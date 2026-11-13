//
// CPU Monte Carlo Search Implementation
// Created for Uni-Dock2 CPU backend
//

#include "mc_cpu.h"
#include "geometry/quaternion.h"
#include <cmath>
#include <vector>
#include <cstring>
#include <chrono>
#include <atomic>
#include <spdlog/spdlog.h>

#if USE_OPENMP
#include <omp.h>
#endif

// Constants from CUDA version
static const Real PHI_CPU = 1.4142136;
static const Real PSI_CPU = 1.5337512;
static const Real PI_CPU = Real(3.1415927);
static const Real EPSILON_MC = std::numeric_limits<Real>::epsilon();

// Skip BFGS for severely clashing conformations.  A cheap energy-only eval
// is done first; if the energy exceeds this ceiling the pose is hopeless and
// full BFGS is skipped.  Only applied when natom >= ENERGY_CEILING_MIN_ATOMS
// to avoid interfering with position-bias / align docking on normal ligands
// where initial randomised energy can legitimately be high.
static const Real ENERGY_CEILING = 1000.0;
static const int  ENERGY_CEILING_MIN_ATOMS = 600;

// Per-pose wall-time budget (seconds).  Propagated as a thread-local deadline
// into line_search_cpu and bfgs_cpu as a safety net.
static const double POSE_TIMEOUT_SEC = 2.0;


void randomize_pose_cpu(FlexPose* out_pose, const FlexPose* pose_old,
                        const FlexTopo& flex_topo,
                        CPURandom& rng, int pose_id, int total_poses) {
    const auto& c = g_cpu_constants;
    
    // Copy coordinates from old pose
    std::memcpy(out_pose->coords, pose_old->coords, flex_topo.natom * 3 * sizeof(Real));
    out_pose->energy = pose_old->energy;
    
    // Copy center and orientation
    out_pose->center[0] = pose_old->center[0];
    out_pose->center[1] = pose_old->center[1];
    out_pose->center[2] = pose_old->center[2];
    out_pose->rot_vec[0] = pose_old->rot_vec[0];
    out_pose->rot_vec[1] = pose_old->rot_vec[1];
    out_pose->rot_vec[2] = pose_old->rot_vec[2];
    
    // Copy dihedrals
    for (int i = 0; i < flex_topo.ntorsion; i++) {
        out_pose->dihedrals[i] = pose_old->dihedrals[i];
    }
    
    // Temporary gradient for applying randomization
    std::vector<Real> dihedrals_g(flex_topo.ntorsion);
    FlexPoseGradient aux_g;
    aux_g.dihedrals_g = dihedrals_g.data();
    
    if (c.FLAG_CONSTRAINT_DOCK) {
        // Center and orientation are fixed
        aux_g.center_g[0] = 0;
        aux_g.center_g[1] = 0;
        aux_g.center_g[2] = 0;
        aux_g.orientation_g[0] = 0;
        aux_g.orientation_g[1] = 0;
        aux_g.orientation_g[2] = 0;
    } else {
        // Random center within box
        Real new_center[3];
        if (c.USE_INNER_BOX) {
            // Inner box: every chain starts with its reference point inside it
            new_center[0] = rng.rand_real_within(c.INNER_X_LO, c.INNER_X_HI);
            new_center[1] = rng.rand_real_within(c.INNER_Y_LO, c.INNER_Y_HI);
            new_center[2] = rng.rand_real_within(c.INNER_Z_LO, c.INNER_Z_HI);
        } else {
            Real a = gyration_radius_cpu(pose_old, &flex_topo);
            new_center[0] = rng.rand_real_within(c.BOX_X_LO + a, c.BOX_X_HI - a);
            new_center[1] = rng.rand_real_within(c.BOX_Y_LO + a, c.BOX_Y_HI - a);
            new_center[2] = rng.rand_real_within(c.BOX_Z_LO + a, c.BOX_Z_HI - a);
        }
        
        aux_g.center_g[0] = new_center[0] - out_pose->center[0];
        aux_g.center_g[1] = new_center[1] - out_pose->center[1];
        aux_g.center_g[2] = new_center[2] - out_pose->center[2];
        
        // Random orientation using Super-Fibonacci spiral
        // Alexa, M. (2022). Super-Fibonacci Spirals
        Real s = pose_id + 0.5;
        Real r = std::sqrt(s / total_poses);
        Real R = std::sqrt(1.0 - s / total_poses);
        Real alpha = 2.0 * PI_CPU * s / PHI_CPU;
        Real beta = 2.0 * PI_CPU * s / PSI_CPU;
        
        Real q[4] = {
            r * std::sin(alpha),
            r * std::cos(alpha),
            R * std::sin(beta),
            R * std::cos(beta)
        };
        quaternion_to_rotvec(aux_g.orientation_g, q);
    }
    
    // Random torsions
    for (int i = 0; i < flex_topo.ntorsion; i++) {
        int i_lo = flex_topo.range_inds[i * 2];
        int nrange = flex_topo.range_inds[i * 2 + 1];
        
        // Pick a random range
        int range_idx = rng.randint(0, nrange - 1);
        int range_offset = i_lo + range_idx * 2;
        
        Real lo = flex_topo.range_list[range_offset];
        Real hi = flex_topo.range_list[range_offset + 1];
        
        Real new_dihe = rng.rand_real_within(lo, hi);
        aux_g.dihedrals_g[i] = new_dihe - out_pose->dihedrals[i];
    }
    
    // Apply the randomization
    apply_grad_update_pose_cpu(out_pose, &aux_g, flex_topo, 1.0);
}


void mutate_pose_cpu(FlexPose* out_pose, const FlexTopo* flex_topo,
                     CPURandom& rng, Real amplitude) {
    const auto& c = g_cpu_constants;
    
    int num_mutable = 2 + flex_topo->ntorsion;  // center, orientation, torsions
    int which;
    
    if (c.FLAG_CONSTRAINT_DOCK) {
        if (num_mutable < 3) {
            return;  // Nothing to mutate
        }
        which = rng.randint(2, num_mutable - 1);  // Only torsions
    } else {
        which = rng.randint(0, num_mutable - 1);
    }
    
    // Generate random values: 3 in unit sphere + 2 uniform in (0,1]
    // Matches CUDA gen_4_rand_in_sphere + curand_uniform
    Real rand_5[5];
    while (true) {
        rand_5[0] = (rng.uniform() - 0.5) * 2.0;  // [-1, 1]
        rand_5[1] = (rng.uniform() - 0.5) * 2.0;
        rand_5[2] = (rng.uniform() - 0.5) * 2.0;
        Real norm_sq = rand_5[0]*rand_5[0] + rand_5[1]*rand_5[1] + rand_5[2]*rand_5[2];
        if (norm_sq < 1.0) {
            rand_5[3] = rng.uniform();  // for range selection
            rand_5[4] = rng.uniform();  // for position within range
            break;
        }
    }
    
    if (which == 0) {
        // Translation: clamp to box boundaries, directly update coords
        // Matches CUDA mutate_pose_warp which==0
        Real dx = clamp_by_range_cpu(amplitude * rand_5[0] + out_pose->center[0],
                                     c.USE_INNER_BOX ? c.INNER_X_HI : c.BOX_X_HI, c.USE_INNER_BOX ? c.INNER_X_LO : c.BOX_X_LO) - out_pose->center[0];
        Real dy = clamp_by_range_cpu(amplitude * rand_5[1] + out_pose->center[1],
                                     c.USE_INNER_BOX ? c.INNER_Y_HI : c.BOX_Y_HI, c.USE_INNER_BOX ? c.INNER_Y_LO : c.BOX_Y_LO) - out_pose->center[1];
        Real dz = clamp_by_range_cpu(amplitude * rand_5[2] + out_pose->center[2],
                                     c.USE_INNER_BOX ? c.INNER_Z_HI : c.BOX_Z_HI, c.USE_INNER_BOX ? c.INNER_Z_LO : c.BOX_Z_LO) - out_pose->center[2];
        
        for (int i = 0; i < flex_topo->natom; i++) {
            out_pose->coords[i * 3]     += dx;
            out_pose->coords[i * 3 + 1] += dy;
            out_pose->coords[i * 3 + 2] += dz;
        }
        out_pose->center[0] += dx;
        out_pose->center[1] += dy;
        out_pose->center[2] += dz;
        
    } else if (which == 1) {
        // Rotation: scale by 1/gyration_radius, directly rotate coords
        // Matches CUDA mutate_pose_warp which==1
        Real a = gyration_radius_cpu(out_pose, flex_topo);
        if (a > EPSILON_MC) {
            Real rot_vec[3] = {
                amplitude / a * rand_5[0],
                amplitude / a * rand_5[1],
                amplitude / a * rand_5[2]
            };
            
            Real q[4];
            rotvec_to_quaternion(q, rot_vec);
            out_pose->rot_vec[0] = rot_vec[0];
            out_pose->rot_vec[1] = rot_vec[1];
            out_pose->rot_vec[2] = rot_vec[2];
            
            // Rotate all atoms around center
            for (int i = 0; i < flex_topo->natom; i++) {
                Real tmp[3] = {
                    out_pose->coords[i * 3]     - out_pose->center[0],
                    out_pose->coords[i * 3 + 1] - out_pose->center[1],
                    out_pose->coords[i * 3 + 2] - out_pose->center[2]
                };
                rotate_vec_by_quaternion(tmp, q);
                out_pose->coords[i * 3]     = tmp[0] + out_pose->center[0];
                out_pose->coords[i * 3 + 1] = tmp[1] + out_pose->center[1];
                out_pose->coords[i * 3 + 2] = tmp[2] + out_pose->center[2];
            }
        }
        
    } else {
        // Torsion: sample a new angle from allowed ranges
        // Matches CUDA mutate_pose_warp which>=2
        int i_tor = which - 2;
        Real a = get_radian_in_ranges_cpu(
            flex_topo->range_list + flex_topo->range_inds[2 * i_tor],
            flex_topo->range_inds[2 * i_tor + 1],
            rand_5 + 3) - out_pose->dihedrals[i_tor];
        apply_grad_update_dihe_cpu(out_pose, flex_topo, i_tor, a);
    }
}


// MCThreadMemory is now defined in mc_cpu.h

void mc_single_pose_cpu(FlexPose* out_pose, const FlexTopo& flex_topo,
                        const FixMol& fix_mol, const FlexParamVina& flex_param,
                        const FixParamVina& fix_param,
                        CPURandom& rng,
                        int mc_steps, int opt_steps, bool randomize,
                        int pose_id, int total_poses,
                        MCThreadMemory& mem) {
    
    int ntorsion = flex_topo.ntorsion;
    int natom = flex_topo.natom;
    int dim_x = dof_x + ntorsion;
    
    // Use pre-allocated memory
    mem.resize(natom, ntorsion);
    
    // pose_candidate: the mutated pose for each MC step
    FlexPose pose_candidate;
    pose_candidate.coords = mem.aux_coords.data();
    pose_candidate.dihedrals = mem.aux_dihedrals.data();
    
    // pose_accepted: the currently accepted pose in the MC chain
    FlexPose pose_accepted;
    pose_accepted.coords = mem.current_coords.data();
    pose_accepted.dihedrals = mem.current_dihedrals.data();
    
    // Randomize if needed
    if (randomize) {
        randomize_pose_cpu(&pose_accepted, out_pose, flex_topo, rng, pose_id, total_poses);
    } else {
        duplicate_pose_cpu(&pose_accepted, out_pose, dim_x, natom);
    }
    
    // Handle mc_steps == 0: just compute energy and return
    // Matches CUDA mc_kernel mc_steps==0 path
    if (mc_steps == 0) {
        pose_accepted.energy = cal_e_f_cpu(&pose_accepted, flex_topo, fix_mol, flex_param, fix_param, mem.forces.data());
        duplicate_pose_cpu(out_pose, &pose_accepted, dim_x, natom);
        return;
    }
    
    // Set thread-local deadline — checked inside line_search_cpu & bfgs_cpu
    set_pose_deadline(POSE_TIMEOUT_SEC);

    // Track best energy seen (matches CUDA best_e tracking)
    Real best_e = Real(1e9);

    // Warm start: the loop mutates before it scores and unconditionally accepts
    // step 0, so the incoming pose is never scored and can never be returned.
    // When the caller supplied a specific pose (randomize == false) that silently
    // discarded it however good it was — handing the engine the crystal pose came
    // back 9 A away with a 5 kcal/mol worse score.  Evaluate (and locally
    // optimize) it up front so it can win.  Skipped when randomizing, where the
    // starting pose is arbitrary and this would only burn an extra BFGS.
    if (!randomize) {
        if (opt_steps > 0) {
            bfgs_with_memory_cpu(&pose_accepted, flex_topo, fix_mol, flex_param,
                                 fix_param, opt_steps, mem.bfgs_mem);
        } else {
            pose_accepted.energy = cal_e_f_cpu(&pose_accepted, flex_topo, fix_mol,
                                               flex_param, fix_param,
                                               mem.forces.data());
        }
        duplicate_pose_cpu(out_pose, &pose_accepted, dim_x, natom);
        best_e = pose_accepted.energy;
    }

    // Monte Carlo loop
    for (int step = 0; step < mc_steps; step++) {
        if (pose_deadline_exceeded()) break;

        // Copy accepted to candidate
        duplicate_pose_cpu(&pose_candidate, &pose_accepted, dim_x, natom);
        
        // Mutate candidate
        mutate_pose_cpu(&pose_candidate, &flex_topo, rng);
        
        // Quick energy check: skip BFGS for severely clashing poses (safety net).
        // Only for large systems (>= ENERGY_CEILING_MIN_ATOMS) where catastrophic
        // clashes cause numerical issues.  Small/medium ligands with position-bias
        // or align can legitimately have high initial energies that BFGS resolves.
        Real quick_e = cal_e_f_cpu(&pose_candidate, flex_topo, fix_mol,
                                    flex_param, fix_param, mem.forces.data());

        if (natom >= ENERGY_CEILING_MIN_ATOMS && quick_e > ENERGY_CEILING) {
            pose_candidate.energy = quick_e;
        } else if (opt_steps == 0) {
            // No optimization, just compute energy and gradient
            std::vector<Real> dihedrals_g_tmp(ntorsion);
            FlexPoseGradient aux_g;
            aux_g.dihedrals_g = dihedrals_g_tmp.data();
            pose_candidate.energy = cal_e_grad_cpu(&pose_candidate, &aux_g, flex_topo, fix_mol, flex_param, fix_param, mem.forces.data());
        } else {
            // BFGS optimization (using pre-allocated memory)
            bfgs_with_memory_cpu(&pose_candidate, flex_topo, fix_mol, flex_param, fix_param,
                                 opt_steps, mem.bfgs_mem);
        }

        // Post-BFGS hbond angle correction (legacy binary filter).
        // When USE_ANGULAR_HBOND is active, the smooth angular weight in cal_e_f_cpu
        // handles angle dependence inline during BFGS — this post-hoc correction
        // is redundant and disabled.
        if (g_cpu_constants.USE_POST_BFGS_HBOND_CORR) {
            pose_candidate.energy += hbond_angle_correction_cpu(
                &pose_candidate, flex_topo, fix_mol);
        }

        // Metropolis criterion with temperature=1.2 (matches CUDA)
        bool accepted = false;
        if (step == 0) {
            // First step is always accepted (matches CUDA: "if (step == 0 || accepted)")
            accepted = true;
        } else {
            Real rand_x = rng.uniform();
            // metropolis_accept from search/mc.h: temperature=1.2
            if (pose_candidate.energy < pose_accepted.energy) {
                accepted = true;
            } else {
                Real accept_p = std::exp((pose_accepted.energy - pose_candidate.energy) / Real(1.2));
                accepted = (rand_x < accept_p);
            }
        }
        
        if (accepted) {
            // Update accepted pose
            duplicate_pose_cpu(&pose_accepted, &pose_candidate, dim_x, natom);
            
            // Track best pose ever seen (matches CUDA best_e logic)
            if (pose_accepted.energy < best_e) {
                duplicate_pose_cpu(out_pose, &pose_accepted, dim_x, natom);
                best_e = pose_accepted.energy;
            }
        }
    }
}


void mc_cpu(FlexPose* out_poses, const FlexTopo* flex_topos,
            const FixMol& fix_mol, const FlexParamVina* flex_params,
            const FixParamVina& fix_param,
            int mc_steps, int opt_steps, int nflex, int exhaustiveness,
            int seed, bool randomize) {

    spdlog::info("CPU Monte Carlo: nflex={}, exhaustiveness={}, mc_steps={}, opt_steps={}",
                 nflex, exhaustiveness, mc_steps, opt_steps);
    
    int total_poses = nflex * exhaustiveness;
    
#if USE_OPENMP
    int num_threads = omp_get_max_threads();
    spdlog::info("Using OpenMP with {} threads", num_threads);
    
    // Pre-allocate thread-local memory to avoid allocations in hot loop
    std::vector<MCThreadMemory> thread_mem(num_threads);
    
    std::atomic<int> poses_done{0};
    auto mc_wall_start = std::chrono::steady_clock::now();

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i_flex = 0; i_flex < nflex; i_flex++) {
        for (int i_pose = 0; i_pose < exhaustiveness; i_pose++) {
            int global_idx = i_flex * exhaustiveness + i_pose;
            int tid = omp_get_thread_num();
            
            // Each pose gets its own random generator with unique seed
            CPURandom rng(seed + global_idx * 1000);
            
            FlexPose* pose = &out_poses[global_idx];
            const FlexTopo& topo = flex_topos[i_flex];
            const FlexParamVina& param = flex_params[i_flex];
            
            auto t0 = std::chrono::steady_clock::now();
            mc_single_pose_cpu(pose, topo, fix_mol, param, fix_param,
                              rng, mc_steps, opt_steps, randomize,
                              global_idx, total_poses, thread_mem[tid]);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            int done = poses_done.fetch_add(1) + 1;

            if (ms > 1000.0) {
                spdlog::warn("Slow pose {}: {:.0f} ms (E={:.1f})", global_idx, ms, pose->energy);
            }
            if (done % 256 == 0 || done == total_poses) {
                double wall = std::chrono::duration<double>(t1 - mc_wall_start).count();
                spdlog::info("MC progress: {}/{} poses done, wall={:.1f}s", done, total_poses, wall);
            }
        }
    }
#else
    // Single-threaded version
    MCThreadMemory mem;
    for (int i_flex = 0; i_flex < nflex; i_flex++) {
        for (int i_pose = 0; i_pose < exhaustiveness; i_pose++) {
            int global_idx = i_flex * exhaustiveness + i_pose;

            CPURandom rng(seed + global_idx * 1000);

            FlexPose* pose = &out_poses[global_idx];
            const FlexTopo& topo = flex_topos[i_flex];
            const FlexParamVina& param = flex_params[i_flex];

            mc_single_pose_cpu(pose, topo, fix_mol, param, fix_param,
                              rng, mc_steps, opt_steps, randomize,
                              global_idx, total_poses, mem);
        }
    }
#endif
    
    spdlog::info("CPU Monte Carlo completed");
}
