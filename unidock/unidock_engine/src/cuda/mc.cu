//
// Created by Congcong Liu on 24-9-25.
//

#include <cooperative_groups.h>

#include "myutils/myio.h"
#include "model/model.h"
#include "search/mc.h"

#include "common.cuh"
#include <chrono>
#include <cstdlib>
#include "myutils/errors.h"
#include "myutils/mymath.h"
#include "geometry/quaternion.h"
#include "bfgs.cuh"
#include <type_traits>



template<bool TORLIB, class Tile>
__device__ __forceinline__ void randomize_pose_warp(const Tile& tile,
                                                    FlexPose* out_pose_new, Real* aux_g, Real* aux_scratch,
                                                    const FlexPose* pose_old, const FlexTopo& flex_topo,
                                                    int n,
                                                    curandStatePhilox4_32_10_t* state){
    float4 rf4 = curand_uniform4(state);
    Real tmp4[4] = {0};
    Real rotvec[3] = {map_01_to_dot5(rf4.x), map_01_to_dot5(rf4.y), map_01_to_dot5(rf4.z)};
    uint4 ri4 = curand4(state);

    // copy cartesian coordinates
    for (int i = tile.thread_rank(); i < flex_topo.natom * 3; i += tile.num_threads()){
        out_pose_new->coords[i] = pose_old->coords[i];
    }
    tile.sync();

    // generate a random pose inside the box
    if (tile.thread_rank() == 0){
        // set energy
        out_pose_new->energy = pose_old->energy;

        // copy center & orientation
        out_pose_new->center[0] = pose_old->center[0];
        out_pose_new->center[1] = pose_old->center[1];
        out_pose_new->center[2] = pose_old->center[2];
        out_pose_new->rot_vec[0] = pose_old->rot_vec[0];
        out_pose_new->rot_vec[1] = pose_old->rot_vec[1];
        out_pose_new->rot_vec[2] = pose_old->rot_vec[2];

        if (FLAG_CONSTRAINT_DOCK){
            // center and orientation are fixed
            aux_g[0] = 0;
            aux_g[STRIDE_G] = 0;
            aux_g[2 * STRIDE_G] = 0;
            aux_g[3 * STRIDE_G] = 0;
            aux_g[4 * STRIDE_G] = 0;
            aux_g[5 * STRIDE_G] = 0;
        }
        else{
            // random center, set gradient
            if (USE_INNER_BOX){
                // Inner box: every chain starts with its reference point inside it
                tmp4[0] = get_real_within_by_int(ri4.x, INNER_X_LO, INNER_X_HI, ceil((INNER_X_HI - INNER_X_LO) / BOX_PREC) + 1);
                tmp4[1] = get_real_within_by_int(ri4.y, INNER_Y_LO, INNER_Y_HI, ceil((INNER_Y_HI - INNER_Y_LO) / BOX_PREC) + 1);
                tmp4[2] = get_real_within_by_int(ri4.z, INNER_Z_LO, INNER_Z_HI, ceil((INNER_Z_HI - INNER_Z_LO) / BOX_PREC) + 1);
            }
            else{
                Real a = gyration_radius(pose_old, &flex_topo);
                tmp4[0] = get_real_within_by_int(ri4.x, BOX_X_LO + a, BOX_X_HI - a, ceil((BOX_X_HI - BOX_X_LO - 2 * a) / BOX_PREC) + 1);
                tmp4[1] = get_real_within_by_int(ri4.y, BOX_Y_LO + a, BOX_Y_HI - a, ceil((BOX_Y_HI - BOX_Y_LO - 2 * a) / BOX_PREC) + 1);
                tmp4[2] = get_real_within_by_int(ri4.z, BOX_Z_LO + a, BOX_Z_HI - a, ceil((BOX_Z_HI - BOX_Z_LO - 2 * a) / BOX_PREC) + 1);
            }

            aux_g[0] = tmp4[0] - out_pose_new->center[0];
            aux_g[STRIDE_G] = tmp4[1] - out_pose_new->center[1];
            aux_g[2 * STRIDE_G] = tmp4[2] - out_pose_new->center[2];

            // random orientation, set gradient.
            // Alexa, M. (2022). Super-Fibonacci Spirals: Fast, Low-Discrepancy Sampling of SO(3). Proceedings of the
            // IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2022-June(3), 8281–8290.
            // https://doi.org/10.1109/CVPR52688.2022.00811

            int id_pose = ((blockIdx.x * blockDim.x + threadIdx.x) % (n * TILE_SIZE)) / TILE_SIZE; // global idx of the thread
            DPrint1("id_pose: %d\n", id_pose);

            Real s = id_pose + 0.5;
            Real r = sqrt( s / n);
            Real R = sqrt(1.0 - s / n);
            Real alpha = 2.0 * PI * s / PHI;
            Real beta = 2.0 * PI * s / PSI;
            tmp4[0] = r * sin(alpha);
            tmp4[1] = r * cos(alpha);
            tmp4[2] = R * sin(beta);
            tmp4[3] = R * cos(beta);
            Real rv[3];
            quaternion_to_rotvec(rv, tmp4);
            aux_g[3 * STRIDE_G] = rv[0];
            aux_g[4 * STRIDE_G] = rv[1];
            aux_g[5 * STRIDE_G] = rv[2];
            // printf("[RAND] %f, %f, %f\n", rv[0], rv[1], rv[2]);
        }

        // generate random uints for all torsions // todo: change sampling of torsions
        for (int i = 0; i < flex_topo.ntorsion; i ++){
            // copy dihedrals
            tmp4[3] = pose_old->dihedrals[i];

            out_pose_new->dihedrals[i] = tmp4[3];
            // set gradient
            ri4.x = curand(state);
            ri4.w = ri4.x % flex_topo.range_inds[i * 2 + 1]; // save index of range_list
            ri4.z = flex_topo.range_inds[i * 2] + ri4.w * 2; // save a tmp index
            ri4.x = curand(state);

            tmp4[1] = flex_topo.range_list[ri4.z + 1] - flex_topo.range_list[ri4.z];
            ri4.y = ceil(tmp4[1] / TOR_PREC) + 1; // 10 degree as precision

            tmp4[0] = get_real_within_by_int(ri4.x, flex_topo.range_list[ri4.z], flex_topo.range_list[ri4.z + 1], ri4.y);
            aux_g[(dof_g + i) * STRIDE_G] = tmp4[0] - tmp4[3];
        }
    }
    tile.sync();

    apply_grad_update_pose_warp<TORLIB>(tile, out_pose_new, aux_g, flex_topo, 1., aux_scratch);

}




/**
 * @brief Mutate one pose and update coords.
 * 
 * @param tile Cooperative group
 * @param out_pose Pointer to the pose to be mutated
 * @param flex_topo Topology of the flex
 * @param state cuRand state
 * @param amplitude Amplitude of the mutation
 */
template<bool TORLIB, class Tile>
__forceinline__ __device__ void mutate_pose_warp(const Tile& tile, FlexPose* out_pose,
                                                 const FlexTopo* flex_topo,
                                                 curandStatePhilox4_32_10_t* state, Real amplitude = 1){ //amplitude:2.0
    // DOF, which as an index of DOF
    Real rand_5[5] = {0};
    Real q[4] = {0}; //todo: use rand_5 instead of q to save registers
    Real tmp1[3] = {0};
    Real a = 0;
    int which = -1;

    if (tile.thread_rank() == 0){
        int num_mutable = 2 + flex_topo->ntorsion; //center, orientation, torsions
        if (FLAG_CONSTRAINT_DOCK){
            if (num_mutable < 3){
                which = 3; // no mutation
            } else{
                which = gen_rand_int_within(state, 2, num_mutable - 1);
            }
        }
        else{
            which = gen_rand_int_within(state, 0, num_mutable - 1);
        }
        // DPrint1("which is %d\n", which);
        // prepare random values for choosing DOF to mutate
        gen_4_rand_in_sphere(rand_5, state);
        rand_5[4] = curand_uniform(state);
    }
    tile.sync();

    which = tile.shfl(which, 0);
    rand_5[0] = tile.shfl(rand_5[0], 0);
    rand_5[1] = tile.shfl(rand_5[1], 0);
    rand_5[2] = tile.shfl(rand_5[2], 0);
    rand_5[3] = tile.shfl(rand_5[3], 0);
    rand_5[4] = tile.shfl(rand_5[4], 0);

    // 0 for translation
    if (which == 0){
        // compute a translation under box constraint
        tmp1[0] = clamp_by_range(amplitude * rand_5[0] + out_pose->center[0], USE_INNER_BOX ? INNER_X_HI : BOX_X_HI, USE_INNER_BOX ? INNER_X_LO : BOX_X_LO) - out_pose->center[0]; //amplitude * rand_5[0];
        tmp1[1] = clamp_by_range(amplitude * rand_5[1] + out_pose->center[1], USE_INNER_BOX ? INNER_Y_HI : BOX_Y_HI, USE_INNER_BOX ? INNER_Y_LO : BOX_Y_LO) - out_pose->center[1]; //amplitude * rand_5[1];
        tmp1[2] = clamp_by_range(amplitude * rand_5[2] + out_pose->center[2], USE_INNER_BOX ? INNER_Z_HI : BOX_Z_HI, USE_INNER_BOX ? INNER_Z_LO : BOX_Z_LO) - out_pose->center[2]; //amplitude * rand_5[2];

        for (int i_at = tile.thread_rank(); i_at < flex_topo->natom; i_at += tile.num_threads()){
            // move to the new center
            out_pose->coords[i_at * 3] = out_pose->coords[i_at * 3] + tmp1[0];
            out_pose->coords[i_at * 3 + 1] = out_pose->coords[i_at * 3 + 1] + tmp1[1];
            out_pose->coords[i_at * 3 + 2] = out_pose->coords[i_at * 3 + 2] + tmp1[2];
        }
        tile.sync();
        if (tile.thread_rank() == 0){
            out_pose->center[0] = out_pose->center[0] + tmp1[0];
            out_pose->center[1] = out_pose->center[1] + tmp1[1];
            out_pose->center[2] = out_pose->center[2] + tmp1[2];
        }
        tile.sync();
    }
    else if (which == 1){
        // 1 for rotation
        if (tile.thread_rank() == 0){
            //1 for rotation of the whole molecule
            a = gyration_radius(out_pose, flex_topo); // an indicator of the size
            if (a > EPSILON){
                // add a random rotation to temporary quaternion
                // the movement step of an atom is roughly amplitude Angstrom
                tmp1[0] = amplitude / a * rand_5[0];
                tmp1[1] = amplitude / a * rand_5[1];
                tmp1[2] = amplitude / a * rand_5[2];

                rotvec_to_quaternion(q, tmp1);
                out_pose->rot_vec[0] = tmp1[0];
                out_pose->rot_vec[1] = tmp1[1];
                out_pose->rot_vec[2] = tmp1[2];
            }
        }
        tile.sync();
        a = tile.shfl(a, 0);

        if (a > EPSILON){
            q[0] = tile.shfl(q[0], 0);
            q[1] = tile.shfl(q[1], 0);
            q[2] = tile.shfl(q[2], 0);
            q[3] = tile.shfl(q[3], 0);

            // rotate all atoms fixme: the rotation has low precision and leads to error over 0.001
            for (int i_at = tile.thread_rank(); i_at < flex_topo->natom; i_at += tile.num_threads()){
                tmp1[0] = out_pose->coords[i_at * 3] - out_pose->center[0];
                tmp1[1] = out_pose->coords[i_at * 3 + 1] - out_pose->center[1];
                tmp1[2] = out_pose->coords[i_at * 3 + 2] - out_pose->center[2];
                rotate_vec_by_quaternion(tmp1, q);
                out_pose->coords[i_at * 3] = tmp1[0] + out_pose->center[0];
                out_pose->coords[i_at * 3 + 1] = tmp1[1] + out_pose->center[1];
                out_pose->coords[i_at * 3 + 2] = tmp1[2] + out_pose->center[2];
            }
            tile.sync();
        }
    }
    else if (which - 2 < flex_topo->ntorsion){
        // rotate one dihedral
        which -= 2;
        // change lig dihedral
        if (tile.thread_rank() == 0){
            a = get_radian_in_ranges(flex_topo->range_list + flex_topo->range_inds[2 * which],
                                     flex_topo->range_inds[2 * which + 1], rand_5 + 3) - out_pose->dihedrals[which];
            // printf("which is %d, a is %f\n", which,  a);
        }
        a = tile.shfl(a, 0); // increment of dihedral value
        apply_grad_update_dihe_warp<TORLIB>(tile, out_pose, flex_topo, which, a);
    }
    else{
        // no mutation
        // assert(which - 2 < flex_topo->ntorsion);
    }
}


/**
 * @brief Monte Carlo Kernel. Each warp tackles one pose.
 *
 * @param out_poses Prepared poses that have been initialized, size: nflex * num_pose_per_flex
 * @param flex_topos Topology of flex, size: nflex
 * @param aux_poses Auxiliary poses, size: STRIDE_POSE * nflex * num_pose_per_flex
 * @param aux_hessians Per-pose BFGS state blocks (Hessian + DOF vectors, see bfgs_state_ptrs), size: nflex * num_pose_per_flex
 * @param aux_forces Auxiliary forces, size: nflex * num_pose_per_flex
 * @param states Random states, each pose owns one, size: nflex * num_pose_per_flex
 * @param seed Random seed
 * @param mc_steps Number of MC steps
 * @param opt_steps Number of optimization steps after a pose is accepted
 * @param num_pose_per_flex Number of poses per flex
 * @param max_thread Maximum number of threads used
 * @param rng_slots Stream slot of each flex, size: nflex; pose i of flex f draws curand subsequence
 *                  rng_slots[f] * num_pose_per_flex + i. nullptr: subsequence = id_pose
 */
// Occupancy target, in warps resident per multiprocessor. Asking for 20 caps the compiler near
// 102 registers per thread, which on Ada (sm_89) forces 152 bytes of local-memory spill and costs
// 4-19 % on this kernel; 12 fits in 140 registers with no spill. The kernel is latency-bound and
// far from occupancy-limited anyway: 512 single-warp blocks leave most of the device idle, and
// running four times as many poses costs no extra time.
#ifndef UD2_MC_MIN_BLOCKS
#define UD2_MC_MIN_BLOCKS 12
#endif
// Launch-shape dispatch: the receptor-map kernel also compiles at 16 blocks per SM (127 registers, no
// spill on sm_89), which is 1.12x in the screening regime (a full-device wave, thousands of blocks) and
// 1.00x at 512 poses. mc_cu takes it for GRID launches of at least UD2_MC_LARGE_LAUNCH blocks (default
// 16 blocks x 128 SMs x 4 waves; env UD2_MC_LARGE_LAUNCH overrides, 0 = never) and UD2_MC_MIN_BLOCKS
// otherwise. The analytic kernel at 16 spills 72 B and is never instantiated.
#define UD2_MC_LARGE_MIN_BLOCKS 16
#ifndef UD2_MC_LARGE_LAUNCH
#define UD2_MC_LARGE_LAUNCH 8192
#endif
// GRID selects the energy model of every evaluation in this kernel (bfgs.cuh cal_e_f_warp<GRID>):
// true = receptor maps + exact hydrogen-bond pass (g_maps / g_pgrid built), false = analytic pairs.
// MINB is the __launch_bounds__ minimum blocks per SM. The body depends on it only through PREFETCH
// (cal_e_f_warp's one-step-ahead loads, bit-identical results): on in the large-launch receptor-map
// instantiation (screening batches, about 3 % faster), off in the others, whose source is then the code
// without it. Enabled everywhere it measured about 6 % faster on single 512-pose runs but about 1.5 % slower
// on udbench re-docking under 8-client MPS (PoseX under MPS +4 %), so the small-launch kernels keep the plain code.
// TORLIB: torsion-library variant (bfgs.cuh), taken by launches with a limited torsion (mc_cu torlib).
template<bool GRID, int MINB, bool TORLIB>
__global__ void __launch_bounds__(TILE_SIZE, MINB) mc_kernel(FlexPose* out_poses, const FlexTopo* flex_topos, const FixMol& fix_mol,
                          const FlexParamVina* flex_params, const FixParamVina& fix_param,
                          FlexPose* aux_poses, FlexPoseHessian* aux_hessians,
                          FlexForce* aux_forces,
                          curandStatePhilox4_32_10_t* states, int seed, bool randomize,
                          int mc_steps, int opt_steps, int num_pose_per_flex, int max_thread, int state_flags,
                          const int* rng_slots){
    constexpr bool PREFETCH = GRID && MINB >= UD2_MC_LARGE_MIN_BLOCKS;
    // Just for ONE best pose

    int id_thread = blockIdx.x * blockDim.x + threadIdx.x; // global idx of the thread
    if (id_thread < max_thread){
        int id_pose = id_thread / TILE_SIZE; // 一个pose由1个tile处理，也就是包含多个threads
        int id_flex = id_pose / num_pose_per_flex;

        // Use alias
        FlexPose& out_pose = out_poses[id_pose]; // pointer to global data
        const FlexTopo& flex_topo = flex_topos[id_flex];
        const FlexParamVina& flex_param = flex_params[id_flex];

        FlexPose& pose_candidate = aux_poses[id_pose * STRIDE_POSE];
        FlexPose& pose_accepted = aux_poses[id_pose * STRIDE_POSE + 1];
        FlexPose& aux_pose_ori = aux_poses[id_pose * STRIDE_POSE + 2];
        FlexPose& aux_pose_new = aux_poses[id_pose * STRIDE_POSE + 3];

        FlexPoseHessian& aux_h = aux_hessians[id_pose];

        FlexForce& aux_f = aux_forces[id_pose]; //todo: use struct or just Real* ?

        curandStatePhilox4_32_10_t& state = states[id_pose];


        // Cooperative group. The screening instantiation (PREFETCH) synchronises with the block barrier (AlignedTile).
        auto cta = cg::this_thread_block();
        using KernelTile = std::conditional_t<PREFETCH, AlignedTile, cg::thread_block_tile<TILE_SIZE>>;
        const KernelTile tile(cg::tiled_partition<TILE_SIZE>(cta));
        // Init curand states
        if (tile.thread_rank() == 0){
            int id_rng = rng_slots ? rng_slots[id_flex] * num_pose_per_flex + (id_pose - id_flex * num_pose_per_flex) : id_pose;
            curand_init(seed, id_rng, 0, &state);
        }
        tile.sync();

        // flat BFGS state of this pose; slot BV_G (the start of V) is also randomize_pose_warp's displacement
        Real* bfgs_h = nullptr;
        Real* bfgs_v = nullptr;
        bfgs_state_ptrs(tile, aux_h.matrix, flex_topo.natom, flex_topo.ntorsion, state_flags, &bfgs_h, &bfgs_v);
        Real* aux_g = bfgs_v + BV_G;

        int dim = 3 + 4 + flex_topo.ntorsion;
        Real best_e = 1e9; // large value for finding minimum energy

        PROF_DECL;
        if (randomize){
            // prepare the initial pose: each pose is a random pose!
            randomize_pose_warp<TORLIB>(tile, &pose_accepted, aux_g, bfgs_v + BV_Y, &out_pose, flex_topo, num_pose_per_flex, &state);
        }else{
            duplicate_pose_warp(tile, &pose_accepted, &out_pose, dim, flex_topo.natom);
        }
        PROF_MARK(PROF_RANDOMIZE);

        if (mc_steps == 0){
            Real energy = cal_e_f_warp<GRID, PREFETCH>(tile, &pose_accepted, flex_topo, fix_mol, flex_param, fix_param, aux_f.f);

            if (tile.thread_rank() == 0){
                pose_accepted.energy = energy;
            }
            duplicate_pose_warp(tile, &out_pose, &pose_accepted, dim, flex_topo.natom);
        }
        else{
            for (int step = 0; step < mc_steps; step++){
                // 1. mutate conf, PRODUCE a random conf
                DPrint1("========= MC step %d \n", step);

                PROF_RESET(); PROF_COUNT(PROF_N_MC_STEP);
                duplicate_pose_warp(tile, &pose_candidate, &pose_accepted, dim, flex_topo.natom);

                mutate_pose_warp<TORLIB>(tile, &pose_candidate, &flex_topo, &state);
                PROF_MARK(PROF_MUTATE);

                // todo: add clash-detection for efficiency

                // Record initial energy and gradient. E_ori is energy, aux_g is set as current gradient
                // If max_steps == 0, this func only records the energy of original structure.
                if (opt_steps == 0){
                    Real energy = cal_e_grad_warp<GRID, PREFETCH>(tile, &pose_candidate, aux_g, flex_topo, fix_mol,
                                                                  flex_param, fix_param, aux_f.f);

                    if (tile.thread_rank() == 0){
                        pose_candidate.energy = energy;
                    }
                    tile.sync();
                }

                // Local minimisation of the candidate at the hunt cap. Vina's second pass (re-minimising a new
                // best pose at authentic_v inside the chain) is not run: every search output is rescored and
                // refined analytically at CURL_V_AUTHENTIC_CU after the search (rescore_kernel, opt_kernel),
                // and the in-chain pass measured 1.3-1.4x of engine time for no top-1 change on 1,454
                // systems x 2 seeds. The CPU backend never had it.
                if (opt_steps > 0){
                    // essential optimization. coords are updated inside bfgs
                    bfgs_warp<GRID, PREFETCH, TORLIB>(tile,
                                                      &pose_candidate, flex_topo, fix_mol,
                                                      flex_param, fix_param,
                                                      &aux_pose_new, &aux_pose_ori,
                                                      bfgs_v, bfgs_h, &aux_f, opt_steps, CURL_V_CU);
                }

                // 2. Metropolis
                PROF_RESET();
                bool accepted = false;
                if (tile.thread_rank() == 0){
                    Real rand_x = curand_uniform(&states[id_pose]);
                    accepted = metropolis_accept(pose_accepted.energy, pose_candidate.energy, 1.2, rand_x);
                }
                accepted = tile.shfl(accepted, 0);
                if (step == 0 || accepted){
                    // set accepted pose as this lately accepted candidate
                    duplicate_pose_warp(tile, &pose_accepted, &pose_candidate, dim, flex_topo.natom);
                    if (pose_accepted.energy < best_e){
                        duplicate_pose_warp(tile, &out_pose, &pose_accepted, dim, flex_topo.natom);
                        best_e = pose_accepted.energy;
                    }
                }
                PROF_MARK(PROF_MC_BOOKKEEP);
            }
        }
    }
}


// The launch of mc_cu for one TORLIB: instantiation by launch shape, its BFGS state placement and carveout, the kernel.
template<bool TORLIB>
static void launch_mc_kernel(bool use_grid, bool large, int npose, int block_size,
                             FlexPose* out_poses, const FlexTopo* topos,
                             const FixMol& fix_mol, const FlexParamVina* flex_param, const FixParamVina& fix_param,
                             FlexPose* aux_poses, FlexPoseHessian* aux_hessians, FlexForce* aux_forces,
                             curandStatePhilox4_32_10_t* states, int seed, bool randomize,
                             int mc_steps, int opt_steps, int exhuastiveness, const int* rng_slots){
    // unified L1/shared split for this launch (UD2_CARVE overrides the occ32 rule), set on the
    // instantiation that is launched: the receptor-map kernel (use_grid, g_maps built) or the analytic one
    const void* kernel = !use_grid ? (const void*)mc_kernel<false, UD2_MC_MIN_BLOCKS, TORLIB>
                       : large     ? (const void*)mc_kernel<true, UD2_MC_LARGE_MIN_BLOCKS, TORLIB>
                                   : (const void*)mc_kernel<true, UD2_MC_MIN_BLOCKS, TORLIB>;
    // BFGS state placement (shared memory where it keeps this launch's carveout and residency), then the carveout
    const int state_flags = bfgs_state_flags(kernel, "mc_kernel", npose, block_size);
    const size_t dyn = kernel_state_dyn_shared_bytes(state_flags);
    set_carveout(kernel, "mc_kernel", "UD2_CARVE", "occ32", npose, block_size, dyn);
    if (use_grid && large){
        mc_kernel<true, UD2_MC_LARGE_MIN_BLOCKS, TORLIB><<<npose, block_size, dyn>>>(out_poses, topos, fix_mol,
                                     flex_param, fix_param,
                                     aux_poses, aux_hessians, aux_forces,
                                     states, seed, randomize,
                                     mc_steps, opt_steps, exhuastiveness, npose * block_size, state_flags, rng_slots);
    } else if (use_grid){
        mc_kernel<true, UD2_MC_MIN_BLOCKS, TORLIB><<<npose, block_size, dyn>>>(out_poses, topos, fix_mol,
                                     flex_param, fix_param,
                                     aux_poses, aux_hessians, aux_forces,
                                     states, seed, randomize,
                                     mc_steps, opt_steps, exhuastiveness, npose * block_size, state_flags, rng_slots);
    } else {
        mc_kernel<false, UD2_MC_MIN_BLOCKS, TORLIB><<<npose, block_size, dyn>>>(out_poses, topos, fix_mol,
                                     flex_param, fix_param,
                                     aux_poses, aux_hessians, aux_forces,
                                     states, seed, randomize,
                                     mc_steps, opt_steps, exhuastiveness, npose * block_size, state_flags, rng_slots);
    }
}


void mc_cu(FlexPose* out_poses, const FlexTopo* topos,
           const FixMol& fix_mol, const FlexParamVina* flex_param, const FixParamVina& fix_param,
           FlexPose* aux_poses, FlexPoseHessian* aux_hessians, FlexForce* aux_forces,
           int mc_steps, int opt_steps, int nflex, int exhuastiveness, int seed, bool randomize, bool use_grid,
           const int* rng_slots, bool torlib){
    //------- perform MC on GPU -------//

    const int block_size = TILE_SIZE; // One block for one tile (for 32, namely one warp per block)
    int npose = nflex * exhuastiveness;

    // initilize curand states
    curandStatePhilox4_32_10_t* states;
    checkCUDA(cudaMalloc(&states, sizeof(curandStatePhilox4_32_10_t) * npose));

    // run the kernel
#ifdef UD2_PROFILE
    prof_begin(npose);
    auto prof_t0 = std::chrono::steady_clock::now();
#endif
    // opt_steps < 0 used to run a zero-iteration bfgs_warp, which yields the same candidate energy as the
    // opt_steps == 0 branch and no re-minimisation; clamp here so the kernel's hot code stays unchanged
    if (opt_steps < 0) opt_steps = 0;

    // launch shape: GRID launches of at least large_launch blocks (env UD2_MC_LARGE_LAUNCH, 0 = never)
    // take the 16-blocks-per-SM instantiation, everything else UD2_MC_MIN_BLOCKS
    const char* large_env = std::getenv("UD2_MC_LARGE_LAUNCH");
    const int large_launch = large_env ? std::atoi(large_env) : UD2_MC_LARGE_LAUNCH;
    const bool large = use_grid && large_launch > 0 && npose >= large_launch;
    const int min_blocks = large ? UD2_MC_LARGE_MIN_BLOCKS : UD2_MC_MIN_BLOCKS;
    spdlog::info("mc_kernel launch: {} blocks, {} kernel, {} min blocks/SM (large-launch threshold {})",
                 npose, use_grid ? "grid" : "analytic", min_blocks, large_launch);
    spdlog::debug("mc_kernel torsion-library variant: {}", torlib);
    if (torlib){
        launch_mc_kernel<true>(use_grid, large, npose, block_size, out_poses, topos, fix_mol, flex_param, fix_param,
                               aux_poses, aux_hessians, aux_forces, states, seed, randomize,
                               mc_steps, opt_steps, exhuastiveness, rng_slots);
    } else {
        launch_mc_kernel<false>(use_grid, large, npose, block_size, out_poses, topos, fix_mol, flex_param, fix_param,
                                aux_poses, aux_hessians, aux_forces, states, seed, randomize,
                                mc_steps, opt_steps, exhuastiveness, rng_slots);
    }
    checkCUDA(cudaDeviceSynchronize());
#ifdef UD2_PROFILE
    prof_report("mc", npose, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prof_t0).count());
#endif

    // free mem
    checkCUDA(cudaFree(states));
}
