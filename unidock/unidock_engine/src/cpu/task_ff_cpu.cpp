//
// Host-side force-field (GAFF2) minimization stage of DockTask.
//
// This stage runs on the host in both backends: it operates on the filtered
// poses already copied back to host memory (flex_pose_list_res). The CPU
// backend hands it its host-resident topologies directly; the CUDA backend
// rebuilds host copies because its *_cu members point at device memory.
//

#include "common_cpu.h"
#include "score_ff_cpu.h"
#include "bfgs_cpu.h"
#include "bfgs_ff_cpu.h"
#include "host_topo.h"
#include "task/DockTask.h"
#include <spdlog/spdlog.h>
#include <vector>
#include <cstring>
#include <cmath>


// ==================== FF Minimization ====================

/// Line search for FF scoring (mirrors line_search_cpu but uses cal_e_grad_ff_cpu)
static void line_search_ff_cpu(const FixMol& fix_mol, const FFMinData& ff,
                               const FlexPose* x, const FlexPoseGradient* g,
                               const FlexTopo& flex_topo, const FlexPoseGradient* p,
                               Real* aux_f,
                               FlexPose* out_x_new, FlexPoseGradient* out_g_new,
                               Real e0, int dim_g, int dim_x, Real* out_e, Real* out_alpha) {
    Real alpha = 1.0;
    Real e_new = 0;
    const Real pg = g_dot_product_cpu(p, g, dim_g);

    for (int trial = 0; trial < LINE_SEARCH_STEPS_CPU; trial++) {
        duplicate_pose_cpu(out_x_new, x, dim_x, flex_topo.natom);
        // FF minimization: skip torsion library range clamping (uses its own torsion potential)
        apply_grad_update_pose_cpu(out_x_new, p, flex_topo, alpha, /*skip_range_clamp=*/true);
        e_new = cal_e_grad_ff_cpu(out_x_new, out_g_new, flex_topo, fix_mol, ff, aux_f);

        if (e_new - e0 < LINE_SEARCH_C0_CPU * alpha * pg) {
            break;
        }
        alpha *= LINE_SEARCH_MULTIPLIER_CPU;
    }
    *out_e = e_new;
    *out_alpha = alpha;
}


/// Copy ligand cartesian coords + energy only (Cartesian BFGS path).
static inline void duplicate_pose_cartesian_cpu(FlexPose* dst, const FlexPose* src, int natom) {
    std::memcpy(dst->coords, src->coords, natom * 3 * sizeof(Real));
    dst->energy = src->energy;
}


/// Update pose coordinates by a Cartesian step: coords += alpha * p.
static inline void apply_grad_update_pose_cartesian_cpu(FlexPose* pose, const Real* p,
                                                       Real alpha, int natom) {
    int n = natom * 3;
    for (int k = 0; k < n; k++) {
        pose->coords[k] += alpha * p[k];
    }
}


/// Dot product over a flat real vector of length dim.
static inline Real dot_flat_cpu(const Real* a, const Real* b, int dim) {
    Real s = 0;
    for (int i = 0; i < dim; i++) s += a[i] * b[i];
    return s;
}


/// Matrix-vector product: out_p = -H * g, where H is upper-triangular packed.
static void minus_mat_vec_product_flat_cpu(Real* out_p, const Real* h, const Real* g, int dim) {
    for (int i = 0; i < dim; i++) {
        Real sum = 0;
        for (int j = 0; j < dim; j++) {
            sum += h[tri_mat_index(i, j)] * g[j];
        }
        out_p[i] = -sum;
    }
}


/// BFGS Hessian update over a flat-vector dim, mirroring bfgs_update_hessian_cpu.
static void bfgs_update_hessian_flat_cpu(Real* out_h, int dim, const Real* p, const Real* y,
                                         Real* aux_minus_hy, Real alpha) {
    Real yp = dot_flat_cpu(y, p, dim);
    if (alpha * yp < std::numeric_limits<Real>::epsilon()) return;

    minus_mat_vec_product_flat_cpu(aux_minus_hy, out_h, y, dim);
    Real yhy = -dot_flat_cpu(y, aux_minus_hy, dim);
    Real r = 1.0f / (alpha * yp);

    for (int i = 0; i < dim; i++) {
        for (int j = i; j < dim; j++) {
            Real tmp = alpha * r * (aux_minus_hy[i] * p[j] + aux_minus_hy[j] * p[i])
                     + alpha * alpha * (r * r * yhy + r) * p[i] * p[j];
            out_h[i + j * (j + 1) / 2] += tmp;
        }
    }
}


static inline void apply_atom_mask_to_grad(Real* grad,
                                           const std::vector<bool>* mask,
                                           int natom) {
    if (!mask || mask->empty()) return;
    for (int a = 0; a < natom; a++) {
        if ((*mask)[a]) {
            grad[a*3+0] = 0;
            grad[a*3+1] = 0;
            grad[a*3+2] = 0;
        }
    }
}


/// Line search for Cartesian-DOF FF BFGS.
/// First-step cap: clamp initial alpha so max_k |alpha * p[k]| <= 0.1 A.
static void line_search_ff_cartesian_cpu(const FixMol& fix_mol, const FFMinData& ff,
                                         const FlexPose* x, const Real* g,
                                         const FlexTopo& flex_topo, const Real* p,
                                         Real* aux_f, Real* grad_new,
                                         FlexPose* out_x_new,
                                         Real e0, int dim, Real* out_e, Real* out_alpha,
                                         const std::vector<bool>* mask) {
    Real p_inf = 0;
    for (int k = 0; k < dim; k++) {
        Real ap = std::fabs(p[k]);
        if (ap > p_inf) p_inf = ap;
    }
    Real alpha = (p_inf > 1e-6f) ? std::min<Real>(1.0f, 0.1f / p_inf) : 1.0f;

    Real e_new = 0;
    Real pg = dot_flat_cpu(p, g, dim);
    int natom = flex_topo.natom;

    for (int trial = 0; trial < LINE_SEARCH_STEPS_CPU; trial++) {
        if (pose_deadline_exceeded()) break;
        duplicate_pose_cartesian_cpu(out_x_new, x, natom);
        apply_grad_update_pose_cartesian_cpu(out_x_new, p, alpha, natom);
        e_new = cal_e_grad_cartesian_cpu(out_x_new, flex_topo, fix_mol, ff, grad_new, aux_f);
        apply_atom_mask_to_grad(grad_new, mask, natom);

        if (e_new - e0 < LINE_SEARCH_C0_CPU * alpha * pg) break;
        alpha *= LINE_SEARCH_MULTIPLIER_CPU;
    }
    *out_e = e_new;
    *out_alpha = alpha;
}


/// Cartesian-DOF BFGS optimizer for FF scoring (3N DOF, full ligand Cartesian).
/// Pose center/rot_vec/dihedrals are left stale; only pose->coords is the optimization variable.
void bfgs_ff_cartesian_cpu(FlexPose* out_x, const FlexTopo& flex_topo,
                                  const FixMol& fix_mol, const FFMinData& ff,
                                  int max_steps, BFGSThreadMemory& mem,
                                  const std::vector<bool>* constrained_atom_mask) {
    int natom = flex_topo.natom;
    int dim = 3 * natom;
    mem.resize_cartesian(natom);

    FlexPose aux_x_new, aux_x_ori;
    aux_x_new.coords = mem.coords_new.data();
    aux_x_ori.coords = mem.coords_ori.data();

    Real* grad      = mem.grad.data();
    Real* grad_new  = mem.grad_new.data();
    Real* p         = mem.p.data();
    Real* y         = mem.y.data();
    Real* minus_hy  = mem.minus_hy.data();
    Real* h         = mem.h_matrix.data();
    Real* aux_f     = mem.forces.data();

    Real E_ori = cal_e_grad_cartesian_cpu(out_x, flex_topo, fix_mol, ff, grad, aux_f);
    apply_atom_mask_to_grad(grad, constrained_atom_mask, natom);
    Real E = E_ori;
    Real E1 = 0, alpha = 0;

    // Initial Hessian: scaled identity. Target per-atom RMS displacement = 0.05 A.
    Real g_sumsq = 0;
    for (int k = 0; k < dim; k++) g_sumsq += grad[k] * grad[k];
    Real g_rms_per_atom = std::sqrt(g_sumsq / std::max(1, natom)) / std::sqrt(static_cast<Real>(3));
    Real h_scale = 0.05f / std::max(g_rms_per_atom, static_cast<Real>(1e-6));

    int h_size = dim * (dim + 1) / 2;
    std::memset(h, 0, h_size * sizeof(Real));
    for (int i = 0; i < dim; i++) {
        h[tri_mat_index(i, i)] = h_scale;
    }

    // Save originals for rollback.
    duplicate_pose_cartesian_cpu(&aux_x_ori, out_x, natom);
    duplicate_pose_cartesian_cpu(&aux_x_new, out_x, natom);
    std::memcpy(grad_new, grad, dim * sizeof(Real));

    int consecutive_rejects = 0;
    constexpr int MAX_CONSECUTIVE_REJECTS = 2;

    for (int step = 0; step < max_steps; step++) {
        if (pose_deadline_exceeded()) break;

        // Gradient sanity: detect unphysical states.
        Real g_inf = 0;
        for (int k = 0; k < dim; k++) {
            Real ag = std::fabs(grad[k]);
            if (ag > g_inf) g_inf = ag;
        }
        if (!std::isfinite(g_inf) || g_inf > 1e6f) {
            spdlog::warn("FF Cartesian BFGS: |grad|_inf={:.2e} at step {}; rolling back.", g_inf, step);
            duplicate_pose_cartesian_cpu(out_x, &aux_x_ori, natom);
            E = E_ori;
            break;
        }

        // Search direction p = -H * g
        minus_mat_vec_product_flat_cpu(p, h, grad, dim);

        line_search_ff_cartesian_cpu(fix_mol, ff, out_x, grad, flex_topo,
                                     p, aux_f, grad_new, &aux_x_new,
                                     E, dim, &E1, &alpha, constrained_atom_mask);

        // Reject step if line search did not actually descend.
        if (E1 >= E - 1e-6f) {
            consecutive_rejects++;
            if (consecutive_rejects >= MAX_CONSECUTIVE_REJECTS) {
                spdlog::info("FF Cartesian BFGS: no descent in {} consecutive line searches at step {}; stopping.",
                             consecutive_rejects, step);
                break;
            }
            for (int i = 0; i < dim; i++) {
                h[tri_mat_index(i, i)] *= 0.5f;
            }
            continue;
        }
        consecutive_rejects = 0;

        // y = grad_new - grad
        for (int k = 0; k < dim; k++) y[k] = grad_new[k] - grad[k];

        // Accept step: copy coords from aux_x_new to out_x.
        E = E1;
        std::memcpy(out_x->coords, aux_x_new.coords, natom * 3 * sizeof(Real));

        // Bond drift guard: rollback if any |r - r0| / r0 > 0.20.
        bool drift_violated = false;
        for (const auto& b : ff.lig_bonds) {
            int i = b.i, j = b.j;
            Real dx = out_x->coords[j*3]   - out_x->coords[i*3];
            Real dy = out_x->coords[j*3+1] - out_x->coords[i*3+1];
            Real dz = out_x->coords[j*3+2] - out_x->coords[i*3+2];
            Real r = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (b.r0 > 1e-6f && std::fabs(r - b.r0) / b.r0 > 0.20f) {
                spdlog::warn("FF Cartesian BFGS: bond ({},{}) drifted r={:.3f} A vs r0={:.3f} A "
                             "(ratio={:.2f}) at step {}; rolling back.",
                             i, j, r, b.r0, std::fabs(r - b.r0) / b.r0, step);
                drift_violated = true;
                break;
            }
        }
        if (drift_violated) {
            duplicate_pose_cartesian_cpu(out_x, &aux_x_ori, natom);
            E = E_ori;
            break;
        }

        // Convergence check (per-atom RMS gradient).
        Real gg = 0;
        for (int k = 0; k < dim; k++) gg += grad[k] * grad[k];
        if (std::sqrt(gg) < 0.1f) {
            std::memcpy(grad, grad_new, dim * sizeof(Real));
            break;
        }

        bfgs_update_hessian_flat_cpu(h, dim, p, y, minus_hy, alpha);

        // Promote grad_new -> grad for next iteration.
        std::memcpy(grad, grad_new, dim * sizeof(Real));
    }

    if (E > E_ori) {
        E = E_ori;
        duplicate_pose_cartesian_cpu(out_x, &aux_x_ori, natom);
    }
    out_x->energy = E;
}


/// BFGS optimizer for FF scoring (mirrors bfgs_cpu but uses FF energy functions)
void bfgs_ff_cpu(FlexPose* out_x, const FlexTopo& flex_topo,
                        const FixMol& fix_mol, const FFMinData& ff,
                        int max_steps, BFGSThreadMemory& mem) {
    int natom = flex_topo.natom;
    int ntorsion = flex_topo.ntorsion;
    mem.resize(natom, ntorsion);

    FlexPose aux_x_new, aux_x_ori;
    aux_x_new.coords = mem.coords_new.data();
    aux_x_new.dihedrals = mem.dihedrals_new.data();
    aux_x_ori.coords = mem.coords_ori.data();
    aux_x_ori.dihedrals = mem.dihedrals_ori.data();

    FlexPoseGradient aux_g, aux_g_new, aux_g_ori, aux_p, aux_y, aux_minus_hy;
    aux_g.dihedrals_g = mem.dihedrals_g.data();
    aux_g_new.dihedrals_g = mem.dihedrals_g_new.data();
    aux_g_ori.dihedrals_g = mem.dihedrals_g_ori.data();
    aux_p.dihedrals_g = mem.dihedrals_p.data();
    aux_y.dihedrals_g = mem.dihedrals_y.data();
    aux_minus_hy.dihedrals_g = mem.dihedrals_minus_hy.data();

    FlexPoseHessian aux_h;
    aux_h.matrix = mem.h_matrix.data();

    Real* aux_f = mem.forces.data();

    int dim_g = dof_g + ntorsion;
    int dim_x = dof_x + ntorsion;

    // Initial energy and gradient
    Real E_ori = cal_e_grad_ff_cpu(out_x, &aux_g, flex_topo, fix_mol, ff, aux_f);
    Real E = E_ori;
    Real E1 = 0, alpha = 0;

    // Initialize Hessian as identity
    int h_size = dim_g * (dim_g + 1) / 2;
    std::memset(aux_h.matrix, 0, h_size * sizeof(Real));
    for (int i = 0; i < dim_g; i++) {
        aux_h.matrix[tri_mat_index(i, i)] = 1.0;
    }

    duplicate_grad_cpu(&aux_g_new, &aux_g, dim_g);
    duplicate_pose_cpu(&aux_x_new, out_x, dim_x, natom);
    duplicate_grad_cpu(&aux_g_ori, &aux_g, dim_g);
    duplicate_pose_cpu(&aux_x_ori, out_x, dim_x, natom);
    duplicate_grad_cpu(&aux_p, &aux_g, dim_g);

    for (int step = 0; step < max_steps; step++) {
        minus_mat_vec_product_cpu(&aux_p, aux_h.matrix, &aux_g, dim_g);

        line_search_ff_cpu(fix_mol, ff, out_x, &aux_g, flex_topo,
                           &aux_p, aux_f, &aux_x_new, &aux_g_new,
                           E, dim_g, dim_x, &E1, &alpha);

        // y = g_new - g
        duplicate_grad_cpu(&aux_y, &aux_g_new, dim_g);
        for (int i = 0; i < dim_g; i++) {
            Real tmp = grad_index_read(&aux_y, i) - grad_index_read(&aux_g, i);
            grad_index_write(&aux_y, i, tmp);
        }

        E = E1;
        duplicate_pose_cpu(out_x, &aux_x_new, dim_x, natom);

        Real gg = g_dot_product_cpu(&aux_g, &aux_g, dim_g);
        // FF gradients are much larger than Vina (hundreds of kcal/mol scale),
        // so use a looser convergence threshold than Vina's 1e-5.
        if (std::sqrt(gg) < 0.1f) break;

        duplicate_grad_cpu(&aux_g, &aux_g_new, dim_g);

        if (step == 0) {
            Real yy = g_dot_product_cpu(&aux_y, &aux_y, dim_g);
            if (std::abs(yy) > std::numeric_limits<Real>::epsilon()) {
                Real yp = g_dot_product_cpu(&aux_y, &aux_p, dim_g);
                Real scale = alpha * yp / yy;
                for (int i = 0; i < dim_g; i++) {
                    aux_h.matrix[tri_mat_index(i, i)] = scale;
                }
            }
        }

        bfgs_update_hessian_cpu(aux_h.matrix, dim_g, &aux_p, &aux_y, &aux_minus_hy, alpha);
    }

    // Restore if optimization failed
    if (E > E_ori) {
        E = E_ori;
        duplicate_pose_cpu(out_x, &aux_x_ori, dim_x, natom);
    }
    out_x->energy = E;
}


/// Validate that an FFMinData carries the harmonic bonded terms required for
/// Cartesian-DOF minimization. Without bond stretch + angle bend params the 3N
/// Cartesian DOF are unrestrained and minimization distorts the geometry, so
/// each missing component is reported as an error. Returns true if complete.
static bool validate_ff_for_cartesian(const FFMinData& ff, int flex_idx) {
    bool ok = true;
    if (ff.lig_atoms.natom <= 0) {
        spdlog::error("FF Cartesian param check (flex {}): no ligand atoms / nonbonded params loaded.",
                      flex_idx);
        ok = false;
    }
    if (ff.lig_atoms.natom >= 2 && ff.lig_bonds.empty()) {
        spdlog::error("FF Cartesian param check (flex {}): missing bond stretch params "
                      "(stretch_harm) — bond lengths would be unrestrained. "
                      "Ligand DMS must contain stretch_harm_term/stretch_harm_param.", flex_idx);
        ok = false;
    }
    if (ff.lig_atoms.natom >= 3 && ff.lig_angles.empty()) {
        spdlog::error("FF Cartesian param check (flex {}): missing angle bend params "
                      "(angle_harm) — bond angles would be unrestrained. "
                      "Ligand DMS must contain angle_harm_term/angle_harm_param.", flex_idx);
        ok = false;
    }
    return ok;
}


/// The ligand with its FF-only torsions after the search torsions (dihedrals likewise): the fields
/// fill_flex_topo_host reads, for the FF topology.
static UDFlexMol ff_torsion_mol(const UDFlexMol& m) {
    UDFlexMol t;
    t.natom = m.natom;
    t.vina_types = m.vina_types;
    t.torsions = m.torsions;
    t.torsions.insert(t.torsions.end(), m.ff_only_torsions.begin(), m.ff_only_torsions.end());
    t.dihedrals = m.dihedrals;
    t.dihedrals.insert(t.dihedrals.end(), m.ff_only_dihedrals.begin(), m.ff_only_dihedrals.end());
    return t;
}


void DockTask::run_ff_minimize() {
    bool has_ff = !ff_min_data_list.empty() || ff_min_data.loaded;
    if (!has_ff || dock_param.ff_minimize_steps <= 0) {
        return;
    }

#if USE_CUDA
    // Device-resident topologies are unreadable here; rebuild host copies.
    std::vector<HostFlexTopo> host_topos(nflex);
    for (int i = 0; i < nflex; i++) {
        if (udflex_mols[i].ff_only_torsions.empty()) {
            build_host_flex_topo(udflex_mols[i], host_topos[i]);
        } else {
            build_host_flex_topo(ff_torsion_mol(udflex_mols[i]), host_topos[i]);
        }
    }
    FixMol fix_mol_host;
    fix_mol_host.natom = udfix_mol.natom;
    fix_mol_host.coords = udfix_mol.coords.data();
    const FixMol& fix_mol = fix_mol_host;
    auto topo_at = [&host_topos](int i) -> const FlexTopo& { return host_topos[i].topo; };
#else
    const FixMol& fix_mol = *fix_mol_cu;
    // The search topology misses the FF-only torsions: build the FF one in place for those ligands
    std::vector<HostFlexTopo> ff_topos(nflex);
    for (int i = 0; i < nflex; i++) {
        if (udflex_mols[i].ff_only_torsions.empty()) continue;
        build_host_flex_topo(ff_torsion_mol(udflex_mols[i]), ff_topos[i]);
    }
    auto topo_at = [this, &ff_topos](int i) -> const FlexTopo& {
        return udflex_mols[i].ff_only_torsions.empty() ? flex_topo_list_cu[i] : ff_topos[i].topo;
    };
#endif

    spdlog::info("FF Minimization (BFGS, mode={}) for {} steps on {} flex molecules...",
                 FFMIN_MODE_NAMES[static_cast<int>(dock_param.ff_minimize_mode)],
                 dock_param.ff_minimize_steps, nflex);

    for (int i = 0; i < nflex; i++) {
        // Select per-ligand or single FF data
        const FFMinData& ff_ref = !ff_min_data_list.empty() ? ff_min_data_list[i] : ff_min_data;
        // Merged batches can mix ligands with and without FF parameters
        if (!ff_ref.loaded) continue;

        // Cartesian mode needs full bonded FF; skip (don't distort) if incomplete.
        if (dock_param.ff_minimize_mode == FFMIN_CARTESIAN &&
            !validate_ff_for_cartesian(ff_ref, i)) {
            spdlog::error("Skipping FF minimization for flex {}: incomplete FF params for Cartesian "
                          "mode. Supply a ligand DMS with stretch_harm + angle_harm, or use "
                          "ff_minimize_mode=0 (INTERNAL). Poses left unminimized.", i);
            continue;
        }

        auto& filtered_inds = filtered_pose_inds_list[i];
        spdlog::info("  Flex {}: {} poses to minimize", i, filtered_inds.size());

        // Hessian memory warning for Cartesian mode on large ligands.
        if (dock_param.ff_minimize_mode == FFMIN_CARTESIAN &&
            topo_at(i).natom > 500) {
            spdlog::warn("FF Cartesian BFGS on flex {} with natom={} : Hessian is 3N x 3N "
                         "(~{:.1f} MB/thread). Consider reducing exhaustiveness or using INTERNAL mode.",
                         i, topo_at(i).natom,
                         (3.0 * topo_at(i).natom) *
                         (3.0 * topo_at(i).natom + 1) / 2.0 * sizeof(Real) / (1024.0 * 1024.0));
        }

        // Thread-local BFGS memory
        std::vector<BFGSThreadMemory> thread_mems(get_num_threads());

        // Build per-ligand position-constraint mask once (read-only inside parallel region).
        const int natom_i_outer = topo_at(i).natom;
        std::vector<bool> ff_mask;
        const auto& cap = dock_param.constrained_atom_indices_per_flex;
        if (dock_param.ff_minimize_mode == FFMIN_CARTESIAN &&
            i < static_cast<int>(cap.size()) && !cap[i].empty()) {
            ff_mask.assign(natom_i_outer, false);
            for (int idx : cap[i]) {
                if (idx < 0 || idx >= natom_i_outer) {
                    spdlog::warn("constrained_atom_indices[{}] idx {} out of [0,{}) -- ignored",
                                 i, idx, natom_i_outer);
                    continue;
                }
                ff_mask[idx] = true;
            }
        }
        if (!ff_mask.empty() && dock_param.ff_minimize_mode != FFMIN_CARTESIAN) {
            spdlog::warn("constrained_atom_indices ignored: only Cartesian FF mode honors mask");
        }
        const std::vector<bool>* mask_ptr = ff_mask.empty() ? nullptr : &ff_mask;
        const std::vector<Real>& ff_only_dihedrals = udflex_mols[i].ff_only_dihedrals;
        const int n_search_dihe = udflex_mols[i].dihedrals.size();

        #pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < static_cast<int>(filtered_inds.size()); k++) {
            int j = filtered_inds[k];
            FlexPose* res_pose = flex_pose_list_res + j;
            FlexPose* pose = res_pose;

            // With FF-only torsions, minimise a view sharing the coords whose dihedrals are the search
            // ones followed by the FF-only ones at their input values
            FlexPose ff_pose;
            std::vector<Real> ff_dihedrals;
            if (!ff_only_dihedrals.empty()) {
                ff_pose = *res_pose;
                ff_dihedrals.assign(res_pose->dihedrals, res_pose->dihedrals + n_search_dihe);
                ff_dihedrals.insert(ff_dihedrals.end(), ff_only_dihedrals.begin(), ff_only_dihedrals.end());
                ff_pose.dihedrals = ff_dihedrals.data();
                pose = &ff_pose;
            }

            BFGSThreadMemory& mem = thread_mems[get_thread_id()];

            // Compute initial FF energy (for logging; BFGS also computes it internally)
            int natom_i = topo_at(i).natom;
            std::vector<Real> coords_before(pose->coords, pose->coords + natom_i * 3);
            std::vector<Real> tmp_f(natom_i * 3, 0);
            Real ff_e_before = cal_e_f_ff_cpu(pose, topo_at(i), fix_mol, ff_ref, tmp_f.data());

            if (dock_param.ff_minimize_mode == FFMIN_CARTESIAN) {
                bfgs_ff_cartesian_cpu(pose, topo_at(i), fix_mol, ff_ref,
                                      dock_param.ff_minimize_steps, mem, mask_ptr);
            } else {
                bfgs_ff_cpu(pose, topo_at(i), fix_mol, ff_ref,
                            dock_param.ff_minimize_steps, mem);
            }
            Real ff_e_after = pose->energy;
            if (pose != res_pose) {
                res_pose->energy = pose->energy;
                std::memcpy(res_pose->center, pose->center, sizeof(res_pose->center));
                std::memcpy(res_pose->rot_vec, pose->rot_vec, sizeof(res_pose->rot_vec));
                std::memcpy(res_pose->dihedrals, pose->dihedrals, n_search_dihe * sizeof(Real));
            }

            // Compute RMSD displacement
            Real sum_sq = 0;
            for (int a = 0; a < natom_i * 3; a++) {
                Real d = pose->coords[a] - coords_before[a];
                sum_sq += d * d;
            }
            Real rmsd = std::sqrt(sum_sq / natom_i);
            bool rolled_back = (ff_e_after >= ff_e_before - 1e-6f);
            spdlog::info("    Pose {}: FF E {:.3f} -> {:.3f} (delta={:.3f}), RMSD={:.4f}{}",
                         k, ff_e_before, ff_e_after, ff_e_after - ff_e_before, rmsd,
                         rolled_back ? " [no improvement]" : "");
        }
    }

    spdlog::info("FF Minimization is done.");
}


