//
// CPU Energy and Force Calculation Implementation
// Created for Uni-Dock2 CPU backend
//
// OPTIMIZATIONS APPLIED:
// 1. Static Vina scorer (avoid repeated construction)
// 2. Distance check BEFORE box penalty (skip ~95% of pairs early)
// 3. Pre-cache ligand coordinates (reduce repeated memory lookups)
// 4. Combined penalty calculation with inter-molecular loop
//

#include "score_cpu.h"
#include "myutils/matrix.h"
#include <cstring>
#include <cmath>
#include <algorithm>

// Epsilon for numerical stability
static const Real EPSILON_CPU = std::numeric_limits<Real>::epsilon();

// Vina-style energy capping (curl) to smooth pathological energy landscapes.
// Caps repulsive (positive) pairwise energy at ~v, squashes gradient by (v/(v+e))^2.
// Matches AutoDock Vina's hunt_cap approach (v=10 for pair-based evaluation).
// Without this, deeply clashing poses produce extreme gradients that make BFGS
// extremely slow on CPU, causing apparent hangs.
static const Real CURL_V = 10.0;

inline void curl_cpu(Real& e, Real& f, Real v) {
    if (e > 0) {
        Real tmp = v / (v + e);
        e *= tmp;
        f *= tmp * tmp;
    }
}

// Thread-local scorer to avoid repeated construction (optimization #1)
static thread_local Vina g_vina_scorer;

Real cal_e_f_cpu(const FlexPose* pose, const FlexTopo& flex_topo,
                 const FixMol& fix_mol, const FlexParamVina& flex_param,
                 const FixParamVina& fix_param, Real* aux_f) {
    Real energy = 0.0;
    g_vina_scorer.set_weights(g_cpu_constants.weights);
    const Real r2_cutoff = g_vina_scorer.r2_cutoff;
    const Real w_elec = g_vina_scorer.weight_elec_val();
    const Real* lig_q = (w_elec != Real(0)) ? flex_param.charges : nullptr;

    const int natom = flex_topo.natom;
    const Real* __restrict__ pose_coords = pose->coords;
    const Real* __restrict__ fix_coords = fix_mol.coords;

    // Initialize aux_f as zero
    std::memset(aux_f, 0, natom * 3 * sizeof(Real));

    // 1. Compute inter-molecular energy: flex-protein via spatial grid
    const int* __restrict__ flex_types = flex_param.atom_types;
    const SpatialGrid& grid = g_cpu_constants.receptor_grid;
    const bool use_angular = g_cpu_constants.USE_ANGULAR_HBOND;
    const auto& atp = g_cpu_constants.atp;

    // Use pre-built bonded-H topology for angular HBond scoring
    const BondedHTopo& h_topo = flex_param.bonded_h_topo;
    const RecDonorHTopo& rh_topo = g_cpu_constants.rec_donor_h_topo;

    if (grid.built) {
        const int ny_nz = grid.ny * grid.nz;

        for (int i_lig = 0; i_lig < natom; i_lig++) {
            const int lig_type = flex_types[i_lig];
            if (atp.is_hydrogen[lig_type]) continue;

            const int i1_3 = i_lig * 3;
            const Real lig_x = pose_coords[i1_3];
            const Real lig_y = pose_coords[i1_3 + 1];
            const Real lig_z = pose_coords[i1_3 + 2];

            // Pre-compute ligand atom VDW radius (data-driven)
            const Real r1 = g_cpu_constants.vdw_radii[lig_type];

            // Check ligand atom donor/acceptor status (for angular HBond)
            const bool lig_is_donor = use_angular && atp.is_donor[lig_type];
            const bool lig_is_acceptor = use_angular && atp.is_acceptor[lig_type];

            // Determine grid cell of this ligand atom
            int cx = (int)((lig_x - grid.origin[0]) * grid.inv_cell_size);
            int cy = (int)((lig_y - grid.origin[1]) * grid.inv_cell_size);
            int cz = (int)((lig_z - grid.origin[2]) * grid.inv_cell_size);
            cx = std::clamp(cx, 0, grid.nx - 1);
            cy = std::clamp(cy, 0, grid.ny - 1);
            cz = std::clamp(cz, 0, grid.nz - 1);

            // Determine neighbor cell range (clamp to grid bounds)
            const int cx_lo = std::max(0, cx - grid.half);
            const int cx_hi = std::min(grid.nx - 1, cx + grid.half);
            const int cy_lo = std::max(0, cy - grid.half);
            const int cy_hi = std::min(grid.ny - 1, cy + grid.half);
            const int cz_lo = std::max(0, cz - grid.half);
            const int cz_hi = std::min(grid.nz - 1, cz + grid.half);

            // Loop over 27 (or fewer at boundaries) neighboring cells
            for (int ix = cx_lo; ix <= cx_hi; ix++) {
                for (int iy = cy_lo; iy <= cy_hi; iy++) {
                    for (int iz = cz_lo; iz <= cz_hi; iz++) {
                        const int cell = ix * ny_nz + iy * grid.nz + iz;
                        const int start = grid.cell_start[cell];
                        const int end   = grid.cell_start[cell + 1];

                        for (int k = start; k < end; k++) {
                            const GridAtom& rec = grid.atoms[k];
                            const int i2_3 = rec.index * 3;
                            const Real rec_x = fix_coords[i2_3];
                            const Real rec_y = fix_coords[i2_3 + 1];
                            const Real rec_z = fix_coords[i2_3 + 2];

                            Real r_vec_x = rec_x - lig_x;
                            Real r_vec_y = rec_y - lig_y;
                            Real r_vec_z = rec_z - lig_z;
                            Real rr = r_vec_x*r_vec_x + r_vec_y*r_vec_y + r_vec_z*r_vec_z;

                            if (rr >= r2_cutoff) continue;

                            Real r = std::sqrt(rr);
                            Real d = r - (r1 + rec.radius);
                            Real f_div_r = 0;
                            Real e_pair = g_vina_scorer.eval_ef(
                                d, lig_type, rec.type,
                                &f_div_r, atp);
                            if (lig_q != nullptr) {
                                // united-atom screened Coulomb (dE/dr adds to dE/dd: d = r - const)
                                Real f_el = 0;
                                Real e_el = g_vina_scorer.elec_pair(lig_q[i_lig] * rec.q, r, &f_el);
                                e_pair += w_elec * e_el;
                                f_div_r += w_elec * f_el;
                            }

                            // Angular HBond: adjust HBond term by w(angle) BEFORE curl.
                            // HBond energy is always <= 0, so curl (which only clips e>0)
                            // does not affect it — the adjustment commutes with curl.
                            // Handles BOTH directions:
                            //   1. Ligand donor -> receptor acceptor (ligand bonded H, or the best
                            //      hydrogen on the bond cone for rotatable donors: -OH, -NH2, -NH3+)
                            //   2. Receptor donor -> ligand acceptor (receptor H table, same cone rule)
                            // Donors without hydrogens and without a heavy neighbour (water O,
                            // metals) carry no directional information and keep full weight.
                            // For DA-DA pairs, picks the direction with better angular weight.
                            // ang_mode: 0 none, 1 explicit ligand H (index ang_idx), 2 ligand cone
                            // (ang_idx = ligand antecedent), 3 explicit receptor H (coords ang_h*),
                            // 4 receptor cone (ang_idx = receptor antecedent)
                            int ang_mode = 0, ang_idx = -1;
                            Real ang_hx = 0, ang_hy = 0, ang_hz = 0;
                            Real angular_dw_dcos = 0, angular_cos = 1, angular_e_hb = 0;
                            const bool hb_possible = (lig_is_donor || lig_is_acceptor)
                                && xs_h_bond_possible(lig_type, rec.type, atp);
                            if (hb_possible) {
                                Real f_hb = 0;
                                Real e_hb = g_vina_scorer.weight_hbond_val()
                                            * g_vina_scorer.vina_hbond(d, &f_hb);
                                f_hb *= g_vina_scorer.weight_hbond_val();

                                if (std::abs(e_hb) > Real(1e-12)) {
                                    Real best_w = Real(0), best_dw = Real(0), best_cos = Real(1);
                                    int best_mode = 0, best_idx = -1;
                                    Real bhx = 0, bhy = 0, bhz = 0;
                                    const Real D1[3] = {lig_x, lig_y, lig_z};
                                    const Real A1[3] = {rec_x, rec_y, rec_z};

                                    auto consider_cos = [&](Real cos_dha, int mode, int idx) {
                                        Real dw = 0;
                                        Real w = angular_hbond_weight(cos_dha, &dw);
                                        if (w > best_w) {
                                            best_w = w; best_dw = dw; best_cos = cos_dha;
                                            best_mode = mode; best_idx = idx;
                                        }
                                    };
                                    auto consider_h = [&](Real hx, Real hy, Real hz, const Real* Dp, const Real* Ap,
                                                          int mode, int idx) {
                                        Real hd_x = Dp[0]-hx, hd_y = Dp[1]-hy, hd_z = Dp[2]-hz;
                                        Real ha_x = Ap[0]-hx, ha_y = Ap[1]-hy, ha_z = Ap[2]-hz;
                                        Real hd_sq = hd_x*hd_x + hd_y*hd_y + hd_z*hd_z;
                                        Real ha_sq = ha_x*ha_x + ha_y*ha_y + ha_z*ha_z;
                                        if (hd_sq < Real(1e-12) || ha_sq < Real(1e-12)) return;
                                        Real cos_dha = (hd_x*ha_x + hd_y*ha_y + hd_z*ha_z)
                                                       / (std::sqrt(hd_sq) * std::sqrt(ha_sq));
                                        const Real w_before = best_w;
                                        consider_cos(cos_dha, mode, idx);
                                        if (best_w > w_before) { bhx = hx; bhy = hy; bhz = hz; }
                                    };
                                    auto full_credit = [&]() {
                                        if (best_w < Real(1)) { best_w = Real(1); best_dw = 0; best_mode = 0; }
                                    };

                                    // Direction 1: ligand donor -> receptor acceptor
                                    if (lig_is_donor && atp.is_acceptor[rec.type]) {
                                        const int hb = h_topo.h_start[i_lig];
                                        const int he = h_topo.h_start[i_lig + 1];
                                        const int n_h = he - hb;
                                        const bool rot = h_topo.donor_rotatable ? h_topo.donor_rotatable[i_lig] != 0
                                                                                : hbond_donor_rotatable(lig_type, n_h);
                                        if (rot) {
                                            const int x = h_topo.antecedent ? h_topo.antecedent[i_lig] : -1;
                                            if (x >= 0) {
                                                consider_cos(hbond_cone_cos(D1, pose_coords + x * 3, A1, nullptr, nullptr, nullptr), 2, x);
                                            } else {
                                                full_credit();
                                            }
                                        } else if (n_h == 0) {
                                            full_credit();
                                        } else {
                                            for (int ih = hb; ih < he; ih++) {
                                                const int hi = h_topo.h_indices[ih];
                                                consider_h(pose_coords[hi*3], pose_coords[hi*3+1], pose_coords[hi*3+2], D1, A1, 1, hi);
                                            }
                                        }
                                    }

                                    // Direction 2: receptor donor -> ligand acceptor
                                    if (lig_is_acceptor && atp.is_donor[rec.type]
                                        && rh_topo.natom > 0) {
                                        const int rs = rh_topo.h_start[rec.index];
                                        const int re = rh_topo.h_start[rec.index + 1];
                                        const int n_h = re - rs;
                                        const bool rot = rh_topo.donor_rotatable ? rh_topo.donor_rotatable[rec.index] != 0
                                                                                 : hbond_donor_rotatable(rec.type, n_h);
                                        if (rot) {
                                            const int x = rh_topo.antecedent ? rh_topo.antecedent[rec.index] : -1;
                                            if (x >= 0) {
                                                consider_cos(hbond_cone_cos(A1, fix_coords + x * 3, D1, nullptr, nullptr, nullptr), 4, x);
                                            } else {
                                                full_credit();
                                            }
                                        } else if (n_h == 0) {
                                            full_credit();
                                        } else {
                                            for (int ih = rs; ih < re; ih++) {
                                                consider_h(rh_topo.h_coords[ih*3], rh_topo.h_coords[ih*3+1], rh_topo.h_coords[ih*3+2],
                                                           A1, D1, 3, -1);
                                            }
                                        }
                                    }

                                    // Adjust: E = E_non_hb + w*E_hb = E_total + (w-1)*E_hb
                                    e_pair  += (best_w - Real(1)) * e_hb;
                                    f_div_r += (best_w - Real(1)) * f_hb;

                                    ang_mode = best_mode; ang_idx = best_idx;
                                    ang_hx = bhx; ang_hy = bhy; ang_hz = bhz;
                                    angular_dw_dcos = best_dw;
                                    angular_cos = best_cos;
                                    angular_e_hb = e_hb;
                                }
                            }

                            // Curl the (possibly adjusted) pair energy
                            curl_cpu(e_pair, f_div_r, CURL_V);
                            energy += e_pair;

                            if (r < EPSILON_CPU) r = EPSILON_CPU;
                            f_div_r /= r;

                            aux_f[i1_3]     += -f_div_r * r_vec_x;
                            aux_f[i1_3 + 1] += -f_div_r * r_vec_y;
                            aux_f[i1_3 + 2] += -f_div_r * r_vec_z;

                            // Angular gradient on movable ligand atoms
                            // E_ang = w(theta)*E_hb(d), dE/d(atom) = E_hb * dw/dcos * dcos/d(atom)
                            // HBond energy is negative, curl doesn't affect it -> use raw e_hb.
                            if (ang_mode != 0 && std::abs(angular_dw_dcos) > Real(1e-12)) {
                                const Real scale = angular_e_hb * angular_dw_dcos;
                                if (ang_mode == 1) {
                                    // Direction 1, explicit H: ligand donor D and ligand H
                                    const Real hx = ang_hx, hy = ang_hy, hz = ang_hz;
                                    Real hd_x = lig_x-hx, hd_y = lig_y-hy, hd_z = lig_z-hz;
                                    Real ha_x = rec_x-hx, ha_y = rec_y-hy, ha_z = rec_z-hz;
                                    Real inv_hd = Real(1)/std::sqrt(hd_x*hd_x + hd_y*hd_y + hd_z*hd_z);
                                    Real inv_ha = Real(1)/std::sqrt(ha_x*ha_x + ha_y*ha_y + ha_z*ha_z);
                                    Real inv_hd_ha = inv_hd * inv_ha;
                                    Real hd_hat_x = hd_x*inv_hd, hd_hat_y = hd_y*inv_hd, hd_hat_z = hd_z*inv_hd;
                                    Real ha_hat_x = ha_x*inv_ha, ha_hat_y = ha_y*inv_ha, ha_hat_z = ha_z*inv_ha;

                                    Real dcos_dH_x = -inv_hd_ha*(lig_x+rec_x-2*hx) + angular_cos*(hd_hat_x*inv_hd + ha_hat_x*inv_ha);
                                    Real dcos_dH_y = -inv_hd_ha*(lig_y+rec_y-2*hy) + angular_cos*(hd_hat_y*inv_hd + ha_hat_y*inv_ha);
                                    Real dcos_dH_z = -inv_hd_ha*(lig_z+rec_z-2*hz) + angular_cos*(hd_hat_z*inv_hd + ha_hat_z*inv_ha);

                                    Real dcos_dD_x = inv_hd_ha*ha_x - angular_cos*hd_hat_x*inv_hd;
                                    Real dcos_dD_y = inv_hd_ha*ha_y - angular_cos*hd_hat_y*inv_hd;
                                    Real dcos_dD_z = inv_hd_ha*ha_z - angular_cos*hd_hat_z*inv_hd;

                                    const int h3 = ang_idx * 3;
                                    aux_f[h3]     += scale * dcos_dH_x;
                                    aux_f[h3 + 1] += scale * dcos_dH_y;
                                    aux_f[h3 + 2] += scale * dcos_dH_z;
                                    aux_f[i1_3]     += scale * dcos_dD_x;
                                    aux_f[i1_3 + 1] += scale * dcos_dD_y;
                                    aux_f[i1_3 + 2] += scale * dcos_dD_z;
                                }
                                else if (ang_mode == 2) {
                                    // Direction 1, cone H: ligand donor D and its antecedent X
                                    const Real D1[3] = {lig_x, lig_y, lig_z};
                                    const Real A1[3] = {rec_x, rec_y, rec_z};
                                    Real gD[3], gX[3], gA[3];
                                    hbond_cone_cos(D1, pose_coords + ang_idx * 3, A1, gD, gX, gA);
                                    const int x3 = ang_idx * 3;
                                    aux_f[i1_3]     += scale * gD[0];
                                    aux_f[i1_3 + 1] += scale * gD[1];
                                    aux_f[i1_3 + 2] += scale * gD[2];
                                    aux_f[x3]     += scale * gX[0];
                                    aux_f[x3 + 1] += scale * gX[1];
                                    aux_f[x3 + 2] += scale * gX[2];
                                }
                                else if (ang_mode == 3) {
                                    // Direction 2, explicit receptor H: gradient only on the ligand
                                    // acceptor A; receptor D and H are static.
                                    const Real hx = ang_hx, hy = ang_hy, hz = ang_hz;
                                    Real hd_x = rec_x-hx, hd_y = rec_y-hy, hd_z = rec_z-hz;
                                    Real ha_x = lig_x-hx, ha_y = lig_y-hy, ha_z = lig_z-hz;
                                    Real inv_hd = Real(1)/std::sqrt(hd_x*hd_x + hd_y*hd_y + hd_z*hd_z);
                                    Real inv_ha = Real(1)/std::sqrt(ha_x*ha_x + ha_y*ha_y + ha_z*ha_z);
                                    Real inv_hd_ha = inv_hd * inv_ha;
                                    Real ha_hat_x = ha_x*inv_ha, ha_hat_y = ha_y*inv_ha, ha_hat_z = ha_z*inv_ha;

                                    aux_f[i1_3]     += scale * (inv_hd_ha*hd_x - angular_cos*ha_hat_x*inv_ha);
                                    aux_f[i1_3 + 1] += scale * (inv_hd_ha*hd_y - angular_cos*ha_hat_y*inv_ha);
                                    aux_f[i1_3 + 2] += scale * (inv_hd_ha*hd_z - angular_cos*ha_hat_z*inv_ha);
                                }
                                else {
                                    // Direction 2, receptor cone H: gradient only on the ligand acceptor
                                    const Real D2[3] = {rec_x, rec_y, rec_z};
                                    const Real A2[3] = {lig_x, lig_y, lig_z};
                                    Real gD[3], gX[3], gA[3];
                                    hbond_cone_cos(D2, fix_coords + ang_idx * 3, A2, gD, gX, gA);
                                    aux_f[i1_3]     += scale * gA[0];
                                    aux_f[i1_3 + 1] += scale * gA[1];
                                    aux_f[i1_3 + 2] += scale * gA[2];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Compute penalty energy for all non-H atoms
    // NOTE: This must be done for ALL atoms, not tracked through inter loop
    for (int i = 0; i < natom; i++) {
        if (!g_cpu_constants.atp.is_hydrogen[flex_types[i]]) {
            Real coord_adj_x = pose_coords[i * 3];
            Real coord_adj_y = pose_coords[i * 3 + 1];
            Real coord_adj_z = pose_coords[i * 3 + 2];
            Real f_tmp[3] = {0, 0, 0};
            energy += cal_box_penalty_clamp_coord_cpu(&coord_adj_x, &coord_adj_y, &coord_adj_z, f_tmp);
            
            if (std::abs(f_tmp[0]) > EPSILON_CPU) aux_f[i * 3] += f_tmp[0];
            if (std::abs(f_tmp[1]) > EPSILON_CPU) aux_f[i * 3 + 1] += f_tmp[1];
            if (std::abs(f_tmp[2]) > EPSILON_CPU) aux_f[i * 3 + 2] += f_tmp[2];
        }
    }
    
    // 2b. Inner box: wall penalty on the heavy-atom centroid (same slope as the box walls),
    //     force shared equally by the heavy atoms (aux_f holds +dE/dx)
    if (g_cpu_constants.USE_INNER_BOX) {
        const CPUConstants& c = g_cpu_constants;
        Real sx = 0, sy = 0, sz = 0; int nh = 0;
        for (int i = 0; i < natom; i++) {
            if (c.atp.is_hydrogen[flex_types[i]]) continue;
            sx += pose_coords[i * 3]; sy += pose_coords[i * 3 + 1]; sz += pose_coords[i * 3 + 2]; nh++;
        }
        if (nh > 0) {
            const Real inv = Real(1) / Real(nh);
            const Real cx = sx * inv, cy = sy * inv, cz = sz * inv;
            Real pen = 0, gx = 0, gy = 0, gz = 0;
            if (cx < c.INNER_X_LO) { pen += c.INNER_X_LO - cx; gx = -c.PENALTY_SLOPE; } else if (cx > c.INNER_X_HI) { pen += cx - c.INNER_X_HI; gx = c.PENALTY_SLOPE; }
            if (cy < c.INNER_Y_LO) { pen += c.INNER_Y_LO - cy; gy = -c.PENALTY_SLOPE; } else if (cy > c.INNER_Y_HI) { pen += cy - c.INNER_Y_HI; gy = c.PENALTY_SLOPE; }
            if (cz < c.INNER_Z_LO) { pen += c.INNER_Z_LO - cz; gz = -c.PENALTY_SLOPE; } else if (cz > c.INNER_Z_HI) { pen += cz - c.INNER_Z_HI; gz = c.PENALTY_SLOPE; }
            if (pen > 0) {
                energy += pen * c.PENALTY_SLOPE;
                for (int i = 0; i < natom; i++) {
                    if (c.atp.is_hydrogen[flex_types[i]]) continue;
                    aux_f[i * 3] += gx * inv; aux_f[i * 3 + 1] += gy * inv; aux_f[i * 3 + 2] += gz * inv;
                }
            }
        }
    }

    // 3. Compute intra-molecular energy
    const int npair_intra = flex_param.npair_intra;
    const int* __restrict__ pairs_intra = flex_param.pairs_intra;
    const Real* __restrict__ r1_plus_r2_intra = flex_param.r1_plus_r2_intra;
    
    const Real w_intra = g_vina_scorer.weight_intra_val();
    for (int i = 0; i < npair_intra; i++) {
        const int i1 = pairs_intra[i * 2];
        const int i2 = pairs_intra[i * 2 + 1];
        const int i1_3 = i1 * 3;
        const int i2_3 = i2 * 3;
        
        const Real r_vec_x = pose_coords[i2_3] - pose_coords[i1_3];
        const Real r_vec_y = pose_coords[i2_3 + 1] - pose_coords[i1_3 + 1];
        const Real r_vec_z = pose_coords[i2_3 + 2] - pose_coords[i1_3 + 2];
        const Real rr = r_vec_x*r_vec_x + r_vec_y*r_vec_y + r_vec_z*r_vec_z;
        
        if (rr < r2_cutoff) {
            Real r = std::sqrt(rr);
            Real f_div_r = 0;
            Real e_tmp = g_vina_scorer.eval_ef(r - r1_plus_r2_intra[i],
                                               flex_types[i1],
                                               flex_types[i2], &f_div_r, g_cpu_constants.atp);
            e_tmp *= w_intra;
            f_div_r *= w_intra;
            curl_cpu(e_tmp, f_div_r, CURL_V);
            energy += e_tmp;
            
            if (r < EPSILON_CPU) {
                r = EPSILON_CPU;
            }
            f_div_r /= r;
            
            aux_f[i1_3]     += -f_div_r * r_vec_x;
            aux_f[i1_3 + 1] += -f_div_r * r_vec_y;
            aux_f[i1_3 + 2] += -f_div_r * r_vec_z;
            aux_f[i2_3]     += f_div_r * r_vec_x;
            aux_f[i2_3 + 1] += f_div_r * r_vec_y;
            aux_f[i2_3 + 2] += f_div_r * r_vec_z;
        }
    }

    // 4. Extension terms (score/terms.h): TermSet::accumulate runs every term's per-atom loop under the settings
    //    the ligand's FlexParamVina carries (terms.settings), each term guarded by itself; the position bias is
    //    the former loop of this file, one body shared with the CUDA kernel (ext/bias/bias_term.h).
    flex_param.terms.accumulate(natom, pose_coords, flex_types, energy, aux_f);

    // 4. ligand torsional strain, identical to the GPU block in cuda/bfgs.cuh and to the value
    //    torsion_strain_energy computes for the host rescoring. aux_f holds +dE/dx.
    if (flex_param.n_tors_penalty > 0){
        const Real w_ts = g_vina_scorer.weight_torsion_strain_val();
        if (w_ts != Real(0)){
            for (int r = 0; r < flex_param.n_tors_penalty; r++){
                const TorsionPenaltyRow& row = flex_param.tors_penalty[r];
                Real d[12];
                const Real pen = torsion_penalty_ef(pose->coords + row.atoms[0] * 3,
                                                    pose->coords + row.atoms[1] * 3,
                                                    pose->coords + row.atoms[2] * 3,
                                                    pose->coords + row.atoms[3] * 3,
                                                    row.kind, row.tolerance, d);
                energy += w_ts * row.weight * pen;
                const Real sc = w_ts * row.weight;
                for (int a = 0; a < 4; a++){
                    const int base = row.atoms[a] * 3;
                    aux_f[base]     += sc * d[a * 3];
                    aux_f[base + 1] += sc * d[a * 3 + 1];
                    aux_f[base + 2] += sc * d[a * 3 + 2];
                }
            }
        }
    }

    return energy;
}


void cal_grad_cpu(const Real* aux_f, const FlexTopo& flex_topo,
                  const FlexPose* pose, FlexPoseGradient* out_g) {
    Real tmp1[3], tmp2[3], tmp3[3], tmp5[3];
    Real mat[9] = {0};
    
    // Calculate gradient on dihedrals
    for (int i_tor = 0; i_tor < flex_topo.ntorsion; i_tor++) {
        int begin = flex_topo.rotated_inds[i_tor * 2];
        int end = flex_topo.rotated_inds[i_tor * 2 + 1];
        int iat_from = flex_topo.axis_atoms[i_tor * 2];
        int iat_to = flex_topo.axis_atoms[i_tor * 2 + 1];
        
        tmp5[0] = tmp5[1] = tmp5[2] = 0.0;
        
        for (int j = begin; j <= end; j++) {
            int i_at = flex_topo.rotated_atoms[j];
            // NOTE: H atoms are NOT skipped — angular HBond scoring puts forces
            // on H atoms that must propagate through torsion gradients.
            // H atoms without forces contribute zero torque (no cost).

            // Compute lever arm vector
            tmp1[0] = pose->coords[i_at * 3] - pose->coords[iat_to * 3];
            tmp1[1] = pose->coords[i_at * 3 + 1] - pose->coords[iat_to * 3 + 1];
            tmp1[2] = pose->coords[i_at * 3 + 2] - pose->coords[iat_to * 3 + 2];
            
            // Take gradient from aux_f
            tmp2[0] = aux_f[i_at * 3];
            tmp2[1] = aux_f[i_at * 3 + 1];
            tmp2[2] = aux_f[i_at * 3 + 2];
            
            // Compute torque: tmp3 = tmp1 x tmp2
            cross_cpu(tmp1, tmp2, tmp3);
            
            // Accumulate torque
            tmp5[0] += tmp3[0];
            tmp5[1] += tmp3[1];
            tmp5[2] += tmp3[2];
        }
        
        // Take the axis
        tmp3[0] = pose->coords[iat_to * 3] - pose->coords[iat_from * 3];
        tmp3[1] = pose->coords[iat_to * 3 + 1] - pose->coords[iat_from * 3 + 1];
        tmp3[2] = pose->coords[iat_to * 3 + 2] - pose->coords[iat_from * 3 + 2];
        
        // Projection on axis as gradient on dihedral
        Real axis_norm = norm_cpu(tmp3);
        if (axis_norm < EPSILON_CPU) axis_norm = EPSILON_CPU;
        out_g->dihedrals_g[i_tor] = dot_cpu(tmp5, tmp3) / axis_norm;
    }
    
    // Derivative on center and orientation (only needed for free docking)
    if (!g_cpu_constants.FLAG_CONSTRAINT_DOCK) {
        Real tmp4[3] = {0, 0, 0};
        std::memset(mat, 0, sizeof(mat));

        Real q[4];
        rotvec_to_quaternion(q, pose->rot_vec);
        quaternion_conjugate(q);

        for (int i_at = 0; i_at < flex_topo.natom; i_at++) {
            // NOTE: H atoms included — angular HBond forces on H must
            // propagate to translation and rotation gradients.

            tmp1[0] = pose->coords[i_at * 3] - pose->center[0];
            tmp1[1] = pose->coords[i_at * 3 + 1] - pose->center[1];
            tmp1[2] = pose->coords[i_at * 3 + 2] - pose->center[2];

            rotate_vec_by_quaternion(tmp1, q);

            tmp2[0] = aux_f[i_at * 3];
            tmp2[1] = aux_f[i_at * 3 + 1];
            tmp2[2] = aux_f[i_at * 3 + 2];

            tmp4[0] += tmp2[0];
            tmp4[1] += tmp2[1];
            tmp4[2] += tmp2[2];

            Real mat_tmp[9];
            outer_product(tmp2, tmp1, mat_tmp);
            for (int j = 0; j < 9; j++) {
                mat[j] += mat_tmp[j];
            }
        }

        out_g->center_g[0] = tmp4[0];
        out_g->center_g[1] = tmp4[1];
        out_g->center_g[2] = tmp4[2];

        for (int i = 0; i < 3; i++) {
            Real dR_dv[9];
            cal_grad_of_rot_over_vec(dR_dv, pose->rot_vec, i);
            out_g->orientation_g[i] = frobenius_product(mat, dR_dv);
        }
    }
}


Real cal_e_grad_cpu(const FlexPose* pose, FlexPoseGradient* out_g,
                    const FlexTopo& flex_topo,
                    const FixMol& fix_mol, const FlexParamVina& flex_param,
                    const FixParamVina& fix_param, Real* aux_f) {
    Real energy = cal_e_f_cpu(pose, flex_topo, fix_mol, flex_param, fix_param, aux_f);
    cal_grad_cpu(aux_f, flex_topo, pose, out_g);
    
    // Add torsion library penalty (soft constraint instead of hard clamping)
    if (g_cpu_constants.USE_TORSION_PENALTY) {
        const Real k = g_cpu_constants.TORSION_PENALTY_K;
        
        for (int i_tor = 0; i_tor < flex_topo.ntorsion; i_tor++) {
            int i_lo = flex_topo.range_inds[i_tor * 2];
            int nrange = flex_topo.range_inds[i_tor * 2 + 1];
            
            if (nrange > 0) {
                Real grad_sign = 0;
                Real dist = distance_to_nearest_range_cpu(
                    pose->dihedrals[i_tor], 
                    flex_topo.range_list + i_lo, 
                    nrange, 
                    &grad_sign
                );
                
                // Quadratic penalty: E_penalty = k * dist^2
                energy += k * dist * dist;
                
                // Gradient: dE/dtheta = 2 * k * dist * sign
                out_g->dihedrals_g[i_tor] += 2.0 * k * dist * grad_sign;
            }
        }
    }
    
    return energy;
}


void duplicate_grad_cpu(FlexPoseGradient* out_g_new, const FlexPoseGradient* g_old, int dim) {
    for (int i = 0; i < 3; i++) {
        out_g_new->center_g[i] = g_old->center_g[i];
    }
    for (int i = 0; i < 3; i++) {
        out_g_new->orientation_g[i] = g_old->orientation_g[i];
    }
    int ntorsion = dim - dof_g;
    for (int i = 0; i < ntorsion; i++) {
        out_g_new->dihedrals_g[i] = g_old->dihedrals_g[i];
    }
}


void duplicate_pose_cpu(FlexPose* out_pose_new, const FlexPose* pose_old, int dim, int natom) {
    for (int i = 0; i < 3; i++) {
        out_pose_new->center[i] = pose_old->center[i];
    }
    for (int i = 0; i < 4; i++) {
        out_pose_new->rot_vec[i] = pose_old->rot_vec[i];
    }
    
    int ntorsion = dim - dof_x;
    for (int i = 0; i < ntorsion; i++) {
        out_pose_new->dihedrals[i] = pose_old->dihedrals[i];
    }
    
    // Copy cartesian coordinates
    std::memcpy(out_pose_new->coords, pose_old->coords, natom * 3 * sizeof(Real));
    
    out_pose_new->energy = pose_old->energy;
}


/**
 * @brief Calculate signed distance to nearest allowed range
 * Returns 0 if angle is within any range, otherwise returns distance to nearest boundary
 * Also outputs the gradient sign (-1 if should decrease, +1 if should increase)
 */
Real distance_to_nearest_range_cpu(Real angle, const Real* range_list, int nrange, Real* out_grad_sign) {
    if (nrange == 0) {
        *out_grad_sign = 0;
        return 0;
    }
    
    // Check if angle is within any range
    for (int i = 0; i < nrange; i++) {
        Real lo = range_list[i * 2];
        Real hi = range_list[i * 2 + 1];
        if (angle >= lo && angle <= hi) {
            *out_grad_sign = 0;
            return 0;
        }
    }
    
    // Find distance to nearest boundary
    Real min_dist = std::numeric_limits<Real>::max();
    Real grad_sign = 0;
    
    for (int i = 0; i < nrange; i++) {
        Real lo = range_list[i * 2];
        Real hi = range_list[i * 2 + 1];
        
        Real dist_lo = angle - lo;  // positive if angle > lo
        Real dist_hi = angle - hi;  // positive if angle > hi
        
        if (std::abs(dist_lo) < min_dist) {
            min_dist = std::abs(dist_lo);
            grad_sign = (dist_lo > 0) ? 1.0 : -1.0;  // push toward lo
        }
        if (std::abs(dist_hi) < min_dist) {
            min_dist = std::abs(dist_hi);
            grad_sign = (dist_hi > 0) ? 1.0 : -1.0;  // push toward hi
        }
    }
    
    *out_grad_sign = grad_sign;
    return min_dist;
}


Real clamp_by_ranges_cpu(Real angle, const Real* range_list, int nrange) {
    if (nrange == 0) return angle;
    
    // Check if angle is within any range
    for (int i = 0; i < nrange; i++) {
        Real lo = range_list[i * 2];
        Real hi = range_list[i * 2 + 1];
        if (angle >= lo && angle <= hi) {
            return angle;
        }
    }
    
    // Find closest range boundary
    Real best_angle = angle;
    Real min_dist = std::numeric_limits<Real>::max();
    
    for (int i = 0; i < nrange; i++) {
        Real lo = range_list[i * 2];
        Real hi = range_list[i * 2 + 1];
        
        Real dist_lo = std::abs(angle - lo);
        Real dist_hi = std::abs(angle - hi);
        
        if (dist_lo < min_dist) {
            min_dist = dist_lo;
            best_angle = lo;
        }
        if (dist_hi < min_dist) {
            min_dist = dist_hi;
            best_angle = hi;
        }
    }
    
    return best_angle;
}


void apply_grad_update_dihe_cpu(FlexPose* out_x, const FlexTopo* flex_topo,
                                 int i_tor, Real dihe_incre_raw,
                                 bool skip_range_clamp) {
    Real dihe_incre = normalize_angle_cpu(dihe_incre_raw);
    
    Real dihe_new = normalize_angle_cpu(out_x->dihedrals[i_tor] + dihe_incre);
    
    // When using penalty method, skip hard clamping - let gradient guide the optimization
    // When not using penalty, apply hard clamping as before
    // FF minimization always skips range clamping (uses its own torsion potential)
    if (!skip_range_clamp && !g_cpu_constants.USE_TORSION_PENALTY) {
        int i_lo = flex_topo->range_inds[i_tor * 2];
        int nrange = flex_topo->range_inds[i_tor * 2 + 1];
        dihe_new = clamp_by_ranges_cpu(dihe_new, flex_topo->range_list + i_lo, nrange);
    }
    
    // Update dihedral value
    dihe_incre = dihe_new - out_x->dihedrals[i_tor];
    out_x->dihedrals[i_tor] = dihe_new;
    
    // Get influenced atoms
    int begin = flex_topo->rotated_inds[i_tor * 2];
    int end = flex_topo->rotated_inds[i_tor * 2 + 1];
    
    // Define rotation axis
    int iat_from = flex_topo->axis_atoms[i_tor * 2];
    int iat_to = flex_topo->axis_atoms[i_tor * 2 + 1];
    
    Real coord_to[3] = {
        out_x->coords[iat_to * 3],
        out_x->coords[iat_to * 3 + 1],
        out_x->coords[iat_to * 3 + 2]
    };
    
    Real axis_unit[3] = {
        coord_to[0] - out_x->coords[iat_from * 3],
        coord_to[1] - out_x->coords[iat_from * 3 + 1],
        coord_to[2] - out_x->coords[iat_from * 3 + 2]
    };
    
    Real l = norm_cpu(axis_unit);
    if (l <= EPSILON_CPU) {
        axis_unit[0] = 1; axis_unit[1] = 0; axis_unit[2] = 0;
    } else {
        axis_unit[0] /= l; axis_unit[1] /= l; axis_unit[2] /= l;
    }
    
    // Rotate all influenced atoms
    for (int j = begin; j <= end; j++) {
        int i_at = flex_topo->rotated_atoms[j];
        
        // Move rotation origin to the axis
        Real tmp[3] = {
            out_x->coords[i_at * 3] - coord_to[0],
            out_x->coords[i_at * 3 + 1] - coord_to[1],
            out_x->coords[i_at * 3 + 2] - coord_to[2]
        };
        
        // Rotate by Rodrigues' formula
        rotate_vec_by_rodrigues(tmp, axis_unit, dihe_incre);
        
        // Move back
        out_x->coords[i_at * 3] = tmp[0] + coord_to[0];
        out_x->coords[i_at * 3 + 1] = tmp[1] + coord_to[1];
        out_x->coords[i_at * 3 + 2] = tmp[2] + coord_to[2];
    }
    // NOTE: Center update removed - done once in apply_grad_update_pose_cpu
}


void apply_grad_update_pose_cpu(FlexPose* out_x, const FlexPoseGradient* g,
                                 const FlexTopo& flex_topo, Real alpha,
                                 bool skip_range_clamp) {
    const auto& c = g_cpu_constants;
    
    // Torsion increment
    for (int i_tor = 0; i_tor < flex_topo.ntorsion; i_tor++) {
        apply_grad_update_dihe_cpu(out_x, &flex_topo, i_tor, g->dihedrals_g[i_tor] * alpha, skip_range_clamp);
    }
    
    // Update center ONCE after all torsion updates (was done per-torsion before)
    Real center_new[3] = {0, 0, 0};
    for (int j = 0; j < flex_topo.natom; j++) {
        center_new[0] += out_x->coords[j * 3];
        center_new[1] += out_x->coords[j * 3 + 1];
        center_new[2] += out_x->coords[j * 3 + 2];
    }
    out_x->center[0] = center_new[0] / flex_topo.natom;
    out_x->center[1] = center_new[1] / flex_topo.natom;
    out_x->center[2] = center_new[2] / flex_topo.natom;
    
    if (!c.FLAG_CONSTRAINT_DOCK) {
        // Save original center
        Real old_center[3] = {out_x->center[0], out_x->center[1], out_x->center[2]};
        
        // Update center
        out_x->center[0] = clamp_to_center_cpu(old_center[0] + alpha * g->center_g[0], c.BOX_X_HI, c.BOX_X_LO);
        out_x->center[1] = clamp_to_center_cpu(old_center[1] + alpha * g->center_g[1], c.BOX_Y_HI, c.BOX_Y_LO);
        out_x->center[2] = clamp_to_center_cpu(old_center[2] + alpha * g->center_g[2], c.BOX_Z_HI, c.BOX_Z_LO);
        
        // Update orientation
        Real rot_incre[3] = {
            g->orientation_g[0] * alpha,
            g->orientation_g[1] * alpha,
            g->orientation_g[2] * alpha
        };
        
        Real q[4];
        rotvec_to_quaternion(q, rot_incre);
        
        out_x->rot_vec[0] = rot_incre[0];
        out_x->rot_vec[1] = rot_incre[1];
        out_x->rot_vec[2] = rot_incre[2];
        
        // Apply rotation and translation to all atoms
        for (int i_at = 0; i_at < flex_topo.natom; i_at++) {
            // Locate rotation origin on original center
            Real tmp[3] = {
                out_x->coords[i_at * 3] - old_center[0],
                out_x->coords[i_at * 3 + 1] - old_center[1],
                out_x->coords[i_at * 3 + 2] - old_center[2]
            };
            
            rotate_vec_by_quaternion(tmp, q);
            
            // Move to new center
            out_x->coords[i_at * 3] = tmp[0] + out_x->center[0];
            out_x->coords[i_at * 3 + 1] = tmp[1] + out_x->center[1];
            out_x->coords[i_at * 3 + 2] = tmp[2] + out_x->center[2];
        }
    }
}


// ==================== Bonded-H topology ====================

void build_bonded_h_topo(BondedHTopo& topo, const Real* coords,
                         const int* vn_types, int natom) {
    const auto& atp = g_cpu_constants.atp;
    const Real max_bond_dist_sq = Real(1.3 * 1.3);

    topo.natom = natom;
    topo.h_start = new int[natom + 1]();

    // Pass 1: count bonded H per atom
    for (int i = 0; i < natom; i++) {
        if (atp.is_hydrogen[vn_types[i]]) continue; // H atoms don't have bonded H
        const Real ix = coords[i * 3];
        const Real iy = coords[i * 3 + 1];
        const Real iz = coords[i * 3 + 2];
        for (int h = 0; h < natom; h++) {
            if (!atp.is_hydrogen[vn_types[h]]) continue;
            Real dx = coords[h * 3]     - ix;
            Real dy = coords[h * 3 + 1] - iy;
            Real dz = coords[h * 3 + 2] - iz;
            if (dx * dx + dy * dy + dz * dz < max_bond_dist_sq) {
                topo.h_start[i + 1]++;
            }
        }
    }

    // Pass 2: prefix sum
    for (int i = 0; i < natom; i++) {
        topo.h_start[i + 1] += topo.h_start[i];
    }
    topo.total_h = topo.h_start[natom];

    // Pass 3: fill indices
    topo.h_indices = new int[std::max(topo.total_h, 1)]; // at least 1 to avoid nullptr
    int* cursor = new int[natom];
    std::memcpy(cursor, topo.h_start, natom * sizeof(int));

    for (int i = 0; i < natom; i++) {
        if (atp.is_hydrogen[vn_types[i]]) continue;
        const Real ix = coords[i * 3];
        const Real iy = coords[i * 3 + 1];
        const Real iz = coords[i * 3 + 2];
        for (int h = 0; h < natom; h++) {
            if (!atp.is_hydrogen[vn_types[h]]) continue;
            Real dx = coords[h * 3]     - ix;
            Real dy = coords[h * 3 + 1] - iy;
            Real dz = coords[h * 3 + 2] - iz;
            if (dx * dx + dy * dy + dz * dz < max_bond_dist_sq) {
                topo.h_indices[cursor[i]++] = h;
            }
        }
    }
    delete[] cursor;

    // Pass 4: heavy neighbour (antecedent) of every atom, nearest heavy atom within a bond length
    const Real max_heavy_bond_sq = Real(2.0 * 2.0);
    topo.antecedent = new int[natom];
    for (int i = 0; i < natom; i++) {
        topo.antecedent[i] = -1;
        if (atp.is_hydrogen[vn_types[i]]) continue;
        Real best = max_heavy_bond_sq;
        for (int j = 0; j < natom; j++) {
            if (j == i || atp.is_hydrogen[vn_types[j]]) continue;
            Real dx = coords[j * 3] - coords[i * 3];
            Real dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            Real dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            Real d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < best) { best = d2; topo.antecedent[i] = j; }
        }
    }

    // Pass 5: free-rotor donors (cone hydrogen) vs fixed donors (explicit hydrogens)
    topo.donor_rotatable = new int[natom];
    for (int i = 0; i < natom; i++) {
        const int hb = topo.h_start[i], he = topo.h_start[i + 1];
        const int n_h = he - hb;
        const Real* X = topo.antecedent[i] >= 0 ? coords + topo.antecedent[i] * 3 : nullptr;
        const Real* H1 = n_h >= 1 ? coords + topo.h_indices[hb] * 3 : nullptr;
        const Real* H2 = n_h >= 2 ? coords + topo.h_indices[hb + 1] * 3 : nullptr;
        topo.donor_rotatable[i] = hbond_donor_rotatable_geom(vn_types[i], n_h, coords + i * 3, X, H1, H2) ? 1 : 0;
    }

    // Pass 6: pairs of free-rotor donors on one antecedent (gem-diol C(OH)2, B(OH)2, P(OH)2). The GPU polar
    // pass adds their cone gradients to the shared atom from two lanes and keeps the g_pgrid walk for such
    // ligands so the accumulation order is unchanged (bfgs.cuh cal_e_f_warp cand_pass, maps.cu upload_polar_cands)
    topo.n_shared_cone_antecedent = 0;
    for (int i = 0; i < natom; i++) {
        if (!topo.donor_rotatable[i] || topo.antecedent[i] < 0) continue;
        for (int j = i + 1; j < natom; j++) {
            if (topo.donor_rotatable[j] && topo.antecedent[j] == topo.antecedent[i]) topo.n_shared_cone_antecedent++;
        }
    }
}

void free_bonded_h_topo(BondedHTopo& topo) {
    delete[] topo.h_start;
    delete[] topo.h_indices;
    delete[] topo.antecedent;
    delete[] topo.donor_rotatable;
    topo.h_start = nullptr;
    topo.h_indices = nullptr;
    topo.antecedent = nullptr;
    topo.donor_rotatable = nullptr;
    topo.natom = 0;
    topo.total_h = 0;
}

void free_rec_donor_h_topo(RecDonorHTopo& topo) {
    delete[] topo.h_start;
    delete[] topo.h_coords;
    delete[] topo.antecedent;
    delete[] topo.donor_rotatable;
    topo.h_start = nullptr;
    topo.h_coords = nullptr;
    topo.antecedent = nullptr;
    topo.donor_rotatable = nullptr;
    topo.natom = 0;
    topo.total_h = 0;
}


// ==================== Cone-donor hydrogen placement ====================

// Heavy atoms within 2.0 A of atom i, the distance test of build_bonded_h_topo pass 4
static std::vector<int> heavy_neighbours(const Real* coords, const int* vn_types, int natom, int i,
                                         const AtomTypeProps& atp) {
    const Real max_heavy_bond_sq = Real(2.0 * 2.0);
    std::vector<int> nb;
    for (int j = 0; j < natom; j++) {
        if (j == i || atp.is_hydrogen[vn_types[j]]) continue;
        Real dx = coords[j * 3] - coords[i * 3];
        Real dy = coords[j * 3 + 1] - coords[i * 3 + 1];
        Real dz = coords[j * 3 + 2] - coords[i * 3 + 2];
        if (dx * dx + dy * dy + dz * dz < max_heavy_bond_sq) nb.push_back(j);
    }
    return nb;
}

// sp2 test for the antecedent X of a hydroxyl: two neighbours, or three whose angles at X sum to at
// least 345 deg (the planarity limit of hbond_donor_rotatable_geom)
static bool is_planar_center(const Real* coords, int x, const std::vector<int>& nb) {
    if (nb.size() == 2) return true;
    if (nb.size() != 3) return false;
    Real v[3][3];
    for (int k = 0; k < 3; k++) {
        for (int c = 0; c < 3; c++) v[k][c] = coords[nb[k] * 3 + c] - coords[x * 3 + c];
        Real n = std::sqrt(v[k][0] * v[k][0] + v[k][1] * v[k][1] + v[k][2] * v[k][2]);
        if (n < Real(1e-6)) return false;
        for (int c = 0; c < 3; c++) v[k][c] /= n;
    }
    Real sum = 0;
    const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (const auto& p : pairs) {
        Real c = v[p[0]][0] * v[p[1]][0] + v[p[0]][1] * v[p[1]][1] + v[p[0]][2] * v[p[1]][2];
        sum += std::acos(std::max(Real(-1), std::min(Real(1), c)));
    }
    return sum >= Real(345.0 * 3.14159265358979 / 180.0);
}

// Angular weight of D-H...A at an explicit hydrogen: the consider() rule of angular_hbond_pair_weight
// (score/score.h), 0 for a degenerate geometry
static Real explicit_h_weight(const Real* H, const Real* D, const Real* A) {
    Real hd_x = D[0] - H[0], hd_y = D[1] - H[1], hd_z = D[2] - H[2];
    Real ha_x = A[0] - H[0], ha_y = A[1] - H[1], ha_z = A[2] - H[2];
    Real hd_sq = hd_x * hd_x + hd_y * hd_y + hd_z * hd_z;
    Real ha_sq = ha_x * ha_x + ha_y * ha_y + ha_z * ha_z;
    if (hd_sq < Real(1e-12) || ha_sq < Real(1e-12)) return 0;
    Real cos_dha = (hd_x * ha_x + hd_y * ha_y + hd_z * ha_z) / (std::sqrt(hd_sq) * std::sqrt(ha_sq));
    Real dw = 0;
    return angular_hbond_weight(cos_dha, &dw);
}

// Weight score() gives receptor donor a with its hydrogens toward ligand atom L: direction 2 of
// angular_hbond_pair_weight (score/score.h); 0 without a receptor H table, as there
static Real rec_donor_weight(const UDFixMol& fix, int a, const Real* L) {
    const RecDonorHTopo& rh = fix.rec_donor_h;
    if (rh.natom <= 0) return 0;
    const Real* R = fix.coords.data() + a * 3;
    const int rs = rh.h_start[a], re = rh.h_start[a + 1];
    const int n_h = re - rs;
    const bool rot = rh.donor_rotatable ? rh.donor_rotatable[a] != 0 : hbond_donor_rotatable(fix.vina_types[a], n_h);
    if (rot) {
        const int x = rh.antecedent ? rh.antecedent[a] : -1;
        if (x < 0) return 1;
        Real dw = 0;
        return angular_hbond_weight(hbond_cone_cos(R, fix.coords.data() + x * 3, L, nullptr, nullptr, nullptr), &dw);
    }
    if (n_h == 0) return 1;
    Real best = 0;
    for (int ih = rs; ih < re; ih++) best = std::max(best, explicit_h_weight(rh.h_coords + ih * 3, R, L));
    return best;
}

std::vector<PlaceHDonor> find_placeable_cone_donors(const UDFlexMol& mol, int n_write, const AtomTypeProps& atp) {
    const int natom = mol.natom;
    const Real* coords = mol.coords.data();
    const int* types = mol.vina_types.data();
    BondedHTopo h_topo = {};
    build_bonded_h_topo(h_topo, coords, types, natom);

    // Number of heavy-atom H lists holding each H: moving an H listed by another atom too would
    // move that atom's explicit hydrogen
    std::vector<int> n_lists(natom, 0);
    for (int k = 0; k < h_topo.total_h; k++) n_lists[h_topo.h_indices[k]]++;

    std::vector<PlaceHDonor> donors;
    for (int d = 0; d < natom; d++) {
        const int td = types[d];
        if (!h_topo.donor_rotatable[d] || !atp.is_donor[td]) continue;
        const int x = h_topo.antecedent[d];
        const int hb = h_topo.h_start[d], he = h_topo.h_start[d + 1];
        if (x < 0 || he - hb < 1 || d >= n_write) continue;
        if (heavy_neighbours(coords, types, natom, d, atp).size() != 1) continue;
        bool own_h = true;
        for (int k = hb; k < he; k++) {
            const int h = h_topo.h_indices[k];
            if (n_lists[h] != 1 || h >= n_write) own_h = false;
        }
        if (!own_h) continue;

        PlaceHDonor dn;
        dn.d = d;
        dn.x = x;
        dn.h.assign(h_topo.h_indices + hb, h_topo.h_indices + he);
        if (td == VN_TYPE_O_D || td == VN_TYPE_O_DA) {
            std::vector<int> nb_x = heavy_neighbours(coords, types, natom, x, atp);
            nb_x.insert(nb_x.end(), h_topo.h_indices + h_topo.h_start[x], h_topo.h_indices + h_topo.h_start[x + 1]);
            if (is_planar_center(coords, x, nb_x)) {
                // phenol: in-plane flip only; acid, enol, oxime, ... O-H keep their prepared hydrogen
                const bool aromatic_x = mol.aromatic.size() == static_cast<size_t>(natom) && mol.aromatic[x] != 0;
                if (!aromatic_x) continue;
                dn.flip_only = true;
            }
        }
        donors.push_back(std::move(dn));
    }
    free_bonded_h_topo(h_topo);
    return donors;
}

int place_cone_donor_h(Real* coords, int natom, const std::vector<PlaceHDonor>& donors, const UDFixMol& fix,
                       const std::vector<int>& rec_acceptors, const Real* radii, const AtomTypeProps& atp,
                       const int* lig_types, int& n_h_moved) {
    const Real min_hh_sq = Real(2.0 * 2.0);
    Vina vina;
    int n_moved = 0;
    n_h_moved = 0;
    std::vector<int> lig_h;
    for (int a = 0; a < natom; a++) {
        if (atp.is_hydrogen[lig_types[a]]) lig_h.push_back(a);
    }
    std::vector<int> acc;
    std::vector<Real> ramp;
    std::vector<Real> h_try, h_best;
    std::vector<Real> thetas;
    for (const auto& dn : donors) {
        const Real* D = coords + dn.d * 3;
        const Real* X = coords + dn.x * 3;
        const int td = lig_types[dn.d];
        const int n_h = static_cast<int>(dn.h.size());

        // Acceptor set: receptor acceptors where score() applies the angular rule to D
        acc.clear();
        ramp.clear();
        for (int a : rec_acceptors) {
            const int ta = fix.vina_types[a];
            const Real* A = fix.coords.data() + a * 3;
            Real r_vec[3] = {A[0] - D[0], A[1] - D[1], A[2] - D[2]};
            Real rr = r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2];
            Real f = 0;
            const Real ramp_a = vina.vina_hbond(std::sqrt(rr) - (radii[td] + radii[ta]), &f);
            if (!(ramp_a > 0)) continue;
            if (atp.is_acceptor[td] && atp.is_donor[ta] && rec_donor_weight(fix, a, D) >= Real(1)) continue;
            acc.push_back(a);
            ramp.push_back(ramp_a);
        }
        if (acc.empty()) continue;

        Real u[3] = {D[0] - X[0], D[1] - X[1], D[2] - X[2]};
        const Real nu = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        if (nu < Real(1e-6)) continue;
        for (int c = 0; c < 3; c++) u[c] /= nu;

        auto objective = [&](const Real* hc) {
            Real f = 0;
            for (size_t k = 0; k < acc.size(); k++) {
                const Real* A = fix.coords.data() + acc[k] * 3;
                Real best = 0;
                for (int ih = 0; ih < n_h; ih++) best = std::max(best, explicit_h_weight(hc + ih * 3, D, A));
                f += ramp[k] * best;
            }
            return f;
        };
        // component of P - D perpendicular to the axis
        auto perp = [&](const Real* P, Real* out) {
            Real v[3] = {P[0] - D[0], P[1] - D[1], P[2] - D[2]};
            Real proj = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
            for (int c = 0; c < 3; c++) out[c] = v[c] - proj * u[c];
            return std::sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2]);
        };
        // a turned H closer than 2.0 A to a ligand H of another atom, at the current coordinates
        auto clashes = [&](const Real* hc) {
            for (int o : lig_h) {
                if (std::find(dn.h.begin(), dn.h.end(), o) != dn.h.end()) continue;
                const Real* P = coords + o * 3;
                for (int ih = 0; ih < n_h; ih++) {
                    Real dx = hc[ih * 3] - P[0], dy = hc[ih * 3 + 1] - P[1], dz = hc[ih * 3 + 2] - P[2];
                    if (dx * dx + dy * dy + dz * dz < min_hh_sq) return true;
                }
            }
            return false;
        };

        thetas.clear();
        if (dn.flip_only) {
            thetas.push_back(PI);
        } else {
            for (int a : acc) {
                Real cone_h[3], b[3];
                hbond_cone_h(D, X, fix.coords.data() + a * 3, cone_h);
                if (perp(cone_h, b) < Real(1e-6)) continue;
                for (int h : dn.h) {
                    Real p[3];
                    if (perp(coords + h * 3, p) < Real(1e-6)) continue;
                    Real pxb[3];
                    cross_product(p, b, pxb);
                    thetas.push_back(std::atan2(u[0] * pxb[0] + u[1] * pxb[1] + u[2] * pxb[2],
                                                p[0] * b[0] + p[1] * b[1] + p[2] * b[2]));
                }
            }
            // several acceptors: the optimum can lie between the cone directions
            if (acc.size() >= 2) {
                for (int k = 1; k < 360; k++) thetas.push_back(Real(k) * PI / Real(180));
            }
        }

        h_best.clear();
        for (int h : dn.h) h_best.insert(h_best.end(), coords + h * 3, coords + h * 3 + 3);
        Real f_best = objective(h_best.data());
        bool moved = false;
        h_try.resize(n_h * 3);
        for (Real theta : thetas) {
            for (int ih = 0; ih < n_h; ih++) {
                const Real* H = coords + dn.h[ih] * 3;
                Real v[3] = {H[0] - D[0], H[1] - D[1], H[2] - D[2]};
                rotate_vec_by_rodrigues(v, u, theta);
                for (int c = 0; c < 3; c++) h_try[ih * 3 + c] = D[c] + v[c];
            }
            const Real f = objective(h_try.data());
            if (!(f > f_best + Real(1e-6)) || clashes(h_try.data())) continue;
            f_best = f;
            h_best = h_try;
            moved = true;
        }
        if (!moved) continue;
        for (int ih = 0; ih < n_h; ih++) {
            for (int c = 0; c < 3; c++) coords[dn.h[ih] * 3 + c] = h_best[ih * 3 + c];
        }
        n_moved++;
        n_h_moved += n_h;
    }
    return n_moved;
}


// ==================== Hbond angle correction ====================

Real hbond_angle_correction_cpu(const FlexPose* pose, const FlexTopo& flex_topo,
                                 const FixMol& fix_mol) {
    const auto& atp = g_cpu_constants.atp;
    const SpatialGrid& grid = g_cpu_constants.receptor_grid;
    if (!grid.built) return 0;

    const Real* lig_coords = pose->coords;
    const Real* rec_coords = fix_mol.coords;
    const int natom = flex_topo.natom;
    const int* lig_types = flex_topo.vn_types;

    const Real max_bond_dist_sq = Real(1.3 * 1.3); // bonded-H detection radius
    const Real min_dha_cos = Real(-0.17365);        // cos(100°)
    const Real max_ha_dist_sq = Real(3.0 * 3.0);   // H...A distance cutoff
    g_vina_scorer.set_weights(g_cpu_constants.weights);
    const Real r2_cutoff = g_vina_scorer.r2_cutoff;
    const int ny_nz = grid.ny * grid.nz;

    Real correction = 0;

    for (int i_lig = 0; i_lig < natom; i_lig++) {
        const int lig_type = lig_types[i_lig];
        if (!atp.is_donor[lig_type]) continue;

        const Real dx0 = lig_coords[i_lig * 3];
        const Real dy0 = lig_coords[i_lig * 3 + 1];
        const Real dz0 = lig_coords[i_lig * 3 + 2];

        // Collect bonded H atoms on the ligand (distance-based, max 8)
        int bonded_h[8];
        int n_h = 0;
        for (int h = 0; h < natom && n_h < 8; h++) {
            if (!atp.is_hydrogen[lig_types[h]]) continue;
            Real bx = lig_coords[h * 3]     - dx0;
            Real by = lig_coords[h * 3 + 1] - dy0;
            Real bz = lig_coords[h * 3 + 2] - dz0;
            if (bx * bx + by * by + bz * bz < max_bond_dist_sq) {
                bonded_h[n_h++] = h;
            }
        }
        if (n_h == 0) continue; // no bonded H → can't check angle

        const Real r1 = g_cpu_constants.vdw_radii[lig_type];

        // Grid neighbor lookup
        int cx = (int)((dx0 - grid.origin[0]) * grid.inv_cell_size);
        int cy = (int)((dy0 - grid.origin[1]) * grid.inv_cell_size);
        int cz = (int)((dz0 - grid.origin[2]) * grid.inv_cell_size);
        cx = std::clamp(cx, 0, grid.nx - 1);
        cy = std::clamp(cy, 0, grid.ny - 1);
        cz = std::clamp(cz, 0, grid.nz - 1);

        const int cx_lo = std::max(0, cx - grid.half), cx_hi = std::min(grid.nx - 1, cx + grid.half);
        const int cy_lo = std::max(0, cy - grid.half), cy_hi = std::min(grid.ny - 1, cy + grid.half);
        const int cz_lo = std::max(0, cz - grid.half), cz_hi = std::min(grid.nz - 1, cz + grid.half);

        for (int ix = cx_lo; ix <= cx_hi; ix++) {
            for (int iy = cy_lo; iy <= cy_hi; iy++) {
                for (int iz = cz_lo; iz <= cz_hi; iz++) {
                    const int cell = ix * ny_nz + iy * grid.nz + iz;
                    const int start = grid.cell_start[cell];
                    const int end   = grid.cell_start[cell + 1];

                    for (int k = start; k < end; k++) {
                        const GridAtom& rec = grid.atoms[k];
                        // Only receptor acceptors form hbond with ligand donors
                        if (!atp.is_acceptor[rec.type]) continue;

                        const int i2_3 = rec.index * 3;
                        const Real ax = rec_coords[i2_3];
                        const Real ay = rec_coords[i2_3 + 1];
                        const Real az = rec_coords[i2_3 + 2];

                        Real rx = ax - dx0;
                        Real ry = ay - dy0;
                        Real rz = az - dz0;
                        Real rr = rx * rx + ry * ry + rz * rz;
                        if (rr >= r2_cutoff) continue;

                        Real r = std::sqrt(rr);
                        Real d = r - r1 - rec.radius;

                        // Check what hbond energy eval_ef awarded for this pair
                        VinaTerms terms = g_vina_scorer.eval_ef_decomp(
                            d, lig_type, rec.type, atp);
                        if (terms.hbond == Real(0)) continue;

                        // If both directions are possible (lig=DA, rec=DA), skip:
                        // we can't verify the reverse direction without receptor H
                        if (atp.is_acceptor[lig_type] && atp.is_donor[rec.type])
                            continue;

                        // Check D-H...A angle using each bonded H
                        bool good_angle = false;
                        for (int ih = 0; ih < n_h && !good_angle; ih++) {
                            const int h3 = bonded_h[ih] * 3;
                            const Real hx = lig_coords[h3];
                            const Real hy = lig_coords[h3 + 1];
                            const Real hz = lig_coords[h3 + 2];

                            // H → A vector
                            Real ha_x = ax - hx, ha_y = ay - hy, ha_z = az - hz;
                            Real ha_sq = ha_x * ha_x + ha_y * ha_y + ha_z * ha_z;
                            if (ha_sq >= max_ha_dist_sq) continue;
                            Real ha_dist = std::sqrt(ha_sq);

                            // H → D vector
                            Real hd_x = dx0 - hx, hd_y = dy0 - hy, hd_z = dz0 - hz;
                            Real hd_dist = std::sqrt(
                                hd_x * hd_x + hd_y * hd_y + hd_z * hd_z);

                            if (ha_dist < Real(1e-6) || hd_dist < Real(1e-6))
                                continue;

                            // cos(D-H...A) = dot(H→D, H→A) / (|H→D| |H→A|)
                            Real cos_angle =
                                (hd_x * ha_x + hd_y * ha_y + hd_z * ha_z)
                                / (hd_dist * ha_dist);

                            // Good geometry: angle > 100° → cos < cos(100°)
                            if (cos_angle < min_dha_cos) {
                                good_angle = true;
                            }
                        }

                        if (!good_angle) {
                            // Remove the favorable hbond energy (terms.hbond < 0)
                            correction -= terms.hbond;
                        }
                    }
                }
            }
        }
    }

    return correction;
}
