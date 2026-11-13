//
// Created by Congcong Liu on 24-12-26.
//

#ifndef SCORE_H
#define SCORE_H
#include "myutils/common.h"
#include "score/rescore_hbond.h"
#include <algorithm>
#include <cmath>


Real cal_box_penalty_atom(Real x, Real y, Real z, const Box& box){
    Real penalty_slope = 1e6;
    Real penalty = 0.;
    if (x < box.x_lo){
        penalty += box.x_lo - x;
    }
    else if (x > box.x_hi){
        penalty += x - box.x_hi;
    }

    if (y < box.y_lo){
        penalty += box.y_lo - y;
    }
    else if (y > box.y_hi){
        penalty += y - box.y_hi;
    }

    if (z < box.z_lo){
        penalty += box.z_lo - z;
    }
    else if (z > box.z_hi){
        penalty += z - box.z_hi;
    }

    return penalty * penalty_slope;
}


// Thread-local (as score_cpu.cpp's g_vina_scorer): DockTask::run_score re-weights it per ligand from
// parallel threads, each thread seeing only its own copy. Constant-initialised, so no TLS wrapper.
thread_local Vina SF;
Vina SF_INTRA;  // standard Vina weights; the intra term is only scaled, never re-weighted

// Apolar-environment credit: logistic neighbour counter centred at POLAR_ENV_RADIUS with width
// POLAR_ENV_WIDTH, saturated per atom through 1 - exp(-n / POLAR_ENV_CAP).
constexpr Real POLAR_ENV_RADIUS = 5.0;
constexpr Real POLAR_ENV_WIDTH = 0.5;
constexpr Real POLAR_ENV_CAP = 6.0;

/**
 * @brief Best angular HBond weight for ligand atom i1 against receptor atom i2, the same
 * rule the search energy applies (cal_e_f_cpu / cal_e_f_warp): direction 1, ligand donor
 * with its bonded H toward a receptor acceptor; direction 2, receptor donor with its H
 * table toward a ligand acceptor; the better of the two. Returns 0 when no hydrogen
 * geometry is available, which removes the HBond term like the search does.
 */
inline Real angular_hbond_pair_weight(int i1, int i2, const Real* flex_coords,
                                      const UDFixMol& udfix_mol, const UDFlexMol& udflex_mol,
                                      const AtomTypeProps& atp,
                                      const BondedHTopo* h_topo, const RecDonorHTopo* rh_topo){
    const int t1 = udflex_mol.vina_types[i1], t2 = udfix_mol.vina_types[i2];
    const Real* L = flex_coords + i1 * 3;
    const Real* R = udfix_mol.coords.data() + i2 * 3;
    Real best_w = 0;
    auto consider = [&](Real hx, Real hy, Real hz, const Real* D, const Real* A){
        Real hd_x = D[0] - hx, hd_y = D[1] - hy, hd_z = D[2] - hz;
        Real ha_x = A[0] - hx, ha_y = A[1] - hy, ha_z = A[2] - hz;
        Real hd_sq = hd_x * hd_x + hd_y * hd_y + hd_z * hd_z;
        Real ha_sq = ha_x * ha_x + ha_y * ha_y + ha_z * ha_z;
        if (hd_sq < Real(1e-12) || ha_sq < Real(1e-12)) return;
        Real cos_dha = (hd_x * ha_x + hd_y * ha_y + hd_z * ha_z) / (std::sqrt(hd_sq) * std::sqrt(ha_sq));
        Real dw = 0;
        best_w = std::max(best_w, angular_hbond_weight(cos_dha, &dw));
    };
    // Rotatable donors (-OH, -NH2, -NH3+): score the best hydrogen on the bond cone
    // instead of the prep-placed one. Donors with no hydrogen and no heavy neighbour
    // (water O, metals) carry no directional information and keep full weight.
    if (atp.is_donor[t1] && atp.is_acceptor[t2] && h_topo != nullptr && h_topo->h_start != nullptr){
        const int hb = h_topo->h_start[i1], he = h_topo->h_start[i1 + 1];
        const int n_h = he - hb;
        const bool rot = h_topo->donor_rotatable ? h_topo->donor_rotatable[i1] != 0 : hbond_donor_rotatable(t1, n_h);
        if (rot){
            const int x = h_topo->antecedent ? h_topo->antecedent[i1] : -1;
            if (x >= 0){
                Real dw = 0;
                best_w = std::max(best_w, angular_hbond_weight(hbond_cone_cos(L, flex_coords + x * 3, R, nullptr, nullptr, nullptr), &dw));
            } else {
                best_w = Real(1);
            }
        } else if (n_h == 0){
            best_w = Real(1);
        } else {
            for (int ih = hb; ih < he; ih++){
                const Real* H = flex_coords + h_topo->h_indices[ih] * 3;
                consider(H[0], H[1], H[2], L, R);
            }
        }
    }
    if (atp.is_acceptor[t1] && atp.is_donor[t2] && rh_topo != nullptr && rh_topo->natom > 0){
        const int rs = rh_topo->h_start[i2], re = rh_topo->h_start[i2 + 1];
        const int n_h = re - rs;
        const bool rot = rh_topo->donor_rotatable ? rh_topo->donor_rotatable[i2] != 0 : hbond_donor_rotatable(t2, n_h);
        if (rot){
            const int x = rh_topo->antecedent ? rh_topo->antecedent[i2] : -1;
            if (x >= 0){
                Real dw = 0;
                best_w = std::max(best_w, angular_hbond_weight(hbond_cone_cos(R, udfix_mol.coords.data() + x * 3, L, nullptr, nullptr, nullptr), &dw));
            } else {
                best_w = Real(1);
            }
        } else if (n_h == 0){
            best_w = Real(1);
        } else {
            for (int ih = rs; ih < re; ih++){
                const Real* H = rh_topo->h_coords + ih * 3;
                consider(H[0], H[1], H[2], R, L);
            }
        }
    }
    return best_w;
}



/**
 * @brief Ligand torsional strain: sum over the preparation-time quadruples of a penalty in [0, 1].
 * kind 0 penalises twisting a conjugated single bond out of plane beyond `tolerance` degrees,
 * kind 1 penalises an sp3-sp3 bond approaching an eclipse, kind 2 the cis amide / E ester.
 */
inline Real torsion_strain_energy(const std::vector<TorsionPenaltyRow>& rows, const Real* coords){
    Real total = 0;
    for (const auto& r : rows){
        const Real* pi = coords + r.atoms[0] * 3;
        const Real* pj = coords + r.atoms[1] * 3;
        const Real* pk = coords + r.atoms[2] * 3;
        const Real* pl = coords + r.atoms[3] * 3;
        Real b0[3], b1[3], b2[3];
        for (int c = 0; c < 3; c++){ b0[c] = pi[c] - pj[c]; b1[c] = pk[c] - pj[c]; b2[c] = pl[c] - pk[c]; }
        Real n1 = std::sqrt(b1[0]*b1[0] + b1[1]*b1[1] + b1[2]*b1[2]);
        if (n1 < Real(1e-8)) continue;
        for (int c = 0; c < 3; c++) b1[c] /= n1;
        Real d0 = b0[0]*b1[0] + b0[1]*b1[1] + b0[2]*b1[2];
        Real d2 = b2[0]*b1[0] + b2[1]*b1[1] + b2[2]*b1[2];
        Real v[3], w[3];
        for (int c = 0; c < 3; c++){ v[c] = b0[c] - d0 * b1[c]; w[c] = b2[c] - d2 * b1[c]; }
        Real x = v[0]*w[0] + v[1]*w[1] + v[2]*w[2];
        Real cross[3] = {b1[1]*v[2] - b1[2]*v[1], b1[2]*v[0] - b1[0]*v[2], b1[0]*v[1] - b1[1]*v[0]};
        Real y = cross[0]*w[0] + cross[1]*w[1] + cross[2]*w[2];
        Real phi = std::atan2(y, x);
        Real pen = 0;
        if (r.kind == 0){
            Real dev = std::asin(std::min(std::abs(std::sin(phi)), Real(1))) * Real(180.0 / M_PI);
            pen = (dev - r.tolerance) / (Real(90) - r.tolerance);
        } else if (r.kind == 1){
            Real c3 = std::cos(3 * phi);
            c3 = std::max(Real(-1), std::min(Real(1), c3));
            Real dev = std::acos(c3) * Real(180.0 / M_PI) / Real(3);
            pen = (Real(60) - r.tolerance - dev) / (Real(60) - r.tolerance);
        } else {
            pen = (Real(1) - std::cos(phi)) / Real(2);
        }
        total += r.weight * std::max(Real(0), std::min(Real(1), pen));
    }
    return total;
}


/**
 * @brief Rescore one pose (intra, inter, box penalty into out_pose->center[0..2]).
 * With angular_hbond the inter HBond term carries the same angular weight the search
 * used, so the reported score ranks poses on the landscape they were optimized on.
 */
void score(FlexPose* out_pose, const Real* flex_coords, const UDFixMol& udfix_mol, const UDFlexMol& udflex_mol, const Box& box, const AtomTypeProps& atp,
           bool angular_hbond = false, const BondedHTopo* h_topo = nullptr, const RecDonorHTopo* rh_topo = nullptr,
           ScoreDiag* out_diag = nullptr){
    Real rr = 0;
    Real f = 0;
    Real e_intra = 0., e_inter = 0., e_penalty = 0.;
    Vina sf = SF;  // local copy of the thread-local scorer (56 B): stack-relative loads in the pair loops


    // 1. Compute Pairwise energy and forces
    // -- Compute intra-molecular energy
    for (int i = 0; i < udflex_mol.intra_pairs.size() / 2; i ++){
        int i1 = udflex_mol.intra_pairs[i * 2], i2 = udflex_mol.intra_pairs[i * 2 + 1];

        // Cartesian distances won't be saved
        Real r_vec[3] = {
            // cuda vector multiply v3 v4
            flex_coords[i2 * 3] - flex_coords[i1 * 3],
            flex_coords[i2 * 3 + 1] - flex_coords[i1 * 3 + 1],
            flex_coords[i2 * 3 + 2] - flex_coords[i1 * 3 + 2]
        };
        rr = r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2];
        // DPrintCPU("Pair: %d %d r2 = %f", i1, i2, rr);
        if (rr < sf.r2_cutoff){
            rr = sqrt(rr); // use r2 as a container for |r|
            // if ((i1 == 17) and i2 == 20){
            //     int mmm = 1;
            //     for (int n = 200; n < 500; n++) {
            //         float r = n * 0.01;
            //         float tmp_f=0;
            //         float res = sf.eval_ef(r - udflex_mol.r1_plus_r2_intra[i], udflex_mol.vina_types[i1],
            //                         udflex_mol.vina_types[i2], &tmp_f);
            //         // printf("r=%f, e=%f\n", r, res);
            //     }
            // }
            // The intramolecular energy is evaluated with the standard Vina term weights and only
            // scaled by `intra`, so that changing the intermolecular weights (or using a separate
            // rescoring weight set) does not silently re-parameterise the ligand strain term.
            // The search kernels do NOT follow this rule: cal_e_f_cpu / cal_e_f_warp weight the intra
            // pairs with the search set (Score) times `intra`. Aligning them to this rule was measured
            // accuracy-neutral (1,454 systems x 2 seeds, same binary) and the kernel
            // behaviour was kept, so the search energy and this score differ on the intra term.
            Real tmp = SF_INTRA.eval_ef(rr - udflex_mol.r1_plus_r2_intra[i], udflex_mol.vina_types[i1],
                                        udflex_mol.vina_types[i2], &f, atp);
            e_intra += sf.weight_intra_val() * tmp;
            // DPrintCPU(" e = %f", tmp);
        }
        // DPrintCPU("\n", 1);
    }
    // DPrintCPU("\n", 1);

    out_pose->center[0] = e_intra;
    if (!udflex_mol.torsion_penalties.empty()
        && (sf.weight_torsion_strain_val() != Real(0) || out_diag != nullptr)){
        const Real ts = torsion_strain_energy(udflex_mol.torsion_penalties, flex_coords);
        if (out_diag) out_diag->torsion_strain = ts;
        out_pose->center[0] += sf.weight_torsion_strain_val() * ts;
    }

    // -- Compute inter-molecular energy: flex-protein
    int atom_id = udflex_mol.inter_pairs[0]; // begin from
    Real atom_e = 0;
    // Apolar-environment credit: soft count of receptor hydrophobic carbons around every weakly
    // polar ligand atom, saturated per atom. Accumulated in the pair loop, reduced afterwards.
    const Real w_polar_env = sf.weight_polar_env_val();
    const bool have_polar_env = udflex_mol.polar_env_class.size() == (size_t)udflex_mol.natom;
    const bool use_polar_env = (w_polar_env != Real(0) || out_diag != nullptr) && have_polar_env;
    std::vector<Real> polar_env_count(use_polar_env ? udflex_mol.natom : 0, Real(0));
    std::vector<Real> polar_env_hb(use_polar_env ? udflex_mol.natom : 0, Real(0));
    // Diagnostics over EVERY heavy atom, not just the class ones, so terms defined on the
    // complement (the strong donors and donor-acceptors) can be fitted from the output.
    const bool diag = out_diag != nullptr;
    std::vector<Real> dg_apolar(diag ? udflex_mol.natom : 0, Real(0));
    std::vector<Real> dg_polar(diag ? udflex_mol.natom : 0, Real(0));
    std::vector<Real> dg_hbond(diag ? udflex_mol.natom : 0, Real(0));
    // Extra weight for ligand-aromatic x receptor-ring-carbon contacts, on the hydrophobic ramp.
    const Real w_arom = sf.weight_arom_contact_val();
    const bool have_arom = udflex_mol.aromatic.size() == (size_t)udflex_mol.natom
        && udfix_mol.arom_ring_c.size() == (size_t)udfix_mol.natom;
    const bool use_arom = (w_arom != Real(0) || out_diag != nullptr) && have_arom;
    Real arom_raw = 0;
    const bool have_elec = udflex_mol.charges_ua.size() == (size_t)udflex_mol.natom
        && udflex_mol.charged_group.size() == (size_t)udflex_mol.natom
        && udfix_mol.charges_ua.size() == (size_t)udfix_mol.natom;
    const bool use_elec = sf.weight_elec_val() != Real(0) && have_elec;
    Real elec_neu = 0, elec_chg = 0;
    for (int i = 0; i < udflex_mol.inter_pairs.size() / 2; i ++){
        int i1 = udflex_mol.inter_pairs[i * 2], i2 = udflex_mol.inter_pairs[i * 2 + 1];
        if (i1 > atom_id){
            // DPrintCPU("Atom %d e_inter = %f\n", atom_id, atom_e);
            atom_e = 0;
            atom_id ++;
        }
        // Cartesian distances won't be saved
        Real r_vec[3] = {
            udfix_mol.coords[i2 * 3] - flex_coords[i1 * 3],
            udfix_mol.coords[i2 * 3 + 1] - flex_coords[i1 * 3 + 1],
            udfix_mol.coords[i2 * 3 + 2] - flex_coords[i1 * 3 + 2]
        };
        rr = r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2];

        if (rr < sf.r2_cutoff){
            rr = sqrt(rr); // use r2 as a container for |r|
            const Real d = rr - udflex_mol.r1_plus_r2_inter[i];
            const int t1 = udflex_mol.vina_types[i1], t2 = udfix_mol.vina_types[i2];
            Real e = sf.eval_ef(d, t1, t2, &f, atp);
            if (use_elec){
                Real q1 = udflex_mol.charges_ua[i1];
                if (udflex_mol.charged_group[i1]) q1 *= sf.elec_charged_scale_val();
                Real f_el = 0;
                e += sf.weight_elec_val() * sf.elec_pair(q1 * udfix_mol.charges_ua[i2], rr, &f_el);
            }
            if (diag && have_elec){
                Real f_el = 0;
                const Real ep = sf.elec_pair(udflex_mol.charges_ua[i1] * udfix_mol.charges_ua[i2], rr, &f_el);
                if (udflex_mol.charged_group[i1]) elec_chg += ep; else elec_neu += ep;
            }
            if (use_polar_env && udflex_mol.polar_env_class[i1] && t2 == VN_TYPE_C_H){
                polar_env_count[i1] += Real(1) / (Real(1) + std::exp((rr - POLAR_ENV_RADIUS) / POLAR_ENV_WIDTH));
            }
            if (diag){
                const Real sw = Real(1) / (Real(1) + std::exp((rr - POLAR_ENV_RADIUS) / POLAR_ENV_WIDTH));
                if (t2 == VN_TYPE_C_H) dg_apolar[i1] += sw;
                if (atp.is_donor[t2] || atp.is_acceptor[t2]) dg_polar[i1] += sw;
            }
            if (use_arom && udflex_mol.aromatic[i1] && udfix_mol.arom_ring_c[i2]
                && atp.is_hydrophobic[t1] && atp.is_hydrophobic[t2]){
                Real f_ar = 0;
                const Real e_ar = sf.vina_hydrophobic(d, &f_ar);
                arom_raw += e_ar;
                e += w_arom * e_ar;
            }
            if (angular_hbond && (atp.is_donor[t1] || atp.is_acceptor[t1]) && xs_h_bond_possible(t1, t2, atp)){
                Real f_hb = 0;
                Real e_hb = sf.weight_hbond_val() * sf.vina_hbond(d, &f_hb);
                if (std::abs(e_hb) > Real(1e-12)){
                    const Real aw = angular_hbond_pair_weight(i1, i2, flex_coords, udfix_mol, udflex_mol, atp, h_topo, rh_topo);
                    e += (aw - Real(1)) * e_hb;
                    if (use_polar_env) polar_env_hb[i1] += std::abs(aw * e_hb);
                    if (diag) dg_hbond[i1] += std::abs(aw * e_hb);
                }
            } else if ((use_polar_env || diag) && xs_h_bond_possible(t1, t2, atp)){
                Real f_hb = 0;
                const Real mag = std::abs(sf.weight_hbond_val() * sf.vina_hbond(d, &f_hb));
                if (use_polar_env) polar_env_hb[i1] += mag;
                if (diag) dg_hbond[i1] += mag;
            }
            // printf("Atom %d %d e=%f\n", i1, i2, e);
            e_inter += e;
            atom_e += e;
        }
    }
    // DPrintCPU("Atom %d e_inter = %f\n", atom_id, atom_e);

    if (use_polar_env){
        Real credit = 0;
        for (int i = 0; i < udflex_mol.natom; i++){
            if (udflex_mol.polar_env_class[i]){
                credit += Real(1) - std::exp(-polar_env_count[i] / POLAR_ENV_CAP);
                if (out_diag){
                    out_diag->n_class += 1;
                    out_diag->burial += polar_env_count[i];
                    out_diag->hbond_sat += polar_env_hb[i];
                }
            }
        }
        if (out_diag) out_diag->polar_env = credit;
        e_inter += w_polar_env * credit;
    }
    if (out_diag){
        out_diag->arom_contact = arom_raw;
        out_diag->elec_neutral = elec_neu;
        out_diag->elec_charged = elec_chg;
        const bool have_class = udflex_mol.polar_env_class.size() == (size_t)udflex_mol.natom;
        for (int i = 0; i < udflex_mol.natom; i++){
            if (atp.is_hydrogen[udflex_mol.vina_types[i]]) continue;
            out_diag->n_heavy += 1;
            out_diag->heavy_atom.push_back(i);
            out_diag->heavy_type.push_back(static_cast<Real>(udflex_mol.vina_types[i]));
            out_diag->heavy_class.push_back(have_class && udflex_mol.polar_env_class[i] ? Real(1) : Real(0));
            out_diag->heavy_apolar.push_back(dg_apolar[i]);
            out_diag->heavy_polar.push_back(dg_polar[i]);
            out_diag->heavy_hbond.push_back(dg_hbond[i]);
        }
    }
    out_pose->center[1] = e_inter;


    // -- Compute inter-molecular energy: penalty
    for (int i = 0; i < udflex_mol.natom; i ++){
        if (atp.is_hydrogen[udflex_mol.vina_types[i]]){
            continue;
        }
        Real atom_penalty = cal_box_penalty_atom(flex_coords[i * 3], flex_coords[i * 3 + 1], flex_coords[i * 3 + 2], box);
        if (atom_penalty > 0){
            // DPrintCPU("Atom %d penalty = %f\n", i, atom_penalty);
            cal_box_penalty_atom(flex_coords[i * 3], flex_coords[i * 3 + 1], flex_coords[i * 3 + 2], box);
        }
        e_penalty += atom_penalty;
    }
    out_pose->center[2] = e_penalty;
}


/**
 * @brief Per-atom and per-residue decomposition of the inter energy score() puts in center[1].
 *
 * Mirrors the inter loop of score() with out_diag == nullptr, expression for expression, on the
 * weight set held by `sf`: the 5 Vina terms, the screened Coulomb term, the aromatic-contact term,
 * the per-pair angular HBond weight (the same h_topo / rh_topo) and the saturated apolar-environment
 * credit. Pairwise terms go to exactly one ligand atom and one receptor residue; the polar_env
 * credit is exact per atom and split over residues in proportion to the atom's raw soft counts.
 * H atoms have no inter pairs and stay zero.
 *
 * @param sf            weight set of the pass being decomposed (a copy taken when it was installed)
 * @param out_atoms     one VinaTerms per ligand atom
 * @param out_residues  one VinaTerms per receptor residue (nullptr: skipped)
 * @param out_inter     running sum in score()'s order; equals that pass's center[1] bit for bit
 */
void score_decomp(Vina& sf, const Real* flex_coords, const UDFixMol& udfix_mol, const UDFlexMol& udflex_mol,
                  const AtomTypeProps& atp, bool angular_hbond, const BondedHTopo* h_topo,
                  const RecDonorHTopo* rh_topo, std::vector<VinaTerms>& out_atoms,
                  std::vector<VinaTerms>* out_residues = nullptr, Real* out_inter = nullptr){
    out_atoms.assign(udflex_mol.natom, VinaTerms{});
    if (out_residues) out_residues->assign(udfix_mol.residues.size(), VinaTerms{});

    // Same gates as score() without diagnostics
    const Real w_polar_env = sf.weight_polar_env_val();
    const bool use_polar_env = w_polar_env != Real(0)
        && udflex_mol.polar_env_class.size() == (size_t)udflex_mol.natom;
    const Real w_arom = sf.weight_arom_contact_val();
    const bool use_arom = w_arom != Real(0)
        && udflex_mol.aromatic.size() == (size_t)udflex_mol.natom
        && udfix_mol.arom_ring_c.size() == (size_t)udfix_mol.natom;
    const bool use_elec = sf.weight_elec_val() != Real(0)
        && udflex_mol.charges_ua.size() == (size_t)udflex_mol.natom
        && udflex_mol.charged_group.size() == (size_t)udflex_mol.natom
        && udfix_mol.charges_ua.size() == (size_t)udfix_mol.natom;
    std::vector<Real> polar_env_count(use_polar_env ? udflex_mol.natom : 0, Real(0));
    std::vector<std::pair<int, Real>> polar_env_pairs;  // (pair index, soft count), for the residue split

    Real e_inter = 0;
    for (int i = 0; i < udflex_mol.inter_pairs.size() / 2; i++){
        const int i1 = udflex_mol.inter_pairs[i * 2], i2 = udflex_mol.inter_pairs[i * 2 + 1];
        Real r_vec[3] = {
            udfix_mol.coords[i2 * 3] - flex_coords[i1 * 3],
            udfix_mol.coords[i2 * 3 + 1] - flex_coords[i1 * 3 + 1],
            udfix_mol.coords[i2 * 3 + 2] - flex_coords[i1 * 3 + 2]
        };
        Real rr = r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2];
        if (!(rr < sf.r2_cutoff)) continue;  // not `>=`: a NaN pair is skipped, as in score()

        rr = sqrt(rr);
        const Real d = rr - udflex_mol.r1_plus_r2_inter[i];
        const int t1 = udflex_mol.vina_types[i1], t2 = udfix_mol.vina_types[i2];
        // Each `e += w * x` stays one expression, as in score(), so the compiler contracts it the
        // same way and the running sum matches center[1] exactly.
        Real f = 0;
        Real e = sf.eval_ef(d, t1, t2, &f, atp);
        VinaTerms pt = sf.eval_ef_decomp(d, t1, t2, atp);
        if (use_elec){
            Real q1 = udflex_mol.charges_ua[i1];
            if (udflex_mol.charged_group[i1]) q1 *= sf.elec_charged_scale_val();
            Real f_el = 0;
            const Real e_el = sf.elec_pair(q1 * udfix_mol.charges_ua[i2], rr, &f_el);
            e += sf.weight_elec_val() * e_el;
            pt.elec = sf.weight_elec_val() * e_el;
        }
        if (use_polar_env && udflex_mol.polar_env_class[i1] && t2 == VN_TYPE_C_H){
            const Real sw = Real(1) / (Real(1) + std::exp((rr - POLAR_ENV_RADIUS) / POLAR_ENV_WIDTH));
            polar_env_count[i1] += sw;
            if (out_residues) polar_env_pairs.push_back({i, sw});
        }
        if (use_arom && udflex_mol.aromatic[i1] && udfix_mol.arom_ring_c[i2]
            && atp.is_hydrophobic[t1] && atp.is_hydrophobic[t2]){
            Real f_ar = 0;
            const Real e_ar = sf.vina_hydrophobic(d, &f_ar);
            e += w_arom * e_ar;
            pt.arom_contact = w_arom * e_ar;
        }
        if (angular_hbond && (atp.is_donor[t1] || atp.is_acceptor[t1]) && xs_h_bond_possible(t1, t2, atp)){
            Real f_hb = 0;
            Real e_hb = sf.weight_hbond_val() * sf.vina_hbond(d, &f_hb);
            if (std::abs(e_hb) > Real(1e-12)){
                const Real aw = angular_hbond_pair_weight(i1, i2, flex_coords, udfix_mol, udflex_mol, atp, h_topo, rh_topo);
                e += (aw - Real(1)) * e_hb;
                pt.hbond += (aw - Real(1)) * e_hb;
            }
        }
        pt.total = e;
        e_inter += e;
        out_atoms[i1] += pt;
        if (out_residues) (*out_residues)[udfix_mol.atom_to_residue[i2]] += pt;
    }

    if (use_polar_env){
        Real credit = 0;
        for (int i = 0; i < udflex_mol.natom; i++){
            if (!udflex_mol.polar_env_class[i]) continue;
            const Real c = Real(1) - std::exp(-polar_env_count[i] / POLAR_ENV_CAP);
            credit += c;
            out_atoms[i].polar_env = w_polar_env * c;
            out_atoms[i].total += out_atoms[i].polar_env;
        }
        e_inter += w_polar_env * credit;
        // Residue share of an atom's saturated credit: proportional to that residue's raw counts
        for (const auto& [ip, sw] : polar_env_pairs){
            const int i1 = udflex_mol.inter_pairs[ip * 2], i2 = udflex_mol.inter_pairs[ip * 2 + 1];
            const Real share = out_atoms[i1].polar_env * (sw / polar_env_count[i1]);
            VinaTerms& res = (*out_residues)[udfix_mol.atom_to_residue[i2]];
            res.polar_env += share;
            res.total += share;
        }
    }
    if (out_inter) *out_inter = e_inter;
}

#endif //SCORE_H
