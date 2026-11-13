//
// Geometric H-bond validation independent of Vina hbond scoring.
// Checks all donor/acceptor ligand atoms against all receptor atoms.
//

#ifndef RESCORE_HBOND_H
#define RESCORE_HBOND_H

#include "score/vina.h"
#include "model/model.h"

/**
 * @brief Geometric H-bond rescore based on all atom coordinates.
 *
 * Evaluates every ligand donor/acceptor atom — independent of whether Vina
 * assigned an hbond term — and zeros the hbond contribution when no valid
 * hydrogen-bond geometry exists.
 *
 * - Ligand DONOR atoms: require at least one bonded H with D-H...A angle > 100 deg
 *   and H...A distance < 3.0 A.
 * - Ligand ACCEPTOR-only atoms: uses RecDonorHTopo for proper D-H...A angle check
 *   when receptor bonded-H data is available; falls back to D...A distance < 3.5 A
 *   if receptor H positions are unavailable.
 * - DONOR+ACCEPTOR atoms: try donor check first, then acceptor check.
 *
 * If validation fails, hbond is zeroed and total is recalculated.
 */
inline void rescore_hbond(std::vector<VinaTerms>& atom_terms,
                          const Real* flex_coords,
                          const UDFixMol& udfix_mol,
                          const UDFlexMol& udflex_mol,
                          const AtomTypeProps& atp){

    const Real max_ha_dist = 3.0;       // H...A distance threshold
    const Real max_da_dist = 3.5;       // D...A distance fallback for acceptor check
    const Real min_dha_angle_cos = -0.17365; // cos(100 deg); angle > 100 => cos < -0.17365
    const Real max_bond_dist_sq = 1.3 * 1.3; // bonded-H detection radius squared
    const Real r2_cutoff = 64.0;        // same cutoff as Vina SF

    for (int i = 0; i < udflex_mol.natom; i++){
        if (atom_terms[i].hbond == 0) continue;

        int type_i = udflex_mol.vina_types[i];
        bool is_donor = atp.is_donor[type_i];
        bool is_acceptor = atp.is_acceptor[type_i];

        // Skip atoms that are neither donor nor acceptor
        if (!is_donor && !is_acceptor) continue;

        const Real* pos_i = flex_coords + i * 3;

        // --- Collect bonded H atoms for donor check ---
        std::vector<int> bonded_h;
        if (is_donor){
            for (int h = 0; h < udflex_mol.natom; h++){
                if (udflex_mol.vina_types[h] != VN_TYPE_H) continue;
                const Real* pos_h = flex_coords + h * 3;
                Real dx = pos_h[0] - pos_i[0];
                Real dy = pos_h[1] - pos_i[1];
                Real dz = pos_h[2] - pos_i[2];
                if (dx*dx + dy*dy + dz*dz < max_bond_dist_sq){
                    bonded_h.push_back(h);
                }
            }
        }

        bool valid = false;

        // --- Case A: Ligand atom is a donor (with bonded H) ---
        if (is_donor && !bonded_h.empty()){
            for (int p = 0; p < (int)udflex_mol.inter_pairs.size() / 2 && !valid; p++){
                if (udflex_mol.inter_pairs[p * 2] != i) continue;
                int j = udflex_mol.inter_pairs[p * 2 + 1];
                int type_j = udfix_mol.vina_types[j];
                if (!atp.is_acceptor[type_j]) continue;

                const Real* pos_j = udfix_mol.coords.data() + j * 3;
                Real rr = 0;
                for (int d = 0; d < 3; d++){
                    Real dd = pos_j[d] - pos_i[d];
                    rr += dd * dd;
                }
                if (rr >= r2_cutoff) continue;

                // Check each bonded H
                for (int hi : bonded_h){
                    const Real* pos_h = flex_coords + hi * 3;
                    // H...A distance
                    Real ha_vec[3] = {pos_j[0] - pos_h[0],
                                      pos_j[1] - pos_h[1],
                                      pos_j[2] - pos_h[2]};
                    Real ha_dist = sqrt(ha_vec[0]*ha_vec[0] + ha_vec[1]*ha_vec[1] + ha_vec[2]*ha_vec[2]);
                    if (ha_dist >= max_ha_dist) continue;

                    // D-H vector (from H toward D)
                    Real dh_vec[3] = {pos_i[0] - pos_h[0],
                                      pos_i[1] - pos_h[1],
                                      pos_i[2] - pos_h[2]};
                    Real dh_dist = sqrt(dh_vec[0]*dh_vec[0] + dh_vec[1]*dh_vec[1] + dh_vec[2]*dh_vec[2]);

                    if (dh_dist < 1e-6 || ha_dist < 1e-6) continue;

                    // cos(D-H...A) = dot(H->D, H->A) / (|H->D| * |H->A|)
                    Real cos_angle = (dh_vec[0]*ha_vec[0] + dh_vec[1]*ha_vec[1] + dh_vec[2]*ha_vec[2])
                                     / (dh_dist * ha_dist);

                    // angle > 100 deg  =>  cos(angle) < cos(100) ~ -0.17365
                    if (cos_angle < min_dha_angle_cos){
                        valid = true;
                        break;
                    }
                }
            }
        }

        // --- Case B: Ligand atom is acceptor (receptor is donor) ---
        // Use RecDonorHTopo for proper D-H...A angle check when available;
        // fall back to D...A distance check if receptor H data is missing.
        if (!valid && is_acceptor){
            const auto& rh = udfix_mol.rec_donor_h;
            for (int p = 0; p < (int)udflex_mol.inter_pairs.size() / 2 && !valid; p++){
                if (udflex_mol.inter_pairs[p * 2] != i) continue;
                int j = udflex_mol.inter_pairs[p * 2 + 1];
                int type_j = udfix_mol.vina_types[j];
                if (!atp.is_donor[type_j]) continue;

                const Real* pos_j = udfix_mol.coords.data() + j * 3;
                Real da_dist_sq = 0;
                for (int d = 0; d < 3; d++){
                    Real dd = pos_j[d] - pos_i[d];
                    da_dist_sq += dd * dd;
                }
                if (da_dist_sq >= r2_cutoff) continue;

                // Try proper D-H...A angle check using receptor bonded-H
                if (rh.natom > 0 && j < rh.natom
                    && rh.h_start[j] < rh.h_start[j + 1]) {
                    for (int ih = rh.h_start[j]; ih < rh.h_start[j + 1] && !valid; ih++){
                        const Real* pos_h = rh.h_coords + ih * 3;
                        // H...A (ligand acceptor)
                        Real ha_vec[3] = {pos_i[0] - pos_h[0],
                                          pos_i[1] - pos_h[1],
                                          pos_i[2] - pos_h[2]};
                        Real ha_dist = sqrt(ha_vec[0]*ha_vec[0] + ha_vec[1]*ha_vec[1] + ha_vec[2]*ha_vec[2]);
                        if (ha_dist >= max_ha_dist) continue;

                        // D-H vector (from H toward D)
                        Real dh_vec[3] = {pos_j[0] - pos_h[0],
                                          pos_j[1] - pos_h[1],
                                          pos_j[2] - pos_h[2]};
                        Real dh_dist = sqrt(dh_vec[0]*dh_vec[0] + dh_vec[1]*dh_vec[1] + dh_vec[2]*dh_vec[2]);

                        if (dh_dist < 1e-6 || ha_dist < 1e-6) continue;

                        Real cos_angle = (dh_vec[0]*ha_vec[0] + dh_vec[1]*ha_vec[1] + dh_vec[2]*ha_vec[2])
                                         / (dh_dist * ha_dist);

                        if (cos_angle < min_dha_angle_cos){
                            valid = true;
                        }
                    }
                } else {
                    // Fallback: no receptor H data for this atom — use D...A distance
                    if (da_dist_sq < max_da_dist * max_da_dist){
                        valid = true;
                    }
                }
            }
        }

        // --- Zero out hbond if no valid geometry found ---
        if (!valid){
            atom_terms[i].total -= atom_terms[i].hbond;
            atom_terms[i].hbond = 0;
        }
    }
}

#endif //RESCORE_HBOND_H
