//
// SDF V2000 writer implementation.
//

#include "sdf_writer.h"
#include <spdlog/spdlog.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <string>

namespace {

// Element symbols indexed by atomic number (0 = dummy, 1 = H, ..., 118 = Og).
// Only elements likely to appear in drug-like molecules are spelled out;
// the rest are left as "??" which is safe (SDF readers will ignore them).
const char* ELEMENT_SYMBOL[] = {
    "* ", "H ", "He", "Li", "Be", "B ", "C ", "N ", "O ", "F ", "Ne",
    "Na", "Mg", "Al", "Si", "P ", "S ", "Cl", "Ar", "K ", "Ca",
    "Sc", "Ti", "V ", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y ", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I ", "Xe"
};
constexpr int MAX_KNOWN_ANUM = 54;

/// Return element symbol string (trimmed) for an atomic number.
const char* anum_to_symbol(int anum) {
    if (anum >= 0 && anum <= MAX_KNOWN_ANUM) return ELEMENT_SYMBOL[anum];
    return "??";
}

/// Write a single SDF property tag.
void write_prop(FILE* f, const char* key, const char* value) {
    fprintf(f, ">  <%s>\n%s\n\n", key, value);
}

/// Write a floating-point property.
void write_prop_real(FILE* f, const char* key, Real value) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.3f", static_cast<double>(value));
    write_prop(f, key, buf);
}

/// Write a space-separated array of floats as a property.
void write_prop_array(FILE* f, const char* key, const Real* data, int n) {
    fprintf(f, ">  <%s>\n", key);
    for (int i = 0; i < n; i++) {
        if (i > 0) fputc(' ', f);
        fprintf(f, "%.4f", static_cast<double>(data[i]));
    }
    fprintf(f, "\n\n");
}

/// Write a per-atom array over the written molecule (the first n_written of n flex atoms). When
/// the flex system is larger, the sum over the unwritten atoms goes to decomp_unwritten_<X> for
/// key atom_<X>, so the pose totals still reconcile.
void write_prop_atom_array(FILE* f, const char* key, const Real* data, int n, int n_written) {
    write_prop_array(f, key, data, std::min(n, n_written));
    if (n <= n_written) return;
    double rest = 0;
    for (int k = n_written; k < n; k++) rest += data[k];
    const std::string rest_key = std::string("decomp_unwritten_") + (key + std::strlen("atom_"));
    write_prop_real(f, rest_key.c_str(), static_cast<Real>(rest));
}

} // anonymous namespace


void write_poses_to_sdf(
    const std::string& fp_sdf,
    const std::vector<SDFMolData>& sdf_mol_list,
    const std::vector<std::string>& flex_names,
    const std::vector<std::vector<int>>& filtered_pose_inds_list,
    const FlexPose* flex_pose_list_res,
    const Real* flex_pose_list_real_res,
    const std::vector<int>& list_i_real,
    const std::vector<std::vector<std::vector<VinaTerms>>>& decomp_data,
    const std::vector<std::vector<std::vector<FFAtomTerms>>>& ff_decomp_data,
    const std::vector<std::vector<std::array<Real, 3>>>& ff_pose_energies,
    const std::vector<std::vector<std::vector<VinaTerms>>>& residue_decomp_data,
    const std::vector<ResidueKey>& residues,
    const std::vector<std::vector<ScoreDiag>>& pose_diag,
    const std::vector<std::vector<std::array<Real, 3>>>& pose_score_parts,
    const VinaWeights& decomp_weights)
{
    FILE* f = fopen(fp_sdf.c_str(), "w");
    if (!f) {
        spdlog::error("Cannot open SDF output file: {}", fp_sdf);
        return;
    }

    bool has_decomp = !decomp_data.empty();
    bool has_ff_decomp = !ff_decomp_data.empty();
    bool has_ff_energies = !ff_pose_energies.empty();
    bool has_residue_decomp = !residue_decomp_data.empty() && !residues.empty();

    // The records of ligand i, written to `f` with the same calls whatever `f` is (the output file
    // or a memory stream of this ligand)
    auto write_ligand = [&](int i, FILE* f) {
        // Select per-ligand SDFMolData: use index i if available, else first
        const SDFMolData& sdf_mol = (i < static_cast<int>(sdf_mol_list.size()))
            ? sdf_mol_list[i] : sdf_mol_list[0];

        int natom = sdf_mol.natom;
        int nbond = static_cast<int>(sdf_mol.bonds.size());

        // Count non-zero formal charges for M  CHG lines
        std::vector<std::pair<int, int>> charged_atoms;
        for (int a = 0; a < natom; a++) {
            if (sdf_mol.formal_charge[a] != 0) {
                charged_atoms.push_back({a, sdf_mol.formal_charge[a]});
            }
        }

        int pose_local_idx = 0;
        for (auto& j : filtered_pose_inds_list[i]) {
            // --- Header block ---
            fprintf(f, "%s\n", sdf_mol.mol_name.c_str());
            fprintf(f, "     UniDock2       3D\n");
            fprintf(f, "\n");

            // --- Counts line ---
            fprintf(f, "%3d%3d  0  0  0  0  0  0  0  0999 V2000\n", natom, nbond);

            // --- Atom block ---
            int coord_start = list_i_real[j * 2];
            for (int a = 0; a < natom; a++) {
                Real x = flex_pose_list_real_res[coord_start + a * 3 + 0];
                Real y = flex_pose_list_real_res[coord_start + a * 3 + 1];
                Real z = flex_pose_list_real_res[coord_start + a * 3 + 2];
                const char* sym = anum_to_symbol(sdf_mol.anum[a]);
                fprintf(f, "%10.4f%10.4f%10.4f %-3s 0  0  0  0  0  0  0  0  0  0  0  0\n",
                        static_cast<double>(x),
                        static_cast<double>(y),
                        static_cast<double>(z),
                        sym);
            }

            // --- Bond block ---
            // Stereo column 0: DMS wedge flags describe the input orientation, not the docked pose
            for (const auto& b : sdf_mol.bonds) {
                fprintf(f, "%3d%3d%3d%3d\n",
                        b.p0 + 1, b.p1 + 1, b.order, 0);
            }

            // --- M  CHG lines (formal charges) ---
            if (!charged_atoms.empty()) {
                for (size_t ci = 0; ci < charged_atoms.size(); ci += 8) {
                    size_t count = std::min(charged_atoms.size() - ci, size_t(8));
                    fprintf(f, "M  CHG%3d", static_cast<int>(count));
                    for (size_t k = 0; k < count; k++) {
                        fprintf(f, " %3d %3d",
                                charged_atoms[ci + k].first + 1,
                                charged_atoms[ci + k].second);
                    }
                    fprintf(f, "\n");
                }
            }
            fprintf(f, "M  END\n");

            // --- Properties block ---
            for (const auto& p : sdf_mol.properties) {
                write_prop(f, p.key.c_str(), p.value.c_str());
            }

            write_prop(f, "ud2_molecule_name", sdf_mol.mol_name.c_str());
            {
                char buf[32];
                snprintf(buf, sizeof(buf), "%d", pose_local_idx);
                write_prop(f, "ud2_pose_index", buf);
            }

            // Two scores, because choosing the best pose of one ligand and comparing different
            // ligands are different questions:
            //   vina_pose_score      ranks the poses of THIS ligand; the engine writes them in this order
            //   vina_affinity_score  compares ligands; this is what a screening run should rank on
            // They differ even under one weight set: the pose score is intra + inter + box penalty,
            // the affinity subtracts the rank-1 pose's intra energy and divides by 1 + 0.05846 * Nrot
            // (Vina). Settings.affinity_weights additionally gives the affinity its own weight set.
            write_prop_real(f, "vina_pose_score", flex_pose_list_res[j].rot_vec[2]);
            write_prop_real(f, "vina_affinity_score", flex_pose_list_res[j].rot_vec[0]);

            // Vina energies
            write_prop_real(f, "vina_binding_free_energy", flex_pose_list_res[j].rot_vec[0]);
            write_prop_real(f, "vina_intra_inter", flex_pose_list_res[j].rot_vec[1]);
            write_prop_real(f, "vina_intra", flex_pose_list_res[j].center[0]);
            write_prop_real(f, "vina_inter", flex_pose_list_res[j].center[1]);
            write_prop_real(f, "vina_box_penalty", flex_pose_list_res[j].center[2]);
            write_prop_real(f, "vina_torsion_number_energy", flex_pose_list_res[j].rot_vec[3]);

            // Raw, unweighted accumulators of the rescoring terms. They are written whatever the
            // weights are, so a different functional form can be fitted from one docking run
            // instead of one run per candidate.
            if (i < static_cast<int>(pose_diag.size())
                && pose_local_idx < static_cast<int>(pose_diag[i].size())) {
                const ScoreDiag& dg = pose_diag[i][pose_local_idx];
                write_prop_real(f, "ud2_polar_env_raw", dg.polar_env);
                write_prop_real(f, "ud2_arom_contact_raw", dg.arom_contact);
                write_prop_real(f, "ud2_torsion_strain_raw", dg.torsion_strain);
                write_prop_real(f, "ud2_elec_neutral", dg.elec_neutral);
                write_prop_real(f, "ud2_elec_charged", dg.elec_charged);
                write_prop_real(f, "ud2_polar_env_burial", dg.burial);
                write_prop_real(f, "ud2_polar_env_hbond_sat", dg.hbond_sat);
                write_prop_real(f, "ud2_n_class", static_cast<Real>(dg.n_class));
                write_prop_real(f, "ud2_n_heavy", static_cast<Real>(dg.n_heavy));
                if (dg.n_ring_bias > 0) write_prop_real(f, "ud2_ring_bias_raw", dg.ring_bias);
                if (dg.has_bias_energy) write_prop_real(f, "ud2_bias_energy", dg.bias_energy);
                if (!dg.heavy_type.empty()) {
                    // Heavy atoms of the written molecule only (heavy_atom is ascending)
                    int nh = static_cast<int>(dg.heavy_type.size());
                    if (dg.heavy_atom.size() == dg.heavy_type.size()) {
                        nh = static_cast<int>(std::lower_bound(dg.heavy_atom.begin(), dg.heavy_atom.end(), natom)
                                              - dg.heavy_atom.begin());
                    }
                    write_prop_array(f, "ud2_heavy_type",   dg.heavy_type.data(),   nh);
                    write_prop_array(f, "ud2_heavy_class",  dg.heavy_class.data(),  nh);
                    write_prop_array(f, "ud2_heavy_apolar", dg.heavy_apolar.data(), nh);
                    write_prop_array(f, "ud2_heavy_polar",  dg.heavy_polar.data(),  nh);
                    write_prop_array(f, "ud2_heavy_hbond",  dg.heavy_hbond.data(),  nh);
                }
            }

            // FF energy summary
            if (has_ff_energies && i < static_cast<int>(ff_pose_energies.size())
                && pose_local_idx < static_cast<int>(ff_pose_energies[i].size())) {
                const auto& ffe = ff_pose_energies[i][pose_local_idx];
                write_prop_real(f, "ff_total_energy", ffe[0]);
                write_prop_real(f, "ff_interaction_energy", ffe[1]);
                write_prop_real(f, "ff_intra_ligand_energy", ffe[2]);
            }

            // Parts of vina_pose_score: inter + intra + penalty == vina_pose_score
            if (i < static_cast<int>(pose_score_parts.size())
                && pose_local_idx < static_cast<int>(pose_score_parts[i].size())) {
                const auto& parts = pose_score_parts[i][pose_local_idx];
                write_prop_real(f, "vina_pose_inter", parts[1]);
                write_prop_real(f, "vina_pose_intra", parts[0]);
                write_prop_real(f, "vina_pose_penalty", parts[2]);
            }

            // Per-atom Vina decomposition
            if (has_decomp && i < static_cast<int>(decomp_data.size())
                && pose_local_idx < static_cast<int>(decomp_data[i].size())) {
                const auto& at = decomp_data[i][pose_local_idx];
                int n = static_cast<int>(at.size());
                std::vector<Real> buf(n);
                auto write_atom = [&](auto field, const char* key) {
                    for (int k = 0; k < n; k++) buf[k] = at[k].*field;
                    write_prop_atom_array(f, key, buf.data(), n, natom);
                };
                write_atom(&VinaTerms::total, "atom_inter_energy");
                write_atom(&VinaTerms::gauss1, "atom_gauss1");
                write_atom(&VinaTerms::gauss2, "atom_gauss2");
                write_atom(&VinaTerms::repulsion, "atom_repulsion");
                write_atom(&VinaTerms::hydrophobic, "atom_hydrophobic");
                write_atom(&VinaTerms::hbond, "atom_hbond");
                // Extra terms only when the pose weight set uses them, so every pose of a run has
                // the same columns and default-weight output keeps the Vina ones
                if (decomp_weights.elec != Real(0)) write_atom(&VinaTerms::elec, "atom_elec");
                if (decomp_weights.arom_contact != Real(0)) write_atom(&VinaTerms::arom_contact, "atom_arom_contact");
                if (decomp_weights.polar_env != Real(0)) write_atom(&VinaTerms::polar_env, "atom_polar_env");
            }

            // Per-residue Vina decomposition
            if (has_residue_decomp && i < static_cast<int>(residue_decomp_data.size())
                && pose_local_idx < static_cast<int>(residue_decomp_data[i].size())) {
                const auto& rt = residue_decomp_data[i][pose_local_idx];
                int nres = static_cast<int>(rt.size());

                {
                    fprintf(f, ">  <residue_labels>\n");
                    for (int k = 0; k < nres; k++) {
                        if (k > 0) fputc(' ', f);
                        const auto& rk = residues[k];
                        fprintf(f, "%s:%s:%d", rk.chain.c_str(), rk.resname.c_str(), rk.resid);
                    }
                    fprintf(f, "\n\n");
                }

                std::vector<Real> rbuf(nres);
                auto extract_res = [&](auto field) {
                    for (int k = 0; k < nres; k++) rbuf[k] = rt[k].*field;
                };
                extract_res(&VinaTerms::total);
                write_prop_array(f, "residue_inter_energy", rbuf.data(), nres);
                extract_res(&VinaTerms::gauss1);
                write_prop_array(f, "residue_gauss1", rbuf.data(), nres);
                extract_res(&VinaTerms::gauss2);
                write_prop_array(f, "residue_gauss2", rbuf.data(), nres);
                extract_res(&VinaTerms::repulsion);
                write_prop_array(f, "residue_repulsion", rbuf.data(), nres);
                extract_res(&VinaTerms::hydrophobic);
                write_prop_array(f, "residue_hydrophobic", rbuf.data(), nres);
                extract_res(&VinaTerms::hbond);
                write_prop_array(f, "residue_hbond", rbuf.data(), nres);
                if (decomp_weights.elec != Real(0)) {
                    extract_res(&VinaTerms::elec);
                    write_prop_array(f, "residue_elec", rbuf.data(), nres);
                }
                if (decomp_weights.arom_contact != Real(0)) {
                    extract_res(&VinaTerms::arom_contact);
                    write_prop_array(f, "residue_arom_contact", rbuf.data(), nres);
                }
                if (decomp_weights.polar_env != Real(0)) {
                    extract_res(&VinaTerms::polar_env);
                    write_prop_array(f, "residue_polar_env", rbuf.data(), nres);
                }
            }

            // Per-atom FF decomposition
            if (has_ff_decomp && i < static_cast<int>(ff_decomp_data.size())
                && pose_local_idx < static_cast<int>(ff_decomp_data[i].size())) {
                const auto& ft = ff_decomp_data[i][pose_local_idx];
                int n = static_cast<int>(ft.size());
                std::vector<Real> buf(n);
                auto write_atom = [&](auto field, const char* key) {
                    for (int k = 0; k < n; k++) buf[k] = ft[k].*field;
                    write_prop_atom_array(f, key, buf.data(), n, natom);
                };
                write_atom(&FFAtomTerms::total, "atom_ff_total");
                write_atom(&FFAtomTerms::inter_lj, "atom_ff_inter_lj");
                write_atom(&FFAtomTerms::inter_coulomb, "atom_ff_inter_coulomb");
                write_atom(&FFAtomTerms::intra_lj, "atom_ff_intra_lj");
                write_atom(&FFAtomTerms::intra_coulomb, "atom_ff_intra_coulomb");
                write_atom(&FFAtomTerms::torsion, "atom_ff_torsion");
                // Harmonic bond / angle parts, present when the ligand DMS carries stretch_harm /
                // angle_harm tables; with them the written parts sum to atom_ff_total
                const bool has_bonded = std::any_of(ft.begin(), ft.end(), [](const FFAtomTerms& t) {
                    return t.bond != Real(0) || t.angle != Real(0);
                });
                if (has_bonded) {
                    write_atom(&FFAtomTerms::bond, "atom_ff_bond");
                    write_atom(&FFAtomTerms::angle, "atom_ff_angle");
                }
            }

            // --- Record separator ---
            fprintf(f, "$$$$\n");
            pose_local_idx++;
        }
    };

    // Format the ligands in parallel, each into its own memory stream, then write the streams in
    // ligand order. Every byte of a ligand comes from the same fprintf sequence as a direct write,
    // and the streams are concatenated in order, so the file is identical to the serial write.
    const int nlig = static_cast<int>(flex_names.size());
    std::vector<std::string> blocks(nlig);
    std::atomic<int> nfail{0};
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = std::max(1, std::min(omp_get_max_threads(), nlig / 8));  // serial below 8 ligands per thread
#endif
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) if(nthreads > 1)
#endif
    for (int i = 0; i < nlig; i++) {
        char* buf = nullptr;
        size_t len = 0;
        FILE* mf = open_memstream(&buf, &len);
        if (!mf) {
            nfail++;
            continue;
        }
        write_ligand(i, mf);
        fclose(mf);
        blocks[i].assign(buf, len);
        free(buf);
    }
    if (nfail > 0) {
        spdlog::warn("open_memstream failed for {} ligands; writing the records directly", nfail.load());
        for (int i = 0; i < nlig; i++) write_ligand(i, f);
    } else {
        for (const auto& b : blocks) fwrite(b.data(), 1, b.size(), f);
    }

    fclose(f);
    spdlog::info("Wrote SDF output: {} ({} ligands)", fp_sdf, flex_names.size());
}
