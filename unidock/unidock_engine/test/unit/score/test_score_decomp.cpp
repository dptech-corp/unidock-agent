//
// score_decomp(): the per-atom / per-residue decomposition reconciles with the inter part of
// score() under the same weight set (angular HBond on and off, non-default weights, no receptor
// H table, a charged group), each residue matches the ligand scored against that residue alone,
// and DockTask runs write SDF records that satisfy the file-level contract:
//   vina_pose_inter + vina_pose_intra + vina_pose_penalty == vina_pose_score
//   sum(atom_inter_energy) == vina_pose_inter
//   atom_inter_energy == sum of that atom's written term columns
//   sum(residue_X) == sum(atom_X) for every term X
//   ff_total_energy == ff_interaction_energy + ff_intra_ligand_energy
// plus the DockTask FF rules: no pose-score parts without --decomp, FF minimisation leaves a
// ligand without a loaded FF alone and turns the hydrogen-only torsions kept out of the search,
// which the affinity still counts; a ligand's aromatic-ring bias rows reach its own search
// and SDF record only; and a biased run's pose pass (BT_POS and BT_ALIGN, atom and ring rows)
// orders by vina_pose_score + bias energy, writes ud2_bias_energy, references every affinity to
// the intra of the written top pose as an unbiased run does, and leaves every other tag of a pose
// as the unbiased run writes it, while rows without a bias type, also under constraint docking,
// change no byte (plan step 7).
//
// score.h defines non-inline globals (SF, score, score_decomp) that DockTask.o already carries,
// so this file declares what it uses instead of including score.h (see test_box_penalty.cpp).
// A signature mismatch is a link error.
//

#include <catch2/catch_amalgamated.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include "model/model.h"
#include "score/vina.h"
#include "format/dms_reader.h"
#include "cpu/score_cpu.h"
#include "constants/constants.h"
#include "task/DockTask.h"
#include "config/parse.h"

extern thread_local Vina SF;
void score(FlexPose* out_pose, const Real* flex_coords, const UDFixMol& udfix_mol, const UDFlexMol& udflex_mol,
           const Box& box, const AtomTypeProps& atp, bool angular_hbond, const BondedHTopo* h_topo,
           const RecDonorHTopo* rh_topo, ScoreDiag* out_diag);
void score_decomp(Vina& sf, const Real* flex_coords, const UDFixMol& udfix_mol, const UDFlexMol& udflex_mol,
                  const AtomTypeProps& atp, bool angular_hbond, const BondedHTopo* h_topo,
                  const RecDonorHTopo* rh_topo, std::vector<VinaTerms>& out_atoms,
                  std::vector<VinaTerms>* out_residues, Real* out_inter);

namespace {

const std::string TYK2_DIR = std::string(TEST_DATA_DIR) + "ff_minimize/tyk2/";

// the engine's default_rescore_weights() of model.h, plus the Coulomb term, so every inter term is active
VinaWeights rescore_set(){
    VinaWeights w;
    w.gauss1 = -0.041677; w.gauss2 = -0.002722; w.hydrophobic = -0.042773; w.hbond = -0.791096;
    w.intra = 0.60294; w.polar_env = -0.398679; w.arom_contact = 0.03834; w.torsion_strain = 0.216366;
    w.elec = 0.114; w.elec_charged_scale = 0.57;
    return w;
}

// Rigid shift of the DMS pose that brings an aromatic ligand carbon within hydrophobic range of a
// receptor ring carbon (TYR 91 sits at 5.5 A, beyond the 5.3 A C-C ramp end, in the DMS pose) and
// moves hbond pairs off the full-weight angle, so arom_contact and the angular hbond weight are
// exercised; the DMS pose alone has neither.
const Real SHIFT[3] = {0.0, 1.5, -1.0};

// The pairwise columns go to one ligand atom and one residue; polar_env is the per-atom credit
Real VinaTerms::* const PAIR_COLS[] = {
    &VinaTerms::gauss1, &VinaTerms::gauss2, &VinaTerms::repulsion, &VinaTerms::hydrophobic,
    &VinaTerms::hbond, &VinaTerms::elec, &VinaTerms::arom_contact,
};
Real VinaTerms::* const TERM_COLS[] = {
    &VinaTerms::gauss1, &VinaTerms::gauss2, &VinaTerms::repulsion, &VinaTerms::hydrophobic,
    &VinaTerms::hbond, &VinaTerms::elec, &VinaTerms::arom_contact, &VinaTerms::polar_env,
};

/// score_decomp's running sum against score()'s center[1]. Both evaluate the same expressions
/// in the same order, and agree bit for bit in the builds tested, but whether `e += w * x`
/// is contracted to an FMA is up to the compiler at each inlined call site, so a few ulp are
/// allowed.
bool same_up_to_ulp(Real a, Real b){
    return std::fabs(a - b) <= 4 * std::numeric_limits<Real>::epsilon() * std::max(Real(1), std::fabs(b));
}

/// Puts the global SF weights back when a test case ends, pass or fail (DockTask re-weights SF).
struct SFWeightsGuard {
    const VinaWeights saved = SF.get_weights();
    ~SFWeightsGuard(){ SF.set_weights(saved); }
};

/// Path in the system temp directory, removed when the guard goes out of scope (also when a
/// REQUIRE fails).
struct TmpFile {
    const std::string path;
    TmpFile(const std::string& stem, const std::string& ext)
        : path((std::filesystem::temp_directory_path()
                / (stem + "_" + std::to_string(static_cast<int>(getpid())) + ext)).string()) {}
    ~TmpFile(){ std::remove(path.c_str()); }
};

/// tyk2 ejm42 (enriched DMS) in its receptor, pairs and radii set up as main.cpp / DockTask do.
struct Tyk2 {
    AtomTypeProps atp;
    UDFlexMol lig; FFMinData ff_lig;
    UDFixMol rec; FFMinData ff_rec;
    Box box;
    BondedHTopo h_topo = {};

    Tyk2(){
        init_default_atom_type_props(atp);
        read_ligand_dms_full(TYK2_DIR + "ligand_ejm42_enriched.dms", lig, ff_lig, atp, false);
        // The test DMS has no ud2_atom_aromatic table; flag the aromatic ring atoms (RDKit on
        // ligand_ejm42.sdf, same atom order) so the aromatic-contact term is exercised.
        lig.aromatic.assign(lig.natom, 0);
        for (int a : {0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15}) lig.aromatic[a] = 1;

        const Real half = 15, cutoff = 8;
        box.x_lo = lig.center[0] - half; box.x_hi = lig.center[0] + half;
        box.y_lo = lig.center[1] - half; box.y_hi = lig.center[1] + half;
        box.z_lo = lig.center[2] - half; box.z_hi = lig.center[2] + half;
        Box box_protein = box;
        box_protein.x_lo -= cutoff; box_protein.x_hi += cutoff;
        box_protein.y_lo -= cutoff; box_protein.y_hi += cutoff;
        box_protein.z_lo -= cutoff; box_protein.z_hi += cutoff;
        read_receptor_dms_full(TYK2_DIR + "receptor_enriched.dms", box_protein, rec, ff_rec, atp);

        for (int i = 0; i < lig.natom; i++){
            if (atp.is_hydrogen[lig.vina_types[i]]) continue;
            for (int j = 0; j < rec.natom; j++){
                if (atp.is_hydrogen[rec.vina_types[j]]) continue;
                lig.inter_pairs.push_back(i);
                lig.inter_pairs.push_back(j);
            }
        }
        build_bonded_h_topo(h_topo, lig.coords.data(), lig.vina_types.data(), lig.natom);
    }
    ~Tyk2(){ free_bonded_h_topo(h_topo); }

    /// Copy with r1 + r2 filled, for direct score() / score_decomp() calls.
    UDFlexMol scored_lig() const {
        UDFlexMol m = lig;
        for (size_t p = 0; p < m.inter_pairs.size(); p += 2){
            m.r1_plus_r2_inter.push_back(VN_VDW_RADII[m.vina_types[m.inter_pairs[p]]]
                                         + VN_VDW_RADII[rec.vina_types[m.inter_pairs[p + 1]]]);
        }
        for (size_t p = 0; p < m.intra_pairs.size(); p += 2){
            m.r1_plus_r2_intra.push_back(VN_VDW_RADII[m.vina_types[m.intra_pairs[p]]]
                                         + VN_VDW_RADII[m.vina_types[m.intra_pairs[p + 1]]]);
        }
        return m;
    }

    std::vector<Real> shifted(Real dx, Real dy, Real dz) const {
        std::vector<Real> c = lig.coords;
        for (int a = 0; a < lig.natom; a++){ c[a * 3] += dx; c[a * 3 + 1] += dy; c[a * 3 + 2] += dz; }
        return c;
    }

    /// The ligand itself moved by SHIFT, as a DockTask input.
    UDFlexMol shifted_lig() const {
        UDFlexMol m = lig;
        m.coords = shifted(SHIFT[0], SHIFT[1], SHIFT[2]);
        for (int k = 0; k < 3; k++) m.center[k] += SHIFT[k];
        return m;
    }
};

Real col_sum(const std::vector<VinaTerms>& v, Real VinaTerms::* f){
    Real s = 0;
    for (const auto& t : v) s += t.*f;
    return s;
}

/// Per-residue terms recomputed the slow way: the ligand scored against one residue at a time.
/// A residue's pairwise columns are the sums over the atoms of its own run. polar_env saturates
/// per atom over all residues, c(n) = w (1 - exp(-n / cap)), so the atom's full credit is split in
/// proportion to the residue's raw soft count: n_r / n = log(1 - c_r / w) / log(1 - c / w).
/// `mag` gets the matching sums of absolute values, for the tolerances.
std::vector<VinaTerms> residues_one_by_one(Vina& sf, const Real* coords, const Tyk2& t, const UDFlexMol& lig,
                                           bool angular, const std::vector<VinaTerms>& atoms,
                                           std::vector<VinaTerms>& mag){
    std::vector<std::vector<int>> pairs_of(t.rec.residues.size());
    for (size_t p = 0; p < lig.inter_pairs.size() / 2; p++){
        pairs_of[t.rec.atom_to_residue[lig.inter_pairs[p * 2 + 1]]].push_back(static_cast<int>(p));
    }
    const double w = sf.weight_polar_env_val();
    std::vector<VinaTerms> out(pairs_of.size());
    mag.assign(pairs_of.size(), VinaTerms{});
    UDFlexMol one = lig;
    std::vector<VinaTerms> at;
    for (size_t r = 0; r < pairs_of.size(); r++){
        if (pairs_of[r].empty()) continue;
        one.inter_pairs.clear();
        one.r1_plus_r2_inter.clear();
        for (int p : pairs_of[r]){
            one.inter_pairs.push_back(lig.inter_pairs[p * 2]);
            one.inter_pairs.push_back(lig.inter_pairs[p * 2 + 1]);
            one.r1_plus_r2_inter.push_back(lig.r1_plus_r2_inter[p]);
        }
        score_decomp(sf, coords, t.rec, one, t.atp, angular, &t.h_topo, &t.rec.rec_donor_h, at, nullptr, nullptr);
        for (int a = 0; a < lig.natom; a++){
            for (auto f : PAIR_COLS){
                out[r].*f += at[a].*f;
                mag[r].*f += std::fabs(at[a].*f);
            }
            const Real pair_total = at[a].total - at[a].polar_env;
            out[r].total += pair_total;
            mag[r].total += std::fabs(pair_total);
            if (at[a].polar_env == Real(0)) continue;
            const Real share = static_cast<Real>(atoms[a].polar_env * std::log1p(-at[a].polar_env / w)
                                                 / std::log1p(-atoms[a].polar_env / w));
            out[r].polar_env += share;
            out[r].total += share;
            mag[r].polar_env += std::fabs(share);
            mag[r].total += std::fabs(share);
        }
    }
    return out;
}

std::string read_text(const std::string& path){
    std::ifstream in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

/// The records of an SDF text, each up to and including its "$$$$" line.
std::vector<std::string> sdf_records(const std::string& text){
    std::vector<std::string> out;
    size_t start = 0, end;
    while ((end = text.find("$$$$\n", start)) != std::string::npos){
        out.push_back(text.substr(start, end + 5 - start));
        start = end + 5;
    }
    return out;
}

using Record = std::map<std::string, std::string>;

std::vector<Record> sdf_props(const std::string& text){
    std::vector<Record> out;
    for (const auto& rec : sdf_records(text)){
        std::istringstream in(rec);
        Record r;
        std::string line, value;
        while (std::getline(in, line)){
            if (line.rfind(">  <", 0) != 0) continue;
            std::getline(in, value);
            r[line.substr(4, line.size() - 5)] = value;
        }
        out.push_back(std::move(r));
    }
    return out;
}

std::vector<double> numbers(const std::string& s){
    std::istringstream is(s);
    std::vector<double> v;
    double x;
    while (is >> x) v.push_back(x);
    return v;
}

double vsum(const std::vector<double>& v){
    double s = 0;
    for (double x : v) s += x;
    return s;
}

/// DockParam of main.cpp's `task: score`: the input pose is scored as given, not searched.
DockParam score_only_param(const Box& box){
    DockParam dp;
    dp.box = box;
    dp.randomize = false;
    dp.exhaustiveness = 1;
    dp.mc_steps = 0;
    dp.opt_steps = 0;
    dp.refine_steps = 0;
    dp.num_pose = 1;
    dp.energy_range = 999;
    dp.rmsd_limit = 999;
    dp.seed = 1234;
    dp.angular_hbond = true;
    return dp;
}

/// Runs a DockTask on the tyk2 receptor and returns the SDF it writes. One ligand takes the
/// single-ligand path (sdf_mol); several take the merged-DMS path (sdf_mol_list and the
/// per-ligand ff_list).
std::string run_task(const Tyk2& t, const DockParam& dp, const UDFlexMolList& ligs,
                     const std::vector<FFMinData>& ff_list = {}){
    std::vector<std::string> names;
    for (size_t k = 0; k < ligs.size(); k++) names.push_back("ejm42_" + std::to_string(k));
    const std::vector<Real> radii(VN_VDW_RADII, VN_VDW_RADII + 21);
    DockTask task(t.rec, ligs, dp, names, radii, t.atp);
    task.show_score = false;
    SDFMolData sdf;
    read_sdf_mol_data(TYK2_DIR + "ligand_ejm42_enriched.dms", sdf);
    REQUIRE(sdf.natom == t.lig.natom);
    if (ligs.size() == 1) task.sdf_mol = sdf;
    else task.sdf_mol_list.assign(ligs.size(), sdf);
    task.ff_min_data_list = ff_list;
    const TmpFile out("test_score_decomp", ".sdf");
    task.fp_sdf = out.path;
    task.run();
    return read_text(out.path);
}

/// The file-level decomposition contract (docs/energy_decomposition.md), record by record.
/// Returns how many records have vina_pose_inter != vina_inter (the affinity pass re-weighted SF).
int check_decomp_records(const std::vector<Record>& records, const Tyk2& t){
    int n_heavy = 0;
    for (int a = 0; a < t.lig.natom; a++) n_heavy += t.atp.is_hydrogen[t.lig.vina_types[a]] ? 0 : 1;
    const std::vector<std::string> terms = {"gauss1", "gauss2", "repulsion", "hydrophobic", "hbond",
                                            "elec", "arom_contact", "polar_env"};
    int n_pose_ne_last = 0;
    for (const auto& r : records){
        for (const char* key : {"vina_pose_inter", "vina_pose_intra", "vina_pose_penalty", "residue_labels"}){
            REQUIRE(r.count(key) == 1);
        }
        const double pose_score = std::stod(r.at("vina_pose_score"));
        const double p_inter = std::stod(r.at("vina_pose_inter"));
        const double p_intra = std::stod(r.at("vina_pose_intra"));
        const double p_pen = std::stod(r.at("vina_pose_penalty"));
        if (std::fabs(p_inter - std::stod(r.at("vina_inter"))) > 1e-3) n_pose_ne_last++;

        // (1) the parts add up to vina_pose_score (3 printed values at %.3f)
        REQUIRE(std::fabs(p_inter + p_intra + p_pen - pose_score) <= 2e-3);

        // (2) the atoms add up to the pose inter energy
        const auto at_total = numbers(r.at("atom_inter_energy"));
        REQUIRE(at_total.size() == (size_t)t.lig.natom);
        INFO("sum(atom_inter_energy) = " << vsum(at_total) << " vina_pose_inter = " << p_inter);
        REQUIRE(std::fabs(vsum(at_total) - p_inter) <= 5e-4 + 5e-5 * n_heavy + 1e-3);

        // (3) each atom's total is the sum of its written columns
        std::vector<std::vector<double>> cols;
        for (const auto& x : terms){
            REQUIRE(r.count("atom_" + x) == 1);
            cols.push_back(numbers(r.at("atom_" + x)));
            REQUIRE(cols.back().size() == (size_t)t.lig.natom);
        }
        for (int a = 0; a < t.lig.natom; a++){
            double s = 0;
            for (const auto& c : cols) s += c[a];
            REQUIRE(std::fabs(at_total[a] - s) <= 5e-5 * (1 + terms.size()) + 1e-6);
        }

        // (4) residues add up to atoms, per term
        const auto res_total = numbers(r.at("residue_inter_energy"));
        auto nonzero = [](const std::vector<double>& v){ int n = 0; for (double x : v) n += x != 0; return n; };
        REQUIRE(std::fabs(vsum(res_total) - vsum(at_total)) <= 5e-5 * (nonzero(res_total) + t.lig.natom) + 1e-6);
        for (size_t k = 0; k < terms.size(); k++){
            REQUIRE(r.count("residue_" + terms[k]) == 1);
            const auto res = numbers(r.at("residue_" + terms[k]));
            INFO("term " << terms[k] << " sum(residue) = " << vsum(res) << " sum(atom) = " << vsum(cols[k]));
            REQUIRE(std::fabs(vsum(res) - vsum(cols[k])) <= 5e-5 * (nonzero(res) + t.lig.natom) + 1e-6);
        }
    }
    return n_pose_ne_last;
}

} // anonymous namespace


TEST_CASE("score_decomp reconciles with score() under a non-default weight set", "[score_decomp]"){
    Tyk2 t;
    SFWeightsGuard guard;
    const UDFlexMol lig = t.scored_lig();
    const std::vector<Real> crystal = t.lig.coords, moved = t.shifted(SHIFT[0], SHIFT[1], SHIFT[2]);

    for (bool angular : {true, false}){
        Real seen[8] = {};  // largest |column sum| over the poses
        for (int ipose = 0; ipose < 2; ipose++){
            const Real* coords = ipose == 0 ? crystal.data() : moved.data();
            INFO("angular = " << angular << " pose = " << (ipose == 0 ? "crystal" : "shifted"));
            SF.set_weights(rescore_set());
            Vina sf = SF;
            FlexPose p{};
            score(&p, coords, t.rec, lig, t.box, t.atp, angular, &t.h_topo, &t.rec.rec_donor_h, nullptr);
            std::vector<VinaTerms> at, rt;
            Real inter = 0;
            score_decomp(sf, coords, t.rec, lig, t.atp, angular, &t.h_topo, &t.rec.rec_donor_h, at, &rt, &inter);

            REQUIRE(at.size() == (size_t)lig.natom);
            REQUIRE(rt.size() == t.rec.residues.size());
            // The running sum follows score()'s order and expressions
            INFO("inter = " << inter << " center[1] = " << p.center[1]);
            REQUIRE(same_up_to_ulp(inter, p.center[1]));
            // Atom totals regroup the same per-pair values: float rounding only
            const Real sum_at = col_sum(at, &VinaTerms::total);
            INFO("center[1] = " << p.center[1] << " sum(atom total) = " << sum_at);
            REQUIRE(std::fabs(sum_at - p.center[1]) <= Real(1e-4) * std::max(Real(1), std::fabs(p.center[1])));

            for (int a = 0; a < lig.natom; a++){
                Real s = 0, s_abs = 0;
                for (auto f : TERM_COLS){ s += at[a].*f; s_abs += std::fabs(at[a].*f); }
                REQUIRE(std::fabs(at[a].total - s) <= Real(1e-5) * (1 + s_abs));
                if (t.atp.is_hydrogen[lig.vina_types[a]]){
                    for (auto f : TERM_COLS) REQUIRE(at[a].*f == Real(0));
                    REQUIRE(at[a].total == Real(0));
                }
            }
            for (auto f : TERM_COLS) REQUIRE(std::fabs(col_sum(rt, f) - col_sum(at, f)) <= Real(1e-4));
            REQUIRE(std::fabs(col_sum(rt, &VinaTerms::total) - sum_at) <= Real(1e-4));

            // Each residue against the ligand scored against that residue alone: pairwise terms
            // exactly, polar_env as the proportional share of the atom credit
            std::vector<VinaTerms> mag;
            const auto one_by_one = residues_one_by_one(sf, coords, t, lig, angular, at, mag);
            int n_res_contact = 0, n_res_polar = 0;
            for (size_t r = 0; r < rt.size(); r++){
                INFO("residue " << r);
                for (auto f : TERM_COLS){
                    REQUIRE(std::fabs(rt[r].*f - one_by_one[r].*f) <= Real(1e-5) * (1 + mag[r].*f));
                }
                REQUIRE(std::fabs(rt[r].total - one_by_one[r].total) <= Real(1e-5) * (1 + mag[r].total));
                if (std::fabs(one_by_one[r].total) > Real(1e-3)) n_res_contact++;
                if (std::fabs(one_by_one[r].polar_env) > Real(1e-4)) n_res_polar++;
            }
            // Several residues on both counts, so attributing to the wrong residue shows
            REQUIRE(n_res_contact >= 5);
            REQUIRE(n_res_polar >= 2);

            for (int k = 0; k < 8; k++) seen[k] = std::max(seen[k], std::fabs(col_sum(at, TERM_COLS[k])));
        }
        // Every term contributes in this pose set, so none of the checks above is vacuous
        for (int k = 0; k < 8; k++){
            INFO("angular = " << angular << " term column " << k);
            REQUIRE(seen[k] > Real(1e-3));
        }
    }
}


TEST_CASE("score_decomp: the angular weight changes only the hbond column", "[score_decomp]"){
    Tyk2 t;
    SFWeightsGuard guard;
    const UDFlexMol lig = t.scored_lig();
    const std::vector<Real> moved = t.shifted(SHIFT[0], SHIFT[1], SHIFT[2]);
    SF.set_weights(rescore_set());
    Vina sf = SF;

    FlexPose p_on{}, p_off{};
    std::vector<VinaTerms> on, off;
    score(&p_on, moved.data(), t.rec, lig, t.box, t.atp, true, &t.h_topo, &t.rec.rec_donor_h, nullptr);
    score(&p_off, moved.data(), t.rec, lig, t.box, t.atp, false, &t.h_topo, &t.rec.rec_donor_h, nullptr);
    score_decomp(sf, moved.data(), t.rec, lig, t.atp, true, &t.h_topo, &t.rec.rec_donor_h, on, nullptr, nullptr);
    score_decomp(sf, moved.data(), t.rec, lig, t.atp, false, &t.h_topo, &t.rec.rec_donor_h, off, nullptr, nullptr);

    int n_hb_diff = 0;
    for (int a = 0; a < lig.natom; a++){
        for (auto f : TERM_COLS){
            if (f == &VinaTerms::hbond) continue;
            REQUIRE(on[a].*f == off[a].*f);
        }
        if (on[a].hbond != off[a].hbond) n_hb_diff++;
    }
    REQUIRE(n_hb_diff > 0);
    const Real d_center = p_on.center[1] - p_off.center[1];
    const Real d_hbond = col_sum(on, &VinaTerms::hbond) - col_sum(off, &VinaTerms::hbond);
    INFO("center[1] on - off = " << d_center << " hbond column on - off = " << d_hbond);
    REQUIRE(std::fabs(d_center - d_hbond) <= Real(1e-4));
}


TEST_CASE("score_decomp: default weights leave the extra columns empty", "[score_decomp]"){
    Tyk2 t;
    SFWeightsGuard guard;
    const UDFlexMol lig = t.scored_lig();
    SF.set_weights(VinaWeights{});
    Vina sf = SF;
    FlexPose p{};
    score(&p, t.lig.coords.data(), t.rec, lig, t.box, t.atp, true, &t.h_topo, &t.rec.rec_donor_h, nullptr);
    std::vector<VinaTerms> at, rt;
    Real inter = 0;
    score_decomp(sf, t.lig.coords.data(), t.rec, lig, t.atp, true, &t.h_topo, &t.rec.rec_donor_h, at, &rt, &inter);
    REQUIRE(same_up_to_ulp(inter, p.center[1]));
    for (const auto* v : {&at, &rt}){
        for (const auto& x : *v){
            REQUIRE(x.elec == Real(0));
            REQUIRE(x.arom_contact == Real(0));
            REQUIRE(x.polar_env == Real(0));
        }
    }
}


TEST_CASE("score_decomp: a copy of the pose weights survives the affinity re-weighting", "[score_decomp]"){
    Tyk2 t;
    SFWeightsGuard guard;
    const UDFlexMol lig = t.scored_lig();
    SF.set_weights(rescore_set());
    Vina sf_pose = SF;  // what DockTask::run_score keeps for the decomposition
    FlexPose c_pose{}, c_aff{};
    score(&c_pose, t.lig.coords.data(), t.rec, lig, t.box, t.atp, true, &t.h_topo, &t.rec.rec_donor_h, nullptr);
    SF.set_weights(VinaWeights{});
    score(&c_aff, t.lig.coords.data(), t.rec, lig, t.box, t.atp, true, &t.h_topo, &t.rec.rec_donor_h, nullptr);
    REQUIRE(c_aff.center[1] != c_pose.center[1]);

    std::vector<VinaTerms> at;
    Real inter = 0;
    score_decomp(sf_pose, t.lig.coords.data(), t.rec, lig, t.atp, true, &t.h_topo, &t.rec.rec_donor_h, at, nullptr, &inter);
    REQUIRE(same_up_to_ulp(inter, c_pose.center[1]));
}


TEST_CASE("score_decomp: a receptor without an H table has no receptor-donor HBond", "[score_decomp]"){
    Tyk2 t;
    SFWeightsGuard guard;
    const UDFlexMol lig = t.scored_lig();
    const RecDonorHTopo no_h_table{};  // natom == 0: the receptor-donor direction has weight 0
    SF.set_weights(rescore_set());
    Vina sf = SF;

    int n_acc_hbond = 0;  // acceptor-only ligand atoms with an HBond when the table is there
    for (const auto& coords : {t.lig.coords, t.shifted(SHIFT[0], SHIFT[1], SHIFT[2])}){
        FlexPose p{};
        score(&p, coords.data(), t.rec, lig, t.box, t.atp, true, &t.h_topo, &no_h_table, nullptr);
        std::vector<VinaTerms> at, with_table;
        Real inter = 0;
        score_decomp(sf, coords.data(), t.rec, lig, t.atp, true, &t.h_topo, &no_h_table, at, nullptr, &inter);
        score_decomp(sf, coords.data(), t.rec, lig, t.atp, true, &t.h_topo, &t.rec.rec_donor_h, with_table, nullptr, nullptr);
        INFO("inter = " << inter << " center[1] = " << p.center[1]);
        REQUIRE(same_up_to_ulp(inter, p.center[1]));
        for (int a = 0; a < lig.natom; a++){
            const int ty = lig.vina_types[a];
            if (!t.atp.is_acceptor[ty] || t.atp.is_donor[ty]) continue;
            // An acceptor-only ligand atom hbonds only with receptor donors
            INFO("atom " << a << " hbond " << at[a].hbond << " with the H table " << with_table[a].hbond);
            REQUIRE(std::fabs(at[a].hbond) <= Real(1e-6));
            if (std::fabs(with_table[a].hbond) > Real(1e-3)) n_acc_hbond++;
        }
    }
    REQUIRE(n_acc_hbond > 0);
}


TEST_CASE("score_decomp: elec_charged_scale scales the elec column of charged-group atoms only", "[score_decomp]"){
    Tyk2 t;
    SFWeightsGuard guard;
    UDFlexMol lig = t.scored_lig();
    REQUIRE(lig.charges_ua.size() == (size_t)lig.natom);
    REQUIRE(lig.charged_group.size() == (size_t)lig.natom);
    // ejm42 is neutral: put its three most charged heavy atoms in a formally charged group
    std::vector<int> heavy;
    for (int a = 0; a < lig.natom; a++){
        if (!t.atp.is_hydrogen[lig.vina_types[a]]) heavy.push_back(a);
    }
    std::sort(heavy.begin(), heavy.end(), [&](int a, int b){ return std::fabs(lig.charges_ua[a]) > std::fabs(lig.charges_ua[b]); });
    for (int k = 0; k < 3; k++) lig.charged_group[heavy[k]] = 1;

    const std::vector<Real> moved = t.shifted(SHIFT[0], SHIFT[1], SHIFT[2]);
    VinaWeights w_scaled = rescore_set(), w_unit = rescore_set();
    w_unit.elec_charged_scale = 1;
    const Real scale = w_scaled.elec_charged_scale;

    SF.set_weights(w_scaled);
    Vina sf_scaled = SF;
    FlexPose p{};
    score(&p, moved.data(), t.rec, lig, t.box, t.atp, true, &t.h_topo, &t.rec.rec_donor_h, nullptr);
    std::vector<VinaTerms> scaled, unit;
    Real inter = 0;
    score_decomp(sf_scaled, moved.data(), t.rec, lig, t.atp, true, &t.h_topo, &t.rec.rec_donor_h, scaled, nullptr, &inter);
    REQUIRE(same_up_to_ulp(inter, p.center[1]));
    SF.set_weights(w_unit);
    Vina sf_unit = SF;
    score_decomp(sf_unit, moved.data(), t.rec, lig, t.atp, true, &t.h_topo, &t.rec.rec_donor_h, unit, nullptr, nullptr);

    int n_charged_elec = 0;
    for (int a = 0; a < lig.natom; a++){
        INFO("atom " << a << " charged " << int(lig.charged_group[a]) << " elec " << scaled[a].elec << " at scale 1 " << unit[a].elec);
        if (lig.charged_group[a]){
            // The ligand charge is scaled before the (linear) Coulomb pair energy
            REQUIRE(std::fabs(scaled[a].elec - scale * unit[a].elec) <= Real(1e-5) * (1 + std::fabs(unit[a].elec)));
            if (std::fabs(unit[a].elec) > Real(1e-2)) n_charged_elec++;
        } else {
            REQUIRE(scaled[a].elec == unit[a].elec);
        }
    }
    REQUIRE(n_charged_elec > 0);
}


TEST_CASE("DockTask --decomp output reconciles with vina_pose_score", "[score_decomp][docktask]"){
    Tyk2 t;
    SFWeightsGuard guard;

    SECTION("searched poses, affinity pass re-weighting SF") {
        DockParam dp;
        dp.box = t.box;
        dp.exhaustiveness = 8;
        dp.mc_steps = 5;
        dp.opt_steps = 10;
        dp.refine_steps = 5;
        dp.num_pose = 3;
        dp.seed = 1234;
        dp.decomp = true;
        dp.angular_hbond = true;
        dp.rescore_weights = rescore_set();
        dp.has_rescore_weights = true;
        dp.affinity_weights = VinaWeights{};
        dp.has_affinity_weights = true;

        const auto records = sdf_props(run_task(t, dp, {t.lig}));
        REQUIRE(!records.empty());
        // The affinity pass re-weights SF; the decomposition must still follow the pose pass
        REQUIRE(check_decomp_records(records, t) > 0);
    }

    SECTION("shifted input pose scored as given: HBond pairs off the full-weight angle") {
        // The angular weight moves this pose's inter energy well beyond the file tolerance, so a
        // decomposition without it cannot reconcile
        const UDFlexMol lig = t.scored_lig();
        const std::vector<Real> moved = t.shifted(SHIFT[0], SHIFT[1], SHIFT[2]);
        SF.set_weights(rescore_set());
        FlexPose p_on{}, p_off{};
        score(&p_on, moved.data(), t.rec, lig, t.box, t.atp, true, &t.h_topo, &t.rec.rec_donor_h, nullptr);
        score(&p_off, moved.data(), t.rec, lig, t.box, t.atp, false, &t.h_topo, &t.rec.rec_donor_h, nullptr);
        INFO("center[1] angular on " << p_on.center[1] << " off " << p_off.center[1]);
        REQUIRE(std::fabs(p_on.center[1] - p_off.center[1]) > 0.05);

        DockParam dp = score_only_param(t.box);
        dp.decomp = true;
        dp.rescore_weights = rescore_set();
        dp.has_rescore_weights = true;
        const auto records = sdf_props(run_task(t, dp, {t.shifted_lig()}));
        REQUIRE(records.size() == 1);
        check_decomp_records(records, t);
        REQUIRE(std::fabs(std::stod(records[0].at("vina_pose_inter")) - p_on.center[1]) <= 1e-3);
    }
}


TEST_CASE("DockTask without --decomp writes no pose-score parts or decomposition arrays", "[score_decomp][docktask]"){
    Tyk2 t;
    SFWeightsGuard guard;
    DockParam dp = score_only_param(t.box);
    dp.decomp = false;
    dp.rescore_weights = rescore_set();
    dp.has_rescore_weights = true;

    const auto records = sdf_props(run_task(t, dp, {t.shifted_lig()}));
    REQUIRE(records.size() == 1);
    REQUIRE(records[0].count("vina_pose_score") == 1);
    for (const auto& kv : records[0]){
        INFO("property " << kv.first);
        for (const char* owned : {"vina_pose_inter", "vina_pose_intra", "vina_pose_penalty", "atom_", "residue_",
                                  "decomp_unwritten_"}){
            REQUIRE(kv.first.rfind(owned, 0) != 0);
        }
    }
}


TEST_CASE("DockTask: ff_intra_ligand_energy includes the harmonic bond and angle energy", "[score_decomp][docktask][ff_bonded]"){
    Tyk2 t;
    SFWeightsGuard guard;
    DockParam dp = score_only_param(t.box);
    dp.decomp = true;
    dp.ligand_dms_path = TYK2_DIR + "ligand_ejm42_enriched.dms";
    dp.protein_dms_path = TYK2_DIR + "receptor_enriched.dms";

    const auto records = sdf_props(run_task(t, dp, {t.lig}));
    REQUIRE(records.size() == 1);
    const auto& r = records[0];
    for (const char* key : {"ff_total_energy", "ff_interaction_energy", "ff_intra_ligand_energy",
                            "atom_ff_bond", "atom_ff_angle"}){
        REQUIRE(r.count(key) == 1);
    }
    const double bonded = vsum(numbers(r.at("atom_ff_bond"))) + vsum(numbers(r.at("atom_ff_angle")));
    const double total = std::stod(r.at("ff_total_energy"));
    const double inter = std::stod(r.at("ff_interaction_energy"));
    const double intra = std::stod(r.at("ff_intra_ligand_energy"));
    INFO("ff total " << total << " inter " << inter << " intra " << intra << " bond + angle " << bonded);
    // A split without the harmonic terms would miss by this much
    REQUIRE(std::fabs(bonded) > 0.05);
    REQUIRE(std::fabs(total - inter - intra) <= 2e-3);
}


TEST_CASE("DockTask: FF minimisation leaves a ligand without a loaded FF alone", "[docktask][ff_populated]"){
    Tyk2 t;
    SFWeightsGuard guard;
    // Merged-DMS layout, two copies of ejm42: ligand 0 with its FF and the receptor FF; ligand 1
    // not loaded. In production an unloaded ligand of a merged batch comes from the merged
    // reader's per-ligand gate and carries no FF data; keeping the data here is the stricter case
    // (the skip must rest on the loaded flag alone)
    REQUIRE(t.ff_lig.loaded);
    FFMinData ff_on = t.ff_lig, ff_off = t.ff_lig;
    read_receptor_dms(TYK2_DIR + "receptor_enriched.dms", ff_on);
    REQUIRE(ff_on.loaded);
    ff_off.loaded = false;

    auto run = [&](int ff_steps){
        DockParam dp = score_only_param(t.box);
        dp.ff_minimize_steps = ff_steps;
        // Cartesian: the DMS pose is off the GAFF2 bond / angle minimum, so the loaded ligand moves
        dp.ff_minimize_mode = FFMIN_CARTESIAN;
        return sdf_records(run_task(t, dp, {t.lig, t.lig}, {ff_on, ff_off}));
    };
    const auto minimised = run(20), reference = run(0);
    REQUIRE(minimised.size() == 2);
    REQUIRE(reference.size() == 2);
    // Ligand 0 is minimised (so the run did minimise), ligand 1 is written as without minimisation
    REQUIRE(minimised[0] != reference[0]);
    REQUIRE(minimised[1] == reference[1]);
    REQUIRE(minimised[1].find("ff_total_energy") == std::string::npos);
}


// ==================== FF minimisation of the hydrogen-only torsions ====================

namespace {

/// Coordinates of the first natom atoms of an SDF record (4 header lines, then the V2000 atom block)
std::vector<double> record_coords(const std::string& record, int natom){
    std::istringstream in(record);
    std::string line;
    for (int k = 0; k < 4; k++) std::getline(in, line);
    std::vector<double> c;
    for (int a = 0; a < natom; a++){
        std::getline(in, line);
        std::istringstream ls(line);
        double x, y, z;
        ls >> x >> y >> z;
        c.insert(c.end(), {x, y, z});
    }
    return c;
}

/// Dihedral i-j-k-l in degrees
template <typename T>
double dihedral_deg(const std::vector<T>& c, int i, int j, int k, int l){
    double b[3][3];
    const int at[4] = {i, j, k, l};
    for (int m = 0; m < 3; m++){
        for (int x = 0; x < 3; x++) b[m][x] = c[at[m + 1] * 3 + x] - c[at[m] * 3 + x];
    }
    auto cross = [](const double* u, const double* v, double* w){
        w[0] = u[1] * v[2] - u[2] * v[1]; w[1] = u[2] * v[0] - u[0] * v[2]; w[2] = u[0] * v[1] - u[1] * v[0];
    };
    double n1[3], n2[3], m1[3];
    cross(b[0], b[1], n1);
    cross(b[1], b[2], n2);
    cross(n1, b[1], m1);
    const double len_b1 = std::sqrt(b[1][0] * b[1][0] + b[1][1] * b[1][1] + b[1][2] * b[1][2]);
    const double x = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
    const double y = (m1[0] * n2[0] + m1[1] * n2[1] + m1[2] * n2[2]) / len_b1;
    return std::atan2(y, x) * 180.0 / M_PI;
}

} // anonymous namespace


TEST_CASE("DockTask: FF minimisation turns the hydrogen-only torsions kept out of the search", "[docktask][ff_only_torsions]"){
    Tyk2 t;
    SFWeightsGuard guard;
    // The C19-C30 methyl rotor of ejm42 as the DMS reader keeps it out of the search, its hydrogens
    // turned 60 deg (eclipsed) about C19-C30 so that the FF has a torque on it
    UDFlexMol lig = t.lig;
    UDTorsion methyl;
    methyl.atoms[0] = 17; methyl.atoms[1] = 19; methyl.atoms[2] = 30; methyl.atoms[3] = 32;
    methyl.axis[0] = 19; methyl.axis[1] = 30;
    methyl.rotated_atoms = {30, 32, 33, 34};
    methyl.range_list = {static_cast<Real>(-M_PI), static_cast<Real>(M_PI)};
    double u[3], len = 0;
    for (int x = 0; x < 3; x++){
        u[x] = lig.coords[30 * 3 + x] - lig.coords[19 * 3 + x];
        len += u[x] * u[x];
    }
    for (double& x : u) x /= std::sqrt(len);
    const double cos_a = std::cos(M_PI / 3), sin_a = std::sin(M_PI / 3);
    for (int h : {32, 33, 34}){
        double v[3], w[3];
        for (int x = 0; x < 3; x++) v[x] = lig.coords[h * 3 + x] - lig.coords[30 * 3 + x];
        const double dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        w[0] = u[1] * v[2] - u[2] * v[1]; w[1] = u[2] * v[0] - u[0] * v[2]; w[2] = u[0] * v[1] - u[1] * v[0];
        for (int x = 0; x < 3; x++){
            lig.coords[h * 3 + x] = static_cast<Real>(lig.coords[30 * 3 + x] + v[x] * cos_a + w[x] * sin_a
                                                      + u[x] * dot * (1 - cos_a));
        }
    }
    lig.ff_only_torsions = {methyl};
    lig.ff_only_dihedrals = {static_cast<Real>(dihedral_deg(lig.coords, 17, 19, 30, 32) * M_PI / 180.0)};

    // Merged-DMS layout, ligand 1 without a loaded FF: its pose follows ligand 0's in memory. Ligand
    // FF only: with the receptor FF the internal-mode minimisation of this pose rolls back unmoved
    REQUIRE(t.ff_lig.loaded);
    FFMinData ff_on = t.ff_lig, ff_off = t.ff_lig;
    ff_off.loaded = false;
    auto run = [&](int ff_steps){
        DockParam dp = score_only_param(t.box);
        dp.ff_minimize_steps = ff_steps;
        dp.ff_minimize_mode = FFMIN_INTERNAL;
        return sdf_records(run_task(t, dp, {lig, lig}, {ff_on, ff_off}));
    };
    const auto minimised = run(20), reference = run(0);
    REQUIRE(minimised.size() == 2);
    REQUIRE(reference.size() == 2);

    // The FF topology turns the methyl; the search topology has no torsion for it
    const double before = dihedral_deg(record_coords(reference[0], lig.natom), 17, 19, 30, 32);
    const double after = dihedral_deg(record_coords(minimised[0], lig.natom), 17, 19, 30, 32);
    double turn = after - before;
    turn -= 360.0 * std::round(turn / 360.0);
    INFO("H32 dihedral " << before << " -> " << after);
    REQUIRE(std::fabs(turn) > 10.0);
    // The pose holds no slot for the FF-only dihedral: minimising without the view would write it
    // into ligand 1's coordinates
    REQUIRE(minimised[1] == reference[1]);

    // The affinity counts every DMS torsion, the FF-only one included
    for (const auto& r : sdf_props(reference[0] + reference[1])){
        const double affinity = std::stod(r.at("vina_binding_free_energy"));
        const double e_inter = affinity - std::stod(r.at("vina_torsion_number_energy"));
        INFO("affinity " << affinity << " inter " << e_inter);
        REQUIRE(std::fabs((e_inter / affinity - 1) / 0.05846 - 5) < 0.05);
    }
}


// ==================== aromatic-ring bias rows through DockTask ====================

namespace {

/// Distance from the centroid of atoms 0-5 to target
double ring_centroid_dist(const std::vector<double>& c, const Real* target){
    double d2 = 0;
    for (int x = 0; x < 3; x++){
        double m = 0;
        for (int a = 0; a < 6; a++) m += c[a * 3 + x];
        d2 += (m / 6 - target[x]) * (m / 6 - target[x]);
    }
    return std::sqrt(d2);
}

} // anonymous namespace


TEST_CASE("DockTask: ring-bias rows reach the search and the SDF of their own ligand only", "[docktask][ring_bias]"){
    Tyk2 t;
    SFWeightsGuard guard;
    // Merged-DMS layout, three copies of ejm42 at the crystal pose. Ligand 0 has no ring rows; ligands 1 and 2
    // have one row each on the benzene ring (atoms 0-5, ring order), the target centroid 0.8 A off the crystal
    // one along +x and -x, the target normal the crystal one. A refinement from the input pose (no MC) pulls
    // each ring toward its own target; a backend that drops the rows, or hands a ligand another one's rows,
    // cannot. The CUDA build runs the same case through the pooled device arena.
    RingBiasRow row;
    row.n = 6;
    row.v_set = 4;
    for (int k = 0; k < 6; k++){
        row.atoms[k] = k;
        for (int x = 0; x < 3; x++) row.c[x] += t.lig.coords[k * 3 + x] / 6;
    }
    Real nn[3] = {0, 0, 0};
    for (int k = 0; k < 6; k++){
        const Real* p = &t.lig.coords[k * 3];
        const Real* q = &t.lig.coords[((k + 1) % 6) * 3];
        const Real a1[3] = {p[0] - row.c[0], p[1] - row.c[1], p[2] - row.c[2]};
        const Real a2[3] = {q[0] - row.c[0], q[1] - row.c[1], q[2] - row.c[2]};
        Real cr[3];
        cross_product(a1, a2, cr);
        for (int x = 0; x < 3; x++) nn[x] += cr[x];
    }
    const Real len = cal_norm(nn);
    for (int x = 0; x < 3; x++) row.nt[x] = nn[x] / len;
    RingBiasRow plus = row, minus = row;
    plus.c[0] += 0.8;
    minus.c[0] -= 0.8;
    UDFlexMol lig_plus = t.lig, lig_minus = t.lig;
    lig_plus.ring.rows = {plus};
    lig_minus.ring.rows = {minus};

    DockParam dp = score_only_param(t.box);
    dp.refine_steps = 30;
    set_bias(dp, BT_ALIGN, dp.bias_k);   // with the policy the October bias_type test implied (template hydrogens kept)
    const auto with_rows = sdf_records(run_task(t, dp, {t.lig, lig_plus, lig_minus}));
    const auto without = sdf_records(run_task(t, dp, {t.lig, t.lig, t.lig}));
    REQUIRE(with_rows.size() == 3);
    REQUIRE(without.size() == 3);

    // The ligand without rows is written as in a batch without any, and only ligands with rows report them
    REQUIRE(with_rows[0] == without[0]);
    for (const auto& rec : without) REQUIRE(rec.find("ud2_ring_bias_raw") == std::string::npos);
    REQUIRE(with_rows[0].find("ud2_ring_bias_raw") == std::string::npos);
    // A biased run writes the pose pass's bias energy on every record (plan step 7); without rows it is 0
    for (const auto& rec : without) REQUIRE(std::stod(sdf_props(rec)[0].at("ud2_bias_energy")) == 0.0);

    for (int l : {1, 2}){
        const RingBiasRow& r = l == 1 ? plus : minus;
        const auto c_with = record_coords(with_rows[l], t.lig.natom);
        const double d_with = ring_centroid_dist(c_with, r.c);
        const double d_without = ring_centroid_dist(record_coords(without[l], t.lig.natom), r.c);
        const auto props = sdf_props(with_rows[l]);
        REQUIRE(props.size() == 1);
        REQUIRE(props[0].count("ud2_ring_bias_raw") == 1);
        const double raw = std::stod(props[0].at("ud2_ring_bias_raw"));
        const std::vector<Real> xyz(c_with.begin(), c_with.end());
        RingBiasFrame fr;
        const double e = ring_bias_frame(xyz.data(), r, dp.bias_k, dp.ring_bias, fr);
        INFO("ligand " << l << ": ring centroid to its target " << d_without << " A without rows, " << d_with
             << " A with; ud2_ring_bias_raw " << raw << ", ring_bias_frame of the written pose " << e);
        REQUIRE(d_with < 0.5 * d_without);
        REQUIRE(std::fabs(raw - e) <= 2e-3 + 1e-3 * std::fabs(e));
        // no atom rows: the bias energy of the pose pass (atom + ring) is the ring term alone
        REQUIRE(props[0].count("ud2_bias_energy") == 1);
        INFO("ud2_bias_energy " << props[0].at("ud2_bias_energy"));
        REQUIRE(std::fabs(std::stod(props[0].at("ud2_bias_energy")) - raw) <= 2e-3);
    }
}


// ==================== bias-aware pose pass (plan step 7) ====================

namespace {

/// Gaussian position-bias energy (BT_POS) of rows on the coordinates c, double precision (ext/bias/bias_term.h eval_pos)
double pos_bias_closed_form(const std::vector<Bias>& rows, const std::vector<double>& c, double bias_k){
    double e = 0;
    for (const auto& b : rows){
        double rr = 0;
        for (int x = 0; x < 3; x++) rr += (double(b.param[x]) - c[b.i * 3 + x]) * (double(b.param[x]) - c[b.i * 3 + x]);
        e += double(b.param[3]) * bias_k * std::exp(-rr / double(b.param[4]));
    }
    return e;
}

/// Z-align bias energy (BT_ALIGN) of rows on heavy atoms at the coordinates c, double precision (ext/bias/bias_term.h
/// eval_zalign: s_a 3, c_a 1, k_a 30)
double zalign_bias_closed_form(const std::vector<Bias>& rows, const std::vector<double>& c, double bias_k){
    double e = 0;
    for (const auto& b : rows){
        double rr = 0;
        for (int x = 0; x < 3; x++) rr += (double(b.param[x]) - c[b.i * 3 + x]) * (double(b.param[x]) - c[b.i * 3 + x]);
        e += double(b.param[3]) * bias_k * 30.0 / (1.0 + std::exp(3.0 * (1.0 - std::sqrt(rr))));
    }
    return e;
}

/// What the 4-decimal rounding of the written coordinates (5e-5 A per axis) can move a closed form by, with a factor 2:
/// for each biased atom the largest change over the corners of its rounding cube (a row's energy is monotone in the
/// atom's distance r to the row, so the farthest point is a corner; at the target pose r ~ 0 sits at the tip of the
/// Z-align cone, where a central difference would see no slope)
double rounding_slack(double (*closed_form)(const std::vector<Bias>&, const std::vector<double>&, double),
                      const std::vector<Bias>& rows, const std::vector<double>& c, double bias_k){
    const double d = 5e-5;
    const double e0 = closed_form(rows, c, bias_k);
    std::vector<int> atoms;
    for (const auto& b : rows) if (std::find(atoms.begin(), atoms.end(), b.i) == atoms.end()) atoms.push_back(b.i);
    double slack = 0;
    for (int a : atoms){
        double worst = 0;
        for (int corner = 0; corner < 8; corner++){
            std::vector<double> x = c;
            for (int k = 0; k < 3; k++) x[a * 3 + k] += (corner >> k & 1) ? d : -d;
            worst = std::max(worst, std::fabs(closed_form(rows, x, bias_k) - e0));
        }
        slack += worst;
    }
    return 2 * slack;
}

} // anonymous namespace


TEST_CASE("DockTask: a biased run's pose pass orders by vina_pose_score + bias energy, the scores stay unbiased",
          "[docktask][bias][pose_pass]"){
    Tyk2 t;
    SFWeightsGuard guard;
    // Four randomised starts scored as placed (no MC step, no refinement), so a bias moves no atom and every arm
    // has the same four candidates; rmsd_limit, energy_range and rescore_pool keep all four in the pose pass.
    DockParam dp = score_only_param(t.box);
    set_randomize(dp, true);
    dp.exhaustiveness = 4;
    dp.num_pose = 4;
    dp.rescore_pool = 10;
    dp.rmsd_limit = 0.01;
    dp.energy_range = 1e9;
    const int natom = t.lig.natom;

    const std::string plain_sdf = run_task(t, dp, {t.lig});
    const auto plain = sdf_props(plain_sdf);
    const auto plain_rec = sdf_records(plain_sdf);
    REQUIRE(plain.size() == 4);
    std::vector<double> score(4);
    std::vector<std::vector<double>> xyz(4);
    for (int m = 0; m < 4; m++){
        REQUIRE(plain[m].count("ud2_bias_energy") == 0);   // unbiased run: no tag
        score[m] = std::stod(plain[m].at("vina_pose_score"));
        xyz[m] = record_coords(plain_rec[m], natom);
        if (m > 0) REQUIRE(score[m] >= score[m - 1]);
    }
    auto candidate_of = [&](const std::vector<double>& c){
        for (int n = 0; n < 4; n++) if (c == xyz[n]) return n;
        return -1;
    };

    // One narrow Gaussian row (width 0.25 A^2) on atom 0 at a worse-scoring pose's atom 0, the best-scoring such pose
    // whose atom 0 is more than 2 A from every other pose's (so the row reaches no other candidate), 10 kcal/mol
    // deeper than its score gap to the first pose
    int target = -1;
    for (int n = 1; n < 4 && target < 0; n++){
        bool apart = true;
        for (int o = 0; o < 4; o++){
            if (o == n) continue;
            double d2 = 0;
            for (int x = 0; x < 3; x++) d2 += (xyz[o][x] - xyz[n][x]) * (xyz[o][x] - xyz[n][x]);
            apart = apart && d2 > 4.0;
        }
        if (apart) target = n;
    }
    REQUIRE(target > 0);
    const double gap = score[target] - score[0];
    Bias row;
    row.i = 0;
    for (int x = 0; x < 3; x++) row.param[x] = float(xyz[target][x]);
    row.param[4] = 0.25f;
    row.param[3] = float(-(gap + 10.0) / double(dp.bias_k));
    UDFlexMol lig_row = t.lig;
    lig_row.bias.rows = {row};

    // The Z-align row (BT_ALIGN, the type every ud2align facade case runs): a_qt k_a / (1 + exp(3 (1 - r))) with
    // k_a 30 on a heavy atom, a well of depth 0.905 a_qt k_a between r = 0 and r = 2 A, the same margin as above
    REQUIRE(t.lig.vina_types[0] != VN_TYPE_H);
    Bias row_align = row;
    row_align.param[3] = float((gap + 10.0) / (0.9 * 30.0 * double(dp.bias_k)));
    UDFlexMol lig_align = t.lig;
    lig_align.bias.rows = {row_align};

    // The affinity trio of a run's records: every affinity is referenced to the intra of the run's first written
    // record (the top pose), affinity - vina_torsion_number_energy = vina_intra_inter - that intra (for the top pose
    // its own vina_inter + vina_box_penalty), and affinity = that inter / (1 + w N_rot), the scale measured on the
    // unbiased run (the same ligand and weight set). The tags carry 3 decimals of float values (the CUDA build's
    // clashing random starts reach ~1e6, hence the relative slack).
    auto affinity_tags = [](const std::string& tag){
        return tag == "vina_affinity_score" || tag == "vina_binding_free_energy" || tag == "vina_torsion_number_energy";
    };
    auto tag_of = [](const Record& r, const char* tag){ return std::stod(r.at(tag)); };
    auto e_inter_of = [&](const Record& r){
        return tag_of(r, "vina_binding_free_energy") - tag_of(r, "vina_torsion_number_energy");
    };
    int scale_from = 0;                     // the unbiased record with the largest inter, for the conf-indep scale
    for (int m = 1; m < 4; m++){
        if (std::fabs(e_inter_of(plain[m])) > std::fabs(e_inter_of(plain[scale_from]))) scale_from = m;
    }
    const double e_scale = e_inter_of(plain[scale_from]);
    REQUIRE(std::fabs(e_scale) > 1.0);
    const double conf_indep_scale = tag_of(plain[scale_from], "vina_binding_free_energy") / e_scale;
    auto check_affinity_reference = [&](const std::vector<Record>& recs){
        const double intra_top = tag_of(recs[0], "vina_intra");
        for (size_t m = 0; m < recs.size(); m++){
            const double total = tag_of(recs[m], "vina_intra_inter");
            const double e_inter = e_inter_of(recs[m]);
            const double affinity = tag_of(recs[m], "vina_binding_free_energy");
            const double fp = 1e-6 * (std::fabs(total) + std::fabs(intra_top));
            INFO("record " << m << ": affinity " << affinity << ", inter " << e_inter << ", vina_intra_inter " << total
                 << ", the top pose's intra " << intra_top << ", conf-indep scale " << conf_indep_scale);
            REQUIRE(recs[m].at("vina_affinity_score") == recs[m].at("vina_binding_free_energy"));
            REQUIRE(std::fabs(e_inter - (total - intra_top)) <= 3e-3 + fp);
            REQUIRE(std::fabs(affinity - conf_indep_scale * e_inter)
                    <= 3e-3 + std::fabs(e_inter) * (3e-3 / std::fabs(e_scale) + 1e-6));
        }
    };
    check_affinity_reference(plain);        // the unbiased run: referenced to its own first pose, as before

    // A biased run of the same four candidates, num_pose of them written: the target pose is written first, the key
    // vina_pose_score + bias energy is non-decreasing, ud2_bias_energy is the closed form, every affinity is
    // referenced to the intra of the written top pose (the target), as an unbiased run references its own first
    // pose, and every other tag but ud2_pose_index is the one the unbiased num_pose 4 run
    // writes for the same pose (vina_pose_score and every energy stay unbiased). The target's intra differs from the
    // unbiased first pose's, so an affinity referenced to the unbiased order fails. With num_pose < 4 the pool
    // (rescore_pool 10) still holds all four, and the one written pose's affinity uses its own intra, not that of
    // the unbiased first pose of the pool (plain pose 0, not written). These rows also put the target at the head of
    // the pool, so the order of the sort and the num_pose cut, and the affinity reference taken after the sort (not
    // from the pool's head), are checked in their own section below
    REQUIRE(std::fabs(tag_of(plain[target], "vina_intra") - tag_of(plain[0], "vina_intra")) > 0.05);
    auto check_biased = [&](BiasType bt, const UDFlexMol& lig, double (*closed_form)(const std::vector<Bias>&,
                                                                                   const std::vector<double>&, double),
                            int num_pose){
        DockParam dpb = dp;
        set_bias(dpb, bt, dp.bias_k);
        dpb.num_pose = num_pose;
        REQUIRE(dpb.rescore_pool >= 4);
        const std::string biased_sdf = run_task(t, dpb, {lig});
        const auto biased = sdf_props(biased_sdf);
        const auto biased_rec = sdf_records(biased_sdf);
        const int n_out = std::min(num_pose, 4);
        REQUIRE(int(biased.size()) == n_out);
        std::vector<double> key(n_out);
        std::vector<int> from(n_out, -1);
        for (int m = 0; m < n_out; m++){
            const auto c = record_coords(biased_rec[m], natom);
            from[m] = candidate_of(c);
            REQUIRE(from[m] >= 0);                                  // the same four candidates
            REQUIRE(biased[m].count("ud2_bias_energy") == 1);
            const double e_bias = std::stod(biased[m].at("ud2_bias_energy"));
            const double e_ref = closed_form(lig.bias.rows, c, dpb.bias_k);
            // the engine evaluates the unrounded coordinates: the CUDA build's random starts clash (scores ~1e6), so
            // the Z-align row is deep and its slope at r = 0 turns the coordinates' rounding into ~20 kcal/mol
            const double slack = rounding_slack(closed_form, lig.bias.rows, c, dpb.bias_k);
            INFO("bias type " << int(bt) << ", num_pose " << num_pose << ", pose " << m << " = unbiased pose " << from[m]
                 << ": vina_pose_score " << score[from[m]] << ", ud2_bias_energy " << e_bias << ", closed form " << e_ref
                 << ", rounding slack " << slack);
            REQUIRE(std::fabs(e_bias - e_ref) <= 1e-3 + 1e-5 * std::fabs(e_ref) + slack);
            REQUIRE(biased[m].size() == plain[from[m]].size() + 1);  // + ud2_bias_energy
            for (const auto& [tag, value] : plain[from[m]]){
                if (tag == "ud2_pose_index" || affinity_tags(tag)) continue;
                INFO("tag " << tag);
                REQUIRE(biased[m].count(tag) == 1);
                REQUIRE(biased[m].at(tag) == value);
            }
            key[m] = score[from[m]] + e_bias;
            if (m > 0) REQUIRE(key[m] >= key[m - 1] - 2e-3);       // the pose-pass key, at the tags' 3 decimals
        }
        REQUIRE(from[0] == target);                                 // the order flipped
        check_affinity_reference(biased);                           // referenced to the written top pose
    };

    SECTION("BT_POS: the biased pose is written first, affinities of the written top pose, the rest unbiased"){
        check_biased(BT_POS, lig_row, pos_bias_closed_form, 4);
    }

    SECTION("BT_ALIGN: the biased pose is written first, affinities of the written top pose, the rest unbiased"){
        check_biased(BT_ALIGN, lig_align, zalign_bias_closed_form, 4);
    }

    SECTION("BT_POS, num_pose 1 from a pool of 4: the biased pose is written, its affinity of its own intra"){
        check_biased(BT_POS, lig_row, pos_bias_closed_form, 1);
    }

    SECTION("BT_ALIGN, num_pose 1 from a pool of 4: the biased pose is written, its affinity of its own intra"){
        check_biased(BT_ALIGN, lig_align, zalign_bias_closed_form, 1);
    }

    SECTION("BT_POS, num_pose 1 from a pool of 4 headed by another pose: the whole pool is sorted before the cut"){
        // The row on the target at depth gap +- margin: for some margin the pool's head (the search energy, the bias
        // included; a pool of one, rescore_pool 1, writes it) and the pose pass's first pose (vina_pose_score + bias; the
        // num_pose 4 run sorts the whole pool and cuts nothing) differ, and then only a sort of the whole pool before
        // the num_pose cut writes the pose pass's first. The num_pose 1 run writes the num_pose 4 run's first record.
        // Both runs reference their affinities to the intra of that written first record, which is not the pool's
        // head, so a reference taken before the sort fails here.
        // Which margin does it depends on the build's random starts (Mac clang: +1, the target first; box gcc and CUDA:
        // -1, plain pose 0 first and the target at the pool's head).
        DockParam dpb = dp;
        set_bias(dpb, BT_POS, dp.bias_k);
        bool headed_by_another = false;
        for (double margin : {1.0, -1.0, 0.3, -0.3}){
            Bias r = row;
            r.param[3] = float(-(gap + margin) / double(dp.bias_k));
            UDFlexMol lig_m = t.lig;
            lig_m.bias.rows = {r};
            const std::string full_sdf = run_task(t, dpb, {lig_m});
            const auto full = sdf_records(full_sdf);
            REQUIRE(full.size() == 4);
            const int first = candidate_of(record_coords(full[0], natom));
            DockParam dp_head = dpb;
            dp_head.num_pose = 1;
            dp_head.rescore_pool = 1;
            const auto head = sdf_records(run_task(t, dp_head, {lig_m}));
            REQUIRE(head.size() == 1);
            const int head_pose = candidate_of(record_coords(head[0], natom));
            INFO("margin " << margin << ": the pose pass's first is unbiased pose " << first << ", the pool's head "
                 << head_pose);
            REQUIRE(first >= 0);
            REQUIRE(head_pose >= 0);
            if (first == head_pose) continue;
            DockParam dp_one = dpb;
            dp_one.num_pose = 1;
            const std::string one_sdf = run_task(t, dp_one, {lig_m});
            const auto one = sdf_records(one_sdf);
            REQUIRE(one.size() == 1);
            REQUIRE(one[0] == full[0]);
            check_affinity_reference(sdf_props(full_sdf));
            check_affinity_reference(sdf_props(one_sdf));
            headed_by_another = true;
            break;
        }
        REQUIRE(headed_by_another);
    }

    SECTION("bias type off, rows present: byte-identical to the run without rows"){
        REQUIRE(run_task(t, dp, {lig_row}) == plain_sdf);
        REQUIRE(run_task(t, dp, {lig_align}) == plain_sdf);
    }

    SECTION("constraint docking, bias type off, rows present: the unbiased order and no tag"){
        // static_root keeps the template hydrogens as a bias does (policy.keep_template_hydrogens), but the pose pass
        // keys on the bias type, so the rows change no byte
        DockParam dpc = dp;
        set_constraint_docking(dpc, true);
        REQUIRE(dpc.policy.keep_template_hydrogens);
        REQUIRE(dpc.bias_type == BT_NO);
        const std::string constrained_sdf = run_task(t, dpc, {t.lig});
        REQUIRE(run_task(t, dpc, {lig_row}) == constrained_sdf);
        const auto constrained = sdf_props(constrained_sdf);
        REQUIRE(!constrained.empty());
        for (size_t m = 0; m < constrained.size(); m++){
            REQUIRE(constrained[m].count("ud2_bias_energy") == 0);
            if (m > 0) REQUIRE(std::stod(constrained[m].at("vina_pose_score"))
                               >= std::stod(constrained[m - 1].at("vina_pose_score")));
        }
    }
}
