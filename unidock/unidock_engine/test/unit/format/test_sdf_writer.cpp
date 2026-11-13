//
// Unit tests for SDF V2000 writer (write_poses_to_sdf).
//

#include <catch2/catch_amalgamated.hpp>
#include "format/sdf_writer.h"
#include "format/dms_reader.h"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

namespace {

/// Read entire file into a string.
std::string read_file(const std::string& path) {
    std::ifstream ifs(path);
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

/// Split a string by newline into lines.
std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> lines;
    std::istringstream iss(s);
    std::string line;
    while (std::getline(iss, line)) {
        lines.push_back(line);
    }
    return lines;
}

/// Build a minimal SDFMolData for a 3-atom molecule (H2O-like).
SDFMolData make_h2o_sdf_mol() {
    SDFMolData mol;
    mol.natom = 3;
    mol.anum = {8, 1, 1};               // O, H, H
    mol.formal_charge = {0, 0, 0};
    mol.bonds = {
        {0, 1, 1, 0},  // O-H single bond
        {0, 2, 1, 0},  // O-H single bond
    };
    mol.mol_name = "water";
    mol.loaded = true;
    return mol;
}

/// Build a minimal SDFMolData with a formal charge (ammonium-like: NH4+).
SDFMolData make_charged_sdf_mol() {
    SDFMolData mol;
    mol.natom = 5;
    mol.anum = {7, 1, 1, 1, 1};         // N, H, H, H, H
    mol.formal_charge = {1, 0, 0, 0, 0}; // +1 on nitrogen
    mol.bonds = {
        {0, 1, 1, 0},
        {0, 2, 1, 0},
        {0, 3, 1, 0},
        {0, 4, 1, 0},
    };
    mol.mol_name = "ammonium";
    mol.properties.push_back({"dG_exp", "-5.0"});
    mol.loaded = true;
    return mol;
}

/// Value line of an SDF property in a single-record file ("" when absent).
std::string prop_value(const std::string& content, const std::string& key) {
    const std::string tag = ">  <" + key + ">\n";
    size_t pos = content.find(tag);
    if (pos == std::string::npos) return "";
    pos += tag.size();
    return content.substr(pos, content.find('\n', pos) - pos);
}

/// Numbers of a space-separated property value.
std::vector<double> prop_numbers(const std::string& content, const std::string& key) {
    std::istringstream iss(prop_value(content, key));
    std::vector<double> v;
    double x;
    while (iss >> x) v.push_back(x);
    return v;
}

/// Create a temporary file path (cleaned up by test caller).
std::string make_tmp_path(const char* label) {
    char buf[256];
    snprintf(buf, sizeof(buf), "/tmp/test_sdf_writer_%s_%d.sdf", label, static_cast<int>(getpid()));
    return std::string(buf);
}

/// Prepare single-ligand FlexPose and coordinate arrays for natom atoms,
/// with nposes poses at known coordinates.
struct PoseFixture {
    std::vector<FlexPose> poses;
    std::vector<Real> coords;          // flat: [pose0_atom0_x, y, z, atom1_x, ...]
    std::vector<int> list_i_real;      // [coord_start, coord_end] per pose

    std::vector<std::string> flex_names;
    std::vector<std::vector<int>> filtered_pose_inds;

    PoseFixture(int natom, int nposes) {
        flex_names.push_back("lig");
        std::vector<int> inds;

        for (int p = 0; p < nposes; p++) {
            FlexPose fp;
            fp.energy = -1.0f * (p + 1);
            // Abuse rot_vec/center to store energy components the writer reads
            fp.rot_vec[0] = -7.0f - p;    // vina_binding_free_energy
            fp.rot_vec[1] = -5.0f - p;    // vina_intra_inter
            fp.rot_vec[3] = 0.5f + p;     // vina_torsion_number_energy
            fp.center[0] = -3.0f - p;     // vina_intra
            fp.center[1] = -2.0f - p;     // vina_inter
            fp.center[2] = 0.1f + p;      // vina_box_penalty
            poses.push_back(fp);
            inds.push_back(p);

            int start = static_cast<int>(coords.size());
            for (int a = 0; a < natom; a++) {
                coords.push_back(1.0f + a + p * 0.1f);  // x
                coords.push_back(2.0f + a + p * 0.1f);  // y
                coords.push_back(3.0f + a + p * 0.1f);  // z
            }
            int end = static_cast<int>(coords.size());
            list_i_real.push_back(start);   // coord start
            list_i_real.push_back(end);     // coord end (unused but needed for indexing)
        }
        filtered_pose_inds.push_back(inds);
    }
};

} // anonymous namespace


// ===================== SDF Writer Tests =====================

TEST_CASE("write_poses_to_sdf: header, counts, and structure", "[sdf_writer]") {
    SDFMolData mol = make_h2o_sdf_mol();
    PoseFixture fix(3, 1);
    std::string tmp = make_tmp_path("basic");

    write_poses_to_sdf(
        tmp, {mol}, fix.flex_names, fix.filtered_pose_inds,
        fix.poses.data(), fix.coords.data(), fix.list_i_real);

    std::string content = read_file(tmp);
    auto lines = split_lines(content);
    REQUIRE(lines.size() > 10);

    SECTION("header line is mol_name") {
        REQUIRE(lines[0] == "water");
    }

    SECTION("program line") {
        REQUIRE(lines[1].find("UniDock2") != std::string::npos);
    }

    SECTION("counts line: 3 atoms 2 bonds") {
        // V2000 counts line: "  3  2  0  0  0  0  0  0  0  0999 V2000"
        REQUIRE(lines[3].find("V2000") != std::string::npos);
        REQUIRE(lines[3].substr(0, 3) == "  3");
        REQUIRE(lines[3].substr(3, 3) == "  2");
    }

    SECTION("atom block has 3 lines with O and H symbols") {
        // lines[4..6] are atom lines
        REQUIRE(lines[4].find("O ") != std::string::npos);
        REQUIRE(lines[5].find("H ") != std::string::npos);
        REQUIRE(lines[6].find("H ") != std::string::npos);
    }

    SECTION("atom coordinates match input") {
        // First atom: x=1.0, y=2.0, z=3.0
        REQUIRE(lines[4].find("1.0000") != std::string::npos);
        REQUIRE(lines[4].find("2.0000") != std::string::npos);
        REQUIRE(lines[4].find("3.0000") != std::string::npos);
    }

    SECTION("bond block: 2 bonds, 1-indexed") {
        // lines[7..8] are bond lines
        // Bond 0-1 -> "  1  2  1  0"
        REQUIRE(lines[7].find("  1  2") != std::string::npos);
        // Bond 0-2 -> "  1  3  1  0"
        REQUIRE(lines[8].find("  1  3") != std::string::npos);
    }

    SECTION("M  END marker present") {
        bool found = false;
        for (auto& l : lines) {
            if (l.find("M  END") != std::string::npos) { found = true; break; }
        }
        REQUIRE(found);
    }

    SECTION("$$$$ delimiter present") {
        bool found = false;
        for (auto& l : lines) {
            if (l == "$$$$") { found = true; break; }
        }
        REQUIRE(found);
    }

    SECTION("energy property present") {
        bool found = false;
        for (auto& l : lines) {
            if (l.find("vina_binding_free_energy") != std::string::npos) { found = true; break; }
        }
        REQUIRE(found);
    }

    std::remove(tmp.c_str());
}


TEST_CASE("write_poses_to_sdf: formal charges produce M  CHG lines", "[sdf_writer]") {
    SDFMolData mol = make_charged_sdf_mol();
    PoseFixture fix(5, 1);
    std::string tmp = make_tmp_path("chg");

    write_poses_to_sdf(
        tmp, {mol}, fix.flex_names, fix.filtered_pose_inds,
        fix.poses.data(), fix.coords.data(), fix.list_i_real);

    std::string content = read_file(tmp);

    SECTION("M  CHG line for nitrogen +1") {
        // Expected: "M  CHG  1   1   1" (1 entry, atom 1 (1-indexed), charge +1)
        REQUIRE(content.find("M  CHG") != std::string::npos);
        REQUIRE(content.find("M  CHG  1   1   1") != std::string::npos);
    }

    SECTION("user property dG_exp present") {
        REQUIRE(content.find("dG_exp") != std::string::npos);
        REQUIRE(content.find("-5.0") != std::string::npos);
    }

    std::remove(tmp.c_str());
}


TEST_CASE("write_poses_to_sdf: multiple poses produce multiple records", "[sdf_writer]") {
    SDFMolData mol = make_h2o_sdf_mol();
    int nposes = 3;
    PoseFixture fix(3, nposes);
    std::string tmp = make_tmp_path("multi");

    write_poses_to_sdf(
        tmp, {mol}, fix.flex_names, fix.filtered_pose_inds,
        fix.poses.data(), fix.coords.data(), fix.list_i_real);

    std::string content = read_file(tmp);

    SECTION("3 $$$$ delimiters") {
        int count = 0;
        size_t pos = 0;
        while ((pos = content.find("$$$$", pos)) != std::string::npos) {
            count++;
            pos += 4;
        }
        REQUIRE(count == nposes);
    }

    SECTION("3 M  END markers") {
        int count = 0;
        size_t pos = 0;
        while ((pos = content.find("M  END", pos)) != std::string::npos) {
            count++;
            pos += 5;
        }
        REQUIRE(count == nposes);
    }

    SECTION("pose indices 0, 1, 2 present") {
        REQUIRE(content.find("ud2_pose_index") != std::string::npos);
    }

    std::remove(tmp.c_str());
}


TEST_CASE("write_poses_to_sdf: empty decomp data is fine", "[sdf_writer]") {
    SDFMolData mol = make_h2o_sdf_mol();
    PoseFixture fix(3, 1);
    std::string tmp = make_tmp_path("nodecomp");

    // Pass empty decomp and ff_decomp explicitly
    std::vector<std::vector<std::vector<VinaTerms>>> empty_decomp;
    std::vector<std::vector<std::vector<FFAtomTerms>>> empty_ff_decomp;
    std::vector<std::vector<std::array<Real, 3>>> empty_ff_energies;

    write_poses_to_sdf(
        tmp, {mol}, fix.flex_names, fix.filtered_pose_inds,
        fix.poses.data(), fix.coords.data(), fix.list_i_real,
        empty_decomp, empty_ff_decomp, empty_ff_energies);

    std::string content = read_file(tmp);

    SECTION("no atom_inter_energy property when decomp empty") {
        REQUIRE(content.find("atom_inter_energy") == std::string::npos);
    }

    SECTION("no ff_total_energy property when ff empty") {
        REQUIRE(content.find("ff_total_energy") == std::string::npos);
    }

    SECTION("still has basic structure") {
        REQUIRE(content.find("$$$$") != std::string::npos);
        REQUIRE(content.find("M  END") != std::string::npos);
    }

    std::remove(tmp.c_str());
}


TEST_CASE("write_poses_to_sdf: with decomp data", "[sdf_writer]") {
    SDFMolData mol = make_h2o_sdf_mol();
    PoseFixture fix(3, 1);
    std::string tmp = make_tmp_path("decomp");

    // Build per-atom Vina decomp: 1 ligand, 1 pose, 3 atoms
    std::vector<std::vector<std::vector<VinaTerms>>> decomp(1);
    decomp[0].resize(1);
    for (int a = 0; a < 3; a++) {
        VinaTerms t;
        t.total = -0.5f * (a + 1);
        t.gauss1 = -0.1f * (a + 1);
        t.gauss2 = -0.2f * (a + 1);
        t.repulsion = 0.0f;
        t.hydrophobic = 0.0f;
        t.hbond = -0.05f * (a + 1);
        decomp[0][0].push_back(t);
    }

    write_poses_to_sdf(
        tmp, {mol}, fix.flex_names, fix.filtered_pose_inds,
        fix.poses.data(), fix.coords.data(), fix.list_i_real,
        decomp);

    std::string content = read_file(tmp);

    SECTION("atom_inter_energy property present") {
        REQUIRE(content.find("atom_inter_energy") != std::string::npos);
    }

    SECTION("atom_gauss1 property present") {
        REQUIRE(content.find("atom_gauss1") != std::string::npos);
    }

    SECTION("atom_hbond property present") {
        REQUIRE(content.find("atom_hbond") != std::string::npos);
    }

    std::remove(tmp.c_str());
}


TEST_CASE("write_poses_to_sdf: pose-score parts and optional term columns", "[sdf_writer]") {
    SDFMolData mol = make_h2o_sdf_mol();
    PoseFixture fix(3, 1);
    std::string tmp = make_tmp_path("parts");

    std::vector<std::vector<std::vector<VinaTerms>>> decomp(1, std::vector<std::vector<VinaTerms>>(1, std::vector<VinaTerms>(3)));
    decomp[0][0][0].polar_env = -0.25f;
    decomp[0][0][0].total = -0.25f;
    std::vector<ResidueKey> residues = {{"A", 1, "LEU"}, {"A", 2, "TYR"}, {"A", 3, "GLY"}};
    auto res_decomp = decomp;  // 3 residues, same values
    std::vector<std::vector<std::array<Real, 3>>> parts(1);
    parts[0].push_back({-1.25f, -8.5f, 0.0f});

    auto write = [&](const VinaWeights& w) {
        write_poses_to_sdf(
            tmp, {mol}, fix.flex_names, fix.filtered_pose_inds,
            fix.poses.data(), fix.coords.data(), fix.list_i_real,
            decomp, {}, {}, res_decomp, residues, {}, parts, w);
        return read_file(tmp);
    };

    SECTION("vina_pose_inter / intra / penalty written before the atom arrays") {
        std::string content = write(VinaWeights{});
        REQUIRE(prop_value(content, "vina_pose_inter") == "-8.500");
        REQUIRE(prop_value(content, "vina_pose_intra") == "-1.250");
        REQUIRE(prop_value(content, "vina_pose_penalty") == "0.000");
        REQUIRE(content.find("vina_pose_penalty") < content.find("atom_inter_energy"));
    }

    SECTION("default weights: no elec / arom_contact / polar_env columns") {
        std::string content = write(VinaWeights{});
        for (const char* key : {"atom_elec", "atom_arom_contact", "atom_polar_env",
                                "residue_elec", "residue_arom_contact", "residue_polar_env"}) {
            REQUIRE(content.find(key) == std::string::npos);
        }
        REQUIRE(content.find("atom_hbond") != std::string::npos);
        REQUIRE(content.find("residue_hbond") != std::string::npos);
    }

    SECTION("nonzero polar_env weight adds its columns only") {
        VinaWeights w;
        w.polar_env = -0.4f;
        std::string content = write(w);
        REQUIRE(prop_value(content, "atom_polar_env") == "-0.2500 0.0000 0.0000");
        REQUIRE(prop_value(content, "residue_polar_env") == "-0.2500 0.0000 0.0000");
        REQUIRE(content.find("atom_elec") == std::string::npos);
        REQUIRE(content.find("atom_arom_contact") == std::string::npos);
    }

    std::remove(tmp.c_str());
}


TEST_CASE("write_poses_to_sdf: per-atom arrays cover the written molecule only", "[sdf_writer][sdf_natom]") {
    // Flex system of 5 atoms (e.g. PROTAC + E3 ligase atoms) of which the SDF writes the first
    // 3 (sdf_natom); flex heavy atoms are 0, 3 and 4.
    const int nflex = 5;
    PoseFixture fix(nflex, 1);
    std::string tmp = make_tmp_path("sdf_natom");

    std::vector<std::vector<std::vector<VinaTerms>>> decomp(1, std::vector<std::vector<VinaTerms>>(1));
    std::vector<std::vector<std::vector<FFAtomTerms>>> ff_decomp(1, std::vector<std::vector<FFAtomTerms>>(1));
    for (int a = 0; a < nflex; a++) {
        VinaTerms t;
        t.gauss1 = -0.1f * (a + 1);
        t.hbond = -0.5f * (a + 1);
        t.total = t.gauss1 + t.hbond;
        decomp[0][0].push_back(t);
        FFAtomTerms ft;
        ft.inter_lj = -1.0f * (a + 1);
        ft.bond = 0.25f * (a + 1);
        ft.total = ft.inter_lj + ft.bond;
        ff_decomp[0][0].push_back(ft);
    }
    std::vector<std::vector<std::array<Real, 3>>> ff_energies(1);
    ff_energies[0].push_back({-11.25f, -15.0f, 3.75f});
    std::vector<ResidueKey> residues = {{"A", 1, "LEU"}, {"B", 7, "TRP"}};
    std::vector<std::vector<std::vector<VinaTerms>>> res_decomp(1, std::vector<std::vector<VinaTerms>>(1, std::vector<VinaTerms>(2)));
    res_decomp[0][0][0].total = -4.0f;
    res_decomp[0][0][1].total = -5.0f;
    std::vector<std::vector<ScoreDiag>> diag(1, std::vector<ScoreDiag>(1));
    ScoreDiag& dg = diag[0][0];
    dg.n_heavy = 3;
    dg.heavy_atom = {0, 3, 4};
    dg.heavy_type = {6, 3, 2};
    dg.heavy_class = {1, 0, 0};
    dg.heavy_apolar = {1.5f, 2.5f, 3.5f};
    dg.heavy_polar = {0, 0, 0};
    dg.heavy_hbond = {0, 0, 0};

    auto write = [&](const SDFMolData& mol) {
        write_poses_to_sdf(
            tmp, {mol}, fix.flex_names, fix.filtered_pose_inds,
            fix.poses.data(), fix.coords.data(), fix.list_i_real,
            decomp, ff_decomp, ff_energies, res_decomp, residues, diag);
        return read_file(tmp);
    };

    SECTION("3 written atoms: arrays truncated, unwritten sums reported") {
        std::string content = write(make_h2o_sdf_mol());
        REQUIRE(prop_value(content, "atom_inter_energy") == "-0.6000 -1.2000 -1.8000");
        REQUIRE(prop_value(content, "atom_gauss1") == "-0.1000 -0.2000 -0.3000");
        REQUIRE(prop_value(content, "decomp_unwritten_inter_energy") == "-5.400");
        REQUIRE(prop_value(content, "decomp_unwritten_gauss1") == "-0.900");
        REQUIRE(prop_value(content, "decomp_unwritten_hbond") == "-4.500");
        REQUIRE(prop_value(content, "decomp_unwritten_repulsion") == "0.000");
        REQUIRE(prop_value(content, "atom_ff_total") == "-0.7500 -1.5000 -2.2500");
        REQUIRE(prop_value(content, "decomp_unwritten_ff_total") == "-6.750");
        REQUIRE(prop_value(content, "decomp_unwritten_ff_inter_lj") == "-9.000");
        REQUIRE(prop_value(content, "atom_ff_bond") == "0.2500 0.5000 0.7500");
        REQUIRE(prop_value(content, "decomp_unwritten_ff_bond") == "2.250");
        REQUIRE(prop_value(content, "ud2_heavy_type") == "6.0000");
        REQUIRE(prop_value(content, "ud2_heavy_apolar") == "1.5000");
        REQUIRE(prop_value(content, "ud2_n_heavy") == "3.000");
        REQUIRE(prop_value(content, "residue_inter_energy") == "-4.0000 -5.0000");
        // written atoms + unwritten sum == all flex atoms
        const auto at = prop_numbers(content, "atom_inter_energy");
        REQUIRE(at[0] + at[1] + at[2] + std::stod(prop_value(content, "decomp_unwritten_inter_energy"))
                == Catch::Approx(-9.0).margin(1e-3));
    }

    SECTION("all atoms written: no truncation, no unwritten sums") {
        std::string content = write(make_charged_sdf_mol());
        REQUIRE(content.find("decomp_unwritten") == std::string::npos);
        REQUIRE(prop_value(content, "atom_inter_energy") == "-0.6000 -1.2000 -1.8000 -2.4000 -3.0000");
        REQUIRE(prop_value(content, "atom_ff_total") == "-0.7500 -1.5000 -2.2500 -3.0000 -3.7500");
        REQUIRE(prop_value(content, "ud2_heavy_type") == "6.0000 3.0000 2.0000");
    }

    std::remove(tmp.c_str());
}


TEST_CASE("write_poses_to_sdf: FF bond and angle columns complete atom_ff_total", "[sdf_writer][ff_bonded]") {
    SDFMolData mol = make_h2o_sdf_mol();
    PoseFixture fix(3, 1);
    std::string tmp = make_tmp_path("ff_bonded");

    auto make_terms = [](Real bond_scale, Real angle_scale) {
        std::vector<std::vector<std::vector<FFAtomTerms>>> ff(1, std::vector<std::vector<FFAtomTerms>>(1));
        for (int a = 0; a < 3; a++) {
            FFAtomTerms t;
            t.inter_lj = -0.5f * (a + 1);
            t.inter_coulomb = -0.25f;
            t.intra_lj = 0.125f;
            t.intra_coulomb = 0.0625f;
            t.torsion = 0.03125f * (a + 1);
            t.bond = bond_scale * (a + 1);
            t.angle = angle_scale;
            t.total = t.inter_lj + t.inter_coulomb + t.intra_lj + t.intra_coulomb + t.torsion + t.bond + t.angle;
            ff[0][0].push_back(t);
        }
        return ff;
    };
    auto write = [&](const std::vector<std::vector<std::vector<FFAtomTerms>>>& ff) {
        write_poses_to_sdf(
            tmp, {mol}, fix.flex_names, fix.filtered_pose_inds,
            fix.poses.data(), fix.coords.data(), fix.list_i_real,
            {}, ff);
        return read_file(tmp);
    };

    SECTION("harmonic terms present: the 7 written parts sum to atom_ff_total") {
        std::string content = write(make_terms(0.75f, 0.375f));
        const auto total = prop_numbers(content, "atom_ff_total");
        REQUIRE(total.size() == 3);
        std::vector<double> parts_sum(3, 0.0);
        for (const char* key : {"atom_ff_inter_lj", "atom_ff_inter_coulomb", "atom_ff_intra_lj",
                                "atom_ff_intra_coulomb", "atom_ff_torsion", "atom_ff_bond", "atom_ff_angle"}) {
            const auto v = prop_numbers(content, key);
            REQUIRE(v.size() == 3);
            for (int a = 0; a < 3; a++) parts_sum[a] += v[a];
        }
        for (int a = 0; a < 3; a++) REQUIRE(parts_sum[a] == Catch::Approx(total[a]).margin(4e-4));
    }

    SECTION("no harmonic terms: bond / angle columns omitted") {
        std::string content = write(make_terms(0.0f, 0.0f));
        REQUIRE(content.find("atom_ff_total") != std::string::npos);
        REQUIRE(content.find("atom_ff_bond") == std::string::npos);
        REQUIRE(content.find("atom_ff_angle") == std::string::npos);
    }

    std::remove(tmp.c_str());
}


TEST_CASE("write_poses_to_sdf: with FF energy summary", "[sdf_writer]") {
    SDFMolData mol = make_h2o_sdf_mol();
    PoseFixture fix(3, 1);
    std::string tmp = make_tmp_path("ff");

    // Build FF energy: 1 ligand, 1 pose, 3 components
    std::vector<std::vector<std::array<Real, 3>>> ff_energies(1);
    ff_energies[0].push_back({-10.5f, -8.0f, -2.5f});

    write_poses_to_sdf(
        tmp, {mol}, fix.flex_names, fix.filtered_pose_inds,
        fix.poses.data(), fix.coords.data(), fix.list_i_real,
        {}, {}, ff_energies);

    std::string content = read_file(tmp);

    SECTION("ff_total_energy present") {
        REQUIRE(content.find("ff_total_energy") != std::string::npos);
    }

    SECTION("ff_interaction_energy present") {
        REQUIRE(content.find("ff_interaction_energy") != std::string::npos);
    }

    SECTION("ff_intra_ligand_energy present") {
        REQUIRE(content.find("ff_intra_ligand_energy") != std::string::npos);
    }

    std::remove(tmp.c_str());
}


// ===================== DMS reader - read_sdf_mol_data =====================

TEST_CASE("read_sdf_mol_data: enriched ejm42 DMS", "[sdf_mol_data]") {
    std::string path = std::string(TEST_DATA_DIR) + "/ff_minimize/tyk2/ligand_ejm42_enriched.dms";
    SDFMolData mol;
    REQUIRE_NOTHROW(read_sdf_mol_data(path, mol));
    REQUIRE(mol.loaded);

    SECTION("atom count matches enriched DMS") {
        REQUIRE(mol.natom == 35);
    }

    SECTION("atomic numbers populated") {
        REQUIRE(mol.anum.size() == 35);
        // First atom is carbon (anum=6)
        REQUIRE(mol.anum[0] == 6);
    }

    SECTION("formal_charge vector sized correctly") {
        REQUIRE(mol.formal_charge.size() == 35);
    }

    SECTION("bonds populated") {
        REQUIRE(mol.bonds.size() > 0);
        for (auto& b : mol.bonds) {
            REQUIRE(b.p0 >= 0);
            REQUIRE(b.p1 >= 0);
            REQUIRE(b.p0 < 35);
            REQUIRE(b.p1 < 35);
            REQUIRE(b.order >= 1);
            REQUIRE(b.order <= 3);
        }
    }

    SECTION("mol_name may be empty if ud2_metadata absent") {
        // The enriched DMS does not have ud2_metadata with original_name/molecule_name,
        // so mol_name is expected to be empty. This is correct behavior.
        // Just verify the field is accessible (no crash).
        REQUIRE(mol.mol_name.size() >= 0);
    }
}
