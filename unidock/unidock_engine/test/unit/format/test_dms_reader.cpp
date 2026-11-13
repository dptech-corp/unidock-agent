//
// Unit tests for DMS reader
//

#include <catch2/catch_amalgamated.hpp>
#include "format/dms_reader.h"
#include <sqlite3.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <filesystem>

static const std::string DMS_TEST_DIR = std::string(TEST_DATA_DIR) + "ff_minimize/tyk2/";

static std::string dms_path(const std::string& fname) {
    return DMS_TEST_DIR + fname;
}

TEST_CASE("read_ligand_dms: ejm42 ligand", "[dms_reader]") {
    FFMinData ff;
    std::string path = dms_path("ligand_ejm42.dms");
    REQUIRE_NOTHROW(read_ligand_dms(path, ff));
    REQUIRE(ff.loaded);

    SECTION("atom count") {
        REQUIRE(ff.lig_atoms.natom == 35);
    }

    SECTION("all atoms have LJ params") {
        REQUIRE(ff.lig_atoms.sigma.size() == 35);
        REQUIRE(ff.lig_atoms.epsilon.size() == 35);
        REQUIRE(ff.lig_atoms.charge.size() == 35);
        REQUIRE(ff.lig_atoms.anum.size() == 35);
    }

    SECTION("first atom is carbon") {
        REQUIRE(ff.lig_atoms.anum[0] == 6);  // C1
    }

    SECTION("charges are non-zero and reasonable") {
        // C1 has charge -0.096
        REQUIRE(ff.lig_atoms.charge[0] == Catch::Approx(-0.096f).margin(0.001));
        // Cl1 (atom 6) has charge -0.0599
        REQUIRE(ff.lig_atoms.charge[6] == Catch::Approx(-0.0599f).margin(0.001));
    }

    SECTION("sigma/epsilon are positive for heavy atoms") {
        for (int i = 0; i < ff.lig_atoms.natom; i++) {
            if (ff.lig_atoms.anum[i] > 1) {
                REQUIRE(ff.lig_atoms.sigma[i] > 0);
                REQUIRE(ff.lig_atoms.epsilon[i] > 0);
            }
        }
    }

    SECTION("1-4 pairs") {
        REQUIRE(ff.lig_pairs_14.size() == 73);
        // All aij should be positive (repulsive LJ A term)
        for (auto& p : ff.lig_pairs_14) {
            REQUIRE(p.aij > 0);
            REQUIRE(p.bij > 0);
            REQUIRE(p.i >= 0);
            REQUIRE(p.j >= 0);
            REQUIRE(p.i < 35);
            REQUIRE(p.j < 35);
        }
    }

    SECTION("torsions") {
        REQUIRE(ff.lig_torsions.size() > 0);
        // phi0 should be in radians (converted from degrees)
        for (auto& t : ff.lig_torsions) {
            REQUIRE(t.phi0 >= -M_PI - 0.01);
            REQUIRE(t.phi0 <= M_PI + 0.01);
            REQUIRE(t.p0 >= 0);
            REQUIRE(t.p3 < 35);
        }
    }

    SECTION("exclusions") {
        REQUIRE(ff.lig_exclusions.size() > 0);
    }
}


TEST_CASE("read_receptor_dms: tyk2 protein", "[dms_reader]") {
    FFMinData ff;
    std::string path = dms_path("protein.dms");
    REQUIRE_NOTHROW(read_receptor_dms(path, ff));

    SECTION("atom count") {
        REQUIRE(ff.rec_atoms.natom == 4667);
    }

    SECTION("all atoms have LJ params") {
        REQUIRE(ff.rec_atoms.sigma.size() == 4667);
        REQUIRE(ff.rec_atoms.epsilon.size() == 4667);
        REQUIRE(ff.rec_atoms.charge.size() == 4667);
    }

    SECTION("first atom is nitrogen (THR N-terminus)") {
        REQUIRE(ff.rec_atoms.anum[0] == 7);  // N
    }

    SECTION("charges sum is approximately integer") {
        double total_charge = 0;
        for (auto q : ff.rec_atoms.charge) {
            total_charge += q;
        }
        // Total protein charge should be close to an integer
        double frac = total_charge - std::round(total_charge);
        REQUIRE(std::abs(frac) < 0.1);
    }

    SECTION("coordinates are populated") {
        REQUIRE(ff.rec_atoms.x.size() == 4667);
        // Coordinates should not all be zero
        bool has_nonzero = false;
        for (int i = 0; i < 10; i++) {
            if (ff.rec_atoms.x[i] != 0 || ff.rec_atoms.y[i] != 0 || ff.rec_atoms.z[i] != 0) {
                has_nonzero = true;
                break;
            }
        }
        REQUIRE(has_nonzero);
    }
}


TEST_CASE("read_ligand_dms: small ligand", "[dms_reader]") {
    FFMinData ff;
    std::string path = dms_path("ligand_small1.dms");
    REQUIRE_NOTHROW(read_ligand_dms(path, ff));
    REQUIRE(ff.loaded);
    REQUIRE(ff.lig_atoms.natom == 17);
    REQUIRE(ff.lig_pairs_14.size() > 0);
    REQUIRE(ff.lig_torsions.size() > 0);
}


// ==================== Enriched DMS reader tests ====================

TEST_CASE("read_ligand_dms_full: enriched ejm42", "[dms_reader_full]") {
    std::string path = dms_path("ligand_ejm42_enriched.dms");
    UDFlexMol flex;
    FFMinData ff;
    AtomTypeProps type_props;
    init_default_atom_type_props(type_props);

    REQUIRE_NOTHROW(read_ligand_dms_full(path, flex, ff, type_props, true));

    SECTION("atom count") {
        REQUIRE(flex.natom == 35);
    }

    SECTION("coords populated") {
        REQUIRE(flex.coords.size() == 35 * 3);
        // Center should be reasonable (not all zeros)
        REQUIRE(std::abs(flex.center[0]) + std::abs(flex.center[1]) + std::abs(flex.center[2]) > 0);
    }

    SECTION("vina types populated") {
        REQUIRE(flex.vina_types.size() == 35);
        // First atom is a carbon (vina_type 2 = C_H or 3 = C_P)
        REQUIRE((flex.vina_types[0] == 2 || flex.vina_types[0] == 3));
    }

    SECTION("charges populated") {
        REQUIRE(flex.charges.size() == 35);
        REQUIRE(flex.charges[0] == Catch::Approx(-0.096f).margin(0.001));
    }

    SECTION("exclusion pairs") {
        REQUIRE(flex.pairs_1213.size() > 0);
        REQUIRE(flex.pairs_14.size() > 0);
    }

    SECTION("torsions") {
        REQUIRE(flex.torsions.size() == 4);
        for (auto& t : flex.torsions) {
            REQUIRE(t.atoms[0] >= 0);
            REQUIRE(t.atoms[3] < 35);
            REQUIRE(t.axis[0] == t.atoms[1]);
            REQUIRE(t.axis[1] == t.atoms[2]);
            REQUIRE(t.rotated_atoms.size() > 0);
            REQUIRE(t.range_list.size() >= 2);
        }
    }

    SECTION("dihedrals") {
        REQUIRE(flex.dihedrals.size() == 4);
    }

    SECTION("intra pairs") {
        REQUIRE(flex.intra_pairs.size() > 0);
        // Every pair index must be valid
        for (size_t i = 0; i < flex.intra_pairs.size(); i++) {
            REQUIRE(flex.intra_pairs[i] >= 0);
            REQUIRE(flex.intra_pairs[i] < 35);
        }
    }

    SECTION("FF data also loaded") {
        REQUIRE(ff.loaded);
        REQUIRE(ff.lig_atoms.natom == 35);
        REQUIRE(ff.lig_pairs_14.size() == 73);
    }

    SECTION("molecule name derived from filename") {
        REQUIRE(flex.name == "ligand_ejm42_enriched");
    }
}


// ==================== Bond + Angle loader tests ====================

TEST_CASE("read_ligand_dms: ejm42 harmonic bonds loaded", "[dms_reader][bonds]") {
    FFMinData ff;
    std::string path = dms_path("ligand_ejm42.dms");
    REQUIRE_NOTHROW(read_ligand_dms(path, ff));
    REQUIRE(ff.loaded);

    SECTION("bond count matches stretch_harm_term rows") {
        REQUIRE(ff.lig_bonds.size() == 36);
    }

    SECTION("all bond indices are in-range") {
        for (auto& b : ff.lig_bonds) {
            REQUIRE(b.i >= 0);
            REQUIRE(b.j >= 0);
            REQUIRE(b.i < 35);
            REQUIRE(b.j < 35);
        }
    }

    SECTION("spot-check bond (i=0, j=1): r0 and fc") {
        // First bond ordered by (p0, p1): p0=0, p1=1, r0=1.3986, fc=354.25
        bool found = false;
        for (auto& b : ff.lig_bonds) {
            if (b.i == 0 && b.j == 1) {
                REQUIRE(b.r0 == Catch::Approx(1.3986).margin(0.001));
                REQUIRE(b.fc == Catch::Approx(354.25).margin(0.1));
                found = true;
                break;
            }
        }
        REQUIRE(found);
    }

    SECTION("r0 and fc are positive") {
        for (auto& b : ff.lig_bonds) {
            REQUIRE(b.r0 > 0.0f);
            REQUIRE(b.fc > 0.0f);
        }
    }
}


TEST_CASE("read_ligand_dms: ejm42 harmonic angles loaded", "[dms_reader][angles]") {
    FFMinData ff;
    std::string path = dms_path("ligand_ejm42.dms");
    REQUIRE_NOTHROW(read_ligand_dms(path, ff));
    REQUIRE(ff.loaded);

    SECTION("angle count matches angle_harm_term rows") {
        REQUIRE(ff.lig_angles.size() == 58);
    }

    SECTION("all angle indices are in-range") {
        for (auto& a : ff.lig_angles) {
            REQUIRE(a.i >= 0);
            REQUIRE(a.j >= 0);
            REQUIRE(a.k >= 0);
            REQUIRE(a.i < 35);
            REQUIRE(a.j < 35);
            REQUIRE(a.k < 35);
        }
    }

    SECTION("spot-check angle (i=0, j=1, k=2): theta0 in radians and fc") {
        // From DB: p0=0, p1=1, p2=2, theta0=120.020051666836 deg, fc=63.67
        const double expected_theta0_rad = 120.020051666836 * M_PI / 180.0;
        bool found = false;
        for (auto& a : ff.lig_angles) {
            if (a.i == 0 && a.j == 1 && a.k == 2) {
                REQUIRE(a.theta0 == Catch::Approx(expected_theta0_rad).margin(0.001));
                REQUIRE(a.fc == Catch::Approx(63.67).margin(0.1));
                found = true;
                break;
            }
        }
        REQUIRE(found);
    }

    SECTION("theta0 values are in plausible range (radians)") {
        // All angles should be between 60 and 180 degrees expressed in radians
        for (auto& a : ff.lig_angles) {
            REQUIRE(a.theta0 > 60.0 * M_PI / 180.0);
            REQUIRE(a.theta0 < 180.0 * M_PI / 180.0);
        }
    }

    SECTION("fc values are positive") {
        for (auto& a : ff.lig_angles) {
            REQUIRE(a.fc > 0.0f);
        }
    }
}


TEST_CASE("read_receptor_dms_full: enriched tyk2", "[dms_reader_full]") {
    std::string path = dms_path("receptor_enriched.dms");
    UDFixMol fix;
    FFMinData ff;
    AtomTypeProps type_props;
    init_default_atom_type_props(type_props);

    // Use a very large box to include all atoms
    Box big_box;
    big_box.x_lo = -1000; big_box.x_hi = 1000;
    big_box.y_lo = -1000; big_box.y_hi = 1000;
    big_box.z_lo = -1000; big_box.z_hi = 1000;

    REQUIRE_NOTHROW(read_receptor_dms_full(path, big_box, fix, ff, type_props));

    SECTION("atom count (minus hydrogens)") {
        // 4667 total - H atoms that are filtered by is_hydrogen
        REQUIRE(fix.natom > 0);
        REQUIRE(fix.natom < 4667);  // Some H removed
    }

    SECTION("vina types populated") {
        REQUIRE(fix.vina_types.size() == static_cast<size_t>(fix.natom));
    }

    SECTION("FF receptor data loaded") {
        REQUIRE(ff.rec_atoms.natom == 4667);
    }
}


// ==================== FF loaded only when populated ====================

namespace {

/// Copy of a test DMS in the system temp directory, removed when the guard goes out of scope
/// (also when a REQUIRE fails).
struct TmpDmsCopy {
    const std::string path;
    TmpDmsCopy(const std::string& fname, const char* label)
        : path((std::filesystem::temp_directory_path()
                / ("test_dms_ff_populated_" + std::string(label) + "_"
                   + std::to_string(static_cast<int>(getpid())) + ".dms")).string()) {
        std::filesystem::copy_file(dms_path(fname), path, std::filesystem::copy_options::overwrite_existing);
    }
    ~TmpDmsCopy() { std::remove(path.c_str()); }
};

void exec_sql(const std::string& path, const char* sql) {
    sqlite3* db = nullptr;
    REQUIRE(sqlite3_open(path.c_str(), &db) == SQLITE_OK);
    const int rc = sqlite3_exec(db, sql, nullptr, nullptr, nullptr);
    INFO("sqlite: " << sqlite3_errmsg(db));
    sqlite3_close(db);
    REQUIRE(rc == SQLITE_OK);
}

// Edits applied to the temporary copies
const char* SQL_EMPTY_NB = "DELETE FROM nonbonded_param";
const char* SQL_DROP_NB = "DROP TABLE nonbonded_param";
const char* SQL_UNTYPE_ATOM0 = "UPDATE particle SET nbtype = 9999 WHERE id = 0";

/// Turns a copy of ligand_ejm42_enriched.dms (35 atoms) into a two-ligand merged DMS with the
/// ud2_molecules layout of the Python merge: atoms 0-34 are ligand 0, 35-69 the same molecule
/// again as ligand 1, FF and bonded terms duplicated with shifted atom ids. Torsion rows are left
/// out (rigid ligands), which the merged reader accepts.
const char* SQL_MERGE_TWO =
    "CREATE TEMP TABLE p2 AS SELECT * FROM particle;"
    "UPDATE p2 SET id = id + 35;"
    "INSERT INTO particle SELECT * FROM p2;"
    "INSERT INTO bond SELECT p0 + 35, p1 + 35, [order] FROM bond;"
    "INSERT INTO exclusion SELECT p0 + 35, p1 + 35 FROM exclusion;"
    "INSERT INTO pair_12_6_es_term SELECT p0 + 35, p1 + 35, param FROM pair_12_6_es_term;"
    "INSERT INTO stretch_harm_term SELECT p0 + 35, p1 + 35, constrained, param FROM stretch_harm_term;"
    "INSERT INTO angle_harm_term SELECT p0 + 35, p1 + 35, p2 + 35, constrained, param FROM angle_harm_term;"
    "INSERT INTO dihedral_trig_term SELECT p0 + 35, p1 + 35, p2 + 35, p3 + 35, param FROM dihedral_trig_term;"
    "CREATE TABLE ap AS SELECT atom_id, 0 AS molecule_id, vina_type, ff_type FROM ud2_atom_props;"
    "INSERT INTO ap SELECT atom_id + 35, 1, vina_type, ff_type FROM ud2_atom_props;"
    "DROP TABLE ud2_atom_props; ALTER TABLE ap RENAME TO ud2_atom_props;"
    "CREATE TABLE ra AS SELECT atom_id, 0 AS molecule_id FROM ud2_root_atoms;"
    "INSERT INTO ra SELECT atom_id + 35, 1 FROM ud2_root_atoms;"
    "DROP TABLE ud2_root_atoms; ALTER TABLE ra RENAME TO ud2_root_atoms;"
    "DROP TABLE ud2_torsions; DROP TABLE ud2_torsion_ranges;"
    "CREATE TABLE ud2_torsions (torsion_id INTEGER, molecule_id INTEGER, atom_i INTEGER, atom_j INTEGER,"
    " atom_k INTEGER, atom_l INTEGER, dihedral_value REAL, mobile_atoms TEXT);"
    "CREATE TABLE ud2_molecules (molecule_id INTEGER PRIMARY KEY, mol_name TEXT, atom_offset INTEGER, natom INTEGER);"
    "INSERT INTO ud2_molecules VALUES (0, 'lig_0', 0, 35), (1, 'lig_1', 35, 35);";

/// FFMinData as an earlier successful read leaves it (flag and bonded lists), to show that a read
/// finding no FF clears it
FFMinData stale_ligand_ff() {
    FFMinData ff;
    ff.loaded = true;
    ff.lig_pairs_14.push_back({});
    ff.lig_torsions.push_back({});
    ff.lig_exclusions.push_back({});
    ff.lig_bonds.push_back({});
    ff.lig_angles.push_back({});
    return ff;
}

void require_no_ligand_ff(const FFMinData& ff) {
    REQUIRE(!ff.loaded);
    REQUIRE(ff.lig_atoms.natom == 0);
    REQUIRE(ff.lig_atoms.charge.empty());
    REQUIRE(ff.lig_pairs_14.empty());
    REQUIRE(ff.lig_torsions.empty());
    REQUIRE(ff.lig_exclusions.empty());
    REQUIRE(ff.lig_bonds.empty());
    REQUIRE(ff.lig_angles.empty());
}

} // anonymous namespace


TEST_CASE("read_ligand_dms: FF loaded only when every atom has nonbonded parameters", "[dms_reader][ff_populated]") {
    const TmpDmsCopy tmp("ligand_ejm42.dms", "lig");

    SECTION("untouched copy") {
        FFMinData ff;
        read_ligand_dms(tmp.path, ff);
        REQUIRE(ff.loaded);
        REQUIRE(ff.lig_atoms.natom == 35);
    }

    SECTION("empty nonbonded_param table (Vina-only DMS)") {
        exec_sql(tmp.path, SQL_EMPTY_NB);
        FFMinData ff = stale_ligand_ff();
        read_ligand_dms(tmp.path, ff);
        require_no_ligand_ff(ff);
    }

    SECTION("one atom without parameters") {
        exec_sql(tmp.path, SQL_UNTYPE_ATOM0);
        FFMinData ff = stale_ligand_ff();
        read_ligand_dms(tmp.path, ff);
        require_no_ligand_ff(ff);
    }

    SECTION("nonbonded_param absent") {
        exec_sql(tmp.path, SQL_DROP_NB);
        FFMinData ff = stale_ligand_ff();
        read_ligand_dms(tmp.path, ff);
        require_no_ligand_ff(ff);
    }
}


TEST_CASE("read_ligand_dms_full: empty FF tables leave the flex read intact", "[dms_reader_full][ff_populated]") {
    const TmpDmsCopy tmp("ligand_ejm42_enriched.dms", "lig_full");
    exec_sql(tmp.path, SQL_EMPTY_NB);
    AtomTypeProps type_props;
    init_default_atom_type_props(type_props);
    UDFlexMol flex;
    FFMinData ff;
    REQUIRE_NOTHROW(read_ligand_dms_full(tmp.path, flex, ff, type_props, false));
    REQUIRE(flex.natom == 35);
    REQUIRE(flex.torsions.size() > 0);
    REQUIRE(!ff.loaded);
    REQUIRE(ff.lig_atoms.natom == 0);
}


TEST_CASE("read_multi_ligand_dms_full: FF per ligand only when every atom has parameters", "[dms_reader_full][ff_populated]") {
    const TmpDmsCopy tmp("ligand_ejm42_enriched.dms", "merged");
    exec_sql(tmp.path, SQL_MERGE_TWO);
    AtomTypeProps type_props;
    init_default_atom_type_props(type_props);
    UDFlexMolList flex;
    std::vector<FFMinData> ff;
    std::vector<SDFMolData> sdf;

    SECTION("both ligands parameterised") {
        REQUIRE_NOTHROW(read_multi_ligand_dms_full(tmp.path, flex, ff, sdf, type_props, false));
        REQUIRE(flex.size() == 2);
        REQUIRE(ff.size() == 2);
        for (int k = 0; k < 2; k++) {
            INFO("ligand " << k);
            REQUIRE(flex[k].natom == 35);
            REQUIRE(ff[k].loaded);
            REQUIRE(ff[k].lig_atoms.natom == 35);
            REQUIRE(!ff[k].lig_bonds.empty());
        }
    }

    SECTION("one atom of ligand 1 without parameters") {
        exec_sql(tmp.path, "UPDATE particle SET nbtype = 9999 WHERE id = 35");
        REQUIRE_NOTHROW(read_multi_ligand_dms_full(tmp.path, flex, ff, sdf, type_props, false));
        REQUIRE(flex.size() == 2);
        REQUIRE(ff.size() == 2);
        REQUIRE(flex[1].natom == 35);
        REQUIRE(ff[0].loaded);
        REQUIRE(ff[0].lig_atoms.natom == 35);
        require_no_ligand_ff(ff[1]);
    }
}


TEST_CASE("read_receptor_dms: an empty or partial receptor table switches a loaded FF off", "[dms_reader][ff_populated]") {
    const TmpDmsCopy tmp("protein.dms", "rec");
    FFMinData ff;
    ff.loaded = true;  // as after a successful read_ligand_dms

    SECTION("populated receptor keeps it loaded") {
        read_receptor_dms(tmp.path, ff);
        REQUIRE(ff.loaded);
        REQUIRE(ff.rec_atoms.natom == 4667);
    }

    SECTION("empty nonbonded_param table") {
        exec_sql(tmp.path, SQL_EMPTY_NB);
        read_receptor_dms(tmp.path, ff);
        REQUIRE(!ff.loaded);
        REQUIRE(ff.rec_atoms.natom == 0);
    }

    SECTION("one atom without parameters") {
        exec_sql(tmp.path, SQL_UNTYPE_ATOM0);
        read_receptor_dms(tmp.path, ff);
        REQUIRE(!ff.loaded);
        REQUIRE(ff.rec_atoms.natom == 0);
    }

    SECTION("nonbonded_param absent: nothing read, loaded left as it is") {
        exec_sql(tmp.path, SQL_DROP_NB);
        read_receptor_dms(tmp.path, ff);
        REQUIRE(ff.loaded);
        REQUIRE(ff.rec_atoms.natom == 0);
    }
}


// ==================== Hydrogen-only torsions leave the search ====================

namespace {

const std::string HROTOR_TEST_DIR = std::string(TEST_DATA_DIR) + "hrotor/";

/// The value read_ligand_dms_full stores for a DMS dihedral_value in degrees
Real dms_dihedral(double deg) {
    return static_cast<Real>(deg) * static_cast<Real>(M_PI / 180.0);
}

void require_torsion(const UDTorsion& t, int i, int j, int k, int l) {
    INFO("torsion " << t.atoms[0] << "," << t.atoms[1] << "," << t.atoms[2] << "," << t.atoms[3]);
    REQUIRE(t.atoms[0] == i);
    REQUIRE(t.atoms[1] == j);
    REQUIRE(t.atoms[2] == k);
    REQUIRE(t.atoms[3] == l);
    REQUIRE(t.axis[0] == j);
    REQUIRE(t.axis[1] == k);
}

/// ejm42 Cl20 on the root ring becomes a phenol-like O (O_DA, out of the root). A row rotating it with
/// H23 then has an axis on the root, so its fragment holds two intra pairs out ((5,20), (6,20)).
const char* SQL_ROOT_OH =
    "UPDATE ud2_atom_props SET vina_type = 11 WHERE atom_id = 20;"
    "DELETE FROM ud2_root_atoms WHERE atom_id = 20;";

/// Keeps the ejm42 torsion rows for SQL_HROT_ROWS (SQL_MERGE_TWO drops the table)
const char* SQL_KEEP_TORSIONS = "CREATE TABLE tor_src AS SELECT * FROM ud2_torsions";

/// Torsion rows of the merged copy (after SQL_MERGE_TWO and SQL_ROOT_OH). Ligand 0: the four ejm42
/// torsions with an -NH3 rotor inserted as row 2 (methyl C30 retyped N_D, rotated atoms C30 and its
/// three H) and the root O-H as row 5. Ligand 1: a one-H N-H row with the axis listed H-bearing atom
/// first (axis N16-C17, rotated N16 and H29) and a hydrogen-only row that rotates neither axis atom.
const char* SQL_HROT_ROWS =
    "INSERT INTO ud2_torsions SELECT torsion_id + (torsion_id >= 2), 0, atom_i, atom_j, atom_k, atom_l,"
    " dihedral_value, mobile_atoms FROM tor_src;"
    "INSERT INTO ud2_torsions VALUES (2, 0, 17, 19, 30, 32, 60.0, '30,32,33,34'), (5, 0, 1, 2, 20, 23, 0.0, '20,23');"
    "UPDATE ud2_atom_props SET vina_type = 5 WHERE atom_id = 30;"
    "INSERT INTO ud2_torsions VALUES (0, 1, 64, 51, 52, 53, -150.0, '51,64'), (1, 1, 52, 54, 65, 67, 45.0, '67,68,69');"
    "DROP TABLE tor_src;";

} // anonymous namespace


TEST_CASE("read_ligand_dms_full: hydrogen-only rotors of CHEMBL191003 leave the search", "[dms_reader][ff_only_torsions]") {
    AtomTypeProps type_props;
    init_default_atom_type_props(type_props);
    UDFlexMol flex;
    FFMinData ff;
    REQUIRE_NOTHROW(read_ligand_dms_full(HROTOR_TEST_DIR + "CHEMBL191003_e2e.dms", flex, ff, type_props, true));

    // DMS torsions 1, 2, 3, 5, 6 in order
    REQUIRE(flex.torsions.size() == 5);
    REQUIRE(flex.dihedrals.size() == 5);
    require_torsion(flex.torsions[0], 5, 4, 3, 2);
    require_torsion(flex.torsions[1], 6, 5, 4, 3);
    require_torsion(flex.torsions[2], 7, 8, 9, 10);
    require_torsion(flex.torsions[3], 1, 16, 17, 18);
    require_torsion(flex.torsions[4], 18, 17, 19, 20);
    REQUIRE(flex.dihedrals[0] == dms_dihedral(38.53718988883868));
    REQUIRE(flex.dihedrals[1] == dms_dihedral(59.04222019137608));
    REQUIRE(flex.dihedrals[2] == dms_dihedral(-139.3948187355399));
    REQUIRE(flex.dihedrals[3] == dms_dihedral(-90.37666508239856));
    REQUIRE(flex.dihedrals[4] == dms_dihedral(89.26377510870479));

    // DMS torsion 0 (guanidine C1-N0H2) and 4 (sulfonamide S9-N10H2)
    REQUIRE(flex.ff_only_torsions.size() == 2);
    REQUIRE(flex.ff_only_dihedrals.size() == 2);
    require_torsion(flex.ff_only_torsions[0], 2, 1, 0, 27);
    require_torsion(flex.ff_only_torsions[1], 8, 9, 10, 32);
    REQUIRE(flex.ff_only_torsions[0].rotated_atoms == std::vector<int>{0, 27, 28});
    REQUIRE(flex.ff_only_torsions[1].rotated_atoms == std::vector<int>{10, 32, 33});
    REQUIRE(flex.ff_only_dihedrals[0] == dms_dihedral(-0.005699502657121795));
    REQUIRE(flex.ff_only_dihedrals[1] == dms_dihedral(142.6357882694499));

    // Fragments still come from all seven torsions (count read before the split)
    REQUIRE(flex.intra_pairs.size() / 2 == 227);
}


TEST_CASE("read_ligand_dms_full: the one-H sulfonamide N-H of 3DD0 stays in the search", "[dms_reader][ff_only_torsions]") {
    AtomTypeProps type_props;
    init_default_atom_type_props(type_props);
    UDFlexMol flex;
    FFMinData ff;
    REQUIRE_NOTHROW(read_ligand_dms_full(HROTOR_TEST_DIR + "3DD0_ligand.dms", flex, ff, type_props, true));

    REQUIRE(flex.torsions.size() == 4);
    REQUIRE(flex.dihedrals.size() == 4);
    require_torsion(flex.torsions[0], 6, 8, 0, 16);
    require_torsion(flex.torsions[1], 3, 4, 14, 5);
    require_torsion(flex.torsions[2], 4, 14, 5, 15);
    require_torsion(flex.torsions[3], 10, 9, 8, 0);
    REQUIRE(flex.dihedrals[0] == dms_dihedral(147.2802412678392));
    REQUIRE(flex.dihedrals[3] == dms_dihedral(-49.11701595967716));
    REQUIRE(flex.ff_only_torsions.empty());
    REQUIRE(flex.ff_only_dihedrals.empty());
    REQUIRE(flex.intra_pairs.size() / 2 == 43);
}


TEST_CASE("read_ligand_dms_full: a root O-H rotor leaves the search after the fragments", "[dms_reader][ff_only_torsions]") {
    const TmpDmsCopy tmp("ligand_ejm42_enriched.dms", "hrot_root_oh");
    exec_sql(tmp.path, SQL_ROOT_OH);
    exec_sql(tmp.path, "INSERT INTO ud2_torsions VALUES (4, 1, 2, 20, 23, 0.0, '20,23')");
    AtomTypeProps type_props;
    init_default_atom_type_props(type_props);
    UDFlexMol flex;
    FFMinData ff;
    REQUIRE_NOTHROW(read_ligand_dms_full(tmp.path, flex, ff, type_props, false));

    REQUIRE(flex.torsions.size() == 4);
    REQUIRE(flex.dihedrals.size() == 4);
    require_torsion(flex.torsions[3], 18, 17, 19, 30);
    REQUIRE(flex.dihedrals[3] == dms_dihedral(17.54520830361666));
    REQUIRE(flex.ff_only_torsions.size() == 1);
    REQUIRE(flex.ff_only_dihedrals.size() == 1);
    require_torsion(flex.ff_only_torsions[0], 1, 2, 20, 23);
    REQUIRE(flex.ff_only_dihedrals[0] == dms_dihedral(0.0));
    REQUIRE(flex.intra_pairs.size() / 2 == 138);
}


TEST_CASE("read_multi_ligand_dms_full: hydrogen-only torsions leave the search per ligand", "[dms_reader][ff_only_torsions]") {
    const TmpDmsCopy tmp("ligand_ejm42_enriched.dms", "merged_hrot");
    exec_sql(tmp.path, SQL_KEEP_TORSIONS);
    exec_sql(tmp.path, SQL_MERGE_TWO);
    exec_sql(tmp.path, SQL_ROOT_OH);
    exec_sql(tmp.path, SQL_HROT_ROWS);
    AtomTypeProps type_props;
    init_default_atom_type_props(type_props);
    UDFlexMolList flex;
    std::vector<FFMinData> ff;
    std::vector<SDFMolData> sdf;

    SECTION("-NH3 and root O-H rotors moved, one-H N-H and ambiguous rows kept") {
        REQUIRE_NOTHROW(read_multi_ligand_dms_full(tmp.path, flex, ff, sdf, type_props, false));
        REQUIRE(flex.size() == 2);

        REQUIRE(flex[0].torsions.size() == 4);
        REQUIRE(flex[0].dihedrals.size() == 4);
        require_torsion(flex[0].torsions[0], 2, 3, 7, 8);
        require_torsion(flex[0].torsions[1], 7, 9, 10, 15);
        require_torsion(flex[0].torsions[2], 15, 14, 16, 17);
        require_torsion(flex[0].torsions[3], 18, 17, 19, 30);
        REQUIRE(flex[0].dihedrals[1] == dms_dihedral(-173.2229914305481));
        REQUIRE(flex[0].dihedrals[2] == dms_dihedral(30.10055885784124));
        REQUIRE(flex[0].ff_only_torsions.size() == 2);
        REQUIRE(flex[0].ff_only_dihedrals.size() == 2);
        require_torsion(flex[0].ff_only_torsions[0], 17, 19, 30, 32);
        require_torsion(flex[0].ff_only_torsions[1], 1, 2, 20, 23);
        REQUIRE(flex[0].ff_only_torsions[0].rotated_atoms == std::vector<int>{30, 32, 33, 34});
        REQUIRE(flex[0].ff_only_dihedrals[0] == dms_dihedral(60.0));
        REQUIRE(flex[0].ff_only_dihedrals[1] == dms_dihedral(0.0));
        REQUIRE(flex[0].intra_pairs.size() / 2 == 138);

        REQUIRE(flex[1].torsions.size() == 2);
        REQUIRE(flex[1].dihedrals.size() == 2);
        require_torsion(flex[1].torsions[0], 29, 16, 17, 18);
        require_torsion(flex[1].torsions[1], 17, 19, 30, 32);
        REQUIRE(flex[1].dihedrals[0] == dms_dihedral(-150.0));
        REQUIRE(flex[1].ff_only_torsions.empty());
        REQUIRE(flex[1].ff_only_dihedrals.empty());
        REQUIRE(flex[1].intra_pairs.size() / 2 == 142);
    }

    SECTION("a one-H O-H rotor is moved") {
        exec_sql(tmp.path, "UPDATE ud2_atom_props SET vina_type = 11 WHERE atom_id = 51");
        REQUIRE_NOTHROW(read_multi_ligand_dms_full(tmp.path, flex, ff, sdf, type_props, false));
        REQUIRE(flex.size() == 2);
        REQUIRE(flex[1].torsions.size() == 1);
        REQUIRE(flex[1].dihedrals.size() == 1);
        require_torsion(flex[1].torsions[0], 17, 19, 30, 32);
        REQUIRE(flex[1].dihedrals[0] == dms_dihedral(45.0));
        REQUIRE(flex[1].ff_only_torsions.size() == 1);
        require_torsion(flex[1].ff_only_torsions[0], 29, 16, 17, 18);
        REQUIRE(flex[1].ff_only_dihedrals[0] == dms_dihedral(-150.0));
    }

    SECTION("a thiol S-H rotor is moved, rows rotating both axis atoms or no hydrogen are kept") {
        // N16 retyped S_P; the methyl row lists both axis atoms; a new row rotates only its axis atom C19
        exec_sql(tmp.path,
                 "UPDATE ud2_atom_props SET vina_type = 15 WHERE atom_id = 51;"
                 "UPDATE ud2_torsions SET mobile_atoms = '54,65,67,68,69' WHERE molecule_id = 1 AND torsion_id = 1;"
                 "INSERT INTO ud2_torsions VALUES (2, 1, 53, 52, 54, 56, 10.0, '54');");
        REQUIRE_NOTHROW(read_multi_ligand_dms_full(tmp.path, flex, ff, sdf, type_props, false));
        REQUIRE(flex.size() == 2);
        REQUIRE(flex[1].torsions.size() == 2);
        REQUIRE(flex[1].dihedrals.size() == 2);
        require_torsion(flex[1].torsions[0], 17, 19, 30, 32);
        require_torsion(flex[1].torsions[1], 18, 17, 19, 21);
        REQUIRE(flex[1].dihedrals[1] == dms_dihedral(10.0));
        REQUIRE(flex[1].ff_only_torsions.size() == 1);
        require_torsion(flex[1].ff_only_torsions[0], 29, 16, 17, 18);
        REQUIRE(flex[1].ff_only_dihedrals[0] == dms_dihedral(-150.0));
    }
}


// ==================== aromatic-ring bias tables ====================

namespace {

const char* SQL_RING_TABLES =
    "CREATE TABLE ud2_ring_bias (ring_row INTEGER, cx REAL, cy REAL, cz REAL, nx REAL, ny REAL, nz REAL, v_set REAL);"
    "CREATE TABLE ud2_ring_bias_atoms (ring_row INTEGER, pos INTEGER, atom_id INTEGER);";

/// One ring row; the atom rows are inserted in reverse pos order so the reader has to sort them
std::string ring_row_sql(int ring_row, const std::vector<int>& atoms, double cx, double cy, double cz,
                         double nx, double ny, double nz, double v_set){
    std::string s = "INSERT INTO ud2_ring_bias VALUES (" + std::to_string(ring_row) + ", " + std::to_string(cx) + ", "
        + std::to_string(cy) + ", " + std::to_string(cz) + ", " + std::to_string(nx) + ", " + std::to_string(ny) + ", "
        + std::to_string(nz) + ", " + std::to_string(v_set) + ");";
    for (int k = (int)atoms.size() - 1; k >= 0; k--){
        s += "INSERT INTO ud2_ring_bias_atoms VALUES (" + std::to_string(ring_row) + ", " + std::to_string(k) + ", "
           + std::to_string(atoms[k]) + ");";
    }
    return s;
}

void require_ring(const RingBiasRow& r, const std::vector<int>& atoms){
    REQUIRE(r.n == (int)atoms.size());
    for (int k = 0; k < r.n; k++) REQUIRE(r.atoms[k] == atoms[k]);
}

void read_single(const std::string& path, UDFlexMol& flex){
    AtomTypeProps type_props;
    init_default_atom_type_props(type_props);
    FFMinData ff;
    read_ligand_dms_full(path, flex, ff, type_props, false);
}

}  // namespace


TEST_CASE("read_ligand_dms_full: ring-bias tables", "[dms_reader][ring_bias]") {
    const TmpDmsCopy tmp("ligand_ejm42_enriched.dms", "ring");

    SECTION("tables absent: no rows") {
        UDFlexMol flex;
        REQUIRE_NOTHROW(read_single(tmp.path, flex));
        REQUIRE(flex.natom == 35);
        REQUIRE(flex.ring.rows.empty());
    }

    SECTION("rows in ring_row order, atoms in pos order, normal normalised") {
        exec_sql(tmp.path, SQL_RING_TABLES);
        exec_sql(tmp.path, (ring_row_sql(7, {0, 1, 2, 3, 4, 5}, 1.0, 2.0, 3.0, 0.0, 0.0, 2.0, 4.0)
                            + ring_row_sql(3, {10, 11, 12, 13, 14, 15}, -1.5, 0.5, 2.5, 3.0, 4.0, 0.0, 2.0)
                            + ring_row_sql(9, {14, 13, 12, 11, 10}, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)).c_str());
        UDFlexMol flex;
        REQUIRE_NOTHROW(read_single(tmp.path, flex));
        REQUIRE(flex.ring.rows.size() == 3);
        const RingBiasRow& py = flex.ring.rows[0];
        const RingBiasRow& bz = flex.ring.rows[1];
        const RingBiasRow& five = flex.ring.rows[2];
        require_ring(py, {10, 11, 12, 13, 14, 15});
        require_ring(bz, {0, 1, 2, 3, 4, 5});
        require_ring(five, {14, 13, 12, 11, 10});
        REQUIRE(py.c[0] == Catch::Approx(-1.5));
        REQUIRE(py.c[1] == Catch::Approx(0.5));
        REQUIRE(py.c[2] == Catch::Approx(2.5));
        REQUIRE(py.nt[0] == Catch::Approx(0.6));
        REQUIRE(py.nt[1] == Catch::Approx(0.8));
        REQUIRE(py.nt[2] == Catch::Approx(0.0).margin(1e-7));
        REQUIRE(py.v_set == Catch::Approx(2.0));
        REQUIRE(bz.nt[2] == Catch::Approx(1.0));
        REQUIRE(bz.v_set == Catch::Approx(4.0));
        REQUIRE(five.nt[0] == Catch::Approx(1.0 / std::sqrt(3.0)));
    }

    SECTION("empty tables: no rows") {
        exec_sql(tmp.path, SQL_RING_TABLES);
        UDFlexMol flex;
        REQUIRE_NOTHROW(read_single(tmp.path, flex));
        REQUIRE(flex.ring.rows.empty());
    }

    SECTION("one table without the other") {
        exec_sql(tmp.path, "CREATE TABLE ud2_ring_bias (ring_row INTEGER, cx REAL, cy REAL, cz REAL, "
                           "nx REAL, ny REAL, nz REAL, v_set REAL)");
        UDFlexMol flex;
        REQUIRE_THROWS_WITH(read_single(tmp.path, flex), Catch::Matchers::ContainsSubstring("only one of the ring-bias tables"));
    }

    SECTION("bad ring sizes") {
        exec_sql(tmp.path, SQL_RING_TABLES);
        for (const std::vector<int>& atoms : {std::vector<int>{0, 1, 2, 3}, std::vector<int>{0, 1, 2, 3, 4, 5, 6},
                                              std::vector<int>{}}){
            INFO("ring of " << atoms.size() << " atoms");
            exec_sql(tmp.path, "DELETE FROM ud2_ring_bias; DELETE FROM ud2_ring_bias_atoms;");
            exec_sql(tmp.path, ring_row_sql(0, atoms, 0, 0, 0, 0, 0, 1, 1).c_str());
            UDFlexMol flex;
            REQUIRE_THROWS_WITH(read_single(tmp.path, flex),
                                Catch::Matchers::ContainsSubstring("has " + std::to_string(atoms.size()) + " atoms")
                                && Catch::Matchers::ContainsSubstring("5- or 6-membered"));
        }
    }

    SECTION("ring atom outside the ligand") {
        exec_sql(tmp.path, SQL_RING_TABLES);
        exec_sql(tmp.path, ring_row_sql(0, {30, 31, 32, 33, 34, 35}, 0, 0, 0, 0, 0, 1, 1).c_str());
        UDFlexMol flex;
        REQUIRE_THROWS_WITH(read_single(tmp.path, flex), Catch::Matchers::ContainsSubstring("atom_id 35 outside"));
    }

    SECTION("zero template normal") {
        exec_sql(tmp.path, SQL_RING_TABLES);
        exec_sql(tmp.path, ring_row_sql(0, {0, 1, 2, 3, 4, 5}, 0, 0, 0, 0, 0, 0, 1).c_str());
        UDFlexMol flex;
        REQUIRE_THROWS_WITH(read_single(tmp.path, flex), Catch::Matchers::ContainsSubstring("zero template normal"));
    }
}


TEST_CASE("read_multi_ligand_dms_full: ring-bias rows per ligand, atom ids rebased", "[dms_reader][ring_bias]") {
    const TmpDmsCopy tmp("ligand_ejm42_enriched.dms", "merged_ring");
    exec_sql(tmp.path, SQL_MERGE_TWO);
    exec_sql(tmp.path, SQL_RING_TABLES);
    AtomTypeProps type_props;
    init_default_atom_type_props(type_props);
    UDFlexMolList flex;
    std::vector<FFMinData> ff;
    std::vector<SDFMolData> sdf;

    SECTION("each ligand gets its own rows") {
        exec_sql(tmp.path, (ring_row_sql(1, {0, 1, 2, 3, 4, 5}, 1, 1, 1, 0, 0, 1, 2.0)
                            + ring_row_sql(2, {45, 46, 47, 48, 49, 50}, 2, 2, 2, 0, 1, 0, 4.0)
                            + ring_row_sql(3, {35, 36, 37, 38, 39, 40}, 3, 3, 3, 1, 0, 0, 2.0)
                            + ring_row_sql(0, {10, 11, 12, 13, 14, 15}, 4, 4, 4, 0, 0, 1, 4.0)).c_str());
        REQUIRE_NOTHROW(read_multi_ligand_dms_full(tmp.path, flex, ff, sdf, type_props, false));
        REQUIRE(flex.size() == 2);
        REQUIRE(flex[0].ring.rows.size() == 2);
        require_ring(flex[0].ring.rows[0], {10, 11, 12, 13, 14, 15});
        require_ring(flex[0].ring.rows[1], {0, 1, 2, 3, 4, 5});
        REQUIRE(flex[0].ring.rows[0].c[0] == Catch::Approx(4.0));
        REQUIRE(flex[1].ring.rows.size() == 2);
        require_ring(flex[1].ring.rows[0], {10, 11, 12, 13, 14, 15});
        require_ring(flex[1].ring.rows[1], {0, 1, 2, 3, 4, 5});
        REQUIRE(flex[1].ring.rows[0].nt[1] == Catch::Approx(1.0));
        REQUIRE(flex[1].ring.rows[1].c[0] == Catch::Approx(3.0));
    }

    SECTION("no rows: both ligands empty") {
        REQUIRE_NOTHROW(read_multi_ligand_dms_full(tmp.path, flex, ff, sdf, type_props, false));
        REQUIRE(flex.size() == 2);
        REQUIRE(flex[0].ring.rows.empty());
        REQUIRE(flex[1].ring.rows.empty());
    }

    SECTION("a ring across two ligands") {
        exec_sql(tmp.path, ring_row_sql(0, {32, 33, 34, 35, 36, 37}, 0, 0, 0, 0, 0, 1, 1).c_str());
        REQUIRE_THROWS_WITH(read_multi_ligand_dms_full(tmp.path, flex, ff, sdf, type_props, false),
                            Catch::Matchers::ContainsSubstring("outside the ligand"));
    }

    SECTION("ring_row repeated, as in per-ligand files merged without renumbering") {
        exec_sql(tmp.path, (ring_row_sql(0, {0, 1, 2, 3, 4, 5}, 1, 1, 1, 0, 0, 1, 2.0)
                            + ring_row_sql(0, {35, 36, 37, 38, 39, 40}, 3, 3, 3, 1, 0, 0, 2.0)).c_str());
        REQUIRE_THROWS_WITH(read_multi_ligand_dms_full(tmp.path, flex, ff, sdf, type_props, false),
                            Catch::Matchers::ContainsSubstring("ring_row repeats"));
    }

    SECTION("rows that belong to no ligand") {
        for (const std::vector<int>& atoms : {std::vector<int>{}, std::vector<int>{70, 71, 72, 73, 74, 75}}){
            INFO("ring atoms " << atoms.size());
            exec_sql(tmp.path, "DELETE FROM ud2_ring_bias; DELETE FROM ud2_ring_bias_atoms;");
            exec_sql(tmp.path, (ring_row_sql(1, {0, 1, 2, 3, 4, 5}, 1, 1, 1, 0, 0, 1, 2.0)
                                + ring_row_sql(2, atoms, 0, 0, 0, 0, 0, 1, 1)).c_str());
            flex.clear(); ff.clear(); sdf.clear();
            REQUIRE_THROWS_WITH(read_multi_ligand_dms_full(tmp.path, flex, ff, sdf, type_props, false),
                                Catch::Matchers::ContainsSubstring("1 of 2 ring rows belong to no ligand"));
        }
    }
}
