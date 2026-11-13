//
// The DMS reader's extension-table registry (format/dms_reader.h register_table_reader) and its two entries, the
// ud2_bias reader (ext/bias/bias_reader.cpp) and the ring-bias reader (ext/bias/ring_reader.cpp): a DMS without the
// table yields no rows, with it the rows in rowid order; the merged read hands every ligand its own scope (offset,
// atom count) and the rows come back rebased; a ligand's ring rows in a merged file are its single-file rows.
//

#include <catch2/catch_amalgamated.hpp>
#include "format/dms_reader.h"
#include "ext/bias/bias_reader.h"
#include "ext/bias/ring_reader.h"
#include "model/model.h"
#include "constants/constants.h"

#include <sqlite3.h>
#include <unistd.h>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

const std::string DMS_TEST_DIR = std::string(TEST_DATA_DIR) + "ff_minimize/tyk2/";

/// Copy of the tyk2 enriched ligand DMS (35 atoms) in the temp directory, removed with the guard
struct TmpDms {
    const std::string path;
    explicit TmpDms(const char* label)
        : path((std::filesystem::temp_directory_path()
                / ("test_table_registry_" + std::string(label) + "_" + std::to_string(static_cast<int>(getpid())) + ".dms")).string()) {
        std::filesystem::copy_file(DMS_TEST_DIR + "ligand_ejm42_enriched.dms", path,
                                   std::filesystem::copy_options::overwrite_existing);
    }
    ~TmpDms() { std::remove(path.c_str()); }
};

void exec_sql(const std::string& path, const std::string& sql) {
    sqlite3* db = nullptr;
    REQUIRE(sqlite3_open(path.c_str(), &db) == SQLITE_OK);
    const int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, nullptr);
    INFO("sqlite: " << sqlite3_errmsg(db));
    sqlite3_close(db);
    REQUIRE(rc == SQLITE_OK);
}

/// ud2_bias rows (atom_id, ref_x, ref_y, ref_z, v_set, r), inserted in the order given (= rowid order)
void inject_bias(const std::string& path, const std::vector<std::vector<double>>& rows) {
    std::string sql = "DROP TABLE IF EXISTS ud2_bias;"
                      "CREATE TABLE ud2_bias (atom_id INTEGER, ref_x REAL, ref_y REAL, ref_z REAL, v_set REAL, r REAL);";
    for (const auto& r : rows) {
        sql += "INSERT INTO ud2_bias VALUES (" + std::to_string(static_cast<int>(r[0]));
        for (size_t k = 1; k < 6; k++) sql += ", " + std::to_string(r[k]);
        sql += ");";
    }
    exec_sql(path, sql);
}

/// Two-ligand merge of the tyk2 ligand (the layout of the Python merge, as test_dms_reader.cpp builds it):
/// atoms 0-34 are ligand 0, 35-69 the same molecule again as ligand 1. Torsion rows are left out (rigid ligands).
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

// A probe reader registered by this test file for a table no real DMS carries: records every call
std::vector<TableScope> probe_calls;
void probe_reader(sqlite3*, const TableScope& scope, UDFlexMol& out) {
    probe_calls.push_back(scope);
    out.name += "+probe";
}
struct ProbeRegistrar {
    ProbeRegistrar() { register_table_reader("ud2_registry_probe", probe_reader); }
} probe_registrar;

int count_registered(const char* table) {
    int n = 0;
    for (const auto& entry : registered_table_readers()) if (entry.table == table) n++;
    return n;
}

} // namespace


TEST_CASE("table registry: ud2_bias is registered once, by ext/bias/bias_reader.cpp", "[table_registry]") {
    REQUIRE(count_registered("ud2_bias") == 1);
    REQUIRE(count_registered("ud2_registry_probe") == 1);
    bool found = false;
    for (const auto& entry : registered_table_readers()) {
        if (entry.table == "ud2_bias") {
            found = true;
            REQUIRE(entry.read == &read_bias_rows);
            REQUIRE(entry.companion.empty());
            REQUIRE(entry.before_merged == nullptr);
            REQUIRE(entry.after_merged == nullptr);
        }
    }
    REQUIRE(found);
}


TEST_CASE("table registry: a DMS without the tables gives no rows and no reader call", "[table_registry]") {
    probe_calls.clear();
    UDFlexMol flex; FFMinData ff; AtomTypeProps atp; init_default_atom_type_props(atp);
    REQUIRE_NOTHROW(read_ligand_dms_full(DMS_TEST_DIR + "ligand_ejm42_enriched.dms", flex, ff, atp, false));
    REQUIRE(flex.natom == 35);
    REQUIRE(flex.bias.rows.empty());
    REQUIRE(probe_calls.empty());
    REQUIRE(flex.name.find("+probe") == std::string::npos);
}


TEST_CASE("table registry: single-ligand read calls the registered readers after the core tables, rows in rowid order", "[table_registry]") {
    TmpDms tmp("single");
    // rowid order is not atom order: atom 5 first, then two rows on atom 0, then atom 2
    inject_bias(tmp.path, {{5, 1.0, 2.0, 3.0, 0.5, 0.0}, {0, 4.0, 5.0, 6.0, 0.3, 0.0}, {0, 7.0, 8.0, 9.0, 0.8, 1.0}, {2, 1.5, 2.5, 3.5, 0.1, 0.0}});
    exec_sql(tmp.path, "CREATE TABLE ud2_registry_probe (x INTEGER); INSERT INTO ud2_registry_probe VALUES (1);");
    probe_calls.clear();

    UDFlexMol flex; FFMinData ff; AtomTypeProps atp; init_default_atom_type_props(atp);
    REQUIRE_NOTHROW(read_ligand_dms_full(tmp.path, flex, ff, atp, false));
    REQUIRE(flex.natom == 35);                                        // the core tables were read

    REQUIRE(flex.bias.rows.size() == 4);
    const int expected_atoms[4] = {5, 0, 0, 2};
    const float expected_x[4] = {1.0f, 4.0f, 7.0f, 1.5f};
    for (int i = 0; i < 4; i++) {
        REQUIRE(flex.bias.rows[i].i == expected_atoms[i]);
        REQUIRE(flex.bias.rows[i].param[0] == Catch::Approx(expected_x[i]));
    }
    REQUIRE(flex.bias.rows[2].param[4] == Catch::Approx(1.0f));

    REQUIRE(probe_calls.size() == 1);                                 // once, with the whole-file scope
    REQUIRE(probe_calls[0].off == 0);
    REQUIRE(probe_calls[0].natom == 35);
    REQUIRE(!probe_calls[0].merged);
    REQUIRE(flex.name.find("+probe") != std::string::npos);
}


TEST_CASE("table registry: merged read hands every ligand its scope; bias rows filtered and rebased per ligand", "[table_registry]") {
    TmpDms tmp("merged");
    exec_sql(tmp.path, SQL_MERGE_TWO);
    // ligand 1's rows (atom ids 35..69) interleaved with ligand 0's in rowid order
    inject_bias(tmp.path, {{36, 1.0, 0.0, 0.0, 0.9, 0.0}, {3, 2.0, 0.0, 0.0, 0.8, 0.0}, {35, 3.0, 0.0, 0.0, 0.7, 0.0},
                           {3, 4.0, 0.0, 0.0, 0.6, 0.0}, {69, 5.0, 0.0, 0.0, 0.5, 0.0}, {0, 6.0, 0.0, 0.0, 0.4, 0.0}});
    exec_sql(tmp.path, "CREATE TABLE ud2_registry_probe (x INTEGER);");
    probe_calls.clear();

    UDFlexMolList flex; std::vector<FFMinData> ff; std::vector<SDFMolData> sdf;
    AtomTypeProps atp; init_default_atom_type_props(atp);
    REQUIRE_NOTHROW(read_multi_ligand_dms_full(tmp.path, flex, ff, sdf, atp, false));
    REQUIRE(flex.size() == 2);
    REQUIRE(flex[0].natom == 35);
    REQUIRE(flex[1].natom == 35);

    // ligand 0: rows with atom_id in [0, 35) in rowid order: 3, 3, 0
    REQUIRE(flex[0].bias.rows.size() == 3);
    REQUIRE(flex[0].bias.rows[0].i == 3); REQUIRE(flex[0].bias.rows[0].param[0] == Catch::Approx(2.0f));
    REQUIRE(flex[0].bias.rows[1].i == 3); REQUIRE(flex[0].bias.rows[1].param[0] == Catch::Approx(4.0f));
    REQUIRE(flex[0].bias.rows[2].i == 0); REQUIRE(flex[0].bias.rows[2].param[0] == Catch::Approx(6.0f));
    // ligand 1: rows with atom_id in [35, 70), rebased by 35, in rowid order: 36 -> 1, 35 -> 0, 69 -> 34
    REQUIRE(flex[1].bias.rows.size() == 3);
    REQUIRE(flex[1].bias.rows[0].i == 1);  REQUIRE(flex[1].bias.rows[0].param[0] == Catch::Approx(1.0f));
    REQUIRE(flex[1].bias.rows[1].i == 0);  REQUIRE(flex[1].bias.rows[1].param[0] == Catch::Approx(3.0f));
    REQUIRE(flex[1].bias.rows[2].i == 34); REQUIRE(flex[1].bias.rows[2].param[0] == Catch::Approx(5.0f));
    REQUIRE(flex[1].bias.rows[2].param[3] == Catch::Approx(0.5f));

    REQUIRE(probe_calls.size() == 2);
    REQUIRE(probe_calls[0].off == 0);  REQUIRE(probe_calls[0].natom == 35); REQUIRE(probe_calls[0].merged);
    REQUIRE(probe_calls[1].off == 35); REQUIRE(probe_calls[1].natom == 35); REQUIRE(probe_calls[1].merged);
}


TEST_CASE("read_bias_rows: the scope selects the SQL (whole table vs the ligand's id range, rebased)", "[table_registry][bias_reader]") {
    TmpDms tmp("scope");
    inject_bias(tmp.path, {{40, 1.0, 0.0, 0.0, 0.9, 0.0}, {2, 2.0, 0.0, 0.0, 0.8, 0.0}, {37, 3.0, 0.0, 0.0, 0.7, 0.0}, {70, 4.0, 0.0, 0.0, 0.6, 0.0}});
    sqlite3* db = nullptr;
    REQUIRE(sqlite3_open_v2(tmp.path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) == SQLITE_OK);

    UDFlexMol whole;
    read_bias_rows(db, TableScope{0, 35, false}, whole);              // single-ligand file: every row, ids as stored
    REQUIRE(whole.bias.rows.size() == 4);
    REQUIRE(whole.bias.rows[0].i == 40);
    REQUIRE(whole.bias.rows[3].i == 70);

    UDFlexMol second;
    read_bias_rows(db, TableScope{35, 35, true}, second);             // merged file, ligand at offset 35: [35, 70)
    REQUIRE(second.bias.rows.size() == 2);
    REQUIRE(second.bias.rows[0].i == 5);  REQUIRE(second.bias.rows[0].param[0] == Catch::Approx(1.0f));
    REQUIRE(second.bias.rows[1].i == 2);  REQUIRE(second.bias.rows[1].param[0] == Catch::Approx(3.0f));

    UDFlexMol appended = second;                                      // rows are appended, never cleared
    read_bias_rows(db, TableScope{0, 35, true}, appended);
    REQUIRE(appended.bias.rows.size() == 3);
    REQUIRE(appended.bias.rows[2].i == 2);
    sqlite3_close(db);
}


// ==================== the ring-bias entry (plan step 6) ====================

namespace {

/// One ring row (ring_row, centroid, normal, v_set) and its atom rows in pos order
std::string ring_sql(int ring_row, const std::vector<int>& atoms, double cx, double nz, double v_set) {
    std::string sql = "INSERT INTO ud2_ring_bias VALUES (" + std::to_string(ring_row) + ", " + std::to_string(cx)
                    + ", 0.5, -1.0, 0.0, 0.0, " + std::to_string(nz) + ", " + std::to_string(v_set) + ");";
    for (size_t k = 0; k < atoms.size(); k++) {
        sql += "INSERT INTO ud2_ring_bias_atoms VALUES (" + std::to_string(ring_row) + ", " + std::to_string(k) + ", "
             + std::to_string(atoms[k]) + ");";
    }
    return sql;
}

const char* SQL_RING_TABLES =
    "CREATE TABLE ud2_ring_bias (ring_row INTEGER, cx REAL, cy REAL, cz REAL, nx REAL, ny REAL, nz REAL, v_set REAL);"
    "CREATE TABLE ud2_ring_bias_atoms (ring_row INTEGER, pos INTEGER, atom_id INTEGER);";

bool same_rows(const std::vector<RingBiasRow>& a, const std::vector<RingBiasRow>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].n != b[i].n || a[i].v_set != b[i].v_set) return false;
        for (int k = 0; k < 6; k++) if (a[i].atoms[k] != b[i].atoms[k]) return false;
        for (int j = 0; j < 3; j++) if (a[i].c[j] != b[i].c[j] || a[i].nt[j] != b[i].nt[j]) return false;
    }
    return true;
}

} // namespace


TEST_CASE("table registry: the ring-bias reader is the second entry, with its companion table and file checks", "[table_registry][ring_bias]") {
    REQUIRE(count_registered("ud2_ring_bias") == 1);
    REQUIRE(count_registered("ud2_ring_bias_atoms") == 0);            // the companion, not an entry of its own
    int i_bias = -1, i_ring = -1, i = 0;
    for (const auto& entry : registered_table_readers()) {
        if (entry.table == "ud2_bias") i_bias = i;
        if (entry.table == "ud2_ring_bias") {
            i_ring = i;
            REQUIRE(entry.read == &read_ring_bias_rows);
            REQUIRE(entry.companion == "ud2_ring_bias_atoms");
            REQUIRE(entry.before_merged != nullptr);
            REQUIRE(entry.after_merged != nullptr);
        }
        i++;
    }
    REQUIRE(i_bias >= 0);
    REQUIRE(i_ring > i_bias);                                          // read after the position bias, as in October
}


TEST_CASE("table registry: a ligand's ring rows in a merged DMS are its single-file rows, atoms shifted by its offset", "[table_registry][ring_bias]") {
    // single-ligand file: two rings (ring_row 0, 1), atoms in ligand numbering
    TmpDms single("ring_single");
    exec_sql(single.path, SQL_RING_TABLES);
    exec_sql(single.path, ring_sql(0, {0, 1, 2, 3, 4, 5}, 1.0, 1.0, 4.0) + ring_sql(1, {10, 11, 12, 13, 14}, -2.0, 2.0, 2.0));
    UDFlexMol one; FFMinData ff1; AtomTypeProps atp; init_default_atom_type_props(atp);
    REQUIRE_NOTHROW(read_ligand_dms_full(single.path, one, ff1, atp, false));
    REQUIRE(one.ring.rows.size() == 2);
    REQUIRE(one.ring.rows[1].n == 5);
    REQUIRE(one.ring.rows[1].atoms[4] == 14);

    // the same rows as ligand 1 of a two-ligand merge (the layout BiasPlugin.annotate_ligand_dms writes): ring_row
    // continued after ligand 0's (none here: ligand 0 carries one row of its own first), atom ids + 35
    TmpDms merged("ring_merged");
    exec_sql(merged.path, SQL_MERGE_TWO);
    exec_sql(merged.path, SQL_RING_TABLES);
    exec_sql(merged.path, ring_sql(0, {20, 21, 22, 23, 24, 25}, 3.0, 1.0, 1.0)
                          + ring_sql(1, {35, 36, 37, 38, 39, 40}, 1.0, 1.0, 4.0)
                          + ring_sql(2, {45, 46, 47, 48, 49}, -2.0, 2.0, 2.0));
    UDFlexMolList flex; std::vector<FFMinData> ff; std::vector<SDFMolData> sdf;
    REQUIRE_NOTHROW(read_multi_ligand_dms_full(merged.path, flex, ff, sdf, atp, false));
    REQUIRE(flex.size() == 2);
    REQUIRE(flex[0].ring.rows.size() == 1);
    REQUIRE(flex[0].ring.rows[0].atoms[0] == 20);
    REQUIRE(same_rows(flex[1].ring.rows, one.ring.rows));
    REQUIRE(flex[1].bias.rows.empty());
}
