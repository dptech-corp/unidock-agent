//
// Unit tests for reading ud2_bias table from enriched DMS files.
//

#include <catch2/catch_amalgamated.hpp>
#include "format/dms_reader.h"
#include "model/model.h"
#include "constants/constants.h"

#include <sqlite3.h>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

static const std::string DMS_TEST_DIR = std::string(TEST_DATA_DIR) + "ff_minimize/tyk2/";

// Helper: copy a file (binary safe)
static bool copy_file(const std::string& src, const std::string& dst) {
    std::ifstream  in(src,  std::ios::binary);
    std::ofstream out(dst, std::ios::binary);
    if (!in || !out) return false;
    out << in.rdbuf();
    return true;
}

// Helper: inject a ud2_bias table with known data into a DMS file
static void inject_bias_table(const std::string& dms_path,
                              const std::vector<int>& atom_ids,
                              const std::vector<float>& ref_x,
                              const std::vector<float>& ref_y,
                              const std::vector<float>& ref_z,
                              const std::vector<float>& v_set,
                              const std::vector<float>& r_param) {
    sqlite3* db = nullptr;
    REQUIRE(sqlite3_open(dms_path.c_str(), &db) == SQLITE_OK);

    char* errmsg = nullptr;
    REQUIRE(sqlite3_exec(db, "DROP TABLE IF EXISTS ud2_bias", nullptr, nullptr, &errmsg) == SQLITE_OK);
    REQUIRE(sqlite3_exec(db,
        "CREATE TABLE ud2_bias ("
        "atom_id INTEGER, ref_x REAL, ref_y REAL, ref_z REAL, "
        "v_set REAL, r REAL)",
        nullptr, nullptr, &errmsg) == SQLITE_OK);

    sqlite3_stmt* stmt = nullptr;
    REQUIRE(sqlite3_prepare_v2(db,
        "INSERT INTO ud2_bias VALUES (?, ?, ?, ?, ?, ?)",
        -1, &stmt, nullptr) == SQLITE_OK);

    for (size_t i = 0; i < atom_ids.size(); i++) {
        sqlite3_bind_int(stmt,    1, atom_ids[i]);
        sqlite3_bind_double(stmt, 2, ref_x[i]);
        sqlite3_bind_double(stmt, 3, ref_y[i]);
        sqlite3_bind_double(stmt, 4, ref_z[i]);
        sqlite3_bind_double(stmt, 5, v_set[i]);
        sqlite3_bind_double(stmt, 6, r_param[i]);
        REQUIRE(sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);
}


TEST_CASE("read_ligand_dms_full reads ud2_bias table", "[dms_bias]") {
    std::string src_path = DMS_TEST_DIR + "ligand_ejm42_enriched.dms";
    std::string tmp_path = "test_bias_tmp.dms";

    REQUIRE(copy_file(src_path, tmp_path));

    // Inject 3 bias entries: atoms 0, 0, 2
    std::vector<int>   ids   = {0,    0,    2};
    std::vector<float> rx    = {1.0f, 2.0f, 3.0f};
    std::vector<float> ry    = {4.0f, 5.0f, 6.0f};
    std::vector<float> rz    = {7.0f, 8.0f, 9.0f};
    std::vector<float> vs    = {0.5f, 0.3f, 0.8f};
    std::vector<float> rp    = {0.0f, 0.0f, 0.0f};

    inject_bias_table(tmp_path, ids, rx, ry, rz, vs, rp);

    UDFlexMol flex;
    FFMinData ff;
    AtomTypeProps type_props;
    init_default_atom_type_props(type_props);

    REQUIRE_NOTHROW(read_ligand_dms_full(tmp_path, flex, ff, type_props, false));

    SECTION("bias entries loaded") {
        REQUIRE(flex.bias.rows.size() == 3);
    }

    SECTION("bias atom indices correct") {
        REQUIRE(flex.bias.rows[0].i == 0);
        REQUIRE(flex.bias.rows[1].i == 0);
        REQUIRE(flex.bias.rows[2].i == 2);
    }

    SECTION("bias params correct") {
        // First entry
        REQUIRE(flex.bias.rows[0].param[0] == Catch::Approx(1.0f));
        REQUIRE(flex.bias.rows[0].param[1] == Catch::Approx(4.0f));
        REQUIRE(flex.bias.rows[0].param[2] == Catch::Approx(7.0f));
        REQUIRE(flex.bias.rows[0].param[3] == Catch::Approx(0.5f));
        REQUIRE(flex.bias.rows[0].param[4] == Catch::Approx(0.0f));

        // Third entry
        REQUIRE(flex.bias.rows[2].param[0] == Catch::Approx(3.0f));
        REQUIRE(flex.bias.rows[2].param[1] == Catch::Approx(6.0f));
        REQUIRE(flex.bias.rows[2].param[2] == Catch::Approx(9.0f));
        REQUIRE(flex.bias.rows[2].param[3] == Catch::Approx(0.8f));
    }

    // Clean up
    std::remove(tmp_path.c_str());
}


TEST_CASE("read_ligand_dms_full without ud2_bias table", "[dms_bias]") {
    // The original enriched DMS has no ud2_bias table — biases should be empty
    std::string path = DMS_TEST_DIR + "ligand_ejm42_enriched.dms";

    UDFlexMol flex;
    FFMinData ff;
    AtomTypeProps type_props;
    init_default_atom_type_props(type_props);

    REQUIRE_NOTHROW(read_ligand_dms_full(path, flex, ff, type_props, false));

    REQUIRE(flex.bias.rows.empty());
    // Other data should still be present
    REQUIRE(flex.natom == 35);
}


TEST_CASE("read_ligand_dms_full with empty ud2_bias table", "[dms_bias]") {
    std::string src_path = DMS_TEST_DIR + "ligand_ejm42_enriched.dms";
    std::string tmp_path = "test_bias_empty_tmp.dms";

    REQUIRE(copy_file(src_path, tmp_path));

    // Create empty ud2_bias table
    inject_bias_table(tmp_path, {}, {}, {}, {}, {}, {});

    UDFlexMol flex;
    FFMinData ff;
    AtomTypeProps type_props;
    init_default_atom_type_props(type_props);

    REQUIRE_NOTHROW(read_ligand_dms_full(tmp_path, flex, ff, type_props, false));

    REQUIRE(flex.bias.rows.empty());
    // Other data should still be present
    REQUIRE(flex.natom == 35);

    std::remove(tmp_path.c_str());
}


TEST_CASE("read_ligand_dms_full many bias entries per atom", "[dms_bias]") {
    std::string src_path = DMS_TEST_DIR + "ligand_ejm42_enriched.dms";
    std::string tmp_path = "test_bias_many_tmp.dms";

    REQUIRE(copy_file(src_path, tmp_path));

    // Inject 10 entries all for atom 0 (simulating N template atoms → 1 query atom)
    const int N = 10;
    std::vector<int>   ids(N, 0);
    std::vector<float> rx(N), ry(N), rz(N), vs(N), rp(N);
    for (int i = 0; i < N; i++) {
        rx[i] = static_cast<float>(i * 1.0);
        ry[i] = static_cast<float>(i * 2.0);
        rz[i] = static_cast<float>(i * 3.0);
        vs[i] = static_cast<float>(0.1 * (i + 1));
        rp[i] = 0.0f;
    }

    inject_bias_table(tmp_path, ids, rx, ry, rz, vs, rp);

    UDFlexMol flex;
    FFMinData ff;
    AtomTypeProps type_props;
    init_default_atom_type_props(type_props);

    REQUIRE_NOTHROW(read_ligand_dms_full(tmp_path, flex, ff, type_props, false));

    SECTION("all 10 entries loaded") {
        REQUIRE(flex.bias.rows.size() == 10);
    }

    SECTION("all entries refer to atom 0") {
        for (int i = 0; i < N; i++) {
            REQUIRE(flex.bias.rows[i].i == 0);
        }
    }

    SECTION("params are correct") {
        for (int i = 0; i < N; i++) {
            REQUIRE(flex.bias.rows[i].param[0] == Catch::Approx(i * 1.0f));
            REQUIRE(flex.bias.rows[i].param[1] == Catch::Approx(i * 2.0f));
            REQUIRE(flex.bias.rows[i].param[2] == Catch::Approx(i * 3.0f));
            REQUIRE(flex.bias.rows[i].param[3] == Catch::Approx(0.1f * (i + 1)));
        }
    }

    std::remove(tmp_path.c_str());
}


TEST_CASE("read_ligand_dms_full bias entries ordered by atom_id", "[dms_bias]") {
    std::string src_path = DMS_TEST_DIR + "ligand_ejm42_enriched.dms";
    std::string tmp_path = "test_bias_order_tmp.dms";

    REQUIRE(copy_file(src_path, tmp_path));

    // Inject entries for atoms {0, 0, 2, 2, 5} — grouped by atom_id
    std::vector<int>   ids = {0,    0,    2,    2,    5};
    std::vector<float> rx  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> ry  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> rz  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vs  = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    std::vector<float> rp  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    inject_bias_table(tmp_path, ids, rx, ry, rz, vs, rp);

    UDFlexMol flex;
    FFMinData ff;
    AtomTypeProps type_props;
    init_default_atom_type_props(type_props);

    REQUIRE_NOTHROW(read_ligand_dms_full(tmp_path, flex, ff, type_props, false));

    SECTION("correct count") {
        REQUIRE(flex.bias.rows.size() == 5);
    }

    SECTION("ordering preserved: atom_ids are non-decreasing") {
        for (size_t i = 1; i < flex.bias.rows.size(); i++) {
            REQUIRE(flex.bias.rows[i].i >= flex.bias.rows[i - 1].i);
        }
    }

    SECTION("grouping correct") {
        // Atoms 0: entries 0,1  Atoms 2: entries 2,3  Atom 5: entry 4
        REQUIRE(flex.bias.rows[0].i == 0);
        REQUIRE(flex.bias.rows[1].i == 0);
        REQUIRE(flex.bias.rows[2].i == 2);
        REQUIRE(flex.bias.rows[3].i == 2);
        REQUIRE(flex.bias.rows[4].i == 5);
    }

    SECTION("params match injected values") {
        REQUIRE(flex.bias.rows[0].param[0] == Catch::Approx(1.0f));
        REQUIRE(flex.bias.rows[1].param[0] == Catch::Approx(2.0f));
        REQUIRE(flex.bias.rows[4].param[0] == Catch::Approx(5.0f));
        REQUIRE(flex.bias.rows[4].param[3] == Catch::Approx(0.5f));
    }

    std::remove(tmp_path.c_str());
}
