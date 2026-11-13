//
// ud2_bias table reader; formerly "8. Read position-bias data" of format/dms_reader.cpp (both the single-ligand
// read and the per-ligand loop of the merged read), same SQL, same column reads, same rowid order.
//

#include "ext/bias/bias_reader.h"

#include <sqlite3.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

void read_bias_rows(sqlite3* db, const TableScope& scope, UDFlexMol& out){
    std::string sql = "SELECT atom_id, ref_x, ref_y, ref_z, v_set, r FROM ud2_bias";
    if (scope.merged){
        sql += " WHERE atom_id >= " + std::to_string(scope.off) + " AND atom_id < " + std::to_string(scope.off + scope.natom);
    }
    sql += " ORDER BY rowid";

    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK){
        throw std::runtime_error("SQL prepare failed: " + sql + " (" + sqlite3_errmsg(db) + ")");
    }
    while (sqlite3_step(stmt) == SQLITE_ROW){
        Bias b;
        b.i = sqlite3_column_int(stmt, 0) - scope.off;
        b.param[0] = static_cast<Real>(sqlite3_column_double(stmt, 1));  // ref_x
        b.param[1] = static_cast<Real>(sqlite3_column_double(stmt, 2));  // ref_y
        b.param[2] = static_cast<Real>(sqlite3_column_double(stmt, 3));  // ref_z
        b.param[3] = static_cast<Real>(sqlite3_column_double(stmt, 4));  // V_set (similarity)
        b.param[4] = static_cast<Real>(sqlite3_column_double(stmt, 5));  // r (width param)
        out.bias.rows.push_back(b);
    }
    sqlite3_finalize(stmt);
    if (!scope.merged){
        spdlog::info("  Position-bias entries from DMS: {}", out.bias.rows.size());
    }
}

namespace {
// Registers the reader before main runs. ud_lib is an OBJECT library (src/CMakeLists.txt): every object file of
// it is on the link line of the binary, the Python module and the unit tests, so this translation unit, and with
// it the registration, cannot be dropped the way a member of a static archive can.
struct BiasTableRegistrar {
    BiasTableRegistrar(){ register_table_reader("ud2_bias", read_bias_rows); }
} bias_table_registrar;
} // namespace
