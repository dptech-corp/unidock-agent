//
// ud2_ring_bias / ud2_ring_bias_atoms reader; the October read_ring_bias / count_merged_ring_bias_rows of
// format/dms_reader.cpp (feat/ud2align-ring-bias) and its orphan check after the per-ligand loop of the merged read,
// same SQL and column reads, now a registry entry (format/dms_reader.h register_table_reader).
//

#include "ext/bias/ring_reader.h"

#include <sqlite3.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/// RAII prepared statement (the subset of format/dms_reader.cpp SqliteStmt the reader uses: the same sqlite calls).
struct Stmt {
    sqlite3_stmt* stmt = nullptr;
    Stmt(sqlite3* db, const std::string& sql) {
        if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("SQL prepare failed: " + sql + " (" + sqlite3_errmsg(db) + ")");
        }
    }
    ~Stmt() { if (stmt) sqlite3_finalize(stmt); }
    bool step() { return sqlite3_step(stmt) == SQLITE_ROW; }
    int col_int(int i)    { return sqlite3_column_int(stmt, i); }
    double col_dbl(int i) { return sqlite3_column_double(stmt, i); }
    Stmt(const Stmt&) = delete;
    Stmt& operator=(const Stmt&) = delete;
};

} // namespace

void read_ring_bias_rows(sqlite3* db, const TableScope& scope, UDFlexMol& out) {
    std::vector<RingBiasRow>& out_rows = out.ring.rows;
    const int off = scope.off, natom = scope.natom;
    out_rows.clear();
    const bool has_rows = dms_has_table(db, "ud2_ring_bias");
    const bool has_atoms = dms_has_table(db, "ud2_ring_bias_atoms");
    if (!has_rows && !has_atoms) return;
    if (!has_rows || !has_atoms) {
        throw std::runtime_error("DMS has only one of the ring-bias tables ud2_ring_bias / ud2_ring_bias_atoms");
    }
    std::string sql =
        "SELECT r.ring_row, r.cx, r.cy, r.cz, r.nx, r.ny, r.nz, r.v_set, a.atom_id, a.atom_id IS NULL "
        "FROM ud2_ring_bias r LEFT JOIN ud2_ring_bias_atoms a ON a.ring_row = r.ring_row";
    if (scope.merged) {
        // format/dms_reader.cpp range_clause("atom_id", off, natom)
        sql += " WHERE r.ring_row IN (SELECT ring_row FROM ud2_ring_bias_atoms"
             + std::string(" WHERE atom_id >= ") + std::to_string(off)
             + " AND atom_id < " + std::to_string(off + natom) + ")";
    }
    sql += " ORDER BY r.ring_row, a.pos";

    std::vector<int> ids;  // ring_row of each entry of out_rows
    std::vector<int> counts;
    Stmt q(db, sql);
    while (q.step()) {
        const int id = q.col_int(0);
        if (ids.empty() || ids.back() != id) {
            const double nx = q.col_dbl(4), ny = q.col_dbl(5), nz = q.col_dbl(6);
            const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len < 1e-6) {
                throw std::runtime_error("ud2_ring_bias: ring_row " + std::to_string(id) + " has a zero template normal");
            }
            RingBiasRow r;
            for (int k = 0; k < 3; k++) r.c[k] = static_cast<Real>(q.col_dbl(1 + k));
            r.nt[0] = static_cast<Real>(nx / len);
            r.nt[1] = static_cast<Real>(ny / len);
            r.nt[2] = static_cast<Real>(nz / len);
            r.v_set = static_cast<Real>(q.col_dbl(7));
            out_rows.push_back(r);
            ids.push_back(id);
            counts.push_back(0);
        }
        if (q.col_int(9)) continue;  // a ring row without atoms
        const int k = counts.back()++;
        const int atom = q.col_int(8) - off;
        if (atom < 0 || atom >= natom) {
            throw std::runtime_error("ud2_ring_bias_atoms: ring_row " + std::to_string(id) + " has atom_id "
                                     + std::to_string(atom + off) + " outside the ligand's atoms ["
                                     + std::to_string(off) + ", " + std::to_string(off + natom) + ")");
        }
        if (k < 6) out_rows.back().atoms[k] = atom;
    }
    for (size_t i = 0; i < out_rows.size(); i++) {
        if (counts[i] < 5 || counts[i] > 6) {
            throw std::runtime_error("ud2_ring_bias: ring_row " + std::to_string(ids[i]) + " has "
                                     + std::to_string(counts[i]) + " atoms; ring bias needs a 5- or 6-membered ring");
        }
        out_rows[i].n = counts[i];
    }
}

int count_merged_ring_bias_rows(sqlite3* db) {
    if (!dms_has_table(db, "ud2_ring_bias")) return 0;
    Stmt q(db, "SELECT COUNT(*), COUNT(DISTINCT ring_row) FROM ud2_ring_bias");
    q.step();
    if (q.col_int(0) != q.col_int(1)) {
        throw std::runtime_error("ud2_ring_bias: ring_row repeats; in a merged DMS ring_row must be unique across all ligands");
    }
    return q.col_int(0);
}

namespace {

/// Before the first ligand of a merged read: ring_row unique across the file.
void check_ring_rows_unique(sqlite3* db, const UDFlexMol*, size_t) {
    count_merged_ring_bias_rows(db);
}

/// After the last ligand: every ring row was read by some ligand.
void check_ring_rows_owned(sqlite3* db, const UDFlexMol* ligands, size_t n) {
    const int n_ring_rows = count_merged_ring_bias_rows(db);
    int n_ring_read = 0;
    for (size_t i = 0; i < n; i++) n_ring_read += static_cast<int>(ligands[i].ring.rows.size());
    if (n_ring_read != n_ring_rows) {
        throw std::runtime_error("ud2_ring_bias: " + std::to_string(n_ring_rows - n_ring_read) + " of "
                                 + std::to_string(n_ring_rows)
                                 + " ring rows belong to no ligand (no atoms, or atoms outside every ligand)");
    }
}

// Registers the reader before main runs, after ud2_bias (ext/bias/bias_reader.cpp precedes this file in ud_lib's
// source list, src/CMakeLists.txt, so its registrar runs first). ud_lib is an OBJECT library: every object file of it
// is linked, so this registration cannot be dropped.
struct RingTableRegistrar {
    RingTableRegistrar() {
        TableReaderEntry entry;
        entry.table = "ud2_ring_bias";
        entry.companion = "ud2_ring_bias_atoms";
        entry.read = read_ring_bias_rows;
        entry.before_merged = check_ring_rows_unique;
        entry.after_merged = check_ring_rows_owned;
        register_table_reader(entry);
    }
} ring_table_registrar;

} // namespace
