//
// DMS reader implementation using SQLite3 amalgamation.
//

#include "dms_reader.h"
#include "score/vina.h"
#include <sqlite3.h>
#include <spdlog/spdlog.h>
#include <cmath>
#include <stdexcept>
#include <set>
#include <numeric>
#include <algorithm>
#include <unordered_map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

/// RAII wrapper for sqlite3 database handle.
struct SqliteDB {
    sqlite3* db = nullptr;

    SqliteDB(const std::string& path) {
        int rc = sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr);
        if (rc != SQLITE_OK) {
            std::string err = sqlite3_errmsg(db);
            sqlite3_close(db);
            db = nullptr;
            throw std::runtime_error("Failed to open DMS file: " + path + " (" + err + ")");
        }
    }

    ~SqliteDB() {
        if (db) sqlite3_close(db);
    }

    // Non-copyable
    SqliteDB(const SqliteDB&) = delete;
    SqliteDB& operator=(const SqliteDB&) = delete;
};

/// RAII wrapper for sqlite3 prepared statement.
struct SqliteStmt {
    sqlite3_stmt* stmt = nullptr;

    SqliteStmt(sqlite3* db, const std::string& sql) {
        int rc = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            throw std::runtime_error(
                "SQL prepare failed: " + sql + " (" + sqlite3_errmsg(db) + ")");
        }
    }

    ~SqliteStmt() {
        if (stmt) sqlite3_finalize(stmt);
    }

    bool step() {
        return sqlite3_step(stmt) == SQLITE_ROW;
    }

    int col_int(int i)    { return sqlite3_column_int(stmt, i); }
    double col_dbl(int i) { return sqlite3_column_double(stmt, i); }
    const char* col_str(int i) {
        const unsigned char* s = sqlite3_column_text(stmt, i);
        return s ? reinterpret_cast<const char*>(s) : "";
    }

    // Non-copyable
    SqliteStmt(const SqliteStmt&) = delete;
    SqliteStmt& operator=(const SqliteStmt&) = delete;
};

/// " WHERE col >= off AND col < off + natom", so a per-ligand read of a MERGED multi-ligand
/// DMS touches only its own rows. Without it each of these queries scans the whole table once
/// per ligand, which is quadratic in the ligand count: on a 7316-ligand screening file the
/// loader spent 324 s of a 1740 s run. Integers only, formatted in, so no binding is needed.
static std::string range_clause(const char* col, int off, int natom){
    return std::string(" WHERE ") + col + " >= " + std::to_string(off)
         + " AND " + col + " < " + std::to_string(off + natom);
}

/// Read per-atom FF parameters (charge, LJ sigma/epsilon, anum, coords).
void read_atom_params(sqlite3* db, FFAtomParams& out) {
    // Join particle with nonbonded_param via nbtype
    SqliteStmt q(db,
        "SELECT p.id, p.anum, p.charge, p.x, p.y, p.z, np.sigma, np.epsilon "
        "FROM particle p "
        "JOIN nonbonded_param np ON p.nbtype = np.id "
        "ORDER BY p.id");

    while (q.step()) {
        // int id = q.col_int(0);  // DMS atom index (should match vector index)
        out.anum.push_back(q.col_int(1));
        out.charge.push_back(static_cast<Real>(q.col_dbl(2)));
        out.x.push_back(static_cast<Real>(q.col_dbl(3)));
        out.y.push_back(static_cast<Real>(q.col_dbl(4)));
        out.z.push_back(static_cast<Real>(q.col_dbl(5)));
        out.sigma.push_back(static_cast<Real>(q.col_dbl(6)));
        out.epsilon.push_back(static_cast<Real>(q.col_dbl(7)));
    }
    out.natom = static_cast<int>(out.anum.size());
}

/// Number of rows in the particle table.
int count_particles(sqlite3* db) {
    SqliteStmt q(db, "SELECT COUNT(*) FROM particle");
    return q.step() ? q.col_int(0) : 0;
}

} // anonymous namespace

/// Check whether a SQLite table exists in the database.
static bool table_exists(sqlite3* db, const char* table_name) {
    std::string sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='";
    sql += table_name;
    sql += "'";
    SqliteStmt q(db, sql);
    return q.step();
}


// ===================== Extension table registry =====================

static std::vector<TableReaderEntry>& table_registry() {
    static std::vector<TableReaderEntry> registry;   // function-local: valid during static init
    return registry;
}

void register_table_reader(const char* table, TableReader fn) {
    TableReaderEntry entry;
    entry.table = table;
    entry.read = fn;
    table_registry().push_back(entry);
}

void register_table_reader(const TableReaderEntry& entry) {
    table_registry().push_back(entry);
}

const std::vector<TableReaderEntry>& registered_table_readers() {
    return table_registry();
}

bool dms_has_table(sqlite3* db, const char* table) {
    return table_exists(db, table);
}

/// The entry's table, or its companion, is in the file.
static bool entry_present(sqlite3* db, const TableReaderEntry& entry) {
    return table_exists(db, entry.table.c_str())
        || (!entry.companion.empty() && table_exists(db, entry.companion.c_str()));
}

/// The registered extension tables of one ligand, each when the table (or its companion) exists.
static void read_registered_tables(sqlite3* db, const TableScope& scope, UDFlexMol& out) {
    for (const auto& entry : table_registry()) {
        if (entry_present(db, entry)) entry.read(db, scope, out);
    }
}

/// The whole-file checks of a merged read: before the first ligand (ligands nullptr) or after the last.
static void check_registered_tables(sqlite3* db, bool before, const UDFlexMol* ligands, size_t n) {
    for (const auto& entry : table_registry()) {
        const MergedTableCheck check = before ? entry.before_merged : entry.after_merged;
        if (check != nullptr && entry_present(db, entry)) check(db, ligands, n);
    }
}

void read_ligand_dms(const std::string& dms_path, FFMinData& out, bool warn_unpopulated) {
    spdlog::info("Reading ligand DMS: {}", dms_path);
    SqliteDB db(dms_path);
    // No FF: loaded = false and no ligand FF data, whatever `out` held before
    auto drop_ligand_ff = [&out]() {
        out.lig_atoms = FFAtomParams();
        out.lig_pairs_14.clear();
        out.lig_torsions.clear();
        out.lig_exclusions.clear();
        out.lig_bonds.clear();
        out.lig_angles.clear();
        out.pair_type.clear();
        out.loaded = false;
    };

    // If core FF tables are absent (no antechamber/GAFF2 parameterization),
    // skip all FF loading.
    if (!table_exists(db.db, "nonbonded_param")) {
        spdlog::info("  No FF tables (nonbonded_param absent); FF scoring disabled for this ligand");
        drop_ligand_ff();
        return;
    }

    // 1. Per-atom params. The FF arrays are index-aligned with the flex atoms, so every particle
    // needs nonbonded parameters. An empty table is a deliberately Vina-only DMS (info, like an
    // absent one); a partial table is a broken parameterisation (warning).
    read_atom_params(db.db, out.lig_atoms);
    const int n_particle = count_particles(db.db);
    if (out.lig_atoms.natom == 0) {
        spdlog::info("  No FF parameters (nonbonded_param empty); FF scoring disabled for this ligand");
        drop_ligand_ff();
        return;
    }
    if (out.lig_atoms.natom != n_particle) {
        const std::string msg = fmt::format("  Ligand DMS {}: nonbonded parameters for {} of {} atoms; "
                                            "FF scoring disabled for this ligand",
                                            dms_path, out.lig_atoms.natom, n_particle);
        if (warn_unpopulated) spdlog::warn(msg);
        else spdlog::info(msg);
        drop_ligand_ff();
        return;
    }
    spdlog::info("  Ligand atoms: {}", out.lig_atoms.natom);

    // 2. 1-4 pairs (pre-scaled)
    if (table_exists(db.db, "pair_12_6_es_term")) {
        SqliteStmt q(db.db,
            "SELECT t.p0, t.p1, p.aij, p.bij, p.qij "
            "FROM pair_12_6_es_term t "
            "JOIN pair_12_6_es_param p ON t.param = p.id");

        while (q.step()) {
            FFPair14 pair;
            pair.i   = q.col_int(0);
            pair.j   = q.col_int(1);
            pair.aij = static_cast<Real>(q.col_dbl(2));
            pair.bij = static_cast<Real>(q.col_dbl(3));
            pair.qij = static_cast<Real>(q.col_dbl(4));
            out.lig_pairs_14.push_back(pair);
        }
    }
    spdlog::info("  Ligand 1-4 pairs: {}", out.lig_pairs_14.size());

    // 3. Torsion parameters
    if (table_exists(db.db, "dihedral_trig_term")) {
        SqliteStmt q(db.db,
            "SELECT t.p0, t.p1, t.p2, t.p3, "
            "p.phi0, p.fc0, p.fc1, p.fc2, p.fc3, p.fc4, p.fc5, p.fc6 "
            "FROM dihedral_trig_term t "
            "JOIN dihedral_trig_param p ON t.param = p.id");

        while (q.step()) {
            FFTorsion tor;
            tor.p0 = q.col_int(0);
            tor.p1 = q.col_int(1);
            tor.p2 = q.col_int(2);
            tor.p3 = q.col_int(3);
            tor.phi0 = static_cast<Real>(q.col_dbl(4) * M_PI / 180.0);  // degrees -> radians
            for (int k = 0; k < 7; k++) {
                tor.fc[k] = static_cast<Real>(q.col_dbl(5 + k));
            }
            out.lig_torsions.push_back(tor);
        }
    }
    spdlog::info("  Ligand torsions: {}", out.lig_torsions.size());

    // 4. Exclusions (1-2 and 1-3 pairs)
    if (table_exists(db.db, "exclusion")) {
        SqliteStmt q(db.db, "SELECT p0, p1 FROM exclusion");
        while (q.step()) {
            FFExclusion ex;
            ex.i = q.col_int(0);
            ex.j = q.col_int(1);
            out.lig_exclusions.push_back(ex);
        }
    }
    spdlog::info("  Ligand exclusions: {}", out.lig_exclusions.size());

    // 5. Bond harmonic parameters (optional; legacy DMS may lack stretch_harm_*)
    if (table_exists(db.db, "stretch_harm_term")) {
        SqliteStmt q(db.db,
            "SELECT t.p0, t.p1, p.r0, p.fc "
            "FROM stretch_harm_term t "
            "JOIN stretch_harm_param p ON t.param = p.id");
        while (q.step()) {
            FFBond b;
            b.i  = q.col_int(0);
            b.j  = q.col_int(1);
            b.r0 = static_cast<Real>(q.col_dbl(2));
            b.fc = static_cast<Real>(q.col_dbl(3));
            out.lig_bonds.push_back(b);
        }
    }
    spdlog::info("  Ligand bonds: {}", out.lig_bonds.size());

    // 6. Angle harmonic parameters (optional; legacy DMS may lack angle_harm_*)
    if (table_exists(db.db, "angle_harm_term")) {
        SqliteStmt q(db.db,
            "SELECT t.p0, t.p1, t.p2, p.theta0, p.fc "
            "FROM angle_harm_term t "
            "JOIN angle_harm_param p ON t.param = p.id");
        while (q.step()) {
            FFAngle a;
            a.i      = q.col_int(0);
            a.j      = q.col_int(1);
            a.k      = q.col_int(2);
            a.theta0 = static_cast<Real>(q.col_dbl(3) * M_PI / 180.0);  // degrees -> radians
            a.fc     = static_cast<Real>(q.col_dbl(4));
            out.lig_angles.push_back(a);
        }
    }
    spdlog::info("  Ligand angles: {}", out.lig_angles.size());

    out.build_pair_lookup();
    out.loaded = true;
    spdlog::info("Ligand DMS loaded successfully.");
}


void read_receptor_dms(const std::string& dms_path, FFMinData& out) {
    spdlog::info("Reading receptor DMS: {}", dms_path);
    SqliteDB db(dms_path);

    // A receptor built without a force field (bonds + ud2_atom_props only) is enough for
    // Vina scoring; skip the LJ/charge read as the ligand reader does.
    if (!table_exists(db.db, "nonbonded_param")) {
        spdlog::info("  No FF tables (nonbonded_param absent); FF scoring disabled for the receptor");
        return;
    }

    // Per-atom params only (receptor is rigid, no bonded terms needed). Every particle needs
    // nonbonded parameters, as for the ligand; FF scoring needs both sides, so an empty or
    // partial table switches a ligand FF loaded before off.
    read_atom_params(db.db, out.rec_atoms);
    const int n_particle = count_particles(db.db);
    if (out.rec_atoms.natom == 0 || out.rec_atoms.natom != n_particle) {
        const std::string msg = fmt::format("  Receptor DMS {}: nonbonded parameters for {} of {} atoms; "
                                            "FF scoring disabled", dms_path, out.rec_atoms.natom, n_particle);
        // Warn only when a ligand FF was waiting for it
        if (out.loaded) spdlog::warn(msg);
        else spdlog::info(msg);
        out.rec_atoms = FFAtomParams();
        out.loaded = false;
        return;
    }
    spdlog::info("  Receptor atoms: {}", out.rec_atoms.natom);
    spdlog::info("Receptor DMS loaded successfully.");
}


// ===================== Enriched DMS readers (replace JSON) =====================

// Helpers for torsion-based fragment splitting (used to exclude intra pairs)

bool checkInAnySameSet(const std::vector<std::set<int>>& frags, int v1, int v2) {
    for (const auto& frag: frags) {
        if (frag.find(v1) != frag.end() && frag.find(v2) != frag.end()) {
            return true;
        }
    }
    return false;
}


/**
 * @brief Vina excludes intramolecular pairs separated by up to three bonds (1-2, 1-3, 1-4).
 * DMS files from a force-field preparation carry them in the exclusion / pair_12_6_es_term tables;
 * the lightweight preparation writes neither, which used to leave every 1-4 pair across a rotatable
 * bond in the intra energy (repulsive at 1-4 distances). Derive them from the bond table instead.
 * Atom ids in [off, off + natom) are rebased to 0.
 */
static void derive_bond_path_exclusions(sqlite3* db, int off, int natom,
                                        std::set<std::pair<int, int>>& pairs_1213,
                                        std::set<std::pair<int, int>>& pairs_14) {
    std::vector<std::vector<int>> nb(natom);
    {
        SqliteStmt q(db, "SELECT p0, p1 FROM bond" + range_clause("p0", off, natom));
        while (q.step()) {
            int i = q.col_int(0) - off, j = q.col_int(1) - off;
            if (i < 0 || j < 0 || i >= natom || j >= natom) continue;
            nb[i].push_back(j); nb[j].push_back(i);
        }
    }
    for (int a = 0; a < natom; a++) {
        std::vector<int> depth(natom, -1);
        std::vector<int> frontier = {a};
        depth[a] = 0;
        for (int d = 1; d <= 3; d++) {
            std::vector<int> next;
            for (int x : frontier) {
                for (int y : nb[x]) {
                    if (depth[y] >= 0) continue;
                    depth[y] = d;
                    next.push_back(y);
                }
            }
            frontier.swap(next);
        }
        for (int b = a + 1; b < natom; b++) {
            if (depth[b] < 0) continue;
            auto p = std::make_pair(a, b);
            if (depth[b] <= 2) pairs_1213.insert(p);
            else pairs_14.insert(p);
        }
    }
}


/**
 * @brief United-atom charges and formally charged groups of a ligand read from a DMS.
 * charges_ua[i] = charge[i] + charges of the hydrogens bonded to i (hydrogens themselves 0).
 * charged_group[i] = 1 for atoms carrying a formal charge (ud2_formal_charges) and for their
 * resonance partners: same-element terminal atoms on the same heavy neighbour (carboxylate,
 * phosphate, sulfonate oxygens) and, for charged nitrogens, the other nitrogens of the same
 * carbon (amidinium, guanidinium). Atom ids are rebased by off; only [off, off + natom) is read.
 */
/**
 * @brief Ligand atoms eligible for the apolar-environment credit: weakly polar atoms that tolerate
 * burial in a hydrophobic pocket. Polar carbons (C_P), fluorine and sulfur always qualify; nitrogen
 * and oxygen only when they carry no hydrogen, no formal charge, and are of the ether / sulfonyl /
 * nitro / aromatic-N kind. Strong donors and acceptors (hydroxyl, carbonyl, carboxylate) do not.
 */
/**
 * @brief Phenylalanine, tyrosine and tryptophan aromatic ring carbons, by residue and atom name.
 * Tyr CZ and Trp CD1 / CE2 are left out: they carry a heteroatom and are typed C_P, not hydrophobic.
 * Histidine ring carbons are all C_P for the same reason. Amber terminal residue names (NPHE, CTYR,
 * ...) are stripped first.
 */
static bool is_receptor_ring_carbon(const std::string& resname, const std::string& atom_name) {
    static const std::set<std::string> amino_acids = {
        "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","HIE","HID","HIP","ILE","LEU","LYS",
        "MET","PHE","PRO","SER","THR","TRP","TYR","VAL","CYX","ASH","GLH","LYN","CYM"};
    std::string rn = resname;
    if (rn.size() == 4 && (rn[0] == 'N' || rn[0] == 'C') && amino_acids.count(rn.substr(1))) {
        rn = rn.substr(1);
    }
    if (rn == "PHE") {
        return atom_name == "CG" || atom_name == "CD1" || atom_name == "CD2"
            || atom_name == "CE1" || atom_name == "CE2" || atom_name == "CZ";
    }
    if (rn == "TYR") {
        return atom_name == "CG" || atom_name == "CD1" || atom_name == "CD2"
            || atom_name == "CE1" || atom_name == "CE2";
    }
    if (rn == "TRP") {
        return atom_name == "CG" || atom_name == "CD2" || atom_name == "CE3"
            || atom_name == "CZ2" || atom_name == "CZ3" || atom_name == "CH2";
    }
    return false;
}


/**
 * @brief Optional per-ligand annotations for the rescoring terms: aromatic flags
 * (ud2_atom_aromatic) and torsional-strain quadruples (ud2_torsion_penalties). Both tables are
 * written by the ligand preparation; when they are missing the corresponding terms score as zero.
 * Atom ids in [off, off + natom) are rebased to 0.
 */
static void read_ligand_scoring_annotations(sqlite3* db, int off, int natom,
                                            std::vector<int>& out_aromatic,
                                            std::vector<TorsionPenaltyRow>& out_rows) {
    out_aromatic.assign(natom, 0);
    if (table_exists(db, "ud2_atom_aromatic")) {
        SqliteStmt q(db, "SELECT atom_id FROM ud2_atom_aromatic" + range_clause("atom_id", off, natom));
        while (q.step()) {
            int i = q.col_int(0) - off;
            if (i >= 0 && i < natom) out_aromatic[i] = 1;
        }
    }
    out_rows.clear();
    if (!table_exists(db, "ud2_torsion_penalties")) return;
    SqliteStmt q(db, "SELECT atom_i, atom_j, atom_k, atom_l, kind, tolerance, weight_share "
                     "FROM ud2_torsion_penalties" + range_clause("atom_j", off, natom)
                     + " ORDER BY rowid");
    while (q.step()) {
        TorsionPenaltyRow r;
        bool ok = true;
        for (int k = 0; k < 4; k++) {
            r.atoms[k] = q.col_int(k) - off;
            if (r.atoms[k] < 0 || r.atoms[k] >= natom) ok = false;
        }
        if (!ok) continue;  // belongs to another ligand of a merged DMS
        r.kind = q.col_int(4);
        r.tolerance = static_cast<Real>(q.col_dbl(5));
        r.weight = static_cast<Real>(q.col_dbl(6));
        out_rows.push_back(r);
    }
}


static void fill_polar_env_class(int natom, const std::vector<int>& vina_types,
                                 const std::vector<int>& anum,
                                 const std::vector<std::vector<int>>& nb,
                                 const AtomTypeProps& type_props,
                                 const std::vector<int>& formal_group,
                                 std::vector<int>& out) {
    out.assign(natom, 0);
    auto is_h = [&](int i){ return type_props.is_hydrogen[vina_types[i]]; };
    for (int a = 0; a < natom; a++) {
        if (is_h(a)) continue;
        const int t = vina_types[a];
        if (t == VN_TYPE_C_P || t == VN_TYPE_F_H || t == VN_TYPE_S_P) { out[a] = 1; continue; }
        if (formal_group[a]) continue;
        if (t == VN_TYPE_N_P || t == VN_TYPE_N_A) { out[a] = 1; continue; }
        if (t == VN_TYPE_O_P || t == VN_TYPE_O_A) {
            int n_heavy = 0, last = -1;
            for (int x : nb[a]) if (!is_h(x)) { n_heavy++; last = x; }
            if (n_heavy == 2) out[a] = 1;
            else if (n_heavy == 1 && (anum[last] == 7 || anum[last] == 16)) out[a] = 1;
        }
    }
}


static void ligand_charge_annotations(sqlite3* db, int off, int natom, const std::vector<int>& vina_types,
                                      const AtomTypeProps& type_props, const std::vector<Real>& charges,
                                      std::vector<Real>& charges_ua, std::vector<int>& charged_group,
                                      std::vector<int>* polar_env_class = nullptr) {
    std::vector<std::vector<int>> nb(natom);
    {
        SqliteStmt q(db, "SELECT p0, p1 FROM bond" + range_clause("p0", off, natom));
        while (q.step()) {
            int i = q.col_int(0) - off, j = q.col_int(1) - off;
            if (i < 0 || j < 0 || i >= natom || j >= natom) continue;
            nb[i].push_back(j); nb[j].push_back(i);
        }
    }
    std::vector<int> anum(natom, 0);
    {
        SqliteStmt q(db, "SELECT id, anum FROM particle" + range_clause("id", off, natom));
        while (q.step()) {
            int i = q.col_int(0) - off;
            if (i >= 0 && i < natom) anum[i] = q.col_int(1);
        }
    }
    auto is_h = [&](int i){ return type_props.is_hydrogen[vina_types[i]]; };
    charges_ua.assign(natom, Real(0));
    for (int i = 0; i < natom; i++) {
        if (is_h(i)) continue;
        Real q = charges[i];
        for (int j : nb[i]) if (is_h(j)) q += charges[j];
        charges_ua[i] = q;
    }
    charged_group.assign(natom, 0);
    if (polar_env_class) polar_env_class->assign(natom, 0);
    if (!table_exists(db, "ud2_formal_charges")) {
        if (polar_env_class) {
            // no formal charges: every atom counts as neutral for the class test
            std::vector<int> no_charge(natom, 0);
            fill_polar_env_class(natom, vina_types, anum, nb, type_props, no_charge, *polar_env_class);
        }
        return;
    }
    std::vector<int> fc(natom, 0);
    {
        SqliteStmt q(db, "SELECT atom_id, formal_charge FROM ud2_formal_charges" + range_clause("atom_id", off, natom));
        while (q.step()) {
            int i = q.col_int(0) - off;
            if (i >= 0 && i < natom) fc[i] = q.col_int(1);
        }
    }
    auto n_heavy_nb = [&](int i){ int n = 0; for (int j : nb[i]) if (!is_h(j)) n++; return n; };
    // formal_group is the same resonance spreading WITHOUT the zwitterion exception: the
    // apolar-environment class needs nitro / N-oxide oxygens to count as formally charged.
    std::vector<int> formal_group(natom, 0);
    for (int a = 0; a < natom; a++) {
        if (fc[a] == 0 || is_h(a)) continue;
        bool zwitterion = false;
        for (int x : nb[a]) if (fc[x] * fc[a] < 0) zwitterion = true;
        std::vector<int> group = {a};
        for (int x : nb[a]) {
            if (is_h(x)) continue;
            for (int y : nb[x]) {
                if (y == a || is_h(y) || anum[y] != anum[a]) continue;
                if (n_heavy_nb(y) == 1 || anum[a] == 7) group.push_back(y);
            }
        }
        for (int y : group) formal_group[y] = 1;
        if (zwitterion) continue;  // internal zwitterions (nitro, N-oxide) are neutral groups
        for (int y : group) charged_group[y] = 1;
    }
    if (polar_env_class) {
        fill_polar_env_class(natom, vina_types, anum, nb, type_props, formal_group, *polar_env_class);
    }
}

void split_torsions_into_frags(const std::set<int>& root, const std::vector<UDTorsion>& torsions,
                               std::vector<std::set<int>>& out_frags) {
    // The first frag is the root atoms
    std::set<int> frag0;
    for (int i : root){
        frag0.insert(i);
    }

    // use index list
    std::vector<int> range(torsions.size());
    std::iota(range.begin(), range.end(), 0);

    // Step 1: Identify big torsions and add them as fragments
    std::vector<int> range_tmp = range;
    for (int itor : range_tmp) {
        auto& torsion = torsions[itor];
        // Check if either axis atom is in the root set
        if (root.find(torsion.axis[0]) != root.end() || root.find(torsion.axis[1]) != root.end()) {
            range.erase(std::remove(range.begin(), range.end(), itor), range.end());

            frag0.insert(torsion.axis[0]);
            frag0.insert(torsion.axis[1]);

            std::set<int> frag_tmp;
            for (int i : torsion.rotated_atoms){
                frag_tmp.insert(i);
            }
            frag_tmp.insert(torsion.axis[0]);
            frag_tmp.insert(torsion.axis[1]);
            out_frags.push_back(frag_tmp);
        }
    }
    out_frags.push_back(frag0);

    // Step 2: Split existing fragments using other torsions
    for (int itor : range) {
        auto& torsion = torsions[itor];

        for (auto& frag : out_frags) {
            if ((frag.find(torsion.axis[0]) != frag.end()) and (frag.find(torsion.axis[1]) != frag.end())) {
                std::set<int> frag1, frag2;
                std::set<int> tor_set(torsion.rotated_atoms.begin(), torsion.rotated_atoms.end());
                std::set_intersection(frag.begin(), frag.end(),
                                      tor_set.begin(), tor_set.end(),
                                      std::inserter(frag1, frag1.begin()));
                std::set_difference(frag.begin(), frag.end(),
                                    tor_set.begin(), tor_set.end(),
                                    std::inserter(frag2, frag2.begin()));

                frag1.insert(torsion.axis[0]);
                frag1.insert(torsion.axis[1]);
                frag2.insert(torsion.axis[0]);
                frag2.insert(torsion.axis[1]);

                frag = frag1;
                out_frags.push_back(frag2);
                break;
            }
        }
    }
}

/**
 * @brief A torsion stays in the search unless every atom it rotates besides the axis is a hydrogen.
 * Of those hydrogen-only torsions, the N-H of a fixed donor with one hydrogen (e.g. anionic sulfonamide
 * S-N(-)H) stays too. The H-bearing atom is the axis atom listed among the rotated atoms; when neither or
 * both are, the torsion stays.
 */
static bool keep_torsion_in_search(const UDTorsion& t, const std::vector<int>& vina_types,
                                   const AtomTypeProps& type_props) {
    int n_h = 0;
    for (int a : t.rotated_atoms) {
        if (a == t.axis[0] || a == t.axis[1]) continue;
        if (!type_props.is_hydrogen[vina_types[a]]) return true;
        n_h++;
    }
    if (n_h == 0) return true;
    const bool rot0 = std::find(t.rotated_atoms.begin(), t.rotated_atoms.end(), t.axis[0]) != t.rotated_atoms.end();
    const bool rot1 = std::find(t.rotated_atoms.begin(), t.rotated_atoms.end(), t.axis[1]) != t.rotated_atoms.end();
    if (rot0 == rot1) return true;
    const int d = vina_types[rot0 ? t.axis[0] : t.axis[1]];
    return n_h == 1 && type_props.is_donor[d] && !hbond_donor_rotatable(d, 1);
}

/**
 * @brief Moves the hydrogen-only torsions (keep_torsion_in_search) and their dihedrals, in order, from
 * torsions / dihedrals to ff_only_torsions / ff_only_dihedrals, where FF minimisation still rotates them.
 * Called after the intra pairs, so the fragments include every DMS torsion. Returns the number moved.
 */
static int split_ff_only_torsions(UDFlexMol& m, const AtomTypeProps& type_props) {
    std::vector<char> keep(m.torsions.size());
    for (size_t t = 0; t < m.torsions.size(); t++) {
        keep[t] = keep_torsion_in_search(m.torsions[t], m.vina_types, type_props);
    }
    const int n_moved = static_cast<int>(std::count(keep.begin(), keep.end(), 0));
    if (n_moved == 0) return 0;

    std::vector<UDTorsion> torsions;
    std::vector<Real> dihedrals;
    for (size_t t = 0; t < keep.size(); t++) {
        if (keep[t]) {
            torsions.push_back(std::move(m.torsions[t]));
            dihedrals.push_back(m.dihedrals[t]);
        } else {
            m.ff_only_torsions.push_back(std::move(m.torsions[t]));
            m.ff_only_dihedrals.push_back(m.dihedrals[t]);
        }
    }
    m.torsions = std::move(torsions);
    m.dihedrals = std::move(dihedrals);
    return n_moved;
}

namespace {

/// Parse a comma-separated string of integers.
std::vector<int> parse_csv_ints(const char* text) {
    std::vector<int> result;
    if (!text || text[0] == '\0') return result;
    std::string s(text);
    size_t pos = 0;
    while (pos < s.size()) {
        size_t next = s.find(',', pos);
        if (next == std::string::npos) next = s.size();
        if (next > pos) {
            result.push_back(std::stoi(s.substr(pos, next - pos)));
        }
        pos = next + 1;
    }
    return result;
}

} // anonymous namespace


void read_ligand_dms_full(const std::string& dms_path,
                          UDFlexMol& out_flex,
                          FFMinData& out_ff,
                          const AtomTypeProps& type_props,
                          bool use_tor_lib) {
    spdlog::info("Reading enriched ligand DMS: {}", dms_path);
    SqliteDB db(dms_path);

    // Verify enrichment tables exist
    if (!table_exists(db.db, "ud2_atom_props")) {
        throw std::runtime_error("DMS missing ud2_atom_props table: " + dms_path);
    }

    // --- 1. Read per-atom data ---
    // Coordinates and charges from particle table; vina types from ud2_atom_props
    {
        SqliteStmt q(db.db,
            "SELECT p.id, p.x, p.y, p.z, p.charge, "
            "       ua.vina_type "
            "FROM particle p "
            "JOIN ud2_atom_props ua ON p.id = ua.atom_id "
            "ORDER BY p.id");

        Real coords_sum[3] = {0};
        while (q.step()) {
            Real x = static_cast<Real>(q.col_dbl(1));
            Real y = static_cast<Real>(q.col_dbl(2));
            Real z = static_cast<Real>(q.col_dbl(3));
            Real charge = static_cast<Real>(q.col_dbl(4));
            int vina_type = q.col_int(5);

            out_flex.coords.push_back(x);
            out_flex.coords.push_back(y);
            out_flex.coords.push_back(z);
            out_flex.vina_types.push_back(vina_type);
            out_flex.charges.push_back(charge);
            coords_sum[0] += x;
            coords_sum[1] += y;
            coords_sum[2] += z;
        }
        out_flex.natom = static_cast<int>(out_flex.charges.size());
        if (out_flex.natom > 0) {
            out_flex.center[0] = coords_sum[0] / out_flex.natom;
            out_flex.center[1] = coords_sum[1] / out_flex.natom;
            out_flex.center[2] = coords_sum[2] / out_flex.natom;
        }
    }
    spdlog::info("  Ligand atoms: {}", out_flex.natom);

    // Read molecule name from ud2_metadata if available, else derive from filename
    if (table_exists(db.db, "ud2_metadata")) {
        SqliteStmt qm(db.db, "SELECT value FROM ud2_metadata WHERE key = 'molecule_name'");
        if (qm.step()) {
            const char* val = reinterpret_cast<const char*>(sqlite3_column_text(qm.stmt, 0));
            if (val) out_flex.name = val;
        }
    }
    if (out_flex.name.empty()) {
        auto pos = dms_path.find_last_of("/\\");
        std::string basename = (pos == std::string::npos) ? dms_path : dms_path.substr(pos + 1);
        auto dot = basename.rfind('.');
        out_flex.name = (dot != std::string::npos) ? basename.substr(0, dot) : basename;
    }

    // --- 2. Read 1-4 pairs (from pair_12_6_es_term, absent when no FF) ---
    if (table_exists(db.db, "pair_12_6_es_term")) {
        SqliteStmt q(db.db, "SELECT p0, p1 FROM pair_12_6_es_term");
        while (q.step()) {
            int i = q.col_int(0);
            int j = q.col_int(1);
            out_flex.pairs_14.insert(std::make_pair(std::min(i,j), std::max(i,j)));
        }
    }

    // --- 3. Read exclusions (1-2 & 1-3 = all exclusions minus 1-4) ---
    if (table_exists(db.db, "exclusion")) {
        SqliteStmt q(db.db, "SELECT p0, p1 FROM exclusion");
        while (q.step()) {
            int i = q.col_int(0);
            int j = q.col_int(1);
            auto p = std::make_pair(std::min(i,j), std::max(i,j));
            if (out_flex.pairs_14.find(p) == out_flex.pairs_14.end()) {
                out_flex.pairs_1213.insert(p);
            }
        }
    }

    if (out_flex.pairs_1213.empty() && out_flex.pairs_14.empty()) {
        derive_bond_path_exclusions(db.db, 0, out_flex.natom, out_flex.pairs_1213, out_flex.pairs_14);
        spdlog::info("  No exclusion tables in DMS: derived {} 1-2/1-3 and {} 1-4 pairs from bonds",
                     out_flex.pairs_1213.size(), out_flex.pairs_14.size());
    }
    ligand_charge_annotations(db.db, 0, out_flex.natom, out_flex.vina_types, type_props, out_flex.charges,
                              out_flex.charges_ua, out_flex.charged_group, &out_flex.polar_env_class);
    read_ligand_scoring_annotations(db.db, 0, out_flex.natom, out_flex.aromatic, out_flex.torsion_penalties);

    // --- 4. Read root atoms ---
    std::set<int> root_atoms;
    {
        SqliteStmt q(db.db, "SELECT atom_id FROM ud2_root_atoms");
        while (q.step()) {
            root_atoms.insert(q.col_int(0));
        }
    }

    // --- 5. Read torsions ---
    {
        SqliteStmt q(db.db,
            "SELECT torsion_id, atom_i, atom_j, atom_k, atom_l, "
            "       dihedral_value, mobile_atoms "
            "FROM ud2_torsions ORDER BY torsion_id");

        while (q.step()) {
            int tor_id = q.col_int(0);
            UDTorsion torsion;
            torsion.atoms[0] = q.col_int(1);
            torsion.atoms[1] = q.col_int(2);
            torsion.atoms[2] = q.col_int(3);
            torsion.atoms[3] = q.col_int(4);
            torsion.axis[0] = torsion.atoms[1];
            torsion.axis[1] = torsion.atoms[2];

            Real dihedral_deg = static_cast<Real>(q.col_dbl(5));
            out_flex.dihedrals.push_back(dihedral_deg * static_cast<Real>(M_PI / 180.0));

            // Parse mobile atoms from CSV text
            const char* mobile_text = reinterpret_cast<const char*>(
                sqlite3_column_text(q.stmt, 6));
            torsion.rotated_atoms = parse_csv_ints(mobile_text);

            // Read torsion ranges for this torsion
            if (use_tor_lib && table_exists(db.db, "ud2_torsion_ranges")) {
                SqliteStmt qr(db.db,
                    "SELECT range_lo, range_hi FROM ud2_torsion_ranges "
                    "WHERE torsion_id = ? ORDER BY rowid");
                sqlite3_bind_int(qr.stmt, 1, tor_id);
                while (qr.step()) {
                    Real lo = static_cast<Real>(qr.col_dbl(0) * M_PI / 180.0);
                    Real hi = static_cast<Real>(qr.col_dbl(1) * M_PI / 180.0);
                    if (lo < hi) {
                        torsion.range_list.push_back(lo);
                        torsion.range_list.push_back(hi);
                    } else if (lo > 0 && hi < 0) {
                        torsion.range_list.push_back(lo);
                        torsion.range_list.push_back(static_cast<Real>(M_PI));
                        torsion.range_list.push_back(static_cast<Real>(-M_PI));
                        torsion.range_list.push_back(hi);
                    } else {
                        spdlog::critical("DMS torsion range error: lo={}, hi={}", lo, hi);
                    }
                }
            }
            if (torsion.range_list.empty()) {
                torsion.range_list.push_back(static_cast<Real>(-M_PI));
                torsion.range_list.push_back(static_cast<Real>(M_PI));
            }

            out_flex.torsions.push_back(torsion);
        }
    }
    spdlog::info("  Torsions: {}", out_flex.torsions.size());

    // --- 6. Compute fragments, then intra_pairs ---
    {
        std::vector<std::set<int>> frags;
        split_torsions_into_frags(root_atoms, out_flex.torsions, frags);

        for (int i = 0; i < out_flex.natom; i++) {
            // Hydrogens contribute exactly zero to the Vina pair energy, so a pair that
            // involves one is dead weight in every intra loop (about three quarters of them).
            if (type_props.is_hydrogen[out_flex.vina_types[i]]) continue;
            for (int j = i + 1; j < out_flex.natom; j++) {
                if (type_props.is_hydrogen[out_flex.vina_types[j]]) continue;
                auto p = std::make_pair(i, j);
                if (out_flex.pairs_1213.find(p) == out_flex.pairs_1213.end() &&
                    out_flex.pairs_14.find(p) == out_flex.pairs_14.end() &&
                    !checkInAnySameSet(frags, i, j)) {
                    out_flex.intra_pairs.push_back(i);
                    out_flex.intra_pairs.push_back(j);
                }
            }
        }
    }
    spdlog::info("  Intra pairs: {}", out_flex.intra_pairs.size() / 2);
    const int n_ff_only = split_ff_only_torsions(out_flex, type_props);
    if (n_ff_only > 0) {
        spdlog::info("  {} hydrogen-only torsions removed from the search (kept for FF minimisation)", n_ff_only);
    }
    // Note: inter_pairs are computed later when receptor is also available

    // --- 7. Read FF params for FF minimization (DockTask::run reads them again and warns) ---
    read_ligand_dms(dms_path, out_ff, false);

    // --- 8. Extension tables (register_table_reader: ud2_bias in ext/bias/bias_reader.cpp, ud2_ring_bias in
    //        ext/bias/ring_reader.cpp), when present ---
    read_registered_tables(db.db, TableScope{0, out_flex.natom, false}, out_flex);
}


void read_receptor_dms_full(const std::string& dms_path,
                            const Box& box,
                            UDFixMol& out_fix,
                            FFMinData& out_ff,
                            const AtomTypeProps& type_props) {
    spdlog::info("Reading enriched receptor DMS: {}", dms_path);
    SqliteDB db(dms_path);

    bool has_ud2 = table_exists(db.db, "ud2_atom_props");

    // DMS atom ID → compacted heavy-atom index (for RecDonorHTopo)
    std::unordered_map<int, int> dms_id_to_compact;

    if (has_ud2) {
        // Read from enriched DMS: particle + ud2_atom_props
        SqliteStmt q(db.db,
            "SELECT p.id, p.x, p.y, p.z, p.charge, "
            "       ua.vina_type, p.resname, p.resid, p.chain, p.name "
            "FROM particle p "
            "JOIN ud2_atom_props ua ON p.id = ua.atom_id "
            "ORDER BY p.id");

        int compact_idx = 0;
        std::map<ResidueKey, int> res_map;
        while (q.step()) {
            int dms_id = q.col_int(0);
            int vina_type = q.col_int(5);
            if (type_props.is_hydrogen[vina_type]) continue;  // skip H

            Real x = static_cast<Real>(q.col_dbl(1));
            Real y = static_cast<Real>(q.col_dbl(2));
            Real z = static_cast<Real>(q.col_dbl(3));

            if (!box.is_inside(x, y, z)) continue;  // box filter

            out_fix.coords.push_back(x);
            out_fix.coords.push_back(y);
            out_fix.coords.push_back(z);
            out_fix.vina_types.push_back(vina_type);
            out_fix.charges.push_back(static_cast<Real>(q.col_dbl(4)));
            out_fix.arom_ring_c.push_back(
                (vina_type == VN_TYPE_C_H && is_receptor_ring_carbon(q.col_str(6), q.col_str(9))) ? 1 : 0);
            dms_id_to_compact[dms_id] = compact_idx++;

            std::string chain = q.col_str(8);
            if (chain.empty()) chain = "A";
            ResidueKey key{chain, q.col_int(7), q.col_str(6)};
            auto it = res_map.find(key);
            int res_idx;
            if (it == res_map.end()) {
                res_idx = static_cast<int>(out_fix.residues.size());
                res_map[key] = res_idx;
                out_fix.residues.push_back(key);
            } else {
                res_idx = it->second;
            }
            out_fix.atom_to_residue.push_back(res_idx);
        }
    } else {
        spdlog::warn("Receptor DMS has no ud2_atom_props; cannot populate UDFixMol");
    }

    out_fix.natom = static_cast<int>(out_fix.charges.size());
    spdlog::info("  Receptor atoms in box: {}", out_fix.natom);
    spdlog::info("  Receptor residues in box: {}", out_fix.residues.size());

    // Build RecDonorHTopo: receptor donor-H side table for angular HBond scoring
    if (has_ud2 && out_fix.natom > 0) {
        // Step 1: Read H atom coordinates (DMS ID → coords)
        struct HCoord { Real x, y, z, q; };
        std::unordered_map<int, HCoord> h_atoms;
        {
            SqliteStmt q(db.db,
                "SELECT p.id, p.x, p.y, p.z, ua.vina_type, p.charge "
                "FROM particle p "
                "JOIN ud2_atom_props ua ON p.id = ua.atom_id "
                "ORDER BY p.id");
            while (q.step()) {
                int vina_type = q.col_int(4);
                if (!type_props.is_hydrogen[vina_type]) continue;
                int dms_id = q.col_int(0);
                h_atoms[dms_id] = {
                    static_cast<Real>(q.col_dbl(1)),
                    static_cast<Real>(q.col_dbl(2)),
                    static_cast<Real>(q.col_dbl(3)),
                    static_cast<Real>(q.col_dbl(5))
                };
            }
        }
        out_fix.charges_ua = out_fix.charges;  // heavy-atom charges, hydrogens folded in below

        // Step 2: Read bond table, collect bonded H coords per compacted donor and one
        //         bonded heavy neighbour (antecedent) per compacted atom
        std::vector<std::vector<Real>> h_coords_per_atom(out_fix.natom);
        std::vector<int> antecedent(out_fix.natom, -1);
        {
            SqliteStmt q(db.db, "SELECT p0, p1 FROM bond");
            while (q.step()) {
                int p0 = q.col_int(0);
                int p1 = q.col_int(1);
                {
                    auto it0 = dms_id_to_compact.find(p0);
                    auto it1 = dms_id_to_compact.find(p1);
                    if (it0 != dms_id_to_compact.end() && it1 != dms_id_to_compact.end()) {
                        if (antecedent[it0->second] < 0) antecedent[it0->second] = it1->second;
                        if (antecedent[it1->second] < 0) antecedent[it1->second] = it0->second;
                    }
                }
                // Check both bond orientations: (heavy, H) or (H, heavy)
                auto try_bond = [&](int heavy_id, int h_id) {
                    auto it_heavy = dms_id_to_compact.find(heavy_id);
                    if (it_heavy == dms_id_to_compact.end()) return;
                    auto it_h = h_atoms.find(h_id);
                    if (it_h == h_atoms.end()) return;
                    int ci = it_heavy->second;
                    out_fix.charges_ua[ci] += it_h->second.q;
                    if (type_props.is_donor[out_fix.vina_types[ci]]) {
                        h_coords_per_atom[ci].push_back(it_h->second.x);
                        h_coords_per_atom[ci].push_back(it_h->second.y);
                        h_coords_per_atom[ci].push_back(it_h->second.z);
                    }
                };
                try_bond(p0, p1);
                try_bond(p1, p0);
            }
        }

        // Step 3: Build CSR arrays
        auto& rh = out_fix.rec_donor_h;
        rh.natom = out_fix.natom;
        rh.h_start = new int[out_fix.natom + 1]();
        for (int i = 0; i < out_fix.natom; i++) {
            rh.h_start[i + 1] = rh.h_start[i]
                                 + static_cast<int>(h_coords_per_atom[i].size()) / 3;
        }
        rh.total_h = rh.h_start[out_fix.natom];
        rh.antecedent = new int[out_fix.natom];
        for (int i = 0; i < out_fix.natom; i++) rh.antecedent[i] = antecedent[i];
        rh.h_coords = new Real[std::max(rh.total_h * 3, 1)];
        for (int i = 0; i < out_fix.natom; i++) {
            int offset = rh.h_start[i] * 3;
            for (size_t j = 0; j < h_coords_per_atom[i].size(); j++) {
                rh.h_coords[offset + j] = h_coords_per_atom[i][j];
            }
        }

        // Step 4: free-rotor donors (cone hydrogen) vs fixed donors (explicit hydrogens)
        rh.donor_rotatable = new int[out_fix.natom];
        int n_donors_with_h = 0, n_rotatable = 0;
        for (int i = 0; i < rh.natom; i++) {
            const int n_h = rh.h_start[i + 1] - rh.h_start[i];
            if (n_h > 0) n_donors_with_h++;
            const Real* D = out_fix.coords.data() + i * 3;
            const Real* X = rh.antecedent[i] >= 0 ? out_fix.coords.data() + rh.antecedent[i] * 3 : nullptr;
            const Real* H1 = n_h >= 1 ? rh.h_coords + rh.h_start[i] * 3 : nullptr;
            const Real* H2 = n_h >= 2 ? rh.h_coords + (rh.h_start[i] + 1) * 3 : nullptr;
            rh.donor_rotatable[i] = hbond_donor_rotatable_geom(out_fix.vina_types[i], n_h, D, X, H1, H2) ? 1 : 0;
            n_rotatable += rh.donor_rotatable[i];
        }
        spdlog::info("  Receptor donor-H topology: {} donors with {} bonded H total, {} free-rotor donors",
                     n_donors_with_h, rh.total_h, n_rotatable);
    }

    // Read FF params for FF minimization
    read_receptor_dms(dms_path, out_ff);
    spdlog::info("Receptor DMS loaded successfully.");
}


// ===================== Multi-ligand merged DMS =====================

bool is_multi_ligand_dms(const std::string& dms_path) {
    try {
        SqliteDB db(dms_path);
        return table_exists(db.db, "ud2_molecules");
    } catch (...) {
        return false;
    }
}


/// Molecule boundary info from ud2_molecules table.
struct MolBoundary {
    int molecule_id;
    std::string mol_name;
    int atom_offset;
    int natom;
};

/// Read molecule boundaries from merged DMS.
static std::vector<MolBoundary> read_mol_boundaries(sqlite3* db) {
    std::vector<MolBoundary> result;
    SqliteStmt q(db,
        "SELECT molecule_id, mol_name, atom_offset, natom "
        "FROM ud2_molecules ORDER BY molecule_id");
    while (q.step()) {
        MolBoundary mb;
        mb.molecule_id = q.col_int(0);
        const char* name = reinterpret_cast<const char*>(
            sqlite3_column_text(q.stmt, 1));
        mb.mol_name = name ? name : "";
        mb.atom_offset = q.col_int(2);
        mb.natom = q.col_int(3);
        result.push_back(mb);
    }
    return result;
}


/// Rows of one molecule-keyed table (ud2_atom_props, ud2_root_atoms, ...) grouped by molecule_id.
/// The protocol runner writes these tables without a molecule_id index, so a "WHERE molecule_id = k"
/// query per ligand scans the whole table: quadratic in the ligand count (16.6 s of a 2,000-ligand
/// read, 66 s at 4,000). Each table is read once instead, on first use, and a ligand takes its rows
/// from its bucket wherever that provably gives the rows of its per-ligand query; the callers run the
/// per-ligand query otherwise.
template <typename Row>
struct MolBuckets {
    bool loaded = false;
    bool usable = false;   // every non-NULL molecule_id is stored as an INTEGER
    std::unordered_map<sqlite3_int64, std::vector<Row>> rows;
    std::vector<Row> empty;

    /// Rows whose molecule_id equals mol_id, in the order of the bulk query.
    std::vector<Row>& of(int mol_id) {
        auto it = rows.find(mol_id);
        return it == rows.end() ? empty : it->second;
    }
};

/// Run `sql` (column 0 = molecule_id) once and fill the buckets with fill(q). A molecule_id stored as an
/// INTEGER equals the integer literal k of "WHERE molecule_id = k" exactly when its value is k, whatever the
/// column affinity, and NULL never does; any other storage class (or a failed step) leaves the buckets unusable.
template <typename Row, typename Fill>
static void load_mol_buckets(sqlite3* db, const std::string& sql, MolBuckets<Row>& b, Fill fill) {
    b.loaded = true;
    SqliteStmt q(db, sql);
    int rc;
    while ((rc = sqlite3_step(q.stmt)) == SQLITE_ROW) {
        int type = sqlite3_column_type(q.stmt, 0);
        if (type == SQLITE_NULL) continue;
        if (type != SQLITE_INTEGER) break;
        b.rows[sqlite3_column_int64(q.stmt, 0)].push_back(fill(q));
    }
    b.usable = rc == SQLITE_DONE;
    if (!b.usable) b.rows.clear();
}

/// The rows of a per-ligand query, in its order.
template <typename Row, typename Fill>
static std::vector<Row> query_rows(sqlite3* db, const std::string& sql, Fill fill) {
    std::vector<Row> rows;
    SqliteStmt q(db, sql);
    while (q.step()) rows.push_back(fill(q));
    return rows;
}

/// Whether a table has any index (including the automatic ones of PRIMARY KEY / UNIQUE). Without one, a
/// query on it with no ORDER BY can only scan it, so its rows come in rowid order.
static bool table_has_index(sqlite3* db, const char* table_name) {
    SqliteStmt q(db, std::string("PRAGMA index_list(") + table_name + ")");
    return q.step();
}


void read_multi_ligand_dms_full(const std::string& dms_path,
                                UDFlexMolList& out_flex_list,
                                std::vector<FFMinData>& out_ff_list,
                                std::vector<SDFMolData>& out_sdf_list,
                                const AtomTypeProps& type_props,
                                bool use_tor_lib) {
    spdlog::info("Reading multi-ligand merged DMS: {}", dms_path);
    SqliteDB db(dms_path);

    auto boundaries = read_mol_boundaries(db.db);
    spdlog::info("  Found {} ligands in merged DMS", boundaries.size());
    int n_ff_unpopulated = 0;  // ligands with FF tables but not every atom parameterised
    const size_t first_ligand = out_flex_list.size();  // the ligands this read appends start here
    check_registered_tables(db.db, true, nullptr, 0);

    // Molecule-keyed tables, each read once (see MolBuckets). A row keeps the column values exactly as
    // the per-ligand loop reads them (same sqlite3_column_* call on the same stored value).
    struct AtomRow { sqlite3_int64 id; bool id_int; double x, y, z, charge; int vina_type; };
    struct TorsionRow {
        sqlite3_int64 key; bool key_int; int tor_id; int atoms[4]; double dihedral;
        bool mobile_null; std::string mobile;
    };
    struct RangeRow { int tid_type; sqlite3_int64 tid; double lo, hi; };
    struct ChargeRow { int atom_id, formal_charge; };
    struct StereoRow { int p0, p1, stereo; };
    struct PropRow { bool both; std::string key, value; };
    MolBuckets<AtomRow> atom_buckets;
    MolBuckets<int> root_buckets;
    MolBuckets<TorsionRow> torsion_buckets;
    MolBuckets<RangeRow> range_buckets;
    MolBuckets<ChargeRow> charge_buckets;
    MolBuckets<StereoRow> stereo_buckets;
    MolBuckets<PropRow> prop_buckets;
    bool props_scan_order = false;   // ud2_sdf_properties has no index: per-ligand rows come in rowid order
    auto atom_row = [](SqliteStmt& q, int c) {
        AtomRow r;
        r.id_int = sqlite3_column_type(q.stmt, c) == SQLITE_INTEGER;
        r.id = sqlite3_column_int64(q.stmt, c);
        r.x = q.col_dbl(c + 1);
        r.y = q.col_dbl(c + 2);
        r.z = q.col_dbl(c + 3);
        r.charge = q.col_dbl(c + 4);
        r.vina_type = q.col_int(c + 5);
        return r;
    };
    auto torsion_row = [](SqliteStmt& q, int c) {
        TorsionRow r;
        r.key_int = sqlite3_column_type(q.stmt, c) == SQLITE_INTEGER;
        r.key = sqlite3_column_int64(q.stmt, c);
        r.tor_id = q.col_int(c);
        for (int k = 0; k < 4; k++) r.atoms[k] = q.col_int(c + 1 + k);
        r.dihedral = q.col_dbl(c + 5);
        const char* text = reinterpret_cast<const char*>(sqlite3_column_text(q.stmt, c + 6));
        r.mobile_null = text == nullptr;
        if (text) r.mobile = text;
        return r;
    };
    auto charge_row = [](SqliteStmt& q, int c) { return ChargeRow{q.col_int(c), q.col_int(c + 1)}; };
    auto stereo_row = [](SqliteStmt& q, int c) { return StereoRow{q.col_int(c), q.col_int(c + 1), q.col_int(c + 2)}; };
    auto prop_row = [](SqliteStmt& q, int c) {
        const char* k = reinterpret_cast<const char*>(sqlite3_column_text(q.stmt, c));
        const char* v = reinterpret_cast<const char*>(sqlite3_column_text(q.stmt, c + 1));
        PropRow r;
        r.both = k && v;
        if (r.both) { r.key = k; r.value = v; }
        return r;
    };

    for (const auto& mb : boundaries) {
        int off = mb.atom_offset;
        int na = mb.natom;
        int mol_id = mb.molecule_id;

        UDFlexMol flex;
        FFMinData ff;
        SDFMolData sdf;

        // --- 1. Per-atom data (coords, types) ---
        {
            if (!atom_buckets.loaded) {
                load_mol_buckets(db.db,
                    "SELECT ua.molecule_id, p.id, p.x, p.y, p.z, p.charge, ua.vina_type "
                    "FROM particle p "
                    "JOIN ud2_atom_props ua ON p.id = ua.atom_id",
                    atom_buckets, [&](SqliteStmt& q) { return atom_row(q, 1); });
            }
            // ORDER BY p.id: the sorted bucket is that order when the ids are distinct integers
            std::vector<AtomRow> rows;
            bool from_bucket = atom_buckets.usable;
            if (from_bucket) {
                rows = atom_buckets.of(mol_id);
                for (const auto& r : rows) from_bucket = from_bucket && r.id_int;
            }
            if (from_bucket) {
                std::sort(rows.begin(), rows.end(),
                          [](const AtomRow& a, const AtomRow& b) { return a.id < b.id; });
                for (size_t i = 1; i < rows.size(); i++) from_bucket = from_bucket && rows[i - 1].id < rows[i].id;
            }
            if (!from_bucket) {
                std::string sql =
                    "SELECT p.id, p.x, p.y, p.z, p.charge, "
                    "       ua.vina_type "
                    "FROM particle p "
                    "JOIN ud2_atom_props ua ON p.id = ua.atom_id "
                    "WHERE ua.molecule_id = " + std::to_string(mol_id) +
                    " ORDER BY p.id";
                rows = query_rows<AtomRow>(db.db, sql, [&](SqliteStmt& q) { return atom_row(q, 0); });
            }

            Real coords_sum[3] = {0};
            for (const auto& r : rows) {
                Real x = static_cast<Real>(r.x);
                Real y = static_cast<Real>(r.y);
                Real z = static_cast<Real>(r.z);
                Real charge = static_cast<Real>(r.charge);
                int vina_type = r.vina_type;

                flex.coords.push_back(x);
                flex.coords.push_back(y);
                flex.coords.push_back(z);
                flex.vina_types.push_back(vina_type);
                flex.charges.push_back(charge);
                coords_sum[0] += x;
                coords_sum[1] += y;
                coords_sum[2] += z;
            }
            flex.natom = static_cast<int>(flex.charges.size());
            if (flex.natom > 0) {
                flex.center[0] = coords_sum[0] / flex.natom;
                flex.center[1] = coords_sum[1] / flex.natom;
                flex.center[2] = coords_sum[2] / flex.natom;
            }
        }

        // Molecule name: prefer original_name (input SDF _Name),
        // fall back to molecule_name, then ud2_molecules.mol_name
        flex.name = mb.mol_name;
        if (table_exists(db.db, "ud2_metadata")) {
            // Try original_name first
            std::string sql_orig =
                "SELECT value FROM ud2_metadata WHERE molecule_id = " +
                std::to_string(mol_id) + " AND key = 'original_name'";
            SqliteStmt qo(db.db, sql_orig);
            if (qo.step()) {
                const char* val = reinterpret_cast<const char*>(
                    sqlite3_column_text(qo.stmt, 0));
                if (val && val[0]) flex.name = val;
            }
            // Fall back to molecule_name if original_name was empty
            if (flex.name == mb.mol_name) {
                std::string sql =
                    "SELECT value FROM ud2_metadata WHERE molecule_id = " +
                    std::to_string(mol_id) + " AND key = 'molecule_name'";
                SqliteStmt qm(db.db, sql);
                if (qm.step()) {
                    const char* val = reinterpret_cast<const char*>(
                        sqlite3_column_text(qm.stmt, 0));
                    if (val) flex.name = val;
                }
            }
        }

        spdlog::info("  Ligand {} ({}): {} atoms, offset {}",
                      mol_id, flex.name, flex.natom, off);

        // --- 2. Read 1-4 pairs (from pair_12_6_es_term, absent when no FF) ---
        if (table_exists(db.db, "pair_12_6_es_term")) {
            std::string sql =
                "SELECT p0, p1 FROM pair_12_6_es_term "
                "WHERE p0 >= " + std::to_string(off) +
                " AND p0 < " + std::to_string(off + na) +
                " AND p1 >= " + std::to_string(off) +
                " AND p1 < " + std::to_string(off + na);
            SqliteStmt q(db.db, sql);
            while (q.step()) {
                int i = q.col_int(0) - off;  // rebase to 0
                int j = q.col_int(1) - off;
                flex.pairs_14.insert(std::make_pair(std::min(i,j), std::max(i,j)));
            }
        }

        // --- 3. Exclusions (1-2 & 1-3) ---
        if (table_exists(db.db, "exclusion")) {
            std::string sql =
                "SELECT p0, p1 FROM exclusion "
                "WHERE p0 >= " + std::to_string(off) +
                " AND p0 < " + std::to_string(off + na) +
                " AND p1 >= " + std::to_string(off) +
                " AND p1 < " + std::to_string(off + na);
            SqliteStmt q(db.db, sql);
            while (q.step()) {
                int i = q.col_int(0) - off;
                int j = q.col_int(1) - off;
                auto p = std::make_pair(std::min(i,j), std::max(i,j));
                if (flex.pairs_14.find(p) == flex.pairs_14.end()) {
                    flex.pairs_1213.insert(p);
                }
            }
        }
        if (flex.pairs_1213.empty() && flex.pairs_14.empty()) {
            derive_bond_path_exclusions(db.db, off, na, flex.pairs_1213, flex.pairs_14);
        }
        ligand_charge_annotations(db.db, off, na, flex.vina_types, type_props, flex.charges,
                                  flex.charges_ua, flex.charged_group, &flex.polar_env_class);
        read_ligand_scoring_annotations(db.db, off, na, flex.aromatic, flex.torsion_penalties);

        // --- 4. Root atoms ---
        std::set<int> root_atoms;
        {
            if (!root_buckets.loaded) {
                load_mol_buckets(db.db, "SELECT molecule_id, atom_id FROM ud2_root_atoms",
                                 root_buckets, [](SqliteStmt& q) { return q.col_int(1); });
            }
            // a set: the row order does not matter
            std::vector<int> rows;
            if (root_buckets.usable) {
                rows = root_buckets.of(mol_id);
            } else {
                std::string sql =
                    "SELECT atom_id FROM ud2_root_atoms "
                    "WHERE molecule_id = " + std::to_string(mol_id);
                rows = query_rows<int>(db.db, sql, [](SqliteStmt& q) { return q.col_int(0); });
            }
            for (int atom_id : rows) {
                root_atoms.insert(atom_id - off);  // rebase
            }
        }

        // --- 5. Torsions ---
        {
            if (!torsion_buckets.loaded) {
                load_mol_buckets(db.db,
                    "SELECT molecule_id, torsion_id, atom_i, atom_j, atom_k, atom_l, "
                    "       dihedral_value, mobile_atoms "
                    "FROM ud2_torsions",
                    torsion_buckets, [&](SqliteStmt& q) { return torsion_row(q, 1); });
            }
            // ORDER BY torsion_id: the sorted bucket is that order when the ids are distinct integers
            std::vector<TorsionRow> rows;
            bool from_bucket = torsion_buckets.usable;
            if (from_bucket) {
                rows = torsion_buckets.of(mol_id);
                for (const auto& r : rows) from_bucket = from_bucket && r.key_int;
            }
            if (from_bucket) {
                std::sort(rows.begin(), rows.end(),
                          [](const TorsionRow& a, const TorsionRow& b) { return a.key < b.key; });
                for (size_t i = 1; i < rows.size(); i++) from_bucket = from_bucket && rows[i - 1].key < rows[i].key;
            }
            if (!from_bucket) {
                std::string sql =
                    "SELECT torsion_id, atom_i, atom_j, atom_k, atom_l, "
                    "       dihedral_value, mobile_atoms "
                    "FROM ud2_torsions WHERE molecule_id = " + std::to_string(mol_id) +
                    " ORDER BY torsion_id";
                rows = query_rows<TorsionRow>(db.db, sql, [&](SqliteStmt& q) { return torsion_row(q, 0); });
            }

            for (const auto& r : rows) {
                int tor_id = r.tor_id;
                UDTorsion torsion;
                torsion.atoms[0] = r.atoms[0] - off;
                torsion.atoms[1] = r.atoms[1] - off;
                torsion.atoms[2] = r.atoms[2] - off;
                torsion.atoms[3] = r.atoms[3] - off;
                torsion.axis[0] = torsion.atoms[1];
                torsion.axis[1] = torsion.atoms[2];

                Real dihedral_deg = static_cast<Real>(r.dihedral);
                flex.dihedrals.push_back(dihedral_deg * static_cast<Real>(M_PI / 180.0));

                // Parse mobile atoms from CSV text and rebase
                const char* mobile_text = r.mobile_null ? nullptr : r.mobile.c_str();
                auto mobile = parse_csv_ints(mobile_text);
                for (auto& a : mobile) a -= off;
                torsion.rotated_atoms = std::move(mobile);

                // Read torsion ranges
                if (use_tor_lib && table_exists(db.db, "ud2_torsion_ranges")) {
                    if (!range_buckets.loaded) {
                        load_mol_buckets(db.db,
                            "SELECT molecule_id, torsion_id, range_lo, range_hi "
                            "FROM ud2_torsion_ranges ORDER BY rowid",
                            range_buckets, [](SqliteStmt& q) {
                                return RangeRow{sqlite3_column_type(q.stmt, 1), sqlite3_column_int64(q.stmt, 1),
                                                q.col_dbl(2), q.col_dbl(3)};
                            });
                    }
                    // WHERE torsion_id = tor_id ... ORDER BY rowid: the bucket rows with that integer
                    // torsion_id, in rowid order, unless a row of this molecule stores another kind of value
                    std::vector<RangeRow> range_rows;
                    bool ranges_from_bucket = range_buckets.usable;
                    if (ranges_from_bucket) {
                        for (const auto& rr : range_buckets.of(mol_id)) {
                            if (rr.tid_type != SQLITE_INTEGER && rr.tid_type != SQLITE_NULL) ranges_from_bucket = false;
                            if (rr.tid_type == SQLITE_INTEGER && rr.tid == tor_id) range_rows.push_back(rr);
                        }
                    }
                    if (!ranges_from_bucket) {
                        std::string sql_r =
                            "SELECT range_lo, range_hi FROM ud2_torsion_ranges "
                            "WHERE torsion_id = " + std::to_string(tor_id) +
                            " AND molecule_id = " + std::to_string(mol_id) +
                            " ORDER BY rowid";
                        range_rows = query_rows<RangeRow>(db.db, sql_r, [](SqliteStmt& q) {
                            return RangeRow{SQLITE_NULL, 0, q.col_dbl(0), q.col_dbl(1)};
                        });
                    }
                    for (const auto& rr : range_rows) {
                        Real lo = static_cast<Real>(rr.lo * M_PI / 180.0);
                        Real hi = static_cast<Real>(rr.hi * M_PI / 180.0);
                        if (lo < hi) {
                            torsion.range_list.push_back(lo);
                            torsion.range_list.push_back(hi);
                        } else if (lo > 0 && hi < 0) {
                            torsion.range_list.push_back(lo);
                            torsion.range_list.push_back(static_cast<Real>(M_PI));
                            torsion.range_list.push_back(static_cast<Real>(-M_PI));
                            torsion.range_list.push_back(hi);
                        } else {
                            spdlog::critical("DMS torsion range error: lo={}, hi={}", lo, hi);
                        }
                    }
                }
                if (torsion.range_list.empty()) {
                    torsion.range_list.push_back(static_cast<Real>(-M_PI));
                    torsion.range_list.push_back(static_cast<Real>(M_PI));
                }

                flex.torsions.push_back(torsion);
            }
        }
        spdlog::info("    Torsions: {}", flex.torsions.size());

        // --- 6. Compute fragments, then intra_pairs ---
        {
            std::vector<std::set<int>> frags;
            split_torsions_into_frags(root_atoms, flex.torsions, frags);

            for (int i = 0; i < flex.natom; i++) {
                if (type_props.is_hydrogen[flex.vina_types[i]]) continue;  // zero energy, dead weight
                for (int j = i + 1; j < flex.natom; j++) {
                    if (type_props.is_hydrogen[flex.vina_types[j]]) continue;
                    auto p = std::make_pair(i, j);
                    if (flex.pairs_1213.find(p) == flex.pairs_1213.end() &&
                        flex.pairs_14.find(p) == flex.pairs_14.end() &&
                        !checkInAnySameSet(frags, i, j)) {
                        flex.intra_pairs.push_back(i);
                        flex.intra_pairs.push_back(j);
                    }
                }
            }
        }
        spdlog::info("    Intra pairs: {}", flex.intra_pairs.size() / 2);
        const int n_ff_only = split_ff_only_torsions(flex, type_props);
        if (n_ff_only > 0) {
            spdlog::info("    {} hydrogen-only torsions removed from the search (kept for FF minimisation)", n_ff_only);
        }

        // --- 7. Read FF params for this molecule (skip when FF tables absent) ---
        bool has_ff_tables = table_exists(db.db, "nonbonded_param") &&
                             table_exists(db.db, "pair_12_6_es_term");
        if (has_ff_tables) {
            // Per-atom FF params (ligand)
            {
                std::string sql =
                    "SELECT p.id, p.anum, p.charge, p.x, p.y, p.z, np.sigma, np.epsilon "
                    "FROM particle p "
                    "JOIN nonbonded_param np ON p.nbtype = np.id "
                    "WHERE p.id >= " + std::to_string(off) +
                    " AND p.id < " + std::to_string(off + na) +
                    " ORDER BY p.id";
                SqliteStmt q(db.db, sql);
                while (q.step()) {
                    ff.lig_atoms.anum.push_back(q.col_int(1));
                    ff.lig_atoms.charge.push_back(static_cast<Real>(q.col_dbl(2)));
                    ff.lig_atoms.x.push_back(static_cast<Real>(q.col_dbl(3)));
                    ff.lig_atoms.y.push_back(static_cast<Real>(q.col_dbl(4)));
                    ff.lig_atoms.z.push_back(static_cast<Real>(q.col_dbl(5)));
                    ff.lig_atoms.sigma.push_back(static_cast<Real>(q.col_dbl(6)));
                    ff.lig_atoms.epsilon.push_back(static_cast<Real>(q.col_dbl(7)));
                }
                ff.lig_atoms.natom = static_cast<int>(ff.lig_atoms.anum.size());
            }
        }
        // The FF arrays are index-aligned with the flex atoms: a ligand gets FF data only when
        // every one of its atoms has nonbonded parameters
        const bool ff_populated = has_ff_tables && na > 0 && ff.lig_atoms.natom == na;
        if (has_ff_tables && !ff_populated) {
            n_ff_unpopulated++;
            ff = FFMinData();
        }
        if (ff_populated) {
            // 1-4 pairs (pre-scaled, rebased)
            {
                std::string sql =
                    "SELECT t.p0, t.p1, p.aij, p.bij, p.qij "
                    "FROM pair_12_6_es_term t "
                    "JOIN pair_12_6_es_param p ON t.param = p.id "
                    "WHERE t.p0 >= " + std::to_string(off) +
                    " AND t.p0 < " + std::to_string(off + na) +
                    " AND t.p1 >= " + std::to_string(off) +
                    " AND t.p1 < " + std::to_string(off + na);
                SqliteStmt q(db.db, sql);
                while (q.step()) {
                    FFPair14 pair;
                    pair.i   = q.col_int(0) - off;
                    pair.j   = q.col_int(1) - off;
                    pair.aij = static_cast<Real>(q.col_dbl(2));
                    pair.bij = static_cast<Real>(q.col_dbl(3));
                    pair.qij = static_cast<Real>(q.col_dbl(4));
                    ff.lig_pairs_14.push_back(pair);
                }
            }

            // Torsion parameters (rebased)
            if (table_exists(db.db, "dihedral_trig_term")) {
                std::string sql =
                    "SELECT t.p0, t.p1, t.p2, t.p3, "
                    "p.phi0, p.fc0, p.fc1, p.fc2, p.fc3, p.fc4, p.fc5, p.fc6 "
                    "FROM dihedral_trig_term t "
                    "JOIN dihedral_trig_param p ON t.param = p.id "
                    "WHERE t.p0 >= " + std::to_string(off) +
                    " AND t.p0 < " + std::to_string(off + na) +
                    " AND t.p3 >= " + std::to_string(off) +
                    " AND t.p3 < " + std::to_string(off + na);
                SqliteStmt q(db.db, sql);
                while (q.step()) {
                    FFTorsion tor;
                    tor.p0 = q.col_int(0) - off;
                    tor.p1 = q.col_int(1) - off;
                    tor.p2 = q.col_int(2) - off;
                    tor.p3 = q.col_int(3) - off;
                    tor.phi0 = static_cast<Real>(q.col_dbl(4) * M_PI / 180.0);
                    for (int k = 0; k < 7; k++) {
                        tor.fc[k] = static_cast<Real>(q.col_dbl(5 + k));
                    }
                    ff.lig_torsions.push_back(tor);
                }
            }

            // Exclusions (rebased)
            if (table_exists(db.db, "exclusion")) {
                std::string sql =
                    "SELECT p0, p1 FROM exclusion "
                    "WHERE p0 >= " + std::to_string(off) +
                    " AND p0 < " + std::to_string(off + na) +
                    " AND p1 >= " + std::to_string(off) +
                    " AND p1 < " + std::to_string(off + na);
                SqliteStmt q(db.db, sql);
                while (q.step()) {
                    FFExclusion ex;
                    ex.i = q.col_int(0) - off;
                    ex.j = q.col_int(1) - off;
                    ff.lig_exclusions.push_back(ex);
                }
            }

            // Bond harmonic parameters (rebased; optional)
            if (table_exists(db.db, "stretch_harm_term")) {
                std::string sql =
                    "SELECT t.p0, t.p1, p.r0, p.fc "
                    "FROM stretch_harm_term t "
                    "JOIN stretch_harm_param p ON t.param = p.id "
                    "WHERE t.p0 >= " + std::to_string(off) +
                    " AND t.p0 < " + std::to_string(off + na) +
                    " AND t.p1 >= " + std::to_string(off) +
                    " AND t.p1 < " + std::to_string(off + na);
                SqliteStmt q(db.db, sql);
                while (q.step()) {
                    FFBond b;
                    b.i  = q.col_int(0) - off;
                    b.j  = q.col_int(1) - off;
                    b.r0 = static_cast<Real>(q.col_dbl(2));
                    b.fc = static_cast<Real>(q.col_dbl(3));
                    ff.lig_bonds.push_back(b);
                }
            }

            // Angle harmonic parameters (rebased; optional)
            if (table_exists(db.db, "angle_harm_term")) {
                std::string sql =
                    "SELECT t.p0, t.p1, t.p2, p.theta0, p.fc "
                    "FROM angle_harm_term t "
                    "JOIN angle_harm_param p ON t.param = p.id "
                    "WHERE t.p0 >= " + std::to_string(off) +
                    " AND t.p0 < " + std::to_string(off + na) +
                    " AND t.p1 >= " + std::to_string(off) +
                    " AND t.p1 < " + std::to_string(off + na) +
                    " AND t.p2 >= " + std::to_string(off) +
                    " AND t.p2 < " + std::to_string(off + na);
                SqliteStmt q(db.db, sql);
                while (q.step()) {
                    FFAngle a;
                    a.i      = q.col_int(0) - off;
                    a.j      = q.col_int(1) - off;
                    a.k      = q.col_int(2) - off;
                    a.theta0 = static_cast<Real>(q.col_dbl(3) * M_PI / 180.0);
                    a.fc     = static_cast<Real>(q.col_dbl(4));
                    ff.lig_angles.push_back(a);
                }
            }

            ff.build_pair_lookup();
            ff.loaded = true;
        }
        spdlog::info("    FF: {} atoms, {} 1-4 pairs, {} torsions, {} excl, {} bonds, {} angles",
                      ff.lig_atoms.natom, ff.lig_pairs_14.size(),
                      ff.lig_torsions.size(), ff.lig_exclusions.size(),
                      ff.lig_bonds.size(), ff.lig_angles.size());

        // --- 8. Extension tables (register_table_reader: ud2_bias in ext/bias/bias_reader.cpp, ud2_ring_bias in
        //        ext/bias/ring_reader.cpp), when present ---
        read_registered_tables(db.db, TableScope{off, na, true}, flex);

        // --- 9. SDF output data ---
        // Atomic numbers
        {
            std::string sql =
                "SELECT id, anum FROM particle "
                "WHERE id >= " + std::to_string(off) +
                " AND id < " + std::to_string(off + na) +
                " ORDER BY id";
            SqliteStmt q(db.db, sql);
            while (q.step()) {
                sdf.anum.push_back(q.col_int(1));
            }
            sdf.natom = static_cast<int>(sdf.anum.size());
        }
        sdf.formal_charge.assign(sdf.natom, 0);

        // Bonds (rebased)
        {
            std::string sql =
                "SELECT p0, p1, [order] FROM bond "
                "WHERE p0 >= " + std::to_string(off) +
                " AND p0 < " + std::to_string(off + na) +
                " AND p1 >= " + std::to_string(off) +
                " AND p1 < " + std::to_string(off + na);
            SqliteStmt q(db.db, sql);
            while (q.step()) {
                SDFBond b;
                b.p0 = q.col_int(0) - off;
                b.p1 = q.col_int(1) - off;
                b.order = q.col_int(2);
                b.stereo = 0;
                sdf.bonds.push_back(b);
            }
        }

        // Formal charges
        if (table_exists(db.db, "ud2_formal_charges")) {
            if (!charge_buckets.loaded) {
                load_mol_buckets(db.db, "SELECT molecule_id, atom_id, formal_charge FROM ud2_formal_charges",
                                 charge_buckets, [&](SqliteStmt& q) { return charge_row(q, 1); });
            }
            // the last row of an atom wins: the row order does not matter when the atom ids are distinct
            std::vector<ChargeRow> rows;
            bool from_bucket = charge_buckets.usable;
            if (from_bucket) {
                rows = charge_buckets.of(mol_id);
                std::vector<int> ids;
                for (const auto& r : rows) ids.push_back(r.atom_id);
                std::sort(ids.begin(), ids.end());
                from_bucket = std::adjacent_find(ids.begin(), ids.end()) == ids.end();
            }
            if (!from_bucket) {
                std::string sql =
                    "SELECT atom_id, formal_charge FROM ud2_formal_charges "
                    "WHERE molecule_id = " + std::to_string(mol_id);
                rows = query_rows<ChargeRow>(db.db, sql, [&](SqliteStmt& q) { return charge_row(q, 0); });
            }
            for (const auto& r : rows) {
                int aid = r.atom_id - off;
                int fc  = r.formal_charge;
                if (aid >= 0 && aid < sdf.natom) {
                    sdf.formal_charge[aid] = fc;
                }
            }
        }

        // Bond stereo (rebased)
        if (table_exists(db.db, "ud2_bond_stereo")) {
            if (!stereo_buckets.loaded) {
                load_mol_buckets(db.db, "SELECT molecule_id, bond_p0, bond_p1, stereo FROM ud2_bond_stereo",
                                 stereo_buckets, [&](SqliteStmt& q) { return stereo_row(q, 1); });
            }
            // the last row of a bond wins: the row order does not matter when the bonds are distinct
            std::vector<StereoRow> rows;
            bool from_bucket = stereo_buckets.usable;
            if (from_bucket) {
                rows = stereo_buckets.of(mol_id);
                std::vector<std::pair<int, int>> bonds;
                for (const auto& r : rows) bonds.emplace_back(r.p0, r.p1);
                std::sort(bonds.begin(), bonds.end());
                from_bucket = std::adjacent_find(bonds.begin(), bonds.end()) == bonds.end();
            }
            if (!from_bucket) {
                std::string sql =
                    "SELECT bond_p0, bond_p1, stereo FROM ud2_bond_stereo "
                    "WHERE molecule_id = " + std::to_string(mol_id);
                rows = query_rows<StereoRow>(db.db, sql, [&](SqliteStmt& q) { return stereo_row(q, 0); });
            }
            for (const auto& r : rows) {
                int p0 = r.p0 - off;
                int p1 = r.p1 - off;
                int st = r.stereo;
                for (auto& b : sdf.bonds) {
                    if (b.p0 == p0 && b.p1 == p1) {
                        b.stereo = st;
                        break;
                    }
                }
            }
        }

        // User SDF properties
        if (table_exists(db.db, "ud2_sdf_properties")) {
            if (!prop_buckets.loaded) {
                props_scan_order = !table_has_index(db.db, "ud2_sdf_properties");
                load_mol_buckets(db.db, "SELECT molecule_id, key, value FROM ud2_sdf_properties",
                                 prop_buckets, [&](SqliteStmt& q) { return prop_row(q, 1); });
            }
            // no ORDER BY: without an index both queries can only scan the table, so both return rowid
            // order; with one, the bucket stands in only for a ligand with at most one row
            std::vector<PropRow> rows;
            bool from_bucket = prop_buckets.usable;
            if (from_bucket) {
                rows = prop_buckets.of(mol_id);
                from_bucket = props_scan_order || rows.size() <= 1;
            }
            if (!from_bucket) {
                std::string sql =
                    "SELECT key, value FROM ud2_sdf_properties "
                    "WHERE molecule_id = " + std::to_string(mol_id);
                rows = query_rows<PropRow>(db.db, sql, [&](SqliteStmt& q) { return prop_row(q, 0); });
            }
            for (const auto& r : rows) {
                if (r.both) {
                    sdf.properties.push_back({r.key, r.value});
                }
            }
        }

        // Molecule name for SDF output: prefer original_name
        if (table_exists(db.db, "ud2_metadata")) {
            std::string sql1 =
                "SELECT value FROM ud2_metadata "
                "WHERE molecule_id = " + std::to_string(mol_id) +
                " AND key = 'original_name'";
            SqliteStmt q1(db.db, sql1);
            if (q1.step()) {
                const char* val = reinterpret_cast<const char*>(
                    sqlite3_column_text(q1.stmt, 0));
                if (val) sdf.mol_name = val;
            }
            if (sdf.mol_name.empty()) {
                std::string sql2 =
                    "SELECT value FROM ud2_metadata "
                    "WHERE molecule_id = " + std::to_string(mol_id) +
                    " AND key = 'molecule_name'";
                SqliteStmt q2(db.db, sql2);
                if (q2.step()) {
                    const char* val = reinterpret_cast<const char*>(
                        sqlite3_column_text(q2.stmt, 0));
                    if (val) sdf.mol_name = val;
                }
            }
        }

        sdf.loaded = true;

        out_flex_list.push_back(std::move(flex));
        out_ff_list.push_back(std::move(ff));
        out_sdf_list.push_back(std::move(sdf));
    }
    check_registered_tables(db.db, false, out_flex_list.data() + first_ligand, out_flex_list.size() - first_ligand);

    if (n_ff_unpopulated > 0) {
        spdlog::warn("  {} of {} ligands have FF tables without nonbonded parameters for every atom; "
                     "FF scoring disabled for them", n_ff_unpopulated, boundaries.size());
    }
    spdlog::info("Multi-ligand DMS loaded: {} ligands", boundaries.size());
}


// ===================== SDF output data reader =====================

void read_sdf_mol_data(const std::string& dms_path, SDFMolData& out) {
    spdlog::info("Reading SDF mol data from DMS: {}", dms_path);
    SqliteDB db(dms_path);

    // Check for sdf_natom in ud2_metadata (PROTAC: write only first N atoms)
    int sdf_natom = -1;  // -1 = no limit, use all atoms
    if (table_exists(db.db, "ud2_metadata")) {
        SqliteStmt q(db.db, "SELECT value FROM ud2_metadata WHERE key = 'sdf_natom'");
        if (q.step()) {
            const char* val = reinterpret_cast<const char*>(
                sqlite3_column_text(q.stmt, 0));
            if (val) sdf_natom = std::atoi(val);
        }
    }
    if (sdf_natom > 0) {
        spdlog::info("  sdf_natom={} (filtering to first {} atoms)", sdf_natom, sdf_natom);
    }

    // 1. Atomic numbers from particle table
    {
        std::string sql = (sdf_natom > 0)
            ? "SELECT id, anum FROM particle WHERE id < " + std::to_string(sdf_natom) + " ORDER BY id"
            : "SELECT id, anum FROM particle ORDER BY id";
        SqliteStmt q(db.db, sql);
        while (q.step()) {
            out.anum.push_back(q.col_int(1));
        }
        out.natom = static_cast<int>(out.anum.size());
    }
    spdlog::info("  SDF atoms: {}", out.natom);

    // Initialise formal charges to zero
    out.formal_charge.assign(out.natom, 0);

    // 2. Bond table (p0, p1, order)
    {
        std::string sql = (sdf_natom > 0)
            ? "SELECT p0, p1, [order] FROM bond WHERE p0 < "
              + std::to_string(sdf_natom) + " AND p1 < " + std::to_string(sdf_natom)
            : "SELECT p0, p1, [order] FROM bond";
        SqliteStmt q(db.db, sql);
        while (q.step()) {
            SDFBond b;
            b.p0 = q.col_int(0);
            b.p1 = q.col_int(1);
            b.order = q.col_int(2);
            b.stereo = 0;
            out.bonds.push_back(b);
        }
    }
    spdlog::info("  SDF bonds: {}", out.bonds.size());

    // 3. Formal charges from ud2_formal_charges (sparse, only non-zero)
    if (table_exists(db.db, "ud2_formal_charges")) {
        SqliteStmt q(db.db, "SELECT atom_id, formal_charge FROM ud2_formal_charges");
        while (q.step()) {
            int aid = q.col_int(0);
            int fc  = q.col_int(1);
            if (aid >= 0 && aid < out.natom) {
                out.formal_charge[aid] = fc;
            }
        }
    }

    // 4. Bond stereo from ud2_bond_stereo (sparse, only non-zero)
    if (table_exists(db.db, "ud2_bond_stereo")) {
        SqliteStmt q(db.db, "SELECT bond_p0, bond_p1, stereo FROM ud2_bond_stereo");
        while (q.step()) {
            int p0 = q.col_int(0);
            int p1 = q.col_int(1);
            int st = q.col_int(2);
            // Find the matching bond and set its stereo
            for (auto& b : out.bonds) {
                if (b.p0 == p0 && b.p1 == p1) {
                    b.stereo = st;
                    break;
                }
            }
        }
    }

    // 5. User SDF properties from ud2_sdf_properties
    if (table_exists(db.db, "ud2_sdf_properties")) {
        SqliteStmt q(db.db, "SELECT key, value FROM ud2_sdf_properties");
        while (q.step()) {
            const char* k = reinterpret_cast<const char*>(
                sqlite3_column_text(q.stmt, 0));
            const char* v = reinterpret_cast<const char*>(
                sqlite3_column_text(q.stmt, 1));
            if (k && v) {
                out.properties.push_back({std::string(k), std::string(v)});
            }
        }
    }

    // 6. Molecule name: prefer original_name (from input SDF _Name),
    //    fall back to molecule_name (internal MOL_0)
    if (table_exists(db.db, "ud2_metadata")) {
        SqliteStmt q1(db.db, "SELECT value FROM ud2_metadata WHERE key = 'original_name'");
        if (q1.step()) {
            const char* val = reinterpret_cast<const char*>(
                sqlite3_column_text(q1.stmt, 0));
            if (val) out.mol_name = val;
        }
        if (out.mol_name.empty()) {
            SqliteStmt q2(db.db, "SELECT value FROM ud2_metadata WHERE key = 'molecule_name'");
            if (q2.step()) {
                const char* val = reinterpret_cast<const char*>(
                    sqlite3_column_text(q2.stmt, 0));
                if (val) out.mol_name = val;
            }
        }
    }

    out.loaded = true;
    spdlog::info("  SDF mol data loaded: {} atoms, {} bonds, {} properties",
                 out.natom, out.bonds.size(), out.properties.size());
}
