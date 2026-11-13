//
// DMS file reader via SQLite3.
// Reads force field parameters directly from DMS files produced by
// uff prm (ligand, GAFF2) and utop prm (protein, ff14SB).
//
// Also reads enriched DMS files (with ud2_* tables written by Python)
// to populate UDFlexMol / UDFixMol for the docking engine, replacing JSON.
//

#ifndef DMS_READER_H
#define DMS_READER_H

#include <string>
#include <utility>
#include <vector>
#include "myutils/common.h"
#include "model/model.h"
#include "constants/constants.h"

struct sqlite3;   // <sqlite3.h> typedefs this struct; the readers below take the open handle

/// Per-atom force field parameters read from a DMS file.
struct FFAtomParams {
    std::vector<Real> sigma;     // LJ sigma, Angstrom
    std::vector<Real> epsilon;   // LJ epsilon, kcal/mol
    std::vector<Real> charge;    // partial charge, elementary charge units
    std::vector<int>  anum;      // atomic number
    std::vector<Real> x, y, z;   // coordinates (for atom index mapping)
    int natom = 0;
};

/// Pre-scaled 1-4 pair interaction from DMS pair_12_6_es table.
struct FFPair14 {
    int i, j;       // atom indices (DMS ordering)
    Real aij;       // LJ A parameter (kcal/mol * A^12), pre-scaled
    Real bij;       // LJ B parameter (kcal/mol * A^6), pre-scaled
    Real qij;       // charge product, pre-scaled
};

/// Torsion parameters from DMS dihedral_trig table.
struct FFTorsion {
    int p0, p1, p2, p3;   // atom indices (DMS ordering)
    Real phi0;             // phase angle, radians
    Real fc[7];            // fc0..fc6 for periodicities 0-6, kcal/mol
};

/// Harmonic bond stretch from DMS stretch_harm table.
/// E = fc * (r - r0)^2   (fc absorbs the 1/2 factor)
struct FFBond {
    int i, j;     // atom indices (DMS ordering)
    Real r0;      // equilibrium length, Angstrom
    Real fc;      // force constant, kcal/mol/A^2
};

/// Harmonic angle bend from DMS angle_harm table.
/// E = fc * (theta - theta0)^2  (theta0 stored in radians after conversion)
struct FFAngle {
    int i, j, k;  // atom indices, j is the vertex (DMS ordering)
    Real theta0;  // equilibrium angle, radians (converted from degrees at load)
    Real fc;      // force constant, kcal/mol/rad^2
};

/// Exclusion pair (1-2 and 1-3 bonded pairs that are excluded from nonbonded).
struct FFExclusion {
    int i, j;
};

/// Pair type codes for the precomputed exclusion lookup.
enum FFPairType : uint8_t {
    FF_PAIR_NORMAL  = 0,   // compute full nonbonded interaction
    FF_PAIR_EXCLUDE = 1,   // 1-2 or 1-3: skip entirely
    FF_PAIR_14      = 2,   // 1-4: use pre-scaled pair14 params
};

/// All force field data needed for FF minimization.
struct FFMinData {
    FFAtomParams lig_atoms;                  // ligand per-atom FF params
    FFAtomParams rec_atoms;                  // receptor per-atom FF params
    std::vector<FFPair14> lig_pairs_14;      // ligand 1-4 pairs (pre-scaled)
    std::vector<FFTorsion> lig_torsions;     // ligand torsion params
    std::vector<FFExclusion> lig_exclusions; // ligand 1-2/1-3 exclusions
    std::vector<FFBond>  lig_bonds;          // ligand harmonic bond params
    std::vector<FFAngle> lig_angles;         // ligand harmonic angle params
    // Set by read_ligand_dms when every ligand atom has nonbonded parameters and cleared by it
    // otherwise; cleared by read_receptor_dms when the receptor's nonbonded_param table is empty
    // or partial.
    bool loaded = false;

    // Precomputed triangular lookup: pair_type[tri(i,j)] for i > j.
    // Avoids rebuilding std::set on every energy call.  Built once by
    // build_pair_lookup() after DMS loading.
    std::vector<uint8_t> pair_type;

    /// Triangular index for (a, b) with a > b.
    static inline size_t tri_index(int a, int b) {
        return static_cast<size_t>(a) * (a - 1) / 2 + b;
    }

    /// Build pair_type from lig_exclusions and lig_pairs_14.
    void build_pair_lookup() {
        int n = lig_atoms.natom;
        if (n <= 1) return;
        pair_type.assign(static_cast<size_t>(n) * (n - 1) / 2, FF_PAIR_NORMAL);
        for (const auto& ex : lig_exclusions) {
            int a = std::max(ex.i, ex.j);
            int b = std::min(ex.i, ex.j);
            pair_type[tri_index(a, b)] = FF_PAIR_EXCLUDE;
        }
        for (const auto& p : lig_pairs_14) {
            int a = std::max(p.i, p.j);
            int b = std::min(p.i, p.j);
            pair_type[tri_index(a, b)] = FF_PAIR_14;
        }
    }
};

/// Read ligand force field parameters from a DMS file.
/// Populates lig_atoms, lig_pairs_14, lig_torsions, lig_exclusions, lig_bonds, lig_angles and
/// sets loaded, but only when every particle has nonbonded parameters. Otherwise the ligand FF
/// data is cleared, loaded = false, and one line is logged: info for an absent or empty
/// nonbonded_param table (Vina-only DMS), a warning for a partial one (info with
/// warn_unpopulated = false).
void read_ligand_dms(const std::string& dms_path, FFMinData& out, bool warn_unpopulated = true);

/// Read receptor force field parameters from a DMS file.
/// Populates rec_atoms only (receptor is rigid, no bonded terms needed). When the nonbonded_param
/// table is absent, nothing is read and loaded is left as it is. When it exists but does not cover
/// every particle (empty or partial), rec_atoms stays empty and loaded is cleared (FF scoring
/// needs both sides); that is a warning only if a ligand FF was loaded.
void read_receptor_dms(const std::string& dms_path, FFMinData& out);


// ===================== Extension table registry =====================

/// Where one ligand's rows are in the open file: off / natom its particle id range, merged whether the file is a
/// multi-ligand merge (ud2_molecules), in which case a reader restricts its query to [off, off + natom) and
/// rebases the atom ids by off; a single-ligand file is read whole (off 0, merged false).
struct TableScope {
    int off = 0;
    int natom = 0;
    bool merged = false;
};

/// Reader of one optional per-ligand table (an ud2_* extension table such as ud2_bias): appends the ligand's rows
/// to `out`. Called only when the table (or the entry's companion table) exists.
using TableReader = void (*)(sqlite3* db, const TableScope& scope, UDFlexMol& out);

/// Whole-file check of an extension in a merged read (read_multi_ligand_dms_full), for what no single ligand's read
/// can see (a key repeated across ligands, rows that belong to no ligand). Throws on a bad file. Called once before
/// the first ligand's tables (ligands nullptr, n 0) or once after the last (the ligands this read produced, in
/// order), when the table or its companion exists.
using MergedTableCheck = void (*)(sqlite3* db, const UDFlexMol* ligands, size_t n);

/// One registered extension. `read` runs for every ligand, after the core tables, when `table` exists or, for a
/// two-table extension, its `companion` does (the reader then checks the pair itself: ext/bias/ring_reader.cpp).
struct TableReaderEntry {
    std::string table;
    TableReader read = nullptr;
    std::string companion;                       // "" = a one-table extension
    MergedTableCheck before_merged = nullptr;    // nullptr = no whole-file check
    MergedTableCheck after_merged = nullptr;
};

/// Register `fn` for `table` (one table, no whole-file checks). read_ligand_dms_full and read_multi_ligand_dms_full
/// call the registered readers, in registration order, after the core tables of every ligand. Extensions register
/// from a static object in their own translation unit (ext/bias/bias_reader.cpp, ext/bias/ring_reader.cpp).
void register_table_reader(const char* table, TableReader fn);
/// Register a full entry (companion table, whole-file checks of a merged read).
void register_table_reader(const TableReaderEntry& entry);

/// The registered entries in registration order.
const std::vector<TableReaderEntry>& registered_table_readers();

/// Whether the open DMS has the table (sqlite_master), for registered readers.
bool dms_has_table(sqlite3* db, const char* table);


// ===================== Enriched DMS readers (replace JSON) =====================

/// Read an enriched ligand DMS to populate both UDFlexMol (docking topology)
/// and FFMinData (force field params) from a single file.
/// The DMS must have ud2_atom_props, ud2_torsions, ud2_torsion_ranges,
/// ud2_root_atoms tables (written by Python enrichment). The registered extension tables
/// (register_table_reader) are read after them when present.
void read_ligand_dms_full(const std::string& dms_path,
                          UDFlexMol& out_flex,
                          FFMinData& out_ff,
                          const AtomTypeProps& type_props,
                          bool use_tor_lib);

/// Read an enriched receptor DMS to populate UDFixMol (Vina types + coords
/// inside the box) from ud2_atom_props, and optionally FFMinData.rec_atoms.
void read_receptor_dms_full(const std::string& dms_path,
                            const Box& box,
                            UDFixMol& out_fix,
                            FFMinData& out_ff,
                            const AtomTypeProps& type_props);


// ===================== SDF output data from DMS =====================

/// Bond data for SDF output.
struct SDFBond {
    int p0, p1;   // atom indices (0-based, DMS ordering)
    int order;     // bond order (1=single, 2=double, 3=triple)
    int stereo;    // SDF V2000 stereo flag (0=none, 1=Up, 6=Down)
};

/// Key-value pair for SDF property block.
struct SDFProperty {
    std::string key;
    std::string value;
};

/// Molecular structure data needed to write SDF output.
/// Read from an enriched DMS file; coordinates are NOT included here
/// (they come from the docking engine's pose results).
struct SDFMolData {
    int natom = 0;
    std::vector<int> anum;                  // atomic number per atom
    std::vector<int> formal_charge;         // formal charge per atom (0 for most)
    std::vector<SDFBond> bonds;             // bond table with orders and stereo
    std::vector<SDFProperty> properties;    // user SDF tags (dG_exp, etc.)
    std::string mol_name;                   // molecule name for SDF header
    bool loaded = false;
};

/// Read SDF output data (bonds, formal charges, stereo, user properties)
/// from an enriched ligand DMS file.
void read_sdf_mol_data(const std::string& dms_path, SDFMolData& out);


// ===================== Multi-ligand merged DMS =====================

/// Check if a DMS file contains multiple ligands (has ud2_molecules table).
bool is_multi_ligand_dms(const std::string& dms_path);

/// Read a merged enriched DMS containing N ligands.
/// Populates per-ligand topology (UDFlexMolList), FF data, and SDF data.
void read_multi_ligand_dms_full(const std::string& dms_path,
                                UDFlexMolList& out_flex_list,
                                std::vector<FFMinData>& out_ff_list,
                                std::vector<SDFMolData>& out_sdf_list,
                                const AtomTypeProps& type_props,
                                bool use_tor_lib);

#endif // DMS_READER_H
