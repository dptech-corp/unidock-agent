//
// Spatial Grid for fast receptor neighbor lookup in CPU inter-molecular scoring.
// Replaces O(N_lig * N_rec) brute-force with O(N_lig * ~27 cells) grid query.
//

#ifndef SPATIAL_GRID_H
#define SPATIAL_GRID_H

#include "myutils/common.h"
#include "constants/constants.h"
#include <cstring>
#include <algorithm>
#include <cmath>

struct alignas(16) GridAtom {
    Real  x, y, z; // receptor atom coordinates (copied so a cell walk needs no gather into fix_mol->coords)
    Real  radius;  // pre-computed vdw_radii[type]
    int   index;   // receptor atom index into fix_mol->coords
    int   type;    // Vina atom type (for eval_ef)
    Real  q;       // united-atom charge (Coulomb term; 0 when charges are not provided)
    int   pad;     // 32-byte entries: two 16-byte vector loads on the device
};

// Cells per cutoff radius. 1 gives 8 A cells and a 3x3x3 walk, of which only ~16 % of the
// visited volume is inside the 8 A sphere; finer cells trade cell bookkeeping for fewer
// distance tests (2 -> 27 %, 3 -> 33 %).
#ifndef UD2_GRID_REFINE
#define UD2_GRID_REFINE 1
#endif

struct SpatialGrid {
    Real origin[3];      // grid origin (box_lo - half * cell_size)
    Real inv_cell_size;  // 1.0 / cell_size
    int  half;           // cells to walk on each side: the neighbourhood is (2*half+1)^3
    int  nx, ny, nz;     // cell counts per axis
    int  ncells;         // nx * ny * nz

    // CSR layout: atoms in cell c are atoms[cell_start[c] .. cell_start[c+1])
    int*      cell_start;  // size: ncells + 1
    GridAtom* atoms;       // size: total_atoms (sorted by cell)
    int       total_atoms;

    bool built;            // true after successful build
};

/**
 * @brief Build spatial grid over receptor (fixed molecule) atoms.
 *
 * Cell size = 8.0 A = sqrt(r2_cutoff=64.0), so any pair within cutoff
 * is guaranteed to be in the same cell or adjacent cells (27-cell neighborhood).
 *
 * Uses a classic 3-pass histogram sort:
 *   1. Count atoms per cell
 *   2. Prefix sum -> cell_start
 *   3. Place atoms into flat array
 *
 * Only non-hydrogen heavy atoms are inserted (H atoms are skipped by eval_ef anyway).
 *
 * orig_index (optional): GridAtom.index is taken from orig_index[i] instead of i, so a cell list
 * over a subset of the receptor keeps the original atom indices. cell_size <= 0 keeps the default
 * 8 A / half; half is the walk half-width stored in the grid.
 */
inline void build_spatial_grid(SpatialGrid& grid,
                               const Real* coords, const int* atom_types,
                               int natom, const AtomTypeProps& atp,
                               const Real* vdw_radii, const Real* charges = nullptr,
                               const int* orig_index = nullptr, Real cell_size = Real(0),
                               int half = UD2_GRID_REFINE) {
    const int HALF = half;
    const Real CELL_SIZE = cell_size > Real(0) ? cell_size : Real(8.0) / HALF;  // sqrt(r2_cutoff) / HALF

    // Find bounding box of receptor atoms
    Real lo[3] = { 1e30,  1e30,  1e30};
    Real hi[3] = {-1e30, -1e30, -1e30};

    int nheavy = 0;
    for (int i = 0; i < natom; i++) {
        if (atp.is_hydrogen[atom_types[i]]) continue;
        nheavy++;
        for (int d = 0; d < 3; d++) {
            Real v = coords[i * 3 + d];
            if (v < lo[d]) lo[d] = v;
            if (v > hi[d]) hi[d] = v;
        }
    }

    if (nheavy == 0) {
        grid.built = false;
        grid.cell_start = nullptr;
        grid.atoms = nullptr;
        grid.total_atoms = 0;
        grid.ncells = 0;
        return;
    }

    // Pad by the walk half-width on each side so boundary atoms have valid neighbours
    grid.origin[0] = lo[0] - HALF * CELL_SIZE;
    grid.origin[1] = lo[1] - HALF * CELL_SIZE;
    grid.origin[2] = lo[2] - HALF * CELL_SIZE;
    grid.inv_cell_size = Real(1.0) / CELL_SIZE;
    grid.half = HALF;

    grid.nx = std::max(1, (int)std::ceil((hi[0] - lo[0]) / CELL_SIZE) + 2 * HALF);
    grid.ny = std::max(1, (int)std::ceil((hi[1] - lo[1]) / CELL_SIZE) + 2 * HALF);
    grid.nz = std::max(1, (int)std::ceil((hi[2] - lo[2]) / CELL_SIZE) + 2 * HALF);
    grid.ncells = grid.nx * grid.ny * grid.nz;
    grid.total_atoms = nheavy;

    // Pass 1: count atoms per cell
    grid.cell_start = new int[grid.ncells + 1]();  // zero-initialized

    for (int i = 0; i < natom; i++) {
        if (atp.is_hydrogen[atom_types[i]]) continue;
        int cx = (int)((coords[i * 3]     - grid.origin[0]) * grid.inv_cell_size);
        int cy = (int)((coords[i * 3 + 1] - grid.origin[1]) * grid.inv_cell_size);
        int cz = (int)((coords[i * 3 + 2] - grid.origin[2]) * grid.inv_cell_size);
        cx = std::clamp(cx, 0, grid.nx - 1);
        cy = std::clamp(cy, 0, grid.ny - 1);
        cz = std::clamp(cz, 0, grid.nz - 1);
        int cell = cx * grid.ny * grid.nz + cy * grid.nz + cz;
        grid.cell_start[cell + 1]++;
    }

    // Pass 2: prefix sum
    for (int c = 0; c < grid.ncells; c++) {
        grid.cell_start[c + 1] += grid.cell_start[c];
    }

    // Pass 3: place atoms (use a write-cursor array)
    grid.atoms = new GridAtom[nheavy];
    int* cursor = new int[grid.ncells];
    std::memcpy(cursor, grid.cell_start, grid.ncells * sizeof(int));

    for (int i = 0; i < natom; i++) {
        if (atp.is_hydrogen[atom_types[i]]) continue;
        int cx = (int)((coords[i * 3]     - grid.origin[0]) * grid.inv_cell_size);
        int cy = (int)((coords[i * 3 + 1] - grid.origin[1]) * grid.inv_cell_size);
        int cz = (int)((coords[i * 3 + 2] - grid.origin[2]) * grid.inv_cell_size);
        cx = std::clamp(cx, 0, grid.nx - 1);
        cy = std::clamp(cy, 0, grid.ny - 1);
        cz = std::clamp(cz, 0, grid.nz - 1);
        int cell = cx * grid.ny * grid.nz + cy * grid.nz + cz;

        int pos = cursor[cell]++;
        grid.atoms[pos].x      = coords[i * 3];
        grid.atoms[pos].y      = coords[i * 3 + 1];
        grid.atoms[pos].z      = coords[i * 3 + 2];
        grid.atoms[pos].index  = orig_index ? orig_index[i] : i;
        grid.atoms[pos].type   = atom_types[i];
        grid.atoms[pos].radius = vdw_radii[atom_types[i]];
        grid.atoms[pos].q      = charges ? charges[i] : Real(0);
        grid.atoms[pos].pad    = 0;
    }

    delete[] cursor;
    grid.built = true;
}

/**
 * @brief Free spatial grid memory.
 */
inline void free_spatial_grid(SpatialGrid& grid) {
    delete[] grid.cell_start;
    delete[] grid.atoms;
    grid.cell_start = nullptr;
    grid.atoms = nullptr;
    grid.total_atoms = 0;
    grid.ncells = 0;
    grid.built = false;
}

#endif // SPATIAL_GRID_H
