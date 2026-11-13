//
// Receptor maps for the CUDA GRID path (see output/gpu_grid_design_1008/spec.md).
//
// One map per ligand heavy type present in the batch, two cap sets (CURL_V_CU and
// CURL_V_AUTHENTIC_CU), holding the per-pair-curled sum of eval_ef_nohb over the receptor
// heavy atoms within Score.r2_cutoff. The hydrogen-bond ramp is not in the maps: the kernel
// adds it exactly in a polar pass over g_pgrid, the donor/acceptor-only receptor cell list.
//

#ifndef MAPS_CUH
#define MAPS_CUH

#include "common.cuh"
#include <string>
#include <vector>

struct RecMaps {
    Real  origin[3];      // map origin (A): box low corner minus grid_margin
    Real  h, inv_h;       // spacing and its inverse
    int   nx, ny, nz;     // points per axis (>= 2 each); the last point lies beyond box high + grid_margin
    int   nslot;          // number of ligand types present in the batch
    int   slot_of_type[MAX_VN_TYPES]; // -1 when the type has no map
    float* data;          // device, size 2 * nslot * nx * ny * nz,
                          // index = ((cap * nslot + slot) * nx + ix) * ny * nz + iy * nz + iz
    bool  built;
};

extern __constant__ RecMaps g_maps;          // maps (built == false -> the GRID instantiation must not be launched)
extern __constant__ SpatialGrid g_pgrid;     // polar-only receptor cell list (donor or acceptor heavy atoms),
                                             // cell 4.0 A, half 1, GridAtom.index = original receptor atom index

// Polar-pass candidate lists: per MAPS_PCAND_CELL_SIZE cell, the g_pgrid atoms that can pass the polar-pass
// distance test (d < hbond_high) from anywhere in the cell, as inline GridAtom copies in CSR layout, in the
// order the g_pgrid walk visits them (ordering argument at upload_polar_cands in maps.cu).
struct PolarCands {
    Real  origin[3];    // cell grid origin (A): polar bounding box low corner minus one cell
    Real  inv_cell;     // 1 / MAPS_PCAND_CELL_SIZE
    int   nx, ny, nz;   // cells per axis
    Real  r_cut;        // largest R_lig + R_rec over the hbond-possible type pairs plus hbond_high (A)
    int*  cell_start;   // device, nx * ny * nz + 1
    GridAtom* atoms;    // device, candidates of cell c at [cell_start[c], cell_start[c + 1])
    bool  built;        // false -> cal_e_f_warp<true> walks g_pgrid
};
extern __constant__ PolarCands g_pcand;

// Eligibility limits of the GRID path (spec invariant 5)
#ifndef MAPS_MAX_POINTS
#define MAPS_MAX_POINTS 8000000LL
#endif
#ifndef MAPS_MAX_BYTES
#define MAPS_MAX_BYTES (2LL * 1024 * 1024 * 1024)
#endif
#define MAPS_POLAR_CELL_SIZE Real(4.0)
#define MAPS_PCAND_CELL_SIZE Real(2.0)
#define MAPS_PCAND_SLACK Real(0.05)   // A added to the candidate radius: float rounding of cell indices and distances

/**
 * @brief Build and upload the receptor maps (g_maps) and the polar-only cell list (g_pgrid).
 * Requires init_constants and upload_receptor_grid to have run (the build kernel walks g_grid and
 * reads g_vdw_radii / g_atp / Score). Applies the eligibility rules (use_grid, weight_elec == 0,
 * max heavy atoms, point and byte limits).
 * @param ligand_types_present Vina types of the ligand heavy atoms in the batch (any order, duplicates allowed)
 * @param summary On success "{nslot} types, {nx}x{ny}x{nz}, {MB} MB, build {ms} ms"; otherwise the reason
 * @return true when the maps were built and uploaded
 */
bool build_receptor_maps(const UDFixMol& fix, const AtomTypeProps& atp, const std::vector<Real>& vdw_radii,
                         const DockParam& dock_param, const std::vector<int>& ligand_types_present,
                         int max_heavy_atoms, std::string& summary);
void free_receptor_maps();

#endif // MAPS_CUH
