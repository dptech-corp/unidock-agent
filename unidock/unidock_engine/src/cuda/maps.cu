#include <cstdlib>
//
// Receptor maps for the CUDA GRID path: host build, upload and release (see maps.cuh).
//
#include "maps.cuh"
#include "bfgs.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <spdlog/spdlog.h>

__constant__ RecMaps g_maps = {};
__constant__ SpatialGrid g_pgrid = {};
__constant__ PolarCands g_pcand = {};


// One thread per (point, slot); blockIdx.y is the slot, so the ligand type is block-uniform.
// The thread walks the (2 * half + 1)^3 cells of g_grid around its point with the bounds logic of
// cal_e_f_warp (bfgs.cuh:599-610): the cell index of the point is clamped into the cell list, so a
// point beyond the list's padding tests the boundary cells, which hold nothing within the cutoff.
// Over the receptor heavy atoms with r2 < Score.r2_cutoff it sums curl(eval_ef_nohb) at both caps.
__global__ void build_maps_kernel(RecMaps maps, const int* __restrict__ slot_type){
    // Per-type tables are read with thread-divergent receptor types; stage them in shared memory
    // (same reason as cal_e_f_warp, bfgs.cuh:451-462).
    __shared__ AtomTypeProps s_atp;
    __shared__ Real s_vdw[MAX_VN_TYPES];
    {
        const int* src = reinterpret_cast<const int*>(&g_atp);
        int* dst = reinterpret_cast<int*>(&s_atp);
        for (int i = threadIdx.x; i < (int)(sizeof(AtomTypeProps) / sizeof(int)); i += blockDim.x) dst[i] = src[i];
        for (int i = threadIdx.x; i < MAX_VN_TYPES; i += blockDim.x) s_vdw[i] = g_vdw_radii[i];
    }
    __syncthreads();

    const long long npts = (long long)maps.nx * maps.ny * maps.nz;
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npts) return;
    const int slot = blockIdx.y;
    const int t = slot_type[slot];
    const int ny_nz = maps.ny * maps.nz;
    const int ix = (int)(p / ny_nz);
    const int rem = (int)(p - (long long)ix * ny_nz);
    const int iy = rem / maps.nz;
    const int iz = rem - iy * maps.nz;
    const Real x = maps.origin[0] + ix * maps.h;
    const Real y = maps.origin[1] + iy * maps.h;
    const Real z = maps.origin[2] + iz * maps.h;
    const Real r1 = s_vdw[t];
    const Real r2_cutoff = Score.r2_cutoff;

    int cx = (int)((x - g_grid.origin[0]) * g_grid.inv_cell_size);
    int cy = (int)((y - g_grid.origin[1]) * g_grid.inv_cell_size);
    int cz = (int)((z - g_grid.origin[2]) * g_grid.inv_cell_size);
    cx = min(max(cx, 0), g_grid.nx - 1); cy = min(max(cy, 0), g_grid.ny - 1); cz = min(max(cz, 0), g_grid.nz - 1);
    const int sx_lo = max(0, cx - g_grid.half), sx_hi = min(g_grid.nx - 1, cx + g_grid.half);
    const int sy_lo = max(0, cy - g_grid.half), sy_hi = min(g_grid.ny - 1, cy + g_grid.half);
    const int zlo = max(0, cz - g_grid.half), zspan = min(g_grid.nz - 1, cz + g_grid.half) - zlo + 1;
    const int g_ny_nz = g_grid.ny * g_grid.nz;
    const int* __restrict__ cs = g_grid.cell_start;

    Real e_hunt = 0, e_auth = 0;
    for (int sx = sx_lo; sx <= sx_hi; sx++){
        for (int sy = sy_lo; sy <= sy_hi; sy++){
            const int c = sx * g_ny_nz + sy * g_grid.nz + zlo;
            const int k_end = __ldg(cs + c + zspan);
            for (int k = __ldg(cs + c); k < k_end; k++){
                const GridAtom rec = g_grid.atoms[k];
                const Real dx = rec.x - x, dy = rec.y - y, dz = rec.z - z;
                const Real r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= r2_cutoff) continue;
                const Real d = sqrtf(r2) - (r1 + rec.radius);
                Real f = 0;
                const Real e = Score.eval_ef_nohb(d, t, rec.type, &f, s_atp);
                Real e0 = e, f0 = f;
                curl_warp(e0, f0, CURL_V_CU);
                e_hunt += e0;
                Real e1 = e, f1 = f;
                curl_warp(e1, f1, CURL_V_AUTHENTIC_CU);
                e_auth += e1;
            }
        }
    }
    const long long base = ((long long)slot * maps.nx + ix) * ny_nz + (long long)iy * maps.nz + iz;
    maps.data[base] = e_hunt;                              // cap set 0: CURL_V_CU
    maps.data[(long long)maps.nslot * npts + base] = e_auth;  // cap set 1: CURL_V_AUTHENTIC_CU
}


// Polar-pass candidate lists over the polar cell list `host` (the g_pgrid build: atoms sorted by cell).
//
// Ordering argument. The g_pgrid walk of cal_e_f_warp<true> (bfgs.cuh, "walk over g_pgrid") visits, for one
// ligand atom, the cells wx in [wx_lo, wx_hi] (outer), wy in [wy_lo, wy_hi] and for each (wx, wy) the contiguous
// CSR range cell_start[c] .. cell_start[c + zspan) of its z-span, c = wx * ny * nz + wy * nz + zlo. Consecutive
// spans start at least nz cells apart and zspan <= nz, so the position k in host.atoms increases monotonically
// over the whole walk: the walk visits atoms by g_pgrid cell index (x-major, then y, then z), then by position
// within the cell, i.e. in increasing k. A pair contributes only when xs_h_bond_possible(t1, t2) and
// d = r - (R1 + R2) < hbond_high, which needs r < r_cut (the largest such R1 + R2 plus hbond_high). Every such
// atom lies (a) inside the walk's cells, because the walk reaches at least half * MAPS_POLAR_CELL_SIZE from the
// ligand atom on every side and r_cut is required to be smaller than that reach, and (b) within
// r_cut + half cell diagonal of the centre of the lane's candidate cell: the lane's atom is inside that cell,
// or, when its cell index was clamped, every polar atom lies on the far side of the cell's face, so the clamped
// point of the cell is nearer to the atom than the ligand atom is. The candidate list of a cell (all atoms within
// that radius of its centre, plus MAPS_PCAND_SLACK for float rounding) is therefore a superset of the
// contributing atoms of any lane in that cell. Filled in increasing k, it yields the same contributing atoms in
// the same order as the walk, so each lane runs the same per-pair code on the same sequence of pairs and
// accumulates the same floating-point sequence.
// The argument is per lane. The one cross-lane accumulation of the polar pass is the cone gradient of a free-rotor
// donor, added to its antecedent's s_force slot from inside the loop (bfgs.cuh hbond_angular_pair_warp mode 2).
// When two such donors share an antecedent (gem-diol C(OH)2; BondedHTopo::n_shared_cone_antecedent > 0) the two
// lanes' atomics on that slot interleave by loop iteration, which the lists change (walk visit index -> list
// index), so those ligands keep the walk (cal_e_f_warp cand_pass).
static void upload_polar_cands(const SpatialGrid& host, const AtomTypeProps& atp, const Real* radii){
    const auto t0 = std::chrono::steady_clock::now();
    // UD2_PCAND=0 keeps the g_pgrid walk (A/B switch for the candidate lists)
    if (const char* e = std::getenv("UD2_PCAND"); e != nullptr && e[0] == '0'){
        spdlog::info("GRID polar candidates disabled by UD2_PCAND=0; the polar pass walks g_pgrid");
        return;
    }
    Real r_cut = 0;
    for (int t1 = 0; t1 < MAX_VN_TYPES; t1++){
        if (atp.is_hydrogen[t1]) continue;
        for (int t2 = 0; t2 < MAX_VN_TYPES; t2++){
            if (atp.is_hydrogen[t2] || !xs_h_bond_possible(t1, t2, atp)) continue;
            r_cut = std::max(r_cut, radii[t1] + radii[t2]);
        }
    }
    r_cut += Vina::hbond_high_val();
    const Real reach = Real(host.half) * MAPS_POLAR_CELL_SIZE;
    if (r_cut <= Real(0) || r_cut + MAPS_PCAND_SLACK > reach){
        spdlog::info("GRID polar candidates not built (r_cut {:.3f} A vs walk reach {:.1f} A); the polar pass walks g_pgrid", r_cut, reach);
        return;
    }

    const Real cell = MAPS_PCAND_CELL_SIZE;
    Real lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
    for (int k = 0; k < host.total_atoms; k++){
        const Real p[3] = {host.atoms[k].x, host.atoms[k].y, host.atoms[k].z};
        for (int d = 0; d < 3; d++){ lo[d] = std::min(lo[d], p[d]); hi[d] = std::max(hi[d], p[d]); }
    }
    PolarCands pc = {};
    for (int d = 0; d < 3; d++) pc.origin[d] = lo[d] - cell;
    pc.inv_cell = Real(1) / cell;
    pc.nx = std::max(1, (int)std::ceil((hi[0] - lo[0]) / cell) + 2);
    pc.ny = std::max(1, (int)std::ceil((hi[1] - lo[1]) / cell) + 2);
    pc.nz = std::max(1, (int)std::ceil((hi[2] - lo[2]) / cell) + 2);
    pc.r_cut = r_cut;
    const int ncells = pc.nx * pc.ny * pc.nz;
    const int n[3] = {pc.nx, pc.ny, pc.nz};
    const Real R = r_cut + Real(0.5) * cell * std::sqrt(Real(3)) + MAPS_PCAND_SLACK;
    const Real R2 = R * R;
    // cells whose centre lies within R of atom a (index range from the bounding cube, widened by one cell)
    auto for_each_cell = [&](const GridAtom& a, auto&& f){
        const Real p[3] = {a.x, a.y, a.z};
        int c_lo[3], c_hi[3];
        for (int d = 0; d < 3; d++){
            c_lo[d] = std::clamp((int)std::floor((p[d] - R - pc.origin[d]) * pc.inv_cell) - 1, 0, n[d] - 1);
            c_hi[d] = std::clamp((int)std::floor((p[d] + R - pc.origin[d]) * pc.inv_cell) + 1, 0, n[d] - 1);
        }
        for (int cx = c_lo[0]; cx <= c_hi[0]; cx++){
            const Real dx = pc.origin[0] + (Real(cx) + Real(0.5)) * cell - p[0];
            for (int cy = c_lo[1]; cy <= c_hi[1]; cy++){
                const Real dy = pc.origin[1] + (Real(cy) + Real(0.5)) * cell - p[1];
                for (int cz = c_lo[2]; cz <= c_hi[2]; cz++){
                    const Real dz = pc.origin[2] + (Real(cz) + Real(0.5)) * cell - p[2];
                    if (dx * dx + dy * dy + dz * dz > R2) continue;
                    f((cx * pc.ny + cy) * pc.nz + cz);
                }
            }
        }
    };
    std::vector<int> start(ncells + 1, 0);
    for (int k = 0; k < host.total_atoms; k++) for_each_cell(host.atoms[k], [&](int c){ start[c + 1]++; });
    for (int c = 0; c < ncells; c++) start[c + 1] += start[c];
    const int total = start[ncells];
    std::vector<GridAtom> atoms(std::max(total, 1));
    std::vector<int> cursor(start.begin(), start.end() - 1);
    for (int k = 0; k < host.total_atoms; k++) for_each_cell(host.atoms[k], [&](int c){ atoms[cursor[c]++] = host.atoms[k]; });
    int per_cell_max = 0;
    for (int c = 0; c < ncells; c++) per_cell_max = std::max(per_cell_max, start[c + 1] - start[c]);

    checkCUDA(cudaMalloc(&pc.cell_start, (ncells + 1) * sizeof(int)));
    checkCUDA(cudaMemcpy(pc.cell_start, start.data(), (ncells + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCUDA(cudaMalloc(&pc.atoms, atoms.size() * sizeof(GridAtom)));
    checkCUDA(cudaMemcpy(pc.atoms, atoms.data(), atoms.size() * sizeof(GridAtom), cudaMemcpyHostToDevice));
    pc.built = true;
    checkCUDA(cudaMemcpyToSymbol(g_pcand, &pc, sizeof(PolarCands)));
    const double bytes = (ncells + 1) * (double)sizeof(int) + atoms.size() * (double)sizeof(GridAtom);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    spdlog::info("GRID polar candidates: r_cut {:.2f} A, {:.1f} A cells {}x{}x{} over {} polar atoms, {} candidates ({:.2f} mean / {} max per cell), {:.2f} MB, build {:.1f} ms",
                 r_cut, cell, pc.nx, pc.ny, pc.nz, host.total_atoms, total, (double)total / ncells, per_cell_max,
                 bytes / (1024.0 * 1024.0), ms);
}


// Polar-only receptor cell list: donor or acceptor heavy atoms, 4 A cells, half 1, original indices.
// Leaves g_pgrid empty (built == false) when the receptor has no polar heavy atom.
static void upload_polar_grid(const UDFixMol& fix, const AtomTypeProps& atp, const Real* radii){
    std::vector<Real> coords;
    std::vector<int> types, orig;
    for (int i = 0; i < fix.natom; i++){
        const int t = fix.vina_types[i];
        if (atp.is_hydrogen[t]) continue;
        if (!(atp.is_donor[t] || atp.is_acceptor[t])) continue;
        coords.push_back(fix.coords[i * 3]);
        coords.push_back(fix.coords[i * 3 + 1]);
        coords.push_back(fix.coords[i * 3 + 2]);
        types.push_back(t);
        orig.push_back(i);
    }
    SpatialGrid host = {};
    build_spatial_grid(host, coords.data(), types.data(), (int)types.size(), atp, radii, nullptr,
                       orig.data(), MAPS_POLAR_CELL_SIZE, 1);
    if (!host.built) return;

    SpatialGrid dev = host;  // dims / origin / counts; pointer fields replaced below
    checkCUDA(cudaMalloc(&dev.cell_start, (host.ncells + 1) * sizeof(int)));
    checkCUDA(cudaMemcpy(dev.cell_start, host.cell_start, (host.ncells + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCUDA(cudaMalloc(&dev.atoms, std::max(host.total_atoms, 1) * sizeof(GridAtom)));
    checkCUDA(cudaMemcpy(dev.atoms, host.atoms, host.total_atoms * sizeof(GridAtom), cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(g_pgrid, &dev, sizeof(SpatialGrid)));
    upload_polar_cands(host, atp, radii);
    free_spatial_grid(host);
}


bool build_receptor_maps(const UDFixMol& fix, const AtomTypeProps& atp, const std::vector<Real>& vdw_radii,
                         const DockParam& dock_param, const std::vector<int>& ligand_types_present,
                         int max_heavy_atoms, std::string& summary){
    free_receptor_maps();
    if (!dock_param.use_grid){
        summary = "use_grid is false";
        return false;
    }
    if (dock_param.weights.elec != Real(0)){
        summary = spdlog::fmt_lib::format("weight_elec = {} (the maps carry no Coulomb term)", dock_param.weights.elec);
        return false;
    }
    (void)max_heavy_atoms;  // no ligand-size gate: the maps depend on the box and the types, not on the ligand
    if (dock_param.grid_spacing <= Real(0) || dock_param.grid_margin < Real(0)){
        summary = spdlog::fmt_lib::format("grid_spacing {} / grid_margin {} out of range", dock_param.grid_spacing, dock_param.grid_margin);
        return false;
    }
    SpatialGrid grid_dev = {};
    checkCUDA(cudaMemcpyFromSymbol(&grid_dev, g_grid, sizeof(SpatialGrid)));
    if (!grid_dev.built){
        summary = "receptor cell list not built";
        return false;
    }

    RecMaps maps = {};
    for (int t = 0; t < MAX_VN_TYPES; t++) maps.slot_of_type[t] = -1;
    for (int t : ligand_types_present){
        if (t < 0 || t >= MAX_VN_TYPES || atp.is_hydrogen[t] || maps.slot_of_type[t] >= 0) continue;
        maps.slot_of_type[t] = maps.nslot++;
    }
    if (maps.nslot == 0){
        summary = "no ligand heavy atom type in the batch";
        return false;
    }
    const Real h = dock_param.grid_spacing, m = dock_param.grid_margin;
    const Box& box = dock_param.box;
    maps.origin[0] = box.x_lo - m;
    maps.origin[1] = box.y_lo - m;
    maps.origin[2] = box.z_lo - m;
    maps.h = h;
    maps.inv_h = Real(1) / h;
    maps.nx = (int)std::floor((box.x_hi + m - maps.origin[0]) / h) + 2;
    maps.ny = (int)std::floor((box.y_hi + m - maps.origin[1]) / h) + 2;
    maps.nz = (int)std::floor((box.z_hi + m - maps.origin[2]) / h) + 2;
    const long long npts = (long long)maps.nx * maps.ny * maps.nz;
    if (npts > MAPS_MAX_POINTS){
        summary = spdlog::fmt_lib::format("{}x{}x{} = {} map points > {}", maps.nx, maps.ny, maps.nz, npts, MAPS_MAX_POINTS);
        return false;
    }
    const long long bytes = 2LL * maps.nslot * npts * (long long)sizeof(float);
    if (bytes > MAPS_MAX_BYTES){
        summary = spdlog::fmt_lib::format("{} types x {} points = {:.1f} MB > {:.0f} MB", maps.nslot, npts,
                              bytes / (1024.0 * 1024.0), MAPS_MAX_BYTES / (1024.0 * 1024.0));
        return false;
    }

    std::vector<int> slot_type(maps.nslot);
    for (int t = 0; t < MAX_VN_TYPES; t++){
        if (maps.slot_of_type[t] >= 0) slot_type[maps.slot_of_type[t]] = t;
    }
    const auto t0 = std::chrono::steady_clock::now();
    checkCUDA(cudaMalloc(&maps.data, bytes));
    int* slot_type_cu = nullptr;
    checkCUDA(cudaMalloc(&slot_type_cu, maps.nslot * sizeof(int)));
    checkCUDA(cudaMemcpy(slot_type_cu, slot_type.data(), maps.nslot * sizeof(int), cudaMemcpyHostToDevice));

    const int block_size = 256;
    const dim3 nblock((unsigned)((npts + block_size - 1) / block_size), (unsigned)maps.nslot);
    set_carveout((const void*)build_maps_kernel, "build_maps_kernel", "UD2_CARVE_MAPS", "occ32",
                 (int)(nblock.x * nblock.y), block_size, 0);
    build_maps_kernel<<<nblock, block_size>>>(maps, slot_type_cu);
    checkCUDA(cudaGetLastError());
    checkCUDA(cudaDeviceSynchronize());
    checkCUDA(cudaFree(slot_type_cu));
    maps.built = true;
    checkCUDA(cudaMemcpyToSymbol(g_maps, &maps, sizeof(RecMaps)));

    Real radii[MAX_VN_TYPES];
    fill_vdw_radii(radii, vdw_radii);
    upload_polar_grid(fix, atp, radii);
    checkCUDA(cudaDeviceSynchronize());
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    summary = spdlog::fmt_lib::format("{} types, {}x{}x{}, {:.1f} MB, build {:.1f} ms", maps.nslot, maps.nx, maps.ny, maps.nz,
                          bytes / (1024.0 * 1024.0), ms);
    return true;
}


void free_receptor_maps(){
    RecMaps maps = {};
    checkCUDA(cudaMemcpyFromSymbol(&maps, g_maps, sizeof(RecMaps)));
    if (maps.built){
        checkCUDA(cudaFree(maps.data));
        RecMaps empty = {};
        for (int t = 0; t < MAX_VN_TYPES; t++) empty.slot_of_type[t] = -1;
        checkCUDA(cudaMemcpyToSymbol(g_maps, &empty, sizeof(RecMaps)));
    }
    PolarCands pc = {};
    checkCUDA(cudaMemcpyFromSymbol(&pc, g_pcand, sizeof(PolarCands)));
    if (pc.cell_start){   // uploaded (built may have been cleared by set_polar_cands_enabled in a test)
        checkCUDA(cudaFree(pc.cell_start));
        checkCUDA(cudaFree(pc.atoms));
        PolarCands empty = {};
        checkCUDA(cudaMemcpyToSymbol(g_pcand, &empty, sizeof(PolarCands)));
    }
    SpatialGrid pgrid = {};
    checkCUDA(cudaMemcpyFromSymbol(&pgrid, g_pgrid, sizeof(SpatialGrid)));
    if (!pgrid.built) return;
    checkCUDA(cudaFree(pgrid.cell_start));
    checkCUDA(cudaFree(pgrid.atoms));
    SpatialGrid empty = {};
    checkCUDA(cudaMemcpyToSymbol(g_pgrid, &empty, sizeof(SpatialGrid)));
}
