//
// Created by Congcong Liu on 24-12-12.
//
#include "common.cuh"
#include <vector>
#include <algorithm>
#include <spdlog/spdlog.h>
#include <string>
#include <cstdlib>

__constant__ bool FLAG_CONSTRAINT_DOCK = false;
__constant__ Real BOX_X_HI = 30;
__constant__ Real BOX_X_LO = -30;
__constant__ Real BOX_Y_HI = 30;
__constant__ Real BOX_Y_LO = -30;
__constant__ Real BOX_Z_HI = 30;
__constant__ Real BOX_Z_LO = -30;
__constant__ Real TOR_PREC = 0.3;
__constant__ Real BOX_PREC = 1.0;
#ifndef UD2_PENALTY_SLOPE
#define UD2_PENALTY_SLOPE 10.0
#endif
__constant__ Real PENALTY_SLOPE = UD2_PENALTY_SLOPE;  // 10 matches init_constants_cpu; upstream uses 1e6 (-DUD2_PENALTY_SLOPE=1e6 to test)
__constant__ bool USE_INNER_BOX = false;
__constant__ Real INNER_X_LO = 0;
__constant__ Real INNER_X_HI = 0;
__constant__ Real INNER_Y_LO = 0;
__constant__ Real INNER_Y_HI = 0;
__constant__ Real INNER_Z_LO = 0;
__constant__ Real INNER_Z_HI = 0;
__constant__ TermSettings g_term_settings;   // every term off (the struct's defaults); always overwritten by init_constants


#if true
__constant__ Vina Score;
#else
__constant__ Gaff2 Score;
#endif

__constant__ AtomTypeProps g_atp;
__constant__ Real g_vdw_radii[MAX_VN_TYPES];
__constant__ bool USE_ANGULAR_HBOND = true;
__constant__ SpatialGrid g_grid = {};


// Per-type vdW radii: JSON/DMS-provided if available, else the built-in table (same rule as CPU)
void fill_vdw_radii(Real* out, const std::vector<Real>& vdw_radii){
    for (int i = 0; i < MAX_VN_TYPES; i++) out[i] = Real(0);
    if (!vdw_radii.empty()){
        int n = std::min(static_cast<int>(vdw_radii.size()), MAX_VN_TYPES);
        for (int i = 0; i < n; i++) out[i] = vdw_radii[i];
    } else {
        int n = sizeof(VN_VDW_RADII) / sizeof(VN_VDW_RADII[0]);
        for (int i = 0; i < n; i++) out[i] = VN_VDW_RADII[i];
    }
}


void init_constants(const DockParam& dock_param, const AtomTypeProps& type_props,
                    const std::vector<Real>& vdw_radii){
    //======================= constants ======================
    checkCUDA(cudaMemcpyToSymbol(FLAG_CONSTRAINT_DOCK, &dock_param.policy.static_root, sizeof(bool)));
    checkCUDA(cudaMemcpyToSymbol(BOX_X_HI, &dock_param.box.x_hi, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(BOX_X_LO, &dock_param.box.x_lo, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(BOX_Y_HI, &dock_param.box.y_hi, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(BOX_Y_LO, &dock_param.box.y_lo, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(BOX_Z_HI, &dock_param.box.z_hi, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(BOX_Z_LO, &dock_param.box.z_lo, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(USE_INNER_BOX, &dock_param.use_inner_box, sizeof(bool)));
    checkCUDA(cudaMemcpyToSymbol(INNER_X_LO, &dock_param.inner_box.x_lo, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(INNER_X_HI, &dock_param.inner_box.x_hi, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(INNER_Y_LO, &dock_param.inner_box.y_lo, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(INNER_Y_HI, &dock_param.inner_box.y_hi, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(INNER_Z_LO, &dock_param.inner_box.z_lo, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(INNER_Z_HI, &dock_param.inner_box.z_hi, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(TOR_PREC, &dock_param.tor_prec, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(BOX_PREC, &dock_param.box_prec, sizeof(Real), 0, cudaMemcpyHostToDevice));
    // optimize_cu raises the slope to 100 for refinement; reset it so every task searches with the compiled value
    const Real penalty_slope = UD2_PENALTY_SLOPE;
    checkCUDA(cudaMemcpyToSymbol(PENALTY_SLOPE, &penalty_slope, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(USE_ANGULAR_HBOND, &dock_param.angular_hbond, sizeof(bool)));
    // Device-side mirror of the term settings (every ligand's FlexParamVina::terms.settings is filled from the same
    // DockParam in task.cu); one struct in constant memory so the kernel's term guard is a constant operand.
    const TermSettings term_settings = term_settings_of(dock_param);
    checkCUDA(cudaMemcpyToSymbol(g_term_settings, &term_settings, sizeof(TermSettings), 0, cudaMemcpyHostToDevice));
    {
        Vina host_score;
        host_score.set_weights(dock_param.weights);
        checkCUDA(cudaMemcpyToSymbol(Score, &host_score, sizeof(Vina)));
    }
    // Copy atom type properties to constant memory
    checkCUDA(cudaMemcpyToSymbol(g_atp, &type_props, sizeof(AtomTypeProps)));

    Real radii[MAX_VN_TYPES];
    fill_vdw_radii(radii, vdw_radii);
    checkCUDA(cudaMemcpyToSymbol(g_vdw_radii, radii, sizeof(radii)));
    checkCUDA(cudaDeviceSynchronize());// assure that memcpy is finished
}


void upload_receptor_grid(const Real* coords, const int* atom_types, int natom,
                          const AtomTypeProps& type_props, const std::vector<Real>& vdw_radii,
                          const Real* charges){
    free_receptor_grid();
    Real radii[MAX_VN_TYPES];
    fill_vdw_radii(radii, vdw_radii);

    SpatialGrid host = {};
    build_spatial_grid(host, coords, atom_types, natom, type_props, radii, charges);
    if (!host.built){
        spdlog::warn("Receptor grid not built ({} atoms); GPU falls back to brute-force receptor loop", natom);
        return;
    }

    SpatialGrid dev = host;  // dims / origin / counts; pointer fields replaced below
    checkCUDA(cudaMalloc(&dev.cell_start, (host.ncells + 1) * sizeof(int)));
    checkCUDA(cudaMemcpy(dev.cell_start, host.cell_start, (host.ncells + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCUDA(cudaMalloc(&dev.atoms, std::max(host.total_atoms, 1) * sizeof(GridAtom)));
    checkCUDA(cudaMemcpy(dev.atoms, host.atoms, host.total_atoms * sizeof(GridAtom), cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(g_grid, &dev, sizeof(SpatialGrid)));
    checkCUDA(cudaDeviceSynchronize());
    spdlog::info("Receptor grid uploaded to GPU: {}x{}x{} cells, {} heavy atoms",
                 host.nx, host.ny, host.nz, host.total_atoms);
    free_spatial_grid(host);
}


void reorder_pairs_round_robin(const int* pairs, int npair, int natom, std::vector<int>& out){
    std::vector<std::vector<int>> buckets(std::max(natom, 1));
    for (int p = 0; p < npair; p++) buckets[pairs[p * 2]].push_back(p);
    out.clear(); out.reserve(npair * 2);
    std::vector<size_t> cursor(buckets.size(), 0);
    int emitted = 0;
    while (emitted < npair){
        for (size_t b = 0; b < buckets.size(); b++){
            if (cursor[b] < buckets[b].size()){
                int p = buckets[b][cursor[b]++];
                out.push_back(pairs[p * 2]); out.push_back(pairs[p * 2 + 1]);
                emitted++;
            }
        }
    }
}


static int s_kernel_max_natom = 0;
static int s_kernel_max_natom_shared = 0;   // largest natom <= KERNEL_SHARED_FORCE_MAX_ATOMS (shared force accumulator)
static int s_kernel_max_ntorsion = 0;
void set_kernel_max_natom(int natom){
    s_kernel_max_natom = std::max(s_kernel_max_natom, natom);
    if (natom <= KERNEL_SHARED_FORCE_MAX_ATOMS) s_kernel_max_natom_shared = std::max(s_kernel_max_natom_shared, natom);
}
void set_kernel_max_ntorsion(int ntorsion){ s_kernel_max_ntorsion = std::max(s_kernel_max_ntorsion, ntorsion); }
size_t kernel_dyn_shared_bytes(){ return (size_t)std::min(std::max(s_kernel_max_natom, 1), KERNEL_SHARED_FORCE_MAX_ATOMS) * 3 * sizeof(Real); }

size_t kernel_state_dyn_shared_bytes(int state_flags){
    const size_t dim_g = dof_g + s_kernel_max_ntorsion;
    const size_t st = ((state_flags & BFGS_STATE_V_SHARED) ? STRIDE_G * dim_g : 0)
                    + ((state_flags & BFGS_STATE_H_SHARED) ? dim_g * (dim_g + 1) / 2 : 0);
    const size_t n = std::max({kernel_dyn_shared_bytes() / sizeof(Real), (size_t)s_kernel_max_natom_shared * 3 + st, st});
    return n * sizeof(Real);
}

namespace{
// unified L1/shared configurations of sm_86 / sm_89, KB
const int CARVE_KB[] = {0, 8, 16, 32, 64, 100};

// smallest configuration holding n blocks of s bytes of shared memory plus the per-block reserve
int carve_kb_holding(int n, int s, int reserve){
    for (int kb : CARVE_KB){
        if ((long)n * (s + reserve) <= kb * 1024L) return kb;
    }
    return 100;
}
}

void set_carveout(const void* func, const char* label, const char* env, const char* fallback, int nblock, int block_size,
                  size_t dyn){
    const char* e = std::getenv(env);
    const std::string mode = e ? e : fallback;
    if (mode == "default") return;
    if (mode.rfind("kb", 0) == 0){
        // fixed target configuration (KB) through the driver-580 hint model
        int target = std::atoi(mode.c_str() + 2), reserve = 0, dev = 0;
        checkCUDA(cudaGetDevice(&dev));
        checkCUDA(cudaDeviceGetAttribute(&reserve, cudaDevAttrReservedSharedMemoryPerBlock, dev));
        cudaFuncAttributes fa;
        checkCUDA(cudaFuncGetAttributes(&fa, func));
        int s = (int)((fa.sharedSizeBytes + dyn + 127) / 128 * 128);
        int lo = -1, hi = -1;
        for (int p = 1; p <= 100; p++){
            if (carve_kb_holding(std::max(1, p * 1024 / s), s, reserve) != target) continue;
            if (lo < 0) lo = p;
            hi = p;
        }
        int pct = lo < 0 ? (int)cudaSharedmemCarveoutDefault : (lo + hi) / 2;
        checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributePreferredSharedMemoryCarveout, pct));
        spdlog::info("{} carveout: fixed target {} KB, {} B shared/block, hint {}", label, target, s, pct);
        return;
    }
    if (mode != "occ" && mode != "occw" && mode != "occ32"){
        int pct = std::atoi(mode.c_str());
        checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributePreferredSharedMemoryCarveout, pct));
        spdlog::info("{} carveout: explicit hint {} %", label, pct);
        return;
    }
    if (nblock <= 0) return;    // empty launch: no waves to balance, driver default kept
    int dev = 0, major = 0, minor = 0, nsm = 0, reserve = 0;
    checkCUDA(cudaGetDevice(&dev));
    checkCUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    checkCUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    if (major != 8 || (minor != 6 && minor != 9)){
        spdlog::info("{} carveout: occ rule knows only sm_86 / sm_89 configurations, driver default kept", label);
        return;
    }
    checkCUDA(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    checkCUDA(cudaDeviceGetAttribute(&reserve, cudaDevAttrReservedSharedMemoryPerBlock, dev));
    cudaFuncAttributes fa;
    checkCUDA(cudaFuncGetAttributes(&fa, func));
    int s = (int)((fa.sharedSizeBytes + dyn + 127) / 128 * 128);
    // blocks per SM with shared memory out of the way (registers, threads, block slots)
    checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    int reg_blocks = 0;
    checkCUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&reg_blocks, func, block_size, dyn));
    int needed = std::max(1, std::min((nblock + nsm - 1) / nsm, reg_blocks));
    if ((mode == "occw" || mode == "occ32") && reg_blocks > 0){
        // wave-balanced: as many blocks per SM as keep the register-limited number of waves
        int waves = (nblock + nsm * reg_blocks - 1) / (nsm * reg_blocks);
        needed = std::max(1, (nblock + nsm * waves - 1) / (nsm * waves));
    }
    int target = carve_kb_holding(needed, s, reserve);
    // occ32: never below 32 KB, so that launches of different sizes (and MPS clients) share one configuration
    if (mode == "occ32") target = std::max(target, 32);
    int dflt = carve_kb_holding(reg_blocks, s, reserve);
    // driver 580: a hint of p % asks for floor(p * 1 KB / s) blocks and gets the smallest configuration holding them;
    // take the middle of the hint range that lands on the target
    int lo = -1, hi = -1;
    for (int p = 1; target < dflt && p <= 100; p++){
        if (carve_kb_holding(std::max(1, p * 1024 / s), s, reserve) != target) continue;
        if (lo < 0) lo = p;
        hi = p;
    }
    int pct = lo < 0 ? (int)cudaSharedmemCarveoutDefault : (lo + hi) / 2;
    checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributePreferredSharedMemoryCarveout, pct));
    spdlog::info("{} carveout: {} rule, {} B shared/block, {} blocks on {} SMs, {} blocks/SM by registers, {} needed, "
                 "target {} KB (driver default {} KB), hint {}", label, mode, s, nblock, nsm, reg_blocks, needed, target, dflt, pct);
}


namespace{
// set_carveout's occ32 decision for a launch: configuration (KB), blocks per SM it needs, blocks per SM it holds.
// false when the occ rule does not apply (not sm_86 / sm_89, empty launch).
bool occ32_plan(const void* func, int nblock, int block_size, size_t dyn, int* out_kb, int* out_needed, int* out_resident){
    if (nblock <= 0) return false;
    int dev = 0, major = 0, minor = 0, nsm = 0, reserve = 0;
    checkCUDA(cudaGetDevice(&dev));
    checkCUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    checkCUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    if (major != 8 || (minor != 6 && minor != 9)) return false;
    checkCUDA(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    checkCUDA(cudaDeviceGetAttribute(&reserve, cudaDevAttrReservedSharedMemoryPerBlock, dev));
    cudaFuncAttributes fa;
    checkCUDA(cudaFuncGetAttributes(&fa, func));
    const int s = (int)((fa.sharedSizeBytes + dyn + 127) / 128 * 128);
    // blocks per SM with shared memory out of the way, as set_carveout counts them; the kernel's carveout preference
    // is put back afterwards (set_carveout leaves it alone in its "default" mode)
    checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    int reg_blocks = 0;
    checkCUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&reg_blocks, func, block_size, dyn));
    checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributePreferredSharedMemoryCarveout, fa.preferredShmemCarveout));
    if (reg_blocks <= 0) return false;
    const int waves = (nblock + nsm * reg_blocks - 1) / (nsm * reg_blocks);
    const int needed = std::max(1, (nblock + nsm * waves - 1) / (nsm * waves));
    const int kb = std::max(carve_kb_holding(needed, s, reserve), 32);
    *out_kb = kb;
    *out_needed = needed;
    *out_resident = std::min(reg_blocks, (int)(kb * 1024L / (s + reserve)));
    return true;
}
}

int bfgs_state_flags(const void* func, const char* label, int nblock, int block_size){
    const char* e = std::getenv("UD2_BFGS_STATE");
    const std::string mode = e ? e : "auto";
    int flags = 0;
    if (mode == "v") flags = BFGS_STATE_V_SHARED;
    else if (mode == "h") flags = BFGS_STATE_H_SHARED;
    else if (mode == "vh") flags = BFGS_STATE_V_SHARED | BFGS_STATE_H_SHARED;
    else if (mode == "auto"){
        int kb0 = 0, needed0 = 0, res0 = 0;
        if (occ32_plan(func, nblock, block_size, kernel_state_dyn_shared_bytes(0), &kb0, &needed0, &res0)){
            for (int f : {BFGS_STATE_V_SHARED | BFGS_STATE_H_SHARED, BFGS_STATE_V_SHARED, BFGS_STATE_H_SHARED}){
                int kb = 0, needed = 0, res = 0;
                if (occ32_plan(func, nblock, block_size, kernel_state_dyn_shared_bytes(f), &kb, &needed, &res)
                    && kb == kb0 && res >= needed0){
                    flags = f;
                    break;
                }
            }
        }
    }
    spdlog::info("{} BFGS state: {} rule, vectors in {}, Hessian in {} memory, {} B dynamic shared/block (max {} torsions)",
                 label, mode, (flags & BFGS_STATE_V_SHARED) ? "shared" : "global",
                 (flags & BFGS_STATE_H_SHARED) ? "shared" : "global", kernel_state_dyn_shared_bytes(flags), s_kernel_max_ntorsion);
    return flags;
}


void free_receptor_grid(){
    SpatialGrid dev = {};
    checkCUDA(cudaMemcpyFromSymbol(&dev, g_grid, sizeof(SpatialGrid)));
    if (!dev.built) return;
    checkCUDA(cudaFree(dev.cell_start));
    checkCUDA(cudaFree(dev.atoms));
    SpatialGrid empty = {};
    checkCUDA(cudaMemcpyToSymbol(g_grid, &empty, sizeof(SpatialGrid)));
}


void build_intra_csr(const int* pairs, int npair, int natom,
                     std::vector<int>& out_start, std::vector<int>& out_nbr){
    out_start.assign(natom + 1, 0);
    for (int p = 0; p < npair; p++){
        out_start[pairs[p * 2] + 1]++;
        out_start[pairs[p * 2 + 1] + 1]++;
    }
    for (int i = 0; i < natom; i++){
        out_start[i + 1] += out_start[i];
    }
    out_nbr.assign(std::max(npair * 2, 1), 0);
    std::vector<int> cursor(out_start.begin(), out_start.end() - 1);
    for (int p = 0; p < npair; p++){
        int i = pairs[p * 2], j = pairs[p * 2 + 1];
        out_nbr[cursor[i]++] = j;
        out_nbr[cursor[j]++] = i;
    }
}

#ifdef UD2_PROFILE
__device__ unsigned long long* g_prof_ptr = nullptr;
static unsigned long long* s_prof_dev = nullptr;
static int s_prof_nwarps = 0;

void prof_begin(int nwarps){
    if (s_prof_dev) checkCUDA(cudaFree(s_prof_dev));
    s_prof_nwarps = nwarps;
    checkCUDA(cudaMalloc(&s_prof_dev, (size_t)nwarps * PROF_SLOTS * sizeof(unsigned long long)));
    checkCUDA(cudaMemset(s_prof_dev, 0, (size_t)nwarps * PROF_SLOTS * sizeof(unsigned long long)));
    checkCUDA(cudaMemcpyToSymbol(g_prof_ptr, &s_prof_dev, sizeof(s_prof_dev)));
}

void prof_report(const char* label, int nwarps, double wall_ms){
    std::vector<unsigned long long> h((size_t)nwarps * PROF_SLOTS);
    checkCUDA(cudaMemcpy(h.data(), s_prof_dev, h.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    double sum[PROF_SLOTS] = {0};
    for (int w = 0; w < nwarps; w++) for (int k = 0; k < PROF_SLOTS; k++) sum[k] += (double)h[w * PROF_SLOTS + k];
    const char* names[8] = {"randomize", "mutate", "energy:inter", "energy:penalty+intra+reduce", "gradient", "line-search copy+update", "bfgs algebra/copies", "mc bookkeeping"};
    double timed = 0; for (int k = 0; k < 8; k++) timed += sum[k];
    int dev = 0; cudaGetDevice(&dev); int khz = 0; cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev);
    double cyc_per_ms = khz;  // kHz -> cycles per ms
    spdlog::info("[prof:{}] {} warps, wall {:.1f} ms, sum of per-warp timed cycles {:.3e} ({:.1f} warp-ms), per-warp mean {:.2f} ms",
                 label, nwarps, wall_ms, timed, timed / cyc_per_ms, timed / cyc_per_ms / nwarps);
    for (int k = 0; k < 8; k++){
        spdlog::info("[prof:{}]   {:28s} {:6.1f} %", label, names[k], 100.0 * sum[k] / std::max(timed, 1.0));
    }
    double ne = sum[PROF_N_EVAL], nls = sum[PROF_N_LS_TRIAL], nb = sum[PROF_N_BFGS_STEP], nmc = sum[PROF_N_MC_STEP];
    spdlog::info("[prof:{}] counts: mc steps {:.0f}, bfgs steps {:.0f} ({:.2f}/mc step), line-search trials {:.0f} ({:.2f}/bfgs step), energy evals {:.0f} ({:.1f}/mc step)",
                 label, nmc, nb, nb / std::max(nmc, 1.0), nls, nls / std::max(nb, 1.0), ne, ne / std::max(nmc, 1.0));
    double cl = sum[PROF_LS_CLAMPED_TRIALS], cl_fail = sum[PROF_LS_CLAMPED_TRIALS_FAIL], fr_fail = sum[PROF_LS_FREE_TRIALS_FAIL];
    double exh = sum[PROF_LS_EXHAUSTED], nb_cl = sum[PROF_BFGS_STEPS_ANY_CLAMP], tr_cl = sum[PROF_TRIALS_IN_ANY_CLAMP_STEPS];
    double snaps = sum[PROF_SNAPS], far = sum[PROF_SNAPS_FAR];
    double pinned = sum[PROF_PINNED_TORSIONS], nb_pin = sum[PROF_BFGS_STEPS_PINNED], tr_pin = sum[PROF_TRIALS_IN_PINNED_STEPS];
    auto pct = [](double a, double b){ return 100.0 * a / std::max(b, 1.0); };
    auto per = [](double a, double b){ return a / std::max(b, 1.0); };
    double active = sum[PROF_ACTIVE_TORSIONS], nb_act = sum[PROF_STEPS_WITH_ACTIVE], nb_steep = sum[PROF_STEPS_STEEPEST_GUARD];
    double nb_stop = sum[PROF_STEPS_CONSTRAINED_STOP], clipped_tr = sum[PROF_CLIPPED_TRIALS], clipped = sum[PROF_CLIPPED_TORSIONS];
    double clipped_fail = sum[PROF_LS_CLIPPED_TRIALS_FAIL], unclipped_fail = sum[PROF_LS_UNCLIPPED_TRIALS_FAIL];
    double nondescent = sum[PROF_CLIPPED_NONDESCENT], exh_nondescent = sum[PROF_LS_EXHAUSTED_WITH_NONDESCENT];
    spdlog::info("[prof:{}] torlib: clamped trials {:.0f} ({:.1f}% of trials), failed clamped {:.0f} ({:.1f}%) vs failed free {:.0f} ({:.1f}% of free trials), "
                 "exhausted line searches {:.0f} ({:.2f}% of bfgs steps), any-clamp bfgs steps {:.0f} with {:.0f} trials ({:.2f}/step vs {:.2f}/step in the others), "
                 "snaps {:.0f}, far {:.0f} ({:.1f}%), pinned torsions {:.0f}, pinned bfgs steps {:.0f} ({:.2f}% of steps) with {:.0f} trials ({:.2f}/step), outside torsions {:.0f}; "
                 "active set: active torsions {:.0f} ({:.3f}/bfgs step), steps with active {:.0f} ({:.2f}%), steepest guard {:.0f} ({:.2f}%), "
                 "constrained stops {:.0f} ({:.2f}%); bent step: clipped trials {:.0f} ({:.2f}% of trials), clipped torsions {:.0f} ({:.2f}/clipped trial), "
                 "failed clipped {:.0f} ({:.1f}% of clipped trials) vs failed unclipped {:.0f} ({:.1f}% of unclipped trials), "
                 "clipped nondescent {:.0f} ({:.2f}% of clipped trials), exhausted with nondescent {:.0f} ({:.2f}% of exhausted)",
                 label, cl, pct(cl, nls), cl_fail, pct(cl_fail, cl), fr_fail, pct(fr_fail, nls - cl),
                 exh, pct(exh, nb), nb_cl, tr_cl, per(tr_cl, nb_cl), per(nls - tr_cl, nb - nb_cl),
                 snaps, far, pct(far, snaps), pinned, nb_pin, pct(nb_pin, nb), tr_pin, per(tr_pin, nb_pin), sum[PROF_OUTSIDE_TORSIONS],
                 active, per(active, nb), nb_act, pct(nb_act, nb), nb_steep, pct(nb_steep, nb),
                 nb_stop, pct(nb_stop, nb), clipped_tr, pct(clipped_tr, nls), clipped, per(clipped, clipped_tr),
                 clipped_fail, pct(clipped_fail, clipped_tr), unclipped_fail, pct(unclipped_fail, nls - clipped_tr),
                 nondescent, pct(nondescent, clipped_tr), exh_nondescent, pct(exh_nondescent, exh));
    spdlog::info("[prof:{}] inter loop per eval: receptor atoms visited {:.0f} (sum over lanes), max-lane {:.0f}, scored pairs {:.0f} ({:.0f}% of visited) -> {:.0f} cyc per max-lane visit, {:.0f} cyc per scored pair (warp-serial)",
                 label, sum[8] / ne, sum[9] / ne, sum[10] / ne, 100.0 * sum[10] / std::max(sum[8], 1.0), sum[PROF_E_INTER] / std::max(sum[9], 1.0), sum[PROF_E_INTER] / std::max(sum[10], 1.0) * 32.0);
    spdlog::info("[prof:{}] per energy eval: inter {:.0f} cyc, intra+pen {:.0f} cyc, grad {:.0f} cyc, ls copy+update {:.0f} cyc  (total timed/eval {:.0f} cyc = {:.1f} us at {:.2f} GHz)",
                 label, sum[PROF_E_INTER] / ne, sum[PROF_E_INTRA] / ne, sum[PROF_GRAD] / ne, sum[PROF_LS_UPDATE] / ne, timed / ne, timed / ne / cyc_per_ms * 1000.0, khz / 1e6);
}
#endif
