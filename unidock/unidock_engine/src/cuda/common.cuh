//
// Created by Congcong Liu on 24-10-9.
//

#ifndef COMMON_CUH
#define COMMON_CUH

#include <curand_kernel.h>
#include "myutils/mymath.h"
#include "score/vina.h"
#include "cpu/spatial_grid.h"
#include <cuda_runtime.h>

// -------------- Constants -----------------
extern __constant__ bool FLAG_CONSTRAINT_DOCK;
extern __constant__ Real BOX_X_HI;
extern __constant__ Real BOX_X_LO;
extern __constant__ Real BOX_Y_HI;
extern __constant__ Real BOX_Y_LO;
extern __constant__ Real BOX_Z_HI;
extern __constant__ Real BOX_Z_LO;
extern __constant__ Real TOR_PREC;
extern __constant__ Real BOX_PREC;
extern __constant__ Real PENALTY_SLOPE;
extern __constant__ bool USE_INNER_BOX;   // inner box on the ligand heavy-atom centroid
extern __constant__ Real INNER_X_LO;
extern __constant__ Real INNER_X_HI;
extern __constant__ Real INNER_Y_LO;
extern __constant__ Real INNER_Y_HI;
extern __constant__ Real INNER_Z_LO;
extern __constant__ Real INNER_Z_HI;
extern __constant__ TermSettings g_term_settings;   // device-side mirror of FlexParamVina::terms.settings (score/terms.h): the kernel's term guard and settings

#if true
extern __constant__ Vina Score;
#else
extern __constant__ Gaff2 Score;
#endif

extern __constant__ AtomTypeProps g_atp;
extern __constant__ Real g_vdw_radii[MAX_VN_TYPES];  // per-type vdW radii; r1 + r2 is derived from types on device
extern __constant__ bool USE_ANGULAR_HBOND;          // angle-weighted HBond term (mirrors CPU backend)
extern __constant__ SpatialGrid g_grid;              // receptor cell list (device pointers); built == false -> brute force

/**
 * @brief Build the receptor cell list on the host (same builder as the CPU backend) and upload it.
 * Replaces the per-ligand-atom loop over every receptor atom by a 27-cell neighborhood walk.
 */
void upload_receptor_grid(const Real* coords, const int* atom_types, int natom,
                          const AtomTypeProps& type_props, const std::vector<Real>& vdw_radii = {},
                          const Real* charges = nullptr);
void free_receptor_grid();
// Per-type vdW radii table (MAX_VN_TYPES entries): JSON/DMS-provided if available, else the built-in table
void fill_vdw_radii(Real* out, const std::vector<Real>& vdw_radii);

/**
 * @brief Dynamic shared memory the energy kernels need: a per-block force accumulator of
 * max_natom * 3 Reals (see cal_e_f_warp). Call set_kernel_max_natom before launching.
 */
void set_kernel_max_natom(int natom);
size_t kernel_dyn_shared_bytes();
// Sets the L1/shared carveout of func for a launch of nblock blocks of block_size threads with dyn bytes of dynamic
// shared memory. Environment variable env (else fallback) picks the mode: occ32 (smallest configuration holding the
// wave-balanced blocks per SM of this launch, never below 32 KB), occ / occw (same without the 32 KB floor; occ without
// wave balancing), kb<N> (fixed configuration), default (driver) or an explicit percent.
void set_carveout(const void* func, const char* label, const char* env, const char* fallback, int nblock, int block_size,
                  size_t dyn);
// Ligands up to this many atoms accumulate forces in shared memory; larger ones use
// global atomics on aux_f so the per-block shared footprint does not throttle residency.
#define KERNEL_SHARED_FORCE_MAX_ATOMS 512

// Placement of the per-pose BFGS state (bfgs_state_ptrs in bfgs.cuh): bit set = in dynamic shared memory behind the
// force accumulator, clear = in the pose's global block.
#define BFGS_STATE_V_SHARED 1   // the STRIDE_G DOF vectors
#define BFGS_STATE_H_SHARED 2   // the Hessian
// Largest torsion count of the ligands the kernels will see (sizes the shared BFGS state); grows only, like the natom one.
void set_kernel_max_ntorsion(int ntorsion);
// Dynamic shared memory of a BFGS kernel launch with this placement: the force accumulator plus the state parts in
// shared memory (a ligand above KERNEL_SHARED_FORCE_MAX_ATOMS atoms has no shared accumulator and puts them at 0).
size_t kernel_state_dyn_shared_bytes(int state_flags);
// Placement for a launch of nblock blocks of func. Environment UD2_BFGS_STATE: global | v | h | vh fixes it; auto
// (default) takes the most of the state into shared memory (vectors and Hessian, else the vectors, else the Hessian)
// that leaves the occ32 carveout configuration of the launch unchanged and still holds the blocks per SM it needs.
int bfgs_state_flags(const void* func, const char* label, int nblock, int block_size);

/**
 * @brief Upload box, precision, scoring switches, atom-type table and vdW radii to constant memory.
 * vdw_radii may be empty, in which case the built-in VN_VDW_RADII table is used.
 */
void init_constants(const DockParam& dock_param, const AtomTypeProps& type_props,
                    const std::vector<Real>& vdw_radii = {});

/**
 * @brief Build the symmetric CSR neighbor view of an intra pair list (host side).
 * out_start has natom + 1 entries, out_nbr has npair * 2 entries.
 */
void build_intra_csr(const int* pairs, int npair, int natom,
                     std::vector<int>& out_start, std::vector<int>& out_nbr);

/**
 * @brief Reorder an intra pair list round-robin over the first atom (host side).
 * The device scores 32 consecutive pairs per batch with shared-memory atomics on both
 * atoms; an i-major order makes a whole batch hit one force slot and serialize.
 */
void reorder_pairs_round_robin(const int* pairs, int npair, int natom, std::vector<int>& out);


#define TILE_SIZE 32

// ---------------- Optional phase profiler (compile with -DUD2_PROFILE) ----------------
// Lane 0 of each warp accumulates clock64() deltas per phase into its own slot
// (g_prof_ptr[blockIdx.x * PROF_SLOTS + phase]); no atomics, host reduces afterwards.
#define PROF_SLOTS 40
enum ProfPhase {
    PROF_RANDOMIZE = 0, PROF_MUTATE = 1, PROF_E_INTER = 2, PROF_E_INTRA = 3, PROF_GRAD = 4,
    PROF_LS_UPDATE = 5, PROF_BFGS_ALGEBRA = 6, PROF_MC_BOOKKEEP = 7,
    PROF_N_EVAL = 12, PROF_N_LS_TRIAL = 13, PROF_N_BFGS_STEP = 14, PROF_N_MC_STEP = 15,
    // torsion-library line-search counters (line_search_warp)
    PROF_LS_CLAMPED_TRIALS = 16, PROF_LS_CLAMPED_TRIALS_FAIL = 17, PROF_LS_FREE_TRIALS_FAIL = 18,
    PROF_LS_EXHAUSTED = 19, PROF_BFGS_STEPS_ANY_CLAMP = 20, PROF_TRIALS_IN_ANY_CLAMP_STEPS = 21,
    PROF_SNAPS = 22, PROF_SNAPS_FAR = 23, PROF_PINNED_TORSIONS = 24, PROF_BFGS_STEPS_PINNED = 25,
    PROF_TRIALS_IN_PINNED_STEPS = 26, PROF_OUTSIDE_TORSIONS = 27,
    // torsion-range active set and bent line-search step counters (bfgs_warp, line_search_warp)
    PROF_ACTIVE_TORSIONS = 28, PROF_STEPS_WITH_ACTIVE = 29, PROF_STEPS_STEEPEST_GUARD = 30,
    PROF_STEPS_CONSTRAINED_STOP = 31, PROF_CLIPPED_TRIALS = 32, PROF_CLIPPED_TORSIONS = 33,
    // failed trials split by the bent step (clipped or not); clipped trials whose step taken predicts no decrease
    // (alpha p.g + d_clip >= 0), and the exhausted line searches that had one
    PROF_LS_CLIPPED_TRIALS_FAIL = 34, PROF_LS_UNCLIPPED_TRIALS_FAIL = 35, PROF_CLIPPED_NONDESCENT = 36,
    PROF_LS_EXHAUSTED_WITH_NONDESCENT = 37
};
#ifdef UD2_PROFILE
extern __device__ unsigned long long* g_prof_ptr;
#define PROF_DECL long long _prof_t = clock64()
#define PROF_MARK(phase) do { long long _t = clock64(); if (g_prof_ptr && threadIdx.x % TILE_SIZE == 0) g_prof_ptr[blockIdx.x * PROF_SLOTS + (phase)] += (unsigned long long)(_t - _prof_t); _prof_t = _t; } while (0)
#define PROF_RESET() do { _prof_t = clock64(); } while (0)
#define PROF_COUNT(slot) do { if (g_prof_ptr && threadIdx.x % TILE_SIZE == 0) g_prof_ptr[blockIdx.x * PROF_SLOTS + (slot)] += 1ULL; } while (0)
#define PROF_ADD(slot, n) do { if (g_prof_ptr && threadIdx.x % TILE_SIZE == 0) g_prof_ptr[blockIdx.x * PROF_SLOTS + (slot)] += (unsigned long long)(n); } while (0)
void prof_begin(int nwarps);
void prof_report(const char* label, int nwarps, double wall_ms);
#else
#define PROF_DECL do {} while (0)
#define PROF_MARK(phase) do {} while (0)
#define PROF_RESET() do {} while (0)
#define PROF_COUNT(slot) do {} while (0)
#define PROF_ADD(slot, n) do {} while (0)
#endif
#define STRIDE_POSE 4
#define STRIDE_G 6



/**
 * @brief Generate an integer in [min, max] on CUDA
 * 
 * @param state CUDA random state
 * @param min Minimum value (int)
 * @param max Maximum value (int)
 * @return An integer in [min, max]
 */
__forceinline__ __device__ int gen_rand_int_within(curandStatePhilox4_32_10_t* state, int min, int max) {
    int range = max - min + 1;
    return min + (curand(state) % range);
}


/**
 * @brief Generate a real number in [min, max]
 * 
 * @param c A random integer within Integer range   
 * @param min Minimum value (Real)
 * @param max Maximum value (Real)
 * @param n Number of divisions. The larger n is, the more accurate the result is.
 * @return A real number in [min, max]
 */
SCOPE_INLINE Real get_real_within_by_int(int c, Real min, Real max, int n=1001) {
    Real v = Real(c % n) * (max - min) / n + min;
    if (v < min){
        return min;
    }
    if(v > max){
        return max;
    }
    return v;
}



/**
 * @brief Generate a real number in [min, max] on CUDA
 * 
 * @param state CUDA random state
 * @param min Minimum value (Real)
 * @param max Maximum value (Real)
 * @return A real number in [min, max]
 */
__forceinline__ __device__ Real gen_rand_real_within(curandStatePhilox4_32_10_t* state, Real min, Real max) {
    return get_real_within_by_int(curand(state), min, max);
}




__forceinline__ __device__ void gen_4_rand_in_sphere(Real *out_rand_4, curandStatePhilox4_32_10_t *state) {
    float4 rand_4;
    while (true) {  // on average, this will have to be run about twice
        rand_4 = curand_uniform4(state);  // ~ U[0,1]
        out_rand_4[0] = (rand_4.x - 0.5) * 2.0; // [-1, 1]
        out_rand_4[1] = (rand_4.y - 0.5) * 2.0;
        out_rand_4[2] = (rand_4.z - 0.5) * 2.0;

        if (cal_norm(out_rand_4) < 1) { // not a specific length, also a randomization on rotation degree
            out_rand_4[3] = rand_4.w;
            return;
        }
    }
}


SCOPE_INLINE Real gyration_radius(const FlexPose* flex_pose, const FlexTopo* flex_topo){
    // compute gyration radius:
    Real d_sq_sum = 0;
    int natom = 0;
    // for each atom
    for (int i = 0; i < flex_topo->natom; i++){
        // only tackle non-H atoms
        if (!g_atp.is_hydrogen[flex_topo->vn_types[i]]){
            d_sq_sum += dist_sq(flex_pose->coords + 3 * i, flex_pose->center);
            ++natom;
        }
    }
    return natom > 0 ? sqrtf(d_sq_sum / natom) : 0;
}



#endif //COMMON_CUH


