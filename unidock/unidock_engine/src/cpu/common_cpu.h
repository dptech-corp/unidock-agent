//
// CPU Common Definitions and Utilities
// Created for Uni-Dock2 CPU backend
//

#ifndef COMMON_CPU_H
#define COMMON_CPU_H

#include <limits>
#include <cmath>
#include <algorithm>

#if USE_OPENMP
#include <omp.h>
#endif

#include "model/model.h"
#include "score/vina.h"
#include "cpu/spatial_grid.h"

// CPU version doesn't need CUDA decorators
#ifndef SCOPE_CU
#define SCOPE_CU
#endif

#ifndef SCOPE_INLINE
#define SCOPE_INLINE inline
#endif

// Constants that were in CUDA constant memory
struct CPUConstants {
    bool FLAG_CONSTRAINT_DOCK = false;
    Real BOX_X_HI = 0;
    Real BOX_X_LO = 0;
    Real BOX_Y_HI = 0;
    Real BOX_Y_LO = 0;
    Real BOX_Z_HI = 0;
    Real BOX_Z_LO = 0;
    Real TOR_PREC = 0.3;
    Real BOX_PREC = 2.0;
    Real PENALTY_SLOPE = 10.0;
    bool USE_INNER_BOX = false;   // inner box on the ligand heavy-atom centroid
    Real INNER_X_LO = 0, INNER_X_HI = 0, INNER_Y_LO = 0, INNER_Y_HI = 0, INNER_Z_LO = 0, INNER_Z_HI = 0;
    Vina Score;
    AtomTypeProps atp;  // Data-driven atom type properties
    Real vdw_radii[MAX_VN_TYPES];  // Data-driven VDW radii (from JSON or fallback)
    // The term settings (bias type, k) live in each ligand's FlexParamVina::terms.settings (score/terms.h)

    // Torsion library penalty method settings
    bool USE_TORSION_PENALTY = true;   // Use soft penalty instead of hard clamping
    Real TORSION_PENALTY_K = 5.0;      // Penalty strength (kcal/mol per radian^2)

    // Spatial grid for fast receptor neighbor lookup
    SpatialGrid receptor_grid = {};

    // Receptor donor-H topology for angular HBond (receptor-donor direction)
    RecDonorHTopo rec_donor_h_topo = {};

    // Inline angular HBond scoring (replaces post-BFGS binary correction)
    bool USE_ANGULAR_HBOND = true;       // Enable smooth angular weight in scoring loop
    VinaWeights weights;                 // scoring-function term weights (search and host scoring)
    bool USE_POST_BFGS_HBOND_CORR = false; // Legacy post-BFGS binary correction (off when angular is on)
};

// Global CPU constants (initialized in init_constants_cpu)
extern CPUConstants g_cpu_constants;

/**
 * @brief Initialize CPU constants from DockParam
 * Replacement for CUDA's cudaMemcpyToSymbol
 */
void init_constants_cpu(const DockParam& dock_param, const AtomTypeProps& type_props,
                        const std::vector<Real>& vdw_radii_vec = {});

/**
 * @brief Get number of threads for OpenMP
 */
inline int get_num_threads() {
#if USE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/**
 * @brief Get current thread ID
 */
inline int get_thread_id() {
#if USE_OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// ==================== Math utilities ====================

/**
 * @brief Calculate squared distance between two 3D points
 */
SCOPE_INLINE Real dist_sq_cpu(const Real* a, const Real* b) {
    Real dx = a[0] - b[0];
    Real dy = a[1] - b[1];
    Real dz = a[2] - b[2];
    return dx*dx + dy*dy + dz*dz;
}

/**
 * @brief Calculate distance between two 3D points
 */
SCOPE_INLINE Real dist_cpu(const Real* a, const Real* b) {
    return std::sqrt(dist_sq_cpu(a, b));
}

/**
 * @brief Calculate norm of a 3D vector
 */
SCOPE_INLINE Real norm_cpu(const Real* v) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

/**
 * @brief Calculate squared norm of a 3D vector
 */
SCOPE_INLINE Real norm_sq_cpu(const Real* v) {
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}

/**
 * @brief Normalize a 3D vector in place
 */
SCOPE_INLINE void normalize_cpu(Real* v) {
    Real n = norm_cpu(v);
    if (n > 1e-10) {
        v[0] /= n;
        v[1] /= n;
        v[2] /= n;
    }
}

/**
 * @brief Cross product of two 3D vectors: out = a x b
 */
SCOPE_INLINE void cross_cpu(const Real* a, const Real* b, Real* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

/**
 * @brief Dot product of two 3D vectors
 */
SCOPE_INLINE Real dot_cpu(const Real* a, const Real* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/**
 * @brief Get real value within range based on integer
 * Replacement for CUDA get_real_within_by_int
 */
SCOPE_INLINE Real get_real_within_by_int_cpu(int c, Real min, Real max, int n = 1001) {
    Real v = Real(c % n) * (max - min) / n + min;
    return std::clamp(v, min, max);
}

/**
 * @brief Clamp a value to [lo, hi] range
 * CPU version of clamp_by_range from mymath.h
 */
SCOPE_INLINE Real clamp_by_range_cpu(Real val, Real hi, Real lo) {
    if (val > hi) return hi;
    if (val < lo) return lo;
    return val;
}

/**
 * @brief Pick a random radian from allowed ranges
 * CPU version of get_radian_in_ranges from mymath.h
 * @param ranges Array of [lo, hi] pairs
 * @param num_range Number of ranges
 * @param rand_2 Two random numbers in (0, 1]
 * @return A randomly selected angle within the given ranges
 */
SCOPE_INLINE Real get_radian_in_ranges_cpu(const Real* ranges, int num_range, const Real* rand_2) {
    int i_range = (int)std::ceil(rand_2[0] * num_range - 1);
    if (i_range < 0) i_range = 0;
    if (i_range >= num_range) i_range = num_range - 1;
    Real from = ranges[i_range * 2];
    Real to = ranges[i_range * 2 + 1];
    return from + (to - from) * rand_2[1];
}

/**
 * @brief Compute gyration radius of a pose
 */
SCOPE_INLINE Real gyration_radius_cpu(const FlexPose* flex_pose, const FlexTopo* flex_topo) {
    Real d_sq_sum = 0;
    int natom = 0;
    
    for (int i = 0; i < flex_topo->natom; i++) {
        // Only tackle non-H atoms
        if (!g_cpu_constants.atp.is_hydrogen[flex_topo->vn_types[i]]) {
            d_sq_sum += dist_sq_cpu(flex_pose->coords + 3 * i, flex_pose->center);
            ++natom;
        }
    }
    return natom > 0 ? std::sqrt(d_sq_sum / natom) : 0;
}


// ==================== Quaternion operations ====================

/**
 * @brief Normalize quaternion in place
 */
SCOPE_INLINE void normalize_quaternion_cpu(Real* q) {
    Real n = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (n > 1e-10) {
        q[0] /= n;
        q[1] /= n;
        q[2] /= n;
        q[3] /= n;
    }
}

/**
 * @brief Quaternion multiplication: out = a * b
 */
SCOPE_INLINE void quat_multiply_cpu(const Real* a, const Real* b, Real* out) {
    out[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    out[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    out[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    out[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

/**
 * @brief Rotate a 3D vector by quaternion: out = q * v * q^(-1)
 */
SCOPE_INLINE void rotate_by_quaternion_cpu(const Real* q, const Real* v, Real* out) {
    // Using the formula: v' = v + 2*w*(q_xyz x v) + 2*(q_xyz x (q_xyz x v))
    // where q = (w, q_xyz) = (q[0], q[1], q[2], q[3])
    Real w = q[0];
    Real qv[3] = {q[1], q[2], q[3]};
    
    Real cross1[3];
    cross_cpu(qv, v, cross1);
    
    Real cross2[3];
    cross_cpu(qv, cross1, cross2);
    
    out[0] = v[0] + 2*w*cross1[0] + 2*cross2[0];
    out[1] = v[1] + 2*w*cross1[1] + 2*cross2[1];
    out[2] = v[2] + 2*w*cross1[2] + 2*cross2[2];
}

/**
 * @brief Create quaternion from axis-angle representation
 * @param axis Rotation axis (normalized)
 * @param angle Rotation angle in radians
 * @param out Output quaternion (w, x, y, z)
 */
SCOPE_INLINE void axis_angle_to_quaternion_cpu(const Real* axis, Real angle, Real* out) {
    Real half_angle = angle * 0.5;
    Real s = std::sin(half_angle);
    out[0] = std::cos(half_angle);
    out[1] = axis[0] * s;
    out[2] = axis[1] * s;
    out[3] = axis[2] * s;
}

/**
 * @brief Convert rotation vector to quaternion
 * @param rot_vec Rotation vector (3D)
 * @param out Output quaternion (w, x, y, z)
 */
SCOPE_INLINE void rot_vec_to_quaternion_cpu(const Real* rot_vec, Real* out) {
    Real angle = norm_cpu(rot_vec);
    if (angle < 1e-10) {
        out[0] = 1.0;
        out[1] = 0.0;
        out[2] = 0.0;
        out[3] = 0.0;
    } else {
        Real axis[3] = {rot_vec[0]/angle, rot_vec[1]/angle, rot_vec[2]/angle};
        axis_angle_to_quaternion_cpu(axis, angle, out);
    }
}


#endif // COMMON_CPU_H
