//
// CPU BFGS Optimization
// Created for Uni-Dock2 CPU backend
//

#ifndef BFGS_CPU_H
#define BFGS_CPU_H

#include "common_cpu.h"
#include "score_cpu.h"
#include "model/model.h"
#include "myutils/matrix.h"

// Per-pose wall-time deadline (thread-local, set by MC, checked by line_search/BFGS)
void set_pose_deadline(double seconds);
bool pose_deadline_exceeded();

// Line search parameters
static const int LINE_SEARCH_STEPS_CPU = 10;
static const Real LINE_SEARCH_C0_CPU = 0.0001;
static const Real LINE_SEARCH_MULTIPLIER_CPU = 0.5;

/**
 * @brief Matrix-vector product: out_p = -H * g
 */
void minus_mat_vec_product_cpu(FlexPoseGradient* out_p, const Real* h,
                                const FlexPoseGradient* g, int dim);

/**
 * @brief Dot product of two gradients
 */
Real g_dot_product_cpu(const FlexPoseGradient* a, const FlexPoseGradient* b, int dim);

/**
 * @brief Line search with backtracking
 * 
 * @param fix_mol Fixed molecule
 * @param fix_param Fix parameters
 * @param flex_param Flex parameters
 * @param x Current pose
 * @param g Current gradient
 * @param flex_topo Flex topology
 * @param p Search direction
 * @param aux_f Auxiliary force array
 * @param out_x_new Output new pose
 * @param out_g_new Output new gradient
 * @param e0 Current energy
 * @param dim_g Gradient dimension
 * @param dim_x Pose dimension
 * @param out_e Output new energy
 * @param out_alpha Output step size
 */
void line_search_cpu(const FixMol& fix_mol, const FixParamVina& fix_param,
                     const FlexParamVina& flex_param, const FlexPose* x,
                     const FlexPoseGradient* g, const FlexTopo& flex_topo,
                     const FlexPoseGradient* p, Real* aux_f,
                     FlexPose* out_x_new, FlexPoseGradient* out_g_new,
                     Real e0, int dim_g, int dim_x, Real* out_e, Real* out_alpha);

/**
 * @brief Update Hessian matrix using BFGS formula
 */
void bfgs_update_hessian_cpu(Real* out_h, int dim, const FlexPoseGradient* p,
                              const FlexPoseGradient* y,
                              FlexPoseGradient* aux_minus_hy, Real alpha);

/**
 * @brief Main BFGS optimization
 * 
 * @param out_x Pose to optimize (input/output)
 * @param flex_topo Flex topology
 * @param fix_mol Fixed molecule
 * @param flex_param Flex parameters
 * @param fix_param Fix parameters
 * @param aux_x_new Auxiliary pose
 * @param aux_x_ori Auxiliary pose for original state
 * @param aux_g Auxiliary gradient
 * @param aux_g_new Auxiliary gradient
 * @param aux_g_ori Auxiliary gradient for original state
 * @param aux_p Search direction
 * @param aux_y y vector for BFGS
 * @param aux_minus_hy Auxiliary vector
 * @param aux_h Hessian matrix
 * @param aux_f Force array
 * @param max_steps Maximum optimization steps
 */
void bfgs_cpu(FlexPose* out_x, const FlexTopo& flex_topo, const FixMol& fix_mol,
              const FlexParamVina& flex_param, const FixParamVina& fix_param,
              FlexPose* aux_x_new, FlexPose* aux_x_ori,
              FlexPoseGradient* aux_g, FlexPoseGradient* aux_g_new,
              FlexPoseGradient* aux_g_ori,
              FlexPoseGradient* aux_p, FlexPoseGradient* aux_y,
              FlexPoseGradient* aux_minus_hy,
              FlexPoseHessian* aux_h, FlexForce* aux_f, int max_steps);

/**
 * @brief Simplified BFGS that allocates its own auxiliary memory
 * WARNING: This allocates memory on every call - use bfgs_with_memory for hot loops
 */
void bfgs_simple_cpu(FlexPose* out_x, const FlexTopo& flex_topo,
                     const FixMol& fix_mol, const FlexParamVina& flex_param,
                     const FixParamVina& fix_param, int max_steps);


/**
 * @brief Pre-allocated memory structure for BFGS optimization
 * Use this to avoid repeated allocations in hot loops
 */
struct BFGSThreadMemory {
    std::vector<Real> coords_new;
    std::vector<Real> dihedrals_new;
    std::vector<Real> coords_ori;
    std::vector<Real> dihedrals_ori;
    std::vector<Real> dihedrals_g;
    std::vector<Real> dihedrals_g_new;
    std::vector<Real> dihedrals_g_ori;
    std::vector<Real> dihedrals_p;
    std::vector<Real> dihedrals_y;
    std::vector<Real> dihedrals_minus_hy;
    std::vector<Real> h_matrix;
    std::vector<Real> forces;

    // Cartesian-mode buffers (size = 3 * natom). Unused in internal mode.
    std::vector<Real> grad;
    std::vector<Real> grad_new;
    std::vector<Real> p;
    std::vector<Real> y;
    std::vector<Real> minus_hy;

    void resize(int natom, int ntorsion) {
        int h_size = (dof_g + ntorsion) * (dof_g + ntorsion + 1) / 2;
        coords_new.resize(natom * 3);
        dihedrals_new.resize(ntorsion);
        coords_ori.resize(natom * 3);
        dihedrals_ori.resize(ntorsion);
        dihedrals_g.resize(ntorsion);
        dihedrals_g_new.resize(ntorsion);
        dihedrals_g_ori.resize(ntorsion);
        dihedrals_p.resize(ntorsion);
        dihedrals_y.resize(ntorsion);
        dihedrals_minus_hy.resize(ntorsion);
        h_matrix.resize(h_size);
        forces.resize(natom * 3);
    }

    void resize_cartesian(int natom) {
        int dim = 3 * natom;
        int h_size = dim * (dim + 1) / 2;
        coords_new.resize(natom * 3);
        coords_ori.resize(natom * 3);
        forces.resize(natom * 3);
        grad.resize(dim);
        grad_new.resize(dim);
        p.resize(dim);
        y.resize(dim);
        minus_hy.resize(dim);
        h_matrix.resize(h_size);
    }
};

/**
 * @brief BFGS optimization using pre-allocated memory (for hot loops)
 */
void bfgs_with_memory_cpu(FlexPose* out_x, const FlexTopo& flex_topo,
                          const FixMol& fix_mol, const FlexParamVina& flex_param,
                          const FixParamVina& fix_param, int max_steps,
                          BFGSThreadMemory& mem);

#endif // BFGS_CPU_H
