//
// CPU BFGS Optimization Implementation
// Created for Uni-Dock2 CPU backend
//

#include "bfgs_cpu.h"
#include <cmath>
#include <vector>
#include <cstring>
#include <chrono>

static const Real EPSILON_CPU = std::numeric_limits<Real>::epsilon();

// Thread-local pose deadline — set by mc_single_pose_cpu, checked in line_search
// to abort expensive evaluations on pathological conformations.
thread_local std::chrono::steady_clock::time_point g_pose_deadline;

void set_pose_deadline(double seconds) {
    g_pose_deadline = std::chrono::steady_clock::now()
                    + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                          std::chrono::duration<double>(seconds));
}

bool pose_deadline_exceeded() {
    if (g_pose_deadline.time_since_epoch().count() == 0) return false;
    return std::chrono::steady_clock::now() > g_pose_deadline;
}


void minus_mat_vec_product_cpu(FlexPoseGradient* out_p, const Real* h,
                                const FlexPoseGradient* g, int dim) {
    // out_p = -H * g
    // H is stored as upper triangular matrix
    for (int i = 0; i < dim; i++) {
        Real sum = 0;
        for (int j = 0; j < dim; j++) {
            sum += h[tri_mat_index(i, j)] * grad_index_read(g, j);
        }
        grad_index_write(out_p, i, -sum);
    }
}


Real g_dot_product_cpu(const FlexPoseGradient* a, const FlexPoseGradient* b, int dim) {
    Real result = 0;
    for (int i = 0; i < dim; i++) {
        result += grad_index_read(a, i) * grad_index_read(b, i);
    }
    return result;
}


void line_search_cpu(const FixMol& fix_mol, const FixParamVina& fix_param,
                     const FlexParamVina& flex_param, const FlexPose* x,
                     const FlexPoseGradient* g, const FlexTopo& flex_topo,
                     const FlexPoseGradient* p, Real* aux_f,
                     FlexPose* out_x_new, FlexPoseGradient* out_g_new,
                     Real e0, int dim_g, int dim_x, Real* out_e, Real* out_alpha) {
    Real alpha = 1.0;
    Real e_new = 0;
    
    const Real pg = g_dot_product_cpu(p, g, dim_g);
    
    for (int trial = 0; trial < LINE_SEARCH_STEPS_CPU; trial++) {
        // Bail if the per-pose wall-time deadline has passed
        if (pose_deadline_exceeded()) break;

        // Copy x to x_new
        duplicate_pose_cpu(out_x_new, x, dim_x, flex_topo.natom);
        
        // Apply alpha * gradient
        apply_grad_update_pose_cpu(out_x_new, p, flex_topo, alpha);
        
        // Calculate new energy and gradient
        e_new = cal_e_grad_cpu(out_x_new, out_g_new, flex_topo, fix_mol, flex_param, fix_param, aux_f);
        
        // Armijo condition
        if (e_new - e0 < LINE_SEARCH_C0_CPU * alpha * pg) {
            break;
        }
        
        alpha *= LINE_SEARCH_MULTIPLIER_CPU;
    }
    
    *out_e = e_new;
    *out_alpha = alpha;
}


void bfgs_update_hessian_cpu(Real* out_h, int dim, const FlexPoseGradient* p,
                              const FlexPoseGradient* y,
                              FlexPoseGradient* aux_minus_hy, Real alpha) {
    Real yp = g_dot_product_cpu(y, p, dim);
    
    if (alpha * yp < EPSILON_CPU) {
        return;
    }
    
    // aux_minus_hy = -H * y
    minus_mat_vec_product_cpu(aux_minus_hy, out_h, y, dim);
    
    // yhy = y^T * H * y
    Real yhy = -g_dot_product_cpu(y, aux_minus_hy, dim);
    Real r = 1.0 / (alpha * yp);
    
    // Update Hessian
    for (int i = 0; i < dim; i++) {
        for (int j = i; j < dim; j++) {
            Real tmp = alpha * r * (grad_index_read(aux_minus_hy, i) * grad_index_read(p, j) +
                                   grad_index_read(aux_minus_hy, j) * grad_index_read(p, i))
                     + alpha * alpha * (r * r * yhy + r) * grad_index_read(p, i) * grad_index_read(p, j);
            
            out_h[i + j * (j + 1) / 2] += tmp;
        }
    }
}


void bfgs_cpu(FlexPose* out_x, const FlexTopo& flex_topo, const FixMol& fix_mol,
              const FlexParamVina& flex_param, const FixParamVina& fix_param,
              FlexPose* aux_x_new, FlexPose* aux_x_ori,
              FlexPoseGradient* aux_g, FlexPoseGradient* aux_g_new,
              FlexPoseGradient* aux_g_ori,
              FlexPoseGradient* aux_p, FlexPoseGradient* aux_y,
              FlexPoseGradient* aux_minus_hy,
              FlexPoseHessian* aux_h, FlexForce* aux_f, int max_steps) {
    
    int dim_g = dof_g + flex_topo.ntorsion;
    int dim_x = dof_x + flex_topo.ntorsion;
    
    // Calculate initial energy and gradient
    Real E_ori = cal_e_grad_cpu(out_x, aux_g, flex_topo, fix_mol, flex_param, fix_param, aux_f->f);
    Real E = E_ori;
    Real E1 = 0;
    Real alpha = 0;
    
    // Initialize Hessian as identity matrix
    int h_size = dim_g * (dim_g + 1) / 2;
    std::memset(aux_h->matrix, 0, h_size * sizeof(Real));
    for (int i = 0; i < dim_g; i++) {
        aux_h->matrix[tri_mat_index(i, i)] = 1.0;
    }
    
    // Initialize auxiliary structures
    duplicate_grad_cpu(aux_g_new, aux_g, dim_g);
    duplicate_pose_cpu(aux_x_new, out_x, dim_x, flex_topo.natom);
    duplicate_grad_cpu(aux_g_ori, aux_g, dim_g);
    duplicate_pose_cpu(aux_x_ori, out_x, dim_x, flex_topo.natom);
    duplicate_grad_cpu(aux_p, aux_g, dim_g);
    
    for (int step = 0; step < max_steps; step++) {
        if (pose_deadline_exceeded()) break;

        // Compute search direction: p = -H * g
        minus_mat_vec_product_cpu(aux_p, aux_h->matrix, aux_g, dim_g);
        
        // Line search
        line_search_cpu(fix_mol, fix_param, flex_param, out_x, aux_g, flex_topo,
                       aux_p, aux_f->f, aux_x_new, aux_g_new, E, dim_g, dim_x, &E1, &alpha);
        
        // aux_y = aux_g_new - aux_g
        duplicate_grad_cpu(aux_y, aux_g_new, dim_g);
        for (int i = 0; i < dim_g; i++) {
            Real tmp = grad_index_read(aux_y, i) - grad_index_read(aux_g, i);
            grad_index_write(aux_y, i, tmp);
        }
        
        // Update energy
        E = E1;
        
        // Update pose: out_x = aux_x_new
        duplicate_pose_cpu(out_x, aux_x_new, dim_x, flex_topo.natom);
        
        // Check convergence
        Real gg = g_dot_product_cpu(aux_g, aux_g, dim_g);
        if (std::sqrt(gg) < 1e-5f) {
            break;
        }
        
        // Update gradient
        duplicate_grad_cpu(aux_g, aux_g_new, dim_g);
        
        // Heuristic initial Hessian at first step
        if (step == 0) {
            Real yy = g_dot_product_cpu(aux_y, aux_y, dim_g);
            if (std::abs(yy) > EPSILON_CPU) {
                Real yp = g_dot_product_cpu(aux_y, aux_p, dim_g);
                Real scale = alpha * yp / yy;
                for (int i = 0; i < dim_g; i++) {
                    aux_h->matrix[tri_mat_index(i, i)] = scale;
                }
            }
        }
        
        // Update Hessian
        bfgs_update_hessian_cpu(aux_h->matrix, dim_g, aux_p, aux_y, aux_minus_hy, alpha);
    }
    
    // If optimization failed, restore original state
    if (E > E_ori) {
        E = E_ori;
        duplicate_pose_cpu(out_x, aux_x_ori, dim_x, flex_topo.natom);
        duplicate_grad_cpu(aux_g, aux_g_ori, dim_g);
    }
    
    out_x->energy = E;
}


void bfgs_with_memory_cpu(FlexPose* out_x, const FlexTopo& flex_topo,
                          const FixMol& fix_mol, const FlexParamVina& flex_param,
                          const FixParamVina& fix_param, int max_steps,
                          BFGSThreadMemory& mem) {
    int natom = flex_topo.natom;
    int ntorsion = flex_topo.ntorsion;
    
    // Ensure memory is properly sized (no-op if already correct size)
    mem.resize(natom, ntorsion);
    
    FlexPose aux_x_new, aux_x_ori;
    aux_x_new.coords = mem.coords_new.data();
    aux_x_new.dihedrals = mem.dihedrals_new.data();
    aux_x_ori.coords = mem.coords_ori.data();
    aux_x_ori.dihedrals = mem.dihedrals_ori.data();
    
    FlexPoseGradient aux_g, aux_g_new, aux_g_ori, aux_p, aux_y, aux_minus_hy;
    aux_g.dihedrals_g = mem.dihedrals_g.data();
    aux_g_new.dihedrals_g = mem.dihedrals_g_new.data();
    aux_g_ori.dihedrals_g = mem.dihedrals_g_ori.data();
    aux_p.dihedrals_g = mem.dihedrals_p.data();
    aux_y.dihedrals_g = mem.dihedrals_y.data();
    aux_minus_hy.dihedrals_g = mem.dihedrals_minus_hy.data();
    
    FlexPoseHessian aux_h;
    aux_h.matrix = mem.h_matrix.data();
    
    FlexForce aux_f;
    aux_f.f = mem.forces.data();
    
    bfgs_cpu(out_x, flex_topo, fix_mol, flex_param, fix_param,
             &aux_x_new, &aux_x_ori, &aux_g, &aux_g_new, &aux_g_ori,
             &aux_p, &aux_y, &aux_minus_hy, &aux_h, &aux_f, max_steps);
}


void bfgs_simple_cpu(FlexPose* out_x, const FlexTopo& flex_topo,
                     const FixMol& fix_mol, const FlexParamVina& flex_param,
                     const FixParamVina& fix_param, int max_steps) {
    // Use thread-local memory to avoid repeated allocations
    static thread_local BFGSThreadMemory mem;
    bfgs_with_memory_cpu(out_x, flex_topo, fix_mol, flex_param, fix_param, max_steps, mem);
}
