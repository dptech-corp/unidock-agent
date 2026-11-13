//
// Created by Congcong Liu on 24-12-9.
//


// src/cuda/bfgs_wrapper.cu
#include "model/model.h"
#include "bfgs.cuh"


// The device routines keep DOF vectors flat (bfgs.cuh); the test API passes FlexPoseGradient. Copy between the two.
__device__ __forceinline__ void grad_to_flat(const cg::thread_block_tile<TILE_SIZE>& tile, const FlexPoseGradient* g,
                                             Real* out_v, int dim){
    for (int i = tile.thread_rank(); i < dim; i += tile.num_threads()){
        out_v[i * STRIDE_G] = grad_index_read(g, i);
    }
    tile.sync();
}

__device__ __forceinline__ void flat_to_grad(const cg::thread_block_tile<TILE_SIZE>& tile, const Real* v,
                                             FlexPoseGradient* out_g, int dim){
    for (int i = tile.thread_rank(); i < dim; i += tile.num_threads()){
        grad_index_write(out_g, i, v[i * STRIDE_G]);
    }
    tile.sync();
}

// zero-filled device buffer of STRIDE_G flat DOF vectors
static Real* alloc_flat_vectors_gpu(int ntorsion){
    Real* v = nullptr;
    const size_t bytes = (size_t)STRIDE_G * (dof_g + ntorsion) * sizeof(Real);
    checkCUDA(cudaMalloc(&v, bytes));
    checkCUDA(cudaMemset(v, 0, bytes));
    return v;
}


__global__ void line_search_one_original_kernel(const FixMol* fix_mol, const FixParamVina* fix_param,
                                   const FlexParamVina* flex_param, const FlexPose* x, const FlexTopo* flex_topo,
                                   FlexPoseGradient* aux_g, FlexPoseGradient* aux_p,
                                   FlexPoseHessian* aux_h, FlexForce* aux_f,
                                   FlexPose* out_x_new, FlexPoseGradient* out_g_new,
                                   Real* out_e, Real* out_alpha, Real* aux_v) {
    auto tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());

    //DOF, the vector x has this dimension
    int dim_g = 3 + 3 + flex_topo->ntorsion; // center, orientation, torsion
    int dim_x = 3 + 4 + flex_topo->ntorsion;
    Real* g = aux_v + BV_G;
    Real* g_new = aux_v + BV_G_NEW;
    Real* p = aux_v + BV_P;
    grad_to_flat(tile, aux_g, g, dim_g);
    grad_to_flat(tile, aux_p, p, dim_g);
    grad_to_flat(tile, out_g_new, g_new, dim_g);

    Real e0 = cal_e_grad_warp(tile, x, g, *flex_topo, *fix_mol, *flex_param, *fix_param, aux_f->f);
    // Use unit matrix as initial guess for Hessian
    init_tri_mat_warp(tile, aux_h->matrix, dim_g, 0); // set zero
    set_tri_mat_diagonal_warp(tile, aux_h->matrix, dim_g, 1); // set diagonal to 1

    // compute line search direction aux_p = -Hg (dim*1 vector)
    minus_mat_vec_product_warp(tile, p, aux_h->matrix, g, dim_g);

    // run the target device function
    line_search_warp(tile, *fix_mol, *fix_param,
        *flex_param, x,
        g, *flex_topo, p,
        aux_f->f,
        out_x_new, g_new, aux_v + BV_Y,
        e0, dim_g, dim_x,
        out_e, out_alpha);
    flat_to_grad(tile, g, aux_g, dim_g);
    flat_to_grad(tile, p, aux_p, dim_g);
    flat_to_grad(tile, g_new, out_g_new, dim_g);
}


__global__ void cal_e_grad_one_kernel(const FixMol* fix_mol, const FixParamVina* fix_param,
        const FlexParamVina* flex_param, const FlexPose* x, const FlexTopo* flex_topo,
        FlexPoseGradient* out_g, FlexForce* aux_f, Real* out_e, Real* aux_v) {

    auto tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
    const int dim_g = dof_g + flex_topo->ntorsion;
    grad_to_flat(tile, out_g, aux_v, dim_g);
    Real e = cal_e_grad_warp(tile, x, aux_v, *flex_topo, *fix_mol, *flex_param, *fix_param, aux_f->f);
    flat_to_grad(tile, aux_v, out_g, dim_g);
    if (tile.thread_rank() == 0) *out_e = e;
}


void cal_e_grad_one_gpu(const FixMol& fix_mol, const FixParamVina& fix_param,
        const FlexParamVina& flex_param, const FlexPose& x, const FlexTopo& flex_topo,
        FlexPoseGradient* out_g, Real* out_e) {

    // Allocate memory on GPU
    set_kernel_max_natom(flex_topo.natom);
    FixMol* fix_mol_cu = alloccp_FixMol_gpu(fix_mol);
    FixParamVina* fix_param_cu = alloccp_FixParamVina_gpu(fix_param, fix_mol.natom);
    FlexParamVina* flex_param_cu = alloccp_FlexParamVina_gpu(flex_param, flex_topo.natom);
    FlexPose* x_cu = alloccp_FlexPose_gpu(x, flex_topo.natom, flex_topo.ntorsion);
    FlexTopo* flex_topo_cu = alloccp_FlexTopo_gpu(flex_topo);
    FlexPoseGradient* out_g_cu = alloccp_FlexPoseGradient_gpu(*out_g, flex_topo.ntorsion);
    FlexForce* aux_f_cu = alloc_FlexForce_gpu(fix_mol.natom);
    Real* aux_v_cu = alloc_flat_vectors_gpu(flex_topo.ntorsion);
    Real* out_e_cu;
    checkCUDA(cudaMalloc(&out_e_cu, sizeof(Real)));

    // Run the kernel
    cal_e_grad_one_kernel<<<1, TILE_SIZE, kernel_dyn_shared_bytes()>>>(fix_mol_cu, fix_param_cu, flex_param_cu, x_cu, flex_topo_cu,
        out_g_cu, aux_f_cu, out_e_cu, aux_v_cu);
    cudaDeviceSynchronize();
    checkCUDA(cudaFree(aux_v_cu));

    // Copy the results back to CPU
    checkCUDA(cudaMemcpy(out_e, out_e_cu, sizeof(Real), cudaMemcpyDeviceToHost));
    freecp_FlexPoseGradient_gpu(out_g_cu, out_g, flex_topo.ntorsion);

    // Free on GPU
    free_FixMol_gpu(fix_mol_cu);
    free_FixParamVina_gpu(fix_param_cu);
    free_FlexParamVina_gpu(flex_param_cu);
    free_FlexPose_gpu(x_cu);
    free_FlexTopo_gpu(flex_topo_cu);
    free_FlexForce_gpu(aux_f_cu);
    checkCUDA(cudaFree(out_e_cu));

}


/**
 * @brief Wrapper of line search on GPU, only perform on one pose using one warp/block.
 */
void line_search_one_gpu(const FixMol& fix_mol, const FixParamVina& fix_param,
                     const FlexParamVina& flex_param, const FlexPose& x,
                     const FlexTopo& flex_topo,
                     FlexPose* out_x_new, FlexPoseGradient* out_g_new,
                     Real* out_e, Real* out_alpha) {

    // Allocate memory on GPU
    set_kernel_max_natom(flex_topo.natom);
    FixMol* fix_mol_cu = alloccp_FixMol_gpu(fix_mol);
    FixParamVina* fix_param_cu = alloccp_FixParamVina_gpu(fix_param, fix_mol.natom);
    FlexParamVina* flex_param_cu = alloccp_FlexParamVina_gpu(flex_param, flex_topo.natom);
    FlexPose* x_cu = alloccp_FlexPose_gpu(x, fix_mol.natom, flex_topo.ntorsion);
    FlexTopo* flex_topo_cu = alloccp_FlexTopo_gpu(flex_topo);
    FlexPoseGradient* aux_g_cu = alloccp_FlexPoseGradient_gpu(*out_g_new, flex_topo.ntorsion);
    FlexPoseGradient* aux_p_cu = alloccp_FlexPoseGradient_gpu(*out_g_new, flex_topo.ntorsion);
    FlexPoseHessian* aux_h_cu = alloc_FlexPoseHessian_gpu(flex_topo.ntorsion);
    FlexForce* aux_f_cu = alloc_FlexForce_gpu(fix_mol.natom );
    FlexPose* out_x_new_cu = alloccp_FlexPose_gpu(*out_x_new, flex_topo.natom, flex_topo.ntorsion);

    FlexPoseGradient* out_g_new_cu = alloccp_FlexPoseGradient_gpu(*out_g_new, flex_topo.ntorsion);
    Real* out_e_cu;
    checkCUDA(cudaMalloc(&out_e_cu, sizeof(Real)));
    Real* out_alpha_cu;
    checkCUDA(cudaMalloc(&out_alpha_cu, sizeof(Real)));
    Real* aux_v_cu = alloc_flat_vectors_gpu(flex_topo.ntorsion);


    // Run the kernel
    line_search_one_original_kernel<<<1, TILE_SIZE, kernel_dyn_shared_bytes()>>>(fix_mol_cu, fix_param_cu,
        flex_param_cu, x_cu, flex_topo_cu,
        aux_g_cu, aux_p_cu, aux_h_cu, aux_f_cu,
        out_x_new_cu, out_g_new_cu,
        out_e_cu, out_alpha_cu, aux_v_cu);
    cudaDeviceSynchronize();
    checkCUDA(cudaFree(aux_v_cu));

    // Copy the results back to CPU
    freecp_FlexPose_gpu(out_x_new_cu, out_x_new, flex_topo.natom, flex_topo.ntorsion);
    freecp_FlexPoseGradient_gpu(out_g_new_cu, out_g_new, flex_topo.ntorsion);

    checkCUDA(cudaMemcpy(out_e, out_e_cu, sizeof(Real), cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(out_alpha, out_alpha_cu, sizeof(Real), cudaMemcpyDeviceToHost));

    // Free on GPU
    free_FixMol_gpu(fix_mol_cu);
    free_FixParamVina_gpu(fix_param_cu);
    free_FlexParamVina_gpu(flex_param_cu);
    free_FlexPose_gpu(x_cu);
    free_FlexTopo_gpu(flex_topo_cu);

    checkCUDA(cudaFree(out_e_cu));
    checkCUDA(cudaFree(out_alpha_cu));
    
    free_FlexPoseGradient_gpu(aux_g_cu);
    free_FlexPoseGradient_gpu(aux_p_cu);
    free_FlexPoseHessian_gpu(aux_h_cu);
    free_FlexForce_gpu(aux_f_cu);
}

__global__ void bfgs_one_kernel(const FixMol* fix_mol, const FixParamVina* fix_param,
        const FlexParamVina* flex_param, FlexPose* x, const FlexTopo* flex_topo,
        FlexPose* aux_x_new, FlexPose* aux_x_ori,
        Real* aux_v, FlexPoseHessian* aux_h, FlexForce* aux_f, int max_steps) {
    auto tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
    bfgs_warp(tile, x, *flex_topo, *fix_mol, *flex_param, *fix_param,
              aux_x_new, aux_x_ori, aux_v, aux_h->matrix, aux_f, max_steps);
}


void bfgs_one_gpu(const FixMol& fix_mol, const FixParamVina& fix_param,
                  const FlexParamVina& flex_param, FlexPose* x, const FlexTopo& flex_topo,
                  int max_steps) {
    set_kernel_max_natom(flex_topo.natom);
    FixMol* fix_mol_cu = alloccp_FixMol_gpu(fix_mol);
    FixParamVina* fix_param_cu = alloccp_FixParamVina_gpu(fix_param, fix_mol.natom);
    FlexParamVina* flex_param_cu = alloccp_FlexParamVina_gpu(flex_param, flex_topo.natom);
    FlexTopo* flex_topo_cu = alloccp_FlexTopo_gpu(flex_topo);
    FlexPose* x_cu = alloccp_FlexPose_gpu(*x, flex_topo.natom, flex_topo.ntorsion);
    FlexPose* aux_x_new_cu = alloccp_FlexPose_gpu(*x, flex_topo.natom, flex_topo.ntorsion);
    FlexPose* aux_x_ori_cu = alloccp_FlexPose_gpu(*x, flex_topo.natom, flex_topo.ntorsion);
    Real* aux_v_cu = alloc_flat_vectors_gpu(flex_topo.ntorsion);
    FlexPoseHessian* aux_h_cu = alloc_FlexPoseHessian_gpu(flex_topo.ntorsion);
    FlexForce* aux_f_cu = alloc_FlexForce_gpu(flex_topo.natom);

    bfgs_one_kernel<<<1, TILE_SIZE, kernel_dyn_shared_bytes()>>>(fix_mol_cu, fix_param_cu, flex_param_cu, x_cu, flex_topo_cu,
                                      aux_x_new_cu, aux_x_ori_cu, aux_v_cu,
                                      aux_h_cu, aux_f_cu, max_steps);
    checkCUDA(cudaDeviceSynchronize());

    freecp_FlexPose_gpu(x_cu, x, flex_topo.natom, flex_topo.ntorsion);
    free_FlexPose_gpu(aux_x_new_cu); free_FlexPose_gpu(aux_x_ori_cu);
    checkCUDA(cudaFree(aux_v_cu));
    free_FlexPoseHessian_gpu(aux_h_cu); free_FlexForce_gpu(aux_f_cu);
    free_FixMol_gpu(fix_mol_cu); free_FixParamVina_gpu(fix_param_cu); free_FlexParamVina_gpu(flex_param_cu); free_FlexTopo_gpu(flex_topo_cu);
}


// ---------------- GRID path test wrappers (test/unit/cuda/test_grid_maps.cpp) ----------------
template<bool GRID>
__global__ void cal_e_grad_one_cap_kernel(const FixMol* fix_mol, const FixParamVina* fix_param,
        const FlexParamVina* flex_param, const FlexPose* x, const FlexTopo* flex_topo,
        FlexPoseGradient* out_g, FlexForce* aux_f, Real* out_e, Real v_cap, Real* aux_v) {
    auto tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
    const int dim_g = dof_g + flex_topo->ntorsion;
    grad_to_flat(tile, out_g, aux_v, dim_g);
    Real e = cal_e_grad_warp<GRID>(tile, x, aux_v, *flex_topo, *fix_mol, *flex_param, *fix_param, aux_f->f, v_cap);
    flat_to_grad(tile, aux_v, out_g, dim_g);
    if (tile.thread_rank() == 0) *out_e = e;
}



Real curl_v_cap(bool authentic){
    return authentic ? CURL_V_AUTHENTIC_CU : CURL_V_CU;
}


void cal_e_grad_one_gpu_cap(const FixMol& fix_mol, const FixParamVina& fix_param,
        const FlexParamVina& flex_param, const FlexPose& x, const FlexTopo& flex_topo,
        FlexPoseGradient* out_g, Real* out_e, bool grid, bool authentic) {
    const Real v_cap = curl_v_cap(authentic);
    set_kernel_max_natom(flex_topo.natom);
    FixMol* fix_mol_cu = alloccp_FixMol_gpu(fix_mol);
    FixParamVina* fix_param_cu = alloccp_FixParamVina_gpu(fix_param, fix_mol.natom);
    FlexParamVina* flex_param_cu = alloccp_FlexParamVina_gpu(flex_param, flex_topo.natom);
    FlexPose* x_cu = alloccp_FlexPose_gpu(x, flex_topo.natom, flex_topo.ntorsion);
    FlexTopo* flex_topo_cu = alloccp_FlexTopo_gpu(flex_topo);
    FlexPoseGradient* out_g_cu = alloccp_FlexPoseGradient_gpu(*out_g, flex_topo.ntorsion);
    FlexForce* aux_f_cu = alloc_FlexForce_gpu(fix_mol.natom);
    Real* aux_v_cu = alloc_flat_vectors_gpu(flex_topo.ntorsion);
    Real* out_e_cu;
    checkCUDA(cudaMalloc(&out_e_cu, sizeof(Real)));

    if (grid){
        cal_e_grad_one_cap_kernel<true><<<1, TILE_SIZE, kernel_dyn_shared_bytes()>>>(fix_mol_cu, fix_param_cu, flex_param_cu,
            x_cu, flex_topo_cu, out_g_cu, aux_f_cu, out_e_cu, v_cap, aux_v_cu);
    } else {
        cal_e_grad_one_cap_kernel<false><<<1, TILE_SIZE, kernel_dyn_shared_bytes()>>>(fix_mol_cu, fix_param_cu, flex_param_cu,
            x_cu, flex_topo_cu, out_g_cu, aux_f_cu, out_e_cu, v_cap, aux_v_cu);
    }
    checkCUDA(cudaGetLastError());
    checkCUDA(cudaDeviceSynchronize());
    checkCUDA(cudaFree(aux_v_cu));

    checkCUDA(cudaMemcpy(out_e, out_e_cu, sizeof(Real), cudaMemcpyDeviceToHost));
    freecp_FlexPoseGradient_gpu(out_g_cu, out_g, flex_topo.ntorsion);

    free_FixMol_gpu(fix_mol_cu);
    free_FixParamVina_gpu(fix_param_cu);
    free_FlexParamVina_gpu(flex_param_cu);
    free_FlexPose_gpu(x_cu);
    free_FlexTopo_gpu(flex_topo_cu);
    free_FlexForce_gpu(aux_f_cu);
    checkCUDA(cudaFree(out_e_cu));
}


void get_receptor_maps(RecMaps& out){
    checkCUDA(cudaMemcpyFromSymbol(&out, g_maps, sizeof(RecMaps)));
}



void get_polar_grid(SpatialGrid& out, std::vector<int>& cell_start, std::vector<GridAtom>& atoms){
    checkCUDA(cudaMemcpyFromSymbol(&out, g_pgrid, sizeof(SpatialGrid)));
    cell_start.clear(); atoms.clear();
    if (!out.built) return;
    cell_start.resize(out.ncells + 1);
    checkCUDA(cudaMemcpy(cell_start.data(), out.cell_start, (out.ncells + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    atoms.resize(out.total_atoms);
    if (out.total_atoms > 0) checkCUDA(cudaMemcpy(atoms.data(), out.atoms, out.total_atoms * sizeof(GridAtom), cudaMemcpyDeviceToHost));
}


void get_polar_cands(PolarCands& out, std::vector<int>& cell_start, std::vector<GridAtom>& atoms){
    checkCUDA(cudaMemcpyFromSymbol(&out, g_pcand, sizeof(PolarCands)));
    cell_start.clear(); atoms.clear();
    if (!out.cell_start) return;
    const int ncells = out.nx * out.ny * out.nz;
    cell_start.resize(ncells + 1);
    checkCUDA(cudaMemcpy(cell_start.data(), out.cell_start, (ncells + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    atoms.resize(cell_start[ncells]);
    if (!atoms.empty()) checkCUDA(cudaMemcpy(atoms.data(), out.atoms, atoms.size() * sizeof(GridAtom), cudaMemcpyDeviceToHost));
}


void set_polar_cands_enabled(bool enabled){
    PolarCands pc = {};
    checkCUDA(cudaMemcpyFromSymbol(&pc, g_pcand, sizeof(PolarCands)));
    if (!pc.cell_start) return;   // nothing uploaded
    pc.built = enabled;
    checkCUDA(cudaMemcpyToSymbol(g_pcand, &pc, sizeof(PolarCands)));
}
