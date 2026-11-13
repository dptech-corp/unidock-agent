//
// Created by Congcong Liu on 24-11-21.
//

#include <cooperative_groups.h>
#include "optimize/optimize.h"
#include "model/model.h"
#include "common.cuh"
#include "bfgs.cuh"

namespace cg = cooperative_groups;


template<bool TORLIB>
__global__ void opt_kernel(FlexPose* out_poses, const int* pose_inds, const FlexTopo* flex_topos, const FixMol& fix_mol,
                          const FlexParamVina* flex_params, const FixParamVina& fix_param,
                          FlexPose* aux_poses, FlexPoseHessian* aux_hessians,
                          FlexForce* aux_forces,
                          int refine_steps, int npose_per_flex, int state_flags){

    // nflex, each flex has num_pose poses to optimize, each pose uses a warp

    int id_pose = pose_inds[blockIdx.x];

    int id_flex = id_pose / npose_per_flex;

    // Use alias
    FlexPose& out_pose = out_poses[id_pose]; // pointer to global data
    const FlexTopo& flex_topo = flex_topos[id_flex];
    const FlexParamVina& flex_param = flex_params[id_flex];

    FlexPose& aux_pose_ori = aux_poses[id_pose * STRIDE_POSE + 2];
    FlexPose& aux_pose_new = aux_poses[id_pose * STRIDE_POSE + 3];

    FlexPoseHessian& aux_h = aux_hessians[id_pose];

    FlexForce& aux_f = aux_forces[id_pose];


    // Cooperative group
    auto cta = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(cta);

    Real* bfgs_h = nullptr;
    Real* bfgs_v = nullptr;
    bfgs_state_ptrs(tile, aux_h.matrix, flex_topo.natom, flex_topo.ntorsion, state_flags, &bfgs_h, &bfgs_v);
    bfgs_warp<false, false, TORLIB>(tile,
               &out_pose, flex_topo, fix_mol, flex_param, fix_param,
               &aux_pose_new, &aux_pose_ori,
               bfgs_v, bfgs_h, &aux_f, refine_steps, CURL_V_AUTHENTIC_CU);
    tile.sync();
}



void optimize_cu(FlexPose* out_poses, const int* pose_inds, const FlexTopo* flex_topos, const FixMol& fix_mol,
                          const FlexParamVina* flex_params, const FixParamVina& fix_param,
                          FlexPose* aux_poses, FlexPoseHessian* aux_hessians,
                          FlexForce* aux_forces,
                          int refine_steps, int nblock, int npose_per_flex, bool torlib){

    // todo: quasi_newton_par.max_steps = unsigned((25 + m_model_gpu[l].num_movable_atoms()) / 3);
    // todo: through refinement, poses outside the box can be purged into the box????
    Real slope_for_refine = 100;
    checkCUDA(cudaMemcpyToSymbol(PENALTY_SLOPE, &slope_for_refine, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaDeviceSynchronize());

    const void* kernel = torlib ? (const void*)opt_kernel<true> : (const void*)opt_kernel<false>;
    const int state_flags = bfgs_state_flags(kernel, "opt_kernel", nblock, TILE_SIZE);
    const size_t dyn = kernel_state_dyn_shared_bytes(state_flags);
    set_carveout(kernel, "opt_kernel", "UD2_CARVE_OPT", "occ32", nblock, TILE_SIZE, dyn);
    if (torlib){
        opt_kernel<true><<<nblock, TILE_SIZE, dyn>>>(out_poses, pose_inds, flex_topos, fix_mol,
                          flex_params, fix_param,
                          aux_poses, aux_hessians,
                          aux_forces, refine_steps, npose_per_flex, state_flags);
    } else {
        opt_kernel<false><<<nblock, TILE_SIZE, dyn>>>(out_poses, pose_inds, flex_topos, fix_mol,
                          flex_params, fix_param,
                          aux_poses, aux_hessians,
                          aux_forces, refine_steps, npose_per_flex, state_flags);
    }
    checkCUDA(cudaDeviceSynchronize());

}


// Analytic energy of one search output (GRID path): the chains were minimised on the receptor maps,
// so every stored energy is re-evaluated with cal_e_f_warp<false> at CURL_V_AUTHENTIC_CU before the
// clustering orders the poses. PENALTY_SLOPE is left as the search had it. One warp per pose.
__global__ void rescore_kernel(FlexPose* out_poses, const FlexTopo* flex_topos, const FixMol& fix_mol,
                               const FlexParamVina* flex_params, const FixParamVina& fix_param,
                               FlexForce* aux_forces, int npose_per_flex){
    const int id_pose = blockIdx.x;
    const int id_flex = id_pose / npose_per_flex;
    FlexPose& out_pose = out_poses[id_pose];

    auto cta = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(cta);

    const Real energy = cal_e_f_warp<false>(tile, &out_pose, flex_topos[id_flex], fix_mol, flex_params[id_flex], fix_param,
                                            aux_forces[id_pose].f, CURL_V_AUTHENTIC_CU);
    if (tile.thread_rank() == 0){
        out_pose.energy = energy;
    }
    tile.sync();
}


void rescore_cu(FlexPose* out_poses, const FlexTopo* flex_topos, const FixMol& fix_mol,
                const FlexParamVina* flex_params, const FixParamVina& fix_param,
                FlexForce* aux_forces, int nflex, int npose_per_flex){
    const int npose = nflex * npose_per_flex;
    set_carveout((const void*)rescore_kernel, "rescore_kernel", "UD2_CARVE_RESCORE", "occ32", npose, TILE_SIZE, kernel_dyn_shared_bytes());
    rescore_kernel<<<npose, TILE_SIZE, kernel_dyn_shared_bytes()>>>(out_poses, flex_topos, fix_mol,
                          flex_params, fix_param, aux_forces, npose_per_flex);
    checkCUDA(cudaDeviceSynchronize());
}
