//
// Created by lccdp on 24-8-15.
//

#ifndef UD2_MC_H
#define UD2_MC_H

#include "model/model.h"
#include "score/vina.h"
#if USE_CUDA
#include <curand_kernel.h>
#endif

SCOPE_INLINE bool metropolis_accept(Real e_old, Real e_new, Real temperature, Real rand_x) {
    if (e_new < e_old){
        return true;
    }
    Real accept_p = exp((e_old - e_new) / temperature);
    return rand_x < accept_p;
}


/**
 * @brief Use Monte Carlo to search for the best pose, starting from out_pose.
 * use_grid: evaluate on the receptor maps (mc_kernel<true>; g_maps must be built) instead of the analytic pairs.
 * rng_slots: device array, one stream slot per ligand (Advanced.rng_period); nullptr = the ligand's index in this launch.
 * torlib: some torsion of the launch has torsion-library ranges (not the full circle): run the kernel variant with the
 * range-aware search (bfgs.cuh TORLIB); false runs the one without it.
 */
#if USE_CUDA
void mc_cu(FlexPose* out_poses, const FlexTopo* topos,
           const FixMol& fix_mol, const FlexParamVina* flex_param, const FixParamVina& fix_param,
           FlexPose* aux_poses, FlexPoseHessian* aux_hessians, FlexForce* aux_forces,
           int mc_steps, int opt_steps, int nflex, int exhuastiveness, int seed, bool randomize, bool use_grid,
           const int* rng_slots, bool torlib);
#endif




#endif //UD2_MC_H
