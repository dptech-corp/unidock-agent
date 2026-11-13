//
// Created by Congcong Liu on 24-11-21.
//

#ifndef OPTIMIZE_H
#define OPTIMIZE_H

#include "model/model.h"
#include "score/vina.h"

/**
 * @brief BFGS refinement of the nblock poses pose_inds; torlib as in mc_cu (search/mc.h)
 */
void optimize_cu(FlexPose* out_poses, const int* pose_inds, const FlexTopo* flex_topos, const FixMol& fix_mol,
                          const FlexParamVina* flex_params, const FixParamVina& fix_param,
                          FlexPose* aux_poses, FlexPoseHessian* aux_hessians,
                          FlexForce* aux_forces,
                          int refine_steps, int nblock, int npose_per_flex, bool torlib);

/**
 * @brief Analytic energy (CURL_V_AUTHENTIC_CU) of all nflex * npose_per_flex search outputs, written to
 * out_pose.energy. Used after an MC search on the receptor maps (GRID path), before clustering.
 */
void rescore_cu(FlexPose* out_poses, const FlexTopo* flex_topos, const FixMol& fix_mol,
                const FlexParamVina* flex_params, const FixParamVina& fix_param,
                FlexForce* aux_forces, int nflex, int npose_per_flex);


#endif //OPTIMIZE_H
