//
// Created by Congcong Liu on 24-12-9.
//

#ifndef BFGS_H
#define BFGS_H

#include "model/model.h"
#include "score/vina.h"
#include <vector>

void line_search_one_gpu(const FixMol& fix_mol, const FixParamVina& fix_param,
                     const FlexParamVina& flex_param, const FlexPose& x,
                     const FlexTopo& flex_topo,
                     FlexPose* out_x_new, FlexPoseGradient* out_g_new,
                     Real* out_e, Real* out_alpha);


void cal_e_grad_one_gpu(const FixMol& fix_mol, const FixParamVina& fix_param,
        const FlexParamVina& flex_param, const FlexPose& x, const FlexTopo& flex_topo,
        FlexPoseGradient* out_g, Real* out_e);

/**
 * @brief Run the full BFGS of one pose on the GPU (one warp) and copy the optimized pose back.
 * The pose is updated in place (coords, dihedrals, center, rot_vec, energy).
 */
void bfgs_one_gpu(const FixMol& fix_mol, const FixParamVina& fix_param,
                  const FlexParamVina& flex_param, FlexPose* x, const FlexTopo& flex_topo,
                  int max_steps);


// ---- GRID path test wrappers (bfgs_wrapper.cu; test/unit/cuda/test_grid_maps.cpp) ----
struct RecMaps;
/**
 * @brief cal_e_grad_warp<grid> of one pose on one warp at the hunt cap (CURL_V_CU) or the authentic cap
 * (CURL_V_AUTHENTIC_CU). grid = true needs g_maps / g_pgrid built (build_receptor_maps).
 */
void cal_e_grad_one_gpu_cap(const FixMol& fix_mol, const FixParamVina& fix_param,
        const FlexParamVina& flex_param, const FlexPose& x, const FlexTopo& flex_topo,
        FlexPoseGradient* out_g, Real* out_e, bool grid, bool authentic);
/** @brief Host copy of the g_maps header (data stays the device buffer). */
void get_receptor_maps(RecMaps& out);
struct PolarCands; struct SpatialGrid; struct GridAtom;
/** @brief Host copies of g_pgrid (header, CSR cell_start, atoms); empty vectors when not built. */
void get_polar_grid(SpatialGrid& out, std::vector<int>& cell_start, std::vector<GridAtom>& atoms);
/** @brief Host copies of g_pcand (header, CSR cell_start, candidate atoms); empty vectors when not uploaded. */
void get_polar_cands(PolarCands& out, std::vector<int>& cell_start, std::vector<GridAtom>& atoms);
/** @brief Flip g_pcand.built so the GRID polar pass uses the candidate lists (true) or walks g_pgrid (false). */
void set_polar_cands_enabled(bool enabled);
/** @brief The curl cap the kernels use: CURL_V_AUTHENTIC_CU when authentic, else CURL_V_CU. */
Real curl_v_cap(bool authentic);


#endif //BFGS_H
