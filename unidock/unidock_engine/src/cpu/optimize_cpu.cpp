//
// CPU Optimization Wrapper Implementation
// Created for Uni-Dock2 CPU backend
//

#include "optimize_cpu.h"
#include <spdlog/spdlog.h>

#if USE_OPENMP
#include <omp.h>
#endif


void optimize_cpu(FlexPose* poses, const int* pose_inds,
                  const FlexTopo* flex_topos, const FixMol& fix_mol,
                  const FlexParamVina* flex_params, const FixParamVina& fix_param,
                  int refine_steps, int npose, int exhaustiveness) {
    
    spdlog::info("CPU Optimization: {} poses, {} steps", npose, refine_steps);
    
#if USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < npose; i++) {
        int global_idx = pose_inds[i];
        int i_flex = global_idx / exhaustiveness;
        
        FlexPose* pose = &poses[global_idx];
        const FlexTopo& topo = flex_topos[i_flex];
        const FlexParamVina& param = flex_params[i_flex];
        
        bfgs_simple_cpu(pose, topo, fix_mol, param, fix_param, refine_steps);
    }
    
    spdlog::info("CPU Optimization completed");
}
