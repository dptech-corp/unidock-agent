//
// CPU Optimization Wrapper
// Created for Uni-Dock2 CPU backend
//

#ifndef OPTIMIZE_CPU_H
#define OPTIMIZE_CPU_H

#include "common_cpu.h"
#include "bfgs_cpu.h"
#include "model/model.h"

/**
 * @brief Run refinement optimization on clustered poses
 * CPU version of optimize_cu
 * 
 * @param poses Array of poses
 * @param pose_inds Indices of poses to optimize
 * @param flex_topos Array of flex topologies
 * @param fix_mol Fixed molecule
 * @param flex_params Array of flex parameters
 * @param fix_param Fix parameters
 * @param refine_steps Number of refinement steps
 * @param npose Number of poses to optimize
 * @param exhaustiveness Original exhaustiveness (for indexing)
 */
void optimize_cpu(FlexPose* poses, const int* pose_inds,
                  const FlexTopo* flex_topos, const FixMol& fix_mol,
                  const FlexParamVina* flex_params, const FixParamVina& fix_param,
                  int refine_steps, int npose, int exhaustiveness);


#endif // OPTIMIZE_CPU_H
