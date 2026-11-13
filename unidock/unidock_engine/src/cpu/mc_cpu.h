//
// CPU Monte Carlo Search
// Created for Uni-Dock2 CPU backend
//

#ifndef MC_CPU_H
#define MC_CPU_H

#include "common_cpu.h"
#include "random_cpu.h"
#include "score_cpu.h"
#include "bfgs_cpu.h"
#include "model/model.h"

// Thread-local memory structure to avoid allocations in hot loop
struct MCThreadMemory {
    std::vector<Real> aux_coords;
    std::vector<Real> aux_dihedrals;
    std::vector<Real> forces;
    std::vector<Real> current_coords;
    std::vector<Real> current_dihedrals;
    std::vector<Real> best_coords;      // For best-pose tracking
    std::vector<Real> best_dihedrals;   // For best-pose tracking
    BFGSThreadMemory bfgs_mem;  // Pre-allocated BFGS memory

    void resize(int natom, int ntorsion) {
        aux_coords.resize(natom * 3);
        aux_dihedrals.resize(ntorsion);
        forces.resize(natom * 3);
        current_coords.resize(natom * 3);
        current_dihedrals.resize(ntorsion);
        best_coords.resize(natom * 3);
        best_dihedrals.resize(ntorsion);
        bfgs_mem.resize(natom, ntorsion);
    }
};

/**
 * @brief Randomize pose to a random position/orientation within the box
 * CPU version of randomize_pose_warp
 */
void randomize_pose_cpu(FlexPose* out_pose, const FlexPose* pose_old,
                        const FlexTopo& flex_topo,
                        CPURandom& rng, int pose_id, int total_poses);

/**
 * @brief Mutate pose by random perturbation on one DOF
 * CPU version of mutate_pose_warp
 * Directly modifies coordinates (translation/rotation) or calls apply_grad_update_dihe_cpu (torsion)
 */
void mutate_pose_cpu(FlexPose* out_pose, const FlexTopo* flex_topo,
                     CPURandom& rng, Real amplitude = 1.0);

/**
 * @brief Run Monte Carlo search for a single pose
 * CPU version of one iteration of mc_kernel
 * @param mem Pre-allocated thread-local memory for performance
 */
void mc_single_pose_cpu(FlexPose* out_pose, const FlexTopo& flex_topo,
                        const FixMol& fix_mol, const FlexParamVina& flex_param,
                        const FixParamVina& fix_param,
                        CPURandom& rng,
                        int mc_steps, int opt_steps, bool randomize,
                        int pose_id, int total_poses,
                        MCThreadMemory& mem);

/**
 * @brief Run Monte Carlo search for all poses (parallelized)
 * CPU version of mc_cu
 * 
 * @param out_poses Array of poses (size: nflex * exhaustiveness)
 * @param flex_topos Array of flex topologies (size: nflex)
 * @param fix_mol Fixed molecule
 * @param flex_params Array of flex parameters (size: nflex)
 * @param fix_param Fix parameters
 * @param mc_steps Number of MC steps
 * @param opt_steps Number of optimization steps per MC step (-1 for heuristic)
 * @param nflex Number of flex molecules
 * @param exhaustiveness Number of poses per flex
 * @param seed Random seed
 * @param randomize Whether to randomize initial poses
 */
void mc_cpu(FlexPose* out_poses, const FlexTopo* flex_topos,
            const FixMol& fix_mol, const FlexParamVina* flex_params,
            const FixParamVina& fix_param,
            int mc_steps, int opt_steps, int nflex, int exhaustiveness,
            int seed, bool randomize);


#endif // MC_CPU_H
