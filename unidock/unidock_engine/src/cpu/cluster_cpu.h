//
// CPU Clustering
// Created for Uni-Dock2 CPU backend
//

#ifndef CLUSTER_CPU_H
#define CLUSTER_CPU_H

#include "common_cpu.h"
#include "model/model.h"
#include <vector>

/**
 * @brief Calculate RMSD between two poses using specified atom indices
 * @param atom_inds Array of atom indices to include in RMSD
 * @param n_atoms Number of indices in atom_inds
 */
Real cal_rmsd_pair_cpu(const FlexPose* pose1, const FlexPose* pose2,
                       const int* atom_inds, int n_atoms);

/**
 * @brief Cluster poses by RMSD and energy
 * CPU version of cluster_cu
 * 
 * @param clustered_pose_inds_out Output array for clustered pose indices
 * @param npose_clustered_out Output number of clustered poses
 * @param clustered_pose_inds_list Output list of clustered indices per flex
 * @param poses Array of poses
 * @param flex_topos Array of flex topologies
 * @param nflex Number of flex molecules
 * @param exhaustiveness Number of poses per flex
 * @param rmsd_limit RMSD threshold for clustering
 * @param cluster_atom_indices_per_flex Per-ligand atom indices for RMSD.
 *        Empty outer = use heavy atoms for all. Empty inner = heavy atoms for that ligand.
 */
void cluster_cpu(int* clustered_pose_inds_out, int* npose_clustered_out,
                 std::vector<std::vector<int>>* clustered_pose_inds_list,
                 const FlexPose* poses, const FlexTopo* flex_topos,
                 int nflex, int exhaustiveness, Real rmsd_limit,
                 const std::vector<std::vector<int>>& cluster_atom_indices_per_flex);


#endif // CLUSTER_CPU_H
