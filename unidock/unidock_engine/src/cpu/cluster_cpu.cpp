//
// CPU Clustering Implementation
// Created for Uni-Dock2 CPU backend
//

#include "cluster_cpu.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <spdlog/spdlog.h>

#if USE_OPENMP
#include <omp.h>
#endif


Real cal_rmsd_pair_cpu(const FlexPose* pose1, const FlexPose* pose2,
                       const int* atom_inds, int n_atoms) {
    Real rmsd_sq = 0;
    
    for (int k = 0; k < n_atoms; k++) {
        int i = atom_inds[k];
        Real dx = pose1->coords[i * 3]     - pose2->coords[i * 3];
        Real dy = pose1->coords[i * 3 + 1] - pose2->coords[i * 3 + 1];
        Real dz = pose1->coords[i * 3 + 2] - pose2->coords[i * 3 + 2];
        rmsd_sq += dx*dx + dy*dy + dz*dz;
    }
    
    return n_atoms > 0 ? std::sqrt(rmsd_sq / n_atoms) : 0;
}


void cluster_cpu(int* clustered_pose_inds_out, int* npose_clustered_out,
                 std::vector<std::vector<int>>* clustered_pose_inds_list,
                 const FlexPose* poses, const FlexTopo* flex_topos,
                 int nflex, int exhaustiveness, Real rmsd_limit,
                 const std::vector<std::vector<int>>& cluster_atom_indices_per_flex) {
    
    spdlog::info("CPU Clustering: nflex={}, exhaustiveness={}, rmsd_limit={}",
                 nflex, exhaustiveness, rmsd_limit);
    
    clustered_pose_inds_list->clear();
    clustered_pose_inds_list->resize(nflex);
    
    int total_clustered = 0;
    
    for (int i_flex = 0; i_flex < nflex; i_flex++) {
        const FlexTopo& topo = flex_topos[i_flex];
        int natom = topo.natom;
        int base_idx = i_flex * exhaustiveness;
        
        // Determine atom indices for RMSD calculation
        std::vector<int> rmsd_atom_inds;
        bool has_custom = !cluster_atom_indices_per_flex.empty()
                          && i_flex < (int)cluster_atom_indices_per_flex.size()
                          && !cluster_atom_indices_per_flex[i_flex].empty();
        if (has_custom) {
            rmsd_atom_inds = cluster_atom_indices_per_flex[i_flex];
        } else {
            // Default: heavy atoms only
            for (int i = 0; i < natom; i++) {
                if (!g_cpu_constants.atp.is_hydrogen[topo.vn_types[i]]) {
                    rmsd_atom_inds.push_back(i);
                }
            }
        }
        const int* atom_inds_ptr = rmsd_atom_inds.data();
        int n_rmsd_atoms = (int)rmsd_atom_inds.size();
        
        spdlog::debug("Flex {}: using {} atoms for clustering RMSD ({})",
                      i_flex, n_rmsd_atoms, has_custom ? "custom" : "heavy-atom default");
        
        // Get indices sorted by energy
        std::vector<int> sorted_indices(exhaustiveness);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        
        std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&poses, base_idx](int a, int b) {
                return poses[base_idx + a].energy < poses[base_idx + b].energy;
            });
        
        // Greedy clustering. Use vector<char> (not vector<bool>): the deactivation
        // loop below writes active[j] from an OpenMP parallel-for, and vector<bool>
        // bit-packing makes concurrent writes to neighboring indices a data race.
        std::vector<char> active(exhaustiveness, 1);
        std::vector<int>& selected = (*clustered_pose_inds_list)[i_flex];
        
        for (int idx : sorted_indices) {
            if (!active[idx]) continue;
            
            // Add this pose to cluster
            int global_idx = base_idx + idx;
            selected.push_back(global_idx);
            clustered_pose_inds_out[total_clustered++] = global_idx;
            
            // Deactivate similar poses
#if USE_OPENMP
            #pragma omp parallel for
#endif
            for (int j = 0; j < exhaustiveness; j++) {
                if (active[j] && j != idx) {
                    Real rmsd = cal_rmsd_pair_cpu(&poses[base_idx + idx],
                                                  &poses[base_idx + j],
                                                  atom_inds_ptr, n_rmsd_atoms);
                    if (rmsd < rmsd_limit) {
                        active[j] = false;
                    }
                }
            }
        }
        
        spdlog::debug("Flex {}: {} poses clustered to {}", i_flex, exhaustiveness, selected.size());
    }
    
    *npose_clustered_out = total_clustered;
    spdlog::info("CPU Clustering completed: {} total clustered poses", total_clustered);
}
