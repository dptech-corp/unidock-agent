//
// Created by Congcong Liu on 24-9-23.
//

#ifndef DOCKTASK_H
#define DOCKTASK_H


#include "model/model.h"
#include <string>
#include "score/vina.h"
#include "format/dms_reader.h"
#include "score/gaff2.h"


/**
 * @brief Docking calculation model. Aiming to perform a docking computation method, thus
 * managing all related concepts, including the structure to compute on; the method to
 * apply, results to show...
 */
class DockTask{
public:
    // -------------------- Settings
    // Both - Parameters
    DockParam dock_param;
    int nflex = 0;
    bool show_score = true;

    // -------------------- Molecules
    // CPU - Input Model
    UDFixMol udfix_mol;
    UDFlexMolList udflex_mols;
    std::vector<std::string> fns_flex;
    std::vector<Real> vdw_radii;  // vdW radii for Vina scoring (indexed by atom type)
    AtomTypeProps type_props;     // Data-driven atom type properties for scoring
    FFMinData ff_min_data;        // Force field parameters from DMS (single-ligand)
    SDFMolData sdf_mol;           // Molecular structure data for direct SDF output (single-ligand)
    std::vector<FFMinData> ff_min_data_list;  // Per-ligand FF (multi-ligand merged DMS)
    std::vector<SDFMolData> sdf_mol_list;     // Per-ligand SDF data (multi-ligand merged DMS)
    std::vector<int> rng_slots;               // Per-ligand MC stream slot (Advanced.rng_period); empty = slot in the launch

    // CPU - Output Model
    std::string fp_sdf;           // SDF output path (empty = skip SDF output)

    // Construction
    // flex_mol_list by value: a caller that hands over its list (std::move) saves a copy of the per-ligand
    // receptor pair lists, ~1 MB per ligand on a 2,000-atom pocket
    DockTask(const UDFixMol& fix_mol, UDFlexMolList flex_mol_list, DockParam dock_param,
             std::vector<std::string> fns_flex,
             const std::vector<Real>& vdw_radii = {},
             const AtomTypeProps& type_props = AtomTypeProps()):
        udfix_mol(fix_mol), udflex_mols(std::move(flex_mol_list)), dock_param(dock_param), fns_flex(fns_flex),
        vdw_radii(vdw_radii), type_props(type_props){
        nflex = udflex_mols.size();
    };

    /**
     * @brief Run a whole process: global search, cluster by RMSD, refinement by optimization and final output.
     */
    void run();

    // Create dpfix_mol and dpflex_mols
private:
    // CPU
    int n_atom_all_flex = 0;
    int n_dihe_all_flex = 0;
    std::vector<int> list_i_real; // record indices/ranges of coords & dihedrals in Real data, size: nflex * exhaustiveness
    FlexPose* flex_pose_list_res; // size: nflex * exhaustiveness todo: add a member score instead of these usurpers
    // Finally, use center[3] to record intra, inter, penalty
    // then use orientation[4] to record Predicted Free Energy of Binding, Total score, inter(contains penalty) score, conf_independent part

    Real* flex_pose_list_real_res;
    std::vector<std::vector<int>> clustered_pose_inds_list; // global ind
    std::vector<std::vector<int>> filtered_pose_inds_list; // global ind

    // GPU
    FixMol* fix_mol_cu;
    Real* fix_mol_real_cu;

    FlexPose* flex_pose_list_cu; // size: nflex * exhaustiveness
    Real* flex_pose_list_real_cu;

    FlexTopo* flex_topo_list_cu; // size: nflex
    int* flex_topo_list_int_cu;
    Real* flex_topo_list_real_cu;

    FlexPoseGradient* flex_grad_list_cu;
    FlexPoseHessian* flex_hessian_list_cu;

    FlexParamVina* flex_param_list_cu;
    int* flex_param_list_int_cu;
    Real* flex_param_list_real_cu;
    /// pooled device arena for the ligand torsional-strain rows, sliced per ligand
    TorsionPenaltyRow* flex_param_list_tors_cu = nullptr;
    /// pooled device arena for the aromatic-ring bias rows, sliced per ligand
    RingBiasRow* flex_param_list_ring_cu = nullptr;
    int* flex_param_list_bias_int_cu = nullptr;
    Real* flex_param_list_bias_real_cu = nullptr;
    FixParamVina* fix_param_cu;
    int* fix_param_int_cu;
    bool use_grid_now = false;   // receptor maps built for this task (CUDA GRID path)
    bool torlib_kernels = false; // tor_lib on and a torsion of the batch limited: search / refine take the TORLIB kernels

    Real* aux_list_e_cu; // saves energy of all poses of all flexes, size: nflex * exhaustiveness
    int npose_clustered = 0;
    int* clustered_pose_inds_cu; // pose indices after clustering, -1 indicates that no pose is selected

    // GPU - auxiliary
    FlexPose* aux_poses_cu;
    Real* aux_poses_real_cu;
    FlexPoseHessian* aux_hessians_cu;   // per-pose BFGS state blocks: Hessian + DOF vectors (task.cu, bfgs_state_ptrs)
    Real* aux_hessians_real_cu;
    FlexForce* aux_forces_cu;
    Real* aux_forces_real_cu;

    int* aux_rmsd_ij_cu; // saves pose indices of tri mat (no diagonal), {(i, j) | 0<= i,j < exhaustiveness; i < j}
    int* aux_list_cluster_cu; // saves all pose state: 1 for left, 0 for abandoned (after merging)


    // Per-atom decomposition data: [flex_idx][pose_local_idx] -> vector<VinaTerms> (one per atom)
    std::vector<std::vector<std::vector<VinaTerms>>> decomp_data;
    // Per-atom FF decomposition data (same layout, populated when --decomp + DMS loaded)
    std::vector<std::vector<std::vector<FFAtomTerms>>> ff_decomp_data;
    // Per-pose FF energy summary: [flex_idx][pose_local_idx] -> {total, inter, intra}
    // Populated when ff_min_data.loaded, regardless of --decomp
    std::vector<std::vector<std::array<Real, 3>>> ff_pose_energies;
    std::vector<std::vector<std::vector<VinaTerms>>> residue_decomp_data;
    /// [flex_idx][pose_local_idx] -> center[0..2] (intra, inter, penalty) of the pose-score pass,
    /// whose sum is vina_pose_score. Populated with --decomp only.
    std::vector<std::vector<std::array<Real, 3>>> pose_score_parts;
    /// Raw rescoring-term accumulators per ligand, per output pose. Written to the SDF so a
    /// different polar_env / arom_contact / torsion_strain form can be fitted offline.
    std::vector<std::vector<ScoreDiag>> pose_diag;

    // -------------------- Functions
    void prepare_vina();
    void run_search();
    void run_rescore();   // CUDA GRID path: analytic energies of the search outputs before clustering
    void run_cluster();
    void run_refine();
    void run_filter();
    void run_place_h();
    void run_ff_minimize();
    void run_score();
    void dump_poses();

    void alloc_device();
    void cp_to_cpu();
    void free_device();
    void free_device_mc();
};


#endif //DOCKTASK_H
