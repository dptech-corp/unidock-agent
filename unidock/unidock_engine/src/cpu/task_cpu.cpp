//
// CPU Task Implementation - Memory management and DockTask integration
// Created for Uni-Dock2 CPU backend
//

#include "common_cpu.h"
#include "mc_cpu.h"
#include "cluster_cpu.h"
#include "optimize_cpu.h"
#include "bfgs_cpu.h"
#include "host_topo.h"
#include "task/DockTask.h"
#include <spdlog/spdlog.h>
#include <vector>
#include <cstring>
#include <stdexcept>

// CPU memory allocation for poses
static void alloc_cpu_flex_pose_list(FlexPose** flex_pose_list, Real** flex_pose_list_real,
                                     std::vector<int>* list_i_real,
                                     const UDFlexMolList& udflex_mols, int npose, int nflex,
                                     int exhaustiveness, int* list_natom_flex, int n_atom_all_flex,
                                     int* list_ndihe, int n_dihe_all_flex) {
    
    // Allocate pose array
    *flex_pose_list = new FlexPose[npose];
    
    // Allocate continuous memory for coords and dihedrals
    *flex_pose_list_real = new Real[exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex)];
    
    Real* p_real = *flex_pose_list_real;
    list_i_real->push_back(0);
    
    for (int i = 0; i < nflex; i++) {
        const auto& m = udflex_mols[i];
        for (int j = 0; j < exhaustiveness; j++) {
            FlexPose& pose = (*flex_pose_list)[i * exhaustiveness + j];
            
            // Initialize pose
            pose.center[0] = m.center[0];
            pose.center[1] = m.center[1];
            pose.center[2] = m.center[2];
            pose.rot_vec[0] = 0;
            pose.rot_vec[1] = 0;
            pose.rot_vec[2] = 0;
            pose.energy = 999;
            
            // Set pointers
            pose.coords = p_real;
            list_i_real->push_back(list_i_real->back() + list_natom_flex[i] * 3);
            
            // Copy initial coords
            std::memcpy(pose.coords, m.coords.data(), m.coords.size() * sizeof(Real));
            p_real += list_natom_flex[i] * 3;
            
            pose.dihedrals = p_real;
            list_i_real->push_back(list_i_real->back() + list_ndihe[i]);
            
            // Copy initial dihedrals
            std::memcpy(pose.dihedrals, m.dihedrals.data(), m.dihedrals.size() * sizeof(Real));
            p_real += list_ndihe[i];
        }
    }
}


// CPU implementation of DockTask methods
void DockTask::alloc_device() {
    spdlog::info("Memory Allocation on CPU...");
    
    // Initialize CPU constants
    init_constants_cpu(dock_param, type_props, vdw_radii);
    
    int npose = nflex * dock_param.exhaustiveness;
    
    // Calculate sizes
    n_atom_all_flex = 0;
    n_dihe_all_flex = 0;
    
    std::vector<int> list_natom_flex(nflex);
    std::vector<int> list_ndihe(nflex);
    
    for (int i = 0; i < nflex; i++) {
        const auto& m = udflex_mols[i];
        list_natom_flex[i] = m.natom;
        list_ndihe[i] = m.dihedrals.size();
        n_atom_all_flex += m.natom;
        n_dihe_all_flex += m.dihedrals.size();
    }
    
    // Allocate flex_pose_list
    alloc_cpu_flex_pose_list(&flex_pose_list_cu, &flex_pose_list_real_cu, &list_i_real,
                            udflex_mols, npose, nflex, dock_param.exhaustiveness,
                            list_natom_flex.data(), n_atom_all_flex,
                            list_ndihe.data(), n_dihe_all_flex);
    
    // Allocate flex_topo_list (CPU version - just create array)
    flex_topo_list_cu = new FlexTopo[nflex];
    
    // Allocate continuous memory for topo data
    int total_vn_types = 0, total_axis_atoms = 0, total_range_inds = 0;
    int total_rotated_inds = 0, total_rotated_atoms = 0, total_range_list = 0;
    
    for (int i = 0; i < nflex; i++) {
        const auto& m = udflex_mols[i];
        total_vn_types += m.natom;
        total_axis_atoms += m.torsions.size() * 2;
        total_range_inds += m.torsions.size() * 2;
        total_rotated_inds += m.torsions.size() * 2;
        for (const auto& t : m.torsions) {
            total_rotated_atoms += t.rotated_atoms.size();
            total_range_list += t.range_list.size();
        }
    }
    
    flex_topo_list_int_cu = new int[total_vn_types + total_axis_atoms + total_range_inds + 
                                    total_rotated_inds + total_rotated_atoms];
    flex_topo_list_real_cu = new Real[total_range_list];
    
    int* p_int = flex_topo_list_int_cu;
    Real* p_real = flex_topo_list_real_cu;
    
    for (int i = 0; i < nflex; i++) {
        fill_flex_topo_host(udflex_mols[i], flex_topo_list_cu[i], p_int, p_real);
    }
    
    // Allocate fix_mol
    fix_mol_cu = new FixMol;
    fix_mol_real_cu = new Real[udfix_mol.coords.size()];
    fix_mol_cu->natom = udfix_mol.natom;
    fix_mol_cu->coords = fix_mol_real_cu;
    std::memcpy(fix_mol_cu->coords, udfix_mol.coords.data(), udfix_mol.coords.size() * sizeof(Real));
    
    // Allocate flex_param_list
    flex_param_list_cu = new FlexParamVina[nflex];
    
    int total_pairs = 0;
    for (int i = 0; i < nflex; i++) {
        total_pairs += udflex_mols[i].intra_pairs.size() + udflex_mols[i].inter_pairs.size() +
                      udflex_mols[i].natom;
    }
    int total_r1r2 = 0;
    for (int i = 0; i < nflex; i++) {
        total_r1r2 += (udflex_mols[i].intra_pairs.size() + udflex_mols[i].inter_pairs.size()) / 2;
    }
    
    int total_natom = 0;
    for (int i = 0; i < nflex; i++) total_natom += udflex_mols[i].natom;
    flex_param_list_int_cu = new int[total_pairs];
    flex_param_list_real_cu = new Real[total_r1r2 + total_natom];
    
    p_int = flex_param_list_int_cu;
    p_real = flex_param_list_real_cu;
    
    for (int i = 0; i < nflex; i++) {
        const auto& m = udflex_mols[i];
        FlexParamVina& param = flex_param_list_cu[i];
        
        param.npair_intra = m.intra_pairs.size() / 2;
        param.pairs_intra = p_int;
        std::memcpy(param.pairs_intra, m.intra_pairs.data(), m.intra_pairs.size() * sizeof(int));
        p_int += m.intra_pairs.size();
        
        param.npair_inter = m.inter_pairs.size() / 2;
        param.pairs_inter = p_int;
        std::memcpy(param.pairs_inter, m.inter_pairs.data(), m.inter_pairs.size() * sizeof(int));
        p_int += m.inter_pairs.size();
        
        param.atom_types = p_int;
        std::memcpy(param.atom_types, m.vina_types.data(), m.natom * sizeof(int));
        p_int += m.natom;

        // torsional-strain rows live in the ligand for the run's lifetime; no copy needed
        param.n_tors_penalty = static_cast<int>(m.torsion_penalties.size());
        param.tors_penalty = m.torsion_penalties.empty() ? nullptr : m.torsion_penalties.data();

        // effective united-atom charges for the Coulomb term (charged groups scaled)
        param.charges = p_real;
        for (int k = 0; k < m.natom; k++) {
            Real q = k < (int)m.charges_ua.size() ? m.charges_ua[k] : Real(0);
            if (k < (int)m.charged_group.size() && m.charged_group[k]) q *= g_cpu_constants.weights.elec_charged_scale;
            p_real[k] = q;
        }
        p_real += m.natom;
        
        param.r1_plus_r2_intra = p_real;
        std::memcpy(param.r1_plus_r2_intra, m.r1_plus_r2_intra.data(),
                   m.r1_plus_r2_intra.size() * sizeof(Real));
        p_real += m.r1_plus_r2_intra.size();
        
        param.r1_plus_r2_inter = p_real;
        std::memcpy(param.r1_plus_r2_inter, m.r1_plus_r2_inter.data(),
                   m.r1_plus_r2_inter.size() * sizeof(Real));
        p_real += m.r1_plus_r2_inter.size();
    }

    // Allocate and flatten bias data (PositionBiasTerm, ext/bias/bias_term.h): the rows per ligand, the run's
    // term settings (score/terms.h TermSettings) into every ligand's TermSet. The aromatic-ring rows
    // (RingBiasTerm, ext/bias/ring_term.h) live in the ligand for the run's lifetime; no copy needed
    {
        int total_bias_ints = 0;
        int total_bias_reals = 0;
        for (int i = 0; i < nflex; i++) {
            total_bias_ints += udflex_mols[i].natom * 2;
            total_bias_reals += udflex_mols[i].bias.rows.size() * 5;
        }
        if (total_bias_ints > 0) {
            flex_param_list_bias_int_cu = new int[total_bias_ints]();
            flex_param_list_bias_real_cu = new Real[std::max(1, total_bias_reals)];
        }
        int* p_bias_int = flex_param_list_bias_int_cu;
        Real* p_bias_real = flex_param_list_bias_real_cu;
        for (int i = 0; i < nflex; i++) {
            const auto& m = udflex_mols[i];
            flex_param_list_cu[i].terms.settings = term_settings_of(dock_param);
            RingBiasTerm& ring = flex_param_list_cu[i].terms.ring;
            ring.n = static_cast<int>(m.ring.rows.size());
            ring.rows = m.ring.rows.empty() ? nullptr : m.ring.rows.data();
            PositionBiasTerm& bias = flex_param_list_cu[i].terms.bias;
            bias.inds = p_bias_int;
            bias.params = p_bias_real;
            const int offset = flatten_bias_rows(m, bias.inds, bias.params);
            p_bias_int += m.natom * 2;
            p_bias_real += offset;
        }
        if (total_bias_reals > 0) {
            spdlog::info("Position-bias loaded: {} total bias entries across {} ligands",
                         total_bias_reals / 5, nflex);
        }
    }

    // Build per-ligand bonded-H topology for angular HBond scoring
    for (int i = 0; i < nflex; i++) {
        const auto& m = udflex_mols[i];
        build_bonded_h_topo(flex_param_list_cu[i].bonded_h_topo,
                            m.coords.data(), m.vina_types.data(), m.natom);
    }

    // Allocate fix_param
    fix_param_cu = new FixParamVina;
    fix_param_int_cu = new int[udfix_mol.natom];
    fix_param_cu->atom_types = fix_param_int_cu;
    std::memcpy(fix_param_cu->atom_types, udfix_mol.vina_types.data(),
               udfix_mol.natom * sizeof(int));
    fix_param_cu->charges = udfix_mol.charges_ua.size() == (size_t)udfix_mol.natom
                            ? const_cast<Real*>(udfix_mol.charges_ua.data()) : nullptr;

    // Build spatial grid over receptor heavy atoms for fast neighbor lookup
    build_spatial_grid(g_cpu_constants.receptor_grid,
                       fix_mol_cu->coords, fix_param_cu->atom_types,
                       fix_mol_cu->natom, g_cpu_constants.atp,
                       g_cpu_constants.vdw_radii, fix_param_cu->charges);
    spdlog::info("Spatial grid built: {}x{}x{} = {} cells, {} heavy atoms",
                 g_cpu_constants.receptor_grid.nx,
                 g_cpu_constants.receptor_grid.ny,
                 g_cpu_constants.receptor_grid.nz,
                 g_cpu_constants.receptor_grid.ncells,
                 g_cpu_constants.receptor_grid.total_atoms);

    // Copy receptor donor-H topology (shallow copy — pointers owned by udfix_mol)
    g_cpu_constants.rec_donor_h_topo = udfix_mol.rec_donor_h;

    // Allocate clustering arrays
    clustered_pose_inds_cu = new int[npose];
    
    spdlog::info("Memory allocation on CPU is done.");
}


void DockTask::run_search() {
    if (dock_param.rng_period > 0) {
        // mc_cpu seeds by the pose index in the batch; per-ligand stream slots are implemented on CUDA only
        throw std::runtime_error("Advanced.rng_period > 0 is not supported by the CPU backend");
    }
    spdlog::info("Global Search (MC) on CPU...");
    
    mc_cpu(flex_pose_list_cu, flex_topo_list_cu, *fix_mol_cu, flex_param_list_cu,
           *fix_param_cu, dock_param.mc_steps, dock_param.opt_steps,
           nflex, dock_param.exhaustiveness, dock_param.seed, dock_param.randomize);
    
    spdlog::info("Global Search (MC) on CPU is done.");
}


void DockTask::run_cluster() {
    spdlog::info("Clustering {} * {} = {} poses on CPU...",
                 nflex, dock_param.exhaustiveness, nflex * dock_param.exhaustiveness);
    
    cluster_cpu(clustered_pose_inds_cu, &npose_clustered, &clustered_pose_inds_list,
               flex_pose_list_cu, flex_topo_list_cu,
               nflex, dock_param.exhaustiveness, dock_param.rmsd_limit,
               dock_param.cluster_atom_indices_per_flex);
    
    spdlog::info("Clustering on CPU is done. {} poses are left.", npose_clustered);
}


void DockTask::run_refine() {
    spdlog::info("Run Refinement (BFGS) on CPU for {} steps...", dock_param.refine_steps);
    
    optimize_cpu(flex_pose_list_cu, clustered_pose_inds_cu, flex_topo_list_cu,
                *fix_mol_cu, flex_param_list_cu, *fix_param_cu,
                dock_param.refine_steps, npose_clustered, dock_param.exhaustiveness);
    
    spdlog::info("Refinement (BFGS) on CPU is done.");
}


void DockTask::cp_to_cpu() {
    // For CPU backend, data is already on CPU
    // Just create the result arrays with same pointers
    flex_pose_list_res = flex_pose_list_cu;
    flex_pose_list_real_res = flex_pose_list_real_cu;
}


void DockTask::free_device() {
    spdlog::info("Memory free on CPU...");
    
    // For CPU backend, we own the memory
    // Note: flex_pose_list_res points to same memory as flex_pose_list_cu
    // So we only free once
    
    delete[] flex_pose_list_cu;
    delete[] flex_pose_list_real_cu;
    
    delete[] flex_topo_list_cu;
    delete[] flex_topo_list_int_cu;
    delete[] flex_topo_list_real_cu;
    
    delete fix_mol_cu;
    delete[] fix_mol_real_cu;
    
    for (int i = 0; i < nflex; i++) {
        free_bonded_h_topo(flex_param_list_cu[i].bonded_h_topo);
    }
    delete[] flex_param_list_cu;
    delete[] flex_param_list_int_cu;
    delete[] flex_param_list_real_cu;
    delete[] flex_param_list_bias_int_cu;
    delete[] flex_param_list_bias_real_cu;
    
    delete fix_param_cu;
    delete[] fix_param_int_cu;

    free_spatial_grid(g_cpu_constants.receptor_grid);
    // NOTE: do NOT free rec_donor_h_topo here — it is a shallow copy of
    // udfix_mol.rec_donor_h, which is owned by the caller and reused
    // across batches.
    g_cpu_constants.rec_donor_h_topo = {};

    delete[] clustered_pose_inds_cu;
    
    // Don't delete flex_pose_list_res and flex_pose_list_real_res
    // as they point to the same memory
    flex_pose_list_res = nullptr;
    flex_pose_list_real_res = nullptr;
    
    spdlog::info("Memory free on CPU is done.");
}


void DockTask::free_device_mc() {
    // For CPU backend, nothing special to do
}
