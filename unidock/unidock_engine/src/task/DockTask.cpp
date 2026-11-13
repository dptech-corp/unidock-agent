//
// Created by Congcong Liu on 24-9-23.
//

#include <spdlog/spdlog.h>
#include <algorithm>  // 为std::sort
#include <numeric>  // 为std::iota
#include <cstdlib>  // std::getenv
#include <map>
#include <memory>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "DockTask.h"
#if USE_CUDA
#include "search/mc.h"
#include "cluster/cluster.h"
#include "optimize/optimize.h"
#endif
#include "format/sdf_writer.h"
#include "score/vina.h"
#include "score/score.h"
#include "cpu/score_ff_cpu.h"




void DockTask::run(){
    spdlog::debug("All file names of this Task: ");
    for(auto& f : fns_flex){
        spdlog::debug(f);
    }

    // Load FF parameters from DMS if paths are provided (enables FF energy reporting
    // even when ff_minimize_steps == 0; minimization itself is gated separately)
    if (!ff_min_data_list.empty()) {
        // Pre-loaded from multi-ligand merged DMS — nothing to do
        spdlog::info("Using pre-loaded per-ligand FF data ({} ligands)", ff_min_data_list.size());
    } else if (!dock_param.ligand_dms_path.empty() && !dock_param.protein_dms_path.empty()) {
        try {
            read_ligand_dms(dock_param.ligand_dms_path, ff_min_data);
            // FF scoring needs both sides; a ligand without FF parameters (most Vina-only inputs)
            // must not pull in a copy of the receptor FF.
            if (ff_min_data.loaded) {
                read_receptor_dms(dock_param.protein_dms_path, ff_min_data);
            }
            if (ff_min_data.loaded) {
                spdlog::info("FF parameter source: ligand {} / receptor {}",
                             dock_param.ligand_dms_path, dock_param.protein_dms_path);
            }
        } catch (const std::exception& e) {
            spdlog::error("Failed to load DMS files, FF scoring disabled: {}", e.what());
            ff_min_data.loaded = false;
        }
    }

    // Load SDF mol data from ligand DMS for direct SDF output
    if (!sdf_mol_list.empty()) {
        // Pre-loaded from multi-ligand merged DMS
        spdlog::info("Using pre-loaded per-ligand SDF data ({} ligands)", sdf_mol_list.size());
    } else if (!dock_param.ligand_dms_path.empty() && !fp_sdf.empty()) {
        try {
            read_sdf_mol_data(dock_param.ligand_dms_path, sdf_mol);
        } catch (const std::exception& e) {
            spdlog::warn("Failed to load SDF mol data from DMS, "
                         "SDF output disabled: {}", e.what());
            sdf_mol.loaded = false;
        }
    }

#if USE_CUDA
    // Host-side stages (FF scoring in run_score, run_ff_minimize) read the CPU
    // constants; the CUDA alloc_device does not populate them.
    init_constants_cpu(dock_param, type_props, vdw_radii);
#endif

    try {
        // compute necessary sizes
        prepare_vina();
    } catch (const std::exception& e) {
        spdlog::critical("Failed to prepare Vina: {}", e.what());
        throw;  // the device state is unusable after this; a later stage would read missing buffers
    }

    try {
        alloc_device();
    } catch (const std::exception& e) {
        spdlog::critical("Failed to allocate device memory: {}", e.what());
        throw;  // the device state is unusable after this; a later stage would read missing buffers
    }

    try {
        run_search();
    } catch (const std::exception& e) {
        spdlog::critical("Failed to run search: {}", e.what());
        throw;  // the device state is unusable after this; a later stage would read missing buffers
    }
#if USE_CUDA
    // GRID path: the chains were minimised on the receptor maps; the clustering and the shortlist
    // must order analytic energies
    if (use_grid_now){
        try {
            run_rescore();
        } catch (const std::exception& e) {
            spdlog::critical("Failed to run rescore: {}", e.what());
        }
    }
#endif
    // todo: free some unnecessary data


    try {
        run_cluster();
    } catch (const std::exception& e) {
        spdlog::critical("Failed to run cluster: {}", e.what());
    }

    // todo: free some unnecessary data


    if (dock_param.refine_steps > 0){
        try {
            run_refine();
        } catch (const std::exception& e) {
            spdlog::critical("Failed to run refine: {}", e.what());
        }
    }

    // Move to CPU
    try{
        run_filter();
    } catch (const std::exception& e) {
        spdlog::critical("Failed to copy results from GPU to CPU: {}", e.what());
    }

    // Cone-donor hydrogens toward the receptor acceptors: H coordinates only, so the Vina scores stay
    // unchanged unless an FF minimisation follows; FF energies see the placed H
    try {
        run_place_h();
    } catch (const std::exception& e) {
        spdlog::critical("Failed to place donor hydrogens: {}", e.what());
    }

    // FF Minimization (optional, after filter, before Vina rescore)
    try {
        run_ff_minimize();
    } catch (const std::exception& e) {
        spdlog::critical("Failed to run FF minimize: {}", e.what());
    }

    // Run on CPU
    // todo: if not cluster or not refine?
    try {
        run_score();
    } catch (const std::exception& e) {
        spdlog::critical("Failed to run score: {}", e.what());
    }

    // Run on CPU
    try {
        spdlog::info("Dumping the poses...");
        dump_poses();
        spdlog::info("Dumping is done.");
    } catch (const std::exception& e) {
        spdlog::critical("Failed to dump poses: {}", e.what());
    }

    // Run on CPU
    try {
        free_device();
    } catch (const std::exception& e) {
        spdlog::critical("Failed to free device memory: {}", e.what());
    }
}



void DockTask::prepare_vina(){
    // compute r1 + r2 values for acceleration
    
    // Use vdw_radii if available, otherwise fall back to hardcoded defaults
    const Real* radii = vdw_radii.empty() ? VN_VDW_RADII : vdw_radii.data();
    const int max_type = vdw_radii.empty() ? 21 : static_cast<int>(vdw_radii.size());

    for (auto& flex_mol : udflex_mols){
        // compute r1 + r2 for all intra pairs
        for (int j = 0; j < flex_mol.intra_pairs.size(); j += 2){
            int i1 = flex_mol.intra_pairs[j];
            int i2 = flex_mol.intra_pairs[j + 1];
            int t1 = flex_mol.vina_types[i1];
            int t2 = flex_mol.vina_types[i2];
            if (t1 >= max_type || t2 >= max_type) {
                spdlog::warn("Atom type {} or {} exceeds vdw_radii size {}", t1, t2, max_type);
            }
            flex_mol.r1_plus_r2_intra.push_back(radii[t1] + radii[t2]);
        }

        // compute r1 + r2 for all inter pairs
        for (int j = 0; j < flex_mol.inter_pairs.size(); j += 2){
            int i1 = flex_mol.inter_pairs[j];
            int i2 = flex_mol.inter_pairs[j + 1];
            int t1 = flex_mol.vina_types[i1];
            int t2 = udfix_mol.vina_types[i2];
            if (t1 >= max_type || t2 >= max_type) {
                spdlog::warn("Atom type {} or {} exceeds vdw_radii size {}", t1, t2, max_type);
            }
            flex_mol.r1_plus_r2_inter.push_back(radii[t1] + radii[t2]);
        }
    }
}


#if USE_CUDA
/**
 * One Step of MC to mutate?
 */
void DockTask::run_search(){
    spdlog::info("Global Search (MC)...");
    // Advanced.rng_period: per-ligand stream slots on the device; nullptr keeps each ligand's slot in the launch
    int* rng_slots_cu = nullptr;
    if (!rng_slots.empty()){
        checkCUDA(cudaMalloc(&rng_slots_cu, sizeof(int) * rng_slots.size()));
        checkCUDA(cudaMemcpy(rng_slots_cu, rng_slots.data(), sizeof(int) * rng_slots.size(), cudaMemcpyHostToDevice));
    }
    // Perform Metropolis-MC and update the flex_pose_list
    mc_cu(flex_pose_list_cu, flex_topo_list_cu,
          *fix_mol_cu, flex_param_list_cu, *fix_param_cu,
          aux_poses_cu, aux_hessians_cu, aux_forces_cu,
          dock_param.mc_steps, dock_param.opt_steps, nflex, dock_param.exhaustiveness,
          dock_param.seed, dock_param.randomize, use_grid_now, rng_slots_cu, torlib_kernels);
    if (rng_slots_cu){
        checkCUDA(cudaFree(rng_slots_cu));
    }
    spdlog::info("Global Search (MC) is done.");
}

void DockTask::run_rescore(){
    rescore_cu(flex_pose_list_cu, flex_topo_list_cu, *fix_mol_cu, flex_param_list_cu, *fix_param_cu,
               aux_forces_cu, nflex, dock_param.exhaustiveness);
}

/**
 * @brief Remove duplicated poses by rmsd limit. Poses with lower energy is preferred.
 */
void DockTask::run_cluster(){
    // clustering according to rmsd_limit, num_pose. [ energy_range(not loaded)]
    spdlog::info("Clustering {} * {} = {} poses ...", nflex, dock_param.exhaustiveness, nflex * dock_param.exhaustiveness);
    cluster_cu(clustered_pose_inds_cu, &npose_clustered, &clustered_pose_inds_list, flex_pose_list_cu, flex_topo_list_cu,
        aux_list_e_cu, aux_list_cluster_cu, aux_rmsd_ij_cu,
        nflex, dock_param.exhaustiveness, dock_param.rmsd_limit);
    spdlog::info("Clustering is done. {} poses are left.", npose_clustered);
}


void DockTask::run_refine(){
    int refine_steps = dock_param.refine_steps;
    if (use_grid_now){
        // GRID path: the search minima are map minima, so the analytic polish gets at least the
        // opt_steps heuristic (Vina's (25 + natom) / 3) over the largest ligand of the batch
        int natom_max = 0;
        for (const auto& m : udflex_mols) natom_max = std::max(natom_max, m.natom);
        refine_steps = std::max(refine_steps, (25 + natom_max) / 3);
        spdlog::info("GRID refine steps: max({}, (25 + {}) / 3) = {}", dock_param.refine_steps, natom_max, refine_steps);
    }
    spdlog::info("Run Refinement (BFGS) for {} steps...", refine_steps);
    optimize_cu(flex_pose_list_cu, clustered_pose_inds_cu, flex_topo_list_cu, *fix_mol_cu,
                          flex_param_list_cu, *fix_param_cu,
                          aux_poses_cu, aux_hessians_cu,
                          aux_forces_cu,
                          refine_steps, npose_clustered, dock_param.exhaustiveness, torlib_kernels);
    spdlog::info("Refinement (BFGS) is done.");
}
#endif  // USE_CUDA

void DockTask::run_filter(){
    spdlog::info("Filtering by rescore_pool / num_pose & energy_range...");

    // copy results to cpu
    cp_to_cpu();

    // Object: all clutered poses
    for (int i = 0; i < nflex; i ++){ // for each flex
        auto& clustered_inds = clustered_pose_inds_list[i];

        // prepare indices
        std::vector<int> sorted_indices(clustered_inds.size()); // find the best num_modes poses for each flex
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0); // 填充0,1,2,...
        std::sort(sorted_indices.begin(), sorted_indices.end(),
        [this, &clustered_inds](int a, int b) { return this->flex_pose_list_res[clustered_inds[a]].energy < this->flex_pose_list_res[clustered_inds[b]].energy; });

        // UD2_LOG_POSE_ENERGY set: log each clustered pose energy (search / refinement energy, bias included)
        if (std::getenv("UD2_LOG_POSE_ENERGY") != nullptr){
            for (auto& j: sorted_indices){
                spdlog::info("pose_energy ligand {} name {} pose {} energy {:.9g}", i, udflex_mols[i].name,
                             clustered_inds[j], flex_pose_list_res[clustered_inds[j]].energy);
            }
        }

        if (dock_param.use_inner_box){
            // Inner box: drop poses whose heavy-atom centroid left it (the search wall makes
            // this rare); keep everything if no pose qualifies
            const UDFlexMol& m = udflex_mols[i];
            std::vector<int> kept;
            for (auto& j: sorted_indices){
                const Real* c = flex_pose_list_res[clustered_inds[j]].coords;
                Real cx = 0, cy = 0, cz = 0; int nh = 0;
                for (int a = 0; a < m.natom; a++){
                    if (type_props.is_hydrogen[m.vina_types[a]]) continue;
                    cx += c[a * 3]; cy += c[a * 3 + 1]; cz += c[a * 3 + 2]; nh++;
                }
                if (nh > 0 && dock_param.inner_box.is_inside(cx / nh, cy / nh, cz / nh)) kept.push_back(j);
            }
            if (kept.empty()) spdlog::warn("Flex {}: no pose inside the inner box, keeping all", i);
            else sorted_indices = kept;
        }
        int count = 0;
        Real e_min = flex_pose_list_res[clustered_inds[sorted_indices[0]]].energy;
        std::vector<int> filtered_inds;
        for (auto& j: sorted_indices){
            if(flex_pose_list_res[clustered_inds[j]].energy > e_min + dock_param.energy_range){
                break;
            }
            filtered_inds.push_back(clustered_inds[j]);
            count ++;
            // keep the re-ranking pool; run_score cuts the sorted pool to num_pose before the write
            if (count >= std::max(dock_param.rescore_pool, dock_param.num_pose)){
                break;
            }
        }
        filtered_pose_inds_list.push_back(filtered_inds);
    }
    spdlog::info("Filtering is done.");
}


void DockTask::run_place_h(){
    if (!dock_param.angular_hbond) return;
    // task score / randomize: no search or refinement ran, the input hydrogens are kept
    if (dock_param.mc_steps == 0 && dock_param.refine_steps == 0) return;
    // constraint and position-bias runs keep the hydrogens of the template (SearchPolicy, config/parse.cpp)
    if (dock_param.policy.static_root || dock_param.policy.keep_template_hydrogens) return;

    const Real* radii = vdw_radii.empty() ? VN_VDW_RADII : vdw_radii.data();
    std::vector<int> rec_acceptors;
    for (int a = 0; a < udfix_mol.natom; a++){
        if (type_props.is_acceptor[udfix_mol.vina_types[a]]) rec_acceptors.push_back(a);
    }

    int n_hydrogen = 0, n_pose = 0;
    // a run_filter that threw can leave fewer ligands than nflex
    const int n_lig = std::min(nflex, static_cast<int>(filtered_pose_inds_list.size()));
    for (int i = 0; i < n_lig; i ++){
        const UDFlexMol& mol = udflex_mols[i];
        // PROTAC: the flexible system also holds E3 atoms that the SDF does not write; never touch them
        int n_write = mol.natom;
        if (!sdf_mol_list.empty()){
            n_write = sdf_mol_list[i < static_cast<int>(sdf_mol_list.size()) ? i : 0].natom;
        } else if (sdf_mol.loaded){
            n_write = sdf_mol.natom;
        }
        const std::vector<PlaceHDonor> donors = find_placeable_cone_donors(mol, n_write, type_props);
        if (donors.empty()) continue;

        for (auto& j: filtered_pose_inds_list[i]){
            FlexPose& pose = flex_pose_list_res[j];
            int n_h = 0;
            if (place_cone_donor_h(pose.coords, mol.natom, donors, udfix_mol, rec_acceptors, radii, type_props,
                                   mol.vina_types.data(), n_h) == 0) continue;
            n_hydrogen += n_h;
            n_pose ++;
            // all-atom mean, the rotation pivot of the FF minimisation
            Real center_new[3] = {0, 0, 0};
            for (int a = 0; a < mol.natom; a++){
                center_new[0] += pose.coords[a * 3];
                center_new[1] += pose.coords[a * 3 + 1];
                center_new[2] += pose.coords[a * 3 + 2];
            }
            pose.center[0] = center_new[0] / mol.natom;
            pose.center[1] = center_new[1] / mol.natom;
            pose.center[2] = center_new[2] / mol.natom;
        }
    }
    spdlog::info("{} donor hydrogens placed in {} poses.", n_hydrogen, n_pose);
}



void DockTask::run_score(){
    spdlog::info("Scoring...");
    // only score once
    Vina v;
    std::string show_format = "{:<10}{:<20}";

    // Determine if we have FF data (either per-ligand list or single)
    bool has_ff = !ff_min_data_list.empty() || ff_min_data.loaded;

    pose_diag.assign(nflex, {});

    if (dock_param.decomp){
        decomp_data.resize(nflex);
        residue_decomp_data.resize(nflex);
        pose_score_parts.resize(nflex);
        spdlog::info("Energy decomposition enabled: per-atom Vina term breakdown");
        if (has_ff) {
            ff_decomp_data.resize(nflex);
            spdlog::info("FF per-atom decomposition also enabled (DMS data loaded)");
        }
    }

    // Always compute per-pose FF energy summary when DMS data is loaded
    if (has_ff) {
        ff_pose_energies.resize(nflex);
    }

    // Pass 1: the scores, in parallel over ligands. Ligand i reads its own topology and poses and
    // writes only its own slots (flex_pose_list_res[j] for j in filtered_pose_inds_list[i], and the
    // [i] entries of pose_diag / ff_pose_energies / decomp_data / residue_decomp_data /
    // pose_score_parts / ff_decomp_data, all sized before the loop), and the scorer SF is
    // thread-local and re-weighted at the start of every ligand. No sum crosses ligands and the
    // loops inside a ligand are the serial ones, so every ligand performs the serial sequence of
    // floating-point operations whatever the thread count. Pass 2 writes the log in ligand order.
    // Threads: Hardware.ncpu caps the team (the CUDA build compiles main.cpp without OpenMP, so the
    // cap is applied here), and batches under 8 ligands per thread run serially: a re-docking task
    // or an MPS worker scores one ligand, where spawning a team only adds latency and, with eight
    // workers on one host, contention.
    int nthreads = 1;
#ifdef _OPENMP
    if (dock_param.ncpu > 0) omp_set_num_threads(std::max(1, std::min(dock_param.ncpu, omp_get_num_procs())));
    nthreads = omp_get_max_threads();
    nthreads = std::max(1, std::min(nthreads, nflex / 8));
    spdlog::debug("Scoring with {} OpenMP threads", nthreads);
#endif
    std::string first_error;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) if(nthreads > 1)
#endif
    for (int i = 0; i < nflex; i ++){
      try {
        const auto& mol = udflex_mols[i];
        int n_tors = mol.torsions.size() + mol.ff_only_torsions.size();

        // Select per-ligand or single FF data
        const FFMinData& ff_ref = !ff_min_data_list.empty() ? ff_min_data_list[i] : ff_min_data;

        // use center[3] to record intra, inter, penalty
        // then use orientation[4] to record Predicted Free Energy of Binding, Total score, inter(contains penalty) score, conf_independent part
        // Angular HBond in the reported score: same bonded-H topology (from the input
        // conformer, bonds are preserved by the search) and receptor H table as the search.
        SF.set_weights(dock_param.pose_weights());
        // The decomposition reconciles with vina_pose_score, so it keeps this weight set even when
        // the affinity pass below re-weights SF.
        Vina sf_pose = SF;
        BondedHTopo h_topo = {};
        if (dock_param.angular_hbond){
            build_bonded_h_topo(h_topo, mol.coords.data(), mol.vina_types.data(), mol.natom);
        }
        const RecDonorHTopo* rh_topo = &udfix_mol.rec_donor_h;

        int j_r1 = filtered_pose_inds_list[i][0];
        score(flex_pose_list_res + j_r1, flex_pose_list_real_res + list_i_real[j_r1 * 2], udfix_mol, mol, dock_param.box, type_props,
              dock_param.angular_hbond, &h_topo, rh_topo);
        Real e_intra_rank1 = flex_pose_list_res[j_r1].center[0];

        // Order the output poses by the pose-selection score, so the first pose is the chosen one
        // even when the reported affinity comes from a different function.
        // A biased run (bias type set) orders them by vina_pose_score + the candidate's bias energy, the search's atom
        // and ring terms evaluated here on the host (score/terms.h HostTermSet) at the coordinates score() rescores,
        // so the pose the bias pulled onto the template is not displaced by a better-scoring unaligned one (plan
        // step 7). vina_pose_score (rot_vec[2]), the decomposition and every energy stay unbiased; the affinity
        // pass takes its reference intra from the written top pose, the first of this order, as in an unbiased run,
        // so when the bias changes the top-1 every affinity of the ligand is relative to
        // the new top pose. The bias energy is written as ud2_bias_energy. An unbiased run sorts by vina_pose_score
        // alone, as before.
        const bool biased = dock_param.bias_type != BT_NO;
        std::unique_ptr<HostTermSet> host_terms;
        if (biased) host_terms = std::make_unique<HostTermSet>(mol, term_settings_of(dock_param));
        std::map<int, Real> pose_bias;  // pose -> its bias energy (biased runs)
        std::map<int, std::array<Real, 3>> pose_pass_parts;  // pose -> center[0..2] of this pass (--decomp)
        for (auto& j: filtered_pose_inds_list[i]){
            score(flex_pose_list_res + j, flex_pose_list_real_res + list_i_real[j * 2], udfix_mol, mol, dock_param.box, type_props,
                  dock_param.angular_hbond, &h_topo, rh_topo);
            flex_pose_list_res[j].rot_vec[2] = flex_pose_list_res[j].center[0] + flex_pose_list_res[j].center[1]
                                             + flex_pose_list_res[j].center[2];
            if (biased){
                pose_bias[j] = host_terms->energy(flex_pose_list_real_res + list_i_real[j * 2]);
            }
            if (dock_param.decomp){
                pose_pass_parts[j] = {flex_pose_list_res[j].center[0], flex_pose_list_res[j].center[1],
                                      flex_pose_list_res[j].center[2]};
            }
        }
        if (biased){
            std::stable_sort(filtered_pose_inds_list[i].begin(), filtered_pose_inds_list[i].end(),
                             [&](int a, int b){ return flex_pose_list_res[a].rot_vec[2] + pose_bias.at(a)
                                                       < flex_pose_list_res[b].rot_vec[2] + pose_bias.at(b); });
        } else {
            std::stable_sort(filtered_pose_inds_list[i].begin(), filtered_pose_inds_list[i].end(),
                             [&](int a, int b){ return flex_pose_list_res[a].rot_vec[2] < flex_pose_list_res[b].rot_vec[2]; });
        }
        // The pose pass chose among the pool (rescore_pool); only the first num_pose are labelled and written
        if (static_cast<int>(filtered_pose_inds_list[i].size()) > dock_param.num_pose){
            filtered_pose_inds_list[i].resize(dock_param.num_pose);
        }
        if (dock_param.has_affinity_weights){
            // the reported number is a cross-ligand quantity and gets its own weight set
            SF.set_weights(dock_param.affinity_weights);
            j_r1 = filtered_pose_inds_list[i][0];
            score(flex_pose_list_res + j_r1, flex_pose_list_real_res + list_i_real[j_r1 * 2], udfix_mol, mol, dock_param.box, type_props,
                  dock_param.angular_hbond, &h_topo, rh_topo);
            e_intra_rank1 = flex_pose_list_res[j_r1].center[0];
        }

        pose_diag[i].assign(filtered_pose_inds_list[i].size(), ScoreDiag{});
        int diag_slot = 0;
        for (auto& j: filtered_pose_inds_list[i]){
            ScoreDiag& dg = pose_diag[i][diag_slot++];
            score(flex_pose_list_res + j, flex_pose_list_real_res + list_i_real[j * 2], udfix_mol, mol, dock_param.box, type_props,
                  dock_param.angular_hbond, &h_topo, rh_topo, &dg);
            if (!mol.ring.rows.empty()){
                // the search's ring term on the written pose (ext/bias/ring_term.h), reported as ud2_ring_bias_raw
                dg.n_ring_bias = static_cast<int>(mol.ring.rows.size());
                dg.ring_bias = ring_bias_energy(mol.ring.rows, flex_pose_list_real_res + list_i_real[j * 2],
                                                dock_param.bias_k, dock_param.ring_bias);
            }
            if (biased){
                // the pose pass's bias energy of this pose (same coordinates), reported as ud2_bias_energy
                dg.bias_energy = pose_bias.at(j);
                dg.has_bias_energy = true;
            }
            flex_pose_list_res[j].rot_vec[1] = flex_pose_list_res[j].center[0] + flex_pose_list_res[j].center[1] +
                flex_pose_list_res[j].center[2]; // Total

            Real e_inter = flex_pose_list_res[j].rot_vec[1] - e_intra_rank1; // Real adopted inter
            // Free Energy of Binding
            flex_pose_list_res[j].rot_vec[0] = v.vina_conf_indep(e_inter, n_tors);  // Affinity
            flex_pose_list_res[j].rot_vec[3] = flex_pose_list_res[j].rot_vec[0] - e_inter;  // Conf-Independent

            // FF energy computation (when DMS data loaded, regardless of --decomp)
            if (ff_ref.loaded) {
                auto ff_terms = score_decomp_ff(flex_pose_list_res + j, ff_ref);

                // Sum per-atom terms to get molecule-level energies
                Real ff_total = 0, ff_inter_sum = 0, ff_intra_sum = 0;
                for (const auto& t : ff_terms) {
                    ff_inter_sum += t.inter_lj + t.inter_coulomb;
                    ff_intra_sum += t.intra_lj + t.intra_coulomb + t.torsion + t.bond + t.angle;
                    ff_total     += t.total;
                }
                ff_pose_energies[i].push_back({ff_total, ff_inter_sum, ff_intra_sum});

                // Per-atom decomp data (only when --decomp is enabled)
                if (dock_param.decomp) {
                    ff_decomp_data[i].push_back(std::move(ff_terms));
                }
            }

            // Per-atom / per-residue decomposition of the inter part of vina_pose_score
            // (only when --decomp is enabled)
            if (dock_param.decomp){
                std::vector<VinaTerms> atom_terms, res_terms;
                score_decomp(sf_pose, flex_pose_list_real_res + list_i_real[j * 2], udfix_mol, mol, type_props,
                             dock_param.angular_hbond, &h_topo, rh_topo, atom_terms,
                             udfix_mol.residues.empty() ? nullptr : &res_terms);
                if (!udfix_mol.residues.empty()) {
                    residue_decomp_data[i].push_back(std::move(res_terms));
                }
                decomp_data[i].push_back(std::move(atom_terms));
                pose_score_parts[i].push_back(pose_pass_parts.at(j));
            }
        }
        if (h_topo.h_start != nullptr){
            free_bonded_h_topo(h_topo);
        }
      } catch (const std::exception& e) {
        // an exception must not leave the parallel region; the first one is rethrown below
#ifdef _OPENMP
        #pragma omp critical(run_score_error)
#endif
        if (first_error.empty()) first_error = e.what();
      }
    }
    if (!first_error.empty()){
        throw std::runtime_error(first_error);
    }

    // Pass 2: the per-ligand affinity table, serial and in ligand order (same lines as before)
    for (int i = 0; i < nflex; i ++){
        if(show_score){
            spdlog::info("-------------------------------------------");
            spdlog::info(show_format, "Rank", "Affinity (kcal/mol)");
            spdlog::info("-------------------------------------------");
        }
        int pose_num = 0;
        for (auto& j: filtered_pose_inds_list[i]){
            pose_num ++;
            if(show_score){
                spdlog::info(show_format, pose_num, flex_pose_list_res[j].rot_vec[0]);
            }
        }
        if(show_score){
            spdlog::info("-------------------------------------------");
        }
    }

    spdlog::info("Scoring is done.");
}



void DockTask::dump_poses(){
    if (!sdf_mol_list.empty() && !fp_sdf.empty()) {
        // Multi-ligand: per-ligand SDF data
        write_poses_to_sdf(fp_sdf, sdf_mol_list, fns_flex, filtered_pose_inds_list,
            flex_pose_list_res, flex_pose_list_real_res, list_i_real,
            decomp_data, ff_decomp_data, ff_pose_energies,
            residue_decomp_data, udfix_mol.residues, pose_diag,
            pose_score_parts, dock_param.pose_weights());
    } else if (sdf_mol.loaded && !fp_sdf.empty()) {
        // Single-ligand: wrap in a vector
        write_poses_to_sdf(fp_sdf, std::vector<SDFMolData>{sdf_mol}, fns_flex,
            filtered_pose_inds_list,
            flex_pose_list_res, flex_pose_list_real_res, list_i_real,
            decomp_data, ff_decomp_data, ff_pose_energies,
            residue_decomp_data, udfix_mol.residues, pose_diag,
            pose_score_parts, dock_param.pose_weights());
    } else {
        spdlog::warn("No SDF data available; skipping pose output");
    }
}

