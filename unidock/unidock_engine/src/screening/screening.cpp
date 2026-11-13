//
// Created by Congcong Liu on 24-10-9.
//

#include "model/model.h"
#include "myutils/myio.h"
#include "task/DockTask.h"
#include <stdexcept>
#include "screening.h"
#if USE_CUDA
#include "cuda/common.cuh"
#else
// Define STRIDE constants used by predict_gpu_flex (originally from cuda/common.cuh)
static const int STRIDE_POSE = 4;
static const int STRIDE_G = 6;   // task.cu allocates STRIDE_G (common.cuh) gradient slots per pose
#endif
#include <spdlog/spdlog.h>

/**
 * @brief Group ligands into groups, for better managing cuda kernels and optimizing performances
 * @param flex_mol_list Flxible molecules
 * @return Groups of ligands
 */
std::vector<std::vector<int>> group_flex_mols(const UDFlexMolList &flex_mol_list) {
    std::vector<std::vector<int>> groups(NGroup + 1);
    bool flag_overflow;
    for (int i = 0; i < flex_mol_list.size(); i ++) {
        flag_overflow = true;
        for (int igroup = 0; igroup < NGroup; ++igroup){
            if (flex_mol_list[i].vina_types.size() <= NatomThresholds[igroup] &&
                flex_mol_list[i].torsions.size() <= NTorsionThresholds[igroup]) {
                groups[igroup].push_back(i);
                flag_overflow = false;
                break;
            }
        }

        if(flag_overflow){
            groups[NGroup].push_back(i);
        }
    }

    // print groups info    
    for (int igroup  = 0; igroup < NGroup + 1; igroup ++){
        spdlog::info("Group {} size: {}", GroupNames[igroup], groups[igroup].size());
    }
    return groups;
}


int predict_gpu_fix(UDFixMol& udfix_mol){
    // Only consider > 1 MB
    int s1 = sizeof(FixMol) + udfix_mol.coords.size() * sizeof(Real);
    // fix_param_cu
    int s2 = sizeof(FixParamVina) + udfix_mol.natom * sizeof(int);

    return (s1 + s2) / 1048576;
}

/**
 * @brief The per-ligand sums predict_gpu_flex evaluates, kept as a running total so that batch
 * assembly adds one ligand per candidate instead of re-summing the batch. add() performs the same
 * integer += statements, in the same ligand order, as the one-pass loop, so the totals and the
 * predicted size are identical.
 */
struct FlexBatchTotals {
    int n_range_all_flex = 0, n_rotated_atoms_all_flex = 0;
    int n_dim_all_flex = 0, n_dim_tri_mat_all_flex = 0;
    int size_inter_all_flex = 0, size_intra_all_flex = 0;
    int n_atom_all_flex = 0;
    int n_dihe_all_flex = 0;

    void add(const UDFlexMol& m){
        size_intra_all_flex += m.intra_pairs.size();
        size_inter_all_flex += m.inter_pairs.size(); // all possible pairs

        n_atom_all_flex += m.natom;
        n_dihe_all_flex += m.dihedrals.size();

        int dim = 3 + 4 + m.dihedrals.size();
        n_dim_all_flex += dim;
        n_dim_tri_mat_all_flex += dim * (dim + 1) / 2;
        for (auto t : m.torsions){
            n_rotated_atoms_all_flex += t.rotated_atoms.size();
            n_range_all_flex += t.range_list.size() / 2;
        }
    }
};

static int predict_gpu_flex(const FlexBatchTotals& t, int nflex, int exhaustiveness){
    // Byte sums in 64-bit: in int the sum wrapped once a candidate batch passed ~2 GB, so the MB total
    // never reached the device limit and a size group was never split (out of memory at ~9,000 ligands, exh 512).
    typedef long long i64;
    const i64 npose = (i64)exhaustiveness * nflex;
    const i64 n_range_all_flex = t.n_range_all_flex, n_rotated_atoms_all_flex = t.n_rotated_atoms_all_flex;
    const i64 n_dim_tri_mat_all_flex = t.n_dim_tri_mat_all_flex;
    const i64 size_inter_all_flex = t.size_inter_all_flex, size_intra_all_flex = t.size_intra_all_flex;
    const i64 n_atom_all_flex = t.n_atom_all_flex;
    const i64 n_dihe_all_flex = t.n_dihe_all_flex;

    //
    i64 s1 = npose * (i64)sizeof(FlexPose) + (i64)exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex) * (i64)sizeof(Real);
    i64 s2 = (i64)nflex * (i64)sizeof(FlexTopo) + (n_atom_all_flex + n_dihe_all_flex * 2 + n_dihe_all_flex * 2 +
        n_dihe_all_flex * 2 + n_rotated_atoms_all_flex) * (i64)sizeof(int) +
            n_range_all_flex * 2 * (i64)sizeof(Real);
    // flex_param_list_cu
    i64 s3 = (i64)nflex * (i64)sizeof(FlexParamVina) +
        (size_intra_all_flex + size_inter_all_flex + n_atom_all_flex) * (i64)sizeof(int) +
            (size_intra_all_flex + size_inter_all_flex) / 2 * (i64)sizeof(Real);
    // aux_pose_cu
    i64 s4 = (i64)STRIDE_POSE * npose * (i64)sizeof(FlexPose) +
        (i64)STRIDE_POSE * exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex) * (i64)sizeof(Real);
    // aux_grads_cu
    i64 s5 = (i64)STRIDE_G * npose * (i64)sizeof(FlexPoseGradient) +
        (i64)STRIDE_G * exhaustiveness * n_dihe_all_flex * (i64)sizeof(Real);

    // aux_hessians_cu
    i64 s6 = npose * (i64)sizeof(FlexPoseHessian) +
        (i64)exhaustiveness * n_dim_tri_mat_all_flex * (i64)sizeof(Real);
    // aux_forces_cu
    i64 s7 = npose * (i64)sizeof(FlexForce) +
        (i64)exhaustiveness * n_atom_all_flex * 3 * (i64)sizeof(Real);
    // Clustering
    i64 npair = (i64)exhaustiveness * (exhaustiveness - 1) / 2; // tri-mat with diagonal
    i64 s8 = npose * (i64)sizeof(Real) +
        (i64)nflex * exhaustiveness * (i64)sizeof(int) +
            (i64)nflex * npair * 2 * (i64)sizeof(int) +
                (i64)nflex * exhaustiveness * (i64)sizeof(int);
    i64 total = (s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8) / 1048576;
    return total > 2147483647LL ? 2147483647 : (int)total;
}

int predict_gpu_flex(UDFlexMolList& udflex_mols, int exhaustiveness){
    int nflex = udflex_mols.size();
    FlexBatchTotals totals;
    for (int i = 0; i < nflex; i++){
        totals.add(udflex_mols[i]);
    }
    return predict_gpu_flex(totals, nflex, exhaustiveness);
}


/**
 * @brief The main entrance of docking process. Apply cross-device or multi-threads strategies here.
 *          Avoiding further penetration.
 * @param dpfix_mol
 * @param dpflex_mols
 * @param fns_flex
 * @param dp_out
 * @param dock_param
 * @param device_max_memory
 */
void run_screening(UDFixMol & dpfix_mol, UDFlexMolList &dpflex_mols, const std::vector<std::string>& fns_flex,
                   const std::string &dp_out, DockParam& dock_param, int device_max_memory,
                   const std::vector<Real>& vdw_radii, const AtomTypeProps& type_props, std::string name_base,
                   const std::vector<FFMinData>& ff_data_list, const std::vector<SDFMolData>& sdf_data_list){

    // print docking prameters
    dock_param.show();

    //------------------ Init CUDA RANDOM
    auto groups = group_flex_mols(dpflex_mols);
    spdlog::debug("Group count: {}", groups.size());

    int batch_id = 0;
    int memory_fix = predict_gpu_fix(dpfix_mol);
    device_max_memory -= memory_fix;
    // 10 % headroom for what the predictor does not model (receptor grid maps, polar candidate lists, pinned
    // staging): a batch sized to the full free memory failed at the cluster pair matrix (8,449 ligands, exh 512).
    device_max_memory = static_cast<int>(device_max_memory * 0.9);
    spdlog::info("Batch memory budget: {} MB (90 % of the free memory after the receptor)", device_max_memory);
    std::vector<int> batch_sizes;   // logged at the end: the split must not depend on how the prediction is summed

    // For each group, divide flex_mol_list into small batches according to GPU Mem Limit
    for (int igroup=0; igroup < groups.size(); igroup ++){
        spdlog::info("@@@@@@@@@@@@@@ Tackling Group: {} @@@@@@@@@@@@@@", GroupNames[igroup]);
        std::vector<int> & group = groups[igroup];
        int num_flex_processed = 0;

        // perform batch by batch
        while (num_flex_processed < group.size()) {
            // For each batch
            ++batch_id;
            int batch_size = 0;
            UDFlexMolList batch_flex_mol_list;
            std::vector<std::string> batch_fns_flex;
            std::vector<FFMinData> batch_ff_list;
            std::vector<SDFMolData> batch_sdf_list;
            std::vector<int> batch_rng_slots;   // Advanced.rng_period: input index mod period, per ligand

            int natom_noH = 0;
            FlexBatchTotals batch_totals;   // sums over batch_flex_mol_list, the input of predict_gpu_flex
            while (num_flex_processed + batch_size < group.size()) {
                int imol = num_flex_processed + batch_size;
                int global_idx = group[imol];
                auto& m = dpflex_mols[global_idx];

                // check GPU memory: the batch sums plus this ligand, which is what predict_gpu_flex
                // over the batch with this ligand appended would compute (same += sequence)
                FlexBatchTotals trial = batch_totals;
                trial.add(m);
                if (predict_gpu_flex(trial, batch_size + 1, dock_param.exhaustiveness) < device_max_memory){
                    batch_totals = trial;
                    batch_fns_flex.emplace_back(fns_flex[global_idx]);
                    if (!ff_data_list.empty() && global_idx < static_cast<int>(ff_data_list.size()))
                        batch_ff_list.push_back(ff_data_list[global_idx]);
                    if (!sdf_data_list.empty() && global_idx < static_cast<int>(sdf_data_list.size()))
                        batch_sdf_list.push_back(sdf_data_list[global_idx]);
                    if (dock_param.rng_period > 0)
                        batch_rng_slots.push_back(global_idx % dock_param.rng_period);
                    int natom_noH_tmp = 0;
                    for (auto t : m.vina_types) { if (!type_props.is_hydrogen[t]) natom_noH_tmp++; }
                    if (natom_noH_tmp > natom_noH){
                        natom_noH = natom_noH_tmp;
                    }
                    // moved, not copied: each ligand is read here once, and its receptor pair list is
                    // ~1 MB on a 2,000-atom pocket (run_screening consumes dpflex_mols)
                    batch_flex_mol_list.emplace_back(std::move(m));
                    batch_size++;
                }else{
                    break;
                }
            }

            if (batch_size == 0){
                // A single ligand above the budget used to go through as an empty batch and crash in alloc_device
                throw std::runtime_error("ligand " + fns_flex[group[num_flex_processed]] +
                    " alone exceeds the batch memory budget of " + std::to_string(device_max_memory) + " MB");
            }
            spdlog::info("Batch {} size: {}", batch_id, batch_size);
            batch_sizes.push_back(batch_size);
            num_flex_processed += batch_size;

            // run on this batch. An individual procedure for global_search_with_local_optimize
            spdlog::info("Perform the Task...");
            auto start = std::chrono::high_resolution_clock::now();

            // opt_steps < 0 asks for the heuristic (Vina / upstream Uni-Dock2 rule,
            // (25 + num_atoms) / 3 with the largest ligand of the batch); explicit
            // values are honored. Note: this used to be a hard override to 15.
            // Each batch gets its own budget from its own largest ligand; dock_param keeps the request (-1)
            // so later batches are not stuck with the first batch's value.
            DockParam batch_param = dock_param;
            if (dock_param.opt_steps < 0){
                int natom_max = 0;
                for (const auto& m : batch_flex_mol_list) natom_max = std::max(natom_max, m.natom);
                batch_param.opt_steps = (25 + natom_max) / 3;
                spdlog::info("opt_steps heuristic: (25 + {}) / 3 = {}", natom_max, batch_param.opt_steps);
            }
            DockTask task(dpfix_mol, std::move(batch_flex_mol_list), batch_param, batch_fns_flex, vdw_radii, type_props);
            // Attach per-ligand FF/SDF data (from multi-ligand merged DMS)
            if (!batch_ff_list.empty()) {
                task.ff_min_data_list = std::move(batch_ff_list);
            }
            if (!batch_sdf_list.empty()) {
                task.sdf_mol_list = std::move(batch_sdf_list);
            }
            if (!batch_rng_slots.empty()) {
                task.rng_slots = std::move(batch_rng_slots);
            }
            task.fp_sdf = gen_filepath(
                name_base + "_" + std::to_string(batch_id) + ".sdf", dp_out);
            try {
                task.run();
            } catch (const std::exception& e) {
                // The remaining batches would run on a device left in an unknown state; stop the run here.
                // The CLI turns this into exit code 1, the Python module into a RuntimeError.
                throw std::runtime_error("Batch " + std::to_string(batch_id) + " failed: " + e.what());
            }
            spdlog::info("Task is done.");

            std::chrono::duration<double, std::milli> duration = std::chrono::high_resolution_clock::now() - start;
            spdlog::info("Batch {} running time: {:.1f} ms", batch_id, duration.count());

        }
    }
    std::string split;
    for (size_t b = 0; b < batch_sizes.size(); b++){
        split += (b > 0 ? "/" : "") + std::to_string(batch_sizes[b]);
    }
    spdlog::info("Batch split: {} ({} batches)", split, batch_sizes.size());
}



