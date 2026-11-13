#include <string>
#include <vector>
#include <filesystem>

#if USE_CUDA
#include <cuda_runtime.h>
#endif
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <spdlog/spdlog.h>

#include "model/model.h"
#include "config/parse.h"
#include "format/dms_reader.h"
#include "myutils/errors.h"
#include "screening/screening.h"


namespace py = pybind11;

void RunDockingPipeline(
    // Input parameters (enriched DMS paths)
    std::string ligand_dms_path,
    std::string receptor_dms_path,

    // Output parameter
    std::string output_dir,

    // Settings parameters
    Real center_x,
    Real center_y,
    Real center_z,
    Real size_x,
    Real size_y,
    Real size_z,

    // task type
    std::string task = "screen",

    // search mode
    std::string search_mode = "balance",

    // dock parameters
    int exhaustiveness = 512,
    bool randomize = true,
    int mc_steps = 40,
    int opt_steps = -1,
    int refine_steps = 5,
    int num_pose = 10,
    int rescore_pool = 10,
    Real rmsd_limit = 1.0,
    Real energy_range = 5.0,
    int seed = 1234567,

    // Advanced parameters
    bool constraint_docking = false,
    bool use_tor_lib = false,
    std::vector<std::vector<int>> cluster_atom_indices = {},
    std::vector<std::vector<int>> constrained_atom_indices = {},
    bool decomp = false,
    int ff_minimize_steps = 0,
    int ff_minimize_mode = 0,
    std::string bias = "no",
    Real bias_k = 0.0,

    // Hardware parameters
    int gpu_device_id = 0
) {
    auto start = std::chrono::high_resolution_clock::now();

    // Check DMS files exist
    if (!std::filesystem::exists(ligand_dms_path)) {
        throw std::runtime_error("Ligand DMS file does not exist: " + ligand_dms_path);
    }
    if (!std::filesystem::exists(receptor_dms_path)) {
        throw std::runtime_error("Receptor DMS file does not exist: " + receptor_dms_path);
    }

    // Derive base name from ligand DMS for output naming
    std::string name_base = std::filesystem::path(ligand_dms_path).stem().string();

    // The docking parameters of the arguments: config/parse.cpp param_from_args (plan step 5e), the one parser
    // the binary's YAML goes through as well. The box, the search settings, this route's search_mode presets,
    // FF, bias and the policy; the caller's randomize is honoured (see param_from_args).
    PipelineArgs args;
    args.ligand_dms_path = ligand_dms_path;
    args.receptor_dms_path = receptor_dms_path;
    args.output_dir = output_dir;
    args.center_x = center_x; args.center_y = center_y; args.center_z = center_z;
    args.size_x = size_x; args.size_y = size_y; args.size_z = size_z;
    args.task = task;
    args.search_mode = search_mode;
    args.exhaustiveness = exhaustiveness;
    args.randomize = randomize;
    args.mc_steps = mc_steps;
    args.opt_steps = opt_steps;
    args.refine_steps = refine_steps;
    args.num_pose = num_pose;
    args.rescore_pool = rescore_pool;
    args.rmsd_limit = rmsd_limit;
    args.energy_range = energy_range;
    args.seed = seed;
    args.constraint_docking = constraint_docking;
    args.use_tor_lib = use_tor_lib;
    args.cluster_atom_indices = cluster_atom_indices;
    args.constrained_atom_indices = constrained_atom_indices;
    args.decomp = decomp;
    args.ff_minimize_steps = ff_minimize_steps;
    args.ff_minimize_mode = ff_minimize_mode;
    args.bias = bias;
    args.bias_k = bias_k;
    args.gpu_device_id = gpu_device_id;
    DockParam dock_param = param_from_args(args);

    // Real cutoff for protein box
    Real cutoff = 8.0;
    Box box_protein;
    box_protein.x_lo = dock_param.box.x_lo - cutoff;
    box_protein.x_hi = dock_param.box.x_hi + cutoff;
    box_protein.y_lo = dock_param.box.y_lo - cutoff;
    box_protein.y_hi = dock_param.box.y_hi + cutoff;
    box_protein.z_lo = dock_param.box.z_lo - cutoff;
    box_protein.z_hi = dock_param.box.z_hi + cutoff;

    // Create data structures for molecules
    UDFixMol fix_mol;
    UDFlexMolList flex_mol_list;
    std::vector<std::string> fns_flex;
    std::vector<Real> vdw_radii;
    AtomTypeProps type_props;
    std::vector<FFMinData> ff_data_list;
    std::vector<SDFMolData> sdf_data_list;

    // Initialize atom type properties and vdW radii
    init_default_atom_type_props(type_props);
    vdw_radii.assign(VN_VDW_RADII, VN_VDW_RADII + 21);

    // Read receptor from enriched DMS
    FFMinData ff_tmp_rec;
    read_receptor_dms_full(receptor_dms_path, box_protein, fix_mol, ff_tmp_rec, type_props);

    if (is_multi_ligand_dms(ligand_dms_path)) {
        // Multi-ligand merged DMS
        spdlog::info("Detected multi-ligand merged DMS");
        read_multi_ligand_dms_full(ligand_dms_path, flex_mol_list,
                                   ff_data_list, sdf_data_list,
                                   type_props, use_tor_lib);

        for (auto& flex_mol : flex_mol_list) {
            for (int i = 0; i < flex_mol.natom; i++) {
                if (type_props.is_hydrogen[flex_mol.vina_types[i]]) continue;
                for (int j = 0; j < fix_mol.natom; j++) {
                    if (type_props.is_hydrogen[fix_mol.vina_types[j]]) continue;
                    flex_mol.inter_pairs.push_back(i);
                    flex_mol.inter_pairs.push_back(j);
                }
            }
            fns_flex.push_back(flex_mol.name);
        }

        // No receptor FF copy for ligands without FF tables (FF scoring needs both sides)
        for (auto& ff : ff_data_list) {
            if (!ff.loaded) continue;
            read_receptor_dms(receptor_dms_path, ff);
            if (ff.loaded) continue;
            // Receptor FF table empty or partial: FF is off for every ligand, found out once
            for (auto& other : ff_data_list) other.loaded = false;
            break;
        }
    } else {
        // Single-ligand DMS
        UDFlexMol flex_mol;
        FFMinData ff_tmp_lig;
        read_ligand_dms_full(ligand_dms_path, flex_mol, ff_tmp_lig, type_props, use_tor_lib);

        for (int i = 0; i < flex_mol.natom; i++) {
            if (type_props.is_hydrogen[flex_mol.vina_types[i]]) continue;
            for (int j = 0; j < fix_mol.natom; j++) {
                if (type_props.is_hydrogen[fix_mol.vina_types[j]]) continue;
                flex_mol.inter_pairs.push_back(i);
                flex_mol.inter_pairs.push_back(j);
            }
        }

        fns_flex.push_back(flex_mol.name);
        flex_mol_list.push_back(std::move(flex_mol));
    }

    spdlog::info("Receptor has {:d} atoms in box", fix_mol.natom);
    spdlog::info("Flexible molecules count: {:d}", flex_mol_list.size());
    if (flex_mol_list.size() == 0) {
        spdlog::error("No flexible molecules found");
    }

    // get total memory in MB and leave 5%
    float max_memory = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE) / 1024 / 1024 * 0.95;
#if USE_CUDA
    int deviceCount = 0;
    checkCUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        spdlog::critical("No CUDA device is found!");
        exit(1);
    }
    checkCUDA(cudaSetDevice(gpu_device_id));
    spdlog::info("Set GPU device id to {:d}", gpu_device_id);
    size_t avail, total;
    cudaMemGetInfo(&avail, &total);
    spdlog::info("Available Memory = {:d} MB   Total Memory = {:d} MB",
        avail / 1024 / 1024, total / 1024 / 1024);
    int max_gpu_memory = avail / 1024 / 1024 * 0.95; // leave 5%
    if (max_gpu_memory > 0 && max_gpu_memory < max_memory) {
        max_memory = (float) max_gpu_memory;
    }
#else
    spdlog::info("Running on CPU backend");
    spdlog::info("Available Memory = {:.0f} MB", max_memory);
#endif

    if (!dock_param.constrained_atom_indices_per_flex.empty() &&
        dock_param.constrained_atom_indices_per_flex.size() != flex_mol_list.size()) {
        spdlog::warn("constrained_atom_indices size ({}) != nflex ({}); ignoring.",
                     dock_param.constrained_atom_indices_per_flex.size(), flex_mol_list.size());
        dock_param.constrained_atom_indices_per_flex.clear();
    }

    // Create output directory if it doesn't exist
    if (!std::filesystem::exists(output_dir)) {
        try {
            std::filesystem::create_directories(output_dir);
        } catch (const std::filesystem::filesystem_error& e) {
            spdlog::critical("Failed to create output directory {}: {}", output_dir, e.what());
            exit(1);
        }
    }

    // Run the docking
    if (task == "screen"){
        spdlog::info("----------------------- RUN Screening -----------------------");
        run_screening(fix_mol, flex_mol_list, fns_flex, output_dir, dock_param, max_memory, vdw_radii, type_props, name_base, ff_data_list, sdf_data_list);

    } else if (task == "score"){
        spdlog::info("----------------------- RUN Only Scoring -----------------------");
        set_randomize(dock_param, false);
        dock_param.exhaustiveness = 1;
        dock_param.mc_steps = 0;
        dock_param.opt_steps = 0;
        dock_param.refine_steps = 0;
        dock_param.num_pose = 1;
        dock_param.energy_range = 999;
        dock_param.rmsd_limit = 999;
        run_screening(fix_mol, flex_mol_list, fns_flex, output_dir, dock_param, max_memory, vdw_radii, type_props, name_base, ff_data_list, sdf_data_list);

    } else if (task == "benchmark_one"){
        spdlog::warn("benchmark task is not implemented");

    } else if (task == "mc"){
        set_randomize(dock_param, true);
        dock_param.opt_steps = 0;
        dock_param.refine_steps = 0;
        spdlog::info("----------------------- RUN Only Monte Carlo Random Walking -----------------------");
        run_screening(fix_mol, flex_mol_list, fns_flex, output_dir, dock_param, max_memory, vdw_radii, type_props, name_base, ff_data_list, sdf_data_list);

    } else{
        spdlog::critical("Not supported task: {} doesn't belong to (screen, local_only, mc)", task);
        exit(1);
    }

    std::chrono::duration<double, std::milli> duration = std::chrono::high_resolution_clock::now() - start;
    spdlog::info("UD2 Total Cost: {:.1f} ms", duration.count());
}

// Define Python module
PYBIND11_MODULE(pipeline, m) {
    m.doc() = "Python bindings for the Uni-Dock2 molecular docking engine pipeline";
    // The engine version (VERSION_NUMBER, as `ud2 --version` prints it); unidock_processing.core.engine checks it
    // before passing input an older module would ignore (the ring-bias tables, ud2 0.4.6).
    m.attr("__version__") = VERSION_NUMBER;

    m.def(
        "run_docking_pipeline",
        &RunDockingPipeline,
        "Run docking pipeline using the Uni-Dock2 engine",
        py::arg("ligand_dms_path"),
        py::arg("receptor_dms_path"),
        py::arg("output_dir"),
        py::arg("center_x"),
        py::arg("center_y"),
        py::arg("center_z"),
        py::arg("size_x"),
        py::arg("size_y"),
        py::arg("size_z"),
        py::arg("task") = "screen",
        py::arg("search_mode") = "balance",
        py::arg("exhaustiveness") = 512,
        py::arg("randomize") = true,
        py::arg("mc_steps") = 40,
        py::arg("opt_steps") = -1,
        py::arg("refine_steps") = 5,
        py::arg("num_pose") = 10,
        py::arg("rescore_pool") = 10,
        py::arg("rmsd_limit") = 1.0,
        py::arg("energy_range") = 5.0,
        py::arg("seed") = 1234567,
        py::arg("constraint_docking") = false,
        py::arg("use_tor_lib") = false,
        py::arg("cluster_atom_indices") = std::vector<std::vector<int>>{},
        py::arg("constrained_atom_indices") = std::vector<std::vector<int>>{},
        py::arg("decomp") = false,
        py::arg("ff_minimize_steps") = 0,
        py::arg("ff_minimize_mode") = 0,
        py::arg("bias") = "no",
        py::arg("bias_k") = 0.0,
        py::arg("gpu_device_id") = 0
    );
}
