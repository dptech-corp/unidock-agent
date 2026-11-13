#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>
#include <vector> // ligand paths
#include <algorithm> // std::any_of
#if USE_CUDA
#include <cuda_runtime.h>
#endif
#include <yaml-cpp/yaml.h>

#include <unistd.h>
#if USE_OPENMP
#include <omp.h>
#endif
#include "myutils/errors.h"
#include "model/model.h"
#include "config/parse.h"
#include "format/dms_reader.h"
#include "screening/screening.h"
#include "main.h"


void print_help(){
    const char* STR_HELP = R"(
Usage: ud2 [options] [config_path]

Options:
  --version             Show version information
  --help                Show this help message
  --dump_config [path]  Dump default config to file (default: default.yaml)
  --log <path>          Specify log file path, only take effects when [config_path] is given
)";
    std::cout << STR_HELP;
}

void print_sign() {
    // ANSI Shadow
        std::cout << R"(
    ██╗ccc██╗██████╗c██████╗c
    ██║ccc██║██╔══██╗╚════██╗
    ██║ccc██║██║cc██║c█████╔╝
    ██║ccc██║██║cc██║██╔═══╝c
    ╚██████╔╝██████╔╝███████╗
    c╚═════╝c╚═════╝c╚══════╝
    )" << std::endl;

}

void print_version(){
    std::cout << "UD2 C++ Engine Version: " << VERSION_NUMBER << "\n";
}


void dump_config_template(const std::string& p){
    std::ofstream f(p);
    if (!f){
        std::cout << "Failed to create config template file: " << p.c_str() << "\n";
        exit(1);
    }

    f << STR_CONFIG_TEMPLATE;
}



int main(int argc, char* argv[])
{
#ifdef DEBUG
    int log_level = 0;
#else
    int log_level = 1;
#endif
    std::string fp_log = "ud.log"; // default under working directory
    std::string fp_config;

    // parse arguments
    for (int i = 1; i < argc; ++i){
        std::string arg = argv[i];
        if (arg == "--version"){
            print_version();
            return 0;
        }
        if (arg == "--help"){
            print_help();
            return 0;
        }
        if (arg == "--dump_config"){
            std::string fp_dump = "default.yaml";
            if ((i + 1 < argc) && (argv[i + 1][0] != '-')){
                fp_dump = argv[++i];
            }
            dump_config_template(fp_dump);
            return 0;
        }

        if (arg == "--log"){
            if (argc > i + 1){
                fp_log = argv[i + 1];
                i += 1; // log parameter done
            } else{
                print_help();
                printf("--log requires a log path\n");
                return 1;
            }
        } else{
            if (fp_config.empty()){
                fp_config = arg;
                i ++;
            } else{
                print_help();
                printf("After config path is parsed as %s, unexpected argument: %s\n", fp_config.c_str(), arg.c_str());
                return 1;
            }
        }
    }

    if (fp_config.empty()){
        print_help();
        printf("Missing argument: config file path\n");
        exit(1);
    }

    init_logger(fp_log, log_level);
    spdlog::info("Using config file: {}", fp_config);

    print_sign();
    print_version();

    spdlog::info("==================== UD2 Starts! ======================\n");
    auto start = std::chrono::high_resolution_clock::now();

    YAML::Node config = YAML::LoadFile(fp_config);

    // Every key, default, check and warning of the YAML: config/parse.cpp (plan step 5e)
    RunConfig run;
    DockParam dock_param = parse_config(config, run);
    const int ncpu = dock_param.ncpu;
#if USE_OPENMP
    omp_set_num_threads(ncpu);
    spdlog::info("OpenMP threads set to {}", ncpu);
#endif
    const std::string& task = run.task;
    const std::string& fp_lig_dms_full = run.ligand_dms_full;
    const std::string& fp_rec_dms_full = run.receptor_dms_full;
    const bool use_tor_lib = dock_param.policy.use_torsion_ranges;

    UDFlexMolList flex_mol_list;
    UDFixMol fix_mol;
    std::vector<std::string> fns_flex;
    std::vector<Real> vdw_radii;
    AtomTypeProps type_props;

    Real cutoff = 8.0;
    Box box_protein;
    box_protein.x_lo = dock_param.box.x_lo - cutoff;
    box_protein.x_hi = dock_param.box.x_hi + cutoff;
    box_protein.y_lo = dock_param.box.y_lo - cutoff;
    box_protein.y_hi = dock_param.box.y_hi + cutoff;
    box_protein.z_lo = dock_param.box.z_lo - cutoff;
    box_protein.z_hi = dock_param.box.z_hi + cutoff;

    // Input source name for output file naming
    std::string name_base;

    // Per-ligand FF and SDF data (populated only for multi-ligand merged DMS)
    std::vector<FFMinData> ff_data_list;
    std::vector<SDFMolData> sdf_data_list;

    {
        // ===== DMS input path (parse_config has required both Inputs paths) =====
        spdlog::info("Using DMS-only input path");
        init_default_atom_type_props(type_props);
        vdw_radii.assign(VN_VDW_RADII, VN_VDW_RADII + 21);

        // Read receptor from enriched DMS
        FFMinData ff_tmp_rec;
        read_receptor_dms_full(fp_rec_dms_full, box_protein, fix_mol, ff_tmp_rec, type_props);

        if (is_multi_ligand_dms(fp_lig_dms_full)) {
            // ===== Multi-ligand merged DMS =====
            spdlog::info("Detected multi-ligand merged DMS");
            read_multi_ligand_dms_full(fp_lig_dms_full, flex_mol_list,
                                       ff_data_list, sdf_data_list,
                                       type_props, use_tor_lib);

            // Compute inter_pairs for each ligand
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

            // Read receptor FF into each per-ligand FFMinData. Skipped when the ligand DMS has no
            // FF tables: FF scoring needs both sides, and screening libraries would otherwise
            // hold one copy of the whole receptor per ligand.
            for (auto& ff : ff_data_list) {
                if (!ff.loaded) continue;
                read_receptor_dms(fp_rec_dms_full, ff);
                if (ff.loaded) continue;
                // Receptor FF table empty or partial: FF is off for every ligand, found out once
                for (auto& other : ff_data_list) other.loaded = false;
                break;
            }

        } else {
            // ===== Single-ligand DMS =====
            UDFlexMol flex_mol;
            FFMinData ff_tmp_lig;
            read_ligand_dms_full(fp_lig_dms_full, flex_mol, ff_tmp_lig, type_props, use_tor_lib);

            // Compute inter_pairs: flex vs fix (data-driven H check)
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

            // Read SDF mol data for output (bonds, formal charges, etc.)
            SDFMolData sdf_mol_single;
            read_sdf_mol_data(fp_lig_dms_full, sdf_mol_single);
            if (sdf_mol_single.natom > 0) {
                sdf_data_list.push_back(std::move(sdf_mol_single));
            }
        }

        // Name for output
        auto p = fp_lig_dms_full.find_last_of("/\\");
        name_base = (p == std::string::npos) ? fp_lig_dms_full : fp_lig_dms_full.substr(p + 1);
        auto dot = name_base.rfind('.');
        if (dot != std::string::npos) name_base = name_base.substr(0, dot);
    }

    spdlog::info("Receptor has {:d} atoms in box", fix_mol.natom);
    spdlog::info("Flexible molecules count: {:d}", flex_mol_list.size());
    if (flex_mol_list.size() == 0){
        spdlog::error("No flexible molecules found");
    }

    // get total memory in MB and leave 5%
    float max_memory = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE) / 1024 / 1024 * 0.95;
    
#if USE_CUDA
    int deviceCount = 0;
    checkCUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount > 0) {
        const int device_id = run.gpu_device_id;
        checkCUDA(cudaSetDevice(device_id));
        spdlog::info("Set GPU device id to {:d}", device_id);
        size_t avail, total;
        cudaMemGetInfo(&avail, &total);
        spdlog::info("Available Memory = {:d} MB   Total Memory = {:d} MB",
            avail / 1024 / 1024, total / 1024 / 1024);
        int max_gpu_memory = avail / 1024 / 1024 * 0.95; // leave 5%
        if (max_gpu_memory > 0 && max_gpu_memory < max_memory) {
            max_memory = (float) max_gpu_memory;
        }
    } else{
        spdlog::critical("No CUDA device is found!");
        exit(1);
    }
#else
    // CPU backend - use system memory
    spdlog::info("Running on CPU backend");
    spdlog::info("Available Memory = {:.0f} MB", max_memory);
#endif

    // Merged multi-ligand input: the per-ligand FF was read from the Inputs files above
    if (std::any_of(ff_data_list.begin(), ff_data_list.end(), [](const FFMinData& ff) { return ff.loaded; })) {
        spdlog::info("FF parameter source: ligand {} / receptor {}", fp_lig_dms_full, fp_rec_dms_full);
    }


    // -------------------------------  Perform Task -------------------------------
    const std::string& dp_out = run.output_dir;
    if (!std::filesystem::exists(dp_out)) {
        try {
            std::filesystem::create_directories(dp_out);
        } catch (const std::filesystem::filesystem_error& e) {
            spdlog::critical("Failed to create output directory {}: {}", dp_out, e.what());
            exit(1);
        }
    }

    try {
    if (task == "screen"){ // allow changing every parameter
        spdlog::info("----------------------- RUN Screening -----------------------");
        run_screening(fix_mol, flex_mol_list, fns_flex, dp_out, dock_param, max_memory, vdw_radii, type_props, name_base, ff_data_list, sdf_data_list);

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
        run_screening(fix_mol, flex_mol_list, fns_flex, dp_out, dock_param, max_memory, vdw_radii, type_props, name_base, ff_data_list, sdf_data_list);

    } else if (task == "benchmark_one"){
        spdlog::warn("benchmark task is not implemented");
        spdlog::info("----------------------- RUN Benchmark on One-Crystal-Ligand Cases -----------------------");
        spdlog::info("----------------------- Given poses are deemed as reference poses -----------------------");

    } else if (task == "mc"){
        set_randomize(dock_param, true);
        dock_param.opt_steps = 0;
        dock_param.refine_steps = 0;
        spdlog::info("----------------------- RUN Only Monte Carlo Random Walking (With Clustering) -----------------------");
        run_screening(fix_mol, flex_mol_list, fns_flex, dp_out, dock_param, max_memory, vdw_radii, type_props, name_base, ff_data_list, sdf_data_list);

    } else if (task == "randomize"){
        set_randomize(dock_param, true);
        dock_param.mc_steps = 0;
        dock_param.opt_steps = 0;
        dock_param.refine_steps = 0;
        dock_param.num_pose = dock_param.exhaustiveness;
        dock_param.energy_range = 1e9;
        dock_param.rmsd_limit = 0.;
        spdlog::info("----------------------- RUN Only Randomization (No Clustering) -----------------------");
        run_screening(fix_mol, flex_mol_list, fns_flex, dp_out, dock_param, max_memory, vdw_radii, type_props, name_base, ff_data_list, sdf_data_list);

    } else if (task == "optimize"){
        set_randomize(dock_param, false);
        dock_param.exhaustiveness = 1;
        dock_param.mc_steps = 0;
        dock_param.opt_steps = 0;
        dock_param.num_pose = 1;
        dock_param.energy_range = 1e9;
        dock_param.rmsd_limit = 0.;
        spdlog::info("----------------------- RUN Only Optimization on Input Pose (for `refine_steps) -----------------------");
        run_screening(fix_mol, flex_mol_list, fns_flex, dp_out, dock_param, max_memory, vdw_radii, type_props, name_base, ff_data_list, sdf_data_list);

    } else{
        spdlog::critical("Not supported task: {} doesn't belong to (screen, local_only, mc)", task);
        exit(1);
    }
    } catch (const std::exception& e) {
        spdlog::critical("Run aborted: {}", e.what());
        return 1;
    }

    std::chrono::duration<double, std::milli> duration = std::chrono::high_resolution_clock::now() - start;
    spdlog::info("UD2 Total Cost: {:.1f} ms", duration.count());

    return 0;
}
