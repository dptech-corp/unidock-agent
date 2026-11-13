//
// The one config parser of the engine (plan step 5e): the YAML of the ud2 binary and the arguments of the Python
// pipeline both become a DockParam here, so a new block or key is added once. parse_config carries every key,
// default, range check, warning and log line that bin/unidock/main.cpp read until step 5; param_from_args carries
// the argument handling of api/python/pipeline.cpp (its own search_mode presets, the caller's randomize honoured).
// The bias has its own top-level block `bias: {type: no|pos|align, k: <Real>}`, with the optional sub-block
// `ring: {k, s, d0}` of the aromatic-ring term (defaults 30, 3, 1; plan step 6); Advanced.bias / Advanced.bias_k
// are read as aliases for one release, with a deprecation warning (the block wins when both are given). The pipeline
// arguments carry no ring parameters: param_from_args keeps their defaults.
//

#ifndef UD2_CONFIG_PARSE_H
#define UD2_CONFIG_PARSE_H

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "model/model.h"

/// The items of the engine YAML that are not DockParam fields: what main.cpp needs besides the DockParam.
struct RunConfig {
    std::string task = "screen";       // Settings.task
    std::string ligand_dms_full;       // Inputs.ligand_dms_full
    std::string receptor_dms_full;     // Inputs.receptor_dms_full
    std::string output_dir;            // Outputs.dir
    int gpu_device_id = 0;             // Hardware.gpu_device_id (read in the CUDA build only, as before)
};

/// Parse the engine YAML. Exits the process (spdlog critical / error, exit code 1) on what main.cpp exited on:
/// an unreadable item, a missing input pair, an unknown bias type or search_mode, a bad rng_period or
/// ff_minimize_mode.
DockParam parse_config(const YAML::Node& config, RunConfig& run);

/// The arguments of RunDockingPipeline (api/python/pipeline.cpp), in its order and with its defaults.
struct PipelineArgs {
    std::string ligand_dms_path;
    std::string receptor_dms_path;
    std::string output_dir;
    Real center_x = 0, center_y = 0, center_z = 0;
    Real size_x = 0, size_y = 0, size_z = 0;
    std::string task = "screen";
    std::string search_mode = "balance";
    int exhaustiveness = 512;
    bool randomize = true;
    int mc_steps = 40;
    int opt_steps = -1;
    int refine_steps = 5;
    int num_pose = 10;
    int rescore_pool = 10;
    Real rmsd_limit = 1.0;
    Real energy_range = 5.0;
    int seed = 1234567;
    bool constraint_docking = false;
    bool use_tor_lib = false;
    std::vector<std::vector<int>> cluster_atom_indices;
    std::vector<std::vector<int>> constrained_atom_indices;
    bool decomp = false;
    int ff_minimize_steps = 0;
    int ff_minimize_mode = 0;
    std::string bias = "no";
    Real bias_k = 0.0;
    int gpu_device_id = 0;
};

/// The DockParam of the pipeline arguments (the box, the search settings, the pipeline's search_mode presets,
/// FF, bias, policy, the index lists as given). Exits (critical) on an unknown bias type / search_mode or a bad
/// ff_minimize_mode, as the pipeline did. The nflex check of constrained_atom_indices stays with the caller.
DockParam param_from_args(const PipelineArgs& args);

// ---- shared by both parsers: the fields and the SearchPolicy they mirror are set together ----

/// BT_NO / BT_POS / BT_ALIGN for "no" / "pos" / "align"; anything else is the critical error it always was (exit 1).
BiasType bias_type_from_name(const std::string& name);
/// bias_type / bias_k and policy.keep_template_hydrogens (a bias keeps the template hydrogens).
void set_bias(DockParam& dp, BiasType type, Real k);
/// constraint_docking and policy.static_root / keep_template_hydrogens.
void set_constraint_docking(DockParam& dp, bool constraint_docking);
/// randomize and policy.randomize.
void set_randomize(DockParam& dp, bool randomize);
/// policy.use_torsion_ranges (Advanced.tor_lib / use_tor_lib).
void set_tor_lib(DockParam& dp, bool use_tor_lib);

#endif // UD2_CONFIG_PARSE_H
