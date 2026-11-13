//
// The engine's config parser: parse_config (the YAML of the ud2 binary, every read that bin/unidock/main.cpp made
// until plan step 5, in its order) and param_from_args (the argument handling of api/python/pipeline.cpp). See
// parse.h. The DockParam fields and the SearchPolicy that mirrors some of them are set through the set_* helpers,
// so the two never disagree.
//

#include "config/parse.h"

#include <algorithm>
#include <cstdlib>
#include <thread>
#include <spdlog/spdlog.h>

// ===================== shared helpers =====================

BiasType bias_type_from_name(const std::string& bias){
    if (bias == "no"){
        return BT_NO;
    } else if (bias == "pos"){
        return BT_POS;
    } else if (bias == "align"){
        return BT_ALIGN;
    }
    spdlog::critical("Not supported bias: {} doesn't belong to (no, pos, align)", bias);
    exit(1);
}

void set_bias(DockParam& dp, BiasType type, Real k){
    dp.bias_type = type;
    dp.bias_k = k;
    dp.policy.keep_template_hydrogens = dp.policy.static_root || type != BT_NO;
}

void set_constraint_docking(DockParam& dp, bool constraint_docking){
    dp.constraint_docking = constraint_docking;
    dp.policy.static_root = constraint_docking;
    dp.policy.keep_template_hydrogens = constraint_docking || dp.bias_type != BT_NO;
}

void set_randomize(DockParam& dp, bool randomize){
    dp.randomize = randomize;
    dp.policy.randomize = randomize;
}

void set_tor_lib(DockParam& dp, bool use_tor_lib){
    dp.policy.use_torsion_ranges = use_tor_lib;
}

/// Scoring-function term weights. The defaults are the fitted sets (model.h default_*_weights):
///   Settings.score_weights    : the search set (Monte Carlo + local optimisation)
///   Settings.rescore_weights  : the pose pass that orders the output poses (vina_pose_score)
///   Settings.affinity_weights : the reported affinity (vina_affinity_score)
/// Each YAML block overrides single keys of its set; the other keys keep the default.
/// Keys: gauss1, gauss2, repulsion, hydrophobic, hbond, intra, elec, elec_charged_scale,
///       polar_env, arom_contact, torsion_strain.
static void read_weights(const YAML::Node& sw, VinaWeights& w){
    if (sw["gauss1"]) w.gauss1 = sw["gauss1"].as<Real>();
    if (sw["gauss2"]) w.gauss2 = sw["gauss2"].as<Real>();
    if (sw["repulsion"]) w.repulsion = sw["repulsion"].as<Real>();
    if (sw["hydrophobic"]) w.hydrophobic = sw["hydrophobic"].as<Real>();
    if (sw["hbond"]) w.hbond = sw["hbond"].as<Real>();
    if (sw["intra"]) w.intra = sw["intra"].as<Real>();
    if (sw["elec"]) w.elec = sw["elec"].as<Real>();
    if (sw["elec_charged_scale"]) w.elec_charged_scale = sw["elec_charged_scale"].as<Real>();
    if (sw["polar_env"]) w.polar_env = sw["polar_env"].as<Real>();
    if (sw["arom_contact"]) w.arom_contact = sw["arom_contact"].as<Real>();
    if (sw["torsion_strain"]) w.torsion_strain = sw["torsion_strain"].as<Real>();
}

static void log_weights(const char* label, const VinaWeights& w){
    spdlog::info("{}: gauss1 {} gauss2 {} repulsion {} hydrophobic {} hbond {} intra {} elec {} "
                 "elec_charged_scale {} polar_env {} arom_contact {} torsion_strain {}",
                 label, w.gauss1, w.gauss2, w.repulsion, w.hydrophobic, w.hbond, w.intra, w.elec,
                 w.elec_charged_scale, w.polar_env, w.arom_contact, w.torsion_strain);
}

static const char* bias_name(BiasType type){
    return type == BT_ALIGN ? "align" : (type == BT_POS ? "pos" : "no");
}

/// The resolved bias block, logged at start like the weight sets (the ring parameters apply to ring rows only).
static void log_bias(const DockParam& dp){
    spdlog::info("Bias: type {} k {} ring k {} s {} d0 {}", bias_name(dp.bias_type), dp.bias_k,
                 dp.ring_bias.k, dp.ring_bias.s, dp.ring_bias.d0);
}

/// The optional sub-block `bias: {ring: {k, s, d0}}` of the aromatic-ring bias (ext/bias/ring_term.h): each key given
/// overrides the default of RingBiasParams (RING_BIAS_K / _S / _D0: 30, 3, 1). Exits (critical) on a block that is
/// not a map or an unreadable value, as for the other items.
static void read_ring_bias_params(const YAML::Node& ring, RingBiasParams& p){
    if (!ring.IsMap()){
        spdlog::critical("The ring block of bias must be a map: bias: {{ring: {{k: <number>, s: <number>, d0: <number>}}}}");
        exit(1);
    }
    for (const auto& item : {std::make_pair("k", &p.k), std::make_pair("s", &p.s), std::make_pair("d0", &p.d0)}){
        if (!ring[item.first]) continue;
        try {
            *item.second = ring[item.first].as<Real>();
        } catch (const YAML::Exception& e) {
            spdlog::critical("Failed to read config item bias.ring.{}: {}", item.first, e.what());
            exit(1);
        }
    }
}


// ===================== the YAML of the binary =====================

template<typename T>
static T get_config_with_err(const YAML::Node& config, const std::string& section, const std::string& key,
                             const T& default_value = T()) {
    try {
        if (!config[section] || !config[section][key]) {
            spdlog::warn("Config item {}.{} doesn't exist, using default value", section, key);
            return default_value;
        }
        return config[section][key].as<T>();
    } catch (const YAML::Exception& e) {
        spdlog::critical("Failed to read config item {}.{}: {}", section, key, e.what());
        exit(1);
    }
}

DockParam parse_config(const YAML::Node& config, RunConfig& run){
    int mc_steps = 0;
    DockParam dock_param;
    int ncpu = std::thread::hardware_concurrency();

    // Load ncpu from config (the caller sets the OpenMP threads)
    ncpu = get_config_with_err<int>(config, "Hardware", "ncpu", ncpu);
    dock_param.ncpu = ncpu;

    // -------------------------------  Parse Advanced -------------------------------
    dock_param.exhaustiveness = get_config_with_err<int>(config, "Advanced", "exhaustiveness", dock_param.exhaustiveness);;
    set_randomize(dock_param, get_config_with_err<bool>(config, "Advanced", "randomize", dock_param.randomize));
    dock_param.mc_steps = get_config_with_err<int>(config, "Advanced", "mc_steps", mc_steps);
    dock_param.opt_steps = get_config_with_err<int>(config, "Advanced", "opt_steps", dock_param.opt_steps);
    if (dock_param.opt_steps < 0){ //heuristic
        dock_param.opt_steps = -1;
        spdlog::info("Use heuristic method to decide opt_steps");
    }

    dock_param.refine_steps = get_config_with_err<int>(config, "Advanced", "refine_steps", dock_param.refine_steps);
    dock_param.use_grid = get_config_with_err<bool>(config, "Advanced", "use_grid", dock_param.use_grid);
    dock_param.grid_spacing = get_config_with_err<Real>(config, "Advanced", "grid_spacing", dock_param.grid_spacing);
    dock_param.grid_margin = get_config_with_err<Real>(config, "Advanced", "grid_margin", dock_param.grid_margin);

    // box
    Real center_x = get_config_with_err<Real>(config, "Settings", "center_x");
    Real center_y = get_config_with_err<Real>(config, "Settings", "center_y");
    Real center_z = get_config_with_err<Real>(config, "Settings", "center_z");
    Real size_x = get_config_with_err<Real>(config, "Settings", "size_x");
    Real size_y = get_config_with_err<Real>(config, "Settings", "size_y");
    Real size_z = get_config_with_err<Real>(config, "Settings", "size_z");
    dock_param.box.x_lo = center_x - size_x / 2;
    dock_param.box.x_hi = center_x + size_x / 2;
    dock_param.box.y_lo = center_y - size_y / 2;
    dock_param.box.y_hi = center_y + size_y / 2;
    dock_param.box.z_lo = center_z - size_z / 2;
    dock_param.box.z_hi = center_z + size_z / 2;
    // Settings.inner_box: edge length (A) of an inner box around the box center that the ligand
    // heavy-atom centroid must stay in; 0 = off (10 A is the benchmark protocol).
    Real inner_box_size = get_config_with_err<Real>(config, "Settings", "inner_box", Real(0));
    if (inner_box_size > 0){
        dock_param.use_inner_box = true;
        dock_param.inner_box.x_lo = std::max(center_x - inner_box_size / 2, dock_param.box.x_lo);
        dock_param.inner_box.x_hi = std::min(center_x + inner_box_size / 2, dock_param.box.x_hi);
        dock_param.inner_box.y_lo = std::max(center_y - inner_box_size / 2, dock_param.box.y_lo);
        dock_param.inner_box.y_hi = std::min(center_y + inner_box_size / 2, dock_param.box.y_hi);
        dock_param.inner_box.z_lo = std::max(center_z - inner_box_size / 2, dock_param.box.z_lo);
        dock_param.inner_box.z_hi = std::min(center_z + inner_box_size / 2, dock_param.box.z_hi);
        spdlog::info("Inner box: ligand heavy-atom centroid kept within {:.1f} A of the box center", inner_box_size / 2);
    }

    run.task = get_config_with_err<std::string>(config, "Settings", "task", "screen");

    // Input
    run.ligand_dms_full = get_config_with_err<std::string>(config, "Inputs", "ligand_dms_full", std::string(""));
    run.receptor_dms_full = get_config_with_err<std::string>(config, "Inputs", "receptor_dms_full", std::string(""));

    bool use_tor_lib = get_config_with_err<bool>(config, "Advanced", "tor_lib", true);
    if (not use_tor_lib){
        spdlog::warn("Torsion Library is NOT used.");
    }
    set_tor_lib(dock_param, use_tor_lib);
    dock_param.tor_prec = get_config_with_err<Real>(config, "Advanced", "tor_prec", dock_param.tor_prec);
    dock_param.box_prec = get_config_with_err<Real>(config, "Advanced", "box_prec", dock_param.box_prec);

    if (run.ligand_dms_full.empty() || run.receptor_dms_full.empty()) {
        spdlog::critical("No input specified: need Inputs.ligand_dms_full + Inputs.receptor_dms_full");
        exit(1);
    }
    // DMS paths for DockTask to load FF params during run() (single-ligand uses these; multi-ligand has
    // pre-loaded ff_data_list); Advanced.ligand_dms / protein_dms below override them when given
    dock_param.ligand_dms_path = run.ligand_dms_full;
    dock_param.protein_dms_path = run.receptor_dms_full;

#if USE_CUDA
    run.gpu_device_id = get_config_with_err<int>(config, "Hardware", "gpu_device_id", 0);
#endif

    // Advanced
    dock_param.num_pose = get_config_with_err<int>(config, "Advanced", "num_pose", dock_param.num_pose);
    dock_param.rescore_pool = get_config_with_err<int>(config, "Advanced", "rescore_pool", dock_param.rescore_pool);
    dock_param.rmsd_limit = get_config_with_err<Real>(config, "Advanced", "rmsd_limit", dock_param.rmsd_limit);
    dock_param.energy_range = get_config_with_err<Real>(config, "Advanced", "energy_range", dock_param.energy_range);
    dock_param.seed = get_config_with_err<int>(config, "Advanced", "seed", dock_param.seed);
    // read only when present: an absent key would add a "doesn't exist" warning to every default run
    if (config["Advanced"] && config["Advanced"]["rng_period"]) {
        dock_param.rng_period = get_config_with_err<int>(config, "Advanced", "rng_period", dock_param.rng_period);
    }
    if (dock_param.rng_period < 0){
        spdlog::critical("Advanced.rng_period must be >= 0 (0 = off), got {}", dock_param.rng_period);
        exit(1);
    }
    if (dock_param.rng_period > 0){
        spdlog::info("rng_period = {}: input ligand k draws the MC random streams of ligand k mod {}",
                     dock_param.rng_period, dock_param.rng_period);
    }

    // Bias: the own top-level block `bias: {type: no|pos|align, k: <Real>, ring: {k, s, d0}}` (ring optional, the
    // aromatic-ring term's parameters). Advanced.bias / Advanced.bias_k are
    // read as aliases for one release (the same reads as before, so a run without them logs what it logged); the
    // block wins when both are given.
    std::string bias = get_config_with_err<std::string>(config, "Advanced", "bias", "no");
    Real bias_k = get_config_with_err<Real>(config, "Advanced", "bias_k", dock_param.bias_k);
    const bool bias_alias_given = config["Advanced"] && (config["Advanced"]["bias"] || config["Advanced"]["bias_k"]);
    if (bias_alias_given){
        spdlog::warn("Advanced.bias / Advanced.bias_k are deprecated and read for one more release: "
                     "use the top-level block `bias: {{type: {}, k: {}}}`", bias, bias_k);
    }
    if (config["bias"]){
        if (!config["bias"].IsMap()){
            spdlog::critical("The bias block must be a map: bias: {{type: no|pos|align, k: <number>}}");
            exit(1);
        }
        if (bias_alias_given){
            spdlog::warn("Both the bias block and Advanced.bias / Advanced.bias_k are given; the bias block wins");
        }
        if (config["bias"]["type"]) bias = get_config_with_err<std::string>(config, "bias", "type", bias);
        if (config["bias"]["k"]) bias_k = get_config_with_err<Real>(config, "bias", "k", bias_k);
        if (config["bias"]["ring"]) read_ring_bias_params(config["bias"]["ring"], dock_param.ring_bias);
    }
    set_bias(dock_param, bias_type_from_name(bias), bias_k);

    // Per-ligand atom indices for clustering RMSD (optional)
    if (config["Advanced"] && config["Advanced"]["cluster_atom_indices"]) {
        const auto& node = config["Advanced"]["cluster_atom_indices"];
        if (node.IsSequence()) {
            for (size_t i = 0; i < node.size(); i++) {
                std::vector<int> inds;
                if (node[i].IsSequence()) {
                    for (size_t j = 0; j < node[i].size(); j++) {
                        inds.push_back(node[i][j].as<int>());
                    }
                }
                // null / empty YAML entry => empty vector => heavy-atom default
                dock_param.cluster_atom_indices_per_flex.push_back(std::move(inds));
            }
            spdlog::info("Parsed cluster_atom_indices for {} ligands",
                         dock_param.cluster_atom_indices_per_flex.size());
        }
    }

    // -------------------------------  Parse Settings -------------------------------
    std::string search_mode = get_config_with_err<std::string>(config, "Settings", "search_mode", "balance");
    if (search_mode == "fast"){
#if USE_CUDA
        dock_param.exhaustiveness = 128;
#else
        dock_param.exhaustiveness = 64;
#endif
        dock_param.mc_steps = 20;
    } else if (search_mode == "balance"){
#if USE_CUDA
        dock_param.exhaustiveness = 256;
#else
        dock_param.exhaustiveness = 128;
#endif
        dock_param.mc_steps = 30;
    } else if (search_mode == "detail"){
#if USE_CUDA
        dock_param.exhaustiveness = 512;
#else
        dock_param.exhaustiveness = 256;
#endif
        dock_param.mc_steps = 40;
    } else if (search_mode == "free"){
        //
    } else{
        spdlog::critical("Not supported search_mode: {} doesn't belong to (fast, balance, detail, free)" , search_mode);
        exit(1);
    }

    if (dock_param.exhaustiveness < ncpu) {
        spdlog::warn("Low exhaustiveness doesn't utilize all CPUs");
    }

    set_constraint_docking(dock_param, get_config_with_err<bool>(config, "Settings", "constraint_docking", false));
    if (dock_param.constraint_docking){
        set_randomize(dock_param, false);
    }
    dock_param.decomp = get_config_with_err<bool>(config, "Settings", "decomp", false);
    dock_param.angular_hbond = get_config_with_err<bool>(config, "Settings", "angular_hbond", true);

    if (config["Settings"] && config["Settings"]["score_weights"]) {
        read_weights(config["Settings"]["score_weights"], dock_param.weights);
    }
    if (config["Settings"] && config["Settings"]["rescore_weights"]) {
        read_weights(config["Settings"]["rescore_weights"], dock_param.rescore_weights);
    }
    if (config["Settings"] && config["Settings"]["affinity_weights"]) {
        read_weights(config["Settings"]["affinity_weights"], dock_param.affinity_weights);
    }
    log_weights("Score weights", dock_param.weights);
    log_weights("Rescore weights", dock_param.rescore_weights);
    log_weights("Affinity weights", dock_param.affinity_weights);
    // the resolved bias block, logged at start like the weight sets
    log_bias(dock_param);

    // FF minimization (optional) and FF scoring (when DMS paths provided)
    dock_param.ff_minimize_steps = get_config_with_err<int>(config, "Advanced", "ff_minimize_steps", 0);
    int ff_mode_int = get_config_with_err<int>(config, "Advanced", "ff_minimize_mode", 0);
    if (ff_mode_int < 0 || ff_mode_int > 1) {
        spdlog::error("ff_minimize_mode must be 0 (internal) or 1 (cartesian)");
        exit(1);
    }
    dock_param.ff_minimize_mode = static_cast<FFMinimizeMode>(ff_mode_int);
    if (dock_param.ff_minimize_mode == FFMIN_CARTESIAN && dock_param.ff_minimize_steps <= 0) {
        spdlog::error("ff_minimize_mode=cartesian requires ff_minimize_steps > 0");
        exit(1);
    }
    // FF parameter source. Single-ligand input: Advanced.ligand_dms / protein_dms when given, else
    // the Inputs DMS files; DockTask::run reads them and logs the source when the FF loads.
    // Merged multi-ligand input: the per-ligand FF was read from the Inputs files (main.cpp logs the source).
    const std::string adv_ligand_dms = get_config_with_err<std::string>(config, "Advanced", "ligand_dms", std::string(""));
    const std::string adv_protein_dms = get_config_with_err<std::string>(config, "Advanced", "protein_dms", std::string(""));
    if (!adv_ligand_dms.empty()) dock_param.ligand_dms_path = adv_ligand_dms;
    if (!adv_protein_dms.empty()) dock_param.protein_dms_path = adv_protein_dms;
    if (!dock_param.ligand_dms_path.empty() && !dock_param.protein_dms_path.empty()) {
        if (dock_param.ff_minimize_steps > 0) {
            spdlog::info("FF minimization enabled: {} steps", dock_param.ff_minimize_steps);
        }
    } else if (dock_param.ff_minimize_steps > 0) {
        spdlog::warn("FF minimization requires both ligand_dms and protein_dms paths; disabling.");
        dock_param.ff_minimize_steps = 0;
    }

    // Output
    run.output_dir = get_config_with_err<std::string>(config, "Outputs", "dir");

    return dock_param;
}


// ===================== the arguments of the Python pipeline =====================

DockParam param_from_args(const PipelineArgs& args){
    DockParam dock_param;

    // Set box parameters from center and size
    dock_param.box.x_lo = args.center_x - args.size_x / 2;
    dock_param.box.x_hi = args.center_x + args.size_x / 2;
    dock_param.box.y_lo = args.center_y - args.size_y / 2;
    dock_param.box.y_hi = args.center_y + args.size_y / 2;
    dock_param.box.z_lo = args.center_z - args.size_z / 2;
    dock_param.box.z_hi = args.center_z + args.size_z / 2;

    // Set DMS paths for DockTask
    dock_param.ligand_dms_path = args.ligand_dms_path;
    dock_param.protein_dms_path = args.receptor_dms_path;

    // Set docking parameters
    dock_param.exhaustiveness = args.exhaustiveness;
    set_randomize(dock_param, args.randomize);
    dock_param.mc_steps = args.mc_steps;
    dock_param.opt_steps = args.opt_steps;
    dock_param.refine_steps = args.refine_steps;
    dock_param.num_pose = args.num_pose;
    dock_param.rescore_pool = args.rescore_pool;
    dock_param.rmsd_limit = args.rmsd_limit;
    dock_param.energy_range = args.energy_range;
    dock_param.seed = args.seed;
    dock_param.ff_minimize_steps = args.ff_minimize_steps;
    if (args.ff_minimize_mode < 0 || args.ff_minimize_mode > 1) {
        spdlog::critical("ff_minimize_mode must be 0 (internal) or 1 (cartesian)");
        exit(1);
    }
    dock_param.ff_minimize_mode = static_cast<FFMinimizeMode>(args.ff_minimize_mode);
    if (dock_param.ff_minimize_mode == FFMIN_CARTESIAN && dock_param.ff_minimize_steps <= 0) {
        spdlog::critical("ff_minimize_mode=cartesian requires ff_minimize_steps > 0");
        exit(1);
    }
    dock_param.decomp = args.decomp;
    set_tor_lib(dock_param, args.use_tor_lib);

    // Bias
    set_bias(dock_param, bias_type_from_name(args.bias), args.bias_k);

    if (args.search_mode == "fast"){
        dock_param.exhaustiveness = 64;
        dock_param.mc_steps = 30;
        dock_param.opt_steps = 3;
    } else if (args.search_mode == "balance"){
        dock_param.exhaustiveness = 64;
        dock_param.mc_steps = 200;
        dock_param.opt_steps = 5;
    } else if (args.search_mode == "detail"){
        dock_param.exhaustiveness = 512;
        dock_param.mc_steps = 300;
        dock_param.opt_steps = 5;
    } else if (args.search_mode == "free"){
        //
    } else{
        spdlog::critical("Not supported search_mode: {} doesn't belong to (fast, balance, detail, free)" , args.search_mode);
        exit(1);
    }

    if (dock_param.exhaustiveness < std::thread::hardware_concurrency()) {
        spdlog::warn("Low exhaustiveness doesn't utilize all CPUs");
    }

    set_constraint_docking(dock_param, args.constraint_docking);
    // NOTE: constraint docking used to force randomize=false here, which meant
    // randomize_pose_cpu never ran and every pose started from the same input
    // conformation -- exhaustiveness bought short MC walks from one point, not
    // independent starts.  randomize_pose_cpu already honours the constraint
    // (it zeroes the centre/orientation update and randomises only torsions),
    // so the caller's choice is respected.

    dock_param.cluster_atom_indices_per_flex = args.cluster_atom_indices;
    dock_param.constrained_atom_indices_per_flex = args.constrained_atom_indices;

    log_bias(dock_param);
    return dock_param;
}
