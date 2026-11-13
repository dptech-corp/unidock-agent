//
// config/parse.cpp: the bias block (with its ring sub-block) and its Advanced aliases give the same DockParam, the block wins when both are
// given, constraint docking turns randomize off and sets the policy, and the pipeline's param_from_args equals
// parse_config for the same values (search_mode free: the two routes keep their own presets otherwise).
//

#include <catch2/catch_amalgamated.hpp>
#include <string>
#include <yaml-cpp/yaml.h>

#include "config/parse.h"
#include "model/model.h"

namespace {

const char* BASE_YAML = R"(
Advanced:
  seed: 181129
  exhaustiveness: 512
  randomize: true
  mc_steps: 40
  opt_steps: 15
  refine_steps: 5
  rmsd_limit: 1.0
  num_pose: 1
  rescore_pool: 10
  energy_range: 10.0
  tor_lib: false
  ff_minimize_steps: 0
  ligand_dms: /w/lig.dms
  protein_dms: /w/rec.dms
Hardware:
  ncpu: 4
Settings:
  task: screen
  search_mode: free
  constraint_docking: false
  decomp: false
  center_x: 1.0
  center_y: 2.0
  center_z: 3.0
  size_x: 22.5
  size_y: 22.5
  size_z: 22.5
Outputs:
  dir: /w/res2
Inputs:
  ligand_dms_full: /w/lig.dms
  receptor_dms_full: /w/rec.dms
)";

DockParam parse(const std::string& yaml, RunConfig* run_out = nullptr){
    RunConfig run;
    DockParam dp = parse_config(YAML::Load(yaml), run);
    if (run_out) *run_out = run;
    return dp;
}

void require_same_weights(const VinaWeights& a, const VinaWeights& b){
    REQUIRE(a.gauss1 == b.gauss1); REQUIRE(a.gauss2 == b.gauss2); REQUIRE(a.repulsion == b.repulsion);
    REQUIRE(a.hydrophobic == b.hydrophobic); REQUIRE(a.hbond == b.hbond); REQUIRE(a.intra == b.intra);
    REQUIRE(a.elec == b.elec); REQUIRE(a.elec_charged_scale == b.elec_charged_scale);
    REQUIRE(a.polar_env == b.polar_env); REQUIRE(a.arom_contact == b.arom_contact);
    REQUIRE(a.torsion_strain == b.torsion_strain);
}

void require_same_box(const Box& a, const Box& b){
    REQUIRE(a.x_lo == b.x_lo); REQUIRE(a.x_hi == b.x_hi); REQUIRE(a.y_lo == b.y_lo);
    REQUIRE(a.y_hi == b.y_hi); REQUIRE(a.z_lo == b.z_lo); REQUIRE(a.z_hi == b.z_hi);
}

/// Every field of DockParam (the search settings, the three weight sets, the box, FF, bias, the policy)
void require_same_param(const DockParam& a, const DockParam& b){
    REQUIRE(a.seed == b.seed); REQUIRE(a.rng_period == b.rng_period);
    REQUIRE(a.constraint_docking == b.constraint_docking); REQUIRE(a.exhaustiveness == b.exhaustiveness);
    REQUIRE(a.randomize == b.randomize); REQUIRE(a.mc_steps == b.mc_steps); REQUIRE(a.opt_steps == b.opt_steps);
    REQUIRE(a.refine_steps == b.refine_steps); REQUIRE(a.num_pose == b.num_pose); REQUIRE(a.rescore_pool == b.rescore_pool);
    REQUIRE(a.energy_range == b.energy_range); REQUIRE(a.rmsd_limit == b.rmsd_limit);
    REQUIRE(a.tor_prec == b.tor_prec); REQUIRE(a.box_prec == b.box_prec); REQUIRE(a.decomp == b.decomp);
    REQUIRE(a.cluster_atom_indices_per_flex == b.cluster_atom_indices_per_flex);
    REQUIRE(a.constrained_atom_indices_per_flex == b.constrained_atom_indices_per_flex);
    REQUIRE(a.ff_minimize_steps == b.ff_minimize_steps); REQUIRE(a.ligand_dms_path == b.ligand_dms_path);
    REQUIRE(a.protein_dms_path == b.protein_dms_path); REQUIRE(a.ff_minimize_mode == b.ff_minimize_mode);
    REQUIRE(a.search_score == b.search_score); REQUIRE(a.opt_score == b.opt_score);
    require_same_box(a.box, b.box); require_same_box(a.inner_box, b.inner_box); REQUIRE(a.use_inner_box == b.use_inner_box);
    REQUIRE(a.bias_type == b.bias_type); REQUIRE(a.bias_k == b.bias_k); REQUIRE(a.angular_hbond == b.angular_hbond);
    REQUIRE(a.ring_bias.k == b.ring_bias.k); REQUIRE(a.ring_bias.s == b.ring_bias.s); REQUIRE(a.ring_bias.d0 == b.ring_bias.d0);
    REQUIRE(a.use_grid == b.use_grid); REQUIRE(a.grid_spacing == b.grid_spacing); REQUIRE(a.grid_margin == b.grid_margin);
    require_same_weights(a.weights, b.weights); require_same_weights(a.rescore_weights, b.rescore_weights);
    require_same_weights(a.affinity_weights, b.affinity_weights);
    REQUIRE(a.has_rescore_weights == b.has_rescore_weights); REQUIRE(a.has_affinity_weights == b.has_affinity_weights);
    REQUIRE(a.policy.randomize == b.policy.randomize); REQUIRE(a.policy.static_root == b.policy.static_root);
    REQUIRE(a.policy.keep_template_hydrogens == b.policy.keep_template_hydrogens);
    REQUIRE(a.policy.use_torsion_ranges == b.policy.use_torsion_ranges);
}

} // namespace


TEST_CASE("parse_config: the plain YAML, its defaults and the run items", "[config][parse]"){
    RunConfig run;
    DockParam dp = parse(BASE_YAML, &run);
    REQUIRE(run.task == "screen");
    REQUIRE(run.ligand_dms_full == "/w/lig.dms");
    REQUIRE(run.receptor_dms_full == "/w/rec.dms");
    REQUIRE(run.output_dir == "/w/res2");
    REQUIRE(dp.ncpu == 4);
    REQUIRE(dp.seed == 181129); REQUIRE(dp.exhaustiveness == 512); REQUIRE(dp.mc_steps == 40);
    REQUIRE(dp.opt_steps == 15); REQUIRE(dp.refine_steps == 5); REQUIRE(dp.num_pose == 1);
    REQUIRE(dp.rescore_pool == 10); REQUIRE(dp.energy_range == Real(10)); REQUIRE(dp.rmsd_limit == Real(1));
    REQUIRE(dp.box.x_lo == Real(1.0 - 22.5 / 2)); REQUIRE(dp.box.z_hi == Real(3.0 + 22.5 / 2));
    REQUIRE(!dp.use_inner_box);
    // the defaults of the keys the YAML leaves out, as DockParam / main.cpp had them
    REQUIRE(dp.bias_type == BT_NO); REQUIRE(dp.bias_k == Real(0.3));
    REQUIRE(dp.ring_bias.k == Real(30)); REQUIRE(dp.ring_bias.s == Real(3)); REQUIRE(dp.ring_bias.d0 == Real(1));
    REQUIRE(dp.angular_hbond); REQUIRE(!dp.decomp); REQUIRE(dp.use_grid); REQUIRE(dp.rng_period == 0);
    REQUIRE(dp.tor_prec == Real(0.3)); REQUIRE(dp.box_prec == Real(1.0));
    REQUIRE(dp.ff_minimize_steps == 0); REQUIRE(dp.ff_minimize_mode == FFMIN_INTERNAL);
    REQUIRE(dp.ligand_dms_path == "/w/lig.dms"); REQUIRE(dp.protein_dms_path == "/w/rec.dms");
    require_same_weights(dp.weights, default_search_weights());
    require_same_weights(dp.rescore_weights, default_rescore_weights());
    require_same_weights(dp.affinity_weights, default_affinity_weights());
    // the policy mirrors
    REQUIRE(dp.randomize); REQUIRE(dp.policy.randomize);
    REQUIRE(!dp.constraint_docking); REQUIRE(!dp.policy.static_root);
    REQUIRE(!dp.policy.keep_template_hydrogens);
    REQUIRE(!dp.policy.use_torsion_ranges);                       // tor_lib: false
    REQUIRE(parse(std::string(BASE_YAML) + "").policy.use_torsion_ranges == false);
    std::string tor_default = BASE_YAML;
    tor_default.replace(tor_default.find("  tor_lib: false\n"), 17, "");
    REQUIRE(parse(tor_default).policy.use_torsion_ranges);        // Advanced.tor_lib defaults to true
}


TEST_CASE("parse_config: the bias block, the Advanced aliases and both give the same DockParam", "[config][parse][bias]"){
    const DockParam plain = parse(BASE_YAML);
    const DockParam block = parse(std::string(BASE_YAML) + "bias:\n  type: align\n  k: 0.05\n");
    std::string alias = BASE_YAML;
    alias.replace(alias.find("  tor_lib: false\n"), 17, "  tor_lib: false\n  bias: align\n  bias_k: 0.05\n");
    const DockParam aliased = parse(alias);

    REQUIRE(block.bias_type == BT_ALIGN);
    REQUIRE(block.bias_k == Real(0.05));
    REQUIRE(block.policy.keep_template_hydrogens);                // a bias keeps the template hydrogens
    REQUIRE(!block.policy.static_root);
    require_same_param(block, aliased);
    // the bias is the only difference to the plain parse
    REQUIRE(plain.bias_type == BT_NO);
    REQUIRE(!plain.policy.keep_template_hydrogens);
    DockParam plain_with_bias = plain;
    set_bias(plain_with_bias, BT_ALIGN, Real(0.05));
    require_same_param(plain_with_bias, block);

    // both given: the block wins (per key)
    const DockParam both = parse(alias + "bias:\n  type: pos\n  k: 0.2\n");
    REQUIRE(both.bias_type == BT_POS);
    REQUIRE(both.bias_k == Real(0.2));
    const DockParam both_k_only = parse(alias + "bias:\n  k: 0.2\n");
    REQUIRE(both_k_only.bias_type == BT_ALIGN);
    REQUIRE(both_k_only.bias_k == Real(0.2));

    // the pos type, and "no" through the block
    REQUIRE(parse(std::string(BASE_YAML) + "bias:\n  type: pos\n").bias_type == BT_POS);
    REQUIRE(parse(std::string(BASE_YAML) + "bias:\n  type: pos\n").bias_k == Real(0.3));   // k keeps the default
    const DockParam off = parse(std::string(BASE_YAML) + "bias:\n  type: no\n  k: 0.7\n");
    REQUIRE(off.bias_type == BT_NO);
    REQUIRE(off.bias_k == Real(0.7));
    REQUIRE(!off.policy.keep_template_hydrogens);
}


TEST_CASE("parse_config: the ring sub-block of the bias block sets the ring parameters, each key alone", "[config][parse][bias][ring_bias]"){
    const DockParam plain = parse(BASE_YAML);
    const DockParam full = parse(std::string(BASE_YAML) + "bias:\n  type: align\n  k: 0.05\n  ring:\n    k: 12\n    s: 2.5\n    d0: 1.5\n");
    REQUIRE(full.bias_type == BT_ALIGN);
    REQUIRE(full.bias_k == Real(0.05));
    REQUIRE(full.ring_bias.k == Real(12)); REQUIRE(full.ring_bias.s == Real(2.5)); REQUIRE(full.ring_bias.d0 == Real(1.5));
    // the ring block is the only difference to the same bias block without it
    DockParam no_ring = parse(std::string(BASE_YAML) + "bias:\n  type: align\n  k: 0.05\n");
    REQUIRE(no_ring.ring_bias.k == plain.ring_bias.k); REQUIRE(no_ring.ring_bias.s == plain.ring_bias.s);
    REQUIRE(no_ring.ring_bias.d0 == plain.ring_bias.d0);
    no_ring.ring_bias = full.ring_bias;
    require_same_param(no_ring, full);
    // a key left out keeps its default
    const DockParam s_only = parse(std::string(BASE_YAML) + "bias:\n  type: align\n  ring:\n    s: 4\n");
    REQUIRE(s_only.ring_bias.k == Real(30)); REQUIRE(s_only.ring_bias.s == Real(4)); REQUIRE(s_only.ring_bias.d0 == Real(1));
    REQUIRE(s_only.bias_k == Real(0.3));
    // the pipeline arguments carry no ring parameters: the defaults
    PipelineArgs args;
    args.bias = "align"; args.bias_k = 0.05;
    const DockParam from_args = param_from_args(args);
    REQUIRE(from_args.ring_bias.k == Real(30)); REQUIRE(from_args.ring_bias.s == Real(3)); REQUIRE(from_args.ring_bias.d0 == Real(1));
}


TEST_CASE("parse_config: constraint docking turns randomize off and sets the policy", "[config][parse][policy]"){
    std::string yaml = BASE_YAML;
    yaml.replace(yaml.find("  constraint_docking: false\n"), 28, "  constraint_docking: true\n");
    const DockParam dp = parse(yaml);
    REQUIRE(dp.constraint_docking);
    REQUIRE(dp.policy.static_root);
    REQUIRE(dp.policy.keep_template_hydrogens);
    REQUIRE(!dp.randomize);                                       // Advanced.randomize true is overridden
    REQUIRE(!dp.policy.randomize);
    REQUIRE(dp.bias_type == BT_NO);

    // the helpers keep the fields and the policy together, in either order
    DockParam h;
    set_bias(h, BT_ALIGN, Real(0.1));
    REQUIRE(h.policy.keep_template_hydrogens);
    set_constraint_docking(h, true);
    REQUIRE((h.constraint_docking && h.policy.static_root && h.policy.keep_template_hydrogens));
    set_bias(h, BT_NO, Real(0.1));
    REQUIRE(h.policy.keep_template_hydrogens);                    // still constrained
    set_constraint_docking(h, false);
    REQUIRE(!h.policy.keep_template_hydrogens);
    set_randomize(h, false);
    REQUIRE((!h.randomize && !h.policy.randomize));
    set_tor_lib(h, true);
    REQUIRE(h.policy.use_torsion_ranges);
}


TEST_CASE("parse_config: search_mode presets, inner box, weights, cluster indices, FF", "[config][parse]"){
    std::string yaml = BASE_YAML;
    yaml.replace(yaml.find("  search_mode: free\n"), 20, "  search_mode: fast\n  inner_box: 10.0\n"
                 "  score_weights: {gauss1: -0.05, elec: 0.1}\n  rescore_weights: {hbond: -0.7}\n  affinity_weights: {intra: 0.9}\n");
    yaml.replace(yaml.find("  ff_minimize_steps: 0\n"), 22, "  ff_minimize_steps: 50\n  ff_minimize_mode: 1\n"
                 "  cluster_atom_indices: [[0, 1, 2], [], [4]]\n  rng_period: 3\n");
    const DockParam dp = parse(yaml);
#if USE_CUDA
    REQUIRE(dp.exhaustiveness == 128);
#else
    REQUIRE(dp.exhaustiveness == 64);
#endif
    REQUIRE(dp.mc_steps == 20);
    REQUIRE(dp.use_inner_box);
    REQUIRE(dp.inner_box.x_lo == Real(1.0 - 5.0)); REQUIRE(dp.inner_box.x_hi == Real(1.0 + 5.0));
    REQUIRE(dp.weights.gauss1 == Real(-0.05)); REQUIRE(dp.weights.elec == Real(0.1));
    REQUIRE(dp.weights.gauss2 == default_search_weights().gauss2);
    REQUIRE(dp.rescore_weights.hbond == Real(-0.7)); REQUIRE(dp.affinity_weights.intra == Real(0.9));
    REQUIRE(dp.ff_minimize_steps == 50); REQUIRE(dp.ff_minimize_mode == FFMIN_CARTESIAN);
    REQUIRE(dp.cluster_atom_indices_per_flex == std::vector<std::vector<int>>{{0, 1, 2}, {}, {4}});
    REQUIRE(dp.rng_period == 3);
}


TEST_CASE("param_from_args equals parse_config for the same values (search_mode free)", "[config][parse][pipeline]"){
    PipelineArgs args;
    args.ligand_dms_path = "/w/lig.dms"; args.receptor_dms_path = "/w/rec.dms"; args.output_dir = "/w/res2";
    args.center_x = 1.0; args.center_y = 2.0; args.center_z = 3.0;
    args.size_x = 22.5; args.size_y = 22.5; args.size_z = 22.5;
    args.task = "screen"; args.search_mode = "free";
    args.exhaustiveness = 512; args.randomize = true; args.mc_steps = 40; args.opt_steps = 15; args.refine_steps = 5;
    args.num_pose = 1; args.rescore_pool = 10; args.rmsd_limit = 1.0; args.energy_range = 10.0; args.seed = 181129;
    args.constraint_docking = false; args.use_tor_lib = false; args.decomp = false;
    args.ff_minimize_steps = 0; args.ff_minimize_mode = 0;

    for (const char* bias : {"no", "align", "pos"}){
        args.bias = bias; args.bias_k = bias == std::string("no") ? Real(0.3) : Real(0.05);
        std::string yaml = BASE_YAML;
        if (bias != std::string("no")) yaml += std::string("bias:\n  type: ") + bias + "\n  k: 0.05\n";
        DockParam from_yaml = parse(yaml);
        DockParam from_args = param_from_args(args);
        from_args.ncpu = from_yaml.ncpu;                          // Hardware.ncpu has no pipeline argument
        INFO("bias " << bias);
        require_same_param(from_yaml, from_args);
    }

    // the pipeline honours the caller's randomize under constraint docking (the binary turns it off)
    args.bias = "no"; args.bias_k = 0.0;
    args.constraint_docking = true; args.randomize = true;
    DockParam constrained = param_from_args(args);
    REQUIRE((constrained.constraint_docking && constrained.policy.static_root && constrained.policy.keep_template_hydrogens));
    REQUIRE((constrained.randomize && constrained.policy.randomize));
    args.cluster_atom_indices = {{0, 1}, {}};
    args.constrained_atom_indices = {{3}, {4, 5}};
    DockParam lists = param_from_args(args);
    REQUIRE(lists.cluster_atom_indices_per_flex == args.cluster_atom_indices);
    REQUIRE(lists.constrained_atom_indices_per_flex == args.constrained_atom_indices);

    // the pipeline's own presets (unchanged from pipeline.cpp)
    args.search_mode = "balance";
    DockParam balance = param_from_args(args);
    REQUIRE((balance.exhaustiveness == 64 && balance.mc_steps == 200 && balance.opt_steps == 5));
}
