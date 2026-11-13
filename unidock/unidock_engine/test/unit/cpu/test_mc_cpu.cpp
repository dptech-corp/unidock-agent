//
// Unit tests for CPU Monte Carlo search (mc_single_pose_cpu)
// Tests: energy decrease, determinism, mc_steps=0, timeout
//

#include <catch2/catch_amalgamated.hpp>
#include "cpu/mc_cpu.h"
#include "cpu/common_cpu.h"
#include "cpu/score_cpu.h"
#include "cpu/spatial_grid.h"
#include "constants/constants.h"
#include "score/vina.h"
#include <cmath>
#include <vector>
#include <chrono>

using Catch::Approx;

// ========== Helpers (same pattern as test_angular_hbond.cpp) ==========

static void init_test_atp() {
    init_default_atom_type_props(g_cpu_constants.atp);
    int n = sizeof(VN_VDW_RADII) / sizeof(VN_VDW_RADII[0]);
    for (int i = 0; i < n; i++) g_cpu_constants.vdw_radii[i] = VN_VDW_RADII[i];
    for (int i = n; i < MAX_VN_TYPES; i++) g_cpu_constants.vdw_radii[i] = Real(0);
}

static void set_large_box() {
    g_cpu_constants.BOX_X_LO = -50; g_cpu_constants.BOX_X_HI = 50;
    g_cpu_constants.BOX_Y_LO = -50; g_cpu_constants.BOX_Y_HI = 50;
    g_cpu_constants.BOX_Z_LO = -50; g_cpu_constants.BOX_Z_HI = 50;
}

static FixMol make_receptor_grid(std::vector<Real>& rec_coords,
                                  Real rx, Real ry, Real rz, int rec_type) {
    rec_coords = {rx, ry, rz};
    std::vector<int> rec_types_vec = {rec_type};

    free_spatial_grid(g_cpu_constants.receptor_grid);
    build_spatial_grid(g_cpu_constants.receptor_grid,
                       rec_coords.data(), rec_types_vec.data(),
                       1, g_cpu_constants.atp,
                       g_cpu_constants.vdw_radii);

    FixMol fix;
    fix.natom = 1;
    fix.coords = rec_coords.data();
    return fix;
}

struct MinimalFlexParam {
    FlexParamVina param;
    std::vector<int> atom_types_vec;
    std::vector<int> inds_bias_vec;

    void build(const std::vector<int>& vn_types, const Real* coords) {
        int natom = (int)vn_types.size();
        atom_types_vec = vn_types;
        inds_bias_vec.assign(natom * 2, 0);

        param.npair_intra = 0;
        param.pairs_intra = nullptr;
        param.r1_plus_r2_intra = nullptr;
        param.npair_inter = 0;
        param.pairs_inter = nullptr;
        param.r1_plus_r2_inter = nullptr;
        param.atom_types = atom_types_vec.data();
        param.terms.bias.inds = inds_bias_vec.data();
        param.terms.bias.params = nullptr;

        param.bonded_h_topo = {};
        build_bonded_h_topo(param.bonded_h_topo, coords, vn_types.data(), natom);
    }

    ~MinimalFlexParam() {
        free_bonded_h_topo(param.bonded_h_topo);
    }
};


// ========== Fixture builder for 2-atom rigid ligand ==========
// N_D at ligand_pos, bonded H at ligand_pos - (1,0,0)
// Receptor: O_A at rec_pos
struct MCTestFixture {
    // Ligand
    std::vector<Real> lig_coords;
    std::vector<int>  lig_vn_types;
    FlexPose          pose;
    FlexTopo          topo;
    MinimalFlexParam  flex_param;

    // Receptor
    std::vector<Real> rec_coords;
    std::vector<int>  rec_types;
    FixMol            fix_mol;
    FixParamVina      fix_param;

    void build(Real lig_x, Real lig_y, Real lig_z,
               Real rec_x, Real rec_y, Real rec_z) {
        init_test_atp();
        set_large_box();
        g_cpu_constants.USE_ANGULAR_HBOND = false;
        g_cpu_constants.USE_POST_BFGS_HBOND_CORR = false;
        g_cpu_constants.USE_TORSION_PENALTY = false;
        g_cpu_constants.FLAG_CONSTRAINT_DOCK = false;

        // 2-atom ligand: N_D + bonded H
        lig_coords = {lig_x, lig_y, lig_z,
                      lig_x - Real(1.0), lig_y, lig_z};
        lig_vn_types = {VN_TYPE_N_D, VN_TYPE_H};

        // Receptor: O_A
        fix_mol = make_receptor_grid(rec_coords, rec_x, rec_y, rec_z, VN_TYPE_O_A);
        rec_types = {VN_TYPE_O_A};
        fix_param.atom_types = rec_types.data();

        // FlexTopo: ntorsion=0
        topo.natom = 2;
        topo.ntorsion = 0;
        topo.vn_types = lig_vn_types.data();
        topo.axis_atoms = nullptr;
        topo.range_inds = nullptr;
        topo.range_list = nullptr;
        topo.rotated_inds = nullptr;
        topo.rotated_atoms = nullptr;

        // FlexParamVina
        flex_param.build(lig_vn_types, lig_coords.data());

        // FlexPose
        pose.coords = lig_coords.data();
        pose.dihedrals = nullptr;
        pose.center[0] = lig_x;
        pose.center[1] = lig_y;
        pose.center[2] = lig_z;
        pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
        pose.energy = 999;
    }

    ~MCTestFixture() {
        free_spatial_grid(g_cpu_constants.receptor_grid);
    }
};


// ===================== TEST 1: energy decreases =====================

TEST_CASE("mc_single_pose energy decreases after search", "[mc][cpu]") {
    MCTestFixture f;
    // Ligand at (5,0,0), receptor at (0,0,0) — suboptimal distance
    f.build(5.0, 0.0, 0.0,  0.0, 0.0, 0.0);

    // Compute initial energy
    std::vector<Real> aux_f(f.topo.natom * 3, 0);
    Real initial_energy = cal_e_f_cpu(&f.pose, f.topo, f.fix_mol,
                                       f.flex_param.param, f.fix_param,
                                       aux_f.data());

    // Run MC search
    CPURandom rng(42);
    MCThreadMemory mem;
    FlexPose out_pose;
    std::vector<Real> out_coords(f.topo.natom * 3);
    out_pose.coords = out_coords.data();
    out_pose.dihedrals = nullptr;
    // Copy initial pose to out_pose
    duplicate_pose_cpu(&out_pose, &f.pose, dof_x + f.topo.ntorsion, f.topo.natom);

    mc_single_pose_cpu(&out_pose, f.topo, f.fix_mol,
                       f.flex_param.param, f.fix_param,
                       rng, /*mc_steps=*/10, /*opt_steps=*/5,
                       /*randomize=*/false, /*pose_id=*/0, /*total_poses=*/1,
                       mem);

    REQUIRE(std::isfinite(out_pose.energy));
    REQUIRE(out_pose.energy < initial_energy);
}


// ===================== TEST 2: deterministic with same seed =====================

TEST_CASE("mc_single_pose deterministic with same seed", "[mc][cpu]") {
    auto run_mc = [](unsigned int seed) -> std::pair<Real, std::vector<Real>> {
        MCTestFixture f;
        f.build(5.0, 0.0, 0.0,  0.0, 0.0, 0.0);

        CPURandom rng(seed);
        MCThreadMemory mem;
        FlexPose out_pose;
        std::vector<Real> out_coords(f.topo.natom * 3);
        out_pose.coords = out_coords.data();
        out_pose.dihedrals = nullptr;
        duplicate_pose_cpu(&out_pose, &f.pose, dof_x + f.topo.ntorsion, f.topo.natom);

        mc_single_pose_cpu(&out_pose, f.topo, f.fix_mol,
                           f.flex_param.param, f.fix_param,
                           rng, /*mc_steps=*/5, /*opt_steps=*/3,
                           /*randomize=*/false, /*pose_id=*/0, /*total_poses=*/1,
                           mem);

        return {out_pose.energy, out_coords};
    };

    auto [e1, coords1] = run_mc(42);
    auto [e2, coords2] = run_mc(42);

    REQUIRE_THAT(e1, Catch::Matchers::WithinAbs(e2, 1e-8));
    for (size_t i = 0; i < coords1.size(); i++) {
        REQUIRE_THAT(coords1[i], Catch::Matchers::WithinAbs(coords2[i], 1e-8));
    }
}


// ===================== TEST 3: mc_steps=0 scores without mutation =====================

TEST_CASE("mc_single_pose mc_steps=0 scores without mutation", "[mc][cpu]") {
    MCTestFixture f;
    f.build(3.0, 0.0, 0.0,  0.0, 0.0, 0.0);

    // Save original coordinates
    std::vector<Real> orig_coords(f.lig_coords.begin(), f.lig_coords.end());

    CPURandom rng(42);
    MCThreadMemory mem;
    FlexPose out_pose;
    std::vector<Real> out_coords(f.topo.natom * 3);
    out_pose.coords = out_coords.data();
    out_pose.dihedrals = nullptr;
    duplicate_pose_cpu(&out_pose, &f.pose, dof_x + f.topo.ntorsion, f.topo.natom);

    mc_single_pose_cpu(&out_pose, f.topo, f.fix_mol,
                       f.flex_param.param, f.fix_param,
                       rng, /*mc_steps=*/0, /*opt_steps=*/5,
                       /*randomize=*/false, /*pose_id=*/0, /*total_poses=*/1,
                       mem);

    REQUIRE(std::isfinite(out_pose.energy));
    // Coordinates should be unchanged
    for (size_t i = 0; i < orig_coords.size(); i++) {
        REQUIRE_THAT(out_coords[i], Catch::Matchers::WithinAbs(orig_coords[i], 1e-6));
    }
}


// ===================== TEST 4: timeout prevents hang =====================

TEST_CASE("mc_single_pose timeout prevents hang", "[mc][cpu]") {
    MCTestFixture f;
    f.build(5.0, 0.0, 0.0,  0.0, 0.0, 0.0);

    CPURandom rng(42);
    MCThreadMemory mem;
    FlexPose out_pose;
    std::vector<Real> out_coords(f.topo.natom * 3);
    out_pose.coords = out_coords.data();
    out_pose.dihedrals = nullptr;
    duplicate_pose_cpu(&out_pose, &f.pose, dof_x + f.topo.ntorsion, f.topo.natom);

    // Set a very short deadline
    set_pose_deadline(0.001);

    auto t0 = std::chrono::steady_clock::now();

    mc_single_pose_cpu(&out_pose, f.topo, f.fix_mol,
                       f.flex_param.param, f.fix_param,
                       rng, /*mc_steps=*/1000, /*opt_steps=*/10,
                       /*randomize=*/false, /*pose_id=*/0, /*total_poses=*/1,
                       mem);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    REQUIRE(std::isfinite(out_pose.energy));
    REQUIRE(elapsed < 1.0);
}
