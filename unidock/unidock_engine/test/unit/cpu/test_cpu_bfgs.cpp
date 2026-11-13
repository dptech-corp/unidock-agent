//
// Unit tests for CPU BFGS optimization
// Created for Uni-Dock2 CPU backend validation
//

#include <catch2/catch_amalgamated.hpp>
#include "cpu/bfgs_cpu.h"
#include "cpu/score_cpu.h"
#include "cpu/common_cpu.h"
#include "cpu/random_cpu.h"
#include "cpu/spatial_grid.h"
#include "constants/constants.h"
#include "score/vina.h"
#include "myutils/matrix.h"
#include <vector>
#include <cmath>
#include <limits>

using Catch::Approx;

TEST_CASE("CPU Matrix-Vector Product", "[cpu][bfgs]") {
    
    SECTION("minus_mat_vec_product_cpu identity matrix") {
        const int ntorsion = 1;
        const int dim = dof_g + ntorsion;  // 6 + 1 = 7
        
        // Identity matrix (stored as upper triangular)
        std::vector<Real> h(dim * (dim + 1) / 2, 0);
        for (int i = 0; i < dim; i++) {
            h[tri_mat_index(i, i)] = 1.0;
        }
        
        // Input gradient
        std::vector<Real> dihedrals_g(ntorsion, 1.0);
        FlexPoseGradient g;
        g.center_g[0] = 1; g.center_g[1] = 2; g.center_g[2] = 3;
        g.orientation_g[0] = 4; g.orientation_g[1] = 5; g.orientation_g[2] = 6;
        g.dihedrals_g = dihedrals_g.data();
        
        // Output
        std::vector<Real> dihedrals_p(ntorsion, 0);
        FlexPoseGradient p;
        p.dihedrals_g = dihedrals_p.data();
        
        minus_mat_vec_product_cpu(&p, h.data(), &g, dim);
        
        // -I * g = -g
        REQUIRE(p.center_g[0] == Approx(-1.0));
        REQUIRE(p.center_g[1] == Approx(-2.0));
        REQUIRE(p.center_g[2] == Approx(-3.0));
    }
}


TEST_CASE("CPU Gradient Dot Product", "[cpu][bfgs]") {
    
    SECTION("g_dot_product_cpu") {
        const int ntorsion = 2;
        const int dim = dof_g + ntorsion;
        
        std::vector<Real> dihedrals_a(ntorsion, 1.0);
        std::vector<Real> dihedrals_b(ntorsion, 2.0);
        
        FlexPoseGradient a, b;
        a.center_g[0] = 1; a.center_g[1] = 1; a.center_g[2] = 1;
        a.orientation_g[0] = 1; a.orientation_g[1] = 1; a.orientation_g[2] = 1;
        a.dihedrals_g = dihedrals_a.data();
        
        b.center_g[0] = 2; b.center_g[1] = 2; b.center_g[2] = 2;
        b.orientation_g[0] = 2; b.orientation_g[1] = 2; b.orientation_g[2] = 2;
        b.dihedrals_g = dihedrals_b.data();
        
        Real dot = g_dot_product_cpu(&a, &b, dim);
        
        // (1*2 + 1*2 + 1*2) + (1*2 + 1*2 + 1*2) + (1*2 + 1*2) = 6 + 6 + 4 = 16
        REQUIRE(dot == Approx(16.0));
    }
}


TEST_CASE("CPU Random Number Generator", "[cpu][random]") {
    
    SECTION("CPURandom uniform distribution") {
        CPURandom rng(12345);
        
        int count_below_05 = 0;
        int total = 10000;
        
        for (int i = 0; i < total; i++) {
            Real r = rng.uniform();
            REQUIRE(r >= 0.0);
            REQUIRE(r < 1.0);
            if (r < 0.5) count_below_05++;
        }
        
        // Should be roughly 50% below 0.5
        Real ratio = static_cast<Real>(count_below_05) / total;
        REQUIRE(ratio == Approx(0.5).margin(0.05));
    }
    
    SECTION("CPURandom randint range") {
        CPURandom rng(42);
        
        for (int i = 0; i < 100; i++) {
            int r = rng.randint(5, 10);
            REQUIRE(r >= 5);
            REQUIRE(r <= 10);
        }
    }
    
    SECTION("CPURandom rand_in_sphere") {
        CPURandom rng(99);
        
        for (int i = 0; i < 100; i++) {
            Real out[4];
            rng.rand_in_sphere(out);
            
            Real norm_sq = out[0]*out[0] + out[1]*out[1] + out[2]*out[2];
            REQUIRE(norm_sq < 1.0);
            REQUIRE(out[3] >= 0.0);
            REQUIRE(out[3] < 1.0);
        }
    }
    
    SECTION("CPURandom reproducibility") {
        CPURandom rng1(12345);
        CPURandom rng2(12345);
        
        for (int i = 0; i < 100; i++) {
            REQUIRE(rng1.uniform() == rng2.uniform());
        }
    }
}


// ==================== normalize_angle_cpu extreme value tests ====================

TEST_CASE("normalize_angle_cpu handles extreme values", "[cpu][bfgs][normalize]") {
    const Real PI_VAL = Real(3.1415927);

    SECTION("normal angle passes through unchanged") {
        REQUIRE(normalize_angle_cpu(Real(0.5)) == Approx(0.5));
        REQUIRE(normalize_angle_cpu(Real(-0.5)) == Approx(-0.5));
        REQUIRE(normalize_angle_cpu(Real(0.0)) == Approx(0.0));
    }

    SECTION("+Inf does not hang and returns 0") {
        Real result = normalize_angle_cpu(std::numeric_limits<Real>::infinity());
        REQUIRE(result == Approx(0.0));
    }

    SECTION("-Inf does not hang and returns 0") {
        Real result = normalize_angle_cpu(-std::numeric_limits<Real>::infinity());
        REQUIRE(result == Approx(0.0));
    }

    SECTION("NaN does not hang and returns 0") {
        Real result = normalize_angle_cpu(std::numeric_limits<Real>::quiet_NaN());
        REQUIRE(result == Approx(0.0));
    }

    SECTION("large positive angle normalizes to [-PI, PI]") {
        Real result = normalize_angle_cpu(100 * PI_VAL);
        REQUIRE(result >= -PI_VAL);
        REQUIRE(result <= PI_VAL);
    }

    SECTION("large negative angle normalizes to [-PI, PI]") {
        Real result = normalize_angle_cpu(-100 * PI_VAL);
        REQUIRE(result >= -PI_VAL);
        REQUIRE(result <= PI_VAL);
    }

    SECTION("angle just above PI wraps correctly") {
        Real result = normalize_angle_cpu(PI_VAL + Real(0.1));
        REQUIRE(result == Approx(-PI_VAL + Real(0.1)).margin(1e-4));
    }

    SECTION("angle just below -PI wraps correctly") {
        Real result = normalize_angle_cpu(-PI_VAL - Real(0.1));
        REQUIRE(result == Approx(PI_VAL - Real(0.1)).margin(1e-4));
    }
}


// ==================== cal_grad_cpu with overlapping axis atoms ====================

TEST_CASE("cal_grad_cpu with near-zero axis norm produces finite gradient", "[cpu][bfgs][grad]") {
    // Minimal system: 3 atoms, 1 torsion
    // Axis atoms 0 and 1 at same position (axis_norm ~ 0)
    // Rotated atom 2 with non-zero force

    const int natom = 3;
    const int ntorsion = 1;

    // Coords: atom 0 and atom 1 at the SAME position, atom 2 offset
    std::vector<Real> coords = {
        0, 0, 0,   // atom 0 (axis_from)
        0, 0, 0,   // atom 1 (axis_to) — overlapping!
        5, 0, 0    // atom 2 (rotated atom)
    };

    std::vector<Real> dihedrals = {0.5};

    FlexPose pose;
    pose.coords = coords.data();
    pose.dihedrals = dihedrals.data();
    pose.center[0] = pose.center[1] = pose.center[2] = 0;
    pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
    pose.energy = 0;

    // FlexTopo: 1 torsion, axis atoms 0→1, rotated atom = {2}
    std::vector<int> axis_atoms = {0, 1};
    std::vector<int> rotated_inds = {0, 0};  // begin=0, end=0 → one rotated atom
    std::vector<int> rotated_atoms = {2};
    std::vector<int> vn_types = {0, 0, 0};   // all non-hydrogen type
    std::vector<int> range_inds = {0, 0};
    std::vector<Real> range_list;

    FlexTopo topo;
    topo.natom = natom;
    topo.ntorsion = ntorsion;
    topo.axis_atoms = axis_atoms.data();
    topo.rotated_inds = rotated_inds.data();
    topo.rotated_atoms = rotated_atoms.data();
    topo.vn_types = vn_types.data();
    topo.range_inds = range_inds.data();
    topo.range_list = range_list.data();

    // Force on atom 2: large force that would produce huge torque
    std::vector<Real> forces = {
        0, 0, 0,       // atom 0
        0, 0, 0,       // atom 1
        0, 100, 0      // atom 2: large force in y
    };

    // Ensure type 0 is not hydrogen
    g_cpu_constants.atp.is_hydrogen[0] = false;
    g_cpu_constants.FLAG_CONSTRAINT_DOCK = true;  // skip center/orientation grad

    std::vector<Real> dihedrals_g(ntorsion, 0);
    FlexPoseGradient grad;
    grad.dihedrals_g = dihedrals_g.data();

    cal_grad_cpu(forces.data(), topo, &pose, &grad);

    // With the axis_norm guard, gradient must be finite (not Inf/NaN)
    REQUIRE(std::isfinite(grad.dihedrals_g[0]));
}


// ==================== apply_grad_update_dihe_cpu with Inf input ====================

TEST_CASE("apply_grad_update_dihe_cpu with Inf increment does not hang", "[cpu][bfgs][dihe]") {
    const int natom = 3;
    const int ntorsion = 1;

    // Coords: simple linear molecule along x-axis
    std::vector<Real> coords = {
        0, 0, 0,    // atom 0 (axis_from)
        1, 0, 0,    // atom 1 (axis_to)
        2, 0, 0     // atom 2 (rotated)
    };
    std::vector<Real> dihedrals = {0.0};

    FlexPose pose;
    pose.coords = coords.data();
    pose.dihedrals = dihedrals.data();
    pose.center[0] = 1; pose.center[1] = 0; pose.center[2] = 0;
    pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
    pose.energy = 0;

    std::vector<int> axis_atoms = {0, 1};
    std::vector<int> rotated_inds = {0, 0};
    std::vector<int> rotated_atoms = {2};
    std::vector<int> vn_types = {0, 0, 0};
    std::vector<int> range_inds = {0, 0};
    std::vector<Real> range_list;

    FlexTopo topo;
    topo.natom = natom;
    topo.ntorsion = ntorsion;
    topo.axis_atoms = axis_atoms.data();
    topo.rotated_inds = rotated_inds.data();
    topo.rotated_atoms = rotated_atoms.data();
    topo.vn_types = vn_types.data();
    topo.range_inds = range_inds.data();
    topo.range_list = range_list.data();

    g_cpu_constants.USE_TORSION_PENALTY = false;

    SECTION("Inf increment does not hang") {
        apply_grad_update_dihe_cpu(&pose, &topo, 0,
                                    std::numeric_limits<Real>::infinity());
        // Must return (not hang). Dihedral should be clamped to 0 by isfinite guard.
        REQUIRE(std::isfinite(pose.dihedrals[0]));
    }

    SECTION("-Inf increment does not hang") {
        apply_grad_update_dihe_cpu(&pose, &topo, 0,
                                    -std::numeric_limits<Real>::infinity());
        REQUIRE(std::isfinite(pose.dihedrals[0]));
    }

    SECTION("NaN increment does not hang") {
        apply_grad_update_dihe_cpu(&pose, &topo, 0,
                                    std::numeric_limits<Real>::quiet_NaN());
        REQUIRE(std::isfinite(pose.dihedrals[0]));
    }
}


// ==================== Helpers for BFGS convergence tests ====================

static void init_test_atp_bfgs() {
    init_default_atom_type_props(g_cpu_constants.atp);
    int n = sizeof(VN_VDW_RADII) / sizeof(VN_VDW_RADII[0]);
    for (int i = 0; i < n; i++) g_cpu_constants.vdw_radii[i] = VN_VDW_RADII[i];
    for (int i = n; i < MAX_VN_TYPES; i++) g_cpu_constants.vdw_radii[i] = Real(0);
}

static void set_large_box_bfgs() {
    g_cpu_constants.BOX_X_LO = -50; g_cpu_constants.BOX_X_HI = 50;
    g_cpu_constants.BOX_Y_LO = -50; g_cpu_constants.BOX_Y_HI = 50;
    g_cpu_constants.BOX_Z_LO = -50; g_cpu_constants.BOX_Z_HI = 50;
}

static FixMol make_receptor_grid_bfgs(std::vector<Real>& rec_coords,
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

struct MinimalFlexParamBFGS {
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

    ~MinimalFlexParamBFGS() {
        free_bonded_h_topo(param.bonded_h_topo);
    }
};


// ==================== BFGS convergence: energy never increases ====================

TEST_CASE("bfgs_cpu never increases energy (restore logic)", "[cpu][bfgs]") {
    init_test_atp_bfgs();
    set_large_box_bfgs();
    g_cpu_constants.USE_ANGULAR_HBOND = false;
    g_cpu_constants.USE_TORSION_PENALTY = false;
    g_cpu_constants.USE_POST_BFGS_HBOND_CORR = false;
    g_cpu_constants.FLAG_CONSTRAINT_DOCK = false;

    // Test across multiple starting positions.  BFGS should never increase
    // energy — if the line search fails, the restore logic at line 199 of
    // bfgs_cpu.cpp kicks in.
    struct TestCase { Real x; int rec_type; int lig_type; };
    TestCase cases[] = {
        {2.0, VN_TYPE_C_H, VN_TYPE_C_H},  // clashing
        {4.0, VN_TYPE_O_A, VN_TYPE_N_D},   // near minimum
        {6.0, VN_TYPE_C_H, VN_TYPE_C_H},   // attractive tail
    };

    for (auto& tc : cases) {
        std::vector<Real> rec_coords;
        FixMol fix_mol = make_receptor_grid_bfgs(rec_coords, 0, 0, 0, tc.rec_type);
        std::vector<int> rec_types = {tc.rec_type};
        FixParamVina fix_param;
        fix_param.atom_types = rec_types.data();

        const int natom = 1;
        std::vector<Real> lig_coords = {tc.x, 0, 0};
        std::vector<int> lig_vn_types = {tc.lig_type};

        FlexTopo topo;
        topo.natom = natom;
        topo.ntorsion = 0;
        topo.vn_types = lig_vn_types.data();
        topo.axis_atoms = nullptr;
        topo.range_inds = nullptr;
        topo.range_list = nullptr;
        topo.rotated_inds = nullptr;
        topo.rotated_atoms = nullptr;

        MinimalFlexParamBFGS flex_param;
        flex_param.build(lig_vn_types, lig_coords.data());

        FlexPose pose;
        pose.coords = lig_coords.data();
        pose.dihedrals = nullptr;
        pose.center[0] = tc.x; pose.center[1] = 0; pose.center[2] = 0;
        pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
        pose.energy = 999;

        // Compute initial energy
        std::vector<Real> aux_f(natom * 3, 0);
        Real initial_energy = cal_e_f_cpu(&pose, topo, fix_mol,
                                           flex_param.param, fix_param, aux_f.data());

        BFGSThreadMemory mem;
        bfgs_with_memory_cpu(&pose, topo, fix_mol, flex_param.param, fix_param,
                              /*max_steps=*/20, mem);

        REQUIRE(std::isfinite(pose.energy));
        REQUIRE(pose.energy <= initial_energy + Real(0.01));

        free_spatial_grid(g_cpu_constants.receptor_grid);
    }
}


// ==================== BFGS idempotent on converged pose ====================

TEST_CASE("bfgs_cpu idempotent on converged pose", "[cpu][bfgs]") {
    init_test_atp_bfgs();
    set_large_box_bfgs();
    g_cpu_constants.USE_ANGULAR_HBOND = false;
    g_cpu_constants.USE_TORSION_PENALTY = false;
    g_cpu_constants.FLAG_CONSTRAINT_DOCK = false;

    std::vector<Real> rec_coords;
    FixMol fix_mol = make_receptor_grid_bfgs(rec_coords, 0, 0, 0, VN_TYPE_O_A);
    std::vector<int> rec_types = {VN_TYPE_O_A};
    FixParamVina fix_param;
    fix_param.atom_types = rec_types.data();

    const int natom = 2;
    std::vector<Real> lig_coords = {4.0, 0, 0,  3.0, 0, 0};
    std::vector<int> lig_vn_types = {VN_TYPE_N_D, VN_TYPE_H};

    FlexTopo topo;
    topo.natom = natom;
    topo.ntorsion = 0;
    topo.vn_types = lig_vn_types.data();
    topo.axis_atoms = nullptr;
    topo.range_inds = nullptr;
    topo.range_list = nullptr;
    topo.rotated_inds = nullptr;
    topo.rotated_atoms = nullptr;

    MinimalFlexParamBFGS flex_param;
    flex_param.build(lig_vn_types, lig_coords.data());

    FlexPose pose;
    pose.coords = lig_coords.data();
    pose.dihedrals = nullptr;
    pose.center[0] = 4.0; pose.center[1] = 0; pose.center[2] = 0;
    pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
    pose.energy = 999;

    // First BFGS run — converge
    BFGSThreadMemory mem;
    bfgs_with_memory_cpu(&pose, topo, fix_mol, flex_param.param, fix_param,
                          /*max_steps=*/30, mem);

    Real e1 = pose.energy;
    std::vector<Real> coords1(lig_coords.begin(), lig_coords.end());

    // Second BFGS run on the converged pose must be STABLE: the energy is
    // unchanged and atoms drift less than the convergence tolerance. It is not
    // bit-identical — line-search BFGS converges on the gradient norm, so at a
    // flat minimum (here delta-E ~ 1e-3 kcal/mol) the position is only pinned to
    // the gradient threshold and can slide a sub-0.05 A amount on re-entry.
    bfgs_with_memory_cpu(&pose, topo, fix_mol, flex_param.param, fix_param,
                          /*max_steps=*/30, mem);

    Real e2 = pose.energy;
    REQUIRE(std::abs(e2 - e1) < 0.01);

    for (int i = 0; i < natom * 3; i++) {
        REQUIRE(std::abs(lig_coords[i] - coords1[i]) < 0.05);
    }

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


// ==================== BFGS restores original when optimization fails ====================

TEST_CASE("bfgs_cpu restores original when optimization fails", "[cpu][bfgs]") {
    init_test_atp_bfgs();
    set_large_box_bfgs();
    g_cpu_constants.USE_ANGULAR_HBOND = false;
    g_cpu_constants.USE_TORSION_PENALTY = false;
    g_cpu_constants.FLAG_CONSTRAINT_DOCK = false;

    // Receptor at origin
    std::vector<Real> rec_coords;
    FixMol fix_mol = make_receptor_grid_bfgs(rec_coords, 0, 0, 0, VN_TYPE_O_A);
    std::vector<int> rec_types = {VN_TYPE_O_A};
    FixParamVina fix_param;
    fix_param.atom_types = rec_types.data();

    // Ligand far away (30 Å) — flat energy landscape, E ≈ 0
    const int natom = 2;
    std::vector<Real> lig_coords = {30.0, 0, 0,  29.0, 0, 0};
    std::vector<int> lig_vn_types = {VN_TYPE_N_D, VN_TYPE_H};

    FlexTopo topo;
    topo.natom = natom;
    topo.ntorsion = 0;
    topo.vn_types = lig_vn_types.data();
    topo.axis_atoms = nullptr;
    topo.range_inds = nullptr;
    topo.range_list = nullptr;
    topo.rotated_inds = nullptr;
    topo.rotated_atoms = nullptr;

    MinimalFlexParamBFGS flex_param;
    flex_param.build(lig_vn_types, lig_coords.data());

    FlexPose pose;
    pose.coords = lig_coords.data();
    pose.dihedrals = nullptr;
    pose.center[0] = 30.0; pose.center[1] = 0; pose.center[2] = 0;
    pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
    pose.energy = 999;

    // Compute initial energy
    std::vector<Real> aux_f(natom * 3, 0);
    Real initial_energy = cal_e_f_cpu(&pose, topo, fix_mol,
                                       flex_param.param, fix_param, aux_f.data());
    pose.energy = initial_energy;

    BFGSThreadMemory mem;
    bfgs_with_memory_cpu(&pose, topo, fix_mol, flex_param.param, fix_param,
                          /*max_steps=*/5, mem);

    // BFGS should not make energy worse (restores if E increased)
    REQUIRE(pose.energy <= initial_energy + Real(0.01));

    free_spatial_grid(g_cpu_constants.receptor_grid);
}
