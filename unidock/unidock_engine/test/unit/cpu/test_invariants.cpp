//
// Unit tests for translation/rotation invariance and box penalty FD gradient
//

#include <catch2/catch_amalgamated.hpp>
#include "cpu/common_cpu.h"
#include "cpu/score_cpu.h"
#include "cpu/spatial_grid.h"
#include "constants/constants.h"
#include "score/vina.h"
#include "geometry/quaternion.h"
#include <cmath>
#include <cstring>
#include <vector>

using Catch::Approx;

// ========== Helpers (same pattern as other CPU tests) ==========

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


// ===================== TEST 1: scoring invariant to system translation =====================

TEST_CASE("scoring invariant to system translation", "[cpu][invariant]") {
    init_test_atp();
    g_cpu_constants.USE_ANGULAR_HBOND = false;
    g_cpu_constants.USE_TORSION_PENALTY = false;
    g_cpu_constants.FLAG_CONSTRAINT_DOCK = false;

    // Box: [-10, 10] on all axes
    g_cpu_constants.BOX_X_LO = -10; g_cpu_constants.BOX_X_HI = 10;
    g_cpu_constants.BOX_Y_LO = -10; g_cpu_constants.BOX_Y_HI = 10;
    g_cpu_constants.BOX_Z_LO = -10; g_cpu_constants.BOX_Z_HI = 10;

    // Ligand: N_D at (3,0,0), H at (2,0,0)
    const int natom = 2;
    std::vector<Real> lig_coords = {3.0, 0, 0,  2.0, 0, 0};
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

    // Receptor: O_A at (0,0,0)
    std::vector<Real> rec_coords;
    FixMol fix_mol = make_receptor_grid(rec_coords, 0, 0, 0, VN_TYPE_O_A);
    std::vector<int> rec_types = {VN_TYPE_O_A};
    FixParamVina fix_param;
    fix_param.atom_types = rec_types.data();

    MinimalFlexParam flex_param;
    flex_param.build(lig_vn_types, lig_coords.data());

    FlexPose pose1;
    pose1.coords = lig_coords.data();
    pose1.dihedrals = nullptr;
    pose1.center[0] = 3.0; pose1.center[1] = 0; pose1.center[2] = 0;
    pose1.rot_vec[0] = pose1.rot_vec[1] = pose1.rot_vec[2] = 0;
    pose1.energy = 999;

    std::vector<Real> aux_f1(natom * 3, 0);
    Real E1 = cal_e_f_cpu(&pose1, topo, fix_mol, flex_param.param, fix_param, aux_f1.data());

    // --- Translate entire system by T = (100, 100, 100) ---
    const Real T = Real(100.0);

    // Shift ligand coords
    std::vector<Real> lig_coords2 = {Real(3.0) + T, T, T,  Real(2.0) + T, T, T};

    // Shift receptor and rebuild grid
    std::vector<Real> rec_coords2;
    FixMol fix_mol2 = make_receptor_grid(rec_coords2, 0 + T, 0 + T, 0 + T, VN_TYPE_O_A);

    // Shift box
    g_cpu_constants.BOX_X_LO = -10 + T; g_cpu_constants.BOX_X_HI = 10 + T;
    g_cpu_constants.BOX_Y_LO = -10 + T; g_cpu_constants.BOX_Y_HI = 10 + T;
    g_cpu_constants.BOX_Z_LO = -10 + T; g_cpu_constants.BOX_Z_HI = 10 + T;

    // Rebuild flex param with shifted coords (for bonded_h_topo)
    MinimalFlexParam flex_param2;
    flex_param2.build(lig_vn_types, lig_coords2.data());

    FlexPose pose2;
    pose2.coords = lig_coords2.data();
    pose2.dihedrals = nullptr;
    pose2.center[0] = 3.0 + T; pose2.center[1] = 0 + T; pose2.center[2] = 0 + T;
    pose2.rot_vec[0] = pose2.rot_vec[1] = pose2.rot_vec[2] = 0;
    pose2.energy = 999;

    std::vector<Real> aux_f2(natom * 3, 0);
    Real E2 = cal_e_f_cpu(&pose2, topo, fix_mol2, flex_param2.param, fix_param, aux_f2.data());

    // Energy must be identical
    REQUIRE(std::abs(E1 - E2) < 1e-4);

    // Forces must be identical
    for (int i = 0; i < natom * 3; i++) {
        REQUIRE(std::abs(aux_f1[i] - aux_f2[i]) < 1e-4);
    }

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


// ===================== TEST 2: scoring invariant to system rotation =====================

TEST_CASE("scoring invariant to system rotation", "[cpu][invariant]") {
    init_test_atp();
    set_large_box();  // [-50, 50] — no box penalty before or after rotation
    g_cpu_constants.USE_ANGULAR_HBOND = false;  // depends on absolute H geometry
    g_cpu_constants.USE_TORSION_PENALTY = false;
    g_cpu_constants.FLAG_CONSTRAINT_DOCK = false;

    // Ligand: N_D at (3,0,0), H at (2,0,0)
    const int natom = 2;
    std::vector<Real> lig_coords = {3.0, 0, 0,  2.0, 0, 0};
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

    // Receptor: O_A at (0,0,0)
    std::vector<Real> rec_coords;
    FixMol fix_mol = make_receptor_grid(rec_coords, 0, 0, 0, VN_TYPE_O_A);
    std::vector<int> rec_types = {VN_TYPE_O_A};
    FixParamVina fix_param;
    fix_param.atom_types = rec_types.data();

    MinimalFlexParam flex_param;
    flex_param.build(lig_vn_types, lig_coords.data());

    FlexPose pose1;
    pose1.coords = lig_coords.data();
    pose1.dihedrals = nullptr;
    pose1.center[0] = 3.0; pose1.center[1] = 0; pose1.center[2] = 0;
    pose1.rot_vec[0] = pose1.rot_vec[1] = pose1.rot_vec[2] = 0;
    pose1.energy = 999;

    std::vector<Real> aux_f1(natom * 3, 0);
    Real E1 = cal_e_f_cpu(&pose1, topo, fix_mol, flex_param.param, fix_param, aux_f1.data());

    // --- Rotate entire system by 90 degrees around Z axis ---
    // Rotation vector: (0, 0, PI/2)
    const Real PI_VAL = Real(3.1415927);
    Real rot_vec[3] = {0, 0, PI_VAL / 2};
    Real q[4];
    rotvec_to_quaternion(q, rot_vec);

    // Rotate ligand coords
    std::vector<Real> lig_coords2(natom * 3);
    for (int i = 0; i < natom; i++) {
        Real v[3] = {lig_coords[i*3], lig_coords[i*3+1], lig_coords[i*3+2]};
        rotate_vec_by_quaternion(v, q);
        lig_coords2[i*3]   = v[0];
        lig_coords2[i*3+1] = v[1];
        lig_coords2[i*3+2] = v[2];
    }

    // Rotate receptor coords and rebuild grid
    Real rec_rotated[3] = {rec_coords[0], rec_coords[1], rec_coords[2]};
    rotate_vec_by_quaternion(rec_rotated, q);
    std::vector<Real> rec_coords2;
    FixMol fix_mol2 = make_receptor_grid(rec_coords2, rec_rotated[0], rec_rotated[1], rec_rotated[2], VN_TYPE_O_A);

    // Rebuild flex param with rotated coords
    MinimalFlexParam flex_param2;
    flex_param2.build(lig_vn_types, lig_coords2.data());

    // Compute rotated centroid for pose.center
    Real center_rot[3] = {3.0, 0, 0};
    rotate_vec_by_quaternion(center_rot, q);

    FlexPose pose2;
    pose2.coords = lig_coords2.data();
    pose2.dihedrals = nullptr;
    pose2.center[0] = center_rot[0]; pose2.center[1] = center_rot[1]; pose2.center[2] = center_rot[2];
    pose2.rot_vec[0] = pose2.rot_vec[1] = pose2.rot_vec[2] = 0;
    pose2.energy = 999;

    std::vector<Real> aux_f2(natom * 3, 0);
    Real E2 = cal_e_f_cpu(&pose2, topo, fix_mol2, flex_param2.param, fix_param, aux_f2.data());

    // Energy must be identical (Vina scoring is distance-based)
    REQUIRE(std::abs(E1 - E2) < 1e-4);

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


// ===================== TEST 3: box penalty FD gradient check =====================

TEST_CASE("box penalty FD gradient check", "[cpu][invariant]") {
    const Real PENALTY_SLOPE = g_cpu_constants.PENALTY_SLOPE;  // default 10.0

    // Box: [0, 10] on all axes
    g_cpu_constants.BOX_X_LO = 0; g_cpu_constants.BOX_X_HI = 10;
    g_cpu_constants.BOX_Y_LO = 0; g_cpu_constants.BOX_Y_HI = 10;
    g_cpu_constants.BOX_Z_LO = 0; g_cpu_constants.BOX_Z_HI = 10;

    SECTION("atom inside box: penalty = 0, forces = 0") {
        Real x = 5, y = 5, z = 5;
        Real f[3] = {0, 0, 0};
        Real penalty = cal_box_penalty_clamp_coord_cpu(&x, &y, &z, f);

        REQUIRE(penalty == Approx(0.0));
        REQUIRE(f[0] == Approx(0.0));
        REQUIRE(f[1] == Approx(0.0));
        REQUIRE(f[2] == Approx(0.0));
    }

    SECTION("atom outside +X: correct penalty and force, FD check") {
        // 2 A outside +X boundary
        Real x = 12, y = 5, z = 5;
        Real f[3] = {0, 0, 0};
        Real penalty = cal_box_penalty_clamp_coord_cpu(&x, &y, &z, f);

        // penalty = distance_outside * PENALTY_SLOPE = 2 * 10 = 20
        REQUIRE(penalty == Approx(2.0 * PENALTY_SLOPE));
        // Force on X: +PENALTY_SLOPE (pushes back inside)
        REQUIRE(f[0] == Approx(PENALTY_SLOPE));
        REQUIRE(f[1] == Approx(0.0));
        REQUIRE(f[2] == Approx(0.0));

        // FD check: perturb x by +/-h, recompute penalty
        const Real h = Real(0.01);

        Real xp = 12 + h, yp = 5.0, zp = 5.0;
        Real fp[3] = {0, 0, 0};
        Real ep = cal_box_penalty_clamp_coord_cpu(&xp, &yp, &zp, fp);

        Real xm = 12 - h, ym = 5.0, zm = 5.0;
        Real fm[3] = {0, 0, 0};
        Real em = cal_box_penalty_clamp_coord_cpu(&xm, &ym, &zm, fm);

        // dE/dx via FD — note force is -dE/dx, so analytic force = -fd
        Real fd = (ep - em) / (2 * h);
        REQUIRE(f[0] == Approx(fd).margin(0.1));
    }

    SECTION("atom outside -Y: correct penalty and force, FD check") {
        // 3 A outside -Y boundary
        Real x = 5, y = -3, z = 5;
        Real f[3] = {0, 0, 0};
        Real penalty = cal_box_penalty_clamp_coord_cpu(&x, &y, &z, f);

        REQUIRE(penalty == Approx(3.0 * PENALTY_SLOPE));
        // Force on Y: -PENALTY_SLOPE (pushes back toward +Y, i.e. inside)
        REQUIRE(f[1] == Approx(-PENALTY_SLOPE));

        // FD check on Y
        const Real h = Real(0.01);

        Real x1 = 5.0, yp = -3 + h, z1 = 5.0;
        Real fp[3] = {0, 0, 0};
        Real ep = cal_box_penalty_clamp_coord_cpu(&x1, &yp, &z1, fp);

        Real x2 = 5.0, ym = -3 - h, z2 = 5.0;
        Real fm[3] = {0, 0, 0};
        Real em = cal_box_penalty_clamp_coord_cpu(&x2, &ym, &z2, fm);

        Real fd = (ep - em) / (2 * h);
        REQUIRE(f[1] == Approx(fd).margin(0.1));
    }

    SECTION("atom outside on two axes simultaneously") {
        // Outside +X by 2, outside -Y by 3, Z inside
        Real x = 12, y = -3, z = 5;
        Real f[3] = {0, 0, 0};
        Real penalty = cal_box_penalty_clamp_coord_cpu(&x, &y, &z, f);

        // Total penalty = (2 + 3) * PENALTY_SLOPE
        REQUIRE(penalty == Approx(5.0 * PENALTY_SLOPE));
        REQUIRE(f[0] == Approx(PENALTY_SLOPE));
        REQUIRE(f[1] == Approx(-PENALTY_SLOPE));
        REQUIRE(f[2] == Approx(0.0));

        // FD check on all 3 axes
        const Real h = Real(0.01);
        Real coords[3] = {12, -3, 5};

        for (int axis = 0; axis < 3; axis++) {
            Real cp[3] = {coords[0], coords[1], coords[2]};
            Real cm[3] = {coords[0], coords[1], coords[2]};
            cp[axis] += h;
            cm[axis] -= h;

            Real fp[3] = {0, 0, 0}, fm_arr[3] = {0, 0, 0};
            Real ep = cal_box_penalty_clamp_coord_cpu(&cp[0], &cp[1], &cp[2], fp);
            Real em = cal_box_penalty_clamp_coord_cpu(&cm[0], &cm[1], &cm[2], fm_arr);

            Real fd = (ep - em) / (2 * h);
            REQUIRE(f[axis] == Approx(fd).margin(0.1));
        }
    }
}


// ===================== TEST 4: full cal_e_f_cpu box penalty gradient =====================

TEST_CASE("full cal_e_f_cpu box penalty gradient", "[cpu][invariant]") {
    init_test_atp();
    g_cpu_constants.USE_ANGULAR_HBOND = false;
    g_cpu_constants.USE_TORSION_PENALTY = false;
    g_cpu_constants.FLAG_CONSTRAINT_DOCK = false;

    // Box: [-10, 10]
    g_cpu_constants.BOX_X_LO = -10; g_cpu_constants.BOX_X_HI = 10;
    g_cpu_constants.BOX_Y_LO = -10; g_cpu_constants.BOX_Y_HI = 10;
    g_cpu_constants.BOX_Z_LO = -10; g_cpu_constants.BOX_Z_HI = 10;

    // 1-atom ligand: C_H at (10.5, 0, 0) — 0.5 A outside +X
    const int natom = 1;
    std::vector<Real> lig_coords = {10.5, 0, 0};
    std::vector<int> lig_vn_types = {VN_TYPE_C_H};

    FlexTopo topo;
    topo.natom = natom;
    topo.ntorsion = 0;
    topo.vn_types = lig_vn_types.data();
    topo.axis_atoms = nullptr;
    topo.range_inds = nullptr;
    topo.range_list = nullptr;
    topo.rotated_inds = nullptr;
    topo.rotated_atoms = nullptr;

    // Receptor: C_H at (0, 0, 0) — far from ligand (minimal Vina interaction)
    std::vector<Real> rec_coords;
    FixMol fix_mol = make_receptor_grid(rec_coords, 0, 0, 0, VN_TYPE_C_H);
    std::vector<int> rec_types = {VN_TYPE_C_H};
    FixParamVina fix_param;
    fix_param.atom_types = rec_types.data();

    MinimalFlexParam flex_param;
    flex_param.build(lig_vn_types, lig_coords.data());

    FlexPose pose;
    pose.coords = lig_coords.data();
    pose.dihedrals = nullptr;
    pose.center[0] = 10.5; pose.center[1] = 0; pose.center[2] = 0;
    pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
    pose.energy = 999;

    std::vector<Real> aux_f(natom * 3, 0);
    Real E0 = cal_e_f_cpu(&pose, topo, fix_mol, flex_param.param, fix_param, aux_f.data());

    // FD check on X coordinate (the one outside the box)
    const Real h = Real(1e-4);

    std::vector<Real> coords_p = {Real(10.5) + h, 0, 0};
    FlexPose pose_p;
    pose_p.coords = coords_p.data();
    pose_p.dihedrals = nullptr;
    pose_p.center[0] = 10.5 + h; pose_p.center[1] = 0; pose_p.center[2] = 0;
    pose_p.rot_vec[0] = pose_p.rot_vec[1] = pose_p.rot_vec[2] = 0;
    std::vector<Real> f_p(natom * 3, 0);
    Real Ep = cal_e_f_cpu(&pose_p, topo, fix_mol, flex_param.param, fix_param, f_p.data());

    std::vector<Real> coords_m = {Real(10.5) - h, 0, 0};
    FlexPose pose_m;
    pose_m.coords = coords_m.data();
    pose_m.dihedrals = nullptr;
    pose_m.center[0] = 10.5 - h; pose_m.center[1] = 0; pose_m.center[2] = 0;
    pose_m.rot_vec[0] = pose_m.rot_vec[1] = pose_m.rot_vec[2] = 0;
    std::vector<Real> f_m(natom * 3, 0);
    Real Em = cal_e_f_cpu(&pose_m, topo, fix_mol, flex_param.param, fix_param, f_m.data());

    Real fd = (Ep - Em) / (Real(2.0) * h);
    // aux_f stores dE/dx (gradient direction), FD also computes dE/dx
    REQUIRE_THAT(aux_f[0], Catch::Matchers::WithinAbs(fd, 0.5));

    free_spatial_grid(g_cpu_constants.receptor_grid);
}
