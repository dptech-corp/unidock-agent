//
// Unit tests for cal_e_f_cpu + cal_grad_cpu combined FD gradient check
// Tests the full scoring pipeline (inter + intra + box penalty) against
// finite-difference derivatives on atomic coordinates and dihedrals.
//

#include <catch2/catch_amalgamated.hpp>
#include "cpu/common_cpu.h"
#include "cpu/score_cpu.h"
#include "cpu/spatial_grid.h"
#include "constants/constants.h"
#include "score/vina.h"
#include <cmath>
#include <cstring>
#include <vector>

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


// ===================== TEST 1: rigid 2-atom ligand FD check =====================

TEST_CASE("cal_e_grad_cpu FD check: rigid 2-atom ligand", "[cpu][gradient]") {
    init_test_atp();
    set_large_box();
    g_cpu_constants.USE_ANGULAR_HBOND = false;
    g_cpu_constants.USE_TORSION_PENALTY = false;
    g_cpu_constants.FLAG_CONSTRAINT_DOCK = false;

    // Receptor: O_A at origin
    std::vector<Real> rec_coords;
    FixMol fix_mol = make_receptor_grid(rec_coords, 0, 0, 0, VN_TYPE_O_A);
    std::vector<int> rec_types = {VN_TYPE_O_A};
    FixParamVina fix_param;
    fix_param.atom_types = rec_types.data();

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

    MinimalFlexParam flex_param;
    flex_param.build(lig_vn_types, lig_coords.data());

    // Compute energy + forces at reference position
    FlexPose pose;
    pose.coords = lig_coords.data();
    pose.dihedrals = nullptr;
    pose.center[0] = 3.0; pose.center[1] = 0; pose.center[2] = 0;
    pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
    pose.energy = 999;

    std::vector<Real> aux_f(natom * 3, 0);
    cal_e_f_cpu(&pose, topo, fix_mol, flex_param.param, fix_param, aux_f.data());

    // FD check on all 6 ligand coordinates (3 per atom)
    const Real h = Real(1e-4);
    for (int i = 0; i < natom * 3; i++) {
        std::vector<Real> coords_p(lig_coords);
        std::vector<Real> coords_m(lig_coords);
        coords_p[i] += h;
        coords_m[i] -= h;

        FlexPose pose_p;
        pose_p.coords = coords_p.data();
        pose_p.dihedrals = nullptr;
        pose_p.center[0] = 3.0; pose_p.center[1] = 0; pose_p.center[2] = 0;
        pose_p.rot_vec[0] = pose_p.rot_vec[1] = pose_p.rot_vec[2] = 0;

        FlexPose pose_m;
        pose_m.coords = coords_m.data();
        pose_m.dihedrals = nullptr;
        pose_m.center[0] = 3.0; pose_m.center[1] = 0; pose_m.center[2] = 0;
        pose_m.rot_vec[0] = pose_m.rot_vec[1] = pose_m.rot_vec[2] = 0;

        std::vector<Real> f_p(natom * 3, 0), f_m(natom * 3, 0);
        Real e_plus = cal_e_f_cpu(&pose_p, topo, fix_mol, flex_param.param,
                                   fix_param, f_p.data());
        Real e_minus = cal_e_f_cpu(&pose_m, topo, fix_mol, flex_param.param,
                                    fix_param, f_m.data());

        Real fd = (e_plus - e_minus) / (Real(2.0) * h);
        // aux_f stores dE/dx (gradient, not force)
        REQUIRE_THAT(aux_f[i], Catch::Matchers::WithinAbs(fd, 5e-2));
    }

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


// ===================== TEST 2: 3-atom ligand with 1 torsion FD check =====================

TEST_CASE("cal_e_grad_cpu FD check: 3-atom ligand with 1 torsion", "[cpu][gradient]") {
    init_test_atp();
    set_large_box();
    g_cpu_constants.USE_ANGULAR_HBOND = false;
    g_cpu_constants.USE_TORSION_PENALTY = false;
    g_cpu_constants.FLAG_CONSTRAINT_DOCK = false;

    // Receptor: C_H at origin
    std::vector<Real> rec_coords;
    FixMol fix_mol = make_receptor_grid(rec_coords, 0, 0, 0, VN_TYPE_C_H);
    std::vector<int> rec_types = {VN_TYPE_C_H};
    FixParamVina fix_param;
    fix_param.atom_types = rec_types.data();

    // Ligand: 3 atoms with 1 torsion
    // atom0=N_D at (3,0,0), atom1=C_P at (4.5,0,0), atom2=O_A at (6,0,0.5)
    const int natom = 3;
    const int ntorsion = 1;
    std::vector<Real> lig_coords = {3.0, 0.0, 0.0,
                                     4.5, 0.0, 0.0,
                                     6.0, 0.0, 0.5};
    std::vector<int> lig_vn_types = {VN_TYPE_N_D, VN_TYPE_C_P, VN_TYPE_O_A};

    // Torsion: axis = (0, 1), rotated_atoms = {2}
    std::vector<int> axis_atoms = {0, 1};
    std::vector<int> rotated_inds = {0, 0};  // inclusive [begin, end]: torsion 0 rotates rotated_atoms[0]
    std::vector<int> rotated_atoms = {2};
    std::vector<int> range_inds = {0, 0};    // no range constraints
    std::vector<Real> range_list;

    FlexTopo topo;
    topo.natom = natom;
    topo.ntorsion = ntorsion;
    topo.vn_types = lig_vn_types.data();
    topo.axis_atoms = axis_atoms.data();
    topo.range_inds = range_inds.data();
    topo.range_list = range_list.data();
    topo.rotated_inds = rotated_inds.data();
    topo.rotated_atoms = rotated_atoms.data();

    MinimalFlexParam flex_param;
    flex_param.build(lig_vn_types, lig_coords.data());

    std::vector<Real> dihedrals = {Real(0.3)};

    FlexPose pose;
    pose.coords = lig_coords.data();
    pose.dihedrals = dihedrals.data();
    pose.center[0] = 3.0; pose.center[1] = 0; pose.center[2] = 0;
    pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
    pose.energy = 999;

    // Compute reference energy + forces + gradient
    std::vector<Real> aux_f(natom * 3, 0);
    std::vector<Real> dihedrals_g(ntorsion, 0);
    FlexPoseGradient out_g;
    out_g.dihedrals_g = dihedrals_g.data();
    Real e_ref = cal_e_grad_cpu(&pose, &out_g, topo, fix_mol,
                                 flex_param.param, fix_param, aux_f.data());

    SECTION("coordinate FD check") {
        const Real h = Real(1e-4);
        for (int i = 0; i < natom * 3; i++) {
            std::vector<Real> coords_p(lig_coords);
            std::vector<Real> coords_m(lig_coords);
            coords_p[i] += h;
            coords_m[i] -= h;

            FlexPose pose_p;
            std::vector<Real> dihe_p(dihedrals);
            pose_p.coords = coords_p.data();
            pose_p.dihedrals = dihe_p.data();
            pose_p.center[0] = 3.0; pose_p.center[1] = 0; pose_p.center[2] = 0;
            pose_p.rot_vec[0] = pose_p.rot_vec[1] = pose_p.rot_vec[2] = 0;

            FlexPose pose_m;
            std::vector<Real> dihe_m(dihedrals);
            pose_m.coords = coords_m.data();
            pose_m.dihedrals = dihe_m.data();
            pose_m.center[0] = 3.0; pose_m.center[1] = 0; pose_m.center[2] = 0;
            pose_m.rot_vec[0] = pose_m.rot_vec[1] = pose_m.rot_vec[2] = 0;

            std::vector<Real> f_p(natom * 3, 0), f_m(natom * 3, 0);
            Real e_plus = cal_e_f_cpu(&pose_p, topo, fix_mol, flex_param.param,
                                       fix_param, f_p.data());
            Real e_minus = cal_e_f_cpu(&pose_m, topo, fix_mol, flex_param.param,
                                        fix_param, f_m.data());

            Real fd = (e_plus - e_minus) / (Real(2.0) * h);
            // aux_f stores dE/dx (gradient, not force)
            REQUIRE_THAT(aux_f[i], Catch::Matchers::WithinAbs(fd, 5e-2));
        }
    }

    SECTION("dihedral FD check") {
        const Real h = Real(1e-4);

        // Plus side: perturb dihedral and recompute coordinates
        std::vector<Real> coords_p(lig_coords);
        std::vector<Real> dihe_p(dihedrals);
        FlexPose pose_p;
        pose_p.coords = coords_p.data();
        pose_p.dihedrals = dihe_p.data();
        pose_p.center[0] = 3.0; pose_p.center[1] = 0; pose_p.center[2] = 0;
        pose_p.rot_vec[0] = pose_p.rot_vec[1] = pose_p.rot_vec[2] = 0;
        apply_grad_update_dihe_cpu(&pose_p, &topo, 0, h, /*skip_range_clamp=*/true);
        std::vector<Real> f_p(natom * 3, 0);
        Real e_plus = cal_e_f_cpu(&pose_p, topo, fix_mol, flex_param.param,
                                   fix_param, f_p.data());

        // Minus side
        std::vector<Real> coords_m(lig_coords);
        std::vector<Real> dihe_m(dihedrals);
        FlexPose pose_m;
        pose_m.coords = coords_m.data();
        pose_m.dihedrals = dihe_m.data();
        pose_m.center[0] = 3.0; pose_m.center[1] = 0; pose_m.center[2] = 0;
        pose_m.rot_vec[0] = pose_m.rot_vec[1] = pose_m.rot_vec[2] = 0;
        apply_grad_update_dihe_cpu(&pose_m, &topo, 0, -h, /*skip_range_clamp=*/true);
        std::vector<Real> f_m(natom * 3, 0);
        Real e_minus = cal_e_f_cpu(&pose_m, topo, fix_mol, flex_param.param,
                                    fix_param, f_m.data());

        Real fd = (e_plus - e_minus) / (Real(2.0) * h);
        // Dihedral gradient has higher noise due to coordinate reconstruction
        REQUIRE_THAT(dihedrals_g[0], Catch::Matchers::WithinAbs(fd, 0.01));
    }

    free_spatial_grid(g_cpu_constants.receptor_grid);
}
