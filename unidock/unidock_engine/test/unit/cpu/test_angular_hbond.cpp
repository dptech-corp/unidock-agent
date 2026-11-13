//
// Unit tests for angular HBond scoring components:
//   1. angular_hbond_weight() — known angle values + finite-difference derivative
//   2. build_bonded_h_topo() — CSR construction correctness
//   3. RecDonorHTopo — manual construction + cleanup
//   4. cal_e_f_cpu() — FD gradient check, direction 1 (ligand donor)
//   5. cal_e_f_cpu() — FD gradient check, direction 2 (receptor donor)
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ========== Helpers (same pattern as test_hbond_angle.cpp) ==========

static void init_test_atp() {
    init_default_atom_type_props(g_cpu_constants.atp);
    int n = sizeof(VN_VDW_RADII) / sizeof(VN_VDW_RADII[0]);
    for (int i = 0; i < n; i++) g_cpu_constants.vdw_radii[i] = VN_VDW_RADII[i];
    for (int i = n; i < MAX_VN_TYPES; i++) g_cpu_constants.vdw_radii[i] = Real(0);
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

// Set box bounds large enough so all test atoms are inside (no penalty)
static void set_large_box() {
    g_cpu_constants.BOX_X_LO = -50; g_cpu_constants.BOX_X_HI = 50;
    g_cpu_constants.BOX_Y_LO = -50; g_cpu_constants.BOX_Y_HI = 50;
    g_cpu_constants.BOX_Z_LO = -50; g_cpu_constants.BOX_Z_HI = 50;
}

// Build a minimal FlexParamVina for a ligand with given atom_types and bonded_h_topo.
// No intra/inter pairs, no bias.
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


// ======================== TEST 1: angular_hbond_weight ========================

TEST_CASE("angular_hbond_weight: known angle values", "[angular_hbond]") {
    // cos_min = -0.17365 (cos 100 deg), inv_range = 1/(cos_min + 1) = 1/0.82635
    // Linear ramp in cos between 90 deg (w = 0) and 140 deg (w = 1)
    const Real cos_zero = Real(ANGULAR_HBOND_COS_ZERO);
    const Real cos_full = Real(ANGULAR_HBOND_COS_FULL);
    const Real inv_range = Real(1.0) / (cos_zero - cos_full);

    SECTION("cos=-1 (180 deg) => w=1, dw=0 (clamped at t>=1)") {
        Real dw = -999;
        Real w = angular_hbond_weight(Real(-1.0), &dw);
        REQUIRE(w == Approx(1.0));
        REQUIRE(dw == Approx(0.0));
    }

    SECTION("cos=cos_full (140 deg) => w=1, dw=0 (clamped at t>=1)") {
        Real dw = -999;
        Real w = angular_hbond_weight(cos_full, &dw);
        REQUIRE(w == Approx(1.0));
        REQUIRE(dw == Approx(0.0));
    }

    SECTION("cos=cos_zero (90 deg) => w=0, dw=0 (clamped at t<=0)") {
        Real dw = -999;
        Real w = angular_hbond_weight(cos_zero, &dw);
        REQUIRE(w == Approx(0.0));
        REQUIRE(dw == Approx(0.0));
    }

    SECTION("cos=+0.5 (60 deg) => well below threshold, w=0, dw=0") {
        Real dw = -999;
        Real w = angular_hbond_weight(Real(0.5), &dw);
        REQUIRE(w == Approx(0.0));
        REQUIRE(dw == Approx(0.0));
    }

    SECTION("cos=-0.5 (120 deg) => t = (cos_zero - (-0.5))*inv_range") {
        Real cos_dha = Real(-0.5);
        Real t = (cos_zero - cos_dha) * inv_range;
        // t = 0.5 / 0.76604 ~ 0.6527
        REQUIRE(t > 0);
        REQUIRE(t < 1);

        Real dw = 0;
        Real w = angular_hbond_weight(cos_dha, &dw);
        REQUIRE(w == Approx(t).margin(1e-5));
        REQUIRE(dw == Approx(-inv_range).margin(1e-5));
    }

    SECTION("cos=-0.2588 (105 deg) => intermediate") {
        Real cos_dha = Real(-0.2588);
        Real t = (cos_zero - cos_dha) * inv_range;
        REQUIRE(t > 0);
        REQUIRE(t < 1);

        Real dw = 0;
        Real w = angular_hbond_weight(cos_dha, &dw);
        REQUIRE(w == Approx(t).margin(1e-5));
        REQUIRE(dw == Approx(-inv_range).margin(1e-5));
    }
}


TEST_CASE("angular_hbond_weight: derivative matches finite difference", "[angular_hbond]") {
    const Real eps = Real(1e-5);

    // Test at several cos values in the smooth region (cos_min < cos < -1 not possible,
    // so test in the active range: cos in (-1, cos_min) where t is in (0,1))
    std::vector<Real> test_cos = {Real(-0.1), Real(-0.3), Real(-0.5), Real(-0.7)};

    for (Real c : test_cos) {
        Real dw_analytic = 0;
        angular_hbond_weight(c, &dw_analytic);

        Real dw_plus = 0, dw_minus = 0;
        Real w_plus = angular_hbond_weight(c + eps, &dw_plus);
        Real w_minus = angular_hbond_weight(c - eps, &dw_minus);

        Real dw_numeric = (w_plus - w_minus) / (2 * eps);
        REQUIRE(dw_analytic == Approx(dw_numeric).margin(5e-3));
    }
}


// ======================== TEST 2: build_bonded_h_topo ========================

TEST_CASE("build_bonded_h_topo: CSR construction", "[angular_hbond]") {
    init_test_atp();

    SECTION("Case A: one donor + one bonded H (d=1.0 < 1.3)") {
        // atom 0: N_D at (0,0,0), atom 1: H at (0,0,1)
        Real coords[] = {0, 0, 0,  0, 0, 1};
        int types[] = {VN_TYPE_N_D, VN_TYPE_H};
        BondedHTopo topo = {};
        build_bonded_h_topo(topo, coords, types, 2);

        REQUIRE(topo.natom == 2);
        REQUIRE(topo.total_h == 1);
        // N_D (atom 0) has 1 bonded H
        REQUIRE(topo.h_start[0] == 0);
        REQUIRE(topo.h_start[1] == 1);
        // H (atom 1) is hydrogen, so no bonded H counted for it
        REQUIRE(topo.h_start[2] == 1);
        REQUIRE(topo.h_indices[0] == 1);

        free_bonded_h_topo(topo);
    }

    SECTION("Case B: one donor + one H too far (d=2.0 > 1.3)") {
        Real coords[] = {0, 0, 0,  0, 0, 2};
        int types[] = {VN_TYPE_N_D, VN_TYPE_H};
        BondedHTopo topo = {};
        build_bonded_h_topo(topo, coords, types, 2);

        REQUIRE(topo.natom == 2);
        REQUIRE(topo.total_h == 0);
        REQUIRE(topo.h_start[0] == 0);
        REQUIRE(topo.h_start[1] == 0);
        REQUIRE(topo.h_start[2] == 0);

        free_bonded_h_topo(topo);
    }

    SECTION("Case C: NH3-like donor with 3 bonded H") {
        // atom 0: N_D at origin, atoms 1-3: H at ~1.0 A away
        Real coords[] = {
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        };
        int types[] = {VN_TYPE_N_D, VN_TYPE_H, VN_TYPE_H, VN_TYPE_H};
        BondedHTopo topo = {};
        build_bonded_h_topo(topo, coords, types, 4);

        REQUIRE(topo.natom == 4);
        REQUIRE(topo.total_h == 3);
        REQUIRE(topo.h_start[0] == 0);
        REQUIRE(topo.h_start[1] == 3);  // N_D has 3 bonded H
        // Remaining atoms are H, so their entries should be unchanged
        REQUIRE(topo.h_start[4] == 3);

        // Check indices contain {1, 2, 3} in some order
        std::vector<int> h_ids(topo.h_indices, topo.h_indices + 3);
        std::sort(h_ids.begin(), h_ids.end());
        REQUIRE(h_ids == std::vector<int>{1, 2, 3});

        free_bonded_h_topo(topo);
    }

    SECTION("Case D: no H atoms in molecule") {
        Real coords[] = {0, 0, 0,  3, 0, 0};
        int types[] = {VN_TYPE_N_D, VN_TYPE_C_H};
        BondedHTopo topo = {};
        build_bonded_h_topo(topo, coords, types, 2);

        REQUIRE(topo.natom == 2);
        REQUIRE(topo.total_h == 0);
        REQUIRE(topo.h_start[0] == 0);
        REQUIRE(topo.h_start[1] == 0);
        REQUIRE(topo.h_start[2] == 0);

        free_bonded_h_topo(topo);
    }

    SECTION("Case E: H just beyond 1.3 A boundary is excluded") {
        // Place H at 1.31 A — clearly beyond 1.3 A cutoff
        Real coords[] = {0, 0, 0,  Real(1.31), 0, 0};
        int types[] = {VN_TYPE_N_D, VN_TYPE_H};
        BondedHTopo topo = {};
        build_bonded_h_topo(topo, coords, types, 2);

        REQUIRE(topo.total_h == 0);

        free_bonded_h_topo(topo);
    }
}


// ======================== TEST 3: RecDonorHTopo manual construction ========================

TEST_CASE("RecDonorHTopo: manual construction and cleanup", "[angular_hbond]") {
    // Manually build a RecDonorHTopo for 1 receptor donor with 1 bonded H
    RecDonorHTopo topo;
    topo.natom = 2;  // e.g., 2 heavy atoms in receptor, only index 0 is donor
    topo.total_h = 1;
    topo.h_start = new int[3];
    topo.h_start[0] = 0;  // atom 0 starts at 0
    topo.h_start[1] = 1;  // atom 0 has 1 H
    topo.h_start[2] = 1;  // atom 1 has 0 H
    topo.h_coords = new Real[3];
    topo.h_coords[0] = Real(1.0);  // H at (1, 0, 0)
    topo.h_coords[1] = Real(0.0);
    topo.h_coords[2] = Real(0.0);

    // Verify structure
    REQUIRE(topo.natom == 2);
    REQUIRE(topo.total_h == 1);
    REQUIRE(topo.h_start[0] == 0);
    REQUIRE(topo.h_start[1] == 1);
    REQUIRE(topo.h_start[2] == 1);
    REQUIRE(topo.h_coords[0] == Approx(1.0));
    REQUIRE(topo.h_coords[1] == Approx(0.0));
    REQUIRE(topo.h_coords[2] == Approx(0.0));

    // Cleanup
    free_rec_donor_h_topo(topo);
    REQUIRE(topo.h_start == nullptr);
    REQUIRE(topo.h_coords == nullptr);
    REQUIRE(topo.natom == 0);
    REQUIRE(topo.total_h == 0);
}


// ======================== TEST 4: FD gradient — direction 1 (ligand donor) ========================

// Helper to compute energy via cal_e_f_cpu with a given ligand coords buffer.
// Caller provides all persistent structures; only pose_coords is modified.
static Real compute_energy(FlexPose& pose, const FlexTopo& topo,
                           const FixMol& fix, const FlexParamVina& fparam,
                           const FixParamVina& fixparam, int natom) {
    std::vector<Real> aux_f(natom * 3, 0);
    return cal_e_f_cpu(&pose, topo, fix, fparam, fixparam, aux_f.data());
}

TEST_CASE("cal_e_f_cpu: FD gradient for angular HBond direction 1 (lig donor)", "[angular_hbond]") {
    init_test_atp();
    set_large_box();
    g_cpu_constants.USE_ANGULAR_HBOND = true;

    // Geometry: 3-atom ligand [N_D donor, H, C_H filler]
    // The filler is placed far away so it doesn't interact.
    // Receptor: 1 O_A acceptor
    // RecDonorHTopo: empty (direction 2 inactive)

    struct TestGeometry {
        const char* name;
        Real dx, dy, dz;    // donor position
        Real hx, hy, hz;    // H position
        Real rx, ry, rz;    // receptor acceptor position
    };

    // Choose geometries where D-H...A angle varies.
    // D at origin, H at (0,0,1). Place receptor at different positions.
    TestGeometry geos[] = {
        // ~175 deg: nearly linear but slightly off-axis to avoid degenerate zero transverse grad
        {"175deg", 0, 0, 0,  0, 0, 1,  Real(0.1), Real(0.05), Real(3.5)},
        // ~140 deg: cos(140)=-0.766, use off-axis position
        {"140deg", 0, 0, 0,  0, 0, 1,  Real(1.5), 0, Real(2.5)},
        // ~120 deg: cos(120)=-0.5, further off-axis
        {"120deg", 0, 0, 0,  0, 0, 1,  Real(2.0), 0, Real(2.0)},
    };

    // Set up empty rec_donor_h_topo
    RecDonorHTopo saved_rh = g_cpu_constants.rec_donor_h_topo;
    RecDonorHTopo empty_rh = {};
    g_cpu_constants.rec_donor_h_topo = empty_rh;

    for (auto& geo : geos) {
        DYNAMIC_SECTION("geometry: " << geo.name) {
            // Build receptor grid
            std::vector<Real> rec_coords;
            FixMol fix = make_receptor_grid(rec_coords, geo.rx, geo.ry, geo.rz, VN_TYPE_O_A);

            FixParamVina fixparam;
            std::vector<int> fix_atom_types = {VN_TYPE_O_A};
            fixparam.atom_types = fix_atom_types.data();

            // Build ligand
            const int natom = 3;
            std::vector<Real> coords = {
                geo.dx, geo.dy, geo.dz,
                geo.hx, geo.hy, geo.hz,
                Real(20), Real(20), Real(20)  // filler far away
            };
            std::vector<int> vn_types = {VN_TYPE_N_D, VN_TYPE_H, VN_TYPE_C_H};

            FlexPose pose;
            pose.coords = coords.data();
            pose.dihedrals = nullptr;
            pose.center[0] = geo.dx; pose.center[1] = geo.dy; pose.center[2] = geo.dz;
            pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
            pose.energy = 0;

            FlexTopo topo;
            topo.natom = natom;
            topo.ntorsion = 0;
            topo.vn_types = vn_types.data();
            topo.axis_atoms = nullptr;
            topo.range_inds = nullptr;
            topo.range_list = nullptr;
            topo.rotated_inds = nullptr;
            topo.rotated_atoms = nullptr;

            MinimalFlexParam mfp;
            mfp.build(vn_types, coords.data());

            // Compute analytical gradient
            std::vector<Real> aux_f(natom * 3, 0);
            Real e0 = cal_e_f_cpu(&pose, topo, fix, mfp.param, fixparam, aux_f.data());
            REQUIRE(std::isfinite(e0));

            // Finite-difference check
            const Real delta = Real(5e-4);
            int n_checked = 0, n_sign_ok = 0, n_rel_ok = 0;

            // Only check atoms 0 (donor) and 1 (H) — atom 2 is far away
            for (int i = 0; i < 2; i++) {
                for (int d = 0; d < 3; d++) {
                    int idx = i * 3 + d;
                    Real orig = coords[idx];

                    coords[idx] = orig + delta;
                    Real e_plus = compute_energy(pose, topo, fix, mfp.param, fixparam, natom);

                    coords[idx] = orig - delta;
                    Real e_minus = compute_energy(pose, topo, fix, mfp.param, fixparam, natom);

                    coords[idx] = orig;

                    Real dE_numeric = (e_plus - e_minus) / (2 * delta);
                    Real dE_analytic = aux_f[idx];

                    n_checked++;

                    // Skip near-zero gradients (noise-dominated for float32)
                    if (std::abs(dE_analytic) < Real(0.05) && std::abs(dE_numeric) < Real(0.05)) {
                        n_sign_ok++;
                        n_rel_ok++;
                        continue;
                    }

                    // Sign agreement
                    if (dE_analytic * dE_numeric >= 0) n_sign_ok++;

                    // Relative error
                    Real abs_diff = std::abs(dE_analytic - dE_numeric);
                    Real scale = std::max(Real(0.01),
                                 std::max(std::abs(dE_analytic), std::abs(dE_numeric)));
                    if (abs_diff / scale < Real(0.1) || abs_diff < Real(0.05)) {
                        n_rel_ok++;
                    }
                }
            }

            // All 6 components of donor + H should pass
            REQUIRE(n_sign_ok == n_checked);
            REQUIRE(n_rel_ok == n_checked);

            free_spatial_grid(g_cpu_constants.receptor_grid);
        }
    }

    g_cpu_constants.rec_donor_h_topo = saved_rh;
}


// ======================== TEST 5: FD gradient — direction 2 (receptor donor) ========================

TEST_CASE("cal_e_f_cpu: FD gradient for angular HBond direction 2 (rec donor)", "[angular_hbond]") {
    init_test_atp();
    set_large_box();
    g_cpu_constants.USE_ANGULAR_HBOND = true;

    // Geometry: 2-atom ligand [O_A acceptor, C_H filler]
    // Receptor: 1 N_D donor
    // RecDonorHTopo: 1 receptor donor (index 0) with 1 bonded H
    // BondedHTopo: built from ligand (O_A has no bonded H → direction 1 inactive)

    struct TestGeometry {
        const char* name;
        Real ax, ay, az;    // ligand acceptor position
        Real rx, ry, rz;    // receptor donor position
        Real rhx, rhy, rhz; // receptor bonded H position
    };

    TestGeometry geos[] = {
        // ~175 deg: nearly linear, slightly off-axis to avoid degenerate zero gradients
        {"175deg", Real(0.1), Real(0.05), Real(3.5),  0, 0, 0,  0, 0, 1},
        // ~140 deg: A off-axis from H
        {"140deg", Real(1.5), 0, Real(2.5),  0, 0, 0,  0, 0, 1},
        // ~120 deg: further off-axis
        {"120deg", Real(2.0), 0, Real(2.0),  0, 0, 0,  0, 0, 1},
    };

    for (auto& geo : geos) {
        DYNAMIC_SECTION("geometry: " << geo.name) {
            // Build receptor grid with N_D donor
            std::vector<Real> rec_coords;
            FixMol fix = make_receptor_grid(rec_coords, geo.rx, geo.ry, geo.rz, VN_TYPE_N_D);

            FixParamVina fixparam;
            std::vector<int> fix_atom_types = {VN_TYPE_N_D};
            fixparam.atom_types = fix_atom_types.data();

            // Set up RecDonorHTopo: receptor atom 0 (N_D) has 1 bonded H
            RecDonorHTopo saved_rh = g_cpu_constants.rec_donor_h_topo;
            RecDonorHTopo rh;
            rh.natom = 1;
            rh.total_h = 1;
            rh.h_start = new int[2];
            rh.h_start[0] = 0;
            rh.h_start[1] = 1;
            rh.h_coords = new Real[3];
            rh.h_coords[0] = geo.rhx;
            rh.h_coords[1] = geo.rhy;
            rh.h_coords[2] = geo.rhz;
            g_cpu_constants.rec_donor_h_topo = rh;

            // Build 2-atom ligand: [O_A acceptor, C_H filler far away]
            const int natom = 2;
            std::vector<Real> coords = {
                geo.ax, geo.ay, geo.az,
                Real(20), Real(20), Real(20)
            };
            std::vector<int> vn_types = {VN_TYPE_O_A, VN_TYPE_C_H};

            FlexPose pose;
            pose.coords = coords.data();
            pose.dihedrals = nullptr;
            pose.center[0] = geo.ax; pose.center[1] = geo.ay; pose.center[2] = geo.az;
            pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
            pose.energy = 0;

            FlexTopo topo;
            topo.natom = natom;
            topo.ntorsion = 0;
            topo.vn_types = vn_types.data();
            topo.axis_atoms = nullptr;
            topo.range_inds = nullptr;
            topo.range_list = nullptr;
            topo.rotated_inds = nullptr;
            topo.rotated_atoms = nullptr;

            MinimalFlexParam mfp;
            mfp.build(vn_types, coords.data());

            // Compute analytical gradient
            std::vector<Real> aux_f(natom * 3, 0);
            Real e0 = cal_e_f_cpu(&pose, topo, fix, mfp.param, fixparam, aux_f.data());
            REQUIRE(std::isfinite(e0));

            // FD check on atom 0 (O_A acceptor) only — atom 1 is far away
            const Real delta = Real(5e-4);
            int n_checked = 0, n_sign_ok = 0, n_rel_ok = 0;

            for (int d = 0; d < 3; d++) {
                int idx = d;  // atom 0, dimension d
                Real orig = coords[idx];

                coords[idx] = orig + delta;
                Real e_plus = compute_energy(pose, topo, fix, mfp.param, fixparam, natom);

                coords[idx] = orig - delta;
                Real e_minus = compute_energy(pose, topo, fix, mfp.param, fixparam, natom);

                coords[idx] = orig;

                Real dE_numeric = (e_plus - e_minus) / (2 * delta);
                Real dE_analytic = aux_f[idx];

                n_checked++;

                if (std::abs(dE_analytic) < Real(0.001) && std::abs(dE_numeric) < Real(0.001)) {
                    n_sign_ok++;
                    n_rel_ok++;
                    continue;
                }

                if (dE_analytic * dE_numeric >= 0) n_sign_ok++;

                Real abs_diff = std::abs(dE_analytic - dE_numeric);
                Real scale = std::max(Real(0.01),
                             std::max(std::abs(dE_analytic), std::abs(dE_numeric)));
                if (abs_diff / scale < Real(0.05) || abs_diff < Real(0.01)) {
                    n_rel_ok++;
                }
            }

            REQUIRE(n_sign_ok == n_checked);
            REQUIRE(n_rel_ok == n_checked);

            // Cleanup
            free_spatial_grid(g_cpu_constants.receptor_grid);
            g_cpu_constants.rec_donor_h_topo = saved_rh;
            delete[] rh.h_start;
            delete[] rh.h_coords;
        }
    }
}


// ======================== TEST 6: rotatable donors score the best cone hydrogen ========================

TEST_CASE("angular_hbond: rotatable donor classification", "[angular_hbond]") {
    REQUIRE(hbond_donor_rotatable(VN_TYPE_O_DA, 1));
    REQUIRE(hbond_donor_rotatable(VN_TYPE_O_D, 1));
    REQUIRE_FALSE(hbond_donor_rotatable(VN_TYPE_N_D, 1));    // amide / aromatic N-H: fixed
    REQUIRE(hbond_donor_rotatable(VN_TYPE_N_D, 2));          // -NH2
    REQUIRE(hbond_donor_rotatable(VN_TYPE_N_DA, 3));         // -NH3+
    REQUIRE_FALSE(hbond_donor_rotatable(VN_TYPE_Met_D, 0));
}

TEST_CASE("hbond_cone_h: hydrogen on the X-D cone toward the acceptor", "[angular_hbond]") {
    const Real D[3] = {0, 0, 0};
    const Real X[3] = {Real(-1.43), 0, 0};
    Real H[3];

    SECTION("acceptor at the tetrahedral angle from X: H points straight at it") {
        const Real A[3] = {Real(2.9 * 0.33380686), Real(2.9 * 0.94264149), 0};
        hbond_cone_h(D, X, A, H);
        REQUIRE(std::sqrt(H[0]*H[0] + H[1]*H[1] + H[2]*H[2]) == Approx(1.0).margin(1e-5));
        // X-D-H = 109.47 deg
        Real cos_xdh = (X[0]*H[0] + X[1]*H[1] + X[2]*H[2]) / Real(1.43);
        REQUIRE(cos_xdh == Approx(-0.33380686).margin(1e-5));
        // D-H...A = 180 deg
        Real hd[3] = {D[0]-H[0], D[1]-H[1], D[2]-H[2]}, ha[3] = {A[0]-H[0], A[1]-H[1], A[2]-H[2]};
        Real c = (hd[0]*ha[0] + hd[1]*ha[1] + hd[2]*ha[2])
                 / (std::sqrt(hd[0]*hd[0]+hd[1]*hd[1]+hd[2]*hd[2]) * std::sqrt(ha[0]*ha[0]+ha[1]*ha[1]+ha[2]*ha[2]));
        REQUIRE(c == Approx(-1.0).margin(1e-5));
        Real dw = 0;
        REQUIRE(angular_hbond_weight(c, &dw) == Approx(1.0));
    }

    SECTION("acceptor on the X-D axis: no cone hydrogen can reach it (w = 0)") {
        const Real A[3] = {Real(2.9), 0, 0};
        hbond_cone_h(D, X, A, H);
        REQUIRE(std::sqrt(H[0]*H[0] + H[1]*H[1] + H[2]*H[2]) == Approx(1.0).margin(1e-5));
        Real hd[3] = {D[0]-H[0], D[1]-H[1], D[2]-H[2]}, ha[3] = {A[0]-H[0], A[1]-H[1], A[2]-H[2]};
        Real c = (hd[0]*ha[0] + hd[1]*ha[1] + hd[2]*ha[2])
                 / (std::sqrt(hd[0]*hd[0]+hd[1]*hd[1]+hd[2]*hd[2]) * std::sqrt(ha[0]*ha[0]+ha[1]*ha[1]+ha[2]*ha[2]));
        REQUIRE(c > Real(-0.05));   // ~89 deg
        Real dw = 0;
        REQUIRE(angular_hbond_weight(c, &dw) == Approx(0.0));
    }
}

TEST_CASE("cal_e_f_cpu: hydroxyl donor ignores the prepared H and uses the cone", "[angular_hbond]") {
    init_test_atp();
    set_large_box();
    g_cpu_constants.PENALTY_SLOPE = 10;
    RecDonorHTopo saved_rh = g_cpu_constants.rec_donor_h_topo;
    g_cpu_constants.rec_donor_h_topo = RecDonorHTopo{};

    // Ligand: hydroxyl O_DA at origin, its H prepared pointing AWAY from the acceptor,
    // carbon antecedent on -x. Receptor: one O_A acceptor.
    auto run = [&](Real ax, Real ay, Real az, bool angular) {
        std::vector<Real> rec_coords;
        FixMol fix = make_receptor_grid(rec_coords, ax, ay, az, VN_TYPE_O_A);
        FixParamVina fixparam;
        std::vector<int> fix_atom_types = {VN_TYPE_O_A};
        fixparam.atom_types = fix_atom_types.data();

        std::vector<Real> coords = {
            0, 0, 0,                                   // O_DA
            Real(0.3338), Real(-0.9426), 0,            // H, on the cone but on the far side
            Real(-1.43), 0, 0                          // C antecedent
        };
        std::vector<int> vn_types = {VN_TYPE_O_DA, VN_TYPE_H, VN_TYPE_C_H};
        FlexPose pose;
        pose.coords = coords.data(); pose.dihedrals = nullptr;
        pose.center[0] = pose.center[1] = pose.center[2] = 0;
        pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
        pose.energy = 0;
        FlexTopo topo;
        topo.natom = 3; topo.ntorsion = 0; topo.vn_types = vn_types.data();
        topo.axis_atoms = nullptr; topo.range_inds = nullptr; topo.range_list = nullptr;
        topo.rotated_inds = nullptr; topo.rotated_atoms = nullptr;
        MinimalFlexParam mfp;
        mfp.build(vn_types, coords.data());
        REQUIRE(mfp.param.bonded_h_topo.antecedent[0] == 2);

        g_cpu_constants.USE_ANGULAR_HBOND = angular;
        Real e = compute_energy(pose, topo, fix, mfp.param, fixparam, 3);
        free_spatial_grid(g_cpu_constants.receptor_grid);
        return e;
    };

    // d = 2.9 - (1.7 + 1.7) = -0.5 -> vina_hbond = 0.5 / 0.7
    const Real e_hb = g_cpu_constants.weights.hbond * Real(0.5 / 0.7);

    SECTION("acceptor at the tetrahedral direction: full credit although the prepared H points away") {
        Real ax = Real(2.9 * 0.33380686), ay = Real(2.9 * 0.94264149);
        Real e_off = run(ax, ay, 0, false);
        Real e_on = run(ax, ay, 0, true);
        REQUIRE(e_on == Approx(e_off).margin(1e-4));
    }

    SECTION("acceptor on the C-O axis: the hydroxyl cannot reach it, hbond term removed") {
        Real e_off = run(Real(2.9), 0, 0, false);
        Real e_on = run(Real(2.9), 0, 0, true);
        // Without the hbond term the O...O pair turns repulsive (> 0) and is curled with
        // V = 10 (e -> V e / (V + e)); the C...O pair is the same in both runs.
        Vina scorer;
        scorer.set_weights(g_cpu_constants.weights);
        Real f = 0;
        Real p_oo = scorer.eval_ef(Real(-0.5), VN_TYPE_O_DA, VN_TYPE_O_A, &f, g_cpu_constants.atp);
        Real p_co = scorer.eval_ef(Real(4.33 - 3.6), VN_TYPE_C_H, VN_TYPE_O_A, &f, g_cpu_constants.atp);
        REQUIRE(e_off == Approx(p_oo + p_co).margin(1e-4));
        Real e_oo_on = p_oo - e_hb;
        REQUIRE(e_oo_on > 0);
        e_oo_on *= Real(10) / (Real(10) + e_oo_on);
        REQUIRE(e_on == Approx(e_oo_on + p_co).margin(1e-4));
    }

    g_cpu_constants.USE_ANGULAR_HBOND = false;
    g_cpu_constants.rec_donor_h_topo = saved_rh;
}


// ======================== TEST 7: closed-form cone cosine and its gradients ========================

TEST_CASE("hbond_cone_cos: matches the explicit cone hydrogen and finite differences", "[angular_hbond]") {
    // Deterministic pseudo-random geometries: donor at origin-ish, antecedent ~1.4 A away,
    // acceptor 2.6-3.4 A away in a random direction.
    unsigned s = 12345u;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return Real((s >> 8) & 0xFFFF) / Real(65535) - Real(0.5); };
    int n_ramp = 0;
    for (int trial = 0; trial < 200; trial++) {
        Real D[3] = {rnd(), rnd(), rnd()};
        Real ux[3] = {rnd(), rnd(), rnd()};
        Real nu = std::sqrt(ux[0]*ux[0] + ux[1]*ux[1] + ux[2]*ux[2]);
        if (nu < Real(0.05)) continue;
        Real X[3] = {D[0] + Real(1.43) * ux[0] / nu, D[1] + Real(1.43) * ux[1] / nu, D[2] + Real(1.43) * ux[2] / nu};
        Real ua[3] = {rnd(), rnd(), rnd()};
        Real na = std::sqrt(ua[0]*ua[0] + ua[1]*ua[1] + ua[2]*ua[2]);
        if (na < Real(0.05)) continue;
        Real r = Real(2.6) + Real(0.8) * (rnd() + Real(0.5));
        Real A[3] = {D[0] + r * ua[0] / na, D[1] + r * ua[1] / na, D[2] + r * ua[2] / na};

        // (a) same cosine as the explicit construction
        Real H[3];
        hbond_cone_h(D, X, A, H);
        Real hd[3] = {D[0]-H[0], D[1]-H[1], D[2]-H[2]}, ha[3] = {A[0]-H[0], A[1]-H[1], A[2]-H[2]};
        Real c_explicit = (hd[0]*ha[0] + hd[1]*ha[1] + hd[2]*ha[2])
                          / (std::sqrt(hd[0]*hd[0]+hd[1]*hd[1]+hd[2]*hd[2]) * std::sqrt(ha[0]*ha[0]+ha[1]*ha[1]+ha[2]*ha[2]));
        Real gD[3], gX[3], gA[3];
        Real c_closed = hbond_cone_cos(D, X, A, gD, gX, gA);
        INFO("trial " << trial << " cos explicit " << c_explicit << " closed " << c_closed);
        REQUIRE(c_closed == Approx(c_explicit).margin(2e-5));

        // (b) translation invariance
        REQUIRE(gD[0] + gX[0] + gA[0] == Approx(0).margin(1e-5));
        REQUIRE(gD[1] + gX[1] + gA[1] == Approx(0).margin(1e-5));
        REQUIRE(gD[2] + gX[2] + gA[2] == Approx(0).margin(1e-5));

        // (c) central finite differences on every coordinate (skip the on-axis cusp)
        Real u[3] = {X[0]-D[0], X[1]-D[1], X[2]-D[2]}, a[3] = {A[0]-D[0], A[1]-D[1], A[2]-D[2]};
        Real cos_phi = (u[0]*a[0] + u[1]*a[1] + u[2]*a[2]) / (Real(1.43) * r);
        if (std::abs(cos_phi) > Real(0.98)) continue;
        if (c_closed < Real(-0.76) || c_closed > Real(0.0)) n_ramp += 0; else n_ramp++;
        const Real h = Real(1e-3);
        Real* P[3] = {D, X, A};
        Real* G[3] = {gD, gX, gA};
        for (int p = 0; p < 3; p++) {
            for (int k = 0; k < 3; k++) {
                Real orig = P[p][k];
                P[p][k] = orig + h;
                Real cp = hbond_cone_cos(D, X, A, nullptr, nullptr, nullptr);
                P[p][k] = orig - h;
                Real cm = hbond_cone_cos(D, X, A, nullptr, nullptr, nullptr);
                P[p][k] = orig;
                Real fd = (cp - cm) / (2 * h);
                INFO("trial " << trial << " point " << p << " coord " << k << " analytic " << G[p][k] << " fd " << fd);
                REQUIRE(G[p][k] == Approx(fd).margin(2e-3));
            }
        }
    }
    REQUIRE(n_ramp > 20);  // enough geometries inside the 90-140 deg ramp were checked
}


// ======================== TEST 8: FD gradient through cal_e_f_cpu for cone donors ========================

TEST_CASE("cal_e_f_cpu: FD gradient for cone hydrogens (both directions)", "[angular_hbond]") {
    init_test_atp();
    set_large_box();
    g_cpu_constants.PENALTY_SLOPE = 10;
    g_cpu_constants.USE_ANGULAR_HBOND = true;
    RecDonorHTopo saved_rh = g_cpu_constants.rec_donor_h_topo;

    auto fd_check = [&](FlexPose& pose, const FlexTopo& topo, const FixMol& fix, const FlexParamVina& fparam,
                        const FixParamVina& fixparam, std::vector<Real>& coords, const std::vector<int>& atoms) {
        std::vector<Real> aux_f(topo.natom * 3, 0);
        Real e0 = cal_e_f_cpu(&pose, topo, fix, fparam, fixparam, aux_f.data());
        REQUIRE(std::isfinite(e0));
        const Real delta = Real(5e-4);
        Real max_abs = 0;
        for (int i : atoms) {
            for (int d = 0; d < 3; d++) {
                int idx = i * 3 + d;
                Real orig = coords[idx];
                coords[idx] = orig + delta;
                Real ep = compute_energy(pose, topo, fix, fparam, fixparam, topo.natom);
                coords[idx] = orig - delta;
                Real em = compute_energy(pose, topo, fix, fparam, fixparam, topo.natom);
                coords[idx] = orig;
                Real fd = (ep - em) / (2 * delta);
                INFO("atom " << i << " coord " << d << " analytic " << aux_f[idx] << " fd " << fd);
                REQUIRE(aux_f[idx] == Approx(fd).margin(0.02));
                max_abs = std::max(max_abs, std::abs(aux_f[idx]));
            }
        }
        return max_abs;
    };

    SECTION("ligand hydroxyl donor (cone) toward a fixed receptor acceptor, mid-ramp geometry") {
        g_cpu_constants.rec_donor_h_topo = RecDonorHTopo{};
        // O at origin, C on -x; acceptor placed so that the cone hydrogen gives ~120 deg
        std::vector<Real> rec_coords;
        FixMol fix = make_receptor_grid(rec_coords, Real(2.4), Real(1.6), Real(0.4), VN_TYPE_O_A);
        FixParamVina fixparam;
        std::vector<int> fix_atom_types = {VN_TYPE_O_A};
        fixparam.atom_types = fix_atom_types.data();

        std::vector<Real> coords = {0, 0, 0,  Real(0.3338), Real(-0.9426), 0,  Real(-1.43), 0, 0};
        std::vector<int> vn_types = {VN_TYPE_O_DA, VN_TYPE_H, VN_TYPE_C_H};
        FlexPose pose;
        pose.coords = coords.data(); pose.dihedrals = nullptr;
        pose.center[0] = pose.center[1] = pose.center[2] = 0;
        pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0; pose.energy = 0;
        FlexTopo topo;
        topo.natom = 3; topo.ntorsion = 0; topo.vn_types = vn_types.data();
        topo.axis_atoms = nullptr; topo.range_inds = nullptr; topo.range_list = nullptr;
        topo.rotated_inds = nullptr; topo.rotated_atoms = nullptr;
        MinimalFlexParam mfp;
        mfp.build(vn_types, coords.data());

        // the geometry must sit inside the ramp, otherwise the angular gradient is trivially zero
        Real gD[3], gX[3], gA[3];
        Real c = hbond_cone_cos(coords.data(), coords.data() + 6, rec_coords.data(), gD, gX, gA);
        INFO("cone cos " << c);
        REQUIRE(c > Real(-0.76)); REQUIRE(c < Real(-0.05));

        Real max_abs = fd_check(pose, topo, fix, mfp.param, fixparam, coords, {0, 2});
        REQUIRE(max_abs > Real(0.05));  // the antecedent really receives a force
        free_spatial_grid(g_cpu_constants.receptor_grid);
    }

    SECTION("receptor hydroxyl donor (cone) toward a ligand acceptor, mid-ramp geometry") {
        // receptor: Ser-like O_DA at origin with C antecedent on -x and one H (ignored: rotatable)
        std::vector<Real> rec_coords = {0, 0, 0,  Real(-1.43), 0, 0};
        std::vector<int> rec_types = {VN_TYPE_O_DA, VN_TYPE_C_H};
        free_spatial_grid(g_cpu_constants.receptor_grid);
        build_spatial_grid(g_cpu_constants.receptor_grid, rec_coords.data(), rec_types.data(), 2,
                           g_cpu_constants.atp, g_cpu_constants.vdw_radii);
        FixMol fix; fix.natom = 2; fix.coords = rec_coords.data();
        FixParamVina fixparam; fixparam.atom_types = rec_types.data();
        RecDonorHTopo rh{};
        rh.natom = 2; rh.total_h = 1;
        rh.h_start = new int[3]{0, 1, 1};
        rh.h_coords = new Real[3]{Real(0.3338), Real(-0.9426), 0};
        rh.antecedent = new int[2]{1, -1};
        g_cpu_constants.rec_donor_h_topo = rh;

        std::vector<Real> coords = {Real(2.4), Real(1.6), Real(0.4)};
        std::vector<int> vn_types = {VN_TYPE_O_A};
        FlexPose pose;
        pose.coords = coords.data(); pose.dihedrals = nullptr;
        pose.center[0] = coords[0]; pose.center[1] = coords[1]; pose.center[2] = coords[2];
        pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0; pose.energy = 0;
        FlexTopo topo;
        topo.natom = 1; topo.ntorsion = 0; topo.vn_types = vn_types.data();
        topo.axis_atoms = nullptr; topo.range_inds = nullptr; topo.range_list = nullptr;
        topo.rotated_inds = nullptr; topo.rotated_atoms = nullptr;
        MinimalFlexParam mfp;
        mfp.build(vn_types, coords.data());

        Real gD[3], gX[3], gA[3];
        Real c = hbond_cone_cos(rec_coords.data(), rec_coords.data() + 3, coords.data(), gD, gX, gA);
        INFO("cone cos " << c);
        REQUIRE(c > Real(-0.76)); REQUIRE(c < Real(-0.05));

        fd_check(pose, topo, fix, mfp.param, fixparam, coords, {0});
        free_spatial_grid(g_cpu_constants.receptor_grid);
        delete[] rh.h_start; delete[] rh.h_coords; delete[] rh.antecedent;
    }

    g_cpu_constants.USE_ANGULAR_HBOND = false;
    g_cpu_constants.rec_donor_h_topo = saved_rh;
}


// ======================== TEST 9: planar -NH2 keeps its hydrogens, pyramidal -NH2 is a rotor ========================

TEST_CASE("hbond_donor_rotatable_geom: planar vs pyramidal NH2", "[angular_hbond]") {
    const Real N[3] = {0, 0, 0};
    const Real C[3] = {Real(-1.33), 0, 0};
    // planar (amide / guanidinium): H at +-120 deg from the C-N bond, in plane
    const Real Hp1[3] = {Real(1.01 * 0.5), Real(1.01 * 0.866), 0};
    const Real Hp2[3] = {Real(1.01 * 0.5), Real(-1.01 * 0.866), 0};
    REQUIRE_FALSE(hbond_donor_rotatable_geom(VN_TYPE_N_D, 2, N, C, Hp1, Hp2));
    // pyramidal (amine): tetrahedral arrangement
    const Real Ht1[3] = {Real(1.01 * 0.3338), Real(1.01 * 0.9426), 0};
    const Real Ht2[3] = {Real(1.01 * 0.3338), Real(-1.01 * 0.4713), Real(1.01 * 0.8164)};
    REQUIRE(hbond_donor_rotatable_geom(VN_TYPE_N_D, 2, N, C, Ht1, Ht2));
    // ammonium and hydroxyl are rotors regardless of geometry; single N-H is fixed
    REQUIRE(hbond_donor_rotatable_geom(VN_TYPE_N_D, 3, N, C, Hp1, Hp2));
    REQUIRE(hbond_donor_rotatable_geom(VN_TYPE_O_DA, 1, N, C, Hp1, nullptr));
    REQUIRE_FALSE(hbond_donor_rotatable_geom(VN_TYPE_N_D, 1, N, C, Hp1, nullptr));
    // NH2 without a heavy neighbour (ammonia-like): rotor, scored with full credit downstream
    REQUIRE(hbond_donor_rotatable_geom(VN_TYPE_N_D, 2, N, nullptr, Hp1, Hp2));
}

TEST_CASE("build_bonded_h_topo: donor_rotatable table for an amide and an amine", "[angular_hbond]") {
    init_test_atp();
    // atoms: 0 C, 1 N (amide, planar H), 2 H, 3 H, 4 C, 5 N (amine, pyramidal H), 6 H, 7 H
    std::vector<Real> coords = {
        Real(-1.33), 0, 0,
        0, 0, 0,
        Real(1.01 * 0.5), Real(1.01 * 0.866), 0,
        Real(1.01 * 0.5), Real(-1.01 * 0.866), 0,
        Real(10 - 1.47), 0, 0,
        10, 0, 0,
        Real(10 + 1.01 * 0.3338), Real(1.01 * 0.9426), 0,
        Real(10 + 1.01 * 0.3338), Real(-1.01 * 0.4713), Real(1.01 * 0.8164),
    };
    std::vector<int> types = {VN_TYPE_C_P, VN_TYPE_N_D, VN_TYPE_H, VN_TYPE_H,
                              VN_TYPE_C_P, VN_TYPE_N_D, VN_TYPE_H, VN_TYPE_H};
    BondedHTopo topo{};
    build_bonded_h_topo(topo, coords.data(), types.data(), 8);
    REQUIRE(topo.h_start[2] - topo.h_start[1] == 2);
    REQUIRE(topo.h_start[6] - topo.h_start[5] == 2);
    REQUIRE(topo.antecedent[1] == 0);
    REQUIRE(topo.antecedent[5] == 4);
    REQUIRE(topo.donor_rotatable[1] == 0);
    REQUIRE(topo.donor_rotatable[5] == 1);
    REQUIRE(topo.donor_rotatable[0] == 0);
    free_bonded_h_topo(topo);
    REQUIRE(topo.donor_rotatable == nullptr);
}
