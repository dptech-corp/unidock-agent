//
// Unit tests for hbond_angle_correction_cpu
//
// Verifies that the post-BFGS hbond angle filter correctly:
//   - Returns zero correction for good D-H...A geometry (angle > 100°)
//   - Returns positive correction for bad geometry (angle < 100°)
//   - Handles edge cases: no bonded H, DA+DA pairs, non-hbond pairs
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

// ========== Helper: set up g_cpu_constants with default atom type props ==========
static void init_test_atp() {
    init_default_atom_type_props(g_cpu_constants.atp);
    int n = sizeof(VN_VDW_RADII) / sizeof(VN_VDW_RADII[0]);
    for (int i = 0; i < n; i++) g_cpu_constants.vdw_radii[i] = VN_VDW_RADII[i];
    for (int i = n; i < MAX_VN_TYPES; i++) g_cpu_constants.vdw_radii[i] = Real(0);
}

// ========== Helper: build a 1-atom receptor grid ==========
// Places a single receptor atom at (rx, ry, rz) with type rec_type.
// Returns the FixMol (caller owns coords lifetime via the passed vector).
static FixMol make_receptor_grid(std::vector<Real>& rec_coords,
                                  Real rx, Real ry, Real rz, int rec_type) {
    rec_coords = {rx, ry, rz};

    std::vector<int> rec_types_vec = {rec_type};

    // Build the spatial grid into g_cpu_constants
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

// ========== Helper: build a ligand pose ==========
// Creates a 3-atom ligand: [donor, hydrogen, carbon_filler]
// donor at (dx, dy, dz), H at (hx, hy, hz), filler at origin.
struct TestPose {
    std::vector<Real> coords;
    std::vector<int> vn_types;
    FlexPose pose;
    FlexTopo topo;

    void build(int donor_type,
               Real dx, Real dy, Real dz,
               Real hx, Real hy, Real hz) {
        // atom 0: donor
        // atom 1: H bonded to donor
        // atom 2: C_H filler (needed so topo.natom >= 1 heavy non-donor)
        coords = {dx, dy, dz,   hx, hy, hz,   0, 0, 0};
        vn_types = {donor_type, VN_TYPE_H, VN_TYPE_C_H};

        pose.coords = coords.data();
        pose.dihedrals = nullptr;
        pose.center[0] = dx; pose.center[1] = dy; pose.center[2] = dz;
        pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
        pose.energy = 0;

        topo.natom = 3;
        topo.ntorsion = 0;
        topo.vn_types = vn_types.data();
        topo.axis_atoms = nullptr;
        topo.range_inds = nullptr;
        topo.range_list = nullptr;
        topo.rotated_inds = nullptr;
        topo.rotated_atoms = nullptr;
    }
};


// ========================== TESTS ==========================

TEST_CASE("hbond_angle_correction: good angle returns zero", "[hbond_angle][cpu]") {
    init_test_atp();

    // Geometry: N_D donor at origin, H at (0, 0, 1), receptor O_A at (0, 0, 4)
    // D-H...A angle: H is between D and A on the z-axis → angle = 180° (linear)
    // This is an ideal hbond geometry.
    std::vector<Real> rec_coords;
    FixMol fix = make_receptor_grid(rec_coords, 0, 0, 4, VN_TYPE_O_A);

    TestPose tp;
    tp.build(VN_TYPE_N_D,  0, 0, 0,  0, 0, 1);

    Real corr = hbond_angle_correction_cpu(&tp.pose, tp.topo, fix);

    // Good angle → no correction
    REQUIRE(corr == Approx(0.0).margin(1e-6));

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


TEST_CASE("hbond_angle_correction: bad angle returns positive correction", "[hbond_angle][cpu]") {
    init_test_atp();

    // Geometry: N_D donor at origin, H at (0, 0, 1)
    // Receptor O_A at (0, 0, -2):  A is BEHIND the donor relative to H
    // D-H...A angle: vectors H→D = (0,0,-1), H→A = (0,0,-3)
    // cos(angle) = dot / (|H→D| |H→A|) = 3/3 = 1.0 → angle = 0° (terrible)
    //
    // The surface distance: r = 2.0, d = 2.0 - 1.8(N_D) - 1.7(O_A) = -1.5
    // hbond_low = -0.7, so d < hbond_low → vina_hbond = 1.0
    // hbond energy = weight_hbond * 1.0 (favorable; weight_hbond = the active search set's hbond weight)
    //
    // Correction should remove this → correction ≈ -weight_hbond
    std::vector<Real> rec_coords;
    FixMol fix = make_receptor_grid(rec_coords, 0, 0, -2, VN_TYPE_O_A);

    TestPose tp;
    tp.build(VN_TYPE_N_D,  0, 0, 0,  0, 0, 1);

    Real corr = hbond_angle_correction_cpu(&tp.pose, tp.topo, fix);

    // Bad angle → positive correction (removes favorable hbond energy)
    REQUIRE(corr > 0.5);
    REQUIRE(corr == Approx(-g_cpu_constants.weights.hbond).margin(0.01));

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


TEST_CASE("hbond_angle_correction: no bonded H gives zero correction", "[hbond_angle][cpu]") {
    init_test_atp();

    // Donor at origin but no H atom nearby → can't check angle → no correction
    std::vector<Real> rec_coords;
    FixMol fix = make_receptor_grid(rec_coords, 0, 0, 3, VN_TYPE_O_A);

    // Build ligand with donor but H is far away (10 Å, well beyond 1.3 Å bonded cutoff)
    TestPose tp;
    tp.build(VN_TYPE_N_D,  0, 0, 0,  10, 10, 10);

    Real corr = hbond_angle_correction_cpu(&tp.pose, tp.topo, fix);
    REQUIRE(corr == Approx(0.0).margin(1e-6));

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


TEST_CASE("hbond_angle_correction: non-hbond pair gives zero", "[hbond_angle][cpu]") {
    init_test_atp();

    // Receptor is C_H (not acceptor, not donor) → no hbond possible
    std::vector<Real> rec_coords;
    FixMol fix = make_receptor_grid(rec_coords, 0, 0, 3, VN_TYPE_C_H);

    TestPose tp;
    tp.build(VN_TYPE_N_D,  0, 0, 0,  0, 0, 1);

    Real corr = hbond_angle_correction_cpu(&tp.pose, tp.topo, fix);
    REQUIRE(corr == Approx(0.0).margin(1e-6));

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


TEST_CASE("hbond_angle_correction: DA+DA pair is skipped (conservative)", "[hbond_angle][cpu]") {
    init_test_atp();

    // Both ligand and receptor are N_DA (donor+acceptor).
    // The reverse direction (receptor-donor → ligand-acceptor) might be valid
    // but we can't verify it (no receptor H). So correction should be zero.
    //
    // Place receptor at (0,0,-2) to create bad angle (same geometry as bad-angle test)
    std::vector<Real> rec_coords;
    FixMol fix = make_receptor_grid(rec_coords, 0, 0, -2, VN_TYPE_N_DA);

    TestPose tp;
    tp.build(VN_TYPE_N_DA,  0, 0, 0,  0, 0, 1);

    Real corr = hbond_angle_correction_cpu(&tp.pose, tp.topo, fix);

    // DA+DA → skipped (conservative) → no correction
    REQUIRE(corr == Approx(0.0).margin(1e-6));

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


TEST_CASE("hbond_angle_correction: donor-only lig + DA receptor IS checked", "[hbond_angle][cpu]") {
    init_test_atp();

    // Ligand N_D (donor only) + receptor N_DA (donor+acceptor)
    // Since ligand is NOT acceptor, the reverse direction is impossible.
    // The pair is only valid as lig-donor → rec-acceptor.
    // Bad angle should produce correction.
    std::vector<Real> rec_coords;
    FixMol fix = make_receptor_grid(rec_coords, 0, 0, -2, VN_TYPE_N_DA);

    TestPose tp;
    tp.build(VN_TYPE_N_D,  0, 0, 0,  0, 0, 1);

    Real corr = hbond_angle_correction_cpu(&tp.pose, tp.topo, fix);

    // N_D is donor-only, so the pair IS checked despite rec being DA
    REQUIRE(corr > 0.5);

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


TEST_CASE("hbond_angle_correction: acceptor-only ligand gives zero", "[hbond_angle][cpu]") {
    init_test_atp();

    // Ligand O_A (acceptor only, not donor) + receptor N_D (donor)
    // is_donor[O_A] = false → the function skips this atom entirely.
    std::vector<Real> rec_coords;
    FixMol fix = make_receptor_grid(rec_coords, 0, 0, 3, VN_TYPE_N_D);

    TestPose tp;
    tp.build(VN_TYPE_O_A,  0, 0, 0,  0, 0, 1);

    Real corr = hbond_angle_correction_cpu(&tp.pose, tp.topo, fix);
    REQUIRE(corr == Approx(0.0).margin(1e-6));

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


TEST_CASE("hbond_angle_correction: borderline angle ~100 deg", "[hbond_angle][cpu]") {
    init_test_atp();

    // Place receptor so that D-H...A angle is exactly ~100°.
    // D at origin, H at (0, 0, 1).  H→D = (0,0,-1).
    // cos(100°) ≈ -0.17365, so H→A_hat_z = 0.17365
    // H→A_hat = (sin(acos(0.17365)), 0, 0.17365) ≈ (0.98481, 0, 0.17365)
    // A = H + 2.5 * H→A_hat = (2.4620, 0, 1.4341)

    Real a_x = Real(2.4620);
    Real a_y = Real(0);
    Real a_z = Real(1.4341);

    std::vector<Real> rec_coords;
    FixMol fix = make_receptor_grid(rec_coords, a_x, a_y, a_z, VN_TYPE_O_A);

    TestPose tp;
    tp.build(VN_TYPE_N_D,  0, 0, 0,  0, 0, 1);

    // Check that the distance is within cutoff (r < 8)
    Real r = std::sqrt(a_x*a_x + a_y*a_y + a_z*a_z);
    REQUIRE(r < 8.0);

    Real corr = hbond_angle_correction_cpu(&tp.pose, tp.topo, fix);

    // At exactly 100°, cos = cos(100°) ≈ -0.17365, threshold is cos < -0.17365
    // This is right at the boundary. Due to float precision, it could go either way.
    // Just verify it's near zero (either zero or a small correction)
    REQUIRE(corr >= 0.0);
    // The important thing: angle slightly > 100° gives 0, slightly < 100° gives > 0
    // We test both sides below.

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


TEST_CASE("hbond_angle_correction: 120 deg (good) vs 60 deg (bad)", "[hbond_angle][cpu]") {
    init_test_atp();

    // D at origin, H at (0, 0, 1).  H→D = (0,0,-1).
    //
    // For angle θ at H: cos(θ) = dot(H→D, H→A) / (|H→D||H→A|)
    //   = -H→A_hat_z
    //
    // 120° case: cos(120°) = -0.5 → H→A_hat_z = 0.5
    //   H→A_hat = (0.86603, 0, 0.5)
    //   A = H + 2*(0.86603, 0, 0.5) = (1.73205, 0, 2.0)
    //   angle > 100° → good
    //
    // 60° case: cos(60°) = 0.5 → H→A_hat_z = -0.5
    //   H→A_hat = (0.86603, 0, -0.5)
    //   A = H + 2*(0.86603, 0, -0.5) = (1.73205, 0, 0.0)
    //   angle < 100° → bad

    TestPose tp;
    tp.build(VN_TYPE_N_D,  0, 0, 0,  0, 0, 1);

    // --- 120° (good) ---
    {
        std::vector<Real> rec_coords;
        FixMol fix = make_receptor_grid(rec_coords,
            Real(1.73205), Real(0), Real(2.0), VN_TYPE_O_A);

        Real corr = hbond_angle_correction_cpu(&tp.pose, tp.topo, fix);
        REQUIRE(corr == Approx(0.0).margin(1e-6));

        free_spatial_grid(g_cpu_constants.receptor_grid);
    }

    // --- 60° (bad) ---
    {
        std::vector<Real> rec_coords;
        FixMol fix = make_receptor_grid(rec_coords,
            Real(1.73205), Real(0), Real(0.0), VN_TYPE_O_A);

        Real corr = hbond_angle_correction_cpu(&tp.pose, tp.topo, fix);
        REQUIRE(corr > 0.0);  // bad angle → positive correction

        free_spatial_grid(g_cpu_constants.receptor_grid);
    }
}


TEST_CASE("hbond_angle_correction: pair beyond cutoff gives zero", "[hbond_angle][cpu]") {
    init_test_atp();

    // Place receptor acceptor very far away (> 8 Å = sqrt(r2_cutoff))
    std::vector<Real> rec_coords;
    FixMol fix = make_receptor_grid(rec_coords, 0, 0, 20, VN_TYPE_O_A);

    TestPose tp;
    tp.build(VN_TYPE_N_D,  0, 0, 0,  0, 0, 1);

    Real corr = hbond_angle_correction_cpu(&tp.pose, tp.topo, fix);
    REQUIRE(corr == Approx(0.0).margin(1e-6));

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


// ==================== Cross-check tests ====================
// These verify that the correction EXACTLY matches eval_ef_decomp's hbond term.

TEST_CASE("hbond_angle: bad angle correction == eval_ef_decomp hbond", "[hbond_angle][cpu][crosscheck]") {
    init_test_atp();

    // Geometry: N_D at origin, H at (0,0,1), O_A at (0,0,-2)
    // D-H...A angle = 0° (terrible) → correction should remove the hbond energy.
    //
    // Manually compute the same surface distance that hbond_angle_correction_cpu uses:
    //   r = dist(donor, acceptor) = 2.0
    //   d = r - VN_VDW_RADII[N_D] - VN_VDW_RADII[O_A] = 2.0 - 1.8 - 1.7 = -1.5
    Real r = 2.0;
    Real d = r - VN_VDW_RADII[VN_TYPE_N_D] - VN_VDW_RADII[VN_TYPE_O_A];

    // Get the hbond energy that eval_ef_decomp awards for this pair (same weights as the CPU path)
    Vina scorer;
    scorer.set_weights(g_cpu_constants.weights);
    VinaTerms terms = scorer.eval_ef_decomp(d, VN_TYPE_N_D, VN_TYPE_O_A, g_cpu_constants.atp);
    REQUIRE(terms.hbond < 0);  // must be favorable (negative)

    // Now get the correction from hbond_angle_correction_cpu
    std::vector<Real> rec_coords;
    FixMol fix = make_receptor_grid(rec_coords, 0, 0, -2, VN_TYPE_O_A);

    TestPose tp;
    tp.build(VN_TYPE_N_D, 0, 0, 0,  0, 0, 1);

    Real corr = hbond_angle_correction_cpu(&tp.pose, tp.topo, fix);

    // The correction should EXACTLY equal -terms.hbond (removes the hbond energy)
    REQUIRE(corr == Approx(-terms.hbond).margin(1e-6));

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


TEST_CASE("hbond_angle: good angle has zero correction despite non-zero hbond", "[hbond_angle][cpu][crosscheck]") {
    init_test_atp();

    // Geometry: N_D at origin, H at (0,0,1), O_A at (0,0,4)
    // D-H...A angle = 180° (perfect linear) → correction should be zero.
    //
    // But first verify that eval_ef_decomp DOES award non-zero hbond for this pair.
    Real r = 4.0;
    Real d = r - VN_VDW_RADII[VN_TYPE_N_D] - VN_VDW_RADII[VN_TYPE_O_A];
    // d = 4.0 - 1.8 - 1.7 = 0.5 → inside hbond range [-0.7, 0]

    Vina scorer;
    VinaTerms terms = scorer.eval_ef_decomp(d, VN_TYPE_N_D, VN_TYPE_O_A, g_cpu_constants.atp);

    // The hbond IS non-zero — this pair has a valid distance-based hbond
    // (d=0.5 is within [hbond_low=-0.7, hbond_high=0] → partial hbond)
    // Note: d=0.5 > hbond_high=0, so actually hbond=0. Adjust distance.
    // Let's use O_A at (0,0,3) instead: r=3.0, d=3.0-1.8-1.7=-0.5
    // d=-0.5 is in [-0.7, 0] → partial hbond
    Real r2 = 3.0;
    Real d2 = r2 - VN_VDW_RADII[VN_TYPE_N_D] - VN_VDW_RADII[VN_TYPE_O_A];
    VinaTerms terms2 = scorer.eval_ef_decomp(d2, VN_TYPE_N_D, VN_TYPE_O_A, g_cpu_constants.atp);
    REQUIRE(terms2.hbond < 0);  // confirm hbond IS awarded

    // Now run the correction with good angle (D at origin, H at (0,0,1), A at (0,0,3))
    // D-H...A: H→D=(0,0,-1), H→A=(0,0,2) → cos = -2/2 = -1.0 → angle=180° (perfect)
    std::vector<Real> rec_coords;
    FixMol fix = make_receptor_grid(rec_coords, 0, 0, 3, VN_TYPE_O_A);

    TestPose tp;
    tp.build(VN_TYPE_N_D, 0, 0, 0,  0, 0, 1);

    Real corr = hbond_angle_correction_cpu(&tp.pose, tp.topo, fix);

    // Hbond exists (terms2.hbond < 0) but angle is good → NO correction
    REQUIRE(corr == Approx(0.0).margin(1e-6));

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


TEST_CASE("hbond_angle: correction equals hbond for partial hbond distance", "[hbond_angle][cpu][crosscheck]") {
    init_test_atp();

    // Test with a partial hbond (d between hbond_low and hbond_high).
    // Place donor N_D at origin, H at (0,0,1).
    // Place receptor O_A behind donor to create bad angle.
    //
    // Choose distance so hbond is partial (not full):
    //   r1+r2 = 1.8+1.7 = 3.5, hbond_low=-0.7, hbond_high=0
    //   For d=-0.3 (partial): r = 3.5 + (-0.3) = 3.2
    //   Place A at (0,0,-3.2) so it's behind the donor.
    //   D-H...A angle: H at (0,0,1), A at (0,0,-3.2)
    //     H→D=(0,0,-1), H→A=(0,0,-4.2) → cos=4.2/4.2=1.0 → angle=0° (bad)
    Real target_r = Real(3.2);
    Real d = target_r - VN_VDW_RADII[VN_TYPE_N_D] - VN_VDW_RADII[VN_TYPE_O_A];
    // d = 3.2 - 3.5 = -0.3, which is between hbond_low(-0.7) and hbond_high(0)

    Vina scorer;
    scorer.set_weights(g_cpu_constants.weights);
    VinaTerms terms = scorer.eval_ef_decomp(d, VN_TYPE_N_D, VN_TYPE_O_A, g_cpu_constants.atp);
    REQUIRE(terms.hbond < 0);                                   // partial hbond is favorable
    REQUIRE(terms.hbond > g_cpu_constants.weights.hbond);       // but weaker than full hbond

    // Place A at (0, 0, -3.2) → bad angle
    std::vector<Real> rec_coords;
    FixMol fix = make_receptor_grid(rec_coords, 0, 0, Real(-target_r), VN_TYPE_O_A);

    TestPose tp;
    tp.build(VN_TYPE_N_D, 0, 0, 0,  0, 0, 1);

    Real corr = hbond_angle_correction_cpu(&tp.pose, tp.topo, fix);

    // Correction exactly removes the partial hbond
    REQUIRE(corr == Approx(-terms.hbond).margin(1e-6));
    // And it's less than the full hbond removal
    REQUIRE(corr < -g_cpu_constants.weights.hbond);
    REQUIRE(corr > 0.0);

    free_spatial_grid(g_cpu_constants.receptor_grid);
}
