//
// Numerical force-check tests for ff_bond and ff_angle energy terms.
//
// Strategy: build minimal FFMinData with only bond or angle terms active
// (zero LJ/charge atoms, no torsions, no receptor), compute analytic
// gradient via cal_e_f_ff_cpu, then compare against central-difference
// finite-difference gradient on each coordinate component.
//
// Sign convention: aux_f stores +dE/dx_i (energy gradient, not force).
// F_fd[k] = (E(x+h) - E(x-h)) / (2h) for component k.
//

#include <catch2/catch_amalgamated.hpp>
#include "cpu/score_ff_cpu.h"
#include "format/dms_reader.h"
#include <cmath>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Helpers
// ============================================================================

/// Build a minimal FFMinData with natom atoms, all-zero LJ/charge,
/// no receptor, no torsions, no exclusions, no pairs.
/// Caller fills lig_bonds or lig_angles afterward.
static FFMinData make_empty_ff(int natom) {
    FFMinData ff;
    ff.lig_atoms.natom = natom;
    ff.lig_atoms.sigma.assign(natom, 0.0);
    ff.lig_atoms.epsilon.assign(natom, 0.0);
    ff.lig_atoms.charge.assign(natom, 0.0);
    ff.lig_atoms.anum.assign(natom, 6);
    ff.lig_atoms.x.assign(natom, 0.0);
    ff.lig_atoms.y.assign(natom, 0.0);
    ff.lig_atoms.z.assign(natom, 0.0);
    ff.rec_atoms.natom = 0;
    ff.loaded = true;
    // No exclusions -> pair_type stays empty (pair_type is only consulted for
    // the intra nonbonded loop, which contributes zero because sigma/eps = 0).
    // build_pair_lookup with natom <= 1 returns early; for natom = 2 we need
    // to call it so the triangular table has the right size (though all entries
    // default to FF_PAIR_NORMAL — fine because eps_ij = sqrt(0*0) = 0).
    ff.build_pair_lookup();
    return ff;
}

/// Build minimal FlexTopo + FlexPose + FixMol from coordinate vector.
/// Sets a wide box centred at origin so box-penalty never fires.
struct MinimalSetup {
    FlexTopo topo;
    FlexPose pose;
    FixMol   fix_mol;
    std::vector<int>  vn_types;
    std::vector<Real> rec_coords; // empty

    void init(int natom, std::vector<Real>& coords) {
        topo.natom    = natom;
        topo.ntorsion = 0;
        vn_types.assign(natom, 0);
        topo.vn_types      = vn_types.data();
        topo.axis_atoms    = nullptr;
        topo.range_inds    = nullptr;
        topo.range_list    = nullptr;
        topo.rotated_inds  = nullptr;
        topo.rotated_atoms = nullptr;

        pose.coords    = coords.data();
        pose.dihedrals = nullptr;
        pose.center[0] = pose.center[1] = pose.center[2] = 0;
        pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;

        fix_mol.natom  = 0;
        fix_mol.coords = nullptr;

        // Wide box so atoms are always inside; no box penalty.
        DockParam dp;
        dp.box.x_lo = -100; dp.box.x_hi = 100;
        dp.box.y_lo = -100; dp.box.y_hi = 100;
        dp.box.z_lo = -100; dp.box.z_hi = 100;
        AtomTypeProps atp;
        init_constants_cpu(dp, atp);
    }
};

/// Compute energy only (no force output needed) — reuse cal_e_f_ff_cpu
/// with a scratch force buffer.
static Real energy_only(std::vector<Real>& coords, int natom,
                        FlexTopo& topo, FixMol& fix_mol, FFMinData& ff) {
    FlexPose p;
    p.coords    = coords.data();
    p.dihedrals = nullptr;
    p.center[0] = p.center[1] = p.center[2] = 0;
    p.rot_vec[0] = p.rot_vec[1] = p.rot_vec[2] = 0;
    std::vector<Real> scratch(natom * 3, 0.0);
    return cal_e_f_ff_cpu(&p, topo, fix_mol, ff, scratch.data());
}


// ============================================================================
// Bond stretch: analytic gradient vs finite difference
// ============================================================================

TEST_CASE("ff_bond: analytic gradient matches finite difference", "[ff_bond_angle]") {
    // 2-atom system; only bond term contributes (LJ/Coulomb: eps=0, q=0).
    // Bond: i=0, j=1, r0=1.5 A, fc=300 kcal/mol/A^2 (E=fc*(r-r0)^2, fc absorbs the 1/2 factor)
    const int natom = 2;
    FFMinData ff = make_empty_ff(natom);
    FFBond bond;
    bond.i  = 0; bond.j = 1;
    bond.r0 = 1.5f;
    bond.fc = 300.0f;
    ff.lig_bonds.push_back(bond);

    // Perturbed geometry: r = 1.6 A along x-axis, slight y/z to avoid singularity.
    std::vector<Real> coords = {
        0.0f,  0.0f,  0.0f,   // atom 0
        1.6f,  0.05f, -0.02f  // atom 1
    };

    MinimalSetup setup;
    setup.init(natom, coords);

    // Analytic gradient
    std::vector<Real> aux_f(natom * 3, 0.0f);
    Real E0 = cal_e_f_ff_cpu(&setup.pose, setup.topo, setup.fix_mol, ff, aux_f.data());
    REQUIRE(std::isfinite(E0));
    REQUIRE(E0 > 0.0f);  // displaced from equilibrium => positive energy

    // Finite-difference check.
    // Use h=1e-3 A: large enough to avoid float cancellation, small enough
    // for central-difference accuracy (O(h^2) ~ 1e-6 for this smooth potential).
    // Tolerance: relative error < 1e-2 for components with |analytic| > 1e-3;
    // absolute error < 1e-3 kcal/mol/A for near-zero components.
    const Real h        = 1e-3f;
    const Real rel_tol  = 1e-2f;
    const Real abs_tol  = 1e-3f;
    const Real zero_thr = 1e-3f;  // below this, use absolute test

    Real max_relerr = 0.0f;
    for (int k = 0; k < natom * 3; k++) {
        std::vector<Real> coords_plus  = coords;
        std::vector<Real> coords_minus = coords;
        coords_plus[k]  += h;
        coords_minus[k] -= h;

        Real Ep = energy_only(coords_plus,  natom, setup.topo, setup.fix_mol, ff);
        Real Em = energy_only(coords_minus, natom, setup.topo, setup.fix_mol, ff);
        Real fd = (Ep - Em) / (2.0f * h);

        Real analytic = aux_f[k];
        Real abs_err  = std::abs(analytic - fd);

        if (std::abs(analytic) < zero_thr && std::abs(fd) < zero_thr) {
            // Both near-zero: use absolute tolerance
            INFO("Bond coord " << k
                 << ": analytic=" << analytic
                 << " fd=" << fd
                 << " abs_err=" << abs_err << " (near-zero, abs check)");
            REQUIRE(abs_err < abs_tol);
        } else {
            Real denom   = std::max(std::abs(analytic), std::abs(fd));
            Real relerr  = abs_err / denom;
            max_relerr   = std::max(max_relerr, relerr);
            INFO("Bond coord " << k
                 << ": analytic=" << analytic
                 << " fd=" << fd
                 << " relerr=" << relerr);
            REQUIRE(relerr < rel_tol);
        }
    }
    INFO("Bond max relative error: " << max_relerr);
}


// ============================================================================
// Angle bend: analytic gradient vs finite difference
// ============================================================================

TEST_CASE("ff_angle: analytic gradient matches finite difference", "[ff_bond_angle]") {
    // 3-atom system; only angle term contributes (LJ/Coulomb: eps=0, q=0; no bonds).
    // Angle: i=0, j=1(apex), k=2, theta0=109.5 deg, fc=50 kcal/mol/rad^2
    const int natom = 3;
    FFMinData ff = make_empty_ff(natom);
    FFAngle ang;
    ang.i      = 0; ang.j = 1; ang.k = 2;
    ang.theta0 = static_cast<Real>(109.5 * M_PI / 180.0);
    ang.fc     = 50.0f;
    ff.lig_angles.push_back(ang);

    // Geometry: angle ~100 degrees at apex (atom 1 at origin).
    // atom 0: (1, 0, 0)
    // atom 1: (0, 0, 0)  <- apex
    // atom 2: (cos100, sin100, 0)
    const double ang100_deg = 100.0 * M_PI / 180.0;
    std::vector<Real> coords = {
        1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        static_cast<Real>(std::cos(ang100_deg)),
        static_cast<Real>(std::sin(ang100_deg)),
        0.0f
    };

    MinimalSetup setup;
    setup.init(natom, coords);

    // Analytic gradient
    std::vector<Real> aux_f(natom * 3, 0.0f);
    Real E0 = cal_e_f_ff_cpu(&setup.pose, setup.topo, setup.fix_mol, ff, aux_f.data());
    REQUIRE(std::isfinite(E0));
    REQUIRE(E0 > 0.0f);

    // Finite-difference check (same strategy as bond test above).
    const Real h        = 1e-3f;
    const Real rel_tol  = 1e-2f;
    const Real abs_tol  = 1e-3f;
    const Real zero_thr = 1e-3f;

    Real max_relerr = 0.0f;
    for (int k = 0; k < natom * 3; k++) {
        std::vector<Real> coords_plus  = coords;
        std::vector<Real> coords_minus = coords;
        coords_plus[k]  += h;
        coords_minus[k] -= h;

        Real Ep = energy_only(coords_plus,  natom, setup.topo, setup.fix_mol, ff);
        Real Em = energy_only(coords_minus, natom, setup.topo, setup.fix_mol, ff);
        Real fd = (Ep - Em) / (2.0f * h);

        Real analytic = aux_f[k];
        Real abs_err  = std::abs(analytic - fd);

        if (std::abs(analytic) < zero_thr && std::abs(fd) < zero_thr) {
            INFO("Angle coord " << k
                 << ": analytic=" << analytic
                 << " fd=" << fd
                 << " abs_err=" << abs_err << " (near-zero, abs check)");
            REQUIRE(abs_err < abs_tol);
        } else {
            Real denom   = std::max(std::abs(analytic), std::abs(fd));
            Real relerr  = abs_err / denom;
            max_relerr   = std::max(max_relerr, relerr);
            INFO("Angle coord " << k
                 << ": analytic=" << analytic
                 << " fd=" << fd
                 << " relerr=" << relerr);
            REQUIRE(relerr < rel_tol);
        }
    }
    INFO("Angle max relative error: " << max_relerr);
}


// ============================================================================
// Sign / descent check: taking a step along -aux_f reduces energy
// ============================================================================

TEST_CASE("ff_bond: energy decreases along negative gradient", "[ff_bond_angle]") {
    const int natom = 2;
    FFMinData ff = make_empty_ff(natom);
    FFBond bond;
    bond.i = 0; bond.j = 1; bond.r0 = 1.5f; bond.fc = 300.0f;
    ff.lig_bonds.push_back(bond);

    std::vector<Real> coords = {0.0f, 0.0f, 0.0f, 1.6f, 0.05f, -0.02f};
    MinimalSetup setup;
    setup.init(natom, coords);

    std::vector<Real> aux_f(natom * 3, 0.0f);
    Real E0 = cal_e_f_ff_cpu(&setup.pose, setup.topo, setup.fix_mol, ff, aux_f.data());

    // Steepest descent step: coords_new = coords - alpha * aux_f
    const Real alpha = 1e-4f;
    std::vector<Real> coords_new = coords;
    for (int k = 0; k < natom * 3; k++) {
        coords_new[k] -= alpha * aux_f[k];
    }

    Real E1 = energy_only(coords_new, natom, setup.topo, setup.fix_mol, ff);
    INFO("E0=" << E0 << " E1=" << E1);
    REQUIRE(E1 < E0);
}

TEST_CASE("ff_angle: energy decreases along negative gradient", "[ff_bond_angle]") {
    const int natom = 3;
    FFMinData ff = make_empty_ff(natom);
    FFAngle ang;
    ang.i = 0; ang.j = 1; ang.k = 2;
    ang.theta0 = static_cast<Real>(109.5 * M_PI / 180.0);
    ang.fc     = 50.0f;
    ff.lig_angles.push_back(ang);

    const double ang100_deg = 100.0 * M_PI / 180.0;
    std::vector<Real> coords = {
        1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        static_cast<Real>(std::cos(ang100_deg)),
        static_cast<Real>(std::sin(ang100_deg)),
        0.0f
    };
    MinimalSetup setup;
    setup.init(natom, coords);

    std::vector<Real> aux_f(natom * 3, 0.0f);
    Real E0 = cal_e_f_ff_cpu(&setup.pose, setup.topo, setup.fix_mol, ff, aux_f.data());

    const Real alpha = 1e-4f;
    std::vector<Real> coords_new = coords;
    for (int k = 0; k < natom * 3; k++) {
        coords_new[k] -= alpha * aux_f[k];
    }

    Real E1 = energy_only(coords_new, natom, setup.topo, setup.fix_mol, ff);
    INFO("E0=" << E0 << " E1=" << E1);
    REQUIRE(E1 < E0);
}


// ============================================================================
// Energy units sanity: 1-degree perturbation of angle with fc=50 should
// give E ~ fc * (1*pi/180)^2 = 50 * (0.01745)^2 ~ 0.01521 kcal/mol
// ============================================================================

TEST_CASE("ff_angle: energy magnitude consistent with kcal/mol units", "[ff_bond_angle]") {
    const int natom = 3;
    FFMinData ff = make_empty_ff(natom);
    FFAngle ang;
    ang.i = 0; ang.j = 1; ang.k = 2;
    ang.theta0 = static_cast<Real>(109.5 * M_PI / 180.0);  // equilibrium
    ang.fc     = 50.0f;
    ff.lig_angles.push_back(ang);

    // Geometry at exactly theta0 + 1 degree
    const double perturb_rad = 1.0 * M_PI / 180.0;  // 1 degree in radians
    const double theta_actual = 109.5 * M_PI / 180.0 + perturb_rad;

    // Atom 0: unit vector along x; atom 1: apex at origin; atom 2: unit vector at theta_actual
    std::vector<Real> coords = {
        1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        static_cast<Real>(std::cos(theta_actual)),
        static_cast<Real>(std::sin(theta_actual)),
        0.0f
    };
    MinimalSetup setup;
    setup.init(natom, coords);

    std::vector<Real> aux_f(natom * 3, 0.0f);
    Real E = cal_e_f_ff_cpu(&setup.pose, setup.topo, setup.fix_mol, ff, aux_f.data());

    // Expected: fc * dtheta^2 = 50 * (pi/180)^2 ~ 0.01521 kcal/mol
    Real dt     = static_cast<Real>(perturb_rad);
    Real E_expected = 50.0f * dt * dt;  // ~0.01521

    INFO("E_actual=" << E << " E_expected=" << E_expected);
    REQUIRE(std::isfinite(E));
    REQUIRE(E > 0.0f);
    // Should match within 1% (float precision)
    REQUIRE(std::abs(E - E_expected) / E_expected < 0.01f);
}
