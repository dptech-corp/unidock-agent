//
// Unit tests for low-level FF functions in gaff2.h
// Tests analytical gradients against finite differences.
//

#include <catch2/catch_amalgamated.hpp>
#include "score/gaff2.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr Real FD_DELTA = 1e-4f;

// Helper: finite difference derivative dE/dr
static Real fd_deriv(Real (*func)(Real, Real, Real, Real*),
                     Real r, Real p1, Real p2) {
    Real dummy;
    Real e_plus  = func(r + FD_DELTA, p1, p2, &dummy);
    Real e_minus = func(r - FD_DELTA, p1, p2, &dummy);
    return (e_plus - e_minus) / (2 * FD_DELTA);
}

// ============================================================================
// ff_lj: LJ 12-6
// ============================================================================

TEST_CASE("ff_lj: energy at equilibrium distance", "[gaff2]") {
    // At r = sigma, s6 = 1, E = 4*eps*(1-1) = 0
    Real dE = 0;
    Real e = ff_lj(3.0f, 3.0f, 0.1f, &dE);
    REQUIRE(e == Catch::Approx(0.0f).margin(1e-5));
}

TEST_CASE("ff_lj: energy at r = 2^(1/6)*sigma (minimum)", "[gaff2]") {
    // At r = 2^(1/6)*sigma, E = -eps
    Real sigma = 3.0f;
    Real eps = 0.1f;
    Real r_min = sigma * std::pow(2.0, 1.0/6.0);
    Real dE = 0;
    Real e = ff_lj(r_min, sigma, eps, &dE);
    REQUIRE(e == Catch::Approx(-eps).margin(1e-4));
    // At minimum, dE/dr = 0
    REQUIRE(std::abs(dE) < 1e-3);
}

TEST_CASE("ff_lj: gradient matches finite difference", "[gaff2]") {
    Real sigma = 3.4f, eps = 0.12f;
    Real test_distances[] = {2.5f, 3.0f, 3.5f, 4.0f, 5.0f, 8.0f};

    for (Real r : test_distances) {
        Real dE_analytic = 0;
        ff_lj(r, sigma, eps, &dE_analytic);
        Real dE_numeric = fd_deriv(ff_lj, r, sigma, eps);
        Real margin = std::max(1e-3f, std::abs(dE_analytic) * 0.01f);
        REQUIRE(dE_analytic == Catch::Approx(dE_numeric).margin(margin));
    }
}

TEST_CASE("ff_lj: repulsive at short distance", "[gaff2]") {
    Real dE = 0;
    Real e = ff_lj(2.0f, 3.0f, 0.1f, &dE);
    REQUIRE(e > 0);  // repulsive
    REQUIRE(dE < 0);  // dE/dr < 0 → force pushes atoms apart
}

// ============================================================================
// ff_coulomb: distance-dependent dielectric
// ============================================================================

TEST_CASE("ff_coulomb: opposite charges attract", "[gaff2]") {
    Real dE = 0;
    Real e = ff_coulomb(5.0f, 0.5f, -0.5f, &dE);
    REQUIRE(e < 0);  // attractive
}

TEST_CASE("ff_coulomb: like charges repel", "[gaff2]") {
    Real dE = 0;
    Real e = ff_coulomb(5.0f, 0.5f, 0.5f, &dE);
    REQUIRE(e > 0);  // repulsive
}

TEST_CASE("ff_coulomb: gradient matches finite difference", "[gaff2]") {
    Real test_distances[] = {2.0f, 3.0f, 5.0f, 8.0f, 12.0f};
    Real qi = 0.3f, qj = -0.4f;

    for (Real r : test_distances) {
        Real dE_analytic = 0;
        ff_coulomb(r, qi, qj, &dE_analytic);
        Real dE_numeric = fd_deriv(ff_coulomb, r, qi, qj);
        Real margin = std::max(1e-4f, std::abs(dE_analytic) * 0.01f);
        REQUIRE(dE_analytic == Catch::Approx(dE_numeric).margin(margin));
    }
}

TEST_CASE("ff_coulomb: energy scales as 1/r^2", "[gaff2]") {
    Real dE1 = 0, dE2 = 0;
    Real e1 = ff_coulomb(4.0f, 1.0f, 1.0f, &dE1);
    Real e2 = ff_coulomb(8.0f, 1.0f, 1.0f, &dE2);
    // E ∝ 1/r^2, so E(4)/E(8) = 64/16 = 4
    REQUIRE(e1 / e2 == Catch::Approx(4.0f).margin(0.01));
}

// ============================================================================
// ff_pair14: pre-scaled 1-4 interactions
// ============================================================================

TEST_CASE("ff_pair14: gradient matches finite difference", "[gaff2]") {
    // Typical AMBER 1-4 pair parameters
    Real aij = 1e6f;   // LJ A (kcal/mol * A^12)
    Real bij = 500.0f;  // LJ B (kcal/mol * A^6)
    Real qij = 0.05f;   // pre-scaled charge product
    Real test_distances[] = {3.0f, 4.0f, 5.0f, 7.0f};

    for (Real r : test_distances) {
        Real dE_analytic = 0;
        ff_pair14(r, aij, bij, qij, &dE_analytic);

        // Numerical FD
        Real dummy;
        Real e_plus  = ff_pair14(r + FD_DELTA, aij, bij, qij, &dummy);
        Real e_minus = ff_pair14(r - FD_DELTA, aij, bij, qij, &dummy);
        Real dE_numeric = (e_plus - e_minus) / (2 * FD_DELTA);

        Real margin = std::max(1e-2f, std::abs(dE_analytic) * 0.02f);
        REQUIRE(dE_analytic == Catch::Approx(dE_numeric).margin(margin));
    }
}

// ============================================================================
// ff_torsion: periodic torsion potential
// ============================================================================

TEST_CASE("ff_torsion: cos2 term E = fc2*(1+cos(2*phi))", "[gaff2]") {
    Real fc[7] = {0, 0, 1.5f, 0, 0, 0, 0};  // only fc2
    Real phi0 = 0;
    Real dE = 0;

    // At phi=0: E = 1.5*(1+cos(0)) = 3.0
    Real e0 = ff_torsion(0, phi0, fc, &dE);
    REQUIRE(e0 == Catch::Approx(3.0f).margin(1e-4));

    // At phi=pi/2: E = 1.5*(1+cos(pi)) = 0
    Real e90 = ff_torsion(M_PI/2, phi0, fc, &dE);
    REQUIRE(e90 == Catch::Approx(0.0f).margin(1e-4));

    // At phi=pi: E = 1.5*(1+cos(2*pi)) = 3.0
    Real e180 = ff_torsion(M_PI, phi0, fc, &dE);
    REQUIRE(e180 == Catch::Approx(3.0f).margin(1e-4));
}

TEST_CASE("ff_torsion: gradient matches finite difference", "[gaff2]") {
    Real fc[7] = {0.5f, 1.0f, 0.8f, 0.3f, 0, 0, 0};
    Real phi0 = M_PI / 6;

    Real test_angles[] = {0.0f, 0.5f, 1.0f, 2.0f, -1.5f, M_PI};

    for (Real phi : test_angles) {
        Real dE_analytic = 0;
        ff_torsion(phi, phi0, fc, &dE_analytic);

        Real dummy;
        Real e_plus  = ff_torsion(phi + FD_DELTA, phi0, fc, &dummy);
        Real e_minus = ff_torsion(phi - FD_DELTA, phi0, fc, &dummy);
        Real dE_numeric = (e_plus - e_minus) / (2 * FD_DELTA);

        Real margin = std::max(1e-3f, std::abs(dE_analytic) * 0.01f);
        REQUIRE(dE_analytic == Catch::Approx(dE_numeric).margin(margin));
    }
}

TEST_CASE("ff_torsion: constant term fc0 has zero derivative", "[gaff2]") {
    Real fc[7] = {2.5f, 0, 0, 0, 0, 0, 0};  // only constant
    Real dE = 0;
    Real e = ff_torsion(1.0f, 0, fc, &dE);
    REQUIRE(e == Catch::Approx(2.5f).margin(1e-5));
    REQUIRE(dE == Catch::Approx(0.0f).margin(1e-5));
}

TEST_CASE("ff_torsion: phi0 phase shift works correctly", "[gaff2]") {
    Real fc[7] = {0, 0, 0, 1.0f, 0, 0, 0};  // only fc3
    Real phi0 = M_PI;  // phase = pi
    Real dE = 0;

    // E = fc3*(1+cos(3*phi - pi))
    // At phi=pi/3: cos(3*pi/3 - pi) = cos(0) = 1, E = 2.0
    Real e = ff_torsion(M_PI/3, phi0, fc, &dE);
    REQUIRE(e == Catch::Approx(2.0f).margin(1e-4));
}
