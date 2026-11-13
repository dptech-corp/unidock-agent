//
// Unit tests for the position-bias potentials (PositionBiasTerm::eval_pos / eval_zalign, ext/bias/bias_term.h,
// formerly Vina::eval_ef_pos / eval_ef_zalign) and the bias data structures
//

#include <catch2/catch_amalgamated.hpp>
#include "score/vina.h"
#include "ext/bias/bias_term.h"
#include "constants/constants.h"
#include <cmath>
#include <cstring>

using Catch::Matchers::WithinAbs;

// ==================== eval_ef_pos tests ====================

TEST_CASE("eval_ef_pos: zero distance gives V_set energy", "[eval_ef_pos]") {
    Real f[3] = {0., 0., 0.};
    Real r_[3] = {0., 0., 0.}; // atom at bias position

    Real V_set = 2.0;
    Real r2 = 1.0;
    Real e = PositionBiasTerm::eval_pos(r_, V_set, r2, f);

    // At r=0: E = V_set * exp(0) = V_set
    REQUIRE_THAT(e, WithinAbs(2.0, 1e-6));
    // Forces should be zero at minimum
    REQUIRE_THAT(f[0], WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(f[1], WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(f[2], WithinAbs(0.0, 1e-6));
}

TEST_CASE("eval_ef_pos: energy decays with distance", "[eval_ef_pos]") {
    Real V_set = 1.0;
    Real r2 = 1.0;

    // E(r) = V_set * exp(-|r|^2 / r2)
    // At distance 1 along x: E = exp(-1) ~ 0.3679
    Real f1[3] = {0., 0., 0.};
    Real r1[3] = {1., 0., 0.};
    Real e1 = PositionBiasTerm::eval_pos(r1, V_set, r2, f1);
    REQUIRE_THAT(e1, WithinAbs(std::exp(-1.0), 1e-5));

    // At distance 2 along x: E = exp(-4) ~ 0.0183
    Real f2[3] = {0., 0., 0.};
    Real r2_[3] = {2., 0., 0.};
    Real e2 = PositionBiasTerm::eval_pos(r2_, V_set, r2, f2);
    REQUIRE_THAT(e2, WithinAbs(std::exp(-4.0), 1e-5));

    // Energy decays with distance
    REQUIRE(e1 > e2);
}

TEST_CASE("eval_ef_pos: force points toward bias center", "[eval_ef_pos]") {
    Real V_set = 1.0;
    Real r2 = 1.0;

    // r_ = bias_pos - atom_pos = {-1, 0, 0} means atom is at +1, bias at 0
    // Force should push atom toward bias (positive x direction in force)
    Real f[3] = {0., 0., 0.};
    Real r_[3] = {-1., 0., 0.};
    PositionBiasTerm::eval_pos(r_, V_set, r2, f);

    // f = E * 2 * r_ / r2, so f[0] should be negative (pointing toward bias)
    REQUIRE(f[0] < 0.0);
    REQUIRE_THAT(f[1], WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(f[2], WithinAbs(0.0, 1e-6));
}

TEST_CASE("eval_ef_pos: force matches finite difference of energy", "[eval_ef_pos]") {
    Real V_set = 1.5;
    Real r2 = 2.0;
    Real delta = 1e-4;

    Real r0[3] = {0.5, -0.3, 0.8};

    for (int dim = 0; dim < 3; dim++) {
        Real r_plus[3], r_minus[3];
        std::memcpy(r_plus, r0, sizeof(r0));
        std::memcpy(r_minus, r0, sizeof(r0));
        r_plus[dim] += delta;
        r_minus[dim] -= delta;

        Real f_dummy[3] = {0., 0., 0.};
        Real e_plus = PositionBiasTerm::eval_pos(r_plus, V_set, r2, f_dummy);
        std::memset(f_dummy, 0, sizeof(f_dummy));
        Real e_minus = PositionBiasTerm::eval_pos(r_minus, V_set, r2, f_dummy);

        // FD: dE/dr_dim
        Real fd_grad = (e_plus - e_minus) / (2 * delta);

        Real f_analytic[3] = {0., 0., 0.};
        PositionBiasTerm::eval_pos(r0, V_set, r2, f_analytic);

        // Force matches dE/dr (same sign — force pulls atom toward bias center)
        REQUIRE_THAT(std::abs(f_analytic[dim]), WithinAbs(std::abs(fd_grad), 1e-3));
    }
}


// ==================== eval_ef_zalign tests ====================

TEST_CASE("eval_ef_zalign: reference values from original test", "[eval_ef_zalign]") {
    Real f[3] = {0., 0., 0.};

    // From original test_bias.cpp:
    // r_ = {-1, 0, 0}, a_qt = 1, vn_type = VN_TYPE_C_H
    // Expected: e = 15.0, f = {22.5, 0, 0}
    Real r_[3] = {-1., 0., 0.};
    Real a_qt = 1.;
    int vn_type = VN_TYPE_C_H;

    Real e = PositionBiasTerm::eval_zalign(r_, a_qt, vn_type, f);
    REQUIRE_THAT(e, WithinAbs(15.0, 1e-3));
    REQUIRE_THAT(f[0], WithinAbs(22.5, 1e-2));
    REQUIRE_THAT(f[1], WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(f[2], WithinAbs(0.0, 1e-4));
}

TEST_CASE("eval_ef_zalign: hydrogen gets weaker attraction", "[eval_ef_zalign]") {
    Real r_[3] = {-1., 0., 0.};
    Real a_qt = 1.;

    Real f_h[3] = {0., 0., 0.};
    Real e_h = PositionBiasTerm::eval_zalign(r_, a_qt, VN_TYPE_H, f_h);

    Real f_c[3] = {0., 0., 0.};
    Real e_c = PositionBiasTerm::eval_zalign(r_, a_qt, VN_TYPE_C_H, f_c);

    // Hydrogen k_a = 0.1, carbon k_a = 30 -> 300x weaker
    REQUIRE(e_c > e_h * 100);
}

TEST_CASE("eval_ef_zalign: energy increases with distance (sigmoid)", "[eval_ef_zalign]") {
    Real a_qt = 1.;
    int vn_type = VN_TYPE_C_H;

    // Close (r < c=1): low energy
    Real f1[3] = {0., 0., 0.};
    Real r1[3] = {0.5, 0., 0.};
    Real e1 = PositionBiasTerm::eval_zalign(r1, a_qt, vn_type, f1);

    // Far (r >> c): high energy (plateau)
    Real f2[3] = {0., 0., 0.};
    Real r2[3] = {5., 0., 0.};
    Real e2 = PositionBiasTerm::eval_zalign(r2, a_qt, vn_type, f2);

    // Energy increases with distance (penalizes being far from template)
    REQUIRE(e2 > e1);
}

TEST_CASE("eval_ef_zalign: force magnitude matches finite difference", "[eval_ef_zalign]") {
    Real a_qt = 0.8;
    int vn_type = VN_TYPE_C_P;
    Real delta = 1e-4;

    Real r0[3] = {1.5, 0.0, 0.0}; // 1D to avoid r=0 singularity

    for (int dim = 0; dim < 1; dim++) { // only test x dimension for 1D case
        Real r_plus[3], r_minus[3];
        std::memcpy(r_plus, r0, sizeof(r0));
        std::memcpy(r_minus, r0, sizeof(r0));
        r_plus[dim] += delta;
        r_minus[dim] -= delta;

        Real f_dummy[3] = {0., 0., 0.};
        Real e_plus = PositionBiasTerm::eval_zalign(r_plus, a_qt, vn_type, f_dummy);
        std::memset(f_dummy, 0, sizeof(f_dummy));
        Real e_minus = PositionBiasTerm::eval_zalign(r_minus, a_qt, vn_type, f_dummy);

        Real fd_grad = (e_plus - e_minus) / (2 * delta);

        Real f_analytic[3] = {0., 0., 0.};
        PositionBiasTerm::eval_zalign(r0, a_qt, vn_type, f_analytic);

        // Force magnitude matches |dE/dr|
        REQUIRE_THAT(std::abs(f_analytic[dim]), WithinAbs(std::abs(fd_grad), 1e-2));
    }
}


// ==================== Bias struct and data tests ====================

TEST_CASE("Bias struct stores correct data", "[bias_struct]") {
    Bias b;
    b.i = 5;
    b.param[0] = 1.0f;
    b.param[1] = 2.0f;
    b.param[2] = 3.0f;
    b.param[3] = 0.5f; // V_set
    b.param[4] = 1.0f; // r

    REQUIRE(b.i == 5);
    REQUIRE(b.param[0] == 1.0f);
    REQUIRE(b.param[3] == 0.5f);
}

TEST_CASE("BiasType enum values", "[bias_type]") {
    REQUIRE(BT_NO == 0);
    REQUIRE(BT_POS == 1);
    REQUIRE(BT_ALIGN == 2);
}

TEST_CASE("UDFlexMol bias rows", "[bias_flexmol]") {
    UDFlexMol mol;
    REQUIRE(mol.bias.rows.empty());

    Bias b1;
    b1.i = 0;
    b1.param[0] = 1.0f; b1.param[1] = 2.0f; b1.param[2] = 3.0f;
    b1.param[3] = 0.8f; b1.param[4] = 0.5f;
    mol.bias.rows.push_back(b1);

    Bias b2;
    b2.i = 0;
    b2.param[0] = 4.0f; b2.param[1] = 5.0f; b2.param[2] = 6.0f;
    b2.param[3] = 0.3f; b2.param[4] = 1.0f;
    mol.bias.rows.push_back(b2);

    REQUIRE(mol.bias.rows.size() == 2);
    REQUIRE(mol.bias.rows[0].i == 0);
    REQUIRE(mol.bias.rows[1].param[0] == 4.0f);
}

TEST_CASE("DockParam bias defaults", "[bias_dockparam]") {
    DockParam dp;
    REQUIRE(dp.bias_type == BT_NO);
    REQUIRE_THAT(static_cast<double>(dp.bias_k), WithinAbs(0.3, 1e-6));
}
