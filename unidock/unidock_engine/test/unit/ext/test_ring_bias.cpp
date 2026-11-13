//
// Aromatic-ring bias term ring_bias_frame / ring_bias_atom_grad (ext/bias/ring_term.h): closed form at overlay / flipped /
// perpendicular / far / generic geometries, guards, and central finite differences of every ring-atom coordinate (the
// October test, score/test_ring_bias.cpp of feat/ud2align-ring-bias, at the default RingBiasParams); the run-time
// parameters k, s, d0 (TermSettings::ring) against the same closed form.
//

#include <catch2/catch_amalgamated.hpp>
#include "score/terms.h"   // ext/bias/ring_term.h and TermSettings
#include <cmath>
#include <vector>

using Catch::Matchers::WithinAbs;

namespace {

const double BIAS_K = 0.3;

struct Ring {
    std::vector<Real> coords;  // ring atoms first, in ring order
    RingBiasRow row;
};

void unit(double* v){
    const double l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] /= l; v[1] /= l; v[2] /= l;
}

/// Regular m-ring of radius rad centred at c0 in the plane spanned by e1, e2 (orthonormal); its Newell
/// normal is e1 x e2. Atom k gets an out-of-plane displacement pucker[k] along that normal.
Ring make_ring(int m, double rad, const double* c0, const double* e1, const double* e2,
               const std::vector<double>& pucker = {}){
    Ring r;
    double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    for (int k = 0; k < m; k++){
        const double th = 2 * M_PI * k / m;
        const double z = pucker.empty() ? 0.0 : pucker[k];
        for (int j = 0; j < 3; j++){
            r.coords.push_back(Real(c0[j] + rad * (std::cos(th) * e1[j] + std::sin(th) * e2[j]) + z * n[j]));
        }
        r.row.atoms[k] = k;
    }
    r.row.n = m;
    return r;
}

/// Double-precision closed form of E straight from the definition (Newell sum on raw coordinates).
double energy_ref(const std::vector<Real>& x, const RingBiasRow& row, double bias_k, double K = 30.0, double S = 3.0,
                  double D0 = 1.0){
    const int m = row.n;
    double c[3] = {0, 0, 0}, N[3] = {0, 0, 0};
    for (int k = 0; k < m; k++) for (int j = 0; j < 3; j++) c[j] += x[row.atoms[k] * 3 + j] / m;
    for (int k = 0; k < m; k++){
        const Real* p = x.data() + row.atoms[k] * 3;
        const Real* q = x.data() + row.atoms[(k + 1) % m] * 3;
        N[0] += double(p[1]) * q[2] - double(p[2]) * q[1];
        N[1] += double(p[2]) * q[0] - double(p[0]) * q[2];
        N[2] += double(p[0]) * q[1] - double(p[1]) * q[0];
    }
    unit(N);
    const double d = std::sqrt((c[0] - row.c[0]) * (c[0] - row.c[0]) + (c[1] - row.c[1]) * (c[1] - row.c[1])
                               + (c[2] - row.c[2]) * (c[2] - row.c[2]));
    const double cs = N[0] * row.nt[0] + N[1] * row.nt[1] + N[2] * row.nt[2];
    const double a = double(row.v_set) * bias_k * K;
    return -a * cs * cs / (1.0 + std::exp(S * (d - D0)));
}

void set_target(RingBiasRow& row, const double* c, const double* nt, double v_set){
    double n[3] = {nt[0], nt[1], nt[2]};
    unit(n);
    for (int j = 0; j < 3; j++){ row.c[j] = Real(c[j]); row.nt[j] = Real(n[j]); }
    row.v_set = Real(v_set);
}

double max_abs(const Real* g, int n){
    double m = 0;
    for (int i = 0; i < n; i++) m = std::max(m, std::fabs(double(g[i])));
    return m;
}

const double E1[3] = {1, 0, 0}, E2[3] = {0, 1, 0};

/// E and dE/dx of the row's atoms in ring order (18 Reals, zero past n), as the scoring loops assemble them
Real ring_eval(const Real* x, const RingBiasRow& row, Real bias_k, Real* g, const RingBiasParams& p = RingBiasParams{}){
    RingBiasFrame fr;
    const Real e = ring_bias_frame(x, row, bias_k, p, fr);
    for (int i = 0; i < 18; i++) g[i] = 0;
    for (int k = 0; k < row.n; k++) ring_bias_atom_grad(x, row, fr, k, g + k * 3);
    return e;
}

}  // namespace


TEST_CASE("ring bias: closed form at overlay, flipped, perpendicular and far", "[ring_bias]"){
    const double c0[3] = {0.3, -0.2, 0.5};
    Ring r = make_ring(6, 1.39, c0, E1, E2);
    const double v_set = 2.0, a = v_set * BIAS_K * 30.0;
    const double e_overlay = -a / (1.0 + std::exp(-3.0));
    Real g[18];

    SECTION("overlay: E = -a F(0), no force"){
        const double z[3] = {0, 0, 1};
        set_target(r.row, c0, z, v_set);
        const Real e = ring_eval(r.coords.data(), r.row, Real(BIAS_K), g);
        REQUIRE_THAT(double(e), WithinAbs(e_overlay, 1e-5 * a));
        REQUIRE(max_abs(g, 18) < 1e-4 * a);
    }
    SECTION("flipped template normal counts as parallel"){
        const double z[3] = {0, 0, -1};
        set_target(r.row, c0, z, v_set);
        const Real e = ring_eval(r.coords.data(), r.row, Real(BIAS_K), g);
        REQUIRE_THAT(double(e), WithinAbs(e_overlay, 1e-5 * a));
        REQUIRE(max_abs(g, 18) < 1e-4 * a);
    }
    SECTION("perpendicular: E = 0, no force"){
        const double t[3] = {0.6, 0.8, 0};
        set_target(r.row, c0, t, v_set);
        const Real e = ring_eval(r.coords.data(), r.row, Real(BIAS_K), g);
        REQUIRE_THAT(double(e), WithinAbs(0.0, 1e-5));
        REQUIRE(max_abs(g, 18) < 1e-4);
    }
    SECTION("centroid 10 A away: E ~ 0"){
        const double far[3] = {c0[0] + 6, c0[1] - 8, c0[2]};
        const double z[3] = {0, 0, 1};
        set_target(r.row, far, z, v_set);
        const Real e = ring_eval(r.coords.data(), r.row, Real(BIAS_K), g);
        REQUIRE(std::fabs(double(e)) < 1e-9);
        REQUIRE(max_abs(g, 18) < 1e-8);
    }
}


TEST_CASE("ring bias: closed form at generic geometries, 5- and 6-membered", "[ring_bias]"){
    // ring planes tilted against the template normal, centroids off the target, far from the origin
    double e1[3] = {0.8, 0.6, 0.0}, e2[3] = {-0.36, 0.48, 0.8};
    unit(e2);
    const double c0[3] = {-5.3, 25.1, -31.6};
    for (int m : {5, 6}){
        for (double d : {0.0, 0.4, 1.0, 1.3, 2.5}){
            Ring r = make_ring(m, m == 6 ? 1.39 : 1.18, c0, e1, e2, {0.05, -0.04, 0.03, -0.02, 0.04, -0.06});
            const double tgt[3] = {c0[0] + 0.6 * d, c0[1] - 0.48 * d, c0[2] + 0.64 * d};
            const double nt[3] = {0.2, -0.5, 0.84};
            set_target(r.row, tgt, nt, 4.0);
            Real g[18];
            const double e = ring_eval(r.coords.data(), r.row, Real(BIAS_K), g);
            const double ref = energy_ref(r.coords, r.row, BIAS_K);
            INFO("m " << m << " d " << d << " E " << e << " closed form " << ref);
            REQUIRE(ref < -1e-3);
            REQUIRE_THAT(e, WithinAbs(ref, 2e-5 * std::max(1.0, std::fabs(ref))));
        }
    }
}


TEST_CASE("ring bias: degenerate ring scores zero", "[ring_bias]"){
    Ring r;
    r.row.n = 6;
    for (int k = 0; k < 6; k++){
        r.row.atoms[k] = k;
        r.coords.push_back(Real(1.4 * k)); r.coords.push_back(Real(0.5)); r.coords.push_back(Real(-0.2));
    }
    const double tgt[3] = {3.5, 0.5, -0.2}, nt[3] = {0, 0, 1};
    set_target(r.row, tgt, nt, 2.0);
    Real g[18];
    REQUIRE(ring_eval(r.coords.data(), r.row, Real(BIAS_K), g) == Real(0));
    REQUIRE(max_abs(g, 18) == 0.0);
}


TEST_CASE("ring bias: gradient matches central finite differences on every ring atom", "[ring_bias]"){
    double e1[3] = {0.6, 0.0, 0.8}, e2[3] = {0.0, 1.0, 0.0};
    const double c0[3] = {0.7, -0.4, 1.1};
    for (int m : {5, 6}){
        for (double d : {0.35, 0.9, 1.6}){
            Ring r = make_ring(m, m == 6 ? 1.39 : 1.18, c0, e1, e2, {0.12, -0.08, 0.1, -0.05, 0.07, -0.11});
            const double tgt[3] = {c0[0] + 0.48 * d, c0[1] + 0.6 * d, c0[2] - 0.64 * d};
            const double nt[3] = {0.3, 0.4, 0.87};
            set_target(r.row, tgt, nt, 3.0);
            Real g[18], scratch[18];
            ring_eval(r.coords.data(), r.row, Real(BIAS_K), g);
            const Real h = Real(2e-3);
            for (int i = 0; i < m * 3; i++){
                std::vector<Real> xp = r.coords, xm = r.coords;
                xp[i] += h; xm[i] -= h;
                const double fd = (double(ring_eval(xp.data(), r.row, Real(BIAS_K), scratch))
                                   - double(ring_eval(xm.data(), r.row, Real(BIAS_K), scratch))) / (double(xp[i]) - double(xm[i]));
                INFO("m " << m << " d " << d << " coordinate " << i << " analytic " << g[i] << " fd " << fd);
                REQUIRE_THAT(double(g[i]), WithinAbs(fd, 1e-2 * std::max(1.0, std::fabs(fd))));
            }
            for (int i = m * 3; i < 18; i++) REQUIRE(g[i] == Real(0));
        }
    }
}


TEST_CASE("ring bias: the default parameters are the October constants k 30, s 3, d0 1", "[ring_bias]"){
    const RingBiasParams p;
    REQUIRE(p.k == Real(30.0));
    REQUIRE(p.s == Real(3.0));
    REQUIRE(p.d0 == Real(1.0));
    const TermSettings s;
    REQUIRE(s.ring.k == p.k);
    REQUIRE(s.ring.s == p.s);
    REQUIRE(s.ring.d0 == p.d0);
    DockParam dp;
    dp.ring_bias.k = 12; dp.ring_bias.s = 2; dp.ring_bias.d0 = 1.5;
    const TermSettings t = term_settings_of(dp);
    REQUIRE(t.ring.k == Real(12));
    REQUIRE(t.ring.s == Real(2));
    REQUIRE(t.ring.d0 == Real(1.5));
}


TEST_CASE("ring bias: run-time k, s, d0 give the closed form with those values, and its gradient", "[ring_bias]"){
    const double c0[3] = {0.2, -0.1, 0.3};
    Ring r = make_ring(6, 1.39, c0, E1, E2, {0.05, -0.04, 0.03, -0.05, 0.04, -0.03});
    const double ct[3] = {0.9, 0.4, -0.2}, nt[3] = {0.2, -0.3, 1.0};
    set_target(r.row, ct, nt, 2.5);
    for (const RingBiasParams& p : {RingBiasParams{Real(12), Real(2), Real(1.5)}, RingBiasParams{Real(45), Real(4.5), Real(0.5)}}){
        INFO("k " << p.k << " s " << p.s << " d0 " << p.d0);
        Real g[18];
        const double e = ring_eval(r.coords.data(), r.row, Real(BIAS_K), g, p);
        const double ref = energy_ref(r.coords, r.row, BIAS_K, p.k, p.s, p.d0);
        REQUIRE(ref < -0.1);
        REQUIRE_THAT(e, WithinAbs(ref, 1e-5 * std::fabs(ref)));
        Real scratch[18];
        for (int i = 0; i < 18; i++){
            std::vector<Real> xp = r.coords, xm = r.coords;
            xp[i] += Real(1e-3); xm[i] -= Real(1e-3);
            const double fd = (double(ring_eval(xp.data(), r.row, Real(BIAS_K), scratch, p))
                               - double(ring_eval(xm.data(), r.row, Real(BIAS_K), scratch, p))) / (double(xp[i]) - double(xm[i]));
            CHECK_THAT(double(g[i]), WithinAbs(fd, 2e-3 * std::max(1.0, std::fabs(fd))));
        }
    }
}


TEST_CASE("ring bias: RingBiasTerm::accumulate_as on the host is the per-row sum, forces added per ring atom", "[ring_bias]"){
    // two rows sharing atoms (one query ring on two template rings), serial lane 0 / stride 1 as the CPU path runs it
    const double c0[3] = {0, 0, 0};
    Ring r = make_ring(6, 1.39, c0, E1, E2, {0.05, -0.04, 0.03, -0.05, 0.04, -0.03});
    const double ct1[3] = {0.5, 0.2, 0.1}, nt1[3] = {0.1, 0.2, 1.0}, ct2[3] = {-0.4, 0.3, 0.6}, nt2[3] = {1.0, 0.1, 0.4};
    RingBiasRow rows[2] = {r.row, r.row};
    set_target(rows[0], ct1, nt1, 2.0);
    set_target(rows[1], ct2, nt2, 4.0);
    RingBiasTerm term;
    term.rows = rows;
    term.n = 2;
    const RingBiasParams p;
    std::vector<Real> aux(18, Real(0.25));
    Real energy = Real(-1);
    term.accumulate_as(Real(BIAS_K), p, 0, 1, r.coords.data(), energy, aux.data());
    Real g1[18], g2[18];
    const Real e1 = ring_eval(r.coords.data(), rows[0], Real(BIAS_K), g1);
    const Real e2 = ring_eval(r.coords.data(), rows[1], Real(BIAS_K), g2);
    REQUIRE(energy == (Real(-1) + e1) + e2);
    for (int i = 0; i < 18; i++) REQUIRE(aux[i] == (Real(0.25) + g1[i]) + g2[i]);
    REQUIRE(ring_bias_energy({rows[0], rows[1]}, r.coords.data(), Real(BIAS_K), p) == Real(0) + e1 + e2);
}
