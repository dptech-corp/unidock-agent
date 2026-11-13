//
// Aromatic-ring bias in the CPU scoring pipeline (cal_e_f_cpu / cal_e_grad_cpu) on the tyk2 ligand ejm42
// (benzene ring atoms 0-5 and pyridine ring atoms 10-15, on opposite sides of rotatable bonds):
//   1. energy equals the double-precision closed form; BT_NO and BT_POS ignore the rows (bitwise)
//   2. translation, rotation and torsion gradients match central finite differences, ring bias only
//   3. the ring term adds to the atom bias without changing it
//   4. the pose pass's host term set (score/terms.h HostTermSet, plan step 7) gives the search energy of the atom and
//      ring bias at the same coordinates, under every bias type
// (the October test of feat/ud2align-ring-bias; the bias type / k now sit in FlexParamVina::terms.settings, the rows
// in terms.bias / terms.ring, score/terms.h)
//

#include <catch2/catch_amalgamated.hpp>
#include "cpu/common_cpu.h"
#include "cpu/score_cpu.h"
#include "cpu/host_topo.h"
#include "cpu/spatial_grid.h"
#include "format/dms_reader.h"
#include "score/terms.h"
#include "constants/constants.h"
#include <cmath>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;

namespace {

const std::string TYK2_LIGAND = std::string(TEST_DATA_DIR) + "ff_minimize/tyk2/ligand_ejm42_enriched.dms";
const Real BIAS_K = 0.3;

void unit(double* v){
    const double l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] /= l; v[1] /= l; v[2] /= l;
}

/// Centroid and unit Newell normal of ring atoms ids (double precision)
void ring_frame(const std::vector<Real>& x, const std::vector<int>& ids, double* c, double* n){
    const int m = ids.size();
    c[0] = c[1] = c[2] = 0;
    n[0] = n[1] = n[2] = 0;
    for (int k = 0; k < m; k++) for (int j = 0; j < 3; j++) c[j] += double(x[ids[k] * 3 + j]) / m;
    for (int k = 0; k < m; k++){
        const Real* p = x.data() + ids[k] * 3;
        const Real* q = x.data() + ids[(k + 1) % m] * 3;
        const double a[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};
        const double b[3] = {q[0] - c[0], q[1] - c[1], q[2] - c[2]};
        n[0] += a[1] * b[2] - a[2] * b[1];
        n[1] += a[2] * b[0] - a[0] * b[2];
        n[2] += a[0] * b[1] - a[1] * b[0];
    }
    unit(n);
}

/// Row on ring atoms ids, target = own centroid + shift, target normal = unit(base + tilt)
RingBiasRow make_row(const std::vector<Real>& x, const std::vector<int>& ids, const double* shift,
                     const double* base, const double* tilt, double v_set){
    double c[3], n[3];
    ring_frame(x, ids, c, n);
    RingBiasRow r;
    r.n = ids.size();
    for (int k = 0; k < r.n; k++) r.atoms[k] = ids[k];
    double nt[3] = {base[0] + tilt[0], base[1] + tilt[1], base[2] + tilt[2]};
    unit(nt);
    for (int j = 0; j < 3; j++){ r.c[j] = Real(c[j] + shift[j]); r.nt[j] = Real(nt[j]); }
    r.v_set = Real(v_set);
    return r;
}

/// The test rows: benzene and pyridine each onto a tilted, shifted copy of themselves, the pyridine also
/// onto a benzene-oriented target (many-to-many), and a five-atom row on pyridine atoms 10-14 for the n = 5 path.
std::vector<RingBiasRow> test_rows(const std::vector<Real>& x){
    const std::vector<int> benz = {0, 1, 2, 3, 4, 5}, pyr = {10, 11, 12, 13, 14, 15}, five = {10, 11, 12, 13, 14};
    double cb[3], nb[3], cp[3], np[3], c5[3], n5[3];
    ring_frame(x, benz, cb, nb);
    ring_frame(x, pyr, cp, np);
    ring_frame(x, five, c5, n5);
    const double s1[3] = {0.5, -0.3, 0.4}, t1[3] = {0.3, 0.4, -0.2};
    const double s2[3] = {-0.4, 0.6, 0.3}, t2[3] = {-0.3, 0.2, 0.4};
    const double s3[3] = {1.0, 0.5, -0.5}, t0[3] = {0, 0, 0};
    const double s4[3] = {0.3, 0.3, 0.3}, t4[3] = {0.5, -0.4, 0.1};
    return {make_row(x, benz, s1, nb, t1, 2.0), make_row(x, pyr, s2, np, t2, 4.0),
            make_row(x, pyr, s3, nb, t0, 2.0), make_row(x, five, s4, n5, t4, 1.0)};
}

double closed_form(const std::vector<Real>& x, const std::vector<RingBiasRow>& rows, double bias_k){
    double e = 0;
    for (const auto& r : rows){
        std::vector<int> ids(r.atoms, r.atoms + r.n);
        double c[3], n[3];
        ring_frame(x, ids, c, n);
        const double d = std::sqrt((c[0] - r.c[0]) * (c[0] - r.c[0]) + (c[1] - r.c[1]) * (c[1] - r.c[1])
                                   + (c[2] - r.c[2]) * (c[2] - r.c[2]));
        const double cs = n[0] * r.nt[0] + n[1] * r.nt[1] + n[2] * r.nt[2];
        e += -double(r.v_set) * bias_k * 30.0 * cs * cs / (1.0 + std::exp(3.0 * (d - 1.0)));
    }
    return e;
}

/// tyk2 ejm42 with no receptor and no intra pairs, so only the bias terms score. With `centre`
/// the coordinates are shifted to put the ligand centre at the origin (finite differences in float).
struct RingLigand {
    UDFlexMol lig;
    HostFlexTopo host_topo;
    std::vector<int> inds_bias;
    std::vector<Real> params_bias;
    std::vector<RingBiasRow> rows;
    std::vector<Real> coords, dihedrals;
    FlexParamVina param{};
    FlexPose pose{};
    FixMol fix{};
    FixParamVina fp{};

    explicit RingLigand(bool centre){
        AtomTypeProps atp; init_default_atom_type_props(atp);
        FFMinData ff;
        read_ligand_dms_full(TYK2_LIGAND, lig, ff, atp, false);
        build_host_flex_topo(lig, host_topo);
        coords = lig.coords;
        dihedrals = lig.dihedrals;
        Real cen[3] = {lig.center[0], lig.center[1], lig.center[2]};
        if (centre){
            for (int i = 0; i < lig.natom; i++) for (int j = 0; j < 3; j++) coords[i * 3 + j] -= cen[j];
            cen[0] = cen[1] = cen[2] = 0;
        }
        inds_bias.assign(lig.natom * 2, 0);
        param.npair_intra = 0;
        param.atom_types = lig.vina_types.data();
        param.terms.bias.inds = inds_bias.data();
        param.terms.settings.bias_type = BT_ALIGN;
        param.terms.settings.bias_k = BIAS_K;
        pose.coords = coords.data();
        pose.dihedrals = dihedrals.data();
        for (int j = 0; j < 3; j++) pose.center[j] = cen[j];
        fix.natom = 0;
        fix.coords = nullptr;
        fp.atom_types = nullptr;
    }
    void set_rings(const std::vector<RingBiasRow>& r){
        rows = r;
        param.terms.ring.n = rows.size();
        param.terms.ring.rows = rows.empty() ? nullptr : rows.data();
    }
    void set_atom_bias(const std::vector<Bias>& b){
        inds_bias.assign(lig.natom * 2, 0);
        params_bias.clear();
        int last_i = -1;
        for (const auto& x : b){
            if (x.i != last_i) inds_bias[x.i * 2] = params_bias.size();
            for (int k = 0; k < 5; k++) params_bias.push_back(x.param[k]);
            inds_bias[x.i * 2 + 1] = params_bias.size();
            last_i = x.i;
        }
        param.terms.bias.inds = inds_bias.data();
        param.terms.bias.params = params_bias.empty() ? nullptr : params_bias.data();
    }
    void set_bias_type(BiasType bt){ param.terms.settings.bias_type = bt; }
    Real energy(std::vector<Real>* aux = nullptr){
        std::vector<Real> f(lig.natom * 3, 0);
        const Real e = cal_e_f_cpu(&pose, host_topo.topo, fix, param, fp, f.data());
        if (aux) *aux = f;
        return e;
    }
    Real energy_at(const std::vector<Real>& x){
        FlexPose p = pose;
        std::vector<Real> c = x;
        p.coords = c.data();
        std::vector<Real> f(lig.natom * 3, 0);
        return cal_e_f_cpu(&p, host_topo.topo, fix, param, fp, f.data());
    }
};

/// Constants for a bias-only evaluation: no receptor grid, box far away, no torsion library penalty (the bias type
/// and k are the ligand's terms.settings, RingLigand::set_bias_type)
struct BiasConstants {
    BiasConstants(){
        init_default_atom_type_props(g_cpu_constants.atp);
        free_spatial_grid(g_cpu_constants.receptor_grid);
        g_cpu_constants.FLAG_CONSTRAINT_DOCK = false;
        g_cpu_constants.USE_TORSION_PENALTY = false;
        g_cpu_constants.USE_ANGULAR_HBOND = false;
        g_cpu_constants.USE_INNER_BOX = false;
        g_cpu_constants.PENALTY_SLOPE = 10.0;
        g_cpu_constants.BOX_X_LO = -100; g_cpu_constants.BOX_X_HI = 100;
        g_cpu_constants.BOX_Y_LO = -100; g_cpu_constants.BOX_Y_HI = 100;
        g_cpu_constants.BOX_Z_LO = -100; g_cpu_constants.BOX_Z_HI = 100;
    }
};

void rotate_about(std::vector<Real>& x, const Real* c, int axis, double ang){
    const double cs = std::cos(ang), sn = std::sin(ang);
    const int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
    for (size_t i = 0; i < x.size() / 3; i++){
        const double u = double(x[i * 3 + a1]) - c[a1], v = double(x[i * 3 + a2]) - c[a2];
        x[i * 3 + a1] = Real(c[a1] + cs * u - sn * v);
        x[i * 3 + a2] = Real(c[a2] + sn * u + cs * v);
    }
}

}  // namespace


TEST_CASE("cpu ring bias: energy is the closed form, only under BT_ALIGN", "[cpu][ring_bias]"){
    RingLigand s(false);
    REQUIRE(s.lig.natom == 35);
    const std::vector<RingBiasRow> rows = test_rows(s.coords);
    BiasConstants bc;

    s.set_rings({});
    std::vector<Real> f_none;
    const Real e_none = s.energy(&f_none);
    REQUIRE(e_none == Real(0));

    s.set_rings(rows);
    const double ref = closed_form(s.coords, rows, BIAS_K);
    const Real e = s.energy();
    printf("[cpu ring bias closed form] %zu rows | E %.6f | closed form %.6f\n", rows.size(), double(e), ref);
    REQUIRE(ref < -10);
    REQUIRE_THAT(double(e), WithinAbs(ref, 2e-5 * std::fabs(ref)));

    for (BiasType bt : {BT_NO, BT_POS}){
        INFO("bias type " << int(bt));
        s.set_bias_type(bt);
        std::vector<Real> f_on, f_off;
        s.set_rings(rows);
        const Real e_on = s.energy(&f_on);
        s.set_rings({});
        const Real e_off = s.energy(&f_off);
        REQUIRE(e_on == e_off);
        REQUIRE(f_on == f_off);
    }
}


TEST_CASE("cpu ring bias: translation, rotation and torsion gradients match finite differences", "[cpu][ring_bias]"){
    RingLigand s(true);
    REQUIRE(s.host_topo.topo.ntorsion > 0);
    BiasConstants bc;
    s.set_rings(test_rows(s.coords));

    const int ntor = s.host_topo.topo.ntorsion;
    std::vector<Real> dihe_g(ntor, 0), aux_f(s.lig.natom * 3, 0);
    FlexPoseGradient g;
    g.dihedrals_g = dihe_g.data();
    const Real e0 = cal_e_grad_cpu(&s.pose, &g, s.host_topo.topo, s.fix, s.param, s.fp, aux_f.data());
    REQUIRE(e0 < Real(-10));

    double gmax = 0;
    const double h = 2e-3;
    for (int k = 0; k < 3; k++){
        std::vector<Real> xp = s.coords, xm = s.coords;
        for (int i = 0; i < s.lig.natom; i++){ xp[i * 3 + k] += Real(h); xm[i * 3 + k] -= Real(h); }
        const double fd = (double(s.energy_at(xp)) - double(s.energy_at(xm))) / (2 * h);
        printf("[cpu ring bias fd] center_g[%d] analytic %.5f fd %.5f\n", k, double(g.center_g[k]), fd);
        CHECK_THAT(double(g.center_g[k]), WithinAbs(fd, 1e-2 * std::max(1.0, std::fabs(fd))));
        gmax = std::max(gmax, std::fabs(fd));
    }
    for (int k = 0; k < 3; k++){
        std::vector<Real> xp = s.coords, xm = s.coords;
        rotate_about(xp, s.pose.center, k, h);
        rotate_about(xm, s.pose.center, k, -h);
        const double fd = (double(s.energy_at(xp)) - double(s.energy_at(xm))) / (2 * h);
        printf("[cpu ring bias fd] orientation_g[%d] analytic %.5f fd %.5f\n", k, double(g.orientation_g[k]), fd);
        CHECK_THAT(double(g.orientation_g[k]), WithinAbs(fd, 1e-2 * std::max(1.0, std::fabs(fd))));
        gmax = std::max(gmax, std::fabs(fd));
    }
    auto e_dihe = [&](int i_tor, Real dth){
        std::vector<Real> c2 = s.coords, d2 = s.dihedrals;
        FlexPose p2 = s.pose; p2.coords = c2.data(); p2.dihedrals = d2.data();
        apply_grad_update_dihe_cpu(&p2, &s.host_topo.topo, i_tor, dth, true);
        std::vector<Real> f(s.lig.natom * 3, 0);
        return double(cal_e_f_cpu(&p2, s.host_topo.topo, s.fix, s.param, s.fp, f.data()));
    };
    int n_moving = 0;
    for (int i_tor = 0; i_tor < ntor; i_tor++){
        const double fd = (e_dihe(i_tor, Real(h)) - e_dihe(i_tor, Real(-h))) / (2 * h);
        printf("[cpu ring bias fd] dihe_g[%d] analytic %.5f fd %.5f\n", i_tor, double(dihe_g[i_tor]), fd);
        CHECK_THAT(double(dihe_g[i_tor]), WithinAbs(fd, 1e-2 * std::max(1.0, std::fabs(fd))));
        if (std::fabs(fd) > 0.1) n_moving++;
    }
    REQUIRE(gmax > 1);
    REQUIRE(n_moving > 0);
}


TEST_CASE("cpu ring bias: adds to the atom bias without changing it", "[cpu][ring_bias]"){
    RingLigand s(false);
    BiasConstants bc;
    const std::vector<RingBiasRow> rows = test_rows(s.coords);
    std::vector<Bias> atom_rows;
    for (int a : {0, 3, 7, 12, 16, 19}){
        Bias b;
        b.i = a;
        b.param[0] = float(s.coords[a * 3] + 0.7); b.param[1] = float(s.coords[a * 3 + 1] - 0.4);
        b.param[2] = float(s.coords[a * 3 + 2] + 0.9); b.param[3] = 2.0f; b.param[4] = 2.0f;
        atom_rows.push_back(b);
    }

    std::vector<Real> f_atom, f_ring, f_both;
    s.set_atom_bias(atom_rows); s.set_rings({});
    const Real e_atom = s.energy(&f_atom);
    s.set_atom_bias({}); s.set_rings(rows);
    const Real e_ring = s.energy(&f_ring);
    s.set_atom_bias(atom_rows); s.set_rings(rows);
    const Real e_both = s.energy(&f_both);
    printf("[cpu ring bias additive] E atom %.6f ring %.6f both %.6f\n", double(e_atom), double(e_ring), double(e_both));
    REQUIRE(std::fabs(double(e_atom)) > 1);
    REQUIRE_THAT(double(e_both), WithinAbs(double(e_atom) + double(e_ring), 1e-5 * std::fabs(double(e_both))));
    for (size_t i = 0; i < f_both.size(); i++){
        INFO("aux_f[" << i << "]");
        REQUIRE_THAT(double(f_both[i]), WithinAbs(double(f_atom[i]) + double(f_ring[i]), 1e-5 * std::max(1.0, std::fabs(double(f_both[i])))));
    }
}


TEST_CASE("cpu pose pass: the host term set gives the search's atom + ring bias energy at the same coordinates",
          "[cpu][ring_bias][bias][pose_pass]"){
    RingLigand s(false);
    BiasConstants bc;
    const std::vector<RingBiasRow> rows = test_rows(s.coords);
    int h = -1;   // a hydrogen: the Z-align potential's weaker k_a
    for (int a = 0; a < s.lig.natom && h < 0; a++) if (s.lig.vina_types[a] == VN_TYPE_H) h = a;
    REQUIRE(h >= 0);
    std::vector<Bias> atom_rows;
    for (int a : {0, 3, 3, 7, 12, 16, 19, h}){   // two rows on atom 3; each atom's rows contiguous, as the reader gives them
        Bias b;
        b.i = a;
        b.param[0] = float(s.coords[a * 3] + 0.7); b.param[1] = float(s.coords[a * 3 + 1] - 0.4);
        b.param[2] = float(s.coords[a * 3 + 2] + 0.9); b.param[3] = 2.0f; b.param[4] = 2.0f;
        atom_rows.push_back(b);
    }
    s.set_atom_bias(atom_rows);
    s.set_rings(rows);
    // the ligand as DockTask holds it: its rows in UDFlexMol::bias / ::ring, flattened by the host term set itself
    UDFlexMol m = s.lig;
    m.bias.rows = atom_rows;
    m.ring.rows = rows;

    std::vector<Real> moved = s.coords;
    rotate_about(moved, s.pose.center, 2, 0.4);
    for (Real& v : moved) v += Real(0.3);
    for (BiasType bt : {BT_ALIGN, BT_POS, BT_NO}){
        s.set_bias_type(bt);
        DockParam dp;
        dp.bias_type = bt;
        dp.bias_k = BIAS_K;
        HostTermSet host(m, term_settings_of(dp));
        for (const auto* x : {&s.coords, &moved}){
            const double e_search = s.energy_at(*x);   // cal_e_f_cpu, no receptor and no intra pair: the bias terms only
            const double e_host = host.energy(x->data());
            printf("[cpu pose pass host terms] bias type %d | search %.6f | host %.6f\n", int(bt), e_search, e_host);
            INFO("bias type " << int(bt));
            REQUIRE_THAT(e_host, WithinAbs(e_search, 1e-6 * std::max(1.0, std::fabs(e_search))));
            if (bt == BT_NO) REQUIRE(e_host == 0.0);
            else REQUIRE(std::fabs(e_host) > 1);
        }
    }
}
