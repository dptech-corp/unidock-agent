//
// Receptor maps of the CUDA GRID path (src/cuda/maps.cu, cal_e_f_warp<true> in bfgs.cuh) on the tyk2 fixture:
//   (a) map value (one-atom probe) vs the analytic path and vs the host sum of per-pair-curled eval_ef_nohb
//   (b) interpolant gradient vs central finite differences of the interpolant
//   (c) GRID vs analytic evaluation of the crystal pose; exact, hydrogen bond included, for a pose on map points
//   (d) use_grid disabled: HEAD's analytic wrapper is bit-identical with and without the maps
//

#include <catch2/catch_amalgamated.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "optimize/bfgs.h"
#include "cuda/common.cuh"
#include "cuda/maps.cuh"
#include "cpu/common_cpu.h"
#include "cpu/score_cpu.h"
#include "cpu/host_topo.h"
#include "format/dms_reader.h"

static const std::string TYK2_DIR = std::string(TEST_DATA_DIR) + "ff_minimize/tyk2/";

// tyk2 receptor (30 A box + 8 A around the crystal ligand), the ligand, device constants, receptor cell list and
// the maps for the ligand's heavy types plus N_P / O_P (the retyped probe of test c)
struct Tyk2Maps {
    AtomTypeProps atp;
    UDFlexMol lig; FFMinData ff_lig;
    UDFixMol rec; FFMinData ff_rec;
    DockParam dp;
    FixMol fix{}; FixParamVina fp{};
    HostFlexTopo host_topo;
    std::vector<Real> r1r2; std::vector<int> inds_bias;
    FlexParamVina param{};
    std::vector<Real> coords, dihedrals; FlexPose pose{};
    std::vector<int> types_present; int nheavy = 0;
    bool built = false; std::string summary;
    RecMaps maps{};

    Tyk2Maps(){
        init_default_atom_type_props(atp);
        read_ligand_dms_full(TYK2_DIR + "ligand_ejm42_enriched.dms", lig, ff_lig, atp, false);
        const Real half = 15, cutoff = 8;
        dp.box.x_lo = lig.center[0] - half; dp.box.x_hi = lig.center[0] + half;
        dp.box.y_lo = lig.center[1] - half; dp.box.y_hi = lig.center[1] + half;
        dp.box.z_lo = lig.center[2] - half; dp.box.z_hi = lig.center[2] + half;
        Box box_protein = dp.box;
        box_protein.x_lo -= cutoff; box_protein.x_hi += cutoff; box_protein.y_lo -= cutoff; box_protein.y_hi += cutoff;
        box_protein.z_lo -= cutoff; box_protein.z_hi += cutoff;
        read_receptor_dms_full(TYK2_DIR + "receptor_enriched.dms", box_protein, rec, ff_rec, atp);
        fix.natom = rec.natom; fix.coords = rec.coords.data();
        fp.atom_types = rec.vina_types.data(); fp.rec_donor_h = rec.rec_donor_h;

        build_host_flex_topo(lig, host_topo);
        for (size_t j = 0; j < lig.intra_pairs.size(); j += 2)
            r1r2.push_back(VN_VDW_RADII[lig.vina_types[lig.intra_pairs[j]]] + VN_VDW_RADII[lig.vina_types[lig.intra_pairs[j + 1]]]);
        inds_bias.assign(lig.natom * 2, 0);
        param.npair_intra = lig.intra_pairs.size() / 2; param.pairs_intra = lig.intra_pairs.data();
        param.r1_plus_r2_intra = r1r2.data(); param.atom_types = lig.vina_types.data(); param.terms.bias.inds = inds_bias.data();
        init_constants_cpu(dp, atp);   // build_bonded_h_topo reads g_cpu_constants.atp
        build_bonded_h_topo(param.bonded_h_topo, lig.coords.data(), lig.vina_types.data(), lig.natom);
        coords = lig.coords; dihedrals = lig.dihedrals;
        pose.coords = coords.data(); pose.dihedrals = dihedrals.data();
        pose.center[0] = lig.center[0]; pose.center[1] = lig.center[1]; pose.center[2] = lig.center[2];

        init_constants(dp, atp);
        upload_receptor_grid(rec.coords.data(), rec.vina_types.data(), rec.natom, atp);
        bool present[MAX_VN_TYPES] = {};
        for (int a = 0; a < lig.natom; a++){
            if (atp.is_hydrogen[lig.vina_types[a]]) continue;
            present[lig.vina_types[a]] = true; nheavy++;
        }
        present[VN_TYPE_N_P] = present[VN_TYPE_O_P] = true;
        for (int t = 0; t < MAX_VN_TYPES; t++) if (present[t]) types_present.push_back(t);
        built = build_receptor_maps(rec, atp, {}, dp, types_present, nheavy, summary);
        get_receptor_maps(maps);
    }
    ~Tyk2Maps(){ free_bonded_h_topo(param.bonded_h_topo); free_receptor_maps(); free_receptor_grid(); }

    // distance from p to the nearest receptor heavy atom (polar_only: donors / acceptors only)
    Real min_rec_dist(const Real* p, bool polar_only = false) const {
        double best = 1e30;
        for (int j = 0; j < rec.natom; j++){
            if (polar_only && !(atp.is_donor[rec.vina_types[j]] || atp.is_acceptor[rec.vina_types[j]])) continue;
            const double dx = rec.coords[j * 3] - p[0], dy = rec.coords[j * 3 + 1] - p[1], dz = rec.coords[j * 3 + 2] - p[2];
            best = std::min(best, dx * dx + dy * dy + dz * dz);
        }
        return Real(std::sqrt(best));
    }
    void snap(const double* q, Real* out) const {   // nearest map point
        for (int k = 0; k < 3; k++) out[k] = Real(maps.origin[k] + std::round((q[k] - maps.origin[k]) * maps.inv_h) * maps.h);
    }
};

// Rigid toy ligand without intra pairs: inside the box the inter term is the whole energy
struct RigidToy {
    std::vector<int> types; std::vector<Real> coords; std::vector<int> inds_bias;
    FlexTopo topo{}; FlexParamVina param{}; FlexPose pose{};
    RigidToy(const std::vector<int>& t, const std::vector<Real>& c) : types(t), coords(c), inds_bias(t.size() * 2, 0){
        topo.natom = (int)types.size(); topo.vn_types = types.data();
        param.atom_types = types.data(); param.terms.bias.inds = inds_bias.data();
        build_bonded_h_topo(param.bonded_h_topo, coords.data(), types.data(), topo.natom);
        pose.coords = coords.data();
        for (int i = 0; i < topo.natom; i++) for (int k = 0; k < 3; k++) pose.center[k] += coords[i * 3 + k] / topo.natom;
    }
    ~RigidToy(){ free_bonded_h_topo(param.bonded_h_topo); }
};

struct Result { Real e = 0; FlexPoseGradient g{}; std::vector<Real> dihe_g; };

// grid: cal_e_grad_warp<true> (maps + polar pass); head: HEAD's wrapper cal_e_grad_one_gpu (analytic, hunt cap)
static Result eval(const Tyk2Maps& F, const FlexParamVina& param, const FlexPose& pose, const FlexTopo& topo, bool grid, bool authentic, bool head = false){
    Result r; r.dihe_g.assign(std::max(topo.ntorsion, 1), 0); r.g.dihedrals_g = r.dihe_g.data();
    if (head) cal_e_grad_one_gpu(F.fix, F.fp, param, pose, topo, &r.g, &r.e);
    else cal_e_grad_one_gpu_cap(F.fix, F.fp, param, pose, topo, &r.g, &r.e, grid, authentic);
    return r;
}
// one heavy atom of type t at p: the GRID energy is the interpolant (plus the exact polar pass for donors /
// acceptors) and center_g its gradient
static Result eval_probe(const Tyk2Maps& F, int t, const Real* p, bool grid, bool authentic){
    RigidToy toy({t}, {p[0], p[1], p[2]});
    return eval(F, toy.param, toy.pose, toy.topo, grid, authentic);
}
static bool same_bits(const Result& a, const Result& b, int ntorsion){
    return std::memcmp(&a.e, &b.e, sizeof(Real)) == 0 && std::memcmp(a.g.center_g, b.g.center_g, 3 * sizeof(Real)) == 0
        && std::memcmp(a.g.orientation_g, b.g.orientation_g, 3 * sizeof(Real)) == 0
        && std::memcmp(a.dihe_g.data(), b.dihe_g.data(), ntorsion * sizeof(Real)) == 0;
}

// Trilinear interpolation error of the per-pair-curled sums at 0.375 A, measured on tyk2: up to 0.09 kcal/mol for a
// single probe atom 2.5 A or more from every receptor atom (the convex repulsive wall interpolates high)
static constexpr double GRID_E_TOL = 0.15;

// Random points inside the docking box, kept 0.1 h away from the map cell faces so a central difference of the
// interpolant never straddles a cell
static std::vector<std::array<Real, 3>> box_points(const Tyk2Maps& F, int n, unsigned seed){
    std::mt19937 rng(seed); std::uniform_real_distribution<double> u(0, 1);
    // 0.5 A inside the box: the cell-face snap below must never push a point past a box wall (box penalty)
    const Real lo[3] = {F.dp.box.x_lo + Real(0.5), F.dp.box.y_lo + Real(0.5), F.dp.box.z_lo + Real(0.5)};
    const Real hi[3] = {F.dp.box.x_hi - Real(0.5), F.dp.box.y_hi - Real(0.5), F.dp.box.z_hi - Real(0.5)};
    std::vector<std::array<Real, 3>> pts(n);
    for (auto& p : pts) for (int k = 0; k < 3; k++){
        const double g = (lo[k] + u(rng) * (hi[k] - lo[k]) - F.maps.origin[k]) * F.maps.inv_h;
        p[k] = Real(F.maps.origin[k] + (std::floor(g) + 0.1 + 0.8 * (g - std::floor(g))) * F.maps.h);
    }
    return pts;
}

// What the maps store at p for type t: the per-pair-curled sum of eval_ef_nohb over the receptor heavy atoms within
// the cutoff (build_maps_kernel), summed in double on the host
static double host_map_value(const Tyk2Maps& F, Vina& score, int t, const Real* p, Real v){
    double sum = 0;
    for (int j = 0; j < F.rec.natom; j++){
        const int t2 = F.rec.vina_types[j];
        const double dx = F.rec.coords[j * 3] - p[0], dy = F.rec.coords[j * 3 + 1] - p[1], dz = F.rec.coords[j * 3 + 2] - p[2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= score.r2_cutoff) continue;
        Real f = 0;
        Real e = score.eval_ef_nohb(Real(std::sqrt(r2)) - (VN_VDW_RADII[t] + VN_VDW_RADII[t2]), t, t2, &f, F.atp);
        if (e > 0) e *= v / (v + e);   // curl_warp
        sum += e;
    }
    return sum;
}


TEST_CASE("grid maps: map values vs the analytic path and the host sum (one-atom probes)", "[grid_maps]"){
    Tyk2Maps F;
    INFO(F.summary);
    REQUIRE(F.built);
    REQUIRE(F.maps.built);
    REQUIRE(F.maps.nslot == (int)F.types_present.size());
    for (int t : F.types_present) REQUIRE(F.maps.slot_of_type[t] >= 0);
    printf("[grid_maps] %s | receptor atoms %d, ligand heavy atoms %d, map types %zu\n", F.summary.c_str(), F.rec.natom, F.nheavy, F.types_present.size());
    Vina score; score.set_weights(F.dp.weights);
    const auto pts = box_points(F, 200, 20261008u);
    for (int t : F.types_present){
        const bool polar = F.atp.is_donor[t] || F.atp.is_acceptor[t];
        int n_far = 0; double dev_an = 0, dev_host = 0, dev_auth = 0, e_max = 0;
        for (size_t i = 0; i < pts.size(); i++){
            const Real* p = pts[i].data();
            const Result g = eval_probe(F, t, p, true, false);
            REQUIRE(std::isfinite(g.e));
            if (F.min_rec_dist(p) < Real(2.5)) continue;
            n_far++;
            const Real a = eval_probe(F, t, p, false, false).e;
            dev_an = std::max(dev_an, (double)std::fabs(g.e - a)); e_max = std::max(e_max, (double)std::fabs(a));
            INFO("type " << t << " point " << i << " grid " << g.e << " analytic " << a);
            CHECK_THAT(g.e, Catch::Matchers::WithinAbs(a, GRID_E_TOL));
            if (!polar){   // no polar pass: the energy is the interpolant itself
                const double hv = host_map_value(F, score, t, p, curl_v_cap(false));
                dev_host = std::max(dev_host, std::fabs(g.e - hv));
                CHECK_THAT((double)g.e, Catch::Matchers::WithinAbs(hv, GRID_E_TOL));
            }
            if (i < 50){   // cap set 1
                const Real ga = eval_probe(F, t, p, true, true).e, aa = eval_probe(F, t, p, false, true).e;
                dev_auth = std::max(dev_auth, (double)std::fabs(ga - aa));
                CHECK_THAT(ga, Catch::Matchers::WithinAbs(aa, GRID_E_TOL));
            }
        }
        printf("[grid_maps a] type %2d%s: %3d / %zu points beyond 2.5 A | max |grid - analytic| %.2e (authentic cap %.2e) | max |grid - host sum| %.2e | max |E| %.3f\n",
               t, polar ? " (polar)" : "        ", n_far, pts.size(), dev_an, dev_auth, dev_host, e_max);
    }
}


TEST_CASE("grid maps: interpolant gradient vs central finite differences", "[grid_maps]"){
    Tyk2Maps F;
    REQUIRE(F.built);
    const auto pts = box_points(F, 40, 777u);
    const Real delta = 1e-3;
    double worst = 0, worst_abs = 0; int n = 0;
    for (int t : F.types_present){
        const bool polar = F.atp.is_donor[t] || F.atp.is_acceptor[t];
        for (const auto& pt : pts){
            const Real* p = pt.data();
            // beyond 2.5 A of every atom; polar probes also beyond the hydrogen-bond range of polar atoms
            if (F.min_rec_dist(p) < Real(2.5) || (polar && F.min_rec_dist(p, true) < Real(4))) continue;
            const Result g0 = eval_probe(F, t, p, true, false);
            for (int k = 0; k < 3; k++){
                Real pp[3] = {p[0], p[1], p[2]}, pm[3] = {p[0], p[1], p[2]};
                pp[k] += delta; pm[k] -= delta;
                // divide by the displacement the float coordinates actually carry: at |y| ~ 40 A the float grid is
                // 4e-6 A, 0.4 % of delta, which would otherwise show up as a spurious gradient error
                const double span = (double)pp[k] - (double)pm[k];
                const Real fd = Real(((double)eval_probe(F, t, pp, true, false).e - (double)eval_probe(F, t, pm, true, false).e) / span);
                const double scale = std::max(1.0, (double)std::fabs(g0.g.center_g[k]));
                worst = std::max(worst, std::fabs(g0.g.center_g[k] - fd) / scale);
                worst_abs = std::max(worst_abs, (double)std::fabs(g0.g.center_g[k] - fd));
                INFO("type " << t << " axis " << k << " grad " << g0.g.center_g[k] << " fd " << fd);
                CHECK_THAT(g0.g.center_g[k], Catch::Matchers::WithinAbs(fd, 1e-3 * scale));
                n++;
            }
        }
    }
    printf("[grid_maps b] %d gradient components over %zu types, delta %.0e A: worst |grad - fd| %.2e, worst |grad - fd| / max(1, |grad|) %.2e\n",
           n, F.types_present.size(), (double)delta, worst_abs, worst);
}


TEST_CASE("grid maps: GRID vs analytic evaluation of the crystal pose; exact on map points", "[grid_maps]"){
    Tyk2Maps F;
    REQUIRE(F.built);
    const FlexTopo& topo = F.host_topo.topo;
    for (bool authentic : {false, true}){
        const Result g = eval(F, F.param, F.pose, topo, true, authentic), a = eval(F, F.param, F.pose, topo, false, authentic);
        double gdev = 0;
        for (int k = 0; k < 3; k++) gdev = std::max({gdev, (double)std::fabs(g.g.center_g[k] - a.g.center_g[k]), (double)std::fabs(g.g.orientation_g[k] - a.g.orientation_g[k])});
        for (int i = 0; i < topo.ntorsion; i++) gdev = std::max(gdev, (double)std::fabs(g.dihe_g[i] - a.dihe_g[i]));
        printf("[grid_maps c] crystal pose, %s cap: E grid %.4f analytic %.4f | diff %.4f = %.4f per heavy atom (%d) | max |grad diff| %.3f\n",
               authentic ? "authentic" : "hunt", g.e, a.e, g.e - a.e, (g.e - a.e) / F.nheavy, F.nheavy, gdev);
        REQUIRE(std::isfinite(g.e));
        REQUIRE_THAT(g.e, Catch::Matchers::WithinAbs(a.e, 0.5 * F.nheavy));
    }

    // On map points the interpolant is exact, so GRID equals the analytic sum with the hydrogen bond. The hydrogen-bond
    // term is isolated by retyping N_D -> N_P and O_A -> O_P (same radius and hydrophobicity: same map, no hydrogen bond).
    // Ligand donor N_D 2.9 A from a receptor acceptor along a map axis, its H 3 h towards the acceptor (D-H...A ~ 180 deg);
    // ligand acceptor O_A 2.9 A from a receptor donor along that donor's H direction; C_H and C_P near the ligand centre.
    REQUIRE(VN_VDW_RADII[VN_TYPE_N_D] == VN_VDW_RADII[VN_TYPE_N_P]);
    REQUIRE(VN_VDW_RADII[VN_TYPE_O_A] == VN_VDW_RADII[VN_TYPE_O_P]);
    const RecDonorHTopo& rh = F.rec.rec_donor_h;
    const Real* c = F.lig.center;
    int j_acc = -1, j_don = -1;
    for (int j = 0; j < F.rec.natom && (j_acc < 0 || j_don < 0); j++){
        const int t = F.rec.vina_types[j];
        const Real* x = F.rec.coords.data() + j * 3;
        if ((x[0] - c[0]) * (x[0] - c[0]) + (x[1] - c[1]) * (x[1] - c[1]) + (x[2] - c[2]) * (x[2] - c[2]) > 64) continue;
        if (j_acc < 0 && F.atp.is_acceptor[t] && !F.atp.is_donor[t]) j_acc = j;
        if (j_don < 0 && F.atp.is_donor[t] && rh.h_start[j + 1] > rh.h_start[j]) j_don = j;
    }
    REQUIRE(j_acc >= 0);
    REQUIRE(j_don >= 0);
    const Real* A = F.rec.coords.data() + j_acc * 3;
    const Real* D = F.rec.coords.data() + j_don * 3;
    const Real* Hd = rh.h_coords + rh.h_start[j_don] * 3;
    int ax = 0;
    for (int k = 1; k < 3; k++) if (std::fabs(c[k] - A[k]) > std::fabs(c[ax] - A[ax])) ax = k;
    const double sgn = c[ax] > A[ax] ? 1 : -1;
    double q[3] = {A[0], A[1], A[2]}; q[ax] += 2.9 * sgn;
    Real xn[3], xh[3], xo[3], xc1[3], xc2[3];
    F.snap(q, xn);
    for (int k = 0; k < 3; k++) xh[k] = xn[k]; xh[ax] -= Real(3) * F.maps.h * Real(sgn);
    const double dir[3] = {Hd[0] - D[0], Hd[1] - D[1], Hd[2] - D[2]};
    const double dn = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    for (int k = 0; k < 3; k++) q[k] = D[k] + 2.9 * dir[k] / dn;
    F.snap(q, xo);
    const double q1[3] = {c[0] + 1.5, c[1], c[2]}, q2[3] = {c[0], c[1] - 1.5, c[2]};
    F.snap(q1, xc1); F.snap(q2, xc2);
    const std::vector<Real> xyz = {xn[0], xn[1], xn[2], xh[0], xh[1], xh[2], xo[0], xo[1], xo[2], xc1[0], xc1[1], xc1[2], xc2[0], xc2[1], xc2[2]};
    RigidToy on({VN_TYPE_N_D, VN_TYPE_H, VN_TYPE_O_A, VN_TYPE_C_H, VN_TYPE_C_P}, xyz);
    RigidToy off({VN_TYPE_N_P, VN_TYPE_H, VN_TYPE_O_P, VN_TYPE_C_H, VN_TYPE_C_P}, xyz);
    REQUIRE(on.param.bonded_h_topo.h_start[1] == 1);   // the H is bonded to the N_D
    for (bool authentic : {false, true}){
        const Result g_on = eval(F, on.param, on.pose, on.topo, true, authentic), a_on = eval(F, on.param, on.pose, on.topo, false, authentic);
        const Result g_off = eval(F, off.param, off.pose, off.topo, true, authentic), a_off = eval(F, off.param, off.pose, off.topo, false, authentic);
        const double hb_g = (double)g_on.e - g_off.e, hb_a = (double)a_on.e - a_off.e;
        printf("[grid_maps c] on-grid toy, %s cap: E grid %.5f analytic %.5f (retyped %.5f / %.5f) | hbond term grid %.5f analytic %.5f\n",
               authentic ? "authentic" : "hunt", g_on.e, a_on.e, g_off.e, a_off.e, hb_g, hb_a);
        REQUIRE_THAT(g_on.e, Catch::Matchers::WithinAbs(a_on.e, 1e-3 * std::max(1.0, (double)std::fabs(a_on.e))));
        REQUIRE_THAT(g_off.e, Catch::Matchers::WithinAbs(a_off.e, 1e-3 * std::max(1.0, (double)std::fabs(a_off.e))));
        REQUIRE(hb_a < -0.1);   // the probe forms a hydrogen bond
        REQUIRE_THAT(hb_g, Catch::Matchers::WithinAbs(hb_a, 1e-4 * std::max(1.0, std::fabs(hb_a))));
        // the retyped difference cancels the (identical) map part: what is left is the polar-pass force
        for (int k = 0; k < 3; k++){
            const double dg = (double)g_on.g.center_g[k] - g_off.g.center_g[k], da = (double)a_on.g.center_g[k] - a_off.g.center_g[k];
            const double dgo = (double)g_on.g.orientation_g[k] - g_off.g.orientation_g[k], dao = (double)a_on.g.orientation_g[k] - a_off.g.orientation_g[k];
            REQUIRE_THAT(dg, Catch::Matchers::WithinAbs(da, 1e-3 * std::max(1.0, std::fabs(da))));
            REQUIRE_THAT(dgo, Catch::Matchers::WithinAbs(dao, 1e-3 * std::max(1.0, std::fabs(dao))));
        }
    }
}


TEST_CASE("grid maps: use_grid disabled leaves the analytic wrapper bit-identical", "[grid_maps]"){
    Tyk2Maps F;
    REQUIRE(F.built);
    const FlexTopo& topo = F.host_topo.topo;
    const Result with_maps = eval(F, F.param, F.pose, topo, false, false, true);
    const Result new_off = eval(F, F.param, F.pose, topo, false, false);   // same instantiation, other kernel
    free_receptor_maps();
    RecMaps m; get_receptor_maps(m);
    REQUIRE(!m.built);
    const Result no_maps = eval(F, F.param, F.pose, topo, false, false, true);
    DockParam off = F.dp; off.use_grid = false;
    std::string why;
    REQUIRE(!build_receptor_maps(F.rec, F.atp, {}, off, F.types_present, F.nheavy, why));
    REQUIRE(why == "use_grid is false");
    get_receptor_maps(m);
    REQUIRE(!m.built);
    const Result off_maps = eval(F, F.param, F.pose, topo, false, false, true);
    printf("[grid_maps d] crystal pose E (HEAD wrapper): maps built %.6f, maps freed %.6f, use_grid false %.6f | new wrapper GRID=false %.6f (bit-identical: %d)\n",
           with_maps.e, no_maps.e, off_maps.e, new_off.e, (int)same_bits(with_maps, new_off, topo.ntorsion));
    REQUIRE(same_bits(with_maps, no_maps, topo.ntorsion));
    REQUIRE(same_bits(with_maps, off_maps, topo.ntorsion));
    CHECK(same_bits(with_maps, new_off, topo.ntorsion));
}


// ---------------- polar-pass candidate lists (g_pcand, maps.cu upload_polar_cands) ----------------
// Host replicas of the polar pass's pair selection (bfgs.cuh polar_pair, same float arithmetic): the sequence of
// contributing receptor atoms (original index) for one ligand atom of type t1 at p, visited by the 27-cell walk
// over g_pgrid or by the candidate list of p's g_pcand cell. n_visited counts the atoms tested.
static std::vector<int> walk_pairs(const SpatialGrid& pg, const std::vector<int>& cs, const std::vector<GridAtom>& atoms,
                                   const AtomTypeProps& atp, int t1, const Real* p, int& n_visited){
    std::vector<int> out;
    int cx = (int)((p[0] - pg.origin[0]) * pg.inv_cell_size);
    int cy = (int)((p[1] - pg.origin[1]) * pg.inv_cell_size);
    int cz = (int)((p[2] - pg.origin[2]) * pg.inv_cell_size);
    cx = std::min(std::max(cx, 0), pg.nx - 1); cy = std::min(std::max(cy, 0), pg.ny - 1); cz = std::min(std::max(cz, 0), pg.nz - 1);
    const int wx_lo = std::max(0, cx - pg.half), wx_hi = std::min(pg.nx - 1, cx + pg.half);
    const int wy_lo = std::max(0, cy - pg.half), wy_hi = std::min(pg.ny - 1, cy + pg.half);
    const int zlo = std::max(0, cz - pg.half), zspan = std::min(pg.nz - 1, cz + pg.half) - zlo + 1;
    for (int wx = wx_lo; wx <= wx_hi; wx++){
        for (int wy = wy_lo; wy <= wy_hi; wy++){
            const int c = wx * pg.ny * pg.nz + wy * pg.nz + zlo;
            for (int k = cs[c]; k < cs[c + zspan]; k++){
                const GridAtom& rec = atoms[k];
                n_visited++;
                if (!xs_h_bond_possible(t1, rec.type, atp)) continue;
                const Real dx = rec.x - p[0], dy = rec.y - p[1], dz = rec.z - p[2];
                const Real d = sqrtf(dx * dx + dy * dy + dz * dz) - (VN_VDW_RADII[t1] + rec.radius);
                if (!(d < Vina::hbond_high_val())) continue;
                out.push_back(rec.index);
            }
        }
    }
    return out;
}
static std::vector<int> cand_pairs(const PolarCands& pc, const std::vector<int>& cs, const std::vector<GridAtom>& atoms,
                                   const AtomTypeProps& atp, int t1, const Real* p, int& n_visited){
    std::vector<int> out;
    int cx = (int)((p[0] - pc.origin[0]) * pc.inv_cell);
    int cy = (int)((p[1] - pc.origin[1]) * pc.inv_cell);
    int cz = (int)((p[2] - pc.origin[2]) * pc.inv_cell);
    cx = std::min(std::max(cx, 0), pc.nx - 1); cy = std::min(std::max(cy, 0), pc.ny - 1); cz = std::min(std::max(cz, 0), pc.nz - 1);
    const int c = (cx * pc.ny + cy) * pc.nz + cz;
    for (int k = cs[c]; k < cs[c + 1]; k++){
        const GridAtom& rec = atoms[k];
        n_visited++;
        if (!xs_h_bond_possible(t1, rec.type, atp)) continue;
        const Real dx = rec.x - p[0], dy = rec.y - p[1], dz = rec.z - p[2];
        const Real d = sqrtf(dx * dx + dy * dy + dz * dz) - (VN_VDW_RADII[t1] + rec.radius);
        if (!(d < Vina::hbond_high_val())) continue;
        out.push_back(rec.index);
    }
    return out;
}

// Rigid transform of the crystal pose: coords rotated by `angle` about a unit axis through the centre, then translated
static void rigid_pose(const Tyk2Maps& F, const Real* axis, Real angle, const Real* shift, std::vector<Real>& coords, FlexPose& pose){
    const Real c = cosf(angle), s = sinf(angle), t = 1 - c;
    const Real R[3][3] = {{t * axis[0] * axis[0] + c, t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1]},
                          {t * axis[0] * axis[1] + s * axis[2], t * axis[1] * axis[1] + c, t * axis[1] * axis[2] - s * axis[0]},
                          {t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2] * axis[2] + c}};
    coords = F.lig.coords;
    for (int i = 0; i < F.lig.natom; i++){
        Real v[3];
        for (int k = 0; k < 3; k++) v[k] = F.lig.coords[i * 3 + k] - F.lig.center[k];
        for (int k = 0; k < 3; k++) coords[i * 3 + k] = R[k][0] * v[0] + R[k][1] * v[1] + R[k][2] * v[2] + F.lig.center[k] + shift[k];
    }
    pose = F.pose;
    pose.coords = coords.data();
    for (int k = 0; k < 3; k++) pose.center[k] = F.lig.center[k] + shift[k];
}


TEST_CASE("grid maps: polar candidate lists reproduce the g_pgrid walk and are bit-identical on the GPU", "[grid_maps]"){
    Tyk2Maps F;
    REQUIRE(F.built);
    SpatialGrid pg{}; std::vector<int> pg_cs; std::vector<GridAtom> pg_atoms;
    get_polar_grid(pg, pg_cs, pg_atoms);
    PolarCands pc{}; std::vector<int> pc_cs; std::vector<GridAtom> pc_atoms;
    get_polar_cands(pc, pc_cs, pc_atoms);
    REQUIRE(pg.built);
    REQUIRE(pc.built);
    // r_cut: largest R1 + R2 over the hbond-possible heavy type pairs of the built-in table (N 1.8 + N 1.8)
    Real r_cut = 0;
    for (int t1 = 0; t1 < MAX_VN_TYPES; t1++) for (int t2 = 0; t2 < MAX_VN_TYPES; t2++){
        if (F.atp.is_hydrogen[t1] || F.atp.is_hydrogen[t2] || !xs_h_bond_possible(t1, t2, F.atp)) continue;
        Real radii[MAX_VN_TYPES]; fill_vdw_radii(radii, {});
        r_cut = std::max(r_cut, radii[t1] + radii[t2]);
    }
    REQUIRE(pc.r_cut == r_cut + Vina::hbond_high_val());
    REQUIRE(pc.inv_cell == Real(1) / MAPS_PCAND_CELL_SIZE);
    const int ncells = pc.nx * pc.ny * pc.nz;
    int per_cell_max = 0;
    for (int c = 0; c < ncells; c++) per_cell_max = std::max(per_cell_max, pc_cs[c + 1] - pc_cs[c]);
    // every candidate is a g_pgrid atom, and every cell's list is in increasing g_pgrid position (the walk order)
    for (int c = 0; c < ncells; c++){
        int last = -1;
        for (int k = pc_cs[c]; k < pc_cs[c + 1]; k++){
            int pos = -1;
            for (int j = 0; j < pg.total_atoms; j++) if (pg_atoms[j].index == pc_atoms[k].index){ pos = j; break; }
            REQUIRE(pos >= 0);
            REQUIRE(std::memcmp(&pg_atoms[pos], &pc_atoms[k], sizeof(GridAtom)) == 0);
            REQUIRE(pos > last);
            last = pos;
        }
    }
    printf("[grid_maps e] g_pcand: r_cut %.2f A, cells %dx%dx%d, %d polar atoms, %d candidates (%.2f mean / %d max per cell)\n",
           pc.r_cut, pc.nx, pc.ny, pc.nz, pg.total_atoms, pc_cs[ncells], (double)pc_cs[ncells] / ncells, per_cell_max);

    // (1) host replica: same contributing atoms in the same order for random probes of every polar ligand type,
    //     over the polar bounding box widened by 6 A (probes beyond the candidate grid take the clamped cell)
    std::mt19937 rng(1008u);
    Real lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
    for (const GridAtom& a : pg_atoms){
        const Real q[3] = {a.x, a.y, a.z};
        for (int k = 0; k < 3; k++){ lo[k] = std::min(lo[k], q[k]); hi[k] = std::max(hi[k], q[k]); }
    }
    std::vector<int> polar_types;
    for (int t = 0; t < MAX_VN_TYPES; t++) if (!F.atp.is_hydrogen[t] && (F.atp.is_donor[t] || F.atp.is_acceptor[t])) polar_types.push_back(t);
    long long n_probe = 0, n_contrib = 0, vis_walk = 0, vis_cand = 0;
    for (int rep = 0; rep < 4000; rep++){
        Real p[3];
        for (int k = 0; k < 3; k++) p[k] = Real(lo[k] - 6 + std::uniform_real_distribution<double>(0, 1)(rng) * (hi[k] - lo[k] + 12));
        for (int t1 : polar_types){
            int nw = 0, nc = 0;
            const std::vector<int> w = walk_pairs(pg, pg_cs, pg_atoms, F.atp, t1, p, nw);
            const std::vector<int> c = cand_pairs(pc, pc_cs, pc_atoms, F.atp, t1, p, nc);
            INFO("type " << t1 << " probe " << p[0] << " " << p[1] << " " << p[2] << ": walk " << w.size() << " cand " << c.size());
            REQUIRE(w == c);
            n_probe++; n_contrib += (long long)w.size(); vis_walk += nw; vis_cand += nc;
        }
    }
    printf("[grid_maps e] host replica: %lld probes, %lld contributing pairs, identical sequences; atoms tested per probe: walk %.1f, candidates %.1f\n",
           n_probe, n_contrib, (double)vis_walk / n_probe, (double)vis_cand / n_probe);

    // (2) GPU: energy and gradient of the crystal pose and of 40 rigid perturbations, both caps, bit-identical
    //     with the candidate lists and with the walk (the same per-pair code over the same pair sequence)
    const FlexTopo& topo = F.host_topo.topo;
    int n_pose = 0, n_hb = 0;
    for (int rep = 0; rep < 41; rep++){
        std::vector<Real> coords; FlexPose pose{};
        Real axis[3] = {0, 0, 1}, shift[3] = {0, 0, 0}, angle = 0;
        if (rep > 0){
            std::normal_distribution<double> g(0, 1);
            double n2 = 0;
            for (int k = 0; k < 3; k++){ axis[k] = Real(g(rng)); n2 += axis[k] * axis[k]; }
            for (int k = 0; k < 3; k++){ axis[k] /= Real(std::sqrt(n2)); shift[k] = Real(std::uniform_real_distribution<double>(-1.5, 1.5)(rng)); }
            angle = Real(std::uniform_real_distribution<double>(-0.6, 0.6)(rng));
        }
        rigid_pose(F, axis, angle, shift, coords, pose);
        for (bool authentic : {false, true}){
            set_polar_cands_enabled(true);
            const Result on = eval(F, F.param, pose, topo, true, authentic);
            set_polar_cands_enabled(false);
            const Result off = eval(F, F.param, pose, topo, true, authentic);
            set_polar_cands_enabled(true);
            INFO("pose " << rep << (authentic ? " authentic" : " hunt") << ": E cand " << on.e << " walk " << off.e);
            REQUIRE(std::isfinite(on.e));
            REQUIRE(same_bits(on, off, topo.ntorsion));
            n_pose++;
        }
        // the pose forms at least one hydrogen bond when a polar ligand atom is within r_cut of a polar receptor atom
        for (int i = 0; i < F.lig.natom; i++){
            const int t = F.lig.vina_types[i];
            if (!(F.atp.is_donor[t] || F.atp.is_acceptor[t])) continue;
            if (F.min_rec_dist(coords.data() + i * 3, true) < pc.r_cut){ n_hb++; break; }
        }
    }
    printf("[grid_maps e] GPU: %d (pose, cap) evaluations bit-identical with candidate lists and with the walk; %d of 41 poses have a polar contact within r_cut\n", n_pose, n_hb);
    REQUIRE(n_hb > 0);
}


// Two free-rotor donors on one antecedent (gem-diol C(OH)2): hbond_angular_pair_warp mode 2 adds both lanes' cone
// gradients to the antecedent's s_force slot from inside the polar loop, so their order depends on the loop iteration
// at which each lane reaches its pairs, which the candidate lists change. Such ligands keep the g_pgrid walk
// (BondedHTopo::n_shared_cone_antecedent, cal_e_f_warp cand_pass); the energy never depends on that order.
TEST_CASE("grid maps: ligands with two free-rotor donors on one antecedent keep the polar walk", "[grid_maps]"){
    Tyk2Maps F;
    REQUIRE(F.built);
    PolarCands pc{}; std::vector<int> pc_cs; std::vector<GridAtom> pc_atoms;
    get_polar_cands(pc, pc_cs, pc_atoms);
    REQUIRE(pc.built);

    // pairs of receptor acceptors inside the docking box less than 4.5 A apart (carboxylates, backbone O next to a
    // side-chain O): the toy's carbon goes 1.5-2.5 A from their midpoint, so each hydroxyl can sit within the hbond
    // range of both acceptors with the acceptor on the carbon's side, where the cone angle falls inside the 90-140
    // deg ramp (that is where mode 2 adds to the antecedent)
    std::vector<std::pair<int, int>> acc_pairs;
    {
        std::vector<int> acceptors;
        for (int j = 0; j < F.rec.natom; j++){
            if (!F.atp.is_acceptor[F.rec.vina_types[j]]) continue;
            const Real* q = F.rec.coords.data() + j * 3;
            if (q[0] < F.dp.box.x_lo || q[0] > F.dp.box.x_hi || q[1] < F.dp.box.y_lo || q[1] > F.dp.box.y_hi
                || q[2] < F.dp.box.z_lo || q[2] > F.dp.box.z_hi) continue;
            acceptors.push_back(j);
        }
        for (size_t a = 0; a < acceptors.size(); a++) for (size_t b = a + 1; b < acceptors.size(); b++){
            const Real* p = F.rec.coords.data() + acceptors[a] * 3;
            const Real* q = F.rec.coords.data() + acceptors[b] * 3;
            const Real dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
            if (dx * dx + dy * dy + dz * dz < Real(4.5 * 4.5)) acc_pairs.emplace_back(acceptors[a], acceptors[b]);
        }
    }
    REQUIRE(acc_pairs.size() > 10);

    // gem-diol toy in its local frame: C at the origin, two O_D at 1.43 A, 109.5 deg apart
    const Real local[9] = {0, 0, 0, Real(1.43), 0, 0, Real(-0.477), Real(1.348), 0};
    const std::vector<int> types = {VN_TYPE_C_P, VN_TYPE_O_D, VN_TYPE_O_D};
    std::mt19937 rng(2026u);
    auto unit = [&](Real* u){
        std::normal_distribution<double> g(0, 1);
        double n2 = 0;
        for (int k = 0; k < 3; k++){ u[k] = Real(g(rng)); n2 += u[k] * u[k]; }
        for (int k = 0; k < 3; k++) u[k] /= Real(std::sqrt(n2));
    };
    auto place = [&](std::vector<Real>& coords){
        Real axis[3], u[3]; unit(axis); unit(u);
        const Real ang = Real(std::uniform_real_distribution<double>(-3.14159265358979, 3.14159265358979)(rng));
        const Real c = std::cos(ang), s = std::sin(ang), t = 1 - c;
        const Real R[3][3] = {{t * axis[0] * axis[0] + c, t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1]},
                              {t * axis[0] * axis[1] + s * axis[2], t * axis[1] * axis[1] + c, t * axis[1] * axis[2] - s * axis[0]},
                              {t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2] * axis[2] + c}};
        const auto& pr = acc_pairs[std::uniform_int_distribution<int>(0, (int)acc_pairs.size() - 1)(rng)];
        const Real d = Real(std::uniform_real_distribution<double>(1.5, 2.5)(rng));
        Real cc[3];
        for (int k = 0; k < 3; k++) cc[k] = Real(0.5) * (F.rec.coords[pr.first * 3 + k] + F.rec.coords[pr.second * 3 + k]) + d * u[k];
        coords.assign(9, 0);
        for (int a = 0; a < 3; a++){
            const Real* v = local + a * 3;
            for (int k = 0; k < 3; k++) coords[a * 3 + k] = R[k][0] * v[0] + R[k][1] * v[1] + R[k][2] * v[2] + cc[k];
        }
    };

    const int n_trial = 600;
    int n_grad_diff = 0, n_contact = 0;
    for (int rep = 0; rep < n_trial; rep++){
        std::vector<Real> coords; place(coords);
        RigidToy toy(types, coords);
        REQUIRE(toy.param.bonded_h_topo.n_shared_cone_antecedent == 1);
        REQUIRE(toy.param.bonded_h_topo.antecedent[1] == 0);
        REQUIRE(toy.param.bonded_h_topo.antecedent[2] == 0);
        // (1) the pass over the candidate lists (flag cleared) vs the walk: same energy, the gradient may differ
        toy.param.bonded_h_topo.n_shared_cone_antecedent = 0;
        set_polar_cands_enabled(true);
        const Result cand = eval(F, toy.param, toy.pose, toy.topo, true, false);
        set_polar_cands_enabled(false);
        const Result walk = eval(F, toy.param, toy.pose, toy.topo, true, false);
        INFO("trial " << rep << ": E cand " << cand.e << " walk " << walk.e);
        REQUIRE(std::memcmp(&cand.e, &walk.e, sizeof(Real)) == 0);
        if (!same_bits(cand, walk, 0)) n_grad_diff++;
        // (2) as built (flag set): the ligand walks g_pgrid whether or not the lists are enabled
        toy.param.bonded_h_topo.n_shared_cone_antecedent = 1;
        set_polar_cands_enabled(true);
        const Result kept = eval(F, toy.param, toy.pose, toy.topo, true, false);
        REQUIRE(same_bits(kept, walk, 0));
        bool contact = false;
        for (int a = 1; a < 3 && !contact; a++) contact = F.min_rec_dist(coords.data() + a * 3, true) < pc.r_cut;
        if (contact) n_contact++;
    }
    set_polar_cands_enabled(true);
    printf("[grid_maps f] gem-diol toy, %d placements (%d with a polar contact): lists vs walk with the flag cleared: energy identical in all, "
           "gradient differs in %d; with the flag set: bit-identical with the walk in all\n", n_trial, n_contact, n_grad_diff);
}
