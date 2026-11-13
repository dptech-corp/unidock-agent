//
// GPU energy / gradient kernel vs the CPU backend (the validated reference).
//   1. toy C-C pairs at several distances (curl, cutoff)
//   2. angular HBond, direction 1: ligand donor N-H ... receptor acceptor O
//   3. angular HBond, direction 2: receptor donor N-H ... ligand acceptor O
//   4. real system (tyk2 enriched DMS): energy, translation / rotation / torsion gradients,
//      with angular HBond on and off, plus finite-difference checks of the GPU gradient
//

#include <catch2/catch_amalgamated.hpp>
#include <cmath>
#include <string>
#include <vector>

#include "optimize/bfgs.h"
#include "cuda/common.cuh"
#include "cpu/common_cpu.h"
#include "cpu/score_cpu.h"
#include "cpu/spatial_grid.h"
#include "cpu/host_topo.h"
#include "format/dms_reader.h"
#include "cpu/mc_cpu.h"
#include "cpu/bfgs_cpu.h"
#include "cpu/random_cpu.h"
#include "score/terms.h"

static const std::string TYK2_DIR = std::string(TEST_DATA_DIR) + "ff_minimize/tyk2/";


// Upload the same constants to both backends and (re)build the CPU receptor grid.
static bool use_device_grid = true;  // toggled per test to cover both receptor loops on the GPU

// The bias settings of the last sync_backends: the CPU path reads them from FlexParamVina::terms.settings (which
// DockTask::alloc_device fills from DockParam), the kernel from the __constant__ mirror g_term_settings that
// init_constants uploads; eval_cpu / eval_gpu stamp them into the hand-built params so both backends see the
// synced DockParam.
static int s_sync_bias_type = BT_NO;
static Real s_sync_bias_k = Real(0.3);

static void sync_backends(const DockParam& dp, const AtomTypeProps& atp,
                          const FixMol& fix, const int* fix_types, const Real* fix_charges = nullptr){
    init_constants_cpu(dp, atp);
    init_constants(dp, atp);
    s_sync_bias_type = dp.bias_type;
    s_sync_bias_k = dp.bias_k;
    g_cpu_constants.USE_TORSION_PENALTY = false;  // not implemented on GPU; compare pure scoring
    free_spatial_grid(g_cpu_constants.receptor_grid);
    build_spatial_grid(g_cpu_constants.receptor_grid, fix.coords, fix_types, fix.natom,
                       g_cpu_constants.atp, g_cpu_constants.vdw_radii, fix_charges);
    if (use_device_grid) upload_receptor_grid(fix.coords, fix_types, fix.natom, atp, {}, fix_charges);
    else free_receptor_grid();
}

static DockParam large_box_param(bool angular){
    DockParam dp;
    dp.box.x_lo = -50; dp.box.x_hi = 50;
    dp.box.y_lo = -50; dp.box.y_hi = 50;
    dp.box.z_lo = -50; dp.box.z_hi = 50;
    dp.angular_hbond = angular;
    return dp;
}

// Flatten bias rows the way DockTask::alloc_device does (task_cpu.cpp, task.cu): inds[a*2], inds[a*2+1] are
// the half-open row range of atom a in Real slots, [0,0] means no bias; the rows of one atom are contiguous.
static void flatten_bias(const std::vector<Bias>& rows, int natom, std::vector<int>& inds, std::vector<Real>& params){
    inds.assign(natom * 2, 0);
    params.clear();
    int last_i = -1;
    for (const auto& b : rows){
        if (b.i != last_i) inds[b.i * 2] = (int)params.size();
        for (int k = 0; k < 5; k++) params.push_back(static_cast<Real>(b.param[k]));
        inds[b.i * 2 + 1] = (int)params.size();
        last_i = b.i;
    }
}

// One bias row on atom i at (atom + offset) with depth v_set and width slot w (r2 for BT_POS, unused by BT_ALIGN).
static Bias make_bias_row(int i, const Real* atom, double dx, double dy, double dz, float v_set, float w){
    Bias b;
    b.i = i;
    b.param[0] = float(atom[0] + dx); b.param[1] = float(atom[1] + dy); b.param[2] = float(atom[2] + dz);
    b.param[3] = v_set; b.param[4] = w;
    return b;
}

// Double-precision closed form of the bias energy (vina.h eval_ef_pos / eval_ef_zalign).
static double bias_energy_closed_form(const std::vector<Bias>& rows, const int* types, const Real* coords,
                                      BiasType bias_type, double bias_k){
    double e = 0;
    for (const auto& b : rows){
        const double dx = double(b.param[0]) - coords[b.i * 3];
        const double dy = double(b.param[1]) - coords[b.i * 3 + 1];
        const double dz = double(b.param[2]) - coords[b.i * 3 + 2];
        const double rr = dx * dx + dy * dy + dz * dz;
        const double a = double(b.param[3]) * bias_k;
        if (bias_type == BT_POS){
            e += a * std::exp(-rr / double(b.param[4]));
        } else if (bias_type == BT_ALIGN){
            const double k_a = types[b.i] == VN_TYPE_H ? 0.1 : 30.0;
            e += a * k_a / (1.0 + std::exp(3.0 * (1.0 - std::sqrt(rr))));
        }
    }
    return e;
}

// Puts bias_type back to BT_NO on both backends when a bias test case ends, also after a failed REQUIRE.
struct BiasOffGuard {
    DockParam dp;
    AtomTypeProps atp;
    ~BiasOffGuard(){ dp.bias_type = BT_NO; init_constants_cpu(dp, atp); init_constants(dp, atp); s_sync_bias_type = BT_NO; }
};

// A self-owned rigid ligand (no torsions) with all-pairs intra list.
struct ToyLigand {
    std::vector<int> types;
    std::vector<Real> coords;
    std::vector<int> pairs_intra;
    std::vector<Real> r1r2_intra;
    std::vector<int> inds_bias;
    std::vector<Real> params_bias;
    FlexTopo topo{};
    FlexParamVina param{};
    FlexPose pose{};

    void build(const std::vector<int>& t, const std::vector<Real>& c){
        types = t; coords = c;
        int natom = types.size();
        topo.natom = natom; topo.ntorsion = 0;
        topo.vn_types = types.data();
        pairs_intra.clear(); r1r2_intra.clear();
        for (int i = 0; i < natom; i++){
            for (int j = i + 1; j < natom; j++){
                pairs_intra.push_back(i); pairs_intra.push_back(j);
                r1r2_intra.push_back(VN_VDW_RADII[types[i]] + VN_VDW_RADII[types[j]]);
            }
        }
        inds_bias.assign(natom * 2, 0);
        param = FlexParamVina{};
        param.npair_intra = pairs_intra.size() / 2;
        param.pairs_intra = pairs_intra.data();
        param.r1_plus_r2_intra = r1r2_intra.data();
        param.atom_types = types.data();
        param.terms.bias.inds = inds_bias.data();
        build_bonded_h_topo(param.bonded_h_topo, coords.data(), types.data(), natom);
        pose = FlexPose{};
        pose.coords = coords.data();
        pose.dihedrals = nullptr;
        Real cx = 0, cy = 0, cz = 0;
        for (int i = 0; i < natom; i++){ cx += coords[i*3]; cy += coords[i*3+1]; cz += coords[i*3+2]; }
        pose.center[0] = cx / natom; pose.center[1] = cy / natom; pose.center[2] = cz / natom;
    }
    // terms.bias.params stays nullptr without rows, as in the tests that never set a bias
    void set_bias(const std::vector<Bias>& rows){
        flatten_bias(rows, topo.natom, inds_bias, params_bias);
        param.terms.bias.inds = inds_bias.data();
        param.terms.bias.params = params_bias.empty() ? nullptr : params_bias.data();
    }
    ~ToyLigand(){ if (param.bonded_h_topo.h_start) free_bonded_h_topo(param.bonded_h_topo); }
};

struct Result { Real e; FlexPoseGradient g; std::vector<Real> dihe_g; };

// param with the synced bias settings in its term set (what DockTask::alloc_device writes there)
static FlexParamVina with_synced_bias(const FlexParamVina& param){
    FlexParamVina p = param;
    p.terms.settings.bias_type = s_sync_bias_type;
    p.terms.settings.bias_k = s_sync_bias_k;
    return p;
}

static Result eval_cpu(const FixMol& fix, const FixParamVina& fp, const FlexParamVina& param,
                       const FlexPose& pose, const FlexTopo& topo){
    Result r; r.dihe_g.assign(std::max(topo.ntorsion, 1), 0); r.g.dihedrals_g = r.dihe_g.data();
    std::vector<Real> aux_f(topo.natom * 3, 0);
    const FlexParamVina p = with_synced_bias(param);
    r.e = cal_e_grad_cpu(&pose, &r.g, topo, fix, p, fp, aux_f.data());
    return r;
}

static Result eval_gpu(const FixMol& fix, const FixParamVina& fp, const FlexParamVina& param,
                       const FlexPose& pose, const FlexTopo& topo){
    Result r; r.dihe_g.assign(std::max(topo.ntorsion, 1), 0); r.g.dihedrals_g = r.dihe_g.data();
    const FlexParamVina p = with_synced_bias(param);
    cal_e_grad_one_gpu(fix, fp, p, pose, topo, &r.g, &r.e);
    return r;
}

static void require_same(const Result& a, const Result& b, int ntorsion, Real tol){
    REQUIRE_THAT(b.e, Catch::Matchers::WithinAbs(a.e, tol));
    for (int i = 0; i < 3; i++){
        REQUIRE_THAT(b.g.center_g[i], Catch::Matchers::WithinAbs(a.g.center_g[i], tol));
        REQUIRE_THAT(b.g.orientation_g[i], Catch::Matchers::WithinAbs(a.g.orientation_g[i], tol));
    }
    for (int i = 0; i < ntorsion; i++){
        REQUIRE_THAT(b.dihe_g[i], Catch::Matchers::WithinAbs(a.dihe_g[i], tol));
    }
}

// Finite-difference check of the GPU translation gradient by rigidly shifting the ligand.
static void fd_check_gpu_center(const FixMol& fix, const FixParamVina& fp, ToyLigand& lig, Real tol, bool central = false){
    Result r0 = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
    const Real dr = 1e-3;
    auto e_shifted = [&](int k, Real step){
        std::vector<Real> saved = lig.coords;
        for (int i = 0; i < lig.topo.natom; i++) lig.coords[i * 3 + k] += step;
        lig.pose.center[k] += step;
        Result r1 = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
        lig.coords = saved; lig.pose.center[k] -= step;
        return r1.e;
    };
    for (int k = 0; k < 3; k++){
        const Real fd = central ? (e_shifted(k, dr) - e_shifted(k, -dr)) / (2 * dr) : (e_shifted(k, dr) - r0.e) / dr;
        REQUIRE_THAT(r0.g.center_g[k], Catch::Matchers::WithinAbs(fd, tol));
    }
}


TEST_CASE("gpu vs cpu: C-C pair energies and gradients", "[gpu_vs_cpu]"){
    AtomTypeProps atp; init_default_atom_type_props(atp);
    std::vector<Real> rec_coords = {0, 0, 0};
    std::vector<int> rec_types = {VN_TYPE_C_H};
    FixMol fix; fix.natom = 1; fix.coords = rec_coords.data();
    FixParamVina fp; fp.atom_types = rec_types.data();
    sync_backends(large_box_param(true), atp, fix, rec_types.data());

    for (bool grid : {true, false}){
        use_device_grid = grid;
        sync_backends(large_box_param(true), atp, fix, rec_types.data());
        for (Real r : {Real(0.5), Real(1.0), Real(1.5), Real(2.5), Real(4.0), Real(7.9), Real(8.5)}){
            ToyLigand lig; lig.build({VN_TYPE_C_H}, {r, 0, 0});
            Result c = eval_cpu(fix, fp, lig.param, lig.pose, lig.topo);
            Result g = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
            INFO("grid = " << grid << " r = " << r << " cpu e = " << c.e << " gpu e = " << g.e);
            require_same(c, g, 0, 1e-4);
        }
    }
    use_device_grid = true;
    sync_backends(large_box_param(true), atp, fix, rec_types.data());

    // two-atom ligand: intra pair + rotation gradient
    ToyLigand lig; lig.build({VN_TYPE_C_H, VN_TYPE_C_H}, {1.0, 0, 0, 1.0, 1.0, 0});
    Result c = eval_cpu(fix, fp, lig.param, lig.pose, lig.topo);
    Result g = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
    require_same(c, g, 0, 1e-4);
    fd_check_gpu_center(fix, fp, lig, 0.02);
}


TEST_CASE("gpu vs cpu: angular hbond, ligand donor N-H ... receptor acceptor O", "[gpu_vs_cpu][hbond]"){
    AtomTypeProps atp; init_default_atom_type_props(atp);
    std::vector<Real> rec_coords = {2.9, 0.6, 0.0};
    std::vector<int> rec_types = {VN_TYPE_O_A};
    FixMol fix; fix.natom = 1; fix.coords = rec_coords.data();
    FixParamVina fp; fp.atom_types = rec_types.data();

    for (bool angular : {true, false}){
        sync_backends(large_box_param(angular), atp, fix, rec_types.data());
        // N at origin, H along +x, O off-axis: D-H...A angle ~ 150 degrees
        ToyLigand lig; lig.build({VN_TYPE_N_D, VN_TYPE_H}, {0, 0, 0, 1.01, 0, 0});
        REQUIRE(lig.param.bonded_h_topo.total_h == 1);
        Result c = eval_cpu(fix, fp, lig.param, lig.pose, lig.topo);
        Result g = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
        INFO("angular = " << angular << " cpu e = " << c.e << " gpu e = " << g.e);
        REQUIRE(c.e < 0);  // an hbond is formed
        require_same(c, g, 0, 1e-4);
        fd_check_gpu_center(fix, fp, lig, 0.02);
    }

    // The angular term must change the energy for a bent geometry
    sync_backends(large_box_param(true), atp, fix, rec_types.data());
    ToyLigand bent; bent.build({VN_TYPE_N_D, VN_TYPE_H}, {0, 0, 0, 0, 1.01, 0});  // H perpendicular
    Result g_bent = eval_gpu(fix, fp, bent.param, bent.pose, bent.topo);
    Result c_bent = eval_cpu(fix, fp, bent.param, bent.pose, bent.topo);
    require_same(c_bent, g_bent, 0, 1e-4);
    sync_backends(large_box_param(false), atp, fix, rec_types.data());
    Result g_bent_off = eval_gpu(fix, fp, bent.param, bent.pose, bent.topo);
    REQUIRE(g_bent.e > g_bent_off.e + 0.05);  // weaker hbond when angle is poor
}


TEST_CASE("gpu vs cpu: angular hbond, receptor donor N-H ... ligand acceptor O", "[gpu_vs_cpu][hbond]"){
    AtomTypeProps atp; init_default_atom_type_props(atp);
    // receptor: a single donor N at origin with one bonded H along +x (side table)
    std::vector<Real> rec_coords = {0, 0, 0};
    std::vector<int> rec_types = {VN_TYPE_N_D};
    std::vector<int> rh_start = {0, 1};
    std::vector<Real> rh_coords = {1.01, 0, 0};
    FixMol fix; fix.natom = 1; fix.coords = rec_coords.data();
    FixParamVina fp; fp.atom_types = rec_types.data();
    fp.rec_donor_h.natom = 1; fp.rec_donor_h.total_h = 1;
    fp.rec_donor_h.h_start = rh_start.data(); fp.rec_donor_h.h_coords = rh_coords.data();

    for (bool angular : {true, false}){
        sync_backends(large_box_param(angular), atp, fix, rec_types.data());
        g_cpu_constants.rec_donor_h_topo = fp.rec_donor_h;
        ToyLigand lig; lig.build({VN_TYPE_O_A}, {2.9, 0.6, 0.0});
        Result c = eval_cpu(fix, fp, lig.param, lig.pose, lig.topo);
        Result g = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
        INFO("angular = " << angular << " cpu e = " << c.e << " gpu e = " << g.e);
        REQUIRE(c.e < 0);
        require_same(c, g, 0, 1e-4);
        fd_check_gpu_center(fix, fp, lig, 0.02);
        g_cpu_constants.rec_donor_h_topo = {};
    }
}


TEST_CASE("gpu vs cpu: angular hbond, ligand hydroxyl (cone hydrogen) ... receptor acceptor O", "[gpu_vs_cpu][hbond]"){
    AtomTypeProps atp; init_default_atom_type_props(atp);
    // acceptor placed so that the best cone hydrogen gives a mid-ramp (~120 deg) angle
    std::vector<Real> rec_coords = {2.4, 1.6, 0.4};
    std::vector<int> rec_types = {VN_TYPE_O_A};
    FixMol fix; fix.natom = 1; fix.coords = rec_coords.data();
    FixParamVina fp; fp.atom_types = rec_types.data();

    Result e_on, e_off;
    for (bool angular : {true, false}){
        sync_backends(large_box_param(angular), atp, fix, rec_types.data());
        // O_DA at origin, prepared H on the far side of the cone, C antecedent on -x
        ToyLigand lig; lig.build({VN_TYPE_O_DA, VN_TYPE_H, VN_TYPE_C_H},
                                 {0, 0, 0,  0.3338, -0.9426, 0,  -1.43, 0, 0});
        REQUIRE(lig.param.bonded_h_topo.antecedent[0] == 2);
        Result c = eval_cpu(fix, fp, lig.param, lig.pose, lig.topo);
        Result g = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
        INFO("angular = " << angular << " cpu e = " << c.e << " gpu e = " << g.e);
        require_same(c, g, 0, 1e-4);
        fd_check_gpu_center(fix, fp, lig, 0.02);
        (angular ? e_on : e_off) = g;
    }
    // mid-ramp: partial credit, so the angular term changes the energy but keeps some hbond
    REQUIRE(e_on.e > e_off.e + 0.02);
    REQUIRE(e_on.e < e_off.e + 0.40);
}


TEST_CASE("gpu vs cpu: angular hbond, receptor hydroxyl (cone hydrogen) ... ligand acceptor O", "[gpu_vs_cpu][hbond]"){
    AtomTypeProps atp; init_default_atom_type_props(atp);
    // receptor: Ser-like O_DA at origin, C antecedent on -x, one prepared H pointing away
    std::vector<Real> rec_coords = {0, 0, 0,  -1.43, 0, 0};
    std::vector<int> rec_types = {VN_TYPE_O_DA, VN_TYPE_C_H};
    std::vector<int> rh_start = {0, 1, 1};
    std::vector<Real> rh_coords = {0.3338, -0.9426, 0};
    std::vector<int> rh_ante = {1, -1};
    FixMol fix; fix.natom = 2; fix.coords = rec_coords.data();
    FixParamVina fp; fp.atom_types = rec_types.data();
    fp.rec_donor_h.natom = 2; fp.rec_donor_h.total_h = 1;
    fp.rec_donor_h.h_start = rh_start.data(); fp.rec_donor_h.h_coords = rh_coords.data();
    fp.rec_donor_h.antecedent = rh_ante.data();

    Result e_on, e_off;
    for (bool angular : {true, false}){
        sync_backends(large_box_param(angular), atp, fix, rec_types.data());
        g_cpu_constants.rec_donor_h_topo = fp.rec_donor_h;
        ToyLigand lig; lig.build({VN_TYPE_O_A}, {2.4, 1.6, 0.4});
        Result c = eval_cpu(fix, fp, lig.param, lig.pose, lig.topo);
        Result g = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
        INFO("angular = " << angular << " cpu e = " << c.e << " gpu e = " << g.e);
        require_same(c, g, 0, 1e-4);
        fd_check_gpu_center(fix, fp, lig, 0.02);
        (angular ? e_on : e_off) = g;
        g_cpu_constants.rec_donor_h_topo = {};
    }
    REQUIRE(e_on.e > e_off.e + 0.02);
    REQUIRE(e_on.e < e_off.e + 0.40);
}


TEST_CASE("gpu vs cpu: real system tyk2 (enriched DMS)", "[gpu_vs_cpu][real]"){
    AtomTypeProps atp; init_default_atom_type_props(atp);

    UDFlexMol lig; FFMinData ff_lig;
    read_ligand_dms_full(TYK2_DIR + "ligand_ejm42_enriched.dms", lig, ff_lig, atp, false);
    REQUIRE(lig.natom > 10);
    REQUIRE(lig.torsions.size() > 0);

    DockParam dp;
    const Real half = 15, cutoff = 8;
    dp.box.x_lo = lig.center[0] - half; dp.box.x_hi = lig.center[0] + half;
    dp.box.y_lo = lig.center[1] - half; dp.box.y_hi = lig.center[1] + half;
    dp.box.z_lo = lig.center[2] - half; dp.box.z_hi = lig.center[2] + half;
    Box box_protein = dp.box;
    box_protein.x_lo -= cutoff; box_protein.x_hi += cutoff;
    box_protein.y_lo -= cutoff; box_protein.y_hi += cutoff;
    box_protein.z_lo -= cutoff; box_protein.z_hi += cutoff;

    UDFixMol rec; FFMinData ff_rec;
    read_receptor_dms_full(TYK2_DIR + "receptor_enriched.dms", box_protein, rec, ff_rec, atp);
    REQUIRE(rec.natom > 100);
    REQUIRE(rec.rec_donor_h.natom == rec.natom);

    FixMol fix; fix.natom = rec.natom; fix.coords = rec.coords.data();
    FixParamVina fp; fp.atom_types = rec.vina_types.data(); fp.rec_donor_h = rec.rec_donor_h;

    HostFlexTopo host_topo; build_host_flex_topo(lig, host_topo);
    const FlexTopo& topo = host_topo.topo;

    std::vector<Real> r1r2;
    for (size_t j = 0; j < lig.intra_pairs.size(); j += 2){
        r1r2.push_back(VN_VDW_RADII[lig.vina_types[lig.intra_pairs[j]]] + VN_VDW_RADII[lig.vina_types[lig.intra_pairs[j + 1]]]);
    }
    std::vector<int> inds_bias(lig.natom * 2, 0);
    FlexParamVina param{};
    param.npair_intra = lig.intra_pairs.size() / 2;
    param.pairs_intra = lig.intra_pairs.data();
    param.r1_plus_r2_intra = r1r2.data();
    param.atom_types = lig.vina_types.data();
    param.terms.bias.inds = inds_bias.data();

    std::vector<Real> coords = lig.coords, dihedrals = lig.dihedrals;
    FlexPose pose{}; pose.coords = coords.data(); pose.dihedrals = dihedrals.data();
    pose.center[0] = lig.center[0]; pose.center[1] = lig.center[1]; pose.center[2] = lig.center[2];

    // variant 3: united-atom Coulomb term on (ligand charged groups scaled), grid on
    REQUIRE(lig.charges_ua.size() == (size_t)lig.natom);
    REQUIRE(rec.charges_ua.size() == (size_t)rec.natom);
    std::vector<Real> lig_q_eff(lig.natom);
    Real e_no_elec = 0;
    for (int variant = 0; variant < 4; variant++){
        const bool angular = (variant != 1);
        const bool elec = (variant == 3);
        use_device_grid = (variant != 2);
        dp.angular_hbond = angular;
        dp.weights.elec = elec ? Real(0.114) : Real(0);
        dp.weights.elec_charged_scale = Real(0.57);
        for (int k = 0; k < lig.natom; k++) lig_q_eff[k] = lig.charges_ua[k] * (lig.charged_group[k] ? dp.weights.elec_charged_scale : Real(1));
        param.charges = elec ? lig_q_eff.data() : nullptr;
        fp.charges = elec ? rec.charges_ua.data() : nullptr;
        sync_backends(dp, atp, fix, rec.vina_types.data(), fp.charges);
        g_cpu_constants.rec_donor_h_topo = rec.rec_donor_h;
        if (param.bonded_h_topo.h_start) free_bonded_h_topo(param.bonded_h_topo);
        build_bonded_h_topo(param.bonded_h_topo, coords.data(), lig.vina_types.data(), lig.natom);

        Result c = eval_cpu(fix, fp, param, pose, topo);
        Result g = eval_gpu(fix, fp, param, pose, topo);
        INFO("angular = " << angular << " natom = " << lig.natom << " ntor = " << topo.ntorsion
             << " rec atoms = " << rec.natom << " cpu e = " << c.e << " gpu e = " << g.e);
        Real tol = Real(2e-3) * std::max(Real(1), std::fabs(c.e));
        {
            Real gmax = 0, dmax = 0;
            for (int i = 0; i < 3; i++){ gmax = std::max(gmax, std::fabs(c.g.center_g[i])); dmax = std::max(dmax, std::fabs(c.g.center_g[i] - g.g.center_g[i]));
                                        gmax = std::max(gmax, std::fabs(c.g.orientation_g[i])); dmax = std::max(dmax, std::fabs(c.g.orientation_g[i] - g.g.orientation_g[i])); }
            for (int i = 0; i < topo.ntorsion; i++){ gmax = std::max(gmax, std::fabs(c.dihe_g[i])); dmax = std::max(dmax, std::fabs(c.dihe_g[i] - g.dihe_g[i])); }
            printf("[tyk2 angular=%d grid=%d elec=%d] energy cpu %.5f gpu %.5f |diff| %.2e | max |grad| %.3f, max |grad diff| %.2e\n",
                   (int)angular, (int)use_device_grid, (int)elec, c.e, g.e, std::fabs(c.e - g.e), gmax, dmax);
        }
        if (variant == 0) e_no_elec = c.e;
        if (elec) REQUIRE(std::fabs(c.e - e_no_elec) > Real(0.01));  // the Coulomb term changes the energy

        // Central finite differences of the torsion gradient (CPU dihedral update on copies).
        // Two step sizes: convergence toward the analytic value distinguishes curvature from a
        // wrong gradient. Only the converged (smaller) step is asserted.
        auto e_gpu_at = [&](int i_tor, Real dtheta){
            std::vector<Real> c2 = coords, d2 = dihedrals;
            FlexPose p2 = pose; p2.coords = c2.data(); p2.dihedrals = d2.data();
            apply_grad_update_dihe_cpu(&p2, &topo, i_tor, dtheta, true);
            return eval_gpu(fix, fp, param, p2, topo).e;
        };
        for (int i_tor = 0; i_tor < topo.ntorsion; i_tor++){
            const Real h1 = 1e-3, h2 = 2e-4;
            Real fd1 = (e_gpu_at(i_tor, h1) - e_gpu_at(i_tor, -h1)) / (2 * h1);
            Real fd2 = (e_gpu_at(i_tor, h2) - e_gpu_at(i_tor, -h2)) / (2 * h2);
            printf("[tyk2 angular=%d grid=%d] torsion %d: axis(%d->%d) rotated %d atoms | cpu grad %.4f | gpu grad %.4f | central fd h=1e-3 %.4f h=2e-4 %.4f\n",
                   (int)angular, (int)use_device_grid, i_tor, topo.axis_atoms[i_tor*2], topo.axis_atoms[i_tor*2+1],
                   topo.rotated_inds[i_tor*2+1] - topo.rotated_inds[i_tor*2] + 1,
                   c.dihe_g[i_tor], g.dihe_g[i_tor], fd1, fd2);
            CHECK_THAT(g.dihe_g[i_tor], Catch::Matchers::WithinAbs(fd2, Real(0.05) * std::max(Real(1), std::fabs(fd2))));
        }
        require_same(c, g, topo.ntorsion, tol);
        g_cpu_constants.rec_donor_h_topo = {};
    }
    free_bonded_h_topo(param.bonded_h_topo);
    free_spatial_grid(g_cpu_constants.receptor_grid);
    free_receptor_grid();
    use_device_grid = true;
}


// Same random starts, both optimizers: the search quality gap seen in docking runs must be
// explained either here (local optimization from clashing starts) or in the MC loop.
TEST_CASE("gpu vs cpu: BFGS from randomized starts, real system tyk2", "[gpu_vs_cpu][bfgs_random]"){
    AtomTypeProps atp; init_default_atom_type_props(atp);
    UDFlexMol lig; FFMinData ff_lig;
    read_ligand_dms_full(TYK2_DIR + "ligand_ejm42_enriched.dms", lig, ff_lig, atp, false);
    DockParam dp;
    const Real half = 15, cutoff = 8;
    dp.box.x_lo = lig.center[0] - half; dp.box.x_hi = lig.center[0] + half;
    dp.box.y_lo = lig.center[1] - half; dp.box.y_hi = lig.center[1] + half;
    dp.box.z_lo = lig.center[2] - half; dp.box.z_hi = lig.center[2] + half;
    Box box_protein = dp.box;
    box_protein.x_lo -= cutoff; box_protein.x_hi += cutoff; box_protein.y_lo -= cutoff; box_protein.y_hi += cutoff;
    box_protein.z_lo -= cutoff; box_protein.z_hi += cutoff;
    UDFixMol rec; FFMinData ff_rec;
    read_receptor_dms_full(TYK2_DIR + "receptor_enriched.dms", box_protein, rec, ff_rec, atp);
    FixMol fix; fix.natom = rec.natom; fix.coords = rec.coords.data();
    FixParamVina fp; fp.atom_types = rec.vina_types.data(); fp.rec_donor_h = rec.rec_donor_h;
    HostFlexTopo host_topo; build_host_flex_topo(lig, host_topo);
    const FlexTopo& topo = host_topo.topo;
    std::vector<Real> r1r2;
    for (size_t j = 0; j < lig.intra_pairs.size(); j += 2)
        r1r2.push_back(VN_VDW_RADII[lig.vina_types[lig.intra_pairs[j]]] + VN_VDW_RADII[lig.vina_types[lig.intra_pairs[j + 1]]]);
    std::vector<int> inds_bias(lig.natom * 2, 0);
    FlexParamVina param{};
    param.npair_intra = lig.intra_pairs.size() / 2; param.pairs_intra = lig.intra_pairs.data();
    param.r1_plus_r2_intra = r1r2.data(); param.atom_types = lig.vina_types.data(); param.terms.bias.inds = inds_bias.data();
    build_bonded_h_topo(param.bonded_h_topo, lig.coords.data(), lig.vina_types.data(), lig.natom);

    use_device_grid = true;
    sync_backends(dp, atp, fix, rec.vina_types.data());
    g_cpu_constants.USE_TORSION_PENALTY = true;   // as in production CPU runs
    g_cpu_constants.rec_donor_h_topo = rec.rec_donor_h;

    FlexPose ref{}; std::vector<Real> c0 = lig.coords, d0 = lig.dihedrals;
    ref.coords = c0.data(); ref.dihedrals = d0.data();
    ref.center[0] = lig.center[0]; ref.center[1] = lig.center[1]; ref.center[2] = lig.center[2];

    const int N = 64, steps = 15;
    BFGSThreadMemory mem;
    int n_cpu_better = 0, n_gpu_better = 0, n_same = 0;
    double sum_cpu = 0, sum_gpu = 0;
    for (int i = 0; i < N; i++){
        CPURandom rng(1234 + i * 1000);
        std::vector<Real> cs(lig.natom * 3), ds(std::max(topo.ntorsion, 1));
        FlexPose start{}; start.coords = cs.data(); start.dihedrals = ds.data();
        randomize_pose_cpu(&start, &ref, topo, rng, i, N);

        std::vector<Real> cc = cs, dc = ds; FlexPose xc = start; xc.coords = cc.data(); xc.dihedrals = dc.data();
        std::vector<Real> cg_ = cs, dg_ = ds; FlexPose xg = start; xg.coords = cg_.data(); xg.dihedrals = dg_.data();
        Result e0 = eval_cpu(fix, fp, param, start, topo);
        bfgs_with_memory_cpu(&xc, topo, fix, param, fp, steps, mem);
        bfgs_one_gpu(fix, fp, param, &xg, topo, steps);
        // rescore both results with the CPU scorer so the comparison is apples to apples
        Real ec = eval_cpu(fix, fp, param, xc, topo).e, eg = eval_cpu(fix, fp, param, xg, topo).e;
        sum_cpu += ec; sum_gpu += eg;
        if (std::fabs(ec - eg) < 1e-2) n_same++; else if (ec < eg) n_cpu_better++; else n_gpu_better++;
        if (i < 12 || std::fabs(ec - eg) > 1)
            printf("[bfgs_random] start %2d: E0 %12.2f -> cpu %10.3f (reported %10.3f) | gpu %10.3f (reported %10.3f)\n",
                   i, e0.e, ec, xc.energy, eg, xg.energy);
    }
    printf("[bfgs_random] %d starts, %d steps: same %d | cpu lower %d | gpu lower %d | mean E cpu %.3f gpu %.3f\n",
           N, steps, n_same, n_cpu_better, n_gpu_better, sum_cpu / N, sum_gpu / N);
    free_bonded_h_topo(param.bonded_h_topo);
    free_spatial_grid(g_cpu_constants.receptor_grid);
    free_receptor_grid();
    g_cpu_constants.rec_donor_h_topo = {};
    SUCCEED();
}


static const char* bias_name(BiasType bt){ return bt == BT_ALIGN ? "align" : (bt == BT_POS ? "pos" : "no"); }

static Real max_grad_dev(const Result& a, const Result& b, int ntorsion){
    Real d = 0;
    for (int i = 0; i < 3; i++){
        d = std::max(d, std::fabs(a.g.center_g[i] - b.g.center_g[i]));
        d = std::max(d, std::fabs(a.g.orientation_g[i] - b.g.orientation_g[i]));
    }
    for (int i = 0; i < ntorsion; i++) d = std::max(d, std::fabs(a.dihe_g[i] - b.dihe_g[i]));
    return d;
}


// Position bias on a toy ligand: two rows on atom 0, none on atom 1 (gap), one on N, one on the H.
TEST_CASE("gpu vs cpu: position bias toy", "[gpu_vs_cpu][bias]"){
    AtomTypeProps atp; init_default_atom_type_props(atp);
    // receptor: 4 C atoms more than 8 A from every ligand atom (inter = 0), and at least as many atoms as
    // the ligand because cal_e_grad_one_gpu sizes aux_f by the receptor natom
    std::vector<Real> rec_coords = {20, 20, 20,  24, 20, 20,  20, 24, 20,  20, 20, 24};
    std::vector<int> rec_types(4, VN_TYPE_C_H);
    FixMol fix; fix.natom = 4; fix.coords = rec_coords.data();
    FixParamVina fp; fp.atom_types = rec_types.data();
    BiasOffGuard guard{large_box_param(true), atp};

    ToyLigand lig; lig.build({VN_TYPE_C_H, VN_TYPE_C_H, VN_TYPE_N_A, VN_TYPE_H},
                             {0, 0, 0,  3.2, 0.4, 0,  0.3, 3.1, 0.5,  0.3, 4.11, 0.5});
    const Real* x = lig.coords.data();
    const std::vector<Bias> rows = {
        make_bias_row(0, x,     0.6, 0.8, 0.0,       1.0f, 2.0f),   // r = 1.00
        make_bias_row(0, x,     0.0, 0.948, -1.264,  2.5f, 2.0f),   // r = 1.58
        make_bias_row(2, x + 6, 0.7, -0.4, 0.9,      3.0f, 2.0f),   // r = 1.208
        make_bias_row(3, x + 9, 1.0, 0.5, -1.0,      4.0f, 2.0f),   // r = 1.50, hydrogen
    };

    for (BiasType bt : {BT_ALIGN, BT_POS}){
        for (Real k : {Real(0.05), Real(0.3)}){
            DockParam dp = large_box_param(true);
            dp.bias_k = k;
            dp.bias_type = BT_NO;
            sync_backends(dp, atp, fix, rec_types.data());
            lig.set_bias({});
            Result g_zero = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
            lig.set_bias(rows);
            Result c_off = eval_cpu(fix, fp, lig.param, lig.pose, lig.topo);
            Result g_off = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);

            dp.bias_type = bt;
            sync_backends(dp, atp, fix, rec_types.data());
            Result c_on = eval_cpu(fix, fp, lig.param, lig.pose, lig.topo);
            Result g_on = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);

            const double e_ref = bias_energy_closed_form(rows, lig.types.data(), lig.coords.data(), bt, double(k));
            const double de_cpu = double(c_on.e) - double(c_off.e), de_gpu = double(g_on.e) - double(g_off.e);
            printf("[bias toy %s k=%.2f] E_off cpu %.6f gpu %.6f (no rows %.6f) | E_on cpu %.6f gpu %.6f max |grad diff| %.2e"
                   " | dE cpu %.6f gpu %.6f closed form %.6f rel dev cpu %.2e gpu %.2e\n",
                   bias_name(bt), (double)k, c_off.e, g_off.e, g_zero.e, c_on.e, g_on.e, max_grad_dev(c_on, g_on, 0),
                   de_cpu, de_gpu, e_ref, std::fabs(de_cpu - e_ref) / std::fabs(e_ref), std::fabs(de_gpu - e_ref) / std::fabs(e_ref));
            INFO("bias " << bias_name(bt) << " k " << k);

            // BT_NO with rows present gives the same GPU result as no rows
            REQUIRE_THAT(g_off.e, Catch::Matchers::WithinAbs(g_zero.e, 1e-6));
            for (int d = 0; d < 3; d++) REQUIRE_THAT(g_off.g.center_g[d], Catch::Matchers::WithinAbs(g_zero.g.center_g[d], 1e-6));
            require_same(c_on, g_on, 0, Real(1e-4) * std::max(Real(1), std::fabs(c_on.e)));
            REQUIRE_THAT(de_cpu, Catch::Matchers::WithinAbs(e_ref, 1e-4 * std::fabs(e_ref)));
            REQUIRE_THAT(de_gpu, Catch::Matchers::WithinAbs(e_ref, 1e-4 * std::fabs(e_ref)));
            if (k == Real(0.3)) fd_check_gpu_center(fix, fp, lig, 0.02);
        }
    }
}


// The TermSet route (plan step 5): the CPU path reads the bias type / k from FlexParamVina::terms.settings itself,
// the CUDA kernel from the __constant__ mirror g_term_settings init_constants uploads from the same DockParam. A
// biased pose evaluated through a term set filled the way DockTask::alloc_device fills it agrees with the GPU, and
// a term set left at BT_NO on the CPU is the unbiased energy whatever the device mirror says.
TEST_CASE("gpu vs cpu: biased pose through FlexParamVina::terms (TermSet route)", "[gpu_vs_cpu][bias][terms]"){
    AtomTypeProps atp; init_default_atom_type_props(atp);
    std::vector<Real> rec_coords = {20, 20, 20,  24, 20, 20,  20, 24, 20,  20, 20, 24};
    std::vector<int> rec_types(4, VN_TYPE_C_H);
    FixMol fix; fix.natom = 4; fix.coords = rec_coords.data();
    FixParamVina fp; fp.atom_types = rec_types.data();
    BiasOffGuard guard{large_box_param(true), atp};

    ToyLigand lig; lig.build({VN_TYPE_C_H, VN_TYPE_N_A, VN_TYPE_H}, {0, 0, 0,  3.1, 0.4, 0,  3.3, 1.4, 0.3});
    const Real* x = lig.coords.data();
    const std::vector<Bias> rows = {
        make_bias_row(0, x,     0.6, 0.8, 0.0,   1.0f, 2.0f),
        make_bias_row(1, x + 3, 0.7, -0.4, 0.9,  3.0f, 2.0f),
        make_bias_row(2, x + 6, 1.0, 0.5, -1.0,  4.0f, 2.0f),   // hydrogen
    };
    lig.set_bias(rows);
    auto eval_cpu_term = [&](const FlexParamVina& param){   // no stamping: the term as given
        Result r; r.dihe_g.assign(1, 0); r.g.dihedrals_g = r.dihe_g.data();
        std::vector<Real> aux_f(lig.topo.natom * 3, 0);
        r.e = cal_e_grad_cpu(&lig.pose, &r.g, lig.topo, fix, param, fp, aux_f.data());
        return r;
    };

    DockParam dp = large_box_param(true);
    dp.bias_type = BT_NO;
    sync_backends(dp, atp, fix, rec_types.data());
    const Result c_off = eval_cpu(fix, fp, lig.param, lig.pose, lig.topo);
    const Result g_off = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
    require_same(c_off, g_off, 0, Real(1e-4) * std::max(Real(1), std::fabs(c_off.e)));

    for (BiasType bt : {BT_ALIGN, BT_POS}){
        const Real k = Real(0.3);
        dp.bias_type = bt; dp.bias_k = k;
        sync_backends(dp, atp, fix, rec_types.data());    // device mirrors on

        FlexParamVina term_param = lig.param;              // the term set as DockTask::alloc_device fills it
        term_param.terms.settings.bias_type = bt; term_param.terms.settings.bias_k = k;
        const Result c_term = eval_cpu_term(term_param);
        const Result g_on = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);

        FlexParamVina no_param = lig.param;                // the term off: the CPU ignores the device mirror
        no_param.terms.settings.bias_type = BT_NO;
        const Result c_no = eval_cpu_term(no_param);

        const double e_ref = bias_energy_closed_form(rows, lig.types.data(), lig.coords.data(), bt, double(k));
        printf("[terms route %s] E_off cpu %.6f | E_on term %.6f gpu %.6f | term off under device mirror %.6f | closed-form dE %.6f\n",
               bias_name(bt), c_off.e, c_term.e, g_on.e, c_no.e, e_ref);
        INFO("bias " << bias_name(bt));
        REQUIRE_THAT(c_no.e, Catch::Matchers::WithinAbs(c_off.e, 1e-6));
        for (int d = 0; d < 3; d++) REQUIRE_THAT(c_no.g.center_g[d], Catch::Matchers::WithinAbs(c_off.g.center_g[d], 1e-6));
        require_same(c_term, g_on, 0, Real(1e-4) * std::max(Real(1), std::fabs(c_term.e)));
        REQUIRE_THAT(double(c_term.e) - double(c_off.e), Catch::Matchers::WithinAbs(e_ref, 1e-4 * std::fabs(e_ref)));
    }
}


// Finite-difference force of one bias row on a one-atom heavy and a one-atom hydrogen ligand
// (center_g is dE/dx of that atom), rows at r = 0.3 / 1.0 / 2.0 A along a non-axial direction.
TEST_CASE("gpu vs cpu: position bias finite-difference force, one-atom ligands", "[gpu_vs_cpu][bias]"){
    AtomTypeProps atp; init_default_atom_type_props(atp);
    std::vector<Real> rec_coords = {25, 25, 25};
    std::vector<int> rec_types = {VN_TYPE_C_H};
    FixMol fix; fix.natom = 1; fix.coords = rec_coords.data();
    FixParamVina fp; fp.atom_types = rec_types.data();
    BiasOffGuard guard{large_box_param(true), atp};

    const Real bias_k = 0.3;
    const float v_set = 1.0f;
    const double u[3] = {1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0};
    for (int type : {VN_TYPE_C_H, VN_TYPE_H}){
        for (BiasType bt : {BT_ALIGN, BT_POS}){
            DockParam dp = large_box_param(true);
            dp.bias_type = bt;
            dp.bias_k = bias_k;
            sync_backends(dp, atp, fix, rec_types.data());
            for (double r : {0.3, 1.0, 2.0}){
                ToyLigand lig; lig.build({type}, {2.0, -1.0, 3.0});
                lig.set_bias({make_bias_row(0, lig.coords.data(), r * u[0], r * u[1], r * u[2], v_set, 2.0f)});
                Result c = eval_cpu(fix, fp, lig.param, lig.pose, lig.topo);
                Result g = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
                const Real dr = 1e-3;
                Real fd[3];
                for (int d = 0; d < 3; d++){
                    std::vector<Real> saved = lig.coords;
                    lig.coords[d] += dr; lig.pose.center[d] += dr;
                    Result g1 = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
                    lig.coords = saved; lig.pose.center[d] -= dr;
                    fd[d] = (g1.e - g.e) / dr;
                }
                const double gnorm = std::sqrt(double(g.g.center_g[0]) * g.g.center_g[0] + double(g.g.center_g[1]) * g.g.center_g[1]
                                               + double(g.g.center_g[2]) * g.g.center_g[2]);
                printf("[bias fd %s %s r=%.1f] E cpu %.6f gpu %.6f | grad cpu (%.5f %.5f %.5f) gpu (%.5f %.5f %.5f) fd (%.5f %.5f %.5f) | |g| %.5f\n",
                       type == VN_TYPE_H ? "H" : "C", bias_name(bt), r, c.e, g.e,
                       c.g.center_g[0], c.g.center_g[1], c.g.center_g[2], g.g.center_g[0], g.g.center_g[1], g.g.center_g[2],
                       fd[0], fd[1], fd[2], gnorm);
                INFO("type " << type << " bias " << bias_name(bt) << " r " << r);
                REQUIRE_THAT(g.e, Catch::Matchers::WithinAbs(c.e, 1e-4 * std::max(Real(1), std::fabs(c.e))));
                for (int d = 0; d < 3; d++){
                    REQUIRE_THAT(g.g.center_g[d], Catch::Matchers::WithinAbs(c.g.center_g[d], 1e-4 * std::max(Real(1), std::fabs(c.g.center_g[d]))));
                    REQUIRE_THAT(g.g.center_g[d], Catch::Matchers::WithinAbs(fd[d], 1e-2 * std::max(Real(1), std::fabs(g.g.center_g[d]))));
                }
                if (bt == BT_ALIGN && r == 1.0){
                    // inflection of the sigmoid: |dE/dr| = 0.75 * a * k_a
                    const double peak = 0.75 * double(v_set) * double(bias_k) * (type == VN_TYPE_H ? 0.1 : 30.0);
                    REQUIRE_THAT(gnorm, Catch::Matchers::WithinAbs(peak, 1e-3 * peak));
                }
            }
        }
    }
}


// Real system with position bias (tyk2, angular HBond, device grid): parity, torsion finite differences,
// and the bias-only differential (E_on - E_off, g_on - g_off) CPU vs GPU, which the absolute parity
// tolerance cannot resolve row by row.
TEST_CASE("gpu vs cpu: real system tyk2 with position bias", "[gpu_vs_cpu][real][bias]"){
    AtomTypeProps atp; init_default_atom_type_props(atp);

    UDFlexMol lig; FFMinData ff_lig;
    read_ligand_dms_full(TYK2_DIR + "ligand_ejm42_enriched.dms", lig, ff_lig, atp, false);
    REQUIRE(lig.natom > 10);
    REQUIRE(lig.torsions.size() > 0);

    DockParam dp;
    const Real half = 15, cutoff = 8;
    dp.box.x_lo = lig.center[0] - half; dp.box.x_hi = lig.center[0] + half;
    dp.box.y_lo = lig.center[1] - half; dp.box.y_hi = lig.center[1] + half;
    dp.box.z_lo = lig.center[2] - half; dp.box.z_hi = lig.center[2] + half;
    Box box_protein = dp.box;
    box_protein.x_lo -= cutoff; box_protein.x_hi += cutoff;
    box_protein.y_lo -= cutoff; box_protein.y_hi += cutoff;
    box_protein.z_lo -= cutoff; box_protein.z_hi += cutoff;

    UDFixMol rec; FFMinData ff_rec;
    read_receptor_dms_full(TYK2_DIR + "receptor_enriched.dms", box_protein, rec, ff_rec, atp);
    REQUIRE(rec.natom > 100);
    REQUIRE(rec.rec_donor_h.natom == rec.natom);

    FixMol fix; fix.natom = rec.natom; fix.coords = rec.coords.data();
    FixParamVina fp; fp.atom_types = rec.vina_types.data(); fp.rec_donor_h = rec.rec_donor_h;

    HostFlexTopo host_topo; build_host_flex_topo(lig, host_topo);
    const FlexTopo& topo = host_topo.topo;

    std::vector<Real> r1r2;
    for (size_t j = 0; j < lig.intra_pairs.size(); j += 2){
        r1r2.push_back(VN_VDW_RADII[lig.vina_types[lig.intra_pairs[j]]] + VN_VDW_RADII[lig.vina_types[lig.intra_pairs[j + 1]]]);
    }
    std::vector<int> inds_bias;
    std::vector<Real> params_bias;
    FlexParamVina param{};
    param.npair_intra = lig.intra_pairs.size() / 2;
    param.pairs_intra = lig.intra_pairs.data();
    param.r1_plus_r2_intra = r1r2.data();
    param.atom_types = lig.vina_types.data();
    auto install = [&](const std::vector<Bias>& rows){
        flatten_bias(rows, lig.natom, inds_bias, params_bias);
        param.terms.bias.inds = inds_bias.data();
        param.terms.bias.params = params_bias.empty() ? nullptr : params_bias.data();
    };

    std::vector<Real> coords = lig.coords, dihedrals = lig.dihedrals;
    FlexPose pose{}; pose.coords = coords.data(); pose.dihedrals = dihedrals.data();
    pose.center[0] = lig.center[0]; pose.center[1] = lig.center[1]; pose.center[2] = lig.center[2];

    // variant 0 of the tyk2 parity test: angular HBond, device grid, no Coulomb
    dp.angular_hbond = true;
    dp.weights.elec = Real(0);
    dp.bias_k = Real(0.3);
    use_device_grid = true;
    param.charges = nullptr;
    fp.charges = nullptr;
    build_bonded_h_topo(param.bonded_h_topo, coords.data(), lig.vina_types.data(), lig.natom);
    BiasOffGuard guard{dp, atp};
    auto sync = [&](BiasType bt){
        dp.bias_type = bt;
        sync_backends(dp, atp, fix, rec.vina_types.data());
        g_cpu_constants.rec_donor_h_topo = rec.rec_donor_h;
    };

    // rows: every heavy atom at its crystal coordinate + (0.7, -0.4, 0.9) A, two more rows on each of the
    // first three heavy atoms, one row on the first hydrogen (v_set 2.0, width slot 2.0).
    // rows_h: one row on every hydrogen, so the hydrogen switch k_a = 0.1 is resolved on its own.
    std::vector<Bias> rows_all, rows_h;
    int n_extra = 0, first_h = -1;
    for (int a = 0; a < lig.natom; a++){
        const Real* xa = lig.coords.data() + a * 3;
        if (!atp.is_hydrogen[lig.vina_types[a]]){
            rows_all.push_back(make_bias_row(a, xa, 0.7, -0.4, 0.9, 2.0f, 2.0f));
            if (n_extra < 3){
                rows_all.push_back(make_bias_row(a, xa, -0.5, 0.8, 0.6, 2.0f, 2.0f));
                rows_all.push_back(make_bias_row(a, xa, 0.3, 0.9, -1.1, 2.0f, 2.0f));
                n_extra++;
            }
            continue;
        }
        if (first_h < 0){
            first_h = a;
            rows_all.push_back(make_bias_row(a, xa, -0.6, -0.7, 0.5, 2.0f, 2.0f));
        }
        rows_h.push_back(make_bias_row(a, xa, -0.6, -0.7, 0.5, 2.0f, 2.0f));
    }
    REQUIRE(first_h >= 0);
    REQUIRE(n_extra == 3);

    sync(BT_NO);
    install({});
    Result c_off = eval_cpu(fix, fp, param, pose, topo);
    Result g_off = eval_gpu(fix, fp, param, pose, topo);

    for (BiasType bt : {BT_ALIGN, BT_POS}){
        sync(bt);
        install(rows_all);
        Result c = eval_cpu(fix, fp, param, pose, topo);
        Result g = eval_gpu(fix, fp, param, pose, topo);
        const double e_ref = bias_energy_closed_form(rows_all, lig.vina_types.data(), coords.data(), bt, double(dp.bias_k));
        const double de_cpu = double(c.e) - double(c_off.e), de_gpu = double(g.e) - double(g_off.e);
        const Real tol = Real(2e-3) * std::max(Real(1), std::fabs(c.e));
        printf("[tyk2 bias=%s] natom %d, %zu rows | E_off cpu %.5f gpu %.5f | E_on cpu %.5f gpu %.5f |diff| %.2e | max |grad diff| %.2e"
               " | dE cpu %.5f gpu %.5f closed form %.5f\n",
               bias_name(bt), lig.natom, rows_all.size(), c_off.e, g_off.e, c.e, g.e, std::fabs(c.e - g.e),
               max_grad_dev(c, g, topo.ntorsion), de_cpu, de_gpu, e_ref);
        INFO("bias " << bias_name(bt) << " cpu e = " << c.e << " gpu e = " << g.e);

        REQUIRE(std::fabs(de_cpu) > 1);
        REQUIRE(std::fabs(de_gpu) > 1);
        REQUIRE_THAT(de_gpu, Catch::Matchers::WithinAbs(de_cpu, 1e-3 * std::fabs(de_cpu)));

        auto e_gpu_at = [&](int i_tor, Real dtheta){
            std::vector<Real> c2 = coords, d2 = dihedrals;
            FlexPose p2 = pose; p2.coords = c2.data(); p2.dihedrals = d2.data();
            apply_grad_update_dihe_cpu(&p2, &topo, i_tor, dtheta, true);
            return eval_gpu(fix, fp, param, p2, topo).e;
        };
        for (int i_tor = 0; i_tor < topo.ntorsion; i_tor++){
            const Real h2 = 2e-4;
            Real fd2 = (e_gpu_at(i_tor, h2) - e_gpu_at(i_tor, -h2)) / (2 * h2);
            printf("[tyk2 bias=%s] torsion %d | cpu grad %.4f | gpu grad %.4f | central fd h=2e-4 %.4f\n",
                   bias_name(bt), i_tor, c.dihe_g[i_tor], g.dihe_g[i_tor], fd2);
            CHECK_THAT(g.dihe_g[i_tor], Catch::Matchers::WithinAbs(fd2, Real(0.05) * std::max(Real(1), std::fabs(fd2))));
        }
        require_same(c, g, topo.ntorsion, tol);

        // bias-only differential, CPU vs GPU, within 1e-4 * max(1, |CPU difference|)
        for (int set = 0; set < 2; set++){
            const std::vector<Bias>& rows = set == 0 ? rows_all : rows_h;
            install(rows);
            Result c_on = eval_cpu(fix, fp, param, pose, topo);
            Result g_on = eval_gpu(fix, fp, param, pose, topo);
            std::vector<double> d_cpu, d_gpu;
            std::vector<std::string> label;
            d_cpu.push_back(double(c_on.e) - c_off.e); d_gpu.push_back(double(g_on.e) - g_off.e); label.push_back("E");
            for (int i = 0; i < 3; i++){
                d_cpu.push_back(double(c_on.g.center_g[i]) - c_off.g.center_g[i]);
                d_gpu.push_back(double(g_on.g.center_g[i]) - g_off.g.center_g[i]);
                label.push_back("center_g[" + std::to_string(i) + "]");
            }
            for (int i = 0; i < 3; i++){
                d_cpu.push_back(double(c_on.g.orientation_g[i]) - c_off.g.orientation_g[i]);
                d_gpu.push_back(double(g_on.g.orientation_g[i]) - g_off.g.orientation_g[i]);
                label.push_back("orientation_g[" + std::to_string(i) + "]");
            }
            for (int i = 0; i < topo.ntorsion; i++){
                d_cpu.push_back(double(c_on.dihe_g[i]) - c_off.dihe_g[i]);
                d_gpu.push_back(double(g_on.dihe_g[i]) - g_off.dihe_g[i]);
                label.push_back("dihe_g[" + std::to_string(i) + "]");
            }
            double worst = 0;
            size_t i_worst = 0;
            for (size_t i = 0; i < d_cpu.size(); i++){
                const double rel = std::fabs(d_gpu[i] - d_cpu[i]) / std::max(1.0, std::fabs(d_cpu[i]));
                if (rel > worst){ worst = rel; i_worst = i; }
            }
            printf("[tyk2 bias=%s differential %s] %zu rows | dE cpu %.6f gpu %.6f | d center_g cpu (%.5f %.5f %.5f) gpu (%.5f %.5f %.5f)"
                   " | d orientation_g cpu (%.5f %.5f %.5f) gpu (%.5f %.5f %.5f) | worst %s cpu %.6f gpu %.6f scaled dev %.2e\n",
                   bias_name(bt), set == 0 ? "all rows" : "hydrogen rows", rows.size(), d_cpu[0], d_gpu[0],
                   d_cpu[1], d_cpu[2], d_cpu[3], d_gpu[1], d_gpu[2], d_gpu[3],
                   d_cpu[4], d_cpu[5], d_cpu[6], d_gpu[4], d_gpu[5], d_gpu[6],
                   label[i_worst].c_str(), d_cpu[i_worst], d_gpu[i_worst], worst);
            for (size_t i = 0; i < d_cpu.size(); i++){
                INFO("bias " << bias_name(bt) << " set " << set << " " << label[i]);
                CHECK_THAT(d_gpu[i], Catch::Matchers::WithinAbs(d_cpu[i], 1e-4 * std::max(1.0, std::fabs(d_cpu[i]))));
            }
        }
    }
    free_bonded_h_topo(param.bonded_h_topo);
    free_spatial_grid(g_cpu_constants.receptor_grid);
    free_receptor_grid();
    g_cpu_constants.rec_donor_h_topo = {};
    use_device_grid = true;
}


// ---------------------------------------------------------------------------------------------------------------
// Large-ligand kernel paths with position bias.
// Synthetic rigid ligand: heavy atoms (C_H, C_H, N_A, O_A cycling) on an nx*ny*nz lattice centred on the origin,
// a hydrogen 1.0 A off every fourth heavy atom (a C_H). Intra pairs are the heavy-heavy pairs closer than
// pair_rmax (hydrogens skipped, as in the DMS reader).
static void build_lattice_ligand(ToyLigand& lig, const AtomTypeProps& atp, int nx, int ny, int nz, Real spacing, Real pair_rmax){
    const int cycle[4] = {VN_TYPE_C_H, VN_TYPE_C_H, VN_TYPE_N_A, VN_TYPE_O_A};
    const Real h = Real(1.0 / std::sqrt(3.0));
    std::vector<int> types;
    std::vector<Real> coords;
    int n_heavy = 0;
    for (int ix = 0; ix < nx; ix++){
        for (int iy = 0; iy < ny; iy++){
            for (int iz = 0; iz < nz; iz++){
                const Real x = (ix - (nx - 1) * Real(0.5)) * spacing;
                const Real y = (iy - (ny - 1) * Real(0.5)) * spacing;
                const Real z = (iz - (nz - 1) * Real(0.5)) * spacing;
                types.push_back(cycle[n_heavy % 4]);
                coords.insert(coords.end(), {x, y, z});
                if (n_heavy % 4 == 0){
                    types.push_back(VN_TYPE_H);
                    coords.insert(coords.end(), {x + h, y + h, z + h});
                }
                n_heavy++;
            }
        }
    }
    lig.build(types, coords);

    std::vector<int> pairs;
    std::vector<Real> r1r2;
    const Real rr_max = pair_rmax * pair_rmax;
    for (size_t k = 0; k < lig.pairs_intra.size(); k += 2){
        const int i = lig.pairs_intra[k], j = lig.pairs_intra[k + 1];
        if (atp.is_hydrogen[lig.types[i]] || atp.is_hydrogen[lig.types[j]]) continue;
        const Real dx = lig.coords[i * 3] - lig.coords[j * 3];
        const Real dy = lig.coords[i * 3 + 1] - lig.coords[j * 3 + 1];
        const Real dz = lig.coords[i * 3 + 2] - lig.coords[j * 3 + 2];
        if (dx * dx + dy * dy + dz * dz >= rr_max) continue;
        pairs.push_back(i); pairs.push_back(j);
        r1r2.push_back(lig.r1r2_intra[k / 2]);
    }
    lig.pairs_intra = pairs;
    lig.r1r2_intra = r1r2;
    lig.param.npair_intra = lig.pairs_intra.size() / 2;
    lig.param.pairs_intra = lig.pairs_intra.data();
    lig.param.r1_plus_r2_intra = lig.r1r2_intra.data();
}

// Receptor: a C_H plane 3.5 A under the ligand lattice (inter contacts through the receptor queue, which writes
// s_force) plus a far C_H plane at z = 45 (27 x 27, no contacts) that pads the receptor to more atoms than the
// ligand, because cal_e_grad_one_gpu sizes aux_f by the receptor natom.
static void build_plane_receptor(std::vector<Real>& coords, std::vector<int>& types, int nx, int ny, Real spacing, Real z_near){
    coords.clear(); types.clear();
    for (int ix = 0; ix < nx; ix++){
        for (int iy = 0; iy < ny; iy++){
            coords.insert(coords.end(), {(ix - (nx - 1) * Real(0.5)) * spacing, (iy - (ny - 1) * Real(0.5)) * spacing, z_near});
            types.push_back(VN_TYPE_C_H);
        }
    }
    for (int ix = 0; ix < 27; ix++){
        for (int iy = 0; iy < 27; iy++){
            coords.insert(coords.end(), {(ix - 13) * Real(3.5), (iy - 13) * Real(3.5), Real(45)});
            types.push_back(VN_TYPE_C_H);
        }
    }
}

// Bias rows on the lattice ligand, in atom order (one contiguous run per atom). Offsets have a tangential part
// around the z axis so the bias torque does not cancel over the symmetric lattice.
//   heavy atoms: one row, except every 7th heavy atom from the 4th on (gap atoms, no rows); a second row on every
//   11th heavy atom; hydrogens: one row each (also collected in rows_h).
static void lattice_bias_rows(const ToyLigand& lig, const AtomTypeProps& atp,
                              std::vector<Bias>& rows_all, std::vector<Bias>& rows_h, int& n_gap, int& n_double){
    rows_all.clear(); rows_h.clear();
    n_gap = 0; n_double = 0;
    int n_heavy = 0;
    for (int a = 0; a < lig.topo.natom; a++){
        const Real* xa = lig.coords.data() + a * 3;
        double tx = -(double(xa[1]) - lig.pose.center[1]), ty = double(xa[0]) - lig.pose.center[0];
        const double tn = std::sqrt(tx * tx + ty * ty);
        if (tn < 1e-3){ tx = 1; ty = 0; } else { tx /= tn; ty /= tn; }
        const Bias first = make_bias_row(a, xa, 0.8 * tx + 0.3, 0.8 * ty - 0.2, 0.4, 2.0f, 2.0f);
        if (atp.is_hydrogen[lig.types[a]]){
            rows_all.push_back(first);
            rows_h.push_back(first);
            continue;
        }
        const int k = n_heavy++;
        if (k % 7 == 3){ n_gap++; continue; }
        rows_all.push_back(first);
        if (k % 11 == 0){
            rows_all.push_back(make_bias_row(a, xa, -0.6 * tx - 0.4, -0.6 * ty + 0.5, 0.7, 1.5f, 2.0f));
            n_double++;
        }
    }
}

// Bias-only differential (E_on - E_off, center / orientation gradient differences), CPU vs GPU, within
// 1e-4 * max(1, |CPU difference|). Returns the worst scaled deviation.
static double check_bias_differential(const char* label, const Result& c_off, const Result& g_off,
                                      const Result& c_on, const Result& g_on){
    std::vector<double> d_cpu, d_gpu;
    std::vector<std::string> name;
    d_cpu.push_back(double(c_on.e) - c_off.e); d_gpu.push_back(double(g_on.e) - g_off.e); name.push_back("E");
    for (int i = 0; i < 3; i++){
        d_cpu.push_back(double(c_on.g.center_g[i]) - c_off.g.center_g[i]);
        d_gpu.push_back(double(g_on.g.center_g[i]) - g_off.g.center_g[i]);
        name.push_back("center_g[" + std::to_string(i) + "]");
    }
    for (int i = 0; i < 3; i++){
        d_cpu.push_back(double(c_on.g.orientation_g[i]) - c_off.g.orientation_g[i]);
        d_gpu.push_back(double(g_on.g.orientation_g[i]) - g_off.g.orientation_g[i]);
        name.push_back("orientation_g[" + std::to_string(i) + "]");
    }
    double worst = 0;
    size_t i_worst = 0;
    for (size_t i = 0; i < d_cpu.size(); i++){
        const double rel = std::fabs(d_gpu[i] - d_cpu[i]) / std::max(1.0, std::fabs(d_cpu[i]));
        if (rel > worst){ worst = rel; i_worst = i; }
    }
    printf("[%s] dE cpu %.6f gpu %.6f | d center_g cpu (%.5f %.5f %.5f) gpu (%.5f %.5f %.5f)"
           " | d orientation_g cpu (%.4f %.4f %.4f) gpu (%.4f %.4f %.4f) | worst %s cpu %.6f gpu %.6f scaled dev %.2e\n",
           label, d_cpu[0], d_gpu[0], d_cpu[1], d_cpu[2], d_cpu[3], d_gpu[1], d_gpu[2], d_gpu[3],
           d_cpu[4], d_cpu[5], d_cpu[6], d_gpu[4], d_gpu[5], d_gpu[6], name[i_worst].c_str(), d_cpu[i_worst], d_gpu[i_worst], worst);
    for (size_t i = 0; i < d_cpu.size(); i++){
        INFO(label << " " << name[i]);
        CHECK_THAT(d_gpu[i], Catch::Matchers::WithinAbs(d_cpu[i], 1e-4 * std::max(1.0, std::fabs(d_cpu[i]))));
    }
    return worst;
}

static void run_large_bias_case(const char* label, int nx, int ny, int nz, Real spacing, Real pair_rmax,
                                bool expect_aliased_force, bool expect_compacted_intra){
    AtomTypeProps atp; init_default_atom_type_props(atp);
    ToyLigand lig;
    build_lattice_ligand(lig, atp, nx, ny, nz, spacing, pair_rmax);
    const int natom = lig.topo.natom;
    // bfgs.cuh: s_force aliases aux_f above KERNEL_SHARED_FORCE_MAX_ATOMS; compacted intra above INTRA_COMPACT_MIN_PAIRS = 4096
    REQUIRE((natom > KERNEL_SHARED_FORCE_MAX_ATOMS) == expect_aliased_force);
    REQUIRE((lig.param.npair_intra > 4096) == expect_compacted_intra);

    Real z_min = lig.coords[2];
    for (int a = 0; a < natom; a++) z_min = std::min(z_min, lig.coords[a * 3 + 2]);
    std::vector<Real> rec_coords;
    std::vector<int> rec_types;
    build_plane_receptor(rec_coords, rec_types, nx, ny, spacing, z_min - Real(3.5));
    FixMol fix; fix.natom = rec_types.size(); fix.coords = rec_coords.data();
    FixParamVina fp; fp.atom_types = rec_types.data();
    REQUIRE(fix.natom >= natom);

    std::vector<Bias> rows_all, rows_h;
    int n_gap = 0, n_double = 0;
    lattice_bias_rows(lig, atp, rows_all, rows_h, n_gap, n_double);
    REQUIRE(n_gap > 0);
    REQUIRE(n_double > 0);
    REQUIRE(!rows_h.empty());

    DockParam dp = large_box_param(true);
    dp.bias_k = Real(0.3);
    BiasOffGuard guard{dp, atp};

    dp.bias_type = BT_NO;
    sync_backends(dp, atp, fix, rec_types.data());
    lig.set_bias({});
    Result c_off = eval_cpu(fix, fp, lig.param, lig.pose, lig.topo);
    Result g_off = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
    printf("[%s] natom %d (hydrogens %zu), intra pairs %d, receptor atoms %d, rows %zu (gap atoms %d, two-row atoms %d)"
           " | bias off: E cpu %.5f gpu %.5f |diff| %.2e max |grad diff| %.2e\n",
           label, natom, rows_h.size(), lig.param.npair_intra, fix.natom, rows_all.size(), n_gap, n_double,
           c_off.e, g_off.e, std::fabs(c_off.e - g_off.e), max_grad_dev(c_off, g_off, 0));
    REQUIRE(std::fabs(c_off.e) > Real(0.1));   // the non-bias terms are live (receptor contacts, intra pairs)
    require_same(c_off, g_off, 0, Real(2e-3) * std::max(Real(1), std::fabs(c_off.e)));

    for (BiasType bt : {BT_ALIGN, BT_POS}){
        dp.bias_type = bt;
        sync_backends(dp, atp, fix, rec_types.data());
        for (int set = 0; set < 2; set++){
            const std::vector<Bias>& rows = set == 0 ? rows_all : rows_h;
            lig.set_bias(rows);
            Result c_on = eval_cpu(fix, fp, lig.param, lig.pose, lig.topo);
            Result g_on = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
            const double e_ref = bias_energy_closed_form(rows, lig.types.data(), lig.coords.data(), bt, double(dp.bias_k));
            const std::string tag = std::string(label) + " bias=" + bias_name(bt) + (set == 0 ? " all rows" : " hydrogen rows");
            printf("[%s] %zu rows | E_on cpu %.5f gpu %.5f |diff| %.2e max |grad diff| %.2e | closed-form dE %.6f\n",
                   tag.c_str(), rows.size(), c_on.e, g_on.e, std::fabs(c_on.e - g_on.e), max_grad_dev(c_on, g_on, 0), e_ref);
            INFO(tag);
            if (set == 0){
                REQUIRE(std::fabs(double(c_on.e) - c_off.e) > 1);
                REQUIRE(std::fabs(double(g_on.e) - g_off.e) > 1);
            }
            require_same(c_on, g_on, 0, Real(2e-3) * std::max(Real(1), std::fabs(c_on.e)));
            check_bias_differential(tag.c_str(), c_off, g_off, c_on, g_on);
        }
    }
    free_spatial_grid(g_cpu_constants.receptor_grid);
    free_receptor_grid();
}


TEST_CASE("gpu vs cpu: position bias, synthetic ligand over 512 atoms (aliased force path)", "[gpu_vs_cpu][bias][large]"){
    // 9 x 9 x 7 heavy atoms at 4 A + 142 hydrogens = 709 atoms; face-neighbour pairs only (1494 <= 4096)
    run_large_bias_case("large aliased", 9, 9, 7, Real(4.0), Real(4.5), true, false);
    // same ligand, pairs up to 8.5 A (> 4096): aliased force together with the compacted intra path
    run_large_bias_case("large aliased+compacted", 9, 9, 7, Real(4.0), Real(8.5), true, true);
}


TEST_CASE("gpu vs cpu: position bias, synthetic ligand with over 4096 intra pairs (compacted intra path)", "[gpu_vs_cpu][bias][large]"){
    // 5 x 5 x 5 heavy atoms at 3.5 A + 32 hydrogens = 157 atoms (shared-force path); all 7750 heavy pairs
    run_large_bias_case("compacted intra", 5, 5, 5, Real(3.5), Real(1000), false, true);
}


// ---------------------------------------------------------------------------------------------------------------
// Aromatic-ring bias (ext/bias/ring_term.h ring_bias_frame / ring_bias_atom_grad / RingBiasTerm, run by score/terms.h TermSet
// from score_cpu.cpp "4." and bfgs.cuh "3b."; the October cases of feat/ud2align-ring-bias).

static void ring_frame(const Real* x, const int* ids, int m, double* c, double* n){
    c[0] = c[1] = c[2] = 0;
    n[0] = n[1] = n[2] = 0;
    for (int k = 0; k < m; k++) for (int j = 0; j < 3; j++) c[j] += double(x[ids[k] * 3 + j]) / m;
    for (int k = 0; k < m; k++){
        const Real* p = x + ids[k] * 3;
        const Real* q = x + ids[(k + 1) % m] * 3;
        const double a[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};
        const double b[3] = {q[0] - c[0], q[1] - c[1], q[2] - c[2]};
        n[0] += a[1] * b[2] - a[2] * b[1];
        n[1] += a[2] * b[0] - a[0] * b[2];
        n[2] += a[0] * b[1] - a[1] * b[0];
    }
    const double l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    n[0] /= l; n[1] /= l; n[2] /= l;
}

// Double-precision closed form of the ring-bias energy: E = -v_set bias_k 30 (n . n_t)^2 / (1 + exp(3 (d - 1)))
static double ring_energy_closed_form(const std::vector<RingBiasRow>& rows, const Real* x, double bias_k){
    double e = 0;
    for (const auto& r : rows){
        double c[3], n[3];
        ring_frame(x, r.atoms, r.n, c, n);
        const double d = std::sqrt((c[0] - r.c[0]) * (c[0] - r.c[0]) + (c[1] - r.c[1]) * (c[1] - r.c[1])
                                   + (c[2] - r.c[2]) * (c[2] - r.c[2]));
        const double cs = n[0] * r.nt[0] + n[1] * r.nt[1] + n[2] * r.nt[2];
        e -= double(r.v_set) * bias_k * 30.0 * cs * cs / (1.0 + std::exp(3.0 * (d - 1.0)));
    }
    return e;
}

// Row on ring atoms ids: target centroid = own centroid + shift, target normal = unit(normal of `base` atoms + tilt)
static RingBiasRow make_ring_row(const Real* x, const std::vector<int>& ids, const double* shift,
                                 const std::vector<int>& base, const double* tilt, float v_set){
    RingBiasRow r;
    r.n = ids.size();
    for (int k = 0; k < r.n; k++) r.atoms[k] = ids[k];
    double c[3], n[3], cb[3], nb[3];
    ring_frame(x, r.atoms, r.n, c, n);
    ring_frame(x, base.data(), base.size(), cb, nb);
    double nt[3] = {nb[0] + tilt[0], nb[1] + tilt[1], nb[2] + tilt[2]};
    const double l = std::sqrt(nt[0] * nt[0] + nt[1] * nt[1] + nt[2] * nt[2]);
    for (int j = 0; j < 3; j++){ r.c[j] = Real(c[j] + shift[j]); r.nt[j] = Real(nt[j] / l); }
    r.v_set = v_set;
    return r;
}

static void set_rings(FlexParamVina& param, const std::vector<RingBiasRow>& rows){
    param.terms.ring.n = rows.size();
    param.terms.ring.rows = rows.empty() ? nullptr : rows.data();
}


// Toy rigid ligand: a hexagon and a pentagon 4 A apart plus one hydrogen; rows: hexagon and pentagon on
// shifted, tilted copies of themselves and the hexagon on a pentagon-oriented target, so two lanes add to
// the same atoms. No intra pairs and no receptor contact: the ring rows are the only energy under BT_ALIGN.
TEST_CASE("gpu vs cpu: ring bias toy", "[gpu_vs_cpu][bias][ring_bias]"){
    AtomTypeProps atp; init_default_atom_type_props(atp);
    std::vector<Real> rec_coords;
    for (int i = 0; i < 12; i++){ rec_coords.push_back(30 + 4 * (i % 3)); rec_coords.push_back(30 + 4 * (i / 3)); rec_coords.push_back(30); }
    std::vector<int> rec_types(12, VN_TYPE_C_H);
    FixMol fix; fix.natom = 12; fix.coords = rec_coords.data();
    FixParamVina fp; fp.atom_types = rec_types.data();
    BiasOffGuard guard{large_box_param(true), atp};

    std::vector<int> types;
    std::vector<Real> xyz;
    for (int k = 0; k < 6; k++){
        const double th = 2 * M_PI * k / 6;
        types.push_back(VN_TYPE_C_H);
        xyz.push_back(Real(1.39 * std::cos(th))); xyz.push_back(Real(1.39 * std::sin(th) * 0.8)); xyz.push_back(Real(1.39 * std::sin(th) * 0.6));
    }
    for (int k = 0; k < 5; k++){
        const double th = 2 * M_PI * k / 5;
        types.push_back(k == 0 ? VN_TYPE_N_A : VN_TYPE_C_H);
        xyz.push_back(Real(4.0 + 1.18 * std::cos(th) * 0.6)); xyz.push_back(Real(1.18 * std::sin(th))); xyz.push_back(Real(1.18 * std::cos(th) * 0.8));
    }
    types.push_back(VN_TYPE_H);
    xyz.push_back(Real(2.0)); xyz.push_back(Real(-1.5)); xyz.push_back(Real(0.3));
    ToyLigand lig; lig.build(types, xyz);
    lig.param.npair_intra = 0;
    const Real* x = lig.coords.data();
    const std::vector<int> hex = {0, 1, 2, 3, 4, 5}, pent = {6, 7, 8, 9, 10};
    const double s1[3] = {0.5, -0.3, 0.4}, t1[3] = {0.3, 0.4, -0.2};
    const double s2[3] = {-0.4, 0.2, 0.5}, t2[3] = {-0.2, 0.1, 0.3};
    const double s3[3] = {0.9, 0.4, -0.3}, t0[3] = {0, 0, 0};
    const std::vector<RingBiasRow> rows = {make_ring_row(x, hex, s1, hex, t1, 2.0f), make_ring_row(x, pent, s2, pent, t2, 4.0f),
                                           make_ring_row(x, hex, s3, pent, t0, 2.0f)};

    for (Real k : {Real(0.05), Real(0.3)}){
        DockParam dp = large_box_param(true);
        dp.bias_k = k;
        for (BiasType bt : {BT_NO, BT_POS}){
            dp.bias_type = bt;
            sync_backends(dp, atp, fix, rec_types.data());
            set_rings(lig.param, {});
            Result g_zero = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
            set_rings(lig.param, rows);
            Result g_rows = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
            INFO("bias " << bias_name(bt) << " k " << k);
            REQUIRE_THAT(g_rows.e, Catch::Matchers::WithinAbs(g_zero.e, 1e-6));
            require_same(g_zero, g_rows, 0, Real(1e-6));
        }

        dp.bias_type = BT_ALIGN;
        sync_backends(dp, atp, fix, rec_types.data());
        set_rings(lig.param, {});
        Result c_off = eval_cpu(fix, fp, lig.param, lig.pose, lig.topo);
        Result g_off = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
        set_rings(lig.param, rows);
        Result c_on = eval_cpu(fix, fp, lig.param, lig.pose, lig.topo);
        Result g_on = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
        const double e_ref = ring_energy_closed_form(rows, lig.coords.data(), double(k));
        const double de_cpu = double(c_on.e) - double(c_off.e), de_gpu = double(g_on.e) - double(g_off.e);
        printf("[ring bias toy k=%.2f] E_off cpu %.6f gpu %.6f | E_on cpu %.6f gpu %.6f max |grad diff| %.2e"
               " | dE cpu %.6f gpu %.6f closed form %.6f\n",
               (double)k, c_off.e, g_off.e, c_on.e, g_on.e, max_grad_dev(c_on, g_on, 0), de_cpu, de_gpu, e_ref);
        INFO("ring bias toy k " << k);
        REQUIRE(e_ref < -0.5);
        require_same(c_on, g_on, 0, Real(1e-4) * std::max(Real(1), std::fabs(c_on.e)));
        REQUIRE_THAT(de_cpu, Catch::Matchers::WithinAbs(e_ref, 1e-4 * std::fabs(e_ref)));
        REQUIRE_THAT(de_gpu, Catch::Matchers::WithinAbs(e_ref, 1e-4 * std::fabs(e_ref)));
        check_bias_differential(("ring bias toy k=" + std::to_string(k)).c_str(), c_off, g_off, c_on, g_on);
        // central: the forward difference is ~0.021 off here at k = 0.3 even in exact arithmetic
        if (k == Real(0.3)) fd_check_gpu_center(fix, fp, lig, 0.02, true);
    }
    set_rings(lig.param, {});
}


// The pose pass (DockTask::run_score, plan step 7) evaluates the bias on the host with HostTermSet (score/terms.h),
// built from the ligand's rows by flatten_bias_rows, as the CUDA fill flattens them before the upload. On the same
// pose its energy is the device term's (kernel E_on - E_off) and the CPU kernel's, for the atom rows alone (BT_POS)
// and the atom + ring rows (BT_ALIGN). Toy rigid ligand: a hexagon, a pentagon and a hydrogen, no intra pairs, no
// receptor contact.
TEST_CASE("gpu vs cpu: the pose pass's host term set gives the device bias energy", "[gpu_vs_cpu][bias][pose_pass]"){
    AtomTypeProps atp; init_default_atom_type_props(atp);
    std::vector<Real> rec_coords;
    for (int i = 0; i < 12; i++){ rec_coords.push_back(30 + 4 * (i % 3)); rec_coords.push_back(30 + 4 * (i / 3)); rec_coords.push_back(30); }
    std::vector<int> rec_types(12, VN_TYPE_C_H);
    FixMol fix; fix.natom = 12; fix.coords = rec_coords.data();
    FixParamVina fp; fp.atom_types = rec_types.data();
    BiasOffGuard guard{large_box_param(true), atp};

    std::vector<int> types;
    std::vector<Real> xyz;
    for (int k = 0; k < 6; k++){
        const double th = 2 * M_PI * k / 6;
        types.push_back(VN_TYPE_C_H);
        xyz.push_back(Real(1.39 * std::cos(th))); xyz.push_back(Real(1.39 * std::sin(th) * 0.8)); xyz.push_back(Real(1.39 * std::sin(th) * 0.6));
    }
    for (int k = 0; k < 5; k++){
        const double th = 2 * M_PI * k / 5;
        types.push_back(k == 0 ? VN_TYPE_N_A : VN_TYPE_C_H);
        xyz.push_back(Real(4.0 + 1.18 * std::cos(th) * 0.6)); xyz.push_back(Real(1.18 * std::sin(th))); xyz.push_back(Real(1.18 * std::cos(th) * 0.8));
    }
    types.push_back(VN_TYPE_H);
    xyz.push_back(Real(2.0)); xyz.push_back(Real(-1.5)); xyz.push_back(Real(0.3));
    ToyLigand lig; lig.build(types, xyz);
    lig.param.npair_intra = 0;
    const Real* x = lig.coords.data();
    const std::vector<Bias> atom_rows = {
        make_bias_row(0, x,       0.6, 0.8, 0.0,      1.0f, 2.0f),
        make_bias_row(0, x,       0.0, 0.9, -1.2,     2.5f, 2.0f),
        make_bias_row(6, x + 18,  0.7, -0.4, 0.9,     3.0f, 2.0f),
        make_bias_row(11, x + 33, 1.0, 0.5, -1.0,     4.0f, 2.0f),   // hydrogen
    };
    const std::vector<int> hex = {0, 1, 2, 3, 4, 5}, pent = {6, 7, 8, 9, 10};
    const double s1[3] = {0.5, -0.3, 0.4}, t1[3] = {0.3, 0.4, -0.2};
    const double s2[3] = {-0.4, 0.2, 0.5}, t2[3] = {-0.2, 0.1, 0.3};
    const std::vector<RingBiasRow> ring_rows = {make_ring_row(x, hex, s1, hex, t1, 2.0f), make_ring_row(x, pent, s2, pent, t2, 4.0f)};
    UDFlexMol m;                      // the ligand as DockTask holds it: rows in UDFlexMol::bias / ::ring
    m.natom = static_cast<int>(types.size());
    m.vina_types = types;
    m.bias.rows = atom_rows;
    m.ring.rows = ring_rows;

    for (BiasType bt : {BT_POS, BT_ALIGN}){
        DockParam dp = large_box_param(true);
        dp.bias_type = bt;
        dp.bias_k = Real(0.3);
        sync_backends(dp, atp, fix, rec_types.data());
        lig.set_bias({});
        set_rings(lig.param, {});
        const Result c_off = eval_cpu(fix, fp, lig.param, lig.pose, lig.topo);
        const Result g_off = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);
        lig.set_bias(atom_rows);
        set_rings(lig.param, ring_rows);
        const Result c_on = eval_cpu(fix, fp, lig.param, lig.pose, lig.topo);
        const Result g_on = eval_gpu(fix, fp, lig.param, lig.pose, lig.topo);

        HostTermSet host(m, term_settings_of(dp));
        const double e_host = host.energy(lig.coords.data());
        const double de_cpu = double(c_on.e) - double(c_off.e), de_gpu = double(g_on.e) - double(g_off.e);
        printf("[pose pass host terms %s] host %.6f | kernel dE cpu %.6f gpu %.6f\n", bias_name(bt), e_host, de_cpu, de_gpu);
        INFO("bias " << bias_name(bt));
        REQUIRE(std::fabs(e_host) > 0.5);
        REQUIRE_THAT(de_gpu, Catch::Matchers::WithinAbs(e_host, 1e-4 * std::max(1.0, std::fabs(e_host))));
        REQUIRE_THAT(de_cpu, Catch::Matchers::WithinAbs(e_host, 1e-4 * std::max(1.0, std::fabs(e_host))));
    }
    lig.set_bias({});
    set_rings(lig.param, {});
}


// Real system with ring bias (tyk2, angular HBond, device grid; benzene atoms 0-5 and pyridine 10-15 on opposite
// sides of rotatable bonds): parity, GPU torsion finite differences of the bias part, and the ring-only differential
// (E_on - E_off and every DOF gradient) CPU vs GPU, alone and together with atom-bias rows.
TEST_CASE("gpu vs cpu: real system tyk2 with ring bias", "[gpu_vs_cpu][real][bias][ring_bias]"){
    AtomTypeProps atp; init_default_atom_type_props(atp);

    UDFlexMol lig; FFMinData ff_lig;
    read_ligand_dms_full(TYK2_DIR + "ligand_ejm42_enriched.dms", lig, ff_lig, atp, false);
    REQUIRE(lig.natom == 35);
    REQUIRE(lig.torsions.size() > 0);

    DockParam dp;
    const Real half = 15, cutoff = 8;
    dp.box.x_lo = lig.center[0] - half; dp.box.x_hi = lig.center[0] + half;
    dp.box.y_lo = lig.center[1] - half; dp.box.y_hi = lig.center[1] + half;
    dp.box.z_lo = lig.center[2] - half; dp.box.z_hi = lig.center[2] + half;
    Box box_protein = dp.box;
    box_protein.x_lo -= cutoff; box_protein.x_hi += cutoff;
    box_protein.y_lo -= cutoff; box_protein.y_hi += cutoff;
    box_protein.z_lo -= cutoff; box_protein.z_hi += cutoff;

    UDFixMol rec; FFMinData ff_rec;
    read_receptor_dms_full(TYK2_DIR + "receptor_enriched.dms", box_protein, rec, ff_rec, atp);
    REQUIRE(rec.natom > 100);

    FixMol fix; fix.natom = rec.natom; fix.coords = rec.coords.data();
    FixParamVina fp; fp.atom_types = rec.vina_types.data(); fp.rec_donor_h = rec.rec_donor_h;

    HostFlexTopo host_topo; build_host_flex_topo(lig, host_topo);
    const FlexTopo& topo = host_topo.topo;

    std::vector<Real> r1r2;
    for (size_t j = 0; j < lig.intra_pairs.size(); j += 2){
        r1r2.push_back(VN_VDW_RADII[lig.vina_types[lig.intra_pairs[j]]] + VN_VDW_RADII[lig.vina_types[lig.intra_pairs[j + 1]]]);
    }
    std::vector<int> inds_bias;
    std::vector<Real> params_bias;
    FlexParamVina param{};
    param.npair_intra = lig.intra_pairs.size() / 2;
    param.pairs_intra = lig.intra_pairs.data();
    param.r1_plus_r2_intra = r1r2.data();
    param.atom_types = lig.vina_types.data();
    auto install_atoms = [&](const std::vector<Bias>& rows){
        flatten_bias(rows, lig.natom, inds_bias, params_bias);
        param.terms.bias.inds = inds_bias.data();
        param.terms.bias.params = params_bias.empty() ? nullptr : params_bias.data();
    };
    install_atoms({});

    std::vector<Real> coords = lig.coords, dihedrals = lig.dihedrals;
    FlexPose pose{}; pose.coords = coords.data(); pose.dihedrals = dihedrals.data();
    pose.center[0] = lig.center[0]; pose.center[1] = lig.center[1]; pose.center[2] = lig.center[2];

    dp.angular_hbond = true;
    dp.weights.elec = Real(0);
    dp.bias_k = Real(0.3);
    use_device_grid = true;
    param.charges = nullptr;
    fp.charges = nullptr;
    build_bonded_h_topo(param.bonded_h_topo, coords.data(), lig.vina_types.data(), lig.natom);
    BiasOffGuard guard{dp, atp};
    auto sync = [&](BiasType bt){
        dp.bias_type = bt;
        sync_backends(dp, atp, fix, rec.vina_types.data());
        g_cpu_constants.rec_donor_h_topo = rec.rec_donor_h;
    };

    const Real* x = coords.data();
    const std::vector<int> benz = {0, 1, 2, 3, 4, 5}, pyr = {10, 11, 12, 13, 14, 15}, five = {10, 11, 12, 13, 14};
    const double s1[3] = {0.5, -0.3, 0.4}, t1[3] = {0.3, 0.4, -0.2};
    const double s2[3] = {-0.4, 0.6, 0.3}, t2[3] = {-0.3, 0.2, 0.4};
    const double s3[3] = {1.0, 0.5, -0.5}, t0[3] = {0, 0, 0};
    const double s4[3] = {0.3, 0.3, 0.3}, t4[3] = {0.5, -0.4, 0.1};
    const std::vector<RingBiasRow> rings = {make_ring_row(x, benz, s1, benz, t1, 2.0f), make_ring_row(x, pyr, s2, pyr, t2, 4.0f),
                                            make_ring_row(x, pyr, s3, benz, t0, 2.0f), make_ring_row(x, five, s4, five, t4, 1.0f)};
    std::vector<Bias> atom_rows;
    for (int a : {0, 3, 7, 12, 16, 19}) atom_rows.push_back(make_bias_row(a, x + a * 3, 0.7, -0.4, 0.9, 2.0f, 2.0f));

    sync(BT_NO);
    set_rings(param, {});
    Result c_off = eval_cpu(fix, fp, param, pose, topo);
    Result g_off = eval_gpu(fix, fp, param, pose, topo);

    sync(BT_ALIGN);
    for (int set = 0; set < 2; set++){
        const char* label = set == 0 ? "rings only" : "rings + atom rows";
        install_atoms(set == 0 ? std::vector<Bias>{} : atom_rows);
        set_rings(param, rings);
        Result c = eval_cpu(fix, fp, param, pose, topo);
        Result g = eval_gpu(fix, fp, param, pose, topo);
        const double de_cpu = double(c.e) - double(c_off.e), de_gpu = double(g.e) - double(g_off.e);
        printf("[tyk2 ring bias, %s] %zu rings | E_off cpu %.5f gpu %.5f | E_on cpu %.5f gpu %.5f |diff| %.2e"
               " | max |grad diff| %.2e | dE cpu %.5f gpu %.5f\n",
               label, rings.size(), c_off.e, g_off.e, c.e, g.e, std::fabs(c.e - g.e), max_grad_dev(c, g, topo.ntorsion), de_cpu, de_gpu);
        INFO(label << " cpu e = " << c.e << " gpu e = " << g.e);
        if (set == 0){
            const double e_ref = ring_energy_closed_form(rings, coords.data(), double(dp.bias_k));
            REQUIRE(e_ref < -10);
            REQUIRE_THAT(de_cpu, Catch::Matchers::WithinAbs(e_ref, 1e-4 * std::fabs(e_ref)));
        }
        REQUIRE_THAT(de_gpu, Catch::Matchers::WithinAbs(de_cpu, 1e-3 * std::fabs(de_cpu)));
        require_same(c, g, topo.ntorsion, Real(2e-3) * std::max(Real(1), std::fabs(c.e)));

        // finite differences of the bias part only (rows on minus rows off at the same pose): this cancels the small
        // steps of the non-bias energy along torsions 0 and 1, and h = 1e-3 stays above the float rounding of the
        // rotated coordinates far from the origin, which shifts a total-energy difference at h = 2e-4 by ~0.3
        auto de_gpu_at = [&](int i_tor, Real dtheta){
            std::vector<Real> c2 = coords, d2 = dihedrals;
            FlexPose p2 = pose; p2.coords = c2.data(); p2.dihedrals = d2.data();
            apply_grad_update_dihe_cpu(&p2, &topo, i_tor, dtheta, true);
            const double e_on = eval_gpu(fix, fp, param, p2, topo).e;
            set_rings(param, {});
            install_atoms({});
            const double e_bare = eval_gpu(fix, fp, param, p2, topo).e;
            set_rings(param, rings);
            install_atoms(set == 0 ? std::vector<Bias>{} : atom_rows);
            return e_on - e_bare;
        };
        for (int i_tor = 0; i_tor < topo.ntorsion; i_tor++){
            const Real h = 1e-3;
            const double fd = (de_gpu_at(i_tor, h) - de_gpu_at(i_tor, -h)) / (2 * h);
            const double dg = double(g.dihe_g[i_tor]) - g_off.dihe_g[i_tor];
            printf("[tyk2 ring bias, %s] torsion %d | cpu grad %.4f | gpu grad %.4f | bias part gpu %.4f central fd h=1e-3 %.4f\n",
                   label, i_tor, c.dihe_g[i_tor], g.dihe_g[i_tor], dg, fd);
            CHECK_THAT(dg, Catch::Matchers::WithinAbs(fd, 0.05 * std::max(1.0, std::fabs(fd))));
        }

        // differential over every degree of freedom, CPU vs GPU, within 1e-4 * max(1, |CPU difference|)
        std::vector<double> d_cpu, d_gpu;
        std::vector<std::string> name;
        d_cpu.push_back(de_cpu); d_gpu.push_back(de_gpu); name.push_back("E");
        for (int i = 0; i < 3; i++){
            d_cpu.push_back(double(c.g.center_g[i]) - c_off.g.center_g[i]);
            d_gpu.push_back(double(g.g.center_g[i]) - g_off.g.center_g[i]);
            name.push_back("center_g[" + std::to_string(i) + "]");
            d_cpu.push_back(double(c.g.orientation_g[i]) - c_off.g.orientation_g[i]);
            d_gpu.push_back(double(g.g.orientation_g[i]) - g_off.g.orientation_g[i]);
            name.push_back("orientation_g[" + std::to_string(i) + "]");
        }
        for (int i = 0; i < topo.ntorsion; i++){
            d_cpu.push_back(double(c.dihe_g[i]) - c_off.dihe_g[i]);
            d_gpu.push_back(double(g.dihe_g[i]) - g_off.dihe_g[i]);
            name.push_back("dihe_g[" + std::to_string(i) + "]");
        }
        double worst = 0;
        size_t i_worst = 0;
        for (size_t i = 0; i < d_cpu.size(); i++){
            const double rel = std::fabs(d_gpu[i] - d_cpu[i]) / std::max(1.0, std::fabs(d_cpu[i]));
            if (rel > worst){ worst = rel; i_worst = i; }
        }
        printf("[tyk2 ring bias, %s differential] worst %s cpu %.6f gpu %.6f scaled dev %.2e\n",
               label, name[i_worst].c_str(), d_cpu[i_worst], d_gpu[i_worst], worst);
        for (size_t i = 0; i < d_cpu.size(); i++){
            INFO(label << " " << name[i]);
            CHECK_THAT(d_gpu[i], Catch::Matchers::WithinAbs(d_cpu[i], 1e-4 * std::max(1.0, std::fabs(d_cpu[i]))));
        }
    }
    set_rings(param, {});
    free_bonded_h_topo(param.bonded_h_topo);
    free_spatial_grid(g_cpu_constants.receptor_grid);
    free_receptor_grid();
    g_cpu_constants.rec_donor_h_topo = {};
    use_device_grid = true;
}
