//
// Unit tests for position-bias in CPU scoring pipeline (cal_e_f_cpu).
//
// Tests:
//  1. BT_NO:    no bias entries → zero bias contribution
//  2. BT_ALIGN: single atom near reference → attractive energy and force
//  3. BT_ALIGN: force magnitude vs distance (sigmoid profile, vanishes > 2 Å)
//  4. BT_ALIGN: translation gradient is frozen (center_g == 0)
//  5. BT_POS:   Gaussian potential (energy decays with distance)
//

#include <catch2/catch_amalgamated.hpp>
#include "cpu/common_cpu.h"
#include "cpu/score_cpu.h"
#include "model/model.h"
#include "constants/constants.h"
#include <vector>
#include <cstring>
#include <cmath>

using Catch::Approx;
using Catch::Matchers::WithinAbs;

// The bias settings of the current test case (setup_bias_constants); the setups below stamp them into their
// FlexParamVina::terms.settings, which is where cal_e_f_cpu reads them (there is no CPU constant for them).
static BiasType s_bias_type = BT_NO;
static Real s_bias_k = 0.1;

// ---------------------------------------------------------------
// Helper: build a minimal 1-atom "molecule" with optional bias.
//
// The single atom is placed at `atom_pos` with no receptor, no
// intra/inter pairs, so the ONLY energy contribution comes from bias.
// ---------------------------------------------------------------
struct MinimalBiasSetup {
    // Molecule data
    std::vector<Real> coords;
    std::vector<Real> forces;

    // FlexTopo (1 atom, 0 torsions)
    FlexTopo topo;
    std::vector<int> vn_types_vec;
    std::vector<int> rotated_inds_vec;
    std::vector<int> axis_atoms_vec;
    std::vector<int> range_inds_vec;
    std::vector<Real> range_list_vec;
    std::vector<int> rotated_atoms_vec;

    // FlexParamVina (no vina pairs, only bias)
    FlexParamVina flex_param;
    std::vector<int> bias_inds;    // size: natom * 2
    std::vector<Real> bias_params; // size: nbias * 5

    // FixMol & FixParam (empty receptor)
    FixMol fix_mol;
    FixParamVina fix_param;
    std::vector<int> fix_types_vec;

    // FlexPose
    FlexPose pose;

    MinimalBiasSetup(Real ax, Real ay, Real az,
                     int vn_type,
                     const std::vector<Bias>& biases) {
        const int natom = 1;

        // -- coords & forces
        coords = {ax, ay, az};
        forces.resize(natom * 3, 0.0);

        // -- FlexTopo
        topo.natom = natom;
        topo.ntorsion = 0;
        vn_types_vec = {vn_type};
        rotated_inds_vec = {};
        axis_atoms_vec = {};
        range_inds_vec = {};
        range_list_vec = {};
        rotated_atoms_vec = {};
        topo.vn_types = vn_types_vec.data();
        topo.rotated_inds = rotated_inds_vec.data();
        topo.axis_atoms = axis_atoms_vec.data();
        topo.range_inds = range_inds_vec.data();
        topo.range_list = range_list_vec.data();
        topo.rotated_atoms = rotated_atoms_vec.data();

        // -- FlexPose (center = atom pos, identity rotation)
        pose.coords = coords.data();
        pose.center[0] = ax; pose.center[1] = ay; pose.center[2] = az;
        pose.rot_vec[0] = 0; pose.rot_vec[1] = 0; pose.rot_vec[2] = 0;
        pose.energy = 0;
        pose.dihedrals = nullptr;

        // -- FlexParamVina: no intra/inter pairs
        flex_param.npair_intra = 0;
        flex_param.pairs_intra = nullptr;
        flex_param.npair_inter = 0;
        flex_param.pairs_inter = nullptr;
        flex_param.r1_plus_r2_intra = nullptr;
        flex_param.r1_plus_r2_inter = nullptr;
        std::vector<int> atom_types_vec = {vn_type};
        flex_param.atom_types = atom_types_vec.data();
        // Copy to keep alive (stored in member for lifetime)
        this->atom_types_storage = atom_types_vec;
        flex_param.atom_types = this->atom_types_storage.data();

        // -- Flatten biases into inds_bias / params_bias
        bias_inds.resize(natom * 2, 0);
        bias_params.clear();
        int last_i = -1;
        int offset = 0;
        for (const auto& b : biases) {
            if (b.i != last_i) {
                bias_inds[b.i * 2] = offset;
            }
            for (int k = 0; k < 5; k++) bias_params.push_back(b.param[k]);
            offset += 5;
            bias_inds[b.i * 2 + 1] = offset;
            last_i = b.i;
        }
        if (bias_params.empty()) bias_params.push_back(0); // avoid nullptr
        flex_param.terms.bias.inds = bias_inds.data();
        flex_param.terms.bias.params = bias_params.data();
        flex_param.terms.settings.bias_type = s_bias_type;
        flex_param.terms.settings.bias_k = s_bias_k;

        // -- FixMol (empty receptor)
        fix_mol.natom = 0;
        fix_mol.coords = nullptr;
        fix_types_vec = {};
        fix_param.atom_types = nullptr;
    }

    Real run_energy() {
        return cal_e_f_cpu(&pose, topo, fix_mol, flex_param, fix_param, forces.data());
    }

private:
    std::vector<int> atom_types_storage;
};


// ---------------------------------------------------------------
//  Helper: set up g_cpu_constants and the bias settings for bias testing
// ---------------------------------------------------------------
static void setup_bias_constants(BiasType bt, Real bias_k = 0.1) {
    // the setups built after this call carry these in their terms.settings
    s_bias_type = bt;
    s_bias_k = bias_k;
    // Cannot copy-assign CPUConstants (Vina has const members), set fields individually
    g_cpu_constants.FLAG_CONSTRAINT_DOCK = false;
    // Large box so no penalty
    g_cpu_constants.BOX_X_LO = -100; g_cpu_constants.BOX_X_HI = 100;
    g_cpu_constants.BOX_Y_LO = -100; g_cpu_constants.BOX_Y_HI = 100;
    g_cpu_constants.BOX_Z_LO = -100; g_cpu_constants.BOX_Z_HI = 100;
    g_cpu_constants.PENALTY_SLOPE = 10.0;
    init_default_atom_type_props(g_cpu_constants.atp);
}


// ==================== TEST CASES ====================

TEST_CASE("bias: BT_NO produces zero bias energy", "[cpu][bias]") {
    setup_bias_constants(BT_NO);

    // Atom at (0,0,0) with one bias entry — but bias disabled
    Bias b;
    b.i = 0;
    b.param[0] = 5.0; b.param[1] = 0; b.param[2] = 0;
    b.param[3] = 1.0; b.param[4] = 0.0;

    MinimalBiasSetup s(0, 0, 0, VN_TYPE_C_H, {b});
    Real e = s.run_energy();

    // No bias contribution (and no vina pairs), so energy == 0
    REQUIRE_THAT(e, WithinAbs(0.0, 1e-6));
}


TEST_CASE("bias: BT_ALIGN single atom at reference gives low energy", "[cpu][bias]") {
    setup_bias_constants(BT_ALIGN, 0.1);

    // Reference at (5,0,0), atom also at (5,0,0)  →  r = 0
    Bias b;
    b.i = 0;
    b.param[0] = 5.0; b.param[1] = 0; b.param[2] = 0;
    b.param[3] = 1.0; // a_qt = 1
    b.param[4] = 0.0; // unused for BT_ALIGN

    MinimalBiasSetup s_on(5, 0, 0, VN_TYPE_C_H, {b});
    Real e_on = s_on.run_energy();

    // Now place atom 3 Å away  →  r = 3, sigmoid saturated
    MinimalBiasSetup s_far(2, 0, 0, VN_TYPE_C_H, {b});
    Real e_far = s_far.run_energy();

    INFO("e_on_target=" << e_on << "  e_far=" << e_far);
    // Far atom has higher energy (sigmoid penalty)
    REQUIRE(e_far > e_on);
}


TEST_CASE("bias: BT_ALIGN sigmoid force profile", "[cpu][bias]") {
    // Verify force is concentrated around c_a=1 Å and vanishes > 2 Å.
    // eval_ef_zalign: E = a_qt * k_a / (1 + exp(s_a*(c_a - r)))
    //   s_a=3, c_a=1, k_a=30 (non-H), bias_k scales a_qt
    setup_bias_constants(BT_ALIGN, 0.1);

    Bias b;
    b.i = 0;
    b.param[0] = 0; b.param[1] = 0; b.param[2] = 0; // reference at origin
    b.param[3] = 1.0; // a_qt
    b.param[4] = 0.0;

    auto get_force_x = [&](Real dist) -> Real {
        MinimalBiasSetup s(dist, 0, 0, VN_TYPE_C_H, {b});
        s.run_energy();
        return s.forces[0]; // force on atom in x
    };

    Real f_at_0_5 = std::abs(get_force_x(0.5));
    Real f_at_1_0 = std::abs(get_force_x(1.0));
    Real f_at_1_5 = std::abs(get_force_x(1.5));
    Real f_at_3_0 = std::abs(get_force_x(3.0));
    Real f_at_5_0 = std::abs(get_force_x(5.0));

    INFO("f@0.5=" << f_at_0_5 << " f@1.0=" << f_at_1_0
         << " f@1.5=" << f_at_1_5 << " f@3.0=" << f_at_3_0
         << " f@5.0=" << f_at_5_0);

    // Force peaks near c_a=1.0
    REQUIRE(f_at_1_0 > f_at_0_5);
    REQUIRE(f_at_1_0 > f_at_1_5);

    // Force decays strongly beyond 2 Å
    REQUIRE(f_at_3_0 < f_at_1_0 * 0.05); // < 5% of peak
    REQUIRE(f_at_5_0 < f_at_1_0 * 0.001); // < 0.1% of peak
}


TEST_CASE("bias: BT_ALIGN allows translation gradient (matches GPU)", "[cpu][bias]") {
    setup_bias_constants(BT_ALIGN, 0.1);

    // Atom 2 Å from reference → non-zero force
    Bias b;
    b.i = 0;
    b.param[0] = 0; b.param[1] = 0; b.param[2] = 0;
    b.param[3] = 1.0; b.param[4] = 0.0;

    MinimalBiasSetup s(2, 0, 0, VN_TYPE_C_H, {b});
    Real e = s.run_energy();

    // Forces should be non-zero (atom not at reference)
    REQUIRE(std::abs(s.forces[0]) > 0.001);

    // Translation gradient is NOT frozen — matches original GPU behavior
    FlexPoseGradient g;
    std::vector<Real> dihedrals_g;
    g.dihedrals_g = dihedrals_g.data();
    cal_grad_cpu(s.forces.data(), s.topo, &s.pose, &g);

    // center_g should reflect the bias force (non-zero)
    REQUIRE(std::abs(g.center_g[0]) > 0.001);
}


TEST_CASE("bias: BT_POS Gaussian energy decays", "[cpu][bias]") {
    setup_bias_constants(BT_POS, 0.1);

    // BT_POS uses eval_ef_pos: E = V_set * exp(-|r|^2 / r2)
    Bias b;
    b.i = 0;
    b.param[0] = 0; b.param[1] = 0; b.param[2] = 0; // reference at origin
    b.param[3] = 1.0; // V_set
    b.param[4] = 1.0; // r2 (sigma^2)

    MinimalBiasSetup s_close(0.5, 0, 0, VN_TYPE_C_H, {b});
    Real e_close = s_close.run_energy();

    MinimalBiasSetup s_far(3.0, 0, 0, VN_TYPE_C_H, {b});
    Real e_far = s_far.run_energy();

    INFO("e_close=" << e_close << "  e_far=" << e_far);
    // BT_POS: energy is highest at r=0 and decays — close atom has more energy
    REQUIRE(e_close > e_far);
}


TEST_CASE("bias: BT_ALIGN hydrogen gets much weaker force", "[cpu][bias]") {
    setup_bias_constants(BT_ALIGN, 0.1);

    Bias b;
    b.i = 0;
    b.param[0] = 0; b.param[1] = 0; b.param[2] = 0;
    b.param[3] = 1.0; b.param[4] = 0.0;

    // Heavy atom (k_a = 30)
    MinimalBiasSetup s_c(1.0, 0, 0, VN_TYPE_C_H, {b});
    s_c.run_energy();
    Real f_c = std::abs(s_c.forces[0]);

    // Hydrogen (k_a = 0.1)
    MinimalBiasSetup s_h(1.0, 0, 0, VN_TYPE_H, {b});
    s_h.run_energy();
    Real f_h = std::abs(s_h.forces[0]);

    INFO("f_carbon=" << f_c << "  f_hydrogen=" << f_h);
    REQUIRE(f_c > f_h * 100); // k_a ratio = 30/0.1 = 300x
}


TEST_CASE("bias: multiple bias entries on same atom are summed", "[cpu][bias]") {
    setup_bias_constants(BT_ALIGN, 0.1);

    // One bias entry
    Bias b1;
    b1.i = 0;
    b1.param[0] = 2.0; b1.param[1] = 0; b1.param[2] = 0;
    b1.param[3] = 1.0; b1.param[4] = 0.0;

    MinimalBiasSetup s1(0, 0, 0, VN_TYPE_C_H, {b1});
    Real e1 = s1.run_energy();

    // Two identical bias entries → energy should double
    Bias b2 = b1;
    MinimalBiasSetup s2(0, 0, 0, VN_TYPE_C_H, {b1, b2});
    Real e2 = s2.run_energy();

    INFO("e_single=" << e1 << "  e_double=" << e2);
    REQUIRE_THAT(static_cast<double>(e2), WithinAbs(static_cast<double>(2.0 * e1), 1e-4));
}


// ---------------------------------------------------------------
// Helper: build an N-atom "molecule" with optional per-atom bias.
//
// All atoms placed in a line along x, no receptor, no intra/inter
// pairs, so the ONLY energy contribution comes from bias.
// ---------------------------------------------------------------
struct MultiAtomBiasSetup {
    std::vector<Real> coords;
    std::vector<Real> forces;

    FlexTopo topo;
    std::vector<int> vn_types_vec;
    std::vector<int> rotated_inds_vec;
    std::vector<int> axis_atoms_vec;
    std::vector<int> range_inds_vec;
    std::vector<Real> range_list_vec;
    std::vector<int> rotated_atoms_vec;

    FlexParamVina flex_param;
    std::vector<int> bias_inds;
    std::vector<Real> bias_params;
    std::vector<int> atom_types_storage;

    FixMol fix_mol;
    FixParamVina fix_param;

    FlexPose pose;

    MultiAtomBiasSetup(const std::vector<Real>& atom_coords,
                       const std::vector<int>& atom_types,
                       const std::vector<Bias>& biases) {
        const int natom = static_cast<int>(atom_types.size());

        // -- coords & forces
        coords = atom_coords;
        forces.resize(natom * 3, 0.0);

        // -- FlexTopo
        topo.natom = natom;
        topo.ntorsion = 0;
        vn_types_vec = atom_types;
        topo.vn_types = vn_types_vec.data();
        topo.rotated_inds = nullptr;
        topo.axis_atoms = nullptr;
        topo.range_inds = nullptr;
        topo.range_list = nullptr;
        topo.rotated_atoms = nullptr;

        // -- FlexPose (center = centroid, identity rotation)
        pose.coords = coords.data();
        pose.center[0] = 0; pose.center[1] = 0; pose.center[2] = 0;
        for (int a = 0; a < natom; a++) {
            pose.center[0] += coords[a * 3 + 0];
            pose.center[1] += coords[a * 3 + 1];
            pose.center[2] += coords[a * 3 + 2];
        }
        pose.center[0] /= natom;
        pose.center[1] /= natom;
        pose.center[2] /= natom;
        pose.rot_vec[0] = 0; pose.rot_vec[1] = 0; pose.rot_vec[2] = 0;
        pose.energy = 0;
        pose.dihedrals = nullptr;

        // -- FlexParamVina: no intra/inter pairs
        flex_param.npair_intra = 0;
        flex_param.pairs_intra = nullptr;
        flex_param.npair_inter = 0;
        flex_param.pairs_inter = nullptr;
        flex_param.r1_plus_r2_intra = nullptr;
        flex_param.r1_plus_r2_inter = nullptr;
        atom_types_storage = atom_types;
        flex_param.atom_types = atom_types_storage.data();

        // -- Flatten biases into inds_bias / params_bias
        bias_inds.resize(natom * 2, 0);
        bias_params.clear();
        int last_i = -1;
        int offset = 0;
        for (const auto& b : biases) {
            if (b.i != last_i) {
                bias_inds[b.i * 2] = offset;
            }
            for (int k = 0; k < 5; k++) bias_params.push_back(b.param[k]);
            offset += 5;
            bias_inds[b.i * 2 + 1] = offset;
            last_i = b.i;
        }
        if (bias_params.empty()) bias_params.push_back(0);
        flex_param.terms.bias.inds = bias_inds.data();
        flex_param.terms.bias.params = bias_params.data();
        flex_param.terms.settings.bias_type = s_bias_type;
        flex_param.terms.settings.bias_k = s_bias_k;

        // -- FixMol (empty receptor)
        fix_mol.natom = 0;
        fix_mol.coords = nullptr;
        fix_param.atom_types = nullptr;
    }

    Real run_energy() {
        std::fill(forces.begin(), forces.end(), 0.0);
        return cal_e_f_cpu(&pose, topo, fix_mol, flex_param, fix_param, forces.data());
    }
};


// ==================== MULTI-ATOM TEST CASES ====================

TEST_CASE("bias: multi-atom only biased atoms contribute energy", "[cpu][bias]") {
    setup_bias_constants(BT_ALIGN, 0.1);

    // Bias on atoms 0 and 2, no bias on atom 1
    Bias b0; b0.i = 0;
    b0.param[0] = 0; b0.param[1] = 0; b0.param[2] = 0;
    b0.param[3] = 1.0; b0.param[4] = 0.0;

    Bias b2; b2.i = 2;
    b2.param[0] = 6; b2.param[1] = 0; b2.param[2] = 0;
    b2.param[3] = 1.0; b2.param[4] = 0.0;

    // 3-atom molecule: atom0 at (1,0,0), atom1 at (3,0,0), atom2 at (5,0,0)
    std::vector<Real> coords = {1,0,0, 3,0,0, 5,0,0};
    std::vector<int> types = {VN_TYPE_C_H, VN_TYPE_C_H, VN_TYPE_C_H};

    MultiAtomBiasSetup s3(coords, types, {b0, b2});
    Real e3 = s3.run_energy();

    // Individual atom energies via MinimalBiasSetup
    Bias b0s; b0s.i = 0;
    b0s.param[0] = 0; b0s.param[1] = 0; b0s.param[2] = 0;
    b0s.param[3] = 1.0; b0s.param[4] = 0.0;
    MinimalBiasSetup s_a0(1, 0, 0, VN_TYPE_C_H, {b0s});
    Real e_a0 = s_a0.run_energy();

    Bias b2s; b2s.i = 0;
    b2s.param[0] = 6; b2s.param[1] = 0; b2s.param[2] = 0;
    b2s.param[3] = 1.0; b2s.param[4] = 0.0;
    MinimalBiasSetup s_a2(5, 0, 0, VN_TYPE_C_H, {b2s});
    Real e_a2 = s_a2.run_energy();

    INFO("e3=" << e3 << " e_a0=" << e_a0 << " e_a2=" << e_a2);
    REQUIRE_THAT(static_cast<double>(e3),
                 WithinAbs(static_cast<double>(e_a0 + e_a2), 1e-4));
}


TEST_CASE("bias: multi-atom forces on unbiased atom are zero", "[cpu][bias]") {
    setup_bias_constants(BT_ALIGN, 0.1);

    Bias b0; b0.i = 0;
    b0.param[0] = 0; b0.param[1] = 0; b0.param[2] = 0;
    b0.param[3] = 1.0; b0.param[4] = 0.0;

    Bias b2; b2.i = 2;
    b2.param[0] = 6; b2.param[1] = 0; b2.param[2] = 0;
    b2.param[3] = 1.0; b2.param[4] = 0.0;

    std::vector<Real> coords = {1,0,0, 3,0,0, 5,0,0};
    std::vector<int> types = {VN_TYPE_C_H, VN_TYPE_C_H, VN_TYPE_C_H};

    MultiAtomBiasSetup s(coords, types, {b0, b2});
    s.run_energy();

    // Atom 1 (index 1) has no bias → forces should be zero
    REQUIRE_THAT(static_cast<double>(s.forces[3]), WithinAbs(0.0, 1e-8));
    REQUIRE_THAT(static_cast<double>(s.forces[4]), WithinAbs(0.0, 1e-8));
    REQUIRE_THAT(static_cast<double>(s.forces[5]), WithinAbs(0.0, 1e-8));

    // Atoms 0 and 2 should have non-zero forces (displaced from reference)
    Real f0_mag = std::sqrt(s.forces[0]*s.forces[0] + s.forces[1]*s.forces[1] + s.forces[2]*s.forces[2]);
    Real f2_mag = std::sqrt(s.forces[6]*s.forces[6] + s.forces[7]*s.forces[7] + s.forces[8]*s.forces[8]);
    REQUIRE(f0_mag > 1e-4);
    REQUIRE(f2_mag > 1e-4);
}


TEST_CASE("bias: bias_k scaling is linear", "[cpu][bias]") {
    Bias b; b.i = 0;
    b.param[0] = 0; b.param[1] = 0; b.param[2] = 0;
    b.param[3] = 1.0; b.param[4] = 0.0;

    // 2-atom molecule, bias only on atom 0
    std::vector<Real> coords = {1,0,0, 5,0,0};
    std::vector<int> types = {VN_TYPE_C_H, VN_TYPE_C_H};

    setup_bias_constants(BT_ALIGN, 0.1);
    MultiAtomBiasSetup s1(coords, types, {b});
    Real e1 = s1.run_energy();

    setup_bias_constants(BT_ALIGN, 0.2);
    MultiAtomBiasSetup s2(coords, types, {b});
    Real e2 = s2.run_energy();

    INFO("e_k0.1=" << e1 << " e_k0.2=" << e2);
    REQUIRE(std::abs(e1) > 1e-6); // non-zero baseline
    REQUIRE_THAT(static_cast<double>(e2),
                 WithinAbs(static_cast<double>(2.0 * e1), std::abs(e1) * 0.01));
}


TEST_CASE("bias: BT_ALIGN 3D gradient points away from reference", "[cpu][bias]") {
    setup_bias_constants(BT_ALIGN, 0.1);

    // Reference at (5,3,2), atom displaced to (6,1,4)
    Bias b; b.i = 0;
    b.param[0] = 5; b.param[1] = 3; b.param[2] = 2;
    b.param[3] = 1.0; b.param[4] = 0.0;

    std::vector<Real> coords = {6,1,4, 10,0,0}; // atom 0 displaced, atom 1 unbiased
    std::vector<int> types = {VN_TYPE_C_H, VN_TYPE_C_H};

    MultiAtomBiasSetup s(coords, types, {b});
    s.run_energy();

    // Displacement from atom toward reference: (5-6, 3-1, 2-4) = (-1, 2, -2)
    Real dx = 5 - 6, dy = 3 - 1, dz = 2 - 4;
    Real gx = s.forces[0], gy = s.forces[1], gz = s.forces[2];

    // BT_ALIGN eval_ef_zalign accumulates gradient, not force.
    // Gradient should point away from reference (energy decreases toward reference).
    // So dot(gradient, disp_toward_ref) < 0.
    Real dot = gx * dx + gy * dy + gz * dz;
    INFO("grad=(" << gx << "," << gy << "," << gz << ") disp=(" << dx << "," << dy << "," << dz << ") dot=" << dot);
    REQUIRE(dot < 0);
}


TEST_CASE("bias: BT_POS 3D force points toward reference", "[cpu][bias]") {
    setup_bias_constants(BT_POS, 0.1);

    // Reference at (5,3,2), atom displaced to (6,1,4)
    Bias b; b.i = 0;
    b.param[0] = 5; b.param[1] = 3; b.param[2] = 2;
    b.param[3] = 1.0; b.param[4] = 1.0; // V_set=1, r2=1

    std::vector<Real> coords = {6,1,4, 10,0,0};
    std::vector<int> types = {VN_TYPE_C_H, VN_TYPE_C_H};

    MultiAtomBiasSetup s(coords, types, {b});
    s.run_energy();

    Real dx = 5 - 6, dy = 3 - 1, dz = 2 - 4;
    Real fx = s.forces[0], fy = s.forces[1], fz = s.forces[2];

    Real dot = fx * dx + fy * dy + fz * dz;
    INFO("force=(" << fx << "," << fy << "," << fz << ") disp=(" << dx << "," << dy << "," << dz << ") dot=" << dot);
    REQUIRE(dot > 0);
}


TEST_CASE("bias: atom_id gap in flattening produces correct inds_bias", "[cpu][bias]") {
    setup_bias_constants(BT_ALIGN, 0.1);

    // 3-atom molecule with bias on atoms 0 and 2 (gap at 1)
    Bias b0; b0.i = 0;
    b0.param[0] = 0; b0.param[1] = 0; b0.param[2] = 0;
    b0.param[3] = 1.0; b0.param[4] = 0.0;

    Bias b2; b2.i = 2;
    b2.param[0] = 6; b2.param[1] = 0; b2.param[2] = 0;
    b2.param[3] = 1.0; b2.param[4] = 0.0;

    std::vector<Real> coords = {1,0,0, 3,0,0, 5,0,0};
    std::vector<int> types = {VN_TYPE_C_H, VN_TYPE_C_H, VN_TYPE_C_H};

    MultiAtomBiasSetup s(coords, types, {b0, b2});

    // Atom 1 (gap): inds_bias[1*2] == inds_bias[1*2+1] == 0 (no bias range)
    REQUIRE(s.bias_inds[1 * 2] == 0);
    REQUIRE(s.bias_inds[1 * 2 + 1] == 0);

    // Atom 0: start=0, end=5 (one entry of 5 floats)
    REQUIRE(s.bias_inds[0 * 2] == 0);
    REQUIRE(s.bias_inds[0 * 2 + 1] == 5);

    // Atom 2: start=5, end=10 (one entry of 5 floats)
    REQUIRE(s.bias_inds[2 * 2] == 5);
    REQUIRE(s.bias_inds[2 * 2 + 1] == 10);

    // Run energy to make sure it doesn't crash with gaps
    Real e = s.run_energy();
    REQUIRE(std::isfinite(e));
}
