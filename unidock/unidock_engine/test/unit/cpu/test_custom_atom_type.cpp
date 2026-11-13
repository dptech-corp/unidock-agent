//
// Unit tests for custom (injected) atom types via data-driven VDW radii & AtomTypeProps.
//
// Verifies that the CPU scoring pipeline is fully data-driven:
//   - Custom type IDs beyond the original 21 (e.g., ID=21, 22) work without crash
//   - Custom VDW radii are used in inter-molecular scoring (cal_e_f_cpu)
//   - Custom VDW radii are stored correctly in the spatial grid
//   - Custom boolean properties (hydrophobic, donor, acceptor) activate the correct scoring terms
//   - Changing VDW radius produces a different energy (proving the hardcoded array is not used)
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

// ========== Custom type IDs ==========
static constexpr int CUSTOM_TYPE_A = 21;  // First custom type (beyond original 21)
static constexpr int CUSTOM_TYPE_B = 22;  // Second custom type

// ========== Helper: Set up g_cpu_constants with defaults + custom types ==========
// Starts from standard 21 types, then adds custom types at IDs 21 and 22.
static void init_custom_atp(Real radius_a, Real radius_b,
                            bool a_hydrophobic, bool a_donor, bool a_acceptor,
                            bool b_hydrophobic, bool b_donor, bool b_acceptor) {
    // Start with standard defaults
    init_default_atom_type_props(g_cpu_constants.atp);
    int n_default = sizeof(VN_VDW_RADII) / sizeof(VN_VDW_RADII[0]);
    for (int i = 0; i < n_default; i++) g_cpu_constants.vdw_radii[i] = VN_VDW_RADII[i];
    for (int i = n_default; i < MAX_VN_TYPES; i++) g_cpu_constants.vdw_radii[i] = Real(0);

    // Extend: custom type A
    g_cpu_constants.vdw_radii[CUSTOM_TYPE_A] = radius_a;
    g_cpu_constants.atp.is_hydrogen[CUSTOM_TYPE_A] = false;
    g_cpu_constants.atp.is_hydrophobic[CUSTOM_TYPE_A] = a_hydrophobic;
    g_cpu_constants.atp.is_donor[CUSTOM_TYPE_A] = a_donor;
    g_cpu_constants.atp.is_acceptor[CUSTOM_TYPE_A] = a_acceptor;

    // Extend: custom type B
    g_cpu_constants.vdw_radii[CUSTOM_TYPE_B] = radius_b;
    g_cpu_constants.atp.is_hydrogen[CUSTOM_TYPE_B] = false;
    g_cpu_constants.atp.is_hydrophobic[CUSTOM_TYPE_B] = b_hydrophobic;
    g_cpu_constants.atp.is_donor[CUSTOM_TYPE_B] = b_donor;
    g_cpu_constants.atp.is_acceptor[CUSTOM_TYPE_B] = b_acceptor;

    g_cpu_constants.atp.num_types = 23;
}


// ========================== SECTION 1: Storage tests ==========================

TEST_CASE("custom_atom_type: radii stored correctly in g_cpu_constants", "[custom_type][cpu]") {
    init_custom_atp(3.0, 2.5, false, false, false, false, false, false);

    REQUIRE(g_cpu_constants.vdw_radii[CUSTOM_TYPE_A] == Approx(3.0));
    REQUIRE(g_cpu_constants.vdw_radii[CUSTOM_TYPE_B] == Approx(2.5));
    // Standard types unchanged
    REQUIRE(g_cpu_constants.vdw_radii[VN_TYPE_H] == Approx(1.1));
    REQUIRE(g_cpu_constants.vdw_radii[VN_TYPE_C_H] == Approx(1.9));
    REQUIRE(g_cpu_constants.vdw_radii[VN_TYPE_O_A] == Approx(1.7));
}


TEST_CASE("custom_atom_type: boolean props stored correctly", "[custom_type][cpu]") {
    init_custom_atp(3.0, 2.5,
                    /*a: hydrophobic*/ true, /*donor*/ false, /*acceptor*/ false,
                    /*b: hydrophobic*/ false, /*donor*/ true, /*acceptor*/ true);

    REQUIRE(g_cpu_constants.atp.is_hydrogen[CUSTOM_TYPE_A] == false);
    REQUIRE(g_cpu_constants.atp.is_hydrophobic[CUSTOM_TYPE_A] == true);
    REQUIRE(g_cpu_constants.atp.is_donor[CUSTOM_TYPE_A] == false);
    REQUIRE(g_cpu_constants.atp.is_acceptor[CUSTOM_TYPE_A] == false);

    REQUIRE(g_cpu_constants.atp.is_hydrogen[CUSTOM_TYPE_B] == false);
    REQUIRE(g_cpu_constants.atp.is_hydrophobic[CUSTOM_TYPE_B] == false);
    REQUIRE(g_cpu_constants.atp.is_donor[CUSTOM_TYPE_B] == true);
    REQUIRE(g_cpu_constants.atp.is_acceptor[CUSTOM_TYPE_B] == true);

    REQUIRE(g_cpu_constants.atp.num_types == 23);
}


// ========================== SECTION 2: Spatial grid tests ==========================

TEST_CASE("custom_atom_type: spatial grid stores custom radius", "[custom_type][cpu][grid]") {
    init_custom_atp(3.0, 2.5, false, false, false, false, false, false);

    // Build grid with one receptor atom of custom type A
    std::vector<Real> coords = {5.0, 5.0, 5.0};
    std::vector<int> types = {CUSTOM_TYPE_A};

    free_spatial_grid(g_cpu_constants.receptor_grid);
    build_spatial_grid(g_cpu_constants.receptor_grid,
                       coords.data(), types.data(),
                       1, g_cpu_constants.atp,
                       g_cpu_constants.vdw_radii);

    REQUIRE(g_cpu_constants.receptor_grid.built == true);
    REQUIRE(g_cpu_constants.receptor_grid.total_atoms == 1);
    // The stored radius should be the custom value, not any hardcoded one
    REQUIRE(g_cpu_constants.receptor_grid.atoms[0].radius == Approx(3.0));
    REQUIRE(g_cpu_constants.receptor_grid.atoms[0].type == CUSTOM_TYPE_A);

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


TEST_CASE("custom_atom_type: grid with mixed standard + custom types", "[custom_type][cpu][grid]") {
    init_custom_atp(3.0, 2.5, false, false, false, false, false, false);

    // Two receptor atoms: one standard O_A, one custom type B
    std::vector<Real> coords = {1.0, 1.0, 1.0,  2.0, 2.0, 2.0};
    std::vector<int> types = {VN_TYPE_O_A, CUSTOM_TYPE_B};

    free_spatial_grid(g_cpu_constants.receptor_grid);
    build_spatial_grid(g_cpu_constants.receptor_grid,
                       coords.data(), types.data(),
                       2, g_cpu_constants.atp,
                       g_cpu_constants.vdw_radii);

    REQUIRE(g_cpu_constants.receptor_grid.total_atoms == 2);

    // Find each atom by type and verify its radius
    bool found_oa = false, found_custom = false;
    for (int i = 0; i < 2; i++) {
        if (g_cpu_constants.receptor_grid.atoms[i].type == VN_TYPE_O_A) {
            REQUIRE(g_cpu_constants.receptor_grid.atoms[i].radius == Approx(1.7));
            found_oa = true;
        }
        if (g_cpu_constants.receptor_grid.atoms[i].type == CUSTOM_TYPE_B) {
            REQUIRE(g_cpu_constants.receptor_grid.atoms[i].radius == Approx(2.5));
            found_custom = true;
        }
    }
    REQUIRE(found_oa);
    REQUIRE(found_custom);

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


// ========================== SECTION 3: Vina eval_ef with custom types ==========================

TEST_CASE("custom_atom_type: eval_ef with custom hydrophobic pair", "[custom_type][cpu][eval_ef]") {
    init_custom_atp(3.0, 2.5,
                    /*a: hydrophobic*/ true, /*donor*/ false, /*acceptor*/ false,
                    /*b:*/ false, false, false);

    Vina scorer;

    // Custom type A (hydrophobic) + standard C_H (hydrophobic)
    // d = distance - r1 - r2.  Pick d=1.0 (inside hydrophobic range [0.5, 1.5])
    Real d = Real(1.0);
    Real f1 = 0;
    Real e1 = scorer.eval_ef(d, CUSTOM_TYPE_A, VN_TYPE_C_H, &f1, g_cpu_constants.atp);

    // Also get decomposition to verify hydrophobic term is active
    VinaTerms terms = scorer.eval_ef_decomp(d, CUSTOM_TYPE_A, VN_TYPE_C_H, g_cpu_constants.atp);
    REQUIRE(terms.hydrophobic != Approx(0.0));

    // Compare: two non-hydrophobic types at same d should NOT have hydrophobic term
    VinaTerms terms_no_hp = scorer.eval_ef_decomp(d, VN_TYPE_N_P, VN_TYPE_O_P, g_cpu_constants.atp);
    REQUIRE(terms_no_hp.hydrophobic == Approx(0.0));

    // The hydrophobic pair should have more negative energy
    REQUIRE(terms.total < terms_no_hp.total);
}


TEST_CASE("custom_atom_type: eval_ef with custom hbond donor + acceptor", "[custom_type][cpu][eval_ef]") {
    init_custom_atp(3.0, 2.5,
                    /*a:*/ false, false, false,
                    /*b: donor+acceptor*/ false, /*donor*/ true, /*acceptor*/ false);

    Vina scorer;

    // Custom type B (donor) + standard O_A (acceptor) → hbond should be possible
    Real d = Real(-0.5);  // in hbond range [-0.7, 0]
    VinaTerms terms = scorer.eval_ef_decomp(d, CUSTOM_TYPE_B, VN_TYPE_O_A, g_cpu_constants.atp);
    REQUIRE(terms.hbond < 0);  // favorable hbond

    // Custom type A (not donor/acceptor) + O_A → no hbond
    VinaTerms terms_no_hb = scorer.eval_ef_decomp(d, CUSTOM_TYPE_A, VN_TYPE_O_A, g_cpu_constants.atp);
    REQUIRE(terms_no_hb.hbond == Approx(0.0));
}


TEST_CASE("custom_atom_type: eval_ef with custom acceptor", "[custom_type][cpu][eval_ef]") {
    init_custom_atp(3.0, 2.5,
                    /*a: acceptor*/ false, false, /*acceptor*/ true,
                    /*b:*/ false, false, false);

    Vina scorer;

    // Standard N_D (donor) + custom type A (acceptor) → hbond should be possible
    Real d = Real(-0.3);
    VinaTerms terms = scorer.eval_ef_decomp(d, VN_TYPE_N_D, CUSTOM_TYPE_A, g_cpu_constants.atp);
    REQUIRE(terms.hbond < 0);
}


// ========================== SECTION 4: Full cal_e_f_cpu with custom type ==========================

// Helper: build minimal scoring setup for a 1-atom ligand vs 1-atom receptor
struct CustomScoreSetup {
    std::vector<Real> lig_coords;
    std::vector<Real> rec_coords;
    std::vector<int> lig_types_vec;
    std::vector<int> rec_types_vec;
    std::vector<Real> aux_f;

    FlexPose pose;
    FlexTopo topo;
    FixMol fix_mol;
    FlexParamVina flex_param;
    FixParamVina fix_param;

    // intra pairs (empty)
    std::vector<int> pairs_intra;
    std::vector<Real> r1_plus_r2_intra;
    // bias (none)
    std::vector<int> inds_bias;
    std::vector<Real> params_bias;

    void build(int lig_type, Real lx, Real ly, Real lz,
               int rec_type, Real rx, Real ry, Real rz) {
        lig_coords = {lx, ly, lz};
        rec_coords = {rx, ry, rz};
        lig_types_vec = {lig_type};
        rec_types_vec = {rec_type};
        aux_f.resize(3, 0);

        // Pose
        pose.coords = lig_coords.data();
        pose.dihedrals = nullptr;
        pose.center[0] = lx; pose.center[1] = ly; pose.center[2] = lz;
        pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
        pose.energy = 0;

        // Topo
        topo.natom = 1;
        topo.ntorsion = 0;
        topo.vn_types = lig_types_vec.data();
        topo.axis_atoms = nullptr;
        topo.range_inds = nullptr;
        topo.range_list = nullptr;
        topo.rotated_inds = nullptr;
        topo.rotated_atoms = nullptr;

        // Fix mol
        fix_mol.natom = 1;
        fix_mol.coords = rec_coords.data();

        // Flex param: no intra pairs, 1 atom
        flex_param.npair_intra = 0;
        flex_param.pairs_intra = nullptr;
        flex_param.r1_plus_r2_intra = nullptr;
        flex_param.npair_inter = 0;
        flex_param.pairs_inter = nullptr;
        flex_param.r1_plus_r2_inter = nullptr;
        flex_param.atom_types = lig_types_vec.data();

        // Bias: none
        inds_bias = {0, 0};
        params_bias.clear();
        flex_param.terms.bias.inds = inds_bias.data();
        flex_param.terms.bias.params = nullptr;

        // Fix param
        fix_param.atom_types = rec_types_vec.data();

        // Build spatial grid for the receptor atom
        free_spatial_grid(g_cpu_constants.receptor_grid);
        build_spatial_grid(g_cpu_constants.receptor_grid,
                           rec_coords.data(), rec_types_vec.data(),
                           1, g_cpu_constants.atp,
                           g_cpu_constants.vdw_radii);
    }

    Real score() {
        std::fill(aux_f.begin(), aux_f.end(), 0);
        return cal_e_f_cpu(&pose, topo, fix_mol, flex_param, fix_param, aux_f.data());
    }

    ~CustomScoreSetup() {
        free_spatial_grid(g_cpu_constants.receptor_grid);
    }
};


TEST_CASE("custom_atom_type: cal_e_f_cpu uses custom VDW radius for ligand", "[custom_type][cpu][scoring]") {
    // Custom type A with radius=3.0 (much larger than standard C_P=1.9)
    init_custom_atp(3.0, 2.5, false, false, false, false, false, false);

    // Set up box
    g_cpu_constants.BOX_X_LO = -20; g_cpu_constants.BOX_X_HI = 20;
    g_cpu_constants.BOX_Y_LO = -20; g_cpu_constants.BOX_Y_HI = 20;
    g_cpu_constants.BOX_Z_LO = -20; g_cpu_constants.BOX_Z_HI = 20;
    g_cpu_constants.PENALTY_SLOPE = 10.0;

    // Score with custom type A (radius=3.0) as ligand, O_A (radius=1.7) as receptor
    // Distance r = 5.0, so d = 5.0 - 3.0 - 1.7 = 0.3
    CustomScoreSetup setup1;
    setup1.build(CUSTOM_TYPE_A, 0, 0, 0,  VN_TYPE_O_A, 5.0, 0, 0);
    Real e_custom = setup1.score();

    // Score with standard C_P (radius=1.9) as ligand at same positions
    // Distance r = 5.0, so d = 5.0 - 1.9 - 1.7 = 1.4
    init_custom_atp(3.0, 2.5, false, false, false, false, false, false);
    g_cpu_constants.BOX_X_LO = -20; g_cpu_constants.BOX_X_HI = 20;
    g_cpu_constants.BOX_Y_LO = -20; g_cpu_constants.BOX_Y_HI = 20;
    g_cpu_constants.BOX_Z_LO = -20; g_cpu_constants.BOX_Z_HI = 20;
    g_cpu_constants.PENALTY_SLOPE = 10.0;

    CustomScoreSetup setup2;
    setup2.build(VN_TYPE_C_P, 0, 0, 0,  VN_TYPE_O_A, 5.0, 0, 0);
    Real e_standard = setup2.score();

    // The energies MUST differ because different VDW radii → different surface distance
    // If hardcoded VN_VDW_RADII were still used, custom type A would crash (OOB) or give wrong result
    REQUIRE(e_custom != Approx(e_standard).margin(1e-4));

    // Verify the direction: larger radius → smaller surface distance → more repulsive at close range
    // d_custom = 0.3, d_standard = 1.4
    // At d=0.3: gauss1 ~ exp(-0.36) ≈ 0.70, at d=1.4: gauss1 ~ exp(-3.92) ≈ 0.02
    // So custom should have more negative energy (gauss1 is attractive)
    // Actually it depends on the balance of terms. Let's just verify they differ.
    INFO("e_custom=" << e_custom << " e_standard=" << e_standard);
}


TEST_CASE("custom_atom_type: same geometry, different radius → different energy", "[custom_type][cpu][scoring]") {
    // This is the KEY test: prove that changing vdw_radii[CUSTOM_TYPE_A] changes the energy
    g_cpu_constants.BOX_X_LO = -20; g_cpu_constants.BOX_X_HI = 20;
    g_cpu_constants.BOX_Y_LO = -20; g_cpu_constants.BOX_Y_HI = 20;
    g_cpu_constants.BOX_Z_LO = -20; g_cpu_constants.BOX_Z_HI = 20;
    g_cpu_constants.PENALTY_SLOPE = 10.0;

    // Run 1: custom type A with radius = 1.5
    init_custom_atp(1.5, 2.5, false, false, false, false, false, false);
    CustomScoreSetup s1;
    s1.build(CUSTOM_TYPE_A, 0, 0, 0,  VN_TYPE_C_P, 4.0, 0, 0);
    Real e1 = s1.score();

    // Run 2: custom type A with radius = 3.0 (same positions)
    init_custom_atp(3.0, 2.5, false, false, false, false, false, false);
    g_cpu_constants.BOX_X_LO = -20; g_cpu_constants.BOX_X_HI = 20;
    g_cpu_constants.BOX_Y_LO = -20; g_cpu_constants.BOX_Y_HI = 20;
    g_cpu_constants.BOX_Z_LO = -20; g_cpu_constants.BOX_Z_HI = 20;
    g_cpu_constants.PENALTY_SLOPE = 10.0;

    CustomScoreSetup s2;
    s2.build(CUSTOM_TYPE_A, 0, 0, 0,  VN_TYPE_C_P, 4.0, 0, 0);
    Real e2 = s2.score();

    // Different radius → different surface distance → different energy
    // r = 4.0; d1 = 4.0 - 1.5 - 1.9 = 0.6;  d2 = 4.0 - 3.0 - 1.9 = -0.9
    REQUIRE(e1 != Approx(e2).margin(1e-4));

    // Verify manually: d2 < 0 triggers repulsion, d1 > 0 does not
    Vina scorer;
    Real f_dummy;
    Real e1_manual = scorer.eval_ef(Real(0.6), CUSTOM_TYPE_A, VN_TYPE_C_P, &f_dummy, g_cpu_constants.atp);
    Real e2_manual = scorer.eval_ef(Real(-0.9), CUSTOM_TYPE_A, VN_TYPE_C_P, &f_dummy, g_cpu_constants.atp);

    // d2=-0.9 has repulsion (d^2 * weight_repulsion), d1=0.6 does not
    REQUIRE(e2_manual > e1_manual);  // more repulsive with larger radius
    REQUIRE(e2 > e1);  // same direction in full scoring

    INFO("e1(r=1.5)=" << e1 << " e2(r=3.0)=" << e2);
    INFO("e1_manual(d=0.6)=" << e1_manual << " e2_manual(d=-0.9)=" << e2_manual);
}


TEST_CASE("custom_atom_type: cal_e_f_cpu energy matches manual eval_ef", "[custom_type][cpu][scoring]") {
    // Verify exact agreement between cal_e_f_cpu and manual eval_ef for custom type
    init_custom_atp(2.2, 2.5, false, false, false, false, false, false);

    g_cpu_constants.BOX_X_LO = -20; g_cpu_constants.BOX_X_HI = 20;
    g_cpu_constants.BOX_Y_LO = -20; g_cpu_constants.BOX_Y_HI = 20;
    g_cpu_constants.BOX_Z_LO = -20; g_cpu_constants.BOX_Z_HI = 20;
    g_cpu_constants.PENALTY_SLOPE = 10.0;

    // Ligand at origin (custom type A, radius=2.2), receptor at (4, 0, 0) (C_P, radius=1.9)
    Real r = 4.0;
    Real d = r - 2.2 - 1.9;  // d = -0.1

    Vina scorer;
    scorer.set_weights(g_cpu_constants.weights);
    Real f_dummy;
    Real e_manual = scorer.eval_ef(d, CUSTOM_TYPE_A, VN_TYPE_C_P, &f_dummy, g_cpu_constants.atp);

    CustomScoreSetup setup;
    setup.build(CUSTOM_TYPE_A, 0, 0, 0,  VN_TYPE_C_P, 4.0, 0, 0);
    Real e_full = setup.score();

    // cal_e_f_cpu should match eval_ef (no penalty since inside box, no intra pairs)
    REQUIRE(e_full == Approx(e_manual).margin(1e-5));
}


// ========================== SECTION 5: hbond_angle_correction with custom donor ==========================

TEST_CASE("custom_atom_type: hbond_angle_correction with custom donor type", "[custom_type][cpu][hbond]") {
    // Custom type B as donor (radius=2.5)
    init_custom_atp(3.0, 2.5,
                    /*a:*/ false, false, false,
                    /*b: donor*/ false, /*donor*/ true, /*acceptor*/ false);

    // Bad angle geometry: donor at origin, H at (0,0,1), receptor O_A at (0,0,-2)
    // r = 2.0, d = 2.0 - 2.5(custom) - 1.7(O_A) = -2.2
    // This uses custom radius 2.5 in hbond_angle_correction_cpu
    std::vector<Real> rec_coords = {0, 0, -2};
    std::vector<int> rec_types = {VN_TYPE_O_A};

    free_spatial_grid(g_cpu_constants.receptor_grid);
    build_spatial_grid(g_cpu_constants.receptor_grid,
                       rec_coords.data(), rec_types.data(),
                       1, g_cpu_constants.atp,
                       g_cpu_constants.vdw_radii);

    // Build ligand: custom donor B at origin, H at (0,0,1), filler C_H at (10,10,10)
    std::vector<Real> lig_coords = {0, 0, 0,  0, 0, 1,  10, 10, 10};
    std::vector<int> lig_types = {CUSTOM_TYPE_B, VN_TYPE_H, VN_TYPE_C_H};

    FlexPose pose;
    pose.coords = lig_coords.data();
    pose.dihedrals = nullptr;
    pose.center[0] = 0; pose.center[1] = 0; pose.center[2] = 0;
    pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
    pose.energy = 0;

    FlexTopo topo;
    topo.natom = 3;
    topo.ntorsion = 0;
    topo.vn_types = lig_types.data();
    topo.axis_atoms = nullptr;
    topo.range_inds = nullptr;
    topo.range_list = nullptr;
    topo.rotated_inds = nullptr;
    topo.rotated_atoms = nullptr;

    FixMol fix;
    fix.natom = 1;
    fix.coords = rec_coords.data();

    Real corr = hbond_angle_correction_cpu(&pose, topo, fix);

    // Bad angle → positive correction (removing favorable hbond)
    REQUIRE(corr > 0.0);

    // Verify it matches eval_ef_decomp with the custom surface distance
    Real r = 2.0;
    Real d = r - g_cpu_constants.vdw_radii[CUSTOM_TYPE_B] - g_cpu_constants.vdw_radii[VN_TYPE_O_A];
    // d = 2.0 - 2.5 - 1.7 = -2.2 (deep inside hbond range → full hbond)

    Vina scorer;
    scorer.set_weights(g_cpu_constants.weights);
    VinaTerms terms = scorer.eval_ef_decomp(d, CUSTOM_TYPE_B, VN_TYPE_O_A, g_cpu_constants.atp);
    REQUIRE(terms.hbond < 0);
    REQUIRE(corr == Approx(-terms.hbond).margin(1e-6));

    free_spatial_grid(g_cpu_constants.receptor_grid);
}


TEST_CASE("custom_atom_type: hbond correction differs with different custom radius", "[custom_type][cpu][hbond]") {
    // Place receptor far enough that with one radius the hbond is partial,
    // and with another radius the hbond is full.
    //
    // O_A receptor at (0,0,-4.0), donor at origin, H at (0,0,1)
    // r = 4.0
    // With radius=2.5: d = 4.0 - 2.5 - 1.7 = -0.2 (partial hbond, d in [-0.7, 0])
    // With radius=1.0: d = 4.0 - 1.0 - 1.7 = 1.3  (d > 0, NO hbond)

    auto run_with_radius = [](Real radius) -> Real {
        init_custom_atp(3.0, radius,
                        false, false, false,
                        false, /*donor*/ true, false);

        std::vector<Real> rec_coords = {0, 0, -4};
        std::vector<int> rec_types = {VN_TYPE_O_A};
        free_spatial_grid(g_cpu_constants.receptor_grid);
        build_spatial_grid(g_cpu_constants.receptor_grid,
                           rec_coords.data(), rec_types.data(),
                           1, g_cpu_constants.atp,
                           g_cpu_constants.vdw_radii);

        std::vector<Real> lig_coords = {0, 0, 0,  0, 0, 1,  10, 10, 10};
        std::vector<int> lig_types = {CUSTOM_TYPE_B, VN_TYPE_H, VN_TYPE_C_H};

        FlexPose pose;
        pose.coords = lig_coords.data();
        pose.dihedrals = nullptr;
        pose.center[0] = 0; pose.center[1] = 0; pose.center[2] = 0;
        pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
        pose.energy = 0;

        FlexTopo topo;
        topo.natom = 3;
        topo.ntorsion = 0;
        topo.vn_types = lig_types.data();
        topo.axis_atoms = nullptr;
        topo.range_inds = nullptr;
        topo.range_list = nullptr;
        topo.rotated_inds = nullptr;
        topo.rotated_atoms = nullptr;

        FixMol fix;
        fix.natom = 1;
        fix.coords = rec_coords.data();

        Real corr = hbond_angle_correction_cpu(&pose, topo, fix);
        free_spatial_grid(g_cpu_constants.receptor_grid);
        return corr;
    };

    Real corr_large = run_with_radius(2.5);  // d = -0.2, partial hbond → positive correction
    Real corr_small = run_with_radius(1.0);  // d = 1.3, no hbond → zero correction

    REQUIRE(corr_large > 0.0);
    REQUIRE(corr_small == Approx(0.0).margin(1e-6));

    INFO("corr(r=2.5)=" << corr_large << " corr(r=1.0)=" << corr_small);
}


// ========================== SECTION 6: init_constants_cpu with custom vector ==========================

TEST_CASE("custom_atom_type: init_constants_cpu accepts extended vdw_radii vector", "[custom_type][cpu][init]") {
    // Simulate what DockTask would do: pass a 23-element radii vector
    DockParam dp;
    dp.constraint_docking = false;
    dp.box = {-10, 10, -10, 10, -10, 10};
    dp.tor_prec = 0.3;
    dp.box_prec = 2.0;
    dp.bias_type = BT_NO;
    dp.bias_k = 0.1;

    AtomTypeProps atp;
    init_default_atom_type_props(atp);
    // Extend to 23 types
    atp.is_hydrogen[21] = false;
    atp.is_hydrophobic[21] = true;
    atp.is_donor[21] = false;
    atp.is_acceptor[21] = false;
    atp.is_hydrogen[22] = false;
    atp.is_hydrophobic[22] = false;
    atp.is_donor[22] = true;
    atp.is_acceptor[22] = true;
    atp.num_types = 23;

    // Build radii vector: standard 21 + 2 custom
    std::vector<Real> radii(23);
    for (int i = 0; i < 21; i++) radii[i] = VN_VDW_RADII[i];
    radii[21] = Real(4.0);  // custom
    radii[22] = Real(0.8);  // custom

    init_constants_cpu(dp, atp, radii);

    // Verify standard types preserved
    REQUIRE(g_cpu_constants.vdw_radii[0] == Approx(1.1));   // H
    REQUIRE(g_cpu_constants.vdw_radii[2] == Approx(1.9));   // C_H
    REQUIRE(g_cpu_constants.vdw_radii[10] == Approx(1.7));  // O_A

    // Verify custom types stored
    REQUIRE(g_cpu_constants.vdw_radii[21] == Approx(4.0));
    REQUIRE(g_cpu_constants.vdw_radii[22] == Approx(0.8));

    // Verify boolean props
    REQUIRE(g_cpu_constants.atp.is_hydrophobic[21] == true);
    REQUIRE(g_cpu_constants.atp.is_donor[22] == true);
    REQUIRE(g_cpu_constants.atp.is_acceptor[22] == true);
    REQUIRE(g_cpu_constants.atp.num_types == 23);
}


TEST_CASE("custom_atom_type: init_constants_cpu with empty vector uses defaults", "[custom_type][cpu][init]") {
    DockParam dp;
    dp.constraint_docking = false;
    dp.box = {-10, 10, -10, 10, -10, 10};
    dp.tor_prec = 0.3;
    dp.box_prec = 2.0;
    dp.bias_type = BT_NO;
    dp.bias_k = 0.1;

    AtomTypeProps atp;
    init_default_atom_type_props(atp);

    // Pass empty vector → should use VN_VDW_RADII defaults
    init_constants_cpu(dp, atp, {});

    REQUIRE(g_cpu_constants.vdw_radii[0] == Approx(1.1));   // H
    REQUIRE(g_cpu_constants.vdw_radii[2] == Approx(1.9));   // C_H
    REQUIRE(g_cpu_constants.vdw_radii[20] == Approx(1.2));  // Met_D
    // Beyond default 21 types should be zero
    REQUIRE(g_cpu_constants.vdw_radii[21] == Approx(0.0));
}
