//
// Unit tests for custom (injected) GAFF2/FF parameters via data-driven FFMinData.
//
// Verifies that the CPU FF scoring pipeline is fully data-driven:
//   - Custom sigma/epsilon values produce different intra-molecular LJ energy
//   - Custom charges produce different Coulomb energy
//   - Custom torsion fc values produce different torsion energy
//   - Custom 1-4 pair parameters produce different 1-4 energy
//   - Changing parameters on real DMS data shifts the total energy
//
// These tests prove that the C++ engine does NOT use hardcoded FF parameters;
// it relies entirely on the sigma/epsilon/charge/torsion values passed in
// via FFMinData (populated from the DMS file at runtime).
//

#include <catch2/catch_amalgamated.hpp>
#include "cpu/score_ff_cpu.h"
#include "format/dms_reader.h"
#include "score/gaff2.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// ========== Helper: minimal 2-atom FFMinData for isolated LJ/Coulomb tests ==========
static FFMinData make_two_atom_ff(Real sig0, Real sig1, Real eps0, Real eps1,
                                  Real q0, Real q1) {
    FFMinData ff;
    ff.lig_atoms.natom = 2;
    ff.lig_atoms.sigma  = {sig0, sig1};
    ff.lig_atoms.epsilon = {eps0, eps1};
    ff.lig_atoms.charge  = {q0, q1};
    ff.lig_atoms.anum    = {6, 6};  // carbon-like
    ff.lig_atoms.x = {0, 0};
    ff.lig_atoms.y = {0, 0};
    ff.lig_atoms.z = {0, 0};
    ff.rec_atoms.natom = 0;  // no receptor
    ff.build_pair_lookup();
    ff.loaded = true;
    return ff;
}

// Helper: create FlexTopo + FlexPose + FixMol for a simple ligand
struct FFTestSetup {
    std::vector<Real> coords;
    std::vector<int> vn_types;
    std::vector<Real> aux_f;
    FlexPose pose;
    FlexTopo topo;
    FixMol fix_mol;

    void build(int natom) {
        vn_types.assign(natom, 0);
        aux_f.assign(natom * 3, 0);

        topo.natom = natom;
        topo.ntorsion = 0;
        topo.vn_types = vn_types.data();
        topo.axis_atoms = nullptr;
        topo.range_inds = nullptr;
        topo.range_list = nullptr;
        topo.rotated_inds = nullptr;
        topo.rotated_atoms = nullptr;

        pose.coords = coords.data();
        pose.dihedrals = nullptr;
        pose.center[0] = pose.center[1] = pose.center[2] = 0;
        pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
        pose.energy = 0;

        fix_mol.natom = 0;
        fix_mol.coords = nullptr;
    }

    Real score(const FFMinData& ff) {
        std::fill(aux_f.begin(), aux_f.end(), 0);
        return cal_e_f_ff_cpu(&pose, topo, fix_mol, ff, aux_f.data());
    }
};


// ========================== SECTION 1: LJ sigma/epsilon tests ==========================

TEST_CASE("custom_ff_param: different sigma produces different LJ energy", "[custom_ff][cpu]") {
    // Two atoms at distance 4.0 A, no exclusions, no 1-4 pairs
    // Changing sigma shifts the LJ energy.

    // Set up box so no box penalty
    g_cpu_constants.BOX_X_LO = -50; g_cpu_constants.BOX_X_HI = 50;
    g_cpu_constants.BOX_Y_LO = -50; g_cpu_constants.BOX_Y_HI = 50;
    g_cpu_constants.BOX_Z_LO = -50; g_cpu_constants.BOX_Z_HI = 50;
    g_cpu_constants.PENALTY_SLOPE = 10.0;

    Real r = 4.0;

    // Run 1: sigma = 1.9 A (GAFF2 "ca" carbon-like)
    FFMinData ff1 = make_two_atom_ff(1.9, 1.9, 0.086, 0.086, 0.0, 0.0);
    FFTestSetup s1;
    s1.coords = {0, 0, 0, r, 0, 0};
    s1.build(2);
    Real e1 = s1.score(ff1);

    // Run 2: sigma = 3.5 A (much larger)
    FFMinData ff2 = make_two_atom_ff(3.5, 3.5, 0.086, 0.086, 0.0, 0.0);
    FFTestSetup s2;
    s2.coords = {0, 0, 0, r, 0, 0};
    s2.build(2);
    Real e2 = s2.score(ff2);

    // Energies must differ because different sigma → different LJ potential
    REQUIRE(std::isfinite(e1));
    REQUIRE(std::isfinite(e2));
    REQUIRE(e1 != Catch::Approx(e2).margin(1e-4));

    // Verify manually: sig_ij = sigma (both same), E = 4*eps*[(sig/r)^12 - (sig/r)^6]
    Real sig1_ij = 1.9;
    Real sig2_ij = 3.5;
    Real eps = 0.086;
    Real inv_r = 1.0 / r;
    Real s6_1 = std::pow(sig1_ij * inv_r, 6);
    Real s6_2 = std::pow(sig2_ij * inv_r, 6);
    Real e1_manual = 4.0 * eps * (s6_1 * s6_1 - s6_1);
    Real e2_manual = 4.0 * eps * (s6_2 * s6_2 - s6_2);
    REQUIRE(e1 == Catch::Approx(e1_manual).margin(1e-3));
    REQUIRE(e2 == Catch::Approx(e2_manual).margin(1e-3));

    INFO("e1(sig=1.9)=" << e1 << " e2(sig=3.5)=" << e2);
}


TEST_CASE("custom_ff_param: different epsilon produces different LJ energy", "[custom_ff][cpu]") {
    g_cpu_constants.BOX_X_LO = -50; g_cpu_constants.BOX_X_HI = 50;
    g_cpu_constants.BOX_Y_LO = -50; g_cpu_constants.BOX_Y_HI = 50;
    g_cpu_constants.BOX_Z_LO = -50; g_cpu_constants.BOX_Z_HI = 50;
    g_cpu_constants.PENALTY_SLOPE = 10.0;

    Real r = 4.0;

    // Run 1: eps = 0.086 kcal/mol (GAFF2 "ca")
    FFMinData ff1 = make_two_atom_ff(1.9, 1.9, 0.086, 0.086, 0.0, 0.0);
    FFTestSetup s1;
    s1.coords = {0, 0, 0, r, 0, 0};
    s1.build(2);
    Real e1 = s1.score(ff1);

    // Run 2: eps = 0.5 kcal/mol (much larger well depth)
    FFMinData ff2 = make_two_atom_ff(1.9, 1.9, 0.5, 0.5, 0.0, 0.0);
    FFTestSetup s2;
    s2.coords = {0, 0, 0, r, 0, 0};
    s2.build(2);
    Real e2 = s2.score(ff2);

    REQUIRE(std::isfinite(e1));
    REQUIRE(std::isfinite(e2));
    REQUIRE(e1 != Catch::Approx(e2).margin(1e-4));

    // eps_ij = sqrt(eps_i * eps_j). Since both same: eps_ij = eps.
    // Energy scales linearly with eps → e2/e1 ≈ 0.5/0.086
    Real ratio = e2 / e1;
    REQUIRE(ratio == Catch::Approx(0.5 / 0.086).margin(0.01));

    INFO("e1(eps=0.086)=" << e1 << " e2(eps=0.5)=" << e2);
}


// ========================== SECTION 2: Coulomb charge tests ==========================

TEST_CASE("custom_ff_param: different charges produce different Coulomb energy", "[custom_ff][cpu]") {
    g_cpu_constants.BOX_X_LO = -50; g_cpu_constants.BOX_X_HI = 50;
    g_cpu_constants.BOX_Y_LO = -50; g_cpu_constants.BOX_Y_HI = 50;
    g_cpu_constants.BOX_Z_LO = -50; g_cpu_constants.BOX_Z_HI = 50;
    g_cpu_constants.PENALTY_SLOPE = 10.0;

    Real r = 5.0;
    // Use zero LJ to isolate Coulomb
    Real sig = 0.001;  // tiny sigma → negligible LJ at r=5
    Real eps = 0.0;

    // Run 1: opposite charges → attractive
    FFMinData ff1 = make_two_atom_ff(sig, sig, eps, eps, 0.5, -0.5);
    FFTestSetup s1;
    s1.coords = {0, 0, 0, r, 0, 0};
    s1.build(2);
    Real e1 = s1.score(ff1);

    // Run 2: same-sign charges → repulsive
    FFMinData ff2 = make_two_atom_ff(sig, sig, eps, eps, 0.5, 0.5);
    FFTestSetup s2;
    s2.coords = {0, 0, 0, r, 0, 0};
    s2.build(2);
    Real e2 = s2.score(ff2);

    REQUIRE(std::isfinite(e1));
    REQUIRE(std::isfinite(e2));

    // Opposite charges → negative energy; same charges → positive energy
    REQUIRE(e1 < 0);
    REQUIRE(e2 > 0);

    // They should be equal in magnitude (q1*q2 changes sign)
    REQUIRE(std::abs(e1 + e2) < 0.01);

    INFO("e1(+/-)=" << e1 << " e2(+/+)=" << e2);
}


// ========================== SECTION 3: Torsion parameter tests ==========================

TEST_CASE("custom_ff_param: different torsion fc produces different torsion energy", "[custom_ff][cpu]") {
    g_cpu_constants.BOX_X_LO = -50; g_cpu_constants.BOX_X_HI = 50;
    g_cpu_constants.BOX_Y_LO = -50; g_cpu_constants.BOX_Y_HI = 50;
    g_cpu_constants.BOX_Z_LO = -50; g_cpu_constants.BOX_Z_HI = 50;
    g_cpu_constants.PENALTY_SLOPE = 10.0;

    // 4-atom chain with ~90 degree dihedral (atom 0 up, atom 3 into screen)
    // Use zero sigma/eps/charge to isolate torsion energy
    FFMinData ff;
    ff.lig_atoms.natom = 4;
    ff.lig_atoms.sigma   = {0.001, 0.001, 0.001, 0.001};
    ff.lig_atoms.epsilon  = {0.0, 0.0, 0.0, 0.0};
    ff.lig_atoms.charge   = {0.0, 0.0, 0.0, 0.0};
    ff.lig_atoms.anum     = {6, 6, 6, 6};
    ff.lig_atoms.x = {0, 0, 0, 0};
    ff.lig_atoms.y = {0, 0, 0, 0};
    ff.lig_atoms.z = {0, 0, 0, 0};
    ff.rec_atoms.natom = 0;
    // Exclude all pairs (1-2 chain: 0-1, 1-2, 2-3 and 1-3: 0-2, 1-3)
    ff.lig_exclusions = {{0,1}, {1,2}, {2,3}, {0,2}, {1,3}, {0,3}};
    ff.build_pair_lookup();
    ff.loaded = true;

    // Torsion 1: fc3 = 2.0 kcal/mol (standard GAFF2 sp3-sp3)
    FFTorsion tor1;
    tor1.p0 = 0; tor1.p1 = 1; tor1.p2 = 2; tor1.p3 = 3;
    tor1.phi0 = 0;
    std::memset(tor1.fc, 0, sizeof(tor1.fc));
    tor1.fc[3] = 2.0;  // periodicity 3

    FFMinData ff1 = ff;
    ff1.lig_torsions = {tor1};

    FFTestSetup s1;
    s1.coords = {-1, 1, 0,   0, 0, 0,   1, 0, 0,   2, 0, 1};
    s1.build(4);
    Real e1 = s1.score(ff1);

    // Torsion 2: fc3 = 10.0 kcal/mol (much stronger barrier)
    FFTorsion tor2 = tor1;
    tor2.fc[3] = 10.0;

    FFMinData ff2 = ff;
    ff2.lig_torsions = {tor2};

    FFTestSetup s2;
    s2.coords = {-1, 1, 0,   0, 0, 0,   1, 0, 0,   2, 0, 1};
    s2.build(4);
    Real e2 = s2.score(ff2);

    REQUIRE(std::isfinite(e1));
    REQUIRE(std::isfinite(e2));
    REQUIRE(e1 != Catch::Approx(e2).margin(1e-4));

    // The ratio should be close to 10.0/2.0 = 5 (torsion energy linear in fc)
    // At 90 degrees: E = fc3*(1 + cos(3*pi/2)) = fc3*(1 + 0) = fc3
    // Actually: cos(3 * ~pi/2) depends on exact angle. But ratio should hold.
    if (e1 != 0) {
        REQUIRE(std::abs(e2 / e1 - 5.0) < 0.5);
    }

    INFO("e1(fc3=2.0)=" << e1 << " e2(fc3=10.0)=" << e2);
}


// ========================== SECTION 4: 1-4 pair parameter tests ==========================

TEST_CASE("custom_ff_param: different 1-4 pair aij/bij changes energy", "[custom_ff][cpu]") {
    g_cpu_constants.BOX_X_LO = -50; g_cpu_constants.BOX_X_HI = 50;
    g_cpu_constants.BOX_Y_LO = -50; g_cpu_constants.BOX_Y_HI = 50;
    g_cpu_constants.BOX_Z_LO = -50; g_cpu_constants.BOX_Z_HI = 50;
    g_cpu_constants.PENALTY_SLOPE = 10.0;

    // 2 atoms at distance 4.0, connected as a 1-4 pair
    // Zero sigma/eps on atoms to kill intra-nonbonded; energy comes only from pair14.
    FFMinData ff;
    ff.lig_atoms.natom = 2;
    ff.lig_atoms.sigma   = {0.001, 0.001};
    ff.lig_atoms.epsilon  = {0.0, 0.0};
    ff.lig_atoms.charge   = {0.0, 0.0};
    ff.lig_atoms.anum     = {6, 6};
    ff.lig_atoms.x = {0, 0};
    ff.lig_atoms.y = {0, 0};
    ff.lig_atoms.z = {0, 0};
    ff.rec_atoms.natom = 0;
    ff.lig_exclusions = {};
    ff.build_pair_lookup();
    ff.loaded = true;

    Real r = 4.0;

    // 1-4 pair 1: moderate aij/bij, zero charge product
    FFPair14 p1;
    p1.i = 0; p1.j = 1;
    p1.aij = 1000.0;
    p1.bij = 50.0;
    p1.qij = 0.0;

    FFMinData ff1 = ff;
    ff1.lig_pairs_14 = {p1};
    ff1.build_pair_lookup();

    FFTestSetup s1;
    s1.coords = {0, 0, 0, r, 0, 0};
    s1.build(2);
    Real e1 = s1.score(ff1);

    // 1-4 pair 2: much larger aij (stronger repulsion at short range)
    FFPair14 p2 = p1;
    p2.aij = 100000.0;

    FFMinData ff2 = ff;
    ff2.lig_pairs_14 = {p2};
    ff2.build_pair_lookup();

    FFTestSetup s2;
    s2.coords = {0, 0, 0, r, 0, 0};
    s2.build(2);
    Real e2 = s2.score(ff2);

    REQUIRE(std::isfinite(e1));
    REQUIRE(std::isfinite(e2));
    REQUIRE(e1 != Catch::Approx(e2).margin(1e-4));

    INFO("e1(aij=1e3)=" << e1 << " e2(aij=1e5)=" << e2);
}


// ========================== SECTION 5: Real DMS data — parameter perturbation ==========================

static const std::string DMS_FF_CUSTOM_DIR = std::string(TEST_DATA_DIR) + "ff_minimize/tyk2/";

// Helper: load DMS data and build the scoring setup
struct DMSScoreSetup {
    FFMinData ff;
    FlexTopo topo;
    FlexPose pose;
    FixMol fix_mol;
    std::vector<Real> coords;
    std::vector<Real> rec_coords;
    std::vector<int> vn_types;
    std::vector<Real> aux_f;

    bool load(const std::string& lig_dms, const std::string& rec_dms) {
        read_ligand_dms(lig_dms, ff);
        read_receptor_dms(rec_dms, ff);
        if (!ff.loaded) return false;

        int n = ff.lig_atoms.natom;
        vn_types.assign(n, 0);
        aux_f.assign(n * 3, 0);
        coords.resize(n * 3);

        topo.natom = n;
        topo.ntorsion = 0;
        topo.vn_types = vn_types.data();
        topo.axis_atoms = nullptr;
        topo.range_inds = nullptr;
        topo.range_list = nullptr;
        topo.rotated_inds = nullptr;
        topo.rotated_atoms = nullptr;

        for (int i = 0; i < n; i++) {
            coords[i*3]   = ff.lig_atoms.x[i];
            coords[i*3+1] = ff.lig_atoms.y[i];
            coords[i*3+2] = ff.lig_atoms.z[i];
        }
        pose.coords = coords.data();
        pose.dihedrals = nullptr;
        pose.center[0] = pose.center[1] = pose.center[2] = 0;
        pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;
        for (int i = 0; i < n; i++) {
            pose.center[0] += coords[i*3];
            pose.center[1] += coords[i*3+1];
            pose.center[2] += coords[i*3+2];
        }
        pose.center[0] /= n;
        pose.center[1] /= n;
        pose.center[2] /= n;

        int nr = ff.rec_atoms.natom;
        rec_coords.resize(nr * 3);
        for (int i = 0; i < nr; i++) {
            rec_coords[i*3]   = ff.rec_atoms.x[i];
            rec_coords[i*3+1] = ff.rec_atoms.y[i];
            rec_coords[i*3+2] = ff.rec_atoms.z[i];
        }
        fix_mol.natom = nr;
        fix_mol.coords = rec_coords.data();

        DockParam dp;
        dp.box.x_lo = pose.center[0] - 20;
        dp.box.x_hi = pose.center[0] + 20;
        dp.box.y_lo = pose.center[1] - 20;
        dp.box.y_hi = pose.center[1] + 20;
        dp.box.z_lo = pose.center[2] - 20;
        dp.box.z_hi = pose.center[2] + 20;
        AtomTypeProps atp;
        init_constants_cpu(dp, atp);

        return true;
    }

    Real score() {
        std::fill(aux_f.begin(), aux_f.end(), 0);
        return cal_e_f_ff_cpu(&pose, topo, fix_mol, ff, aux_f.data());
    }
};


TEST_CASE("custom_ff_param: perturbing sigma on real DMS changes total energy", "[custom_ff][cpu][dms]") {
    DMSScoreSetup s;
    REQUIRE(s.load(DMS_FF_CUSTOM_DIR + "ligand_ejm42.dms",
                   DMS_FF_CUSTOM_DIR + "protein.dms"));

    Real e_orig = s.score();
    REQUIRE(std::isfinite(e_orig));

    // Perturb: double the sigma of the first heavy atom
    Real orig_sig = s.ff.lig_atoms.sigma[0];
    s.ff.lig_atoms.sigma[0] = orig_sig * 2.0;
    Real e_perturbed = s.score();

    REQUIRE(std::isfinite(e_perturbed));
    REQUIRE(e_orig != Catch::Approx(e_perturbed).margin(0.01));

    // Restore → should recover original energy
    s.ff.lig_atoms.sigma[0] = orig_sig;
    Real e_restored = s.score();
    REQUIRE(e_restored == Catch::Approx(e_orig).margin(1e-5));

    INFO("e_orig=" << e_orig << " e_perturbed=" << e_perturbed);
}


TEST_CASE("custom_ff_param: perturbing epsilon on real DMS changes total energy", "[custom_ff][cpu][dms]") {
    DMSScoreSetup s;
    REQUIRE(s.load(DMS_FF_CUSTOM_DIR + "ligand_ejm42.dms",
                   DMS_FF_CUSTOM_DIR + "protein.dms"));

    Real e_orig = s.score();

    // Perturb: scale epsilon of all atoms by 2x
    std::vector<Real> orig_eps = s.ff.lig_atoms.epsilon;
    for (auto& e : s.ff.lig_atoms.epsilon) e *= 2.0;
    Real e_perturbed = s.score();

    REQUIRE(std::isfinite(e_perturbed));
    REQUIRE(e_orig != Catch::Approx(e_perturbed).margin(0.01));

    INFO("e_orig=" << e_orig << " e_scaled_eps=" << e_perturbed);
}


TEST_CASE("custom_ff_param: perturbing charge on real DMS changes total energy", "[custom_ff][cpu][dms]") {
    DMSScoreSetup s;
    REQUIRE(s.load(DMS_FF_CUSTOM_DIR + "ligand_ejm42.dms",
                   DMS_FF_CUSTOM_DIR + "protein.dms"));

    Real e_orig = s.score();

    // Perturb: flip all charges (changes sign of all Coulomb terms)
    for (auto& q : s.ff.lig_atoms.charge) q = -q;
    Real e_flipped = s.score();

    REQUIRE(std::isfinite(e_flipped));
    // Flipping all ligand charges changes lig-rec Coulomb energy;
    // intra Coulomb with flipped charges stays the same (qi*qj unchanged).
    // But inter Coulomb changes sign → different total energy.
    REQUIRE(e_orig != Catch::Approx(e_flipped).margin(0.01));

    INFO("e_orig=" << e_orig << " e_flipped_q=" << e_flipped);
}


TEST_CASE("custom_ff_param: zeroing all torsion fc changes total energy", "[custom_ff][cpu][dms]") {
    DMSScoreSetup s;
    REQUIRE(s.load(DMS_FF_CUSTOM_DIR + "ligand_ejm42.dms",
                   DMS_FF_CUSTOM_DIR + "protein.dms"));

    Real e_orig = s.score();

    // Save original torsions, then zero all fc
    auto orig_torsions = s.ff.lig_torsions;
    REQUIRE(orig_torsions.size() > 0);

    for (auto& t : s.ff.lig_torsions) {
        for (int n = 0; n < 7; n++) t.fc[n] = 0;
    }
    Real e_no_torsion = s.score();

    REQUIRE(std::isfinite(e_no_torsion));
    REQUIRE(e_orig != Catch::Approx(e_no_torsion).margin(0.01));

    INFO("e_orig=" << e_orig << " e_no_torsion=" << e_no_torsion);
}


// ========================== SECTION 6: score_decomp_ff with custom params ==========================

TEST_CASE("custom_ff_param: score_decomp_ff reflects custom sigma in inter_lj", "[custom_ff][cpu][decomp]") {
    DMSScoreSetup s;
    REQUIRE(s.load(DMS_FF_CUSTOM_DIR + "ligand_ejm42.dms",
                   DMS_FF_CUSTOM_DIR + "protein.dms"));

    auto terms_orig = score_decomp_ff(&s.pose, s.ff);

    // Sum inter_lj across all atoms
    Real inter_lj_orig = 0;
    for (auto& t : terms_orig) inter_lj_orig += t.inter_lj;

    // Perturb sigma of first heavy atom
    s.ff.lig_atoms.sigma[0] *= 2.0;
    auto terms_perturbed = score_decomp_ff(&s.pose, s.ff);

    Real inter_lj_perturbed = 0;
    for (auto& t : terms_perturbed) inter_lj_perturbed += t.inter_lj;

    REQUIRE(inter_lj_orig != Catch::Approx(inter_lj_perturbed).margin(0.01));

    INFO("inter_lj_orig=" << inter_lj_orig << " inter_lj_perturbed=" << inter_lj_perturbed);
}
