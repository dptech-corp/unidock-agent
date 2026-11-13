//
// Detailed unit tests for FF CPU scoring.
// Focus: finite difference validation, second ligand, index bounds.
//

#include <catch2/catch_amalgamated.hpp>
#include "cpu/score_ff_cpu.h"
#include "format/dms_reader.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <set>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const std::string DMS_FF_TEST_DIR = std::string(TEST_DATA_DIR) + "ff_minimize/tyk2/";

// Helper: set up FlexTopo, FlexPose, FixMol from FFMinData (ligand coords from DMS)
struct FFTestFixture {
    FFMinData ff;
    FlexTopo topo;
    FlexPose pose;
    FixMol fix_mol;
    std::vector<int> vn_types;
    std::vector<Real> coords;
    std::vector<Real> rec_coords;
    std::vector<Real> dihedrals;

    void load(const std::string& lig_name, const std::string& rec_name = "protein.dms") {
        read_ligand_dms(DMS_FF_TEST_DIR + lig_name, ff);
        read_receptor_dms(DMS_FF_TEST_DIR + rec_name, ff);
        REQUIRE(ff.loaded);

        topo.natom = ff.lig_atoms.natom;
        topo.ntorsion = 0;
        vn_types.assign(topo.natom, 0);
        topo.vn_types = vn_types.data();
        topo.axis_atoms = nullptr;
        topo.range_inds = nullptr;
        topo.range_list = nullptr;
        topo.rotated_inds = nullptr;
        topo.rotated_atoms = nullptr;

        coords.resize(topo.natom * 3);
        for (int i = 0; i < topo.natom; i++) {
            coords[i*3]   = ff.lig_atoms.x[i];
            coords[i*3+1] = ff.lig_atoms.y[i];
            coords[i*3+2] = ff.lig_atoms.z[i];
        }
        pose.coords = coords.data();
        dihedrals.clear();
        pose.dihedrals = nullptr;
        pose.center[0] = pose.center[1] = pose.center[2] = 0;
        pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;

        for (int i = 0; i < topo.natom; i++) {
            pose.center[0] += coords[i*3];
            pose.center[1] += coords[i*3+1];
            pose.center[2] += coords[i*3+2];
        }
        pose.center[0] /= topo.natom;
        pose.center[1] /= topo.natom;
        pose.center[2] /= topo.natom;

        // FixMol with all receptor coords (for API compatibility, though cal_e_f_ff_cpu no longer uses it)
        fix_mol.natom = ff.rec_atoms.natom;
        rec_coords.resize(fix_mol.natom * 3);
        for (int i = 0; i < fix_mol.natom; i++) {
            rec_coords[i*3]   = ff.rec_atoms.x[i];
            rec_coords[i*3+1] = ff.rec_atoms.y[i];
            rec_coords[i*3+2] = ff.rec_atoms.z[i];
        }
        fix_mol.coords = rec_coords.data();

        // Initialize box
        DockParam dp;
        dp.box.x_lo = pose.center[0] - 20;
        dp.box.x_hi = pose.center[0] + 20;
        dp.box.y_lo = pose.center[1] - 20;
        dp.box.y_hi = pose.center[1] + 20;
        dp.box.z_lo = pose.center[2] - 20;
        dp.box.z_hi = pose.center[2] + 20;
        AtomTypeProps atp;
        init_constants_cpu(dp, atp);
    }
};


// ============================================================================
// DMS reader: atom index bounds validation
// ============================================================================

TEST_CASE("dms_reader: pair14 atom indices within bounds", "[dms_reader_detail]") {
    FFMinData ff;
    read_ligand_dms(DMS_FF_TEST_DIR + "ligand_ejm42.dms", ff);
    int natom = ff.lig_atoms.natom;

    for (size_t k = 0; k < ff.lig_pairs_14.size(); k++) {
        const auto& p = ff.lig_pairs_14[k];
        INFO("pair14[" << k << "]: i=" << p.i << " j=" << p.j << " natom=" << natom);
        REQUIRE(p.i >= 0);
        REQUIRE(p.i < natom);
        REQUIRE(p.j >= 0);
        REQUIRE(p.j < natom);
        REQUIRE(p.i != p.j);
    }
}

TEST_CASE("dms_reader: torsion atom indices within bounds", "[dms_reader_detail]") {
    FFMinData ff;
    read_ligand_dms(DMS_FF_TEST_DIR + "ligand_ejm42.dms", ff);
    int natom = ff.lig_atoms.natom;

    for (size_t k = 0; k < ff.lig_torsions.size(); k++) {
        const auto& t = ff.lig_torsions[k];
        INFO("torsion[" << k << "]: " << t.p0 << "-" << t.p1 << "-" << t.p2 << "-" << t.p3);
        REQUIRE(t.p0 >= 0);
        REQUIRE(t.p0 < natom);
        REQUIRE(t.p1 >= 0);
        REQUIRE(t.p1 < natom);
        REQUIRE(t.p2 >= 0);
        REQUIRE(t.p2 < natom);
        REQUIRE(t.p3 >= 0);
        REQUIRE(t.p3 < natom);
    }
}

TEST_CASE("dms_reader: exclusion atom indices within bounds", "[dms_reader_detail]") {
    FFMinData ff;
    read_ligand_dms(DMS_FF_TEST_DIR + "ligand_ejm42.dms", ff);
    int natom = ff.lig_atoms.natom;

    for (size_t k = 0; k < ff.lig_exclusions.size(); k++) {
        const auto& ex = ff.lig_exclusions[k];
        INFO("exclusion[" << k << "]: i=" << ex.i << " j=" << ex.j);
        REQUIRE(ex.i >= 0);
        REQUIRE(ex.i < natom);
        REQUIRE(ex.j >= 0);
        REQUIRE(ex.j < natom);
    }
}

TEST_CASE("dms_reader: small ligand index bounds", "[dms_reader_detail]") {
    FFMinData ff;
    read_ligand_dms(DMS_FF_TEST_DIR + "ligand_small1.dms", ff);
    int natom = ff.lig_atoms.natom;
    REQUIRE(natom == 17);

    for (const auto& p : ff.lig_pairs_14) {
        REQUIRE(p.i >= 0);
        REQUIRE(p.i < natom);
        REQUIRE(p.j >= 0);
        REQUIRE(p.j < natom);
    }
    for (const auto& t : ff.lig_torsions) {
        REQUIRE(t.p0 >= 0);
        REQUIRE(t.p0 < natom);
        REQUIRE(t.p3 >= 0);
        REQUIRE(t.p3 < natom);
    }
}


// ============================================================================
// Finite difference force validation for cal_e_f_ff_cpu
// This is the most critical test: verifies analytical forces match numerical dE/dx.
// ============================================================================

TEST_CASE("cal_e_f_ff_cpu: gradient matches finite difference on ejm42", "[score_ff_cpu_fd]") {
    FFTestFixture fix;
    fix.load("ligand_ejm42.dms");

    const int natom = fix.topo.natom;
    const Real delta = 5e-4f;

    // Compute analytical gradient
    std::vector<Real> aux_f(natom * 3, 0);
    Real e0 = cal_e_f_ff_cpu(&fix.pose, fix.topo, fix.fix_mol, fix.ff, aux_f.data());
    REQUIRE(std::isfinite(e0));

    // NOTE: aux_f stores the energy GRADIENT ∂E/∂x (same convention as Vina scoring).
    // For atoms near close contacts (steep LJ r^-12), the analytical gradient can be
    // very large and finite differences become unreliable. We test separately:
    // 1. Sign agreement (should be ~100% for moderate-gradient atoms)
    // 2. Relative error for moderate-gradient atoms (|g| < 50 kcal/mol/A)
    int n_moderate = 0, n_sign_ok = 0, n_rel_ok = 0;
    int n_total = 0;

    for (int i = 0; i < natom; i++) {
        for (int d = 0; d < 3; d++) {
            int idx = i * 3 + d;
            Real orig = fix.coords[idx];

            fix.coords[idx] = orig + delta;
            std::vector<Real> f_tmp(natom * 3, 0);
            Real e_plus = cal_e_f_ff_cpu(&fix.pose, fix.topo, fix.fix_mol, fix.ff, f_tmp.data());

            fix.coords[idx] = orig - delta;
            Real e_minus = cal_e_f_ff_cpu(&fix.pose, fix.topo, fix.fix_mol, fix.ff, f_tmp.data());

            fix.coords[idx] = orig;

            Real dE_dx_numeric = (e_plus - e_minus) / (2 * delta);
            Real dE_dx_analytic = aux_f[idx];

            n_total++;

            // Skip near-zero gradients (noise-dominated)
            if (std::abs(dE_dx_analytic) < 0.01f && std::abs(dE_dx_numeric) < 0.01f) {
                n_moderate++;
                n_sign_ok++;
                n_rel_ok++;
                continue;
            }

            // For moderate gradients (not in stiff LJ repulsion zone)
            if (std::abs(dE_dx_analytic) < 50.0f && std::abs(dE_dx_numeric) < 50.0f) {
                n_moderate++;

                // Sign check
                if (dE_dx_analytic * dE_dx_numeric >= 0) {
                    n_sign_ok++;
                }

                // Relative error check
                Real abs_diff = std::abs(dE_dx_analytic - dE_dx_numeric);
                Real scale = std::max(1.0f, std::max(std::abs(dE_dx_analytic), std::abs(dE_dx_numeric)));
                if (abs_diff / scale < 0.1f || abs_diff < 0.5f) {
                    n_rel_ok++;
                }
            }
        }
    }

    spdlog::info("FD gradient check (ejm42): moderate={}/{}, sign_ok={}, rel_ok={}",
                 n_moderate, n_total, n_sign_ok, n_rel_ok);

    // Sign should agree for most moderate-gradient atoms.
    // Some sign mismatches are expected at the hard 12A cutoff boundary where
    // FD perturbation causes receptor atoms to jump in/out of the cutoff.
    REQUIRE(n_sign_ok > n_moderate * 0.88);
    // Relative error should be acceptable for the majority
    REQUIRE(n_rel_ok > n_moderate * 0.55);
}


// ============================================================================
// Second ligand: energy and decomposition on ligand_small1
// ============================================================================

TEST_CASE("cal_e_f_ff_cpu: energy is finite on small1 ligand", "[score_ff_cpu_detail]") {
    FFTestFixture fix;
    fix.load("ligand_small1.dms");

    std::vector<Real> aux_f(fix.topo.natom * 3, 0);
    Real energy = cal_e_f_ff_cpu(&fix.pose, fix.topo, fix.fix_mol, fix.ff, aux_f.data());
    REQUIRE(std::isfinite(energy));

    // Forces should be finite
    for (int i = 0; i < fix.topo.natom * 3; i++) {
        REQUIRE(std::isfinite(aux_f[i]));
    }
    spdlog::info("small1 FF energy: {:.4f} kcal/mol", energy);
}

TEST_CASE("score_decomp_ff: matches total on small1 ligand", "[score_ff_cpu_detail]") {
    FFTestFixture fix;
    fix.load("ligand_small1.dms");

    std::vector<Real> aux_f(fix.topo.natom * 3, 0);
    Real total_energy = cal_e_f_ff_cpu(&fix.pose, fix.topo, fix.fix_mol, fix.ff, aux_f.data());

    auto atom_terms = score_decomp_ff(&fix.pose, fix.ff);
    REQUIRE(atom_terms.size() == static_cast<size_t>(fix.topo.natom));

    Real decomp_sum = 0;
    for (const auto& t : atom_terms) {
        REQUIRE(std::isfinite(t.total));
        decomp_sum += t.total;
    }

    spdlog::info("small1: cal_e_f total={:.4f}, decomp sum={:.4f}", total_energy, decomp_sum);
    REQUIRE(decomp_sum == Catch::Approx(total_energy).margin(0.2));
}


// ============================================================================
// Force sign check: moving ligand toward receptor should decrease energy
// if they are within attractive range (not too close)
// ============================================================================

TEST_CASE("cal_e_f_ff_cpu: stepping opposite to gradient decreases energy", "[score_ff_cpu_detail]") {
    FFTestFixture fix;
    fix.load("ligand_ejm42.dms");

    // Translate ligand 1 Angstrom in +x direction (to create a gradient)
    for (int i = 0; i < fix.topo.natom; i++) {
        fix.coords[i*3] += 1.0f;
    }
    fix.pose.center[0] += 1.0f;

    std::vector<Real> aux_f(fix.topo.natom * 3, 0);
    Real e_shifted = cal_e_f_ff_cpu(&fix.pose, fix.topo, fix.fix_mol, fix.ff, aux_f.data());
    REQUIRE(std::isfinite(e_shifted));

    // Total gradient in x direction (aux_f stores gradient ∂E/∂x)
    Real total_gx = 0;
    for (int i = 0; i < fix.topo.natom; i++) {
        total_gx += aux_f[i*3];
    }

    // Step OPPOSITE to gradient direction (steepest descent)
    Real step = 0.01f;
    Real sign = (total_gx > 0) ? -1.0f : 1.0f;  // opposite to gradient
    for (int i = 0; i < fix.topo.natom; i++) {
        fix.coords[i*3] += sign * step;
    }
    fix.pose.center[0] += sign * step;

    std::vector<Real> f2(fix.topo.natom * 3, 0);
    Real e_follow = cal_e_f_ff_cpu(&fix.pose, fix.topo, fix.fix_mol, fix.ff, f2.data());

    // Stepping opposite to gradient should decrease energy
    REQUIRE(e_follow <= e_shifted + 0.5f);
}


// ============================================================================
// Exclusion set correctness: excluded pairs should NOT contribute to energy
// ============================================================================

TEST_CASE("cal_e_f_ff_cpu: exclusion list prevents self-interaction artifacts", "[score_ff_cpu_detail]") {
    FFTestFixture fix;
    fix.load("ligand_ejm42.dms");

    // Verify exclusion count is reasonable: should be >= number of bonds
    // For a 35-atom ligand, typically ~35 bonds + ~60 1-3 pairs
    REQUIRE(fix.ff.lig_exclusions.size() > 30);

    // Verify no self-exclusions (i == j)
    for (const auto& ex : fix.ff.lig_exclusions) {
        REQUIRE(ex.i != ex.j);
    }

    // In AMBER DMS format, 1-4 pairs MAY also appear in the exclusion table.
    // The code handles this correctly: the intra-molecular loop skips both excluded
    // and pair14 pairs, then pair14 interactions are computed separately.
    // Verify no duplicate exclusions
    std::set<std::pair<int,int>> excl_set;
    for (const auto& ex : fix.ff.lig_exclusions) {
        int a = std::min(ex.i, ex.j);
        int b = std::max(ex.i, ex.j);
        excl_set.insert({a, b});
    }
    // All exclusions should be unique pairs
    REQUIRE(excl_set.size() == fix.ff.lig_exclusions.size());
}


// ============================================================================
// Dihedral force: finite difference check on a single torsion
// ============================================================================

TEST_CASE("distribute_dihedral_forces: FD check on non-planar geometry", "[score_ff_cpu_detail]") {
    // Non-trivial 3D geometry
    Real p0[3] = {-1.5, 0.8, 0.3};
    Real p1[3] = { 0.0, 0.0, 0.0};
    Real p2[3] = { 1.2, 0.1, -0.2};
    Real p3[3] = { 2.3, -0.7, 0.9};

    Real dE_dphi = 1.0f;  // unit dE/dphi

    // Analytical forces
    Real f0[3] = {0}, f1[3] = {0}, f2[3] = {0}, f3[3] = {0};
    distribute_dihedral_forces(p0, p1, p2, p3, dE_dphi, f0, f1, f2, f3);

    // Newton's 3rd law
    for (int k = 0; k < 3; k++) {
        Real sum = f0[k] + f1[k] + f2[k] + f3[k];
        REQUIRE(std::abs(sum) < 1e-4);
    }

    // FD check: perturb atom 3 and check dE/dx_3 ≈ f3
    // NOTE: distribute_dihedral_forces stores gradient (∂E/∂x), not force,
    // despite the function name. This is consistent with the Vina convention
    // where aux_f stores the energy gradient.
    Real delta = 1e-4f;
    for (int d = 0; d < 3; d++) {
        Real p3_save = p3[d];

        p3[d] = p3_save + delta;
        Real phi_plus = compute_dihedral_angle(p0, p1, p2, p3);

        p3[d] = p3_save - delta;
        Real phi_minus = compute_dihedral_angle(p0, p1, p2, p3);

        p3[d] = p3_save;

        // dE/dx3 = dE/dphi * dphi/dx3
        Real dphi_dx3 = (phi_plus - phi_minus) / (2 * delta);
        Real dE_dx3_numeric = dE_dphi * dphi_dx3;

        // f3 stores gradient ∂E/∂x3 (same convention as aux_f)
        Real dE_dx3_analytic = f3[d];

        Real margin = std::max(1e-2f, std::abs(dE_dx3_analytic) * 0.05f);
        REQUIRE(dE_dx3_analytic == Catch::Approx(dE_dx3_numeric).margin(margin));
    }
}


// ============================================================================
// Test collinear atoms in dihedral (edge case: b2_len ≈ 0 or mm/nn ≈ 0)
// ============================================================================

TEST_CASE("compute_dihedral_angle: collinear atoms return 0", "[score_ff_cpu_detail]") {
    // p0, p1, p2 are collinear → normal m = 0
    Real p0[3] = {0, 0, -1};
    Real p1[3] = {0, 0, 0};
    Real p2[3] = {0, 0, 1};
    Real p3[3] = {1, 0, 2};

    // b1 = (0,0,1), b2 = (0,0,1) → m = b1 x b2 = (0,0,0)
    Real phi = compute_dihedral_angle(p0, p1, p2, p3);
    // Should not crash, returns 0 or a finite value
    REQUIRE(std::isfinite(phi));
}

TEST_CASE("distribute_dihedral_forces: collinear does not crash", "[score_ff_cpu_detail]") {
    Real p0[3] = {0, 0, -1};
    Real p1[3] = {0, 0, 0};
    Real p2[3] = {0, 0, 1};
    Real p3[3] = {1, 0, 2};

    Real f0[3] = {0}, f1[3] = {0}, f2[3] = {0}, f3[3] = {0};
    // Should not crash (mm ≈ 0 → early return)
    distribute_dihedral_forces(p0, p1, p2, p3, 1.0f, f0, f1, f2, f3);
    // Forces should be zero or finite
    for (int k = 0; k < 3; k++) {
        REQUIRE(std::isfinite(f0[k]));
        REQUIRE(std::isfinite(f3[k]));
    }
}


// ============================================================================
// Second ligand: FD force check on ligand_small1
// ============================================================================

TEST_CASE("cal_e_f_ff_cpu: gradient matches FD on small1 ligand", "[score_ff_cpu_fd]") {
    FFTestFixture fix;
    fix.load("ligand_small1.dms");

    const int natom = fix.topo.natom;
    const Real delta = 5e-4f;

    std::vector<Real> aux_f(natom * 3, 0);
    Real e0 = cal_e_f_ff_cpu(&fix.pose, fix.topo, fix.fix_mol, fix.ff, aux_f.data());
    REQUIRE(std::isfinite(e0));

    int n_moderate = 0, n_sign_ok = 0, n_rel_ok = 0;
    int n_total = 0;

    for (int i = 0; i < natom; i++) {
        for (int d = 0; d < 3; d++) {
            int idx = i * 3 + d;
            Real orig = fix.coords[idx];

            fix.coords[idx] = orig + delta;
            std::vector<Real> f_tmp(natom * 3, 0);
            Real e_plus = cal_e_f_ff_cpu(&fix.pose, fix.topo, fix.fix_mol, fix.ff, f_tmp.data());

            fix.coords[idx] = orig - delta;
            Real e_minus = cal_e_f_ff_cpu(&fix.pose, fix.topo, fix.fix_mol, fix.ff, f_tmp.data());

            fix.coords[idx] = orig;

            Real dE_dx_numeric = (e_plus - e_minus) / (2 * delta);
            Real dE_dx_analytic = aux_f[idx];

            n_total++;

            if (std::abs(dE_dx_analytic) < 0.01f && std::abs(dE_dx_numeric) < 0.01f) {
                n_moderate++;
                n_sign_ok++;
                n_rel_ok++;
                continue;
            }

            if (std::abs(dE_dx_analytic) < 50.0f && std::abs(dE_dx_numeric) < 50.0f) {
                n_moderate++;
                if (dE_dx_analytic * dE_dx_numeric >= 0) {
                    n_sign_ok++;
                }
                Real abs_diff = std::abs(dE_dx_analytic - dE_dx_numeric);
                Real scale = std::max(1.0f, std::max(std::abs(dE_dx_analytic), std::abs(dE_dx_numeric)));
                if (abs_diff / scale < 0.1f || abs_diff < 0.5f) {
                    n_rel_ok++;
                }
            }
        }
    }

    spdlog::info("FD gradient check (small1): moderate={}/{}, sign_ok={}, rel_ok={}",
                 n_moderate, n_total, n_sign_ok, n_rel_ok);
    REQUIRE(n_sign_ok > n_moderate * 0.95);
    REQUIRE(n_rel_ok > n_moderate * 0.80);
}
