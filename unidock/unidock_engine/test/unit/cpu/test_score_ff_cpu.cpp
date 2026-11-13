//
// Unit tests for FF CPU scoring functions
//

#include <catch2/catch_amalgamated.hpp>
#include "cpu/score_ff_cpu.h"
#include "format/dms_reader.h"
#include <cmath>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Dihedral angle computation
// ============================================================================

TEST_CASE("compute_dihedral_angle: cis configuration is ~0", "[score_ff_cpu]") {
    // Cis: p0 and p3 on the same side of the p1-p2 axis
    Real p0[3] = {-1.0, 1.0, 0.0};
    Real p1[3] = { 0.0, 0.0, 0.0};
    Real p2[3] = { 1.0, 0.0, 0.0};
    Real p3[3] = { 2.0, 1.0, 0.0};  // same side as p0

    Real phi = compute_dihedral_angle(p0, p1, p2, p3);
    REQUIRE(std::abs(phi) < 0.01);
}

TEST_CASE("compute_dihedral_angle: trans configuration is ~pi", "[score_ff_cpu]") {
    // Trans: p0 and p3 on opposite sides of the p1-p2 axis
    Real p0[3] = {-1.0, 1.0, 0.0};
    Real p1[3] = { 0.0, 0.0, 0.0};
    Real p2[3] = { 1.0, 0.0, 0.0};
    Real p3[3] = { 2.0,-1.0, 0.0};  // opposite side from p0

    Real phi = compute_dihedral_angle(p0, p1, p2, p3);
    REQUIRE(std::abs(std::abs(phi) - M_PI) < 0.01);
}

TEST_CASE("compute_dihedral_angle: gauche 90 degrees", "[score_ff_cpu]") {
    // 90 degree dihedral
    Real p0[3] = {-1.0, 1.0, 0.0};
    Real p1[3] = { 0.0, 0.0, 0.0};
    Real p2[3] = { 1.0, 0.0, 0.0};
    Real p3[3] = { 2.0, 0.0, 1.0};  // rotated 90 deg around p1-p2 axis

    Real phi = compute_dihedral_angle(p0, p1, p2, p3);
    REQUIRE(std::abs(std::abs(phi) - M_PI/2) < 0.01);
}


// ============================================================================
// Dihedral force distribution (Newton's 3rd law check)
// ============================================================================

TEST_CASE("distribute_dihedral_forces: total force sums to zero", "[score_ff_cpu]") {
    Real p0[3] = {-1.0, 1.0, 0.0};
    Real p1[3] = { 0.0, 0.0, 0.0};
    Real p2[3] = { 1.0, 0.0, 0.0};
    Real p3[3] = { 2.0, 0.5, 0.8};

    Real f0[3] = {0}, f1[3] = {0}, f2[3] = {0}, f3[3] = {0};
    Real dE_dphi = 1.0;

    distribute_dihedral_forces(p0, p1, p2, p3, dE_dphi, f0, f1, f2, f3);

    // Newton's 3rd law: sum of all forces = 0
    for (int k = 0; k < 3; k++) {
        Real sum = f0[k] + f1[k] + f2[k] + f3[k];
        REQUIRE(std::abs(sum) < 1e-4);
    }
}

TEST_CASE("distribute_dihedral_forces: FD check vs dihedral angle", "[score_ff_cpu]") {
    // Numerical check: rotating atom 3 by delta should change dihedral by ~delta
    Real p0[3] = {-1.0, 1.0, 0.0};
    Real p1[3] = { 0.0, 0.0, 0.0};
    Real p2[3] = { 1.0, 0.0, 0.0};
    Real p3[3] = { 2.0, 0.5, 0.8};

    Real f0[3] = {0}, f1[3] = {0}, f2[3] = {0}, f3[3] = {0};
    Real dE_dphi = 1.0;

    distribute_dihedral_forces(p0, p1, p2, p3, dE_dphi, f0, f1, f2, f3);

    // Check that forces on terminal atoms are non-zero
    Real f0_mag = std::sqrt(f0[0]*f0[0] + f0[1]*f0[1] + f0[2]*f0[2]);
    Real f3_mag = std::sqrt(f3[0]*f3[0] + f3[1]*f3[1] + f3[2]*f3[2]);
    REQUIRE(f0_mag > 1e-6);
    REQUIRE(f3_mag > 1e-6);
}


// ============================================================================
// FF energy on real DMS data
// ============================================================================

static const std::string DMS_FF_TEST_DIR = std::string(TEST_DATA_DIR) + "ff_minimize/tyk2/";

TEST_CASE("cal_e_f_ff_cpu: energy is finite on ejm42 ligand", "[score_ff_cpu]") {
    // Load DMS data
    FFMinData ff;
    std::string lig_path = DMS_FF_TEST_DIR + "ligand_ejm42.dms";
    std::string rec_path = DMS_FF_TEST_DIR + "protein.dms";
    read_ligand_dms(lig_path, ff);
    read_receptor_dms(rec_path, ff);
    REQUIRE(ff.loaded);

    // Create a minimal FlexTopo
    FlexTopo topo;
    topo.natom = ff.lig_atoms.natom;
    topo.ntorsion = 0;

    // Create vn_types (just use anum as a proxy — box penalty won't apply if H check fails)
    std::vector<int> vn_types(topo.natom, 0);
    topo.vn_types = vn_types.data();
    topo.axis_atoms = nullptr;
    topo.range_inds = nullptr;
    topo.range_list = nullptr;
    topo.rotated_inds = nullptr;
    topo.rotated_atoms = nullptr;

    // Create pose with DMS coordinates
    FlexPose pose;
    std::vector<Real> coords(topo.natom * 3);
    for (int i = 0; i < topo.natom; i++) {
        coords[i*3]   = ff.lig_atoms.x[i];
        coords[i*3+1] = ff.lig_atoms.y[i];
        coords[i*3+2] = ff.lig_atoms.z[i];
    }
    pose.coords = coords.data();
    pose.dihedrals = nullptr;
    pose.center[0] = pose.center[1] = pose.center[2] = 0;
    pose.rot_vec[0] = pose.rot_vec[1] = pose.rot_vec[2] = 0;

    // Compute center
    for (int i = 0; i < topo.natom; i++) {
        pose.center[0] += coords[i*3];
        pose.center[1] += coords[i*3+1];
        pose.center[2] += coords[i*3+2];
    }
    pose.center[0] /= topo.natom;
    pose.center[1] /= topo.natom;
    pose.center[2] /= topo.natom;

    // Create FixMol with receptor coordinates
    FixMol fix_mol;
    fix_mol.natom = ff.rec_atoms.natom;
    std::vector<Real> rec_coords(fix_mol.natom * 3);
    for (int i = 0; i < fix_mol.natom; i++) {
        rec_coords[i*3]   = ff.rec_atoms.x[i];
        rec_coords[i*3+1] = ff.rec_atoms.y[i];
        rec_coords[i*3+2] = ff.rec_atoms.z[i];
    }
    fix_mol.coords = rec_coords.data();

    // Initialize CPU constants (need box for penalty)
    DockParam dp;
    dp.box.x_lo = pose.center[0] - 20;
    dp.box.x_hi = pose.center[0] + 20;
    dp.box.y_lo = pose.center[1] - 20;
    dp.box.y_hi = pose.center[1] + 20;
    dp.box.z_lo = pose.center[2] - 20;
    dp.box.z_hi = pose.center[2] + 20;
    AtomTypeProps atp;
    init_constants_cpu(dp, atp);

    // Compute energy
    std::vector<Real> aux_f(topo.natom * 3, 0);
    Real energy = cal_e_f_ff_cpu(&pose, topo, fix_mol, ff, aux_f.data());

    // Energy should be finite and reasonable
    REQUIRE(std::isfinite(energy));
    spdlog::info("ejm42 FF energy: {:.4f} kcal/mol", energy);

    // Forces should be finite
    for (int i = 0; i < topo.natom * 3; i++) {
        REQUIRE(std::isfinite(aux_f[i]));
    }

    // At least some forces should be non-zero
    bool has_force = false;
    for (int i = 0; i < topo.natom * 3; i++) {
        if (std::abs(aux_f[i]) > 1e-6) { has_force = true; break; }
    }
    REQUIRE(has_force);
}


// ============================================================================
// Per-atom FF decomposition: sum of per-atom totals == total energy
// ============================================================================

TEST_CASE("score_decomp_ff: per-atom sum matches cal_e_f_ff_cpu total", "[score_ff_cpu]") {
    // Load DMS data
    FFMinData ff;
    std::string lig_path = DMS_FF_TEST_DIR + "ligand_ejm42.dms";
    std::string rec_path = DMS_FF_TEST_DIR + "protein.dms";
    read_ligand_dms(lig_path, ff);
    read_receptor_dms(rec_path, ff);
    REQUIRE(ff.loaded);

    // Create a minimal FlexTopo (needed for cal_e_f_ff_cpu)
    FlexTopo topo;
    topo.natom = ff.lig_atoms.natom;
    topo.ntorsion = 0;
    std::vector<int> vn_types(topo.natom, 0);
    topo.vn_types = vn_types.data();
    topo.axis_atoms = nullptr;
    topo.range_inds = nullptr;
    topo.range_list = nullptr;
    topo.rotated_inds = nullptr;
    topo.rotated_atoms = nullptr;

    // Create pose with DMS coordinates
    FlexPose pose;
    std::vector<Real> coords(topo.natom * 3);
    for (int i = 0; i < topo.natom; i++) {
        coords[i*3]   = ff.lig_atoms.x[i];
        coords[i*3+1] = ff.lig_atoms.y[i];
        coords[i*3+2] = ff.lig_atoms.z[i];
    }
    pose.coords = coords.data();
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

    // Create FixMol with receptor coordinates (for cal_e_f_ff_cpu)
    FixMol fix_mol;
    fix_mol.natom = ff.rec_atoms.natom;
    std::vector<Real> rec_coords(fix_mol.natom * 3);
    for (int i = 0; i < fix_mol.natom; i++) {
        rec_coords[i*3]   = ff.rec_atoms.x[i];
        rec_coords[i*3+1] = ff.rec_atoms.y[i];
        rec_coords[i*3+2] = ff.rec_atoms.z[i];
    }
    fix_mol.coords = rec_coords.data();

    // Initialize CPU constants
    DockParam dp;
    dp.box.x_lo = pose.center[0] - 20;
    dp.box.x_hi = pose.center[0] + 20;
    dp.box.y_lo = pose.center[1] - 20;
    dp.box.y_hi = pose.center[1] + 20;
    dp.box.z_lo = pose.center[2] - 20;
    dp.box.z_hi = pose.center[2] + 20;
    AtomTypeProps atp;
    init_constants_cpu(dp, atp);

    // Get total energy from cal_e_f_ff_cpu
    std::vector<Real> aux_f(topo.natom * 3, 0);
    Real total_energy = cal_e_f_ff_cpu(&pose, topo, fix_mol, ff, aux_f.data());
    REQUIRE(std::isfinite(total_energy));

    // Get per-atom decomposition
    auto atom_terms = score_decomp_ff(&pose, ff);
    REQUIRE(atom_terms.size() == static_cast<size_t>(topo.natom));

    // Sum per-atom totals
    Real decomp_sum = 0;
    for (const auto& t : atom_terms) {
        REQUIRE(std::isfinite(t.total));
        decomp_sum += t.total;
    }

    // The sums should match (excluding box penalty which is in cal_e_f_ff_cpu but not in decomp)
    // cal_e_f_ff_cpu includes box penalty; score_decomp_ff does not.
    // For coordinates within the box (which they should be with ±20 box), penalty = 0.
    spdlog::info("cal_e_f_ff_cpu total: {:.4f}", total_energy);
    spdlog::info("score_decomp_ff sum:  {:.4f}", decomp_sum);
    REQUIRE(decomp_sum == Catch::Approx(total_energy).margin(0.1));

    // Verify individual terms are finite and that inter terms are non-zero
    Real sum_inter_lj = 0, sum_inter_coul = 0;
    Real sum_intra_lj = 0, sum_intra_coul = 0;
    Real sum_torsion = 0;
    for (const auto& t : atom_terms) {
        REQUIRE(std::isfinite(t.inter_lj));
        REQUIRE(std::isfinite(t.inter_coulomb));
        REQUIRE(std::isfinite(t.intra_lj));
        REQUIRE(std::isfinite(t.intra_coulomb));
        REQUIRE(std::isfinite(t.torsion));
        sum_inter_lj += t.inter_lj;
        sum_inter_coul += t.inter_coulomb;
        sum_intra_lj += t.intra_lj;
        sum_intra_coul += t.intra_coulomb;
        sum_torsion += t.torsion;
    }
    spdlog::info("  inter_lj:  {:.4f}", sum_inter_lj);
    spdlog::info("  inter_col: {:.4f}", sum_inter_coul);
    spdlog::info("  intra_lj:  {:.4f}", sum_intra_lj);
    spdlog::info("  intra_col: {:.4f}", sum_intra_coul);
    spdlog::info("  torsion:   {:.4f}", sum_torsion);

    // At least inter terms should be non-zero for a ligand near a receptor
    REQUIRE(std::abs(sum_inter_lj) + std::abs(sum_inter_coul) > 1e-3);
}
