//
// Direct unit test for bfgs_ff_cartesian_cpu and bfgs_ff_cpu.
//
// Strategy:
//   1. Load ejm42 ligand + tyk2 receptor DMS.
//   2. Strain the initial pose: push atom b of the first bond along (b-a) so
//      the bond length becomes r0 + 0.1 A.
//   3. Run bfgs_ff_cartesian_cpu (200 steps) — Cartesian mode has bond DOF so
//      it must relax the stretched bond.
//   4. Run bfgs_ff_cpu (200 steps) — internal mode has no bond DOF so the bond
//      remains stretched.
//   5. Assert:
//       A. E_final_cartesian <= E_initial   (energy descended)
//       B. E_final_internal  <= E_initial   (energy descended)
//       C. stretched_bond_final_cartesian < stretched_bond_initial - 0.05  (bond relaxed >= 0.05 A)
//       D. stretched_bond_final_internal  > stretched_bond_initial - 0.01  (bond frozen in int mode)
//       E. rmsd_cartesian_vs_internal > 0.02  (poses are genuinely different)
//       F. max_bond_drift_cartesian < 0.10  (no bond blow-up)
//       G. all final coords are finite (no NaN / inf)
//

#include <catch2/catch_amalgamated.hpp>
#include "cpu/bfgs_ff_cpu.h"
#include "cpu/score_ff_cpu.h"
#include "cpu/common_cpu.h"
#include "cpu/bfgs_cpu.h"
#include "format/dms_reader.h"
#include "model/model.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <limits>

static const std::string DMS_BFGS_REAL_DIR = std::string(TEST_DATA_DIR) + "ff_minimize/tyk2/";

// ============================================================================
// Helper: compute bond length between atoms i and j from a flat coord array.
// ============================================================================
static Real bond_length(const Real* coords, int i, int j) {
    Real dx = coords[j*3]   - coords[i*3];
    Real dy = coords[j*3+1] - coords[i*3+1];
    Real dz = coords[j*3+2] - coords[i*3+2];
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// ============================================================================
// Main test
// ============================================================================
TEST_CASE("bfgs_ff_cartesian_cpu and bfgs_ff_cpu: strained bond relaxation", "[bfgs_ff_real]") {

    // --- 1. Load DMS data ---
    FFMinData ff;
    read_ligand_dms(DMS_BFGS_REAL_DIR + "ligand_ejm42.dms", ff);
    read_receptor_dms(DMS_BFGS_REAL_DIR + "protein.dms", ff);
    REQUIRE(ff.loaded);
    REQUIRE(!ff.lig_bonds.empty());

    int natom = ff.lig_atoms.natom;
    REQUIRE(natom > 0);

    // --- 2. Build FlexTopo / FixMol ---
    FlexTopo topo;
    topo.natom    = natom;
    topo.ntorsion = 0;
    std::vector<int> vn_types(natom, 0);
    topo.vn_types      = vn_types.data();
    topo.axis_atoms    = nullptr;
    topo.range_inds    = nullptr;
    topo.range_list    = nullptr;
    topo.rotated_inds  = nullptr;
    topo.rotated_atoms = nullptr;

    FixMol fix_mol;
    fix_mol.natom = ff.rec_atoms.natom;
    std::vector<Real> rec_coords(fix_mol.natom * 3);
    for (int i = 0; i < fix_mol.natom; i++) {
        rec_coords[i*3]   = ff.rec_atoms.x[i];
        rec_coords[i*3+1] = ff.rec_atoms.y[i];
        rec_coords[i*3+2] = ff.rec_atoms.z[i];
    }
    fix_mol.coords = rec_coords.data();

    // Init box constants around ligand centroid
    std::vector<Real> base_coords(natom * 3);
    for (int i = 0; i < natom; i++) {
        base_coords[i*3]   = ff.lig_atoms.x[i];
        base_coords[i*3+1] = ff.lig_atoms.y[i];
        base_coords[i*3+2] = ff.lig_atoms.z[i];
    }
    Real cx = 0, cy = 0, cz = 0;
    for (int i = 0; i < natom; i++) {
        cx += base_coords[i*3];
        cy += base_coords[i*3+1];
        cz += base_coords[i*3+2];
    }
    cx /= natom; cy /= natom; cz /= natom;

    DockParam dp;
    dp.box.x_lo = cx - 20; dp.box.x_hi = cx + 20;
    dp.box.y_lo = cy - 20; dp.box.y_hi = cy + 20;
    dp.box.z_lo = cz - 20; dp.box.z_hi = cz + 20;
    AtomTypeProps atp;
    init_constants_cpu(dp, atp);

    // --- 3. Pick first bond and strain it ---
    const FFBond& first_bond = ff.lig_bonds[0];
    int a = first_bond.i;
    int b = first_bond.j;
    Real r0 = first_bond.r0;
    REQUIRE(r0 > 1e-6f);

    // Direction vector (b - a), normalized
    Real bax = base_coords[b*3]   - base_coords[a*3];
    Real bay = base_coords[b*3+1] - base_coords[a*3+1];
    Real baz = base_coords[b*3+2] - base_coords[a*3+2];
    Real len = std::sqrt(bax*bax + bay*bay + baz*baz);
    REQUIRE(len > 1e-6f);
    bax /= len; bay /= len; baz /= len;

    // Push atom b by (target_r - current_r) along direction
    Real target_r = r0 + 0.1f;
    Real push = target_r - len;

    // Strained coords: copy base, then move atom b
    std::vector<Real> strained_coords = base_coords;
    strained_coords[b*3]   += push * bax;
    strained_coords[b*3+1] += push * bay;
    strained_coords[b*3+2] += push * baz;

    Real stretched_bond_initial = bond_length(strained_coords.data(), a, b);
    INFO("bond (" << a << "," << b << "): r0=" << r0 << " stretched_initial=" << stretched_bond_initial);
    REQUIRE(stretched_bond_initial > r0 + 0.08f);  // strain actually applied

    // --- 4. Measure E_initial from strained coords ---
    std::vector<Real> aux_f0(natom * 3, 0.0f);
    FlexPose pose_tmp;
    pose_tmp.coords    = strained_coords.data();
    pose_tmp.dihedrals = nullptr;
    pose_tmp.center[0] = cx; pose_tmp.center[1] = cy; pose_tmp.center[2] = cz;
    pose_tmp.rot_vec[0] = pose_tmp.rot_vec[1] = pose_tmp.rot_vec[2] = 0;

    Real E_initial = cal_e_f_ff_cpu(&pose_tmp, topo, fix_mol, ff, aux_f0.data());
    REQUIRE(std::isfinite(E_initial));
    INFO("E_initial (strained) = " << E_initial);

    // --- 5a. Set up Cartesian BFGS ---
    std::vector<Real> coords_cartesian = strained_coords;
    FlexPose pose_cartesian;
    pose_cartesian.coords    = coords_cartesian.data();
    pose_cartesian.dihedrals = nullptr;
    pose_cartesian.center[0] = cx; pose_cartesian.center[1] = cy; pose_cartesian.center[2] = cz;
    pose_cartesian.rot_vec[0] = pose_cartesian.rot_vec[1] = pose_cartesian.rot_vec[2] = 0;

    BFGSThreadMemory mem_cartesian;
    bfgs_ff_cartesian_cpu(&pose_cartesian, topo, fix_mol, ff, /*max_steps=*/200, mem_cartesian);
    Real E_final_cartesian = pose_cartesian.energy;

    // --- 5b. Set up Internal BFGS ---
    std::vector<Real> coords_internal = strained_coords;
    FlexPose pose_internal;
    pose_internal.coords    = coords_internal.data();
    pose_internal.dihedrals = nullptr;
    pose_internal.center[0] = cx; pose_internal.center[1] = cy; pose_internal.center[2] = cz;
    pose_internal.rot_vec[0] = pose_internal.rot_vec[1] = pose_internal.rot_vec[2] = 0;

    BFGSThreadMemory mem_internal;
    bfgs_ff_cpu(&pose_internal, topo, fix_mol, ff, /*max_steps=*/200, mem_internal);
    Real E_final_internal = pose_internal.energy;

    // --- 6. Measure final bond lengths ---
    Real stretched_bond_final_cartesian = bond_length(coords_cartesian.data(), a, b);
    Real stretched_bond_final_internal  = bond_length(coords_internal.data(), a, b);

    // --- 7. Max bond drift for Cartesian mode ---
    Real max_bond_drift_cartesian = 0.0f;
    for (const auto& bd : ff.lig_bonds) {
        int i = bd.i, j = bd.j;
        if (bd.r0 < 1e-6f) continue;
        Real r = bond_length(coords_cartesian.data(), i, j);
        Real drift = std::fabs(r - bd.r0) / bd.r0;
        if (drift > max_bond_drift_cartesian) max_bond_drift_cartesian = drift;
    }

    // --- 8. RMSD between the two final poses ---
    Real rmsd_sum_sq = 0.0f;
    for (int k = 0; k < natom * 3; k++) {
        Real d = coords_cartesian[k] - coords_internal[k];
        rmsd_sum_sq += d * d;
    }
    Real rmsd_cartesian_vs_internal = std::sqrt(rmsd_sum_sq / natom);

    // --- Log all numbers ---
    spdlog::info("[bfgs_ff_real] E_initial={:.4f}  E_final_cart={:.4f}  E_final_int={:.4f}  "
                 "stretched_initial={:.4f}  stretched_final_cart={:.4f}  stretched_final_int={:.4f}  "
                 "rmsd_cart_vs_int={:.4f} A  max_bond_drift_cart={:.4f}",
                 E_initial, E_final_cartesian, E_final_internal,
                 stretched_bond_initial, stretched_bond_final_cartesian, stretched_bond_final_internal,
                 rmsd_cartesian_vs_internal, max_bond_drift_cartesian);

    // =========================================================================
    // ASSERTIONS
    // =========================================================================

    // A. Cartesian mode descended energy
    INFO("E_final_cartesian=" << E_final_cartesian << " E_initial=" << E_initial);
    CHECK(E_final_cartesian <= E_initial);

    // B. Internal mode also descended energy
    INFO("E_final_internal=" << E_final_internal << " E_initial=" << E_initial);
    CHECK(E_final_internal <= E_initial);

    // C. Cartesian mode relaxed the stretched bond by at least 0.05 A
    INFO("stretched_bond_final_cartesian=" << stretched_bond_final_cartesian
         << " threshold=" << (stretched_bond_initial - 0.05f));
    CHECK(stretched_bond_final_cartesian < stretched_bond_initial - 0.05f);

    // D. Internal mode CANNOT relax bonds — bond stays frozen at stretched value
    INFO("stretched_bond_final_internal=" << stretched_bond_final_internal
         << " threshold=" << (stretched_bond_initial - 0.01f));
    CHECK(stretched_bond_final_internal > stretched_bond_initial - 0.01f);

    // E. Poses are genuinely different (not just the same computation path)
    INFO("rmsd_cartesian_vs_internal=" << rmsd_cartesian_vs_internal);
    CHECK(rmsd_cartesian_vs_internal > 0.02f);

    // F. Cartesian mode doesn't blow up bonds
    INFO("max_bond_drift_cartesian=" << max_bond_drift_cartesian);
    CHECK(max_bond_drift_cartesian < 0.10f);

    // G. No NaN / inf in Cartesian final coords
    for (int k = 0; k < natom * 3; k++) {
        INFO("coords_cartesian[" << k << "] = " << coords_cartesian[k]);
        CHECK(std::isfinite(coords_cartesian[k]));
    }

    // G2. No NaN / inf in Internal final coords
    for (int k = 0; k < natom * 3; k++) {
        INFO("coords_internal[" << k << "] = " << coords_internal[k]);
        CHECK(std::isfinite(coords_internal[k]));
    }

    // H. At least one meaningful assertion executed
    CHECK(natom > 0);
}
