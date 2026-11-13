//
// Unit tests for the constrained_atom_mask feature in bfgs_ff_cartesian_cpu.
//
// Two test cases:
//
// 1. "bfgs_ff_cartesian_cpu with atom mask: constrained atoms stay fixed"
//    - Loads ejm42 ligand + tyk2 receptor.
//    - Strains the pose (stretch first bond by 0.1 A).
//    - Constrains atoms 0-9 via a mask, runs 200 BFGS steps.
//    - Verifies constrained atoms have zero displacement (bit-identical),
//      unconstrained atoms moved meaningfully, and energy descended.
//
// 2. "bfgs_ff_cartesian_cpu with empty mask: same as nullptr"
//    - Empty std::vector<bool> mask should behave like no mask.
//    - Final energy should match the unconstrained result (~-237.16 kcal/mol).
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
#include <bit>

static const std::string DMS_MASK_DIR = std::string(TEST_DATA_DIR) + "ff_minimize/tyk2/";

// ============================================================================
// Helper: Euclidean distance between two 3D points stored in a flat coord array
// ============================================================================
static float atom_displacement(const Real* a, const Real* b) {
    float dx = a[0] - b[0];
    float dy = a[1] - b[1];
    float dz = a[2] - b[2];
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// ============================================================================
// Common fixture builder — avoids duplicating DMS load + FlexTopo + FixMol setup
// ============================================================================
struct MaskTestFixture {
    FFMinData ff;
    FlexTopo topo;
    FixMol fix_mol;
    std::vector<int> vn_types;
    std::vector<Real> rec_coords;
    std::vector<Real> base_coords;
    std::vector<Real> strained_coords;
    int natom = 0;
    int bond_a = 0, bond_b = 0;
    Real cx = 0, cy = 0, cz = 0;

    bool build() {
        read_ligand_dms(DMS_MASK_DIR + "ligand_ejm42.dms", ff);
        read_receptor_dms(DMS_MASK_DIR + "protein.dms", ff);
        if (!ff.loaded || ff.lig_bonds.empty()) return false;

        natom = ff.lig_atoms.natom;
        if (natom < 10) return false;  // need at least 10 atoms for scaffold mask

        // FlexTopo: Cartesian mode, no torsions
        topo.natom      = natom;
        topo.ntorsion   = 0;
        vn_types.assign(natom, 0);
        topo.vn_types      = vn_types.data();
        topo.axis_atoms    = nullptr;
        topo.range_inds    = nullptr;
        topo.range_list    = nullptr;
        topo.rotated_inds  = nullptr;
        topo.rotated_atoms = nullptr;

        // FixMol from receptor
        fix_mol.natom = ff.rec_atoms.natom;
        rec_coords.resize(fix_mol.natom * 3);
        for (int i = 0; i < fix_mol.natom; i++) {
            rec_coords[i*3]   = ff.rec_atoms.x[i];
            rec_coords[i*3+1] = ff.rec_atoms.y[i];
            rec_coords[i*3+2] = ff.rec_atoms.z[i];
        }
        fix_mol.coords = rec_coords.data();

        // Base ligand coords
        base_coords.resize(natom * 3);
        for (int i = 0; i < natom; i++) {
            base_coords[i*3]   = ff.lig_atoms.x[i];
            base_coords[i*3+1] = ff.lig_atoms.y[i];
            base_coords[i*3+2] = ff.lig_atoms.z[i];
        }

        // Centroid for box
        cx = cy = cz = 0;
        for (int i = 0; i < natom; i++) {
            cx += base_coords[i*3];
            cy += base_coords[i*3+1];
            cz += base_coords[i*3+2];
        }
        cx /= natom; cy /= natom; cz /= natom;

        // Init box constants
        DockParam dp;
        dp.box.x_lo = cx - 20; dp.box.x_hi = cx + 20;
        dp.box.y_lo = cy - 20; dp.box.y_hi = cy + 20;
        dp.box.z_lo = cz - 20; dp.box.z_hi = cz + 20;
        AtomTypeProps atp;
        init_constants_cpu(dp, atp);

        // Strain first bond: push atom b so bond = r0 + 0.1 A
        const FFBond& fb = ff.lig_bonds[0];
        bond_a = fb.i;
        bond_b = fb.j;
        Real r0 = fb.r0;

        Real bax = base_coords[bond_b*3]   - base_coords[bond_a*3];
        Real bay = base_coords[bond_b*3+1] - base_coords[bond_a*3+1];
        Real baz = base_coords[bond_b*3+2] - base_coords[bond_a*3+2];
        Real len = std::sqrt(bax*bax + bay*bay + baz*baz);
        if (len < 1e-6f) return false;
        bax /= len; bay /= len; baz /= len;

        Real push = (r0 + 0.1f) - len;
        strained_coords = base_coords;
        strained_coords[bond_b*3]   += push * bax;
        strained_coords[bond_b*3+1] += push * bay;
        strained_coords[bond_b*3+2] += push * baz;

        return true;
    }
};

// ============================================================================
// TEST CASE 1: constrained atoms stay bit-identical
// ============================================================================
TEST_CASE("bfgs_ff_cartesian_cpu with atom mask: constrained atoms stay fixed", "[bfgs_ff_mask]") {

    MaskTestFixture fix;
    REQUIRE(fix.build());

    int natom = fix.natom;
    REQUIRE(natom == 35);

    // Measure E_initial from strained coords
    std::vector<Real> aux_f0(natom * 3, 0.0f);
    FlexPose pose_tmp;
    pose_tmp.coords    = fix.strained_coords.data();
    pose_tmp.dihedrals = nullptr;
    pose_tmp.center[0] = fix.cx; pose_tmp.center[1] = fix.cy; pose_tmp.center[2] = fix.cz;
    pose_tmp.rot_vec[0] = pose_tmp.rot_vec[1] = pose_tmp.rot_vec[2] = 0;
    Real E_initial = cal_e_f_ff_cpu(&pose_tmp, fix.topo, fix.fix_mol, fix.ff, aux_f0.data());
    REQUIRE(std::isfinite(E_initial));

    // Build the mask: constrain scaffold = atoms 0-9
    std::vector<bool> mask(natom, false);
    for (int i = 0; i < 10; i++) mask[i] = true;

    // Set up the pose for masked BFGS
    std::vector<Real> coords_masked = fix.strained_coords;
    FlexPose pose_masked;
    pose_masked.coords    = coords_masked.data();
    pose_masked.dihedrals = nullptr;
    pose_masked.center[0] = fix.cx; pose_masked.center[1] = fix.cy; pose_masked.center[2] = fix.cz;
    pose_masked.rot_vec[0] = pose_masked.rot_vec[1] = pose_masked.rot_vec[2] = 0;

    // Set deadline so the loop runs
    set_pose_deadline(60.0);

    BFGSThreadMemory mem;
    bfgs_ff_cartesian_cpu(&pose_masked, fix.topo, fix.fix_mol, fix.ff, 200, mem, &mask);
    Real E_final = pose_masked.energy;

    // -------------------------------------------------------------------------
    // Measurements
    // -------------------------------------------------------------------------

    // Max displacement among constrained atoms (0-9)
    float max_disp_constrained = 0.0f;
    for (int i = 0; i < 10; i++) {
        float d = atom_displacement(&coords_masked[i*3], &fix.strained_coords[i*3]);
        if (d > max_disp_constrained) max_disp_constrained = d;
    }

    // Max displacement among unconstrained atoms (10-34)
    float max_disp_unconstrained = 0.0f;
    for (int i = 10; i < natom; i++) {
        float d = atom_displacement(&coords_masked[i*3], &fix.strained_coords[i*3]);
        if (d > max_disp_unconstrained) max_disp_unconstrained = d;
    }

    // Bit-identical count: 30 float components for atoms 0-9
    int bit_identical_count = 0;
    for (int i = 0; i < 10; i++) {
        for (int c = 0; c < 3; c++) {
            uint32_t orig, final_val;
            std::memcpy(&orig,      &fix.strained_coords[i*3+c], sizeof(uint32_t));
            std::memcpy(&final_val, &coords_masked[i*3+c],       sizeof(uint32_t));
            if (orig == final_val) bit_identical_count++;
        }
    }

    // Log all numbers
    spdlog::info("[bfgs_ff_mask] E_initial={:.4f} E_final={:.4f} "
                 "max_disp_constrained={:.6e} max_disp_unconstrained={:.4f} "
                 "bit_identical={}/30",
                 E_initial, E_final,
                 max_disp_constrained, max_disp_unconstrained,
                 bit_identical_count);

    // =========================================================================
    // ASSERTIONS
    // =========================================================================

    // A. Constrained atoms must not move at all (gradient zeroed -> zero p -> zero displacement)
    INFO("max_disp_constrained=" << max_disp_constrained);
    CHECK(max_disp_constrained == 0.0f);

    // B. Unconstrained atoms moved meaningfully (BFGS actually ran)
    INFO("max_disp_unconstrained=" << max_disp_unconstrained);
    CHECK(max_disp_unconstrained > 0.05f);

    // C. Energy descended
    INFO("E_final=" << E_final << " E_initial=" << E_initial);
    CHECK(E_final <= E_initial);

    // D. All 30 float components of constrained atoms are bit-identical
    INFO("bit_identical_count=" << bit_identical_count);
    CHECK(bit_identical_count == 30);

    // E. No NaN in final coords
    for (int k = 0; k < natom * 3; k++) {
        INFO("coords_masked[" << k << "] = " << coords_masked[k]);
        CHECK(std::isfinite(coords_masked[k]));
    }
}

// ============================================================================
// TEST CASE 2: empty mask behaves like nullptr (no constraints)
// ============================================================================
TEST_CASE("bfgs_ff_cartesian_cpu with empty mask: same as nullptr", "[bfgs_ff_mask]") {

    MaskTestFixture fix;
    REQUIRE(fix.build());

    int natom = fix.natom;
    REQUIRE(natom == 35);

    // Empty mask: apply_atom_mask_to_grad returns immediately on empty check
    std::vector<bool> emptyMask;

    std::vector<Real> coords_empty = fix.strained_coords;
    FlexPose pose_empty;
    pose_empty.coords    = coords_empty.data();
    pose_empty.dihedrals = nullptr;
    pose_empty.center[0] = fix.cx; pose_empty.center[1] = fix.cy; pose_empty.center[2] = fix.cz;
    pose_empty.rot_vec[0] = pose_empty.rot_vec[1] = pose_empty.rot_vec[2] = 0;

    set_pose_deadline(60.0);

    BFGSThreadMemory mem;
    bfgs_ff_cartesian_cpu(&pose_empty, fix.topo, fix.fix_mol, fix.ff, 200, mem, &emptyMask);
    Real E_final_empty = pose_empty.energy;

    spdlog::info("[bfgs_ff_mask/empty] E_final_empty={:.4f}", E_final_empty);

    // Should reach roughly the same minimum as the unconstrained run (~-237.16 kcal/mol)
    INFO("E_final_empty=" << E_final_empty << " expected ~-237.16");
    CHECK(std::fabs(E_final_empty - (-237.16f)) < 0.5f);
}
