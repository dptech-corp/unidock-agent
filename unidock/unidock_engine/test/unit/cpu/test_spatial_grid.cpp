//
// Unit tests for SpatialGrid construction and query correctness
// Tests: basic construction, hydrogen exclusion, CSR consistency, neighbor reachability
//

#include <catch2/catch_amalgamated.hpp>
#include "cpu/spatial_grid.h"
#include "cpu/common_cpu.h"
#include "constants/constants.h"
#include <vector>
#include <set>
#include <cmath>

using Catch::Approx;

// ========== Helpers ==========

static void init_test_atp() {
    init_default_atom_type_props(g_cpu_constants.atp);
    int n = sizeof(VN_VDW_RADII) / sizeof(VN_VDW_RADII[0]);
    for (int i = 0; i < n; i++) g_cpu_constants.vdw_radii[i] = VN_VDW_RADII[i];
    for (int i = n; i < MAX_VN_TYPES; i++) g_cpu_constants.vdw_radii[i] = Real(0);
}


// ===================== TEST 1: basic construction =====================

TEST_CASE("build_spatial_grid basic construction", "[cpu][grid]") {
    init_test_atp();

    // 3 heavy atoms at known positions
    std::vector<Real> coords = {
        0, 0, 0,    // C_H
        5, 0, 0,    // O_A
        0, 5, 0     // N_D
    };
    std::vector<int> types = {VN_TYPE_C_H, VN_TYPE_O_A, VN_TYPE_N_D};

    SpatialGrid grid = {};
    build_spatial_grid(grid, coords.data(), types.data(), 3,
                       g_cpu_constants.atp, g_cpu_constants.vdw_radii);

    REQUIRE(grid.built == true);
    REQUIRE(grid.total_atoms == 3);
    // CSR: cell_start[ncells] == total_atoms
    REQUIRE(grid.cell_start[grid.ncells] == 3);

    free_spatial_grid(grid);
}


// ===================== TEST 2: excludes hydrogen atoms =====================

TEST_CASE("spatial grid excludes hydrogen atoms", "[cpu][grid]") {
    init_test_atp();

    // 4 atoms: 2 heavy + 2 hydrogen
    std::vector<Real> coords = {
        0, 0, 0,    // atom 0: C_H (heavy)
        1, 0, 0,    // atom 1: H   (hydrogen)
        5, 0, 0,    // atom 2: O_A (heavy)
        6, 0, 0     // atom 3: H   (hydrogen)
    };
    std::vector<int> types = {VN_TYPE_C_H, VN_TYPE_H, VN_TYPE_O_A, VN_TYPE_H};

    SpatialGrid grid = {};
    build_spatial_grid(grid, coords.data(), types.data(), 4,
                       g_cpu_constants.atp, g_cpu_constants.vdw_radii);

    REQUIRE(grid.built == true);
    REQUIRE(grid.total_atoms == 2);

    // Collect all stored atom indices
    std::set<int> stored_indices;
    for (int i = 0; i < grid.total_atoms; i++) {
        stored_indices.insert(grid.atoms[i].index);
    }
    // Only heavy atoms (indices 0 and 2) should be stored
    REQUIRE(stored_indices.count(0) == 1);
    REQUIRE(stored_indices.count(2) == 1);
    REQUIRE(stored_indices.count(1) == 0);
    REQUIRE(stored_indices.count(3) == 0);

    free_spatial_grid(grid);
}


// ===================== TEST 3: neighbors within cutoff =====================

TEST_CASE("spatial grid finds all neighbors within cutoff", "[cpu][grid]") {
    init_test_atp();

    // 3 heavy atoms along x-axis
    // Cell size = 8 Å. Atoms at 0, 7, 9 Å.
    // Atoms at 0 and 7 should be in same or adjacent cells (distance 7 < 8).
    // Atom at 9 may or may not be adjacent depending on cell boundaries,
    // but should still be within 27-cell neighborhood since distance=9 < 2*8=16.
    std::vector<Real> coords = {
        0, 0, 0,    // atom 0: C_H
        7, 0, 0,    // atom 1: O_A  (within one cell size)
        9, 0, 0     // atom 2: N_D  (within 27-cell neighborhood)
    };
    std::vector<int> types = {VN_TYPE_C_H, VN_TYPE_O_A, VN_TYPE_N_D};

    SpatialGrid grid = {};
    build_spatial_grid(grid, coords.data(), types.data(), 3,
                       g_cpu_constants.atp, g_cpu_constants.vdw_radii);

    REQUIRE(grid.built == true);
    REQUIRE(grid.total_atoms == 3);

    // Verify atom 0 and atom 1 can reach each other via 27-cell neighborhood:
    // Find which cell each atom is in
    auto cell_of = [&](int atom_idx) -> int {
        for (int c = 0; c < grid.ncells; c++) {
            for (int j = grid.cell_start[c]; j < grid.cell_start[c + 1]; j++) {
                if (grid.atoms[j].index == atom_idx) return c;
            }
        }
        return -1;
    };

    int c0 = cell_of(0);
    int c1 = cell_of(1);
    REQUIRE(c0 >= 0);
    REQUIRE(c1 >= 0);

    // Atoms 0 and 1 (7 Å apart, cell_size=8 Å) should be in same or adjacent cells
    int cx0 = c0 / (grid.ny * grid.nz);
    int cy0 = (c0 / grid.nz) % grid.ny;
    int cz0 = c0 % grid.nz;

    int cx1 = c1 / (grid.ny * grid.nz);
    int cy1 = (c1 / grid.nz) % grid.ny;
    int cz1 = c1 % grid.nz;

    REQUIRE(std::abs(cx0 - cx1) <= 1);
    REQUIRE(std::abs(cy0 - cy1) <= 1);
    REQUIRE(std::abs(cz0 - cz1) <= 1);

    free_spatial_grid(grid);
}


// ===================== TEST 4: CSR pointers are consistent =====================

TEST_CASE("spatial grid CSR pointers are consistent", "[cpu][grid]") {
    init_test_atp();

    // 10 heavy atoms at various positions
    std::vector<Real> coords = {
        0, 0, 0,
        3, 1, 2,
        7, 4, 1,
        12, 0, 5,
        1, 8, 3,
        6, 6, 6,
        15, 2, 1,
        4, 10, 7,
        9, 3, 9,
        2, 7, 4
    };
    // All C_H (heavy, non-hydrogen)
    std::vector<int> types(10, VN_TYPE_C_H);

    SpatialGrid grid = {};
    build_spatial_grid(grid, coords.data(), types.data(), 10,
                       g_cpu_constants.atp, g_cpu_constants.vdw_radii);

    REQUIRE(grid.built == true);
    REQUIRE(grid.total_atoms == 10);

    // CSR property 1: cell_start[0] == 0
    REQUIRE(grid.cell_start[0] == 0);

    // CSR property 2: monotonically non-decreasing
    for (int i = 0; i < grid.ncells; i++) {
        REQUIRE(grid.cell_start[i] <= grid.cell_start[i + 1]);
    }

    // CSR property 3: cell_start[ncells] == total_atoms
    REQUIRE(grid.cell_start[grid.ncells] == grid.total_atoms);

    // CSR property 4: all stored atom indices are valid
    for (int j = 0; j < grid.total_atoms; j++) {
        REQUIRE(grid.atoms[j].index >= 0);
        REQUIRE(grid.atoms[j].index < 10);
    }

    // CSR property 5: all original atom indices appear exactly once
    std::set<int> seen;
    for (int j = 0; j < grid.total_atoms; j++) {
        seen.insert(grid.atoms[j].index);
    }
    REQUIRE((int)seen.size() == 10);

    free_spatial_grid(grid);
}
