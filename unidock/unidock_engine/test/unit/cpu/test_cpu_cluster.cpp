//
// Unit tests for CPU clustering functions
// Created for Uni-Dock2 CPU backend validation
//

#include <catch2/catch_amalgamated.hpp>
#include "cpu/cluster_cpu.h"
#include "model/model.h"
#include <cmath>
#include <vector>
#include <numeric>

using Catch::Approx;

// Helper: build sequential indices [0, 1, ..., n-1]
static std::vector<int> seq_inds(int n) {
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    return v;
}


TEST_CASE("cal_rmsd_pair_cpu identical poses", "[cpu][cluster]") {
    const int natom = 4;
    std::vector<Real> coords(natom * 3, 0.0);
    // Set up known coordinates: (1,0,0), (0,1,0), (0,0,1), (1,1,1)
    coords[0] = 1; coords[1] = 0; coords[2] = 0;
    coords[3] = 0; coords[4] = 1; coords[5] = 0;
    coords[6] = 0; coords[7] = 0; coords[8] = 1;
    coords[9] = 1; coords[10] = 1; coords[11] = 1;

    FlexPose pose1, pose2;
    pose1.coords = coords.data();
    pose2.coords = coords.data();

    auto inds = seq_inds(natom);
    Real rmsd = cal_rmsd_pair_cpu(&pose1, &pose2, inds.data(), natom);
    REQUIRE(rmsd == Approx(0.0));
}


TEST_CASE("cal_rmsd_pair_cpu known displacement", "[cpu][cluster]") {
    const int natom = 3;
    // Pose1: (0,0,0), (1,0,0), (0,1,0)
    std::vector<Real> coords1 = {0, 0, 0, 1, 0, 0, 0, 1, 0};
    // Pose2: shift all by (1,0,0) => (1,0,0), (2,0,0), (1,1,0)
    std::vector<Real> coords2 = {1, 0, 0, 2, 0, 0, 1, 1, 0};

    FlexPose pose1, pose2;
    pose1.coords = coords1.data();
    pose2.coords = coords2.data();

    // RMSD = sqrt( (1^2 + 1^2 + 1^2) / 3 ) = sqrt(3/3) = 1.0
    auto inds = seq_inds(natom);
    Real rmsd = cal_rmsd_pair_cpu(&pose1, &pose2, inds.data(), natom);
    REQUIRE(rmsd == Approx(1.0));
}


TEST_CASE("cal_rmsd_pair_cpu single atom displacement", "[cpu][cluster]") {
    const int natom = 1;
    std::vector<Real> coords1 = {0, 0, 0};
    std::vector<Real> coords2 = {3, 4, 0};

    FlexPose pose1, pose2;
    pose1.coords = coords1.data();
    pose2.coords = coords2.data();

    // RMSD = sqrt( (9 + 16 + 0) / 1 ) = 5.0
    auto inds = seq_inds(natom);
    Real rmsd = cal_rmsd_pair_cpu(&pose1, &pose2, inds.data(), natom);
    REQUIRE(rmsd == Approx(5.0));
}


TEST_CASE("cal_rmsd_pair_cpu zero atoms", "[cpu][cluster]") {
    FlexPose pose1, pose2;
    pose1.coords = nullptr;
    pose2.coords = nullptr;

    Real rmsd = cal_rmsd_pair_cpu(&pose1, &pose2, nullptr, 0);
    REQUIRE(rmsd == Approx(0.0));
}


TEST_CASE("cluster_cpu all identical poses", "[cpu][cluster]") {
    const int nflex = 1;
    const int exhaustiveness = 4;
    const int natom = 2;
    const Real rmsd_limit = 1.0;

    // All poses have identical coordinates
    std::vector<Real> coords(natom * 3, 0.0);
    coords[0] = 1; coords[1] = 2; coords[2] = 3;
    coords[3] = 4; coords[4] = 5; coords[5] = 6;

    std::vector<FlexPose> poses(nflex * exhaustiveness);
    // Use separate coord buffers so pointers are valid, but identical values
    std::vector<std::vector<Real>> coord_buffers(nflex * exhaustiveness, coords);
    for (int i = 0; i < nflex * exhaustiveness; i++) {
        poses[i].coords = coord_buffers[i].data();
        poses[i].energy = static_cast<Real>(i);  // Different energies
    }

    FlexTopo topo;
    topo.natom = natom;

    std::vector<int> clustered_inds(nflex * exhaustiveness, -1);
    int npose_clustered = 0;
    std::vector<std::vector<int>> clustered_list;

    // Pass explicit all-atom indices for each flex
    std::vector<std::vector<int>> atom_inds_per_flex = {seq_inds(natom)};
    cluster_cpu(clustered_inds.data(), &npose_clustered,
                &clustered_list, poses.data(), &topo,
                nflex, exhaustiveness, rmsd_limit, atom_inds_per_flex);

    // All identical => only 1 cluster representative (the lowest-energy one)
    REQUIRE(npose_clustered == 1);
    REQUIRE(clustered_list.size() == 1);
    REQUIRE(clustered_list[0].size() == 1);
    REQUIRE(clustered_list[0][0] == 0);  // Lowest energy pose
}


TEST_CASE("cluster_cpu well-separated poses", "[cpu][cluster]") {
    const int nflex = 1;
    const int exhaustiveness = 3;
    const int natom = 1;
    const Real rmsd_limit = 1.0;

    // Three poses far apart: (0,0,0), (10,0,0), (20,0,0)
    std::vector<Real> coords0 = {0, 0, 0};
    std::vector<Real> coords1 = {10, 0, 0};
    std::vector<Real> coords2 = {20, 0, 0};

    std::vector<FlexPose> poses(exhaustiveness);
    poses[0].coords = coords0.data();
    poses[0].energy = 1.0;
    poses[1].coords = coords1.data();
    poses[1].energy = 2.0;
    poses[2].coords = coords2.data();
    poses[2].energy = 3.0;

    FlexTopo topo;
    topo.natom = natom;

    std::vector<int> clustered_inds(exhaustiveness, -1);
    int npose_clustered = 0;
    std::vector<std::vector<int>> clustered_list;

    std::vector<std::vector<int>> atom_inds_per_flex = {seq_inds(natom)};
    cluster_cpu(clustered_inds.data(), &npose_clustered,
                &clustered_list, poses.data(), &topo,
                nflex, exhaustiveness, rmsd_limit, atom_inds_per_flex);

    // All well-separated => 3 cluster representatives
    REQUIRE(npose_clustered == 3);
    REQUIRE(clustered_list[0].size() == 3);
}


TEST_CASE("cluster_cpu multiple flex molecules", "[cpu][cluster]") {
    const int nflex = 2;
    const int exhaustiveness = 2;
    const int natom = 1;
    const Real rmsd_limit = 1.0;

    // Flex 0: two identical poses => 1 cluster
    // Flex 1: two far-apart poses => 2 clusters
    std::vector<Real> c0a = {0, 0, 0};
    std::vector<Real> c0b = {0, 0, 0};
    std::vector<Real> c1a = {0, 0, 0};
    std::vector<Real> c1b = {100, 0, 0};

    std::vector<FlexPose> poses(nflex * exhaustiveness);
    poses[0].coords = c0a.data(); poses[0].energy = 1.0;
    poses[1].coords = c0b.data(); poses[1].energy = 2.0;
    poses[2].coords = c1a.data(); poses[2].energy = 1.0;
    poses[3].coords = c1b.data(); poses[3].energy = 2.0;

    FlexTopo topos[2];
    topos[0].natom = natom;
    topos[1].natom = natom;

    std::vector<int> clustered_inds(nflex * exhaustiveness, -1);
    int npose_clustered = 0;
    std::vector<std::vector<int>> clustered_list;

    std::vector<std::vector<int>> atom_inds_per_flex = {seq_inds(natom), seq_inds(natom)};
    cluster_cpu(clustered_inds.data(), &npose_clustered,
                &clustered_list, poses.data(), topos,
                nflex, exhaustiveness, rmsd_limit, atom_inds_per_flex);

    REQUIRE(clustered_list.size() == 2);
    REQUIRE(clustered_list[0].size() == 1);  // flex 0: 1 cluster
    REQUIRE(clustered_list[1].size() == 2);  // flex 1: 2 clusters
    REQUIRE(npose_clustered == 3);
}


TEST_CASE("cluster_cpu energy ordering", "[cpu][cluster]") {
    const int nflex = 1;
    const int exhaustiveness = 3;
    const int natom = 1;
    const Real rmsd_limit = 0.5;

    // Two close poses and one far: (0,0,0), (0.1,0,0), (10,0,0)
    // Give second the lowest energy so it becomes the cluster representative
    std::vector<Real> c0 = {0, 0, 0};
    std::vector<Real> c1 = {0.1, 0, 0};
    std::vector<Real> c2 = {10, 0, 0};

    std::vector<FlexPose> poses(exhaustiveness);
    poses[0].coords = c0.data(); poses[0].energy = 5.0;
    poses[1].coords = c1.data(); poses[1].energy = -2.0;  // lowest energy
    poses[2].coords = c2.data(); poses[2].energy = 3.0;

    FlexTopo topo;
    topo.natom = natom;

    std::vector<int> clustered_inds(exhaustiveness, -1);
    int npose_clustered = 0;
    std::vector<std::vector<int>> clustered_list;

    std::vector<std::vector<int>> atom_inds_per_flex = {seq_inds(natom)};
    cluster_cpu(clustered_inds.data(), &npose_clustered,
                &clustered_list, poses.data(), &topo,
                nflex, exhaustiveness, rmsd_limit, atom_inds_per_flex);

    // Pose 1 (lowest energy) should be first in sorted order
    // Poses 0 and 1 are close (RMSD = 0.1 < 0.5), so pose 0 is deactivated
    // Result: pose 1 and pose 2 survive
    REQUIRE(npose_clustered == 2);
    REQUIRE(clustered_list[0].size() == 2);
    // First selected should be pose 1 (lowest energy)
    REQUIRE(clustered_list[0][0] == 1);
}


TEST_CASE("cal_rmsd_pair_cpu custom atom subset", "[cpu][cluster]") {
    // 3 atoms: atom0 differs between poses, atoms 1-2 are identical
    // Pose A: (0,0,0), (5,5,5), (9,9,9)
    // Pose B: (10,0,0), (5,5,5), (9,9,9)
    std::vector<Real> cA = {0,0,0,  5,5,5,  9,9,9};
    std::vector<Real> cB = {10,0,0, 5,5,5,  9,9,9};

    FlexPose pA, pB;
    pA.coords = cA.data();
    pB.coords = cB.data();

    // All atoms => RMSD = sqrt((100+0+0)/3) ≈ 5.77
    auto all = seq_inds(3);
    Real rmsd_all = cal_rmsd_pair_cpu(&pA, &pB, all.data(), 3);
    REQUIRE(rmsd_all == Approx(std::sqrt(100.0 / 3)));

    // Only atom 0 => RMSD = sqrt(100/1) = 10.0
    std::vector<int> only0 = {0};
    Real rmsd_0 = cal_rmsd_pair_cpu(&pA, &pB, only0.data(), 1);
    REQUIRE(rmsd_0 == Approx(10.0));

    // Only atoms 1,2 => RMSD = 0 (identical)
    std::vector<int> only12 = {1, 2};
    Real rmsd_12 = cal_rmsd_pair_cpu(&pA, &pB, only12.data(), 2);
    REQUIRE(rmsd_12 == Approx(0.0));
}


TEST_CASE("cluster_cpu custom atom subset changes clustering result", "[cpu][cluster]") {
    // Two poses with 3 atoms each. They differ only in atom 0.
    // Atom 0: (0,0,0) vs (10,0,0)  => RMSD on atom 0 alone = 10
    // Atoms 1,2: identical          => RMSD on atoms 1,2 = 0
    //
    // With rmsd_limit=5:
    //   - clustering on atom subset {0}    => both survive (RMSD=10 > 5)
    //   - clustering on atom subset {1,2}  => merged to 1 (RMSD=0 < 5)
    const int nflex = 1;
    const int exhaustiveness = 2;
    const int natom = 3;
    const Real rmsd_limit = 5.0;

    std::vector<Real> cA = {0,0,0,  5,5,5,  9,9,9};
    std::vector<Real> cB = {10,0,0, 5,5,5,  9,9,9};

    std::vector<FlexPose> poses(exhaustiveness);
    poses[0].coords = cA.data(); poses[0].energy = 1.0;
    poses[1].coords = cB.data(); poses[1].energy = 2.0;

    FlexTopo topo;
    topo.natom = natom;

    // --- Case 1: cluster using only atom 0 => RMSD=10 > 5, both survive ---
    {
        std::vector<int> clustered_inds(exhaustiveness, -1);
        int npose_clustered = 0;
        std::vector<std::vector<int>> clustered_list;

        std::vector<std::vector<int>> atom_inds = {{0}};
        cluster_cpu(clustered_inds.data(), &npose_clustered,
                    &clustered_list, poses.data(), &topo,
                    nflex, exhaustiveness, rmsd_limit, atom_inds);

        REQUIRE(npose_clustered == 2);
        REQUIRE(clustered_list[0].size() == 2);
    }

    // --- Case 2: cluster using atoms 1,2 => RMSD=0 < 5, merged to 1 ---
    {
        std::vector<int> clustered_inds(exhaustiveness, -1);
        int npose_clustered = 0;
        std::vector<std::vector<int>> clustered_list;

        std::vector<std::vector<int>> atom_inds = {{1, 2}};
        cluster_cpu(clustered_inds.data(), &npose_clustered,
                    &clustered_list, poses.data(), &topo,
                    nflex, exhaustiveness, rmsd_limit, atom_inds);

        REQUIRE(npose_clustered == 1);
        REQUIRE(clustered_list[0].size() == 1);
        REQUIRE(clustered_list[0][0] == 0);  // lowest energy
    }
}
