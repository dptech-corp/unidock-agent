//
// Unit tests for CPU scoring functions
// Created for Uni-Dock2 CPU backend validation
//

#include <catch2/catch_amalgamated.hpp>
#include "cpu/common_cpu.h"
#include "cpu/score_cpu.h"
#include "model/model.h"
#include <cmath>
#include <vector>

using Catch::Approx;

TEST_CASE("CPU Common Math Functions", "[cpu][math]") {
    
    SECTION("dist_sq_cpu") {
        Real a[3] = {0, 0, 0};
        Real b[3] = {1, 0, 0};
        REQUIRE(dist_sq_cpu(a, b) == Approx(1.0));
        
        Real c[3] = {1, 1, 1};
        REQUIRE(dist_sq_cpu(a, c) == Approx(3.0));
    }
    
    SECTION("norm_cpu") {
        Real v[3] = {3, 4, 0};
        REQUIRE(norm_cpu(v) == Approx(5.0));
    }
    
    SECTION("cross_cpu") {
        Real a[3] = {1, 0, 0};
        Real b[3] = {0, 1, 0};
        Real out[3];
        cross_cpu(a, b, out);
        REQUIRE(out[0] == Approx(0.0));
        REQUIRE(out[1] == Approx(0.0));
        REQUIRE(out[2] == Approx(1.0));
    }
    
    SECTION("dot_cpu") {
        Real a[3] = {1, 2, 3};
        Real b[3] = {4, 5, 6};
        REQUIRE(dot_cpu(a, b) == Approx(32.0));
    }
}


TEST_CASE("CPU Quaternion Operations", "[cpu][quaternion]") {
    
    SECTION("rot_vec_to_quaternion_cpu identity") {
        Real rot_vec[3] = {0, 0, 0};
        Real q[4];
        rot_vec_to_quaternion_cpu(rot_vec, q);
        REQUIRE(q[0] == Approx(1.0));
        REQUIRE(q[1] == Approx(0.0));
        REQUIRE(q[2] == Approx(0.0));
        REQUIRE(q[3] == Approx(0.0));
    }
    
    SECTION("rotate_by_quaternion_cpu") {
        // Rotate (1, 0, 0) by 90 degrees around z-axis should give (0, 1, 0)
        Real q[4] = {static_cast<Real>(std::cos(M_PI/4)), 0, 0, static_cast<Real>(std::sin(M_PI/4))};  // 90 deg around z
        Real v[3] = {1, 0, 0};
        Real out[3];
        rotate_by_quaternion_cpu(q, v, out);
        REQUIRE(out[0] == Approx(0.0).margin(1e-5));
        REQUIRE(out[1] == Approx(1.0).margin(1e-5));
        REQUIRE(out[2] == Approx(0.0).margin(1e-5));
    }
}


TEST_CASE("CPU Box Penalty", "[cpu][score]") {
    // Initialize constants for test
    g_cpu_constants.BOX_X_LO = -10;
    g_cpu_constants.BOX_X_HI = 10;
    g_cpu_constants.BOX_Y_LO = -10;
    g_cpu_constants.BOX_Y_HI = 10;
    g_cpu_constants.BOX_Z_LO = -10;
    g_cpu_constants.BOX_Z_HI = 10;
    g_cpu_constants.PENALTY_SLOPE = 10.0;
    
    SECTION("Inside box - no penalty") {
        Real x = 0, y = 0, z = 0;
        Real f[3] = {0, 0, 0};
        Real penalty = cal_box_penalty_clamp_coord_cpu(&x, &y, &z, f);
        REQUIRE(penalty == Approx(0.0));
        REQUIRE(x == Approx(0.0));
    }
    
    SECTION("Outside box - penalty applied") {
        Real x = 15, y = 0, z = 0;  // 5 units outside
        Real f[3] = {0, 0, 0};
        Real penalty = cal_box_penalty_clamp_coord_cpu(&x, &y, &z, f);
        REQUIRE(penalty > 0);
        REQUIRE(x == Approx(10.0));  // Clamped to box boundary
    }
}


TEST_CASE("CPU Pose Duplication", "[cpu][pose]") {
    
    SECTION("duplicate_pose_cpu") {
        const int natom = 10;
        const int ntorsion = 3;
        const int dim = dof_x + ntorsion;  // 7 + 3 = 10
        
        std::vector<Real> coords1(natom * 3, 1.0);
        std::vector<Real> coords2(natom * 3, 0.0);
        std::vector<Real> dihedrals1(ntorsion, 0.5);
        std::vector<Real> dihedrals2(ntorsion, 0.0);
        
        FlexPose pose1, pose2;
        pose1.coords = coords1.data();
        pose1.dihedrals = dihedrals1.data();
        pose1.center[0] = 1; pose1.center[1] = 2; pose1.center[2] = 3;
        pose1.energy = 42.0;
        
        pose2.coords = coords2.data();
        pose2.dihedrals = dihedrals2.data();
        
        duplicate_pose_cpu(&pose2, &pose1, dim, natom);
        
        REQUIRE(pose2.center[0] == Approx(1.0));
        REQUIRE(pose2.center[1] == Approx(2.0));
        REQUIRE(pose2.center[2] == Approx(3.0));
        REQUIRE(pose2.energy == Approx(42.0));
        REQUIRE(pose2.coords[0] == Approx(1.0));
        REQUIRE(pose2.dihedrals[0] == Approx(0.5));
    }
}


TEST_CASE("CPU Gradient Operations", "[cpu][gradient]") {
    
    SECTION("duplicate_grad_cpu") {
        const int ntorsion = 3;
        const int dim = dof_g + ntorsion;  // 6 + 3 = 9
        
        std::vector<Real> dihedrals_g1(ntorsion, 0.1);
        std::vector<Real> dihedrals_g2(ntorsion, 0.0);
        
        FlexPoseGradient g1, g2;
        g1.dihedrals_g = dihedrals_g1.data();
        g1.center_g[0] = 1; g1.center_g[1] = 2; g1.center_g[2] = 3;
        g1.orientation_g[0] = 4; g1.orientation_g[1] = 5; g1.orientation_g[2] = 6;
        
        g2.dihedrals_g = dihedrals_g2.data();
        
        duplicate_grad_cpu(&g2, &g1, dim);
        
        REQUIRE(g2.center_g[0] == Approx(1.0));
        REQUIRE(g2.orientation_g[2] == Approx(6.0));
        REQUIRE(g2.dihedrals_g[0] == Approx(0.1));
    }
}


TEST_CASE("CPU Normalize Angle", "[cpu][math]") {
    
    SECTION("normalize_angle_cpu") {
        REQUIRE(normalize_angle_cpu(0) == Approx(0.0));
        REQUIRE(normalize_angle_cpu(static_cast<Real>(M_PI)) == Approx(M_PI).margin(1e-5));
        REQUIRE(normalize_angle_cpu(static_cast<Real>(2 * M_PI)) == Approx(0.0).margin(1e-5));
        // 3*PI - 2*PI = PI, so normalize(3*PI) = PI (not -PI)
        REQUIRE(normalize_angle_cpu(static_cast<Real>(3 * M_PI)) == Approx(M_PI).margin(1e-5));
    }
}
