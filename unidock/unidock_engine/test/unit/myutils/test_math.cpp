//
// Created by Congcong Liu on 24-10-16.
//

#include <cmath>
#include <catch2/catch_amalgamated.hpp>
#include "myutils/mymath.h"

TEST_CASE("normalize_angle normalizes an angle to the range [-π, π)", "[normalize_angle]") {
    Real margin = 1e-4;
    REQUIRE_THAT(normalize_angle(0.0f), Catch::Matchers::WithinAbs(0.0f, margin));
    REQUIRE_THAT(normalize_angle(PI), Catch::Matchers::WithinAbs(PI, margin)); // 注意，PI
    REQUIRE_THAT(normalize_angle(-PI), Catch::Matchers::WithinAbs(-PI, margin));
    REQUIRE_THAT(normalize_angle(2 * PI), Catch::Matchers::WithinAbs(0.0f, margin));
    REQUIRE_THAT(normalize_angle(-2 * PI), Catch::Matchers::WithinAbs(0.0f, margin));
    REQUIRE_THAT(normalize_angle(3 * PI), Catch::Matchers::WithinAbs(PI, margin));
    REQUIRE_THAT(normalize_angle(-3 * PI), Catch::Matchers::WithinAbs(-PI, margin));
    //![normalize_angle]
}

TEST_CASE("normalize_angle normalizes an angle to the range [0, 2π)", "[normalize_angle]") {
    Real margin = 1e-4;
    REQUIRE_THAT(normalize_angle_2pi(0.0f), Catch::Matchers::WithinAbs(0.0f, margin));
    REQUIRE_THAT(normalize_angle_2pi(PI), Catch::Matchers::WithinAbs(PI, margin)); // 注意，PI 会被转换为 -PI
    REQUIRE_THAT(normalize_angle_2pi(-PI), Catch::Matchers::WithinAbs(PI, margin));
    REQUIRE_THAT(normalize_angle_2pi(2 * PI), Catch::Matchers::WithinAbs(0.0f, margin));
    REQUIRE_THAT(normalize_angle_2pi(-2 * PI), Catch::Matchers::WithinAbs(0.0f, margin));
    REQUIRE_THAT(normalize_angle_2pi(0.3 + 13 * PI), Catch::Matchers::WithinAbs(3.441592, margin));
    REQUIRE_THAT(normalize_angle_2pi(0.3 - 13 * PI), Catch::Matchers::WithinAbs(3.441592, margin));
    //![normalize_angle]
}


TEST_CASE("ang_to_rad converts an angle in degree to radian", "[ang_to_rad]") {
    REQUIRE_THAT(ang_to_rad(0), Catch::Matchers::WithinAbs(0, 1e-4));
    REQUIRE_THAT(ang_to_rad(180), Catch::Matchers::WithinAbs(PI, 1e-4));
    REQUIRE_THAT(ang_to_rad(90), Catch::Matchers::WithinAbs(PI / 2, 1e-4));
    REQUIRE_THAT(ang_to_rad(-180), Catch::Matchers::WithinAbs(-PI, 1e-4));
    //![ang_to_rad]
}


TEST_CASE("get_radian_in_ranges randomly selects a value within a given range", "[get_radian_in_ranges]") {
    Real ranges[] = {-180, -160, 20, 90, 110, 150};
    int num_range = 3;

    Real rand[] = {0.54f, 0.1f};
    REQUIRE_THAT(get_radian_in_ranges(ranges, num_range, rand), Catch::Matchers::WithinAbs(27, 1e-4));

    Real rand2[] = {0.89f, 1e-6};
    REQUIRE_THAT(get_radian_in_ranges(ranges, num_range, rand2), Catch::Matchers::WithinAbs(110, 1e-4));

    Real rand3[] = {0.21f, 1.0f}  ;
    REQUIRE_THAT(get_radian_in_ranges(ranges, num_range, rand3), Catch::Matchers::WithinAbs(-160, 1e-4));
    //![get_radian_in_ranges]
}


TEST_CASE("circular_distance is the smallest angle around the circle", "[circular_distance]") {
    Real margin = 1e-4;
    REQUIRE_THAT(circular_distance(0.1f, -0.1f), Catch::Matchers::WithinAbs(0.2f, margin));
    REQUIRE_THAT(circular_distance(PI - 0.1f, -PI + 0.1f), Catch::Matchers::WithinAbs(0.2f, margin));
    REQUIRE_THAT(circular_distance(-PI + 0.1f, PI - 0.1f), Catch::Matchers::WithinAbs(0.2f, margin));
    REQUIRE_THAT(circular_distance(0.0f, PI), Catch::Matchers::WithinAbs(PI, margin));
    REQUIRE_THAT(circular_distance(PI, -PI), Catch::Matchers::WithinAbs(0.0f, margin));
    REQUIRE_THAT(circular_distance(1.0f, 1.0f), Catch::Matchers::WithinAbs(0.0f, margin));
    //![circular_distance]
}


TEST_CASE("clamp_by_ranges keeps an angle inside a range and snaps it to the nearest boundary round the circle",
          "[clamp_by_ranges]") {
    Real margin = 1e-6;

    SECTION("inside a range: unchanged") {
        Real ranges[] = {-2.0f, -1.0f, 2.0f, PI};
        REQUIRE(clamp_by_ranges(-1.5f, ranges, 2) == -1.5f);
        REQUIRE(clamp_by_ranges(2.0f, ranges, 2) == 2.0f);
        REQUIRE(clamp_by_ranges(PI, ranges, 2) == PI);
        Real full[] = {-PI, PI};
        REQUIRE(clamp_by_ranges(-PI, full, 1) == -PI);
        REQUIRE(clamp_by_ranges(0.7f, full, 1) == 0.7f);
    }

    SECTION("a step 1 ulp past +pi on an unpaired [lo, pi] range wraps to ~-pi and comes back to pi") {
        Real ranges[] = {2.0f, PI};
        const Real wrapped = normalize_angle(std::nextafter(PI, Real(10)));
        REQUIRE(wrapped < 0);
        REQUIRE(clamp_by_ranges(wrapped, ranges, 1) == PI);
        // the same with a range below it that is nearer by linear distance
        Real ranges_two[] = {-2.0f, -1.0f, 2.0f, PI};
        REQUIRE(clamp_by_ranges(wrapped, ranges_two, 2) == PI);
        // and the mirror case at -pi
        Real ranges_lo[] = {-PI, -2.0f};
        REQUIRE(clamp_by_ranges(PI - 1e-6f, ranges_lo, 1) == -PI);
    }

    SECTION("outside every range: nearest boundary across the seam") {
        Real ranges[] = {-1.0f, 2.5f};
        // -3.0 is 2 pi - 5.5 = 0.783 from 2.5 round the seam, 2.0 from -1.0 directly
        REQUIRE_THAT(clamp_by_ranges(-3.0f, ranges, 1), Catch::Matchers::WithinAbs(2.5f, margin));
        // 3.0 is 0.5 from 2.5 directly
        REQUIRE_THAT(clamp_by_ranges(3.0f, ranges, 1), Catch::Matchers::WithinAbs(2.5f, margin));
        // -1.5 is 0.5 from -1.0
        REQUIRE_THAT(clamp_by_ranges(-1.5f, ranges, 1), Catch::Matchers::WithinAbs(-1.0f, margin));
        Real ranges_two[] = {-2.8f, -2.0f, 0.0f, 1.0f};
        // 3.0 is 2 pi - 5.8 = 0.483 from -2.8 round the seam, 2.0 from 1.0 directly
        REQUIRE_THAT(clamp_by_ranges(3.0f, ranges_two, 2), Catch::Matchers::WithinAbs(-2.8f, margin));
    }
    //![clamp_by_ranges]
}


TEST_CASE("torsion_room is the room to the edge of the allowed arc in a direction", "[torsion_room]") {
    Real margin = 1e-4;

    SECTION("split arc [2, pi] + [-pi, -2.5] across the seam") {
        Real ranges[] = {2.0f, PI, -PI, -2.5f};
        // from the [2, pi] part across +-pi into [-pi, -2.5]
        REQUIRE_THAT(torsion_room(3.0f, ranges, 2, 1), Catch::Matchers::WithinAbs(2 * PI - 5.5f, margin));
        REQUIRE_THAT(torsion_room(3.0f, ranges, 2, -1), Catch::Matchers::WithinAbs(1.0f, margin));
        // from the [-pi, -2.5] part across -+pi into [2, pi]
        REQUIRE_THAT(torsion_room(-3.0f, ranges, 2, -1), Catch::Matchers::WithinAbs(2 * PI - 5.0f, margin));
        REQUIRE_THAT(torsion_room(-3.0f, ranges, 2, 1), Catch::Matchers::WithinAbs(0.5f, margin));
        // the seam itself is inside the arc
        REQUIRE_THAT(torsion_room(PI, ranges, 2, 1), Catch::Matchers::WithinAbs(PI - 2.5f, margin));
        REQUIRE_THAT(torsion_room(-PI, ranges, 2, -1), Catch::Matchers::WithinAbs(PI - 2.0f, margin));
    }

    SECTION("theta exactly on a boundary") {
        Real ranges[] = {2.0f, PI, -PI, -2.5f};
        REQUIRE_THAT(torsion_room(2.0f, ranges, 2, -1), Catch::Matchers::WithinAbs(0.0f, margin));
        REQUIRE_THAT(torsion_room(2.0f, ranges, 2, 1), Catch::Matchers::WithinAbs(2 * PI - 4.5f, margin));
        REQUIRE_THAT(torsion_room(-2.5f, ranges, 2, 1), Catch::Matchers::WithinAbs(0.0f, margin));
        // -pi is a real boundary when no range ends at +pi
        Real ranges_lo[] = {-PI, -2.0f};
        REQUIRE_THAT(torsion_room(-PI, ranges_lo, 1, -1), Catch::Matchers::WithinAbs(0.0f, margin));
        REQUIRE_THAT(torsion_room(-PI, ranges_lo, 1, 1), Catch::Matchers::WithinAbs(PI - 2.0f, margin));
    }

    SECTION("theta outside every range") {
        Real ranges[] = {2.0f, PI, -PI, -2.5f};
        REQUIRE(torsion_room(0.0f, ranges, 2, 1) < 0);
        REQUIRE(torsion_room(0.0f, ranges, 2, -1) < 0);
        REQUIRE(torsion_room(1.99f, ranges, 2, 1) == TOR_ROOM_OUTSIDE);
    }

    SECTION("full circle") {
        Real ranges[] = {-PI, PI};
        REQUIRE(torsion_room(0.3f, ranges, 1, 1) == TOR_ROOM_FREE);
        REQUIRE(torsion_room(-PI, ranges, 1, -1) == TOR_ROOM_FREE);
        // two ranges that touch at 0 and join at the seam
        Real ranges_joined[] = {-PI, 0.0f, 0.0f, PI};
        REQUIRE(torsion_room(0.5f, ranges_joined, 2, 1) == TOR_ROOM_FREE);
        REQUIRE(torsion_room(0.5f, ranges_joined, 2, -1) == TOR_ROOM_FREE);
    }

    SECTION("single interior range, and touching ranges joined") {
        Real ranges[] = {-1.0f, 0.5f};
        REQUIRE_THAT(torsion_room(0.0f, ranges, 1, 1), Catch::Matchers::WithinAbs(0.5f, margin));
        REQUIRE_THAT(torsion_room(0.0f, ranges, 1, -1), Catch::Matchers::WithinAbs(1.0f, margin));
        REQUIRE_THAT(torsion_room(0.5f, ranges, 1, 1), Catch::Matchers::WithinAbs(0.0f, margin));
        Real ranges_touch[] = {-1.0f, 0.0f, 0.0f, 1.0f};
        REQUIRE_THAT(torsion_room(-0.5f, ranges_touch, 2, 1), Catch::Matchers::WithinAbs(1.5f, margin));
        REQUIRE_THAT(torsion_room(0.5f, ranges_touch, 2, -1), Catch::Matchers::WithinAbs(1.5f, margin));
    }
    //![torsion_room]
}


TEST_CASE("get_real_within_by_int", "[get_real_within_by_int]") {
    // default n = 1001
    REQUIRE_THAT(get_real_within_by_int(0, 0, 10), Catch::Matchers::WithinAbs(0, 1e-4));
    REQUIRE_THAT(get_real_within_by_int(11, 0, 10, 1001), Catch::Matchers::WithinAbs(11.0/1001 * 10, 1e-4));
    REQUIRE_THAT(get_real_within_by_int(1002, 0, 10, 1001), Catch::Matchers::WithinAbs(1.0/1001 * 10, 1e-4));
    // x
    REQUIRE_THAT(get_real_within_by_int(633093634, -30.5, -5.5), Catch::Matchers::WithinAbs(-23.2419, 1e-4));
    REQUIRE_THAT(get_real_within_by_int(633, -PI, PI), Catch::Matchers::WithinAbs(-0.5067, 1e-4));

    //![get_real_within_by_int]
}


TEST_CASE("compute gaussian", "[gaussian]") {
    Real e, f;
    e = gaussian(0, 0, 3, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(1, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0, 1e-4));

    e = gaussian(1, 1, 3, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(1, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0, 1e-4));

    e = gaussian(0, 1, 4, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(0.939413071f, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0.117426634f, 1e-4));
}


TEST_CASE("computes the cross product of two 3D vectors", "[cross_product]") {
    Real vec1[] = {1.0f, 2.0f, 3.0f};
    Real vec2[] = {4.0f, 5.0f, 6.0f};
    Real result[3];
    cross_product(vec1, vec2, result);
    REQUIRE_THAT(result[0], Catch::Matchers::WithinAbs(-3.0f, 1e-4));
    REQUIRE_THAT(result[1], Catch::Matchers::WithinAbs(6.0f, 1e-4));
    REQUIRE_THAT(result[2], Catch::Matchers::WithinAbs(-3.0f, 1e-4));


    double vec3[] = {1.0, 0.0, 0.0};
    double vec4[] = {0.0, 1.0, 0.0};
    double result2[3];
    cross_product(vec3, vec4, result2);
    REQUIRE_THAT(result2[0], Catch::Matchers::WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(result2[1], Catch::Matchers::WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(result2[2], Catch::Matchers::WithinAbs(1.0, 1e-4));

    int vec5[] = {1, 0, 0};
    int vec6[] = {0, 1, 0};
    int result3[3];
    cross_product(vec5, vec6, result3);
    REQUIRE(result3[0] == 0);
    REQUIRE(result3[1] == 0);
    REQUIRE(result3[2] == 1);   
    //![cross_product]
}


TEST_CASE("computes the outer product of two 3D vectors", "[outer_product]") {
    Real vec1[] = {1.0f, 2.0f, 3.0f};
    Real vec2[] = {4.0f, 5.0f, 6.0f};
    Real expected[9] = {4, 5, 6, 8, 10, 12, 12, 15, 18};
    Real result[9];
    outer_product(vec1, vec2, result);
    for (int i = 0; i < 9; ++i){
        REQUIRE_THAT(result[i], Catch::Matchers::WithinAbs(expected[i], 1e-4));
    }
    //![outer_product]
}


TEST_CASE("compute dot product", "[dot_product]") {
    Real vec1[] = {1.0f, 2.0f, 3.0f};
    Real vec2[] = {4.0f, 5.0f, 6.0f};
    REQUIRE_THAT(dot_product(vec1, vec2), Catch::Matchers::WithinAbs(32.0f, 1e-4));
    //![dot_product]
}

TEST_CASE("cal_norm computes the Euclidean norm of a 3D vector", "[cal_norm]") {
    Real margin = 1e-4;

    Real vec1[] = {3.0f, 4.0f, 0.0f};
    REQUIRE_THAT(cal_norm(vec1), Catch::Matchers::WithinAbs(5.0f, margin));

    Real vec2[] = {1.0f, 2.0f, 2.0f};
    REQUIRE_THAT(cal_norm(vec2), Catch::Matchers::WithinAbs(3.0f, margin));

    Real vec3[] = {0.0f, 0.0f, 0.0f};
    REQUIRE_THAT(cal_norm(vec3), Catch::Matchers::WithinAbs(0.0f, margin));
    //![cal_norm]
}

TEST_CASE("dist2 computes the squared Euclidean distance between two 3D points", "[dist2]") {
    Real margin = 1e-4;

    Real point1[] = {1.0f, 2.0f, 3.0f};
    Real point2[] = {4.0f, 6.0f, 8.0f};
    REQUIRE_THAT(dist_sq(point1, point2), Catch::Matchers::WithinAbs(50.0f, margin));

    Real point3[] = {0.0f, 0.0f, 0.0f};
    Real point4[] = {0.0f, 0.0f, 0.0f};
    REQUIRE_THAT(dist_sq(point3, point4), Catch::Matchers::WithinAbs(0.0f, margin));

    Real point5[] = {1.0f, 1.0f, 1.0f};
    Real point6[] = {2.0f, 2.0f, 2.0f};
    REQUIRE_THAT(dist_sq(point5, point6), Catch::Matchers::WithinAbs(3.0f, margin));
    //![dist2]
}



