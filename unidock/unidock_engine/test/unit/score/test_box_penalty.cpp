//
// Unit tests for Box::is_inside from model.h
// Note: cal_box_penalty_atom (score.h) cannot be tested independently due to
// duplicate symbol issues caused by non-inline globals in that header.
// The CPU version cal_box_penalty_clamp_coord_cpu is tested in test_cpu_score.cpp.
//

#include <catch2/catch_amalgamated.hpp>
#include "model/model.h"


TEST_CASE("Box::is_inside center of box", "[score][box]") {
    Box box;
    box.x_lo = -10; box.x_hi = 10;
    box.y_lo = -10; box.y_hi = 10;
    box.z_lo = -10; box.z_hi = 10;

    REQUIRE(box.is_inside(0, 0, 0));
    REQUIRE(box.is_inside(5, -5, 3));
}


TEST_CASE("Box::is_inside on boundary", "[score][box]") {
    Box box;
    box.x_lo = -10; box.x_hi = 10;
    box.y_lo = -10; box.y_hi = 10;
    box.z_lo = -10; box.z_hi = 10;

    // Boundaries are inclusive
    REQUIRE(box.is_inside(-10, -10, -10));
    REQUIRE(box.is_inside(10, 10, 10));
    REQUIRE(box.is_inside(-10, 0, 10));
}


TEST_CASE("Box::is_inside outside", "[score][box]") {
    Box box;
    box.x_lo = -10; box.x_hi = 10;
    box.y_lo = -10; box.y_hi = 10;
    box.z_lo = -10; box.z_hi = 10;

    REQUIRE_FALSE(box.is_inside(15, 0, 0));
    REQUIRE_FALSE(box.is_inside(0, -11, 0));
    REQUIRE_FALSE(box.is_inside(0, 0, 10.1));
    REQUIRE_FALSE(box.is_inside(100, 100, 100));
}


TEST_CASE("Box::is_inside asymmetric box", "[score][box]") {
    Box box;
    box.x_lo = 0; box.x_hi = 5;
    box.y_lo = -2; box.y_hi = 3;
    box.z_lo = 10; box.z_hi = 20;

    REQUIRE(box.is_inside(2, 0, 15));
    REQUIRE(box.is_inside(0, -2, 10));
    REQUIRE(box.is_inside(5, 3, 20));

    REQUIRE_FALSE(box.is_inside(-1, 0, 15));
    REQUIRE_FALSE(box.is_inside(2, 0, 5));
    REQUIRE_FALSE(box.is_inside(2, 4, 15));
}
