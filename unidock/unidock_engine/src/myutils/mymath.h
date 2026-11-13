//
// Created by Congcong Liu on 24-9-30.
//

#ifndef MATH_H
#define MATH_H

#include "myutils/common.h"
#include <cassert>


SCOPE_INLINE Real map_01_to_dot5(Real a){
    // todo: test
    assert(0 < a && a <= 1);
    return a - 0.5;
}

/**
 * @brief Normalize an angle to the range [0, 2pi]
 * @snippet test_math.cpp normalize_angle
 * @param x The angle, in radiant
 * @return The normalized angle
 */
SCOPE_INLINE Real normalize_angle_2pi(Real x){
    x = fmod(x, 2 * PI); // Normalize to the range [-2π, 2π)
    if (x < 0){
        x += 2 * PI; // If the result is negative, normalize to [0, 2π)
    }
    return x;
}

/**
 * @brief Normalize an angle to the range [-π, π)
 * @snippet test_math.cpp normalize_angle
 * @param x The angle, in radiant
 * @return The normalized angle
 */
SCOPE_INLINE Real normalize_angle(Real x){
    if (!(fabs(x) < 2 * PI)) x = fmod(x, 2 * PI); // Normalize to the range [-2π, 2π); fmod is exact, so |x| < 2π is returned unchanged
    if (x < -PI){
        x += 2 * PI; // purge to [0, π)
    }
    if (x > PI){
        x -= 2 * PI; // purge to [-π, 0)
    }
    return x;
}


/**
 * @brief Transform an angle in [-180, 180] to radian in [-π, π]
 * @snippet test_math.cpp ang_to_rad
 * @param angle The angle in degree
 * @return The angle in radian
 */
SCOPE_INLINE Real ang_to_rad(Real angle){
    assert(angle >= -180 && angle <= 180);
    return angle * DEG2RAD;
}

/**
 * @brief Transform an radian angle in [-π, π] to degree in [-180, 180]
 * @snippet test_math.cpp rad_to_ang
 * @param rad The angle in radian
 * @return The angle in degree
 */
SCOPE_INLINE Real rad_to_ang(Real rad){
    assert(rad >= -PI && rad <= PI);
    return rad * RAD2DEG;
}


//------------------------------------------------------
//------------------- Range Operations -----------------
//------------------------------------------------------

/**
 * @brief Check if a value is in a given range
 * @snippet test/unit/myutils/test_math.cpp isInRanges
 * @param value The value to check
 * @param ranges The range array, each range is represented by two values, indicating the start and end of the range
 * @param num_range The number of ranges
 * @return True if the value is in any range, false otherwise
 */
SCOPE_INLINE bool isInRanges(Real value, int* ranges, int num_range){
    for (int i = 0; i < num_range; i++){
        if (value >= ranges[i * 2] && value <= ranges[i * 2 + 1]){
            return true;
        }
    }
    return false;
}


/**
 * @brief Randomly select a value from a given range fixme: the boundary values can be missed
 * @snippet test/unit/myutils/test_math.cpp get_radian_in_ranges
 * @param ranges The range array, each range is represented by two values, indicating the start and end of the range
 * @param num_range The number of ranges
 * @param rand_2 2 random numbers, both in the range (0, 1]
 * @return The randomly selected value in the given range
 */
SCOPE_INLINE Real get_radian_in_ranges(Real* ranges, int num_range, Real* rand_2){
    int i_range = ceil(rand_2[0] * num_range - 1); //(0, 1] mapped to [0, 1, ..., num_range - 1]

    // choose a value:
    Real from = ranges[i_range * 2];
    Real to = ranges[i_range * 2 + 1];
    return from + (to - from) * rand_2[1]; // (0, 1] mapped to [from, ..., to]
}

SCOPE_INLINE Real clamp_to_center(Real a, Real hi, Real lo){
    if ((a > hi) or (a < lo)){
        return (hi + lo) / 2.;
    }

    return a;
}


SCOPE_INLINE Real clamp_by_range(Real ang, Real hi, Real lo){
    if (ang > hi){
        return hi;
    }
    if (ang < lo){
        return lo;
    }
    return ang;
}


constexpr Real TOR_TOL = 1e-5; // radian; tolerance for "theta inside a range" and for two ranges joining
constexpr Real TOR_ROOM_FREE = 1e6; // torsion_room: the allowed set is the full circle
constexpr Real TOR_ROOM_OUTSIDE = -1; // torsion_room: theta is outside every range

/**
 * @brief Smallest angle between a and b around the circle
 * @snippet test/unit/myutils/test_math.cpp circular_distance
 * @return The distance in radian, in [0, π]
 */
SCOPE_INLINE Real circular_distance(Real a, Real b){
    return fabs(normalize_angle(a - b));
}


/**
 * @brief Clamp an angle to a set of ranges
 * @snippet test/unit/myutils/test_math.cpp clamp_by_ranges
 * @param ang The angle in radian, in [-π, π]
 * @param ranges nrange [lo, hi] pairs in radian
 * @tparam CIRCULAR false: nearest by |ang - boundary| instead, the rule of the kernels without the torsion library
 * (bfgs.cuh TORLIB = false), which keep the code they had before the wrap-aware rule
 * @return ang if it is in a range, else the range boundary nearest to it round the circle
 */
template<bool CIRCULAR = true>
SCOPE_INLINE Real clamp_by_ranges(Real ang, Real* ranges, int nrange){
    // return a1 if a1 in any range
    for (int i = 0; i < nrange; i++){
        if (ang >= ranges[2 * i] && ang <= ranges[2 * i + 1]){
            return ang;
        }
    }

    // return the closest boundary value if ang is not in any range
    Real closest_ang = ang;
    Real min_distance = 100.;
    for (int i = 0; i < nrange; i++){
        Real distance_to_min, distance_to_max;
        if constexpr (CIRCULAR){
            distance_to_min = circular_distance(ang, ranges[2 * i]);
            distance_to_max = circular_distance(ang, ranges[2 * i + 1]);
        }
        else{
            distance_to_min = fabs(ang - ranges[2 * i]);
            distance_to_max = fabs(ang - ranges[2 * i + 1]);
        }

        if (distance_to_min < min_distance){
            min_distance = distance_to_min;
            closest_ang = ranges[2 * i];
        }
        if (distance_to_max < min_distance){
            min_distance = distance_to_max;
            closest_ang = ranges[2 * i + 1];
        }
    }
    return closest_ang;
}

/**
 * @brief How far theta can turn in direction dir before leaving the allowed set: the union of the ranges on the
 * circle, where ranges that touch or overlap form one arc and a range ending at +π continues into one starting at -π
 * @snippet test/unit/myutils/test_math.cpp torsion_room
 * @param theta The angle in radian, in [-π, π]
 * @param ranges nrange [lo, hi] pairs in radian, lo < hi
 * @param dir +1 (increasing angle) or -1
 * @return The room in radian; TOR_ROOM_FREE if the union is the full circle; TOR_ROOM_OUTSIDE if theta is in no range
 */
SCOPE_INLINE Real torsion_room(Real theta, const Real* ranges, int nrange, int dir){
    if (nrange == 1 && ranges[0] <= -PI + TOR_TOL && ranges[1] >= PI - TOR_TOL){
        return TOR_ROOM_FREE;
    }
    // Walk towards increasing angle; for dir < 0 on the mirrored circle (x -> -x), where [lo, hi] becomes [-hi, -lo].
    const bool fwd = dir > 0;
    const Real t = fwd ? theta : -theta;
    Real end = -TOR_ROOM_FREE;
    for (int i = 0; i < nrange; i++){
        const Real lo = fwd ? ranges[2 * i] : -ranges[2 * i + 1];
        const Real hi = fwd ? ranges[2 * i + 1] : -ranges[2 * i];
        if (t >= lo - TOR_TOL && t <= hi + TOR_TOL && hi > end){
            end = hi;
        }
    }
    if (end == -TOR_ROOM_FREE){
        return TOR_ROOM_OUTSIDE;
    }
    Real room = end - t;
    // Each pass joins one more range or crosses the seam, so one lap round the circle takes at most nrange + 1 passes.
    for (int pass = 0; pass < 2 * nrange + 1; pass++){
        const bool seam = end >= PI - TOR_TOL;
        const Real from = seam ? -PI : end;
        Real next = from;
        for (int i = 0; i < nrange; i++){
            const Real lo = fwd ? ranges[2 * i] : -ranges[2 * i + 1];
            const Real hi = fwd ? ranges[2 * i + 1] : -ranges[2 * i];
            if (lo <= from + TOR_TOL && hi > next){
                next = hi;
            }
        }
        if (next <= from + TOR_TOL){
            break;
        }
        room += (seam ? PI - end : 0) + (next - from);
        end = next;
        if (room >= 2 * PI - TOR_TOL){
            return TOR_ROOM_FREE;
        }
    }
    return room > 0 ? room : 0;
}

/**
 * @brief Generate a real number in [min, max]
 *
 * @param c A random integer within Integer range
 * @param min Minimum value (Real)
 * @param max Maximum value (Real)
 * @param n Number of divisions. The larger n is, the more accurate the result is.
 * @return A real number in [min, max]
 */
SCOPE_INLINE Real get_real_within_by_int(uint c, Real min, Real max, int n = 31){
    //todo: automatically adjust n
    Real tmp = c % n;
    Real v = tmp * (max - min) / n + min;
    if (v < min){
        return min;
    }
    if (v > max){
        return max;
    }
    return v;
}


//------------------------------------------------------
//------------------- Math functions -------------------
//------------------------------------------------------

/**
 * @brief Compute the value and derivative of a Gaussian function
 * @snippet test/unit/myutils/test_math.cpp gaussian
 * @param d Distance
 * @param offset Offset of the Gaussian function
 * @param width Width of the Gaussian function
 * @param out_f Output derivative
 * @return Value of the Gaussian function
 */
SCOPE_INLINE Real gaussian(Real d, Real offset, Real width, Real* out_f){
    Real e = exp(-square((d - offset) / width));
    *out_f = e * 2 * (offset - d) / width / width;
    return e;
}


//------------------------------------------------------
//------------------- Vector Operations ----------------
//------------------------------------------------------

/**
 * @brief Compute the cross product of two 3x1 vectors
 * @snippet test/unit/myutils/test_math.cpp cross_product
 * @param a The first vector
 * @param b The second vector
 * @param out_res The result vector
 */
template <typename T1, typename T2, typename T3>
SCOPE_INLINE void cross_product(const T1* a, const T2* b, T3* out_res){
    out_res[0] = a[1] * b[2] - a[2] * b[1];
    out_res[1] = a[2] * b[0] - a[0] * b[2];
    out_res[2] = a[0] * b[1] - a[1] * b[0];
}

/**
 * @brief Compute the outer product of two 3x1 vectors
 * @snippet test/unit/myutils/test_math.cpp outer_product
 * @param a The first vector
 * @param b The second vector
 * @param out_res The result matrix (3x3)
 */
template <typename T1, typename T2, typename T3>
SCOPE_INLINE void outer_product(const T1* a, const T2* b, T3* out_res){
    out_res[0] = a[0] * b[0];
    out_res[1] = a[0] * b[1];
    out_res[2] = a[0] * b[2];
    out_res[3] = a[1] * b[0];
    out_res[4] = a[1] * b[1];
    out_res[5] = a[1] * b[2];
    out_res[6] = a[2] * b[0];
    out_res[7] = a[2] * b[1];
    out_res[8] = a[2] * b[2];
}


/**
 * @brief Compute the dot product of two 3D vectors
 * @snippet test/unit/myutils/test_math.cpp dot_product
 * @param a The first vector
 * @param b The second vector
 * @return The dot product of the two vectors
 */
template <typename T1, typename T2>
SCOPE_INLINE Real dot_product(const T1* a, const T2* b){
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}


/**
 * @brief Calculate the norm of a vector
 * @snippet test_math.cpp cal_norm
 * @param a The vector
 * @return The norm of the vector
 */
SCOPE_INLINE Real cal_norm(const Real* a){
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

/**
 * @brief Normalize a rotation vector
 * @snippet test_math.cpp norm_rotvec
 * @param v The vector
 */
SCOPE_INLINE void norm_rotvec(Real* out_v){
    Real l = cal_norm(out_v);
    Real angle  = normalize_angle(l);
    // the real rotation angle
    if (l > EPSILON){
        out_v[0] = out_v[0] / l * angle;
        out_v[1] = out_v[1] / l * angle;
        out_v[2] = out_v[2] / l * angle;

    } else {
        out_v[0] = 0;
        out_v[1] = 0;
        out_v[2] = 0;
    }
}


/**
 * @brief Calculate the squared distance between two vectors
 * @snippet test_math.cpp dist_norm2
 * @param a The first vector
 * @param b The second vector
 * @return The squared distance between the two vectors
 */
SCOPE_INLINE Real dist_sq(const Real* a, const Real* b){
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])
        + (a[2] - b[2]) * (a[2] - b[2]);
}

SCOPE_INLINE Real dist(const Real* a, const Real* b){
    return sqrt(dist_sq(a, b));
}


#endif //MATH_H
