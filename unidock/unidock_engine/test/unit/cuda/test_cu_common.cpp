//
// Created by Congcong Liu on 24-10-15.
//
// All functions in common.cuh require CUDA runtime or device-side execution:
//   - gen_rand_int_within: __device__ only, uses curandStatePhilox4_32_10_t
//   - gen_rand_real_within: __device__ only, uses curandStatePhilox4_32_10_t
//   - gen_4_rand_in_sphere: __device__ only, uses curand_uniform4
//   - gyration_radius: uses __constant__ g_atp (requires cudaMemcpyToSymbol)
//   - get_real_within_by_int: host-callable but already tested via myutils/test_math.cpp
//   - init_constants: requires cudaMemcpyToSymbol
//
// No host-only functions remain untested. CUDA kernel tests would require
// a GPU device and are out of scope for CPU-side unit tests.
//

#include <catch2/catch_amalgamated.hpp>
#include "cuda/common.cuh"


TEST_CASE("cuda common.cuh - no host-testable functions", "[test_cuda]") {
    // Placeholder: all functions in common.cuh require CUDA device.
    // get_real_within_by_int is the only host-callable function and is
    // already covered by myutils/test_math.cpp tests.
    SUCCEED("No host-side functions to test in common.cuh");
}
