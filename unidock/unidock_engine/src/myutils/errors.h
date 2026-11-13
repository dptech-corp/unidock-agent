//
// Created by Congcong Liu on 24-9-19.
//
#ifndef ERRORS_H
#define ERRORS_H

#include <string>
#include "spdlog/spdlog.h"

#if USE_CUDA
#include <cuda_runtime.h>
void check_cuda(cudaError_t cuda_err, char const *const func, const char *const file, int const line);
#define checkCUDA(val) check_cuda((val), #val, __FILE__, __LINE__)

#define CUDA_ERROR(...) printf("[CUDA error][%s:%d][%s][Block:%d,Thread:%d] ", \
__FILE__, __LINE__, __func__, \
blockIdx.x, threadIdx.x); \
printf(__VA_ARGS__); \
printf("\n")

#else
// No-op macros for CPU builds
#define checkCUDA(val) ((void)0)
#define CUDA_ERROR(...) ((void)0)
#endif

void init_logger(const std::string& fp_log = "ud.log", int level=1);

#endif //ERRORS_H