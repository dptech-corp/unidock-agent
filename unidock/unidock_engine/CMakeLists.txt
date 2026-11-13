cmake_minimum_required(VERSION 3.18)

# ------------------------------ Options for Users
option(BUILD_BIN "build binaries" ON)
option(BUILD_API "build api" OFF)
option(BUILD_TEST "build unit tests" ON)
option(BUILD_DOC "build documents" OFF)

# Precision
option(USE_DOUBLE "If TRUE, use double-precision instead of single-precision for the whole project." FALSE)

# Backend selection: CUDA or CPU. Defaults to the CPU backend; pass -DUSE_CUDA=ON
# to build the CUDA backend (src/cuda).
option(USE_CUDA "Build with CUDA GPU support" OFF)
option(USE_OPENMP "Build with OpenMP parallelization (for CPU backend)" ON)

# ------------------------------ Options for Developers
set(VERSION_NUMBER 0.4.6)

# For Building Stage
link_directories(${CMAKE_SOURCE_DIR}/lib)
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

set(PRJ_NAME ud2)
set(LIB_NAME ud_lib)
set(THIRD_NAME_CATCH2 catch2)
set(THIRD_NAME_YAML yaml-cpp)
set(THIRD_NAME_SQLITE3 sqlite3)
set(BIN_NAME ud2)
set(TESTS_NAME unit_test)

# install
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) # default path is
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Installation root" FORCE)
endif()

# ------------------------------ Fixed Settings
# use C++ 17 standard for std::filesystem
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------ Backend Configuration
if(USE_CUDA)
    message(STATUS "Building with CUDA backend")
    
    # https://en.wikipedia.org/wiki/CUDA#GPUs_supported
    set(CMAKE_CUDA_ARCHITECTURES # At least 70!
            70 # V100
            # 75 # RTX 20, T4
            80 # A100, A800
            86 # RTX 30
            89 # RTX 40, L40
            90 # H100
    )
    message("CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    
    project(${PRJ_NAME} LANGUAGES CUDA CXX C)
    
    find_package(CUDAToolkit REQUIRED)
    
    add_compile_definitions(USE_CUDA=1)
    add_compile_definitions(FuncScope=__host__\ __device__)
    # Fast intrinsics for exp/sqrt/div in the scoring kernels: measured deviation from the
    # CPU scorer stays at float rounding level (~1e-5 kcal/mol) while eval cost drops ~18 %.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
    add_compile_definitions(USE_OPENMP=0)  # the CPU-backend stages keep their single-threaded paths in the CUDA build;
                                           # OpenMP for the host tail is set per source in src/CMakeLists.txt
else()
    message(STATUS "Building with CPU backend (no CUDA)")
    
    project(${PRJ_NAME} LANGUAGES CXX C)
    
    add_compile_definitions(USE_CUDA=0)
    add_compile_definitions(FuncScope=)
    
    # OpenMP for CPU parallelization
    if(USE_OPENMP)
        # Apple Clang needs Homebrew libomp hints
        if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
            execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(HOMEBREW_LIBOMP_PREFIX)
                set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
                set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
                set(OpenMP_C_LIB_NAMES "omp")
                set(OpenMP_CXX_LIB_NAMES "omp")
                set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
            endif()
        endif()
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
            add_compile_definitions(USE_OPENMP=1)
        else()
            message(WARNING "OpenMP not found, falling back to single-threaded CPU")
            set(USE_OPENMP OFF)
            add_compile_definitions(USE_OPENMP=0)
        endif()
    else()
        add_compile_definitions(USE_OPENMP=0)
    endif()
endif()

set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)

# Default is a release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (USE_DOUBLE)
    add_compile_definitions(Real=double)
    message("  Precision: double")
else ()
    add_compile_definitions(Real=float)
    message("  Precision: float")
endif ()

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
endif ()

# ------------------------------ Start building
add_subdirectory("lib")

add_subdirectory("src")

# python binding
if (BUILD_API)
    add_subdirectory("api")
endif ()

if (BUILD_BIN)
    add_subdirectory("bin")
endif ()

if (BUILD_TEST) # Only support GLIB >= 2.35 (Ubuntu 22.04) todo: GLIB 2.31
    add_subdirectory("include")
    add_subdirectory("test")
endif ()

if (BUILD_DOC)
    add_subdirectory("doc")
endif ()



