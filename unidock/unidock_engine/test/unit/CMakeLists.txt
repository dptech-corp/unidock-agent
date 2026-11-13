
# Use Catch2 for all tests (instead of gtest)
message("      [Sub-Section] Building unit tests ...")

# Common tests (backend-agnostic)
set(common_sources
        unit_test.cpp
        format/test_dms_reader.cpp
        format/test_dms_bias.cpp
        format/test_table_registry.cpp
        ext/test_bias_term.cpp
        ext/test_ring_bias.cpp
        config/test_parse.cpp
        format/test_sdf_writer.cpp
        geometry/test_quaternion.cpp
        geometry/test_rotation.cpp
        myutils/test_math.cpp
        myutils/test_matrix.cpp
        score/test_vina.cpp
        score/test_rescore_hbond.cpp
        score/test_score_decomp.cpp
        score/test_gaff2.cpp
        score/test_bias.cpp
        score/test_box_penalty.cpp
        score/test_place_h.cpp
)

# Backend-specific tests
if(USE_CUDA)
    set(backend_sources
            cuda/test_bfgs.cpp
            cuda/test_cu_common.cpp
            cuda/test_score_gpu_vs_cpu.cpp
            cuda/test_grid_maps.cpp
            ../../src/cpu/mc_cpu.cpp   # host MC routines used as reference by the GPU tests
            search/test_mc.cpp
    )
else()
    set(backend_sources
            cpu/test_cpu_score.cpp
            cpu/test_cpu_bfgs.cpp
            cpu/test_cpu_bias.cpp
            cpu/test_cpu_ring_bias.cpp
            cpu/test_score_ff_cpu.cpp
            cpu/test_score_ff_cpu_detailed.cpp
            cpu/test_cpu_cluster.cpp
            cpu/test_hbond_angle.cpp
            cpu/test_angular_hbond.cpp
            cpu/test_custom_atom_type.cpp
            cpu/test_custom_ff_param.cpp
            cpu/test_mc_cpu.cpp
            cpu/test_spatial_grid.cpp
            cpu/test_cal_e_f_gradient.cpp
            cpu/test_invariants.cpp
            cpu/test_ff_bond_angle.cpp
            cpu/test_ff_minimize_cartesian.cpp
            cpu/test_bfgs_ff_real.cpp
            cpu/test_bfgs_ff_mask.cpp
    )
endif()

set(sources ${common_sources} ${backend_sources})

add_executable(${TESTS_NAME} ${sources})
target_link_libraries(${TESTS_NAME} ${LIB_NAME} ${THIRD_NAME_CATCH2} ${THIRD_NAME_SQLITE3} ${THIRD_NAME_YAML})

# Provide absolute path to test data so tests work from any working directory
target_compile_definitions(${TESTS_NAME} PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)


