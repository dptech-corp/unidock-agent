# Common sources (backend-agnostic)
set(common_sources
        cluster/cluster.h

        config/parse.cpp        config/parse.h

        constants/constants.h

        ext/bias/bias_term.h
        ext/bias/bias_reader.cpp ext/bias/bias_reader.h
        ext/bias/ring_term.h
        ext/bias/ring_reader.cpp ext/bias/ring_reader.h

        format/dms_reader.cpp   format/dms_reader.h
        format/sdf_writer.cpp   format/sdf_writer.h

        geometry/quaternion.h
        geometry/rotation.h

        model/model.h

        myutils/common.h
        myutils/errors.h        myutils/errors.cpp
        myutils/matrix.h
        myutils/myio.h          myutils/myio.cpp
        myutils/mymath.h

        optimize/bfgs.h
        optimize/optimize.h

        screening/screening.h   screening/screening.cpp

        search/mc.h
        search/bvh.cpp          search/bvh.h

        score/gaff2.cpp         score/gaff2.h
        score/score.h
        score/terms.h
        score/vina.h

        task/DockTask.cpp       task/DockTask.h
)

# Backend-specific sources
if(USE_CUDA)
    set(backend_sources
            cuda/bfgs.cuh
            cuda/bfgs_wrapper.cu
            cuda/cluster.cu
            cuda/common.cuh cuda/common.cu
            cuda/maps.cuh cuda/maps.cu
            cuda/mc.cu
            cuda/model.cu
            cuda/mymath_warp.cuh
            cuda/optimize.cu
            cuda/task.cu
            cuda/vina.cu
            # Host-side stages shared with the CPU backend (FF scoring / minimization)
            cpu/common_cpu.h        cpu/common_cpu.cpp
            cpu/score_cpu.h         cpu/score_cpu.cpp
            cpu/score_ff_cpu.h      cpu/score_ff_cpu.cpp
            cpu/bfgs_cpu.h          cpu/bfgs_cpu.cpp
            cpu/host_topo.h         cpu/host_topo.cpp
            cpu/task_ff_cpu.cpp
    )
else()
    set(backend_sources
            cpu/spatial_grid.h
            cpu/random_cpu.h
            cpu/common_cpu.h        cpu/common_cpu.cpp
            cpu/score_cpu.h         cpu/score_cpu.cpp
            cpu/score_ff_cpu.h      cpu/score_ff_cpu.cpp
            cpu/bfgs_cpu.h          cpu/bfgs_cpu.cpp
            cpu/mc_cpu.h            cpu/mc_cpu.cpp
            cpu/cluster_cpu.h       cpu/cluster_cpu.cpp
            cpu/optimize_cpu.h      cpu/optimize_cpu.cpp
            cpu/host_topo.h         cpu/host_topo.cpp
            cpu/task_cpu.cpp
            cpu/task_ff_cpu.cpp
    )
endif()

set(sources ${common_sources} ${backend_sources})

# compile lib for the bin to link to
add_library(${LIB_NAME} OBJECT ${sources})

# include src directory
if(USE_CUDA)
    target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/lib/sqlite3 . )
    target_link_libraries(${LIB_NAME} PUBLIC CUDA::cudart_static)
    # OpenMP for the host tail of a task only: task/DockTask.cpp (final scoring over ligands) and
    # format/sdf_writer.cpp (SDF formatting), both guarded by _OPENMP. The flag goes on these two
    # sources, not on the target through OpenMP::OpenMP_CXX: that target's INTERFACE_COMPILE_OPTIONS
    # would reach every CXX source and activate the unguarded pragma in cpu/task_ff_cpu.cpp while
    # get_num_threads()/get_thread_id() (USE_OPENMP=0) still report a single thread. find_package
    # runs in this directory scope so OpenMP_CXX_FOUND stays unset for bin/ and test/.
    if(USE_OPENMP)
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            message(STATUS "OpenMP found for the host tail: ${OpenMP_CXX_VERSION}")
            separate_arguments(host_tail_omp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
            set_source_files_properties(task/DockTask.cpp format/sdf_writer.cpp
                    PROPERTIES COMPILE_OPTIONS "${host_tail_omp_flags}")
            target_link_libraries(${LIB_NAME} PUBLIC ${OpenMP_CXX_LIBRARIES})
        else()
            message(WARNING "OpenMP not found; the host tail runs single-threaded")
        endif()
    endif()
else()
    target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/lib/sqlite3 . )
    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(${LIB_NAME} PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

# config/parse.cpp reads the YAML: every consumer of ud_lib (bin, api, test) links ${THIRD_NAME_YAML} itself
# (an OBJECT library's objects do not propagate through another OBJECT library)

set_target_properties(${LIB_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# install
install(TARGETS ${LIB_NAME}
        LIBRARY DESTINATION lib
)