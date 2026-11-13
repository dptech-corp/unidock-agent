message("   [Section] Building BIN ...")

add_executable(${BIN_NAME}
        unidock/main.cpp
        unidock/main.h
)
target_compile_definitions(${BIN_NAME} PRIVATE VERSION_NUMBER="${VERSION_NUMBER}")
target_link_libraries(${BIN_NAME} PRIVATE ${THIRD_NAME_YAML} ${THIRD_NAME_SQLITE3} ${LIB_NAME})
if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(${BIN_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(${BIN_NAME}
        PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)

install(
        TARGETS ${BIN_NAME}
        RUNTIME DESTINATION bin
)
message("   [Section Done]  BIN !")
