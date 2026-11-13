message("   [Section] lib (3rd Party) ...")
add_definitions(-DCATCH_AMALGAMATED_CUSTOM_MAIN)

add_library(${THIRD_NAME_CATCH2} OBJECT catch2/catch_amalgamated.cpp)
#target_include_directories(${THIRD_NAME_CATCH2} PUBLIC ${CMAKE_SOURCE_DIR}/include/yaml-cpp yaml-cpp/)

file(GLOB_RECURSE YAML_SOURCES yaml-cpp/*)
add_library(${THIRD_NAME_YAML} OBJECT ${YAML_SOURCES})
# linked into ud_lib (config/parse.cpp) and so into the Python module: position independent
set_target_properties(${THIRD_NAME_YAML} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${THIRD_NAME_CATCH2} PUBLIC .)

# SQLite3 amalgamation for DMS file reading
set(THIRD_NAME_SQLITE3 sqlite3 PARENT_SCOPE)
add_library(sqlite3 OBJECT sqlite3/sqlite3.c)
target_include_directories(sqlite3 PUBLIC sqlite3/)
set_target_properties(sqlite3 PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 11)
# Suppress warnings from third-party C code
target_compile_options(sqlite3 PRIVATE -w)

message("   [Section Done] lib (3rd Party) !")


