cmake_minimum_required(VERSION 3.10)
project(Bayesian-Multinomial-Group-Testing)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_COMPILER "mpicxx")  # Set the compiler to mpicxx
set(CMAK_C_COMPILER "mpicc")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -Wall")


set(EXEC_SOURCE_DIR "exec")
set(ANALYSIS_SOURCE_DIR "statistical_analysis")
set(LATTICE_SOURCE_DIR "product_lattice")
set(TREE_SOURCE_DIR "${ANALYSIS_SOURCE_DIR}/tree")
set(UTIL_SOURCE_DIR "${ANALYSIS_SOURCE_DIR}/util")
set(EXTERNAL_SOURCE_DIR "external")
set(LIB_SOURCE_DIR lib) 

include_directories(include)
include_directories(${EXTERNAL_SOURCE_DIR}/include)

# Recursively find all source files in the specified directories
file(GLOB_RECURSE EXEC_SOURCES "${EXEC_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE ANALYSIS_SOURCES "${ANALYSIS_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE TREE_SOURCES "${TREE_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE UTIL_SOURCES "${UTIL_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE LATTICE_SOURCES "${LATTICE_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE EXTERNAL_SOURCES "${EXTERNAL_SOURCE_DIR}/*.c*")

add_library(BMGT STATIC
    ${ANALYSIS_SOURCES}
    ${TREE_SOURCES}
    ${UTIL_SOURCES}
    ${LATTICE_SOURCES}
    ${EXTERNAL_SOURCES}
)

# Set the output directory for the library
set_target_properties(BMGT PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${LIB_SOURCE_DIR}
)

add_executable(gt_intra
    ${EXEC_SOURCE_DIR}/run_glob_tree_intra.cpp
)

add_executable(gt_intra_dbg
    ${EXEC_SOURCE_DIR}/run_glob_tree_intra.cpp
)

add_executable(gt
    ${EXEC_SOURCE_DIR}/run_glob_tree.cpp
)

add_executable(ft
    ${EXEC_SOURCE_DIR}/run_fusion_tree.cpp
)

add_executable(dt_intra
    ${EXEC_SOURCE_DIR}/run_dist_tree_intra.cpp
)

add_executable(dt
    ${EXEC_SOURCE_DIR}/run_dist_tree.cpp
)

add_executable(dt_gt
    ${EXEC_SOURCE_DIR}/run_dist_glob_tree.cpp
)

add_executable(dt_gt_trim
    ${EXEC_SOURCE_DIR}/run_dist_glob_tree_trim.cpp
)

add_executable(dt_gt_symm
    ${EXEC_SOURCE_DIR}/run_dist_glob_tree_symm.cpp
)

add_executable(gpt_trim
    ${EXEC_SOURCE_DIR}/run_global_partial_tree.cpp
)

add_executable(ts_intra
    ${EXEC_SOURCE_DIR}/run_test_selection_intra.cpp
)

add_executable(ts
    ${EXEC_SOURCE_DIR}/run_test_selection.cpp
)

add_dependencies(gt_intra BMGT)
add_dependencies(gt_intra_dbg BMGT)
add_dependencies(gt BMGT)
add_dependencies(ft BMGT)
add_dependencies(dt_intra BMGT)
add_dependencies(dt BMGT)
add_dependencies(dt_gt BMGT)
add_dependencies(dt_gt_trim BMGT)
add_dependencies(dt_gt_symm BMGT)
add_dependencies(gpt_trim BMGT)
add_dependencies(ts_intra BMGT)
add_dependencies(ts BMGT)

target_link_libraries(gt_intra
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)

target_link_libraries(gt_intra_dbg
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)

target_link_libraries(gt
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)

target_link_libraries(ft
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)

target_link_libraries(dt_intra
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)

target_link_libraries(dt
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)

target_link_libraries(dt_gt
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)

target_link_libraries(dt_gt_trim
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)

target_link_libraries(dt_gt_symm
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)

target_link_libraries(gpt_trim
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)

target_link_libraries(ts_intra
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)

target_link_libraries(ts
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)
