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

add_executable(gt_omp
    ${EXEC_SOURCE_DIR}/start_global_tree.cpp
)

add_executable(gt_omp_debug
    ${EXEC_SOURCE_DIR}/start_global_tree.cpp
)

add_executable(gt_mpi
    ${EXEC_SOURCE_DIR}/start_global_tree_mpi.cpp
)

add_executable(ft_mpi
    ${EXEC_SOURCE_DIR}/start_fusion_tree_mpi.cpp
)

add_executable(ts_mpi
    ${EXEC_SOURCE_DIR}/start_test_selection_mpi.cpp
)

add_executable(ts
    ${EXEC_SOURCE_DIR}/start_test_selection.cpp
)

add_dependencies(gt_omp BMGT)
add_dependencies(gt_omp_debug BMGT)
add_dependencies(gt_mpi BMGT)
add_dependencies(ft_mpi BMGT)
add_dependencies(ts_mpi BMGT)
add_dependencies(ts BMGT)

target_link_libraries(gt_omp
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)

target_link_libraries(gt_omp_debug
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)

target_link_libraries(gt_mpi
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)

target_link_libraries(ft_mpi
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)

target_link_libraries(ts_mpi
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)

target_link_libraries(ts
    ${CMAKE_SOURCE_DIR}/${LIB_SOURCE_DIR}/libBMGT.a
)
