cmake_minimum_required(VERSION 3.10)
project(BMGT VERSION 1.0 DESCRIPTION "Scalable Bayesian Multinomial Group Testing Framework")
set(CMAKE_VERBOSE_MAKEFILE ON)

set(BMGT_MAJOR 0)
set(BMGT_MINOR 0)
set(BMGT_PATCH 1)
set(BMGT_SUFFIX "alpha")
set(BMGT_VERSION_CODE ${BMGT_MAJOR}${BMGT_MINOR}${BMGT_PATCH})

configure_file(${CMAKE_SOURCE_DIR}/include/bmgt.h.in ${CMAKE_BINARY_DIR}/include/bmgt.h @ONLY)

set(CMAKE_CXX_COMPILER "mpicxx")
set(CMAKE_C_COMPILER "mpicc")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -fopenmp -fvisibility=hidden")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -Wall -fvisibility=hidden")

set(EXAMPLE_SOURCE_DIR "${PROJECT_SOURCE_DIR}/examples")
set(EXEC_SOURCE_DIR "${PROJECT_SOURCE_DIR}/exec")
set(ANALYSIS_SOURCE_DIR "${PROJECT_SOURCE_DIR}/statistical_analysis")
set(LATTICE_SOURCE_DIR "${PROJECT_SOURCE_DIR}/product_lattice")
set(TREE_SOURCE_DIR "${ANALYSIS_SOURCE_DIR}/tree")
set(UTIL_SOURCE_DIR "${ANALYSIS_SOURCE_DIR}/util")
set(3RD_PARTY_SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rd-party")
set(CMAKE_BINARY_DIR "${PROJECT_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${3RD_PARTY_SOURCE_DIR}/include")

# Recursively find all source files in the specified directories
file(GLOB_RECURSE EXAMPLE_SOURCES "${EXAMPLE_SOURCE_DIR}/*.c*")
file(GLOB_RECURSE EXEC_SOURCES "${EXEC_SOURCE_DIR}/*.c*")
file(GLOB_RECURSE ANALYSIS_SOURCES "${ANALYSIS_SOURCE_DIR}/*.c*")
file(GLOB_RECURSE TREE_SOURCES "${TREE_SOURCE_DIR}/*.c*")
file(GLOB_RECURSE UTIL_SOURCES "${UTIL_SOURCE_DIR}/*.c*")
file(GLOB_RECURSE LATTICE_SOURCES "${LATTICE_SOURCE_DIR}/*.c*")
file(GLOB_RECURSE 3RD_PARTY_SOURCES "${3RD_PARTY_SOURCE_DIR}/*.c*")

# Define a function to create both shared and static library targets
function(add_bmgt_library TARGET_NAME)
    # Shared library
    add_library(${TARGET_NAME}_shared SHARED
        ${ANALYSIS_SOURCES}
        ${TREE_SOURCES}
        ${UTIL_SOURCES}
        ${LATTICE_SOURCES}
        ${3RD_PARTY_SOURCES}
        ${EXEC_SOURCES}
    )

    # Static library (same target name, but different properties)
    add_library(${TARGET_NAME}_static STATIC
        ${ANALYSIS_SOURCES}
        ${TREE_SOURCES}
        ${UTIL_SOURCES}
        ${LATTICE_SOURCES}
        ${3RD_PARTY_SOURCES}
        ${EXEC_SOURCES}
    )

    # OPTIONAL: Set output name to differentiate the static library file
    set_target_properties(${TARGET_NAME}_shared PROPERTIES OUTPUT_NAME ${TARGET_NAME})
    set_target_properties(${TARGET_NAME}_static PROPERTIES OUTPUT_NAME ${TARGET_NAME})
    set_target_properties(${TARGET_NAME}_shared PROPERTIES VERSION ${PROJECT_VERSION})
    set_target_properties(${TARGET_NAME}_shared PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR})
    set_target_properties(${TARGET_NAME}_static PROPERTIES VERSION ${PROJECT_VERSION})
    set_target_properties(${TARGET_NAME}_static PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/bmgt.h)
    set_target_properties(${TARGET_NAME}_shared PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/bmgt.h)
    install(TARGETS ${TARGET_NAME}_shared
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(TARGETS ${TARGET_NAME}_static
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${CMAKE_SOURCE_DIR}/bmgt.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

    # Apply compile definitions if provided
    if(ARGN)
        target_compile_definitions(${TARGET_NAME}_shared PRIVATE ${ARGN})
        target_compile_definitions(${TARGET_NAME}_static PRIVATE ${ARGN})
    endif()
endfunction()

include(GNUInstallDirs)

# Create library targets
add_bmgt_library(bmgt_baseline BBPA_NAIVE)
add_bmgt_library(bmgt_op1 BBPA_OP1)
add_bmgt_library(bmgt_op2 BBPA_OP2)
add_bmgt_library(bmgt_op3 ENABLE_SIMD USE_INTEL_INTRINSICS)
add_bmgt_library(bmgt_op4 ENABLE_SIMD USE_INTEL_INTRINSICS ENABLE_OMP)
add_bmgt_library(bmgt ENABLE_SIMD USE_INTEL_INTRINSICS)

function(create_executable TARGET_NAME SOURCE_FILE LIBRARY_NAME SHOULD_INSTALL)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    add_dependencies(${TARGET_NAME} ${LIBRARY_NAME})
    target_link_libraries(${TARGET_NAME} ${LIBRARY_NAME})
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_BINARY_DIR}/include)

    # Check if the executable should be installed
    if(SHOULD_INSTALL)
        install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endfunction()


create_executable(global_tree_intra ${EXAMPLE_SOURCE_DIR}/run_glob_tree_intra.cpp bmgt_shared TRUE)
create_executable(global_tree_intra_dbg ${EXAMPLE_SOURCE_DIR}/run_glob_tree_intra.cpp bmgt_shared TRUE)
create_executable(global_tree ${EXAMPLE_SOURCE_DIR}/run_glob_tree.cpp bmgt_shared TRUE)
create_executable(fusion_tree ${EXAMPLE_SOURCE_DIR}/run_fusion_tree.cpp bmgt_shared TRUE)
create_executable(distributed_tree_intra ${EXAMPLE_SOURCE_DIR}/run_dist_tree_intra.cpp bmgt_shared TRUE)
create_executable(distributed_tree ${EXAMPLE_SOURCE_DIR}/run_dist_tree.cpp bmgt_shared TRUE)
create_executable(distributed_global_hybrid_tree ${EXAMPLE_SOURCE_DIR}/run_dist_glob_tree.cpp bmgt_shared TRUE)
create_executable(distributed_global_hybrid_tree_with_trimming ${EXAMPLE_SOURCE_DIR}/run_dist_glob_tree_trim.cpp bmgt_shared TRUE)
create_executable(distributed_global_hybrid_tree_with_symmetric ${EXAMPLE_SOURCE_DIR}/run_dist_glob_tree_symm.cpp bmgt_shared TRUE)
create_executable(global_partial_tree_with_trim ${EXAMPLE_SOURCE_DIR}/run_glob_partial_tree_trim.cpp bmgt_shared TRUE)
create_executable(global_partial_tree_with_symmetric ${EXAMPLE_SOURCE_DIR}/run_glob_partial_tree_symm.cpp bmgt_shared TRUE)
create_executable(global_partial_tree_with_misspecification_trim ${EXAMPLE_SOURCE_DIR}/run_glob_partial_tree_trim_misspec.cpp bmgt_shared TRUE)
create_executable(model_manipulation_benchmark ${EXAMPLE_SOURCE_DIR}/run_model_manipulation_benchmark.cpp bmgt_shared TRUE)
create_executable(BBPA_benchmark_intra ${EXAMPLE_SOURCE_DIR}/run_BBPA_benchmark_intra.cpp bmgt_shared TRUE)
create_executable(BBPA_benchmark ${EXAMPLE_SOURCE_DIR}/run_BBPA_benchmark.cpp bmgt_shared TRUE)

# Versions of BBPA benchmarks that link to specific versions of libraries
create_executable(BBPA_benchmark_intra_baseline ${EXAMPLE_SOURCE_DIR}/run_BBPA_benchmark_intra.cpp bmgt_baseline_shared TRUE)
create_executable(BBPA_benchmark_intra_op1 ${EXAMPLE_SOURCE_DIR}/run_BBPA_benchmark_intra.cpp bmgt_op1_shared TRUE)
create_executable(BBPA_benchmark_intra_op2 ${EXAMPLE_SOURCE_DIR}/run_BBPA_benchmark_intra.cpp bmgt_op2_shared TRUE)
create_executable(BBPA_benchmark_intra_op3 ${EXAMPLE_SOURCE_DIR}/run_BBPA_benchmark_intra.cpp bmgt_op3_shared TRUE)
create_executable(BBPA_benchmark_intra_op4 ${EXAMPLE_SOURCE_DIR}/run_BBPA_benchmark_intra.cpp bmgt_op4_shared TRUE)
create_executable(BBPA_benchmark_baseline ${EXAMPLE_SOURCE_DIR}/run_BBPA_benchmark.cpp bmgt_baseline_shared TRUE)
create_executable(BBPA_benchmark_op1 ${EXAMPLE_SOURCE_DIR}/run_BBPA_benchmark.cpp bmgt_op1_shared TRUE)
create_executable(BBPA_benchmark_op2 ${EXAMPLE_SOURCE_DIR}/run_BBPA_benchmark.cpp bmgt_op2_shared TRUE)
create_executable(BBPA_benchmark_op3 ${EXAMPLE_SOURCE_DIR}/run_BBPA_benchmark.cpp bmgt_op3_shared TRUE)
create_executable(BBPA_benchmark_op4 ${EXAMPLE_SOURCE_DIR}/run_BBPA_benchmark.cpp bmgt_op4_shared TRUE)

include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/license.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "${BMGT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${BMGT_VERSION_MINOR}")
set(CPACK_SOURCE_GENERATOR "TGZ")
include(CPack)