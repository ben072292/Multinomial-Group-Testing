cmake_minimum_required(VERSION 3.10)
project(BMGT VERSION 1.0 DESCRIPTION "Scalable Bayesian Multinomial Group Testing Framework")
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_COMPILER "mpicxx") # Set the compiler to mpicxx
set(CMAKE_C_COMPILER "mpicc") # Corrected typo
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
add_bmgt_library(bmgt_v1 BBPA_V1)
add_bmgt_library(bmgt_v2 BBPA_V2)
add_bmgt_library(bmgt_v3 BBPA_V3)
add_bmgt_library(bmgt ENABLE_SIMD)

add_executable(gt_intra
    ${EXAMPLE_SOURCE_DIR}/run_glob_tree_intra.cpp
)

add_executable(gt_intra_dbg
    ${EXAMPLE_SOURCE_DIR}/run_glob_tree_intra.cpp
)

add_executable(gt
    ${EXAMPLE_SOURCE_DIR}/run_glob_tree.cpp
)

add_executable(ft
    ${EXAMPLE_SOURCE_DIR}/run_fusion_tree.cpp
)

add_executable(dt_intra
    ${EXAMPLE_SOURCE_DIR}/run_dist_tree_intra.cpp
)

add_executable(dt
    ${EXAMPLE_SOURCE_DIR}/run_dist_tree.cpp
)

add_executable(dt_gt
    ${EXAMPLE_SOURCE_DIR}/run_dist_glob_tree.cpp
)

add_executable(dt_gt_trim
    ${EXAMPLE_SOURCE_DIR}/run_dist_glob_tree_trim.cpp
)

add_executable(dt_gt_symm
    ${EXAMPLE_SOURCE_DIR}/run_dist_glob_tree_symm.cpp
)

add_executable(gpt_trim
    ${EXAMPLE_SOURCE_DIR}/run_glob_partial_tree_trim.cpp
)

add_executable(gpt_symm
    ${EXAMPLE_SOURCE_DIR}/run_glob_partial_tree_symm.cpp
)

add_executable(gpt_trim_misspecification
    ${EXAMPLE_SOURCE_DIR}/run_glob_partial_tree_trim_misspec.cpp
)

add_executable(ts_intra
    ${EXAMPLE_SOURCE_DIR}/run_test_selection_intra.cpp
)

add_executable(ts
    ${EXAMPLE_SOURCE_DIR}/run_test_selection.cpp
)

add_executable(ts_intra_v1 ${EXAMPLE_SOURCE_DIR}/run_test_selection_intra.cpp)
add_executable(ts_intra_v2 ${EXAMPLE_SOURCE_DIR}/run_test_selection_intra.cpp)
add_executable(ts_intra_v3 ${EXAMPLE_SOURCE_DIR}/run_test_selection_intra.cpp)

add_executable(ts_v1 ${EXAMPLE_SOURCE_DIR}/run_test_selection.cpp)
add_executable(ts_v2 ${EXAMPLE_SOURCE_DIR}/run_test_selection.cpp)
add_executable(ts_v3 ${EXAMPLE_SOURCE_DIR}/run_test_selection.cpp)

add_dependencies(gt_intra bmgt_shared)
add_dependencies(gt_intra_dbg bmgt_shared)
add_dependencies(gt bmgt_shared)
add_dependencies(ft bmgt_shared)
add_dependencies(dt_intra bmgt_shared)
add_dependencies(dt bmgt_shared)
add_dependencies(dt_gt bmgt_shared)
add_dependencies(dt_gt_trim bmgt_shared)
add_dependencies(dt_gt_symm bmgt_shared)
add_dependencies(gpt_trim bmgt_shared)
add_dependencies(gpt_trim_misspecification bmgt_shared)
add_dependencies(gpt_symm bmgt_shared)
add_dependencies(ts_intra bmgt_shared)
add_dependencies(ts bmgt_shared)
add_dependencies(ts_intra_v1 bmgt_v1_shared)
add_dependencies(ts_intra_v2 bmgt_v2_shared)
add_dependencies(ts_intra_v3 bmgt_v3_shared)

add_dependencies(ts_v1 bmgt_v1_shared)
add_dependencies(ts_v2 bmgt_v2_shared)
add_dependencies(ts_v3 bmgt_v3_shared)

target_link_libraries(gt_intra bmgt_shared)
target_include_directories(gt_intra PRIVATE ${CMAKE_BINARY_DIR}/include)

target_link_libraries(gt_intra_dbg
    bmgt_shared
)

target_include_directories(gt_intra_dbg PRIVATE ${CMAKE_BINARY_DIR}/include)

target_link_libraries(gt
    bmgt_shared
)
target_include_directories(gt PRIVATE ${CMAKE_BINARY_DIR}/include)

target_link_libraries(ft
    bmgt_shared
)
target_include_directories(ft PRIVATE ${CMAKE_BINARY_DIR}/include)

target_link_libraries(dt_intra
    bmgt_shared
)
target_include_directories(dt_intra PRIVATE ${CMAKE_BINARY_DIR}/include)

target_link_libraries(dt
    bmgt_shared
)
target_include_directories(dt PRIVATE ${CMAKE_BINARY_DIR}/include)

target_link_libraries(dt_gt
    bmgt_shared
)
target_include_directories(dt_gt PRIVATE ${CMAKE_BINARY_DIR}/include)

target_link_libraries(dt_gt_trim
    bmgt_shared
)
target_include_directories(dt_gt_trim PRIVATE ${CMAKE_BINARY_DIR}/include)

target_link_libraries(dt_gt_symm
    bmgt_shared
)
target_include_directories(dt_gt_symm PRIVATE ${CMAKE_BINARY_DIR}/include)

target_link_libraries(gpt_trim
    bmgt_shared
)
target_include_directories(gpt_trim PRIVATE ${CMAKE_BINARY_DIR}/include)

target_link_libraries(gpt_trim_misspecification
    bmgt_shared
)
target_include_directories(gpt_trim_misspecification PRIVATE ${CMAKE_BINARY_DIR}/include)

target_link_libraries(gpt_symm
    bmgt_shared
)
target_include_directories(gpt_symm PRIVATE ${CMAKE_BINARY_DIR}/include)

target_link_libraries(ts_intra
    bmgt_shared
)
target_include_directories(ts_intra PRIVATE ${CMAKE_BINARY_DIR}/include)

target_link_libraries(ts
    bmgt_shared
)
target_include_directories(ts PRIVATE ${CMAKE_BINARY_DIR}/include)

target_link_libraries(ts_intra_v1 bmgt_v1_shared)
target_include_directories(ts_intra_v1 PRIVATE ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ts_intra_v2 bmgt_v2_shared)
target_include_directories(ts_intra_v2 PRIVATE ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ts_intra_v3 bmgt_v3_shared)
target_include_directories(ts_intra_v3 PRIVATE ${CMAKE_BINARY_DIR}/include)

target_link_libraries(ts_v1 bmgt_v1_shared)
target_include_directories(ts_v1 PRIVATE ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ts_v2 bmgt_v2_shared)
target_include_directories(ts_v2 PRIVATE ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ts_v3 bmgt_v3_shared)
target_include_directories(ts_v3 PRIVATE ${CMAKE_BINARY_DIR}/include)

include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "${BMGT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${BMGT_VERSION_MINOR}")
set(CPACK_SOURCE_GENERATOR "TGZ")
include(CPack)