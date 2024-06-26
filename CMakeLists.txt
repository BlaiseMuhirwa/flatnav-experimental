cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(FlatNav CXX)

# print ${PROJECT_SOURCE_DIR} and ${PROJECT_BINARY_DIR}
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")
message(STATUS "CMake CXX Compiler: ${CMAKE_CXX_COMPILER}")

# Add the parent directory to the include path This allows us to write #include
# <flatnav/...> instead of #include "../flatnav/..."
include_directories(${CMAKE_SOURCE_DIR})

# Include cereal
include_directories(${CMAKE_SOURCE_DIR}/external/cereal/include/)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} \
    -Ofast \
    -DHAVE_CXX0X \
    -DNDEBUG \
    -fopenmp \
    -fpic \
    -w \
    -ffast-math \
    -funroll-loops")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(cmake/find_avx.cmake)

option(CMAKE_BUILD_TYPE "Build type" Release)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  # Add debug compile flags
  message(STATUS "Building in Debug mode")
  # Address sanitizer: https://clang.llvm.org/docs/AddressSanitizer.html
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -fsanitize=address")
endif()

include(FeatureSummary)
# All options summary
feature_summary(WHAT ALL)

include_directories("${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}")


if(APPLE)
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
  set(OpenMP_CXX_LIB_NAMES "omp")
  set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  message(STATUS "OpenMP Found. Building the Package using the system OpenMP.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
  message(
    FATAL_ERROR
      "OpenMP Not Found. Building the Package using LLVM's OpenMP. This is slower than the system OpenMP."
  )
endif(OpenMP_FOUND)

option(BUILD_TESTS "Build all tests")
option(BUIL_EXAMPLES "Build examples")
option(NO_SIMD_VECTORIZATION "Disable using SIMD instructions")
message(STATUS "Building tests: ${BUILD_TESTS}")
message(STATUS "Building examples: ${BUILD_EXAMPLES}")

# Enable auto-vectorization if we are not using SIMD.
if(NO_SIMD_VECTORIZATION)
  message(STATUS "Disabling using SIMD instructions")
  add_definitions(-DNO_SIMD_VECTORIZATION)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize")
endif()

# TODO: Using globbing or some other command that does not require writing down
# every header file
set(HEADERS
    ${PROJECT_SOURCE_DIR}/flatnav/distances/inner_product_distance.h
    ${PROJECT_SOURCE_DIR}/flatnav/distances/squared_l2_distance.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/squared_l2_simd_extensions.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/inner_product_simd_extensions.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/visited_set_pool.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/gorder_priority_queue.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/reordering.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/parallel_constructs.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/macros.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/simd_base_types.h
    ${PROJECT_SOURCE_DIR}/flatnav/distance_interface.h
    ${PROJECT_SOURCE_DIR}/flatnav/index.h
    ${PROJECT_SOURCE_DIR}/quantization/product_quantization.h
    ${PROJECT_SOURCE_DIR}/quantization/centroids_generator.h
    ${PROJECT_SOURCE_DIR}/quantization/utils.h)

# Interface here means that the library is header only
add_library(FLAT_NAV_LIB INTERFACE)
target_sources(FLAT_NAV_LIB INTERFACE ${HEADERS})
target_include_directories(FLAT_NAV_LIB INTERFACE ${PROJECT_SOURCE_DIR})

target_link_libraries(FLAT_NAV_LIB INTERFACE OpenMP::OpenMP_CXX)

if(BUILD_EXAMPLES)
  message(STATUS "Building examples")
  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
  include(cmake/find_cnpy_and_zlib.cmake)
  add_subdirectory(${PROJECT_SOURCE_DIR}/tools)
endif()

if(BUILD_TESTS)
  message(STATUS "Building flatnav + quantization unit tests using gtest")
  include(cmake/find_google_test.cmake)
  add_subdirectory(${PROJECT_SOURCE_DIR}/flatnav/tests)
  # add_subdirectory(${PROJECT_SOURCE_DIR}/quantization/tests)
endif()

