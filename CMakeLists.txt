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
include(cmake/FindAVX.cmake)

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

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  message(STATUS "OpenMP Found. Building the Package using the system OpenMP.")
else()
  message(
    FATAL_ERROR
      "OpenMP Not Found. Building the Package using LLVM's OpenMP. This is slower than the system OpenMP."
  )
endif(OpenMP_FOUND)

option(BUILD_TESTS "Build all tests")
option(BUIL_EXAMPLES "Build examples")
option(BUILD_BENCHMARKS "Build ANNS benchmarks")
option(NO_SIMD_VECTORIZATION "Disable using SIMD instructions")
message(STATUS "Building tests: ${BUILD_TESTS}")
message(STATUS "Building examples: ${BUILD_EXAMPLES}")
message(STATUS "Building benchmarks: ${BUILD_BENCHMARKS}")

# Enable auto-vectorization if we are not using SIMD.
if(NO_SIMD_VECTORIZATION)
  message(STATUS "Disabling using SIMD instructions")
  add_definitions(-DNO_SIMD_VECTORIZATION)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize")
endif()

# TODO: Using globbing or some other command that does not require writing down
# every header file
set(HEADERS
    ${PROJECT_SOURCE_DIR}/flatnav/distances/InnerProductDistance.h
    ${PROJECT_SOURCE_DIR}/flatnav/distances/SquaredL2Distance.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/SquaredL2SimdExtensions.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/InnerProductSimdExtensions.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/VisitedSetPool.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/GorderPriorityQueue.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/Reordering.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/ParallelConstructs.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/Macros.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/Datatype.h
    ${PROJECT_SOURCE_DIR}/flatnav/util/SimdBaseTypes.h
    ${PROJECT_SOURCE_DIR}/flatnav/DistanceInterface.h
    ${PROJECT_SOURCE_DIR}/flatnav/Index.h
    # ${PROJECT_SOURCE_DIR}/quantization/ProductQuantization.h
    ${PROJECT_SOURCE_DIR}/quantization/CentroidsGenerator.h
    ${PROJECT_SOURCE_DIR}/quantization/Utils.h)

# Interface here means that the library is header only
add_library(FLAT_NAV_LIB INTERFACE)
target_sources(FLAT_NAV_LIB INTERFACE ${HEADERS})
target_include_directories(FLAT_NAV_LIB INTERFACE ${PROJECT_SOURCE_DIR})

target_link_libraries(FLAT_NAV_LIB INTERFACE OpenMP::OpenMP_CXX)

if(BUILD_EXAMPLES)
  message(STATUS "Building examples")
  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
  include(cmake/FindCNPYAndZLIB.cmake)
  add_subdirectory(${PROJECT_SOURCE_DIR}/tools)
endif()

if(BUILD_TESTS)
  message(STATUS "Building flatnav + quantization unit tests using gtest")
  include(cmake/FindGoogleTest.cmake)
  add_subdirectory(${PROJECT_SOURCE_DIR}/flatnav/tests)
  # add_subdirectory(${PROJECT_SOURCE_DIR}/quantization/tests)
endif()

if(BUILD_BENCHMARKS)
  message(STATUS "Building benchmarks")
  if(NOT TARGET CNPY)
    include(cmake/FindCNPYAndZLIB.cmake)
  endif()
  add_subdirectory(${PROJECT_SOURCE_DIR}/benchmarks)
endif()
