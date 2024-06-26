cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(GOOGLE_TEST_DIR "${PROJECT_BINARY_DIR}/_deps/googletest-src")


include(FetchContent)
if(NOT EXISTS ${GOOGLE_TEST_DIR})
  message(
    STATUS
      "Downloading googletest to ${PROJECT_BINARY_DIR}/_deps/googletest-src")
endif()
# This does not download googletest again if its already available in the
# CMakeCache file
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main)

FetchContent_MakeAvailable(googletest)

FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  message(STATUS "GoogleTest not populated")
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
else()
  message(STATUS "GoogleTest already populated")
endif()

message(STATUS "googletest_BINARY_DIR: ${googletest_BINARY_DIR}")
message(STATUS "googletest_SOURCE_DIR: ${googletest_SOURCE_DIR}")

target_include_directories(
  FLAT_NAV_LIB INTERFACE ${googletest_SOURCE_DIR}/googletest/include)
