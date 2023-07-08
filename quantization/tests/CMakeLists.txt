cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
enable_testing()
include(GoogleTest)

# Add test executables here 
set(PQ_TESTS test_kmeans test_pq_manager test_pq_e2e)


foreach(PQ_TEST IN LISTS PQ_TESTS)
  add_executable(${PQ_TEST} ${PQ_TEST}.cpp)
  target_link_libraries(${PQ_TEST} gtest gtest_main FLAT_NAV_LIB)

  gtest_discover_tests(${PQ_TEST} DISCOVERY_TIMEOUT 60)
  install(TARGETS ${PQ_TEST} RUNTIME DESTINATION bin)

endforeach(PQ_TEST IN LISTS PQ_TESTS)