include(Catch)

add_executable(onion_tests
  test_sanity.cpp
)
target_link_libraries(onion_tests PRIVATE Catch2::Catch2WithMain onion_options Threads::Threads)
catch_discover_tests(onion_tests)
