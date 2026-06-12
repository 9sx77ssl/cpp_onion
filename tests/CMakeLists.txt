include(Catch)

add_executable(onion_tests
  test_sanity.cpp
  test_sha3.cpp
  test_base32.cpp
)
target_link_libraries(onion_tests PRIVATE Catch2::Catch2WithMain onion_options Threads::Threads
  onion_crypto onion_core)
catch_discover_tests(onion_tests)
