include(Catch)

add_executable(onion_tests
  test_sanity.cpp
  test_sha3.cpp
  test_base32.cpp
  test_onion_address.cpp
  test_keys.cpp
  test_matcher.cpp
  test_engine_primitives.cpp
)
target_link_libraries(onion_tests PRIVATE Catch2::Catch2WithMain onion_options Threads::Threads
  onion_crypto onion_core onion_engine)
catch_discover_tests(onion_tests)
