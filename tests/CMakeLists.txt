include(Catch)

add_executable(onion_tests
  test_sanity.cpp
  test_sha3.cpp
  test_base32.cpp
  test_onion_address.cpp
  test_keys.cpp
  test_matcher.cpp
  test_engine_primitives.cpp
  test_naive_engine.cpp
  test_verifier.cpp
  test_tor_key_writer.cpp
  test_fe25519.cpp
  test_fe25519x4.cpp
  test_ge25519.cpp
  test_ge25519x4.cpp
  test_incremental.cpp
  test_incremental_engine.cpp
  test_incremental_x4.cpp
  test_incremental_engine_x4.cpp
)
target_link_libraries(onion_tests PRIVATE Catch2::Catch2WithMain onion_options Threads::Threads
  onion_crypto onion_core onion_engine onion_engine_cpu onion_io)
# test_fe25519x4.cpp and test_ge25519x4.cpp pass Fe4/GeP3x4 (which contain
# __m256i) by value, so they need AVX2 even in debug/asan builds where
# -march=native is not set.
set_source_files_properties(test_fe25519x4.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
set_source_files_properties(test_ge25519x4.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
# test_incremental_x4.cpp includes incremental_x4.hpp -> ge25519x4.hpp -> Fe4 (__m256i).
set_source_files_properties(test_incremental_x4.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
# test_incremental_engine_x4.cpp includes incremental_engine_x4.hpp which transitively
# pulls in incremental_x4.hpp -> ge25519x4.hpp -> Fe4 (__m256i by value).
set_source_files_properties(test_incremental_engine_x4.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
catch_discover_tests(onion_tests)

# CUDA cross-validation gate: only built/registered when ONION_CUDA is ON, so
# CPU-only presets (debug/release/asan) are unaffected and need no nvcc.
if(ONION_CUDA)
  add_executable(onion_cuda_tests test_cuda_xval.cpp)
  target_link_libraries(onion_cuda_tests PRIVATE
    Catch2::Catch2WithMain onion_options onion_engine_cuda PkgConfig::SODIUM)
  catch_discover_tests(onion_cuda_tests TEST_PREFIX "cuda.")
endif()

add_test(NAME oracle_self_test
         COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/verify_onion.py --self-test)

add_test(NAME e2e_search_and_oracle
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/e2e/run_e2e.sh
                 $<TARGET_FILE:onion>
                 ${CMAKE_SOURCE_DIR}/tools/verify_onion.py)
set_tests_properties(e2e_search_and_oracle PROPERTIES TIMEOUT 120)
