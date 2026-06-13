add_library(onion_crypto STATIC
  sha3.cpp
  keys.cpp
  fe25519.cpp
  fe25519x4.cpp
  ge25519.cpp
  ge25519x4.cpp
  incremental.cpp
  incremental_x4.cpp
)
# fe25519x4.cpp, ge25519x4.cpp, and incremental_x4.cpp need AVX2 even in
# debug/asan builds (where -march=native is not set).
set_source_files_properties(fe25519x4.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
set_source_files_properties(ge25519x4.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
set_source_files_properties(incremental_x4.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
target_include_directories(onion_crypto PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(onion_crypto
  PUBLIC PkgConfig::SODIUM
  PRIVATE onion_options)
