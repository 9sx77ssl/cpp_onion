add_library(onion_engine_cpu STATIC
  naive_engine.cpp
  incremental_engine.cpp
  incremental_engine_x4.cpp
)
# incremental_engine_x4.cpp includes incremental_x4.hpp -> ge25519x4.hpp -> Fe4
# (__m256i by value), so it must be compiled with AVX2 even in debug/asan builds
# where -march=native is not set.
set_source_files_properties(incremental_engine_x4.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
target_link_libraries(onion_engine_cpu
  PUBLIC onion_engine onion_core
  PRIVATE onion_crypto onion_options Threads::Threads)
