add_library(onion_engine_cpu STATIC
  naive_engine.cpp
  incremental_engine.cpp
)
target_link_libraries(onion_engine_cpu
  PUBLIC onion_engine onion_core
  PRIVATE onion_crypto onion_options Threads::Threads)
