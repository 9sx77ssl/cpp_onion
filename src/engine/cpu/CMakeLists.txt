add_library(onion_engine_cpu STATIC
  naive_engine.cpp
)
target_link_libraries(onion_engine_cpu
  PUBLIC onion_engine onion_core
  PRIVATE onion_options Threads::Threads)
