add_executable(onion main.cpp)
target_link_libraries(onion PRIVATE
  onion_core onion_engine_cpu onion_io CLI11::CLI11 onion_options Threads::Threads)
