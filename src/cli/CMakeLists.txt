add_executable(onion main.cpp)
target_link_libraries(onion PRIVATE
  onion_core onion_engine_cpu onion_io CLI11::CLI11 onion_options Threads::Threads)

# CUDA engine: only linked + enabled when the CUDA backend is built. The
# #include and the --engine cuda branch in main.cpp are #ifdef ONION_CUDA.
if(ONION_CUDA)
  target_link_libraries(onion PRIVATE onion_engine_cuda)
  target_compile_definitions(onion PRIVATE ONION_CUDA)
endif()
