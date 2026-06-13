# CUDA backend (optional target; only added when ONION_CUDA is ON).
# Builds a static lib from the .cu files. The boundary headers (xval.hpp) are
# plain C++ included by host C++23; all CUDA code stays inside this lib.

add_library(onion_engine_cuda STATIC
  xval_kernel.cu
  search_kernel.cu
  cuda_engine.cu
)
# Device-core headers live under src/; let the .cu find "engine/cuda/..." and
# "crypto/...". onion_crypto is linked for the host-side scalar setup;
# onion_core for core::CompiledPattern (the CudaEngine pattern type).
target_include_directories(onion_engine_cuda PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(onion_engine_cuda PUBLIC onion_crypto onion_core)

# Turing (sm_75); position-independent so it can link into shared host targets.
set_target_properties(onion_engine_cuda PROPERTIES
  CUDA_ARCHITECTURES 75
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF)

# cuda_engine.cu includes core/matcher.hpp, which declares compile_prefix with
# std::expected (a C++23 facility). CMake's CUDA dialect table doesn't expose
# "cuda_std_23" for this nvcc, so request the dialect directly: nvcc 13.3
# accepts -std=c++23 and forwards the matching flag to the g++-15 host pass.
target_compile_options(onion_engine_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-std=c++23>)

# Device field backend selector. OFF (default) = validated 5x51-bit __int128
# field (device_field.cuh). ON = native 8x32-bit limb field (device_field32.cuh),
# the Turing speedup path. The xval gate cross-validates whichever is selected
# against libsodium bit-for-bit, so a wrong field can never ship.
option(ONION_CUDA_FIELD32 "Use the native 32-bit limb device field on GPU" ON)
if(ONION_CUDA_FIELD32)
  target_compile_definitions(onion_engine_cuda PRIVATE ONION_CUDA_FIELD32)
  message(STATUS "CUDA device field: native 8x32-bit limbs (ONION_CUDA_FIELD32=ON)")
else()
  message(STATUS "CUDA device field: 5x51-bit __int128 (ONION_CUDA_FIELD32=OFF)")
endif()
