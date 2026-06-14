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
# We deliberately do NOT set a CUDA_STANDARD property here: CMake 4.x + nvcc
# 13.3 expose CUDA20 but not CUDA23 (requesting 23 errors), and a CUDA_STANDARD
# value would re-introduce CMake's auto -std flag. The top-level CMakeLists
# satisfies CMake's dialect machinery with standard 20 but clears its
# -std flag mapping, so the explicit -std=c++23 below is the single -std
# reaching nvcc (no "incompatible redefinition for option std" warning).
set_target_properties(onion_engine_cuda PROPERTIES
  CUDA_ARCHITECTURES 75
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF)

# cuda_engine.cu includes core/matcher.hpp, which declares compile_prefix with
# std::expected (a C++23 facility). nvcc 13.3 accepts -std=c++23 and forwards
# the matching flag to the g++-15 host pass; this is the ONLY -std reaching nvcc
# (the top-level CMakeLists clears CMake's auto CUDA -std flag so there is no
# duplicate that would make nvcc warn "incompatible redefinition for option std").
target_compile_options(onion_engine_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-std=c++23>)

# Steps-per-thread (M): the Montgomery batch-inversion chain length. Compile-time
# because it sizes the per-thread local arrays and loop bounds. Default 0 keeps
# the header's built-in 3072 knee (search_kernel.cuh); set ONION_CUDA_M to a
# positive value (e.g. -DONION_CUDA_M=512) to sweep a smaller per-thread local
# footprint. Applied to the whole onion_engine_cuda target so the kernel (array
# sizes) and host (per_launch accounting) both observe the same M.
set(ONION_CUDA_M "0" CACHE STRING "Steps-per-thread M (0 = header default of 3072)")
if(NOT ONION_CUDA_M STREQUAL "0")
  target_compile_definitions(onion_engine_cuda PRIVATE ONION_CUDA_M=${ONION_CUDA_M})
  message(STATUS "CUDA steps-per-thread M overridden: ${ONION_CUDA_M}")
endif()

# Device field backend selector. ON (default) = native 8x32-bit limb field
# (device_field32.cuh), the Turing speedup path. OFF = validated 5x51-bit
# __int128 reference field (device_field.cuh). The xval gate cross-validates
# whichever is selected against libsodium bit-for-bit, so a wrong field can
# never ship; only the selected field is built (the other path is compiled on
# demand by reconfiguring with the option flipped).
option(ONION_CUDA_FIELD32 "Use the native 32-bit limb device field on GPU" ON)
if(ONION_CUDA_FIELD32)
  target_compile_definitions(onion_engine_cuda PRIVATE ONION_CUDA_FIELD32)
  message(STATUS "CUDA device field: native 8x32-bit limbs (ONION_CUDA_FIELD32=ON)")
else()
  message(STATUS "CUDA device field: 5x51-bit __int128 (ONION_CUDA_FIELD32=OFF)")
endif()
