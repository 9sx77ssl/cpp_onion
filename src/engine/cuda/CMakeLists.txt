# CUDA backend (optional target; only added when ONION_CUDA is ON).
# Builds a static lib from the .cu files. The boundary headers (xval.hpp) are
# plain C++ included by host C++23; all CUDA code stays inside this lib.

add_library(onion_engine_cuda STATIC
  xval_kernel.cu
)
# Device-core headers live under src/; let the .cu find "engine/cuda/..." and
# "crypto/...". onion_crypto is linked for the host-side scalar setup.
target_include_directories(onion_engine_cuda PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(onion_engine_cuda PUBLIC onion_crypto)

# Turing (sm_75); position-independent so it can link into shared host targets.
set_target_properties(onion_engine_cuda PROPERTIES
  CUDA_ARCHITECTURES 75
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF)
