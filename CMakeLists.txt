cmake_minimum_required(VERSION 3.28)
project(cpp_onion LANGUAGES CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ONION_NATIVE "Build with -march=native" ON)
option(ONION_SANITIZE "Enable ASan+UBSan" OFF)
option(ONION_USE_MOLD "Link with the mold linker when available" ON)
option(ONION_CUDA "Build the CUDA backend" OFF)

# CUDA is an OPTIONAL target (design §6 toolchain firewall): the rest of the
# project stays on its default host compiler (GCC 16) and builds CPU-only when
# nvcc is absent. Only when ONION_CUDA=ON do we enable the language so default
# presets are unaffected.
if(ONION_CUDA)
  # Pin the CUDA host compiler to g++-15 (the proven combo with nvcc 13.3 on
  # sm_75). NVCC_CCBIN may also be set system-wide; this is explicit and safe.
  if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-15)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 75)
  # CMake needs a KNOWN CUDA dialect to drive the .cu compiles; CMake 4.x + nvcc
  # 13.3 expose CUDA20 (not CUDA23 -- requesting 23 errors "does not support this
  # dialect"), so we satisfy the dialect machinery with standard 20. The device
  # TUs are actually compiled at C++23 via the explicit -std=c++23 in
  # src/engine/cuda/CMakeLists.txt (cuda_engine.cu needs matcher.hpp's
  # std::expected). To avoid CMake ALSO emitting its own -std=c++20 (which made
  # nvcc warn "incompatible redefinition for option std" on every .cu), clear
  # the CUDA20 standard/extension flag mappings so the explicit -std=c++23 is the
  # ONLY -std reaching nvcc.
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
  set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
  message(STATUS "CUDA backend enabled (sm_75, host compiler ${CMAKE_CUDA_HOST_COMPILER})")
endif()

if(ONION_USE_MOLD)
  find_program(MOLD_LINKER mold)
  if(MOLD_LINKER)
    add_link_options(-fuse-ld=mold)
    message(STATUS "Linking with mold: ${MOLD_LINKER}")
  else()
    message(STATUS "mold not found; using the default linker")
  endif()
endif()

add_library(onion_options INTERFACE)
target_compile_options(onion_options INTERFACE -Wall -Wextra -Wpedantic)
if(ONION_NATIVE)
  target_compile_options(onion_options INTERFACE -march=native)
endif()
# Release-only: Link-Time Optimization so cross-TU fe_mul/fe_sq inline into the
# point-add hot loop, plus loop unrolling and no semantic interposition (lets the
# linker treat global symbols as non-interposable, enabling cross-TU inlining).
target_compile_options(onion_options INTERFACE
  $<$<CONFIG:Release>:-flto=auto -fno-semantic-interposition -funroll-loops>)
target_link_options(onion_options INTERFACE
  $<$<CONFIG:Release>:-flto=auto -fno-semantic-interposition -funroll-loops>)
if(ONION_SANITIZE)
  target_compile_options(onion_options INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(onion_options INTERFACE -fsanitize=address,undefined)
endif()

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SODIUM REQUIRED IMPORTED_TARGET libsodium)

include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
FetchContent_Declare(Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.7.1
  GIT_SHALLOW TRUE)
FetchContent_Declare(CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.4.2
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(Catch2 CLI11)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_subdirectory(src)

add_subdirectory(tests)
