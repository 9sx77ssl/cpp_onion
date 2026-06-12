cmake_minimum_required(VERSION 3.28)
project(cpp_onion LANGUAGES CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ONION_NATIVE "Build with -march=native" ON)
option(ONION_SANITIZE "Enable ASan+UBSan" OFF)

add_library(onion_options INTERFACE)
target_compile_options(onion_options INTERFACE -Wall -Wextra -Wpedantic)
if(ONION_NATIVE)
  target_compile_options(onion_options INTERFACE -march=native)
endif()
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
