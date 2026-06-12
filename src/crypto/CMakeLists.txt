add_library(onion_crypto STATIC
  sha3.cpp
  keys.cpp
)
target_include_directories(onion_crypto PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(onion_crypto
  PUBLIC PkgConfig::SODIUM
  PRIVATE onion_options)
