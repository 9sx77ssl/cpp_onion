add_library(onion_core STATIC
  base32.cpp
)
target_include_directories(onion_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(onion_core
  PUBLIC onion_crypto
  PRIVATE onion_options)
