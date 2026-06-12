add_library(onion_core STATIC
  base32.cpp
  onion_address.cpp
  matcher.cpp
)
target_include_directories(onion_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(onion_core
  PRIVATE onion_crypto onion_options)
