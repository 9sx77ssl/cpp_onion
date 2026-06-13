add_library(onion_engine INTERFACE)
target_include_directories(onion_engine INTERFACE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(onion_engine INTERFACE onion_crypto)

add_subdirectory(cpu)

if(ONION_CUDA)
  add_subdirectory(cuda)
endif()
