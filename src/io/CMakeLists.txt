add_library(onion_io STATIC
  verifier.cpp
  tor_key_writer.cpp
)
target_link_libraries(onion_io
  PUBLIC onion_core onion_engine
  PRIVATE onion_options)
