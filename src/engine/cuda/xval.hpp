#pragma once
//
// Plain-C++ boundary for the CUDA cross-validation harness (design §6: the
// boundary header is plain C++ -- no CUDA headers, no std::expected/std::print).
// Host C++23 code (the test) includes only this; the .cu owns all CUDA calls.

#include <cstdint>

namespace onion::cuda {

// Walk the incremental chain for `n_chains` independent base scalars on the
// device and write, for each chain c and step i, the 32-byte little-endian
// y-encoding of point (a0_c + 8*i)*B.
//
// Inputs:
//   a0        : n_chains * 32 bytes, each a clamped 32-byte LE scalar.
//   n_chains  : number of independent chains (one device thread each).
//   steps     : chain length N per chain (points 0..N-1).
// Output:
//   out_y     : n_chains * steps * 32 bytes; out_y[(c*steps + i)*32 .. +32) is
//               the y-encoding of the i-th point of chain c. Leading 31 bytes
//               are the pure y (the sign bit in byte 31 is not set).
//
// Host-side setup (start point A0 and the 8B cached-affine step) is done with
// the validated scalar code on the CPU and uploaded; the device only walks.
//
// Returns 0 on success, non-zero CUDA error code otherwise.
int run_incremental_xval(const uint8_t* a0, uint8_t* out_y, int n_chains, int steps);

}  // namespace onion::cuda
