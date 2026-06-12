#pragma once

#include "crypto/fe25519.hpp"

#include <array>
#include <cstddef>
#include <span>

namespace onion::crypto {

struct GeP3 { Fe X, Y, Z, T; };                 // extended: x=X/Z, y=Y/Z, T=XY/Z
struct GeCached { Fe YplusX, YminusX, Z, T2d; }; // precomputed addend

[[nodiscard]] GeP3 ge_basepoint();
[[nodiscard]] GeP3 ge_double(const GeP3& p);
[[nodiscard]] GeCached ge_to_cached(const GeP3& p);
[[nodiscard]] GeP3 ge_add(const GeP3& p, const GeCached& q);
[[nodiscard]] GeP3 ge_scalarmult_base(std::span<const std::byte, 32> scalar);  // setup only

// Full 32-byte encoding (little-endian y with x's sign in the top bit). Used by
// tests and on a match; the hot loop uses ge_affine_y_bytes with a batched Z^-1.
[[nodiscard]] std::array<std::byte, 32> ge_encode(const GeP3& p);

}  // namespace onion::crypto
