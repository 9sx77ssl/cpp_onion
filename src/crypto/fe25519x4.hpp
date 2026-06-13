#pragma once

// AVX2 4-wide field arithmetic over GF(2^255-19), radix 2^25.5 (10 unsigned
// limbs, agl's curve25519-donna 32-bit layout). Each Fe4 holds four
// independent field elements in struct-of-arrays form: v[i] is a __m256i whose
// four 64-bit lanes hold limb i of candidates 0..3. The four lanes never
// interact — carries propagate only along the limb axis within a lane.
//
// Correctness is gated by tests/test_fe25519x4.cpp, which cross-validates every
// operation bit-exactly against the scalar onion::crypto::Fe field.

#include <immintrin.h>

#include <array>
#include <cstddef>

namespace onion::crypto {

struct Fe4 {
    __m256i v[10];
};

// Load four 32-byte little-endian field elements into SoA form.
[[nodiscard]] Fe4 fe4_load(const std::array<std::byte, 32>& a,
                           const std::array<std::byte, 32>& b,
                           const std::array<std::byte, 32>& c,
                           const std::array<std::byte, 32>& d);

// Store the four lanes as canonical 32-byte little-endian encodings.
void fe4_store_y(const Fe4& f, std::array<std::byte, 32>& a, std::array<std::byte, 32>& b,
                 std::array<std::byte, 32>& c, std::array<std::byte, 32>& d);

[[nodiscard]] Fe4 fe4_mul(const Fe4& a, const Fe4& b);
[[nodiscard]] Fe4 fe4_sq(const Fe4& a);
[[nodiscard]] Fe4 fe4_add(const Fe4& a, const Fe4& b);
[[nodiscard]] Fe4 fe4_sub(const Fe4& a, const Fe4& b);
[[nodiscard]] Fe4 fe4_invert(const Fe4& a);  // a^(p-2)

}  // namespace onion::crypto
