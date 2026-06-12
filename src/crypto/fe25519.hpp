#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace onion::crypto {

// Field element mod 2^255-19 in 5 x 51-bit limbs (donna64 representation).
struct Fe {
    std::uint64_t v[5];
};

[[nodiscard]] Fe fe_from_bytes(std::span<const std::byte, 32> s);
void fe_to_bytes(std::span<std::byte, 32> out, const Fe& f);

[[nodiscard]] Fe fe_add(const Fe& a, const Fe& b);
[[nodiscard]] Fe fe_sub(const Fe& a, const Fe& b);
[[nodiscard]] Fe fe_mul(const Fe& a, const Fe& b);
[[nodiscard]] Fe fe_sq(const Fe& a);
[[nodiscard]] Fe fe_invert(const Fe& a);  // a^(p-2)

}  // namespace onion::crypto
