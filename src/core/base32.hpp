#pragma once

#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace onion::core {

inline constexpr std::string_view kBase32Alphabet = "abcdefghijklmnopqrstuvwxyz234567";

enum class Base32Error { invalid_char, nonzero_padding, invalid_length };

// RFC 4648 base32, lowercase, no '=' padding (Tor onion convention).
[[nodiscard]] std::string base32_encode(std::span<const std::byte> in);
[[nodiscard]] std::expected<std::vector<std::byte>, Base32Error>
base32_decode(std::string_view s);

}  // namespace onion::core
