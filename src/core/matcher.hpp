#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>

namespace onion::core {

enum class MatcherError { empty, invalid_char, too_long };

// A base32 prefix compiled to a byte pattern + mask over the leading
// public-key bytes. Limit 49 chars: beyond stream bit 247 the x-sign bit
// (bit 248) participates, and Phase 1's hot path matches on y alone
// (design doc §0).
struct CompiledPattern {
    std::array<std::uint8_t, 32> bytes{};
    std::array<std::uint8_t, 32> mask{};
    std::size_t nbytes = 0;
    std::string prefix;
};

inline constexpr std::size_t kMaxPrefixLen = 49;

[[nodiscard]] std::expected<CompiledPattern, MatcherError>
compile_prefix(std::string_view prefix);

[[nodiscard]] inline bool matches(const CompiledPattern& p,
                                  std::span<const std::byte, 32> pubkey) noexcept {
    for (std::size_t i = 0; i < p.nbytes; ++i)
        if ((std::to_integer<std::uint8_t>(pubkey[i]) & p.mask[i]) != p.bytes[i])
            return false;
    return true;
}

}  // namespace onion::core
