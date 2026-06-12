#include "core/base32.hpp"

#include <cstdint>

namespace onion::core {

std::string base32_encode(std::span<const std::byte> in) {
    std::string out;
    out.reserve((in.size() * 8 + 4) / 5);
    std::uint32_t acc = 0;
    int bits = 0;
    for (std::byte b : in) {
        acc = (acc << 8) | std::to_integer<std::uint32_t>(b);
        bits += 8;
        while (bits >= 5) {
            bits -= 5;
            out.push_back(kBase32Alphabet[(acc >> bits) & 0x1f]);
        }
    }
    if (bits > 0) out.push_back(kBase32Alphabet[(acc << (5 - bits)) & 0x1f]);
    return out;
}

std::expected<std::vector<std::byte>, Base32Error> base32_decode(std::string_view s) {
    // RFC 4648 unpadded base32 lengths satisfy size % 8 in {0,2,4,5,7};
    // {1,3,6} are structurally impossible (would need fractional bytes).
    static constexpr bool kValidMod8[8] = {true, false, true, false,
                                           true, true, false, true};
    if (!kValidMod8[s.size() % 8])
        return std::unexpected(Base32Error::invalid_length);

    std::vector<std::byte> out;
    out.reserve(s.size() * 5 / 8);
    std::uint32_t acc = 0;
    int bits = 0;
    for (char c : s) {
        const auto pos = kBase32Alphabet.find(c);
        if (pos == std::string_view::npos)
            return std::unexpected(Base32Error::invalid_char);
        acc = (acc << 5) | static_cast<std::uint32_t>(pos);
        bits += 5;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(std::byte((acc >> bits) & 0xff));
        }
    }
    if (bits > 0 && (acc & ((1u << bits) - 1)) != 0)
        return std::unexpected(Base32Error::nonzero_padding);
    return out;
}

}  // namespace onion::core
