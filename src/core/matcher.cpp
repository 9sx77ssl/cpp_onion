#include "core/matcher.hpp"

#include "core/base32.hpp"

namespace onion::core {

std::expected<CompiledPattern, MatcherError> compile_prefix(std::string_view prefix) {
    if (prefix.empty()) return std::unexpected(MatcherError::empty);
    if (prefix.size() > kMaxPrefixLen) return std::unexpected(MatcherError::too_long);

    CompiledPattern out;
    out.prefix = std::string(prefix);

    std::size_t bit = 0;
    for (char c : prefix) {
        const auto pos = kBase32Alphabet.find(c);
        if (pos == std::string_view::npos)
            return std::unexpected(MatcherError::invalid_char);
        for (int k = 4; k >= 0; --k, ++bit) {
            const auto byte_idx = bit / 8;
            const auto bit_mask = std::uint8_t(0x80u >> (bit % 8));
            if ((pos >> k) & 1) out.bytes[byte_idx] |= bit_mask;
            out.mask[byte_idx] |= bit_mask;
        }
    }
    out.nbytes = (bit + 7) / 8;
    return out;
}

}  // namespace onion::core
