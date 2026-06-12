#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace onion::crypto {

// SHA3-256 (FIPS 202). Compact, byte-oriented; cold path only (address
// checksums + verification), never on the search hot path.
class Sha3_256 {
public:
    static constexpr std::size_t digest_size = 32;
    static constexpr std::size_t rate = 136;  // bytes per block at 256-bit security

    void update(std::span<const std::byte> data);
    [[nodiscard]] std::array<std::byte, digest_size> finalize();

    [[nodiscard]] static std::array<std::byte, digest_size>
    hash(std::span<const std::byte> data) {
        Sha3_256 h;
        h.update(data);
        return h.finalize();
    }

private:
    void absorb_block();

    std::array<std::uint64_t, 25> state_{};
    std::array<std::byte, rate> buf_{};
    std::size_t buf_len_ = 0;
};

}  // namespace onion::crypto
