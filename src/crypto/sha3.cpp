#include "crypto/sha3.hpp"

#include <algorithm>

namespace onion::crypto {
namespace {

constexpr std::array<std::uint64_t, 24> kRoundConstants = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};
constexpr std::array<int, 24> kRotc = {1,  3,  6,  10, 15, 21, 28, 36,
                                       45, 55, 2,  14, 27, 41, 56, 8,
                                       25, 43, 62, 18, 39, 61, 20, 44};
constexpr std::array<int, 24> kPiln = {10, 7,  11, 17, 18, 3,  5,  16,
                                       8,  21, 24, 4,  15, 23, 19, 13,
                                       12, 2,  20, 14, 22, 9,  6,  1};

constexpr std::uint64_t rotl64(std::uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

void keccakf(std::array<std::uint64_t, 25>& st) {
    for (int round = 0; round < 24; ++round) {
        // theta
        std::uint64_t bc[5];
        for (int i = 0; i < 5; ++i)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        for (int i = 0; i < 5; ++i) {
            const std::uint64_t t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
        }
        // rho + pi
        std::uint64_t t = st[1];
        for (int i = 0; i < 24; ++i) {
            const int j = kPiln[i];
            const std::uint64_t tmp = st[j];
            st[j] = rotl64(t, kRotc[i]);
            t = tmp;
        }
        // chi
        for (int j = 0; j < 25; j += 5) {
            std::uint64_t b[5];
            for (int i = 0; i < 5; ++i) b[i] = st[j + i];
            for (int i = 0; i < 5; ++i)
                st[j + i] = b[i] ^ (~b[(i + 1) % 5] & b[(i + 2) % 5]);
        }
        // iota
        st[0] ^= kRoundConstants[round];
    }
}

}  // namespace

void Sha3_256::update(std::span<const std::byte> data) {
    for (std::byte b : data) {
        buf_[buf_len_++] = b;
        if (buf_len_ == rate) {
            absorb_block();
            buf_len_ = 0;
        }
    }
}

void Sha3_256::absorb_block() {
    for (std::size_t i = 0; i < rate / 8; ++i) {
        std::uint64_t lane = 0;
        for (int j = 7; j >= 0; --j)
            lane = (lane << 8) | std::to_integer<std::uint64_t>(buf_[i * 8 + std::size_t(j)]);
        state_[i] ^= lane;
    }
    keccakf(state_);
}

std::array<std::byte, Sha3_256::digest_size> Sha3_256::finalize() {
    std::fill(buf_.begin() + std::ptrdiff_t(buf_len_), buf_.end(), std::byte{0});
    buf_[buf_len_] = std::byte{0x06};   // SHA3 domain separation + pad10*1 start
    buf_[rate - 1] |= std::byte{0x80};  // pad10*1 end
    absorb_block();

    std::array<std::byte, digest_size> out;
    for (std::size_t i = 0; i < digest_size / 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            out[i * 8 + j] = std::byte((state_[i] >> (8 * j)) & 0xff);
    return out;
}

}  // namespace onion::crypto
