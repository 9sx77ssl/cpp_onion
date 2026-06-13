// THE GATE: prove the CUDA device crypto core matches libsodium bit-for-bit.
//
// For several random clamped base scalars a0 and N points each, run the device
// incremental chain and check that every point's leading 31 y-bytes equal
// crypto_scalarmult_ed25519_base_noclamp(a0 + 8i). Registered with ctest only
// when ONION_CUDA is ON, so CPU-only builds/CI are unaffected. Never weaken.

#include <catch2/catch_test_macros.hpp>

#include "engine/cuda/xval.hpp"

#include <sodium.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

// a += 8 as a 256-bit little-endian integer (matches base_noclamp's stepping).
void add8_le(unsigned char* a) {
    unsigned int carry = 8;
    for (int k = 0; k < 32 && carry; ++k) {
        unsigned int s = a[k] + (carry & 0xff);
        a[k] = (unsigned char)(s & 0xff);
        carry = s >> 8;
    }
}

}  // namespace

TEST_CASE("CUDA device chain matches libsodium base_noclamp(a0+8i) bit-for-bit") {
    REQUIRE(sodium_init() >= 0);

    constexpr int kChains = 4;
    constexpr int kSteps = 1024;  // N >= 1024 per chain

    std::mt19937_64 rng(20260613);
    std::vector<std::uint8_t> a0((size_t)kChains * 32);
    for (int c = 0; c < kChains; ++c) {
        std::uint8_t* s = a0.data() + (size_t)c * 32;
        for (int k = 0; k < 32; ++k) s[k] = (std::uint8_t)(rng() & 0xff);
        s[0] &= 0xf8;                       // clamp low 3 bits
        s[31] = (s[31] & 0x7f) | 0x40;      // clamp high bits
    }

    std::vector<std::uint8_t> out_y((size_t)kChains * kSteps * 32);
    int rc = onion::cuda::run_incremental_xval(a0.data(), out_y.data(), kChains, kSteps);
    REQUIRE(rc == 0);  // device executed cleanly

    for (int c = 0; c < kChains; ++c) {
        std::array<unsigned char, 32> a;
        std::memcpy(a.data(), a0.data() + (size_t)c * 32, 32);
        const std::uint8_t* chain = out_y.data() + (size_t)c * kSteps * 32;
        for (int i = 0; i < kSteps; ++i) {
            std::array<unsigned char, 32> ref;
            REQUIRE(crypto_scalarmult_ed25519_base_noclamp(ref.data(), a.data()) == 0);
            // Leading 31 bytes are the pure y; byte 31 differs by the sign bit.
            const std::uint8_t* dev = chain + (size_t)i * 32;
            bool ok = std::memcmp(dev, ref.data(), 31) == 0;
            if (!ok) {
                INFO("chain " << c << " step " << i << " mismatch");
            }
            REQUIRE(ok);
            add8_le(a.data());
        }
    }
}
