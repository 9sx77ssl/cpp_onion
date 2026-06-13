#include <catch2/catch_test_macros.hpp>
#include "crypto/incremental_x4.hpp"
#include "crypto/incremental.hpp"   // scalar_add_8i

#include <sodium.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <random>
#include <vector>

// ── helpers ───────────────────────────────────────────────────────────────────

// Advance a libsodium-style 32-byte LE scalar by 8 in place.
static void advance_by_8(std::array<unsigned char, 32>& a) {
    unsigned carry = 8;
    for (int k = 0; k < 32 && carry; ++k) {
        unsigned s = a[k] + (carry & 0xff);
        a[k] = static_cast<unsigned char>(s & 0xff);
        carry = s >> 8;
    }
}

// ── master cross-check test ───────────────────────────────────────────────────

TEST_CASE("IncrementalStepperX4 y-bytes match libsodium on all 4 lanes, 10 trials × 1024 steps") {
    using namespace onion::crypto;
    REQUIRE(sodium_init() >= 0);

    std::mt19937_64 rng(20240613ULL);

    for (int trial = 0; trial < 10; ++trial) {
        // Generate 4 independent clamped scalars.
        std::array<std::array<std::byte, 32>, 4> a0;
        for (auto& lane_a0 : a0) {
            for (auto& x : lane_a0) x = std::byte(rng() & 0xff);
            lane_a0[0]  = std::byte(std::to_integer<unsigned>(lane_a0[0])  & 0xf8);
            lane_a0[31] = std::byte((std::to_integer<unsigned>(lane_a0[31]) & 0x7f) | 0x40);
        }

        constexpr std::size_t N = 1024;

        IncrementalStepperX4 stepper(a0[0], a0[1], a0[2], a0[3]);
        REQUIRE(stepper.consumed() == 0);

        std::array<std::vector<std::array<std::byte, 32>>, 4> out;
        for (auto& v : out) v.resize(N);
        stepper.next_batch(out, N);
        REQUIRE(stepper.consumed() == N);

        // Verify each lane against libsodium.
        for (int L = 0; L < 4; ++L) {
            std::array<unsigned char, 32> a;
            std::memcpy(a.data(), a0[L].data(), 32);

            for (std::size_t i = 0; i < N; ++i) {
                std::array<unsigned char, 32> ref;
                REQUIRE(crypto_scalarmult_ed25519_base_noclamp(ref.data(), a.data()) == 0);

                // Compare leading 31 bytes (pure y; byte 31 differs by sign bit).
                bool ok = (std::memcmp(out[L][i].data(), ref.data(), 31) == 0);
                if (!ok) {
                    FAIL("Trial " << trial << " lane " << L << " step " << i
                         << ": stepper byte[0]=" << std::to_integer<unsigned>(out[L][i][0])
                         << " ref byte[0]=" << static_cast<unsigned>(ref[0]));
                }
                advance_by_8(a);
            }
        }
    }
}

TEST_CASE("IncrementalStepperX4 consumed counter increments correctly across multiple batches") {
    using namespace onion::crypto;
    REQUIRE(sodium_init() >= 0);

    std::mt19937_64 rng(20240614ULL);
    std::array<std::array<std::byte, 32>, 4> a0;
    for (auto& lane_a0 : a0) {
        for (auto& x : lane_a0) x = std::byte(rng() & 0xff);
        lane_a0[0]  = std::byte(std::to_integer<unsigned>(lane_a0[0])  & 0xf8);
        lane_a0[31] = std::byte((std::to_integer<unsigned>(lane_a0[31]) & 0x7f) | 0x40);
    }

    IncrementalStepperX4 stepper(a0[0], a0[1], a0[2], a0[3]);
    REQUIRE(stepper.consumed() == 0);

    std::array<std::vector<std::array<std::byte, 32>>, 4> out;
    for (auto& v : out) v.resize(16);

    stepper.next_batch(out, 16);
    REQUIRE(stepper.consumed() == 16);

    stepper.next_batch(out, 16);
    REQUIRE(stepper.consumed() == 32);
}
