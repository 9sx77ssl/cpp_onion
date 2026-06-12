#include <catch2/catch_test_macros.hpp>
#include "crypto/incremental.hpp"

#include <sodium.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <random>

TEST_CASE("incremental y-bytes match libsodium base_noclamp(a0+8i) for the whole batch") {
    using namespace onion::crypto;
    REQUIRE(sodium_init() >= 0);
    std::mt19937_64 rng(2024);

    for (int trial = 0; trial < 20; ++trial) {
        std::array<std::byte, 32> a0;
        for (auto& x : a0) x = std::byte(rng() & 0xff);
        a0[0] = std::byte(std::to_integer<unsigned>(a0[0]) & 0xf8);   // clamp low 3 bits
        a0[31] = std::byte((std::to_integer<unsigned>(a0[31]) & 0x7f) | 0x40);

        constexpr std::size_t N = 1024;
        IncrementalStepper stepper(a0);
        std::array<std::array<std::byte, 32>, N> ybatch;
        stepper.next_batch(ybatch);

        // reference scalar a = a0; check each i
        std::array<unsigned char, 32> a;
        std::memcpy(a.data(), a0.data(), 32);
        for (std::size_t i = 0; i < N; ++i) {
            std::array<unsigned char, 32> ref;
            REQUIRE(crypto_scalarmult_ed25519_base_noclamp(ref.data(), a.data()) == 0);
            // compare the leading 31 bytes (pure y; byte 31 differs by the sign bit)
            CHECK(std::memcmp(ybatch[i].data(), ref.data(), 31) == 0);
            // advance reference scalar by 8
            unsigned int carry = 8;
            for (int k = 0; k < 32 && carry; ++k) { unsigned s = a[k] + (carry & 0xff); a[k] = s & 0xff; carry = s >> 8; }
        }
    }
}

TEST_CASE("scalar_add_8i reconstructs a0 + 8*i") {
    using namespace onion::crypto;
    std::array<std::byte, 32> a0{};
    a0[0] = std::byte{0x10};
    auto s = scalar_add_8i(a0, 3);  // +24 -> 0x28
    CHECK(std::to_integer<unsigned>(s[0]) == 0x28);

    std::array<std::byte, 32> b{};
    b[0] = std::byte{0xf8};
    auto s2 = scalar_add_8i(b, 1);  // 0xf8 + 8 = 0x100 -> byte0=0, byte1=1
    CHECK(std::to_integer<unsigned>(s2[0]) == 0x00);
    CHECK(std::to_integer<unsigned>(s2[1]) == 0x01);
}
