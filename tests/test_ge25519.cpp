#include <catch2/catch_test_macros.hpp>
#include "crypto/ge25519.hpp"

#include <sodium.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <random>
#include <string_view>

namespace {
std::array<std::byte, 32> hex32(std::string_view h) {
    auto nib = [](char c) { return (c >= 'a') ? unsigned(c - 'a' + 10) : unsigned(c - '0'); };
    std::array<std::byte, 32> o;
    for (std::size_t i = 0; i < 32; ++i) o[i] = std::byte((nib(h[2*i]) << 4) | nib(h[2*i+1]));
    return o;
}
// Full 32-byte encoding of a P3 point (y little-endian + x sign bit), for tests only.
std::array<std::byte, 32> encode(const onion::crypto::GeP3& p) {
    return onion::crypto::ge_encode(p);
}
}

TEST_CASE("ge scalarmult_base matches the known a0*B vector") {
    using namespace onion::crypto;
    auto a0 = hex32("400000000000000000000000000000000000000000000000107e2d9c5ebf4a00");
    CHECK(encode(ge_scalarmult_base(a0)) ==
          hex32("494759ec2b42faec2b989685783762a15d87fb11dd2012f34765676cf96728d0"));
}

TEST_CASE("ge scalarmult_base(8) matches the 8B vector and equals 3 doublings of B") {
    using namespace onion::crypto;
    std::array<std::byte, 32> eight{};
    eight[0] = std::byte{8};
    CHECK(encode(ge_scalarmult_base(eight)) ==
          hex32("b4b937fca95b2f1e93e41e62fc3c78818ff38a66096fad6e7973e5c90006d321"));

    GeP3 b = ge_basepoint();
    GeP3 b8 = ge_double(ge_double(ge_double(b)));
    CHECK(encode(b8) == encode(ge_scalarmult_base(eight)));
}

TEST_CASE("ge scalarmult_base matches libsodium base_noclamp on random scalars") {
    using namespace onion::crypto;
    REQUIRE(sodium_init() >= 0);
    std::mt19937_64 rng(123);
    for (int t = 0; t < 200; ++t) {
        std::array<std::byte, 32> s;
        for (auto& x : s) x = std::byte(rng() & 0xff);
        s[31] = std::byte(std::to_integer<unsigned>(s[31]) & 0x7f);  // < 2^255

        std::array<unsigned char, 32> ref;
        REQUIRE(crypto_scalarmult_ed25519_base_noclamp(
                    ref.data(), reinterpret_cast<const unsigned char*>(s.data())) == 0);

        auto mine = encode(ge_scalarmult_base(s));
        // compare the full 32-byte encoding (y + sign)
        CHECK(std::memcmp(mine.data(), ref.data(), 32) == 0);
    }
}

TEST_CASE("ge_add via cached equals encode of point sum (P + 8B)") {
    using namespace onion::crypto;
    REQUIRE(sodium_init() >= 0);
    std::array<std::byte, 32> a0 = hex32("400000000000000000000000000000000000000000000000107e2d9c5ebf4a00");
    std::array<std::byte, 32> eight{}; eight[0] = std::byte{8};

    GeP3 A = ge_scalarmult_base(a0);
    GeCached eightB = ge_to_cached(ge_scalarmult_base(eight));
    GeP3 sum = ge_add(A, eightB);

    // reference: (a0 + 8) * B
    std::array<unsigned char, 32> a0p8;
    std::memcpy(a0p8.data(), a0.data(), 32);
    unsigned int carry = 8;
    for (int i = 0; i < 32 && carry; ++i) { unsigned s = a0p8[i] + (carry & 0xff); a0p8[i] = s & 0xff; carry = s >> 8; }
    std::array<unsigned char, 32> ref;
    REQUIRE(crypto_scalarmult_ed25519_base_noclamp(ref.data(), a0p8.data()) == 0);
    CHECK(std::memcmp(encode(sum).data(), ref.data(), 32) == 0);
}
