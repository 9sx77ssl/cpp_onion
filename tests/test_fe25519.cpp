#include <catch2/catch_test_macros.hpp>
#include "crypto/fe25519.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string_view>

namespace {
std::array<std::byte, 32> hex32(std::string_view h) {
    auto nib = [](char c) { return (c >= 'a') ? unsigned(c - 'a' + 10) : unsigned(c - '0'); };
    std::array<std::byte, 32> o;
    for (std::size_t i = 0; i < 32; ++i) o[i] = std::byte((nib(h[2*i]) << 4) | nib(h[2*i+1]));
    return o;
}
std::array<std::byte, 32> to_bytes(const onion::crypto::Fe& f) {
    std::array<std::byte, 32> o;
    onion::crypto::fe_to_bytes(o, f);
    return o;
}
}

TEST_CASE("fe roundtrip from_bytes/to_bytes") {
    auto a = hex32("ddccbbaa0099887766554433221100998877665544332211efcdab9078563412");
    CHECK(to_bytes(onion::crypto::fe_from_bytes(a)) == a);
}

TEST_CASE("fe_mul matches Python big-int product") {
    using namespace onion::crypto;
    auto a = fe_from_bytes(hex32("ddccbbaa0099887766554433221100998877665544332211efcdab9078563412"));
    auto b = fe_from_bytes(hex32("01efcdab78563412908f7e6d5c4b3a291807f6e5d4c3b2a121436587a9cbed0f"));
    CHECK(to_bytes(fe_mul(a, b)) ==
          hex32("6c549ee33167ba4ccd73a41253186ddd4225ad6c5ee8d8b55b7a615ad92b0a22"));
}

TEST_CASE("fe_sq matches Python and equals fe_mul(a,a)") {
    using namespace onion::crypto;
    auto a = fe_from_bytes(hex32("ddccbbaa0099887766554433221100998877665544332211efcdab9078563412"));
    CHECK(to_bytes(fe_sq(a)) ==
          hex32("8688b02610872860d0f7eee66990690a0fe2c837fac937eb098e0d043b54e773"));
    CHECK(to_bytes(fe_sq(a)) == to_bytes(fe_mul(a, a)));
}

TEST_CASE("fe_invert matches Python and a*inv(a)==1") {
    using namespace onion::crypto;
    auto a = fe_from_bytes(hex32("ddccbbaa0099887766554433221100998877665544332211efcdab9078563412"));
    CHECK(to_bytes(fe_invert(a)) ==
          hex32("6c1316d8a841051ef542f904fb62baffdc6ee9c930b63bbcad91b918e0e39f66"));
    auto one = hex32("0100000000000000000000000000000000000000000000000000000000000000");
    CHECK(to_bytes(fe_mul(a, fe_invert(a))) == one);
}

TEST_CASE("fe add/sub are inverse and reduce correctly") {
    using namespace onion::crypto;
    std::mt19937_64 rng(7);
    for (int t = 0; t < 500; ++t) {
        std::array<std::byte, 32> ab, bb;
        for (auto& x : ab) x = std::byte(rng() & 0xff);
        for (auto& x : bb) x = std::byte(rng() & 0xff);
        ab[31] = std::byte(std::to_integer<unsigned>(ab[31]) & 0x7f);  // < 2^255
        bb[31] = std::byte(std::to_integer<unsigned>(bb[31]) & 0x7f);
        auto a = fe_from_bytes(ab), b = fe_from_bytes(bb);
        // (a+b)-b == a
        CHECK(to_bytes(fe_sub(fe_add(a, b), b)) == to_bytes(a));
    }
}
