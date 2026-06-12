#include <catch2/catch_test_macros.hpp>
#include "crypto/keys.hpp"
#include "core/onion_address.hpp"

#include <array>
#include <cstddef>
#include <string_view>

namespace {
std::array<std::byte, 32> from_hex32(std::string_view hex) {
    auto nib = [](char c) -> unsigned {
        return (c >= 'a') ? unsigned(c - 'a' + 10) : unsigned(c - '0');
    };
    std::array<std::byte, 32> out;
    for (std::size_t i = 0; i < 32; ++i)
        out[i] = std::byte((nib(hex[2 * i]) << 4) | nib(hex[2 * i + 1]));
    return out;
}
}

TEST_CASE("RFC 8032 TEST1: seed -> expanded -> pubkey -> address") {
    using namespace onion;

    const auto seed = from_hex32(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
    const auto expected_pk = from_hex32(
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");

    const auto key = crypto::expand_seed(seed);

    // clamping invariants: low 3 bits clear, bit 254 set, bit 255 clear
    CHECK((std::to_integer<unsigned>(key.scalar[0]) & 0x07) == 0);
    CHECK((std::to_integer<unsigned>(key.scalar[31]) & 0xc0) == 0x40);

    const auto pk = crypto::pubkey_from_scalar(key.scalar);
    REQUIRE(pk.has_value());
    CHECK(*pk == expected_pk);

    CHECK(core::onion_address_from_pubkey(*pk).to_string() ==
          "25njqamcweflpvkl73j4szahhihoc4xt3ktcgjnpaingr5yhkenl5sid.onion");
}

TEST_CASE("ExpandedSecretKey::wipe zeroizes") {
    using namespace onion;
    const auto seed = from_hex32(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
    auto key = crypto::expand_seed(seed);
    key.wipe();
    std::array<std::byte, 32> zero{};
    CHECK(key.scalar == zero);
    CHECK(key.prf_prefix == zero);
}
