#include <catch2/catch_test_macros.hpp>
#include "core/onion_address.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <string_view>

namespace {
std::array<std::byte, 32> make_pubkey(unsigned char fill) {
    std::array<std::byte, 32> pk;
    pk.fill(std::byte{fill});
    return pk;
}
}

TEST_CASE("onion address known answers") {
    using onion::core::onion_address_from_pubkey;

    std::array<std::byte, 32> seq;
    for (std::size_t i = 0; i < 32; ++i) seq[i] = std::byte(i);
    CHECK(onion_address_from_pubkey(seq).to_string() ==
          "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3kead.onion");

    CHECK(onion_address_from_pubkey(make_pubkey(0x42)).to_string() ==
          "ijbeeqscijbeeqscijbeeqscijbeeqscijbeeqscijbeeqscijbezhid.onion");
}

TEST_CASE("all v3 addresses are 56 chars and end in 'd'") {
    using onion::core::onion_address_from_pubkey;
    std::mt19937_64 rng(12345);  // deterministic test, NOT key material
    for (int trial = 0; trial < 100; ++trial) {
        std::array<std::byte, 32> pk;
        for (auto& b : pk) b = std::byte(rng() & 0xff);
        const auto addr = onion_address_from_pubkey(pk);
        CHECK(addr.chars.size() == 56);
        CHECK(addr.chars[55] == 'd');  // version byte 0x03 -> final base32 char 'd'
    }
}

TEST_CASE("address decode recovers pubkey and validates checksum") {
    using onion::core::onion_address_from_pubkey;
    using onion::core::pubkey_from_onion_address;

    const auto pk = make_pubkey(0x42);
    const auto addr = onion_address_from_pubkey(pk).to_string();

    auto recovered = pubkey_from_onion_address(addr);
    REQUIRE(recovered.has_value());
    CHECK(*recovered == pk);

    // Corrupt one address character -> checksum failure
    auto bad = addr;
    bad[3] = (bad[3] == 'a') ? 'b' : 'a';
    auto corrupted = pubkey_from_onion_address(bad);
    CHECK_FALSE(corrupted.has_value());
    CHECK(corrupted.error() == onion::core::AddressError::bad_checksum);

    using onion::core::AddressError;
    // bad_length: not 56 chars
    {
        auto r = pubkey_from_onion_address(std::string_view{addr}.substr(0, 55));
        REQUIRE_FALSE(r.has_value());
        CHECK(r.error() == AddressError::bad_length);
    }
    // bad_base32: 56 chars but contains a non-alphabet character ('1')
    {
        auto bad32 = addr.substr(0, 56);
        bad32[0] = '1';
        auto r = pubkey_from_onion_address(bad32);
        REQUIRE_FALSE(r.has_value());
        CHECK(r.error() == AddressError::bad_base32);
    }
}
