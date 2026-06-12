#include <catch2/catch_test_macros.hpp>
#include "core/matcher.hpp"
#include "core/onion_address.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string>

TEST_CASE("compile_prefix produces exact bytes and mask") {
    using onion::core::compile_prefix;
    // "abc" -> 5-bit groups 00000 00001 00010 -> bitstream 00000000 0100010x
    const auto pat = compile_prefix("abc");
    REQUIRE(pat.has_value());
    CHECK(pat->nbytes == 2);
    CHECK(pat->bytes[0] == 0x00);
    CHECK(pat->bytes[1] == 0x44);
    CHECK(pat->mask[0] == 0xff);
    CHECK(pat->mask[1] == 0xfe);
    CHECK(pat->prefix == "abc");
}

TEST_CASE("compile_prefix validates input") {
    using onion::core::compile_prefix;
    CHECK_FALSE(compile_prefix("").has_value());
    CHECK_FALSE(compile_prefix("ab1").has_value());   // '1' not in base32 alphabet
    CHECK_FALSE(compile_prefix("ab0").has_value());   // '0' not in alphabet
    CHECK_FALSE(compile_prefix("ABC").has_value());   // uppercase rejected (strict)
    CHECK_FALSE(compile_prefix(std::string(50, 'a')).has_value());  // > 49 chars
    CHECK(compile_prefix(std::string(49, 'a')).has_value());
}

TEST_CASE("matches agrees with naive encode-then-compare reference") {
    using namespace onion::core;
    std::mt19937_64 rng(99);  // deterministic test, NOT key material
    int positives = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        std::array<std::byte, 32> pk;
        for (auto& b : pk) b = std::byte(rng() & 0xff);
        const std::string addr{onion_address_from_pubkey(pk).view()};

        // True prefixes of this address must match, for every length 1..49
        const std::size_t len = 1 + trial % 49;
        const auto pat = compile_prefix(addr.substr(0, len));
        REQUIRE(pat.has_value());
        CHECK(matches(*pat, pk));

        // A single mutated final char must not match
        std::string bad = addr.substr(0, len);
        bad.back() = (bad.back() == 'a') ? 'b' : 'a';
        const auto bad_pat = compile_prefix(bad);
        REQUIRE(bad_pat.has_value());
        if (matches(*bad_pat, pk)) ++positives;
    }
    CHECK(positives == 0);
}
