#include <catch2/catch_test_macros.hpp>
#include "io/verifier.hpp"
#include "core/matcher.hpp"
#include "crypto/keys.hpp"

#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

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

onion::engine::MatchCandidate test1_candidate() {
    const auto seed = from_hex32(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
    onion::engine::MatchCandidate c;
    c.secret = onion::crypto::expand_seed(seed);
    c.claimed_pubkey = *onion::crypto::pubkey_from_scalar(c.secret.scalar);
    c.pattern_index = 0;
    return c;
}
}

TEST_CASE("verifier accepts a genuine candidate") {
    using namespace onion;
    // TEST1's address starts with '2'
    std::vector patterns{*core::compile_prefix("2")};
    const auto result = io::verify(test1_candidate(), patterns);
    REQUIRE(result.has_value());
    CHECK(result->address.to_string() ==
          "25njqamcweflpvkl73j4szahhihoc4xt3ktcgjnpaingr5yhkenl5sid.onion");
}

TEST_CASE("verifier rejects a tampered pubkey") {
    using namespace onion;
    std::vector patterns{*core::compile_prefix("2")};
    auto cand = test1_candidate();
    cand.claimed_pubkey[5] ^= std::byte{0x01};
    const auto result = io::verify(cand, patterns);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error() == io::VerifyError::pubkey_mismatch);
}

TEST_CASE("verifier rejects a candidate that does not match its pattern") {
    using namespace onion;
    std::vector patterns{*core::compile_prefix("zz")};  // TEST1 addr starts "25"
    const auto result = io::verify(test1_candidate(), patterns);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error() == io::VerifyError::pattern_mismatch);
}
