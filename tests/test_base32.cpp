#include <catch2/catch_test_macros.hpp>
#include "core/base32.hpp"

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace {
std::span<const std::byte> as_bytes_sv(std::string_view s) {
    return {reinterpret_cast<const std::byte*>(s.data()), s.size()};
}
}

TEST_CASE("base32 encode RFC 4648 vectors (lowercase, unpadded)") {
    using onion::core::base32_encode;
    CHECK(base32_encode(as_bytes_sv("")) == "");
    CHECK(base32_encode(as_bytes_sv("f")) == "my");
    CHECK(base32_encode(as_bytes_sv("fo")) == "mzxq");
    CHECK(base32_encode(as_bytes_sv("foo")) == "mzxw6");
    CHECK(base32_encode(as_bytes_sv("foob")) == "mzxw6yq");
    CHECK(base32_encode(as_bytes_sv("fooba")) == "mzxw6ytb");
    CHECK(base32_encode(as_bytes_sv("foobar")) == "mzxw6ytboi");
}

TEST_CASE("base32 decode round-trips and rejects junk") {
    using onion::core::base32_decode;
    using onion::core::base32_encode;

    auto decoded = base32_decode("mzxw6ytboi");
    REQUIRE(decoded.has_value());
    CHECK(base32_encode(*decoded) == "mzxw6ytboi");
    CHECK(decoded->size() == 6);

    CHECK_FALSE(base32_decode("MZXW6").has_value());   // uppercase rejected (strict)
    CHECK_FALSE(base32_decode("mzx w").has_value());   // whitespace rejected
    CHECK_FALSE(base32_decode("mzxw1").has_value());   // '1' not in alphabet
    CHECK_FALSE(base32_decode("mz======").has_value()); // padding chars rejected

    CHECK_FALSE(base32_decode("a").has_value());        // length 1 mod 8 = 1
    CHECK_FALSE(base32_decode("aaa").has_value());      // length 3 mod 8 = 3
    CHECK_FALSE(base32_decode("aaaaaa").has_value());   // length 6 mod 8 = 6
}
