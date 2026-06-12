#include <catch2/catch_test_macros.hpp>
#include "crypto/sha3.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace {
std::string to_hex(std::span<const std::byte> bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    for (std::byte b : bytes) {
        out.push_back(digits[std::to_integer<unsigned>(b) >> 4]);
        out.push_back(digits[std::to_integer<unsigned>(b) & 0xf]);
    }
    return out;
}
std::span<const std::byte> as_bytes_sv(std::string_view s) {
    return {reinterpret_cast<const std::byte*>(s.data()), s.size()};
}
} // namespace

TEST_CASE("SHA3-256 NIST known answers") {
    using onion::crypto::Sha3_256;
    CHECK(to_hex(Sha3_256::hash(as_bytes_sv(""))) ==
          "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a");
    CHECK(to_hex(Sha3_256::hash(as_bytes_sv("abc"))) ==
          "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532");
}

TEST_CASE("SHA3-256 streaming equals one-shot") {
    using onion::crypto::Sha3_256;
    // 200 bytes crosses the 136-byte rate boundary
    std::string msg(200, 'x');
    Sha3_256 h;
    h.update(as_bytes_sv(std::string_view{msg}.substr(0, 7)));
    h.update(as_bytes_sv(std::string_view{msg}.substr(7, 150)));
    h.update(as_bytes_sv(std::string_view{msg}.substr(157)));
    CHECK(h.finalize() == Sha3_256::hash(as_bytes_sv(msg)));
}
