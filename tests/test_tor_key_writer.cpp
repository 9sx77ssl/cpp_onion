#include <catch2/catch_test_macros.hpp>
#include "io/tor_key_writer.hpp"
#include "io/verifier.hpp"
#include "core/matcher.hpp"
#include "crypto/keys.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

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

fs::path make_temp_dir() {
    std::string tmpl = (fs::temp_directory_path() / "onion_test_XXXXXX").string();
    REQUIRE(::mkdtemp(tmpl.data()) != nullptr);
    return fs::path(tmpl);
}

std::vector<std::byte> read_all(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    REQUIRE(in.good());
    std::vector<char> raw{std::istreambuf_iterator<char>(in),
                          std::istreambuf_iterator<char>()};
    return {reinterpret_cast<std::byte*>(raw.data()),
            reinterpret_cast<std::byte*>(raw.data()) + raw.size()};
}
}

TEST_CASE("writer emits Tor-format files with exact bytes and 0700/0600 permissions") {
    using namespace onion;

    const auto seed = from_hex32(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
    engine::MatchCandidate cand;
    cand.secret = crypto::expand_seed(seed);
    cand.claimed_pubkey = *crypto::pubkey_from_scalar(cand.secret.scalar);
    cand.pattern_index = 0;
    std::vector patterns{*core::compile_prefix("2")};
    const auto verified = io::verify(cand, patterns);
    REQUIRE(verified.has_value());

    const auto outdir = make_temp_dir();
    const auto dir = io::write_tor_keys(*verified, outdir);
    REQUIRE(dir.has_value());

    constexpr std::string_view kAddr56 =
        "25njqamcweflpvkl73j4szahhihoc4xt3ktcgjnpaingr5yhkenl5sid";
    CHECK(dir->filename() == fs::path(kAddr56));
    CHECK(fs::status(*dir).permissions() ==
          (fs::perms::owner_read | fs::perms::owner_write | fs::perms::owner_exec));

    // hostname
    {
        std::ifstream host(*dir / "hostname");
        std::string line;
        std::getline(host, line);
        CHECK(line == std::string(kAddr56) + ".onion");
    }

    // secret key file: 32-byte tag header + scalar + PRF prefix
    {
        const auto bytes = read_all(*dir / "hs_ed25519_secret_key");
        REQUIRE(bytes.size() == 96);
        constexpr std::string_view tag = "== ed25519v1-secret: type0 ==";
        CHECK(std::memcmp(bytes.data(), tag.data(), tag.size()) == 0);
        for (std::size_t i = tag.size(); i < 32; ++i)
            CHECK(bytes[i] == std::byte{0});
        CHECK(std::memcmp(bytes.data() + 32, verified->secret.scalar.data(), 32) == 0);
        CHECK(std::memcmp(bytes.data() + 64, verified->secret.prf_prefix.data(), 32) == 0);
        CHECK(fs::status(*dir / "hs_ed25519_secret_key").permissions() ==
              (fs::perms::owner_read | fs::perms::owner_write));
    }

    // public key file: 32-byte tag header + pubkey
    {
        const auto bytes = read_all(*dir / "hs_ed25519_public_key");
        REQUIRE(bytes.size() == 64);
        constexpr std::string_view tag = "== ed25519v1-public: type0 ==";
        CHECK(std::memcmp(bytes.data(), tag.data(), tag.size()) == 0);
        CHECK(std::memcmp(bytes.data() + 32, verified->pubkey.data(), 32) == 0);
    }

    fs::remove_all(outdir);
}

TEST_CASE("writer refuses to overwrite an existing result") {
    using namespace onion;
    const auto seed = from_hex32(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
    engine::MatchCandidate cand;
    cand.secret = crypto::expand_seed(seed);
    cand.claimed_pubkey = *crypto::pubkey_from_scalar(cand.secret.scalar);
    cand.pattern_index = 0;
    std::vector patterns{*core::compile_prefix("2")};
    const auto verified = io::verify(cand, patterns);
    REQUIRE(verified.has_value());

    const auto outdir = make_temp_dir();
    REQUIRE(io::write_tor_keys(*verified, outdir).has_value());
    CHECK_FALSE(io::write_tor_keys(*verified, outdir).has_value());  // O_EXCL
    fs::remove_all(outdir);
}
