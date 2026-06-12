#pragma once

#include <array>
#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <string_view>

namespace onion::core {

enum class AddressError { bad_length, bad_base32, bad_version, bad_checksum };

struct OnionAddress {
    std::array<char, 56> chars;

    [[nodiscard]] std::string to_string() const {
        return std::string(chars.data(), chars.size()) + ".onion";
    }
    // The 56-char base32 body, without the ".onion" suffix.
    [[nodiscard]] std::string_view view() const {
        return {chars.data(), chars.size()};
    }
};

// rend-spec-v3: base32(PUBKEY || SHA3-256(".onion checksum" || PUBKEY || 0x03)[0..2) || 0x03)
[[nodiscard]] OnionAddress
onion_address_from_pubkey(std::span<const std::byte, 32> pubkey);

// Inverse for verification/tests: accepts with or without the ".onion" suffix,
// validates length, base32, version byte, and checksum.
[[nodiscard]] std::expected<std::array<std::byte, 32>, AddressError>
pubkey_from_onion_address(std::string_view address);

}  // namespace onion::core
