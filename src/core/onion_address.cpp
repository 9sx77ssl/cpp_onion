#include "core/onion_address.hpp"

#include "core/base32.hpp"
#include "crypto/sha3.hpp"

#include <algorithm>

namespace onion::core {
namespace {

constexpr std::string_view kChecksumTag = ".onion checksum";
constexpr std::byte kVersion{0x03};

std::array<std::byte, 2> checksum(std::span<const std::byte, 32> pubkey) {
    crypto::Sha3_256 h;
    h.update({reinterpret_cast<const std::byte*>(kChecksumTag.data()), kChecksumTag.size()});
    h.update(pubkey);
    h.update({&kVersion, 1});
    const auto digest = h.finalize();
    return {digest[0], digest[1]};
}

}  // namespace

OnionAddress onion_address_from_pubkey(std::span<const std::byte, 32> pubkey) {
    std::array<std::byte, 35> blob;
    std::ranges::copy(pubkey, blob.begin());
    const auto chk = checksum(pubkey);
    blob[32] = chk[0];
    blob[33] = chk[1];
    blob[34] = kVersion;

    const std::string encoded = base32_encode(blob);  // exactly 56 chars for 35 bytes
    OnionAddress out;
    std::ranges::copy(encoded, out.chars.begin());
    return out;
}

std::expected<std::array<std::byte, 32>, AddressError>
pubkey_from_onion_address(std::string_view address) {
    if (address.ends_with(".onion")) address.remove_suffix(6);
    if (address.size() != 56) return std::unexpected(AddressError::bad_length);

    const auto decoded = base32_decode(address);
    if (!decoded || decoded->size() != 35)
        return std::unexpected(AddressError::bad_base32);

    if ((*decoded)[34] != kVersion) return std::unexpected(AddressError::bad_version);

    std::array<std::byte, 32> pubkey;
    std::copy_n(decoded->begin(), 32, pubkey.begin());
    const auto chk = checksum(pubkey);
    if ((*decoded)[32] != chk[0] || (*decoded)[33] != chk[1])
        return std::unexpected(AddressError::bad_checksum);
    return pubkey;
}

}  // namespace onion::core
