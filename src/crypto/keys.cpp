#include "crypto/keys.hpp"

#include <sodium.h>

#include <algorithm>
#include <cstdlib>

namespace onion::crypto {
namespace {

void ensure_sodium() {
    static const int rc = sodium_init();  // thread-safe magic static
    if (rc < 0) std::abort();             // no entropy source: unrecoverable
}

}  // namespace

void ExpandedSecretKey::wipe() noexcept {
    sodium_memzero(scalar.data(), scalar.size());
    sodium_memzero(prf_prefix.data(), prf_prefix.size());
}

ExpandedSecretKey expand_seed(std::span<const std::byte, 32> seed) {
    ensure_sodium();
    std::array<std::byte, 64> h{};
    crypto_hash_sha512(reinterpret_cast<unsigned char*>(h.data()),
                       reinterpret_cast<const unsigned char*>(seed.data()),
                       seed.size());
    h[0] &= std::byte{0xf8};
    h[31] &= std::byte{0x7f};
    h[31] |= std::byte{0x40};

    ExpandedSecretKey key;
    std::copy_n(h.begin(), 32, key.scalar.begin());
    std::copy_n(h.begin() + 32, 32, key.prf_prefix.begin());
    sodium_memzero(h.data(), h.size());
    return key;
}

std::optional<std::array<std::byte, 32>>
pubkey_from_scalar(std::span<const std::byte, 32> scalar) {
    ensure_sodium();
    std::array<std::byte, 32> pk;
    if (crypto_scalarmult_ed25519_base_noclamp(
            reinterpret_cast<unsigned char*>(pk.data()),
            reinterpret_cast<const unsigned char*>(scalar.data())) != 0)
        return std::nullopt;
    return pk;
}

void random_bytes(std::span<std::byte> out) {
    ensure_sodium();
    randombytes_buf(out.data(), out.size());
}

}  // namespace onion::crypto
