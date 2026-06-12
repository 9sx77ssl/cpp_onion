#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <span>

namespace onion::crypto {

// Tor's hs_ed25519_secret_key stores this *expanded* form (not a seed):
// 32-byte clamped scalar `a` followed by the 32-byte PRF prefix `RH`
// used for deterministic signature nonces.
struct ExpandedSecretKey {
    std::array<std::byte, 32> scalar{};
    std::array<std::byte, 32> prf_prefix{};

    ExpandedSecretKey() = default;
    ExpandedSecretKey(const ExpandedSecretKey&) = default;
    ExpandedSecretKey& operator=(const ExpandedSecretKey&) = default;
    ExpandedSecretKey(ExpandedSecretKey&& o) noexcept
        : scalar(o.scalar), prf_prefix(o.prf_prefix) { o.wipe(); }
    ExpandedSecretKey& operator=(ExpandedSecretKey&& o) noexcept {
        if (this != &o) {
            scalar = o.scalar;
            prf_prefix = o.prf_prefix;
            o.wipe();
        }
        return *this;
    }
    ~ExpandedSecretKey() { wipe(); }

    void wipe() noexcept;
};

// RFC 8032: SHA-512(seed), clamp first half -> scalar; second half -> RH.
[[nodiscard]] ExpandedSecretKey expand_seed(std::span<const std::byte, 32> seed);

// A = a*B via libsodium (no re-clamping; scalar is already in clamped form).
// nullopt on libsodium failure (degenerate scalar) — callers skip the candidate.
[[nodiscard]] std::optional<std::array<std::byte, 32>>
pubkey_from_scalar(std::span<const std::byte, 32> scalar);

// Fill with CSPRNG bytes (getrandom-backed via libsodium).
void random_bytes(std::span<std::byte> out);

}  // namespace onion::crypto
