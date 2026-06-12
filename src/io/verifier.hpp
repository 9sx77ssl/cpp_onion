#pragma once

#include "core/matcher.hpp"
#include "core/onion_address.hpp"
#include "engine/engine.hpp"

#include <expected>
#include <span>

namespace onion::io {

enum class VerifyError {
    pubkey_mismatch,
    bad_pattern_index,
    pattern_mismatch,
    address_prefix_mismatch,
};

struct VerifiedResult {
    crypto::ExpandedSecretKey secret;
    std::array<std::byte, 32> pubkey{};
    core::OnionAddress address{};
};

// The firewall between engine arithmetic and user-visible output: re-derives
// the pubkey via libsodium and re-checks the match before anything is
// written (design doc §9). A failure here means an engine bug.
[[nodiscard]] std::expected<VerifiedResult, VerifyError>
verify(const engine::MatchCandidate& candidate,
       std::span<const core::CompiledPattern> patterns);

}  // namespace onion::io
