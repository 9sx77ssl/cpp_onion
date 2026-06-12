#include "io/verifier.hpp"

#include "crypto/keys.hpp"

namespace onion::io {

std::expected<VerifiedResult, VerifyError>
verify(const engine::MatchCandidate& candidate,
       std::span<const core::CompiledPattern> patterns) {
    const auto pk = crypto::pubkey_from_scalar(candidate.secret.scalar);
    if (!pk || *pk != candidate.claimed_pubkey)
        return std::unexpected(VerifyError::pubkey_mismatch);

    if (candidate.pattern_index >= patterns.size())
        return std::unexpected(VerifyError::bad_pattern_index);
    const auto& pattern = patterns[candidate.pattern_index];

    if (!core::matches(pattern, *pk))
        return std::unexpected(VerifyError::pattern_mismatch);

    const auto address = core::onion_address_from_pubkey(*pk);
    if (!address.view().starts_with(pattern.prefix))
        return std::unexpected(VerifyError::address_prefix_mismatch);

    return VerifiedResult{candidate.secret, *pk, address};
}

}  // namespace onion::io
