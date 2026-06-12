#include "crypto/incremental.hpp"

namespace onion::crypto {

std::array<std::byte, 32> scalar_add_8i(std::span<const std::byte, 32> a0, std::uint64_t i) {
    std::array<std::byte, 32> s;
    for (std::size_t k = 0; k < 32; ++k) s[k] = a0[k];
    std::uint64_t carry = i * 8;
    for (std::size_t k = 0; k < 32 && carry; ++k) {
        std::uint64_t v = std::to_integer<std::uint64_t>(s[k]) + (carry & 0xff);
        s[k] = std::byte(v & 0xff);
        carry = (carry >> 8) + (v >> 8);
    }
    return s;
}

IncrementalStepper::IncrementalStepper(std::span<const std::byte, 32> a0) {
    cur_ = ge_scalarmult_base(a0);
    std::array<std::byte, 32> eight{};
    eight[0] = std::byte{8};
    step8b_ = ge_to_cached(ge_scalarmult_base(eight));
}

void IncrementalStepper::next_batch_impl(std::array<std::byte, 32>* out, std::size_t n) {
    // Keep only Y and Z per candidate, then Montgomery batch-invert the Z values.
    // The scratch is reused across batches (resize is a no-op once sized).
    if (ys_.size() < n) {
        ys_.resize(n);
        z_.resize(n);
        prefix_.resize(n);
    }
    for (std::size_t i = 0; i < n; ++i) {
        ys_[i] = cur_.Y;
        z_[i] = cur_.Z;
        cur_ = ge_add(cur_, step8b_);
    }
    // prefix products: prefix_[i] = z_[0]*...*z_[i]
    prefix_[0] = z_[0];
    for (std::size_t i = 1; i < n; ++i) prefix_[i] = fe_mul(prefix_[i - 1], z_[i]);
    Fe inv = fe_invert(prefix_[n - 1]);  // 1 / (z_[0]*...*z_[n-1])
    // back-substitute: zinv[i] = inv * prefix_[i-1]; inv *= z_[i]
    for (std::size_t i = n; i-- > 0;) {
        Fe zinv = (i == 0) ? inv : fe_mul(inv, prefix_[i - 1]);
        Fe y = fe_mul(ys_[i], zinv);
        fe_to_bytes(out[i], y);
        if (i != 0) inv = fe_mul(inv, z_[i]);
    }
    consumed_ += n;
}

}  // namespace onion::crypto
