#pragma once

#include "crypto/ge25519.hpp"

#include <array>
#include <cstddef>
#include <span>

namespace onion::crypto {

// a0 + 8*i as a 256-bit little-endian scalar (no reduction; i is small).
[[nodiscard]] std::array<std::byte, 32>
scalar_add_8i(std::span<const std::byte, 32> a0, std::uint64_t i);

// Walks A0=a0*B, A0+8B, A0+16B, ... in projective coordinates and recovers the
// little-endian affine-y encoding of each point a batch at a time (one field
// inversion per batch via Montgomery's trick). Byte 31's sign bit is NOT set
// (irrelevant to prefixes <= 49 chars; the verifier re-derives the full key).
class IncrementalStepper {
public:
    explicit IncrementalStepper(std::span<const std::byte, 32> a0);

    // Fills out[k] with the y-bytes of the (consumed + k)-th point, then advances.
    template <std::size_t N>
    void next_batch(std::array<std::array<std::byte, 32>, N>& out) {
        next_batch_impl(out.data(), N);
    }
    // Runtime-sized overload (used by the engine, which sizes batches at construction).
    void next_batch(std::span<std::array<std::byte, 32>> out) {
        next_batch_impl(out.data(), out.size());
    }

    [[nodiscard]] std::uint64_t consumed() const { return consumed_; }

private:
    void next_batch_impl(std::array<std::byte, 32>* out, std::size_t n);

    GeP3 cur_;
    GeCached step8b_;
    std::uint64_t consumed_ = 0;
};

}  // namespace onion::crypto
