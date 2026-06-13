#pragma once

// AVX2 4-wide incremental stepper for Ed25519 y-byte extraction.
//
// IncrementalStepperX4 accepts 4 independent base scalars (one per lane) and
// walks each lane's point sequence  A0, A0+8B, A0+16B, ...  in projective
// extended coordinates, batching Z^-1 recovery via Montgomery's trick applied
// to Fe4 (all 4 lanes in lockstep).
//
// Output layout: out[lane][k] is the 32-byte y-encoding for lane `lane`, step
// `consumed+k` (before the call advances consumed).  Byte 31's sign bit is NOT
// set (irrelevant for prefix matching; the verifier re-derives the full key).

#include "crypto/ge25519x4.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace onion::crypto {

class IncrementalStepperX4 {
public:
    // 4 independent base scalars, one per lane (each 32-byte LE, clamped).
    IncrementalStepperX4(std::span<const std::byte, 32> a0_0,
                         std::span<const std::byte, 32> a0_1,
                         std::span<const std::byte, 32> a0_2,
                         std::span<const std::byte, 32> a0_3);

    // Fills out[lane][k] (lane 0..3, k 0..n-1) with the y-bytes of that
    // lane's (consumed+k)-th point, then advances consumed by n.
    // Each out[lane] must be pre-sized to at least n elements.
    void next_batch(std::array<std::vector<std::array<std::byte, 32>>, 4>& out,
                    std::size_t n);

    [[nodiscard]] std::uint64_t consumed() const { return consumed_; }

private:
    GeP3x4           cur_;       // current 4-wide projective point
    GeCachedAffinex4 step8b_;    // 8B broadcast into all 4 lanes (constant)
    std::uint64_t    consumed_ = 0;

    // Reusable scratch — sized on first batch, reused thereafter (no hot-path alloc).
    std::vector<Fe4> ys_, zs_, prefix_;
};

}  // namespace onion::crypto
