#pragma once

// AVX2 4-wide extended-coordinates point operations on Ed25519.
//
// GeP3x4 holds four independent GeP3 points in struct-of-arrays form via Fe4
// (see fe25519x4.hpp). Lane k corresponds to candidate k in the 4-wide batch.
//
// ge_madd_x4 is the only hot operation: it performs mixed addition (affine q)
// in lockstep across all 4 lanes using the same formula as the scalar ge_madd.
// The scalar GeCachedAffine addend is broadcast into all 4 lanes of a
// GeCachedAffinex4 before the hot loop so we do a single broadcast per 8B step.
//
// Pack/unpack helpers go through canonical bytes (fe_to_bytes / fe4_load /
// fe4_store_y) and are intended for setup and test only, not the hot path.

#include "crypto/fe25519x4.hpp"
#include "crypto/ge25519.hpp"

namespace onion::crypto {

// 4-wide extended-coordinates point: (X:Y:Z:T) with XY=ZT.
struct GeP3x4 { Fe4 X, Y, Z, T; };

// 4-wide precomputed affine addend (Z normalised to 1 in each lane).
struct GeCachedAffinex4 { Fe4 YplusX, YminusX, T2d; };

// Mixed addition of 4 lanes: each lane computes p[k] + q[k] where q is affine.
// Identical formula to scalar ge_madd, operating on fe4_* ops.
[[nodiscard]] GeP3x4 ge_madd_x4(const GeP3x4& p, const GeCachedAffinex4& q);

// Pack four scalar GeP3 points into a GeP3x4 (lane k = pts[k]).
[[nodiscard]] GeP3x4 ge_p3x4_pack(const GeP3& l0, const GeP3& l1,
                                   const GeP3& l2, const GeP3& l3);

// Broadcast a single scalar GeCachedAffine into all 4 lanes of a GeCachedAffinex4.
[[nodiscard]] GeCachedAffinex4 ge_cached_affinex4_broadcast(const GeCachedAffine& q);

// Extract one lane back to a scalar GeP3 (for testing / y-extraction).
[[nodiscard]] GeP3 ge_p3x4_unpack(const GeP3x4& p, int lane);

}  // namespace onion::crypto
