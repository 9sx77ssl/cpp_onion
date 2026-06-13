#include <catch2/catch_test_macros.hpp>

#include "crypto/ge25519.hpp"
#include "crypto/ge25519x4.hpp"

#include <array>
#include <cstddef>
#include <random>

namespace {

using onion::crypto::Fe;
using onion::crypto::GeCachedAffine;
using onion::crypto::GeCachedAffinex4;
using onion::crypto::GeP3;
using onion::crypto::GeP3x4;

using Bytes32 = std::array<std::byte, 32>;

// Generate a random 32-byte scalar (clamped to < 2^255 so ge_scalarmult_base
// is happy with it — we don't need full clamping for this test).
Bytes32 rand_scalar(std::mt19937_64& rng) {
    Bytes32 s;
    for (auto& b : s) b = std::byte(rng() & 0xff);
    s[31] = std::byte(std::to_integer<unsigned>(s[31]) & 0x7f);  // top bit clear
    return s;
}

}  // namespace

// ─── Cross-validation: ge_madd_x4 must match scalar ge_madd on all 4 lanes ───

TEST_CASE("ge_madd_x4 cross-validates against scalar ge_madd (500 trials)") {
    using namespace onion::crypto;
    std::mt19937_64 rng(0xDEADBEEF);

    for (int trial = 0; trial < 500; ++trial) {
        // Build 4 random valid points via scalar multiplication of the basepoint.
        Bytes32 s0 = rand_scalar(rng);
        Bytes32 s1 = rand_scalar(rng);
        Bytes32 s2 = rand_scalar(rng);
        Bytes32 s3 = rand_scalar(rng);
        GeP3 P0 = ge_scalarmult_base(s0);
        GeP3 P1 = ge_scalarmult_base(s1);
        GeP3 P2 = ge_scalarmult_base(s2);
        GeP3 P3 = ge_scalarmult_base(s3);

        // Build a random affine addend.
        Bytes32 sq = rand_scalar(rng);
        GeP3 Pq   = ge_scalarmult_base(sq);
        GeCachedAffine q = ge_to_cached_affine(Pq);

        // Vectorized path.
        GeP3x4 packed = ge_p3x4_pack(P0, P1, P2, P3);
        GeCachedAffinex4 qx4 = ge_cached_affinex4_broadcast(q);
        GeP3x4 res = ge_madd_x4(packed, qx4);

        // Scalar reference for each lane.
        GeP3 r0 = ge_madd(P0, q);
        GeP3 r1 = ge_madd(P1, q);
        GeP3 r2 = ge_madd(P2, q);
        GeP3 r3 = ge_madd(P3, q);

        Bytes32 want0 = ge_encode(r0);
        Bytes32 want1 = ge_encode(r1);
        Bytes32 want2 = ge_encode(r2);
        Bytes32 want3 = ge_encode(r3);

        Bytes32 got0 = ge_encode(ge_p3x4_unpack(res, 0));
        Bytes32 got1 = ge_encode(ge_p3x4_unpack(res, 1));
        Bytes32 got2 = ge_encode(ge_p3x4_unpack(res, 2));
        Bytes32 got3 = ge_encode(ge_p3x4_unpack(res, 3));

        INFO("trial = " << trial);
        CHECK(got0 == want0);
        CHECK(got1 == want1);
        CHECK(got2 == want2);
        CHECK(got3 == want3);
    }
}

// ─── Pack/unpack round-trip: pack then unpack must recover the original point ─

TEST_CASE("ge_p3x4_pack/unpack round-trips for all 4 lanes (200 trials)") {
    using namespace onion::crypto;
    std::mt19937_64 rng(0xC0FFEE42);

    for (int trial = 0; trial < 200; ++trial) {
        Bytes32 s0 = rand_scalar(rng);
        Bytes32 s1 = rand_scalar(rng);
        Bytes32 s2 = rand_scalar(rng);
        Bytes32 s3 = rand_scalar(rng);
        GeP3 P0 = ge_scalarmult_base(s0);
        GeP3 P1 = ge_scalarmult_base(s1);
        GeP3 P2 = ge_scalarmult_base(s2);
        GeP3 P3 = ge_scalarmult_base(s3);

        GeP3x4 packed = ge_p3x4_pack(P0, P1, P2, P3);

        INFO("trial = " << trial);
        CHECK(ge_encode(ge_p3x4_unpack(packed, 0)) == ge_encode(P0));
        CHECK(ge_encode(ge_p3x4_unpack(packed, 1)) == ge_encode(P1));
        CHECK(ge_encode(ge_p3x4_unpack(packed, 2)) == ge_encode(P2));
        CHECK(ge_encode(ge_p3x4_unpack(packed, 3)) == ge_encode(P3));
    }
}
