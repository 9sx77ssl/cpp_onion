#include "crypto/fe25519.hpp"

namespace onion::crypto {
namespace {

using u64 = std::uint64_t;
__extension__ typedef unsigned __int128 u128;
constexpr u64 M = (u64{1} << 51) - 1;  // reduce_mask_51

u64 load64(std::span<const std::byte, 32> s, std::size_t off) {
    u64 r = 0;
    for (int i = 7; i >= 0; --i) r = (r << 8) | std::to_integer<u64>(s[off + std::size_t(i)]);
    return r;
}

}  // namespace

Fe fe_from_bytes(std::span<const std::byte, 32> s) {
    const u64 x0 = load64(s, 0), x1 = load64(s, 8), x2 = load64(s, 16), x3 = load64(s, 24);
    Fe h;
    h.v[0] = x0 & M;
    h.v[1] = ((x0 >> 51) | (x1 << 13)) & M;
    h.v[2] = ((x1 >> 38) | (x2 << 26)) & M;
    h.v[3] = ((x2 >> 25) | (x3 << 39)) & M;
    h.v[4] = (x3 >> 12) & M;  // ignores bit 255 (the sign bit)
    return h;
}

void fe_to_bytes(std::span<std::byte, 32> out, const Fe& in) {
    u64 t[5] = {in.v[0], in.v[1], in.v[2], in.v[3], in.v[4]};

    auto carry = [&] {
        t[1] += t[0] >> 51; t[0] &= M;
        t[2] += t[1] >> 51; t[1] &= M;
        t[3] += t[2] >> 51; t[2] &= M;
        t[4] += t[3] >> 51; t[3] &= M;
    };
    auto carry_full = [&] { carry(); t[0] += 19 * (t[4] >> 51); t[4] &= M; };

    carry_full();
    carry_full();
    t[0] += 19;
    carry_full();
    t[0] += (M + 1) - 19;
    t[1] += M; t[2] += M; t[3] += M; t[4] += M;
    carry();
    t[4] &= M;

    u64 f;
    std::size_t o = 0;
    auto write51 = [&](int n, int shift) {
        f = (t[n] >> shift) | (t[n + 1] << (51 - shift));
        for (int i = 0; i < 8; ++i, f >>= 8) out[o++] = std::byte(f & 0xff);
    };
    write51(0, 0);
    write51(1, 13);
    write51(2, 26);
    write51(3, 39);
}

Fe fe_add(const Fe& a, const Fe& b) {
    return {{a.v[0] + b.v[0], a.v[1] + b.v[1], a.v[2] + b.v[2], a.v[3] + b.v[3], a.v[4] + b.v[4]}};
}

Fe fe_sub(const Fe& a, const Fe& b) {
    // bias by 8p so the result stays non-negative, then carry-reduce so the
    // output limbs are <= M+epsilon. The reduction is essential: without it a
    // fe_sub result has limbs ~2^54 and, in the worst case (a bias-form zero
    // such as fe_sub(1,1)), overflows fe_mul's `o0 += c*19` u64 step — which
    // silently drifts the identity point over repeated doublings.
    constexpr u64 two54m152 = (u64{1} << 54) - 152;  // 8*(2^51-19)
    constexpr u64 two54m8 = (u64{1} << 54) - 8;      // 8*(2^51-1)
    u64 t0 = a.v[0] + two54m152 - b.v[0];
    u64 t1 = a.v[1] + two54m8 - b.v[1];
    u64 t2 = a.v[2] + two54m8 - b.v[2];
    u64 t3 = a.v[3] + two54m8 - b.v[3];
    u64 t4 = a.v[4] + two54m8 - b.v[4];
    t1 += t0 >> 51; t0 &= M;
    t2 += t1 >> 51; t1 &= M;
    t3 += t2 >> 51; t2 &= M;
    t4 += t3 >> 51; t3 &= M;
    t0 += 19 * (t4 >> 51); t4 &= M;
    t1 += t0 >> 51; t0 &= M;
    Fe r = {{t0, t1, t2, t3, t4}};
    return r;
}

Fe fe_mul(const Fe& a, const Fe& b) {
    u64 r0 = a.v[0], r1 = a.v[1], r2 = a.v[2], r3 = a.v[3], r4 = a.v[4];
    const u64 s0 = b.v[0], s1 = b.v[1], s2 = b.v[2], s3 = b.v[3], s4 = b.v[4];

    u128 t0 = (u128)r0 * s0;
    u128 t1 = (u128)r0 * s1 + (u128)r1 * s0;
    u128 t2 = (u128)r0 * s2 + (u128)r2 * s0 + (u128)r1 * s1;
    u128 t3 = (u128)r0 * s3 + (u128)r3 * s0 + (u128)r1 * s2 + (u128)r2 * s1;
    u128 t4 = (u128)r0 * s4 + (u128)r4 * s0 + (u128)r3 * s1 + (u128)r1 * s3 + (u128)r2 * s2;

    r4 *= 19; r1 *= 19; r2 *= 19; r3 *= 19;

    t0 += (u128)r4 * s1 + (u128)r1 * s4 + (u128)r2 * s3 + (u128)r3 * s2;
    t1 += (u128)r4 * s2 + (u128)r2 * s4 + (u128)r3 * s3;
    t2 += (u128)r4 * s3 + (u128)r3 * s4;
    t3 += (u128)r4 * s4;

    u64 c;
    u64 o0, o1, o2, o3, o4;
    o0 = (u64)t0 & M; c = (u64)(t0 >> 51);
    t1 += c; o1 = (u64)t1 & M; c = (u64)(t1 >> 51);
    t2 += c; o2 = (u64)t2 & M; c = (u64)(t2 >> 51);
    t3 += c; o3 = (u64)t3 & M; c = (u64)(t3 >> 51);
    t4 += c; o4 = (u64)t4 & M; c = (u64)(t4 >> 51);
    o0 += c * 19; c = o0 >> 51; o0 &= M;
    o1 += c;      c = o1 >> 51; o1 &= M;
    o2 += c;
    return {{o0, o1, o2, o3, o4}};
}

Fe fe_sq(const Fe& a) {
    const u64 r0 = a.v[0], r1 = a.v[1], r2 = a.v[2], r3 = a.v[3], r4 = a.v[4];
    const u64 d0 = r0 * 2, d1 = r1 * 2, d2 = r2 * 2 * 19, d419 = r4 * 19, d4 = d419 * 2;

    u128 t0 = (u128)r0 * r0 + (u128)d4 * r1 + (u128)d2 * r3;
    u128 t1 = (u128)d0 * r1 + (u128)d4 * r2 + (u128)r3 * (r3 * 19);
    u128 t2 = (u128)d0 * r2 + (u128)r1 * r1 + (u128)d4 * r3;
    u128 t3 = (u128)d0 * r3 + (u128)d1 * r2 + (u128)r4 * d419;
    u128 t4 = (u128)d0 * r4 + (u128)d1 * r3 + (u128)r2 * r2;

    u64 c;
    u64 o0, o1, o2, o3, o4;
    o0 = (u64)t0 & M; c = (u64)(t0 >> 51);
    t1 += c; o1 = (u64)t1 & M; c = (u64)(t1 >> 51);
    t2 += c; o2 = (u64)t2 & M; c = (u64)(t2 >> 51);
    t3 += c; o3 = (u64)t3 & M; c = (u64)(t3 >> 51);
    t4 += c; o4 = (u64)t4 & M; c = (u64)(t4 >> 51);
    o0 += c * 19; c = o0 >> 51; o0 &= M;
    o1 += c;      c = o1 >> 51; o1 &= M;
    o2 += c;
    return {{o0, o1, o2, o3, o4}};
}

Fe fe_invert(const Fe& z) {
    // z^(p-2) = z^(2^255 - 21) via the standard addition chain.
    Fe z2, z9, z11, z2_5_0, z2_10_0, z2_20_0, z2_50_0, z2_100_0, t;
    auto sq = [](const Fe& x) { return fe_sq(x); };

    z2 = sq(z);                                  // 2
    t = sq(z2); t = sq(t);                        // 8
    z9 = fe_mul(t, z);                            // 9
    z11 = fe_mul(z9, z2);                         // 11
    t = sq(z11);                                  // 22
    z2_5_0 = fe_mul(t, z9);                       // 2^5 - 2^0 = 31

    t = sq(z2_5_0); for (int i = 1; i < 5; ++i) t = sq(t);
    z2_10_0 = fe_mul(t, z2_5_0);                  // 2^10 - 2^0

    t = sq(z2_10_0); for (int i = 1; i < 10; ++i) t = sq(t);
    z2_20_0 = fe_mul(t, z2_10_0);                 // 2^20 - 2^0

    t = sq(z2_20_0); for (int i = 1; i < 20; ++i) t = sq(t);
    t = fe_mul(t, z2_20_0);                       // 2^40 - 2^0

    t = sq(t); for (int i = 1; i < 10; ++i) t = sq(t);
    z2_50_0 = fe_mul(t, z2_10_0);                 // 2^50 - 2^0

    t = sq(z2_50_0); for (int i = 1; i < 50; ++i) t = sq(t);
    z2_100_0 = fe_mul(t, z2_50_0);                // 2^100 - 2^0

    t = sq(z2_100_0); for (int i = 1; i < 100; ++i) t = sq(t);
    t = fe_mul(t, z2_100_0);                      // 2^200 - 2^0

    t = sq(t); for (int i = 1; i < 50; ++i) t = sq(t);
    t = fe_mul(t, z2_50_0);                       // 2^250 - 2^0

    t = sq(t); t = sq(t); t = sq(t); t = sq(t); t = sq(t);  // *2^5
    return fe_mul(t, z11);                        // 2^255 - 21
}

}  // namespace onion::crypto
