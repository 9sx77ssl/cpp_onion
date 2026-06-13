#pragma once
//
// Device port of the validated 5x51 donna64 field + ed25519 point ops.
//
// This is a verbatim translation of the host code that was cross-validated
// bit-for-bit against libsodium:
//   - src/crypto/fe25519.cpp  (Fe, fe_mul/sq/add/sub/invert/from_bytes/to_bytes)
//   - src/crypto/ge25519.cpp  (GeP3, ge_madd, ge_to_cached_affine, kD2)
//
// It uses `unsigned __int128`, which is verified to compile AND run correctly
// in device code on sm_75 (Turing). Self-contained: no std::, only raw arrays,
// uint8_t and uint64_t, so it can live entirely in __device__ code.
//
// Algorithm changes vs the host are NONE. Only the surface (struct/function
// signatures, span->pointer) is adapted so it compiles as CUDA device code.

#include <cstdint>

namespace onion::cuda {

__extension__ typedef unsigned __int128 u128_t;

// reduce_mask_51
__device__ __forceinline__ constexpr uint64_t fed_mask() { return (uint64_t{1} << 51) - 1; }

// Field element mod 2^255-19 in 5 x 51-bit limbs (donna64 representation).
struct Fe {
    uint64_t v[5];
};

// Extended-coordinate point: x = X/Z, y = Y/Z, T = XY/Z.
struct GeP3 {
    Fe X, Y, Z, T;
};

// Cached affine addend (Z normalized to 1); ged_madd uses d = 2*p.Z.
struct GeCachedAffine {
    Fe YplusX, YminusX, T2d;
};

// 2d = 2*(-121665/121666) mod p, as 5x51 limbs (matches ge25519.cpp kD2).
__device__ __forceinline__ Fe fed_kD2() {
    return Fe{{0x69b9426b2f159ULL, 0x35050762add7aULL, 0x3cf44c0038052ULL,
               0x6738cc7407977ULL, 0x2406d9dc56dffULL}};
}

// ---- field ops (verbatim port of fe25519.cpp) ----

__device__ __forceinline__ uint64_t fed_load64(const uint8_t* s, int off) {
    uint64_t r = 0;
    for (int i = 7; i >= 0; --i) r = (r << 8) | (uint64_t)s[off + i];
    return r;
}

__device__ __forceinline__ Fe fed_from_bytes(const uint8_t* s) {
    const uint64_t M = fed_mask();
    const uint64_t x0 = fed_load64(s, 0), x1 = fed_load64(s, 8),
                   x2 = fed_load64(s, 16), x3 = fed_load64(s, 24);
    Fe h;
    h.v[0] = x0 & M;
    h.v[1] = ((x0 >> 51) | (x1 << 13)) & M;
    h.v[2] = ((x1 >> 38) | (x2 << 26)) & M;
    h.v[3] = ((x2 >> 25) | (x3 << 39)) & M;
    h.v[4] = (x3 >> 12) & M;  // ignores bit 255 (the sign bit)
    return h;
}

// Writes 32 bytes (little-endian y; sign bit NOT set, matching the hot path).
__device__ __forceinline__ void fed_to_bytes(uint8_t* out, const Fe& in) {
    const uint64_t M = fed_mask();
    uint64_t t[5] = {in.v[0], in.v[1], in.v[2], in.v[3], in.v[4]};

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

    uint64_t f;
    int o = 0;
    auto write51 = [&](int n, int shift) {
        f = (t[n] >> shift) | (t[n + 1] << (51 - shift));
        for (int i = 0; i < 8; ++i, f >>= 8) out[o++] = (uint8_t)(f & 0xff);
    };
    write51(0, 0);
    write51(1, 13);
    write51(2, 26);
    write51(3, 39);
}

__device__ __forceinline__ Fe fed_add(const Fe& a, const Fe& b) {
    return Fe{{a.v[0] + b.v[0], a.v[1] + b.v[1], a.v[2] + b.v[2],
               a.v[3] + b.v[3], a.v[4] + b.v[4]}};
}

__device__ __forceinline__ Fe fed_sub(const Fe& a, const Fe& b) {
    // bias by 8p then carry-reduce so output limbs are <= M+epsilon (essential:
    // an unreduced bias-form zero overflows fed_mul's `o0 += c*19` u64 step).
    const uint64_t M = fed_mask();
    constexpr uint64_t two54m152 = (uint64_t{1} << 54) - 152;  // 8*(2^51-19)
    constexpr uint64_t two54m8 = (uint64_t{1} << 54) - 8;      // 8*(2^51-1)
    uint64_t t0 = a.v[0] + two54m152 - b.v[0];
    uint64_t t1 = a.v[1] + two54m8 - b.v[1];
    uint64_t t2 = a.v[2] + two54m8 - b.v[2];
    uint64_t t3 = a.v[3] + two54m8 - b.v[3];
    uint64_t t4 = a.v[4] + two54m8 - b.v[4];
    t1 += t0 >> 51; t0 &= M;
    t2 += t1 >> 51; t1 &= M;
    t3 += t2 >> 51; t2 &= M;
    t4 += t3 >> 51; t3 &= M;
    t0 += 19 * (t4 >> 51); t4 &= M;
    t1 += t0 >> 51; t0 &= M;
    return Fe{{t0, t1, t2, t3, t4}};
}

__device__ __forceinline__ Fe fed_mul(const Fe& a, const Fe& b) {
    const uint64_t M = fed_mask();
    uint64_t r0 = a.v[0], r1 = a.v[1], r2 = a.v[2], r3 = a.v[3], r4 = a.v[4];
    const uint64_t s0 = b.v[0], s1 = b.v[1], s2 = b.v[2], s3 = b.v[3], s4 = b.v[4];

    u128_t t0 = (u128_t)r0 * s0;
    u128_t t1 = (u128_t)r0 * s1 + (u128_t)r1 * s0;
    u128_t t2 = (u128_t)r0 * s2 + (u128_t)r2 * s0 + (u128_t)r1 * s1;
    u128_t t3 = (u128_t)r0 * s3 + (u128_t)r3 * s0 + (u128_t)r1 * s2 + (u128_t)r2 * s1;
    u128_t t4 = (u128_t)r0 * s4 + (u128_t)r4 * s0 + (u128_t)r3 * s1 + (u128_t)r1 * s3 + (u128_t)r2 * s2;

    r4 *= 19; r1 *= 19; r2 *= 19; r3 *= 19;

    t0 += (u128_t)r4 * s1 + (u128_t)r1 * s4 + (u128_t)r2 * s3 + (u128_t)r3 * s2;
    t1 += (u128_t)r4 * s2 + (u128_t)r2 * s4 + (u128_t)r3 * s3;
    t2 += (u128_t)r4 * s3 + (u128_t)r3 * s4;
    t3 += (u128_t)r4 * s4;

    uint64_t c;
    uint64_t o0, o1, o2, o3, o4;
    o0 = (uint64_t)t0 & M; c = (uint64_t)(t0 >> 51);
    t1 += c; o1 = (uint64_t)t1 & M; c = (uint64_t)(t1 >> 51);
    t2 += c; o2 = (uint64_t)t2 & M; c = (uint64_t)(t2 >> 51);
    t3 += c; o3 = (uint64_t)t3 & M; c = (uint64_t)(t3 >> 51);
    t4 += c; o4 = (uint64_t)t4 & M; c = (uint64_t)(t4 >> 51);
    o0 += c * 19; c = o0 >> 51; o0 &= M;
    o1 += c;      c = o1 >> 51; o1 &= M;
    o2 += c;
    return Fe{{o0, o1, o2, o3, o4}};
}

__device__ __forceinline__ Fe fed_sq(const Fe& a) {
    const uint64_t M = fed_mask();
    const uint64_t r0 = a.v[0], r1 = a.v[1], r2 = a.v[2], r3 = a.v[3], r4 = a.v[4];
    const uint64_t d0 = r0 * 2, d1 = r1 * 2, d2 = r2 * 2 * 19, d419 = r4 * 19, d4 = d419 * 2;

    u128_t t0 = (u128_t)r0 * r0 + (u128_t)d4 * r1 + (u128_t)d2 * r3;
    u128_t t1 = (u128_t)d0 * r1 + (u128_t)d4 * r2 + (u128_t)r3 * (r3 * 19);
    u128_t t2 = (u128_t)d0 * r2 + (u128_t)r1 * r1 + (u128_t)d4 * r3;
    u128_t t3 = (u128_t)d0 * r3 + (u128_t)d1 * r2 + (u128_t)r4 * d419;
    u128_t t4 = (u128_t)d0 * r4 + (u128_t)d1 * r3 + (u128_t)r2 * r2;

    uint64_t c;
    uint64_t o0, o1, o2, o3, o4;
    o0 = (uint64_t)t0 & M; c = (uint64_t)(t0 >> 51);
    t1 += c; o1 = (uint64_t)t1 & M; c = (uint64_t)(t1 >> 51);
    t2 += c; o2 = (uint64_t)t2 & M; c = (uint64_t)(t2 >> 51);
    t3 += c; o3 = (uint64_t)t3 & M; c = (uint64_t)(t3 >> 51);
    t4 += c; o4 = (uint64_t)t4 & M; c = (uint64_t)(t4 >> 51);
    o0 += c * 19; c = o0 >> 51; o0 &= M;
    o1 += c;      c = o1 >> 51; o1 &= M;
    o2 += c;
    return Fe{{o0, o1, o2, o3, o4}};
}

__device__ __forceinline__ Fe fed_invert(const Fe& z) {
    // z^(p-2) = z^(2^255 - 21) via the standard addition chain (verbatim).
    Fe z2, z9, z11, z2_5_0, z2_10_0, z2_20_0, z2_50_0, z2_100_0, t;

    z2 = fed_sq(z);                                  // 2
    t = fed_sq(z2); t = fed_sq(t);                   // 8
    z9 = fed_mul(t, z);                              // 9
    z11 = fed_mul(z9, z2);                           // 11
    t = fed_sq(z11);                                 // 22
    z2_5_0 = fed_mul(t, z9);                         // 2^5 - 2^0 = 31

    t = fed_sq(z2_5_0); for (int i = 1; i < 5; ++i) t = fed_sq(t);
    z2_10_0 = fed_mul(t, z2_5_0);                    // 2^10 - 2^0

    t = fed_sq(z2_10_0); for (int i = 1; i < 10; ++i) t = fed_sq(t);
    z2_20_0 = fed_mul(t, z2_10_0);                   // 2^20 - 2^0

    t = fed_sq(z2_20_0); for (int i = 1; i < 20; ++i) t = fed_sq(t);
    t = fed_mul(t, z2_20_0);                         // 2^40 - 2^0

    t = fed_sq(t); for (int i = 1; i < 10; ++i) t = fed_sq(t);
    z2_50_0 = fed_mul(t, z2_10_0);                   // 2^50 - 2^0

    t = fed_sq(z2_50_0); for (int i = 1; i < 50; ++i) t = fed_sq(t);
    z2_100_0 = fed_mul(t, z2_50_0);                  // 2^100 - 2^0

    t = fed_sq(z2_100_0); for (int i = 1; i < 100; ++i) t = fed_sq(t);
    t = fed_mul(t, z2_100_0);                        // 2^200 - 2^0

    t = fed_sq(t); for (int i = 1; i < 50; ++i) t = fed_sq(t);
    t = fed_mul(t, z2_50_0);                         // 2^250 - 2^0

    t = fed_sq(t); t = fed_sq(t); t = fed_sq(t); t = fed_sq(t); t = fed_sq(t);  // *2^5
    return fed_mul(t, z11);                          // 2^255 - 21
}

// ---- point ops (verbatim port of ge25519.cpp) ----

// r = p + q, q affine (Z=1). Mixed addition (the hot op).
__device__ __forceinline__ GeP3 ged_madd(const GeP3& p, const GeCachedAffine& q) {
    Fe a = fed_mul(fed_sub(p.Y, p.X), q.YminusX);
    Fe b = fed_mul(fed_add(p.Y, p.X), q.YplusX);
    Fe c = fed_mul(q.T2d, p.T);
    Fe d = fed_add(p.Z, p.Z);  // 2 * p.Z  (q.Z == 1)
    Fe e = fed_sub(b, a);
    Fe f = fed_sub(d, c);
    Fe g = fed_add(d, c);
    Fe h = fed_add(b, a);
    return GeP3{fed_mul(e, f), fed_mul(g, h), fed_mul(f, g), fed_mul(e, h)};
}

}  // namespace onion::cuda
