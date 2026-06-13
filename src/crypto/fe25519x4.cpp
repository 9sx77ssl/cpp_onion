#include "crypto/fe25519x4.hpp"

#include <cstdint>
#include <cstring>

// 4-wide AVX2 port of agl's curve25519-donna 32-bit field (radix 2^25.5, 10
// unsigned limbs). Each scalar int64 in fproduct/fsquare_inner/freduce becomes
// a __m256i with four 64-bit lanes (one per candidate); each 32x32->64 product
// becomes _mm256_mul_epu32 (vpmuludq, low-32 of each lane); accumulation is
// _mm256_add_epi64; carries are _mm256_srli_epi64 + _mm256_and_si256 with the
// 26/25-bit masks. Lanes are fully independent: carries move only along the
// limb axis, never across lanes.
//
// All limbs stay non-negative: fexpand yields limbs < 2^26, products of
// non-negatives are non-negative, freduce_degree adds non-negative *19 terms,
// and the bias used by fe4_sub keeps every limb >= 0. That lets us use plain
// unsigned right-shift carries instead of donna's signed fcontract carry path.

namespace onion::crypto {
namespace {

using i64 = std::int64_t;

constexpr i64 kMask26 = (i64{1} << 26) - 1;  // 0x3ffffff
constexpr i64 kMask25 = (i64{1} << 25) - 1;  // 0x1ffffff

inline __m256i v_set1(i64 x) { return _mm256_set1_epi64x(x); }
inline __m256i v_add(__m256i a, __m256i b) { return _mm256_add_epi64(a, b); }
inline __m256i v_mul(__m256i a, __m256i b) { return _mm256_mul_epu32(a, b); }  // low32*low32
inline __m256i v_and(__m256i a, __m256i x) { return _mm256_and_si256(a, x); }
inline __m256i v_shr(__m256i a, int n) { return _mm256_srli_epi64(a, n); }
inline __m256i v_shl(__m256i a, int n) { return _mm256_slli_epi64(a, n); }

// Scale a 64-bit lane value by a small constant. NOTE: we cannot use vpmuludq
// here because the operand (often a sum of products) already exceeds 32 bits,
// and vpmuludq only reads the low 32 bits of each lane. Use shift-and-add,
// which is exact on the full 64-bit lane.
//   *2  -> <<1 ;  *4 -> <<2 ;  *19 -> <<4 + <<1 + x  (16+2+1)
inline __m256i v_scale(__m256i a, i64 k) {
    switch (k) {
        case 2:  return v_shl(a, 1);
        case 4:  return v_shl(a, 2);
        case 19: return v_add(v_add(v_shl(a, 4), v_shl(a, 1)), a);
        default: return v_mul(a, v_set1(k));  // only used for <=32-bit operands
    }
}

// --- degree reduction: fold limbs 10..18 (the 2^255 wrap) back into 0..8 ---
// scalar: output[i] += output[i+10]*19, done as <<4 + <<1 + <<1(implicit) ...
// here the products are well under 2^59 so a single *19 multiply is exact.
void freduce_degree(__m256i t[19]) {
    for (int i = 8; i >= 0; --i) {
        // output[i] += output[i+10] * 19
        t[i] = v_add(t[i], v_scale(t[i + 10], 19));
    }
}

// --- coefficient reduction: bring every limb back to 26/25 bits (unsigned) ---
// mirrors freduce_coefficients, but limbs are guaranteed non-negative so the
// carry is just an arithmetic/unsigned right shift.
void freduce_coefficients(__m256i t[19]) {
    t[10] = v_set1(0);
    for (int i = 0; i < 10; i += 2) {
        __m256i over = v_shr(t[i], 26);
        t[i] = v_and(t[i], v_set1(kMask26));
        t[i + 1] = v_add(t[i + 1], over);

        over = v_shr(t[i + 1], 25);
        t[i + 1] = v_and(t[i + 1], v_set1(kMask25));
        t[i + 2] = v_add(t[i + 2], over);
    }
    // fold the carry that spilled into limb 10 back into limb 0 (*19)
    t[0] = v_add(t[0], v_scale(t[10], 19));
    t[10] = v_set1(0);

    // one more carry out of limb 0
    __m256i over = v_shr(t[0], 26);
    t[0] = v_and(t[0], v_set1(kMask26));
    t[1] = v_add(t[1], over);
}

// --- the 10x10 product, degree-19, exactly mirroring agl's fproduct ---
void fproduct(__m256i out[19], const __m256i in2[10], const __m256i in[10]) {
    out[0] = v_mul(in2[0], in[0]);

    out[1] = v_add(v_mul(in2[0], in[1]),
                   v_mul(in2[1], in[0]));

    out[2] = v_add(v_scale(v_mul(in2[1], in[1]), 2),
              v_add(v_mul(in2[0], in[2]),
                    v_mul(in2[2], in[0])));

    out[3] = v_add(v_add(v_mul(in2[1], in[2]), v_mul(in2[2], in[1])),
                   v_add(v_mul(in2[0], in[3]), v_mul(in2[3], in[0])));

    out[4] = v_add(v_mul(in2[2], in[2]),
              v_add(v_scale(v_add(v_mul(in2[1], in[3]), v_mul(in2[3], in[1])), 2),
              v_add(v_mul(in2[0], in[4]), v_mul(in2[4], in[0]))));

    out[5] = v_add(v_add(v_mul(in2[2], in[3]), v_mul(in2[3], in[2])),
              v_add(v_add(v_mul(in2[1], in[4]), v_mul(in2[4], in[1])),
                    v_add(v_mul(in2[0], in[5]), v_mul(in2[5], in[0]))));

    out[6] = v_add(v_scale(v_add(v_mul(in2[3], in[3]),
                            v_add(v_mul(in2[1], in[5]), v_mul(in2[5], in[1]))), 2),
              v_add(v_add(v_mul(in2[2], in[4]), v_mul(in2[4], in[2])),
                    v_add(v_mul(in2[0], in[6]), v_mul(in2[6], in[0]))));

    out[7] = v_add(v_add(v_mul(in2[3], in[4]), v_mul(in2[4], in[3])),
              v_add(v_add(v_mul(in2[2], in[5]), v_mul(in2[5], in[2])),
              v_add(v_add(v_mul(in2[1], in[6]), v_mul(in2[6], in[1])),
                    v_add(v_mul(in2[0], in[7]), v_mul(in2[7], in[0])))));

    out[8] = v_add(v_mul(in2[4], in[4]),
              v_add(v_scale(v_add(v_add(v_mul(in2[3], in[5]), v_mul(in2[5], in[3])),
                                  v_add(v_mul(in2[1], in[7]), v_mul(in2[7], in[1]))), 2),
              v_add(v_add(v_mul(in2[2], in[6]), v_mul(in2[6], in[2])),
                    v_add(v_mul(in2[0], in[8]), v_mul(in2[8], in[0])))));

    out[9] = v_add(v_add(v_mul(in2[4], in[5]), v_mul(in2[5], in[4])),
              v_add(v_add(v_mul(in2[3], in[6]), v_mul(in2[6], in[3])),
              v_add(v_add(v_mul(in2[2], in[7]), v_mul(in2[7], in[2])),
              v_add(v_add(v_mul(in2[1], in[8]), v_mul(in2[8], in[1])),
                    v_add(v_mul(in2[0], in[9]), v_mul(in2[9], in[0]))))));

    out[10] = v_add(v_scale(v_add(v_mul(in2[5], in[5]),
                             v_add(v_add(v_mul(in2[3], in[7]), v_mul(in2[7], in[3])),
                                   v_add(v_mul(in2[1], in[9]), v_mul(in2[9], in[1])))), 2),
               v_add(v_add(v_mul(in2[4], in[6]), v_mul(in2[6], in[4])),
                     v_add(v_mul(in2[2], in[8]), v_mul(in2[8], in[2]))));

    out[11] = v_add(v_add(v_mul(in2[5], in[6]), v_mul(in2[6], in[5])),
               v_add(v_add(v_mul(in2[4], in[7]), v_mul(in2[7], in[4])),
               v_add(v_add(v_mul(in2[3], in[8]), v_mul(in2[8], in[3])),
                     v_add(v_mul(in2[2], in[9]), v_mul(in2[9], in[2])))));

    out[12] = v_add(v_mul(in2[6], in[6]),
               v_add(v_scale(v_add(v_mul(in2[5], in[7]), v_mul(in2[7], in[5])), 2),
               v_add(v_scale(v_add(v_mul(in2[3], in[9]), v_mul(in2[9], in[3])), 2),
                     v_add(v_mul(in2[4], in[8]), v_mul(in2[8], in[4])))));

    out[13] = v_add(v_add(v_mul(in2[6], in[7]), v_mul(in2[7], in[6])),
               v_add(v_add(v_mul(in2[5], in[8]), v_mul(in2[8], in[5])),
                     v_add(v_mul(in2[4], in[9]), v_mul(in2[9], in[4]))));

    out[14] = v_add(v_scale(v_add(v_mul(in2[7], in[7]),
                             v_add(v_mul(in2[5], in[9]), v_mul(in2[9], in[5]))), 2),
               v_add(v_mul(in2[6], in[8]), v_mul(in2[8], in[6])));

    out[15] = v_add(v_add(v_mul(in2[7], in[8]), v_mul(in2[8], in[7])),
                    v_add(v_mul(in2[6], in[9]), v_mul(in2[9], in[6])));

    out[16] = v_add(v_mul(in2[8], in[8]),
                    v_scale(v_add(v_mul(in2[7], in[9]), v_mul(in2[9], in[7])), 2));

    out[17] = v_add(v_mul(in2[8], in[9]), v_mul(in2[9], in[8]));

    out[18] = v_scale(v_mul(in2[9], in[9]), 2);
}

// --- the squaring, degree-19, exactly mirroring agl's fsquare_inner ---
void fsquare_inner(__m256i out[19], const __m256i in[10]) {
    out[0] = v_mul(in[0], in[0]);

    out[1] = v_scale(v_mul(in[0], in[1]), 2);

    out[2] = v_scale(v_add(v_mul(in[1], in[1]), v_mul(in[0], in[2])), 2);

    out[3] = v_scale(v_add(v_mul(in[1], in[2]), v_mul(in[0], in[3])), 2);

    out[4] = v_add(v_mul(in[2], in[2]),
              v_add(v_scale(v_mul(in[1], in[3]), 4),
                    v_scale(v_mul(in[0], in[4]), 2)));

    out[5] = v_scale(v_add(v_mul(in[2], in[3]),
                      v_add(v_mul(in[1], in[4]), v_mul(in[0], in[5]))), 2);

    out[6] = v_scale(v_add(v_mul(in[3], in[3]),
                      v_add(v_mul(in[2], in[4]),
                      v_add(v_mul(in[0], in[6]),
                            v_scale(v_mul(in[1], in[5]), 2)))), 2);

    out[7] = v_scale(v_add(v_mul(in[3], in[4]),
                      v_add(v_mul(in[2], in[5]),
                      v_add(v_mul(in[1], in[6]), v_mul(in[0], in[7])))), 2);

    out[8] = v_add(v_mul(in[4], in[4]),
              v_scale(v_add(v_mul(in[2], in[6]),
                       v_add(v_mul(in[0], in[8]),
                             v_scale(v_add(v_mul(in[1], in[7]), v_mul(in[3], in[5])), 2))), 2));

    out[9] = v_scale(v_add(v_mul(in[4], in[5]),
                      v_add(v_mul(in[3], in[6]),
                      v_add(v_mul(in[2], in[7]),
                      v_add(v_mul(in[1], in[8]), v_mul(in[0], in[9]))))), 2);

    out[10] = v_scale(v_add(v_mul(in[5], in[5]),
                       v_add(v_mul(in[4], in[6]),
                       v_add(v_mul(in[2], in[8]),
                             v_scale(v_add(v_mul(in[3], in[7]), v_mul(in[1], in[9])), 2)))), 2);

    out[11] = v_scale(v_add(v_mul(in[5], in[6]),
                       v_add(v_mul(in[4], in[7]),
                       v_add(v_mul(in[3], in[8]), v_mul(in[2], in[9])))), 2);

    out[12] = v_add(v_mul(in[6], in[6]),
               v_scale(v_add(v_mul(in[4], in[8]),
                             v_scale(v_add(v_mul(in[5], in[7]), v_mul(in[3], in[9])), 2)), 2));

    out[13] = v_scale(v_add(v_mul(in[6], in[7]),
                       v_add(v_mul(in[5], in[8]), v_mul(in[4], in[9]))), 2);

    out[14] = v_scale(v_add(v_mul(in[7], in[7]),
                       v_add(v_mul(in[6], in[8]),
                             v_scale(v_mul(in[5], in[9]), 2))), 2);

    out[15] = v_scale(v_add(v_mul(in[7], in[8]), v_mul(in[6], in[9])), 2);

    out[16] = v_add(v_mul(in[8], in[8]), v_scale(v_mul(in[7], in[9]), 4));

    out[17] = v_scale(v_mul(in[8], in[9]), 2);

    out[18] = v_scale(v_mul(in[9], in[9]), 2);
}

}  // namespace

Fe4 fe4_load(const std::array<std::byte, 32>& a, const std::array<std::byte, 32>& b,
             const std::array<std::byte, 32>& c, const std::array<std::byte, 32>& d) {
    // fexpand each candidate into 10 limbs, then transpose into SoA lanes.
    auto expand = [](const std::array<std::byte, 32>& in, i64 limb[10]) {
        auto rd32 = [&](int start) -> std::uint64_t {
            std::uint64_t r = 0;
            for (int k = 3; k >= 0; --k)
                r = (r << 8) | std::to_integer<std::uint64_t>(in[std::size_t(start + k)]);
            return r;
        };
        // F(n, start, shift, mask): donna fexpand layout
        limb[0] = i64((rd32(0) >> 0) & 0x3ffffffu);
        limb[1] = i64((rd32(3) >> 2) & 0x1ffffffu);
        limb[2] = i64((rd32(6) >> 3) & 0x3ffffffu);
        limb[3] = i64((rd32(9) >> 5) & 0x1ffffffu);
        limb[4] = i64((rd32(12) >> 6) & 0x3ffffffu);
        limb[5] = i64((rd32(16) >> 0) & 0x1ffffffu);
        limb[6] = i64((rd32(19) >> 1) & 0x3ffffffu);
        limb[7] = i64((rd32(22) >> 3) & 0x1ffffffu);
        limb[8] = i64((rd32(25) >> 4) & 0x3ffffffu);
        limb[9] = i64((rd32(28) >> 6) & 0x1ffffffu);
    };

    i64 la[10], lb[10], lc[10], ld[10];
    expand(a, la);
    expand(b, lb);
    expand(c, lc);
    expand(d, ld);

    Fe4 f;
    for (int i = 0; i < 10; ++i) {
        // lane 0 = candidate a, ..., lane 3 = candidate d
        f.v[i] = _mm256_set_epi64x(ld[i], lc[i], lb[i], la[i]);
    }
    return f;
}

void fe4_store_y(const Fe4& f, std::array<std::byte, 32>& a, std::array<std::byte, 32>& b,
                 std::array<std::byte, 32>& c, std::array<std::byte, 32>& d) {
    // Pull the four lanes back out into per-candidate signed-32 limb arrays,
    // then run agl's fcontract (canonical reduction + LE serialisation).
    alignas(32) i64 tmp[10][4];
    for (int i = 0; i < 10; ++i)
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp[i]), f.v[i]);

    auto contract = [](i64 src[10], std::array<std::byte, 32>& out) {
        std::int32_t input[10];
        for (int i = 0; i < 10; ++i) input[i] = std::int32_t(src[i]);

        // Two passes of signed carry (handles any residual negatives, though
        // our limbs are non-negative here), matching donna fcontract.
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 9; ++i) {
                if (i & 1) {
                    const std::int32_t mask = input[i] >> 31;
                    const std::int32_t carry = -((input[i] & mask) >> 25);
                    input[i] = input[i] + (carry << 25);
                    input[i + 1] = input[i + 1] - carry;
                } else {
                    const std::int32_t mask = input[i] >> 31;
                    const std::int32_t carry = -((input[i] & mask) >> 26);
                    input[i] = input[i] + (carry << 26);
                    input[i + 1] = input[i + 1] - carry;
                }
            }
            const std::int32_t mask = input[9] >> 31;
            const std::int32_t carry = -((input[9] & mask) >> 25);
            input[9] = input[9] + (carry << 25);
            input[0] = input[0] - (carry * 19);
        }
        {
            const std::int32_t mask = input[0] >> 31;
            const std::int32_t carry = -((input[0] & mask) >> 26);
            input[0] = input[0] + (carry << 26);
            input[1] = input[1] - carry;
        }
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 9; ++i) {
                if (i & 1) {
                    const std::int32_t carry = input[i] >> 25;
                    input[i] &= 0x1ffffff;
                    input[i + 1] += carry;
                } else {
                    const std::int32_t carry = input[i] >> 26;
                    input[i] &= 0x3ffffff;
                    input[i + 1] += carry;
                }
            }
            const std::int32_t carry = input[9] >> 25;
            input[9] &= 0x1ffffff;
            input[0] += 19 * carry;
        }

        // Conditional subtraction of p to reach the canonical representative.
        auto s32_gte = [](std::int32_t x, std::int32_t y) -> std::int32_t {
            x -= y;  // x>=0 iff x>=y (no overflow for our ranges)
            return ~(x >> 31);
        };
        auto s32_eq = [](std::int32_t x, std::int32_t y) -> std::int32_t {
            x = ~(x ^ y);
            x &= x << 16;
            x &= x << 8;
            x &= x << 4;
            x &= x << 2;
            x &= x << 1;
            return x >> 31;
        };
        std::int32_t mask = s32_gte(input[0], 0x3ffffed);
        for (int i = 1; i < 10; ++i)
            mask &= (i & 1) ? s32_eq(input[i], 0x1ffffff) : s32_eq(input[i], 0x3ffffff);
        input[0] -= mask & 0x3ffffed;
        for (int i = 1; i < 10; ++i)
            input[i] -= mask & ((i & 1) ? 0x1ffffff : 0x3ffffff);

        input[1] <<= 2;
        input[2] <<= 3;
        input[3] <<= 5;
        input[4] <<= 6;
        input[6] <<= 1;
        input[7] <<= 3;
        input[8] <<= 4;
        input[9] <<= 6;

        std::uint8_t o[32];
        std::memset(o, 0, sizeof(o));
        auto F = [&](int i, int s) {
            o[s + 0] |= std::uint8_t(input[i] & 0xff);
            o[s + 1] = std::uint8_t((input[i] >> 8) & 0xff);
            o[s + 2] = std::uint8_t((input[i] >> 16) & 0xff);
            o[s + 3] = std::uint8_t((input[i] >> 24) & 0xff);
        };
        F(0, 0);
        F(1, 3);
        F(2, 6);
        F(3, 9);
        F(4, 12);
        F(5, 16);
        F(6, 19);
        F(7, 22);
        F(8, 25);
        F(9, 28);
        for (int i = 0; i < 32; ++i) out[std::size_t(i)] = std::byte(o[i]);
    };

    i64 sa[10], sb[10], sc[10], sd[10];
    for (int i = 0; i < 10; ++i) {
        sa[i] = tmp[i][0];
        sb[i] = tmp[i][1];
        sc[i] = tmp[i][2];
        sd[i] = tmp[i][3];
    }
    contract(sa, a);
    contract(sb, b);
    contract(sc, c);
    contract(sd, d);
}

Fe4 fe4_mul(const Fe4& a, const Fe4& b) {
    __m256i t[19];
    fproduct(t, a.v, b.v);
    freduce_degree(t);
    freduce_coefficients(t);
    Fe4 r;
    for (int i = 0; i < 10; ++i) r.v[i] = t[i];
    return r;
}

Fe4 fe4_sq(const Fe4& a) {
    __m256i t[19];
    fsquare_inner(t, a.v);
    freduce_degree(t);
    freduce_coefficients(t);
    Fe4 r;
    for (int i = 0; i < 10; ++i) r.v[i] = t[i];
    return r;
}

Fe4 fe4_add(const Fe4& a, const Fe4& b) {
    // Add limb-wise then carry-reduce so the result is small enough for fe4_mul.
    __m256i t[19];
    for (int i = 0; i < 10; ++i) t[i] = v_add(a.v[i], b.v[i]);
    for (int i = 10; i < 19; ++i) t[i] = v_set1(0);
    freduce_coefficients(t);
    Fe4 r;
    for (int i = 0; i < 10; ++i) r.v[i] = t[i];
    return r;
}

Fe4 fe4_sub(const Fe4& a, const Fe4& b) {
    // a - b, biased by 8p (a multiple of p, so value-preserving mod p) to keep
    // every limb non-negative before the unsigned carry-reduce. 8p's limbs are
    // 8 * p_limb[i], all >= the max input limb (~2^26), so a + bias - b >= 0.
    // Reduce afterwards so the result limbs are <26/25 bits and fe4_mul cannot
    // overflow on a bias-form zero (the scalar fe_sub guards the same hazard).
    static constexpr i64 P0 = 0x3ffffed, P_ODD = 0x1ffffff, P_EVEN = 0x3ffffff;
    __m256i bias[10];
    bias[0] = v_set1(8 * P0);
    for (int i = 1; i < 10; ++i) bias[i] = v_set1(8 * ((i & 1) ? P_ODD : P_EVEN));

    __m256i t[19];
    for (int i = 0; i < 10; ++i) t[i] = v_add(v_add(a.v[i], bias[i]), _mm256_sub_epi64(v_set1(0), b.v[i]));
    // The above subtracts b via two's complement add; since bias[i] >> b limb,
    // each lane stays non-negative. Equivalent to a[i] + bias[i] - b[i].
    for (int i = 10; i < 19; ++i) t[i] = v_set1(0);
    freduce_coefficients(t);
    Fe4 r;
    for (int i = 0; i < 10; ++i) r.v[i] = t[i];
    return r;
}

Fe4 fe4_invert(const Fe4& z) {
    // z^(p-2) = z^(2^255 - 21) via the same addition chain as scalar fe_invert.
    Fe4 z2, z9, z11, z2_5_0, z2_10_0, z2_20_0, z2_50_0, z2_100_0, t;

    z2 = fe4_sq(z);                               // 2
    t = fe4_sq(z2); t = fe4_sq(t);                // 8
    z9 = fe4_mul(t, z);                           // 9
    z11 = fe4_mul(z9, z2);                        // 11
    t = fe4_sq(z11);                              // 22
    z2_5_0 = fe4_mul(t, z9);                      // 2^5 - 2^0 = 31

    t = fe4_sq(z2_5_0); for (int i = 1; i < 5; ++i) t = fe4_sq(t);
    z2_10_0 = fe4_mul(t, z2_5_0);                 // 2^10 - 2^0

    t = fe4_sq(z2_10_0); for (int i = 1; i < 10; ++i) t = fe4_sq(t);
    z2_20_0 = fe4_mul(t, z2_10_0);                // 2^20 - 2^0

    t = fe4_sq(z2_20_0); for (int i = 1; i < 20; ++i) t = fe4_sq(t);
    t = fe4_mul(t, z2_20_0);                      // 2^40 - 2^0

    t = fe4_sq(t); for (int i = 1; i < 10; ++i) t = fe4_sq(t);
    z2_50_0 = fe4_mul(t, z2_10_0);                // 2^50 - 2^0

    t = fe4_sq(z2_50_0); for (int i = 1; i < 50; ++i) t = fe4_sq(t);
    z2_100_0 = fe4_mul(t, z2_50_0);               // 2^100 - 2^0

    t = fe4_sq(z2_100_0); for (int i = 1; i < 100; ++i) t = fe4_sq(t);
    t = fe4_mul(t, z2_100_0);                     // 2^200 - 2^0

    t = fe4_sq(t); for (int i = 1; i < 50; ++i) t = fe4_sq(t);
    t = fe4_mul(t, z2_50_0);                      // 2^250 - 2^0

    t = fe4_sq(t); t = fe4_sq(t); t = fe4_sq(t); t = fe4_sq(t); t = fe4_sq(t);  // *2^5
    return fe4_mul(t, z11);                       // 2^255 - 21
}

}  // namespace onion::crypto
