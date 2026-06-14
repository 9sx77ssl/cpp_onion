#pragma once
//
// NATIVE 32-bit-limb device field for ed25519 (sm_75 / Turing), drop-in
// alternative to device_field.cuh. Same public surface (Fe, GeP3,
// GeCachedAffine, fed_add/sub/mul/sq/invert/from_bytes/to_bytes, ged_madd,
// fed_kD2), so search_kernel.cu / xval_kernel.cu can switch with a #define.
//
// WHY: device_field.cuh uses `unsigned __int128`, which Turing has no hardware
// for. nvcc emulates each 64x64->128 multiply as a chain of mul.wide.u32 +
// add-with-carry, which is register-heavy (178 regs/thread) and instruction
// heavy. This file represents a field element as 8 x uint32_t limbs (radix
// 2^32) and multiplies with NATIVE mad.lo/mad.hi (mul.wide.u32 + __umulhi)
// carry chains, then reduces using 2^256 = 38 (mod p) -- no u128 anywhere.
//
// REPRESENTATION: Fe.w[0..7], value = sum w[i] * 2^(32 i), a 256-bit integer
// that is a representative of its residue class mod p = 2^255 - 19. After every
// mul/sq/add/sub the value is brought to a "weakly reduced" form: < 2^256 with
// the top bit semantics handled by the 2^256=38 fold, and add/sub keep limbs
// in 32 bits via carry. Only fed_to_bytes performs the final canonical
// reduction to [0, p). Internal ops therefore work with representatives that
// may be in [0, 2^256), which is closed under the operations below.
//
// CORRECTNESS: when selected (ONION_CUDA_FIELD32=ON, the default), this field
// is cross-validated bit-for-bit against libsodium base_noclamp(a0+8i) by the
// xval gate (tests/test_cuda_xval.cpp -> run_incremental_xval). Never weaken
// that gate.

#include <cstdint>

namespace onion::cuda {

// Field element: 8 x 32-bit limbs, little-endian (w[0] = bits 0..31).
struct Fe {
    uint32_t w[8];
};

// Extended-coordinate point: x = X/Z, y = Y/Z, T = XY/Z.
struct GeP3 {
    Fe X, Y, Z, T;
};

// Cached affine addend (Z normalized to 1); ged_madd uses d = 2*p.Z.
struct GeCachedAffine {
    Fe YplusX, YminusX, T2d;
};

// ---- byte I/O (MUST match the 51-bit path's encoding) --------------------

// Decode 32 little-endian bytes; ignores bit 255 (the sign bit), matching the
// hot path and the 51-bit fed_from_bytes.
__device__ __forceinline__ Fe fed_from_bytes(const uint8_t* s) {
    Fe h;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        h.w[i] = (uint32_t)s[4 * i] | ((uint32_t)s[4 * i + 1] << 8) |
                 ((uint32_t)s[4 * i + 2] << 16) | ((uint32_t)s[4 * i + 3] << 24);
    }
    h.w[7] &= 0x7fffffffu;  // drop the sign bit (bit 255)
    return h;
}

// Canonical reduction of an 8-limb representative in [0, 2^256) to [0, p),
// then write 32 little-endian bytes (sign bit left clear, matching the hot
// path). Standard refextract: try t = w - p; if no final borrow, t was >= p
// so keep t, else keep w.
__device__ __forceinline__ void fed_to_bytes(uint8_t* out, const Fe& in) {
    uint32_t w[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = in.w[i];

    // First fold any value >= 2^255 down: since 2^255 = 19 (mod p), but our
    // representative is < 2^256 here. Fully reduce mod p by conditional
    // subtraction of p, done twice to cover w in [0, 2^256) (w can be up to
    // ~2p+epsilon after weak reduction). p = 2^255 - 19.
    // p limbs: w0 = 0xffffffed, w1..w6 = 0xffffffff, w7 = 0x7fffffff.
    auto sub_p = [&](uint32_t* x) -> uint32_t {
        // returns borrow (1 if x < p, i.e. subtraction underflowed)
        uint64_t borrow = 0;
        uint64_t d;
        d = (uint64_t)x[0] - 0xffffffedull - borrow; x[0] = (uint32_t)d; borrow = (d >> 63);
#pragma unroll
        for (int i = 1; i < 7; ++i) {
            d = (uint64_t)x[i] - 0xffffffffull - borrow; x[i] = (uint32_t)d; borrow = (d >> 63);
        }
        d = (uint64_t)x[7] - 0x7fffffffull - borrow; x[7] = (uint32_t)d; borrow = (d >> 63);
        return (uint32_t)borrow;
    };

    // Reduce while >= p. Worst-case representative < 2^256 < 3p, so two passes
    // suffice; do up to three to be safe (cheap, branch-predicted, uniform).
#pragma unroll
    for (int pass = 0; pass < 3; ++pass) {
        uint32_t t[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) t[i] = w[i];
        uint32_t borrow = sub_p(t);
        if (!borrow) {  // w >= p, keep the subtracted value
#pragma unroll
            for (int i = 0; i < 8; ++i) w[i] = t[i];
        }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[4 * i] = (uint8_t)(w[i] & 0xff);
        out[4 * i + 1] = (uint8_t)((w[i] >> 8) & 0xff);
        out[4 * i + 2] = (uint8_t)((w[i] >> 16) & 0xff);
        out[4 * i + 3] = (uint8_t)((w[i] >> 24) & 0xff);
    }
}

// ---- 2d constant ---------------------------------------------------------

// 2d = 2*(-121665/121666) mod p. We materialize the SAME field element as the
// 51-bit fed_kD2() by reusing its known canonical byte encoding. Rather than
// hardcode bytes, derive the 8x32 limbs from the 5x51 limbs at compile time:
//   51-bit limbs (from ge25519.cpp kD2):
//     0x69b9426b2f159, 0x35050762add7a, 0x3cf44c0038052, 0x6738cc7407977, 0x2406d9dc56dff
// The 256-bit integer = sum limb[i] << (51 i). Precomputed 8x32 limbs below
// (verified by the xval gate, which compares the device chain -- built with
// THIS constant -- against libsodium).
__device__ __forceinline__ Fe fed_kD2() {
    // Computed from the 51-bit limbs:  value = l0 + l1*2^51 + l2*2^102 +
    // l3*2^153 + l4*2^204, then split into 32-bit limbs.
    return Fe{{0x26b2f159u, 0xebd69b94u, 0x8283b156u, 0x00e0149au,
               0xeef3d130u, 0x198e80f2u, 0x56dffce7u, 0x2406d9dcu}};
}

// ---- field arithmetic -----------------------------------------------------

// Weak reduction: given a 16-limb product r[0..15] (each 32 bits, value =
// sum r[i] 2^(32 i) < 2^512), fold the high half via 2^256 = 38 (mod p) into
// an 8-limb representative < 2^256. Used by fed_mul/fed_sq.
//
// Step 1: lo = r[0..7], hi = r[8..15]. result = lo + 38*hi  (since
//         2^256 = 38 mod p).  38*hi is up to ~2^261, so it spills into a few
//         extra limbs; capture them as carry2 and fold again (38*carry2),
//         which is tiny. Two folds bring it below 2^256 + small, and one final
//         carry pass + top fold (2^256=38) settles it to < 2^256.
__device__ __forceinline__ Fe fed32_reduce_wide(const uint32_t r[16]) {
    // acc[0..7] = r[0..7] + 38 * r[8..15]
    uint64_t acc[9];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = (uint64_t)r[i];
    acc[8] = 0;

    // Add 38*hi: 38 * r[8+i] contributes to limb i. 38*2^32 fits in 64 bits.
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] += (uint64_t)38u * (uint64_t)r[8 + i];

    // Propagate carries across acc[0..8].
    uint32_t out[8];
    uint64_t carry = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint64_t v = acc[i] + carry;
        out[i] = (uint32_t)v;
        carry = v >> 32;
    }
    // `carry` now holds bits >= 2^256. Fold once more: top * 38 into limb 0..
    // carry is small (< ~2^30 worst case), 38*carry < 2^36.
    uint64_t fold = carry * 38u;
#pragma unroll
    for (int i = 0; i < 8 && fold; ++i) {
        uint64_t v = (uint64_t)out[i] + fold;
        out[i] = (uint32_t)v;
        fold = v >> 32;
    }
    // After this second fold the residual `fold` is 0 or 1*38-scale; one more
    // tiny ripple guaranteed to settle (the value is now < 2^256 + 38).
    if (fold) {
        uint64_t v = (uint64_t)out[0] + fold;
        out[0] = (uint32_t)v;
        uint64_t c = v >> 32;
#pragma unroll
        for (int i = 1; i < 8 && c; ++i) {
            uint64_t vv = (uint64_t)out[i] + c;
            out[i] = (uint32_t)vv;
            c = vv >> 32;
        }
    }
    Fe o;
#pragma unroll
    for (int i = 0; i < 8; ++i) o.w[i] = out[i];
    return o;
}

// Schoolbook 8x8 -> 16-limb product, then weak-reduce (2^256 = 38 mod p). Uses
// native 32-bit MACs with a 32-bit running carry (a 64-bit accumulate per term
// extracts lo/hi). Measured the lowest register count (the kernel settles at
// ~96 regs/thread with the early M=256/1024 sweep, ~113 regs/thread at the
// committed M=3072, 0 spills either way on sm_75 per ptxas -v) and the best
// throughput of the variants tried -- an interleaved-reduction column
// form needs a 9-limb 64-bit accumulator that costs ~24 more registers and ran
// slightly slower here. Validated on host bit-for-bit vs the libsodium field.
__device__ __forceinline__ Fe fed_mul(const Fe& a, const Fe& b) {
    uint32_t r[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) r[i] = 0;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t carry = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            // p = a[i]*b[j] + r[i+j] + carry  (full 64-bit accumulate)
            uint64_t p = (uint64_t)a.w[i] * (uint64_t)b.w[j] +
                         (uint64_t)r[i + j] + (uint64_t)carry;
            r[i + j] = (uint32_t)p;
            carry = (uint32_t)(p >> 32);
        }
        r[i + 8] = carry;
    }
    return fed32_reduce_wide(r);
}

// Squaring: reuse the multiply structure (the doubling-of-cross-products
// optimization saves a few MACs but the schoolbook path is already native and
// register-light; keep it simple and provably correct).
__device__ __forceinline__ Fe fed_sq(const Fe& a) {
    return fed_mul(a, a);
}

// Addition: limbwise add with carry; result < 2^256 + 2^33, fold the top
// carry via 2^256 = 38 so the representative stays < 2^256.
__device__ __forceinline__ Fe fed_add(const Fe& a, const Fe& b) {
    uint32_t out[8];
    uint64_t carry = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint64_t v = (uint64_t)a.w[i] + (uint64_t)b.w[i] + carry;
        out[i] = (uint32_t)v;
        carry = v >> 32;
    }
    // Fold overflow bit(s): carry in {0,1}. 2^256 = 38 (mod p).
    uint64_t fold = carry * 38u;
#pragma unroll
    for (int i = 0; i < 8 && fold; ++i) {
        uint64_t v = (uint64_t)out[i] + fold;
        out[i] = (uint32_t)v;
        fold = v >> 32;
    }
    Fe o;
#pragma unroll
    for (int i = 0; i < 8; ++i) o.w[i] = out[i];
    return o;
}

// Subtraction: add 2*p to keep non-negative, subtract b, fold. Using a
// multiple of p (= 0 mod p) keeps the value a valid representative.
//   2p = 2^256 - 38.  a + 2p - b is in (0, 2^257), so after limbwise compute
//   we fold the carry (which is 1 in the common case) via 2^256 = 38.
__device__ __forceinline__ Fe fed_sub(const Fe& a, const Fe& b) {
    // 2p limbs: 2p = 2^256 - 38 => limb0 = 0xffffffda, limb1..7 = 0xffffffff.
    uint32_t out[8];
    uint64_t carry = 0;  // signed-ish accumulate via 64-bit with borrow folded
    // Compute a + 2p first (a is < 2^256, 2p < 2^256, sum < 2^257).
    const uint32_t twop0 = 0xffffffdau;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t twop_i = (i == 0) ? twop0 : 0xffffffffu;
        uint64_t v = (uint64_t)a.w[i] + (uint64_t)twop_i + carry;
        out[i] = (uint32_t)v;
        carry = v >> 32;
    }
    // Now subtract b.
    uint64_t borrow = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint64_t d = (uint64_t)out[i] - (uint64_t)b.w[i] - borrow;
        out[i] = (uint32_t)d;
        borrow = (d >> 63);
    }
    // Net top word: carry (from +2p, 0/1) minus borrow (from -b, 0/1).
    // Value = (carry - borrow) * 2^256 + out. Fold via 2^256 = 38.
    int64_t top = (int64_t)carry - (int64_t)borrow;  // in {-1, 0, 1}
    if (top > 0) {
        uint64_t fold = (uint64_t)top * 38u;
#pragma unroll
        for (int i = 0; i < 8 && fold; ++i) {
            uint64_t v = (uint64_t)out[i] + fold;
            out[i] = (uint32_t)v;
            fold = v >> 32;
        }
    } else if (top < 0) {
        // Borrowed past the top: add p back (one multiple) to restore. Since
        // a + 2p - b >= a + p > 0 always (a >= 0, b < 2^256 < 2p), top < 0 is
        // impossible; handle defensively by adding 2^256 worth as 38.
        uint64_t fold = 38u;  // add 2^256 = 38 once
#pragma unroll
        for (int i = 0; i < 8 && fold; ++i) {
            uint64_t v = (uint64_t)out[i] + fold;
            out[i] = (uint32_t)v;
            fold = v >> 32;
        }
    }
    Fe o;
#pragma unroll
    for (int i = 0; i < 8; ++i) o.w[i] = out[i];
    return o;
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

// ---- point ops (verbatim structure of ge25519.cpp ged_madd) --------------

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
