# Phase 1: Incremental Scalar ed25519 CPU Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the naive per-candidate `NaiveCpuEngine` (full scalar multiplication, ~0.34 M keys/s) with an incremental engine that computes one point addition per candidate (`A += 8B`) plus amortized batch inversion — a 100–250× speedup — behind the existing `IEngine` seam, with every step cross-validated bit-for-bit against libsodium.

**Architecture:** Custom ed25519 field arithmetic (`fe25519`, 5×51-bit limbs, donna64) and group operations (`ge25519`: extended-coordinate point double/add, fixed-base scalar mult for epoch setup). An `IncrementalStepper` walks `A₀, A₀+8B, A₀+16B, …` in projective coordinates, collecting `Z` values and recovering affine `y` for a whole batch with a single field inversion (Montgomery's trick). `IncrementalCpuEngine` drives steppers across threads, matches leading `y`-bytes, and on a hit reconstructs the scalar `a₀+8i` and emits a `MatchCandidate` (with a fresh random PRF prefix). The verifier firewall and Tor key writer from Phase 0 are reused unchanged.

**Tech Stack:** C++23, GCC 16, `unsigned __int128` for field products, libsodium (cross-validation oracle only — never on the hot path).

**Why this is correct despite hand-written crypto:** The proven invariant (validated against libsodium for 5001 steps before this plan was written) is:
> `incremental_chain(a₀)[i]` encodes the same point as `crypto_scalarmult_ed25519_base_noclamp(a₀ + 8i)`, for the leading 31 bytes (pure `y`).

Task 3's cross-validation test is the **source of truth**: any transcription error in the donna64 field code or the point formulas makes it fail, and is fixed against libsodium — never by weakening the test. Field ops are additionally checked against Python-computed vectors (big-int mod `p = 2²⁵⁵−19`) in Task 1.

**Conventions:** namespaces `onion::crypto` (fe/ge/incremental) and `onion::engine`. Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. `mask51 = (uint64_t{1}<<51)-1`.

**Verified constants & test vectors (computed with Python, `p = 2²⁵⁵−19`, before this plan):**

| Quantity | Value |
|---|---|
| `fe_mul`: a (LE32 hex) | `ddccbbaa0099887766554433221100998877665544332211efcdab9078563412` |
| `fe_mul`: b (LE32 hex) | `01efcdab78563412908f7e6d5c4b3a291807f6e5d4c3b2a121436587a9cbed0f` |
| `fe_mul`: a·b mod p | `6c549ee33167ba4ccd73a41253186ddd4225ad6c5ee8d8b55b7a615ad92b0a22` |
| `fe_sq`: a² mod p | `8688b02610872860d0f7eee66990690a0fe2c837fac937eb098e0d043b54e773` |
| `fe_invert`: a⁻¹ mod p | `6c1316d8a841051ef542f904fb62baffdc6ee9c930b63bbcad91b918e0e39f66` |
| basepoint `Bx` limbs | `{0x62d608f25d51a,0x412a4b4f6592a,0x75b7171a4b31d,0x1ff60527118fe,0x216936d3cd6e5}` |
| basepoint `By` limbs | `{0x6666666666658,0x4cccccccccccc,0x1999999999999,0x3333333333333,0x6666666666666}` |
| basepoint `T=Bx·By` limbs | `{0x68ab3a5b7dda3,0x0eea2a5eadbb,0x2af8df483c27e,0x332b375274732,0x67875f0fd78b7}` |
| `8B` encoded (LE32 hex) | `b4b937fca95b2f1e93e41e62fc3c78818ff38a66096fad6e7973e5c90006d321` |
| scalarmult test: a₀ (LE32) | `400000000000000000000000000000000000000000000000107e2d9c5ebf4a00` |
| → a₀·B encoded (LE32) | `494759ec2b42faec2b989685783762a15d87fb11dd2012f34765676cf96728d0` |

---

### Task 1: Field arithmetic `fe25519` (`onion_crypto`)

5×51-bit field element over GF(2²⁵⁵−19). This is the donna64 reference (Andrew Moon / Adam Langley lineage, public domain). The Python vectors above are the correctness gate.

**Files:**
- Create: `src/crypto/fe25519.hpp`, `src/crypto/fe25519.cpp`
- Modify: `src/crypto/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `tests/test_fe25519.cpp`

- [ ] **Step 1: Write the failing test — `tests/test_fe25519.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "crypto/fe25519.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string_view>

namespace {
std::array<std::byte, 32> hex32(std::string_view h) {
    auto nib = [](char c) { return (c >= 'a') ? unsigned(c - 'a' + 10) : unsigned(c - '0'); };
    std::array<std::byte, 32> o;
    for (std::size_t i = 0; i < 32; ++i) o[i] = std::byte((nib(h[2*i]) << 4) | nib(h[2*i+1]));
    return o;
}
std::array<std::byte, 32> to_bytes(const onion::crypto::Fe& f) {
    std::array<std::byte, 32> o;
    onion::crypto::fe_to_bytes(o, f);
    return o;
}
}

TEST_CASE("fe roundtrip from_bytes/to_bytes") {
    auto a = hex32("ddccbbaa0099887766554433221100998877665544332211efcdab9078563412");
    CHECK(to_bytes(onion::crypto::fe_from_bytes(a)) == a);
}

TEST_CASE("fe_mul matches Python big-int product") {
    using namespace onion::crypto;
    auto a = fe_from_bytes(hex32("ddccbbaa0099887766554433221100998877665544332211efcdab9078563412"));
    auto b = fe_from_bytes(hex32("01efcdab78563412908f7e6d5c4b3a291807f6e5d4c3b2a121436587a9cbed0f"));
    CHECK(to_bytes(fe_mul(a, b)) ==
          hex32("6c549ee33167ba4ccd73a41253186ddd4225ad6c5ee8d8b55b7a615ad92b0a22"));
}

TEST_CASE("fe_sq matches Python and equals fe_mul(a,a)") {
    using namespace onion::crypto;
    auto a = fe_from_bytes(hex32("ddccbbaa0099887766554433221100998877665544332211efcdab9078563412"));
    CHECK(to_bytes(fe_sq(a)) ==
          hex32("8688b02610872860d0f7eee66990690a0fe2c837fac937eb098e0d043b54e773"));
    CHECK(to_bytes(fe_sq(a)) == to_bytes(fe_mul(a, a)));
}

TEST_CASE("fe_invert matches Python and a*inv(a)==1") {
    using namespace onion::crypto;
    auto a = fe_from_bytes(hex32("ddccbbaa0099887766554433221100998877665544332211efcdab9078563412"));
    CHECK(to_bytes(fe_invert(a)) ==
          hex32("6c1316d8a841051ef542f904fb62baffdc6ee9c930b63bbcad91b918e0e39f66"));
    auto one = hex32("0100000000000000000000000000000000000000000000000000000000000000");
    CHECK(to_bytes(fe_mul(a, fe_invert(a))) == one);
}

TEST_CASE("fe add/sub are inverse and reduce correctly") {
    using namespace onion::crypto;
    std::mt19937_64 rng(7);
    for (int t = 0; t < 500; ++t) {
        std::array<std::byte, 32> ab, bb;
        for (auto& x : ab) x = std::byte(rng() & 0xff);
        for (auto& x : bb) x = std::byte(rng() & 0xff);
        ab[31] = std::byte(std::to_integer<unsigned>(ab[31]) & 0x7f);  // < 2^255
        bb[31] = std::byte(std::to_integer<unsigned>(bb[31]) & 0x7f);
        auto a = fe_from_bytes(ab), b = fe_from_bytes(bb);
        // (a+b)-b == a
        CHECK(to_bytes(fe_sub(fe_add(a, b), b)) == to_bytes(a));
    }
}
```

- [ ] **Step 2: Run to verify it fails** — add `test_fe25519.cpp` to `tests/CMakeLists.txt` sources, then `cmake --build --preset debug`. Expected: FAIL — `crypto/fe25519.hpp` not found.

- [ ] **Step 3: Write the implementation**

`src/crypto/fe25519.hpp`:

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace onion::crypto {

// Field element mod 2^255-19 in 5 x 51-bit limbs (donna64 representation).
struct Fe {
    std::uint64_t v[5];
};

[[nodiscard]] Fe fe_from_bytes(std::span<const std::byte, 32> s);
void fe_to_bytes(std::span<std::byte, 32> out, const Fe& f);

[[nodiscard]] Fe fe_add(const Fe& a, const Fe& b);
[[nodiscard]] Fe fe_sub(const Fe& a, const Fe& b);
[[nodiscard]] Fe fe_mul(const Fe& a, const Fe& b);
[[nodiscard]] Fe fe_sq(const Fe& a);
[[nodiscard]] Fe fe_invert(const Fe& a);  // a^(p-2)

}  // namespace onion::crypto
```

`src/crypto/fe25519.cpp` (donna64 — the Task-1 vectors validate every operation; if any vector fails, the bug is a transcription error here, fix it against the vector, never edit the test):

```cpp
#include "crypto/fe25519.hpp"

namespace onion::crypto {
namespace {

using u64 = std::uint64_t;
using u128 = unsigned __int128;
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
    // bias by 8p so the result stays non-negative; outputs feed straight into fe_mul.
    constexpr u64 two54m152 = (u64{1} << 54) - 152;  // 8*(2^51-19)
    constexpr u64 two54m8 = (u64{1} << 54) - 8;      // 8*(2^51-1)
    Fe r = {{a.v[0] + two54m152 - b.v[0],
             a.v[1] + two54m8 - b.v[1],
             a.v[2] + two54m8 - b.v[2],
             a.v[3] + two54m8 - b.v[3],
             a.v[4] + two54m8 - b.v[4]}};
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
```

`src/crypto/CMakeLists.txt`: add `fe25519.cpp` to the `onion_crypto` sources list.
`tests/CMakeLists.txt`: add `test_fe25519.cpp` to the sources list.

- [ ] **Step 4: Run tests** — `cmake --build --preset debug && ctest --preset debug`. Expected: all pass (the 5 new fe tests included). If `fe_mul`/`fe_sq`/`fe_invert` vector tests fail, the bug is in this file (compare against the Python vector); do NOT modify the expected vectors.

- [ ] **Step 5: Commit**
```bash
git add -A
git commit -m "feat(crypto): fe25519 5x51 field arithmetic (donna64) with Python KATs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Group operations `ge25519` (`onion_crypto`)

Extended twisted-Edwards point ops. Hardcoded basepoint; double + add (formula sequences are representation-independent, using the Task-1 fe ops); fixed-base scalar mult (double-and-add, epoch setup only); affine-`y` encoding given a precomputed `Z⁻¹`.

**Files:**
- Create: `src/crypto/ge25519.hpp`, `src/crypto/ge25519.cpp`
- Modify: `src/crypto/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `tests/test_ge25519.cpp` (cross-checks against libsodium — links `PkgConfig::SODIUM`, already a dep of onion_crypto)

- [ ] **Step 1: Write the failing test — `tests/test_ge25519.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "crypto/ge25519.hpp"

#include <sodium.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <random>
#include <string_view>

namespace {
std::array<std::byte, 32> hex32(std::string_view h) {
    auto nib = [](char c) { return (c >= 'a') ? unsigned(c - 'a' + 10) : unsigned(c - '0'); };
    std::array<std::byte, 32> o;
    for (std::size_t i = 0; i < 32; ++i) o[i] = std::byte((nib(h[2*i]) << 4) | nib(h[2*i+1]));
    return o;
}
// Full 32-byte encoding of a P3 point (y little-endian + x sign bit), for tests only.
std::array<std::byte, 32> encode(const onion::crypto::GeP3& p) {
    return onion::crypto::ge_encode(p);
}
}

TEST_CASE("ge scalarmult_base matches the known a0*B vector") {
    using namespace onion::crypto;
    auto a0 = hex32("400000000000000000000000000000000000000000000000107e2d9c5ebf4a00");
    CHECK(encode(ge_scalarmult_base(a0)) ==
          hex32("494759ec2b42faec2b989685783762a15d87fb11dd2012f34765676cf96728d0"));
}

TEST_CASE("ge scalarmult_base(8) matches the 8B vector and equals 3 doublings of B") {
    using namespace onion::crypto;
    std::array<std::byte, 32> eight{};
    eight[0] = std::byte{8};
    CHECK(encode(ge_scalarmult_base(eight)) ==
          hex32("b4b937fca95b2f1e93e41e62fc3c78818ff38a66096fad6e7973e5c90006d321"));

    GeP3 b = ge_basepoint();
    GeP3 b8 = ge_double(ge_double(ge_double(b)));
    CHECK(encode(b8) == encode(ge_scalarmult_base(eight)));
}

TEST_CASE("ge scalarmult_base matches libsodium base_noclamp on random scalars") {
    using namespace onion::crypto;
    REQUIRE(sodium_init() >= 0);
    std::mt19937_64 rng(123);
    for (int t = 0; t < 200; ++t) {
        std::array<std::byte, 32> s;
        for (auto& x : s) x = std::byte(rng() & 0xff);
        s[31] = std::byte(std::to_integer<unsigned>(s[31]) & 0x7f);  // < 2^255

        std::array<unsigned char, 32> ref;
        REQUIRE(crypto_scalarmult_ed25519_base_noclamp(
                    ref.data(), reinterpret_cast<const unsigned char*>(s.data())) == 0);

        auto mine = encode(ge_scalarmult_base(s));
        // compare the full 32-byte encoding (y + sign)
        CHECK(std::memcmp(mine.data(), ref.data(), 32) == 0);
    }
}

TEST_CASE("ge_add via cached equals encode of point sum (P + 8B)") {
    using namespace onion::crypto;
    REQUIRE(sodium_init() >= 0);
    std::array<std::byte, 32> a0 = hex32("400000000000000000000000000000000000000000000000107e2d9c5ebf4a00");
    std::array<std::byte, 32> eight{}; eight[0] = std::byte{8};

    GeP3 A = ge_scalarmult_base(a0);
    GeCached eightB = ge_to_cached(ge_scalarmult_base(eight));
    GeP3 sum = ge_add(A, eightB);

    // reference: (a0 + 8) * B
    std::array<unsigned char, 32> a0p8;
    std::memcpy(a0p8.data(), a0.data(), 32);
    unsigned int carry = 8;
    for (int i = 0; i < 32 && carry; ++i) { unsigned s = a0p8[i] + (carry & 0xff); a0p8[i] = s & 0xff; carry = s >> 8; }
    std::array<unsigned char, 32> ref;
    REQUIRE(crypto_scalarmult_ed25519_base_noclamp(ref.data(), a0p8.data()) == 0);
    CHECK(std::memcmp(encode(sum).data(), ref.data(), 32) == 0);
}
```

- [ ] **Step 2: Run to verify it fails** — add `test_ge25519.cpp` to `tests/CMakeLists.txt` sources, then `cmake --build --preset debug`. Expected: FAIL — `crypto/ge25519.hpp` not found.

- [ ] **Step 3: Write the implementation**

`src/crypto/ge25519.hpp`:

```cpp
#pragma once

#include "crypto/fe25519.hpp"

#include <array>
#include <cstddef>
#include <span>

namespace onion::crypto {

struct GeP3 { Fe X, Y, Z, T; };                 // extended: x=X/Z, y=Y/Z, T=XY/Z
struct GeCached { Fe YplusX, YminusX, Z, T2d; }; // precomputed addend

[[nodiscard]] GeP3 ge_basepoint();
[[nodiscard]] GeP3 ge_double(const GeP3& p);
[[nodiscard]] GeCached ge_to_cached(const GeP3& p);
[[nodiscard]] GeP3 ge_add(const GeP3& p, const GeCached& q);
[[nodiscard]] GeP3 ge_scalarmult_base(std::span<const std::byte, 32> scalar);  // setup only

// Full 32-byte encoding (little-endian y with x's sign in the top bit). Used by
// tests and on a match; the hot loop uses ge_affine_y_bytes with a batched Z^-1.
[[nodiscard]] std::array<std::byte, 32> ge_encode(const GeP3& p);

}  // namespace onion::crypto
```

**Constant regeneration (do this if any Task-2 vector test fails — never hand-edit limbs):** the basepoint and `d` limbs below are emitted by this exact snippet; re-run it and paste the output rather than trusting a hand-typed constant:
```bash
python3 - <<'PY'
p=2**255-19; mask=(1<<51)-1
def inv(x): return pow(x,p-2,p)
d=(-121665*inv(121666))%p; By=(4*inv(5))%p
def xrec(y):
    xx=(y*y-1)*inv(d*y*y+1)%p; x=pow(xx,(p+3)//8,p)
    if (x*x-xx)%p!=0: x=x*pow(2,(p-1)//4,p)%p
    if x%2: x=p-x
    return x
Bx=xrec(By); T=(Bx*By)%p
for n,v in [("Bx",Bx),("By",By),("T",T),("d",d)]:
    print(n, "Fe{{"+", ".join(f"0x{(v>>(51*i))&mask:x}ULL" for i in range(5))+"}}")
PY
```

`src/crypto/ge25519.cpp` (point formulas: ref10 add/double sequences over the Task-1 fe ops; the libsodium cross-checks in Task 2 + the bit-exact chain test in Task 3 are the gate):

```cpp
#include "crypto/ge25519.hpp"

namespace onion::crypto {
namespace {

// d = -121665/121666 and 2d, as 5x51 limbs (verified in Python).
const Fe kD2 = [] {
    // 2*d mod p limbs (computed in Python: 2d = ...). We derive 2d from d limbs.
    // d limbs:
    Fe d{{0x34dca135978a3ULL, 0x1a8283b156ebdULL, 0x5e7a26001c029ULL,
          0x739c663a03cbbULL, 0x52036cee2b6ffULL}};
    return fe_add(d, d);
}();

}  // namespace

GeP3 ge_basepoint() {
    GeP3 b;
    b.X = Fe{{0x62d608f25d51aULL, 0x412a4b4f6592aULL, 0x75b7171a4b31dULL,
              0x1ff60527118feULL, 0x216936d3cd6e5ULL}};
    b.Y = Fe{{0x6666666666658ULL, 0x4ccccccccccccULL, 0x1999999999999ULL,
              0x3333333333333ULL, 0x6666666666666ULL}};
    b.Z = Fe{{1, 0, 0, 0, 0}};
    b.T = Fe{{0x68ab3a5b7dda3ULL, 0x0eea2a5eadbbULL, 0x2af8df483c27eULL,
              0x332b375274732ULL, 0x67875f0fd78b7ULL}};
    return b;
}

GeCached ge_to_cached(const GeP3& p) {
    return {fe_add(p.Y, p.X), fe_sub(p.Y, p.X), p.Z, fe_mul(p.T, kD2)};
}

// r = p + q  (p3 + cached -> p3), unified/complete Edwards addition.
GeP3 ge_add(const GeP3& p, const GeCached& q) {
    Fe a = fe_mul(fe_sub(p.Y, p.X), q.YminusX);
    Fe b = fe_mul(fe_add(p.Y, p.X), q.YplusX);
    Fe c = fe_mul(q.T2d, p.T);
    Fe d = fe_mul(p.Z, q.Z);
    d = fe_add(d, d);
    Fe e = fe_sub(b, a);
    Fe f = fe_sub(d, c);
    Fe g = fe_add(d, c);
    Fe h = fe_add(b, a);
    return {fe_mul(e, f), fe_mul(g, h), fe_mul(f, g), fe_mul(e, h)};
}

// r = 2p. Standard ext-coords doubling (uses only squarings + adds/subs).
GeP3 ge_double(const GeP3& p) {
    Fe a = fe_sq(p.X);
    Fe b = fe_sq(p.Y);
    Fe c = fe_add(fe_sq(p.Z), fe_sq(p.Z));   // 2*Z^2
    Fe na = fe_sub(Fe{{0,0,0,0,0}}, a);      // -A  (== p - a); fe_sub biases, ok
    Fe x1y1 = fe_add(p.X, p.Y);
    Fe e = fe_sub(fe_sub(fe_sq(x1y1), a), b);  // (X+Y)^2 - A - B
    Fe g = fe_add(na, b);                      // -A + B
    Fe f = fe_sub(g, c);                       // G - 2Z^2
    Fe h = fe_sub(na, b);                      // -A - B
    return {fe_mul(e, f), fe_mul(g, h), fe_mul(f, g), fe_mul(e, h)};
}

GeP3 ge_scalarmult_base(std::span<const std::byte, 32> scalar) {
    // Simple double-and-add over the hardcoded basepoint. Setup-only (once per
    // epoch), so constant-time / windowing is unnecessary.
    GeP3 r = {Fe{{0,0,0,0,0}}, Fe{{1,0,0,0,0}}, Fe{{1,0,0,0,0}}, Fe{{0,0,0,0,0}}};  // identity
    GeCached base = ge_to_cached(ge_basepoint());
    // process bits MSB-first
    for (int byte = 31; byte >= 0; --byte) {
        unsigned b = std::to_integer<unsigned>(scalar[std::size_t(byte)]);
        for (int bit = 7; bit >= 0; --bit) {
            r = ge_double(r);
            if ((b >> bit) & 1) r = ge_add(r, base);
        }
    }
    return r;
}

std::array<std::byte, 32> ge_encode(const GeP3& p) {
    Fe zinv = fe_invert(p.Z);
    Fe x = fe_mul(p.X, zinv);
    Fe y = fe_mul(p.Y, zinv);
    std::array<std::byte, 32> out;
    fe_to_bytes(out, y);
    std::array<std::byte, 32> xb;
    fe_to_bytes(xb, x);
    // set top bit = low bit of x (the sign)
    out[31] = std::byte(std::to_integer<unsigned>(out[31]) |
                        ((std::to_integer<unsigned>(xb[0]) & 1) << 7));
    return out;
}

}  // namespace onion::crypto
```

(Note: the `By` limb `0x4cccccccccccc` is masked to 51 bits in the literal above — the Python value is `0x4cccccccccccc`; if the build warns about the literal width, write it as the bare `0x4ccccccccccccULL`. The Task-2 vector test will fail loudly if any basepoint limb is wrong.)

`src/crypto/CMakeLists.txt`: add `ge25519.cpp` to `onion_crypto` sources.
`tests/CMakeLists.txt`: add `test_ge25519.cpp` to sources (onion_crypto already links libsodium, so `<sodium.h>` is available).

- [ ] **Step 4: Run tests** — `cmake --build --preset debug && ctest --preset debug`. Expected: all pass, including the 200-scalar libsodium cross-check and the `P+8B` check. A failure here means a point-formula or basepoint-constant bug — fix against libsodium, never weaken the test.

- [ ] **Step 5: Commit**
```bash
git add -A
git commit -m "feat(crypto): ge25519 extended-coord point ops + fixed-base scalarmult (libsodium-validated)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Incremental stepper + scalar reconstruction + the cross-validation gate (`onion_crypto`)

The batched chain: walk `A₀, A₀+8B, …` keeping points projective, recover affine `y` for a whole batch with one inversion (Montgomery's trick), and reconstruct the matching scalar `a₀+8i`. **This task's libsodium cross-validation is the project's correctness keystone.**

**Files:**
- Create: `src/crypto/incremental.hpp`, `src/crypto/incremental.cpp`
- Modify: `src/crypto/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `tests/test_incremental.cpp` (links libsodium)

- [ ] **Step 1: Write the failing test — `tests/test_incremental.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "crypto/incremental.hpp"

#include <sodium.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <random>

TEST_CASE("incremental y-bytes match libsodium base_noclamp(a0+8i) for the whole batch") {
    using namespace onion::crypto;
    REQUIRE(sodium_init() >= 0);
    std::mt19937_64 rng(2024);

    for (int trial = 0; trial < 20; ++trial) {
        std::array<std::byte, 32> a0;
        for (auto& x : a0) x = std::byte(rng() & 0xff);
        a0[0] = std::byte(std::to_integer<unsigned>(a0[0]) & 0xf8);   // clamp low 3 bits
        a0[31] = std::byte((std::to_integer<unsigned>(a0[31]) & 0x7f) | 0x40);

        constexpr std::size_t N = 1024;
        IncrementalStepper stepper(a0);
        std::array<std::array<std::byte, 32>, N> ybatch;
        stepper.next_batch(ybatch);

        // reference scalar a = a0; check each i
        std::array<unsigned char, 32> a;
        std::memcpy(a.data(), a0.data(), 32);
        for (std::size_t i = 0; i < N; ++i) {
            std::array<unsigned char, 32> ref;
            REQUIRE(crypto_scalarmult_ed25519_base_noclamp(ref.data(), a.data()) == 0);
            // compare the leading 31 bytes (pure y; byte 31 differs by the sign bit)
            CHECK(std::memcmp(ybatch[i].data(), ref.data(), 31) == 0);
            // advance reference scalar by 8
            unsigned int carry = 8;
            for (int k = 0; k < 32 && carry; ++k) { unsigned s = a[k] + (carry & 0xff); a[k] = s & 0xff; carry = s >> 8; }
        }
    }
}

TEST_CASE("scalar_add_8i reconstructs a0 + 8*i") {
    using namespace onion::crypto;
    std::array<std::byte, 32> a0{};
    a0[0] = std::byte{0x10};
    auto s = scalar_add_8i(a0, 3);  // +24 -> 0x28
    CHECK(std::to_integer<unsigned>(s[0]) == 0x28);

    std::array<std::byte, 32> b{};
    b[0] = std::byte{0xf8};
    auto s2 = scalar_add_8i(b, 1);  // 0xf8 + 8 = 0x100 -> byte0=0, byte1=1
    CHECK(std::to_integer<unsigned>(s2[0]) == 0x00);
    CHECK(std::to_integer<unsigned>(s2[1]) == 0x01);
}
```

- [ ] **Step 2: Run to verify it fails** — add `test_incremental.cpp` to `tests/CMakeLists.txt` sources, then build. Expected: FAIL — `crypto/incremental.hpp` not found.

- [ ] **Step 3: Write the implementation**

`src/crypto/incremental.hpp`:

```cpp
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
```

`src/crypto/incremental.cpp`:

```cpp
#include "crypto/incremental.hpp"

#include <vector>

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
    // Collect points and their Z, then Montgomery batch-invert the Z values.
    std::vector<GeP3> pts(n);
    std::vector<Fe> z(n), prefix(n);
    for (std::size_t i = 0; i < n; ++i) {
        pts[i] = cur_;
        z[i] = cur_.Z;
        cur_ = ge_add(cur_, step8b_);
    }
    // prefix products: prefix[i] = z[0]*...*z[i]
    prefix[0] = z[0];
    for (std::size_t i = 1; i < n; ++i) prefix[i] = fe_mul(prefix[i - 1], z[i]);
    Fe inv = fe_invert(prefix[n - 1]);  // 1 / (z[0]*...*z[n-1])
    // back-substitute: zinv[i] = inv * prefix[i-1]; inv *= z[i]
    for (std::size_t i = n; i-- > 0;) {
        Fe zinv = (i == 0) ? inv : fe_mul(inv, prefix[i - 1]);
        Fe y = fe_mul(pts[i].Y, zinv);
        fe_to_bytes(out[i], y);
        if (i != 0) inv = fe_mul(inv, z[i]);
    }
    consumed_ += n;
}

}  // namespace onion::crypto
```

`src/crypto/CMakeLists.txt`: add `incremental.cpp`.
`tests/CMakeLists.txt`: add `test_incremental.cpp`.

- [ ] **Step 4: Run tests** — `cmake --build --preset debug && ctest --preset debug`, then ALSO `cmake --preset asan && cmake --build --preset asan && ctest --preset asan`. Expected: all pass in both. The cross-validation test compares 20×1024 incremental points against libsodium — if it fails, a field/point bug exists; fix it against libsodium, never weaken the comparison.

- [ ] **Step 5: Commit**
```bash
git add -A
git commit -m "feat(crypto): incremental batched stepper, scalar reconstruction, libsodium cross-validation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `IncrementalCpuEngine` (`onion_engine_cpu`)

Drives steppers across threads, matches leading `y`-bytes, emits verified candidates with a fresh PRF prefix.

**Files:**
- Create: `src/engine/cpu/incremental_engine.hpp`, `src/engine/cpu/incremental_engine.cpp`
- Modify: `src/engine/cpu/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `tests/test_incremental_engine.cpp`

- [ ] **Step 1: Write the failing test — `tests/test_incremental_engine.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "engine/cpu/incremental_engine.hpp"
#include "core/matcher.hpp"
#include "crypto/keys.hpp"
#include "io/verifier.hpp"

#include <chrono>
#include <stop_token>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

TEST_CASE("incremental engine finds a 3-char prefix and the candidate verifies") {
    using namespace onion;
    auto pat = core::compile_prefix("abc");
    REQUIRE(pat.has_value());
    std::vector patterns{*pat};

    engine::ResultQueue queue;
    engine::StatsBoard stats(4);
    engine::IncrementalCpuEngine eng(patterns, 4, queue, stats);

    std::stop_source stop;
    std::jthread runner([&] { eng.run(stop.get_token()); });

    auto cand = queue.pop_wait_for(30s);  // ~32768 expected tries
    stop.request_stop();
    REQUIRE(cand.has_value());

    // the firewall must accept it (independent libsodium re-derivation)
    const auto verified = io::verify(*cand, patterns);
    REQUIRE(verified.has_value());
    CHECK(verified->address.view().starts_with("abc"));

    runner.join();
    CHECK(stats.total() > 0);
}

TEST_CASE("incremental engine honors stop_token promptly") {
    using namespace onion;
    auto pat = core::compile_prefix("zzzzzzzzzzzz");  // 12 chars: never matches in-test
    REQUIRE(pat.has_value());
    std::vector patterns{*pat};
    engine::ResultQueue queue;
    engine::StatsBoard stats(4);
    engine::IncrementalCpuEngine eng(patterns, 4, queue, stats);

    std::stop_source stop;
    std::jthread runner([&] { eng.run(stop.get_token()); });
    std::this_thread::sleep_for(100ms);
    const auto t0 = std::chrono::steady_clock::now();
    stop.request_stop();
    runner.join();
    CHECK(std::chrono::steady_clock::now() - t0 < 2s);
}
```

- [ ] **Step 2: Run to verify it fails** — add to `tests/CMakeLists.txt`, build. Expected: FAIL — header not found.

- [ ] **Step 3: Write the implementation**

`src/engine/cpu/incremental_engine.hpp`:

```cpp
#pragma once

#include "core/matcher.hpp"
#include "engine/engine.hpp"

#include <cstddef>
#include <stop_token>
#include <vector>

namespace onion::engine {

// Phase 1 engine: per-worker incremental ed25519 search (A += 8B + batched
// inversion). ~100-250x the naive engine. Same IEngine seam.
class IncrementalCpuEngine final : public IEngine {
public:
    IncrementalCpuEngine(std::vector<core::CompiledPattern> patterns, unsigned num_threads,
                         ResultQueue& results, StatsBoard& stats, std::size_t batch = 1024,
                         std::uint64_t epoch_candidates = 1u << 22)
        : patterns_(std::move(patterns)), num_threads_(num_threads), results_(results),
          stats_(stats), batch_(batch), epoch_candidates_(epoch_candidates) {}

    void run(std::stop_token stop) override;

private:
    void worker(std::stop_token stop, unsigned index);

    std::vector<core::CompiledPattern> patterns_;
    unsigned num_threads_;
    ResultQueue& results_;
    StatsBoard& stats_;
    std::size_t batch_;
    std::uint64_t epoch_candidates_;
};

}  // namespace onion::engine
```

`src/engine/cpu/incremental_engine.cpp`:

```cpp
#include "engine/cpu/incremental_engine.hpp"

#include "crypto/incremental.hpp"
#include "crypto/keys.hpp"

#include <sodium.h>

#include <array>
#include <span>
#include <thread>
#include <vector>

namespace onion::engine {

void IncrementalCpuEngine::run(std::stop_token stop) {
    std::vector<std::jthread> workers;
    workers.reserve(num_threads_);
    for (unsigned i = 0; i < num_threads_; ++i)
        workers.emplace_back([this, stop, i] { worker(stop, i); });
}

void IncrementalCpuEngine::worker(std::stop_token stop, unsigned index) {
    std::vector<std::array<std::byte, 32>> ybatch(batch_);
    std::uint64_t local_total = 0;

    while (!stop.stop_requested()) {
        // New epoch: fresh clamped base scalar.
        std::array<std::byte, 32> a0;
        crypto::random_bytes(a0);
        a0[0] = std::byte(std::to_integer<unsigned>(a0[0]) & 0xf8);
        a0[31] = std::byte((std::to_integer<unsigned>(a0[31]) & 0x7f) | 0x40);

        crypto::IncrementalStepper stepper(a0);
        const std::uint64_t batches = epoch_candidates_ / batch_;

        for (std::uint64_t b = 0; b < batches && !stop.stop_requested(); ++b) {
            const std::uint64_t base_index = stepper.consumed();
            stepper.next_batch(std::span<std::array<std::byte, 32>>{ybatch});
            for (std::size_t k = 0; k < batch_; ++k) {
                std::span<const std::byte, 32> y{ybatch[k]};
                for (std::size_t p = 0; p < patterns_.size(); ++p) {
                    if (core::matches(patterns_[p], y)) {
                        auto scalar = crypto::scalar_add_8i(a0, base_index + k);
                        crypto::ExpandedSecretKey secret;
                        secret.scalar = scalar;
                        crypto::random_bytes(secret.prf_prefix);  // fresh RH (design §9)
                        auto pk = crypto::pubkey_from_scalar(secret.scalar);
                        if (pk) results_.push({secret, *pk, p});
                    }
                }
            }
            local_total += batch_;
            stats_.set(index, local_total);
        }
        sodium_memzero(a0.data(), a0.size());
    }
}

}  // namespace onion::engine
```

(The `next_batch(std::span<...>)` overload used here is defined in Task 3's `incremental.hpp`; `ybatch` is a `std::vector<std::array<std::byte,32>>` of size `batch_`.)

`src/engine/cpu/CMakeLists.txt` (full new content):
```cmake
add_library(onion_engine_cpu STATIC
  naive_engine.cpp
  incremental_engine.cpp
)
target_link_libraries(onion_engine_cpu
  PUBLIC onion_engine onion_core
  PRIVATE onion_crypto onion_options Threads::Threads)
```
`tests/CMakeLists.txt`: add `test_incremental_engine.cpp` to sources (onion_io already linked from Phase 0).

- [ ] **Step 4: Run tests** — debug then asan: `cmake --build --preset debug && ctest --preset debug` and `cmake --build --preset asan && ctest --preset asan`. Expected: all pass, the found candidate verifies through the firewall. Also do a one-off TSan build of the engine tests (per Phase 0 practice) and confirm clean.

- [ ] **Step 5: Commit**
```bash
git add -A
git commit -m "feat(engine): incremental CPU engine (A+=8B, batched inversion) behind IEngine

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: CLI engine selection + `onion bench` + measurement

Wire the new engine into the CLI (default), keep `--engine naive` for cross-checking, add a throughput benchmark, extend the e2e test to the incremental engine, and record the measured speedup.

**Files:**
- Modify: `src/cli/main.cpp`, `tests/e2e/run_e2e.sh`, `tests/CMakeLists.txt`
- Test: extend e2e

- [ ] **Step 1: Add `--engine` and a `bench` mode to `src/cli/main.cpp`**

After the existing option declarations add:
```cpp
    std::string engine_name = "incremental";
    double bench_seconds = 0.0;
    app.add_option("--engine", engine_name, "engine: incremental (default) or naive");
    app.add_option("--bench", bench_seconds, "benchmark: run N seconds against an impossible prefix, report keys/s");
```
Replace the construction of the engine
```cpp
    onion::engine::NaiveCpuEngine engine(patterns, threads, queue, stats);
```
with a factory that selects by name (include `engine/cpu/incremental_engine.hpp`):
```cpp
    std::unique_ptr<onion::engine::IEngine> engine;
    if (engine_name == "naive")
        engine = std::make_unique<onion::engine::NaiveCpuEngine>(patterns, threads, queue, stats);
    else
        engine = std::make_unique<onion::engine::IncrementalCpuEngine>(patterns, threads, queue, stats);
```
and change `engine.run(...)` to `engine->run(...)`. Add `#include <memory>`.

For `--bench`: if `bench_seconds > 0`, after spawning the runner, sleep that long, read `stats.total()`, print `keys/s`, request stop, and exit 0 — bypassing the match/verify loop. Insert near the top of the run loop:
```cpp
    if (bench_seconds > 0.0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(bench_seconds));
        const double secs = std::chrono::duration<double>(clock::now() - start).count();
        std::println("bench: {:.2f} M keys/s ({} candidates in {:.1f}s, {} threads, {})",
                     double(stats.total()) / secs / 1e6, stats.total(), secs, threads, engine_name);
        stop.request_stop();
        return 0;
    }
```
(Run `--bench` against a long impossible prefix so nothing matches, e.g. `onion zzzzzzzzzzzzzzzz --bench 10`.)

- [ ] **Step 2: Build and measure both engines**
```bash
cmake --build --preset release
./build/release/src/cli/onion zzzzzzzzzzzzzzzz --bench 10 --engine naive -t 12
./build/release/src/cli/onion zzzzzzzzzzzzzzzz --bench 10 --engine incremental -t 6
./build/release/src/cli/onion zzzzzzzzzzzzzzzz --bench 10 --engine incremental -t 12
```
Record the three numbers. Expected: incremental ≫ naive (target ≥ 30 M keys/s aggregate on 6 cores; ≥ 100× the naive number). Note whether 6 or 12 threads is faster (design §7 hypothesis).

- [ ] **Step 3: Extend the e2e test — `tests/e2e/run_e2e.sh`** (parameterize the engine; default already exercises incremental)

Change the search line to use the incremental engine explicitly and prove the firewall+oracle still pass end-to-end:
```bash
"$BIN" ab --count 1 --threads 2 --out "$OUT" --quiet --engine incremental
```
(The rest of the script — dir check, oracle validation — is unchanged.)

- [ ] **Step 4: Run the full suite** — `ctest --preset debug && ctest --preset asan`. Expected: all pass (e2e now finds `ab` via the incremental engine and the Python oracle validates it).

- [ ] **Step 5: Update README and commit**

Update `README.md`: change "Status: Phase 0" to note the incremental engine is the default, replace the speed table with the measured numbers from Step 2, and add the `--engine`/`--bench` flags to usage.

```bash
git add -A
git commit -m "feat(cli): default to incremental engine; add --engine and --bench; measured speedup

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Definition of Done (Phase 1)

- [ ] All field/point/incremental ops cross-validated bit-for-bit against libsodium (Tasks 1–3); `ctest` green in debug + asan.
- [ ] `IncrementalCpuEngine` finds prefixes and every emitted candidate passes the verifier firewall and the Python oracle (e2e).
- [ ] `onion --bench` shows ≥ 100× the naive engine on the same machine; number recorded in README.
- [ ] Manual acceptance: generate a 5–6 char vanity address with the incremental engine in seconds–minutes, oracle-validated.

## Beyond Phase 1 (future, not in this plan)

Per design §5/§6, the next multipliers stack on this foundation: **AVX2** (4 candidates/lane-group, ~2–3×), **AVX-512 IFMA** (52-bit limbs, the known-fastest x86 path), and **CUDA** (one lane/thread, ~1–2 G/s — turns 8-char prefixes from hours into minutes). Each reuses the matcher/verifier/writer/oracle and is gated on this engine being correct and measured.
