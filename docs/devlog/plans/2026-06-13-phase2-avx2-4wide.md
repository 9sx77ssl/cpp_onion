# Phase 2: AVX2 4-wide Batched Field — Implementation Plan

> Execute task-by-task. Each step is gated by **bit-exact cross-validation against the already-validated scalar path / libsodium**. A fast-but-wrong generator is worthless — never weaken a cross-check.

**Goal:** Process **4 independent candidates per worker in lockstep** using AVX2, ~+60–100% over the scalar incremental engine on Zen 2 (honest estimate; Zen 2 runs 256-bit AVX2 as 2×128 and has no native `u64×u64→u128`, so we use 32-bit-limb `vpmuludq`).

**Architecture (additive — does NOT touch the validated scalar path):**
- `crypto/fe25519x4` — a 4-lane field element `Fe4 = __m256i[10]` in **radix 2²⁵·⁵ (10 unsigned limbs, donna-32 layout)**: lane `k` (0..3) of `v[i]` holds limb `i` of candidate `k` as a 64-bit value. Ops: `mul, sq, add, sub, mul121666(n/a), invert`, plus `load4` (4×32-byte → SoA) and `store4_y` (extract the 4 little-endian y encodings). This is **agl's curve25519-donna 32-bit `fmul`/`fsquare`/`freduce` vectorized 4-wide** — same 10×10 schoolbook, same 2²⁵⁵≡19 folding, same carry chain, with each scalar `int64` op replaced by the `__m256i` equivalent (`vpmuludq` for 32×32→64 products, `vpaddq`, `vpsrlq`/`vpand` for per-lane carries — carries stay within a lane along the limb axis; lanes never interact).
- `crypto/ge25519x4` — `GeP3x4 {Fe4 X,Y,Z,T}`, `GeCachedAffinex4`; `ge_madd_x4` (the hot op: same formula as scalar `ge_madd`, vectorized). Setup (A0 per lane, 8B) reuses the **scalar** `ge_scalarmult_base` per lane then packs into SoA — setup is once per epoch, no need to vectorize it.
- `crypto/incremental_x4` — `IncrementalStepperX4`: 4 independent base scalars `a0[0..3]`; `next_batch` walks all 4 lanes via `ge_madd_x4`, 4-wide Montgomery batch inversion (per-lane prefix products + one 4-wide `fe_invert` + back-substitute), emits 4×N y-byte encodings.
- `engine/cpu/incremental_engine.cpp` — add a `--simd` path (or auto-detect) where each worker drives an `IncrementalStepperX4` (4 lanes), matching all 4 y-streams. Keep the scalar engine as fallback (CPUs without AVX2, and for cross-checking).

**Cross-validation gates (the whole point):**
1. `fe25519x4`: for 4 random field elements, `fe_mul_x4(a,b)` lane `k` == scalar `fe_mul(a_k,b_k)` (and same for sq/add/sub/invert/load/store). Run ≥1000 random vectors.
2. `ge_madd_x4`: 4 random points + affine 8B, lane `k` == scalar `ge_madd`.
3. `IncrementalStepperX4`: for 4 random clamped `a0`, lane `k`, step `i`, the y-bytes == `libsodium crypto_scalarmult_ed25519_base_noclamp(a0_k + 8i)` leading 31 bytes (the same keystone test as scalar, ×4 lanes). **This is the master gate.**
4. Engine: a found candidate from the SIMD engine passes `io::verify` (libsodium firewall).
5. asan/ubsan green; full `ctest` green.

**Tasks (each: write failing cross-check test → implement → pass → commit):**

### Task 1 — `fe25519x4` field arithmetic
Files: `src/crypto/fe25519x4.{hpp,cpp}`, `tests/test_fe25519x4.cpp`. Add `-mavx2` to the crypto lib (or guard the file). API mirrors scalar `Fe` but 4-wide. Test = gate #1 (every op vs scalar `fe25519`, 1000+ random vectors, including the radix-2²⁵·⁵ load/store transpose). The donna-32 10-limb `fmul`/`fsquare` is the template; if a vector mismatches, the bug is in the vectorized carry/fold — fix against the scalar reference, never weaken the test.

### Task 2 — `ge25519x4` point ops
Files: `src/crypto/ge25519x4.{hpp,cpp}`, `tests/test_ge25519x4.cpp`. `GeP3x4`, `GeCachedAffinex4`, `ge_madd_x4`, pack/unpack helpers (scalar GeP3 ↔ lane). Test = gate #2.

### Task 3 — `IncrementalStepperX4` + master cross-validation
Files: `src/crypto/incremental_x4.{hpp,cpp}`, `tests/test_incremental_x4.cpp`. 4-wide batch inversion. Test = gate #3 (vs libsodium, 4 lanes). **The keystone.**

### Task 4 — SIMD engine path + measure
Modify `src/engine/cpu/incremental_engine.cpp` (or add `IncrementalCpuEngineX4`) + `--simd auto|on|off` CLI flag (default: auto = use x4 when AVX2 present). Worker drives the x4 stepper, matches 4 lanes, emits verified candidates. Test = gate #4 + stop test. Then `--bench 8 -t 12` A/B vs the scalar engine; record the speedup in README. Target ≥ 1.5× (accept and document if Zen 2 yields less).

**Build note:** AVX2 intrinsics need `-mavx2` (already covered by `-march=native` in release, but the x4 TU must also get it in debug/asan — add `target_compile_options(... -mavx2)` to the file or crypto lib, guarded so non-x86 still builds the scalar path).

**Fallback if 4-wide AVX2 underperforms or proves too costly to land cleanly:** a 2-way *scalar* interleave (two independent steppers per worker, manually interleaved to expose ILP) is lower-risk and still hides some `fe_mul` latency — document the measured result either way.
