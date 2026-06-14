# CLAUDE.md

Guidance for an AI assistant (Claude or other models) working in this repo.
Keep it load-bearing; the source code is the source of truth. Verify every
claim in the code before relying on it — do not invent architecture.

## Read this first (the protocol)

**THE CARDINAL RULE — bit-exactness vs libsodium is non-negotiable.**
Every field/point/engine/kernel change MUST stay bit-for-bit identical to
libsodium. The test suite cross-validates the whole search chain against
`crypto_scalarmult_ed25519_base_noclamp(a₀ + 8i)`. A fast-but-wrong generator
is worthless: it emits unusable keys.

- **Never weaken a cross-check** to make a test pass. If a cross-validation
  test fails, the new code is wrong, not the test.
- **The firewall runs before every write.** `io::verify` (`src/io/verifier.hpp`)
  independently re-derives the pubkey via libsodium and re-checks the match
  before any key reaches disk. Match ≠ result (design §1). Never route output
  around it.
- **`tools/verify_onion.py`** is an independent pure-Python oracle (a second
  opinion, not sharing our code). The e2e test and `oracle_self_test` run it.
- **Verify claims in source, don't invent.** If a fact isn't in the code or
  `docs/design.md`, don't assume it. Check the actual file path.

## Product

A high-performance **Tor v3 vanity `.onion` address generator** in modern
C++23 (CPU + CUDA). Give it a prefix; it searches ed25519 keypairs until an
address starts with it, then writes key material in the exact format Tor
consumes. See [`README.md`](README.md) (pitch + benchmarks) and
[`docs/design.md`](docs/design.md) (full design review).

## How it's fast (the one algorithm that matters)

Naive: one full scalar mult (`A = a·B`, ~250 point ops) per candidate.
Chosen (the `mkp224o` approach): pick one random clamped base scalar `a₀` per
worker/epoch, compute `A₀ = a₀·B` once, then walk `Aᵢ₊₁ = Aᵢ + 8B` —
**one point addition per candidate**. `+8` (not `+1`) preserves the clamp
invariant (low 3 scalar bits zero). The address needs affine `y = Y·Z⁻¹`, so a
whole batch's `Z` values are inverted with **one** field inversion via
Montgomery's trick (~3.3 mults/candidate amortized at batch 1024). The leading
pubkey bytes are matched against a precompiled byte+mask pattern; SHA3 +
base32 are off the hot path (run only on a rare match). Full rationale:
`docs/design.md` §0.

## Layout — lib layering behind the `IEngine` seam

Libraries stack bottom-up; engines are peers behind one narrow interface.

```
src/crypto/   (onion_crypto)  field/point/scalar arithmetic + hashing
  fe25519.{hpp,cpp}        GF(2²⁵⁵−19), 5×51-bit limbs (the perf kernel)
  ge25519.{hpp,cpp}        twisted-Edwards points, mixed addition, 8B step
  incremental.{hpp,cpp}    IncrementalStepper: A+=8B, batched inversion, y recovery
  keys.{hpp,cpp}           ExpandedSecretKey (scalar+RH), expand_seed, pubkey_from_scalar
  sha3.{hpp,cpp}           vendored Keccak-f[1600] (libsodium has NO SHA3) — cold path
  fe25519x4 / ge25519x4 / incremental_x4   AVX2 4-wide (__m256i SoA) variants
src/core/     (onion_core)   off-hot-path string/match logic
  base32.{hpp,cpp}         a–z2–7 codec
  onion_address.{hpp,cpp}  rend-spec-v3 address build + inverse (validates checksum)
  matcher.{hpp,cpp}        CompiledPattern (bytes+mask), compile_prefix, matches() (≤49 chars)
src/engine/   (onion_engine) the seam + implementations
  engine.hpp               IEngine, MatchCandidate, ResultQueue, StatsBoard  <-- the seam
  cpu/naive_engine          reference: libsodium per candidate (slow, correct)
  cpu/incremental_engine     DEFAULT CPU engine (A+=8B + batch inversion)
  cpu/incremental_engine_x4  AVX2 4-wide CPU engine (--simd on)
  cuda/cuda_engine.{hpp,cu}  GPU engine (--engine cuda)
  cuda/device_field32.cuh    native 8×32-bit device field (DEFAULT on device)
  cuda/device_field.cuh      __int128 device field (slower; kept for reference)
  cuda/device_field_select.cuh  selects which device field to compile
  cuda/search_kernel.cu      the interleaved-chain search kernel
  cuda/xval_kernel.cu        device-vs-libsodium cross-validation kernel
src/io/       (onion_io)
  verifier.{hpp,cpp}       THE FIREWALL: io::verify re-derives pubkey before write
  tor_key_writer.{hpp,cpp} writes hostname / hs_ed25519_{secret,public}_key (dir 0700, files 0600, O_EXCL)
src/cli/main.cpp           arg parsing (CLI11), engine selection, monitor loop, SIGINT
```

**The seam (`src/engine/engine.hpp`):** `IEngine::run(std::stop_token)` runs
once per run (virtual dispatch is fine here). Workers report a
`MatchCandidate { secret, claimed_pubkey, pattern_index }` into a mutex+condvar
`ResultQueue`; throughput goes through a cache-line-padded `StatsBoard`
(relaxed `atomic_ref`, no RMW). A candidate is NOT a result until `io::verify`
passes.

## Build & test

CPU presets need GCC 14+ (targets GCC 16), CMake ≥ 3.28, libsodium, Python 3.
The [mold](https://github.com/rui314/mold) linker is auto-used when present.

```sh
# CPU
cmake --preset release   # -O3 -march=native -flto=auto -fno-semantic-interposition
cmake --build --preset release       # binary: build/release/src/cli/onion
ctest --preset release               # or: --preset debug  (53 tests)
cmake --preset debug && ctest --preset debug    # 53 tests
cmake --preset asan  && ctest --preset asan     # ASan+UBSan

# CUDA  (only this preset enables nvcc; CPU presets are unaffected)
cmake --preset cuda && cmake --build --preset cuda
ctest --preset cuda                  # 56 tests (= 53 + 3 GPU: cuda.* prefix + e2e)
```

**CUDA requirements (pinned, hard-won):** `nvcc` (`/opt/cuda/bin/nvcc`) with
host compiler **g++-15** (`CMAKE_CUDA_HOST_COMPILER`; nvcc rejects GCC 16),
targeting **sm_75** (`CMAKE_CUDA_ARCHITECTURES 75`). All set in the `cuda`
preset and top `CMakeLists.txt` under `if(ONION_CUDA)`. Without a CUDA
toolchain the CUDA target and its tests are simply not built.

CLI flags (see `src/cli/main.cpp`): `-o/--out`, `-t/--threads`, `-n/--count`,
`-q/--quiet`, `--engine {naive|incremental|cuda}`, `--simd {on|off|auto}`,
`--batch N` (default 1024), `--bench SECONDS` (run against an impossible prefix,
report keys/s). User-facing command guide (Russian): [`USAGE.md`](USAGE.md).

## Conventions

- **C++23.** Single namespace tree `onion::{crypto,core,engine,io}`. Prefer
  `std::expected`/`std::optional` for fallible returns (see `matcher.hpp`,
  `onion_address.hpp`, `verifier.hpp`), `std::span` over raw ptr+len,
  `std::byte` for key material, `std::stop_token` for shutdown.
- **TDD + cross-validation discipline.** Each unit has a Catch2 test in
  `tests/` (registered in `tests/CMakeLists.txt`). Crypto tests cross-check
  against libsodium KATs and `base_noclamp(a₀+8i)`; do not add a fast path
  without a matching cross-validation test.
- **Comments are short and factual** — explain *why*, not what the code says.
- **No speculative scaffolding.** Don't add abstractions a task doesn't need.

### Adding a new engine (behind `IEngine`)

1. Implement `IEngine::run(std::stop_token)` in a new TU under
   `src/engine/{cpu,cuda}/`; report hits as `MatchCandidate`, update the
   `StatsBoard` at batch granularity (≥ ~2¹⁰ candidates), never per candidate.
2. **It MUST cross-validate.** Add a test that proves the engine's device/
   field chain is bit-exact vs libsodium `base_noclamp(a₀+8i)` (model the GPU
   path on `tests/test_cuda_xval.cpp`; the CPU path on `tests/test_incremental*`).
3. **It MUST pass the firewall.** Every emitted candidate goes through
   `io::verify` before `tor_key_writer` — drive the engine through the seam in
   a test the way `tests/test_cuda_engine.cpp` does.
4. Wire selection in `src/cli/main.cpp` and registration in the CMake target;
   keep CUDA behind `if(ONION_CUDA)` so CPU presets never need nvcc.

## Gotchas / architectural pins (hard-won — do not regress)

- **`fe_sub` MUST carry-reduce its 8p-biased output.** It biases by 8p to stay
  non-negative; without a following carry-reduce, a worst-case bias-form zero
  (e.g. `fe_sub(1,1)`) overflows `fe_mul`'s `o0 += c*19` u64 step and silently
  drifts the identity over repeated doublings. See the comment at
  `src/crypto/fe25519.cpp` (~line 66). This bit us once.
- **AVX2 4-wide is SLOWER on Zen 2.** The 256-bit units are two fused 128-bit
  µops and a 4-lane point spills the register file. Scalar `incremental` is the
  default; `--simd auto` stays scalar here. `--simd on` forces x4 (wins on
  Zen 4+/Intel AVX-512-class cores). Kept, not deleted.
- **CUDA: prefer the native 32-bit field, not `__int128`.** `__int128` works on
  device but is emulated/slow on Turing (fast 32-bit IMAD, emulated 64-bit
  mul). `device_field32.cuh` cut the kernel from 178 → **~113 regs/thread at the
  committed M=3072 (0 spills; ~96 at the early M=256/1024 sweep)**, raising
  occupancy. Combined with the fused 2-array batch inversion and the M=3072 /
  double-buffered pipeline (see `docs/design.md` §6), throughput rose ~275 →
  **~390 M/s**. `device_field.cuh` is the slower reference. The ~390 M/s ceiling
  is **integer-ALU / `fed_mul`-issue bound (and pinned at the 50 W power cap)**,
  NOT local-memory-bandwidth bound — every memory-traffic-cutting experiment lost
  throughput; see `docs/devlog/gpu-optimization-research.md` for the on-card
  profiling that established this.
- **CUDA host compiler is g++-15** (nvcc rejects GCC 16). Target sm_75.
- **NEVER commit secret keys.** `keys/` and the `.claude` session dirs are
  gitignored. Generated `hs_ed25519_secret_key` files are real secrets.
- **SHA3 is vendored** (`src/crypto/sha3.cpp`) — libsodium provides SHA-512 but
  no SHA3-256, which Tor's checksum needs.

## Performance snapshot (measured)

Release `-O3 -march=native -flto`, AMD Ryzen 5 4600H (Zen 2, 6C/12T) + NVIDIA
GTX 1650 (Turing, sm_75, 14 SMs, 4 GB):

| Engine | Flags | Threads | Keys/s |
|---|---|---|---|
| naive (libsodium per candidate) | | 12 | ~0.34 M/s |
| **incremental** (`A+=8B` + batch inversion) — CPU default | | 12 | **~25–28 M/s** |
| incremental + AVX2 4-wide | `--simd on` | 12 | ~21 M/s (slower on Zen 2) |
| **CUDA** (interleaved chains, M=3072, native 32-bit field) | `--engine cuda` | GPU | **~390 M/s** |

**vs `mkp224o`** (same machine): mkp224o CPU (amd64-51-30k, 12t) ≈ 28.9 M/s;
cpp_onion CPU incremental 12t ≈ 28.1 M/s (on par). mkp224o has no GPU backend;
cpp_onion CUDA ≈ 390 M/s ≈ **~13–15× the 12-thread CPU** and ~13× mkp224o
on this GPU. The design's optimistic ~10⁹ assumes a far larger GPU; honest
numbers here are GPU-bound. The GPU build is leak-checked
(`compute-sanitizer --tool memcheck --leak-check full`: 0 bytes / 0 errors).
Search cost is ~`32^L` for an `L`-char prefix: ≤6 chars seconds, 7 minutes,
8 hours.

## Pointers

- [`docs/design.md`](docs/design.md) — full design review (§0 algorithm, §1
  architecture, §6 GPU, §9 crypto/firewall, §16 testing, §22 roadmap).
- [`USAGE.md`](USAGE.md) — user command guide (Russian).
- [`README.md`](README.md) — pitch + the live benchmark table.
- [`docs/devlog/plans/`](docs/devlog/plans) — phase history (phase0 reference
  generator, phase1 incremental engine, phase2 AVX2 4-wide, phase3 CUDA).
- `tools/verify_onion.py` — independent Python oracle (run with `--self-test`).
```
