# Engineering Design: High-Performance Tor v3 Vanity Onion Address Generator (C++23, CPU + CUDA)

## Context

We are designing, from scratch, a vanity address generator for Tor v3 onion services: given user-supplied prefixes (e.g. `mysite`), it searches ed25519 keypairs until one's `.onion` address starts with a requested prefix, then emits the key material in the exact format Tor consumes. The project's purpose is twofold: (1) be genuinely fast — competitive with and eventually beyond `mkp224o`, the current open-source reference — and (2) serve as a showcase of modern C++23 systems engineering: clean architecture, measured optimization, CPU SIMD, CUDA, and rigorous testing of cryptographic code.

Target: Arch Linux, GCC 16+, CMake, 6C/12T CPU, 32 GB RAM, one CUDA-capable NVIDIA GPU. Working directory `/home/rsz/Desktop/cpp_onion` is empty (greenfield, not yet a git repo).

This document is the design review artifact. No code yet.

---

## 0. Problem model — what actually has to be fast

Everything downstream follows from the math, so we fix it first.

**Address construction (Tor rend-spec-v3):**

```
onion_address = base32( PUBKEY ‖ CHECKSUM ‖ VERSION ) + ".onion"
PUBKEY   = 32-byte ed25519 public key
CHECKSUM = SHA3-256(".onion checksum" ‖ PUBKEY ‖ VERSION)[0..2)
VERSION  = 0x03
```

35 bytes = 280 bits = exactly 56 base32 characters (alphabet `a–z 2–7`, lowercase). Two consequences that shape the whole design:

1. **Character `i` covers bits `[5i, 5i+5)` of the byte string.** The 32-byte pubkey covers bits 0–255, so characters 0–50 depend *only* on the public key. Any realistic prefix (≤ 16 chars) is therefore a pure function of the leading pubkey bytes. **The SHA3 checksum and base32 encoding are off the hot path entirely** — they run only on a match (rare) and at output time. The hot path reduces to: *produce next candidate pubkey, compare its leading ⌈5L/8⌉ bytes against a precompiled byte pattern + mask.*

2. **Search cost is geometric:** each character is 5 uniform bits, so a length-L prefix needs ~`32^L` candidates on average:

| Prefix len | Expected tries | @ 75 M/s (CPU est.) | @ 2 G/s (GPU est.) |
|---|---|---|---|
| 5 | 3.4 × 10⁷ | < 1 s | instant |
| 6 | 1.1 × 10⁹ | 14 s | < 1 s |
| 7 | 3.4 × 10¹⁰ | ~8 min | 17 s |
| 8 | 1.1 × 10¹² | ~4 h | ~9 min |
| 9 | 3.5 × 10¹³ | ~5.4 d | ~4.9 h |
| 10 | 1.1 × 10¹⁵ | ~174 d | ~6.5 d |

(Throughput figures are order-of-magnitude design targets, to be validated in Phase 1 — see §13/§21.)

**The single most important algorithmic decision** (dominates every micro-optimization by ~100×):

- **Alternative A — naive:** per candidate: random 32-byte seed → SHA-512 → clamp → full scalar multiplication `A = a·B`. Cost ≈ a SHA-512 plus ~250–300 point doublings/additions ≈ tens of microseconds. Yields ~10⁴–10⁵ keys/s/core. This is what a naive wrapper around libsodium gives you. Correct, simple, secure, and hopelessly slow.
- **Alternative B — incremental search (chosen, the `mkp224o` approach):** generate one random clamped base scalar `a₀` per worker per epoch, compute `A₀ = a₀·B` once, then iterate `Aᵢ₊₁ = Aᵢ + 8B` where `8B` is a precomputed constant point. Each candidate's secret scalar is implicitly `a₀ + 8i`. The step `+8` (not `+1`) preserves the clamping invariant that the low 3 bits of the scalar are zero. Per-candidate cost collapses from a full scalar multiplication to **one point addition (~8–12 field multiplications)**.
- **Alternative C — hierarchical/derived keys:** no usable structure exists; Tor v3 keys are raw ed25519, no BIP32-style derivation that preserves the public-key relation cheaply beyond what B already gives. Rejected.

Chosen: **B**, with one supporting trick: point addition in extended twisted-Edwards coordinates yields projective `(X:Y:Z)`; the address needs affine `y = Y·Z⁻¹` (the ed25519 pubkey encoding is little-endian `y` with the sign of `x` in the top bit of byte 31 — that sign bit is stream bit 248, which lands in base32 char 49, so matching on `y` alone is exact for prefixes ≤ 49 chars and the hot path needs *only* `y`; spec validation enforces the ≤ 49 bound). Field inversion is ~265 field-mults via Fermat, so we use **Montgomery batch inversion**: invert N candidates' Z values with `3(N−1)` mults + 1 inversion. At N = 1024, amortized inversion cost ≈ 3.3 mults/candidate.

**Hot-path cost budget per candidate:** ~8–12 M (point add incl. coordinate bookkeeping) + ~3.3 M (batch inversion share) + 1 M (`Y·Z⁻¹`) ≈ **13–16 field multiplications ≈ 350–500 cycles scalar** → ~8–12 M keys/s/core at 4 GHz, before SIMD. This budget is the project's north star; every design choice below is judged against it.

---

## 1. High-level architecture

```
                ┌─────────────────────────────────────────────┐
                │                  CLI / main                  │
                │  parse args → SearchSpec → spawn engines →  │
                │  monitor loop (stats, ETA) → shutdown        │
                └──────┬───────────────────────────┬──────────┘
                       │                           │
              SearchSpec (compiled)         ResultSink (writer thread)
                       │                           ▲
        ┌──────────────┴──────────────┐            │ rare MatchCandidate
        ▼                             ▼            │
┌────────────────┐          ┌──────────────────┐   │
│   CpuEngine    │          │    CudaEngine    │   │
│ N worker       │          │ 1 host thread,   │───┤
│ jthreads, each │──────────│ kernel launches, │   │
│ fully isolated │          │ hit readback     │   │
└────────────────┘          └──────────────────┘   │
        │                             │            │
        └──── per-worker padded ──────┘            ▼
              stat counters              Verifier (libsodium + SHA3,
              (read by monitor)          independent re-derivation)
                                                   │
                                                   ▼
                                         Tor-format key files on disk
```

Principles, with alternatives considered:

- **Engines are peers behind a narrow interface.** One `SearchSpec` in; `MatchCandidate`s and throughput counters out. CPU and GPU engines run concurrently on disjoint random subspaces (no coordination needed — collision probability is cryptographically nil).
  - *Alternative:* a work-queue model where a scheduler hands out scalar ranges. Rejected: the search is stateless and embarrassingly parallel; ranges/coordination add complexity and a sharing point with zero benefit. This also means future distributed operation is trivial (§17).
- **Match ≠ result.** Engines report *candidates* (worker id, base scalar id, iteration index). A separate **Verifier** independently recomputes the public key from the reconstructed scalar via libsodium, recomputes checksum and address, and only then writes files. This firewalls hand-written/GPU field arithmetic bugs — the class of bug that silently produces unusable keys — out of the user-visible output path. The cost is irrelevant (matches are rare by construction).
- **Hot path is engine-private.** Nothing crosses a thread boundary per candidate. Shared state is touched at batch granularity (≥ 2¹⁰ candidates) or rarer.
- **Static polymorphism inside engines, dynamic at the boundary.** The engine boundary is called once per run — a plain abstract base class (`IEngine`) is correct there; virtual dispatch cost is irrelevant and it keeps CUDA behind a compile-time firewall. Inside `CpuEngine`, the field/point kernel is a template parameter (concept-constrained) so scalar / AVX2 / AVX-512-IFMA backends are zero-cost to compose, selected once at startup by CPU feature detection.
  - *Alternative:* virtual dispatch per batch inside the engine too. Viable (batch granularity amortizes it) but loses cross-kernel inlining within the batch loop, and concepts give us better compile-time error messages than an ABI. Templates chosen.

## 2. Core components and responsibilities

| Component | Responsibility | Key design point |
|---|---|---|
| `SearchSpec` / `CompiledMatcher` | Parse + validate prefixes (charset `a–z2–7`), compile to byte+mask patterns | Compilation happens once; hot loop sees only `(bytes[], mask, len)` |
| `crypto::fe` | Field arithmetic over GF(2²⁵⁵−19); 5×51-bit limbs scalar; AVX2 / IFMA variants | The performance kernel; concept `FieldOps` |
| `crypto::ge` | Edwards point ops: precomputed-point mixed addition, fixed-base scalar mult (setup only), batch inversion | `8B` and window tables as `constexpr` data computed at compile time |
| `crypto::sha3` | Vendored compact Keccak-f[1600] (SHA3-256) | Cold path only; libsodium does **not** provide SHA3 — this must be vendored |
| `crypto::keys` | Scalar reconstruction `a₀+8i`, clamping invariants, expanded-secret-key assembly | Owns the carry/overflow guard (§9) |
| `CpuEngine` | Worker threads, batching, epochs, matching | All buffers preallocated per worker |
| `CudaEngine` | Device memory, kernel lifecycle, hit readback | Isolated CMake target; off by default if no CUDA toolchain |
| `Verifier` | Independent recomputation of pubkey/address before output | Uses libsodium `crypto_scalarmult_ed25519_base_noclamp` |
| `ResultSink` / `TorKeyWriter` | mutex+condvar queue → writer thread → Tor-format files (`hostname`, `hs_ed25519_secret_key`, `hs_ed25519_public_key`), dir 0700 / files 0600 | Golden-file tested against a real Tor-generated service |
| `Stats` | Per-worker cacheline-padded counters, monitor aggregation, ETA | Relaxed atomics, no RMW contention |
| `rng` | `getrandom(2)`-backed seeding (via libsodium `randombytes_buf`) | Per worker, per epoch; never userspace-only PRNG for key material |
| `cli` | Arg parsing, lifecycle, signal handling (SIGINT → `stop_source`) | Thin |

## 3. Project structure

```
cpp_onion/
├── CMakeLists.txt            # top-level: options ONION_CUDA, ONION_NATIVE, ONION_SANITIZE
├── CMakePresets.json         # debug / release-native / asan / ubsan / tsan / cuda
├── cmake/                    # toolchain helpers, CPM or FetchContent pins
├── docs/
│   └── design.md             # this document, kept current
├── src/
│   ├── core/                 # SearchSpec, CompiledMatcher, base32, types (lib: onion_core)
│   ├── crypto/               # fe/, ge/, sha3/, keys.{hpp,cpp}   (lib: onion_crypto)
│   ├── engine/
│   │   ├── engine.hpp        # IEngine, MatchCandidate, Stats interfaces
│   │   ├── cpu/              # CpuEngine + kernel templates (lib: onion_engine_cpu)
│   │   └── cuda/             # .cu files, host wrapper (lib: onion_engine_cuda, OPTIONAL)
│   ├── io/                   # TorKeyWriter, Verifier (lib: onion_io)
│   └── cli/                  # main.cpp (exe: onion)
├── tests/                    # Catch2: unit/, kat/ (vectors), e2e/, golden/
├── fuzz/                     # libFuzzer harnesses: matcher, base32, key-file parser
├── bench/                    # nanobench micro + `onion bench` e2e; results/*.json
└── tools/
    └── verify_onion.py       # independent Python oracle (pure-python ed25519 + sha3)
```

Each `src/` subdirectory is a separate CMake library with explicit dependency edges (`onion_engine_cpu → onion_core → onion_crypto`). Rationale: enforces layering (CUDA can't leak into core), gives per-library test binaries, and keeps TUs small for fast incremental builds. Single-header "unity" style rejected — it hides coupling and slows iteration.

## 4. Dependencies — choices and rationale

Philosophy: **minimal, pinned, nothing in the hot path.** Every hot-path instruction is ours; dependencies provide correctness oracles and entropy.

| Dependency | Use | Alternatives considered | Why this |
|---|---|---|---|
| **libsodium** | Verifier (independent pubkey derivation), `randombytes_buf`, `sodium_memzero`/`sodium_mlock` | OpenSSL (heavyweight, slower ed25519 API churn); hand-rolled only (loses the independent-oracle property) | Audited, ubiquitous on Arch, exactly the right primitives. Never on the hot path |
| **Vendored compact Keccak** (tiny-sha3-class, single file) | SHA3-256 checksum | OpenSSL EVP (dependency weight for a cold path); XKCP full (build complexity) | libsodium lacks SHA3; checksum is cold-path so a ~150-line audited compact implementation is ideal. Cross-checked against NIST vectors *and* the Python oracle |
| **CLI11** (FetchContent, pinned tag) | CLI parsing | hand-rolled (fine but boilerplate); Boost.ProgramOptions (heavy) | Header-only, ergonomic, no runtime cost |
| **Catch2 v3** | Tests | GoogleTest (heavier build); doctest (fine, fewer features) | Sections + generators fit property-style crypto tests well |
| **nanobench** | Microbenchmarks | google/benchmark (heavier, more rigid) | Single header, robust timing, JSON output for regression tracking |
| **CUDA Toolkit** (optional) | GPU backend | OpenCL (worse tooling, no Nsight Compute); SYCL (immature for this) | NVIDIA GPU available; Nsight tooling is decisive |

Explicitly **not** dependencies: Boost (nothing needed), fmt (GCC 16 has `std::print`/`std::format`), TBB/OpenMP (hand-rolled `jthread` pool is trivial here and we need precise control over per-thread state and affinity; OpenMP's runtime adds opacity to profiling).

## 5. CPU-only design

**Field representation:** 5 limbs × 51 bits in `uint64_t`, products via `unsigned __int128` (GCC native). This is the proven fast scalar representation (donna-64 lineage) and maps directly to AVX-512 IFMA's 52-bit lanes later.
- *Alternative:* 10×25.5-bit (donna-32) — only wins on 32-bit targets; rejected. 4×64 with carry chains — fewer limbs but carry-chain serialization hurts on modern OoO cores and vectorizes poorly; rejected.

**Hot loop per worker (batch size N ≈ 1024, tunable):**

```
epoch:  seed ← getrandom; a₀ ← clamp(SHA-512(seed)); A ← fixed-base-mult(a₀)   [setup, ~µs]
loop:   for i in 0..N:  A ← A + 8B (mixed add, precomputed niels form); save Zᵢ, Yᵢ
        batch-invert {Zᵢ} (Montgomery, two interleaved chains for ILP — the
            product chain is serially dependent; interleaving 2–4 independent
            chains hides multiplier latency)
        for i: yᵢ ← Yᵢ·Zᵢ⁻¹; serialize first ⌈5L/8⌉ bytes; masked compare
        on hit → push MatchCandidate{worker, epoch_seed_id, i}
        every batch: counter += N (relaxed); check stop_token
        every ~2²⁴ candidates: new epoch (re-seed)
```

**Batch size tradeoff:** larger N amortizes the single inversion (1/N · ~265 M each) but grows the working set (N × (point 160 B + scratch)); N = 1024 ≈ 200 KB fits L2 comfortably, and inversion amortization has diminished to noise (0.26 M/candidate). Beyond ~4096 there is no win and L2 pressure begins. Default 1024, exposed as a tunable for benchmarking.

**SIMD roadmap (phased, each gated on measurement):**
1. **Scalar 51-bit baseline** — correctness reference, already ~8–12 M/s/core.
2. **AVX2:** process 4 *independent* candidates per vector lane-group (vertical/SoA batching: 4 point-additions in lockstep, one field element limb per `ymm` across 4 candidates). This is the standard way to vectorize crypto without redesigning the math: no cross-lane dependencies, pure throughput. Expected ~2–3× per core (AVX2 has no 64×64→128 mul; uses 32-bit `vpmuludq` limbs — the AVX2 path gets its own representation, 10×25.5-bit (donna-32 style) or 9×29-bit, chosen by benchmarking carry-handling cost).
3. **AVX-512 IFMA** (`vpmadd52luq/huq`) if the CPU supports it: near-perfect fit for 51/52-bit limbs, the known-fastest ed25519 approach on current x86. Runtime-dispatched.
- *Alternative:* `std::experimental::simd`. Rejected for the kernel: no access to widening 52-bit multiply-add semantics; we need intrinsics. A thin in-house `simd<T,N>` wrapper keeps intrinsics contained and testable.
- SoA layout for batched candidates (`X[4][5]` limb-major) — mandatory for clean vector loads; AoS rejected.

**Matching:** prefix compiled to `(prefix_bytes, last_byte_mask)`. For k prefixes, linear scan of masked compares (k ≤ ~16 typical — branch-predictable, all in registers). For large prefix sets (future), first-byte 256-entry dispatch table, then per-bucket compare. Regex matching deliberately excluded from the hot path (§17, §20).

## 6. GPU / CUDA design

> **Status: IMPLEMENTED, measured, and deep-optimized (Phase 3, 2026-06-13).** `CudaEngine` (behind `IEngine`) ships in `src/engine/cuda/`, selected by `--engine cuda` on the `cuda` CMake preset. On the GTX 1650 (Turing, sm_75, 14 SMs, 4 GB) it sustains **390.1 M keys/s** (median of 3 × 8 s `--bench` runs; all 3 runs identical), **15.4× the 12-thread CPU incremental baseline** (25.4 M/s) — clearing the Phase-3 gate (≥ 10× whole-CPU). **Device-field choice:** the planned native **8×32-bit limb** field (`device_field32.cuh`, `mul.wide`/`mad.hi` carry chains) is the default (`ONION_CUDA_FIELD32=ON`); it replaced the initial validated 5×51-bit `__int128` field (`device_field.cuh`, kept as the cross-checked reference path) because nvcc *emulates* 64-bit/`__int128` multiplies on Turing. The 32-bit field cut the search kernel from **178 → 96 registers/thread with 0 spill stores/loads** (`ptxas -v`), raising occupancy and giving the ~275 → ~306 M/s gain.
>
> **Deep-optimization pass (306 → 390 M/s, +27%).** Profiling (analytical + `ptxas -v`; `ncu`/`nsys` are not installed on this box) showed the kernel is **local-memory-bandwidth bound** — the per-thread Montgomery batch-inversion scratch (Y/Z chains) streams from DRAM-backed local memory at ~0% cache residency, not ALU and not register spills (0 spills). Three changes that attacked or hid that traffic were **KEPT**: (1) a **fused 2-array batch inversion** that drops the redundant `prefix[]` array by folding `Y·pe` into one stored value — same field-op count, −33% local footprint, **302 → 323 M/s** (commit `48ef121`); (2) a **re-swept steps-per-thread knee** — with the smaller per-thread footprint the inversion amortizes essentially for free as M grows, so M was raised **1024 → 3072** (the largest that still launches at T=2¹⁵, 2.83/4 GB), **323 → 340 M/s** (commit `502cb61`); (3) a **double-buffered epoch pipeline** — each in-flight epoch owns its own device buffers/stream so epoch N+1's host point-add setup + H2D overlaps epoch N's running kernel and the GPU never idles between epochs, **340 → 390 M/s** (commit `27ca210`). Two levers were honestly tried and **REVERTED** as losses on this card: **warp-cooperative inversion** (one `__shfl` Montgomery chain across 32 lanes) cut local memory 32× but the prefix/suffix scan added ~15–20 field-muls per key vs the baseline's amortized ~0.26 mul/key — a **−46% regression** (the bus traffic it saved was smaller than the ALU it added); and **lifting occupancy past 62.5%** (`__launch_bounds__(128,5)`) was a flat **0%** because the memory bus is already saturated, so extra resident warps merely contend (and `(128,6)` spilled). Both are documented experiments, not committed.
>
> **Hardware ceiling, stated honestly.** ~390 M/s is the genuine GTX 1650 ceiling for this workload — the kernel is bus-bound and the cheap algorithmic headroom is exhausted. The gap to multi-Gk/s is **hardware, not software**: a Blackwell-class GPU has ~20–30× this Turing card's compute, so the *same code* scales onto it to multi-Gk/s (a friend's Blackwell measures ~7 Gk/s; 7 Gk/s ÷ ~24 ≈ this card's ~310–390 M/s). **1 Gk/s is not reachable on a 14-SM, 4 GB GTX 1650** and was never claimed.
>
> **Correctness:** the device chain is cross-validated bit-for-bit against libsodium `base_noclamp(a0+8i)` (test `cuda.CUDA device chain matches libsodium`), every emitted key passes the host Verifier firewall (libsodium re-derive) plus the Python oracle — full CUDA suite **56/56 green**, CPU suite **53/53 green**. **Leak/safety verification (hard requirement):** `compute-sanitizer --tool memcheck --leak-check full` reports **0 bytes leaked in 0 allocations / 0 errors** on both the test binary and a real generate run — the long-running `run()` loop `cudaMalloc`s per epoch (now per-pipeline-slot) and frees every allocation via RAII (no per-epoch leak, no host leak, no races on the readback path). The rest of this section is the original design rationale.

Same algorithm, different parallelization grain: **one search lane per thread**, tens of thousands of threads.

**Kernel structure:**
- **Epoch setup kernel:** each thread `t` derives its start scalar `aₜ = a₀ + 8·t·K` (K = candidates per thread per epoch; thread ranges are disjoint by construction) and computes its start point with a **fixed-base windowed scalar multiplication** (4-bit windows ⇒ ~64 mixed additions, one-time per epoch). The window table lives in **shared memory** here — per-thread scalars make lookups warp-divergent, and the constant cache serializes divergent accesses; `__constant__` is reserved for the warp-uniform `8B` in the search kernel. Setup is amortized to noise for K ≥ 4096. Note the GPU epoch spans `T·K` candidates under one `a₀` (e.g. ~2²⁸–2³² for 10⁵–10⁶ threads) — far larger than the CPU's 2²⁴ window; the scalar-overflow guard (§9) takes `epoch_len = T·K` (overflow probability still ~2⁻²²² at 2³²) and the related-key window statement in §18 is per-epoch, sized accordingly.
- **Search kernel (the hot one):** each thread iterates `A += 8B`, accumulating a per-thread chain of M Z-values (M ≈ 16–32, register/local-memory budget permitting), runs per-thread Montgomery inversion over its chain, serializes leading y-bytes, masked-compares. Hits: `atomicAdd` on a global counter + write `(tid, iter)` into a small fixed slot buffer. The host reads back only the counter + slots. Per-thread points persist across launches in a global-memory state array (~160 B/thread, tens of MB at 10⁵–10⁶ threads — registers don't survive kernel boundaries), loaded once per launch; amortized over ≥ 10³ candidates per launch, **DRAM and PCIe traffic round to zero; this is a pure compute workload.**
- **Honest cost accounting:** short per-thread chains amortize inversion far worse than the CPU's N = 1024 batches — per candidate, the inversion share is `3 + 265/M` muls ≈ 19.6 M at M = 16 (the Fermat share alone exceeding the point addition) or 11.3 M at M = 32. v1 GPU cost is therefore **~20–29 field muls/candidate, roughly 2× the CPU budget** — the GPU wins on raw multiplier count, not per-candidate efficiency. This is the explicit target of experiment #1 below; GPU throughput projections must use this budget, not the CPU's.
- *Alternative considered — full scalar-mult per candidate per thread:* ~100× more arithmetic; only upside is unrelated keys; rejected (same reasoning as CPU).
- *Alternative considered — warp-cooperative batch inversion* (one Montgomery chain across 32 lanes via shuffles, effective M = 32·M_thread): directly attacks the amortization gap above, at the cost of synchronization and divergence complexity. Slated as **experiment #1** after the simple version is measured, not in v1.

**Field arithmetic on GPU:** 8×32-bit limbs with `mad.lo/mad.hi` PTX carry chains (`__umulhi`-based) — GPUs have fast 32-bit IMAD and no native 64-bit multiply. The 51-bit representation is wrong for GPU; this is a *second* field implementation, which doubles the correctness surface — mitigated by cross-validation tests (same seeds ⇒ identical pubkey streams CPU vs GPU, §16) and the Verifier firewall.

**Register pressure is the central GPU risk:** extended point = 4 fe × 8 limbs = 32 registers minimum live, plus temporaries and the Z-chain. Expect 128–255 registers/thread, i.e. low occupancy (~12–25%). That can still be near-peak for IMAD-bound code (occupancy ≠ utilization), but it must be measured in Nsight Compute, and the Z-chain length M is the knob: longer chain = better inversion amortization but more spill to local memory. Tuned empirically per-architecture; `__launch_bounds__` pinned after measurement.

**Launch strategy:** repeated grid launches sized to ~50–200 ms each.
- *Alternative:* persistent kernel + device-side loop. Saves ~10 µs launch overhead per launch (noise at 100 ms granularity) but complicates cancellation, watchdog timeouts on display GPUs, and error recovery. Rejected for v1.

**Toolchain firewall:** `onion_engine_cuda` is an optional CMake target. The constraint is no longer the C++ dialect — CUDA 13.3 nvcc supports `--std=c++23` — but the **host-compiler matrix: nvcc caps at GCC 15.x and rejects GCC 16**, the project's mandated compiler. So the CUDA TUs are compiled with a separately pinned host compiler (GCC 15 or clang), which forces the same discipline anyway: the boundary header stays dialect-conservative (plain structs, byte spans, no stdlib types whose ABI/feature level could diverge between the two toolchains), and host C++23 code never includes CUDA headers. If the toolchain or GPU is absent, the build degrades gracefully to CPU-only.
- *Alternative:* clang as CUDA compiler (single-frontend simplicity) — kept as a documented escape hatch; nvcc default because Nsight integration and IMAD codegen are primary.
- *Alternative:* runtime `dlopen` plugin for the GPU backend — over-engineering for one binary on one machine; compile-time option chosen. Revisit only if binary distribution becomes a goal (§17).

## 7. Work distribution across threads

- **CPU:** `W` worker `std::jthread`s, default = **physical cores (6)**, not logical (12): the kernel saturates the multiplier ports, so SMT siblings mostly fight for the same execution units. *This is a hypothesis to falsify in benchmarking* — `--threads` stays a first-class flag and the bench matrix covers 6 vs 12. One additional GPU-host thread per GPU (mostly blocked in `cudaStreamSynchronize`), one monitor thread, one writer thread — these three are I/O-ish and coexist fine with 6 busy workers.
- **No shared work queue, no work stealing.** Workers own disjoint random subspaces by construction (independent seeds); the search has no notion of "finishing" a range, so there is nothing to balance — every thread is 100% busy until stop. Work-stealing machinery would be pure overhead.
- **Thread affinity:** optional `--pin` (pin worker *n* → physical core *n*). Default off; measure. On a 6-core single-CCX/single-socket part, migration cost is minor, but pinning stabilizes benchmark variance.
- **Stop discipline:** `std::stop_token` checked once per batch (~100 µs granularity). On match in "find one and exit" mode, `stop_source.request_stop()`; workers drain current batch and exit. SIGINT handled via `signalfd`-style self-pipe in the monitor → same stop path (no async-signal-unsafety).

## 8. Randomness strategy

- **Key material entropy:** every epoch's 32-byte seed comes from the kernel CSPRNG (`randombytes_buf` → `getrandom(2)`). Frequency: once per worker per ~2²⁴ candidates ⇒ a few syscalls per second per machine — performance-irrelevant, so **no userspace PRNG is ever used for key material**. (A ChaCha20 userspace generator was considered and rejected: it adds an attack/bug surface to save syscalls we don't make.)
- `std::random_device` rejected: standard gives no CSPRNG guarantee; libsodium's API is explicit and audited.
- **Within an epoch**, candidates are `a₀ + 8i` — *not* fresh randomness, and deliberately so (§0). Uniformity: `a₀` is uniform over the clamped set; the walk covers a contiguous coset slice; resulting public keys are computationally indistinguishable from uniform for any feasible epoch length. Epoch re-seeding bounds the window of related keys (§18) and refreshes entropy.
- **Non-crypto randomness** (bench shuffling etc.): `std::mt19937_64`, clearly segregated in `rng/` with type-distinct wrappers so the two can't be confused at a call site.

## 9. Cryptographic considerations

- **Clamping invariant under increment:** clamped scalars have bits {0,1,2} clear and bit 254 set, bit 255 clear. Step `+8` preserves the low bits. Carry into bit 255 is possible in principle (`a₀` uniform in `[2²⁵⁴, 2²⁵⁵)`); probability within an epoch is ~`8·2²⁴/2²⁵⁴` ≈ 2⁻²²⁷ — but the guard is one byte-compare at epoch setup plus a bound check, so we implement it anyway (resample if `a₀ + 8·epoch_len` would overflow 2²⁵⁵). Cheap insurance against a "can't happen" that would silently break the clamped-form assumption some ed25519 code makes.
- **Expanded secret key correctness:** Tor's `hs_ed25519_secret_key` holds the *expanded* 64-byte secret `(a ‖ RH)` — scalar plus the PRF prefix used for deterministic nonces — not a seed. For a found key, `a = a₀ + 8i` (plain 256-bit add, no reduction). **RH must NOT be shared across related keys:** RFC 8032 derives RH from the seed, but our emitted scalar no longer corresponds to any seed; reusing one RH across scalars that differ by known amounts risks correlated nonces. Decision: **RH ← fresh 32 random bytes at output time.** Tor only requires that `(a, RH)` be consistent with the published pubkey, and nonce = SHA-512(RH ‖ msg) stays unpredictable, so this is sound on its own merits. (`mkp224o` solves it differently — it keeps the seed-derived RH but never emits two keys from one seed, re-seeding after each match; both schemes are sound, ours doesn't depend on the one-key-per-epoch discipline.)
- **Verification firewall (restated as a crypto control):** every candidate is re-derived via libsodium (`crypto_scalarmult_ed25519_base_noclamp` on the reconstructed scalar) and the full address re-computed (vendored SHA3 + base32) before anything is written. A mismatch is a hard assertion failure — it means a kernel bug, and we want to know loudly.
- **Independent oracle:** `tools/verify_onion.py` (pure-Python ed25519 + hashlib sha3) re-validates outputs in tests, so even a shared-library bug or our SHA3 vendoring can't self-confirm.
- **Constant-time is explicitly a non-goal for the search loop** (§18 threat model) — candidates are not secrets until selected. The *output path* (scalar reconstruction, file writing) uses constant-time primitives where free, and wipes intermediates.

## 10. Memory management strategy

The working set is tiny — this is a compute-bound, cache-resident workload, and the strategy is mostly about *not doing things*:

- **Zero allocation on the hot path.** Each worker owns one preallocated, 64-byte-aligned arena: batch arrays (SoA), scratch, serialization buffer. Allocated once at engine start, sized by batch config. Steady-state heap traffic = zero; the allocator is irrelevant by design.
- *Alternatives rejected:* `std::pmr` pools (solve allocation churn we don't have), custom allocators (same), hugepages (working set ≈ 200 KB/worker; TLB pressure is nil).
- **Layout over allocation:** SoA for batches (§5), `alignas(64)` on per-worker mutable shared state (counters) using `std::hardware_destructive_interference_size` to prevent false sharing — on this workload a single false-shared counter line measurably taxes every worker.
- **Secrets hygiene:** epoch seeds/scalars live in a small per-worker `SecretArena`: `sodium_mlock`ed (best-effort, flag-controlled), `sodium_memzero`ed on epoch end and engine teardown. Not `memset` — it gets dead-store-eliminated.
- **GPU:** hot state in registers/local memory *within* a launch; global memory holds the per-thread point-state array that persists across launches (§6, tens of MB — touched once per launch, amortized to nothing), the hit buffer (KBs), and constant tables. Pinned host memory for the readback buffer (one `cudaMemcpyAsync` per launch).
- **NUMA:** N/A (single socket); noted as a future concern only if the engine ever runs on multi-socket servers (§17).

## 11. Synchronization strategy

Inventory of *all* cross-thread communication — kept deliberately short:

| Channel | Mechanism | Frequency | Rationale |
|---|---|---|---|
| Stop signal | `std::stop_token` (+ one `atomic<bool>` mirror for the CUDA host loop) | checked per batch | stdlib-native, composes with `jthread` dtor |
| Throughput counters | per-worker `alignas(64)` plain `uint64_t` slots; worker writes and monitor reads via relaxed `std::atomic_ref` — no RMW anywhere | write 1/batch, read 2 Hz | single-writer ⇒ plain stores suffice; `atomic_ref` keeps the worker's own view non-atomic (§19); relaxed is sufficient (monotonic counter, no ordering dependency) |
| Match candidates | `std::mutex` + `std::condition_variable` MPSC queue | ~once per minutes/hours | see §12 |
| Result files | single writer thread | rare | serializes I/O, owns fsync + permissions |
| Engine lifecycle | plain calls before threads start / after they join | once | no concurrency at all |

Design rule: **synchronization cost must round to zero**, achieved by making synchronization *rare*, not by making it clever. TSan build must stay clean — exotic schemes that TSan can't see through are themselves disqualified (§20).

## 12. Lock-free opportunities — and why we mostly decline them

- **Stat counters:** already effectively lock-free (single-writer relaxed atomics, no RMW, no contention). This is the one place lock-freedom is both free and correct. Taken.
- **Match queue:** a lock-free MPSC queue is the classic résumé-driven mistake here. Expected throughput: *one item per minutes-to-hours*. An uncontended `std::mutex` is ~20 ns and obviously correct; a Michael-Scott queue buys nothing measurable and costs review effort and TSan/ABA reasoning. **Declined, with prejudice.** Revisit only if a future mode (e.g. "stream all near-misses") changes the rate by ~6 orders of magnitude.
- **Seed/epoch handout:** none needed — workers seed independently from the kernel; there is no shared sequence to coordinate (a shared atomic epoch counter exists only for labeling/diagnostics, relaxed).
- **GPU hit buffer:** device-side `atomicAdd` slot allocation — uncontended in practice (hits are rare), and the canonical CUDA pattern. Taken.

The honest summary: this workload's parallelism is so clean that the right amount of lock-free engineering is *almost none*, and claiming otherwise would be self-deception. The design's concurrency sophistication goes into *isolation* (per-worker everything) rather than *coordination*.

## 13. Likely performance bottlenecks (ranked, with detection plan)

1. **Multiplier throughput (CPU):** the kernel is a dense stream of 64×64→128 multiplies; ports saturate. Detect: `perf stat` IPC + port utilization (toplev); fix: IFMA path, better scheduling (interleave 2 candidates' dependency chains even in scalar code).
2. **Batch-inversion serial chain:** Montgomery's trick is inherently sequential in the product pass. Detect: it shows as low IPC in that phase; fix: 2–4 interleaved chains (planned from day one, §5).
3. **GPU register pressure / occupancy cliff:** §6. Detect: Nsight Compute occupancy + spill counters; knobs: Z-chain length, limb scheduling, `__launch_bounds__`.
4. **SMT contention:** 12 threads may *reduce* throughput vs 6. Detect: bench matrix; fix: default to physical cores.
5. **Thermal/turbo collapse on sustained AVX-512:** 6-core desktop parts downclock under heavy 512-bit streams. Detect: long-run bench with frequency logging (`turbostat`); fix: it may still net-win — *measure, never assume*; possibly prefer 256-bit IFMA encodings.
6. **False sharing / accidental hot-path sharing:** prevented structurally (§10/§11); detect regression via `perf c2c`.
7. **What is deliberately NOT a bottleneck:** SHA3, base32, matching (byte compares), result I/O, PCIe — all off the hot path by construction (§0). If profiling ever shows them, the architecture has been violated, not the code being slow.

## 14. Benchmarking strategy

- **Canonical metric:** *candidates fully matched per second* (pubkey derived to comparable bytes + compared) — not "point adds/s", not seeds/s. Defined once in `docs/`; every reported number uses it. Inflated metrics are how vanity-generator READMEs lie; we won't.
- **Three tiers:**
  1. **Microbenchmarks (nanobench):** `fe_mul`, `fe_sq`, point add, batch inversion (N sweep), matcher — each kernel variant (scalar/AVX2/IFMA), pinned core, fixed frequency where possible. Output JSON → `bench/results/<git-sha>.json`.
  2. **Engine bench (`onion bench --duration 30s`):** in-binary, real hot loop, impossible-prefix (e.g. 16 chars) so no early exit; reports candidates/s per worker and total, plus thread-count and batch-size sweep matrices.
  3. **External baseline:** same machine run of `mkp224o` (its `-s` stats) as the honesty anchor. Target: parity in Phase 2, exceed with IFMA/CUDA phases.
- **Methodology:** ≥ 5 runs, report median + spread; `cpupower frequency-set` performance governor; record temperature/frequency (`turbostat`) alongside; reject runs with thermal variance > threshold. Compare across commits via a small script over the JSON history — regressions > 3% flag red in review.
- **GPU:** Nsight-derived achieved-throughput plus the same canonical end-to-end metric; never report kernel-only numbers as system numbers.
- **mkp224o head-to-head (measured):** same machine (Ryzen 5 4600H + GTX 1650), mkp224o's fastest amd64-51-30k build, 12 threads. mkp224o CPU ~28.9 M keys/s; cpp_onion CPU incremental ~28.1 M keys/s (on par with the reference); cpp_onion CUDA ~390 M keys/s ≈ 13.5× mkp224o. The GPU is the differentiator (mkp224o has no GPU backend); the ~13× is GPU-hardware-bound on this GTX 1650 — hundreds-of-× would need a larger GPU.

## 15. Profiling strategy

- **CPU, macro:** `perf record -g` + flamegraphs (expectation: one giant plateau in `fe_mul` — anything else visible is a finding); `perf stat -d` for IPC/cache/branch; **toplev (pmu-tools)** for port-pressure attribution (the kernel's limiting resource is execution ports, which plain perf won't name).
- **CPU, micro:** `llvm-mca` / uiCA on the emitted inner-loop asm to validate scheduling models before trusting wall-clock deltas; `objdump`-diff of the kernel between commits to catch GCC codegen regressions (GCC 16 is bleeding-edge — §21).
- **GPU:** **Nsight Compute** per-kernel (occupancy, register count, spills, IMAD pipe utilization, achieved vs theoretical issue rate); **Nsight Systems** for timeline sanity (launch gaps, readback overlap). CUPTI counters wired into `onion bench --cuda` for headless regression numbers.
- **Cadence:** profile *at each phase gate* (§21 roadmap) and on any unexplained bench delta; no continuous-profiling machinery — the program has one hot loop.

## 16. Testing strategy

Crypto code that silently produces wrong-but-plausible output is the project's existential risk; testing is layered accordingly:

1. **Known-answer tests:** RFC 8032 ed25519 vectors (pubkey derivation); NIST SHA3-256 vectors; RFC 4648 base32 vectors; real Tor-generated onion service as golden files (address ↔ key files round-trip, exact file bytes).
2. **Property/cross-implementation tests (the workhorse):**
   - `∀ random a₀, i ≤ 10⁴: incremental_chain(a₀)[i] == libsodium_scalarmult_base(a₀ + 8i)` — validates the entire incremental scheme against an independent implementation.
   - batch-inverted `y` == per-element Fermat inversion == libsodium field ops where exposed.
   - Every kernel variant (scalar/AVX2/IFMA/CUDA) run on identical seeds must produce **bit-identical pubkey streams** — one test harness, parameterized by kernel; this is the contract that lets us optimize fearlessly.
3. **End-to-end:** search a 1–2 char prefix (sub-second), then re-verify the written files with `tools/verify_onion.py` (independent language, independent libraries) and check file permissions and formats.
4. **Fuzzing (libFuzzer + ASan):** prefix parser/matcher compiler (user input!), base32, key-file writer/reader round-trip. Matcher correctness also property-tested against a naive "base32-encode-then-strcmp" reference.
5. **Sanitizer matrix in CI:** ASan+UBSan test run, TSan engine smoke run (short multi-threaded search), plus a `-D_GLIBCXX_ASSERTIONS` debug run. CUDA: `compute-sanitizer` (memcheck/racecheck) on the test kernels.
6. **Statistical smoke test:** χ² on leading-byte distribution over ~10⁷ generated candidates — catches catastrophic bias bugs (wrong limb carry producing structured keys) cheaply.

TDD discipline: field/point/matcher units are built test-first against the KAT/property suites (the references exist before our implementations do, which is the natural TDD shape for crypto).

## 17. Future extensibility (designed-for, not built)

- **New match semantics:** `CompiledMatcher` is an interface compiled from `SearchSpec`; suffix/substring/regex matchers slot in on CPU. **Explicit boundary:** GPU supports only prefix-set matching; complex matchers fall back to CPU (documented, enforced by spec validation). Regex-on-GPU is a tar pit we name and refuse.
- **Distributed search:** because workers are stateless and coordination-free (§7), distribution = run the binary on more machines + collect results; a thin `--report-to` HTTP sink is the only addition needed. No scheduler will ever be required — this falls out of the no-work-queue decision.
- **New backends:** `IEngine` is the seam — candidates: multi-GPU (one `CudaEngine` per device, trivial), ROCm/HIP port of the CUDA kernel, OpenCL for non-NVIDIA. The Verifier/cross-validation harness (§16) is backend-agnostic, which is what actually makes new backends *cheap to trust*.
- **Batch "mining" mode:** keep-searching-after-match, multiple prefixes to separate output dirs. Sink policy *plus one engine rule that is not optional*: **a worker that emits a match immediately starts a new epoch (fresh seed).** Two scalars from one epoch are publicly linkable (`A_j − A_i = 8(j−i)·B`, checkable in ≤ epoch-length point adds) and compromise of one reveals the other — the same discipline mkp224o enforces. §18's security argument assumes exactly this rule.
- **What we refuse to pre-build:** plugin ABIs, config-file frameworks, daemon mode. YAGNI; the seams above are enough.

## 18. Security considerations

**Threat model:** the machine generating keys is trusted; adversaries are (a) remote parties who later see the public address, (b) post-hoc local artifact leakage (swap, logs, dropped files), (c) our own bugs producing weak/invalid keys. Out of scope: live local attackers doing microarchitectural side-channel extraction during generation (if your box is owned while generating, the key is lost regardless).

- **Key strength is not weakened by vanity search:** the published prefix constrains public-key bits, which are public anyway; the secret scalar retains full dlog hardness. The *related-key* structure within an epoch never leaves the process — **at most one scalar is ever materialized per epoch** (workers re-seed immediately after emitting a match, §17), and `a₀`/neighbors are wiped (§10). An attacker holding the published key cannot derive epoch relations without `a₀`. (Epoch length is ~2²⁴ candidates on CPU and T·K on GPU — the invariant is per-epoch, not per-size.)
- **Nonce-prefix independence (RH)** — §9; this is the one subtle crypto-correctness trap in incremental vanity generation and is handled explicitly.
- **Artifact hygiene:** output dir `0700`, key files `0600` (matching Tor's expectations); never print secret bytes; seeds/scalars in `mlock`ed, zeroized arenas; no core dumps of workers (`PR_SET_DUMPABLE` cleared when `--paranoid`).
- **Verification before emission** (§9) doubles as a security control: a flipped bit in a secret key file is an unusable-service incident.
- **Supply chain:** dependencies pinned by tag+hash in FetchContent; vendored Keccak imported with provenance note and KAT-locked.
- **Responsible-use note in README:** vanity prefixes are also a phishing tool (humans verify only the prefix of 56-char addresses); we document that full-address verification is the user's responsibility. The tool itself is standard, legitimate Tor operator tooling (Facebook, NYT, etc. use vanity onions).

## 19. Modern C++ features to use — each with a job, not a tour

The build standard is C++23; honesty requires labeling vintages, since most of the workhorse features are C++17/20 and a reviewer should not see them sold as C++23. Genuinely-C++23 items are bolded in the vintage column.

| Feature | Vintage | Where | Why it earns its place |
|---|---|---|---|
| `std::expected<T, Error>` | **C++23** | spec parsing, io, engine setup | error handling in library code without exceptions on any path that could be warm; explicit at call sites |
| `std::print`/`println` | **C++23** | CLI/monitor | drops fmt dependency and iostream locale baggage |
| Deducing `this` | **C++23** | small CRTP-replacement mixins (kernel variants) | removes CRTP boilerplate where static dispatch composes kernels |
| `[[assume]]`, `std::unreachable()` | **C++23** | kernel inner loops | communicates limb-bound invariants to the optimizer (measured before kept — see §20 discipline) |
| `std::stacktrace` | **C++23** lib (needs `-lstdc++exp` on libstdc++) | verifier hard-failure reports | actionable diagnostics (§9 — failures must be loud and located) |
| `if consteval`, extended `constexpr` | **C++23** (+ C++20 `consteval`) | `8B`, fixed-base window tables | tables are *computed, not pasted* — compile-time-verified through the same field-arithmetic code paths the tests cover; removes a transcription-error class |
| `std::jthread` + `std::stop_token` | C++20 | engines, monitor | cooperative cancellation without hand-rolled flags; RAII join kills a whole bug class |
| Concepts (`FieldOps`, `PointOps`, `MatcherLike`) | C++20 | kernel composition | zero-cost polymorphism with readable constraint errors; the mechanism behind swappable SIMD kernels |
| `std::span` / `std::byte` | C++20 / C++17 | all buffer interfaces | bounds-carrying views; no decay, no `void*`; `as_bytes` for serialization seams |
| `std::atomic_ref` | C++20 | stats slots (§11) | atomic access to per-worker plain `uint64_t` without forcing atomic types into the worker's private view |
| `std::hardware_destructive_interference_size` | C++17 | counters, arenas | false-sharing prevention as a *declared* property |
| `std::format` | C++20 | CLI/monitor | shared formatting machinery with `std::print` |
| `std::source_location` | C++20 | assertion macros | call-site capture without `__FILE__` macros |
| Ranges (cold paths only) | C++20/23 | CLI, result formatting, tests | clarity where performance is irrelevant; banned from the hot loop (§20) |

## 20. Features explicitly avoided — and the reasoning

| Avoided | Why |
|---|---|
| **Modules** (incl. `import std`) | GCC 16 + CMake support still rough, and the **mixed-toolchain CUDA boundary (§6: GCC 16 host code vs GCC 15/clang CUDA TUs) makes a module-based boundary a hard blocker** — it must stay header-based. Cost (build-system fragility) exceeds benefit (compile time on a small codebase). Revisit as an experiment branch only |
| **Coroutines** | no async I/O exists; the hot loop must be a branch-predictable straight line. Any "elegant generator-of-candidates" coroutine shape would hide allocation and frame overhead inside the kernel |
| **Exceptions on hot/warm paths** | not `-fno-exceptions` (libstdc++ and cold startup code may use them) — but no `throw` reachable from engine loops; fallible operations return `std::expected`. Predictability + nvcc friendliness |
| **Virtual dispatch inside kernels** | indirect calls defeat inlining in the one loop that matters; virtuality lives only at the once-per-run engine boundary |
| **`std::regex`** | catastrophically slow, locale-dependent; matcher is compiled by us (and regex is excluded from hot path by design anyway) |
| **`shared_ptr`/general shared ownership** | ownership here is a tree with scoped lifetimes; refcounting would advertise sharing that must not exist |
| **`std::async`/OpenMP/TBB** | opaque scheduling vs. our requirement of exact per-thread state, affinity, and profile attribution |
| **Lock-free queues** (§12) | complexity without measurable benefit at our event rates |
| **`std::experimental::simd` for the kernel** | lacks widening/IFMA semantics; intrinsics behind our wrapper instead |
| **Clever UB-adjacent micro-opts** (type-punning unions, hand-rolled `restrict` casts) | UBSan/TSan cleanliness is a standing requirement; optimizations that sanitizers can't certify are rejected on policy |
| **Premature `[[assume]]`/builtin-expect spray** | each one requires a measured win to stay, otherwise deleted — annotations rot |

## 21. Self-critique: failure points, scalability limits, open risks

1. **Biggest risk — silently wrong field arithmetic** (esp. the second, GPU implementation). Mitigations are the test matrix (§16) and Verifier firewall (§9); residual risk: a bug that biases *which* keys are found but still finds valid ones — χ² smoke test partially covers; accepted.
2. **Performance estimates are estimates.** The 8–12 M/s/core scalar and 2–3× AVX2 numbers are informed by donna-class implementations and `mkp224o`, not measured here. Phase gates (below) exist so a miss forces re-planning, not denial. If AVX2 yields < 1.5×, the right answer may be "ship scalar+IFMA+CUDA and skip AVX2 polish" — committed to deciding by data.
3. **GCC 16 bleeding edge:** ICEs and codegen regressions are realistic. Mitigation: clang must stay green in CI as a second compiler; kernel asm diffs (§15) catch silent regressions.
4. **CUDA host-compiler split** is a permanent tax: nvcc (CUDA 13.3) accepts C++23 but caps the host compiler at GCC 15.x, so the CUDA TUs build with a different toolchain than the GCC 16 host code (§6). Mixed-toolchain ABI discipline at the boundary header *will* chafe, and every CUDA release shifts the matrix. Accepted consciously; pinned versions in CI.
5. **Unknown GPU model:** occupancy/tuning numbers can't be pre-planned; the design keeps every GPU knob runtime- or build-time-tunable, and the CUDA phase starts with a measurement matrix, not targets.
6. **SMT/AVX-512 questions are open** by design — flagged as hypotheses with kill criteria (§7, §13), not assumptions.
7. **Scalability ceiling:** single machine ≈ CPU O(10⁸)/s + GPU O(10⁹)/s ⇒ 10-char prefixes are ~weeks even on GPU; that's physics (`32^L`), not architecture. The honest scaling story is distribution (§17), which the stateless design makes nearly free — but result *collection/dedup-of-effort* across machines is unsolved-by-design (acceptable: duplicated work is wasted electricity, not incorrectness).
8. **Over-engineering watchlist:** the engine abstraction and kernel-concept machinery are justified by the *committed* second backend (CUDA). If CUDA were ever descoped, the right move is to collapse the abstraction — noted so the codebase doesn't keep paying for a ghost.

## 22. Phased roadmap with gates

| Phase | Deliverable | Exit gate |
|---|---|---|
| 0 | Scaffold: CMake presets, CI (clang+gcc, sanitizers), libsodium-naive reference generator (correct, slow), KAT suite, Python oracle, golden Tor files | e2e test finds a 2-char prefix; outputs verified by oracle + (manually) by real Tor |
| 1 | Scalar 51-bit field/point kernel + incremental search + batch inversion, CPU engine, stats/ETA | bit-identical to libsodium on property suite; ≥ 5 M/s/core measured; mkp224o gap quantified |
| 2 | AVX2 batched kernel + runtime dispatch; bench/profiling harness mature | ≥ 1.8× scalar or documented decision to deprioritize; TSan/ASan/UBSan green |
| 3 | CUDA engine + cross-validation + compute-sanitizer | GPU ≥ 10× whole-CPU throughput; identical pubkey streams vs CPU on shared seeds |
| 4 | IFMA path (if HW), tuning sweeps, docs, README honesty pass, mkp224o head-to-head | published reproducible benchmark; all phase-1–3 gates still green |

## Verification (how we'll know the eventual implementation is right)

- `ctest` runs: KATs, property/cross-impl suites, e2e prefix search with Python-oracle re-verification, golden Tor file round-trip.
- Sanitizer matrix (ASan/UBSan/TSan, compute-sanitizer) green.
- Manual acceptance: generate a real vanity key, install into a Tor `HiddenServiceDir`, confirm Tor serves the expected address.
- Bench: `onion bench` JSON vs phase gates above; side-by-side `mkp224o` run on the same machine.
