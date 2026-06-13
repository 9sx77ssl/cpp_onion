# cpp_onion

Полное руководство по командам: [USAGE.md](USAGE.md)

A high-performance **Tor v3 vanity onion address generator** in modern C++23.

Give it a prefix; it searches ed25519 keypairs until one's `.onion` address starts
with it, then writes the key material in the exact format Tor expects.

```
$ onion rsz -t 12 -o ./keys
found: rszzquwmwlnthf3ue5n7gtveepnb4224zrq42qutaqbxf4qpas2txmyd.onion -> ./keys/rsz...
```

## Why it's fast

A naive generator does a full ed25519 scalar multiplication (~250 point ops) per
candidate. cpp_onion instead uses an **incremental search** (the `mkp224o` approach):
pick one random base scalar `a₀`, then walk `A, A+8B, A+16B, …` — **one point
addition per candidate** — and recover the affine `y` for a whole batch with a single
field inversion (Montgomery's trick). That's ~100× less arithmetic per key.

The field and group arithmetic (`fe25519`, `ge25519`) are hand-written (5×51-bit
limbs, extended twisted-Edwards coordinates, mixed addition with an affine step
point) and **cross-validated bit-for-bit against libsodium** — every step of the
search provably matches `a₀+8i` times the basepoint. The release build uses
link-time optimization to inline the field multiplies into the point-addition loop.

### Speed (measured, release `-O3 -march=native -flto`)

Benchmarked on an AMD Ryzen 5 4600H (Zen 2, 6C/12T) + NVIDIA GTX 1650 (Turing,
sm_75, 14 SMs, 4 GB):

| Engine | Flags | Threads | Keys/s |
|---|---|---|---|
| naive (libsodium per candidate) | | 12 | ~0.34 M/s |
| **incremental** (`A+=8B` + batched inversion) — **CPU default** | | 12 | **~25 M/s** |
| incremental + AVX2 4-wide (Fe4 SoA, 4 lanes) | `--simd on` | 12 | ~21 M/s |
| **CUDA** (interleaved chains, batched inversion, M=1024, native 32-bit field) | `--engine cuda` | GPU | **~306 M/s** |

The CPU incremental engine is ~70–80× the naive baseline (hardware/thermal
dependent). The **CUDA backend is ~12.0× the 12-thread CPU incremental engine**
(306.2 vs 25.4 M keys/s, median of 3 × 8 s runs). All engines produce bit-exact
keys validated by libsodium and the independent Python oracle
(`tools/verify_onion.py`); the GPU device chain is additionally cross-checked
bit-for-bit against libsodium `base_noclamp(a0 + 8i)` in the test suite. The GPU
build is **leak-checked**: `compute-sanitizer --tool memcheck --leak-check full`
reports **0 bytes leaked / 0 errors** on both the test binary and a real
generate run (the per-epoch `cudaMalloc`/free loop is RAII-clean).

The default is the scalar incremental engine. An AVX2 4-wide path (`--simd on`)
processes 4 independent lanes per step via `__m256i` field arithmetic — but on
**Zen 2** the 256-bit units are two fused 128-bit µops and a 4-lane point spills
the register file, so it measures *slower* than scalar here; it's kept for
microarchitectures with full 256-bit throughput (Zen 4+, Intel AVX-512-class).

Search cost scales as `32^L` for an `L`-char prefix: ≤6 chars is seconds, 7 is
~minutes, 8 is hours. The **CUDA backend** (`--engine cuda`) now exists and is
the fastest engine: each device thread walks one interleaved `A += T·8B` chain
and amortizes a single field inversion across M=1024 points via a Montgomery
batch inversion. The device field uses **native 8×32-bit limbs** with
`mul.wide`/`mad.hi` carry chains — Turing has fast 32-bit IMAD and only emulates
64-bit/`__int128` multiplies, so the 32-bit field cut the kernel from 178 to
**96 registers/thread with zero spills**, raising occupancy. On the GTX 1650 it
sustains ~306 M keys/s (up from 275 with the `__int128` field) — honest and well
short of the design's optimistic ~10⁹ (that figure assumes a far larger,
higher-end GPU). The next levers would be a wider/newer GPU or warp-cooperative
batch inversion to grow the effective chain length.

## Benchmarks vs mkp224o

[mkp224o](https://github.com/cathugger/mkp224o) is the established reference vanity
onion generator. Head-to-head on the same machine (AMD Ryzen 5 4600H + NVIDIA GTX
1650), comparing against mkp224o's fastest amd64-51-30k build, both at 12 threads:

| Generator | Backend | Keys/s |
|---|---|---|
| mkp224o (amd64-51-30k) | CPU, 12 threads | ~28.9 M/s |
| cpp_onion incremental | CPU, 12 threads | ~28.1 M/s |
| cpp_onion CUDA | GPU (GTX 1650) | ~310 M/s (≈ 10.7× mkp224o) |

On CPU cpp_onion is on par with the reference (~28.1 vs ~28.9 M/s); the GPU backend
is the differentiator, since mkp224o has no GPU backend at all. The ~10.7× here is
GPU-hardware-bound: hundreds-of-× over mkp224o is achievable on a larger GPU, but on
this GTX 1650 it's ~11×.

## Build

Requires GCC 14+ (targets GCC 16), CMake ≥ 3.28, libsodium, Python 3. Uses the
[mold](https://github.com/rui314/mold) linker automatically when present.

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release        # KATs, libsodium cross-validation, e2e + oracle
```

Optional CUDA backend (requires the CUDA toolkit; the `cuda` preset uses
`nvcc` with a g++-15 host compiler and targets sm_75):

```sh
cmake --preset cuda
cmake --build --preset cuda
ctest --preset cuda           # adds the GPU device-chain libsodium xval + firewall
```

## Usage

```sh
# Generate (incremental + AVX2 4-wide is the default via --simd auto)
./build/release/src/cli/onion myname -t 12 -o ./keys

# Force scalar incremental path (no SIMD):
./build/release/src/cli/onion myname -t 12 --simd off

# Multiple prefixes (stops at the first match), naive engine for cross-checking,
# and a throughput benchmark against an impossible prefix:
./build/release/src/cli/onion abc xyz23 -t 12 -o ./keys
./build/release/src/cli/onion myname --engine naive -t 12
./build/release/src/cli/onion zzzzzzzzzzzzzzzz --bench 10 -t 12

# GPU (CUDA build): generate and benchmark
./build/cuda/src/cli/onion myname --engine cuda -o ./keys
./build/cuda/src/cli/onion zzzzzzzzzzzzzzzz --engine cuda --bench 10
./build/release/src/cli/onion zzzzzzzzzzzzzzzz --bench 10 -t 12 --simd off   # scalar A/B
```

Allowed prefix characters are base32: `a–z` and `2–7` (no `0 1 8 9`). The result
directory is Tor `HiddenServiceDir`-compatible (`hostname`,
`hs_ed25519_secret_key`, `hs_ed25519_public_key`; dir `0700`, files `0600`):

```
HiddenServiceDir /path/to/keys/myname.../
HiddenServicePort 80 127.0.0.1:8080
```

Every found key is independently re-derived with libsodium before being written,
and `tools/verify_onion.py` re-validates it with a from-scratch pure-Python ed25519
implementation that shares no code with the generator.

## Architecture

Layered libraries behind a swappable engine interface (`IEngine`):

```
crypto/  fe25519, ge25519, incremental stepper, key derivation, SHA3-256
core/    base32, Tor v3 address construction, prefix→byte/mask matcher
engine/  IEngine, StatsBoard, ResultQueue, NaiveCpuEngine, IncrementalCpuEngine, IncrementalCpuEngineX4
io/      verification firewall, Tor key-file writer
cli/     the `onion` binary
```

The full engineering design (algorithm, CPU/SIMD/GPU roadmap, threading,
security model) is in [`docs/design.md`](docs/design.md).

## A note on vanity addresses

A recognizable prefix does **not** make an address verifiable — people checking only
the first few characters of a 56-char address is exactly what phishing relies on.
Always publish and verify your full address.

## License

MIT — see [LICENSE](LICENSE).
