# cpp_onion

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

Benchmarked on an AMD Ryzen 5 4600H (Zen 2, 6C/12T):

| Engine | Flags | Threads | Keys/s |
|---|---|---|---|
| naive (libsodium per candidate) | | 12 | ~0.34 M/s |
| **incremental** (`A+=8B` + batched inversion) | `--simd off` | 12 | **~23.8 M/s** |
| **incremental + AVX2 4-wide** (Fe4 SoA, 4 lanes) | `--simd on` | 12 | **~21.3 M/s** |

The AVX2 4-wide (`--simd on`, default) path exercises 4 independent lanes per
step via __m256i field arithmetic; on Zen 2 the 256-bit execution units are two
fused 128-bit µops, so the theoretical 4× SIMD gain is offset by register
pressure — the measured result is within noise of the scalar engine on this
microarchitecture. Both engines produce bit-exact keys validated by libsodium
and the independent Python oracle (`tools/verify_onion.py`).

~70× the naive baseline (hardware/thermal dependent). Search cost scales as
`32^L` for an `L`-char prefix: ≤6 chars is seconds, 7 is ~minutes, 8 is hours.
The next big lever is a CUDA backend (the design targets ~10⁹ keys/s on a GPU).

## Build

Requires GCC 14+ (targets GCC 16), CMake ≥ 3.28, libsodium, Python 3. Uses the
[mold](https://github.com/rui314/mold) linker automatically when present.

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release        # KATs, libsodium cross-validation, e2e + oracle
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
