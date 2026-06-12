# cpp_onion

A high-performance Tor v3 vanity onion address generator in modern C++23.

**Status: Phase 1** — incremental ed25519 engine is the default (~62x the
naive baseline; see speed table below). The naive engine is retained for
cross-checking via `--engine naive`. See `docs/design.md` for the full
engineering design.

## Build

Requires: GCC 14+ (project targets GCC 16), CMake 3.28+, libsodium, Python 3.
Optionally uses the [mold](https://github.com/rui314/mold) linker when present.

    cmake --preset release
    cmake --build --preset release
    ctest --preset release

## Speed (measured, AMD/Intel, release build, `-O3`)

| Engine | Threads | Keys/s |
|---|---|---|
| naive (Phase 0, libsodium per-candidate) | 12 | 0.33 M/s |
| incremental (Phase 1, A+=8B + batch inversion) | 6 | 16.21 M/s |
| incremental (Phase 1, A+=8B + batch inversion) | 12 | 20.66 M/s |

Incremental engine at 12 threads is **~62x** the naive baseline.

## Usage

    ./build/release/src/cli/onion myname -o ./keys -t 12

Searches for `myname...onion` using the incremental engine by default.
Writes a Tor `HiddenServiceDir`-compatible directory: `hostname`,
`hs_ed25519_secret_key`, `hs_ed25519_public_key` (dir 0700, files 0600).
Point Tor at it:

Additional flags:

    # Use the naive engine for cross-checking
    ./build/release/src/cli/onion myname --engine naive -t 12

    # Benchmark throughput (impossible prefix, no output written)
    ./build/release/src/cli/onion zzzzzzzzzzzzzzzz --bench 10 --engine incremental -t 12

    HiddenServiceDir /path/to/keys/myname.../
    HiddenServicePort 80 127.0.0.1:8080

Every found key is independently re-verified (libsodium re-derivation) before
being written; `tools/verify_onion.py` provides a second, pure-Python oracle.

Expected work scales as 32^L for an L-char prefix: 6 chars is seconds-to-
minutes territory for the Phase 1 engine, 8+ chars wants the future GPU
backend (design doc §0).

## A note on vanity addresses

A recognizable prefix does not make an address verifiable — humans checking
only the first few characters is exactly what phishing relies on. Publish
and verify your full 56-character address.
