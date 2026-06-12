# cpp_onion

A high-performance Tor v3 vanity onion address generator in modern C++23.

**Status: Phase 0** — correct, slow reference engine (libsodium per-candidate
derivation). Phase 1 replaces it with an incremental ed25519 search
(~1000x faster); see `docs/design.md` for the full engineering design.

## Build

Requires: GCC 14+ (project targets GCC 16), CMake 3.28+, libsodium, Python 3.
Optionally uses the [mold](https://github.com/rui314/mold) linker when present.

    cmake --preset release
    cmake --build --preset release
    ctest --preset release

## Usage

    ./build/release/src/cli/onion myname -o ./keys -t 6

Searches for `myname...onion`, writes a Tor `HiddenServiceDir`-compatible
directory: `hostname`, `hs_ed25519_secret_key`, `hs_ed25519_public_key`
(dir 0700, files 0600). Point Tor at it:

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
