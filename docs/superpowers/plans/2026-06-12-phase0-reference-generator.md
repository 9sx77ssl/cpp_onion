# Phase 0: Scaffold + Reference Onion Generator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A correct (slow) end-to-end Tor v3 vanity onion generator — CLI, multithreaded libsodium-based search, independent verification, Tor-format output — plus the full test/oracle infrastructure that Phase 1's fast kernel will be validated against.

**Architecture:** Layered static libraries (`onion_crypto` ← `onion_core` ← `onion_engine_cpu`/`onion_io` ← `onion` CLI). Hot-path interfaces (`IEngine`, `CompiledMatcher`, `StatsBoard`, `ResultQueue`, Verifier firewall) are established now so Phase 1 swaps the engine without touching anything else. Design doc: `docs/design.md` (copied in Task 1 from `/home/rsz/.claude/plans/you-are-a-senior-fancy-garden.md`).

**Tech Stack:** C++23, GCC 16, CMake ≥ 3.28 + presets, libsodium (system, via pkg-config), Catch2 v3 + CLI11 (FetchContent, pinned), vendored compact SHA3, Python 3 oracle (stdlib only).

**Conventions:** All headers live under `src/` and are included as `#include "crypto/sha3.hpp"` etc. Namespaces: `onion::core`, `onion::crypto`, `onion::engine`, `onion::io`. Every commit ends with:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

**Verified test vectors used throughout (generated and cross-checked with Python hashlib before writing this plan):**

| Vector | Value |
|---|---|
| SHA3-256("") | `a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a` |
| SHA3-256("abc") | `3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532` |
| base32("foobar") | `mzxw6ytboi` (and: f→`my`, fo→`mzxq`, foo→`mzxw6`, foob→`mzxw6yq`, fooba→`mzxw6ytb`) |
| RFC 8032 TEST1 seed | `9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60` |
| RFC 8032 TEST1 pubkey | `d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a` |
| onion(TEST1 pubkey) | `25njqamcweflpvkl73j4szahhihoc4xt3ktcgjnpaingr5yhkenl5sid.onion` |
| onion(pubkey=bytes 00..1f) | `aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3kead.onion` |
| onion(pubkey=0x42×32) | `ijbeeqscijbeeqscijbeeqscijbeeqscijbeeqscijbeeqscijbezhid.onion` |

---

### Task 1: Repository scaffold and build skeleton

**Files:**
- Create: `.gitignore`, `CMakeLists.txt`, `CMakePresets.json`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`, `tests/test_sanity.cpp`, `docs/design.md`

- [ ] **Step 1: Initialize repo and copy the design doc**

```bash
cd /home/rsz/Desktop/cpp_onion
git init -b main
mkdir -p src tests docs tools
cp /home/rsz/.claude/plans/you-are-a-senior-fancy-garden.md docs/design.md
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
build/
.cache/
__pycache__/
```

- [ ] **Step 3: Create top-level `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.28)
project(cpp_onion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ONION_NATIVE "Build with -march=native" ON)
option(ONION_SANITIZE "Enable ASan+UBSan" OFF)

add_library(onion_options INTERFACE)
target_compile_options(onion_options INTERFACE -Wall -Wextra -Wpedantic)
if(ONION_NATIVE)
  target_compile_options(onion_options INTERFACE -march=native)
endif()
if(ONION_SANITIZE)
  target_compile_options(onion_options INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(onion_options INTERFACE -fsanitize=address,undefined)
endif()

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SODIUM REQUIRED IMPORTED_TARGET libsodium)

include(FetchContent)
FetchContent_Declare(Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.7.1)
FetchContent_Declare(CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.4.2)
FetchContent_MakeAvailable(Catch2 CLI11)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
```

- [ ] **Step 4: Create `CMakePresets.json`**

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "debug",
      "binaryDir": "build/debug",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Debug", "ONION_NATIVE": "OFF" }
    },
    {
      "name": "release",
      "binaryDir": "build/release",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Release", "ONION_NATIVE": "ON" }
    },
    {
      "name": "asan",
      "binaryDir": "build/asan",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Debug", "ONION_NATIVE": "OFF", "ONION_SANITIZE": "ON" }
    }
  ],
  "buildPresets": [
    { "name": "debug", "configurePreset": "debug" },
    { "name": "release", "configurePreset": "release" },
    { "name": "asan", "configurePreset": "asan" }
  ],
  "testPresets": [
    { "name": "debug", "configurePreset": "debug", "output": { "outputOnFailure": true } },
    { "name": "release", "configurePreset": "release", "output": { "outputOnFailure": true } },
    { "name": "asan", "configurePreset": "asan", "output": { "outputOnFailure": true } }
  ]
}
```

- [ ] **Step 5: Create `src/CMakeLists.txt`** (empty for now; libraries are appended by later tasks)

```cmake
# Library subdirectories are added here as they are created.
```

- [ ] **Step 6: Create `tests/test_sanity.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>

TEST_CASE("toolchain sanity") {
    CHECK(1 + 1 == 2);
}
```

- [ ] **Step 7: Create `tests/CMakeLists.txt`**

```cmake
include(Catch)

add_executable(onion_tests
  test_sanity.cpp
)
target_link_libraries(onion_tests PRIVATE Catch2::Catch2WithMain onion_options Threads::Threads)
catch_discover_tests(onion_tests)
```

- [ ] **Step 8: Configure, build, and run the sanity test**

Run: `cmake --preset debug && cmake --build --preset debug && ctest --preset debug`
Expected: configure succeeds (libsodium found, Catch2/CLI11 fetched), build succeeds, `100% tests passed, 1 test`.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "chore: scaffold CMake project with presets, Catch2, CLI11, libsodium

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Vendored SHA3-256 (`onion_crypto`)

Tor's address checksum uses SHA3-256, which libsodium does not provide. Compact vendored Keccak-f[1600]; cold path only, clarity over speed.

**Files:**
- Create: `src/crypto/sha3.hpp`, `src/crypto/sha3.cpp`, `src/crypto/CMakeLists.txt`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `tests/test_sha3.cpp`

- [ ] **Step 1: Write the failing test — `tests/test_sha3.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "crypto/sha3.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace {
std::string to_hex(std::span<const std::byte> bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    for (std::byte b : bytes) {
        out.push_back(digits[std::to_integer<unsigned>(b) >> 4]);
        out.push_back(digits[std::to_integer<unsigned>(b) & 0xf]);
    }
    return out;
}
std::span<const std::byte> as_bytes_sv(std::string_view s) {
    return {reinterpret_cast<const std::byte*>(s.data()), s.size()};
}
} // namespace

TEST_CASE("SHA3-256 NIST known answers") {
    using onion::crypto::Sha3_256;
    CHECK(to_hex(Sha3_256::hash(as_bytes_sv(""))) ==
          "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a");
    CHECK(to_hex(Sha3_256::hash(as_bytes_sv("abc"))) ==
          "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532");
}

TEST_CASE("SHA3-256 streaming equals one-shot") {
    using onion::crypto::Sha3_256;
    // 200 bytes crosses the 136-byte rate boundary
    std::string msg(200, 'x');
    Sha3_256 h;
    h.update(as_bytes_sv(std::string_view{msg}.substr(0, 7)));
    h.update(as_bytes_sv(std::string_view{msg}.substr(7, 150)));
    h.update(as_bytes_sv(std::string_view{msg}.substr(157)));
    CHECK(h.finalize() == Sha3_256::hash(as_bytes_sv(msg)));
}
```

- [ ] **Step 2: Run test to verify it fails**

Add `test_sha3.cpp` to `tests/CMakeLists.txt` `add_executable` list, then:
Run: `cmake --build --preset debug`
Expected: FAIL to compile — `crypto/sha3.hpp` not found.

- [ ] **Step 3: Write the implementation**

`src/crypto/sha3.hpp`:

```cpp
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace onion::crypto {

// SHA3-256 (FIPS 202). Compact, byte-oriented; cold path only (address
// checksums + verification), never on the search hot path.
class Sha3_256 {
public:
    static constexpr std::size_t digest_size = 32;
    static constexpr std::size_t rate = 136;  // bytes per block at 256-bit security

    void update(std::span<const std::byte> data);
    [[nodiscard]] std::array<std::byte, digest_size> finalize();

    [[nodiscard]] static std::array<std::byte, digest_size>
    hash(std::span<const std::byte> data) {
        Sha3_256 h;
        h.update(data);
        return h.finalize();
    }

private:
    void absorb_block();

    std::array<std::uint64_t, 25> state_{};
    std::array<std::byte, rate> buf_{};
    std::size_t buf_len_ = 0;
};

}  // namespace onion::crypto
```

`src/crypto/sha3.cpp`:

```cpp
#include "crypto/sha3.hpp"

#include <algorithm>

namespace onion::crypto {
namespace {

constexpr std::array<std::uint64_t, 24> kRoundConstants = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};
constexpr std::array<int, 24> kRotc = {1,  3,  6,  10, 15, 21, 28, 36,
                                       45, 55, 2,  14, 27, 41, 56, 8,
                                       25, 43, 62, 18, 39, 61, 20, 44};
constexpr std::array<int, 24> kPiln = {10, 7,  11, 17, 18, 3,  5,  16,
                                       8,  21, 24, 4,  15, 23, 19, 13,
                                       12, 2,  20, 14, 22, 9,  6,  1};

constexpr std::uint64_t rotl64(std::uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

void keccakf(std::array<std::uint64_t, 25>& st) {
    for (int round = 0; round < 24; ++round) {
        // theta
        std::uint64_t bc[5];
        for (int i = 0; i < 5; ++i)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        for (int i = 0; i < 5; ++i) {
            const std::uint64_t t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
        }
        // rho + pi
        std::uint64_t t = st[1];
        for (int i = 0; i < 24; ++i) {
            const int j = kPiln[i];
            const std::uint64_t tmp = st[j];
            st[j] = rotl64(t, kRotc[i]);
            t = tmp;
        }
        // chi
        for (int j = 0; j < 25; j += 5) {
            std::uint64_t b[5];
            for (int i = 0; i < 5; ++i) b[i] = st[j + i];
            for (int i = 0; i < 5; ++i)
                st[j + i] = b[i] ^ (~b[(i + 1) % 5] & b[(i + 2) % 5]);
        }
        // iota
        st[0] ^= kRoundConstants[round];
    }
}

}  // namespace

void Sha3_256::update(std::span<const std::byte> data) {
    for (std::byte b : data) {
        buf_[buf_len_++] = b;
        if (buf_len_ == rate) {
            absorb_block();
            buf_len_ = 0;
        }
    }
}

void Sha3_256::absorb_block() {
    for (std::size_t i = 0; i < rate / 8; ++i) {
        std::uint64_t lane = 0;
        for (int j = 7; j >= 0; --j)
            lane = (lane << 8) | std::to_integer<std::uint64_t>(buf_[i * 8 + std::size_t(j)]);
        state_[i] ^= lane;
    }
    keccakf(state_);
}

std::array<std::byte, Sha3_256::digest_size> Sha3_256::finalize() {
    std::fill(buf_.begin() + std::ptrdiff_t(buf_len_), buf_.end(), std::byte{0});
    buf_[buf_len_] = std::byte{0x06};   // SHA3 domain separation + pad10*1 start
    buf_[rate - 1] |= std::byte{0x80};  // pad10*1 end
    absorb_block();

    std::array<std::byte, digest_size> out;
    for (std::size_t i = 0; i < digest_size / 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            out[i * 8 + j] = std::byte((state_[i] >> (8 * j)) & 0xff);
    return out;
}

}  // namespace onion::crypto
```

`src/crypto/CMakeLists.txt`:

```cmake
add_library(onion_crypto STATIC
  sha3.cpp
)
target_include_directories(onion_crypto PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(onion_crypto
  PUBLIC PkgConfig::SODIUM
  PRIVATE onion_options)
```

`src/CMakeLists.txt` (full new content):

```cmake
# Library subdirectories are added here as they are created.
add_subdirectory(crypto)
```

`tests/CMakeLists.txt` (full new content):

```cmake
include(Catch)

add_executable(onion_tests
  test_sanity.cpp
  test_sha3.cpp
)
target_link_libraries(onion_tests PRIVATE Catch2::Catch2WithMain onion_options Threads::Threads
  onion_crypto)
catch_discover_tests(onion_tests)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build --preset debug && ctest --preset debug`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(crypto): vendored compact SHA3-256 with NIST KATs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Base32 codec (`onion_core`)

**Files:**
- Create: `src/core/base32.hpp`, `src/core/base32.cpp`, `src/core/CMakeLists.txt`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `tests/test_base32.cpp`

- [ ] **Step 1: Write the failing test — `tests/test_base32.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "core/base32.hpp"

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace {
std::span<const std::byte> as_bytes_sv(std::string_view s) {
    return {reinterpret_cast<const std::byte*>(s.data()), s.size()};
}
}

TEST_CASE("base32 encode RFC 4648 vectors (lowercase, unpadded)") {
    using onion::core::base32_encode;
    CHECK(base32_encode(as_bytes_sv("")) == "");
    CHECK(base32_encode(as_bytes_sv("f")) == "my");
    CHECK(base32_encode(as_bytes_sv("fo")) == "mzxq");
    CHECK(base32_encode(as_bytes_sv("foo")) == "mzxw6");
    CHECK(base32_encode(as_bytes_sv("foob")) == "mzxw6yq");
    CHECK(base32_encode(as_bytes_sv("fooba")) == "mzxw6ytb");
    CHECK(base32_encode(as_bytes_sv("foobar")) == "mzxw6ytboi");
}

TEST_CASE("base32 decode round-trips and rejects junk") {
    using onion::core::base32_decode;
    using onion::core::base32_encode;

    auto decoded = base32_decode("mzxw6ytboi");
    REQUIRE(decoded.has_value());
    CHECK(base32_encode(*decoded) == "mzxw6ytboi");
    CHECK(decoded->size() == 6);

    CHECK_FALSE(base32_decode("MZXW6").has_value());   // uppercase rejected (strict)
    CHECK_FALSE(base32_decode("mzx w").has_value());   // whitespace rejected
    CHECK_FALSE(base32_decode("mzxw1").has_value());   // '1' not in alphabet
    CHECK_FALSE(base32_decode("mz======").has_value()); // padding chars rejected
}
```

- [ ] **Step 2: Run test to verify it fails**

Add `test_base32.cpp` to `tests/CMakeLists.txt` sources and `onion_core` to its `target_link_libraries`, then:
Run: `cmake --build --preset debug`
Expected: FAIL to compile — `core/base32.hpp` not found (and `onion_core` target missing until Step 3's CMake edits).

- [ ] **Step 3: Write the implementation**

`src/core/base32.hpp`:

```cpp
#pragma once

#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace onion::core {

inline constexpr std::string_view kBase32Alphabet = "abcdefghijklmnopqrstuvwxyz234567";

enum class Base32Error { invalid_char, nonzero_padding };

// RFC 4648 base32, lowercase, no '=' padding (Tor onion convention).
[[nodiscard]] std::string base32_encode(std::span<const std::byte> in);
[[nodiscard]] std::expected<std::vector<std::byte>, Base32Error>
base32_decode(std::string_view s);

}  // namespace onion::core
```

`src/core/base32.cpp`:

```cpp
#include "core/base32.hpp"

#include <cstdint>

namespace onion::core {

std::string base32_encode(std::span<const std::byte> in) {
    std::string out;
    out.reserve((in.size() * 8 + 4) / 5);
    std::uint32_t acc = 0;
    int bits = 0;
    for (std::byte b : in) {
        acc = (acc << 8) | std::to_integer<std::uint32_t>(b);
        bits += 8;
        while (bits >= 5) {
            bits -= 5;
            out.push_back(kBase32Alphabet[(acc >> bits) & 0x1f]);
        }
    }
    if (bits > 0) out.push_back(kBase32Alphabet[(acc << (5 - bits)) & 0x1f]);
    return out;
}

std::expected<std::vector<std::byte>, Base32Error> base32_decode(std::string_view s) {
    std::vector<std::byte> out;
    out.reserve(s.size() * 5 / 8);
    std::uint32_t acc = 0;
    int bits = 0;
    for (char c : s) {
        const auto pos = kBase32Alphabet.find(c);
        if (pos == std::string_view::npos)
            return std::unexpected(Base32Error::invalid_char);
        acc = (acc << 5) | static_cast<std::uint32_t>(pos);
        bits += 5;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(std::byte((acc >> bits) & 0xff));
        }
    }
    if (bits > 0 && (acc & ((1u << bits) - 1)) != 0)
        return std::unexpected(Base32Error::nonzero_padding);
    return out;
}

}  // namespace onion::core
```

`src/core/CMakeLists.txt`:

```cmake
add_library(onion_core STATIC
  base32.cpp
)
target_include_directories(onion_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(onion_core
  PUBLIC onion_crypto
  PRIVATE onion_options)
```

(`onion_core` links `onion_crypto` because Task 4's address construction needs SHA3. Note: this flips the dependency arrow drawn in the design doc §3 — sha3 has no dependencies, so `crypto` is the bottom layer and `core` sits on it. Recorded here deliberately.)

`src/CMakeLists.txt` (full new content):

```cmake
# Library subdirectories are added here as they are created.
add_subdirectory(crypto)
add_subdirectory(core)
```

`tests/CMakeLists.txt` (full new content):

```cmake
include(Catch)

add_executable(onion_tests
  test_sanity.cpp
  test_sha3.cpp
  test_base32.cpp
)
target_link_libraries(onion_tests PRIVATE Catch2::Catch2WithMain onion_options Threads::Threads
  onion_crypto onion_core)
catch_discover_tests(onion_tests)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build --preset debug && ctest --preset debug`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(core): RFC 4648 base32 codec, lowercase unpadded

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Onion address construction (`onion_core`)

**Files:**
- Create: `src/core/onion_address.hpp`, `src/core/onion_address.cpp`
- Modify: `src/core/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `tests/test_onion_address.cpp`

- [ ] **Step 1: Write the failing test — `tests/test_onion_address.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "core/onion_address.hpp"

#include <array>
#include <cstddef>
#include <random>

namespace {
std::array<std::byte, 32> make_pubkey(unsigned char fill) {
    std::array<std::byte, 32> pk;
    pk.fill(std::byte{fill});
    return pk;
}
}

TEST_CASE("onion address known answers") {
    using onion::core::onion_address_from_pubkey;

    std::array<std::byte, 32> seq;
    for (std::size_t i = 0; i < 32; ++i) seq[i] = std::byte(i);
    CHECK(onion_address_from_pubkey(seq).to_string() ==
          "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3kead.onion");

    CHECK(onion_address_from_pubkey(make_pubkey(0x42)).to_string() ==
          "ijbeeqscijbeeqscijbeeqscijbeeqscijbeeqscijbeeqscijbezhid.onion");
}

TEST_CASE("all v3 addresses are 56 chars and end in 'd'") {
    using onion::core::onion_address_from_pubkey;
    std::mt19937_64 rng(12345);  // deterministic test, NOT key material
    for (int trial = 0; trial < 100; ++trial) {
        std::array<std::byte, 32> pk;
        for (auto& b : pk) b = std::byte(rng() & 0xff);
        const auto addr = onion_address_from_pubkey(pk);
        CHECK(addr.chars.size() == 56);
        CHECK(addr.chars[55] == 'd');  // version byte 0x03 -> final base32 char 'd'
    }
}

TEST_CASE("address decode recovers pubkey and validates checksum") {
    using onion::core::onion_address_from_pubkey;
    using onion::core::pubkey_from_onion_address;

    const auto pk = make_pubkey(0x42);
    const auto addr = onion_address_from_pubkey(pk).to_string();

    auto recovered = pubkey_from_onion_address(addr);
    REQUIRE(recovered.has_value());
    CHECK(*recovered == pk);

    // Corrupt one address character -> checksum failure
    auto bad = addr;
    bad[3] = (bad[3] == 'a') ? 'b' : 'a';
    CHECK_FALSE(pubkey_from_onion_address(bad).has_value());
}
```

- [ ] **Step 2: Run test to verify it fails**

Add `test_onion_address.cpp` to `tests/CMakeLists.txt` sources, then:
Run: `cmake --build --preset debug`
Expected: FAIL to compile — `core/onion_address.hpp` not found.

- [ ] **Step 3: Write the implementation**

`src/core/onion_address.hpp`:

```cpp
#pragma once

#include <array>
#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <string_view>

namespace onion::core {

enum class AddressError { bad_length, bad_base32, bad_version, bad_checksum };

struct OnionAddress {
    std::array<char, 56> chars;

    [[nodiscard]] std::string to_string() const {
        return std::string(chars.data(), chars.size()) + ".onion";
    }
    [[nodiscard]] std::string_view view() const {
        return {chars.data(), chars.size()};
    }
};

// rend-spec-v3: base32(PUBKEY || SHA3-256(".onion checksum" || PUBKEY || 0x03)[0..2) || 0x03)
[[nodiscard]] OnionAddress
onion_address_from_pubkey(std::span<const std::byte, 32> pubkey);

// Inverse for verification/tests: accepts with or without the ".onion" suffix,
// validates length, base32, version byte, and checksum.
[[nodiscard]] std::expected<std::array<std::byte, 32>, AddressError>
pubkey_from_onion_address(std::string_view address);

}  // namespace onion::core
```

`src/core/onion_address.cpp`:

```cpp
#include "core/onion_address.hpp"

#include "core/base32.hpp"
#include "crypto/sha3.hpp"

#include <algorithm>

namespace onion::core {
namespace {

constexpr std::string_view kChecksumTag = ".onion checksum";
constexpr std::byte kVersion{0x03};

std::array<std::byte, 2> checksum(std::span<const std::byte, 32> pubkey) {
    crypto::Sha3_256 h;
    h.update({reinterpret_cast<const std::byte*>(kChecksumTag.data()), kChecksumTag.size()});
    h.update(pubkey);
    h.update({&kVersion, 1});
    const auto digest = h.finalize();
    return {digest[0], digest[1]};
}

}  // namespace

OnionAddress onion_address_from_pubkey(std::span<const std::byte, 32> pubkey) {
    std::array<std::byte, 35> blob;
    std::ranges::copy(pubkey, blob.begin());
    const auto chk = checksum(pubkey);
    blob[32] = chk[0];
    blob[33] = chk[1];
    blob[34] = kVersion;

    const std::string encoded = base32_encode(blob);  // exactly 56 chars for 35 bytes
    OnionAddress out;
    std::ranges::copy(encoded, out.chars.begin());
    return out;
}

std::expected<std::array<std::byte, 32>, AddressError>
pubkey_from_onion_address(std::string_view address) {
    if (address.ends_with(".onion")) address.remove_suffix(6);
    if (address.size() != 56) return std::unexpected(AddressError::bad_length);

    const auto decoded = base32_decode(address);
    if (!decoded || decoded->size() != 35)
        return std::unexpected(AddressError::bad_base32);

    if ((*decoded)[34] != kVersion) return std::unexpected(AddressError::bad_version);

    std::array<std::byte, 32> pubkey;
    std::copy_n(decoded->begin(), 32, pubkey.begin());
    const auto chk = checksum(pubkey);
    if ((*decoded)[32] != chk[0] || (*decoded)[33] != chk[1])
        return std::unexpected(AddressError::bad_checksum);
    return pubkey;
}

}  // namespace onion::core
```

`src/core/CMakeLists.txt` (full new content):

```cmake
add_library(onion_core STATIC
  base32.cpp
  onion_address.cpp
)
target_include_directories(onion_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(onion_core
  PUBLIC onion_crypto
  PRIVATE onion_options)
```

`tests/CMakeLists.txt`: add `test_onion_address.cpp` to the sources list (everything else unchanged).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build --preset debug && ctest --preset debug`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(core): Tor v3 onion address construction and checksum validation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Key derivation via libsodium (`onion_crypto`)

**Files:**
- Create: `src/crypto/keys.hpp`, `src/crypto/keys.cpp`
- Modify: `src/crypto/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `tests/test_keys.cpp`

- [ ] **Step 1: Write the failing test — `tests/test_keys.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "crypto/keys.hpp"
#include "core/onion_address.hpp"

#include <array>
#include <cstddef>
#include <string_view>

namespace {
std::array<std::byte, 32> from_hex32(std::string_view hex) {
    auto nib = [](char c) -> unsigned {
        return (c >= 'a') ? unsigned(c - 'a' + 10) : unsigned(c - '0');
    };
    std::array<std::byte, 32> out;
    for (std::size_t i = 0; i < 32; ++i)
        out[i] = std::byte((nib(hex[2 * i]) << 4) | nib(hex[2 * i + 1]));
    return out;
}
}

TEST_CASE("RFC 8032 TEST1: seed -> expanded -> pubkey -> address") {
    using namespace onion;

    const auto seed = from_hex32(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
    const auto expected_pk = from_hex32(
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");

    const auto key = crypto::expand_seed(seed);

    // clamping invariants: low 3 bits clear, bit 254 set, bit 255 clear
    CHECK((std::to_integer<unsigned>(key.scalar[0]) & 0x07) == 0);
    CHECK((std::to_integer<unsigned>(key.scalar[31]) & 0xc0) == 0x40);

    const auto pk = crypto::pubkey_from_scalar(key.scalar);
    REQUIRE(pk.has_value());
    CHECK(*pk == expected_pk);

    CHECK(core::onion_address_from_pubkey(*pk).to_string() ==
          "25njqamcweflpvkl73j4szahhihoc4xt3ktcgjnpaingr5yhkenl5sid.onion");
}

TEST_CASE("ExpandedSecretKey::wipe zeroizes") {
    using namespace onion;
    const auto seed = from_hex32(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
    auto key = crypto::expand_seed(seed);
    key.wipe();
    std::array<std::byte, 32> zero{};
    CHECK(key.scalar == zero);
    CHECK(key.prf_prefix == zero);
}
```

- [ ] **Step 2: Run test to verify it fails**

Add `test_keys.cpp` to `tests/CMakeLists.txt` sources, then:
Run: `cmake --build --preset debug`
Expected: FAIL to compile — `crypto/keys.hpp` not found.

- [ ] **Step 3: Write the implementation**

`src/crypto/keys.hpp`:

```cpp
#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <span>

namespace onion::crypto {

// Tor's hs_ed25519_secret_key stores this *expanded* form (not a seed):
// 32-byte clamped scalar `a` followed by the 32-byte PRF prefix `RH`
// used for deterministic signature nonces.
struct ExpandedSecretKey {
    std::array<std::byte, 32> scalar{};
    std::array<std::byte, 32> prf_prefix{};

    ExpandedSecretKey() = default;
    ExpandedSecretKey(const ExpandedSecretKey&) = default;
    ExpandedSecretKey& operator=(const ExpandedSecretKey&) = default;
    ~ExpandedSecretKey() { wipe(); }

    void wipe() noexcept;
};

// RFC 8032: SHA-512(seed), clamp first half -> scalar; second half -> RH.
[[nodiscard]] ExpandedSecretKey expand_seed(std::span<const std::byte, 32> seed);

// A = a*B via libsodium (no re-clamping; scalar is already in clamped form).
// nullopt on libsodium failure (degenerate scalar) — callers skip the candidate.
[[nodiscard]] std::optional<std::array<std::byte, 32>>
pubkey_from_scalar(std::span<const std::byte, 32> scalar);

// Fill with CSPRNG bytes (getrandom-backed via libsodium).
void random_bytes(std::span<std::byte> out);

}  // namespace onion::crypto
```

`src/crypto/keys.cpp`:

```cpp
#include "crypto/keys.hpp"

#include <sodium.h>

#include <algorithm>
#include <cstdlib>

namespace onion::crypto {
namespace {

void ensure_sodium() {
    static const int rc = sodium_init();  // thread-safe magic static
    if (rc < 0) std::abort();             // no entropy source: unrecoverable
}

}  // namespace

void ExpandedSecretKey::wipe() noexcept {
    sodium_memzero(scalar.data(), scalar.size());
    sodium_memzero(prf_prefix.data(), prf_prefix.size());
}

ExpandedSecretKey expand_seed(std::span<const std::byte, 32> seed) {
    ensure_sodium();
    std::array<std::byte, 64> h;
    crypto_hash_sha512(reinterpret_cast<unsigned char*>(h.data()),
                       reinterpret_cast<const unsigned char*>(seed.data()),
                       seed.size());
    h[0] &= std::byte{0xf8};
    h[31] &= std::byte{0x7f};
    h[31] |= std::byte{0x40};

    ExpandedSecretKey key;
    std::copy_n(h.begin(), 32, key.scalar.begin());
    std::copy_n(h.begin() + 32, 32, key.prf_prefix.begin());
    sodium_memzero(h.data(), h.size());
    return key;
}

std::optional<std::array<std::byte, 32>>
pubkey_from_scalar(std::span<const std::byte, 32> scalar) {
    ensure_sodium();
    std::array<std::byte, 32> pk;
    if (crypto_scalarmult_ed25519_base_noclamp(
            reinterpret_cast<unsigned char*>(pk.data()),
            reinterpret_cast<const unsigned char*>(scalar.data())) != 0)
        return std::nullopt;
    return pk;
}

void random_bytes(std::span<std::byte> out) {
    ensure_sodium();
    randombytes_buf(out.data(), out.size());
}

}  // namespace onion::crypto
```

`src/crypto/CMakeLists.txt` (full new content):

```cmake
add_library(onion_crypto STATIC
  sha3.cpp
  keys.cpp
)
target_include_directories(onion_crypto PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(onion_crypto
  PUBLIC PkgConfig::SODIUM
  PRIVATE onion_options)
```

`tests/CMakeLists.txt`: add `test_keys.cpp` to the sources list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build --preset debug && ctest --preset debug`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(crypto): seed expansion and pubkey derivation via libsodium (RFC 8032 KAT)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Prefix matcher (`onion_core`)

Compiles a base32 prefix into `(bytes, mask)` over the leading pubkey bytes — the exact representation Phase 1's hot loop will consume.

**Files:**
- Create: `src/core/matcher.hpp`, `src/core/matcher.cpp`
- Modify: `src/core/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `tests/test_matcher.cpp`

- [ ] **Step 1: Write the failing test — `tests/test_matcher.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "core/matcher.hpp"
#include "core/onion_address.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string>

TEST_CASE("compile_prefix produces exact bytes and mask") {
    using onion::core::compile_prefix;
    // "abc" -> 5-bit groups 00000 00001 00010 -> bitstream 00000000 0100010x
    const auto pat = compile_prefix("abc");
    REQUIRE(pat.has_value());
    CHECK(pat->nbytes == 2);
    CHECK(pat->bytes[0] == 0x00);
    CHECK(pat->bytes[1] == 0x44);
    CHECK(pat->mask[0] == 0xff);
    CHECK(pat->mask[1] == 0xfe);
    CHECK(pat->prefix == "abc");
}

TEST_CASE("compile_prefix validates input") {
    using onion::core::compile_prefix;
    CHECK_FALSE(compile_prefix("").has_value());
    CHECK_FALSE(compile_prefix("ab1").has_value());   // '1' not in base32 alphabet
    CHECK_FALSE(compile_prefix("ab0").has_value());   // '0' not in alphabet
    CHECK_FALSE(compile_prefix("ABC").has_value());   // uppercase rejected (strict)
    CHECK_FALSE(compile_prefix(std::string(50, 'a')).has_value());  // > 49 chars
    CHECK(compile_prefix(std::string(49, 'a')).has_value());
}

TEST_CASE("matches agrees with naive encode-then-compare reference") {
    using namespace onion::core;
    std::mt19937_64 rng(99);  // deterministic test, NOT key material
    int positives = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        std::array<std::byte, 32> pk;
        for (auto& b : pk) b = std::byte(rng() & 0xff);
        const std::string addr{onion_address_from_pubkey(pk).view()};

        // True prefixes of this address must match, for every length 1..12
        const std::size_t len = 1 + trial % 12;
        const auto pat = compile_prefix(addr.substr(0, len));
        REQUIRE(pat.has_value());
        CHECK(matches(*pat, pk));

        // A single mutated final char must not match
        std::string bad = addr.substr(0, len);
        bad.back() = (bad.back() == 'a') ? 'b' : 'a';
        const auto bad_pat = compile_prefix(bad);
        REQUIRE(bad_pat.has_value());
        if (matches(*bad_pat, pk)) ++positives;
    }
    CHECK(positives == 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Add `test_matcher.cpp` to `tests/CMakeLists.txt` sources, then:
Run: `cmake --build --preset debug`
Expected: FAIL to compile — `core/matcher.hpp` not found.

- [ ] **Step 3: Write the implementation**

`src/core/matcher.hpp`:

```cpp
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>

namespace onion::core {

enum class MatcherError { empty, invalid_char, too_long };

// A base32 prefix compiled to a byte pattern + mask over the leading
// public-key bytes. Limit 49 chars: beyond stream bit 247 the x-sign bit
// (bit 248) participates, and Phase 1's hot path matches on y alone
// (design doc §0).
struct CompiledPattern {
    std::array<std::uint8_t, 32> bytes{};
    std::array<std::uint8_t, 32> mask{};
    std::size_t nbytes = 0;
    std::string prefix;
};

inline constexpr std::size_t kMaxPrefixLen = 49;

[[nodiscard]] std::expected<CompiledPattern, MatcherError>
compile_prefix(std::string_view prefix);

[[nodiscard]] inline bool matches(const CompiledPattern& p,
                                  std::span<const std::byte, 32> pubkey) noexcept {
    for (std::size_t i = 0; i < p.nbytes; ++i)
        if ((std::to_integer<std::uint8_t>(pubkey[i]) & p.mask[i]) != p.bytes[i])
            return false;
    return true;
}

}  // namespace onion::core
```

`src/core/matcher.cpp`:

```cpp
#include "core/matcher.hpp"

#include "core/base32.hpp"

namespace onion::core {

std::expected<CompiledPattern, MatcherError> compile_prefix(std::string_view prefix) {
    if (prefix.empty()) return std::unexpected(MatcherError::empty);
    if (prefix.size() > kMaxPrefixLen) return std::unexpected(MatcherError::too_long);

    CompiledPattern out;
    out.prefix = std::string(prefix);

    std::size_t bit = 0;
    for (char c : prefix) {
        const auto pos = kBase32Alphabet.find(c);
        if (pos == std::string_view::npos)
            return std::unexpected(MatcherError::invalid_char);
        for (int k = 4; k >= 0; --k, ++bit) {
            const auto byte_idx = bit / 8;
            const auto bit_mask = std::uint8_t(0x80u >> (bit % 8));
            if ((pos >> k) & 1) out.bytes[byte_idx] |= bit_mask;
            out.mask[byte_idx] |= bit_mask;
        }
    }
    out.nbytes = (bit + 7) / 8;
    return out;
}

}  // namespace onion::core
```

`src/core/CMakeLists.txt` (full new content):

```cmake
add_library(onion_core STATIC
  base32.cpp
  onion_address.cpp
  matcher.cpp
)
target_include_directories(onion_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(onion_core
  PUBLIC onion_crypto
  PRIVATE onion_options)
```

`tests/CMakeLists.txt`: add `test_matcher.cpp` to the sources list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build --preset debug && ctest --preset debug`
Expected: PASS (13 tests).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(core): prefix matcher compiled to byte+mask patterns

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Engine interfaces — `StatsBoard`, `ResultQueue`, `IEngine` (`onion_engine`)

**Files:**
- Create: `src/engine/engine.hpp` (header-only target `onion_engine`)
- Create: `src/engine/CMakeLists.txt`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `tests/test_engine_primitives.cpp`

- [ ] **Step 1: Write the failing test — `tests/test_engine_primitives.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "engine/engine.hpp"

#include <chrono>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

TEST_CASE("StatsBoard sums per-worker totals written from worker threads") {
    onion::engine::StatsBoard stats(4);
    {
        std::vector<std::jthread> workers;
        for (unsigned w = 0; w < 4; ++w)
            workers.emplace_back([&stats, w] {
                for (std::uint64_t total = 0; total <= 1000; total += 100)
                    stats.set(w, total);
            });
    }  // join
    CHECK(stats.total() == 4000);
}

TEST_CASE("ResultQueue delivers pushed candidates") {
    onion::engine::ResultQueue q;
    onion::engine::MatchCandidate c;
    c.pattern_index = 7;
    q.push(c);
    auto got = q.pop_wait_for(100ms);
    REQUIRE(got.has_value());
    CHECK(got->pattern_index == 7);
}

TEST_CASE("ResultQueue pop times out empty") {
    onion::engine::ResultQueue q;
    const auto t0 = std::chrono::steady_clock::now();
    CHECK_FALSE(q.pop_wait_for(50ms).has_value());
    CHECK(std::chrono::steady_clock::now() - t0 >= 50ms);
}
```

- [ ] **Step 2: Run test to verify it fails**

Add `test_engine_primitives.cpp` to `tests/CMakeLists.txt` sources and `onion_engine` to its link libraries, then:
Run: `cmake --build --preset debug`
Expected: FAIL to compile — `engine/engine.hpp` not found.

- [ ] **Step 3: Write the implementation**

`src/engine/engine.hpp`:

```cpp
#pragma once

#include "crypto/keys.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <new>
#include <optional>
#include <stop_token>
#include <vector>

namespace onion::engine {

// What an engine reports on a hit. NOT yet a result: the Verifier (onion_io)
// independently re-derives the pubkey before anything reaches disk
// (design doc §1 "Match != result").
struct MatchCandidate {
    crypto::ExpandedSecretKey secret;
    std::array<std::byte, 32> claimed_pubkey{};
    std::size_t pattern_index = 0;
};

// Single-writer-per-slot throughput counters. Workers own plain uint64_t
// slots (padded against false sharing); cross-thread access goes through
// relaxed atomic_ref. No RMW anywhere (design doc §11).
class StatsBoard {
public:
    explicit StatsBoard(unsigned workers) : slots_(workers) {}

    void set(unsigned worker, std::uint64_t total) noexcept {
        std::atomic_ref<std::uint64_t>(slots_[worker].value)
            .store(total, std::memory_order_relaxed);
    }

    [[nodiscard]] std::uint64_t total() const noexcept {
        std::uint64_t sum = 0;
        for (const auto& slot : slots_)
            // atomic_ref requires a mutable lvalue; the underlying object is
            // never actually written through this path.
            sum += std::atomic_ref<std::uint64_t>(
                       const_cast<std::uint64_t&>(slot.value))
                       .load(std::memory_order_relaxed);
        return sum;
    }

private:
    struct alignas(std::hardware_destructive_interference_size) Slot {
        std::uint64_t value = 0;
    };
    std::vector<Slot> slots_;
};

// Mutex+condvar MPSC queue. Matches are rare (minutes-to-hours apart);
// lock-free here is deliberately declined (design doc §12).
class ResultQueue {
public:
    void push(MatchCandidate c) {
        {
            std::lock_guard lock(mutex_);
            queue_.push_back(std::move(c));
        }
        cv_.notify_one();
    }

    [[nodiscard]] std::optional<MatchCandidate>
    pop_wait_for(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); }))
            return std::nullopt;
        MatchCandidate c = std::move(queue_.front());
        queue_.pop_front();
        return c;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<MatchCandidate> queue_;
};

// Engine boundary: called once per run; virtual dispatch is fine here
// (design doc §1). run() blocks until the stop_token fires.
class IEngine {
public:
    virtual ~IEngine() = default;
    virtual void run(std::stop_token stop) = 0;
};

}  // namespace onion::engine
```

`src/engine/CMakeLists.txt`:

```cmake
add_library(onion_engine INTERFACE)
target_include_directories(onion_engine INTERFACE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(onion_engine INTERFACE onion_crypto)
```

`src/CMakeLists.txt` (full new content):

```cmake
# Library subdirectories are added here as they are created.
add_subdirectory(crypto)
add_subdirectory(core)
add_subdirectory(engine)
```

`tests/CMakeLists.txt` (full new content):

```cmake
include(Catch)

add_executable(onion_tests
  test_sanity.cpp
  test_sha3.cpp
  test_base32.cpp
  test_onion_address.cpp
  test_keys.cpp
  test_matcher.cpp
  test_engine_primitives.cpp
)
target_link_libraries(onion_tests PRIVATE Catch2::Catch2WithMain onion_options Threads::Threads
  onion_crypto onion_core onion_engine)
catch_discover_tests(onion_tests)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build --preset debug && ctest --preset debug`
Expected: PASS (16 tests).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(engine): IEngine boundary, padded StatsBoard, mutex ResultQueue

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Naive CPU engine (`onion_engine_cpu`)

The Phase 0 reference engine: fresh random seed per candidate, full libsodium derivation. ~10⁴–10⁵ keys/s/core — slow by design; Phase 1 replaces exactly this class behind `IEngine`.

**Files:**
- Create: `src/engine/cpu/naive_engine.hpp`, `src/engine/cpu/naive_engine.cpp`, `src/engine/cpu/CMakeLists.txt`
- Modify: `src/engine/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `tests/test_naive_engine.cpp`

- [ ] **Step 1: Write the failing test — `tests/test_naive_engine.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "engine/cpu/naive_engine.hpp"
#include "core/matcher.hpp"
#include "crypto/keys.hpp"

#include <chrono>
#include <stop_token>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

TEST_CASE("naive engine finds a single-char prefix and reports a true candidate") {
    using namespace onion;
    auto pat = core::compile_prefix("a");
    REQUIRE(pat.has_value());
    std::vector<core::CompiledPattern> patterns{*pat};

    engine::ResultQueue queue;
    engine::StatsBoard stats(2);
    engine::NaiveCpuEngine eng(patterns, 2, queue, stats);

    std::stop_source stop;
    std::jthread runner([&] { eng.run(stop.get_token()); });

    auto cand = queue.pop_wait_for(30s);  // expected ~32 tries; 30s is paranoia
    stop.request_stop();
    REQUIRE(cand.has_value());

    auto pk = crypto::pubkey_from_scalar(cand->secret.scalar);
    REQUIRE(pk.has_value());
    CHECK(*pk == cand->claimed_pubkey);
    CHECK(cand->pattern_index == 0);
    CHECK(core::matches(patterns[0], *pk));

    runner.join();
    CHECK(stats.total() > 0);
}

TEST_CASE("naive engine honors stop_token promptly") {
    using namespace onion;
    // 12-char prefix: will never match within the test
    auto pat = core::compile_prefix("aaaaaaaaaaaa");
    REQUIRE(pat.has_value());
    std::vector<core::CompiledPattern> patterns{*pat};

    engine::ResultQueue queue;
    engine::StatsBoard stats(2);
    engine::NaiveCpuEngine eng(patterns, 2, queue, stats);

    std::stop_source stop;
    std::jthread runner([&] { eng.run(stop.get_token()); });
    std::this_thread::sleep_for(100ms);

    const auto t0 = std::chrono::steady_clock::now();
    stop.request_stop();
    runner.join();
    CHECK(std::chrono::steady_clock::now() - t0 < 2s);
}
```

- [ ] **Step 2: Run test to verify it fails**

Add `test_naive_engine.cpp` to `tests/CMakeLists.txt` sources and `onion_engine_cpu` to its link libraries, then:
Run: `cmake --build --preset debug`
Expected: FAIL to compile — `engine/cpu/naive_engine.hpp` not found.

- [ ] **Step 3: Write the implementation**

`src/engine/cpu/naive_engine.hpp`:

```cpp
#pragma once

#include "core/matcher.hpp"
#include "engine/engine.hpp"

#include <stop_token>
#include <vector>

namespace onion::engine {

// Phase 0 reference engine: fresh CSPRNG seed per candidate, full libsodium
// derivation. Correct and slow; the architectural placeholder Phase 1's
// incremental engine replaces.
class NaiveCpuEngine final : public IEngine {
public:
    NaiveCpuEngine(std::vector<core::CompiledPattern> patterns,
                   unsigned num_threads, ResultQueue& results, StatsBoard& stats)
        : patterns_(std::move(patterns)),
          num_threads_(num_threads),
          results_(results),
          stats_(stats) {}

    void run(std::stop_token stop) override;

private:
    void worker(std::stop_token stop, unsigned index);

    std::vector<core::CompiledPattern> patterns_;
    unsigned num_threads_;
    ResultQueue& results_;
    StatsBoard& stats_;
};

}  // namespace onion::engine
```

`src/engine/cpu/naive_engine.cpp`:

```cpp
#include "engine/cpu/naive_engine.hpp"

#include "crypto/keys.hpp"

#include <sodium.h>

#include <thread>

namespace onion::engine {

void NaiveCpuEngine::run(std::stop_token stop) {
    std::vector<std::jthread> workers;
    workers.reserve(num_threads_);
    for (unsigned i = 0; i < num_threads_; ++i)
        workers.emplace_back([this, stop, i] { worker(stop, i); });
}  // jthread destructors join; workers exit when `stop` fires

void NaiveCpuEngine::worker(std::stop_token stop, unsigned index) {
    constexpr int kCheckInterval = 256;  // naive derivation ~20-60us/candidate
    std::array<std::byte, 32> seed;
    std::uint64_t local_total = 0;

    while (!stop.stop_requested()) {
        for (int i = 0; i < kCheckInterval; ++i) {
            crypto::random_bytes(seed);
            const auto secret = crypto::expand_seed(seed);
            const auto pk = crypto::pubkey_from_scalar(secret.scalar);
            if (!pk) continue;  // degenerate scalar; skip (defensive)
            ++local_total;
            for (std::size_t p = 0; p < patterns_.size(); ++p)
                if (core::matches(patterns_[p], *pk))
                    results_.push({secret, *pk, p});
        }
        stats_.set(index, local_total);
    }
    sodium_memzero(seed.data(), seed.size());
}

}  // namespace onion::engine
```

`src/engine/cpu/CMakeLists.txt`:

```cmake
add_library(onion_engine_cpu STATIC
  naive_engine.cpp
)
target_link_libraries(onion_engine_cpu
  PUBLIC onion_engine onion_core
  PRIVATE onion_options Threads::Threads)
```

`src/engine/CMakeLists.txt` (full new content):

```cmake
add_library(onion_engine INTERFACE)
target_include_directories(onion_engine INTERFACE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(onion_engine INTERFACE onion_crypto)

add_subdirectory(cpu)
```

`tests/CMakeLists.txt`: add `test_naive_engine.cpp` to sources and `onion_engine_cpu` to link libraries.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build --preset debug && ctest --preset debug`
Expected: PASS (18 tests). The single-char search typically completes in well under a second even in Debug.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(engine): naive multithreaded libsodium reference engine

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Verifier (`onion_io`)

**Files:**
- Create: `src/io/verifier.hpp`, `src/io/verifier.cpp`, `src/io/CMakeLists.txt`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `tests/test_verifier.cpp`

- [ ] **Step 1: Write the failing test — `tests/test_verifier.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "io/verifier.hpp"
#include "core/matcher.hpp"
#include "crypto/keys.hpp"

#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

namespace {
std::array<std::byte, 32> from_hex32(std::string_view hex) {
    auto nib = [](char c) -> unsigned {
        return (c >= 'a') ? unsigned(c - 'a' + 10) : unsigned(c - '0');
    };
    std::array<std::byte, 32> out;
    for (std::size_t i = 0; i < 32; ++i)
        out[i] = std::byte((nib(hex[2 * i]) << 4) | nib(hex[2 * i + 1]));
    return out;
}

onion::engine::MatchCandidate test1_candidate() {
    const auto seed = from_hex32(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
    onion::engine::MatchCandidate c;
    c.secret = onion::crypto::expand_seed(seed);
    c.claimed_pubkey = *onion::crypto::pubkey_from_scalar(c.secret.scalar);
    c.pattern_index = 0;
    return c;
}
}

TEST_CASE("verifier accepts a genuine candidate") {
    using namespace onion;
    // TEST1's address starts with '2'
    std::vector patterns{*core::compile_prefix("2")};
    const auto result = io::verify(test1_candidate(), patterns);
    REQUIRE(result.has_value());
    CHECK(result->address.to_string() ==
          "25njqamcweflpvkl73j4szahhihoc4xt3ktcgjnpaingr5yhkenl5sid.onion");
}

TEST_CASE("verifier rejects a tampered pubkey") {
    using namespace onion;
    std::vector patterns{*core::compile_prefix("2")};
    auto cand = test1_candidate();
    cand.claimed_pubkey[5] ^= std::byte{0x01};
    const auto result = io::verify(cand, patterns);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error() == io::VerifyError::pubkey_mismatch);
}

TEST_CASE("verifier rejects a candidate that does not match its pattern") {
    using namespace onion;
    std::vector patterns{*core::compile_prefix("zz")};  // TEST1 addr starts "25"
    const auto result = io::verify(test1_candidate(), patterns);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error() == io::VerifyError::pattern_mismatch);
}
```

- [ ] **Step 2: Run test to verify it fails**

Add `test_verifier.cpp` to `tests/CMakeLists.txt` sources and `onion_io` to link libraries, then:
Run: `cmake --build --preset debug`
Expected: FAIL to compile — `io/verifier.hpp` not found.

- [ ] **Step 3: Write the implementation**

`src/io/verifier.hpp`:

```cpp
#pragma once

#include "core/matcher.hpp"
#include "core/onion_address.hpp"
#include "engine/engine.hpp"

#include <expected>
#include <span>

namespace onion::io {

enum class VerifyError {
    pubkey_mismatch,
    bad_pattern_index,
    pattern_mismatch,
    address_prefix_mismatch,
};

struct VerifiedResult {
    crypto::ExpandedSecretKey secret;
    std::array<std::byte, 32> pubkey{};
    core::OnionAddress address{};
};

// The firewall between engine arithmetic and user-visible output: re-derives
// the pubkey via libsodium and re-checks the match before anything is
// written (design doc §9). A failure here means an engine bug.
[[nodiscard]] std::expected<VerifiedResult, VerifyError>
verify(const engine::MatchCandidate& candidate,
       std::span<const core::CompiledPattern> patterns);

}  // namespace onion::io
```

`src/io/verifier.cpp`:

```cpp
#include "io/verifier.hpp"

#include "crypto/keys.hpp"

#include <string_view>

namespace onion::io {

std::expected<VerifiedResult, VerifyError>
verify(const engine::MatchCandidate& candidate,
       std::span<const core::CompiledPattern> patterns) {
    const auto pk = crypto::pubkey_from_scalar(candidate.secret.scalar);
    if (!pk || *pk != candidate.claimed_pubkey)
        return std::unexpected(VerifyError::pubkey_mismatch);

    if (candidate.pattern_index >= patterns.size())
        return std::unexpected(VerifyError::bad_pattern_index);
    const auto& pattern = patterns[candidate.pattern_index];

    if (!core::matches(pattern, *pk))
        return std::unexpected(VerifyError::pattern_mismatch);

    const auto address = core::onion_address_from_pubkey(*pk);
    if (!address.view().starts_with(pattern.prefix))
        return std::unexpected(VerifyError::address_prefix_mismatch);

    return VerifiedResult{candidate.secret, *pk, address};
}

}  // namespace onion::io
```

`src/io/CMakeLists.txt`:

```cmake
add_library(onion_io STATIC
  verifier.cpp
)
target_link_libraries(onion_io
  PUBLIC onion_core onion_engine
  PRIVATE onion_options)
```

`src/CMakeLists.txt` (full new content):

```cmake
# Library subdirectories are added here as they are created.
add_subdirectory(crypto)
add_subdirectory(core)
add_subdirectory(engine)
add_subdirectory(io)
```

`tests/CMakeLists.txt`: add `test_verifier.cpp` to sources and `onion_io` to link libraries.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build --preset debug && ctest --preset debug`
Expected: PASS (21 tests).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(io): independent verification firewall before output

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Tor key file writer (`onion_io`)

**Files:**
- Create: `src/io/tor_key_writer.hpp`, `src/io/tor_key_writer.cpp`
- Modify: `src/io/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `tests/test_tor_key_writer.cpp`

- [ ] **Step 1: Write the failing test — `tests/test_tor_key_writer.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "io/tor_key_writer.hpp"
#include "io/verifier.hpp"
#include "core/matcher.hpp"
#include "crypto/keys.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {
std::array<std::byte, 32> from_hex32(std::string_view hex) {
    auto nib = [](char c) -> unsigned {
        return (c >= 'a') ? unsigned(c - 'a' + 10) : unsigned(c - '0');
    };
    std::array<std::byte, 32> out;
    for (std::size_t i = 0; i < 32; ++i)
        out[i] = std::byte((nib(hex[2 * i]) << 4) | nib(hex[2 * i + 1]));
    return out;
}

fs::path make_temp_dir() {
    std::string tmpl = (fs::temp_directory_path() / "onion_test_XXXXXX").string();
    REQUIRE(::mkdtemp(tmpl.data()) != nullptr);
    return fs::path(tmpl);
}

std::vector<std::byte> read_all(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    REQUIRE(in.good());
    std::vector<char> raw{std::istreambuf_iterator<char>(in),
                          std::istreambuf_iterator<char>()};
    return {reinterpret_cast<std::byte*>(raw.data()),
            reinterpret_cast<std::byte*>(raw.data()) + raw.size()};
}
}

TEST_CASE("writer emits Tor-format files with exact bytes and 0700/0600 permissions") {
    using namespace onion;

    const auto seed = from_hex32(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
    engine::MatchCandidate cand;
    cand.secret = crypto::expand_seed(seed);
    cand.claimed_pubkey = *crypto::pubkey_from_scalar(cand.secret.scalar);
    cand.pattern_index = 0;
    std::vector patterns{*core::compile_prefix("2")};
    const auto verified = io::verify(cand, patterns);
    REQUIRE(verified.has_value());

    const auto outdir = make_temp_dir();
    const auto dir = io::write_tor_keys(*verified, outdir);
    REQUIRE(dir.has_value());

    constexpr std::string_view kAddr56 =
        "25njqamcweflpvkl73j4szahhihoc4xt3ktcgjnpaingr5yhkenl5sid";
    CHECK(dir->filename() == fs::path(kAddr56));
    CHECK(fs::status(*dir).permissions() ==
          (fs::perms::owner_read | fs::perms::owner_write | fs::perms::owner_exec));

    // hostname
    {
        std::ifstream host(*dir / "hostname");
        std::string line;
        std::getline(host, line);
        CHECK(line == std::string(kAddr56) + ".onion");
    }

    // secret key file: 32-byte tag header + scalar + PRF prefix
    {
        const auto bytes = read_all(*dir / "hs_ed25519_secret_key");
        REQUIRE(bytes.size() == 96);
        constexpr std::string_view tag = "== ed25519v1-secret: type0 ==";
        CHECK(std::memcmp(bytes.data(), tag.data(), tag.size()) == 0);
        for (std::size_t i = tag.size(); i < 32; ++i)
            CHECK(bytes[i] == std::byte{0});
        CHECK(std::memcmp(bytes.data() + 32, verified->secret.scalar.data(), 32) == 0);
        CHECK(std::memcmp(bytes.data() + 64, verified->secret.prf_prefix.data(), 32) == 0);
        CHECK(fs::status(*dir / "hs_ed25519_secret_key").permissions() ==
              (fs::perms::owner_read | fs::perms::owner_write));
    }

    // public key file: 32-byte tag header + pubkey
    {
        const auto bytes = read_all(*dir / "hs_ed25519_public_key");
        REQUIRE(bytes.size() == 64);
        constexpr std::string_view tag = "== ed25519v1-public: type0 ==";
        CHECK(std::memcmp(bytes.data(), tag.data(), tag.size()) == 0);
        CHECK(std::memcmp(bytes.data() + 32, verified->pubkey.data(), 32) == 0);
    }

    fs::remove_all(outdir);
}

TEST_CASE("writer refuses to overwrite an existing result") {
    using namespace onion;
    const auto seed = from_hex32(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
    engine::MatchCandidate cand;
    cand.secret = crypto::expand_seed(seed);
    cand.claimed_pubkey = *crypto::pubkey_from_scalar(cand.secret.scalar);
    cand.pattern_index = 0;
    std::vector patterns{*core::compile_prefix("2")};
    const auto verified = io::verify(cand, patterns);
    REQUIRE(verified.has_value());

    const auto outdir = make_temp_dir();
    REQUIRE(io::write_tor_keys(*verified, outdir).has_value());
    CHECK_FALSE(io::write_tor_keys(*verified, outdir).has_value());  // O_EXCL
    fs::remove_all(outdir);
}
```

- [ ] **Step 2: Run test to verify it fails**

Add `test_tor_key_writer.cpp` to `tests/CMakeLists.txt` sources, then:
Run: `cmake --build --preset debug`
Expected: FAIL to compile — `io/tor_key_writer.hpp` not found.

- [ ] **Step 3: Write the implementation**

`src/io/tor_key_writer.hpp`:

```cpp
#pragma once

#include "io/verifier.hpp"

#include <expected>
#include <filesystem>

namespace onion::io {

enum class WriteError { create_dir_failed, open_failed, write_failed };

// Writes <outdir>/<56-char-address>/{hostname, hs_ed25519_secret_key,
// hs_ed25519_public_key} in the exact format Tor's HiddenServiceDir expects.
// Directory 0700, files 0600, O_EXCL (never overwrites). Returns the
// created directory.
[[nodiscard]] std::expected<std::filesystem::path, WriteError>
write_tor_keys(const VerifiedResult& result, const std::filesystem::path& outdir);

}  // namespace onion::io
```

`src/io/tor_key_writer.cpp`:

```cpp
#include "io/tor_key_writer.hpp"

#include <fcntl.h>
#include <sodium.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <string_view>

namespace fs = std::filesystem;

namespace onion::io {
namespace {

constexpr std::string_view kSecretTag = "== ed25519v1-secret: type0 ==";
constexpr std::string_view kPublicTag = "== ed25519v1-public: type0 ==";

std::array<std::byte, 32> make_header(std::string_view tag) {
    std::array<std::byte, 32> h{};  // zero-padded to 32 bytes
    std::memcpy(h.data(), tag.data(), tag.size());
    return h;
}

std::expected<void, WriteError> write_file_0600(const fs::path& path,
                                                std::span<const std::byte> data) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return std::unexpected(WriteError::open_failed);
    std::size_t off = 0;
    while (off < data.size()) {
        const ssize_t n = ::write(fd, data.data() + off, data.size() - off);
        if (n <= 0) {
            ::close(fd);
            return std::unexpected(WriteError::write_failed);
        }
        off += static_cast<std::size_t>(n);
    }
    ::fsync(fd);
    ::close(fd);
    return {};
}

}  // namespace

std::expected<fs::path, WriteError>
write_tor_keys(const VerifiedResult& result, const fs::path& outdir) {
    const std::string addr56{result.address.view()};
    const fs::path dir = outdir / addr56;

    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return std::unexpected(WriteError::create_dir_failed);
    fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace, ec);
    if (ec) return std::unexpected(WriteError::create_dir_failed);

    // hostname
    const std::string hostname = result.address.to_string() + "\n";
    if (auto r = write_file_0600(
            dir / "hostname",
            {reinterpret_cast<const std::byte*>(hostname.data()), hostname.size()});
        !r)
        return std::unexpected(r.error());

    // hs_ed25519_secret_key: header || scalar || prf_prefix
    std::array<std::byte, 96> secret_file;
    const auto sec_header = make_header(kSecretTag);
    std::ranges::copy(sec_header, secret_file.begin());
    std::ranges::copy(result.secret.scalar, secret_file.begin() + 32);
    std::ranges::copy(result.secret.prf_prefix, secret_file.begin() + 64);
    const auto sec_result = write_file_0600(dir / "hs_ed25519_secret_key", secret_file);
    sodium_memzero(secret_file.data(), secret_file.size());
    if (!sec_result) return std::unexpected(sec_result.error());

    // hs_ed25519_public_key: header || pubkey
    std::array<std::byte, 64> public_file;
    const auto pub_header = make_header(kPublicTag);
    std::ranges::copy(pub_header, public_file.begin());
    std::ranges::copy(result.pubkey, public_file.begin() + 32);
    if (auto r = write_file_0600(dir / "hs_ed25519_public_key", public_file); !r)
        return std::unexpected(r.error());

    return dir;
}

}  // namespace onion::io
```

`src/io/CMakeLists.txt` (full new content):

```cmake
add_library(onion_io STATIC
  verifier.cpp
  tor_key_writer.cpp
)
target_link_libraries(onion_io
  PUBLIC onion_core onion_engine
  PRIVATE onion_options)
```

`tests/CMakeLists.txt`: add `test_tor_key_writer.cpp` to the sources list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build --preset debug && ctest --preset debug`
Expected: PASS (23 tests).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(io): Tor HiddenServiceDir key file writer with strict permissions

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: Python verification oracle

Independent re-validation in a different language with different libraries (pure-Python ed25519 + hashlib SHA3). The derivation algorithm below was validated against RFC 8032 TEST1 before this plan was written.

**Files:**
- Create: `tools/verify_onion.py`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the oracle — `tools/verify_onion.py`**

```python
#!/usr/bin/env python3
"""Independent oracle for cpp_onion results.

Re-derives the public key from the secret scalar using a from-scratch
pure-Python ed25519 (no shared code with the C++ implementation or
libsodium), recomputes the onion address, and checks it against the
hostname file. Exit code 0 = OK, 1 = failure.

Usage:
  verify_onion.py <result-dir>     verify a generated key directory
  verify_onion.py --self-test      run internal KATs (RFC 8032 TEST1)
"""

import base64
import hashlib
import sys
from pathlib import Path

P = 2**255 - 19


def inv(x: int) -> int:
    return pow(x, P - 2, P)


D = -121665 * inv(121666) % P
BY = 4 * inv(5) % P


def xrecover(y: int) -> int:
    xx = (y * y - 1) * inv(D * y * y + 1) % P
    x = pow(xx, (P + 3) // 8, P)
    if (x * x - xx) % P != 0:
        x = x * pow(2, (P - 1) // 4, P) % P
    if x % 2 != 0:
        x = P - x
    return x


BX = xrecover(BY)
B = (BX, BY, 1, BX * BY % P)  # extended coordinates (X, Y, Z, T)
IDENTITY = (0, 1, 1, 0)


def pt_add(p, q):
    x1, y1, z1, t1 = p
    x2, y2, z2, t2 = q
    a = (y1 - x1) * (y2 - x2) % P
    b = (y1 + x1) * (y2 + x2) % P
    c = 2 * t1 * t2 * D % P
    d = 2 * z1 * z2 % P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % P, g * h % P, f * g % P, e * h % P)


def scalarmult(p, e: int):
    q = IDENTITY
    while e > 0:
        if e & 1:
            q = pt_add(q, p)
        p = pt_add(p, p)
        e >>= 1
    return q


def encodepoint(p) -> bytes:
    x, y, z, _ = p
    zi = inv(z)
    x, y = x * zi % P, y * zi % P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def onion_address(pub: bytes) -> str:
    chk = hashlib.sha3_256(b".onion checksum" + pub + b"\x03").digest()[:2]
    return base64.b32encode(pub + chk + b"\x03").decode().lower() + ".onion"


SECRET_TAG = b"== ed25519v1-secret: type0 ==" + b"\x00" * 3
PUBLIC_TAG = b"== ed25519v1-public: type0 ==" + b"\x00" * 3


def fail(msg: str):
    print(f"FAIL: {msg}")
    sys.exit(1)


def check_dir(result_dir: Path):
    secret = (result_dir / "hs_ed25519_secret_key").read_bytes()
    public = (result_dir / "hs_ed25519_public_key").read_bytes()
    hostname = (result_dir / "hostname").read_text().strip()

    if len(secret) != 96 or secret[:32] != SECRET_TAG:
        fail("malformed hs_ed25519_secret_key")
    if len(public) != 64 or public[:32] != PUBLIC_TAG:
        fail("malformed hs_ed25519_public_key")

    a = int.from_bytes(secret[32:64], "little")
    if a & 7:
        fail("scalar low 3 bits not clear (clamping violated)")

    derived_pub = encodepoint(scalarmult(B, a))
    if derived_pub != public[32:]:
        fail("public key file does not equal scalar*B")
    if onion_address(derived_pub) != hostname:
        fail("hostname does not match address derived from key")
    print(f"OK: {hostname}")


def self_test():
    seed = bytes.fromhex(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    pub = encodepoint(scalarmult(B, a))
    assert pub.hex() == ("d75a980182b10ab7d54bfed3c964073a"
                         "0ee172f3daa62325af021a68f707511a"), "RFC8032 TEST1 pubkey"
    assert onion_address(pub) == (
        "25njqamcweflpvkl73j4szahhihoc4xt3ktcgjnpaingr5yhkenl5sid.onion"
    ), "TEST1 onion address"
    print("self-test OK")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "--self-test":
        self_test()
    else:
        check_dir(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
```

Then: `chmod +x tools/verify_onion.py`

- [ ] **Step 2: Register the self-test with ctest**

Append to `tests/CMakeLists.txt`:

```cmake
add_test(NAME oracle_self_test
         COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/verify_onion.py --self-test)
```

- [ ] **Step 3: Run to verify it passes**

Run: `python3 tools/verify_onion.py --self-test && cmake --build --preset debug && ctest --preset debug`
Expected: `self-test OK`; ctest PASS (24 tests).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(tools): pure-Python independent verification oracle

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: CLI and end-to-end test

**Files:**
- Create: `src/cli/main.cpp`, `src/cli/CMakeLists.txt`, `tests/e2e/run_e2e.sh`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`

- [ ] **Step 1: Write the CLI — `src/cli/main.cpp`**

```cpp
#include "core/matcher.hpp"
#include "engine/cpu/naive_engine.hpp"
#include "io/tor_key_writer.hpp"
#include "io/verifier.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <print>
#include <stop_token>
#include <thread>
#include <vector>

namespace {
volatile std::sig_atomic_t g_interrupted = 0;
extern "C" void on_sigint(int) { g_interrupted = 1; }
}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"cpp_onion - Tor v3 vanity onion address generator (Phase 0 reference engine)"};

    std::vector<std::string> prefixes;
    std::filesystem::path outdir = ".";
    unsigned threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    std::size_t count = 1;
    bool quiet = false;

    app.add_option("prefix", prefixes, "base32 prefix(es) to search for (a-z, 2-7)")
        ->required();
    app.add_option("-o,--out", outdir, "output directory");
    app.add_option("-t,--threads", threads, "worker threads");
    app.add_option("-n,--count", count, "number of keys to find before exiting");
    app.add_flag("-q,--quiet", quiet, "suppress progress output");
    CLI11_PARSE(app, argc, argv);
    threads = std::max(1u, threads);

    std::vector<onion::core::CompiledPattern> patterns;
    std::size_t shortest = std::numeric_limits<std::size_t>::max();
    for (const auto& p : prefixes) {
        auto compiled = onion::core::compile_prefix(p);
        if (!compiled) {
            std::println(stderr, "error: invalid prefix '{}' (allowed: a-z 2-7, length 1-49)", p);
            return 1;
        }
        shortest = std::min(shortest, p.size());
        patterns.push_back(std::move(*compiled));
    }
    const double expected_tries = std::pow(32.0, double(shortest));
    if (!quiet)
        std::println("searching {} pattern(s), {} threads, ~{:.3g} expected candidates per match",
                     patterns.size(), threads, expected_tries);

    onion::engine::ResultQueue queue;
    onion::engine::StatsBoard stats(threads);
    onion::engine::NaiveCpuEngine engine(patterns, threads, queue, stats);

    std::signal(SIGINT, on_sigint);
    std::stop_source stop;
    std::jthread runner([&] { engine.run(stop.get_token()); });

    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    std::size_t found = 0;
    int exit_code = 0;

    while (found < count && !g_interrupted) {
        auto candidate = queue.pop_wait_for(std::chrono::milliseconds(500));

        if (candidate) {
            const auto verified = onion::io::verify(*candidate, patterns);
            if (!verified) {
                std::println(stderr,
                             "FATAL: verification failed (engine bug, error {})",
                             int(verified.error()));
                exit_code = 2;
                break;
            }
            const auto dir = onion::io::write_tor_keys(*verified, outdir);
            if (!dir) {
                std::println(stderr, "error: failed to write keys (error {})",
                             int(dir.error()));
                exit_code = 3;
                break;
            }
            std::println("found: {}  ->  {}", verified->address.to_string(),
                         dir->string());
            ++found;
        }

        if (!quiet) {
            const auto elapsed = std::chrono::duration<double>(clock::now() - start).count();
            const auto total = stats.total();
            std::print("\r{:.0f} keys/s | {} checked | {:.1f}s elapsed   ",
                       elapsed > 0 ? double(total) / elapsed : 0.0, total, elapsed);
            std::fflush(stdout);
        }
    }

    if (!quiet) std::println("");
    stop.request_stop();
    return exit_code;
}
```

- [ ] **Step 2: Create `src/cli/CMakeLists.txt` and register**

`src/cli/CMakeLists.txt`:

```cmake
add_executable(onion main.cpp)
target_link_libraries(onion PRIVATE
  onion_core onion_engine_cpu onion_io CLI11::CLI11 onion_options Threads::Threads)
```

`src/CMakeLists.txt` (full new content):

```cmake
# Library subdirectories are added here as they are created.
add_subdirectory(crypto)
add_subdirectory(core)
add_subdirectory(engine)
add_subdirectory(io)
add_subdirectory(cli)
```

- [ ] **Step 3: Build and smoke-test manually**

Run: `cmake --build --preset debug && ./build/debug/src/cli/onion a -n 1 -t 4 -o /tmp/onion_smoke`
Expected: finds a key within seconds; prints `found: a....onion -> /tmp/onion_smoke/a...`; clean exit. Then `rm -rf /tmp/onion_smoke`.

- [ ] **Step 4: Write the e2e test — `tests/e2e/run_e2e.sh`**

```bash
#!/usr/bin/env bash
# End-to-end: search a 1-char prefix, then verify the output with the
# independent Python oracle.
set -euo pipefail

BIN="$1"
ORACLE="$2"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# 2-char prefix: ~1024 expected tries — the Phase 0 exit gate (design doc §22)
"$BIN" ab --count 1 --threads 2 --out "$OUT" --quiet

DIR="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d | head -n1)"
test -n "$DIR" || { echo "FAIL: no result directory produced"; exit 1; }

case "$(basename "$DIR")" in
  ab*) ;;
  *) echo "FAIL: result does not start with requested prefix"; exit 1 ;;
esac

python3 "$ORACLE" "$DIR"
```

Then: `chmod +x tests/e2e/run_e2e.sh`

Append to `tests/CMakeLists.txt`:

```cmake
add_test(NAME e2e_search_and_oracle
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/e2e/run_e2e.sh
                 $<TARGET_FILE:onion>
                 ${CMAKE_SOURCE_DIR}/tools/verify_onion.py)
set_tests_properties(e2e_search_and_oracle PROPERTIES TIMEOUT 120)
```

- [ ] **Step 5: Run the full suite**

Run: `cmake --build --preset debug && ctest --preset debug`
Expected: PASS (25 tests) including `e2e_search_and_oracle` printing `OK: ab...onion`.

- [ ] **Step 6: Run the sanitizer preset as a gate**

Run: `cmake --preset asan && cmake --build --preset asan && ctest --preset asan`
Expected: all tests PASS with no ASan/UBSan reports.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(cli): onion CLI with progress stats and oracle-backed e2e test

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 13: README and CI workflow

**Files:**
- Create: `README.md`, `.github/workflows/ci.yml`

- [ ] **Step 1: Write `README.md`**

```markdown
# cpp_onion

A high-performance Tor v3 vanity onion address generator in modern C++23.

**Status: Phase 0** — correct, slow reference engine (libsodium per-candidate
derivation). Phase 1 replaces it with an incremental ed25519 search
(~1000x faster); see `docs/design.md` for the full engineering design.

## Build

Requires: GCC 14+ (project targets GCC 16), CMake 3.28+, libsodium, Python 3.

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
```

- [ ] **Step 2: Write `.github/workflows/ci.yml`**

```yaml
name: ci
on: [push, pull_request]

jobs:
  build-test:
    runs-on: ubuntu-24.04
    strategy:
      fail-fast: false
      matrix:
        include:
          - { cxx: g++-14, packages: g++-14 }
          - { cxx: clang++-18, packages: clang-18 }
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: sudo apt-get update && sudo apt-get install -y ${{ matrix.packages }} libsodium-dev ninja-build
      - name: Configure
        run: >
          cmake -S . -B build -G Ninja
          -DCMAKE_BUILD_TYPE=Debug
          -DCMAKE_CXX_COMPILER=${{ matrix.cxx }}
          -DONION_NATIVE=OFF
          -DONION_SANITIZE=ON
      - name: Build
        run: cmake --build build
      - name: Test
        run: ctest --test-dir build --output-on-failure
```

(GCC 16 is the local toolchain; CI uses g++-14, the newest GCC on `ubuntu-24.04` runners, which covers every C++23 feature Phase 0 uses. clang is the design-mandated second compiler — the canary for GCC-16-only code (design doc §21.3). Revisit versions when runners ship newer toolchains.)

- [ ] **Step 3: Verify the full local matrix one last time**

Run: `ctest --preset debug && ctest --preset asan`
Expected: all PASS in both presets.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: README and CI workflow

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Definition of Done (Phase 0 exit gate, design doc §22)

- [ ] `ctest` green in `debug` and `asan` presets (25 tests: KATs, property tests, unit, e2e).
- [ ] e2e finds a short prefix and the Python oracle validates the written files.
- [ ] Manual acceptance (optional, requires Tor installed): point a Tor `HiddenServiceDir` at a generated directory, confirm Tor serves the expected address.

## What Phase 1 builds on top of this (separate plan, written after Phase 0 lands)

- `crypto::fe` / `crypto::ge`: 5×51-limb field arithmetic, extended-coordinate point ops, `+8B` incremental chains, Montgomery batch inversion — tested against Task 5's libsodium path (`incremental_chain(a0)[i] == pubkey_from_scalar(a0 + 8i)`).
- `IncrementalCpuEngine` replacing `NaiveCpuEngine` behind `IEngine` — every other component (matcher, verifier, writer, oracle, CLI) is reused unchanged.
- Scalar reconstruction `a0 + 8i` with the bit-255 overflow guard; fresh random `prf_prefix` at output time (design doc §9).
- `onion bench` subcommand + nanobench microbenchmarks.
