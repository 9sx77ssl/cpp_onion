#include <catch2/catch_test_macros.hpp>

#include "crypto/fe25519.hpp"
#include "crypto/fe25519x4.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string_view>

namespace {

using onion::crypto::Fe;
using onion::crypto::Fe4;

using Bytes = std::array<std::byte, 32>;

// Reduce a scalar Fe to its canonical 32-byte little-endian encoding.
Bytes scalar_to_bytes(const Fe& f) {
    Bytes o;
    onion::crypto::fe_to_bytes(o, f);
    return o;
}

// A random < 2^255 field element (top bit of byte 31 cleared, matching the
// generator that feeds the real engine).
Bytes rand_fe(std::mt19937_64& rng) {
    Bytes b;
    for (auto& x : b) x = std::byte(rng() & 0xff);
    b[31] = std::byte(std::to_integer<unsigned>(b[31]) & 0x7f);
    return b;
}

// Pull the four lane encodings out of a Fe4.
std::array<Bytes, 4> store4(const Fe4& f) {
    std::array<Bytes, 4> o;
    onion::crypto::fe4_store_y(f, o[0], o[1], o[2], o[3]);
    return o;
}

}  // namespace

TEST_CASE("fe4 load/store round-trips against scalar from/to bytes") {
    std::mt19937_64 rng(0xC0FFEE);
    for (int trial = 0; trial < 1000; ++trial) {
        Bytes a = rand_fe(rng), b = rand_fe(rng), c = rand_fe(rng), d = rand_fe(rng);
        Fe4 f = onion::crypto::fe4_load(a, b, c, d);
        auto got = store4(f);
        // The scalar canonical form of the same inputs (already < 2^255, so the
        // canonicalisation is the identity on the represented value).
        CHECK(got[0] == scalar_to_bytes(onion::crypto::fe_from_bytes(a)));
        CHECK(got[1] == scalar_to_bytes(onion::crypto::fe_from_bytes(b)));
        CHECK(got[2] == scalar_to_bytes(onion::crypto::fe_from_bytes(c)));
        CHECK(got[3] == scalar_to_bytes(onion::crypto::fe_from_bytes(d)));
    }
}

TEST_CASE("fe4_mul cross-validates against scalar fe_mul") {
    using namespace onion::crypto;
    std::mt19937_64 rng(1);
    for (int trial = 0; trial < 1000; ++trial) {
        Bytes a0 = rand_fe(rng), a1 = rand_fe(rng), a2 = rand_fe(rng), a3 = rand_fe(rng);
        Bytes b0 = rand_fe(rng), b1 = rand_fe(rng), b2 = rand_fe(rng), b3 = rand_fe(rng);
        Fe4 va = fe4_load(a0, a1, a2, a3);
        Fe4 vb = fe4_load(b0, b1, b2, b3);
        auto got = store4(fe4_mul(va, vb));
        CHECK(got[0] == scalar_to_bytes(fe_mul(fe_from_bytes(a0), fe_from_bytes(b0))));
        CHECK(got[1] == scalar_to_bytes(fe_mul(fe_from_bytes(a1), fe_from_bytes(b1))));
        CHECK(got[2] == scalar_to_bytes(fe_mul(fe_from_bytes(a2), fe_from_bytes(b2))));
        CHECK(got[3] == scalar_to_bytes(fe_mul(fe_from_bytes(a3), fe_from_bytes(b3))));
    }
}

TEST_CASE("fe4_sq cross-validates against scalar fe_sq") {
    using namespace onion::crypto;
    std::mt19937_64 rng(2);
    for (int trial = 0; trial < 1000; ++trial) {
        Bytes a0 = rand_fe(rng), a1 = rand_fe(rng), a2 = rand_fe(rng), a3 = rand_fe(rng);
        Fe4 va = fe4_load(a0, a1, a2, a3);
        auto got = store4(fe4_sq(va));
        CHECK(got[0] == scalar_to_bytes(fe_sq(fe_from_bytes(a0))));
        CHECK(got[1] == scalar_to_bytes(fe_sq(fe_from_bytes(a1))));
        CHECK(got[2] == scalar_to_bytes(fe_sq(fe_from_bytes(a2))));
        CHECK(got[3] == scalar_to_bytes(fe_sq(fe_from_bytes(a3))));
    }
}

TEST_CASE("fe4_add cross-validates against scalar fe_add") {
    using namespace onion::crypto;
    std::mt19937_64 rng(3);
    for (int trial = 0; trial < 1000; ++trial) {
        Bytes a0 = rand_fe(rng), a1 = rand_fe(rng), a2 = rand_fe(rng), a3 = rand_fe(rng);
        Bytes b0 = rand_fe(rng), b1 = rand_fe(rng), b2 = rand_fe(rng), b3 = rand_fe(rng);
        Fe4 va = fe4_load(a0, a1, a2, a3);
        Fe4 vb = fe4_load(b0, b1, b2, b3);
        auto got = store4(fe4_add(va, vb));
        CHECK(got[0] == scalar_to_bytes(fe_add(fe_from_bytes(a0), fe_from_bytes(b0))));
        CHECK(got[1] == scalar_to_bytes(fe_add(fe_from_bytes(a1), fe_from_bytes(b1))));
        CHECK(got[2] == scalar_to_bytes(fe_add(fe_from_bytes(a2), fe_from_bytes(b2))));
        CHECK(got[3] == scalar_to_bytes(fe_add(fe_from_bytes(a3), fe_from_bytes(b3))));
    }
}

TEST_CASE("fe4_sub cross-validates against scalar fe_sub") {
    using namespace onion::crypto;
    std::mt19937_64 rng(4);
    for (int trial = 0; trial < 1000; ++trial) {
        Bytes a0 = rand_fe(rng), a1 = rand_fe(rng), a2 = rand_fe(rng), a3 = rand_fe(rng);
        Bytes b0 = rand_fe(rng), b1 = rand_fe(rng), b2 = rand_fe(rng), b3 = rand_fe(rng);
        Fe4 va = fe4_load(a0, a1, a2, a3);
        Fe4 vb = fe4_load(b0, b1, b2, b3);
        auto got = store4(fe4_sub(va, vb));
        CHECK(got[0] == scalar_to_bytes(fe_sub(fe_from_bytes(a0), fe_from_bytes(b0))));
        CHECK(got[1] == scalar_to_bytes(fe_sub(fe_from_bytes(a1), fe_from_bytes(b1))));
        CHECK(got[2] == scalar_to_bytes(fe_sub(fe_from_bytes(a2), fe_from_bytes(b2))));
        CHECK(got[3] == scalar_to_bytes(fe_sub(fe_from_bytes(a3), fe_from_bytes(b3))));
    }
}

TEST_CASE("fe4_sub returns reduced limbs that fe4_mul can consume (bias-zero overflow)") {
    using namespace onion::crypto;
    std::mt19937_64 rng(5);
    for (int trial = 0; trial < 1000; ++trial) {
        // Force the worst case where a==b so the difference is a bias-form zero,
        // then immediately feed it through fe4_mul and compare to scalar.
        Bytes a0 = rand_fe(rng), a1 = rand_fe(rng), a2 = rand_fe(rng), a3 = rand_fe(rng);
        Bytes m0 = rand_fe(rng), m1 = rand_fe(rng), m2 = rand_fe(rng), m3 = rand_fe(rng);
        Fe4 va = fe4_load(a0, a1, a2, a3);
        Fe4 vm = fe4_load(m0, m1, m2, m3);
        Fe4 diff = fe4_sub(va, va);  // bias-form zero in every lane
        auto got = store4(fe4_mul(diff, vm));

        auto sref = [&](const Bytes& a, const Bytes& m) {
            Fe fa = fe_from_bytes(a), fm = fe_from_bytes(m);
            return scalar_to_bytes(fe_mul(fe_sub(fa, fa), fm));
        };
        CHECK(got[0] == sref(a0, m0));
        CHECK(got[1] == sref(a1, m1));
        CHECK(got[2] == sref(a2, m2));
        CHECK(got[3] == sref(a3, m3));
    }
}

TEST_CASE("fe4_invert cross-validates against scalar fe_invert") {
    using namespace onion::crypto;
    std::mt19937_64 rng(6);
    for (int trial = 0; trial < 1000; ++trial) {
        Bytes a0 = rand_fe(rng), a1 = rand_fe(rng), a2 = rand_fe(rng), a3 = rand_fe(rng);
        Fe4 va = fe4_load(a0, a1, a2, a3);
        auto got = store4(fe4_invert(va));
        CHECK(got[0] == scalar_to_bytes(fe_invert(fe_from_bytes(a0))));
        CHECK(got[1] == scalar_to_bytes(fe_invert(fe_from_bytes(a1))));
        CHECK(got[2] == scalar_to_bytes(fe_invert(fe_from_bytes(a2))));
        CHECK(got[3] == scalar_to_bytes(fe_invert(fe_from_bytes(a3))));
    }
}
