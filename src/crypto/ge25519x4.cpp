#include "crypto/ge25519x4.hpp"

#include "crypto/fe25519.hpp"

#include <array>
#include <cassert>
#include <cstddef>

namespace onion::crypto {

// ── helpers: move a scalar Fe through bytes into one lane of an Fe4 ──────────

namespace {

// Encode a scalar Fe to 32 bytes (wrapper to keep this file self-contained).
static std::array<std::byte, 32> fe_encode(const Fe& f) {
    std::array<std::byte, 32> b;
    fe_to_bytes(b, f);
    return b;
}

// Build an Fe4 where all 4 lanes hold the same scalar Fe.
static Fe4 fe4_broadcast(const Fe& f) {
    std::array<std::byte, 32> b = fe_encode(f);
    return fe4_load(b, b, b, b);
}

// Extract lane `lane` of an Fe4 into a scalar Fe (goes through bytes).
static Fe fe4_extract_lane(const Fe4& f, int lane) {
    std::array<std::byte, 32> out[4];
    fe4_store_y(f, out[0], out[1], out[2], out[3]);
    return fe_from_bytes(out[lane]);
}

}  // namespace

// ── pack / unpack ─────────────────────────────────────────────────────────────

GeP3x4 ge_p3x4_pack(const GeP3& l0, const GeP3& l1, const GeP3& l2, const GeP3& l3) {
    // Encode each coordinate of each point to bytes, then load into Fe4 SoA.
    auto load4 = [](const Fe& a, const Fe& b, const Fe& c, const Fe& d) -> Fe4 {
        return fe4_load(fe_encode(a), fe_encode(b), fe_encode(c), fe_encode(d));
    };
    GeP3x4 out;
    out.X = load4(l0.X, l1.X, l2.X, l3.X);
    out.Y = load4(l0.Y, l1.Y, l2.Y, l3.Y);
    out.Z = load4(l0.Z, l1.Z, l2.Z, l3.Z);
    out.T = load4(l0.T, l1.T, l2.T, l3.T);
    return out;
}

GeCachedAffinex4 ge_cached_affinex4_broadcast(const GeCachedAffine& q) {
    GeCachedAffinex4 out;
    out.YplusX  = fe4_broadcast(q.YplusX);
    out.YminusX = fe4_broadcast(q.YminusX);
    out.T2d     = fe4_broadcast(q.T2d);
    return out;
}

GeP3 ge_p3x4_unpack(const GeP3x4& p, int lane) {
    assert(lane >= 0 && lane < 4);
    GeP3 out;
    out.X = fe4_extract_lane(p.X, lane);
    out.Y = fe4_extract_lane(p.Y, lane);
    out.Z = fe4_extract_lane(p.Z, lane);
    out.T = fe4_extract_lane(p.T, lane);
    return out;
}

// ── ge_madd_x4: mixed addition, identical formula to scalar ge_madd ───────────
//
//   a = (p.Y - p.X) * q.YminusX
//   b = (p.Y + p.X) * q.YplusX
//   c = q.T2d * p.T
//   d = 2 * p.Z                       (q.Z == 1)
//   e = b - a
//   f = d - c
//   g = d + c
//   h = b + a
//   result = (e*f, g*h, f*g, e*h)

GeP3x4 ge_madd_x4(const GeP3x4& p, const GeCachedAffinex4& q) {
    Fe4 a = fe4_mul(fe4_sub(p.Y, p.X), q.YminusX);
    Fe4 b = fe4_mul(fe4_add(p.Y, p.X), q.YplusX);
    Fe4 c = fe4_mul(q.T2d, p.T);
    Fe4 d = fe4_add(p.Z, p.Z);  // 2 * p.Z  (q.Z == 1)
    Fe4 e = fe4_sub(b, a);
    Fe4 f = fe4_sub(d, c);
    Fe4 g = fe4_add(d, c);
    Fe4 h = fe4_add(b, a);
    return {fe4_mul(e, f), fe4_mul(g, h), fe4_mul(f, g), fe4_mul(e, h)};
}

}  // namespace onion::crypto
