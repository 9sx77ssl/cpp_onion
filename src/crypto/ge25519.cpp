#include "crypto/ge25519.hpp"

namespace onion::crypto {
namespace {

// 2d = 2*(-121665/121666) mod p, as 5x51 limbs (computed in Python, all limbs < 2^51).
const Fe kD2 = {{0x69b9426b2f159ULL, 0x35050762add7aULL, 0x3cf44c0038052ULL,
                 0x6738cc7407977ULL, 0x2406d9dc56dffULL}};

}  // namespace

GeP3 ge_basepoint() {
    GeP3 b;
    b.X = Fe{{0x62d608f25d51aULL, 0x412a4b4f6592aULL, 0x75b7171a4b31dULL,
              0x1ff60527118feULL, 0x216936d3cd6e5ULL}};
    b.Y = Fe{{0x6666666666658ULL, 0x4ccccccccccccULL, 0x1999999999999ULL,
              0x3333333333333ULL, 0x6666666666666ULL}};
    b.Z = Fe{{1, 0, 0, 0, 0}};
    b.T = Fe{{0x68ab3a5b7dda3ULL, 0x0eea2a5eadbbULL, 0x2af8df483c27eULL,
              0x332b375274732ULL, 0x67875f0fd78b7ULL}};
    return b;
}

GeCached ge_to_cached(const GeP3& p) {
    return {fe_add(p.Y, p.X), fe_sub(p.Y, p.X), p.Z, fe_mul(p.T, kD2)};
}

// r = p + q  (p3 + cached -> p3), unified/complete Edwards addition.
GeP3 ge_add(const GeP3& p, const GeCached& q) {
    Fe a = fe_mul(fe_sub(p.Y, p.X), q.YminusX);
    Fe b = fe_mul(fe_add(p.Y, p.X), q.YplusX);
    Fe c = fe_mul(q.T2d, p.T);
    Fe d = fe_mul(p.Z, q.Z);
    d = fe_add(d, d);
    Fe e = fe_sub(b, a);
    Fe f = fe_sub(d, c);
    Fe g = fe_add(d, c);
    Fe h = fe_add(b, a);
    return {fe_mul(e, f), fe_mul(g, h), fe_mul(f, g), fe_mul(e, h)};
}

// r = 2p. Extended-coordinate doubling (dbl-2008-hwcd, a=-1). The sign flips
// vs the textbook (E=H-(X+Y)^2 = -E_std, G=A-B = -G_std, etc.) cancel in every
// output product, so this matches the standard doubling. fe_sub now returns
// reduced limbs, so all fe_mul inputs are safely bounded.
// Outputs: X3=E*F, Y3=G*H, Z3=F*G, T3=E*H.
GeP3 ge_double(const GeP3& p) {
    Fe A = fe_sq(p.X);
    Fe B = fe_sq(p.Y);
    Fe C = fe_add(fe_sq(p.Z), fe_sq(p.Z));     // 2*Z^2
    Fe H = fe_add(A, B);                        // A + B
    Fe E = fe_sub(H, fe_sq(fe_add(p.X, p.Y))); // H - (X+Y)^2
    Fe G = fe_sub(A, B);                        // A - B
    Fe F = fe_add(C, G);                        // C + G
    return {fe_mul(E, F), fe_mul(G, H), fe_mul(F, G), fe_mul(E, H)};
}

GeP3 ge_scalarmult_base(std::span<const std::byte, 32> scalar) {
    // Simple double-and-add over the hardcoded basepoint. Setup-only (once per
    // epoch), so constant-time / windowing is unnecessary.
    GeP3 r = {Fe{{0,0,0,0,0}}, Fe{{1,0,0,0,0}}, Fe{{1,0,0,0,0}}, Fe{{0,0,0,0,0}}};  // identity
    GeCached base = ge_to_cached(ge_basepoint());
    // process bits MSB-first; crypto_scalarmult_ed25519_base_noclamp treats the
    // scalar as a 255-bit integer (ignores bit 255 = bit 7 of byte 31).
    for (int byte = 31; byte >= 0; --byte) {
        unsigned b = std::to_integer<unsigned>(scalar[std::size_t(byte)]);
        if (byte == 31) b &= 0x7fu;  // mask out bit 255 to match noclamp semantics
        for (int bit = 7; bit >= 0; --bit) {
            r = ge_double(r);
            if ((b >> bit) & 1) r = ge_add(r, base);
        }
    }
    return r;
}

std::array<std::byte, 32> ge_encode(const GeP3& p) {
    Fe zinv = fe_invert(p.Z);
    Fe x = fe_mul(p.X, zinv);
    Fe y = fe_mul(p.Y, zinv);
    std::array<std::byte, 32> out;
    fe_to_bytes(out, y);
    std::array<std::byte, 32> xb;
    fe_to_bytes(xb, x);
    // set top bit = low bit of x (the sign)
    out[31] = std::byte(std::to_integer<unsigned>(out[31]) |
                        ((std::to_integer<unsigned>(xb[0]) & 1) << 7));
    return out;
}

}  // namespace onion::crypto
