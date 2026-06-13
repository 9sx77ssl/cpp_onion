#pragma once
//
// Compile-time selector between the two device field backends:
//   - device_field.cuh    : validated 5x51-bit limbs using emulated __int128
//                           (the trusted baseline / reference path).
//   - device_field32.cuh  : native 8x32-bit limbs (no u128 emulation), the
//                           Turing speedup path.
//
// Define ONION_CUDA_FIELD32 (via the CMake option ONION_CUDA_FIELD32=ON) to use
// the 32-bit field in the search kernel. The xval gate compiles BOTH paths into
// separate harness entry points so the new field is always cross-validated.
//
// Both headers export the identical surface (Fe, GeP3, GeCachedAffine,
// fed_add/sub/mul/sq/invert/from_bytes/to_bytes, fed_kD2, ged_madd), so the
// kernel sources are field-agnostic. Only the host->device Fe bridge differs;
// to_device_fe() below handles both representations.

#ifdef ONION_CUDA_FIELD32
#include "engine/cuda/device_field32.cuh"
#else
#include "engine/cuda/device_field.cuh"
#endif

#include "crypto/fe25519.hpp"
#include "crypto/ge25519.hpp"

#include <array>
#include <cstring>

namespace onion::cuda {

// Bridge a host crypto::Fe (5x51 limbs) into the selected device-layout Fe.
//
// For the 51-bit device field this is a direct limb copy (identical layout).
// For the 32-bit device field we go through the canonical 32-byte encoding:
// crypto::fe_to_bytes() reduces the host element mod p to its canonical little-
// endian bytes, which the device field re-decodes into 8x32 limbs. Coordinates
// X/Y/Z/T are residues mod p, so reducing each individually preserves the
// projective point exactly. This host-side helper is NOT __device__ code.
inline Fe to_device_fe(const onion::crypto::Fe& f) {
    Fe r;
#ifdef ONION_CUDA_FIELD32
    std::array<std::byte, 32> b{};
    onion::crypto::fe_to_bytes(b, f);
    for (int i = 0; i < 8; ++i) {
        const std::uint8_t* s = reinterpret_cast<const std::uint8_t*>(b.data()) + 4 * i;
        r.w[i] = (std::uint32_t)s[0] | ((std::uint32_t)s[1] << 8) |
                 ((std::uint32_t)s[2] << 16) | ((std::uint32_t)s[3] << 24);
    }
    r.w[7] &= 0x7fffffffu;  // canonical y has the sign bit clear; mirror device
#else
    for (int i = 0; i < 5; ++i) r.v[i] = f.v[i];
#endif
    return r;
}

inline GeP3 to_device_p3(const onion::crypto::GeP3& p) {
    return GeP3{to_device_fe(p.X), to_device_fe(p.Y), to_device_fe(p.Z), to_device_fe(p.T)};
}

inline GeCachedAffine to_device_cached(const onion::crypto::GeCachedAffine& q) {
    return GeCachedAffine{to_device_fe(q.YplusX), to_device_fe(q.YminusX),
                          to_device_fe(q.T2d)};
}

}  // namespace onion::cuda
