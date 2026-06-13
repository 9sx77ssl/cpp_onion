#include "crypto/incremental_x4.hpp"

#include "crypto/ge25519.hpp"

namespace onion::crypto {

IncrementalStepperX4::IncrementalStepperX4(std::span<const std::byte, 32> a0_0,
                                            std::span<const std::byte, 32> a0_1,
                                            std::span<const std::byte, 32> a0_2,
                                            std::span<const std::byte, 32> a0_3)
{
    // Compute the starting points for each lane (scalar, setup only).
    GeP3 p0 = ge_scalarmult_base(a0_0);
    GeP3 p1 = ge_scalarmult_base(a0_1);
    GeP3 p2 = ge_scalarmult_base(a0_2);
    GeP3 p3 = ge_scalarmult_base(a0_3);
    cur_ = ge_p3x4_pack(p0, p1, p2, p3);

    // 8B is the same constant in every lane.
    std::array<std::byte, 32> eight{};
    eight[0] = std::byte{8};
    step8b_ = ge_cached_affinex4_broadcast(ge_to_cached_affine(ge_scalarmult_base(eight)));
}

void IncrementalStepperX4::next_batch(
    std::array<std::vector<std::array<std::byte, 32>>, 4>& out,
    std::size_t n)
{
    // Grow scratch once; reuse across batches (design §10: no hot-path alloc).
    if (ys_.size() < n) {
        ys_.resize(n);
        zs_.resize(n);
        prefix_.resize(n);
    }

    // Hot loop: collect Y and Z for all n steps, then advance cur_.
    for (std::size_t i = 0; i < n; ++i) {
        ys_[i] = cur_.Y;
        zs_[i] = cur_.Z;
        cur_ = ge_madd_x4(cur_, step8b_);
    }

    // Montgomery batch inversion of the Z values (per-lane in lockstep via Fe4).
    //
    // prefix_[i] = zs_[0] * zs_[1] * ... * zs_[i]   (all 4 lanes simultaneously)
    prefix_[0] = zs_[0];
    for (std::size_t i = 1; i < n; ++i)
        prefix_[i] = fe4_mul(prefix_[i - 1], zs_[i]);

    // inv = 1 / (zs_[0] * ... * zs_[n-1])
    Fe4 inv = fe4_invert(prefix_[n - 1]);

    // Back-substitute: for each step (in reverse), recover y = Y * Z^-1 for all 4 lanes.
    for (std::size_t i = n; i-- > 0;) {
        Fe4 zinv = (i == 0) ? inv : fe4_mul(inv, prefix_[i - 1]);
        Fe4 y    = fe4_mul(ys_[i], zinv);

        // Decode all 4 lanes into the output arrays.
        fe4_store_y(y, out[0][i], out[1][i], out[2][i], out[3][i]);

        if (i != 0) inv = fe4_mul(inv, zs_[i]);
    }

    consumed_ += n;
}

}  // namespace onion::crypto
