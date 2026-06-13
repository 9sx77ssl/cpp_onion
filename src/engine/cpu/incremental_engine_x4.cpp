#include "engine/cpu/incremental_engine_x4.hpp"

#include "crypto/incremental.hpp"
#include "crypto/incremental_x4.hpp"
#include "crypto/keys.hpp"

#include <sodium.h>

#include <array>
#include <thread>
#include <vector>

namespace onion::engine {

void IncrementalCpuEngineX4::run(std::stop_token stop) {
    std::vector<std::jthread> workers;
    workers.reserve(num_threads_);
    for (unsigned i = 0; i < num_threads_; ++i)
        workers.emplace_back([this, stop, i] { worker(stop, i); });
}

void IncrementalCpuEngineX4::worker(std::stop_token stop, unsigned index) {
    // Per-batch output buffers: 4 lanes, each sized to batch_.
    // Sized once, reused across batches and epochs (design §10: no hot-path alloc).
    std::array<std::vector<std::array<std::byte, 32>>, 4> ybatch;
    for (auto& lane : ybatch)
        lane.resize(batch_);

    std::uint64_t local_total = 0;

    while (!stop.stop_requested()) {
        // New epoch: 4 independent fresh clamped base scalars, one per lane.
        std::array<std::array<std::byte, 32>, 4> a0;
        for (auto& scalar : a0) {
            crypto::random_bytes(scalar);
            scalar[0]  = std::byte(std::to_integer<unsigned>(scalar[0])  & 0xf8);
            scalar[31] = std::byte((std::to_integer<unsigned>(scalar[31]) & 0x7f) | 0x40);
        }

        crypto::IncrementalStepperX4 stepper(a0[0], a0[1], a0[2], a0[3]);
        const std::uint64_t batches = epoch_candidates_ / batch_;

        for (std::uint64_t b = 0; b < batches && !stop.stop_requested(); ++b) {
            const std::uint64_t base_index = stepper.consumed();
            stepper.next_batch(ybatch, batch_);

            for (std::size_t lane = 0; lane < 4; ++lane) {
                for (std::size_t k = 0; k < batch_; ++k) {
                    std::span<const std::byte, 32> y{ybatch[lane][k]};
                    for (std::size_t p = 0; p < patterns_.size(); ++p) {
                        if (core::matches(patterns_[p], y)) {
                            auto scalar = crypto::scalar_add_8i(a0[lane], base_index + k);
                            crypto::ExpandedSecretKey secret;
                            secret.scalar = scalar;
                            crypto::random_bytes(secret.prf_prefix);  // fresh RH (design §9)
                            auto pk = crypto::pubkey_from_scalar(secret.scalar);
                            if (pk) results_.push({secret, *pk, p});
                        }
                    }
                }
            }
            // 4 lanes × batch_ candidates advanced this batch.
            local_total += 4 * batch_;
            stats_.set(index, local_total);
        }

        for (auto& scalar : a0)
            sodium_memzero(scalar.data(), scalar.size());
    }
}

}  // namespace onion::engine
