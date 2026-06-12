#include "engine/cpu/incremental_engine.hpp"

#include "crypto/incremental.hpp"
#include "crypto/keys.hpp"

#include <sodium.h>

#include <array>
#include <span>
#include <thread>
#include <vector>

namespace onion::engine {

void IncrementalCpuEngine::run(std::stop_token stop) {
    std::vector<std::jthread> workers;
    workers.reserve(num_threads_);
    for (unsigned i = 0; i < num_threads_; ++i)
        workers.emplace_back([this, stop, i] { worker(stop, i); });
}

void IncrementalCpuEngine::worker(std::stop_token stop, unsigned index) {
    std::vector<std::array<std::byte, 32>> ybatch(batch_);
    std::uint64_t local_total = 0;

    while (!stop.stop_requested()) {
        // New epoch: fresh clamped base scalar.
        std::array<std::byte, 32> a0;
        crypto::random_bytes(a0);
        a0[0] = std::byte(std::to_integer<unsigned>(a0[0]) & 0xf8);
        a0[31] = std::byte((std::to_integer<unsigned>(a0[31]) & 0x7f) | 0x40);

        crypto::IncrementalStepper stepper(a0);
        const std::uint64_t batches = epoch_candidates_ / batch_;

        for (std::uint64_t b = 0; b < batches && !stop.stop_requested(); ++b) {
            const std::uint64_t base_index = stepper.consumed();
            stepper.next_batch(std::span<std::array<std::byte, 32>>{ybatch});
            for (std::size_t k = 0; k < batch_; ++k) {
                std::span<const std::byte, 32> y{ybatch[k]};
                for (std::size_t p = 0; p < patterns_.size(); ++p) {
                    if (core::matches(patterns_[p], y)) {
                        auto scalar = crypto::scalar_add_8i(a0, base_index + k);
                        crypto::ExpandedSecretKey secret;
                        secret.scalar = scalar;
                        crypto::random_bytes(secret.prf_prefix);  // fresh RH (design §9)
                        auto pk = crypto::pubkey_from_scalar(secret.scalar);
                        if (pk) results_.push({secret, *pk, p});
                    }
                }
            }
            local_total += batch_;
            stats_.set(index, local_total);
        }
        sodium_memzero(a0.data(), a0.size());
    }
}

}  // namespace onion::engine
