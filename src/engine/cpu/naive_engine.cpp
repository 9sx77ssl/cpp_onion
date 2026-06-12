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
