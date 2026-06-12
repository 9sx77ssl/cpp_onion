#pragma once

#include "core/matcher.hpp"
#include "engine/engine.hpp"

#include <cstddef>
#include <stop_token>
#include <vector>

namespace onion::engine {

// Phase 1 engine: per-worker incremental ed25519 search (A += 8B + batched
// inversion). ~100-250x the naive engine. Same IEngine seam.
class IncrementalCpuEngine final : public IEngine {
public:
    IncrementalCpuEngine(std::vector<core::CompiledPattern> patterns, unsigned num_threads,
                         ResultQueue& results, StatsBoard& stats, std::size_t batch = 1024,
                         std::uint64_t epoch_candidates = 1u << 22)
        : patterns_(std::move(patterns)), num_threads_(num_threads), results_(results),
          stats_(stats), batch_(batch), epoch_candidates_(epoch_candidates) {}

    void run(std::stop_token stop) override;

private:
    void worker(std::stop_token stop, unsigned index);

    std::vector<core::CompiledPattern> patterns_;
    unsigned num_threads_;
    ResultQueue& results_;
    StatsBoard& stats_;
    std::size_t batch_;
    std::uint64_t epoch_candidates_;
};

}  // namespace onion::engine
