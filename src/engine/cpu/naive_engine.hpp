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
