#pragma once

#include "core/matcher.hpp"
#include "engine/engine.hpp"

#include <cstddef>
#include <stop_token>
#include <vector>

namespace onion::engine {

// AVX2 4-wide incremental engine. Same IEngine seam as IncrementalCpuEngine.
// Each worker epoch drives an IncrementalStepperX4 with 4 independent clamped
// base scalars. Each batch yields 4 lanes × batch_ candidates. Throughput
// accounting: local_total += 4 * n per batch (honest 4x numerator vs scalar).
class IncrementalCpuEngineX4 final : public IEngine {
public:
    IncrementalCpuEngineX4(std::vector<core::CompiledPattern> patterns, unsigned num_threads,
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
