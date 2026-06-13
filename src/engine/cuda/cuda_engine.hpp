#pragma once
//
// Plain-C++ boundary for the CUDA search engine (design §6 toolchain firewall).
// Includes NO CUDA headers, so the GCC-16 CLI translation unit can include it.
// All cudaMalloc/launch/sync calls live in cuda_engine.cu behind an opaque
// device-state pimpl.

#include "core/matcher.hpp"
#include "engine/engine.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stop_token>
#include <vector>

namespace onion::cuda {

// Tunable launch knobs (all have defaults sized for a 14-SM Turing GTX 1650).
struct CudaKnobs {
    // T = number of device threads (= interleaved chains). 2^16 saturates the
    // GPU; each thread walks kStepsPerThread (M=32) points per launch.
    int threads = 1 << 16;
    // CUDA block size (threads per block). 128 gives good occupancy on sm_75.
    int block = 128;
};

// GPU vanity-search engine. Same IEngine seam as the CPU engines: run() blocks
// until the stop_token fires. Candidates are pushed to the ResultQueue; the CLI
// io::verify firewall is the final gate (a non-verifying hit never reaches
// disk and must not crash this loop).
class CudaEngine final : public onion::engine::IEngine {
public:
    CudaEngine(std::vector<onion::core::CompiledPattern> patterns,
               onion::engine::ResultQueue& results,
               onion::engine::StatsBoard& stats,
               CudaKnobs knobs = {});
    ~CudaEngine() override;

    CudaEngine(const CudaEngine&) = delete;
    CudaEngine& operator=(const CudaEngine&) = delete;

    void run(std::stop_token stop) override;

private:
    struct Impl;                 // owns all device state (cudaMalloc'd buffers).
    std::unique_ptr<Impl> impl_;
};

}  // namespace onion::cuda
