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

// Tunable launch knobs (defaults are the measured optimum for a 14-SM Turing
// GTX 1650; both can be overridden at run time via the ONION_CUDA_THREADS /
// ONION_CUDA_BLOCK env vars for sweeps, see cuda_engine.cu::init).
struct CudaKnobs {
    // T = number of device threads (= interleaved chains). The kernel is bound
    // by field arithmetic / local memory (~113 regs/thread for the default
    // 32-bit field at M=3072, 178 for the 51-bit reference field; both leave
    // ~2 resident blocks/SM, fixed in T), so throughput is flat for T >= ~14k
    // (enough waves to fill all 14 SMs). 2^14 is the measured knee: identical
    // throughput to 2^15/2^16 but with ~half the per-thread local-memory
    // reservation. Each thread walks kStepsPerThread (M) points per launch.
    int threads = 1 << 14;
    // CUDA block size (threads per block). 128 measured best on sm_75 (>= 256
    // is within noise; 128 keeps 2 blocks resident at the kernel's reg count).
    int block = 128;
    // Which StatsBoard slot this engine writes its cumulative count to. Default
    // 0 (standalone). In the cpu+gpu composite the CPU engine owns slots
    // [0, threads); the GPU is handed slot `threads` so the two never clobber.
    unsigned stats_slot = 0;
    // Block (sleep) the host thread on GPU sync instead of spin-polling a CPU
    // core. Standalone GPU keeps spinning (lowest latency; the CPU is otherwise
    // idle); in the cpu+gpu composite we set this so the freed core runs a CPU
    // worker — the kernel is long (~258 ms/epoch) so the wakeup latency is noise.
    bool blocking_sync = false;
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
