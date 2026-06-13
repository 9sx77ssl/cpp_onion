#pragma once
//
// CUDA-side interface to the vanity-search hot kernel. Included ONLY by .cu
// translation units (it pulls in cuda_runtime via the kernel launch wrappers
// and device_field.cuh layouts). The plain-C++ engine boundary is in
// cuda_engine.hpp; this header is the .cu <-> .cu seam.

#include "engine/cuda/device_field.cuh"

#include <cstdint>

namespace onion::cuda {

// Compile-time steps per thread (M). Each thread amortizes ONE fed_invert
// (~265 field muls) across this many points via a Montgomery batch inversion,
// so the per-step inversion cost is ~265/M muls. The kernel uses 178 registers
// (occupancy is register-bound, constant in M), so larger M wins purely by
// amortizing the inversion -- measured 65/116/214/252 M keys/s at M=32/64/256/
// 512 on a GTX 1650. The per-thread Fe scratch (3 * M * 40 bytes) lives in
// local memory: at M=256 that is ~30 KB/thread, so the local-memory reservation
// (~ max_resident_threads * 30 KB) stays safely under the 4 GB card. M=512
// (61 KB/thread) is ~15% faster but its reservation approaches VRAM, risking a
// launch OOM, so 256 is the robust default knee of the curve.
inline constexpr int kStepsPerThread = 256;

// Max compiled patterns held in __constant__ memory (broadcast to all threads).
inline constexpr int kMaxConstPatterns = 16;

// Max hits recorded per kernel launch. Overflow is detected by the host via the
// hit counter (atomicAdd returns the pre-increment value); excess hits beyond
// this cap are simply dropped for that launch and rediscovered next epoch.
inline constexpr int kMaxHitSlots = 256;

// Device-side compiled pattern (mirrors core::CompiledPattern's hot fields).
struct DevPattern {
    uint8_t bytes[32];
    uint8_t mask[32];
    int nbytes;
};

// A recorded hit: candidate index = t + j*T, scalar = a0 + 8*(t + j*T).
struct HitSlot {
    int t;
    int j;
    int pattern_index;
};

// Upload compiled patterns into __constant__ memory (n <= kMaxConstPatterns).
cudaError_t upload_patterns(const DevPattern* patterns, int n);

// Launch the search kernel: T threads, each walks kStepsPerThread points by
// BIGSTEP and records hits into d_slots/d_hit_count. Caller sizes block.
void launch_search_kernel(const GeP3* d_starts,
                          const GeCachedAffine* d_bigstep,
                          int T,
                          int block,
                          HitSlot* d_slots,
                          int* d_hit_count,
                          cudaStream_t stream);

}  // namespace onion::cuda
