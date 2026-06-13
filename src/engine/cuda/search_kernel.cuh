#pragma once
//
// CUDA-side interface to the vanity-search hot kernel. Included ONLY by .cu
// translation units (it pulls in cuda_runtime via the kernel launch wrappers
// and device_field.cuh layouts). The plain-C++ engine boundary is in
// cuda_engine.hpp; this header is the .cu <-> .cu seam.

#include "engine/cuda/device_field_select.cuh"

#include <cstdint>

namespace onion::cuda {

// Compile-time steps per thread (M). Each thread amortizes ONE fed_invert
// (~265 field muls) across this many points via a Montgomery batch inversion,
// so the per-step inversion cost is ~265/M muls. Larger M wins purely by
// amortizing the inversion -- measured on a GTX 1650 at T=2^14: M=256->214,
// 512->256, 768->267, 1024->275 M keys/s. The per-thread scratch lives in
// local memory (global-backed, L1/L2-cached); its bandwidth is the kernel's
// real bottleneck, so the default search_kernel uses a FUSED 2-array batch
// inversion (combo[]=Y*prefix and zs[]=z; see search_kernel.cu) instead of the
// old 3-array (ys[]/zs[]/prefix[]) scheme. With the 32-bit field (Fe=32 B) that
// cuts the stack frame from ~96 KB to ~64 KB/thread (2 * M * 32 B), reducing
// local-memory traffic ~33% for a measured +~7% (302 -> 323 M keys/s) on the
// GTX 1650 at T=2^15. The runtime reservation at T=2^15 fits the 4 GB card. The
// curve is knee-flattening (768->1024 is only ~3%) and beyond 1024 the
// reservation approaches VRAM with negligible gain, so 1024 is the robust knee.
inline constexpr int kStepsPerThread = 1024;

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
