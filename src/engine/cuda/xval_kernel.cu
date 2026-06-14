// CUDA cross-validation kernel + host harness (Phase 3 Task 1, THE GATE).
//
// One device thread per chain walks A0, A0+8B, A0+16B, ... using the device
// 5x51 field/point core (device_field.cuh) and writes each point's 32-byte
// y-encoding. Host setup uses the validated CPU scalar code to compute the
// per-chain start point A0 and the shared 8B cached-affine step, then uploads
// them -- the device only walks (ged_madd) and recovers y = Y/Z (fed_invert).
//
// The host test then checks every (chain, i) against
// crypto_scalarmult_ed25519_base_noclamp(a0 + 8i). Never weaken that gate.

#include "engine/cuda/device_field_select.cuh"
#include "engine/cuda/xval.hpp"

#include "crypto/ge25519.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

namespace onion::cuda {
namespace {

__global__ void incremental_chain(const GeP3* starts,
                                  const GeCachedAffine* step8b,
                                  uint8_t* out_y,
                                  int n_chains,
                                  int steps) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_chains) return;

    GeP3 cur = starts[c];
    const GeCachedAffine s = *step8b;
    uint8_t* base = out_y + (size_t)c * steps * 32;

    for (int i = 0; i < steps; ++i) {
        // y = Y / Z, encoded little-endian (sign bit not set, like the hot path).
        Fe zinv = fed_invert(cur.Z);
        Fe y = fed_mul(cur.Y, zinv);
        fed_to_bytes(base + (size_t)i * 32, y);
        cur = ged_madd(cur, s);
    }
}

#define CUDA_CHK(expr)                                                        \
    do {                                                                      \
        cudaError_t err__ = (expr);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr,       \
                         __FILE__, __LINE__, cudaGetErrorString(err__));      \
            return (int)err__;                                                \
        }                                                                     \
    } while (0)

// Minimal RAII owner of a cudaMalloc'd buffer for this test harness, mirroring
// cuda_engine.cu's CudaBuf discipline: any CUDA_CHK early-return below frees
// every already-acquired buffer via the destructor (cudaFree(nullptr) is a
// no-op). Keeps the one non-production code path in the CUDA backend leak-clean
// on the error paths too, not just on success.
template <typename T>
class ScopedCudaBuf {
public:
    ScopedCudaBuf() = default;
    ~ScopedCudaBuf() { if (p_) cudaFree(p_); }
    ScopedCudaBuf(const ScopedCudaBuf&) = delete;
    ScopedCudaBuf& operator=(const ScopedCudaBuf&) = delete;
    cudaError_t alloc(size_t n) { return cudaMalloc(&p_, n * sizeof(T)); }
    [[nodiscard]] T* get() const noexcept { return p_; }

private:
    T* p_ = nullptr;
};

}  // namespace

int run_incremental_xval(const uint8_t* a0, uint8_t* out_y, int n_chains, int steps) {
    using namespace onion::crypto;

    // --- Host setup (validated scalar code): start points + shared 8B step. ---
    std::vector<GeP3> h_starts(n_chains);
    for (int c = 0; c < n_chains; ++c) {
        std::array<std::byte, 32> scalar;
        std::memcpy(scalar.data(), a0 + (size_t)c * 32, 32);
        onion::crypto::GeP3 A0 = ge_scalarmult_base(scalar);
        h_starts[c] = to_device_p3(A0);
    }
    std::array<std::byte, 32> eight{};
    eight[0] = std::byte{8};
    onion::crypto::GeCachedAffine step = ge_to_cached_affine(ge_scalarmult_base(eight));
    GeCachedAffine h_step = to_device_cached(step);

    // --- Upload, launch, download. ---
    // RAII-owned device buffers: every CUDA_CHK early-return below frees them.
    ScopedCudaBuf<GeP3> d_starts;
    ScopedCudaBuf<GeCachedAffine> d_step;
    ScopedCudaBuf<uint8_t> d_out;
    const size_t out_bytes = (size_t)n_chains * steps * 32;

    CUDA_CHK(d_starts.alloc((size_t)n_chains));
    CUDA_CHK(d_step.alloc(1));
    CUDA_CHK(d_out.alloc(out_bytes));
    CUDA_CHK(cudaMemcpy(d_starts.get(), h_starts.data(), (size_t)n_chains * sizeof(GeP3),
                        cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_step.get(), &h_step, sizeof(GeCachedAffine),
                        cudaMemcpyHostToDevice));

    const int block = 64;
    const int grid = (n_chains + block - 1) / block;
    incremental_chain<<<grid, block>>>(d_starts.get(), d_step.get(), d_out.get(),
                                       n_chains, steps);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaMemcpy(out_y, d_out.get(), out_bytes, cudaMemcpyDeviceToHost));
    return 0;
}

}  // namespace onion::cuda
