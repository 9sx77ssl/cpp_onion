// CUDA vanity-search hot kernel (Phase 3 Task 2).
//
// INTERLEAVED scheme (cheap host setup, see cuda_engine.cu run loop):
//   Host picks one random clamped base scalar a0, computes A0 = a0*B and the
//   cached affine step 8B (validated scalar code). It then walks T cheap point
//   additions P_0 = A0, P_{t+1} = P_t + 8B to get one start point per thread,
//   and computes BIGSTEP = T*(8B) ONCE. Both are uploaded.
//
//   Kernel: thread t loads its start point P_t and walks A += BIGSTEP for M
//   steps. Step j visits candidate index  idx = t + j*T  (scalar a0 + 8*idx).
//   To avoid one fed_invert per step (which would make the GPU slower than the
//   CPU), each thread batches the M Z-values and does ONE Montgomery batch
//   inversion, then recovers y = Y/Z for all M points, masked-compares against
//   the compiled patterns, and on a hit records (t, j) into a slot buffer.
//
// The host (cuda_engine.cu) reconstructs scalar = a0 + 8*(t + j*T), rebuilds
// the candidate via the validated scalar pubkey path, and pushes it; the CLI's
// io::verify firewall is the final gate. A bogus hit cannot escape.

#include "engine/cuda/device_field.cuh"
#include "engine/cuda/search_kernel.cuh"

namespace onion::cuda {

// Patterns live in __constant__ memory: every thread reads the same bytes,
// so the constant cache broadcasts them with zero divergence. Capacity is
// fixed; the host asserts patterns.size() <= kMaxConstPatterns.
__constant__ DevPattern c_patterns[kMaxConstPatterns];
__constant__ int c_num_patterns;

// Encode the recovered affine y and masked-compare against every compiled
// pattern (held in __constant__, broadcast to all threads). On a match, record
// (t, j, pattern_index) into the hit-slot buffer via an atomic counter; the
// host reconstructs the scalar and the io::verify firewall is the final gate.
__device__ __forceinline__ void
match_and_record(const Fe& y, int t, int j, HitSlot* slots, int* hit_count) {
    uint8_t enc[32];
    fed_to_bytes(enc, y);
    for (int p = 0; p < c_num_patterns; ++p) {
        const DevPattern& pat = c_patterns[p];
        bool ok = true;
        for (int b = 0; b < pat.nbytes; ++b) {
            if ((enc[b] & pat.mask[b]) != pat.bytes[b]) { ok = false; break; }
        }
        if (ok) {
            int slot = atomicAdd(hit_count, 1);
            if (slot < kMaxHitSlots) {
                slots[slot].t = t;
                slots[slot].j = j;
                slots[slot].pattern_index = p;
            }
            // One pattern match is enough to record this candidate; the host
            // re-checks all patterns during verify anyway.
            return;
        }
    }
}

// grid = ceil(T / block). Each thread owns one chain start point.
// __launch_bounds__ was measured to be neutral: ptxas already settles at 178
// registers (=> 2 resident blocks/SM at block=128 on sm_75), and the kernel is
// field-arithmetic/local-memory bound rather than occupancy bound, so pinning
// the launch bounds neither lowered the register count nor improved throughput.
__global__ void search_kernel(const GeP3* __restrict__ starts,
                              const GeCachedAffine* __restrict__ bigstep,
                              int T,
                              HitSlot* slots,
                              int* hit_count) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    GeP3 cur = starts[t];
    const GeCachedAffine step = *bigstep;

    // Per-thread scratch for the Montgomery batch inversion over M Z-values.
    Fe ys[kStepsPerThread];
    Fe zs[kStepsPerThread];
    Fe prefix[kStepsPerThread];

    // Walk M steps, stashing Y and Z; advance by BIGSTEP each step.
    #pragma unroll 1
    for (int j = 0; j < kStepsPerThread; ++j) {
        ys[j] = cur.Y;
        zs[j] = cur.Z;
        cur = ged_madd(cur, step);
    }

    // prefix[i] = z[0]*...*z[i]
    prefix[0] = zs[0];
    #pragma unroll 1
    for (int i = 1; i < kStepsPerThread; ++i)
        prefix[i] = fed_mul(prefix[i - 1], zs[i]);

    Fe inv = fed_invert(prefix[kStepsPerThread - 1]);  // 1 / prod(z)

    // Back-substitute: zinv[i] = inv * prefix[i-1]; recover y; inv *= z[i].
    #pragma unroll 1
    for (int i = kStepsPerThread; i-- > 0;) {
        Fe zinv = (i == 0) ? inv : fed_mul(inv, prefix[i - 1]);
        Fe y = fed_mul(ys[i], zinv);
        match_and_record(y, t, i, slots, hit_count);
        if (i != 0) inv = fed_mul(inv, zs[i]);
    }
}

void launch_search_kernel(const GeP3* d_starts,
                          const GeCachedAffine* d_bigstep,
                          int T,
                          int block,
                          HitSlot* d_slots,
                          int* d_hit_count,
                          cudaStream_t stream) {
    const int grid = (T + block - 1) / block;
    search_kernel<<<grid, block, 0, stream>>>(d_starts, d_bigstep, T, d_slots, d_hit_count);
}

cudaError_t upload_patterns(const DevPattern* patterns, int n) {
    cudaError_t err = cudaMemcpyToSymbol(c_patterns, patterns, sizeof(DevPattern) * n);
    if (err != cudaSuccess) return err;
    return cudaMemcpyToSymbol(c_num_patterns, &n, sizeof(int));
}

}  // namespace onion::cuda
