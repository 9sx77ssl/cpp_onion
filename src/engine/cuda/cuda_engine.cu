// CUDA search engine host loop (Phase 3 Task 2). All CUDA calls live here,
// behind the plain-C++ CudaEngine boundary (cuda_engine.hpp).
//
// Per epoch:
//   1. Draw a fresh clamped base scalar a0.
//   2. Compute A0 = a0*B and the cached affine step 8B (validated scalar code).
//   3. Walk T cheap point-additions P_0=A0, P_{t+1}=P_t+8B -> one start per
//      thread. Compute BIGSTEP = T*(8B) ONCE (T-1 additions). Upload both.
//   4. Launch the search kernel launches_per_epoch times. Each launch: every
//      thread walks M=kStepsPerThread points by BIGSTEP, recovering y with one
//      batched inversion, masked-comparing against the patterns. Candidate
//      index of thread t, step j is  idx_base + t + j*T  where idx_base grows
//      by T*M each launch (so successive launches scan fresh, disjoint ranges).
//   5. Read back hits; reconstruct scalar = a0 + 8*idx; build MatchCandidate;
//      push. The CLI io::verify firewall is the final gate.

#include "engine/cuda/cuda_engine.hpp"
#include "engine/cuda/device_field.cuh"
#include "engine/cuda/search_kernel.cuh"

#include "crypto/ge25519.hpp"
#include "crypto/incremental.hpp"
#include "crypto/keys.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

namespace onion::cuda {
namespace {

// Bridge a host crypto::Fe (5x51 limbs) into the device-layout Fe (identical).
Fe to_device_fe(const onion::crypto::Fe& f) {
    Fe r;
    for (int i = 0; i < 5; ++i) r.v[i] = f.v[i];
    return r;
}

GeP3 to_device_p3(const onion::crypto::GeP3& p) {
    return GeP3{to_device_fe(p.X), to_device_fe(p.Y), to_device_fe(p.Z), to_device_fe(p.T)};
}

GeCachedAffine to_device_cached(const onion::crypto::GeCachedAffine& q) {
    return GeCachedAffine{to_device_fe(q.YplusX), to_device_fe(q.YminusX), to_device_fe(q.T2d)};
}

#define CUDA_OK(expr)                                                        \
    do {                                                                     \
        cudaError_t err__ = (expr);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr,      \
                         __FILE__, __LINE__, cudaGetErrorString(err__));     \
            return false;                                                    \
        }                                                                    \
    } while (0)

}  // namespace

struct CudaEngine::Impl {
    std::vector<onion::core::CompiledPattern> patterns;
    onion::engine::ResultQueue& results;
    onion::engine::StatsBoard& stats;
    CudaKnobs knobs;

    // Device buffers (allocated once, reused every epoch).
    GeP3* d_starts = nullptr;
    GeCachedAffine* d_bigstep = nullptr;
    HitSlot* d_slots = nullptr;
    int* d_hit_count = nullptr;
    cudaStream_t stream = nullptr;
    bool ok = false;  // false => construction failed; run() reports and returns.

    Impl(std::vector<onion::core::CompiledPattern> p,
         onion::engine::ResultQueue& r,
         onion::engine::StatsBoard& s,
         CudaKnobs k)
        : patterns(std::move(p)), results(r), stats(s), knobs(k) {}

    ~Impl() {
        if (d_starts) cudaFree(d_starts);
        if (d_bigstep) cudaFree(d_bigstep);
        if (d_slots) cudaFree(d_slots);
        if (d_hit_count) cudaFree(d_hit_count);
        if (stream) cudaStreamDestroy(stream);
    }

    // One-time device setup: validate knobs, upload patterns, allocate buffers.
    bool init() {
        if (patterns.empty()) {
            std::fprintf(stderr, "CudaEngine: no patterns\n");
            return false;
        }
        if (static_cast<int>(patterns.size()) > kMaxConstPatterns) {
            std::fprintf(stderr, "CudaEngine: too many patterns (%zu > %d)\n",
                         patterns.size(), kMaxConstPatterns);
            return false;
        }
        if (knobs.threads <= 0 || knobs.block <= 0) {
            std::fprintf(stderr, "CudaEngine: invalid knobs\n");
            return false;
        }

        // Pack patterns into the device-constant form and upload.
        std::vector<DevPattern> dev(patterns.size());
        for (std::size_t i = 0; i < patterns.size(); ++i) {
            const auto& src = patterns[i];
            std::memcpy(dev[i].bytes, src.bytes.data(), 32);
            std::memcpy(dev[i].mask, src.mask.data(), 32);
            dev[i].nbytes = static_cast<int>(src.nbytes);
        }
        CUDA_OK(upload_patterns(dev.data(), static_cast<int>(dev.size())));

        const int T = knobs.threads;
        CUDA_OK(cudaMalloc(&d_starts, static_cast<std::size_t>(T) * sizeof(GeP3)));
        CUDA_OK(cudaMalloc(&d_bigstep, sizeof(GeCachedAffine)));
        CUDA_OK(cudaMalloc(&d_slots, sizeof(HitSlot) * kMaxHitSlots));
        CUDA_OK(cudaMalloc(&d_hit_count, sizeof(int)));
        CUDA_OK(cudaStreamCreate(&stream));
        return true;
    }

    // Build per-thread start points and BIGSTEP on the host with cheap point
    // adds, then upload. start[t] = A0 + t*(8B); bigstep = T*(8B). Returns
    // false on a CUDA upload error.
    bool setup_epoch(std::span<const std::byte, 32> a0);
};

bool CudaEngine::Impl::setup_epoch(std::span<const std::byte, 32> a0) {
    using namespace onion::crypto;
    const int T = knobs.threads;

    // A0 = a0*B and the cached affine step 8B (validated scalar code).
    onion::crypto::GeP3 A0 = ge_scalarmult_base(a0);
    std::array<std::byte, 32> eight{};
    eight[0] = std::byte{8};
    onion::crypto::GeP3 step8b_p3 = ge_scalarmult_base(eight);
    onion::crypto::GeCachedAffine step8b = ge_to_cached_affine(step8b_p3);

    // T cheap point additions: P_0 = A0, P_{t+1} = P_t + 8B. Each thread's
    // start point. BIGSTEP = T*(8B) accumulated alongside.
    std::vector<GeP3> h_starts(static_cast<std::size_t>(T));
    onion::crypto::GeP3 cur = A0;
    onion::crypto::GeP3 bigstep_p3 = step8b_p3;  // running k*(8B), k starts at 1
    for (int t = 0; t < T; ++t) {
        h_starts[t] = to_device_p3(cur);
        cur = ge_madd(cur, step8b);             // P_{t+1} = P_t + 8B
        if (t + 1 < T)                          // accumulate T*(8B): T-1 adds
            bigstep_p3 = ge_madd(bigstep_p3, step8b);
    }
    // bigstep_p3 now holds (1 + (T-1)) * 8B = T * 8B.
    GeCachedAffine h_bigstep = to_device_cached(ge_to_cached_affine(bigstep_p3));

    CUDA_OK(cudaMemcpyAsync(d_starts, h_starts.data(),
                            static_cast<std::size_t>(T) * sizeof(GeP3),
                            cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_bigstep, &h_bigstep, sizeof(GeCachedAffine),
                            cudaMemcpyHostToDevice, stream));
    return true;
}

CudaEngine::CudaEngine(std::vector<onion::core::CompiledPattern> patterns,
                       onion::engine::ResultQueue& results,
                       onion::engine::StatsBoard& stats,
                       CudaKnobs knobs)
    : impl_(std::make_unique<Impl>(std::move(patterns), results, stats, knobs)) {
    impl_->ok = impl_->init();
}

CudaEngine::~CudaEngine() = default;

void CudaEngine::run(std::stop_token stop) {
    Impl& s = *impl_;
    if (!s.ok) {
        std::fprintf(stderr, "CudaEngine: device init failed; engine idle\n");
        return;
    }

    const int T = s.knobs.threads;
    const std::uint64_t per_launch = static_cast<std::uint64_t>(T) * kStepsPerThread;
    std::uint64_t total = 0;
    std::vector<HitSlot> h_slots(kMaxHitSlots);

    // One epoch == one kernel launch over a fresh, fully-disjoint candidate
    // space. The device walks each thread t from its start point P_t = A0+t*8B
    // for M=kStepsPerThread steps of BIGSTEP=T*8B, so thread t, step j visits
    // candidate index exactly  t + j*T  (covering [0, T*M) with no gaps/overlap
    // within the epoch). Re-seeding a0 each epoch draws a brand-new random
    // T*M-sized slab of the keyspace, so epochs never recheck the same key.
    // The host setup (T cheap point-adds) is ~one M-th of the device work, a
    // few-percent overhead, so per-epoch re-seed costs little.
    while (!stop.stop_requested()) {
        std::array<std::byte, 32> a0;
        onion::crypto::random_bytes(a0);
        a0[0] = std::byte(std::to_integer<unsigned>(a0[0]) & 0xf8);
        a0[31] = std::byte((std::to_integer<unsigned>(a0[31]) & 0x7f) | 0x40);

        if (!s.setup_epoch(a0)) {
            if (stop.stop_requested()) break;
            continue;  // transient setup error: re-seed next iteration.
        }

        const int zero = 0;
        if (cudaMemcpyAsync(s.d_hit_count, &zero, sizeof(int),
                            cudaMemcpyHostToDevice, s.stream) != cudaSuccess)
            continue;

        launch_search_kernel(s.d_starts, s.d_bigstep, T, s.knobs.block,
                             s.d_slots, s.d_hit_count, s.stream);
        if (cudaGetLastError() != cudaSuccess) continue;

        int hit_count = 0;
        if (cudaMemcpyAsync(&hit_count, s.d_hit_count, sizeof(int),
                            cudaMemcpyDeviceToHost, s.stream) != cudaSuccess)
            continue;
        if (cudaStreamSynchronize(s.stream) != cudaSuccess) continue;

        if (hit_count > 0) {
            const int n = hit_count < kMaxHitSlots ? hit_count : kMaxHitSlots;
            if (cudaMemcpy(h_slots.data(), s.d_slots, sizeof(HitSlot) * n,
                           cudaMemcpyDeviceToHost) == cudaSuccess) {
                for (int i = 0; i < n; ++i) {
                    const HitSlot& hit = h_slots[i];
                    // candidate index = t + j*T (this epoch's a0).
                    const std::uint64_t idx =
                        static_cast<std::uint64_t>(hit.t) +
                        static_cast<std::uint64_t>(hit.j) * static_cast<std::uint64_t>(T);
                    auto scalar = onion::crypto::scalar_add_8i(a0, idx);
                    auto pk = onion::crypto::pubkey_from_scalar(scalar);
                    if (!pk) continue;  // degenerate scalar: skip (firewall safe)
                    onion::engine::MatchCandidate cand;
                    cand.secret.scalar = scalar;
                    onion::crypto::random_bytes(cand.secret.prf_prefix);  // fresh RH
                    cand.claimed_pubkey = *pk;
                    cand.pattern_index = static_cast<std::size_t>(hit.pattern_index);
                    s.results.push(std::move(cand));
                }
            }
        }

        total += per_launch;
        s.stats.set(0, total);
    }
}

}  // namespace onion::cuda
