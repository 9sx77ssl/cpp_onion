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
#include "engine/cuda/device_field_select.cuh"
#include "engine/cuda/search_kernel.cuh"

#include "crypto/ge25519.hpp"
#include "crypto/incremental.hpp"
#include "crypto/keys.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

namespace onion::cuda {
namespace {

// --- RAII device-memory ownership -----------------------------------------
// A move-only owner of a cudaMalloc'd buffer that frees it via cudaFree in its
// destructor. This makes init() all-or-nothing for free: if any later
// allocation fails (or throws), every already-acquired buffer is released when
// the Impl is torn down, with no manual NULL-check teardown. cudaFree(nullptr)
// is a documented no-op, so a default-constructed (empty) owner is safe.
template <typename T>
class CudaBuf {
public:
    CudaBuf() = default;
    ~CudaBuf() { reset(); }

    CudaBuf(const CudaBuf&) = delete;
    CudaBuf& operator=(const CudaBuf&) = delete;
    CudaBuf(CudaBuf&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
    CudaBuf& operator=(CudaBuf&& o) noexcept {
        if (this != &o) { reset(); p_ = std::exchange(o.p_, nullptr); }
        return *this;
    }

    // Allocate n elements; frees any previously held buffer first. Returns the
    // cudaError_t so the caller can route it through CUDA_OK.
    cudaError_t alloc(std::size_t n) {
        reset();
        return cudaMalloc(&p_, n * sizeof(T));
    }
    void reset() noexcept {
        if (p_) { cudaFree(p_); p_ = nullptr; }
    }
    [[nodiscard]] T* get() const noexcept { return p_; }

private:
    T* p_ = nullptr;
};

// RAII owner of a pinned (page-locked) host buffer (cudaMallocHost/cudaFreeHost).
// Pinned host memory is REQUIRED for cudaMemcpyAsync to be truly asynchronous
// AND so the source survives a deferred copy: the per-epoch start points and
// bigstep are uploaded with cudaMemcpyAsync, so their host storage must outlive
// the copy. Owning it in Impl (allocate-once-reuse) gives that lifetime and
// also drops the per-epoch heap allocation.
template <typename T>
class PinnedBuf {
public:
    PinnedBuf() = default;
    ~PinnedBuf() { reset(); }

    PinnedBuf(const PinnedBuf&) = delete;
    PinnedBuf& operator=(const PinnedBuf&) = delete;
    PinnedBuf(PinnedBuf&& o) noexcept
        : p_(std::exchange(o.p_, nullptr)), n_(std::exchange(o.n_, 0)) {}
    PinnedBuf& operator=(PinnedBuf&& o) noexcept {
        if (this != &o) {
            reset();
            p_ = std::exchange(o.p_, nullptr);
            n_ = std::exchange(o.n_, 0);
        }
        return *this;
    }

    cudaError_t alloc(std::size_t n) {
        reset();
        cudaError_t e = cudaMallocHost(&p_, n * sizeof(T));
        if (e == cudaSuccess) n_ = n;
        return e;
    }
    void reset() noexcept {
        if (p_) { cudaFreeHost(p_); p_ = nullptr; n_ = 0; }
    }
    [[nodiscard]] T* get() const noexcept { return p_; }
    [[nodiscard]] std::size_t size() const noexcept { return n_; }

private:
    T* p_ = nullptr;
    std::size_t n_ = 0;
};

// RAII owner of a cudaStream_t (cudaStreamCreate/cudaStreamDestroy).
class CudaStream {
public:
    CudaStream() = default;
    ~CudaStream() { reset(); }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    // Move-constructible only: these are fixed pipeline-slot array members that
    // are never reassigned, so a move-assignment operator would be dead code
    // (and nvcc flags the unreferenced concrete-class operator with #177-D).
    CudaStream(CudaStream&& o) noexcept : s_(std::exchange(o.s_, nullptr)) {}

    cudaError_t create() {
        reset();
        return cudaStreamCreate(&s_);
    }
    void reset() noexcept {
        if (s_) { cudaStreamDestroy(s_); s_ = nullptr; }
    }
    [[nodiscard]] cudaStream_t get() const noexcept { return s_; }

private:
    cudaStream_t s_ = nullptr;
};

// Host->device Fe/GeP3/GeCachedAffine bridges live in device_field_select.cuh
// (they handle both the 51-bit and 32-bit device layouts). Pulled in above.

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

    // Double-buffered ("ping-pong") per-epoch resources. The host setup of
    // epoch N+1 (T cheap point-adds + its H2D upload) is overlapped with the
    // long-running kernel of epoch N by giving each in-flight epoch its OWN copy
    // of every per-epoch buffer and its own stream, so writing N+1's start
    // points never races N's kernel reads. kPipelineDepth==1 collapses this to
    // the original serial single-buffer behaviour (used as the fallback path).
    //
    // Allocated ONCE in init(), reused every epoch; the per-epoch loop performs
    // ZERO cudaMalloc/cudaFree. RAII owners free them on Impl teardown.
#ifdef ONION_CUDA_NO_PIPELINE
    static constexpr int kPipelineDepth = 1;
#else
    static constexpr int kPipelineDepth = 2;
#endif
    CudaBuf<GeP3> d_starts[kPipelineDepth];
    CudaBuf<GeCachedAffine> d_bigstep[kPipelineDepth];
    CudaBuf<HitSlot> d_slots[kPipelineDepth];
    CudaBuf<int> d_hit_count[kPipelineDepth];
    CudaStream stream[kPipelineDepth];

    // Persistent PINNED host staging buffers (allocated once, one set per
    // pipeline slot). The per-epoch upload uses cudaMemcpyAsync, so these MUST
    // outlive the copy and be page-locked; owning them here gives that lifetime
    // and removes the per-epoch heap allocation. h_starts holds the T
    // per-thread start points; the small ones stage bigstep / hit_count for
    // async H2D/D2H. Each slot's a0 is retained so its hits can be reconstructed
    // after the (deferred) sync.
    PinnedBuf<GeP3> h_starts[kPipelineDepth];
    PinnedBuf<GeCachedAffine> h_bigstep[kPipelineDepth];
    PinnedBuf<int> h_hit_count[kPipelineDepth];
    PinnedBuf<HitSlot> h_slots[kPipelineDepth];
    std::array<std::byte, 32> slot_a0[kPipelineDepth];

    // Robustness accounting (printed once on shutdown if non-zero). None of
    // these affect correctness -- the io::verify firewall is the final gate --
    // but they surface silent loss so a systematic bug cannot hide.
    std::uint64_t dropped_epoch_errors = 0;   // epoch skipped on a CUDA error
    std::uint64_t dropped_degenerate = 0;     // hit -> scalar failed clamp/pubkey
    std::uint64_t hit_overflow_events = 0;     // launch produced > kMaxHitSlots hits

    bool ok = false;  // false => construction failed; run() reports and returns.

    Impl(std::vector<onion::core::CompiledPattern> p,
         onion::engine::ResultQueue& r,
         onion::engine::StatsBoard& s,
         CudaKnobs k)
        : patterns(std::move(p)), results(r), stats(s), knobs(k) {}

    // RAII members free all device/host/stream resources automatically; no
    // hand-written destructor is needed (and none can leak a buffer).

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
#ifdef ONION_CUDA_FIELD32
        // The native 32-bit field uses ~113 regs/thread at the committed M=3072
        // (0 spills; ~96 at the early M=256/1024 sweep) vs 178 for the 51-bit
        // __int128 field, and a 32-byte Fe (vs 40), so more threads stay
        // resident AND the per-thread local-memory reservation is smaller. T =
        // 2^15 measured fastest (~302 M keys/s vs ~290 at 2^14) and its ~3.2 GB
        // local reservation fits the 4 GB card. Only bump if the caller left
        // the shared default; env override still wins below.
        if (knobs.threads == (1 << 14)) knobs.threads = 1 << 15;
#endif
        // Optional env overrides for tuning sweeps (T and block). Compile-time
        // M lives in search_kernel.cuh. Invalid/absent values keep the defaults.
        if (const char* e = std::getenv("ONION_CUDA_THREADS")) {
            int v = std::atoi(e);
            if (v > 0) knobs.threads = v;
        }
        if (const char* e = std::getenv("ONION_CUDA_BLOCK")) {
            int v = std::atoi(e);
            if (v > 0) knobs.block = v;
        }
        if (knobs.threads <= 0 || knobs.block <= 0) {
            std::fprintf(stderr, "CudaEngine: invalid knobs\n");
            return false;
        }

        // Pack patterns into the device-constant form and upload. The count is
        // validated above (<= kMaxConstPatterns), and upload_patterns()'s
        // return is checked, so c_patterns / c_num_patterns are always
        // consistent on the device.
        std::vector<DevPattern> dev(patterns.size());
        for (std::size_t i = 0; i < patterns.size(); ++i) {
            const auto& src = patterns[i];
            std::memcpy(dev[i].bytes, src.bytes.data(), 32);
            std::memcpy(dev[i].mask, src.mask.data(), 32);
            dev[i].nbytes = static_cast<int>(src.nbytes);
        }
        CUDA_OK(upload_patterns(dev.data(), static_cast<int>(dev.size())));

        const std::size_t T = static_cast<std::size_t>(knobs.threads);
        // Per-slot device + pinned-host buffers and streams. On any failure
        // CUDA_OK returns false and the RAII owners free whatever was already
        // acquired (all-or-nothing). kPipelineDepth sets are allocated; the
        // doubled per-slot footprint is tiny (d_starts is the largest at
        // T*128 B ~= 4 MB/slot) and does NOT change the kernel's dominant
        // per-thread local-memory reservation (only one kernel is ever in
        // flight), so the VRAM peak is effectively unchanged.
        for (int s = 0; s < kPipelineDepth; ++s) {
            CUDA_OK(d_starts[s].alloc(T));
            CUDA_OK(d_bigstep[s].alloc(1));
            CUDA_OK(d_slots[s].alloc(kMaxHitSlots));
            CUDA_OK(d_hit_count[s].alloc(1));
            CUDA_OK(h_starts[s].alloc(T));
            CUDA_OK(h_bigstep[s].alloc(1));
            CUDA_OK(h_hit_count[s].alloc(1));
            CUDA_OK(h_slots[s].alloc(kMaxHitSlots));
            CUDA_OK(stream[s].create());
        }
        return true;
    }

    // Build per-thread start points and BIGSTEP on the host with cheap point
    // adds (into pipeline slot `slot`'s pinned buffers), then enqueue the H2D
    // uploads + hit-counter reset on that slot's stream. start[t] = A0 + t*(8B);
    // bigstep = T*(8B). All async work for this epoch lives on stream[slot].
    // Returns false on a CUDA upload error.
    bool setup_epoch(int slot, std::span<const std::byte, 32> a0);

    // Enqueue the search kernel for `slot` plus its hit-count readback on the
    // slot's stream. Assumes setup_epoch(slot, ...) already enqueued the
    // uploads + hit-counter reset on the SAME stream (so the kernel is ordered
    // after them). Returns false on a launch error. No host sync here -- the
    // run loop defers the sync so the next epoch's host setup can overlap.
    bool launch_epoch(int slot) {
        launch_search_kernel(d_starts[slot].get(), d_bigstep[slot].get(),
                             knobs.threads, knobs.block, d_slots[slot].get(),
                             d_hit_count[slot].get(), stream[slot].get());
        if (cudaGetLastError() != cudaSuccess) return false;
        // hit_count readback into this slot's pinned staging; one later sync of
        // this slot's stream drains upload+reset+kernel+readback together.
        return cudaMemcpyAsync(h_hit_count[slot].get(), d_hit_count[slot].get(),
                               sizeof(int), cudaMemcpyDeviceToHost,
                               stream[slot].get()) == cudaSuccess;
    }

    // Sync `slot`'s stream and turn any recorded hits into pushed candidates,
    // reconstructing scalars from this slot's retained a0. Returns false only on
    // a CUDA error (so the caller can account a dropped epoch).
    bool drain_epoch(int slot);
};

bool CudaEngine::Impl::setup_epoch(int slot, std::span<const std::byte, 32> a0) {
    using namespace onion::crypto;
    const int T = knobs.threads;

    // A0 = a0*B and the cached affine step 8B (validated scalar code).
    onion::crypto::GeP3 A0 = ge_scalarmult_base(a0);
    std::array<std::byte, 32> eight{};
    eight[0] = std::byte{8};
    onion::crypto::GeP3 step8b_p3 = ge_scalarmult_base(eight);
    onion::crypto::GeCachedAffine step8b = ge_to_cached_affine(step8b_p3);

    // Retain this epoch's a0 in the slot so its hits (read back after the
    // deferred sync) can be reconstructed even though the next iteration is
    // already preparing the OTHER slot's a0.
    std::memcpy(slot_a0[slot].data(), a0.data(), 32);

    // T cheap point additions, written straight into this slot's persistent
    // pinned staging buffer: P_0 = A0, P_{t+1} = P_t + 8B. Each thread's start
    // point. BIGSTEP = T*(8B) accumulated alongside. This CPU-only work is what
    // the pipeline overlaps with the previous epoch's in-flight kernel.
    GeP3* starts = h_starts[slot].get();
    onion::crypto::GeP3 cur = A0;
    onion::crypto::GeP3 bigstep_p3 = step8b_p3;  // running k*(8B), k starts at 1
    for (int t = 0; t < T; ++t) {
        starts[t] = to_device_p3(cur);
        cur = ge_madd(cur, step8b);             // P_{t+1} = P_t + 8B
        if (t + 1 < T)                          // accumulate T*(8B): T-1 adds
            bigstep_p3 = ge_madd(bigstep_p3, step8b);
    }
    // bigstep_p3 now holds (1 + (T-1)) * 8B = T * 8B.
    *h_bigstep[slot].get() = to_device_cached(ge_to_cached_affine(bigstep_p3));

    cudaStream_t st = stream[slot].get();
    // The source buffers (this slot's h_starts/h_bigstep/h_hit_count) are
    // persistent and pinned, so these async copies are safe and stay async. The
    // slot is not reused until its own stream is synchronized (run loop), so the
    // next overwrite of these host buffers cannot race an in-flight copy.
    CUDA_OK(cudaMemcpyAsync(d_starts[slot].get(), starts,
                            static_cast<std::size_t>(T) * sizeof(GeP3),
                            cudaMemcpyHostToDevice, st));
    CUDA_OK(cudaMemcpyAsync(d_bigstep[slot].get(), h_bigstep[slot].get(),
                            sizeof(GeCachedAffine),
                            cudaMemcpyHostToDevice, st));
    // Reset this slot's device hit counter on the same stream (ordered before
    // the kernel the run loop launches on this stream).
    *h_hit_count[slot].get() = 0;
    CUDA_OK(cudaMemcpyAsync(d_hit_count[slot].get(), h_hit_count[slot].get(),
                            sizeof(int), cudaMemcpyHostToDevice, st));
    return true;
}

bool CudaEngine::Impl::drain_epoch(int slot) {
    const int T = knobs.threads;
    // Drain this slot's whole epoch (upload, reset, kernel, hit_count readback).
    if (cudaStreamSynchronize(stream[slot].get()) != cudaSuccess) return false;

    const int hit_count = *h_hit_count[slot].get();
    if (hit_count <= 0) return true;

    // The kernel bounds-checks every slot write (slot < kMaxHitSlots), so there
    // is never an out-of-bounds device write. If more than kMaxHitSlots
    // candidates hit in one launch the surplus is simply not recorded; we read
    // the first n and surface the overflow. Those extra winners are re-found
    // next epoch (fresh a0), so none are lost for good -- but a persistent
    // overflow would signal a pattern far denser than expected, hence the
    // accounting.
    if (hit_count > kMaxHitSlots) ++hit_overflow_events;
    const int n = hit_count < kMaxHitSlots ? hit_count : kMaxHitSlots;
    // Synchronous copy of exactly the n recorded slots into this slot's pinned
    // host memory; the prior stream sync guarantees the kernel is done.
    if (cudaMemcpy(h_slots[slot].get(), d_slots[slot].get(), sizeof(HitSlot) * n,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    std::span<const std::byte, 32> a0{slot_a0[slot]};
    for (int i = 0; i < n; ++i) {
        const HitSlot& hit = h_slots[slot].get()[i];
        // candidate index = t + j*T (this epoch's a0).
        const std::uint64_t idx =
            static_cast<std::uint64_t>(hit.t) +
            static_cast<std::uint64_t>(hit.j) * static_cast<std::uint64_t>(T);
        auto scalar = onion::crypto::scalar_add_8i(a0, idx);
        auto pk = onion::crypto::pubkey_from_scalar(scalar);
        if (!pk) {            // degenerate scalar: skip (firewall safe)
            ++dropped_degenerate;
            continue;
        }
        onion::engine::MatchCandidate cand;
        cand.secret.scalar = scalar;
        onion::crypto::random_bytes(cand.secret.prf_prefix);  // fresh RH
        cand.claimed_pubkey = *pk;
        cand.pattern_index = static_cast<std::size_t>(hit.pattern_index);
        results.push(std::move(cand));
    }
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

    // One epoch == one kernel launch over a fresh, fully-disjoint candidate
    // space. The device walks each thread t from its start point P_t = A0+t*8B
    // for M=kStepsPerThread steps of BIGSTEP=T*8B, so thread t, step j visits
    // candidate index exactly  t + j*T  (covering [0, T*M) with no gaps/overlap
    // within the epoch). Re-seeding a0 each epoch draws a brand-new random
    // T*M-sized slab of the keyspace, so epochs never recheck the same key.
    //
    // SOFTWARE PIPELINE (double-buffered, kPipelineDepth==2): the host setup of
    // epoch N (T cheap point-adds, ~12 ms on this box) plus its H2D upload runs
    // on slot N%2's own stream while epoch N-1's ~300 ms kernel is still in
    // flight on the OTHER slot's stream. Because each in-flight epoch has its
    // own device + pinned-host buffers and its own stream, N's host writes never
    // race N-1's kernel reads, and the deferred per-slot sync replaces the old
    // immediate sync. This hides the host setup (previously serial GPU-idle time
    // between epochs) behind the kernel. With kPipelineDepth==1 (compile
    // -DONION_CUDA_NO_PIPELINE) it degrades to the original serial single-buffer
    // loop. Correctness is identical to the serial path -- no field math, kernel,
    // or candidate-index arithmetic changed -- so the libsodium device-chain
    // xval is unaffected.
    //
    // Helper: draw a fresh clamped base scalar and run epoch setup+launch on the
    // given slot. Returns false on any CUDA error (slot left un-launched).
    auto start_epoch = [&](int slot) -> bool {
        std::array<std::byte, 32> a0;
        onion::crypto::random_bytes(a0);
        a0[0] = std::byte(std::to_integer<unsigned>(a0[0]) & 0xf8);
        a0[31] = std::byte((std::to_integer<unsigned>(a0[31]) & 0x7f) | 0x40);
        if (!s.setup_epoch(slot, a0)) return false;
        return s.launch_epoch(slot);
    };

    // Prime the pipeline: launch the first epoch (slot 0) so a kernel is in
    // flight before we begin overlapping the next epoch's host setup with it.
    int cur = 0;
    bool cur_live = !stop.stop_requested() && start_epoch(cur);
    if (!cur_live && !stop.stop_requested()) ++s.dropped_epoch_errors;

    while (cur_live && !stop.stop_requested()) {
        // While slot `cur`'s kernel runs on its stream, build the NEXT epoch on
        // the other slot. Its host point-adds (CPU) and H2D overlap cur's
        // kernel; its kernel is enqueued on the other stream and starts as the
        // GPU frees up. This is the overlap that hides the host setup time.
        const int nxt = (cur + 1) % Impl::kPipelineDepth;
        bool nxt_live = !stop.stop_requested() && start_epoch(nxt);
        if (!nxt_live && !stop.stop_requested()) ++s.dropped_epoch_errors;

        // Now wait on cur's epoch and harvest its hits. (For kPipelineDepth==1,
        // nxt==cur, so this drains the epoch just launched above -- the serial
        // path, identical ordering to the original loop.)
        if (!s.drain_epoch(cur)) ++s.dropped_epoch_errors;

        total += per_launch;
        s.stats.set(0, total);

        cur = nxt;
        cur_live = nxt_live;
    }

    // Drain any epoch still in flight before the RAII members free the buffers
    // its streams reference. cur's was already drained on a clean loop exit; on
    // a stop_request mid-flight there may be one outstanding -- sync every slot
    // defensively (cudaStreamSynchronize on idle/empty streams is a no-op).
    for (int slot = 0; slot < Impl::kPipelineDepth; ++slot)
        cudaStreamSynchronize(s.stream[slot].get());

    // One-line summary ONLY on a real, actionable problem (a CUDA error dropped an
    // epoch, or a degenerate scalar). Hit-slot overflow is benign and expected for
    // short prefixes (surplus hits are simply re-found next epoch), so it never
    // prints on its own — that was just noise on normal runs.
    if (s.dropped_epoch_errors || s.dropped_degenerate) {
        std::fprintf(stderr,
                     "CudaEngine: shutdown stats -- dropped epochs (CUDA errors): "
                     "%llu, degenerate scalars skipped: %llu, hit-slot overflow "
                     "launches: %llu\n",
                     static_cast<unsigned long long>(s.dropped_epoch_errors),
                     static_cast<unsigned long long>(s.dropped_degenerate),
                     static_cast<unsigned long long>(s.hit_overflow_events));
    }
}

}  // namespace onion::cuda
