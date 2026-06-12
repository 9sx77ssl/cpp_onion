#pragma once

#include "crypto/keys.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <new>
#include <optional>
#include <stop_token>
#include <vector>

namespace onion::engine {

// What an engine reports on a hit. NOT yet a result: the Verifier (onion_io)
// independently re-derives the pubkey before anything reaches disk
// (design doc §1 "Match != result").
struct MatchCandidate {
    crypto::ExpandedSecretKey secret;
    std::array<std::byte, 32> claimed_pubkey{};
    std::size_t pattern_index = 0;
};

// Single-writer-per-slot throughput counters. Workers own plain uint64_t
// slots (padded against false sharing); cross-thread access goes through
// relaxed atomic_ref. No RMW anywhere (design doc §11).
class StatsBoard {
public:
    explicit StatsBoard(unsigned workers) : slots_(workers) {}

    void set(unsigned worker, std::uint64_t total) noexcept {
        std::atomic_ref<std::uint64_t>(slots_[worker].value)
            .store(total, std::memory_order_relaxed);
    }

    [[nodiscard]] std::uint64_t total() const noexcept {
        std::uint64_t sum = 0;
        for (auto& slot : slots_)  // slots_ is mutable: no const_cast needed
            sum += std::atomic_ref<std::uint64_t>(slot.value)
                       .load(std::memory_order_relaxed);
        return sum;
    }

private:
    // Cache-line padded so each worker's counter sits on its own line
    // (no false sharing). The -Winterference-size ABI caveat does not apply:
    // this is one binary with one set of flags, never shared across DSOs.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winterference-size"
#endif
    struct alignas(std::hardware_destructive_interference_size) Slot {
        std::uint64_t value = 0;
    };
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
    mutable std::vector<Slot> slots_;
};

// Mutex+condvar MPSC queue. Matches are rare (minutes-to-hours apart);
// lock-free here is deliberately declined (design doc §12).
class ResultQueue {
public:
    void push(MatchCandidate c) {
        {
            std::lock_guard lock(mutex_);
            queue_.push_back(std::move(c));
        }
        cv_.notify_one();
    }

    [[nodiscard]] std::optional<MatchCandidate>
    pop_wait_for(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); }))
            return std::nullopt;
        MatchCandidate c = std::move(queue_.front());
        queue_.pop_front();
        return c;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<MatchCandidate> queue_;
};

// Engine boundary: called once per run; virtual dispatch is fine here
// (design doc §1). run() blocks until the stop_token fires.
class IEngine {
public:
    virtual ~IEngine() = default;
    virtual void run(std::stop_token stop) = 0;
};

}  // namespace onion::engine
