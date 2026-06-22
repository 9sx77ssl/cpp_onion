#pragma once

#include "engine/engine.hpp"

#include <memory>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace onion::engine {

// Runs several engines concurrently (e.g. the CPU incremental engine alongside
// the CUDA engine) for combined throughput. Each child engine seeds its own
// random base scalars, so they search fully disjoint subspaces with no
// coordination — collision probability is cryptographically nil (design §1/§7).
//
// All children share ONE ResultQueue, so every candidate — CPU or GPU — still
// passes the same io::verify firewall before any key is written. They write
// DISJOINT StatsBoard slots (the CPU engine owns [0, threads), the GPU is given
// slot `threads` via CudaKnobs::stats_slot), so StatsBoard::total() is the true
// combined keys/s. A child that dies on its own (e.g. CUDA init failure) just
// stops contributing; the others keep running — the search degrades gracefully
// to whatever backends are live instead of failing.
//
// run() spawns one jthread per child (passing the SHARED parent stop_token, not
// each jthread's own) and blocks at scope exit until all children's run()
// return — which they do when the parent stop_token fires.
class CompositeEngine final : public IEngine {
public:
    explicit CompositeEngine(std::vector<std::unique_ptr<IEngine>> engines)
        : engines_(std::move(engines)) {}

    void run(std::stop_token stop) override {
        std::vector<std::jthread> threads;
        threads.reserve(engines_.size());
        for (auto& e : engines_)
            threads.emplace_back([p = e.get(), stop] { p->run(stop); });
        // jthreads join on destruction here: run() blocks until every child's
        // run() returns (each observes the same `stop`).
    }

private:
    std::vector<std::unique_ptr<IEngine>> engines_;
};

}  // namespace onion::engine
