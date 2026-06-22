#include <catch2/catch_test_macros.hpp>

#include "engine/composite_engine.hpp"
#include "engine/cpu/incremental_engine.hpp"
#include "core/matcher.hpp"
#include "io/verifier.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <stop_token>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace {
// Minimal engine that records that it ran and blocks on the shared stop_token,
// exactly like the real engines. Lets us assert the composite started every
// child concurrently and that run() blocks until all of them return.
struct FakeEngine final : onion::engine::IEngine {
    std::atomic<int>& started;
    std::atomic<int>& stopped;
    FakeEngine(std::atomic<int>& s, std::atomic<int>& e) : started(s), stopped(e) {}
    void run(std::stop_token st) override {
        started.fetch_add(1, std::memory_order_relaxed);
        while (!st.stop_requested()) std::this_thread::sleep_for(2ms);
        stopped.fetch_add(1, std::memory_order_relaxed);
    }
};
}  // namespace

TEST_CASE("composite runs every child concurrently and blocks run() until stop") {
    using namespace onion;
    std::atomic<int> started{0}, stopped{0};

    std::vector<std::unique_ptr<engine::IEngine>> children;
    for (int i = 0; i < 3; ++i)
        children.push_back(std::make_unique<FakeEngine>(started, stopped));
    engine::CompositeEngine comp(std::move(children));

    std::stop_source stop;
    std::atomic<bool> run_returned{false};
    std::jthread runner([&] {
        comp.run(stop.get_token());
        run_returned.store(true, std::memory_order_release);
    });

    // All three children must start (run concurrently) within a short window.
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (started.load() < 3 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(2ms);
    CHECK(started.load() == 3);
    // run() must NOT have returned while children are still alive.
    CHECK_FALSE(run_returned.load(std::memory_order_acquire));

    stop.request_stop();
    runner.join();
    CHECK(stopped.load() == 3);              // every child observed the shared stop_token
    CHECK(run_returned.load(std::memory_order_acquire));  // run() returned after all joined
}

TEST_CASE("composite drives a real engine end-to-end through the firewall") {
    using namespace onion;
    auto pat = core::compile_prefix("abc");
    REQUIRE(pat.has_value());
    std::vector patterns{*pat};

    engine::ResultQueue queue;
    engine::StatsBoard stats(4);
    std::vector<std::unique_ptr<engine::IEngine>> children;
    children.push_back(
        std::make_unique<engine::IncrementalCpuEngine>(patterns, 4, queue, stats));
    engine::CompositeEngine comp(std::move(children));

    std::stop_source stop;
    std::jthread runner([&] { comp.run(stop.get_token()); });

    auto cand = queue.pop_wait_for(30s);  // ~32768 expected tries
    stop.request_stop();
    REQUIRE(cand.has_value());

    // The candidate routed through the composite must still pass the firewall
    // (independent libsodium re-derivation) — composing engines changes nothing
    // about the match != result contract.
    const auto verified = io::verify(*cand, patterns);
    REQUIRE(verified.has_value());
    CHECK(verified->address.view().starts_with("abc"));

    runner.join();
    CHECK(stats.total() > 0);
}
