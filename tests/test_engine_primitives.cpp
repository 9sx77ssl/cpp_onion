#include <catch2/catch_test_macros.hpp>
#include "engine/engine.hpp"

#include <chrono>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

TEST_CASE("StatsBoard sums per-worker totals written from worker threads") {
    onion::engine::StatsBoard stats(4);
    {
        std::vector<std::jthread> workers;
        for (unsigned w = 0; w < 4; ++w)
            workers.emplace_back([&stats, w] {
                for (std::uint64_t total = 0; total <= 1000; total += 100)
                    stats.set(w, total);
            });
    }  // join
    CHECK(stats.total() == 4000);
}

TEST_CASE("ResultQueue delivers pushed candidates") {
    onion::engine::ResultQueue q;
    onion::engine::MatchCandidate c;
    c.pattern_index = 7;
    q.push(c);
    auto got = q.pop_wait_for(100ms);
    REQUIRE(got.has_value());
    CHECK(got->pattern_index == 7);
}

TEST_CASE("ResultQueue pop times out empty") {
    onion::engine::ResultQueue q;
    const auto t0 = std::chrono::steady_clock::now();
    CHECK_FALSE(q.pop_wait_for(50ms).has_value());
    CHECK(std::chrono::steady_clock::now() - t0 >= 50ms);
}
