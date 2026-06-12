#include <catch2/catch_test_macros.hpp>
#include "engine/cpu/naive_engine.hpp"
#include "core/matcher.hpp"
#include "crypto/keys.hpp"

#include <chrono>
#include <stop_token>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

TEST_CASE("naive engine finds a single-char prefix and reports a true candidate") {
    using namespace onion;
    auto pat = core::compile_prefix("a");
    REQUIRE(pat.has_value());
    std::vector<core::CompiledPattern> patterns{*pat};

    engine::ResultQueue queue;
    engine::StatsBoard stats(2);
    engine::NaiveCpuEngine eng(patterns, 2, queue, stats);

    std::stop_source stop;
    std::jthread runner([&] { eng.run(stop.get_token()); });

    auto cand = queue.pop_wait_for(30s);  // expected ~32 tries; 30s is paranoia
    stop.request_stop();
    REQUIRE(cand.has_value());

    auto pk = crypto::pubkey_from_scalar(cand->secret.scalar);
    REQUIRE(pk.has_value());
    CHECK(*pk == cand->claimed_pubkey);
    CHECK(cand->pattern_index == 0);
    CHECK(core::matches(patterns[0], *pk));

    runner.join();
    CHECK(stats.total() > 0);
}

TEST_CASE("naive engine honors stop_token promptly") {
    using namespace onion;
    // 12-char prefix: will never match within the test
    auto pat = core::compile_prefix("aaaaaaaaaaaa");
    REQUIRE(pat.has_value());
    std::vector<core::CompiledPattern> patterns{*pat};

    engine::ResultQueue queue;
    engine::StatsBoard stats(2);
    engine::NaiveCpuEngine eng(patterns, 2, queue, stats);

    std::stop_source stop;
    std::jthread runner([&] { eng.run(stop.get_token()); });
    std::this_thread::sleep_for(100ms);

    const auto t0 = std::chrono::steady_clock::now();
    stop.request_stop();
    runner.join();
    CHECK(std::chrono::steady_clock::now() - t0 < 2s);
}
