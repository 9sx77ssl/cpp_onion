#include <catch2/catch_test_macros.hpp>
#include "engine/cpu/incremental_engine_x4.hpp"
#include "core/matcher.hpp"
#include "crypto/keys.hpp"
#include "io/verifier.hpp"

#include <chrono>
#include <stop_token>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

TEST_CASE("x4 engine finds a 3-char prefix and the candidate passes io::verify") {
    using namespace onion;
    auto pat = core::compile_prefix("abc");
    REQUIRE(pat.has_value());
    std::vector patterns{*pat};

    engine::ResultQueue queue;
    engine::StatsBoard stats(4);
    engine::IncrementalCpuEngineX4 eng(patterns, 4, queue, stats);

    std::stop_source stop;
    std::jthread runner([&] { eng.run(stop.get_token()); });

    auto cand = queue.pop_wait_for(30s);  // ~32768 expected tries; x4 lanes are fast
    stop.request_stop();
    REQUIRE(cand.has_value());

    // The verifier firewall: independent libsodium re-derivation must pass.
    const auto verified = io::verify(*cand, patterns);
    REQUIRE(verified.has_value());
    CHECK(verified->address.view().starts_with("abc"));

    runner.join();
    // 4 lanes × batch_ candidates per batch: total must be positive.
    CHECK(stats.total() > 0);
}

TEST_CASE("x4 engine honors stop_token promptly") {
    using namespace onion;
    auto pat = core::compile_prefix("zzzzzzzzzzzz");  // 12 chars: effectively never matches
    REQUIRE(pat.has_value());
    std::vector patterns{*pat};

    engine::ResultQueue queue;
    engine::StatsBoard stats(4);
    engine::IncrementalCpuEngineX4 eng(patterns, 4, queue, stats);

    std::stop_source stop;
    std::jthread runner([&] { eng.run(stop.get_token()); });
    std::this_thread::sleep_for(100ms);
    const auto t0 = std::chrono::steady_clock::now();
    stop.request_stop();
    runner.join();
    CHECK(std::chrono::steady_clock::now() - t0 < 2s);
}
