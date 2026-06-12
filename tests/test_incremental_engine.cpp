#include <catch2/catch_test_macros.hpp>
#include "engine/cpu/incremental_engine.hpp"
#include "core/matcher.hpp"
#include "crypto/keys.hpp"
#include "io/verifier.hpp"

#include <chrono>
#include <stop_token>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

TEST_CASE("incremental engine finds a 3-char prefix and the candidate verifies") {
    using namespace onion;
    auto pat = core::compile_prefix("abc");
    REQUIRE(pat.has_value());
    std::vector patterns{*pat};

    engine::ResultQueue queue;
    engine::StatsBoard stats(4);
    engine::IncrementalCpuEngine eng(patterns, 4, queue, stats);

    std::stop_source stop;
    std::jthread runner([&] { eng.run(stop.get_token()); });

    auto cand = queue.pop_wait_for(30s);  // ~32768 expected tries
    stop.request_stop();
    REQUIRE(cand.has_value());

    // the firewall must accept it (independent libsodium re-derivation)
    const auto verified = io::verify(*cand, patterns);
    REQUIRE(verified.has_value());
    CHECK(verified->address.view().starts_with("abc"));

    runner.join();
    CHECK(stats.total() > 0);
}

TEST_CASE("incremental engine honors stop_token promptly") {
    using namespace onion;
    auto pat = core::compile_prefix("zzzzzzzzzzzz");  // 12 chars: never matches in-test
    REQUIRE(pat.has_value());
    std::vector patterns{*pat};
    engine::ResultQueue queue;
    engine::StatsBoard stats(4);
    engine::IncrementalCpuEngine eng(patterns, 4, queue, stats);

    std::stop_source stop;
    std::jthread runner([&] { eng.run(stop.get_token()); });
    std::this_thread::sleep_for(100ms);
    const auto t0 = std::chrono::steady_clock::now();
    stop.request_stop();
    runner.join();
    CHECK(std::chrono::steady_clock::now() - t0 < 2s);
}
