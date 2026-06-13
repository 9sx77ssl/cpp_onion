// End-to-end CUDA search-engine gate (Phase 3 Task 2). Runs the real
// CudaEngine on the GPU through the IEngine seam, then proves a GPU-found
// candidate survives the io::verify firewall (independent libsodium
// re-derivation) and that the stop_token is honored promptly. Registered with
// ctest only when ONION_CUDA is ON, so CPU-only presets are unaffected.
//
// This complements test_cuda_xval.cpp (device crypto core == libsodium): here
// we exercise the full host loop -- random clamped a0, host point-add setup,
// kernel launch, hit readback, scalar reconstruction, candidate push -- and
// confirm the assembled MatchCandidate is genuinely correct.

#include <catch2/catch_test_macros.hpp>

#include "core/matcher.hpp"
#include "engine/cuda/cuda_engine.hpp"
#include "io/verifier.hpp"

#include <chrono>
#include <stop_token>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

TEST_CASE("CUDA engine finds a 3-char prefix and the candidate verifies") {
    using namespace onion;
    auto pat = core::compile_prefix("abc");
    REQUIRE(pat.has_value());
    std::vector patterns{*pat};

    engine::ResultQueue queue;
    engine::StatsBoard stats(1);  // CudaEngine reports through worker slot 0
    cuda::CudaEngine eng(patterns, queue, stats);

    std::stop_source stop;
    std::jthread runner([&] { eng.run(stop.get_token()); });

    // One epoch scans T*M = 2^16 * 256 ~= 16.7M candidates; a 3-char prefix
    // (~32k expected) is found in the first launch. 60s is a wide safety margin
    // that also tolerates device init / first-launch JIT.
    auto cand = queue.pop_wait_for(60s);
    stop.request_stop();
    REQUIRE(cand.has_value());

    // THE GATE: the firewall must accept a GPU-found candidate. This is an
    // independent libsodium re-derivation of the pubkey from the reconstructed
    // scalar -- if the GPU's hit -> scalar -> pubkey path were wrong, this
    // would reject it.
    const auto verified = io::verify(*cand, patterns);
    REQUIRE(verified.has_value());
    CHECK(verified->address.view().starts_with("abc"));

    runner.join();
    CHECK(stats.total() > 0);
}

TEST_CASE("CUDA engine honors stop_token promptly") {
    using namespace onion;
    // 12-char prefix: ~2^60 expected tries, never matches during the test, so
    // run() stays in its epoch loop until the stop_token fires.
    auto pat = core::compile_prefix("zzzzzzzzzzzz");
    REQUIRE(pat.has_value());
    std::vector patterns{*pat};

    engine::ResultQueue queue;
    engine::StatsBoard stats(1);
    cuda::CudaEngine eng(patterns, queue, stats);

    std::stop_source stop;
    std::jthread runner([&] { eng.run(stop.get_token()); });
    std::this_thread::sleep_for(200ms);  // let at least one epoch begin
    const auto t0 = std::chrono::steady_clock::now();
    stop.request_stop();
    runner.join();
    // The loop checks stop_requested() each epoch; one in-flight epoch must
    // finish first, so allow generous slack for a single kernel launch + sync.
    CHECK(std::chrono::steady_clock::now() - t0 < 10s);
}
