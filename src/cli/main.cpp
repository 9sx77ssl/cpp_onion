#include "core/matcher.hpp"
#include "engine/cpu/incremental_engine.hpp"
#include "engine/cpu/incremental_engine_x4.hpp"
#include "engine/cpu/naive_engine.hpp"
#include "io/tor_key_writer.hpp"
#include "io/verifier.hpp"

#ifdef ONION_CUDA
#include "engine/cuda/cuda_engine.hpp"
#endif

#include <CLI/CLI.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <print>
#include <stop_token>
#include <thread>
#include <vector>

namespace {
volatile std::sig_atomic_t g_interrupted = 0;
extern "C" void on_sigint(int) { g_interrupted = 1; }
}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"cpp_onion - Tor v3 vanity onion address generator (Phase 0 reference engine)"};

    std::vector<std::string> prefixes;
    std::filesystem::path outdir = ".";
    unsigned threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    std::size_t count = 1;
    bool quiet = false;

    app.add_option("prefix", prefixes, "base32 prefix(es) to search for (a-z, 2-7)")
        ->required();
    app.add_option("-o,--out", outdir, "output directory");
    app.add_option("-t,--threads", threads, "worker threads");
    app.add_option("-n,--count", count, "number of keys to find before exiting");
    app.add_flag("-q,--quiet", quiet, "suppress progress output");
    std::string engine_name = "incremental";
    std::string simd_mode = "auto";  // auto -> fastest on this CPU (scalar on Zen 2); on -> force AVX2 x4
    double bench_seconds = 0.0;
    std::size_t batch = 1024;
    app.add_option("--engine", engine_name,
                   "engine: incremental (default), naive, or cuda (GPU; requires CUDA build)");
    app.add_option("--simd", simd_mode, "AVX2 4-wide engine: on | off | auto (default). NOTE: 4-wide is slower than scalar on Zen 2 (2x128 AVX2 + register spills), so auto uses the scalar engine here; --simd on forces x4 (wins on Zen 4+/Intel AVX-512-class cores)");
    app.add_option("--bench", bench_seconds, "benchmark: run N seconds against an impossible prefix, report keys/s");
    app.add_option("--batch", batch, "incremental engine batch size (default 1024)");
    CLI11_PARSE(app, argc, argv);
    threads = std::max(1u, threads);

    std::vector<onion::core::CompiledPattern> patterns;
    std::size_t shortest = std::numeric_limits<std::size_t>::max();
    for (const auto& p : prefixes) {
        auto compiled = onion::core::compile_prefix(p);
        if (!compiled) {
            std::println(stderr, "error: invalid prefix '{}' (allowed: a-z 2-7, length 1-49)", p);
            return 1;
        }
        shortest = std::min(shortest, p.size());
        patterns.push_back(std::move(*compiled));
    }
    const double expected_tries = std::pow(32.0, double(shortest));
    if (!quiet) {
        if (engine_name == "cuda")
            std::println("searching {} pattern(s) on GPU (CUDA — tens of thousands of device "
                         "threads; --threads does not apply), ~{:.3g} expected candidates per match",
                         patterns.size(), expected_tries);
        else
            std::println("searching {} pattern(s), {} threads, ~{:.3g} expected candidates per match",
                         patterns.size(), threads, expected_tries);
    }

    onion::engine::ResultQueue queue;
    onion::engine::StatsBoard stats(threads);
    std::unique_ptr<onion::engine::IEngine> engine;
    // Determine which incremental variant to use.
    // --engine naive always wins; otherwise only --simd on forces the x4 engine.
    // auto/off use the scalar engine because 4-wide AVX2 measured SLOWER on Zen 2.
    const bool use_x4 = (engine_name != "naive") && (engine_name != "cuda") && (simd_mode == "on");
    if (engine_name == "cuda") {
#ifdef ONION_CUDA
        engine = std::make_unique<onion::cuda::CudaEngine>(patterns, queue, stats);
#else
        std::println(stderr,
                     "error: --engine cuda requires a CUDA build "
                     "(configure with -DONION_CUDA=ON, e.g. cmake --preset cuda)");
        return 1;
#endif
    } else if (engine_name == "naive")
        engine = std::make_unique<onion::engine::NaiveCpuEngine>(patterns, threads, queue, stats);
    else if (use_x4)
        engine = std::make_unique<onion::engine::IncrementalCpuEngineX4>(patterns, threads, queue, stats, batch);
    else
        engine = std::make_unique<onion::engine::IncrementalCpuEngine>(patterns, threads, queue, stats, batch);

    std::signal(SIGINT, on_sigint);
    std::stop_source stop;
    std::jthread runner([&] { engine->run(stop.get_token()); });

    using clock = std::chrono::steady_clock;
    const auto start = clock::now();

    if (bench_seconds > 0.0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(bench_seconds));
        const double secs = std::chrono::duration<double>(clock::now() - start).count();
        const double mkeys = double(stats.total()) / secs / 1e6;
        if (engine_name == "cuda")
            std::println("bench: {:.2f} M keys/s ({} candidates in {:.1f}s, GPU/CUDA)",
                         mkeys, stats.total(), secs);
        else {
            const std::string variant = (engine_name == "naive") ? "naive"
                                        : use_x4 ? "incremental+avx2x4"
                                                 : "incremental";
            std::println("bench: {:.2f} M keys/s ({} candidates in {:.1f}s, {} threads, {})",
                         mkeys, stats.total(), secs, threads, variant);
        }
        stop.request_stop();
        return 0;
    }

    std::size_t found = 0;
    int exit_code = 0;

    while (found < count && !g_interrupted) {
        auto candidate = queue.pop_wait_for(std::chrono::milliseconds(500));

        if (candidate) {
            const auto verified = onion::io::verify(*candidate, patterns);
            if (!verified) {
                std::println(stderr,
                             "FATAL: verification failed (engine bug, error {})",
                             int(verified.error()));
                exit_code = 2;
                break;
            }
            const auto dir = onion::io::write_tor_keys(*verified, outdir);
            if (!dir) {
                std::println(stderr, "error: failed to write keys (error {})",
                             int(dir.error()));
                exit_code = 3;
                break;
            }
            std::println("found: {}  ->  {}", verified->address.to_string(),
                         dir->string());
            ++found;
        }

        if (!quiet) {
            const auto elapsed = std::chrono::duration<double>(clock::now() - start).count();
            const auto total = stats.total();
            std::print("\r{:.0f} keys/s | {} checked | {:.1f}s elapsed   ",
                       elapsed > 0 ? double(total) / elapsed : 0.0, total, elapsed);
            std::fflush(stdout);
        }
    }

    if (!quiet) std::println("");
    stop.request_stop();
    return exit_code;
}
