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
#include <format>
#include <limits>
#include <memory>
#include <print>
#include <string>
#include <stop_token>
#include <thread>
#include <vector>

namespace {
volatile std::sig_atomic_t g_interrupted = 0;
extern "C" void on_sigint(int) { g_interrupted = 1; }

// Humanized count: 12345 -> "12.3K", 1500000 -> "1.5M", 2.0e9 -> "2.0G".
std::string humanize_count(double n) {
    static const char* units[] = {"", "K", "M", "G", "T", "P"};
    int u = 0;
    while (n >= 1000.0 && u < 5) {
        n /= 1000.0;
        ++u;
    }
    if (u == 0) return std::format("{}", static_cast<std::uint64_t>(n));
    return std::format("{:.1f}{}", n, units[u]);
}

// Humanized rate in keys/s: "25 M/s", "1.2 G/s", "302 M/s".
std::string humanize_rate(double per_sec) {
    static const char* units[] = {"/s", " K/s", " M/s", " G/s", " T/s", " P/s"};
    int u = 0;
    while (per_sec >= 1000.0 && u < 5) {
        per_sec /= 1000.0;
        ++u;
    }
    if (per_sec >= 100.0 || u == 0) return std::format("{:.0f}{}", per_sec, units[u]);
    return std::format("{:.1f}{}", per_sec, units[u]);
}
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
    app.add_option("-n,--count", count,
                   "keys to find before exiting; 0 = run forever (Ctrl+C to stop)");
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
    const std::string target = (count == 0) ? "∞" : std::format("{}", count);

    // Minimalist startup: banner (the only ^.^) + a single factual status line.
    // Suppressed entirely under --quiet, and the status line is omitted in --bench.
    if (!quiet) {
        std::println("cpp_onion ^.^");
        if (bench_seconds <= 0.0) {
            std::string prefix_list;
            for (std::size_t i = 0; i < prefixes.size(); ++i) {
                if (i) prefix_list += ", ";
                prefix_list += prefixes[i];
            }
            const std::string engine_label =
                (engine_name == "cuda") ? "CUDA" : std::format("{} threads", threads);
            std::println("{} · {} · target {} · ~{:.3g} tries/match", prefix_list,
                         engine_label, target, expected_tries);
        }
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
    const bool infinite = (count == 0);

    while ((infinite || found < count) && !g_interrupted) {
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
            if (!quiet) std::print("\r");  // clear in-progress line before the found line
            std::println("  found  {}  →  {}", verified->address.to_string(),
                         dir->string());
            ++found;
        }

        if (!quiet) {
            const auto elapsed = std::chrono::duration<double>(clock::now() - start).count();
            const auto total = stats.total();
            const double rate = elapsed > 0 ? double(total) / elapsed : 0.0;
            std::print("\r{} · {} tried · {:.0f}s · {}/{}   ", humanize_rate(rate),
                       humanize_count(double(total)), elapsed, found, target);
            std::fflush(stdout);
        }
    }

    if (!quiet) std::println("");
    stop.request_stop();
    return exit_code;
}
