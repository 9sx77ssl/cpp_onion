#include "core/matcher.hpp"
#include "engine/cpu/naive_engine.hpp"
#include "io/tor_key_writer.hpp"
#include "io/verifier.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
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
    if (!quiet)
        std::println("searching {} pattern(s), {} threads, ~{:.3g} expected candidates per match",
                     patterns.size(), threads, expected_tries);

    onion::engine::ResultQueue queue;
    onion::engine::StatsBoard stats(threads);
    onion::engine::NaiveCpuEngine engine(patterns, threads, queue, stats);

    std::signal(SIGINT, on_sigint);
    std::stop_source stop;
    std::jthread runner([&] { engine.run(stop.get_token()); });

    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
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
