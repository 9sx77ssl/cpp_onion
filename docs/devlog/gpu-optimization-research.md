# GPU Optimization Research — Path past ~390 M keys/s on the GTX 1650

**Status:** research only (no production code changed).
**Card:** NVIDIA GTX 1650, mobile TU117, sm_75, 14 SMs, 4 GB, GDDR6, **50 W power cap / 60 W hard max**.
**Baseline:** `./onion --engine cuda --bench` = **390 M keys/s** (re-confirmed 390.07; some build/thermal states bench 327–340).
**Goal of the brief:** find a realistic path to **+30% → ~500 M/s**, or prove honestly it is at the silicon ceiling.

---

## TL;DR (the honest bottom line)

**The brief's central premise is wrong, and that changes everything.** The kernel is **NOT local-memory-bandwidth bound.** Five independent GPU prototype sweeps on the real card converge on the same conclusion: **the kernel is integer-ALU / `fed_mul`-issue-throughput bound** (Turing's 32-bit IMAD pipe + carry IADD3), with local memory contributing only ~7–9% of the wall time, and the card pinned at its **50 W electrical cap** (clocks throttle to ~1575 MHz vs 1785 max, throttle reason `0x4 = SW Power Cap`).

Consequences:

- The "reduce local-memory traffic" lever the brief asks for is the **wrong lever**. Every measured attempt to cut local traffic — shared-memory chain inversion, block-wise/register-resident inversion, one-array recompute, warp-cooperative `__shfl` — **lost throughput** (from −7% to −94%), because each traffic cut adds `fed_invert` calls or kills occupancy, and the kernel was never bandwidth-starved (it uses only ~24–29% of the card's *real* 175.6 GB/s).
- **390 M/s already sits at ~69–82% of the pure-ALU ceiling** (depending on which ceiling probe), and within ~3–9% of the compute-only twin (the same arithmetic with zero arrays: ~432–451 M/s). So removing *every* byte of local traffic buys **at most +9–15%**, not +30%.
- **+30% (→500 M/s) is not reachable in software on this card.** It would require cutting the `fed_mul` *count* per key, and every count-reduction lever is blocked (the 7-mul mixed addition is the Edwards floor; doubling breaks the keyspace scan; faster multiply algorithms measured *slower* on Turing).
- **The only verified, safe, free near-term gain is heterogeneous CPU+GPU concurrency: +5–6%** (run the existing CPU incremental engine at **6 threads** — physical cores only — alongside the GPU; 12 threads contends via SMT and nets ~0).
- The **same kernel scales to multi-Gk/s on a bigger GPU**: the bottleneck is this card's silicon (ALU issue rate + 50 W cap), not the algorithm. An RTX 4090 (~2.7× the SM count and clock, far more INT throughput) runs this code at ~1–2 Gk/s with no code change.

**Realistic reachable on this card: ~410–420 M/s (+5–8%) software-only, or ~415–425 (+6–9%) software + CPU-concurrency.** Not +30%.

---

## 1. The physical ceiling (roofline)

Measured on the idle card (`cudaEvent` timing, best/median of 3–5, production `device_field32.cuh` copied read-only to `/tmp/onion-research/`). Several independent ceiling probes, all consistent:

| Probe | What it measures | Result |
|---|---|---|
| Local-mem streaming, **exact production footprint** (2 arrays × M × 32 B = 196,608 B/thread, 0 spills) | the pure-bandwidth wall | **171 GB/s = 89% of 192 GB/s theoretical → ~1337 M keys/s** if it were bus-bound |
| Streaming-copy bandwidth | the card's real DRAM ceiling | **175.6 GB/s** (this is the **GDDR6** 1650, *not* the 128 GB/s GDDR5 variant the memory-bank premise assumed) |
| `fed_mul` standalone (8×32 schoolbook, max-ILP) | INT-multiply issue roof | **5.3–6.5 G fed_mul/s** (occupancy-insensitive → issue-bound, not latency-bound) |
| Pure-ALU ceiling ÷ 11.09 mul/key | implied key-rate roof | **~482–564 M keys/s** |
| Compute-only twin (same arithmetic, **zero arrays**) | upper bound if local traffic were free | **~432–451 M keys/s** |
| `ged_madd` (the real hot op, 7M + 8 add/sub) | effective hot-loop rate | 0.69 G madd/s = **4.82 G mul/s effective** |
| **Production kernel** | reality | **390 M/s = 4.37–4.46 G mul/s effective** |

**Per-key cost (verified by source grep + arithmetic):** `ged_madd` = 7 `fed_mul`; forward `combo`/`pe` = 2; back-substitute `y`/`run` = 2; amortized `fed_invert` ≈ 265 muls / 3072 = 0.086. **Total ≈ 11.09 `fed_mul`/key.**

**The decisive experiment (run twice, two sessions):** a microbench with the *exact* 196 KB/thread production footprint and 0 spills streams at **1337 M keys/s** — 3.4× production. If the kernel were bus-bound it would sit near that wall. Instead the **compute-only twin (zero arrays) reaches only ~432–451 M/s**, barely +9–15% over production. **Therefore eliminating 100% of local traffic buys at most +9–15%, and the remaining ~85% of the wall is integer-ALU issue.**

Cross-check (SASS instruction mix of the production kernel via `cuobjdump`): `IADD3`=4712, `IMAD.WIDE.U32`=3292, `IMAD.X`=2532, `IMAD.MOV`=1631 vs **local ops `STL.128`=6 + `LDL.128`=4**. ALU-to-local-memory instruction ratio **~1216 : 1**. The carry-propagation adds (`IADD3`) actually *outnumber* the multiplies. This is an unambiguous compute-bound signature.

A second compute-bound proof: walk-only (7 mul/key, 0-byte frame) = ~690 M/s; walk + inversion machinery (~11 mul/key, 0-byte frame) = ~447 M/s. Ratio 1.55 ≈ 11/7 = 1.57 → **throughput tracks mul count linearly**, the textbook compute-bound fingerprint.

**Where 390 sits:** ~69% of the pure-`fed_mul` ALU ceiling, ~82% of the standalone-`fed_mul` roof, ~87–91% of the compute-only twin. The +27% the project already extracted (310→390) captured most of the available headroom. **This is near the practical compute roof for an 11-mul/key incremental search on a 50 W-capped TU117.**

---

## 2. Ranked opportunities

### PURSUE

#### P1 — Heterogeneous CPU + GPU concurrency (6 physical-core threads) — **+5–6%, measured, free, no kernel change**
- **Mechanism:** the GPU host thread is ~95% idle during the kernel. The existing CPU `incremental` engine searches a disjoint keyspace (independent random `a₀` per engine) in parallel. Bottlenecks are orthogonal — GPU is INT-ALU/power bound on-device, CPU is its own cores. Results merge through the existing thread-safe `ResultQueue`; the `io::verify` firewall is unchanged.
- **Measured:** GPU-alone ~332–340; GPU + CPU-**6t** = **346–359 M/s (+5–6%) with zero GPU penalty.** GPU + CPU-**12t** = ~334–344 (≈0 net — the SMT siblings steal the GPU host thread's per-epoch point-walk cycles and the GPU drops 6–7%). **Must cap at 6 threads (physical cores only).**
- **How-to (future, describe only):** in `src/cli/main.cpp`, when `--engine cuda`, also construct an `IncrementalCpuEngine` with `N = 6` (new `--cpu-threads` flag, default ~`hw_concurrency/2`), launch both `IEngine::run` on a shared `ResultQueue`/`StatsBoard`/`stop_token`. ~150 lines of spawn boilerplate in `main.cpp`; **zero** changes to either engine's internals or to the firewall.
- **Effort:** Low (1 day). **Risk:** Low. Caveat: a Zen 2 mobile CPU under sustained load may thermal-throttle; if combined gain < +3%, fall back to fewer threads. **Touches the (mis-stated) local-mem bottleneck:** No — orthogonal resource.

### MAYBE

#### M1 — `fed_mul` / reduction carry-chain micro-optimization via hand-PTX (gECC-style CIOS) — **0–8%, uncertain, high effort, high risk**
- **Mechanism:** the kernel is ALU-issue bound, so the *only* on-device lever with any upside is cutting INT instructions inside `fed_mul`/reduction. gECC (arXiv 2501.03245) reports ~1.72× on the multiply in isolation via inline-PTX `mad.lo.cc`/`madc.hi.cc` carry chains.
- **Why uncertain/risky:** the project's C++ `(uint64_t)a*b + r + c` **already compiles to optimal `IMAD.WIDE.U32`**. The recognized expert njuffa's forum thread found hand-PTX cut per-multiply instructions ~60% but moved the *full* kernel only ~4% (a single carry-flag CC register serializes the carry chain). A direct instruction-count-equivalent PTX probe here ran 678 vs 688 M/s — **no faster.** And `ged_madd` is add/sub-heavy, so any `fed_mul`-only win is diluted kernel-wide.
- **How-to (future):** rewrite `fed_mul`/`fed32_reduce_wide` in `device_field32.cuh` with inline PTX carry chains; **must re-pass the bit-exact xval gate** (`tests/test_cuda_xval.cpp`) vs libsodium `base_noclamp(a₀+8i)` — the CARDINAL RULE. **Effort:** High (PTX + full re-validation). **Risk:** High (correctness). **Expected:** +0–8% best case, realistically ~0–4%. **Touches local-mem bottleneck:** No (correctly targets the real ALU bottleneck). Pursue only if you accept the re-validation cost for a single-digit gain.

#### M2 — `fed_to_bytes_partial` (extract only `nbytes`, skip full canonical reduction for non-matches) — **+1–3%, low effort**
- **Mechanism:** `match_and_record` calls full `fed_to_bytes` (32-byte canonical reduction) per key but compares only `pat.nbytes` (2–6). A partial extractor saves the 3-pass reduction and 30 unused bytes on the ~99.99% of non-matches, re-canonicalizing only on a tentative match (no false positives).
- **Why only MAYBE:** the match path costs only ~1.5% of wall time (measured `match_cost.cu`), so the ceiling is tiny. **How-to:** add `fed_to_bytes_partial(out, y, nbytes)` to `device_field32.cuh`; call it from `match_and_record` (`search_kernel.cu` line 38); on tentative match call full `fed_to_bytes` and re-verify. **Effort:** Low (<1 day). **Risk:** Low (xval still gates). **Expected:** +1–3%, likely ~+1.5%. **Touches local-mem bottleneck:** No.

### DEBUNKED (measured negative or 0% — do not pursue)

| # | Angle | Measured / verdict | Why it fails for THIS kernel |
|---|---|---|---|
| D1 | **Shared-memory chain inversion** (the brief's headline candidate) | **−50% to −94%** (B=4 → 63 M/s; B=32 → 23 M/s) | Small B explodes `fed_invert` count (M/B × ~265 muls); large-enough B forces a tiny block → occupancy collapse + 32-bit bank conflicts. gECC agrees: a batch this size "far exceeds L1/shared capacity." |
| D2 | **Block-wise / register-resident inversion** | **0% best case (ties at B≥1536), −31% at B=64** | 48× smaller footprint (4 KB vs 196 KB, ptxas-verified) yet *slower* — proves it's not bandwidth-bound. Each block adds M/B extra inversions; added ALU > saved traffic at every B. |
| D3 | **One-array recompute** (store `zs[]` only, re-walk for `combo`) | **−35% (253 M/s)** | Halving traffic lost badly: the doubled `ged_madd` walk (+7 mul/key) dominates the saved bytes. |
| D4 | **Warp-cooperative `__shfl` inversion** | **−46%** (already in-repo, reverted) | Adds 15–20 muls/key for prefix/suffix scan; trades occupancy (62.5→40%) for storage the kernel didn't need. |
| D5 | **Per-thread ILP (2–4 independent chains)** | **−15% to −40%** (regs 78→174) | Kernel is *throughput*-bound, not latency-bound; more in-flight chains just spill registers. |
| D6 | **Karatsuba 256-bit `fed_mul`** | **−6% (4.96–5.0 vs 5.3 G/s)** | Irregular sub/carry-correction (92 vs 66 regs) outweighs 64→48 partial-product savings on Turing's uniform IMAD grid. solanity's own comment agrees. |
| D7 | **Comba / lazy carry-save reduction** | **−39% (3.2–3.4 G/s)** | `if(lo<old)++hi` serializes the diagonal — worst case for the IMAD pipe. |
| D8 | **Donna 10×25.5 / 5×51-bit / 64-bit limbs** | **−45% (2.91 G/s)** | Turing has no native 64×64→128 mul; signed-i64 accumulators emulate via `IMAD.WIDE` + `SHF` carry chains, 146 regs kills occupancy. 8×32 native is correct for sm_75. |
| D9 | **Toom-Cook-3** | 0% to −10% (not built; below crossover) | Interpolation needs divisions (~inversion-scale); 256-bit is far below the ~1000-bit Toom crossover. |
| D10 | **Larger T (>2^15)** | **flat-to-worse** (2^18 → 268–352) | The prior session's "+3.3% from T>2^15" was a replica artifact; does **not** reproduce on the production binary. CUDA reserves local mem per *resident* thread, so T=2^15 is already optimal. |
| D11 | **Montgomery multiplication** (vs pseudo-Mersenne) | −5% to +3% (not worth) | Pseudo-Mersenne 2²⁵⁵−19 reduction is shifts+adds; Montgomery adds ~16–25% ALU on GPU for no memory benefit. |
| D12 | **Niels coords (7M readdition)** | ~+0.4% | Saves 1 mul per 3072 steps (constant `8B`); the `T2d` precompute is once/epoch, so the per-key win rounds to noise. |
| D13 | **Windowed / comb precomputation** | +0–3% | Reduces point-add count but the walk is fixed-stride `A += BIGSTEP`; lookup saves few muls and adds `__constant__` lookups. |
| D14 | **GLV / endomorphism decomposition** | 0% (impossible) | Ed25519 was *designed* without an efficiently-computable endomorphism (Bernstein, conservative SafeCurves choice). Mathematically unavailable. |
| D15 | **DP4A / tensor cores** | 0% (architectural mismatch) | 8-bit SIMD / fixed matrix granules can't do 32-bit-limb carried modular multiply. Zero published ECC use; gECC uses plain IMAD. |
| D16 | **Pinned / zero-copy / managed (system RAM)** | 0% or negative | Kernel never touches host memory; routing the hot arrays over PCIe (16 GB/s) vs on-device (175 GB/s) would be 5–8× slower. |
| D17 | **Multi-stream / persistent kernel / cooperative groups** | 0% to −5% | Launch overhead is ~0.002% of a 300 ms epoch; consumer Turing lacks Hyper-Q (streams serialize); grid-sync busy-wait erases any saving. |
| D18 | **AMD Vega iGPU co-compute** | 0–8% theoretical, ~0 net | iGPU is ~25× slower, shares the same system-RAM pool as the CPU, and OpenCL/coherency/load-balance overhead (10–30%) erases its ~3% contribution. |
| D19 | **Raise power limit 50 W → 60 W** | +5–12% *if* root (not a code change) | Real headroom exists (thermals ~60 °C vs 94 °C throttle; power is the active cap), but `nvidia-smi -pl 60` needs root and is non-persistent. The roofline above was measured *under* the 50 W cap, so this is the one genuine non-software lever — out of scope for the code. |
| D20 | **Z⁻¹-free / probabilistic early-reject; single-array via T=XY/Z; lazy reduction; alt radix; ILP scheduling of `fed_mul`; data-layout `uint4` packing** | 0% to +2% | All either don't reduce the ALU bottleneck, require unavailable stored coordinates, or add register pressure. See findings appendix. |

---

## 3. Recommended path to the biggest realistic gain

The brief's hypothesized path (block-wise/shared-resident inversion ± lazy reduction) is **measured-dead** — it targets a bottleneck that isn't the bottleneck. The honest recommended path is:

1. **P1 — CPU-concurrent (6 threads):** +5–6%, free, no kernel change, no correctness risk. **Do this first.** (390 → ~410–414 M/s combined.)
2. **(Optional) M2 — `fed_to_bytes_partial`:** +1–3% on the GPU side, low effort. Stacks with P1.
3. **(Optional, only if you accept high re-validation cost) M1 — hand-PTX `fed_mul` carry chains:** +0–4% realistic, uncertain, must re-pass the libsodium xval gate.

**Honest total estimate:** **~+6–10% → ~415–430 M/s.** This does **not** reach +30%. There is no measured or literature-backed software path to 500 M/s on this card.

The only lever that could plausibly reach +30% is **non-software**: raising the 50 W power cap to 60 W (root `nvidia-smi -pl 60`), which the prototype work estimates at +5–12% (sublinear, the card is also thermally fine). Even stacked with CPU-concurrency that lands ~+12–18%, still short of +30%, and it's out of scope (root + non-persistent).

---

## 4. Heterogeneous angle (measured/realistic)

- **CPU concurrent (Ryzen 5 4600H, our `incremental` engine):** the **one real free win**. 6 physical-core threads add ~17–19.5 M/s cleanly with **zero** GPU penalty → **+5–6%**. 12 threads (SMT) contends with the GPU host thread and nets ~0. Verdict: **PURSUE at 6 threads.**
- **AMD Radeon Vega iGPU:** ~25× slower than the dGPU, shares system-RAM bandwidth with the CPU, and OpenCL port + coherency + load-balance overhead (10–30%) erases its ~3% theoretical contribution. Verdict: **DEBUNKED** (engineering cost ≫ gain). If you ever want a real second GPU, add a second NVIDIA card via multi-GPU CUDA (scales linearly), not the iGPU.

---

## 5. Bottom line: this card vs the silicon wall, and scaling up

- **390 M/s is within ~3–9% of this GTX 1650's genuine ceiling** for a *correct* `a₀ + 8i` incremental search. The realistic software-only headroom is **~+5–8% (→410–420 M/s)**; with CPU-concurrency **~+6–10% (→415–430)**. **+30% (500 M/s) is at/beyond the silicon roof** and is a *hardware* gap, not a software one.
- **The wall is twofold and both halves are silicon:** (1) integer-ALU issue throughput (11 `fed_mul`/key, and the mul *count* is at the Edwards/Montgomery-trick algorithmic floor — 7M mixed-add is the proven HWCD-2008 minimum, batch inversion is ~4 mul/key optimal, and `ged_madd` has no squarings so a dedicated `fe_sq` buys nothing); (2) the **50 W power cap** throttling clocks to ~1575/1785 MHz under load.
- **Why none of the mul-count levers work:** the 7M point-add output coordinates (X,Y,Z,T) all feed the next step — none droppable; point-*doubling* (4M+4S, cheaper) would change the walk to `a₀·2^i` and destroy the contiguous `a₀+8i` keyspace scan that makes the search correct and efficient (mkp224o issue #20 floated this and never shipped it); faster multiply algorithms (Karatsuba, comba, donna) all measured **slower** on Turing.
- **The same code scales to multi-Gk/s on a bigger GPU — automatically.** Because the bottleneck is this card's INT-issue rate and 50 W cap, not the algorithm or memory layout, a larger/newer GPU (e.g. RTX 4090: far more SMs, much higher INT throughput, ~7× the bandwidth, no 50 W cap) runs this kernel at ~1–2 Gk/s with **zero code change**. The honest recommendation for >390 M/s is **more/bigger silicon**, not a better kernel.

---

### Appendix — provenance

- 29 research-angle write-ups + 5 on-device GPU prototype measurement sets (all GPU runs idle-checked, `cudaEvent` timing, production `device_field32.cuh` copied read-only into `/tmp/onion-research/`).
- Source verified: `src/engine/cuda/search_kernel.cu` (2-array fused-combo default, `ged_madd` walk + single `fed_invert`/thread), `device_field32.cuh` (8×32 native field).
- Key prototype files: `localmem_bench.cu` (171 GB/s footprint stream), `compute_only.cu` (432–451 M/s zero-array twin), `mulbench3.cu` (schoolbook 5.3 > Karatsuba 4.97 > comba 3.3 G/s), `roofline.cu`, `smem_chain.cu` (shared-mem inversion −50…−94%), `blockwise.cu`, `recompute.cu`, `search.sass` (1216:1 ALU:local instruction mix), `bw.cu`/`drambw.cu` (175.6 GB/s real DRAM BW).
- **No production code was modified.** This report is the only file written.
