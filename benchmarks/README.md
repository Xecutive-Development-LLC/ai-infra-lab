# Baseline Benchmark

`bench_baseline.py` is Phase A of the roadmap ([`docs/AI_Inference_Server_Build_Log_and_Roadmap.md`](../docs/AI_Inference_Server_Build_Log_and_Roadmap.md) §10): a single-request, controlled benchmark against the vLLM OpenAI-compatible API. It exists to replace the doc's earlier "97–163 tok/s observed in logs" — informal interval readings — with real, repeatable, recorded numbers.

No dependencies beyond the Python 3 standard library.

## What it measures

For each request, streamed via SSE (`stream: true`, `stream_options.include_usage: true`):

- **TTFT** — time to first token (time from request sent to first generated token received)
- **Total latency** — full request wall-clock time
- **Output tokens/sec** — `completion_tokens / total_latency` (includes TTFT, so it's the number a client actually experiences)
- **Decode tokens/sec** — `(completion_tokens - 1) / (total_latency - TTFT)` — generation speed *after* the first token, i.e. steady-state decode throughput isolated from prefill
- **Prompt tokens/sec** — `prompt_tokens / TTFT`, a rough proxy for prefill throughput (also includes network + scheduling, so treat as approximate)

`prompt_tokens` / `completion_tokens` come from the API's own `usage` field, not from a local tokenizer guess.

## Shapes

| Shape | Prompt | max_tokens |
|---|---|---|
| `short_short` | ~1 sentence (~19 tokens) | 64 |
| `short_long` | ~1 sentence | 512 |
| `long_short` | ~3.3K tokens (rotating filler paragraphs) | 64 |
| `long_long` | ~3.3K tokens | 512 |
| `vlong_short` | ~9.1K tokens | 128 |
| `vlong_long` | ~9.1K tokens | 512 |
| `xlong_short` | ~91K tokens | 128 |
| `xlong_long` | ~91K tokens | 512 |

`vlong_*`/`xlong_*` exist to answer a specific question: what happens when a user pastes something large, or an agentic tool-calling loop's conversation history has grown deep. Short output on `vlong_short`/`xlong_short` deliberately mimics a tool-call decision (a small JSON blob), not an essay — that's the more common shape in a real tool-calling loop; the `*_long` variants cover the final-summarized-response case. `xlong_*` needs the server started with `--max-model-len` >= ~92K (131072/128K in practice, see the concurrency doc below) or it'll fail with a context-length error.

**Every `prompt_fn()` call generates a fresh, unique prompt** (a random nonce prefix) — never a reused string. This isn't cosmetic: vLLM's prefix caching is on by default, and two requests sharing a leading substring get a (near-)free prefill on the second one. Reusing one static long-prompt string across repeats/shapes silently understated real TTFT/latency for `long_*`/`vlong_*`/`xlong_*` by a large margin — see the correction note at the top of the concurrency section below.

`max_tokens` is a cap, not a target — at `temperature=0.0` the model may stop earlier on a natural end-of-sequence token (this is real signal, e.g. `short_long` almost always finishes well under 512 because a two-sentence answer doesn't need that much room). `long_long` reliably hits the cap since a long-context continuation is more open-ended.

## Usage

```bash
# defaults to http://192.168.60.157:8000/v1, 5 repeats, all 4 shapes
python3 bench_baseline.py

# point at a different host, more repeats for a tighter p95, subset of shapes
python3 bench_baseline.py --base-url http://192.168.60.157:8000/v1 --repeats 8 --shapes short_short,long_long

# if/when Phase C adds an API key requirement
VLLM_API_KEY=sk-... python3 bench_baseline.py
```

Each run prints a summary table and writes a full JSON record (per-request raw results + summary stats) to `results/baseline_<UTC timestamp>.json`. Commit result files you want to keep as reference points — that's the whole point of tracking them here instead of letting them scroll off in a terminal.

## What this intentionally does *not* cover

- **GPU telemetry** (VRAM, utilization, power, temperature) — Phase D territory, meant to come from vLLM's `/metrics` endpoint + `nvidia-smi` into Prometheus/Grafana, not bolted onto these scripts.
- **Maximum stable context length** — needs its own sweep (64K/96K/128K) against the current 32,768 `--max-model-len` ceiling.

---

# Concurrency Benchmark

> **Correction (2026-09-14, same day):** early runs of this sweep for `long_long` and `vlong_short` — and two baseline runs, `results/baseline_2026-09-14T19-32-08Z.json` (existing shapes at the new 128K config) and `results/baseline_2026-09-14T19-32-53Z.json` (first `xlong_*` attempt) — reused one static prompt string across every repeat/request. vLLM's prefix caching (on by default) turned repeats into a near-free prefill after the first hit — confirmed via the server's own log, `Prefix cache hit rate: 81.3%`, and directly visible in the xlong file as an 8.7s vs. 0.29s TTFT split between two requests using the *same* ~91K-token prompt. Real traffic doesn't share prefixes across unrelated requests, so those numbers were unrealistically optimistic — including, initially, an incorrect FP8-vs-BF16 TTFT comparison below that got caught and fixed before being written down (using `baseline_2026-09-14T19-32-08Z.json`'s contaminated vlong_short TTFT as "real" BF16 made FP8 look far worse than it is). Fixed in `shapes.py` (every prompt now carries a random nonce, defeating the cache by construction) and **every number below is from a corrected, cache-defeated run** — the original result files stay in `results/` for the record but are contaminated; don't cite them for anything beyond `short_short`/`short_long` (spot-checked and confirmed unaffected — 19-token prompts have no meaningful prefill to cache): `concurrency_2026-09-14T18-43-44Z/18-44-44Z/18-47-36Z/18-47-52Z.json`, `baseline_2026-09-14T19-32-08Z.json`, `baseline_2026-09-14T19-32-53Z.json`.

`bench_concurrency.py` is Phase B: fires N requests at the same time (`concurrent.futures.ThreadPoolExecutor`, one thread per in-flight request — each does a blocking streamed HTTP call, so real concurrent requests land on the server regardless of Python's GIL) for a range of concurrency levels, using a single fixed prompt/output shape throughout so the only variable being changed is concurrency itself.

For each level, it repeats the batch (`--repeats`, default 3) and reports two different things:

- **Aggregate throughput** (`sum(completion_tokens across the batch) / batch_wall_clock_time`) — median/p95 *across batches*, since this is a rate that only makes sense computed per-batch.
- **TTFT / total latency** — median/p95 pooled *across every individual request* at that level, since these are per-request numbers and pooling more repeats gives a better percentile estimate.

```bash
python3 bench_concurrency.py --levels 1,2,4,8,16,32,64,128,256 --repeats 3
python3 bench_concurrency.py --shape long_short   # any shape from shapes.py
```

## Reference run — where does it stop scaling?

Three runs against `Qwen/Qwen3-4B-Instruct-2507` (`max_model_len=32768`), shape `short_long`, temperature 0.0 — `results/concurrency_2026-09-14T17-29-16Z.json` (levels 1–32), `...17-29-32Z.json` (64–256), `...17-29-56Z.json` (192–512, 1 repeat to probe the ceiling):

| concurrency | agg tok/s (p50) | TTFT p50 | latency p50 | latency p95 |
|---|---|---|---|---|
| 1 | 165 | 0.021s | 0.77s | 0.77s |
| 2 | 302 | 0.026s | 0.83s | 0.86s |
| 4 | 600 | 0.031s | 0.82s | 0.83s |
| 8 | 1,090 | 0.032s | 0.92s | 0.93s |
| 16 | 1,925 | 0.041s | 1.04s | 1.05s |
| 32 | 4,265 | 0.125s | 0.93s | 1.01s |
| 64 | 8,395 | 0.068s | 0.94s | 0.95s |
| 128 | 13,106 | 0.119s | 1.13s | 1.23s |
| 192 | 14,217 | 0.162s | 1.47s | 1.68s |
| **256** | **15,367 (peak)** | 0.174s | 1.88s | 2.04s |
| 384 | 10,355 (↓) | 0.266s | 2.19s | 3.39s |
| 512 | 10,762 | 1.070s (↑10x) | 3.79s | 4.80s |

**Findings:**
- Throughput scales near-linearly all the way to 32 concurrent requests (1.7–2.2x throughput for every 2x concurrency), and keeps scaling — just sub-linearly — through 256.
- **Peak aggregate throughput is ~15,400 tok/s at concurrency 256** — roughly **93x** the concurrency-1 baseline (165 tok/s) for 256x the concurrency, i.e. vLLM's continuous batching is doing real work, not just serializing requests.
- **256 is the knee.** Past it, throughput *regresses* (256→384 drops from 15,367 to 10,355 tok/s) while TTFT p50 jumps 6x (0.174s → 1.070s at 512) and p95 latency nearly triples (2.04s → 4.80s). This isn't gentle plateauing — it's the server falling behind and requests queuing.
- No request failures at any level tested, up to 512 concurrent. The ceiling here is a *quality-of-service* wall, not a hard capacity/OOM wall — whatever's limiting it (scheduler, `max_num_seqs`, KV-cache pressure) degrades service before it rejects anything outright, on this default `vllm serve` config with no explicit batching flags set.
- **Practical takeaway:** for this shape/model/hardware, keep steady-state concurrency at or below ~128–192 to stay on the good side of the latency curve; 256 is the max-throughput point but already costs noticeably more latency per request than 128 does.

This was one fixed shape (`short_long`: ~19 prompt tokens, up to 512 output) — a workload with longer prompts or a different output-length distribution would hit a different knee. Re-run with `--shape long_long` or a custom shape before trusting these exact numbers for a different traffic pattern. Tuning *why* 256 is the wall (scheduler settings, `max_num_seqs`, `gpu_memory_utilization`) is Phase G, not this script's job.

## The knee moves a *lot* with prompt size — three shapes, three very different ceilings

Prompt length dominates the concurrency ceiling far more than output length does, because it's KV-cache pressure and prefill compute (proportional to total tokens in flight) that saturate first, not decode. Three sweeps against the same server, same model, all with the prefix-cache fix (unique prompt per request):

| shape | prompt tokens | peak agg tok/s | concurrency at peak | median latency at peak concurrency | ceiling vs. `short_long` |
|---|---|---|---|---|---|
| `short_long` | ~19 | ~15,400 | 256 | 1.9s | baseline |
| `long_long` | ~3,358 | **~1,200** (flat 32→128) | 32 (already flat) | 13.4s | **~13x lower** |
| `vlong_short` | ~9,157 | **~180–186** (flat 8→64) | 8 (already flat) | 5.6s | **~83x lower** |

(`long_long`: `results/concurrency_2026-09-14T19-42-43Z.json`. `vlong_short`: `results/concurrency_2026-09-14T19-45-52Z.json`; single-request baseline in `results/baseline_2026-09-14T19-37-18Z.json` region.)

**This matters directly for any tool-calling / agentic use case**, not just raw chat: a tool-calling loop resends its growing conversation history on every round-trip, so effective prompt size climbs within a single turn, not just across a session. `vlong_short` is the closest proxy here to a real tool-calling round-trip (large input, small structured output) — and its ceiling is brutal: throughput is already flat by **concurrency 8**, and by concurrency 64 median latency is 30s with TTFT alone at 21s. At ~9K input tokens, this single GPU's realistic concurrent-user ceiling for that traffic shape is **single digits**, not the 32–48 the (contaminated) earlier numbers suggested. For `long_long`'s ~3.3K tokens the ceiling is a still-modest 32. Prompt size, not request count, is the number to watch — and it bites much harder than the first pass through this benchmark suggested.

All three shapes hit a *soft* ceiling (queuing/latency degradation), never a hard one in this range — zero request failures were observed at any concurrency level tested, up to 512 for `short_long`. Don't rely on error rate as a signal that you've found the limit; watch the throughput-vs-concurrency curve and the latency percentiles together.

## Pushing context to 128K — where the wall actually is

Restarted vLLM with `--max-model-len 131072` (128K) instead of the original 32,768. The server's own startup log gave the headline number immediately, before running a single benchmark:

```
GPU KV cache size: 138,064 tokens
Maximum concurrency for 131,072 tokens per request: 1.05x
```

That's the whole story in one line: this RTX 5090, at this model/dtype/`gpu-memory-utilization`, has room for barely more than *one* full-length 128K request at a time. Confirmed empirically with `xlong_short`/`xlong_long` (~91K-token prompts, well short of the 131K ceiling but the largest shape in this suite):

| shape | prompt tokens | TTFT (cold, unique prompt) | decode tok/s | concurrency 2 |
|---|---|---|---|---|
| `xlong_short` (128 out) | ~91,056 | **17.6s** | 70.7 | 27.2s TTFT, 1.01x throughput for 2x concurrency — flat |
| `xlong_long` (512 out) | ~91,054 | **17.6s** | 70.0 | — |

(`results/baseline_2026-09-14T19-37-18Z.json`, `results/concurrency_2026-09-14T19-38-24Z.json`.)

**Findings:**
- Raising `--max-model-len` to 128K cost *nothing* for requests that don't use it — `short_short`/`short_long` decode throughput at 128K config is within noise of the same shapes at the 32K config (167.6 vs 167.7 tok/s). The ceiling is only paid by whoever actually sends a long prompt, not as a tax on every request. (`long_long`/`vlong_short` were also measured at the 128K config in this same baseline run, but that run predates the prefix-cache fix above and is contaminated for those two shapes specifically — see the correction note.)
- A genuinely fresh ~91K-token prompt costs **~17.6s just to first token**. Decode throughput (~70 tok/s) is roughly half the short-prompt rate (~165 tok/s) — attention cost per generated token grows with context length, so it's not only prefill that gets slower.
- Concurrency at this size is exactly what the startup log predicted: **essentially none.** A second concurrent ~91K-token request doesn't add throughput (1.01x for 2x concurrency) — it just makes both requests wait roughly twice as long.
- **Practical takeaway:** 128K context is usable for a single request at a time, with the understanding that the user is waiting ~17-20+ seconds before anything starts streaming back, and that a second simultaneous huge request will queue behind it, not run alongside it. This is not a "raise a flag and move on" config — if the application needs to serve multiple users near this context size concurrently, this single GPU cannot do that at 128K; sharding across requests to different context tiers (short/medium context on this box, genuinely huge context routed elsewhere or serialized) is the realistic near-term answer, not a bigger `--max-model-len`.

---

# Quantization: FP8 vs. BF16

Tested [`Qwen/Qwen3-4B-Instruct-2507-FP8`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8) — Qwen's own official checkpoint, fine-grained block-scaled FP8, not a community requantization — against the same BF16 baseline, same hardware, same prompts (unique per request throughout).

## The RTX 5090-specific landmine

vLLM has an open bug ([vllm-project/vllm#51884](https://github.com/vllm-project/vllm/issues/51884)): block-scaled FP8 weights fail to load on sm120 (RTX 5090 / consumer Blackwell) because vLLM routes them through DeepGEMM, whose kernels reject sm120's scale-factor layout. Confirmed real on this box — the fix is two environment variables set before `vllm serve`:

```bash
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
```

With that set, the server log shows `Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod` (not DeepGEMM) and loads cleanly. Without it, expect a load-time crash, not a runtime one.

## VRAM / KV cache

| | BF16 | FP8 | 
|---|---|---|
| Weights + non-torch memory | 7.64 GiB | 5.4 GiB |
| Available for KV cache (32K config, 92% util) | 18.96 GiB | 22.27 GiB |
| GPU KV cache size (32K config) | — | 162,144 tokens |
| Max concurrency at 32,768 tokens/request | — | 4.95x |

FP8 frees up roughly **3.3 GiB more KV cache headroom** at the same `gpu-memory-utilization`, not quite the full ~half-the-weights savings would suggest once non-torch overhead is counted, but a real, usable gain.

## Throughput and latency

Comparing against BF16 numbers pulled from the *corrected* (cache-defeated) concurrency sweeps above, concurrency=1 — not the contaminated baseline file, see the correction note:

| shape | metric | BF16 (real) | FP8 (real) | delta |
|---|---|---|---|---|
| `long_long` (~3.4K prompt) | TTFT | ~0.19s | ~0.12–0.18s | same or better |
| `long_long` | decode tok/s | ~160 | 195.6 | **+22%** |
| `vlong_short` (~9.1K prompt) | TTFT | 0.517s | 0.378s | **~27% better** |
| `vlong_short` | decode tok/s | 143.7 | 173–180 | **+21–25%** |

(`results/baseline_2026-09-14T20-07-28Z.json` for FP8 baseline numbers.)

FP8 wins on **both** axes here — faster decode (smaller weights, less memory bandwidth per token fetched) and, once measured correctly, faster-or-equal TTFT too. There's no real tradeoff visible in this data once the DeepGEMM workaround is applied; the only cost is remembering to set those two environment variables.

## Concurrency: does the extra KV cache headroom translate to a higher ceiling?

`vlong_short` concurrency sweep, FP8 vs. the corrected BF16 sweep from earlier:

| concurrency | BF16 agg tok/s | FP8 agg tok/s | BF16 TTFT p50 | FP8 TTFT p50 |
|---|---|---|---|---|
| 1 | 91.2 | 115.1 | 0.517s | 0.378s |
| 2 | 126.2 | 166.4 | 0.772s | 0.561s |
| 4 | 159.5 | 205.5 | 1.272s | 0.944s |
| 8 | 179.2 | 237.5 | 2.277s | 1.644s |
| 16 | 179.8 | **247.4 (peak)** | 4.383s | 3.202s |
| 32 | 183.8 | 241.8 | 10.369s | 6.674s |
| 64 | 185.9 | 238.5 | 21.195s | 15.384s |

(`results/concurrency_2026-09-14T20-10-16Z.json`.)

**FP8's peak throughput is ~33% higher** (247 vs 186 tok/s) and its knee sits one level higher (16 vs 8) — consistent with the extra KV cache headroom buying a bit more room before the wall. At every concurrency level tested, FP8 has both higher throughput *and* lower latency than BF16 for the same shape. Past the knee, both still hit the same kind of soft wall (latency climbing, throughput flat) — FP8 shifts the wall, it doesn't remove it.

## Verdict

For this model/hardware, **FP8 (with the DeepGEMM workaround) looks like a strict upgrade over BF16** for this benchmark suite: faster decode, faster-or-equal TTFT, more KV cache headroom, and a meaningfully higher concurrency ceiling for the long-input/short-output shape that most resembles real tool-calling traffic. Model quality/accuracy was not evaluated here — this is a speed/capacity comparison only; a quality regression check (task-specific evals, not generic benchmarks) belongs to Phase E before treating this as a production decision, per the roadmap.

## Reference run

`results/baseline_2026-09-14T16-06-53Z.json` — first real baseline, `Qwen/Qwen3-4B-Instruct-2507`, `max_model_len=32768`, 8 repeats/shape, single request at a time (concurrency=1), `temperature=0.0`:

| shape | prompt_tok | out_tok | TTFT p50 | TTFT p95 | latency p50 | latency p95 | decode tok/s p50 | decode tok/s p95 |
|---|---|---|---|---|---|---|---|---|
| short_short | 19 | 64 | 0.020s | 0.021s | 0.396s | 0.441s | 167.7 | 168.0 |
| short_long | 19 | 126 | 0.019s | 0.020s | 0.768s | 0.809s | 167.1 | 167.7 |
| long_short | 3358 | 64 | 0.080s | 0.156s | 0.475s | 0.549s | 159.7 | 160.5 |
| long_long | 3358 | 512 | 0.038s | 0.111s | 3.249s | 3.257s | 159.2 | 162.4 |

Steady-state decode throughput sits around 160–168 tok/s at concurrency=1, consistent with (a bit above) the doc's earlier uncontrolled log readings of 97–163 tok/s. TTFT stays under 160ms even at ~3.3K prompt tokens on this 4B model.
