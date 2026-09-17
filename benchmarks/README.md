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

# once Phase C's auth is deployed, get the key from deploy/.env on the
# VM -- unauthenticated requests to /v1/* get a 401
export VLLM_API_KEY=<value from deploy/.env>
python3 bench_baseline.py
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

## FP8 at 128K — does the extra KV cache headroom matter at the extreme?

Same `--max-model-len 131072` restart as the BF16 128K test, FP8 model, same DeepGEMM workaround. Startup log, directly comparable to BF16's:

| | BF16 @ 128K | FP8 @ 128K |
|---|---|---|
| GPU KV cache size | 138,064 tokens | **162,144 tokens (+17%)** |
| Max concurrency at 131,072 tokens/request | 1.05x | **1.24x** |

A real, measurable gain in the *theoretical* ceiling. In practice, tested against `xlong_short`/`xlong_long` (~91K-token prompts — the same shapes used for the BF16 128K test):

| shape | BF16 TTFT | FP8 TTFT | BF16 decode | FP8 decode |
|---|---|---|---|---|
| `xlong_short` | 17.6s | **16.2s** | 70.7 | **77.4** |
| `xlong_long` | 17.6s | **16.2s** | 70.0 | **76.1** |

(`results/baseline_2026-09-14T20-35-03Z.json`.)

Modest wins (~8-9%) — smaller than the 20-25% gains seen at shorter contexts, because at ~91K tokens attention cost (unaffected by FP8 weight quantization) dominates more of the total compute than the linear/MLP layers that actually get the speedup. **Concurrency at this prompt size is still flat**: two concurrent `xlong_short` requests gave 1.00x throughput for 2x concurrency (TTFT 16.2s → 25.0s) — no better than BF16's 1.01x. The reason: two ~91K-token requests need ~182K tokens of KV cache between them, and FP8's 162,144-token pool still isn't enough to hold two, even though it holds meaningfully more than BF16's 138,064. **The 1.05x→1.24x theoretical improvement is real but only bites at request sizes closer to the max_model_len ceiling itself** — for genuinely huge concurrent prompts like these test shapes, both precisions are still effectively single-request-at-a-time.

## Verdict

For this model/hardware, **FP8 (with the DeepGEMM workaround) looks like a strict upgrade over BF16** for this benchmark suite: faster decode (+20-25% at moderate context, a smaller but real +8-9% even at ~91K tokens), faster-or-equal TTFT everywhere tested, more KV cache headroom, and a meaningfully higher concurrency ceiling for the ~9K-token long-input/short-output shape closest to real tool-calling traffic (247 vs 186 tok/s peak). That concurrency win doesn't carry all the way to the context extreme, though — at ~91K-token prompts, both precisions are effectively single-request-at-a-time; FP8's larger KV cache pool (162,144 vs 138,064 tokens) helps, but not enough to fit two requests that size at once. Model quality/accuracy was not evaluated here — this is a speed/capacity comparison only; a quality regression check (task-specific evals, not generic benchmarks) belongs to Phase E before treating this as a production decision, per the roadmap.

## Reference run

`results/baseline_2026-09-14T16-06-53Z.json` — first real baseline, `Qwen/Qwen3-4B-Instruct-2507`, `max_model_len=32768`, 8 repeats/shape, single request at a time (concurrency=1), `temperature=0.0`:

| shape | prompt_tok | out_tok | TTFT p50 | TTFT p95 | latency p50 | latency p95 | decode tok/s p50 | decode tok/s p95 |
|---|---|---|---|---|---|---|---|---|
| short_short | 19 | 64 | 0.020s | 0.021s | 0.396s | 0.441s | 167.7 | 168.0 |
| short_long | 19 | 126 | 0.019s | 0.020s | 0.768s | 0.809s | 167.1 | 167.7 |
| long_short | 3358 | 64 | 0.080s | 0.156s | 0.475s | 0.549s | 159.7 | 160.5 |
| long_long | 3358 | 512 | 0.038s | 0.111s | 3.249s | 3.257s | 159.2 | 162.4 |

Steady-state decode throughput sits around 160–168 tok/s at concurrency=1, consistent with (a bit above) the doc's earlier uncontrolled log readings of 97–163 tok/s. TTFT stays under 160ms even at ~3.3K prompt tokens on this 4B model.

---

# Model Comparison: Qwen3.5-4B vs. Phi-4-mini-instruct vs. Gemma-4-E4B-it

Full raw numbers for every model/precision/context combination live in a
dedicated doc — [`docs/MODEL_COMPARISON_RESULTS.md`](../docs/MODEL_COMPARISON_RESULTS.md)
— since there are twelve separate runs' worth of tables. This section is the
narrative summary and cross-model synthesis.

Same-size-class candidates from the roadmap ("bigger models cut context capacity
the wrong direction for a growing tool-calling use case, so compare against
same-size alternatives instead"): `Qwen/Qwen3.5-4B`, `microsoft/Phi-4-mini-instruct`,
`google/gemma-4-E4B-it` (corrected from the README's placeholder `gemma-4-E4B` —
that's the non-instruct base checkpoint), each tested in BF16 and an FP8-dynamic
quant, at both 32K and 128K `--max-model-len`, same harness and methodology as
every other benchmark in this repo (unique-nonce prompts, `temperature=0.0`).

## The headline: Qwen3.5-4B beats the incumbent at its own game

At the shape that matters most for this repo's use case — a ~91K-token prompt at
128K context, standing in for a deep tool-calling conversation history —
Qwen3.5-4B-FP8 delivers:

| | Incumbent (Qwen3-4B-Instruct-2507-FP8) | Qwen3.5-4B-FP8 | Delta |
|---|---|---|---|
| TTFT @ xlong_short | 16.2s | 6.3s | **~2.6x faster** |
| Decode tok/s @ xlong_short | 77.4 | 161.1 | **~2.1x faster** |
| GPU KV cache size @ 128K | 162,144 tokens | 634,799 tokens | **~3.9x more headroom** |
| Max concurrency @ 128K | 1.24x | 4.84x | **~3.9x higher ceiling** |

This isn't a quantization artifact — BF16-vs-BF16 shows the same gap (Qwen3.5-4B
BF16: 7.1s TTFT / 129 tok/s decode vs. the incumbent's BF16: 17.6s / 70.7 tok/s).
The likely cause: Qwen3.5-4B mixes in linear-attention layers (Gated DeltaNet-style
— visible in its FP8 quant config's per-layer `linear_attn` targets and in vLLM's
compiled op list, `qwen_gdn_attention_core`) alongside standard attention. Linear
attention's per-token cost doesn't grow with sequence length the way standard
attention's does, so a hybrid architecture pays less of the long-context tax —
consistent with both the smaller TTFT/decode gap at long context *and* the ~4x
larger raw KV-cache-token capacity at the same `--max-model-len`.

At shorter shapes the two are closer (Qwen3.5-4B still wins on decode throughput
by 20-25% with FP8, same pattern as the original FP8-vs-BF16 comparison), but the
long-context gap is where this result actually matters for the roadmap's stated
priority ("growing tool-calling conversations want *more* context, not less").

## Phi-4-mini-instruct: fine at 32K, falls off a cliff at 128K

Competitive at moderate context — FP8 decode reaches 145-261 tok/s across
short/long/vlong shapes, comparable to or better than the incumbent. But at 128K,
decode collapses to **~11 tok/s** (both precisions) — a 6-7x drop from its own 32K
numbers, and *worse* than the incumbent's own 128K decode (70-77 tok/s). Its raw
KV-cache-token capacity (152K-177K) is in the same range as the incumbent's
(138K-162K) rather than the ~500K-900K the other two candidates show — consistent
with `Phi3ForCausalLM` being a conventional dense-attention architecture with none
of Qwen3.5's or Gemma's hybrid-attention relief. Not a fit for this repo's
long-context priority.

## Gemma-4-E4B-it: architecturally the most promising, but currently broken

Gemma-4-E4B-it has the **largest KV-cache headroom of any model tested in this
repo** (596K-890K tokens, 18-22x max concurrency at 32K) and the **fastest raw
throughput measured here** (16,403 tok/s peak aggregate at concurrency 256, beating
even the incumbent's reference sweep) — on short prompts. **Every single request
with a prompt longer than ~30 tokens failed**, identically across both precisions
and both context configs: the server returns `200 OK` with `completion_tokens: 1`
and zero visible content, i.e. an immediate end-of-sequence token, with no error
anywhere in the server log. Reproduced directly (not just inferred) — see
`docs/MODEL_COMPARISON_RESULTS.md` for the full repro and a best-effort root-cause
hypothesis (vLLM falls back to `TRITON_ATTN` for this architecture's heterogeneous
sliding/full-attention heads since FA4 isn't available, and a correctness bug in
that fallback for long sequences is the most consistent explanation for what was
observed). **Not usable for this repo's use case until this is fixed upstream** —
flagging it here rather than silently excluding it, same as the RTX 5090 DeepGEMM
landmine got flagged rather than worked around quietly.

## Verdict

**Qwen3.5-4B-FP8 looks like a genuine upgrade over the current production
incumbent** for this repo's actual priority (long-context speed and concurrency),
backed by a real architectural difference rather than a lucky benchmark run — the
gap holds across every shape tested. Phi-4-mini-instruct doesn't fit the
long-context priority despite being fine at short context. Gemma-4-E4B-it is the
most architecturally interesting of the three but can't be evaluated for the real
use case until its long-prompt bug is resolved.

As always: **speed and capacity only, quality not evaluated.** Before touching the
production server config, Phase E (task-specific quality evals) still needs to
happen — this comparison narrows the field, it doesn't make the final call.

---

# Model Comparison Round 2: Casting a Wider Net

Full raw results: [`docs/MODEL_COMPARISON_ROUND2_RESULTS.md`](../docs/MODEL_COMPARISON_ROUND2_RESULTS.md).
This section is the summary.

Round 1 stuck to same-size-class Qwen/Phi/Gemma alternatives. Round 2 went
back to vLLM's actual supported-architectures list
(<https://docs.vllm.ai/en/latest/models/supported_models/>) and picked 5 new
families, prioritizing hybrid/linear-attention designs — the strongest
predictor of long-context performance found so far. Every candidate was
verified against the live HuggingFace API *and* the installed vLLM's own
`ModelRegistry` before downloading anything, after round 1's Gemma bug made
clear that "vLLM supports it" and "this exact install actually serves it
correctly" are different claims.

**Net result: no clean new champion, four confirmed landmines (one
corrected after a same-day follow-up investigation), and one
clearly-adoptable VRAM win.**

## Granite-4.0-H-Micro: impressive numbers, real unreliability — and a corrected root cause

`ibm-granite/granite-4.0-h-micro` — 3.2B params, less than half the size of
round 1's Qwen3.5-4B winner — with an AWQ quant hits **350-437 tok/s decode
and 4.1s TTFT on a real ~91K-token prompt at 128K context**, more than double
Qwen3.5-4B-FP8's 161 tok/s / 6.3s at the identical shape. Its KV-cache
capacity (2.1-3.0 *million* tokens) dwarfs every other model tested in this
project by one to two orders of magnitude — its `GraniteMoeHybridForCausalLM`
architecture runs Mamba2 for 9 of every 10 layers, and Mamba state doesn't
grow with sequence length the way attention KV cache does.

The original writeup here said this checkpoint was "broken at 32K, clean at
128K" and left it as an open question. **That was wrong, and a same-day
follow-up investigation found the real cause**: it's not about
`--max-model-len` at all. Direct `curl` testing (bypassing the benchmark
harness, to rule out a client bug) found the actual trigger is **prompt
length alone** — prompts under ~12,000 tokens fail reliably (`200 OK`,
`completion_tokens: 1`, empty text) **at both 32K and 128K**, and prompts
above that threshold succeed reliably at either config. Three plausible
mechanisms were ruled out by direct A/B test (prefix caching, CUDA graph
capture/replay — neither changed the outcome at all). The unquantized BF16
checkpoint shows the identical failure signature, just far narrower — ~40-60%
of attempts fail on trivial ~20-30-token prompts, content-dependent, while
its ~3.3K-token prompts succeed reliably. **Best-supported explanation**: a
real fragility in this hybrid Mamba2/MoE architecture at short-to-medium
prompt lengths (plausibly insufficient "warm-up" signal in the Mamba state
and/or MoE router on a fresh short sequence), which this community AWQ
4-bit quantization amplifies from "rare, ~30 tokens" to "reliable, ~12,000
tokens" — consistent with a brand-new hybrid architecture being unusually
sensitive to quantization noise in exactly the cases where it was already
marginal. Full investigation and evidence in `docs/MODEL_COMPARISON_ROUND2_RESULTS.md`.

This is a **stronger** disqualifier than the original "broken at 32K"
framing, not a weaker one: a real tool-calling conversation starts short and
grows, so it spends most of its life exactly in the prompt-length range where
this model is unreliable — the excellent 91K-token numbers don't matter if
the model can't reliably get through the early turns of that same
conversation. Not a production candidate as this checkpoint stands, but the
128K numbers are real enough to be worth revisiting if a fix (official
checkpoint, different quant, or an upstream vLLM/architecture fix) ever
shows up.

## The speed champion that can't do the job it needs to do

`LiquidAI/LFM2.5-2.6B` — the smallest model tested in this project (2.7B) —
posted the **highest raw throughput measured anywhere in this repo**: 20,997
tok/s peak aggregate (beats Gemma-4-E4B-it's round-1 record of 16,403), with
245-333 tok/s single-request decode, up to ~9.8K real prompt tokens. Then it
**fails 100% of requests at 128K context** — same "clean `200 OK`,
`completion_tokens: 1`, zero content" signature Gemma showed in round 1, just
triggered by a different condition (long context here, vs. any prompt over
~30 tokens for Gemma). Excellent candidate for a short/medium-context use
case; unusable for this repo's actual long-tool-calling-history priority.

## Three architectures that don't load at all on this install

- **`tiiuae/Falcon-H1-7B-Instruct`** (BF16, no quant available) — hits a hard
  ~16.84 GiB CUDA-graph-profiling allocation that doesn't shrink under either
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` or a lower
  `--gpu-memory-utilization` — genuinely too tight for this 32GB GPU under
  vLLM 0.29's current memory-profiling behavior for this hybrid architecture.
- **`nvidia/Nemotron-H-8B-Reasoning-128K-FP8`** (the *official* NVIDIA FP8
  checkpoint) — fails during weight loading (`'MergedColumnParallelLinear'
  object has no attribute 'data'`), a real vLLM/checkpoint incompatibility.
  The BF16 version of the same model works fine. **Root-caused in a same-day
  follow-up** (see `docs/MODEL_COMPARISON_ROUND2_RESULTS.md`): vLLM 0.29
  doesn't auto-detect this checkpoint's quantization config for this
  architecture at all, and refuses an explicit `--quantization modelopt`
  override — genuinely not fixable from the outside this session.
- ~~`mistralai/Ministral-3-8B-Instruct-2512` (both precisions) — fails to
  even import (`ImportError: cannot import name 'PixtralRotaryEmbedding'`)~~
  **Fixed in the same follow-up.** A two-symbol `transformers`/vLLM
  version-skew bug in vLLM's vision-tower import path (loads unconditionally
  even for text-only use) — patched with a small venv-scoped compatibility
  shim, now loads and serves cleanly at 32K and 128K. See the full root
  cause, the fix, and first real benchmark numbers in
  `docs/MODEL_COMPARISON_ROUND2_RESULTS.md`'s "Follow-up (2026-09-16)"
  section. Along the way, also found that its "FP8" and "BF16" checkpoints
  are byte-identical (Mistral's base release already ships natively
  mixed-precision) — there's no separate BF16 baseline for this model.

Falcon-H1-7B remains unresolved; the other two are now root-caused, one
fixed and one confirmed not fixable without a vLLM code change.

## The adoptable win: `--kv-cache-dtype fp8`

Separate from every weight quantization tested so far, vLLM can quantize the
**KV cache** itself. Tested on the round-1 champion (Qwen3.5-4B-FP8 @ 128K),
changing nothing else:

| | baseline (`auto`) | `--kv-cache-dtype fp8` | Delta |
|---|---|---|---|
| KV cache size | 634,799 tokens | 1,205,662 tokens | **+90%** |
| Max concurrency @ 128K | 4.84x | 9.20x | **~1.9x** |
| Decode tok/s | 161.1 | 185.3 | **+15%** |
| TTFT | 6.3s | 7.6s | -20% (worse) |

Roughly double the KV-cache headroom and concurrency ceiling, plus slightly
*faster* decode (less memory bandwidth per attention read), for one flag on
an already-running model — no download, no new model risk. The tradeoff is
~20% worse TTFT (extra quantize/dequantize overhead during prefill), and the
extra headroom doesn't help genuinely huge simultaneous prompts (~91K tokens)
scale any better — that's compute-bound, not memory-bound, confirming round
1's finding. **This is a real, low-effort software-level capacity win,
directly answering "optimize before buying more hardware"** — worth adopting
on whichever model ends up in production.

**Doesn't generalize to every model, though — and that includes the
currently-deployed one.** Tried the same flag on Nemotron-H-8B (BF16) and
Ministral-3-8B in a same-day follow-up — both fail outright, a genuine sm120
(RTX 5090 / consumer Blackwell) gap in FlashInfer's fused `xqa` decode
kernel, separate from the DeepGEMM weight-FP8 bug above. Three different
mitigations tried (a FlashInfer patch-version bump, forcing
`VLLM_ATTENTION_BACKEND=FLASH_ATTN`, forcing `TRITON_ATTN`) all failed
identically or were silently ignored. **Then tested it against the actual
production model** (`Qwen/Qwen3-4B-Instruct-2507-FP8`) and it fails there
too, identical signature — 3 of 4 architectures tested this session, only
Qwen3.5-4B-FP8 works. Confirmed Qwen3.5-4B is a genuinely hybrid
linear+full-attention architecture, but that alone doesn't explain it since
Nemotron-H (also hybrid) fails the same way — not fully root-caused.
**Practical effect: this lever can't be "adopted on production" as an
independent step; it's coupled to switching production to Qwen3.5-4B-FP8
first**, which is itself gated on Phase E quality evals. See
`docs/MODEL_COMPARISON_ROUND2_RESULTS.md` for the full trace.

Other untested-but-real levers found in `vllm serve --help`:
`--language-model-only` (skip loading vision/audio towers on multimodal
models used text-only), `--cpu-offload-gb` (offload weights to the 62GB of
system RAM, virtually extending VRAM at a PCIe-latency cost), and
`--kv-offloading-size` (offload cold KV-cache blocks to CPU RAM rather than
the whole model).

---

# Phase E: Snow-Zone Tool-Calling Quality Eval

Everything above measures speed and capacity. This is the first eval in the
repo that measures whether the output is *correct* — built against the real
production tool registry (`Operator-Portal`'s `LlmTools::Registry`), not an
invented schema. Full writeup: `docs/PHASE_E_TOOL_CALLING_EVAL_RESULTS.md`.
Harness: `eval_tool_calling.py` / `eval_cases_snowzone.py` / `eval_tool_schema.py`.

```bash
python3 eval_tool_calling.py --model RedHatAI/Qwen3.5-4B-FP8-dynamic --repeats 3
python3 eval_tool_calling.py --model Qwen/Qwen3-4B-Instruct-2507-FP8 --repeats 3
```

**Headline: Qwen3.5-4B-FP8 clears its first quality gate.** 39/42 overall vs.
the incumbent's 33/42 (after tightening the grader — one fix was itself worth
catching: a first attempt at "accept a clarifying question" accidentally also
accepted the incumbent's false-success hallucination, since it happened to
mention the right zone name too), but the safety-relevant subset is the real
story — 9/12 vs. 3/12. The incumbent fabricated a snowfall depth reading
(`inches: 2.5`) the user never gave, on the one tool that writes to
payroll-linked records, and separately narrated a successful report that
**never actually called the tool**. The champion's one remaining issue is
getting stuck re-querying `list_snowfall_zones` instead of committing to a
resolved zone or asking directly — a minor loop-risk, not fabrication.

Two things worth remembering before reusing this harness on a different
model: the tool-call parser is architecture-specific and picking the wrong
one silently produces plain-text rambling instead of a `tool_calls` response
(`qwen3_xml` for Qwen3.5-4B, `hermes` for the plain Qwen3 incumbent — same
"wrong parser looks like a broken model" trap as everything else
architecture-specific in this repo), and Qwen3.5-4B visibly reasons in
`content` before every tool call (more completion tokens per call than the
incumbent's direct calls) — a real latency/cost tradeoff, not free caution.
