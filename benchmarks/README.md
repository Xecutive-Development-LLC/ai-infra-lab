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

## Reference run

`results/baseline_2026-09-14T16-06-53Z.json` — first real baseline, `Qwen/Qwen3-4B-Instruct-2507`, `max_model_len=32768`, 8 repeats/shape, single request at a time (concurrency=1), `temperature=0.0`:

| shape | prompt_tok | out_tok | TTFT p50 | TTFT p95 | latency p50 | latency p95 | decode tok/s p50 | decode tok/s p95 |
|---|---|---|---|---|---|---|---|---|
| short_short | 19 | 64 | 0.020s | 0.021s | 0.396s | 0.441s | 167.7 | 168.0 |
| short_long | 19 | 126 | 0.019s | 0.020s | 0.768s | 0.809s | 167.1 | 167.7 |
| long_short | 3358 | 64 | 0.080s | 0.156s | 0.475s | 0.549s | 159.7 | 160.5 |
| long_long | 3358 | 512 | 0.038s | 0.111s | 3.249s | 3.257s | 159.2 | 162.4 |

Steady-state decode throughput sits around 160–168 tok/s at concurrency=1, consistent with (a bit above) the doc's earlier uncontrolled log readings of 97–163 tok/s. TTFT stays under 160ms even at ~3.3K prompt tokens on this 4B model.
