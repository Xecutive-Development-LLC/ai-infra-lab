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

- **Concurrency** (Phase B — 2/4/8+ concurrent requests, aggregate throughput, continuous batching behavior) — separate script, since it needs an async/concurrent client rather than the sequential loop here.
- **GPU telemetry** (VRAM, utilization, power, temperature) — Phase D territory, meant to come from vLLM's `/metrics` endpoint + `nvidia-smi` into Prometheus/Grafana, not bolted onto this script.
- **Maximum stable context length** — needs its own sweep (64K/96K/128K) against the current 32,768 `--max-model-len` ceiling.

## Reference run

`results/baseline_2026-09-14T16-06-53Z.json` — first real baseline, `Qwen/Qwen3-4B-Instruct-2507`, `max_model_len=32768`, 8 repeats/shape, single request at a time (concurrency=1), `temperature=0.0`:

| shape | prompt_tok | out_tok | TTFT p50 | TTFT p95 | latency p50 | latency p95 | decode tok/s p50 | decode tok/s p95 |
|---|---|---|---|---|---|---|---|---|
| short_short | 19 | 64 | 0.020s | 0.021s | 0.396s | 0.441s | 167.7 | 168.0 |
| short_long | 19 | 126 | 0.019s | 0.020s | 0.768s | 0.809s | 167.1 | 167.7 |
| long_short | 3358 | 64 | 0.080s | 0.156s | 0.475s | 0.549s | 159.7 | 160.5 |
| long_long | 3358 | 512 | 0.038s | 0.111s | 3.249s | 3.257s | 159.2 | 162.4 |

Steady-state decode throughput sits around 160–168 tok/s at concurrency=1, consistent with (a bit above) the doc's earlier uncontrolled log readings of 97–163 tok/s. TTFT stays under 160ms even at ~3.3K prompt tokens on this 4B model.
