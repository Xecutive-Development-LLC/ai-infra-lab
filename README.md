# AI Infra Lab

A running log of learnings, work, and discoveries from building and operating self-hosted AI infrastructure — inference servers, benchmarking, evals, and the agent/application layer on top of them.

## Current Project: Self-Hosted LLM Inference Server

Self-hosted, OpenAI-API-compatible LLM inference stack on bare-metal GPU hardware, built from the ground up: hypervisor → GPU passthrough → driver/CUDA → Python/PyTorch → vLLM → served model.

**Stack:** RTX 5090 · Proxmox VE · Ubuntu 24.04 · CUDA 13.2 · PyTorch 2.13 · vLLM 0.29 · Qwen3-4B-Instruct

**Status:** End-to-end path is working — GPU passthrough, driver/CUDA, PyTorch, vLLM, and the OpenAI-compatible API have all been validated. The VM now has a static IP (`192.168.60.157`, survives reboot) instead of DHCP. Phase A (single-request baseline) and Phase B (concurrency scaling) are both done — see [`benchmarks/`](benchmarks/). Concurrency ceiling is **highly prompt-size-dependent**: peak throughput/knee moves from ~15,400 tok/s @ concurrency 256 for trivial prompts down to ~180 tok/s @ concurrency 8 at ~9K prompt tokens — prompt size, not request count, is the real capacity constraint. **FP8 quantization tested and looks like a strict upgrade** over BF16 (official `Qwen/Qwen3-4B-Instruct-2507-FP8` checkpoint, needs a documented env-var workaround for an RTX-5090-specific vLLM bug) at both 32K and 128K context: faster decode, better-or-equal TTFT, more KV cache headroom, higher concurrency ceiling — speed/capacity only, quality wasn't evaluated. **Currently running: FP8 model at 128K context (`--max-model-len 131072`)** — same live config on `192.168.60.157:8000` as before this comparison started; restored automatically once the benchmark run finished.

**Model comparison done:** benchmarked three same-size-class candidates (`Qwen/Qwen3.5-4B`, `microsoft/Phi-4-mini-instruct`, `google/gemma-4-E4B-it`) against the incumbent, each in BF16 and FP8, at 32K and 128K context — see [`benchmarks/README.md`](benchmarks/README.md)'s "Model Comparison" section for the writeup and [`docs/MODEL_COMPARISON_RESULTS.md`](docs/MODEL_COMPARISON_RESULTS.md) for every raw number. **Qwen3.5-4B-FP8 looks like a genuine upgrade over the incumbent** — at a ~91K-token prompt (128K context), it's ~2.6x faster TTFT, ~2.1x faster decode, and has ~3.9x more KV-cache headroom/concurrency ceiling, most likely from hybrid linear-attention layers that don't pay the same long-context tax as standard attention. Phi-4-mini-instruct is competitive at 32K but its decode throughput collapses to ~11 tok/s at 128K (worse than the incumbent). Gemma-4-E4B-it has the best raw numbers of all four models on short prompts but **fails on 100% of requests longer than ~30 tokens** (immediate end-of-sequence, no server-side error) — reproduced and written up, likely a vLLM day-0-support gap for this brand-new architecture's attention backend, not a model defect. As always: speed/capacity only, quality not evaluated — Phase E still gates any actual production switch.

Full build log, architecture, commands, concepts learned, problems/fixes, and the phased roadmap (A–G) live in:
- [`docs/AI_Inference_Server_Build_Log_and_Roadmap.md`](docs/AI_Inference_Server_Build_Log_and_Roadmap.md) — working copy, kept current
- [`docs/AI_Inference_Server_Build_Log_and_Roadmap.docx`](docs/AI_Inference_Server_Build_Log_and_Roadmap.docx) — original source doc
- [`docs/MODEL_COMPARISON_RESULTS.md`](docs/MODEL_COMPARISON_RESULTS.md) — full raw results for every model/precision/context combination

### Immediate next steps
1. ~~Build a repeatable benchmark harness (TTFT, latency, tokens/sec, p50/p95).~~ Done — [`benchmarks/bench_baseline.py`](benchmarks/bench_baseline.py).
2. ~~Test concurrency (2/4/8+ requests) and observe vLLM continuous batching behavior.~~ Done — [`benchmarks/bench_concurrency.py`](benchmarks/bench_concurrency.py).
3. ~~Push context length to 128K and find the real wall.~~ Done — see `benchmarks/README.md`'s "Pushing context to 128K" section.
4. ~~Benchmark a quantized model (FP8) and compare, including at 128K context.~~ Done — see `benchmarks/README.md`. FP8 raises the 128K ceiling from 1.05x to 1.24x max concurrency (138,064 → 162,144 KV cache tokens), a real but modest gain — at ~91K-token prompts both precisions are still effectively single-request-at-a-time.
5. ~~Download and benchmark `Qwen/Qwen3.5-4B`, `microsoft/Phi-4-mini-instruct`, and `google/gemma-4-E4B-it` against the current baseline.~~ Done — see `benchmarks/README.md`'s "Model Comparison" section and `docs/MODEL_COMPARISON_RESULTS.md`. Qwen3.5-4B-FP8 wins decisively on long-context speed/capacity; Phi-4-mini-instruct falls off at 128K; Gemma-4-E4B-it is currently broken on long prompts (vLLM-side, not benchmarked further). AWQ/int4 as a second quantization data point remains open but lower priority.
6. **Next session:** either (a) chase down the Gemma-4-E4B-it long-prompt bug (check for an upstream vLLM fix / try a non-Triton attention backend) if it's worth the time given Qwen3.5-4B already looks like the stronger candidate, or (b) move straight to Phase E — task-specific quality/accuracy evals for Qwen3.5-4B-FP8 (the current front-runner) before considering a production switch away from the incumbent; task-specific model routing as AI use cases grow, rather than one model for everything.
7. Make CUDA PATH persistent (still shell-session-only); daemonize vLLM (systemd or Docker + NVIDIA Container Toolkit) so it survives reboots.
8. Then: observability (Prometheus/Grafana — GPU utilization/power/VRAM/KV-cache during load is still unmeasured), and the first real agent/tool-calling use case.
9. Visual charts/graphs of the benchmark results for non-technical stakeholders — deferred until requested; raw data is all in place under `benchmarks/results/` and `docs/MODEL_COMPARISON_RESULTS.md` to build them from whenever needed.

### A methodology lesson worth flagging

Early long-context concurrency numbers in this repo were quietly wrong: `shapes.py` generated the *same* deterministic prompt text on every repeat/request, and vLLM's prefix caching (on by default) turned repeats into near-free cached prefills — confirmed via the server's own `Prefix cache hit rate: 81.3%` log line. Real traffic never shares prefixes like that. Fixed by making every generated prompt carry a random nonce; the corrected numbers are markedly worse (e.g. `long_long`'s peak throughput dropped from a reported ~4,000 tok/s to a real ~1,200 tok/s). The same contamination briefly produced a second, opposite-direction mistake: comparing FP8 against a BF16 baseline file that turned out to still be contaminated made FP8's TTFT look much worse than BF16's, when the corrected comparison shows FP8 is actually as-good-or-better on TTFT too. Caught before being written down as a conclusion, but a reminder that one bug can produce more than one wrong downstream number. Full story in `benchmarks/README.md`.

## Repo Structure

```
docs/       Build logs, architecture notes, roadmaps for each project
benchmarks/ Benchmark scripts, harnesses, and results (as they're built)
```

## Conventions

- Every benchmark result is recorded with hardware, model, precision/quantization, context size, concurrency, and sampling parameters — no bare numbers.
- Work happens on feature branches merged via PR; `main` stays the record of what's actually been validated.
