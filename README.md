# AI Infra Lab

A running log of learnings, work, and discoveries from building and operating self-hosted AI infrastructure — inference servers, benchmarking, evals, and the agent/application layer on top of them.

## Current Project: Self-Hosted LLM Inference Server

Self-hosted, OpenAI-API-compatible LLM inference stack on bare-metal GPU hardware, built from the ground up: hypervisor → GPU passthrough → driver/CUDA → Python/PyTorch → vLLM → served model.

**Stack:** RTX 5090 · Proxmox VE · Ubuntu 24.04 · CUDA 13.2 · PyTorch 2.13 · vLLM 0.29 · Qwen3-4B-Instruct

**Status:** End-to-end path is working — GPU passthrough, driver/CUDA, PyTorch, vLLM, and the OpenAI-compatible API have all been validated. The VM now has a static IP (`192.168.60.157`, survives reboot) instead of DHCP. Phase A (single-request baseline) and Phase B (concurrency scaling) are both done — see [`benchmarks/`](benchmarks/). Concurrency ceiling is **highly prompt-size-dependent**: peak throughput/knee moves from ~15,400 tok/s @ concurrency 256 for trivial prompts down to ~180 tok/s @ concurrency 8 at ~9K prompt tokens — prompt size, not request count, is the real capacity constraint. Server can run at 128K context (`--max-model-len 131072`), but at that size the GPU can barely serve *one* full-length request at a time (measured: ~17.6s TTFT for a fresh ~91K-token prompt, concurrency flat at 2). **FP8 quantization tested and looks like a strict upgrade** over BF16 (official `Qwen/Qwen3-4B-Instruct-2507-FP8` checkpoint, needs a documented env-var workaround for an RTX-5090-specific vLLM bug): +20-25% decode throughput, better-or-equal TTFT, ~3.3 GiB more KV cache headroom, ~33% higher concurrency ceiling on the long-input shape — speed/capacity only, quality wasn't evaluated. Persistence (systemd/Docker), multi-model comparison, security, observability, and the application/agent layer are next.

Full build log, architecture, commands, concepts learned, problems/fixes, and the phased roadmap (A–G) live in:
- [`docs/AI_Inference_Server_Build_Log_and_Roadmap.md`](docs/AI_Inference_Server_Build_Log_and_Roadmap.md) — working copy, kept current
- [`docs/AI_Inference_Server_Build_Log_and_Roadmap.docx`](docs/AI_Inference_Server_Build_Log_and_Roadmap.docx) — original source doc

### Immediate next steps
1. ~~Build a repeatable benchmark harness (TTFT, latency, tokens/sec, p50/p95).~~ Done — [`benchmarks/bench_baseline.py`](benchmarks/bench_baseline.py).
2. ~~Test concurrency (2/4/8+ requests) and observe vLLM continuous batching behavior.~~ Done — [`benchmarks/bench_concurrency.py`](benchmarks/bench_concurrency.py).
3. ~~Push context length to 128K and find the real wall.~~ Done — see `benchmarks/README.md`'s "Pushing context to 128K" section.
4. ~~Benchmark a quantized model (FP8) and compare.~~ Done — see `benchmarks/README.md`'s "Quantization: FP8 vs. BF16" section. Still open: FP8 at 128K context (does the extra KV cache headroom meaningfully raise the 1.05x ceiling?), AWQ/int4 as a second data point, and quality/accuracy eval (Phase E) before trusting this for production.
5. Compare across different model families/sizes for quality and speed, then task-specific model routing as AI use cases grow, rather than one model for everything.
6. Make CUDA PATH persistent (still shell-session-only); daemonize vLLM (systemd or Docker + NVIDIA Container Toolkit) so it survives reboots.
7. Then: observability (Prometheus/Grafana — GPU utilization/power/VRAM/KV-cache during load is still unmeasured), model evals, and the first real agent/tool-calling use case.

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
