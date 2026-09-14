# AI Infra Lab

A running log of learnings, work, and discoveries from building and operating self-hosted AI infrastructure — inference servers, benchmarking, evals, and the agent/application layer on top of them.

## Current Project: Self-Hosted LLM Inference Server

Self-hosted, OpenAI-API-compatible LLM inference stack on bare-metal GPU hardware, built from the ground up: hypervisor → GPU passthrough → driver/CUDA → Python/PyTorch → vLLM → served model.

**Stack:** RTX 5090 · Proxmox VE · Ubuntu 24.04 · CUDA 13.2 · PyTorch 2.13 · vLLM 0.29 · Qwen3-4B-Instruct

**Status:** End-to-end path is working — GPU passthrough, driver/CUDA, PyTorch, vLLM, and the OpenAI-compatible API have all been validated. The VM now has a static IP (`192.168.60.157`, survives reboot) instead of DHCP. Phase A (single-request baseline) and Phase B (concurrency scaling) are both done — see [`benchmarks/`](benchmarks/). Concurrency ceiling is **highly prompt-size-dependent**: peak throughput/knee moves from ~15,400 tok/s @ concurrency 256 for trivial prompts down to ~180 tok/s @ concurrency 8 at ~9K prompt tokens — prompt size, not request count, is the real capacity constraint. **FP8 quantization tested and looks like a strict upgrade** over BF16 (official `Qwen/Qwen3-4B-Instruct-2507-FP8` checkpoint, needs a documented env-var workaround for an RTX-5090-specific vLLM bug) at both 32K and 128K context: faster decode, better-or-equal TTFT, more KV cache headroom, higher concurrency ceiling — speed/capacity only, quality wasn't evaluated. **Currently running: FP8 model at 128K context (`--max-model-len 131072`)** — that's the live state on `192.168.60.157:8000` as of the last session.

**Paused here, resuming next session:** researched bigger models (Qwen3.6-27B, Mistral-Small-3.2-24B, Qwen3.6-35B-A3B MoE) for a quality comparison, but real VRAM math ruled them out — at 22-25GB real weight size (AWQ int4), they'd cut context capacity from 128K+ down to 23K-80K tokens, which conflicts with the actual need (growing tool-calling conversations want *more* context, not less). Pivoted to same-size-class candidates instead, verified against HuggingFace directly: **`Qwen/Qwen3.5-4B`** (same family, newer gen, 262K native context), **`microsoft/Phi-4-mini-instruct`** (different family, function-calling-focused, 128K native context), **`google/gemma-4-E4B`** (different family, multimodal-capable, 128K native context). All three fit comfortably without sacrificing context headroom. `ministral/Ministral-4b-instruct` was considered and dropped — fits VRAM but is capped at 32K context regardless of hardware. **Next up: download and benchmark these three against the current Qwen3-4B(-FP8) baseline.**

Full build log, architecture, commands, concepts learned, problems/fixes, and the phased roadmap (A–G) live in:
- [`docs/AI_Inference_Server_Build_Log_and_Roadmap.md`](docs/AI_Inference_Server_Build_Log_and_Roadmap.md) — working copy, kept current
- [`docs/AI_Inference_Server_Build_Log_and_Roadmap.docx`](docs/AI_Inference_Server_Build_Log_and_Roadmap.docx) — original source doc

### Immediate next steps
1. ~~Build a repeatable benchmark harness (TTFT, latency, tokens/sec, p50/p95).~~ Done — [`benchmarks/bench_baseline.py`](benchmarks/bench_baseline.py).
2. ~~Test concurrency (2/4/8+ requests) and observe vLLM continuous batching behavior.~~ Done — [`benchmarks/bench_concurrency.py`](benchmarks/bench_concurrency.py).
3. ~~Push context length to 128K and find the real wall.~~ Done — see `benchmarks/README.md`'s "Pushing context to 128K" section.
4. ~~Benchmark a quantized model (FP8) and compare, including at 128K context.~~ Done — see `benchmarks/README.md`. FP8 raises the 128K ceiling from 1.05x to 1.24x max concurrency (138,064 → 162,144 KV cache tokens), a real but modest gain — at ~91K-token prompts both precisions are still effectively single-request-at-a-time.
5. **Next session:** download and benchmark `Qwen/Qwen3.5-4B`, `microsoft/Phi-4-mini-instruct`, and `google/gemma-4-E4B` against the current baseline — quality and speed, same shapes/methodology as the FP8 comparison. AWQ/int4 as a second quantization data point remains open but lower priority than model comparison now.
6. Then: quality/accuracy eval (Phase E task-specific evals, not generic benchmarks) before trusting any speed numbers for a production decision; task-specific model routing as AI use cases grow, rather than one model for everything.
7. Make CUDA PATH persistent (still shell-session-only); daemonize vLLM (systemd or Docker + NVIDIA Container Toolkit) so it survives reboots.
8. Then: observability (Prometheus/Grafana — GPU utilization/power/VRAM/KV-cache during load is still unmeasured), and the first real agent/tool-calling use case.

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
