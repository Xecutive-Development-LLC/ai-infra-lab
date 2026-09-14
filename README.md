# AI Infra Lab

A running log of learnings, work, and discoveries from building and operating self-hosted AI infrastructure — inference servers, benchmarking, evals, and the agent/application layer on top of them.

## Current Project: Self-Hosted LLM Inference Server

Self-hosted, OpenAI-API-compatible LLM inference stack on bare-metal GPU hardware, built from the ground up: hypervisor → GPU passthrough → driver/CUDA → Python/PyTorch → vLLM → served model.

**Stack:** RTX 5090 · Proxmox VE · Ubuntu 24.04 · CUDA 13.2 · PyTorch 2.13 · vLLM 0.29 · Qwen3-4B-Instruct

**Status:** End-to-end path is working — GPU passthrough, driver/CUDA, PyTorch, vLLM, and the OpenAI-compatible API have all been validated. The VM now has a static IP (`192.168.60.157`, survives reboot) instead of DHCP. Phase A (single-request baseline) and Phase B (concurrency scaling) are both done — see [`benchmarks/`](benchmarks/): peak throughput ~15,400 tok/s at concurrency 256, safe steady-state ceiling ~128–192 before latency degrades. Persistence (systemd/Docker), security, observability, model evals, and the application/agent layer are next.

Full build log, architecture, commands, concepts learned, problems/fixes, and the phased roadmap (A–G) live in:
- [`docs/AI_Inference_Server_Build_Log_and_Roadmap.md`](docs/AI_Inference_Server_Build_Log_and_Roadmap.md) — working copy, kept current
- [`docs/AI_Inference_Server_Build_Log_and_Roadmap.docx`](docs/AI_Inference_Server_Build_Log_and_Roadmap.docx) — original source doc

### Immediate next steps
1. ~~Build a repeatable benchmark harness (TTFT, latency, tokens/sec, p50/p95).~~ Done — [`benchmarks/bench_baseline.py`](benchmarks/bench_baseline.py).
2. ~~Test concurrency (2/4/8+ requests) and observe vLLM continuous batching behavior.~~ Done — [`benchmarks/bench_concurrency.py`](benchmarks/bench_concurrency.py).
3. Make CUDA PATH persistent (still shell-session-only); daemonize vLLM (systemd or Docker + NVIDIA Container Toolkit) so it survives reboots.
4. Then: observability (Prometheus/Grafana — GPU utilization/power/VRAM/KV-cache during load is still unmeasured), model evals, and the first real agent/tool-calling use case.

## Repo Structure

```
docs/       Build logs, architecture notes, roadmaps for each project
benchmarks/ Benchmark scripts, harnesses, and results (as they're built)
```

## Conventions

- Every benchmark result is recorded with hardware, model, precision/quantization, context size, concurrency, and sampling parameters — no bare numbers.
- Work happens on feature branches merged via PR; `main` stays the record of what's actually been validated.
