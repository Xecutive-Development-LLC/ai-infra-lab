# AI Inference Server — Build Log & Roadmap

**Stack:** RTX 5090 · Proxmox · Ubuntu 24.04 · CUDA 13.2 · PyTorch · vLLM · Qwen3.5-4B-FP8

**Current state:** production switched to `Qwen3.5-4B-FP8` (2026-09-16) with tool-calling enabled, after round 1/2 benchmarks and a Phase E quality eval both favored it over the original `Qwen3-4B` this section documents standing up. §1-§9 below describe the original build; see §2's Current Architecture table and `README.md` for the current model.

> Source of truth: this file is a Markdown transcription of `AI_Inference_Server_Build_Log_and_Roadmap.docx` (kept alongside it in this folder) for easier diffing/editing over time. Update both, or retire the `.docx` once this file is the working copy.

## 1. Executive Snapshot

- Physical server runs Proxmox VE with an NVIDIA GeForce RTX 5090 passed through directly to an Ubuntu inference VM.
- Ubuntu guest has the NVIDIA 595.84 open driver, CUDA Toolkit 13.2, Python 3.12, PyTorch 2.13.0+cu132, uv, and vLLM 0.29.0 installed.
- `Qwen/Qwen3-4B-Instruct-2507` is loaded and served by vLLM on port 8000 through an OpenAI-compatible API.
- Health endpoint and model endpoint both respond successfully; the first chat-completion request completed successfully on the RTX 5090.
- Current configured context length is 32,768 tokens. A 262,144-token attempt failed because KV-cache requirements exceeded available VRAM; the logs estimated a practical upper bound around 138K tokens for that specific memory configuration.
- Observed load test behavior: ~29.3 GiB VRAM resident at idle, 0% GPU utilization idle, and 100% GPU utilization with roughly 435–449 W draw during active generation.
- Observed vLLM log throughput during initial single-request tests ranged from roughly 97 to 163 generated tokens/sec in interval-level measurements. **These are not yet controlled benchmark numbers.**

## 2. Current Architecture

| Layer | Current configuration |
|---|---|
| Physical host | Proxmox VE 9.1.x server with NVIDIA GeForce RTX 5090 (~32 GiB VRAM) |
| Virtual machine | VM 100, `ai-inference`, Ubuntu Server 24.04.4 LTS, q35 + OVMF UEFI |
| CPU / RAM | 16 vCPU cores, 64 GiB RAM, ballooning disabled |
| Storage | 500 GB VM disk; root LV expanded to ~450 GB ext4 |
| Network | VirtIO on vmbr0; static address 192.168.60.157/24 (converted from DHCP — see §7) |
| GPU passthrough | RTX 5090 at host PCI 41:00.0 plus audio function 41:00.1, isolated with IOMMU and bound to vfio-pci |
| Guest GPU driver | NVIDIA 595.84 open driver |
| CUDA | Driver compatibility reports CUDA 13.2; full CUDA Toolkit 13.2 installed at `/usr/local/cuda-13.2` |
| Python environment | Python 3.12.3 virtual environment at `~/llm-env` — kept for ad-hoc model investigation (this project's whole landmine-debugging history); no longer what production runs on, see below |
| Inference stack | PyTorch 2.13.0+cu132 + vLLM 0.29.0 (bare-metal venv); production runs the pinned `vllm/vllm-openai:v0.29.0` Docker image instead (PyTorch 2.13.0+cu130, same vLLM/FlashInfer versions) via Docker + NVIDIA Container Toolkit — see `deploy/README.md` |
| Model | `RedHatAI/Qwen3.5-4B-FP8-dynamic` — switched 2026-09-16 from the original `Qwen/Qwen3-4B-Instruct-2507`(-FP8) after round 1/2 benchmarks and a Phase E quality eval both favored it; see `README.md`'s "Production switched" note |
| API | vLLM OpenAI-compatible server in a Docker Compose stack (`restart: unless-stopped` + healthcheck-driven `autoheal`, survives crashes and host reboots), `http://0.0.0.0:8000`, tool-calling enabled (`--enable-auto-tool-choice --tool-call-parser qwen3_xml`) — Phase C, see `deploy/README.md` |
| Configured context | 131,072 tokens (128K) — raised from the original 32,768; see §10 Phase A |

## 3. Work Completed

### 3.1 Proxmox VM and Ubuntu
- Created VM 100 (`ai-inference`) with q35 machine type, OVMF UEFI, QEMU guest agent, 16 cores, 64 GiB RAM, 500 GB SCSI disk, and VirtIO networking.
- Installed Ubuntu Server 24.04.4 LTS and OpenSSH.
- Expanded the root logical volume to approximately 450 GB.
- Confirmed the VM is reachable over SSH from the Mac terminal.

### 3.2 GPU Passthrough
- Confirmed AMD-Vi/IOMMU support on the Proxmox host.
- Identified the RTX 5090 GPU and its HDMI/DisplayPort audio function in the same IOMMU group.
- Blacklisted Nouveau on the host and configured vfio-pci binding for the GPU and audio PCI IDs.
- Updated initramfs and rebooted the Proxmox host.
- Added the raw PCI device to the Ubuntu VM with All Functions enabled.
- Confirmed both NVIDIA functions are visible inside the guest using `lspci`.

### 3.3 NVIDIA Driver and CUDA
- Used `ubuntu-drivers` to identify `nvidia-driver-595-open` as the recommended guest driver.
- Installed NVIDIA driver 595.84 and confirmed the GPU through `nvidia-smi`.
- Added NVIDIA's CUDA repository for Ubuntu 24.04.
- Installed `cuda-toolkit-13-2` system-wide.
- Confirmed `/usr/local/cuda -> /usr/local/cuda-13.2` and `nvcc V13.2.86`.
- Exported `/usr/local/cuda/bin` into PATH for the current shell.

### 3.4 Python, PyTorch, uv, and vLLM
- Installed `python3.12-venv` after the first venv creation attempt failed.
- Created and activated `~/llm-env`.
- Upgraded pip and installed `uv`.
- Installed vLLM with `uv` and its compatible PyTorch stack.
- Confirmed `torch.cuda.is_available()` returns `True`.
- Confirmed PyTorch version 2.13.0+cu132 and `torch.version.cuda` reports 13.2.
- Installed `python3.12-dev` after a vLLM JIT compilation failed due to missing `Python.h`.

### 3.5 Model Serving
- Selected the official `Qwen/Qwen3-4B-Instruct-2507` model from Hugging Face.
- Allowed vLLM to download the model automatically from Hugging Face.
- First attempted 262,144-token context; vLLM rejected the configuration because KV cache would require roughly 36 GiB while only about 18.96 GiB was available for KV cache.
- Restarted at 32,768-token context and successfully initialized the engine.
- Confirmed vLLM exposes `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, and related routes.
- Validated `/health` with HTTP 200 OK.
- Validated `/v1/models` and confirmed `Qwen/Qwen3-4B-Instruct-2507` with `max_model_len` 32768.
- Sent the first successful `/v1/chat/completions` request and received a Qwen-generated response.

## 4. Key Commands Used

This is a practical command record, not an exhaustive shell history.

```bash
# Create/activate Python environment
python3 -m venv ~/llm-env
source ~/llm-env/bin/activate

# Install venv support
sudo apt install python3.12-venv

# Install uv
pip install uv

# Install vLLM
uv pip install vllm --torch-backend=auto

# Check CUDA access from PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Install Python headers
sudo apt install python3.12-dev

# Check nvcc
nvcc --version

# Temporary CUDA PATH export
export PATH=/usr/local/cuda/bin:$PATH

# Launch vLLM
vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 32768

# Health check
curl http://localhost:8000/health

# List served models
curl http://localhost:8000/v1/models

# GPU monitor
watch -n 0.5 nvidia-smi
```

## 5. What Happens During an Inference Request

| Step | Stage | What happens |
|---|---|---|
| 1 | Client sends an HTTP request | `curl` sends JSON to vLLM's OpenAI-compatible endpoint |
| 2 | vLLM parses the request | The server reads the model name, messages, sampling parameters, and token limits |
| 3 | Prompt is tokenized | The natural-language prompt is converted into token IDs. The JSON itself is not what gets tokenized |
| 4 | Tokens become tensors | Token IDs are represented as tensors and passed into the model execution path |
| 5 | PyTorch + CUDA drive GPU execution | PyTorch dispatches GPU operations through CUDA; vLLM orchestrates efficient serving and scheduling |
| 6 | GPU performs transformer math | Model weights, activations, and KV-cache state live in VRAM while CUDA kernels execute matrix operations and attention |
| 7 | Model produces logits and selects tokens | The model predicts a distribution for the next token, sampling/decoding selects one, then the process repeats autoregressively |
| 8 | Tokens are decoded to text | Generated token IDs are converted back into readable text |
| 9 | vLLM returns the HTTP response | The response is packaged as OpenAI-compatible JSON and returned to the client |

## 6. Core Concepts Learned

| Concept | Working definition |
|---|---|
| IOMMU | Provides DMA/device isolation and address translation so a physical PCIe device can be safely assigned to a VM |
| VFIO | Linux framework used to bind and expose the physical GPU to the VM; it relies on IOMMU rather than bypassing it |
| Nouveau | Open-source NVIDIA Linux driver. It was removed from ownership of the GPU on the Proxmox host so vfio-pci could own the device |
| NVIDIA guest driver | The kernel/user-space driver inside Ubuntu that makes the passed-through physical GPU usable |
| CUDA | NVIDIA GPU-computing platform/programming model and ecosystem. It is not itself an LLM inference server |
| CUDA Toolkit | Developer toolchain including nvcc, headers, and libraries. The driver can support CUDA workloads without nvcc; vLLM needed nvcc for JIT compilation in this setup |
| PyTorch | General machine-learning tensor/framework layer that can execute operations on CUDA-enabled GPUs |
| vLLM | High-throughput LLM serving engine with scheduling, KV-cache management, continuous batching, and OpenAI-compatible APIs |
| VRAM | GPU memory used for model weights, activations, KV cache, CUDA graph/runtime allocations, and other inference state |
| KV cache | Temporary attention key/value state used to avoid recomputing previous tokens during autoregressive generation. It is runtime state, not permanent conversation memory |
| Activations | Temporary intermediate tensors produced while the neural network executes |
| CUDA graphs | Captured GPU operation sequences that reduce repeated CPU kernel-launch overhead, at the cost of some memory overhead |
| Context window | Maximum number of input + output tokens the model/server configuration can handle for a request. Longer context generally requires more KV-cache memory |

## 7. Problems Encountered and Fixes

| Issue | Why it happened | Fix |
|---|---|---|
| `python3 -m venv` failed | `python3.12-venv` was missing | Installed `python3.12-venv`, then recreated the environment |
| vLLM JIT failed: `Python.h` missing | Python development headers were absent | Installed `python3.12-dev` |
| vLLM JIT failed: `nvcc` not found | NVIDIA driver was installed, but the full CUDA Toolkit was not. `nvidia-smi`'s CUDA version only indicated driver compatibility | Installed NVIDIA CUDA Toolkit 13.2 and exported `/usr/local/cuda/bin` into PATH |
| 262K context failed | KV-cache requirement was about 36 GiB, larger than the ~18.96 GiB vLLM had available for KV cache | Reduced `--max-model-len` to 32768 |
| `curl` reported `Could not resolve host: hcurl` | An accidental extra string was included before the valid curl invocation | Ignored the malformed fragment; the subsequent valid request succeeded |
| VM's DHCP address changed on restart (`.81` → `.157`), breaking the assumed SSH/API endpoint | No DHCP reservation; guest used a dynamic lease that wasn't guaranteed to persist across a host/VM restart | Converted the guest to a static IP via netplan (`/etc/netplan/50-cloud-init.yaml`, `dhcp4: no`) and disabled cloud-init's network management (`/etc/cloud/cloud.cfg.d/99-disable-network-config.cfg`) so it doesn't get silently reverted on next boot |
| `mistralai/Ministral-3-8B-Instruct-2512` failed to import (`ImportError: cannot import name 'PixtralRotaryEmbedding'`) | `transformers`/vLLM version-skew: transformers 5.17.0 renamed/removed two symbols vLLM 0.29.0's `pixtral.py` still imports unconditionally at module load, even for text-only models | **Standing fix, still in place**: `~/llm-env/lib/python3.12/site-packages/zzz_pixtral_shim.pth` + `ai_infra_lab_pixtral_shim.py` — a venv-scoped compatibility shim. Delete both files once vLLM ships a release matching current transformers names. Full root cause in `docs/MODEL_COMPARISON_ROUND2_RESULTS.md`'s "Follow-up (2026-09-16)" section |
| `RedHatAI/Qwen3.5-4B-FP8-dynamic` crashed on the production switch: `RuntimeError: FlashInfer backend is not available` (the same `arch=sm120` `xqa` decode-kernel gap documented in `MODEL_COMPARISON_ROUND2_RESULTS.md`) — but only with `--enable-auto-tool-choice` and `--kv-cache-dtype fp8` **combined**; each flag works fine alone on this exact model | Enabling tool-calling appears to change vLLM's internal kernel/backend selection in a way that now routes FP8-KV-cache decode through the broken FlashInfer `xqa` kernel, even on the one model where plain `--kv-cache-dtype fp8` (no tool-calling) was previously confirmed working | Shipped production without `--kv-cache-dtype fp8` — tool-calling is non-negotiable (the whole point of the switch), the KV-cache win is not. Confirmed by isolating: tool-calling alone at 128K loads and serves correctly; re-adding `--kv-cache-dtype fp8` alongside it reproduces the crash immediately. Not investigated further than this (three separate FlashInfer/sm120 mitigation attempts already failed earlier the same day for a different model combination — see the round 2 doc) |

## 8. Current Measurements and Observations

| Metric | Observed | Interpretation |
|---|---|---|
| Idle GPU utilization | 0% | Expected: model can remain loaded while no kernels are actively executing |
| Idle power | ~19 W in initial idle capture | Low compute activity even though VRAM remained allocated |
| Idle temperature | ~38 °C | Healthy idle reading |
| VRAM while server loaded | ~29,334 MiB / 32,607 MiB | vLLM holds model/runtime allocations and reserves a large memory pool |
| Active GPU utilization | 100% | The longer generation workload fully utilized the GPU during sampled intervals |
| Active power | ~435–449 W | Large jump from idle confirms active GPU compute |
| Active temperature | ~49–55 °C during the captured run | Temperature rose under sustained inference load and later declined |
| Performance state | P1 during load | GPU moved into a high-performance state |
| Interval-level vLLM generation throughput | ~97.5, 163.1, 139.4 tokens/sec observed in logs | Useful evidence of performance, but not yet a controlled benchmark |
| KV-cache utilization during short tests | Small percentages (e.g., ~0.8–1.9% in log snapshots) | Short prompts are using only a small fraction of the reserved KV-cache capacity |

> **Important:** high VRAM occupancy with 0% GPU utilization is not a contradiction. Memory residency and active compute utilization are different metrics.

## 9. Resume Bullet — Current Safe Version

> Built and deployed a self-hosted LLM inference stack on an NVIDIA RTX 5090, configuring GPU passthrough, CUDA 13.2, PyTorch, and vLLM to serve Qwen3-4B through an OpenAI-compatible API; optimized and validated GPU memory allocation across model weights, KV cache, activations, and runtime overhead for long-context inference.

Do not add a maximum-context or tokens/sec claim yet. Replace this with benchmarked figures only after controlled testing.

## 10. What Still Needs To Be Done

### Phase A — Finish the Baseline
- ~~Run a controlled single-request benchmark that reports time-to-first-token (TTFT), end-to-end latency, output tokens/sec, prompt processing throughput, and total generated tokens.~~ **Done** — `benchmarks/bench_baseline.py`.
- ~~Repeat the same benchmark multiple times and record median/p50 and tail/p95 values rather than relying on one run.~~ **Done**.
- ~~Test several prompt/output shapes: short prompt + short output, short prompt + long output, long prompt + short output, and long prompt + long output.~~ **Done** — all four shapes.
- ~~Find a stable maximum context configuration empirically (for example 64K, 96K, 128K) rather than assuming the theoretical estimate is production-safe.~~ **Done** — server restarted at `--max-model-len 131072` (128K), which loads and serves successfully. But "stable" needs a caveat: the server's own startup log reports `GPU KV cache size: 138,064 tokens, Maximum concurrency for 131,072 tokens per request: 1.05x` — confirmed empirically (a genuine ~91K-token prompt costs ~17.6s TTFT, and a second concurrent one of similar size doesn't add throughput, just doubles the wait). 128K context is usable for one request at a time on this GPU, not for concurrent long-context traffic. See `benchmarks/README.md`.

### Phase B — Concurrency and vLLM Behavior
- ~~Send 2, 4, 8, and higher concurrent requests and measure aggregate throughput plus per-request latency.~~ **Done** — `benchmarks/bench_concurrency.py`, swept 1→512.
- ~~Observe vLLM continuous batching and scheduling behavior.~~ **Done, empirically** — for a trivial (~19-token) prompt, throughput scales near-linearly to 32, sub-linearly to a peak of ~15,400 tok/s at concurrency 256, then *regresses* past that (queuing, not an OOM/error wall — zero request failures even at 512 concurrent). **This number is highly prompt-size dependent, not a general server capacity figure** — see below and `benchmarks/README.md`.
- Monitor GPU utilization, power, VRAM, KV-cache usage, request queue depth, and throughput while concurrency rises. — **not done**; this is client-side-only data so far. Deferred to Phase D (vLLM `/metrics` + `nvidia-smi` → Prometheus/Grafana) rather than bolted onto the concurrency script.
- ~~Determine the concurrency point where throughput stops scaling or latency becomes unacceptable.~~ **Done, and the answer is "it depends entirely on prompt size"**: knee at concurrency 256 for ~19-token prompts (~15,400 tok/s peak), concurrency 32 for ~3.3K-token prompts (~1,200 tok/s peak), concurrency 8 for ~9K-token prompts (~180 tok/s peak). (First-pass numbers for the two longer shapes were initially overstated by a prefix-caching methodology bug — corrected same day; see `benchmarks/README.md`.) Prompt size, not request count, is the real capacity constraint on this GPU.

### Phase C — Make the Server Persistent and Production-Friendly
- ~~Make CUDA PATH persistent in the shell environment so `nvcc` is available in fresh sessions.~~ **Done (2026-09-16)** — `export PATH=/usr/local/cuda/bin:$PATH` added to `~/.bashrc`, verified in a real interactive shell. Now only matters for host-level ad-hoc debugging (bare-metal `vllm serve` investigations like this project's whole landmine history) — production itself runs inside a container with its own bundled CUDA toolkit and never touches host PATH.
- ~~Run vLLM under systemd, Docker, or another supervisor so inference survives SSH disconnects and restarts automatically after reboot.~~ **Done (2026-09-16), Docker + NVIDIA Container Toolkit** — see `deploy/README.md` for the full setup and verification. `restart: unless-stopped` + a `willfarrell/autoheal` companion container for health-based restarts; `docker.service` already enabled on boot. Verified crash-recovery via `RestartCount` incrementing on a real (non-zero-exit) crash; did *not* verify via a literal host reboot (see `deploy/README.md` for why) — rests on Docker's own documented `unless-stopped` + boot-enabled-daemon behavior instead.
- ~~Add authentication/API-key protection before exposing the service beyond localhost/trusted network.~~ **Done (2026-09-17)** — `docker-compose.yml` reads `VLLM_API_KEY` from `.env` (`env_file:`, gitignored, key generated and written entirely server-side via `openssl rand -hex 32`, never displayed or logged). Deployed and verified live: `curl /v1/models` returns 401 without a key, 200 with the correct one; `curl /health` and `curl /metrics` confirmed to return 200 with no key, exactly matching the documented scope (only `/v1`, `/v2`, `/inference` are gated). Neither the Operator Portal's AI features nor the agentic-orchestration layer are live yet, so this caused no active outage — but it's a real prerequisite: both will need this same key in their own config before they can call this server at all once they go live. See `deploy/README.md`'s "Authentication" section.
- Bind or firewall the API appropriately; avoid exposing port 8000 publicly without authentication and network controls. — **still not done**, deliberately out of scope for this pass (see `deploy/README.md`); acceptable while LAN-only, no reverse proxy or TLS termination exists yet if that changes.
- ~~Add structured logging and log rotation.~~ **Done (2026-09-17)** — both `vllm` and `autoheal` services now use the `json-file` driver with `max-size: 10m`/`max-file: 3` (30MB cap per container) instead of Docker's unbounded default. See `deploy/README.md`'s "Logging" section.
- ~~Add health/readiness monitoring and automatic restart policies.~~ **Done (2026-09-16)** — `/health`-based Docker healthcheck (`start_period: 240s` to tolerate cold-start compile time) + `autoheal` restarting anything Docker reports unhealthy. Covers both "process exited" and "process alive but wedged" failure modes.

### Phase D — Observability and Benchmarking
- ~~Collect vLLM metrics from its metrics endpoint and feed them into Prometheus/Grafana or an equivalent monitoring stack.~~ **Stack built (2026-09-17), not yet deployed** — `deploy/observability-compose.yml`: Prometheus + Grafana + `utkuozdemir/nvidia_gpu_exporter` (chosen over NVIDIA DCGM, which targets datacenter/Kubernetes GPU fleets — overkill for one consumer card). Kept as a separate Compose file from production, deliberately: independent lifecycle, and Prometheus scrapes vLLM's already-published port 8000 via `host.docker.internal` rather than joining production's Docker network by name. Confirmed via vLLM's own `--help` text that `/metrics` is not gated by the `--api-key` flag Phase C introduces (only `/v1`, `/v2`, `/inference` are), so no auth wiring is needed here regardless of Phase C's state. Grafana's datasource and vLLM's own official dashboard (vendored from vLLM's `examples/observability/prometheus_grafana/grafana.json` at the `v0.29.0` tag) are provisioned automatically; the GPU dashboard (community ID `14574`) is a documented one-time manual import. See `deploy/README.md`'s "Observability (Phase D)" section. Still needs an actual `docker compose -f observability-compose.yml up -d` on the VM to go live.
- GPU utilization/thermals/power tracking during load — mechanism is in place per above (the GPU exporter), not yet observed under real traffic since the stack isn't deployed yet.
- Track TTFT, inter-token latency, request latency, tokens/sec, queue time, batch size, KV-cache occupancy — vLLM's native `/metrics` already exposes most of these; the vendored Grafana dashboard graphs them. Verify coverage once deployed; add custom panels for anything the stock dashboard is missing.
- ~~Create a repeatable benchmark harness so model/version/config changes can be compared apples-to-apples.~~ **Done, well before this phase** — `benchmarks/bench_baseline.py` / `bench_concurrency.py`, used consistently across every model comparison round in this doc.
- ~~Record hardware, model precision/quantization, context size, concurrency, and sampling parameters with every benchmark result.~~ **Done** — see `benchmarks/README.md`'s and `docs/MODEL_COMPARISON*_RESULTS.md`'s result tables throughout.

### Phase E — Model Evaluation
- Download and test additional model families and sizes that fit the RTX 5090. — **research done, downloads not started.** Initially considered bigger models (Qwen3.6-27B, Mistral-Small-3.2-24B, Qwen3.6-35B-A3B MoE, all via AWQ int4) but real VRAM math ruled them out: at 22-25GB real weight size, they leave only ~2-6GB for KV cache, cutting context capacity from 128K+ down to ~23K-80K tokens (worse for the MoE model specifically, whose thin ~2GB activation margin risks a failed startup). Since context headroom matters more than raw parameter count for the target use case (growing tool-calling conversations), pivoted to same-size-class candidates instead: **`Qwen/Qwen3.5-4B`** (same family, newer gen, 262K native context, 9.3GB), **`microsoft/Phi-4-mini-instruct`** (different family, function-calling-focused, 128K native context, 7.7GB), **`google/gemma-4-E4B`** (different family, multimodal-capable, 128K native context, 16GB). All three verified via HuggingFace config/file-size checks, all comfortably preserve the current model's context capability. `ministral/Ministral-4b-instruct` was considered and dropped — fits VRAM fine but is architecturally capped at 32K context, well short of what's needed. Next session: download and benchmark these three against the Qwen3-4B(-FP8) baseline.
- **Re-checked 2026-09-16** against a fresh round of "best model for RTX 5090" recommendations (generic web advice suggesting 27B-70B quantized models). Verified real numbers rather than trusting the advice at face value: `Qwen/Qwen3.8-27B-FP8` (a genuinely newer release than the original check), `Qwen/Qwen3.6-27B-FP8`, and `Qwen/Qwen3.5-27B-GPTQ-Int4` all land at ~30-31GB real weight size (same hybrid linear+full-attention family as the champion, just scaled up, plus a vision tower) — leaves ~0-1GB of this GPU's 31.36 GiB for KV cache/activations, likely failing to load at all under vLLM's default memory profiling. `casperhansen/llama-3.3-70b-instruct-awq` is 39.8GB — larger than total VRAM before any other overhead; any 70B-class model is categorically out for this single 32GB card, at any quantization. `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` genuinely fits (19.3GB, ~12GB headroom) but is hard-capped at 32,768 native context — same disqualifier as Ministral-4b above, and it's coding-specialized besides. `unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit` clears both VRAM (19.2GB) and context (131K native) but is a reasoning-distillation model (long hidden chain-of-thought before every answer — the opposite of what a low-latency tool-calling assistant needs) using `bitsandbytes` quantization (weaker vLLM kernel support than AWQ/GPTQ). **Conclusion: the generic advice is calibrated for short-context enthusiast chat use, a different profile than this project's 128K tool-calling target — none of it changes the recommendation.** Qwen3.5-4B-FP8 remains the front-runner. Same session: cleaned up the HF cache of every confirmed-broken/ruled-out download from rounds 1-2 (Falcon-H1-7B, Nemotron-H FP8, Gemma-4-E4B-it BF16+FP8, Granite-4.0-H-Micro AWQ, LFM2.5-2.6B FP8) — freed ~55GB (46% → 33% disk used). Kept every model that actually works (even non-champions) as reference/fallback, and both variants of the current production model and the front-runner.
- ~~Compare quality, speed, VRAM use, context length, and tool-calling ability.~~ **Fallback-candidate pass done (2026-09-17)** — speed/VRAM/context were already covered in round 1; ran the missing piece, the quality/tool-calling gate, against `Qwen/Qwen3.5-4B` (BF16) and `RedHatAI/Phi-4-mini-instruct-FP8-dynamic`. `google/gemma-4-E4B-it` dropped entirely — fails on any prompt >~30 tokens, a real vLLM bug, not worth chasing for a fallback pick. **Qwen3.5-4B-BF16: 36/42 (85.7%)**, safety subset 6/12 — a workable same-family fallback, but with a real zone-resolution regression versus the FP8 champion (not just a precision difference). **Phi-4-mini-instruct-FP8: 24/42 (57.1%)**, safety subset 9/12 (ties the champion) — the low score is a genuine `phi4_mini_json` parser/output-format mismatch (the model reliably emits its own JSON tool-call format instead of the `functools[...]` marker the parser expects), not a tool-selection failure. Not currently viable as a fallback. Full writeup: `docs/PHASE_E_FALLBACK_CANDIDATES_EVAL_RESULTS.md`.
- Build task-specific evals for the company use cases rather than choosing models from generic benchmarks alone. — not started.
- ~~Test quantized variants where useful and measure the quality/performance tradeoff.~~ **Done** — FP8 tested extensively (baseline, concurrency, and at 128K context) against BF16 for `Qwen3-4B-Instruct-2507`; looks like a strict upgrade with a documented RTX-5090-specific workaround. See `benchmarks/README.md`.

### Phase F — Application / Agent Layer
- Build the first internal client that calls the OpenAI-compatible vLLM API instead of raw curl.
- Implement system prompts, structured outputs, tool/function calling, validation, retries, and safe permission boundaries.
- Connect the first target use case: snow-zone trigger updates/history retrieval with explicit safe tool calls and audit logs.
- Add persistent conversation/application state outside the model; do not confuse application memory with the transient KV cache.
- Later extend to customer/operator SMS responses, sales-portal assistant, customer portal, and other internal systems.

### Phase G — Advanced Inference Optimization
- Tune `gpu-memory-utilization`, context length, batch limits, scheduler settings, and KV-cache configuration only after baseline numbers exist.
- Evaluate prefix caching where workloads share large prompt prefixes.
- Test quantization and alternative dtypes for larger models or more concurrency.
- Evaluate tensor parallelism only if additional GPUs are introduced; the current RTX 5090 is one GPU, not multiple GPUs.
- Compare vLLM against alternatives only for a concrete reason (for example TensorRT-LLM, llama.cpp, or specialized serving paths).

## 11. Exact Next Session Plan

1. Confirm the vLLM server is still healthy.
2. Create a repeatable benchmark command/script instead of manually reading interval log lines.
3. Capture a clean single-request baseline.
4. Run 2 concurrent requests, then 4 concurrent requests.
5. Record aggregate throughput and latency changes.
6. Review what continuous batching is doing and explain it in interview language.
7. Make CUDA PATH persistent.
8. Choose whether to daemonize vLLM with systemd or containerize it with Docker + NVIDIA Container Toolkit.
9. Only after the serving layer is stable, move to model evaluation and the first real company agent/tool-calling use case.

## 12. Interview-Ready Talking Points

**How did you pass the GPU into the VM?**
I enabled IOMMU on the Proxmox host, removed the GPU from the host graphics driver, bound the RTX 5090 and its audio function to vfio-pci, then passed the PCIe device directly into an Ubuntu VM. Inside the VM, the NVIDIA driver exposes the physical GPU to CUDA workloads.

**Why did `nvidia-smi` work before `nvcc`?**
`nvidia-smi` comes with the NVIDIA driver and its reported CUDA version represents driver compatibility. `nvcc` is part of the CUDA Toolkit, which is a separate developer toolchain and had not yet been installed.

**Why can VRAM be almost full while GPU utilization is 0%?**
The model weights and vLLM memory pools can remain resident in VRAM while the GPU is idle. GPU utilization measures active compute, not memory occupancy. During inference I observed utilization jump to 100% while VRAM stayed roughly constant.

**Why did 262K context fail even though the 4B model weights fit?**
Weights are only one part of the memory budget. Long context dramatically increases KV-cache requirements. The model weights used roughly 7.6 GiB, but the requested 262K context required about 36 GiB of KV cache, exceeding the memory available for KV cache on the 32 GiB GPU.

**What is vLLM doing for you?**
It is the serving engine around the model: it exposes OpenAI-compatible APIs, schedules requests, manages KV cache, batches work efficiently, and drives model execution through the PyTorch/CUDA stack.

## 13. Current Status

Milestone reached: the full self-hosted inference path is working end-to-end.

- Hardware virtualization and GPU passthrough: **working**
- NVIDIA driver and CUDA Toolkit: **working**
- PyTorch sees the GPU: **working**
- vLLM engine: **working**
- Qwen3-4B model loading: **working**
- OpenAI-compatible API: **working**
- First inference and GPU-load validation: **working**
- Controlled benchmarking, concurrency testing, persistence, security, observability, evals, and application integration: **next**
