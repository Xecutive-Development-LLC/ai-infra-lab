# AI Inference Server — Build Log & Roadmap

**Stack:** RTX 5090 · Proxmox · Ubuntu 24.04 · CUDA 13.2 · PyTorch · vLLM · Qwen3-4B

**Current state:** first self-hosted LLM is running successfully through an OpenAI-compatible API.

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
| Python environment | Python 3.12.3 virtual environment at `~/llm-env` |
| Inference stack | PyTorch 2.13.0+cu132 + vLLM 0.29.0 |
| Model | `Qwen/Qwen3-4B-Instruct-2507` |
| API | vLLM OpenAI-compatible server on `http://0.0.0.0:8000` |
| Configured context | 32,768 tokens |

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
- Find a stable maximum context configuration empirically (for example 64K, 96K, 128K) rather than assuming the theoretical estimate is production-safe. — **not done yet**.

### Phase B — Concurrency and vLLM Behavior
- ~~Send 2, 4, 8, and higher concurrent requests and measure aggregate throughput plus per-request latency.~~ **Done** — `benchmarks/bench_concurrency.py`, swept 1→512.
- ~~Observe vLLM continuous batching and scheduling behavior.~~ **Done, empirically** — throughput scales near-linearly to 32, sub-linearly to a peak of ~15,400 tok/s at concurrency 256, then *regresses* past that (queuing, not an OOM/error wall — zero request failures even at 512 concurrent). See `benchmarks/README.md`.
- Monitor GPU utilization, power, VRAM, KV-cache usage, request queue depth, and throughput while concurrency rises. — **not done**; this is client-side-only data so far. Deferred to Phase D (vLLM `/metrics` + `nvidia-smi` → Prometheus/Grafana) rather than bolted onto the concurrency script.
- ~~Determine the concurrency point where throughput stops scaling or latency becomes unacceptable.~~ **Done** — knee at 256 (peak throughput), practical safe ceiling ~128–192 before latency degrades noticeably.

### Phase C — Make the Server Persistent and Production-Friendly
- Make CUDA PATH persistent in the shell environment so `nvcc` is available in fresh sessions.
- Run vLLM under systemd, Docker, or another supervisor so inference survives SSH disconnects and restarts automatically after reboot.
- Add authentication/API-key protection before exposing the service beyond localhost/trusted network.
- Bind or firewall the API appropriately; avoid exposing port 8000 publicly without authentication and network controls.
- Add structured logging and log rotation.
- Add health/readiness monitoring and automatic restart policies.

### Phase D — Observability and Benchmarking
- Collect vLLM metrics from its metrics endpoint and feed them into Prometheus/Grafana or an equivalent monitoring stack.
- Track TTFT, inter-token latency, request latency, tokens/sec, queue time, batch size, KV-cache occupancy, GPU utilization, VRAM, thermals, and power.
- Create a repeatable benchmark harness so model/version/config changes can be compared apples-to-apples.
- Record hardware, model precision/quantization, context size, concurrency, and sampling parameters with every benchmark result.

### Phase E — Model Evaluation
- Download and test additional model families and sizes that fit the RTX 5090.
- Compare quality, speed, VRAM use, context length, and tool-calling ability.
- Build task-specific evals for the company use cases rather than choosing models from generic benchmarks alone.
- Test quantized variants where useful and measure the quality/performance tradeoff.

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
