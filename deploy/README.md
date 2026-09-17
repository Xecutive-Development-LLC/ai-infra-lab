# Production Deployment (Phase C)

The inference server runs as a Docker Compose stack on the VM
(`192.168.60.157`), not a manually-launched `nohup vllm serve` process — that
was the setup for all of Phases A/B/D/E's research and benchmarking, but
production now depends on this server for real (Operator Portal), so it
needed to survive reboots and crashes without someone SSHing in.

## What's here

`docker-compose.yml` — the actual deployed config, kept in sync with what's
running on the VM at `~/vllm-deploy/docker-compose.yml`. Two services:

- **`vllm`** — `vllm/vllm-openai:v0.29.0` (pinned to match the version this
  project's whole history of RTX-5090-specific landmines was found and fixed
  against — see `docs/AI_Inference_Server_Build_Log_and_Roadmap.md` section 7
  and `docs/MODEL_COMPARISON_ROUND2_RESULTS.md`). Serves
  `RedHatAI/Qwen3.5-4B-FP8-dynamic` at 128K context with tool-calling
  enabled. `restart: unless-stopped` plus a `/health` healthcheck.
- **`autoheal`** ([willfarrell/autoheal](https://github.com/willfarrell/docker-autoheal)) —
  watches any container labeled `autoheal=true` and restarts it if Docker
  reports it unhealthy. `restart: unless-stopped` alone only recovers a
  container whose process actually exited; it does nothing for a process
  that's alive but wedged/not responding, which is exactly what the
  healthcheck exists to catch.

## Prerequisites (already done on this VM, documented here for a rebuild)

Docker Engine + NVIDIA Container Toolkit, installed via the official
install docs for Ubuntu 24.04:

```bash
# Docker Engine
apt-get install -y ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list
apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
usermod -aG docker <user>

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Verify GPU access from a container
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

`docker.service` is enabled by the package install (`systemctl is-enabled
docker` → `enabled`), so the daemon itself starts on boot without any extra
step.

## Deploying

```bash
mkdir -p ~/vllm-deploy && cd ~/vllm-deploy
# copy docker-compose.yml here
docker compose up -d
```

First boot takes ~3.5 minutes (model load + a cold `torch.compile` +
FlashInfer autotune, all uncached). Every restart after that reuses the
`vllm-compile-cache` named volume and takes ~75 seconds — confirmed by timing
a real restart (`210s` cold vs. `76s` warm).

## Verified behavior (2026-09-16)

- **Container image parity with the bare-metal venv**: same vLLM 0.29.0,
  same FlashInfer 0.6.18 (only the CUDA point version differs, 13.0 vs.
  13.2 — no observed effect). Confirmed identical KV-cache size (~636K
  tokens @ 128K), identical FP8 kernel selection
  (`CutlassFP8ScaledMMLinearKernel`, not the broken DeepGEMM path), and a
  correct live tool-calling response before cutting over from bare-metal.
- **Crash recovery**: a container whose process exits unexpectedly (tested
  with a plain container that self-exits non-zero) is restarted
  automatically by `unless-stopped` — confirmed via `RestartCount`
  incrementing.
- **`docker kill`/`docker stop` do NOT auto-restart** while the Docker
  daemon keeps running — this is `unless-stopped`'s designed behavior
  (distinguishes "operator deliberately stopped this" from "it crashed"),
  not a gap. Confirmed directly: killing the `vllm` container left it
  `Exited` with `RestartCount=0` for 150+ seconds, no auto-recovery,
  exactly as documented. Restarting it is `docker compose up -d vllm`.
- **Survives a host reboot**: not tested with a literal reboot (too
  disruptive to trigger casually on a box with its own history of
  reboot-related networking issues — see the static-IP fix in the roadmap
  doc's problems table). Rests on Docker's own well-documented behavior:
  `unless-stopped` containers restart when the daemon starts, regardless of
  whether they were in a manually-stopped state before the daemon last
  stopped — combined with `docker.service` being enabled at the systemd
  level, a host reboot should bring the whole stack back without manual
  intervention. Worth an actual reboot test if/when a reboot is needed for
  another reason anyway, rather than triggering one solely to prove this.

## Rolling back to bare-metal (if ever needed)

The original `~/llm-env` venv is untouched and still has every model this
project has downloaded and tested. `vllm serve <model> --max-model-len N
[--kv-cache-dtype fp8] [--enable-auto-tool-choice --tool-call-parser X]`
with `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0` set works exactly as
documented throughout this repo's history — useful for any future one-off
model investigation (as opposed to production serving, which should stay on
the Compose stack).

**Gotcha confirmed 2026-09-17 (Phase E fallback-candidate eval):** for
`microsoft/Phi-4-mini-instruct` / `RedHatAI/Phi-4-mini-instruct-FP8-dynamic`,
the tool-call parser is `phi4_mini_json`, and it **requires an explicit
`--chat-template`** override — Phi-4-mini's own tokenizer chat template has
no branch for assistant `tool_calls`/`role: tool` messages, so multi-round
tool-calling silently mis-renders without one. Use the vendored
`deploy/chat_templates/tool_chat_template_phi4_mini.jinja` (copied from
vLLM's own `examples/tool_chat_template_phi4_mini.jinja` at the `v0.29.0`
tag). Even with the correct parser and template, this model/checkpoint
frequently emits its own generic JSON tool-call format instead of the
`functools[...]` marker the parser expects — see
`docs/PHASE_E_FALLBACK_CANDIDATES_EVAL_RESULTS.md` for the full writeup;
not currently reliable for tool-calling in this stack.
