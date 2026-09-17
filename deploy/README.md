# Production Deployment (Phase C)

The inference server runs as a Docker Compose stack on the VM
(`192.168.60.157`), not a manually-launched `nohup vllm serve` process — that
was the setup for all of Phases A/B/D/E's research and benchmarking, but this
is meant to be the real production target for the Operator Portal and the
future agentic-orchestration layer, so it needed to survive reboots and
crashes without someone SSHing in.

**Current state (2026-09-17): idle.** Neither the Operator Portal's AI
features nor the agentic-orchestration layer are live yet — nothing is
currently sending real traffic to this server besides this repo's own
benchmark/eval runs. That means stop/restart windows here carry no active
outage risk today; the auth and observability work below still needs to
land *before* those layers go live, not urgently right now.

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

## Authentication

The `vllm` service reads `VLLM_API_KEY` from `.env` (via `env_file:` in
`docker-compose.yml`) -- vLLM picks this up natively from the environment,
no `--api-key` CLI flag needed. Copy `deploy/.env.example` to
`~/vllm-deploy/.env` on the VM and fill in a real value
(`openssl rand -hex 32`); `.env` is gitignored and must never be committed.

**Scope, confirmed via vLLM's own `--help` text**: the key only protects
endpoints under `/v1`, `/v2`, and `/inference`. `/health` and `/metrics`
stay open regardless -- this is why Phase D's Prometheus scrape config
needs no auth wiring, and why a healthcheck hitting `/health` keeps working
unmodified.

**Prerequisite, not implemented here**: neither the Operator Portal's AI
features nor the LangGraph agentic-orchestration layer are live yet, so
flipping this on today causes no active outage -- but both will need this
same `VLLM_API_KEY` in their own config before they can call this server at
all, once they do go live. Land it in their config as part of standing up
each of those integrations, not as a fire drill after the fact.

## Logging

Both services use the `json-file` driver with `max-size: 10m` /
`max-file: 3` (30MB cap per container) instead of Docker's unbounded
default -- container logs would otherwise grow indefinitely on a
long-lived container. Check current usage with:
```bash
docker inspect --format='{{.LogPath}}' vllm
```

## Deploying

```bash
mkdir -p ~/vllm-deploy && cd ~/vllm-deploy
# copy docker-compose.yml here, plus .env (see Authentication above)
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
