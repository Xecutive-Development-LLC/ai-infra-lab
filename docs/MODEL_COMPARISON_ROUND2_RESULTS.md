# Model Comparison Round 2: Full Results

> **Correction (2026-09-15, same day):** this doc originally attributed
> Granite-4.0-H-Micro-AWQ's failures to `--max-model-len` ("broken at 32K,
> clean at 128K"). That was wrong — an artifact of the test matrix, not the
> real cause. The 128K config's baseline only ever runs `xlong_short`/`xlong_long`
> (~91K-token prompts); it was **never actually tested with a short prompt**,
> so "clean at 128K" was a claim about long prompts only, not about the
> config in general. Direct follow-up testing (see "Landmine 4, corrected"
> below) found the real trigger is **prompt length alone, independent of
> `--max-model-len`**: the exact same failure reproduces at 128K too for
> anything under roughly 12,000-13,000 tokens, and the exact same success
> reproduces at 32K for anything above that threshold. Every number in this
> doc is still accurate; only the causal story for Landmine 4 needed fixing.
> Full corrected writeup below, kept rather than silently edited away — same
> policy as the earlier prefix-caching correction in `benchmarks/README.md`.

Round 2 casts a wider net across vLLM's full supported-architectures list
(<https://docs.vllm.ai/en/latest/models/supported_models/>), rather than
just same-family follow-ups. Companion to
[`MODEL_COMPARISON_RESULTS.md`](MODEL_COMPARISON_RESULTS.md) (round 1: Qwen3.5-4B,
Phi-4-mini-instruct, Gemma-4-E4B-it). [`benchmarks/README.md`](../benchmarks/README.md)
has the narrative synthesis; this doc is the reference for every raw number.

## Candidate selection and verification

Five model families were selected from vLLM's real supported-architecture list
(fetched directly, not assumed), filtered to: not already tested (excludes all
Qwen/Phi/Gemma), roughly 1B-14B params, and — given hybrid/linear-attention
architecture was the single strongest predictor of long-context performance in
round 1 — prioritized toward Mamba/SSM/hybrid designs. Every candidate was
verified directly against the HuggingFace API (`gated`, real param count from
`safetensors.total`, `architectures[0]`, `max_position_embeddings` from
`config.json`) and cross-checked against the **live vLLM 0.29 install's
`ModelRegistry`** over SSH, not just the docs page:

| Model | Repo ID | Arch class | Params | Native ctx | Gated? | Quant tested |
|---|---|---|---|---|---|---|
| Falcon-H1-7B-Instruct | `tiiuae/Falcon-H1-7B-Instruct` | `FalconH1ForCausalLM` | 7.59B | 262,144 | No | none found |
| Nemotron-H-8B-Reasoning-128K | `nvidia/Nemotron-H-8B-Reasoning-128K` | `NemotronHForCausalLM` | 8.10B | 131,072 | No | official FP8 |
| Granite-4.0-H-Micro | `ibm-granite/granite-4.0-h-micro` | `GraniteMoeHybridForCausalLM` | 3.19B | 131,072 | No | AWQ (`cyankiwi/granite-4.0-h-micro-AWQ-4bit`) |
| LFM2.5-2.6B | `LiquidAI/LFM2.5-2.6B` | `Lfm2ForCausalLM` | 2.70B | 131,072 | No | FP8 (`vrfai/LFM2.5-2.6B-FP8`) |
| Ministral-3-8B-Instruct-2512 | `mistralai/Ministral-3-8B-Instruct-2512` | `Mistral3ForConditionalGeneration` | 8.92B | 262,144 (claimed) | No | FP8 (`unsloth/...-FP8`) |

Rejected during verification, with the specific real reason:
- **Zyphra/Zamba2-7B-Instruct** — hybrid Mamba2+transformer, architecturally
  appealing, but `config.json` caps `max_position_embeddings` at **4,096**,
  confirmed directly, not a docs typo. Disqualifying for this project's
  long-context priority regardless of architecture.
- **openbmb/MiniCPM-SALA** — best spec on paper (524,288-token native
  context, hybrid sparse+linear attention, 9.48B, ungated) but
  `MiniCPMSALAForCausalLM` is **not present in the installed vLLM 0.29's
  `ModelRegistry`**, confirmed against the live install rather than the docs
  page. Not currently servable.
- `JambaForCausalLM` (AI21) — smallest real release is ~52B total, too big.
  `Exaone4ForCausalLM` (LG) — only 1.2B/32B public sizes, no usable middle
  size. `OlmoHybridForCausalLM` — no released checkpoint found.

vLLM 0.29 ships its own Triton Mamba/SSM kernels
(`vllm/model_executor/layers/mamba/ops/`), confirmed on the remote install —
the external `mamba_ssm`/`causal_conv1d` pip packages are absent and not
required.

## What actually worked — 10 of 18 planned runs

9 model/quant variants × 2 contexts (32K, 128K) = 18 planned runs. **10
servers came up and produced a result file; 3 combinations failed to load at
all (2 different root causes); 2 more loaded but a large fraction of
individual requests failed on specific prompt shapes/lengths** (Granite's
"loaded" runs specifically should not be read as "worked" — see "Landmine 4,
corrected" below for what was actually reliable within them).

| Model | Precision | Status | Notes |
|---|---|---|---|
| Falcon-H1-7B-Instruct | BF16 | **Failed to load, both contexts** | Hard VRAM ceiling — see below |
| Nemotron-H-8B-Reasoning-128K | BF16 | OK, both contexts | |
| Nemotron-H-8B-Reasoning-128K | FP8 (official) | **Failed to load, both contexts** | Weight-loader bug — see below |
| Granite-4.0-H-Micro | BF16 | Loaded both contexts; **unreliable on short prompts** | See "Landmine 4, corrected" |
| Granite-4.0-H-Micro | AWQ | Loaded both contexts; **unreliable under ~12K-token prompts, both contexts** | See "Landmine 4, corrected" |
| LFM2.5-2.6B | BF16 | OK at 32K (exceptional); **100% fails at 128K** | Same signature as Gemma's bug — see below |
| LFM2.5-2.6B | FP8 | OK at 32K (exceptional); **100% fails at 128K** | Same |
| Ministral-3-8B-Instruct | BF16 + FP8 | **Failed to load, all 4 combos** | `transformers`/vLLM version mismatch — see below |

## Landmine 1: Falcon-H1-7B-Instruct doesn't fit this GPU under vLLM 0.29's defaults

BF16 weights alone take ~15.1-15.9 GiB. At server startup, vLLM's CUDA-graph
memory-profiling step tries to allocate a **16.84 GiB** "minimal" KV-cache
buffer to test graph capture — and that number didn't budge across two
different remedies:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.84 GiB.
GPU 0 has a total capacity of 31.36 GiB of which ~16 GiB is free.
```

- Attempt 1: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the exact
  remedy PyTorch's own error message suggests, for fragmentation) — identical
  16.84 GiB allocation attempt, identical failure.
- Attempt 2: `--gpu-memory-utilization 0.75` (down from the 0.92 default, to
  shrink the executor's memory budget) — identical 16.84 GiB allocation
  attempt, identical failure.

Since the target allocation size didn't change under either remedy, this isn't
fragmentation and isn't standard headroom tuning — it looks like the "minimal"
profiling buffer size for this specific hybrid Mamba+attention architecture is
computed independent of `--gpu-memory-utilization`, and it's simply too large
to coexist with ~15GB of BF16 weights on a 31GB GPU. Not root-caused further
than this within the session budget — plausibly a real vLLM sizing
inefficiency for this architecture rather than a fundamental impossibility,
but unresolved. No quantized checkpoint was found for this model to test as
an alternative.

## Landmine 2: Nemotron-H-8B's official FP8 checkpoint fails to load

The BF16 checkpoint of the same model loads and serves perfectly (see results
below). The **official NVIDIA FP8 checkpoint** (`nvidia/Nemotron-H-8B-Reasoning-128K-FP8`)
fails during weight loading:

```
File ".../vllm/model_executor/layers/linear.py", line 756, in weight_loader
    param_data = param.data
AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'
```

This happens inside vLLM's `MergedColumnParallelLinear.weight_loader`, used
for this hybrid architecture's Mamba in-projection layers — looks like a
genuine incompatibility between this FP8 checkpoint's tensor layout and vLLM
0.29's loader for that specific layer type on a hybrid Mamba2 architecture,
not a memory or config issue (fails identically at both 32K and 128K, fails
immediately during weight loading before any memory profiling happens). Worth
retrying on a newer vLLM release.

## Landmine 3: `Mistral3ForConditionalGeneration` doesn't import on this install

Both Ministral-3-8B precisions fail identically, immediately, for all 4
combos — not a model or checkpoint issue, a `transformers`/vLLM version
mismatch:

```
File ".../vllm/model_executor/models/pixtral.py", line 19, in <module>
    from transformers.models.pixtral.modeling_pixtral import (
ImportError: cannot import name 'PixtralRotaryEmbedding' from
'transformers.models.pixtral.modeling_pixtral'. Did you mean:
'PixtralVisionRotaryEmbedding'?
```

`Mistral3ForConditionalGeneration` (the class vLLM uses even though this is
listed as a text-chat model) pulls in Pixtral vision-tower code at import
time — regardless of whether the request uses images — and the installed
`transformers` version renamed the class vLLM's `pixtral.py` still imports by
its old name. This is a version-skew bug between the two packages on this
install, not something fixable by changing serve flags. Blocks this model
entirely until `transformers` or vLLM is updated to match.

## Landmine 4, corrected: the "immediate EOS" bug is about prompt length, not `--max-model-len`

Round 1 found Gemma-4-E4B-it failing 100% of requests past ~30 prompt tokens,
with a `200 OK` / `completion_tokens: 1` / zero-visible-content signature.
**The exact same signature shows up on two more, architecturally unrelated
models this round.**

- **LFM2.5-2.6B (both BF16 and FP8) fails 100% of `xlong_short`/`xlong_long`
  requests at 128K context** (~91-97K-token prompts) — but is completely
  clean at 32K, including its own `vlong_short`/`vlong_long` shapes up to
  ~9.8K real tokens. Not re-investigated after the Granite finding below;
  the same "is it really about `--max-model-len`, or about prompt length
  independent of it?" question applies here too and is still open.
- **Granite-4.0-H-Micro-AWQ** was originally reported as "broken at 32K,
  clean at 128K." **This was wrong.** Direct follow-up investigation (raw
  `curl` requests against a live server, bypassing the benchmark harness
  entirely, to rule out a client-side bug) found:

  1. The 128K config's baseline test only ever sends `xlong_short`/`xlong_long`
     (~91K tokens) — it was never tested with a short prompt. Sending the
     exact same short prompt that fails at 32K to the **128K** server
     produces the **identical failure** (`completion_tokens: 1`, empty text).
     `--max-model-len` was never the variable.
  2. Binary-searching prompt length against a single server (`--max-model-len 32768`,
     unchanged) found a hard crossover: prompts up to ~11,878 tokens fail
     100% of the time; prompts at ~12,813 tokens and above succeed reliably.
     This holds at both 32K and 128K `--max-model-len` — it's the same
     threshold either way.
  3. Ruled out three plausible mechanisms by direct A/B test, each a
     clean retry changing exactly one flag: **prefix caching**
     (`--no-enable-prefix-caching` — still fails), and **CUDA graph
     capture/replay** (`--enforce-eager` — still fails). Neither moved the
     failure at all.
  4. **The unquantized BF16 checkpoint shows the same shape of bug, just far
     narrower**: repeating the ~20-30-token `short_short` shape 5 times with
     different random content, BF16 failed 3/5 times (empty completion,
     same signature) — genuinely non-deterministic per exact prompt content
     at trivial length, matching round 1's original 2/5 partial-failure
     finding for this shape — while its `long_short` (~3.3K tokens) prompt
     succeeded cleanly every time tested. **AWQ shows the identical failure
     mode but the affected range is roughly 400x wider** (unreliable up to
     ~12,000 tokens instead of ~30).

  **Best-supported conclusion**: there's a real, pre-existing fragility in
  `GraniteMoeHybridForCausalLM` at short-to-medium prompt lengths — plausibly
  the Mamba2 state and/or MoE router not having "warmed up" enough signal
  yet on a fresh, short sequence, causing the model to occasionally argmax
  straight to an end-of-sequence token as its very first output. This
  fragility exists in full-precision BF16 (rare, content-dependent, only at
  the shortest shapes) and gets dramatically amplified by this community AWQ
  4-bit quantization (reliable failure across a ~400x wider length range) —
  consistent with a hybrid Mamba/MoE architecture being unusually sensitive
  to quantization noise in exactly the marginal cases where its own
  first-principles behavior was already fragile. Not root-caused further
  than this (would need to inspect actual logits/hidden states at the
  failure boundary, out of scope here) — but this is a materially different,
  better-supported story than "broken at 32K."

**This actually strengthens the case against using Granite-4.0-H-Micro in
production as-is**, not weakens it: a real tool-calling conversation *starts*
short and grows over many turns — it spends most of its life in exactly the
prompt-length range (under ~12K tokens for AWQ, under ~30 tokens even for
BF16) where this model is unreliable. The excellent 128K/91K-token numbers
below are real and reproduced, but a model that can't reliably handle the
early turns of the exact conversation it would eventually see at 91K tokens
isn't usable for this repo's target use case regardless of how good it gets
once a conversation is already enormous.

## Follow-up (2026-09-16): root-causing Landmines 2 and 3

Went back to actually root-cause (not just document) Nemotron-H's FP8 loader
crash and Ministral-3-8B's import failure, since both looked like they might
be fixable rather than fundamental. One fixed, one didn't — and a third,
unrelated finding fell out of testing `--kv-cache-dtype fp8` on both.

### Ministral-3-8B: fixed. `Mistral3ForConditionalGeneration` now loads and serves.

Root cause, precisely: vLLM 0.29.0's `pixtral.py` (imported unconditionally
by the Mistral3 vision-tower code path, even for text-only use) imports two
symbols from `transformers.models.pixtral.modeling_pixtral` that transformers
5.17.0 no longer provides under those names:

1. **`PixtralRotaryEmbedding`** — renamed to `PixtralVisionRotaryEmbedding`.
   Verified identical `__init__(config, device)` signature and `forward()`
   body, so this is a pure rename, not a behavior change.
2. **`position_ids_in_meshgrid`** — deleted outright (confirmed absent
   anywhere in the transformers 5.17.0 source tree, not moved/inlined). Only
   used inside `PixtralVisionModel.forward()` when actually encoding images —
   unreachable for text-only chat completions, but still imported eagerly at
   module load time, so it blocks loading regardless.

Both packages are already at their latest PyPI release (vllm 0.29.0 is
`LATEST`, transformers 5.17.0 is `LATEST`), confirmed via `pip index
versions` on the live install — so this is a genuine unpatched version-skew
bug between the two projects, not something an upgrade fixes.

**Fix applied**: a two-file, venv-scoped shim —
`~/llm-env/lib/python3.12/site-packages/zzz_pixtral_shim.pth` (a one-line
`.pth` loader; `sitecustomize.py` was tried first but is shadowed by
Ubuntu's own `/usr/lib/python3.12/sitecustomize.py`, which loads first on
`sys.path`) plus `ai_infra_lab_pixtral_shim.py`, which on import:
- aliases `PixtralRotaryEmbedding = PixtralVisionRotaryEmbedding`, and
- re-adds `position_ids_in_meshgrid`, sourced verbatim from
  `transformers==4.57.6`'s `modeling_pixtral.py` (downloaded from PyPI and
  diffed by hand for this fix, not reconstructed from memory) — the last
  release still carrying it.

This is now a **standing modification to the production venv**, not a
benchmark-time-only flag — it stays in effect for every future `vllm serve`
invocation until vLLM ships a release whose `pixtral.py` matches current
transformers names, at which point both shim files should be deleted. Verified
`import vllm.model_executor.models.pixtral` succeeds cleanly with the shim in
place, and the model loads, serves, and answers real requests at both 32K and
128K (see results below).

### A second finding, discovered while benchmarking Ministral: the "FP8" checkpoint isn't a separate quantization

`unsloth/Ministral-3-8B-Instruct-2512-FP8` and `mistralai/Ministral-3-8B-Instruct-2512`
(catalogued in round 2 as the FP8 and BF16 variants respectively) turned out
to be **the same underlying weights**. Checked directly by reading the raw
safetensors headers on both checkpoints: every shard is byte-identical in
size (4,983,094,790 / 4,992,892,530 / 444,667,504 bytes, exactly, across all
3 shards), and the same tensor key
(`language_model.model.layers.0.mlp.down_proj.weight`) has the **identical
dtype (`F8_E4M3`) and identical `data_offsets`** in both files. Mistral's own
"base" release already ships natively mixed-precision — 91 tensors
pre-quantized to FP8 (attention/MLP linear weights), 212 left at BF16
(norms, embeddings, vision tower, `lm_head`) — confirmed via each tensor's
declared dtype in the safetensors header. `unsloth`'s "-FP8" repo is a mirror
of the same files, not a distinct further-quantized variant.

This fully explains why the "BF16 vs FP8" baseline numbers below are nearly
identical (154.3 vs 154.7 decode tok/s at `short_short`, 10.25 vs 10.28 GiB
consumed weight memory): **both benchmark runs loaded the same weights.**
There is no actual BF16 baseline for this model in this repo — only this one
native mixed-precision checkpoint, tested under two names. Worth remembering
before citing "Ministral-3-8B BF16" as a quality/speed baseline anywhere.

### Nemotron-H-8B: root-caused, not fixed. A vLLM quantization-auto-detection gap.

The official `nvidia/Nemotron-H-8B-Reasoning-128K-FP8` checkpoint's
`hf_quant_config.json` / `config.json` declare real, static-activation FP8
quantization for every linear layer except each Mamba layer's `conv1d`
(explicitly listed in `ignore`/`exclude_modules` — `in_proj` is **not**
excluded, so it's supposed to be quantized). Instrumented vLLM's weight
loader with a temporary diagnostic print (`linear.py`, reverted immediately
after, confirmed byte-identical to the original via `diff`) and found the
crash happens on the checkpoint's `in_proj.input_scale` tensor: vLLM built
that Mamba merged-linear layer with **no `input_scale` parameter at all**, so
the generic `getattr(submodule, attr, self)` fallback in `linear.py`
(`load_weights`) silently returns the *layer module itself* when the
attribute doesn't exist, and calling `weight_loader` on that gives the
already-documented `'MergedColumnParallelLinear' object has no attribute
'data'`.

Chased one level further: forcing `--quantization modelopt` explicitly (the
dedicated vLLM quant method for this checkpoint's NVIDIA ModelOpt-produced
`hf_quant_config.json` format) fails immediately with a clear `pydantic`
validation error —

```
Quantization method specified in the model config (None) does not match
the quantization method specified in the `quantization` argument (modelopt).
```

— meaning **vLLM 0.29.0's auto-detection finds no quantization config at all
for this checkpoint on the `NemotronHForCausalLM` architecture**, and vLLM's
own config validation refuses to let an explicit `--quantization` override a
`None` auto-detection result. With no quant_config applied anywhere, every
layer (including `in_proj`) is built as plain unquantized — which is exactly
consistent with the missing `input_scale` parameter above. This is a real
gap in vLLM's quant-config auto-detection for this specific
architecture/checkpoint-format combination, not a flag or environment
variable away from working — there's no supported way to force it from the
outside. Deliberately did **not** patch around this by fabricating or
dropping the mismatched scale tensor: doing so would silently change the
model's numerics on a model whose quality was never evaluated in the first
place, exactly the kind of shortcut this repo's own methodology section
warns about. Left the installed vLLM untouched; the BF16 checkpoint (already
confirmed working, see Results below) remains the only usable path for this
model.

### A third finding: `--kv-cache-dtype fp8` is blocked on this GPU for both architectures, by a different bug than the weight-FP8 one

Tried applying the proven `--kv-cache-dtype fp8` lever (real win for
Qwen3.5-4B-FP8, see below) to Nemotron-H-8B (BF16) and Ministral-3-8B, since
neither has a working weight-FP8 path. Both fail identically, at request
time (Ministral) or KV-cache-profiling time (Nemotron-H after a FlashInfer
version bump — see below):

```
File ".../vllm/v1/attention/backends/flashinfer.py", line 2403, in forward
    flashinfer_xqa_batch_decode_with_kv_cache(...)
File ".../vllm/utils/flashinfer.py", line 140, in _missing
RuntimeError: FlashInfer backend is not available. Please install the
package to enable FlashInfer kernels: ...
```

`flashinfer-python` **is** installed (0.6.18) and imports fine — the
"backend not available" message is a generic placeholder vLLM raises when a
*specific* FlashInfer kernel entry point resolves to a stub, not when the
package itself is missing. Here, it's the fused `xqa` batch-decode kernel
vLLM selects specifically for FP8 KV-cache on this GPU
(`arch=sm120` — RTX 5090 / consumer Blackwell, logged explicitly by vLLM at
startup) that isn't available in the installed FlashInfer build. Ruled out
three mitigations directly, each a clean retry changing exactly one thing:

1. **Patch-version bump** (`flashinfer-python` 0.6.18 → 0.6.18.post1, the
   only newer version on PyPI) — same failure, one step further into
   startup (fails during KV-cache profiling instead of the first request),
   suggesting the newer version changed something but didn't add the
   missing sm120 kernel. Reverted back to 0.6.18 (the vLLM-pinned version)
   immediately after ruling this out, to avoid leaving the shared
   production venv off-spec.
2. **`VLLM_ATTENTION_BACKEND=FLASH_ATTN`** — vLLM logged
   `Using FLASHINFER attention backend out of potential backends:
   ['FLASHINFER', 'TRITON_ATTN']` — FLASH_ATTN isn't even offered as an
   option for this architecture/GPU, so the override was silently ignored.
3. **`VLLM_ATTENTION_BACKEND=TRITON_ATTN`** — same backend list logged,
   same FlashInfer selection, identical crash. The FP8-KV-cache code path
   appears hard-routed to FlashInfer's `xqa` kernel regardless of the
   general attention-backend setting for this GPU/architecture combination.

**Conclusion: this is a distinct, separate consumer-Blackwell (sm120)
support gap from the already-documented DeepGEMM weight-FP8 bug
(vllm-project/vllm#51884)** — same root category (fast-moving CUDA kernel
libraries lagging on brand-new consumer hardware), different specific
kernel, different workaround needed (none found this session). Whatever
makes `--kv-cache-dtype fp8` work cleanly on Qwen3.5-4B-FP8 is architecture/
attention-config-specific and doesn't generalize to Nemotron-H's hybrid
Mamba2 attention layers or Ministral-3-8B's — worth checking on any future
candidate before assuming this lever is a free VRAM win everywhere.

## Results: what worked

### Nemotron-H-8B-Reasoning-128K (BF16 only — FP8 broken, see above)

| shape | prompt tok | TTFT p50 | decode tok/s p50 |
|---|---|---|---|
| short_short (32K) | 30 | 0.037s | 102.3 |
| vlong_short (32K) | 9,370 | 0.802s | 109.3 |
| xlong_short (128K) | 93,159 | 8.792s | 97.0 |

Solid, unspectacular — beats the original incumbent's BF16 xlong numbers
(17.6s TTFT / 70.7 tok/s) but well behind Qwen3.5-4B-FP8 (6.3s / 161.1 tok/s)
at the same shape. GPU KV cache: 513,117 tokens @ 32K (15.66x), 649,702 @
128K (4.96x) — similar order of magnitude to Qwen3.5-4B, consistent with
Nemotron-H's own hybrid Mamba2 design (only 4 full-attention layers total per
its model card).

### Ministral-3-8B-Instruct-2512 — now loads, thanks to the pixtral shim above

First real numbers for this model, at both context sizes. Reminder: "bf16"
and "fp8" folders in `benchmarks/results/ministral-3-8b/` contain the same
underlying weights (see the finding above) — one results table, not two.

| shape | prompt tok | TTFT p50 | decode tok/s p50 |
|---|---|---|---|
| short_short (32K) | 29 | 0.020s | 154.7 |
| long_short (32K) | 3,468 | 0.179s | 147.5 |
| vlong_short (32K) | 9,371 | 0.530s | 134.8 |
| xlong_short (128K) | 93,159 | 17.747s | 68.6 |

Beats Nemotron-H-8B at every shape except `xlong_short`, where it lands right
next to it (68.6 vs 97.0 tok/s — Nemotron-H wins here) — but still well
behind Qwen3.5-4B-FP8's 6.3s TTFT / 161.1 tok/s at the same shape. GPU KV
cache: 137,408 tokens @ 32K (4.19x), 137,168 tokens @ 128K (1.05x) — a
conventional dense-transformer profile (no Mamba/linear-attention layers to
shrink KV-cache cost the way Nemotron-H's hybrid design does), and the
smallest KV-cache headroom of any round-2 candidate at 128K.

Concurrency (32K): `short_long` scales cleanly to 11,560 tok/s peak median at
concurrency 256, zero failures at any level tested — essentially identical
scaling to Qwen3.5-4B-FP8's own `short_long` sweep. `vlong_short`
(~9.4K-token prompts, closer to real agentic-conversation traffic) plateaus
at concurrency 8, ~115 tok/s aggregate — a real ceiling, but the model never
fails a request outright at any concurrency level tested up to 64. At 128K,
same story as every other model this size class: concurrency 2 barely beats
concurrency 1 (4.2 vs 4.1 tok/s aggregate) — single-request-only territory,
as expected.

### Granite-4.0-H-Micro — impressive at scale, unreliable at exactly the lengths that matter first

| config | shape | TTFT p50 | decode tok/s p50 | KV cache tokens @ this ctx | max concurrency |
|---|---|---|---|---|---|
| BF16 @ 32K | vlong_short | 0.379s | 204.5 | 2,143,269 | 65.41x |
| BF16 @ 128K | xlong_short | 3.917s | 203.4 | 2,600,821 | 19.84x |
| AWQ @ 128K | xlong_short | **4.158s** | **350.7** | 3,038,710 | 23.18x |
| AWQ @ 128K | xlong_long | **4.137s** | **437.0** | — | — |

At 128K context on a real ~91K-token prompt, **Granite-4.0-H-Micro-AWQ hits
350-437 tok/s decode with a 4.1s TTFT** — more than double Qwen3.5-4B-FP8's
161 tok/s / 6.3s TTFT at the identical shape, on a model less than half the
size (3.2B vs 4.66B params). Its raw KV-cache-token capacity (2.1-3.0
*million* tokens) dwarfs everything else tested in this project — one to two
orders of magnitude beyond Qwen3.5-4B's already-large ~600K, consistent with
`GraniteMoeHybridForCausalLM`'s 9:1 Mamba2-to-transformer layer ratio (Mamba
state size doesn't grow with sequence length the way attention KV cache
does).

**The caveat is serious, and it's not about `--max-model-len`** (see
"Landmine 4, corrected" above for the full investigation): both AWQ and BF16
fail unreliably on short-to-medium prompts — AWQ up to ~12,000 tokens, BF16
occasionally even at ~20-30 tokens — **at any `--max-model-len`, including
128K**. A real tool-calling conversation spends most of its life exactly in
that broken range before it ever reaches 91K tokens, so **this is not yet a
usable model for this repo's target use case**, independent of how good the
numbers get once a conversation is already enormous. Flagging the 128K
numbers as a genuinely exciting data point for a *fixed* future version of
this checkpoint, not as something to build on today.

### LFM2.5-2.6B — new project-wide throughput champion at moderate context, hard-blocked at 128K

| shape | precision | decode tok/s p50 | concurrency peak (short_long, level) |
|---|---|---|---|
| short_short | BF16 | 245.2 | 20,997 tok/s @ 256 |
| short_short | FP8 | 333.3 | 18,312 tok/s @ 256 |
| vlong_short | BF16 | 240.4 | 380.7 tok/s @ 64 |
| vlong_short | FP8 | 327.8 | 369.6 tok/s @ 64 |
| xlong_short (128K) | both | **0 — 100% failure** | **0 — 100% failure** |

20,997 tok/s is the highest raw aggregate throughput measured anywhere in
this project (beats Gemma-4-E4B-it's round-1 record of 16,403 tok/s), on the
smallest model tested (2.7B). Purpose-built for agentic/tool-calling use per
its vendor positioning, and the numbers back that up — up to ~9.8K real
prompt tokens. But it's flatly unusable for this repo's actual target
scenario (deep tool-calling history approaching 128K) per Landmine 4 above.

## VRAM optimization: testing `--kv-cache-dtype fp8`

Separate from every quantization tested so far (all of which quantize
*weights*), vLLM has a **KV-cache dtype** flag that's untouched in this
project until now. Tested against the round-1 champion, Qwen3.5-4B-FP8 @
128K, changing nothing else:

| | `--kv-cache-dtype auto` (baseline) | `--kv-cache-dtype fp8` | Delta |
|---|---|---|---|
| GPU KV cache size | 634,799 tokens | **1,205,662 tokens** | **+90%** |
| Max concurrency @ 128K | 4.84x | **9.20x** | **~1.9x** |
| TTFT @ xlong_short | 6.314s | 7.598s | -20% (worse) |
| Decode tok/s @ xlong_short | 161.1 | 185.3 | **+15%** |

**Real, substantial capacity gain — roughly double the KV-cache headroom and
concurrency ceiling — for a model that's already loaded and already
weight-quantized, at essentially zero extra effort.** Decode throughput even
improved slightly (quantized KV cache means less memory bandwidth per
attention read). The cost: TTFT got ~20% worse, likely from the extra
quantize/dequantize step during prefill. A concurrency check (levels 1/2/4 at
`xlong_short`) showed the extra memory headroom doesn't translate into better
scaling at this prompt size — throughput stayed flat (15.5 → 16.5 tok/s from
concurrency 1 to 4) while TTFT queued up badly (7.6s → 19.0s) — confirming
what round 1 already found: at ~91K-token prompts, **compute is the
bottleneck, not memory**, so more KV-cache room alone doesn't unlock real
concurrent throughput at this extreme. It does, however, directly answer the
"can we save VRAM / serve more with the same GPU" question for anything
short of that extreme — 90% more KV cache for one flag, no download, no
quality risk beyond the (untested) precision loss on cached keys/values
themselves.

## Other software-level VRAM levers found (not yet tested)

From `vllm serve --help=CacheConfig` / `--help=OffloadConfig` /
`--help=MultiModalConfig` on the live install:

- **`--language-model-only`** — disables all multimodal inputs (sets every
  modality limit to 0), skipping the vision/audio encoder weights entirely.
  Relevant to any multimodal-tagged model used text-only (Ministral-3-8B has
  an unused vision tower; so did Gemma-4-E4B-it in round 1) — real weight-VRAM
  savings, untested this round since both candidates were blocked by other
  issues first.
- **`--cpu-offload-gb`** — offloads model weights to system RAM (62GB
  available on this box) via UVA zero-copy access, "virtually" extending GPU
  VRAM at a PCIe-transfer latency cost per forward pass. Could revive some of
  the bigger models ruled out for VRAM reasons in the original model-comparison
  research session — untested, real latency impact unknown.
- **`--kv-offloading-size` / `--kv-offloading-backend`** (`native` or
  `lmcache`) — offloads *cold* KV-cache blocks to CPU RAM rather than the
  whole model, extending effective context/concurrency capacity beyond raw
  GPU VRAM. Untested.

## Verdict

Still no new champion, including after the follow-up. **Qwen3.5-4B-FP8
(round 1) remains the recommendation.** Granite-4.0-H-Micro-AWQ's exceptional
91K-token numbers don't translate into a usable model — the follow-up
investigation found the model unreliable across most of the prompt-length
range a real conversation would actually traverse (see "Landmine 4,
corrected"), not just at one context config. Nemotron-H-8B (BF16) and
Ministral-3-8B (now that it loads) are both solidly *usable* — no
reliability landmines like Granite's, real working numbers at both context
sizes — but both trail Qwen3.5-4B-FP8 at every shape tested, most visibly at
128K (68.6-97.0 tok/s decode vs. 161.1). Nemotron-H's official FP8 checkpoint
and `--kv-cache-dtype fp8` on either model would have narrowed or closed that
gap, but both are blocked by real, root-caused vLLM/FlashInfer support gaps
on this specific GPU (see the follow-up section above) — not weaknesses of
the models themselves. This is a real finding worth having, though: it rules
Granite out with much higher confidence than "seemed to work at 128K, didn't
investigate why 32K failed" would have, and gives a concrete signal
(short-prompt reliability) to check first on any future hybrid Mamba/MoE
candidate before trusting its long-context numbers.

`--kv-cache-dtype fp8` is an unambiguous win, though, and cheap to adopt
immediately on the current production model regardless of which model
ultimately wins: +90% KV-cache headroom, +15% decode throughput, at the cost
of ~20% worse TTFT — a real, no-download, no-new-model way to serve more
concurrent long-context requests on the same GPU, directly answering "can we
optimize at the software level before buying more hardware."
