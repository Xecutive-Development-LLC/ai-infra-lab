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

No new champion this round. **Qwen3.5-4B-FP8 (round 1) remains the
recommendation.** Granite-4.0-H-Micro-AWQ's exceptional 91K-token numbers
don't translate into a usable model — the follow-up investigation found the
model unreliable across most of the prompt-length range a real conversation
would actually traverse (see "Landmine 4, corrected"), not just at one
context config. The most reliable new candidate, Nemotron-H-8B, doesn't beat
round 1's champion either. This is a real finding worth having, though: it
rules Granite out with much higher confidence than "seemed to work at 128K,
didn't investigate why 32K failed" would have, and gives a concrete signal
(short-prompt reliability) to check first on any future hybrid Mamba/MoE
candidate before trusting its long-context numbers.

`--kv-cache-dtype fp8` is an unambiguous win, though, and cheap to adopt
immediately on the current production model regardless of which model
ultimately wins: +90% KV-cache headroom, +15% decode throughput, at the cost
of ~20% worse TTFT — a real, no-download, no-new-model way to serve more
concurrent long-context requests on the same GPU, directly answering "can we
optimize at the software level before buying more hardware."
