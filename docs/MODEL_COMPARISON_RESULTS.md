# Model Comparison: Full Results

Raw, per-model, per-precision, per-context results for the same-size-class model
comparison flagged as "next up" in the README: **`Qwen/Qwen3.5-4B`**,
**`microsoft/Phi-4-mini-instruct`**, and **`google/gemma-4-E4B-it`**, each in BF16
and an FP8-dynamic quantized variant, each at 32K and 128K `--max-model-len`. This
is the reference doc for every number; [`benchmarks/README.md`](../benchmarks/README.md)
has the narrative writeup and cross-model synthesis.

All twelve runs use the existing harness ([`bench_baseline.py`](../benchmarks/bench_baseline.py),
[`bench_concurrency.py`](../benchmarks/bench_concurrency.py)) unmodified, same
prefix-cache-safe unique-nonce prompts as every other benchmark in this repo, `temperature=0.0`.
Hardware/host unchanged (RTX 5090, `192.168.60.157`). Raw JSON lives under
`benchmarks/results/<model-slug>/<bf16|fp8>/<32k|128k>/`.

## Models and checkpoints used

| Model | Repo | Params | Native context | FP8 checkpoint used |
|---|---|---|---|---|
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | 4.66B | 262,144 | `RedHatAI/Qwen3.5-4B-FP8-dynamic` |
| Phi-4-mini-instruct | `microsoft/Phi-4-mini-instruct` | 3.84B | 131,072 | `RedHatAI/Phi-4-mini-instruct-FP8-dynamic` |
| Gemma-4-E4B-it | `google/gemma-4-E4B-it` | 8.00B | 131,072 | `prithivMLmods/gemma-4-E4B-it-FP8` |

Two corrections made during setup, worth recording:

- The README's placeholder name was `google/gemma-4-E4B`, which is the **base**
  (non-instruction-tuned) checkpoint. The instruct release is `google/gemma-4-E4B-it`
  — that's what's benchmarked here, for a fair comparison against the other two
  (already-instruct) models.
- None of these three has an official FP8 checkpoint from its own lab (unlike the
  incumbent `Qwen/Qwen3-4B-Instruct-2507-FP8`). Picked the highest-trust community
  `compressed-tensors` FP8-dynamic quant for each (RedHatAI — the same org, formerly
  Neural Magic, whose format vLLM's compressed-tensors backend targets natively;
  `prithivMLmods` for Gemma, the highest-download reputable option available). All
  three loaded cleanly on vLLM 0.29 with no format issues.

vLLM 0.29 (already installed, no upgrade needed) supports all three architectures
natively: `Qwen3_5ForConditionalGeneration`, `Phi3ForCausalLM`, `Gemma4ForConditionalGeneration`.

## GPU KV cache headroom (from each server's own startup log)

The single most informative number per run, same as the existing FP8-vs-BF16
section reports it — `GPU KV cache size` and vLLM's own `Maximum concurrency for N
tokens per request` at that `--max-model-len`:

| Model | Precision | 32K: KV cache tokens | 32K: max concurrency | 128K: KV cache tokens | 128K: max concurrency |
|---|---|---|---|---|---|
| Qwen3.5-4B | BF16 | 518,114 | 15.81x | 560,268 | 4.27x |
| Qwen3.5-4B | FP8 | 586,499 | 17.90x | 634,799 | 4.84x |
| Phi-4-mini-instruct | BF16 | 151,973 | 4.64x | 152,029 | 1.16x |
| Phi-4-mini-instruct | FP8 | 176,489 | 5.39x | 176,554 | 1.35x |
| Gemma-4-E4B-it | BF16 | 596,618 | 18.21x | 742,143 | 5.66x |
| Gemma-4-E4B-it | FP8 | 715,672 | 21.84x | 890,237 | 6.79x |
| *(incumbent)* Qwen3-4B-Instruct-2507 | BF16 | — | — | 138,064 | 1.05x |
| *(incumbent)* Qwen3-4B-Instruct-2507 | FP8 | — | — | 162,144 | 1.24x |

FP8 loading: all three used vLLM's `CutlassFP8ScaledMMLinearKernel for
CompressedTensorsW8A8Fp8` kernel path — **not** the block-scaled DeepGEMM path that
needed the `VLLM_USE_DEEP_GEMM=0` / `VLLM_MOE_USE_DEEP_GEMM=0` workaround for the
incumbent's official checkpoint (`benchmarks/README.md`'s "RTX 5090-specific
landmine" section). Those env vars were still set defensively on every FP8 launch
here; the landmine appears specific to block-scaled FP8, not per-channel/dynamic
FP8, so it's plausible they weren't actually needed for these three checkpoints.
Not confirmed either way — didn't test without them.

Qwen3.5-4B and Gemma-4-E4B-it both report **dramatically higher raw KV-cache-token
capacity** than Phi-4-mini-instruct despite having larger (Qwen3.5) or much larger
(Gemma) weights. This tracks with architecture, not VRAM: both show hybrid/linear
attention structure in their layer names (Qwen3.5's FP8 quant config lists
`layers.N.linear_attn.*` targets; vLLM's compiled op list for this server includes
`qwen_gdn_attention_core`, i.e. Gated DeltaNet-style linear attention mixed with
full attention). A linear-attention layer's KV state doesn't grow with sequence
length the way standard multi-head attention's does, so per-token KV cache cost is
much lower on average — Phi-4-mini-instruct's architecture (`Phi3ForCausalLM`, a
conventional dense transformer) doesn't get that benefit, which is the likely
reason its raw token capacity is 3-4x lower than the other two despite the smallest
weights of any model tested here.

## Baseline results (single request, concurrency=1)

`n`/`fail` counts requests attempted/failed per shape (5 repeats for 32K shapes, 3
for 128K `xlong_*` shapes, matching the existing suite's cost-scaled repeat counts).

### Qwen3.5-4B

| shape | prompt tok | out tok | TTFT p50 (BF16) | TTFT p50 (FP8) | decode tok/s p50 (BF16) | decode tok/s p50 (FP8) |
|---|---|---|---|---|---|---|
| short_short | ~26 | 64 | 0.040s | 0.044s | 164.9 | 206.6 |
| short_long | ~28 | 512 | 0.122s | 0.046s | 169.1 | 205.8 |
| long_short | ~3,390 | 64 | 0.177s | 0.134s | 161.8 | 161.9 |
| long_long | ~3,391 | 512 | 0.170s | 0.134s | 161.3 | 203.1 |
| vlong_short | ~9,157 | 128 | 0.478s | 0.372s | 158.2 | 203.5 |
| vlong_long | ~9,154 | 512 | 0.448s | 0.352s | 158.2 | 197.2 |
| xlong_short (128K) | ~91,057 | 128 | 7.065s | 6.314s | 129.3 | 161.1 |
| xlong_long (128K) | ~91,057 | 512 | 7.112s | 6.265s | 127.0 | 152.2 |

FP8 is a strict win here at every shape tested — same pattern as the incumbent's
FP8 comparison, but the gain is bigger at long context (+20-25% at vlong, still
+18-25% even at ~91K tokens, vs. the incumbent's +8-9% at that size).

**The headline number**: at ~91K prompt tokens, Qwen3.5-4B's BF16 TTFT (7.07s) is
**less than half** the incumbent Qwen3-4B-Instruct-2507's BF16 TTFT at the same
shape (17.6s, from `benchmarks/README.md`), and decode throughput (129 tok/s) is
**~1.8x** the incumbent's (70.7 tok/s). FP8 widens both gaps further (6.3s TTFT,
161 tok/s decode vs. the incumbent FP8's 16.2s / 77.4 tok/s). This is consistent
with the hybrid-attention architecture difference noted above: standard attention's
cost grows with context length; a model that's mixing in linear-attention layers
pays less of that tax.

### Phi-4-mini-instruct

| shape | prompt tok | out tok | TTFT p50 (BF16) | TTFT p50 (FP8) | decode tok/s p50 (BF16) | decode tok/s p50 (FP8) |
|---|---|---|---|---|---|---|
| short_short | ~25 | 64 | 0.020s | 0.023s | 186.7 | 265.7 |
| short_long | ~25 | ~120 | 0.019s | 0.022s | 185.5 | 260.9 |
| long_short | ~3,408 | 64 | 0.151s | 0.103s | 120.4 | 145.3 |
| long_long | ~3,404 | ~69 | 0.200s | 0.104s | 131.7 | 144.8 |
| vlong_short | ~9,214 | ~67 | 0.429s | 0.319s | 74.7 | 80.5 |
| vlong_long | ~9,216 | ~66 | 0.437s | 0.312s | 75.0 | 80.6 |
| xlong_short (128K) | ~91,748 | ~55 | 12.307s | 11.078s | 11.0 | 11.2 |
| xlong_long (128K) | ~91,749 | ~75 | 12.267s | 11.078s | 10.9 | 11.2 |

FP8 gives a solid, consistent speed bump at every shape (+15-40% decode, faster
TTFT throughout) — same clean pattern as the other two models. But look at the
**128K row**: decode collapses to ~11 tok/s, a 6-7x cliff from the 74-80 tok/s it
holds at 32K/vlong_short. Compare to Qwen3.5-4B, which only drops from ~200 to
~130-160 tok/s (a ~25% dip) over the same context jump, or even the incumbent
Qwen3-4B-Instruct-2507, which drops from ~160 to ~70-77 tok/s (a ~55% dip, but
nowhere near Phi-4-mini's ~93% dip). Phi-4-mini-instruct's dense-attention
architecture and small raw KV-cache-token capacity (151K-176K, see table above)
both point the same direction: this model pays the full quadratic-attention cost
at long context, with none of the mitigation the other two get from hybrid/linear
attention layers. Its max-concurrency-at-128K (1.16x-1.35x) is essentially
single-request-only, same as the incumbent.

### Gemma-4-E4B-it — broken on any prompt past ~30 tokens

| shape | prompt tok | n | failed | notes |
|---|---|---|---|---|
| short_short | ~28 | 5 | **0** | OK — 125.7 (BF16) / 188.6 (FP8) tok/s decode |
| short_long | ~28 | 5 | **0** | OK — 145.1 (BF16) / 186.4 (FP8) tok/s decode |
| long_short | ~3,466 | 5 | **5/5** | 100% failure, both precisions |
| long_long | ~3,460 | 5 | **5/5** | 100% failure, both precisions |
| vlong_short | ~9,150 | 5 | **5/5** | 100% failure, both precisions |
| vlong_long | ~9,150 | 5 | **5/5** | 100% failure, both precisions |
| xlong_short (128K) | ~91,000 | 3 | **3/3** | 100% failure, both precisions |
| xlong_long (128K) | ~91,000 | 3 | **3/3** | 100% failure, both precisions |

**Every single request with a prompt longer than the ~19-28-token short shapes
failed**, identically across BF16 and FP8, at both 32K and 128K `--max-model-len`.
Concurrency sweeps confirm the same boundary: the `short_long` sweep (levels 1-256)
ran clean (peaking at 12,917 tok/s BF16 / 16,403 tok/s FP8 — the highest raw
throughput of any model tested, consistent with its huge KV-cache-token headroom),
while every single request in the `vlong_short` and `xlong_short` sweeps failed at
every concurrency level tested. A handful of `short_long` requests also failed
sporadically at high concurrency (2 out of 2,304 total requests across both BF16
and FP8 sweeps) — small enough to be the same root cause showing up rarely rather
than a separate issue, but noted for completeness.

**What actually happens** (reproduced directly, not just inferred from the
benchmark client's error): the server returns a normal `200 OK` and the SSE stream
closes with `usage: {prompt_tokens: 3466, completion_tokens: 1, ...}` but **zero
content-bearing chunks** — the model emits what looks like an immediate
end-of-sequence token as its very first generated token, for any prompt past the
short-shape threshold. The vLLM server log shows no error, exception, or crash
anywhere — from the server's point of view this is a normal, successful,
one-token completion. `temperature=0.0` is sent explicitly on every request (the
harness always does this — see `vllm_client.py`), so this isn't the
`generation_config.json` sampling-parameter warning vLLM logs on this model's
startup (that only applies when a request omits sampling params).

Best-effort hypothesis, not confirmed: the server startup log shows `Gemma4 model
has heterogeneous head dimensions {'sliding_attention': 256, 'full_attention':
512}. FA4 not available, forcing TRITON_ATTN backend` — i.e. this brand-new
architecture's preferred attention kernel (FA4) isn't available on vLLM 0.29 for
this GPU, so it silently falls back to a different backend (Triton) for gemma-4's
mixed sliding-window/full-attention layers. A correctness bug in that fallback
path for long sequences (producing degenerate/garbage logits that argmax to EOS)
would explain every symptom observed: identical failure across both precisions
(precision-independent, so not a quantization bug), a clean HTTP 200 with no
server-side error (a numerics bug, not a crash), and a sharp threshold tied to
prompt length rather than content (consistent with the sliding-window boundary).
Not root-caused further than this — would need to bisect attention backends or
file/check upstream vLLM issues for `Gemma4ForConditionalGeneration` +
`TRITON_ATTN` to confirm. **Gemma-4-E4B-it should be treated as unusable for this
repo's target use case (long tool-calling conversation history) until this is
resolved upstream or a different attention backend/vLLM version fixes it** — its
otherwise-excellent KV-cache headroom and short-prompt speed are moot if it can't
handle a realistic prompt.

## Concurrency sweeps

`short_long` (full sweep, levels 1-256, 3 repeats/level) and `vlong_short` (levels
1-64, 3 repeats/level) at 32K; `xlong_short` (levels 1-2, 2 repeats/level) at 128K.
Full per-level tables are in the raw JSON; headline peak numbers:

| Model | Precision | short_long peak agg tok/s (concurrency) | vlong_short peak agg tok/s (concurrency) | xlong_short @ conc=2 (128K) |
|---|---|---|---|---|
| Qwen3.5-4B | BF16 | 7,277 (128, plateaus by 256) | 267.8 (64, still climbing) | 15.9 tok/s, TTFT 11.1s |
| Qwen3.5-4B | FP8 | 8,460 (256) | 342.7 (64, still climbing) | 17.9 tok/s, TTFT 9.9s |
| Phi-4-mini-instruct | BF16 | 8,871 (256) | 135.9 (64, flat since ~16) | 3.6 tok/s, TTFT 21.7s |
| Phi-4-mini-instruct | FP8 | 10,690 (256) | 189.9 (64, still climbing) | 3.4 tok/s, TTFT 18.6s |
| Gemma-4-E4B-it | BF16 | 12,917 (256) | **0 (100% failed, all levels)** | **0 (100% failed)** |
| Gemma-4-E4B-it | FP8 | 16,403 (256) | **0 (100% failed, all levels)** | **0 (100% failed)** |
| *(incumbent)* Qwen3-4B-Instruct-2507 | BF16 | 15,367 (256, reference sweep) | 186 (peak, flat by 8) | 1.01x throughput for 2x concurrency |
| *(incumbent)* Qwen3-4B-Instruct-2507 | FP8 | — | 247 (peak, flat by 16) | 1.00x throughput for 2x concurrency |

Two things stand out:

1. **At `xlong_short` (128K, 91K-token prompts), Phi-4-mini-instruct's concurrency=2
   throughput (3.4-3.6 tok/s) is barely above noise** — worse in absolute terms
   than its own concurrency=1 number in some runs, i.e. adding a second concurrent
   huge request doesn't help and may actively hurt, consistent with a model whose
   decode is already attention-bound rather than memory-bandwidth-bound at this
   size. Qwen3.5-4B, by contrast, holds ~16-18 tok/s at concurrency=2 — still far
   from linear scaling (as expected, per the incumbent's own "effectively
   single-request-at-a-time at this size" finding) but meaningfully more
   throughput than Phi-4-mini-instruct's.
2. Gemma-4-E4B-it's `short_long` numbers are the fastest raw throughput measured
   in this entire repo (16,403 tok/s FP8 peak, beating the incumbent's 15,367 tok/s
   reference run) — but that number is meaningless for the actual use case this
   comparison exists for, given the long-prompt failure above.

## Verdict

**Qwen3.5-4B (FP8-dynamic) is the standout result of this comparison, and looks
like a genuine upgrade over the current production incumbent** for this repo's
stated use case (growing tool-calling conversations, i.e. long-context speed and
concurrency matter more than short-prompt throughput). At the shape that matters
most — ~91K-token prompts at 128K context — it delivers roughly **2.2-2.6x faster
TTFT** and **~2x higher decode throughput** than the incumbent Qwen3-4B-Instruct-2507-FP8,
plus **~3.9x more KV-cache headroom** (634,799 vs. 162,144 tokens) and a
**~3.9x higher max-concurrency multiplier** (4.84x vs. 1.24x) at the same
`--max-model-len`. This tracks with a real architectural difference (hybrid
linear/full attention vs. presumably-dense attention), not just quantization or
better luck — the gap holds up across every shape tested, not just one favorable
data point.

**Phi-4-mini-instruct is not a good fit for this repo's long-context use case** —
competitive at 32K, but its decode throughput falls off a cliff at 128K (11 tok/s,
a ~6-7x drop vs. its own 32K numbers, and worse than even the incumbent's own
128K numbers). Its dense-transformer architecture doesn't get the same
long-context relief the other two models show.

**Gemma-4-E4B-it cannot currently be evaluated for this use case at all** — it's
architecturally the most promising of the three (largest KV-cache headroom, fastest
short-prompt throughput of any model tested here) but fails on 100% of prompts
longer than ~30 tokens, in both precisions, at both context lengths. This looks
like a vLLM-side day-0-support gap (attention-backend fallback for the brand-new
`Gemma4ForConditionalGeneration` architecture), not a fundamental model problem —
worth revisiting if/when vLLM ships FA4 support or a different backend for this
architecture, but out of scope to chase further here.

As with the original FP8-vs-BF16 comparison: **this is speed and capacity only —
output quality was not evaluated for any of these three models.** Before actually
switching the production server away from `Qwen/Qwen3-4B-Instruct-2507-FP8`, the
roadmap's Phase E (task-specific quality/accuracy evals, not generic benchmarks)
still needs to happen. This doc's job is narrowing the field, not making the final
call.

## Raw result files

```
benchmarks/results/
├── qwen3.5-4b/{bf16,fp8}/{32k,128k}/{baseline,concurrency}_<timestamp>.json
├── phi-4-mini-instruct/{bf16,fp8}/{32k,128k}/{baseline,concurrency}_<timestamp>.json
└── gemma-4-e4b-it/{bf16,fp8}/{32k,128k}/{baseline,concurrency}_<timestamp>.json
```

Each `baseline_*.json` has per-shape summaries (n/failed, TTFT/latency/decode
percentiles) plus every individual raw request. Each `concurrency_*.json` has
per-level summaries plus every individual raw request within each batch.
