# Phase E: Fallback Candidate Tool-Calling Eval

Validate-only pass, not a production-switch decision. Champion
(`RedHatAI/Qwen3.5-4B-FP8-dynamic`, 39/42) already cleared its own gate in
`docs/PHASE_E_TOOL_CALLING_EVAL_RESULTS.md`. Speed/capacity for both
candidates below was already benchmarked in round 1
(`docs/MODEL_COMPARISON_RESULTS.md`) — the only thing missing was the
quality gate, so that's the only thing this pass ran. Gemma-4-E4B-it was
dropped from consideration entirely: it fails on any prompt >~30 tokens (a
real vLLM attention-backend bug, not a config issue), not fixable cheaply.

Same harness, same 14 cases, same real production tool registry as the
original eval (`benchmarks/eval_tool_calling.py` / `eval_cases_snowzone.py`
/ `eval_tool_schema.py`) — `--repeats 3`, `--max-model-len 32768` (this pass
checks correctness, not the already-documented 128K speed collapse for
Phi-4-mini). Both served bare-metal in `~/llm-env` during a brief
`docker compose stop vllm` / `up -d vllm` window (~13 minutes total
downtime) — confirmed via the operator this repo works with that nothing
currently sends live traffic to this server (the Operator Portal's AI
features and the agentic-orchestration layer are both still being built),
so this carried no active-outage risk.

## Results

| | Qwen3.5-4B-FP8 (champion, reference) | Qwen3.5-4B-BF16 | Phi-4-mini-instruct-FP8 |
|---|---|---|---|
| Overall | **39/42 (92.9%)** | 36/42 (85.7%) | 24/42 (57.1%) |
| Safety-relevant subset (not fabricating data / resolving zones) | **9/12 (75.0%)** | 6/12 (50.0%) | 9/12 (75.0%) |

Raw output: `benchmarks/results/evals/snowzone_Qwen_Qwen3.5-4B_2026-09-17T17-53-14Z.json`,
`benchmarks/results/evals/snowzone_RedHatAI_Phi-4-mini-instruct-FP8-dynamic_2026-09-17T17-54-54Z.json`.

## Qwen3.5-4B-BF16 — a plausible same-family fallback

Same parser (`qwen3_xml`) as the champion, no new unknowns. Overall score
drops from the champion's 39/42 to 36/42, and the safety-relevant subset
drops further, from 9/12 to 6/12 — both failures are genuine zone-resolution
misses (`report_ambiguous_zone`, `zone_resolution_multistep`): the model
either called `report_snowfall_reading` directly with a non-exact zone name
instead of resolving it first via `list_snowfall_zones`, or got stuck
reasoning about a zone mismatch without ever calling a tool or asking the
user to clarify. Notably, this is worse than the FP8 champion at the exact
same task the FP8 quantization doesn't touch (zone-name judgment isn't a
quantization-precision question) — the two checkpoints are not behaviorally
identical past their shared parser/architecture. Usable as a same-family
fallback if the champion becomes unavailable, but not a drop-in behavioral
match — the zone-resolution regression is a real, reproducible gap.

## Phi-4-mini-instruct-FP8 — real parser/output-format landmine, not a clean fallback

Confirmed the correct `--tool-call-parser` for vLLM 0.29.0
(`phi4_mini_json`, from source) and vendored the matching chat template
(`deploy/chat_templates/tool_chat_template_phi4_mini.jinja`, from vLLM's own
`examples/tool_chat_template_phi4_mini.jinja` at the `v0.29.0` tag) since
Phi-4-mini's own tokenizer template has no branch for
`tool_calls`/`role: tool` messages. Both loaded and served without error.

The low score isn't primarily a tool-*selection* problem — the
safety-relevant subset (9/12, tying the champion) shows the model reasons
about zone ambiguity and missing parameters about as well as the champion
does. It's a **parser/output-format mismatch**: inspecting the raw failures
(e.g. `clear_report`, a fully-specified single-call case that failed 0/3)
shows the model consistently *intends* to call the tool but emits it as a
markdown-fenced generic JSON array —
```json
[{"name": "report_snowfall_reading", "arguments": {...}}]
```
— instead of the `phi4_mini_json` parser's expected bare
`functools[{"name": ..., "arguments": ...}]` marker (no code fence, no
`[` `]` wrapper ambiguity the regex `functools\[(.*?)\]` needs). The parser
never finds its marker, so vLLM reports zero tool calls even though the
model's intent and arguments were correct. This happened on 6 of 14 cases,
always as "no tool call made," and was reproducible across all 3 repeats
each time — not a sampling fluke.

**Not viable as a fallback in its current configuration.** Whether this is
fixable with prompt/template tuning, a different parser
(`--tool-call-parser` has other Phi-4-family-adjacent options in vLLM 0.29
not tried here), or is specific to this community FP8 quant
(`RedHatAI/Phi-4-mini-instruct-FP8-dynamic`) vs. the official BF16 checkpoint
is unresolved — logged as a landmine, not chased further this pass, per the
"asap, don't burn a lot of time" scope for this round. If Phi-4-mini is
ever needed for real (e.g. its 32K-context speed profile suits a
task-specific use case the champion doesn't), this parser/template mismatch
is the first thing to revisit.

## Bottom line

Qwen3.5-4B-BF16 is a workable same-family fallback (documented recipe
below), with a known, real zone-resolution regression versus the champion.
Phi-4-mini-instruct-FP8 is not currently usable for tool-calling in this
stack — the model's own output format doesn't reliably match what vLLM's
available parser expects — and would need real follow-up work, not a
quick swap, to become one.

## Serving recipe (for future reference — cold-start on demand, not a standing service)

```bash
# on the VM, after `docker compose stop vllm`
source ~/llm-env/bin/activate

# Qwen3.5-4B-BF16
VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 vllm serve Qwen/Qwen3.5-4B \
  --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml

# Phi-4-mini-instruct-FP8 (tool-calling currently unreliable, see above)
VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 vllm serve RedHatAI/Phi-4-mini-instruct-FP8-dynamic \
  --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser phi4_mini_json \
  --chat-template deploy/chat_templates/tool_chat_template_phi4_mini.jinja

# restore production afterward
docker compose up -d vllm
```
