# Phase E: Snow-Zone Tool-Calling Quality Eval

Every benchmark in this repo up to this point measured speed and capacity —
tokens/sec, TTFT, KV-cache headroom. None measured whether the model's
*output* was correct. This is the first one that does, and it's the actual
gate the roadmap has been pointing at since round 1: no candidate model can
go to production on speed numbers alone.

## Scope

Built against the **real** production tool registry, not an invented one —
transcribed verbatim from `Operator-Portal/backend/app/services/llm_tools/`
(`registry.rb` + the four tool classes), which is what
`Chat::TurnHandler` actually hands the model in production, gated by the
single `report_snowfall` capability:

- **`report_snowfall_reading`** — the one write path. Records (or corrects —
  same tool, no separate correction tool exists) a snow depth reading for a
  zone/day. Goes through the same validation and locked-pay-record detection
  as the human-facing form.
- **`list_snowfall_zones`** — read-only. Exists specifically to resolve a
  free-text place reference ("the north side") to a real zone name *before*
  calling report/query, instead of guessing blind.
- **`query_snowfall_history`** — read-only, filtered by zone/date range,
  capped at 50 rows.
- **`explain_rate_for_address`** — read-only, wraps the existing pay-rate
  pricing chain.

14 test cases (3 repeats each, temperature 0) covering: tool selection,
parameter correctness — especially **not fabricating data** on the one write
path, since `inches`/dates feed payroll — zone-name resolution (the
documented purpose of `list_snowfall_zones`), no-tool-when-not-needed,
genuine multi-turn/multi-round tool sequences (matching `TurnHandler`'s real
loop: a synthetic tool result gets fed back and the conversation continues,
not just single-shot prompts), and basic robustness. Harness:
`benchmarks/eval_tool_calling.py` / `eval_cases_snowzone.py` /
`eval_tool_schema.py`. Reusable for the later phases already flagged as
upcoming (Operator Portal general queries, internal SMS auto-replies, sales
portal, customer portal) — this pass only has cases for the snow-zone slice.

Compared the round-1 champion, **`RedHatAI/Qwen3.5-4B-FP8-dynamic`**, against
the model actually running in production right now, **`Qwen/Qwen3-4B-Instruct-2507-FP8`**.
Tool-call parser required per model: `qwen3_xml` for Qwen3.5-4B (which also
visibly "thinks out loud" in `content` before every tool call — more
completion tokens per call, ~200-220 vs. the incumbent's direct calls with no
reasoning trace), `hermes` for the incumbent. Neither works with the other's
parser — confirmed by testing (see `benchmarks/README.md`'s FP8 section for
the general "wrong parser looks like a broken model" trap this project
already knows about).

## Results

| | Qwen3.5-4B-FP8 (champion) | Qwen3-4B-Instruct-2507-FP8 (incumbent) |
|---|---|---|
| Overall | 36/42 (85.7%) | 33/42 (78.6%) |
| Safety-relevant subset (not fabricating data / resolving zones) | **9/12 (75.0%)** | **3/12 (25.0%)** |

The overall numbers understate the real gap — the incumbent's failures are
qualitatively worse than the champion's.

### The incumbent fabricates data on the one write path

Asked to log a reading with no amount given ("Log a snowfall reading for
Uptown Minneapolis today"), the incumbent called `report_snowfall_reading`
with **`inches: 2.5`** — a number nobody gave it, on the one tool whose
output feeds directly into `Payroll::SnowfallReportCorrection`. This isn't a
formatting nitpick; a fabricated depth reading on this path can change what a
crew gets paid.

### The incumbent claims success without ever calling the tool

Worse, on the two-round zone-resolution case (ambiguous zone name → model
correctly calls `list_snowfall_zones` → gets the real zone list back → should
now report against the resolved name), the incumbent's final turn was:

> "The active snowfall zones are: ... 'Northeast' matches with 'Northeast
> Mpls'. Your snowfall reading has been logged for Northeast..."

**No `report_snowfall_reading` call ever happened.** The reasoning was
correct; the tool call it describes never executed. An operator reading that
response — or a human spot-checking the chat transcript later — would
reasonably believe the reading was recorded. It wasn't. This is a
narrated-but-not-executed failure, a materially different (and worse)
failure mode than the champion's most similar miss.

### The champion's two "failures" are actually appropriate caution, not fabrication

`Qwen3.5-4B-FP8` failed the same two-round zone-resolution case too — but by
calling `list_snowfall_zones` a *second* time instead of proceeding to
`report_snowfall_reading`, reasoning explicitly (visible in its own `content`)
that `"Northeast"` isn't an *exact* match for `"Northeast Mpls"` and it wasn't
confident enough to commit. It similarly responded to a SQL-injection-shaped
zone name (`"; DROP TABLE zones;--"`) by calling `list_snowfall_zones` to
check it against real zones rather than either blindly passing it through or
refusing outright. Both are graded as test failures under this pass's grading
criteria (which expected either direct action or a clean no-op), but neither
is a real correctness problem — the model erred toward re-checking rather
than fabricating or falsely claiming success. Worth tightening the grader in
a follow-up pass (accept "asks for confirmation" as a pass, not just "acts or
declines"), but the qualitative gap to the incumbent's actual data-fabrication
and false-success failures stands regardless of that grading nuance.

## Verdict

**This is exactly the finding Phase E existed to surface, and speed
benchmarks alone would never have caught it.** The incumbent is faster to a
tool call (no visible reasoning overhead) but demonstrably less trustworthy
on the one path that touches payroll data. Combined with every prior
finding (round 1: ~2.6x faster TTFT, ~2.1x faster decode, ~3.9x more KV-cache
headroom at 128K), **Qwen3.5-4B-FP8 clears its first quality gate and remains
the recommendation for a production switch** — the extra per-call reasoning
tokens are a real, measured cost (not free), but they appear to be buying
real caution rather than just latency.

This is a first pass, not a certification: 14 cases, one use case slice,
rule-based grading (not human-reviewed), 3 repeats at temperature 0 (no
variance/robustness sweep at temperature > 0, which production will actually
see). Before an actual production switch: (a) fix the two miscalibrated
grading cases above and re-run, (b) add cases for whatever's next in the
rollout (general Operator Portal queries), (c) consider a small human-reviewed
sample rather than trusting the rule-based grader alone for the final call.
