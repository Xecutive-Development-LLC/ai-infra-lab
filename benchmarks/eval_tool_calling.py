#!/usr/bin/env python3
"""
Phase E: task-specific quality eval for snow-zone tool-calling.

Not a speed/capacity benchmark like bench_baseline.py/bench_concurrency.py --
this measures whether the model actually calls the right tool, with the right
parameters, against the real production tool registry (see
eval_tool_schema.py's header). Every prior benchmark in this repo has
measured tokens/sec; none has measured whether the output is *correct*. This
is the first one that does.

Supports multi-step cases: if a case defines `simulate_tool_result`, the
harness feeds back a synthetic tool-role result (matching the real shape
Chat::TurnHandler persists) and sends the conversation for a second round,
mirroring the real tool-calling loop (capped at 2 rounds here, not the real
MAX_TOOL_ROUNDTRIPS=5 -- these cases only need one round-trip to exercise the
behavior of interest).

Usage:
    python3 eval_tool_calling.py --model RedHatAI/Qwen3.5-4B-FP8-dynamic --repeats 3
    python3 eval_tool_calling.py --model Qwen/Qwen3-4B-Instruct-2507-FP8 --repeats 3
"""

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from eval_cases_snowzone import CASES
from eval_tool_schema import SYSTEM_PROMPT, TOOLS

DEFAULT_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://192.168.60.157:8000/v1")
SAFETY_RELEVANT_CATEGORIES = ("param_correctness", "zone_resolution")


def _post(base_url, model, messages, tools, api_key, max_tokens, temperature):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "messages": messages, "tools": tools, "max_tokens": max_tokens, "temperature": temperature}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_one(base_url, model, case, api_key=None, max_tokens=400, temperature=0.0):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(case["messages"])
    t0 = time.monotonic()
    total_completion_tokens = 0
    try:
        data = _post(base_url, model, messages, TOOLS, api_key, max_tokens, temperature)
    except Exception as exc:
        return {"case": case["id"], "category": case["category"], "ok": False, "passed": False,
                "note": f"request failed: {exc}", "latency_s": time.monotonic() - t0}

    msg = data["choices"][0]["message"]
    tool_calls = msg.get("tool_calls") or []
    content = msg.get("content")
    total_completion_tokens += data.get("usage", {}).get("completion_tokens") or 0
    finish_reason = data["choices"][0].get("finish_reason")

    simulate = case.get("simulate_tool_result")
    if simulate and tool_calls:
        # Persist round 1's assistant turn, then one synthetic tool result per
        # call, then any scripted follow-up user message, then ask again --
        # matches TurnHandler's history-append-and-resend loop for one hop.
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        for call in tool_calls:
            result = simulate(call)
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result),
            })
        messages.extend(case.get("followup_messages", []))
        try:
            data = _post(base_url, model, messages, TOOLS, api_key, max_tokens, temperature)
        except Exception as exc:
            return {"case": case["id"], "category": case["category"], "ok": False, "passed": False,
                    "note": f"round-2 request failed: {exc}", "latency_s": time.monotonic() - t0}
        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content")
        total_completion_tokens += data.get("usage", {}).get("completion_tokens") or 0
        finish_reason = data["choices"][0].get("finish_reason")

    latency = time.monotonic() - t0
    passed, note = case["check"](tool_calls, content)
    return {
        "case": case["id"],
        "category": case["category"],
        "ok": True,
        "passed": passed,
        "note": note,
        "latency_s": round(latency, 3),
        "completion_tokens": total_completion_tokens,
        "tool_calls": tool_calls,
        "content_preview": (content or "")[:200],
        "finish_reason": finish_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", required=True, help="Model id being served (must match /v1/models)")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per case (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results" / "evals"))
    args = parser.parse_args()

    print(f"Target:  {args.base_url}")
    print(f"Model:   {args.model}")
    print(f"Repeats: {args.repeats} per case")
    print(f"Cases:   {len(CASES)}")
    print()

    all_results = []
    per_case_summary = {}
    for case in CASES:
        runs = [run_one(args.base_url, args.model, case, args.api_key, args.max_tokens) for _ in range(args.repeats)]
        all_results.extend(runs)
        n_pass = sum(1 for r in runs if r["passed"])
        lat = [r["latency_s"] for r in runs if r["ok"]]
        per_case_summary[case["id"]] = {
            "category": case["category"],
            "pass_rate": round(n_pass / len(runs), 2),
            "n_pass": n_pass,
            "n_total": len(runs),
            "latency_p50": round(statistics.median(lat), 3) if lat else None,
            "notes": list({r["note"] for r in runs}),
        }
        status = "PASS" if n_pass == len(runs) else ("PARTIAL" if n_pass > 0 else "FAIL")
        print(f"[{status:7s}] {case['id']:35s} {n_pass}/{len(runs)}  ({case['category']})")
        for r in runs:
            if not r["passed"]:
                print(f"          -> {r['note']}")

    total_pass = sum(s["n_pass"] for s in per_case_summary.values())
    total_n = sum(s["n_total"] for s in per_case_summary.values())
    print()
    print(f"Overall: {total_pass}/{total_n} ({100*total_pass/total_n:.1f}%)")

    safety_cases = [cid for cid, s in per_case_summary.items()
                    if any(tag in s["category"] for tag in SAFETY_RELEVANT_CATEGORIES)]
    safety_pass = sum(per_case_summary[c]["n_pass"] for c in safety_cases)
    safety_total = sum(per_case_summary[c]["n_total"] for c in safety_cases)
    if safety_total:
        print(f"Safety-relevant subset (not fabricating data / resolving zones correctly): "
              f"{safety_pass}/{safety_total} ({100*safety_pass/safety_total:.1f}%)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    safe_model = args.model.replace("/", "_")
    out_path = out_dir / f"snowzone_{safe_model}_{timestamp}.json"
    payload = {
        "timestamp_utc": timestamp,
        "base_url": args.base_url,
        "model": args.model,
        "repeats": args.repeats,
        "overall_pass_rate": round(total_pass / total_n, 3),
        "safety_relevant_pass_rate": round(safety_pass / safety_total, 3) if safety_total else None,
        "per_case_summary": per_case_summary,
        "raw_results": all_results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
