#!/usr/bin/env python3
"""
Single-request baseline benchmark for an OpenAI-compatible completions endpoint
(built for vLLM). Measures TTFT, end-to-end latency, and decode throughput
across four prompt/output shapes, repeated N times each, and reports
median (p50) and tail (p95).

This is Phase A of the roadmap in docs/AI_Inference_Server_Build_Log_and_Roadmap.md:
"Finish the Baseline" — controlled single-request numbers before concurrency.

Usage:
    python3 bench_baseline.py
    python3 bench_baseline.py --base-url http://192.168.60.157:8000/v1 --repeats 8
    python3 bench_baseline.py --shapes short_short,long_long
"""

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from shapes import SHAPES
from vllm_client import RequestResult, detect_model, percentile, run_one_request

DEFAULT_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://192.168.60.157:8000/v1")


def summarize(results: list[RequestResult]) -> dict:
    ok = [r for r in results if r.ok]
    failed = len(results) - len(ok)
    if not ok:
        return {"n": len(results), "failed": failed}

    def stats(field: str) -> dict:
        vals = [getattr(r, field) for r in ok]
        return {
            "median": round(statistics.median(vals), 3),
            "p95": round(percentile(vals, 95), 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
        }

    return {
        "n": len(results),
        "failed": failed,
        "prompt_tokens_typical": ok[0].prompt_tokens,
        "completion_tokens_typical": ok[0].completion_tokens,
        "ttft_s": stats("ttft_s"),
        "total_latency_s": stats("total_latency_s"),
        "output_tokens_per_s": stats("output_tokens_per_s"),
        "decode_tokens_per_s": stats("decode_tokens_per_s"),
        "prompt_tokens_per_s": stats("prompt_tokens_per_s"),
    }


def print_summary_table(summaries: dict[str, dict]) -> None:
    header = (
        f"{'shape':<12} {'n':>3} {'fail':>4} {'prompt_tok':>10} {'out_tok':>7} "
        f"{'TTFT p50':>9} {'TTFT p95':>9} {'lat p50':>8} {'lat p95':>8} "
        f"{'decode p50':>11} {'decode p95':>11}"
    )
    print(header)
    print("-" * len(header))
    for shape, s in summaries.items():
        if "ttft_s" not in s:
            print(f"{shape:<12} {s['n']:>3} {s['failed']:>4}  (no successful requests)")
            continue
        print(
            f"{shape:<12} {s['n']:>3} {s['failed']:>4} "
            f"{s['prompt_tokens_typical']:>10} {s['completion_tokens_typical']:>7} "
            f"{s['ttft_s']['median']:>9.3f} {s['ttft_s']['p95']:>9.3f} "
            f"{s['total_latency_s']['median']:>8.3f} {s['total_latency_s']['p95']:>8.3f} "
            f"{s['decode_tokens_per_s']['median']:>11.1f} {s['decode_tokens_per_s']['p95']:>11.1f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                         help=f"OpenAI-compatible base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--model", default=None,
                         help="Model id; auto-detected from /v1/models if omitted")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"),
                         help="Bearer token, if the server requires one")
    parser.add_argument("--repeats", type=int, default=5,
                         help="Requests per shape (default: 5)")
    parser.add_argument("--shapes", default=",".join(SHAPES.keys()),
                         help=f"Comma-separated subset of: {','.join(SHAPES.keys())}")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results"),
                         help="Where to write the JSON result file")
    args = parser.parse_args()

    shapes_to_run = [s.strip() for s in args.shapes.split(",") if s.strip()]
    for s in shapes_to_run:
        if s not in SHAPES:
            print(f"Unknown shape '{s}'. Choices: {', '.join(SHAPES.keys())}", file=sys.stderr)
            return 1

    try:
        model = args.model or detect_model(args.base_url, args.api_key)
    except Exception as exc:
        print(f"Could not reach {args.base_url} to detect model: {exc}", file=sys.stderr)
        return 1

    print(f"Target:  {args.base_url}")
    print(f"Model:   {model}")
    print(f"Repeats: {args.repeats} per shape")
    print(f"Shapes:  {', '.join(shapes_to_run)}")
    print()

    all_results: dict[str, list[RequestResult]] = {}
    for shape in shapes_to_run:
        cfg = SHAPES[shape]
        print(f"Running {shape} ({args.repeats}x, max_tokens={cfg['max_tokens']})...", end=" ", flush=True)
        results = []
        for _ in range(args.repeats):
            r = run_one_request(args.base_url, model, cfg["prompt"], cfg["max_tokens"], shape, args.api_key)
            results.append(r)
        all_results[shape] = results
        ok = sum(1 for r in results if r.ok)
        print(f"{ok}/{len(results)} ok")

    summaries = {shape: summarize(results) for shape, results in all_results.items()}

    print()
    print_summary_table(summaries)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = out_dir / f"baseline_{timestamp}.json"

    payload = {
        "timestamp_utc": timestamp,
        "base_url": args.base_url,
        "model": model,
        "repeats": args.repeats,
        "sampling": {"temperature": 0.0},
        "summaries": summaries,
        "raw_results": {
            shape: [asdict(r) for r in results] for shape, results in all_results.items()
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
