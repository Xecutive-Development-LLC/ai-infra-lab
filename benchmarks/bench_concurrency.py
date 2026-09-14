#!/usr/bin/env python3
"""
Concurrency benchmark for an OpenAI-compatible completions endpoint (built for
vLLM). Fires N requests at the same time for each concurrency level, measures
aggregate throughput for the whole batch plus per-request latency/TTFT, and
repeats each level to report median/p95.

This is Phase B of the roadmap in docs/AI_Inference_Server_Build_Log_and_Roadmap.md:
find where aggregate throughput stops scaling with concurrency, and where
per-request latency starts degrading.

Usage:
    python3 bench_concurrency.py
    python3 bench_concurrency.py --levels 1,2,4,8,16,32 --repeats 3
    python3 bench_concurrency.py --shape long_short
"""

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from shapes import SHAPES
from vllm_client import RequestResult, detect_model, percentile, run_one_request

DEFAULT_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://192.168.60.157:8000/v1")


def run_batch(base_url: str, model: str, shape: str, max_tokens: int, prompt_fn,
              concurrency: int, api_key: str | None) -> tuple[list[RequestResult], float]:
    """Fire `concurrency` requests at once, wait for all of them, return the
    results plus the batch's own wall-clock duration (for aggregate throughput).
    Each request gets its own freshly-generated prompt via prompt_fn() — never
    one shared string — so none of them can prefix-cache off a sibling."""
    import time
    batch_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(run_one_request, base_url, model, prompt_fn(), max_tokens, shape, api_key)
            for _ in range(concurrency)
        ]
        results = [f.result() for f in futures]
    batch_duration = time.perf_counter() - batch_start
    return results, batch_duration


def summarize_level(concurrency: int, batch_throughputs: list[float],
                     pooled_results: list[RequestResult]) -> dict:
    ok = [r for r in pooled_results if r.ok]
    failed = len(pooled_results) - len(ok)

    def stats(vals: list[float]) -> dict:
        if not vals:
            return {"median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
        return {
            "median": round(statistics.median(vals), 3),
            "p95": round(percentile(vals, 95), 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
        }

    return {
        "concurrency": concurrency,
        "n_requests": len(pooled_results),
        "failed": failed,
        "aggregate_output_tokens_per_s": stats(batch_throughputs),
        "ttft_s": stats([r.ttft_s for r in ok]),
        "total_latency_s": stats([r.total_latency_s for r in ok]),
    }


def print_summary_table(summaries: list[dict]) -> None:
    header = (
        f"{'conc':>4} {'n':>4} {'fail':>4} {'agg tok/s p50':>14} {'agg tok/s p95':>14} "
        f"{'TTFT p50':>9} {'TTFT p95':>9} {'lat p50':>8} {'lat p95':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['concurrency']:>4} {s['n_requests']:>4} {s['failed']:>4} "
            f"{s['aggregate_output_tokens_per_s']['median']:>14.1f} "
            f"{s['aggregate_output_tokens_per_s']['p95']:>14.1f} "
            f"{s['ttft_s']['median']:>9.3f} {s['ttft_s']['p95']:>9.3f} "
            f"{s['total_latency_s']['median']:>8.3f} {s['total_latency_s']['p95']:>8.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                         help=f"OpenAI-compatible base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--model", default=None,
                         help="Model id; auto-detected from /v1/models if omitted")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"),
                         help="Bearer token, if the server requires one")
    parser.add_argument("--levels", default="1,2,4,8,16",
                         help="Comma-separated concurrency levels (default: 1,2,4,8,16)")
    parser.add_argument("--repeats", type=int, default=3,
                         help="Batches per concurrency level, for median/p95 (default: 3)")
    parser.add_argument("--shape", default="short_long", choices=list(SHAPES.keys()),
                         help="Fixed prompt/output shape used at every concurrency level (default: short_long)")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results"),
                         help="Where to write the JSON result file")
    args = parser.parse_args()

    try:
        levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    except ValueError:
        print(f"--levels must be comma-separated integers, got '{args.levels}'", file=sys.stderr)
        return 1

    try:
        model = args.model or detect_model(args.base_url, args.api_key)
    except Exception as exc:
        print(f"Could not reach {args.base_url} to detect model: {exc}", file=sys.stderr)
        return 1

    cfg = SHAPES[args.shape]

    print(f"Target:     {args.base_url}")
    print(f"Model:      {model}")
    print(f"Shape:      {args.shape} (max_tokens={cfg['max_tokens']})")
    print(f"Levels:     {levels}")
    print(f"Repeats:    {args.repeats} batches per level")
    print()

    summaries = []
    raw_by_level: dict[int, dict] = {}

    for concurrency in levels:
        print(f"Concurrency {concurrency}: ", end="", flush=True)
        batch_throughputs = []
        pooled_results: list[RequestResult] = []
        for rep in range(args.repeats):
            results, batch_duration = run_batch(
                args.base_url, model, args.shape, cfg["max_tokens"], cfg["prompt_fn"], concurrency, args.api_key
            )
            ok_results = [r for r in results if r.ok]
            total_completion_tokens = sum(r.completion_tokens for r in ok_results)
            agg_throughput = total_completion_tokens / batch_duration if batch_duration > 0 else 0.0
            batch_throughputs.append(agg_throughput)
            pooled_results.extend(results)
            print(".", end="", flush=True)

        summary = summarize_level(concurrency, batch_throughputs, pooled_results)
        summaries.append(summary)
        raw_by_level[concurrency] = {
            "batch_throughputs": batch_throughputs,
            "results": [asdict(r) for r in pooled_results],
        }
        print(
            f" agg {summary['aggregate_output_tokens_per_s']['median']:.1f} tok/s median, "
            f"TTFT p50 {summary['ttft_s']['median']:.3f}s, "
            f"{summary['failed']}/{summary['n_requests']} failed"
        )

    print()
    print_summary_table(summaries)

    # crude scaling signal: aggregate throughput at each level vs. the level below it
    print()
    print("Scaling vs. previous level:")
    for i, s in enumerate(summaries):
        if i == 0:
            print(f"  concurrency {s['concurrency']:>3}: baseline")
            continue
        prev = summaries[i - 1]
        prev_tp = prev["aggregate_output_tokens_per_s"]["median"]
        cur_tp = s["aggregate_output_tokens_per_s"]["median"]
        ratio = cur_tp / prev_tp if prev_tp > 0 else 0.0
        conc_ratio = s["concurrency"] / prev["concurrency"]
        note = "scaling well" if ratio >= conc_ratio * 0.7 else "throughput plateauing"
        print(f"  concurrency {s['concurrency']:>3}: {ratio:.2f}x throughput for {conc_ratio:.0f}x concurrency ({note})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = out_dir / f"concurrency_{timestamp}.json"

    payload = {
        "timestamp_utc": timestamp,
        "base_url": args.base_url,
        "model": model,
        "shape": args.shape,
        "levels": levels,
        "repeats": args.repeats,
        "sampling": {"temperature": 0.0},
        "summaries": summaries,
        "raw_by_level": raw_by_level,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
