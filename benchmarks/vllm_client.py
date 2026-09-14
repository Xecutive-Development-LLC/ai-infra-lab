"""
Shared client for talking to an OpenAI-compatible completions endpoint (built
for vLLM) and turning the streamed response into timing/throughput numbers.
Used by bench_baseline.py (Phase A, sequential) and bench_concurrency.py
(Phase B, concurrent batches) so both scripts measure things the same way.

Standard library only, on purpose — no dependency install needed to run a
benchmark against a box that just came up.
"""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class RequestResult:
    shape: str
    ok: bool
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_s: float = 0.0
    total_latency_s: float = 0.0
    output_tokens_per_s: float = 0.0
    decode_tokens_per_s: float = 0.0
    prompt_tokens_per_s: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0


def run_one_request(base_url: str, model: str, prompt: str, max_tokens: int,
                     shape: str, api_key: str | None, temperature: float = 0.0) -> RequestResult:
    url = f"{base_url.rstrip('/')}/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )

    start = time.perf_counter()
    ttft = None
    usage = None
    first_token_seen = False

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)

                choices = chunk.get("choices") or []
                if choices and choices[0].get("text") and not first_token_seen:
                    ttft = time.perf_counter() - start
                    first_token_seen = True

                if chunk.get("usage"):
                    usage = chunk["usage"]
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        return RequestResult(shape=shape, ok=False, error=str(exc), start_time=start,
                              end_time=time.perf_counter())

    end = time.perf_counter()
    total_latency = end - start

    if ttft is None or usage is None:
        return RequestResult(
            shape=shape, ok=False,
            error=f"incomplete stream (ttft={ttft}, usage={usage})",
            start_time=start, end_time=end,
        )

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    decode_window = max(end - (start + ttft), 1e-6)

    return RequestResult(
        shape=shape,
        ok=True,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_s=ttft,
        total_latency_s=total_latency,
        output_tokens_per_s=completion_tokens / total_latency if total_latency > 0 else 0.0,
        decode_tokens_per_s=(completion_tokens - 1) / decode_window if completion_tokens > 1 else 0.0,
        prompt_tokens_per_s=prompt_tokens / ttft if ttft > 0 else 0.0,
        start_time=start,
        end_time=end,
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def detect_model(base_url: str, api_key: str | None) -> str:
    url = f"{base_url.rstrip('/')}/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["data"][0]["id"]
