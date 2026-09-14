"""Prompt/output shapes shared by the benchmark scripts.

Every `prompt_fn()` call returns a freshly-generated, unique string — never a
cached/precomputed one. This matters because vLLM has prefix caching on by
default: two requests sharing a leading substring get a (near-)free prefill
on the second one, silently understating real cold-prefill cost. Confirmed
via `Prefix cache hit rate: 81.3%` in the server log after a run that reused
the same deterministic long-prompt text across shapes/repeats. A random nonce
at the very start of every long/vlong/xlong prompt defeats that by
construction, so every request pays real prefill cost, matching how actual
traffic behaves (different users send different text).
"""

import uuid

# Rotating filler sentences so a "long" prompt isn't just one sentence repeated
# verbatim within itself.
_FILLER_SENTENCES = [
    "The inference server processes each request by tokenizing the prompt, "
    "running it through the model, and streaming generated tokens back to the client.",
    "GPU memory is shared between model weights, activations, and the KV cache, "
    "and the KV cache grows with both context length and batch size.",
    "Continuous batching lets a serving engine interleave prefill and decode work "
    "across multiple in-flight requests instead of processing them strictly one at a time.",
    "Benchmarking a language model server means separating prompt-processing throughput "
    "from decode throughput, since the two phases have very different performance characteristics.",
    "A stable production deployment needs persistence, authentication, structured logging, "
    "and health monitoring in addition to raw inference speed.",
]

SHORT_PROMPT_TEMPLATE = (
    "Request {nonce}. Explain in two sentences what a KV cache is and why it matters for LLM inference."
)


def build_short_prompt() -> str:
    return SHORT_PROMPT_TEMPLATE.format(nonce=uuid.uuid4().hex[:8])


def build_long_prompt(target_words: int = 2800) -> str:
    """~1.3 tokens/word for English BPE tokenizers, so this lands roughly in the
    multi-thousand-token range. Actual token count is read back from the API's
    `usage` field rather than trusted here. Starts with a random nonce so no
    two calls — even with the same target_words — share a cacheable prefix."""
    words = 0
    parts = [f"Request {uuid.uuid4().hex}."]
    i = 0
    while words < target_words:
        sentence = _FILLER_SENTENCES[i % len(_FILLER_SENTENCES)]
        parts.append(sentence)
        words += len(sentence.split())
        i += 1
    parts.append(
        "\n\nGiven all of the above, explain in two sentences what a KV cache is "
        "and why it matters for LLM inference."
    )
    return " ".join(parts)


# Each entry's "prompt_fn" must be called fresh for every single request — never
# reuse one generated string across repeats or concurrent calls in the same batch.
SHAPES = {
    "short_short": {"prompt_fn": build_short_prompt, "max_tokens": 64},
    "short_long": {"prompt_fn": build_short_prompt, "max_tokens": 512},
    "long_short": {"prompt_fn": build_long_prompt, "max_tokens": 64},
    "long_long": {"prompt_fn": build_long_prompt, "max_tokens": 512},
    # ~10K tokens: a pasted document/log, or a deep multi-turn agentic
    # conversation history — closer to real tool-calling traffic than
    # long_*'s ~3.3K. Paired with a short output (a tool-call decision is a
    # small JSON blob, not an essay) and a longer one (a final summarized
    # response) since real agentic loops produce both.
    "vlong_short": {"prompt_fn": lambda: build_long_prompt(target_words=7700), "max_tokens": 128},
    "vlong_long": {"prompt_fn": lambda: build_long_prompt(target_words=7700), "max_tokens": 512},
    # ~90-100K tokens: stress-tests context length itself, not just concurrency.
    # Only usable when the server was started with a --max-model-len large
    # enough to fit prompt + max_tokens (needs >=131072 for xlong_long).
    "xlong_short": {"prompt_fn": lambda: build_long_prompt(target_words=77000), "max_tokens": 128},
    "xlong_long": {"prompt_fn": lambda: build_long_prompt(target_words=77000), "max_tokens": 512},
}
