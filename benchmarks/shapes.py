"""Prompt/output shapes shared by the benchmark scripts."""

# Rotating filler sentences so a "long" prompt isn't just one sentence repeated
# verbatim (which would make prefix-caching effects, if ever enabled, misleading).
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

SHORT_PROMPT = (
    "Explain in two sentences what a KV cache is and why it matters for LLM inference."
)


def build_long_prompt(target_words: int = 2800) -> str:
    """~1.3 tokens/word for English BPE tokenizers, so this lands roughly in the
    multi-thousand-token range. Actual token count is read back from the API's
    `usage` field rather than trusted here."""
    words = 0
    parts = []
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


SHAPES = {
    "short_short": {"prompt": SHORT_PROMPT, "max_tokens": 64},
    "short_long": {"prompt": SHORT_PROMPT, "max_tokens": 512},
    "long_short": {"prompt": build_long_prompt(), "max_tokens": 64},
    "long_long": {"prompt": build_long_prompt(), "max_tokens": 512},
    # ~10K tokens: a pasted document/log, or a deep multi-turn agentic
    # conversation history — closer to real tool-calling traffic than
    # long_*'s ~3.3K. Paired with a short output (a tool-call decision is a
    # small JSON blob, not an essay) and a longer one (a final summarized
    # response) since real agentic loops produce both.
    "vlong_short": {"prompt": build_long_prompt(target_words=7700), "max_tokens": 128},
    "vlong_long": {"prompt": build_long_prompt(target_words=7700), "max_tokens": 512},
}
