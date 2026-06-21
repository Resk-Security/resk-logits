"""Streaming generation helpers for use with ReskLogits processors."""

from threading import Thread
from typing import Any, Dict, Iterator, List, Optional

from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase, TextIteratorStreamer


def stream_generate(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    logits_processors: Optional[List] = None,
    **generate_kwargs: Any,
) -> Iterator[str]:
    """Generate text with streaming, applying logits processors.

    Automatically wraps the generation in ``processor.stream()`` for every
    stateful processor in the list, ensuring state is reset between runs.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        prompt: Input text.
        logits_processors: Processors to apply (can include stateful ones
            like ``ShadowBanProcessor``, ``ForceLastPhraseLogitsProcessor``,
            ``TriggerPhraseLogitsProcessor``).
        **generate_kwargs: Additional kwargs forwarded to ``model.generate()``
            (e.g. ``max_new_tokens=100``, ``temperature=0.7``).

    Yields:
        Text chunks as they are generated.

    Usage::

        from resklogits import ShadowBanProcessor, stream_generate

        shadow_ban = ShadowBanProcessor(tokenizer, banned_phrases, device="cpu")
        for chunk in stream_generate(
            model, tokenizer, "Tell me about",
            logits_processors=[shadow_ban],
            max_new_tokens=50,
        ):
            print(chunk, end="", flush=True)
    """
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    gen_kwargs: Dict[str, Any] = dict(
        **inputs,
        streamer=streamer,
        logits_processor=logits_processors or [],
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_kwargs.update(generate_kwargs)

    # Auto-wrap stateful processors in stream() context
    _stateful_reset(gen_kwargs.get("logits_processor", []))

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for text in streamer:
        yield text

    thread.join()


def _stateful_reset(processors: List):
    """Call ``stream()`` on any processor that has the method."""
    for proc in processors:
        if hasattr(proc, "stream"):
            proc.reset()
