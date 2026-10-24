#!/usr/bin/env python3
"""Reproducible overhead benchmark for the shadow-ban logits processor.

This measures the cost the processor adds to every decoding step, not model
quality. It uses a deterministic in-memory tokenizer so the numbers do not
depend on a model download or on GPU availability.

    python3 benchmarks/benchmark_processor.py --patterns 100,1000,10000
    python3 benchmarks/benchmark_processor.py --json benchmarks/results.json

The report includes the machine and library versions. Publish the command and
the raw JSON with any number you quote.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resklogits import ShadowBanProcessor  # noqa: E402


class SyntheticTokenizer:
    """Minimal deterministic tokenizer for benchmarking the automaton.

    Words map to stable ids. The vocabulary is padded to 50,257 entries so the
    danger mask and the logits tensor have a realistic size.
    """

    eos_token_id = 0

    def __init__(self, words: list[str], vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self._word_to_id = {word: i + 1 for i, word in enumerate(dict.fromkeys(words))}
        self._id_to_word = {i: word for word, i in self._word_to_id.items()}
        self._id_to_word[0] = "<eos>"

    def __call__(self, texts, add_special_tokens=False, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [self._encode(text) for text in texts]}

    def _encode(self, text: str) -> list[int]:
        return [self._word_to_id[w] for w in text.split() if w in self._word_to_id]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self._id_to_word.get(int(i), "") for i in ids)


def build_phrases(count: int) -> list[str]:
    return [f"danger term{i} payload{i}" for i in range(count)]


def time_calls(processor, tokenizer, batch: int, iterations: int, seq_len: int = 32) -> dict:
    vocab = tokenizer.vocab_size
    input_ids = torch.randint(1, 1000, (batch, seq_len), dtype=torch.long)
    scores = torch.randn(batch, vocab)
    for _ in range(10):
        processor(input_ids, scores.clone())
    samples = []
    for _ in range(iterations):
        processor.reset()
        start = time.perf_counter()
        processor(input_ids, scores.clone())
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return {
        "batch": batch,
        "iterations": iterations,
        "mean_ms": round(statistics.mean(samples), 4),
        "p50_ms": round(samples[len(samples) // 2], 4),
        "p95_ms": round(samples[min(len(samples) - 1, int(len(samples) * 0.95))], 4),
        "per_item_us": round(statistics.mean(samples) * 1000 / batch, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", default="100,1000,10000")
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--json", default="benchmarks/results.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    pattern_counts = [int(x) for x in args.patterns.split(",") if x.strip()]
    batches = [int(x) for x in args.batches.split(",") if x.strip()]

    all_words = []
    for count in pattern_counts:
        for phrase in build_phrases(count):
            all_words.extend(phrase.split())
    tokenizer = SyntheticTokenizer(all_words + ["safe", "text", "continuation"])

    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "device": args.device,
        "vocab_size": tokenizer.vocab_size,
        "runs": [],
    }

    # Warm up tokenizer caches and torch kernels before timing builds.
    ShadowBanProcessor(
        tokenizer=tokenizer,
        banned_phrases=build_phrases(10),
        shadow_penalty=-15.0,
        device=args.device,
    )

    for count in pattern_counts:
        phrases = build_phrases(count)
        start = time.perf_counter()
        processor = ShadowBanProcessor(
            tokenizer=tokenizer,
            banned_phrases=phrases,
            shadow_penalty=-15.0,
            device=args.device,
        )
        build_ms = (time.perf_counter() - start) * 1000
        run = {
            "patterns": count,
            "build_ms": round(build_ms, 2),
            "danger_tokens": int(processor.ac.danger_mask.sum().item()),
            "calls": [],
        }
        for batch in batches:
            run["calls"].append(
                time_calls(processor, tokenizer, batch, args.iterations)
            )
        results["runs"].append(run)
        print(
            f"{count:>6} patterns  build {build_ms:8.1f} ms  "
            f"danger_mask {run['danger_tokens']:>5} tokens"
        )
        for call in run["calls"]:
            print(
                f"        batch {call['batch']:>2}  mean {call['mean_ms']:7.3f} ms  "
                f"p95 {call['p95_ms']:7.3f} ms  per item {call['per_item_us']:7.2f} us"
            )

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
