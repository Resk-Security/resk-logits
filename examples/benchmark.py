"""
Performance Benchmark for Shadow Ban Logits Processor

Measures:
- Build time for automaton
- Per-token processing overhead
- Memory usage
- Throughput comparison
- Scaling with pattern count
"""

import gc
import json
import time

import torch
from resklogits import ShadowBanProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer


def benchmark_build_time(tokenizer, phrases, device="cuda"):
    """Benchmark automaton construction time."""
    start = time.time()
    processor = ShadowBanProcessor(
        tokenizer=tokenizer, banned_phrases=phrases, shadow_penalty=-15.0, device=device
    )
    build_time = time.time() - start
    return processor, build_time


def benchmark_processing_overhead(processor, tokenizer, iterations=1000, device="cuda"):
    """Benchmark per-token overhead."""
    vocab_size = tokenizer.vocab_size
    batch_sizes = [1, 4, 8, 16]
    seq_len = 20

    results = {}

    for batch_size in batch_sizes:
        dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        dummy_scores = torch.randn(batch_size, vocab_size, device=device)

        # Warmup
        for _ in range(10):
            _ = processor(dummy_input_ids, dummy_scores)

        # Benchmark
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(iterations):
            _ = processor(dummy_input_ids, dummy_scores)

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - start
        avg_time_ms = (elapsed / iterations) * 1000
        throughput = (batch_size * iterations) / elapsed

        results[batch_size] = {"avg_time_ms": avg_time_ms, "throughput": throughput}

    return results


def benchmark_generation_overhead(
    model, tokenizer, processor, prompts, max_tokens=50, device="cuda"
):
    """Benchmark generation with/without processor."""
    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Without processor
        start = time.time()
        with torch.no_grad():
            outputs_without = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        time_without = time.time() - start

        # With processor
        processor.reset()
        start = time.time()
        with torch.no_grad():
            outputs_with = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                logits_processor=[processor],
                pad_token_id=tokenizer.eos_token_id,
            )
        time_with = time.time() - start

        overhead_ms = (time_with - time_without) * 1000
        overhead_pct = ((time_with - time_without) / time_without * 100) if time_without > 0 else 0

        results.append(
            {
                "prompt": prompt,
                "time_without": time_without,
                "time_with": time_with,
                "overhead_ms": overhead_ms,
                "overhead_pct": overhead_pct,
            }
        )

    return results


def benchmark_scaling(tokenizer, all_phrases, device="cuda"):
    """Benchmark how performance scales with pattern count."""
    pattern_counts = [10, 50, 100, 250, 500, 1000]
    results = {}

    for count in pattern_counts:
        if count > len(all_phrases):
            break

        phrases = all_phrases[:count]
        processor, build_time = benchmark_build_time(tokenizer, phrases, device)

        vocab_size = tokenizer.vocab_size
        dummy_input_ids = torch.randint(0, vocab_size, (1, 10), device=device)
        dummy_scores = torch.randn(1, vocab_size, device=device)

        # Warmup
        for _ in range(5):
            _ = processor(dummy_input_ids, dummy_scores)

        # Benchmark
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(100):
            _ = processor(dummy_input_ids, dummy_scores)

        if device == "cuda":
            torch.cuda.synchronize()

        process_time = (time.time() - start) / 100 * 1000

        results[count] = {
            "build_time": build_time,
            "process_time_ms": process_time,
            "danger_tokens": processor.ac.danger_mask.sum().item(),
            "states": len(processor.ac.trie),
        }

        del processor
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


def benchmark_memory(processor, device="cuda"):
    """Estimate memory usage."""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Trigger some operations
        vocab_size = processor.tokenizer.vocab_size
        dummy_input_ids = torch.randint(0, vocab_size, (1, 10), device=device)
        dummy_scores = torch.randn(1, vocab_size, device=device)
        _ = processor(dummy_input_ids, dummy_scores)

        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        return memory_mb
    else:
        return None


def main():
    print("=" * 80)
    print("SHADOW BAN LOGITS PROCESSOR - PERFORMANCE BENCHMARK")
    print("=" * 80)
    print()

    # Setup
    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("⚠️  WARNING: Running on CPU (results will be slower)")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Model: {model_name}")
    print()

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()
    print("✓ Model loaded")
    print()

    # Load banned phrases
    print("Loading banned phrases...")
    with open("../src/resklogits/data/banned_phrases.json", "r") as f:
        data = json.load(f)
    all_phrases = [phrase for phrases in data.values() for phrase in phrases]
    print(f"✓ Loaded {len(all_phrases)} phrases")
    print()

    # Benchmark 1: Build Time
    print("=" * 80)
    print("BENCHMARK 1: Automaton Build Time")
    print("=" * 80)
    processor, build_time = benchmark_build_time(tokenizer, all_phrases, device)
    print(f"Patterns: {len(all_phrases)}")
    print(f"Build time: {build_time:.3f}s")
    print(f"States created: {len(processor.ac.trie)}")
    print(f"Danger tokens: {processor.ac.danger_mask.sum().item()} / {tokenizer.vocab_size}")
    print()

    # Benchmark 2: Processing Overhead
    if device == "cuda":
        print("=" * 80)
        print("BENCHMARK 2: Per-Token Processing Overhead")
        print("=" * 80)
        overhead_results = benchmark_processing_overhead(
            processor, tokenizer, iterations=1000, device=device
        )

        print(f"{'Batch Size':<12} {'Avg Time (ms)':<15} {'Throughput (tok/s)':<20}")
        print("-" * 80)
        for batch_size, result in overhead_results.items():
            print(f"{batch_size:<12} {result['avg_time_ms']:<15.4f} {result['throughput']:<20.0f}")
        print()

    # Benchmark 3: Generation Overhead
    print("=" * 80)
    print("BENCHMARK 3: End-to-End Generation Overhead")
    print("=" * 80)
    test_prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In a galaxy far",
        "To be or not to",
        "It was the best",
    ]

    gen_results = benchmark_generation_overhead(
        model, tokenizer, processor, test_prompts, max_tokens=30, device=device
    )

    print(f"{'Prompt':<25} {'No Filter (s)':<15} {'With Filter (s)':<17} {'Overhead':<15}")
    print("-" * 80)
    total_overhead = 0
    for result in gen_results:
        prompt_short = (
            result["prompt"][:22] + "..." if len(result["prompt"]) > 22 else result["prompt"]
        )
        print(
            f"{prompt_short:<25} {result['time_without']:<15.3f} {result['time_with']:<17.3f} "
            f"{result['overhead_pct']:<14.1f}%"
        )
        total_overhead += result["overhead_pct"]

    avg_overhead = total_overhead / len(gen_results)
    print("-" * 80)
    print(f"{'Average':<25} {'':<15} {'':<17} {avg_overhead:<14.1f}%")
    print()

    # Benchmark 4: Scaling
    print("=" * 80)
    print("BENCHMARK 4: Scaling with Pattern Count")
    print("=" * 80)
    scaling_results = benchmark_scaling(tokenizer, all_phrases, device)

    print(
        f"{'Patterns':<12} {'Build (s)':<12} {'Process (ms)':<15} {'States':<10} {'Danger Tokens':<15}"
    )
    print("-" * 80)
    for count, result in scaling_results.items():
        print(
            f"{count:<12} {result['build_time']:<12.3f} {result['process_time_ms']:<15.4f} "
            f"{result['states']:<10} {result['danger_tokens']:<15}"
        )
    print()

    # Benchmark 5: Memory
    if device == "cuda":
        print("=" * 80)
        print("BENCHMARK 5: Memory Usage")
        print("=" * 80)
        memory_mb = benchmark_memory(processor, device)
        print(f"Peak GPU memory: {memory_mb:.2f} MB")
        print(
            f"Danger mask size: {processor.ac.danger_mask.numel() * processor.ac.danger_mask.element_size() / 1024 / 1024:.2f} MB"
        )
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Patterns: {len(all_phrases)}")
    print(f"✓ Build time: {build_time:.3f}s")
    if device == "cuda":
        print(f"✓ Processing overhead: ~{overhead_results[1]['avg_time_ms']:.4f}ms/token (batch=1)")
        print(
            f"✓ Max throughput: ~{max(r['throughput'] for r in overhead_results.values()):.0f} tokens/s"
        )
    print(f"✓ Generation overhead: ~{avg_overhead:.1f}%")
    print(f"✓ Danger tokens: {processor.ac.danger_mask.sum().item()} / {tokenizer.vocab_size}")
    if device == "cuda":
        print(f"✓ Memory: ~{memory_mb:.1f} MB")
    print()
    print("🚀 Ultra-fast GPU-accelerated filtering ready for production!")


if __name__ == "__main__":
    main()
