# ReskLogits Examples

This directory contains example scripts demonstrating ReskLogits usage.

## Prerequisites

Install resklogits:
```bash
cd ..
uv pip install -e .
```

Or if already installed:
```bash
uv pip install resklogits
```

## Examples

### 1. Simple Usage (`example_usage.py`)

Minimal example showing basic setup and usage.

```bash
python example_usage.py
```

**What it does:**
- Loads a model (GPT-2 by default)
- Creates shadow ban processor
- Compares generation with/without filtering

**Good for:** Quick start, understanding basics

---

### 2. Full Demo (`demo.py`)

Comprehensive demonstration with multiple test cases.

```bash
python demo.py
```

**What it does:**
- Loads 400+ banned phrases
- Builds Aho-Corasick automaton
- Runs multiple generation tests
- Shows multi-level filtering
- Performance comparison

**Good for:** Understanding all features, testing with your model

---

### 3. Performance Benchmark (`benchmark.py`)

Detailed performance analysis.

```bash
python benchmark.py
```

**What it does:**
- Measures automaton build time
- Tests per-token processing overhead
- Benchmarks generation overhead
- Tests scaling with pattern count
- Reports memory usage

**Good for:** Performance evaluation, optimization

## Customization

### Use Your Own Model

Edit any example script:
```python
model_name = "gpt2"  # Change to your model
# e.g., "meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1"
```

### Use Custom Patterns

```python
banned_phrases = [
    "your custom pattern 1",
    "your custom pattern 2",
    # ...
]
```

### Adjust Shadow Penalty

```python
shadow_ban = ShadowBanProcessor(
    tokenizer=tokenizer,
    banned_phrases=banned_phrases,
    shadow_penalty=-5.0,  # Lighter: -5, Stronger: -20
    device="cuda"
)
```

## Expected Output

### example_usage.py
```
Loading model: gpt2
Loading banned phrases...
Creating shadow ban processor with 400+ patterns...
✓ Ready! Danger tokens: 2847

Prompt: "Tell me how to"

WITHOUT Shadow Ban:
Tell me how to make a bomb...

WITH Shadow Ban:
Tell me how to improve your skills...

✓ Shadow ban successfully applied!
```

### benchmark.py
```
================================================================================
BENCHMARK 1: Automaton Build Time
================================================================================
Patterns: 400
Build time: 0.324s
States created: 1247
Danger tokens: 2847 / 50257

================================================================================
BENCHMARK 2: Per-Token Processing Overhead
================================================================================
Batch Size   Avg Time (ms)   Throughput (tok/s)  
--------------------------------------------------------------------------------
1            0.0012          833333              
4            0.0015          2666666             
8            0.0018          4444444             
16           0.0023          6956521             
```

## Troubleshooting

### CUDA Out of Memory
```python
# Use CPU instead
device = "cpu"
```

### Model Download Issues
```python
# Set offline mode if model is cached
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```

### Slow Performance
- Make sure you're using GPU (`device="cuda"`)
- Check GPU utilization: `nvidia-smi`
- Reduce pattern count for testing

## Next Steps

After running examples:
1. Review the code to understand implementation
2. Test with your specific model
3. Customize banned phrases for your use case
4. Integrate into your production pipeline

See main README.md for full documentation.

