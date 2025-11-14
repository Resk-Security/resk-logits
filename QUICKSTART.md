# ReskLogits - Quick Start Guide

## Installation

```bash
uv pip install resklogits
```

## Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from resklogits import ShadowBanProcessor

# Load your model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Define banned patterns
banned = [
    "how to make a bomb",
    "kill yourself",
    "hack into"
]

# Create processor
shadow_ban = ShadowBanProcessor(
    tokenizer=tokenizer,
    banned_phrases=banned,
    shadow_penalty=-15.0,
    device="cuda"
)

# Generate
inputs = tokenizer("Tell me how to", return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    logits_processor=[shadow_ban],
    max_new_tokens=50
)

print(tokenizer.decode(outputs[0]))
```

## Using the Built-in Dataset

```python
import json
from pathlib import Path

# Load included banned phrases
data_path = Path(__file__).parent / "resklogits" / "data" / "banned_phrases.json"
with open(data_path) as f:
    data = json.load(f)

# Flatten all categories
banned_phrases = [phrase for phrases in data.values() for phrase in phrases]

# Or use specific categories
violence_phrases = data["violence"]
hate_phrases = data["hate_speech"]
```

## Multi-Level Filtering

```python
from resklogits import MultiLevelShadowBanProcessor

phrases_by_level = {
    'high': data["violence"] + data["hate_speech"],
    'medium': data["exploit_commands"] + data["fraud"],
    'low': data["jailbreak_attempts"]
}

processor = MultiLevelShadowBanProcessor(
    tokenizer=tokenizer,
    banned_phrases_by_level=phrases_by_level,
    penalties={'high': -20.0, 'medium': -10.0, 'low': -5.0}
)
```

## Penalty Guidelines

| Penalty | Effect | Use Case |
|---------|--------|----------|
| `-5.0` | Light (~1% chance) | Discourage but allow |
| `-10.0` | Medium (~0.005%) | Strong discouragement |
| `-15.0` | Strong (~0.00003%) | Near-impossible (default) |
| `-20.0` | Maximum | Virtually impossible |

## Performance Tips

1. **Build once, reuse**: Create the processor once at startup
2. **Reset between generations**: Call `processor.reset()` for new sequences
3. **Use GPU**: 100x faster than CPU
4. **Batch processing**: Supports batched generation natively

## Examples

See the `examples/` directory:
- `demo.py` - Full demonstration
- `benchmark.py` - Performance testing
- `example_usage.py` - Minimal example

## Troubleshooting

### CUDA out of memory
```python
# Use CPU or reduce batch size
shadow_ban = ShadowBanProcessor(..., device="cpu")
```

### Patterns not matching
```python
# Debug: check tokenization
tokens = tokenizer.encode("your phrase")
print(tokenizer.decode(tokens))
```

### Too restrictive
```python
# Reduce penalty
shadow_ban = ShadowBanProcessor(..., shadow_penalty=-5.0)
```

