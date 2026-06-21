# ReskLogits

[![PyPI version](https://img.shields.io/pypi/v/resklogits.svg)](https://pypi.org/project/resklogits/)
[![Python Versions](https://img.shields.io/pypi/pyversions/resklogits.svg)](https://pypi.org/project/resklogits/)
[![License](https://img.shields.io/pypi/l/resklogits.svg)](https://github.com/Resk-Security/resk-logits/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/resklogits)](https://pepy.tech/project/resklogits)
[![GitHub issues](https://img.shields.io/github/issues/Resk-Security/resk-logits.svg)](https://github.com/Resk-Security/resk-logits/issues)
[![GitHub stars](https://img.shields.io/github/stars/Resk-Security/resk-logits.svg)](https://github.com/Resk-Security/resk-logits/stargazers)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![GitHub last commit](https://img.shields.io/github/last-commit/Resk-Security/resk-logits)](https://github.com/Resk-Security/resk-logits/commits/main)
[![PyPI - Implementation](https://img.shields.io/pypi/implementation/resklogits)](https://pypi.org/project/resklogits/)
[![LLM Security](https://img.shields.io/badge/LLM-Security-red)](https://github.com/Resk-Security/resk-logits)

## 🎯 What is ReskLogits?

**ReskLogits** is a logits processor that implements a "shadow ban" system to filter dangerous content during text generation by language models (LLMs).

### Key Concept: Shadow Ban vs Hard Block

Unlike traditional methods that completely block certain tokens (hard block), ReskLogits applies an **invisible penalty** to dangerous tokens, making them extremely unlikely without explicitly blocking generation. This creates a more natural user experience while maintaining high security.

### How It Works

The library uses a vectorized **Aho-Corasick** algorithm on GPU or (CPU) to detect dangerous patterns in generated text. It pre-computes a binary mask that identifies all dangerous tokens, then applies a penalty to corresponding logits in real-time.

**Example: with GPT2**

Prompt: "Tell me how to make a bomb"

**WITHOUT Shadow Ban:**
```


With that, the man turned and went to where the bomb lay.

"I am going to get a bomb," he said.

Well, he had the bomb ready.

"I am going to get a
```

**WITH Shadow Ban:**
```
"There will be four," said I, "one with two guns, one with one man." The two men looked like they were about eighty years old, but, "There will be two." I took out my pistol, opened it,
```

```mermaid
graph LR
    A[User Prompt] --> B[LLM Model]
    B --> C["Raw Logits<br/>1×vocab_size"]
    C --> D["VectorizedAhoCorasick<br/>State + GPU Mask"]
    D --> E["Danger Mask<br/>1×vocab_size"]
    E --> F["Apply Penalty<br/>logits mask += -15.0"]
    F --> G[Penalized Logits]
    G --> H[Token Generation]
    H --> I{Dangerous Token?}
    I -->|Yes| J["Probability ~0.00003%"]
    I -->|No| K[Normal Generation]
    J --> L[Safe Text Generated]
    K --> L
    
    style D fill:#e1f5ff
    style E fill:#fff4e1
    style F fill:#ffe1e1
    style J fill:#ffcccc
```

### Concrete Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from resklogits import ShadowBanProcessor
import torch

# 1. Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. Define banned phrases
banned_phrases = [
    "how to make a bomb",
    "kill yourself",
    "hack into system",
    "create explosives"
]

# 3. Create shadow ban processor
shadow_ban = ShadowBanProcessor(
    tokenizer=tokenizer,
    banned_phrases=banned_phrases,
    shadow_penalty=-15.0,  # Strong penalty (probability ~0.00003%)
    device="cuda"  # Use GPU
)

# 4. Generate text with protection
prompt = "Tell me how to"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Reset state for new generation
shadow_ban.reset()

# Generate with shadow ban
outputs = model.generate(
    **inputs,
    logits_processor=[shadow_ban], 
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7
)

# Result: Model naturally avoids dangerous tokens
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")

```

### Key Advantages

- 🎭 **Invisible**: User doesn't notice the filtering
- 🛡️ **Jailbreak-resistant**: Stateful detection captures partial generations
- 📈 **Scalable**: Handles 1000+ banned phrases
- 🔧 **Easy to integrate**: Compatible with HuggingFace Transformers, vLLM, TGI
- ⚡ **Streaming-ready**: Built-in ``stream()`` context manager and ``stream_generate()`` helper

---

## Streaming

All stateful processors (``ShadowBanProcessor``, ``ForceLastPhraseLogitsProcessor``,
``TriggerPhraseLogitsProcessor``) provide a ``stream()`` context manager that
auto-resets internal state on enter and exit.

### With ``stream_generate()`` (recommended)

```python
from resklogits import ShadowBanProcessor, stream_generate

shadow_ban = ShadowBanProcessor(tokenizer, banned_phrases, device="cpu")

for chunk in stream_generate(
    model, tokenizer, "Tell me about",
    logits_processors=[shadow_ban],
    max_new_tokens=50,
    temperature=0.7,
):
    print(chunk, end="", flush=True)
```

```python
# Works with any combination of processors
from resklogits import (
    GenLengthLogitsProcessor,
    BanTokenProcessor,
    ForceLastPhraseLogitsProcessor,
    stream_generate,
)

procs = [
    GenLengthLogitsProcessor(tokenizer, min_length=20, max_length=100),
    BanTokenProcessor(tokenizer, banned_tokens=["rm"]),
    ForceLastPhraseLogitsProcessor(tokenizer, phrase="\n\nFIN_ACTION", trigger_length=98),
]

for chunk in stream_generate(model, tokenizer, "Explain RAG", logits_processors=procs):
    print(chunk, end="", flush=True)
```

### With ``TextIteratorStreamer`` (manual)

```python
from threading import Thread
from transformers import TextIteratorStreamer
from resklogits import ShadowBanProcessor

shadow_ban = ShadowBanProcessor(tokenizer, banned_phrases, device="cpu")
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

inputs = tokenizer("Tell me about", return_tensors="pt")

with shadow_ban.stream():  # auto reset on enter + exit
    thread = Thread(target=model.generate, kwargs={
        **inputs,
        max_new_tokens=50,
        logits_processor=[shadow_ban],
        streamer=streamer,
    })
    thread.start()
    for text in streamer:
        print(text, end="", flush=True)
```

### With vLLM

```python
from resklogits import to_vllm, BanTokenProcessor
from vllm import LLM, SamplingParams

ban = to_vllm(BanTokenProcessor(tokenizer, banned_tokens=["rm"]))
llm = LLM(model="gpt2")
params = SamplingParams(temperature=0.7, logits_processors=[ban])

for output in llm.generate(["Tell me about"], params):
    for token in output.outputs[0].token_ids:
        print(tokenizer.decode(token), end="", flush=True)
```

---

## Architecture

## Architecture

```
[GPU] → Logits (1×vocab_size) → [Vectorized Aho-Corasick] → Mask (1×vocab_size) → Penalized Logits
```

## Installation

### Using uv (recommended)

```bash
uv pip install resklogits
```

### Using pip

```bash
pip install resklogits
```

### From source

```bash
git clone https://github.com/resk-team/resklogits.git
cd resklogits
uv pip install -e .
```


## Shadow Ban vs Hard Block

| Method | Approach | Probability | User Experience |
|--------|----------|-------------|-----------------|
| **Hard Block** | `logits[token] = -inf` | 0% | Unnatural, obvious filtering |
| **Shadow Ban** | `logits[token] += -15.0` | ~0.00003% | Natural, invisible filtering |

## Penalty Levels

| Penalty | Probability | Use Case |
|---------|-------------|----------|
| `-5.0` | ~1% | Light filtering |
| `-10.0` | ~0.005% | Medium filtering |
| `-15.0` | ~0.00003% | Strong filtering (default) |
| `-20.0` | ~impossible | Maximum filtering |

## Multi-Level Filtering

For tiered safety filtering by severity:

```python
from resklogits import MultiLevelShadowBanProcessor

phrases_by_level = {
    'high': ['bomb', 'kill', 'murder'],      # -20.0 penalty
    'medium': ['hack', 'exploit', 'crack'],  # -10.0 penalty
    'low': ['jailbreak', 'bypass']           # -5.0 penalty
}

multi_level = MultiLevelShadowBanProcessor(
    tokenizer=tokenizer,
    banned_phrases_by_level=phrases_by_level,
    penalties={'high': -20.0, 'medium': -10.0, 'low': -5.0}
)
```

## Utility Logits Processors

ReskLogits provides a collection of ready-to-use logits processors for
common generation control tasks.

### GenLengthLogitsProcessor

Adjusts the EOS token logit to control sequence length:

```python
from resklogits import GenLengthLogitsProcessor

length_ctrl = GenLengthLogitsProcessor(
    tokenizer,
    min_length=30,      # Penalise EOS before 30 tokens
    max_length=200,     # Boost EOS after 200 tokens
    eos_penalty=-10.0,  # Penalty when below min
    eos_boost=5.0,      # Boost when above max
)
```

### CiteFromPromptLogitsProcessor

Boosts tokens that appear in the prompt — ideal for RAG / attentive reading:

```python
from resklogits import CiteFromPromptLogitsProcessor

prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
cite = CiteFromPromptLogitsProcessor(
    tokenizer,
    prompt_ids=prompt_ids,
    boost_factor=2.0,  # Boost prompt tokens by +2.0
)
```

### ForceLastPhraseLogitsProcessor

Forces a specific phrase at the end of the generated sequence.
Useful for structured output (signatures, footers, validation tokens):

```python
from resklogits import ForceLastPhraseLogitsProcessor

# Force "FIN_ACTION" at the end (trigger_length = max_new_tokens - phrase_len)
force = ForceLastPhraseLogitsProcessor(
    tokenizer,
    phrase="\n\nFIN_ACTION",
    trigger_length=195,  # Start forcing when seq_len reaches 195
)

# Or trigger manually at any point:
force.force_now()
```

### MultipleChoiceLogitsProcessor

Restricts generation to a predefined set of choices (MCQ, True/False, etc.):

```python
from resklogits import MultipleChoiceLogitsProcessor

mcq = MultipleChoiceLogitsProcessor(
    tokenizer,
    choices=["0", "1", "2", "3"],
)

# All logits except those for "0", "1", "2", "3" are set to -inf
```

### BanTokenProcessor

Hard-blocks specific tokens (strings or IDs) — complementary to
ShadowBan's phrase-level penalty approach:

```python
from resklogits import BanTokenProcessor

ban = BanTokenProcessor(
    tokenizer,
    banned_tokens=["rm", "DROP", "SELECT"],
    # or banned_token_ids=[0, 1, 2]
)

# Add / remove tokens dynamically:
ban.add_banned_tokens([100, 200])
ban.remove_banned_tokens([0])
ban.add_banned_strings(tokenizer, ["FORMAT", "TRUNCATE"])
```

### TriggerPhraseLogitsProcessor

Watches for a trigger phrase token-by-token. When detected, it forces
the model to generate a predefined response. Uses a state machine for
exact trigger matching:

```python
from resklogits import TriggerPhraseLogitsProcessor

# Auto-complete code after ```python marker
trigger = TriggerPhraseLogitsProcessor(
    tokenizer,
    trigger="\n```python",
    response="import torch\nimport numpy as np\n\ndef main():",
)
```

### YAML Configuration

All utility processors can be configured via YAML file and loaded in
one call:

```yaml
# processors.yaml
processors:
  gen_length:
    min_length: 10
    max_length: 200

  force_last_phrase:
    phrase: "\n\nFIN_ACTION"
    trigger_length: 197

  ban_token:
    banned_tokens: ["rm -rf", "DROP TABLE"]

  trigger_phrases:
    - trigger: "\n```python"
      response: "import torch\n"
```

```python
from resklogits.config_parser import load_processors_from_yaml

procs = load_processors_from_yaml("processors.yaml", tokenizer)

# Pass directly to generate
outputs = model.generate(..., logits_processor=procs)
```

### vLLM Compatibility

The `VLLMWrapper` adapts any Transformers `LogitsProcessor` for use
with vLLM's `SamplingParams`:

```python
from resklogits import to_vllm, to_vllm_list
from resklogits import ShadowBanProcessor, BanTokenProcessor

# Single processor
shadow_ban = ShadowBanProcessor(tokenizer, banned_phrases, device="cuda")
vllm_proc = to_vllm(shadow_ban)

# List of processors
procs = to_vllm_list([
    ShadowBanProcessor(tokenizer, banned_phrases, device="cuda"),
    BanTokenProcessor(tokenizer, banned_tokens=["rm"]),
])

# Pass to vLLM
from vllm import SamplingParams
params = SamplingParams(logits_processors=vllm_proc)
```

## Symbolic Rule Generator

Generate patterns from YAML rules instead of manually listing them:

### Create YAML Rules

```yaml
rules:
  violence:
    severity: high
    penalty: -20.0
    templates:
      - pattern: "{instruction} {action} {weapon}"
        instruction: ["how to", "guide to"]
        action: ["make", "build", "create"]
        weapon: ["a bomb", "an explosive"]
    exact:
      - "kill yourself"
```

### Generate Patterns

```bash
# CLI
resklogits generate rules.yaml -o patterns.json

# Python
from resklogits import load_rules_from_yaml
patterns = load_rules_from_yaml("rules.yaml")
```

### Features

- **Templates**: Variable substitution and combinatorial expansion
- **Logic rules**: AND, OR, NOT operators
- **Synonyms**: Automatic synonym expansion
- **Caching**: Hash-based caching avoids regeneration
- **CLI**: Full command-line interface

See [RULE_BUILDER.md](RULE_BUILDER.md) for complete guide.

## How It Works

### 1. Aho-Corasick Automaton

Classical multi-pattern matching with:
- Trie structure for pattern storage
- Failure links for efficient transitions
- Output functions for match detection

### 2. GPU Vectorization

Pre-computes a binary mask `[vocab_size]` where:
- `mask[i] = True` if token `i` is dangerous
- Applied via vectorized operation: `scores[:, mask] += penalty`

### 3. State Tracking

Maintains automaton state across generation:
- Tracks partial matches in progress
- Detects complete pattern matches
- Forces EOS on successful matches

## Banned Phrases Dataset

The library includes a comprehensive dataset of 400+ dangerous phrases across 20 categories in `src/resklogits/data/banned_phrases.json`:
- Violence and weapons
- Hate speech and slurs
- Exploitation and trafficking
- Hacking and exploits
- Fraud and scams
- Drug synthesis
- Self-harm content
- Jailbreak attempts

You can use your own patterns or extend the provided dataset.

## Examples

The `examples/` directory contains:

### Demo Script
```bash
cd examples
python demo.py
```

Tests:
- Loading and building automaton
- Generation with/without shadow ban
- Multi-level filtering

### Benchmark Script
```bash
cd examples
python benchmark.py
```

Comprehensive benchmarks:
- Automaton build time
- Scaling with pattern count
- Memory usage

### Simple Usage
```bash
cd examples
python example_usage.py
```

Minimal example showing basic setup.

### Rule Generator Demo
```bash
cd examples
python rule_generator_demo.py
```

Demonstrates symbolic rule generation with templates and caching.

### Cache Management Demo
```bash
cd examples
python cache_demo.py
```

Shows cache functionality and management.

## API Reference

### VectorizedAhoCorasick

```python
from resklogits import VectorizedAhoCorasick

class VectorizedAhoCorasick:
    def __init__(self, tokenizer, banned_phrases, device="cuda")
    def step(self, state: int, token: int) -> int
    def has_match(self, state: int) -> bool
    def get_matched_patterns(self, state: int) -> List[int]
```

### ShadowBanProcessor

```python
from resklogits import ShadowBanProcessor

class ShadowBanProcessor(LogitsProcessor):
    def __init__(self, tokenizer, banned_phrases, shadow_penalty=-15.0, device="cuda")
    def __call__(self, input_ids, scores) -> torch.FloatTensor
    def reset(self)
    def get_current_matches(self, batch_idx=0) -> List[str]
```

### MultiLevelShadowBanProcessor

```python
from resklogits import MultiLevelShadowBanProcessor

class MultiLevelShadowBanProcessor(ShadowBanProcessor):
    def __init__(self, tokenizer, banned_phrases_by_level, penalties=None, device="cuda")
```

### ConfigParser (Rule Generator)

```python
from resklogits import ConfigParser, load_rules_from_yaml

# Parse YAML rules
parser = ConfigParser()
results = parser.generate_all_patterns("rules.yaml")

# Convenience function
patterns = load_rules_from_yaml("rules.yaml", use_cache=True)
```

### RuleCache

```python
from resklogits import RuleCache

cache = RuleCache()
if cache.exists(rule_hash):
    patterns = cache.load(rule_hash)
else:
    patterns = generate()
    cache.save(rule_hash, patterns)
```


## Development

### Setup Development Environment

```bash
git clone https://github.com/resk-team/resklogits.git
cd resklogits
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Run Tests

```bash
# Tests unitaires
pytest tests/ -v

# Avec couverture
pytest tests/ --cov=resklogits --cov-report=html

# Script de test complet
# Linux/Mac:
bash scripts/test_all.sh
# Windows:
scripts\test_all.bat
```

### Build Local

```bash
# Build du package
uv build

# Vérifier le package
twine check dist/*

# Tester l'installation
# Linux/Mac:
bash scripts/build_and_test.sh
# Windows:
scripts\build_and_test.bat
```

### Code Formatting

```bash
# Formater
black src/ tests/ examples/

# Vérifier
black --check src/ tests/ examples/

# Linter
ruff check src/ tests/ examples/

# Type checking
mypy src/
```

See [LOCAL_TESTING.md](LOCAL_TESTING.md) for complete testing guide.

## Project Structure

```
resklogits/
├── src/
│   └── resklogits/
│       ├── __init__.py
│       ├── vectorized_aho_corasick.py
│       ├── shadow_ban_processor.py
│       ├── vllm_adapter.py              # vLLM compatibility wrapper
│       ├── config_parser.py             # YAML parser (rules + processors)
│       ├── processors/
│       │   ├── __init__.py
│       │   ├── gen_length.py
│       │   ├── cite_from_prompt.py
│       │   ├── force_last_phrase.py
│       │   ├── multiple_choice.py
│       │   ├── ban_token.py
│       │   └── trigger_phrase.py
│       └── data/
│           └── banned_phrases.json
├── examples/
│   ├── demo.py
│   ├── example_usage.py
│   └── benchmark.py
├── tests/
│   ├── test_basic.py
│   └── test_processors.py              # 30+ tests for utility processors
├── pyproject.toml
└── README.md
```

## License

APACHE 2

## Citation

If you use this in research, please cite:

```bibtex
@software{resklogits_2025,
  title={ReskLogits: GPU-Accelerated Shadow Ban Logits Processor},
  author={RESK},
  year={2025},
  url={https://github.com/Resk-Security/resk-logits}
}
```
