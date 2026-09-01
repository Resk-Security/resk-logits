# ReskLogits

> Blocks dangerous LLM output at the token level — invisibly, before it exists.

[![PyPI version](https://img.shields.io/pypi/v/resklogits.svg)](https://pypi.org/project/resklogits/)
[![Python Versions](https://img.shields.io/pypi/pyversions/resklogits.svg)](https://pypi.org/project/resklogits/)
[![License](https://img.shields.io/pypi/l/resklogits.svg)](https://github.com/Resk-Security/resk-logits/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/resklogits)](https://pepy.tech/project/resklogits)
[![GitHub stars](https://img.shields.io/github/stars/Resk-Security/resk-logits.svg)](https://github.com/Resk-Security/resk-logits/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/Resk-Security/resk-logits.svg)](https://github.com/Resk-Security/resk-logits/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/Resk-Security/resk-logits)](https://github.com/Resk-Security/resk-logits/commits/main)

## Installation

```bash
pip install resklogits
```

## Usage rapide (30 seconds)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from resklogits import ShadowBanProcessor

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

shadow_ban = ShadowBanProcessor(
    tokenizer=tokenizer,
    banned_phrases=["how to make a bomb", "kill yourself"],
    shadow_penalty=-15.0,   # probability ~0.00003%
    device="cuda",
)

shadow_ban.reset()
outputs = model.generate(
    **tokenizer("Tell me how to", return_tensors="pt").to("cuda"),
    logits_processor=[shadow_ban],
    max_new_tokens=50,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

The model naturally steers away from dangerous tokens — no exception raised, no truncated output, nothing visible to the user.

## Pourquoi ReskLogits ?

Content filters scan text **after** generation: by the time they catch a banned phrase, the model already produced it and your user already saw it (or a broken "regenerate" loop). Hard blocking (`logits[token] = -inf`) is visible and unnatural. ReskLogits applies an **invisible penalty** inside the generation loop, using a GPU/CPU-vectorized Aho-Corasick automaton that tracks partial matches across tokens — so jailbreak-style multi-token phrasings are still caught.

| | **ReskLogits** | LLM Guard / NeMo (post-hoc filters) | Hard block (`-inf`) |
|---|---|---|---|
| When it acts | During generation, per token | After full output exists | During generation |
| Forbidden content ever emitted | No | Yes — before detection | No |
| User experience | Invisible, natural | Broken outputs, retries | Abrupt truncation |
| Multi-token / partial-match detection | ✅ stateful automaton | ⚠️ regex on final text | Token-exact only |
| Scaling to 1000+ phrases | ✅ vectorized mask, GPU | Runtime cost per request | Mask size explodes |
| Streaming support | ✅ `stream()` / `stream_generate()` | ⚠️ | ✅ |
| vLLM / TGI compatible | ✅ `to_vllm()` adapter | Varies | ⚠️ |

## Documentation

- [QUICKSTART.md](QUICKSTART.md) — get running in minutes
- [QUICKSTART_RULES.md](QUICKSTART_RULES.md) — YAML rule generation
- [RULE_BUILDER.md](RULE_BUILDER.md) — complete rule-builder guide
- [CHANGELOG.md](CHANGELOG.md)

## Highlights

- **Streaming built-in**: `stream()` context manager auto-resets state; `stream_generate()` yields text chunks directly.
- **Penalty levels**: `-5.0` (light) → `-15.0` (default) → `-20.0` (near-impossible).
- **Multi-level filtering**: `MultiLevelShadowBanProcessor` applies different penalties per severity (`high`/`medium`/`low`).
- **Batteries included**: 400+ dangerous phrases across 20 categories in the bundled dataset; symbolic YAML rule generator with templates, logic operators and synonyms; 8 utility logits processors (length control, forced endings, MCQ restriction, token bans, trigger phrases, prompt-grounding boost).
- **vLLM compatible**: `to_vllm(processor)` adapts any processor to `SamplingParams`.

```python
from resklogits import MultiLevelShadowBanProcessor

multi = MultiLevelShadowBanProcessor(
    tokenizer=tokenizer,
    banned_phrases_by_level={
        "high": ["bomb", "kill"],
        "medium": ["hack", "exploit"],
        "low": ["jailbreak"],
    },
    penalties={"high": -20.0, "medium": -10.0, "low": -5.0},
)
```

## Examples

```bash
cd examples
python demo.py            # generation with/without shadow ban
python benchmark.py       # build time, pattern scaling, memory
python rule_generator_demo.py
```

## Development

```bash
git clone https://github.com/Resk-Security/resk-logits.git
cd resk-logits
uv pip install -e ".[dev]"
pytest tests/ -v --cov=resklogits
```

## Ecosystem

- **[Resk-LLM](https://github.com/Resk-Security/Resk-LLM)** — input-time detection; integrates ReskLogits for generation-time defense.
- **[resksecure](https://github.com/Resk-Security/reskSecure)** — per-user bitmask firewall built on this engine.
- **[Resk](https://github.com/Resk-Security/Resk)** — full-stack LLM firewall app.

## License

Apache 2.0. If you use this in research, please cite:

```bibtex
@software{resklogits_2025,
  title={ReskLogits: GPU-Accelerated Shadow Ban Logits Processor},
  author={RESK},
  year={2025},
  url={https://github.com/Resk-Security/resk-logits}
}
```
