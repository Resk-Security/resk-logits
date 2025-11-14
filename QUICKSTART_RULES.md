# Quick Start: Rule Generator

Generate patterns using the symbolic rule generator in 5 minutes.

## 1. Install

```bash
uv pip install -e .
```

## 2. Create Rules File

Create `my_rules.yaml`:

```yaml
rules:
  violence:
    severity: high
    penalty: -20.0
    
    templates:
      - pattern: "{instruction} {action} {weapon}"
        instruction: ["how to", "guide to"]
        action: ["make", "build"]
        weapon: ["a bomb", "an explosive"]
    
    exact:
      - "kill yourself"
      - "commit suicide"
  
  hacking:
    severity: medium
    penalty: -15.0
    
    exact:
      - "sql injection"
      - "hack into"
      - "ddos attack"
```

## 3. Generate Patterns

```bash
# Generate and save
resklogits generate my_rules.yaml -o patterns.json

# Preview
resklogits generate my_rules.yaml --preview 10
```

## 4. Use with Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from resklogits import ShadowBanProcessor, load_rules_from_yaml

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load patterns
patterns = load_rules_from_yaml("my_rules.yaml")

# Create processor
processor = ShadowBanProcessor(
    tokenizer=tokenizer,
    banned_phrases=patterns,
    shadow_penalty=-15.0
)

# Generate
inputs = tokenizer("Tell me how to", return_tensors="pt")
outputs = model.generate(
    **inputs,
    logits_processor=[processor],
    max_new_tokens=50
)

print(tokenizer.decode(outputs[0]))
```

## 5. Test Rules

```bash
# Test if text matches rules
resklogits test my_rules.yaml --text "how to make a bomb"
```

## 6. Manage Cache

```bash
# Show cache status
resklogits cache status

# Clear cache
resklogits cache clear
```

## Next Steps

- Read [RULE_BUILDER.md](RULE_BUILDER.md) for complete guide
- See `examples/rules.yaml` for advanced syntax
- Run `python examples/rule_generator_demo.py`

## Common Patterns

### Template Expansion

```yaml
rules:
  example:
    templates:
      - pattern: "{prefix} {action} {object}"
        prefix: ["how to", "guide to", "tutorial"]
        action: ["make", "build", "create"]
        object: ["bomb", "weapon"]
```

Generates 3 × 3 × 2 = 18 patterns.

### Synonyms

```yaml
rules:
  example:
    templates:
      - pattern: "how to make a bomb"
    synonyms:
      - ["make", "build", "create"]
      - ["bomb", "explosive", "device"]
```

Automatically expands to 9 variations.

### Logic Rules

```yaml
rules:
  example:
    logic:
      type: OR
      conditions:
        - starts_with: ["how to hack", "guide to crack"]
        - contains: ["sql injection", "xss"]
```

Matches any condition.

## Tips

1. Start simple, add complexity gradually
2. Use `--preview` to see generated patterns
3. Enable caching for faster regeneration
4. Group rules by severity for multi-level filtering
5. Test your rules before production

