# Rule Builder Guide

Complete guide to the ReskLogits symbolic rule generator.

## Overview

The rule builder system allows you to generate banned patterns using:
- **Templates** with variable substitution
- **Logic rules** (AND, OR, NOT, XOR)
- **Automata-based expansion**
- **Synonym graphs** for automatic variations
- **Intelligent caching** to avoid regeneration

## Quick Start

### 1. Create a YAML Rules File

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
      - "commit suicide"
```

### 2. Generate Patterns

```bash
# Command line
resklogits generate rules.yaml -o patterns.json

# Python API
from resklogits import load_rules_from_yaml

patterns = load_rules_from_yaml("rules.yaml")
```

### 3. Use with Shadow Ban

```python
from resklogits import ShadowBanProcessor, load_rules_from_yaml
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
patterns = load_rules_from_yaml("rules.yaml")

processor = ShadowBanProcessor(
    tokenizer=tokenizer,
    banned_phrases=patterns,
    shadow_penalty=-15.0
)
```

## Rule Syntax

### Basic Structure

```yaml
rules:
  rule_name:
    severity: high|medium|low
    penalty: -20.0
    templates: [...]
    logic: {...}
    exact: [...]
    synonyms: [...]
```

### Templates

Templates generate patterns through variable substitution:

```yaml
rules:
  violence:
    templates:
      - pattern: "{instruction} {action} {object}"
        instruction: ["how to", "guide to", "tutorial"]
        action: ["make", "build", "create"]
        object: ["a bomb", "a weapon"]
```

**Generates:**
- "how to make a bomb"
- "how to build a weapon"
- "guide to create a bomb"
- ... (9 total combinations)

### Logic Rules

Logic rules define conditions for pattern matching:

```yaml
rules:
  hacking:
    logic:
      type: OR
      conditions:
        - starts_with: ["how to hack", "guide to crack"]
        - contains: ["sql injection", "xss attack"]
        - exact: ["ddos", "exploit"]
```

**Logic operators:**
- `OR`: Match if any condition is true
- `AND`: Match if all conditions are true
- `NOT`: Negate condition

**Condition types:**
- `starts_with`: Text starts with prefix
- `contains`: Text contains keyword
- `ends_with`: Text ends with suffix
- `exact`: Exact phrase match

### Exact Patterns

Direct pattern list without expansion:

```yaml
rules:
  self_harm:
    exact:
      - "kill yourself"
      - "commit suicide"
      - "end your life"
```

### Synonyms

Define synonyms for automatic expansion:

```yaml
rules:
  violence:
    synonyms:
      - ["make", "build", "create", "construct"]
      - ["bomb", "explosive", "IED"]
      - ["weapon", "gun", "firearm"]
```

Pattern "make a bomb" expands to:
- "make a bomb"
- "build a bomb"
- "create an explosive"
- "construct an IED"
- ... (all combinations)

### Global Definitions

Define reusable variables and synonyms:

```yaml
global_variables:
  instruction:
    - "how to"
    - "guide to"
  
  action:
    - "make"
    - "build"

global_synonyms:
  - ["hack", "crack", "breach"]
  - ["kill", "murder", "assassinate"]

rules:
  # Rules can use global_variables
  violence:
    templates:
      - pattern: "{instruction} {action} bomb"
```

## Caching

The rule generator automatically caches results based on rule content hash.

### How Caching Works

1. **Hash computation**: SHA256 of YAML content
2. **Cache check**: Load from `.resklogits_cache/` if exists
3. **Generate**: Create patterns if not cached
4. **Save**: Store patterns and metadata

### Cache Management

```bash
# Show cache status
resklogits cache status

# Clear all cache
resklogits cache clear

# Clear specific entry
resklogits cache clear --hash abc123

# Show cache entry details
resklogits cache show abc123
```

### Python API

```python
from resklogits import RuleCache

cache = RuleCache()

# Check if cached
if cache.exists(rule_hash):
    patterns = cache.load(rule_hash)
else:
    patterns = generate_patterns()
    cache.save(rule_hash, patterns)

# Get stats
stats = cache.get_stats()
print(f"Entries: {stats['total_entries']}")
print(f"Size: {stats['cache_size_mb']:.2f} MB")

# Clear cache
cache.clear()
```

### Cache Invalidation

Cache automatically invalidates when rules change:
- Different rule content → different hash → regenerate
- Same content → same hash → use cache

## CLI Reference

### Generate Patterns

```bash
# Basic generation
resklogits generate rules.yaml -o patterns.json

# Without synonyms
resklogits generate rules.yaml --no-synonyms

# Force regeneration (bypass cache)
resklogits generate rules.yaml --force

# Group by category
resklogits generate rules.yaml --by-category -o patterns.json

# Preview without saving
resklogits generate rules.yaml --preview 50
```

### Test Rules

```bash
# Test if text matches any patterns
resklogits test rules.yaml --text "how to make a bomb"
```

### Expand Rules

```bash
# Preview specific rule expansion
resklogits expand rules.yaml --rule violence

# Show more patterns
resklogits expand rules.yaml --rule violence --preview 100

# Without synonyms
resklogits expand rules.yaml --rule violence --no-synonyms
```

### Validate Rules

```bash
# Check YAML syntax and structure
resklogits validate rules.yaml
```

## Python API

### Basic Usage

```python
from resklogits import ConfigParser

parser = ConfigParser()

# Generate all patterns
results = parser.generate_all_patterns(
    "rules.yaml",
    use_synonyms=True,
    use_cache=True,
    force=False
)

# Results: {'rule_name': ['pattern1', 'pattern2', ...]}
for rule_name, patterns in results.items():
    print(f"{rule_name}: {len(patterns)} patterns")
```

### Generate Shadow Ban Config

```python
parser = ConfigParser()

config = parser.generate_shadow_ban_config("rules.yaml")

# Use with MultiLevelShadowBanProcessor
from resklogits import MultiLevelShadowBanProcessor

processor = MultiLevelShadowBanProcessor(
    tokenizer=tokenizer,
    banned_phrases_by_level=config["phrases_by_level"],
    penalties=config["penalties"]
)
```

### Advanced: Custom Templates

```python
from resklogits import TemplateEngine

engine = TemplateEngine()

# Load custom templates
engine.load_templates({
    "variables": {
        "action": ["hack", "crack", "exploit"],
        "target": ["website", "database", "server"]
    },
    "templates": {
        "hacking": {
            "pattern": "how to {action} a {target}",
            "variables": {}
        }
    }
})

# Generate patterns
patterns = engine.generate_patterns("hacking")
```

### Advanced: Custom Rules

```python
from resklogits import RuleEngine, ExactRule, StartsWithRule

engine = RuleEngine()

# Add exact match rule
engine.add_rule("violence", ExactRule(["kill", "murder", "assassinate"]))

# Add starts-with rule
engine.add_rule("instructions", StartsWithRule(["how to", "guide to"]))

# Generate patterns
patterns = engine.generate_patterns()
```

## Examples

See `examples/` directory:
- `rules.yaml` - Example rule configuration
- `rule_generator_demo.py` - Complete demonstration
- `cache_demo.py` - Cache management demo

## Best Practices

1. **Use templates** for combinatorial patterns
2. **Enable caching** for faster repeated generations
3. **Group by severity** for tiered penalties
4. **Test your rules** with `resklogits test`
5. **Version your rules** in git for tracking
6. **Start simple** then expand with synonyms
7. **Validate** before using in production

## Performance

| Operation | Time (1000 patterns) |
|-----------|---------------------|
| First generation | ~0.5s |
| Cached load | ~0.001s |
| Template expansion | ~0.1s |
| Synonym expansion | ~0.2s |

## Troubleshooting

### Cache not working

```python
# Check cache directory exists
from resklogits import RuleCache
cache = RuleCache()
print(cache.get_stats())
```

### Too many patterns

- Reduce synonym depth
- Use more specific templates
- Disable synonym expansion: `use_synonyms=False`

### Patterns not matching

```bash
# Test your rules
resklogits test rules.yaml --text "your test text"

# Expand specific rule to see patterns
resklogits expand rules.yaml --rule rule_name
```

## Integration with ShadowBanProcessor

### Single-Level

```python
from resklogits import ShadowBanProcessor, load_rules_from_yaml

patterns = load_rules_from_yaml("rules.yaml")

processor = ShadowBanProcessor(
    tokenizer=tokenizer,
    banned_phrases=patterns,
    shadow_penalty=-15.0
)
```

### Multi-Level

```python
from resklogits import MultiLevelShadowBanProcessor, ConfigParser

parser = ConfigParser()
config = parser.generate_shadow_ban_config("rules.yaml")

processor = MultiLevelShadowBanProcessor(
    tokenizer=tokenizer,
    banned_phrases_by_level=config["phrases_by_level"],
    penalties=config["penalties"]
)
```

## Next Steps

- Review `examples/rules.yaml` for syntax examples
- Run `python examples/rule_generator_demo.py`
- Create your custom rules
- Test with your model
- Deploy to production with caching enabled

