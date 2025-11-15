"""
Demo: Symbolic Rule Generator

Demonstrates how to generate patterns from YAML rules using templates,
logic, and caching.
"""

import json
from pathlib import Path

from resklogits.cache import RuleCache
from resklogits.config_parser import ConfigParser, load_rules_from_yaml


def main():
    print("=" * 80)
    print("Symbolic Rule Generator Demo")
    print("=" * 80)
    print()

    # Path to rules file
    rules_file = Path(__file__).parent / "rules.yaml"

    if not rules_file.exists():
        print(f"[ERROR] Rules file not found: {rules_file}")
        return 1

    # Example 1: Basic pattern generation
    print("Example 1: Generate patterns from YAML")
    print("-" * 80)

    parser = ConfigParser()
    results = parser.generate_all_patterns(str(rules_file), use_synonyms=True, use_cache=True)

    print(f"[OK] Generated patterns from {len(results)} rules:")
    for rule_name, patterns in results.items():
        print(f"  {rule_name}: {len(patterns)} patterns")
    print()

    # Show sample patterns
    all_patterns = []
    for patterns in results.values():
        all_patterns.extend(patterns)

    print(f"Sample patterns (showing 10 of {len(set(all_patterns))}):")
    for pattern in sorted(set(all_patterns))[:10]:
        print(f"  - {pattern}")
    print()

    # Example 2: Generate specific rule
    print("Example 2: Generate specific rule category")
    print("-" * 80)

    rule_configs = parser.parse_file(str(rules_file))
    violence_config = rule_configs["violence"]
    violence_patterns = parser.generate_patterns(violence_config, use_synonyms=True)

    print(f"[OK] Generated {len(violence_patterns)} patterns for 'violence' rule")
    print("Sample patterns:")
    for pattern in sorted(violence_patterns)[:5]:
        print(f"  - {pattern}")
    print()

    # Example 3: Cache demonstration
    print("Example 3: Cache demonstration")
    print("-" * 80)

    cache = RuleCache()
    stats = cache.get_stats()

    print("Cache status:")
    print(f"  Directory: {stats['cache_dir']}")
    print(f"  Entries: {stats['total_entries']}")
    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  Size: {stats['cache_size_mb']:.2f} MB")
    print()

    # Example 4: Performance with caching
    print("Example 4: Performance with caching")
    print("-" * 80)

    import time

    # With cache (should be instant)
    start = time.time()
    patterns_cached = load_rules_from_yaml(str(rules_file), use_cache=True)
    time_cached = time.time() - start

    # Without cache (regenerate)
    start = time.time()
    patterns_no_cache = load_rules_from_yaml(str(rules_file), use_cache=False, force=True)
    time_no_cache = time.time() - start

    print(f"[OK] With cache: {time_cached:.4f}s ({len(patterns_cached)} patterns)")
    print(f"[OK] Without cache: {time_no_cache:.4f}s ({len(patterns_no_cache)} patterns)")
    print(f"[OK] Speedup: {time_no_cache/time_cached:.1f}x faster with cache")
    print()

    # Example 5: Generate Shadow Ban configuration
    print("Example 5: Generate Shadow Ban configuration")
    print("-" * 80)

    shadow_config = parser.generate_shadow_ban_config(str(rules_file))

    print("[OK] Shadow Ban configuration by severity:")
    for level, patterns in shadow_config["phrases_by_level"].items():
        penalty = shadow_config["penalties"].get(level, -10.0)
        print(f"  {level}: {len(patterns)} patterns (penalty: {penalty})")
    print()

    # Example 6: Save to JSON
    print("Example 6: Save generated patterns")
    print("-" * 80)

    output_file = Path(__file__).parent / "generated_patterns.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "patterns": sorted(set(all_patterns)),
                "count": len(set(all_patterns)),
                "rules": list(results.keys()),
            },
            f,
            indent=2,
        )

    print(f"[OK] Saved patterns to: {output_file}")
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"[OK] Total unique patterns: {len(set(all_patterns))}")
    print(f"[OK] Rules processed: {len(results)}")
    print("[OK] Cache enabled: Yes")
    print("[OK] Synonym expansion: Yes")
    print()
    print("[SUCCESS] Ready to use with ShadowBanProcessor!")


if __name__ == "__main__":
    main()
