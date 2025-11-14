"""
Demo: Cache Management

Demonstrates cache functionality for rule generation.
"""

from pathlib import Path

from resklogits.cache import RuleCache
from resklogits.config_parser import load_rules_from_yaml


def main():
    print("=" * 80)
    print("Cache Management Demo")
    print("=" * 80)
    print()

    # Initialize cache
    cache = RuleCache()

    # Demo 1: Cache status
    print("Demo 1: Cache Status")
    print("-" * 80)
    stats = cache.get_stats()
    print(f"Cache directory: {stats['cache_dir']}")
    print(f"Total entries: {stats['total_entries']}")
    print(f"Total patterns: {stats['total_patterns']}")
    print(f"Size: {stats['cache_size_mb']:.2f} MB")
    print()

    # Demo 2: Generate with cache
    print("Demo 2: Generate Patterns (creates cache)")
    print("-" * 80)

    rules_file = Path(__file__).parent / "rules.yaml"

    if rules_file.exists():
        # Read content for hashing
        with open(rules_file, 'r') as f:
            rules_content = f.read()

        rule_hash = cache.compute_hash(rules_content)
        print(f"Rule hash: {rule_hash}")

        # Generate patterns (will cache)
        patterns = load_rules_from_yaml(str(rules_file), use_cache=True)
        print(f"✓ Generated {len(patterns)} patterns")

        # Check if cached
        if cache.exists(rule_hash):
            print("✓ Patterns cached successfully")
        print()

    # Demo 3: List cache entries
    print("Demo 3: List Cache Entries")
    print("-" * 80)

    entries = cache.list_entries()
    if entries:
        print(f"Found {len(entries)} cache entries:")
        for entry in entries[:5]:
            print(f"  Hash: {entry['hash']}")
            print(f"    Patterns: {entry['pattern_count']}")
            print(f"    Created: {entry['created_str']}")
            print(f"    Last accessed: {entry['last_accessed_str']}")
            print()
    else:
        print("No cache entries found")
    print()

    # Demo 4: Load from cache
    if entries:
        print("Demo 4: Load from Cache")
        print("-" * 80)

        first_hash = entries[0]['hash']
        cached_patterns = cache.load(first_hash)
        print(f"✓ Loaded {len(cached_patterns)} patterns from cache")

        # Show metadata
        metadata = cache.get_metadata(first_hash)
        if metadata:
            print("Metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        print()

    # Demo 5: Cache invalidation
    print("Demo 5: Cache Invalidation")
    print("-" * 80)
    print("Modifying rules content...")

    modified_content = rules_content + "\n# Modified"
    modified_hash = cache.compute_hash(modified_content)

    print(f"Original hash: {rule_hash}")
    print(f"Modified hash: {modified_hash}")
    print(f"Hashes match: {rule_hash == modified_hash}")
    print()

    if rule_hash != modified_hash:
        print("✓ Cache invalidation works - different content = different hash")
    print()

    # Demo 6: Cache cleanup
    print("Demo 6: Cache Cleanup Options")
    print("-" * 80)
    print("Available cleanup operations:")
    print("  - Clear all: cache.clear()")
    print("  - Clear specific: cache.clear(hash)")
    print("  - Cleanup old: cache.cleanup_old(max_age_days=30)")
    print()
    print("Note: Not executing cleanup in demo")
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print("✓ Cache provides instant pattern loading")
    print("✓ Hash-based invalidation ensures correctness")
    print("✓ Metadata tracking for debugging")
    print("✓ Easy cleanup and management")
    print()
    print("💡 Use CLI for cache management:")
    print("   resklogits cache status")
    print("   resklogits cache clear")
    print("   resklogits cache show <hash>")


if __name__ == "__main__":
    main()

