"""
CLI Tool for Rule Generation

Command-line interface for generating, testing, and managing rules.
"""

import argparse
import json
import sys
from pathlib import Path

from .cache import RuleCache
from .config_parser import ConfigParser, load_rules_from_yaml


def cmd_generate(args):
    """Generate patterns from YAML rules."""
    print(f"Generating patterns from: {args.rules}")

    try:
        parser = ConfigParser()
        results = parser.generate_all_patterns(
            args.rules,
            use_synonyms=args.synonyms,
            use_cache=not args.no_cache,
            force=args.force
        )

        # Flatten patterns
        all_patterns = []
        for rule_name, patterns in results.items():
            all_patterns.extend(patterns)

        unique_patterns = list(set(all_patterns))

        print(f"✓ Generated {len(unique_patterns)} unique patterns from {len(results)} rules")

        if args.output:
            # Save to JSON
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                if args.by_category:
                    json.dump(results, f, indent=2)
                else:
                    json.dump(unique_patterns, f, indent=2)
            print(f"✓ Saved to: {args.output}")
        else:
            # Print to stdout
            if args.by_category:
                print(json.dumps(results, indent=2))
            else:
                for pattern in sorted(unique_patterns)[:args.preview]:
                    print(f"  - {pattern}")
                if len(unique_patterns) > args.preview:
                    print(f"  ... and {len(unique_patterns) - args.preview} more")

        return 0

    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_test(args):
    """Test rule matching against text."""
    print(f"Testing rules from: {args.rules}")
    print(f"Test text: \"{args.text}\"")
    print()

    try:
        patterns = load_rules_from_yaml(args.rules, use_cache=not args.no_cache)

        matches = [p for p in patterns if p in args.text.lower()]

        if matches:
            print(f"✓ Found {len(matches)} matches:")
            for match in matches:
                print(f"  - {match}")
        else:
            print("✗ No matches found")

        return 0

    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1


def cmd_expand(args):
    """Expand and preview patterns for a specific rule."""
    print(f"Expanding rule: {args.rule}")

    try:
        parser = ConfigParser()
        all_rules = parser.parse_file(args.rules)

        if args.rule not in all_rules:
            print(f"✗ Rule '{args.rule}' not found", file=sys.stderr)
            print(f"Available rules: {', '.join(all_rules.keys())}")
            return 1

        rule_config = all_rules[args.rule]
        patterns = parser.generate_patterns(rule_config, use_synonyms=not args.no_synonyms)

        print(f"✓ Generated {len(patterns)} patterns:")
        for pattern in sorted(patterns)[:args.preview]:
            print(f"  - {pattern}")
        if len(patterns) > args.preview:
            print(f"  ... and {len(patterns) - args.preview} more")

        return 0

    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1


def cmd_cache_status(args):
    """Show cache status."""
    cache = RuleCache(args.cache_dir)

    stats = cache.get_stats()
    print("Cache Status:")
    print(f"  Directory: {stats['cache_dir']}")
    print(f"  Entries: {stats['total_entries']}")
    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  Size: {stats['cache_size_mb']:.2f} MB")
    print()

    if args.verbose and stats['total_entries'] > 0:
        print("Entries:")
        for entry in cache.list_entries()[:10]:
            print(f"  {entry['hash']}: {entry['pattern_count']} patterns (accessed: {entry['last_accessed_str']})")

    return 0


def cmd_cache_clear(args):
    """Clear cache."""
    cache = RuleCache(args.cache_dir)

    if args.hash:
        cache.clear(args.hash)
        print(f"✓ Cleared cache entry: {args.hash}")
    else:
        stats = cache.get_stats()
        if args.yes or input(f"Clear all {stats['total_entries']} cache entries? (y/N): ").lower() == 'y':
            cache.clear()
            print("✓ Cache cleared")
        else:
            print("Cancelled")

    return 0


def cmd_cache_show(args):
    """Show cache entry details."""
    cache = RuleCache(args.cache_dir)

    try:
        patterns = cache.load(args.hash)
        metadata = cache.get_metadata(args.hash)

        print(f"Cache Entry: {args.hash}")
        print(f"  Patterns: {len(patterns)}")

        if metadata:
            print("  Metadata:")
            for key, value in metadata.items():
                print(f"    {key}: {value}")

        if args.show_patterns:
            print("\n  Patterns:")
            for pattern in patterns[:args.limit]:
                print(f"    - {pattern}")
            if len(patterns) > args.limit:
                print(f"    ... and {len(patterns) - args.limit} more")

        return 0

    except FileNotFoundError:
        print(f"✗ Cache entry not found: {args.hash}", file=sys.stderr)
        return 1


def cmd_validate(args):
    """Validate YAML rules file."""
    print(f"Validating: {args.rules}")

    try:
        parser = ConfigParser()
        rules = parser.parse_file(args.rules)

        print("✓ Valid YAML configuration")
        print(f"  Rules: {len(rules)}")

        for name, config in rules.items():
            print(f"  - {name}: severity={config.severity}, penalty={config.penalty}")

        return 0

    except Exception as e:
        print(f"✗ Validation failed: {e}", file=sys.stderr)
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ReskLogits Rule Generator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate patterns from rules')
    gen_parser.add_argument('rules', help='Path to YAML rules file')
    gen_parser.add_argument('-o', '--output', help='Output file (JSON)')
    gen_parser.add_argument('--no-synonyms', action='store_true', help='Disable synonym expansion')
    gen_parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    gen_parser.add_argument('--force', action='store_true', help='Force regeneration')
    gen_parser.add_argument('--by-category', action='store_true', help='Group output by rule category')
    gen_parser.add_argument('--preview', type=int, default=20, help='Number of patterns to preview')
    gen_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test rule matching')
    test_parser.add_argument('rules', help='Path to YAML rules file')
    test_parser.add_argument('--text', required=True, help='Text to test against')
    test_parser.add_argument('--no-cache', action='store_true', help='Disable caching')

    # Expand command
    expand_parser = subparsers.add_parser('expand', help='Expand and preview a specific rule')
    expand_parser.add_argument('rules', help='Path to YAML rules file')
    expand_parser.add_argument('--rule', required=True, help='Rule name to expand')
    expand_parser.add_argument('--no-synonyms', action='store_true', help='Disable synonym expansion')
    expand_parser.add_argument('--preview', type=int, default=50, help='Number of patterns to show')

    # Cache subcommands
    cache_parser = subparsers.add_parser('cache', help='Cache management')
    cache_subparsers = cache_parser.add_subparsers(dest='cache_command')

    cache_status_parser = cache_subparsers.add_parser('status', help='Show cache status')
    cache_status_parser.add_argument('--cache-dir', default='.resklogits_cache', help='Cache directory')
    cache_status_parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed info')

    cache_clear_parser = cache_subparsers.add_parser('clear', help='Clear cache')
    cache_clear_parser.add_argument('--cache-dir', default='.resklogits_cache', help='Cache directory')
    cache_clear_parser.add_argument('--hash', help='Clear specific entry')
    cache_clear_parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation')

    cache_show_parser = cache_subparsers.add_parser('show', help='Show cache entry')
    cache_show_parser.add_argument('hash', help='Cache entry hash')
    cache_show_parser.add_argument('--cache-dir', default='.resklogits_cache', help='Cache directory')
    cache_show_parser.add_argument('--show-patterns', action='store_true', help='Show patterns')
    cache_show_parser.add_argument('--limit', type=int, default=20, help='Pattern limit')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate YAML rules file')
    validate_parser.add_argument('rules', help='Path to YAML rules file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    if args.command == 'generate':
        return cmd_generate(args)
    elif args.command == 'test':
        return cmd_test(args)
    elif args.command == 'expand':
        return cmd_expand(args)
    elif args.command == 'cache':
        if args.cache_command == 'status':
            return cmd_cache_status(args)
        elif args.cache_command == 'clear':
            return cmd_cache_clear(args)
        elif args.cache_command == 'show':
            return cmd_cache_show(args)
        else:
            cache_parser.print_help()
            return 1
    elif args.command == 'validate':
        return cmd_validate(args)

    return 1


if __name__ == '__main__':
    sys.exit(main())

