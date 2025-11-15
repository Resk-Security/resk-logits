"""
ReskLogits - GPU-Accelerated Shadow Ban Logits Processor

Ultra-fast vectorized Aho-Corasick pattern matching for LLM safety filtering
with symbolic rule generation and intelligent caching.
"""

from .cache import CachedGenerator, RuleCache
from .config_parser import ConfigParser, load_rules_from_yaml
from .pattern_automata import PatternExpander, SynonymGraph
from .rule_engine import ContainsRule, ExactRule, Rule, RuleEngine, StartsWithRule
from .rule_templates import Template, TemplateEngine
from .shadow_ban_processor import MultiLevelShadowBanProcessor, ShadowBanProcessor
from .vectorized_aho_corasick import VectorizedAhoCorasick

__version__ = "0.1.0"
__author__ = "RESK Team"
__all__ = [
    # Core logits processors
    "VectorizedAhoCorasick",
    "ShadowBanProcessor",
    "MultiLevelShadowBanProcessor",
    # Rule generation
    "ConfigParser",
    "load_rules_from_yaml",
    "TemplateEngine",
    "Template",
    "RuleEngine",
    "Rule",
    "ExactRule",
    "StartsWithRule",
    "ContainsRule",
    "PatternExpander",
    "SynonymGraph",
    # Caching
    "RuleCache",
    "CachedGenerator",
]
