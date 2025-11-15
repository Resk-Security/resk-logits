"""
Logic Rule Engine for Pattern Matching

Provides symbolic rule system with logic operators (AND, OR, NOT, XOR)
and pattern matching capabilities.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional


class LogicOperator(Enum):
    """Logic operators for rule composition."""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    XOR = "XOR"


class Rule(ABC):
    """Abstract base class for rules."""

    @abstractmethod
    def matches(self, text: str) -> bool:
        """Check if rule matches given text."""
        pass

    @abstractmethod
    def generate_patterns(self) -> List[str]:
        """Generate patterns from this rule."""
        pass

    def __and__(self, other: "Rule") -> "CompositeRule":
        """Compose rules with AND operator."""
        return CompositeRule(LogicOperator.AND, [self, other])

    def __or__(self, other: "Rule") -> "CompositeRule":
        """Compose rules with OR operator."""
        return CompositeRule(LogicOperator.OR, [self, other])

    def __invert__(self) -> "CompositeRule":
        """Negate rule with NOT operator."""
        return CompositeRule(LogicOperator.NOT, [self])

    def __xor__(self, other: "Rule") -> "CompositeRule":
        """Compose rules with XOR operator."""
        return CompositeRule(LogicOperator.XOR, [self, other])


class ExactRule(Rule):
    """Rule that matches exact phrases."""

    def __init__(self, phrases: List[str]):
        self.phrases = [p.lower() for p in phrases]

    def matches(self, text: str) -> bool:
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in self.phrases)

    def generate_patterns(self) -> List[str]:
        return self.phrases.copy()

    def __repr__(self):
        return f"ExactRule({len(self.phrases)} phrases)"


class StartsWithRule(Rule):
    """Rule that matches text starting with specific prefixes."""

    def __init__(self, prefixes: List[str]):
        self.prefixes = [p.lower() for p in prefixes]

    def matches(self, text: str) -> bool:
        text_lower = text.lower()
        return any(text_lower.startswith(prefix) for prefix in self.prefixes)

    def generate_patterns(self) -> List[str]:
        return self.prefixes.copy()

    def __repr__(self):
        return f"StartsWithRule({len(self.prefixes)} prefixes)"


class ContainsRule(Rule):
    """Rule that matches text containing specific keywords."""

    def __init__(self, keywords: List[str]):
        self.keywords = [k.lower() for k in keywords]

    def matches(self, text: str) -> bool:
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.keywords)

    def generate_patterns(self) -> List[str]:
        return self.keywords.copy()

    def __repr__(self):
        return f"ContainsRule({len(self.keywords)} keywords)"


class EndsWithRule(Rule):
    """Rule that matches text ending with specific suffixes."""

    def __init__(self, suffixes: List[str]):
        self.suffixes = [s.lower() for s in suffixes]

    def matches(self, text: str) -> bool:
        text_lower = text.lower()
        return any(text_lower.endswith(suffix) for suffix in self.suffixes)

    def generate_patterns(self) -> List[str]:
        return self.suffixes.copy()

    def __repr__(self):
        return f"EndsWithRule({len(self.suffixes)} suffixes)"


class CompositeRule(Rule):
    """Rule composed of multiple rules with logic operators."""

    def __init__(self, operator: LogicOperator, rules: List[Rule]):
        self.operator = operator
        self.rules = rules

    def matches(self, text: str) -> bool:
        """Evaluate composite rule."""
        if self.operator == LogicOperator.AND:
            return all(rule.matches(text) for rule in self.rules)
        elif self.operator == LogicOperator.OR:
            return any(rule.matches(text) for rule in self.rules)
        elif self.operator == LogicOperator.NOT:
            return not self.rules[0].matches(text)
        elif self.operator == LogicOperator.XOR:
            matches = [rule.matches(text) for rule in self.rules]
            return sum(matches) == 1
        return False

    def generate_patterns(self) -> List[str]:
        """Generate patterns from composite rule."""
        if self.operator == LogicOperator.NOT:
            # NOT rules don't generate patterns
            return []

        # For AND/OR/XOR, combine patterns from all subrules
        all_patterns = []
        for rule in self.rules:
            all_patterns.extend(rule.generate_patterns())

        # Remove duplicates while preserving order
        seen = set()
        unique_patterns = []
        for pattern in all_patterns:
            if pattern not in seen:
                seen.add(pattern)
                unique_patterns.append(pattern)

        return unique_patterns

    def __repr__(self):
        return f"CompositeRule({self.operator.value}, {len(self.rules)} rules)"


class PatternRule(Rule):
    """Rule for generating patterns from templates."""

    def __init__(self, patterns: List[str]):
        self.patterns = [p.lower() for p in patterns]

    def matches(self, text: str) -> bool:
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self.patterns)

    def generate_patterns(self) -> List[str]:
        return self.patterns.copy()

    def __repr__(self):
        return f"PatternRule({len(self.patterns)} patterns)"


class CombinationRule(Rule):
    """Rule that generates combinations of components."""

    def __init__(self, name: str):
        self.name = name
        self.components: Dict[str, List[str]] = {}

    def with_component(self, component_name: str, values: List[str]) -> "CombinationRule":
        """Add a component with possible values."""
        self.components[component_name] = values
        return self

    def with_action(self, actions: List[str]) -> "CombinationRule":
        """Convenience method for action component."""
        return self.with_component("action", actions)

    def with_object(self, objects: List[str]) -> "CombinationRule":
        """Convenience method for object component."""
        return self.with_component("object", objects)

    def with_prefix(self, prefixes: List[str]) -> "CombinationRule":
        """Convenience method for prefix component."""
        return self.with_component("prefix", prefixes)

    def combine(self) -> List[str]:
        """Generate all combinations of components."""
        if not self.components:
            return []

        from itertools import product

        # Get component names and values in consistent order
        names = sorted(self.components.keys())
        value_lists = [self.components[name] for name in names]

        # Generate all combinations
        patterns = []
        for combo in product(*value_lists):
            # Join with spaces
            pattern = " ".join(combo)
            patterns.append(pattern)

        return patterns

    def matches(self, text: str) -> bool:
        """Check if any combination matches."""
        patterns = self.combine()
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in patterns)

    def generate_patterns(self) -> List[str]:
        return self.combine()

    def __repr__(self):
        return f"CombinationRule({self.name}, {len(self.components)} components)"


class RuleEngine:
    """Engine for managing and evaluating rules."""

    def __init__(self) -> None:
        self.rules: Dict[str, Rule] = {}

    def add_rule(self, name: str, rule: Rule):
        """Add a named rule to the engine."""
        self.rules[name] = rule

    def get_rule(self, name: str) -> Rule:
        """Get rule by name."""
        if name not in self.rules:
            raise KeyError(f"Rule '{name}' not found")
        return self.rules[name]

    def evaluate(self, name: str, text: str) -> bool:
        """Evaluate a rule against text."""
        rule = self.get_rule(name)
        return rule.matches(text)

    def generate_patterns(self, name: Optional[str] = None) -> List[str]:
        """
        Generate patterns from rules.

        Args:
            name: Specific rule name (None = all rules)

        Returns:
            List of generated patterns
        """
        if name:
            return self.get_rule(name).generate_patterns()
        else:
            all_patterns = []
            for rule in self.rules.values():
                all_patterns.extend(rule.generate_patterns())

            # Deduplicate
            return list(set(all_patterns))

    def load_from_config(self, config: Dict[str, Any]):
        """
        Load rules from configuration dict.

        Expected format:
        {
            "rule_name": {
                "type": "exact|starts_with|contains|ends_with",
                "values": ["pattern1", "pattern2"]
            }
        }
        """
        for name, rule_config in config.items():
            rule_type = rule_config.get("type", "exact")
            values = rule_config.get("values", [])

            rule: Rule
            if rule_type == "exact":
                rule = ExactRule(values)
            elif rule_type == "starts_with":
                rule = StartsWithRule(values)
            elif rule_type == "contains":
                rule = ContainsRule(values)
            elif rule_type == "ends_with":
                rule = EndsWithRule(values)
            else:
                raise ValueError(f"Unknown rule type: {rule_type}")

            self.add_rule(name, rule)

    def __repr__(self):
        return f"RuleEngine({len(self.rules)} rules)"
