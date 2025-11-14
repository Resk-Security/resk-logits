"""
YAML Config Parser for Symbolic Rules

Parses YAML configuration files and generates patterns using
templates, logic rules, and automata.
"""

from typing import Any, Dict, List

import yaml

from .cache import RuleCache
from .pattern_automata import PatternExpander
from .rule_engine import (
    RuleEngine,
)
from .rule_templates import TemplateEngine


class RuleConfig:
    """Parsed rule configuration."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.severity = config.get("severity", "medium")
        self.penalty = config.get("penalty", -10.0)
        self.templates = config.get("templates", [])
        self.logic = config.get("logic", {})
        self.exact = config.get("exact", [])
        self.synonyms = config.get("synonyms", [])

    def __repr__(self):
        return f"RuleConfig({self.name}, severity={self.severity})"


class ConfigParser:
    """Parser for YAML rule configurations."""

    def __init__(self):
        self.template_engine = TemplateEngine()
        self.rule_engine = RuleEngine()
        self.pattern_expander = PatternExpander()
        self.global_variables = {}  # Store global variables

    def parse_file(self, yaml_path: str) -> Dict[str, RuleConfig]:
        """
        Parse YAML configuration file.
        
        Args:
            yaml_path: Path to YAML file
            
        Returns:
            Dict of rule name -> RuleConfig
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return self.parse_dict(data)

    def parse_dict(self, config: Dict[str, Any]) -> Dict[str, RuleConfig]:
        """
        Parse configuration dictionary.
        
        Args:
            config: Configuration dict
            
        Returns:
            Dict of rule name -> RuleConfig
        """
        rules_config = config.get("rules", {})

        # Load global variables if present
        if "global_variables" in config:
            self.global_variables = config["global_variables"]

        # Load global templates if present
        if "global_templates" in config:
            self.template_engine.load_templates(config["global_templates"])

        # Load global synonyms if present
        if "global_synonyms" in config:
            self.pattern_expander.add_synonyms(config["global_synonyms"])

        # Parse each rule
        parsed_rules = {}
        for name, rule_data in rules_config.items():
            parsed_rules[name] = RuleConfig(name, rule_data)

        return parsed_rules

    def generate_patterns_from_templates(self, templates: List[Dict[str, Any]]) -> List[str]:
        """
        Generate patterns from template definitions.
        
        Args:
            templates: List of template dicts with 'pattern' and variables
            
        Returns:
            List of generated patterns
        """
        all_patterns = []

        for template_def in templates:
            pattern = template_def.get("pattern")
            if not pattern:
                continue

            # Extract variables (all keys except 'pattern')
            template_variables = {k: v for k, v in template_def.items() if k != "pattern"}
            
            # Merge global variables with template-specific variables
            # Template variables override global ones
            merged_variables = {**self.global_variables, **template_variables}

            # Create temporary template engine
            temp_engine = TemplateEngine()
            temp_engine.load_templates({
                "templates": {
                    "temp": {
                        "pattern": pattern,
                        "variables": merged_variables
                    }
                }
            })

            patterns = temp_engine.generate_patterns("temp")
            all_patterns.extend(patterns)

        return all_patterns

    def generate_patterns_from_logic(self, logic: Dict[str, Any]) -> List[str]:
        """
        Generate patterns from logic rules.
        
        Args:
            logic: Logic configuration dict
            
        Returns:
            List of generated patterns
        """
        logic_type = logic.get("type", "OR")
        conditions = logic.get("conditions", [])

        all_patterns = []

        for condition in conditions:
            if "exact" in condition:
                all_patterns.extend(condition["exact"])
            elif "starts_with" in condition:
                all_patterns.extend(condition["starts_with"])
            elif "contains" in condition:
                all_patterns.extend(condition["contains"])
            elif "ends_with" in condition:
                all_patterns.extend(condition["ends_with"])

        return all_patterns

    def generate_patterns(self, rule_config: RuleConfig, use_synonyms: bool = True) -> List[str]:
        """
        Generate all patterns for a rule configuration.
        
        Args:
            rule_config: Parsed rule configuration
            use_synonyms: Whether to expand with synonyms
            
        Returns:
            List of generated patterns
        """
        all_patterns = []

        # Generate from templates
        if rule_config.templates:
            patterns = self.generate_patterns_from_templates(rule_config.templates)
            all_patterns.extend(patterns)

        # Generate from logic rules
        if rule_config.logic:
            patterns = self.generate_patterns_from_logic(rule_config.logic)
            all_patterns.extend(patterns)

        # Add exact patterns
        if rule_config.exact:
            all_patterns.extend(rule_config.exact)

        # Add rule-specific synonyms
        if rule_config.synonyms:
            self.pattern_expander.add_synonyms(rule_config.synonyms)

        # Expand with synonyms if enabled
        if use_synonyms and all_patterns:
            all_patterns = self.pattern_expander.expand_all(all_patterns, use_synonyms=True)

        # Deduplicate and lowercase
        unique_patterns = list(set(p.lower() for p in all_patterns))

        return unique_patterns

    def generate_all_patterns(
        self,
        yaml_path: str,
        use_synonyms: bool = True,
        use_cache: bool = True,
        force: bool = False
    ) -> Dict[str, List[str]]:
        """
        Generate patterns for all rules in configuration.
        
        Args:
            yaml_path: Path to YAML configuration
            use_synonyms: Whether to expand with synonyms
            use_cache: Whether to use caching
            force: Force regeneration even if cached
            
        Returns:
            Dict mapping rule names to pattern lists
        """
        # Read YAML content for hashing
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_content = f.read()

        # Use cache if enabled
        if use_cache and not force:
            cache = RuleCache()
            rule_hash = cache.compute_hash(yaml_content)

            if cache.exists(rule_hash):
                # Load from cache
                patterns = cache.load(rule_hash)
                # Convert flat list back to dict format
                # (For simplicity, return as single category)
                return {"all": patterns}

        # Parse configuration
        rule_configs = self.parse_file(yaml_path)

        # Generate patterns for each rule
        results = {}
        for name, config in rule_configs.items():
            patterns = self.generate_patterns(config, use_synonyms)
            results[name] = patterns

        # Save to cache if enabled
        if use_cache:
            cache = RuleCache()
            rule_hash = cache.compute_hash(yaml_content)

            # Flatten all patterns for cache
            all_patterns = []
            for patterns in results.values():
                all_patterns.extend(patterns)

            metadata = {
                "rule_count": len(results),
                "yaml_file": yaml_path,
                "use_synonyms": use_synonyms
            }
            cache.save(rule_hash, list(set(all_patterns)), metadata)

        return results

    def generate_shadow_ban_config(
        self,
        yaml_path: str,
        use_synonyms: bool = True
    ) -> Dict[str, Any]:
        """
        Generate configuration for ShadowBanProcessor.
        
        Args:
            yaml_path: Path to YAML configuration
            use_synonyms: Whether to expand with synonyms
            
        Returns:
            Dict with 'phrases_by_level' and 'penalties'
        """
        rule_configs = self.parse_file(yaml_path)

        phrases_by_level = {
            "high": [],
            "medium": [],
            "low": []
        }

        penalties = {}

        for name, config in rule_configs.items():
            patterns = self.generate_patterns(config, use_synonyms)

            severity = config.severity.lower()
            if severity in phrases_by_level:
                phrases_by_level[severity].extend(patterns)

            penalties[severity] = config.penalty

        return {
            "phrases_by_level": phrases_by_level,
            "penalties": penalties
        }

    def __repr__(self):
        return "ConfigParser()"


def load_rules_from_yaml(
    yaml_path: str,
    use_synonyms: bool = True,
    use_cache: bool = True,
    force: bool = False
) -> List[str]:
    """
    Convenience function to load rules from YAML.
    
    Args:
        yaml_path: Path to YAML configuration
        use_synonyms: Whether to expand with synonyms
        use_cache: Whether to use caching
        force: Force regeneration
        
    Returns:
        Flat list of all generated patterns
    """
    parser = ConfigParser()
    results = parser.generate_all_patterns(yaml_path, use_synonyms, use_cache, force)

    # Flatten all patterns
    all_patterns = []
    for patterns in results.values():
        all_patterns.extend(patterns)

    return list(set(all_patterns))

