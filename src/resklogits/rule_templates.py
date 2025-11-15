"""
Rule Templates System for Pattern Generation

Provides template-based pattern generation with variable substitution
and combinatorial expansion.
"""

import re
from itertools import product
from typing import Any, Dict, List, Optional, cast


class TemplateVariable:
    """Represents a template variable with possible values."""

    def __init__(self, name: str, values: List[str]):
        self.name = name
        self.values = values

    def __repr__(self):
        return f"TemplateVariable({self.name}, {len(self.values)} values)"


class Template:
    """Represents a pattern template with variables."""

    def __init__(self, pattern: str, variables: Dict[str, List[str]]):
        """
        Initialize template.

        Args:
            pattern: Template string with {variable} placeholders
            variables: Dict mapping variable names to possible values
        """
        self.pattern = pattern
        self.variables = {name: TemplateVariable(name, vals) for name, vals in variables.items()}
        self._find_placeholders()

    def _find_placeholders(self):
        """Extract placeholder names from pattern."""
        self.placeholders = re.findall(r"\{(\w+)\}", self.pattern)

        # Validate all placeholders have variables
        for placeholder in self.placeholders:
            if placeholder not in self.variables:
                raise ValueError(
                    f"Placeholder '{placeholder}' in pattern has no variable definition"
                )

    def expand(self) -> List[str]:
        """
        Expand template to all combinations.

        Returns:
            List of expanded patterns
        """
        if not self.placeholders:
            return [self.pattern]

        # Get all combinations of variable values
        variable_combos = product(*[self.variables[ph].values for ph in self.placeholders])

        patterns = []
        for combo in variable_combos:
            # Substitute variables in pattern
            result = self.pattern
            for placeholder, value in zip(self.placeholders, combo, strict=True):
                result = result.replace(f"{{{placeholder}}}", value)
            patterns.append(result)

        return patterns

    def __repr__(self):
        return f"Template({self.pattern}, {len(self.placeholders)} vars)"


class TemplateLibrary:
    """Collection of reusable templates."""

    def __init__(self) -> None:
        self.templates: Dict[str, Template] = {}
        self.variables: Dict[str, List[str]] = {}

    def add_variable(self, name: str, values: List[str]):
        """Add a reusable variable definition."""
        self.variables[name] = values

    def add_template(
        self, name: str, pattern: str, variables: Optional[Dict[str, List[str]]] = None
    ):
        """
        Add a template to the library.

        Args:
            name: Template identifier
            pattern: Template pattern string
            variables: Variable definitions (or None to use library variables)
        """
        if variables is None:
            variables = {}

        # Merge with library variables
        merged_vars = {**self.variables, **variables}

        self.templates[name] = Template(pattern, merged_vars)

    def get_template(self, name: str) -> Template:
        """Get template by name."""
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found")
        return self.templates[name]

    def expand_template(self, name: str) -> List[str]:
        """Expand template to patterns."""
        return self.get_template(name).expand()

    def expand_all(self) -> Dict[str, List[str]]:
        """Expand all templates."""
        return {name: template.expand() for name, template in self.templates.items()}

    def __repr__(self):
        return f"TemplateLibrary({len(self.templates)} templates, {len(self.variables)} variables)"


class TemplateEngine:
    """High-level template processing engine."""

    def __init__(self):
        self.library = TemplateLibrary()

    def load_templates(self, config: Dict[str, Any]):
        """
        Load templates from configuration dict.

        Expected format:
        {
            "variables": {
                "action": ["make", "build", "create"],
                "object": ["bomb", "weapon"]
            },
            "templates": {
                "violence": {
                    "pattern": "{instruction} {action} {object}",
                    "variables": {
                        "instruction": ["how to", "guide to"]
                    }
                }
            }
        }
        """
        # Load global variables
        if "variables" in config:
            for name, values in config["variables"].items():
                self.library.add_variable(name, values)

        # Load templates
        if "templates" in config:
            for name, template_def in config["templates"].items():
                pattern = template_def.get("pattern")
                variables = template_def.get("variables", {})

                if pattern:
                    self.library.add_template(name, pattern, variables)

    def generate_patterns(self, template_name: Optional[str] = None) -> List[str]:
        """
        Generate patterns from templates.

        Args:
            template_name: Specific template to expand (None = all)

        Returns:
            List of generated patterns
        """
        if template_name:
            return cast(List[str], self.library.expand_template(template_name))
        else:
            # Expand all and flatten
            all_patterns = cast(Dict[str, List[str]], self.library.expand_all())
            return [pattern for patterns in all_patterns.values() for pattern in patterns]

    def generate_by_category(self) -> Dict[str, List[str]]:
        """Generate patterns grouped by template name."""
        return cast(Dict[str, List[str]], self.library.expand_all())

    def __repr__(self):
        return f"TemplateEngine({self.library})"


# Predefined common templates
COMMON_TEMPLATES = {
    "variables": {
        "instruction_prefix": [
            "how to",
            "guide to",
            "tutorial",
            "instructions for",
            "steps to",
            "way to",
            "method to",
        ],
        "action_make": [
            "make",
            "build",
            "create",
            "construct",
            "craft",
            "assemble",
            "produce",
            "manufacture",
        ],
        "action_use": ["use", "operate", "employ", "utilize", "apply"],
        "article": ["a", "an", "the", ""],
    },
    "templates": {
        "instruction_action_object": {
            "pattern": "{instruction_prefix} {action_make} {article} {object}",
            "variables": {},
        }
    },
}


def create_default_engine() -> TemplateEngine:
    """Create template engine with common templates."""
    engine = TemplateEngine()
    engine.load_templates(COMMON_TEMPLATES)
    return engine
