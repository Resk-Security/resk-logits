"""
Automata-based Pattern Expansion

Implements finite state machines for systematic pattern generation
and synonym expansion through graph traversal.
"""

from collections import defaultdict, deque
from itertools import product
from typing import Dict, List, Set


class SynonymGraph:
    """Graph structure for synonym relationships."""

    def __init__(self):
        self.graph: Dict[str, Set[str]] = defaultdict(set)

    def add_synonym(self, word1: str, word2: str):
        """Add bidirectional synonym relationship."""
        self.graph[word1].add(word2)
        self.graph[word2].add(word1)

    def add_synonym_group(self, words: List[str]):
        """Add a group of mutually synonymous words."""
        for i, word1 in enumerate(words):
            for word2 in words[i+1:]:
                self.add_synonym(word1, word2)

    def get_synonyms(self, word: str, max_depth: int = 1) -> Set[str]:
        """
        Get all synonyms of a word up to max depth.
        
        Args:
            word: Source word
            max_depth: How many hops to traverse (1 = direct synonyms only)
            
        Returns:
            Set of synonyms including the original word
        """
        if word not in self.graph:
            return {word}

        visited = {word}
        queue = deque([(word, 0)])

        while queue:
            current, depth = queue.popleft()

            if depth < max_depth:
                for synonym in self.graph[current]:
                    if synonym not in visited:
                        visited.add(synonym)
                        queue.append((synonym, depth + 1))

        return visited

    def expand_phrase(self, phrase: str, max_depth: int = 1) -> List[str]:
        """
        Expand phrase by replacing words with synonyms.
        
        Args:
            phrase: Input phrase
            max_depth: Synonym traversal depth
            
        Returns:
            List of phrase variations
        """
        words = phrase.split()
        word_variants = [list(self.get_synonyms(word, max_depth)) for word in words]

        # Generate all combinations
        variations = []
        for combo in product(*word_variants):
            variations.append(" ".join(combo))

        return variations

    def __repr__(self):
        return f"SynonymGraph({len(self.graph)} words)"


class PatternFSM:
    """Finite State Machine for pattern generation."""

    def __init__(self):
        self.states: Set[int] = {0}  # State 0 is start state
        self.transitions: Dict[int, Dict[str, int]] = defaultdict(dict)
        self.accepting: Set[int] = set()
        self.state_counter = 1

    def add_state(self, accepting: bool = False) -> int:
        """Add a new state and return its ID."""
        state_id = self.state_counter
        self.states.add(state_id)
        if accepting:
            self.accepting.add(state_id)
        self.state_counter += 1
        return state_id

    def add_transition(self, from_state: int, to_state: int, symbol: str):
        """Add a transition between states."""
        if from_state not in self.states or to_state not in self.states:
            raise ValueError("Invalid state")
        self.transitions[from_state][symbol] = to_state

    def add_pattern(self, pattern: str):
        """Add a linear pattern to the FSM."""
        tokens = pattern.split()
        current_state = 0

        for i, token in enumerate(tokens):
            if token in self.transitions[current_state]:
                # Transition already exists
                current_state = self.transitions[current_state][token]
            else:
                # Create new state and transition
                is_final = (i == len(tokens) - 1)
                next_state = self.add_state(accepting=is_final)
                self.add_transition(current_state, next_state, token)
                current_state = next_state

    def generate_patterns(self, max_length: int = 10) -> List[str]:
        """
        Generate all patterns accepted by the FSM.
        
        Args:
            max_length: Maximum pattern length (in tokens)
            
        Returns:
            List of accepted patterns
        """
        patterns = []

        # BFS through state graph
        queue = deque([(0, [])])  # (state, path)

        while queue:
            state, path = queue.popleft()

            # Check if accepting state
            if state in self.accepting:
                patterns.append(" ".join(path))

            # Don't explore beyond max length
            if len(path) >= max_length:
                continue

            # Explore transitions
            for symbol, next_state in self.transitions[state].items():
                queue.append((next_state, path + [symbol]))

        return patterns

    def __repr__(self):
        return f"PatternFSM({len(self.states)} states, {len(self.accepting)} accepting)"


class GrammarExpander:
    """Expands patterns using grammar rules."""

    def __init__(self):
        self.rules: Dict[str, List[List[str]]] = {}

    def add_rule(self, non_terminal: str, productions: List[List[str]]):
        """
        Add a grammar rule.
        
        Args:
            non_terminal: Non-terminal symbol (e.g., "NP", "VP")
            productions: List of production rules (list of symbols)
        """
        self.rules[non_terminal] = productions

    def expand(self, start_symbol: str, max_depth: int = 3) -> List[str]:
        """
        Expand a start symbol using grammar rules.
        
        Args:
            start_symbol: Symbol to start expansion
            max_depth: Maximum recursion depth
            
        Returns:
            List of terminal strings
        """
        def expand_recursive(symbol: str, depth: int) -> List[List[str]]:
            if depth > max_depth:
                return [[symbol]]

            # Terminal symbol
            if symbol not in self.rules:
                return [[symbol]]

            # Non-terminal - expand all productions
            result = []
            for production in self.rules[symbol]:
                # Expand each symbol in production
                expanded_parts = [expand_recursive(s, depth + 1) for s in production]

                # Cartesian product of all expansions
                for combo in product(*expanded_parts):
                    # Flatten the combination
                    flattened = []
                    for part in combo:
                        flattened.extend(part)
                    result.append(flattened)

            return result

        expansions = expand_recursive(start_symbol, 0)
        return [" ".join(exp) for exp in expansions]

    def __repr__(self):
        return f"GrammarExpander({len(self.rules)} rules)"


class PatternExpander:
    """High-level pattern expansion system combining multiple techniques."""

    def __init__(self):
        self.synonym_graph = SynonymGraph()
        self.fsm = PatternFSM()
        self.grammar = GrammarExpander()

    def add_synonyms(self, word_groups: List[List[str]]):
        """Add synonym groups."""
        for group in word_groups:
            self.synonym_graph.add_synonym_group(group)

    def add_base_patterns(self, patterns: List[str]):
        """Add base patterns to FSM."""
        for pattern in patterns:
            self.fsm.add_pattern(pattern)

    def expand_with_synonyms(self, patterns: List[str], max_depth: int = 1) -> List[str]:
        """Expand patterns using synonyms."""
        expanded = []
        for pattern in patterns:
            expanded.extend(self.synonym_graph.expand_phrase(pattern, max_depth))

        # Deduplicate
        return list(set(expanded))

    def expand_all(self, base_patterns: List[str], use_synonyms: bool = True) -> List[str]:
        """
        Comprehensive pattern expansion.
        
        Args:
            base_patterns: Initial patterns
            use_synonyms: Whether to apply synonym expansion
            
        Returns:
            Expanded pattern list
        """
        # Start with base patterns
        all_patterns = set(base_patterns)

        # Apply synonym expansion
        if use_synonyms:
            for pattern in base_patterns:
                variants = self.synonym_graph.expand_phrase(pattern, max_depth=1)
                all_patterns.update(variants)

        return list(all_patterns)

    def __repr__(self):
        return f"PatternExpander({self.synonym_graph}, {self.fsm})"


# Predefined synonym groups
DEFAULT_SYNONYMS = [
    ["make", "build", "create", "construct", "craft", "produce"],
    ["how to", "guide to", "tutorial", "instructions", "steps to"],
    ["bomb", "explosive", "device"],
    ["weapon", "gun", "firearm"],
    ["kill", "murder", "assassinate", "eliminate"],
    ["hack", "crack", "breach", "exploit", "break into"],
]


def create_default_expander() -> PatternExpander:
    """Create pattern expander with default synonyms."""
    expander = PatternExpander()
    expander.add_synonyms(DEFAULT_SYNONYMS)
    return expander

