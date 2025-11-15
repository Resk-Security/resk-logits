"""
GPU-Accelerated Vectorized Aho-Corasick Pattern Matcher

This module implements a vectorized version of the Aho-Corasick algorithm
optimized for GPU operations with PyTorch. It pre-computes a danger mask
for ultra-fast token filtering during generation.
"""

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, cast

import torch


class VectorizedAhoCorasick:
    """
    Vectorized Aho-Corasick automaton for GPU-accelerated pattern matching.

    This class builds a trie with failure links and pre-computes a binary mask
    of dangerous tokens for O(1) GPU-based filtering.
    """

    def __init__(self, tokenizer, banned_phrases: List[str], device: str = "cuda"):
        """
        Initialize the Aho-Corasick automaton.

        Args:
            tokenizer: HuggingFace tokenizer
            banned_phrases: List of phrases to ban
            device: Device for GPU operations ('cuda' or 'cpu')
        """
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.device = device

        # 1. Tokenize all banned phrases
        self.patterns = []
        for phrase in banned_phrases:
            ids = tokenizer.encode(phrase, add_special_tokens=False)
            if ids:
                self.patterns.append(tuple(ids))

        # 2. Build Aho-Corasick automaton (CPU, one-time at startup)
        self.trie, self.failure, self.output = self._build_aho_corasick()

        # 3. Pre-compute state transitions
        self.state_to_tokens = self._precompute_state_transitions()

        # 4. Build danger mask (GPU tensor)
        self.danger_mask = self._build_danger_mask()

    def _build_aho_corasick(self) -> Tuple[Dict, Dict, Dict]:
        """
        Build the Aho-Corasick automaton with trie and failure links.

        Returns:
            (trie, failure, output) tuple:
            - trie: Dict[state, Dict[token, next_state]]
            - failure: Dict[state, fallback_state]
            - output: Dict[state, List[pattern_indices]]
        """
        trie: Dict[int, Dict[int, int]] = defaultdict(dict)
        failure: Dict[int, int] = {}
        output = defaultdict(list)

        # Build trie structure
        for idx, pattern in enumerate(self.patterns):
            node = 0
            for token in pattern:
                if token not in trie[node]:
                    trie[node][token] = len(trie)
                node = trie[node][token]
            output[node].append(idx)

        # Build failure links using BFS
        queue: deque[int] = deque()

        # Initialize root's children
        for _token, child in trie[0].items():
            failure[child] = 0
            queue.append(child)

        # Build failure links for all other nodes
        while queue:
            current = queue.popleft()

            for token, child in trie[current].items():
                queue.append(child)

                # Find failure link
                fail_state = failure.get(current, 0)
                while fail_state != 0 and token not in trie[fail_state]:
                    fail_state = failure.get(fail_state, 0)

                if token in trie[fail_state]:
                    failure[child] = trie[fail_state][token]
                else:
                    failure[child] = 0

                # Merge output from failure state
                output[child].extend(output[failure[child]])

        return dict(trie), failure, dict(output)

    def _precompute_state_transitions(self) -> Dict[int, Set[int]]:
        """
        Pre-compute which tokens are valid from each state.

        Returns:
            Dict mapping state -> set of valid tokens from that state
        """
        state_to_tokens = defaultdict(set)
        for state, transitions in self.trie.items():
            for token in transitions.keys():
                state_to_tokens[state].add(token)
        return dict(state_to_tokens)

    def _build_danger_mask(self) -> torch.Tensor:
        """
        Build a GPU binary mask indicating dangerous tokens.

        This mask is used for vectorized penalty application.
        The mask marks tokens that:
        1. Appear in any banned pattern
        2. Lead to states with output (complete matches)

        Returns:
            Boolean tensor of shape [vocab_size] on GPU
        """
        danger_tokens: Set[int] = set()

        # Add all tokens from banned patterns
        for pattern in self.patterns:
            danger_tokens.update(pattern)

        # Pre-compute reverse mapping: state -> set of tokens that lead to it
        # This avoids O(n²) nested loops
        state_to_incoming_tokens: Dict[int, Set[int]] = defaultdict(set)
        for parent_state, transitions in self.trie.items():
            for token, next_state in transitions.items():
                state_to_incoming_tokens[next_state].add(token)

        # Add tokens that can lead to matches by traversing the trie
        for state, outputs in self.output.items():
            if outputs:
                # Walk back through failure links to find all contributing tokens
                current = state
                visited = set()

                while current != 0 and current not in visited:
                    visited.add(current)

                    # Use pre-computed reverse mapping (O(1) lookup instead of O(n))
                    if current in state_to_incoming_tokens:
                        danger_tokens.update(state_to_incoming_tokens[current])

                    current = self.failure.get(current, 0)

        # Create GPU mask efficiently using tensor operations
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        if danger_tokens:
            token_ids = torch.tensor(list(danger_tokens), dtype=torch.long, device=self.device)
            valid_mask = token_ids < self.vocab_size
            mask[token_ids[valid_mask]] = True

        return mask

    def step(self, state: int, token: int) -> int:
        """
        Advance the automaton by one token.

        Args:
            state: Current state
            token: Input token ID

        Returns:
            Next state after consuming token
        """
        # Follow failure links until we find a valid transition or reach root
        while state != 0 and token not in self.trie.get(state, {}):
            state = self.failure.get(state, 0)

        # Take transition if it exists
        return cast(int, self.trie.get(state, {}).get(token, 0))

    def has_match(self, state: int) -> bool:
        """
        Check if current state represents a complete pattern match.

        Args:
            state: Current state

        Returns:
            True if state has an output (complete match)
        """
        return state in self.output and len(self.output[state]) > 0

    def get_matched_patterns(self, state: int) -> List[int]:
        """
        Get indices of matched patterns at current state.

        Args:
            state: Current state

        Returns:
            List of pattern indices that match
        """
        return cast(List[int], self.output.get(state, []))

    def __repr__(self) -> str:
        return (
            f"VectorizedAhoCorasick("
            f"patterns={len(self.patterns)}, "
            f"states={len(self.trie)}, "
            f"danger_tokens={self.danger_mask.sum().item()})"
        )
