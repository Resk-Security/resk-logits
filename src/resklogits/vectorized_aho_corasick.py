"""
GPU-Accelerated Vectorized Aho-Corasick Pattern Matcher

This module implements a vectorized version of the Aho-Corasick algorithm
optimized for GPU operations with PyTorch. It pre-computes a danger mask
for ultra-fast token filtering during generation.
"""

import time
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, cast

import torch


class VectorizedAhoCorasick:
    """
    Vectorized Aho-Corasick automaton for GPU-accelerated pattern matching.

    This class builds a trie with failure links and pre-computes a binary mask
    of dangerous tokens for O(1) GPU-based filtering.
    """

    def __init__(self, tokenizer, banned_phrases: List[str], device: str = "cuda", verbose: bool = False):
        """
        Initialize the Aho-Corasick automaton.

        Args:
            tokenizer: HuggingFace tokenizer
            banned_phrases: List of phrases to ban
            device: Device for GPU operations ('cuda' or 'cpu')
            verbose: Print timing information for profiling
        """
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.device = device
        self.verbose = verbose

        t0 = time.time()
        
        # 1. Tokenize all banned phrases using batch processing
        if banned_phrases:
            # Use batch_encode_plus for faster tokenization
            encoded = tokenizer.batch_encode_plus(
                banned_phrases,
                add_special_tokens=False,
                return_attention_mask=False,
                padding=False,
                truncation=False
            )
            self.patterns = [tuple(ids) for ids in encoded['input_ids'] if ids]
        else:
            self.patterns = []
        
        t1 = time.time()
        if self.verbose:
            print(f"[Profile] Tokenization (batch): {t1-t0:.3f}s")

        # 2. Build Aho-Corasick automaton (CPU, one-time at startup)
        self.trie, self.failure, self.output = self._build_aho_corasick()
        
        t2 = time.time()
        if self.verbose:
            print(f"[Profile] Build trie: {t2-t1:.3f}s")

        # 3. Pre-compute state transitions
        self.state_to_tokens = self._precompute_state_transitions()
        
        t3 = time.time()
        if self.verbose:
            print(f"[Profile] Precompute transitions: {t3-t2:.3f}s")

        # 4. Build danger mask (GPU tensor)
        self.danger_mask = self._build_danger_mask()
        
        t4 = time.time()
        if self.verbose:
            print(f"[Profile] Build danger mask: {t4-t3:.3f}s")
            print(f"[Profile] TOTAL BUILD TIME: {t4-t0:.3f}s")

    def _build_aho_corasick(self) -> Tuple[Dict, Dict, Dict]:
        """
        Build the Aho-Corasick automaton with trie and failure links.

        Returns:
            (trie, failure, output) tuple:
            - trie: Dict[state, Dict[token, next_state]]
            - failure: Dict[state, fallback_state]
            - output: Dict[state, List[pattern_indices]] (only direct matches, not failure chain)
        """
        trie: Dict[int, Dict[int, int]] = defaultdict(dict)
        failure: Dict[int, int] = {}
        output: Dict[int, List[int]] = defaultdict(list)
        
        # State counter for efficient state creation
        state_counter = 1  # 0 is root

        # Build trie structure
        for idx, pattern in enumerate(self.patterns):
            node = 0
            for token in pattern:
                if token not in trie[node]:
                    trie[node][token] = state_counter
                    state_counter += 1
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

        # Return without pre-computing all outputs - we'll check failure chain dynamically
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
        Simply marks all tokens that appear in any banned pattern as dangerous.

        Returns:
            Boolean tensor of shape [vocab_size] on GPU
        """
        danger_tokens: Set[int] = set()

        # Add all tokens from all banned patterns
        # This is O(n) where n = total tokens across all patterns
        for pattern in self.patterns:
            danger_tokens.update(pattern)

        # Create GPU mask efficiently using tensor operations
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        if danger_tokens:
            # Convert to sorted list for better memory locality
            token_list = sorted(danger_tokens)
            # Create tensor on CPU first, then move to device if needed
            token_ids = torch.tensor(token_list, dtype=torch.long, device='cpu')
            # Filter out any tokens >= vocab_size
            valid_tokens = token_ids[token_ids < self.vocab_size]
            # Move to target device and set mask
            if self.device == 'cuda':
                valid_tokens = valid_tokens.to('cuda')
            mask[valid_tokens] = True

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
        Follows failure links to check all possible matches.

        Args:
            state: Current state

        Returns:
            True if state has an output (complete match)
        """
        # Check current state
        if state in self.output and len(self.output[state]) > 0:
            return True
        
        # Check states reachable via failure links
        current = state
        visited = set()
        while current in self.failure and current not in visited:
            visited.add(current)
            current = self.failure[current]
            if current in self.output and len(self.output[current]) > 0:
                return True
        
        return False

    def get_matched_patterns(self, state: int) -> List[int]:
        """
        Get indices of matched patterns at current state.
        Follows failure links to collect all matches.

        Args:
            state: Current state

        Returns:
            List of pattern indices that match
        """
        matches: Set[int] = set(self.output.get(state, []))
        # Collect matches from failure chain
        current = state
        visited = set()
        while current in self.failure and current not in visited:
            visited.add(current)
            current = self.failure[current]
            matches.update(self.output.get(current, []))
        
        return list(matches)

    def __repr__(self) -> str:
        return (
            f"VectorizedAhoCorasick("
            f"patterns={len(self.patterns)}, "
            f"states={len(self.trie)}, "
            f"danger_tokens={self.danger_mask.sum().item()})"
        )
