"""
Shadow Ban Logits Processor for HuggingFace Transformers

This module implements a LogitsProcessor that uses vectorized Aho-Corasick
to apply GPU-accelerated shadow bans on dangerous tokens during generation.
"""

from typing import Dict, List, Optional

import torch
from transformers import LogitsProcessor

from .vectorized_aho_corasick import VectorizedAhoCorasick


class ShadowBanProcessor(LogitsProcessor):
    """
    GPU-accelerated logits processor with shadow banning.

    Instead of hard-blocking tokens (-inf), this processor applies penalties
    to make dangerous tokens extremely unlikely while maintaining natural generation.

    Key features:
    - Vectorized GPU operations for zero-latency filtering
    - Shadow ban: penalties instead of hard blocks
    - Automatic EOS forcing on complete matches
    - Jailbreak-resistant through stateful pattern matching
    """

    def __init__(
        self,
        tokenizer,
        banned_phrases: List[str],
        shadow_penalty: float = -15.0,
        device: str = "cuda",
    ):
        """
        Initialize the shadow ban processor.

        Args:
            tokenizer: HuggingFace tokenizer
            banned_phrases: List of phrases to shadow ban
            shadow_penalty: Penalty to apply to dangerous tokens (default: -15.0)
                           Typical ranges:
                           -5.0:  Light penalty (~1% chance)
                           -10.0: Medium penalty (~0.005% chance)
                           -15.0: Strong penalty (~0.00003% chance)
                           -20.0: Very strong penalty (virtually impossible)
            device: Device for GPU operations
        """
        self.tokenizer = tokenizer
        self.shadow_penalty = shadow_penalty
        self.device = device

        # Build Aho-Corasick automaton
        self.ac = VectorizedAhoCorasick(tokenizer, banned_phrases, device)

        # State tracking (per batch item)
        self.current_state = 0
        self.states: Dict[int, int] = {}  # batch_idx -> state

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply shadow ban penalties to logits.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            scores: Raw logits [batch_size, vocab_size]

        Returns:
            Modified logits with shadow ban penalties applied
        """
        batch_size = input_ids.shape[0]

        # Initialize states for new batch items
        for batch_idx in range(batch_size):
            if batch_idx not in self.states:
                self.states[batch_idx] = 0

        # Update automaton state based on last generated token
        if input_ids.shape[1] > 0:
            for batch_idx in range(batch_size):
                last_token = int(input_ids[batch_idx, -1].item())
                self.states[batch_idx] = self.ac.step(self.states[batch_idx], last_token)

        # Apply shadow ban: penalize dangerous tokens (GPU vectorized)
        if self.ac.danger_mask.any():
            # Broadcast penalty across batch dimension
            scores[:, self.ac.danger_mask] += self.shadow_penalty

        # Force EOS on complete matches (hard block)
        for batch_idx in range(batch_size):
            if self.ac.has_match(self.states[batch_idx]):
                # Pattern detected - force end of sequence
                scores[batch_idx, :] = -float("inf")
                scores[batch_idx, self.tokenizer.eos_token_id] = 0.0

        return scores

    def reset(self):
        """Reset automaton state (call between generations)."""
        self.current_state = 0
        self.states.clear()

    def get_current_matches(self, batch_idx: int = 0) -> List[str]:
        """
        Get currently matched patterns for a batch item.

        Args:
            batch_idx: Batch index to check

        Returns:
            List of matched phrase strings
        """
        state = self.states.get(batch_idx, 0)
        pattern_indices = self.ac.get_matched_patterns(state)
        return [
            self.tokenizer.decode(self.ac.patterns[idx], skip_special_tokens=True)
            for idx in pattern_indices
        ]

    def __repr__(self) -> str:
        return (
            f"ShadowBanProcessor("
            f"penalty={self.shadow_penalty}, "
            f"patterns={len(self.ac.patterns)}, "
            f"danger_tokens={self.ac.danger_mask.sum().item()})"
        )


class MultiLevelShadowBanProcessor(ShadowBanProcessor):
    """
    Shadow ban processor with tiered penalty levels.

    Applies different penalties based on severity categories.
    """

    def __init__(
        self,
        tokenizer,
        banned_phrases_by_level: dict,
        penalties: Optional[Dict[str, float]] = None,
        device: str = "cuda",
    ):
        """
        Initialize multi-level shadow ban processor.

        Args:
            tokenizer: HuggingFace tokenizer
            banned_phrases_by_level: Dict of {level: [phrases]}
                                    e.g., {'high': [...], 'medium': [...], 'low': [...]}
            penalties: Dict of {level: penalty_value}
                      Defaults: {'high': -20.0, 'medium': -10.0, 'low': -5.0}
            device: Device for GPU operations
        """
        # Default penalties
        if penalties is None:
            penalties = {"high": -20.0, "medium": -10.0, "low": -5.0}

        self.tokenizer = tokenizer
        self.device = device
        self.penalties = penalties

        # Build separate automaton for each level
        self.automatons: Dict[str, VectorizedAhoCorasick] = {}
        for level, phrases in banned_phrases_by_level.items():
            if phrases:
                self.automatons[level] = VectorizedAhoCorasick(tokenizer, phrases, device)

        # State tracking per level (use different name to avoid type conflict with parent)
        self.level_states: Dict[str, Dict[int, int]] = {
            level: {} for level in self.automatons.keys()
        }

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply tiered shadow ban penalties."""
        batch_size = input_ids.shape[0]

        # Initialize states
        for level in self.automatons.keys():
            for batch_idx in range(batch_size):
                if batch_idx not in self.level_states[level]:
                    self.level_states[level][batch_idx] = 0

        # Update states
        if input_ids.shape[1] > 0:
            for batch_idx in range(batch_size):
                last_token = int(input_ids[batch_idx, -1].item())
                for level, ac in self.automatons.items():
                    self.level_states[level][batch_idx] = ac.step(
                        self.level_states[level][batch_idx], last_token
                    )

        # Apply penalties by level
        for level, ac in self.automatons.items():
            penalty = self.penalties.get(level, -10.0)
            if ac.danger_mask.any():
                scores[:, ac.danger_mask] += penalty

        # Force EOS on any high-level matches
        if "high" in self.automatons:
            high_ac = self.automatons["high"]
            for batch_idx in range(batch_size):
                if high_ac.has_match(self.level_states["high"][batch_idx]):
                    scores[batch_idx, :] = -float("inf")
                    scores[batch_idx, self.tokenizer.eos_token_id] = 0.0

        return scores

    def reset(self):
        """Reset all automaton states."""
        for level in self.automatons.keys():
            self.level_states[level].clear()
