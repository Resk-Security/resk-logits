from typing import Optional

import torch
from transformers import LogitsProcessor


class GenLengthLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        min_length: int = 0,
        max_length: Optional[int] = None,
        eos_penalty: float = -10.0,
        eos_boost: float = 5.0,
    ):
        self.eos_token_id = tokenizer.eos_token_id
        self.min_length = min_length
        self.max_length = max_length
        self.eos_penalty = eos_penalty
        self.eos_boost = eos_boost

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        seq_len = input_ids.shape[-1]

        if self.max_length is not None and seq_len >= self.max_length:
            scores[:, self.eos_token_id] += self.eos_boost
        elif seq_len < self.min_length:
            scores[:, self.eos_token_id] += self.eos_penalty

        return scores

    def __repr__(self) -> str:
        return (
            f"GenLengthLogitsProcessor("
            f"min={self.min_length}, "
            f"max={self.max_length}, "
            f"penalty={self.eos_penalty}, "
            f"boost={self.eos_boost})"
        )
