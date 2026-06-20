from typing import List, Set

import torch
from transformers import LogitsProcessor


class MultipleChoiceLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        choices: List[str],
    ):
        self.choices = choices
        self.valid_token_ids: Set[int] = set()

        for choice in choices:
            tokens = tokenizer.encode(choice, add_special_tokens=False)
            if len(tokens) >= 1:
                self.valid_token_ids.add(tokens[0])

        if not self.valid_token_ids:
            raise ValueError(f"No valid tokens found for choices: {choices}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        valid_ids = torch.tensor(
            list(self.valid_token_ids), device=scores.device, dtype=torch.long
        )
        mask = torch.full_like(scores, -float("inf"))
        mask[:, valid_ids] = scores[:, valid_ids]
        return mask

    def __repr__(self) -> str:
        return (
            f"MultipleChoiceLogitsProcessor("
            f"choices={self.choices}, "
            f"valid_tokens={len(self.valid_token_ids)})"
        )
