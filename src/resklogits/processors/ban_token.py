from typing import List, Optional, Set

import torch
from transformers import LogitsProcessor


class BanTokenProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        banned_tokens: Optional[List[str]] = None,
        banned_token_ids: Optional[List[int]] = None,
    ):
        self.banned_ids: Set[int] = set()

        if banned_token_ids is not None:
            self.banned_ids.update(banned_token_ids)

        if banned_tokens is not None:
            for token in banned_tokens:
                encoded = tokenizer.encode(token, add_special_tokens=False)
                self.banned_ids.update(encoded)

        if not self.banned_ids:
            import warnings

            warnings.warn(
                "BanTokenProcessor created with no banned tokens. "
                "The processor will be a no-op.",
                stacklevel=2,
            )

        self._banned_tensor: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.banned_ids:
            return scores

        if self._device != scores.device:
            self._banned_tensor = torch.tensor(
                list(self.banned_ids), device=scores.device, dtype=torch.long
            )
            self._device = scores.device

        scores[:, self._banned_tensor] = -float("inf")
        return scores

    def add_banned_tokens(self, token_ids: List[int]):
        self.banned_ids.update(token_ids)
        self._banned_tensor = None

    def add_banned_strings(self, tokenizer, tokens: List[str]):
        for token in tokens:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            self.banned_ids.update(encoded)
        self._banned_tensor = None

    def remove_banned_tokens(self, token_ids: List[int]):
        for tid in token_ids:
            self.banned_ids.discard(tid)
        self._banned_tensor = None

    def __repr__(self) -> str:
        return f"BanTokenProcessor(banned_tokens={len(self.banned_ids)})"
