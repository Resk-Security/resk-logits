from contextlib import contextmanager
from typing import Generator, List, Optional

import torch
from transformers import LogitsProcessor


class ForceLastPhraseLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        phrase: str,
        trigger_length: Optional[int] = None,
    ):
        self.phrase_ids: List[int] = tokenizer.encode(phrase, add_special_tokens=False)
        self.trigger_length = trigger_length
        self.position: int = -1

    def force_now(self):
        self.position = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        seq_len = input_ids.shape[-1]
        batch_size = input_ids.shape[0]

        if self.position >= len(self.phrase_ids):
            return scores

        if self.position >= 0:
            for batch_idx in range(batch_size):
                scores[batch_idx, :] = -float("inf")
                scores[batch_idx, self.phrase_ids[self.position]] = 0.0
            self.position += 1
            return scores

        if self.trigger_length is not None and seq_len >= self.trigger_length:
            self.position = 0
            for batch_idx in range(batch_size):
                scores[batch_idx, :] = -float("inf")
                scores[batch_idx, self.phrase_ids[0]] = 0.0
            self.position += 1

        return scores

    def reset(self):
        self.position = -1

    @contextmanager
    def stream(self) -> Generator[None, None, None]:
        self.reset()
        try:
            yield
        finally:
            self.reset()

    def __repr__(self) -> str:
        return (
            f"ForceLastPhraseLogitsProcessor("
            f"phrase_len={len(self.phrase_ids)}, "
            f"trigger={self.trigger_length})"
        )
