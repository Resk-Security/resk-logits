from typing import List

import torch
from transformers import LogitsProcessor


class TriggerPhraseLogitsProcessor(LogitsProcessor):
    """
    Forces insertion of a response phrase after detecting a trigger phrase.

    Uses a state machine to track the trigger token-by-token. Once the full
    trigger is detected in the generated sequence, the processor switches to
    response-forcing mode and greedily generates the response phrase.

    Typical use cases:
      - Force Python code after ``\\n```python``
      - Insert a citation format after a ``[source]`` marker
      - Auto-complete structured output after a section header
    """

    def __init__(
        self,
        tokenizer,
        trigger: str,
        response: str,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer.
            trigger: The phrase that triggers the forced response.
            response: The phrase to insert after the trigger is detected.
        """
        self.trigger_ids: List[int] = tokenizer.encode(trigger, add_special_tokens=False)
        self.response_ids: List[int] = tokenizer.encode(response, add_special_tokens=False)

        if not self.trigger_ids:
            raise ValueError(f"Trigger phrase '{trigger}' produced no tokens.")

        self.match_position: int = 0
        self.response_position: int = -1
        self._done: bool = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        seq_len = input_ids.shape[-1]

        if self._done:
            return scores

        if self.response_position >= 0:
            self._force_response(scores)
            return scores

        seq_len = input_ids.shape[-1]
        if seq_len > 0:
            last_token = int(input_ids[0, -1].item())
            if last_token == self.trigger_ids[self.match_position]:
                self.match_position += 1
                if self.match_position >= len(self.trigger_ids):
                    self.response_position = 0
                    self._force_response(scores)
            else:
                self.match_position = 1 if last_token == self.trigger_ids[0] else 0

        return scores

    def _force_response(self, scores: torch.FloatTensor):
        batch_size = scores.shape[0]
        if self.response_position < len(self.response_ids):
            for batch_idx in range(batch_size):
                scores[batch_idx, :] = -float("inf")
                scores[batch_idx, self.response_ids[self.response_position]] = 0.0
            self.response_position += 1
        else:
            self._done = True

    def reset(self):
        self.match_position = 0
        self.response_position = -1
        self._done = False

    def __repr__(self) -> str:
        return (
            f"TriggerPhraseLogitsProcessor("
            f"trigger_len={len(self.trigger_ids)}, "
            f"response_len={len(self.response_ids)})"
        )
