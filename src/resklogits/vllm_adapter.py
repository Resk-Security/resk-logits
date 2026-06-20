from typing import Callable, List

import torch
from transformers import LogitsProcessor

VLLMLogitsProcessorFn = Callable[[List[int], torch.Tensor], torch.Tensor]


class VLLMWrapper:
    def __init__(self, processor: LogitsProcessor):
        self.processor = processor

    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        input_ids = torch.tensor([token_ids], device=logits.device, dtype=torch.long)
        scores = logits.unsqueeze(0)
        modified = self.processor(input_ids, scores)
        return modified.squeeze(0)

    def __repr__(self) -> str:
        return f"VLLMWrapper({self.processor!r})"


def to_vllm(processor: LogitsProcessor) -> VLLMWrapper:
    return VLLMWrapper(processor)


def to_vllm_list(processors: List[LogitsProcessor]) -> List[VLLMWrapper]:
    return [to_vllm(p) for p in processors]
