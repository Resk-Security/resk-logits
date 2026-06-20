import torch
from transformers import LogitsProcessor


class CiteFromPromptLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        prompt_ids: torch.LongTensor,
        boost_factor: float = 2.0,
    ):
        self.vocab_size = tokenizer.vocab_size
        self.boost_factor = boost_factor

        prompt_ids_t = torch.as_tensor(prompt_ids, dtype=torch.long)
        unique_ids = torch.unique(prompt_ids_t)
        unique_ids = unique_ids[unique_ids < self.vocab_size]

        self.prompt_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.prompt_mask[unique_ids] = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.prompt_mask.any():
            prompt_mask = self.prompt_mask.to(scores.device)
            scores[:, prompt_mask] += self.boost_factor
        return scores

    def __repr__(self) -> str:
        return (
            f"CiteFromPromptLogitsProcessor("
            f"boost={self.boost_factor}, "
            f"prompt_tokens={int(self.prompt_mask.sum().item())})"
        )
