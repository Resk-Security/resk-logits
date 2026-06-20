"""
Tests for the new logits processors and vLLM adapter.
"""

import pytest
import torch
from transformers import AutoTokenizer


def _get_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


# ── GenLengthLogitsProcessor ────────────────────────────────────────────────


def test_gen_length_imports():
    from resklogits import GenLengthLogitsProcessor
    assert GenLengthLogitsProcessor is not None


def test_gen_length_penalises_eos_below_min():
    from resklogits import GenLengthLogitsProcessor

    tokenizer = _get_tokenizer()
    processor = GenLengthLogitsProcessor(
        tokenizer, min_length=10, eos_penalty=-100.0
    )

    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 3))
    scores = torch.randn(1, vocab_size)

    original_eos = scores[0, tokenizer.eos_token_id].clone()
    modified = processor(input_ids, scores)
    assert modified[0, tokenizer.eos_token_id] < original_eos


def test_gen_length_boosts_eos_above_max():
    from resklogits import GenLengthLogitsProcessor

    tokenizer = _get_tokenizer()
    processor = GenLengthLogitsProcessor(
        tokenizer, max_length=5, eos_boost=100.0
    )

    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 10))
    scores = torch.randn(1, vocab_size)

    original_eos = scores[0, tokenizer.eos_token_id].clone()
    modified = processor(input_ids, scores)
    assert modified[0, tokenizer.eos_token_id] > original_eos


def test_gen_length_no_modification_in_middle():
    from resklogits import GenLengthLogitsProcessor

    tokenizer = _get_tokenizer()
    processor = GenLengthLogitsProcessor(
        tokenizer, min_length=0, max_length=100
    )

    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 5))
    scores = torch.randn(1, vocab_size)

    original = scores.clone()
    modified = processor(input_ids, scores)
    assert torch.equal(original, modified)


# ── CiteFromPromptLogitsProcessor ───────────────────────────────────────────


def test_cite_from_prompt_imports():
    from resklogits import CiteFromPromptLogitsProcessor
    assert CiteFromPromptLogitsProcessor is not None


def test_cite_from_prompt_boosts_prompt_tokens():
    from resklogits import CiteFromPromptLogitsProcessor

    tokenizer = _get_tokenizer()
    prompt_ids = tokenizer.encode("The quick brown fox", return_tensors="pt")[0]

    processor = CiteFromPromptLogitsProcessor(
        tokenizer, prompt_ids=prompt_ids, boost_factor=50.0
    )

    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 5))
    scores = torch.randn(1, vocab_size)

    original = scores.clone()
    modified = processor(input_ids, scores)

    # Prompt tokens should have increased
    for tid in torch.unique(prompt_ids):
        if tid < vocab_size:
            assert modified[0, tid] > original[0, tid]


def test_cite_from_prompt_with_empty_prompt_ids():
    from resklogits import CiteFromPromptLogitsProcessor

    tokenizer = _get_tokenizer()
    processor = CiteFromPromptLogitsProcessor(
        tokenizer, prompt_ids=[], boost_factor=2.0
    )

    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 5))
    scores = torch.randn(1, vocab_size)

    original = scores.clone()
    modified = processor(input_ids, scores)
    assert torch.equal(original, modified)


# ── ForceLastPhraseLogitsProcessor ──────────────────────────────────────────


def test_force_last_phrase_imports():
    from resklogits import ForceLastPhraseLogitsProcessor
    assert ForceLastPhraseLogitsProcessor is not None


def test_force_last_phrase_triggers_at_length():
    from resklogits import ForceLastPhraseLogitsProcessor

    tokenizer = _get_tokenizer()
    phrase = "Done."
    processor = ForceLastPhraseLogitsProcessor(
        tokenizer, phrase=phrase, trigger_length=5
    )

    vocab_size = tokenizer.vocab_size
    # Below trigger length
    input_ids = torch.randint(0, vocab_size, (1, 3))
    scores = torch.randn(1, vocab_size)
    modified = processor(input_ids, scores)
    # Should not force yet
    assert modified.shape == scores.shape

    # At trigger length
    input_ids = torch.randint(0, vocab_size, (1, 5))
    scores = torch.randn(1, vocab_size)
    modified = processor(input_ids, scores)
    # Should force the first phrase token
    assert modified[0, processor.phrase_ids[0]] == 0.0


def test_force_last_phrase_force_now():
    from resklogits import ForceLastPhraseLogitsProcessor

    tokenizer = _get_tokenizer()
    phrase = "END"
    processor = ForceLastPhraseLogitsProcessor(tokenizer, phrase=phrase)
    processor.force_now()

    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 3))
    scores = torch.randn(1, vocab_size)
    modified = processor(input_ids, scores)

    assert modified[0, processor.phrase_ids[0]] == 0.0


def test_force_last_phrase_reset():
    from resklogits import ForceLastPhraseLogitsProcessor

    tokenizer = _get_tokenizer()
    processor = ForceLastPhraseLogitsProcessor(
        tokenizer, phrase="Test", trigger_length=5
    )

    processor.force_now()
    assert processor.position == 0
    processor.reset()
    assert processor.position == -1


# ── MultipleChoiceLogitsProcessor ───────────────────────────────────────────


def test_multiple_choice_imports():
    from resklogits import MultipleChoiceLogitsProcessor
    assert MultipleChoiceLogitsProcessor is not None


def test_multiple_choice_restricts_to_valid_tokens():
    from resklogits import MultipleChoiceLogitsProcessor

    tokenizer = _get_tokenizer()
    processor = MultipleChoiceLogitsProcessor(
        tokenizer, choices=["0", "1", "2", "3"]
    )

    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 5))
    scores = torch.randn(1, vocab_size)

    modified = processor(input_ids, scores)

    # Non-choice tokens should be -inf
    for tid in range(min(vocab_size, 50)):
        if tid not in processor.valid_token_ids:
            assert modified[0, tid] == -float("inf")

    # Choice tokens should retain their original values
    for tid in processor.valid_token_ids:
        if tid < vocab_size:
            assert modified[0, tid] == scores[0, tid]


def test_multiple_choice_raises_on_empty():
    from resklogits import MultipleChoiceLogitsProcessor

    tokenizer = _get_tokenizer()
    with pytest.raises(ValueError, match="No valid tokens"):
        MultipleChoiceLogitsProcessor(tokenizer, choices=[])


# ── BanTokenProcessor ───────────────────────────────────────────────────────


def test_ban_token_imports():
    from resklogits import BanTokenProcessor
    assert BanTokenProcessor is not None


def test_ban_token_blocks_by_id():
    from resklogits import BanTokenProcessor

    tokenizer = _get_tokenizer()
    processor = BanTokenProcessor(
        tokenizer, banned_token_ids=[0, 1, 2]
    )

    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 5))
    scores = torch.randn(1, vocab_size)

    modified = processor(input_ids, scores)

    assert modified[0, 0] == -float("inf")
    assert modified[0, 1] == -float("inf")
    assert modified[0, 2] == -float("inf")


def test_ban_token_blocks_by_string():
    from resklogits import BanTokenProcessor

    tokenizer = _get_tokenizer()
    processor = BanTokenProcessor(
        tokenizer, banned_tokens=["rm", "DROP"]
    )

    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 5))
    scores = torch.randn(1, vocab_size)

    modified = processor(input_ids, scores)

    # Encode the banned strings and verify they're blocked
    for token in ["rm", "DROP"]:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        for tid in encoded:
            assert modified[0, tid] == -float("inf")


def test_ban_token_add_remove():
    from resklogits import BanTokenProcessor

    tokenizer = _get_tokenizer()
    processor = BanTokenProcessor(tokenizer, banned_token_ids=[0])

    assert 0 in processor.banned_ids

    processor.add_banned_tokens([5, 10])
    assert 5 in processor.banned_ids
    assert 10 in processor.banned_ids

    processor.remove_banned_tokens([0])
    assert 0 not in processor.banned_ids


# ── vLLM Adapter ────────────────────────────────────────────────────────────


def test_to_vllm_imports():
    from resklogits import VLLMWrapper, to_vllm, to_vllm_list
    assert VLLMWrapper is not None
    assert to_vllm is not None
    assert to_vllm_list is not None


def test_to_vllm_wraps_processor():
    from resklogits import BanTokenProcessor, to_vllm

    tokenizer = _get_tokenizer()
    original = BanTokenProcessor(tokenizer, banned_token_ids=[0])
    wrapped = to_vllm(original)

    assert isinstance(wrapped, object)
    assert hasattr(wrapped, "__call__")


def test_vllm_wrapper_call_signature():
    from resklogits import BanTokenProcessor, to_vllm

    tokenizer = _get_tokenizer()
    original = BanTokenProcessor(tokenizer, banned_token_ids=[0])
    wrapped = to_vllm(original)

    # vLLM-style call: (token_ids: List[int], logits: Tensor) -> Tensor
    vocab_size = tokenizer.vocab_size
    token_ids = [1, 2, 3]
    logits = torch.randn(vocab_size)

    modified = wrapped(token_ids, logits)

    assert modified.shape == (vocab_size,)
    assert modified[0] == -float("inf")


def test_vllm_list():
    from resklogits import (
        BanTokenProcessor,
        GenLengthLogitsProcessor,
        to_vllm_list,
    )

    tokenizer = _get_tokenizer()
    processors = [
        BanTokenProcessor(tokenizer, banned_token_ids=[0]),
        GenLengthLogitsProcessor(tokenizer, min_length=5),
    ]

    wrapped_list = to_vllm_list(processors)
    assert len(wrapped_list) == 2


# ── TriggerPhraseLogitsProcessor ────────────────────────────────────────────


def test_trigger_phrase_imports():
    from resklogits import TriggerPhraseLogitsProcessor
    assert TriggerPhraseLogitsProcessor is not None


def test_trigger_phrase_no_match_passthrough():
    from resklogits import TriggerPhraseLogitsProcessor

    tokenizer = _get_tokenizer()
    processor = TriggerPhraseLogitsProcessor(
        tokenizer, trigger="```python", response="print('hello')"
    )

    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 5))
    scores = torch.randn(1, vocab_size)

    original = scores.clone()
    modified = processor(input_ids, scores)
    assert torch.equal(original, modified)


def test_trigger_phrase_detection_triggers_response():
    from resklogits import TriggerPhraseLogitsProcessor

    tokenizer = _get_tokenizer()
    trigger = "Hello"
    response = " World"
    processor = TriggerPhraseLogitsProcessor(
        tokenizer, trigger=trigger, response=response
    )

    # Build input_ids ending with the trigger
    trigger_ids = tokenizer.encode(trigger, add_special_tokens=False)
    vocab_size = tokenizer.vocab_size
    scores = torch.randn(1, vocab_size)

    # Step through each trigger token
    for i, tid in enumerate(trigger_ids):
        input_ids = torch.tensor([trigger_ids[: i + 1]])
        scores = torch.randn(1, vocab_size)
        original = scores.clone()
        modified = processor(input_ids, scores)

        if i < len(trigger_ids) - 1:
            # Trigger not yet complete
            assert torch.equal(original, modified)
        else:
            # Trigger complete → first response token forced
            expected_id = processor.response_ids[0]
            assert modified[0, expected_id] == 0.0
            total_inf = (modified == -float("inf")).sum().item()
            assert total_inf >= vocab_size - 1


def test_trigger_phrase_full_flow():
    from resklogits import TriggerPhraseLogitsProcessor

    tokenizer = _get_tokenizer()
    processor = TriggerPhraseLogitsProcessor(
        tokenizer, trigger="Hi", response=" there"
    )

    trigger_ids = processor.trigger_ids
    response_ids = processor.response_ids
    vocab_size = tokenizer.vocab_size

    # Step through trigger — last step also forces first response token
    for i in range(len(trigger_ids)):
        input_ids = torch.tensor([trigger_ids[: i + 1]])
        scores = torch.randn(1, vocab_size)
        modified = processor(input_ids, scores)

        if i == len(trigger_ids) - 1:
            # Trigger just completed → first response token forced
            assert modified[0, response_ids[0]] == 0.0

    # Remaining response tokens (skip position 0, already forced)
    for i in range(1, len(response_ids)):
        scores = torch.randn(1, vocab_size)
        modified = processor(input_ids, scores)
        assert modified[0, response_ids[i]] == 0.0

    # After response, should be done → passthrough
    scores = torch.randn(1, vocab_size)
    modified = processor(input_ids, scores)
    assert modified is scores


def test_trigger_phrase_reset():
    from resklogits import TriggerPhraseLogitsProcessor

    tokenizer = _get_tokenizer()
    processor = TriggerPhraseLogitsProcessor(
        tokenizer, trigger="Hi", response=" there"
    )

    # Trigger it
    trigger_ids = processor.trigger_ids
    for tid in trigger_ids:
        input_ids = torch.tensor([[tid]])
        scores = torch.randn(1, tokenizer.vocab_size)
        processor(input_ids, scores)

    assert processor.response_position >= 0
    processor.reset()
    assert processor.match_position == 0
    assert processor.response_position == -1
    assert not processor._done


def test_trigger_phrase_raises_on_empty_trigger():
    from resklogits import TriggerPhraseLogitsProcessor

    tokenizer = _get_tokenizer()
    with pytest.raises(ValueError, match="no tokens"):
        TriggerPhraseLogitsProcessor(
            tokenizer, trigger="", response="test"
        )


# ── YAML config loading ─────────────────────────────────────────────────────


def test_load_processors_from_yaml(tmp_path):
    from resklogits.config_parser import load_processors_from_yaml

    tokenizer = _get_tokenizer()

    yaml_content = """
processors:
  gen_length:
    enabled: true
    min_length: 5
    max_length: 100
    eos_penalty: -10.0
    eos_boost: 5.0
  ban_token:
    enabled: true
    banned_tokens:
      - "evil"
      - "bad"
  trigger_phrases:
    - trigger: "```python"
      response: "import os"
"""
    yaml_path = tmp_path / "test_processors.yaml"
    yaml_path.write_text(yaml_content)

    procs = load_processors_from_yaml(str(yaml_path), tokenizer)
    assert len(procs) == 3

    from resklogits import (
        BanTokenProcessor,
        GenLengthLogitsProcessor,
        TriggerPhraseLogitsProcessor,
    )
    assert isinstance(procs[0], GenLengthLogitsProcessor)
    assert isinstance(procs[1], BanTokenProcessor)
    assert isinstance(procs[2], TriggerPhraseLogitsProcessor)


def test_load_processors_from_yaml_no_processors_section(tmp_path):
    from resklogits.config_parser import load_processors_from_yaml

    tokenizer = _get_tokenizer()

    yaml_content = "rules:\n  violence:\n    exact:\n      - bad"
    yaml_path = tmp_path / "no_proc.yaml"
    yaml_path.write_text(yaml_content)

    procs = load_processors_from_yaml(str(yaml_path), tokenizer)
    assert procs == []
