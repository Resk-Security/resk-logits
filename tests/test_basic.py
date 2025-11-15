"""
Basic tests for resklogits
"""

import pytest
import torch
from transformers import AutoTokenizer


def test_imports():
    """Test that imports work correctly."""
    from resklogits import MultiLevelShadowBanProcessor, ShadowBanProcessor, VectorizedAhoCorasick

    assert VectorizedAhoCorasick is not None
    assert ShadowBanProcessor is not None
    assert MultiLevelShadowBanProcessor is not None


def test_vectorized_aho_corasick():
    """Test VectorizedAhoCorasick basic functionality."""
    from resklogits import VectorizedAhoCorasick

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    banned_phrases = ["bad word", "dangerous phrase"]

    ac = VectorizedAhoCorasick(tokenizer, banned_phrases, device="cpu")

    assert ac.vocab_size == tokenizer.vocab_size
    assert len(ac.patterns) == 2
    assert ac.danger_mask.shape[0] == tokenizer.vocab_size


def test_shadow_ban_processor():
    """Test ShadowBanProcessor basic functionality."""
    from resklogits import ShadowBanProcessor

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    banned_phrases = ["bad word"]

    processor = ShadowBanProcessor(
        tokenizer=tokenizer, banned_phrases=banned_phrases, shadow_penalty=-10.0, device="cpu"
    )

    # Test call
    batch_size = 1
    seq_len = 5
    vocab_size = tokenizer.vocab_size

    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_scores = torch.randn(batch_size, vocab_size)

    modified_scores = processor(dummy_input_ids, dummy_scores)

    assert modified_scores.shape == dummy_scores.shape
    assert modified_scores.dtype == dummy_scores.dtype


def test_multi_level_processor():
    """Test MultiLevelShadowBanProcessor basic functionality."""
    from resklogits import MultiLevelShadowBanProcessor

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    phrases_by_level = {"high": ["very bad"], "medium": ["somewhat bad"], "low": ["slightly bad"]}

    processor = MultiLevelShadowBanProcessor(
        tokenizer=tokenizer,
        banned_phrases_by_level=phrases_by_level,
        penalties={"high": -20.0, "medium": -10.0, "low": -5.0},
        device="cpu",
    )

    assert len(processor.automatons) == 3
    assert "high" in processor.automatons
    assert "medium" in processor.automatons
    assert "low" in processor.automatons


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
