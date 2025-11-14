"""
Quick verification script to test resklogits installation
"""

import sys

def verify_imports():
    """Verify all imports work."""
    print("Testing imports...")
    try:
        from resklogits import VectorizedAhoCorasick, ShadowBanProcessor, MultiLevelShadowBanProcessor
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def verify_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    try:
        from resklogits import ShadowBanProcessor
        from transformers import AutoTokenizer
        import torch
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Create processor
        banned = ["test phrase"]
        processor = ShadowBanProcessor(
            tokenizer=tokenizer,
            banned_phrases=banned,
            shadow_penalty=-10.0,
            device="cpu"
        )
        
        # Test call
        dummy_ids = torch.randint(0, 1000, (1, 5))
        dummy_scores = torch.randn(1, tokenizer.vocab_size)
        result = processor(dummy_ids, dummy_scores)
        
        assert result.shape == dummy_scores.shape
        print("✓ Basic functionality working")
        return True
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False


def verify_version():
    """Check version info."""
    print("\nChecking version...")
    try:
        import resklogits
        print(f"✓ Version: {resklogits.__version__}")
        return True
    except Exception as e:
        print(f"✗ Version check failed: {e}")
        return False


def main():
    print("=" * 60)
    print("ReskLogits Installation Verification")
    print("=" * 60)
    print()
    
    results = []
    results.append(verify_imports())
    results.append(verify_version())
    results.append(verify_basic_functionality())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All checks passed! ReskLogits is ready to use.")
        return 0
    else:
        print("✗ Some checks failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

