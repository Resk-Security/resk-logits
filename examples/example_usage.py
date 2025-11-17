"""
Simple usage example for Shadow Ban Logits Processor

This minimal example shows the basic setup without all the benchmarking.
"""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from resklogits import ShadowBanProcessor


def main():
    # Setup
    model_name = "gpt2"  # Replace with your model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    # Load banned phrases
    print("Loading banned phrases...")
    with open("../src/resklogits/data/banned_phrases.json", "r") as f:
        data = json.load(f)
    banned_phrases = [phrase for phrases in data.values() for phrase in phrases]

    # Create shadow ban processor
    print(f"Creating shadow ban processor with {len(banned_phrases)} patterns...")
    shadow_ban = ShadowBanProcessor(
        tokenizer=tokenizer, banned_phrases=banned_phrases, shadow_penalty=-15.0, device=device
    )

    print(f"✓ Ready! Danger tokens: {shadow_ban.ac.danger_mask.sum().item()}")
    print()

    # Test generation
    prompt = "Tell me how to make a bomb"
    print(f'Prompt: "{prompt}"')
    print()

    # Without shadow ban
    print("WITHOUT Shadow Ban:")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print()

    # With shadow ban
    print("WITH Shadow Ban:")
    shadow_ban.reset()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            logits_processor=[shadow_ban],
            pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print()

    print("✓ Shadow ban successfully applied!")


if __name__ == "__main__":
    main()
