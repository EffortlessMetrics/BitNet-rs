#!/usr/bin/env python3
"""
Tokenizer parity verification script.
Tests that BitNet and HF tokenizers produce identical results.
"""

import sys
import json
import subprocess
from pathlib import Path


def test_tokenizer_parity(bitnet_bin: str, model_path: str, tokenizer_path: str, hf_model_id: str):
    """
    Test that BitNet tokenizer matches HF tokenizer exactly.
    """
    # Test strings covering various edge cases
    test_strings = [
        # Basic ASCII
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        
        # Numbers and punctuation
        "Testing 123... Is this working?",
        "Price: $99.99 (20% off!)",
        
        # Unicode and emoji
        "Hello 世界 🌍",
        "Café résumé naïve",
        
        # Code snippets
        "def foo(x): return x * 2",
        "if (x > 0) { console.log('positive'); }",
        
        # Edge cases
        "",
        " ",
        "    ",
        "\n\n",
        "a" * 1000,  # Long string
        
        # Special tokens (if any)
        "<s>",
        "</s>",
        "<pad>",
        "<unk>",
    ]
    
    # Load HF tokenizer
    try:
        from transformers import AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    except ImportError:
        print("Error: transformers library required for parity testing")
        return False
    except Exception as e:
        print(f"Error loading HF tokenizer: {e}")
        return False
    
    all_match = True
    mismatches = []
    
    for text in test_strings:
        # BitNet tokenization (via CLI)
        cmd = [
            bitnet_bin, "tokenize",
            "--tokenizer", tokenizer_path,
            "--text", text,
            "--json"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                print(f"BitNet tokenization failed for: {repr(text)}")
                print(f"Error: {result.stderr}")
                all_match = False
                continue
            
            bitnet_output = json.loads(result.stdout)
            bitnet_ids = bitnet_output.get("token_ids", [])
            
        except Exception as e:
            print(f"Error running BitNet tokenizer: {e}")
            all_match = False
            continue
        
        # HF tokenization
        hf_output = hf_tokenizer(text, add_special_tokens=False)
        hf_ids = hf_output["input_ids"]
        
        # Compare
        if bitnet_ids != hf_ids:
            all_match = False
            mismatches.append({
                "text": text,
                "bitnet": bitnet_ids,
                "hf": hf_ids,
                "diff": f"BitNet: {bitnet_ids[:10]}... HF: {hf_ids[:10]}..."
            })
    
    # Test with special tokens
    print("\nTesting with special tokens...")
    
    for add_special in [True, False]:
        text = "Hello world"
        
        # HF with/without special tokens
        hf_output = hf_tokenizer(text, add_special_tokens=add_special)
        hf_ids = hf_output["input_ids"]
        
        # BitNet equivalent
        cmd = [
            bitnet_bin, "tokenize",
            "--tokenizer", tokenizer_path,
            "--text", text,
            "--json"
        ]
        
        if add_special:
            cmd.append("--add-special-tokens")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                bitnet_output = json.loads(result.stdout)
                bitnet_ids = bitnet_output.get("token_ids", [])
                
                if bitnet_ids != hf_ids:
                    print(f"  Mismatch with add_special={add_special}")
                    print(f"    BitNet: {bitnet_ids}")
                    print(f"    HF: {hf_ids}")
                    all_match = False
                else:
                    print(f"  ✓ Match with add_special={add_special}")
        except Exception as e:
            print(f"  Error with add_special={add_special}: {e}")
    
    # Report results
    if mismatches:
        print(f"\n❌ Found {len(mismatches)} tokenization mismatches:")
        for m in mismatches[:5]:  # Show first 5
            print(f"  Text: {repr(m['text'][:50])}")
            print(f"  {m['diff']}")
    
    if all_match:
        print(f"\n✅ All {len(test_strings)} test strings tokenized identically!")
        return True
    else:
        print(f"\n❌ Tokenizer parity check failed")
        return False


def main():
    import os
    
    bitnet_bin = os.environ.get("BITNET_BIN", "target/release/bitnet")
    model_path = os.environ.get("MODEL_PATH")
    tokenizer_path = os.environ.get("TOKENIZER")
    hf_model_id = os.environ.get("HF_MODEL_ID")
    
    if not all([model_path, tokenizer_path, hf_model_id]):
        print("Error: Set MODEL_PATH, TOKENIZER, and HF_MODEL_ID environment variables")
        sys.exit(1)
    
    success = test_tokenizer_parity(bitnet_bin, model_path, tokenizer_path, hf_model_id)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()