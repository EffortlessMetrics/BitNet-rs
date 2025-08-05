#!/usr/bin/env python3
"""
FastGen compatibility example for bitnet_py

This example demonstrates how the bitnet_py library provides a drop-in
replacement for the original FastGen implementation, maintaining full
API compatibility while providing improved performance.
"""

import bitnet_py as fast  # Drop-in replacement for original 'import model as fast'
import time
import sys

def main():
    print("BitNet.cpp Python Bindings - FastGen Compatibility Example")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Usage: python fastgen_compatibility.py <checkpoint_dir>")
        print("\nExample:")
        print("  python fastgen_compatibility.py models/bitnet_checkpoint/")
        return
    
    ckpt_dir = sys.argv[1]
    print(f"Checkpoint directory: {ckpt_dir}")
    print()
    
    try:
        # Create generation arguments (matching original API)
        gen_args = fast.GenArgs(
            gen_length=128,
            gen_bsz=2,
            prompt_length=64,
            use_sampling=True,
            temperature=0.8,
            top_p=0.9,
        )
        print(f"Generation args: {gen_args}")
        print()
        
        # Build FastGen engine (matching original API exactly)
        print("Building FastGen engine...")
        start_time = time.time()
        
        device = "cpu"  # Use "cuda:0" if GPU is available
        g = fast.FastGen.build(
            ckpt_dir=ckpt_dir,
            gen_args=gen_args,
            device=device,
            tokenizer_path="./tokenizer.model",
            num_layers=13,
            use_full_vocab=False,
        )
        
        build_time = time.time() - start_time
        print(f"FastGen engine built in {build_time:.2f} seconds")
        print(f"Engine info: {g}")
        print()
        
        # Test prompts (matching original example)
        prompts = [
            "Hello, my name is",
            "The future of AI is",
        ]
        
        print("Encoding prompts...")
        tokens = [g.tokenizer.encode(prompt, bos=False, eos=False) for prompt in prompts]
        print(f"Encoded tokens: {tokens}")
        print()
        
        # Generate responses (matching original API)
        print("Generating responses...")
        print("-" * 50)
        
        start_time = time.time()
        stats, out_tokens = g.generate_all(
            tokens,
            use_cuda_graphs=False,  # Set to True if using GPU
            use_sampling=gen_args.use_sampling,
        )
        generation_time = time.time() - start_time
        
        # Display results (matching original format)
        for i, prompt in enumerate(prompts):
            print(f"> {prompt}")
            answer = g.tokenizer.decode(out_tokens[i])
            print(answer)
            print("---------------")
        
        # Show statistics (matching original API)
        print(f"Generation completed in {generation_time:.2f} seconds")
        print(f"Stats: {stats.show()}")
        print()
        
        # Test chat format compatibility
        print("Testing chat format compatibility...")
        chat_tokenizer = fast.ChatFormat(g.tokenizer)
        
        # Create dialog messages
        dialog = [
            fast.Message(role="user", content="What is artificial intelligence?"),
        ]
        
        # Encode dialog (matching original API)
        dialog_tokens = chat_tokenizer.encode_dialog_prompt(
            dialog=dialog,
            completion=True,
            return_target=False,
        )
        print(f"Dialog tokens: {dialog_tokens[:10]}... (showing first 10)")
        
        # Generate response for dialog
        stats, dialog_results = g.generate_all(
            [dialog_tokens],
            use_cuda_graphs=False,
            use_sampling=True,
        )
        
        dialog_response = chat_tokenizer.decode(dialog_results[0])
        print(f"Dialog response: {dialog_response}")
        print()
        
        # Performance comparison (if original implementation is available)
        print("Performance Summary:")
        print(f"  Tokens per second: {stats.tokens_per_second:.2f}")
        print(f"  Total tokens: {stats.total_tokens}")
        print(f"  Memory used: {stats.memory_used:.2f} GB")
        print(f"  Prefill time: {stats.prefill_time:.3f} seconds")
        print(f"  Decode time: {stats.decode_time:.3f} seconds")
        
        # Test model cache functions (matching original API)
        print("\nTesting cache functions...")
        model_args = fast.ModelArgs(
            dim=2560,
            n_layers=13,
            n_heads=20,
            n_kv_heads=5,
            vocab_size=128256,
        )
        
        cache = fast.make_cache(
            model_args=model_args,
            length=1000,
            device="cpu",
        )
        print(f"Created cache with {len(cache)} layers")
        
        # Test cache prefix
        prefix_cache = fast.cache_prefix(cache, 500)
        print(f"Created prefix cache with {len(prefix_cache)} layers")
        
    except fast.BitNetError as e:
        print(f"BitNet Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nFastGen compatibility test completed successfully!")
    print("The bitnet_py library provides full API compatibility with the original implementation.")
    return 0

if __name__ == "__main__":
    sys.exit(main())