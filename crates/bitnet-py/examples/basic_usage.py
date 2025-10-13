#!/usr/bin/env python3
"""
Basic usage example for bitnet_py - BitNet.cpp Python bindings

This example demonstrates the basic functionality of the bitnet_py library,
showing how to load models, create tokenizers, and perform inference.
"""

import bitnet_py as bitnet
import time
import sys
from pathlib import Path

def main():
    print("BitNet.cpp Python Bindings - Basic Usage Example")
    print("=" * 50)

    # Check if model path is provided
    if len(sys.argv) < 2:
        print("Usage: python basic_usage.py <model_path> [tokenizer_path]")
        print("\nExample:")
        print("  python basic_usage.py models/bitnet_b1_58-3B/ggml-model-i2_s.gguf")
        print("  python basic_usage.py models/checkpoint/ tokenizer.model")
        return

    model_path = sys.argv[1]
    tokenizer_path = sys.argv[2] if len(sys.argv) > 2 else "./tokenizer.model"

    print(f"Model path: {model_path}")
    print(f"Tokenizer path: {tokenizer_path}")
    print()

    try:
        # Load model and tokenizer
        print("Loading model...")
        start_time = time.time()
        model = bitnet.load_model(model_path, device="cpu")
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        print(f"Model info: {model}")
        print(f"Parameters: {model.num_parameters():,}")
        print()

        print("Loading tokenizer...")
        tokenizer = bitnet.create_tokenizer(tokenizer_path)
        print(f"Tokenizer info: {tokenizer}")
        print(f"Vocabulary size: {tokenizer.n_words:,}")
        print()

        # Test basic tokenization
        test_text = "Hello, my name is"
        print(f"Testing tokenization with: '{test_text}'")
        tokens = tokenizer.encode(test_text, bos=True, eos=False)
        decoded = tokenizer.decode(tokens)
        print(f"Tokens: {tokens}")
        print(f"Decoded: '{decoded}'")
        print()

        # Create inference configuration
        config = bitnet.InferenceConfig(
            max_new_tokens=50,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
        )
        print(f"Inference config: {config}")
        print()

        # Create simple inference engine
        print("Creating inference engine...")
        engine = bitnet.SimpleInference(model, tokenizer, config)
        print("Inference engine created successfully")
        print()

        # Test prompts
        prompts = [
            "Hello, my name is",
            "The capital of France is",
            "In the year 2024,",
            "Artificial intelligence is",
        ]

        print("Generating responses...")
        print("-" * 40)

        for i, prompt in enumerate(prompts, 1):
            print(f"Prompt {i}: {prompt}")

            start_time = time.time()
            response = engine.generate(prompt)
            generation_time = time.time() - start_time

            print(f"Response: {response}")
            print(f"Generation time: {generation_time:.2f} seconds")
            print("-" * 40)

        # Benchmark performance
        print("\nRunning performance benchmark...")
        benchmark_results = bitnet.benchmark_inference(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            num_runs=3,
            warmup_runs=1,
        )

        print("Benchmark Results:")
        print(f"  Average tokens/second: {benchmark_results['avg_tokens_per_second']:.2f}")
        print(f"  Average latency: {benchmark_results['avg_latency']:.3f} seconds")
        print(f"  Total tokens generated: {benchmark_results['total_tokens']}")
        print(f"  Total time: {benchmark_results['total_time']:.2f} seconds")

        # System information
        print("\nSystem Information:")
        sys_info = bitnet.get_system_info()
        for key, value in sys_info.items():
            print(f"  {key}: {value}")

    except bitnet.BitNetError as e:
        print(f"BitNet Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

    print("\nExample completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
