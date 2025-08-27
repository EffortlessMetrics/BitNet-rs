#!/usr/bin/env python3
"""
Async streaming example for bitnet_py

This example demonstrates the async streaming capabilities of the bitnet_py
library, showing how to generate text incrementally with proper async/await
support and cancellation handling.
"""

import asyncio
import bitnet_py as bitnet
import time
import sys
from typing import Iterator

def stream_tokens(engine: bitnet.InferenceEngine, prompt: str) -> Iterator[str]:
    """
    Stream tokens from the inference engine using the new streaming API.
    """
    # Use the actual streaming generator
    stream = engine.generate_stream(prompt)
    for token in stream:
        yield token

async def generate_with_timeout(engine: bitnet.SimpleInference, prompt: str, timeout: float = 30.0) -> str:
    """Generate text with a timeout to prevent hanging."""
    try:
        response = await asyncio.wait_for(
            engine.generate_stream(prompt),
            timeout=timeout
        )
        return response
    except asyncio.TimeoutError:
        return f"[Generation timed out after {timeout} seconds]"

async def concurrent_generation(engine: bitnet.SimpleInference, prompts: list[str]) -> list[str]:
    """Generate responses for multiple prompts concurrently."""
    tasks = [generate_with_timeout(engine, prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)

async def interactive_chat(engine: bitnet.SimpleInference, tokenizer: bitnet.Tokenizer):
    """Interactive chat session with streaming responses."""
    print("Interactive Chat Session (type 'quit' to exit)")
    print("=" * 50)
    
    # Create chat format tokenizer
    chat_tokenizer = bitnet.ChatFormat(tokenizer)
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Add user message to history
            user_message = bitnet.Message(role="user", content=user_input)
            conversation_history.append(user_message)
            
            # Create dialog prompt
            dialog_tokens = chat_tokenizer.encode_dialog_prompt(
                dialog=conversation_history,
                completion=True,
                return_target=False,
            )
            
            # Convert tokens back to text for the engine
            dialog_text = tokenizer.decode(dialog_tokens)
            
            # Generate streaming response
            print("Assistant: ", end="", flush=True)
            
            full_response = ""
            async for token in stream_tokens(engine, dialog_text):
                print(token, end="", flush=True)
                full_response += token
            
            print()  # New line after response
            
            # Add assistant response to history
            assistant_message = bitnet.Message(role="assistant", content=full_response.strip())
            conversation_history.append(assistant_message)
            
        except KeyboardInterrupt:
            print("\n\nChat session interrupted.")
            break
        except Exception as e:
            print(f"\nError during chat: {e}")
            continue
    
    print("Chat session ended.")

async def benchmark_async_performance(engine: bitnet.SimpleInference, prompts: list[str]):
    """Benchmark async generation performance."""
    print("Benchmarking async performance...")
    
    # Sequential generation
    print("Sequential generation:")
    start_time = time.time()
    sequential_results = []
    for prompt in prompts:
        result = await engine.generate_stream(prompt)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    print(f"  Time: {sequential_time:.2f} seconds")
    print(f"  Tokens/second: {sum(len(r.split()) for r in sequential_results) / sequential_time:.2f}")
    
    # Concurrent generation
    print("Concurrent generation:")
    start_time = time.time()
    concurrent_results = await concurrent_generation(engine, prompts)
    concurrent_time = time.time() - start_time
    
    print(f"  Time: {concurrent_time:.2f} seconds")
    print(f"  Tokens/second: {sum(len(r.split()) for r in concurrent_results) / concurrent_time:.2f}")
    print(f"  Speedup: {sequential_time / concurrent_time:.2f}x")

async def main():
    print("BitNet.cpp Python Bindings - Async Streaming Example")
    print("=" * 55)
    
    if len(sys.argv) < 2:
        print("Usage: python async_streaming.py <model_path> [tokenizer_path]")
        print("\nExample:")
        print("  python async_streaming.py models/bitnet_b1_58-3B/ggml-model-i2_s.gguf")
        return 1
    
    model_path = sys.argv[1]
    tokenizer_path = sys.argv[2] if len(sys.argv) > 2 else "./tokenizer.model"
    
    try:
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        model = bitnet.load_model(model_path, device="cpu")
        tokenizer = bitnet.create_tokenizer(tokenizer_path)
        
        # Create inference engine with streaming config
        config = bitnet.InferenceConfig(
            max_new_tokens=100,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
        )
        
        engine = bitnet.SimpleInference(model, tokenizer, config)
        print("Setup completed successfully!")
        print()
        
        # Test basic streaming
        print("Testing basic streaming generation...")
        test_prompt = "The future of artificial intelligence"
        print(f"Prompt: {test_prompt}")
        print("Response: ", end="", flush=True)
        
        async for token in stream_tokens(engine, test_prompt):
            print(token, end="", flush=True)
        print("\n")
        
        # Test concurrent generation
        print("Testing concurrent generation...")
        test_prompts = [
            "Hello, my name is",
            "The capital of France is",
            "In the year 2024,",
            "Machine learning is",
        ]
        
        concurrent_results = await concurrent_generation(engine, test_prompts)
        
        for prompt, result in zip(test_prompts, concurrent_results):
            print(f"Prompt: {prompt}")
            print(f"Result: {result}")
            print("-" * 40)
        
        # Benchmark async performance
        await benchmark_async_performance(engine, test_prompts)
        print()
        
        # Interactive chat (optional)
        if len(sys.argv) > 3 and sys.argv[3] == "--interactive":
            await interactive_chat(engine, tokenizer)
        else:
            print("Add --interactive flag for interactive chat session")
        
    except bitnet.BitNetError as e:
        print(f"BitNet Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("Async streaming example completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))