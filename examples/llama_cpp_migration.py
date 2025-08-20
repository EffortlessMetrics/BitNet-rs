#!/usr/bin/env python3
"""
Example showing how to migrate from llama-cpp-python to BitNet.rs

This demonstrates that the same code works with both libraries,
requiring only a single import change!
"""

# Original llama-cpp-python code:
# from llama_cpp import Llama

# BitNet.rs drop-in replacement (ONE LINE CHANGE!):
from bitnet.llama_compat import Llama

def main():
    """
    This is EXACTLY the same code you would use with llama-cpp-python!
    No other changes needed - BitNet.rs handles everything.
    """
    
    # Load model - identical API
    model = Llama(
        model_path="models/bitnet-b1.58-2B.gguf",
        n_ctx=2048,
        n_batch=512,
        n_threads=4,
        n_gpu_layers=32,  # GPU acceleration if available
        verbose=True,
    )
    
    # Tokenization - identical API
    prompt = "The capital of France is"
    tokens = model.tokenize(prompt.encode('utf-8'))
    print(f"Tokens: {tokens}")
    
    # Generation - identical API
    output = model(
        prompt,
        max_tokens=32,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        stop=[".", "\n"],
    )
    
    print(f"Generated: {output['choices'][0]['text']}")
    print(f"Tokens used: {output['usage']}")
    
    # Streaming generation - identical API
    print("\nStreaming generation:")
    for token in model.generate(
        tokens,
        top_k=40,
        top_p=0.95,
        temperature=0.8,
    ):
        text = model.detokenize([token]).decode('utf-8')
        print(text, end='', flush=True)
        
        # Stop after 20 tokens for demo
        if len(tokens) > 20:
            break
    
    print("\n")
    
    # Embeddings - identical API
    embeddings = model.create_embedding("Hello world")
    print(f"Embedding shape: {len(embeddings['data'][0]['embedding'])}")
    
    # Batch completion - identical API
    prompts = [
        "The weather today is",
        "Machine learning is",
        "Python programming is",
    ]
    
    batch_output = model.create_completion(
        prompts,
        max_tokens=20,
        temperature=0.8,
    )
    
    print("\nBatch completions:")
    for i, choice in enumerate(batch_output['choices']):
        print(f"{i+1}. {choice['text']}")
    
    print("\n✅ Migration successful! BitNet.rs works with your existing llama-cpp code!")
    print("Benefits you get with BitNet.rs:")
    print("  • Memory safety (no segfaults)")
    print("  • Better performance (SIMD optimizations)")
    print("  • Handles models that llama.cpp can't (GPT-2 tokenizers)")
    print("  • Built-in HTTP server and streaming")
    print("  • Cross-platform support")


if __name__ == "__main__":
    main()