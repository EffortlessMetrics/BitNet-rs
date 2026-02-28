"""
BitNet.cpp Python bindings - High-performance 1-bit LLM inference library

This package provides Python bindings for the BitNet.cpp Rust implementation,
offering significant performance improvements over the original Python implementation
while maintaining API compatibility.

Key Features:
- Drop-in replacement for existing BitNet Python code
- 2-10x performance improvement over Python baseline
- Support for CPU and GPU inference
- Streaming generation with async support
- Comprehensive model format support (GGUF, SafeTensors, HuggingFace)
- Production-ready with proper error handling and logging

Basic Usage:
    >>> import bitnet_py as bitnet
    >>>
    >>> # Load a model
    >>> model = bitnet.load_model("path/to/model.gguf")
    >>> tokenizer = bitnet.create_tokenizer("path/to/tokenizer.model")
    >>>
    >>> # Create inference engine
    >>> engine = bitnet.InferenceEngine(model, tokenizer)
    >>>
    >>> # Generate text
    >>> result = engine.generate("Hello, my name is")
    >>> print(result)

Advanced Usage:
    >>> # Configure generation parameters
    >>> gen_args = bitnet.GenArgs(
    ...     gen_length=128,
    ...     temperature=0.8,
    ...     top_p=0.9,
    ...     use_sampling=True
    ... )
    >>>
    >>> # Create FastGen engine (matching original API)
    >>> engine = bitnet.FastGen.build(
    ...     ckpt_dir="path/to/checkpoint",
    ...     gen_args=gen_args,
    ...     device="cuda:0"
    ... )
    >>>
    >>> # Batch generation
    >>> prompts = ["Hello", "How are you?", "What is AI?"]
    >>> tokens = [tokenizer.encode(p, bos=True, eos=False) for p in prompts]
    >>> stats, results = engine.generate_all(tokens, use_cuda_graphs=True)
    >>>
    >>> for i, prompt in enumerate(prompts):
    ...     answer = tokenizer.decode(results[i])
    ...     print(f"> {prompt}")
    ...     print(answer)

Migration from Original Python:
    The API is designed to be a drop-in replacement. Simply change:

    ```python
    # Old
    import model as fast

    # New
    import bitnet_py as fast
    ```

    All existing code should work without modification.
"""

from ._bitnet_py import (
    # Core classes
    BitNetModel,
    InferenceEngine,
    Tokenizer,
    ChatFormat,

    # Configuration classes
    ModelArgs,
    GenArgs,
    InferenceConfig,

    # Statistics and utilities
    Stats,
    Message,

    # Utility functions
    load_model,
    create_tokenizer,
    benchmark_inference,
    compare_performance,
    validate_outputs,
    get_system_info,

    # Cache functions (matching original API)
    make_cache,
    cache_prefix,

    # Exception types
    BitNetError,

    # Version
    __version__,
)

# Aliases for backward compatibility with original API
FastGen = InferenceEngine
Transformer = BitNetModel

# Re-export everything for convenience
__all__ = [
    # Core classes
    "BitNetModel",
    "InferenceEngine",
    "Tokenizer",
    "ChatFormat",

    # Configuration
    "ModelArgs",
    "GenArgs",
    "InferenceConfig",

    # Utilities
    "Stats",
    "Message",

    # Functions
    "load_model",
    "create_tokenizer",
    "benchmark_inference",
    "compare_performance",
    "validate_outputs",
    "get_system_info",
    "make_cache",
    "cache_prefix",

    # Aliases
    "FastGen",
    "Transformer",

    # Exceptions
    "BitNetError",

    # Version
    "__version__",
]

# Module-level convenience functions for common use cases
def quick_inference(model_path: str, prompt: str, **kwargs) -> str:
    """
    Quick inference function for simple use cases.

    Args:
        model_path: Path to model file (GGUF, SafeTensors, or checkpoint directory)
        prompt: Input text prompt
        **kwargs: Additional arguments for generation

    Returns:
        Generated text string

    Example:
        >>> result = bitnet_py.quick_inference("model.gguf", "Hello, world!")
        >>> print(result)
    """
    model = load_model(model_path)
    tokenizer = create_tokenizer(kwargs.get("tokenizer_path", "./tokenizer.model"))

    config = InferenceConfig(
        max_new_tokens=kwargs.get("max_new_tokens", 128),
        temperature=kwargs.get("temperature", 0.8),
        top_p=kwargs.get("top_p", 0.9),
        do_sample=kwargs.get("do_sample", True),
    )

    from ._bitnet_py import SimpleInference
    engine = SimpleInference(model, tokenizer, config)
    return engine.generate(prompt)

def benchmark_model(model_path: str, prompts: list[str], **kwargs) -> dict:
    """
    Benchmark a model's performance on given prompts.

    Args:
        model_path: Path to model file
        prompts: List of input prompts
        **kwargs: Additional benchmark parameters

    Returns:
        Dictionary with benchmark results

    Example:
        >>> prompts = ["Hello", "How are you?", "What is AI?"]
        >>> results = bitnet_py.benchmark_model("model.gguf", prompts)
        >>> print(f"Tokens/sec: {results['avg_tokens_per_second']:.2f}")
    """
    model = load_model(model_path)
    tokenizer = create_tokenizer(kwargs.get("tokenizer_path", "./tokenizer.model"))

    return benchmark_inference(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        num_runs=kwargs.get("num_runs", 3),
        warmup_runs=kwargs.get("warmup_runs", 1),
    )
