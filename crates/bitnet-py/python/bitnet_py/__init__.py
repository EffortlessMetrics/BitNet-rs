"""
bitnet_py — High-performance 1-bit LLM inference (Rust/PyO3 bindings)

Basic usage::

    import bitnet_py as bitnet

    model = bitnet.load_model("path/to/model.gguf")
    engine = bitnet.InferenceEngine(model, device="cpu")
    print(engine.generate("Hello!", max_tokens=32))

Streaming::

    for token in engine.generate_stream("Once upon a time"):
        print(token, end="", flush=True)
"""

from .bitnet_py import (  # type: ignore[import]
    # Core classes
    BitNetModel,
    InferenceEngine,
    Tokenizer,
    BitNetConfig,
    GenerationConfig,
    ModelLoader,
    ModelInfo,
    StreamingGenerator,
    # Exception hierarchy
    BitNetBaseError,
    ModelError,
    QuantizationError,
    InferenceError,
    KernelError,
    ConfigError,
    ValidationError,
    # Module-level functions
    load_model,
    list_available_models,
    get_device_info,
    set_num_threads,
    batch_generate,
    get_model_info,
    is_cuda_available,
    is_metal_available,
    get_cuda_device_count,
    # Constants
    __version__,
)

__all__ = [
    # Classes
    "BitNetModel",
    "InferenceEngine",
    "Tokenizer",
    "BitNetConfig",
    "GenerationConfig",
    "ModelLoader",
    "ModelInfo",
    "StreamingGenerator",
    # Exceptions
    "BitNetBaseError",
    "ModelError",
    "QuantizationError",
    "InferenceError",
    "KernelError",
    "ConfigError",
    "ValidationError",
    # Functions
    "load_model",
    "list_available_models",
    "get_device_info",
    "set_num_threads",
    "batch_generate",
    "get_model_info",
    "is_cuda_available",
    "is_metal_available",
    "get_cuda_device_count",
    # Version
    "__version__",
]
