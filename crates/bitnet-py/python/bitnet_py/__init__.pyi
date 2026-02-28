"""
Type stubs for bitnet_py — accurate to the Rust (PyO3) module surface.

Provides IDE autocompletion and type checking for all exported classes,
functions, constants, and exception types.
"""

from typing import Any, Dict, Iterator, List, Optional

# ── version / constants ──────────────────────────────────────────────
__version__: str
__author__: str
__description__: str

CPU: str
CUDA: str
METAL: str

QuantizationType: Dict[str, str]

# ── exception hierarchy ──────────────────────────────────────────────
class BitNetBaseError(Exception):
    """Base exception for all BitNet errors."""
    ...

class ModelError(BitNetBaseError):
    """Model loading / format errors."""
    ...

class QuantizationError(BitNetBaseError):
    """Quantization-related errors."""
    ...

class InferenceError(BitNetBaseError):
    """Inference runtime errors."""
    ...

class KernelError(BitNetBaseError):
    """Compute kernel errors."""
    ...

class ConfigError(BitNetBaseError):
    """Configuration errors."""
    ...

class ValidationError(BitNetBaseError):
    """Model validation errors."""
    ...


# ── configuration ────────────────────────────────────────────────────
class BitNetConfig:
    """Wrapper around the Rust ``BitNetConfig``."""

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...


class GenerationConfig:
    """Sampling / generation parameters."""

    max_tokens: int
    temperature: float

    def __init__(
        self,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> None: ...
    def __repr__(self) -> str: ...


# ── tokenizer ────────────────────────────────────────────────────────
class Tokenizer:
    """HuggingFace / SentencePiece tokenizer loaded from a pretrained name."""

    vocab_size: int

    def __init__(self, name: str) -> None: ...
    def encode(
        self, text: str, add_special_tokens: Optional[bool] = True
    ) -> List[int]: ...
    def decode(
        self, tokens: List[int], skip_special_tokens: Optional[bool] = None
    ) -> str: ...
    def __repr__(self) -> str: ...


# ── model ────────────────────────────────────────────────────────────
class BitNetModel:
    """A loaded BitNet model (GGUF / SafeTensors)."""

    config: Dict[str, Any]
    device: str
    parameter_count: int
    memory_usage: int
    architecture: str
    quantization: str
    supports_streaming: bool

    def __init__(
        self,
        path: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None: ...
    def info(self) -> Dict[str, Any]: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...


class ModelLoader:
    """Loader that can open models and extract metadata."""

    device: str

    def __init__(self, device: str = "cpu") -> None: ...
    def load(self, path: str, **kwargs: Any) -> BitNetModel: ...
    def extract_metadata(self, path: str) -> Dict[str, Any]: ...
    def available_formats(self) -> List[str]: ...
    def __repr__(self) -> str: ...


class ModelInfo:
    """Lightweight model metadata snapshot (read-only properties)."""

    name: str
    version: str
    architecture: str
    vocab_size: int
    context_length: int
    quantization: Optional[str]
    fingerprint: Optional[str]

    def to_dict(self) -> Dict[str, Any]: ...
    def __repr__(self) -> str: ...


# ── inference engine ─────────────────────────────────────────────────
class InferenceEngine:
    """High-level inference engine wrapping a model and tokenizer."""

    model_config: Dict[str, Any]
    device: str

    def __init__(
        self,
        model: BitNetModel,
        tokenizer: Optional[str] = None,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None: ...

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs: Any,
    ) -> str: ...

    def generate_with_metrics(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Returns ``{"text", "latency_ms", "token_count", "tokens_per_second", ...}``."""
        ...

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs: Any,
    ) -> "StreamingGenerator": ...

    def get_logits(self, token_ids: List[int]) -> List[float]:
        """Run a forward pass on *token_ids* and return the raw logit vector."""
        ...

    def get_stats(self) -> Dict[str, Any]: ...
    def clear_cache(self) -> None: ...
    def __repr__(self) -> str: ...


class StreamingGenerator(Iterator[str]):
    """Token-by-token streaming iterator returned by ``InferenceEngine.generate_stream``."""

    def __iter__(self) -> "StreamingGenerator": ...
    def __next__(self) -> str: ...
    def cancel(self) -> None: ...
    def is_active(self) -> bool: ...
    def get_stream_stats(self) -> Dict[str, Any]: ...
    def __repr__(self) -> str: ...


# ── module-level functions ───────────────────────────────────────────
def load_model(
    path: str,
    device: str = "cpu",
    **kwargs: Any,
) -> BitNetModel: ...


def list_available_models(path: str) -> List[str]: ...


def get_device_info() -> Dict[str, Any]: ...


def set_num_threads(num_threads: int) -> None: ...


def batch_generate(
    engine: InferenceEngine,
    prompts: List[str],
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    """Returns a list of ``{"text", "latency_ms", "prompt_index"}`` dicts."""
    ...


def get_model_info(path: str, device: str = "cpu") -> ModelInfo: ...


def is_cuda_available() -> bool: ...
def is_metal_available() -> bool: ...
def get_cuda_device_count() -> int: ...
