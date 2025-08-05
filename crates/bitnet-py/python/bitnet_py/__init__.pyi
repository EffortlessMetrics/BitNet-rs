"""
Type stubs for bitnet_py - BitNet.cpp Python bindings

This file provides comprehensive type hints for all classes and functions
in the bitnet_py module, ensuring proper IDE support and type checking.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, AsyncIterator, Iterator
from typing_extensions import Self
import numpy as np
from numpy.typing import NDArray

__version__: str

class BitNetError(Exception):
    """Base exception class for BitNet errors."""
    message: str
    error_type: str
    
    def __init__(self, message: str, error_type: Optional[str] = None) -> None: ...

class ModelArgs:
    """Model configuration parameters."""
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: Optional[int]
    vocab_size: int
    ffn_dim: int
    norm_eps: float
    rope_theta: float
    use_kernel: bool
    
    def __init__(
        self,
        dim: int = 2560,
        n_layers: int = 30,
        n_heads: int = 20,
        n_kv_heads: Optional[int] = None,
        vocab_size: int = 128256,
        ffn_dim: int = 6912,
        norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        use_kernel: bool = False,
    ) -> None: ...
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Any]) -> Self: ...
    
    def to_dict(self) -> Dict[str, Any]: ...

class GenArgs:
    """Generation configuration parameters."""
    gen_length: int
    gen_bsz: int
    prompt_length: int
    use_sampling: bool
    temperature: float
    top_p: float
    top_k: Optional[int]
    repetition_penalty: float
    
    def __init__(
        self,
        gen_length: int = 32,
        gen_bsz: int = 1,
        prompt_length: int = 64,
        use_sampling: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
    ) -> None: ...

class InferenceConfig:
    """Inference configuration for the Rust engine."""
    max_length: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: Optional[int]
    repetition_penalty: float
    do_sample: bool
    seed: Optional[int]
    pad_token_id: Optional[int]
    eos_token_id: Optional[int]
    
    def __init__(
        self,
        max_length: int = 2048,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        seed: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> None: ...

class Message:
    """Message for chat formatting."""
    role: str
    content: str
    
    def __init__(self, role: str, content: str) -> None: ...

class Stats:
    """Statistics for inference performance."""
    total_tokens: int
    total_time: float
    tokens_per_second: float
    prefill_time: float
    decode_time: float
    memory_used: float
    
    def __init__(self) -> None: ...
    def show(self) -> str: ...

class Tokenizer:
    """Tokenizer for encoding and decoding text."""
    n_words: int
    bos_id: int
    eos_id: int
    eot_id: int
    pad_id: int
    special_tokens: Dict[str, int]
    
    def __init__(self, model_path: str) -> None: ...
    
    def encode(
        self,
        text: str,
        bos: bool = True,
        eos: bool = False,
        allowed_special: Optional[Any] = None,
        disallowed_special: Optional[Any] = None,
    ) -> List[int]: ...
    
    def decode(self, tokens: List[int]) -> str: ...

class ChatFormat:
    """Chat format wrapper for dialog-based interactions."""
    eot_id: int
    
    def __init__(self, tokenizer: Tokenizer) -> None: ...
    
    def decode(self, tokens: List[int]) -> str: ...
    def encode_header(self, message: Message) -> List[int]: ...
    def encode_message(
        self, message: Message, return_target: bool = False
    ) -> Union[List[int], Tuple[List[int], List[int]]]: ...
    def encode_dialog_prompt(
        self,
        dialog: List[Message],
        completion: bool = False,
        return_target: bool = False,
    ) -> Union[List[int], Tuple[List[int], List[int]]]: ...

class BitNetModel:
    """BitNet model wrapper."""
    config: ModelArgs
    device: str
    dtype: str
    model_path: str
    
    def __init__(
        self,
        model_args: ModelArgs,
        device: str = "cpu",
        dtype: str = "bfloat16",
    ) -> None: ...
    
    @classmethod
    def from_pretrained(
        cls,
        ckpt_dir: str,
        model_args: Optional[ModelArgs] = None,
        device: str = "cpu",
        dtype: str = "bfloat16",
    ) -> Self: ...
    
    @classmethod
    def from_gguf(cls, model_path: str, device: Optional[str] = None) -> Self: ...
    
    @classmethod
    def from_safetensors(cls, model_path: str, device: Optional[str] = None) -> Self: ...
    
    def forward(
        self,
        token_values: NDArray[np.int32],
        token_lengths: Optional[NDArray[np.int32]] = None,
        start_pos: Optional[NDArray[np.int32]] = None,
        cache: Optional[List[Any]] = None,
        kv_padding: Optional[int] = None,
    ) -> NDArray[np.float32]: ...
    
    def forward_with_attn_bias(
        self,
        token_values: NDArray[np.int32],
        attn_bias: Any,
        cache: List[Any],
    ) -> NDArray[np.float32]: ...
    
    def to(self, device: str) -> None: ...
    def eval(self) -> None: ...
    def train(self, mode: Optional[bool] = None) -> None: ...
    def num_parameters(self) -> int: ...

class InferenceEngine:
    """Inference engine for text generation."""
    gen_args: GenArgs
    device: str
    tokenizer: Tokenizer
    
    def __init__(
        self,
        prefill_model: BitNetModel,
        decode_model: BitNetModel,
        tokenizer: Tokenizer,
        gen_args: GenArgs,
        device: Optional[str] = None,
    ) -> None: ...
    
    @classmethod
    def build(
        cls,
        ckpt_dir: str,
        gen_args: GenArgs,
        device: str,
        tokenizer_path: Optional[str] = None,
        num_layers: int = 13,
        use_full_vocab: bool = False,
    ) -> Self: ...
    
    def generate_all(
        self,
        prompts: List[List[int]],
        use_cuda_graphs: bool = True,
        use_sampling: Optional[bool] = None,
    ) -> Tuple[Stats, List[List[int]]]: ...
    
    def generate(self, prompt: str) -> str: ...
    
    async def generate_stream(self, prompt: str) -> str: ...
    
    def compile_prefill(self) -> None: ...
    def compile_generate(self) -> None: ...

class SimpleInference:
    """Simple inference engine for basic use cases."""
    config: InferenceConfig
    
    def __init__(
        self,
        model: BitNetModel,
        tokenizer: Tokenizer,
        config: Optional[InferenceConfig] = None,
    ) -> None: ...
    
    def generate(self, prompt: str) -> str: ...
    async def generate_stream(self, prompt: str) -> str: ...

# Utility functions
def load_model(
    model_path: str,
    model_format: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> BitNetModel: ...

def create_tokenizer(
    tokenizer_path: str,
    chat_format: bool = False,
) -> Union[Tokenizer, ChatFormat]: ...

def benchmark_inference(
    model: BitNetModel,
    tokenizer: Tokenizer,
    prompts: List[str],
    gen_args: Optional[GenArgs] = None,
    num_runs: int = 1,
    warmup_runs: int = 1,
) -> Dict[str, Any]: ...

def compare_performance(
    rust_model: BitNetModel,
    python_model: Any,
    tokenizer: Tokenizer,
    prompts: List[str],
    **kwargs: Any,
) -> Dict[str, Any]: ...

def validate_outputs(
    rust_model: BitNetModel,
    python_model: Any,
    tokenizer: Tokenizer,
    prompts: List[str],
    tolerance: float = 1e-6,
) -> Dict[str, Any]: ...

def get_system_info() -> Dict[str, Any]: ...

def make_cache(
    model_args: ModelArgs,
    length: int,
    device: Optional[str] = None,
    n_layers: Optional[int] = None,
    dtype: Optional[str] = None,
) -> List[Any]: ...

def cache_prefix(cache: List[Any], length: int) -> List[Any]: ...

# Convenience functions
def quick_inference(model_path: str, prompt: str, **kwargs: Any) -> str: ...

def benchmark_model(model_path: str, prompts: List[str], **kwargs: Any) -> Dict[str, Any]: ...

# Aliases for backward compatibility
FastGen = InferenceEngine
Transformer = BitNetModel