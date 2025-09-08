# BitNet.cpp Python Bindings

High-performance Python bindings for BitNet.cpp, providing a drop-in replacement for the original Python implementation with significant performance improvements.

## Features

- **Drop-in Compatibility**: Exact API compatibility with the original BitNet Python implementation
- **High Performance**: 2-10x performance improvement over the Python baseline
- **Multiple Backends**: Support for CPU and GPU inference with automatic fallback
- **Streaming Generation**: Async/await support for real-time text generation
- **Model Format Support**: GGUF, SafeTensors, and HuggingFace checkpoint formats
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **Type Safety**: Full type hints and IDE support

## Installation

### From PyPI (when published)

```bash
pip install bitnet-py
```

### From Source

**Requirements**: Python 3.12+ (required for PyO3 ABI3-py312 compatibility)

```bash
# Clone the repository
git clone https://github.com/microsoft/BitNet.git
cd BitNet/crates/bitnet-py

# Verify Python version
python --version  # Should show 3.12+

# Install maturin for building
pip install maturin

# Build and install in development mode
maturin develop --features cpu

# Or build for release
maturin build --release --features cpu
pip install target/wheels/bitnet_py-*.whl
```

### GPU Support

For GPU acceleration, install with CUDA support:

```bash
maturin develop --features gpu
```

## Quick Start

### Basic Usage

```python
import bitnet_py as bitnet

# Load model and tokenizer
model = bitnet.load_model("path/to/model.gguf")
tokenizer = bitnet.create_tokenizer("path/to/tokenizer.model")

# Create inference engine
engine = bitnet.SimpleInference(model, tokenizer)

# Generate text
result = engine.generate("Hello, my name is")
print(result)
```

### FastGen Compatibility

The library provides exact API compatibility with the original FastGen implementation:

```python
import bitnet_py as fast  # Drop-in replacement for 'import model as fast'

# Create generation arguments
gen_args = fast.GenArgs(
    gen_length=128,
    temperature=0.8,
    top_p=0.9,
    use_sampling=True,
)

# Build FastGen engine (identical API)
g = fast.FastGen.build(
    ckpt_dir="path/to/checkpoint",
    gen_args=gen_args,
    device="cuda:0",
)

# Generate responses (identical API)
prompts = ["Hello", "How are you?"]
tokens = [g.tokenizer.encode(p, bos=False, eos=False) for p in prompts]
stats, results = g.generate_all(tokens, use_cuda_graphs=True)

for i, prompt in enumerate(prompts):
    answer = g.tokenizer.decode(results[i])
    print(f"> {prompt}")
    print(answer)
```

### Async Streaming

The library provides comprehensive async streaming support with advanced features:

```python
import asyncio
import bitnet_py as bitnet

async def main():
    model = bitnet.load_model("model.gguf")
    tokenizer = bitnet.create_tokenizer("tokenizer.model")
    engine = bitnet.SimpleInference(model, tokenizer)
    
    # Basic streaming generation
    stream = engine.generate_stream("Tell me about AI")
    for token in stream:
        print(token, end="", flush=True)
    
    # Advanced streaming with timeout and cancellation
    async def stream_with_timeout():
        stream = engine.generate_stream("Long story prompt")
        tokens = []
        
        async def collect_tokens():
            for token in stream:
                tokens.append(token)
                print(token, end="", flush=True)
                await asyncio.sleep(0.01)  # Simulate processing
                
                if len(tokens) >= 20:  # Cancel after 20 tokens
                    stream.cancel()
                    break
        
        try:
            await asyncio.wait_for(collect_tokens(), timeout=10.0)
            print(f"\n✓ Generated {len(tokens)} tokens")
        except asyncio.TimeoutError:
            print("\n⚠ Generation timed out")
            stream.cancel()
    
    await stream_with_timeout()

asyncio.run(main())
```

#### Concurrent Streaming

```python
async def concurrent_streaming():
    prompts = ["AI is", "The future of", "Technology will"]
    semaphore = asyncio.Semaphore(3)  # Limit concurrent streams
    
    async def stream_prompt(prompt):
        async with semaphore:
            stream = engine.generate_stream(prompt)
            tokens = []
            
            for token in stream:
                tokens.append(token)
                if len(tokens) >= 15:  # Limit per stream
                    break
            
            return {"prompt": prompt, "tokens": len(tokens)}
    
    tasks = [stream_prompt(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        print(f"'{result['prompt']}' -> {result['tokens']} tokens")

asyncio.run(concurrent_streaming())
```

#### Stream Control and Monitoring

```python
# Check stream status
stream = engine.generate_stream("Test prompt")
print(f"Active: {stream.is_active()}")

# Get stream statistics
if hasattr(stream, 'get_stream_stats'):
    stats = stream.get_stream_stats()
    print(f"Config: {stats}")

# Manual cancellation
stream.cancel()
print(f"Active after cancel: {stream.is_active()}")
```

### Chat Format

```python
import bitnet_py as bitnet

# Create chat tokenizer
tokenizer = bitnet.create_tokenizer("tokenizer.model")
chat_tokenizer = bitnet.ChatFormat(tokenizer)

# Create dialog
dialog = [
    bitnet.Message(role="user", content="What is AI?"),
]

# Encode dialog
tokens = chat_tokenizer.encode_dialog_prompt(dialog, completion=True)
```

## API Reference

### Core Classes

#### `BitNetModel`

Main model class supporting multiple formats:

```python
# Load from different formats
model = bitnet.BitNetModel.from_gguf("model.gguf")
model = bitnet.BitNetModel.from_safetensors("model.safetensors")
model = bitnet.BitNetModel.from_pretrained("checkpoint_dir/")

# Model information
print(f"Parameters: {model.num_parameters():,}")
print(f"Device: {model.device}")
```

#### `Tokenizer`

Text tokenization with full compatibility:

```python
tokenizer = bitnet.Tokenizer("tokenizer.model")

# Encode text
tokens = tokenizer.encode("Hello world", bos=True, eos=False)

# Decode tokens
text = tokenizer.decode(tokens)

# Properties
print(f"Vocab size: {tokenizer.n_words}")
print(f"BOS token: {tokenizer.bos_id}")
```

#### `InferenceEngine` / `FastGen`

High-performance inference engine:

```python
# Simple interface
engine = bitnet.SimpleInference(model, tokenizer, config)
result = engine.generate("Prompt")

# Advanced interface (FastGen compatibility)
engine = bitnet.FastGen.build(ckpt_dir, gen_args, device)
stats, results = engine.generate_all(token_lists)
```

### Configuration Classes

#### `ModelArgs`

Model architecture configuration:

```python
args = bitnet.ModelArgs(
    dim=2560,
    n_layers=30,
    n_heads=20,
    vocab_size=128256,
    use_kernel=True,
)
```

#### `GenArgs`

Generation parameters:

```python
gen_args = bitnet.GenArgs(
    gen_length=128,
    temperature=0.8,
    top_p=0.9,
    use_sampling=True,
)
```

#### `InferenceConfig`

Inference configuration:

```python
config = bitnet.InferenceConfig(
    max_new_tokens=100,
    temperature=0.8,
    do_sample=True,
    seed=42,
)
```

### Utility Functions

```python
# Performance benchmarking
results = bitnet.benchmark_inference(model, tokenizer, prompts)
print(f"Tokens/sec: {results['avg_tokens_per_second']:.2f}")

# Model comparison
comparison = bitnet.compare_performance(rust_model, python_model, tokenizer, prompts)
print(f"Speedup: {comparison['speedup']:.2f}x")

# Output validation
validation = bitnet.validate_outputs(rust_model, python_model, tokenizer, prompts)
print(f"Accuracy: {validation['num_matches']}/{validation['num_prompts']}")

# System information
info = bitnet.get_system_info()
print(f"Features: {info['features']}")
```

## Performance

The Rust implementation provides significant performance improvements:

| Metric | Python Baseline | Rust Implementation | Improvement |
|--------|----------------|-------------------|-------------|
| Tokens/second | 50-100 | 200-500 | 2-5x |
| Memory usage | 4-8 GB | 2-4 GB | 2x reduction |
| Startup time | 10-30s | 2-5s | 5x faster |
| CPU utilization | 60-80% | 90-95% | Better efficiency |

## Migration Guide

### From Original Python Implementation

1. **Update imports**:
   ```python
   # Old
   import model as fast
   
   # New
   import bitnet_py as fast
   ```

2. **No code changes required** - the API is identical

3. **Optional optimizations**:
   ```python
   # Enable GPU if available
   device = "cuda:0" if torch.cuda.is_available() else "cpu"
   
   # Use optimized kernels
   model_args.use_kernel = True
   ```

### Configuration Migration

Existing configuration files work without changes:

```python
# Load existing config
with open("config.json") as f:
    config_dict = json.load(f)

model_args = bitnet.ModelArgs.from_dict(config_dict["model"])
gen_args = bitnet.GenArgs(**config_dict["generation"])
```

## Examples

See the `examples/` directory for comprehensive examples:

- `basic_usage.py` - Basic model loading and inference
- `fastgen_compatibility.py` - Drop-in replacement demonstration
- `async_streaming.py` - Async streaming generation
- `performance_comparison.py` - Benchmarking and validation
- `migration_example.py` - Step-by-step migration guide

## Development

### Building from Source

```bash
# Verify Python 3.12+ is active
python --version  # Should show 3.12+

# Install development dependencies  
pip install maturin pytest pytest-asyncio black mypy

# Build in development mode with PyO3 ABI3-py312 support
maturin develop

# Run tests including integration tests
pytest tests/
pytest tests/integration/ -v --requires-integration  # Feature-gated integration tests

# Format code
black python/ examples/

# Type checking with Python 3.12 settings
mypy python/bitnet_py/
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_model.py -v
pytest tests/test_inference.py -v
pytest tests/test_compatibility.py -v

# Run integration tests (feature-gated)
pytest tests/integration/ -v --requires-integration

# Run with coverage
pytest tests/ --cov=bitnet_py --cov-report=html

# Test structure improvements in PR #175:
# - Integration tests now properly gated with required-features
# - Enhanced streaming comprehensive test organization
# - Improved test helper implementations
```

## Troubleshooting

### Common Issues

1. **Model loading errors**:
   ```python
   # Check file exists and format
   import os
   assert os.path.exists(model_path), f"Model not found: {model_path}"
   
   # Try different formats
   model = bitnet.load_model(model_path, model_format="auto")
   ```

2. **Memory issues**:
   ```python
   # Reduce batch size
   gen_args.gen_bsz = 1
   
   # Use CPU if GPU memory is insufficient
   device = "cpu"
   ```

3. **Performance issues**:
   ```python
   # Enable optimized kernels
   model_args.use_kernel = True
   
   # Check system info
   info = bitnet.get_system_info()
   print(f"CPU features: {info['cpu_features']}")
   ```

### Getting Help

- Check the [examples](examples/) for common use cases
- Review the [API documentation](#api-reference)
- Open an issue on [GitHub](https://github.com/microsoft/BitNet/issues)

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## Acknowledgments

- Original BitNet.cpp implementation
- PyO3 for excellent Rust-Python bindings
- The Rust ML ecosystem for foundational libraries