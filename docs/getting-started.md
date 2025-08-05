# Getting Started with BitNet Rust

This guide will help you get up and running with BitNet Rust, a high-performance implementation of BitNet models in Rust.

## Installation

### Prerequisites

- Rust 1.75 or later
- CUDA 11.8+ (optional, for GPU acceleration)
- Python 3.8+ (optional, for Python bindings)

### Install from crates.io

```bash
cargo install bitnet-cli
```

### Build from source

```bash
git clone https://github.com/your-org/bitnet-rust.git
cd bitnet-rust
cargo build --release
```

### Feature flags

BitNet Rust supports several feature flags for customization:

- `gpu`: Enable CUDA GPU acceleration (default: enabled)
- `python`: Enable Python bindings (default: disabled)
- `wasm`: Enable WebAssembly support (default: disabled)
- `cli`: Enable CLI tool (default: enabled)

```bash
# Install with specific features
cargo install bitnet-cli --features "gpu,python"

# Build without GPU support
cargo build --release --no-default-features --features "cli"
```

## Quick Start

### Using the CLI

1. **Download a model**:
```bash
bitnet-cli model download microsoft/bitnet-b1_58-large
```

2. **Run inference**:
```bash
bitnet-cli inference --model microsoft/bitnet-b1_58-large --prompt "Hello, world!"
```

3. **Stream generation**:
```bash
bitnet-cli inference --model microsoft/bitnet-b1_58-large --prompt "Tell me a story" --stream
```

### Using the Rust API

Add BitNet to your `Cargo.toml`:

```toml
[dependencies]
bitnet = "0.1.0"
```

Basic usage:

```rust
use bitnet::{BitNetModel, InferenceConfig, GenerationConfig};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Load a model
    let model = BitNetModel::from_pretrained("microsoft/bitnet-b1_58-large").await?;
    
    // Configure generation
    let config = GenerationConfig {
        max_new_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
        ..Default::default()
    };
    
    // Generate text
    let output = model.generate("Hello, world!", &config).await?;
    println!("Generated: {}", output);
    
    Ok(())
}
```

### Streaming Generation

```rust
use bitnet::{BitNetModel, GenerationConfig};
use futures_util::StreamExt;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let model = BitNetModel::from_pretrained("microsoft/bitnet-b1_58-large").await?;
    let config = GenerationConfig::default();
    
    let mut stream = model.generate_stream("Tell me a story", &config);
    
    while let Some(token) = stream.next().await {
        match token {
            Ok(text) => print!("{}", text),
            Err(e) => eprintln!("Error: {}", e),
        }
    }
    
    Ok(())
}
```

## Model Formats

BitNet Rust supports multiple model formats:

### GGUF Format
```bash
# Load GGUF model
bitnet-cli inference --model path/to/model.gguf --prompt "Hello"
```

### SafeTensors Format
```bash
# Load SafeTensors model
bitnet-cli inference --model path/to/model.safetensors --prompt "Hello"
```

### HuggingFace Hub
```bash
# Load from HuggingFace Hub
bitnet-cli inference --model microsoft/bitnet-b1_58-large --prompt "Hello"
```

## Configuration

### Configuration File

Create a `bitnet.toml` configuration file:

```toml
[model]
default_model = "microsoft/bitnet-b1_58-large"
cache_dir = "~/.cache/bitnet"

[inference]
device = "auto"  # "cpu", "cuda", or "auto"
max_batch_size = 8
kv_cache_size = 2048

[generation]
max_new_tokens = 512
temperature = 0.7
top_p = 0.9
top_k = 50
repetition_penalty = 1.0
```

### Environment Variables

BitNet Rust respects these environment variables:

- `BITNET_MODEL_CACHE`: Model cache directory
- `BITNET_DEVICE`: Default device ("cpu", "cuda", "auto")
- `BITNET_LOG_LEVEL`: Log level ("trace", "debug", "info", "warn", "error")
- `CUDA_VISIBLE_DEVICES`: GPU device selection

## Performance Optimization

### CPU Optimization

1. **Enable CPU features**:
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

2. **Tune thread count**:
```bash
export RAYON_NUM_THREADS=8
bitnet-cli inference --model model.gguf --prompt "Hello"
```

### GPU Optimization

1. **Enable mixed precision**:
```rust
let config = InferenceConfig {
    use_mixed_precision: true,
    ..Default::default()
};
```

2. **Optimize batch size**:
```rust
let config = InferenceConfig {
    max_batch_size: 16,  // Adjust based on GPU memory
    ..Default::default()
};
```

## Troubleshooting

### Common Issues

1. **CUDA not found**:
   - Install CUDA 11.8 or later
   - Set `CUDA_HOME` environment variable
   - Build with `--no-default-features --features cli` for CPU-only

2. **Model loading fails**:
   - Check model format compatibility
   - Verify model file integrity
   - Ensure sufficient disk space and memory

3. **Poor performance**:
   - Enable native CPU features with `RUSTFLAGS="-C target-cpu=native"`
   - Use GPU acceleration if available
   - Adjust batch size and thread count

### Debug Mode

Enable debug logging:

```bash
RUST_LOG=debug bitnet-cli inference --model model.gguf --prompt "Hello"
```

### Memory Issues

Monitor memory usage:

```bash
bitnet-cli benchmark --model model.gguf --monitor-memory
```

## Next Steps

- Read the [API Reference](api-reference.md) for detailed API documentation
- Check out [Examples](examples/) for more usage patterns
- See [Migration Guide](migration-guide.md) for migrating from Python/C++
- Review [Performance Tuning](performance-tuning.md) for optimization tips

## Getting Help

- [GitHub Issues](https://github.com/your-org/bitnet-rust/issues)
- [Documentation](https://docs.rs/bitnet)
- [Discord Community](https://discord.gg/bitnet-rust)