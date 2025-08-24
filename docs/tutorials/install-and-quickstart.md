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