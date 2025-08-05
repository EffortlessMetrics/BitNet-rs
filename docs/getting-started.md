# Getting Started with BitNet.rs

This guide will help you get up and running with BitNet.rs quickly.

## Table of Contents

- [Installation](#installation)
- [System Requirements](#system-requirements)
- [First Steps](#first-steps)
- [Basic Usage](#basic-usage)
- [Configuration](#configuration)
- [Next Steps](#next-steps)

## Installation

### Prerequisites

- Rust 1.75 or later
- Git (for source installation)
- CUDA toolkit (optional, for GPU acceleration)

### Option 1: From Crates.io (Recommended)

Add BitNet.rs to your `Cargo.toml`:

```toml
[dependencies]
bitnet-rs = "0.1.0"

# Optional features
bitnet-rs = { version = "0.1.0", features = ["gpu", "server"] }
```

Available features:
- `cpu` (default): CPU-optimized kernels
- `gpu`: CUDA GPU acceleration
- `server`: HTTP server functionality
- `python`: Python bindings
- `wasm`: WebAssembly support
- `full`: All features enabled

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/bitnet-rs/bitnet-rs.git
cd bitnet-rs

# Build with default features
cargo build --release

# Build with specific features
cargo build --release --features="gpu,server"

# Install the CLI tool
cargo install --path crates/bitnet-cli
```

### Option 3: Using Docker

```bash
# Pull the latest image
docker pull bitnet/bitnet-rs:latest

# Run with default configuration
docker run -p 3000:3000 bitnet/bitnet-rs:latest

# Run with custom model
docker run -v /path/to/models:/models -p 3000:3000 \
  -e BITNET_MODEL_PATH=/models/your-model.gguf \
  bitnet/bitnet-rs:latest
```

## System Requirements

### Minimum Requirements

- **CPU**: x86_64 or ARM64 processor
- **RAM**: 4GB (for 1.58B parameter models)
- **Storage**: 2GB free space
- **OS**: Linux, macOS, or Windows

### Recommended Requirements

- **CPU**: Modern x86_64 with AVX2 or ARM64 with NEON
- **RAM**: 8GB or more
- **GPU**: NVIDIA GPU with CUDA 11.0+ (optional)
- **Storage**: SSD with 10GB+ free space

### GPU Requirements (Optional)

- NVIDIA GPU with Compute Capability 6.0+
- CUDA 11.0 or later
- 4GB+ VRAM for 1.58B models

## First Steps

### 1. Verify Installation

```bash
# Check if CLI is installed
bitnet --version

# Run basic health check
bitnet health
```

### 2. Download a Model

```bash
# Download a pre-quantized model
bitnet download bitnet-1.58b-i2s

# Or specify a custom path
bitnet download bitnet-1.58b-i2s --output ./models/
```

### 3. Test Basic Inference

```bash
# Simple text generation
bitnet generate "Hello, world!" --model ./models/bitnet-1.58b-i2s.gguf

# With custom parameters
bitnet generate "The future of AI is" \
  --model ./models/bitnet-1.58b-i2s.gguf \
  --max-tokens 50 \
  --temperature 0.7
```

## Basic Usage

### Command Line Interface

The CLI provides the easiest way to get started:

```bash
# Generate text
bitnet generate "Your prompt here" --model path/to/model.gguf

# Start HTTP server
bitnet serve --model path/to/model.gguf --port 3000

# Convert model formats
bitnet convert input.safetensors output.gguf --quantization i2s

# Benchmark performance
bitnet benchmark --model path/to/model.gguf
```

### Rust Library

```rust
use bitnet_rs::prelude::*;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Load model from file
    let model = BitNetModel::from_file("models/bitnet-1.58b.gguf").await?;
    println!("Model loaded: {} parameters", model.parameter_count());

    // Create inference engine
    let device = Device::best_available()?; // Auto-select best device
    let mut engine = InferenceEngine::new(model, device)?;

    // Simple generation
    let response = engine.generate("Hello, world!").await?;
    println!("Response: {}", response);

    // Generation with custom config
    let config = GenerationConfig {
        max_new_tokens: 100,
        temperature: 0.8,
        top_p: 0.9,
        top_k: 50,
        ..Default::default()
    };

    let response = engine.generate_with_config("The future of AI is", &config).await?;
    println!("Custom response: {}", response);

    Ok(())
}
```

### HTTP Server

Start a server for API access:

```rust
use bitnet_rs::server::BitNetServer;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let server = BitNetServer::builder()
        .model_path("models/bitnet-1.58b.gguf")
        .bind("0.0.0.0:3000")
        .build()
        .await?;

    println!("Server running on http://0.0.0.0:3000");
    server.run().await?;

    Ok(())
}
```

Then make HTTP requests:

```bash
curl -X POST http://localhost:3000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 50}'
```

## Configuration

### Configuration File

Create a `bitnet.toml` configuration file:

```toml
[model]
path = "models/bitnet-1.58b.gguf"
device = "auto"  # "cpu", "cuda:0", or "auto"

[inference]
max_tokens = 100
temperature = 0.7
top_p = 0.9
top_k = 50

[server]
host = "0.0.0.0"
port = 3000
workers = 4

[logging]
level = "info"
format = "json"  # "json" or "pretty"
```

### Environment Variables

```bash
export BITNET_MODEL_PATH="models/bitnet-1.58b.gguf"
export BITNET_DEVICE="cuda:0"
export BITNET_LOG_LEVEL="debug"
export RUST_LOG="bitnet=debug"
```

### Programmatic Configuration

```rust
use bitnet_rs::config::*;

let config = BitNetConfig::builder()
    .model_path("models/bitnet-1.58b.gguf")
    .device(Device::Cuda(0))
    .inference_config(
        InferenceConfig::builder()
            .max_tokens(100)
            .temperature(0.7)
            .build()
    )
    .build();
```

## Next Steps

Now that you have BitNet.rs running, explore these topics:

1. **[API Reference](api-reference.md)** - Complete API documentation
2. **[Examples](../examples/)** - Practical usage examples
3. **[Performance Guide](performance-guide.md)** - Optimization tips
4. **[Migration Guide](migration-guide.md)** - Migrating from other implementations

### Common Use Cases

- **[Text Generation](../examples/basic/cpu_inference.rs)** - Basic text generation
- **[Streaming](../examples/basic/streaming.rs)** - Real-time token streaming
- **[Web Server](../examples/integrations/axum_server.rs)** - HTTP API server
- **[Batch Processing](../examples/batch_processing.rs)** - Process multiple requests
- **[Model Conversion](../examples/model_conversion.rs)** - Convert between formats

### Advanced Topics

- **[Custom Kernels](advanced/custom-kernels.md)** - Implementing custom compute kernels
- **[Quantization](advanced/quantization.md)** - Understanding quantization algorithms
- **[Deployment](../examples/deployment/)** - Production deployment guides
- **[Monitoring](../examples/integrations/tracing_observability.rs)** - Observability setup

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Search [GitHub Issues](https://github.com/bitnet-rs/bitnet-rs/issues)
3. Ask in [GitHub Discussions](https://github.com/bitnet-rs/bitnet-rs/discussions)
4. Join our [Discord Community](https://discord.gg/bitnet-rs)

## Contributing

We welcome contributions! See our [Contributing Guide](../CONTRIBUTING.md) for details on:

- Setting up the development environment
- Running tests
- Submitting pull requests
- Code style guidelines