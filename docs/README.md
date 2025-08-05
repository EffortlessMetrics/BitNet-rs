# BitNet.rs Documentation

Welcome to the comprehensive documentation for BitNet.rs, a high-performance Rust implementation of 1-bit Large Language Models.

## Documentation Structure

- **[Getting Started](getting-started.md)** - Installation and basic usage
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Migration Guide](migration-guide.md)** - Migrating from Python/C++ implementations
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Performance Guide](performance-guide.md)** - Optimization and tuning
- **[Examples](../examples/)** - Practical usage examples
- **[Architecture](architecture.md)** - System design and internals

## Quick Links

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [API Documentation](api-reference.md)
- [Examples](../examples/)
- [Contributing](../CONTRIBUTING.md)

## Installation

### From Crates.io

```bash
cargo add bitnet-rs
```

### From Source

```bash
git clone https://github.com/bitnet-rs/bitnet-rs.git
cd bitnet-rs
cargo build --release
```

### Docker

```bash
docker pull bitnet/bitnet-rs:latest
docker run -p 3000:3000 bitnet/bitnet-rs:latest
```

## Basic Usage

```rust
use bitnet_rs::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Load model
    let model = BitNetModel::from_file("models/bitnet-1.58b.gguf").await?;
    
    // Create inference engine
    let mut engine = InferenceEngine::new(model, Device::Cpu)?;
    
    // Generate text
    let response = engine.generate("Hello, world!").await?;
    println!("Generated: {}", response);
    
    Ok(())
}
```

## Features

- **High Performance**: Optimized SIMD kernels for CPU and GPU
- **Multiple Formats**: Support for GGUF, SafeTensors, and HuggingFace models
- **Quantization**: I2S, TL1, and TL2 quantization algorithms
- **Streaming**: Real-time token generation
- **Cross-Platform**: Linux, macOS, Windows, WebAssembly
- **Production Ready**: Comprehensive monitoring and observability

## Community

- [GitHub Issues](https://github.com/bitnet-rs/bitnet-rs/issues)
- [Discussions](https://github.com/bitnet-rs/bitnet-rs/discussions)
- [Discord](https://discord.gg/bitnet-rs)

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.