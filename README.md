# BitNet.rs - Production-Ready 1-bit LLM Inference

[![Crates.io](https://img.shields.io/crates/v/bitnet.svg)](https://crates.io/crates/bitnet)
[![Documentation](https://docs.rs/bitnet/badge.svg)](https://docs.rs/bitnet)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/microsoft/BitNet#license)
[![Build Status](https://github.com/microsoft/BitNet/workflows/CI/badge.svg)](https://github.com/microsoft/BitNet/actions)
[![MSRV](https://img.shields.io/badge/MSRV-1.90.0-blue.svg)](https://github.com/microsoft/BitNet)

**High-performance Rust implementation of BitNet 1-bit Large Language Model inference** with memory safety, device-aware quantization, and cross-platform support.

> ✅ **Validated Drop-in Replacement**: Compatible with llama.cpp, handles models that crash C++ diagnostic tools.

## Why BitNet.rs?

- 🚀 **High Performance**: SIMD kernels (AVX2/AVX-512/NEON), CUDA acceleration, zero-copy operations
- 🛡️ **Memory Safe**: No segfaults or leaks, comprehensive error handling
- 🌐 **Cross-Platform**: Linux/macOS/Windows, CPU/GPU backends, language bindings
- 🔧 **Developer Friendly**: Modern Rust tooling, extensive documentation

## Quick Start

### Installation

```bash
# Rust library
cargo add bitnet

# CLI tools
cargo install bitnet-cli bitnet-server

# Python bindings
pip install bitnet-rs
```

### Basic Usage

```bash
# Build and test (CPU)
cargo build --no-default-features --features cpu
cargo test --workspace --no-default-features --features cpu

# GPU support
cargo build --no-default-features --features gpu

# Download and run inference
cargo run -p xtask -- download-model
cargo run -p xtask -- infer --model path/to/model.gguf --prompt "Hello"
```

### Rust API

```rust
use bitnet::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Load GGUF model
    let model = BitNetModel::from_file("model.gguf").await?;

    // Create inference engine with device auto-detection
    let engine = InferenceEngine::builder()
        .model(model)
        .backend(Backend::Auto)  // GPU if available, CPU fallback
        .build()?;

    // Generate text
    let response = engine.generate("Explain quantum computing").await?;
    println!("Generated: {}", response.text);

    Ok(())
}
```

## Core Features

### Quantization Support

- **I2_S**: Production 2-bit signed quantization (99%+ accuracy vs FP32)
- **TL1/TL2**: Table lookup quantization with device-aware selection
- **IQ2_S**: GGML-compatible via FFI bridge

### Device-Aware Computing

- Automatic GPU detection and fallback to optimized CPU kernels
- Mixed precision support (FP16/BF16) with Tensor Core acceleration
- Cross-validation against Microsoft BitNet C++ reference

### Universal Tokenizer

- Auto-detection from GGUF metadata
- BPE, SentencePiece, and custom format support
- O(1) byte lookup performance

## Architecture

BitNet.rs is organized as a Rust workspace:

- **`bitnet`**: Main library with unified API
- **`bitnet-inference`**: Autoregressive generation engine
- **`bitnet-quantization`**: 1-bit quantization algorithms
- **`bitnet-kernels`**: SIMD/CUDA compute kernels
- **`bitnet-models`**: GGUF/SafeTensors model loading
- **`bitnet-tokenizers`**: Universal tokenizer system

## Documentation

- 📚 **[Getting Started](docs/getting-started.md)** - Comprehensive setup guide
- 🚀 **[Quick Start](docs/quickstart.md)** - 5-minute setup
- 🏗️ **[Architecture](docs/architecture-overview.md)** - System design
- 🔧 **[Development](docs/development/)** - Build, test, and contribution guides
- 📖 **[API Reference](https://docs.rs/bitnet)** - Complete Rust documentation

### Key Guides

- [GPU Setup](docs/GPU_SETUP.md) - CUDA configuration
- [Performance](docs/performance-benchmarking.md) - Optimization and benchmarking
- [Migration](docs/migration-guide.md) - From C++/Python implementations
- [Troubleshooting](docs/troubleshooting/troubleshooting.md) - Common issues

## Language Bindings

### Python

```python
import bitnet

model = bitnet.BitNetModel("model.gguf")
response = model.generate("Hello, world!")
print(response)
```

### C API

```c
#include "bitnet.h"

BitNetModel* model = bitnet_model_load("model.gguf");
char* response = bitnet_generate(model, "Hello, world!");
printf("%s\n", response);
```

### WebAssembly

```javascript
import init, { BitNetModel } from './pkg/bitnet_wasm.js';

await init();
const model = new BitNetModel('model.gguf');
const response = await model.generate('Hello, world!');
```

## Production Status

✅ **Production Ready** - 100% validation pass rate across all acceptance gates:

- Build, unit tests, tensor mapping, tokenization
- Performance benchmarks, FFI compatibility
- Deterministic outputs, cross-validation

See [VALIDATION.md](VALIDATION.md) for detailed specifications.

## Development

```bash
# Development cycle
cargo test --workspace --no-default-features --features cpu
cargo fmt --all && cargo clippy --all-targets --all-features -- -D warnings

# Cross-validation (when changing inference)
export BITNET_GGUF="path/to/model.gguf"
cargo run -p xtask -- crossval
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

This project is licensed under the MIT OR Apache-2.0 license.

## Acknowledgments

- **Microsoft Research** for the original BitNet architecture
- **Rust ML ecosystem** contributors for excellent tooling
- **Community contributors** who make BitNet.rs better every day

---

**MSRV**: 1.90.0 | **Status**: Production Ready | **Performance**: Benchmarking framework complete
