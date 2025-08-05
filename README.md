# BitNet.rs - High-Performance 1-bit LLM Inference

[![Crates.io](https://img.shields.io/crates/v/bitnet.svg)](https://crates.io/crates/bitnet)
[![Documentation](https://docs.rs/bitnet/badge.svg)](https://docs.rs/bitnet)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/microsoft/BitNet#license)
[![Build Status](https://github.com/microsoft/BitNet/workflows/CI/badge.svg)](https://github.com/microsoft/BitNet/actions)
[![MSRV](https://img.shields.io/badge/MSRV-1.70.0-blue.svg)](https://github.com/microsoft/BitNet)

A high-performance Rust implementation of BitNet 1-bit Large Language Model inference, providing drop-in compatibility with the original Python/C++ implementation while achieving superior performance and safety.

## Features

- **🚀 High Performance**: Optimized SIMD kernels for x86_64 (AVX2/AVX-512) and ARM64 (NEON)
- **🌐 Cross-Platform**: Support for Linux, macOS, and Windows
- **⚡ Multiple Backends**: CPU and GPU (CUDA) inference engines
- **📁 Format Support**: GGUF, SafeTensors, and HuggingFace model formats
- **🔢 Quantization**: I2_S, TL1 (ARM), and TL2 (x86) quantization algorithms
- **🔗 Language Bindings**: C API, Python bindings, and WebAssembly support
- **🏭 Production Ready**: Comprehensive testing, benchmarking, and monitoring

## Quick Start

### Installation

Add BitNet.rs to your `Cargo.toml`:

```toml
[dependencies]
bitnet = "0.1"
```

Or install the CLI tool:

```bash
cargo install bitnet-cli
```

### Basic Usage

```rust
use bitnet::prelude::*;

// Load a BitNet model
let device = Device::Cpu;
let model = BitNetModel::load("model.gguf", &device)?;

// Create inference engine
let mut engine = InferenceEngine::new(model)?;

// Generate text
let response = engine.generate("Hello, world!")?;
println!("{}", response);
```

### CLI Usage

```bash
# Run inference
bitnet inference --model model.gguf --prompt "Hello, world!"

# Convert model formats
bitnet convert --input model.safetensors --output model.gguf

# Benchmark performance
bitnet benchmark --model model.gguf
```

## Feature Flags

BitNet.rs uses feature flags to enable optional functionality:

- `cpu` (default): CPU inference with optimized kernels
- `gpu`: GPU acceleration via CUDA
- `avx2`: x86_64 AVX2 SIMD optimizations
- `avx512`: x86_64 AVX-512 SIMD optimizations
- `neon`: ARM64 NEON SIMD optimizations
- `full`: Enable all features

See [FEATURES.md](FEATURES.md) for detailed feature documentation.

## Architecture

The project is organized as a Rust workspace:

| Crate | Description |
|-------|-------------|
| `bitnet` | Main library crate |
| `bitnet-common` | Shared types and utilities |
| `bitnet-models` | Model loading and definitions |
| `bitnet-quantization` | Quantization algorithms |
| `bitnet-kernels` | High-performance compute kernels |
| `bitnet-inference` | Inference engines |
| `bitnet-tokenizers` | Tokenization support |
| `bitnet-cli` | Command-line interface |
| `bitnet-server` | HTTP server |
| `bitnet-ffi` | C API bindings |
| `bitnet-py` | Python bindings |
| `bitnet-wasm` | WebAssembly bindings |

## Performance

BitNet.rs achieves significant performance improvements over the original Python implementation:

- **2-5x faster inference** through zero-cost abstractions and SIMD optimization
- **Reduced memory footprint** via zero-copy operations and efficient memory management
- **Better scalability** with async/await support and batch processing

## Language Bindings

### Python

```python
import bitnet

# Load model and generate text
model = bitnet.BitNetModel("model.gguf")
response = model.generate("Hello, world!")
print(response)
```

### C API

```c
#include "bitnet.h"

// Load model and run inference
BitNetModel* model = bitnet_model_load("model.gguf");
char* response = bitnet_generate(model, "Hello, world!");
printf("%s\n", response);
bitnet_free_string(response);
bitnet_model_free(model);
```

### WebAssembly

```javascript
import init, { BitNetModel } from './pkg/bitnet_wasm.js';

await init();
const model = new BitNetModel('model.gguf');
const response = await model.generate('Hello, world!');
console.log(response);
```

## Development

### Prerequisites

- Rust 1.70.0 or later
- Python 3.8+ (for Python bindings)
- CUDA Toolkit 11.0+ (for GPU support)

### Building

```bash
# Build with default features (CPU only)
cargo build

# Build with GPU support
cargo build --features gpu

# Build with all optimizations
cargo build --features full

# Run tests
cargo test --workspace

# Run benchmarks
cargo bench --workspace
```

### Code Quality

```bash
# Format code
cargo fmt --all

# Run clippy with pedantic lints
cargo clippy --all-targets --all-features -- -D warnings

# Security audit
cargo audit

# License compliance
cargo deny check
```

## Cross-Validation

The implementation includes comprehensive cross-validation against the original Python/C++ codebase to ensure numerical accuracy and functional parity within 1e-6 tolerance.

## Documentation

- [API Documentation](https://docs.rs/bitnet)
- [Feature Flags](FEATURES.md)
- [Migration Guide](crates/bitnet-py/MIGRATION_GUIDE.md)
- [Performance Guide](docs/performance-guide.md)
- [Troubleshooting](docs/troubleshooting.md)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT OR Apache-2.0 license. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Original BitNet.cpp implementation
- [Candle](https://github.com/huggingface/candle) tensor library
- Rust ML ecosystem contributors

---

**Minimum Supported Rust Version (MSRV)**: 1.70.0