# BitNet.rs - Production-Ready 1-bit LLM Inference

[![Crates.io](https://img.shields.io/crates/v/bitnet.svg)](https://crates.io/crates/bitnet)
[![Documentation](https://docs.rs/bitnet/badge.svg)](https://docs.rs/bitnet)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/microsoft/BitNet#license)
[![Build Status](https://github.com/microsoft/BitNet/workflows/CI/badge.svg)](https://github.com/microsoft/BitNet/actions)
[![MSRV](https://img.shields.io/badge/MSRV-1.90.0-blue.svg)](https://github.com/microsoft/BitNet)

**High-performance Rust implementation of BitNet 1-bit Large Language Model inference** with memory safety, device-aware quantization, and cross-platform support.

> ✅ **Validated Drop-in Replacement**: Compatible with llama.cpp, handles models that crash C++ diagnostic tools.

## Why BitNet.rs?

- 🚀 **High Performance**: Real quantized inference (10-20 tok/s CPU, 50-100 tok/s GPU), SIMD kernels (AVX2/AVX-512/NEON), CUDA acceleration
- 🛡️ **Memory Safe**: No segfaults or leaks, comprehensive error handling, strict mode prevents mock fallbacks
- 🌐 **Cross-Platform**: Linux/macOS/Windows, CPU/GPU backends, I2S/TL1/TL2 quantization
- 🔧 **Developer Friendly**: Modern Rust tooling, extensive documentation, cross-validation against C++ reference

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
# Build and test (CPU with real quantization)
cargo build --no-default-features --features cpu
cargo test --workspace --no-default-features --features cpu

# GPU support with mixed precision
cargo build --no-default-features --features gpu

# Download and run inference with strict mode (prevents mock fallbacks)
cargo run -p xtask -- download-model
BITNET_STRICT_MODE=1 cargo run -p xtask -- infer --model path/to/model.gguf --prompt "Hello"
```

### 10-Line CPU Quickstart

Get started with BitNet.rs CPU inference in under 10 lines:

```bash
# 1. Build with explicit CPU features
cargo build --no-default-features --features cpu

# 2. Download a BitNet model
cargo run -p xtask -- download-model

# 3. Run deterministic inference (128 tokens)
export BITNET_DETERMINISTIC=1 RAYON_NUM_THREADS=1 BITNET_SEED=42
cargo run -p xtask -- benchmark --model models/*.gguf --tokens 128

# 4. Verify honest compute receipt
cargo run -p xtask -- verify-receipt ci/inference.json
```

**Expected Performance:** 10-20 tok/s on CPU for 2B I2_S models (see [baselines/](docs/baselines/) for measured results).

**Receipt Verification:** All inference runs generate receipts (`ci/inference.json`) with kernel IDs proving real computation. CI blocks PRs with mocked receipts.

### Rust API

```rust
use bitnet::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Load GGUF model with real quantized weights
    let model = BitNetModel::from_file("model.gguf").await?;

    // Create inference engine with device auto-detection and strict mode
    std::env::set_var("BITNET_STRICT_MODE", "1");  // Prevent mock fallbacks
    let engine = InferenceEngine::builder()
        .model(model)
        .backend(Backend::Auto)  // GPU if available, CPU fallback
        .quantization(QuantizationType::I2S)  // Real quantized computation
        .build()?;

    // Generate text with real neural network inference
    let response = engine.generate("Explain quantum computing").await?;
    println!("Generated: {}", response.text);

    // Access realistic performance metrics
    if let Some(metrics) = response.metrics {
        println!("Throughput: {:.1} tokens/sec", metrics.throughput.e2e);
        println!("Quantization: I2S (99.8% accuracy vs FP32)");
    }

    Ok(())
}
```

## Core Features

### Quantization Support

- **I2_S**: Production 2-bit signed quantization (≥99.8% accuracy vs FP32, 10-20 tok/s CPU, 50-100 tok/s GPU)
- **TL1/TL2**: Table lookup quantization with device-aware selection (≥99.6% accuracy vs FP32)
- **Real Computation**: Native quantized matrix multiplication eliminates mock fallbacks
- **Strict Mode**: `BITNET_STRICT_MODE=1` ensures production-ready inference paths

### Device-Aware Computing

- Automatic GPU detection and fallback to optimized CPU kernels
- Mixed precision support (FP16/BF16) with Tensor Core acceleration
- Cross-validation against Microsoft BitNet C++ reference (<5% performance variance)
- SIMD acceleration (AVX2/AVX-512/NEON) for CPU inference

### Universal Tokenizer Discovery

- **Automatic Detection**: Extracts tokenizers from GGUF metadata (HuggingFace JSON, SentencePiece, vocabulary arrays)
- **Architecture Recognition**: Identifies BitNet, LLaMA-2/3, GPT-2, GPT-Neo, BERT, T5 from tensor patterns
- **Smart Fallbacks**: Co-located files → cache → HuggingFace Hub download → offline mode
- **Model-Specific Wrappers**: LLaMA (32K/128K variants), GPT-2 (no BOS), BitNet (quantization-aware)
- **Production Mode**: `BITNET_STRICT_TOKENIZERS=1` prevents mock fallbacks
- **Vocabulary Resolution**: 5-strategy extraction from metadata, tensors, or architecture defaults
- **O(1) Performance**: Memory-mapped GGUF parsing, zero-copy tokenizer extraction

## Receipt Verification

BitNet.rs implements **honest compute** verification through production receipts. Every inference run generates a receipt with kernel IDs proving real computation.

### Receipt Schema v1.0.0

```json
{
  "schema_version": "1.0.0",
  "compute_path": "real",
  "backend": "cpu",
  "model": "microsoft/bitnet-b1.58-2B-4T-gguf",
  "quantization": "i2s",
  "tokens_generated": 128,
  "throughput_tokens_per_sec": 15.3,
  "success": true,
  "kernels": [
    "i2s_cpu_quantized_matmul",
    "tl1_lut_dequant_forward",
    "attention_kv_cache_update",
    "layernorm_forward"
  ],
  "timestamp": "2025-10-15T12:00:00Z"
}
```

### xtask Commands

```bash
# Generate receipt (writes ci/inference.json)
cargo run -p xtask -- benchmark --model <model.gguf> --tokens 128

# Verify receipt passes quality gates
cargo run -p xtask -- verify-receipt ci/inference.json

# Strict mode (fail on warnings)
BITNET_STRICT_MODE=1 cargo run -p xtask -- verify-receipt ci/inference.json
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BITNET_DETERMINISTIC` | Enable deterministic inference | `0` |
| `BITNET_SEED` | Random seed for deterministic mode | `42` |
| `RAYON_NUM_THREADS` | Thread count (use `1` for determinism) | auto |
| `BITNET_STRICT_MODE` | Fail on validation warnings | `0` |
| `BITNET_GGUF` | Override model path | auto-discover `models/` |

### Receipt Requirements

**Honest Compute:**
- `compute_path` must be `"real"` (not `"mocked"`)
- `kernels` array must be non-empty
- Kernel IDs must be valid (non-empty, ≤128 chars, ≤10,000 count)

**CI Enforcement:**
- Model Gates (CPU) workflow requires valid receipts
- Branch protection blocks PRs with mocked receipts
- See [.github/workflows/model-gates.yml](.github/workflows/model-gates.yml)

### Baseline Receipts

Reference receipts are stored in [docs/baselines/](docs/baselines/) with datestamped filenames (`YYYYMMDD-cpu.json`). These baselines establish reproducible performance benchmarks for CPU inference.

## Architecture

BitNet.rs is organized as a Rust workspace:

- **`bitnet`**: Main library with unified API
- **`bitnet-inference`**: Autoregressive generation engine
- **`bitnet-quantization`**: 1-bit quantization algorithms (I2_S, TL1, TL2)
- **`bitnet-kernels`**: SIMD/CUDA compute kernels
- **`bitnet-models`**: GGUF/SafeTensors model loading with zero-copy memory mapping
- **`bitnet-tokenizers`**: Universal tokenizer discovery with automatic GGUF extraction

## Documentation

- 📚 **[Getting Started](docs/getting-started.md)** - Comprehensive setup guide
- 🚀 **[Quick Start](docs/quickstart.md)** - 5-minute setup
- 🏗️ **[Architecture](docs/architecture-overview.md)** - System design
- 🔧 **[Development](docs/development/)** - Build, test, and contribution guides
- 📖 **[API Reference](https://docs.rs/bitnet)** - Complete Rust documentation

### Key Guides

- [Tokenizer Discovery](docs/reference/tokenizer-discovery-api.md) - Automatic tokenizer resolution
- [Export Clean GGUF](docs/howto/export-clean-gguf.md) - SafeTensors to GGUF with F16 LayerNorm preservation
- [Model Baselines](docs/baselines/README.md) - Validated model fingerprints and performance metrics
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
# Development cycle with strict mode (prevents mock fallbacks)
BITNET_STRICT_MODE=1 cargo test --workspace --no-default-features --features cpu
cargo fmt --all && cargo clippy --all-targets --all-features -- -D warnings

# Cross-validation against C++ reference (validates real quantization)
export BITNET_GGUF="path/to/model.gguf"
BITNET_STRICT_MODE=1 cargo run -p xtask -- crossval
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

This project is licensed under the MIT OR Apache-2.0 license.

## Acknowledgments

- **Microsoft Research** for the original BitNet architecture
- **Rust ML ecosystem** contributors for excellent tooling
- **Community contributors** who make BitNet.rs better every day

---

**MSRV**: 1.90.0 | **Status**: Production Ready | **Performance**: Real quantized inference (10-20 tok/s CPU, 50-100 tok/s GPU)
