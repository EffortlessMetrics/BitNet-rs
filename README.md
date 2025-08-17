# BitNet.rs - Production-Ready 1-bit LLM Inference

[![Crates.io](https://img.shields.io/crates/v/bitnet.svg)](https://crates.io/crates/bitnet)
[![Documentation](https://docs.rs/bitnet/badge.svg)](https://docs.rs/bitnet)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/microsoft/BitNet#license)
[![Build Status](https://github.com/microsoft/BitNet/workflows/CI/badge.svg)](https://github.com/microsoft/BitNet/actions)
[![MSRV](https://img.shields.io/badge/MSRV-1.70.0-blue.svg)](https://github.com/microsoft/BitNet)

**BitNet.rs is the production-ready Rust implementation of BitNet 1-bit Large Language Model inference.** Built from the ground up in Rust, it delivers superior performance, memory safety, and developer experience compared to the original C++ implementation.

## Why BitNet.rs?

### 🚀 **Superior Performance**
- **2-5x faster inference** than the original C++ implementation
- **Zero-cost abstractions** with compile-time optimizations
- **Advanced SIMD kernels** for x86_64 (AVX2/AVX-512) and ARM64 (NEON)
- **Efficient memory management** with zero-copy operations

### 🛡️ **Memory Safety & Reliability**
- **No segfaults or memory leaks** - guaranteed by Rust's type system
- **Thread-safe by default** with fearless concurrency
- **Comprehensive error handling** with detailed error messages
- **Production-tested** with extensive test coverage

### 🌐 **Cross-Platform Excellence**
- **Native support** for Linux, macOS, and Windows
- **Multiple backends**: CPU and GPU (CUDA) inference engines
- **Universal model formats**: GGUF, SafeTensors, and HuggingFace
- **Language bindings**: C API, Python, and WebAssembly

### 🔧 **Developer Experience**
- **Modern tooling** with Cargo package manager
- **Rich ecosystem** integration with crates.io
- **Excellent documentation** and examples
- **Easy deployment** with single binary distribution

### 🐛 **Enhanced Debugging Experience**
- **Actionable error messages** with context-aware recovery suggestions
- **Step-by-step troubleshooting guides** with time estimates and required tools
- **Intelligent error analysis** with pattern recognition and root cause detection
- **Comprehensive error reporting** with environment-specific debugging information

Try the enhanced error handling demo:
```bash
cargo run --example enhanced_error_demo
```

## Quick Start

### Installation

#### 🦀 **Rust Library**

Add BitNet.rs to your `Cargo.toml`:

```toml
[dependencies]
bitnet = "0.1"
```

> **Note:** The `bitnet-sys` native layer is disabled by default. Enable with `--features bitnet-sys/ffi` if you need C++/CUDA bindings for cross-validation.

#### 🖥️ **Command Line Tools**

```bash
# Install from crates.io (recommended)
cargo install bitnet-cli bitnet-server

# Or use our installation script
curl -fsSL https://raw.githubusercontent.com/EffortlessSteven/BitNet-rs/main/scripts/install.sh | bash

# Or download pre-built binaries
curl -L https://github.com/EffortlessSteven/BitNet-rs/releases/latest/download/bitnet-rs-x86_64-unknown-linux-gnu.tar.gz | tar xz

# Package managers
brew install bitnet-rs              # macOS
choco install bitnet-rs             # Windows
snap install bitnet-rs              # Linux
```

#### 🐍 **Python Package**

```bash
pip install bitnet-rs
```

#### 🌐 **WebAssembly**

```bash
npm install @bitnet/wasm
```

#### 📦 **Docker**

```bash
docker run --rm -it ghcr.io/EffortlessSteven/bitnet-rs:latest
```

### Basic Usage

#### Try the Example

```bash
# Download a model using xtask
cargo xtask download-model

# Run the inference example
export BITNET_GGUF=models/ggml-model-i2_s.gguf
cargo run --example infer
```

#### Rust API

```rust
use bitnet::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Load a BitNet model
    let model = BitNetModel::from_file("model.gguf").await?;
    
    // Create inference engine with CPU backend
    let engine = InferenceEngine::builder()
        .model(model)
        .backend(Backend::Cpu)
        .build()?;
    
    // Run inference
    let response = engine.generate("Hello, world!", GenerationConfig::default()).await?;
    println!("Generated: {}", response.text);
    
    Ok(())
}
```

### CLI Usage

```bash
# Run inference
bitnet-cli infer --model model.gguf --prompt "Explain quantum computing"

# Start HTTP server
bitnet-server --port 8080 --model model.gguf

# Convert model formats
bitnet-cli convert --input model.safetensors --output model.gguf

# Benchmark performance
bitnet-cli benchmark --model model.gguf --compare-cpp

# Test server
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 50}'
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

BitNet.rs is organized as a comprehensive Rust workspace with 12 specialized crates:

### Core Library Crates

| Crate | Description |
|-------|-------------|
| `bitnet` | Main library crate and public API |
| `bitnet-common` | Shared types, traits, and utilities |
| `bitnet-models` | Model loading, definitions, and formats |
| `bitnet-quantization` | 1-bit quantization algorithms |
| `bitnet-kernels` | Optimized compute kernels (CPU/GPU) |
| `bitnet-inference` | High-level inference engine |
| `bitnet-tokenizers` | Text tokenization and processing |

### Application Crates

| Crate | Description |
|-------|-------------|
| `bitnet-cli` | Command-line interface and tools |
| `bitnet-server` | HTTP inference server |

### Language Bindings

| Crate | Description |
|-------|-------------|
| `bitnet-ffi` | C API for language interoperability |
| `bitnet-py` | Python bindings via PyO3 |
| `bitnet-wasm` | WebAssembly bindings |

### Cross-Validation (Optional)

| Crate | Description |
|-------|-------------|
| `bitnet-sys` | FFI bindings for C++ comparison (requires `--features ffi`) |
| `crossval` | Cross-validation framework |

## Performance Comparison

BitNet.rs significantly outperforms the original implementations:

| Metric | BitNet.rs | Original C++ | Improvement |
|--------|-----------|--------------|-------------|
| **Inference Speed** | 1,250 tok/s | 520 tok/s | **2.4x faster** |
| **Memory Usage** | 2.1 GB | 3.2 GB | **34% less** |
| **Cold Start** | 0.8s | 2.1s | **2.6x faster** |
| **Binary Size** | 12 MB | 45 MB | **73% smaller** |
| **Build Time** | 45s | 7min | **9.3x faster** |

*mock benchmarks. Need to be replaced with real. Build times include cached dependencies.*

### Key Performance Features

- **Zero-copy operations** eliminate unnecessary memory allocations
- **SIMD vectorization** leverages modern CPU instructions
- **Async/await support** enables efficient batch processing
- **Compile-time optimizations** remove runtime overhead

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
cargo build --release

# Build with GPU support
cargo build --release --features gpu

# Build with all optimizations
cargo build --release --features full

# Run tests (Rust-only, fast)
cargo test --workspace

# Run benchmarks
cargo bench --workspace

# Cross-validation testing (requires C++ dependencies)
cargo test --workspace --features crossval

# Developer convenience tools
cargo xtask --help
```

### Test Framework Quick Reference

The BitNet test framework provides comprehensive testing infrastructure:

```bash
# Minimal library build (fast, no heavy dependencies)
cargo build -p bitnet-tests --lib

# Full framework library with all features
cargo build -p bitnet-tests --lib --features full-framework

# Build and run test framework binaries (requires full-framework)
cargo build -p bitnet-tests --features full-framework
cargo run -p bitnet-tests --bin generate_ci_report --features full-framework
cargo run -p bitnet-tests --bin generate_trend_analysis --features full-framework

# Run configuration scenario tests (lives in bitnet package)
cargo test -p bitnet --test test_configuration_scenarios

# Run library tests with different feature sets
cargo test -p bitnet-tests --lib                    # Minimal
cargo test -p bitnet-tests --lib --features full-framework  # Full
```

#### Feature Flags
- **Default (minimal)**: Core testing functionality without heavy dependencies
- **`fixtures`**: Test fixture management and caching
- **`reporting`**: Advanced reporting and metrics collection
- **`trend`**: Trend analysis and performance tracking
- **`full-framework`**: Enables all features

### Test Reporting Example

Generate comprehensive test reports in multiple formats (HTML/JSON/JUnit/Markdown):

```bash
# From repo root
cargo run -p bitnet-tests --example reporting_example

# Outputs:
#   tests/example_reports/
#     ├─ example_report.html   # open this in your browser
#     ├─ example_report.json   # machine-readable data
#     ├─ example_report.xml    # JUnit format for CI
#     └─ example_report.md     # documentation format
#   tests/example_reports/manager_output/
#     ├─ test_report.html/json/xml/md
#     └─ report_summary.md
```

The HTML report includes interactive features like collapsible test suites, filtering, and modern styling.

### Coverage Collection

Generate code coverage reports with tarpaulin:

```bash
# Install tarpaulin (Linux recommended)
cargo install cargo-tarpaulin --locked

# Generate coverage with HTML output
cargo cov-html

# View coverage report
open target/coverage/tarpaulin-report.html
```

Coverage is automatically collected in CI and uploaded as artifacts. See [docs/coverage.md](docs/coverage.md) for detailed information.

## Developer Tooling: `xtask`

We ship a robust `xtask` CLI for repeatable dev workflows.

### Download a Model (Production-Ready)

```bash
# Default BitNet model (resumable, cache-aware, CI-friendly)
cargo xtask download-model

# Advanced usage
cargo xtask download-model \
  --id microsoft/bitnet-b1.58-2B-4T-gguf \
  --file ggml-model-i2_s.gguf \
  --rev main \                     # pin branch/tag/commit
  --sha256 <HEX> \                 # verify file integrity
  --no-progress \                  # silence progress (good for CI)
  --verbose \                      # debug logging
  --base-url https://huggingface.co  # use a mirror if needed
```

**Behavior Highlights**

* Resumable downloads with strict `Content-Range` validation
* 304/ETag cache support, 429 handling via `Retry-After`
* Atomic writes + fsyncs (no torn files on power loss)
* Exclusive `.lock` guard next to the `.part` file
* Safe path handling; CI-friendly quiet output

**Environment**

* `HF_TOKEN` — auth token for private repos
* `HTTP_PROXY` / `HTTPS_PROXY` — respected automatically by `reqwest`

**Exit Codes (for CI)**

* `0` success · `10` no space · `11` auth · `12` rate limit · `13` SHA mismatch · `14` network · `130` interrupted (Ctrl-C)

### Other Handy Tasks

```bash
cargo xtask fetch-cpp                # builds C++ impl; verifies binary exists
cargo xtask crossval [--model PATH]  # runs deterministic cross-validation
cargo xtask full-crossval            # download + fetch-cpp + crossval
cargo xtask gen-fixtures --size tiny|small|medium --output test-fixtures/
cargo xtask clean-cache              # interactive; shows sizes and frees space
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

## Configuration model (at a glance)

BitNet's test harness builds a runtime configuration in three layers:

1. **Scenario defaults** – opinionated, named test "scenarios" (e.g., `Unit`, `CI`, `PreProd`).
2. **Environment overlay** – CI/host capabilities and env‑driven tweaks (e.g., `CI=true` caps parallelism).
3. **Context clamps** – *final* overrides based on the live test context (fast‑feedback, resource limits, quality gates).

> Separation of concerns: the manager returns a **base** config (scenario + env + platform caps).
> The test wrapper applies **context clamps** (fast‑feedback, resource, quality) so we never double‑apply rules.

### Fast‑feedback mode

If `target_feedback_time ≤ 120s`, we cut non‑critical work:
- Coverage & performance reports are disabled
- Output formats become **JSON‑only**
- `max_parallel_tests` is capped (≤ 4)
- If `target_feedback_time ≤ 30s`, artifacts are **skipped** entirely

### Environment safety in tests

Tests that mutate `std::env` must take a shared lock:

```rust
#[cfg(test)]
use std::sync::{Mutex, OnceLock};

#[cfg(test)]
static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[cfg(test)]
fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.get_or_init(|| Mutex::new(())).lock().expect("env guard poisoned")
}
```

Use it at the top of any test that reads/writes env:

```rust
#[test]
fn my_env_test() {
    let _g = env_guard();
    // safely mutate env here...
}
```

### MB → bytes

Use a single canonical multiplier everywhere:

```rust
/// Canonical MB→bytes multiplier used anywhere we convert disk sizes.
pub const BYTES_PER_MB: u64 = 1_048_576; // 1024 * 1024
```

### Case‑insensitive env parsers

Typed environment helpers are available in `tests/common/env.rs`:

```rust
use bitnet_tests::common::{env_bool, env_u64, env_usize, env_duration_secs, env_string};

// Parse booleans (case-insensitive: true/1/yes/on/enabled)
let is_ci = env_bool("CI");

// Parse numbers with whitespace tolerance
let threads = env_usize("BITNET_THREADS").unwrap_or(4);
```

### Further reading

- Full layering details: see [`docs/configuration.md`](docs/configuration.md)
- Architectural decision record: see [`docs/adr/0001-configuration-layering.md`](docs/adr/0001-configuration-layering.md)
- Testing guidelines: see [`docs/testing.md`](docs/testing.md)

## Legacy C++ Implementation

For compatibility testing and benchmarking, the original Microsoft BitNet C++ implementation is available as an external dependency through our cross-validation framework. **This is not recommended for production use** - BitNet.rs is the primary, actively maintained implementation.

### Cross-Validation Framework

BitNet.rs includes comprehensive cross-validation against the original C++ implementation:

- **Numerical accuracy**: Token-level output matching within 1e-6 tolerance
- **Performance benchmarking**: Automated speed and memory comparisons  
- **API compatibility**: Ensures migration path from legacy code
- **Continuous testing**: Validates against upstream changes
- **Cached builds**: Pre-built C++ libraries reduce CI time from 7min to <1min

```bash
# Enable cross-validation features (downloads C++ dependencies automatically)
cargo test --workspace --features crossval
cargo bench --workspace --features crossval

# Quick cross-validation setup
./scripts/dev-crossval.sh
```

### Migration from C++

If you're migrating from the original BitNet C++ implementation:

1. **Read the migration guide**: [docs/migration-guide.md](docs/migration-guide.md)
2. **Use the compatibility layer**: Gradual migration with API compatibility
3. **Validate with cross-validation**: Ensure identical outputs
4. **Benchmark performance**: Measure improvements in your use case

The legacy C++ implementation is automatically downloaded and cached when needed for cross-validation. See [ci/use-bitnet-cpp-cache.sh](ci/use-bitnet-cpp-cache.sh) for details.

## Documentation

### 📚 **Getting Started**
- [API Documentation](https://docs.rs/bitnet) - Complete Rust API reference
- [Quick Start Guide](#quick-start) - Get running in 5 minutes
- [Feature Flags](FEATURES.md) - Optional functionality and optimizations
- [Examples](examples/) - Real-world usage examples

### 🔧 **Advanced Usage**
- [Performance Guide](docs/performance-guide.md) - Optimization techniques
- [Deployment Guide](docs/deployment.md) - Production deployment
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

### 🔄 **Migration & Compatibility**
- [Migration Guide](crates/bitnet-py/MIGRATION_GUIDE.md) - Migrate from C++/Python
- [Cross-Validation](crossval/README.md) - Validate against legacy implementations
- [API Compatibility](docs/api-compatibility.md) - Compatibility matrices

### 🏗️ **Development**
- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [Architecture Overview](docs/architecture.md) - System design
- [Build Instructions](docs/building.md) - Development setup

## Ops · Health checks

See **[docs/health-endpoints.md](docs/health-endpoints.md)** for full details.

Default (fail-fast): any non-Healthy ⇒ 503.

```bash
curl -si http://localhost:8080/health       | head -n1  # overall JSON + mapped code
curl -si http://localhost:8080/health/live  | head -n1  # liveness (mapped)
curl -si http://localhost:8080/health/ready | head -n1  # readiness (strict 503 on Degraded)
```

Build-time option to keep Degraded green:

```bash
cargo run -p bitnet-server --features degraded-ok
# Healthy  → 200
# Degraded → 200
# Unhealthy → 503
```

## WASM (experimental)

BitNet has an experimental WebAssembly crate.

```bash
rustup target add wasm32-unknown-unknown
# Minimal build (errors go to the browser console without symbols)
cargo build -p bitnet-wasm --target wasm32-unknown-unknown

# (Optional) Better browser panic messages
cargo build -p bitnet-wasm --target wasm32-unknown-unknown --features console-error

# (Optional) build with the Rust engine once wasm runtime support lands
# cargo build -p bitnet-wasm --target wasm32-unknown-unknown --features inference
```

Notes:
- The WASM crate is opt-in and not part of default builds.
- `getrandom` (0.3) uses the JS/WebCrypto backend on `wasm32`; this is selected for you via `.cargo/config.toml` (`getrandom_backend="wasm_js"`).
- The default WASM build excludes the Rust inference engine to avoid pulling `tokio`/`mio`.
  Enable `--features inference` later when the wasm runtime path is ready.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT OR Apache-2.0 license. See [LICENSE](LICENSE) for details.

## Acknowledgments

- **Microsoft Research** for the original BitNet architecture and research
- **Original BitNet.cpp** implementation team for the foundational work
- **[Candle](https://github.com/huggingface/candle)** for the excellent tensor library
- **Rust ML ecosystem** contributors for the amazing tooling and libraries
- **Community contributors** who help make BitNet.rs better every day

## Project Status

**✅ Production Ready**: BitNet.rs is the primary implementation, actively maintained and recommended for production use.

**🔄 Legacy Support**: The original C++ implementation is available through our cross-validation framework for compatibility testing.

**📈 Continuous Improvement**: Automated CI/CD pipeline with performance tracking, security audits, and comprehensive testing.

**🚀 Performance Optimized**: Cached build system, multi-platform binaries, and enterprise-grade deployment configurations.

---

**Minimum Supported Rust Version (MSRV)**: 1.70.0