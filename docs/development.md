# BitNet-rs Development Guide

This is the main development guide for BitNet-rs, providing an overview of development workflows, tools, and best practices for neural network quantization and inference.

## Quick Start

1. **Setup Development Environment**
   ```bash
   git clone https://github.com/EffortlessMetrics/BitNet-rs.git
   cd BitNet-rs
   rustup update stable  # Requires Rust 1.92.0+
   ```

2. **Run Tests**
   ```bash
   # Quick CPU tests
   cargo test --no-default-features --workspace --no-default-features --features cpu

   # GPU tests (requires CUDA)
   cargo test --no-default-features --workspace --no-default-features --features gpu
   ```

3. **Download Models for Testing**
   ```bash
   cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf
   ```

## Development Workflows

### Core Development Commands

```bash
# Format code
cargo fmt --all

# Lint with Clippy
cargo clippy --all-targets --all-features -- -D warnings

# Build documentation
cargo doc --workspace --no-default-features --features cpu --no-deps

# Run specific test suite
cargo test --package bitnet-quantization --no-default-features --features cpu
```

### Neural Network Development

```bash
# Cross-validate against C++ reference
cargo run -p xtask -- crossval

# Verify model compatibility
cargo run -p xtask -- verify --model models/test.gguf

# Benchmark performance
cargo run -p xtask -- benchmark --model models/test.gguf --tokens 128
```

## Development Guides

### Core Development
- **[Build Commands](development/build-commands.md)** - Comprehensive build and test commands
- **[Development Standards](development/development-standards.md)** - Coding standards and best practices
- **[CI Integration](development/ci-integration.md)** - Continuous integration setup
- **[Test Suite](development/test-suite.md)** - Testing framework and practices

### Specialized Development
- **[GPU Development](development/gpu-development.md)** - CUDA kernel development and optimization
- **[GPU Setup Guide](development/gpu-setup-guide.md)** - GPU environment setup
- **[Cross-validation Setup](development/cross-validation-setup.md)** - C++ reference comparison
- **[Validation Framework](development/validation-framework.md)** - Quality assurance processes

### Tools and Automation
- **[xtask Guide](development/xtask.md)** - Task runner and automation tools

## Feature Flags

BitNet-rs uses feature flags for conditional compilation. **Default features are EMPTY** - always specify explicitly:

```bash
# CPU inference with SIMD optimizations
cargo build --no-default-features --features cpu

# GPU inference with CUDA kernels
cargo build --no-default-features --features gpu

# FFI bridge for C++ compatibility
cargo build --no-default-features --features cpu,ffi

# WebAssembly support
cargo build --target wasm32-unknown-unknown --no-default-features --features browser
```

## Architecture Overview

### Workspace Structure

- **`bitnet`** - Main library with unified public API
- **`bitnet-quantization`** - 1-bit quantization algorithms (I2S, TL1, TL2, IQ2_S)
- **`bitnet-kernels`** - High-performance SIMD/CUDA kernels
- **`bitnet-inference`** - Inference engine with streaming support
- **`bitnet-models`** - Model loading (GGUF, SafeTensors)
- **`bitnet-tokenizers`** - Universal tokenizer with GGUF integration
- **`crossval`** - Cross-validation against C++ reference
- **`xtask`** - Development task automation

### Key Design Principles

1. **Zero-Copy Operations** - Memory-mapped models, careful lifetime management
2. **Device-Aware Computing** - Automatic GPU/CPU selection with graceful fallback
3. **Neural Network Focus** - Optimized for 1-bit quantization and inference
4. **Cross-Platform** - Rust-first with WebAssembly and C++ FFI support

## Testing Strategy

### Test Types

1. **Unit Tests** - Individual function and module testing
2. **Integration Tests** - Cross-module functionality
3. **Property Tests** - Randomized testing with QuickCheck
4. **Cross-validation Tests** - Comparison with C++ reference implementation
5. **Mutation Tests** - Robustness testing (CI only)

### Neural Network Specific Testing

- **Accuracy Validation** - >99% quantization accuracy requirement
- **Performance Benchmarks** - Inference speed and memory usage
- **GGUF Compatibility** - Format compliance and model loading
- **GPU/CPU Parity** - Consistent results across devices

## Performance Optimization

### Quantization Performance
- Target >99% accuracy for I2S quantization
- Vectorized operations with SIMD intrinsics
- GPU acceleration with mixed precision (FP16/BF16)
- Memory-efficient bit packing and alignment

### Inference Performance
- Streaming generation with low latency
- Batch processing optimization
- Cache-friendly memory access patterns
- Zero-copy tensor operations

## Getting Help

- **Issues**: Report bugs and request features on GitHub
- **Documentation**: Comprehensive guides in `docs/` directory
- **Examples**: Working code examples in `examples/` directory
- **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines

## Next Steps

1. **New Contributors**: Start with [CONTRIBUTING.md](../CONTRIBUTING.md)
2. **GPU Development**: See [GPU Development Guide](development/gpu-development.md)
3. **Testing**: Review [Test Suite Guide](development/test-suite.md)
4. **Performance**: Check [Performance Benchmarking](performance-benchmarking.md)

---

For comprehensive information on specific development topics, explore the linked guides above.
