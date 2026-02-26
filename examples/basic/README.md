# Basic Examples

This directory contains simple, standalone examples that demonstrate core BitNet-rs functionality.

## Examples

- **`cpu_inference.rs`** - Basic CPU inference example
- **`gpu_inference.rs`** - GPU-accelerated inference example
- **`streaming.rs`** - Streaming text generation
- **`convolution_demo.rs`** - 2D convolution with quantization

## Running Examples

```bash
# CPU inference
cargo run --example cpu_inference --no-default-features --features cpu

# GPU inference (requires CUDA)
cargo run --example gpu_inference --no-default-features --features gpu

# Streaming generation
cargo run --example streaming --no-default-features --features cpu
```

## Prerequisites

- **Rust 1.90.0+** (supports Rust 2024 edition)
- For GPU examples: CUDA toolkit and compatible GPU
- Model files in GGUF format (use `cargo run -p xtask -- download-model` to get started)
