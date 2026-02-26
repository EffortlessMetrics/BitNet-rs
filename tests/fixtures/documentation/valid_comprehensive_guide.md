# BitNet-rs Feature Flag Comprehensive Guide (VALID Pattern)

## Feature Flag Philosophy

BitNet-rs uses **empty default features** to ensure explicit dependency control and prevent bloat. You must always specify features:

```bash
cargo build --no-default-features --features cpu|gpu
```

## Core Features

### CPU Inference (`--features cpu`)

SIMD-optimized CPU inference with architecture-specific kernels:

```bash
# Build CPU inference
cargo build --no-default-features --features cpu

# Run inference
cargo run -p bitnet-cli --no-default-features --features cpu -- \
  infer --model model.gguf --prompt "Hello"

# Test CPU kernels
cargo test --workspace --no-default-features --features cpu
```

**Supported architectures:**
- x86_64: AVX2, AVX-512 SIMD
- aarch64: ARM NEON SIMD

### GPU Inference (`--features gpu`)

CUDA-accelerated inference with mixed precision support:

```bash
# Build GPU inference
cargo build --no-default-features --features gpu

# Run GPU inference
CUDA_VISIBLE_DEVICES=0 cargo run -p bitnet-cli \
  --no-default-features --features gpu -- \
  infer --model model.gguf --device cuda:0

# Test GPU kernels (requires CUDA toolkit)
cargo test --workspace --no-default-features --features gpu
```

**Requirements:**
- CUDA toolkit 11.8+ or 12.x
- NVIDIA GPU with compute capability 7.0+

**Note**: `cuda` is a temporary alias for `gpu` feature. Prefer `--features gpu`.

### Mixed Precision (`--features gpu,mixed_precision`)

FP16/BF16 acceleration for Tensor Core GPUs:

```bash
cargo build --no-default-features --features gpu,mixed_precision
```

### Cross-Validation (`--features crossval`)

Systematic comparison with C++ reference implementation:

```bash
export BITNET_GGUF=path/to/model.gguf
cargo run -p xtask -- crossval
```

## Development Commands

### Quality Checks

```bash
# Format all code
cargo fmt --all

# Lint CPU features
cargo clippy --all-targets --no-default-features --features cpu -- -D warnings

# Lint GPU features
cargo clippy --all-targets --no-default-features --features gpu -- -D warnings

# Lint all features
cargo clippy --all-targets --all-features -- -D warnings
```

### Testing

```bash
# CPU tests
cargo test --workspace --no-default-features --features cpu

# GPU tests
cargo test --workspace --no-default-features --features gpu

# Integration tests
cargo test --workspace --no-default-features --features cpu,crossval
```

## Feature Composition

Combine features as needed:

```bash
# CPU + FFI bridge
cargo build --no-default-features --features cpu,ffi

# GPU + mixed precision + cross-validation
cargo build --no-default-features --features gpu,mixed_precision,crossval

# CPU + SentencePiece tokenizer
cargo build --no-default-features --features cpu,spm
```

## Troubleshooting

### "GPU not available" errors

```bash
# Check GPU detection
cargo run -p xtask -- preflight

# Force fake GPU for testing
BITNET_GPU_FAKE=cuda cargo run -p xtask -- preflight

# Disable GPU for CPU fallback testing
BITNET_GPU_FAKE=none cargo test --no-default-features --features gpu
```

### Feature compilation verification

```rust
// Check if GPU was compiled
use bitnet_kernels::device_features::gpu_compiled;

fn main() {
    if gpu_compiled() {
        println!("GPU support compiled");
    } else {
        println!("CPU-only build");
    }
}
```

## Summary

✅ **Always use**: `--no-default-features --features cpu|gpu`
✅ **Prefer**: `--features gpu` over `--features cuda`
✅ **Test with**: `cargo test --workspace --no-default-features --features <feature>`
✅ **Quality**: Run clippy and fmt before committing
