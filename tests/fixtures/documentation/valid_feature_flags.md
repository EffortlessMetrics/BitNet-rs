# Building BitNet.rs (VALID Documentation Pattern)

## Feature Flags

BitNet.rs uses **empty default features**. Always specify features explicitly:

```bash
# CPU build (SIMD-optimized)
cargo build --no-default-features --features cpu

# GPU build (CUDA acceleration)
cargo build --no-default-features --features gpu

# GPU with mixed precision (FP16/BF16)
cargo build --no-default-features --features gpu,mixed_precision
```

## Feature Flag Reference

- `cpu`: SIMD-optimized CPU inference (AVX2/AVX-512/NEON)
- `gpu`: CUDA acceleration with automatic fallback
- `ffi`: C++ FFI bridge for gradual migration
- `crossval`: Cross-validation against C++ reference
- `spm`: SentencePiece tokenizer support

**Note**: `cuda` is a temporary alias for `gpu` (will be removed in future versions). Prefer using `--features gpu`.

## Testing

```bash
# CPU tests
cargo test --workspace --no-default-features --features cpu

# GPU tests (requires CUDA toolkit)
cargo test --workspace --no-default-features --features gpu

# Cross-validation tests
cargo test --workspace --no-default-features --features cpu,crossval
```

## Development Workflow

```bash
# Standard development cycle
cargo fmt --all
cargo clippy --all-targets --no-default-features --features cpu -- -D warnings
cargo test --workspace --no-default-features --features cpu

# GPU development
cargo clippy --all-targets --no-default-features --features gpu -- -D warnings
cargo test --workspace --no-default-features --features gpu
```

## Common Pitfalls

❌ **WRONG**: `cargo build` (uses empty default features)
✅ **RIGHT**: `cargo build --no-default-features --features cpu`

❌ **WRONG**: `cargo test` (no features specified)
✅ **RIGHT**: `cargo test --no-default-features --features cpu`
