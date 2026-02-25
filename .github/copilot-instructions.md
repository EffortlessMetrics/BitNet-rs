# BitNet.rs — Copilot Instructions

High-performance Rust implementation of BitNet 1-bit LLM inference. MSRV 1.92.0, Rust 2024 edition.

## Build & Test Commands

**Default features are intentionally empty. Always specify features explicitly.**

```bash
# Build
cargo build --no-default-features --features cpu
cargo build --no-default-features --features gpu

# Test (full workspace)
cargo nextest run --workspace --no-default-features --features cpu
cargo nextest run --profile ci   # CI profile: 4 threads, 5-min timeout

# Run a single test by name
cargo nextest run --workspace --no-default-features --features cpu -E 'test(my_test_name)'
# Or with cargo test
cargo test -p bitnet-inference --no-default-features --features cpu -- sampling::tests::test_temperature

# Fixture-based integration tests
cargo test -p bitnet-models --test qk256_dual_flavor_tests --no-default-features --features fixtures

# Lint & format
cargo fmt --all
cargo clippy --all-targets --no-default-features --features cpu -- -D warnings

# Skip slow QK256 scalar-kernel tests
BITNET_SKIP_SLOW_TESTS=1 cargo nextest run --workspace --no-default-features --features cpu

# Release build with native SIMD
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo build --release --no-default-features --features cpu,full-cli
```

## Architecture

The workspace is a multi-crate Rust library for neural network inference. Data flows like this:

```
bitnet-tokenizers → bitnet-models (GGUF loader) → bitnet-quantization → bitnet-kernels → bitnet-inference → bitnet-server / bitnet-cli
```

**Key crates:**
- `bitnet-common` — foundational types (`BitNetTensor`, `Device`, `QuantizationType`, errors, `warn_once!`, `StrictModeEnforcer`)
- `bitnet-kernels` — SIMD/CUDA compute kernels; `KernelManager` selects best available provider (CUDA > CPU fallback) via `KernelProvider` trait
- `bitnet-models` — GGUF/SafeTensors loading; dual-flavor I2_S detection (BitNet32-F16 vs QK256 256-element blocks)
- `bitnet-quantization` — I2_S, TL1, TL2, QK256 algorithms; QK256 dispatch in `qk256_dispatch`
- `bitnet-inference` — `ProductionInferenceEngine` (primary) and `InferenceEngine`; `SamplingStrategy` created fresh per request; streaming via `GenerationStream`
- `bitnet-server` — axum HTTP server with batch engine, concurrency manager, execution router
- `bitnet-trace` — tensor activation tracing for cross-validation against C++ reference
- `crossval` — C++ reference validation framework (bitnet.cpp + llama.cpp)
- `xtask` — developer tooling (`download-model`, `crossval`, `verify-receipt`, `setup-cpp-auto`)

**`default-members`** in `Cargo.toml` limits `cargo build/test` to the core crates. Wasm, Python bindings, FFI, and tooling crates must be addressed explicitly.

## Key Conventions

### Feature gates
GPU code must always use the unified predicate — never `#[cfg(feature = "cuda")]` alone:
```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_only_function() { ... }
```
Use `bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime}` for runtime checks.

### Environment-mutating tests
Use `temp_env::with_var` (preferred) or `EnvGuard` (RAII fallback), always with `#[serial(bitnet_env)]`:
```rust
use serial_test::serial;

#[test]
#[serial(bitnet_env)]
fn test_strict_mode() {
    temp_env::with_var("BITNET_STRICT_MODE", Some("1"), || {
        // ...
    });
}
```
`EnvGuard` is included via the `include!()` macro from `tests/support/env_guard.rs` — not a normal module import.

### Ignored tests must have justifications
Pre-commit hooks reject bare `#[ignore]`. Always add a reason:
```rust
// Model-gated test (one ignore per test function)
#[ignore = "requires model file - run manually or in CI with BITNET_GGUF set"]
fn test_inference_with_real_model() { ... }

// Performance-sensitive test
#[ignore = "Slow: QK256 scalar kernels (~0.1 tok/s); run manually with --ignored"]
fn test_qk256_full_model_inference() { ... }
```

### TDD scaffolding
~1070 `#[ignore]` tests across the workspace (all with justification strings) represent
intentional scaffolding, not bugs. ~112 in core crates, ~129 in the integration test
crate, ~652 in xtask, ~173 in crossval. Categories: real-model tests, CUDA tests, slow
mock-inference tests, crossval tests, and TDD scaffolds for unimplemented features.

### Rate-limited logging
Use `warn_once!` from `bitnet_common` for hot-path warnings that would otherwise spam logs:
```rust
use bitnet_common::warn_once;
warn_once!("unique_key", "message shown only once at WARN, then at DEBUG");
```

### Receipts
Benchmarks write a receipt to `ci/inference.json` (schema v1.0.0). `compute_path` must be `"real"` — never `"mock"`. Verify with `cargo run -p xtask -- verify-receipt`.

### BITNET_STRICT_MODE
When `BITNET_STRICT_MODE=1`: validation fails with exit code 8 on suspicious LayerNorm weights, `BITNET_GPU_FAKE` is ignored, and mock inference paths are rejected.

### Quantization flavor detection
When loading I2_S tensors, QK256 (256-element blocks) is checked before BitNet32-F16 (32-element blocks with inline F16 scales) for closer-match disambiguation. See `bitnet-models/src/formats/` for the loader.

### clippy & formatting
`max_width = 100`, `tab_spaces = 4`, Rust 2024 edition. `cognitive-complexity-threshold = 30`. `allow-expect-in-tests = true`, `allow-unwrap-in-tests = true`.
