# Feature Gates Quick Reference

## One-Minute Summary

BitNet.rs **requires explicit feature specification** - defaults are empty. All builds must use `--no-default-features --features <list>`.

```bash
# Three most common commands:
cargo build --no-default-features --features cpu        # CPU inference (most common)
cargo build --no-default-features --features gpu        # GPU (requires CUDA)
cargo test --no-default-features --features cpu,fixtures # Full test suite
```

---

## Feature Combinations by Use Case

### Development

```bash
# Fast CPU iteration
cargo build --no-default-features --features cpu
cargo test --no-default-features --features cpu

# With all tests
cargo test --no-default-features --features cpu,fixtures

# GPU development (if CUDA available)
cargo build --no-default-features --features gpu
cargo test --no-default-features --features gpu
```

### Production

```bash
# Optimized CPU binary
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo build --release --no-default-features --features cpu

# GPU binary (requires CUDA)
RUSTFLAGS="-C opt-level=3 -C lto=thin" \
  cargo build --release --no-default-features --features gpu
```

### Testing & Validation

```bash
# Core tests (quantization, models, kernels)
cargo test --no-default-features --features cpu

# With fixtures (GGUF integration tests)
cargo test --no-default-features --features cpu,fixtures

# Cross-validation (requires C++ setup)
cargo test --no-default-features --features cpu,crossval

# Full CI suite
cargo nextest run --workspace --no-default-features --features cpu --profile ci
cargo nextest run --workspace --no-default-features --features gpu --profile ci
```

### CLI Tools

```bash
# Run inference
cargo run -p bitnet-cli --no-default-features --features cpu -- run \
  --model model.gguf --prompt "Test" --max-tokens 32

# Interactive chat
cargo run -p bitnet-cli --no-default-features --features cpu -- chat \
  --model model.gguf --tokenizer tokenizer.json

# Full-featured CLI (with benchmarking)
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- benchmark \
  --model model.gguf --tokens 128
```

---

## Feature Groups

### By Device

| Feature | Includes | Binary Size |
|---------|----------|------------|
| `cpu` | CPU kernels + inference | ~40MB |
| `gpu` | CUDA kernels + inference | ~80MB |
| (empty) | Models only | ~5MB |

### By Optimization

| Feature | Effect | x86_64 | aarch64 |
|---------|--------|--------|---------|
| `avx2` | ✅ Enabled if compiled | ✅ Auto-detected | ❌ N/A |
| `avx512` | ✅ Enabled if compiled | ✅ Auto-detected | ❌ N/A |
| `neon` | ✅ Enabled if compiled | ❌ N/A | ✅ Auto-detected |

*Note*: SIMD features are always runtime-detected. Compile-time gate ensures code compiles on all archs.

### By Test Category

| Feature | Tests Enabled | Purpose |
|---------|---------------|---------|
| (none) | Core tests | ~50 tests |
| `fixtures` | +GGUF integration | ~30 additional tests |
| `full-framework` | +reporting, trend, coverage | ~20 additional bins |

---

## Environment Variables (Runtime Overrides)

```bash
# GPU detection (testing only)
BITNET_GPU_FAKE=cuda   # Pretend GPU available
BITNET_GPU_FAKE=none   # Pretend no GPU
BITNET_STRICT_MODE=1   # Ignore GPU_FAKE, use real detection

# Testing
BITNET_SKIP_SLOW_TESTS=1  # Skip QK256 scalar kernel tests (slow)

# Logging
RUST_LOG=warn    # Reduce noise (recommended for inference output)
RUST_LOG=debug   # Full debug output

# Determinism
BITNET_DETERMINISTIC=1
BITNET_SEED=42
RAYON_NUM_THREADS=1
```

---

## What Each Feature Does

### Primary Features

**`cpu`** - CPU-based inference
- Enables bitnet-inference/cpu
- Enables bitnet-kernels/cpu-optimized
- SIMD detection at runtime (avx2/avx512/neon)
- ~40MB binary

**`gpu`** - GPU-based inference (alias: `cuda`)
- Requires CUDA toolkit installed
- Enables CUDA kernel compilation
- Graceful fallback to CPU at runtime
- ~80MB binary

### Component Features (Usually Auto-Enabled)

**`kernels`** - Compute kernels module
- Auto-enabled by `cpu` or `gpu`
- Provides KernelManager for device selection

**`inference`** - Autoregressive generation engine
- Auto-enabled by `cpu` or `gpu`
- Provides streaming generation API

**`tokenizers`** - Universal tokenizer support
- Auto-enabled by `cpu` or `gpu`
- Auto-discovery, embedding, dynamic loading

### Test Features (Optional)

**`fixtures`** - GGUF fixture integration tests
- ~30 additional tests
- Requires models in `models/` directory
- Can be slow (~5 min)

**`full-framework`** - Complete test infrastructure
- Includes `fixtures`, `reporting`, `trend`
- CI binaries and report generation
- Slow tests included

### Cross-Validation Features

**`crossval`** - C++ reference comparison
- Requires BitNet.cpp C++ repository
- Requires C++ compiler (gcc/clang)
- Generates parity reports with cosine similarity

**`trace`** - Tensor activation tracing
- Enables debug traces for divergence analysis
- Small runtime overhead

---

## Common Problems & Solutions

### Problem: Build fails with no inference code

```bash
❌ cargo build
   error[E0463]: can't find crate for `bitnet_inference`

✅ cargo build --no-default-features --features cpu
```

**Why**: Features are empty by default. Must specify.

### Problem: GPU feature doesn't enable GPU kernels

```bash
❌ #[cfg(feature = "gpu")]
   // Only compiled with feature=gpu, breaks with cuda feature

✅ #[cfg(any(feature = "gpu", feature = "cuda"))]
   // Compiled with either feature
```

**Why**: `cuda` is an alias for `gpu`, must use `any()`.

### Problem: Test only runs with specific feature

```bash
❌ #[cfg(feature = "fixtures")]
   #[test]
   fn test_with_fixtures() { }
   // Test silently skipped if fixtures not enabled

✅ // In Cargo.toml:
   [[test]]
   name = "fixture_tests"
   required-features = ["fixtures"]
```

**Why**: Use `required-features` - prevents silent skips.

### Problem: GPU code compiles but fails at runtime

```bash
❌ if gpu_compiled() {
       use_gpu()  // Fails if GPU not available at runtime
   }

✅ if gpu_compiled() && gpu_available_runtime() {
       use_gpu()  // Safe - hardware actually available
   } else if gpu_compiled() {
       use_cpu()  // Fallback to CPU
   }
```

**Why**: Compile-time check !== runtime hardware check.

---

## Feature Matrix for CI

```bash
# Minimal testing
cargo test --no-default-features --features cpu

# Full matrix
cargo test --no-default-features --features cpu
cargo test --no-default-features --features gpu      # (if CUDA available)
cargo test --no-default-features --features cpu,fixtures
cargo test --no-default-features --features cpu,crossval  # (if C++ setup)

# Recommended with nextest (prevents hangs)
cargo nextest run --workspace --no-default-features --features cpu --profile ci
```

---

## See Also

- **Full Documentation**: `docs/reference/FEATURE_GATES_PATTERNS.md`
- **Build Guide**: `docs/development/build-commands.md`
- **GPU Setup**: `docs/GPU_SETUP.md`
- **Cross-Validation**: `docs/explanation/cpp-setup.md`
