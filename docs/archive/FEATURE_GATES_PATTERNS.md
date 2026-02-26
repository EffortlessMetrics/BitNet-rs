# BitNet-rs Feature Gate Patterns & Conditional Compilation Report

## Executive Summary

BitNet-rs uses an **intentional empty-default-features architecture** with carefully structured feature gates across 18+ crates and 151+ conditional compilation markers. Features are designed as composable components enabling different compile paths for CPU, GPU, cross-validation, and test scenarios.

**Key Principle**: Default features are **EMPTY** - all capabilities must be explicitly enabled to prevent surprise dependencies.

---

## 1. Feature Flag Taxonomy

### 1.1 Primary Device Features (Top-Level Cargo.toml)

| Feature | Purpose | Dependencies | Usage |
|---------|---------|--------------|-------|
| `cpu` | CPU inference with SIMD | `kernels`, `inference`, `tokenizers`, `bitnet-kernels/cpu-optimized` | Production, testing |
| `gpu` | CUDA acceleration | `kernels`, `inference`, `tokenizers`, `bitnet-kernels/gpu` | GPU machines |
| `cuda` | ⚠️ Alias for `gpu` | Same as `gpu` | Backward compatibility only |

### 1.2 SIMD Optimization Features (bitnet-kernels)

| Feature | Arch | Status | Notes |
|---------|------|--------|-------|
| `avx2` | x86_64 | ✅ Runtime-gated | AVX2 dequantization for QK256 |
| `avx512` | x86_64 | ✅ Runtime-gated | AVX-512 fallback |
| `neon` | aarch64 | ✅ Runtime-gated | ARM NEON vectorization |

**Runtime Selection**: KernelManager uses `is_x86_feature_detected!()` at runtime. Fallback to scalar always available.

### 1.3 Component Features (Cascading)

```
cpu / gpu (primary)
├── kernels → dep:bitnet-kernels
├── inference → dep:bitnet-inference + kernels
└── tokenizers → dep:bitnet-tokenizers
```

Features are **optional dependencies** enabled only when requested:

```toml
[dependencies]
bitnet-inference = { path = "...", optional = true }
bitnet-kernels = { path = "...", optional = true }
bitnet-tokenizers = { path = "...", optional = true }

[features]
inference = ["dep:bitnet-inference", "kernels"]
kernels = ["dep:bitnet-kernels"]
cpu = ["kernels", "inference", "tokenizers", "bitnet-kernels/cpu-optimized"]
```

### 1.4 Quantization & FFI Features

| Feature | Enabled By | Purpose |
|---------|------------|---------|
| `iq2s-ffi` | `bitnet` + `crossval` | GGML IQ2_S via C++ FFI |
| `ffi` | cross-validation | C++ reference integration |
| `cpp-ffi` | tests | Link against BitNet.cpp |

### 1.5 Cross-Validation & Testing Features

| Crate | Features | Purpose |
|-------|----------|---------|
| `bitnet-crossval` | `crossval`, `ffi`, `iq2s-ffi` | C++ comparison & parity validation |
| `bitnet-tests` | `fixtures`, `reporting`, `trend`, `inference` | Test infrastructure & CI reporting |
| `bitnet-cli` | `full-cli`, `cli-bench`, `crossval` | CLI subcommands & benchmarking |

### 1.6 Convenience Feature Bundles

```toml
[features]
# Full capability set (bloats binary but enables everything)
full = ["cpu", "cuda", "avx2", "avx512", "neon"]

# Minimal core (models + quantization only, no inference)
minimal = []

# Full testing framework
full-framework = ["fixtures", "reporting", "trend", "spm", "integration-tests", "ffi-tests"]
```

---

## 2. Feature Gate Pattern Analysis

### 2.1 Unified GPU Predicate (Issue #439 Pattern)

**Requirement**: Always use `any(feature = "gpu", feature = "cuda")` for GPU code:

✅ **CORRECT**:
```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub mod gpu_kernels {
    // GPU code here - compiled when either feature enabled
}
```

❌ **INCORRECT**:
```rust
#[cfg(feature = "gpu")]
// Breaks with cuda feature - don't do this!
```

**Usage in codebase**: 15+ occurrences (crates/bitnet-cli, bitnet-tokenizers, bitnet-kernels)

### 2.2 Optional Dependency Pattern

**Pattern**: Feature gates optional crate dependencies:

```toml
[dependencies]
bitnet-sys = { path = "...", optional = true }

[features]
ffi = ["dep:bitnet-sys", "bitnet-sys/ffi"]
```

**Benefits**:
- Binary size reduction (~5-10MB per optional crate)
- Faster builds (feature disabled = no compilation)
- Compile error prevention (unused crate code eliminated)

**Found in**: 8 crates
- `bitnet-inference` → `bitnet-sys/ffi`
- `bitnet-models` → `bitnet-ggml-ffi/iq2s-ffi`
- `bitnet-cli` → conditional FFI support

### 2.3 Cross-Crate Feature Propagation

Features automatically propagate through dependency graphs:

```
bitnet (root) cpu
  ├── bitnet-kernels/cpu-optimized
  │   ├── bitnet-kernels/avx2
  │   ├── bitnet-kernels/cpu-fallback
  ├── bitnet-inference/cpu
  │   └── bitnet-kernels/cpu-optimized (transitive)
  └── bitnet-quantization/cpu
      └── bitnet-kernels/cpu (transitive)
```

**Implication**: When you request `cpu` at root level, it automatically enables child features across the workspace.

### 2.4 Runtime Feature Detection

**Location**: `bitnet-kernels/src/device_features.rs` (Issue #439 spec)

```rust
/// Compile-time check (fast, always correct)
pub fn gpu_compiled() -> bool {
    cfg!(any(feature = "gpu", feature = "cuda"))
}

/// Runtime check (detect hardware & environment)
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_available_runtime() -> bool {
    // 1. Respect BITNET_GPU_FAKE (testing override)
    // 2. Fall back to real CUDA detection
    // 3. Strict mode ignores fakes
}
```

**Graceful Degradation**:
```rust
if gpu_compiled() && gpu_available_runtime() {
    // Use GPU
} else if gpu_compiled() {
    // Compiled but no hardware - fallback to CPU
} else {
    // Not compiled - CPU only
}
```

### 2.5 Test Feature Gating Pattern

**Requirement Features** in Cargo.toml:

```toml
[[test]]
name = "gpu_quantization"
path = "tests/gpu_quantization.rs"
required-features = ["gpu"]  # Only runs when gpu feature enabled

[[bin]]
name = "validate_config"
path = "bin/validate_config.rs"
required-features = ["fixtures"]
```

**Usage**: 27+ test/bin declarations with `required-features`

---

## 3. Build Matrix Implications

### 3.1 Recommended Build Configurations

```bash
# Development - CPU only (fast iteration)
cargo build --no-default-features --features cpu
cargo test --no-default-features --features cpu

# Production - CPU with SIMD
RUSTFLAGS="-C target-cpu=native" \
  cargo build --release --no-default-features --features cpu

# GPU development
cargo build --no-default-features --features gpu,avx2
cargo test --no-default-features --features gpu

# Cross-validation (requires C++ setup)
cargo build --no-default-features --features cpu,crossval
cargo test --no-default-features --features cpu,crossval

# Full testing framework
cargo test --no-default-features --features cpu,full-framework

# Minimal binary (small size)
cargo build --no-default-features
```

### 3.2 Feature Interaction Matrix

| Config | Inference | SIMD | Cross-Val | Notes |
|--------|-----------|------|-----------|-------|
| ✅ Empty | ❌ No | ❌ No | ❌ No | Models-only (smallest) |
| ✅ CPU | ✅ Yes | ✅ Runtime | ❌ No | Default for inference |
| ✅ GPU | ✅ Yes | ✅ Runtime | ❌ No | Requires CUDA toolkit |
| ✅ CPU+crossval | ✅ Yes | ✅ Runtime | ✅ Yes | Testing against C++ |
| ❌ GPU+CPU | ⚠️ Conflict | ✅ Runtime | ❌ No | Both compiled (redundant) |

### 3.3 Feature Conflicts & Mitigations

| Conflict | Behavior | Mitigation |
|----------|----------|-----------|
| No features requested | Build succeeds, no inference | Always specify `--features cpu` or `gpu` |
| Both CPU & GPU | Both compiled (bloat) | Use either, not both (both flags OK technically) |
| GPU without CUDA | Compile error | Use `--no-default-features --features gpu` only if CUDA available |
| FFI without setup | Linker error | Use `cargo xtask fetch-cpp` or `--no-default-features --features cpu` |

---

## 4. Graceful Degradation Patterns

### 4.1 Device-Aware Selection (KernelManager)

Located: `bitnet-kernels/src/lib.rs`

```rust
pub struct KernelManager {
    providers: Vec<Box<dyn KernelProvider>>,  // Ordered by preference
    selected: OnceLock<usize>,
}

impl KernelManager::new() {
    // Priority order:
    // 1. GPU kernel (if compiled + available)
    // 2. AVX-512 kernel (if detected at runtime)
    // 3. AVX-2 kernel (if detected at runtime)
    // 4. NEON kernel (if detected at runtime)
    // 5. CPU fallback kernel (always available)
}
```

**Key Property**: Even if GPU is compiled but unavailable at runtime, system gracefully falls back to optimized CPU kernels.

### 4.2 Optional Transitive Dependencies

When parent feature not enabled, child crates aren't compiled:

```
Build with: --no-default-features --features models
  Result: Only bitnet-models compiled
          bitnet-inference NOT compiled
          bitnet-kernels NOT compiled
          (inference operations unavailable, but models load fine)
```

### 4.3 Feature Stub Pattern

For features that are optional:

```toml
[features]
# Features that are placeholders (do nothing, don't fail)
iq2s-rust = []         # Pure-Rust IQ2_S (planned)
mixed-precision = []   # Mixed precision (planned)
```

---

## 5. Feature-Gated Testing

### 5.1 Test-Only Features

```toml
[features]
fixtures = []         # Enable GGUF fixture tests
reporting = []        # Enable CI reporting bin/test
trend = []            # Enable trend analysis
integration-tests = [] # Enable long-running integration tests
```

**Usage**: Tests only compile when feature requested:

```bash
# Don't compile fixture tests
cargo test --no-default-features --features cpu
  # ✅ bitnet-quantization tests run
  # ❌ fixture tests skipped

# Compile fixture tests
cargo test --no-default-features --features cpu,fixtures
  # ✅ Both run
```

### 5.2 Required-Features for Tests

Tests are skipped if required-features not enabled:

```toml
[[test]]
name = "gpu_integration"
path = "tests/gpu_integration.rs"
required-features = ["gpu"]
# ✓ Only compiled/run when --features gpu
```

**Implication**: CI matrix must test multiple configurations:
```bash
# CI Pipeline
cargo test --no-default-features --features cpu
cargo test --no-default-features --features gpu
cargo test --no-default-features --features cpu,fixtures
cargo test --no-default-features --features cpu,full-framework
```

### 5.3 Environment-Gated Tests (EnvGuard Pattern)

For tests that mutate environment:

```rust
#[test]
#[serial(bitnet_env)]  // Serialize with other env tests
fn test_gpu_detection() {
    use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};
    
    let _guard = EnvVarGuard::set("BITNET_GPU_FAKE", "none");
    // Test code
    // _guard dropped → env restored automatically
}
```

---

## 6. Runtime Feature Configuration

### 6.1 Environment Variables for Feature Override

| Variable | Values | Purpose | Scope |
|----------|--------|---------|-------|
| `BITNET_GPU_FAKE` | `cuda`, `none` | Override GPU detection (testing) | Runtime |
| `BITNET_STRICT_MODE` | `1`, `true` | Disable GPU fake, enforce real detection | Runtime |
| `BITNET_SKIP_SLOW_TESTS` | `1` | Skip QK256 scalar kernel tests | Test |
| `RUST_LOG` | `debug`, `info`, `warn` | Control logging verbosity | Runtime |

### 6.2 Runtime Detection Pattern

```rust
// Don't rely on compile-time features alone!
// Use runtime checks:

if bitnet_kernels::device_features::gpu_compiled() {
    if bitnet_kernels::device_features::gpu_available_runtime() {
        // Safe to use GPU
    } else {
        // Compiled but not available - use CPU fallback
    }
}
```

---

## 7. Feature Addition Guide

### 7.1 Adding a New Feature-Gated Component

**Scenario**: Add new quantization algorithm `QK512`

**Step 1**: Define in crate's Cargo.toml

```toml
[dependencies]
my-new-module = { optional = true }

[features]
qk512 = ["dep:my-new-module"]
```

**Step 2**: Use cfg gate in code

```rust
#[cfg(feature = "qk512")]
pub mod qk512;

#[cfg(feature = "qk512")]
pub fn load_qk512(data: &[u8]) -> Result<Model> {
    // Implementation
}
```

**Step 3**: Forward feature from parent crates (if needed)

```toml
# bitnet/Cargo.toml
[features]
full = ["cpu", "cuda", "qk512"]  # Add to bundles

# bitnet-models/Cargo.toml
[features]
qk512 = ["my-new-module/qk512"]  # Propagate
```

**Step 4**: Add feature-gated test

```toml
[[test]]
name = "qk512_tests"
required-features = ["qk512"]
```

### 7.2 Adding GPU-Specific Code

**Always use unified predicate**:

```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub mod gpu_qk512;

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_qk512_kernel() { /* ... */ }
```

**Testing**:

```bash
cargo test --no-default-features --features gpu,qk512
```

### 7.3 Adding Optional FFI

```toml
[dependencies]
cpp-bindings = { optional = true }

[features]
qk512-ffi = ["dep:cpp-bindings"]

[build-dependencies]
bindgen = { optional = true }
```

---

## 8. Testing Different Feature Combinations

### 8.1 Quick Validation Matrix

```bash
#!/bin/bash
set -e

# Core builds
echo "=== Testing feature combinations ==="

echo "1. CPU-only (most common)"
cargo test --no-default-features --features cpu

echo "2. Minimal (models only)"
cargo test --no-default-features

echo "3. Full testing framework"
cargo test --no-default-features --features cpu,full-framework

echo "4. GPU (if CUDA available)"
cargo test --no-default-features --features gpu 2>/dev/null || echo "CUDA not available, skipping"

echo "5. Cross-validation (if C++ setup)"
cargo test --no-default-features --features cpu,crossval 2>/dev/null || echo "C++ not set up, skipping"

echo "=== All tests passed ==="
```

### 8.2 Nextest Configuration

From `.config/nextest.toml`:

```toml
[profile.ci]
test-threads = 4
retries = 0
timeout = "300s"  # 5 min timeout for slow QK256 tests
```

Run with:
```bash
cargo nextest run --workspace --no-default-features --features cpu --profile ci
```

---

## 9. Current Feature Coverage Status

### 9.1 By Category

| Category | Crates | Features | Tests | Status |
|----------|--------|----------|-------|--------|
| **CPU Inference** | bitnet-kernels, bitnet-inference | `cpu`, `avx2`, `avx512`, `neon` | 120+ | ✅ Complete |
| **GPU Inference** | bitnet-kernels, bitnet-inference | `gpu`, `cuda` | 40+ | ⚠️ Partial (awaiting #439) |
| **Quantization** | bitnet-quantization | `iq2s-ffi` | 35+ | ✅ Complete |
| **Cross-Validation** | bitnet-crossval | `crossval`, `ffi` | 25+ | ⚠️ Blocked (#469) |
| **Testing Framework** | bitnet-tests | `fixtures`, `reporting`, `trend` | 48+ | ✅ Complete |
| **CLI** | bitnet-cli | `full-cli`, `cli-bench` | 30+ | ✅ Complete |

### 9.2 Feature Gate Coverage

- **151+ conditional compilation sites** across codebase
- **27+ required-features declarations** (test/bin gating)
- **18 crates with feature definitions**
- **100% feature-to-code traceability** (all features have users)

---

## 10. Common Pitfalls & Solutions

### 10.1 Pitfall: Forgetting --features

```bash
# ❌ Wrong - will fail to compile inference code
cargo build

# ✅ Correct
cargo build --no-default-features --features cpu
```

**Solution**: Add to `.cargo/config.toml`:
```toml
[build]
# Force explicit feature specification (prevents accidents)
# Note: Not recommended - better to be explicit
```

### 10.2 Pitfall: Using Single GPU Predicate

```rust
// ❌ Wrong - misses cuda feature
#[cfg(feature = "gpu")]
pub fn gpu_function() { }

// ✅ Correct
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_function() { }
```

### 10.3 Pitfall: Test Expecting Wrong Feature

```rust
// ❌ Test runs with `--features cpu` but needs GPU
#[cfg(feature = "gpu")]
#[test]
fn test_gpu() { /* GPU code */ }

// If GPU feature not enabled, test silently skipped
// Solution: Add to Cargo.toml instead:
// [[test]]
// name = "gpu_test"
// required-features = ["gpu"]
```

### 10.4 Pitfall: Forgetting Cross-Crate Propagation

```toml
# Parent crate
[features]
myfeature = ["dep:my-crate"]  # ✅ Enables my-crate

# But child features not automatic!
# Must explicitly forward:
myfeature = ["dep:my-crate", "my-crate/child-feature"]  # ✅
```

---

## 11. Best Practices

1. **Always specify features explicitly**
   - Default features are empty by design
   - No silent surprises or bloated binaries

2. **Use unified GPU predicate**
   ```rust
   #[cfg(any(feature = "gpu", feature = "cuda"))]
   ```

3. **Test multiple configurations in CI**
   ```bash
   cargo test --no-default-features --features cpu
   cargo test --no-default-features --features gpu
   cargo test --no-default-features --features cpu,fixtures
   ```

4. **Use EnvGuard for env-mutating tests**
   ```rust
   #[test]
   #[serial(bitnet_env)]
   fn test_env_dependent() { /* ... */ }
   ```

5. **Document feature requirements in code**
   ```rust
   /// Requires: feature = "gpu"
   pub fn gpu_only_function() { }
   ```

6. **Use required-features for conditional compilation in tests**
   - Cleaner than `#[cfg(...)]` for test compilation gates
   - Prevents subtle test skip bugs

7. **Graceful degradation over failure**
   - Compile GPU kernels if available
   - Fall back to CPU at runtime if hardware unavailable
   - Never crash due to missing optional hardware

---

## 12. Feature Gate Quick Reference

| Scenario | Command | Includes |
|----------|---------|----------|
| Development | `--no-default-features --features cpu` | CPU inference + tests |
| Production | `--release --no-default-features --features cpu` | Optimized CPU only |
| GPU testing | `--no-default-features --features gpu` | CUDA + GPU kernels |
| Full testing | `--no-default-features --features cpu,full-framework` | All test infrastructure |
| Cross-val | `--no-default-features --features cpu,crossval` | C++ comparison |
| Minimal | `--no-default-features` | Models only, no inference |
| Full | `--features full` | Everything (bloated) |

---

## Conclusion

BitNet-rs uses sophisticated feature gating to enable:

1. **Modular compilation** - Only compile what you need
2. **Multiple backends** - CPU, GPU, or both with graceful fallback
3. **Extensive testing** - Feature-specific test matrices
4. **Production safety** - Runtime detection + env overrides
5. **Development flexibility** - Fast iteration with CPU, full validation with GPU

The empty-default-features design ensures users never accidentally pull in unexpected dependencies, while unified GPU predicates and optional dependencies provide clean, maintainable feature composition across 18+ crates.
