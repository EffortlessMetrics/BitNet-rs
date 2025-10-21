# BitNet.rs Configuration Pattern Inventory

**Generated**: 2025-10-21
**Analysis Scope**: Comprehensive cfg pattern scan across crates/ directory (1,672 total patterns)
**Focus**: Feature flags, test enablement, architecture-specific code

## Executive Summary

This inventory documents all conditional compilation (`#[cfg(...)]`) patterns in the BitNet.rs codebase, enabling visibility into:
- Which tests are gated by feature flags
- Infrastructure-gated tests (GPU, environment variables, network)
- Architecture-specific code paths
- Feature flag dependencies and interactions

**Key Finding**: The codebase uses **653 unique cfg patterns** (281 unique pattern prefixes) across **17 crates in crates/ directory** (20 workspace members total including root, crossval, tests), with careful attention to feature gate consistency (especially GPU/CUDA unification).

**Note on Scope**: This analysis covers the `crates/` directory only. The full workspace includes additional crates in the root, crossval/, and tests/ directories not counted in these statistics.

---

## 1. Summary Statistics

### Overall Patterns
- **Total cfg patterns**: 1,672 instances (in crates/ directory)
- **Unique complete patterns**: 653 distinct combinations
- **Unique pattern prefixes**: 281 distinct prefixes (ignoring feature/type values)
- **Top 5 patterns** (by frequency):
  - `#[cfg(feature = "cpu")]`: 387 instances (23.1%)
  - `#[cfg(test)]`: 211 instances (12.6%)
  - `#[cfg(feature = "gpu")]`: 146 instances (8.7%)
  - `#[cfg(any(feature = "gpu", feature = "cuda"))]`: 87 instances (5.2%)
  - `#[cfg(feature = "inference")]`: 72 instances (4.3%)

### Pattern Categories

| Category | Count | Examples |
|----------|-------|----------|
| Feature flags | 1,150 | `cpu`, `gpu`, `inference`, `crossval`, `ffi` |
| Test conditional | 211 | `#[cfg(test)]` |
| Architecture | 140 | `x86_64`, `aarch64`, `wasm32` |
| Build mode | 12 | `debug_assertions` |
| Platform OS | 7 | `target_os = "linux"`, `target_os = "macos"` |
| Combinations (all/any/not) | 160 | Complex predicates |

### Distribution by Crate

| Crate | Count | Primary Focus |
|-------|-------|---------------|
| bitnet-inference | 347 | Test fixtures, GPU/CPU dispatch |
| bitnet-tokenizers | 298 | Model detection, downloads |
| bitnet-kernels | 260 | GPU/CPU kernels, device-aware |
| bitnet-quantization | 233 | Device quantizers, accuracy |
| bitnet-server | 195 | Feature-gated endpoints |
| bitnet-models | 176 | Loader variants, format detection |
| bitnet-cli | 64 | Full-cli feature, inference modes |
| bitnet-common | 27 | Device config, diagnostics |
| bitnet-sys | 21 | FFI bindings, CUDA detection |
| bitnet-ggml-ffi | 18 | Alternative implementations |
| bitnet-wasm | 15 | Browser/Node.js targets |
| bitnet-ffi | 14 | C++ bridge, interop |
| bitnet-compat | 4 | Metadata fixing |
| bitnet-py | 4 | Python bindings |
| bitnet-st2gguf | 4 | SafeTensors conversion |
| bitnet-st-tools | 0 | Utilities only |

---

## 2. Feature Flag Enablement Map

### Primary Feature Flags

#### `feature = "cpu"` (387 instances)
Primary CPU inference features. Required for any non-GPU build.

**Enables**:
- CPU-only kernels and fallbacks
- SIMD optimizations (AVX2, AVX-512, NEON)
- TDD test scaffolding for CPU quantization
- Mock elimination tests for ARM/x86

**Crates**:
- bitnet-kernels: 125 instances
- bitnet-inference: 89 instances
- bitnet-quantization: 78 instances
- bitnet-models: 52 instances
- Others: 43 instances

**Key Tests** (Sample):
- `/crates/bitnet-kernels/tests/issue_260_feature_gated_tests.rs`: CPU feature gate tests
- `/crates/bitnet-quantization/tests/device_aware_quantization.rs`: Device-aware quantizers
- `/crates/bitnet-inference/tests/ac1_quantized_linear_layers.rs`: Linear layer tests

---

#### `feature = "gpu"` (146 instances)
GPU acceleration via CUDA. Can be used alone or with CPU.

**Enables**:
- CUDA kernel compilation
- GPU memory management
- Mixed precision inference
- GPU-specific validation
- CUDA testing infrastructure

**Also Enabled By**: `feature = "cuda"` (backward-compatible alias)

**Unified Predicate Pattern**:
```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]  // 87 instances
```

**Crates**:
- bitnet-inference: 76 instances
- bitnet-kernels: 68 instances  
- bitnet-quantization: 32 instances
- bitnet-server: 28 instances
- Others: 42 instances

**Key Tests** (Sample):
- `/crates/bitnet-kernels/tests/gpu_integration.rs`: GPU device integration
- `/crates/bitnet-kernels/tests/mixed_precision_gpu_kernels.rs`: Mixed precision (8 instances gated)
- `/crates/bitnet-server/tests/device_config_test.rs`: Device selection

---

#### `feature = "inference"` (72 instances)
Inference engine and autoregressive generation.

**Enables**:
- Full production inference pipeline
- Autoregressive token generation
- Prompt template formatting
- Streaming response support

**Often Combined With**:
- `feature = "inference" + feature = "crossval"` (18 instances)
- `feature = "inference" + feature = "gpu"` (3+ instances)

**Crates**:
- bitnet-inference: 48 instances
- bitnet-cli: 14 instances
- bitnet-models: 8 instances
- Others: 2 instances

---

#### `feature = "crossval"` (50 instances)
Cross-validation against C++ reference implementation.

**Enables**:
- Parity testing with BitNet.cpp
- Accuracy validation tests
- Reference baseline comparisons
- FFI bridge testing

**Environment Gate**: `BITNET_CPP_DIR` (must be set)

**Crates**:
- bitnet-quantization: 22 instances
- bitnet-inference: 16 instances
- bitnet-kernels: 8 instances
- bitnet-models: 4 instances

**Key Tests**:
- `/crates/bitnet-quantization/tests/gpu_parity.rs`: GPU parity testing
- `/crates/bitnet-inference/tests/ac4_cross_validation_accuracy.rs`: Accuracy validation

---

#### `feature = "ffi"` (29 instances)
C++ FFI bridge and interoperability.

**Enables**:
- C++ function bindings
- Alternative quantization implementations
- FFI-based quantization kernels
- Build-time C++ integration

**Requires At Build Time**: `have_cpp` (detected via build script)

**Safe Mode**: Uses `#[cfg(any(not(feature = "ffi"), not(have_cpp)))]` (8 instances) to provide fallbacks

**Crates**:
- bitnet-kernels: 14 instances
- bitnet-quantization: 10 instances
- bitnet-inference: 3 instances
- Others: 2 instances

**Key Tests**:
- `/crates/bitnet-kernels/tests/issue_260_feature_gated_tests.rs`: FFI availability tests (4 instances)
- `/crates/bitnet-quantization/tests/fixture_integration_test.rs`: Fixture tests with FFI

---

#### `feature = "opentelemetry"` (32 instances)
OTEL observability and metrics.

**Enables**:
- OTEL tracing
- Performance metrics collection
- Event span recording
- Async runtime integration

**Crates**:
- bitnet-server: 20 instances
- bitnet-inference: 8 instances
- bitnet-quantization: 4 instances

---

#### `feature = "spm"` (46 instances)
SentencePiece tokenizer support.

**Enables**:
- SentencePiece tokenizer backend
- Byte-pair encoding (BPE) fallback
- Multilingual tokenization
- Alternative to GGUF-embedded tokenizers

**Requires At Build Time**: `sentencepiece` C++ library

**Tests**:
- `/crates/bitnet-tokenizers/tests/sp_roundtrip.rs`: SPM round-trip testing

---

#### Other Primary Features

| Feature | Instances | Purpose |
|---------|-----------|---------|
| `full-cli` | 23 | Full CLI with all subcommands |
| `full-engine` | 15 | Full inference engine (vs mocked) |
| `iq2s-ffi` | 26 | IQ2_S via FFI (GGML format) |
| `prometheus` | 9 | Prometheus metrics (deprecated) |
| `rt-tokio` | 5 | Tokio async runtime |
| `downloads` | 5 | Model downloading capability |
| `cli-bench` | 5 | Benchmarking CLI utilities |
| `integration-tests` | 1 | Full integration test suite |

---

### Architecture-Specific Gates

#### `target_arch = "x86_64"` (65 instances)
Intel/AMD 64-bit architecture.

**Enables**:
- AVX2 SIMD kernels
- AVX-512 kernels
- x86-specific optimizations
- 64-bit addressing mode

**Patterns**:
- `#[cfg(target_arch = "x86_64")]`: 65 instances (feature-independent)
- `#[cfg(all(target_arch = "x86_64", feature = "avx2"))]`: 14 instances
- `#[cfg(all(target_arch = "x86_64", feature = "avx512"))]`: 1 instance

**Crates**:
- bitnet-kernels: 38 instances
- bitnet-models: 12 instances
- bitnet-quantization: 10 instances
- Others: 5 instances

**Key Files**:
- `/crates/bitnet-kernels/src/cpu/x86.rs`: x86 SIMD implementations
- `/crates/bitnet-models/src/quant/i2s_qk256_avx2.rs`: AVX2-specific quantization

---

#### `target_arch = "aarch64"` (45 instances)
ARM 64-bit architecture (Apple M1/M2, Raspberry Pi 5, AWS Graviton).

**Enables**:
- NEON SIMD vectorization
- ARM-specific optimizations
- Thumb-2 instruction set
- 32-bit exception handling

**Patterns**:
- `#[cfg(target_arch = "aarch64")]`: 45 instances
- `#[cfg(all(target_arch = "aarch64", feature = "neon"))]`: 13 instances

**Crates**:
- bitnet-kernels: 28 instances
- bitnet-models: 9 instances
- bitnet-quantization: 6 instances
- Others: 2 instances

**Key Files**:
- `/crates/bitnet-kernels/src/cpu/arm.rs`: ARM NEON implementations

---

#### `target_arch = "wasm32"` (3 instances)
WebAssembly targets (browser/Node.js).

**Tests**:
- `/crates/bitnet-kernels/tests/issue_260_feature_gated_tests.rs`: WASM unavailability (2 instances)

---

#### Fallback Patterns
- `#[cfg(not(target_arch = "x86_64"))]`: 14 instances (generic fallback for non-x86)
- `#[cfg(not(target_arch = "aarch64"))]`: 11 instances (generic fallback for non-ARM)
- `#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]`: 8 instances (portable fallback)

---

### Complex Feature Combinations

#### GPU + CPU Combinations
- `#[cfg(all(feature = "cpu", feature = "gpu"))]`: 12 instances
  - Tests requiring both backends
  - Device fallback scenarios
- `#[cfg(all(feature = "cpu", not(feature = "gpu")))]`: 1 instance
  - CPU-only validation
- `#[cfg(all(feature = "gpu", not(feature = "cpu")))]`: 1 instance
  - GPU-only configurations

#### Inference Combinations
- `#[cfg(all(feature = "inference", feature = "crossval"))]`: 18 instances
  - Parity validation tests
  - Accuracy baseline comparisons
- `#[cfg(all(feature = "inference", feature = "gpu"))]`: 3+ instances
  - GPU inference pipeline tests

#### Special Combinations
- `#[cfg(all(test, feature = "crossval"))]`: 4 instances
  - Test-only cross-validation
- `#[cfg(all(test, feature = "gpu"))]`: 1 instance
  - GPU-specific test infrastructure
- `#[cfg(all(feature = "gpu", not(feature = "strict")))]`: 8 instances
  - Relaxed GPU validation in non-strict mode

---

## 3. Test Coverage Matrix

### CPU-Only Tests (387 instances with `feature = "cpu"`)

**Enabled By**: `cargo test --no-default-features --features cpu`

**Test Categories**:

| Category | Crate | Example Files | Count |
|----------|-------|--------------|-------|
| Quantization kernels | bitnet-kernels | `cpu_simd_receipts.rs`, `comprehensive_kernel_unit_tests.rs` | ~50 |
| Linear layers | bitnet-quantization | `device_aware_quantization.rs`, `i2s_property_fuzz_tests.rs` | ~40 |
| Model loading | bitnet-models | `qk256_avx2_correctness.rs`, `i2s_dequant.rs` | ~30 |
| Inference engine | bitnet-inference | `real_inference_engine.rs`, `issue_254_*.rs` tests | ~35 |
| CLI operations | bitnet-cli | `cli_smoke.rs`, `qa_greedy_math_confidence.rs` | ~20 |

**Sample Test Files**:
- `/crates/bitnet-kernels/tests/comprehensive_kernel_unit_tests.rs`: CPU SIMD tests with platform dispatch
- `/crates/bitnet-inference/tests/ac1_quantized_linear_layers.rs`: Quantized linear operations
- `/crates/bitnet-models/tests/qk256_avx2_correctness.rs`: QK256 AVX2 verification

---

### GPU Tests (146 instances with `feature = "gpu"`)

**Enabled By**: `cargo test --no-default-features --features gpu`

**Infrastructure Gate**: GPU hardware required at runtime

**Key Tests**:
- `/crates/bitnet-kernels/tests/gpu_integration.rs`: GPU device integration
- `/crates/bitnet-kernels/tests/mixed_precision_gpu_kernels.rs`: Mixed precision (8 feature gates)
- `/crates/bitnet-kernels/tests/gpu_real_compute.rs`: Real CUDA computation
- `/crates/bitnet-inference/tests/ac9_comprehensive_integration_testing.rs`: Full integration

**Skipped If**:
- GPU not detected at runtime (graceful CPU fallback)
- `BITNET_GPU_FAKE=none` (disable GPU for testing)
- CUDA not available in build environment

---

### Cross-Validation Tests (50 instances with `feature = "crossval"`)

**Enabled By**: 
```bash
export BITNET_CPP_DIR=/path/to/bitnet.cpp
export CROSSVAL_GGUF=/path/to/model.gguf
cargo test --no-default-features --features crossval
```

**Infrastructure Gates**:
- `BITNET_CPP_DIR` must point to valid BitNet.cpp build
- `CROSSVAL_GGUF` or `BITNET_GGUF` environment variable
- C++ reference implementation compiled

**Key Tests**:
- `/crates/bitnet-quantization/tests/gpu_parity.rs`: GPU vs C++ parity
- `/crates/bitnet-inference/tests/ac4_cross_validation_accuracy.rs`: Accuracy validation
- `/crates/bitnet-quantization/tests/fixture_integration_test.rs`: Fixture comparison

---

### Environment-Variable-Gated Tests (50+ instances)

| Variable | Count | Gate Type | Purpose |
|----------|-------|-----------|---------|
| `BITNET_GGUF` | 19 | Model path | Real model inference tests |
| `BITNET_STRICT_MODE` | 35 | Validation | Strict model validation |
| `BITNET_DETERMINISTIC` | 17 | Reproducibility | Reproducible inference |
| `BITNET_SEED` | 16 | RNG | Seeded randomness |
| `BITNET_CPP_DIR` | 5 | Cross-validation | C++ reference integration |
| `CROSSVAL_GGUF` | 8 | Model path | Cross-validation model |
| `BITNET_DEVICE` | 5 | Device selection | Explicit device choice |
| `RAYON_NUM_THREADS` | 8 | Parallelism | Single-threaded testing |
| `BITNET_OFFLINE` | 6 | Network | Offline test runs |

**Usage Pattern**:
```rust
#[test]
#[ignore]  // Infrastructure-gated: requires BITNET_GGUF
fn test_real_model_inference() {
    let model_path = std::env::var("BITNET_GGUF")
        .expect("Set BITNET_GGUF to model path");
    // ...
}
```

---

### FFI-Gated Tests (29 instances)

**Feature**: `feature = "ffi"`

**Environment**: `BITNET_CPP_DIR` at build time (sets `have_cpp`)

**Tests**:
- `/crates/bitnet-kernels/tests/issue_260_feature_gated_tests.rs`: FFI availability (4 tests)
- `/crates/bitnet-quantization/tests/fixture_integration_test.rs`: FFI quantization

**Fallback Safety** (6 instances):
```rust
#[cfg(all(feature = "ffi", have_cpp))]
fn use_cpp_implementation() { /* ... */ }

#[cfg(any(not(feature = "ffi"), not(have_cpp)))]
fn use_rust_fallback() { /* ... */ }
```

---

### Architecture-Specific Tests

#### x86_64 Tests (65 instances)
- `#[cfg(all(target_arch = "x86_64", feature = "avx2"))]`: 14 tests
- `/crates/bitnet-kernels/tests/comprehensive_kernel_unit_tests.rs`: AVX2 correctness
- `/crates/bitnet-models/tests/qk256_avx2_correctness.rs`: QK256 AVX2 verification

#### ARM Tests (45 instances)
- `#[cfg(all(target_arch = "aarch64", feature = "neon"))]`: 13 tests
- `/crates/bitnet-kernels/tests/comprehensive_kernel_unit_tests.rs`: NEON tests

#### WASM Tests (3 instances)
- Tests verify WASM target unavailability
- Checked via `#[cfg(target_arch = "wasm32")]`

---

## 4. Feature Flag Dependency Graph

```
                          [Default: Empty]
                                 |
                  ________________|_________________
                 |                |                |
            [cpu]            [gpu/cuda]      [crossval]
              |                  |                |
           __|__                 |         _______|_______
          |     |                |        |       |       |
    [avx2] [neon]            [mixed-  [inference][ffi] [models]
                              precision]
              |                 |
         [TL1/TL2]         [kernels]

Additional: spm, opentelemetry, prometheus, rt-tokio, downloads, etc.
```

### Mutual Exclusivity
- `cpu` and `gpu` can coexist (fallback behavior)
- `crossval` requires either `cpu` or `gpu` (but tests only with one at a time)

### Strict Dependencies
- `inference` often used with `cpu` or `gpu`
- `crossval` requires `BITNET_CPP_DIR` env var
- `ffi` requires C++ build-time detection (`have_cpp`)

---

## 5. Recommendations

### Documentation Needs

1. **Feature Flag Documentation**
   - Create `/docs/explanation/feature-flags-guide.md`
   - Map each feature to its test coverage
   - Explain mutual compatibility

2. **Test Execution Guide**
   - Document how to run feature-gated tests
   - Environment variable setup for infrastructure-gated tests
   - GPU hardware requirements

3. **CI/CD Integration**
   - Add test matrix for different feature combinations
   - Document cross-validation setup
   - GPU-aware CI configuration

### Code Simplification Opportunities

1. **Unified GPU Predicate** ✅ (Already implemented)
   - Pattern `#[cfg(any(feature = "gpu", feature = "cuda"))]`: 87 instances
   - Good standardization across codebase

2. **Simplify Complex Combinations**
   - 16 instances of `#[cfg(all(feature = "inference", any(...)))]` could be simplified
   - Extract common patterns to shared compile-time conditions

3. **Architecture-Specific Code**
   - x86_64/aarch64 gates could benefit from trait-based dispatch
   - SIMD selection currently uses compile-time gates, could use runtime dispatch for portability

### Missing Coverage

1. **RISC-V Support**
   - No `#[cfg(target_arch = "riscv")]` gates exist
   - Recommended for future embedded support

2. **Explicit Windows Support**
   - Only 1 `#[cfg(target_os = "windows")]` instance
   - Consider platform-specific GPU/threading code

3. **Test Variance Documentation**
   - No explicit gates for noise/variance in timing tests
   - Recommend `#[cfg(not(release_or_higher_optimization))]` pattern

### Pattern Consolidation

**Current scattered patterns**:
```rust
#[cfg(feature = "gpu")]
#[cfg(feature = "cuda")]
#[cfg(any(feature = "gpu", feature = "cuda"))]
```

**Recommendation**: Standardize new code to always use `#[cfg(any(feature = "gpu", feature = "cuda"))]` for visibility.

---

## 6. Feature-Gated Test Execution Quick Reference

### CPU-Only Build
```bash
cargo test --no-default-features --features cpu
# Runs: 387 CPU-gated tests, skips 146 GPU-only, 50 crossval
```

### GPU Build (with CPU fallback)
```bash
cargo test --no-default-features --features gpu,cpu
# Runs: All 387 CPU + 146 GPU tests, skips 50 crossval
```

### Full Validation
```bash
export BITNET_CPP_DIR=/path/to/bitnet.cpp
export CROSSVAL_GGUF=/path/to/model.gguf
cargo test --no-default-features --features cpu,gpu,crossval,ffi
# Runs: CPU + GPU + 50 crossval tests
```

### Inference-Only Build
```bash
cargo test --no-default-features --features cpu,inference
# Runs: CPU + inference pipeline tests only
```

### Strict Validation
```bash
BITNET_STRICT_MODE=1 cargo test --no-default-features --features cpu
# Runs with strict validation gates (35 additional strict tests)
```

### Deterministic Testing
```bash
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
  cargo test --no-default-features --features cpu -- --test-threads=1
# Ensures reproducibility for 17 deterministic tests
```

---

## 7. Configuration Details by Crate

### bitnet-kernels (260 instances)
- **CPU**: 125 instances - SIMD kernel selection
- **GPU**: 68 instances - CUDA kernel compilation
- **Architecture**: 40 instances - x86/ARM specific paths
- **Focus**: Device-aware kernel manager

### bitnet-inference (347 instances)
- **CPU**: 89 instances - CPU inference pipeline
- **GPU**: 76 instances - GPU inference, mixed precision
- **Test infrastructure**: 67 instances - Mock elimination, fixtures
- **Focus**: Production inference engine

### bitnet-tokenizers (298 instances)
- **Feature detection**: 67 instances - Model format detection
- **Downloads**: 45 instances - Network-based tokenizer fetching
- **Fallbacks**: 52 instances - Error handling alternatives
- **SPM support**: 46 instances - SentencePiece integration

### bitnet-quantization (233 instances)
- **Device-aware**: 78 instances - Device-specific quantizers
- **GPU**: 32 instances - GPU quantization kernels
- **Accuracy**: 44 instances - Validation gates
- **Focus**: Quantization format compatibility

### bitnet-server (195 instances)
- **OTEL integration**: 20 instances - Metrics/tracing
- **GPU support**: 28 instances - Device management
- **Endpoint gating**: 31 instances - Feature-specific REST APIs
- **Focus**: Production REST service

### bitnet-models (176 instances)
- **Format detection**: 52 instances - GGUF/SafeTensors variants
- **Architecture**: 21 instances - Quantization flavor detection
- **Loader variants**: 48 instances - Device-aware loading
- **Focus**: Model I/O and format compatibility

---

## Appendix: Full Pattern Reference

### All Unique Patterns (Top 50)

```
387  #[cfg(feature = "cpu")]
211  #[cfg(test)]
146  #[cfg(feature = "gpu")]
 87  #[cfg(any(feature = "gpu", feature = "cuda"))]
 72  #[cfg(feature = "inference")]
 65  #[cfg(target_arch = "x86_64")]
 50  #[cfg(feature = "crossval")]
 46  #[cfg(feature = "spm")]
 45  #[cfg(target_arch = "aarch64")]
 33  #[cfg(not(feature = "gpu"))]
 32  #[cfg(feature = "opentelemetry")]
 31  #[cfg(not(any(feature = "gpu", feature = "cuda")))]
 29  #[cfg(feature = "ffi")]
 26  #[cfg(feature = "iq2s-ffi")]
 23  #[cfg(feature = "full-cli")]
 18  #[cfg(all(feature = "inference", feature = "crossval"))]
 17  #[cfg(not(feature = "iq2s-ffi"))]
 16  #[cfg(all(feature = "inference", any(...)))]
 15  #[cfg(feature = "full-engine")]
 14  #[cfg(not(target_arch = "x86_64"))]
 14  #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
 13  #[cfg(not(feature = "spm"))]
 13  #[cfg(all(target_arch = "aarch64", feature = "neon"))]
 12  #[cfg(debug_assertions)]
 12  #[cfg(all(feature = "cpu", feature = "gpu"))]
 11  #[cfg(not(target_arch = "aarch64"))]
  9  #[cfg(not(feature = "inference"))]
  9  #[cfg(feature = "prometheus")]
  8  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  8  #[cfg(all(feature = "gpu", not(feature = "strict")))]
  7  #[cfg(feature = "degraded-ok")]
  7  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  7  #[cfg(all(feature = "simd", target_arch = "x86_64"))]
  7  #[cfg(all(feature = "simd", target_arch = "aarch64"))]
  6  #[cfg(not(feature = "degraded-ok"))]
  6  #[cfg(all(target_os = "macos", feature = "gpu"))]
  6  #[cfg(all(feature = "cpu", feature = "crossval"))]
  5  #[cfg(not(all(target_os = "macos", feature = "gpu")))]
  5  #[cfg(feature = "rt-tokio")]
  5  #[cfg(feature = "downloads")]
  5  #[cfg(feature = "cli-bench")]
  4  #[cfg(target_os = "linux")]
  4  #[cfg(all(test, feature = "crossval"))]
  4  #[cfg(all(debug_assertions, feature = "cpu"))]
  3  #[cfg(unix)]
  3  #[cfg(target_arch = "wasm32")]
  3  #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
  3  #[cfg(feature = "simd")]
  3  #[cfg(any(feature = "cpu", feature = "gpu"))]
  3  #[cfg(all(feature = "inference", feature = "gpu"))]
```

---

## Document Metadata

- **Analysis Date**: 2025-10-21
- **Codebase Revision**: `c5a1c45b` (feat/mvp-finalization)
- **Total Files Scanned**: 250+ Rust files across 17 crates (crates/ directory only)
- **Pattern Detection Tool**: ripgrep with Bash/Python analysis
- **Completeness**: Comprehensive for crates/ directory (excludes root, crossval/, tests/ workspace members)
- **Workspace Context**: 20 total workspace members (17 in crates/, plus root, crossval, tests)

