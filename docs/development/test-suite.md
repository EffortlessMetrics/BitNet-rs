# Test Suite Guide

This document covers the comprehensive test suite for BitNet.rs, including running tests, configuration, and specialized testing strategies.

## Test Status Summary

**Current Test Results**:
- **Total Enabled Tests**: 3,472 (all pass)
- **Passing Tests**: 3,472 (100%)
- **Properly Skipped Tests**: 462 (intentional: ignored, integration, fixtures)
- **Execution Time**: ~118 seconds (with parallel execution)

**Test Infrastructure Status**:
- ✅ **Receipt Verification**: 25/25 tests passing (schema v1.0.0)
- ✅ **Strict Mode Guards**: 12/12 tests passing (runtime enforcement)
- ✅ **Environment Isolation**: 7/7 tests passing (EnvGuard parallel safety)
- ✅ **GGUF Fixtures**: 12/12 tests passing (QK256 dual-flavor detection)
- ✅ **Snapshot Tests**: 42 test files across the workspace (insta)
- ✅ **Property Tests**: 30 test files across all 30 proptest crates (proptest)
- ✅ **Fuzz Targets**: 13 targets, nightly scheduled (cargo-fuzz)
- ✅ **CPU Golden Path E2E**: deterministic end-to-end inference test

## Running Tests

### Standard Test Execution with cargo test

```bash
# Run all enabled tests with CPU features
cargo test --workspace --no-default-features --features cpu

# Run specific test crates
cargo test -p bitnet-inference --no-default-features --features cpu
cargo test -p bitnet-quantization --no-default-features --features cpu
cargo test -p bitnet-models --no-default-features --features cpu

# Run with GPU features
cargo test --workspace --no-default-features --features gpu

# Skip slow tests (QK256 scalar kernels)
BITNET_SKIP_SLOW_TESTS=1 cargo test --workspace --no-default-features --features cpu

# Run including ignored tests (will encounter blocked tests)
cargo test --workspace --no-default-features --features cpu -- --ignored --include-ignored
```

### Using cargo nextest (Recommended for CI)

Nextest provides timeout protection, clean output, and better diagnostics for the BitNet.rs test suite.

```bash
# Install nextest if needed
cargo install cargo-nextest

# Run all tests with default profile (5-minute timeout, clean output)
cargo nextest run --workspace --no-default-features --features cpu

# Run with CI profile (4 fixed threads, no retries, optimized for CI)
cargo nextest run --profile ci --workspace --no-default-features --features cpu

# Run specific crate
cargo nextest run -p bitnet-inference --no-default-features --features cpu

# Skip slow tests
BITNET_SKIP_SLOW_TESTS=1 cargo nextest run --workspace --no-default-features --features cpu

# Generate JUnit XML report (available at target/nextest/junit.xml)
cargo nextest run --workspace --no-default-features --features cpu
```

**Nextest Configuration**: See `.config/nextest.toml` for profiles, timeout settings, and output options.

**Nextest Benefits:**
- **Global timeout**: 5-minute safety net prevents test hangs
- **Fail-fast**: Immediate failure reporting without waiting for all tests
- **Clean output**: Suppresses success output, shows only failures
- **No retries**: `retries = 0` ensures reproducible test results (no flaky test masking)
- **JUnit reports**: Automatic XML export for CI/CD integration
- **Per-test isolation**: Configurable thread count for parallel execution

### Fixture Management

BitNet.rs uses a structured fixture management system for test data. GGUF fixtures are stored in `ci/fixtures/` and provide deterministic test inputs for quantization and model loading tests.

#### Available Fixtures

**Location**: `/home/steven/code/Rust/BitNet-rs/ci/fixtures/qk256/`

**QK256 Fixtures** (QK256 quantization format - 256-element blocks):
- `qk256_4x256.gguf` - 4×256 tensor block (aligned)
- `qk256_3x300.gguf` - 3×300 tensor block (misaligned)
- `bitnet32_2x64.gguf` - 2×64 tensor block (BitNet32 format)

**SHA256 Validation**: `SHA256SUMS` file provides integrity verification

#### Running Fixture-Based Tests

```bash
# Run GGUF fixture tests with fixtures feature
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features cpu,fixtures

# Run all fixture-based integration tests
cargo test --workspace --no-default-features --features "cpu,fixtures"

# Run with fixture validation enabled
BITNET_FIXTURE_VALIDATE=1 cargo test --no-default-features --features "cpu,fixtures"
```

#### Fixture Test Categories

1. **Dual-Flavor Detection** (12 tests passing):
   - QK256 format detection with automatic fallback
   - Tensor size matching and block alignment validation
   - I2_S vs QK256 flavor selection logic

2. **Alignment Validation**:
   - 256-element block boundary checking
   - Quantized tensor dimension validation
   - Scale factor alignment verification

3. **Numerical Correctness**:
   - Dequantization accuracy across fixtures
   - Cross-flavor result comparison (QK256 vs I2_S when applicable)

#### Creating New Fixtures

For new quantization format testing:

```bash
# 1. Create minimal GGUF file with desired tensor sizes
# 2. Add to ci/fixtures/qk256/ directory
# 3. Generate SHA256 hash
sha256sum new_fixture.gguf >> ci/fixtures/qk256/SHA256SUMS

# 4. Validate in tests
BITNET_GGUF=ci/fixtures/qk256/new_fixture.gguf cargo test \
  --no-default-features --features "cpu,fixtures"
```

### Convolution Tests

```bash
# Run convolution unit tests
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu convolution

# Run PyTorch reference convolution tests (requires Python and PyTorch)
cargo test --no-default-features -p bitnet-kernels conv2d_reference_cases --no-default-features --features cpu -- --ignored

# Test specific convolution functionality
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_conv2d_basic_functionality
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_conv2d_with_bias
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_conv2d_stride
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_conv2d_padding
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_conv2d_dilation

# Test quantized convolution
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_conv2d_quantized_i2s
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_conv2d_quantized_tl1
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_conv2d_quantized_with_bias
```

### GPU-Specific Tests

```bash
# GPU smoke tests (basic availability, run on CI with GPU)
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu --test gpu_smoke

# GPU integration tests (comprehensive, manual execution)
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu --test gpu_quantization --ignored

# GPU performance tests (benchmarking, development only)
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_gpu_performance --ignored

# GPU vs CPU quantization accuracy
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_gpu_vs_cpu_quantization_accuracy --ignored

# GPU fallback mechanism testing
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_gpu_quantization_fallback --ignored

# GPU memory management and leak detection
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_gpu_memory_management

# CUDA device information and memory tracking
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_cuda_device_info_query
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_device_memory_tracking
```

### Memory Tracking Tests

```bash
# Basic CPU memory tracking tests
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_memory_tracking
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_performance_tracking

# Comprehensive memory tracking with device awareness
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_memory_tracking_comprehensive
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_memory_efficiency_tracking

# GPU memory tracking tests (requires CUDA)
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_device_memory_tracking
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_gpu_memory_management

# Memory tracking integration with device-aware quantization
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_device_aware_quantizer_memory_stats
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_cuda_quantizer_memory_integration

# Host memory vs system memory validation
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_host_vs_system_memory_tracking

# Thread-safe memory statistics access
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_concurrent_memory_stats_access
```

### Cross-Validation Tests

```bash
# Cross-validation testing (requires C++ dependencies)
cargo test --no-default-features --workspace --no-default-features --features "cpu,ffi,crossval"

# Full cross-validation workflow
cargo run -p xtask -- full-crossval

# Cross-validation with concurrency caps
scripts/preflight.sh && cargo crossval-capped
```

## Test Configuration

The test suite uses a feature-gated configuration system:

- **`fixtures`**: Enables fixture management and test data generation
- **`reporting`**: Enables test reporting (JSON, HTML, Markdown, JUnit)
- **`trend`**: Enables trend analysis and performance tracking
- **`integration-tests`**: Enables full integration test suite

### Feature-Gated Tests and CI Configuration

BitNet.rs uses feature-gated architecture where **default features are EMPTY**. This means tests that depend on device-specific functionality (CPU/GPU) must be run with explicit feature flags:

```bash
# Correct: Tests run with required features
cargo test --no-default-features --features cpu

# Incorrect: Tests may fail without features
cargo test  # Will fail for device-dependent tests
```

#### Feature-Gated Test Behavior

Some tests validate feature-gated functionality and will behave differently based on enabled features:

- **With `--features cpu` or `--features gpu`**: Tests validate full functionality
- **Without features**: Tests validate graceful degradation (e.g., fixture selection returns `None`)

**Example tests with feature-aware assertions:**
- `test_fixture_selector_functionality` (crates/bitnet-server/tests/test_fixtures_integration.rs:197)
- `test_model_selection` (crates/bitnet-server/tests/fixtures/mod.rs:403)

These tests use `#[cfg(any(feature = "cpu", feature = "gpu"))]` guards to ensure correct behavior regardless of feature configuration.

#### CI Configuration Requirements

All CI workflows must use proper feature flags to ensure test stability:

```yaml
# Correct CI test configuration
- run: cargo test -p bitnet-server --all-targets --no-default-features --features cpu

# Incorrect CI configuration (may cause test failures)
- run: cargo test -p bitnet-server --all-targets
```

**CI Workflows with Required Feature Flags:**
- `.github/workflows/ci.yml`: Main test workflow (uses `--features cpu`)
- `.github/workflows/clippy-cli-server.yml`: Server-specific tests (updated to use `--features cpu`)
- `.github/workflows/testing-framework-unit.yml`: Unit test matrix

For more details on feature flags and build configuration, see [CLAUDE.md](../../CLAUDE.md) and [Feature Flags Documentation](../explanation/FEATURES.md).

## Test Features

- **Parallel Test Execution**: Configurable parallelism with resource limits
- **Fixture Management**: Automatic test data generation and caching
- **CI Integration**: JUnit output, exit codes, and CI-specific optimizations
- **Error Reporting**: Detailed error messages with recovery suggestions
- **Performance Tracking**: Benchmark results and regression detection
- **Mock Infrastructure**: Comprehensive mock model and tokenizer implementations for testing
- **Enhanced Performance Testing**: Structured metrics collection with prefill timing validation
- **Mutation Testing**: Enterprise-grade mutation testing with 80%+ kill rates for critical components

## Test Categories

BitNet.rs test suite is organized into distinct categories, each addressing specific aspects of the inference engine and quantization pipeline.

### Category Summary

| Category | Count | Status | Purpose |
|----------|-------|--------|---------|
| **Quantization Tests** | 180+ | ✅ Passing | I2_S flavor detection, TL1/TL2, IQ2_S via FFI |
| **Model Loading Tests** | 95+ | ✅ Passing | GGUF and SafeTensors parsing |
| **Fixture Tests** | 12 | ✅ Passing | QK256 dual-flavor detection, alignment validation |
| **Snapshot Tests** | 200+ | ✅ Passing | Struct/output stability (insta, 42 test files) |
| **Property Tests** | 100+ | ✅ Passing | Randomised invariants (proptest, 20 test files) |
| **Tokenizer Tests** | 110+ | ✅ Passing | Universal tokenizer, auto-discovery |
| **CLI Tests** | 140+ | ✅ Passing | Command-line parsing, flag validation |
| **Device Feature Tests** | 65+ | ✅ Passing | CPU/GPU compilation, feature guards |
| **Validation Tests** | 85+ | ✅ Passing | LayerNorm inspection, projection statistics |
| **Receipt Verification** | 25 | ✅ Passing | Schema v1.0.0 with 8 gates |
| **Strict Mode Tests** | 12 | ✅ Passing | Runtime guards and enforcement |
| **Environment Isolation** | 7 | ✅ Passing | EnvGuard parallel safety |
| **Performance Tests** | 95+ | ✅ Passing | Benchmarking, memory tracking |
| **Integration Tests** | 110+ | 🟡 Partial | End-to-end workflows (some blocked by issues) |
| **Slow/Ignored Tests** | 70+ | ⏸️ Skipped | QK256 scalar kernels, architecture blockers |

**Total Enabled**: 3,472 tests
**Total Skipped**: 462 tests (intentional)
**Pass Rate**: 100%

### Quantization Tests

Validates quantization algorithm implementation and flavor detection:

```bash
# Run all quantization tests
cargo test -p bitnet-quantization --no-default-features --features cpu

# Test specific quantization formats
cargo test -p bitnet-quantization --no-default-features --features cpu i2s
cargo test -p bitnet-quantization --no-default-features --features cpu tl1
cargo test -p bitnet-quantization --no-default-features --features cpu tl2

# Test QK256-specific functionality
cargo test -p bitnet-models --no-default-features --features cpu qk256
```

**Key Test Areas**:
- Flavor detection algorithm accuracy
- Block size and alignment validation
- Dequantization kernel correctness
- Scale factor computation
- Cross-format compatibility

### Model Loading Tests

Validates GGUF and SafeTensors parsing:

```bash
# Run model loading tests
cargo test -p bitnet-models --no-default-features --features cpu

# Test GGUF parsing
cargo test -p bitnet-models --no-default-features --features cpu gguf

# Test SafeTensors loading
cargo test -p bitnet-models --no-default-features --features cpu safetensors

# Test model validation
cargo test -p bitnet-models --no-default-features --features cpu validation
```

**Key Test Areas**:
- GGUF header parsing
- Tensor metadata extraction
- Model structure validation
- Device-aware tensor mapping
- Format compatibility detection

### Tokenizer Tests

Validates universal tokenizer architecture:

```bash
# Run tokenizer tests
cargo test -p bitnet-tokenizers --no-default-features --features cpu

# Test auto-discovery
cargo test -p bitnet-tokenizers --no-default-features --features cpu auto_discover

# Test builder pattern
cargo test -p bitnet-tokenizers --no-default-features --features cpu builder

# Test SentencePiece integration
cargo test -p bitnet-tokenizers --no-default-features --features cpu sentencepiece
```

**Key Test Areas**:
- Format auto-detection
- SentencePiece loading
- Token encoding/decoding
- Special token handling
- Vocab size validation

### CLI Tests

Validates command-line interface and flag parsing:

```bash
# Run all CLI tests
cargo test -p bitnet-cli --no-default-features --features cpu

# Test flag parsing
cargo test -p bitnet-cli --no-default-features --features cpu flags

# Test inference commands
cargo test -p bitnet-cli --no-default-features --features cpu inference

# Test output formatting
cargo test -p bitnet-cli --no-default-features --features cpu output
```

**Key Test Areas**:
- Argument parsing
- Feature flag validation
- Output formatting
- Error message clarity
- Interactive mode (chat)

### Device Feature Tests

Validates CPU/GPU feature compilation and detection:

```bash
# Run feature compilation tests
cargo test --workspace --no-default-features --features cpu device_features

# Test GPU detection
BITNET_GPU_FAKE=cuda cargo test --no-default-features --features gpu device

# Test fallback behavior
BITNET_GPU_FAKE=none cargo test --no-default-features --features gpu device
```

**Key Test Areas**:
- Feature gate consistency
- Device capability detection
- GPU/CPU kernel selection
- Fallback mechanism correctness
- Runtime device availability

### Validation Tests

Validates model inspection and LayerNorm statistics:

```bash
# Run validation tests
cargo test -p bitnet-cli --no-default-features --features cpu validate

# Test LayerNorm inspection
cargo test -p bitnet-cli --no-default-features --features cpu ln_stats

# Test strict mode validation
BITNET_STRICT_MODE=1 cargo test --no-default-features --features cpu validate

# Test validation policies
cargo test -p bitnet-cli --no-default-features --features cpu policy
```

**Key Test Areas**:
- LayerNorm RMS computation
- Projection statistics accuracy
- Weight distribution analysis
- Policy-driven corrections
- Strict mode enforcement

### Receipt Verification Tests (25 tests, 100% passing)

Validates inference receipt schema and compute path verification:

```bash
# Run all receipt verification tests
cargo test -p xtask --no-default-features --features cpu verify_receipt

# Test schema validation
cargo test -p xtask --no-default-features --features cpu schema

# Test compute path verification
cargo test -p xtask --no-default-features --features cpu compute_path

# Test kernel ID hygiene
cargo test -p xtask --no-default-features --features cpu kernel_id
```

**Key Test Areas**:
- Receipt schema v1.0.0 validation
- Compute path authenticity (real vs mock)
- Kernel ID legitimacy checking
- TPS measurement accuracy
- Auto-GPU enforcement

**See also**: [Receipt Verification Reference](../reference/validation-gates.md)

### Strict Mode Tests (12 tests, 100% passing)

Validates production safety enforcement:

```bash
# Run strict mode tests
BITNET_STRICT_MODE=1 cargo test --no-default-features --features cpu strict

# Test exit codes
BITNET_STRICT_MODE=1 cargo test --no-default-features --features cpu exit_code

# Test LayerNorm warnings
BITNET_STRICT_MODE=1 cargo test --no-default-features --features cpu ln_warnings
```

**Key Test Areas**:
- Suspicious weight detection
- Validation gate failures
- Exit code correctness (8 for strict violations)
- Error message clarity
- Feature compatibility checks

### Environment Isolation Tests (7 tests, 100% passing)

Validates EnvGuard and test isolation:

```bash
# Run environment isolation tests
cargo test --workspace --no-default-features --features cpu env_guard

# Run with serial execution
cargo test --workspace --no-default-features --features cpu -- --test-threads=1

# Verify no test pollution
cargo test --test env_isolation --no-default-features --features cpu
```

**Key Test Areas**:
- EnvGuard restoration correctness
- Panic-safe cleanup
- Mutex synchronization
- Process-level serialization
- No test pollution after execution

**See also**: [Test Isolation Guide](#environment-variable-testing)

### Performance Tests

Validates inference performance and resource usage:

```bash
# Run performance tests
cargo test -p bitnet-inference --no-default-features --features cpu perf

# Run memory tracking tests
cargo test -p bitnet-kernels --no-default-features --features cpu memory

# Run benchmarks
cargo bench --no-default-features --features cpu

# Test with metrics collection
cargo test -p bitnet-cli --no-default-features --features cpu metrics
```

**Key Test Areas**:
- Throughput measurement (tokens/second)
- Memory allocation tracking
- Cache efficiency validation
- Latency profiling
- Regression detection

## Testing Strategy

### Mutation Testing

BitNet.rs uses mutation testing to validate test suite effectiveness and ensure critical code paths are properly covered.

#### Recent Achievements (Issue #462)

| Component | Mutation Score | Mutants Killed | Status |
|-----------|---------------|----------------|--------|
| **TL LUT Helper** | **100%** | 6/6 | ✅ Enterprise-grade |
| **Receipt Validation** | **88%** | 14/16 | ✅ Enterprise-grade |
| **Overall (Issue #462)** | **91%** | 20/22 | ✅ Exceeds 80% threshold |

**TL LUT Helper (`bitnet_kernels::tl_lut`):**
- 100% mutation score (6/6 mutants killed)
- All boundary conditions and overflow checks validated
- Checked arithmetic paths fully exercised

**Receipt CPU Validation (`xtask::verify_receipt`):**
- 88% mutation score (14/16 mutants killed)
- Quantized kernel detection thoroughly tested
- Fallback pattern matching validated
- Silent CPU fallback detection confirmed

**Testing Commands:**
```bash
# Run mutation-tested components
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu tl_lut
cargo test --no-default-features -p xtask test_receipt_cpu_validation

# View mutation testing reports
cat ci/receipts/pr-0462/T3.5-mutation-testing-report.md
cat ci/receipts/pr-0462/generative-gate-mutation-check-run.md
```

**See also:** `ci/receipts/pr-0462/` for detailed mutation testing reports and analysis.

### Resolved Issues: Issue #260 - SIMD Kernel Integration ✅

Issue #260 has been successfully resolved with comprehensive SIMD kernel testing:

**Completed Tests (Now Enabled):**
- `test_cpu_simd_kernel_integration`: Validates SIMD throughput with real quantized computation
- `test_tl2_avx_optimization`: Validates AVX optimization speedup for TL2 lookup tables

**Running Issue #260 Tests:**
```bash
# Run resolved SIMD kernel tests
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_cpu_simd_kernel_integration
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_tl2_avx_optimization

# Run all quantization tests (includes SIMD validation)
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu
```

**Related Documentation:**
- See `docs/explanation/issue-260-mock-elimination-completion.md` for full completion details
- See `docs/explanation/issue-260-spec.md` for original technical specification

### Core Testing Framework
- **Unit tests**: Each crate has comprehensive tests
- **Integration tests**: Cross-crate tests in `tests/`
- **Snapshot tests**: Struct/output stability assertions (insta, 42 test files, ~160 assertions, 192 snapshot files)
- **Property-based tests**: Randomised invariant checks (proptest, 20 test files, 100+ properties)
- **Fuzz Targets**: Parser and kernel robustness (cargo-fuzz, 13 targets, nightly scheduled)
- **Cross-validation**: Automated testing against C++ implementation
- **CI gates**: Compatibility tests block on every PR
- **SIMD Kernel Tests** ✅: Real quantization computation validation (Issue #260 resolved)

### Snapshot Tests (insta)

BitNet.rs uses [insta](https://insta.rs) for snapshot testing across all crates. Snapshots pin the human-readable serialization of structs and public API outputs, making unintended behavioural changes visible as CI failures.

**Running snapshot tests:**

```bash
# Run all snapshot tests
cargo nextest run --workspace --no-default-features --features cpu snapshot

# Review and accept new/changed snapshots interactively
cargo insta review

# Update all snapshots non-interactively (after intentional changes)
INSTA_UPDATE=always cargo nextest run --workspace --no-default-features --features cpu snapshot

# Run snapshot tests for a specific crate
cargo nextest run -p bitnet-common --no-default-features --features cpu snapshot
cargo nextest run -p bitnet-receipts --no-default-features --features cpu snapshot
```

**Snapshot locations:** Each crate stores snapshots in `tests/snapshots/` beside its `snapshot_tests.rs`. They are committed to source control.

**When to update snapshots:** Update snapshots only for intentional API/behaviour changes. CI runs in `INSTA_UPDATE=unseen` mode (accepts new snapshots, rejects changes to existing ones).

### Property Tests (proptest)

BitNet.rs uses [proptest](https://proptest-rs.github.io/proptest/intro.html) to verify invariants across randomised inputs. Property tests complement snapshot tests by covering edge cases that fixed examples miss.

**Running property tests:**

```bash
# Run all property tests
cargo nextest run --workspace --no-default-features --features cpu prop

# Run with more cases for deeper coverage
PROPTEST_CASES=1000 cargo nextest run --workspace --no-default-features --features cpu prop

# Run for a specific crate
cargo nextest run -p bitnet-quantization --no-default-features --features cpu prop
cargo nextest run -p bitnet-sampling --no-default-features --features cpu prop
```

**Key property invariants tested:**
- Quantization round-trip accuracy (I2_S, TL1, TL2)
- Sampling reproducibility with fixed seeds
- Tokenizer encoding round-trips
- GGUF header field ordering invariants

### Fuzz Testing (cargo-fuzz)

BitNet.rs has 13 fuzz targets covering parsers, kernels, and tokenizers. Fuzz targets run nightly via `.github/workflows/fuzz-ci.yml` (60 seconds per target, `01:00 UTC`).

**Running fuzz tests manually:**

```bash
# List available fuzz targets
cargo fuzz list

# Fuzz a specific target (runs indefinitely - Ctrl+C to stop)
cargo fuzz run quantization_i2s

# Run with a time limit (e.g., 60 seconds)
cargo fuzz run gguf_parser -- -max_total_time=60

# Run all targets briefly (CI mode, 30s each)
for target in $(cargo fuzz list); do
  cargo fuzz run "$target" -- -max_total_time=30 || true
done
```

**Available fuzz targets:**
| Target | Tests |
|--------|-------|
| `quantization_i2s` | I2_S dequantization with arbitrary inputs |
| `quantization_tl1` | TL1 lookup table with arbitrary codes |
| `quantization_tl2` | TL2 lookup table with arbitrary codes |
| `gguf_parser` | GGUF file header parsing |
| `safetensors_parser` | SafeTensors format parsing |
| `kernel_matmul` | Matrix multiply kernel correctness |
| `tokenizer_discovery` | Tokenizer file auto-discovery |
| `i2s_quantize_roundtrip` | I2_S quantize-dequantize round-trip |
| `sampling_temperature` | Temperature sampling with extreme values |
| `prompt_template` | Prompt template formatting |
| `receipt_json` | Receipt JSON deserialisation |

**Corpus:** Seed corpora live in `fuzz/corpus/<target>/`. CI uploads crash artifacts to GitHub Actions on failure.

### Enhanced Mock Infrastructure and Tokenizer Testing

BitNet.rs includes comprehensive mock infrastructure for robust testing without external dependencies:

#### Mock Model and Tokenizer Testing

```bash
# Test mock model implementation with prefill functionality
cargo test --no-default-features -p bitnet-inference --test batch_prefill --no-default-features --features cpu
cargo test --no-default-features -p bitnet-inference --no-default-features --features cpu

# Test tokenizer builder pattern and Arc<dyn Tokenizer> architecture
cargo test --no-default-features -p bitnet-tokenizers test_tokenizer_builder_from_file --no-default-features --features cpu
cargo test --no-default-features -p bitnet-tokenizers test_universal_tokenizer_mock_fallback --no-default-features --features cpu

# Validate performance metrics with mock infrastructure
cargo test --no-default-features -p bitnet-cli test_inference_metrics_collection --no-default-features --features cpu
cargo test --no-default-features -p bitnet-cli test_batch_inference_with_mock_model --no-default-features --features cpu
```

#### Safe Environment Variable Handling Tests

```bash
# Test enhanced environment variable management with proper unsafe blocks
cargo test --no-default-features -p bitnet-cli test_safe_environment_setup --no-default-features --features cpu
cargo test --no-default-features -p bitnet-cli test_deterministic_configuration --no-default-features --features cpu

# Validate environment variable handling in different scenarios
BITNET_DETERMINISTIC=1 cargo test --no-default-features -p bitnet-cli test_deterministic_inference --no-default-features --features cpu
BITNET_SEED=42 cargo test --no-default-features -p bitnet-cli test_seeded_generation --no-default-features --features cpu
```

#### Mock Infrastructure Features

- **Mock Model Implementation**: Complete model interface with configurable responses
- **Mock Tokenizer**: Testing-compatible tokenizer with predictable behavior
- **Arc<dyn Tokenizer> Support**: Enhanced tokenizer architecture using `TokenizerBuilder::from_file()`
- **Performance Metrics Validation**: Structured testing of timing and throughput metrics
- **Safe Environment Handling**: Proper unsafe block usage for environment variable operations

### GPU Testing Strategy

GPU testing requires special consideration due to hardware dependencies and resource management. See [GPU Development Guide](gpu-development.md#gpu-testing-strategy) for comprehensive coverage of GPU testing categories, hardware-specific test configuration, and CI/CD considerations.

### Concurrency-Capped Testing

Use concurrency caps to prevent resource exhaustion:

```bash
# Run tests with concurrency caps (prevents resource storms)
scripts/preflight.sh && cargo t2                     # 2-thread CPU tests
scripts/preflight.sh && cargo crossval-capped        # Cross-validation with caps
scripts/e2e-gate.sh cargo test --no-default-features --features crossval   # Gate heavy E2E tests
```

See [Concurrency Caps Guide](concurrency-caps.md) for detailed information on preflight scripts, e2e gates, and resource management strategies.

### Performance Tracking Tests

The performance tracking infrastructure includes comprehensive test coverage for metrics collection, validation, and environment configuration:

```bash
# Run all performance tracking tests
cargo test --no-default-features -p bitnet-inference --no-default-features --features "cpu,integration-tests" --test performance_tracking_tests

# Run specific performance test categories
cargo test --no-default-features --test performance_tracking_tests performance_metrics_tests --no-default-features --features cpu
cargo test --no-default-features --test performance_tracking_tests performance_tracker_tests --no-default-features --features cpu
cargo test --no-default-features --test performance_tracking_tests environment_variable_tests --no-default-features --features cpu

# Test InferenceEngine performance integration
cargo test --no-default-features -p bitnet-inference --no-default-features --features "cpu,integration-tests" test_engine_performance_tracking_integration

# Test platform-specific memory and performance tracking
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_memory_tracking
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_performance_tracking

# GPU performance validation with comprehensive metrics
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_cuda_validation_comprehensive
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_gpu_memory_management
```

#### Performance Test Categories

1. **Performance Metrics Tests**: Validate metric computation, validation, and accuracy
2. **Performance Tracker Tests**: Test state management and metrics aggregation
3. **Environment Variable Tests**: Validate configuration through environment variables
4. **Integration Tests**: End-to-end performance tracking with InferenceEngine
5. **Platform-Specific Tests**: Memory tracking and CPU kernel selection monitoring
6. **GPU Performance Tests**: GPU memory management and performance benchmarking

See [Performance Tracking Guide](performance-tracking.md) for detailed usage examples and configuration options.

## Specialized Test Commands

### GGUF Validation Tests

```bash
# Run GGUF validation tests
cargo test --no-default-features -p bitnet-inference --test gguf_header --no-default-features --features cpu
cargo test --no-default-features -p bitnet-inference --test gguf_fuzz --no-default-features --features cpu
cargo test --no-default-features -p bitnet-inference --test engine_inspect --no-default-features --features cpu

# Run async smoke test with synthetic GGUF
printf "GGUF\x02\x00\x00\x00" > /tmp/t.gguf && \
printf "\x00\x00\x00\x00\x00\x00\x00\x00" >> /tmp/t.gguf && \
printf "\x00\x00\x00\x00\x00\x00\x00\x00" >> /tmp/t.gguf && \
BITNET_GGUF=/tmp/t.gguf cargo test --no-default-features -p bitnet-inference --no-default-features --features rt-tokio --test smoke
```

### Convolution Testing Framework

The convolution testing framework includes comprehensive validation against PyTorch reference implementations and extensive unit testing for various parameter combinations.

#### PyTorch Reference Testing

The convolution implementation includes optional PyTorch reference tests that validate correctness by comparing outputs with PyTorch's `F.conv2d` implementation:

```bash
# Prerequisites: Install Python and PyTorch
pip install torch

# Run PyTorch reference tests (ignored by default)
cargo test --no-default-features -p bitnet-kernels conv2d_reference_cases --no-default-features --features cpu -- --ignored

# Verbose output to see test details
cargo test --no-default-features -p bitnet-kernels conv2d_reference_cases --no-default-features --features cpu -- --ignored --nocapture
```

The reference tests cover:
- **Basic convolution**: Simple 2D convolution operations
- **Stride operations**: Various stride configurations (1x1, 2x2)
- **Padding operations**: Zero padding with different configurations
- **Dilation operations**: Dilated convolutions for expanded receptive fields
- **Parameter combinations**: Mixed stride, padding, and dilation

#### Quantization Testing

Comprehensive testing of quantized convolution operations:

```bash
# Test I2S quantization (2-bit signed)
cargo test --no-default-features -p bitnet-kernels test_conv2d_quantized_i2s --no-default-features --features cpu

# Test TL1 quantization (table lookup)
cargo test --no-default-features -p bitnet-kernels test_conv2d_quantized_tl1 --no-default-features --features cpu

# Test TL2 quantization (advanced table lookup)
cargo test --no-default-features -p bitnet-kernels test_conv2d_quantized_tl2 --no-default-features --features cpu

# Test quantization with bias
cargo test --no-default-features -p bitnet-kernels test_conv2d_quantized_with_bias --no-default-features --features cpu

# Test scale factor application
cargo test --no-default-features -p bitnet-kernels test_conv2d_quantized_scale_factor --no-default-features --features cpu
```

#### Error Handling and Validation

The convolution tests include comprehensive error handling validation:

```bash
# Test dimension mismatch errors
cargo test --no-default-features -p bitnet-kernels test_conv2d_dimension_mismatch --no-default-features --features cpu

# Test invalid input size errors
cargo test --no-default-features -p bitnet-kernels test_conv2d_invalid_input_size --no-default-features --features cpu

# Test invalid bias size errors
cargo test --no-default-features -p bitnet-kernels test_conv2d_invalid_bias_size --no-default-features --features cpu

# Test quantized weight size validation
cargo test --no-default-features -p bitnet-kernels test_conv2d_quantized_invalid_weight_size --no-default-features --features cpu

# Test scale size validation
cargo test --no-default-features -p bitnet-kernels test_conv2d_quantized_invalid_scale_size --no-default-features --features cpu
```

### IQ2_S Backend Tests

```bash
# Build with IQ2_S quantization support (requires GGML FFI)
cargo build --no-default-features --release --no-default-features --features "cpu,iq2s-ffi"

# Run IQ2_S backend validation
./scripts/test-iq2s-backend.sh

# Run unit tests
cargo test --package bitnet-models --no-default-features --features "cpu,iq2s-ffi"
```

### Streaming Tests

```bash
# Test streaming generation
cargo run --example streaming_generation --no-default-features --features cpu

# Test server streaming
cargo test --no-default-features -p bitnet-server --no-default-features --features cpu streaming

# Test token ID accuracy
cargo test --no-default-features -p bitnet-inference --no-default-features --features cpu test_token_id_streaming
```

For more streaming functionality and Server-Sent Events testing, see the [Streaming API Guide](streaming-api.md).

## Ignored and Skipped Tests

BitNet.rs intentionally maintains a set of ignored tests (marked with `#[ignore]`) as part of the TDD development approach. This section categorizes why tests are skipped and how to interpret them.

### Categorization Overview

**Total Skipped Tests**: 192
- **Issue Blockers** (~70): Awaiting architectural fixes for active issues
- **Slow/Performance Tests** (~50): QK256 scalar kernels exceed timeout thresholds
- **Feature Scaffolding** (~40): TDD placeholders for post-MVP features
- **Fixtures/Integration** (~32): Integration tests requiring special setup

### Issue Blockers (70+ tests)

These tests are blocked by active issues and cannot run until root causes are fixed.

#### Issue #254: Shape Mismatch in Layer-Norm

**Status**: In analysis phase
**Impact**: Blocks real inference tests for multiple architectures
**Tests Affected**: ~15 inference end-to-end tests

```rust
#[test]
#[ignore] // Blocked by Issue #254 - shape mismatch in layer-norm
fn test_inference_with_shape_validation() { /* ... */ }
```

**Investigation Resources**:
- GitHub issue #254 for detailed analysis and discussion
- Shape handling during layer normalization
- Device-specific tensor dimension mismatches

#### Issue #260: Mock Elimination Not Complete

**Status**: Awaiting refactoring
**Impact**: Prevents full transition to real inference paths
**Tests Affected**: ~15 inference end-to-end tests

```rust
#[test]
#[ignore] // Blocked by Issue #260 - mock elimination not complete
fn test_real_inference_path() {
    unimplemented!("Waiting for mock elimination refactoring")
}
```

**Investigation Resources**:
- GitHub issue #260 for refactoring plan
- Mock vs real inference path differentiation
- Integration test infrastructure updates

#### Issue #469: Tokenizer Parity and FFI Build Hygiene

**Status**: Active development
**Impact**: Blocks cross-validation tests and FFI integration
**Tests Affected**: ~20 cross-validation and tokenizer tests

```rust
#[test]
#[ignore] // Blocked by Issue #469 - tokenizer parity not ready
fn test_cpp_tokenizer_parity() { /* ... */ }
```

**Investigation Resources**:
- GitHub issue #469 for FFI and tokenizer work
- C++ vs Rust tokenizer behavior differences
- FFI build system hygiene improvements

#### Issue #439: Feature Gate Consistency (RESOLVED)

**Status**: ✅ **RESOLVED** (PR #475 merged)
**Unlock Status**: GPU/CPU feature predicates unified
**Tests Unlocked**: All device selection and fallback tests now passing

### Slow/Performance Tests (50+ tests)

These tests are intentionally skipped due to performance characteristics that exceed timeout thresholds.

#### QK256 Scalar Kernel Tests

**Reason**: QK256 MVP uses scalar-only kernels (~0.1 tok/s for 2B models)
**Performance Impact**: Inference at this speed exceeds 5-minute nextest timeout for full models
**Workaround**: Use `--max-new-tokens 4-16` for quick validation

```bash
# Skip slow tests and run faster suite
BITNET_SKIP_SLOW_TESTS=1 cargo test --workspace --no-default-features --features cpu

# Run slow tests separately with extended timeout (not recommended)
cargo test --workspace --no-default-features --features cpu -- --ignored --include-ignored
```

```rust
#[test]
#[ignore] // Slow: QK256 scalar kernels (~0.1 tok/s). Use --max-new-tokens 4-16.
fn test_qk256_full_model_inference() {
    // Full model inference test - takes 10+ minutes
}
```

**Expected Timeline**: SIMD optimizations planned for post-MVP phase to achieve ≥3× uplift.

#### GPU Performance Benchmarks

**Reason**: GPU benchmarks require extended execution time for meaningful results
**Setup**: Marked as ignored, runs manually in development

```rust
#[test]
#[ignore] // GPU benchmark - run manually: cargo test --ignored -- --nocapture
fn test_gpu_performance_baseline() { /* ... */ }
```

### Feature Scaffolding Tests (40+ tests)

These tests are TDD placeholders for features planned in post-MVP phases.

#### Post-MVP GPU Mixed-Precision (After #439)

```rust
#[test]
#[ignore] // TODO: GPU mixed-precision FP16/BF16 implementation (post-MVP)
fn test_gpu_fp16_dequantization() {
    unimplemented!("Waiting for GPU optimization phase")
}
```

#### Advanced Quantization Formats (Post-v0.2)

```rust
#[test]
#[ignore] // TODO: IQ3_S and higher-precision formats (post-v0.2)
fn test_iq3s_quantization() {
    unimplemented!("Planned for v0.3")
}
```

#### Model Export and Optimization Tools

```rust
#[test]
#[ignore] // TODO: ONNX export pipeline (post-MVP)
fn test_onnx_model_export() {
    unimplemented!("Waiting for export framework")
}
```

### Fixture/Integration Tests (32+ tests)

These tests require special setup or external resources.

```bash
# Run only when fixtures feature is enabled
cargo test --workspace --no-default-features --features "cpu,fixtures"

# Skip fixture tests in normal test runs
cargo test --workspace --no-default-features --features cpu  # Fixture tests skipped
```

```rust
#[test]
#[cfg_attr(not(feature = "fixtures"), ignore)]
fn test_with_real_gguf_fixture() {
    // Only runs when fixtures feature is enabled
}
```

### Understanding Test Markers

#### Pattern 1: Issue Blocker

```rust
#[test]
#[ignore] // Blocked by Issue #254 - shape mismatch in layer-norm
fn test_inference_validation() { /* ... */ }
```

**Action**: Check GitHub issue tracker for status and workarounds.

#### Pattern 2: Slow Test

```rust
#[test]
#[ignore] // Slow: ~10 minutes. Set BITNET_SKIP_SLOW_TESTS=0 to run.
fn test_full_model_inference() { /* ... */ }
```

**Action**: Run with `-- --ignored` if needed, or use `BITNET_SKIP_SLOW_TESTS=0`.

#### Pattern 3: Feature Scaffolding

```rust
#[test]
#[ignore] // TODO: Implement post-MVP feature
fn test_future_feature() {
    unimplemented!("Waiting for feature implementation")
}
```

**Action**: Track in development roadmap; will be enabled when feature is implemented.

#### Pattern 4: Feature-Gated

```rust
#[test]
#[cfg_attr(not(feature = "fixtures"), ignore)]
fn test_with_fixture() { /* ... */ }
```

**Action**: Enable feature flag to run: `cargo test --features fixtures`.

### Working with Ignored Tests

#### Check Status of Specific Blocker

```bash
# Find all tests blocked by Issue #254
grep -r "Issue #254" tests --include="*.rs"

# Count blocked tests
grep -r "Blocked by Issue" tests --include="*.rs" | wc -l
```

#### Run Single Ignored Test (if needed)

```bash
# Run a specific ignored test
cargo test test_name -- --ignored --exact

# Run all ignored tests matching pattern
cargo test pattern -- --ignored
```

#### Debug Ignored Test (understand why it's skipped)

```bash
# View the test and its ignore reason
grep -A 10 "#\[ignore\]" tests/test_file.rs

# Check git history for when test was ignored
git log --oneline -S "#[ignore]" -- tests/test_file.rs

# Find issue references
grep -r "#254\|#260\|#469" tests --include="*.rs"
```

### Expected Timeline for Unblocking Tests

| Issue | Status | Expected Unlock | Test Count |
|-------|--------|-----------------|-----------|
| #254 | Analysis | Q4 2024 | ~15 tests |
| #260 | Refactoring | Q4 2024 | ~15 tests |
| #439 | ✅ Resolved | PR #475 merged | ~12 tests (unlocked) |
| #469 | Active Dev | Q1 2025 | ~20 tests |
| QK256 Perf | SIMD Work | Q1 2025 | ~50 tests |

### CI Behavior with Ignored Tests

**In CI**: Only non-ignored tests run (3,359+ enabled tests)
**Ignored tests**: Tracked separately, not blocking CI
**Skipped tests**: 462 tests properly marked as skipped
**Exit code**: Success (0) even with 462+ skipped tests

To run ignored tests locally:

```bash
# Opt-in to run ignored tests
cargo test --workspace --no-default-features --features cpu -- --ignored --include-ignored

# This will encounter blocked tests and see failures for issues #254, #260, #469
```

## Environment Variable Testing

Environment variables are critical for controlling test behavior, determinism, and feature flags. BitNet.rs provides **EnvGuard** - a thread-safe, RAII-based utility for safe environment variable manipulation in tests that prevents test pollution and data races.

### When to Use EnvGuard

Use EnvGuard whenever your test:

1. **Calls `std::env::set_var()` or `std::env::remove_var()`** - These unsafe operations require proper synchronization
2. **Reads and relies on environment variables** - Ensures isolation from other tests
3. **Tests configuration that depends on environment** - e.g., `BITNET_DETERMINISTIC`, `BITNET_STRICT_MODE`
4. **Needs to validate environment-based behavior** - Device selection, GPU detection, feature flags

### Required Pattern: #[serial(bitnet_env)]

All tests using environment variables **must** use the `#[serial(bitnet_env)]` attribute to prevent process-level races:

```rust
use serial_test::serial;
use tests::support::env_guard::EnvGuard;

#[test]
#[serial(bitnet_env)]  // REQUIRED - prevents races with other env-mutating tests
fn test_with_environment() {
    let guard = EnvGuard::new("BITNET_DETERMINISTIC");
    guard.set("1");

    // Test code here - environment is isolated
}
// Guard drops automatically, restoring original state
```

**Without `#[serial(bitnet_env)]**, your test can race with others and cause flaky failures across the suite.

### Complete Examples

#### Basic Usage: Single Environment Variable

```rust
use serial_test::serial;
use tests::support::env_guard::EnvGuard;

#[test]
#[serial(bitnet_env)]
fn test_strict_mode_enabled() {
    let guard = EnvGuard::new("BITNET_STRICT_MODE");
    guard.set("1");

    // Your code can now check the environment variable
    assert_eq!(std::env::var("BITNET_STRICT_MODE").unwrap(), "1");

    // Guard is automatically dropped at end of scope
}
```

#### Scoped Approach: Using `temp_env` (Preferred for Simple Cases)

For simple, linear test flows, use `temp_env::with_var()` for cleaner syntax:

```rust
use serial_test::serial;
use temp_env::with_var;

#[test]
#[serial(bitnet_env)]
fn test_deterministic_inference() {
    // Closure-based approach - automatically restored on scope exit
    with_var("BITNET_DETERMINISTIC", Some("1"), || {
        with_var("BITNET_SEED", Some("42"), || {
            // Your test code here
            assert_eq!(std::env::var("BITNET_DETERMINISTIC").unwrap(), "1");
            assert_eq!(std::env::var("BITNET_SEED").unwrap(), "42");
        });
    });

    // Both variables automatically restored here
}
```

#### RAII Approach: Multiple Sequential Steps

Use EnvGuard when you need multiple sequential steps or complex setup:

```rust
use serial_test::serial;
use tests::support::env_guard::EnvGuard;

#[test]
#[serial(bitnet_env)]
fn test_complex_environment_setup() {
    // Create guards for multiple variables
    let det_guard = EnvGuard::new("BITNET_DETERMINISTIC");
    let seed_guard = EnvGuard::new("BITNET_SEED");
    let threads_guard = EnvGuard::new("RAYON_NUM_THREADS");

    // Set them sequentially
    det_guard.set("1");
    seed_guard.set("42");
    threads_guard.set("1");

    // Step 1: Verify deterministic mode is enabled
    assert_eq!(std::env::var("BITNET_DETERMINISTIC").unwrap(), "1");

    // Step 2: Verify seed is set
    assert_eq!(std::env::var("BITNET_SEED").unwrap(), "42");

    // Step 3: Verify thread count is limited
    assert_eq!(std::env::var("RAYON_NUM_THREADS").unwrap(), "1");

    // Step 4: Run your test code with these settings
    // ... test implementation ...

    // All variables are automatically restored when guards drop
}
```

#### Removing Environment Variables

Remove an environment variable and automatically restore it on cleanup:

```rust
use serial_test::serial;
use tests::support::env_guard::EnvGuard;

#[test]
#[serial(bitnet_env)]
fn test_missing_env_var() {
    // First, set up a baseline environment variable
    unsafe {
        std::env::set_var("BITNET_GPU", "true");
    }

    // Now test the case where it's missing
    let guard = EnvGuard::new("BITNET_GPU");
    guard.remove();

    // Verify it's gone
    assert!(std::env::var("BITNET_GPU").is_err());

    // Test code that validates behavior when variable is absent
    let has_gpu = std::env::var("BITNET_GPU").is_ok();
    assert!(!has_gpu);

    // Guard drops here, restoring the original value
    drop(guard);

    // Verify it's restored
    assert_eq!(std::env::var("BITNET_GPU").unwrap(), "true");
}
```

#### Checking Original Values

Access the original value to understand what was overridden:

```rust
use serial_test::serial;
use tests::support::env_guard::EnvGuard;

#[test]
#[serial(bitnet_env)]
fn test_preserves_original_value() {
    // Set an initial value
    unsafe {
        std::env::set_var("BITNET_BATCH_SIZE", "32");
    }

    let guard = EnvGuard::new("BITNET_BATCH_SIZE");

    // Check what the original was
    assert_eq!(guard.original_value(), Some("32"));

    // Change it
    guard.set("64");
    assert_eq!(std::env::var("BITNET_BATCH_SIZE").unwrap(), "64");

    // When dropped, automatically restores to original
}
```

#### Panic Safety

EnvGuard is panic-safe - the Drop implementation runs even if the test panics:

```rust
use serial_test::serial;
use tests::support::env_guard::EnvGuard;

#[test]
#[serial(bitnet_env)]
fn test_with_panic_safety() {
    let guard = EnvGuard::new("BITNET_TEST_VAR");
    guard.set("original");

    let result = std::panic::catch_unwind(|| {
        assert_eq!(std::env::var("BITNET_TEST_VAR").unwrap(), "original");

        guard.set("modified");
        assert_eq!(std::env::var("BITNET_TEST_VAR").unwrap(), "modified");

        // Simulate a panic mid-test
        panic!("Test failed!");
    });

    assert!(result.is_err(), "Should have panicked");

    // Even though test panicked, guard was properly dropped and restored
    // This can be verified by a subsequent test
}
```

### Common Pitfalls

#### ❌ ANTI-PATTERN: Missing #[serial(bitnet_env)]

```rust
#[test]
// ❌ WRONG: This test can race with other env-mutating tests!
fn test_without_serialization() {
    unsafe { std::env::set_var("BITNET_STRICT_MODE", "1"); }
    // Test pollution and flaky failures!
}
```

**Why this fails**: Without serialization, multiple tests can modify the same environment variable concurrently, causing unpredictable behavior.

**Fix**: Always add `#[serial(bitnet_env)]`:

```rust
#[test]
#[serial(bitnet_env)]  // ✅ CORRECT
fn test_with_serialization() {
    let _guard = EnvGuard::new("BITNET_STRICT_MODE");
    _guard.set("1");
    // Safe and isolated
}
```

#### ❌ ANTI-PATTERN: Forgetting to Hold the Guard

```rust
#[test]
#[serial(bitnet_env)]
fn test_with_dropped_guard() {
    // ❌ WRONG: Guard is immediately dropped!
    EnvGuard::new("BITNET_SEED").set("42");

    // Variable is already restored!
    assert!(std::env::var("BITNET_SEED").is_err());  // FAILS
}
```

**Why this fails**: The guard goes out of scope immediately after creation, restoring the environment variable.

**Fix**: Bind the guard to a variable:

```rust
#[test]
#[serial(bitnet_env)]
fn test_with_held_guard() {
    let _guard = EnvGuard::new("BITNET_SEED");  // ✅ Bound to variable
    _guard.set("42");

    assert_eq!(std::env::var("BITNET_SEED").unwrap(), "42");
}
```

#### ❌ ANTI-PATTERN: Direct std::env::set_var Without Guard

```rust
#[test]
#[serial(bitnet_env)]
fn test_with_unguarded_env() {
    // ❌ WRONG: No restoration on test end
    unsafe { std::env::set_var("BITNET_BATCH_SIZE", "128"); }

    // Variable persists after test - pollutes subsequent tests!
}
```

**Why this fails**: Without a guard, the environment variable persists after the test ends, affecting other tests.

**Fix**: Always use EnvGuard:

```rust
#[test]
#[serial(bitnet_env)]
fn test_with_guarded_env() {
    let _guard = EnvGuard::new("BITNET_BATCH_SIZE");  // ✅ Will restore
    _guard.set("128");

    // Automatically restored when guard drops
}
```

### CI Enforcement and Validation

BitNet.rs includes automated checks to detect EnvGuard violations:

#### Checking for Missing #[serial(bitnet_env)]

```bash
# Check if tests properly use #[serial(bitnet_env)]
grep -r "std::env::set_var\|std::env::remove_var" tests --include="*.rs" | \
  grep -v "#\[serial(bitnet_env)" | \
  grep -v "unsafe {" || echo "✅ No violations found"
```

#### Validating Guard Usage

```bash
# Identify tests that use environment variables without proper guards
cargo clippy --all-targets --tests -- -W clippy::all
```

#### Test-Specific Validation

```bash
# Run environment variable tests with strict checking
cargo test --test '*env*' --no-default-features --features cpu -- --nocapture
```

### How to Fix Violations

If you find an environment variable test without proper guards:

**Step 1**: Add `#[serial(bitnet_env)]` attribute

```rust
#[test]
#[serial(bitnet_env)]  // ADD THIS
fn test_name() { /* ... */ }
```

**Step 2**: Wrap environment modifications with EnvGuard

```rust
use tests::support::env_guard::EnvGuard;

let guard = EnvGuard::new("VAR_NAME");
guard.set("value");
// or
guard.remove();
```

**Step 3**: Verify the test still passes

```bash
cargo test --test test_name -- --nocapture
```

## Environment Variables for Testing

### Runtime Variables
- `BITNET_GGUF` / `CROSSVAL_GGUF`: Path to test model
- `BITNET_CPP_DIR`: Path to C++ implementation
- `BITNET_DETERMINISTIC`: Enable deterministic mode for testing
- `BITNET_SEED`: Set seed for reproducible runs
- `RAYON_NUM_THREADS`: Control CPU parallelism

### Test-Specific Variables
- `RUST_TEST_THREADS`: Rust test parallelism
- `CROSSVAL_WORKERS`: Cross-validation test workers

For complete list of environment variables, see the main project documentation.
