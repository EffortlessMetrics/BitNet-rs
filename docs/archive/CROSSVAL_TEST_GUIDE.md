# Cross-Validation Test Infrastructure - Quick Reference Guide

## Overview

This guide provides a quick reference to BitNet-rs's comprehensive cross-validation test infrastructure. For detailed documentation, see `crossval-test-infrastructure.md`.

## Quick Links

- **Full Documentation**: `docs/explanation/crossval-test-infrastructure.md`
- **Test Files**: `crossval/tests/*.rs`
- **Test Helpers**: `tests/common/`
- **CI Workflows**: `.github/workflows/`
- **Configuration**: `.config/nextest.toml`

## Test Structure at a Glance

### 13 Test Suites

| Suite | Purpose | Location | Size |
|-------|---------|----------|------|
| **parity** | Rust vs C++ logits | `parity.rs` | 150 LOC |
| **parity_bitnetcpp** | Full BitNet.cpp integration | `parity_bitnetcpp.rs` | 950 LOC |
| **parity_receipts** | Receipt schema v1.0.0 | `parity_receipts.rs` | 600 LOC |
| **qk256_crossval** | QK256 vs FP32 reference | `qk256_crossval.rs` | 530 LOC |
| **per_position_logits** | Per-token divergence detection | `per_position_logits.rs` | 355 LOC |
| **performance_validation** | Throughput/latency benchmarking | `performance_validation.rs` | 650 LOC |
| **ffi_integration** | FFI binding safety | `ffi_integration.rs` | 270 LOC |
| **iq2s_validation** | IQ2S quantization via FFI | `iq2s_validation.rs` | 350 LOC |
| **framework_validation** | Cross-validation harness | `framework_validation.rs` | 590 LOC |
| **token_equivalence** | Tokenizer parity | `token_equivalence.rs` | 170 LOC |
| **smoke** | Minimal environment checks | `smoke.rs` | 100 LOC |
| **cpp_probe** | C++ availability detection | `cpp_probe.rs` | 14 LOC |
| **ms_bitnet_mapping** | Model mapping tests | `ms_bitnet_mapping.rs` | 35 LOC |

## Core Infrastructure

### 1. Test Harness

**File**: `tests/common/harness.rs`

Async parallel test execution with:
- Semaphore-based rate limiting (default: 4 threads)
- Per-test environment isolation
- 5-minute timeout enforcement
- Async reporter callbacks
- Real-time statistics

```rust
// Usage in test
pub async fn run_test_suite(&self, suite: &dyn TestSuite) -> Result<TestSuiteResult>
pub async fn run_single_test(&self, test_case: Box<dyn TestCase>) -> TestRecord
```

### 2. Environment Isolation (EnvGuard)

**File**: `tests/common/env.rs`

RAII pattern for safe environment variable manipulation:

```rust
// Set variable (restored on drop)
let _guard = EnvGuard::set("BITNET_SEED", "42");

// Acquire global lock (serializes env-mutating tests)
let _lock = env_guard();

// Typed parsing
let value = env_bool("BITNET_DETERMINISTIC");
let timeout = env_duration_secs("BITNET_TIMEOUT");
```

### 3. Fixture Management

**File**: `tests/common/fixtures_facade.rs`

Feature-gated fixture access:

```rust
// Create fixtures facade (works with/without feature)
let fixtures = Fixtures::new(&config).await?;

// Load model fixture
let model_path = fixtures.get_model_fixture("qk256-small").await?;
```

### 4. Cross-Validation Module

**Location**: `tests/common/cross_validation/`

Rust vs C++ comparison infrastructure:
- `comparison.rs`: Comparison logic with tolerance
- `rust_implementation.rs`: Rust-side test implementation
- `cpp_implementation.rs`: C++-side test implementation (FFI)
- `test_runner.rs`: Test execution engine
- `test_cases.rs`: Standard test cases

## Running Tests

### Basic Commands

```bash
# Run all crossval tests
cargo test --package crossval --features crossval -- --nocapture

# Run specific test suite
cargo test --package crossval --features crossval parity -- --nocapture

# Run with CI profile (4 fixed threads)
cargo nextest run --profile ci --package crossval --features crossval

# Run with verbose logging
RUST_LOG=debug cargo test --package crossval -- --nocapture --test-threads=1
```

### With Model Fixtures

```bash
# Setup environment
export CROSSVAL_GGUF=path/to/model.gguf
export BITNET_CPP_DIR=/path/to/bitnet.cpp

# Run tests
cargo test --package crossval --features crossval -- --nocapture
```

### Specific Scenarios

```bash
# Skip slow tests (QK256 scalar kernels)
BITNET_SKIP_SLOW_TESTS=1 cargo test --workspace --no-default-features --features cpu

# Run with nextest fast-only filter
cargo nextest run --profile ci --filter-expr 'not slow'

# Run GPU tests (1 thread due to memory constraints)
cargo nextest run --profile gpu --features gpu --test kernel_tests
```

## Key Patterns

### Pattern 1: Feature-Gated Tests

```rust
#[cfg(all(test, feature = "crossval"))]
mod tests {
    // Tests requiring crossval feature
    
    #[test]
    #[cfg(any(feature = "cpu", feature = "gpu"))]
    fn test_with_available_backend() {
        // Runs on both CPU and GPU builds
    }
}
```

### Pattern 2: Fixture-Based Model Testing

```rust
fn test_model_path() -> Option<String> {
    // Check C++ backend available
    if !bitnet_sys::is_available() {
        eprintln!("skipping - C++ backend unavailable");
        return None;
    }
    
    // Resolve model from environment
    std::env::var("CROSSVAL_GGUF").ok()
}

#[test]
fn test_with_model() {
    let Some(model_path) = test_model_path() else {
        return;  // Skip if fixture unavailable
    };
    // Test code...
}
```

### Pattern 3: Environment Isolation

```rust
#[test]
fn test_with_env_isolation() {
    // Acquire global lock (serializes with other env tests)
    let _global_lock = env_guard();
    
    // Set variable (automatically restored on drop)
    let _guard = EnvGuard::set("BITNET_DETERMINISTIC", "1");
    
    // Test code...
}
```

### Pattern 4: Mock Fixtures

```rust
#[tokio::test]
async fn test_with_receipt_fixture() {
    let receipt = InferenceReceipt::generate(
        "cpu",
        vec!["i2s_gemv".to_string(), "rope_apply".to_string()],
    )?
    .with_parity(ParityMetadata {
        cpp_available: true,
        cosine_similarity: Some(0.9923),
        exact_match_rate: Some(1.0),
        status: "ok".to_string(),
    });
    
    // Test receipt structure
    assert_eq!(receipt.schema_version, "1.0.0");
}
```

## Nextest Profiles

**File**: `.config/nextest.toml`

| Profile | Use Case | Threads | Timeout | Notes |
|---------|----------|---------|---------|-------|
| `default` | Development | num-cpus | 5min | Fast-fail, verbose |
| `ci` | CI/CD | 4 fixed | 5min | Reproducible |
| `fixtures` | GGUF loading | 2 | 10min | I/O bound |
| `gpu` | GPU tests | 1 | 5min | Memory constrained |
| `doctests` | Doc examples | num-cpus | 2min | Simple tests |

## CI Workflows

**Location**: `.github/workflows/`

| Workflow | Trigger | Scenarios |
|----------|---------|-----------|
| `crossval-fast.yml` | Nightly, manual, `crossval` label | Token equivalence, performance |
| `testing-framework-crossval.yml` | Nightly, manual, `crossval` label | Accuracy, performance, edge-cases |
| `verify-receipts.yml` | PR, schedule | Schema v1.0.0, 8 validation gates |
| `property-tests.yml` | PR, schedule | Property-based quantization |
| `gpu.yml` | PR labeled `gpu-test` | CUDA kernel tests |

## Adding New Tests

### Step 1: Create Test File

```rust
// crossval/tests/my_new_test.rs
#![cfg(all(test, feature = "crossval"))]

#[cfg(all(test, feature = "crossval"))]
mod tests {
    use anyhow::Result;
    
    fn test_model_path() -> Option<String> {
        if !bitnet_sys::is_available() {
            eprintln!("skipping - C++ backend unavailable");
            return None;
        }
        std::env::var("CROSSVAL_GGUF").ok()
    }
    
    #[test]
    fn test_my_feature() {
        let Some(model_path) = test_model_path() else { return; };
        // Test implementation...
    }
}
```

### Step 2: Declare in Cargo.toml

```toml
[[test]]
name = "my_new_test"
path = "tests/my_new_test.rs"
required-features = ["crossval", "integration-tests"]
```

### Step 3: Add to CI Workflow

```yaml
- name: Run my new test
  run: |
    cargo test --package crossval \
      --features crossval \
      my_new_test
```

### Step 4: Test Locally

```bash
export CROSSVAL_GGUF=path/to/model.gguf
export BITNET_CPP_DIR=/path/to/bitnet.cpp
cargo test --package crossval --features crossval my_new_test -- --nocapture
```

## Debugging Failed Tests

### Enable Verbose Output

```bash
# Show all output
cargo test --package crossval --features crossval -- --nocapture

# With debug logging
RUST_LOG=debug cargo test --package crossval -- --nocapture

# With backtrace
RUST_BACKTRACE=1 cargo test --package crossval -- --nocapture
```

### Collect Artifacts

```bash
# Trace comparison
cargo run -p xtask -- trace-diff /tmp/rs_traces /tmp/cpp_traces

# Inspect receipts
find target -name "*.json" -path "*/crossval/*" -exec cat {} \; | jq .
```

## Feature Flags

| Flag | Purpose | Usage |
|------|---------|-------|
| `crossval` | C++ cross-validation | Full parity testing |
| `ffi` | FFI binding support | C++ backend access |
| `iq2s-ffi` | IQ2S quantization via FFI | Quantization testing |
| `integration-tests` | Integration test scaffolding | Complex test scenarios |
| `cpu` | SIMD-optimized CPU inference | CPU backend tests |
| `gpu` | CUDA acceleration | GPU backend tests |
| `fixtures` | Feature-gated fixtures | Model/data fixtures |

## Best Practices

1. **Always use EnvGuard** for environment mutations
2. **Acquire global lock** before using EnvGuard
3. **Check backend availability** before tests requiring it
4. **Use feature gates** to avoid linking errors
5. **Skip gracefully** when fixtures unavailable
6. **Add tolerance constants** at test module level
7. **Document expected behavior** in test comments
8. **Run locally** before pushing to CI

## Performance Benchmarks

- **CPU Tests**: ~5 minutes (4 parallel threads)
- **GPU Tests**: ~10 minutes (1 thread due to memory)
- **Fixture Loading**: ~2 minutes (GGUF parsing)
- **Per-test Timeout**: 5 minutes (configurable)

## Further Reading

For comprehensive documentation with examples and architectural details, see:
- **Full Report**: `docs/explanation/crossval-test-infrastructure.md`
- **Test Templates**: Section 9 of full report
- **CI Configuration**: Section 7 of full report
- **Infrastructure Details**: Section 3-6 of full report

## Support

For issues or questions:
1. Check the full documentation
2. Review existing test implementations
3. Use `RUST_LOG=debug` for detailed output
4. Check CI workflow artifacts on GitHub
