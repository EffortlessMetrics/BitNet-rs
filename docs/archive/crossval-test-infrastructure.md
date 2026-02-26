# BitNet-rs Crossval Test Infrastructure Report

## Executive Summary

BitNet-rs implements a comprehensive cross-validation framework for testing dual-backend (Rust vs C++) inference implementations. The test infrastructure spans **13 test suites** in `crossval/tests/`, integration helpers in `tests/common/`, and sophisticated CI workflows supporting parallel execution, environment isolation, and multi-scenario validation.

### Key Components

- **13 Crossval Test Suites**: Parity, QK256, FFI, performance, receipts, and smoke tests
- **Test Harness**: Async parallel execution with semaphore-based concurrency control
- **Environment Isolation**: EnvGuard pattern for safe test environment manipulation
- **CI Infrastructure**: 7 dedicated workflow files with label-triggered and scheduled execution
- **Fixtures System**: Feature-gated fixture management with facade pattern
- **Nextest Configuration**: 6 test profiles with timeout protection and output optimization

---

## 1. Test Structure Overview

### Location Hierarchy

```
BitNet-rs/
├── crossval/
│   └── tests/                                    # 13 crossval test files
│       ├── parity.rs                            # (150 LOC) Rust vs C++ logits parity
│       ├── parity_bitnetcpp.rs                  # (950 LOC) Full BitNet.cpp integration
│       ├── parity_receipts.rs                   # (600 LOC) Receipt schema v1.0.0 validation
│       ├── qk256_crossval.rs                    # (530 LOC) QK256 vs FP32 reference
│       ├── ffi_integration.rs                   # (270 LOC) FFI binding safety
│       ├── per_position_logits.rs               # (355 LOC) Per-token divergence detection
│       ├── performance_validation.rs            # (650 LOC) Throughput/latency benchmarking
│       ├── iq2s_validation.rs                   # (350 LOC) IQ2S quantization via FFI
│       ├── token_equivalence.rs                 # (170 LOC) Tokenizer parity
│       ├── framework_validation.rs              # (590 LOC) Cross-validation harness tests
│       ├── smoke.rs                             # (100 LOC) Minimal environment checks
│       ├── cpp_probe.rs                         # (14 LOC) C++ availability detection
│       └── ms_bitnet_mapping.rs                 # (35 LOC) Model mapping tests
│
├── tests/
│   ├── common/                                  # Test infrastructure
│   │   ├── mod.rs                              # Module orchestration with feature gates
│   │   ├── env.rs                              # EnvGuard & env var parsing
│   │   ├── harness.rs                          # TestHarness async parallel execution
│   │   ├── fixtures_facade.rs                  # Feature-gated fixture API
│   │   ├── config.rs                           # Test configuration
│   │   ├── errors.rs                           # Error types
│   │   ├── results.rs                          # Result/metrics types
│   │   └── cross_validation/
│   │       ├── mod.rs                          # CV module exports
│   │       ├── comparison.rs                   # Comparison logic
│   │       ├── rust_implementation.rs          # Rust-side test impl
│   │       ├── cpp_implementation.rs           # C++-side test impl (FFI)
│   │       ├── test_runner.rs                  # Test execution engine
│   │       └── test_cases.rs                   # Standard test cases
│   │
│   ├── support/
│   │   └── env_guard.rs                        # RAII guard implementation
│   │
│   └── [other test files]
│
└── .config/
    └── nextest.toml                            # Nextest profiles (6 profiles)

.github/workflows/
├── crossval-fast.yml                           # Fast cached crossval (nightly + manual)
├── testing-framework-crossval.yml              # Full test framework CV suite
├── verify-receipts.yml                         # Receipt schema validation
├── property-tests.yml                          # Property-based quantization tests
└── [5 other test-related workflows]
```

---

## 2. Integration Test Patterns

### Pattern 1: Backend-Specific Tests

The dual-backend pattern uses **feature gates** to conditionally compile tests for CPU vs GPU paths:

```rust
// crossval/tests/parity_bitnetcpp.rs
#[cfg(all(test, feature = "crossval"))]
mod tests {
    // C++ backend available tests
    
    #[test]
    #[cfg(any(feature = "cpu", feature = "gpu"))]
    fn test_parity_with_available_backend() {
        // Runs on both CPU and GPU builds
    }
    
    #[test]
    #[cfg(all(feature = "gpu", not(feature = "cpu")))]
    fn test_parity_gpu_only() {
        // GPU-specific parity test
    }
}
```

**Key Features:**
- Feature-gated compilation (`crossval`, `ffi`, `integration-tests`)
- Environment detection (`bitnet_sys::is_available()` for C++ backend)
- Graceful skipping when backends unavailable

### Pattern 2: Fixture-Based Model Testing

```rust
// crossval/tests/parity_bitnetcpp.rs (simplified)
fn test_model_path() -> Option<String> {
    // 1. Check if C++ backend available
    if !bitnet_sys::is_available() {
        eprintln!("skipping - C++ backend unavailable (set BITNET_CPP_DIR)");
        return None;
    }
    
    // 2. Resolve model path from env
    match std::env::var("CROSSVAL_GGUF") {
        Ok(path) => Some(path),
        Err(_) => {
            eprintln!("skipping - set CROSSVAL_GGUF to path of test model");
            None
        }
    }
}

#[test]
fn test_parity_first_tokens() {
    let Some(model_path) = test_model_path() else {
        return;  // Skip if fixture unavailable
    };
    
    // Load model and run parity check
    let rust_logits = eval_logits_once(&model_path, "Hello")?;
    let cpp_logits = CppSession::eval_logits(&model_path, "Hello")?;
    
    // Compare with tolerance
    compare_logits(&rust_logits, &cpp_logits, 1)?;
}
```

**Fixture Resolution Priority:**
1. Environment variable (`CROSSVAL_GGUF`, `BITNET_GGUF`)
2. Standard model directory (`models/`)
3. Test scaffold fixtures (feature-gated)
4. Skip with informative message if unavailable

### Pattern 3: Multi-Scenario Testing

```rust
// crossval/tests/per_position_logits.rs
#[tokio::test]
async fn test_per_token_logits_divergence() {
    // Scenario 1: Single token (sanity check)
    validate_token_parity(&model, "Hello", 1).await?;
    
    // Scenario 2: Multi-token (streaming)
    validate_token_parity(&model, "Hello world", 4).await?;
    
    // Scenario 3: Complex prompt (stress test)
    validate_token_parity(&model, "The quick brown fox...", 16).await?;
}
```

---

## 3. Test Harness Infrastructure

### TestHarness: Async Parallel Execution

**Location:** `tests/common/harness.rs`

```rust
pub struct TestHarness {
    config: TestConfig,
    fixtures: Fixtures,
    reporters: Vec<ConsoleReporter>,
    semaphore: Arc<Semaphore>,              // Parallelism control
    execution_stats: Arc<RwLock<ExecutionStats>>,
}

impl TestHarness {
    pub async fn run_test_suite(&self, suite: &dyn TestSuite) -> Result<TestSuiteResult> {
        let test_cases = suite.test_cases();
        
        // Spawn async tasks for each test (controlled by semaphore)
        for test_case in test_cases {
            let harness_clone = self.clone_for_test();
            let handle = tokio::spawn(async move {
                harness_clone.run_single_test_isolated(test_case).await
            });
            handles.push(handle);
        }
        
        // Collect results with timeout enforcement
        for handle in handles {
            let result = timeout(self.config.test_timeout, handle).await?;
            // Process result with reporter callbacks
        }
    }
    
    async fn run_single_test_isolated(&self, test_case: Box<dyn TestCase>) -> TestRecord {
        // 1. Acquire semaphore permit (rate limiting)
        let _permit = self.semaphore.acquire().await?;
        
        // 2. Create isolated environment
        let isolated_env = self.create_isolated_environment(&test_name).await;
        
        // 3. Setup phase
        test_case.setup(self.fixtures.ctx()).await?;
        
        // 4. Execute phase with timeout
        let result = timeout(self.config.test_timeout, test_case.execute()).await;
        
        // 5. Cleanup phase (always runs)
        let _ = test_case.cleanup().await;
        
        // 6. Clean isolated environment
        self.cleanup_isolated_environment(isolated_env).await;
        
        // 7. Return result with timestamps
        TestRecord { ... }
    }
}
```

**Key Properties:**
- **Parallelism**: Semaphore-based with configurable limits (default: 4 threads)
- **Isolation**: Per-test temp directory + env var restoration on drop
- **Timeout**: Global 5-minute timeout per test (nextest) + local timeout enforcement
- **Reporting**: Async callbacks for suite start/test complete/suite complete
- **Statistics**: Real-time tracking of pass/fail/timeout counts

### TestCase Trait: Pluggable Tests

```rust
#[async_trait]
pub trait TestCase: Send + Sync {
    fn name(&self) -> &str;
    
    async fn setup(&self, fixtures: FixtureCtx<'_>) -> Result<()> {
        Ok(())  // Default no-op
    }
    
    async fn execute(&self) -> Result<TestMetrics>;  // Required
    
    async fn cleanup(&self) -> Result<()> {
        Ok(())  // Default no-op
    }
    
    fn metadata(&self) -> HashMap<String, String> { /* ... */ }
    fn should_skip(&self) -> Option<String> { None }
}
```

**Implementation Examples:**
- Parity test: load Rust + C++ models, compare logits
- QK256 test: dequantize QK256 vs FP32 reference
- FFI test: verify C++ bindings work safely

---

## 4. Environment Isolation Patterns

### EnvGuard: RAII Environment Variables

**Location:** `tests/common/env.rs`

```rust
#[must_use = "EnvGuard must be held to ensure cleanup"]
pub struct EnvGuard {
    key: String,
    original: Option<String>,
}

impl EnvGuard {
    pub fn set(key: impl Into<String>, value: impl AsRef<str>) -> Self {
        let key = key.into();
        let original = std::env::var(&key).ok();
        unsafe { std::env::set_var(&key, value.as_ref()); }
        Self { key, original }
    }
    
    pub fn remove(key: impl Into<String>) -> Self {
        let key = key.into();
        let original = std::env::var(&key).ok();
        unsafe { std::env::remove_var(&key); }
        Self { key, original }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(ref value) = self.original {
                std::env::set_var(&self.key, value);
            } else {
                std::env::remove_var(&self.key);
            }
        }
    }
}

// Usage in tests
#[test]
fn test_with_env_isolation() {
    let _guard = EnvGuard::set("BITNET_DETERMINISTIC", "1");
    // Environment variable set
    // Automatically restored on drop
}
```

### Global Environment Lock

```rust
static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[must_use = "Environment guard must be held to prevent race conditions"]
pub fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.get_or_init(|| Mutex::new(())).lock().expect("env guard poisoned")
}

// Usage pattern for safe env mutation in parallel tests
#[test]
fn test_determinism_with_env_flags() {
    let _global_lock = env_guard();  // Acquire global lock
    let _guard = EnvGuard::set("BITNET_DETERMINISTIC", "1");
    // Test code here - serialized with other env-mutating tests
}
```

**Design Advantages:**
- **Safety**: RAII guards automatically restore env (even on panic)
- **Ordering**: Global lock serializes env-mutating tests
- **Clarity**: `#[must_use]` prevents accidental forgetting of guards

---

## 5. Fixture Patterns

### Fixture Facade Pattern (Feature-Gated)

**Location:** `tests/common/fixtures_facade.rs`

```rust
// With features enabled
#[cfg(feature = "fixtures")]
#[derive(Clone)]
pub struct Fixtures(pub Arc<fixtures::FixtureManager>);

// Without features (test compilation)
#[cfg(not(feature = "fixtures"))]
#[derive(Clone, Default)]
pub struct Fixtures;

impl Fixtures {
    pub async fn new(cfg: &TestConfig) -> Result<Self> {
        #[cfg(feature = "fixtures")]
        {
            let inner = fixtures::FixtureManager::new(&cfg.fixtures).await?;
            Ok(Self(Arc::new(inner)))
        }
        #[cfg(not(feature = "fixtures"))]
        Ok(Self)
    }
    
    pub async fn get_model_fixture(&self, name: &str) -> Result<PathBuf> {
        #[cfg(feature = "fixtures")]
        self.0.get_model_fixture(name).await
        
        #[cfg(not(feature = "fixtures"))]
        Err(TestError::setup("Fixtures feature not enabled"))
    }
}
```

**Benefits:**
- Single API regardless of feature flags
- Reduces `#[cfg]` scatter in test code
- Graceful degradation when fixtures disabled

### FixtureCtx Type Adaptation

```rust
// tests/common/harness/fixture_ctx.rs
#[cfg(feature = "fixtures")]
pub type FixtureCtx<'a> = &'a FixtureManager;

#[cfg(not(feature = "fixtures"))]
pub type FixtureCtx<'a> = ();

// TestCase setup can use same signature everywhere
#[async_trait]
pub trait TestCase {
    async fn setup(&self, fixtures: FixtureCtx<'_>) -> Result<()> {
        let _ = fixtures;  // Accepted whether FixtureManager or ()
        Ok(())
    }
}
```

---

## 6. Mocking Strategies

### Strategy 1: C++ Backend Availability Checking

```rust
// crossval/tests/parity_bitnetcpp.rs
use bitnet_sys::wrapper;

#[test]
fn test_with_optional_cpp_backend() {
    if !bitnet_sys::is_available() {
        eprintln!("skipping - C++ backend unavailable (set BITNET_CPP_DIR)");
        return;  // Skip test gracefully
    }
    
    // Test code that requires C++ backend
    let session = wrapper::Session::new("model.gguf")?;
}
```

### Strategy 2: Environment Variable Mocking

```rust
// crossval/tests/performance_validation.rs
#[test]
fn test_performance_with_custom_timeout() {
    let _env = EnvGuard::set("BITNET_INFERENCE_TIMEOUT", "10");
    
    // Test code runs with custom timeout
    // Automatically restored on drop
}
```

### Strategy 3: Mock Result Fixtures

```rust
// crossval/tests/parity_receipts.rs
#[tokio::test]
async fn test_parity_receipt_schema_validation() {
    // AC4: Create a receipt with parity metadata and validate schema
    let receipt = InferenceReceipt::generate(
        "cpu",
        vec!["i2s_gemv".to_string(), "rope_apply".to_string()],
    )
    .expect("Receipt generation should succeed")
    .with_parity(ParityMetadata {
        cpp_available: true,
        cosine_similarity: Some(0.9923),
        exact_match_rate: Some(1.0),
        status: "ok".to_string(),
    });
    
    assert_eq!(receipt.schema_version, "1.0.0");
}
```

### Strategy 4: Synthetic Data Generators

```rust
// crossval/tests/qk256_crossval.rs
fn test_qk256_vs_fp32_reference_small_matrix() -> Result<()> {
    let rows = 8;
    let cols = 256;
    
    // Generate deterministic test data
    let codes: Vec<u8> = (0..total_codes).map(|i| (i % 4) as u8).collect();
    let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01).sin()).collect();
    
    // Compare QK256 vs FP32 reference
    let mut qk256_output = vec![0.0f32; rows];
    gemv_qk256(&packed_data, &input, &mut qk256_output, rows, cols, PACKED_BYTES)?;
    
    let mut fp32_output = vec![0.0f32; rows];
    gemv_fp32_reference(&codes, &input, &mut fp32_output, rows, cols)?;
    
    // Verify tolerance
    assert_tolerance(&qk256_output, &fp32_output, 1e-4);
    Ok(())
}
```

---

## 7. CI Test Configuration

### Nextest Profiles (6 configurations)

**Location:** `.config/nextest.toml`

```toml
# Profile 1: Default (development)
[profile.default]
fail-fast = true
test-threads = "num-cpus"
retries = 0
slow-timeout = { period = "300s", terminate-after = 1 }
failure-output = "immediate"
success-output = "never"
[profile.default.junit]
path = "target/nextest/junit.xml"

# Profile 2: CI (fixed thread count)
[profile.ci]
fail-fast = false
test-threads = 4                           # Fixed for reproducibility
retries = 0
slow-timeout = { period = "300s", terminate-after = 1 }
failure-output = "immediate"
success-output = "never"
[profile.ci.junit]
path = "target/nextest/ci/junit.xml"

# Profile 3: Fixtures (GGUF loading heavy)
[profile.fixtures]
fail-fast = false
test-threads = 2                           # Lower to reduce I/O contention
retries = 0
slow-timeout = { period = "600s", terminate-after = 1 }  # Longer for model loading
[profile.fixtures.junit]
path = "target/nextest/fixtures/junit.xml"

# Profile 4: GPU (memory constrained)
[profile.gpu]
fail-fast = false
test-threads = 1                           # Single thread - GPU memory limits
retries = 0
slow-timeout = { period = "300s", terminate-after = 1 }
[profile.gpu.junit]
path = "target/nextest/gpu/junit.xml"

# Profile 5: Doctests (simpler examples)
[profile.doctests]
fail-fast = false
test-threads = "num-cpus"
retries = 0
slow-timeout = { period = "120s", terminate-after = 1 }  # Shorter timeout
[profile.doctests.junit]
path = "target/nextest/doctests/junit.xml"
```

**Usage:**
```bash
# Development (default profile)
cargo nextest run --workspace --no-default-features --features cpu

# CI with fixed threads
cargo nextest run --profile ci --workspace --no-default-features --features cpu

# Fixture-heavy tests
cargo nextest run --profile fixtures --package bitnet-models --features fixtures

# GPU tests (serial execution)
cargo nextest run --profile gpu --features gpu --test kernel_tests
```

### CI Workflows (7 files)

| Workflow | Trigger | Profile | Scenarios |
|----------|---------|---------|-----------|
| `crossval-fast.yml` | Nightly + manual + `crossval` label | CI | Token equivalence, performance |
| `testing-framework-crossval.yml` | Nightly + manual + `crossval` label | CI (3 scenarios per OS) | Accuracy, performance, edge-cases |
| `verify-receipts.yml` | PR, schedule | CI | Schema v1.0.0, 8 validation gates |
| `property-tests.yml` | PR, schedule | CI | Property-based quantization tests |
| `comprehensive-build-test.yml` | PR push | CI | Multi-feature matrix |
| `validation.yml` | PR, schedule | CI | Model validation gates |
| `gpu.yml` | PR labeled `gpu-test` | GPU profile | CUDA kernel tests |

### Example: Full Crossval Workflow

```bash
# .github/workflows/testing-framework-crossval.yml (simplified)
jobs:
  check-trigger:
    # Only run on: nightly schedule, manual dispatch, or 'crossval' label on PR
    
  setup-cpp-implementation:
    # Provision BitNet.cpp from cache (GitHub Actions cache)
    
  cross-validation-tests:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        test_scenario:
          - accuracy-comparison
          - performance-benchmark
          - edge-case-validation
    steps:
      - Build Rust with crossval features
      - Run scenario-specific tests
      - Collect artifacts (traces, logs, metrics)
      - Generate comparison report
      
  cross-validation-summary:
    # Aggregate results from all platforms
    # Post PR comment with summary
    # Create tracking issue on failure
```

---

## 8. Serial Test Patterns

### Serial Execution with serial_test Crate

**Note:** Current codebase uses `EnvGuard` + global `env_guard()` lock instead of `serial_test`, but here's the pattern for reference:

```rust
#[cfg(test)]
mod tests {
    use serial_test::serial;
    
    #[test]
    #[serial]
    fn test_env_determinism_1() {
        // Executed serially with other tests marked #[serial]
        let _guard = EnvGuard::set("BITNET_SEED", "1");
    }
    
    #[test]
    #[serial(bitnet_env)]
    fn test_env_determinism_2() {
        // Executed serially with other tests in group "bitnet_env"
        let _guard = EnvGuard::set("BITNET_DETERMINISTIC", "1");
    }
}
```

**Current Pattern in BitNet-rs:** Global `env_guard()` lock prevents race conditions

```rust
#[test]
fn test_with_env_isolation() {
    let _global_lock = env_guard();  // Acquire global lock (implicit serialization)
    let _env_guard = EnvGuard::set("BITNET_DETERMINISTIC", "1");
    // Test code...
}
```

---

## 9. Template for New Integration Tests

### Template: Backend-Specific Parity Test

```rust
// file: crossval/tests/new_feature_parity.rs

//! Feature Name: [Feature Description]
//!
//! Tests feature spec: docs/explanation/[feature-spec].md
//! API contract: docs/explanation/specs/[feature-contract].md
//!
//! This test validates [what it validates].

#![cfg(all(test, feature = "crossval"))]

use anyhow::Result;
use bitnet_inference::{eval_logits_once, get_model_config};
use bitnet_models::GgufReader;
use bitnet_sys::wrapper::{self, Session as CppSession};
use bitnet_tokenizers::loader::load_tokenizer_from_gguf_reader;

#[cfg(all(test, feature = "crossval"))]
mod tests {
    use super::*;
    
    // ==================== Setup & Helpers ====================
    
    /// Returns model path if C++ backend available
    fn test_model_path() -> Option<String> {
        if !bitnet_sys::is_available() {
            eprintln!("skipping - C++ backend unavailable (set BITNET_CPP_DIR)");
            return None;
        }
        
        match std::env::var("CROSSVAL_GGUF") {
            Ok(path) => Some(path),
            Err(_) => {
                eprintln!("skipping - set CROSSVAL_GGUF to path of test model");
                None
            }
        }
    }
    
    const TOLERANCE: f32 = 1e-4;
    
    fn compare_outputs(rust_output: &[f32], cpp_output: &[f32], step: usize) -> Result<()> {
        if rust_output.len() != cpp_output.len() {
            anyhow::bail!("Length mismatch at step {}", step);
        }
        
        for (i, (r, c)) in rust_output.iter().zip(cpp_output).enumerate() {
            let diff = (r - c).abs();
            if diff > TOLERANCE {
                anyhow::bail!(
                    "Step {}: Difference {} at index {} exceeds tolerance {}",
                    step, diff, i, TOLERANCE
                );
            }
        }
        Ok(())
    }
    
    // ==================== Test Cases ====================
    
    /// AC{N}: Basic functionality test
    #[test]
    fn test_basic_functionality() {
        let Some(model_path) = test_model_path() else {
            return;
        };
        
        // Load Rust implementation
        let reader = GgufReader::open(&model_path).expect("load Rust model");
        
        // Load C++ implementation
        let session = CppSession::new(&model_path).expect("load C++ model");
        
        assert!(reader.is_valid(), "Rust model should be valid");
        assert!(session.is_ready(), "C++ model should be ready");
    }
    
    /// AC{N}: Parity test with tolerance
    #[test]
    fn test_parity_with_tolerance() {
        let Some(model_path) = test_model_path() else {
            return;
        };
        
        let prompt = "Hello world";
        
        // Evaluate Rust implementation
        let rust_logits = eval_logits_once(&model_path, prompt)
            .expect("Rust eval should succeed");
        
        // Evaluate C++ implementation
        let cpp_logits = CppSession::eval_logits(&model_path, prompt)
            .expect("C++ eval should succeed");
        
        // Compare with tolerance
        compare_outputs(&rust_logits, &cpp_logits, 1)
            .expect("Outputs should match within tolerance");
    }
    
    /// AC{N}: Multi-token scenario
    #[test]
    fn test_multi_token_scenario() {
        let Some(model_path) = test_model_path() else {
            return;
        };
        
        let prompt = "The quick brown fox";
        let max_tokens = 4;
        
        // Run both implementations
        for step in 0..max_tokens {
            let partial_prompt = prompt.split_whitespace()
                .take(2 + step)
                .collect::<Vec<_>>()
                .join(" ");
            
            let rust_logits = eval_logits_once(&model_path, &partial_prompt)
                .expect("Rust eval");
            let cpp_logits = CppSession::eval_logits(&model_path, &partial_prompt)
                .expect("C++ eval");
            
            compare_outputs(&rust_logits, &cpp_logits, step + 1)
                .expect(&format!("Step {} should match", step + 1));
        }
    }
    
    /// AC{N}: Performance regression check
    #[test]
    fn test_no_performance_regression() {
        let Some(model_path) = test_model_path() else {
            return;
        };
        
        use std::time::Instant;
        
        let prompt = "Performance test";
        let iterations = 10;
        
        // Benchmark Rust
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = eval_logits_once(&model_path, prompt);
        }
        let rust_time = start.elapsed();
        
        // Benchmark C++
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = CppSession::eval_logits(&model_path, prompt);
        }
        let cpp_time = start.elapsed();
        
        let ratio = rust_time.as_secs_f64() / cpp_time.as_secs_f64();
        println!("Rust/C++ time ratio: {:.2}x", ratio);
        
        // Rust should not be dramatically slower
        assert!(ratio < 2.0, "Rust should not be >2x slower than C++");
    }
}
```

### Template: Fixture-Based Quantization Test

```rust
// file: tests/quantization_parity_test.rs

#![cfg(all(test, feature = "fixtures", feature = "crossval"))]

use tests::prelude::*;

#[test]
fn test_quantization_with_fixtures() {
    // Use fixture system if available
    #[cfg(feature = "fixtures")]
    {
        let fixtures = Fixtures::default();
        
        // Get test model from fixtures
        let model_path = futures::executor::block_on(async {
            fixtures.get_model_fixture("qk256-small").await
        }).expect("fixture should load");
        
        // Test quantization parity
        let result = test_qk256_vs_reference(&model_path);
        assert!(result.is_ok());
    }
    
    #[cfg(not(feature = "fixtures"))]
    {
        println!("Skipping fixture-based test (compile without --features fixtures)");
    }
}

async fn test_qk256_vs_reference(model_path: &str) -> anyhow::Result<()> {
    // Test implementation
    Ok(())
}
```

### Template: Environment-Isolated Test

```rust
// file: tests/env_isolated_test.rs

#[test]
fn test_with_deterministic_env() {
    // Acquire global lock to serialize env mutations
    let _global_lock = tests::common::env::env_guard();
    
    // Use EnvGuard for automatic restoration
    let _seed_guard = tests::common::env::EnvGuard::set("BITNET_SEED", "42");
    let _determ_guard = tests::common::env::EnvGuard::set("BITNET_DETERMINISTIC", "1");
    
    // Run deterministic test
    let result = inference_with_seed();
    
    // Env vars automatically restored on drop
    assert!(result.is_ok());
}

// Multiple tests can safely run in parallel with env_guard
#[test]
fn test_with_different_seed() {
    let _global_lock = tests::common::env::env_guard();
    let _seed_guard = tests::common::env::EnvGuard::set("BITNET_SEED", "100");
    
    // Different seed, different result expected
    let result = inference_with_seed();
    assert!(result.is_ok());
}
```

---

## 10. Dual-Backend Test Strategy

### Backend Selection Pattern

```rust
// Test can run on both CPU and GPU backends
#[cfg(any(feature = "cpu", feature = "gpu"))]
#[test]
fn test_parity_dual_backend() {
    // Code that works on both backends
}

// GPU-specific test
#[cfg(all(feature = "gpu", test))]
#[test]
fn test_parity_gpu_kernels() {
    // GPU-specific test code
}

// CPU-specific test
#[cfg(all(feature = "cpu", test))]
#[test]
fn test_parity_cpu_simd() {
    // CPU-specific SIMD test code
}
```

### Feature Matrix in CI

```yaml
# .github/workflows/testing-framework-crossval.yml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest]
    features:
      - "cpu,avx2"
      - "cpu,neon"
      - "gpu"
    test_scenario:
      - accuracy-comparison
      - performance-benchmark
      - edge-case-validation
```

---

## 11. Key Infrastructure Components Summary

| Component | Location | Purpose | Key Features |
|-----------|----------|---------|--------------|
| **TestHarness** | `tests/common/harness.rs` | Async parallel test execution | Semaphore control, isolation, timeouts, reporting |
| **EnvGuard** | `tests/common/env.rs` | Safe env var mutation in tests | RAII cleanup, global lock, typed parsing |
| **Fixtures** | `tests/common/fixtures_facade.rs` | Feature-gated model/data access | Facade pattern, lazy loading, cache stats |
| **Nextest** | `.config/nextest.toml` | Test runner config | 6 profiles, timeout protection, JUnit output |
| **CI Workflows** | `.github/workflows/*.yml` | Test orchestration | Label-triggered, caching, artifact collection |
| **CrossValidation** | `tests/common/cross_validation/` | Rust vs C++ comparison | Implementation registry, test runners, metrics |
| **Crossval Tests** | `crossval/tests/*.rs` | Parity & integration tests | 13 test suites, feature-gated, model fixtures |

---

## 12. Adding a New Backend-Specific Test

### Step-by-Step Guide

**1. Create test file in `crossval/tests/`:**
```rust
// crossval/tests/new_backend_test.rs
#![cfg(all(test, feature = "crossval"))]

mod tests {
    // Implementation
}
```

**2. Declare test in `crossval/Cargo.toml`:**
```toml
[[test]]
name = "new_backend_test"
path = "tests/new_backend_test.rs"
required-features = ["crossval", "integration-tests"]
```

**3. Use model fixture pattern:**
```rust
fn test_model_path() -> Option<String> {
    if !bitnet_sys::is_available() {
        eprintln!("skipping - C++ backend unavailable");
        return None;
    }
    std::env::var("CROSSVAL_GGUF").ok()
}
```

**4. Add to CI workflow:**
```yaml
# .github/workflows/testing-framework-crossval.yml
- name: Run new backend tests
  run: |
    cargo test --package crossval \
      --features crossval,cpu \
      new_backend_test
```

**5. Run locally:**
```bash
# Setup environment
export CROSSVAL_GGUF=path/to/model.gguf
export BITNET_CPP_DIR=/path/to/bitnet.cpp

# Run test
cargo test --package crossval --features crossval new_backend_test -- --nocapture
```

---

## 13. Debugging Failed Tests

### Enabling Verbose Output

```bash
# Show test output
cargo test --package crossval --features crossval -- --nocapture

# With logging
RUST_LOG=debug cargo test --package crossval -- --nocapture

# With backtrace
RUST_BACKTRACE=1 cargo test --package crossval -- --nocapture
```

### Collecting Test Artifacts

```bash
# Run with trace collection
cargo run -p xtask -- trace-diff /tmp/rs_traces /tmp/cpp_traces

# Inspect receipt files
find target -name "*.json" -path "*/crossval/*" -exec cat {} \; | jq .
```

### Skipping Slow Tests

```bash
# Skip scalar QK256 tests
BITNET_SKIP_SLOW_TESTS=1 cargo test --workspace --no-default-features --features cpu

# Use nextest with fast-only filter
cargo nextest run --profile ci --filter-expr 'not slow'
```

---

## Conclusion

BitNet-rs's test infrastructure provides:

1. **Comprehensive Coverage**: 13 crossval test suites covering parity, performance, and edge cases
2. **Robust Isolation**: EnvGuard + global locks prevent test interference
3. **Parallel Execution**: Semaphore-based concurrency control with per-test timeouts
4. **Feature Flexibility**: Feature-gated compilation supports CPU, GPU, and FFI variants
5. **CI Integration**: 7 workflows with caching, artifact collection, and PR commenting
6. **Developer Experience**: Clear patterns for adding new tests, fixtures, and mocking strategies

The test infrastructure is production-ready and well-suited for dual-backend validation scenarios.
