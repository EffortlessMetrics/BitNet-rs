# Validation Infrastructure Improvements Specification

**Status**: Draft
**Created**: 2025-10-16
**Audience**: BitNet.rs developers implementing validation and benchmarking infrastructure
**Type**: Explanation (Diátaxis)
**Related Issues**: #261 (mock elimination), #439 (GPU feature gates)

## Executive Summary

This specification defines a comprehensive improvement plan for BitNet.rs validation infrastructure, transforming the current mock-based system into a production-grade validation framework with real models, honest compute verification, and systematic parity testing against the C++ reference implementation.

**Core Objectives:**
1. **Model Fetcher**: Lockfile-based model provisioning with SHA-256 verification
2. **Real llama.cpp Bridge**: Replace mock C++ wrapper with minimal FFI surface
3. **Parity Harness**: Systematic tokenization, logits, and multi-step validation
4. **Real Benchmarks**: Replace fabricated TPS with measured inference performance
5. **Production Readiness**: Remove mocks, fail-fast on errors, honest compute enforcement

**Scope Boundaries:**
- **In Scope**: Model fetching, C++ bridge, parity tests, benchmarks, mock removal
- **Out of Scope**: GPU-specific validation (CPU MVP focus), model training, format converters
- **Feature Flag**: `crossval-cpp` for C++ bridge (optional, gated by `BITNET_CPP_DIR`)

## Problem Statement

### Current State Analysis

**Receipt System (Implemented PR #452):**
- ✅ KernelRecorder infrastructure in `bitnet-inference`
- ✅ Production receipts with measured TPS and real kernel IDs
- ✅ Schema validation (v1.0.0) and honest compute gates
- ✅ CI workflow enforcement via `.github/workflows/model-gates.yml`

**Cross-Validation Infrastructure (Current Gaps):**
- ❌ C++ wrapper is intentionally mocked (`crossval/src/bitnet_cpp_wrapper.c`)
- ❌ Benchmarks use fabricated data (`crossval/benches/performance.rs`)
- ❌ No model fetcher - users manually download models
- ❌ Tests skip when fixtures unavailable (silent failures)
- ✅ All Rust inference components are REAL and working

**Impact of Current Limitations:**
1. **No Parity Validation**: Cannot verify Rust output matches C++ reference
2. **Misleading Baselines**: Fabricated TPS numbers create false performance expectations
3. **Manual Setup Friction**: Developers must manually provision models for testing
4. **Silent Test Skips**: Tests pass when fixtures missing, masking validation gaps
5. **No Regression Detection**: Cannot track performance changes over time

### User Stories

**As a BitNet.rs contributor**, I want automated model provisioning so that I can run cross-validation tests without manual setup steps.

**As a CI pipeline**, I want to verify numerical parity against C++ reference so that regressions are caught before merge.

**As a performance engineer**, I want real benchmark measurements so that baselines reflect actual inference performance.

**As a release manager**, I want fail-fast model validation so that broken models are detected immediately with actionable guidance.

## Architecture Overview

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Validation Infrastructure                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │ Model Fetcher│─────▶│ Parity       │─────▶│ Real         │  │
│  │ (xtask)      │      │ Harness      │      │ Benchmarks   │  │
│  │              │      │ (crossval)   │      │ (crossval)   │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                      │                      │          │
│         │                      │                      │          │
│         ▼                      ▼                      ▼          │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │ Lock File    │      │ llama.cpp    │      │ Receipt      │  │
│  │ (.lock.json) │      │ Bridge (FFI) │      │ Generation   │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                      │                      │          │
│         │                      │                      │          │
│         ▼                      ▼                      ▼          │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │ ~/.cache/    │      │ Rust         │      │ Baseline     │  │
│  │ bitnet/      │      │ Inference    │      │ Validation   │  │
│  │ models/      │      │ Engine       │      │ (xtask)      │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow: Model Fetching and Validation

```
┌────────────┐    fetch-models     ┌────────────┐    verify SHA-256    ┌────────────┐
│ xtask CLI  │──────────────────▶ │ Lock File  │────────────────────▶│ ~/.cache/  │
└────────────┘                     │ Parser     │                      │ bitnet/    │
                                   └────────────┘                      └────────────┘
                                          │                                   │
                                          │                                   │
                                          ▼                                   ▼
                                   ┌────────────┐                      ┌────────────┐
                                   │ Download   │─────────────────────▶│ Model      │
                                   │ Manager    │   HTTP w/ progress   │ Storage    │
                                   └────────────┘                      └────────────┘
                                          │
                                          │
                                          ▼
                                   ┌────────────┐    run parity       ┌────────────┐
                                   │ Parity     │────────────────────▶│ C++ Bridge │
                                   │ Harness    │                      │ (FFI)      │
                                   └────────────┘                      └────────────┘
                                          │                                   │
                                          │                                   │
                                          ▼                                   ▼
                                   ┌────────────┐                      ┌────────────┐
                                   │ Rust       │◀─────compare────────│ C++        │
                                   │ Inference  │    logits/tokens    │ Reference  │
                                   └────────────┘                      └────────────┘
                                          │
                                          │
                                          ▼
                                   ┌────────────┐    emit receipt     ┌────────────┐
                                   │ Parity     │────────────────────▶│ ci/        │
                                   │ Receipt    │    cosine ≥0.99     │ parity.json│
                                   └────────────┘                      └────────────┘
```

### System Integration Points

| Component | Integration Point | Purpose |
|-----------|------------------|---------|
| Model Fetcher | `xtask/src/fetch_models.rs` | CLI command for model provisioning |
| Lock File | `crossval-models.lock.json` | Model registry with SHA-256 hashes |
| Cache Directory | `~/.cache/bitnet/models/<sha256>/` | Persistent model storage |
| C++ Bridge | `crossval/src/cpp_bridge.rs` | Minimal FFI to llama.cpp |
| C Wrapper | `crossval/src/llama_cpp_ffi.c` | Extern C functions for FFI |
| Parity Harness | `crossval/src/parity.rs` | Validation test framework |
| Benchmarks | `crossval/benches/real_performance.rs` | Measured inference TPS |
| Baselines | `docs/baselines/` | Performance fingerprints |
| CI Workflow | `.github/workflows/crossval.yml` | Automated validation |

## Feature Requirements

### 1. Model Fetcher (`xtask fetch-models`)

**Goal**: Automated, reproducible model provisioning with cryptographic verification.

**Acceptance Criteria:**
- AC1: `cargo run -p xtask -- fetch-models` downloads models from lockfile
- AC2: SHA-256 verification prevents corrupted/tampered models
- AC3: Models cached to `~/.cache/bitnet/models/<sha256>/` (no git storage)
- AC4: Two-tier strategy: CI-light (tiny model ~50MB) + integration (real model ~2GB)
- AC5: Progress bars for downloads (via `indicatif` crate)
- AC6: Idempotent: skip download if SHA-256 matches cached model

**Lockfile Schema (`crossval-models.lock.json`):**
```json
{
  "schema_version": "1.0.0",
  "models": [
    {
      "name": "ci-light",
      "tier": "ci",
      "url": "https://huggingface.co/bitnet-community/tiny-bitnet-test/resolve/main/model.gguf",
      "sha256": "abc123...",
      "size_bytes": 52428800,
      "description": "Tiny model for fast CI validation (~50MB)"
    },
    {
      "name": "integration",
      "tier": "integration",
      "url": "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf",
      "sha256": "def456...",
      "size_bytes": 2147483648,
      "description": "Production model for full validation (~2GB)"
    }
  ],
  "updated": "2025-10-16T00:00:00Z"
}
```

**CLI Interface:**
```bash
# Fetch all models (respects tier filter from CI)
cargo run -p xtask -- fetch-models

# Fetch specific tier
cargo run -p xtask -- fetch-models --tier ci

# Verify cached models without downloading
cargo run -p xtask -- fetch-models --verify-only

# Force re-download (ignore cache)
cargo run -p xtask -- fetch-models --force
```

**Implementation Notes:**
- Use `reqwest` for HTTP downloads with retry logic
- Cache directory: `$HOME/.cache/bitnet/models/<sha256>/model.gguf`
- Lock file location: workspace root `crossval-models.lock.json`
- Environment override: `BITNET_MODEL_CACHE_DIR` for custom cache location
- Download resume: partial downloads stored as `model.gguf.partial`

**Error Handling:**
```rust
pub enum FetchError {
    NetworkError(String),           // HTTP failures, timeouts
    ChecksumMismatch { expected: String, actual: String },
    CacheWriteError(std::io::Error),
    LockfileParseError(serde_json::Error),
}
```

**File Structure:**
```
~/.cache/bitnet/
├── models/
│   ├── abc123.../
│   │   ├── model.gguf          # Verified model
│   │   └── .metadata.json      # Download timestamp, source URL
│   └── def456.../
│       ├── model.gguf
│       └── .metadata.json
└── .lock                        # Cache-level lockfile for concurrency
```

### 2. Real llama.cpp Bridge

**Goal**: Replace mock C++ wrapper with minimal FFI surface to llama.cpp for parity validation.

**Acceptance Criteria:**
- AC7: Feature-gated `crossval-cpp` requires `BITNET_CPP_DIR` env variable
- AC8: Minimal FFI surface: `model_new`, `context_new`, `tokenize`, `eval`
- AC9: Build script (`crossval/build.rs`) links against llama.cpp if present
- AC10: Tests skip gracefully if `crossval-cpp` feature disabled
- AC11: Memory safety: proper cleanup with `Drop` implementations

**FFI Surface (C API):**
```c
// crossval/src/llama_cpp_ffi.c
#include <llama.h>

typedef struct llama_model_wrapper {
    struct llama_model* model;
    struct llama_context* ctx;
} llama_model_wrapper_t;

// Create model and context
llama_model_wrapper_t* llama_cpp_create_model(const char* model_path);

// Tokenize input
int llama_cpp_tokenize(
    llama_model_wrapper_t* wrapper,
    const char* text,
    int* tokens_out,
    int max_tokens
);

// Evaluate and get logits
int llama_cpp_eval(
    llama_model_wrapper_t* wrapper,
    const int* tokens,
    int n_tokens,
    float* logits_out,
    int max_logits
);

// Cleanup
void llama_cpp_destroy_model(llama_model_wrapper_t* wrapper);
```

**Rust FFI Bindings:**
```rust
// crossval/src/cpp_bridge.rs
#[cfg(feature = "crossval-cpp")]
mod ffi {
    use std::os::raw::{c_char, c_int, c_float};

    #[repr(C)]
    pub struct LlamaModelWrapper {
        _private: [u8; 0],  // Opaque type
    }

    extern "C" {
        pub fn llama_cpp_create_model(path: *const c_char) -> *mut LlamaModelWrapper;
        pub fn llama_cpp_tokenize(
            wrapper: *mut LlamaModelWrapper,
            text: *const c_char,
            tokens_out: *mut c_int,
            max_tokens: c_int,
        ) -> c_int;
        pub fn llama_cpp_eval(
            wrapper: *mut LlamaModelWrapper,
            tokens: *const c_int,
            n_tokens: c_int,
            logits_out: *mut c_float,
            max_logits: c_int,
        ) -> c_int;
        pub fn llama_cpp_destroy_model(wrapper: *mut LlamaModelWrapper);
    }
}

#[cfg(feature = "crossval-cpp")]
pub struct CppModel {
    wrapper: *mut ffi::LlamaModelWrapper,
}

#[cfg(feature = "crossval-cpp")]
impl CppModel {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let path_cstr = std::ffi::CString::new(path.to_string_lossy().as_ref())?;
        let wrapper = unsafe { ffi::llama_cpp_create_model(path_cstr.as_ptr()) };

        if wrapper.is_null() {
            return Err(CrossvalError::ModelLoadError("C++ model load failed".into()));
        }

        Ok(Self { wrapper })
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        // Implementation
    }

    pub fn eval(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        // Implementation
    }
}

#[cfg(feature = "crossval-cpp")]
impl Drop for CppModel {
    fn drop(&mut self) {
        unsafe { ffi::llama_cpp_destroy_model(self.wrapper) }
    }
}
```

**Build Configuration (`crossval/build.rs`):**
```rust
fn main() {
    #[cfg(feature = "crossval-cpp")]
    {
        let cpp_dir = std::env::var("BITNET_CPP_DIR")
            .expect("BITNET_CPP_DIR required for crossval-cpp feature");

        println!("cargo:rustc-link-search=native={}/build", cpp_dir);
        println!("cargo:rustc-link-lib=static=llama");
        println!("cargo:rustc-link-lib=dylib=stdc++");

        cc::Build::new()
            .file("src/llama_cpp_ffi.c")
            .include(format!("{}/include", cpp_dir))
            .compile("llama_cpp_ffi");
    }
}
```

**Environment Setup:**
```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Set environment for BitNet.rs
export BITNET_CPP_DIR=$(pwd)/..
cargo build -p crossval --features crossval-cpp
```

**Feature Flag Documentation:**
```rust
// crossval/Cargo.toml
[features]
default = []
crossval = []              # Enable cross-validation framework (no C++ required)
crossval-cpp = ["crossval"] # Enable C++ bridge (requires BITNET_CPP_DIR)
```

### 3. Parity Harness

**Goal**: Systematic validation of Rust inference against C++ reference implementation.

**Acceptance Criteria:**
- AC12: Tokenization exact match test (Rust == C++ token IDs)
- AC13: Logits cosine similarity ≥0.99 for first-token inference
- AC14: Multi-step greedy decode produces identical token sequences
- AC15: Parity receipt emitted with model SHA, commit, cosine scores
- AC16: Tests feature-gated with `crossval-cpp` (skip if disabled)

**Parity Test Structure:**
```rust
// crossval/src/parity.rs
use crate::{CppModel, Result};
use bitnet_inference::InferenceEngine;

pub struct ParityTest {
    rust_engine: InferenceEngine,
    cpp_model: CppModel,
    model_sha256: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ParityReceipt {
    pub schema_version: String,           // "1.0.0"
    pub timestamp: String,                // RFC3339
    pub model_sha256: String,             // Model fingerprint
    pub bitnet_commit: String,            // Git commit hash
    pub test_suite: String,               // "tokenization|logits|greedy"
    pub status: String,                   // "pass|fail"
    pub metrics: ParityMetrics,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ParityMetrics {
    pub tokenization_exact_match: bool,
    pub logits_cosine_similarity: f64,
    pub greedy_exact_match: bool,
    pub inference_time_rust_ms: f64,
    pub inference_time_cpp_ms: f64,
}

impl ParityTest {
    pub fn new(model_path: &Path) -> Result<Self> {
        // Load Rust inference engine
        let rust_engine = InferenceEngine::new(model_path)?;

        // Load C++ reference model
        let cpp_model = CppModel::load(model_path)?;

        // Compute model SHA-256
        let model_sha256 = compute_file_sha256(model_path)?;

        Ok(Self { rust_engine, cpp_model, model_sha256 })
    }

    /// Test 1: Tokenization exact match
    pub fn test_tokenization(&self, prompt: &str) -> Result<bool> {
        let rust_tokens = self.rust_engine.tokenize(prompt)?;
        let cpp_tokens = self.cpp_model.tokenize(prompt)?;

        Ok(rust_tokens == cpp_tokens)
    }

    /// Test 2: Logits cosine similarity
    pub fn test_logits(&self, prompt: &str) -> Result<f64> {
        let tokens = self.rust_engine.tokenize(prompt)?;

        let rust_logits = self.rust_engine.eval(&tokens)?;
        let cpp_logits = self.cpp_model.eval(&tokens)?;

        let cosine = cosine_similarity(&rust_logits, &cpp_logits);
        Ok(cosine)
    }

    /// Test 3: Multi-step greedy decode
    pub fn test_greedy_decode(&mut self, prompt: &str, max_tokens: usize) -> Result<bool> {
        let rust_tokens = self.rust_engine.generate_greedy(prompt, max_tokens)?;
        let cpp_tokens = self.cpp_model.generate_greedy(prompt, max_tokens)?;

        Ok(rust_tokens == cpp_tokens)
    }

    /// Run full parity suite and emit receipt
    pub fn run_full_suite(&mut self, prompts: &[&str]) -> Result<ParityReceipt> {
        let mut metrics = ParityMetrics {
            tokenization_exact_match: true,
            logits_cosine_similarity: 1.0,
            greedy_exact_match: true,
            inference_time_rust_ms: 0.0,
            inference_time_cpp_ms: 0.0,
        };

        // Test tokenization for all prompts
        for prompt in prompts {
            if !self.test_tokenization(prompt)? {
                metrics.tokenization_exact_match = false;
                break;
            }
        }

        // Test logits (min cosine across prompts)
        let mut min_cosine = 1.0;
        for prompt in prompts {
            let cosine = self.test_logits(prompt)?;
            min_cosine = min_cosine.min(cosine);
        }
        metrics.logits_cosine_similarity = min_cosine;

        // Test greedy decode
        if !self.test_greedy_decode(prompts[0], 10)? {
            metrics.greedy_exact_match = false;
        }

        let status = if metrics.tokenization_exact_match
            && metrics.logits_cosine_similarity >= 0.99
            && metrics.greedy_exact_match
        {
            "pass"
        } else {
            "fail"
        };

        Ok(ParityReceipt {
            schema_version: "1.0.0".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            model_sha256: self.model_sha256.clone(),
            bitnet_commit: get_git_commit_hash()?,
            test_suite: "full".to_string(),
            status: status.to_string(),
            metrics,
        })
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());

    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();

    dot_product / (norm_a * norm_b)
}
```

**Test Integration:**
```rust
// crossval/tests/parity.rs
#[cfg(feature = "crossval-cpp")]
#[test]
fn test_parity_tokenization() {
    // AC:12 - Tokenization exact match
    let model_path = get_test_model_path();
    let parity = ParityTest::new(&model_path).unwrap();

    let prompts = ["Hello world", "The quick brown fox"];
    for prompt in &prompts {
        assert!(
            parity.test_tokenization(prompt).unwrap(),
            "Tokenization mismatch for: {}",
            prompt
        );
    }
}

#[cfg(feature = "crossval-cpp")]
#[test]
fn test_parity_logits() {
    // AC:13 - Logits cosine similarity ≥0.99
    let model_path = get_test_model_path();
    let parity = ParityTest::new(&model_path).unwrap();

    let cosine = parity.test_logits("Hello world").unwrap();
    assert!(
        cosine >= 0.99,
        "Logits diverged: cosine similarity {:.4} < 0.99",
        cosine
    );
}

#[cfg(feature = "crossval-cpp")]
#[test]
fn test_parity_greedy_decode() {
    // AC:14 - Multi-step greedy decode exact match
    let model_path = get_test_model_path();
    let mut parity = ParityTest::new(&model_path).unwrap();

    assert!(
        parity.test_greedy_decode("Once upon a time", 10).unwrap(),
        "Greedy decode mismatch"
    );
}

#[cfg(feature = "crossval-cpp")]
#[test]
fn test_parity_receipt_generation() {
    // AC:15 - Parity receipt emitted
    let model_path = get_test_model_path();
    let mut parity = ParityTest::new(&model_path).unwrap();

    let prompts = ["Hello world", "Test prompt"];
    let receipt = parity.run_full_suite(&prompts).unwrap();

    assert_eq!(receipt.schema_version, "1.0.0");
    assert_eq!(receipt.status, "pass");
    assert!(receipt.metrics.logits_cosine_similarity >= 0.99);

    // Write receipt for CI
    let receipt_path = std::path::Path::new("ci/parity.json");
    std::fs::write(
        receipt_path,
        serde_json::to_string_pretty(&receipt).unwrap()
    ).unwrap();
}
```

**Parity Receipt Example:**
```json
{
  "schema_version": "1.0.0",
  "timestamp": "2025-10-16T12:00:00Z",
  "model_sha256": "abc123...",
  "bitnet_commit": "a1b2c3d",
  "test_suite": "full",
  "status": "pass",
  "metrics": {
    "tokenization_exact_match": true,
    "logits_cosine_similarity": 0.9987,
    "greedy_exact_match": true,
    "inference_time_rust_ms": 12.3,
    "inference_time_cpp_ms": 15.7
  }
}
```

### 4. Real Benchmarks

**Goal**: Replace fabricated TPS numbers with measured Rust inference performance.

**Acceptance Criteria:**
- AC17: Benchmark measures real `bitnet-inference` engine (no mocks)
- AC18: Emit bench receipts with measured TPS, latency, kernel IDs
- AC19: `xtask gen-baselines` regenerates from receipts
- AC20: Regression detection: fail if TPS drops >10% from baseline

**Benchmark Implementation:**
```rust
// crossval/benches/real_performance.rs
use bitnet_inference::{InferenceEngine, KernelRecorder};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::time::Duration;

fn benchmark_rust_inference(c: &mut Criterion) {
    let model_path = get_benchmark_model_path();
    let mut engine = InferenceEngine::new(&model_path)
        .expect("Failed to load model for benchmarking");

    let mut group = c.benchmark_group("rust_inference");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    let prompts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
    ];

    for prompt in &prompts {
        let token_count = engine.tokenize(prompt).unwrap().len();
        group.throughput(Throughput::Elements(token_count as u64));

        group.bench_function(format!("generate_{}_tokens", token_count), |b| {
            b.iter(|| {
                let recorder = KernelRecorder::new();
                engine.set_kernel_recorder(Some(recorder.clone()));

                let start = std::time::Instant::now();
                let output = engine.generate(prompt, 32).expect("Generation failed");
                let duration = start.elapsed();

                let kernels = recorder.get_kernel_ids();
                let tps = 32.0 / duration.as_secs_f64();

                // Emit receipt for baseline
                emit_bench_receipt(prompt, tps, &kernels, duration);

                output
            });
        });
    }

    group.finish();
}

fn emit_bench_receipt(
    prompt: &str,
    tps: f64,
    kernels: &[String],
    latency: Duration
) {
    let receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "compute_path": "real",
        "backend": "cpu",
        "prompt": prompt,
        "tokens_per_second": tps,
        "latency_ms": latency.as_millis(),
        "kernels": kernels,
    });

    let receipt_path = format!(
        "ci/bench_{}.json",
        prompt.chars().filter(|c| c.is_alphanumeric()).take(20).collect::<String>()
    );

    std::fs::write(
        &receipt_path,
        serde_json::to_string_pretty(&receipt).unwrap()
    ).ok();
}

criterion_group!(benches, benchmark_rust_inference);
criterion_main!(benches);
```

**Baseline Generation (`xtask gen-baselines`):**
```rust
// xtask/src/gen_baselines.rs
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
struct BenchReceipt {
    tokens_per_second: f64,
    latency_ms: u64,
    kernels: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Baseline {
    schema_version: String,
    timestamp: String,
    model_sha256: String,
    cpu_tps: f64,
    cpu_latency_ms: u64,
    cpu_kernels: Vec<String>,
}

pub fn generate_baselines() -> Result<()> {
    // Read all bench receipts from ci/
    let bench_receipts: Vec<BenchReceipt> = glob::glob("ci/bench_*.json")?
        .filter_map(|path| {
            let path = path.ok()?;
            let contents = std::fs::read_to_string(&path).ok()?;
            serde_json::from_str(&contents).ok()
        })
        .collect();

    // Aggregate metrics (median TPS)
    let mut tps_values: Vec<f64> = bench_receipts.iter()
        .map(|r| r.tokens_per_second)
        .collect();
    tps_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_tps = tps_values[tps_values.len() / 2];

    // Extract unique kernels
    let mut all_kernels: Vec<String> = bench_receipts.iter()
        .flat_map(|r| r.kernels.clone())
        .collect();
    all_kernels.sort();
    all_kernels.dedup();

    // Create baseline
    let baseline = Baseline {
        schema_version: "1.0.0".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        model_sha256: compute_model_sha256()?,
        cpu_tps: median_tps,
        cpu_latency_ms: bench_receipts[tps_values.len() / 2].latency_ms,
        cpu_kernels: all_kernels,
    };

    // Write to docs/baselines/
    let baseline_path = format!(
        "docs/baselines/{}-cpu.json",
        chrono::Utc::now().format("%Y%m%d")
    );
    std::fs::write(
        &baseline_path,
        serde_json::to_string_pretty(&baseline)?
    )?;

    println!("✓ Generated baseline: {}", baseline_path);
    println!("  CPU TPS: {:.1} tok/s", baseline.cpu_tps);
    println!("  Kernels: {:?}", baseline.cpu_kernels);

    Ok(())
}
```

**Regression Detection:**
```rust
// xtask/src/verify_baselines.rs
use anyhow::{Context, Result};

pub fn verify_against_baseline(current_tps: f64) -> Result<()> {
    let baseline_path = "docs/baselines/latest-cpu.json";
    let baseline: Baseline = serde_json::from_str(
        &std::fs::read_to_string(baseline_path)?
    )?;

    let baseline_tps = baseline.cpu_tps;
    let change_ratio = (current_tps - baseline_tps) / baseline_tps;

    // Fail if performance dropped >10%
    if change_ratio < -0.10 {
        anyhow::bail!(
            "Performance regression detected!\n\
             Current: {:.1} tok/s\n\
             Baseline: {:.1} tok/s\n\
             Change: {:.1}% (threshold: -10%)",
            current_tps,
            baseline_tps,
            change_ratio * 100.0
        );
    }

    if change_ratio > 0.10 {
        println!(
            "🚀 Performance improvement!\n\
             Current: {:.1} tok/s\n\
             Baseline: {:.1} tok/s\n\
             Change: +{:.1}%",
            current_tps,
            baseline_tps,
            change_ratio * 100.0
        );
    }

    Ok(())
}
```

### 5. Remove Production Mocks

**Goal**: Fail-fast on model load errors, remove mock inference from production paths.

**Acceptance Criteria:**
- AC21: Model load errors fail immediately with actionable guidance
- AC22: Test-only mocks behind `test-mock-model` feature
- AC23: No silent test skips - fixtures required or test fails
- AC24: Clear error messages reference `xtask fetch-models`

**Error Handling Improvements:**
```rust
// bitnet-inference/src/engine.rs
impl InferenceEngine {
    pub fn new(model_path: &Path) -> Result<Self> {
        // Fail-fast: no silent fallback to mock
        let model = GgufModel::load(model_path)
            .context(format!(
                "Failed to load model: {}\n\n\
                 Troubleshooting:\n\
                 1. Verify model exists: ls -lh {}\n\
                 2. Check GGUF format: cargo run -p bitnet-cli -- compat-check {}\n\
                 3. Provision test models: cargo run -p xtask -- fetch-models\n\
                 4. See docs/howto/validate-models.md for detailed guide",
                model_path.display(),
                model_path.display(),
                model_path.display()
            ))?;

        Ok(Self {
            model,
            device: Device::Cpu,
            kernel_recorder: None,
        })
    }
}
```

**Test-Only Mock Feature:**
```rust
// bitnet-inference/Cargo.toml
[features]
default = []
cpu = ["bitnet-kernels/cpu"]
gpu = ["bitnet-kernels/gpu"]
test-mock-model = []  # Enable mock models for unit tests only

// bitnet-inference/tests/mock_model.rs
#[cfg(feature = "test-mock-model")]
pub struct MockModel {
    // Minimal mock for unit tests
}

#[cfg(not(feature = "test-mock-model"))]
compile_error!("MockModel requires test-mock-model feature");
```

**Fixture Validation:**
```rust
// crossval/tests/parity.rs
fn get_test_model_path() -> PathBuf {
    // Try cached model first
    if let Ok(cache_dir) = std::env::var("BITNET_MODEL_CACHE_DIR") {
        let model_path = PathBuf::from(cache_dir).join("ci-light/model.gguf");
        if model_path.exists() {
            return model_path;
        }
    }

    // Fallback to workspace models/
    let model_path = PathBuf::from("models/ci-light.gguf");
    if !model_path.exists() {
        panic!(
            "Test model not found: {}\n\n\
             Run: cargo run -p xtask -- fetch-models --tier ci\n\
             Or set BITNET_MODEL_CACHE_DIR to custom location",
            model_path.display()
        );
    }

    model_path
}
```

## Implementation Phases

### Phase A: Model Fetcher Foundation (PR-A)

**Goal**: Automated model provisioning infrastructure.

**Tasks:**
1. Create lockfile schema and parser (`crossval-models.lock.json`)
2. Implement `xtask fetch-models` command with download manager
3. Add SHA-256 verification and cache management
4. Populate lockfile with CI-light and integration models
5. Document usage in `docs/development/crossval-setup.md`

**Files Modified/Created:**
- `crossval-models.lock.json` (new)
- `xtask/src/fetch_models.rs` (new)
- `xtask/src/main.rs` (add fetch-models command)
- `docs/development/crossval-setup.md` (new)

**Validation:**
```bash
# AC:1 - Fetch models from lockfile
cargo run -p xtask -- fetch-models --tier ci
ls ~/.cache/bitnet/models/*/model.gguf

# AC:2 - SHA-256 verification
# Corrupt cached model and verify fetch detects it
echo "corrupted" >> ~/.cache/bitnet/models/abc123.../model.gguf
cargo run -p xtask -- fetch-models --verify-only  # Should fail

# AC:3 - Cache location
test -f ~/.cache/bitnet/models/abc123.../model.gguf

# AC:4 - Two-tier strategy
cargo run -p xtask -- fetch-models --tier ci
du -sh ~/.cache/bitnet/models/*/  # Verify ~50MB

# AC:5 - Progress bars
cargo run -p xtask -- fetch-models --tier integration  # Visual check

# AC:6 - Idempotent
time cargo run -p xtask -- fetch-models  # Should skip download
```

**Test Coverage:**
```rust
// xtask/tests/fetch_models.rs
#[test]
fn test_lockfile_parsing() { /* AC:1 */ }

#[test]
fn test_sha256_verification() { /* AC:2 */ }

#[test]
fn test_cache_location() { /* AC:3 */ }

#[test]
fn test_tier_filtering() { /* AC:4 */ }

#[test]
fn test_idempotent_fetch() { /* AC:6 */ }
```

### Phase B: C++ Bridge Implementation (PR-B)

**Goal**: Replace mock C++ wrapper with real llama.cpp FFI.

**Dependencies**: Requires PR-A merged (model fetcher).

**Tasks:**
1. Implement C wrapper (`crossval/src/llama_cpp_ffi.c`)
2. Create Rust FFI bindings (`crossval/src/cpp_bridge.rs`)
3. Update build script with `BITNET_CPP_DIR` detection
4. Add feature gate `crossval-cpp` to `crossval/Cargo.toml`
5. Document setup in `docs/development/crossval-cpp-setup.md`

**Files Modified/Created:**
- `crossval/src/llama_cpp_ffi.c` (new, replaces mock)
- `crossval/src/cpp_bridge.rs` (new)
- `crossval/src/bitnet_cpp_wrapper.c` (remove mock)
- `crossval/build.rs` (add BITNET_CPP_DIR linking)
- `crossval/Cargo.toml` (add crossval-cpp feature)
- `docs/development/crossval-cpp-setup.md` (new)

**Validation:**
```bash
# AC:7 - Feature gate requires BITNET_CPP_DIR
cargo build -p crossval --features crossval-cpp  # Should fail without env

export BITNET_CPP_DIR=/path/to/llama.cpp
cargo build -p crossval --features crossval-cpp  # Should succeed

# AC:8 - Minimal FFI surface
# Verify only 4 functions exported
nm target/debug/libcrossval.a | grep llama_cpp

# AC:9 - Build script links llama.cpp
ldd target/debug/libcrossval.so | grep llama

# AC:10 - Tests skip gracefully
cargo test -p crossval  # Without feature, should skip C++ tests

# AC:11 - Memory safety (valgrind check)
valgrind --leak-check=full \
  cargo test -p crossval --features crossval-cpp -- --nocapture
```

**Test Coverage:**
```rust
// crossval/tests/cpp_bridge.rs
#[cfg(feature = "crossval-cpp")]
#[test]
fn test_model_load_unload() { /* AC:11 - Drop cleanup */ }

#[cfg(feature = "crossval-cpp")]
#[test]
fn test_ffi_tokenization() { /* AC:8 - FFI surface */ }

#[cfg(not(feature = "crossval-cpp"))]
#[test]
fn test_feature_gate_skip() { /* AC:10 - Graceful skip */ }
```

### Phase C: Parity Harness (PR-C)

**Goal**: Systematic validation against C++ reference.

**Dependencies**: Requires PR-A and PR-B merged.

**Tasks:**
1. Implement `ParityTest` struct and methods
2. Add tokenization, logits, greedy decode tests
3. Create parity receipt schema and emission
4. Integrate with `xtask crossval` command
5. Add CI workflow for parity validation

**Files Modified/Created:**
- `crossval/src/parity.rs` (new)
- `crossval/tests/parity.rs` (new)
- `xtask/src/crossval.rs` (add parity command)
- `.github/workflows/crossval.yml` (new)
- `ci/parity.json` (receipt output)

**Validation:**
```bash
# AC:12 - Tokenization exact match
cargo test -p crossval --features crossval-cpp test_parity_tokenization

# AC:13 - Logits cosine ≥0.99
cargo test -p crossval --features crossval-cpp test_parity_logits

# AC:14 - Greedy decode exact match
cargo test -p crossval --features crossval-cpp test_parity_greedy_decode

# AC:15 - Receipt generation
cargo test -p crossval --features crossval-cpp test_parity_receipt_generation
test -f ci/parity.json

# AC:16 - Feature gate skip
cargo test -p crossval  # Without feature, should skip
```

**Test Coverage:**
```rust
// crossval/tests/parity.rs
#[cfg(feature = "crossval-cpp")]
#[test]
fn test_parity_tokenization() { /* AC:12 */ }

#[cfg(feature = "crossval-cpp")]
#[test]
fn test_parity_logits() { /* AC:13 */ }

#[cfg(feature = "crossval-cpp")]
#[test]
fn test_parity_greedy_decode() { /* AC:14 */ }

#[cfg(feature = "crossval-cpp")]
#[test]
fn test_parity_receipt_generation() { /* AC:15 */ }
```

### Phase D: Real Benchmarks (PR-D)

**Goal**: Replace fabricated TPS with measured performance.

**Dependencies**: Requires PR-A merged (model fetcher).

**Tasks:**
1. Rewrite `crossval/benches/performance.rs` to use real inference
2. Implement receipt emission in benchmark runs
3. Create `xtask gen-baselines` command
4. Add regression detection to CI
5. Remove placeholder TPS from existing benchmarks

**Files Modified/Created:**
- `crossval/benches/performance.rs` (rewrite, remove mocks)
- `xtask/src/gen_baselines.rs` (new)
- `xtask/src/verify_baselines.rs` (new)
- `docs/baselines/YYYYMMDD-cpu.json` (generated)
- `.github/workflows/benchmarks.yml` (add regression check)

**Validation:**
```bash
# AC:17 - Real inference measurement
cargo bench -p crossval --bench performance

# AC:18 - Emit bench receipts
ls ci/bench_*.json

# AC:19 - Generate baselines
cargo run -p xtask -- gen-baselines
test -f docs/baselines/$(date +%Y%m%d)-cpu.json

# AC:20 - Regression detection
# Simulate performance drop
sed -i 's/"tokens_per_second": [0-9.]*/"tokens_per_second": 5.0/' ci/bench_*.json
cargo run -p xtask -- verify-baselines  # Should fail
```

**Test Coverage:**
```rust
// xtask/tests/baselines.rs
#[test]
fn test_baseline_generation() { /* AC:19 */ }

#[test]
fn test_regression_detection() { /* AC:20 */ }

#[test]
fn test_baseline_schema_validation() { /* Verify v1.0.0 */ }
```

### Phase E: Mock Elimination (PR-E)

**Goal**: Remove production mocks, fail-fast error handling.

**Dependencies**: Requires PR-A merged (model fetcher for error messages).

**Tasks:**
1. Remove mock inference from `bitnet-inference` production code
2. Move remaining mocks behind `test-mock-model` feature
3. Update error messages with actionable guidance
4. Fix silent test skips (require fixtures or fail)
5. Remove fabricated data from crossval benchmarks

**Files Modified/Created:**
- `bitnet-inference/src/engine.rs` (remove production mocks)
- `bitnet-inference/tests/mock_model.rs` (gate behind test-mock-model)
- `crossval/tests/*.rs` (fix silent skips)
- `crossval/benches/performance.rs` (remove placeholder data)

**Validation:**
```bash
# AC:21 - Fail-fast on model load errors
cargo run -p bitnet-cli -- run --model nonexistent.gguf  # Should fail immediately

# AC:22 - Test-only mocks
cargo build -p bitnet-inference --features test-mock-model  # Allowed
cargo build -p bitnet-inference  # Should not include mock

# AC:23 - No silent test skips
# Remove cached models
rm -rf ~/.cache/bitnet/models/*
cargo test -p crossval  # Should fail with actionable error

# AC:24 - Clear error messages
cargo run -p bitnet-cli -- run --model bad.gguf 2>&1 | grep "xtask fetch-models"
```

**Test Coverage:**
```rust
// bitnet-inference/tests/error_handling.rs
#[test]
fn test_fail_fast_missing_model() { /* AC:21 */ }

#[test]
fn test_mock_feature_gate() { /* AC:22 */ }

// crossval/tests/fixture_validation.rs
#[test]
fn test_no_silent_skip() { /* AC:23 */ }

#[test]
fn test_error_message_guidance() { /* AC:24 */ }
```

### Phase F: CI Integration (PR-F)

**Goal**: Automate validation in CI workflows.

**Dependencies**: Requires all previous PRs merged.

**Tasks:**
1. Create `.github/workflows/crossval.yml` for parity tests
2. Add CI label trigger for crossval (`crossval` label)
3. Configure nightly full validation
4. Update branch protection to require parity checks
5. Document CI strategy in `docs/development/ci-validation.md`

**Files Modified/Created:**
- `.github/workflows/crossval.yml` (new)
- `.github/workflows/nightly.yml` (add full validation)
- `docs/development/ci-validation.md` (new)

**Validation:**
```bash
# CI workflow validation (requires GitHub Actions runner)
# PR default: fast, no models
gh pr create --label "crossval"  # Triggers model fetch and parity

# Nightly: full validation
# Verify nightly workflow runs full suite
```

**Test Coverage:**
- Manual verification via GitHub Actions
- Test workflow YAML syntax: `actionlint .github/workflows/crossval.yml`

## API Contracts

### Model Fetcher API

```rust
// xtask/src/fetch_models.rs
pub struct ModelFetcher {
    lockfile_path: PathBuf,
    cache_dir: PathBuf,
}

impl ModelFetcher {
    pub fn new() -> Result<Self>;
    pub fn fetch_all(&self) -> Result<()>;
    pub fn fetch_tier(&self, tier: &str) -> Result<()>;
    pub fn verify_cache(&self) -> Result<()>;
    pub fn get_model_path(&self, name: &str) -> Result<PathBuf>;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelLockfile {
    pub schema_version: String,
    pub models: Vec<ModelEntry>,
    pub updated: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub tier: String,
    pub url: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub description: String,
}
```

### C++ Bridge API

```rust
// crossval/src/cpp_bridge.rs
#[cfg(feature = "crossval-cpp")]
pub struct CppModel {
    wrapper: *mut ffi::LlamaModelWrapper,
}

#[cfg(feature = "crossval-cpp")]
impl CppModel {
    pub fn load(path: &Path) -> Result<Self>;
    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>>;
    pub fn eval(&self, tokens: &[i32]) -> Result<Vec<f32>>;
    pub fn generate_greedy(&self, prompt: &str, max_tokens: usize) -> Result<Vec<i32>>;
}

#[cfg(feature = "crossval-cpp")]
impl Drop for CppModel {
    fn drop(&mut self);
}
```

### Parity Test API

```rust
// crossval/src/parity.rs
pub struct ParityTest {
    rust_engine: InferenceEngine,
    cpp_model: CppModel,
    model_sha256: String,
}

impl ParityTest {
    pub fn new(model_path: &Path) -> Result<Self>;
    pub fn test_tokenization(&self, prompt: &str) -> Result<bool>;
    pub fn test_logits(&self, prompt: &str) -> Result<f64>;
    pub fn test_greedy_decode(&mut self, prompt: &str, max_tokens: usize) -> Result<bool>;
    pub fn run_full_suite(&mut self, prompts: &[&str]) -> Result<ParityReceipt>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityReceipt {
    pub schema_version: String,
    pub timestamp: String,
    pub model_sha256: String,
    pub bitnet_commit: String,
    pub test_suite: String,
    pub status: String,
    pub metrics: ParityMetrics,
}
```

### Baseline Generation API

```rust
// xtask/src/gen_baselines.rs
pub fn generate_baselines() -> Result<()>;
pub fn verify_against_baseline(current_tps: f64) -> Result<()>;

#[derive(Debug, Serialize, Deserialize)]
pub struct Baseline {
    pub schema_version: String,
    pub timestamp: String,
    pub model_sha256: String,
    pub cpu_tps: f64,
    pub cpu_latency_ms: u64,
    pub cpu_kernels: Vec<String>,
}
```

## Testing Strategy

### Unit Tests (TDD with AC Tags)

**Test Organization:**
```
xtask/tests/
├── fetch_models.rs         # AC:1-6 (model fetcher)
├── baselines.rs            # AC:19-20 (baseline generation)

crossval/tests/
├── cpp_bridge.rs           # AC:7-11 (C++ FFI)
├── parity.rs               # AC:12-16 (parity validation)
├── fixture_validation.rs   # AC:23 (no silent skips)

bitnet-inference/tests/
├── error_handling.rs       # AC:21, AC:24 (fail-fast)
├── mock_model.rs           # AC:22 (test-only mocks)
```

**AC Tag Mapping:**
```rust
// Example: xtask/tests/fetch_models.rs
#[test]
fn ac1_fetch_models_from_lockfile() {
    // AC:1 - Fetch models from lockfile
    let fetcher = ModelFetcher::new().unwrap();
    fetcher.fetch_tier("ci").unwrap();
    assert!(fetcher.get_model_path("ci-light").unwrap().exists());
}

#[test]
fn ac2_sha256_verification() {
    // AC:2 - SHA-256 prevents corrupted models
    let fetcher = ModelFetcher::new().unwrap();
    // Corrupt cached model
    let model_path = fetcher.get_model_path("ci-light").unwrap();
    std::fs::write(&model_path, "corrupted").unwrap();
    assert!(fetcher.verify_cache().is_err());
}
```

### Integration Tests

**Crossval Integration:**
```bash
# Full parity test with real models
cargo run -p xtask -- fetch-models --tier ci
export BITNET_CPP_DIR=/path/to/llama.cpp
cargo test -p crossval --features crossval-cpp -- --test-threads=1

# Benchmark integration
cargo bench -p crossval --bench performance
cargo run -p xtask -- gen-baselines
cargo run -p xtask -- verify-baselines
```

**CI Workflows:**
```yaml
# .github/workflows/crossval.yml
name: Cross-Validation

on:
  pull_request:
    types: [labeled]

jobs:
  parity:
    if: contains(github.event.pull_request.labels.*.name, 'crossval')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Fetch CI-light model
        run: cargo run -p xtask -- fetch-models --tier ci
      - name: Setup llama.cpp
        run: ./scripts/setup-llama-cpp.sh
      - name: Run parity tests
        run: cargo test -p crossval --features crossval-cpp
      - name: Verify parity receipt
        run: test -f ci/parity.json && cargo run -p xtask -- verify-receipt ci/parity.json
```

### Manual Validation Checklist

**Phase A (Model Fetcher):**
- [ ] Lockfile parses successfully
- [ ] SHA-256 verification detects corruption
- [ ] Cache directory created at `~/.cache/bitnet/models/`
- [ ] Two-tier strategy (CI-light ~50MB, integration ~2GB)
- [ ] Progress bars display during download
- [ ] Idempotent: re-run skips existing models

**Phase B (C++ Bridge):**
- [ ] `BITNET_CPP_DIR` required for feature gate
- [ ] FFI functions link against llama.cpp
- [ ] Memory leak check passes (valgrind)
- [ ] Tests skip gracefully without feature
- [ ] Build script detects missing llama.cpp

**Phase C (Parity Harness):**
- [ ] Tokenization exact match passes
- [ ] Logits cosine similarity ≥0.99
- [ ] Greedy decode produces identical tokens
- [ ] Parity receipt emitted to `ci/parity.json`
- [ ] Feature gate skips without C++ bridge

**Phase D (Real Benchmarks):**
- [ ] Benchmarks measure real inference (no mocks)
- [ ] Bench receipts emitted to `ci/bench_*.json`
- [ ] `xtask gen-baselines` creates `docs/baselines/YYYYMMDD-cpu.json`
- [ ] Regression detection fails on >10% TPS drop

**Phase E (Mock Elimination):**
- [ ] Model load errors fail immediately
- [ ] Test mocks gated behind `test-mock-model` feature
- [ ] Tests fail on missing fixtures (no silent skips)
- [ ] Error messages reference `xtask fetch-models`

**Phase F (CI Integration):**
- [ ] CI workflow triggers on `crossval` label
- [ ] Nightly full validation runs
- [ ] Branch protection requires parity checks
- [ ] Documentation updated with CI strategy

## Migration Guide

### For Contributors

**Before (Current):**
```bash
# Manual model download
wget https://example.com/model.gguf -O models/model.gguf

# Tests skip silently
cargo test -p crossval  # Skips all tests if fixture missing

# Fabricated benchmarks
cargo bench -p crossval  # Placeholder TPS values
```

**After (Improved):**
```bash
# Automated model provisioning
cargo run -p xtask -- fetch-models --tier ci

# Tests fail-fast with guidance
cargo test -p crossval  # Fails if fixture missing, shows fix command

# Real benchmarks with receipts
cargo bench -p crossval  # Measures actual inference, emits receipts
cargo run -p xtask -- gen-baselines
```

### For CI/CD Pipelines

**Before:**
```yaml
# No model provisioning
- run: cargo test -p crossval  # Skips tests
```

**After:**
```yaml
# Automated setup
- run: cargo run -p xtask -- fetch-models --tier ci
- run: cargo test -p crossval --features crossval-cpp
- run: cargo run -p xtask -- verify-receipt ci/parity.json
```

### For Release Managers

**Before:**
- Manual baseline generation
- No regression detection
- Silent test failures

**After:**
```bash
# Pre-release validation
cargo run -p xtask -- fetch-models --tier integration
cargo bench -p crossval
cargo run -p xtask -- gen-baselines
cargo run -p xtask -- verify-baselines  # Fails if regression

# Baseline published with release
git add docs/baselines/$(date +%Y%m%d)-cpu.json
git commit -m "docs: add v0.2.0 baseline receipt"
```

## Environment Variables

| Variable | Purpose | Default | Example |
|----------|---------|---------|---------|
| `BITNET_MODEL_CACHE_DIR` | Override cache location | `~/.cache/bitnet/models/` | `/tmp/bitnet-cache` |
| `BITNET_CPP_DIR` | llama.cpp installation path | (required for `crossval-cpp`) | `/opt/llama.cpp` |
| `BITNET_GGUF` | Model path for tests | (auto-discover from cache) | `/path/to/model.gguf` |
| `BITNET_STRICT_MODE` | Fail-fast validation | `0` | `1` |

## Performance Targets

### Model Fetching
- CI-light model download: < 30 seconds (50MB @ 2MB/s)
- Integration model download: < 5 minutes (2GB @ 10MB/s)
- Cache lookup: < 100ms (SHA-256 verification)

### Parity Testing
- Tokenization test: < 1 second per prompt
- Logits test: < 5 seconds per prompt (includes inference)
- Greedy decode test: < 30 seconds (10 tokens @ ~3 tok/s)

### Benchmarking
- Benchmark run: 10-60 seconds per configuration
- Baseline generation: < 5 seconds (aggregate receipts)
- Regression detection: < 1 second (compare TPS)

## Risks and Mitigations

### Risk 1: llama.cpp ABI Changes
**Impact**: C++ bridge breaks after llama.cpp updates
**Mitigation**: Pin llama.cpp commit in `crossval-models.lock.json`, document upgrade process
**Contingency**: Maintain mock fallback for local development

### Risk 2: Model Download Failures
**Impact**: CI flakiness due to network issues
**Mitigation**: Retry logic with exponential backoff, mirror models to GitHub Releases
**Contingency**: Cache models in CI artifact store

### Risk 3: Parity Test Numerical Instability
**Impact**: False positives from FP precision variance
**Mitigation**: Relaxed cosine threshold (0.99), deterministic inference mode
**Contingency**: Adaptive thresholds based on model architecture

### Risk 4: Baseline Drift
**Impact**: Performance baselines become stale
**Mitigation**: Nightly regeneration, version baselines with model SHA-256
**Contingency**: Manual baseline review on model updates

### Risk 5: CI Resource Constraints
**Impact**: Long-running parity tests delay PR feedback
**Mitigation**: Two-tier strategy (CI-light for PRs, full for nightly)
**Contingency**: Optional label-triggered validation

## Success Metrics

### Quantitative
- Model fetch success rate: >95% (CI reliability)
- Parity test pass rate: 100% (numerical accuracy)
- Benchmark regression detection: <10% false positives
- CI feedback time: <10 minutes (PR validation)
- Test coverage: >80% (crossval crate)

### Qualitative
- Contributors report easier setup (no manual model download)
- Zero silent test failures (fail-fast guidance)
- Confidence in Rust ↔ C++ parity (validated with real models)
- Performance baselines trusted (measured, not fabricated)

## Related Documentation

### Prerequisites
- `docs/development/test-suite.md` - Testing framework overview
- `docs/reference/validation-gates.md` - Receipt validation system
- `docs/explanation/receipt-validation.md` - Honest compute gates

### Implementation Guides
- `docs/development/crossval-setup.md` (new) - Model fetcher setup
- `docs/development/crossval-cpp-setup.md` (new) - C++ bridge setup
- `docs/development/ci-validation.md` (new) - CI integration guide

### Architecture Decisions
- `docs/explanation/architecture/adr-003-receipt-schema-stability.md` - Receipt versioning
- `docs/explanation/architecture/adr-012-crossval-two-tier-strategy.md` (new) - Model provisioning

### Reference
- `docs/reference/xtask-commands.md` - xtask CLI reference
- `docs/reference/environment-variables.md` - Environment configuration
- `CROSSVAL.md` (new) - Cross-validation user guide

## Appendix: File Structure

```
BitNet-rs/
├── crossval/
│   ├── src/
│   │   ├── cpp_bridge.rs           # C++ FFI bindings (new)
│   │   ├── llama_cpp_ffi.c         # C wrapper for llama.cpp (new)
│   │   ├── parity.rs               # Parity test harness (new)
│   │   ├── fixtures.rs             # Updated with fetcher integration
│   │   └── bitnet_cpp_wrapper.c    # REMOVED (replaced by llama_cpp_ffi.c)
│   ├── benches/
│   │   └── performance.rs          # Rewritten with real inference
│   ├── tests/
│   │   ├── parity.rs               # Parity validation tests (new)
│   │   └── cpp_bridge.rs           # FFI integration tests (new)
│   └── Cargo.toml                  # Add crossval-cpp feature
├── xtask/
│   ├── src/
│   │   ├── fetch_models.rs         # Model fetcher implementation (new)
│   │   ├── gen_baselines.rs        # Baseline generation (new)
│   │   ├── verify_baselines.rs     # Regression detection (new)
│   │   └── main.rs                 # Add fetch-models, gen-baselines commands
│   └── tests/
│       ├── fetch_models.rs         # Model fetcher tests (new)
│       └── baselines.rs            # Baseline generation tests (new)
├── .github/
│   └── workflows/
│       ├── crossval.yml            # Parity validation workflow (new)
│       └── nightly.yml             # Full validation (updated)
├── docs/
│   ├── development/
│   │   ├── crossval-setup.md       # Model fetcher guide (new)
│   │   ├── crossval-cpp-setup.md   # C++ bridge guide (new)
│   │   └── ci-validation.md        # CI integration guide (new)
│   ├── explanation/
│   │   └── architecture/
│   │       └── adr-012-crossval-two-tier-strategy.md  (new)
│   └── baselines/
│       └── YYYYMMDD-cpu.json       # Generated baselines
├── crossval-models.lock.json       # Model registry (new)
└── CROSSVAL.md                     # User guide (new)
```

## Glossary

- **Parity**: Numerical agreement between Rust and C++ implementations
- **Cosine Similarity**: Metric for logits comparison (1.0 = identical, 0.0 = orthogonal)
- **Greedy Decode**: Deterministic token selection (argmax at each step)
- **Receipt**: JSON artifact with performance metrics and kernel IDs
- **Two-Tier Strategy**: CI-light (fast) vs integration (comprehensive) model tiers
- **Honest Compute**: Verification that receipts reflect actual execution (not mocked)
- **SHA-256**: Cryptographic hash for model integrity verification
- **FFI**: Foreign Function Interface (Rust ↔ C/C++ boundary)

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-10-16 | Initial specification |
