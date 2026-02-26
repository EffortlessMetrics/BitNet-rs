# BitNet-rs Cross-Validation Infrastructure - Comprehensive Analysis

**Generated**: 2025-10-24  
**Scope**: Existing cross-validation framework for layer-by-layer comparison infrastructure  
**Location**: `/home/steven/code/Rust/BitNet-rs`

---

## Executive Summary

BitNet-rs has a **mature cross-validation framework** with comprehensive infrastructure for comparing Rust vs C++ implementations. The system includes:

1. **Dedicated `crossval` crate** with parity harness, scoring, and comparison utilities
2. **Receipt system** (Schema v1.0.0) for honest compute verification in CI/CD
3. **Kernel recorder** for tracking which compute paths executed
4. **FFI bridge** to C++ reference implementation for direct comparison
5. **Property-based testing** and baseline management
6. **Multiple parity test suites** across quantization, tokenization, and inference paths

This infrastructure provides **excellent foundations** for building layer-by-layer cross-validation. The key building blocks are already in place; primarily need to add intermediate tensor capture and layer-level comparison logic.

---

## 1. Crossval Crate Architecture

**Location**: `/home/steven/code/Rust/BitNet-rs/crossval/`

### Structure

```
crossval/
├── Cargo.toml              # Feature-gated: crossval, ffi, iq2s-ffi, cpu-gpu
├── README.md               # Comprehensive setup and usage guide
├── src/
│   ├── lib.rs             # Main API entry points
│   ├── comparison.rs      # High-level comparison runner
│   ├── cpp_bindings.rs    # Safe FFI wrappers to C++ 
│   ├── fixtures.rs        # Test model/data management
│   ├── score.rs           # NLL/perplexity evaluation (teacher-forcing)
│   ├── utils.rs           # Token/float comparison + perf utilities
│   └── validation.rs      # Validation gates for accuracy/performance/compatibility
├── tests/
│   ├── parity.rs                 # Deterministic parity tests (LOGIT_TOLERANCE=1e-4)
│   ├── parity_bitnetcpp.rs       # Full BitNet.cpp parity harness
│   ├── qk256_crossval.rs         # QK256 vs FP32 reference validation
│   ├── token_equivalence.rs      # Tokenizer parity tests
│   ├── iq2s_validation.rs        # IQ2_S quantization validation
│   ├── framework_validation.rs   # Test infrastructure validation
│   ├── ffi_integration.rs        # FFI bridge lifecycle tests
│   ├── performance_validation.rs # Throughput/latency comparisons
│   ├── parity_receipts.rs        # Receipt generation and validation
│   ├── cpp_probe.rs              # C++ availability detection
│   └── ms_bitnet_mapping.rs      # Tensor name mapping validation
├── benches/
│   └── performance.rs      # Criterion-based performance benchmarks
├── fixtures/
│   ├── minimal_model.gguf  # Small test model
│   └── test_model_small_*  # Test data
├── docs/
│   ├── PARITY_IMPLEMENTATION.md  # Implementation guide
│   └── baselines/                # Baseline results by date
├── props/                  # Python property-based testing
│   ├── test_greedy_parity.py
│   ├── test_logit_parity.py
│   ├── test_nll_parity.py
│   ├── test_greedy_invariants.py
│   ├── strategies.py
│   ├── metrics.py
│   └── run_model.py
└── build.rs               # C++ binding generation
```

### Key Crate Dependencies

```toml
[features]
default = []
crossval = ["bindgen", "cc", "bitnet-sys/ffi", "bitnet-inference/ffi", "ffi"]
ffi = ["cc", "bitnet-inference/ffi"]
iq2s-ffi = ["bitnet-models/iq2s-ffi", "bitnet-ggml-ffi/iq2s-ffi"]
cpu = ["bitnet-inference/cpu", "bitnet-models/cpu"]
gpu = ["bitnet-inference/gpu", "bitnet-models/gpu"]
```

**Important**: `crossval` feature is **feature-gated** for zero overhead when disabled.

---

## 2. Parity Harness Architecture

### Test Model Path Configuration

**File**: `/home/steven/code/Rust/BitNet-rs/crossval/tests/parity_bitnetcpp.rs`

Tests require:
- `CROSSVAL_GGUF` environment variable → model path
- `BITNET_CPP_DIR` (optional) → C++ reference (when not set, runs Rust-only)
- `PARITY_TEST_TIMEOUT_SECS` (optional) → test timeout (default: `DEFAULT_PARITY_TIMEOUT_SECS`)

**Key Capability**: Tests **gracefully skip** if C++ not available, allowing green CI even without reference.

### Parity Metrics

The harness implements **tolerance-based comparisons**:

```rust
// From parity.rs
const LOGIT_TOLERANCE: f32 = 1e-4;  // ±0.0001 for logit differences

fn compare_logits(rust_logits: &[f32], cpp_logits: &[f32], step: usize) -> Result<()> {
    // Finds max difference and location
    // Returns both absolute difference AND top-5 token rankings
    // Validates cosine similarity >= 0.99 for soft match
}

// From parity_bitnetcpp.rs
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Cosine similarity in [0, 1]
    // 1.0 = identical, 0.0 = orthogonal
}
```

### Parity Receipt Generation

**File**: `/home/steven/code/Rust/BitNet-rs/crossval/tests/parity_bitnetcpp.rs` + receipts.rs

```rust
// Example receipt path: ci/receipts/2025-10-19/parity-bitnetcpp.json
{
  "schema_version": "1.0.0",
  "timestamp": "2025-10-19T...",
  "parity": {
    "cpp_available": true,
    "cosine_similarity": 0.9923,
    "exact_match_rate": 1.0,
    "first_divergence_step": null,
    "status": "ok"  // "ok" | "rust_only" | "divergence" | "timeout"
  },
  "model_info": {
    "sha256": "...",
    "layers": 24,
    "hidden_size": 2048,
    "vocab_size": 32000
  },
  "performance": {
    "rust_tok_s": 42.5,
    "cpp_tok_s": 45.0,
    "throughput_ratio": 0.944
  }
}
```

**Status values**:
- `"ok"`: Exact token sequence match or cosine ≥ 0.99
- `"rust_only"`: C++ reference not available (graceful fallback)
- `"divergence"`: Outputs differ significantly (cosine < 0.99 or mismatch)
- `"timeout"`: Test exceeded timeout threshold

---

## 3. Receipt System (Schema v1.0.0)

**Files**: 
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/receipts.rs`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/tests/support/receipt.rs`
- `/home/steven/code/Rust/BitNet-rs/ci/inference.json` (example)

### Core Receipt Structure

```rust
#[derive(Serialize, Deserialize)]
pub struct InferenceReceipt {
    pub schema_version: String,           // "1.0.0"
    pub timestamp: String,                 // RFC3339 format
    pub compute_path: String,              // "real" | "mock"
    pub backend: String,                   // "cpu" | "cuda" | "metal"
    pub deterministic: bool,
    
    // Honest compute evidence
    pub kernels: Vec<String>,              // ["i2s_gemv", "rope_apply", ...]
    pub kernel_count: usize,
    
    // Model information
    pub model: ModelInfo {
        pub path: String,
        pub quantization_type: Option<String>,
        pub layers: Option<usize>,
        pub hidden_size: Option<usize>,
        pub vocab_size: Option<usize>,
        pub sha256: Option<String>,
    },
    
    // Performance
    pub tokens_generated: usize,
    pub tokens_per_second: f64,
    pub tokens_requested: usize,
    
    // Cross-validation (optional)
    pub parity: Option<ParityMetadata> {
        pub cpp_available: bool,
        pub cosine_similarity: f32,
        pub exact_match_rate: f32,
        pub status: String,  // "ok" | "rust_only" | "divergence"
    },
    
    // Test results (optional)
    pub test_results: Option<TestResults> {
        pub total_tests: usize,
        pub passed: usize,
        pub accuracy_tests: Option<AccuracyTestResults>,
        pub determinism_tests: Option<DeterminismTestResults>,
        pub kv_cache_tests: Option<KVCacheTestResults>,
    },
}
```

### Validation Gates

8 validation gates in `verification.rs`:

1. **Compute Path Gate**: `compute_path == "real"` (no mocks in CI)
2. **Backend Gate**: Valid backend (`cpu|cuda|metal`)
3. **Kernel ID Hygiene**: No empty strings, length ≤ 128, count ≤ 10K
4. **Determinism Gate**: Must match flag or explicit in receipt
5. **GPU Requirement Gate**: `backend="cuda"` requires GPU kernels (auto-enforced)
6. **Model Info Gate**: SHA256 digest and dimensions present
7. **Performance Gate**: Tokens/second ≥ minimum threshold
8. **Parity Gate** (optional): If C++ available, cosine ≥ 0.99 and exact_match_rate = 1.0

---

## 4. Comparison and Validation Infrastructure

### Comparison Module

**File**: `/home/steven/code/Rust/BitNet-rs/crossval/src/comparison.rs`

```rust
pub struct CrossValidator {
    config: CrossvalConfig,
}

impl CrossValidator {
    pub fn validate_fixture(&self, fixture: &TestFixture) -> Result<Vec<ComparisonResult>>;
    
    fn compare_single_prompt(
        &self,
        test_name: &str,
        prompt: &str,
        cpp_model: &CppModel,
    ) -> ComparisonResult;
    
    fn generate_rust(&self, prompt: &str) -> Result<Vec<u32>>;
}

pub struct ComparisonResult {
    pub test_name: String,
    pub prompt: String,
    pub rust_tokens: Vec<u32>,
    pub cpp_tokens: Vec<u32>,
    pub tokens_match: bool,
    pub rust_performance: Option<f64>,  // tok/s
    pub cpp_performance: Option<f64>,
    pub error: Option<String>,
}
```

**Key capability**: Compares token sequences AND performance metrics in parallel.

### Validation Module

**File**: `/home/steven/code/Rust/BitNet-rs/crossval/src/validation.rs`

```rust
pub struct ValidationSuite {
    pub model_path: String,
    pub tokenizer_path: Option<String>,
    pub deterministic: bool,
}

impl ValidationSuite {
    pub fn validate_model_compatibility(&self) -> Result<ValidationResult>;
    pub fn validate_token_parity(&self, prompts: &[String]) -> Result<TokenParityResult>;
    pub fn validate_nll_parity(&self, dataset: &str) -> Result<NllParityResult>;
    pub fn validate_performance(&self, baseline: Option<&Path>) -> Result<PerformanceResult>;
    pub fn run_all(&self) -> Result<Vec<ValidationResult>>;
}

pub struct ValidationResult {
    pub gate: String,
    pub passed: bool,
    pub metrics: HashMap<String, Value>,
    pub message: String,
}
```

**4 validation gates**:
1. Model compatibility (tensor mapping)
2. Token parity (95%+ match rate)
3. NLL parity (< 0.01 delta)
4. Performance (≥ 1.0 tok/s, throughput ratio ≥ 0.95)

### Scoring Module (NLL/Perplexity)

**File**: `/home/steven/code/Rust/BitNet-rs/crossval/src/score.rs`

Teacher-forcing evaluation loop:

```rust
pub fn evaluate_perplexity<M, T>(
    model: &mut M,
    tokenizer: &T,
    input_path: &Path,
    max_context: usize,
) -> Result<ScoreOutput>

// Returns:
pub struct ScoreOutput {
    pub mean_nll: f64,
    pub perplexity: f64,
    pub total_tokens: usize,
    pub tokens_per_second: f64,
    pub model_info: ModelInfo,
}

pub fn validate_parity(
    rust_output: &ScoreOutput,
    cpp_output: &ScoreOutput,
    tolerance: f64,
) -> Result<()>
```

**Validation approach**: 
- Numerically stable log-softmax
- Teacher forcing (true tokens as input)
- Per-position NLL accumulation
- Tolerance-based parity check

---

## 5. Kernel Recorder (Honest Compute Tracking)

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/kernel_recorder.rs`

```rust
pub struct KernelRecorder {
    inner: Arc<Mutex<Vec<String>>>,
}

impl KernelRecorder {
    pub fn new() -> Self;
    pub fn record(&self, id: &'static str);  // Thread-safe
    pub fn snapshot(&self) -> Vec<String>;   // Deduplicated, insertion-order
    pub fn count(&self) -> usize;
    pub fn clear(&self);
}
```

**Usage in inference**:
```rust
// In kernel execution paths
pub fn execute_kernel(recorder: &KernelRecorder) {
    recorder.record("i2s_gemv");  // Records kernel ID
    // ... actual computation ...
}

// In receipt generation
let kernels = recorder.snapshot();  // ["i2s_gemv", "rope_apply", "attention_real"]
receipt.kernels = kernels;
```

**Key properties**:
- Thread-safe (`Arc<Mutex<...>>`)
- O(1) record operation
- Deduplicates while preserving first-occurrence order
- Used for "honest compute" verification in CI/CD

---

## 6. FFI Bridge Architecture

**Files**: 
- `/home/steven/code/Rust/BitNet-rs/crossval/src/cpp_bindings.rs`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/` (FFI wrapper)

### C++ Bindings

```rust
#[cfg(all(feature = "ffi", have_cpp))]
mod imp {
    unsafe extern "C" {
        fn bitnet_cpp_create_model(model_path: *const c_char) -> *mut c_void;
        fn bitnet_cpp_destroy_model(model: *mut c_void);
        fn bitnet_cpp_generate(
            model: *mut c_void,
            prompt: *const c_char,
            max_tokens: c_int,
            tokens_out: *mut u32,
            tokens_count: *mut c_int,
        ) -> c_int;
    }

    pub struct CppModel {
        handle: *mut c_void,
    }

    impl CppModel {
        pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self>;
        pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<Vec<u32>>;
    }
}
```

### Higher-Level FFI Usage (bitnet-sys)

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/`

From parity_bitnetcpp.rs example:

```rust
use bitnet_sys::{
    BitnetContext, BitnetModel,
    bitnet_eval_tokens, bitnet_prefill, bitnet_tokenize_text,
    cpp_decode_greedy, cpp_vocab_size,
};

// Load C++ model
let cpp_model = BitnetModel::from_file(gguf_str)?;
let cpp_ctx = BitnetContext::new(&cpp_model, 4096, 1, 0)?;

// Vocab check
let cpp_vocab = cpp_vocab_size(&cpp_ctx)?;

// Tokenize
let cpp_ids = bitnet_tokenize_text(&cpp_model, prompt, add_bos, parse_special)?;

// Prefill and eval
bitnet_prefill(&cpp_ctx, &cpp_ids_i32)?;
let cpp_logits = bitnet_eval_tokens(&cpp_ctx, &[next_token])?;

// Greedy decode
let token = cpp_decode_greedy(&cpp_logits)?;
```

### Build System

**File**: `/home/steven/code/Rust/BitNet-rs/crossval/build.rs`

- Uses `bindgen` to generate FFI bindings from C++ headers
- Optional compilation (gated by `crossval` feature)
- Requires `BITNET_CPP_DIR` environment variable to be set

---

## 7. Test Suites and Property-Based Testing

### Unit-Level Parity Tests

**File**: `/home/steven/code/Rust/BitNet-rs/crossval/tests/parity.rs`

```rust
#[test]
fn test_model_loading_parity() -> Result<()> {
    // 1. Load C++ model
    let cpp_session = CppSession::load_deterministic(&model_path)?;
    
    // 2. Load Rust model
    let rust_model = load_model(&model_path)?;
    
    // 3. Compare vocab size, dimensions
    assert_eq!(rust_vocab, cpp_vocab);
}

#[test]
fn test_first_token_inference() -> Result<()> {
    // 1. Tokenize prompt
    let tokens = tokenize(prompt)?;
    
    // 2. Get logits from both implementations
    let rust_logits = rust_model.forward(&tokens)?;
    let cpp_logits = cpp_session.eval(&tokens)?;
    
    // 3. Compare with tolerance
    compare_logits(&rust_logits, &cpp_logits, LOGIT_TOLERANCE)?;
}
```

### QK256 Quantization Parity

**File**: `/home/steven/code/Rust/BitNet-rs/crossval/tests/qk256_crossval.rs`

Tests QK256 kernel against FP32 reference:

```rust
#[test]
fn test_qk256_vs_fp32_reference_small_matrix() -> Result<()> {
    // 1. Create test matrix of codes
    let codes: Vec<u8> = (0..total).map(|i| (i % 4) as u8).collect();
    
    // 2. Pack into QK256 format
    let mut packed_data = vec![0u8; rows * QK256_PACKED_BYTES];
    // ... packing logic ...
    
    // 3. Compute QK256 result
    let mut qk256_output = vec![0.0f32; rows];
    gemv_qk256(&packed_data, &input, &mut qk256_output, rows, cols, ...)?;
    
    // 4. Compute FP32 reference
    let mut fp32_output = vec![0.0f32; rows];
    gemv_fp32_reference(&codes, &input, &mut fp32_output, rows, cols)?;
    
    // 5. Compare with 1e-4 tolerance
    for (q, f) in qk256_output.iter().zip(fp32_output.iter()) {
        assert!((q - f).abs() < 1e-4);
    }
}
```

### Python Property-Based Testing

**Location**: `/home/steven/code/Rust/BitNet-rs/crossval/props/`

Uses Hypothesis for property-based test generation:

- `test_greedy_parity.py`: Greedy decode invariants across seeds
- `test_logit_parity.py`: Logit stability across prompts
- `test_nll_parity.py`: NLL consistency via teacher-forcing
- `test_greedy_invariants.py`: Determinism properties
- `strategies.py`: Hypothesis strategies for test data
- `metrics.py`: Statistical metrics (KL divergence, cosine similarity)

**Example**:
```python
@given(strategies.prompts(), strategies.max_tokens())
def test_greedy_decode_parity(prompt, max_tokens):
    rust_tokens = rust_model.generate(prompt, max_tokens, greedy=True, seed=42)
    cpp_tokens = cpp_model.generate(prompt, max_tokens, greedy=True, seed=42)
    assert rust_tokens == cpp_tokens, "Deterministic decode must be identical"
```

---

## 8. Existing Trace and Debug Capabilities

### Kernel Execution Recording

The `KernelRecorder` tracks which compute paths executed:

```rust
// In kernel_recorder.rs
pub fn record(&self, id: &'static str) {
    // Records: "i2s_gemv", "rope_apply", "attention_real", etc.
}
```

This provides **layer-to-kernel mapping** but not intermediate tensor capture.

### Parity Test Diagnostics

**File**: `/home/steven/code/Rust/BitNet-rs/crossval/tests/parity_bitnetcpp.rs`

Includes forensic logging:

```rust
// Token ID diagnostics
let prompt_hash = blake3::hash(formatted_prompt.as_bytes());
eprintln!("parity.prompt_hash={}", prompt_hash);

let head = 16.min(rust_ids.len()).min(cpp_ids_u32.len());
eprintln!("parity.rust.head={:?}", &rust_ids[..head]);
eprintln!("parity.cpp.head ={:?}", &cpp_ids_u32[..head]);

// Top-5 token rankings on divergence
let mut rust_top5: Vec<(usize, f32)> = ...
eprintln!("Rust top-5: {:?}", rust_top5);
eprintln!("C++ top-5: {:?}", cpp_top5);
```

### Model Inspection Utilities

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/` (inspect command)

```bash
cargo run -p bitnet-cli -- inspect --ln-stats --gate auto model.gguf
```

Provides:
- LayerNorm statistics (gamma RMS, beta range)
- Projection shape validation
- Architecture-aware envelope checking

---

## 9. Baseline and Receipt Management

### Baseline Storage

**Location**: `/home/steven/code/Rust/BitNet-rs/crossval/baselines.json`

```json
{
  "timestamp": "2025-10-19T14:32:00Z",
  "baselines": {
    "model_default": {
      "tok_s": 42.5,
      "rss_mb": 512.0
    }
  }
}
```

### Receipt Organization

**Location**: `/home/steven/code/Rust/BitNet-rs/ci/receipts/`

```
ci/receipts/
├── pr-466/
│   ├── gate-benchmarks.json
│   ├── gate-merge-readiness.json
│   └── ...
├── pr-465/
│   ├── gate-fixtures.json
│   ├── gate-quality.json
│   └── impl-finalizer-receipt.json
└── issue-465/
    ├── gate-security.json
    ├── gate-fuzz.json
    └── ...

docs/baselines/
├── 2025-10-16/
│   └── parity-bitnetcpp.json
└── 2025-10-19/
    └── parity-bitnetcpp.json
```

---

## 10. xtask Integration

**Location**: `/home/steven/code/Rust/BitNet-rs/xtask/src/`

Key subcommands:

```bash
# Download and setup C++ reference
cargo run -p xtask -- fetch-cpp

# Mapper gate validation
cargo run -p xtask -- mapper-gate model.gguf

# Receipt verification
cargo run -p xtask -- verify-receipt
cargo run -p xtask -- verify-receipt --require-gpu-kernels

# Benchmarking + receipt generation
cargo run -p xtask -- benchmark --model model.gguf --tokens 128
```

**Key Files**:
- `gates.rs`: Validation gate implementations
- `main.rs`: CLI argument parsing and dispatch

---

## 11. Key Insights for Layer-Level Cross-Validation

### Strengths of Existing Infrastructure

1. **Robust Parity Foundation**:
   - Logit tolerance-based comparison (1e-4)
   - Cosine similarity metrics for soft matching
   - Exact token sequence validation
   - Graceful C++ fallback

2. **Honest Compute Verification**:
   - Kernel recorder tracks execution paths
   - Receipt system documents compute integrity
   - 8-gate validation framework
   - CI-enforced verification

3. **Flexible Testing Framework**:
   - Property-based testing with Hypothesis
   - Fixture-based testing with minimal models
   - Criterion benchmarking integration
   - Cross-platform CI support

4. **Production-Ready Diagnostics**:
   - Forensic logging (token hashes, top-K rankings)
   - Model inspection with architecture awareness
   - Performance baseline tracking
   - Timestamped receipt generation

### Gaps for Layer-Level Validation

1. **No Intermediate Tensor Capture**:
   - Kernel recorder only records IDs, not outputs
   - No layer-by-layer activation dumps
   - No mechanism to compare hidden states

2. **No Layer Instrumentation**:
   - Inference engine doesn't expose layer boundaries
   - No hooks for intermediate output interception
   - No layer-level timing breakdown

3. **Limited Forward Pass Tracing**:
   - No step-by-step computation log
   - No per-layer error accumulation tracking
   - No numerical stability diagnostics at layer level

4. **Receipt System Doesn't Track Layer State**:
   - Current receipt: kernel IDs + top-level metrics
   - Missing: per-layer checksum, activation ranges, intermediate tensor shapes

### Recommended Extension Patterns

**Leverage existing infrastructure**:

1. **Extend KernelRecorder**:
   ```rust
   pub struct LayerRecorder {
       inner: Arc<Mutex<Vec<LayerExecution>>>,
   }
   
   pub struct LayerExecution {
       layer_id: usize,
       layer_type: String,  // "embedding", "attention", "mlp", "output"
       kernel_ids: Vec<String>,
       activation_shape: (usize, usize),
       activation_checksum: u64,
       compute_time_us: u64,
   }
   ```

2. **Build on Validation Suite**:
   - Add `validate_layer_activations()` to ValidationSuite
   - Reuse receipt generation patterns
   - Leverage existing comparison utilities

3. **Utilize FFI Bridge**:
   - C++ side already has layer computation
   - Can capture intermediate tensors via FFI
   - Parity tests already show how to call C++ kernels

4. **Property-Based Layer Testing**:
   - Extend Python test suite with layer-level property checks
   - Use Hypothesis to generate input patterns
   - Validate layer invariants (shape, precision, range)

---

## 12. File Reference Guide

### Core Infrastructure

| File | Purpose | Key Content |
|------|---------|-------------|
| `crossval/src/lib.rs` | API entry point | CrossvalConfig, assert_first_logits_match |
| `crossval/src/comparison.rs` | High-level comparison | CrossValidator, ComparisonResult |
| `crossval/src/validation.rs` | Validation gates | ValidationSuite (4 gates) |
| `crossval/src/score.rs` | NLL evaluation | evaluate_perplexity, log_softmax |
| `crossval/src/utils.rs` | Utilities | compare_tokens, compare_floats, perf measurement |
| `crates/bitnet-inference/src/receipts.rs` | Receipt schema v1.0.0 | InferenceReceipt, ParityMetadata, 8 gates |
| `crates/bitnet-inference/src/kernel_recorder.rs` | Kernel tracking | KernelRecorder (thread-safe) |

### Test Suites

| File | Test Type | Key Feature |
|------|-----------|-------------|
| `crossval/tests/parity.rs` | Unit parity | Logit tolerance comparison |
| `crossval/tests/parity_bitnetcpp.rs` | Full harness | Complete inference comparison, receipts |
| `crossval/tests/qk256_crossval.rs` | Quantization | QK256 vs FP32 reference |
| `crossval/tests/token_equivalence.rs` | Tokenizer | Token ID parity |
| `crossval/tests/iq2s_validation.rs` | Quantization | IQ2_S parity |
| `crossval/tests/performance_validation.rs` | Throughput | Latency/TPS comparison |
| `crossval/tests/parity_receipts.rs` | CI validation | Receipt generation and gates |

### FFI and CLI

| File | Purpose |
|------|---------|
| `crossval/src/cpp_bindings.rs` | Safe C++ wrappers |
| `crates/bitnet-sys/` | Low-level FFI (bitnet_eval_tokens, etc.) |
| `xtask/src/main.rs` | CLI dispatch |
| `xtask/src/gates.rs` | Validation gate implementations |

### Configuration and Data

| File | Content |
|------|---------|
| `crossval/Cargo.toml` | Features: crossval, ffi, iq2s-ffi, cpu, gpu |
| `crossval/baselines.json` | Performance baselines by model |
| `ci/inference.json` | Example receipt (current inference run) |
| `ci/receipts/` | Receipt archive organized by PR/issue |
| `crossval/fixtures/` | Test models and data |
| `crossval/docs/baselines/` | Historical baselines by date |

---

## 13. Environment and Build Configuration

### Required Environment Variables

```bash
# Cross-validation setup
export BITNET_CPP_DIR=/path/to/bitnet.cpp      # For FFI compilation
export CROSSVAL_GGUF=/path/to/model.gguf       # Model for parity tests

# Test execution
export PARITY_TEST_TIMEOUT_SECS=300             # Parity test timeout
export BASELINES_DIR=/path/to/baselines         # Baseline location
export BITNET_DETERMINISTIC=1                   # Reproducible inference
export BITNET_STRICT_MODE=1                     # Strict validation

# Build flags
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
```

### Build Commands

```bash
# Build with cross-validation enabled
cargo build --no-default-features --features crossval

# Run parity tests
cargo test -p bitnet-crossval --features crossval,integration-tests

# Run with C++ reference
BITNET_CPP_DIR=/path/to/bitnet.cpp \
  CROSSVAL_GGUF=models/model.gguf \
  cargo test -p bitnet-crossval --features crossval,integration-tests

# Benchmark
cargo bench -p bitnet-crossval --features crossval

# Generate receipts
cargo run -p xtask -- benchmark --model model.gguf --tokens 128
```

---

## 14. Recommendation: Next Steps for Layer-Level Implementation

1. **Phase 1: Infrastructure (Week 1)**
   - Extend `LayerRecorder` based on `KernelRecorder` pattern
   - Add layer-level fields to receipt schema (v1.1.0-beta)
   - Create `LayerCapture` struct for intermediate tensor metadata

2. **Phase 2: Instrumentation (Week 2-3)**
   - Add layer hooks in inference engine (embedding, attention, mlp, output)
   - Implement tensor capture at layer boundaries
   - Integrate with existing kernel recorder

3. **Phase 3: Comparison (Week 3-4)**
   - Extend `CrossValidator` with `compare_layer_activations()`
   - Implement layer-level cosine similarity
   - Add to validation suite as new gate

4. **Phase 4: Testing & Validation (Week 4-5)**
   - Property-based layer tests
   - Parity suite extension for layer outputs
   - Receipt generation with layer checksums

---

## Conclusion

BitNet-rs has a **mature, well-designed cross-validation framework** ready for extension to layer-level comparison. The kernel recorder, receipt system, FFI bridge, and parity harness provide excellent foundations. Primary work is:

1. Extending infrastructure to capture intermediate tensors
2. Adding layer-level instrumentation hooks
3. Building comparison logic on top of existing utilities
4. Integrating receipts with layer metadata

All existing patterns (threading, error handling, feature gates) can be reused directly.

