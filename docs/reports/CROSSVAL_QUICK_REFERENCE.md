# BitNet-rs Cross-Validation Infrastructure - Quick Reference

## What Exists (Mature, Production-Ready)

### 1. Dedicated `crossval` Crate
- **Location**: `/home/steven/code/Rust/BitNet-rs/crossval/`
- **Purpose**: Compare Rust vs C++ inference implementations
- **Key Modules**:
  - `comparison.rs`: High-level parity runner
  - `validation.rs`: 4 validation gates (compatibility, token parity, NLL, performance)
  - `score.rs`: NLL/perplexity evaluation via teacher-forcing
  - `utils.rs`: Token/float comparison, performance measurement
  - `cpp_bindings.rs`: Safe FFI wrappers to C++ model

### 2. Receipt System (Schema v1.0.0)
- **File**: `crates/bitnet-inference/src/receipts.rs`
- **Purpose**: Document honest compute in CI/CD
- **Contains**:
  - Compute path ("real" vs "mock")
  - Backend identification (cpu/cuda/metal)
  - Kernel IDs (["i2s_gemv", "rope_apply", ...])
  - Model info (SHA256, dimensions)
  - Performance metrics (tok/s)
  - Parity metadata (cosine similarity, exact match rate)
- **Validation**: 8 gates enforced in CI (path, backend, kernels, determinism, GPU requirement, model info, performance, parity)

### 3. Kernel Recorder
- **File**: `crates/bitnet-inference/src/kernel_recorder.rs`
- **Type**: Thread-safe (`Arc<Mutex<Vec<String>>>`)
- **API**:
  - `record(id: &'static str)`: O(1) append
  - `snapshot() -> Vec<String>`: Deduplicated, insertion-order
- **Purpose**: Track which kernels executed during inference

### 4. Parity Harness
- **Primary File**: `crossval/tests/parity_bitnetcpp.rs`
- **Metrics**:
  - Logit tolerance: 1e-4
  - Cosine similarity: ≥ 0.99 for soft match
  - Token ID exact match for hard match
  - Top-5 token ranking diagnostics
- **Features**:
  - Graceful fallback if C++ not available
  - Forensic logging (token hashes, head/tail comparison)
  - Receipt generation per test run
  - Timeout protection (configurable via env var)

### 5. FFI Bridge
- **Lower level**: `crates/bitnet-sys/` (raw C bindings)
- **Higher level**: `crossval/src/cpp_bindings.rs` (safe wrappers)
- **Used for**: Direct comparison with C++ reference implementation
- **Build**: Feature-gated (`crossval` feature requires `BITNET_CPP_DIR`)

### 6. Validation Gates
- **File**: `crossval/src/validation.rs`
- **Gates**:
  1. Model compatibility (tensor mapping)
  2. Token parity (95%+ match)
  3. NLL parity (delta < 0.01)
  4. Performance (≥ 1.0 tok/s, ratio ≥ 0.95)

### 7. Test Suites
- `parity.rs`: Unit-level logit comparison
- `parity_bitnetcpp.rs`: Full inference harness
- `qk256_crossval.rs`: Quantization kernel validation
- `token_equivalence.rs`: Tokenizer parity
- `iq2s_validation.rs`: IQ2_S quantization parity
- `performance_validation.rs`: Throughput/latency
- `parity_receipts.rs`: Receipt generation and validation

### 8. Baseline Management
- **Location**: `crossval/baselines.json`
- **Purpose**: Track performance trends
- **Format**: Model → throughput (tok/s), memory (MB)
- **Historical**: `docs/baselines/<YYYY-MM-DD>/parity-bitnetcpp.json`

### 9. Property-Based Testing
- **Location**: `crossval/props/` (Python + Hypothesis)
- **Tests**:
  - Greedy decode parity across seeds
  - Logit stability across prompts
  - NLL consistency
  - Determinism invariants

## What's Missing for Layer-Level Cross-Validation

### No Intermediate Tensor Capture
- Kernel recorder only tracks IDs, not tensor outputs
- No mechanism to dump activations at layer boundaries
- No layer-by-layer checksum verification

### No Layer Instrumentation Hooks
- Inference engine doesn't expose layer boundaries
- No way to intercept hidden states
- No layer-level timing breakdown

### No Layer Receipt Metadata
- Current receipt: top-level metrics only
- Missing: per-layer activation ranges, checksums, shapes
- No layer divergence tracking

## Building Block: How to Extend

### Pattern 1: Extend KernelRecorder
```rust
pub struct LayerRecorder {
    inner: Arc<Mutex<Vec<LayerExecution>>>,
}

pub struct LayerExecution {
    layer_id: usize,
    layer_type: String,  // "embedding", "attention", "mlp"
    output_shape: (usize, usize),
    output_checksum: u64,
}
```

### Pattern 2: Extend Validation Suite
```rust
impl ValidationSuite {
    pub fn validate_layer_activations(&self) -> Result<LayerValidationResult>;
}
```

### Pattern 3: Extend Receipt Schema
Add to `InferenceReceipt`:
```rust
pub layers: Option<Vec<LayerMetadata>> {
    pub layer_id: usize,
    pub output_checksum: u64,
    pub activation_stats: ActivationStats,
}
```

### Pattern 4: FFI for Layer Outputs
Use existing C++ bridge to capture intermediate tensors:
```rust
let cpp_layer_output = cpp_model.get_layer_output(layer_id)?;
```

## Key Environments Variables

```bash
BITNET_CPP_DIR=/path/to/bitnet.cpp        # For FFI build
CROSSVAL_GGUF=/path/to/model.gguf         # Parity test model
PARITY_TEST_TIMEOUT_SECS=300              # Test timeout
BASELINES_DIR=/path/to/baselines          # Baseline location
BITNET_DETERMINISTIC=1                    # Reproducible inference
BITNET_STRICT_MODE=1                      # Strict validation
```

## Build Commands

```bash
# With cross-validation
cargo build --no-default-features --features crossval

# Run parity tests
BITNET_CPP_DIR=/path/to/bitnet.cpp \
  CROSSVAL_GGUF=models/model.gguf \
  cargo test -p bitnet-crossval --features crossval,integration-tests

# Benchmark + receipt
cargo run -p xtask -- benchmark --model model.gguf --tokens 128

# Verify receipt
cargo run -p xtask -- verify-receipt
```

## File Map - Core Infrastructure

| File | Purpose |
|------|---------|
| `crossval/src/comparison.rs` | CrossValidator (high-level parity) |
| `crossval/src/validation.rs` | ValidationSuite (4 gates) |
| `crossval/src/score.rs` | NLL evaluation + parity check |
| `crates/bitnet-inference/src/receipts.rs` | Receipt schema + structures |
| `crates/bitnet-inference/src/kernel_recorder.rs` | Kernel tracking |
| `crossval/tests/parity_bitnetcpp.rs` | Full parity harness |
| `crossval/tests/qk256_crossval.rs` | Quantization validation |
| `crates/bitnet-sys/` | FFI bindings |
| `xtask/src/gates.rs` | Validation gate implementation |

## Design Patterns to Reuse

1. **Thread-safe Recording**: `Arc<Mutex<Vec<T>>>` pattern (kernel recorder)
2. **Feature Gating**: `#[cfg(feature = "crossval")]` for zero overhead
3. **Tolerance-based Comparison**: 1e-4 for logits, cosine ≥ 0.99 for soft match
4. **Graceful Fallback**: Tests skip if C++ not available
5. **Timestamped Artifacts**: RFC3339 format in receipts
6. **Property-Based Testing**: Hypothesis + manual invariant checks
7. **CI Integration**: Exit codes, structured JSON output

## Conclusion

BitNet-rs has **exceptional cross-validation infrastructure**. To add layer-level comparison:

1. Extend `KernelRecorder` → `LayerRecorder` (capture outputs)
2. Add instrumentation hooks in inference engine
3. Extend `ValidationSuite` with layer validation
4. Update receipt schema to include layer metadata
5. Leverage existing FFI, comparison, and test patterns

No major architectural changes needed - primarily extension of existing, well-designed systems.

