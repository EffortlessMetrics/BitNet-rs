# Crossval Crate Quick Reference

## File Locations

| File | Path | Purpose |
|------|------|---------|
| **Main Library** | `crossval/src/lib.rs` | Root module & error types |
| **Token Parity** | `crossval/src/token_parity.rs` | Pre-gate validation (475 LOC) |
| **Logits Compare** | `crossval/src/logits_compare.rs` | Per-position divergence (200 LOC) |
| **FFI Bindings** | `crossval/src/cpp_bindings.rs` | Safe C++ wrappers (300 LOC) |
| **Comparison** | `crossval/src/comparison.rs` | Orchestration logic (150 LOC) |
| **Validation Suite** | `crossval/src/validation.rs` | Comprehensive gates (200 LOC) |
| **Utilities** | `crossval/src/utils.rs` | Helpers & performance (150 LOC) |
| **Fixtures** | `crossval/src/fixtures.rs` | Test data management (94 LOC) |
| **Score** | `crossval/src/score.rs` | NLL/Perplexity (200 LOC) |
| **C Wrapper** | `crossval/src/bitnet_cpp_wrapper.c` | Extern C functions (64 LOC) |
| **Build Script** | `crossval/build.rs` | FFI compilation (119 LOC) |
| **Configuration** | `crossval/Cargo.toml` | Features & dependencies |

## Key Entry Points

### Public API Functions

```rust
// Token validation (fail-fast pre-gate)
pub fn validate_token_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[i32],
    prompt: &str,
) -> anyhow::Result<()>

// Logits comparison
pub fn compare_per_position_logits(
    rs_logits: &[Vec<f32>],
    cpp_logits: &[Vec<f32>],
) -> LogitsDivergence

// Validation suite
impl ValidationSuite {
    pub fn validate_model_compatibility(&self) -> Result<ValidationResult>
    pub fn validate_token_parity(&self, prompts: &[String]) -> Result<TokenParityResult>
    pub fn validate_nll_parity(&self, dataset: &str) -> Result<NllParityResult>
    pub fn validate_performance(&self, baseline: Option<&Path>) -> Result<PerformanceResult>
}

// FFI bindings
impl CppModel {
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self>
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<Vec<u32>>
    pub fn is_ready(&self) -> bool
}
```

## FFI Boundary

```
Rust Code
  ↓
cpp_bindings.rs (safe wrapper)
  ↓
bitnet_cpp_wrapper.c (extern "C")
  ↓
C++ libraries (llama.cpp, ggml, bitnet.cpp)
```

**Build-time library search (prioritized):**
1. `$BITNET_CROSSVAL_LIBDIR` (explicit)
2. `$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src`
3. `$BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src`
4. `$BITNET_CPP_DIR/build/bin`
5. `$BITNET_CPP_DIR/build/lib`
6. `$BITNET_CPP_DIR/lib`
7. `$BITNET_CPP_DIR/build`

## Feature Flags

```toml
default = []                    # No defaults

# Main crossval features
crossval = [                    # Full validation with C++ FFI
    "dep:bindgen",
    "dep:cc",
    "dep:bitnet-sys",
    "bitnet-sys/ffi",
    "bitnet-inference/ffi",
    "ffi"
]
ffi = ["dep:cc", "bitnet-inference/ffi"]
iq2s-ffi = ["bitnet-models/iq2s-ffi", "bitnet-ggml-ffi/iq2s-ffi"]

# Backend selection
cpu = ["bitnet-inference/cpu", "bitnet-models/cpu"]
gpu = ["bitnet-inference/gpu", "bitnet-models/gpu"]

# Special modes
cpp-probe = []                  # C++ detection
integration-tests = []          # Expensive tests
```

## Test Modules (13 tests, ~165 KB)

| Test | LOC | Purpose | Requirements |
|------|-----|---------|--------------|
| `smoke.rs` | 2.9 KB | Environment checks | - |
| `parity_bitnetcpp.rs` | 34.7 KB | Real parity (async) | crossval + integration-tests |
| `parity_receipts.rs` | 18.7 KB | Receipt validation | - |
| `parity.rs` | 12.8 KB | Logits comparison | crossval + integration-tests |
| `per_position_logits.rs` | 10.5 KB | Divergence detection | crossval + integration-tests |
| `qk256_crossval.rs` | 15.8 KB | QK256 vs FP32 | - |
| `ffi_integration.rs` | 8.1 KB | FFI lifecycle | ffi |
| `framework_validation.rs` | 17.8 KB | Validation suite | - |
| `performance_validation.rs` | 19.5 KB | Throughput/memory | - |
| `iq2s_validation.rs` | 10.4 KB | IQ2_S parity | iq2s-ffi |
| `token_equivalence.rs` | 5.0 KB | Token A/B testing | ffi |
| `cpp_probe.rs` | 0.4 KB | C++ detection | - |
| `ms_bitnet_mapping.rs` | 1.0 KB | Model mapping | - |

## Configuration Patterns

### Environment Variables

```bash
# Build-time
BITNET_CPP_DIR=/path/to/bitnet.cpp
BITNET_CROSSVAL_LIBDIR=/path/to/libs

# Runtime
CROSSVAL_GGUF=/path/to/model.gguf
BITNET_DETERMINISTIC=1
BITNET_SEED=42
RAYON_NUM_THREADS=1
LD_LIBRARY_PATH=/path/to/libs
DYLD_LIBRARY_PATH=/path/to/libs
```

### CrossvalConfig

```rust
pub struct CrossvalConfig {
    pub tolerance: f64,      // Default: 1e-6 (logits)
    pub max_tokens: usize,   // Default: 1000
    pub benchmark: bool,     // Default: false
}
```

## Data Flow

### Token Parity Pre-Gate
```
Prompt
  ↓
[Rust Tokenizer] → Vec<u32>
[C++ Tokenizer]  → Vec<i32>
  ↓
validate_token_parity()
  → If match: Ok(()) [silent]
  → If mismatch: Err + colored diagnostic
```

### Logits Comparison
```
Tokens
  ↓
[Rust Engine] → Vec<Vec<f32>>
[C++ Engine]  → Vec<Vec<f32>>
  ↓
compare_per_position_logits()
  → LogitsDivergence {
      first_divergence_token,
      per_token_cosine_sim,
      per_token_l2_dist,
      max_absolute_diff
    }
```

### Receipt Output
```json
{
  "timestamp": "2025-10-25T...",
  "model_path": "model.gguf",
  "prompt": "What is 2+2?",
  "compute_path": "real",
  "backend": "cpu|cuda",
  "parity": {
    "cpp_available": true,
    "cosine_similarity": 0.9923,
    "exact_match_rate": 1.0,
    "status": "ok"
  },
  "kernel_ids": ["scalar_i2s_qk256"],
  "validation_gates": {
    "token_parity": "pass",
    "logits_parity": "pass"
  }
}
```

## Error Handling

```rust
pub enum CrossvalError {
    CppNotAvailable,
    ModelLoadError(String),
    InferenceError(String),
    ComparisonError(String),
    IoError(#[from] std::io::Error),
    SerializationError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, CrossvalError>;
```

## Command Integration (xtask)

```bash
# Per-token validation
cargo run -p xtask -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4

# Full sweep
cargo run -p xtask -- crossval \
  --model model.gguf \
  --cpp-dir ~/.cache/bitnet_cpp

# Setup
cargo run -p xtask -- setup-crossval
```

## Test Execution

```bash
# All tests
cargo test --workspace --features cpu

# With nextest (recommended)
cargo nextest run --workspace --features cpu

# With C++ FFI
BITNET_CPP_DIR=/path cargo test --features crossval

# Specific test module
cargo test -p bitnet-crossval --test parity
```

## Known Limitations

### C++ Wrapper Status
- **Current**: Mock implementation (checks file exists only)
- **Actual functions**: Hardcoded dummy results
- **Workaround**: Set `BITNET_CPP_DIR` + build with real C++ libs

### Blocked Tests (Issue Dependencies)
- Issue #254 (shape mismatch in layer norm)
- Issue #260 (mock elimination)
- Issue #469 (tokenizer parity + FFI hygiene)

### Coverage Gaps
- No subprocess exit code test
- No stderr capture test
- No negative token handling test
- No large model stress test

## Key Design Decisions

### Token Parity Pre-Gate
- **Why**: Fail-fast before expensive inference
- **Input**: rust_tokens (u32), cpp_tokens (i32), prompt
- **Output**: Ok(()) or Err with diagnostic
- **Exit Code**: Caller should exit(2) on error

### Per-Position Logits
- **Why**: Identify exact divergence point
- **Metrics**: Cosine similarity, L2 distance
- **Threshold**: 1e-4 (cosine), 1e-4 (absolute diff)
- **Scope**: Multi-token generation

### Receipt JSON
- **Why**: CI/CD integration, reproducibility
- **Sections**: Metadata, parity metrics, kernel IDs, gates
- **Versioned**: Schema v1.0.0
- **Signed**: SHA256 of model + prompt

## Dependencies Summary

| Category | Libraries |
|----------|-----------|
| **Serialization** | serde, serde_json |
| **Error Handling** | anyhow, thiserror |
| **FFI** | cc, bindgen, bitnet-sys |
| **Inference** | bitnet-inference, bitnet-models |
| **Tokenization** | bitnet-tokenizers |
| **Terminal UI** | console |
| **Utilities** | scopeguard, dirs, humantime |
| **Async** | tokio |
| **Hashing** | sha2, blake3 |
| **Benchmarking** | criterion |

---

**Last Updated**: 2025-10-25
**Report Generated From**: `/home/steven/code/Rust/BitNet-rs/crossval/`
**Full Report**: See `CROSSVAL_EXPLORATION_REPORT.md`
