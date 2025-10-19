# Issue #469 API Contracts - MVP Sprint Polish

**Document Status:** API Reference
**Created:** 2025-10-18
**Issue:** #469
**Targets:** v0.1.0-mvp release

---

## Overview

This document defines the public API contracts for Issue #469's 8 acceptance criteria. All contracts follow Rust API design guidelines and BitNet.rs workspace conventions.

---

## AC1: Strict Loader Mode CLI Contract

### CLI Argument

```rust
/// Strict GGUF loader mode flag
///
/// # Contract
/// - Flag: `--strict-loader`
/// - Type: `bool`
/// - Default: `false` (permissive mode, backward compatible)
/// - Scope: Global (affects all GGUF loading operations)
///
/// # Behavior
/// - `--strict-loader=true`: Reject ANY size deviation in QK256 tensors
/// - `--strict-loader=false`: Allow up to 0.1% tolerance for alignment padding
///
/// # Error Messages
/// - MUST include: tensor name, expected size, actual size, deviation %
/// - MUST provide: actionable hints (use --strict-loader=false, regenerate GGUF)
///
/// # Example
/// ```bash
/// # Strict mode (fail-fast)
/// bitnet run --model model.gguf --strict-loader --prompt "Test"
///
/// # Permissive mode (default)
/// bitnet run --model model.gguf --prompt "Test"
/// ```
#[arg(long = "strict-loader", default_value_t = false)]
pub strict_loader: bool;
```

### Loader Configuration Contract

```rust
/// GGUF loader configuration
///
/// # Stability: Stable
/// # Since: v0.1.0-mvp
///
/// # Contract
/// - `strict_mode`: Boolean flag for tolerance enforcement
/// - `tolerance_bytes`: Maximum allowed deviation (ignored in strict mode)
/// - Default: Permissive mode with 0.1% tolerance
///
/// # Backward Compatibility
/// - Default behavior MUST remain permissive (strict_mode=false)
/// - tolerance_bytes MUST use centralized constant (AC2: qk256_tolerance_bytes)
///
/// # Thread Safety
/// - Immutable after construction (Send + Sync safe)
#[derive(Debug, Clone)]
pub struct GGUFLoaderConfig {
    pub strict_mode: bool,
    pub tolerance_bytes: usize,
}

impl Default for GGUFLoaderConfig {
    fn default() -> Self {
        Self {
            strict_mode: false,
            tolerance_bytes: qk256_tolerance_bytes(131_072), // AC2: centralized
        }
    }
}
```

### Validation Function Contract

```rust
/// Validate QK256 tensor size
///
/// # Contract
/// - MUST check actual_bytes vs expected_bytes
/// - MUST apply tolerance based on config.strict_mode
/// - MUST log standardized messages (AC2 format)
/// - MUST return anyhow::Result (bail on strict mode violation)
///
/// # Error Handling
/// - Strict mode: anyhow::bail! on ANY deviation
/// - Permissive mode: log::warn! on deviation within tolerance
///
/// # Thread Safety: Safe (no shared mutable state)
fn validate_qk256_tensor_size(
    tensor_name: &str,
    actual_bytes: usize,
    expected_bytes: usize,
    config: &GGUFLoaderConfig,
) -> anyhow::Result<()>;
```

---

## AC2: QK256 Tolerance Constants Contract

### Tolerance Constant

```rust
/// QK256 size tolerance percentage (0.1%)
///
/// # Stability: Stable
/// # Since: v0.1.0-mvp
/// # Value: 0.001 (0.1%)
///
/// # Contract
/// - MUST NOT change without major version bump
/// - MUST be public (exported from bitnet-quantization)
/// - MUST be documented in docs/reference/quantization-support.md
///
/// # Rationale
/// - Accounts for GGUF alignment padding (typically 0-128 bytes)
/// - Rejects structurally corrupted tensors (>0.1% deviation)
/// - Empirically validated across LLaMA, BitNet models
pub const QK256_SIZE_TOLERANCE_PERCENT: f64 = 0.001;
```

### Helper Function Contract

```rust
/// Calculate tolerance bytes for QK256 tensor
///
/// # Stability: Stable
/// # Since: v0.1.0-mvp
///
/// # Contract
/// - MUST apply QK256_SIZE_TOLERANCE_PERCENT to expected_bytes
/// - MUST return ceiling (round up for conservative tolerance)
/// - MUST handle edge case: 0 bytes → 0 tolerance
///
/// # Thread Safety: Safe (pure function, no side effects)
///
/// # Example
/// ```rust
/// assert_eq!(qk256_tolerance_bytes(1_000_000), 1_000); // 1MB → 1KB
/// assert_eq!(qk256_tolerance_bytes(131_072), 131);     // 128KB → 131B
/// assert_eq!(qk256_tolerance_bytes(0), 0);             // Edge case
/// ```
pub fn qk256_tolerance_bytes(expected_bytes: usize) -> usize;
```

### Logging Format Contract

```rust
/// QK256 size mismatch logging format
///
/// # Contract
/// - MUST include: mode (strict|permissive), tensor name, expected, actual, deviation%, threshold%, action
/// - MUST use standardized format for log parsing
/// - Strict mode: log::error! with "REJECTED"
/// - Permissive mode: log::warn! with "ACCEPTED with tolerance"
///
/// # Format Spec
/// "QK256 size mismatch ({mode}): tensor='{name}', expected={exp}B, actual={act}B, \
///  deviation={dev:+.2}% (threshold={thr:.2}%), {ACTION}"
///
/// # Example
/// log::warn!(
///     "QK256 size mismatch (permissive): tensor='blk.0.attn_q.weight', \
///      expected=1048576B, actual=1049000B, deviation=+0.04% (threshold=0.10%), \
///      ACCEPTED with tolerance"
/// );
```

---

## AC3: K/V Cache Validation Contract

### Validation Function Contract

```rust
/// Validate K/V cache dimensions
///
/// # Stability: Stable
/// # Since: v0.1.0-mvp
///
/// # Contract
/// - MUST validate 4D tensor shape: [batch, n_heads, seq_len, head_dim]
/// - MUST use debug_assert! for hot-path checks (zero overhead in release)
/// - MUST emit once-per-layer warnings (no log spam)
/// - MUST return anyhow::Result on structural errors
///
/// # Thread Safety: Safe (uses thread-safe Once guards)
///
/// # Performance
/// - Hot path: debug_assert! only (compiled out in release)
/// - Cold path: Once-per-layer warning (amortized cost)
///
/// # Example
/// ```rust
/// validate_kv_cache_dims(
///     &k_cache,
///     layer_idx=0,
///     expected_batch=1,
///     expected_n_heads=16,
///     max_seq_len=2048,
///     expected_head_dim=64,
/// )?;
/// ```
pub fn validate_kv_cache_dims(
    tensor: &Tensor,
    layer_idx: usize,
    expected_batch: usize,
    expected_n_heads: usize,
    max_seq_len: usize,
    expected_head_dim: usize,
) -> anyhow::Result<()>;
```

### Once-Per-Layer Warning Contract

```rust
/// Emit warning only once per layer
///
/// # Stability: Internal (not public API)
///
/// # Contract
/// - MUST use static Once guards (one per layer, max 64)
/// - MUST be thread-safe (Once::call_once is atomic)
/// - MUST emit log::warn! on first call, no-op on subsequent calls
/// - MUST log::error! if layer_idx exceeds WARNING_FLAGS.len()
///
/// # Thread Safety: Guaranteed by std::sync::Once
fn emit_once_per_layer_warning(layer_idx: usize, message: String);
```

### K/V Cache Initialization Contract

```rust
/// Initialize K/V cache
///
/// # Contract
/// - MUST validate num_layers > 0
/// - MUST validate num_heads > 0
/// - MUST validate head_dim > 0 AND divisible by 4 (SIMD alignment)
/// - MUST validate max_seq_len > 0
/// - MUST return anyhow::Result on invalid config
///
/// # Thread Safety: Safe (immutable after construction)
impl KVCache {
    pub fn new(
        max_seq_len: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> anyhow::Result<Self>;
}
```

---

## AC4: Parity Receipt Schema v1.0.0 Contract

### ParityMetadata Struct Contract

```rust
/// Parity validation metadata
///
/// # Stability: Stable
/// # Schema Version: 1.0.0
/// # Since: v0.1.0-mvp
///
/// # Contract
/// - MUST serialize to JSON (Serialize + Deserialize)
/// - MUST be optional field in InferenceReceipt
/// - MUST validate cosine_similarity in [0.0, 1.0]
/// - MUST validate exact_match_rate in [0.0, 1.0]
/// - MUST validate status in ["ok", "warn", "error", "rust_only"]
///
/// # Status Gates
/// - "ok": cosine_similarity ≥ 0.99 AND exact_match_rate ≥ 0.95
/// - "warn": cosine_similarity ≥ 0.95 (marginal)
/// - "error": cosine_similarity < 0.95 (unacceptable)
/// - "rust_only": C++ reference not available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityMetadata {
    pub cpp_available: bool,
    pub cosine_similarity: f64,
    pub exact_match_rate: f64,
    pub status: String,
}
```

### Receipt Validation Contract

```rust
/// Validate receipt schema v1.0.0
///
/// # Contract
/// - MUST validate base schema (AC9 requirements)
/// - MUST validate parity field (if present)
/// - MUST check status consistency with metrics
/// - MUST return anyhow::Result with descriptive errors
///
/// # Thread Safety: Safe (immutable self)
impl InferenceReceipt {
    pub fn validate_schema_v1(&self) -> anyhow::Result<()>;
}
```

### Timeout Constant Contract

```rust
/// Default inference and parity timeout
///
/// # Stability: Stable
/// # Since: v0.1.0-mvp
/// # Value: 60 seconds
///
/// # Contract
/// - MUST be shared between main inference and parity harness
/// - MUST NOT diverge between constants
/// - MUST be configurable via function parameter (not just constant)
///
/// # Usage
/// ```rust
/// use tokio::time::{timeout, Duration};
/// timeout(Duration::from_secs(DEFAULT_PARITY_TIMEOUT_SECS), async_inference).await
/// ```
pub const DEFAULT_INFERENCE_TIMEOUT_SECS: u64 = 60;
pub const DEFAULT_PARITY_TIMEOUT_SECS: u64 = DEFAULT_INFERENCE_TIMEOUT_SECS;
```

### Parity Harness Contract

```rust
/// Run parity test with receipt generation
///
/// # Stability: Stable (crossval feature)
/// # Since: v0.1.0-mvp
///
/// # Contract
/// - MUST run Rust inference with timeout
/// - MUST run C++ inference if BITNET_CPP_DIR set
/// - MUST calculate cosine_similarity and exact_match_rate
/// - MUST generate InferenceReceipt with parity field
/// - MUST validate receipt before returning
/// - MUST return anyhow::Result on timeout or validation failure
///
/// # Thread Safety: Async-safe (uses tokio::time::timeout)
pub async fn run_parity_test(
    model_path: &str,
    tokens: &[u32],
    timeout_secs: Option<u64>,
) -> anyhow::Result<InferenceReceipt>;
```

---

## AC5: Tokenizer Parity Contract

### Tokenizer Trait Extension Contract

```rust
/// Tokenizer trait
///
/// # Stability: Stable
/// # Since: v0.1.0-mvp
///
/// # New Method: real_vocab_size (AC5)
/// - MUST return actual vocabulary size (no GGUF padding)
/// - MUST have default implementation (backward compatible)
/// - MUST be used for cross-validation parity assertions
///
/// # Backward Compatibility
/// - Existing implementations continue to work (default impl)
/// - New implementations SHOULD override for accurate parity
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
    fn vocab_size(&self) -> usize;

    /// Real vocabulary size (no padding)
    ///
    /// # Contract
    /// - Default: returns vocab_size() (backward compatible)
    /// - GGUF tokenizers: returns real size from tokenizer.ggml.tokens array
    /// - HF tokenizers: returns size without special tokens
    fn real_vocab_size(&self) -> usize {
        self.vocab_size()
    }
}
```

### GGUF Tokenizer Implementation Contract

```rust
/// GGUF tokenizer
///
/// # Contract
/// - MUST distinguish real_vocab_size from padded_vocab_size
/// - MUST detect padding from GGUF metadata
/// - MUST log padding amount for diagnostics
/// - vocab_size() returns padded size (GGUF alignment)
/// - real_vocab_size() returns real size (tokenizer model)
impl Tokenizer for GgufTokenizer {
    fn vocab_size(&self) -> usize;       // Padded (GGUF alignment)
    fn real_vocab_size(&self) -> usize;  // Real (tokenizer model)
}
```

### Parity Assertion Contract

```rust
/// Validate tokenizer parity
///
/// # Contract
/// - MUST use real_vocab_size() for Rust-C++ comparison
/// - MUST log both real and padded sizes for diagnostics
/// - MUST return anyhow::Result with detailed error message
///
/// # Error Message Format
/// "Tokenizer vocab size mismatch breaks parity: \
///  Rust real_vocab_size={r_real}, Rust padded_vocab_size={r_pad}, \
///  C++ vocab_size={cpp}. Rust-C++ mismatch: {diff} tokens."
pub fn validate_tokenizer_parity(
    rust_tokenizer: &dyn Tokenizer,
    cpp_vocab_size: usize,
) -> anyhow::Result<()>;
```

---

## AC6: FFI Build Hygiene Contract

### Unified Compilation Function Contract

```rust
/// Compile C++ shim with unified hygiene settings
///
/// # Stability: Stable (build-time API)
/// # Since: v0.1.0-mvp
///
/// # Contract
/// - MUST use -isystem for system_include_dirs (suppress warnings)
/// - MUST use -I for include_dirs (show warnings)
/// - MUST support C++17 standard
/// - MUST compile with -O2 -fPIC
/// - MUST suppress external header warnings (-Wno-unknown-pragmas, -Wno-deprecated-declarations)
///
/// # Thread Safety: Not applicable (build-time only)
///
/// # Example
/// ```rust
/// compile_cpp_shim(
///     Path::new("csrc/shim.cc"),
///     "bitnet_shim",
///     &[PathBuf::from("csrc/")],  // Show warnings
///     &[PathBuf::from("/usr/local/cuda/include")],  // Suppress warnings
/// )?;
/// ```
pub fn compile_cpp_shim(
    shim_path: &Path,
    output_name: &str,
    include_dirs: &[PathBuf],
    system_include_dirs: &[PathBuf],
) -> Result<(), Box<dyn std::error::Error>>;
```

### System Include Helpers Contract

```rust
/// Get CUDA system include directories
///
/// # Contract
/// - MUST return standard CUDA paths (/usr/local/cuda/include, targets/)
/// - MUST NOT fail (returns best-effort paths)
pub fn cuda_system_includes() -> Vec<PathBuf>;

/// Get BitNet C++ reference include directories
///
/// # Contract
/// - MUST read BITNET_CPP_DIR or fallback to $HOME/.cache/bitnet_cpp
/// - MUST return include/, 3rdparty/llama.cpp paths
/// - MUST return Err if BITNET_CPP_DIR not set and no HOME
pub fn bitnet_cpp_system_includes() -> Result<Vec<PathBuf>, Box<dyn std::error::Error>>;
```

---

## AC7: CI Parity Smoke Test Contract

### CI Workflow Contract

```yaml
# Contract
# - MUST set BITNET_DISABLE_MINIMAL_LOADER=1 (enforce enhanced loader)
# - MUST test both BitNet32-F16 and QK256 formats independently
# - MUST verify receipt generation at workspace root
# - MUST fail CI if cosine_similarity < 0.99 (for C++ available case)
# - MUST validate I2_S flavor in receipt JSON

env:
  BITNET_DISABLE_MINIMAL_LOADER: 1
  BITNET_DETERMINISTIC: 1
  BITNET_SEED: 42
  RAYON_NUM_THREADS: 1
```

### Parity Smoke Script Contract

```bash
# Contract
# - MUST export BITNET_DISABLE_MINIMAL_LOADER=1
# - MUST support BITNET_STRICT_MODE=1 (optional, caller-controlled)
# - MUST validate I2_S flavor in receipt
# - MUST exit with non-zero on parity failure

export BITNET_DISABLE_MINIMAL_LOADER=1

if [ -n "$BITNET_STRICT_MODE" ]; then
    echo "Running in STRICT MODE"
fi

# Run parity test and validate receipt
./scripts/parity_smoke.sh model.gguf
```

---

## AC8: Documentation Contract

### README Quick-Start Contract

```markdown
# Contract
# - MUST include QK256 quick-start section
# - MUST show both automatic and strict loader examples
# - MUST document I2_S dual-flavor support
# - MUST provide cross-validation command examples
# - MUST cross-link to docs/howto/use-qk256-models.md

## Quick Start

### QK256 Format (GGML I2_S, 256-element blocks)

[automatic detection example]
[strict loader example]
[cross-validation example]

**Learn More:**
- [QK256 Usage Guide](docs/howto/use-qk256-models.md)
```

### docs/quickstart.md Contract

```markdown
# Contract
# - MUST include "Using QK256 Models" section
# - MUST document automatic flavor detection
# - MUST document strict loader mode usage
# - MUST document cross-validation workflow
# - MUST provide receipt validation examples

## Using QK256 Models (GGML I2_S)

### Automatic Format Detection
[example]

### Strict Loader Mode
[example with --strict-loader]

### Cross-Validation
[example with cargo run -p xtask -- crossval]
```

---

## Breaking Change Policy

**None of these changes are breaking:**

1. **AC1:** New CLI flag (additive, default preserves existing behavior)
2. **AC2:** New constants (additive, internal refactoring)
3. **AC3:** New validation module (additive, defensive checks)
4. **AC4:** Receipt schema extension (optional field, backward compatible)
5. **AC5:** Trait default method (backward compatible)
6. **AC6:** Build-time consolidation (no runtime API change)
7. **AC7:** CI enhancement (no production code change)
8. **AC8:** Documentation addition (no code change)

**Stability Guarantees:**
- All public APIs marked `Stability: Stable` MUST NOT change without SemVer major bump
- Schema versions (v1.0.0) MUST maintain backward compatibility within major version
- Default behaviors MUST remain unchanged for backward compatibility

---

## Thread Safety Guarantees

All contracts specify thread safety:
- **Safe:** No shared mutable state, Send + Sync safe
- **Async-safe:** Compatible with tokio async runtime
- **Build-time only:** Not applicable (runs at build time, single-threaded)

---

**Document Control:**
- Review Status: API Reference (Ready for Implementation)
- Owner: BitNet.rs Architecture Team
- Issue: #469
- Target: v0.1.0-mvp
