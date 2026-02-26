# Issue #469 MVP Sprint Polish - Technical Specification

**Status:** Draft Technical Specification
**Created:** 2025-10-18
**Author:** BitNet-rs spec-analyzer
**Issue:** #469
**Targets:** v0.1.0-mvp release

---

## Executive Summary

This specification provides a comprehensive technical analysis and implementation roadmap for Issue #469's 8 acceptance criteria (AC1-AC8), which address UX polish, logging consistency, runtime guardrails, and documentation improvements for the QK256 (GGML I2_S with 256-element blocks) MVP release.

**Key Deliverables:**
- Strict loader mode CLI flag with tolerance enforcement
- Centralized QK256 tolerance constants with diagnostic logging
- K/V cache dimension guardrails with once-per-layer warnings
- Cross-validation receipt generation with timeout consistency
- Tokenizer vocab size parity exposure
- FFI build hygiene consolidation with -isystem usage
- CI parity smoke test with strict mode enforcement
- Documentation quick-start additions for QK256 usage

**Implementation Complexity:** Medium
**Estimated Effort:** 5-7 developer-days (sequential implementation)
**Risk Level:** Low (polish work, no breaking architectural changes)

---

## Context and Background

### QK256 (GGML I2_S) Architecture

BitNet-rs now supports dual-flavor I2_S quantization following PR #468:

1. **BitNet32-F16** (existing): 32-element blocks, inline F16 scales (10 bytes/block)
2. **QK256** (new): 256-element blocks, separate scale tensors (64 bytes/block)

The QK256 implementation introduced:
- Pure-Rust quantization kernels with automatic flavor detection
- Enhanced GGUF loader with minimal fallback for compatibility
- Cross-validation against C++ reference implementation
- Feature-gated builds (`cpu`/`gpu`) with no default features

### Current State Assessment

**Strengths (Post-PR #468):**
- QK256 loader working with U8 tensor storage and derived keys
- Automatic flavor detection based on tensor size
- FFI bridge available for cross-validation when `BITNET_CPP_DIR` set
- Deterministic inference with strict mode flag support

**Gaps Requiring Polish:**
- No CLI flag for strict loader mode (tolerance enforcement is internal)
- QK256 tolerance logic scattered across loader code
- K/V cache lacks dimension validation guardrails
- Parity harness doesn't generate consistent receipts
- Tokenizer doesn't expose real vocab size for parity assertions
- FFI build scripts duplicated across crates with verbose warnings
- CI doesn't test both I2_S flavors with strict mode
- Documentation lacks QK256 quick-start guidance

---

## Technical Requirements Analysis

### AC1: Loader Strict Mode UX

**Requirement:**
CLI flag `--strict-loader` to enforce strict QK256 size validation (reject tensors with >0.1% deviation).

**Current State:**
- `gguf_simple.rs:783-834` has QK256 detection with tolerance check
- No CLI flag to toggle strict vs permissive mode
- Error messages exist but could be more actionable

**Technical Approach:**

1. **CLI Flag Addition** (`bitnet-cli/src/main.rs`)
   ```rust
   #[derive(Parser)]
   struct RunArgs {
       // ... existing flags

       /// Enable strict GGUF loader mode (reject QK256 tensors with >0.1% size deviation)
       #[arg(long = "strict-loader", default_value_t = false)]
       strict_loader: bool,
   }
   ```

2. **Loader Configuration** (`bitnet-models/src/gguf_simple.rs`)
   ```rust
   pub struct GGUFLoaderConfig {
       pub strict_mode: bool,  // New field
       pub tolerance_bytes: usize,  // Centralized tolerance
   }

   impl Default for GGUFLoaderConfig {
       fn default() -> Self {
           Self {
               strict_mode: false,  // Permissive by default (backward compat)
               tolerance_bytes: 128,  // 0.1% for typical QK256 tensors
           }
       }
   }
   ```

3. **Validation Logic Update** (`gguf_simple.rs:783-834`)
   ```rust
   let tolerance = if config.strict_mode { 0 } else { config.tolerance_bytes };

   if available.abs_diff(ggml_need) > tolerance {
       if config.strict_mode {
           return Err(anyhow::anyhow!(
               "Tensor '{}' size mismatch (strict mode): expected {} bytes (256-elem blocks), \
                got {} bytes ({:+.2}% deviation). Use --strict-loader to enforce exact alignment \
                or regenerate GGUF with clean export.",
               info.name, ggml_need, available,
               ((available as f64 - ggml_need as f64) / ggml_need as f64) * 100.0
           ));
       } else {
           log::warn!(
               "Tensor '{}' size mismatch: expected {} bytes (256-elem blocks), \
                got {} bytes ({:+.2}% deviation). Proceeding with tolerance (use --strict-loader \
                to enforce exact alignment).",
               info.name, ggml_need, available,
               ((available as f64 - ggml_need as f64) / ggml_need as f64) * 100.0
           );
       }
   }
   ```

**Affected Components:**
- `bitnet-cli/src/main.rs`: CLI argument parsing
- `bitnet-models/src/gguf_simple.rs`: Loader configuration and validation
- `bitnet-models/src/lib.rs`: Config struct export

**Testing Strategy:**
```rust
// AC1: Strict loader mode rejects misaligned QK256 tensors
#[test]
fn test_strict_loader_rejects_misaligned_qk256() {
    let config = GGUFLoaderConfig { strict_mode: true, ..Default::default() };
    let loader = GGUFLoader::new(config);
    let result = loader.load("tests/fixtures/misaligned-qk256.gguf");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("size mismatch"));
    assert!(result.unwrap_err().to_string().contains("strict mode"));
}

#[test]
fn test_permissive_loader_allows_small_deviation() {
    let config = GGUFLoaderConfig { strict_mode: false, ..Default::default() };
    let loader = GGUFLoader::new(config);
    let result = loader.load("tests/fixtures/slightly-misaligned-qk256.gguf");
    assert!(result.is_ok());  // Succeeds with warning
}
```

**Risk Assessment:**
- **Low Risk**: CLI flag is additive, default behavior unchanged
- **Compatibility**: Backward compatible (permissive by default)
- **Performance**: Zero overhead (one-time check at model load)

---

### AC2: QK256 Tolerance & Logs Centralization

**Requirement:**
Single centralized constant `QK256_SIZE_TOLERANCE` (0.001 = 0.1%) with consistent logging.

**Current State:**
- Tolerance logic in `gguf_simple.rs:783-834` uses hardcoded 128 bytes
- No centralized constant exported from quantization crate
- Logging uses `log::warn!` but messages not standardized

**Technical Approach:**

1. **Constant Definition** (`bitnet-quantization/src/lib.rs`)
   ```rust
   /// QK256 size tolerance for GGUF loader validation
   ///
   /// Allows up to 0.1% deviation from expected tensor size to account for
   /// alignment padding in GGUF exporters. Tensors exceeding this threshold
   /// are likely corrupted or use a different quantization format.
   ///
   /// Rationale:
   /// - 0.1% tolerance = ~128 bytes for a 128KB tensor
   /// - Accounts for metadata padding and alignment requirements
   /// - Rejects grossly misaligned tensors (e.g., wrong block size)
   pub const QK256_SIZE_TOLERANCE_PERCENT: f64 = 0.001;  // 0.1%

   /// Calculate tolerance bytes for QK256 tensor
   pub fn qk256_tolerance_bytes(expected_bytes: usize) -> usize {
       (expected_bytes as f64 * QK256_SIZE_TOLERANCE_PERCENT).ceil() as usize
   }
   ```

2. **Re-export in Models Crate** (`bitnet-models/src/lib.rs`)
   ```rust
   pub use bitnet_quantization::{QK256_SIZE_TOLERANCE_PERCENT, qk256_tolerance_bytes};
   ```

3. **Loader Integration** (`bitnet-models/src/gguf_simple.rs`)
   ```rust
   use crate::{QK256_SIZE_TOLERANCE_PERCENT, qk256_tolerance_bytes};

   let tolerance = if config.strict_mode {
       0
   } else {
       qk256_tolerance_bytes(ggml_need)
   };

   if available.abs_diff(ggml_need) > tolerance {
       let deviation_pct = ((available as f64 - ggml_need as f64) / ggml_need as f64) * 100.0;

       if config.strict_mode {
           log::error!(
               "QK256 size mismatch (strict): tensor='{}', expected={}B, actual={}B, \
                deviation={:+.2}% (threshold=0.00%), REJECTED",
               info.name, ggml_need, available, deviation_pct
           );
           return Err(anyhow::anyhow!("QK256 tensor size validation failed"));
       } else {
           log::warn!(
               "QK256 size mismatch (permissive): tensor='{}', expected={}B, actual={}B, \
                deviation={:+.2}% (threshold={:.2}%), ACCEPTED with tolerance",
               info.name, ggml_need, available, deviation_pct,
               QK256_SIZE_TOLERANCE_PERCENT * 100.0
           );
       }
   }
   ```

4. **Documentation Update** (`docs/reference/quantization-support.md`)
   ```markdown
   ### QK256 Tolerance Policy

   **Constant:** `QK256_SIZE_TOLERANCE_PERCENT = 0.001` (0.1%)

   **Rationale:**
   - Accounts for GGUF metadata padding and alignment requirements
   - Rejects tensors with structural issues (wrong block size, corrupted data)
   - Typical padding: 0-128 bytes for tensors in 128KB-10MB range

   **Behavior:**
   - **Permissive mode (default):** Accepts tensors within 0.1% deviation, emits warning
   - **Strict mode (`--strict-loader`):** Rejects any deviation, emits error

   **Example:**
   ```
   Tensor: 2048×2048 QK256 weight
   Expected: 1,048,576 bytes (8 blocks × 512 bytes/row × 2048 rows)
   Tolerance: 1,048 bytes (0.1%)
   Actual range: 1,047,528 - 1,049,624 bytes (ACCEPTED in permissive mode)
   ```
   ```

**Affected Components:**
- `bitnet-quantization/src/lib.rs`: Constant definition and helper functions
- `bitnet-models/src/lib.rs`: Re-export constants
- `bitnet-models/src/gguf_simple.rs`: Tolerance application and logging
- `docs/reference/quantization-support.md`: Documentation

**Testing Strategy:**
```rust
// AC2: QK256 tolerance constant usage
#[test]
fn test_qk256_tolerance_calculation() {
    use bitnet_quantization::{QK256_SIZE_TOLERANCE_PERCENT, qk256_tolerance_bytes};

    // 1MB tensor → 1KB tolerance
    assert_eq!(qk256_tolerance_bytes(1_000_000), 1_000);

    // 128KB tensor → 128 bytes tolerance
    assert_eq!(qk256_tolerance_bytes(131_072), 131);

    // Verify percentage
    assert_eq!(QK256_SIZE_TOLERANCE_PERCENT, 0.001);
}

#[test]
fn test_qk256_tolerance_logging() {
    // Capture logs and verify format
    let logs = capture_logs(|| {
        let loader = GGUFLoader::new(Default::default());
        let _ = loader.load("tests/fixtures/slightly-misaligned-qk256.gguf");
    });

    assert!(logs.contains("QK256 size mismatch (permissive)"));
    assert!(logs.contains("threshold=0.10%"));
    assert!(logs.contains("ACCEPTED with tolerance"));
}
```

**Risk Assessment:**
- **Low Risk**: Refactoring existing logic into centralized constant
- **Compatibility**: Backward compatible (same default tolerance)
- **Performance**: Negligible (constant lookup, one-time calculation)

---

### AC3: K/V Cache Guardrails

**Requirement:**
K/V cache dimension assertions with once-per-layer warnings to avoid log spam.

**Current State:**
- `bitnet-inference/src/layers/attention.rs:84-108` has cache slicing logic
- No dimension validation after slice operations
- No assertions for expected shapes `[batch, n_heads, seq_len, head_dim]`

**Technical Approach:**

1. **Cache Validation Module** (`bitnet-inference/src/layers/kv_cache_validation.rs` - new file)
   ```rust
   use std::sync::Once;
   use bitnet_common::Tensor;

   /// Once-per-layer warning guard to prevent log spam
   static mut WARNING_FLAGS: [Once; 64] = [Once::new(); 64];  // Max 64 layers

   /// Validate K/V cache dimensions post-slice
   ///
   /// # Expected Shape
   /// `[batch, n_heads, seq_len, head_dim]`
   ///
   /// # Guardrails
   /// - Batch dimension == 1 (no batching support yet)
   /// - Number of heads matches model config
   /// - Sequence length ≤ max context length
   /// - Head dimension matches `d_head = d_model / n_heads`
   pub fn validate_kv_cache_dims(
       tensor: &Tensor,
       layer_idx: usize,
       expected_batch: usize,
       expected_n_heads: usize,
       max_seq_len: usize,
       expected_head_dim: usize,
   ) -> anyhow::Result<()> {
       let shape = tensor.shape();

       // Hot-path debug assertion (zero overhead in release)
       debug_assert_eq!(shape.len(), 4, "K/V cache must be 4D tensor");

       if shape.len() != 4 {
           return Err(anyhow::anyhow!(
               "K/V cache shape error (layer {}): expected 4D, got {}D",
               layer_idx, shape.len()
           ));
       }

       let [batch, n_heads, seq_len, head_dim] = [shape[0], shape[1], shape[2], shape[3]];

       // Validate batch dimension
       if batch != expected_batch {
           emit_once_per_layer_warning(
               layer_idx,
               format!(
                   "Layer {} K/V cache batch mismatch: expected {}, got {}. \
                    Batching not supported yet.",
                   layer_idx, expected_batch, batch
               ),
           );
           anyhow::ensure!(
               batch == expected_batch,
               "K/V cache batch dimension mismatch"
           );
       }

       // Validate number of heads
       if n_heads != expected_n_heads {
           emit_once_per_layer_warning(
               layer_idx,
               format!(
                   "Layer {} K/V cache heads mismatch: expected {} (model config), got {}. \
                    This indicates a cache management bug.",
                   layer_idx, expected_n_heads, n_heads
               ),
           );
           anyhow::ensure!(
               n_heads == expected_n_heads,
               "K/V cache heads dimension mismatch"
           );
       }

       // Validate sequence length
       if seq_len > max_seq_len {
           emit_once_per_layer_warning(
               layer_idx,
               format!(
                   "Layer {} K/V cache seq_len exceeds max: {} > {} (max context). \
                    This indicates a cache overflow.",
                   layer_idx, seq_len, max_seq_len
               ),
           );
           anyhow::ensure!(
               seq_len <= max_seq_len,
               "K/V cache sequence length exceeds max"
           );
       }

       // Validate head dimension
       if head_dim != expected_head_dim {
           emit_once_per_layer_warning(
               layer_idx,
               format!(
                   "Layer {} K/V cache head_dim mismatch: expected {}, got {}. \
                    This indicates incorrect d_model/n_heads calculation.",
                   layer_idx, expected_head_dim, head_dim
               ),
           );
           anyhow::ensure!(
               head_dim == expected_head_dim,
               "K/V cache head dimension mismatch"
           );
       }

       Ok(())
   }

   /// Emit warning only once per layer to avoid log spam
   fn emit_once_per_layer_warning(layer_idx: usize, message: String) {
       unsafe {
           if layer_idx < WARNING_FLAGS.len() {
               WARNING_FLAGS[layer_idx].call_once(|| {
                   log::warn!("{}", message);
               });
           } else {
               log::error!(
                   "Layer index {} exceeds max warning flags ({}), \
                    cannot emit once-per-layer warning",
                   layer_idx, WARNING_FLAGS.len()
               );
           }
       }
   }
   ```

2. **Attention Layer Integration** (`bitnet-inference/src/layers/attention.rs`)
   ```rust
   use crate::layers::kv_cache_validation::validate_kv_cache_dims;

   impl KVCache {
       pub fn get(&self, layer_idx: usize) -> Result<(BitNetTensor, BitNetTensor)> {
           self.validate_layer_index(layer_idx)?;

           // Return sliced view for current sequence length
           let k_cache = self.get_sliced_cache(&self.k_cache[layer_idx])?;
           let v_cache = self.get_sliced_cache(&self.v_cache[layer_idx])?;

           // AC3: Validate dimensions post-slice
           validate_kv_cache_dims(
               &k_cache.to_candle()?,
               layer_idx,
               1,  // expected_batch (no batching yet)
               self.num_heads,
               self.max_seq_len,
               self.head_dim,
           )?;

           validate_kv_cache_dims(
               &v_cache.to_candle()?,
               layer_idx,
               1,  // expected_batch
               self.num_heads,
               self.max_seq_len,
               self.head_dim,
           )?;

           Ok((k_cache, v_cache))
       }
   }
   ```

3. **Cache Initialization Guardrails** (`bitnet-inference/src/layers/attention.rs`)
   ```rust
   impl KVCache {
       pub fn new(
           max_seq_len: usize,
           num_layers: usize,
           num_heads: usize,
           head_dim: usize,
           device: &Device,
       ) -> Result<Self> {
           // AC3: Explicit dimension validation at initialization
           anyhow::ensure!(
               num_layers > 0,
               "K/V cache requires at least 1 layer, got {}",
               num_layers
           );
           anyhow::ensure!(
               num_heads > 0,
               "K/V cache requires at least 1 head, got {}",
               num_heads
           );
           anyhow::ensure!(
               head_dim > 0 && head_dim % 4 == 0,
               "K/V cache head_dim must be positive and divisible by 4, got {}",
               head_dim
           );
           anyhow::ensure!(
               max_seq_len > 0,
               "K/V cache max_seq_len must be positive, got {}",
               max_seq_len
           );

           // ... existing initialization code
       }
   }
   ```

**Affected Components:**
- `bitnet-inference/src/layers/kv_cache_validation.rs`: New validation module
- `bitnet-inference/src/layers/attention.rs`: Integration with cache operations
- `bitnet-inference/src/layers/mod.rs`: Module export

**Testing Strategy:**
```rust
// AC3: K/V cache dimension guardrails
#[test]
fn test_kv_cache_dimension_validation() {
    use bitnet_inference::layers::kv_cache_validation::validate_kv_cache_dims;

    // Valid cache tensor
    let valid_cache = Tensor::zeros(&[1, 16, 128, 64], DType::F32, &Device::Cpu).unwrap();
    let result = validate_kv_cache_dims(&valid_cache, 0, 1, 16, 2048, 64);
    assert!(result.is_ok());

    // Invalid batch dimension
    let invalid_batch = Tensor::zeros(&[2, 16, 128, 64], DType::F32, &Device::Cpu).unwrap();
    let result = validate_kv_cache_dims(&invalid_batch, 0, 1, 16, 2048, 64);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("batch dimension mismatch"));

    // Invalid head dimension
    let invalid_heads = Tensor::zeros(&[1, 8, 128, 64], DType::F32, &Device::Cpu).unwrap();
    let result = validate_kv_cache_dims(&invalid_heads, 0, 1, 16, 2048, 64);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("heads dimension mismatch"));
}

#[test]
fn test_once_per_layer_warning() {
    // Capture logs and verify only one warning per layer
    let logs = capture_logs(|| {
        for _ in 0..5 {
            let invalid = Tensor::zeros(&[2, 16, 128, 64], DType::F32, &Device::Cpu).unwrap();
            let _ = validate_kv_cache_dims(&invalid, 0, 1, 16, 2048, 64);
        }
    });

    // Should only have 1 warning (not 5)
    let warning_count = logs.matches("batch mismatch").count();
    assert_eq!(warning_count, 1, "Expected exactly 1 warning for layer 0");
}

#[test]
#[should_panic(expected = "batch dimension mismatch")]
#[cfg(debug_assertions)]
fn test_debug_assertion_on_invalid_cache() {
    let invalid_batch = Tensor::zeros(&[2, 16, 128, 64], DType::F32, &Device::Cpu).unwrap();
    let _ = validate_kv_cache_dims(&invalid_batch, 0, 1, 16, 2048, 64).unwrap();
}
```

**Risk Assessment:**
- **Low Risk**: Defensive validation, no behavior changes in correct code paths
- **Performance**: `debug_assert!` has zero overhead in release builds
- **Once-per-layer warnings**: Amortized cost (static array lookup)

---

### AC4: Parity Harness Receipts & Timeout Consistency

**Requirement:**
Cross-validation receipt generation with v1.0.0 schema and 60s timeout alignment.

**Current State:**
- `bitnet-inference/src/receipts.rs` has receipt infrastructure
- Parity harness in `crossval/` doesn't generate consistent receipts
- Timeout handling differs between main inference and parity tests

**Technical Approach:**

1. **Receipt Schema Validation** (`bitnet-inference/src/receipts.rs`)
   ```rust
   /// Inference receipt v1.0.0 with parity extension
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct InferenceReceipt {
       /// Schema version (must be "1.0.0")
       pub receipt_version: String,

       /// Compute path (must be "real" for production)
       pub compute_path: String,

       /// Backend (cpu, cuda)
       pub backend: String,

       /// Kernel invocations
       pub kernel_ids: Vec<String>,

       /// Parity validation metadata (optional, for cross-validation)
       #[serde(skip_serializing_if = "Option::is_none")]
       pub parity: Option<ParityMetadata>,

       // ... other fields
   }

   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct ParityMetadata {
       /// C++ reference available
       pub cpp_available: bool,

       /// Cosine similarity (1.0 = perfect match)
       pub cosine_similarity: f64,

       /// Exact token match rate (1.0 = all tokens match)
       pub exact_match_rate: f64,

       /// Parity status
       pub status: String,  // "ok" | "warn" | "error"
   }

   impl InferenceReceipt {
       /// Validate receipt schema v1.0.0
       pub fn validate_schema(&self) -> anyhow::Result<()> {
           // AC4: Receipt schema validation
           anyhow::ensure!(
               self.receipt_version == "1.0.0",
               "Invalid receipt version: expected '1.0.0', got '{}'",
               self.receipt_version
           );

           anyhow::ensure!(
               self.compute_path == "real",
               "Invalid compute_path: expected 'real' (no mock inference), got '{}'",
               self.compute_path
           );

           anyhow::ensure!(
               !self.kernel_ids.is_empty(),
               "Receipt must contain at least one kernel invocation"
           );

           // Kernel ID hygiene checks
           for kid in &self.kernel_ids {
               anyhow::ensure!(
                   !kid.is_empty(),
                   "Kernel ID cannot be empty string"
               );
               anyhow::ensure!(
                   kid.len() <= 128,
                   "Kernel ID '{}' exceeds max length (128 chars)",
                   kid
               );
           }

           anyhow::ensure!(
               self.kernel_ids.len() <= 10_000,
               "Receipt contains too many kernel invocations ({}), max 10K",
               self.kernel_ids.len()
           );

           Ok(())
       }
   }
   ```

2. **Parity Harness Receipt Generation** (`crossval/src/parity_harness.rs` - new file)
   ```rust
   use bitnet_inference::receipts::{InferenceReceipt, ParityMetadata};
   use std::time::Duration;
   use tokio::time::timeout;

   /// Parity validation with receipt generation
   pub async fn run_parity_test(
       model_path: &str,
       tokens: &[u32],
       timeout_secs: u64,
   ) -> anyhow::Result<InferenceReceipt> {
       // AC4: Use same timeout as main inference (default: 60s)
       let timeout_duration = Duration::from_secs(timeout_secs);

       // Run Rust inference with timeout
       let rust_result = timeout(
           timeout_duration,
           run_rust_inference(model_path, tokens),
       )
       .await
       .map_err(|_| anyhow::anyhow!("Rust inference timed out after {}s", timeout_secs))??;

       // Run C++ reference if available
       let cpp_result = if let Ok(cpp_dir) = std::env::var("BITNET_CPP_DIR") {
           Some(
               timeout(
                   timeout_duration,
                   run_cpp_inference(&cpp_dir, model_path, tokens),
               )
               .await
               .map_err(|_| anyhow::anyhow!("C++ inference timed out after {}s", timeout_secs))??
           )
       } else {
           None
       };

       // Calculate parity metrics
       let parity = if let Some(cpp_logits) = cpp_result {
           let cosine_sim = calculate_cosine_similarity(&rust_result.logits, &cpp_logits);
           let exact_match = calculate_exact_match_rate(&rust_result.tokens, &cpp_logits);

           ParityMetadata {
               cpp_available: true,
               cosine_similarity: cosine_sim,
               exact_match_rate: exact_match,
               status: if cosine_sim >= 0.99 && exact_match >= 0.95 {
                   "ok".to_string()
               } else if cosine_sim >= 0.95 {
                   "warn".to_string()
               } else {
                   "error".to_string()
               },
           }
       } else {
           ParityMetadata {
               cpp_available: false,
               cosine_similarity: 1.0,  // N/A
               exact_match_rate: 1.0,   // N/A
               status: "rust_only".to_string(),
           }
       };

       // Generate receipt
       let receipt = InferenceReceipt {
           receipt_version: "1.0.0".to_string(),
           compute_path: "real".to_string(),
           backend: rust_result.backend,
           kernel_ids: rust_result.kernel_ids,
           parity: Some(parity),
           // ... other fields
       };

       // AC4: Validate receipt before returning
       receipt.validate_schema()?;

       Ok(receipt)
   }

   /// Default timeout constant (aligned with main inference)
   pub const DEFAULT_PARITY_TIMEOUT_SECS: u64 = 60;
   ```

3. **Timeout Configuration** (`bitnet-inference/src/engine.rs`)
   ```rust
   /// Default inference timeout (shared with parity harness)
   pub const DEFAULT_INFERENCE_TIMEOUT_SECS: u64 = 60;

   /// Re-export for parity harness consistency
   pub use DEFAULT_INFERENCE_TIMEOUT_SECS as PARITY_TIMEOUT_SECS;
   ```

**Affected Components:**
- `bitnet-inference/src/receipts.rs`: Receipt schema validation
- `crossval/src/parity_harness.rs`: New parity test wrapper with receipt generation
- `bitnet-inference/src/engine.rs`: Timeout constant export
- `xtask/src/crossval.rs`: Integration with xtask command

**Testing Strategy:**
```rust
// AC4: Parity harness receipt generation
#[tokio::test]
async fn test_parity_receipt_generation() {
    let model_path = "tests/fixtures/test-model.gguf";
    let tokens = vec![1, 2, 3, 4];

    let receipt = run_parity_test(model_path, &tokens, 60).await.unwrap();

    // Validate schema
    assert_eq!(receipt.receipt_version, "1.0.0");
    assert_eq!(receipt.compute_path, "real");
    assert!(!receipt.kernel_ids.is_empty());

    // Validate parity metadata
    let parity = receipt.parity.unwrap();
    assert!(parity.cosine_similarity >= 0.0 && parity.cosine_similarity <= 1.0);
    assert!(parity.exact_match_rate >= 0.0 && parity.exact_match_rate <= 1.0);
    assert!(["ok", "warn", "error", "rust_only"].contains(&parity.status.as_str()));
}

#[tokio::test]
async fn test_parity_timeout_consistency() {
    use bitnet_inference::engine::DEFAULT_INFERENCE_TIMEOUT_SECS;
    use crossval::parity_harness::DEFAULT_PARITY_TIMEOUT_SECS;

    // AC4: Verify timeout constants match
    assert_eq!(DEFAULT_INFERENCE_TIMEOUT_SECS, DEFAULT_PARITY_TIMEOUT_SECS);
    assert_eq!(DEFAULT_INFERENCE_TIMEOUT_SECS, 60);
}

#[tokio::test]
#[should_panic(expected = "timed out")]
async fn test_parity_timeout_enforcement() {
    // Simulate slow inference
    let slow_model = "tests/fixtures/slow-model.gguf";
    let tokens = vec![1; 1000];  // Large sequence

    // Should timeout after 1 second
    let _ = run_parity_test(slow_model, &tokens, 1).await.unwrap();
}
```

**Risk Assessment:**
- **Low Risk**: Additive receipt generation, no changes to existing inference
- **Timeout Consistency**: Shared constants prevent drift
- **Performance**: Receipt generation adds <1ms overhead

---

### AC5: Tokenizer Parity

**Requirement:**
Expose `real_vocab_size()` method returning actual vocabulary size from tokenizer model.

**Current State:**
- `bitnet-tokenizers/src/lib.rs:84` has `vocab_size()` trait method
- No distinction between real vocab size and GGUF-padded size
- Parity assertions may fail due to alignment padding differences

**Technical Approach:**

1. **Tokenizer Trait Extension** (`bitnet-tokenizers/src/lib.rs`)
   ```rust
   pub trait Tokenizer: Send + Sync {
       fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>>;
       fn decode(&self, tokens: &[u32]) -> Result<String>;

       /// Vocabulary size (may be padded for alignment in GGUF)
       fn vocab_size(&self) -> usize;

       /// Real vocabulary size from tokenizer model (no padding)
       ///
       /// AC5: This is the actual number of tokens in the vocabulary,
       /// not the padded size used in GGUF for alignment (e.g., 32000 vs 32064).
       /// Use this for cross-validation parity assertions.
       fn real_vocab_size(&self) -> usize {
           // Default: assume vocab_size is real (no padding)
           self.vocab_size()
       }

       // ... other methods
   }
   ```

2. **GGUF Tokenizer Implementation** (`bitnet-tokenizers/src/gguf_tokenizer.rs`)
   ```rust
   impl Tokenizer for GgufTokenizer {
       fn vocab_size(&self) -> usize {
           self.padded_vocab_size  // GGUF-aligned size
       }

       fn real_vocab_size(&self) -> usize {
           self.real_vocab_size  // Actual tokenizer vocab size
       }
   }

   impl GgufTokenizer {
       pub fn from_gguf_metadata(metadata: &GgufMetadata) -> Result<Self> {
           let vocab_tokens = metadata.get_array("tokenizer.ggml.tokens")?;
           let real_size = vocab_tokens.len();

           // GGUF may pad vocab size to alignment boundary (e.g., 64)
           let padded_size = metadata
               .get_u32("tokenizer.ggml.vocab_size")
               .unwrap_or(real_size as u32) as usize;

           log::debug!(
               "Tokenizer initialized: real_vocab_size={}, gguf_padded_size={} \
                (padding: {} tokens for alignment)",
               real_size, padded_size, padded_size - real_size
           );

           Ok(Self {
               real_vocab_size: real_size,
               padded_vocab_size: padded_size,
               // ... other fields
           })
       }
   }
   ```

3. **HuggingFace Tokenizer Implementation** (`bitnet-tokenizers/src/hf_tokenizer.rs`)
   ```rust
   impl Tokenizer for HfTokenizer {
       fn vocab_size(&self) -> usize {
           self.inner.get_vocab_size(true)  // With special tokens
       }

       fn real_vocab_size(&self) -> usize {
           self.inner.get_vocab_size(false)  // Without special token padding
       }
   }
   ```

4. **Parity Assertion** (`crossval/src/parity_harness.rs`)
   ```rust
   use bitnet_tokenizers::Tokenizer;

   pub fn validate_tokenizer_parity(
       rust_tokenizer: &dyn Tokenizer,
       cpp_vocab_size: usize,
   ) -> anyhow::Result<()> {
       // AC5: Use real_vocab_size for parity comparison
       let rust_real_size = rust_tokenizer.real_vocab_size();

       anyhow::ensure!(
           rust_real_size == cpp_vocab_size,
           "Tokenizer vocab size mismatch breaks parity: Rust real_vocab_size={}, C++ vocab_size={}",
           rust_real_size, cpp_vocab_size
       );

       log::debug!(
           "Tokenizer parity validated: real_vocab_size={} (Rust padded={})",
           rust_real_size,
           rust_tokenizer.vocab_size()
       );

       Ok(())
   }
   ```

**Affected Components:**
- `bitnet-tokenizers/src/lib.rs`: Trait method addition
- `bitnet-tokenizers/src/gguf_tokenizer.rs`: Implementation with padding detection
- `bitnet-tokenizers/src/hf_tokenizer.rs`: Implementation without padding
- `crossval/src/parity_harness.rs`: Parity assertion update

**Testing Strategy:**
```rust
// AC5: Tokenizer real vocab size
#[test]
fn test_gguf_tokenizer_real_vocab_size() {
    let tokenizer = GgufTokenizer::from_file("tests/fixtures/tokenizer-padded.gguf").unwrap();

    // Real size from tokenizer model
    assert_eq!(tokenizer.real_vocab_size(), 32000);

    // Padded size from GGUF metadata (aligned to 64)
    assert_eq!(tokenizer.vocab_size(), 32064);

    // Padding amount
    assert_eq!(tokenizer.vocab_size() - tokenizer.real_vocab_size(), 64);
}

#[test]
fn test_hf_tokenizer_real_vocab_size() {
    let tokenizer = HfTokenizer::from_file("tests/fixtures/tokenizer.json").unwrap();

    // HF tokenizers don't pad
    assert_eq!(tokenizer.real_vocab_size(), tokenizer.vocab_size());
}

#[test]
fn test_tokenizer_parity_assertion() {
    let rust_tokenizer = GgufTokenizer::from_file("tests/fixtures/tokenizer.gguf").unwrap();
    let cpp_vocab_size = 32000;  // C++ reference

    // Should succeed with real_vocab_size
    let result = validate_tokenizer_parity(&rust_tokenizer, cpp_vocab_size);
    assert!(result.is_ok());

    // Would fail with padded vocab_size
    let padded_result = validate_tokenizer_parity_wrong(&rust_tokenizer, cpp_vocab_size);
    assert!(padded_result.is_err());
}
```

**Risk Assessment:**
- **Low Risk**: New method with backward-compatible default
- **Compatibility**: Existing code uses `vocab_size()`, unchanged behavior
- **Parity Improvement**: Fixes alignment padding mismatches

---

### AC6: FFI Build Hygiene

**Requirement:**
Consolidate `compile_cpp_shim()` function with `-isystem` for third-party includes.

**Current State:**
- Multiple `build.rs` files: `bitnet-kernels/build.rs`, `bitnet-sys/build.rs`, `crossval/build.rs`
- Each uses different C++ compilation flags
- No `-isystem` usage → verbose warnings from CUDA SDK and C++ reference

**Technical Approach:**

1. **Unified FFI Module** (`xtask/src/ffi.rs` - new file)
   ```rust
   use std::path::{Path, PathBuf};

   /// Compile C++ shim with unified hygiene settings
   ///
   /// AC6: Uses -isystem for third-party includes to suppress external warnings
   pub fn compile_cpp_shim(
       shim_path: &Path,
       output_name: &str,
       include_dirs: &[PathBuf],
       system_include_dirs: &[PathBuf],  // Use -isystem for these
   ) -> Result<(), Box<dyn std::error::Error>> {
       let mut builder = cc::Build::new();

       // Standard flags
       builder
           .cpp(true)
           .flag("-std=c++17")
           .flag("-O2")
           .flag("-fPIC");

       // Regular includes (BitNet-rs code - show warnings)
       for dir in include_dirs {
           builder.include(dir);
       }

       // System includes (third-party - suppress warnings)
       for dir in system_include_dirs {
           builder.flag(&format!("-isystem{}", dir.display()));
       }

       // Suppress specific warning categories from external headers
       builder
           .flag("-Wno-unknown-pragmas")  // CUDA pragmas
           .flag("-Wno-deprecated-declarations");  // CUDA deprecated APIs

       // Compile shim
       builder.file(shim_path);
       builder.compile(output_name);

       println!("cargo:warning=Compiled C++ shim: {} (hygiene: -isystem for external headers)", output_name);

       Ok(())
   }

   /// Get CUDA include directories (for -isystem)
   pub fn cuda_system_includes() -> Vec<PathBuf> {
       vec![
           PathBuf::from("/usr/local/cuda/include"),
           PathBuf::from("/usr/local/cuda/targets/x86_64-linux/include"),
           PathBuf::from("/usr/local/cuda/targets/aarch64-linux/include"),
       ]
   }

   /// Get BitNet C++ include directories (for -isystem)
   pub fn bitnet_cpp_system_includes() -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
       let root = std::env::var("BITNET_CPP_DIR")
           .or_else(|_| std::env::var("HOME").map(|h| format!("{}/.cache/bitnet_cpp", h)))?;

       Ok(vec![
           PathBuf::from(&root).join("include"),
           PathBuf::from(&root).join("3rdparty/llama.cpp"),
           PathBuf::from(&root).join("3rdparty/llama.cpp/ggml/include"),
       ])
   }
   ```

2. **Kernels Build Script Migration** (`bitnet-kernels/build.rs`)
   ```rust
   // Add dependency on xtask in build-dependencies
   use xtask::ffi::{compile_cpp_shim, cuda_system_includes};

   fn main() {
       // ... existing GPU/FFI detection

       if ffi_enabled && have_cpp {
           let shim_path = Path::new("csrc/kernels_shim.cc");
           let include_dirs = vec![
               PathBuf::from("csrc/"),
               PathBuf::from("../bitnet-common/include/"),
           ];
           let system_includes = cuda_system_includes();

           compile_cpp_shim(
               shim_path,
               "bitnet_kernels_shim",
               &include_dirs,
               &system_includes,
           ).expect("Failed to compile kernels shim");
       }
   }
   ```

3. **Sys Build Script Migration** (`bitnet-sys/build.rs`)
   ```rust
   use xtask::ffi::{compile_cpp_shim, bitnet_cpp_system_includes};

   fn main() {
       // ... existing FFI detection

       if ffi_enabled {
           let shim_path = Path::new("csrc/bitnet_c_shim.cc");
           let include_dirs = vec![
               PathBuf::from("csrc/"),
           ];
           let system_includes = bitnet_cpp_system_includes()
               .expect("BITNET_CPP_DIR not set");

           compile_cpp_shim(
               shim_path,
               "bitnet_sys_shim",
               &include_dirs,
               &system_includes,
           ).expect("Failed to compile sys shim");
       }
   }
   ```

4. **Crossval Build Script Migration** (`crossval/build.rs`)
   ```rust
   use xtask::ffi::{compile_cpp_shim, bitnet_cpp_system_includes, cuda_system_includes};

   fn main() {
       let mut system_includes = bitnet_cpp_system_includes()
           .unwrap_or_default();
       system_includes.extend(cuda_system_includes());

       compile_cpp_shim(
           Path::new("csrc/crossval_shim.cc"),
           "crossval_shim",
           &[PathBuf::from("csrc/")],
           &system_includes,
       ).expect("Failed to compile crossval shim");
   }
   ```

**Affected Components:**
- `xtask/src/ffi.rs`: New unified FFI compilation module
- `xtask/Cargo.toml`: Add `cc` dependency
- `bitnet-kernels/Cargo.toml`: Add `xtask` build-dependency
- `bitnet-kernels/build.rs`: Migrate to unified function
- `bitnet-sys/build.rs`: Migrate to unified function
- `crossval/build.rs`: Migrate to unified function

**Testing Strategy:**
```bash
# AC6: FFI build hygiene verification
# Test 1: Clean build with FFI
cargo clean
BITNET_CPP_DIR=/path/to/bitnet.cpp cargo build --features ffi 2>&1 | tee build.log

# Verify -isystem usage
grep -E "isystem.*cuda|isystem.*bitnet_cpp" build.log

# Verify warning reduction (should not see CUDA SDK warnings)
! grep -E "warning:.*cuda.*deprecated" build.log

# Test 2: Verify all shims use unified function
for crate in bitnet-kernels bitnet-sys crossval; do
    echo "Checking $crate/build.rs"
    grep -q "compile_cpp_shim" "crates/$crate/build.rs" || echo "FAIL: $crate not using unified function"
done
```

**Risk Assessment:**
- **Medium Risk**: Changes build system, requires testing across platforms
- **Build Dependencies**: Adds `xtask` as build-dependency (circular dependency check needed)
- **Warning Reduction**: Expected >80% reduction in FFI build warnings

---

### AC7: CI/Parity Smoke Test

**Requirement:**
CI parity validation with `BITNET_DISABLE_MINIMAL_LOADER=1` for both I2_S flavors.

**Current State:**
- `scripts/parity_smoke.sh` exists with basic parity testing
- No enforcement of strict mode or minimal loader bypass in CI
- No validation of both BitNet32-F16 and QK256 formats

**Technical Approach:**

1. **CI Workflow Update** (`.github/workflows/parity.yml`)
   ```yaml
   name: Parity Validation

   on:
     push:
       branches: [main]
     pull_request:
       branches: [main]

   env:
     BITNET_DISABLE_MINIMAL_LOADER: 1  # AC7: Fail-fast on loader issues
     BITNET_DETERMINISTIC: 1
     BITNET_SEED: 42
     RAYON_NUM_THREADS: 1

   jobs:
     parity-bitnet32:
       name: Parity - BitNet32-F16 Format
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3

         - name: Download BitNet32-F16 model
           run: |
             cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-bitnet32

         - name: Run parity smoke test
           run: |
             ./scripts/parity_smoke.sh models/bitnet32-f16.gguf

         - name: Verify receipt
           run: |
             RECEIPT=$(find docs/baselines -name "parity-bitnetcpp.json" | head -n1)
             jq -e '.parity.status == "ok" or .parity.status == "rust_only"' "$RECEIPT"
             jq -e '.parity.cosine_similarity >= 0.99' "$RECEIPT" || true

     parity-qk256:
       name: Parity - QK256 Format
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3

         - name: Download QK256 model
           run: |
             cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf

         - name: Run parity smoke test (strict mode)
           run: |
             # AC7: Test strict loader mode with QK256
             export BITNET_STRICT_MODE=1
             ./scripts/parity_smoke.sh models/ggml-model-i2_s.gguf

         - name: Verify receipt (QK256)
           run: |
             RECEIPT=$(find docs/baselines -name "parity-bitnetcpp.json" | head -n1)
             jq -e '.parity.status == "ok" or .parity.status == "rust_only"' "$RECEIPT"
             jq -e '.quant.i2s_flavor_detected == "GgmlQk256NoScale"' "$RECEIPT"
             jq -e '.parity.cosine_similarity >= 0.99' "$RECEIPT" || true

     parity-summary:
       name: Parity Summary
       needs: [parity-bitnet32, parity-qk256]
       runs-on: ubuntu-latest
       steps:
         - name: Report results
           run: |
             echo "✅ Parity validated for both I2_S flavors:"
             echo "  - BitNet32-F16 (32-elem blocks, inline scales)"
             echo "  - QK256 (256-elem blocks, separate scales)"
   ```

2. **Parity Smoke Script Enhancement** (`scripts/parity_smoke.sh`)
   ```bash
   # ... existing setup

   # AC7: Enforce enhanced loader (no minimal fallback)
   export BITNET_DISABLE_MINIMAL_LOADER=1

   # AC7: Optional strict mode (set by caller)
   if [ -n "$BITNET_STRICT_MODE" ]; then
       echo -e "${BLUE}Running in STRICT MODE (enforces exact QK256 alignment)${NC}"
   fi

   # ... existing parity test execution

   # AC7: Validate I2_S flavor in receipt
   if [ "$JQ_AVAILABLE" = true ]; then
       FLAVOR=$(jq -r '.quant.i2s_flavor_detected' "$RECEIPT")
       echo "I2_S Flavor: $FLAVOR"

       if [ "$FLAVOR" = "mixed" ]; then
           echo -e "${YELLOW}Warning: Model uses mixed I2_S flavors${NC}"
       fi
   fi
   ```

3. **Xtask Crossval Command** (`xtask/src/crossval.rs`)
   ```rust
   use std::process::Command;

   pub fn run_crossval_smoke() -> anyhow::Result<()> {
       // AC7: Run parity smoke test with strict enforcement
       std::env::set_var("BITNET_DISABLE_MINIMAL_LOADER", "1");
       std::env::set_var("BITNET_DETERMINISTIC", "1");
       std::env::set_var("BITNET_SEED", "42");
       std::env::set_var("RAYON_NUM_THREADS", "1");

       let model_path = std::env::var("BITNET_GGUF")
           .or_else(|_| {
               // Auto-discover model in models/
               let models_dir = std::path::Path::new("models");
               std::fs::read_dir(models_dir)?
                   .filter_map(|e| e.ok())
                   .filter(|e| e.path().extension() == Some("gguf".as_ref()))
                   .next()
                   .map(|e| e.path().to_string_lossy().to_string())
                   .ok_or_else(|| anyhow::anyhow!("No GGUF model found in models/"))
           })?;

       println!("Running parity smoke test: {}", model_path);

       let status = Command::new("./scripts/parity_smoke.sh")
           .arg(&model_path)
           .status()?;

       anyhow::ensure!(
           status.success(),
           "Parity smoke test failed (exit code: {})",
           status.code().unwrap_or(-1)
       );

       Ok(())
   }
   ```

**Affected Components:**
- `.github/workflows/parity.yml`: CI workflow definition
- `scripts/parity_smoke.sh`: Script enhancement for strict mode
- `xtask/src/crossval.rs`: Crossval command integration

**Testing Strategy:**
```bash
# AC7: CI parity smoke test validation
# Test 1: BitNet32-F16 format
export BITNET_DISABLE_MINIMAL_LOADER=1
./scripts/parity_smoke.sh models/bitnet32-f16.gguf
# Verify receipt flavor
jq -r '.quant.i2s_flavor_detected' docs/baselines/parity-bitnetcpp.json
# Expected: "BitNet32F16"

# Test 2: QK256 format with strict mode
export BITNET_DISABLE_MINIMAL_LOADER=1
export BITNET_STRICT_MODE=1
./scripts/parity_smoke.sh models/ggml-model-i2_s.gguf
# Verify receipt flavor
jq -r '.quant.i2s_flavor_detected' docs/baselines/parity-bitnetcpp.json
# Expected: "GgmlQk256NoScale"

# Test 3: Verify CI failures on cosine similarity < 0.99
# (manually corrupt model to test CI gate)
```

**Risk Assessment:**
- **Low Risk**: CI enhancement, no production code changes
- **Coverage**: Tests both I2_S flavors independently
- **Strict Mode**: Catches loader regressions early

---

### AC8: Docs & README Quick-Start

**Requirement:**
Update `README.md` and `docs/quickstart.md` with QK256-specific quick-start sections.

**Current State:**
- `README.md` has basic usage examples
- `docs/quickstart.md` exists but lacks QK256 guidance
- `docs/howto/use-qk256-models.md` exists but not linked from main docs

**Technical Approach:**

1. **README.md Update**
   ```markdown
   ## Quick Start

   ### Installation

   ```bash
   # Clone repository
   git clone https://github.com/microsoft/BitNet-rs
   cd BitNet-rs

   # Build with CPU support (includes QK256)
   cargo build --release --no-default-features --features cpu
   ```

   ### Running Inference

   BitNet-rs supports two I2_S quantization formats with automatic detection:

   #### BitNet32-F16 Format (32-element blocks)

   ```bash
   # Download BitNet32-F16 model
   cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-bitnet32

   # Run inference
   cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
     --model models/bitnet32-f16.gguf \
     --prompt "What is machine learning?" \
     --max-tokens 32
   ```

   #### QK256 Format (GGML I2_S, 256-element blocks)

   ```bash
   # Download QK256 model
   cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf

   # Run inference (automatic format detection)
   cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
     --model models/ggml-model-i2_s.gguf \
     --prompt "Explain quantum computing" \
     --max-tokens 64

   # Run with strict loader mode (enforce exact QK256 alignment)
   cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
     --model models/ggml-model-i2_s.gguf \
     --strict-loader \
     --prompt "Test" \
     --max-tokens 16
   ```

   ### Cross-Validation

   ```bash
   # Validate QK256 parity against C++ reference
   export BITNET_CPP_DIR=/path/to/bitnet.cpp
   cargo run -p xtask -- crossval

   # Or use one-command smoke test
   ./scripts/parity_smoke.sh models/ggml-model-i2_s.gguf
   ```

   **Learn More:**
   - [QK256 Usage Guide](docs/howto/use-qk256-models.md) - Comprehensive QK256 documentation
   - [Dual I2_S Flavor Architecture](docs/explanation/i2s-dual-flavor.md) - Technical deep dive
   - [Quick Start Guide](docs/quickstart.md) - 5-minute setup walkthrough
   ```

2. **docs/quickstart.md Update**
   ```markdown
   # BitNet-rs Quick Start

   ## Prerequisites

   - Rust 1.90.0+ (MSRV for Rust 2024 edition)
   - CUDA 11.8+ (optional, for GPU inference)
   - 8GB+ RAM (for 2B parameter models)

   ## Installation

   ```bash
   # Clone repository
   git clone https://github.com/microsoft/BitNet-rs
   cd BitNet-rs

   # Build with CPU support
   cargo build --release --no-default-features --features cpu

   # Or build with GPU support (requires CUDA)
   cargo build --release --no-default-features --features gpu
   ```

   ## Using QK256 Models (GGML I2_S)

   BitNet-rs supports GGML's I2_S quantization format (QK256) with 256-element blocks.

   ### Automatic Format Detection

   The loader automatically detects QK256 format based on tensor size:

   ```bash
   # Download QK256 model
   cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf

   # Automatic detection and inference
   cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
     --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
     --prompt "What is 2+2?" \
     --max-tokens 16
   ```

   **Expected Output:**
   ```
   Loading model: models/ggml-model-i2_s.gguf
   INFO: I2_S 'blk.0.attn_q.weight': GGML/llama.cpp format detected (QK_K=256, 64B/block)
   Generating...
   2+2 equals 4.
   ```

   ### Strict Loader Mode

   Enforce exact QK256 alignment (reject tensors with >0.1% size deviation):

   ```bash
   cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
     --model models/ggml-model-i2_s.gguf \
     --strict-loader \
     --prompt "Test" \
     --max-tokens 16
   ```

   **Use strict mode when:**
   - Validating model exports for production deployment
   - Debugging model loading issues
   - Running CI/CD parity tests

   ### Cross-Validation

   Verify QK256 implementation against C++ reference:

   ```bash
   # Set C++ reference path (optional)
   export BITNET_CPP_DIR=/path/to/bitnet.cpp

   # Run cross-validation
   cargo run -p xtask -- crossval

   # Or use one-command smoke test
   ./scripts/parity_smoke.sh models/ggml-model-i2_s.gguf
   ```

   **Receipt validation:**
   ```bash
   # Check parity metrics
   jq '.parity' docs/baselines/parity-bitnetcpp.json

   # Expected output:
   # {
   #   "cpp_available": true,
   #   "cosine_similarity": 0.9923,
   #   "exact_match_rate": 1.0,
   #   "status": "ok"
   # }
   ```

   ## Next Steps

   - Read [QK256 Usage Guide](howto/use-qk256-models.md) for comprehensive documentation
   - Explore [Dual I2_S Flavor Architecture](explanation/i2s-dual-flavor.md) for technical details
   - Review [Model Validation Guide](howto/validate-models.md) for production workflows
   ```

3. **Cross-Link Updates**
   - Add QK256 links to `docs/README.md` (documentation index)
   - Update `docs/explanation/i2s-dual-flavor.md` to reference quick-start
   - Add "See Also" sections linking to new content

**Affected Components:**
- `README.md`: Main repository README
- `docs/quickstart.md`: Quick start guide
- `docs/README.md`: Documentation index
- `docs/explanation/i2s-dual-flavor.md`: Cross-link updates

**Testing Strategy:**
```bash
# AC8: Documentation validation
# Test 1: Verify all links work
for doc in README.md docs/quickstart.md docs/howto/use-qk256-models.md; do
    echo "Checking links in $doc"
    grep -oP '\[.*?\]\(\K[^)]+' "$doc" | while read link; do
        [ -f "$link" ] || echo "BROKEN: $link (from $doc)"
    done
done

# Test 2: Run commands from quick-start examples
./scripts/validate_quickstart_examples.sh

# Test 3: Verify QK256 examples are reproducible
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/ggml-model-i2_s.gguf \
  --prompt "What is 2+2?" \
  --max-tokens 16 \
  --seed 42
# Verify output matches documented example
```

**Risk Assessment:**
- **Low Risk**: Documentation-only changes
- **User Impact**: Improved onboarding for QK256 users
- **Maintenance**: Requires updating examples when CLI changes

---

## Implementation Order and Dependencies

### Dependency Graph

```
AC6 (FFI Build Hygiene)
  ↓
AC2 (QK256 Tolerance)
  ↓
AC1 (Strict Loader Mode) ← depends on AC2
  ↓
AC3 (K/V Cache Guardrails) ← independent
  ↓
AC5 (Tokenizer Parity) ← independent
  ↓
AC4 (Parity Receipts) ← depends on AC5 for vocab assertions
  ↓
AC7 (CI Parity Smoke) ← depends on AC1, AC2, AC4
  ↓
AC8 (Documentation) ← depends on all AC1-AC7 for accurate examples
```

### Recommended Implementation Order

**Phase 1: Build Infrastructure (Day 1)**
1. **AC6** - FFI Build Hygiene
   - Consolidate `compile_cpp_shim()` in `xtask/src/ffi.rs`
   - Migrate all `build.rs` files to unified function
   - Verify >80% warning reduction
   - **Rationale:** Reduces build noise for subsequent development

**Phase 2: Quantization Core (Days 2-3)**
2. **AC2** - QK256 Tolerance Centralization
   - Define `QK256_SIZE_TOLERANCE_PERCENT` in `bitnet-quantization`
   - Update loader to use centralized constant
   - Add documentation in `docs/reference/quantization-support.md`
   - **Rationale:** Foundation for AC1 strict mode

3. **AC1** - Strict Loader Mode UX
   - Add `--strict-loader` CLI flag
   - Implement loader configuration with tolerance enforcement
   - Wire CLI → loader → validation logic
   - **Rationale:** Core UX improvement for QK256 validation

**Phase 3: Runtime Guardrails (Day 4)**
4. **AC3** - K/V Cache Guardrails
   - Implement `kv_cache_validation.rs` module
   - Add dimension assertions with once-per-layer warnings
   - Integrate with attention layer cache operations
   - **Rationale:** Independent safety improvement

5. **AC5** - Tokenizer Parity
   - Add `real_vocab_size()` trait method
   - Implement in GGUF and HF tokenizers
   - Update parity assertions
   - **Rationale:** Independent parity improvement

**Phase 4: Cross-Validation (Day 5)**
6. **AC4** - Parity Receipts & Timeout
   - Implement receipt validation in `bitnet-inference`
   - Create parity harness wrapper with receipt generation
   - Align timeout constants across inference and parity
   - **Rationale:** Depends on AC5 for tokenizer assertions

**Phase 5: CI & Documentation (Days 6-7)**
7. **AC7** - CI Parity Smoke Test
   - Update `.github/workflows/parity.yml`
   - Enhance `scripts/parity_smoke.sh` for strict mode
   - Add xtask crossval command integration
   - **Rationale:** Depends on AC1, AC2, AC4 for meaningful validation

8. **AC8** - Documentation Quick-Start
   - Update `README.md` with QK256 examples
   - Enhance `docs/quickstart.md` with strict mode usage
   - Add cross-links and validation scripts
   - **Rationale:** Final polish after all features implemented

---

## Risk Assessment

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| FFI build hygiene breaks existing builds | High | Low | Test on clean environments, provide rollback path |
| Strict loader mode breaks existing models | Medium | Low | Default to permissive mode, provide clear error messages |
| K/V cache guardrails false positives | Medium | Low | Use once-per-layer warnings, debug assertions |
| Parity receipt schema drift | Low | Medium | Schema validation tests, version checks |
| Tokenizer vocab size confusion | Low | Low | Clear naming (`real_vocab_size` vs `vocab_size`) |
| CI timeout inconsistency | Low | Low | Shared constants, integration tests |
| Documentation examples stale | Low | Medium | Validation scripts in CI |

### Compatibility Risks

| Component | Risk | Backward Compatibility Strategy |
|-----------|------|--------------------------------|
| Strict Loader Mode | New CLI flag | Default: permissive (unchanged behavior) |
| QK256 Tolerance | Centralized constant | Same default value (128 bytes) |
| K/V Cache Validation | New assertions | `debug_assert!` in hot path, warnings in cold path |
| Parity Receipts | New schema fields | Optional `parity` field, v1.0.0 compatible |
| Tokenizer API | New method | Default implementation (backward compat) |
| FFI Build System | Build script changes | Fallback to existing flags if unified function fails |

### Performance Risks

| Component | Overhead | Acceptable Threshold | Mitigation |
|-----------|----------|---------------------|------------|
| Strict Loader Mode | One-time at load | <1% load time increase | Only checks at model load, not inference |
| K/V Cache Guardrails | Per-layer validation | Zero in release builds | `debug_assert!` for hot path |
| Parity Receipts | Receipt generation | <1ms per inference | Async I/O for receipt writes |
| Tokenizer Parity | Vocab size query | Negligible | Simple field access |

---

## Success Criteria

### Acceptance Criteria Validation

**AC1: Strict Loader Mode UX**
- ✅ `--strict-loader` CLI flag parsed correctly
- ✅ Loader rejects QK256 tensors with >0.1% deviation in strict mode
- ✅ Clear error messages with tensor name, expected size, deviation percentage
- ✅ Default: permissive mode (backward compatible)

**AC2: QK256 Tolerance Centralization**
- ✅ `QK256_SIZE_TOLERANCE_PERCENT = 0.001` defined in `bitnet-quantization`
- ✅ Loader uses centralized constant for all tolerance checks
- ✅ Logging references constant value (`threshold=0.10%`)
- ✅ Documentation in `docs/reference/quantization-support.md`

**AC3: K/V Cache Guardrails**
- ✅ Post-slice dimension assertions for `[batch, n_heads, seq_len, head_dim]`
- ✅ Once-per-layer warnings (no log spam)
- ✅ `debug_assert!` in hot path (zero overhead in release)
- ✅ Explicit `anyhow::ensure!` in cache initialization

**AC4: Parity Receipts & Timeout**
- ✅ Receipt schema v1.0.0 with `parity` field
- ✅ `compute_path: "real"`, `kernel_ids: Vec<String>`
- ✅ Timeout: 60s (shared constant with main inference)
- ✅ Receipt validation tests pass

**AC5: Tokenizer Parity**
- ✅ `real_vocab_size()` method in `Tokenizer` trait
- ✅ GGUF tokenizer distinguishes real vs padded size
- ✅ Parity assertions use `real_vocab_size()` for comparison
- ✅ Debug logging shows both sizes

**AC6: FFI Build Hygiene**
- ✅ `compile_cpp_shim()` in `xtask/src/ffi.rs`
- ✅ `-isystem` for CUDA and BitNet C++ includes
- ✅ All `build.rs` files migrated to unified function
- ✅ >80% reduction in FFI build warnings

**AC7: CI Parity Smoke Test**
- ✅ `BITNET_DISABLE_MINIMAL_LOADER=1` in CI workflow
- ✅ Both BitNet32-F16 and QK256 models tested
- ✅ Cosine similarity ≥ 0.99 gate
- ✅ Exact match rate ≥ 0.95 gate

**AC8: Docs & README**
- ✅ `README.md` includes QK256 quick-start section
- ✅ `docs/quickstart.md` includes "Using QK256 Models" section
- ✅ Both docs reference `docs/explanation/i2s-dual-flavor.md`
- ✅ Strict loader mode usage documented
- ✅ Cross-validation command examples included

### Quality Gates

**Testing Coverage:**
- ✅ Unit tests for each AC (tagged with `// AC:<ID>`)
- ✅ Integration tests for loader, K/V cache, parity harness
- ✅ CI tests for both I2_S flavors
- ✅ Documentation validation scripts

**Performance Benchmarks:**
- ✅ No regression in inference throughput (tokens/sec)
- ✅ K/V cache assertions have zero overhead in release builds
- ✅ Strict loader mode adds <1% load time overhead

**Code Quality:**
- ✅ All code passes `cargo clippy --all-targets --all-features`
- ✅ All code formatted with `cargo fmt --all`
- ✅ FFI build warnings reduced by >80%

---

## Conclusion

This specification provides a comprehensive technical roadmap for implementing Issue #469's 8 acceptance criteria, addressing UX polish, logging consistency, runtime guardrails, and documentation improvements for the QK256 MVP release.

**Key Achievements:**
- **Strict Loader Mode:** User-controlled QK256 tolerance enforcement with actionable error messages
- **Centralized Tolerance:** Single source of truth for QK256 size validation (`0.1%`)
- **K/V Cache Safety:** Dimension guardrails with once-per-layer warnings to prevent log spam
- **Parity Consistency:** Receipt generation with v1.0.0 schema and 60s timeout alignment
- **Tokenizer Parity:** Real vocab size exposure for accurate cross-validation
- **FFI Build Hygiene:** Consolidated compilation with `-isystem` for >80% warning reduction
- **CI Coverage:** Both I2_S flavors validated with strict mode enforcement
- **Documentation:** QK256 quick-start guidance for improved developer onboarding

**Estimated Effort:** 5-7 developer-days (sequential implementation)
**Risk Level:** Low (polish work, no breaking changes)
**Release Target:** v0.1.0-mvp

**Next Steps:**
- Route to **spec-finalizer** for requirements validation
- Begin implementation in dependency order (AC6 → AC2 → AC1 → AC3 → AC5 → AC4 → AC7 → AC8)
- Track progress with TDD test coverage (`// AC:<ID>` tags)
- Validate success criteria before merging to main

---

**Document Control:**
- Review Status: Draft
- Next Review: spec-finalizer validation
- Owner: BitNet-rs Architecture Team
