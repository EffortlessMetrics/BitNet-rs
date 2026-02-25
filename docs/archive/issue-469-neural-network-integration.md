# Issue #469 Neural Network Component Integration

**Document Status:** Integration Guide
**Created:** 2025-10-18
**Issue:** #469
**Targets:** v0.1.0-mvp release

---

## Overview

This document describes how Issue #469's 8 acceptance criteria integrate with the BitNet.rs neural network inference pipeline. Each component aligns with specific pipeline stages (Model Loading → Quantization → Inference → Output) and respects workspace crate boundaries.

---

## Neural Network Inference Pipeline

```
┌─────────────────┐
│ Model Loading   │ ← AC1: Strict Loader Mode
│ (bitnet-models) │ ← AC2: QK256 Tolerance
│                 │ ← AC6: FFI Build Hygiene
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Quantization    │ ← AC2: Tolerance Constants (bitnet-quantization)
│ (I2_S)          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Inference       │ ← AC3: K/V Cache Guardrails
│ (Attention)     │ ← AC4: Receipt Generation
│                 │ ← AC5: Tokenizer Parity (bitnet-tokenizers)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output          │ ← AC4: Parity Metadata
│ (Tokens)        │ ← AC7: CI Validation
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Documentation   │ ← AC8: Quick-Start Guides
└─────────────────┘
```

---

## AC1: Strict Loader Mode Integration

### Pipeline Stage: Model Loading

**Component:** `bitnet-models/src/gguf_simple.rs`

**Integration Flow:**

```
User Command:
  bitnet run --strict-loader --model model.gguf --prompt "Test"

    │
    ▼

CLI Parsing (bitnet-cli):
  args.strict_loader = true

    │
    ▼

Loader Config Construction (bitnet-models):
  config = GGUFLoaderConfig {
    strict_mode: true,
    tolerance_bytes: 0  // Ignored in strict mode
  }

    │
    ▼

GGUF Loading (bitnet-models):
  load_gguf_full(model_path, config)
    ├─▶ Parse GGUF header
    ├─▶ Detect I2_S flavor (BitNet32-F16 or QK256)
    ├─▶ For each QK256 tensor:
    │     └─▶ validate_qk256_tensor_size()
    │           ├─▶ Check: actual_bytes vs expected_bytes
    │           ├─▶ Apply tolerance (0 in strict mode)
    │           └─▶ [STRICT FAIL] anyhow::bail!() with actionable error
    └─▶ [OK] Return ModelWeights

    │
    ▼

Quantization Stage:
  Tensors loaded into bitnet-quantization for I2_S processing
```

**Error Handling:**

```rust
// Strict mode error example
Err(anyhow!(
    "Tensor 'blk.0.attn_q.weight' size mismatch (STRICT MODE): \
     expected 1048576B (256-elem blocks), got 1049000B (+0.04% deviation). \n\
     Hint: Use --strict-loader=false for permissive mode, \
     or regenerate GGUF with clean export."
))
```

**Dependencies:**
- AC2: Uses `qk256_tolerance_bytes()` for default tolerance in permissive mode

---

## AC2: QK256 Tolerance Integration

### Pipeline Stage: Quantization

**Component:** `bitnet-quantization/src/lib.rs`

**Integration Flow:**

```
Quantization Constants (bitnet-quantization):
  pub const QK256_SIZE_TOLERANCE_PERCENT: f64 = 0.001;
  pub fn qk256_tolerance_bytes(expected: usize) -> usize { ... }

    │
    ▼

Re-export (bitnet-models):
  pub use bitnet_quantization::{
    QK256_SIZE_TOLERANCE_PERCENT,
    qk256_tolerance_bytes
  };

    │
    ▼

Loader Default Config (bitnet-models):
  impl Default for GGUFLoaderConfig {
    fn default() -> Self {
      Self {
        strict_mode: false,
        tolerance_bytes: qk256_tolerance_bytes(131_072),  // ← AC2 constant
      }
    }
  }

    │
    ▼

Validation Logic (bitnet-models):
  let tolerance = if config.strict_mode { 0 } else { config.tolerance_bytes };
  if actual.abs_diff(expected) > tolerance {
    log_qk256_size_mismatch(...);  // ← AC2 logging format
  }
```

**Logging Format:**

```
log::warn!(
  "QK256 size mismatch (permissive): tensor='blk.0.attn_q.weight', \
   expected=1048576B, actual=1049000B, deviation=+0.04% (threshold=0.10%), \
   ACCEPTED with tolerance"
);
```

**Cross-Crate Dependencies:**
- `bitnet-quantization` exports constants
- `bitnet-models` imports and uses for loader configuration

---

## AC3: K/V Cache Integration

### Pipeline Stage: Inference (Attention Layer)

**Component:** `bitnet-inference/src/layers/attention.rs`

**Integration Flow:**

```
Attention Forward Pass (bitnet-inference):
  fn forward(&mut self, x: Tensor, layer_idx: usize) -> Result<Tensor> {
    // Get K/V cache for this layer
    let (k_cache, v_cache) = self.kv_cache.get(layer_idx)?;
      │
      ▼
    K/V Cache Retrieval (bitnet-inference):
      impl KVCache {
        pub fn get(&self, layer_idx: usize) -> Result<(BitNetTensor, BitNetTensor)> {
          // Slice cache for current sequence length
          let k_cache = self.get_sliced_cache(&self.k_cache[layer_idx])?;
          let v_cache = self.get_sliced_cache(&self.v_cache[layer_idx])?;

          // ← AC3: Validate dimensions post-slice
          validate_kv_cache_dims(
            &k_cache.to_candle()?,
            layer_idx,
            1,  // batch
            self.num_heads,
            self.max_seq_len,
            self.head_dim,
          )?;

          Ok((k_cache, v_cache))
        }
      }
      │
      ▼
    Validation Module (bitnet-inference/layers/kv_cache_validation.rs):
      pub fn validate_kv_cache_dims(...) -> anyhow::Result<()> {
        let shape = tensor.shape();

        // Hot-path assertion (zero overhead in release)
        debug_assert_eq!(shape.len(), 4, "K/V cache must be 4D");

        // Cold-path validation
        if shape.len() != 4 {
          anyhow::bail!("K/V cache shape error (layer {}): ...", layer_idx);
        }

        let [batch, n_heads, seq_len, head_dim] = [...];

        // Validate each dimension with once-per-layer warnings
        if batch != expected_batch {
          emit_once_per_layer_warning(layer_idx, "batch mismatch...");
          anyhow::ensure!(...);
        }

        // ... similar checks for n_heads, seq_len, head_dim
      }
  }
```

**Once-Per-Layer Warning Mechanism:**

```rust
static mut WARNING_FLAGS: [Once; 64] = [Once::new(); 64];

fn emit_once_per_layer_warning(layer_idx: usize, message: String) {
    unsafe {
        if layer_idx < WARNING_FLAGS.len() {
            WARNING_FLAGS[layer_idx].call_once(|| {
                log::warn!("{}", message);
            });
        }
    }
}
```

**Performance Characteristics:**
- **Hot path:** `debug_assert!` (compiled out in release)
- **Cold path:** Once-per-layer warning (amortized cost)
- **Validation overhead:** Negligible (<0.1% inference time)

---

## AC4: Parity Receipt Integration

### Pipeline Stage: Cross-Validation

**Component:** `crossval/src/parity_harness.rs`

**Integration Flow:**

```
Cross-Validation Command:
  cargo run -p xtask -- crossval

    │
    ▼

Parity Harness (crossval):
  async fn run_parity_test(model_path, tokens, timeout) -> Result<InferenceReceipt> {
    // Run Rust inference
    let rust_result = timeout(60s, run_rust_inference(model_path, tokens)).await?;
      │
      ├─▶ Inference Engine (bitnet-inference):
      │     ├─▶ Load model (bitnet-models)
      │     ├─▶ Quantize weights (bitnet-quantization)
      │     ├─▶ Tokenize input (bitnet-tokenizers)
      │     ├─▶ Forward pass (attention with AC3 validation)
      │     └─▶ Collect kernel_ids, backend, logits
      │
      ▼

    // Run C++ reference (if available)
    let cpp_result = if let Ok(cpp_dir) = env::var("BITNET_CPP_DIR") {
      Some(timeout(60s, run_cpp_inference(cpp_dir, model_path, tokens)).await?)
    } else { None };

    // Calculate parity metrics
    let parity = if let Some(cpp_logits) = cpp_result {
      ParityMetadata {
        cpp_available: true,
        cosine_similarity: calculate_cosine_similarity(&rust_result.logits, &cpp_logits),
        exact_match_rate: calculate_exact_match_rate(&rust_result.tokens, &cpp_logits),
        status: if cosine_sim >= 0.99 && exact_match >= 0.95 { "ok" }
                else if cosine_sim >= 0.95 { "warn" }
                else { "error" },
      }
    } else {
      ParityMetadata { cpp_available: false, status: "rust_only", ... }
    };

    // Generate receipt
    let mut receipt = InferenceReceipt::generate(&rust_result.backend, rust_result.kernel_ids)?;
    receipt.parity = Some(parity);
    receipt.validate_schema_v1()?;  // ← AC4: Schema validation

    Ok(receipt)
  }

    │
    ▼

Receipt Storage (xtask):
  receipt.save("docs/baselines/YYYY-MM-DD/parity-bitnetcpp.json")?;
```

**Receipt Schema v1.0.0:**

```json
{
  "schema_version": "1.0.0",
  "compute_path": "real",
  "backend": "cpu",
  "kernels": ["i2s_gemv", "rope_apply", "attention_real"],
  "parity": {
    "cpp_available": true,
    "cosine_similarity": 0.9923,
    "exact_match_rate": 1.0,
    "status": "ok"
  }
}
```

**Dependencies:**
- AC5: Uses tokenizer `real_vocab_size()` for vocab parity assertions
- Shared timeout: `DEFAULT_PARITY_TIMEOUT_SECS = 60`

---

## AC5: Tokenizer Parity Integration

### Pipeline Stage: Inference (Tokenization)

**Component:** `bitnet-tokenizers/src/lib.rs`

**Integration Flow:**

```
Tokenizer Loading (bitnet-tokenizers):
  pub trait Tokenizer: Send + Sync {
    fn vocab_size(&self) -> usize;          // May include GGUF padding
    fn real_vocab_size(&self) -> usize;     // ← AC5: Real vocab (no padding)
  }

    │
    ▼

GGUF Tokenizer Implementation (bitnet-tokenizers):
  impl GgufTokenizer {
    pub fn from_gguf_metadata(metadata: &GgufMetadata) -> Result<Self> {
      let vocab_tokens = metadata.get_array("tokenizer.ggml.tokens")?;
      let real_size = vocab_tokens.len();  // ← AC5: Real vocab size

      let padded_size = metadata
        .get_u32("tokenizer.ggml.vocab_size")
        .unwrap_or(real_size as u32) as usize;  // ← GGUF-aligned size

      log::debug!(
        "Tokenizer: real_vocab_size={}, gguf_padded_size={}, padding={} tokens",
        real_size, padded_size, padded_size - real_size
      );

      Ok(Self { real_vocab_size: real_size, padded_vocab_size: padded_size, ... })
    }
  }

  impl Tokenizer for GgufTokenizer {
    fn vocab_size(&self) -> usize { self.padded_vocab_size }  // GGUF-aligned
    fn real_vocab_size(&self) -> usize { self.real_vocab_size }  // AC5: Real
  }

    │
    ▼

Parity Assertion (crossval):
  pub fn validate_tokenizer_parity(
    rust_tokenizer: &dyn Tokenizer,
    cpp_vocab_size: usize,
  ) -> Result<()> {
    let rust_real = rust_tokenizer.real_vocab_size();  // ← AC5: Use real size
    let rust_padded = rust_tokenizer.vocab_size();

    anyhow::ensure!(
      rust_real == cpp_vocab_size,
      "Tokenizer vocab mismatch: Rust real={}, Rust padded={}, C++ exact={}, \
       mismatch={} tokens",
      rust_real, rust_padded, cpp_vocab_size,
      (rust_real as i64 - cpp_vocab_size as i64).abs()
    );

    log::debug!(
      "Tokenizer parity OK: real_vocab_size={} (Rust padded={}, C++ exact={})",
      rust_real, rust_padded, cpp_vocab_size
    );

    Ok(())
  }
```

**Cross-Validation Example:**

```
Rust GGUF Tokenizer:
  - real_vocab_size() = 32000
  - vocab_size() = 32064 (GGUF padding: 64 tokens)

C++ Reference Tokenizer:
  - vocab_size = 32000 (no padding)

Parity Check:
  rust_real_vocab_size (32000) == cpp_vocab_size (32000) ✅
```

**Dependencies:**
- AC4: Parity harness uses `validate_tokenizer_parity()` before logit comparison

---

## AC6: FFI Build Hygiene Integration

### Pipeline Stage: Build System

**Component:** `xtask/src/ffi.rs` (build-time)

**Integration Flow:**

```
Cargo Build with FFI:
  cargo build --no-default-features --features cpu,ffi

    │
    ▼

Crate Build Scripts (build.rs):
  ├─▶ bitnet-kernels/build.rs
  │     use xtask::ffi::{compile_cpp_shim, cuda_system_includes};
  │     compile_cpp_shim(
  │       "csrc/kernels_shim.cc",
  │       "bitnet_kernels_shim",
  │       &[PathBuf::from("csrc/")],  // -I (show warnings)
  │       &cuda_system_includes(),     // -isystem (suppress warnings)
  │     )?;
  │
  ├─▶ bitnet-sys/build.rs
  │     use xtask::ffi::{compile_cpp_shim, bitnet_cpp_system_includes};
  │     compile_cpp_shim(
  │       "csrc/bitnet_c_shim.cc",
  │       "bitnet_sys_shim",
  │       &[PathBuf::from("csrc/")],
  │       &bitnet_cpp_system_includes()?,
  │     )?;
  │
  └─▶ crossval/build.rs
        use xtask::ffi::{compile_cpp_shim, bitnet_cpp_system_includes, cuda_system_includes};
        let mut system_includes = bitnet_cpp_system_includes()?;
        system_includes.extend(cuda_system_includes());
        compile_cpp_shim(
          "csrc/crossval_shim.cc",
          "crossval_shim",
          &[PathBuf::from("csrc/")],
          &system_includes,
        )?;

    │
    ▼

Unified FFI Compilation (xtask/src/ffi.rs):
  pub fn compile_cpp_shim(...) -> Result<()> {
    let mut builder = cc::Build::new();

    builder
      .cpp(true)
      .flag("-std=c++17")
      .flag("-O2")
      .flag("-fPIC");

    // Regular includes (BitNet.rs code - show warnings)
    for dir in include_dirs {
      builder.include(dir);  // -I{dir}
    }

    // System includes (third-party - suppress warnings)
    for dir in system_include_dirs {
      builder.flag(&format!("-isystem{}", dir.display()));  // ← AC6: -isystem
    }

    // Suppress specific warning categories
    builder
      .flag("-Wno-unknown-pragmas")  // CUDA pragmas
      .flag("-Wno-deprecated-declarations");  // CUDA deprecated APIs

    builder.file(shim_path);
    builder.compile(output_name);

    Ok(())
  }
```

**Warning Reduction:**

| Before AC6 | After AC6 |
|-----------|----------|
| 1200+ FFI build warnings | <200 warnings (>80% reduction) |
| CUDA SDK deprecation warnings | Suppressed via -isystem |
| llama.cpp pragma warnings | Suppressed via -isystem |
| BitNet.rs code warnings | Still shown (desired) |

**Dependencies:**
- `xtask` added as build-dependency in affected crates
- `cc` crate remains primary build dependency

---

## AC7: CI Parity Integration

### Pipeline Stage: CI/CD Validation

**Component:** `.github/workflows/parity.yml`

**Integration Flow:**

```
GitHub Actions Trigger:
  on: [push, pull_request]

    │
    ▼

CI Environment Setup:
  env:
    BITNET_DISABLE_MINIMAL_LOADER: 1  # ← AC7: Enforce enhanced loader
    BITNET_DETERMINISTIC: 1
    BITNET_SEED: 42
    RAYON_NUM_THREADS: 1

    │
    ▼

Job: parity-bitnet32 (BitNet32-F16 format):
  steps:
    - Download BitNet32-F16 model
    - Run parity smoke test: ./scripts/parity_smoke.sh model.gguf
        │
        ├─▶ Rust inference (bitnet-inference)
        ├─▶ C++ inference (if BITNET_CPP_DIR set)
        ├─▶ Generate receipt (AC4)
        └─▶ Save to docs/baselines/parity-bitnetcpp.json
    - Verify receipt:
        jq -e '.parity.status == "ok" or .parity.status == "rust_only"'
        jq -e '.parity.cosine_similarity >= 0.99 or .parity.cpp_available == false'

    │
    ▼

Job: parity-qk256 (QK256 format):
  env:
    BITNET_STRICT_MODE: 1  # ← AC7: Strict loader for QK256
  steps:
    - Download QK256 model
    - Run parity smoke test (strict): ./scripts/parity_smoke.sh model.gguf
        │
        ├─▶ Loader validates QK256 tensors (AC1: strict mode)
        ├─▶ Rust inference (bitnet-inference)
        ├─▶ C++ inference (if BITNET_CPP_DIR set)
        ├─▶ Generate receipt (AC4)
        └─▶ Save to docs/baselines/parity-bitnetcpp.json
    - Verify receipt:
        jq -e '.parity.status == "ok" or .parity.status == "rust_only"'
        jq -e '.quant.i2s_flavor_detected == "GgmlQk256NoScale"'  # ← AC7: Flavor check
        jq -e '.parity.cosine_similarity >= 0.99 or .parity.cpp_available == false'

    │
    ▼

Job: parity-summary:
  needs: [parity-bitnet32, parity-qk256]
  steps:
    - Report results: "✅ Parity validated for both I2_S flavors"
```

**Receipt Location Validation:**

```bash
# AC7: Verify receipt is at workspace root (not subdirectory)
RECEIPT=$(find docs/baselines -name "parity-bitnetcpp.json" | head -n1)
if [[ ! "$RECEIPT" =~ ^docs/baselines/[0-9]{4}-[0-9]{2}-[0-9]{2}/parity-bitnetcpp.json$ ]]; then
  echo "ERROR: Receipt misplaced (expected: docs/baselines/YYYY-MM-DD/parity-bitnetcpp.json)"
  exit 1
fi
```

**Dependencies:**
- AC1: Strict loader mode tested in parity-qk256 job
- AC2: QK256 tolerance applied in loader
- AC4: Receipt generation and validation
- AC5: Tokenizer parity assertions in parity harness

---

## AC8: Documentation Integration

### Pipeline Stage: User Onboarding

**Component:** `README.md`, `docs/quickstart.md`

**Integration Flow:**

```
User Discovery:
  GitHub README.md → Quick Start section
    │
    ├─▶ Installation instructions (cargo build)
    │
    ├─▶ BitNet32-F16 example:
    │     cargo run -p bitnet-cli --features cpu,full-cli -- run \
    │       --model model.gguf --prompt "Test" --max-tokens 32
    │
    ├─▶ QK256 example (automatic detection):
    │     cargo run -p bitnet-cli --features cpu,full-cli -- run \
    │       --model ggml-model-i2_s.gguf --prompt "Test" --max-tokens 64
    │
    ├─▶ QK256 example (strict mode):  # ← AC8: Strict loader documentation
    │     cargo run -p bitnet-cli --features cpu,full-cli -- run \
    │       --model ggml-model-i2_s.gguf --strict-loader --prompt "Test"
    │
    └─▶ Cross-validation example:
          export BITNET_CPP_DIR=/path/to/bitnet.cpp
          cargo run -p xtask -- crossval

    │
    ▼

Detailed Guide:
  docs/quickstart.md → "Using QK256 Models" section
    │
    ├─▶ Automatic format detection explanation
    ├─▶ Strict loader mode usage (AC1)
    ├─▶ Cross-validation workflow (AC4)
    ├─▶ Receipt validation examples (AC4, AC7)
    └─▶ Troubleshooting guide (loader errors, parity failures)

    │
    ▼

Cross-Links:
  ├─▶ docs/howto/use-qk256-models.md (comprehensive QK256 guide)
  ├─▶ docs/explanation/i2s-dual-flavor.md (architecture deep dive)
  ├─▶ docs/reference/quantization-support.md (AC2: tolerance policy)
  └─▶ docs/howto/validate-models.md (validation workflows)
```

**Documentation Coverage Matrix:**

| AC | Documentation Location | Content |
|----|----------------------|---------|
| AC1 | README.md, docs/quickstart.md | `--strict-loader` flag usage |
| AC2 | docs/reference/quantization-support.md | QK256 tolerance policy |
| AC3 | docs/reference/validation-gates.md | K/V cache guardrails (future) |
| AC4 | docs/quickstart.md | Receipt validation examples |
| AC5 | docs/reference/api-docs/ | Tokenizer trait documentation |
| AC6 | docs/development/build-commands.md | FFI build hygiene (future) |
| AC7 | .github/workflows/parity.yml | CI workflow (self-documenting) |
| AC8 | README.md, docs/quickstart.md | QK256 quick-start examples |

---

## Cross-Cutting Concerns

### Feature Flags

All components respect BitNet.rs feature-gated builds:

```toml
# Cargo.toml (bitnet-cli)
[features]
default = []
cpu = ["bitnet-inference/cpu", "bitnet-quantization/cpu"]
gpu = ["bitnet-inference/gpu", "bitnet-kernels/gpu"]
full-cli = ["bitnet-inference", "bitnet-models", "bitnet-tokenizers"]
crossval = ["bitnet-inference/crossval", "crossval"]
```

**Usage:**
```bash
# CPU inference (AC1-AC8 all work)
cargo build --no-default-features --features cpu,full-cli

# GPU inference (AC3 uses GPU K/V cache)
cargo build --no-default-features --features gpu,full-cli

# Cross-validation (AC4, AC5, AC7)
cargo build --no-default-features --features cpu,crossval
```

### Error Propagation

All components use `anyhow::Result` for consistent error handling:

```rust
// AC1: Loader error
Err(anyhow!("Tensor size mismatch (STRICT MODE): ..."))

// AC3: K/V cache error
Err(anyhow!("K/V cache dimension mismatch (layer {}): ...", layer_idx))

// AC4: Receipt validation error
Err(anyhow!("Invalid parity status: '{}' (expected 'ok')", status))

// AC5: Tokenizer parity error
Err(anyhow!("Tokenizer vocab mismatch: Rust={}, C++={}", rust_real, cpp))
```

### Logging Standards

All components use `log` crate with standardized formats:

```rust
// AC1: Loader warnings
log::warn!("QK256 size mismatch (permissive): tensor='...', deviation=...");

// AC2: Tolerance logging
log::debug!("Tokenizer: real_vocab_size={}, gguf_padded_size={}, padding={} tokens");

// AC3: K/V cache warnings (once-per-layer)
log::warn!("Layer {} K/V cache batch mismatch: expected {}, got {}", layer_idx, ...);

// AC4: Parity logging
log::debug!("Tokenizer parity OK: real_vocab_size={} (Rust padded={}, C++ exact={})");
```

---

## Performance Integration

### Model Loading (AC1, AC2)

- **Overhead:** <1% load time increase (one-time validation)
- **Hot path:** Tensor size checks (O(num_tensors), typically <100ms)
- **Cold path:** Error message formatting (only on failures)

### Inference (AC3)

- **Hot path:** `debug_assert!` (zero overhead in release)
- **Cold path:** Once-per-layer warnings (amortized cost, <0.1% inference time)
- **Memory:** Static `Once` guards (64 × 8 bytes = 512 bytes)

### Cross-Validation (AC4)

- **Receipt generation:** <1ms overhead
- **Parity calculation:** O(vocab_size) for cosine similarity (~10ms for 32K vocab)
- **Timeout enforcement:** No overhead (async infrastructure)

---

## Testing Integration

All components follow TDD practices with `// AC:ID` tags:

```rust
// AC1: Strict loader mode rejects misaligned QK256 tensors
#[test]
fn test_strict_loader_rejects_misaligned_qk256() { ... }

// AC2: QK256 tolerance constant usage
#[test]
fn test_qk256_tolerance_calculation() { ... }

// AC3: K/V cache dimension validation
#[test]
fn test_kv_cache_dimension_validation() { ... }

// AC4: Parity harness receipt generation
#[tokio::test]
async fn test_parity_receipt_generation() { ... }

// AC5: Tokenizer real vocab size
#[test]
fn test_gguf_tokenizer_real_vocab_size() { ... }

// AC6: FFI build hygiene verification (bash test)
# Test in scripts/verify_ffi_hygiene.sh

// AC7: CI parity smoke test (GitHub Actions)
# Test in .github/workflows/parity.yml

// AC8: Documentation validation (bash test)
# Test in scripts/validate_quickstart_examples.sh
```

---

## Summary

Issue #469's 8 acceptance criteria integrate seamlessly with the BitNet.rs neural network inference pipeline:

1. **AC1-AC2** enhance Model Loading with strict validation and centralized tolerance
2. **AC3** adds runtime guardrails to Inference (K/V cache validation)
3. **AC4-AC5** improve Cross-Validation with receipts and tokenizer parity
4. **AC6** consolidates FFI build hygiene across all C++ shims
5. **AC7** validates both I2_S flavors in CI
6. **AC8** documents QK256 quick-start for users

All components respect workspace boundaries, feature flags, and BitNet.rs design principles (zero-copy, device-aware, cross-validated, TDD-driven).

---

**Document Control:**
- Review Status: Integration Guide (Ready for Implementation)
- Owner: BitNet.rs Architecture Team
- Issue: #469
- Target: v0.1.0-mvp
