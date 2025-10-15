# ADR-011: Receipt Schema Backward Compatibility (v1.0.0 → v1.1.0)

## Status

**ACCEPTED** - Issue #453 Implementation

## Context

PR #452 established receipt schema v1.0.0 for performance validation. Issue #453 requires extending the schema to include quantization metadata (`kernel_path`, `quantization` section) while maintaining compatibility with existing v1.0.0 readers and writers.

### Current Schema (v1.0.0)

```json
{
  "schema_version": "1.0.0",
  "timestamp": "2025-10-14T01:33:28.076791999+00:00",
  "compute_path": "real",
  "backend": "cuda",
  "deterministic": true,
  "tokens_generated": 128,
  "tokens_per_second": 87.5,
  "kernels": ["gemm_fp16", "i2s_gpu_quantize"]
}
```

### Requirements

1. **Backward Compatibility:** v1.0.0 readers must handle v1.1.0 receipts (ignore unknown fields)
2. **Forward Compatibility:** v1.1.0 readers must handle v1.0.0 receipts (infer missing fields)
3. **CI Stability:** Existing CI workflows must continue without modification
4. **Schema Evolution:** Enable future extensions (v1.2.0, v1.3.0, etc.)

## Decision

We extend the receipt schema to v1.1.0 using **optional fields** with backward-compatible defaults.

### Schema v1.1.0 Definition

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Receipt {
    // Existing v1.0.0 fields (required)
    pub schema_version: String,
    pub timestamp: String,
    pub compute_path: String,
    pub backend: String,
    pub tokens_generated: usize,
    pub tokens_per_second: f64,
    pub kernels: Vec<String>,

    // New v1.1.0 fields (optional, skip if None)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_path: Option<String>,  // "native_quantized" | "fp32_fallback"

    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<QuantizationMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationMetadata {
    pub types_used: Vec<String>,        // ["I2S"], ["TL1", "TL2"], etc.
    pub fallback_count: usize,          // 0 for strict mode
    #[serde(default)]
    pub device_aware_selection: bool,   // Device-aware quantization
}
```

### Backward Compatibility Strategy

**v1.0.0 Reader Behavior:**
```rust
// v1.0.0 readers use serde's default behavior for unknown fields
#[derive(Deserialize)]
struct ReceiptV1_0 {
    pub schema_version: String,
    pub compute_path: String,
    pub backend: String,
    pub kernels: Vec<String>,
    // ... other v1.0.0 fields

    // Unknown fields (kernel_path, quantization) are silently ignored
}

// Deserializing v1.1.0 receipt with v1.0.0 reader:
let receipt: ReceiptV1_0 = serde_json::from_str(v1_1_receipt_json)?;
// kernel_path and quantization fields are skipped (no error)
```

**v1.1.0 Reader Behavior:**
```rust
// v1.1.0 readers handle both v1.0.0 and v1.1.0 receipts
let receipt: Receipt = serde_json::from_str(receipt_json)?;

match receipt.schema_version.as_str() {
    "1.0.0" => {
        // Infer kernel_path from kernels array
        let kernel_path = if receipt.kernel_path.is_none() {
            Some(infer_kernel_path(&receipt.kernels))
        } else {
            receipt.kernel_path
        };

        // quantization field remains None (v1.0.0 doesn't include it)
    }
    "1.1.0" => {
        // Use explicit kernel_path and quantization fields
        let kernel_path = receipt.kernel_path.as_deref().unwrap_or("unknown");
        let quant = receipt.quantization.as_ref();
    }
    _ => return Err(anyhow!("Unsupported schema version")),
}
```

### Field Semantics

**`kernel_path` (optional string):**
- **Values:** `"native_quantized"`, `"fp32_fallback"`, `"unknown"`
- **Default (v1.0.0 compatibility):** Inferred from `kernels` array
- **Purpose:** Explicit declaration of quantization path used

**`quantization` (optional object):**
- **`types_used` (array of strings):** Quantization types (`["I2S"]`, `["TL1", "TL2"]`)
- **`fallback_count` (integer):** Number of FP32 fallback operations (0 for strict mode)
- **`device_aware_selection` (boolean):** Whether device-aware quantization was used
- **Default (v1.0.0 compatibility):** `None` (field not present)

### Serialization Examples

**v1.1.0 Receipt with Native Quantization:**
```json
{
  "schema_version": "1.1.0",
  "timestamp": "2025-10-14T02:15:42.123456789+00:00",
  "compute_path": "real",
  "backend": "cuda",
  "tokens_generated": 16,
  "tokens_per_second": 87.5,
  "kernels": ["gemm_fp16", "i2s_gpu_quantize", "wmma_matmul"],
  "kernel_path": "native_quantized",
  "quantization": {
    "types_used": ["I2S"],
    "fallback_count": 0,
    "device_aware_selection": true
  }
}
```

**v1.1.0 Receipt with FP32 Fallback:**
```json
{
  "schema_version": "1.1.0",
  "compute_path": "real",
  "backend": "cuda",
  "tokens_generated": 16,
  "tokens_per_second": 35.0,
  "kernels": ["dequant_i2s", "fp32_matmul"],
  "kernel_path": "fp32_fallback",
  "quantization": {
    "types_used": [],
    "fallback_count": 16
  }
}
```

**v1.0.0 Receipt (Legacy):**
```json
{
  "schema_version": "1.0.0",
  "compute_path": "real",
  "backend": "cuda",
  "tokens_generated": 128,
  "tokens_per_second": 87.5,
  "kernels": ["gemm_fp16", "i2s_gpu_quantize"]
  // kernel_path and quantization fields not present
}
```

## Rationale

### Why Optional Fields?

**Alternative 1: Required Fields with Defaults**
- **Rejected:** Breaks v1.0.0 readers (strict parsing)
- **Rejected:** Requires migration of all existing receipts

**Alternative 2: New Top-Level Schema**
- **Rejected:** Duplicates existing fields
- **Rejected:** Breaks CI workflows expecting `Receipt` type

**Alternative 3: Nested Schema Version**
- **Rejected:** Complicates parsing logic
- **Rejected:** Difficult to infer missing fields

**Selected: Optional Fields (Best Practice)**
- ✅ v1.0.0 readers ignore unknown fields (serde default)
- ✅ v1.1.0 readers infer missing fields (forward compatibility)
- ✅ No migration required for existing receipts
- ✅ CI workflows continue without modification

### Field Inference Strategy

**`kernel_path` Inference (v1.0.0 → v1.1.0):**
```rust
pub fn infer_kernel_path(kernel_ids: &[String]) -> String {
    let has_quantized = kernel_ids.iter().any(is_quantized_kernel);
    let has_fallback = kernel_ids.iter().any(is_fallback_kernel);

    if has_quantized && !has_fallback {
        "native_quantized".to_string()
    } else if has_fallback {
        "fp32_fallback".to_string()
    } else {
        "unknown".to_string()
    }
}
```

**`quantization` Inference (v1.0.0 → v1.1.0):**
- **Not Inferred:** v1.0.0 receipts lack quantization metadata
- **Default:** `None` (field not present)
- **Validation:** v1.1.0 validation logic handles `None` gracefully

## Implementation Strategy

### Phase 1: Schema Definition (Week 3, Days 15-17)

**Files Modified:**
- `xtask/src/verify_receipt.rs` (define `Receipt` and `QuantizationMetadata`)

**Validation:**
```bash
# Test v1.0.0 receipt parsing with v1.1.0 reader
cargo test -p xtask test_parse_v1_0_receipt_with_v1_1_reader

# Test v1.1.0 receipt parsing
cargo test -p xtask test_parse_v1_1_receipt

# Test field inference
cargo test -p xtask test_infer_kernel_path_from_kernels
```

### Phase 2: Backward Compatibility Testing (Week 3, Days 18-20)

**Test Cases:**
1. **v1.0.0 Reader + v1.1.0 Receipt:** Unknown fields ignored (no error)
2. **v1.1.0 Reader + v1.0.0 Receipt:** Missing fields inferred (no error)
3. **v1.1.0 Reader + v1.1.0 Receipt:** Explicit fields used (no inference)

**Validation:**
```bash
# Test backward compatibility
cargo test -p xtask test_v1_0_reader_v1_1_receipt

# Test forward compatibility
cargo test -p xtask test_v1_1_reader_v1_0_receipt

# Test CI stability (existing workflows)
cargo run -p xtask -- verify-receipt ci/inference_v1_0.json
cargo run -p xtask -- verify-receipt ci/inference_v1_1.json
```

### Phase 3: CI Integration (Week 3, Day 21)

**CI Workflow Updates:**
- No changes required (optional fields are backward compatible)
- Existing workflows continue using v1.0.0 receipts
- New workflows can opt-in to v1.1.0 (set `schema_version: "1.1.0"`)

## Consequences

### Positive

1. **Zero Breaking Changes:** Existing CI workflows continue without modification
2. **Graceful Migration:** v1.0.0 receipts remain valid indefinitely
3. **Field Inference:** v1.1.0 readers infer missing fields from v1.0.0 receipts
4. **Future-Proof:** Optional field pattern enables future schema versions (v1.2.0, etc.)

### Negative

1. **Inference Logic Complexity:** v1.1.0 readers must infer `kernel_path` from `kernels` array
2. **Field Nullability:** Optional fields require `Option<T>` handling (Rust)
3. **Documentation Overhead:** Must document both v1.0.0 and v1.1.0 semantics

### Mitigation

- **Inference Complexity:** Encapsulated in `infer_kernel_path` helper function
- **Field Nullability:** Serde's `#[serde(skip_serializing_if = "Option::is_none")]` simplifies serialization
- **Documentation:** Comprehensive examples in `docs/reference/strict-mode-api.md`

## Validation Metrics

### Success Criteria

- ✅ v1.0.0 readers parse v1.1.0 receipts without error (backward compatibility)
- ✅ v1.1.0 readers parse v1.0.0 receipts without error (forward compatibility)
- ✅ Field inference accuracy: 100% for `kernel_path` (quantized vs fallback)
- ✅ CI stability: Zero workflow modifications required
- ✅ Schema validation: v1.0.0 and v1.1.0 receipts both validate successfully

### Measurable Commands

```bash
# Backward compatibility test
cargo test -p xtask test_v1_0_reader_v1_1_receipt

# Forward compatibility test
cargo test -p xtask test_v1_1_reader_v1_0_receipt

# Field inference test
cargo test -p xtask test_infer_kernel_path_from_kernels

# CI stability verification
cargo run -p xtask -- verify-receipt ci/inference_v1_0.json
cargo run -p xtask -- verify-receipt ci/inference_v1_1.json
```

## Related ADRs

- **ADR-010:** Three-Tier Validation Strategy
- **ADR-012:** Kernel ID Naming Conventions
- **ADR-013:** FP32 Fallback Detection Mechanisms

## References

- **Issue #453:** Strict Quantization Guards
- **PR #452:** Receipt Verification Infrastructure (schema v1.0.0)
- **Serde Documentation:** https://serde.rs/field-attrs.html
