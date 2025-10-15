# Receipt CPU Validation Specification

**Issue:** #462 - CPU Forward Pass with Real Inference (AC3)
**Status:** Specification
**Date:** 2025-10-14

## Context

BitNet.rs uses inference receipts to verify honest compute and prevent silent fallbacks to mock/FP32 paths. The receipt verification system (`xtask/src/verify_receipt.rs`) currently validates GPU backend requirements but lacks CPU backend symmetry.

**Current State:**
- Receipt schema v1.0.0 with compute_path, backend, kernels fields
- GPU validation: `backend="cuda"` requires ≥1 GPU kernel ID
- GPU kernel prefixes: `gemm_`, `wmma_`, `cuda_`, `i2s_gpu_`, `tl*_gpu_`
- No CPU backend validation (silent CPU fallback undetected)

**Problem:**
- CPU backend can report mock kernels without failing validation
- FP32 fallback paths (`fp32_*`, `fallback_*`) not classified as violations
- Dequantization kernels (`dequant*`) incorrectly accepted as quantized

**Required Solution (AC3):**
- Add CPU symmetry: `backend="cpu"` requires ≥1 CPU quantized kernel
- CPU quantized kernel prefixes: `i2s_`, `tl1_`, `tl2_` (use `starts_with()` not `contains()`)
- Excluded prefixes: `dequant*`, `fp32_*`, `fallback_*` (classify as non-quantized)
- Maintain GPU negative test: CUDA backend with CPU kernels fails

## Design

### Receipt Schema (v1.0.0)

**Required Fields:**
```json
{
  "schema_version": "1.0.0",
  "timestamp": "2025-10-14T12:00:00Z",
  "compute_path": "real",  // "real" required, "mock" fails
  "backend": "cpu",        // "cpu" | "cuda" | "metal"
  "kernels": [             // Non-empty array of kernel IDs
    "i2s_gemv",
    "tl1_matmul",
    "rope_apply",
    "attention_real"
  ],
  "deterministic": true,
  "environment": {...},
  "model_info": {...},
  "test_results": {...},
  "performance_baseline": {...}
}
```

**Validation Rules:**
1. `schema_version` ∈ {"1.0.0", "1.0"} (backward compatibility)
2. `compute_path == "real"` (reject "mock")
3. `kernels.length > 0` (at least one kernel)
4. Kernel hygiene: no empty strings, length ≤ 128 chars, count ≤ 10,000
5. Backend-specific kernel requirements:
   - `backend="cuda"` → require GPU kernel (existing)
   - `backend="cpu"` → require CPU quantized kernel (NEW)

### CPU Backend Detection Logic

#### Kernel Classification

**CPU Quantized Kernels (valid for CPU backend):**
```rust
const CPU_QUANTIZED_PREFIXES: &[&str] = &[
    "i2s_",    // I2S quantization (2-bit signed)
    "tl1_",    // TL1 table lookup
    "tl2_",    // TL2 table lookup
];

/// Check if kernel ID represents CPU quantized computation
///
/// # Arguments
/// * `kernel_id` - Kernel identifier string
///
/// # Returns
/// True if kernel starts with CPU quantized prefix (not contains)
///
/// # Safety
/// Uses starts_with() to avoid false positives:
/// - "i2s_gemv" → true (CPU quantized)
/// - "cuda_i2s_gemv" → false (GPU kernel, not CPU)
/// - "dequant_i2s" → false (excluded prefix)
///
/// # Example
/// ```
/// assert!(is_cpu_quantized_kernel("i2s_gemv"));
/// assert!(is_cpu_quantized_kernel("tl1_matmul"));
/// assert!(!is_cpu_quantized_kernel("fp32_matmul"));
/// ```
fn is_cpu_quantized_kernel(kernel_id: &str) -> bool {
    CPU_QUANTIZED_PREFIXES.iter().any(|prefix| kernel_id.starts_with(prefix))
}
```

**Excluded Patterns (fallback, not quantized):**
```rust
const EXCLUDED_PATTERNS: &[&str] = &[
    "dequant",   // Dequantization (FP32 staging)
    "fp32_",     // FP32 fallback matmul
    "fallback_", // Explicit fallback path
];

/// Check if kernel ID is excluded from quantized classification
///
/// # Arguments
/// * `kernel_id` - Kernel identifier string
///
/// # Returns
/// True if kernel matches excluded pattern (contains, not starts_with)
///
/// # Example
/// ```
/// assert!(is_excluded_kernel("dequant_i2s"));
/// assert!(is_excluded_kernel("fp32_matmul"));
/// assert!(!is_excluded_kernel("i2s_gemv"));
/// ```
fn is_excluded_kernel(kernel_id: &str) -> bool {
    EXCLUDED_PATTERNS.iter().any(|pattern| kernel_id.contains(pattern))
}
```

**Combined Validation:**
```rust
/// Validate CPU quantized kernel (AC3 requirement)
///
/// # Logic
/// 1. Check excluded patterns first (fail fast)
/// 2. Check CPU quantized prefixes
/// 3. Reject if neither matches
///
/// # Example
/// ```
/// // Valid CPU quantized
/// assert!(validate_cpu_kernel("i2s_gemv").is_ok());
/// assert!(validate_cpu_kernel("tl2_matmul").is_ok());
///
/// // Invalid: excluded patterns
/// assert!(validate_cpu_kernel("dequant_i2s").is_err());
/// assert!(validate_cpu_kernel("fp32_matmul").is_err());
///
/// // Invalid: GPU kernel (starts with cuda_)
/// assert!(validate_cpu_kernel("cuda_i2s_gemv").is_err());
/// ```
fn validate_cpu_kernel(kernel_id: &str) -> Result<(), String> {
    if is_excluded_kernel(kernel_id) {
        return Err(format!(
            "Kernel '{}' contains excluded pattern (dequant/fp32/fallback)",
            kernel_id
        ));
    }

    if is_cpu_quantized_kernel(kernel_id) {
        Ok(())
    } else {
        Err(format!(
            "Kernel '{}' is not a CPU quantized kernel (expected i2s_/tl1_/tl2_)",
            kernel_id
        ))
    }
}
```

#### Backend-Specific Validation

**CPU Backend Validation (NEW):**
```rust
/// Validate CPU backend receipt (AC3)
///
/// # Requirements
/// - backend == "cpu"
/// - At least one CPU quantized kernel (i2s_/tl1_/tl2_)
/// - No excluded patterns (dequant/fp32/fallback)
///
/// # Errors
/// - No CPU quantized kernels found
/// - All kernels are excluded patterns
/// - Mix of CPU and GPU kernels (suggests silent fallback)
///
/// # Example
/// ```
/// let receipt = json!({
///     "backend": "cpu",
///     "kernels": ["i2s_gemv", "tl1_matmul"]
/// });
/// validate_cpu_receipt(&receipt)?; // OK
///
/// let receipt = json!({
///     "backend": "cpu",
///     "kernels": ["fp32_matmul", "fallback_gemm"]
/// });
/// validate_cpu_receipt(&receipt)?; // Error: no quantized kernels
/// ```
fn validate_cpu_receipt(receipt: &serde_json::Value) -> Result<()> {
    let backend = receipt
        .get("backend")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Receipt missing 'backend' field"))?;

    if backend != "cpu" {
        return Ok(()); // Not CPU backend, skip validation
    }

    let kernels = receipt
        .get("kernels")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("Receipt missing 'kernels' array"))?;

    let kernel_ids: Vec<&str> = kernels
        .iter()
        .filter_map(|v| v.as_str())
        .collect();

    // Count CPU quantized kernels
    let cpu_quantized_count = kernel_ids
        .iter()
        .filter(|&&id| is_cpu_quantized_kernel(id) && !is_excluded_kernel(id))
        .count();

    if cpu_quantized_count == 0 {
        let excluded_count = kernel_ids
            .iter()
            .filter(|&&id| is_excluded_kernel(id))
            .count();

        if excluded_count > 0 {
            bail!(
                "CPU backend verification failed: no quantized kernels, {} excluded patterns found.\n\
                 Excluded kernels: {:?}\n\
                 Expected CPU quantized kernels (examples): {}\n\n\
                 This indicates FP32 fallback or dequantization path. Verify:\n\
                 1) Strict mode: BITNET_STRICT_MODE=1\n\
                 2) Quantization support: cargo test -p bitnet-kernels --features cpu\n\
                 3) Model compatibility: cargo run -p bitnet-cli -- compat-check model.gguf",
                excluded_count,
                kernel_ids.iter().filter(|&&id| is_excluded_kernel(id)).collect::<Vec<_>>(),
                CPU_QUANTIZED_EXAMPLES.join(", ")
            );
        } else {
            bail!(
                "CPU backend verification failed: no quantized kernels found.\n\
                 Actual kernels: {:?}\n\
                 Expected CPU quantized kernels (examples): {}\n\n\
                 This likely indicates mock inference. Verify:\n\
                 1) Build: cargo build --features cpu\n\
                 2) Tests: cargo test -p bitnet-inference --features cpu\n\
                 3) Receipt generation: compute_path should be 'real'",
                kernel_ids,
                CPU_QUANTIZED_EXAMPLES.join(", ")
            );
        }
    }

    Ok(())
}

const CPU_QUANTIZED_EXAMPLES: &[&str] = &[
    "i2s_gemv",
    "i2s_matmul",
    "tl1_matmul",
    "tl2_matmul",
];
```

**GPU Backend Validation (existing, maintained):**
```rust
/// Validate GPU backend receipt (existing logic)
///
/// # Requirements
/// - backend == "cuda"
/// - At least one GPU kernel (gemm_/wmma_/cuda_/i2s_gpu_/tl*_gpu_)
///
/// # Maintained for backward compatibility
fn validate_gpu_receipt(receipt: &serde_json::Value, require_gpu_kernels: bool) -> Result<()> {
    let backend = receipt.get("backend").and_then(|v| v.as_str()).unwrap_or("cpu");
    let must_require_gpu = backend.eq_ignore_ascii_case("cuda");

    let kernels = receipt
        .get("kernels")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("Receipt missing 'kernels' array"))?;

    let kernel_ids: Vec<&str> = kernels.iter().filter_map(|v| v.as_str()).collect();

    let require_gpu = require_gpu_kernels || must_require_gpu;
    if require_gpu {
        let has_gpu_kernel = kernel_ids.iter().any(|id| is_gpu_kernel_id(id));

        if !has_gpu_kernel {
            let reason = if must_require_gpu {
                "backend is 'cuda'"
            } else {
                "--require-gpu-kernels flag set"
            };

            bail!(
                "GPU kernel verification required ({}) but no GPU kernels found.\n\
                 Expected (examples): {}\n\
                 Actual kernels: {:?}\n\n\
                 This likely indicates silent CPU fallback. Verify:\n\
                 1) GPU build: cargo build --features gpu\n\
                 2) CUDA runtime: nvidia-smi\n\
                 3) Device selection: Device::Cuda(0) in inference",
                reason,
                GPU_KERNEL_EXAMPLES.join(", "),
                kernel_ids
            );
        }
    }

    Ok(())
}

const GPU_KERNEL_EXAMPLES: &[&str] = &[
    "gemm_cuda",
    "wmma_i2s",
    "i2s_gpu_gemv",
    "tl1_gpu_matmul",
];

fn is_gpu_kernel_id(kernel_id: &str) -> bool {
    const GPU_KERNEL_PREFIXES: &[&str] = &[
        "gemm_", "wmma_", "cuda_", "i2s_gpu_", "tl1_gpu_", "tl2_gpu_",
    ];
    GPU_KERNEL_PREFIXES.iter().any(|prefix| kernel_id.starts_with(prefix))
}
```

### Integration into verify_receipt_cmd

**Updated Function Flow:**
```rust
/// Verify inference receipt (AC3 integration)
///
/// # Validation Steps
/// 1. Schema version compatibility
/// 2. Compute path verification (real vs mock)
/// 3. Kernel hygiene (non-empty, length, count)
/// 4. GPU backend validation (existing)
/// 5. CPU backend validation (NEW - AC3)
/// 6. Quantization claims verification (existing)
///
/// # Exit Codes
/// - 0: Receipt valid
/// - 1: Receipt invalid or missing
fn verify_receipt_cmd(path: &Path, require_gpu_kernels: bool) -> Result<()> {
    println!("{}", style("🔍 Verifying inference receipt…").bold());

    // Read and parse receipt
    let contents = fs::read_to_string(path)
        .with_context(|| format!("Failed to read receipt: {}", path.display()))?;

    let receipt: Value = serde_json::from_str(&contents)
        .with_context(|| format!("Invalid JSON in receipt: {}", path.display()))?;

    // Step 1: Check schema version
    let schema_version = receipt
        .get("schema_version")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Receipt missing 'schema_version' field"))?;

    if schema_version != "1.0.0" && schema_version != "1.0" {
        bail!("Unsupported schema_version '{}' (expected '1.0.0' or '1.0')", schema_version);
    }

    // Step 2: Check compute_path
    let compute_path = receipt
        .get("compute_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Receipt missing 'compute_path' field"))?;

    if compute_path != "real" {
        bail!("compute_path must be 'real' (got '{}') — mock inference not allowed", compute_path);
    }

    // Step 3: Kernel hygiene
    let kernels = receipt
        .get("kernels")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("Receipt missing 'kernels' array"))?;

    if kernels.is_empty() {
        bail!("Receipt has empty kernels[] — requires at least one real kernel");
    }

    let kernel_ids: Vec<&str> = kernels.iter().filter_map(|v| v.as_str()).collect();

    // Hygiene checks
    if kernel_ids.iter().any(|s| s.trim().is_empty()) {
        bail!("kernels[] contains empty kernel ID");
    }
    if kernel_ids.iter().any(|s| s.len() > 128) {
        bail!("kernels[] contains kernel ID longer than 128 characters");
    }
    if kernel_ids.len() > 10_000 {
        bail!("kernels[] contains too many entries (> 10,000)");
    }

    // Step 4: GPU backend validation (existing)
    validate_gpu_receipt(&receipt, require_gpu_kernels)?;

    // Step 5: CPU backend validation (NEW - AC3)
    validate_cpu_receipt(&receipt)?;

    // Step 6: Quantization claims verification (existing)
    verify_quantization_claims(&receipt)?;

    // Success
    println!("{}", style("✅ Receipt verification passed").green().bold());
    println!("   Schema: {}", schema_version);
    println!("   Compute path: {}", compute_path);
    println!("   Kernels: {} executed", kernels.len());
    println!("   Backend: {}", receipt.get("backend").and_then(|v| v.as_str()).unwrap_or("unknown"));

    Ok(())
}
```

## Validation

### Test Cases

#### Positive Tests (Receipt Should Pass)

```rust
// AC3: CPU backend with valid quantized kernels
#[test]
fn test_ac3_receipt_cpu_kernel_honesty_positive() {
    let receipt = json!({
        "schema_version": "1.0.0",
        "timestamp": "2025-10-14T12:00:00Z",
        "compute_path": "real",
        "backend": "cpu",
        "kernels": [
            "i2s_gemv",
            "tl1_matmul",
            "tl2_matmul",
            "rope_apply",
            "attention_real"
        ],
        "deterministic": true,
        "environment": {},
        "model_info": {},
        "test_results": {},
        "performance_baseline": {}
    });

    // Write to temp file
    let temp_path = temp_dir().join("test_cpu_valid.json");
    std::fs::write(&temp_path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();

    // Verify should pass
    let result = verify_receipt_cmd(&temp_path, false);
    assert!(result.is_ok(), "CPU receipt with quantized kernels should pass");
}

// AC3: CPU backend with mixed kernels (at least one quantized)
#[test]
fn test_ac3_receipt_cpu_mixed_kernels() {
    let receipt = json!({
        "schema_version": "1.0.0",
        "compute_path": "real",
        "backend": "cpu",
        "kernels": [
            "i2s_gemv",         // Quantized (valid)
            "rope_apply",       // Utility (OK)
            "softmax_cpu"       // Utility (OK)
        ]
    });

    let temp_path = temp_dir().join("test_cpu_mixed.json");
    std::fs::write(&temp_path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();

    let result = verify_receipt_cmd(&temp_path, false);
    assert!(result.is_ok(), "CPU receipt with at least one quantized kernel should pass");
}
```

#### Negative Tests (Receipt Should Fail)

```rust
// AC3: CPU backend with no quantized kernels
#[test]
fn test_ac3_receipt_cpu_kernel_honesty_negative() {
    let receipt = json!({
        "schema_version": "1.0.0",
        "compute_path": "real",
        "backend": "cpu",
        "kernels": [
            "rope_apply",    // Utility, not quantized
            "softmax_cpu",   // Utility, not quantized
            "attention_mock" // Mock kernel
        ]
    });

    let temp_path = temp_dir().join("test_cpu_invalid.json");
    std::fs::write(&temp_path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();

    let result = verify_receipt_cmd(&temp_path, false);
    assert!(result.is_err(), "CPU receipt without quantized kernels should fail");
    assert!(result.unwrap_err().to_string().contains("no quantized kernels found"));
}

// AC3: CPU backend with excluded patterns (FP32 fallback)
#[test]
fn test_ac3_receipt_cpu_fp32_fallback() {
    let receipt = json!({
        "schema_version": "1.0.0",
        "compute_path": "real",
        "backend": "cpu",
        "kernels": [
            "fp32_matmul",
            "fallback_gemm",
            "dequant_i2s"
        ]
    });

    let temp_path = temp_dir().join("test_cpu_fp32.json");
    std::fs::write(&temp_path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();

    let result = verify_receipt_cmd(&temp_path, false);
    assert!(result.is_err(), "CPU receipt with FP32 fallback should fail");
    assert!(result.unwrap_err().to_string().contains("excluded patterns"));
}

// AC3: GPU backend with CPU kernels (silent fallback)
#[test]
fn test_ac3_receipt_gpu_cpu_kernel_mismatch() {
    let receipt = json!({
        "schema_version": "1.0.0",
        "compute_path": "real",
        "backend": "cuda",
        "kernels": [
            "i2s_gemv",     // CPU kernel, not GPU
            "tl1_matmul"    // CPU kernel, not GPU
        ]
    });

    let temp_path = temp_dir().join("test_gpu_cpu_mismatch.json");
    std::fs::write(&temp_path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();

    let result = verify_receipt_cmd(&temp_path, false);
    assert!(result.is_err(), "GPU backend with CPU kernels should fail");
    assert!(result.unwrap_err().to_string().contains("no GPU kernels found"));
}

// AC3: Excluded pattern detection
#[test]
fn test_ac3_excluded_pattern_matching() {
    use crate::is_excluded_kernel;

    assert!(is_excluded_kernel("dequant_i2s"));
    assert!(is_excluded_kernel("fp32_matmul"));
    assert!(is_excluded_kernel("fallback_gemm"));
    assert!(is_excluded_kernel("something_dequant_else"));

    assert!(!is_excluded_kernel("i2s_gemv"));
    assert!(!is_excluded_kernel("tl1_matmul"));
}

// AC3: CPU quantized prefix matching (starts_with not contains)
#[test]
fn test_ac3_cpu_quantized_prefix_matching() {
    use crate::is_cpu_quantized_kernel;

    // Valid CPU quantized
    assert!(is_cpu_quantized_kernel("i2s_gemv"));
    assert!(is_cpu_quantized_kernel("i2s_matmul"));
    assert!(is_cpu_quantized_kernel("tl1_matmul"));
    assert!(is_cpu_quantized_kernel("tl2_matmul"));

    // Invalid: GPU kernels (contains i2s_ but starts with cuda_)
    assert!(!is_cpu_quantized_kernel("cuda_i2s_gemv"));
    assert!(!is_cpu_quantized_kernel("gpu_tl1_matmul"));

    // Invalid: excluded patterns
    assert!(!is_cpu_quantized_kernel("dequant_i2s"));
    assert!(!is_cpu_quantized_kernel("fp32_i2s"));

    // Invalid: utility kernels
    assert!(!is_cpu_quantized_kernel("rope_apply"));
    assert!(!is_cpu_quantized_kernel("softmax_cpu"));
}
```

### Integration Tests

```rust
// AC3: Full E2E validation with real receipt generation
#[test]
fn test_ac3_e2e_cpu_receipt_generation() {
    // Generate CPU inference receipt
    let receipt = InferenceReceipt::generate(
        "cpu",
        vec![
            "i2s_gemv".to_string(),
            "tl1_matmul".to_string(),
            "rope_apply".to_string(),
        ],
    )?;

    // Write to file
    let receipt_path = std::env::temp_dir().join("e2e_cpu_receipt.json");
    std::fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)?;

    // Verify receipt
    let result = verify_receipt_cmd(&receipt_path, false);
    assert!(result.is_ok(), "E2E CPU receipt should pass validation");
}

// AC3: GPU receipt continues to work (backward compatibility)
#[test]
fn test_ac3_e2e_gpu_receipt_validation() {
    let receipt = InferenceReceipt::generate(
        "cuda",
        vec![
            "gemm_cuda".to_string(),
            "i2s_gpu_gemv".to_string(),
        ],
    )?;

    let receipt_path = std::env::temp_dir().join("e2e_gpu_receipt.json");
    std::fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)?;

    let result = verify_receipt_cmd(&receipt_path, false);
    assert!(result.is_ok(), "GPU receipt validation should still work");
}
```

### Command-Line Tests

```bash
# AC3: CPU receipt validation (positive)
cargo run -p xtask -- verify-receipt ci/inference.json
# Expected: ✅ Receipt verification passed

# AC3: CPU receipt validation (negative - mock kernels)
echo '{
  "schema_version": "1.0.0",
  "compute_path": "real",
  "backend": "cpu",
  "kernels": ["mock_kernel"]
}' > /tmp/bad_cpu_receipt.json

cargo run -p xtask -- verify-receipt /tmp/bad_cpu_receipt.json
# Expected: Error: CPU backend verification failed: no quantized kernels found

# AC3: CPU receipt validation (negative - FP32 fallback)
echo '{
  "schema_version": "1.0.0",
  "compute_path": "real",
  "backend": "cpu",
  "kernels": ["fp32_matmul", "dequant_i2s"]
}' > /tmp/fp32_receipt.json

cargo run -p xtask -- verify-receipt /tmp/fp32_receipt.json
# Expected: Error: no quantized kernels, 2 excluded patterns found

# AC3: GPU receipt validation (backward compatibility)
cargo run -p xtask -- verify-receipt --require-gpu-kernels ci/gpu_receipt.json
# Expected: ✅ Receipt verification passed (existing test)
```

## Implementation Sequence

### Phase 1: Add CPU Kernel Classification Functions

1. **Add constants:**
   - `CPU_QUANTIZED_PREFIXES`
   - `EXCLUDED_PATTERNS`
   - `CPU_QUANTIZED_EXAMPLES`

2. **Add helper functions:**
   - `is_cpu_quantized_kernel(kernel_id: &str) -> bool`
   - `is_excluded_kernel(kernel_id: &str) -> bool`

3. **Add unit tests:**
   ```bash
   cargo test -p xtask test_ac3_cpu_quantized_prefix_matching
   cargo test -p xtask test_ac3_excluded_pattern_matching
   ```

**Validation:**
```bash
cargo test -p xtask --lib
```

### Phase 2: Implement validate_cpu_receipt

1. **Add function:**
   - `validate_cpu_receipt(receipt: &Value) -> Result<()>`

2. **Add tests:**
   ```bash
   cargo test -p xtask test_ac3_receipt_cpu_kernel_honesty_positive
   cargo test -p xtask test_ac3_receipt_cpu_kernel_honesty_negative
   cargo test -p xtask test_ac3_receipt_cpu_fp32_fallback
   ```

**Validation:**
```bash
cargo test -p xtask test_ac3_receipt
```

### Phase 3: Integrate into verify_receipt_cmd

1. **Update function:**
   - Add `validate_cpu_receipt(&receipt)?;` after GPU validation

2. **Maintain backward compatibility:**
   - GPU validation unchanged
   - Schema validation unchanged
   - Kernel hygiene unchanged

3. **Add integration tests:**
   ```bash
   cargo test -p xtask test_ac3_e2e_cpu_receipt_generation
   cargo test -p xtask test_ac3_receipt_gpu_cpu_kernel_mismatch
   ```

**Validation:**
```bash
cargo test -p xtask verify_receipt
```

### Phase 4: Update Documentation

1. **Update `docs/reference/validation-gates.md`:**
   - Document CPU backend validation
   - Add kernel prefix reference table
   - Include error message examples

2. **Update `README.md`:**
   - Add receipt verification section
   - Include example commands

3. **Update CHANGELOG.md:**
   - Add AC3: CPU backend receipt validation

## Error Messages

### CPU Backend Errors

**No Quantized Kernels:**
```
CPU backend verification failed: no quantized kernels found.
Actual kernels: ["rope_apply", "softmax_cpu", "attention_mock"]
Expected CPU quantized kernels (examples): i2s_gemv, i2s_matmul, tl1_matmul, tl2_matmul

This likely indicates mock inference. Verify:
1) Build: cargo build --features cpu
2) Tests: cargo test -p bitnet-inference --features cpu
3) Receipt generation: compute_path should be 'real'
```

**FP32 Fallback Detected:**
```
CPU backend verification failed: no quantized kernels, 2 excluded patterns found.
Excluded kernels: ["fp32_matmul", "dequant_i2s"]
Expected CPU quantized kernels (examples): i2s_gemv, i2s_matmul, tl1_matmul, tl2_matmul

This indicates FP32 fallback or dequantization path. Verify:
1) Strict mode: BITNET_STRICT_MODE=1
2) Quantization support: cargo test -p bitnet-kernels --features cpu
3) Model compatibility: cargo run -p bitnet-cli -- compat-check model.gguf
```

### GPU Backend Errors (existing)

**Silent CPU Fallback:**
```
GPU kernel verification required (backend is 'cuda') but no GPU kernels found.
Expected (examples): gemm_cuda, wmma_i2s, i2s_gpu_gemv, tl1_gpu_matmul
Actual kernels: ["i2s_gemv", "tl1_matmul"]

This likely indicates silent CPU fallback. Verify:
1) GPU build: cargo build --features gpu
2) CUDA runtime: nvidia-smi
3) Device selection: Device::Cuda(0) in inference
```

## References

### Related Documentation

- `docs/reference/validation-gates.md` - Validation system technical reference
- `docs/explanation/receipt-validation.md` - Receipt schema and validation
- `docs/explanation/cpu-inference-architecture.md` - CPU forward pass design
- `docs/development/test-suite.md` - Testing framework

### Existing Code

- `xtask/src/main.rs:4212` - `verify_receipt_cmd()` function
- `crates/bitnet-inference/src/receipts.rs` - InferenceReceipt generation
- `ci/inference.json` - Production receipt example

### Issue References

- **Issue #462 (AC3):** Receipt CPU validation (this spec)
- **PR #452:** Receipt verification gate
- **Issue #439:** GPU determinism and device testing
