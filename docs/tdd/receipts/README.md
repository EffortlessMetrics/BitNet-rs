# Receipt Test Examples

This directory contains example receipts for testing the `xtask verify-receipt` command.

## Overview

Receipts are JSON artifacts documenting real inference execution. They follow schema version 1.0.0 and are validated against quality gates to ensure honest compute reporting.

## Test Files

### `cpu_positive_example.json` - Valid CPU Receipt

**Purpose**: Demonstrates a **valid** CPU inference receipt that passes all verification checks.

**What it tests**:
- ✅ `schema_version`: "1.0.0" (required)
- ✅ `compute_path`: "real" (honest compute - no mock inference)
- ✅ `backend`: "cpu" (valid backend)
- ✅ `kernels`: Non-empty array with valid CPU quantized kernel IDs
  - `i2s_gemv`: I2_S GEMV (general matrix-vector multiply) kernel
  - `i2s_matmul_avx2`: I2_S AVX2-optimized matrix multiplication
  - `tl1_lookup_neon`: TL1 table lookup with NEON SIMD
  - `tl2_forward`: TL2 forward pass kernel
- ✅ `deterministic`: true (deterministic inference enabled)
- ✅ `environment`: Contains required environment variables
  - `BITNET_DETERMINISTIC=1`: Deterministic mode
  - `BITNET_SEED=42`: Deterministic seed
  - `RAYON_NUM_THREADS=1`: Single-threaded for determinism
  - System metadata (Rust version, OS, CPU brand)
- ✅ `test_results`: All tests passed (0 failures)
- ✅ `performance_baseline`: Valid performance metrics
  - `tokens_per_second: 0.5` (realistic CPU performance for QK256)
  - All timing values are positive
- ✅ `corrections`: Empty (no model corrections applied)

**Expected result**: `cargo run -p xtask -- verify-receipt --path docs/tdd/receipts/cpu_positive_example.json` should **PASS**

**Verification command**:
```bash
cargo run -p xtask -- verify-receipt --path docs/tdd/receipts/cpu_positive_example.json
```

**Expected output**:
```
🔍 Verifying inference receipt…
✅ Receipt verification passed
   Schema: 1.0.0
   Compute path: real
   Kernels: 4 executed
   Backend: cpu
```

---

### `cpu_negative_example.json` - Invalid Receipt (Multiple Violations)

**Purpose**: Demonstrates an **invalid** receipt that fails verification due to multiple violations.

**What it tests (violations)**:
- ❌ `compute_path`: "mock" (violates honest compute requirement - must be "real")
- ❌ `kernels`: Contains empty string `""` (violates kernel ID hygiene)
- ❌ `test_results.failed`: 2 (violates zero-failure requirement)
- ❌ `performance_baseline.tokens_per_second`: -1.0 (invalid negative value)

**Other characteristics**:
- ✅ `schema_version`: "1.0.0" (valid)
- ✅ `backend`: "cpu" (valid)
- ✅ `deterministic`: false (valid - determinism not required)
- ✅ `environment`: Contains basic system metadata

**Expected result**: `cargo run -p xtask -- verify-receipt --path docs/tdd/receipts/cpu_negative_example.json` should **FAIL**

**Verification command**:
```bash
cargo run -p xtask -- verify-receipt --path docs/tdd/receipts/cpu_negative_example.json
```

**Expected output** (should fail with multiple errors):
```
🔍 Verifying inference receipt…
error: compute_path must be 'real' (got 'mock') — mock inference not allowed
```

Alternative failures (depending on validation order):
```
error: kernels[] contains empty kernel ID
```

---

## Receipt Schema v1.0.0 Reference

### Required Fields

| Field | Type | Description | Validation Rules |
|-------|------|-------------|------------------|
| `schema_version` | string | Schema version | Must be "1.0.0" or "1.0" |
| `timestamp` | string | ISO 8601 timestamp | ISO 8601 format |
| `compute_path` | string | Compute path | Must be "real" (not "mock") |
| `backend` | string | Backend used | "cpu" \| "cuda" \| "metal" |
| `kernels` | array[string] | Executed kernels | Non-empty, no empty strings, ≤128 chars each, ≤10K total |
| `deterministic` | boolean | Deterministic mode | true \| false |
| `environment` | object | Environment vars | Map of string → string |
| `model_info` | object | Model config | See ModelInfo schema |
| `test_results` | object | Test results | See TestResults schema |
| `performance_baseline` | object | Performance metrics | See PerformanceBaseline schema |
| `corrections` | array | Applied corrections | Empty for production CI |

### Kernel ID Hygiene Rules

1. **Non-empty array**: `kernels` must contain at least one kernel ID
2. **No empty strings**: Each kernel ID must be non-empty and not whitespace-only
3. **Length limit**: Each kernel ID must be ≤ 128 characters
4. **Count limit**: Total kernel count must be ≤ 10,000
5. **No mock kernels**: Kernel IDs must not contain "mock" (case-insensitive)
6. **Backend-specific**:
   - CPU backend: Should use CPU-specific kernels (`i2s_cpu_*`, `matmul_cpu_*`, etc.)
   - GPU backend (`cuda`): Must contain at least one GPU kernel (`gemm_*`, `wmma_*`, `cuda_*`, `i2s_gpu_*`, etc.)

### Auto-GPU Enforcement

When `backend == "cuda"`, the validator **automatically** requires at least one GPU kernel, even without `--require-gpu-kernels` flag. This prevents silent CPU fallback.

**GPU kernel naming conventions**:
- `gemm_*`: GEMM kernels (e.g., `gemm_fp16`, `gemm_bf16`)
- `wmma_*`: Tensor Core kernels (e.g., `wmma_matmul`)
- `cuda_*`: CUDA utilities (e.g., `cuda_sync`, `cuda_memcpy`)
- `i2s_gpu_*`: I2_S GPU quantization (e.g., `i2s_gpu_quantize`)
- `tl1_gpu_*`: TL1 GPU quantization
- `tl2_gpu_*`: TL2 GPU quantization

### Compute Path Requirements

- **"real"**: Honest compute - actual inference execution (REQUIRED)
- **"mock"**: Mock inference - test scaffolding (REJECTED by validator)

The validator enforces `compute_path == "real"` to ensure receipts document actual inference, not test mocks.

### Test Results Requirements

- `test_results.failed` must be 0 (no failed tests)
- If `accuracy_tests` present, all accuracy tests must pass
- If `deterministic == true` and `determinism_tests` present, `identical_sequences` must be true

### Performance Metrics

While not strictly validated, performance metrics should be realistic:

- **CPU performance**: ~0.1-2.0 tok/s for 2B models (QK256 scalar kernels)
- **GPU performance**: ~30-150 tok/s for 2B models (with proper GPU kernels)
- **Negative values**: Should be avoided (indicates measurement errors)

### Corrections Field

- Must be empty (`[]`) in production CI builds
- Non-empty corrections require `BITNET_ALLOW_CORRECTIONS=1` environment variable
- Each correction documents LayerNorm rescaling or similar model-specific fixes

---

## Usage in Tests

These example receipts are used by:

1. **`xtask/tests/verify_receipt.rs`**: Unit tests for receipt validation logic
2. **`xtask/tests/verify_receipt_cmd.rs`**: Integration tests for CLI verification command
3. **Manual testing**: Developers testing receipt verification behavior

### Example Test Pattern

```rust
#[test]
fn test_cpu_positive_receipt_passes() {
    let receipt_path = workspace_root()
        .join("docs/tdd/receipts/cpu_positive_example.json");

    let result = verify_receipt_cmd(&receipt_path, false);
    assert!(result.is_ok(), "Valid CPU receipt should pass: {:?}", result.err());
}

#[test]
fn test_cpu_negative_receipt_fails() {
    let receipt_path = workspace_root()
        .join("docs/tdd/receipts/cpu_negative_example.json");

    let result = verify_receipt_cmd(&receipt_path, false);
    assert!(result.is_err(), "Invalid receipt should fail validation");

    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("mock") || err_msg.contains("empty kernel ID"));
}
```

---

## Creating New Test Receipts

When creating new test receipts:

1. **Follow schema v1.0.0**: Use `cpu_positive_example.json` as template
2. **Test one violation at a time**: For negative examples, isolate specific validation failures
3. **Use realistic data**: Model paths, performance metrics, and kernel names should be plausible
4. **Document purpose**: Add comments in this README explaining what the receipt tests
5. **Validate manually**: Run `cargo run -p xtask -- verify-receipt --path <receipt.json>` before committing

### Common Test Scenarios

**Positive cases**:
- ✅ Valid CPU receipt (this file)
- ✅ Valid GPU receipt (see `tests/fixtures/receipts/valid-gpu-receipt.json`)
- ✅ Receipt with parity validation (C++ reference comparison)
- ✅ Receipt with deterministic inference

**Negative cases**:
- ❌ Mock compute path (this file)
- ❌ Empty kernel array
- ❌ Empty kernel ID string (this file)
- ❌ Excessive kernel ID length (>128 chars)
- ❌ GPU backend with CPU kernels only
- ❌ Failed tests (this file)
- ❌ Invalid schema version
- ❌ Missing required fields

---

---

## CI Integration

### Automated Receipt Verification Workflow

The receipt verification workflow (`.github/workflows/verify-receipts.yml`) runs automatically on PRs and pushes to main/develop. It performs three levels of testing:

#### 1. Test Positive Example (Should Pass)
```bash
cargo run -p xtask -- verify-receipt \
  --path docs/tdd/receipts/cpu_positive_example.json
```

**Expected**: Exit code 0 (success)
**Validates**: Schema compliance, real compute path, valid kernel IDs

#### 2. Test Negative Example (Should Fail)
```bash
cargo run -p xtask -- verify-receipt \
  --path docs/tdd/receipts/cpu_negative_example.json
```

**Expected**: Non-zero exit code (failure)
**Validates**: Proper rejection of invalid receipts (mock compute path, empty kernels)

#### 3. Verify Generated Receipt (Benchmark Output)
```bash
# Generate receipt
cargo run -p xtask -- benchmark \
  --model models/model.gguf \
  --tokens 8 \
  --json ci/inference.json

# Verify receipt
cargo run -p xtask -- verify-receipt \
  --path ci/inference.json
```

**Expected**: Exit code 0 (success)
**Validates**: Real benchmark receipts pass verification gates

### Workflow Triggers

The workflow runs when:
- PR or push affects `crates/**`, `xtask/**`, `benchmarks/**`, or workflow files
- Changes to receipt examples (`docs/tdd/receipts/**`)
- Manual workflow dispatch

### Failure Scenarios

The workflow **fails** the build if:
1. Positive example does not pass verification
2. Negative example passes verification (should fail)
3. Generated receipt from benchmark is invalid

This ensures the verification logic correctly distinguishes valid from invalid receipts.

---

## See Also

- **Receipt schema**: `crates/bitnet-inference/src/receipts.rs`
- **Verification logic**: `xtask/src/main.rs` (`verify_receipt_cmd()`)
- **Fixture tests**: `xtask/tests/verify_receipt.rs`
- **CI workflow**: `.github/workflows/verify-receipts.yml`
- **CI integration docs**: `docs/development/ci-integration.md`
- **AC9 specification**: `docs/explanation/issue-254-real-inference-spec.md`
- **GPU validation**: `docs/explanation/issue-439-spec.md#receipt-validation`
