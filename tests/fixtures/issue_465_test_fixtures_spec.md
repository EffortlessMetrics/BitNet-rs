# Issue #465 Test Fixtures Specification

## Overview

This document specifies the test fixtures required for Issue #465 test scaffolding validation.

## Required Fixtures

### 1. CPU Baseline Receipt Fixtures

**Location:** `tests/fixtures/issue_465/cpu_baseline_valid.json`

**Purpose:** Valid CPU baseline receipt for AC3/AC4 validation

**Schema:**
```json
{
  "version": "1.0.0",
  "compute_path": "real",
  "kernels": [
    "i2s_cpu_quantized_matmul",
    "tl1_lut_dequant_forward",
    "attention_kv_cache_update",
    "layernorm_forward",
    "ffn_forward"
  ],
  "performance": {
    "tokens_per_sec": 15.3,
    "latency_ms": 65.4,
    "total_tokens": 128
  },
  "success": true,
  "model": {
    "name": "microsoft/bitnet-b1.58-2B-4T-gguf",
    "file": "ggml-model-i2_s.gguf",
    "size_bytes": 2147483648
  },
  "environment": {
    "deterministic": true,
    "seed": 42,
    "num_threads": 1
  }
}
```

### 2. Mocked Receipt Fixtures

**Location:** `tests/fixtures/issue_465/mocked_receipt.json`

**Purpose:** Invalid receipt for AC6 smoke test (should fail verification)

**Schema:**
```json
{
  "version": "1.0.0",
  "compute_path": "mocked",
  "kernels": [],
  "performance": {
    "tokens_per_sec": 15.3
  },
  "success": true
}
```

### 3. Empty Kernels Receipt

**Location:** `tests/fixtures/issue_465/empty_kernels_receipt.json`

**Purpose:** Invalid receipt with empty kernels array (should fail verification)

**Schema:**
```json
{
  "version": "1.0.0",
  "compute_path": "real",
  "kernels": [],
  "performance": {
    "tokens_per_sec": 15.3
  },
  "success": true
}
```

### 4. Invalid Kernels Receipt

**Location:** `tests/fixtures/issue_465/invalid_kernels_receipt.json`

**Purpose:** Receipt with invalid kernel IDs (should fail hygiene checks)

**Schema:**
```json
{
  "version": "1.0.0",
  "compute_path": "real",
  "kernels": [
    "",
    "a_very_long_kernel_id_that_exceeds_128_characters_limit_and_should_fail_validation_with_proper_error_message_explaining_what_went_wrong_in_detail"
  ],
  "performance": {
    "tokens_per_sec": 15.3
  },
  "success": true
}
```

### 5. README Test Fixtures

**Location:** `tests/fixtures/issue_465/readme_templates/`

**Purpose:** Template README sections for AC1/AC2 validation

**Files:**
- `quickstart_section.md`: 10-line CPU quickstart template
- `receipts_section.md`: Receipts documentation template
- `environment_vars_table.md`: Environment variables reference

### 6. Documentation Command Audit Fixtures

**Location:** `tests/fixtures/issue_465/doc_audit/`

**Purpose:** Test data for AC9 feature flag standardization

**Files:**
- `legacy_commands.txt`: List of legacy commands to be replaced
- `standardized_commands.txt`: Replacement commands with feature flags

## Test Data Requirements

### Deterministic Environment

All test fixtures must be generated with:
```bash
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1
```

### Neural Network Context

- **I2_S Quantization:** CPU baseline must reference I2_S quantized kernels
- **Kernel IDs:** Must use realistic kernel naming patterns (`i2s_cpu_*`, `tl1_*`, `tl2_*`)
- **Performance:** CPU baseline should show 10-20 tok/s (realistic for 2B model)

### Validation Rules

1. **Receipt Schema v1.0.0:** All receipts must conform to schema
2. **Kernel Hygiene:**
   - No empty strings in kernels array
   - Kernel IDs ≤ 128 characters each
   - Total kernel count ≤ 10,000
3. **Compute Path:** Must be "real" for honest compute (except mocked test fixtures)
4. **Performance Metrics:** Must be positive numbers

## Fixture Builder Tasks

The fixture-builder agent should:

1. Generate all receipt fixtures with proper schema compliance
2. Create README template fixtures for documentation validation
3. Generate test data for command audit (AC9)
4. Validate all fixtures against receipt schema
5. Include test helper data for GitHub API mocking (AC5, AC7, AC8, AC12)

## Integration Points

### Test Files Using Fixtures

- `issue_465_documentation_tests.rs`: Uses README templates
- `issue_465_baseline_tests.rs`: Uses CPU baseline receipts
- `issue_465_ci_gates_tests.rs`: Uses mocked/invalid receipts
- `issue_465_release_qa_tests.rs`: Uses GitHub API mock data

### Fixture Loading Pattern

```rust
use std::fs;
use std::path::PathBuf;

fn load_fixture(name: &str) -> String {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/issue_465")
        .join(name);

    fs::read_to_string(&fixture_path)
        .expect(&format!("Failed to load fixture: {}", name))
}
```

## Validation

All fixtures should pass:
```bash
cargo test -p bitnet-tests --test issue_465_* --no-run
```

## References

- Issue #465 Specification: `docs/explanation/issue-465-implementation-spec.md`
- Receipt Schema: `crates/bitnet-xtask/src/receipt.rs`
- BitNet-rs Testing Patterns: `tests/lib.rs`
