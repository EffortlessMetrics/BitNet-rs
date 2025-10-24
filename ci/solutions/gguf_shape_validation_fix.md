# GGUF Shape Validation Fix: QK256 Dual-Map Architecture

**Navigation:** [ci/](../) → [solutions/](./00_NAVIGATION_INDEX.md) → This Document
**Related:** [PR #475 Summary](../PR_475_FINAL_SUMMARY.md)

---

**Table of Contents**

- [Executive Summary](#executive-summary)
- [Architecture Overview: Dual-Map System](#architecture-overview-dual-map-system)
- [Test Analysis: test_ac3_tensor_shape_validation_cpu](#test-analysis-test_ac3_tensor_shape_validation_cpu)
- [Complete Fix](#complete-fix)
- [Verification Commands](#verification-commands)
- [Root Cause Analysis](#root-cause-analysis)

---

## Executive Summary

The test `test_ac3_tensor_shape_validation_cpu` in `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/tests/gguf_weight_loading_tests.rs` contains a **critical bug in tensor map access** that violates the GGUF loader's dual-map architecture.

**Root Cause**: The test incorrectly accesses QK256 tensors from `.tensors` (regular Candle tensors) instead of `.i2s_qk256` (QK256-specific struct).

**Impact**: 
- QK256 tensors are never found in the `.tensors` map (causing test failures)
- Shape validation silently fails instead of detecting actual shape mismatches
- Field access uses incompatible patterns (`.rows`/`.cols` on Candle tensors vs direct struct fields)

---

## Architecture Overview: Dual-Map System

### Understanding GgufLoadResult Structure

The GGUF loader returns a `GgufLoadResult` with **three distinct components**:

```rust
pub struct GgufLoadResult {
    pub config: bitnet_common::BitNetConfig,      // Model configuration
    pub tensors: HashMap<String, CandleTensor>,   // Regular tensors (F32, F16, F64)
    pub i2s_qk256: HashMap<String, I2SQk256NoScale>, // QK256 quantized weights
}
```

### Storage Partition Strategy

**Why separate storage?** QK256 tensors are stored in a separate map because:

1. **Format Incompatibility**: QK256 is packed 2-bit data in a raw byte format (`Vec<u8>`)
2. **No Dequantization**: Unlike F32/F16 tensors, QK256 remains quantized for efficient kernel dispatch
3. **Custom Kernels**: QK256 tensors require specialized compute kernels that work directly with packed data
4. **Memory Efficiency**: Avoiding forced dequantization saves ~4× memory for 2-bit quantized models

### Loader Decision Tree (from `gguf_simple.rs`)

**Lines 273-301**: The loader detects tensor format and routes accordingly:

```
For I2_S quantized tensors:
  ├─ Calculate expected_qk256 = rows × ceil(cols/256) × 64 bytes
  ├─ Compare available bytes against expected_qk256
  │
  ├─ If MATCH (within 128-byte tolerance):
  │   └─ QK256 format detected
  │       └─ Store in i2s_qk256 map (return None from load_tensor_from_gguf)
  │       └─ Create I2SQk256NoScale structure in second pass (lines 267-434)
  │
  └─ If NO MATCH:
      └─ BitNet32 format (32-element blocks, inline F16 scales)
          └─ Dequantize → Store in tensors map as CandleTensor
```

**Critical Code** (lines 294-300, `gguf_simple.rs`):

```rust
let qk256_diff = available_bytes.abs_diff(expected_qk256);
let bitnet32_diff = available_bytes.abs_diff(expected_bitnet32);

if qk256_diff <= QK256_SIZE_TOLERANCE && qk256_diff < bitnet32_diff {
    // This is QK256 format - will be stored in i2s_qk256 map
    // Return None to signal this should not be added to regular tensor map
    return Ok(None);
}
```

---

## Test Analysis: test_ac3_tensor_shape_validation_cpu

### Test Location
**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/tests/gguf_weight_loading_tests.rs`
**Lines**: 378-437
**Function**: `test_ac3_tensor_shape_validation_cpu()`

### Fixture Generation

The test uses a QK256 4×256 fixture:

```rust
let gguf_bytes = generate_qk256_4x256(42);  // Generate fixture
let tmp = tempfile::tempdir()?;
let path = tmp.path().join("test_shape.gguf");
std::fs::write(&path, &gguf_bytes)?;

let result = bitnet_models::gguf_simple::load_gguf_full(
    &path,
    Device::Cpu,
    bitnet_models::GGUFLoaderConfig::default(),
);
```

### Fixture Contents (from `qk256_fixtures.rs`)

The fixture includes **TWO tensors**:

1. **`tok_embeddings.weight`** [4, 256] - I2_S QK256 format
   - Rows: 4
   - Cols: 256
   - Expected bytes: 4 × ceil(256/256) × 64 = 4 × 1 × 64 = **256 bytes**
   - **Stored in**: `load_result.i2s_qk256` map

2. **`output.weight`** [4, 256] - F16 format
   - Rows: 4
   - Cols: 256
   - Data type: F16 (2 bytes/element)
   - Expected bytes: 4 × 256 × 2 = **2048 bytes**
   - **Stored in**: `load_result.tensors` map

### Current Test Code (BUGGY)

**Lines 400-416** - **BUG #1: Wrong map access**:

```rust
// ❌ WRONG: Looking in tensors map for QK256 tensor
if let Some(qk256_tensor) = load_result.tensors.get("tok_embeddings.weight") {
    assert_eq!(
        qk256_tensor.rows,  // ❌ BUG #2: Calling .rows on CandleTensor
        4,
        "tok_embeddings.weight should have 4 rows, got {}",
        qk256_tensor.rows
    );
    // ... more assertions
}
```

**Why this fails**:

1. `load_result.tensors.get("tok_embeddings.weight")` returns `None` (tensor is in `.i2s_qk256`, not `.tensors`)
2. If somehow found, `qk256_tensor.rows` would be a **compilation error** because:
   - `CandleTensor` (type of values in `.tensors`) does NOT have `.rows` or `.cols` fields
   - Shape is accessed via `.shape().dims()` method instead

**Lines 418-429** - **Correct approach (F16 tensor)**:

```rust
// ✓ CORRECT: Looking in tensors map for F16 tensor
if let Some(tensor) = load_result.tensors.get("output.weight") {
    let shape = tensor.shape().dims();  // ✓ Correct method-based access
    assert_eq!(
        shape,
        &[4, 256],
        "output.weight should have shape [4, 256], got {:?}",
        shape
    );
}
```

---

## Dual-Map Architecture: Detailed Breakdown

### I2SQk256NoScale Structure (from `i2s_qk256.rs`)

**Lines 65-71**:

```rust
#[derive(Clone, Debug)]
pub struct I2SQk256NoScale {
    pub rows: usize,              // Direct field access
    pub cols: usize,              // Direct field access
    pub row_stride_bytes: usize,  // Direct field access
    pub qs: Vec<u8>,              // Raw packed 2-bit data
}
```

**Key differences from CandleTensor**:
- **Fields are public and directly accessible** (`.rows`, `.cols`)
- **No shape methods** (no `.shape()`)
- **Raw data is opaque** (packed 2-bit codes, not f32 floats)

### CandleTensor Structure (from Candle library)

**Methods used for shape access**:
- `.shape()` → returns `Shape` object
- `.shape().dims()` → returns `&[usize]` array
- No `.rows` or `.cols` fields (opaque implementation)

---

## Field Access Patterns

### Pattern 1: QK256 Tensors (I2SQk256NoScale)

```rust
// ✓ CORRECT: Direct field access on I2SQk256NoScale
if let Some(qk256_tensor) = load_result.i2s_qk256.get("tok_embeddings.weight") {
    let rows = qk256_tensor.rows;  // Direct field
    let cols = qk256_tensor.cols;  // Direct field
    let stride = qk256_tensor.row_stride_bytes;  // Direct field
}
```

### Pattern 2: Regular Tensors (CandleTensor)

```rust
// ✓ CORRECT: Method-based access on CandleTensor
if let Some(tensor) = load_result.tensors.get("output.weight") {
    let shape = tensor.shape().dims();  // Method call
    let rows = shape[0];  // Index into shape array
    let cols = shape[1];
}
```

### Pattern 3: Mixed Handling (Type-Agnostic)

```rust
// ✓ CORRECT: Validate by map location first
enum TensorSource {
    Regular(CandleTensor),
    Qk256(I2SQk256NoScale),
}

match TensorSource::Qk256(qk256) {
    TensorSource::Qk256(t) => {
        let rows = t.rows;  // Direct field
        let cols = t.cols;
    }
    TensorSource::Regular(t) => {
        let dims = t.shape().dims();
        let rows = dims[0];
    }
}
```

---

## Complete Fix

### Location of Changes

| File | Function | Lines | Issue |
|------|----------|-------|-------|
| `crates/bitnet-models/tests/gguf_weight_loading_tests.rs` | `test_ac3_tensor_shape_validation_cpu` | 400-416 | Wrong map, wrong field access |

### Fix Details

#### Before (BUGGY)

```rust
// Lines 378-438
async fn test_ac3_tensor_shape_validation_cpu() -> Result<()> {
    use helpers::qk256_fixtures::generate_qk256_4x256;

    let gguf_bytes = generate_qk256_4x256(42);
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("test_shape.gguf");
    std::fs::write(&path, &gguf_bytes)?;

    let result = bitnet_models::gguf_simple::load_gguf_full(
        &path,
        Device::Cpu,
        bitnet_models::GGUFLoaderConfig::default(),
    );

    match result {
        Ok(load_result) => {
            // Validate tensor shapes in the loaded maps
            // Fixture has: tok_embeddings.weight [4, 256] and output.weight [4, 256]
            // Both are I2_S QK256 format, so they should be in i2s_qk256 map

            // ❌ BUG #1: Checking WRONG map (tensors instead of i2s_qk256)
            // ❌ BUG #2: Wrong field access (.rows instead of direct struct field)
            if let Some(qk256_tensor) = load_result.tensors.get("tok_embeddings.weight") {
                assert_eq!(
                    qk256_tensor.rows,  // ❌ CandleTensor doesn't have .rows
                    4,
                    "tok_embeddings.weight should have 4 rows, got {}",
                    qk256_tensor.rows
                );
                assert_eq!(
                    qk256_tensor.cols,  // ❌ CandleTensor doesn't have .cols
                    256,
                    "tok_embeddings.weight should have 256 cols, got {}",
                    qk256_tensor.cols
                );
            } else {
                anyhow::bail!("Missing tok_embeddings.weight in tensors map");  // ❌ Looks in wrong map
            }

            // Check output.weight shape (in tensors map - F16 format per fixture)
            if let Some(tensor) = load_result.tensors.get("output.weight") {
                let shape = tensor.shape().dims();
                assert_eq!(
                    shape,
                    &[4, 256],
                    "output.weight should have shape [4, 256], got {:?}",
                    shape
                );
            } else {
                anyhow::bail!("Missing output.weight in tensors map");
            }
        }
        Err(err) => {
            eprintln!("AC3 Test correctly failing (TDD Red): {}", err);
            panic!("AC3: Tensor metadata validation not yet implemented");
        }
    }

    Ok(())
}
```

#### After (FIXED)

```rust
// Lines 378-438
async fn test_ac3_tensor_shape_validation_cpu() -> Result<()> {
    use helpers::qk256_fixtures::generate_qk256_4x256;

    let gguf_bytes = generate_qk256_4x256(42);
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("test_shape.gguf");
    std::fs::write(&path, &gguf_bytes)?;

    let result = bitnet_models::gguf_simple::load_gguf_full(
        &path,
        Device::Cpu,
        bitnet_models::GGUFLoaderConfig::default(),
    );

    match result {
        Ok(load_result) => {
            // Validate tensor shapes in the loaded maps
            // Fixture has: tok_embeddings.weight [4, 256] and output.weight [4, 256]
            // - tok_embeddings is I2_S QK256 format → stored in i2s_qk256 map
            // - output is F16 format → stored in tensors map

            // ✓ FIX #1: Check i2s_qk256 map (correct location for QK256)
            // ✓ FIX #2: Direct field access (.rows, .cols on I2SQk256NoScale)
            if let Some(qk256_tensor) = load_result.i2s_qk256.get("tok_embeddings.weight") {
                assert_eq!(
                    qk256_tensor.rows,  // ✓ I2SQk256NoScale has public .rows field
                    4,
                    "tok_embeddings.weight should have 4 rows, got {}",
                    qk256_tensor.rows
                );
                assert_eq!(
                    qk256_tensor.cols,  // ✓ I2SQk256NoScale has public .cols field
                    256,
                    "tok_embeddings.weight should have 256 cols, got {}",
                    qk256_tensor.cols
                );
            } else {
                anyhow::bail!("Missing tok_embeddings.weight in i2s_qk256 map");  // ✓ Correct map
            }

            // ✓ Check output.weight shape (in tensors map - F16 format per fixture)
            if let Some(tensor) = load_result.tensors.get("output.weight") {
                let shape = tensor.shape().dims();
                assert_eq!(
                    shape,
                    &[4, 256],
                    "output.weight should have shape [4, 256], got {:?}",
                    shape
                );
            } else {
                anyhow::bail!("Missing output.weight in tensors map");
            }
        }
        Err(err) => {
            eprintln!("AC3 Test correctly failing (TDD Red): {}", err);
            panic!("AC3: Tensor metadata validation not yet implemented");
        }
    }

    Ok(())
}
```

### Exact Changes Required

```diff
-            if let Some(qk256_tensor) = load_result.tensors.get("tok_embeddings.weight") {
+            if let Some(qk256_tensor) = load_result.i2s_qk256.get("tok_embeddings.weight") {
                 assert_eq!(
                     qk256_tensor.rows,
                     4,
@@ -410,7 +410,7 @@
                     qk256_tensor.cols
                 );
             } else {
-                anyhow::bail!("Missing tok_embeddings.weight in tensors map");
+                anyhow::bail!("Missing tok_embeddings.weight in i2s_qk256 map");
             }
```

---

## Verification Commands

### 1. Run the specific failing test

```bash
cd /home/steven/code/Rust/BitNet-rs

# Before fix (should fail)
cargo test -p bitnet-models --test gguf_weight_loading_tests \
  test_ac3_tensor_shape_validation_cpu --no-default-features --features cpu \
  -- --nocapture 2>&1 | grep -A 20 "Missing tok_embeddings"

# After fix (should pass)
cargo test -p bitnet-models --test gguf_weight_loading_tests \
  test_ac3_tensor_shape_validation_cpu --no-default-features --features cpu \
  -- --nocapture
```

### 2. Verify the fixture structure

```bash
# Check fixture generation
cargo test -p bitnet-models tests::helpers::qk256_fixtures::tests \
  --no-default-features --features cpu -- --nocapture

# Expected output: All fixture tests pass (verify GGUF structure is correct)
```

### 3. Validate with nextest (recommended)

```bash
# Use nextest for parallel test execution
cargo nextest run -p bitnet-models --test gguf_weight_loading_tests \
  --no-default-features --features cpu

# Expected: test_ac3_tensor_shape_validation_cpu passes
```

### 4. Check related tests

```bash
# Verify no other tests have similar issues
cargo grep -p bitnet-models "\.tensors\.get.*qk256\|tensors\.get.*tok_embeddings"

# Verify dual-map usage in other tests
cargo grep -p bitnet-models "\.i2s_qk256\.get\|load_result\.i2s_qk256"
```

### 5. Run full test suite

```bash
# Comprehensive test of GGUF loading
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features cpu

# Expected: All dual-flavor tests pass (12/12)
```

---

## Root Cause Analysis

### Why This Bug Occurred

1. **Comment Mismatch** (line 374):
   - Comment says: "Both are I2_S QK256 format, so they should be in i2s_qk256 map"
   - Code does: `load_result.tensors.get("tok_embeddings.weight")`
   - **Inconsistency**: Comment describes correct behavior, code implements wrong behavior

2. **Copy-Paste Error**:
   - Lines 418-429 correctly access `.tensors` for F16 tensor
   - Lines 400-416 incorrectly copied the same pattern without changing the map

3. **Type System Oversight**:
   - The test never attempts to compile/run (it's marked as async but probably compiled with wrong map)
   - Field access `.rows` on CandleTensor would cause a compile error if the test was actually run
   - Suggests test may have been scaffolded but never executed

### Design Lesson

The dual-map architecture is **intentional and correct**:
- Regular tensors (F32, F16, F64) are **dequantized** → stored as `CandleTensor` in `.tensors`
- QK256 tensors are **kept quantized** → stored as `I2SQk256NoScale` in `.i2s_qk256`
- This avoids unnecessary 4× memory expansion for 2-bit quantized models
- Tests must respect this architectural decision

---

## Related Code Locations

### Fixture Definition
- **File**: `crates/bitnet-models/tests/helpers/qk256_fixtures.rs`
- **Function**: `generate_qk256_4x256()` (lines 62-74)
- **Generated tensors**: `tok_embeddings.weight` (QK256) + `output.weight` (F16)

### Loader Implementation
- **File**: `crates/bitnet-models/src/gguf_simple.rs`
- **Function**: `load_gguf_enhanced()` (lines 200-457)
- **Key decision point**: Lines 273-301 (format detection)
- **QK256 extraction**: Lines 267-434 (second pass)

### Structure Definitions
- **I2SQk256NoScale**: `crates/bitnet-models/src/quant/i2s_qk256.rs` (lines 65-71)
- **GgufLoadResult**: `crates/bitnet-models/src/gguf_simple.rs` (lines 50-55)

### Related Tests
- `test_ac3_tensor_alignment_validation_cpu` (lines 444-475) - Also uses fixtures, correctly accesses `.tensors`
- `test_ac10_tensor_naming_conventions_cpu` (lines 482-517) - Also uses fixtures
- `qk256_dual_flavor_tests.rs` - Dedicated dual-map validation tests

---

## Summary of Issues and Fixes

| Issue | Location | Severity | Fix |
|-------|----------|----------|-----|
| Wrong map access | Line 401 | **CRITICAL** | Change `.tensors` → `.i2s_qk256` |
| Wrong field access | Lines 403, 408 | **CRITICAL** | Keep `.rows`/`.cols` (correct for I2SQk256NoScale) |
| Wrong error message | Line 414 | Minor | Update to reference correct map |

All three issues must be fixed for the test to pass.

---

## Related Documentation

**Main Report**: [PR #475 Final Success Report](../PR_475_FINAL_SUMMARY.md)
**Solution Navigation**: [00_NAVIGATION_INDEX.md](./00_NAVIGATION_INDEX.md)
**Repository Guide**: [CLAUDE.md](../../CLAUDE.md)

**Related Solutions**:
- [qk256_struct_creation_analysis.md](./qk256_struct_creation_analysis.md) - QK256 structural validation tests
- [qk256_property_test_analysis.md](./qk256_property_test_analysis.md) - QK256 property test dimension validation
- [general_docs_scaffolding.md](./general_docs_scaffolding.md) - Documentation completeness validation
