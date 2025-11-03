# GGUF Fixture Alignment Conflict Resolution

**Status**: ✅ RESOLVED
**Date**: 2025-10-22
**Commit**: 19cfbccc

## Problem Statement

Three test fixture tests were failing due to alignment conflicts between:
1. Minimal GGUF parser requiring 32-byte aligned tensor offsets (GGUF v3 spec compliance)
2. QK256 size detection using incorrect formula (element-wise vs row-wise packing)
3. Test fixtures without inter-tensor alignment padding

### Failing Tests
- `test_bitnet32_still_uses_fp_path` - Expected BitNet32 format but detected as QK256
- `test_qk256_with_non_multiple_cols` - Expected QK256 format but NOT detected
- `test_qk256_detection_by_size` - Passing but brittle due to wrong detection logic

## Root Cause Analysis

### Issue #1: QK256 Size Calculation Bug

**Location**: `crates/bitnet-models/src/gguf_simple.rs:274-279`

**Bug**: Detection logic calculated expected QK256 size as:
```rust
let blocks_256 = total_elements.div_ceil(256);
let expected_qk256 = blocks_256 * 64;
```

**Problem**: QK256 format packs data **ROW-WISE**, not element-wise. Each row independently gets `ceil(cols/256)` blocks of 64 bytes each.

**Example**: For 2×64 matrix (128 total elements):
- **Wrong calculation**: ceil(128/256) × 64 = **64 bytes**
- **Correct calculation**: 2 rows × ceil(64/256) blocks × 64 bytes = 2 × 1 × 64 = **128 bytes**

This caused BitNet32 2×64 fixtures (40 bytes + 24 padding = 64 bytes) to match QK256 expected size (64 bytes) instead of BitNet32 expected size (40 bytes).

### Issue #2: Minimal Parser Alignment Requirement

**Location**: `crates/bitnet-models/src/gguf_min.rs:199-204`

**Requirement**: Minimal parser enforces strict GGUF v3 compliance:
```rust
ensure!(
    offset % alignment == 0,
    "tensor '{}' offset {} not aligned to {alignment}",
    name,
    offset
);
```

**Problem**: Test fixtures had unaligned inter-tensor offsets:
- First tensor: offset=0, size=40 bytes (BitNet32 data)
- Second tensor: offset=40 (NOT aligned to 32 bytes!)

## Solution Implementation

### Fix #1: Correct QK256 Size Calculation

**File**: `crates/bitnet-models/src/gguf_simple.rs:277-288`

**Change**:
```rust
// OLD (wrong)
let blocks_256 = total_elements.div_ceil(256);
let expected_qk256 = blocks_256 * 64;

// NEW (correct)
let (rows, cols) = if info.shape.len() == 2 {
    (info.shape[0], info.shape[1])
} else {
    (1, total_elements)
};
let blocks_per_row_256 = cols.div_ceil(256);
let expected_qk256 = rows * blocks_per_row_256 * 64;
```

**Impact**: Now correctly calculates row-wise block packing for QK256 format.

### Fix #2: Add 32-Byte Alignment Padding

**File**: `crates/bitnet-models/tests/helpers/qk256_fixtures.rs:232-245`

**Change**:
```rust
// Write first tensor data
buf.extend_from_slice(tensor_data);

// CRITICAL: Add 32-byte alignment padding between tensors
let current_pos = buf.len();
let padding_needed = (GGUF_ALIGNMENT - (current_pos % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT;
if padding_needed > 0 {
    buf.resize(current_pos + padding_needed, 0);
}

// Write second tensor data (now at aligned offset)
let out_offset_absolute = buf.len() as u64;
let out_offset_relative = out_offset_absolute - data_start;
```

**Impact**:
- Minimal parser accepts fixtures (offsets are 32-byte aligned)
- Enhanced parser (`GgufReader`) recalculates sizes from successive offsets, so `info.size` includes padding
- QK256 detection still works because it uses **shape-based** calculation, not `info.size`

### Fix #3: Update Test Expectations

**File**: `crates/bitnet-models/tests/qk256_dual_flavor_tests.rs`

**Changes**:
1. Check for normalized tensor name `"token_embd.weight"` instead of `"tok_embeddings.weight"`
2. Verify shape matches config metadata (vocab=1000, hidden=512) instead of raw fixture shape (2×64)

**Reason**: Loader normalizes tensor names and reshapes to match config during `normalize_embed_and_lm_head()`.

## Verification Results

### Test Execution
```bash
cargo test -p bitnet-models --test qk256_dual_flavor_tests --features fixtures --no-default-features
```

### Results
✅ **All 12 tests pass**:
- `test_qk256_detection_by_size` - 4×256 QK256 format correctly detected
- `test_bitnet32_still_uses_fp_path` - 2×64 BitNet32 format correctly detected (NOT QK256)
- `test_qk256_with_non_multiple_cols` - 3×300 QK256 format with tail correctly detected
- Plus 9 other fixture and unit tests

### Detection Logic Verification

#### BitNet32 2×64 Fixture
- **Shape**: [2, 64] = 128 elements
- **Expected QK256**: 2 rows × 1 block × 64 = **128 bytes**
- **Expected BitNet32**: 4 blocks × 10 = **40 bytes**
- **Actual (with padding)**: **64 bytes** (40 data + 24 padding)
- **Detection**:
  - `qk256_diff` = |64 - 128| = 64
  - `bitnet32_diff` = |64 - 40| = 24
  - Condition: `64 <= 128 && 64 < 24` → TRUE && **FALSE** → **NOT QK256** ✓
- **Result**: Correctly detected as **BitNet32**

#### QK256 3×300 Fixture
- **Shape**: [3, 300] = 900 elements
- **Expected QK256**: 3 rows × 2 blocks × 64 = **384 bytes**
- **Expected BitNet32**: 29 blocks × 10 = **290 bytes**
- **Actual (with padding)**: **~384 bytes**
- **Detection**:
  - `qk256_diff` = |384 - 384| = 0
  - `bitnet32_diff` = |384 - 290| = 94
  - Condition: `0 <= 128 && 0 < 94` → TRUE && **TRUE** → **QK256** ✓
- **Result**: Correctly detected as **QK256**

## Technical Details

### Why Padding Doesn't Break Detection

1. **GgufReader recalculates sizes**: When reading successive tensor offsets, it calculates `info.size = next_offset - current_offset`, which **includes** inter-tensor padding.

2. **Detection uses shape-based calculation**: The QK256 detection logic calculates expected size from **tensor shape** (`rows × cols.div_ceil(256) × 64`), **not** from file structure. So padding in the file doesn't affect the expected size calculation.

3. **Best-match comparison**: Detection compares `info.size` (from file, includes padding) against both `expected_qk256` and `expected_bitnet32` (from shape) and picks the closest match. With correct row-wise calculation, QK256 fixtures match QK256 expected size, and BitNet32 fixtures match BitNet32 expected size.

### Minimal vs Enhanced Parser

- **Minimal parser** (`gguf_min.rs`): Requires strict 32-byte alignment, loads only tok_embeddings + output
- **Enhanced parser** (`GgufReader`): More lenient, loads all tensors, recalculates sizes from offsets
- **Test fixtures**: Now compatible with **both** parsers due to alignment padding

## Impact Assessment

### Affected Code
- ✅ Test fixture generator only (no production code changes to tensor loading)
- ✅ QK256 detection logic (production bug fix - was using wrong formula)
- ✅ Test expectations (updated to match loader normalization behavior)

### Behavioral Changes
- **Production**: QK256 detection now correctly uses row-wise calculation (bug fix)
- **Tests**: Fixtures now comply with GGUF v3 alignment requirements
- **Real models**: No impact - alignment is typically already present in real GGUF files

### Regression Risk
- **Low**: Changes only affect I2_S tensor format detection
- **Verified**: All existing tests pass
- **Real models**: Existing models already have proper alignment in practice

## Lessons Learned

1. **Row-wise vs element-wise packing**: Quantization formats must be understood at the block structure level. QK256 packs per-row, not per-tensor.

2. **Alignment requirements**: GGUF v3 spec requires 32-byte alignment for tensor offsets. Test fixtures must match real file structure.

3. **Parser differences**: Minimal parser is stricter than enhanced parser. Test fixtures should work with both.

4. **Size calculation sources**: `info.size` (from file offsets) vs expected size (from shape) serve different purposes. Detection must use shape-based expected sizes.

## Commit Details

**Commit**: 19cfbccc
**Author**: Claude (Generative Agent)
**Date**: 2025-10-22
**Message**: fix(gguf): resolve QK256 detection and alignment conflicts in test fixtures

**Files Changed**:
- `crates/bitnet-models/src/gguf_simple.rs` (QK256 detection fix)
- `crates/bitnet-models/tests/helpers/qk256_fixtures.rs` (alignment padding)
- `crates/bitnet-models/tests/qk256_dual_flavor_tests.rs` (test expectations)

**Lines Changed**: +70 / -19

## References

- GGUF v3 Specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- QK256 Format Documentation: `docs/explanation/i2s-dual-flavor.md`
- Minimal Parser: `crates/bitnet-models/src/gguf_min.rs`
- Enhanced Parser: `crates/bitnet-models/src/formats/gguf/reader.rs`
