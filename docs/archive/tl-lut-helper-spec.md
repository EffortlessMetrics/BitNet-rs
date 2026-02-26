# TL LUT Helper Specification

**Issue:** #462 - CPU Forward Pass with Real Inference (AC4)
**Status:** Specification
**Date:** 2025-10-14

## Context

BitNet-rs supports TL1/TL2 (Table Lookup) quantization for efficient 1-bit inference. TL quantization stores activation-weight products in lookup tables (LUTs) indexed by block indices and element positions. Current implementation in `quantized_linear.rs` performs inline LUT indexing without bounds checking, creating potential for index out-of-bounds errors.

**Current State:**
- TL1/TL2 matmul paths exist in `quantized_linear.rs` (lines ~600-800)
- Inline LUT indexing: `lut_offset = block_idx * block_bytes + (elem_in_block / elems_per_block)`
- No bounds checking on computed indices
- TL tests disabled with `#[ignore]` due to index safety concerns

**Problem:**
- Runtime panics from out-of-bounds LUT access
- Difficult to debug indexing errors (no error context)
- Code duplication across TL1/TL2 paths

**Required Solution:**
- Centralized safe LUT index helper with bounds checking
- Clear error messages for debugging
- Reusable across TL1/TL2 implementations
- Re-enable TL tests after integration

## Design

### LUT Indexing Algorithm

**TL Quantization Structure:**
```
Weights: [out_features, in_features] quantized to 1-bit
Activations: [batch_size, in_features] quantized to blocks
LUT: Precomputed activation-weight products per block

Block Structure:
- block_bytes: Physical bytes per block (e.g., 16 bytes)
- elems_per_block: Logical elements per block (e.g., 128 elements)
- Packing ratio: 8 elements per byte (1-bit quantization)

LUT Index Calculation:
  lut_index = block_idx * block_bytes + (elem_in_block / elems_per_pack)
  where:
    - block_idx: Block number in input sequence
    - elem_in_block: Element offset within block (0..elems_per_block)
    - elems_per_pack: Elements per packed byte (8 for 1-bit)
```

**Bounds Requirements:**
1. `elem_in_block < elems_per_block` (element within block)
2. `computed_index < lut_size` (index within LUT bounds)

### API Design

#### Public Interface

```rust
/// Safe LUT index calculation with bounds checking
///
/// # Arguments
/// * `block_idx` - Block index in input sequence (0-indexed)
/// * `elem_in_block` - Element offset within block (0-indexed)
/// * `block_bytes` - Physical bytes per block (e.g., 16)
/// * `elems_per_block` - Logical elements per block (e.g., 128)
///
/// # Returns
/// LUT index for accessing precomputed product table
///
/// # Errors
/// - `LutIndexError::OutOfBounds`: elem_in_block >= elems_per_block
/// - `LutIndexError::IndexOverflow`: Computed index would overflow usize
///
/// # Example
/// ```
/// use bitnet_kernels::tl_lut::lut_index;
///
/// // TL1 configuration: 16-byte blocks, 128 elements per block
/// let idx = lut_index(0, 10, 16, 128)?; // Block 0, element 10
/// assert_eq!(idx, 1); // byte offset: 10 / 8 = 1
///
/// // TL2 configuration: different packing
/// let idx = lut_index(5, 64, 16, 128)?; // Block 5, element 64
/// assert_eq!(idx, 88); // 5*16 + 64/8 = 80 + 8 = 88
/// # Ok::<(), bitnet_kernels::tl_lut::LutIndexError>(())
/// ```
///
/// # Safety
/// - Validates elem_in_block < elems_per_block before computation
/// - Uses checked arithmetic to prevent overflow
/// - Returns descriptive errors for debugging
pub fn lut_index(
    block_idx: usize,
    elem_in_block: usize,
    block_bytes: usize,
    elems_per_block: usize,
) -> Result<usize, LutIndexError>;
```

#### Error Types

```rust
/// LUT indexing errors
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LutIndexError {
    /// Element index exceeds elements per block
    #[error("Element index {0} exceeds elements per block {1}")]
    OutOfBounds(usize, usize),

    /// Computed LUT index exceeds maximum size
    #[error("Computed LUT index {0} exceeds LUT size {1}")]
    IndexOverflow(usize, usize),

    /// Block index computation overflow
    #[error("Block index {block_idx} * block bytes {block_bytes} overflows usize")]
    BlockOffsetOverflow {
        block_idx: usize,
        block_bytes: usize,
    },

    /// Invalid configuration (zero block size)
    #[error("Invalid configuration: block_bytes={block_bytes}, elems_per_block={elems_per_block}")]
    InvalidConfig {
        block_bytes: usize,
        elems_per_block: usize,
    },
}
```

### Implementation

#### Core Algorithm

```rust
pub fn lut_index(
    block_idx: usize,
    elem_in_block: usize,
    block_bytes: usize,
    elems_per_block: usize,
) -> Result<usize, LutIndexError> {
    // Validate configuration
    if block_bytes == 0 || elems_per_block == 0 {
        return Err(LutIndexError::InvalidConfig {
            block_bytes,
            elems_per_block,
        });
    }

    // Bounds check: element must be within block
    if elem_in_block >= elems_per_block {
        return Err(LutIndexError::OutOfBounds(elem_in_block, elems_per_block));
    }

    // Compute block offset with overflow check
    let block_offset = block_idx.checked_mul(block_bytes)
        .ok_or(LutIndexError::BlockOffsetOverflow {
            block_idx,
            block_bytes,
        })?;

    // Compute element offset within block
    // Packing: 8 elements per byte for 1-bit quantization
    const ELEMS_PER_BYTE: usize = 8;
    let elem_offset = elem_in_block / ELEMS_PER_BYTE;

    // Compute final LUT index with overflow check
    let lut_idx = block_offset.checked_add(elem_offset)
        .ok_or(LutIndexError::IndexOverflow(
            usize::MAX, // Overflowed value
            usize::MAX, // Max LUT size
        ))?;

    Ok(lut_idx)
}
```

#### Optional: Runtime LUT Size Validation

```rust
/// Validate LUT index against actual LUT size (optional safety layer)
///
/// # Example
/// ```
/// use bitnet_kernels::tl_lut::{lut_index, validate_lut_index};
///
/// let idx = lut_index(5, 64, 16, 128)?;
/// validate_lut_index(idx, 1024)?; // Ensure idx < 1024
/// # Ok::<(), bitnet_kernels::tl_lut::LutIndexError>(())
/// ```
pub fn validate_lut_index(index: usize, lut_size: usize) -> Result<(), LutIndexError> {
    if index >= lut_size {
        Err(LutIndexError::IndexOverflow(index, lut_size))
    } else {
        Ok(())
    }
}
```

### Integration Points

#### Call Sites in `quantized_linear.rs`

**TL1 Matmul (lines ~600-700):**
```rust
async fn quantized_matmul_tl1(
    &self,
    input: &candle_core::Tensor,
    provider: &dyn bitnet_kernels::KernelProvider,
) -> Result<candle_core::Tensor> {
    use bitnet_kernels::tl_lut::lut_index;

    // ... setup code ...

    for block_idx in 0..num_blocks {
        for elem_in_block in 0..elems_per_block {
            // OLD: Inline indexing (no bounds check)
            // let lut_idx = block_idx * block_bytes + (elem_in_block / 8);

            // NEW: Safe LUT helper
            let lut_idx = lut_index(block_idx, elem_in_block, block_bytes, elems_per_block)
                .context("TL1 LUT indexing failed")?;

            // Use lut_idx to access precomputed products
            let product = lut[lut_idx];
            // ... accumulation ...
        }
    }

    // ... finalization ...
}
```

**TL2 Matmul (lines ~700-800):**
```rust
async fn quantized_matmul_tl2(
    &self,
    input: &candle_core::Tensor,
    provider: &dyn bitnet_kernels::KernelProvider,
) -> Result<candle_core::Tensor> {
    use bitnet_kernels::tl_lut::lut_index;

    // ... setup code ...

    for block_idx in 0..num_blocks {
        for elem_in_block in 0..elems_per_block {
            // NEW: Safe LUT helper
            let lut_idx = lut_index(block_idx, elem_in_block, block_bytes, elems_per_block)
                .context("TL2 LUT indexing failed")?;

            let product = lut[lut_idx];
            // ... accumulation ...
        }
    }

    // ... finalization ...
}
```

#### Module Structure

**New File:** `crates/bitnet-kernels/src/tl_lut.rs`

```rust
//! Table Lookup (TL) quantization utilities
//!
//! Provides safe LUT indexing for TL1/TL2 quantized matmul operations.

mod error;
mod index;

pub use error::LutIndexError;
pub use index::{lut_index, validate_lut_index};

#[cfg(test)]
mod tests;
```

**Update:** `crates/bitnet-kernels/src/lib.rs`

```rust
pub mod tl_lut; // Add new module

// Re-export for convenience
pub use tl_lut::{lut_index, LutIndexError};
```

## Validation

### Unit Tests

#### Bounds Checking Tests

```rust
// AC4: Valid LUT index calculation
#[test]
fn test_ac4_tl_lut_index_bounds_valid() {
    use bitnet_kernels::tl_lut::lut_index;

    // TL1: 16-byte blocks, 128 elements
    assert_eq!(lut_index(0, 0, 16, 128).unwrap(), 0);
    assert_eq!(lut_index(0, 8, 16, 128).unwrap(), 1); // 8/8 = 1 byte
    assert_eq!(lut_index(1, 0, 16, 128).unwrap(), 16); // Block 1 offset
    assert_eq!(lut_index(5, 64, 16, 128).unwrap(), 88); // 5*16 + 64/8 = 88

    // TL2: Different configuration
    assert_eq!(lut_index(0, 0, 32, 256).unwrap(), 0);
    assert_eq!(lut_index(2, 16, 32, 256).unwrap(), 66); // 2*32 + 16/8 = 66
}

// AC4: Out-of-bounds element index
#[test]
fn test_ac4_tl_lut_index_bounds_invalid() {
    use bitnet_kernels::tl_lut::{lut_index, LutIndexError};

    // elem_in_block >= elems_per_block
    let result = lut_index(0, 128, 16, 128);
    assert!(matches!(result, Err(LutIndexError::OutOfBounds(128, 128))));

    let result = lut_index(1, 200, 16, 128);
    assert!(matches!(result, Err(LutIndexError::OutOfBounds(200, 128))));
}

// AC4: Overflow detection
#[test]
fn test_ac4_tl_lut_index_overflow() {
    use bitnet_kernels::tl_lut::{lut_index, LutIndexError};

    // Block offset overflow: usize::MAX * 16
    let result = lut_index(usize::MAX, 0, 16, 128);
    assert!(matches!(result, Err(LutIndexError::BlockOffsetOverflow { .. })));
}

// AC4: Invalid configuration
#[test]
fn test_ac4_tl_lut_index_invalid_config() {
    use bitnet_kernels::tl_lut::{lut_index, LutIndexError};

    // Zero block bytes
    let result = lut_index(0, 0, 0, 128);
    assert!(matches!(result, Err(LutIndexError::InvalidConfig { .. })));

    // Zero elements per block
    let result = lut_index(0, 0, 16, 0);
    assert!(matches!(result, Err(LutIndexError::InvalidConfig { .. })));
}
```

#### Integration Tests

```rust
// AC4: TL1 matmul with safe LUT indexing
#[test]
fn test_ac4_tl1_matmul_with_safe_lut() {
    use bitnet_inference::layers::QuantizedLinear;
    use bitnet_common::QuantizationType;

    let qlinear = create_test_quantized_linear(QuantizationType::TL1)?;
    let input = create_test_input(1, 128)?; // [1, 128]

    // Should not panic with bounds checking
    let output = qlinear.forward(&input).await?;

    assert_eq!(output.shape(), &[1, 256]); // [1, out_features]
}

// AC4: TL2 matmul with safe LUT indexing
#[test]
fn test_ac4_tl2_matmul_with_safe_lut() {
    use bitnet_inference::layers::QuantizedLinear;
    use bitnet_common::QuantizationType;

    let qlinear = create_test_quantized_linear(QuantizationType::TL2)?;
    let input = create_test_input(1, 128)?;

    let output = qlinear.forward(&input).await?;

    assert_eq!(output.shape(), &[1, 256]);
}

// AC4: Re-enable TL tests (remove #[ignore])
#[test] // Remove #[ignore] after integration
fn test_tl1_quantized_linear_forward() {
    // Previously ignored test now runs with safe LUT helper
    let result = run_tl1_forward_pass()?;
    assert!(result.is_ok());
}

#[test] // Remove #[ignore] after integration
fn test_tl2_quantized_linear_forward() {
    let result = run_tl2_forward_pass()?;
    assert!(result.is_ok());
}
```

### Error Message Validation

```rust
// AC4: Descriptive error messages
#[test]
fn test_ac4_lut_error_messages() {
    use bitnet_kernels::tl_lut::lut_index;

    // Out of bounds error
    let err = lut_index(0, 150, 16, 128).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Element index 150 exceeds elements per block 128"
    );

    // Invalid config error
    let err = lut_index(0, 0, 0, 128).unwrap_err();
    assert!(err.to_string().contains("Invalid configuration"));

    // Overflow error
    let err = lut_index(usize::MAX, 0, 16, 128).unwrap_err();
    assert!(err.to_string().contains("overflows usize"));
}
```

### Performance Tests

```rust
// AC4: LUT helper performance (should be negligible overhead)
#[bench]
fn bench_lut_index_helper(b: &mut Bencher) {
    b.iter(|| {
        for block_idx in 0..1000 {
            for elem_in_block in (0..128).step_by(8) {
                black_box(lut_index(block_idx, elem_in_block, 16, 128).unwrap());
            }
        }
    });
}

// Baseline: inline indexing (no bounds check)
#[bench]
fn bench_lut_index_inline(b: &mut Bencher) {
    b.iter(|| {
        for block_idx in 0..1000 {
            for elem_in_block in (0..128).step_by(8) {
                black_box(block_idx * 16 + elem_in_block / 8);
            }
        }
    });
}

// Expected: <5% overhead for bounds checking
```

## Implementation Sequence

### Phase 1: Create LUT Helper Module

1. **Create new file:** `crates/bitnet-kernels/src/tl_lut.rs`
2. **Implement:**
   - `lut_index()` function with bounds checking
   - `LutIndexError` error type
   - `validate_lut_index()` optional helper
3. **Add unit tests:**
   - Valid index calculations (TL1, TL2 configs)
   - Out-of-bounds detection
   - Overflow detection
   - Invalid configuration handling
4. **Update module exports:**
   - Add `pub mod tl_lut;` to `crates/bitnet-kernels/src/lib.rs`
   - Re-export `lut_index` and `LutIndexError`

**Validation:**
```bash
cargo test -p bitnet-kernels test_ac4_tl_lut_index --features cpu
```

### Phase 2: Integrate into QuantizedLinear

1. **Update TL1 path:**
   - Import `use bitnet_kernels::tl_lut::lut_index;`
   - Replace inline indexing with `lut_index()` calls
   - Add `.context()` for error messages
2. **Update TL2 path:**
   - Same integration pattern as TL1
3. **Remove unsafe inline code:**
   - Delete old `lut_idx = block_idx * block_bytes + ...` statements

**Validation:**
```bash
cargo test -p bitnet-inference test_ac4_tl1_matmul_with_safe_lut --features cpu
cargo test -p bitnet-inference test_ac4_tl2_matmul_with_safe_lut --features cpu
```

### Phase 3: Re-enable TL Tests

1. **Find ignored TL tests:**
   ```bash
   grep -r "#\[ignore\]" crates/bitnet-inference/tests | grep -i "tl"
   ```
2. **Remove `#[ignore]` attributes:**
   - Update test files to run TL1/TL2 tests
3. **Verify all tests pass:**
   ```bash
   cargo test -p bitnet-inference --features cpu
   ```

### Phase 4: Documentation and Cleanup

1. **Add doc comments:**
   - Module-level docs in `tl_lut.rs`
   - Function-level docs with examples
2. **Update CHANGELOG.md:**
   - Add entry for AC4: Safe TL LUT indexing helper
3. **Update `docs/reference/quantization-support.md`:**
   - Document TL LUT indexing safety

## Performance Considerations

### Overhead Analysis

**Bounds Checking Cost:**
- 2 comparisons: `elem_in_block < elems_per_block`, `config validation`
- 1 checked multiply: `block_idx * block_bytes`
- 1 checked add: `block_offset + elem_offset`

**Typical TL Matmul:**
- Outer loop: `num_blocks` iterations (e.g., 128 blocks)
- Inner loop: `elems_per_block` iterations (e.g., 128 elements)
- Total LUT lookups: `128 * 128 = 16,384` per layer

**Expected Overhead:**
- Modern CPUs: Branch prediction eliminates bounds check cost
- Checked arithmetic: Same as unchecked for typical values
- Overall: <1% overhead vs. unsafe inline indexing

**Optimization Opportunities:**
1. **Compiler inlining:** Mark `lut_index()` as `#[inline(always)]`
2. **Const generics:** Specialize for common block sizes
3. **SIMD batching:** Process multiple indices in parallel

### Unsafe Optimization (Future)

```rust
/// Unsafe unchecked LUT indexing for performance-critical paths
///
/// # Safety
/// Caller must ensure:
/// - elem_in_block < elems_per_block
/// - computed index < lut.len()
///
/// # Example
/// ```
/// use bitnet_kernels::tl_lut::lut_index_unchecked;
///
/// // Safe: bounds pre-validated
/// let idx = unsafe { lut_index_unchecked(0, 10, 16, 128) };
/// ```
#[inline(always)]
pub unsafe fn lut_index_unchecked(
    block_idx: usize,
    elem_in_block: usize,
    block_bytes: usize,
    _elems_per_block: usize, // Unused in unchecked version
) -> usize {
    block_idx * block_bytes + (elem_in_block / 8)
}
```

**Usage Pattern:**
```rust
// Debug builds: Use safe version
#[cfg(debug_assertions)]
let lut_idx = lut_index(block_idx, elem_in_block, block_bytes, elems_per_block)?;

// Release builds: Use unsafe version (if bounds pre-validated)
#[cfg(not(debug_assertions))]
let lut_idx = unsafe {
    debug_assert!(elem_in_block < elems_per_block);
    lut_index_unchecked(block_idx, elem_in_block, block_bytes, elems_per_block)
};
```

## Error Handling Patterns

### Context Propagation

```rust
// In quantized_linear.rs TL1 path
for block_idx in 0..num_blocks {
    for elem_in_block in 0..elems_per_block {
        let lut_idx = lut_index(block_idx, elem_in_block, block_bytes, elems_per_block)
            .with_context(|| format!(
                "TL1 LUT indexing failed: block={}, elem={}, block_bytes={}, elems_per_block={}",
                block_idx, elem_in_block, block_bytes, elems_per_block
            ))?;

        // Use lut_idx safely
    }
}
```

### Debugging Support

```rust
// Optional: Add debug logging
#[cfg(debug_assertions)]
fn lut_index_debug(
    block_idx: usize,
    elem_in_block: usize,
    block_bytes: usize,
    elems_per_block: usize,
) -> Result<usize, LutIndexError> {
    let idx = lut_index(block_idx, elem_in_block, block_bytes, elems_per_block)?;
    tracing::trace!(
        "LUT index: block={}, elem={}, idx={}, config=({}, {})",
        block_idx, elem_in_block, idx, block_bytes, elems_per_block
    );
    Ok(idx)
}
```

## References

### Related Documentation

- `docs/reference/quantization-support.md` - TL1/TL2 quantization algorithms
- `docs/explanation/cpu-inference-architecture.md` - Forward pass integration
- `docs/development/test-suite.md` - Testing framework

### Existing Code Patterns

- `crates/bitnet-inference/src/layers/quantized_linear.rs:600-800` - TL1/TL2 matmul paths
- `crates/bitnet-kernels/src/cpu.rs` - CPU kernel provider
- `crates/bitnet-common/src/error.rs` - Error handling patterns

### Issue References

- **Issue #462 (AC4):** TL1/TL2 LUT index helper (this spec)
- **Issue #254:** Real inference specification (TL quantization background)
