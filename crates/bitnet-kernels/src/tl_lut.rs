//! Table Lookup (TL) LUT index calculation helper
//!
//! This module provides safe bounds-checked LUT index calculation for TL1/TL2 quantization kernels.
//!
//! # Bit-Packing Layout
//! TL quantization uses bit-packed storage where **8 elements fit in 1 byte**.
//! The LUT backing buffer is byte-addressed, but elements are sub-byte precision.
//!
//! # Index Formula
//! ```text
//! lut_index = block_idx * block_bytes + (elem_in_block / 8)
//! ```
//!
//! The division by 8 maps element indices to byte offsets in the bit-packed LUT.
//!
//! # Safety Guarantees
//! - Validates elems_per_block > 0 (non-empty blocks)
//! - Validates elem_in_block < elems_per_block (bounds check)
//! - Uses checked arithmetic to prevent overflow
//! - Validates final index < lut_len

use bitnet_common::{BitNetError, KernelError, Result};

/// Calculate LUT index with bounds checking for TL quantization kernels
///
/// This function handles bit-packed TL quantization where 8 elements are stored per byte.
/// The element index is divided by 8 to obtain the byte offset within a block.
///
/// # Arguments
/// * `block_idx` - Block index in the quantized tensor
/// * `elem_in_block` - Element index within the block (0..elems_per_block)
/// * `block_bytes` - Number of bytes per block in the LUT
/// * `elems_per_block` - Number of elements per block (must be > 0)
/// * `lut_len` - Total length of the LUT array in bytes
///
/// # Returns
/// * `Ok(usize)` - Valid LUT byte index
/// * `Err(BitNetError)` - If validation fails or overflow occurs
///
/// # Formula
/// ```text
/// byte_offset = elem_in_block / 8  (integer division for bit-packing)
/// lut_index = block_idx * block_bytes + byte_offset
/// ```
///
/// # Element-to-Byte Mapping
/// Elements 0-7 → byte offset 0
/// Elements 8-15 → byte offset 1
/// Elements 16-23 → byte offset 2, etc.
///
/// # Examples
/// ```
/// use bitnet_kernels::tl_lut::lut_index;
///
/// // First 8 elements (0-7) map to byte 0
/// let idx = lut_index(0, 0, 32, 128, 256).unwrap();
/// assert_eq!(idx, 0);
/// let idx = lut_index(0, 7, 32, 128, 256).unwrap();
/// assert_eq!(idx, 0);
///
/// // Elements 8-15 map to byte 1
/// let idx = lut_index(0, 8, 32, 128, 256).unwrap();
/// assert_eq!(idx, 1);
///
/// // Second block, element 8
/// let idx = lut_index(1, 8, 32, 128, 256).unwrap();
/// assert_eq!(idx, 33); // 1 * 32 + (8 / 8) = 32 + 1
///
/// // Bounds check failure
/// let result = lut_index(0, 128, 32, 128, 256);
/// assert!(result.is_err());
/// ```
pub fn lut_index(
    block_idx: usize,
    elem_in_block: usize,
    block_bytes: usize,
    elems_per_block: usize,
    lut_len: usize,
) -> Result<usize> {
    // Validate non-empty blocks
    if elems_per_block == 0 {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: "elems_per_block must be > 0 (non-empty blocks required)".to_string(),
        }));
    }

    // Bounds check: elem_in_block must be < elems_per_block
    if elem_in_block >= elems_per_block {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "Element index {} exceeds elements per block {}",
                elem_in_block, elems_per_block
            ),
        }));
    }

    // Calculate base offset with overflow check
    let base_offset = block_idx.checked_mul(block_bytes).ok_or_else(|| {
        BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "Overflow computing base offset: block_idx={} * block_bytes={}",
                block_idx, block_bytes
            ),
        })
    })?;

    // Calculate element offset (elem_in_block / 8)
    let elem_offset = elem_in_block / 8;

    // Add offsets with overflow check
    let idx = base_offset.checked_add(elem_offset).ok_or_else(|| {
        BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "Overflow computing LUT index: base_offset={} + elem_offset={}",
                base_offset, elem_offset
            ),
        })
    })?;

    // Validate final index is within LUT bounds
    if idx >= lut_len {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("LUT index {} exceeds LUT length {}", idx, lut_len),
        }));
    }

    Ok(idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_indices() {
        // First element of first block
        assert_eq!(lut_index(0, 0, 32, 128, 256).unwrap(), 0);

        // First element of second block
        assert_eq!(lut_index(1, 0, 32, 128, 256).unwrap(), 32);

        // Middle element (8th element = offset 1)
        assert_eq!(lut_index(0, 8, 32, 128, 256).unwrap(), 1);

        // Combined: second block, 8th element
        assert_eq!(lut_index(1, 8, 32, 128, 256).unwrap(), 33);
    }

    #[test]
    fn test_elem_bounds_check() {
        // elem_in_block >= elems_per_block should fail
        let result = lut_index(0, 128, 32, 128, 256);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds elements per block"));

        // elem_in_block == elems_per_block should fail
        let result = lut_index(0, 128, 32, 128, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_elems_per_block_validation() {
        // elems_per_block == 0 should fail
        let result = lut_index(0, 0, 32, 0, 256);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("elems_per_block must be > 0"));
    }

    #[test]
    fn test_lut_length_validation() {
        // Index that would exceed LUT length
        let result = lut_index(10, 0, 32, 128, 256);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds LUT length"));
    }

    #[test]
    fn test_overflow_detection() {
        // Test overflow in base offset calculation (block_idx * block_bytes)
        let result = lut_index(usize::MAX, 0, 32, 128, usize::MAX);
        assert!(result.is_err(), "Expected overflow in base offset calculation");

        // Test overflow in final addition (base_offset + elem_offset)
        // Use block_bytes=1 for precise control: base_offset = (usize::MAX - 5) * 1 = usize::MAX - 5
        // elem_offset = 64 / 8 = 8, so idx = (usize::MAX - 5) + 8 overflows
        let result = lut_index(usize::MAX - 5, 64, 1, 128, usize::MAX);
        assert!(result.is_err(), "Expected overflow in final index calculation");
    }

    #[test]
    fn test_division_by_8() {
        // Verify division by 8 works correctly
        assert_eq!(lut_index(0, 0, 32, 128, 256).unwrap(), 0);
        assert_eq!(lut_index(0, 7, 32, 128, 256).unwrap(), 0); // Still maps to offset 0
        assert_eq!(lut_index(0, 8, 32, 128, 256).unwrap(), 1);
        assert_eq!(lut_index(0, 15, 32, 128, 256).unwrap(), 1);
        assert_eq!(lut_index(0, 16, 32, 128, 256).unwrap(), 2);
    }
}
