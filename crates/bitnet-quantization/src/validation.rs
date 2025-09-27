//! Common validation utilities for quantization operations
//!
//! This module provides shared validation functions used across all quantization types
//! to ensure consistent error handling, security checks, and numerical stability.

use crate::QuantizedTensor;
use bitnet_common::{BitNetError, BitNetTensor, Result, SecurityError, SecurityLimits, Tensor};

/// Validates input tensor against security limits with comprehensive checks
pub fn validate_tensor_input(tensor: &BitNetTensor, limits: &SecurityLimits) -> Result<()> {
    let shape = tensor.shape();

    // Security: Validate shape before any calculations
    if shape.is_empty() {
        return Err(BitNetError::Security(SecurityError::MalformedData {
            reason: "Tensor shape cannot be empty".to_string(),
        }));
    }

    if shape.len() > 8 {
        return Err(BitNetError::Security(SecurityError::ResourceLimit {
            resource: "tensor_dimensions".to_string(),
            value: shape.len() as u64,
            limit: 8,
        }));
    }

    let total_elements = shape.iter().enumerate().try_fold(1u64, |acc, (i, &dim)| {
        if dim == 0 {
            return Err(BitNetError::Security(SecurityError::MalformedData {
                reason: format!("Tensor dimension {} cannot be zero", i),
            }));
        }

        if dim > 1_000_000_000 {
            return Err(BitNetError::Security(SecurityError::ResourceLimit {
                resource: "tensor_dimension_size".to_string(),
                value: dim as u64,
                limit: 1_000_000_000,
            }));
        }

        acc.checked_mul(dim as u64).ok_or_else(|| {
            BitNetError::Security(SecurityError::MemoryBomb {
                reason: format!("Tensor dimension multiplication overflow at dimension {}", i),
            })
        })
    })?;

    // Security: Check tensor element count
    if total_elements > limits.max_tensor_elements {
        return Err(BitNetError::Security(SecurityError::ResourceLimit {
            resource: "tensor_elements".to_string(),
            value: total_elements,
            limit: limits.max_tensor_elements,
        }));
    }

    tracing::debug!(
        "Tensor input validation passed: {} elements, {} dimensions",
        total_elements,
        shape.len()
    );

    Ok(())
}

/// Validates numerical input for potential overflow or invalid values
pub fn validate_numerical_input(data: &[f32]) -> Result<()> {
    let nan_count = data.iter().filter(|&&x| x.is_nan()).count();
    let inf_count = data.iter().filter(|&&x| x.is_infinite()).count();
    let extreme_count = data.iter().filter(|&&x| x.is_finite() && x.abs() > 1e30).count();

    // Security: Log warnings for problematic values but don't fail
    // This allows quantization to proceed with sanitized values
    if nan_count > 0 {
        tracing::warn!(
            "Quantization input contains {} NaN values - these will be mapped to zero",
            nan_count
        );
    }

    if inf_count > 0 {
        tracing::warn!(
            "Quantization input contains {} infinite values - these will be mapped to zero",
            inf_count
        );
    }

    if extreme_count > 0 {
        tracing::warn!(
            "Quantization input contains {} extreme values (>1e30) - these will be clamped",
            extreme_count
        );
    }

    // Security: Fail only if all values are problematic
    let total_problematic = nan_count + inf_count;
    if total_problematic == data.len() && !data.is_empty() {
        return Err(BitNetError::Security(SecurityError::MalformedData {
            reason: "All input values are NaN or infinite - cannot quantize".to_string(),
        }));
    }

    Ok(())
}

/// Validates quantized tensor against security limits
pub fn validate_quantized_tensor(tensor: &QuantizedTensor, limits: &SecurityLimits) -> Result<()> {
    // Security: Validate tensor shape before calculations
    if tensor.shape.is_empty() {
        return Err(BitNetError::Security(SecurityError::MalformedData {
            reason: "Quantized tensor shape cannot be empty".to_string(),
        }));
    }

    if tensor.shape.len() > 8 {
        return Err(BitNetError::Security(SecurityError::ResourceLimit {
            resource: "quantized_tensor_dimensions".to_string(),
            value: tensor.shape.len() as u64,
            limit: 8,
        }));
    }

    // Security: Validate tensor shape and element count
    let total_elements = tensor.shape.iter().enumerate().try_fold(1u64, |acc, (i, &dim)| {
        if dim == 0 {
            return Err(BitNetError::Security(SecurityError::MalformedData {
                reason: format!("Quantized tensor dimension {} cannot be zero", i),
            }));
        }

        if dim > 1_000_000_000 {
            return Err(BitNetError::Security(SecurityError::ResourceLimit {
                resource: "quantized_tensor_dimension_size".to_string(),
                value: dim as u64,
                limit: 1_000_000_000,
            }));
        }

        acc.checked_mul(dim as u64).ok_or_else(|| {
            BitNetError::Security(SecurityError::MemoryBomb {
                reason: format!("Quantized tensor dimension overflow at dimension {}", i),
            })
        })
    })?;

    if total_elements > limits.max_tensor_elements {
        return Err(BitNetError::Security(SecurityError::ResourceLimit {
            resource: "quantized_tensor_elements".to_string(),
            value: total_elements,
            limit: limits.max_tensor_elements,
        }));
    }

    // Security: Validate data and scales array sizes
    if tensor.data.len() > limits.max_memory_allocation {
        return Err(BitNetError::Security(SecurityError::MemoryBomb {
            reason: format!(
                "Quantized data size {} exceeds memory limit {}",
                tensor.data.len(),
                limits.max_memory_allocation
            ),
        }));
    }

    if tensor.scales.len() > limits.max_array_length {
        return Err(BitNetError::Security(SecurityError::ResourceLimit {
            resource: "scales_array_length".to_string(),
            value: tensor.scales.len() as u64,
            limit: limits.max_array_length as u64,
        }));
    }

    // Security: Validate block size is reasonable
    if tensor.block_size == 0 || tensor.block_size > 1024 {
        return Err(BitNetError::Security(SecurityError::MalformedData {
            reason: format!("Invalid block size: {}", tensor.block_size),
        }));
    }

    tracing::debug!(
        "Quantized tensor validation passed: {} elements, {} bytes data, {} scales",
        total_elements,
        tensor.data.len(),
        tensor.scales.len()
    );

    Ok(())
}

/// Validates data length matches expected tensor shape
pub fn validate_data_shape_consistency(data: &[f32], shape: &[usize]) -> Result<()> {
    let expected_elements = shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            BitNetError::Security(SecurityError::MemoryBomb {
                reason: "Shape element count overflow".to_string(),
            })
        })
    })?;

    if data.len() != expected_elements {
        return Err(BitNetError::Security(SecurityError::MalformedData {
            reason: format!(
                "Data length {} does not match shape element count {}",
                data.len(),
                expected_elements
            ),
        }));
    }

    Ok(())
}

/// Estimates memory requirements for quantization
pub fn estimate_quantization_memory(
    total_elements: usize,
    bits_per_element: u8,
    block_size: usize,
) -> usize {
    // Each element becomes `bits_per_element` bits + scale overhead
    let bits_per_element = bits_per_element as f64;
    let element_memory = (total_elements as f64 * bits_per_element / 8.0) as usize;
    let scale_memory = (total_elements / block_size) * 4; // 4 bytes per f32 scale
    element_memory + scale_memory
}

/// Validates memory requirements against limits
pub fn validate_memory_requirements(
    total_elements: usize,
    bits_per_element: u8,
    block_size: usize,
    limits: &SecurityLimits,
) -> Result<()> {
    let memory_estimate =
        estimate_quantization_memory(total_elements, bits_per_element, block_size);

    if memory_estimate > limits.max_memory_allocation {
        return Err(BitNetError::Security(SecurityError::MemoryBomb {
            reason: format!(
                "Quantization memory requirement {} exceeds limit {}",
                memory_estimate, limits.max_memory_allocation
            ),
        }));
    }

    Ok(())
}

/// Fast heuristic to check if data needs detailed validation
#[inline]
pub fn needs_detailed_validation(data: &[f32]) -> bool {
    // Only validate if we detect potential edge cases
    data.len() > 1_000_000 || data.iter().any(|&x| !x.is_finite())
}

/// Validates unpacked data length consistency
pub fn validate_unpacked_data_consistency(
    quantized_data: &[i8],
    expected_elements: usize,
) -> Result<()> {
    if quantized_data.len() != expected_elements {
        return Err(BitNetError::Security(SecurityError::MalformedData {
            reason: format!(
                "Unpacked data length {} does not match expected {}",
                quantized_data.len(),
                expected_elements
            ),
        }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::create_tensor_from_f32;
    use bitnet_common::{SecurityLimits, Tensor};
    use candle_core::Device;

    #[test]
    fn test_validate_tensor_input() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = create_tensor_from_f32(data, &shape, &Device::Cpu).unwrap();
        let limits = SecurityLimits::default();

        assert!(validate_tensor_input(&tensor, &limits).is_ok());
    }

    #[test]
    fn test_validate_numerical_input() {
        let valid_data = vec![1.0, -2.0, 0.5, -0.5];
        assert!(validate_numerical_input(&valid_data).is_ok());

        let invalid_data = vec![f32::NAN; 4];
        assert!(validate_numerical_input(&invalid_data).is_err());
    }

    #[test]
    fn test_validate_data_shape_consistency() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        assert!(validate_data_shape_consistency(&data, &shape).is_ok());

        let wrong_shape = vec![3, 2];
        assert!(validate_data_shape_consistency(&data, &wrong_shape).is_err());
    }

    #[test]
    fn test_memory_estimation() {
        let memory = estimate_quantization_memory(1024, 2, 32);
        assert!(memory > 0);
        assert!(memory < 10000); // Reasonable bound
    }

    #[test]
    fn test_needs_detailed_validation() {
        let small_data = vec![1.0, 2.0, 3.0];
        assert!(!needs_detailed_validation(&small_data));

        let large_data = vec![1.0; 2_000_000];
        assert!(needs_detailed_validation(&large_data));

        let nan_data = vec![f32::NAN, 1.0, 2.0];
        assert!(needs_detailed_validation(&nan_data));
    }
}
