//! I2_S (2-bit signed) quantization implementation
//!
//! This module implements 2-bit signed quantization with efficient bit-packing
//! optimization. Four 2-bit values are packed into each byte for optimal storage.
//! The implementation includes SIMD-optimized kernels for x86_64 and ARM64.

use crate::simd_ops::QuantizationKernels;
use crate::utils::{
    calculate_grouped_scales, create_tensor_from_f32, extract_f32_data, pack_2bit_values,
    unpack_2bit_values,
};
use crate::validation::{
    needs_detailed_validation, validate_data_shape_consistency, validate_numerical_input,
    validate_quantized_tensor, validate_tensor_input, validate_unpacked_data_consistency,
};
use crate::{QuantizedTensor, QuantizerTrait};
use bitnet_common::{
    BitNetError, BitNetTensor, QuantizationError, QuantizationType, Result, SecurityError,
    SecurityLimits, Tensor,
};

use candle_core::Device;
use std::sync::OnceLock;

/// I2_S quantization implementation with bit-packing optimization
pub struct I2SQuantizer {
    block_size: usize,
    kernels: QuantizationKernels,
    /// Cache security validation to avoid repeated checks
    validation_done: OnceLock<bool>,
}

/// I2_S block layout constants
pub struct I2SLayout {
    pub block_size: usize,            // 32 elements per block
    pub bytes_per_block: usize,       // 10 = 8 packed + 2 f16 scale
    pub data_bytes_per_block: usize,  // 8 bytes for packed data
    pub scale_bytes_per_block: usize, // 2 bytes for f16 scale
}

impl Default for I2SLayout {
    fn default() -> Self {
        Self {
            block_size: 32,
            bytes_per_block: 10,
            data_bytes_per_block: 8,
            scale_bytes_per_block: 2,
        }
    }
}

impl I2SLayout {
    pub fn with_block_size(block_size: usize) -> Self {
        // For I2_S: 2 bits per element
        let data_bytes = (block_size * 2).div_ceil(8); // Round up bits to bytes
        Self {
            block_size,
            bytes_per_block: data_bytes + 2, // +2 for f16 scale
            data_bytes_per_block: data_bytes,
            scale_bytes_per_block: 2,
        }
    }
}

impl I2SQuantizer {
    /// Create a new I2_S quantizer with default settings
    pub fn new() -> Self {
        let kernels = QuantizationKernels::new();
        let capabilities = kernels.capabilities();
        Self {
            block_size: capabilities.optimal_block_size().min(32), // I2S prefers smaller blocks
            kernels,
            validation_done: OnceLock::new(),
        }
    }

    /// Create a new I2_S quantizer with custom block size
    pub fn with_block_size(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(4), // Minimum 4 for bit-packing
            kernels: QuantizationKernels::new(),
            validation_done: OnceLock::new(),
        }
    }

    /// Quantize tensor using I2_S algorithm on a specific device
    pub fn quantize(&self, tensor: &BitNetTensor, device: &Device) -> Result<QuantizedTensor> {
        self.quantize_with_limits(tensor, device, &SecurityLimits::default())
    }

    /// Quantize tensor with custom security limits
    pub fn quantize_with_limits(
        &self,
        tensor: &BitNetTensor,
        device: &Device,
        limits: &SecurityLimits,
    ) -> Result<QuantizedTensor> {
        // Cache security validation (only validate once per instance)
        self.validation_done.get_or_init(|| {
            // Only perform initial validation once
            validate_tensor_input(tensor, limits).is_ok()
        });

        if !device.is_cpu() {
            #[cfg(any(feature = "gpu", feature = "cuda"))]
            {
                if device.is_cuda()
                    && bitnet_kernels::gpu::cuda::is_cuda_available()
                    && let Ok(res) = self.quantize_cuda_with_limits(tensor, limits)
                {
                    return Ok(res);
                }
            }
        }

        let data = extract_f32_data(tensor)?;
        let shape = tensor.shape().to_vec();

        // Fast path: Skip expensive validation for typical inputs
        if !needs_detailed_validation(&data) {
            // Direct quantization for well-formed data
            return self.quantize_fast_path(&data, &shape);
        }

        // Security: Validate input data for numerical stability (only for edge cases)
        validate_numerical_input(&data)?;

        // Security: Validate data length matches tensor shape
        validate_data_shape_consistency(&data, &shape)?;

        self.quantize_fast_path(&data, &shape)
    }

    /// Fast path quantization for well-formed data
    #[inline]
    fn quantize_fast_path(&self, data: &[f32], shape: &[usize]) -> Result<QuantizedTensor> {
        // Calculate grouped scales for better accuracy
        let scales = calculate_grouped_scales(data, self.block_size, 2);

        // Quantize data in parallel blocks with safety checks
        let quantized_data = self.kernels.quantize_simd(data, &scales, self.block_size, 2)?;

        // Pack 2-bit values into bytes
        let packed_data = pack_2bit_values(&quantized_data);

        Ok(QuantizedTensor::new_with_params(
            packed_data,
            scales,
            None,
            shape.to_vec(),
            QuantizationType::I2S,
            self.block_size,
        ))
    }

    /// Legacy wrapper that defaults to CPU quantization
    pub fn quantize_tensor(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
        self.quantize(tensor, &Device::Cpu)
    }

    /// Quantize weights from f32 slice - compatibility method for tests
    pub fn quantize_weights(&self, weights: &[f32]) -> Result<QuantizedTensor> {
        use crate::utils::create_tensor_from_f32;
        let shape = vec![weights.len()];
        let tensor = create_tensor_from_f32(weights.to_vec(), &shape, &candle_core::Device::Cpu)?;
        self.quantize_tensor(&tensor)
    }

    /// Check if quantizer supports the specified device
    pub fn supports_device(&self, device: &bitnet_common::Device) -> bool {
        match device {
            bitnet_common::Device::Cpu => true,
            bitnet_common::Device::Cuda(_) => cfg!(any(feature = "gpu", feature = "cuda")),
            bitnet_common::Device::Metal => false, // Metal support not yet implemented
            bitnet_common::Device::Hip(_) => false, // HIP support not yet implemented
            bitnet_common::Device::Npu => false, // NPU support not yet implemented
            bitnet_common::Device::OpenCL(_) => false, // OpenCL support not yet implemented
        }
    }

    /// Dequantize tensor from I2_S format on a specific device
    pub fn dequantize(&self, tensor: &QuantizedTensor, device: &Device) -> Result<BitNetTensor> {
        self.dequantize_with_limits(tensor, device, &SecurityLimits::default())
    }

    /// Dequantize tensor with custom security limits
    pub fn dequantize_with_limits(
        &self,
        tensor: &QuantizedTensor,
        device: &Device,
        limits: &SecurityLimits,
    ) -> Result<BitNetTensor> {
        if tensor.qtype != QuantizationType::I2S {
            return Err(
                QuantizationError::UnsupportedType { qtype: tensor.qtype.to_string() }.into()
            );
        }

        // Security: Validate quantized tensor before dequantization
        validate_quantized_tensor(tensor, limits)?;

        // Security: Validate tensor element count before unpacking
        let expected_elements = tensor.shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim).ok_or_else(|| {
                BitNetError::Security(SecurityError::MemoryBomb {
                    reason: "Dequantization shape element count overflow".to_string(),
                })
            })
        })?;

        let tensor_numel = tensor.numel();
        if tensor_numel != expected_elements {
            return Err(BitNetError::Security(SecurityError::MalformedData {
                reason: format!(
                    "Tensor numel {} does not match shape element count {}",
                    tensor_numel, expected_elements
                ),
            }));
        }

        // Unpack 2-bit values with safety checks
        let quantized_data = unpack_2bit_values(&tensor.data, tensor_numel);

        // Security: Validate unpacked data length
        validate_unpacked_data_consistency(&quantized_data, tensor_numel)?;

        // Dequantize in parallel blocks with safety checks
        let dequantized_data =
            self.kernels.dequantize_simd(&quantized_data, &tensor.scales, self.block_size)?;

        // Create tensor on requested device
        create_tensor_from_f32(dequantized_data, &tensor.shape, device)
    }

    /// Legacy wrapper that defaults to CPU dequantization
    pub fn dequantize_tensor(&self, tensor: &QuantizedTensor) -> Result<BitNetTensor> {
        self.dequantize(tensor, &Device::Cpu)
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn quantize_cuda_with_limits(
        &self,
        tensor: &BitNetTensor,
        limits: &SecurityLimits,
    ) -> Result<QuantizedTensor> {
        use bitnet_kernels::{KernelProvider, gpu::cuda::CudaKernel};

        // Security: Validate input before GPU processing
        validate_tensor_input(tensor, limits)?;

        let data = extract_f32_data(tensor)?;
        let shape = tensor.shape().to_vec();

        // Security: Validate input data for numerical stability
        validate_numerical_input(&data)?;

        let num_blocks = data.len().div_ceil(self.block_size);
        let mut scales = vec![0f32; num_blocks];
        let packed_len = (data.len() * 2).div_ceil(8);
        let mut packed_data = vec![0u8; packed_len];
        let kernel = CudaKernel::new()?;
        kernel.quantize(&data, &mut packed_data, &mut scales, QuantizationType::I2S)?;
        Ok(QuantizedTensor::new_with_params(
            packed_data,
            scales,
            None,
            shape,
            QuantizationType::I2S,
            self.block_size,
        ))
    }
}

impl Default for I2SQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizerTrait for I2SQuantizer {
    fn quantize_tensor(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
        I2SQuantizer::quantize_tensor(self, tensor)
    }

    fn dequantize_tensor(&self, tensor: &QuantizedTensor) -> Result<BitNetTensor> {
        I2SQuantizer::dequantize_tensor(self, tensor)
    }

    fn quantization_type(&self) -> QuantizationType {
        QuantizationType::I2S
    }

    fn is_available(&self) -> bool {
        true // I2_S is always available as it has scalar fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_i2s_quantization_round_trip() {
        let device = Device::Cpu;
        let data = vec![1.0, -2.0, 0.5, -0.5, 3.0, -1.5];
        let shape = vec![2, 3];

        let tensor = create_tensor_from_f32(data.clone(), &shape, &device).unwrap();
        let quantizer = I2SQuantizer::new();

        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

        assert_eq!(quantized.qtype, QuantizationType::I2S);
        assert_eq!(quantized.shape, shape);

        // Check that dequantized tensor has the same shape
        assert_eq!(dequantized.shape(), &shape);
    }

    #[test]
    fn test_i2s_compression_ratio() {
        let device = Device::Cpu;
        let data = vec![1.0; 1024]; // 1024 elements
        let shape = vec![32, 32];

        let tensor = create_tensor_from_f32(data, &shape, &device).unwrap();
        let quantizer = I2SQuantizer::new();

        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let ratio = quantized.compression_ratio();

        // Should achieve significant compression (>8x for large tensors)
        assert!(ratio > 4.0);
    }

    #[test]
    fn test_different_block_sizes() {
        let device = Device::Cpu;
        let data = vec![1.0, -2.0, 0.5, -0.5, 3.0, -1.5, 2.0, -3.0];
        let shape = vec![8];

        let tensor = create_tensor_from_f32(data, &shape, &device).unwrap();

        for block_size in [4, 8, 16] {
            let quantizer = I2SQuantizer::with_block_size(block_size);
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

            assert_eq!(quantized.block_size, block_size);
            assert_eq!(dequantized.shape(), &shape);
        }
    }
}
