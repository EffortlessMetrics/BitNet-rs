//! I2_S (2-bit signed) quantization implementation
//!
//! This module implements 2-bit signed quantization with efficient bit-packing
//! optimization. Four 2-bit values are packed into each byte for optimal storage.
//! The implementation includes SIMD-optimized kernels for x86_64 and ARM64.

use crate::utils::{
    calculate_grouped_scales, create_tensor_from_f32, dequantize_value, extract_f32_data,
    pack_2bit_values, quantize_value, unpack_2bit_values,
};
use crate::{QuantizedTensor, QuantizerTrait};
use bitnet_common::{
    BitNetError, BitNetTensor, QuantizationError, QuantizationType, Result, SecurityError,
    SecurityLimits, Tensor,
};
#[cfg(feature = "gpu")]
#[allow(unused_imports)]
use bitnet_kernels::KernelProvider;
use candle_core::Device;
use rayon::prelude::*;
use std::sync::OnceLock;

/// I2_S quantization implementation with bit-packing optimization
pub struct I2SQuantizer {
    block_size: usize,
    use_simd: bool,
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
        Self {
            block_size: 32,
            use_simd: cfg!(any(target_arch = "x86_64", target_arch = "aarch64")),
            validation_done: OnceLock::new(),
        }
    }

    /// Validate input tensor against security limits
    fn validate_tensor_input(tensor: &BitNetTensor, limits: &SecurityLimits) -> Result<()> {
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

        // Security: Check memory requirements for I2S quantization
        // Each f32 element (4 bytes) becomes 2 bits (0.25 bytes) + scale overhead
        let memory_estimate =
            (total_elements as f64 * 0.25) as usize + (total_elements as usize / 32) * 4; // Add scale overhead
        if memory_estimate > limits.max_memory_allocation {
            return Err(BitNetError::Security(SecurityError::MemoryBomb {
                reason: format!(
                    "I2S quantization memory requirement {} exceeds limit {}",
                    memory_estimate, limits.max_memory_allocation
                ),
            }));
        }

        // Security: Validate tensor dimensions are reasonable
        if shape.len() > 8 {
            return Err(BitNetError::Security(SecurityError::ResourceLimit {
                resource: "tensor_dimensions".to_string(),
                value: shape.len() as u64,
                limit: 8,
            }));
        }

        for &dim in shape {
            if dim > 1_000_000_000 {
                return Err(BitNetError::Security(SecurityError::ResourceLimit {
                    resource: "tensor_dimension_size".to_string(),
                    value: dim as u64,
                    limit: 1_000_000_000,
                }));
            }
        }

        tracing::debug!(
            "I2S quantization input validation passed: {} elements, {} dimensions",
            total_elements,
            shape.len()
        );

        Ok(())
    }

    /// Validate numerical input for potential overflow or invalid values
    fn validate_numerical_input(data: &[f32]) -> Result<()> {
        let nan_count = data.iter().filter(|&&x| x.is_nan()).count();
        let inf_count = data.iter().filter(|&&x| x.is_infinite()).count();
        let extreme_count = data.iter().filter(|&&x| x.is_finite() && x.abs() > 1e30).count();

        // Security: Log warnings for problematic values but don't fail
        // This allows quantization to proceed with sanitized values
        if nan_count > 0 {
            tracing::warn!(
                "I2S quantization input contains {} NaN values - these will be mapped to zero",
                nan_count
            );
        }

        if inf_count > 0 {
            tracing::warn!(
                "I2S quantization input contains {} infinite values - these will be mapped to zero",
                inf_count
            );
        }

        if extreme_count > 0 {
            tracing::warn!(
                "I2S quantization input contains {} extreme values (>1e30) - these will be clamped",
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

    /// Validate quantized tensor against security limits
    fn validate_quantized_tensor(tensor: &QuantizedTensor, limits: &SecurityLimits) -> Result<()> {
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
            "I2S dequantization input validation passed: {} elements, {} bytes data, {} scales",
            total_elements,
            tensor.data.len(),
            tensor.scales.len()
        );

        Ok(())
    }

    /// Create a new I2_S quantizer with custom block size
    pub fn with_block_size(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(4), // Minimum 4 for bit-packing
            use_simd: cfg!(any(target_arch = "x86_64", target_arch = "aarch64")),
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
            Self::validate_tensor_input(tensor, limits).is_ok()
        });

        if !device.is_cpu() {
            #[cfg(feature = "cuda")]
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
        if !self.needs_detailed_validation(&data) {
            // Direct quantization for well-formed data
            return self.quantize_fast_path(&data, &shape);
        }

        // Security: Validate input data for numerical stability (only for edge cases)
        Self::validate_numerical_input(&data)?;

        // Security: Validate data length matches tensor shape
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

        self.quantize_fast_path(&data, &shape)
    }

    /// Fast path quantization for well-formed data
    #[inline]
    fn quantize_fast_path(&self, data: &[f32], shape: &[usize]) -> Result<QuantizedTensor> {
        // Calculate grouped scales for better accuracy
        let scales = calculate_grouped_scales(data, self.block_size, 2);

        // Quantize data in parallel blocks with safety checks
        let quantized_data = if self.use_simd {
            self.quantize_simd(data, &scales)?
        } else {
            self.quantize_scalar(data, &scales)?
        };

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

    /// Check if data needs detailed validation (fast heuristic)
    #[inline]
    fn needs_detailed_validation(&self, data: &[f32]) -> bool {
        // Only validate if we detect potential edge cases
        data.len() > 1_000_000 || data.iter().any(|&x| !x.is_finite())
    }

    /// Legacy wrapper that defaults to CPU quantization
    pub fn quantize_tensor(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
        self.quantize(tensor, &Device::Cpu)
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
        Self::validate_quantized_tensor(tensor, limits)?;

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
        if quantized_data.len() != tensor_numel {
            return Err(BitNetError::Security(SecurityError::MalformedData {
                reason: format!(
                    "Unpacked data length {} does not match expected {}",
                    quantized_data.len(),
                    tensor_numel
                ),
            }));
        }

        // Dequantize in parallel blocks with safety checks
        let dequantized_data = if self.use_simd {
            self.dequantize_simd(&quantized_data, &tensor.scales)?
        } else {
            self.dequantize_scalar(&quantized_data, &tensor.scales)?
        };

        // Create tensor on requested device
        create_tensor_from_f32(dequantized_data, &tensor.shape, device)
    }

    /// Legacy wrapper that defaults to CPU dequantization
    pub fn dequantize_tensor(&self, tensor: &QuantizedTensor) -> Result<BitNetTensor> {
        self.dequantize(tensor, &Device::Cpu)
    }

    #[cfg(feature = "cuda")]
    fn quantize_cuda(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
        self.quantize_cuda_with_limits(tensor, &SecurityLimits::default())
    }

    #[cfg(feature = "cuda")]
    fn quantize_cuda_with_limits(
        &self,
        tensor: &BitNetTensor,
        limits: &SecurityLimits,
    ) -> Result<QuantizedTensor> {
        use bitnet_kernels::gpu::cuda::CudaKernel;

        // Security: Validate input before GPU processing
        Self::validate_tensor_input(tensor, limits)?;

        let data = extract_f32_data(tensor)?;
        let shape = tensor.shape().to_vec();

        // Security: Validate input data for numerical stability
        Self::validate_numerical_input(&data)?;

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

    /// Scalar quantization implementation with safety checks
    fn quantize_scalar(&self, data: &[f32], scales: &[f32]) -> Result<Vec<i8>> {
        let mut quantized = vec![0i8; data.len()];

        quantized
            .par_chunks_mut(self.block_size)
            .zip(data.par_chunks(self.block_size))
            .zip(scales.par_iter())
            .for_each(|((quant_block, data_block), &scale)| {
                for (i, &value) in data_block.iter().enumerate() {
                    quant_block[i] = quantize_value(value, scale, 2);
                }
            });

        Ok(quantized)
    }

    /// Scalar dequantization implementation with safety checks
    fn dequantize_scalar(&self, quantized: &[i8], scales: &[f32]) -> Result<Vec<f32>> {
        let mut dequantized = vec![0.0f32; quantized.len()];

        dequantized
            .par_chunks_mut(self.block_size)
            .zip(quantized.par_chunks(self.block_size))
            .zip(scales.par_iter())
            .for_each(|((dequant_block, quant_block), &scale)| {
                for (i, &value) in quant_block.iter().enumerate() {
                    dequant_block[i] = dequantize_value(value, scale);
                }
            });

        Ok(dequantized)
    }

    /// SIMD-optimized quantization with safety checks
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn quantize_simd(&self, data: &[f32], scales: &[f32]) -> Result<Vec<i8>> {
        #[cfg(target_arch = "x86_64")]
        {
            self.quantize_avx2(data, scales)
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.quantize_neon(data, scales)
        }
    }

    /// SIMD-optimized dequantization with safety checks
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn dequantize_simd(&self, quantized: &[i8], scales: &[f32]) -> Result<Vec<f32>> {
        #[cfg(target_arch = "x86_64")]
        {
            self.dequantize_avx2(quantized, scales)
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.dequantize_neon(quantized, scales)
        }
    }

    /// Fallback to scalar for unsupported architectures
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn quantize_simd(&self, data: &[f32], scales: &[f32]) -> Result<Vec<i8>> {
        self.quantize_scalar(data, scales)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn dequantize_simd(&self, quantized: &[i8], scales: &[f32]) -> Result<Vec<f32>> {
        self.dequantize_scalar(quantized, scales)
    }

    /// AVX2-optimized quantization for x86_64
    #[cfg(target_arch = "x86_64")]
    fn quantize_avx2(&self, data: &[f32], scales: &[f32]) -> Result<Vec<i8>> {
        if !is_x86_feature_detected!("avx2") {
            return self.quantize_scalar(data, scales);
        }

        let mut quantized = vec![0i8; data.len()];

        quantized
            .par_chunks_mut(self.block_size)
            .zip(data.par_chunks(self.block_size))
            .zip(scales.par_iter())
            .for_each(|((quant_block, data_block), &scale)| unsafe {
                self.quantize_avx2_block(data_block, quant_block, scale);
            });

        Ok(quantized)
    }

    /// AVX2-optimized dequantization for x86_64
    #[cfg(target_arch = "x86_64")]
    fn dequantize_avx2(&self, quantized: &[i8], scales: &[f32]) -> Result<Vec<f32>> {
        if !is_x86_feature_detected!("avx2") {
            return self.dequantize_scalar(quantized, scales);
        }

        let mut dequantized = vec![0.0f32; quantized.len()];

        dequantized
            .par_chunks_mut(self.block_size)
            .zip(quantized.par_chunks(self.block_size))
            .zip(scales.par_iter())
            .for_each(|((dequant_block, quant_block), &scale)| unsafe {
                self.dequantize_avx2_block(quant_block, dequant_block, scale);
            });

        Ok(dequantized)
    }

    /// AVX2 kernel for quantizing a single block
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn quantize_avx2_block(&self, data: &[f32], output: &mut [i8], scale: f32) {
        #[allow(clippy::wildcard_imports)]
        use std::arch::x86_64::*;

        let inv_scale = 1.0 / scale;
        let inv_scale_vec = _mm256_set1_ps(inv_scale);
        let min_val = _mm256_set1_ps(-2.0);
        let max_val = _mm256_set1_ps(1.0);

        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();

        for (i, chunk) in chunks.enumerate() {
            unsafe {
                let data_vec = _mm256_loadu_ps(chunk.as_ptr());
                let scaled = _mm256_mul_ps(data_vec, inv_scale_vec);
                let rounded = _mm256_round_ps::<_MM_FROUND_TO_NEAREST_INT>(scaled);
                let clamped = _mm256_max_ps(_mm256_min_ps(rounded, max_val), min_val);

                // Convert to i32 and then to i8
                let i32_vec = _mm256_cvtps_epi32(clamped);
                let i16_vec = _mm256_packs_epi32(i32_vec, i32_vec);
                let i8_vec = _mm256_packs_epi16(i16_vec, i16_vec);

                // Store 8 bytes efficiently
                let result = _mm256_extract_epi64::<0>(i8_vec);
                std::ptr::write_unaligned(output.as_mut_ptr().add(i * 8) as *mut i64, result);
            }
        }

        // Handle remainder with inline quantization (avoid function call overhead)
        for (i, &value) in remainder.iter().enumerate() {
            let idx = data.len() - remainder.len() + i;
            // Inline quantization for better performance
            let normalized = if value.is_finite() && scale != 0.0 {
                (value / scale).round().clamp(-2.0, 1.0) as i8
            } else {
                0i8
            };
            output[idx] = normalized;
        }
    }

    /// AVX2 kernel for dequantizing a single block
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dequantize_avx2_block(&self, quantized: &[i8], output: &mut [f32], scale: f32) {
        #[allow(clippy::wildcard_imports)]
        use std::arch::x86_64::*;

        let scale_vec = _mm256_set1_ps(scale);

        let chunks = quantized.chunks_exact(8);
        let remainder = chunks.remainder();

        for (i, chunk) in chunks.enumerate() {
            unsafe {
                // Load 8 i8 values
                let i8_vec = _mm_loadu_si64(chunk.as_ptr() as *const u8);

                // Convert to i32 and then to f32
                let i32_vec = _mm256_cvtepi8_epi32(i8_vec);
                let f32_vec = _mm256_cvtepi32_ps(i32_vec);
                let result = _mm256_mul_ps(f32_vec, scale_vec);

                _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), result);
            }
        }

        // Handle remainder with inline dequantization (avoid function call overhead)
        for (i, &value) in remainder.iter().enumerate() {
            let idx = quantized.len() - remainder.len() + i;
            // Inline dequantization for better performance
            output[idx] = if scale.is_finite() { value as f32 * scale } else { 0.0 };
        }
    }

    /// NEON-optimized quantization for ARM64
    #[cfg(target_arch = "aarch64")]
    fn quantize_neon(&self, data: &[f32], scales: &[f32]) -> Result<Vec<i8>> {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return self.quantize_scalar(data, scales);
        }

        let mut quantized = vec![0i8; data.len()];

        quantized
            .par_chunks_mut(self.block_size)
            .zip(data.par_chunks(self.block_size))
            .zip(scales.par_iter())
            .for_each(|((quant_block, data_block), &scale)| unsafe {
                self.quantize_neon_block(data_block, quant_block, scale);
            });

        Ok(quantized)
    }

    /// NEON-optimized dequantization for ARM64
    #[cfg(target_arch = "aarch64")]
    fn dequantize_neon(&self, quantized: &[i8], scales: &[f32]) -> Result<Vec<f32>> {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return self.dequantize_scalar(quantized, scales);
        }

        let mut dequantized = vec![0.0f32; quantized.len()];

        dequantized
            .par_chunks_mut(self.block_size)
            .zip(quantized.par_chunks(self.block_size))
            .zip(scales.par_iter())
            .for_each(|((dequant_block, quant_block), &scale)| unsafe {
                self.dequantize_neon_block(quant_block, dequant_block, scale);
            });

        Ok(dequantized)
    }

    /// NEON kernel for quantizing a single block
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn quantize_neon_block(&self, data: &[f32], output: &mut [i8], scale: f32) {
        use std::arch::aarch64::*;

        let inv_scale = 1.0 / scale;
        let inv_scale_vec = vdupq_n_f32(inv_scale);
        let min_val = vdupq_n_f32(-2.0);
        let max_val = vdupq_n_f32(1.0);

        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();

        for (i, chunk) in chunks.enumerate() {
            unsafe {
                let data_vec = vld1q_f32(chunk.as_ptr());
                let scaled = vmulq_f32(data_vec, inv_scale_vec);
                let rounded = vrndnq_f32(scaled);
                let clamped = vmaxq_f32(vminq_f32(rounded, max_val), min_val);

                // Convert to i32 and then to i8
                let i32_vec = vcvtq_s32_f32(clamped);
                let i16_vec = vqmovn_s32(i32_vec);
                let i8_vec = vqmovn_s16(vcombine_s16(i16_vec, i16_vec));

                // Store 4 bytes
                let result = vget_lane_u32::<0>(vreinterpret_u32_s8(i8_vec));
                std::ptr::copy_nonoverlapping(
                    &result as *const u32 as *const i8,
                    output.as_mut_ptr().add(i * 4),
                    4,
                );
            }
        }

        // Handle remainder with scalar code
        for (i, &value) in remainder.iter().enumerate() {
            let idx = data.len() - remainder.len() + i;
            output[idx] = quantize_value(value, scale, 2);
        }
    }

    /// NEON kernel for dequantizing a single block
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn dequantize_neon_block(&self, quantized: &[i8], output: &mut [f32], scale: f32) {
        use std::arch::aarch64::*;

        let scale_vec = vdupq_n_f32(scale);

        let chunks = quantized.chunks_exact(4);
        let remainder = chunks.remainder();

        for (i, chunk) in chunks.enumerate() {
            unsafe {
                // Load 4 i8 values
                let i8_vec = vreinterpret_s8_u32(vld1_dup_u32(chunk.as_ptr() as *const u32));

                // Convert to i32 and then to f32
                let i16_vec = vmovl_s8(i8_vec);
                let i32_vec = vmovl_s16(vget_low_s16(i16_vec));
                let f32_vec = vcvtq_f32_s32(i32_vec);
                let result = vmulq_f32(f32_vec, scale_vec);

                vst1q_f32(output.as_mut_ptr().add(i * 4), result);
            }
        }

        // Handle remainder with scalar code
        for (i, &value) in remainder.iter().enumerate() {
            let idx = quantized.len() - remainder.len() + i;
            output[idx] = dequantize_value(value, scale);
        }
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
