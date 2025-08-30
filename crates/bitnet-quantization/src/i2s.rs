//! I2_S (2-bit signed) quantization implementation
//!
//! This module implements 2-bit signed quantization with efficient bit-packing
//! optimization. Four 2-bit values are packed into each byte for optimal storage.
//! The implementation includes SIMD-optimized kernels for x86_64 and ARM64.

use crate::{QuantizedTensor, QuantizerTrait, utils::*};
use bitnet_common::{BitNetTensor, QuantizationError, QuantizationType, Result, Tensor};
use candle_core::Device;
use rayon::prelude::*;

/// I2_S quantization implementation with bit-packing optimization
pub struct I2SQuantizer {
    block_size: usize,
    use_simd: bool,
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
        }
    }

    /// Create a new I2_S quantizer with custom block size
    pub fn with_block_size(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(4), // Minimum 4 for bit-packing
            use_simd: cfg!(any(target_arch = "x86_64", target_arch = "aarch64")),
        }
    }

    /// Quantize tensor using I2_S algorithm on a specific device
    pub fn quantize(&self, tensor: &BitNetTensor, device: &Device) -> Result<QuantizedTensor> {
        if !device.is_cpu() {
            #[cfg(feature = "cuda")]
            {
                if device.is_cuda() && bitnet_kernels::gpu::cuda::is_cuda_available() {
                    if let Ok(res) = self.quantize_cuda(tensor) {
                        return Ok(res);
                    }
                }
            }
        }

        let data = extract_f32_data(tensor)?;
        let shape = tensor.shape().to_vec();

        // Calculate grouped scales for better accuracy
        let scales = calculate_grouped_scales(&data, self.block_size, 2);

        // Quantize data in parallel blocks
        let quantized_data = if self.use_simd {
            self.quantize_simd(&data, &scales)?
        } else {
            self.quantize_scalar(&data, &scales)?
        };

        // Pack 2-bit values into bytes
        let packed_data = pack_2bit_values(&quantized_data);

        Ok(QuantizedTensor::new_with_params(
            packed_data,
            scales,
            None,
            shape,
            QuantizationType::I2S,
            self.block_size,
        ))
    }

    /// Legacy wrapper that defaults to CPU quantization
    pub fn quantize_tensor(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
        self.quantize(tensor, &Device::Cpu)
    }

    /// Dequantize tensor from I2_S format on a specific device
    pub fn dequantize(&self, tensor: &QuantizedTensor, device: &Device) -> Result<BitNetTensor> {
        if tensor.qtype != QuantizationType::I2S {
            return Err(
                QuantizationError::UnsupportedType { qtype: tensor.qtype.to_string() }.into()
            );
        }

        // Unpack 2-bit values
        let quantized_data = unpack_2bit_values(&tensor.data, tensor.numel());

        // Dequantize in parallel blocks
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
        use bitnet_kernels::gpu::cuda::CudaKernel;
        let data = extract_f32_data(tensor)?;
        let shape = tensor.shape().to_vec();
        let num_blocks = data.len().div_ceil(self.block_size);
        let mut scales = vec![0f32; num_blocks];
        let packed_len = (data.len() * 2 + 7) / 8;
        let mut packed_data = vec![0u8; packed_len];
        let kernel = CudaKernel::new(0)?;
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

    /// Scalar quantization implementation
    fn quantize_scalar(&self, data: &[f32], scales: &[f32]) -> Result<Vec<i8>> {
        let _num_blocks = scales.len();
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

    /// Scalar dequantization implementation
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

    /// SIMD-optimized quantization
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

    /// SIMD-optimized dequantization
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

                // Store 8 bytes
                let result = _mm256_extract_epi64::<0>(i8_vec) as i64;
                std::ptr::copy_nonoverlapping(
                    &result as *const i64 as *const i8,
                    output.as_mut_ptr().add(i * 8),
                    8,
                );
            }
        }

        // Handle remainder with scalar code
        for (i, &value) in remainder.iter().enumerate() {
            let idx = data.len() - remainder.len() + i;
            output[idx] = quantize_value(value, scale, 2);
        }
    }

    /// AVX2 kernel for dequantizing a single block
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dequantize_avx2_block(&self, quantized: &[i8], output: &mut [f32], scale: f32) {
        use std::arch::x86_64::*;

        let scale_vec = _mm256_set1_ps(scale);

        let chunks = quantized.chunks_exact(8);
        let remainder = chunks.remainder();

        for (i, chunk) in chunks.enumerate() {
            unsafe {
                // Load 8 i8 values
                let i8_data = std::ptr::read_unaligned(chunk.as_ptr() as *const i64);
                let i8_vec = _mm_set1_epi64x(i8_data);

                // Convert to i32 and then to f32
                let i32_vec = _mm256_cvtepi8_epi32(i8_vec);
                let f32_vec = _mm256_cvtepi32_ps(i32_vec);
                let result = _mm256_mul_ps(f32_vec, scale_vec);

                _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), result);
            }
        }

        // Handle remainder with scalar code
        for (i, &value) in remainder.iter().enumerate() {
            let idx = quantized.len() - remainder.len() + i;
            output[idx] = dequantize_value(value, scale);
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
                let i8_data = std::ptr::read_unaligned(chunk.as_ptr() as *const u32);
                let i8_vec = vreinterpret_s8_u32(vdup_n_u32(i8_data));

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
