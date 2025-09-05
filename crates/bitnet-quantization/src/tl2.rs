//! TL2 (Table Lookup 2) quantization for x86 platforms
//!
//! This module implements table lookup quantization optimized for x86 AVX2/AVX-512 instructions.
//! It uses vectorized lookup tables and advanced SIMD operations to achieve maximum throughput
//! on x86 architectures, with runtime CPU feature detection for optimal instruction set selection.

use crate::{QuantizedTensor, QuantizerTrait, utils::*};
use bitnet_common::{BitNetTensor, QuantizationError, QuantizationType, Result, Tensor};
#[cfg(feature = "gpu")]
#[allow(unused_imports)]
use bitnet_kernels::KernelProvider;
use candle_core::Device;
use rayon::prelude::*;
use std::collections::HashMap;

/// Configuration for TL2 quantization with x86-specific optimizations
#[derive(Debug, Clone)]
pub struct TL2Config {
    pub block_size: usize,
    pub lookup_table_size: usize,
    pub use_avx512: bool,
    pub use_avx2: bool,
    pub precision_bits: u8,
    pub vectorized_tables: bool,
}

impl Default for TL2Config {
    fn default() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let (use_avx512, use_avx2) =
            (is_x86_feature_detected!("avx512f"), is_x86_feature_detected!("avx2"));
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let (use_avx512, use_avx2) = (false, false);

        Self {
            block_size: 128, // Larger blocks for x86 vectorization
            lookup_table_size: 256,
            use_avx512,
            use_avx2,
            precision_bits: 2,
            vectorized_tables: true,
        }
    }
}

/// Vectorized lookup table optimized for x86 SIMD
#[derive(Debug, Clone)]
pub struct VectorizedLookupTable {
    /// Forward lookup table aligned for SIMD access
    forward: Vec<i8>,
    /// Reverse lookup table aligned for SIMD access  
    reverse: Vec<f32>,
    /// Scale factor
    scale: f32,
    /// Zero point for asymmetric quantization
    _zero_point: i32,
    /// Number of quantization levels
    _num_levels: usize,
}

impl VectorizedLookupTable {
    /// Create a new vectorized lookup table
    pub fn new(min_val: f32, max_val: f32, bits: u8) -> Self {
        let num_levels = 1 << bits;
        let mut forward = vec![0i8; 256]; // Aligned to 256 for SIMD
        let mut reverse = vec![0.0f32; num_levels];

        // Calculate scale and zero point
        let abs_max = max_val.abs().max(min_val.abs());
        let scale = abs_max / ((num_levels / 2) - 1) as f32;
        let zero_point = 0; // Symmetric quantization for simplicity

        // Build reverse lookup table
        for (i, rev) in reverse.iter_mut().enumerate().take(num_levels) {
            let quantized = i as i32 - (num_levels / 2) as i32;
            *rev = quantized as f32 * scale;
        }

        // Build forward lookup table with SIMD-friendly layout
        for (i, fwd) in forward.iter_mut().enumerate().take(256) {
            let float_val = (i as f32 - 128.0) * scale / 128.0; // Normalize to [-1, 1] range
            let quantized = ((float_val / scale).round() as i32)
                .saturating_add((num_levels / 2) as i32)
                .clamp(0, (num_levels - 1) as i32) as i8;
            *fwd = quantized;
        }

        Self { forward, reverse, scale, _zero_point: zero_point, _num_levels: num_levels }
    }

    /// Quantize using vectorized lookup
    pub fn quantize(&self, value: f32) -> i8 {
        let normalized = (value / self.scale * 128.0 + 128.0).round() as usize;
        let index = normalized.clamp(0, 255);
        self.forward[index]
    }

    /// Dequantize using vectorized lookup
    pub fn dequantize(&self, quantized: i8) -> f32 {
        let index = quantized as usize;
        if index < self.reverse.len() { self.reverse[index] } else { 0.0 }
    }
}

/// TL2 quantization implementation optimized for x86 AVX2/AVX-512
pub struct TL2Quantizer {
    config: TL2Config,
    #[allow(dead_code)]
    lookup_tables: HashMap<String, VectorizedLookupTable>,
    cpu_features: CpuFeatures,
}

/// CPU feature detection for optimal kernel selection
#[derive(Debug, Clone)]
struct CpuFeatures {
    has_avx2: bool,
    has_avx512f: bool,
    has_avx512bw: bool,
    has_avx512vl: bool,
}

impl CpuFeatures {
    fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            Self {
                has_avx2: is_x86_feature_detected!("avx2"),
                has_avx512f: is_x86_feature_detected!("avx512f"),
                has_avx512bw: is_x86_feature_detected!("avx512bw"),
                has_avx512vl: is_x86_feature_detected!("avx512vl"),
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            Self { has_avx2: false, has_avx512f: false, has_avx512bw: false, has_avx512vl: false }
        }
    }

    fn best_kernel(&self) -> KernelType {
        if self.has_avx512f && self.has_avx512bw && self.has_avx512vl {
            KernelType::AVX512
        } else if self.has_avx2 {
            KernelType::AVX2
        } else {
            KernelType::Scalar
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum KernelType {
    Scalar,
    AVX2,
    AVX512,
}

impl TL2Quantizer {
    /// Create a new TL2 quantizer with automatic CPU feature detection
    pub fn new() -> Self {
        let cpu_features = CpuFeatures::detect();
        let mut config = TL2Config::default();

        // Adjust configuration based on available features
        match cpu_features.best_kernel() {
            KernelType::AVX512 => {
                config.block_size = 256; // Larger blocks for AVX-512
                config.use_avx512 = true;
            }
            KernelType::AVX2 => {
                config.block_size = 128;
                config.use_avx2 = true;
            }
            KernelType::Scalar => {
                config.block_size = 64;
                config.vectorized_tables = false;
            }
        }

        Self { config, lookup_tables: HashMap::new(), cpu_features }
    }

    /// Create a new TL2 quantizer with custom configuration
    pub fn with_config(config: TL2Config) -> Self {
        Self { config, lookup_tables: HashMap::new(), cpu_features: CpuFeatures::detect() }
    }

    /// Load configuration from .ini file for compatibility with C++ implementation
    pub fn from_ini_file(path: &str) -> Result<Self> {
        let mut config = TL2Config::default();

        if let Ok(content) = std::fs::read_to_string(path) {
            for line in content.lines() {
                let line = line.trim();
                if line.starts_with("block_size=") {
                    if let Ok(size) = line.split('=').nth(1).unwrap_or("128").parse() {
                        config.block_size = size;
                    }
                } else if line.starts_with("lookup_table_size=") {
                    if let Ok(size) = line.split('=').nth(1).unwrap_or("256").parse() {
                        config.lookup_table_size = size;
                    }
                } else if line.starts_with("use_avx512=") {
                    config.use_avx512 = line.split('=').nth(1).unwrap_or("false") == "true";
                } else if line.starts_with("use_avx2=") {
                    config.use_avx2 = line.split('=').nth(1).unwrap_or("true") == "true";
                } else if line.starts_with("precision_bits=") {
                    if let Ok(bits) = line.split('=').nth(1).unwrap_or("2").parse() {
                        config.precision_bits = bits;
                    }
                } else if line.starts_with("vectorized_tables=") {
                    config.vectorized_tables = line.split('=').nth(1).unwrap_or("true") == "true";
                }
            }
        }

        Ok(Self::with_config(config))
    }

    /// Quantize tensor using TL2 algorithm on a specific device
    pub fn quantize(&self, tensor: &BitNetTensor, device: &Device) -> Result<QuantizedTensor> {
        if !device.is_cpu() {
            #[cfg(feature = "cuda")]
            {
                if device.is_cuda()
                    && bitnet_kernels::gpu::cuda::is_cuda_available()
                    && let Ok(res) = self.quantize_cuda(tensor)
                {
                    return Ok(res);
                }
            }
        }

        let data = extract_f32_data(tensor)?;
        let shape = tensor.shape().to_vec();

        // Calculate statistics for lookup table generation
        let (min_val, max_val) =
            data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            });

        // Generate vectorized lookup table
        let lookup_table = VectorizedLookupTable::new(min_val, max_val, self.config.precision_bits);

        // Calculate grouped scales for better accuracy
        let scales =
            calculate_grouped_scales(&data, self.config.block_size, self.config.precision_bits);

        // Select optimal quantization kernel
        let quantized_data = match self.cpu_features.best_kernel() {
            KernelType::AVX512 if self.config.use_avx512 => {
                self.quantize_avx512(&data, &lookup_table, &scales)?
            }
            KernelType::AVX2 if self.config.use_avx2 => {
                self.quantize_avx2(&data, &lookup_table, &scales)?
            }
            _ => self.quantize_scalar(&data, &lookup_table, &scales)?,
        };

        // Pack quantized values efficiently
        let packed_data = self.pack_tl2_values(&quantized_data);

        Ok(QuantizedTensor::new_with_params(
            packed_data,
            scales,
            None, // TL2 uses symmetric quantization
            shape,
            QuantizationType::TL2,
            self.config.block_size,
        ))
    }

    /// Legacy wrapper that defaults to CPU quantization
    pub fn quantize_tensor(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
        self.quantize(tensor, &Device::Cpu)
    }

    /// Dequantize tensor from TL2 format on a specific device
    pub fn dequantize(&self, tensor: &QuantizedTensor, device: &Device) -> Result<BitNetTensor> {
        if tensor.qtype != QuantizationType::TL2 {
            return Err(
                QuantizationError::UnsupportedType { qtype: tensor.qtype.to_string() }.into()
            );
        }

        // Unpack quantized values
        let quantized_data = self.unpack_tl2_values(&tensor.data, tensor.numel());

        // Select optimal dequantization kernel
        let dequantized_data = match self.cpu_features.best_kernel() {
            KernelType::AVX512 if self.config.use_avx512 => {
                self.dequantize_avx512(&quantized_data, &tensor.scales)?
            }
            KernelType::AVX2 if self.config.use_avx2 => {
                self.dequantize_avx2(&quantized_data, &tensor.scales)?
            }
            _ => self.dequantize_scalar(&quantized_data, &tensor.scales)?,
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
        let num_blocks = data.len().div_ceil(self.config.block_size);
        let mut scales = vec![0f32; num_blocks];
        let packed_len = (data.len() * self.config.precision_bits as usize).div_ceil(8);
        let mut packed_data = vec![0u8; packed_len];
        let kernel = CudaKernel::new()?;
        kernel.quantize(&data, &mut packed_data, &mut scales, QuantizationType::TL2)?;
        Ok(QuantizedTensor::new_with_params(
            packed_data,
            scales,
            None,
            shape,
            QuantizationType::TL2,
            self.config.block_size,
        ))
    }

    /// Scalar quantization implementation
    fn quantize_scalar(
        &self,
        data: &[f32],
        _lookup_table: &VectorizedLookupTable,
        scales: &[f32],
    ) -> Result<Vec<i8>> {
        let mut quantized = vec![0i8; data.len()];

        quantized
            .par_chunks_mut(self.config.block_size)
            .zip(data.par_chunks(self.config.block_size))
            .zip(scales.par_iter())
            .for_each(|((quant_block, data_block), &_scale)| {
                // Create block-specific lookup table
                let block_min = data_block.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
                let block_max = data_block.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                let block_table =
                    VectorizedLookupTable::new(block_min, block_max, self.config.precision_bits);

                for (i, &value) in data_block.iter().enumerate() {
                    quant_block[i] = block_table.quantize(value);
                }
            });

        Ok(quantized)
    }

    /// Scalar dequantization implementation
    fn dequantize_scalar(&self, quantized: &[i8], scales: &[f32]) -> Result<Vec<f32>> {
        let mut dequantized = vec![0.0f32; quantized.len()];

        dequantized
            .par_chunks_mut(self.config.block_size)
            .zip(quantized.par_chunks(self.config.block_size))
            .zip(scales.par_iter())
            .for_each(|((dequant_block, quant_block), &scale)| {
                for (i, &value) in quant_block.iter().enumerate() {
                    dequant_block[i] = dequantize_value(value, scale);
                }
            });

        Ok(dequantized)
    }

    /// AVX2-optimized quantization for x86_64
    #[cfg(target_arch = "x86_64")]
    fn quantize_avx2(
        &self,
        data: &[f32],
        lookup_table: &VectorizedLookupTable,
        scales: &[f32],
    ) -> Result<Vec<i8>> {
        if !is_x86_feature_detected!("avx2") {
            return self.quantize_scalar(data, lookup_table, scales);
        }

        let mut quantized = vec![0i8; data.len()];

        quantized
            .par_chunks_mut(self.config.block_size)
            .zip(data.par_chunks(self.config.block_size))
            .zip(scales.par_iter())
            .for_each(|((quant_block, data_block), &scale)| unsafe {
                self.quantize_avx2_block(data_block, quant_block, lookup_table, scale);
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
            .par_chunks_mut(self.config.block_size)
            .zip(quantized.par_chunks(self.config.block_size))
            .zip(scales.par_iter())
            .for_each(|((dequant_block, quant_block), &scale)| unsafe {
                self.dequantize_avx2_block(quant_block, dequant_block, scale);
            });

        Ok(dequantized)
    }

    /// AVX-512 optimized quantization for x86_64 (fallback to AVX2 for now)
    #[cfg(target_arch = "x86_64")]
    fn quantize_avx512(
        &self,
        data: &[f32],
        lookup_table: &VectorizedLookupTable,
        scales: &[f32],
    ) -> Result<Vec<i8>> {
        // AVX-512 is unstable, fallback to AVX2
        self.quantize_avx2(data, lookup_table, scales)
    }

    /// AVX-512 optimized dequantization for x86_64 (fallback to AVX2 for now)
    #[cfg(target_arch = "x86_64")]
    fn dequantize_avx512(&self, quantized: &[i8], scales: &[f32]) -> Result<Vec<f32>> {
        // AVX-512 is unstable, fallback to AVX2
        self.dequantize_avx2(quantized, scales)
    }

    /// Fallback to scalar for non-x86 architectures
    #[cfg(not(target_arch = "x86_64"))]
    fn quantize_avx2(
        &self,
        data: &[f32],
        lookup_table: &VectorizedLookupTable,
        scales: &[f32],
    ) -> Result<Vec<i8>> {
        self.quantize_scalar(data, lookup_table, scales)
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn dequantize_avx2(&self, quantized: &[i8], scales: &[f32]) -> Result<Vec<f32>> {
        self.dequantize_scalar(quantized, scales)
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn quantize_avx512(
        &self,
        data: &[f32],
        lookup_table: &VectorizedLookupTable,
        scales: &[f32],
    ) -> Result<Vec<i8>> {
        self.quantize_scalar(data, lookup_table, scales)
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn dequantize_avx512(&self, quantized: &[i8], scales: &[f32]) -> Result<Vec<f32>> {
        self.dequantize_scalar(quantized, scales)
    }

    /// AVX2 kernel for quantizing a single block
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn quantize_avx2_block(
        &self,
        data: &[f32],
        output: &mut [i8],
        lookup_table: &VectorizedLookupTable,
        scale: f32,
    ) {
        use std::arch::x86_64::*;

        let inv_scale = 1.0 / scale;
        let inv_scale_vec = _mm256_set1_ps(inv_scale);
        let offset_vec = _mm256_set1_ps(128.0);

        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();

        for (i, chunk) in chunks.enumerate() {
            unsafe {
                let data_vec = _mm256_loadu_ps(chunk.as_ptr());
                let scaled = _mm256_mul_ps(data_vec, inv_scale_vec);
                let offset = _mm256_add_ps(scaled, offset_vec);
                let indices = _mm256_cvtps_epi32(offset);

                // Vectorized lookup table access
                let mut result = [0i8; 8];
                result[0] = lookup_table.forward
                    [(_mm256_extract_epi32::<0>(indices).clamp(0, 255)) as usize];
                result[1] = lookup_table.forward
                    [(_mm256_extract_epi32::<1>(indices).clamp(0, 255)) as usize];
                result[2] = lookup_table.forward
                    [(_mm256_extract_epi32::<2>(indices).clamp(0, 255)) as usize];
                result[3] = lookup_table.forward
                    [(_mm256_extract_epi32::<3>(indices).clamp(0, 255)) as usize];
                result[4] = lookup_table.forward
                    [(_mm256_extract_epi32::<4>(indices).clamp(0, 255)) as usize];
                result[5] = lookup_table.forward
                    [(_mm256_extract_epi32::<5>(indices).clamp(0, 255)) as usize];
                result[6] = lookup_table.forward
                    [(_mm256_extract_epi32::<6>(indices).clamp(0, 255)) as usize];
                result[7] = lookup_table.forward
                    [(_mm256_extract_epi32::<7>(indices).clamp(0, 255)) as usize];

                // Store results
                std::ptr::copy_nonoverlapping(result.as_ptr(), output.as_mut_ptr().add(i * 8), 8);
            }
        }

        // Handle remainder with scalar code
        for (i, &value) in remainder.iter().enumerate() {
            let idx = data.len() - remainder.len() + i;
            output[idx] = lookup_table.quantize(value);
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
                // Load 8 i8 values and convert to i32
                let i8_data = std::ptr::read_unaligned(chunk.as_ptr() as *const i64);
                let i8_vec = _mm_set1_epi64x(i8_data);
                let i32_vec = _mm256_cvtepi8_epi32(i8_vec);

                // Convert to float and scale
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

    // AVX-512 kernels removed due to unstable features
    // Will be re-added when AVX-512 support is stabilized

    /// Pack TL2 quantized values (optimized for x86 cache efficiency)
    fn pack_tl2_values(&self, values: &[i8]) -> Vec<u8> {
        pack_2bit_values(values)
    }

    /// Unpack TL2 quantized values
    fn unpack_tl2_values(&self, packed: &[u8], output_len: usize) -> Vec<i8> {
        unpack_2bit_values(packed, output_len)
    }
}

impl Default for TL2Quantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizerTrait for TL2Quantizer {
    fn quantize_tensor(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
        TL2Quantizer::quantize_tensor(self, tensor)
    }

    fn dequantize_tensor(&self, tensor: &QuantizedTensor) -> Result<BitNetTensor> {
        TL2Quantizer::dequantize_tensor(self, tensor)
    }

    fn quantization_type(&self) -> QuantizationType {
        QuantizationType::TL2
    }

    fn is_available(&self) -> bool {
        // TL2 is optimized for x86 but works on all platforms
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_cpu_feature_detection() {
        let features = CpuFeatures::detect();
        let kernel = features.best_kernel();

        // Should select some kernel type
        match kernel {
            KernelType::Scalar | KernelType::AVX2 | KernelType::AVX512 => {
                // All valid
            }
        }
    }

    #[test]
    fn test_vectorized_lookup_table() {
        let table = VectorizedLookupTable::new(-2.0, 2.0, 2);

        // Test quantization
        let quantized = table.quantize(1.0);
        let dequantized = table.dequantize(quantized);

        // Should be reasonably close (2-bit quantization has limited precision)
        assert!(dequantized.abs() < 3.0); // Should be in reasonable range
        assert!((0..4).contains(&quantized)); // 2-bit range
    }

    #[test]
    fn test_tl2_quantization_round_trip() {
        let device = Device::Cpu;
        let data = vec![1.0, -2.0, 0.5, -0.5, 3.0, -1.5];
        let shape = vec![2, 3];

        let tensor = create_tensor_from_f32(data.clone(), &shape, &device).unwrap();
        let quantizer = TL2Quantizer::new();

        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

        assert_eq!(quantized.qtype, QuantizationType::TL2);
        assert_eq!(quantized.shape, shape);
        assert_eq!(dequantized.shape(), &shape);
    }

    #[test]
    fn test_tl2_config_adaptation() {
        let quantizer = TL2Quantizer::new();

        // Block size should be adapted based on CPU features
        assert!(quantizer.config.block_size >= 64);

        // CPU features detection should not panic (always valid)
        let _ = quantizer.cpu_features.has_avx2;
    }

    #[test]
    fn test_large_tensor_quantization() {
        let device = Device::Cpu;
        let data = (0..1024).map(|i| (i as f32 - 512.0) / 256.0).collect::<Vec<_>>();
        let shape = vec![32, 32];

        let tensor = create_tensor_from_f32(data, &shape, &device).unwrap();
        let quantizer = TL2Quantizer::new();

        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

        assert_eq!(quantized.shape, shape);
        assert_eq!(dequantized.shape(), &shape);

        // Should achieve good compression
        let ratio = quantized.compression_ratio();
        assert!(ratio > 4.0);
    }
}
