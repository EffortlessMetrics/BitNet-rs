//! TL1 (Table Lookup 1) quantization for ARM platforms
//!
//! This module implements table lookup quantization optimized for ARM NEON instructions.
//! It uses lookup tables to accelerate quantization and dequantization operations,
//! with configurable block sizes for optimal performance on ARM architectures.

use crate::{utils::*, QuantizedTensor, QuantizerTrait};
use bitnet_common::{BitNetTensor, QuantizationError, QuantizationType, Result, Tensor};
use candle_core::Device;
use rayon::prelude::*;
use std::collections::HashMap;

/// Configuration for TL1 quantization loaded from .ini files
#[derive(Debug, Clone)]
pub struct TL1Config {
    pub block_size: usize,
    pub lookup_table_size: usize,
    pub use_asymmetric: bool,
    pub precision_bits: u8,
}

impl Default for TL1Config {
    fn default() -> Self {
        Self {
            block_size: 64,
            lookup_table_size: 256,
            use_asymmetric: false,
            precision_bits: 2,
        }
    }
}

/// Lookup table for TL1 quantization
#[derive(Debug, Clone)]
pub struct LookupTable {
    /// Forward lookup: float range -> quantized value
    forward: Vec<i8>,
    /// Reverse lookup: quantized value -> float value
    reverse: Vec<f32>,
    /// Scale factor for this table
    scale: f32,
    /// Zero point for asymmetric quantization
    zero_point: i32,
}

impl LookupTable {
    /// Create a new lookup table for the given data range
    pub fn new(min_val: f32, max_val: f32, bits: u8, use_asymmetric: bool) -> Self {
        let num_levels = 1 << bits;
        let mut forward = vec![0i8; 256]; // Index by scaled float value
        let mut reverse = vec![0.0f32; num_levels];
        
        let (scale, zero_point) = if use_asymmetric {
            let scale = (max_val - min_val) / (num_levels - 1) as f32;
            let zero_point = (-min_val / scale).round() as i32;
            (scale, zero_point)
        } else {
            let abs_max = max_val.abs().max(min_val.abs());
            let scale = abs_max / ((num_levels / 2) - 1) as f32;
            (scale, 0)
        };
        
        // Build reverse lookup table
        for i in 0..num_levels {
            let quantized = if use_asymmetric {
                i as i32 - zero_point
            } else {
                i as i32 - (num_levels / 2) as i32
            };
            reverse[i] = quantized as f32 * scale;
        }
        
        // Build forward lookup table
        for i in 0..256 {
            let float_val = (i as f32 - 128.0) * scale; // Map [0,255] to float range
            let quantized = if use_asymmetric {
                ((float_val / scale + zero_point as f32).round() as i32)
                    .clamp(0, num_levels as i32 - 1) as i8
            } else {
                ((float_val / scale).round() as i32 + (num_levels / 2) as i32)
                    .clamp(0, num_levels as i32 - 1) as i8
            };
            forward[i] = quantized;
        }
        
        Self {
            forward,
            reverse,
            scale,
            zero_point,
        }
    }
    
    /// Quantize a value using the lookup table
    pub fn quantize(&self, value: f32) -> i8 {
        let index = ((value / self.scale + 128.0).round() as usize).clamp(0, 255);
        self.forward[index]
    }
    
    /// Dequantize a value using the lookup table
    pub fn dequantize(&self, quantized: i8) -> f32 {
        let index = quantized as usize;
        if index < self.reverse.len() {
            self.reverse[index]
        } else {
            0.0
        }
    }
}

/// TL1 quantization implementation optimized for ARM NEON
pub struct TL1Quantizer {
    config: TL1Config,
    lookup_tables: HashMap<String, LookupTable>,
    use_neon: bool,
}

impl TL1Quantizer {
    /// Create a new TL1 quantizer with default configuration
    pub fn new() -> Self {
        Self {
            config: TL1Config::default(),
            lookup_tables: HashMap::new(),
            use_neon: cfg!(target_arch = "aarch64"),
        }
    }

    /// Create a new TL1 quantizer with custom configuration
    pub fn with_config(config: TL1Config) -> Self {
        Self {
            config,
            lookup_tables: HashMap::new(),
            use_neon: cfg!(target_arch = "aarch64"),
        }
    }

    /// Load configuration from .ini file for compatibility
    pub fn from_ini_file(path: &str) -> Result<Self> {
        // Simplified ini parsing - in practice would use a proper ini parser
        let mut config = TL1Config::default();
        
        if let Ok(content) = std::fs::read_to_string(path) {
            for line in content.lines() {
                let line = line.trim();
                if line.starts_with("block_size=") {
                    if let Ok(size) = line.split('=').nth(1).unwrap_or("64").parse() {
                        config.block_size = size;
                    }
                } else if line.starts_with("lookup_table_size=") {
                    if let Ok(size) = line.split('=').nth(1).unwrap_or("256").parse() {
                        config.lookup_table_size = size;
                    }
                } else if line.starts_with("use_asymmetric=") {
                    config.use_asymmetric = line.split('=').nth(1).unwrap_or("false") == "true";
                } else if line.starts_with("precision_bits=") {
                    if let Ok(bits) = line.split('=').nth(1).unwrap_or("2").parse() {
                        config.precision_bits = bits;
                    }
                }
            }
        }
        
        Ok(Self::with_config(config))
    }

    /// Quantize tensor using TL1 algorithm
    pub fn quantize_tensor(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
        let data = extract_f32_data(tensor)?;
        let shape = tensor.shape().to_vec();
        
        // Calculate statistics for lookup table generation
        let (min_val, max_val) = data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), 
            |(min, max), &val| (min.min(val), max.max(val)));
        
        // Generate lookup table for this tensor
        let lookup_table = LookupTable::new(
            min_val, 
            max_val, 
            self.config.precision_bits, 
            self.config.use_asymmetric
        );
        
        // Calculate grouped scales
        let scales = calculate_grouped_scales(&data, self.config.block_size, self.config.precision_bits);
        
        // Quantize data using lookup tables
        let quantized_data = if self.use_neon {
            self.quantize_neon(&data, &lookup_table, &scales)?
        } else {
            self.quantize_scalar(&data, &lookup_table, &scales)?
        };
        
        // Pack quantized values
        let packed_data = self.pack_tl1_values(&quantized_data);
        
        let zero_points = if self.config.use_asymmetric { 
            Some(vec![lookup_table.zero_point; scales.len()]) 
        } else { 
            None 
        };
        
        Ok(QuantizedTensor::new_with_params(
            packed_data,
            scales,
            zero_points,
            shape,
            QuantizationType::TL1,
            self.config.block_size,
        ))
    }

    /// Dequantize tensor from TL1 format
    pub fn dequantize_tensor(&self, tensor: &QuantizedTensor) -> Result<BitNetTensor> {
        if tensor.qtype != QuantizationType::TL1 {
            return Err(QuantizationError::UnsupportedType {
                qtype: tensor.qtype.to_string(),
            }.into());
        }

        // Unpack quantized values
        let quantized_data = self.unpack_tl1_values(&tensor.data, tensor.numel());
        
        // Reconstruct lookup table from scales and zero points
        let default_zero_points = vec![0; tensor.scales.len()];
        let zero_points = tensor.zero_points.as_ref().unwrap_or(&default_zero_points);
        
        // Dequantize data
        let dequantized_data = if self.use_neon {
            self.dequantize_neon(&quantized_data, &tensor.scales, zero_points)?
        } else {
            self.dequantize_scalar(&quantized_data, &tensor.scales, zero_points)?
        };
        
        // Create tensor
        let device = Device::Cpu; // TODO: Support GPU devices
        create_tensor_from_f32(dequantized_data, &tensor.shape, &device)
    }

    /// Scalar quantization implementation
    fn quantize_scalar(
        &self, 
        data: &[f32], 
        _lookup_table: &LookupTable, 
        scales: &[f32]
    ) -> Result<Vec<i8>> {
        let mut quantized = vec![0i8; data.len()];
        
        quantized
            .par_chunks_mut(self.config.block_size)
            .zip(data.par_chunks(self.config.block_size))
            .zip(scales.par_iter())
            .for_each(|((quant_block, data_block), &scale)| {
                // Create block-specific lookup table
                let block_min = data_block.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
                let block_max = data_block.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                let block_table = LookupTable::new(
                    block_min, 
                    block_max, 
                    self.config.precision_bits, 
                    self.config.use_asymmetric
                );
                
                for (i, &value) in data_block.iter().enumerate() {
                    quant_block[i] = block_table.quantize(value);
                }
            });
        
        Ok(quantized)
    }

    /// Scalar dequantization implementation
    fn dequantize_scalar(
        &self, 
        quantized: &[i8], 
        scales: &[f32], 
        zero_points: &[i32]
    ) -> Result<Vec<f32>> {
        let mut dequantized = vec![0.0f32; quantized.len()];
        
        dequantized
            .par_chunks_mut(self.config.block_size)
            .zip(quantized.par_chunks(self.config.block_size))
            .zip(scales.par_iter())
            .zip(zero_points.par_iter())
            .for_each(|(((dequant_block, quant_block), &scale), &zero_point)| {
                for (i, &value) in quant_block.iter().enumerate() {
                    let adjusted = if self.config.use_asymmetric {
                        value as i32 - zero_point
                    } else {
                        value as i32
                    };
                    dequant_block[i] = adjusted as f32 * scale;
                }
            });
        
        Ok(dequantized)
    }

    /// NEON-optimized quantization for ARM64
    #[cfg(target_arch = "aarch64")]
    fn quantize_neon(
        &self, 
        data: &[f32], 
        lookup_table: &LookupTable, 
        scales: &[f32]
    ) -> Result<Vec<i8>> {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return self.quantize_scalar(data, lookup_table, scales);
        }

        let mut quantized = vec![0i8; data.len()];
        
        quantized
            .par_chunks_mut(self.config.block_size)
            .zip(data.par_chunks(self.config.block_size))
            .zip(scales.par_iter())
            .for_each(|((quant_block, data_block), &scale)| {
                unsafe {
                    self.quantize_neon_block(data_block, quant_block, lookup_table, scale);
                }
            });
        
        Ok(quantized)
    }

    /// NEON-optimized dequantization for ARM64
    #[cfg(target_arch = "aarch64")]
    fn dequantize_neon(
        &self, 
        quantized: &[i8], 
        scales: &[f32], 
        zero_points: &[i32]
    ) -> Result<Vec<f32>> {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return self.dequantize_scalar(quantized, scales, zero_points);
        }

        let mut dequantized = vec![0.0f32; quantized.len()];
        
        dequantized
            .par_chunks_mut(self.config.block_size)
            .zip(quantized.par_chunks(self.config.block_size))
            .zip(scales.par_iter())
            .zip(zero_points.par_iter())
            .for_each(|(((dequant_block, quant_block), &scale), &zero_point)| {
                unsafe {
                    self.dequantize_neon_block(quant_block, dequant_block, scale, zero_point);
                }
            });
        
        Ok(dequantized)
    }

    /// Fallback to scalar for non-ARM architectures
    #[cfg(not(target_arch = "aarch64"))]
    fn quantize_neon(
        &self, 
        data: &[f32], 
        lookup_table: &LookupTable, 
        scales: &[f32]
    ) -> Result<Vec<i8>> {
        self.quantize_scalar(data, lookup_table, scales)
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn dequantize_neon(
        &self, 
        quantized: &[i8], 
        scales: &[f32], 
        zero_points: &[i32]
    ) -> Result<Vec<f32>> {
        self.dequantize_scalar(quantized, scales, zero_points)
    }

    /// NEON kernel for quantizing a single block
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn quantize_neon_block(
        &self, 
        data: &[f32], 
        output: &mut [i8], 
        lookup_table: &LookupTable, 
        scale: f32
    ) {
        use std::arch::aarch64::*;
        
        let inv_scale = 1.0 / scale;
        let inv_scale_vec = vdupq_n_f32(inv_scale);
        let offset_vec = vdupq_n_f32(128.0);
        
        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for (i, chunk) in chunks.enumerate() {
            let data_vec = vld1q_f32(chunk.as_ptr());
            let scaled = vmulq_f32(data_vec, inv_scale_vec);
            let offset = vaddq_f32(scaled, offset_vec);
            let indices = vcvtq_u32_f32(offset);
            
            // Use lookup table for each element
            let mut result = [0i8; 4];
            for j in 0..4 {
                let idx = vgetq_lane_u32(indices, j as i32).min(255) as usize;
                result[j] = lookup_table.forward[idx];
            }
            
            // Store results
            std::ptr::copy_nonoverlapping(
                result.as_ptr(),
                output.as_mut_ptr().add(i * 4),
                4,
            );
        }
        
        // Handle remainder with scalar code
        for (i, &value) in remainder.iter().enumerate() {
            let idx = data.len() - remainder.len() + i;
            output[idx] = lookup_table.quantize(value);
        }
    }

    /// NEON kernel for dequantizing a single block
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn dequantize_neon_block(
        &self, 
        quantized: &[i8], 
        output: &mut [f32], 
        scale: f32, 
        zero_point: i32
    ) {
        use std::arch::aarch64::*;
        
        let scale_vec = vdupq_n_f32(scale);
        let zero_point_vec = vdupq_n_s32(zero_point);
        
        let chunks = quantized.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for (i, chunk) in chunks.enumerate() {
            // Load 4 i8 values and convert to i32
            let i8_data = std::ptr::read_unaligned(chunk.as_ptr() as *const u32);
            let i8_vec = vreinterpret_s8_u32(vdup_n_u32(i8_data));
            let i16_vec = vmovl_s8(i8_vec);
            let i32_vec = vmovl_s16(vget_low_s16(i16_vec));
            
            // Apply zero point adjustment if using asymmetric quantization
            let adjusted = if self.config.use_asymmetric {
                vsubq_s32(i32_vec, zero_point_vec)
            } else {
                i32_vec
            };
            
            // Convert to float and scale
            let f32_vec = vcvtq_f32_s32(adjusted);
            let result = vmulq_f32(f32_vec, scale_vec);
            
            vst1q_f32(output.as_mut_ptr().add(i * 4), result);
        }
        
        // Handle remainder with scalar code
        for (i, &value) in remainder.iter().enumerate() {
            let idx = quantized.len() - remainder.len() + i;
            let adjusted = if self.config.use_asymmetric {
                value as i32 - zero_point
            } else {
                value as i32
            };
            output[idx] = adjusted as f32 * scale;
        }
    }

    /// Pack TL1 quantized values (2-bit packing similar to I2_S)
    fn pack_tl1_values(&self, values: &[i8]) -> Vec<u8> {
        pack_2bit_values(values)
    }

    /// Unpack TL1 quantized values
    fn unpack_tl1_values(&self, packed: &[u8], output_len: usize) -> Vec<i8> {
        unpack_2bit_values(packed, output_len)
    }
}

impl Default for TL1Quantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizerTrait for TL1Quantizer {
    fn quantize_tensor(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
        self.quantize_tensor(tensor)
    }

    fn dequantize_tensor(&self, tensor: &QuantizedTensor) -> Result<BitNetTensor> {
        self.dequantize_tensor(tensor)
    }

    fn quantization_type(&self) -> QuantizationType {
        QuantizationType::TL1
    }

    fn is_available(&self) -> bool {
        // TL1 is optimized for ARM but works on all platforms
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_lookup_table_creation() {
        let table = LookupTable::new(-2.0, 2.0, 2, false);
        
        // Test quantization
        assert_eq!(table.quantize(0.0), 2); // Should map to middle value
        assert_eq!(table.quantize(2.0), 3);  // Should map to max value
        assert_eq!(table.quantize(-2.0), 0); // Should map to min value
        
        // Test dequantization
        assert!((table.dequantize(2) - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_tl1_quantization_round_trip() {
        let device = Device::Cpu;
        let data = vec![1.0, -2.0, 0.5, -0.5, 3.0, -1.5];
        let shape = vec![2, 3];
        
        let tensor = create_tensor_from_f32(data.clone(), &shape, &device).unwrap();
        let quantizer = TL1Quantizer::new();
        
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
        
        assert_eq!(quantized.qtype, QuantizationType::TL1);
        assert_eq!(quantized.shape, shape);
        assert_eq!(dequantized.shape(), &shape);
    }

    #[test]
    fn test_tl1_config_loading() {
        // Test default config
        let quantizer = TL1Quantizer::new();
        assert_eq!(quantizer.config.block_size, 64);
        assert_eq!(quantizer.config.precision_bits, 2);
        
        // Test custom config
        let config = TL1Config {
            block_size: 128,
            lookup_table_size: 512,
            use_asymmetric: true,
            precision_bits: 3,
        };
        let quantizer = TL1Quantizer::with_config(config.clone());
        assert_eq!(quantizer.config.block_size, 128);
        assert_eq!(quantizer.config.use_asymmetric, true);
    }

    #[test]
    fn test_asymmetric_quantization() {
        let device = Device::Cpu;
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]; // All positive values
        let shape = vec![6];
        
        let tensor = create_tensor_from_f32(data, &shape, &device).unwrap();
        
        let config = TL1Config {
            use_asymmetric: true,
            ..Default::default()
        };
        let quantizer = TL1Quantizer::with_config(config);
        
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
        
        assert!(quantized.zero_points.is_some());
        assert_eq!(dequantized.shape(), &shape);
    }
}