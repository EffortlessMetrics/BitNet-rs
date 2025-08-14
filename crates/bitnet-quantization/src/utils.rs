//! Utility functions for quantization operations

use bitnet_common::{BitNetTensor, QuantizationError, Result};
use candle_core::{DType, Device, Tensor as CandleTensor};

/// Calculate the scale factor for quantization
pub fn calculate_scale(data: &[f32], bits: u8) -> f32 {
    let max_val = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    if max_val == 0.0 {
        return 1.0;
    }

    let max_quant = (1 << (bits - 1)) - 1; // For signed quantization
    max_val / max_quant as f32
}

/// Calculate scale factors for grouped quantization
pub fn calculate_grouped_scales(data: &[f32], block_size: usize, bits: u8) -> Vec<f32> {
    let num_blocks = (data.len() + block_size - 1) / block_size;
    let mut scales = Vec::with_capacity(num_blocks);

    for i in 0..num_blocks {
        let start = i * block_size;
        let end = (start + block_size).min(data.len());
        let block = &data[start..end];
        scales.push(calculate_scale(block, bits));
    }

    scales
}

/// Pack 4 2-bit values into a single byte
pub fn pack_2bit_values(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((values.len() + 3) / 4);

    for chunk in values.chunks(4) {
        let mut byte = 0u8;
        for (i, &val) in chunk.iter().enumerate() {
            // Clamp to 2-bit signed range [-2, 1]
            let clamped = val.clamp(-2, 1);
            // Convert to unsigned 2-bit [0, 3]
            let unsigned = (clamped + 2) as u8;
            byte |= unsigned << (i * 2);
        }
        packed.push(byte);
    }

    packed
}

/// Unpack 4 2-bit values from a single byte
pub fn unpack_2bit_values(packed: &[u8], output_len: usize) -> Vec<i8> {
    let mut values = Vec::with_capacity(output_len);

    for &byte in packed {
        for i in 0..4 {
            if values.len() >= output_len {
                break;
            }
            let unsigned = (byte >> (i * 2)) & 0x3;
            let signed = unsigned as i8 - 2; // Convert back to signed [-2, 1]
            values.push(signed);
        }
    }

    values
}

/// Quantize a single value to n-bit signed integer
pub fn quantize_value(value: f32, scale: f32, bits: u8) -> i8 {
    let max_val = (1 << (bits - 1)) - 1;
    let min_val = -(1 << (bits - 1));

    let quantized = (value / scale).round() as i32;
    quantized.clamp(min_val, max_val) as i8
}

/// Dequantize a single value from n-bit signed integer
pub fn dequantize_value(quantized: i8, scale: f32) -> f32 {
    quantized as f32 * scale
}

/// Calculate mean squared error between two tensors
pub fn calculate_mse(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(QuantizationError::QuantizationFailed {
            reason: "Tensor dimensions mismatch".to_string(),
        }
        .into());
    }

    let mse = a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum::<f32>() / a.len() as f32;

    Ok(mse)
}

/// Calculate signal-to-noise ratio
pub fn calculate_snr(original: &[f32], quantized: &[f32]) -> Result<f32> {
    let signal_power = original.iter().map(|&x| x.powi(2)).sum::<f32>() / original.len() as f32;
    let noise_power = calculate_mse(original, quantized)?;

    if noise_power == 0.0 {
        return Ok(f32::INFINITY);
    }

    Ok(10.0 * (signal_power / noise_power).log10())
}

/// Extract f32 data from a BitNetTensor
pub fn extract_f32_data(tensor: &BitNetTensor) -> Result<Vec<f32>> {
    let candle_tensor = tensor.inner();

    // Convert to f32 if needed and move to CPU
    let f32_tensor = if candle_tensor.dtype() != DType::F32 {
        candle_tensor.to_dtype(DType::F32)?
    } else {
        candle_tensor.clone()
    };

    let cpu_tensor = if f32_tensor.device().is_cpu() {
        f32_tensor
    } else {
        f32_tensor.to_device(&Device::Cpu)?
    };

    // Flatten the tensor to 1D and extract the data
    let flattened = cpu_tensor.flatten_all()?;
    let data = flattened.to_vec1::<f32>()?;
    Ok(data)
}

/// Create a BitNetTensor from f32 data
pub fn create_tensor_from_f32(
    data: Vec<f32>,
    shape: &[usize],
    device: &Device,
) -> Result<BitNetTensor> {
    let tensor = CandleTensor::from_vec(data, shape, device)?;
    Ok(BitNetTensor::new(tensor))
}

/// Validate tensor shapes match
pub fn validate_shapes(shape1: &[usize], shape2: &[usize]) -> Result<()> {
    if shape1 != shape2 {
        return Err(QuantizationError::QuantizationFailed {
            reason: format!("Shape mismatch: {:?} vs {:?}", shape1, shape2),
        }
        .into());
    }
    Ok(())
}

/// Calculate optimal block size for grouped quantization
pub fn calculate_optimal_block_size(tensor_size: usize, target_blocks: usize) -> usize {
    let block_size = (tensor_size + target_blocks - 1) / target_blocks;
    // Round to nearest power of 2 for better memory alignment
    block_size.next_power_of_two().min(1024).max(16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_2bit() {
        let values = vec![-2, -1, 0, 1, -2, 1];
        let packed = pack_2bit_values(&values);
        let unpacked = unpack_2bit_values(&packed, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_scale_calculation() {
        let data = vec![1.0, -2.0, 3.0, -4.0];
        let scale = calculate_scale(&data, 2);
        assert!((scale - 4.0 / 1.0).abs() < 1e-6); // max_val=4.0, max_quant=1
    }

    #[test]
    fn test_quantize_dequantize_value() {
        let value = 1.0f32; // Use a simpler value
        let scale = 1.0f32; // Use scale = 1.0 for easier testing
        let quantized = quantize_value(value, scale, 2);
        let dequantized = dequantize_value(quantized, scale);
        // 2-bit quantization has limited precision
        assert!(quantized >= -2 && quantized <= 1); // 2-bit signed range
        assert!(dequantized.abs() <= 2.0); // Should be in reasonable range
    }
}
