//! Int4 quantization for dense SLM (Small Language Model) weight compression.
//!
//! Provides group-wise symmetric and asymmetric int4 quantization with
//! NibblePacked storage (2 values per byte) achieving 4× size reduction
//! over FP32.

use crate::int8_quant::QuantError;

/// Configuration for int4 quantization.
#[derive(Debug, Clone)]
pub struct Int4QuantConfig {
    /// Number of elements per quantization group.
    pub group_size: usize,
    /// Symmetric (true) vs asymmetric (false) quantization.
    pub symmetric: bool,
    /// Block-wise (true) vs per-tensor (false) quantization.
    pub block_wise: bool,
}

impl Default for Int4QuantConfig {
    fn default() -> Self {
        Self { group_size: 128, symmetric: true, block_wise: true }
    }
}

/// Per-group quantization parameters for int4.
#[derive(Debug, Clone)]
pub struct Int4QuantParams {
    /// Scale factor per group.
    pub scales: Vec<f32>,
    /// Zero-points for asymmetric quantization (packed).
    pub zero_points: Vec<u8>,
    /// Group size used during quantization.
    pub group_size: usize,
    /// Number of groups.
    pub num_groups: usize,
    /// Whether symmetric quantization was used.
    pub symmetric: bool,
}

/// Packed int4 storage — two 4-bit signed values per byte.
///
/// Low nibble stores even-indexed values, high nibble stores odd-indexed values.
#[derive(Debug, Clone)]
pub struct NibblePacked {
    /// Packed nibble data.
    pub data: Vec<u8>,
    /// Number of int4 values stored.
    pub len: usize,
}

impl NibblePacked {
    /// Pack a slice of i8 values (expected range [-8, 7]) into nibble-packed storage.
    pub fn pack(values: &[i8]) -> Self {
        let byte_len = values.len().div_ceil(2);
        let mut data = vec![0u8; byte_len];
        for (i, &v) in values.iter().enumerate() {
            let nibble = (v & 0x0F) as u8;
            let byte_idx = i / 2;
            if i % 2 == 0 {
                data[byte_idx] = (data[byte_idx] & 0xF0) | nibble;
            } else {
                data[byte_idx] = (data[byte_idx] & 0x0F) | (nibble << 4);
            }
        }
        Self { data, len: values.len() }
    }

    /// Unpack all nibble-packed values back to i8.
    pub fn unpack(&self) -> Vec<i8> {
        let mut out = Vec::with_capacity(self.len);
        for i in 0..self.len {
            out.push(self.get(i));
        }
        out
    }

    /// Get a single int4 value by index, sign-extended to i8.
    pub fn get(&self, index: usize) -> i8 {
        assert!(index < self.len, "index {index} out of bounds (len={})", self.len);
        let byte_idx = index / 2;
        let raw = if index.is_multiple_of(2) {
            self.data[byte_idx] & 0x0F
        } else {
            (self.data[byte_idx] >> 4) & 0x0F
        };
        // Sign-extend from 4-bit: if bit 3 is set, value is negative
        if raw & 0x08 != 0 { (raw | 0xF0) as i8 } else { raw as i8 }
    }
}

/// Quantize an f32 tensor to int4 using group-wise quantization.
///
/// Returns packed nibble data and per-group quantization parameters.
pub fn quantize_tensor_int4(
    data: &[f32],
    config: &Int4QuantConfig,
) -> (NibblePacked, Int4QuantParams) {
    if data.is_empty() {
        return (
            NibblePacked { data: Vec::new(), len: 0 },
            Int4QuantParams {
                scales: vec![0.0],
                zero_points: vec![0],
                group_size: config.group_size,
                num_groups: 0,
                symmetric: config.symmetric,
            },
        );
    }

    let group_size = if config.block_wise { config.group_size.min(data.len()) } else { data.len() };
    let num_groups = data.len().div_ceil(group_size);

    let mut scales = Vec::with_capacity(num_groups);
    let mut zero_points = Vec::with_capacity(num_groups);
    let mut quantized = Vec::with_capacity(data.len());

    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(data.len());
        let group = &data[start..end];

        if config.symmetric {
            let abs_max = group.iter().copied().fold(0.0f32, |acc, v| acc.max(v.abs()));
            let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 7.0 };
            scales.push(scale);
            zero_points.push(0);

            for &v in group {
                let q = (v / scale).round().clamp(-8.0, 7.0) as i8;
                quantized.push(q);
            }
        } else {
            let min_val = group.iter().copied().fold(f32::INFINITY, f32::min);
            let max_val = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = max_val - min_val;
            let scale = if range == 0.0 { 1.0 } else { range / 15.0 };
            let zp = (-min_val / scale).round().clamp(0.0, 15.0) as u8;
            scales.push(scale);
            zero_points.push(zp);

            for &v in group {
                let q = ((v / scale) + zp as f32).round().clamp(0.0, 15.0) as i8;
                quantized.push(q);
            }
        }
    }

    let packed = NibblePacked::pack(&quantized);
    let params = Int4QuantParams {
        scales,
        zero_points,
        group_size,
        num_groups,
        symmetric: config.symmetric,
    };
    (packed, params)
}

/// Dequantize packed int4 data back to f32.
pub fn dequantize_tensor_int4(packed: &NibblePacked, params: &Int4QuantParams) -> Vec<f32> {
    if packed.len == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(packed.len);
    let values = packed.unpack();

    for (i, &v) in values.iter().enumerate() {
        let group_idx = i / params.group_size;
        let group_idx = group_idx.min(params.num_groups.saturating_sub(1));
        let scale = params.scales[group_idx];

        if params.symmetric {
            // Symmetric: signed value in [-8, 7]
            result.push(v as f32 * scale);
        } else {
            // Asymmetric: recover unsigned nibble [0, 15] from sign-extended i8
            let unsigned_val = (v as u8 & 0x0F) as f32;
            let zp = params.zero_points[group_idx] as f32;
            result.push((unsigned_val - zp) * scale);
        }
    }

    result
}

/// Compute quantization error metrics between original data and int4-quantized data.
pub fn compute_int4_error(
    original: &[f32],
    packed: &NibblePacked,
    params: &Int4QuantParams,
) -> QuantError {
    assert_eq!(original.len(), packed.len, "length mismatch");

    if original.is_empty() {
        return QuantError { max_abs_error: 0.0, mean_abs_error: 0.0, rmse: 0.0, snr_db: 0.0 };
    }

    let deq = dequantize_tensor_int4(packed, params);
    let n = original.len() as f32;

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut sum_sq_err = 0.0f32;
    let mut sum_sq_signal = 0.0f32;

    for (&o, &d) in original.iter().zip(deq.iter()) {
        let err = (o - d).abs();
        max_abs = max_abs.max(err);
        sum_abs += err;
        sum_sq_err += (o - d).powi(2);
        sum_sq_signal += o.powi(2);
    }

    let rmse = (sum_sq_err / n).sqrt();
    let snr_db =
        if sum_sq_err == 0.0 { f32::INFINITY } else { 10.0 * (sum_sq_signal / sum_sq_err).log10() };

    QuantError { max_abs_error: max_abs, mean_abs_error: sum_abs / n, rmse, snr_db }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::int8_quant;

    fn sym_config() -> Int4QuantConfig {
        Int4QuantConfig { group_size: 128, symmetric: true, block_wise: true }
    }

    // ---- NibblePacked: pack/unpack round-trip ----

    #[test]
    fn test_nibble_pack_unpack_even_length() {
        let values: Vec<i8> = vec![0, 1, -1, 7, -8, 3];
        let packed = NibblePacked::pack(&values);
        let unpacked = packed.unpack();
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_nibble_pack_unpack_odd_length() {
        let values: Vec<i8> = vec![0, 1, -1, 7, -8];
        let packed = NibblePacked::pack(&values);
        let unpacked = packed.unpack();
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_nibble_pack_unpack_single() {
        let values: Vec<i8> = vec![5];
        let packed = NibblePacked::pack(&values);
        assert_eq!(packed.len, 1);
        assert_eq!(packed.data.len(), 1);
        assert_eq!(packed.unpack(), values);
    }

    #[test]
    fn test_nibble_pack_unpack_empty() {
        let values: Vec<i8> = vec![];
        let packed = NibblePacked::pack(&values);
        assert_eq!(packed.len, 0);
        assert!(packed.data.is_empty());
        assert!(packed.unpack().is_empty());
    }

    // ---- NibblePacked: get individual elements ----

    #[test]
    fn test_nibble_get_elements() {
        let values: Vec<i8> = vec![-8, -4, 0, 3, 7, -1];
        let packed = NibblePacked::pack(&values);
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(packed.get(i), expected, "mismatch at index {i}");
        }
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_nibble_get_out_of_bounds() {
        let packed = NibblePacked::pack(&[1, 2, 3]);
        packed.get(3);
    }

    // ---- NibblePacked: special values ----

    #[test]
    fn test_nibble_all_zeros() {
        let values = vec![0i8; 16];
        let packed = NibblePacked::pack(&values);
        let unpacked = packed.unpack();
        assert_eq!(values, unpacked);
        assert!(packed.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_nibble_all_same_value() {
        let values = vec![5i8; 10];
        let packed = NibblePacked::pack(&values);
        let unpacked = packed.unpack();
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_nibble_mixed_positive_negative() {
        let values: Vec<i8> = vec![-8, 7, -7, 6, -5, 4, -3, 2, -1, 0];
        let packed = NibblePacked::pack(&values);
        let unpacked = packed.unpack();
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_nibble_boundary_values() {
        let values: Vec<i8> = vec![-8, 7]; // min and max int4
        let packed = NibblePacked::pack(&values);
        assert_eq!(packed.get(0), -8);
        assert_eq!(packed.get(1), 7);
    }

    // ---- Symmetric int4 quantization ----

    #[test]
    fn test_symmetric_basic() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let config = sym_config();
        let (packed, params) = quantize_tensor_int4(&data, &config);
        assert_eq!(packed.len, 5);
        assert_eq!(params.scales.len(), 1); // single group
        // Zero should remain zero
        let deq = dequantize_tensor_int4(&packed, &params);
        assert!((deq[2]).abs() < 1e-6);
    }

    #[test]
    fn test_symmetric_positive_only() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let config = sym_config();
        let (packed, params) = quantize_tensor_int4(&data, &config);
        let deq = dequantize_tensor_int4(&packed, &params);
        // Largest value should be close
        let max_err: f32 =
            data.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err <= params.scales[0] + 1e-6);
    }

    #[test]
    fn test_symmetric_negative_only() {
        let data = vec![-4.0, -3.0, -2.0, -1.0, 0.0];
        let config = sym_config();
        let (packed, params) = quantize_tensor_int4(&data, &config);
        let deq = dequantize_tensor_int4(&packed, &params);
        let max_err: f32 =
            data.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err <= params.scales[0] + 1e-6);
    }

    // ---- Group-wise quantization ----

    #[test]
    fn test_group_size_32() {
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let config = Int4QuantConfig { group_size: 32, symmetric: true, block_wise: true };
        let (packed, params) = quantize_tensor_int4(&data, &config);
        assert_eq!(params.num_groups, 2);
        assert_eq!(params.scales.len(), 2);
        assert_eq!(packed.len, 64);
    }

    #[test]
    fn test_group_size_64() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let config = Int4QuantConfig { group_size: 64, symmetric: true, block_wise: true };
        let (_packed, params) = quantize_tensor_int4(&data, &config);
        assert_eq!(params.num_groups, 4);
        assert_eq!(params.scales.len(), 4);
    }

    #[test]
    fn test_group_size_128() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let config = Int4QuantConfig { group_size: 128, symmetric: true, block_wise: true };
        let (_packed, params) = quantize_tensor_int4(&data, &config);
        assert_eq!(params.num_groups, 2);
        assert_eq!(params.scales.len(), 2);
    }

    // ---- Round-trip fidelity ----

    #[test]
    fn test_roundtrip_fidelity() {
        let data: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let config = sym_config();
        let (packed, params) = quantize_tensor_int4(&data, &config);
        let deq = dequantize_tensor_int4(&packed, &params);
        let max_err: f32 =
            data.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // Error within one quantization step
        assert!(max_err <= params.scales[0] + 1e-6, "max_err={max_err} scale={}", params.scales[0]);
    }

    // ---- Int4 value range clamping ----

    #[test]
    fn test_int4_value_range_clamping() {
        // Large values should clamp to [-8, 7]
        let data = vec![-1000.0, 0.0, 1000.0];
        let config = sym_config();
        let (packed, _params) = quantize_tensor_int4(&data, &config);
        let values = packed.unpack();
        for &v in &values {
            assert!((-8..=7).contains(&v), "value {v} out of int4 range");
        }
    }

    // ---- Config defaults ----

    #[test]
    fn test_config_defaults() {
        let cfg = Int4QuantConfig::default();
        assert_eq!(cfg.group_size, 128);
        assert!(cfg.symmetric);
        assert!(cfg.block_wise);
    }

    // ---- Error computation ----

    #[test]
    fn test_error_computation() {
        let data: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let config = sym_config();
        let (packed, params) = quantize_tensor_int4(&data, &config);
        let err = compute_int4_error(&data, &packed, &params);
        assert!(err.max_abs_error >= 0.0);
        assert!(err.mean_abs_error >= 0.0);
        assert!(err.rmse >= 0.0);
        assert!(err.snr_db > 0.0, "SNR should be positive for non-trivial data");
        assert!(err.mean_abs_error <= err.max_abs_error);
    }

    #[test]
    fn test_error_zeros() {
        let data = vec![0.0f32; 16];
        let config = sym_config();
        let (packed, params) = quantize_tensor_int4(&data, &config);
        let err = compute_int4_error(&data, &packed, &params);
        assert_eq!(err.max_abs_error, 0.0);
        assert_eq!(err.rmse, 0.0);
    }

    // ---- Edge cases ----

    #[test]
    fn test_single_element() {
        let data = vec![42.0];
        let config = sym_config();
        let (packed, params) = quantize_tensor_int4(&data, &config);
        assert_eq!(packed.len, 1);
        let deq = dequantize_tensor_int4(&packed, &params);
        assert!((deq[0] - 42.0).abs() < 7.0); // int4 has limited precision
    }

    #[test]
    fn test_group_size_larger_than_data() {
        let data = vec![1.0, 2.0, 3.0];
        let config = Int4QuantConfig { group_size: 1024, symmetric: true, block_wise: true };
        let (packed, params) = quantize_tensor_int4(&data, &config);
        assert_eq!(params.num_groups, 1);
        assert_eq!(packed.len, 3);
        let deq = dequantize_tensor_int4(&packed, &params);
        assert_eq!(deq.len(), 3);
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<f32> = vec![];
        let config = sym_config();
        let (packed, params) = quantize_tensor_int4(&data, &config);
        assert_eq!(packed.len, 0);
        assert_eq!(params.num_groups, 0);
        let deq = dequantize_tensor_int4(&packed, &params);
        assert!(deq.is_empty());
    }

    // ---- Large tensor ----

    #[test]
    fn test_large_tensor() {
        let data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let config = sym_config();
        let (packed, params) = quantize_tensor_int4(&data, &config);
        assert_eq!(packed.len, 1024);
        assert_eq!(params.num_groups, 8); // 1024 / 128
        let deq = dequantize_tensor_int4(&packed, &params);
        assert_eq!(deq.len(), 1024);
        let max_err: f32 =
            data.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // Reasonable error for int4
        assert!(max_err < 1.0, "max_err={max_err}");
    }

    // ---- Comparison with int8: int4 should have larger error ----

    #[test]
    fn test_int4_vs_int8_error() {
        let data: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();

        // Int4
        let config4 = sym_config();
        let (packed4, params4) = quantize_tensor_int4(&data, &config4);
        let err4 = compute_int4_error(&data, &packed4, &params4);

        // Int8
        let config8 = int8_quant::Int8QuantConfig {
            per_channel: false,
            symmetric: true,
            calibration_method: int8_quant::CalibrationMethod::MinMax,
        };
        let (q8, p8) = int8_quant::quantize_tensor_int8(&data, &config8);
        let err8 = int8_quant::compute_quantization_error(&data, &q8, &p8);

        // Int4 should have larger RMSE than int8
        assert!(
            err4.rmse > err8.rmse,
            "int4 rmse ({}) should exceed int8 rmse ({})",
            err4.rmse,
            err8.rmse
        );
    }

    // ---- Asymmetric quantization ----

    #[test]
    fn test_asymmetric_basic() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let config = Int4QuantConfig { group_size: 128, symmetric: false, block_wise: true };
        let (packed, params) = quantize_tensor_int4(&data, &config);
        assert_eq!(packed.len, 5);
        let deq = dequantize_tensor_int4(&packed, &params);
        let max_err: f32 =
            data.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err <= params.scales[0] + 1e-6, "max_err={max_err}");
    }

    // ---- Per-tensor mode (block_wise=false) ----

    #[test]
    fn test_per_tensor_mode() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let config = Int4QuantConfig { group_size: 32, symmetric: true, block_wise: false };
        let (packed, params) = quantize_tensor_int4(&data, &config);
        // block_wise=false → single group regardless of group_size
        assert_eq!(params.num_groups, 1);
        assert_eq!(packed.len, 256);
    }

    // ---- Nibble storage efficiency ----

    #[test]
    fn test_nibble_storage_efficiency() {
        let n = 100;
        let values = vec![3i8; n];
        let packed = NibblePacked::pack(&values);
        // 100 values → 50 bytes
        assert_eq!(packed.data.len(), 50);
        assert_eq!(packed.len, 100);
    }
}
