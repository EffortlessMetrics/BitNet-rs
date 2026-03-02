//! Int8 quantization for dense SLM (Small Language Model) weight compression.
//!
//! Provides symmetric and asymmetric int8 quantization with per-tensor and
//! per-channel modes, plus calibration methods (MinMax, Percentile, MSE).

/// Calibration method for determining quantization range.
#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationMethod {
    /// Use the full min/max range of the data.
    MinMax,
    /// Clip to a percentile of the data distribution (e.g., 99.9).
    Percentile(f32),
    /// Minimize mean-squared error between original and quantized values.
    MSE,
}

/// Configuration for int8 quantization.
#[derive(Debug, Clone)]
pub struct Int8QuantConfig {
    /// Per-channel (true) vs per-tensor (false) quantization.
    pub per_channel: bool,
    /// Symmetric (true) vs asymmetric (false) quantization.
    pub symmetric: bool,
    /// Method used to determine quantization range.
    pub calibration_method: CalibrationMethod,
}

impl Default for Int8QuantConfig {
    fn default() -> Self {
        Self { per_channel: true, symmetric: true, calibration_method: CalibrationMethod::MinMax }
    }
}

/// Per-tensor or per-channel quantization parameters.
#[derive(Debug, Clone)]
pub struct Int8QuantParams {
    /// Scale factor(s): one per channel or a single element.
    pub scales: Vec<f32>,
    /// Zero-point(s) for asymmetric quantization.
    pub zero_points: Vec<i8>,
    /// Observed minimum value(s).
    pub min_vals: Vec<f32>,
    /// Observed maximum value(s).
    pub max_vals: Vec<f32>,
}

/// Error metrics for quantization quality assessment.
#[derive(Debug, Clone)]
pub struct QuantError {
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
    pub rmse: f32,
    pub snr_db: f32,
}

// ---------------------------------------------------------------------------
// Calibration helpers
// ---------------------------------------------------------------------------

/// Compute the effective (min, max) for a data slice using the given method.
fn calibrated_range(data: &[f32], method: &CalibrationMethod) -> (f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0);
    }

    match method {
        CalibrationMethod::MinMax => {
            let min = data.iter().copied().fold(f32::INFINITY, f32::min);
            let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (min, max)
        }
        CalibrationMethod::Percentile(p) => {
            let p = p.clamp(0.0, 100.0);
            let mut sorted: Vec<f32> = data.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let lo_idx = (((100.0 - p) / 100.0) * (sorted.len() - 1) as f32).round() as usize;
            let hi_idx = ((p / 100.0) * (sorted.len() - 1) as f32).round() as usize;
            let lo_idx = lo_idx.min(sorted.len() - 1);
            let hi_idx = hi_idx.min(sorted.len() - 1);
            (sorted[lo_idx], sorted[hi_idx])
        }
        CalibrationMethod::MSE => {
            // Start from MinMax, then iteratively shrink the range to minimize MSE.
            let abs_max = data.iter().copied().fold(0.0f32, |acc, v| acc.max(v.abs()));
            let min_full = data.iter().copied().fold(f32::INFINITY, f32::min);
            let max_full = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            let steps = 20u32;
            let mut best_min = min_full;
            let mut best_max = max_full;
            let mut best_mse = f32::MAX;

            for i in 0..=steps {
                let shrink = i as f32 / steps as f32 * 0.2; // shrink up to 20 %
                let candidate_max = abs_max * (1.0 - shrink);
                let candidate_min = -candidate_max; // symmetric probe for MSE

                let scale = if candidate_max == 0.0 { 1.0 } else { candidate_max / 127.0 };

                let mse: f32 = data
                    .iter()
                    .map(|&v| {
                        let clamped = v.clamp(candidate_min, candidate_max);
                        let q = (clamped / scale).round().clamp(-128.0, 127.0);
                        let deq = q * scale;
                        (v - deq).powi(2)
                    })
                    .sum::<f32>()
                    / data.len() as f32;

                if mse < best_mse {
                    best_mse = mse;
                    best_min = candidate_min;
                    best_max = candidate_max;
                }
            }
            (best_min, best_max)
        }
    }
}

// ---------------------------------------------------------------------------
// Core quantize / dequantize
// ---------------------------------------------------------------------------

/// Quantize an f32 tensor to int8 using the given configuration (per-tensor).
pub fn quantize_tensor_int8(data: &[f32], config: &Int8QuantConfig) -> (Vec<i8>, Int8QuantParams) {
    if data.is_empty() {
        return (
            Vec::new(),
            Int8QuantParams {
                scales: vec![0.0],
                zero_points: vec![0],
                min_vals: vec![0.0],
                max_vals: vec![0.0],
            },
        );
    }

    let (min_val, max_val) = calibrated_range(data, &config.calibration_method);

    let (scale, zero_point) = if config.symmetric {
        let abs_max = min_val.abs().max(max_val.abs());
        let s = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
        (s, 0i8)
    } else {
        let range = max_val - min_val;
        let s = if range == 0.0 { 1.0 } else { range / 255.0 };
        let zp = (-min_val / s).round() - 128.0;
        let zp = zp.clamp(-128.0, 127.0) as i8;
        (s, zp)
    };

    let quantized: Vec<i8> = data
        .iter()
        .map(|&v| {
            if config.symmetric {
                (v / scale).round().clamp(-128.0, 127.0) as i8
            } else {
                let q = (v / scale).round() + (zero_point as f32 + 128.0);
                // Map to [-128, 127]: subtract 128 back
                (q - 128.0).clamp(-128.0, 127.0).round() as i8
            }
        })
        .collect();

    let params = Int8QuantParams {
        scales: vec![scale],
        zero_points: vec![zero_point],
        min_vals: vec![min_val],
        max_vals: vec![max_val],
    };
    (quantized, params)
}

/// Dequantize int8 data back to f32.
pub fn dequantize_tensor_int8(data: &[i8], params: &Int8QuantParams) -> Vec<f32> {
    if data.is_empty() {
        return Vec::new();
    }

    let scale = params.scales[0];
    let zero_point = params.zero_points[0];

    data.iter().map(|&v| ((v as i32 - zero_point as i32) as f32) * scale).collect()
}

// ---------------------------------------------------------------------------
// Per-channel quantization
// ---------------------------------------------------------------------------

/// Quantize per-channel along `channel_dim` of a tensor with the given `shape`.
///
/// `data` is stored in row-major order.
pub fn quantize_per_channel(
    data: &[f32],
    shape: &[usize],
    channel_dim: usize,
    config: &Int8QuantConfig,
) -> (Vec<i8>, Int8QuantParams) {
    assert!(channel_dim < shape.len(), "channel_dim out of range");
    let total: usize = shape.iter().product();
    assert_eq!(data.len(), total, "data length must match shape product");

    let num_channels = shape[channel_dim];

    // Compute strides
    let mut strides: Vec<usize> = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let channel_stride = strides[channel_dim];
    let elements_per_channel = total / num_channels;

    // Gather indices for each channel
    let mut channel_data: Vec<Vec<f32>> =
        vec![Vec::with_capacity(elements_per_channel); num_channels];
    for (idx, &val) in data.iter().enumerate() {
        let c = (idx / channel_stride) % num_channels;
        channel_data[c].push(val);
    }

    let mut all_scales = Vec::with_capacity(num_channels);
    let mut all_zps = Vec::with_capacity(num_channels);
    let mut all_mins = Vec::with_capacity(num_channels);
    let mut all_maxs = Vec::with_capacity(num_channels);

    let per_tensor_cfg = Int8QuantConfig {
        per_channel: false,
        symmetric: config.symmetric,
        calibration_method: config.calibration_method.clone(),
    };

    // Quantize each channel independently
    let mut channel_quantized: Vec<Vec<i8>> = Vec::with_capacity(num_channels);
    for ch_data in &channel_data {
        let (q, p) = quantize_tensor_int8(ch_data, &per_tensor_cfg);
        all_scales.push(p.scales[0]);
        all_zps.push(p.zero_points[0]);
        all_mins.push(p.min_vals[0]);
        all_maxs.push(p.max_vals[0]);
        channel_quantized.push(q);
    }

    // Scatter back to original layout
    let mut result = vec![0i8; total];
    let mut channel_pos = vec![0usize; num_channels];
    for (idx, out) in result.iter_mut().enumerate() {
        let c = (idx / channel_stride) % num_channels;
        *out = channel_quantized[c][channel_pos[c]];
        channel_pos[c] += 1;
    }

    let params = Int8QuantParams {
        scales: all_scales,
        zero_points: all_zps,
        min_vals: all_mins,
        max_vals: all_maxs,
    };
    (result, params)
}

// ---------------------------------------------------------------------------
// Error metrics
// ---------------------------------------------------------------------------

/// Compute quantization error metrics between the original f32 data and
/// quantized int8 data (using the provided dequantization parameters).
pub fn compute_quantization_error(
    original: &[f32],
    quantized: &[i8],
    params: &Int8QuantParams,
) -> QuantError {
    assert_eq!(original.len(), quantized.len(), "length mismatch");

    if original.is_empty() {
        return QuantError { max_abs_error: 0.0, mean_abs_error: 0.0, rmse: 0.0, snr_db: 0.0 };
    }

    let deq = dequantize_tensor_int8(quantized, params);
    let n = original.len() as f32;

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut sum_sq_err = 0.0f32;
    let mut sum_sq_signal = 0.0f32;

    for ((&o, &d), _) in original.iter().zip(deq.iter()).zip(quantized.iter()) {
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

    fn sym_config() -> Int8QuantConfig {
        Int8QuantConfig { per_channel: false, symmetric: true, ..Default::default() }
    }

    fn asym_config() -> Int8QuantConfig {
        Int8QuantConfig {
            per_channel: false,
            symmetric: false,
            calibration_method: CalibrationMethod::MinMax,
        }
    }

    // ---- Config defaults ----

    #[test]
    fn test_config_defaults() {
        let cfg = Int8QuantConfig::default();
        assert!(cfg.per_channel);
        assert!(cfg.symmetric);
        assert_eq!(cfg.calibration_method, CalibrationMethod::MinMax);
    }

    // ---- Symmetric quantization ----

    #[test]
    fn test_symmetric_zeros() {
        let data = vec![0.0f32; 8];
        let (q, p) = quantize_tensor_int8(&data, &sym_config());
        assert!(q.iter().all(|&v| v == 0));
        assert_eq!(p.zero_points[0], 0);
    }

    #[test]
    fn test_symmetric_positive_only() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let (q, p) = quantize_tensor_int8(&data, &sym_config());
        assert!(p.scales[0] > 0.0);
        // Largest value should map close to 127
        assert_eq!(*q.last().unwrap(), 127);
        assert_eq!(p.zero_points[0], 0);
    }

    #[test]
    fn test_symmetric_negative_only() {
        let data = vec![-4.0, -3.0, -2.0, -1.0, 0.0];
        let (q, p) = quantize_tensor_int8(&data, &sym_config());
        // Most negative should map close to -127
        assert_eq!(q[0], -127);
        assert_eq!(p.zero_points[0], 0);
    }

    #[test]
    fn test_symmetric_mixed() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let (q, _p) = quantize_tensor_int8(&data, &sym_config());
        // Zero should remain zero
        assert_eq!(q[2], 0);
        // Extremes should be symmetric
        assert_eq!(q[0], -q[4]);
    }

    // ---- Asymmetric quantization ----

    #[test]
    fn test_asymmetric_zeros() {
        let data = vec![0.0f32; 8];
        let (q, _p) = quantize_tensor_int8(&data, &asym_config());
        // All zeros should quantize to the same value
        assert!(q.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn test_asymmetric_positive_only() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let (_q, p) = quantize_tensor_int8(&data, &asym_config());
        assert!(p.scales[0] > 0.0);
    }

    #[test]
    fn test_asymmetric_negative_only() {
        let data = vec![-4.0, -3.0, -2.0, -1.0, 0.0];
        let (_q, p) = quantize_tensor_int8(&data, &asym_config());
        assert!(p.scales[0] > 0.0);
    }

    #[test]
    fn test_asymmetric_mixed() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let (q, p) = quantize_tensor_int8(&data, &asym_config());
        // Verify quantized values are in valid int8 range
        assert!(q.iter().all(|&v| (-128..=127).contains(&(v as i16))));
        assert!(p.scales[0] > 0.0);
    }

    // ---- Round-trip fidelity ----

    #[test]
    fn test_roundtrip_symmetric() {
        let data: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let cfg = sym_config();
        let (q, p) = quantize_tensor_int8(&data, &cfg);
        let deq = dequantize_tensor_int8(&q, &p);
        let max_err: f32 =
            data.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // Error should be within one quantization step
        assert!(max_err <= p.scales[0] + 1e-6, "max_err={max_err} scale={}", p.scales[0]);
    }

    #[test]
    fn test_roundtrip_asymmetric() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.05).collect();
        let cfg = asym_config();
        let (q, p) = quantize_tensor_int8(&data, &cfg);
        let deq = dequantize_tensor_int8(&q, &p);
        let max_err: f32 =
            data.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err <= p.scales[0] + 1e-6, "max_err={max_err} scale={}", p.scales[0]);
    }

    #[test]
    fn test_roundtrip_large_range() {
        let data = vec![-1000.0, -500.0, 0.0, 500.0, 1000.0];
        let cfg = sym_config();
        let (q, p) = quantize_tensor_int8(&data, &cfg);
        let deq = dequantize_tensor_int8(&q, &p);
        for (o, d) in data.iter().zip(deq.iter()) {
            assert!((o - d).abs() <= p.scales[0] + 1e-3);
        }
    }

    // ---- Per-channel quantization ----

    #[test]
    fn test_per_channel_2d() {
        // 2x4 matrix, channel_dim=0 → 2 channels
        let data = vec![
            1.0, 2.0, 3.0, 4.0, // channel 0
            -1.0, -2.0, -3.0, -4.0, // channel 1
        ];
        let shape = [2, 4];
        let cfg = Int8QuantConfig { per_channel: true, symmetric: true, ..Default::default() };
        let (q, p) = quantize_per_channel(&data, &shape, 0, &cfg);
        assert_eq!(q.len(), 8);
        assert_eq!(p.scales.len(), 2);
        assert_eq!(p.zero_points.len(), 2);
        // Channel 0 max value (4.0) at index 3 should map to 127
        assert_eq!(q[3], 127);
        // Channel 1 most-negative value (-4.0) at index 7 should map to -127
        assert_eq!(q[7], -127);
    }

    #[test]
    fn test_per_channel_3d() {
        // 2x3x2 tensor, channel_dim=0 → 2 channels of 6 elements each
        let data = vec![
            // channel 0
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, // channel 1
            -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
        ];
        let shape = [2, 3, 2];
        let cfg = Int8QuantConfig { per_channel: true, symmetric: true, ..Default::default() };
        let (q, p) = quantize_per_channel(&data, &shape, 0, &cfg);
        assert_eq!(q.len(), 12);
        assert_eq!(p.scales.len(), 2);
    }

    #[test]
    fn test_per_channel_dim1() {
        // 2x3 matrix, channel_dim=1 → 3 channels
        let data = vec![
            1.0, 10.0, 100.0, // row 0
            2.0, 20.0, 200.0, // row 1
        ];
        let shape = [2, 3];
        let cfg = Int8QuantConfig { per_channel: true, symmetric: true, ..Default::default() };
        let (q, p) = quantize_per_channel(&data, &shape, 1, &cfg);
        assert_eq!(q.len(), 6);
        assert_eq!(p.scales.len(), 3);
        // Each channel should have its own scale
        assert!(p.scales[0] < p.scales[1]);
        assert!(p.scales[1] < p.scales[2]);
    }

    // ---- Calibration methods ----

    #[test]
    fn test_calibration_minmax() {
        let data = vec![-10.0, -1.0, 0.0, 1.0, 10.0];
        let cfg = Int8QuantConfig {
            symmetric: true,
            calibration_method: CalibrationMethod::MinMax,
            ..Default::default()
        };
        let (_q, p) = quantize_tensor_int8(&data, &cfg);
        assert!((p.min_vals[0] - (-10.0)).abs() < 1e-6);
        assert!((p.max_vals[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_calibration_percentile() {
        // With percentile 99.0, extreme outlier should be clipped
        let mut data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        data.push(1000.0); // outlier
        let cfg = Int8QuantConfig {
            per_channel: false,
            symmetric: true,
            calibration_method: CalibrationMethod::Percentile(99.0),
        };
        let (_q, p) = quantize_tensor_int8(&data, &cfg);
        // The max should be clipped below the outlier
        assert!(p.max_vals[0] < 1000.0, "percentile should clip outlier");
    }

    #[test]
    fn test_calibration_mse() {
        let data: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let cfg = Int8QuantConfig {
            per_channel: false,
            symmetric: true,
            calibration_method: CalibrationMethod::MSE,
        };
        let (q, p) = quantize_tensor_int8(&data, &cfg);
        let deq = dequantize_tensor_int8(&q, &p);
        let mse: f32 = data.iter().zip(deq.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>()
            / data.len() as f32;
        // MSE calibration should produce reasonable error
        assert!(mse < 0.01, "mse={mse}");
    }

    // ---- Edge cases ----

    #[test]
    fn test_single_value() {
        let data = vec![42.0];
        let (q, p) = quantize_tensor_int8(&data, &sym_config());
        assert_eq!(q.len(), 1);
        let deq = dequantize_tensor_int8(&q, &p);
        assert!((deq[0] - 42.0).abs() < 1.0);
    }

    #[test]
    fn test_all_same_values() {
        let data = vec![5.0; 16];
        let (q, p) = quantize_tensor_int8(&data, &sym_config());
        // All quantized values should be the same
        assert!(q.windows(2).all(|w| w[0] == w[1]));
        let deq = dequantize_tensor_int8(&q, &p);
        for d in &deq {
            assert!((d - 5.0).abs() < p.scales[0] + 1e-6);
        }
    }

    #[test]
    fn test_very_large_values() {
        let data = vec![-1e6, 0.0, 1e6];
        let (q, p) = quantize_tensor_int8(&data, &sym_config());
        assert_eq!(q[0], -127);
        assert_eq!(q[1], 0);
        assert_eq!(q[2], 127);
        assert!(p.scales[0] > 0.0);
    }

    #[test]
    fn test_very_small_values() {
        let data = vec![-1e-7, 0.0, 1e-7];
        let (q, p) = quantize_tensor_int8(&data, &sym_config());
        let deq = dequantize_tensor_int8(&q, &p);
        for (o, d) in data.iter().zip(deq.iter()) {
            assert!((o - d).abs() < 1e-6);
        }
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<f32> = vec![];
        let (q, p) = quantize_tensor_int8(&data, &sym_config());
        assert!(q.is_empty());
        let deq = dequantize_tensor_int8(&q, &p);
        assert!(deq.is_empty());
    }

    // ---- Quantization error metrics ----

    #[test]
    fn test_quant_error_metrics() {
        let data: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let cfg = sym_config();
        let (q, p) = quantize_tensor_int8(&data, &cfg);
        let err = compute_quantization_error(&data, &q, &p);
        assert!(err.max_abs_error >= 0.0);
        assert!(err.mean_abs_error >= 0.0);
        assert!(err.rmse >= 0.0);
        assert!(err.snr_db > 0.0, "SNR should be positive for non-trivial data");
        assert!(err.rmse <= err.max_abs_error);
        assert!(err.mean_abs_error <= err.max_abs_error);
    }

    #[test]
    fn test_quant_error_perfect() {
        // Zeros should have zero error
        let data = vec![0.0f32; 8];
        let cfg = sym_config();
        let (q, p) = quantize_tensor_int8(&data, &cfg);
        let err = compute_quantization_error(&data, &q, &p);
        assert_eq!(err.max_abs_error, 0.0);
        assert_eq!(err.mean_abs_error, 0.0);
        assert_eq!(err.rmse, 0.0);
    }

    // ---- int8 range bounds ----

    #[test]
    fn test_symmetric_range_bounds() {
        // Symmetric uses [-127, 127] for quantized values (zero_point = 0)
        let data: Vec<f32> = (-200..=200).map(|i| i as f32).collect();
        let cfg = sym_config();
        let (q, p) = quantize_tensor_int8(&data, &cfg);
        assert!(q.iter().all(|&v| (-128..=127).contains(&(v as i16))));
        assert_eq!(p.zero_points[0], 0);
    }

    #[test]
    fn test_asymmetric_range_bounds() {
        // Asymmetric maps [min, max] to [-128, 127]
        let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let cfg = asym_config();
        let (q, _p) = quantize_tensor_int8(&data, &cfg);
        assert!(q.iter().all(|&v| (-128..=127).contains(&(v as i16))));
    }

    // ---- Zero-point correctness ----

    #[test]
    fn test_zero_point_symmetric_is_zero() {
        let data = vec![-5.0, -2.0, 0.0, 3.0, 7.0];
        let cfg = sym_config();
        let (_q, p) = quantize_tensor_int8(&data, &cfg);
        assert_eq!(p.zero_points[0], 0, "symmetric zero_point must be 0");
    }

    #[test]
    fn test_zero_point_asymmetric_offset() {
        // For data in [0, 10], zero_point should be non-zero
        let data: Vec<f32> = (0..=10).map(|i| i as f32).collect();
        let cfg = asym_config();
        let (_q, p) = quantize_tensor_int8(&data, &cfg);
        // Zero-point should offset so that 0.0 maps correctly
        // The zero_point is -128 + round(0 / scale) when min=0
        assert_eq!(p.zero_points[0], -128, "zero maps to -128 in asymmetric with min=0");
    }
}
