#![no_main]

use arbitrary::Arbitrary;
use bitnet_quantization::int8_quant::{
    CalibrationMethod, Int8QuantConfig, dequantize_tensor_int8, quantize_tensor_int8,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct QuantizeRoundtripInput {
    /// Fuzz data length (mapped to realistic tensor size).
    len_byte: u8,
    /// Per-channel vs per-tensor.
    per_channel: bool,
    /// Symmetric vs asymmetric.
    symmetric: bool,
    /// Calibration method selector.
    calibration_byte: u8,
    /// Raw float data.
    data: Vec<u8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: QuantizeRoundtripInput| {
    let target_len = (input.len_byte as usize % 256) + 1;

    let raw = bytes_to_f32(&input.data, target_len);
    if raw.len() < target_len {
        return;
    }

    // Clamp to realistic float range and filter non-finite.
    let data: Vec<f32> = raw[..target_len]
        .iter()
        .map(|&v| if v.is_finite() { v.clamp(-1e4, 1e4) } else { 0.0 })
        .collect();

    let calibration_method = match input.calibration_byte % 3 {
        0 => CalibrationMethod::MinMax,
        1 => CalibrationMethod::Percentile(99.9),
        _ => CalibrationMethod::MSE,
    };

    let config = Int8QuantConfig {
        per_channel: input.per_channel,
        symmetric: input.symmetric,
        calibration_method,
    };

    // --- Int8 quantize → dequantize roundtrip ---
    let (quantized, params) = quantize_tensor_int8(&data, &config);
    assert_eq!(quantized.len(), data.len(), "quantized length mismatch");

    // Scales must be non-negative and finite.
    for (i, &s) in params.scales.iter().enumerate() {
        assert!(s.is_finite(), "scale non-finite at {i}: {s}");
        assert!(s >= 0.0, "scale negative at {i}: {s}");
    }

    let dequantized = dequantize_tensor_int8(&quantized, &params);
    assert_eq!(dequantized.len(), data.len(), "dequantized length mismatch");

    // Invariant: all dequantized values are finite.
    for (i, &val) in dequantized.iter().enumerate() {
        assert!(val.is_finite(), "dequantized value non-finite at {i}: {val}");
    }

    // Invariant: roundtrip error is bounded for int8 quantization.
    // Max absolute error should be proportional to scale (1 LSB = scale).
    let max_scale = params.scales.iter().copied().fold(0.0f32, f32::max);
    if max_scale > 0.0 {
        for (i, (&orig, &deq)) in data.iter().zip(dequantized.iter()).enumerate() {
            let err = (orig - deq).abs();
            // Int8 quantization error should be within 1 scale step.
            assert!(
                err <= max_scale + 1e-5,
                "roundtrip error {err} exceeds max_scale {max_scale} at index {i} \
                 (orig={orig}, deq={deq})"
            );
        }
    }

    // Invariant: zero input quantizes to zero (within tolerance).
    let zeros = vec![0.0f32; target_len];
    let (q_zero, p_zero) = quantize_tensor_int8(&zeros, &config);
    let d_zero = dequantize_tensor_int8(&q_zero, &p_zero);
    for (i, &val) in d_zero.iter().enumerate() {
        assert!(val.abs() < 1e-6, "zero input dequantized to {val} at index {i}");
    }
});
