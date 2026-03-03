#![no_main]

use arbitrary::Arbitrary;
use bitnet_quantization::calibrator::{BitWidth, CalibrationStats, symmetric_params};
use bitnet_quantization::int8_quant::{Int8QuantConfig, Int8QuantParams};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct CalibrationInput {
    /// Raw bytes interpreted as f32 values.
    data: Vec<u8>,
    /// Selector for calibration method.
    method_selector: u8,
    /// Percentile value (only used for Percentile method).
    percentile: u8,
    /// Per-channel vs per-tensor.
    per_channel: bool,
    /// Symmetric vs asymmetric.
    symmetric: bool,
    /// Arbitrary scale values.
    scales: Vec<u8>,
    /// Arbitrary zero-point values.
    zero_points: Vec<i8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    data.chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: CalibrationInput| {
    // Test CalibrationStats with arbitrary data
    let values = bytes_to_f32(&input.data, 256);
    if values.is_empty() {
        return;
    }

    // Filter non-finite values
    let clean: Vec<f32> =
        values.iter().copied().map(|v| if v.is_finite() { v } else { 0.0 }).collect();

    // Invariant 1: CalibrationStats::update must never panic
    let mut stats = CalibrationStats::new();
    stats.update(&clean);

    // Invariant 2: After update, count must equal number of values
    assert_eq!(stats.count, clean.len() as u64);

    // Invariant 3: range must be non-negative
    assert!(stats.range() >= 0.0, "range must be non-negative");

    // Invariant 4: Multiple updates accumulate correctly
    let mut stats2 = CalibrationStats::new();
    for chunk in clean.chunks(4.max(1)) {
        stats2.update(chunk);
    }
    assert_eq!(stats2.count, stats.count);

    // Invariant 5: symmetric_params must not panic
    let _ = symmetric_params(&stats, BitWidth::Int8);
    let _ = symmetric_params(&stats, BitWidth::Int4);

    // Test Int8QuantConfig construction with arbitrary methods
    let method = match input.method_selector % 3 {
        0 => bitnet_quantization::int8_quant::CalibrationMethod::MinMax,
        1 => {
            let p = (input.percentile as f32 / 255.0) * 100.0;
            bitnet_quantization::int8_quant::CalibrationMethod::Percentile(p)
        }
        _ => bitnet_quantization::int8_quant::CalibrationMethod::MSE,
    };

    let config = Int8QuantConfig {
        per_channel: input.per_channel,
        symmetric: input.symmetric,
        calibration_method: method,
    };

    // Invariant 6: quantize_tensor_int8 with valid config must not panic
    let _ = bitnet_quantization::int8_quant::quantize_tensor_int8(&clean, &config);

    // Invariant 7: Int8QuantParams with arbitrary scales/zero-points
    let scales: Vec<f32> = bytes_to_f32(&input.scales, 16);
    if !scales.is_empty() {
        let params = Int8QuantParams {
            scales: scales.clone(),
            zero_points: input.zero_points.iter().copied().take(scales.len()).collect(),
            min_vals: vec![0.0; scales.len()],
            max_vals: vec![1.0; scales.len()],
        };
        // Accessing fields must not panic
        let _ = params.scales.len();
        let _ = params.zero_points.len();
    }

    // Test BitWidth methods
    for bw in &[BitWidth::Int4, BitWidth::Int8] {
        let _ = bw.bits();
        let _ = bw.max_int();
        let _ = bw.min_int();
        let _ = bw.range();
    }
});
