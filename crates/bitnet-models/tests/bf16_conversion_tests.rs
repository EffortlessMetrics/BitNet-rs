//! BF16 ↔ F32 conversion validation tests.
//!
//! Validates that BF16/F16 weight conversions used in SafeTensors loading
//! maintain numerical accuracy suitable for dense model inference.

use half::{bf16, f16};

// ============================================================
// BF16 → F32 conversion accuracy
// ============================================================

#[test]
fn test_bf16_to_f32_zero() {
    let val = bf16::from_f32(0.0);
    assert_eq!(val.to_f32(), 0.0);
}

#[test]
fn test_bf16_to_f32_one() {
    let val = bf16::from_f32(1.0);
    assert_eq!(val.to_f32(), 1.0);
}

#[test]
fn test_bf16_to_f32_negative() {
    let val = bf16::from_f32(-1.0);
    assert_eq!(val.to_f32(), -1.0);
}

#[test]
fn test_bf16_to_f32_small_value() {
    let val = bf16::from_f32(0.001);
    let result = val.to_f32();
    // BF16 has ~3 decimal digits of precision
    assert!((result - 0.001).abs() < 1e-3, "BF16 small value: got {}", result);
}

#[test]
fn test_bf16_to_f32_large_value() {
    // BF16 has 7 mantissa bits, so precision decreases at large values
    let val = bf16::from_f32(65504.0);
    let result = val.to_f32();
    assert!((result - 65504.0).abs() < 256.0, "BF16 large value: got {}", result);
}

#[test]
fn test_bf16_preserves_sign() {
    for &v in &[1.0f32, -1.0, 0.5, -0.5, 100.0, -100.0] {
        let bf = bf16::from_f32(v);
        assert_eq!(bf.to_f32().is_sign_positive(), v.is_sign_positive());
    }
}

#[test]
fn test_bf16_special_values() {
    assert!(bf16::from_f32(f32::INFINITY).to_f32().is_infinite());
    assert!(bf16::from_f32(f32::NEG_INFINITY).to_f32().is_infinite());
    assert!(bf16::from_f32(f32::NAN).to_f32().is_nan());
}

// ============================================================
// F16 → F32 conversion accuracy
// ============================================================

#[test]
fn test_f16_to_f32_zero() {
    let val = f16::from_f32(0.0);
    assert_eq!(val.to_f32(), 0.0);
}

#[test]
fn test_f16_to_f32_one() {
    let val = f16::from_f32(1.0);
    assert_eq!(val.to_f32(), 1.0);
}

#[test]
fn test_f16_to_f32_negative() {
    let val = f16::from_f32(-1.0);
    assert_eq!(val.to_f32(), -1.0);
}

#[test]
fn test_f16_to_f32_precision() {
    // F16 has ~3.3 decimal digits of precision
    let val = f16::from_f32(0.001);
    let result = val.to_f32();
    assert!((result - 0.001).abs() < 1e-3, "F16 precision: got {}", result);
}

#[test]
fn test_f16_special_values() {
    assert!(f16::from_f32(f32::INFINITY).to_f32().is_infinite());
    assert!(f16::from_f32(f32::NEG_INFINITY).to_f32().is_infinite());
    assert!(f16::from_f32(f32::NAN).to_f32().is_nan());
}

// ============================================================
// BF16 vs F16 precision comparison
// ============================================================

#[test]
fn test_bf16_has_larger_range_than_f16() {
    // BF16 range: ±3.4e38 (same as F32)
    // F16 range: ±65504
    let large = 100000.0f32;
    let bf = bf16::from_f32(large);
    let fp = f16::from_f32(large);

    // BF16 can represent large values, F16 saturates to infinity
    assert!(bf.to_f32().is_finite(), "BF16 should handle large values");
    assert!(fp.to_f32().is_infinite(), "F16 should overflow for large values");
}

#[test]
fn test_f16_has_better_precision_than_bf16() {
    // F16 has 10 mantissa bits, BF16 has 7
    let val = 1.001f32;
    let bf_err = (bf16::from_f32(val).to_f32() - val).abs();
    let fp_err = (f16::from_f32(val).to_f32() - val).abs();

    assert!(
        fp_err <= bf_err,
        "F16 should be at least as precise: f16_err={}, bf16_err={}",
        fp_err,
        bf_err
    );
}

// ============================================================
// Batch conversion (simulating weight loading)
// ============================================================

#[test]
fn test_bf16_batch_conversion_accuracy() {
    // Simulate a small weight tensor in BF16
    let f32_weights: Vec<f32> = (0..256)
        .map(|i| (i as f32 - 128.0) / 128.0) // Range [-1, 1)
        .collect();

    let bf16_weights: Vec<bf16> = f32_weights.iter().map(|&v| bf16::from_f32(v)).collect();
    let reconverted: Vec<f32> = bf16_weights.iter().map(|v| v.to_f32()).collect();

    let max_error: f32 = f32_weights
        .iter()
        .zip(reconverted.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // BF16 roundtrip error should be small
    assert!(max_error < 0.01, "BF16 batch roundtrip max error too large: {}", max_error);
}

#[test]
fn test_f16_batch_conversion_accuracy() {
    let f32_weights: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();

    let f16_weights: Vec<f16> = f32_weights.iter().map(|&v| f16::from_f32(v)).collect();
    let reconverted: Vec<f32> = f16_weights.iter().map(|v| v.to_f32()).collect();

    let max_error: f32 = f32_weights
        .iter()
        .zip(reconverted.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // F16 should be more precise than BF16 in this range
    assert!(max_error < 0.001, "F16 batch roundtrip max error too large: {}", max_error);
}

// ============================================================
// Bytemuck casting (same pattern as SafeTensors loader)
// ============================================================

#[test]
fn test_bf16_from_bytes_via_bytemuck() {
    // Simulate reading BF16 from raw bytes (as SafeTensors loader does)
    let bf_val = bf16::from_f32(0.5);
    let bits = bf_val.to_bits();
    let bytes = bits.to_le_bytes();

    // Reconvert from bytes
    let reconstructed_bits = u16::from_le_bytes(bytes);
    let reconstructed = bf16::from_bits(reconstructed_bits);
    assert_eq!(reconstructed.to_f32(), bf_val.to_f32());
}

#[test]
fn test_f16_from_bytes_via_bytemuck() {
    let fp_val = f16::from_f32(0.5);
    let bits = fp_val.to_bits();
    let bytes = bits.to_le_bytes();

    let reconstructed_bits = u16::from_le_bytes(bytes);
    let reconstructed = f16::from_bits(reconstructed_bits);
    assert_eq!(reconstructed.to_f32(), fp_val.to_f32());
}

#[test]
fn test_bf16_bulk_bytemuck_cast() {
    // Simulate the bytemuck::try_cast_slice path from the SafeTensors loader
    let values: Vec<f32> = vec![0.0, 0.5, 1.0, -1.0, 0.25];
    let bf16_values: Vec<bf16> = values.iter().map(|&v| bf16::from_f32(v)).collect();
    let as_u16: Vec<u16> = bf16_values.iter().map(|v| v.to_bits()).collect();
    let raw_bytes: &[u8] = bytemuck::cast_slice(&as_u16);

    // Reconvert (same path as SafeTensors loader)
    let recovered_u16: &[u16] = bytemuck::cast_slice(raw_bytes);
    let recovered_f32: Vec<f32> =
        recovered_u16.iter().map(|&h| bf16::from_bits(h).to_f32()).collect();

    for (i, (&original, &recovered)) in values.iter().zip(recovered_f32.iter()).enumerate() {
        let err = (original - recovered).abs();
        assert!(err < 0.01, "Element {} error too large: {}", i, err);
    }
}

// ============================================================
// Model-scale conversion properties
// ============================================================

#[test]
fn test_phi4_scale_weight_stats() {
    // Simulate Phi-4 scale: 5120-dim weights normalized around [-0.1, 0.1]
    let dim = 5120;
    let weights: Vec<f32> =
        (0..dim).map(|i| ((i * 7 + 3) % 1000) as f32 / 10000.0 - 0.05).collect();

    let bf16_weights: Vec<bf16> = weights.iter().map(|&v| bf16::from_f32(v)).collect();
    let reconverted: Vec<f32> = bf16_weights.iter().map(|v| v.to_f32()).collect();

    // Check sum preservation (important for layer norm)
    let original_sum: f32 = weights.iter().sum();
    let converted_sum: f32 = reconverted.iter().sum();
    let sum_err = (original_sum - converted_sum).abs();

    assert!(
        sum_err < 1.0, // Accumulated error across 5120 elements
        "Sum error for Phi-4 scale weights: {}",
        sum_err
    );

    // Check mean preservation
    let original_mean = original_sum / dim as f32;
    let converted_mean = converted_sum / dim as f32;
    let mean_err = (original_mean - converted_mean).abs();

    assert!(mean_err < 1e-3, "Mean error for Phi-4 scale weights: {}", mean_err);
}

#[test]
fn test_embedding_scale_conversion() {
    // Phi-4 embedding: vocab=100352, hidden=5120
    // Just test a representative slice
    let slice_size = 1024;
    let weights: Vec<f32> =
        (0..slice_size).map(|i| ((i * 13 + 7) % 2000) as f32 / 10000.0 - 0.1).collect();

    let bf16_weights: Vec<bf16> = weights.iter().map(|&v| bf16::from_f32(v)).collect();
    let reconverted: Vec<f32> = bf16_weights.iter().map(|v| v.to_f32()).collect();

    // Max element-wise error
    let max_err: f32 =
        weights.iter().zip(reconverted.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    assert!(max_err < 0.01, "Embedding conversion max error: {}", max_err);
}
