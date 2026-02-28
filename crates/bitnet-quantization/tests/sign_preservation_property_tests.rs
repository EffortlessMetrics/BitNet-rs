//! Property-based tests for quantization round-trip sign preservation.
//!
//! Key invariants tested:
//! - I2S quantize → dequantize preserves sign for sufficiently large magnitudes
//! - TL1 quantize → dequantize preserves sign for sufficiently large magnitudes
//! - TL2 quantize → dequantize preserves sign for sufficiently large magnitudes
//! - `QuantizedTensor::numel()` equals the product of its shape dimensions
//! - `QuantizedTensor::compression_ratio()` is always ≥ 1.0 for non-empty tensors
//! - Quantize is deterministic: same input always yields same output

#![cfg(feature = "cpu")]

use bitnet_common::QuantizationType;
use bitnet_quantization::{I2SQuantizer, TL1Quantizer, TL2Quantizer};
use proptest::prelude::*;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Create a flat BitNetTensor from f32 data using Candle.
fn make_tensor(data: &[f32]) -> bitnet_common::BitNetTensor {
    use candle_core::{Device, Tensor};
    let t = Tensor::from_slice(data, &[data.len()], &Device::Cpu).unwrap();
    bitnet_common::BitNetTensor::new(t)
}

/// Strategy for a vector of f32 values with controllable magnitude.
fn arb_f32_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-10.0f32..10.0, min_len..=max_len)
}

/// Align length to 32 (I2S block size).
fn align32(n: usize) -> usize {
    ((n + 31) / 32) * 32
}

// ── I2S sign preservation ────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80))]

    /// I2S quantize → dequantize preserves the sign of values with |x| > 1.0,
    /// provided they survive quantization (recovered ≠ 0).
    #[test]
    fn prop_i2s_sign_preserved_for_large_values(
        raw in arb_f32_vec(32, 256),
    ) {
        let len = align32(raw.len());
        let mut data = raw;
        data.resize(len, 0.0);

        let tensor = make_tensor(&data);
        let quantizer = I2SQuantizer::new();
        let qt = quantizer.quantize_tensor(&tensor).unwrap();
        let deq = quantizer.dequantize_tensor(&qt).unwrap();
        let deq_data = deq.to_vec().unwrap();

        for (i, (&orig, &recovered)) in data.iter().zip(deq_data.iter()).enumerate() {
            // Skip zeros: quantization may zero-out values small relative to block max
            if orig.abs() > 1.0 && recovered != 0.0 {
                prop_assert!(
                    orig.signum() == recovered.signum(),
                    "I2S sign mismatch at index {}: orig={}, recovered={}", i, orig, recovered
                );
            }
        }
    }

    /// I2S quantization is deterministic: same input → same output bytes.
    #[test]
    fn prop_i2s_deterministic(raw in arb_f32_vec(32, 128)) {
        let len = align32(raw.len());
        let mut data = raw;
        data.resize(len, 0.0);

        let tensor = make_tensor(&data);
        let quantizer = I2SQuantizer::new();
        let qt1 = quantizer.quantize_tensor(&tensor).unwrap();
        let qt2 = quantizer.quantize_tensor(&tensor).unwrap();
        prop_assert_eq!(&qt1.data, &qt2.data, "I2S must produce identical bytes");
        prop_assert_eq!(&qt1.scales, &qt2.scales, "I2S must produce identical scales");
    }
}

// ── TL1 sign preservation ────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80))]

    /// TL1 quantize → dequantize preserves sign for values with |x| > 1.0 that survive quantization.
    #[test]
    fn prop_tl1_sign_preserved_for_large_values(
        raw in arb_f32_vec(32, 256),
    ) {
        let len = align32(raw.len());
        let mut data = raw;
        data.resize(len, 0.0);

        let tensor = make_tensor(&data);
        let quantizer = TL1Quantizer::new();
        let qt = quantizer.quantize_tensor(&tensor).unwrap();
        let deq = quantizer.dequantize_tensor(&qt).unwrap();
        let deq_data = deq.to_vec().unwrap();

        for (i, (&orig, &recovered)) in data.iter().zip(deq_data.iter()).enumerate() {
            if orig.abs() > 1.0 && recovered != 0.0 {
                prop_assert!(
                    orig.signum() == recovered.signum(),
                    "TL1 sign mismatch at index {}: orig={}, recovered={}", i, orig, recovered
                );
            }
        }
    }
}

// ── TL2 sign preservation ────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80))]

    /// TL2 quantize → dequantize preserves sign for values with |x| > 1.0 that survive quantization.
    #[test]
    fn prop_tl2_sign_preserved_for_large_values(
        raw in arb_f32_vec(32, 256),
    ) {
        let len = align32(raw.len());
        let mut data = raw;
        data.resize(len, 0.0);

        let tensor = make_tensor(&data);
        let quantizer = TL2Quantizer::new();
        let qt = quantizer.quantize_tensor(&tensor).unwrap();
        let deq = quantizer.dequantize_tensor(&qt).unwrap();
        let deq_data = deq.to_vec().unwrap();

        for (i, (&orig, &recovered)) in data.iter().zip(deq_data.iter()).enumerate() {
            if orig.abs() > 1.0 && recovered != 0.0 {
                prop_assert!(
                    orig.signum() == recovered.signum(),
                    "TL2 sign mismatch at index {}: orig={}, recovered={}", i, orig, recovered
                );
            }
        }
    }
}

// ── QuantizedTensor invariants ───────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80))]

    /// `numel()` equals the product of the shape dimensions.
    #[test]
    fn prop_quantized_tensor_numel_matches_shape(
        raw in arb_f32_vec(32, 256),
    ) {
        let len = align32(raw.len());
        let mut data = raw;
        data.resize(len, 0.0);

        let tensor = make_tensor(&data);
        let quantizer = I2SQuantizer::new();
        let qt = quantizer.quantize_tensor(&tensor).unwrap();

        let expected: usize = qt.shape.iter().product();
        prop_assert_eq!(
            qt.numel(), expected,
            "numel() must equal product of shape {:?}", qt.shape
        );
    }

    /// `compression_ratio()` is at least 1.0 for non-empty tensors.
    #[test]
    fn prop_compression_ratio_at_least_one(
        raw in arb_f32_vec(32, 256),
    ) {
        let len = align32(raw.len());
        let mut data = raw;
        data.resize(len, 0.0);

        let tensor = make_tensor(&data);
        let quantizer = I2SQuantizer::new();
        let qt = quantizer.quantize_tensor(&tensor).unwrap();

        prop_assert!(
            qt.compression_ratio() >= 1.0,
            "compression_ratio {} must be >= 1.0", qt.compression_ratio()
        );
    }

    /// QuantizedTensor qtype matches the quantizer type.
    #[test]
    fn prop_quantized_tensor_type_matches_quantizer(
        raw in arb_f32_vec(32, 128),
    ) {
        let len = align32(raw.len());
        let mut data = raw;
        data.resize(len, 0.0);
        let tensor = make_tensor(&data);

        let i2s = I2SQuantizer::new();
        let qt = i2s.quantize_tensor(&tensor).unwrap();
        prop_assert_eq!(qt.qtype, QuantizationType::I2S);

        let tl1 = TL1Quantizer::new();
        let qt = tl1.quantize_tensor(&tensor).unwrap();
        prop_assert_eq!(qt.qtype, QuantizationType::TL1);

        let tl2 = TL2Quantizer::new();
        let qt = tl2.quantize_tensor(&tensor).unwrap();
        prop_assert_eq!(qt.qtype, QuantizationType::TL2);
    }
}
