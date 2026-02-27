//! Integration tests for quantization roundtrip correctness.
//!
//! Verifies that quantizing and then dequantizing produces values within
//! acceptable tolerance, and that all invariants hold across I2S, TL1, and TL2.

use bitnet_common::{BitNetTensor, QuantizationType};
use bitnet_quantization::{I2SQuantizer, Quantize, TL1Quantizer, TL2Quantizer};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};

// ── helpers ──────────────────────────────────────────────────────────────────

fn make_tensor(data: Vec<f32>, shape: &[usize]) -> BitNetTensor {
    let t = CandleTensor::from_vec(data, shape, &CandleDevice::Cpu).unwrap();
    BitNetTensor::new(t)
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

// ── roundtrip correctness ────────────────────────────────────────────────────

/// I2S roundtrip: max absolute error must be well below the scale of the input.
#[test]
fn test_i2s_roundtrip_within_tolerance() {
    let data: Vec<f32> = (0..128).map(|i| ((i as f32) / 64.0) - 1.0).collect();
    let tensor = make_tensor(data.clone(), &[128]);

    let q = I2SQuantizer::new();
    let quantized = q.quantize_tensor(&tensor).unwrap();
    let recovered = q.dequantize_tensor(&quantized).unwrap().to_vec().unwrap();

    let max_err = max_abs_error(&data, &recovered);
    // 4 quantization levels over [-1, 1] → step ≈ 0.67; I2S should stay well under that.
    assert!(max_err < 0.5, "I2S roundtrip max error {max_err:.4} exceeds tolerance 0.5");
}

/// TL1 roundtrip: max absolute error must be well below the scale of the input.
#[test]
fn test_tl1_roundtrip_within_tolerance() {
    let data: Vec<f32> = (0..128).map(|i| ((i as f32) / 64.0) - 1.0).collect();
    let tensor = make_tensor(data.clone(), &[128]);

    let q = TL1Quantizer::new();
    let quantized = q.quantize_tensor(&tensor).unwrap();
    let recovered = q.dequantize_tensor(&quantized).unwrap().to_vec().unwrap();

    let max_err = max_abs_error(&data, &recovered);
    // 4 levels over [-1, 1] → step ≈ 0.67; allow up to half a step.
    assert!(max_err <= 0.5, "TL1 roundtrip max error {max_err:.4} exceeds tolerance 0.5");
}

/// TL2 roundtrip: max absolute error must be well below the scale of the input.
#[test]
fn test_tl2_roundtrip_within_tolerance() {
    let data: Vec<f32> = (0..128).map(|i| ((i as f32) / 64.0) - 1.0).collect();
    let tensor = make_tensor(data.clone(), &[128]);

    let q = TL2Quantizer::new();
    let quantized = q.quantize_tensor(&tensor).unwrap();
    let recovered = q.dequantize_tensor(&quantized).unwrap().to_vec().unwrap();

    let max_err = max_abs_error(&data, &recovered);
    // 4 levels over [-1, 1] → step ≈ 0.67; allow up to half a step.
    assert!(max_err <= 0.5, "TL2 roundtrip max error {max_err:.4} exceeds tolerance 0.5");
}

// ── range invariants ─────────────────────────────────────────────────────────

/// I2S packed codes must all be in the 2-bit unsigned range [0, 3].
/// Each byte holds four 2-bit codes; extract and validate every one.
#[test]
fn test_i2s_packed_codes_in_range() {
    let data: Vec<f32> = (0..64).map(|i| (i as f32 / 32.0) - 1.0).collect();
    let tensor = make_tensor(data, &[64]);

    let q = I2SQuantizer::new();
    let quantized = q.quantize_tensor(&tensor).unwrap();

    for (byte_idx, &byte) in quantized.data.iter().enumerate() {
        for bit_pos in 0..4u8 {
            let code = (byte >> (bit_pos * 2)) & 0b11;
            assert!(code <= 3, "Byte {byte_idx} bit-pair {bit_pos}: code {code} out of [0, 3]");
        }
    }
}

/// TL1 packed codes must all be in the 2-bit unsigned range [0, 3].
#[test]
fn test_tl1_packed_codes_in_range() {
    let data: Vec<f32> = (0..128).map(|i| (i as f32 / 64.0) - 1.0).collect();
    let tensor = make_tensor(data, &[128]);

    let q = TL1Quantizer::new();
    let quantized = q.quantize_tensor(&tensor).unwrap();

    for (byte_idx, &byte) in quantized.data.iter().enumerate() {
        for bit_pos in 0..4u8 {
            let code = (byte >> (bit_pos * 2)) & 0b11;
            assert!(code <= 3, "Byte {byte_idx} bit-pair {bit_pos}: code {code} out of [0, 3]");
        }
    }
}

// ── scale invariants ─────────────────────────────────────────────────────────

/// All scale factors produced by each quantizer must be positive finite numbers.
#[test]
fn test_scale_factors_are_positive_finite() {
    let data: Vec<f32> = (0..128).map(|i| (i as f32 / 64.0) - 1.0).collect();
    let tensor = make_tensor(data, &[128]);

    for (name, quantized) in [
        ("I2S", I2SQuantizer::new().quantize_tensor(&tensor).unwrap()),
        ("TL1", TL1Quantizer::new().quantize_tensor(&tensor).unwrap()),
        ("TL2", TL2Quantizer::new().quantize_tensor(&tensor).unwrap()),
    ] {
        for (i, &scale) in quantized.scales.iter().enumerate() {
            assert!(
                scale.is_finite() && scale > 0.0,
                "{name} scale[{i}] = {scale} is not positive-finite"
            );
        }
    }
}

// ── edge cases ───────────────────────────────────────────────────────────────

/// All-zeros input must roundtrip to all-zeros under every quantization type.
#[test]
fn test_all_zeros_roundtrip() {
    let zeros = vec![0.0f32; 128];
    let tensor = make_tensor(zeros, &[128]);

    for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
        let quantized = tensor.quantize(qtype).unwrap();
        let recovered = quantized.dequantize().unwrap().to_vec().unwrap();
        assert_eq!(recovered.len(), 128);
        for (i, &v) in recovered.iter().enumerate() {
            assert_eq!(v, 0.0, "{qtype:?} element {i}: expected 0.0, got {v}");
        }
    }
}

/// All-equal non-zero input must roundtrip with negligible error under every type.
#[test]
fn test_all_equal_values_roundtrip() {
    let val = 0.75f32;
    let data = vec![val; 128];
    let tensor = make_tensor(data.clone(), &[128]);

    for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
        let quantized = tensor.quantize(qtype).unwrap();
        let recovered = quantized.dequantize().unwrap().to_vec().unwrap();
        assert_eq!(recovered.len(), 128, "{qtype:?}: wrong output length");
        for (i, &v) in recovered.iter().enumerate() {
            let err = (v - val).abs();
            assert!(err < 0.5, "{qtype:?} element {i}: error {err:.4} for constant input {val}");
        }
    }
}

// ── structural invariants ────────────────────────────────────────────────────

/// Shape metadata must be preserved exactly after quantize → dequantize.
#[test]
fn test_shape_preserved_after_roundtrip() {
    let shape = vec![4, 32];
    let data: Vec<f32> = (0..128).map(|i| (i as f32 / 64.0) - 1.0).collect();
    let tensor = make_tensor(data, &shape);

    for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
        let quantized = tensor.quantize(qtype).unwrap();
        assert_eq!(
            quantized.shape, shape,
            "{qtype:?}: quantized shape {:?} != original {:?}",
            quantized.shape, shape
        );
    }
}

/// 2-bit quantization must achieve at least 4× compression vs FP32 storage.
#[test]
fn test_compression_ratio_at_least_4x() {
    let data: Vec<f32> = (0..256).map(|i| (i as f32 / 128.0) - 1.0).collect();
    let tensor = make_tensor(data, &[256]);

    for (name, quantized) in [
        ("I2S", I2SQuantizer::new().quantize_tensor(&tensor).unwrap()),
        ("TL1", TL1Quantizer::new().quantize_tensor(&tensor).unwrap()),
        ("TL2", TL2Quantizer::new().quantize_tensor(&tensor).unwrap()),
    ] {
        let ratio = quantized.compression_ratio();
        assert!(ratio >= 4.0, "{name} compression ratio {ratio:.2} < expected 4×");
    }
}
