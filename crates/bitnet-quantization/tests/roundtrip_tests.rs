//! Comprehensive quantization round-trip regression tests.
//!
//! Verifies that encode→decode preserves data within acceptable error
//! bounds for all quantization codecs (I2_S, TL1, TL2). Also tests
//! edge-case patterns, determinism, and the high-level `validate_round_trip`
//! helper.

#![cfg(feature = "cpu")]

use bitnet_common::{BitNetTensor, QuantizationType, Tensor};
use bitnet_quantization::utils::{
    calculate_mse, create_tensor_from_f32, pack_2bit_values, unpack_2bit_values,
};
use bitnet_quantization::{
    I2SQuantizer, Quantize, TL1Quantizer, TL2Quantizer, validate_round_trip,
};
use candle_core::Device;

// -------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------

/// Create a 1-D `BitNetTensor` on CPU from a flat `f32` slice.
fn tensor_1d(data: &[f32]) -> BitNetTensor {
    create_tensor_from_f32(data.to_vec(), &[data.len()], &Device::Cpu).unwrap()
}

/// Extract the flat f32 data from a `BitNetTensor`.
fn to_f32(t: &BitNetTensor) -> Vec<f32> {
    t.to_vec().unwrap()
}

/// Maximum absolute error between two equal-length slices.
fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

// ===================================================================
// I2_S round-trip tests
// ===================================================================

mod i2s {
    use super::*;

    #[test]
    fn roundtrip_ternary_values() {
        // I2_S should faithfully represent {-1, 0, +1} — the core
        // ternary alphabet — with at most a scale-dependent error.
        let data: Vec<f32> = vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0];
        let t = tensor_1d(&data);
        let q = I2SQuantizer::new();
        let enc = q.quantize_tensor(&t).unwrap();
        let dec = q.dequantize_tensor(&enc).unwrap();

        assert_eq!(enc.qtype, QuantizationType::I2S);
        assert_eq!(dec.shape(), &[data.len()]);
        let out = to_f32(&dec);
        assert_eq!(out.len(), data.len());
        // 2-bit quantization: error must be bounded
        let err = max_abs_err(&data, &out);
        assert!(err <= 2.0, "ternary round-trip error too large: {err}");
    }

    #[test]
    fn roundtrip_larger_block() {
        // 64 elements — exercises multi-block path (default block_size ≤32).
        let data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) / 32.0).collect();
        let t = tensor_1d(&data);
        let q = I2SQuantizer::new();
        let enc = q.quantize_tensor(&t).unwrap();
        let dec = q.dequantize_tensor(&enc).unwrap();

        assert_eq!(dec.shape(), &[64]);
        let out = to_f32(&dec);
        let mse = calculate_mse(&data, &out).unwrap();
        // 2-bit quantization on a ramp — MSE < 1.0 is generous
        assert!(mse < 1.0, "I2S 64-elem ramp MSE too large: {mse}");
    }

    #[test]
    fn all_zeros() {
        let data = vec![0.0; 32];
        let t = tensor_1d(&data);
        let q = I2SQuantizer::new();
        let enc = q.quantize_tensor(&t).unwrap();
        let dec = q.dequantize_tensor(&enc).unwrap();

        let out = to_f32(&dec);
        for &v in &out {
            assert!(v.abs() < 1e-6, "all-zero block should dequantize near zero, got {v}");
        }
    }

    #[test]
    fn all_same_nonzero() {
        let data = vec![0.5; 32];
        let t = tensor_1d(&data);
        let q = I2SQuantizer::new();
        let enc = q.quantize_tensor(&t).unwrap();
        let dec = q.dequantize_tensor(&enc).unwrap();

        let out = to_f32(&dec);
        assert_eq!(out.len(), 32);
        // All outputs should be the same value (quantization is uniform).
        let first = out[0];
        for &v in &out[1..] {
            assert!((v - first).abs() < 1e-6, "constant block should dequantize uniformly");
        }
    }

    #[test]
    fn alternating_pattern() {
        let data: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let t = tensor_1d(&data);
        let q = I2SQuantizer::new();
        let enc = q.quantize_tensor(&t).unwrap();
        let dec = q.dequantize_tensor(&enc).unwrap();

        let out = to_f32(&dec);
        assert_eq!(out.len(), 32);
        let err = max_abs_err(&data, &out);
        assert!(err < 2.0, "alternating pattern error too large: {err}");
    }

    #[test]
    fn determinism() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.6).collect();
        let t = tensor_1d(&data);
        let q = I2SQuantizer::new();

        let enc1 = q.quantize_tensor(&t).unwrap();
        let enc2 = q.quantize_tensor(&t).unwrap();

        assert_eq!(enc1.data, enc2.data, "I2S encoding must be deterministic");
        assert_eq!(enc1.scales, enc2.scales, "I2S scales must be deterministic");
    }

    #[test]
    fn shape_preserved() {
        let data = vec![1.0, -1.0, 0.0, 0.5, -0.5, 0.0];
        let shape = vec![2, 3];
        let t = create_tensor_from_f32(data, &shape, &Device::Cpu).unwrap();
        let q = I2SQuantizer::new();
        let enc = q.quantize_tensor(&t).unwrap();
        let dec = q.dequantize_tensor(&enc).unwrap();

        assert_eq!(enc.shape, shape);
        assert_eq!(dec.shape(), &shape);
    }

    #[test]
    fn custom_block_size() {
        let data: Vec<f32> = (0..16).map(|i| i as f32 - 8.0).collect();
        let t = tensor_1d(&data);
        for bs in [4, 8, 16] {
            let q = I2SQuantizer::with_block_size(bs);
            let enc = q.quantize_tensor(&t).unwrap();
            let dec = q.dequantize_tensor(&enc).unwrap();
            assert_eq!(enc.block_size, bs);
            assert_eq!(dec.shape(), &[16]);
        }
    }

    #[test]
    fn compression_ratio_positive() {
        let data = vec![1.0; 256];
        let t = tensor_1d(&data);
        let q = I2SQuantizer::new();
        let enc = q.quantize_tensor(&t).unwrap();
        let ratio = enc.compression_ratio();
        assert!(ratio > 1.0, "compression ratio should exceed 1.0, got {ratio}");
    }

    #[test]
    fn quantize_weights_api() {
        let weights: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let q = I2SQuantizer::new();
        let enc = q.quantize_weights(&weights).unwrap();
        assert_eq!(enc.qtype, QuantizationType::I2S);
        assert_eq!(enc.shape, vec![32]);
    }
}

// ===================================================================
// TL1 round-trip tests
// ===================================================================

mod tl1 {
    use super::*;

    #[test]
    fn roundtrip_basic() {
        let data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) / 10.0).collect();
        let t = tensor_1d(&data);
        let q = TL1Quantizer::new();
        let enc = q.quantize_tensor(&t).unwrap();
        let dec = q.dequantize_tensor(&enc).unwrap();

        assert_eq!(enc.qtype, QuantizationType::TL1);
        assert_eq!(dec.shape(), &[64]);
        let out = to_f32(&dec);
        let mse = calculate_mse(&data, &out).unwrap();
        assert!(mse < 10.0, "TL1 round-trip MSE too large: {mse}");
    }

    #[test]
    fn all_zeros() {
        let data = vec![0.0; 64];
        let t = tensor_1d(&data);
        let q = TL1Quantizer::new();
        let enc = q.quantize_tensor(&t).unwrap();
        let dec = q.dequantize_tensor(&enc).unwrap();

        let out = to_f32(&dec);
        for &v in &out {
            assert!(v.abs() < 1e-5, "all-zero TL1 dequant should be near zero, got {v}");
        }
    }

    #[test]
    fn all_same_nonzero() {
        let data = vec![2.0; 64];
        let t = tensor_1d(&data);
        let q = TL1Quantizer::new();
        let enc = q.quantize_tensor(&t).unwrap();
        let dec = q.dequantize_tensor(&enc).unwrap();

        let out = to_f32(&dec);
        let first = out[0];
        for &v in &out[1..] {
            assert!((v - first).abs() < 1e-5, "constant TL1 block should dequantize uniformly");
        }
    }

    #[test]
    fn alternating_pattern() {
        let data: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 3.0 } else { -3.0 }).collect();
        let t = tensor_1d(&data);
        let q = TL1Quantizer::new();
        let enc = q.quantize_tensor(&t).unwrap();
        let dec = q.dequantize_tensor(&enc).unwrap();

        let out = to_f32(&dec);
        assert_eq!(out.len(), 64);
        let err = max_abs_err(&data, &out);
        assert!(err < 10.0, "TL1 alternating pattern error too large: {err}");
    }

    #[test]
    fn determinism() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05).collect();
        let t = tensor_1d(&data);
        let q = TL1Quantizer::new();
        let enc1 = q.quantize_tensor(&t).unwrap();
        let enc2 = q.quantize_tensor(&t).unwrap();

        assert_eq!(enc1.data, enc2.data, "TL1 encoding must be deterministic");
        assert_eq!(enc1.scales, enc2.scales, "TL1 scales must be deterministic");
    }

    #[test]
    fn shape_preserved() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let shape = vec![8, 8];
        let t = create_tensor_from_f32(data, &shape, &Device::Cpu).unwrap();
        let q = TL1Quantizer::new();
        let enc = q.quantize_tensor(&t).unwrap();
        let dec = q.dequantize_tensor(&enc).unwrap();

        assert_eq!(enc.shape, shape);
        assert_eq!(dec.shape(), &shape);
    }

    #[test]
    fn quantize_weights_api() {
        let weights: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let q = TL1Quantizer::new();
        let enc = q.quantize_weights(&weights).unwrap();
        assert_eq!(enc.qtype, QuantizationType::TL1);
        assert_eq!(enc.shape, vec![64]);
    }
}

// ===================================================================
// TL2 round-trip tests
// ===================================================================

mod tl2 {
    use super::*;

    #[test]
    fn roundtrip_basic() {
        // Use 128+ elements to fill at least one TL2 block.
        let data: Vec<f32> = (0..128).map(|i| ((i as f32) - 64.0) / 20.0).collect();
        let t = tensor_1d(&data);
        let q = TL2Quantizer::new();
        let enc = q.quantize_tensor(&t).unwrap();
        let dec = q.dequantize_tensor(&enc).unwrap();

        assert_eq!(enc.qtype, QuantizationType::TL2);
        assert_eq!(dec.shape(), &[128]);
        let out = to_f32(&dec);
        let mse = calculate_mse(&data, &out).unwrap();
        assert!(mse < 10.0, "TL2 round-trip MSE too large: {mse}");
    }

    #[test]
    fn all_zeros() {
        let data = vec![0.0; 128];
        let t = tensor_1d(&data);
        let q = TL2Quantizer::new();
        let enc = q.quantize_tensor(&t).unwrap();
        let dec = q.dequantize_tensor(&enc).unwrap();

        let out = to_f32(&dec);
        for &v in &out {
            assert!(v.abs() < 1e-5, "all-zero TL2 dequant should be near zero, got {v}");
        }
    }

    #[test]
    fn all_same_nonzero() {
        let data = vec![1.5; 128];
        let t = tensor_1d(&data);
        let q = TL2Quantizer::new();
        let enc = q.quantize_tensor(&t).unwrap();
        let dec = q.dequantize_tensor(&enc).unwrap();

        let out = to_f32(&dec);
        let first = out[0];
        for &v in &out[1..] {
            assert!((v - first).abs() < 1e-5, "constant TL2 block should dequantize uniformly");
        }
    }

    #[test]
    fn alternating_pattern() {
        let data: Vec<f32> = (0..128).map(|i| if i % 2 == 0 { 2.0 } else { -2.0 }).collect();
        let t = tensor_1d(&data);
        let q = TL2Quantizer::new();
        let enc = q.quantize_tensor(&t).unwrap();
        let dec = q.dequantize_tensor(&enc).unwrap();

        let out = to_f32(&dec);
        assert_eq!(out.len(), 128);
        let err = max_abs_err(&data, &out);
        assert!(err < 10.0, "TL2 alternating pattern error too large: {err}");
    }

    #[test]
    fn determinism() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.02 - 1.28).collect();
        let t = tensor_1d(&data);
        let q = TL2Quantizer::new();
        let enc1 = q.quantize_tensor(&t).unwrap();
        let enc2 = q.quantize_tensor(&t).unwrap();

        assert_eq!(enc1.data, enc2.data, "TL2 encoding must be deterministic");
        assert_eq!(enc1.scales, enc2.scales, "TL2 scales must be deterministic");
    }

    #[test]
    fn shape_preserved() {
        let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let shape = vec![16, 16];
        let t = create_tensor_from_f32(data, &shape, &Device::Cpu).unwrap();
        let q = TL2Quantizer::new();
        let enc = q.quantize_tensor(&t).unwrap();
        let dec = q.dequantize_tensor(&enc).unwrap();

        assert_eq!(enc.shape, shape);
        assert_eq!(dec.shape(), &shape);
    }

    #[test]
    fn quantize_weights_api() {
        let weights: Vec<f32> = (0..128).map(|i| (i as f32) * 0.05).collect();
        let q = TL2Quantizer::new();
        let enc = q.quantize_weights(&weights).unwrap();
        assert_eq!(enc.qtype, QuantizationType::TL2);
        assert_eq!(enc.shape, vec![128]);
    }
}

// ===================================================================
// Cross-codec & high-level API tests
// ===================================================================

mod cross_codec {
    use super::*;

    #[test]
    fn validate_round_trip_i2s() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let t = tensor_1d(&data);
        let ok = validate_round_trip(&t, QuantizationType::I2S, 5.0).unwrap();
        assert!(ok, "validate_round_trip should pass for I2S with tolerance 5.0");
    }

    #[test]
    fn validate_round_trip_tl1() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let t = tensor_1d(&data);
        let ok = validate_round_trip(&t, QuantizationType::TL1, 10.0).unwrap();
        assert!(ok, "validate_round_trip should pass for TL1 with tolerance 10.0");
    }

    #[test]
    fn validate_round_trip_tl2() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.05).collect();
        let t = tensor_1d(&data);
        let ok = validate_round_trip(&t, QuantizationType::TL2, 10.0).unwrap();
        assert!(ok, "validate_round_trip should pass for TL2 with tolerance 10.0");
    }

    #[test]
    fn quantize_trait_dispatches_correctly() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let t = tensor_1d(&data);

        // Use the Quantize trait on BitNetTensor
        let enc = t.quantize(QuantizationType::I2S).unwrap();
        assert_eq!(enc.qtype, QuantizationType::I2S);

        // Dequantize via the Quantize trait on QuantizedTensor
        let dec = enc.dequantize().unwrap();
        assert_eq!(dec.shape(), &[32]);
    }

    #[test]
    fn quantizer_trait_type_mismatch_rejected() {
        // Encode as I2S, then try to dequantize with a TL1 quantizer → error.
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let t = tensor_1d(&data);
        let i2s = I2SQuantizer::new();
        let enc = i2s.quantize_tensor(&t).unwrap();

        let tl1 = TL1Quantizer::new();
        let res = tl1.dequantize_tensor(&enc);
        assert!(res.is_err(), "dequantizing I2S with TL1 quantizer should fail");
    }
}

// ===================================================================
// Bit-packing unit tests (I2_S pack/unpack)
// ===================================================================

mod packing {
    use super::*;

    #[test]
    fn pack_unpack_identity() {
        let values: Vec<i8> = vec![-1, 0, 1, -1, 0, 1, -1, 0];
        let packed = pack_2bit_values(&values);
        let unpacked = unpack_2bit_values(&packed, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_unpack_all_zeros() {
        let values = vec![0i8; 32];
        let packed = pack_2bit_values(&values);
        let unpacked = unpack_2bit_values(&packed, 32);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_unpack_all_ones() {
        let values = vec![1i8; 32];
        let packed = pack_2bit_values(&values);
        let unpacked = unpack_2bit_values(&packed, 32);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_unpack_all_neg_ones() {
        let values = vec![-1i8; 32];
        let packed = pack_2bit_values(&values);
        let unpacked = unpack_2bit_values(&packed, 32);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_unpack_alternating() {
        let values: Vec<i8> = (0..32).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let packed = pack_2bit_values(&values);
        let unpacked = unpack_2bit_values(&packed, 32);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_unpack_deterministic() {
        let values: Vec<i8> = vec![1, 0, -1, 1, -2, 0, 1, -1];
        let p1 = pack_2bit_values(&values);
        let p2 = pack_2bit_values(&values);
        assert_eq!(p1, p2, "packing must be deterministic");
    }

    #[test]
    fn pack_unpack_non_multiple_of_four() {
        // 5 elements — last byte is partial
        let values: Vec<i8> = vec![1, 0, -1, 1, -1];
        let packed = pack_2bit_values(&values);
        let unpacked = unpack_2bit_values(&packed, values.len());
        assert_eq!(values, unpacked);
    }
}
