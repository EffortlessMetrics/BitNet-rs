//! Extended integration and property-based tests for bitnet-quantization.
//!
//! Covers I2_S, TL1, TL2, and QK256 quantization formats with:
//! - Pack/unpack round-trips
//! - Scale factor correctness
//! - Edge cases (all-zeros, all-ones, alternating ±1)
//! - Known input → known output assertions
//! - Property-based invariants via proptest

use bitnet_common::{BitNetTensor, QuantizationType};
use bitnet_quantization::{
    I2SQuantizer, QuantizerTrait,
    i2s_qk256::{
        I2SQk256NoScale, QK256_BLOCK, QK256_PACKED_BYTES, code_to_f32, unpack_qk256_block,
    },
    tl1::{LookupTable, TL1Config, TL1Quantizer},
    tl2::{TL2Quantizer, VectorizedLookupTable},
    utils::{calculate_grouped_scales, calculate_scale, pack_2bit_values, unpack_2bit_values},
};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use proptest::prelude::*;

// ── helpers ──────────────────────────────────────────────────────────────────

fn make_tensor(data: Vec<f32>, shape: &[usize]) -> BitNetTensor {
    let t = CandleTensor::from_vec(data, shape, &CandleDevice::Cpu).unwrap();
    BitNetTensor::new(t)
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 1 – I2_S quantization
// ═══════════════════════════════════════════════════════════════════════════

/// All-zeros input round-trips exactly: scale is 1.0 and every element
/// unpacks back to 0.0 (the offset-2 encoding of signed 0 is code 2).
#[test]
fn test_i2s_all_zeros_roundtrip() {
    let data = vec![0.0f32; 32];
    let tensor = make_tensor(data.clone(), &[32]);
    let q = I2SQuantizer::new();
    let qt = q.quantize_tensor(&tensor).unwrap();
    let dq = q.dequantize_tensor(&qt).unwrap();
    let vals = dq.to_vec().unwrap();
    assert_eq!(vals.len(), 32);
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.abs() < 1e-5, "all-zeros input: element {i} should be ~0.0, got {v}");
    }
}

/// All-ones input should round-trip to within the expected quantisation error.
/// For 2-bit signed [-2,1] and max=1.0, scale ≤ 1.0 so error ≤ scale.
#[test]
fn test_i2s_all_ones_roundtrip() {
    let data = vec![1.0f32; 32];
    let tensor = make_tensor(data.clone(), &[32]);
    let q = I2SQuantizer::new();
    let qt = q.quantize_tensor(&tensor).unwrap();
    let dq = q.dequantize_tensor(&qt).unwrap();
    let vals = dq.to_vec().unwrap();
    let err = max_abs_err(&data, &vals);
    // Scale = 1.0 / 1 = 1.0 (max_quant for 2-bit signed is 1), so quantisation error ≤ 1.0
    assert!(err <= 1.1, "all-ones round-trip error {err} exceeds 1.1");
    // And all values should be positive
    for &v in &vals {
        assert!(v >= -1e-5, "dequantised value {v} should not be significantly negative");
    }
}

/// Alternating +1/-1 input: the max absolute value is 1.0 so the scale is
/// exactly 1.0 and each value maps to a non-zero code.
#[test]
fn test_i2s_alternating_sign_roundtrip() {
    let data: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let tensor = make_tensor(data.clone(), &[32]);
    let q = I2SQuantizer::new();
    let qt = q.quantize_tensor(&tensor).unwrap();
    let dq = q.dequantize_tensor(&qt).unwrap();
    let vals = dq.to_vec().unwrap();
    // No value should collapse to zero – signs must be preserved
    for (i, &v) in vals.iter().enumerate() {
        let expected_sign = if i % 2 == 0 { 1.0f32 } else { -1.0f32 };
        assert!(
            v * expected_sign > 0.0,
            "element {i}: sign not preserved (expected {expected_sign}, got {v})"
        );
    }
}

/// `calculate_scale` returns the correct value for a known input.
/// For bits=2, max_quant = (1<<1)-1 = 1. With max_abs = 4.0,
/// expected scale = 4.0 / 1 = 4.0.
#[test]
fn test_i2s_scale_factor_for_known_input() {
    let data = vec![-4.0f32, -2.0, 0.0, 2.0, 4.0];
    let scale = calculate_scale(&data, 2);
    assert!(scale.is_finite() && scale > 0.0, "scale must be positive finite, got {scale}");
    // max_abs = 4.0, max_quant = 1 → scale = 4.0
    assert!((scale - 4.0).abs() < 1e-5, "expected scale ≈ 4.0, got {scale}");
}

/// The packed byte count must equal ceil(n_elements * 2 / 8) = ceil(n/4).
#[test]
fn test_i2s_packed_byte_count() {
    for n in [4usize, 8, 16, 32, 64, 100, 256] {
        let data = vec![0.5f32; n];
        let tensor = make_tensor(data, &[n]);
        let q = I2SQuantizer::new();
        let qt = q.quantize_tensor(&tensor).unwrap();
        let expected_bytes = n.div_ceil(4);
        assert_eq!(
            qt.data.len(),
            expected_bytes,
            "n={n}: expected {expected_bytes} packed bytes, got {}",
            qt.data.len()
        );
    }
}

/// The `QuantizationType` stored in the result must be `I2S`.
#[test]
fn test_i2s_quantization_type_label() {
    let tensor = make_tensor(vec![1.0, -1.0, 0.5], &[3]);
    let qt = I2SQuantizer::new().quantize_tensor(&tensor).unwrap();
    assert_eq!(qt.qtype, QuantizationType::I2S);
}

/// `quantize_weights` is a convenience wrapper; ensure it produces the same
/// result as the full `quantize_tensor` path.
#[test]
fn test_i2s_quantize_weights_helper_matches_tensor_path() {
    let weights = vec![0.5f32, -0.5, 1.0, -1.0, 0.25, -0.25, 0.75, -0.75];
    let q = I2SQuantizer::new();

    let via_helper = q.quantize_weights(&weights).unwrap();
    let tensor = make_tensor(weights.clone(), &[8]);
    let via_tensor = q.quantize_tensor(&tensor).unwrap();

    assert_eq!(via_helper.data, via_tensor.data, "packed bytes must match");
    assert_eq!(via_helper.scales.len(), via_tensor.scales.len(), "scale counts must match");
}

/// `I2SQuantizer::with_block_size(0)` should clamp to the internal minimum (4)
/// and still produce a valid quantization, not panic.
#[test]
fn test_i2s_block_size_clamped_to_minimum() {
    let q = I2SQuantizer::with_block_size(0);
    assert!(q.is_available());
    let tensor = make_tensor(vec![1.0f32; 8], &[8]);
    let qt = q.quantize_tensor(&tensor).unwrap();
    assert!(!qt.data.is_empty());
}

/// Large (out-of-range) input values must not panic; the quantizer clamps them.
#[test]
fn test_i2s_large_values_do_not_panic() {
    let data = vec![1e10f32, -1e10, 1e20, -1e20, f32::MAX, f32::MIN];
    // Pad to a multiple of block size
    let mut padded = data.clone();
    padded.resize(32, 0.0);
    let tensor = make_tensor(padded, &[32]);
    // Should not panic
    let result = I2SQuantizer::new().quantize_tensor(&tensor);
    assert!(result.is_ok(), "large values must not cause an error: {:?}", result.err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 2 – TL1 quantization
// ═══════════════════════════════════════════════════════════════════════════

/// A symmetric `LookupTable` (use_asymmetric=false) maps 0.0 to the zero-
/// level code and dequantizes it back to a value close to zero.
#[test]
fn test_tl1_lookup_table_zero_is_symmetric() {
    let lut = LookupTable::new(-1.0, 1.0, 2, false);
    let code = lut.quantize(0.0);
    let decoded = lut.dequantize(code);
    assert!(decoded.abs() < 0.6, "symmetric LUT: 0.0 should decode near 0.0, got {decoded}");
}

/// `LookupTable` round-trip for a range of values: decoded ≈ original within
/// one quantisation step.
#[test]
fn test_tl1_lookup_table_roundtrip() {
    let lut = LookupTable::new(-2.0, 2.0, 2, false);
    // For 2-bit symmetric, step ≈ 2*abs_max / (num_levels-1) = 4/3 ≈ 1.33
    let step = 4.0f32 / 3.0;
    for &v in &[-1.5f32, -0.5, 0.0, 0.5, 1.5] {
        let decoded = lut.dequantize(lut.quantize(v));
        assert!(
            (v - decoded).abs() <= step + 1e-4,
            "LUT round-trip failed for {v}: decoded {decoded}, step {step}"
        );
    }
}

/// TL1 default config uses 2 bits → 4 levels (2^2).
#[test]
fn test_tl1_precision_bits_gives_four_levels() {
    let cfg = TL1Config::default();
    assert_eq!(cfg.precision_bits, 2);
    assert_eq!(1usize << cfg.precision_bits, 4);
}

/// `TL1Quantizer::quantize_tensor` must label its output as `QuantizationType::TL1`.
#[test]
fn test_tl1_quantization_type_label() {
    let tensor = make_tensor(vec![0.5f32; 64], &[64]);
    let qt = TL1Quantizer::new().quantize_tensor(&tensor).unwrap();
    assert_eq!(qt.qtype, QuantizationType::TL1);
}

/// TL1 round-trip on a multi-block tensor (256 elements, 4 blocks of 64):
/// max error should not exceed twice the quantisation step.
#[test]
fn test_tl1_multi_block_roundtrip_accuracy() {
    let data: Vec<f32> = (0..256).map(|i| (i as f32 / 128.0) - 1.0).collect();
    let tensor = make_tensor(data.clone(), &[256]);
    let q = TL1Quantizer::new();
    let qd = q.quantize_tensor(&tensor).unwrap();
    let dq = q.dequantize_tensor(&qd).unwrap();
    let vals = dq.to_vec().unwrap();
    let err = max_abs_err(&data, &vals);
    // 2-bit over range [-1,1] → step ≈ 2/3; allow 2× step + margin
    assert!(err < 1.5, "TL1 multi-block max error {err} exceeds 1.5");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 3 – TL2 quantization
// ═══════════════════════════════════════════════════════════════════════════

/// `VectorizedLookupTable` must have a forward table of 256 entries and a
/// reverse table of `2^bits` entries (4 for 2-bit).
#[test]
fn test_tl2_lookup_table_sizes() {
    let lut = VectorizedLookupTable::new(-1.0, 1.0, 2);
    assert_eq!(lut.forward_len(), 256, "forward table must have 256 entries");
    assert_eq!(lut.reverse_len(), 4, "reverse table must have 4 entries for 2-bit");
}

/// Sign preservation: positive inputs must decode to positive values and
/// negative inputs to negative values.
#[test]
fn test_tl2_sign_is_preserved() {
    let lut = VectorizedLookupTable::new(-1.0, 1.0, 2);
    let pos_code = lut.quantize(0.8);
    let neg_code = lut.quantize(-0.8);
    assert!(
        lut.dequantize(pos_code) > 0.0,
        "positive input should decode positive, got {}",
        lut.dequantize(pos_code)
    );
    assert!(
        lut.dequantize(neg_code) < 0.0,
        "negative input should decode negative, got {}",
        lut.dequantize(neg_code)
    );
}

/// TL2 quantised result must carry `QuantizationType::TL2`.
#[test]
fn test_tl2_quantization_type_label() {
    let tensor = make_tensor(vec![0.5f32; 128], &[128]);
    let qt = TL2Quantizer::new().quantize_tensor(&tensor).unwrap();
    assert_eq!(qt.qtype, QuantizationType::TL2);
}

/// `get_or_create_lookup_table` returns a consistent table for the same range.
#[test]
fn test_tl2_lookup_table_is_consistent_for_same_range() {
    let q = TL2Quantizer::new();
    let lut_a = q.get_or_create_lookup_table(-1.0, 1.0);
    let lut_b = q.get_or_create_lookup_table(-1.0, 1.0);
    // Both tables must have the same sizes (structural equivalence)
    assert_eq!(lut_a.forward_len(), lut_b.forward_len());
    assert_eq!(lut_a.reverse_len(), lut_b.reverse_len());
}

/// TL2 multi-block round-trip: error bounded by ~2× quantisation step.
#[test]
fn test_tl2_multi_block_roundtrip_accuracy() {
    let data: Vec<f32> = (0..128).map(|i| (i as f32 / 64.0) - 1.0).collect();
    let tensor = make_tensor(data.clone(), &[128]);
    let q = TL2Quantizer::new();
    let qd = q.quantize_tensor(&tensor).unwrap();
    let dq = q.dequantize_tensor(&qd).unwrap();
    let vals = dq.to_vec().unwrap();
    let err = max_abs_err(&data, &vals);
    assert!(err < 1.5, "TL2 multi-block max error {err} exceeds 1.5");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 4 – QK256 block structure
// ═══════════════════════════════════════════════════════════════════════════

/// Constants must match the GGML specification: 256 elements per block and
/// 64 packed bytes per block (2 bits × 256 / 8 = 64).
#[test]
fn test_qk256_constants() {
    assert_eq!(QK256_BLOCK, 256);
    assert_eq!(QK256_PACKED_BYTES, 64);
    assert_eq!(
        QK256_BLOCK * 2 / 8,
        QK256_PACKED_BYTES,
        "2 bits per element × 256 = 512 bits = 64 bytes"
    );
}

/// `code_to_f32` maps exactly to the GGML-verified LUT: {-2, -1, +1, +2}.
#[test]
fn test_qk256_code_to_f32_all_valid_codes() {
    assert_eq!(code_to_f32(0), -2.0);
    assert_eq!(code_to_f32(1), -1.0);
    assert_eq!(code_to_f32(2), 1.0);
    assert_eq!(code_to_f32(3), 2.0);
}

/// A 64-byte block of all 0x00 must unpack to 256 codes of 0.
#[test]
fn test_qk256_unpack_all_zeros_block() {
    let qs: [u8; QK256_PACKED_BYTES] = [0u8; QK256_PACKED_BYTES];
    let mut codes = [0u8; QK256_BLOCK];
    unpack_qk256_block(&qs, &mut codes);
    assert!(codes.iter().all(|&c| c == 0), "all-zero block must unpack to all code-0");
}

/// A 64-byte block of all 0xFF must unpack to 256 codes of 3 (all bits set → 0b11).
#[test]
fn test_qk256_unpack_all_ones_block() {
    let qs: [u8; QK256_PACKED_BYTES] = [0xFFu8; QK256_PACKED_BYTES];
    let mut codes = [0u8; QK256_BLOCK];
    unpack_qk256_block(&qs, &mut codes);
    assert!(codes.iter().all(|&c| c == 3), "0xFF block must unpack to all code-3");
}

/// `I2SQk256NoScale::new` rejects a data buffer that is far too small.
#[test]
fn test_qk256_new_wrong_size_errors() {
    // 4 rows × 256 cols → row_stride = 64 bytes → total = 256 bytes
    // Provide only 1 byte – must fail
    let result = I2SQk256NoScale::new(4, 256, vec![0u8; 1]);
    assert!(result.is_err(), "mismatched size must return Err");
}

/// `I2SQk256NoScale::new` succeeds when the data size matches exactly.
#[test]
fn test_qk256_new_correct_size_succeeds() {
    // 2 rows × 256 cols → row_stride = 64 bytes → total = 128 bytes
    let data = vec![0u8; 128];
    let result = I2SQk256NoScale::new(2, 256, data);
    assert!(result.is_ok(), "correct size must succeed");
    let q = result.unwrap();
    assert_eq!(q.rows, 2);
    assert_eq!(q.cols, 256);
    assert_eq!(q.row_stride_bytes, 64);
}

/// `row_bytes` returns exactly `row_stride_bytes` bytes for each row.
#[test]
fn test_qk256_row_bytes_slice_length() {
    let rows = 3usize;
    let cols = 512usize;
    let blocks_per_row = cols.div_ceil(QK256_BLOCK);
    let row_stride = blocks_per_row * QK256_PACKED_BYTES; // 2 * 64 = 128
    let data = vec![0xABu8; rows * row_stride];
    let q = I2SQk256NoScale::new(rows, cols, data).unwrap();
    for r in 0..rows {
        assert_eq!(q.row_bytes(r).len(), row_stride, "row {r} slice length mismatch");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 5 – Property-based tests (proptest)
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    /// I2_S quantization followed by dequantization must produce values within
    /// ±2 × scale of the original (worst-case 2-bit rounding error).
    #[test]
    fn prop_i2s_roundtrip_error_bounded_by_scale(
        data in prop::collection::vec(-10.0f32..10.0f32, 4..128usize),
    ) {
        // Pad to a multiple of 4 for clean packing
        let mut padded = data.clone();
        let rem = padded.len() % 4;
        if rem != 0 { padded.resize(padded.len() + (4 - rem), 0.0); }
        let n = padded.len();

        let tensor = make_tensor(padded.clone(), &[n]);
        let q = I2SQuantizer::new();
        let qd = q.quantize_tensor(&tensor).unwrap();

        // Each scale corresponds to one block; the max error is bounded by the scale.
        let block_size = qd.block_size;
        let deq = q.dequantize_tensor(&qd).unwrap();
        let vals = deq.to_vec().unwrap();

        for (block_idx, (chunk_in, chunk_out)) in
            padded.chunks(block_size).zip(vals.chunks(block_size)).enumerate()
        {
            let scale = qd.scales[block_idx.min(qd.scales.len() - 1)];
            for (orig, decoded) in chunk_in.iter().zip(chunk_out.iter()) {
                let err = (orig - decoded).abs();
                // Allow 2× scale as the worst-case quantization interval
                prop_assert!(
                    err <= 2.0 * scale + 1e-4,
                    "block {block_idx}: error {err} > 2×scale {}",
                    scale
                );
            }
        }
    }

    /// After I2_S dequantization, all output values must be finite.
    #[test]
    fn prop_i2s_dequantized_values_are_finite(
        data in prop::collection::vec(-100.0f32..100.0f32, 4..64usize),
    ) {
        let mut padded = data.clone();
        let rem = padded.len() % 4;
        if rem != 0 { padded.resize(padded.len() + (4 - rem), 0.0); }
        let n = padded.len();
        let tensor = make_tensor(padded, &[n]);
        let q = I2SQuantizer::new();
        let qd = q.quantize_tensor(&tensor).unwrap();
        let deq = q.dequantize_tensor(&qd).unwrap();
        let vals = deq.to_vec().unwrap();
        for &v in &vals {
            prop_assert!(v.is_finite(), "dequantised value {v} is not finite");
        }
    }

    /// `calculate_grouped_scales` must return exactly `ceil(n / block_size)` scales,
    /// and every scale must be positive and finite.
    #[test]
    fn prop_grouped_scales_count_and_sign(
        data in prop::collection::vec(-1000.0f32..1000.0f32, 1..512usize),
        block_size in 4usize..64usize,
    ) {
        let scales = calculate_grouped_scales(&data, block_size, 2);
        let expected = data.len().div_ceil(block_size);
        prop_assert_eq!(scales.len(), expected, "scale count mismatch");
        for &s in &scales {
            prop_assert!(s.is_finite() && s > 0.0, "scale {s} is not positive finite");
        }
    }

    /// All 2-bit codes unpacked from any 64-byte QK256 block must be in [0, 3].
    #[test]
    fn prop_qk256_unpacked_codes_in_valid_range(
        raw in prop::collection::vec(any::<u8>(), QK256_PACKED_BYTES),
    ) {
        let mut qs = [0u8; QK256_PACKED_BYTES];
        qs.copy_from_slice(&raw);
        let mut codes = [0u8; QK256_BLOCK];
        unpack_qk256_block(&qs, &mut codes);
        for &code in &codes {
            prop_assert!(code <= 3, "code {code} is out of [0,3]");
        }
    }

    /// `pack_2bit_values` then `unpack_2bit_values` is the identity for
    /// values in the 2-bit signed range {-2, -1, 0, 1}.
    #[test]
    fn prop_pack_unpack_2bit_roundtrip(
        values in prop::collection::vec(-2i8..=1i8, 4..128usize),
    ) {
        let packed = pack_2bit_values(&values);
        let unpacked = unpack_2bit_values(&packed, values.len());
        prop_assert_eq!(unpacked, values, "pack→unpack must be identity");
    }

    /// TL1 dequantised output is always finite for finite inputs.
    #[test]
    fn prop_tl1_dequantized_values_finite(
        data in prop::collection::vec(-10.0f32..10.0f32, 1..256usize),
    ) {
        let n = data.len();
        let tensor = make_tensor(data, &[n]);
        let q = TL1Quantizer::new();
        #[allow(clippy::collapsible_if)]
        if let Ok(qd) = q.quantize_tensor(&tensor) {
            if let Ok(deq) = q.dequantize_tensor(&qd) {
                let vals = deq.to_vec().unwrap();
                for &v in &vals {
                    prop_assert!(v.is_finite(), "TL1 dequantised value {v} is not finite");
                }
            }
        }
    }

    /// TL2 dequantised output is always finite for finite inputs.
    #[test]
    fn prop_tl2_dequantized_values_finite(
        data in prop::collection::vec(-10.0f32..10.0f32, 1..256usize),
    ) {
        let n = data.len();
        let tensor = make_tensor(data, &[n]);
        let q = TL2Quantizer::new();
        #[allow(clippy::collapsible_if)]
        if let Ok(qd) = q.quantize_tensor(&tensor) {
            if let Ok(deq) = q.dequantize_tensor(&qd) {
                let vals = deq.to_vec().unwrap();
                for &v in &vals {
                    prop_assert!(v.is_finite(), "TL2 dequantised value {v} is not finite");
                }
            }
        }
    }
}
