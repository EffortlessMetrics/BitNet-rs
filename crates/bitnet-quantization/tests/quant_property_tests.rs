//! Wave 5 property tests for `bitnet-quantization`.
//!
//! These tests cover invariants not already addressed in `property_tests.rs`:
//!
//! 1. **QK256 unpack round-trip** – pack→unpack recovers original 2-bit codes.
//! 2. **QK256 code_to_f32 range** – all valid codes map to {-1, 0, 1}.
//! 3. **Quantize-dequantize sign preservation** – non-zero inputs preserve sign.
//! 4. **Zero vector quantizes to all-zero output** (I2S).
//! 5. **validate_numerical_input accepts all-finite data**.
//! 6. **validate_data_shape_consistency rejects mismatched lengths**.
//! 7. **estimate_quantization_memory is monotonic in element count**.
//! 8. **QuantizerFactory::create is deterministic** (same type → same quantizer name).

#![cfg(feature = "cpu")]

use bitnet_common::QuantizationType;
use bitnet_quantization::i2s_qk256::{
    QK256_BLOCK, QK256_PACKED_BYTES, code_to_f32, unpack_qk256_block,
};
use bitnet_quantization::validation::{
    estimate_quantization_memory, validate_data_shape_consistency, validate_numerical_input,
};
use bitnet_quantization::{I2SQuantizer, QuantizerFactory};
use proptest::prelude::*;

// ── QK256 unpack round-trip ─────────────────────────────────────────────────

proptest! {
    /// Unpacking and re-packing the BitNet.cpp grouped QK256 layout is stable.
    ///
    /// Each byte stores one group position across four 32-value lanes:
    /// [7:6], [5:4], [3:2], [1:0].
    /// This property ensures the bit-shift arithmetic is self-consistent.
    #[test]
    fn prop_qk256_unpack_recovers_packed_codes(
        raw_bytes in prop::collection::vec(any::<u8>(), QK256_PACKED_BYTES..=QK256_PACKED_BYTES),
    ) {
        let qs64: &[u8; QK256_PACKED_BYTES] = raw_bytes.as_slice().try_into().unwrap();
        let mut codes = [0u8; QK256_BLOCK];
        unpack_qk256_block(qs64, &mut codes);

        // Every unpacked code must be in 0..=3.
        for (i, &c) in codes.iter().enumerate() {
            prop_assert!(c <= 3, "code[{}] = {} is out of range 0..=3", i, c);
        }

        for (byte_idx, &original_byte) in qs64.iter().enumerate() {
            let chunk = byte_idx / 32;
            let gp = byte_idx % 32;
            let base = chunk * 128;
            let repacked = (codes[base + gp] << 6)
                | (codes[base + 32 + gp] << 4)
                | (codes[base + 64 + gp] << 2)
                | codes[base + 96 + gp];
            prop_assert_eq!(
                repacked, original_byte,
                "re-packed byte[{}] = {:#04x} != original {:#04x}",
                byte_idx, repacked, original_byte
            );
        }
    }
}

// ── QK256 code_to_f32 range ─────────────────────────────────────────────────

proptest! {
    /// `code_to_f32` maps every valid code (0..=3) to exactly one of {-1, 0, 1}.
    #[test]
    fn prop_code_to_f32_maps_to_expected_set(code in 0u8..=3u8) {
        let v = code_to_f32(code);
        let valid = [-1.0_f32, 0.0, 1.0];
        prop_assert!(
            valid.contains(&v),
            "code_to_f32({}) = {}; expected one of {:?}", code, v, valid
        );
    }

    /// BitNet.cpp I2_S maps both zero-like codes to 0.0.
    #[test]
    fn prop_code_to_f32_zero_codes_are_zero(code in prop_oneof![Just(1u8), Just(3u8)]) {
        prop_assert_eq!(code_to_f32(code), 0.0);
    }
}

// ── I2S sign preservation ───────────────────────────────────────────────────

proptest! {
    /// For a block of same-sign values, I2S quantize→dequantize preserves sign
    /// (positive inputs → non-negative outputs, negative → non-positive).
    ///
    /// We use a block of identical values to avoid per-element sign flips from
    /// quantization noise at near-zero crossings.
    #[test]
    fn prop_i2s_roundtrip_preserves_sign_for_uniform_block(
        magnitude in 0.5f32..100.0f32,
        positive in any::<bool>(),
    ) {
        let value = if positive { magnitude } else { -magnitude };
        let data = vec![value; 32];
        let q = I2SQuantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let deq_tensor = q.dequantize_tensor(&quantized).unwrap();
        let deq = deq_tensor.to_vec().unwrap();

        for (i, &d) in deq.iter().enumerate() {
            if positive {
                prop_assert!(
                    d >= 0.0,
                    "positive input {} → deq[{}] = {} is negative", value, i, d
                );
            } else {
                prop_assert!(
                    d <= 0.0,
                    "negative input {} → deq[{}] = {} is positive", value, i, d
                );
            }
        }
    }
}

// ── I2S zero-vector invariant ───────────────────────────────────────────────

proptest! {
    /// An all-zero input always quantizes to all-zero dequantized output.
    #[test]
    fn prop_i2s_zero_vector_roundtrips_to_zero(
        n_blocks in 1usize..8usize,
    ) {
        let n = n_blocks * 32;
        let data = vec![0.0f32; n];
        let q = I2SQuantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let deq_tensor = q.dequantize_tensor(&quantized).unwrap();
        let deq = deq_tensor.to_vec().unwrap();

        for (i, &v) in deq.iter().enumerate() {
            prop_assert_eq!(v, 0.0, "zero input → deq[{}] = {} should be 0.0", i, v);
        }
    }
}

// ── validate_numerical_input accepts finite data ────────────────────────────

proptest! {
    /// `validate_numerical_input` always succeeds for finite, non-empty data.
    #[test]
    fn prop_validate_numerical_input_accepts_finite(
        data in prop::collection::vec(-1e6f32..1e6f32, 1..256),
    ) {
        let result = validate_numerical_input(&data);
        prop_assert!(
            result.is_ok(),
            "finite data should always pass validation; got {:?}",
            result
        );
    }
}

// ── validate_data_shape_consistency rejects mismatches ──────────────────────

proptest! {
    /// When data length differs from the product of the shape, validation fails.
    #[test]
    fn prop_shape_mismatch_always_rejected(
        data_len in 1usize..128usize,
        shape_dim in 2usize..128usize,
    ) {
        // Ensure mismatch: pick a data_len that != shape_dim.
        prop_assume!(data_len != shape_dim);

        let data = vec![0.0f32; data_len];
        let shape = [shape_dim];
        let result = validate_data_shape_consistency(&data, &shape);
        prop_assert!(
            result.is_err(),
            "data_len={} vs shape=[{}] should be rejected", data_len, shape_dim
        );
    }

    /// When data length matches the product of the shape, validation succeeds.
    #[test]
    fn prop_shape_match_always_accepted(
        rows in 1usize..32usize,
        cols in 1usize..32usize,
    ) {
        let n = rows * cols;
        let data = vec![0.0f32; n];
        let shape = [rows, cols];
        let result = validate_data_shape_consistency(&data, &shape);
        prop_assert!(
            result.is_ok(),
            "data_len={} vs shape=[{}, {}] should succeed; got {:?}",
            n, rows, cols, result
        );
    }
}

// ── estimate_quantization_memory monotonicity ───────────────────────────────

proptest! {
    /// Memory estimate is monotonically non-decreasing in the number of elements.
    #[test]
    fn prop_memory_estimate_monotonic_in_elements(
        a in 32usize..10_000usize,
        b in 32usize..10_000usize,
    ) {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let mem_lo = estimate_quantization_memory(lo, 2, 32);
        let mem_hi = estimate_quantization_memory(hi, 2, 32);
        prop_assert!(
            mem_lo <= mem_hi,
            "estimate({}) = {} > estimate({}) = {}", lo, mem_lo, hi, mem_hi
        );
    }
}

// ── QuantizerFactory determinism ────────────────────────────────────────────

proptest! {
    /// `QuantizerFactory::create` returns the same quantization type for the same input.
    #[test]
    fn prop_quantizer_factory_deterministic(
        qtype in prop_oneof![
            Just(QuantizationType::I2S),
            Just(QuantizationType::TL1),
            Just(QuantizationType::TL2),
        ],
    ) {
        let a = QuantizerFactory::create(qtype);
        let b = QuantizerFactory::create(qtype);
        prop_assert_eq!(
            a.quantization_type(), b.quantization_type(),
            "factory must be deterministic for {:?}", qtype
        );
    }
}
