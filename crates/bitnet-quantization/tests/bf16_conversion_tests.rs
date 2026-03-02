//! BF16 ↔ F16 ↔ F32 conversion correctness tests.
//!
//! Critical for Phi-4 and HuggingFace models that ship BF16 weights.
//! Validates bit-exact representations, cross-format conversions,
//! batch operations, and model-realistic weight patterns.

use half::{bf16, f16};
use std::time::Instant;

// ────────────────────────────────────────────────────────────────────
// BF16 basics (10 tests)
// ────────────────────────────────────────────────────────────────────

#[test]
fn bf16_zero_representation() {
    let pos = bf16::from_f32(0.0);
    let neg = bf16::from_f32(-0.0);
    assert_eq!(pos.to_f32(), 0.0);
    assert_eq!(neg.to_f32(), -0.0_f32);
    // Positive and negative zero have different bit patterns
    assert_eq!(pos.to_bits(), 0x0000);
    assert_eq!(neg.to_bits(), 0x8000);
}

#[test]
fn bf16_one_representation() {
    let one = bf16::from_f32(1.0);
    assert_eq!(one.to_f32(), 1.0);
    // IEEE 754 BF16: sign=0, exp=127 (0x3F), mantissa=0 → 0x3F80
    assert_eq!(one.to_bits(), 0x3F80);
}

#[test]
fn bf16_neg_one_representation() {
    let neg_one = bf16::from_f32(-1.0);
    assert_eq!(neg_one.to_f32(), -1.0);
    assert_eq!(neg_one.to_bits(), 0xBF80);
}

#[test]
fn bf16_f32_roundtrip_preserves_representable_values() {
    // Values exactly representable in BF16 must survive the round-trip
    let representable: &[f32] = &[
        0.0,
        1.0,
        -1.0,
        2.0,
        0.5,
        0.25,
        128.0,
        -256.0,
        0.00390625,         // 2^-8
        bf16::MAX.to_f32(), // exact BF16 max
    ];
    for &v in representable {
        let rt = bf16::from_f32(v).to_f32();
        assert_eq!(rt, v, "round-trip failed for {v}");
    }
}

#[test]
fn bf16_truncation_behavior() {
    // BF16 has 8-bit mantissa (7 explicit + 1 implicit) vs F32's 24-bit.
    // Values not exactly representable get rounded to nearest even.
    let v: f32 = 1.000_976_6; // 1 + 2^-10, not representable in BF16
    let bf = bf16::from_f32(v);
    let rt = bf.to_f32();
    // Should be rounded to nearest BF16 value, not exact
    assert_ne!(rt, v, "non-representable value should not survive round-trip");
    // The error should be bounded by 1 ULP in BF16 (for values near 1.0, ULP ≈ 2^-7)
    let err = (rt - v).abs();
    assert!(err <= 0.0078125, "error {err} exceeds BF16 ULP near 1.0");
}

#[test]
fn bf16_subnormal_handling() {
    // BF16 subnormals: exponent = 0, mantissa != 0
    // Smallest BF16 subnormal: 2^-133 ≈ 9.18e-41
    let tiny = bf16::from_bits(0x0001); // smallest positive subnormal
    let val = tiny.to_f32();
    assert!(val > 0.0, "subnormal must be positive");
    assert!(val < f32::MIN_POSITIVE, "subnormal must be less than MIN_POSITIVE f32");
    // Round-trip
    let rt = bf16::from_f32(val);
    assert_eq!(rt.to_bits(), 0x0001);
}

#[test]
fn bf16_infinity_and_nan() {
    let pos_inf = bf16::from_f32(f32::INFINITY);
    let neg_inf = bf16::from_f32(f32::NEG_INFINITY);
    let nan = bf16::from_f32(f32::NAN);

    assert!(pos_inf.to_f32().is_infinite() && pos_inf.to_f32() > 0.0);
    assert!(neg_inf.to_f32().is_infinite() && neg_inf.to_f32() < 0.0);
    assert!(nan.to_f32().is_nan());

    assert_eq!(pos_inf.to_bits(), 0x7F80);
    assert_eq!(neg_inf.to_bits(), 0xFF80);
}

#[test]
fn bf16_max_min_values() {
    let max = bf16::MAX;
    let min = bf16::MIN;
    // BF16 max ≈ 3.389e38 (same exponent range as F32)
    assert!((max.to_f32() - 3.3895314e38).abs() < 1e34);
    assert!((min.to_f32() - (-3.3895314e38)).abs() < 1e34);
    // Verify symmetry
    assert_eq!(max.to_f32(), -min.to_f32());
}

#[test]
fn bf16_precision_loss_quantification() {
    // Measure max relative error for random values in [0.01, 100.0]
    let mut max_rel_error: f64 = 0.0;
    let count = 10_000;
    for i in 0..count {
        let v = 0.01 + (i as f32 / count as f32) * 99.99;
        let rt = bf16::from_f32(v).to_f32();
        let rel_err = ((rt as f64 - v as f64) / v as f64).abs();
        if rel_err > max_rel_error {
            max_rel_error = rel_err;
        }
    }
    // BF16 has ~7-bit mantissa, so relative error ≤ 2^-7 ≈ 0.78%
    assert!(
        max_rel_error < 0.008,
        "max relative error {max_rel_error} exceeds expected BF16 bound"
    );
}

#[test]
fn bf16_bit_pattern_is_f32_upper_16() {
    // BF16 is defined as the upper 16 bits of the F32 representation (with rounding)
    let values: &[f32] = &[1.0, -2.5, 42.0, 0.1, 1000.0];
    for &v in values {
        let bf = bf16::from_f32(v);
        // For exactly-representable values, BF16 bits == upper 16 bits of F32
        let f32_upper = (v.to_bits() >> 16) as u16;
        if bf16::from_f32(bf.to_f32()).to_bits() == bf.to_bits() {
            // Only check for values that don't require rounding
            let reconstructed = f32::from_bits((bf.to_bits() as u32) << 16);
            let direct = bf.to_f32();
            assert_eq!(reconstructed, direct, "BF16→F32 should zero-extend lower 16 bits for {v}");
        }
        // Always verify that BF16 bits are related to F32 upper bits
        let diff = (bf.to_bits() as i32 - f32_upper as i32).unsigned_abs();
        assert!(diff <= 1, "BF16 bits should be within ±1 of F32 upper 16 for {v}");
    }
}

// ────────────────────────────────────────────────────────────────────
// BF16 → F16 conversion (5 tests)
// ────────────────────────────────────────────────────────────────────

#[test]
fn bf16_to_f16_in_range_values() {
    // F16 range: ≈ ±65504. Values within this range should convert correctly.
    // Use values exactly representable in BF16 to avoid double-rounding differences.
    let values: &[f32] = &[0.5, 1.0, -1.0, 100.0, -100.0, 42.0, 0.125];
    for &v in values {
        let via_bf16 = f16::from_f32(bf16::from_f32(v).to_f32());
        let direct_f16 = f16::from_f32(v);
        // For BF16-exact values in F16 range, both paths should agree within 1 ULP
        let diff = (via_bf16.to_bits() as i32 - direct_f16.to_bits() as i32).unsigned_abs();
        assert!(
            diff <= 1,
            "BF16→F16 mismatch for {v}: via_bf16={} (bits={:#06x}), direct={} (bits={:#06x})",
            via_bf16.to_f32(),
            via_bf16.to_bits(),
            direct_f16.to_f32(),
            direct_f16.to_bits()
        );
    }
}

#[test]
fn bf16_to_f16_out_of_range_saturates() {
    // BF16 supports values up to ~3.4e38 but F16 max is 65504
    let large = bf16::from_f32(100_000.0);
    let f16_val = f16::from_f32(large.to_f32());
    // Should saturate to infinity in F16
    assert!(
        f16_val.to_f32().is_infinite() || f16_val.to_f32() == f16::MAX.to_f32(),
        "out-of-range BF16 should saturate in F16, got {}",
        f16_val.to_f32()
    );

    let neg_large = bf16::from_f32(-100_000.0);
    let neg_f16 = f16::from_f32(neg_large.to_f32());
    assert!(
        neg_f16.to_f32().is_infinite() || neg_f16.to_f32() == f16::MIN.to_f32(),
        "negative out-of-range BF16 should saturate in F16, got {}",
        neg_f16.to_f32()
    );
}

#[test]
fn bf16_to_f16_special_values_preserved() {
    // Zero
    assert_eq!(f16::from_f32(bf16::ZERO.to_f32()).to_f32(), 0.0);
    // Infinity
    let inf_bf16 = bf16::INFINITY;
    assert!(f16::from_f32(inf_bf16.to_f32()).to_f32().is_infinite());
    // NaN
    let nan_bf16 = bf16::NAN;
    assert!(f16::from_f32(nan_bf16.to_f32()).to_f32().is_nan());
    // Negative zero
    let neg_zero = bf16::from_f32(-0.0);
    let f16_neg_zero = f16::from_f32(neg_zero.to_f32());
    assert_eq!(f16_neg_zero.to_f32(), 0.0);
    assert!(f16_neg_zero.is_sign_negative());
}

#[test]
fn bf16_to_f16_subnormal_conversion() {
    // BF16 subnormals are extremely tiny (< ~1.17e-38)
    // These are well within F16 subnormal range, so they'll become F16 zero (flush to zero)
    let bf16_subnormal = bf16::from_bits(0x0001);
    let f32_val = bf16_subnormal.to_f32();
    let f16_val = f16::from_f32(f32_val);
    // BF16 subnormals are much smaller than F16 can represent, expect zero
    assert_eq!(f16_val.to_f32(), 0.0, "BF16 subnormal ({f32_val:e}) should flush to zero in F16");
}

#[test]
fn bf16_f16_f32_roundtrip_error_bound() {
    // Compare: BF16→F16→F32 vs BF16→F32 directly
    // The extra F16 step should add at most F16-level error
    let mut max_extra_error: f64 = 0.0;
    let test_values: Vec<f32> = (0..1000)
        .map(|i| -10.0 + (i as f32 / 1000.0) * 20.0) // [-10, 10]
        .collect();

    for v in &test_values {
        let bf = bf16::from_f32(*v);
        let direct = bf.to_f32() as f64;
        let via_f16 = f16::from_f32(bf.to_f32()).to_f32() as f64;
        let extra_err = (via_f16 - direct).abs();
        if extra_err > max_extra_error {
            max_extra_error = extra_err;
        }
    }
    // F16 precision near 10.0 has ULP ≈ 0.01, so extra error should be small
    assert!(
        max_extra_error < 0.02,
        "BF16→F16→F32 adds too much error vs BF16→F32: {max_extra_error}"
    );
}

// ────────────────────────────────────────────────────────────────────
// Batch conversion (5 tests)
// ────────────────────────────────────────────────────────────────────

#[test]
fn batch_bf16_to_f32_conversion() {
    let bf16_weights: Vec<bf16> =
        (0..1024).map(|i| bf16::from_f32(i as f32 * 0.01 - 5.0)).collect();
    let f32_weights: Vec<f32> = bf16_weights.iter().map(|b| b.to_f32()).collect();

    assert_eq!(f32_weights.len(), 1024);
    for (i, (&bf, &f)) in bf16_weights.iter().zip(f32_weights.iter()).enumerate() {
        assert_eq!(bf.to_f32(), f, "mismatch at index {i}");
    }
}

#[test]
fn batch_conversion_10m_elements_perf() {
    let n = 10_000_000;
    let bf16_data: Vec<bf16> = (0..n).map(|i| bf16::from_f32((i as f32).sin())).collect();

    let start = Instant::now();
    let f32_data: Vec<f32> = bf16_data.iter().map(|b| b.to_f32()).collect();
    let elapsed = start.elapsed();

    assert_eq!(f32_data.len(), n);
    assert!(elapsed.as_secs_f64() < 1.0, "10M BF16→F32 conversions took {elapsed:?}, expected <1s");
}

#[test]
fn batch_conversion_no_extra_allocations() {
    let n = 1000;
    let bf16_data: Vec<bf16> = (0..n).map(|i| bf16::from_f32(i as f32)).collect();

    // Pre-allocate and convert in place to verify no extra allocs needed
    let mut f32_data = Vec::with_capacity(n);
    for b in &bf16_data {
        f32_data.push(b.to_f32());
    }
    assert_eq!(f32_data.len(), n);
    assert!(f32_data.capacity() >= n);
    // Capacity should be exactly what we pre-allocated (no realloc)
    assert_eq!(f32_data.capacity(), n, "unexpected reallocation occurred");
}

#[test]
fn batch_conversion_empty_vector() {
    let empty: Vec<bf16> = vec![];
    let result: Vec<f32> = empty.iter().map(|b| b.to_f32()).collect();
    assert!(result.is_empty());
}

#[test]
fn batch_conversion_single_element() {
    let single = [bf16::from_f32(std::f32::consts::PI)];
    let result: Vec<f32> = single.iter().map(|b| b.to_f32()).collect();
    assert_eq!(result.len(), 1);
    // PI in BF16 should be close to PI
    assert!((result[0] - std::f32::consts::PI).abs() < 0.02);
}

// ────────────────────────────────────────────────────────────────────
// Model-realistic patterns (5 tests)
// ────────────────────────────────────────────────────────────────────

#[test]
fn model_layernorm_weight_range() {
    // LayerNorm weights are typically ~0.9–1.1
    let weights: Vec<f32> = (0..1000).map(|i| 0.9 + (i as f32 / 1000.0) * 0.2).collect();

    for &w in &weights {
        let rt = bf16::from_f32(w).to_f32();
        let err = (rt - w).abs();
        assert!(err < 1e-2, "LayerNorm weight {w} has BF16 error {err} >= 1e-2");
    }
}

#[test]
fn model_attention_weight_range() {
    // Attention weights are typically small: ~-0.01 to 0.01
    let weights: Vec<f32> = (0..1000).map(|i| -0.01 + (i as f32 / 1000.0) * 0.02).collect();

    for &w in &weights {
        let rt = bf16::from_f32(w).to_f32();
        let err = (rt - w).abs();
        // Near zero, absolute error for BF16 is very small
        assert!(err < 1e-4, "attention weight {w} has BF16 error {err} >= 1e-4");
    }
}

#[test]
fn model_large_embedding_matrix_conversion() {
    // Simulate a slice of an embedding matrix: 100352 × 128 = 12_845_056 elements
    // We'll test a representative slice of 100_352 elements
    let n = 100_352;
    let embeddings: Vec<bf16> = (0..n)
        .map(|i| {
            // Simulate typical embedding values: roughly Gaussian, range ~[-2, 2]
            let x = (i as f32 * 0.618_034) % 4.0 - 2.0; // golden ratio hash
            bf16::from_f32(x)
        })
        .collect();

    let f32_emb: Vec<f32> = embeddings.iter().map(|b| b.to_f32()).collect();
    assert_eq!(f32_emb.len(), n);

    // Verify round-trip consistency: bf16→f32→bf16 should be identity
    for (i, (&original, &converted)) in embeddings.iter().zip(f32_emb.iter()).enumerate() {
        let back = bf16::from_f32(converted);
        assert_eq!(
            original.to_bits(),
            back.to_bits(),
            "embedding round-trip failed at index {i}: original bits={:#06x}, back bits={:#06x}",
            original.to_bits(),
            back.to_bits()
        );
    }
}

#[test]
fn model_mixed_sign_preserves_sign() {
    let values: &[f32] =
        &[1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 0.001, -0.001, 1e-10, -1e-10, 1e30, -1e30];

    for &v in values {
        let bf = bf16::from_f32(v);
        let rt = bf.to_f32();
        if v == 0.0 {
            continue;
        }
        assert_eq!(v.is_sign_positive(), rt.is_sign_positive(), "sign flipped for {v} → {rt}");
    }
}

#[test]
fn model_conversion_error_histogram() {
    // Measure conversion error percentiles for random-like weights in [-1, 1]
    let n = 100_000;
    let mut errors: Vec<f64> = (0..n)
        .map(|i| {
            let v = (i as f64 * 0.6180339887).fract() as f32 * 2.0 - 1.0;
            let rt = bf16::from_f32(v).to_f32();
            (rt as f64 - v as f64).abs()
        })
        .collect();

    errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = errors[n / 2];
    let p90 = errors[n * 9 / 10];
    let p99 = errors[n * 99 / 100];
    let p100 = errors[n - 1];

    // BF16 ULP for values in [-1,1]: ~2^-8 = 0.00390625
    assert!(p50 < 0.004, "p50 error {p50} too large (expected <0.004 for BF16)");
    assert!(p90 < 0.004, "p90 error {p90} too large (expected <0.004 for BF16)");
    assert!(p99 < 0.004, "p99 error {p99} too large (expected <0.004 for BF16)");
    assert!(p100 < 0.005, "max error {p100} too large (expected <0.005 for BF16)");

    // Print percentiles for visibility in test output
    eprintln!("BF16 conversion error percentiles (n={n}):");
    eprintln!("  p50:  {p50:.2e}");
    eprintln!("  p90:  {p90:.2e}");
    eprintln!("  p99:  {p99:.2e}");
    eprintln!("  p100: {p100:.2e}");
}
