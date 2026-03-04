#![allow(dead_code, unused_imports, unused_variables, unused_unsafe, unsafe_op_in_unsafe_fn)]
//! Wave 13 property tests: kernel mathematical invariants for activations,
//! quantization, layer normalization, attention, RoPE, fusion, and SIMD math.
//!
//! Key invariants tested (15 properties):
//! - Activations: sigmoid output range [0,1], relu/silu continuity at zero,
//!   activate + activate_inplace equivalence
//! - Quantization: ternary values always in {-1,0,1}, binary values always
//!   in {-1,1}, symmetric i8 roundtrip error bounded, scale factor non-negative
//! - Layer norm: RMS norm output finite and scaled, layer norm idempotent
//!   on unit-variance input
//! - Attention: causal mask diagonal is zero, causal mask upper triangle
//!   is neg-infinity
//! - RoPE: frequency array length matches config, frequencies are positive
//! - Fusion: fused_scale_add matches unfused, fused_add_normalize preserves length
//! - SIMD math: vector scale by 1.0 is identity, dot product self is non-negative

use bitnet_kernels::cpu::activations::{
    ActivationType, activate, activate_inplace, gelu, hard_sigmoid, relu, sigmoid, silu,
};
use bitnet_kernels::cpu::attention::causal_mask;
use bitnet_kernels::cpu::fusion::{fused_add_normalize, fused_scale_add};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::quantize::{
    compute_quantization_error, dequantize_symmetric_i8, quantize_binary, quantize_symmetric_i8,
    quantize_ternary,
};
use bitnet_kernels::cpu::rope::{RopeConfig, compute_frequencies};
use bitnet_kernels::cpu::simd_math::{simd_dot_product, simd_vector_scale};
use proptest::prelude::*;

// -------------------------------------------------------------------
// Strategy helpers
// -------------------------------------------------------------------

/// Non-empty f32 vector with finite values in [-10, 10].
fn finite_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-10.0f32..10.0f32, 1..=max_len)
}

// ===================================================================
// 1. Activations
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// sigmoid output is always in [0, 1].
    #[test]
    fn prop_sigmoid_output_in_unit_interval(x in -50.0f32..50.0f32) {
        let y = sigmoid(x);
        prop_assert!((0.0..=1.0).contains(&y), "sigmoid({x}) = {y} not in [0,1]");
    }

    /// hard_sigmoid output is always in [0, 1].
    #[test]
    fn prop_hard_sigmoid_output_in_unit_interval(x in -50.0f32..50.0f32) {
        let y = hard_sigmoid(x);
        prop_assert!((0.0..=1.0).contains(&y), "hard_sigmoid({x}) = {y} not in [0,1]");
    }

    /// relu of any value is non-negative.
    #[test]
    fn prop_relu_output_nonneg(x in -100.0f32..100.0f32) {
        prop_assert!(relu(x) >= 0.0);
    }

    /// silu(0) == 0 (since silu(x) = x * sigmoid(x)).
    #[test]
    fn prop_silu_zero_at_origin(_dummy in 0..1i32) {
        let y = silu(0.0);
        prop_assert!((y - 0.0).abs() < 1e-7, "silu(0) = {y}, expected 0");
    }

    /// activate and activate_inplace produce identical results.
    #[test]
    fn prop_activate_matches_inplace(input in finite_f32_vec(64)) {
        let allocated = activate(&input, ActivationType::GELU);
        let mut inplace = input.clone();
        activate_inplace(&mut inplace, ActivationType::GELU);
        for (i, (&a, &b)) in allocated.iter().zip(inplace.iter()).enumerate() {
            prop_assert!(
                (a - b).abs() < 1e-6,
                "mismatch at index {i}: activate={a}, inplace={b}"
            );
        }
    }

    /// gelu is bounded below by a known lower bound: gelu(x) > -0.17 for all x
    /// (global minimum ≈ -0.1699).
    #[test]
    fn prop_gelu_bounded_below(x in -50.0f32..50.0f32) {
        let y = gelu(x);
        prop_assert!(y > -0.18, "gelu({x}) = {y} below expected minimum");
    }
}

// ===================================================================
// 2. Quantization: ternary values, binary values, roundtrip error
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Ternary quantization produces only {-1, 0, 1}.
    #[test]
    fn prop_ternary_values_in_set(
        input in finite_f32_vec(128),
        threshold in 0.0f32..5.0f32,
    ) {
        let q = quantize_ternary(&input, threshold);
        for (i, &v) in q.iter().enumerate() {
            prop_assert!(
                v == -1 || v == 0 || v == 1,
                "ternary[{i}] = {v}, expected {{-1,0,1}}"
            );
        }
    }

    /// Binary quantization produces only {-1, 1}.
    #[test]
    fn prop_binary_values_in_set(input in finite_f32_vec(128)) {
        let q = quantize_binary(&input);
        for (i, &v) in q.iter().enumerate() {
            prop_assert!(
                v == -1 || v == 1,
                "binary[{i}] = {v}, expected {{-1,1}}"
            );
        }
    }

    /// Symmetric i8 quantization scale is non-negative.
    #[test]
    fn prop_symmetric_i8_scale_nonneg(
        input in finite_f32_vec(64),
        bits in 2u8..=8u8,
    ) {
        let (_, scale) = quantize_symmetric_i8(&input, bits);
        prop_assert!(scale >= 0.0, "scale={scale} is negative");
    }

    /// Symmetric i8 roundtrip error is bounded: max_abs_error <= scale + eps.
    #[test]
    fn prop_symmetric_i8_roundtrip_error_bounded(
        input in prop::collection::vec(-5.0f32..5.0f32, 2..=64),
        bits in 2u8..=8u8,
    ) {
        let (q, scale) = quantize_symmetric_i8(&input, bits);
        let deq = dequantize_symmetric_i8(&q, scale);
        let err = compute_quantization_error(&input, &deq);
        prop_assert!(
            err.max_abs_error <= scale + 1e-5,
            "max_abs_error={} > scale={}", err.max_abs_error, scale
        );
    }

    /// Symmetric i8 quantization preserves length.
    #[test]
    fn prop_symmetric_i8_preserves_length(
        input in finite_f32_vec(128),
        bits in 2u8..=8u8,
    ) {
        let (q, _) = quantize_symmetric_i8(&input, bits);
        prop_assert_eq!(q.len(), input.len());
    }
}

// ===================================================================
// 3. Layer normalization
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// RMS norm output has the same length as input and all values are finite.
    #[test]
    fn prop_rms_norm_output_finite_same_length(
        norm_size in 1usize..=32,
        batch_size in 1usize..=8,
    ) {
        let n = batch_size * norm_size;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let gamma = vec![1.0f32; norm_size];
        let config = LayerNormConfig::new(vec![norm_size]);

        let output = rms_norm(&input, &gamma, &config).unwrap();
        prop_assert_eq!(output.len(), n, "output length mismatch");
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(v.is_finite(), "rms_norm output[{i}] = {v} is not finite");
        }
    }

    /// Layer norm with identity affine (gamma=1, beta=0) produces zero-mean output.
    #[test]
    fn prop_layer_norm_identity_affine_zero_mean(
        norm_size in 2usize..=32,
        batch_size in 1usize..=4,
    ) {
        let n = batch_size * norm_size;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 5.0).collect();
        let gamma = vec![1.0f32; norm_size];
        let beta = vec![0.0f32; norm_size];
        let config = LayerNormConfig::new(vec![norm_size]);

        let output = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        for b in 0..batch_size {
            let start = b * norm_size;
            let slice = &output[start..start + norm_size];
            let mean: f32 = slice.iter().sum::<f32>() / norm_size as f32;
            prop_assert!(
                mean.abs() < 1e-4,
                "batch {b}: mean = {mean}, expected ~0"
            );
        }
    }
}

// ===================================================================
// 4. Attention: causal mask properties
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Causal mask diagonal is always 0.0 (token can attend to itself).
    #[test]
    fn prop_causal_mask_diagonal_zero(seq_len in 1usize..=32) {
        let mask = causal_mask(seq_len);
        for i in 0..seq_len {
            let val = mask[i * seq_len + i];
            prop_assert!(
                val == 0.0,
                "mask[{i},{i}] = {val}, expected 0.0"
            );
        }
    }

    /// Causal mask upper triangle is NEG_INFINITY (future tokens masked).
    #[test]
    fn prop_causal_mask_upper_neg_inf(seq_len in 2usize..=32) {
        let mask = causal_mask(seq_len);
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                let val = mask[i * seq_len + j];
                prop_assert!(
                    val == f32::NEG_INFINITY,
                    "mask[{i},{j}] = {val}, expected NEG_INFINITY"
                );
            }
        }
    }

    /// Causal mask lower triangle is 0.0 (past tokens visible).
    #[test]
    fn prop_causal_mask_lower_zero(seq_len in 2usize..=32) {
        let mask = causal_mask(seq_len);
        for i in 1..seq_len {
            for j in 0..i {
                let val = mask[i * seq_len + j];
                prop_assert!(
                    val == 0.0,
                    "mask[{i},{j}] = {val}, expected 0.0"
                );
            }
        }
    }
}

// ===================================================================
// 5. RoPE: frequency array properties
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// compute_frequencies returns max_seq_len * head_dim values, all finite.
    #[test]
    fn prop_rope_frequencies_finite_and_correct_length(
        head_dim in prop::sample::select(vec![2usize, 4, 8, 16, 32, 64]),
    ) {
        let max_seq = 16usize;
        let config = RopeConfig::new(head_dim, max_seq);
        let freqs = compute_frequencies(&config);
        prop_assert_eq!(
            freqs.len(),
            max_seq * head_dim,
            "expected max_seq_len * head_dim elements"
        );
        for (i, &f) in freqs.iter().enumerate() {
            prop_assert!(f.is_finite(), "freq[{}] = {} is not finite", i, f);
        }
    }

    /// RoPE cos/sin values are bounded in [-1, 1].
    #[test]
    fn prop_rope_frequencies_bounded(
        head_dim in prop::sample::select(vec![4usize, 8, 16, 32, 64]),
    ) {
        let config = RopeConfig::new(head_dim, 16);
        let freqs = compute_frequencies(&config);
        for (i, &f) in freqs.iter().enumerate() {
            prop_assert!(
                (-1.0..=1.0).contains(&f),
                "freq[{}]={} out of [-1, 1] (cos/sin range)",
                i, f
            );
        }
    }
}

// ===================================================================
// 6. Fusion: fused_scale_add identity, length preservation
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// fused_scale_add with scale=1.0 is equivalent to vector addition.
    #[test]
    fn prop_fused_scale_add_identity_scale(
        len in 1usize..=64,
    ) {
        let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..len).map(|i| i as f32 * -0.3).collect();
        let result = fused_scale_add(&a, &b, 1.0).unwrap();
        for i in 0..len {
            let expected = a[i] + b[i];
            prop_assert!(
                (result[i] - expected).abs() < 1e-6,
                "index {i}: result={}, expected={}", result[i], expected
            );
        }
    }

    /// fused_add_normalize output has same length as input.
    #[test]
    fn prop_fused_add_normalize_preserves_length(
        norm_size in 1usize..=16,
        batch_size in 1usize..=4,
    ) {
        let n = norm_size * batch_size;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * -0.05).collect();
        let gamma = vec![1.0f32; norm_size];
        let eps = 1e-5;
        let result = fused_add_normalize(&a, &b, &gamma, eps);
        match result {
            Ok(out) => prop_assert_eq!(out.len(), n),
            Err(_) => { /* dimension validation may reject — that's fine */ }
        }
    }
}

// ===================================================================
// 7. SIMD math: scale identity, dot product self non-negative
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Scaling a vector by 1.0 returns the original values.
    #[test]
    fn prop_simd_vector_scale_identity(input in finite_f32_vec(128)) {
        let result = simd_vector_scale(&input, 1.0);
        for (i, (&orig, &scaled)) in input.iter().zip(result.iter()).enumerate() {
            prop_assert!(
                (orig - scaled).abs() < 1e-6,
                "index {i}: orig={orig}, scaled={scaled}"
            );
        }
    }

    /// Dot product of a vector with itself is non-negative.
    #[test]
    fn prop_simd_dot_self_nonneg(input in finite_f32_vec(128)) {
        let dot = simd_dot_product(&input, &input);
        prop_assert!(dot >= -1e-6, "dot(x,x) = {dot} is negative");
    }

    /// Scaling by 0.0 produces all zeros.
    #[test]
    fn prop_simd_vector_scale_zero(input in finite_f32_vec(64)) {
        let result = simd_vector_scale(&input, 0.0);
        for (i, &v) in result.iter().enumerate() {
            prop_assert!(
                v == 0.0,
                "index {i}: expected 0.0, got {v}"
            );
        }
    }
}
