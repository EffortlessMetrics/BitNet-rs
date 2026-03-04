#![allow(dead_code, unused_imports, unused_variables, unused_unsafe, unsafe_op_in_unsafe_fn)]
//! Property-based tests — wave 22.
//!
//! Kernel correctness invariants: softmax, layer norm, RoPE, attention,
//! quantize-dequantize round-trip, matrix multiply, residual add,
//! embedding lookup, concat-split round-trip, activation functions,
//! batch norm, and KV cache.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::activations::{
    elu, gelu, gelu_tanh, hard_sigmoid, hard_swish, leaky_relu, mish, relu, selu, sigmoid, silu,
    softplus, swish, tanh_act,
};
use bitnet_kernels::cpu::attention::{apply_mask, causal_mask, scaled_dot_product_attention};
use bitnet_kernels::cpu::batch_norm::{BatchNormConfig, batch_norm_forward};
use bitnet_kernels::cpu::concat::ConcatKernel;
use bitnet_kernels::cpu::embedding::embedding_lookup;
use bitnet_kernels::cpu::kv_cache::{
    KvCache, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear, kv_cache_slice,
};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::quantize::{
    dequantize_asymmetric_u8, dequantize_symmetric_i8, quantize_asymmetric_u8, quantize_binary,
    quantize_symmetric_i8, quantize_ternary,
};
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled};
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, compute_frequencies};
use bitnet_kernels::cpu::simd_matmul::{SimdMatmulConfig, simd_matmul_f32};
use bitnet_kernels::cuda::softmax::{SoftmaxConfig, softmax_cpu};
use proptest::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Generate a Vec<f32> with values in a reasonable range.
fn bounded_vec(len: usize, lo: f32, hi: f32) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(lo..hi, len)
}

// ── 1. Softmax invariants ───────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Softmax output sums to approximately 1.0 per row.
    #[test]
    fn softmax_sums_to_one(input in bounded_vec(16, -10.0f32, 10.0)) {
        let config = SoftmaxConfig::for_shape(input.len(), 1).unwrap();
        let mut output = vec![0.0f32; input.len()];
        softmax_cpu(&input, &mut output, &config).unwrap();
        let sum: f32 = output.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-4, "sum = {sum}");
    }

    /// Softmax output values are all in [0, 1].
    #[test]
    fn softmax_values_in_unit_interval(input in bounded_vec(16, -10.0f32, 10.0)) {
        let config = SoftmaxConfig::for_shape(input.len(), 1).unwrap();
        let mut output = vec![0.0f32; input.len()];
        softmax_cpu(&input, &mut output, &config).unwrap();
        for (i, &v) in output.iter().enumerate() {
            prop_assert!((0.0..=1.0).contains(&v), "output[{i}] = {v}");
        }
    }

    /// Softmax argmax matches input argmax.
    #[test]
    fn softmax_preserves_argmax(input in bounded_vec(16, -10.0f32, 10.0)) {
        let config = SoftmaxConfig::for_shape(input.len(), 1).unwrap();
        let mut output = vec![0.0f32; input.len()];
        softmax_cpu(&input, &mut output, &config).unwrap();
        let in_argmax = input.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let out_argmax = output.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        prop_assert_eq!(in_argmax, out_argmax);
    }

    /// Softmax of uniform input gives equal probabilities.
    #[test]
    fn softmax_uniform_gives_equal(val in -10.0f32..10.0, len in 2usize..32) {
        let input = vec![val; len];
        let config = SoftmaxConfig::for_shape(len, 1).unwrap();
        let mut output = vec![0.0f32; len];
        softmax_cpu(&input, &mut output, &config).unwrap();
        let expected = 1.0 / len as f32;
        for (i, &v) in output.iter().enumerate() {
            prop_assert!((v - expected).abs() < 1e-4, "output[{i}] = {v}, expected {expected}");
        }
    }

    /// Softmax output is non-negative.
    #[test]
    fn softmax_non_negative(input in bounded_vec(32, -50.0f32, 50.0)) {
        let config = SoftmaxConfig::for_shape(input.len(), 1).unwrap();
        let mut output = vec![0.0f32; input.len()];
        softmax_cpu(&input, &mut output, &config).unwrap();
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(v >= 0.0, "output[{i}] = {v}");
        }
    }
}

// ── 2. LayerNorm invariants ─────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// LayerNorm output mean is approximately zero (with gamma=1, beta=0).
    #[test]
    fn layer_norm_zero_mean(input in bounded_vec(32, -10.0f32, 10.0)) {
        let config = LayerNormConfig::new(vec![32]);
        let gamma = vec![1.0f32; 32];
        let beta = vec![0.0f32; 32];
        let output = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let mean: f32 = output.iter().sum::<f32>() / 32.0;
        prop_assert!(mean.abs() < 1e-3, "mean = {mean}");
    }

    /// LayerNorm output variance is approximately 1 (with gamma=1, beta=0).
    #[test]
    fn layer_norm_unit_variance(input in bounded_vec(32, -10.0f32, 10.0)) {
        let config = LayerNormConfig::new(vec![32]);
        let gamma = vec![1.0f32; 32];
        let beta = vec![0.0f32; 32];
        let output = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let mean: f32 = output.iter().sum::<f32>() / 32.0;
        let var: f32 = output.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 32.0;
        prop_assert!((var - 1.0).abs() < 0.1, "variance = {var}");
    }

    /// LayerNorm with beta shifts the mean.
    #[test]
    fn layer_norm_beta_shifts_mean(input in bounded_vec(32, -10.0f32, 10.0), beta_val in -5.0f32..5.0) {
        let config = LayerNormConfig::new(vec![32]);
        let gamma = vec![1.0f32; 32];
        let beta = vec![beta_val; 32];
        let output = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let mean: f32 = output.iter().sum::<f32>() / 32.0;
        prop_assert!((mean - beta_val).abs() < 0.15,
            "mean = {mean}, expected ≈ {beta_val}");
    }

    /// LayerNorm preserves shape (length).
    #[test]
    fn layer_norm_preserves_length(n in 1usize..8) {
        let len = n * 16;
        let input = vec![1.0f32; len];
        let config = LayerNormConfig::new(vec![16]);
        let gamma = vec![1.0f32; 16];
        let beta = vec![0.0f32; 16];
        let output = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        prop_assert_eq!(output.len(), len);
    }

    /// RMSNorm output has finite values.
    #[test]
    fn rms_norm_finite_output(input in bounded_vec(16, -10.0f32, 10.0)) {
        let config = LayerNormConfig::new(vec![16]);
        let gamma = vec![1.0f32; 16];
        let output = rms_norm(&input, &gamma, &config).unwrap();
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(v.is_finite(), "rms_norm output[{i}] = {v} is not finite");
        }
    }
}

// ── 3. RoPE invariants ──────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// RoPE rotation preserves vector magnitude.
    #[test]
    fn rope_preserves_magnitude(input in bounded_vec(8, -10.0f32, 10.0), pos in 0usize..16) {
        let head_dim = 8;
        let config = RopeConfig::new(head_dim, 32);
        let freqs = compute_frequencies(&config);
        let mag_before: f32 = input.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mut data = input.clone();
        apply_rope(&mut data, pos, head_dim, &freqs);
        let mag_after: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();
        prop_assert!((mag_before - mag_after).abs() < 1e-3,
            "before = {mag_before}, after = {mag_after}");
    }

    /// RoPE rotation by position 0 with scaling_factor=1 is near identity.
    #[test]
    fn rope_position_zero_near_identity(input in bounded_vec(8, -10.0f32, 10.0)) {
        let head_dim = 8;
        let config = RopeConfig::new(head_dim, 32);
        let freqs = compute_frequencies(&config);
        let mut data = input.clone();
        apply_rope(&mut data, 0, head_dim, &freqs);
        // At position 0, angle = 0 for all freq dims so cos=1, sin=0
        for (i, (&orig, &rotated)) in input.iter().zip(data.iter()).enumerate() {
            prop_assert!((orig - rotated).abs() < 1e-4,
                "pos=0 mismatch at [{i}]: orig={orig}, rotated={rotated}");
        }
    }

    /// Two successive RoPE rotations at position p and q differ from
    /// a single rotation at position p+q (composition property).
    #[test]
    fn rope_composition(
        input in bounded_vec(4, -5.0f32, 5.0),
        p in 0usize..8,
        q in 0usize..8,
    ) {
        let head_dim = 4;
        let config = RopeConfig::new(head_dim, 32);
        let freqs = compute_frequencies(&config);

        // Single rotation at p + q
        let mut single = input.clone();
        apply_rope(&mut single, p + q, head_dim, &freqs);

        // Two successive rotations: first p, then q
        let mut double = input.clone();
        apply_rope(&mut double, p, head_dim, &freqs);
        apply_rope(&mut double, q, head_dim, &freqs);

        // These should match (rotation composition)
        for (i, (&s, &d)) in single.iter().zip(double.iter()).enumerate() {
            prop_assert!((s - d).abs() < 1e-3,
                "composition mismatch at [{i}]: single={s}, double={d}");
        }
    }

    /// RoPE frequencies have the expected length.
    #[test]
    fn rope_frequency_length(head_dim in (1usize..=8).prop_map(|x| x * 2), max_seq in 1usize..32) {
        let config = RopeConfig::new(head_dim, max_seq);
        let freqs = compute_frequencies(&config);
        prop_assert_eq!(freqs.len(), max_seq * head_dim);
    }

    /// RoPE output has finite values.
    #[test]
    fn rope_finite_output(input in bounded_vec(8, -10.0f32, 10.0), pos in 0usize..16) {
        let head_dim = 8;
        let config = RopeConfig::new(head_dim, 32);
        let freqs = compute_frequencies(&config);
        let mut data = input.clone();
        apply_rope(&mut data, pos, head_dim, &freqs);
        for (i, &v) in data.iter().enumerate() {
            prop_assert!(v.is_finite(), "rope output[{i}] = {v} is not finite");
        }
    }
}

// ── 4. Attention score invariants ───────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Causal mask makes future positions -inf.
    #[test]
    fn causal_mask_future_is_neg_inf(seq_len in 2usize..16) {
        let mask = causal_mask(seq_len);
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                let val = mask[i * seq_len + j];
                prop_assert!(val == f32::NEG_INFINITY,
                    "mask[{},{j}] should be -inf, got {val}", i);
            }
        }
    }

    /// Causal mask diagonal and lower-triangle are zero.
    #[test]
    fn causal_mask_lower_is_zero(seq_len in 2usize..16) {
        let mask = causal_mask(seq_len);
        for i in 0..seq_len {
            for j in 0..=i {
                let val = mask[i * seq_len + j];
                prop_assert!(val == 0.0,
                    "mask[{},{j}] should be 0.0, got {val}", i);
            }
        }
    }

    /// Causal mask has correct size.
    #[test]
    fn causal_mask_size(seq_len in 1usize..32) {
        let mask = causal_mask(seq_len);
        prop_assert_eq!(mask.len(), seq_len * seq_len);
    }

    /// apply_mask does not change elements where mask is zero.
    #[test]
    fn apply_mask_identity_on_zeros(scores in bounded_vec(16, -10.0f32, 10.0)) {
        let mask = vec![0.0f32; scores.len()];
        let mut output = scores.clone();
        apply_mask(&mut output, &mask).unwrap();
        for (i, (&orig, &masked)) in scores.iter().zip(output.iter()).enumerate() {
            prop_assert!((orig - masked).abs() < f32::EPSILON,
                "scores[{}] changed with zero mask", i);
        }
    }

    /// Scaled dot-product attention output has correct shape.
    #[test]
    fn sdpa_output_shape(seq_len in 1usize..8, head_dim in (1usize..=4).prop_map(|x| x * 2)) {
        let n = seq_len * head_dim;
        let q = vec![0.1f32; n];
        let k = vec![0.1f32; n];
        let v = vec![0.1f32; n];
        let output = scaled_dot_product_attention(&q, &k, &v, seq_len, seq_len, head_dim, false).unwrap();
        prop_assert_eq!(output.len(), seq_len * head_dim);
    }
}

// ── 5. Quantize-dequantize round-trip ───────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Symmetric i8 round-trip error is bounded.
    #[test]
    fn symmetric_i8_round_trip_bounded(input in bounded_vec(16, -10.0f32, 10.0)) {
        let (quantized, scale) = quantize_symmetric_i8(&input, 8);
        let reconstructed = dequantize_symmetric_i8(&quantized, scale);
        for (i, (&orig, &recon)) in input.iter().zip(reconstructed.iter()).enumerate() {
            let err = (orig - recon).abs();
            // 8-bit symmetric: max error ~ abs_max / 127
            let max_expected_err = if scale > 0.0 { scale * 1.5 } else { 0.0 };
            prop_assert!(err <= max_expected_err,
                "i={i}: |{orig} - {recon}| = {err} > {max_expected_err}");
        }
    }

    /// Asymmetric u8 round-trip error is bounded.
    #[test]
    fn asymmetric_u8_round_trip_bounded(input in bounded_vec(16, -10.0f32, 10.0)) {
        let (quantized, scale, zero_point) = quantize_asymmetric_u8(&input);
        let reconstructed = dequantize_asymmetric_u8(&quantized, scale, zero_point);
        for (i, (&orig, &recon)) in input.iter().zip(reconstructed.iter()).enumerate() {
            let err = (orig - recon).abs();
            let max_expected_err = if scale > 0.0 { scale * 1.5 } else { 1e-6 };
            prop_assert!(err <= max_expected_err,
                "i={i}: |{orig} - {recon}| = {err} > {max_expected_err}");
        }
    }

    /// Symmetric quantization of all-zeros produces all-zeros.
    #[test]
    fn symmetric_i8_zeros(len in 1usize..64) {
        let input = vec![0.0f32; len];
        let (quantized, scale) = quantize_symmetric_i8(&input, 8);
        prop_assert_eq!(scale, 0.0);
        for &v in &quantized {
            prop_assert_eq!(v, 0);
        }
    }

    /// Ternary quantization outputs are in {-1, 0, 1}.
    #[test]
    fn ternary_values_in_range(input in bounded_vec(32, -10.0f32, 10.0), thresh in 0.0f32..5.0) {
        let quantized = quantize_ternary(&input, thresh);
        for (i, &v) in quantized.iter().enumerate() {
            prop_assert!(v == -1 || v == 0 || v == 1,
                "ternary[{i}] = {v} not in {{-1,0,1}}");
        }
    }

    /// Binary quantization outputs are in {-1, 1}.
    #[test]
    fn binary_values_in_range(input in bounded_vec(32, -10.0f32, 10.0)) {
        let quantized = quantize_binary(&input);
        for (i, &v) in quantized.iter().enumerate() {
            prop_assert!(v == -1 || v == 1,
                "binary[{i}] = {v} not in {{-1,1}}");
        }
    }

    /// Symmetric quantization preserves sign.
    #[test]
    fn symmetric_i8_preserves_sign(input in bounded_vec(16, -10.0f32, 10.0)) {
        let (quantized, _scale) = quantize_symmetric_i8(&input, 8);
        for (i, (&orig, &quant)) in input.iter().zip(quantized.iter()).enumerate() {
            if orig > 0.0 {
                prop_assert!(quant >= 0, "positive input[{i}]={orig} mapped to negative quant={quant}");
            } else if orig < 0.0 {
                prop_assert!(quant <= 0, "negative input[{i}]={orig} mapped to positive quant={quant}");
            }
        }
    }
}

// ── 6. Matrix multiply ─────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// (A*B)^T = B^T * A^T.
    #[test]
    fn matmul_transpose_property(
        a_vals in bounded_vec(6, -5.0f32, 5.0),  // 2x3
        b_vals in bounded_vec(12, -5.0f32, 5.0),  // 3x4
    ) {
        let (m, k, n) = (2, 3, 4);

        // C = A * B
        let cfg = SimdMatmulConfig::new(m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32(&a_vals, &b_vals, &mut c, &cfg).unwrap();

        // Transpose A (2x3 → 3x2) and B (3x4 → 4x3)
        let mut a_t = vec![0.0f32; k * m];
        for i in 0..m {
            for j in 0..k {
                a_t[j * m + i] = a_vals[i * k + j];
            }
        }
        let mut b_t = vec![0.0f32; n * k];
        for i in 0..k {
            for j in 0..n {
                b_t[j * k + i] = b_vals[i * n + j];
            }
        }

        // D = B^T * A^T (4x3 * 3x2 = 4x2)
        let cfg_t = SimdMatmulConfig::new(n, m, k);
        let mut d = vec![0.0f32; n * m];
        simd_matmul_f32(&b_t, &a_t, &mut d, &cfg_t).unwrap();

        // C^T should equal D
        for i in 0..m {
            for j in 0..n {
                let c_t_ij = c[i * n + j];
                let d_ji = d[j * m + i];
                prop_assert!((c_t_ij - d_ji).abs() < 1e-3,
                    "C^T[{j},{i}] = {c_t_ij}, D[{j},{i}] = {d_ji}");
            }
        }
    }

    /// A * I = A (identity matrix).
    #[test]
    fn matmul_identity(a_vals in bounded_vec(9, -5.0f32, 5.0)) {
        let n = 3;
        let mut identity = vec![0.0f32; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        let cfg = SimdMatmulConfig::new(n, n, n);
        let mut c = vec![0.0f32; n * n];
        simd_matmul_f32(&a_vals, &identity, &mut c, &cfg).unwrap();
        for (i, (&a, &c_val)) in a_vals.iter().zip(c.iter()).enumerate() {
            prop_assert!((a - c_val).abs() < 1e-4,
                "A*I != A at [{i}]: {a} vs {c_val}");
        }
    }

    /// A * 0 = 0 (zero matrix).
    #[test]
    fn matmul_zero(a_vals in bounded_vec(6, -5.0f32, 5.0)) {
        let (m, k, n) = (2, 3, 2);
        let zero_b = vec![0.0f32; k * n];
        let cfg = SimdMatmulConfig::new(m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32(&a_vals, &zero_b, &mut c, &cfg).unwrap();
        for (i, &v) in c.iter().enumerate() {
            prop_assert!(v.abs() < 1e-6, "A*0 != 0 at [{i}]: {v}");
        }
    }

    /// Matmul output is finite for finite inputs.
    #[test]
    fn matmul_finite_output(
        a_vals in bounded_vec(4, -5.0f32, 5.0),
        b_vals in bounded_vec(4, -5.0f32, 5.0),
    ) {
        let cfg = SimdMatmulConfig::new(2, 2, 2);
        let mut c = vec![0.0f32; 4];
        simd_matmul_f32(&a_vals, &b_vals, &mut c, &cfg).unwrap();
        for (i, &v) in c.iter().enumerate() {
            prop_assert!(v.is_finite(), "matmul output[{i}] = {v} is not finite");
        }
    }

    /// Matmul output length matches m*n.
    #[test]
    fn matmul_output_shape(
        m in 1usize..5,
        k in 1usize..5,
        n in 1usize..5,
    ) {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let cfg = SimdMatmulConfig::new(m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
        prop_assert_eq!(c.len(), m * n);
    }
}

// ── 7. Residual add ────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Residual add is commutative: a + b = b + a.
    #[test]
    fn residual_add_commutative(
        a in bounded_vec(16, -10.0f32, 10.0),
        b in bounded_vec(16, -10.0f32, 10.0),
    ) {
        let mut out_ab = a.clone();
        add_residual(&mut out_ab, &b).unwrap();
        let mut out_ba = b.clone();
        add_residual(&mut out_ba, &a).unwrap();
        for (i, (&ab, &ba)) in out_ab.iter().zip(out_ba.iter()).enumerate() {
            prop_assert!((ab - ba).abs() < 1e-5,
                "a+b != b+a at [{i}]: {ab} vs {ba}");
        }
    }

    /// Zero residual is identity: x + 0 = x.
    #[test]
    fn residual_add_identity(input in bounded_vec(16, -10.0f32, 10.0)) {
        let zeros = vec![0.0f32; 16];
        let mut output = input.clone();
        add_residual(&mut output, &zeros).unwrap();
        for (i, (&orig, &out)) in input.iter().zip(output.iter()).enumerate() {
            prop_assert!((orig - out).abs() < f32::EPSILON,
                "x+0 != x at [{}]", i);
        }
    }

    /// Scaled residual with scale=1 is same as regular add.
    #[test]
    fn residual_scaled_one(
        a in bounded_vec(16, -10.0f32, 10.0),
        b in bounded_vec(16, -10.0f32, 10.0),
    ) {
        let mut out_add = a.clone();
        add_residual(&mut out_add, &b).unwrap();
        let mut out_scaled = a.clone();
        add_residual_scaled(&mut out_scaled, &b, 1.0).unwrap();
        for (i, (&add_v, &scaled_v)) in out_add.iter().zip(out_scaled.iter()).enumerate() {
            prop_assert!((add_v - scaled_v).abs() < 1e-5,
                "add vs scaled(1.0) at [{i}]: {add_v} vs {scaled_v}");
        }
    }

    /// Scaled residual with scale=0 is identity.
    #[test]
    fn residual_scaled_zero(
        input in bounded_vec(16, -10.0f32, 10.0),
        residual in bounded_vec(16, -10.0f32, 10.0),
    ) {
        let mut output = input.clone();
        add_residual_scaled(&mut output, &residual, 0.0).unwrap();
        for (i, (&orig, &out)) in input.iter().zip(output.iter()).enumerate() {
            prop_assert!((orig - out).abs() < f32::EPSILON,
                "x + 0*r != x at [{}]", i);
        }
    }

    /// Residual add produces finite outputs for finite inputs.
    #[test]
    fn residual_add_finite(
        a in bounded_vec(16, -10.0f32, 10.0),
        b in bounded_vec(16, -10.0f32, 10.0),
    ) {
        let mut output = a.clone();
        add_residual(&mut output, &b).unwrap();
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(v.is_finite(), "residual output[{i}] = {v} is not finite");
        }
    }
}

// ── 8. Embedding lookup ────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Valid indices always succeed.
    #[test]
    fn embedding_valid_indices_succeed(
        idx in proptest::collection::vec(0u32..10, 1..8),
    ) {
        let vocab_size = 10usize;
        let embed_dim = 4usize;
        let table = vec![1.0f32; vocab_size * embed_dim];
        let result = embedding_lookup(&table, &idx, embed_dim);
        prop_assert!(result.is_ok());
        prop_assert_eq!(result.unwrap().len(), idx.len() * embed_dim);
    }

    /// Out-of-range index always errors.
    #[test]
    fn embedding_out_of_range_errors(
        oob_idx in 10u32..100,
    ) {
        let vocab_size = 10usize;
        let embed_dim = 4usize;
        let table = vec![1.0f32; vocab_size * embed_dim];
        let result = embedding_lookup(&table, &[oob_idx], embed_dim);
        prop_assert!(result.is_err());
    }

    /// Embedding output shape matches indices * embed_dim.
    #[test]
    fn embedding_output_shape(
        n_idx in 1usize..8,
        embed_dim in 1usize..16,
    ) {
        let vocab_size = 32usize;
        let table = vec![0.5f32; vocab_size * embed_dim];
        let indices: Vec<u32> = (0..n_idx as u32).collect();
        let output = embedding_lookup(&table, &indices, embed_dim).unwrap();
        prop_assert_eq!(output.len(), n_idx * embed_dim);
    }

    /// Embedding lookup returns correct values from the table.
    #[test]
    fn embedding_lookup_correct_values(idx in 0u32..8) {
        let vocab_size = 8usize;
        let embed_dim = 4usize;
        // Each row has a distinct pattern: row i has values [i*4, i*4+1, ...]
        let table: Vec<f32> = (0..vocab_size * embed_dim)
            .map(|i| i as f32)
            .collect();
        let output = embedding_lookup(&table, &[idx], embed_dim).unwrap();
        let expected_start = (idx as usize) * embed_dim;
        for (i, &v) in output.iter().enumerate() {
            prop_assert_eq!(v, (expected_start + i) as f32);
        }
    }

    /// Empty indices produce empty output.
    #[test]
    fn embedding_empty_indices(_dummy in 0u8..1) {
        let table = vec![1.0f32; 40];
        let result = embedding_lookup(&table, &[], 4).unwrap();
        prop_assert!(result.is_empty());
    }
}

// ── 9. Concat-split round-trip ─────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// split(concat(a, b)) == (a, b) along axis 0 for 1D.
    #[test]
    fn concat_split_round_trip_1d(
        a in bounded_vec(8, -10.0f32, 10.0),
        b in bounded_vec(8, -10.0f32, 10.0),
    ) {
        let shape_a: &[usize] = &[8];
        let shape_b: &[usize] = &[8];
        let concat = ConcatKernel::concat(&[&a, &b], &[shape_a, shape_b], 0).unwrap();
        prop_assert_eq!(concat.len(), 16);
        let parts = ConcatKernel::split(&concat, &[16], 0, 2).unwrap();
        prop_assert_eq!(parts.len(), 2);
        prop_assert_eq!(&parts[0], &a);
        prop_assert_eq!(&parts[1], &b);
    }

    /// split(concat(a, b, c)) == (a, b, c) along axis 0.
    #[test]
    fn concat_split_round_trip_three(
        a in bounded_vec(4, -10.0f32, 10.0),
        b in bounded_vec(4, -10.0f32, 10.0),
        c in bounded_vec(4, -10.0f32, 10.0),
    ) {
        let shape: &[usize] = &[4];
        let concat = ConcatKernel::concat(&[&a, &b, &c], &[shape, shape, shape], 0).unwrap();
        let parts = ConcatKernel::split(&concat, &[12], 0, 3).unwrap();
        prop_assert_eq!(parts.len(), 3);
        prop_assert_eq!(&parts[0], &a);
        prop_assert_eq!(&parts[1], &b);
        prop_assert_eq!(&parts[2], &c);
    }

    /// Concat preserves total element count.
    #[test]
    fn concat_preserves_element_count(
        a in bounded_vec(6, -10.0f32, 10.0),
        b in bounded_vec(6, -10.0f32, 10.0),
    ) {
        let shape: &[usize] = &[6];
        let concat = ConcatKernel::concat(&[&a, &b], &[shape, shape], 0).unwrap();
        prop_assert_eq!(concat.len(), a.len() + b.len());
    }

    /// Split into 1 part returns the original data.
    #[test]
    fn split_into_one(data in bounded_vec(8, -10.0f32, 10.0)) {
        let parts = ConcatKernel::split(&data, &[8], 0, 1).unwrap();
        prop_assert_eq!(parts.len(), 1);
        prop_assert_eq!(&parts[0], &data);
    }

    /// Concat of a single tensor is identity.
    #[test]
    fn concat_single_identity(data in bounded_vec(8, -10.0f32, 10.0)) {
        let shape: &[usize] = &[8];
        let concat = ConcatKernel::concat(&[data.as_slice()], &[shape], 0).unwrap();
        prop_assert_eq!(&concat, &data);
    }
}

// ── 10. Activation functions ───────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// ReLU output is non-negative.
    #[test]
    fn relu_non_negative(x in -100.0f32..100.0) {
        prop_assert!(relu(x) >= 0.0);
    }

    /// ReLU is idempotent.
    #[test]
    fn relu_idempotent(x in -100.0f32..100.0) {
        prop_assert_eq!(relu(relu(x)), relu(x));
    }

    /// Sigmoid output in (0, 1).
    #[test]
    fn sigmoid_in_unit_interval(x in -10.0f32..10.0) {
        let s = sigmoid(x);
        prop_assert!(s > 0.0, "sigmoid({x}) = {s} not > 0");
        prop_assert!(s < 1.0, "sigmoid({x}) = {s} not < 1");
    }

    /// Tanh output in (-1, 1).
    #[test]
    fn tanh_in_range(x in -8.0f32..8.0) {
        let t = tanh_act(x);
        prop_assert!(t > -1.0 && t < 1.0, "tanh({x}) = {t}");
    }

    /// Hard sigmoid in [0, 1].
    #[test]
    fn hard_sigmoid_bounded(x in -100.0f32..100.0) {
        let hs = hard_sigmoid(x);
        prop_assert!((0.0..=1.0).contains(&hs), "hard_sigmoid({x}) = {hs}");
    }

    /// Hard swish is finite.
    #[test]
    fn hard_swish_finite(x in -100.0f32..100.0) {
        prop_assert!(hard_swish(x).is_finite());
    }

    /// GELU is finite for moderate inputs.
    #[test]
    fn gelu_finite(x in -20.0f32..20.0) {
        prop_assert!(gelu(x).is_finite());
    }

    /// GELU-tanh approximation is close to GELU.
    #[test]
    fn gelu_tanh_close_to_gelu(x in -5.0f32..5.0) {
        let diff = (gelu(x) - gelu_tanh(x)).abs();
        prop_assert!(diff < 0.02, "gelu({x})={}, gelu_tanh({x})={}, diff={diff}", gelu(x), gelu_tanh(x));
    }

    /// SiLU(0) ≈ 0.
    #[test]
    fn silu_zero(_dummy in 0u8..1) {
        prop_assert!(silu(0.0).abs() < 1e-6);
    }

    /// Swish equals SiLU (swish is silu with beta=1).
    #[test]
    fn swish_equals_silu(x in -10.0f32..10.0) {
        prop_assert!((swish(x, 1.0) - silu(x)).abs() < 1e-6);
    }

    /// ELU is negative for negative inputs (with alpha > 0).
    #[test]
    fn elu_negative_for_negative_input(x in -100.0f32..-0.01) {
        let e = elu(x, 1.0);
        prop_assert!(e < 0.0, "elu({x}) = {e} should be negative");
    }

    /// Softplus is non-negative for all inputs.
    #[test]
    fn softplus_positive(x in -50.0f32..50.0) {
        prop_assert!(softplus(x) >= 0.0, "softplus({x}) = {}", softplus(x));
    }

    /// Leaky ReLU is non-positive for negative inputs (with positive alpha).
    #[test]
    fn leaky_relu_negative_region(x in -100.0f32..-0.001) {
        let lr = leaky_relu(x, 0.01);
        prop_assert!(lr < 0.0, "leaky_relu({x}) = {lr} should be negative");
    }

    /// SELU is finite.
    #[test]
    fn selu_finite(x in -20.0f32..20.0) {
        prop_assert!(selu(x).is_finite());
    }

    /// Mish is finite for moderate inputs.
    #[test]
    fn mish_finite(x in -20.0f32..20.0) {
        prop_assert!(mish(x).is_finite());
    }

    /// Sigmoid is monotonically non-decreasing.
    #[test]
    fn sigmoid_monotone(a in -50.0f32..50.0, b in -50.0f32..50.0) {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        prop_assert!(sigmoid(lo) <= sigmoid(hi) + 1e-6);
    }
}

// ── 11. Batch norm ─────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Batch norm output has approximately zero mean per channel (gamma=1, beta=0).
    #[test]
    fn batch_norm_zero_mean(batch in 2usize..8) {
        let num_features = 4;
        let len = batch * num_features;
        // Generate non-constant data
        let input: Vec<f32> = (0..len).map(|i| (i as f32) * 0.5 - (len as f32 * 0.25)).collect();
        let gamma = vec![1.0f32; num_features];
        let beta = vec![0.0f32; num_features];
        let running_mean = vec![0.0f32; num_features];
        let running_var = vec![1.0f32; num_features];
        let config = BatchNormConfig { num_features, eps: 1e-5, momentum: 0.1, training: true };
        let (output, _, _) = batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &config).unwrap();

        for ch in 0..num_features {
            let mean: f32 = (0..batch).map(|n| output[n * num_features + ch]).sum::<f32>() / batch as f32;
            prop_assert!(mean.abs() < 0.1, "channel {ch} mean = {mean}");
        }
    }

    /// Batch norm output has approximately unit variance per channel (gamma=1, beta=0).
    #[test]
    fn batch_norm_unit_variance(batch in 4usize..12) {
        let num_features = 4;
        let len = batch * num_features;
        let input: Vec<f32> = (0..len).map(|i| (i as f32) * 0.3 - (len as f32 * 0.15)).collect();
        let gamma = vec![1.0f32; num_features];
        let beta = vec![0.0f32; num_features];
        let running_mean = vec![0.0f32; num_features];
        let running_var = vec![1.0f32; num_features];
        let config = BatchNormConfig { num_features, eps: 1e-5, momentum: 0.1, training: true };
        let (output, _, _) = batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &config).unwrap();

        for ch in 0..num_features {
            let vals: Vec<f32> = (0..batch).map(|n| output[n * num_features + ch]).collect();
            let mean: f32 = vals.iter().sum::<f32>() / batch as f32;
            let var: f32 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / batch as f32;
            prop_assert!((var - 1.0).abs() < 0.3, "channel {ch} variance = {var}");
        }
    }

    /// Batch norm preserves length.
    #[test]
    fn batch_norm_preserves_length(batch in 2usize..8) {
        let num_features = 4;
        let len = batch * num_features;
        let input = vec![1.0f32; len];
        let gamma = vec![1.0f32; num_features];
        let beta = vec![0.0f32; num_features];
        let running_mean = vec![0.0f32; num_features];
        let running_var = vec![1.0f32; num_features];
        let config = BatchNormConfig { num_features, eps: 1e-5, momentum: 0.1, training: true };
        let (output, _, _) = batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &config).unwrap();
        prop_assert_eq!(output.len(), len);
    }

    /// Batch norm output is finite.
    #[test]
    fn batch_norm_finite(batch in 2usize..8) {
        let num_features = 4;
        let len = batch * num_features;
        let input: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
        let gamma = vec![1.0f32; num_features];
        let beta = vec![0.0f32; num_features];
        let running_mean = vec![0.0f32; num_features];
        let running_var = vec![1.0f32; num_features];
        let config = BatchNormConfig { num_features, eps: 1e-5, momentum: 0.1, training: true };
        let (output, _, _) = batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &config).unwrap();
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(v.is_finite(), "batch_norm output[{i}] = {v}");
        }
    }

    /// Batch norm running stats are updated.
    #[test]
    fn batch_norm_updates_running_stats(batch in 2usize..8) {
        let num_features = 2;
        let len = batch * num_features;
        let input: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let gamma = vec![1.0f32; num_features];
        let beta = vec![0.0f32; num_features];
        let running_mean = vec![0.0f32; num_features];
        let running_var = vec![1.0f32; num_features];
        let config = BatchNormConfig { num_features, eps: 1e-5, momentum: 0.1, training: true };
        let (_, updated_mean, _updated_var) = batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &config).unwrap();
        // Running stats should have been updated (not all zeros)
        let all_zero_mean = updated_mean.iter().all(|&v| v == 0.0);
        prop_assert!(!all_zero_mean, "running mean was not updated");
    }
}

// ── 12. KV cache ───────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Append then read returns the appended values.
    #[test]
    fn kv_cache_append_read_round_trip(
        key_vals in bounded_vec(8, -10.0f32, 10.0),
    ) {
        let num_heads = 2;
        let head_dim = 4;
        let config = KvCacheConfig {
            num_layers: 1,
            num_heads,
            head_dim,
            max_seq_len: 16,
            dtype: KvDtype::F32,
        };
        let mut cache = KvCache::new(config).unwrap();
        let values = key_vals.clone();
        kv_cache_append(&mut cache, 0, &key_vals, &values).unwrap();

        let (keys_out, vals_out) = kv_cache_slice(&cache, 0, 0, 1).unwrap();
        prop_assert_eq!(keys_out, key_vals.as_slice());
        prop_assert_eq!(vals_out, values.as_slice());
    }

    /// Multiple appends accumulate tokens.
    #[test]
    fn kv_cache_multiple_appends(n_appends in 1usize..8) {
        let num_heads = 1;
        let head_dim = 4;
        let token_elems = num_heads * head_dim;
        let config = KvCacheConfig {
            num_layers: 1,
            num_heads,
            head_dim,
            max_seq_len: 16,
            dtype: KvDtype::F32,
        };
        let mut cache = KvCache::new(config).unwrap();
        for i in 0..n_appends {
            let kv = vec![i as f32; token_elems];
            kv_cache_append(&mut cache, 0, &kv, &kv).unwrap();
        }
        let seq_len = cache.seq_len(0).unwrap();
        prop_assert_eq!(seq_len, n_appends);
    }

    /// Clear resets sequence length to 0.
    #[test]
    fn kv_cache_clear_resets(n_appends in 1usize..4) {
        let num_heads = 1;
        let head_dim = 2;
        let token_elems = num_heads * head_dim;
        let config = KvCacheConfig {
            num_layers: 2,
            num_heads,
            head_dim,
            max_seq_len: 16,
            dtype: KvDtype::F32,
        };
        let mut cache = KvCache::new(config).unwrap();
        for _ in 0..n_appends {
            let kv = vec![1.0f32; token_elems];
            kv_cache_append(&mut cache, 0, &kv, &kv).unwrap();
            kv_cache_append(&mut cache, 1, &kv, &kv).unwrap();
        }
        kv_cache_clear(&mut cache);
        prop_assert_eq!(cache.seq_len(0).unwrap(), 0);
        prop_assert_eq!(cache.seq_len(1).unwrap(), 0);
    }

    /// Slicing out-of-bounds errors.
    #[test]
    fn kv_cache_slice_oob_errors(_dummy in 0u8..1) {
        let config = KvCacheConfig {
            num_layers: 1,
            num_heads: 1,
            head_dim: 2,
            max_seq_len: 4,
            dtype: KvDtype::F32,
        };
        let cache = KvCache::new(config).unwrap();
        let result = kv_cache_slice(&cache, 0, 0, 1);
        prop_assert!(result.is_err(), "slice beyond seq_len should fail");
    }

    /// Append beyond max_seq_len errors.
    #[test]
    fn kv_cache_overflow_errors(_dummy in 0u8..1) {
        let config = KvCacheConfig {
            num_layers: 1,
            num_heads: 1,
            head_dim: 2,
            max_seq_len: 2,
            dtype: KvDtype::F32,
        };
        let mut cache = KvCache::new(config).unwrap();
        let kv = vec![1.0f32; 2];
        kv_cache_append(&mut cache, 0, &kv, &kv).unwrap();
        kv_cache_append(&mut cache, 0, &kv, &kv).unwrap();
        let result = kv_cache_append(&mut cache, 0, &kv, &kv);
        prop_assert!(result.is_err(), "append beyond max_seq_len should fail");
    }
}
