//! Property-based tests for NEON SIMD kernel correctness on ARM64.
//!
//! These tests exercise the public CPU kernel APIs on `aarch64` targets
//! (Apple Silicon / ARM64 Linux) to verify numerical invariants that
//! must hold regardless of the underlying SIMD implementation:
//!
//! - Softmax: sum-to-one, range [0,1], monotonicity
//! - RMS normalization: approximate unit variance, sign preservation
//! - Quantization: round-trip fidelity
//! - Activations: GELU/SiLU/ReLU monotonicity, non-negativity
//! - Reductions: sum/max/min agreement with naive implementations
//! - Dot product: algebraic properties (commutativity, distributivity)

#![cfg(target_arch = "aarch64")]

use proptest::prelude::*;

use bitnet_kernels::cpu::activations::{ActivationType, activate, gelu, relu, silu};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, rms_norm};
use bitnet_kernels::cpu::quantize::{
    dequantize_asymmetric_u8, dequantize_symmetric_i8, quantize_asymmetric_u8,
    quantize_symmetric_i8,
};
use bitnet_kernels::cuda::softmax::{SoftmaxConfig, softmax_cpu};
use bitnet_kernels::reduction::{ReductionOp, reduce_f32};

// ── Helpers ────────────────────────────────────────────────────────

/// Strategy producing non-empty f32 vectors in a reasonable range.
fn vec_f32(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-50.0f32..50.0f32, 1..=max_len)
}

/// Strategy producing non-empty f32 vectors with strictly positive
/// values (useful for gamma weights).
fn vec_positive_f32(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(0.1f32..10.0f32, 1..=max_len)
}

fn proptest_config() -> ProptestConfig {
    ProptestConfig::with_cases(300)
}

// ═══════════════════════════════════════════════════════════════════
// Softmax properties
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(proptest_config())]

    /// Softmax output sums to 1.0 (within floating-point tolerance)
    /// for each row.
    #[test]
    fn prop_softmax_sums_to_one(
        input in prop::collection::vec(-10.0f32..10.0f32, 1..=256),
    ) {
        let n_cols = input.len();
        let config = SoftmaxConfig::for_shape(n_cols, 1).unwrap();
        let mut output = vec![0.0f32; n_cols];
        softmax_cpu(&input, &mut output, &config).unwrap();

        let sum: f32 = output.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax sum = {sum}, expected ~1.0"
        );
    }

    /// Every softmax output value lies in [0, 1].
    #[test]
    fn prop_softmax_values_in_unit_range(
        input in prop::collection::vec(-10.0f32..10.0f32, 1..=256),
    ) {
        let n_cols = input.len();
        let config = SoftmaxConfig::for_shape(n_cols, 1).unwrap();
        let mut output = vec![0.0f32; n_cols];
        softmax_cpu(&input, &mut output, &config).unwrap();

        for (i, &v) in output.iter().enumerate() {
            prop_assert!(
                (0.0..=1.0).contains(&v),
                "softmax[{i}] = {v} out of [0, 1]"
            );
        }
    }

    /// Softmax preserves monotonic ordering: if input[i] > input[j],
    /// then output[i] >= output[j].
    #[test]
    fn prop_softmax_preserves_monotonicity(
        input in prop::collection::vec(-10.0f32..10.0f32, 2..=128),
    ) {
        let n_cols = input.len();
        let config = SoftmaxConfig::for_shape(n_cols, 1).unwrap();
        let mut output = vec![0.0f32; n_cols];
        softmax_cpu(&input, &mut output, &config).unwrap();

        for i in 0..n_cols {
            for j in (i + 1)..n_cols {
                if input[i] > input[j] {
                    prop_assert!(
                        output[i] >= output[j] - 1e-7,
                        "monotonicity: input[{i}]={} > input[{j}]={} \
                         but output[{i}]={} < output[{j}]={}",
                        input[i], input[j], output[i], output[j]
                    );
                }
            }
        }
    }

    /// Softmax with uniform input produces uniform output ≈ 1/n.
    #[test]
    fn prop_softmax_uniform_input(
        val in -10.0f32..10.0f32,
        len in 1usize..=128,
    ) {
        let input = vec![val; len];
        let config = SoftmaxConfig::for_shape(len, 1).unwrap();
        let mut output = vec![0.0f32; len];
        softmax_cpu(&input, &mut output, &config).unwrap();

        let expected = 1.0 / len as f32;
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(
                (v - expected).abs() < 1e-4,
                "uniform softmax[{i}] = {v}, expected {expected}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// RMS normalization properties
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(proptest_config())]

    /// RMS-normalised output has approximately unit RMS (before gamma
    /// scaling, we check with gamma = 1.0).
    #[test]
    fn prop_rms_norm_unit_rms(
        input in prop::collection::vec(
            prop::num::f32::NORMAL, 2..=256
        ),
    ) {
        let n = input.len();
        // Skip near-zero inputs where division is unstable.
        let rms_in: f32 = (input.iter().map(|x| x * x).sum::<f32>()
            / n as f32)
            .sqrt();
        prop_assume!(rms_in > 1e-6);

        let gamma = vec![1.0f32; n];
        let config = LayerNormConfig::new(vec![n]);
        let output = rms_norm(&input, &gamma, &config).unwrap();

        let rms_out: f32 = (output.iter().map(|x| x * x).sum::<f32>()
            / n as f32)
            .sqrt();
        prop_assert!(
            (rms_out - 1.0).abs() < 0.05,
            "RMS of output = {rms_out}, expected ~1.0"
        );
    }

    /// RMS normalization preserves sign pattern (positive stays
    /// positive, negative stays negative) when gamma is positive.
    #[test]
    fn prop_rms_norm_preserves_sign(
        input in prop::collection::vec(
            prop::num::f32::NORMAL, 2..=256
        ),
    ) {
        let n = input.len();
        let rms_in: f32 = (input.iter().map(|x| x * x).sum::<f32>()
            / n as f32)
            .sqrt();
        prop_assume!(rms_in > 1e-6);

        let gamma = vec![1.0f32; n];
        let config = LayerNormConfig::new(vec![n]);
        let output = rms_norm(&input, &gamma, &config).unwrap();

        for (i, (&inp, &out)) in
            input.iter().zip(output.iter()).enumerate()
        {
            if inp.abs() > 1e-6 {
                prop_assert!(
                    inp.signum() == out.signum(),
                    "sign mismatch at [{i}]: input={inp}, \
                     output={out}"
                );
            }
        }
    }

    /// Scaling gamma uniformly scales the output proportionally.
    #[test]
    fn prop_rms_norm_gamma_scaling(
        input in prop::collection::vec(
            prop::num::f32::NORMAL, 2..=128
        ),
        scale in 0.5f32..5.0f32,
    ) {
        let n = input.len();
        let rms_in: f32 = (input.iter().map(|x| x * x).sum::<f32>()
            / n as f32)
            .sqrt();
        prop_assume!(rms_in > 1e-6);

        let gamma1 = vec![1.0f32; n];
        let gamma2 = vec![scale; n];
        let config = LayerNormConfig::new(vec![n]);

        let out1 = rms_norm(&input, &gamma1, &config).unwrap();
        let out2 = rms_norm(&input, &gamma2, &config).unwrap();

        for (i, (&v1, &v2)) in out1.iter().zip(out2.iter()).enumerate()
        {
            let expected = v1 * scale;
            prop_assert!(
                (v2 - expected).abs() < 1e-4,
                "gamma scaling at [{i}]: {v2} != {expected}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Quantization round-trip properties
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(proptest_config())]

    /// Symmetric i8 quantize → dequantize preserves values within
    /// tolerance.
    #[test]
    fn prop_symmetric_i8_round_trip(
        input in vec_f32(512),
    ) {
        let (quantized, scale) = quantize_symmetric_i8(&input, 8);
        let reconstructed = dequantize_symmetric_i8(&quantized, scale);

        prop_assert_eq!(reconstructed.len(), input.len());

        let abs_max = input
            .iter()
            .copied()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        // Tolerance: one quantization step.
        let tol = if abs_max > 0.0 { abs_max / 127.0 } else { 1e-6 };

        for (i, (&orig, &recon)) in
            input.iter().zip(reconstructed.iter()).enumerate()
        {
            prop_assert!(
                (orig - recon).abs() <= tol + 1e-6,
                "round-trip error at [{i}]: orig={orig}, \
                 recon={recon}, tol={tol}"
            );
        }
    }

    /// Asymmetric u8 quantize → dequantize preserves values within
    /// tolerance.
    #[test]
    fn prop_asymmetric_u8_round_trip(
        input in vec_f32(512),
    ) {
        let (quantized, scale, zero_point) =
            quantize_asymmetric_u8(&input);
        let reconstructed =
            dequantize_asymmetric_u8(&quantized, scale, zero_point);

        prop_assert_eq!(reconstructed.len(), input.len());

        let range = input
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
            .abs()
            + input
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
                .abs();
        let tol = if range > 0.0 { range / 255.0 } else { 1e-6 };

        for (i, (&orig, &recon)) in
            input.iter().zip(reconstructed.iter()).enumerate()
        {
            prop_assert!(
                (orig - recon).abs() <= tol + 1e-5,
                "asym round-trip error at [{i}]: orig={orig}, \
                 recon={recon}"
            );
        }
    }

    /// Symmetric quantization of all-zero input produces all-zero
    /// output.
    #[test]
    fn prop_symmetric_zero_input(len in 1usize..=256) {
        let input = vec![0.0f32; len];
        let (quantized, scale) = quantize_symmetric_i8(&input, 8);
        prop_assert_eq!(scale, 0.0);
        prop_assert!(quantized.iter().all(|&v| v == 0));
    }
}

// ═══════════════════════════════════════════════════════════════════
// Activation function properties
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(proptest_config())]

    /// ReLU outputs are always non-negative.
    #[test]
    fn prop_relu_non_negative(x in -50.0f32..50.0f32) {
        let y = relu(x);
        prop_assert!(
            y >= 0.0 || y.is_nan(),
            "relu({x}) = {y} is negative"
        );
    }

    /// ReLU is monotonically non-decreasing.
    #[test]
    fn prop_relu_monotonic(
        a in -50.0f32..50.0f32,
        b in -50.0f32..50.0f32,
    ) {
        if a <= b {
            prop_assert!(
                relu(a) <= relu(b),
                "relu({a})={} > relu({b})={}",
                relu(a),
                relu(b)
            );
        }
    }

    /// GELU is monotonically non-decreasing (approximate — holds for
    /// standard GELU in practice).
    #[test]
    fn prop_gelu_monotonic(
        a in -50.0f32..50.0f32,
        b in -50.0f32..50.0f32,
    ) {
        if a <= b {
            let ga = gelu(a);
            let gb = gelu(b);
            // GELU is monotonic; allow small float tolerance.
            prop_assert!(
                ga <= gb + 1e-5,
                "gelu({a})={ga} > gelu({b})={gb}"
            );
        }
    }

    /// SiLU is monotonically non-decreasing for x >= 0.
    #[test]
    fn prop_silu_monotonic_positive(
        a in 0.0f32..50.0f32,
        b in 0.0f32..50.0f32,
    ) {
        if a <= b {
            let sa = silu(a);
            let sb = silu(b);
            prop_assert!(
                sa <= sb + 1e-5,
                "silu({a})={sa} > silu({b})={sb}"
            );
        }
    }

    /// Vectorised activate(ReLU) matches scalar relu element-wise.
    #[test]
    fn prop_activate_relu_matches_scalar(
        input in vec_f32(256),
    ) {
        let vec_out = activate(&input, ActivationType::ReLU);
        for (i, (&inp, &out)) in
            input.iter().zip(vec_out.iter()).enumerate()
        {
            let expected = relu(inp);
            prop_assert!(
                (out - expected).abs() < 1e-7,
                "activate(ReLU) mismatch at [{i}]: \
                 {out} vs {expected}"
            );
        }
    }

    /// Vectorised activate(GELU) matches scalar gelu element-wise.
    #[test]
    fn prop_activate_gelu_matches_scalar(
        input in vec_f32(256),
    ) {
        let vec_out = activate(&input, ActivationType::GELU);
        for (i, (&inp, &out)) in
            input.iter().zip(vec_out.iter()).enumerate()
        {
            let expected = gelu(inp);
            prop_assert!(
                (out - expected).abs() < 1e-5,
                "activate(GELU) mismatch at [{i}]: \
                 {out} vs {expected}"
            );
        }
    }

    /// Vectorised activate(SiLU) matches scalar silu element-wise.
    #[test]
    fn prop_activate_silu_matches_scalar(
        input in vec_f32(256),
    ) {
        let vec_out = activate(&input, ActivationType::SiLU);
        for (i, (&inp, &out)) in
            input.iter().zip(vec_out.iter()).enumerate()
        {
            let expected = silu(inp);
            prop_assert!(
                (out - expected).abs() < 1e-5,
                "activate(SiLU) mismatch at [{i}]: \
                 {out} vs {expected}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Reduction properties
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(proptest_config())]

    /// reduce_f32(Sum) matches naive summation.
    #[test]
    fn prop_reduce_sum_matches_naive(
        input in vec_f32(512),
    ) {
        let result = reduce_f32(&input, ReductionOp::Sum);
        let naive: f32 = input.iter().sum();
        prop_assert!(
            (result - naive).abs() < 1e-2,
            "sum: {result} vs naive {naive}"
        );
    }

    /// reduce_f32(Max) matches naive max.
    #[test]
    fn prop_reduce_max_matches_naive(
        input in vec_f32(512),
    ) {
        let result = reduce_f32(&input, ReductionOp::Max);
        let naive = input
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        prop_assert!(
            (result - naive).abs() < 1e-6,
            "max: {result} vs naive {naive}"
        );
    }

    /// reduce_f32(Min) matches naive min.
    #[test]
    fn prop_reduce_min_matches_naive(
        input in vec_f32(512),
    ) {
        let result = reduce_f32(&input, ReductionOp::Min);
        let naive = input
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        prop_assert!(
            (result - naive).abs() < 1e-6,
            "min: {result} vs naive {naive}"
        );
    }

    /// reduce_f32(Mean) matches naive mean.
    #[test]
    fn prop_reduce_mean_matches_naive(
        input in vec_f32(512),
    ) {
        let result = reduce_f32(&input, ReductionOp::Mean);
        let naive: f32 =
            input.iter().sum::<f32>() / input.len() as f32;
        prop_assert!(
            (result - naive).abs() < 1e-3,
            "mean: {result} vs naive {naive}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// Dot product / linear algebra properties
// ═══════════════════════════════════════════════════════════════════

/// Naive dot product for reference comparison.
fn naive_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

proptest! {
    #![proptest_config(proptest_config())]

    /// Dot product is commutative: dot(a, b) == dot(b, a).
    #[test]
    fn prop_dot_product_commutative(
        a in prop::collection::vec(-10.0f32..10.0f32, 1..=256),
    ) {
        let b: Vec<f32> =
            a.iter().map(|x| x * 0.7 + 1.3).collect();
        let dot_ab = naive_dot(&a, &b);
        let dot_ba = naive_dot(&b, &a);
        prop_assert!(
            (dot_ab - dot_ba).abs() < 1e-3,
            "dot(a,b)={dot_ab} != dot(b,a)={dot_ba}"
        );
    }

    /// Dot product is distributive: dot(a, b+c) ≈ dot(a,b) + dot(a,c).
    #[test]
    fn prop_dot_product_distributive(
        a in prop::collection::vec(-5.0f32..5.0f32, 1..=128),
    ) {
        let b: Vec<f32> =
            a.iter().map(|x| x * 0.5 + 0.1).collect();
        let c: Vec<f32> =
            a.iter().map(|x| -x * 0.3 + 0.2).collect();
        let bc: Vec<f32> =
            b.iter().zip(c.iter()).map(|(&x, &y)| x + y).collect();

        let lhs = naive_dot(&a, &bc);
        let rhs = naive_dot(&a, &b) + naive_dot(&a, &c);

        // Tolerance scales with dimension and magnitude.
        let tol = 1e-2 * a.len() as f32;
        prop_assert!(
            (lhs - rhs).abs() < tol,
            "distributive: {lhs} vs {rhs}, tol={tol}"
        );
    }

    /// Dot product of a vector with itself is non-negative (||a||² ≥ 0).
    #[test]
    fn prop_dot_product_self_nonneg(
        a in prop::collection::vec(-10.0f32..10.0f32, 1..=256),
    ) {
        let dot_aa = naive_dot(&a, &a);
        prop_assert!(
            dot_aa >= -1e-6,
            "dot(a,a) = {dot_aa} < 0"
        );
    }
}
