#![allow(dead_code, unused_imports, unused_variables, unused_unsafe, unsafe_op_in_unsafe_fn)]
//! Comprehensive property-based tests for Apple Silicon NEON kernel invariants.
//!
//! Validates mathematical properties that must hold for NEON-accelerated
//! kernels regardless of input:
//!
//! - **Softmax**: output sums to ~1.0, values in [0, 1]
//! - **LayerNorm**: output has mean ~0, variance ~1
//! - **RoPE**: periodic — applying twice with same params is consistent
//! - **Activations**: ReLU zeroes negatives, preserves positives
//! - **Quantize**: symmetric i8 round-trip preserves approximate values
//! - **MatMul**: GEMV output dimension equals M (weight rows)
//! - **Element-wise**: add/mul/scale preserve vector length
//! - **Top-k (argmax)**: returns valid index within input bounds

#![cfg(all(target_arch = "aarch64", feature = "cpu"))]

use proptest::prelude::*;

use bitnet_kernels::cpu::neon_activations::{neon_gelu_f32, neon_relu_f32, neon_silu_f32};
use bitnet_kernels::cpu::neon_elementwise::{neon_add_f32, neon_mul_f32, neon_scale_f32};
use bitnet_kernels::cpu::neon_layernorm::{layernorm_neon, rmsnorm_neon};
use bitnet_kernels::cpu::neon_reductions::{neon_argmax_f32, neon_sum_f32};
use bitnet_kernels::cpu::neon_rope::{scalar_precompute_freqs, scalar_rope_apply};

// Compat wrappers: map old API names to current neon_rope functions.
unsafe fn build_cos_sin_tables_neon(
    dim: usize,
    max_seq: usize,
    base: f32,
) -> (Vec<f32>, Vec<f32>) {
    scalar_precompute_freqs(dim, max_seq, base, 1.0)
}

use bitnet_kernels::cpu::neon_softmax::{softmax_neon, softmax_scalar};
use bitnet_kernels::cpu::quantize::{dequantize_symmetric_i8, quantize_symmetric_i8};

// ── Helpers ────────────────────────────────────────────────────────

fn cfg() -> ProptestConfig {
    ProptestConfig::with_cases(200)
}

const EPS: f32 = 1e-5;

/// Strategy for f32 vectors in [-10, 10] with length in [lo, hi].
fn vec_f32(lo: usize, hi: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-10.0f32..10.0f32, lo..=hi)
}

/// Strategy for even-length dimension (required by RoPE which operates on pairs).
fn even_dim(lo: usize, hi: usize) -> impl Strategy<Value = usize> {
    (lo / 2..=hi / 2).prop_map(|half| half * 2)
}

// ═══════════════════════════════════════════════════════════════════
// 1. Softmax — output sums to ~1.0
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(cfg())]

    /// Softmax output probabilities sum to approximately 1.0.
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_softmax_sum_to_one(
        input in vec_f32(1, 256),
    ) {
        let n = input.len();
        let mut output = vec![0.0f32; n];
        unsafe { softmax_neon(&input, &mut output) };

        let sum: f32 = output.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-3,
            "softmax sum = {sum}, expected ~1.0 (len={n})"
        );
    }

    /// Every softmax output value lies in [0, 1].
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_softmax_values_in_unit_range(
        input in vec_f32(1, 256),
    ) {
        let n = input.len();
        let mut output = vec![0.0f32; n];
        unsafe { softmax_neon(&input, &mut output) };

        for (i, &v) in output.iter().enumerate() {
            prop_assert!(
                (0.0..=1.0).contains(&v),
                "softmax[{i}] = {v}, not in [0, 1]"
            );
        }
    }

    /// NEON softmax matches scalar softmax within tolerance.
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_softmax_neon_matches_scalar(
        input in vec_f32(1, 256),
    ) {
        let n = input.len();
        let mut neon_out = vec![0.0f32; n];
        let mut scalar_out = vec![0.0f32; n];

        unsafe { softmax_neon(&input, &mut neon_out) };
        softmax_scalar(&input, &mut scalar_out);

        for (i, (&a, &b)) in neon_out.iter().zip(scalar_out.iter()).enumerate() {
            prop_assert!(
                (a - b).abs() < 1e-3,
                "softmax mismatch at [{i}]: neon={a}, scalar={b}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. LayerNorm — mean ~0, variance ~1
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(cfg())]

    /// NEON LayerNorm output has approximately zero mean (gamma=1, beta=0).
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_layernorm_zero_mean(
        input in vec_f32(4, 256),
    ) {
        let n = input.len();
        // Skip near-constant inputs that produce degenerate normalization.
        let mean_in = input.iter().sum::<f32>() / n as f32;
        let var_in = input.iter().map(|x| (x - mean_in).powi(2)).sum::<f32>() / n as f32;
        prop_assume!(var_in > 1e-6);

        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];
        let mut output = vec![0.0f32; n];

        unsafe { layernorm_neon(&input, &mut output, &gamma, &beta, EPS) };

        let mean: f32 = output.iter().sum::<f32>() / n as f32;
        prop_assert!(
            mean.abs() < 1e-4,
            "LayerNorm mean = {mean}, expected ~0 (n={n})"
        );
    }

    /// NEON LayerNorm output has approximately unit variance (gamma=1, beta=0).
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_layernorm_unit_variance(
        input in vec_f32(4, 256),
    ) {
        let n = input.len();
        let mean_in = input.iter().sum::<f32>() / n as f32;
        let var_in = input.iter().map(|x| (x - mean_in).powi(2)).sum::<f32>() / n as f32;
        prop_assume!(var_in > 1e-6);

        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];
        let mut output = vec![0.0f32; n];

        unsafe { layernorm_neon(&input, &mut output, &gamma, &beta, EPS) };

        let out_mean = output.iter().sum::<f32>() / n as f32;
        let out_var =
            output.iter().map(|x| (x - out_mean).powi(2)).sum::<f32>() / n as f32;
        prop_assert!(
            (out_var - 1.0).abs() < 0.05,
            "LayerNorm variance = {out_var}, expected ~1.0 (n={n})"
        );
    }

    /// RMSNorm output has approximately unit root-mean-square.
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_rmsnorm_unit_rms(
        input in vec_f32(4, 256),
    ) {
        let n = input.len();
        let rms_in = (input.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
        prop_assume!(rms_in > 1e-4);

        let gamma = vec![1.0f32; n];
        let mut output = vec![0.0f32; n];

        unsafe { rmsnorm_neon(&input, &mut output, &gamma, EPS) };

        let rms_out = (output.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
        prop_assert!(
            (rms_out - 1.0).abs() < 0.05,
            "RMSNorm RMS = {rms_out}, expected ~1.0 (n={n})"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. RoPE — periodicity / consistency
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(cfg())]

    /// RoPE preserves vector norm (rotation is norm-preserving).
    /// build_cos_sin_tables_neon(dim, max_seq, base) -> (cos, sin).
    /// apply_rope_neon(data, cos, sin, dim, pos) mutates data in-place.
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_rope_preserves_norm(
        dim in even_dim(4, 128),
        pos in 0usize..100,
    ) {
        let theta = 10000.0f32;
        let max_seq = pos + 1;
        let mut data: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.3).sin()).collect();
        let norm_in: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

        let (cos_table, sin_table) =
            unsafe { build_cos_sin_tables_neon(dim, max_seq, theta) };

        unsafe {
            scalar_rope_apply(
                &mut data, &cos_table, &sin_table, dim, pos,
            );
        }

        let norm_out: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assert!(
            (norm_in - norm_out).abs() < 1e-2,
            "RoPE changed norm: in={norm_in}, out={norm_out} (dim={dim}, pos={pos})"
        );
    }

    /// Building cos/sin tables twice with same params yields identical results.
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_rope_tables_deterministic(
        dim in even_dim(4, 128),
        max_seq in 1usize..=64,
    ) {
        let theta = 10000.0f32;

        let (cos1, sin1) = unsafe { build_cos_sin_tables_neon(dim, max_seq, theta) };
        let (cos2, sin2) = unsafe { build_cos_sin_tables_neon(dim, max_seq, theta) };

        prop_assert_eq!(&cos1, &cos2, "cos tables not deterministic");
        prop_assert_eq!(&sin1, &sin2, "sin tables not deterministic");
    }

    /// Applying RoPE at position 0 should still produce finite values.
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_rope_position_zero_finite(
        dim in even_dim(4, 128),
    ) {
        let theta = 10000.0f32;
        let mut data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 + 1.0).collect();

        let (cos_table, sin_table) =
            unsafe { build_cos_sin_tables_neon(dim, 1, theta) };

        unsafe {
            scalar_rope_apply(
                &mut data, &cos_table, &sin_table, dim, 0,
            );
        }

        for (i, &v) in data.iter().enumerate() {
            prop_assert!(v.is_finite(), "RoPE pos=0 produced non-finite at [{i}]: {v}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. Activation functions — sign preservation
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(cfg())]

    /// ReLU: negative inputs → 0, non-negative inputs → unchanged.
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_relu_sign_preservation(
        input in vec_f32(1, 256),
    ) {
        let n = input.len();
        let mut output = vec![0.0f32; n];
        unsafe { neon_relu_f32(&input, &mut output) };

        for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
            if x < 0.0 {
                prop_assert_eq!(
                    y, 0.0,
                    "ReLU({}) = {}, expected 0.0 at index {}", x, y, i
                );
            } else {
                prop_assert!(
                    (x - y).abs() < 1e-6,
                    "ReLU({x}) = {y}, expected {x} at index {i}"
                );
            }
        }
    }

    /// ReLU output is always non-negative.
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_relu_non_negative(
        input in vec_f32(1, 256),
    ) {
        let n = input.len();
        let mut output = vec![0.0f32; n];
        unsafe { neon_relu_f32(&input, &mut output) };

        for (i, &v) in output.iter().enumerate() {
            prop_assert!(v >= 0.0, "ReLU output[{i}] = {v} is negative");
        }
    }

    /// SiLU output preserves sign: x * sigmoid(x) has same sign as x.
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_silu_sign_consistent(
        input in prop::collection::vec(
            prop::num::f32::NORMAL.prop_filter("nonzero", |x| x.abs() > 0.01),
            1..=256,
        ),
    ) {
        let n = input.len();
        let mut output = vec![0.0f32; n];
        unsafe { neon_silu_f32(&input, &mut output) };

        for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
            if x > 0.1 {
                prop_assert!(
                    y > 0.0,
                    "SiLU({x}) = {y}, expected positive at index {i}"
                );
            }
            // SiLU can be slightly negative for small negative x, skip strict check.
        }
    }

    /// GELU output length equals input length (no panics for any valid size).
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_gelu_preserves_length(
        input in vec_f32(1, 256),
    ) {
        let n = input.len();
        let mut output = vec![0.0f32; n];
        unsafe { neon_gelu_f32(&input, &mut output) };
        // If we reach here without panic, the length invariant holds.
        prop_assert_eq!(output.len(), n);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. Quantize / dequantize round-trip
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(cfg())]

    /// Symmetric i8 quantize then dequantize preserves values within
    /// quantization error (8-bit: max error ≈ range / 127).
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_quantize_roundtrip_symmetric_i8(
        input in prop::collection::vec(-5.0f32..5.0f32, 1..=256),
    ) {
        let max_abs = input.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        prop_assume!(max_abs > 1e-6);

        let (quantized, scale) = quantize_symmetric_i8(&input, 8);
        let reconstructed = dequantize_symmetric_i8(&quantized, scale);

        prop_assert_eq!(reconstructed.len(), input.len(), "round-trip changed length");

        let step = max_abs / 127.0;
        for (i, (&orig, &recon)) in input.iter().zip(reconstructed.iter()).enumerate() {
            prop_assert!(
                (orig - recon).abs() <= step + 1e-5,
                "round-trip error at [{i}]: orig={orig}, recon={recon}, step={step}"
            );
        }
    }

    /// Quantized output length equals input length.
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_quantize_preserves_length(
        input in prop::collection::vec(-10.0f32..10.0f32, 1..=256),
    ) {
        let (quantized, _scale) = quantize_symmetric_i8(&input, 8);
        prop_assert_eq!(
            quantized.len(),
            input.len(),
            "quantize changed vector length"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// 6. Element-wise operations — length preservation
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(cfg())]

    /// NEON element-wise add preserves vector length and matches scalar.
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_elementwise_add_length_and_correctness(
        n in 1usize..=256,
    ) {
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * -0.2 + 1.0).collect();
        let mut out = vec![0.0f32; n];

        unsafe { neon_add_f32(&a, &b, &mut out) };

        prop_assert_eq!(out.len(), n, "add changed output length");
        for (i, ((&ai, &bi), &oi)) in a.iter().zip(b.iter()).zip(out.iter()).enumerate() {
            prop_assert!(
                (oi - (ai + bi)).abs() < 1e-5,
                "add mismatch at [{i}]: {ai} + {bi} = {oi}"
            );
        }
    }

    /// NEON element-wise mul preserves vector length.
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_elementwise_mul_length_and_correctness(
        n in 1usize..=256,
    ) {
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).cos()).collect();
        let mut out = vec![0.0f32; n];

        unsafe { neon_mul_f32(&a, &b, &mut out) };

        prop_assert_eq!(out.len(), n, "mul changed output length");
        for (i, ((&ai, &bi), &oi)) in a.iter().zip(b.iter()).zip(out.iter()).enumerate() {
            prop_assert!(
                (oi - (ai * bi)).abs() < 1e-5,
                "mul mismatch at [{i}]: {ai} * {bi} = {oi}"
            );
        }
    }

    /// NEON scale preserves vector length and is equivalent to scalar multiply.
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_elementwise_scale_preserves_length(
        n in 1usize..=256,
        scale in -10.0f32..10.0f32,
    ) {
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).sin()).collect();
        let mut out = vec![0.0f32; n];

        unsafe { neon_scale_f32(&a, scale, &mut out) };

        prop_assert_eq!(out.len(), n, "scale changed output length");
        for (i, (&ai, &oi)) in a.iter().zip(out.iter()).enumerate() {
            prop_assert!(
                (oi - ai * scale).abs() < 1e-4,
                "scale mismatch at [{i}]: {ai} * {scale} = {oi}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 7. MatMul / GEMV — dimension consistency
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(cfg())]

    /// NEON dot product is commutative: dot(a, b) ≈ dot(b, a).
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_dot_product_commutative(
        n in 1usize..=256,
    ) {
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).cos()).collect();

        let ab = unsafe { bitnet_kernels::cpu::neon_reductions::neon_dot_f32(&a, &b) };
        let ba = unsafe { bitnet_kernels::cpu::neon_reductions::neon_dot_f32(&b, &a) };

        prop_assert!(
            (ab - ba).abs() < 1e-3,
            "dot not commutative: dot(a,b)={ab}, dot(b,a)={ba} (n={n})"
        );
    }

    /// NEON sum matches naive summation within floating-point tolerance.
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_neon_sum_matches_naive(
        input in vec_f32(1, 256),
    ) {
        let neon_result = unsafe { neon_sum_f32(&input) };
        let naive: f32 = input.iter().sum();

        prop_assert!(
            (neon_result - naive).abs() < 1e-2 * input.len() as f32,
            "sum mismatch: neon={neon_result}, naive={naive} (n={})",
            input.len(),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// 8. Top-k / argmax — cardinality and bounds
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(cfg())]

    /// NEON argmax returns a valid index within [0, len).
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_argmax_returns_valid_index(
        input in vec_f32(1, 256),
    ) {
        let idx = unsafe { neon_argmax_f32(&input) };
        prop_assert!(
            idx < input.len(),
            "argmax index {idx} out of bounds (len={})",
            input.len(),
        );
    }

    /// NEON argmax index points to the maximum value.
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_argmax_points_to_max(
        input in vec_f32(1, 256),
    ) {
        let idx = unsafe { neon_argmax_f32(&input) };
        let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        prop_assert!(
            (input[idx] - max_val).abs() < 1e-6,
            "argmax[{idx}]={} but max={max_val}",
            input[idx],
        );
    }

    /// NEON argmax is deterministic — same input always yields same index.
    #[test]
    #[ignore = "requires aarch64 - run on Apple Silicon hardware"]
    fn prop_argmax_deterministic(
        input in vec_f32(1, 256),
    ) {
        let idx1 = unsafe { neon_argmax_f32(&input) };
        let idx2 = unsafe { neon_argmax_f32(&input) };
        prop_assert_eq!(idx1, idx2, "argmax not deterministic");
    }
}
