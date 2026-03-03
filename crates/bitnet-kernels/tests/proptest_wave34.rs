//! Wave 34 property tests for `bitnet-kernels`.
//!
//! 20 properties covering:
//! - Softmax outputs sum to 1.0 (within tolerance)
//! - Linear transform (W*x + b) shape invariants
//! - Layer norm output mean ≈ 0 (within tolerance)
//! - RMS norm output RMS ≈ 1 (within tolerance)
//! - Quantize→dequantize approximate round-trip
//! - Transpose(Transpose(A)) == A
//! - SIMD vs scalar parity for dot product (via matmul 1×N @ N×1)
//! - Matrix multiplication associativity (within numerical tolerance)

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::linear::{LinearConfig, linear_forward};
use bitnet_kernels::cpu::quantize::{
    dequantize_asymmetric_u8, dequantize_symmetric_i8, quantize_asymmetric_u8,
    quantize_symmetric_i8,
};
use bitnet_kernels::cpu::transpose::TransposeKernel;
use bitnet_kernels::cuda::matmul::{MatmulConfig, matmul_cpu};
use bitnet_kernels::cuda::softmax::{SoftmaxConfig, softmax_cpu};
use proptest::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn vec_strategy(n: usize, lo: f32, hi: f32) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(lo..hi, n..=n)
}

fn bounded_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-5.0f32..5.0, min_len..=max_len)
}

// ── 1. Softmax sum-to-one ───────────────────────────────────────────────────

proptest! {
    /// Softmax output sums to 1.0 (± 1e-5).
    #[test]
    fn prop_softmax_sum_to_one(n in 2usize..64) {
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
        let mut out = vec![0.0f32; n];
        softmax_cpu(&data, &mut out, &cfg).unwrap();
        let sum: f32 = out.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5,
            "softmax sum = {sum}, expected ≈ 1.0");
    }

    /// Softmax outputs are all non-negative.
    #[test]
    fn prop_softmax_non_negative(data in bounded_vec(2, 64)) {
        let n = data.len();
        let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
        let mut out = vec![0.0f32; n];
        softmax_cpu(&data, &mut out, &cfg).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v >= 0.0, "softmax[{i}] = {v} < 0");
        }
    }

    /// Softmax preserves ordering: larger logit → larger probability.
    #[test]
    fn prop_softmax_preserves_order(data in bounded_vec(2, 32)) {
        let n = data.len();
        let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
        let mut out = vec![0.0f32; n];
        softmax_cpu(&data, &mut out, &cfg).unwrap();
        for i in 0..n {
            for j in (i + 1)..n {
                if data[i] > data[j] {
                    prop_assert!(out[i] >= out[j],
                        "ordering: data[{i}]={} > data[{j}]={} but prob {} < {}",
                        data[i], data[j], out[i], out[j]);
                }
            }
        }
    }
}

// ── 2. Linear transform shape invariants ────────────────────────────────────

proptest! {
    /// Linear (W*x + b) produces output of correct length (batch * out_features).
    #[test]
    fn prop_linear_output_length(
        batch in 1usize..4,
        in_f in 2usize..16,
        out_f in 2usize..16,
    ) {
        let x: Vec<f32> = (0..batch * in_f).map(|i| (i as f32) * 0.01).collect();
        let w: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32) * 0.01).collect();
        let bias: Vec<f32> = vec![0.1; out_f];
        let cfg = LinearConfig::new(batch, in_f, out_f).unwrap().with_bias(true);
        let mut out = vec![0.0f32; batch * out_f];
        linear_forward(&x, &w, Some(&bias), &mut out, &cfg).unwrap();
        prop_assert_eq!(out.len(), batch * out_f);
    }

    /// Linear without bias: output is finite for finite inputs.
    #[test]
    fn prop_linear_no_bias_finite(
        in_f in 2usize..8,
        out_f in 2usize..8,
    ) {
        let x: Vec<f32> = (0..in_f).map(|i| (i as f32) * 0.1).collect();
        let w: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32) * 0.01).collect();
        let cfg = LinearConfig::new(1, in_f, out_f).unwrap();
        let mut out = vec![0.0f32; out_f];
        linear_forward(&x, &w, None, &mut out, &cfg).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.is_finite(), "linear out[{i}] = {v} is not finite");
        }
    }
}

// ── 3. Layer norm output mean ≈ 0 ───────────────────────────────────────────

proptest! {
    /// Layer norm (with beta=0, gamma=1) produces near-zero mean output.
    #[test]
    fn prop_layer_norm_zero_mean(n in 4usize..64) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];
        let cfg = LayerNormConfig::new(vec![n]);
        let out = layer_norm(&input, &gamma, Some(&beta), &cfg).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / n as f32;
        prop_assert!(mean.abs() < 0.01,
            "layer_norm mean = {mean}, expected ≈ 0");
    }

    /// Layer norm with non-trivial gamma still centres around 0 when beta=0.
    #[test]
    fn prop_layer_norm_scaled_zero_mean(n in 4usize..32) {
        let input: Vec<f32> = (0..n).map(|i| ((i * 7 % 11) as f32) - 5.0).collect();
        let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.05).collect();
        let beta = vec![0.0f32; n];
        let cfg = LayerNormConfig::new(vec![n]);
        let out = layer_norm(&input, &gamma, Some(&beta), &cfg).unwrap();
        // Weighted mean is not exactly 0, but should be small.
        let mean: f32 = out.iter().sum::<f32>() / n as f32;
        prop_assert!(mean.abs() < 1.0,
            "layer_norm scaled mean = {mean}, expected small");
    }
}

// ── 4. RMS norm output RMS ≈ 1 ─────────────────────────────────────────────

proptest! {
    /// RMS norm with gamma=1 produces output whose RMS ≈ 1.
    #[test]
    fn prop_rms_norm_unit_rms(n in 4usize..64) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.2 - 1.5).collect();
        let gamma = vec![1.0f32; n];
        let cfg = LayerNormConfig::new(vec![n]);
        let out = rms_norm(&input, &gamma, &cfg).unwrap();
        let rms = (out.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
        prop_assert!((rms - 1.0).abs() < 0.05,
            "rms_norm RMS = {rms}, expected ≈ 1.0");
    }

    /// RMS norm preserves sign (element-wise).
    #[test]
    fn prop_rms_norm_preserves_sign(n in 4usize..32) {
        let input: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let gamma = vec![1.0f32; n];
        let cfg = LayerNormConfig::new(vec![n]);
        let out = rms_norm(&input, &gamma, &cfg).unwrap();
        for (i, (&inp, &o)) in input.iter().zip(out.iter()).enumerate() {
            prop_assert!(inp.signum() == o.signum(),
                "sign mismatch at {i}: input {} vs output {}", inp, o);
        }
    }

    /// RMS norm output length matches input length.
    #[test]
    fn prop_rms_norm_output_length(n in 4usize..64) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let gamma = vec![1.0f32; n];
        let cfg = LayerNormConfig::new(vec![n]);
        let out = rms_norm(&input, &gamma, &cfg).unwrap();
        prop_assert_eq!(out.len(), n);
    }
}

// ── 5. Quantize → dequantize round-trip ─────────────────────────────────────

proptest! {
    /// Symmetric i8 quantize→dequantize has bounded error.
    #[test]
    fn prop_symmetric_i8_roundtrip(input in bounded_vec(4, 64)) {
        let (quantized, scale) = quantize_symmetric_i8(&input, 8);
        let recovered = dequantize_symmetric_i8(&quantized, scale);
        for (i, (&orig, &rec)) in input.iter().zip(recovered.iter()).enumerate() {
            let err = (orig - rec).abs();
            // 8-bit quantization: max error ≤ scale (one quantization step).
            prop_assert!(err <= scale + 1e-6,
                "symmetric roundtrip error[{i}] = {err} > scale {scale}");
        }
    }

    /// Asymmetric u8 quantize→dequantize has bounded error.
    #[test]
    fn prop_asymmetric_u8_roundtrip(input in bounded_vec(4, 64)) {
        let (quantized, scale, zp) = quantize_asymmetric_u8(&input);
        let recovered = dequantize_asymmetric_u8(&quantized, scale, zp);
        for (i, (&orig, &rec)) in input.iter().zip(recovered.iter()).enumerate() {
            let err = (orig - rec).abs();
            prop_assert!(err <= scale + 1e-5,
                "asymmetric roundtrip error[{i}] = {err} > scale {scale}");
        }
    }
}

// ── 6. Transpose involution ─────────────────────────────────────────────────

proptest! {
    /// Transpose(Transpose(A)) == A for 2D matrices.
    #[test]
    fn prop_transpose_involution(
        rows in 1usize..16,
        cols in 1usize..16,
    ) {
        let data: Vec<f32> = (0..(rows * cols)).map(|i| i as f32).collect();
        let t1 = TransposeKernel::transpose_2d(&data, rows, cols).unwrap();
        let t2 = TransposeKernel::transpose_2d(&t1, cols, rows).unwrap();
        prop_assert_eq!(t2, data, "transpose²(A) != A");
    }

    /// Transpose of a 1×N matrix is N×1 (shape correctness).
    #[test]
    fn prop_transpose_shape(n in 1usize..32) {
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let transposed = TransposeKernel::transpose_2d(&data, 1, n).unwrap();
        prop_assert_eq!(transposed.len(), n);
        // Transposing 1×N then N×1 back should give original.
        let back = TransposeKernel::transpose_2d(&transposed, n, 1).unwrap();
        prop_assert_eq!(back, data);
    }
}

// ── 7. SIMD vs scalar parity (dot product via 1×N @ N×1 matmul) ────────────

proptest! {
    /// 1×N @ N×1 matmul gives the same result as manual dot product.
    #[test]
    fn prop_dot_product_via_matmul(
        a in vec_strategy(8, -2.0, 2.0),
        b in vec_strategy(8, -2.0, 2.0),
    ) {
        let n = a.len();
        // Manual dot product.
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        // 1×N @ N×1 matmul → 1×1
        let cfg = MatmulConfig::for_shape(1, 1, n).unwrap();
        let mut out = vec![0.0f32; 1];
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();
        prop_assert!((out[0] - expected).abs() < 1e-3,
            "matmul dot {} vs manual {}", out[0], expected);
    }

    /// Dot product is commutative: a·b == b·a.
    #[test]
    fn prop_dot_product_commutative(
        a in vec_strategy(8, -2.0, 2.0),
        b in vec_strategy(8, -2.0, 2.0),
    ) {
        let cfg = MatmulConfig::for_shape(1, 1, 8).unwrap();
        let mut ab = vec![0.0f32; 1];
        let mut ba = vec![0.0f32; 1];
        matmul_cpu(&a, &b, &mut ab, &cfg).unwrap();
        matmul_cpu(&b, &a, &mut ba, &cfg).unwrap();
        prop_assert!((ab[0] - ba[0]).abs() < 1e-4,
            "a·b = {} but b·a = {}", ab[0], ba[0]);
    }
}

// ── 8. Matrix multiplication associativity ──────────────────────────────────

proptest! {
    /// (A*B)*C ≈ A*(B*C) for 2×2 matrices.
    #[test]
    fn prop_matmul_associativity(
        a in vec_strategy(4, -2.0, 2.0),
        b in vec_strategy(4, -2.0, 2.0),
        c in vec_strategy(4, -2.0, 2.0),
    ) {
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap();
        let mut ab = vec![0.0f32; 4];
        matmul_cpu(&a, &b, &mut ab, &cfg).unwrap();
        let mut abc_left = vec![0.0f32; 4];
        matmul_cpu(&ab, &c, &mut abc_left, &cfg).unwrap();

        let mut bc = vec![0.0f32; 4];
        matmul_cpu(&b, &c, &mut bc, &cfg).unwrap();
        let mut abc_right = vec![0.0f32; 4];
        matmul_cpu(&a, &bc, &mut abc_right, &cfg).unwrap();

        for i in 0..4 {
            prop_assert!((abc_left[i] - abc_right[i]).abs() < 1e-3,
                "(AB)C[{i}]={} vs A(BC)[{i}]={}", abc_left[i], abc_right[i]);
        }
    }

    /// Matmul with identity: A*I == A.
    #[test]
    fn prop_matmul_identity(a in vec_strategy(4, -3.0, 3.0)) {
        let identity = vec![1.0, 0.0, 0.0, 1.0f32];
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap();
        let mut result = vec![0.0f32; 4];
        matmul_cpu(&a, &identity, &mut result, &cfg).unwrap();
        for i in 0..4 {
            prop_assert!((result[i] - a[i]).abs() < 1e-5,
                "A*I [{i}]: {} vs {}", result[i], a[i]);
        }
    }

    /// Matmul distributes over addition: A*(B+C) ≈ A*B + A*C.
    #[test]
    fn prop_matmul_distributive(
        a in vec_strategy(4, -1.0, 1.0),
        b in vec_strategy(4, -1.0, 1.0),
        c in vec_strategy(4, -1.0, 1.0),
    ) {
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap();

        // B + C
        let bc_sum: Vec<f32> = b.iter().zip(c.iter()).map(|(x, y)| x + y).collect();
        // A * (B+C)
        let mut a_bc = vec![0.0f32; 4];
        matmul_cpu(&a, &bc_sum, &mut a_bc, &cfg).unwrap();

        // A*B + A*C
        let mut ab = vec![0.0f32; 4];
        let mut ac = vec![0.0f32; 4];
        matmul_cpu(&a, &b, &mut ab, &cfg).unwrap();
        matmul_cpu(&a, &c, &mut ac, &cfg).unwrap();
        let ab_plus_ac: Vec<f32> = ab.iter().zip(ac.iter()).map(|(x, y)| x + y).collect();

        for i in 0..4 {
            prop_assert!((a_bc[i] - ab_plus_ac[i]).abs() < 1e-3,
                "distributive [{i}]: {} vs {}", a_bc[i], ab_plus_ac[i]);
        }
    }

    /// Softmax is shift-invariant: softmax(x) == softmax(x + c).
    #[test]
    fn prop_softmax_shift_invariant(data in bounded_vec(2, 32)) {
        let n = data.len();
        let shifted: Vec<f32> = data.iter().map(|x| x + 3.0).collect();
        let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
        let mut out1 = vec![0.0f32; n];
        let mut out2 = vec![0.0f32; n];
        softmax_cpu(&data, &mut out1, &cfg).unwrap();
        softmax_cpu(&shifted, &mut out2, &cfg).unwrap();
        for i in 0..n {
            prop_assert!((out1[i] - out2[i]).abs() < 1e-5,
                "shift invariance [{i}]: {} vs {}", out1[i], out2[i]);
        }
    }
}
