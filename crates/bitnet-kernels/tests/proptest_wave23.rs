//! Property-based tests — wave 23.
//!
//! Kernel correctness invariants: matmul associativity, transpose involution,
//! softmax sum-to-one & ordering, layer-norm zero-mean & unit-variance,
//! activation bounds & idempotence, embedding determinism, quantize-dequantize
//! bounded error, residual identity, concat associativity, GeLU lower bound,
//! and causal mask lower-triangularity.
//!
//! 50+ property assertions across 14 invariant categories.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::activations::{gelu, relu};
use bitnet_kernels::cpu::attention::causal_mask;
use bitnet_kernels::cpu::concat::ConcatKernel;
use bitnet_kernels::cpu::embedding::embedding_lookup;
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm};
use bitnet_kernels::cpu::quantize::{
    dequantize_asymmetric_u8, dequantize_symmetric_i8, quantize_asymmetric_u8,
    quantize_symmetric_i8,
};
use bitnet_kernels::cpu::residual::add_residual;
use bitnet_kernels::cpu::transpose::TransposeKernel;
use bitnet_kernels::cuda::matmul::{MatmulConfig, matmul_cpu};
use bitnet_kernels::cuda::softmax::{SoftmaxConfig, softmax_cpu};
use proptest::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Generate a Vec<f32> of length `n` with values in `lo..hi`.
fn vec_strategy(n: usize, lo: f32, hi: f32) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(lo..hi, n..=n)
}

// ── 1. MatMul associativity ─────────────────────────────────────────────────

proptest! {
    /// (A*B)*C ≈ A*(B*C) for small square matrices.
    #[test]
    fn matmul_associativity(
        a in vec_strategy(4, -2.0, 2.0),
        b in vec_strategy(4, -2.0, 2.0),
        c in vec_strategy(4, -2.0, 2.0),
    ) {
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap();
        // AB = A * B
        let mut ab = vec![0.0f32; 4];
        matmul_cpu(&a, &b, &mut ab, &cfg).unwrap();
        // (AB)C
        let mut abc_left = vec![0.0f32; 4];
        matmul_cpu(&ab, &c, &mut abc_left, &cfg).unwrap();
        // BC = B * C
        let mut bc = vec![0.0f32; 4];
        matmul_cpu(&b, &c, &mut bc, &cfg).unwrap();
        // A(BC)
        let mut abc_right = vec![0.0f32; 4];
        matmul_cpu(&a, &bc, &mut abc_right, &cfg).unwrap();

        for i in 0..4 {
            prop_assert!(
                (abc_left[i] - abc_right[i]).abs() < 1e-3,
                "(A*B)*C[{}]={} != A*(B*C)[{}]={}",
                i, abc_left[i], i, abc_right[i]
            );
        }
    }

    /// MatMul with identity matrix preserves the input.
    #[test]
    fn matmul_identity(
        a in vec_strategy(4, -5.0, 5.0),
    ) {
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap();
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0f32; 4];
        matmul_cpu(&a, &identity, &mut out, &cfg).unwrap();
        for i in 0..4 {
            prop_assert!(
                (out[i] - a[i]).abs() < 1e-5,
                "A*I[{}]={} != A[{}]={}", i, out[i], i, a[i]
            );
        }
    }

    /// MatMul with zero matrix yields zero.
    #[test]
    fn matmul_zero(
        a in vec_strategy(4, -5.0, 5.0),
    ) {
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap();
        let zeros = vec![0.0f32; 4];
        let mut out = vec![0.0f32; 4];
        matmul_cpu(&a, &zeros, &mut out, &cfg).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(
                v.abs() < 1e-6,
                "A*0[{}]={} != 0", i, v
            );
        }
    }

    /// MatMul output is finite for bounded inputs.
    #[test]
    fn matmul_output_finite(
        m in 1usize..5,
        k in 1usize..5,
        n in 1usize..5,
    ) {
        let a = vec![1.0f32; m * k];
        let b = vec![0.5f32; k * n];
        let cfg = MatmulConfig::for_shape(m, n, k).unwrap();
        let mut out = vec![0.0f32; m * n];
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.is_finite(), "out[{}] not finite", i);
        }
    }
}

// ── 2. Transpose involution ─────────────────────────────────────────────────

proptest! {
    /// transpose(transpose(x)) == x.
    #[test]
    fn transpose_involution(
        rows in 1usize..8,
        cols in 1usize..8,
    ) {
        let n = rows * cols;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let t1 = TransposeKernel::transpose_2d(&data, rows, cols).unwrap();
        let t2 = TransposeKernel::transpose_2d(&t1, cols, rows).unwrap();
        prop_assert_eq!(data, t2);
    }

    /// Transpose preserves element count.
    #[test]
    fn transpose_preserves_length(
        rows in 1usize..16,
        cols in 1usize..16,
    ) {
        let data: Vec<f32> = (0..(rows * cols)).map(|i| i as f32).collect();
        let t = TransposeKernel::transpose_2d(&data, rows, cols).unwrap();
        prop_assert_eq!(t.len(), data.len());
    }

    /// Transpose of a 1×N is an N×1 column vector.
    #[test]
    fn transpose_row_to_col(n in 1usize..32) {
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let t = TransposeKernel::transpose_2d(&data, 1, n).unwrap();
        // t is n×1, same elements
        prop_assert_eq!(t, data);
    }

    /// Transpose of square matrix: diagonal is unchanged.
    #[test]
    fn transpose_diagonal_unchanged(n in 1usize..8) {
        let data: Vec<f32> = (0..(n * n)).map(|i| i as f32).collect();
        let t = TransposeKernel::transpose_2d(&data, n, n).unwrap();
        for i in 0..n {
            prop_assert_eq!(
                data[i * n + i], t[i * n + i],
                "diagonal[{}] changed", i
            );
        }
    }
}

// ── 3. Softmax sum-to-one ───────────────────────────────────────────────────

proptest! {
    /// sum(softmax(x)) ≈ 1.0 for any row.
    #[test]
    fn softmax_sum_to_one(
        data in proptest::collection::vec(-10.0f32..10.0, 2..32),
    ) {
        let n = data.len();
        let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
        let mut out = vec![0.0f32; n];
        softmax_cpu(&data, &mut out, &cfg).unwrap();
        let sum: f32 = out.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax sum = {}, expected ≈1.0", sum
        );
    }

    /// Each softmax output is in (0, 1).
    #[test]
    fn softmax_elements_in_unit_interval(
        data in proptest::collection::vec(-10.0f32..10.0, 2..32),
    ) {
        let n = data.len();
        let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
        let mut out = vec![0.0f32; n];
        softmax_cpu(&data, &mut out, &cfg).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v > 0.0 && v < 1.0,
                "softmax[{}]={} not in (0,1)", i, v);
        }
    }

    /// Multi-row softmax: each row sums to 1.
    #[test]
    fn softmax_multi_row_sum(
        n_cols in 2usize..16,
        n_rows in 1usize..8,
    ) {
        let total = n_cols * n_rows;
        let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let cfg = SoftmaxConfig::for_shape(n_cols, n_rows).unwrap();
        let mut out = vec![0.0f32; total];
        softmax_cpu(&data, &mut out, &cfg).unwrap();
        for row in 0..n_rows {
            let start = row * n_cols;
            let sum: f32 = out[start..start + n_cols].iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-4,
                "row {} sum = {}", row, sum
            );
        }
    }

    /// Softmax output is non-negative.
    #[test]
    fn softmax_non_negative(
        data in proptest::collection::vec(-10.0f32..10.0, 2..32),
    ) {
        let n = data.len();
        let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
        let mut out = vec![0.0f32; n];
        softmax_cpu(&data, &mut out, &cfg).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v >= 0.0, "softmax[{}]={} negative", i, v);
        }
    }
}

// ── 4. Softmax ordering preserved ───────────────────────────────────────────

proptest! {
    /// If x[i] > x[j] then softmax(x)[i] > softmax(x)[j].
    #[test]
    fn softmax_preserves_ordering(
        data in proptest::collection::vec(-10.0f32..10.0, 3..16),
    ) {
        let n = data.len();
        let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
        let mut out = vec![0.0f32; n];
        softmax_cpu(&data, &mut out, &cfg).unwrap();
        for i in 0..n {
            for j in (i + 1)..n {
                if (data[i] - data[j]).abs() > 1e-6 {
                    if data[i] > data[j] {
                        prop_assert!(
                            out[i] >= out[j] - 1e-7,
                            "ordering violated: data[{}]={} > data[{}]={} but softmax[{}]={} < softmax[{}]={}",
                            i, data[i], j, data[j], i, out[i], j, out[j]
                        );
                    } else {
                        prop_assert!(
                            out[j] >= out[i] - 1e-7,
                            "ordering violated: data[{}]={} < data[{}]={} but softmax[{}]={} > softmax[{}]={}",
                            i, data[i], j, data[j], i, out[i], j, out[j]
                        );
                    }
                }
            }
        }
    }

    /// Softmax of equal inputs yields uniform distribution.
    #[test]
    fn softmax_uniform_for_equal_inputs(n in 2usize..32) {
        let data = vec![1.0f32; n];
        let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();
        let mut out = vec![0.0f32; n];
        softmax_cpu(&data, &mut out, &cfg).unwrap();
        let expected = 1.0 / n as f32;
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(
                (v - expected).abs() < 1e-5,
                "softmax[{}]={} != 1/{}", i, v, n
            );
        }
    }
}

// ── 5. LayerNorm zero-mean ──────────────────────────────────────────────────

proptest! {
    /// mean(layernorm(x)) ≈ 0 with gamma=1, no beta.
    #[test]
    fn layernorm_zero_mean(
        dim in 4usize..32,
    ) {
        let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let gamma = vec![1.0f32; dim];
        let mut config = LayerNormConfig::new(vec![dim]);
        config.elementwise_affine = true;
        let out = layer_norm(&input, &gamma, None, &config).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / dim as f32;
        prop_assert!(
            mean.abs() < 1e-4,
            "layernorm mean = {}, expected ≈0", mean
        );
    }

    /// LayerNorm zero-mean holds for random inputs.
    #[test]
    fn layernorm_zero_mean_random(
        data in proptest::collection::vec(-5.0f32..5.0, 4..32),
    ) {
        let dim = data.len();
        let gamma = vec![1.0f32; dim];
        let mut config = LayerNormConfig::new(vec![dim]);
        config.elementwise_affine = true;
        let out = layer_norm(&data, &gamma, None, &config).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / dim as f32;
        prop_assert!(
            mean.abs() < 1e-3,
            "layernorm random mean = {}", mean
        );
    }
}

// ── 6. LayerNorm unit-variance ──────────────────────────────────────────────

proptest! {
    /// var(layernorm(x)) ≈ 1 with gamma=1, no beta.
    #[test]
    fn layernorm_unit_variance(
        dim in 4usize..32,
    ) {
        let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let gamma = vec![1.0f32; dim];
        let mut config = LayerNormConfig::new(vec![dim]);
        config.elementwise_affine = true;
        let out = layer_norm(&input, &gamma, None, &config).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / dim as f32;
        let var: f32 = out.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
        prop_assert!(
            (var - 1.0).abs() < 0.15,
            "layernorm variance = {}, expected ≈1.0", var
        );
    }

    /// LayerNorm unit-variance with random inputs.
    #[test]
    fn layernorm_unit_variance_random(
        data in proptest::collection::vec(-5.0f32..5.0, 8..32),
    ) {
        let dim = data.len();
        let gamma = vec![1.0f32; dim];
        let mut config = LayerNormConfig::new(vec![dim]);
        config.elementwise_affine = true;
        let out = layer_norm(&data, &gamma, None, &config).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / dim as f32;
        let var: f32 = out.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
        prop_assert!(
            (var - 1.0).abs() < 0.2,
            "layernorm random variance = {}", var
        );
    }

    /// LayerNorm output length matches input length.
    #[test]
    fn layernorm_preserves_length(
        dim in 2usize..32,
    ) {
        let input: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let gamma = vec![1.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let out = layer_norm(&input, &gamma, None, &config).unwrap();
        prop_assert_eq!(out.len(), input.len());
    }
}

// ── 7. ReLU non-negative ────────────────────────────────────────────────────

proptest! {
    /// relu(x) >= 0 for all x.
    #[test]
    fn relu_non_negative(x in -1000.0f32..1000.0) {
        prop_assert!(relu(x) >= 0.0, "relu({}) < 0", x);
    }

    /// relu preserves positive values exactly.
    #[test]
    fn relu_preserves_positive(x in 0.0f32..1000.0) {
        prop_assert_eq!(relu(x), x);
    }

    /// relu maps negative values to zero.
    #[test]
    fn relu_maps_negative_to_zero(x in -1000.0f32..0.0) {
        prop_assert_eq!(relu(x), 0.0);
    }
}

// ── 8. ReLU idempotent ──────────────────────────────────────────────────────

proptest! {
    /// relu(relu(x)) == relu(x).
    #[test]
    fn relu_idempotent(x in -1000.0f32..1000.0) {
        let r = relu(x);
        prop_assert_eq!(relu(r), r);
    }

    /// relu applied N times is the same as one application.
    #[test]
    fn relu_multi_idempotent(x in -100.0f32..100.0) {
        let r1 = relu(x);
        let r2 = relu(relu(relu(r1)));
        prop_assert_eq!(r2, r1);
    }
}

// ── 9. Embedding lookup deterministic ───────────────────────────────────────

proptest! {
    /// Same index always returns the same vector.
    #[test]
    fn embedding_lookup_deterministic(
        vocab in 10usize..50,
        dim in 2usize..16,
        idx in 0u32..10,
    ) {
        let idx = idx % vocab as u32;
        let table: Vec<f32> = (0..(vocab * dim)).map(|i| i as f32 * 0.01).collect();
        let out1 = embedding_lookup(&table, &[idx], dim).unwrap();
        let out2 = embedding_lookup(&table, &[idx], dim).unwrap();
        prop_assert_eq!(&out1, &out2);
    }

    /// Repeated indices yield repeated vectors.
    #[test]
    fn embedding_lookup_repeated(
        vocab in 10usize..30,
        dim in 2usize..8,
        idx in 0u32..10,
    ) {
        let idx = idx % vocab as u32;
        let table: Vec<f32> = (0..(vocab * dim)).map(|i| i as f32 * 0.01).collect();
        let out = embedding_lookup(&table, &[idx, idx], dim).unwrap();
        prop_assert_eq!(&out[..dim], &out[dim..2 * dim]);
    }

    /// Embedding output length = num_indices * dim.
    #[test]
    fn embedding_output_length(
        vocab in 5usize..30,
        dim in 1usize..8,
        n in 1usize..10,
    ) {
        let table: Vec<f32> = (0..(vocab * dim)).map(|i| i as f32).collect();
        let indices: Vec<u32> = (0..n).map(|i| (i % vocab) as u32).collect();
        let out = embedding_lookup(&table, &indices, dim).unwrap();
        prop_assert_eq!(out.len(), n * dim);
    }

    /// Embedding lookup output values are finite.
    #[test]
    fn embedding_output_finite(
        vocab in 5usize..20,
        dim in 1usize..8,
    ) {
        let table: Vec<f32> = (0..(vocab * dim)).map(|i| i as f32 * 0.1).collect();
        let indices: Vec<u32> = (0..3).map(|i| (i % vocab) as u32).collect();
        let out = embedding_lookup(&table, &indices, dim).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.is_finite(), "embedding[{}] not finite", i);
        }
    }
}

// ── 10. Quantize-dequantize bounded error ───────────────────────────────────

proptest! {
    /// |dequant(quant(x)) - x| < epsilon for symmetric i8 quantization.
    #[test]
    fn quantize_dequantize_symmetric_bounded_error(
        data in proptest::collection::vec(-10.0f32..10.0, 4..32),
    ) {
        let (quantized, scale) = quantize_symmetric_i8(&data, 8);
        let reconstructed = dequantize_symmetric_i8(&quantized, scale);
        let abs_max = data.iter().copied().fold(0.0f32, |m, v| m.max(v.abs()));
        let epsilon = if abs_max > 0.0 { abs_max / 127.0 + 1e-5 } else { 1e-5 };
        for (i, (&orig, &recon)) in data.iter().zip(reconstructed.iter()).enumerate() {
            prop_assert!(
                (orig - recon).abs() < epsilon * 2.0,
                "symmetric error[{}]: |{} - {}| = {} >= {}",
                i, orig, recon, (orig - recon).abs(), epsilon * 2.0
            );
        }
    }

    /// |dequant(quant(x)) - x| bounded for asymmetric u8 quantization.
    #[test]
    fn quantize_dequantize_asymmetric_bounded_error(
        data in proptest::collection::vec(-10.0f32..10.0, 4..32),
    ) {
        let (quantized, scale, zero_point) = quantize_asymmetric_u8(&data);
        let reconstructed = dequantize_asymmetric_u8(&quantized, scale, zero_point);
        let range = {
            let min = data.iter().copied().fold(f32::INFINITY, f32::min);
            let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            max - min
        };
        let epsilon = if range > 0.0 { range / 255.0 + 1e-5 } else { 1e-5 };
        for (i, (&orig, &recon)) in data.iter().zip(reconstructed.iter()).enumerate() {
            prop_assert!(
                (orig - recon).abs() < epsilon * 2.0,
                "asymmetric error[{}]: |{} - {}| = {} >= {}",
                i, orig, recon, (orig - recon).abs(), epsilon * 2.0
            );
        }
    }

    /// Quantize-dequantize round-trip length is preserved.
    #[test]
    fn quantize_roundtrip_length(
        data in proptest::collection::vec(-5.0f32..5.0, 1..64),
    ) {
        let (q, scale) = quantize_symmetric_i8(&data, 8);
        let recon = dequantize_symmetric_i8(&q, scale);
        prop_assert_eq!(recon.len(), data.len());
    }
}

// ── 11. Residual identity ───────────────────────────────────────────────────

proptest! {
    /// residual(x, zeros) == x.
    #[test]
    fn residual_identity_with_zeros(
        data in proptest::collection::vec(-10.0f32..10.0, 1..64),
    ) {
        let original = data.clone();
        let zeros = vec![0.0f32; data.len()];
        let mut output = data;
        add_residual(&mut output, &zeros).unwrap();
        prop_assert_eq!(output, original);
    }

    /// residual(x, y) == x + y element-wise.
    #[test]
    fn residual_is_addition(
        n in 1usize..32,
    ) {
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let mut output = x.clone();
        add_residual(&mut output, &y).unwrap();
        for i in 0..n {
            prop_assert!(
                (output[i] - (x[i] + y[i])).abs() < 1e-6,
                "residual[{}]={} != {}+{}={}", i, output[i], x[i], y[i], x[i] + y[i]
            );
        }
    }

    /// Residual is commutative: x+y == y+x.
    #[test]
    fn residual_commutative(
        data in proptest::collection::vec(-5.0f32..5.0, 2..16),
    ) {
        let n = data.len() / 2;
        let x = data[..n].to_vec();
        let y = data[n..2 * n].to_vec();
        let mut out_xy = x.clone();
        add_residual(&mut out_xy, &y).unwrap();
        let mut out_yx = y.clone();
        add_residual(&mut out_yx, &x).unwrap();
        for i in 0..n {
            prop_assert!(
                (out_xy[i] - out_yx[i]).abs() < 1e-5,
                "x+y[{}]={} != y+x[{}]={}", i, out_xy[i], i, out_yx[i]
            );
        }
    }
}

// ── 12. Concat associativity ────────────────────────────────────────────────

proptest! {
    /// cat(cat(a,b),c) == cat(a,cat(b,c)) for 1-D tensors on axis 0.
    #[test]
    fn concat_associativity_1d(
        a_len in 1usize..8,
        b_len in 1usize..8,
        c_len in 1usize..8,
    ) {
        let a: Vec<f32> = (0..a_len).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..b_len).map(|i| (i + 100) as f32).collect();
        let c: Vec<f32> = (0..c_len).map(|i| (i + 200) as f32).collect();

        // cat(a, b)
        let ab = ConcatKernel::concat(
            &[a.as_slice(), b.as_slice()],
            &[&[a_len], &[b_len]],
            0,
        ).unwrap();
        // cat(cat(a,b), c)
        let ab_len = a_len + b_len;
        let abc_left = ConcatKernel::concat(
            &[ab.as_slice(), c.as_slice()],
            &[&[ab_len], &[c_len]],
            0,
        ).unwrap();

        // cat(b, c)
        let bc = ConcatKernel::concat(
            &[b.as_slice(), c.as_slice()],
            &[&[b_len], &[c_len]],
            0,
        ).unwrap();
        // cat(a, cat(b,c))
        let bc_len = b_len + c_len;
        let abc_right = ConcatKernel::concat(
            &[a.as_slice(), bc.as_slice()],
            &[&[a_len], &[bc_len]],
            0,
        ).unwrap();

        prop_assert_eq!(abc_left, abc_right);
    }

    /// Concat length is sum of input lengths.
    #[test]
    fn concat_length_sum(
        a_len in 1usize..16,
        b_len in 1usize..16,
    ) {
        let a: Vec<f32> = (0..a_len).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..b_len).map(|i| i as f32).collect();
        let result = ConcatKernel::concat(
            &[a.as_slice(), b.as_slice()],
            &[&[a_len], &[b_len]],
            0,
        ).unwrap();
        prop_assert_eq!(result.len(), a_len + b_len);
    }

    /// Concat preserves element values.
    #[test]
    fn concat_preserves_values(
        a_len in 1usize..8,
        b_len in 1usize..8,
    ) {
        let a: Vec<f32> = (0..a_len).map(|i| i as f32 * 1.1).collect();
        let b: Vec<f32> = (0..b_len).map(|i| i as f32 * 2.2).collect();
        let result = ConcatKernel::concat(
            &[a.as_slice(), b.as_slice()],
            &[&[a_len], &[b_len]],
            0,
        ).unwrap();
        for i in 0..a_len {
            prop_assert_eq!(result[i], a[i]);
        }
        for i in 0..b_len {
            prop_assert_eq!(result[a_len + i], b[i]);
        }
    }
}

// ── Additional concat properties ────────────────────────────────────────────

proptest! {
    /// Concatenating a single tensor returns it unchanged.
    #[test]
    fn concat_single_identity(n in 1usize..16) {
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let result = ConcatKernel::concat(&[a.as_slice()], &[&[n]], 0).unwrap();
        prop_assert_eq!(result, a);
    }
}

// ── 13. GeLU bounded ────────────────────────────────────────────────────────

proptest! {
    /// gelu(x) >= -0.17 for all x.
    #[test]
    fn gelu_lower_bounded(x in -100.0f32..100.0) {
        let g = gelu(x);
        prop_assert!(
            g >= -0.17,
            "gelu({}) = {} < -0.17", x, g
        );
    }

    /// gelu(x) ≈ x for large positive x.
    #[test]
    fn gelu_approx_identity_for_positive(x in 5.0f32..100.0) {
        let g = gelu(x);
        prop_assert!(
            (g - x).abs() < 0.01 * x.abs(),
            "gelu({}) = {}, expected ≈{}", x, g, x
        );
    }

    /// gelu(0) ≈ 0.
    #[test]
    fn gelu_zero(_dummy in 0u8..1) {
        prop_assert!(gelu(0.0).abs() < 1e-6);
    }

    /// gelu output is finite for bounded input.
    #[test]
    fn gelu_finite(x in -100.0f32..100.0) {
        prop_assert!(gelu(x).is_finite());
    }

    /// gelu is monotonically non-decreasing for x >= 0.
    #[test]
    fn gelu_monotone_positive(a in 0.0f32..50.0, b in 0.0f32..50.0) {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        prop_assert!(
            gelu(lo) <= gelu(hi) + 1e-6,
            "gelu({}) = {} > gelu({}) = {}", lo, gelu(lo), hi, gelu(hi)
        );
    }
}

// ── 14. Attention mask symmetry ─────────────────────────────────────────────

proptest! {
    /// Causal mask is lower-triangular (upper triangle is -inf).
    #[test]
    fn causal_mask_lower_triangular(seq_len in 2usize..16) {
        let mask = causal_mask(seq_len);
        for i in 0..seq_len {
            for j in 0..seq_len {
                let v = mask[i * seq_len + j];
                if j > i {
                    prop_assert_eq!(v, f32::NEG_INFINITY,
                        "mask[{},{}] should be -inf, got {}", i, j, v);
                } else {
                    prop_assert_eq!(v, 0.0,
                        "mask[{},{}] should be 0, got {}", i, j, v);
                }
            }
        }
    }

    /// Causal mask size is seq_len².
    #[test]
    fn causal_mask_size(seq_len in 1usize..32) {
        let mask = causal_mask(seq_len);
        prop_assert_eq!(mask.len(), seq_len * seq_len);
    }

    /// Causal mask diagonal is always 0 (self-attention allowed).
    #[test]
    fn causal_mask_diagonal_zero(seq_len in 1usize..32) {
        let mask = causal_mask(seq_len);
        for i in 0..seq_len {
            prop_assert_eq!(mask[i * seq_len + i], 0.0,
                "diagonal[{}] != 0", i);
        }
    }

    /// First row of causal mask: only [0,0] is 0, rest is -inf.
    #[test]
    fn causal_mask_first_row(seq_len in 2usize..16) {
        let mask = causal_mask(seq_len);
        prop_assert_eq!(mask[0], 0.0);
        for (j, &v) in mask.iter().enumerate().take(seq_len).skip(1) {
            prop_assert_eq!(v, f32::NEG_INFINITY,
                "first row[{}] should be -inf", j);
        }
    }
}
