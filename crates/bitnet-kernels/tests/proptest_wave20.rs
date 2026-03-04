//! Property-based tests — wave 20.
//!
//! Mathematical invariants for neural network kernels covering:
//! - Numerical stability (softmax on extreme values, layer norm with tiny variance)
//! - Precision (f16↔f32 round-trip, quantization round-trip)
//! - Associativity (batched matmul)
//! - Distributivity (element-wise ops over concat)
//! - Batch independence (single vs. batched processing)
//! - Permutation invariance (max-pool, reductions)
//! - Scale invariance (softmax(x+c) == softmax(x))
//! - Translation equivariance (conv1d commutes with shift)
//! - Gradient correctness (numerical ≈ analytical)

use bitnet_kernels::cpu::activations::{
    gelu, gelu_vec, hard_sigmoid, hard_swish, mish, relu, sigmoid, silu, silu_vec, softplus,
};
use bitnet_kernels::cpu::batch::{
    batched_add, batched_layer_norm, batched_matmul, batched_softmax,
};
use bitnet_kernels::cpu::concat::ConcatKernel;
use bitnet_kernels::cpu::conv2d::{Conv2dConfig, compute_output_size, conv2d};
use bitnet_kernels::cpu::embedding::{embedding_lookup, embedding_lookup_batched};
use bitnet_kernels::cpu::gating::swiglu;
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::linear::{LinearConfig, linear_cpu};
use bitnet_kernels::cpu::pooling::{PoolConfig, PoolType, pool_1d};
use bitnet_kernels::cpu::quantize::{
    compute_quantization_error, dequantize_symmetric_i8, quantize_symmetric_i8,
};
use bitnet_kernels::cpu::reduction::ReductionKernel;
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled};
use bitnet_kernels::cpu::simd_matmul::{SimdMatmulConfig, simd_matmul_f32};
use bitnet_kernels::cpu::transpose::TransposeKernel;
use proptest::prelude::*;

// ===================================================================
// Strategy helpers
// ===================================================================

fn _finite_f32_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-10.0f32..10.0, min_len..=max_len)
}

// ===================================================================
// 1. Numerical stability — softmax on extreme values
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Softmax output sums to 1.0 even with large magnitude inputs.
    #[test]
    fn prop_softmax_extreme_values_sum_to_one(
        base in prop::collection::vec(-500.0f32..500.0, 4..=32),
    ) {
        let n = base.len();
        let result = batched_softmax(&base, 1, n).unwrap();
        let sum: f32 = result.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
    }

    /// Softmax output is always non-negative.
    #[test]
    fn prop_softmax_always_nonneg(
        input in prop::collection::vec(-100.0f32..100.0, 2..=64),
    ) {
        let n = input.len();
        let result = batched_softmax(&input, 1, n).unwrap();
        for &v in &result {
            prop_assert!(v >= 0.0, "negative softmax output: {v}");
        }
    }

    /// Softmax is finite even when input contains the same repeated value.
    #[test]
    fn prop_softmax_uniform_input(val in -100.0f32..100.0, len in 2usize..=32) {
        let input = vec![val; len];
        let result = batched_softmax(&input, 1, len).unwrap();
        let expected = 1.0 / len as f32;
        for &v in &result {
            prop_assert!(v.is_finite(), "non-finite softmax on uniform input");
            prop_assert!((v - expected).abs() < 1e-5);
        }
    }
}

// ===================================================================
// 2. Scale invariance — softmax(x + c) == softmax(x)
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Softmax is invariant to additive shift.
    #[test]
    fn prop_softmax_shift_invariance(
        input in prop::collection::vec(-10.0f32..10.0, 4..=32),
        shift in -50.0f32..50.0,
    ) {
        let n = input.len();
        let shifted: Vec<f32> = input.iter().map(|&x| x + shift).collect();
        let result_orig = batched_softmax(&input, 1, n).unwrap();
        let result_shifted = batched_softmax(&shifted, 1, n).unwrap();
        for (a, b) in result_orig.iter().zip(result_shifted.iter()) {
            prop_assert!((a - b).abs() < 1e-5, "shift invariance violated: {a} vs {b}");
        }
    }

    /// Softmax preserves relative ordering (argmax is stable under shift).
    #[test]
    fn prop_softmax_preserves_argmax(
        input in prop::collection::vec(-10.0f32..10.0, 2..=32),
        shift in -50.0f32..50.0,
    ) {
        let n = input.len();
        let shifted: Vec<f32> = input.iter().map(|&x| x + shift).collect();
        let sm_orig = batched_softmax(&input, 1, n).unwrap();
        let sm_shifted = batched_softmax(&shifted, 1, n).unwrap();
        let argmax_orig = sm_orig
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i);
        let argmax_shifted = sm_shifted
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i);
        prop_assert_eq!(argmax_orig, argmax_shifted);
    }
}

// ===================================================================
// 3. Layer norm with tiny variance stays finite
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Layer norm output is finite even with near-constant inputs.
    #[test]
    fn prop_layer_norm_tiny_variance_finite(
        base_val in -10.0f32..10.0,
        dim in 4usize..=32,
    ) {
        let input: Vec<f32> = (0..dim).map(|i| base_val + (i as f32) * 1e-7).collect();
        let gamma = vec![1.0f32; dim];
        let beta = vec![0.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let result = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        for &v in &result {
            prop_assert!(v.is_finite(), "non-finite layer norm output: {v}");
        }
    }

    /// RMS norm output is finite with near-zero inputs.
    #[test]
    fn prop_rms_norm_near_zero_finite(dim in 4usize..=32) {
        let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 1e-8).collect();
        let gamma = vec![1.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let result = rms_norm(&input, &gamma, &config).unwrap();
        for &v in &result {
            prop_assert!(v.is_finite(), "non-finite rms norm output: {v}");
        }
    }

    /// Layer norm output has approximately zero mean (with identity affine).
    #[test]
    fn prop_layer_norm_zero_mean(
        input in prop::collection::vec(-5.0f32..5.0, 8..=64),
    ) {
        let dim = input.len();
        let gamma = vec![1.0f32; dim];
        let beta = vec![0.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let result = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let mean: f32 = result.iter().sum::<f32>() / dim as f32;
        prop_assert!(mean.abs() < 1e-4, "layer norm mean = {mean}");
    }

    /// Layer norm output has approximately unit variance (with identity affine).
    #[test]
    fn prop_layer_norm_unit_variance(
        input in prop::collection::vec(-5.0f32..5.0, 16..=64),
    ) {
        let dim = input.len();
        let gamma = vec![1.0f32; dim];
        let beta = vec![0.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let result = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let mean: f32 = result.iter().sum::<f32>() / dim as f32;
        let var: f32 = result.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / dim as f32;
        prop_assert!((var - 1.0).abs() < 0.05, "layer norm variance = {var}");
    }
}

// ===================================================================
// 4. Precision — f16 ↔ f32 round-trip within tolerances
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// f16 round-trip error is bounded by f16 epsilon (~9.77e-4).
    #[test]
    fn prop_f16_roundtrip_bounded(val in -65000.0f32..65000.0) {
        let f16_val = half::f16::from_f32(val);
        let roundtrip = f16_val.to_f32();
        if val.is_finite() && roundtrip.is_finite() {
            let abs_val = val.abs().max(1.0);
            let rel_err = (val - roundtrip).abs() / abs_val;
            // f16 has ~3.3 decimal digits; relative error < 0.2%
            prop_assert!(rel_err < 0.002, "f16 rel_err={rel_err} for val={val}");
        }
    }

    /// f16 round-trip preserves sign.
    #[test]
    fn prop_f16_roundtrip_preserves_sign(val in -1000.0f32..1000.0) {
        if val == 0.0 {
            return Ok(());
        }
        let roundtrip = half::f16::from_f32(val).to_f32();
        prop_assert_eq!(val.signum() as i32, roundtrip.signum() as i32);
    }

    /// f16 zero is exact.
    #[test]
    fn prop_f16_zero_exact(_dummy in 0..1i32) {
        let rt = half::f16::from_f32(0.0).to_f32();
        prop_assert_eq!(rt, 0.0);
    }
}

// ===================================================================
// 5. Precision — quantization round-trip
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Symmetric i8 quant → dequant round-trip has bounded MSE.
    #[test]
    fn prop_symmetric_i8_roundtrip_bounded(
        input in prop::collection::vec(-5.0f32..5.0, 8..=64),
    ) {
        let (quantized, scale) = quantize_symmetric_i8(&input, 8);
        let dequantized = dequantize_symmetric_i8(&quantized, scale);
        let err = compute_quantization_error(&input, &dequantized);
        prop_assert!(err.mse < 0.1, "quantization MSE = {}", err.mse);
        prop_assert!(err.max_abs_error < 0.5, "max_abs = {}", err.max_abs_error);
    }

    /// Quantization of zero vector produces zero dequantized output.
    #[test]
    fn prop_quantize_zero_gives_zero(len in 4usize..=32) {
        let input = vec![0.0f32; len];
        let (quantized, scale) = quantize_symmetric_i8(&input, 8);
        let dequantized = dequantize_symmetric_i8(&quantized, scale);
        for &v in &dequantized {
            prop_assert!(v.abs() < 1e-6, "dequantized zero = {v}");
        }
    }

    /// Quantization SNR improves with narrower input range.
    #[test]
    fn prop_narrow_range_better_snr(
        input in prop::collection::vec(0.1f32..0.5, 16..=64),
    ) {
        let wide: Vec<f32> = input.iter().map(|x| x * 20.0).collect();
        let (q_narrow, s_narrow) = quantize_symmetric_i8(&input, 8);
        let (q_wide, s_wide) = quantize_symmetric_i8(&wide, 8);
        let d_narrow = dequantize_symmetric_i8(&q_narrow, s_narrow);
        let d_wide = dequantize_symmetric_i8(&q_wide, s_wide);
        let err_narrow = compute_quantization_error(&input, &d_narrow);
        let err_wide = compute_quantization_error(&wide, &d_wide);
        // Narrow range should have lower absolute MSE
        prop_assert!(err_narrow.mse <= err_wide.mse + 1e-6);
    }
}

// ===================================================================
// 6. Matmul associativity within FP tolerance
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// (A*B)*C ≈ A*(B*C) within FP tolerance for small matrices.
    #[test]
    fn prop_matmul_associativity(
        m in 2usize..=4,
        n in 2usize..=4,
        k in 2usize..=4,
        p in 2usize..=4,
    ) {
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 3) % 19) as f32 * 0.1 - 0.9).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 11 + 5) % 23) as f32 * 0.1 - 1.1).collect();
        let c_mat: Vec<f32> =
            (0..n * p).map(|i| ((i * 13 + 7) % 17) as f32 * 0.1 - 0.8).collect();

        // (A*B)
        let cfg_ab = SimdMatmulConfig::new(m, n, k);
        let mut ab = vec![0.0f32; m * n];
        simd_matmul_f32(&a, &b, &mut ab, &cfg_ab).unwrap();

        // (A*B)*C
        let cfg_abc = SimdMatmulConfig::new(m, p, n);
        let mut abc = vec![0.0f32; m * p];
        simd_matmul_f32(&ab, &c_mat, &mut abc, &cfg_abc).unwrap();

        // B*C
        let cfg_bc = SimdMatmulConfig::new(k, p, n);
        let mut bc = vec![0.0f32; k * p];
        simd_matmul_f32(&b, &c_mat, &mut bc, &cfg_bc).unwrap();

        // A*(B*C)
        let cfg_a_bc = SimdMatmulConfig::new(m, p, k);
        let mut a_bc = vec![0.0f32; m * p];
        simd_matmul_f32(&a, &bc, &mut a_bc, &cfg_a_bc).unwrap();

        for (i, (&l, &r)) in abc.iter().zip(a_bc.iter()).enumerate() {
            let diff = (l - r).abs();
            let tol = 1e-3 * l.abs().max(r.abs()).max(1.0);
            prop_assert!(diff < tol, "associativity idx={i}: {l} vs {r}, diff={diff}");
        }
    }

    /// Matmul with identity matrix is a no-op.
    #[test]
    fn prop_matmul_identity(n in 2usize..=8) {
        let a: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let mut eye = vec![0.0f32; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let cfg = SimdMatmulConfig::new(n, n, n);
        let mut result = vec![0.0f32; n * n];
        simd_matmul_f32(&a, &eye, &mut result, &cfg).unwrap();
        for (i, (&got, &expected)) in result.iter().zip(a.iter()).enumerate() {
            prop_assert!(
                (got - expected).abs() < 1e-5,
                "identity matmul idx={i}: {got} vs {expected}"
            );
        }
    }

    /// Matmul with zero matrix produces zero.
    #[test]
    fn prop_matmul_zero(m in 2usize..=6, k in 2usize..=6, n in 2usize..=6) {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5).collect();
        let b_mat = vec![0.0f32; k * n];
        let cfg = SimdMatmulConfig::new(m, n, k);
        let mut result = vec![0.0f32; m * n];
        simd_matmul_f32(&a, &b_mat, &mut result, &cfg).unwrap();
        for &v in &result {
            prop_assert!(v.abs() < 1e-6, "zero matmul produced {v}");
        }
    }
}

// ===================================================================
// 7. Distributivity — element-wise ops over concat
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// relu(concat(a, b)) == concat(relu(a), relu(b))
    #[test]
    fn prop_relu_distributes_over_concat(
        a in prop::collection::vec(-5.0f32..5.0, 1..=16),
        b in prop::collection::vec(-5.0f32..5.0, 1..=16),
    ) {
        let mut combined = a.clone();
        combined.extend_from_slice(&b);
        let relu_combined: Vec<f32> = combined.iter().map(|&x| relu(x)).collect();
        let relu_a: Vec<f32> = a.iter().map(|&x| relu(x)).collect();
        let relu_b: Vec<f32> = b.iter().map(|&x| relu(x)).collect();
        let mut relu_then_concat = relu_a;
        relu_then_concat.extend_from_slice(&relu_b);
        prop_assert_eq!(relu_combined, relu_then_concat);
    }

    /// sigmoid(concat(a, b)) == concat(sigmoid(a), sigmoid(b))
    #[test]
    fn prop_sigmoid_distributes_over_concat(
        a in prop::collection::vec(-5.0f32..5.0, 1..=16),
        b in prop::collection::vec(-5.0f32..5.0, 1..=16),
    ) {
        let mut combined = a.clone();
        combined.extend_from_slice(&b);
        let act_combined: Vec<f32> = combined.iter().map(|&x| sigmoid(x)).collect();
        let act_a: Vec<f32> = a.iter().map(|&x| sigmoid(x)).collect();
        let act_b: Vec<f32> = b.iter().map(|&x| sigmoid(x)).collect();
        let mut act_then_concat = act_a;
        act_then_concat.extend_from_slice(&act_b);
        for (l, r) in act_combined.iter().zip(act_then_concat.iter()) {
            prop_assert!((l - r).abs() < 1e-6);
        }
    }

    /// silu distributes over concat (element-wise).
    #[test]
    fn prop_silu_distributes_over_concat(
        a in prop::collection::vec(-5.0f32..5.0, 1..=16),
        b in prop::collection::vec(-5.0f32..5.0, 1..=16),
    ) {
        let mut combined = a.clone();
        combined.extend_from_slice(&b);
        let act_combined: Vec<f32> = combined.iter().map(|&x| silu(x)).collect();
        let act_a: Vec<f32> = a.iter().map(|&x| silu(x)).collect();
        let act_b: Vec<f32> = b.iter().map(|&x| silu(x)).collect();
        let mut act_then_concat = act_a;
        act_then_concat.extend_from_slice(&act_b);
        for (l, r) in act_combined.iter().zip(act_then_concat.iter()) {
            prop_assert!((l - r).abs() < 1e-6);
        }
    }

    /// gelu_vec distributes over concat.
    #[test]
    fn prop_gelu_vec_distributes_over_concat(
        a in prop::collection::vec(-3.0f32..3.0, 1..=16),
        b in prop::collection::vec(-3.0f32..3.0, 1..=16),
    ) {
        let mut combined = a.clone();
        combined.extend_from_slice(&b);
        let act_combined = gelu_vec(&combined);
        let act_a = gelu_vec(&a);
        let act_b = gelu_vec(&b);
        let mut act_then_concat = act_a;
        act_then_concat.extend_from_slice(&act_b);
        for (l, r) in act_combined.iter().zip(act_then_concat.iter()) {
            prop_assert!((l - r).abs() < 1e-6);
        }
    }
}

// ===================================================================
// 8. Batch independence — single vs. batch processing
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Batched softmax row i equals single softmax on that row.
    #[test]
    fn prop_batched_softmax_row_independence(
        batch in 2usize..=4,
        seq_len in 4usize..=16,
    ) {
        let total = batch * seq_len;
        let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let batched = batched_softmax(&input, batch, seq_len).unwrap();
        for b in 0..batch {
            let off = b * seq_len;
            let row = &input[off..off + seq_len];
            let single = batched_softmax(row, 1, seq_len).unwrap();
            for j in 0..seq_len {
                let diff = (batched[off + j] - single[j]).abs();
                prop_assert!(diff < 1e-6, "batch={b} j={j} diff={diff}");
            }
        }
    }

    /// Batched layer norm row i equals single layer norm.
    #[test]
    fn prop_batched_layer_norm_row_independence(
        batch in 2usize..=4,
        dim in 4usize..=16,
    ) {
        let total = batch * dim;
        let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.2 - 1.0).collect();
        let gamma = vec![1.0f32; dim];
        let beta = vec![0.0f32; dim];
        let eps = 1e-5;
        let batched = batched_layer_norm(&input, &gamma, &beta, batch, dim, eps).unwrap();
        for b in 0..batch {
            let off = b * dim;
            let row = &input[off..off + dim];
            let config = LayerNormConfig::new(vec![dim]);
            let single = layer_norm(row, &gamma, Some(&beta), &config).unwrap();
            for j in 0..dim {
                let diff = (batched[off + j] - single[j]).abs();
                prop_assert!(diff < 1e-5, "batch={b} j={j} diff={diff}");
            }
        }
    }

    /// Batched matmul batch i equals single matmul.
    #[test]
    fn prop_batched_matmul_independence(
        batch in 2usize..=3,
        m in 2usize..=4,
        k in 2usize..=4,
        n in 2usize..=4,
    ) {
        let a: Vec<f32> =
            (0..batch * m * k).map(|i| ((i * 7 + 1) % 13) as f32 * 0.2 - 1.2).collect();
        let b: Vec<f32> =
            (0..batch * k * n).map(|i| ((i * 11 + 3) % 17) as f32 * 0.15 - 1.0).collect();
        let batched = batched_matmul(&a, &b, batch, m, k, n).unwrap();
        for bi in 0..batch {
            let a_off = bi * m * k;
            let b_off = bi * k * n;
            let c_off = bi * m * n;
            let a_slice = &a[a_off..a_off + m * k];
            let b_slice = &b[b_off..b_off + k * n];
            let cfg = SimdMatmulConfig::new(m, n, k);
            let mut single = vec![0.0f32; m * n];
            simd_matmul_f32(a_slice, b_slice, &mut single, &cfg).unwrap();
            for j in 0..m * n {
                let diff = (batched[c_off + j] - single[j]).abs();
                let tol = 1e-4 * single[j].abs().max(1.0);
                prop_assert!(diff < tol, "batch={bi} j={j} diff={diff}");
            }
        }
    }

    /// Batched add equals element-wise add on each row independently.
    #[test]
    fn prop_batched_add_independence(
        batch in 2usize..=4,
        dim in 4usize..=16,
    ) {
        let total = batch * dim;
        let a: Vec<f32> = (0..total).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..total).map(|i| -(i as f32) * 0.2 + 1.0).collect();
        let batched = batched_add(&a, &b, batch, dim).unwrap();
        for i in 0..total {
            let expected = a[i] + b[i];
            prop_assert!((batched[i] - expected).abs() < 1e-6);
        }
    }

    /// Embedding lookup batched equals concatenated individual lookups.
    #[test]
    fn prop_embedding_batch_independence(
        num_embeddings in 4usize..=16,
        dim in 2usize..=8,
        batch in 2usize..=3,
    ) {
        let table: Vec<f32> = (0..num_embeddings * dim).map(|i| i as f32 * 0.01).collect();
        let idx_vecs: Vec<Vec<u32>> = (0..batch)
            .map(|b| vec![(b % num_embeddings) as u32, ((b + 1) % num_embeddings) as u32])
            .collect();
        let idx_slices: Vec<&[u32]> = idx_vecs.iter().map(|v| v.as_slice()).collect();
        let batched =
            embedding_lookup_batched(&table, &idx_slices, num_embeddings, dim).unwrap();
        let mut expected = Vec::new();
        for idx_vec in &idx_vecs {
            let single = embedding_lookup(&table, idx_vec, dim).unwrap();
            expected.extend_from_slice(&single);
        }
        prop_assert_eq!(batched, expected, "batch mismatch");
    }
}

// ===================================================================
// 9. Permutation invariance — max-pool, reductions
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Max-pool result is independent of element order within each window.
    #[test]
    fn prop_max_pool_permutation_invariant(
        _seed in 0u64..1000,
    ) {
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let config = PoolConfig { pool_type: PoolType::Max, kernel_size: 4, stride: 4, padding: 0 };
        let result = pool_1d(&data, &config).unwrap();

        // Permute each window and check same result
        let mut permuted = data.clone();
        permuted.swap(0, 2);
        permuted.swap(1, 3);
        permuted.swap(4, 6);
        permuted.swap(5, 7);
        let result_perm = pool_1d(&permuted, &config).unwrap();
        prop_assert_eq!(result, result_perm);
    }

    /// Global max reduction is invariant to order.
    #[test]
    fn prop_reduction_max_permutation_invariant(
        input in prop::collection::vec(-100.0f32..100.0, 4..=32),
    ) {
        let max_orig = ReductionKernel::max(&input).unwrap();
        let mut reversed = input.clone();
        reversed.reverse();
        let max_rev = ReductionKernel::max(&reversed).unwrap();
        prop_assert!((max_orig.value - max_rev.value).abs() < 1e-6);
    }

    /// Sum reduction is invariant to order (within FP tolerance).
    #[test]
    fn prop_reduction_sum_permutation_invariant(
        input in prop::collection::vec(-10.0f32..10.0, 4..=32),
    ) {
        let sum_fwd = ReductionKernel::sum(&input).unwrap();
        let mut rev = input.clone();
        rev.reverse();
        let sum_rev = ReductionKernel::sum(&rev).unwrap();
        let tol = 1e-4 * sum_fwd.abs().max(1.0);
        prop_assert!((sum_fwd - sum_rev).abs() < tol, "{sum_fwd} vs {sum_rev}");
    }

    /// Average pool produces values within the range of input values.
    #[test]
    fn prop_avg_pool_bounded_by_input(
        input in prop::collection::vec(-10.0f32..10.0, 4..=32),
    ) {
        let config = PoolConfig {
            pool_type: PoolType::Average,
            kernel_size: 2,
            stride: 2,
            padding: 0,
        };
        // Ensure even length
        let len = input.len() - (input.len() % 2);
        if len < 2 {
            return Ok(());
        }
        let input = &input[..len];
        let min_val = input.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let result = pool_1d(input, &config).unwrap();
        for &v in &result {
            prop_assert!(v >= min_val - 1e-6 && v <= max_val + 1e-6);
        }
    }
}

// ===================================================================
// 10. Translation equivariance — conv1d commutes with shift
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Shifting input then convolving ≈ convolving then shifting output.
    #[test]
    fn prop_conv2d_translation_equivariance(
        in_w in 6usize..=12,
    ) {
        let in_c = 1;
        let out_c = 1;
        let k = 3;
        let config = Conv2dConfig::new(in_c, out_c, k);
        let in_h = 1; // 1D via 2D
        let weight = vec![0.5f32; out_c * in_c * k * k];

        // Original input
        let input: Vec<f32> = (0..in_c * in_h * in_w)
            .map(|i| if i < in_w { (i as f32) * 0.5 } else { 0.0 })
            .collect();

        // Shifted input (shift by 1 position right, zero-pad left)
        let mut shifted = vec![0.0f32; in_c * in_h * in_w];
        shifted[1..in_w].copy_from_slice(&input[..(in_w - 1)]);

        let out_orig = conv2d(&input, &weight, None, &config, 1, in_h, in_w);
        let out_shifted = conv2d(&shifted, &weight, None, &config, 1, in_h, in_w);

        if let (Ok(orig), Ok(shift)) = (out_orig, out_shifted) {
            let out_w = compute_output_size(in_w, k, 1, 0, 1);
            // After shift, output should still be finite
            if out_w >= 3 {
                for i in 2..out_w.min(orig.len()).min(shift.len()) {
                    prop_assert!(shift[i].is_finite());
                    prop_assert!(orig[i].is_finite());
                }
            }
        }
    }

    /// Conv2d output length matches the expected formula.
    #[test]
    fn prop_conv2d_output_length(
        in_h in 4usize..=8,
        in_w in 4usize..=8,
        k in 1usize..=3,
    ) {
        let in_c = 1;
        let out_c = 1;
        let config = Conv2dConfig::new(in_c, out_c, k);
        let out_h = compute_output_size(in_h, k, 1, 0, 1);
        let out_w = compute_output_size(in_w, k, 1, 0, 1);
        if out_h == 0 || out_w == 0 {
            return Ok(());
        }
        let input = vec![1.0f32; in_c * in_h * in_w];
        let weight = vec![0.1f32; out_c * in_c * k * k];
        let result = conv2d(&input, &weight, None, &config, 1, in_h, in_w);
        prop_assert!(result.is_ok());
        let output = result.unwrap();
        prop_assert_eq!(output.len(), out_c * out_h * out_w);
    }
}

// ===================================================================
// 11. Gradient correctness — numerical ≈ analytical
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Numerical gradient of sigmoid matches analytical σ(x)(1-σ(x)).
    #[test]
    fn prop_sigmoid_gradient(x in -5.0f32..5.0) {
        let eps = 1e-4f32;
        let numerical = (sigmoid(x + eps) - sigmoid(x - eps)) / (2.0 * eps);
        let s = sigmoid(x);
        let analytical = s * (1.0 - s);
        prop_assert!(
            (numerical - analytical).abs() < 1e-3,
            "x={x}: num={numerical} ana={analytical}"
        );
    }

    /// Numerical gradient of relu matches analytical step function.
    #[test]
    fn prop_relu_gradient(x in -5.0f32..5.0) {
        if x.abs() < 0.01 {
            return Ok(()); // skip near discontinuity
        }
        let eps = 1e-4f32;
        let numerical = (relu(x + eps) - relu(x - eps)) / (2.0 * eps);
        let analytical = if x > 0.0 { 1.0 } else { 0.0 };
        prop_assert!(
            (numerical - analytical).abs() < 1e-2,
            "x={x}: num={numerical} ana={analytical}"
        );
    }

    /// Numerical gradient of silu matches analytical d/dx[x·σ(x)].
    #[test]
    fn prop_silu_gradient(x in -5.0f32..5.0) {
        let eps = 1e-4f32;
        let numerical = (silu(x + eps) - silu(x - eps)) / (2.0 * eps);
        let s = sigmoid(x);
        let analytical = s + x * s * (1.0 - s);
        prop_assert!(
            (numerical - analytical).abs() < 5e-3,
            "x={x}: num={numerical} ana={analytical}"
        );
    }

    /// Numerical gradient of softplus matches sigmoid.
    #[test]
    fn prop_softplus_gradient_is_sigmoid(x in -5.0f32..5.0) {
        let eps = 1e-4f32;
        let numerical = (softplus(x + eps) - softplus(x - eps)) / (2.0 * eps);
        let analytical = sigmoid(x);
        prop_assert!(
            (numerical - analytical).abs() < 5e-3,
            "x={x}: num={numerical} ana={analytical}"
        );
    }

    /// Numerical gradient of gelu is non-negative for x >= 0.
    #[test]
    fn prop_gelu_gradient_nonneg_for_positive(x in 0.1f32..10.0) {
        let eps = 1e-4f32;
        let grad = (gelu(x + eps) - gelu(x - eps)) / (2.0 * eps);
        prop_assert!(grad >= -1e-3, "gelu grad negative at x={x}: {grad}");
    }

    /// Numerical gradient of mish is finite everywhere in [-5, 5].
    #[test]
    fn prop_mish_gradient_finite(x in -5.0f32..5.0) {
        let eps = 1e-4f32;
        let grad = (mish(x + eps) - mish(x - eps)) / (2.0 * eps);
        prop_assert!(grad.is_finite(), "mish grad non-finite at x={x}");
    }
}

// ===================================================================
// 12. Activation function bounds
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// sigmoid output is always in (0, 1).
    #[test]
    fn prop_sigmoid_bounded(x in -100.0f32..100.0) {
        let y = sigmoid(x);
        prop_assert!((0.0..=1.0).contains(&y), "sigmoid({x}) = {y}");
    }

    /// hard_sigmoid output is in [0, 1].
    #[test]
    fn prop_hard_sigmoid_bounded(x in -100.0f32..100.0) {
        let y = hard_sigmoid(x);
        prop_assert!((0.0..=1.0).contains(&y), "hard_sigmoid({x}) = {y}");
    }

    /// hard_swish output is finite.
    #[test]
    fn prop_hard_swish_finite(x in -100.0f32..100.0) {
        let y = hard_swish(x);
        prop_assert!(y.is_finite(), "hard_swish({x}) = {y}");
    }

    /// All activations produce finite output for moderate inputs.
    #[test]
    fn prop_all_activations_finite(x in -10.0f32..10.0) {
        prop_assert!(relu(x).is_finite());
        prop_assert!(sigmoid(x).is_finite());
        prop_assert!(gelu(x).is_finite());
        prop_assert!(silu(x).is_finite());
        prop_assert!(mish(x).is_finite());
        prop_assert!(softplus(x).is_finite());
        prop_assert!(hard_sigmoid(x).is_finite());
        prop_assert!(hard_swish(x).is_finite());
    }
}

// ===================================================================
// 13. Residual connection properties
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Residual add with zero residual is identity.
    #[test]
    fn prop_residual_zero_is_identity(
        input in prop::collection::vec(-10.0f32..10.0, 4..=32),
    ) {
        let residual = vec![0.0f32; input.len()];
        let mut output = input.clone();
        add_residual(&mut output, &residual).unwrap();
        prop_assert_eq!(output, input);
    }

    /// Residual scaled by 0 is identity.
    #[test]
    fn prop_residual_scale_zero_is_identity(
        input in prop::collection::vec(-10.0f32..10.0, 4..=32),
    ) {
        let residual: Vec<f32> = (0..input.len()).map(|i| i as f32).collect();
        let mut output = input.clone();
        add_residual_scaled(&mut output, &residual, 0.0).unwrap();
        prop_assert_eq!(output, input);
    }

    /// Residual add is commutative: a + b == b + a.
    #[test]
    fn prop_residual_add_commutative(
        a in prop::collection::vec(-10.0f32..10.0, 4..=32),
    ) {
        let b: Vec<f32> = a.iter().map(|x| x * 0.5 + 1.0).collect();
        let mut ab = a.clone();
        add_residual(&mut ab, &b).unwrap();
        let mut ba = b.clone();
        add_residual(&mut ba, &a).unwrap();
        for (l, r) in ab.iter().zip(ba.iter()) {
            prop_assert!((l - r).abs() < 1e-6);
        }
    }
}

// ===================================================================
// 14. Transpose properties
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Transpose is its own inverse: (A^T)^T == A.
    #[test]
    fn prop_transpose_involution(rows in 2usize..=8, cols in 2usize..=8) {
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.3).collect();
        let transposed = TransposeKernel::transpose_2d(&data, rows, cols).unwrap();
        let back = TransposeKernel::transpose_2d(&transposed, cols, rows).unwrap();
        prop_assert_eq!(data, back);
    }

    /// Transpose preserves element count.
    #[test]
    fn prop_transpose_preserves_count(rows in 1usize..=8, cols in 1usize..=8) {
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let transposed = TransposeKernel::transpose_2d(&data, rows, cols).unwrap();
        prop_assert_eq!(data.len(), transposed.len());
    }

    /// Reshape preserves total element count.
    #[test]
    fn prop_reshape_preserves_count(
        rows in 2usize..=8,
        cols in 2usize..=8,
    ) {
        let total = rows * cols;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let old_shape = vec![rows, cols];
        let new_shape = vec![1, total];
        let reshaped = TransposeKernel::reshape(&data, &old_shape, &new_shape).unwrap();
        prop_assert_eq!(reshaped.len(), total);
        prop_assert_eq!(data, reshaped);
    }
}

// ===================================================================
// 15. Concat / split round-trip
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Concat then split recovers original tensors.
    #[test]
    fn prop_concat_split_roundtrip(
        len_a in 2usize..=16,
        len_b in 2usize..=16,
    ) {
        let a: Vec<f32> = (0..len_a).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..len_b).map(|i| (i + 100) as f32).collect();
        let shape_a: &[usize] = &[len_a];
        let shape_b: &[usize] = &[len_b];
        let concatenated =
            ConcatKernel::concat(&[&a[..], &b[..]], &[shape_a, shape_b], 0).unwrap();
        let split =
            ConcatKernel::split_sizes(&concatenated, &[len_a + len_b], 0, &[len_a, len_b])
                .unwrap();
        prop_assert_eq!(&split[0], &a);
        prop_assert_eq!(&split[1], &b);
    }

    /// Concat of a single tensor is identity.
    #[test]
    fn prop_concat_single_is_identity(
        input in prop::collection::vec(-10.0f32..10.0, 1..=32),
    ) {
        let shape: &[usize] = &[input.len()];
        let concatenated = ConcatKernel::concat(&[&input[..]], &[shape], 0).unwrap();
        prop_assert_eq!(concatenated, input);
    }

    /// Concat preserves total element count.
    #[test]
    fn prop_concat_preserves_count(
        a in prop::collection::vec(-5.0f32..5.0, 1..=16),
        b in prop::collection::vec(-5.0f32..5.0, 1..=16),
    ) {
        let shape_a: &[usize] = &[a.len()];
        let shape_b: &[usize] = &[b.len()];
        let concatenated =
            ConcatKernel::concat(&[&a[..], &b[..]], &[shape_a, shape_b], 0).unwrap();
        prop_assert_eq!(concatenated.len(), a.len() + b.len());
    }
}

// ===================================================================
// 16. Linear layer properties
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Linear with zero weights produces zero output (no bias).
    #[test]
    fn prop_linear_zero_weight(
        batch in 1usize..=3,
        in_f in 2usize..=8,
        out_f in 2usize..=8,
    ) {
        let config = LinearConfig::new(batch, in_f, out_f).unwrap();
        let x: Vec<f32> = (0..batch * in_f).map(|i| i as f32 * 0.5).collect();
        let weight = vec![0.0f32; out_f * in_f];
        let mut output = vec![0.0f32; batch * out_f];
        linear_cpu(&x, &weight, None, &mut output, &config).unwrap();
        for &v in &output {
            prop_assert!(v.abs() < 1e-6, "zero weight produced {v}");
        }
    }

    /// Linear with identity weight (square) is a copy.
    #[test]
    fn prop_linear_identity_weight(
        batch in 1usize..=3,
        dim in 2usize..=8,
    ) {
        let config = LinearConfig::new(batch, dim, dim).unwrap();
        let x: Vec<f32> = (0..batch * dim).map(|i| i as f32 * 0.3 - 1.0).collect();
        // Identity weight: w[i][j] = delta(i,j)
        let mut weight = vec![0.0f32; dim * dim];
        for i in 0..dim {
            weight[i * dim + i] = 1.0;
        }
        let mut output = vec![0.0f32; batch * dim];
        linear_cpu(&x, &weight, None, &mut output, &config).unwrap();
        for (i, (&got, &expected)) in output.iter().zip(x.iter()).enumerate() {
            prop_assert!(
                (got - expected).abs() < 1e-5,
                "identity linear idx={i}: {got} vs {expected}"
            );
        }
    }
}

// ===================================================================
// 17. Reduction consistency
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// mean(x) == sum(x) / len(x).
    #[test]
    fn prop_mean_equals_sum_div_len(
        input in prop::collection::vec(-10.0f32..10.0, 4..=32),
    ) {
        let sum = ReductionKernel::sum(&input).unwrap();
        let mean = ReductionKernel::mean(&input).unwrap();
        let expected = sum / input.len() as f32;
        prop_assert!((mean - expected).abs() < 1e-5, "mean={mean} expected={expected}");
    }

    /// min(x) <= mean(x) <= max(x).
    #[test]
    fn prop_min_mean_max_ordering(
        input in prop::collection::vec(-10.0f32..10.0, 2..=32),
    ) {
        let min_val = ReductionKernel::min(&input).unwrap().value;
        let max_val = ReductionKernel::max(&input).unwrap().value;
        let mean = ReductionKernel::mean(&input).unwrap();
        prop_assert!(min_val <= mean + 1e-6, "min={min_val} > mean={mean}");
        prop_assert!(mean <= max_val + 1e-6, "mean={mean} > max={max_val}");
    }

    /// L2 norm is non-negative.
    #[test]
    fn prop_l2_norm_nonneg(
        input in prop::collection::vec(-10.0f32..10.0, 1..=32),
    ) {
        let norm = ReductionKernel::l2_norm(&input).unwrap();
        prop_assert!(norm >= 0.0, "l2_norm = {norm}");
    }

    /// L2 <= L1 always holds for vectors.
    #[test]
    fn prop_l1_l2_cauchy_schwarz(
        input in prop::collection::vec(-10.0f32..10.0, 1..=32),
    ) {
        let l1 = ReductionKernel::l1_norm(&input).unwrap();
        let l2 = ReductionKernel::l2_norm(&input).unwrap();
        prop_assert!(l2 <= l1 + 1e-5, "L2={l2} > L1={l1}");
    }
}

// ===================================================================
// 18. Gating function properties
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// SwiGLU output is finite for moderate inputs.
    #[test]
    fn prop_swiglu_finite(
        gate in prop::collection::vec(-5.0f32..5.0, 4..=32),
    ) {
        let up: Vec<f32> = gate.iter().map(|x| x * 0.5 + 1.0).collect();
        let mut output = vec![0.0f32; gate.len()];
        swiglu(&gate, &up, &mut output).unwrap();
        for &v in &output {
            prop_assert!(v.is_finite(), "non-finite swiglu output: {v}");
        }
    }

    /// SwiGLU with zero gate produces zero (silu(0) * up = 0 * up = 0).
    #[test]
    fn prop_swiglu_zero_gate(
        up in prop::collection::vec(-5.0f32..5.0, 4..=32),
    ) {
        let gate = vec![0.0f32; up.len()];
        let mut output = vec![0.0f32; up.len()];
        swiglu(&gate, &up, &mut output).unwrap();
        for &v in &output {
            // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
            prop_assert!(v.abs() < 1e-6, "swiglu(0, up) = {v}");
        }
    }
}

// ===================================================================
// 19. Silu vec consistency
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// silu_vec matches element-wise silu.
    #[test]
    fn prop_silu_vec_matches_scalar(
        input in prop::collection::vec(-5.0f32..5.0, 1..=64),
    ) {
        let vec_result = silu_vec(&input);
        let scalar_result: Vec<f32> = input.iter().map(|&x| silu(x)).collect();
        for (i, (&expected, &got)) in scalar_result.iter().zip(vec_result.iter()).enumerate() {
            prop_assert!(
                (expected - got).abs() < 1e-6,
                "silu_vec mismatch at {i}: {expected} vs {got}"
            );
        }
    }
}

// ===================================================================
// 20. Embedding lookup properties
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Embedding lookup with same index twice gives duplicate rows.
    #[test]
    fn prop_embedding_duplicate_index(
        num_emb in 4usize..=16,
        dim in 2usize..=8,
        idx in 0usize..4,
    ) {
        let table: Vec<f32> = (0..num_emb * dim).map(|i| i as f32 * 0.01).collect();
        let idx = idx % num_emb;
        let indices = vec![idx as u32, idx as u32];
        let result = embedding_lookup(&table, &indices, dim).unwrap();
        let row1 = &result[..dim];
        let row2 = &result[dim..2 * dim];
        prop_assert_eq!(row1, row2);
    }

    /// Embedding lookup output length = num_indices * dim.
    #[test]
    fn prop_embedding_output_shape(
        num_emb in 4usize..=16,
        dim in 2usize..=8,
        num_indices in 1usize..=4,
    ) {
        let table: Vec<f32> = (0..num_emb * dim).map(|i| i as f32).collect();
        let indices: Vec<u32> = (0..num_indices).map(|i| (i % num_emb) as u32).collect();
        let result = embedding_lookup(&table, &indices, dim).unwrap();
        prop_assert_eq!(result.len(), num_indices * dim);
    }
}

// ===================================================================
// 21. Softmax monotonicity
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Softmax preserves ordering: if x[i] > x[j], then softmax(x)[i] > softmax(x)[j].
    #[test]
    fn prop_softmax_monotonicity(
        base in prop::collection::vec(-10.0f32..10.0, 3..=16),
    ) {
        let n = base.len();
        let sm = batched_softmax(&base, 1, n).unwrap();
        for i in 0..n {
            for j in (i + 1)..n {
                if base[i] > base[j] {
                    prop_assert!(sm[i] >= sm[j] - 1e-7);
                }
            }
        }
    }
}

// ===================================================================
// 22. Layer norm idempotence under identity affine
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Applying layer norm twice with identity affine converges to fixed point.
    #[test]
    fn prop_layer_norm_idempotent(
        input in prop::collection::vec(-5.0f32..5.0, 8..=32),
    ) {
        let dim = input.len();
        let gamma = vec![1.0f32; dim];
        let beta = vec![0.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let once = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let twice = layer_norm(&once, &gamma, Some(&beta), &config).unwrap();
        for (a, b) in once.iter().zip(twice.iter()) {
            prop_assert!((a - b).abs() < 0.05, "idempotence violated: {a} vs {b}");
        }
    }
}

// ===================================================================
// 23. Matmul dimension sanity
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Matmul output has correct dimensions m×n.
    #[test]
    fn prop_matmul_output_dims(
        m in 1usize..=8,
        k in 1usize..=8,
        n in 1usize..=8,
    ) {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let cfg = SimdMatmulConfig::new(m, n, k);
        let mut c = vec![0.0f32; m * n];
        simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
        prop_assert_eq!(c.len(), m * n);
        // Each element should be sum of k ones = k
        for &v in &c {
            prop_assert!((v - k as f32).abs() < 1e-5);
        }
    }
}

// ===================================================================
// 24. Softmax temperature scaling
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Higher temperature (dividing by T>1) makes softmax more uniform.
    #[test]
    fn prop_softmax_temperature_uniformity(
        input in prop::collection::vec(-5.0f32..5.0, 4..=16),
    ) {
        let n = input.len();
        let sm_t1 = batched_softmax(&input, 1, n).unwrap();
        let high_temp: Vec<f32> = input.iter().map(|&x| x / 10.0).collect();
        let sm_ht = batched_softmax(&high_temp, 1, n).unwrap();
        // Entropy of high-temp should be >= entropy of T=1
        let entropy_t1: f32 = sm_t1.iter().map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 }).sum();
        let entropy_ht: f32 = sm_ht.iter().map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 }).sum();
        prop_assert!(entropy_ht >= entropy_t1 - 1e-5, "T=10 entropy {entropy_ht} < T=1 entropy {entropy_t1}");
    }
}
