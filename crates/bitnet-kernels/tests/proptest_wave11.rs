//! Wave 11 property tests: CPU kernel invariants for batch normalization,
//! scatter/gather, loss functions, pooling, transpose, and shaped reduction.
//!
//! Key invariants tested (20+ properties):
//! - Batch norm: output shape preservation, running mean convergence,
//!   momentum bounds, eval vs train mode consistency
//! - Scatter/gather: scatter then gather roundtrip, index bounds checking,
//!   gather output shape
//! - Loss functions: cross_entropy non-negative, MSE zero for identical
//!   inputs, loss symmetry
//! - Pooling: max_pool output bounds, avg_pool conservation, adaptive_pool
//!   target shape
//! - Transpose: double transpose = identity, shape transformation correctness
//! - Shaped reduction: sum/mean/min/max consistency, reduction dimension removal

use bitnet_kernels::cpu::batch_norm::{BatchNormConfig, batch_norm_forward, batch_norm_inference};
use bitnet_kernels::cpu::loss::{
    LossReduction, cosine_similarity_loss, cross_entropy_loss, l1_loss, mse_loss,
};
use bitnet_kernels::cpu::pooling::{PoolConfig, PoolType, adaptive_avg_pool_1d, pool_1d};
use bitnet_kernels::cpu::scatter_gather::{gather_1d, scatter_1d, scatter_add};
use bitnet_kernels::cpu::transpose::{TransposeConfig, reshape, transpose_2d, transpose_nd};
use bitnet_kernels::reduction::ReductionOp;
use bitnet_kernels::shaped_reduction::{ShapedReductionConfig, reduction_output_shape};
use proptest::prelude::*;

// -------------------------------------------------------------------
// Strategy helpers
// -------------------------------------------------------------------

/// Non-empty f32 vector with finite values in [-100, 100].
fn finite_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-100.0f32..100.0f32, 1..=max_len)
}

/// Non-empty f32 vector with small positive values suitable for logits.
fn small_positive_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(0.01f32..10.0f32, 1..=max_len)
}

// ===================================================================
// 1. Batch normalization: shape, running stats, mode consistency
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// batch_norm_forward output has the same length as the input.
    #[test]
    fn prop_batchnorm_output_shape_preserved(
        num_features in 1usize..=8,
        batch_size in 2usize..=16,
    ) {
        let n = batch_size * num_features;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 5.0).collect();
        let gamma = vec![1.0f32; num_features];
        let beta = vec![0.0f32; num_features];
        let running_mean = vec![0.0f32; num_features];
        let running_var = vec![1.0f32; num_features];
        let config = BatchNormConfig {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            training: true,
        };

        let (output, new_mean, new_var) =
            batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &config)
                .unwrap();

        prop_assert_eq!(output.len(), n, "output length must match input");
        prop_assert_eq!(new_mean.len(), num_features, "running_mean length");
        prop_assert_eq!(new_var.len(), num_features, "running_var length");
    }

    /// Running mean moves towards the batch mean after one forward pass.
    #[test]
    fn prop_batchnorm_running_mean_convergence(
        num_features in 1usize..=4,
        batch_size in 4usize..=16,
    ) {
        let n = batch_size * num_features;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 1.0).collect();
        let gamma = vec![1.0f32; num_features];
        let beta = vec![0.0f32; num_features];
        let running_mean = vec![0.0f32; num_features];
        let running_var = vec![1.0f32; num_features];
        let momentum = 0.1f32;
        let config = BatchNormConfig {
            num_features,
            eps: 1e-5,
            momentum,
            training: true,
        };

        let (_, new_mean, _) =
            batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &config)
                .unwrap();

        // Compute batch mean per channel.
        for ch in 0..num_features {
            let batch_mean: f32 =
                (0..batch_size).map(|b| input[b * num_features + ch]).sum::<f32>()
                    / batch_size as f32;
            // new_mean should be between old (0.0) and batch_mean.
            let expected = (1.0 - momentum) * 0.0 + momentum * batch_mean;
            prop_assert!(
                (new_mean[ch] - expected).abs() < 1e-3,
                "ch {ch}: new_mean={}, expected={expected}",
                new_mean[ch]
            );
        }
    }

    /// Updated running variance is non-negative for all channels.
    #[test]
    fn prop_batchnorm_running_var_nonneg(
        num_features in 1usize..=4,
        batch_size in 2usize..=8,
    ) {
        let n = batch_size * num_features;
        let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.3).sin() * 5.0).collect();
        let gamma = vec![1.0f32; num_features];
        let beta = vec![0.0f32; num_features];
        let running_mean = vec![0.0f32; num_features];
        let running_var = vec![1.0f32; num_features];
        let config = BatchNormConfig {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            training: true,
        };

        let (_, _, new_var) =
            batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &config)
                .unwrap();

        for (ch, &v) in new_var.iter().enumerate() {
            prop_assert!(v >= 0.0, "running_var[{ch}]={v} is negative");
        }
    }

    /// Inference mode using running stats matches the expected formula.
    #[test]
    fn prop_batchnorm_inference_deterministic(
        num_features in 1usize..=4,
        batch_size in 2usize..=8,
    ) {
        let n = batch_size * num_features;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.2).collect();
        let gamma = vec![1.0f32; num_features];
        let beta = vec![0.0f32; num_features];
        let running_mean = vec![0.5f32; num_features];
        let running_var = vec![1.0f32; num_features];
        let eps = 1e-5f32;

        let output = batch_norm_inference(
            &input, &gamma, &beta, &running_mean, &running_var, eps,
        )
        .unwrap();

        prop_assert_eq!(output.len(), n);
        for b in 0..batch_size {
            for ch in 0..num_features {
                let idx = b * num_features + ch;
                let expected =
                    (input[idx] - running_mean[ch]) / (running_var[ch] + eps).sqrt();
                prop_assert!(
                    (output[idx] - expected).abs() < 1e-4,
                    "inference mismatch at [{b},{ch}]: got={}, expected={expected}",
                    output[idx]
                );
            }
        }
    }
}

// ===================================================================
// 2. Scatter/gather: roundtrip, bounds, output shape
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// scatter_1d then gather_1d with identity indices recovers the values.
    #[test]
    fn prop_scatter_gather_1d_roundtrip(n in 1usize..=64) {
        let values: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.7).collect();
        let indices: Vec<usize> = (0..n).collect();
        let mut data = vec![0.0f32; n];

        scatter_1d(&mut data, &indices, &values).unwrap();
        let gathered = gather_1d(&data, &indices).unwrap();

        prop_assert_eq!(&values, &gathered, "scatter→gather roundtrip failed");
    }

    /// gather_1d with out-of-bounds index returns an error.
    #[test]
    fn prop_gather_1d_oob_error(n in 1usize..=32) {
        let data = vec![1.0f32; n];
        let indices = vec![n]; // one past the last valid index
        let result = gather_1d(&data, &indices);
        prop_assert!(result.is_err(), "expected error for OOB index {n}");
    }

    /// gather_1d output length equals the number of indices.
    #[test]
    fn prop_gather_1d_output_shape(
        n in 2usize..=32,
        n_select in 1usize..=16,
    ) {
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        // Select indices within bounds.
        let indices: Vec<usize> = (0..n_select).map(|i| i % n).collect();
        let gathered = gather_1d(&data, &indices).unwrap();
        prop_assert_eq!(
            gathered.len(),
            n_select,
            "gather output length must match index count"
        );
    }

    /// scatter_add accumulates correctly at the same index.
    #[test]
    fn prop_scatter_add_accumulates(
        n in 2usize..=32,
        n_writes in 2usize..=8,
    ) {
        let mut data = vec![0.0f32; n];
        let target_idx = 0usize;
        let indices = vec![target_idx; n_writes];
        let values: Vec<f32> = (0..n_writes).map(|i| (i as f32) + 1.0).collect();

        scatter_add(&mut data, &indices, &values).unwrap();

        let expected: f32 = values.iter().sum();
        let tol = expected.abs() * 1e-5 + 1e-5;
        prop_assert!(
            (data[target_idx] - expected).abs() < tol,
            "scatter_add: got={}, expected={expected}",
            data[target_idx]
        );
    }
}

// ===================================================================
// 3. Loss functions: non-negativity, zero-on-identical, symmetry
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Cross-entropy loss is non-negative for valid inputs.
    #[test]
    fn prop_cross_entropy_nonnegative(
        num_classes in 2usize..=8,
        batch_size in 1usize..=8,
    ) {
        let logits: Vec<f32> = (0..batch_size * num_classes)
            .map(|i| ((i as f32) * 0.7).sin() * 3.0)
            .collect();
        let targets: Vec<usize> = (0..batch_size).map(|i| i % num_classes).collect();

        let (loss, per_sample) =
            cross_entropy_loss(&logits, &targets, num_classes, LossReduction::Mean).unwrap();

        prop_assert!(loss >= -1e-6, "cross_entropy mean loss={loss} is negative");
        for (i, &l) in per_sample.iter().enumerate() {
            prop_assert!(l >= -1e-6, "per_sample[{i}]={l} is negative");
        }
    }

    /// MSE loss is exactly zero when predictions equal targets.
    #[test]
    fn prop_mse_zero_for_identical(input in finite_f32_vec(64)) {
        let loss = mse_loss(&input, &input, LossReduction::Mean).unwrap();
        prop_assert!(
            loss.abs() < 1e-6,
            "MSE of identical inputs should be ~0, got {loss}"
        );
    }

    /// MSE loss is symmetric: mse(a, b) == mse(b, a).
    #[test]
    fn prop_mse_symmetric(
        a in prop::collection::vec(-50.0f32..50.0f32, 1..=64),
    ) {
        let b: Vec<f32> = a.iter().map(|&x| x + 0.1).collect();
        let loss_ab = mse_loss(&a, &b, LossReduction::Mean).unwrap();
        let loss_ba = mse_loss(&b, &a, LossReduction::Mean).unwrap();
        prop_assert!(
            (loss_ab - loss_ba).abs() < 1e-4,
            "MSE not symmetric: {loss_ab} vs {loss_ba}"
        );
    }

    /// L1 loss is symmetric: l1(a, b) == l1(b, a).
    #[test]
    fn prop_l1_symmetric(
        a in prop::collection::vec(-50.0f32..50.0f32, 1..=64),
    ) {
        let b: Vec<f32> = a.iter().map(|&x| x + 0.5).collect();
        let loss_ab = l1_loss(&a, &b, LossReduction::Mean).unwrap();
        let loss_ba = l1_loss(&b, &a, LossReduction::Mean).unwrap();
        prop_assert!(
            (loss_ab - loss_ba).abs() < 1e-4,
            "L1 not symmetric: {loss_ab} vs {loss_ba}"
        );
    }

    /// Cosine similarity loss of a vector with itself is ~0.0 (loss = 1 - sim).
    #[test]
    fn prop_cosine_self_similarity_loss_zero(input in small_positive_vec(64)) {
        let loss = cosine_similarity_loss(&input, &input).unwrap();
        prop_assert!(
            loss.abs() < 1e-4,
            "self cosine similarity loss={loss}, expected ~0.0"
        );
    }

    /// MSE loss is non-negative for all inputs.
    #[test]
    fn prop_mse_nonnegative(
        a in prop::collection::vec(-50.0f32..50.0f32, 1..=64),
        b in prop::collection::vec(-50.0f32..50.0f32, 1..=64),
    ) {
        let len = a.len().min(b.len());
        let loss = mse_loss(&a[..len], &b[..len], LossReduction::Mean).unwrap();
        prop_assert!(loss >= -1e-6, "MSE loss={loss} is negative");
    }
}

// ===================================================================
// 4. Pooling: output bounds, conservation, adaptive target shape
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Max pool output values are bounded by input min and max.
    #[test]
    fn prop_max_pool_output_bounds(
        input in prop::collection::vec(-50.0f32..50.0f32, 4..=128),
        kernel_size in 2usize..=4,
    ) {
        prop_assume!(input.len() >= kernel_size);
        let config = PoolConfig {
            pool_type: PoolType::Max,
            kernel_size,
            stride: 1,
            padding: 0,
        };
        let output = pool_1d(&input, &config).unwrap();
        let input_max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let input_min = input.iter().copied().fold(f32::INFINITY, f32::min);
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(
                v >= input_min - 1e-6 && v <= input_max + 1e-6,
                "max_pool[{i}]={v} out of [{input_min}, {input_max}]"
            );
        }
    }

    /// Average pool with kernel_size=1 stride=1 preserves the input exactly.
    #[test]
    fn prop_avg_pool_identity(input in finite_f32_vec(64)) {
        let config = PoolConfig {
            pool_type: PoolType::Average,
            kernel_size: 1,
            stride: 1,
            padding: 0,
        };
        let output = pool_1d(&input, &config).unwrap();
        prop_assert_eq!(input.len(), output.len());
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            prop_assert!(
                (inp - out).abs() < 1e-6,
                "avg_pool identity mismatch at [{i}]: {inp} vs {out}"
            );
        }
    }

    /// Adaptive avg pool produces exactly the requested output size.
    #[test]
    fn prop_adaptive_pool_target_shape(
        input_len in 4usize..=128,
        output_size in 1usize..=4,
    ) {
        prop_assume!(output_size <= input_len);
        let input: Vec<f32> = (0..input_len).map(|i| i as f32 * 0.1).collect();
        let output = adaptive_avg_pool_1d(&input, output_size).unwrap();
        prop_assert_eq!(
            output.len(),
            output_size,
            "adaptive pool output length mismatch"
        );
    }

    /// Global max pool returns the actual maximum of the input.
    #[test]
    fn prop_global_max_pool_correct(input in finite_f32_vec(128)) {
        let config = PoolConfig {
            pool_type: PoolType::GlobalMax,
            kernel_size: 0,
            stride: 0,
            padding: 0,
        };
        let output = pool_1d(&input, &config).unwrap();
        prop_assert_eq!(output.len(), 1);
        let expected = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        prop_assert!(
            (output[0] - expected).abs() < 1e-6,
            "global max: got={}, expected={expected}",
            output[0]
        );
    }

    /// Max pool output values are always drawn from the input.
    #[test]
    fn prop_max_pool_values_from_input(
        input in prop::collection::vec(-50.0f32..50.0f32, 4..=64),
        kernel_size in 2usize..=3,
    ) {
        prop_assume!(input.len() >= kernel_size);
        let config = PoolConfig {
            pool_type: PoolType::Max,
            kernel_size,
            stride: 1,
            padding: 0,
        };
        let output = pool_1d(&input, &config).unwrap();
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(
                input.iter().any(|&x| (x - v).abs() < 1e-6),
                "max_pool[{i}]={v} not found in input"
            );
        }
    }
}

// ===================================================================
// 5. Transpose: double transpose = identity, shape correctness
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// transpose_2d(transpose_2d(x)) == x (involution).
    #[test]
    fn prop_transpose_2d_double_identity(
        rows in 1usize..=32,
        cols in 1usize..=32,
    ) {
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.3).collect();
        let t1 = transpose_2d(&data, rows, cols);
        let t2 = transpose_2d(&t1, cols, rows);
        prop_assert_eq!(&data, &t2, "double transpose != identity");
    }

    /// transpose_2d output has correct length (rows * cols).
    #[test]
    fn prop_transpose_2d_shape(
        rows in 1usize..=32,
        cols in 1usize..=32,
    ) {
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let transposed = transpose_2d(&data, rows, cols);
        prop_assert_eq!(transposed.len(), rows * cols);
    }

    /// transpose_nd with identity permutation is a no-op.
    #[test]
    fn prop_transpose_nd_identity(
        dim_a in 1usize..=8,
        dim_b in 1usize..=8,
        dim_c in 1usize..=4,
    ) {
        let shape = [dim_a, dim_b, dim_c];
        let total = dim_a * dim_b * dim_c;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let result = transpose_nd(&data, &shape, &[0, 1, 2]);
        prop_assert_eq!(&data, &result, "identity permutation should be no-op");
    }

    /// transpose_nd roundtrip with inverse permutation recovers original.
    #[test]
    fn prop_transpose_nd_roundtrip(
        dim_a in 1usize..=6,
        dim_b in 1usize..=6,
        dim_c in 1usize..=4,
    ) {
        let shape = [dim_a, dim_b, dim_c];
        let total = dim_a * dim_b * dim_c;
        let data: Vec<f32> = (0..total).map(|i| i as f32 * 0.1).collect();
        let perm = [2, 0, 1];
        let inv = [1, 2, 0];
        let t1 = transpose_nd(&data, &shape, &perm);
        let mid_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        let t2 = transpose_nd(&t1, &mid_shape, &inv);
        prop_assert_eq!(&data, &t2, "transpose roundtrip failed");
    }

    /// TransposeConfig::output_shape correctly transforms dimensions.
    #[test]
    fn prop_transpose_config_output_shape(
        dim_a in 1usize..=16,
        dim_b in 1usize..=16,
    ) {
        let config = TransposeConfig {
            shape: vec![dim_a, dim_b],
            permutation: vec![1, 0],
        };
        let out_shape = config.output_shape();
        prop_assert_eq!(out_shape, vec![dim_b, dim_a]);
    }

    /// reshape preserves data when total element count matches.
    #[test]
    fn prop_reshape_preserves_data(
        rows in 1usize..=16,
        cols in 1usize..=16,
    ) {
        let total = rows * cols;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let reshaped = reshape(&data, &[rows, cols], &[total]).unwrap();
        prop_assert_eq!(&data, &reshaped, "reshape should preserve data");
    }
}

// ===================================================================
// 6. Shaped reduction: consistency and dimension removal
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Global sum reduction matches iterative sum.
    #[test]
    fn prop_shaped_reduce_sum_matches_iter(input in finite_f32_vec(128)) {
        let config = ShapedReductionConfig::global(ReductionOp::Sum);
        let result =
            bitnet_kernels::shaped_reduction::reduce_f32(&input, &[input.len()], &config)
                .unwrap();
        prop_assert_eq!(result.len(), 1);
        let expected: f32 = input.iter().sum();
        let tol = input.len() as f32 * 1e-3;
        prop_assert!(
            (result[0] - expected).abs() <= tol,
            "shaped sum={}, iter sum={expected}",
            result[0]
        );
    }

    /// Global mean is bounded by [min, max] of the input.
    #[test]
    fn prop_shaped_reduce_mean_bounded(input in finite_f32_vec(128)) {
        let config = ShapedReductionConfig::global(ReductionOp::Mean);
        let result =
            bitnet_kernels::shaped_reduction::reduce_f32(&input, &[input.len()], &config)
                .unwrap();
        let lo = input.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        prop_assert!(
            result[0] >= lo - 1e-5 && result[0] <= hi + 1e-5,
            "mean={} not in [{lo}, {hi}]",
            result[0]
        );
    }

    /// Global max matches iterator max.
    #[test]
    fn prop_shaped_reduce_max_matches_iter(input in finite_f32_vec(128)) {
        let config = ShapedReductionConfig::global(ReductionOp::Max);
        let result =
            bitnet_kernels::shaped_reduction::reduce_f32(&input, &[input.len()], &config)
                .unwrap();
        let expected = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        prop_assert_eq!(result[0], expected, "shaped max mismatch");
    }

    /// Global min matches iterator min.
    #[test]
    fn prop_shaped_reduce_min_matches_iter(input in finite_f32_vec(128)) {
        let config = ShapedReductionConfig::global(ReductionOp::Min);
        let result =
            bitnet_kernels::shaped_reduction::reduce_f32(&input, &[input.len()], &config)
                .unwrap();
        let expected = input.iter().copied().fold(f32::INFINITY, f32::min);
        prop_assert_eq!(result[0], expected, "shaped min mismatch");
    }

    /// Axis reduction removes the reduced dimension from the output shape.
    #[test]
    fn prop_shaped_reduce_axis_removes_dim(
        dim_a in 2usize..=8,
        dim_b in 2usize..=8,
    ) {
        let config = ShapedReductionConfig::new(ReductionOp::Sum, Some(0), false);
        let out_shape = reduction_output_shape(&[dim_a, dim_b], &config);
        prop_assert_eq!(out_shape, vec![dim_b], "axis-0 reduction should remove first dim");

        let config1 = ShapedReductionConfig::new(ReductionOp::Sum, Some(1), false);
        let out_shape1 = reduction_output_shape(&[dim_a, dim_b], &config1);
        prop_assert_eq!(out_shape1, vec![dim_a], "axis-1 reduction should remove second dim");
    }

    /// Axis reduction with keepdim preserves rank with size-1 dimension.
    #[test]
    fn prop_shaped_reduce_keepdim(
        dim_a in 2usize..=8,
        dim_b in 2usize..=8,
    ) {
        let config = ShapedReductionConfig::new(ReductionOp::Sum, Some(0), true);
        let out_shape = reduction_output_shape(&[dim_a, dim_b], &config);
        prop_assert_eq!(
            out_shape,
            vec![1, dim_b],
            "keepdim should produce [1, dim_b]"
        );
    }

    /// Axis-0 sum reduction: each output element equals the column sum.
    #[test]
    fn prop_shaped_reduce_axis0_sum_correct(
        rows in 2usize..=8,
        cols in 2usize..=8,
    ) {
        let total = rows * cols;
        let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.3).collect();
        let config = ShapedReductionConfig::new(ReductionOp::Sum, Some(0), false);
        let result =
            bitnet_kernels::shaped_reduction::reduce_f32(&input, &[rows, cols], &config).unwrap();

        prop_assert_eq!(result.len(), cols);
        for c in 0..cols {
            let expected: f32 = (0..rows).map(|r| input[r * cols + c]).sum();
            let tol = rows as f32 * 1e-3;
            prop_assert!(
                (result[c] - expected).abs() <= tol,
                "col {c}: got={}, expected={expected}",
                result[c]
            );
        }
    }
}
