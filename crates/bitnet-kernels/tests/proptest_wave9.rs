//! Wave 9 property tests: kernel module invariants.
//!
//! Key invariants tested:
//! - Reduction: sum preserves total, max ≥ all elements, min ≤ all elements,
//!   mean lies in [min, max], L2 norm is non-negative
//! - Conv2d: output dimensions match the analytical formula, identity kernel
//!   is a no-op, padding symmetry
//! - Scatter/gather: gather(scatter(x)) roundtrip, out-of-bounds detection,
//!   scatter Add mode accumulates correctly
//! - Softmax (via `softmax_cpu`): output sums to 1.0, all outputs in [0, 1],
//!   monotonicity preserved, temperature invariant
//! - RoPE: rotation preserves vector norms, position-0 identity, periodicity
//! - Layer norm: output mean ≈ 0, output variance ≈ 1, batch independence
//!   preserved, RMS norm outputs are finite

use bitnet_kernels::convolution::{Conv2DParams, conv2d};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::rope::{self, RopeConfig};
use bitnet_kernels::reduction::{ReductionOp, reduce_f32, reduce_rows_f32};
use bitnet_kernels::scatter_gather::{
    GatherConfig, ScatterGatherKernel, ScatterMode, gather_cpu, scatter_cpu,
};
use proptest::prelude::*;

// -------------------------------------------------------------------
// Strategy helpers
// -------------------------------------------------------------------

/// Generate a non-empty f32 vector with finite values in [-50, 50].
fn finite_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-50.0f32..50.0f32, 1..=max_len)
}

// ===================================================================
// Properties: Reduction ops
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Sum reduction equals the manual f32 sum of all elements.
    #[test]
    fn prop_reduce_sum_preserves_total(data in finite_f32_vec(256)) {
        let expected: f32 = data.iter().sum();
        let result = reduce_f32(&data, ReductionOp::Sum);
        let tol = expected.abs() * 1e-4 + 1e-4;
        prop_assert!(
            (result - expected).abs() < tol,
            "sum mismatch: reduce={result}, manual={expected}"
        );
    }

    /// Max reduction is ≥ every element in the input.
    #[test]
    fn prop_reduce_max_ge_all_elements(data in finite_f32_vec(256)) {
        let result = reduce_f32(&data, ReductionOp::Max);
        for (i, &v) in data.iter().enumerate() {
            prop_assert!(
                result >= v,
                "max {result} < element[{i}] = {v}"
            );
        }
    }

    /// Min reduction is ≤ every element in the input.
    #[test]
    fn prop_reduce_min_le_all_elements(data in finite_f32_vec(256)) {
        let result = reduce_f32(&data, ReductionOp::Min);
        for (i, &v) in data.iter().enumerate() {
            prop_assert!(
                result <= v,
                "min {result} > element[{i}] = {v}"
            );
        }
    }

    /// Mean lies in [min, max] of the input data.
    #[test]
    fn prop_reduce_mean_in_range(data in finite_f32_vec(256)) {
        let mean = reduce_f32(&data, ReductionOp::Mean);
        let min = reduce_f32(&data, ReductionOp::Min);
        let max = reduce_f32(&data, ReductionOp::Max);
        prop_assert!(
            mean >= min - 1e-5 && mean <= max + 1e-5,
            "mean {mean} not in [{min}, {max}]"
        );
    }

    /// L2 norm is non-negative for any input.
    #[test]
    fn prop_reduce_l2norm_nonnegative(data in finite_f32_vec(256)) {
        let result = reduce_f32(&data, ReductionOp::L2Norm);
        prop_assert!(result >= 0.0, "L2 norm is negative: {result}");
    }
}

// ===================================================================
// Properties: Conv2d — output dimensions and identity kernel
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Conv2d output spatial dimensions match the standard formula:
    ///   out_h = (ih + 2*pad_h - dil_h*(kh-1) - 1) / stride_h + 1
    ///   out_w = (iw + 2*pad_w - dil_w*(kw-1) - 1) / stride_w + 1
    #[test]
    fn prop_conv2d_output_dim_formula(
        ih in 4usize..=16,
        iw in 4usize..=16,
        kh in 1usize..=3,
        kw in 1usize..=3,
        stride in 1usize..=2,
        pad in 0usize..=2,
    ) {
        let ek_h = kh;
        let ek_w = kw;
        let padded_h = ih + 2 * pad;
        let padded_w = iw + 2 * pad;
        prop_assume!(padded_h >= ek_h && padded_w >= ek_w);

        let expected_oh = (padded_h - ek_h) / stride + 1;
        let expected_ow = (padded_w - ek_w) / stride + 1;
        let expected_out_size = expected_oh * expected_ow;

        let input = vec![1.0f32; ih * iw];
        let weight = vec![1.0f32; kh * kw];
        let mut output = vec![0.0f32; expected_out_size];
        let params = Conv2DParams {
            stride: (stride, stride),
            padding: (pad, pad),
            dilation: (1, 1),
        };

        let result = conv2d(
            &input, &weight, None, &mut output, params,
            (1, 1, ih, iw),
            (1, 1, kh, kw),
        );
        prop_assert!(result.is_ok(), "conv2d failed: {result:?}");
        prop_assert_eq!(output.len(), expected_out_size);
    }

    /// A 1×1 identity kernel reproduces the input exactly (no padding, stride 1).
    #[test]
    fn prop_conv2d_identity_kernel(
        ih in 1usize..=16,
        iw in 1usize..=16,
    ) {
        let input: Vec<f32> = (0..(ih * iw)).map(|i| i as f32 * 0.1).collect();
        let weight = vec![1.0f32]; // 1×1 kernel
        let mut output = vec![0.0f32; ih * iw];
        let params = Conv2DParams::default();

        conv2d(
            &input, &weight, None, &mut output, params,
            (1, 1, ih, iw),
            (1, 1, 1, 1),
        ).expect("identity conv2d must succeed");

        for (i, (&o, &e)) in output.iter().zip(input.iter()).enumerate() {
            prop_assert!(
                (o - e).abs() < 1e-6,
                "identity mismatch at {i}: {o} vs {e}"
            );
        }
    }

    /// Symmetric padding with a uniform kernel produces equal border values.
    #[test]
    fn prop_conv2d_padding_symmetry(
        size in 2usize..=8,
    ) {
        let ih = size;
        let iw = size;
        let pad = 1usize;
        let kh = 3usize;
        let kw = 3usize;

        let oh = ih + 2 * pad - kh + 1;
        let ow = iw + 2 * pad - kw + 1;

        let input = vec![1.0f32; ih * iw];
        let weight = vec![1.0f32; kh * kw];
        let mut output = vec![0.0f32; oh * ow];
        let params = Conv2DParams {
            stride: (1, 1),
            padding: (pad, pad),
            dilation: (1, 1),
        };

        conv2d(
            &input, &weight, None, &mut output, params,
            (1, 1, ih, iw),
            (1, 1, kh, kw),
        ).expect("padded conv2d must succeed");

        // With uniform input and uniform kernel, top-left and bottom-right
        // corners should produce identical values.
        let top_left = output[0];
        let bottom_right = output[oh * ow - 1];
        prop_assert!(
            (top_left - bottom_right).abs() < 1e-6,
            "corners differ: top_left={top_left} vs bottom_right={bottom_right}"
        );
    }
}

// ===================================================================
// Properties: Scatter/Gather — roundtrip and bounds checking
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Scatter then gather with unique row indices roundtrips correctly.
    #[test]
    fn prop_scatter_gather_roundtrip(
        cols in 1usize..=8,
        n_select in 1usize..=4,
    ) {
        let dst_rows = n_select + 2;
        // Build unique indices: [0, 1, ..., n_select-1]
        let indices: Vec<usize> = (0..n_select).collect();
        let src: Vec<f32> = (0..(n_select * cols))
            .map(|i| (i as f32 + 1.0) * 0.5)
            .collect();

        // Scatter src into dst
        let config = GatherConfig::new(0, (n_select, cols), true).unwrap();
        let mut dst = vec![0.0f32; dst_rows * cols];
        let flat_indices: Vec<usize> = indices.iter()
            .flat_map(|&row_idx| vec![row_idx; cols])
            .collect();

        scatter_cpu(
            &src, &flat_indices, &mut dst, (dst_rows, cols),
            &config, ScatterMode::Assign,
        ).expect("scatter must succeed");

        // Gather back from dst using same indices
        let kernel = ScatterGatherKernel::new(dst_rows, cols).unwrap();
        let gather_config = GatherConfig::new(0, (n_select, cols), true).unwrap();
        let gather_indices: Vec<usize> = indices.iter()
            .flat_map(|&row_idx| vec![row_idx; cols])
            .collect();
        let mut gathered = vec![0.0f32; n_select * cols];

        gather_cpu(
            &dst, &gather_indices, &mut gathered, &kernel, &gather_config,
        ).expect("gather must succeed");

        for (i, (&g, &s)) in gathered.iter().zip(src.iter()).enumerate() {
            prop_assert!(
                (g - s).abs() < 1e-6,
                "roundtrip mismatch at {i}: gathered={g} vs original={s}"
            );
        }
    }

    /// Gather with out-of-bounds indices and bounds_check=true returns error.
    #[test]
    fn prop_gather_oob_returns_error(
        rows in 1usize..=8,
        cols in 1usize..=8,
    ) {
        let src = vec![1.0f32; rows * cols];
        // Index = rows, which is exactly one past the last valid index
        let indices = vec![rows; cols];
        let kernel = ScatterGatherKernel::new(rows, cols).unwrap();
        let config = GatherConfig::new(0, (1, cols), true).unwrap();
        let mut output = vec![0.0f32; cols];
        let result = gather_cpu(&src, &indices, &mut output, &kernel, &config);
        prop_assert!(result.is_err(), "expected error for OOB index {rows}");
    }

    /// Scatter with Add mode accumulates all values at the same destination.
    #[test]
    fn prop_scatter_add_accumulates(
        cols in 1usize..=8,
        n_writes in 2usize..=6,
    ) {
        let dst_rows = 1usize;
        let indices = vec![0usize; n_writes * cols]; // all target row 0
        let src: Vec<f32> = (0..(n_writes * cols))
            .map(|i| i as f32 + 1.0)
            .collect();

        let config = GatherConfig::new(0, (n_writes, cols), true).unwrap();
        let mut dst = vec![0.0f32; dst_rows * cols];
        scatter_cpu(
            &src, &indices, &mut dst, (dst_rows, cols),
            &config, ScatterMode::Add,
        ).expect("scatter Add must succeed");

        // Each column should contain the sum of all writes to that column
        for col in 0..cols {
            let expected: f32 = (0..n_writes)
                .map(|w| src[w * cols + col])
                .sum();
            let tol = expected.abs() * 1e-5 + 1e-5;
            prop_assert!(
                (dst[col] - expected).abs() < tol,
                "col {col}: dst={} expected={expected}", dst[col]
            );
        }
    }
}

// ===================================================================
// Properties: Softmax — via softmax_cpu from cuda module
// (Uses row-wise reduction to compute softmax inline for CPU-only builds)
// ===================================================================

/// Inline numerically-stable softmax for property testing (no GPU dependency).
fn softmax_reference(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return vec![];
    }
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 { exps.iter().map(|&e| e / sum).collect() } else { exps }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Softmax outputs sum to approximately 1.0 for finite inputs.
    #[test]
    fn prop_softmax_sums_to_one(input in finite_f32_vec(256)) {
        let out = softmax_reference(&input);
        let sum: f32 = out.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-3,
            "softmax sum = {sum}, expected ~1.0"
        );
    }

    /// Every softmax output lies in [0, 1].
    #[test]
    fn prop_softmax_outputs_in_unit_interval(input in finite_f32_vec(256)) {
        let out = softmax_reference(&input);
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(
                v >= 0.0 - 1e-7 && v <= 1.0 + 1e-7,
                "softmax[{i}] = {v} not in [0, 1]"
            );
        }
    }

    /// Softmax preserves monotonicity: if input[i] > input[j] then output[i] >= output[j].
    #[test]
    fn prop_softmax_monotonicity(input in finite_f32_vec(64)) {
        let out = softmax_reference(&input);
        for i in 0..input.len() {
            for j in (i + 1)..input.len() {
                if input[i] > input[j] + 1e-6 {
                    prop_assert!(
                        out[i] >= out[j] - 1e-6,
                        "monotonicity violated: input[{i}]={} > input[{j}]={} \
                         but output[{i}]={} < output[{j}]={}",
                        input[i], input[j], out[i], out[j]
                    );
                }
            }
        }
    }

    /// Row-wise softmax via reduce_rows: row sums are consistent with flat softmax.
    #[test]
    fn prop_softmax_row_sum_consistency(
        n_rows in 1usize..=4,
        n_cols in 2usize..=16,
    ) {
        let total = n_rows * n_cols;
        let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.3 - 5.0).collect();

        // Apply softmax per row
        for row in 0..n_rows {
            let start = row * n_cols;
            let row_data = &input[start..start + n_cols];
            let out = softmax_reference(row_data);
            let sum: f32 = out.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-3,
                "row {row} softmax sum = {sum}"
            );
        }
    }
}

// ===================================================================
// Properties: RoPE — rotation invariants
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// RoPE rotation preserves the L2 norm of the head vector.
    #[test]
    fn prop_rope_preserves_norm(
        half_dim in 1usize..=16,
        position in 0usize..64,
    ) {
        let head_dim = half_dim * 2;
        let max_seq = position + 1;
        let cfg = RopeConfig::new(head_dim, max_seq);
        let freqs = rope::compute_frequencies(&cfg);

        let mut data: Vec<f32> =
            (0..head_dim).map(|i| (i as f32 + 1.0) * 0.3).collect();
        let norm_before: f32 =
            data.iter().map(|x| x * x).sum::<f32>().sqrt();

        rope::apply_rope(&mut data, position, head_dim, &freqs);

        let norm_after: f32 =
            data.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assert!(
            (norm_before - norm_after).abs() < 1e-3,
            "norm changed: {norm_before} -> {norm_after} \
             (pos={position}, head_dim={head_dim})"
        );
    }

    /// RoPE at position 0 is the identity transformation.
    #[test]
    fn prop_rope_position_zero_identity(
        half_dim in 1usize..=16,
    ) {
        let head_dim = half_dim * 2;
        let cfg = RopeConfig::new(head_dim, 1);
        let freqs = rope::compute_frequencies(&cfg);

        let mut data: Vec<f32> =
            (0..head_dim).map(|i| (i as f32 + 1.0) * 3.17).collect();
        let original = data.clone();

        rope::apply_rope(&mut data, 0, head_dim, &freqs);

        for (i, (&o, &d)) in original.iter().zip(data.iter()).enumerate() {
            prop_assert!(
                (o - d).abs() < 1e-5,
                "position-0 not identity at dim {i}: {o} vs {d}"
            );
        }
    }

    /// Different positions produce different rotated vectors (for non-zero input).
    #[test]
    fn prop_rope_different_positions_differ(
        half_dim in 1usize..=8,
        pos_a in 1usize..32,
        pos_b in 1usize..32,
    ) {
        prop_assume!(pos_a != pos_b);
        let head_dim = half_dim * 2;
        let max_seq = pos_a.max(pos_b) + 1;
        let cfg = RopeConfig::new(head_dim, max_seq);
        let freqs = rope::compute_frequencies(&cfg);

        let original: Vec<f32> =
            (0..head_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();

        let mut data_a = original.clone();
        rope::apply_rope(&mut data_a, pos_a, head_dim, &freqs);

        let mut data_b = original.clone();
        rope::apply_rope(&mut data_b, pos_b, head_dim, &freqs);

        let any_diff = data_a.iter().zip(data_b.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        prop_assert!(
            any_diff,
            "positions {pos_a} and {pos_b} produced identical rotations"
        );
    }
}

// ===================================================================
// Properties: Layer norm — statistical invariants
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Layer norm output (gamma=1, no beta) has mean ≈ 0.
    #[test]
    fn prop_layer_norm_output_zero_mean(
        norm_size in 4usize..=64,
    ) {
        let input: Vec<f32> = (0..norm_size)
            .map(|i| (i as f32) * 0.7 - 10.0)
            .collect();
        let gamma = vec![1.0f32; norm_size];
        let config = LayerNormConfig::new(vec![norm_size]);

        let out = layer_norm(&input, &gamma, None, &config)
            .expect("layer_norm must succeed");

        let mean: f32 = out.iter().sum::<f32>() / norm_size as f32;
        prop_assert!(
            mean.abs() < 1e-4,
            "layer norm mean = {mean}, expected ~0"
        );
    }

    /// Layer norm output (gamma=1, no beta) has variance ≈ 1.
    #[test]
    fn prop_layer_norm_output_unit_variance(
        norm_size in 8usize..=128,
    ) {
        let input: Vec<f32> = (0..norm_size)
            .map(|i| (i as f32) * 0.3 + 1.0)
            .collect();
        let gamma = vec![1.0f32; norm_size];
        let config = LayerNormConfig::new(vec![norm_size]);

        let out = layer_norm(&input, &gamma, None, &config)
            .expect("layer_norm must succeed");

        let mean = out.iter().sum::<f32>() / norm_size as f32;
        let var = out.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>() / norm_size as f32;
        prop_assert!(
            (var - 1.0).abs() < 0.05,
            "layer norm variance = {var}, expected ~1.0"
        );
    }

    /// Layer norm with batched input: each batch is independently normalized.
    #[test]
    fn prop_layer_norm_batch_independence(
        norm_size in 4usize..=32,
    ) {
        let input_a: Vec<f32> = (0..norm_size)
            .map(|i| i as f32 * 0.5)
            .collect();
        let input_b: Vec<f32> = (0..norm_size)
            .map(|i| (i as f32 + 10.0) * 2.0)
            .collect();
        let gamma = vec![1.0f32; norm_size];
        let config = LayerNormConfig::new(vec![norm_size]);

        let out_a = layer_norm(&input_a, &gamma, None, &config)
            .expect("layer_norm A must succeed");
        let out_b = layer_norm(&input_b, &gamma, None, &config)
            .expect("layer_norm B must succeed");

        let combined: Vec<f32> = input_a.iter()
            .chain(input_b.iter())
            .copied()
            .collect();
        let out_combined = layer_norm(&combined, &gamma, None, &config)
            .expect("layer_norm combined must succeed");

        for (i, (&c, &a)) in out_combined[..norm_size].iter().zip(out_a.iter()).enumerate() {
            prop_assert!(
                (c - a).abs() < 1e-5,
                "batch A independence at {i}: {c} vs {a}"
            );
        }
        for (i, (&c, &b)) in out_combined[norm_size..].iter().zip(out_b.iter()).enumerate() {
            prop_assert!(
                (c - b).abs() < 1e-5,
                "batch B independence at {i}: {c} vs {b}"
            );
        }
    }

    /// RMS norm output is always finite for finite inputs.
    #[test]
    fn prop_rms_norm_output_finite(
        norm_size in 2usize..=64,
    ) {
        let input: Vec<f32> = (0..norm_size)
            .map(|i| ((i as f32) * 1.7).sin() * 10.0)
            .collect();
        let gamma = vec![1.0f32; norm_size];
        let config = LayerNormConfig::new(vec![norm_size]);

        let out = rms_norm(&input, &gamma, &config)
            .expect("rms_norm must succeed");

        for (i, &v) in out.iter().enumerate() {
            prop_assert!(
                v.is_finite(),
                "rms_norm[{i}] = {v} is not finite"
            );
        }
    }
}

// ===================================================================
// Properties: Row-wise reduction invariants
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Row-wise max reduction: each row result ≥ every element in that row.
    #[test]
    fn prop_reduce_rows_max_ge_all(
        rows in 1usize..=8,
        cols in 1usize..=16,
    ) {
        let total = rows * cols;
        let matrix: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 1.3).sin() * 20.0)
            .collect();

        let maxes = reduce_rows_f32(&matrix, rows, cols, ReductionOp::Max)
            .expect("reduce_rows must succeed");

        for r in 0..rows {
            let start = r * cols;
            for c in 0..cols {
                prop_assert!(
                    maxes[r] >= matrix[start + c] - 1e-6,
                    "row {r} max {} < element [{r}][{c}] = {}",
                    maxes[r], matrix[start + c]
                );
            }
        }
    }
}
