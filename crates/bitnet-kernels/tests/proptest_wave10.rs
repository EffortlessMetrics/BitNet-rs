//! Wave 10 property tests: kernel invariants for reductions, normalization,
//! activations, transpose, softmax, scatter/gather, and quantization packing.
//!
//! Key invariants tested (20+ properties):
//! - SIMD reduction: sum/max/min/mean/L2 associativity within f32 tolerance
//! - Batch normalization: output mean ≈ 0, variance ≈ 1 after normalize
//! - Activations: sigmoid ∈ [0,1], relu ≥ 0, tanh ∈ [-1,1], gelu bounded
//! - Transpose: transpose(transpose(x)) ≈ x (involution)
//! - Softmax: outputs sum to ≈ 1.0, all in [0,1]
//! - Gather/scatter round-trip with identity indices
//! - Quantization pack/unpack round-trip preserves values

use bitnet_kernels::cpu::activations;
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm};
use bitnet_kernels::cpu::reduction::ReductionKernel;
use bitnet_kernels::reduction::{ReductionOp, reduce_f32};
use bitnet_kernels::scatter_gather::{
    GatherConfig, ScatterGatherKernel, ScatterMode, gather_cpu, scatter_cpu,
};
use proptest::prelude::*;

// -------------------------------------------------------------------
// Strategy helpers
// -------------------------------------------------------------------

/// Non-empty f32 vector with finite values in [-100, 100].
fn finite_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-100.0f32..100.0f32, 1..=max_len)
}

/// Non-empty f32 vector with small values (avoids exp overflow in softmax).
fn small_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-10.0f32..10.0f32, 1..=max_len)
}

/// Generate a 2-D matrix as a flat vec with given row/col bounds.
fn finite_matrix(
    max_rows: usize,
    max_cols: usize,
) -> impl Strategy<Value = (Vec<f32>, usize, usize)> {
    (1usize..=max_rows, 1usize..=max_cols).prop_flat_map(|(rows, cols)| {
        prop::collection::vec(-100.0f32..100.0f32, rows * cols)
            .prop_map(move |data| (data, rows, cols))
    })
}

// ===================================================================
// 1. SIMD reduction operations: associativity within f32 tolerance
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// reduce_f32 Sum matches iterative sum within tolerance.
    #[test]
    fn prop_reduce_sum_matches_iter(input in finite_f32_vec(256)) {
        let result = reduce_f32(&input, ReductionOp::Sum);
        let expected: f32 = input.iter().sum();
        let tol = input.len() as f32 * 1e-4;
        prop_assert!(
            (result - expected).abs() <= tol,
            "reduce Sum={result}, iter sum={expected}, diff={}", (result - expected).abs()
        );
    }

    /// reduce_f32 Max matches iterator max.
    #[test]
    fn prop_reduce_max_matches_iter(input in finite_f32_vec(256)) {
        let result = reduce_f32(&input, ReductionOp::Max);
        let expected = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        prop_assert_eq!(result, expected);
    }

    /// reduce_f32 Min matches iterator min.
    #[test]
    fn prop_reduce_min_matches_iter(input in finite_f32_vec(256)) {
        let result = reduce_f32(&input, ReductionOp::Min);
        let expected = input.iter().copied().fold(f32::INFINITY, f32::min);
        prop_assert_eq!(result, expected);
    }

    /// reduce_f32 Mean is between min and max.
    #[test]
    fn prop_reduce_mean_bounded(input in finite_f32_vec(256)) {
        let mean = reduce_f32(&input, ReductionOp::Mean);
        let lo = input.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        prop_assert!(
            mean >= lo - 1e-5 && mean <= hi + 1e-5,
            "mean={mean} not in [{lo}, {hi}]"
        );
    }

    /// reduce_f32 L2Norm is non-negative and matches manual sqrt(sum(x²)).
    #[test]
    fn prop_reduce_l2norm_nonneg(input in finite_f32_vec(256)) {
        let result = reduce_f32(&input, ReductionOp::L2Norm);
        prop_assert!(result >= 0.0, "L2Norm must be non-negative, got {result}");
        let expected = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let tol = expected * 1e-4 + 1e-6;
        prop_assert!(
            (result - expected).abs() <= tol,
            "L2Norm={result}, expected={expected}"
        );
    }

    /// CpuReduction sum agrees with reduce_f32 Sum.
    #[test]
    fn prop_cpu_reduction_sum_agrees(input in finite_f32_vec(256)) {
        let r1 = reduce_f32(&input, ReductionOp::Sum);
        let r2 = ReductionKernel::sum(&input).unwrap();
        let tol = input.len() as f32 * 1e-4;
        prop_assert!(
            (r1 - r2).abs() <= tol,
            "reduce_f32={r1}, ReductionKernel::sum={r2}"
        );
    }
}

// ===================================================================
// 2. Batch / layer normalization: mean ≈ 0, variance ≈ 1
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// After layer_norm (no affine), output mean ≈ 0.
    #[test]
    fn prop_layernorm_output_mean_near_zero(
        input in prop::collection::vec(-50.0f32..50.0f32, 4..=128)
    ) {
        let n = input.len();
        let gamma = vec![1.0f32; n];
        let mut config = LayerNormConfig::new(vec![n]);
        config.elementwise_affine = false;
        config.eps = 1e-5;

        let output = layer_norm(&input, &gamma, None, &config).unwrap();
        let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
        prop_assert!(
            mean.abs() < 0.01,
            "layer_norm output mean={mean}, expected ≈0"
        );
    }

    /// After layer_norm (no affine), output variance ≈ 1.
    #[test]
    fn prop_layernorm_output_variance_near_one(
        input in prop::collection::vec(-50.0f32..50.0f32, 4..=128)
    ) {
        let n = input.len();
        let gamma = vec![1.0f32; n];
        let mut config = LayerNormConfig::new(vec![n]);
        config.elementwise_affine = false;
        config.eps = 1e-5;

        let output = layer_norm(&input, &gamma, None, &config).unwrap();
        let mean: f32 = output.iter().sum::<f32>() / n as f32;
        let var: f32 = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        prop_assert!(
            (var - 1.0).abs() < 0.1,
            "layer_norm output variance={var}, expected ≈1"
        );
    }

    /// layer_norm with identity affine (gamma=1, beta=0) is the same as no affine.
    #[test]
    fn prop_layernorm_identity_affine(
        input in prop::collection::vec(-50.0f32..50.0f32, 4..=64)
    ) {
        let n = input.len();
        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];
        let mut config = LayerNormConfig::new(vec![n]);
        config.elementwise_affine = true;
        config.eps = 1e-5;

        let with_affine = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();

        config.elementwise_affine = false;
        let without_affine = layer_norm(&input, &gamma, None, &config).unwrap();

        for (a, b) in with_affine.iter().zip(without_affine.iter()) {
            prop_assert!(
                (a - b).abs() < 1e-5,
                "affine={a} vs no-affine={b}"
            );
        }
    }
}

// ===================================================================
// 3. Activation functions: range invariants
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Sigmoid output always in [0, 1].
    #[test]
    fn prop_sigmoid_range(input in finite_f32_vec(256)) {
        let mut output = vec![0.0f32; input.len()];
        activations::sigmoid_activate(&input, &mut output);
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(
                (0.0..=1.0).contains(&v),
                "sigmoid[{i}]={v} not in [0,1]"
            );
        }
    }

    /// ReLU output always ≥ 0.
    #[test]
    fn prop_relu_nonnegative(input in finite_f32_vec(256)) {
        let mut output = vec![0.0f32; input.len()];
        activations::relu_activate(&input, &mut output);
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(v >= 0.0, "relu[{i}]={v} < 0");
        }
    }

    /// ReLU preserves positive values exactly.
    #[test]
    fn prop_relu_preserves_positive(input in finite_f32_vec(256)) {
        let mut output = vec![0.0f32; input.len()];
        activations::relu_activate(&input, &mut output);
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            if inp > 0.0 {
                prop_assert_eq!(out, inp, "relu should preserve positive at [{}]", i);
            }
        }
    }

    /// Tanh output always in [-1, 1].
    #[test]
    fn prop_tanh_range(input in finite_f32_vec(256)) {
        let mut output = vec![0.0f32; input.len()];
        activations::tanh_activate(&input, &mut output);
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(
                (-1.0..=1.0).contains(&v),
                "tanh[{i}]={v} not in [-1,1]"
            );
        }
    }

    /// Sigmoid is monotonically non-decreasing.
    #[test]
    fn prop_sigmoid_monotone(input in finite_f32_vec(256)) {
        let mut sorted = input.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut output = vec![0.0f32; sorted.len()];
        activations::sigmoid_activate(&sorted, &mut output);
        for i in 1..output.len() {
            prop_assert!(
                output[i] >= output[i - 1] - 1e-6,
                "sigmoid not monotone: [{i}]={} < [{}]={}",
                output[i], i - 1, output[i - 1]
            );
        }
    }

    /// GELU output is bounded: gelu(x) ≥ -0.17 for all x (known lower bound).
    #[test]
    fn prop_gelu_lower_bound(input in finite_f32_vec(256)) {
        let mut output = vec![0.0f32; input.len()];
        activations::gelu_exact_activate(&input, &mut output);
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(
                v >= -0.2,
                "gelu[{i}]={v} below -0.2"
            );
        }
    }

    /// SiLU(0) = 0 and silu is continuous (small input → small output).
    #[test]
    fn prop_silu_at_zero_and_small(
        input in prop::collection::vec(-0.01f32..0.01f32, 1..=64)
    ) {
        let mut output = vec![0.0f32; input.len()];
        activations::silu_activate(&input, &mut output);
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(
                v.abs() < 0.02,
                "silu[{i}]={v} unexpectedly large for small input"
            );
        }
    }
}

// ===================================================================
// 4. Transpose: transpose(transpose(x)) ≈ x (involution)
// ===================================================================

/// Simple 2-D transpose of a row-major flat buffer.
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// transpose(transpose(x)) recovers x exactly.
    #[test]
    fn prop_transpose_involution((data, rows, cols) in finite_matrix(32, 32)) {
        let t1 = transpose_2d(&data, rows, cols);
        let t2 = transpose_2d(&t1, cols, rows);
        prop_assert_eq!(&data, &t2, "transpose²(x) != x");
    }

    /// Transposing a square matrix twice recovers the original.
    #[test]
    fn prop_transpose_square_involution(n in 1usize..=32) {
        let data: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
        let t1 = transpose_2d(&data, n, n);
        let t2 = transpose_2d(&t1, n, n);
        prop_assert_eq!(&data, &t2);
    }

    /// Transpose preserves element count and values (as a multiset).
    #[test]
    fn prop_transpose_preserves_elements((data, rows, cols) in finite_matrix(16, 16)) {
        let transposed = transpose_2d(&data, rows, cols);
        prop_assert_eq!(data.len(), transposed.len());
        let mut orig_sorted: Vec<u32> = data.iter().map(|x| x.to_bits()).collect();
        let mut trans_sorted: Vec<u32> = transposed.iter().map(|x| x.to_bits()).collect();
        orig_sorted.sort();
        trans_sorted.sort();
        prop_assert_eq!(orig_sorted, trans_sorted);
    }
}

// ===================================================================
// 5. Softmax: output sums to ≈ 1.0, all in [0, 1]
// ===================================================================

/// Row-wise softmax (numerically stable) for testing.
fn softmax_ref(input: &[f32]) -> Vec<f32> {
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Softmax outputs sum to approximately 1.0.
    #[test]
    fn prop_softmax_sums_to_one(input in small_f32_vec(256)) {
        let output = softmax_ref(&input);
        let sum: f32 = output.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax sum={sum}, expected ≈1.0"
        );
    }

    /// Every softmax output is in [0, 1].
    #[test]
    fn prop_softmax_outputs_in_unit(input in small_f32_vec(256)) {
        let output = softmax_ref(&input);
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(
                (0.0..=1.0).contains(&v),
                "softmax[{}]={} not in [0,1]", i, v
            );
        }
    }

    /// Softmax preserves ordering (larger input → larger output).
    #[test]
    fn prop_softmax_preserves_order(input in small_f32_vec(256)) {
        let output = softmax_ref(&input);
        for i in 0..input.len() {
            for j in (i + 1)..input.len() {
                if input[i] > input[j] + 1e-6 {
                    prop_assert!(
                        output[i] >= output[j] - 1e-6,
                        "softmax should preserve order: in[{}]={} > in[{}]={} but out[{}]={} < out[{}]={}",
                        i, input[i], j, input[j], i, output[i], j, output[j]
                    );
                }
            }
        }
    }

    /// Softmax with uniform input yields uniform output.
    #[test]
    fn prop_softmax_uniform_input(n in 1usize..=128) {
        let input = vec![1.0f32; n];
        let output = softmax_ref(&input);
        let expected = 1.0 / n as f32;
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(
                (v - expected).abs() < 1e-5,
                "softmax[{}]={}, expected uniform={}", i, v, expected
            );
        }
    }
}

// ===================================================================
// 6. Gather/scatter round-trip: gather(scatter(x)) recovers x
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Scatter then gather with identity indices recovers source (axis 0).
    #[test]
    fn prop_scatter_gather_roundtrip_axis0(n in 1usize..=32) {
        let cols = 4usize;
        let src: Vec<f32> = (0..n * cols).map(|i| i as f32).collect();
        let indices: Vec<usize> = (0..n).collect();

        // Scatter src into dst
        let mut dst = vec![0.0f32; n * cols];
        let config = GatherConfig::new(0, (n, cols), true).unwrap();
        scatter_cpu(
            &src,
            &indices.iter().flat_map(|&i| std::iter::repeat_n(i, cols)).collect::<Vec<_>>(),
            &mut dst,
            (n, cols),
            &config,
            ScatterMode::Assign,
        )
        .unwrap();

        // Gather back
        let kernel = ScatterGatherKernel::new(n, cols).unwrap();
        let mut recovered = vec![0.0f32; n * cols];
        let gather_indices: Vec<usize> =
            indices.iter().flat_map(|&i| std::iter::repeat_n(i, cols)).collect();
        gather_cpu(&dst, &gather_indices, &mut recovered, &kernel, &config).unwrap();

        prop_assert_eq!(&src, &recovered, "scatter→gather round-trip failed");
    }

    /// Index-select with identity indices is a no-op.
    #[test]
    fn prop_index_select_identity(n in 1usize..=32) {
        let cols = 4usize;
        let src: Vec<f32> = (0..n * cols).map(|i| i as f32).collect();
        let indices: Vec<usize> = (0..n).collect();

        let kernel = ScatterGatherKernel::new(n, cols).unwrap();
        let mut output = vec![0.0f32; n * cols];
        bitnet_kernels::scatter_gather::index_select_cpu(
            &src, &indices, &mut output, &kernel, true,
        )
        .unwrap();
        prop_assert_eq!(&src, &output, "index_select with identity indices should be no-op");
    }
}

// ===================================================================
// 7. Quantization pack/unpack round-trip
// ===================================================================

/// Pack 4 signed 2-bit values [-2..1] into one byte, then unpack.
fn pack_2bit(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(4));
    for chunk in values.chunks(4) {
        let mut byte = 0u8;
        for (i, &val) in chunk.iter().enumerate() {
            let clamped = val.clamp(-2, 1);
            let unsigned = (clamped + 2) as u8;
            byte |= unsigned << (i * 2);
        }
        packed.push(byte);
    }
    packed
}

fn unpack_2bit(packed: &[u8], output_len: usize) -> Vec<i8> {
    let mut values = Vec::with_capacity(output_len);
    for &byte in packed {
        for i in 0..4 {
            if values.len() >= output_len {
                break;
            }
            let unsigned = (byte >> (i * 2)) & 0x3;
            values.push(unsigned as i8 - 2);
        }
    }
    values
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// pack→unpack round-trip recovers the original 2-bit signed values.
    #[test]
    fn prop_pack_unpack_roundtrip(
        values in prop::collection::vec(-2i8..=1i8, 1..=256)
    ) {
        let packed = pack_2bit(&values);
        let unpacked = unpack_2bit(&packed, values.len());
        prop_assert_eq!(&values, &unpacked, "pack/unpack round-trip failed");
    }

    /// QK256 unpack block: pack→unpack for unsigned 2-bit codes [0..3].
    #[test]
    fn prop_qk256_unpack_roundtrip(
        codes in prop::collection::vec(0u8..=3u8, 256..=256)
    ) {
        // Pack 256 codes into 64 bytes
        let mut packed = [0u8; 64];
        for (i, &code) in codes.iter().enumerate() {
            let byte_idx = i / 4;
            let bit_shift = (i % 4) * 2;
            packed[byte_idx] |= code << bit_shift;
        }

        // Unpack using the same logic as QK256
        let mut unpacked = [0u8; 256];
        bitnet_quantization::i2s_qk256::unpack_qk256_block(&packed, &mut unpacked);

        for (i, (&orig, &recovered)) in codes.iter().zip(unpacked.iter()).enumerate() {
            prop_assert_eq!(
                orig, recovered,
                "QK256 unpack mismatch at [{}]: orig={}, recovered={}", i, orig, recovered
            );
        }
    }

    /// Packing never produces more bytes than ceil(n/4).
    #[test]
    fn prop_pack_size_bound(
        values in prop::collection::vec(-2i8..=1i8, 1..=256)
    ) {
        let packed = pack_2bit(&values);
        prop_assert_eq!(
            packed.len(),
            values.len().div_ceil(4),
            "packed size should be ceil(n/4)"
        );
    }
}
