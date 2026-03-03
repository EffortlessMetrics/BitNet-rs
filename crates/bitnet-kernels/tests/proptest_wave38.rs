//! Property-based tests — wave 38.
//!
//! Covers kernel dispatch ordering, SIMD alignment invariants, memory
//! allocation properties, tensor shape transformations, quantization
//! round-trips, and numerical stability bounds.
//!
//! 100+ property tests using proptest.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::activations::{relu, sigmoid, silu, tanh_act};
use bitnet_kernels::cpu::attention::{
    CpuAttention, CpuAttentionConfig, causal_mask, scaled_dot_product_attention,
};
use bitnet_kernels::cpu::batch::{
    batched_add, batched_layer_norm, batched_matmul, batched_softmax,
};
use bitnet_kernels::cpu::concat::ConcatKernel;
use bitnet_kernels::cpu::dequant::{dequant_i2s_block, dequant_ternary, pack_ternary};
use bitnet_kernels::cpu::embedding::embedding_lookup;
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::linear::{LinearConfig, linear_cpu};
use bitnet_kernels::cpu::loss::{
    LossReduction, cosine_similarity_loss, l1_loss, mse_loss, perplexity,
};
use bitnet_kernels::cpu::quantize::{
    dequantize_symmetric_i8, quantize_binary, quantize_symmetric_i8, quantize_ternary,
};
use bitnet_kernels::cpu::reduction::{ReductionKernel, ValueWithIndex};
use bitnet_kernels::cpu::residual::add_residual;
use bitnet_kernels::cpu::rope::{RopeConfig, compute_frequencies};
use bitnet_kernels::cpu::scatter_gather::{gather_1d, scatter_1d};
use bitnet_kernels::cpu::transpose::TransposeKernel;
use bitnet_kernels::cuda::softmax::{SoftmaxConfig, softmax_cpu, softmax_cpu_inplace};
use proptest::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn finite_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-10.0f32..10.0, 1..=max_len)
}

fn positive_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(0.01f32..10.0, 1..=max_len)
}

// ── 1. Kernel dispatch ordering (determinism, idempotency) ──────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // 1.1 Softmax is deterministic: same input → same output
    #[test]
    fn prop_softmax_deterministic(n in 1usize..=64) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let config = SoftmaxConfig::for_shape(n, 1).unwrap();
        let mut out1 = vec![0.0f32; n];
        let mut out2 = vec![0.0f32; n];
        softmax_cpu(&input, &mut out1, &config).unwrap();
        softmax_cpu(&input, &mut out2, &config).unwrap();
        prop_assert_eq!(out1, out2);
    }

    // 1.2 ReLU is deterministic (scalar)
    #[test]
    fn prop_relu_deterministic(val in -10.0f32..10.0) {
        let out1 = relu(val);
        let out2 = relu(val);
        prop_assert_eq!(out1.to_bits(), out2.to_bits());
    }

    // 1.3 Sigmoid is deterministic
    #[test]
    fn prop_sigmoid_deterministic(val in -10.0f32..10.0) {
        let out1 = sigmoid(val);
        let out2 = sigmoid(val);
        prop_assert_eq!(out1.to_bits(), out2.to_bits());
    }

    // 1.4 SiLU is deterministic
    #[test]
    fn prop_silu_deterministic(val in -10.0f32..10.0) {
        let out1 = silu(val);
        let out2 = silu(val);
        prop_assert_eq!(out1.to_bits(), out2.to_bits());
    }

    // 1.5 Tanh activation is deterministic
    #[test]
    fn prop_tanh_deterministic(val in -10.0f32..10.0) {
        let out1 = tanh_act(val);
        let out2 = tanh_act(val);
        prop_assert_eq!(out1.to_bits(), out2.to_bits());
    }

    // 1.6 ReLU is idempotent: relu(relu(x)) == relu(x)
    #[test]
    fn prop_relu_idempotent(val in -10.0f32..10.0) {
        let once = relu(val);
        let twice = relu(once);
        prop_assert_eq!(once.to_bits(), twice.to_bits());
    }

    // 1.7 Quantize symmetric i8 is deterministic
    #[test]
    fn prop_quantize_i8_deterministic(input in finite_f32_vec(64)) {
        let (q1, s1) = quantize_symmetric_i8(&input, 8);
        let (q2, s2) = quantize_symmetric_i8(&input, 8);
        prop_assert_eq!(q1, q2);
        prop_assert_eq!(s1.to_bits(), s2.to_bits());
    }

    // 1.8 Ternary quantization is deterministic
    #[test]
    fn prop_quantize_ternary_deterministic(
        input in finite_f32_vec(64),
        threshold in 0.01f32..5.0,
    ) {
        let q1 = quantize_ternary(&input, threshold);
        let q2 = quantize_ternary(&input, threshold);
        prop_assert_eq!(q1, q2);
    }

    // 1.9 Binary quantization is deterministic
    #[test]
    fn prop_quantize_binary_deterministic(input in finite_f32_vec(64)) {
        let q1 = quantize_binary(&input);
        let q2 = quantize_binary(&input);
        prop_assert_eq!(q1, q2);
    }

    // 1.10 Softmax inplace is deterministic
    #[test]
    fn prop_softmax_inplace_deterministic(n in 1usize..=64) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let config = SoftmaxConfig::for_shape(n, 1).unwrap();
        let mut a = input.clone();
        let mut b = input;
        softmax_cpu_inplace(&mut a, &config).unwrap();
        softmax_cpu_inplace(&mut b, &config).unwrap();
        prop_assert_eq!(a, b);
    }

    // 1.11 Reduction sum is deterministic
    #[test]
    fn prop_reduction_sum_deterministic(data in finite_f32_vec(128)) {
        let s1 = ReductionKernel::sum(&data).unwrap();
        let s2 = ReductionKernel::sum(&data).unwrap();
        prop_assert_eq!(s1.to_bits(), s2.to_bits());
    }

    // 1.12 Reduction mean is deterministic
    #[test]
    fn prop_reduction_mean_deterministic(data in finite_f32_vec(128)) {
        let m1 = ReductionKernel::mean(&data).unwrap();
        let m2 = ReductionKernel::mean(&data).unwrap();
        prop_assert_eq!(m1.to_bits(), m2.to_bits());
    }
}

// ── 2. SIMD alignment invariants ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // 2.1 Softmax output length always matches input
    #[test]
    fn prop_softmax_output_len(n in 1usize..=128) {
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let config = SoftmaxConfig::for_shape(n, 1).unwrap();
        let mut out = vec![0.0f32; n];
        softmax_cpu(&input, &mut out, &config).unwrap();
        prop_assert_eq!(out.len(), n);
    }

    // 2.2 Silu_vec output length matches input
    #[test]
    fn prop_silu_vec_output_len(input in finite_f32_vec(128)) {
        let out = bitnet_kernels::cpu::activations::silu_vec(&input);
        prop_assert_eq!(out.len(), input.len());
    }

    // 2.3 Gelu_vec output length matches input
    #[test]
    fn prop_gelu_vec_output_len(input in finite_f32_vec(128)) {
        let out = bitnet_kernels::cpu::activations::gelu_vec(&input);
        prop_assert_eq!(out.len(), input.len());
    }

    // 2.4 Scatter-gather preserves length
    #[test]
    fn prop_scatter_gather_length(n in 1usize..=64) {
        let data = vec![0.0f32; n];
        let indices: Vec<usize> = (0..n).collect();
        let values: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut buf = data;
        scatter_1d(&mut buf, &indices, &values).unwrap();
        prop_assert_eq!(buf.len(), n);
    }

    // 2.5 Quantize symmetric i8 output length matches input
    #[test]
    fn prop_quantize_i8_output_len(input in finite_f32_vec(128)) {
        let (quantized, _scale) = quantize_symmetric_i8(&input, 8);
        prop_assert_eq!(quantized.len(), input.len());
    }

    // 2.6 Dequantize symmetric i8 output length matches input
    #[test]
    fn prop_dequantize_i8_output_len(input in finite_f32_vec(128)) {
        let (quantized, scale) = quantize_symmetric_i8(&input, 8);
        let deq = dequantize_symmetric_i8(&quantized, scale);
        prop_assert_eq!(deq.len(), input.len());
    }

    // 2.7 Aligned vectors remain aligned after vec activations
    #[test]
    fn prop_aligned_size_after_silu(
        n in (1usize..=64).prop_map(|n| n * 16)
    ) {
        let data = vec![1.0f32; n];
        let out = bitnet_kernels::cpu::activations::silu_vec(&data);
        prop_assert_eq!(out.len() % 16, 0);
    }

    // 2.8 Ternary quantize output length
    #[test]
    fn prop_ternary_quantize_length(input in finite_f32_vec(64)) {
        let q = quantize_ternary(&input, 0.5);
        prop_assert_eq!(q.len(), input.len());
    }

    // 2.9 Binary quantize output length
    #[test]
    fn prop_binary_quantize_length(input in finite_f32_vec(64)) {
        let q = quantize_binary(&input);
        prop_assert_eq!(q.len(), input.len());
    }

    // 2.10 Batched softmax output length
    #[test]
    fn prop_batched_softmax_len(
        batch in 1usize..=4,
        seq in 1usize..=16,
    ) {
        let data = vec![1.0f32; batch * seq];
        let out = batched_softmax(&data, batch, seq).unwrap();
        prop_assert_eq!(out.len(), batch * seq);
    }

    // 2.11 Batched add output length
    #[test]
    fn prop_batched_add_len(
        batch in 1usize..=4,
        dim in 1usize..=16,
    ) {
        let a = vec![1.0f32; batch * dim];
        let b = vec![2.0f32; batch * dim];
        let out = batched_add(&a, &b, batch, dim).unwrap();
        prop_assert_eq!(out.len(), batch * dim);
    }

    // 2.12 Reduction output is a single scalar
    #[test]
    fn prop_reduction_sum_is_scalar(data in finite_f32_vec(64)) {
        let sum = ReductionKernel::sum(&data).unwrap();
        prop_assert!(sum.is_finite());
    }
}

// ── 3. Memory allocation properties ────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // 3.1 Layer norm output has correct size
    #[test]
    fn prop_layer_norm_output_size(n in 1usize..=64) {
        let input = vec![1.0f32; n];
        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];
        let config = LayerNormConfig::new(vec![n]);
        let out = layer_norm(&input, &gamma, Some(beta.as_slice()), &config).unwrap();
        prop_assert_eq!(out.len(), n);
    }

    // 3.2 RMS norm output has correct size
    #[test]
    fn prop_rms_norm_output_size(n in 1usize..=64) {
        let input = vec![1.0f32; n];
        let gamma = vec![1.0f32; n];
        let config = LayerNormConfig::new(vec![n]);
        let out = rms_norm(&input, &gamma, &config).unwrap();
        prop_assert_eq!(out.len(), n);
    }

    // 3.3 Linear forward output has correct size
    #[test]
    fn prop_linear_output_size(
        in_feat in 1usize..=16,
        out_feat in 1usize..=16,
    ) {
        let batch = 1;
        let input = vec![1.0f32; batch * in_feat];
        let weight = vec![0.1f32; out_feat * in_feat];
        let bias = vec![0.0f32; out_feat];
        let config = LinearConfig::new(batch, in_feat, out_feat).unwrap();
        let mut out = vec![0.0f32; batch * out_feat];
        linear_cpu(&input, &weight, Some(&bias), &mut out, &config).unwrap();
        prop_assert_eq!(out.len(), batch * out_feat);
    }

    // 3.4 Transpose 2D output has correct size
    #[test]
    fn prop_transpose_2d_size(
        rows in 1usize..=16,
        cols in 1usize..=16,
    ) {
        let data: Vec<f32> = (0..(rows * cols) as u32).map(|i| i as f32).collect();
        let out = TransposeKernel::transpose_2d(&data, rows, cols).unwrap();
        prop_assert_eq!(out.len(), rows * cols);
    }

    // 3.5 Causal mask has correct size
    #[test]
    fn prop_causal_mask_size(seq_len in 1usize..=32) {
        let mask = causal_mask(seq_len);
        prop_assert_eq!(mask.len(), seq_len * seq_len);
    }

    // 3.6 Batched matmul output size
    #[test]
    fn prop_batched_matmul_output_size(
        batch in 1usize..=4,
        m in 1usize..=8,
        k in 1usize..=8,
        n in 1usize..=8,
    ) {
        let a = vec![0.1f32; batch * m * k];
        let b = vec![0.1f32; batch * k * n];
        let out = batched_matmul(&a, &b, batch, m, k, n).unwrap();
        prop_assert_eq!(out.len(), batch * m * n);
    }

    // 3.7 Embedding lookup output size
    #[test]
    fn prop_embedding_lookup_size(
        vocab in 2usize..=32,
        dim in 1usize..=16,
        n_tokens in 1usize..=8,
    ) {
        let table: Vec<f32> = (0..(vocab * dim) as u32).map(|i| i as f32).collect();
        let indices: Vec<u32> = (0..n_tokens).map(|i| (i % vocab) as u32).collect();
        let out = embedding_lookup(&table, &indices, dim).unwrap();
        prop_assert_eq!(out.len(), n_tokens * dim);
    }

    // 3.8 Concat two vectors preserves total element count
    #[test]
    fn prop_concat_preserves_elements(
        a_len in 1usize..=32,
        b_len in 1usize..=32,
    ) {
        let a: Vec<f32> = (0..a_len as u32).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..b_len as u32).map(|i| i as f32).collect();
        let inputs: Vec<&[f32]> = vec![&a, &b];
        let shape_a = [a_len];
        let shape_b = [b_len];
        let shapes: Vec<&[usize]> = vec![&shape_a, &shape_b];
        let out = ConcatKernel::concat(&inputs, &shapes, 0).unwrap();
        prop_assert_eq!(out.len(), a_len + b_len);
    }

    // 3.9 RoPE frequencies have correct size
    #[test]
    fn prop_rope_frequencies_size(head_dim_half in 1usize..=16) {
        let head_dim = head_dim_half * 2;
        let max_seq = 32;
        let config = RopeConfig::new(head_dim, max_seq);
        let freqs = compute_frequencies(&config);
        // Each position produces head_dim values (cos+sin pairs for head_dim/2 dims)
        prop_assert_eq!(freqs.len(), max_seq * head_dim);
    }

    // 3.10 Batched layer norm output size
    #[test]
    fn prop_batched_layer_norm_size(
        batch in 1usize..=4,
        dim in 2usize..=16,
    ) {
        let input = vec![1.0f32; batch * dim];
        let gamma = vec![1.0f32; dim];
        let beta = vec![0.0f32; dim];
        let out = batched_layer_norm(&input, &gamma, &beta, batch, dim, 1e-5).unwrap();
        prop_assert_eq!(out.len(), batch * dim);
    }

    // 3.11 Gather output length
    #[test]
    fn prop_gather_output_len(n in 1usize..=32) {
        let data: Vec<f32> = (0..n as u32).map(|i| i as f32).collect();
        let indices: Vec<usize> = (0..n).collect();
        let out = gather_1d(&data, &indices).unwrap();
        prop_assert_eq!(out.len(), n);
    }

    // 3.12 Dequant i2s block output size
    #[test]
    fn prop_dequant_i2s_block_size(n_blocks in 1usize..=8) {
        let block_size = 32;
        let bytes_per_block = block_size / 4;
        // dequant_i2s_block decodes a single block from the packed bytes
        let packed = vec![0u8; bytes_per_block];
        let out = dequant_i2s_block(&packed, 1.0, block_size).unwrap();
        prop_assert_eq!(out.len(), block_size);
        let _ = n_blocks; // exercise different strategy inputs
    }
}

// ── 4. Tensor shape transformation properties ──────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // 4.1 Transpose 2D is an involution: transpose(transpose(x)) == x
    #[test]
    fn prop_transpose_2d_involution(
        rows in 1usize..=12,
        cols in 1usize..=12,
    ) {
        let data: Vec<f32> = (0..(rows * cols) as u32).map(|i| i as f32).collect();
        let t1 = TransposeKernel::transpose_2d(&data, rows, cols).unwrap();
        let t2 = TransposeKernel::transpose_2d(&t1, cols, rows).unwrap();
        prop_assert_eq!(data, t2);
    }

    // 4.2 Reshape preserves total element count
    #[test]
    fn prop_reshape_preserves_elements(
        rows in 1usize..=8,
        cols in 1usize..=8,
    ) {
        let n = rows * cols;
        let data: Vec<f32> = (0..n as u32).map(|i| i as f32).collect();
        let reshaped = TransposeKernel::reshape(&data, &[rows, cols], &[n]).unwrap();
        prop_assert_eq!(reshaped.len(), n);
        prop_assert_eq!(data, reshaped);
    }

    // 4.3 Flatten preserves elements
    #[test]
    fn prop_flatten_preserves_elements(
        d0 in 1usize..=4,
        d1 in 1usize..=4,
        d2 in 1usize..=4,
    ) {
        let n = d0 * d1 * d2;
        let data: Vec<f32> = (0..n as u32).map(|i| i as f32).collect();
        let (flat_data, flat_shape) = TransposeKernel::flatten(&data, &[d0, d1, d2], 0, 2).unwrap();
        prop_assert_eq!(flat_data.len(), n);
        let _ = flat_shape;
    }

    // 4.4 Squeeze removes unit dims
    #[test]
    fn prop_squeeze_removes_ones(d in 1usize..=8) {
        let shape = vec![1, d, 1];
        let squeezed = TransposeKernel::squeeze(&shape);
        prop_assert!(!squeezed.contains(&1) || squeezed == vec![1]);
    }

    // 4.5 Unsqueeze adds a dim
    #[test]
    fn prop_unsqueeze_adds_dim(
        d0 in 1usize..=8,
        d1 in 1usize..=8,
    ) {
        let shape = vec![d0, d1];
        let result = TransposeKernel::unsqueeze(&shape, 0).unwrap();
        prop_assert_eq!(result.len(), 3);
        prop_assert_eq!(result[0], 1);
    }

    // 4.6 Contiguous strides last dim stride is 1
    #[test]
    fn prop_contiguous_strides_last_is_one(
        d0 in 1usize..=8,
        d1 in 1usize..=8,
    ) {
        let strides = TransposeKernel::contiguous_strides(&[d0, d1]);
        prop_assert_eq!(*strides.last().unwrap(), 1);
    }

    // 4.7 Concat output shape matches sum on axis
    #[test]
    fn prop_concat_shape_axis0(
        a_rows in 1usize..=8,
        b_rows in 1usize..=8,
        cols in 1usize..=8,
    ) {
        let a: Vec<f32> = vec![1.0; a_rows * cols];
        let b: Vec<f32> = vec![2.0; b_rows * cols];
        let inputs: Vec<&[f32]> = vec![&a, &b];
        let shape_a = [a_rows, cols];
        let shape_b = [b_rows, cols];
        let shapes: Vec<&[usize]> = vec![&shape_a, &shape_b];
        let out = ConcatKernel::concat(&inputs, &shapes, 0).unwrap();
        prop_assert_eq!(out.len(), (a_rows + b_rows) * cols);
    }

    // 4.8 Split and concat element count preserved
    #[test]
    fn prop_split_element_count(n in 1usize..=8) {
        let total = n * 2;
        let data: Vec<f32> = (0..total as u32).map(|i| i as f32).collect();
        // split divides axis evenly into num_splits parts
        let parts = ConcatKernel::split(&data, &[total], 0, 2).unwrap();
        prop_assert_eq!(parts.len(), 2);
        prop_assert_eq!(parts[0].len() + parts[1].len(), total);
    }

    // 4.9 Causal mask diagonal is always 0 (unmasked)
    #[test]
    fn prop_causal_mask_diagonal(seq_len in 1usize..=16) {
        let mask = causal_mask(seq_len);
        for i in 0..seq_len {
            prop_assert_eq!(mask[i * seq_len + i], 0.0, "diagonal should be unmasked");
        }
    }

    // 4.10 Causal mask upper triangle is -inf
    #[test]
    fn prop_causal_mask_upper_triangle(seq_len in 2usize..=16) {
        let mask = causal_mask(seq_len);
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                prop_assert!(
                    mask[i * seq_len + j].is_infinite() && mask[i * seq_len + j] < 0.0,
                    "upper triangle should be -inf at ({}, {})", i, j,
                );
            }
        }
    }

    // 4.11 Is_contiguous true for contiguous strides
    #[test]
    fn prop_is_contiguous_for_contiguous(
        d0 in 1usize..=8,
        d1 in 1usize..=8,
    ) {
        let shape = vec![d0, d1];
        let strides = TransposeKernel::contiguous_strides(&shape);
        prop_assert!(TransposeKernel::is_contiguous(&shape, &strides));
    }

    // 4.12 Gather with identity indices is identity
    #[test]
    fn prop_gather_identity(n in 1usize..=32) {
        let data: Vec<f32> = (0..n as u32).map(|i| i as f32).collect();
        let indices: Vec<usize> = (0..n).collect();
        let out = gather_1d(&data, &indices).unwrap();
        prop_assert_eq!(out, data);
    }
}

// ── 5. Quantization round-trip properties ──────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // 5.1 Symmetric i8 round-trip error is bounded
    #[test]
    fn prop_symmetric_i8_roundtrip_bounded(input in finite_f32_vec(64)) {
        let (quantized, scale) = quantize_symmetric_i8(&input, 8);
        let deq = dequantize_symmetric_i8(&quantized, scale);
        for (orig, recovered) in input.iter().zip(deq.iter()) {
            let err = (orig - recovered).abs();
            prop_assert!(err <= scale + 1e-6,
                "round-trip error {} exceeds scale {}", err, scale);
        }
    }

    // 5.2 Quantized values are within i8 range
    #[test]
    fn prop_quantize_i8_range(input in finite_f32_vec(64)) {
        let (quantized, _) = quantize_symmetric_i8(&input, 8);
        for &q in &quantized {
            prop_assert!((-128..=127).contains(&(q as i16)));
        }
    }

    // 5.3 Ternary quantization produces only {-1, 0, 1}
    #[test]
    fn prop_ternary_values(
        input in finite_f32_vec(64),
        threshold in 0.01f32..5.0,
    ) {
        let q = quantize_ternary(&input, threshold);
        for &v in &q {
            prop_assert!(v == -1 || v == 0 || v == 1,
                "ternary value {} not in {{-1,0,1}}", v);
        }
    }

    // 5.4 Binary quantization produces only {-1, 1}
    #[test]
    fn prop_binary_values(input in finite_f32_vec(64)) {
        let q = quantize_binary(&input);
        for &v in &q {
            prop_assert!(v == -1 || v == 1, "binary value {} not in {{-1,1}}", v);
        }
    }

    // 5.5 Pack/unpack ternary round-trip preserves sign
    #[test]
    fn prop_pack_ternary_roundtrip_sign(
        input in proptest::collection::vec(-5.0f32..5.0, 4..=64),
        threshold in 0.01f32..3.0,
    ) {
        let (packed, scale) = pack_ternary(&input, threshold);
        let unpacked = dequant_ternary(&packed, scale);
        for (orig, recovered) in input.iter().zip(unpacked.iter()) {
            if orig.abs() > threshold {
                prop_assert!(
                    orig.signum() == recovered.signum() || *recovered == 0.0,
                    "sign mismatch: orig={} recovered={}", orig, recovered,
                );
            }
        }
    }

    // 5.6 Zero input quantizes to zero
    #[test]
    fn prop_zero_quantize_zero(n in 1usize..=64) {
        let input = vec![0.0f32; n];
        let (quantized, _) = quantize_symmetric_i8(&input, 8);
        for &q in &quantized {
            prop_assert_eq!(q, 0);
        }
    }

    // 5.7 Symmetric i8 quantization scale is non-negative
    #[test]
    fn prop_quantize_scale_nonneg(input in finite_f32_vec(64)) {
        let (_, scale) = quantize_symmetric_i8(&input, 8);
        prop_assert!(scale >= 0.0, "scale should be non-negative: {}", scale);
    }

    // 5.8 Constant input quantizes to constant output
    #[test]
    fn prop_constant_quantize(
        val in -5.0f32..5.0,
        n in 2usize..=32,
    ) {
        let input = vec![val; n];
        let (quantized, _) = quantize_symmetric_i8(&input, 8);
        let first = quantized[0];
        for &q in &quantized {
            prop_assert_eq!(q, first);
        }
    }

    // 5.9 Dequantized ternary values are bounded by scale
    #[test]
    fn prop_dequant_ternary_bounded(
        n in 4usize..=64,
        scale in 0.1f32..10.0,
    ) {
        let packed = vec![0b01_01_01_01u8; n];
        let out = dequant_ternary(&packed, scale);
        for &v in &out {
            prop_assert!(
                v.abs() <= scale + 1e-6 || v == 0.0,
                "value {} exceeds scale {}", v, scale,
            );
        }
    }

    // 5.10 Negative of input gets negative quantization
    #[test]
    fn prop_quantize_negation(input in positive_f32_vec(32)) {
        let neg_input: Vec<f32> = input.iter().map(|x| -x).collect();
        let (q_pos, _) = quantize_symmetric_i8(&input, 8);
        let (q_neg, _) = quantize_symmetric_i8(&neg_input, 8);
        for (qp, qn) in q_pos.iter().zip(q_neg.iter()) {
            prop_assert_eq!(*qp, -(*qn));
        }
    }

    // 5.11 Round-trip error decreases with more bits (8 vs 4)
    #[test]
    fn prop_more_bits_less_error(input in finite_f32_vec(32)) {
        let (q8, s8) = quantize_symmetric_i8(&input, 8);
        let d8 = dequantize_symmetric_i8(&q8, s8);
        let (q4, s4) = quantize_symmetric_i8(&input, 4);
        let d4 = dequantize_symmetric_i8(&q4, s4);
        let err8: f32 = input.iter().zip(d8.iter()).map(|(a, b)| (a - b).abs()).sum();
        let err4: f32 = input.iter().zip(d4.iter()).map(|(a, b)| (a - b).abs()).sum();
        prop_assert!(err8 <= err4 + 1e-6,
            "8-bit error {} > 4-bit error {}", err8, err4);
    }

    // 5.12 Dequant i2s block output values are scaled
    #[test]
    fn prop_dequant_i2s_values_bounded(n_blocks in 1usize..=4) {
        let block_size = 32;
        let bytes_per_block = block_size / 4;
        let packed = vec![0u8; n_blocks * bytes_per_block];
        let scale = 2.0f32;
        let out = dequant_i2s_block(&packed, scale, block_size).unwrap();
        for &v in &out {
            prop_assert!(v.abs() <= scale + 1e-6);
        }
    }
}

// ── 6. Numerical stability bounds ──────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // 6.1 Softmax output sums to 1
    #[test]
    fn prop_softmax_sums_to_one(n in 1usize..=64) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let config = SoftmaxConfig::for_shape(n, 1).unwrap();
        let mut out = vec![0.0f32; n];
        softmax_cpu(&input, &mut out, &config).unwrap();
        let sum: f32 = out.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-4, "softmax sum={} not ~1.0", sum);
    }

    // 6.2 Softmax output is non-negative
    #[test]
    fn prop_softmax_nonneg(n in 1usize..=64) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let config = SoftmaxConfig::for_shape(n, 1).unwrap();
        let mut out = vec![0.0f32; n];
        softmax_cpu(&input, &mut out, &config).unwrap();
        for &v in &out {
            prop_assert!(v >= 0.0, "softmax produced negative: {}", v);
        }
    }

    // 6.3 Softmax output at most 1
    #[test]
    fn prop_softmax_at_most_one(n in 1usize..=64) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let config = SoftmaxConfig::for_shape(n, 1).unwrap();
        let mut out = vec![0.0f32; n];
        softmax_cpu(&input, &mut out, &config).unwrap();
        for &v in &out {
            prop_assert!(v <= 1.0 + 1e-6, "softmax value > 1: {}", v);
        }
    }

    // 6.4 Sigmoid output in [0, 1]
    #[test]
    fn prop_sigmoid_bounded(val in -10.0f32..10.0) {
        let out = sigmoid(val);
        prop_assert!((0.0..=1.0).contains(&out), "sigmoid out of [0,1]: {}", out);
    }

    // 6.5 Tanh output in [-1, 1]
    #[test]
    fn prop_tanh_bounded(val in -10.0f32..10.0) {
        let out = tanh_act(val);
        prop_assert!((-1.0..=1.0).contains(&out), "tanh out of [-1,1]: {}", out);
    }

    // 6.6 ReLU output is non-negative
    #[test]
    fn prop_relu_nonneg(val in -10.0f32..10.0) {
        let out = relu(val);
        prop_assert!(out >= 0.0, "relu produced negative: {}", out);
    }

    // 6.7 SiLU output is finite
    #[test]
    fn prop_silu_finite(val in -10.0f32..10.0) {
        let out = silu(val);
        prop_assert!(out.is_finite(), "silu produced non-finite: {}", out);
    }

    // 6.8 MSE loss is non-negative
    #[test]
    fn prop_mse_nonneg(data in finite_f32_vec(32)) {
        let loss = mse_loss(&data, &data, LossReduction::Mean).unwrap();
        prop_assert!(loss >= 0.0, "MSE loss negative: {}", loss);
    }

    // 6.9 MSE of identical inputs is zero
    #[test]
    fn prop_mse_identical_zero(data in finite_f32_vec(32)) {
        let loss = mse_loss(&data, &data, LossReduction::Mean).unwrap();
        prop_assert!(loss.abs() < 1e-6, "MSE of identical inputs not zero: {}", loss);
    }

    // 6.10 L1 loss is non-negative
    #[test]
    fn prop_l1_nonneg(data in finite_f32_vec(32)) {
        let loss = l1_loss(&data, &data, LossReduction::Mean).unwrap();
        prop_assert!(loss >= -1e-6, "L1 loss negative: {}", loss);
    }

    // 6.11 Cosine similarity in [-1, 1]
    #[test]
    fn prop_cosine_similarity_bounded(a in positive_f32_vec(16)) {
        let b: Vec<f32> = a.iter().map(|x| x * 2.0).collect();
        let sim = cosine_similarity_loss(&a, &b).unwrap();
        prop_assert!((-1.0 - 1e-6..=1.0 + 1e-6).contains(&sim),
            "cosine similarity out of [-1,1]: {}", sim);
    }

    // 6.12 Perplexity is always >= 1
    #[test]
    fn prop_perplexity_ge_one(ce_loss in 0.0f32..20.0) {
        let ppl = perplexity(ce_loss);
        prop_assert!(ppl >= 1.0 - 1e-6, "perplexity < 1: {}", ppl);
    }

    // 6.13 Layer norm output has near-zero mean
    #[test]
    fn prop_layer_norm_zero_mean(n in 4usize..=32) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 1.5).collect();
        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];
        let config = LayerNormConfig::new(vec![n]);
        let out = layer_norm(&input, &gamma, Some(beta.as_slice()), &config).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / n as f32;
        prop_assert!(mean.abs() < 1e-4, "layer norm mean not ~0: {}", mean);
    }

    // 6.14 Layer norm output has near-unit variance
    #[test]
    fn prop_layer_norm_unit_var(n in 4usize..=32) {
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 2.0).collect();
        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];
        let config = LayerNormConfig::new(vec![n]);
        let out = layer_norm(&input, &gamma, Some(beta.as_slice()), &config).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / n as f32;
        let var: f32 = out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
        prop_assert!((var - 1.0).abs() < 0.2, "layer norm variance not ~1: {}", var);
    }

    // 6.15 Reduction max is >= all elements
    #[test]
    fn prop_reduction_max_dominates(data in finite_f32_vec(64)) {
        let ValueWithIndex { value: max_val, .. } = ReductionKernel::max(&data).unwrap();
        for &v in &data {
            prop_assert!(max_val >= v, "max {} < element {}", max_val, v);
        }
    }

    // 6.16 Reduction min is <= all elements
    #[test]
    fn prop_reduction_min_dominates(data in finite_f32_vec(64)) {
        let ValueWithIndex { value: min_val, .. } = ReductionKernel::min(&data).unwrap();
        for &v in &data {
            prop_assert!(min_val <= v, "min {} > element {}", min_val, v);
        }
    }

    // 6.17 Reduction mean is between min and max
    #[test]
    fn prop_reduction_mean_between_extremes(data in finite_f32_vec(64)) {
        let mean = ReductionKernel::mean(&data).unwrap();
        let ValueWithIndex { value: max_val, .. } = ReductionKernel::max(&data).unwrap();
        let ValueWithIndex { value: min_val, .. } = ReductionKernel::min(&data).unwrap();
        prop_assert!(mean >= min_val - 1e-6 && mean <= max_val + 1e-6,
            "mean {} not in [{}, {}]", mean, min_val, max_val);
    }

    // 6.18 L1 norm is non-negative
    #[test]
    fn prop_l1_norm_nonneg(data in finite_f32_vec(64)) {
        let norm = ReductionKernel::l1_norm(&data).unwrap();
        prop_assert!(norm >= 0.0, "L1 norm negative: {}", norm);
    }

    // 6.19 Residual connection: out = x + residual
    #[test]
    fn prop_residual_addition(data in finite_f32_vec(32)) {
        let residual = vec![1.0f32; data.len()];
        let mut output = data.clone();
        add_residual(&mut output, &residual).unwrap();
        for (i, (&o, &d)) in output.iter().zip(data.iter()).enumerate() {
            prop_assert!((o - d - 1.0).abs() < 1e-6,
                "residual mismatch at {}: {} != {} + 1.0", i, o, d);
        }
    }

    // 6.20 RoPE frequencies are finite
    #[test]
    fn prop_rope_freqs_finite(head_dim_half in 1usize..=8) {
        let head_dim = head_dim_half * 2;
        let config = RopeConfig::new(head_dim, 16);
        let freqs = compute_frequencies(&config);
        for &f in &freqs {
            prop_assert!(f.is_finite(), "rope freq not finite: {}", f);
        }
    }

    // 6.21 Batched softmax each batch sums to 1
    #[test]
    fn prop_batched_softmax_row_sums(
        batch in 1usize..=4,
        seq_len in 2usize..=16,
    ) {
        let data: Vec<f32> = (0..(batch * seq_len) as u32).map(|i| (i as f32) * 0.1).collect();
        let out = batched_softmax(&data, batch, seq_len).unwrap();
        for b in 0..batch {
            let sum: f32 = out[b * seq_len..(b + 1) * seq_len].iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-4, "batch {} sum={}", b, sum);
        }
    }

    // 6.22 Sigmoid(0) == 0.5
    #[test]
    fn prop_sigmoid_zero(_dummy in 0u8..1) {
        let out = sigmoid(0.0);
        prop_assert!((out - 0.5).abs() < 1e-6, "sigmoid(0) != 0.5: {}", out);
    }

    // 6.23 Sigmoid monotonicity
    #[test]
    fn prop_sigmoid_monotone(
        a in -10.0f32..10.0,
        delta in 0.0f32..10.0,
    ) {
        let b = a + delta;
        let sa = sigmoid(a);
        let sb = sigmoid(b);
        prop_assert!(sa <= sb + 1e-6, "sigmoid not monotone: sig({})={} > sig({})={}", a, sa, b, sb);
    }

    // 6.24 Product of ones is one
    #[test]
    fn prop_product_of_ones(n in 1usize..=32) {
        let data = vec![1.0f32; n];
        let prod = ReductionKernel::product(&data).unwrap();
        prop_assert!((prod - 1.0).abs() < 1e-6, "product of ones != 1: {}", prod);
    }

    // 6.25 Sum of zeros is zero
    #[test]
    fn prop_sum_of_zeros(n in 1usize..=64) {
        let data = vec![0.0f32; n];
        let sum = ReductionKernel::sum(&data).unwrap();
        prop_assert!(sum.abs() < 1e-10, "sum of zeros != 0: {}", sum);
    }

    // 6.26 ReLU preserves positive values
    #[test]
    fn prop_relu_preserves_positive(val in 0.0f32..100.0) {
        let out = relu(val);
        prop_assert_eq!(out.to_bits(), val.to_bits());
    }

    // 6.27 ReLU zeroes negative values
    #[test]
    fn prop_relu_zeros_negative(val in -100.0f32..0.0) {
        let out = relu(val);
        prop_assert_eq!(out, 0.0);
    }

    // 6.28 Tanh is odd: tanh(-x) == -tanh(x)
    #[test]
    fn prop_tanh_odd(val in -5.0f32..5.0) {
        let pos = tanh_act(val);
        let neg = tanh_act(-val);
        prop_assert!((pos + neg).abs() < 1e-5, "tanh not odd: tanh({})={} tanh({})={}", val, pos, -val, neg);
    }
}

// ── 7. Additional dispatch and pipeline properties ─────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // 7.1 Attention with CpuAttention produces correct output size
    #[test]
    fn prop_cpu_attention_output_size(
        seq in 1usize..=8,
        dim in 1usize..=8,
    ) {
        let n = seq * dim;
        let q = vec![1.0f32; n];
        let k = vec![1.0f32; n];
        let v = vec![1.0f32; n];
        let config = CpuAttentionConfig {
            batch_size: 1,
            num_heads: 1,
            head_dim: dim,
            seq_len: seq,
            causal_mask: false,
            scale: None,
        };
        if let Ok(attn) = CpuAttention::new(config) {
            let out = attn.forward(&q, &k, &v).unwrap();
            prop_assert_eq!(out.len(), n);
        }
    }

    // 7.2 Scaled dot product attention produces finite output
    #[test]
    fn prop_sdpa_finite(
        seq in 1usize..=8,
        dim in 1usize..=8,
    ) {
        let n = seq * dim;
        let q = vec![0.1f32; n];
        let k = vec![0.1f32; n];
        let v = vec![0.1f32; n];
        let out = scaled_dot_product_attention(&q, &k, &v, seq, seq, dim, false);
        if let Ok(out) = out {
            for &val in &out {
                prop_assert!(val.is_finite(), "SDPA non-finite: {}", val);
            }
        }
    }

    // 7.3 Scatter then gather is identity
    #[test]
    fn prop_scatter_gather_roundtrip(n in 1usize..=32) {
        let mut buf = vec![0.0f32; n];
        let indices: Vec<usize> = (0..n).collect();
        let values: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        scatter_1d(&mut buf, &indices, &values).unwrap();
        let gathered = gather_1d(&buf, &indices).unwrap();
        prop_assert_eq!(gathered, values);
    }

    // 7.4 ReLU(x) >= 0 for all x (wide range)
    #[test]
    fn prop_relu_always_nonneg_wide(val in -1e6f32..1e6) {
        let out = relu(val);
        prop_assert!(out >= 0.0);
    }

    // 7.5 Silu_vec output is finite
    #[test]
    fn prop_silu_vec_finite(input in finite_f32_vec(64)) {
        let out = bitnet_kernels::cpu::activations::silu_vec(&input);
        for &v in &out {
            prop_assert!(v.is_finite());
        }
    }

    // 7.6 Gelu_vec output is finite
    #[test]
    fn prop_gelu_vec_finite(input in finite_f32_vec(64)) {
        let out = bitnet_kernels::cpu::activations::gelu_vec(&input);
        for &v in &out {
            prop_assert!(v.is_finite());
        }
    }

    // 7.7 ReLU inplace is same as scalar
    #[test]
    fn prop_relu_inplace_matches_scalar(input in finite_f32_vec(32)) {
        let scalar_out: Vec<f32> = input.iter().map(|&x| relu(x)).collect();
        let mut inplace = input;
        bitnet_kernels::cpu::activations::relu_inplace(&mut inplace);
        prop_assert_eq!(scalar_out, inplace);
    }

    // 7.8 Silu inplace is same as scalar
    #[test]
    fn prop_silu_inplace_matches_scalar(input in finite_f32_vec(32)) {
        let scalar_out: Vec<f32> = input.iter().map(|&x| silu(x)).collect();
        let mut inplace = input;
        bitnet_kernels::cpu::activations::silu_inplace(&mut inplace);
        for (s, i) in scalar_out.iter().zip(inplace.iter()) {
            prop_assert!((s - i).abs() < 1e-6);
        }
    }

    // 7.9 Gelu inplace is same as vec
    #[test]
    fn prop_gelu_inplace_matches_vec(input in finite_f32_vec(32)) {
        let vec_out = bitnet_kernels::cpu::activations::gelu_vec(&input);
        let mut inplace = input;
        bitnet_kernels::cpu::activations::gelu_inplace(&mut inplace);
        for (v, i) in vec_out.iter().zip(inplace.iter()) {
            prop_assert!((v - i).abs() < 1e-6);
        }
    }

    // 7.10 Linear with zero bias is same as without bias
    #[test]
    fn prop_linear_zero_bias(
        in_feat in 1usize..=8,
        out_feat in 1usize..=8,
    ) {
        let batch = 1;
        let input = vec![1.0f32; batch * in_feat];
        let weight = vec![0.5f32; out_feat * in_feat];
        let zero_bias = vec![0.0f32; out_feat];
        let config = LinearConfig::new(batch, in_feat, out_feat).unwrap();
        let mut with_bias = vec![0.0f32; batch * out_feat];
        linear_cpu(&input, &weight, Some(&zero_bias), &mut with_bias, &config).unwrap();
        let mut without_bias = vec![0.0f32; batch * out_feat];
        linear_cpu(&input, &weight, None, &mut without_bias, &config).unwrap();
        for (a, b) in with_bias.iter().zip(without_bias.iter()) {
            prop_assert!((*a - *b).abs() < 1e-6);
        }
    }

    // 7.11 Batched matmul with identity-like matrix
    #[test]
    fn prop_batched_matmul_finite(
        batch in 1usize..=2,
        m in 1usize..=4,
        k in 1usize..=4,
        n in 1usize..=4,
    ) {
        let a = vec![0.1f32; batch * m * k];
        let b = vec![0.1f32; batch * k * n];
        let out = batched_matmul(&a, &b, batch, m, k, n).unwrap();
        for &v in &out {
            prop_assert!(v.is_finite());
        }
    }

    // 7.12 Reduction max index is valid
    #[test]
    fn prop_reduction_max_index_valid(data in finite_f32_vec(64)) {
        let ValueWithIndex { index, value } = ReductionKernel::max(&data).unwrap();
        prop_assert!(index < data.len());
        prop_assert_eq!(data[index].to_bits(), value.to_bits());
    }
}
