//! Property-based tests — wave 24.
//!
//! Covers newer kernel modules: CUDA attention CPU fallbacks, CUDA softmax
//! CPU paths, CUDA elementwise CPU dispatch, CUDA KV cache buffer, CUDA
//! batch-norm kernel, CPU batch-norm forward/inference, CPU scatter-gather
//! round-trips, CPU KV cache append/slice, CPU dequantization, and CPU
//! quantize ↔ dequantize round-trips.
//!
//! 50+ property tests validating: attention score boundedness, output shape
//! preservation, softmax row sums, elementwise idempotence, KV cache
//! monotonicity, batch-norm normalization, scatter-gather fidelity,
//! dequantization tolerance, and more.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::activations::{relu, sigmoid, silu, tanh_act};
use bitnet_kernels::cpu::attention::{
    AttentionKernel, CpuAttention, CpuAttentionConfig, causal_mask, scaled_dot_product_attention,
};
use bitnet_kernels::cpu::batch::batched_softmax;
use bitnet_kernels::cpu::batch_norm::{BatchNormConfig, batch_norm_forward, batch_norm_inference};
use bitnet_kernels::cpu::dequant::{dequant_i2s_block, dequant_ternary, pack_ternary};
use bitnet_kernels::cpu::kv_cache::{
    KvCache, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear, kv_cache_memory_usage,
    kv_cache_slice,
};
use bitnet_kernels::cpu::quantize::{
    dequantize_symmetric_i8, quantize_binary, quantize_symmetric_i8, quantize_ternary,
};
use bitnet_kernels::cpu::scatter_gather::{gather_1d, scatter_1d, scatter_add};
use bitnet_kernels::cuda::attention::{
    AttentionConfig as CudaAttentionCfg, attention_cpu_fallback, batch_attention_cpu,
    multi_head_attention_cpu_fallback,
};
use bitnet_kernels::cuda::elementwise::{
    ElementwiseConfig, ElementwiseOp, elementwise_cpu_fallback, elementwise_unary_cpu,
    fused_elementwise_cpu, launch_elementwise_binary,
};
use bitnet_kernels::cuda::kv_cache::KvCacheBuffer;
use bitnet_kernels::cuda::softmax::{SoftmaxConfig, softmax_cpu};
use proptest::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn finite_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-10.0f32..10.0, 1..=max_len)
}

/// Generate a square matrix (seq_len × head_dim) with bounded values.
fn attention_triple(
    max_seq: usize,
    max_dim: usize,
) -> impl Strategy<Value = (usize, usize, Vec<f32>, Vec<f32>, Vec<f32>)> {
    (1..=max_seq, 1..=max_dim).prop_flat_map(move |(seq, dim)| {
        let n = seq * dim;
        (
            Just(seq),
            Just(dim),
            proptest::collection::vec(-5.0f32..5.0, n..=n),
            proptest::collection::vec(-5.0f32..5.0, n..=n),
            proptest::collection::vec(-5.0f32..5.0, n..=n),
        )
    })
}

// ── Property tests ──────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // ════════════════════════════════════════════════════════════════
    // 1. CUDA attention CPU fallbacks
    // ════════════════════════════════════════════════════════════════

    /// Attention output has the same length as input (seq_len × head_dim).
    #[test]
    fn prop_attention_output_shape(
        (seq, dim, q, k, v) in attention_triple(16, 32)
    ) {
        let config = CudaAttentionCfg::new(1, dim, seq, false).unwrap();
        let out = attention_cpu_fallback(&q, &k, &v, &config).unwrap();
        prop_assert_eq!(out.len(), seq * dim);
    }

    /// Attention output values are always finite.
    #[test]
    fn prop_attention_output_finite(
        (seq, dim, q, k, v) in attention_triple(12, 16)
    ) {
        let config = CudaAttentionCfg::new(1, dim, seq, false).unwrap();
        let out = attention_cpu_fallback(&q, &k, &v, &config).unwrap();
        for &val in &out {
            prop_assert!(val.is_finite(), "attention output contains non-finite: {val}");
        }
    }

    /// Causal attention output equals non-causal when seq_len == 1.
    #[test]
    fn prop_attention_causal_seq1(
        dim in 1usize..=32,
    ) {
        let n = dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let k = q.clone();
        let v = q.clone();
        let cfg_causal = CudaAttentionCfg::new(1, dim, 1, true).unwrap();
        let cfg_normal = CudaAttentionCfg::new(1, dim, 1, false).unwrap();
        let out_c = attention_cpu_fallback(&q, &k, &v, &cfg_causal).unwrap();
        let out_n = attention_cpu_fallback(&q, &k, &v, &cfg_normal).unwrap();
        for (a, b) in out_c.iter().zip(out_n.iter()) {
            prop_assert!((a - b).abs() < 1e-5, "causal != normal at seq=1");
        }
    }

    /// Multi-head attention output shape is num_heads × seq_len × head_dim.
    #[test]
    fn prop_multi_head_attention_shape(
        seq in 1usize..=8,
        dim in 1usize..=16,
        heads in 1usize..=4,
    ) {
        let n = seq * dim;
        let q: Vec<f32> = (0..(heads * n)).map(|i| (i as f32) * 0.01).collect();
        let k = q.clone();
        let v = q.clone();
        let config = CudaAttentionCfg::new(heads, dim, seq, false).unwrap();
        let out = multi_head_attention_cpu_fallback(&q, &k, &v, &config).unwrap();
        prop_assert_eq!(out.len(), heads * seq * dim);
    }

    /// Batch attention preserves total output size = batch * seq * dim.
    #[test]
    fn prop_batch_attention_output_size(
        batch in 1usize..=4,
        seq in 1usize..=8,
        dim in 1usize..=16,
    ) {
        let n = seq * dim;
        let q: Vec<f32> = (0..(batch * n)).map(|i| (i as f32) * 0.01).collect();
        let k = q.clone();
        let v = q.clone();
        let config = CudaAttentionCfg::new(1, dim, seq, false).unwrap();
        let out = batch_attention_cpu(&q, &k, &v, &config, batch).unwrap();
        prop_assert_eq!(out.len(), batch * seq * dim);
    }

    /// Attention scores after softmax are bounded in [0, 1] (verified via
    /// the output being a convex combination of values).
    #[test]
    fn prop_attention_output_bounded_by_values(
        (seq, dim, q, k, v) in attention_triple(8, 8)
    ) {
        let config = CudaAttentionCfg::new(1, dim, seq, false).unwrap();
        let out = attention_cpu_fallback(&q, &k, &v, &config).unwrap();
        let v_min = v.iter().copied().fold(f32::INFINITY, f32::min);
        let v_max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for &o in &out {
            prop_assert!(
                o >= v_min - 1e-4 && o <= v_max + 1e-4,
                "output {o} outside value range [{v_min}, {v_max}]"
            );
        }
    }

    /// CpuAttention single-head forward_single_head output shape.
    #[test]
    fn prop_cpu_attention_single_head_shape(
        seq in 1usize..=12,
        dim in 1usize..=16,
    ) {
        let config = CpuAttentionConfig {
            batch_size: 1,
            num_heads: 1,
            head_dim: dim,
            seq_len: seq,
            causal_mask: false,
            scale: Some(1.0 / (dim as f32).sqrt()),
        };
        let attn = CpuAttention::new(config).unwrap();
        let n = seq * dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let k = q.clone();
        let v = q.clone();
        let out = attn.forward_single_head(&q, &k, &v, seq, seq).unwrap();
        prop_assert_eq!(out.len(), seq * dim);
    }

    // ════════════════════════════════════════════════════════════════
    // 2. CUDA softmax CPU path
    // ════════════════════════════════════════════════════════════════

    /// Softmax row sums to ≈ 1.0 for each row.
    #[test]
    fn prop_softmax_row_sums_to_one(
        cols in 1usize..=128,
        rows in 1usize..=8,
    ) {
        let input: Vec<f32> = (0..(rows * cols)).map(|i| ((i % 37) as f32) * 0.3 - 5.0).collect();
        let mut output = vec![0.0f32; rows * cols];
        let config = SoftmaxConfig::for_shape(cols, rows).unwrap();
        softmax_cpu(&input, &mut output, &config).unwrap();
        for r in 0..rows {
            let sum: f32 = output[r * cols..(r + 1) * cols].iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-4, "row {r} sum = {sum}");
        }
    }

    /// All softmax outputs are in [0, 1].
    #[test]
    fn prop_softmax_values_in_unit_interval(
        cols in 1usize..=64,
        rows in 1usize..=4,
    ) {
        let input: Vec<f32> = (0..(rows * cols)).map(|i| (i as f32) * 0.5 - 10.0).collect();
        let mut output = vec![0.0f32; rows * cols];
        let config = SoftmaxConfig::for_shape(cols, rows).unwrap();
        softmax_cpu(&input, &mut output, &config).unwrap();
        for (i, &v) in output.iter().enumerate() {
            prop_assert!((0.0..=1.0 + 1e-6).contains(&v), "softmax[{i}] = {v} out of [0,1]");
        }
    }

    /// Softmax output length matches input length.
    #[test]
    fn prop_softmax_output_length(
        cols in 1usize..=64,
        rows in 1usize..=8,
    ) {
        let n = rows * cols;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; n];
        let config = SoftmaxConfig::for_shape(cols, rows).unwrap();
        softmax_cpu(&input, &mut output, &config).unwrap();
        prop_assert_eq!(output.len(), n);
    }

    /// Softmax is invariant under uniform translation of inputs.
    #[test]
    fn prop_softmax_translation_invariant(
        v in proptest::collection::vec(-10.0f32..10.0, 2..=64),
        shift in -5.0f32..5.0,
    ) {
        let n = v.len();
        let shifted: Vec<f32> = v.iter().map(|&x| x + shift).collect();
        let config = SoftmaxConfig::for_shape(n, 1).unwrap();
        let mut out_orig = vec![0.0f32; n];
        let mut out_shifted = vec![0.0f32; n];
        softmax_cpu(&v, &mut out_orig, &config).unwrap();
        softmax_cpu(&shifted, &mut out_shifted, &config).unwrap();
        for i in 0..n {
            prop_assert!(
                (out_orig[i] - out_shifted[i]).abs() < 1e-4,
                "translation invariance broken at {i}"
            );
        }
    }

    /// Batched softmax rows each sum to ≈ 1.
    #[test]
    fn prop_batched_softmax_rows_sum(
        batch in 1usize..=8,
        seq in 1usize..=32,
    ) {
        let input: Vec<f32> = (0..(batch * seq)).map(|i| (i as f32) * 0.2 - 3.0).collect();
        let out = batched_softmax(&input, batch, seq).unwrap();
        for b in 0..batch {
            let sum: f32 = out[b * seq..(b + 1) * seq].iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-4, "batch {b} sum = {sum}");
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 3. CPU batch norm
    // ════════════════════════════════════════════════════════════════

    /// Batch-norm output length equals input length.
    #[test]
    fn prop_batch_norm_output_length(
        c in 1usize..=8,
        batch in 2usize..=16,
    ) {
        let n = batch * c;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let gamma = vec![1.0f32; c];
        let beta = vec![0.0f32; c];
        let r_mean = vec![0.0f32; c];
        let r_var = vec![1.0f32; c];
        let config = BatchNormConfig::new(c);
        let (output, _, _) =
            batch_norm_forward(&input, &gamma, &beta, &r_mean, &r_var, &config).unwrap();
        prop_assert_eq!(output.len(), n);
    }

    /// Batch-norm with identity affine (gamma=1, beta=0) gives near-zero
    /// mean per channel.
    #[test]
    fn prop_batch_norm_zero_mean(
        c in 1usize..=4,
        batch in 4usize..=16,
    ) {
        let n = batch * c;
        let input: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 19) as f32 - 9.0).collect();
        let gamma = vec![1.0f32; c];
        let beta = vec![0.0f32; c];
        let r_mean = vec![0.0f32; c];
        let r_var = vec![1.0f32; c];
        let config = BatchNormConfig::new(c);
        let (output, _, _) =
            batch_norm_forward(&input, &gamma, &beta, &r_mean, &r_var, &config).unwrap();
        for ch in 0..c {
            let mean: f32 =
                (0..batch).map(|b| output[b * c + ch]).sum::<f32>() / batch as f32;
            prop_assert!(
                mean.abs() < 0.3,
                "channel {ch} mean = {mean}, expected ≈ 0"
            );
        }
    }

    /// Batch-norm inference output is finite for finite inputs.
    #[test]
    fn prop_batch_norm_inference_finite(
        c in 1usize..=8,
        batch in 1usize..=8,
    ) {
        let n = batch * c;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.2 - 1.0).collect();
        let gamma = vec![1.0f32; c];
        let beta = vec![0.0f32; c];
        let r_mean = vec![0.0f32; c];
        let r_var = vec![1.0f32; c];
        let out = batch_norm_inference(&input, &gamma, &beta, &r_mean, &r_var, 1e-5).unwrap();
        for &v in &out {
            prop_assert!(v.is_finite(), "batch_norm_inference non-finite: {v}");
        }
    }

    /// Batch-norm running stats update stays finite.
    #[test]
    fn prop_batch_norm_running_stats_finite(
        c in 1usize..=4,
        batch in 2usize..=8,
    ) {
        let n = batch * c;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let gamma = vec![1.0f32; c];
        let beta = vec![0.0f32; c];
        let r_mean = vec![0.0f32; c];
        let r_var = vec![1.0f32; c];
        let config = BatchNormConfig::new(c);
        let (_, upd_mean, upd_var) =
            batch_norm_forward(&input, &gamma, &beta, &r_mean, &r_var, &config).unwrap();
        for &m in &upd_mean {
            prop_assert!(m.is_finite(), "updated mean non-finite: {m}");
        }
        for &v in &upd_var {
            prop_assert!(v.is_finite(), "updated var non-finite: {v}");
        }
    }

    /// Batch-norm with gamma=2, beta=3 scales and shifts correctly.
    #[test]
    fn prop_batch_norm_affine_scaling(
        c in 1usize..=4,
        batch in 4usize..=12,
    ) {
        let n = batch * c;
        let input: Vec<f32> = (0..n).map(|i| ((i * 3 + 1) % 11) as f32 - 5.0).collect();
        let gamma = vec![2.0f32; c];
        let beta = vec![3.0f32; c];
        let r_mean = vec![0.0f32; c];
        let r_var = vec![1.0f32; c];
        let config = BatchNormConfig::new(c);
        let (output, _, _) =
            batch_norm_forward(&input, &gamma, &beta, &r_mean, &r_var, &config).unwrap();
        // With gamma=2, beta=3, the per-channel mean should be ≈ 3.0
        for ch in 0..c {
            let mean: f32 =
                (0..batch).map(|b| output[b * c + ch]).sum::<f32>() / batch as f32;
            prop_assert!(
                (mean - 3.0).abs() < 0.5,
                "affine channel {ch} mean = {mean}, expected ≈ 3.0"
            );
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 4. CUDA elementwise CPU dispatch
    // ════════════════════════════════════════════════════════════════

    /// Elementwise add is commutative.
    #[test]
    fn prop_elementwise_add_commutative(
        a in finite_f32_vec(64),
    ) {
        let b: Vec<f32> = a.iter().map(|x| x * 0.5 + 1.0).collect();
        let ab = elementwise_cpu_fallback(&a, &b, ElementwiseOp::Add).unwrap();
        let ba = elementwise_cpu_fallback(&b, &a, ElementwiseOp::Add).unwrap();
        for (i, (&x, &y)) in ab.iter().zip(ba.iter()).enumerate() {
            prop_assert!((x - y).abs() < 1e-5, "add not commutative at {i}");
        }
    }

    /// Elementwise mul is commutative.
    #[test]
    fn prop_elementwise_mul_commutative(
        a in finite_f32_vec(64),
    ) {
        let b: Vec<f32> = a.iter().map(|x| x * 0.3 + 0.5).collect();
        let ab = elementwise_cpu_fallback(&a, &b, ElementwiseOp::Mul).unwrap();
        let ba = elementwise_cpu_fallback(&b, &a, ElementwiseOp::Mul).unwrap();
        for (i, (&x, &y)) in ab.iter().zip(ba.iter()).enumerate() {
            prop_assert!((x - y).abs() < 1e-5, "mul not commutative at {i}");
        }
    }

    /// Elementwise sub(a, a) == 0.
    #[test]
    fn prop_elementwise_sub_self_is_zero(
        a in finite_f32_vec(64),
    ) {
        let result = elementwise_cpu_fallback(&a, &a, ElementwiseOp::Sub).unwrap();
        for (i, &v) in result.iter().enumerate() {
            prop_assert!(v.abs() < 1e-6, "a - a != 0 at {i}: {v}");
        }
    }

    /// Unary relu is idempotent: relu(relu(x)) == relu(x).
    #[test]
    fn prop_unary_relu_idempotent(
        v in finite_f32_vec(64),
    ) {
        let n = v.len();
        let config = ElementwiseConfig::new(n, ElementwiseOp::Relu).unwrap();
        let first = elementwise_unary_cpu(&v, &config).unwrap();
        let second = elementwise_unary_cpu(&first, &config).unwrap();
        for (i, (&a, &b)) in first.iter().zip(second.iter()).enumerate() {
            prop_assert!((a - b).abs() < 1e-6, "relu not idempotent at {i}");
        }
    }

    /// Unary sigmoid output is in (0, 1) for moderate values.
    #[test]
    fn prop_unary_sigmoid_bounded(
        v in proptest::collection::vec(-10.0f32..10.0, 1..=64),
    ) {
        let n = v.len();
        let config = ElementwiseConfig::new(n, ElementwiseOp::Sigmoid).unwrap();
        let out = elementwise_unary_cpu(&v, &config).unwrap();
        for (i, &val) in out.iter().enumerate() {
            prop_assert!(val > 0.0 && val < 1.0, "sigmoid[{i}] = {val} not in (0,1)");
        }
    }

    /// Unary tanh output is in (-1, 1).
    #[test]
    fn prop_unary_tanh_bounded(
        v in proptest::collection::vec(-8.0f32..8.0, 1..=64),
    ) {
        let n = v.len();
        let config = ElementwiseConfig::new(n, ElementwiseOp::Tanh).unwrap();
        let out = elementwise_unary_cpu(&v, &config).unwrap();
        for (i, &val) in out.iter().enumerate() {
            prop_assert!(val > -1.0 && val < 1.0, "tanh[{i}] = {val} not in (-1,1)");
        }
    }

    /// Fused add-mul output is finite for finite inputs.
    #[test]
    fn prop_fused_elementwise_finite(
        v in finite_f32_vec(64),
    ) {
        let bias: Vec<f32> = v.iter().map(|x| x * 0.1).collect();
        let scale: Vec<f32> = v.iter().map(|_| 1.0f32).collect();
        let out = fused_elementwise_cpu(&v, &bias, &scale).unwrap();
        for (i, &val) in out.iter().enumerate() {
            prop_assert!(val.is_finite(), "fused output[{i}] non-finite: {val}");
        }
    }

    /// launch_elementwise_binary Add matches manual addition.
    #[test]
    fn prop_launch_binary_add_correctness(
        a in finite_f32_vec(32),
    ) {
        let b: Vec<f32> = a.iter().map(|x| x * 0.5).collect();
        let result = launch_elementwise_binary(&a, &b, ElementwiseOp::Add).unwrap();
        for (i, ((&ai, &bi), &ri)) in a.iter().zip(b.iter()).zip(result.iter()).enumerate() {
            prop_assert!((ri - (ai + bi)).abs() < 1e-5, "mismatch at {i}");
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 5. CPU scatter-gather
    // ════════════════════════════════════════════════════════════════

    /// gather_1d(scatter_1d(data, idx, vals)) recovers the scattered values.
    #[test]
    fn prop_scatter_then_gather_roundtrip(
        n in 4usize..=32,
    ) {
        let mut data = vec![0.0f32; n];
        let indices: Vec<usize> = (0..n.min(8)).collect();
        let values: Vec<f32> = indices.iter().map(|&i| (i as f32) * 1.5 + 0.1).collect();
        scatter_1d(&mut data, &indices, &values).unwrap();
        let gathered = gather_1d(&data, &indices).unwrap();
        for (i, (&g, &v)) in gathered.iter().zip(values.iter()).enumerate() {
            prop_assert!((g - v).abs() < 1e-6, "roundtrip mismatch at {i}");
        }
    }

    /// scatter_add accumulates correctly for non-overlapping indices.
    #[test]
    fn prop_scatter_add_non_overlapping(
        n in 4usize..=32,
    ) {
        let mut data = vec![1.0f32; n];
        let k = n.min(8);
        let indices: Vec<usize> = (0..k).collect();
        let values: Vec<f32> = vec![2.0f32; k];
        scatter_add(&mut data, &indices, &values).unwrap();
        for &idx in &indices {
            prop_assert!(
                (data[idx] - 3.0).abs() < 1e-6,
                "scatter_add: expected 3.0, got {}", data[idx]
            );
        }
    }

    /// gather_1d output length matches indices length.
    #[test]
    fn prop_gather_output_length(
        n in 2usize..=32,
    ) {
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let k = n.min(8);
        let indices: Vec<usize> = (0..k).collect();
        let out = gather_1d(&data, &indices).unwrap();
        prop_assert_eq!(out.len(), k);
    }

    // ════════════════════════════════════════════════════════════════
    // 6. CPU KV cache
    // ════════════════════════════════════════════════════════════════

    /// Appending to KV cache increases seq_len by 1.
    #[test]
    fn prop_kv_cache_append_increments_seq_len(
        heads in 1usize..=4,
        dim in 1usize..=8,
    ) {
        let config = KvCacheConfig {
            num_layers: 1,
            num_heads: heads,
            head_dim: dim,
            max_seq_len: 16,
            dtype: KvDtype::F32,
        };
        let mut cache = KvCache::new(config).unwrap();
        let tok_size = heads * dim;
        let keys = vec![1.0f32; tok_size];
        let values = vec![2.0f32; tok_size];
        prop_assert_eq!(cache.seq_len(0).unwrap(), 0);
        kv_cache_append(&mut cache, 0, &keys, &values).unwrap();
        prop_assert_eq!(cache.seq_len(0).unwrap(), 1);
        kv_cache_append(&mut cache, 0, &keys, &values).unwrap();
        prop_assert_eq!(cache.seq_len(0).unwrap(), 2);
    }

    /// KV cache slice returns correct data after append.
    #[test]
    fn prop_kv_cache_slice_after_append(
        heads in 1usize..=2,
        dim in 1usize..=4,
    ) {
        let config = KvCacheConfig {
            num_layers: 1,
            num_heads: heads,
            head_dim: dim,
            max_seq_len: 8,
            dtype: KvDtype::F32,
        };
        let mut cache = KvCache::new(config).unwrap();
        let tok_size = heads * dim;
        let keys: Vec<f32> = (0..tok_size).map(|i| i as f32).collect();
        let values: Vec<f32> = (0..tok_size).map(|i| (i as f32) * 10.0).collect();
        kv_cache_append(&mut cache, 0, &keys, &values).unwrap();
        let (k_slice, v_slice) = kv_cache_slice(&cache, 0, 0, 1).unwrap();
        for (i, (&k, &expected)) in k_slice.iter().zip(keys.iter()).enumerate() {
            prop_assert!((k - expected).abs() < 1e-6, "key mismatch at {i}");
        }
        for (i, (&v, &expected)) in v_slice.iter().zip(values.iter()).enumerate() {
            prop_assert!((v - expected).abs() < 1e-6, "value mismatch at {i}");
        }
    }

    /// Clearing KV cache resets seq_len to 0.
    #[test]
    fn prop_kv_cache_clear_resets(
        layers in 1usize..=4,
    ) {
        let config = KvCacheConfig {
            num_layers: layers,
            num_heads: 2,
            head_dim: 4,
            max_seq_len: 8,
            dtype: KvDtype::F32,
        };
        let mut cache = KvCache::new(config).unwrap();
        let tok = vec![1.0f32; 8];
        for layer_idx in 0..layers {
            kv_cache_append(&mut cache, layer_idx, &tok, &tok).unwrap();
        }
        kv_cache_clear(&mut cache);
        for layer_idx in 0..layers {
            prop_assert_eq!(cache.seq_len(layer_idx).unwrap(), 0, "layer not cleared");
        }
    }

    /// KV cache memory usage is positive for any non-empty config.
    #[test]
    fn prop_kv_cache_memory_positive(
        layers in 1usize..=4,
        heads in 1usize..=4,
        dim in 1usize..=8,
    ) {
        let config = KvCacheConfig {
            num_layers: layers,
            num_heads: heads,
            head_dim: dim,
            max_seq_len: 4,
            dtype: KvDtype::F32,
        };
        let cache = KvCache::new(config).unwrap();
        prop_assert!(kv_cache_memory_usage(&cache) > 0);
    }

    // ════════════════════════════════════════════════════════════════
    // 7. CUDA KV cache buffer
    // ════════════════════════════════════════════════════════════════

    /// KvCacheBuffer append_kv increases layer_len.
    #[test]
    fn prop_cuda_kv_buffer_append_increases_len(
        heads in 1usize..=4,
        dim in 1usize..=8,
    ) {
        let config = bitnet_kernels::cuda::kv_cache::KvCacheConfig::new(
            1, heads, dim, 16, bitnet_kernels::cuda::kv_cache::CacheDtype::F32,
        ).unwrap();
        let mut buf = KvCacheBuffer::new(config);
        let tok = vec![1.0f32; heads * dim];
        prop_assert_eq!(buf.layer_len(0).unwrap(), 0);
        buf.append_kv(0, 0, &tok, &tok).unwrap();
        prop_assert_eq!(buf.layer_len(0).unwrap(), 1);
    }

    /// KvCacheBuffer get_kv returns correct data after append.
    #[test]
    fn prop_cuda_kv_buffer_get_roundtrip(
        heads in 1usize..=2,
        dim in 1usize..=4,
    ) {
        let config = bitnet_kernels::cuda::kv_cache::KvCacheConfig::new(
            1, heads, dim, 8, bitnet_kernels::cuda::kv_cache::CacheDtype::F32,
        ).unwrap();
        let mut buf = KvCacheBuffer::new(config);
        let tok_size = heads * dim;
        let keys: Vec<f32> = (0..tok_size).map(|i| i as f32).collect();
        let vals: Vec<f32> = (0..tok_size).map(|i| (i as f32) + 100.0).collect();
        buf.append_kv(0, 0, &keys, &vals).unwrap();
        let (k_out, v_out) = buf.get_kv(0, 0, 1).unwrap();
        for (i, (&k, &expected)) in k_out.iter().zip(keys.iter()).enumerate() {
            prop_assert!((k - expected).abs() < 1e-6, "cuda kv key mismatch at {i}");
        }
        for (i, (&v, &expected)) in v_out.iter().zip(vals.iter()).enumerate() {
            prop_assert!((v - expected).abs() < 1e-6, "cuda kv val mismatch at {i}");
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 8. CPU quantize ↔ dequantize round-trips
    // ════════════════════════════════════════════════════════════════

    /// Symmetric i8 quant→dequant round-trip within tolerance.
    #[test]
    fn prop_symmetric_i8_roundtrip_tolerance(
        v in proptest::collection::vec(-5.0f32..5.0, 4..=64),
    ) {
        let (quantized, scale) = quantize_symmetric_i8(&v, 8);
        let recovered = dequantize_symmetric_i8(&quantized, scale);
        for (i, (&orig, &rec)) in v.iter().zip(recovered.iter()).enumerate() {
            let tol = scale * 0.6;
            prop_assert!(
                (orig - rec).abs() <= tol,
                "quant roundtrip at {i}: orig={orig}, rec={rec}, tol={tol}"
            );
        }
    }

    /// Quantize ternary produces only {-1, 0, 1} values.
    #[test]
    fn prop_quantize_ternary_values(
        v in proptest::collection::vec(-10.0f32..10.0, 1..=64),
    ) {
        let q = quantize_ternary(&v, 0.5);
        for (i, &val) in q.iter().enumerate() {
            prop_assert!(
                val == -1 || val == 0 || val == 1,
                "ternary[{i}] = {val}, expected {{-1,0,1}}"
            );
        }
    }

    /// Quantize binary produces only {-1, 1} values.
    #[test]
    fn prop_quantize_binary_values(
        v in proptest::collection::vec(-10.0f32..10.0, 1..=64),
    ) {
        let q = quantize_binary(&v);
        for (i, &val) in q.iter().enumerate() {
            prop_assert!(
                val == -1 || val == 1,
                "binary[{i}] = {val}, expected {{-1,1}}"
            );
        }
    }

    /// Ternary pack→dequant round-trip preserves sign.
    #[test]
    fn prop_ternary_pack_dequant_sign(
        v in proptest::collection::vec(-10.0f32..10.0, 4..=32),
    ) {
        let (packed, scale) = pack_ternary(&v, 0.5);
        let recovered = dequant_ternary(&packed, scale);
        for (i, (&orig, &rec)) in v.iter().zip(recovered.iter()).enumerate() {
            if orig.abs() > 0.5 {
                prop_assert!(
                    orig.signum() == rec.signum() || rec == 0.0,
                    "sign mismatch at {i}: orig={orig}, rec={rec}"
                );
            }
        }
    }

    /// dequant_i2s_block output length matches expected block_size.
    #[test]
    fn prop_dequant_i2s_block_output_len(
        block_size in prop::sample::select(vec![32usize, 64, 128, 256]),
    ) {
        let packed_bytes = block_size / 4;
        let packed = vec![0xAAu8; packed_bytes];
        let result = dequant_i2s_block(&packed, 1.0, block_size);
        prop_assert!(result.is_ok());
        let out = result.unwrap();
        prop_assert_eq!(out.len(), block_size);
    }

    // ════════════════════════════════════════════════════════════════
    // 9. CPU activation idempotence / bounds
    // ════════════════════════════════════════════════════════════════

    /// abs(abs(x)) == abs(x) via relu double-application.
    #[test]
    fn prop_relu_double_application(
        x in -100.0f32..100.0,
    ) {
        let r1 = relu(x);
        let r2 = relu(r1);
        prop_assert_eq!(r1, r2);
    }

    /// sigmoid is monotonically non-decreasing.
    #[test]
    fn prop_sigmoid_monotone(
        a in -50.0f32..50.0,
        b in -50.0f32..50.0,
    ) {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        prop_assert!(sigmoid(lo) <= sigmoid(hi) + 1e-6);
    }

    /// silu(0) == 0.
    #[test]
    fn prop_silu_zero_is_zero(_dummy in 0u8..1) {
        prop_assert!(silu(0.0).abs() < 1e-6);
    }

    /// tanh output bounded in (-1, 1) for moderate inputs.
    #[test]
    fn prop_tanh_bounded(x in -8.0f32..8.0) {
        let t = tanh_act(x);
        prop_assert!(t > -1.0 && t < 1.0, "tanh({x}) = {t}");
    }

    // ════════════════════════════════════════════════════════════════
    // 10. Causal mask properties
    // ════════════════════════════════════════════════════════════════

    /// Causal mask is lower-triangular (upper values are -inf).
    #[test]
    fn prop_causal_mask_lower_triangular(
        seq in 1usize..=32,
    ) {
        let mask = causal_mask(seq);
        prop_assert_eq!(mask.len(), seq * seq);
        for i in 0..seq {
            for j in 0..seq {
                let val = mask[i * seq + j];
                if j > i {
                    prop_assert!(
                        val == f32::NEG_INFINITY,
                        "mask[{i},{j}] = {val}, expected -inf"
                    );
                } else {
                    prop_assert!(
                        val == 0.0,
                        "mask[{i},{j}] = {val}, expected 0.0"
                    );
                }
            }
        }
    }

    /// Causal mask has exactly seq*(seq+1)/2 zero entries.
    #[test]
    fn prop_causal_mask_zero_count(
        seq in 1usize..=32,
    ) {
        let mask = causal_mask(seq);
        let zeros = mask.iter().filter(|&&v| v == 0.0).count();
        prop_assert_eq!(zeros, seq * (seq + 1) / 2);
    }

    /// AttentionKernel::scaled_dot_product output is finite.
    #[test]
    fn prop_attention_kernel_sdp_finite(
        seq in 1usize..=8,
        dim in 1usize..=8,
    ) {
        let n = seq * dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let k = q.clone();
        let v = q.clone();
        let scale = 1.0 / (dim as f32).sqrt();
        let out = AttentionKernel::scaled_dot_product(
            &q, &k, &v, None, scale, seq, seq, dim,
        ).unwrap();
        for &val in &out {
            prop_assert!(val.is_finite(), "sdp output non-finite: {val}");
        }
    }

    /// scaled_dot_product_attention output shape.
    #[test]
    fn prop_scaled_dot_product_fn_shape(
        seq in 1usize..=8,
        dim in 1usize..=8,
    ) {
        let n = seq * dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02).collect();
        let k = q.clone();
        let v = q.clone();
        let out = scaled_dot_product_attention(&q, &k, &v, seq, seq, dim, true).unwrap();
        prop_assert_eq!(out.len(), seq * dim);
    }

    // ════════════════════════════════════════════════════════════════
    // 11. Additional invariants
    // ════════════════════════════════════════════════════════════════

    /// Softmax with temperature=1 matches default (no temperature scaling).
    #[test]
    fn prop_softmax_temperature_one_is_default(
        cols in 2usize..=32,
    ) {
        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let config_default = SoftmaxConfig::for_shape(cols, 1).unwrap();
        let config_temp1 = SoftmaxConfig::for_shape(cols, 1).unwrap().with_temperature(1.0).unwrap();
        let mut out_default = vec![0.0f32; cols];
        let mut out_temp1 = vec![0.0f32; cols];
        softmax_cpu(&input, &mut out_default, &config_default).unwrap();
        softmax_cpu(&input, &mut out_temp1, &config_temp1).unwrap();
        for i in 0..cols {
            prop_assert!(
                (out_default[i] - out_temp1[i]).abs() < 1e-6,
                "temperature=1 differs from default at {i}"
            );
        }
    }

    /// Elementwise add with zero vector is identity.
    #[test]
    fn prop_elementwise_add_zero_identity(
        a in finite_f32_vec(64),
    ) {
        let zero = vec![0.0f32; a.len()];
        let result = elementwise_cpu_fallback(&a, &zero, ElementwiseOp::Add).unwrap();
        for (i, (&r, &ai)) in result.iter().zip(a.iter()).enumerate() {
            prop_assert!((r - ai).abs() < 1e-6, "add(a,0) != a at {i}");
        }
    }

    /// Elementwise mul with ones vector is identity.
    #[test]
    fn prop_elementwise_mul_ones_identity(
        a in finite_f32_vec(64),
    ) {
        let ones = vec![1.0f32; a.len()];
        let result = elementwise_cpu_fallback(&a, &ones, ElementwiseOp::Mul).unwrap();
        for (i, (&r, &ai)) in result.iter().zip(a.iter()).enumerate() {
            prop_assert!((r - ai).abs() < 1e-6, "mul(a,1) != a at {i}");
        }
    }
}
