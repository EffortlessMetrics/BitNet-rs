#![allow(dead_code, unused_imports, unused_variables, unused_unsafe, unsafe_op_in_unsafe_fn)]
//! Property-based tests — wave 21.
//!
//! Comprehensive CPU kernel invariants: fusion correctness, attention mask
//! algebra, KV-cache monotonicity, transpose involution, concat/split
//! round-trips, quantization error bounds, gating symmetry, pooling
//! monotonicity, linear projection shapes, batch operations, FFN configs,
//! embedding bag aggregation, SIMD-scalar agreement, and residual identity.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::activations::{gelu, sigmoid, silu, tanh_act};
use bitnet_kernels::cpu::attention_mask::{
    combine_masks, create_causal_mask, create_padding_mask, create_sliding_window_mask,
};
use bitnet_kernels::cpu::batch::{batched_add, batched_layer_norm, batched_softmax};
use bitnet_kernels::cpu::concat::ConcatKernel;
use bitnet_kernels::cpu::embedding::{
    embedding_lookup, normalize_embeddings, positional_embedding, positional_encoding,
};
use bitnet_kernels::cpu::fusion::{
    FusionConfig, fused_add_normalize, fused_scale_add, fused_softmax_mask,
};
use bitnet_kernels::cpu::gating::{geglu, reglu, swiglu};
use bitnet_kernels::cpu::kv_cache::{
    KvCache, KvCacheConfig, kv_cache_append, kv_cache_memory_usage,
};
use bitnet_kernels::cpu::layer_norm::{GroupNormConfig, LayerNormConfig, group_norm, rms_norm};
use bitnet_kernels::cpu::linear::{LinearConfig, linear_forward};
use bitnet_kernels::cpu::loss::{
    LossReduction, cosine_similarity_loss, l1_loss, mse_loss, smooth_l1_loss,
};
use bitnet_kernels::cpu::pooling::{PoolConfig, PoolType, adaptive_avg_pool_1d, pool_1d};
use bitnet_kernels::cpu::quantize::{
    compute_quantization_error, dequantize_symmetric_i8, quantize_binary, quantize_symmetric_i8,
    quantize_ternary,
};
use bitnet_kernels::cpu::reduction::ReductionKernel;
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled};
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, compute_frequencies};
use bitnet_kernels::cpu::scatter_gather::{gather_1d, scatter_1d};
use bitnet_kernels::cpu::simd_math::{
    simd_dot_product, simd_l2_norm, simd_vector_add, simd_vector_mul, simd_vector_scale,
};
use bitnet_kernels::cpu::transpose::TransposeKernel;
use proptest::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn finite_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-50.0f32..50.0, 1..=max_len)
}

// ── Property tests ──────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // ── Fusion operations ───────────────────────────────────────────

    /// fused_scale_add with zero second operand is identity.
    #[test]
    fn prop_fused_scale_add_zero_identity(
        input in finite_f32_vec(64)
    ) {
        let b = vec![0.0; input.len()];
        let result = fused_scale_add(&input, &b, 1.0).unwrap();
        for (r, &i) in result.iter().zip(input.iter()) {
            prop_assert!((r - i).abs() < 1e-5, "identity: a + 0*b should equal a");
        }
    }

    /// fused_scale_add with scale=1 is element-wise addition.
    #[test]
    fn prop_fused_scale_add_is_addition(
        a in finite_f32_vec(64),
    ) {
        let b: Vec<f32> = a.iter().map(|x| x * 0.5).collect();
        let result = fused_scale_add(&a, &b, 1.0).unwrap();
        for (i, (r, (&ai, &bi))) in result.iter().zip(a.iter().zip(b.iter())).enumerate() {
            prop_assert!((r - (ai + bi)).abs() < 1e-4, "mismatch at {i}");
        }
    }

    /// fused_scale_add dimension mismatch always errors.
    #[test]
    fn prop_fused_scale_add_mismatch_errors(
        n in 1usize..32,
        m in 1usize..32,
        scale in -10.0f32..10.0
    ) {
        prop_assume!(n != m);
        let a = vec![1.0; n];
        let b = vec![1.0; m];
        let r = fused_scale_add(&a, &b, scale);
        prop_assert!(r.is_err());
    }

    /// fused_softmax_mask output sums to 1 for non-masked inputs.
    #[test]
    fn prop_fused_softmax_mask_sums_to_one(input in finite_f32_vec(64)) {
        let mask = vec![0.0; input.len()];
        let result = fused_softmax_mask(&input, &mask, 1.0).unwrap();
        let sum: f32 = result.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-4, "softmax sum = {}", sum);
    }

    /// fused_softmax_mask outputs are all non-negative (probabilities).
    #[test]
    fn prop_fused_softmax_mask_non_negative(input in finite_f32_vec(64)) {
        let mask = vec![0.0; input.len()];
        let result = fused_softmax_mask(&input, &mask, 1.0).unwrap();
        for &v in &result {
            prop_assert!(v >= 0.0, "negative softmax output: {}", v);
        }
    }

    /// fused_add_normalize produces finite output.
    #[test]
    fn prop_fused_add_normalize_finite(
        a in finite_f32_vec(32),
    ) {
        let b = vec![0.0; a.len()];
        let gamma = vec![1.0; a.len()];
        if let Ok(result) = fused_add_normalize(&a, &b, &gamma, 1e-5) {
            for &v in &result {
                prop_assert!(v.is_finite(), "non-finite output: {}", v);
            }
        }
    }

    /// FusionConfig::disabled sets min_fusion_size to MAX.
    #[test]
    fn prop_fusion_config_disabled_blocks_all(_dummy in 0u8..1) {
        let cfg = FusionConfig::disabled();
        prop_assert_eq!(cfg.min_fusion_size, usize::MAX);
        prop_assert!(!cfg.enable_rmsnorm_linear);
        prop_assert!(!cfg.enable_gelu_linear);
        prop_assert!(!cfg.enable_softmax_mask);
    }

    // ── Attention mask algebra ──────────────────────────────────────

    /// Causal mask has exactly n*(n+1)/2 open (0.0) positions.
    #[test]
    fn prop_causal_mask_open_count(seq_len in 1usize..32) {
        let mask = create_causal_mask(seq_len);
        let open = mask.iter().filter(|&&v| v == 0.0).count();
        prop_assert_eq!(open, seq_len * (seq_len + 1) / 2);
    }

    /// Combining a mask with the zero mask is identity.
    #[test]
    fn prop_combine_masks_identity(seq_len in 1usize..16) {
        let m = create_causal_mask(seq_len);
        let zero = vec![0.0; seq_len * seq_len];
        let combined = combine_masks(&m, &zero, seq_len);
        for (i, (&c, &o)) in combined.iter().zip(m.iter()).enumerate() {
            prop_assert!(
                (c == o) || (c.is_infinite() && o.is_infinite()),
                "mismatch at {i}: combined={c}, original={o}"
            );
        }
    }

    /// Sliding window mask with window >= seq_len equals causal mask.
    #[test]
    fn prop_sliding_window_large_equals_causal(seq_len in 1usize..16) {
        let causal = create_causal_mask(seq_len);
        let sliding = create_sliding_window_mask(seq_len, seq_len + 1);
        prop_assert_eq!(causal, sliding);
    }

    /// Padding mask: valid count = sum of clamped lengths.
    #[test]
    fn prop_padding_mask_valid_count(
        lengths in proptest::collection::vec(0usize..20, 1..8),
        max_len in 1usize..16,
    ) {
        let mask = create_padding_mask(&lengths, max_len);
        let valid = mask.iter().filter(|&&v| v == 0.0).count();
        let expected: usize = lengths.iter().map(|&l| l.min(max_len)).sum();
        prop_assert_eq!(valid, expected);
    }

    /// Sliding window open positions: monotonically increase with window size.
    #[test]
    fn prop_sliding_window_monotone_in_window(
        seq_len in 2usize..16,
        w1 in 1usize..16,
        w2 in 1usize..16,
    ) {
        let (lo, hi) = if w1 <= w2 { (w1, w2) } else { (w2, w1) };
        let m_lo = create_sliding_window_mask(seq_len, lo);
        let m_hi = create_sliding_window_mask(seq_len, hi);
        let open_lo = m_lo.iter().filter(|&&v| v == 0.0).count();
        let open_hi = m_hi.iter().filter(|&&v| v == 0.0).count();
        prop_assert!(open_lo <= open_hi, "open({lo})={open_lo} > open({hi})={open_hi}");
    }

    // ── KV cache ────────────────────────────────────────────────────

    /// KV cache seq_len increases monotonically after appends.
    #[test]
    fn prop_kv_cache_seq_len_monotone(
        num_layers in 1usize..4,
        head_dim in 1usize..8,
        n_heads in 1usize..4,
        appends in 1usize..5,
    ) {
        let cfg = KvCacheConfig {
            num_layers,
            num_heads: n_heads,
            head_dim,
            max_seq_len: 64,
            dtype: bitnet_kernels::cpu::kv_cache::KvDtype::F32,
        };
        if cfg.validate().is_err() { return Ok(()); }
        let mut cache = KvCache::new(cfg).unwrap();
        let step_size = n_heads * head_dim;
        let mut prev_len = 0;
        for _ in 0..appends {
            let k = vec![0.1f32; step_size];
            let v = vec![0.2f32; step_size];
            let _ = kv_cache_append(&mut cache, 0, &k, &v);
            let cur = cache.seq_len(0).unwrap_or(0);
            prop_assert!(cur >= prev_len, "seq_len decreased: {prev_len} -> {cur}");
            prev_len = cur;
        }
    }

    /// KV cache memory usage is non-zero after creation with valid config.
    #[test]
    fn prop_kv_cache_memory_positive(
        num_layers in 1usize..4,
        head_dim in 1usize..8,
        n_heads in 1usize..4,
    ) {
        let cfg = KvCacheConfig {
            num_layers,
            num_heads: n_heads,
            head_dim,
            max_seq_len: 16,
            dtype: bitnet_kernels::cpu::kv_cache::KvDtype::F32,
        };
        if cfg.validate().is_err() { return Ok(()); }
        let cache = KvCache::new(cfg).unwrap();
        let mem = kv_cache_memory_usage(&cache);
        prop_assert!(mem > 0, "memory usage should be positive");
    }

    // ── Transpose involution ────────────────────────────────────────

    /// Transposing a matrix twice yields the original.
    #[test]
    fn prop_transpose_2d_involution(
        rows in 1usize..16,
        cols in 1usize..16,
    ) {
        let n = rows * cols;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let t1 = TransposeKernel::transpose_2d(&data, rows, cols).unwrap();
        let t2 = TransposeKernel::transpose_2d(&t1, cols, rows).unwrap();
        prop_assert_eq!(data, t2);
    }

    /// Transpose preserves element count.
    #[test]
    fn prop_transpose_preserves_count(
        rows in 1usize..16,
        cols in 1usize..16,
    ) {
        let n = rows * cols;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let t = TransposeKernel::transpose_2d(&data, rows, cols).unwrap();
        prop_assert_eq!(t.len(), n);
    }

    /// Squeeze removes all size-1 dimensions.
    #[test]
    fn prop_squeeze_no_ones(
        dims in proptest::collection::vec(1usize..8, 1..6),
    ) {
        let squeezed = TransposeKernel::squeeze(&dims);
        for &d in &squeezed {
            prop_assert!(d > 1 || dims.iter().all(|&x| x == 1),
                "squeeze left a 1-dim: {:?} -> {:?}", dims, squeezed);
        }
    }

    /// Contiguous strides: last stride is 1 for non-empty shapes.
    #[test]
    fn prop_contiguous_strides_last_is_one(
        shape in proptest::collection::vec(1usize..8, 1..5),
    ) {
        let strides = TransposeKernel::contiguous_strides(&shape);
        prop_assert_eq!(*strides.last().unwrap(), 1);
    }

    // ── Concat / Split round-trip ───────────────────────────────────

    /// Split then concat recovers the original for 1D data along axis 0.
    #[test]
    fn prop_split_concat_roundtrip(
        n_chunks in 2usize..6,
        chunk_size in 1usize..16,
    ) {
        let total = n_chunks * chunk_size;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let shape = [total];

        let pieces = ConcatKernel::split(&data, &shape, 0, n_chunks).unwrap();
        let refs: Vec<&[f32]> = pieces.iter().map(|v| v.as_slice()).collect();
        let shapes: Vec<Vec<usize>> = pieces.iter().map(|p| vec![p.len()]).collect();
        let shape_refs: Vec<&[usize]> = shapes.iter().map(|s| s.as_slice()).collect();
        let recovered = ConcatKernel::concat(&refs, &shape_refs, 0).unwrap();
        prop_assert_eq!(data, recovered);
    }

    // ── Quantization round-trip error ───────────────────────────────

    /// Symmetric i8 quantize-dequantize error is bounded.
    #[test]
    fn prop_symmetric_quant_roundtrip_bounded(input in finite_f32_vec(64)) {
        let (q, scale) = quantize_symmetric_i8(&input, 8);
        let dq = dequantize_symmetric_i8(&q, scale);
        let err = compute_quantization_error(&input, &dq);
        prop_assert!(err.max_abs_error.is_finite(), "max error not finite");
        if scale > 0.0 {
            prop_assert!(err.max_abs_error <= scale + 1e-5,
                "max_abs_error {} > scale {}", err.max_abs_error, scale);
        }
    }

    /// Ternary quantization produces only {-1, 0, 1}.
    #[test]
    fn prop_ternary_values_subset(
        input in finite_f32_vec(64),
        threshold in 0.01f32..2.0,
    ) {
        let q = quantize_ternary(&input, threshold);
        for &v in &q {
            prop_assert!(v == -1 || v == 0 || v == 1, "ternary value out of set: {}", v);
        }
    }

    /// Binary quantization produces only {-1, 1}.
    #[test]
    fn prop_binary_values_subset(input in finite_f32_vec(64)) {
        let q = quantize_binary(&input);
        for &v in &q {
            prop_assert!(v == -1 || v == 1, "binary value out of set: {}", v);
        }
    }

    /// Quantization error fields are non-negative.
    #[test]
    fn prop_quantization_error_non_negative(input in finite_f32_vec(32)) {
        let (q, scale) = quantize_symmetric_i8(&input, 8);
        let dq = dequantize_symmetric_i8(&q, scale);
        let err = compute_quantization_error(&input, &dq);
        prop_assert!(err.max_abs_error >= 0.0);
        prop_assert!(err.mse >= 0.0);
    }

    // ── Gating functions ────────────────────────────────────────────

    /// SwiGLU with zero gate produces zero output.
    #[test]
    fn prop_swiglu_zero_gate(n in 1usize..64) {
        let gate = vec![0.0f32; n];
        let up = vec![1.0f32; n];
        let mut output = vec![0.0f32; n];
        swiglu(&gate, &up, &mut output).unwrap();
        for &v in &output {
            prop_assert!((v).abs() < 1e-6, "swiglu(0, 1) = {}", v);
        }
    }

    /// GEGLU with zero gate: gelu(0) * up ≈ 0.
    #[test]
    fn prop_geglu_zero_gate(n in 1usize..64) {
        let gate = vec![0.0f32; n];
        let up = vec![1.0f32; n];
        let mut output = vec![0.0f32; n];
        geglu(&gate, &up, &mut output).unwrap();
        for &v in &output {
            prop_assert!(v.abs() < 1e-5, "geglu(0, 1) = {}", v);
        }
    }

    /// ReGLU with negative gate produces zero.
    #[test]
    fn prop_reglu_negative_gate(n in 1usize..64) {
        let gate = vec![-5.0f32; n];
        let up = vec![1.0f32; n];
        let mut output = vec![0.0f32; n];
        reglu(&gate, &up, &mut output).unwrap();
        for &v in &output {
            prop_assert!(v.abs() < 1e-6, "reglu(-5, 1) = {}", v);
        }
    }

    /// Gating length mismatch always errors.
    #[test]
    fn prop_gating_mismatch_errors(
        n in 1usize..16,
        m in 1usize..16,
    ) {
        prop_assume!(n != m);
        let gate = vec![0.0; n];
        let up = vec![0.0; m];
        let mut out = vec![0.0; n];
        prop_assert!(swiglu(&gate, &up, &mut out).is_err());
    }

    // ── Pooling ─────────────────────────────────────────────────────

    /// Adaptive average pool output has exactly the requested length.
    #[test]
    fn prop_adaptive_avg_pool_output_size(
        input_len in 1usize..64,
        output_size in 1usize..64,
    ) {
        prop_assume!(output_size <= input_len);
        let data: Vec<f32> = (0..input_len).map(|i| i as f32).collect();
        let result = adaptive_avg_pool_1d(&data, output_size).unwrap();
        prop_assert_eq!(result.len(), output_size);
    }

    /// Max pool output >= average pool output element-wise.
    #[test]
    fn prop_max_pool_geq_avg_pool(
        input_len in 4usize..32,
        kernel_size in 2usize..5,
    ) {
        prop_assume!(kernel_size <= input_len);
        let data: Vec<f32> = (0..input_len).map(|i| i as f32).collect();

        let avg_cfg = PoolConfig {
            pool_type: PoolType::Average,
            kernel_size,
            stride: kernel_size,
            padding: 0,
            dilation: 1,
            ceil_mode: false,
        };
        let max_cfg = PoolConfig {
            pool_type: PoolType::Max,
            kernel_size,
            stride: kernel_size,
            padding: 0,
            dilation: 1,
            ceil_mode: false,
        };

        if let (Ok(avg), Ok(max_out)) = (pool_1d(&data, &avg_cfg), pool_1d(&data, &max_cfg)) {
            for (i, (&a, &m)) in avg.iter().zip(max_out.iter()).enumerate() {
                prop_assert!(m >= a - 1e-6, "max[{i}]={m} < avg[{i}]={a}");
            }
        }
    }

    // ── Linear projection ───────────────────────────────────────────

    /// Linear forward output length = batch_size * out_features.
    #[test]
    fn prop_linear_output_shape(
        batch in 1usize..4,
        in_f in 1usize..16,
        out_f in 1usize..16,
    ) {
        let config = LinearConfig::new(batch, in_f, out_f).unwrap();
        let input = vec![0.1f32; batch * in_f];
        let weight = vec![0.01f32; in_f * out_f];
        let bias = vec![0.0f32; out_f];
        let mut output = vec![0.0f32; batch * out_f];
        let r = linear_forward(&input, &weight, Some(&bias), &mut output, &config);
        prop_assert!(r.is_ok());
        prop_assert_eq!(output.len(), batch * out_f);
    }

    /// Linear with zero weights produces bias only.
    #[test]
    fn prop_linear_zero_weight_is_bias(
        in_f in 1usize..8,
        out_f in 1usize..8,
    ) {
        let config = LinearConfig::new(1, in_f, out_f).unwrap();
        let input = vec![1.0f32; in_f];
        let weight = vec![0.0f32; in_f * out_f];
        let bias = vec![0.5f32; out_f];
        let mut output = vec![0.0f32; out_f];
        linear_forward(&input, &weight, Some(&bias), &mut output, &config).unwrap();
        for (i, &v) in output.iter().enumerate() {
            prop_assert!((v - 0.5).abs() < 1e-5, "result[{i}] = {v}, expected 0.5");
        }
    }

    // ── Batch operations ────────────────────────────────────────────

    /// batched_softmax: each row sums to 1.
    #[test]
    fn prop_batched_softmax_row_sums(
        batch in 1usize..4,
        seq_len in 2usize..16,
    ) {
        let data: Vec<f32> = (0..(batch * seq_len)).map(|i| (i as f32) * 0.1).collect();
        if let Ok(result) = batched_softmax(&data, batch, seq_len) {
            for b in 0..batch {
                let row = &result[b * seq_len..(b + 1) * seq_len];
                let sum: f32 = row.iter().sum();
                prop_assert!((sum - 1.0).abs() < 1e-4, "batch {b}: sum = {sum}");
            }
        }
    }

    /// batched_add preserves length.
    #[test]
    fn prop_batched_add_preserves_length(
        batch in 1usize..4,
        dim in 1usize..16,
    ) {
        let n = batch * dim;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
        if let Ok(result) = batched_add(&a, &b, batch, dim) {
            prop_assert_eq!(result.len(), n);
        }
    }

    // ── Embedding ───────────────────────────────────────────────────

    /// Embedding lookup output length = num_indices * embedding_dim.
    #[test]
    fn prop_embedding_lookup_shape(
        vocab_size in 2usize..16,
        embed_dim in 1usize..8,
        num_indices in 1usize..8,
    ) {
        let table: Vec<f32> = (0..(vocab_size * embed_dim)).map(|i| i as f32 * 0.01).collect();
        let indices: Vec<u32> = (0..num_indices).map(|i| (i % vocab_size) as u32).collect();
        let result = embedding_lookup(&table, &indices, embed_dim).unwrap();
        prop_assert_eq!(result.len(), num_indices * embed_dim);
    }

    /// normalize_embeddings produces unit vectors.
    #[test]
    fn prop_normalize_embeddings_unit(
        n_vecs in 1usize..8,
        dim in 2usize..8,
    ) {
        let mut data: Vec<f32> = (0..(n_vecs * dim)).map(|i| (i as f32 + 1.0) * 0.1).collect();
        normalize_embeddings(&mut data, dim);
        for i in 0..n_vecs {
            let vec_slice = &data[i * dim..(i + 1) * dim];
            let norm: f32 = vec_slice.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assert!((norm - 1.0).abs() < 1e-4, "vec {i} norm = {norm}");
        }
    }

    /// Positional embedding output length = seq_len * embedding_dim.
    #[test]
    fn prop_positional_embedding_shape(
        seq_len in 1usize..16,
        embed_dim in 2usize..16,
    ) {
        prop_assume!(embed_dim % 2 == 0);
        let pe = positional_embedding(seq_len, embed_dim);
        prop_assert_eq!(pe.len(), seq_len * embed_dim);
    }

    /// Positional encoding values are finite.
    #[test]
    fn prop_positional_encoding_finite(
        seq_len in 1usize..16,
        embed_dim in (1usize..8).prop_map(|x| x * 2),
    ) {
        let pe = positional_encoding(seq_len, embed_dim, 10000.0);
        for &v in &pe {
            prop_assert!(v.is_finite(), "non-finite positional encoding: {}", v);
        }
    }

    // ── SIMD-scalar agreement ───────────────────────────────────────

    /// SIMD dot product is commutative.
    #[test]
    fn prop_simd_dot_product_commutative(
        a in finite_f32_vec(32),
    ) {
        let b: Vec<f32> = a.iter().rev().copied().collect();
        let ab = simd_dot_product(&a, &b);
        let ba = simd_dot_product(&b, &a);
        prop_assert!((ab - ba).abs() < 1e-3, "dot(a,b)={ab} != dot(b,a)={ba}");
    }

    /// SIMD L2 norm is non-negative.
    #[test]
    fn prop_simd_l2_norm_non_negative(input in finite_f32_vec(32)) {
        let norm = simd_l2_norm(&input);
        prop_assert!(norm >= 0.0, "negative L2 norm: {}", norm);
    }

    /// SIMD vector add is commutative.
    #[test]
    fn prop_simd_vector_add_commutative(a in finite_f32_vec(32)) {
        let b: Vec<f32> = a.iter().map(|x| x * 0.5).collect();
        let ab = simd_vector_add(&a, &b);
        let ba = simd_vector_add(&b, &a);
        for (i, (&x, &y)) in ab.iter().zip(ba.iter()).enumerate() {
            prop_assert!((x - y).abs() < 1e-5, "add not commutative at {i}");
        }
    }

    /// SIMD scale by 1.0 is identity.
    #[test]
    fn prop_simd_scale_identity(input in finite_f32_vec(32)) {
        let scaled = simd_vector_scale(&input, 1.0);
        for (i, (&s, &o)) in scaled.iter().zip(input.iter()).enumerate() {
            prop_assert!((s - o).abs() < 1e-6, "scale(1.0) changed value at {i}");
        }
    }

    /// SIMD scale by 0.0 produces zeros.
    #[test]
    fn prop_simd_scale_zero(input in finite_f32_vec(32)) {
        let scaled = simd_vector_scale(&input, 0.0);
        for (i, &v) in scaled.iter().enumerate() {
            prop_assert!(v.abs() < 1e-6, "scale(0.0) non-zero at {i}: {v}");
        }
    }

    /// SIMD vector_mul element-wise with ones is identity.
    #[test]
    fn prop_simd_mul_ones_identity(input in finite_f32_vec(32)) {
        let ones = vec![1.0f32; input.len()];
        let result = simd_vector_mul(&input, &ones);
        for (i, (&r, &o)) in result.iter().zip(input.iter()).enumerate() {
            prop_assert!((r - o).abs() < 1e-6, "mul by 1 changed at {i}");
        }
    }

    // ── Residual connections ────────────────────────────────────────

    /// add_residual with zero residual is identity.
    #[test]
    fn prop_residual_zero_is_identity(input in finite_f32_vec(32)) {
        let mut output = input.clone();
        let zero = vec![0.0; input.len()];
        add_residual(&mut output, &zero).unwrap();
        prop_assert_eq!(output, input);
    }

    /// add_residual_scaled with scale=0 is identity.
    #[test]
    fn prop_residual_scaled_zero_identity(input in finite_f32_vec(32)) {
        let mut output = input.clone();
        let residual = vec![99.0; input.len()];
        add_residual_scaled(&mut output, &residual, 0.0).unwrap();
        prop_assert_eq!(output, input);
    }

    // ── Loss functions ──────────────────────────────────────────────

    /// MSE loss is non-negative.
    #[test]
    fn prop_mse_non_negative(input in finite_f32_vec(32)) {
        let targets: Vec<f32> = input.iter().map(|x| x + 0.1).collect();
        let loss = mse_loss(&input, &targets, LossReduction::Mean).unwrap();
        prop_assert!(loss >= 0.0, "negative MSE: {}", loss);
    }

    /// MSE of identical inputs is zero.
    #[test]
    fn prop_mse_identical_is_zero(input in finite_f32_vec(32)) {
        let loss = mse_loss(&input, &input, LossReduction::Mean).unwrap();
        prop_assert!(loss.abs() < 1e-6, "MSE of identical: {}", loss);
    }

    /// L1 loss is non-negative.
    #[test]
    fn prop_l1_non_negative(input in finite_f32_vec(32)) {
        let targets: Vec<f32> = input.iter().map(|x| x + 0.5).collect();
        let loss = l1_loss(&input, &targets, LossReduction::Mean).unwrap();
        prop_assert!(loss >= 0.0, "negative L1: {}", loss);
    }

    /// Smooth L1 loss is non-negative.
    #[test]
    fn prop_smooth_l1_non_negative(input in finite_f32_vec(32)) {
        let targets: Vec<f32> = input.iter().map(|x| x + 0.1).collect();
        let loss = smooth_l1_loss(&input, &targets, 1.0, LossReduction::Mean).unwrap();
        prop_assert!(loss >= 0.0, "negative smooth L1: {}", loss);
    }

    /// Cosine similarity loss is finite.
    #[test]
    fn prop_cosine_similarity_loss_finite(a in finite_f32_vec(16)) {
        let b: Vec<f32> = a.iter().rev().copied().collect();
        if let Ok(loss) = cosine_similarity_loss(&a, &b) {
            prop_assert!(loss.is_finite(), "cosine similarity loss not finite: {}", loss);
        }
    }

    // ── RoPE frequencies ────────────────────────────────────────────

    /// RoPE frequencies are all finite.
    #[test]
    fn prop_rope_frequencies_finite(
        head_dim in (1usize..8).prop_map(|x| x * 2),
        max_seq_len in 1usize..64,
    ) {
        let cfg = RopeConfig::new(head_dim, max_seq_len);
        let freqs = compute_frequencies(&cfg);
        for &f in &freqs {
            prop_assert!(f.is_finite(), "non-finite frequency: {}", f);
        }
    }

    /// apply_rope preserves vector length and produces finite values.
    #[test]
    fn prop_rope_preserves_length(
        head_dim in (1usize..4).prop_map(|x| x * 2),
    ) {
        let cfg = RopeConfig::new(head_dim, 32);
        let freqs = compute_frequencies(&cfg);
        let mut data = vec![1.0f32; head_dim];
        apply_rope(&mut data, 0, head_dim, &freqs);
        prop_assert_eq!(data.len(), head_dim);
        for &v in &data {
            prop_assert!(v.is_finite());
        }
    }

    // ── Layer norm / Group norm shape preservation ──────────────────

    /// Group norm output length equals input length.
    #[test]
    fn prop_group_norm_preserves_length(
        n_groups in 1usize..4,
        channels_per_group in 1usize..4,
        spatial in 1usize..8,
    ) {
        let n_channels = n_groups * channels_per_group;
        let total = n_channels * spatial;
        let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 + 0.01).collect();
        let gamma = vec![1.0f32; n_channels];
        let beta = vec![0.0f32; n_channels];
        let cfg = GroupNormConfig::new(n_groups, n_channels, spatial);
        if let Ok(result) = group_norm(&input, &gamma, Some(&beta), &cfg) {
            prop_assert_eq!(result.len(), total);
        }
    }

    /// RMS norm output same length as input.
    #[test]
    fn prop_rms_norm_preserves_length(dim in 2usize..32) {
        let input: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let gamma = vec![1.0f32; dim];
        let cfg = LayerNormConfig::new(vec![dim]);
        let result = rms_norm(&input, &gamma, &cfg).unwrap();
        prop_assert_eq!(result.len(), dim);
    }

    // ── Scatter-Gather round-trip ───────────────────────────────────

    /// gather_1d then scatter_1d recovers original values at indexed positions.
    #[test]
    fn prop_gather_scatter_roundtrip(n in 2usize..32) {
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let indices: Vec<usize> = (0..n).collect();
        let gathered = gather_1d(&data, &indices).unwrap();
        let mut target = vec![0.0f32; n];
        scatter_1d(&mut target, &indices, &gathered).unwrap();
        prop_assert_eq!(target, data);
    }

    // ── Activation function consistency ─────────────────────────────

    /// silu(x) == x * sigmoid(x) for all finite x.
    #[test]
    fn prop_silu_equals_x_times_sigmoid(x in -20.0f32..20.0) {
        let expected = x * sigmoid(x);
        let actual = silu(x);
        prop_assert!((actual - expected).abs() < 1e-5,
            "silu({x}) = {actual}, expected {expected}");
    }

    /// gelu(x) is finite for moderate inputs.
    #[test]
    fn prop_gelu_finite(x in -50.0f32..50.0) {
        let g = gelu(x);
        prop_assert!(g.is_finite(), "gelu({x}) = {g}");
    }

    /// tanh_act output in (-1, 1).
    #[test]
    fn prop_tanh_bounded(x in -20.0f32..20.0) {
        let t = tanh_act(x);
        prop_assert!(t > -1.0 - 1e-6 && t < 1.0 + 1e-6, "tanh({x}) = {t}");
    }

    // ── Reduction consistency ───────────────────────────────────────

    /// ReductionKernel::sum of a constant vector = constant * length.
    #[test]
    fn prop_reduction_sum_constant(n in 1usize..64, val in -10.0f32..10.0) {
        let data = vec![val; n];
        let sum = ReductionKernel::sum(&data).unwrap();
        let expected = val * n as f32;
        prop_assert!((sum - expected).abs() < 1e-2,
            "sum of {n}×{val} = {sum}, expected {expected}");
    }

    /// Product of ones is 1.
    #[test]
    fn prop_reduction_product_ones(n in 1usize..32) {
        let data = vec![1.0f32; n];
        let prod = ReductionKernel::product(&data).unwrap();
        prop_assert!((prod - 1.0).abs() < 1e-5, "product of ones = {prod}");
    }

    /// Mean of identical values equals that value.
    #[test]
    fn prop_reduction_mean_constant(n in 1usize..64, val in -10.0f32..10.0) {
        let data = vec![val; n];
        let mean = ReductionKernel::mean(&data).unwrap();
        prop_assert!((mean - val).abs() < 1e-4, "mean of {n}×{val} = {mean}");
    }

    /// L1 norm of a constant vector = |val| * n.
    #[test]
    fn prop_reduction_l1_norm_constant(n in 1usize..32, val in -10.0f32..10.0) {
        let data = vec![val; n];
        let norm = ReductionKernel::l1_norm(&data).unwrap();
        let expected = val.abs() * n as f32;
        prop_assert!((norm - expected).abs() < 1e-2,
            "l1_norm of {n}×{val} = {norm}, expected {expected}");
    }

    /// L2 norm of a single element is its absolute value.
    #[test]
    fn prop_reduction_l2_norm_single(val in -50.0f32..50.0) {
        let data = vec![val];
        let norm = ReductionKernel::l2_norm(&data).unwrap();
        prop_assert!((norm - val.abs()).abs() < 1e-5,
            "l2_norm([{val}]) = {norm}, expected {}", val.abs());
    }

    /// Unsqueeze increases dimensionality by 1.
    #[test]
    fn prop_unsqueeze_adds_dim(
        dims in proptest::collection::vec(1usize..8, 1..5),
    ) {
        let result = TransposeKernel::unsqueeze(&dims, 0).unwrap();
        prop_assert_eq!(result.len(), dims.len() + 1);
        prop_assert_eq!(result[0], 1);
    }

    /// Reshape preserves total element count.
    #[test]
    fn prop_reshape_preserves_elements(
        rows in 1usize..8,
        cols in 1usize..8,
    ) {
        let total = rows * cols;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let result = TransposeKernel::reshape(&data, &[rows, cols], &[total]).unwrap();
        prop_assert_eq!(result.len(), total);
        prop_assert_eq!(result, data);
    }

    /// batched_layer_norm output is finite and same length.
    #[test]
    fn prop_batched_layer_norm_finite(
        batch in 1usize..3,
        dim in 2usize..8,
    ) {
        let n = batch * dim;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let gamma = vec![1.0f32; dim];
        let beta = vec![0.0f32; dim];
        if let Ok(result) = batched_layer_norm(&data, &gamma, &beta, batch, dim, 1e-5) {
            prop_assert_eq!(result.len(), n);
            for &v in &result {
                prop_assert!(v.is_finite());
            }
        }
    }
}
