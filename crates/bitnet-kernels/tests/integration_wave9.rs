//! Integration Wave 9 — Cross-Crate Kernel Pipeline Tests
//!
//! Validates that kernel primitives compose correctly into realistic
//! inference sub-graphs:
//!
//! - embedding → layer norm → attention
//! - RoPE → multi-head attention → layer norm
//! - quantized matmul → activation → reduction
//! - conv2d → activation → pooling
//! - scatter → gather round-trip
//! - shaped reduction → activation → pooling
//! - fusion → reduction verification
//! - GQA → layer norm → projection
//! - error propagation through composed pipelines

use bitnet_kernels::convolution::{Conv2DParams, conv2d};
use bitnet_kernels::cpu::activations::{
    gelu_exact_activate, relu_activate, relu_activate_inplace, silu_activate,
};
use bitnet_kernels::cpu::attention::{
    AttentionConfig, AttentionKernel, GqaConfig, apply_mask, causal_mask,
};
use bitnet_kernels::cpu::embedding::{
    CpuEmbeddingConfig, embedding_lookup, embedding_with_position, normalize_embeddings,
    pack_embedding_table, unpack_embedding_lookup,
};
use bitnet_kernels::cpu::fusion::{
    fused_add_normalize, fused_gelu_linear, fused_rmsnorm_linear, fused_softmax_mask,
};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::pooling::{PoolConfig, PoolType, PoolingKernel};
use bitnet_kernels::cpu::quantized_matmul::{dequantize_and_matmul, i2s_matmul_f32, pack_i2s};
use bitnet_kernels::cpu::reduction::ReductionKernel;
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, apply_rope_batch, compute_frequencies};
use bitnet_kernels::reduction::{ReductionOp, reduce_f32, reduce_rows_f32};
use bitnet_kernels::scatter_gather::{GatherConfig, ScatterMode, scatter_cpu};
use bitnet_kernels::shaped_reduction::ShapedReductionConfig;

// ── Helpers ────────────────────────────────────────────────────────

fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
    assert!((a - b).abs() <= tol, "{ctx}: expected {b}, got {a} (diff {})", (a - b).abs());
}

fn assert_slice_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
    for (i, (&ai, &bi)) in a.iter().zip(b).enumerate() {
        assert_close(ai, bi, tol, &format!("{ctx}[{i}]"));
    }
}

/// Simple matmul: C[m×n] = A[m×k] × B[k×n] (row-major).
fn naive_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

// ══════════════════════════════════════════════════════════════════
// 1. Embedding → Layer-Norm → Attention pipeline
// ══════════════════════════════════════════════════════════════════

#[test]
fn embedding_layernorm_attention_pipeline() {
    let vocab = 8;
    let dim = 4;
    let seq_len = 2;
    let num_heads = 2;
    let head_dim = dim / num_heads;

    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.1).collect();
    let tokens = [1u32, 3];
    let emb = embedding_lookup(&table, &tokens, dim).unwrap();
    assert_eq!(emb.len(), seq_len * dim);

    let ln_cfg = LayerNormConfig::new(vec![dim]);
    let gamma = vec![1.0f32; dim];
    let normed = layer_norm(&emb, &gamma, None, &ln_cfg).unwrap();
    assert_eq!(normed.len(), seq_len * dim);

    let attn_cfg = AttentionConfig { num_heads, head_dim, seq_len, causal: false, scale: None };
    let out = AttentionKernel::multi_head_attention(&normed, &normed, &normed, &attn_cfg).unwrap();
    assert_eq!(out.len(), seq_len * dim);
    assert!(out.iter().all(|v| v.is_finite()));
}

// ══════════════════════════════════════════════════════════════════
// 2. RoPE → Multi-Head Attention → Layer-Norm
// ══════════════════════════════════════════════════════════════════

#[test]
fn rope_mha_layernorm_pipeline() {
    let num_heads = 2;
    let head_dim = 4;
    let seq_len = 3;
    let model_dim = num_heads * head_dim;

    let rope_cfg = RopeConfig::new(head_dim, 64);
    let freqs = compute_frequencies(&rope_cfg);

    let mut q: Vec<f32> = (0..seq_len * model_dim).map(|i| (i as f32) * 0.05).collect();
    let mut k = q.clone();
    let v = q.clone();

    apply_rope_batch(&mut q, 0, seq_len, num_heads, head_dim, &freqs);
    apply_rope_batch(&mut k, 0, seq_len, num_heads, head_dim, &freqs);

    let attn_cfg = AttentionConfig { num_heads, head_dim, seq_len, causal: true, scale: None };
    let attn_out = AttentionKernel::multi_head_attention(&q, &k, &v, &attn_cfg).unwrap();
    assert_eq!(attn_out.len(), seq_len * model_dim);

    let ln_cfg = LayerNormConfig::new(vec![model_dim]);
    let gamma = vec![1.0f32; model_dim];
    let normed = layer_norm(&attn_out, &gamma, None, &ln_cfg).unwrap();
    assert_eq!(normed.len(), seq_len * model_dim);
    assert!(normed.iter().all(|v| v.is_finite()));
}

// ══════════════════════════════════════════════════════════════════
// 3. Quantized MatMul → ReLU → Reduction
// ══════════════════════════════════════════════════════════════════

#[test]
fn quantized_matmul_relu_reduction_pipeline() {
    let m: usize = 2;
    let k: usize = 8;
    let n: usize = 4;
    let block_size: usize = 4;

    let activations: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let packed_k = k.div_ceil(4);
    let weights: Vec<u8> = vec![pack_i2s([1, 1, 1, 1]); n * packed_k];
    let num_blocks = k.div_ceil(block_size);
    let scales = vec![1.0f32; n * num_blocks];

    let mut logits = vec![0.0f32; m * n];
    i2s_matmul_f32(&activations, &weights, &scales, &mut logits, m, n, k, block_size).unwrap();

    let mut activated = vec![0.0f32; logits.len()];
    relu_activate(&logits, &mut activated);
    assert!(activated.iter().all(|&v| v >= 0.0));

    let means = reduce_rows_f32(&activated, m, n, ReductionOp::Mean).unwrap();
    assert_eq!(means.len(), m);
    assert!(means.iter().all(|v| v.is_finite()));
}

// ══════════════════════════════════════════════════════════════════
// 4. Conv2d → SiLU → Max-Pool
// ══════════════════════════════════════════════════════════════════

#[test]
fn conv2d_silu_maxpool_pipeline() {
    let in_c = 1;
    let out_c = 1;
    let kh = 3;
    let kw = 3;
    let h = 4;
    let w = 4;

    let input: Vec<f32> = (0..in_c * h * w).map(|i| i as f32).collect();
    let weight: Vec<f32> = vec![1.0 / (kh * kw) as f32; out_c * in_c * kh * kw];
    let bias = vec![0.0f32; out_c];

    let params = Conv2DParams { stride: (1, 1), padding: (1, 1), dilation: (1, 1) };
    let out_h = (h + 2 * params.padding.0 - params.dilation.0 * (kh - 1) - 1) / params.stride.0 + 1;
    let out_w = (w + 2 * params.padding.1 - params.dilation.1 * (kw - 1) - 1) / params.stride.1 + 1;
    let mut conv_out = vec![0.0f32; out_c * out_h * out_w];

    conv2d(
        &input,
        &weight,
        Some(&bias),
        &mut conv_out,
        params,
        (1, in_c, h, w),
        (out_c, in_c, kh, kw),
    )
    .unwrap();
    assert_eq!(conv_out.len(), out_c * out_h * out_w);

    let mut activated = vec![0.0f32; conv_out.len()];
    silu_activate(&conv_out, &mut activated);
    assert_eq!(activated.len(), conv_out.len());

    let pool_cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 2, padding: 0 };
    let pooled = PoolingKernel::apply(&activated, &pool_cfg).unwrap();
    assert!(!pooled.is_empty());
    assert!(pooled.iter().all(|v| v.is_finite()));
}

// ══════════════════════════════════════════════════════════════════
// 5. RoPE single-apply vs batch-apply consistency
// ══════════════════════════════════════════════════════════════════

#[test]
fn rope_single_vs_batch_consistency() {
    let head_dim = 4;
    let num_heads = 2;
    let seq_len = 3;
    let model_dim = num_heads * head_dim;

    let rope_cfg = RopeConfig::new(head_dim, 64);
    let freqs = compute_frequencies(&rope_cfg);

    let mut batch_data: Vec<f32> = (0..seq_len * model_dim).map(|i| i as f32 * 0.1).collect();
    apply_rope_batch(&mut batch_data, 0, seq_len, num_heads, head_dim, &freqs);

    let mut single_data: Vec<f32> = (0..seq_len * model_dim).map(|i| i as f32 * 0.1).collect();
    for pos in 0..seq_len {
        for h in 0..num_heads {
            let offset = pos * model_dim + h * head_dim;
            let slice = &mut single_data[offset..offset + head_dim];
            apply_rope(slice, pos, head_dim, &freqs);
        }
    }

    assert_slice_close(&batch_data, &single_data, 1e-6, "rope_batch_vs_single");
}

// ══════════════════════════════════════════════════════════════════
// 6. Embedding → RMS-Norm → Fused RMS-Norm-Linear
// ══════════════════════════════════════════════════════════════════

#[test]
fn embedding_rmsnorm_fused_linear_pipeline() {
    let vocab = 16;
    let dim = 8;
    let tokens = [2u32, 5, 7];

    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.01).collect();
    let emb = embedding_lookup(&table, &tokens, dim).unwrap();

    let gamma = vec![1.0f32; dim];
    let weight: Vec<f32> = vec![0.1; dim * dim];
    for pos in 0..tokens.len() {
        let row = &emb[pos * dim..(pos + 1) * dim];
        let fused = fused_rmsnorm_linear(row, &weight, &gamma, 1e-5).unwrap();
        assert_eq!(fused.len(), dim);
        assert!(fused.iter().all(|v| v.is_finite()));
    }
}

// ══════════════════════════════════════════════════════════════════
// 7. Causal mask → Attention → Row values check
// ══════════════════════════════════════════════════════════════════

#[test]
fn causal_attention_with_all_ones_value() {
    let seq_len = 4;
    let head_dim = 4;

    let q: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 * 0.1).collect();
    let k = q.clone();
    let v: Vec<f32> = vec![1.0; seq_len * head_dim];

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mask = causal_mask(seq_len);

    let out = AttentionKernel::scaled_dot_product(
        &q,
        &k,
        &v,
        Some(&mask),
        scale,
        seq_len,
        seq_len,
        head_dim,
    )
    .unwrap();

    for row in 0..seq_len {
        let row_slice = &out[row * head_dim..(row + 1) * head_dim];
        let sum: f32 = row_slice.iter().sum();
        assert_close(sum, head_dim as f32, 1e-3, &format!("row_{row}_sum"));
    }
}

// ══════════════════════════════════════════════════════════════════
// 8. GQA → Layer-Norm → Linear Projection
// ══════════════════════════════════════════════════════════════════

#[test]
fn gqa_layernorm_projection_pipeline() {
    let num_q_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 4;
    let seq_len = 2;
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let q: Vec<f32> = (0..seq_len * q_dim).map(|i| i as f32 * 0.02).collect();
    let k: Vec<f32> = (0..seq_len * kv_dim).map(|i| i as f32 * 0.03).collect();
    let v: Vec<f32> = (0..seq_len * kv_dim).map(|i| i as f32 * 0.01).collect();

    let gqa_cfg =
        GqaConfig { num_q_heads, num_kv_heads, head_dim, seq_len, causal: false, scale: None };
    let attn_out = AttentionKernel::grouped_query_attention(&q, &k, &v, &gqa_cfg).unwrap();
    assert_eq!(attn_out.len(), seq_len * q_dim);

    let ln_cfg = LayerNormConfig::new(vec![q_dim]);
    let gamma = vec![1.0f32; q_dim];
    let normed = layer_norm(&attn_out, &gamma, None, &ln_cfg).unwrap();

    let proj_weight: Vec<f32> = (0..q_dim * q_dim).map(|i| (i as f32) * 0.001).collect();
    let projected = naive_matmul(&normed, &proj_weight, seq_len, q_dim, q_dim);
    assert_eq!(projected.len(), seq_len * q_dim);
    assert!(projected.iter().all(|v| v.is_finite()));
}

// ══════════════════════════════════════════════════════════════════
// 9. i2s_matmul_f32 agrees with dequantize_and_matmul
// ══════════════════════════════════════════════════════════════════

#[test]
fn i2s_matmul_agrees_with_dequantize_path() {
    let m: usize = 2;
    let k: usize = 8;
    let n: usize = 4;
    let block_size: usize = 4;

    let activations: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
    let packed_k = k.div_ceil(4);
    let weights: Vec<u8> =
        (0..n * packed_k).map(|i| pack_i2s([(i as i8 % 3) - 1, 1, 0, -1])).collect();
    let num_blocks = k.div_ceil(block_size);
    let scales: Vec<f32> = (0..n * num_blocks).map(|i| 0.5 + (i as f32) * 0.1).collect();

    let mut out_direct = vec![0.0f32; m * n];
    i2s_matmul_f32(&activations, &weights, &scales, &mut out_direct, m, n, k, block_size).unwrap();

    let mut out_deq = vec![0.0f32; m * n];
    dequantize_and_matmul(&activations, &weights, &scales, &mut out_deq, m, n, k, block_size)
        .unwrap();

    assert_slice_close(&out_direct, &out_deq, 1e-4, "i2s_vs_deq");
}

// ══════════════════════════════════════════════════════════════════
// 10. Embedding → Normalize → Attention preserves shape
// ══════════════════════════════════════════════════════════════════

#[test]
fn normalized_embedding_attention_preserves_shape() {
    let vocab = 16;
    let dim = 8;
    let seq_len = 3;
    let num_heads = 2;
    let head_dim = dim / num_heads;

    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.1).collect();
    let tokens = [1u32, 4, 7];
    let mut emb = embedding_lookup(&table, &tokens, dim).unwrap();
    normalize_embeddings(&mut emb, dim);

    for row in 0..seq_len {
        let slice = &emb[row * dim..(row + 1) * dim];
        let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_close(norm, 1.0, 1e-5, &format!("row_{row}_norm"));
    }

    let attn_cfg = AttentionConfig { num_heads, head_dim, seq_len, causal: false, scale: None };
    let out = AttentionKernel::multi_head_attention(&emb, &emb, &emb, &attn_cfg).unwrap();
    assert_eq!(out.len(), seq_len * dim);
}

// ══════════════════════════════════════════════════════════════════
// 11. Fused GELU-Linear matches separate GELU + MatMul
// ══════════════════════════════════════════════════════════════════

#[test]
fn fused_gelu_linear_matches_separate_ops() {
    let dim = 8;
    let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.05 - 0.2).collect();
    let weight: Vec<f32> = vec![0.1; dim * dim];
    let bias: Vec<f32> = vec![];

    let fused = fused_gelu_linear(&input, &weight, &bias).unwrap();

    let mut gelu_out = vec![0.0f32; dim];
    gelu_exact_activate(&input, &mut gelu_out);
    let manual: Vec<f32> = weight
        .chunks_exact(dim)
        .map(|row| row.iter().zip(&gelu_out).map(|(w, g)| w * g).sum::<f32>())
        .collect();

    assert_slice_close(&fused, &manual, 1e-4, "fused_gelu_linear");
}

// ══════════════════════════════════════════════════════════════════
// 12. Fused softmax-mask produces valid distribution
// ══════════════════════════════════════════════════════════════════

#[test]
fn fused_softmax_mask_produces_valid_distribution() {
    let n = 8;
    let scores: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let mask: Vec<f32> = vec![0.0; n];
    let scale = 1.0;

    let probs = fused_softmax_mask(&scores, &mask, scale).unwrap();
    assert_eq!(probs.len(), n);

    let sum: f32 = probs.iter().sum();
    assert_close(sum, 1.0, 1e-5, "softmax_sum");
    assert!(probs.iter().all(|&p| p >= 0.0));
}

// ══════════════════════════════════════════════════════════════════
// 13. Fused add-normalize matches manual residual + rms-norm
// ══════════════════════════════════════════════════════════════════

#[test]
fn fused_add_normalize_matches_manual() {
    let dim = 8;
    let residual: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
    let hidden: Vec<f32> = (0..dim).map(|i| (dim - i) as f32 * 0.05).collect();
    let gamma = vec![1.0f32; dim];

    let fused = fused_add_normalize(&residual, &hidden, &gamma, 1e-5).unwrap();

    let sum: Vec<f32> = residual.iter().zip(&hidden).map(|(a, b)| a + b).collect();
    let ln_cfg = LayerNormConfig::new(vec![dim]);
    let manual = rms_norm(&sum, &gamma, &ln_cfg).unwrap();

    assert_slice_close(&fused, &manual, 1e-4, "fused_add_normalize");
}

// ══════════════════════════════════════════════════════════════════
// 14. Scatter writes to correct rows
// ══════════════════════════════════════════════════════════════════

#[test]
fn scatter_writes_to_correct_rows() {
    let rows = 4;
    let cols = 3;

    let src = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]; // 2 rows
    // axis=0 scatter: indices shape (i_rows, i_cols) where i_cols == d_cols
    let indices: Vec<usize> = vec![1, 1, 1, 3, 3, 3];
    let scatter_cfg = GatherConfig::new(0, (2, cols), true).unwrap();

    let mut dst = vec![0.0f32; rows * cols];
    scatter_cpu(&src, &indices, &mut dst, (rows, cols), &scatter_cfg, ScatterMode::Assign).unwrap();

    assert_slice_close(&dst[cols..2 * cols], &[10.0, 20.0, 30.0], 0.0, "scatter_row1");
    assert_slice_close(&dst[3 * cols..4 * cols], &[40.0, 50.0, 60.0], 0.0, "scatter_row3");
}

// ══════════════════════════════════════════════════════════════════
// 15. Shaped reduction axis-0 sum matches manual column sums
// ══════════════════════════════════════════════════════════════════

#[test]
fn shaped_reduction_axis0_sum() {
    let rows = 3;
    let cols = 4;
    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();

    let cfg = ShapedReductionConfig::new(ReductionOp::Sum, Some(0), false);
    let result = bitnet_kernels::shaped_reduction::reduce_f32(&data, &[rows, cols], &cfg).unwrap();
    assert_eq!(result.len(), cols);

    for j in 0..cols {
        let expected: f32 = (0..rows).map(|i| data[i * cols + j]).sum();
        assert_close(result[j], expected, 1e-6, &format!("col_{j}_sum"));
    }
}

// ══════════════════════════════════════════════════════════════════
// 16. Pooling → Activation → Reduction end-to-end
// ══════════════════════════════════════════════════════════════════

#[test]
fn pooling_activation_reduction_pipeline() {
    let data: Vec<f32> = (0..16).map(|i| i as f32 - 8.0).collect();

    let pool_cfg =
        PoolConfig { pool_type: PoolType::Average, kernel_size: 4, stride: 4, padding: 0 };
    let pooled = PoolingKernel::apply(&data, &pool_cfg).unwrap();
    assert_eq!(pooled.len(), 4);

    let mut activated = vec![0.0f32; pooled.len()];
    silu_activate(&pooled, &mut activated);
    assert_eq!(activated.len(), 4);

    let max_val = reduce_f32(&activated, ReductionOp::Max);
    assert!(max_val.is_finite());
    assert!(max_val > 0.0);
}

// ══════════════════════════════════════════════════════════════════
// 17. Layer-norm is idempotent on already-normalized data
// ══════════════════════════════════════════════════════════════════

#[test]
fn layernorm_idempotent_on_normalized_data() {
    let dim = 8;
    let data: Vec<f32> = (0..dim).map(|i| i as f32).collect();

    let cfg = LayerNormConfig::new(vec![dim]);
    let gamma = vec![1.0f32; dim];

    let first = layer_norm(&data, &gamma, None, &cfg).unwrap();
    let second = layer_norm(&first, &gamma, None, &cfg).unwrap();

    assert_slice_close(&first, &second, 1e-5, "layernorm_idempotent");
}

// ══════════════════════════════════════════════════════════════════
// 18. Attention output is bounded by V range
// ══════════════════════════════════════════════════════════════════

#[test]
fn attention_output_bounded_by_value_range() {
    let num_heads = 2;
    let head_dim = 4;
    let seq_len = 3;
    let model_dim = num_heads * head_dim;

    let q: Vec<f32> = vec![0.5; seq_len * model_dim];
    let k: Vec<f32> = vec![0.5; seq_len * model_dim];
    let v: Vec<f32> = (0..seq_len * model_dim)
        .map(|i| -1.0 + (i as f32) * 3.0 / (seq_len * model_dim) as f32)
        .collect();

    let v_min = v.iter().copied().fold(f32::INFINITY, f32::min);
    let v_max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let cfg = AttentionConfig { num_heads, head_dim, seq_len, causal: false, scale: None };
    let out = AttentionKernel::multi_head_attention(&q, &k, &v, &cfg).unwrap();

    for (i, &val) in out.iter().enumerate() {
        assert!(
            val >= v_min - 1e-5 && val <= v_max + 1e-5,
            "out[{i}]={val} outside V range [{v_min}, {v_max}]"
        );
    }
}

// ══════════════════════════════════════════════════════════════════
// 19. RoPE preserves vector norms
// ══════════════════════════════════════════════════════════════════

#[test]
fn rope_preserves_norms() {
    let head_dim = 8;
    let rope_cfg = RopeConfig::new(head_dim, 128);
    let freqs = compute_frequencies(&rope_cfg);

    let original: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();
    let norm_before: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();

    let mut data = original.clone();
    apply_rope(&mut data, 5, head_dim, &freqs);

    let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

    assert_close(norm_before, norm_after, 1e-4, "rope_norm_preservation");
}

// ══════════════════════════════════════════════════════════════════
// 20. Full transformer block: embed → norm → attn → residual → norm
// ══════════════════════════════════════════════════════════════════

#[test]
fn mini_transformer_block_pipeline() {
    let vocab = 32;
    let dim = 8;
    let num_heads = 2;
    let head_dim = dim / num_heads;
    let seq_len = 3;

    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.01).collect();
    let tokens = [3u32, 7, 12];
    let emb = embedding_lookup(&table, &tokens, dim).unwrap();

    let gamma = vec![1.0f32; dim];
    let ln_cfg = LayerNormConfig::new(vec![dim]);
    let normed = rms_norm(&emb, &gamma, &ln_cfg).unwrap();

    let attn_cfg = AttentionConfig { num_heads, head_dim, seq_len, causal: true, scale: None };
    let attn_out =
        AttentionKernel::multi_head_attention(&normed, &normed, &normed, &attn_cfg).unwrap();

    let residual: Vec<f32> = emb.iter().zip(&attn_out).map(|(e, a)| e + a).collect();

    let post_normed = rms_norm(&residual, &gamma, &ln_cfg).unwrap();
    assert_eq!(post_normed.len(), seq_len * dim);
    assert!(post_normed.iter().all(|v| v.is_finite()));

    for pos in 0..seq_len {
        let slice = &post_normed[pos * dim..(pos + 1) * dim];
        let rms: f32 = (slice.iter().map(|x| x * x).sum::<f32>() / dim as f32).sqrt();
        assert_close(rms, 1.0, 0.1, &format!("pos_{pos}_rms"));
    }
}

// ══════════════════════════════════════════════════════════════════
// 21. ReLU in-place matches out-of-place
// ══════════════════════════════════════════════════════════════════

#[test]
fn relu_inplace_matches_outofplace() {
    let data: Vec<f32> = (-5..5).map(|i| i as f32 * 0.3).collect();

    let mut oop = vec![0.0f32; data.len()];
    relu_activate(&data, &mut oop);

    let mut ip = data.clone();
    relu_activate_inplace(&mut ip);

    assert_slice_close(&ip, &oop, 0.0, "relu_inplace_vs_oop");
}

// ══════════════════════════════════════════════════════════════════
// 22. Reduction kernel sum vs mean consistency
// ══════════════════════════════════════════════════════════════════

#[test]
fn reduction_sum_mean_consistency() {
    let data: Vec<f32> = (0..12).map(|i| (i + 1) as f32).collect();

    let sum = ReductionKernel::sum(&data).unwrap();
    let mean = ReductionKernel::mean(&data).unwrap();

    assert_close(mean, sum / data.len() as f32, 1e-6, "mean_eq_sum_div_n");
}

// ══════════════════════════════════════════════════════════════════
// 23. Embedding with sinusoidal position differs from plain
// ══════════════════════════════════════════════════════════════════

#[test]
fn embedding_with_position_differs_from_plain() {
    let vocab = 16;
    let dim = 4;

    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.1).collect();
    let tokens = [1u32, 3];
    let cfg = CpuEmbeddingConfig::new(vocab, dim);

    let plain = embedding_lookup(&table, &tokens, dim).unwrap();
    let with_pos = embedding_with_position(&table, &tokens, &cfg, 0).unwrap();

    assert_ne!(plain, with_pos, "position encoding should modify embeddings");
    assert_eq!(with_pos.len(), plain.len());
}

// ══════════════════════════════════════════════════════════════════
// 24. Apply mask zeroes future positions in scores
// ══════════════════════════════════════════════════════════════════

#[test]
fn apply_mask_zeroes_future_positions() {
    let seq_len = 4;
    let mut scores: Vec<f32> = vec![1.0; seq_len * seq_len];
    let mask = causal_mask(seq_len);

    apply_mask(&mut scores, &mask).unwrap();

    for i in 0..seq_len {
        for j in 0..seq_len {
            let val = scores[i * seq_len + j];
            if j > i {
                assert!(val.is_infinite() && val < 0.0, "mask[{i},{j}] should be -inf");
            } else {
                assert_close(val, 1.0, 0.0, &format!("mask[{i},{j}]"));
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════
// 25. Packed embedding round-trip preserves approximate values
// ══════════════════════════════════════════════════════════════════

#[test]
fn packed_embedding_round_trip() {
    let vocab = 8;
    let dim = 4;
    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.3 - 1.0).collect();
    let packed = pack_embedding_table(&table, vocab, dim);

    let indices = [0u32, 3, 7];
    let original = embedding_lookup(&table, &indices, dim).unwrap();
    let unpacked = unpack_embedding_lookup(&packed, &indices).unwrap();

    assert_eq!(unpacked.len(), original.len());
    for (i, (&o, &u)) in original.iter().zip(&unpacked).enumerate() {
        assert!((o - u).abs() < 0.1, "packed_emb[{i}]: orig={o}, unpacked={u}",);
    }
}

// ══════════════════════════════════════════════════════════════════
// 26. Shaped reduction global mean matches flat reduce
// ══════════════════════════════════════════════════════════════════

#[test]
fn shaped_reduction_global_mean_agrees() {
    let data: Vec<f32> = (0..12).map(|i| i as f32 + 1.0).collect();

    let cfg = ShapedReductionConfig::global(ReductionOp::Mean);
    let result = bitnet_kernels::shaped_reduction::reduce_f32(&data, &[3, 4], &cfg).unwrap();
    assert_eq!(result.len(), 1);

    let expected = reduce_f32(&data, ReductionOp::Mean);
    assert_close(result[0], expected, 1e-6, "global_mean");
}

// ══════════════════════════════════════════════════════════════════
// 27. Error propagation: mismatched dimensions detected early
// ══════════════════════════════════════════════════════════════════

#[test]
fn pipeline_error_propagation_on_shape_mismatch() {
    let cfg = AttentionConfig { num_heads: 2, head_dim: 4, seq_len: 3, causal: false, scale: None };
    let q = vec![0.0f32; 10]; // wrong (expected 24)
    let k = vec![0.0f32; 24];
    let v = vec![0.0f32; 24];
    let result = AttentionKernel::multi_head_attention(&q, &k, &v, &cfg);
    assert!(result.is_err(), "should reject mismatched Q shape");

    let ln_cfg = LayerNormConfig::new(vec![8]);
    let gamma = vec![1.0f32; 4]; // wrong
    let data = vec![1.0f32; 16];
    let result = layer_norm(&data, &gamma, None, &ln_cfg);
    assert!(result.is_err(), "should reject mismatched gamma length");
}

// ══════════════════════════════════════════════════════════════════
// 28. Fused softmax-mask with causal mask zeroes upper triangle
// ══════════════════════════════════════════════════════════════════

#[test]
fn fused_softmax_mask_causal_row() {
    let scores = vec![0.5, 0.8, 0.3, 0.1];
    let mask = vec![0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY];
    let probs = fused_softmax_mask(&scores, &mask, 1.0).unwrap();

    assert!(probs[2] < 1e-6);
    assert!(probs[3] < 1e-6);
    let visible_sum: f32 = probs[0] + probs[1];
    assert_close(visible_sum, 1.0, 1e-5, "visible_sum");
}

// ══════════════════════════════════════════════════════════════════
// 29. Quantized matmul → GELU → layer norm pipeline
// ══════════════════════════════════════════════════════════════════

#[test]
fn quantized_matmul_gelu_layernorm_pipeline() {
    let m: usize = 1;
    let k: usize = 8;
    let n: usize = 4;
    let block_size: usize = 4;

    let activations: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.2 - 0.5).collect();
    let packed_k = k.div_ceil(4);
    let weights: Vec<u8> = vec![pack_i2s([1, -1, 0, 1]); n * packed_k];
    let num_blocks = k.div_ceil(block_size);
    let scales = vec![0.5f32; n * num_blocks];

    let mut logits = vec![0.0f32; m * n];
    i2s_matmul_f32(&activations, &weights, &scales, &mut logits, m, n, k, block_size).unwrap();

    let mut activated = vec![0.0f32; logits.len()];
    gelu_exact_activate(&logits, &mut activated);

    let ln_cfg = LayerNormConfig::new(vec![n]);
    let gamma = vec![1.0f32; n];
    let normed = layer_norm(&activated, &gamma, None, &ln_cfg).unwrap();
    assert_eq!(normed.len(), n);
    assert!(normed.iter().all(|v| v.is_finite()));
}
