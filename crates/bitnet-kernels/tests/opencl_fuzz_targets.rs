//! Fuzz-like proptest targets for OpenCL CPU reference implementations.
//!
//! These tests exercise the CPU reference implementations of the OpenCL kernels
//! with randomised inputs, validating numerical stability, shape handling,
//! quantization round-trips, and pipeline configuration invariants.

use proptest::prelude::*;

// ── Imports from bitnet-kernels ──────────────────────────────────────────────

use bitnet_kernels::opencl_attention::{
    AttentionConfig, AttentionMask, AttentionScores, MultiHeadAttentionRef,
    scaled_dot_product_attention_ref,
};
use bitnet_kernels::opencl_buffer::{AlignedBuffer, BufferPool, optimal_buffer_size};
use bitnet_kernels::opencl_embedding::{embedding_lookup_ref, output_projection_ref};
use bitnet_kernels::opencl_ffn::{ActivationType, ffn_forward_ref, gated_ffn_forward_ref};
use bitnet_kernels::opencl_pipeline::{InferencePipeline, PipelineConfig};
use bitnet_kernels::opencl_quantized::{
    I2sDequantizer, I2sPackedFormat, QuantizedMatVecCpu, build_test_data,
};
use bitnet_kernels::opencl_work_size::{
    ElementwiseWorkSize, MatmulWorkSize, ReductionWorkSize, WorkSizeOptimizer,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Proptest strategies
// ═══════════════════════════════════════════════════════════════════════════════

/// Normal (non-NaN, non-infinite) f32 values.
#[allow(dead_code)]
fn arb_f32_normal() -> impl Strategy<Value = f32> {
    prop::num::f32::NORMAL
}

/// Vector of normal f32 values with length in `1..=max_len`.
#[allow(dead_code)]
fn arb_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(arb_f32_normal(), 1..=max_len)
}

/// Vector of f32 values drawn from a specific range.
#[allow(dead_code)]
fn arb_f32_vec_range(min: f32, max: f32, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(min..=max, 1..=max_len)
}

/// Matrix dimensions `(rows, cols)` with `1..=max`.
#[allow(dead_code)]
fn arb_matrix_dims(max: usize) -> impl Strategy<Value = (usize, usize)> {
    (1..=max, 1..=max)
}

/// Ternary values {-1, 0, +1}.
fn arb_ternary() -> impl Strategy<Value = i8> {
    prop::sample::select(vec![-1i8, 0, 1])
}

/// Vector of ternary values.
#[allow(dead_code)]
fn arb_ternary_vec(max_len: usize) -> impl Strategy<Value = Vec<i8>> {
    prop::collection::vec(arb_ternary(), 1..=max_len)
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Numerical stability fuzzing
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // ── Attention numerical stability ────────────────────────────────────

    #[test]
    fn attention_ref_finite_output(
        seq_len in 1usize..=8,
        kv_len in 1usize..=8,
        head_dim in 1usize..=16,
    ) {
        let q = vec![0.01f32; seq_len * head_dim];
        let k = vec![0.01f32; kv_len * head_dim];
        let v = vec![0.01f32; kv_len * head_dim];
        let mut output = vec![0.0f32; seq_len * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();

        scaled_dot_product_attention_ref(
            &q, &k, &v, &mut output, seq_len, kv_len, head_dim, scale, false,
        );

        for &val in &output {
            prop_assert!(val.is_finite(), "output must be finite, got {val}");
        }
    }

    #[test]
    fn attention_ref_with_random_inputs(
        seq_len in 1usize..=4,
        head_dim in 1usize..=8,
    ) {
        let kv_len = seq_len;
        // Small magnitude to avoid softmax overflow
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.001).collect();
        let k: Vec<f32> = (0..kv_len * head_dim).map(|i| (i as f32) * 0.001).collect();
        let v: Vec<f32> = (0..kv_len * head_dim).map(|i| (i as f32) * 0.01).collect();
        let mut output = vec![0.0f32; seq_len * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();

        scaled_dot_product_attention_ref(
            &q, &k, &v, &mut output, seq_len, kv_len, head_dim, scale, false,
        );

        for &val in &output {
            prop_assert!(val.is_finite(), "attention output must be finite");
        }
    }

    #[test]
    fn attention_causal_mask_finite(
        seq_len in 1usize..=8,
        head_dim in 1usize..=16,
    ) {
        let kv_len = seq_len;
        let q = vec![0.1f32; seq_len * head_dim];
        let k = vec![0.1f32; kv_len * head_dim];
        let v = vec![0.1f32; kv_len * head_dim];
        let mut output = vec![0.0f32; seq_len * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();

        scaled_dot_product_attention_ref(
            &q, &k, &v, &mut output, seq_len, kv_len, head_dim, scale, true,
        );

        for &val in &output {
            prop_assert!(val.is_finite(), "causal attention output must be finite");
        }
    }

    #[test]
    fn attention_near_zero_inputs(
        seq_len in 1usize..=4,
        head_dim in 1usize..=8,
    ) {
        let kv_len = seq_len;
        let q = vec![1e-30f32; seq_len * head_dim];
        let k = vec![1e-30f32; kv_len * head_dim];
        let v = vec![1e-30f32; kv_len * head_dim];
        let mut output = vec![0.0f32; seq_len * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();

        scaled_dot_product_attention_ref(
            &q, &k, &v, &mut output, seq_len, kv_len, head_dim, scale, false,
        );

        for &val in &output {
            prop_assert!(val.is_finite(), "near-zero attention output must be finite");
        }
    }

    #[test]
    fn attention_scores_compute_raw_finite(
        seq_len in 1usize..=8,
        kv_len in 1usize..=8,
        head_dim in 1usize..=16,
    ) {
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i % 7) as f32 * 0.1).collect();
        let k: Vec<f32> = (0..kv_len * head_dim).map(|i| (i % 5) as f32 * 0.1).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let scores = AttentionScores::compute_raw(&q, &k, seq_len, kv_len, head_dim, scale);

        for &w in &scores.weights {
            prop_assert!(w.is_finite(), "raw attention score must be finite");
        }
        prop_assert_eq!(scores.seq_len, seq_len);
        prop_assert_eq!(scores.kv_len, kv_len);
    }

    // ── FFN numerical stability ──────────────────────────────────────────

    #[test]
    fn ffn_forward_finite_output(
        seq_len in 1usize..=4,
        hidden in 1usize..=16,
        inter in 1usize..=16,
    ) {
        let x: Vec<f32> = (0..seq_len * hidden).map(|i| (i as f32) * 0.01).collect();
        let w_up = vec![0.01f32; hidden * inter];
        let w_down = vec![0.01f32; inter * hidden];
        let mut output = vec![0.0f32; seq_len * hidden];

        let result = ffn_forward_ref(
            &x, &w_up, &w_down, &mut output,
            seq_len, hidden, inter, ActivationType::SiLU,
        );
        prop_assert!(result.is_ok(), "ffn_forward_ref should succeed");

        for &val in &output {
            prop_assert!(val.is_finite(), "FFN output must be finite");
        }
    }

    #[test]
    fn gated_ffn_forward_finite_output(
        seq_len in 1usize..=4,
        hidden in 1usize..=16,
        inter in 1usize..=16,
    ) {
        let x: Vec<f32> = (0..seq_len * hidden).map(|i| (i as f32) * 0.01).collect();
        let w_gate = vec![0.01f32; hidden * inter];
        let w_up = vec![0.01f32; hidden * inter];
        let w_down = vec![0.01f32; inter * hidden];
        let mut output = vec![0.0f32; seq_len * hidden];

        let result = gated_ffn_forward_ref(
            &x, &w_gate, &w_up, &w_down, &mut output,
            seq_len, hidden, inter, ActivationType::SiLU,
        );
        prop_assert!(result.is_ok(), "gated_ffn_forward_ref should succeed");

        for &val in &output {
            prop_assert!(val.is_finite(), "gated FFN output must be finite");
        }
    }

    #[test]
    fn ffn_all_activations_finite(
        seq_len in 1usize..=2,
        hidden in 1usize..=8,
        inter in 1usize..=8,
        act_idx in 0usize..=4,
    ) {
        let activations = [
            ActivationType::SiLU,
            ActivationType::GELU,
            ActivationType::GELUApprox,
            ActivationType::ReLU,
            ActivationType::Swish,
        ];
        let activation = activations[act_idx];
        let x: Vec<f32> = (0..seq_len * hidden).map(|i| (i as f32) * 0.02 - 0.5).collect();
        let w_up = vec![0.05f32; hidden * inter];
        let w_down = vec![0.05f32; inter * hidden];
        let mut output = vec![0.0f32; seq_len * hidden];

        let result = ffn_forward_ref(
            &x, &w_up, &w_down, &mut output,
            seq_len, hidden, inter, activation,
        );
        prop_assert!(result.is_ok());
        for &val in &output {
            prop_assert!(val.is_finite(), "FFN output with {:?} must be finite", activation);
        }
    }

    #[test]
    fn activation_apply_large_magnitude(val in -1e10f32..1e10f32) {
        // All activations should produce finite results for large-magnitude inputs
        let activations = [
            ActivationType::SiLU,
            ActivationType::GELU,
            ActivationType::GELUApprox,
            ActivationType::ReLU,
            ActivationType::Swish,
        ];
        for act in &activations {
            let result = act.apply(val);
            prop_assert!(result.is_finite(), "{:?}({val}) = {result} is not finite", act);
        }
    }

    // ── Embedding numerical stability ────────────────────────────────────

    #[test]
    fn embedding_lookup_finite(
        vocab_size in 1usize..=32,
        embed_dim in 1usize..=16,
        seq_len in 1usize..=8,
    ) {
        let weight: Vec<f32> = (0..vocab_size * embed_dim)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let token_ids: Vec<u32> = (0..seq_len).map(|i| (i % vocab_size) as u32).collect();
        let mut output = vec![0.0f32; seq_len * embed_dim];

        let result = embedding_lookup_ref(
            &token_ids, &weight, &mut output, vocab_size, embed_dim, None,
        );
        prop_assert!(result.is_ok());
        for &val in &output {
            prop_assert!(val.is_finite());
        }
    }

    #[test]
    fn output_projection_finite(
        seq_len in 1usize..=4,
        hidden in 1usize..=16,
        vocab in 1usize..=16,
    ) {
        let h: Vec<f32> = (0..seq_len * hidden).map(|i| (i as f32) * 0.01).collect();
        let w = vec![0.01f32; vocab * hidden];
        let mut output = vec![0.0f32; seq_len * vocab];

        let result = output_projection_ref(&h, &w, &mut output, seq_len, hidden, vocab);
        prop_assert!(result.is_ok());
        for &val in &output {
            prop_assert!(val.is_finite(), "output projection must be finite");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Shape fuzzing
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn attention_edge_shapes_1xn(
        kv_len in 1usize..=16,
        head_dim in 1usize..=16,
    ) {
        let seq_len = 1;
        let q = vec![0.1f32; seq_len * head_dim];
        let k = vec![0.1f32; kv_len * head_dim];
        let v = vec![0.1f32; kv_len * head_dim];
        let mut output = vec![0.0f32; seq_len * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();

        scaled_dot_product_attention_ref(
            &q, &k, &v, &mut output, seq_len, kv_len, head_dim, scale, false,
        );

        for &val in &output {
            prop_assert!(val.is_finite());
        }
    }

    #[test]
    fn attention_edge_shapes_nx1(
        seq_len in 1usize..=16,
        head_dim in 1usize..=16,
    ) {
        let kv_len = 1;
        let q = vec![0.1f32; seq_len * head_dim];
        let k = vec![0.1f32; kv_len * head_dim];
        let v = vec![0.1f32; kv_len * head_dim];
        let mut output = vec![0.0f32; seq_len * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();

        scaled_dot_product_attention_ref(
            &q, &k, &v, &mut output, seq_len, kv_len, head_dim, scale, false,
        );

        for &val in &output {
            prop_assert!(val.is_finite());
        }
    }

    #[test]
    fn attention_1x1_shape(head_dim in 1usize..=64) {
        let q = vec![1.0f32; head_dim];
        let k = vec![1.0f32; head_dim];
        let v = vec![1.0f32; head_dim];
        let mut output = vec![0.0f32; head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();

        scaled_dot_product_attention_ref(
            &q, &k, &v, &mut output, 1, 1, head_dim, scale, false,
        );
        for &val in &output {
            prop_assert!((val - 1.0).abs() < 1e-5, "1×1 attention should pass through V");
        }
    }

    #[test]
    fn ffn_square_shape(dim in 1usize..=32) {
        let x = vec![0.1f32; dim];
        let w_up = vec![0.01f32; dim * dim];
        let w_down = vec![0.01f32; dim * dim];
        let mut output = vec![0.0f32; dim];

        let result = ffn_forward_ref(
            &x, &w_up, &w_down, &mut output,
            1, dim, dim, ActivationType::ReLU,
        );
        prop_assert!(result.is_ok());
    }

    #[test]
    fn ffn_wide_intermediate(
        hidden in 1usize..=8,
        multiplier in 2usize..=8,
    ) {
        let inter = hidden * multiplier;
        let x = vec![0.1f32; hidden];
        let w_up = vec![0.01f32; hidden * inter];
        let w_down = vec![0.01f32; inter * hidden];
        let mut output = vec![0.0f32; hidden];

        let result = ffn_forward_ref(
            &x, &w_up, &w_down, &mut output,
            1, hidden, inter, ActivationType::SiLU,
        );
        prop_assert!(result.is_ok());
    }

    #[test]
    fn gated_ffn_shape_variety(
        seq_len in 1usize..=4,
        hidden in 1usize..=8,
        inter in 1usize..=16,
    ) {
        let x = vec![0.02f32; seq_len * hidden];
        let w_gate = vec![0.01f32; hidden * inter];
        let w_up = vec![0.01f32; hidden * inter];
        let w_down = vec![0.01f32; inter * hidden];
        let mut output = vec![0.0f32; seq_len * hidden];

        let result = gated_ffn_forward_ref(
            &x, &w_gate, &w_up, &w_down, &mut output,
            seq_len, hidden, inter, ActivationType::SiLU,
        );
        prop_assert!(result.is_ok());
    }

    #[test]
    fn embedding_oov_tokens_produce_zeros(
        vocab_size in 1usize..=16,
        embed_dim in 1usize..=8,
        seq_len in 1usize..=8,
    ) {
        let weight: Vec<f32> = vec![1.0; vocab_size * embed_dim];
        // All tokens are out-of-vocabulary
        let token_ids: Vec<u32> = vec![vocab_size as u32 + 10; seq_len];
        let mut output = vec![99.0f32; seq_len * embed_dim];

        let result = embedding_lookup_ref(
            &token_ids, &weight, &mut output, vocab_size, embed_dim, None,
        );
        prop_assert!(result.is_ok());
        for &val in &output {
            prop_assert_eq!(val, 0.0, "OOV tokens must produce zero vectors");
        }
    }

    #[test]
    fn embedding_padding_idx_produces_zeros(
        vocab_size in 2usize..=16,
        embed_dim in 1usize..=8,
    ) {
        let weight: Vec<f32> = vec![1.0; vocab_size * embed_dim];
        let padding_idx = 0u32;
        let token_ids = vec![padding_idx];
        let mut output = vec![99.0f32; embed_dim];

        let result = embedding_lookup_ref(
            &token_ids, &weight, &mut output, vocab_size, embed_dim, Some(padding_idx),
        );
        prop_assert!(result.is_ok());
        for &val in &output {
            prop_assert_eq!(val, 0.0, "padding token must produce zero vector");
        }
    }

    #[test]
    fn work_size_1d_nonzero(elements in 1usize..=4096) {
        let result = ElementwiseWorkSize::optimize(elements);
        prop_assert!(result.global_work_size[0] >= elements);
        if let Some(ref local) = result.local_work_size {
            prop_assert!(local[0] > 0);
        }
    }

    #[test]
    fn work_size_2d_nonzero(
        rows in 1usize..=512,
        cols in 1usize..=512,
    ) {
        let optimizer = WorkSizeOptimizer::intel_arc();
        let result = optimizer.optimize_2d(rows, cols);
        prop_assert!(result.global_work_size[0] >= cols);
        prop_assert!(result.global_work_size[1] >= rows);
        if let Some(ref local) = result.local_work_size {
            prop_assert!(local[0] > 0);
            prop_assert!(local[1] > 0);
        }
    }

    #[test]
    fn work_size_3d_nonzero(
        batch in 1usize..=16,
        rows in 1usize..=64,
        cols in 1usize..=64,
    ) {
        let optimizer = WorkSizeOptimizer::intel_arc();
        let result = optimizer.optimize_3d(batch, rows, cols);
        prop_assert!(result.global_work_size[0] > 0);
        prop_assert!(result.global_work_size[1] > 0);
        prop_assert!(result.global_work_size[2] > 0);
    }

    #[test]
    fn matmul_work_size_covers_output(
        m in 1usize..=512,
        n in 1usize..=512,
    ) {
        let result = MatmulWorkSize::optimize(m, n);
        prop_assert!(result.global_work_size[0] >= n);
        prop_assert!(result.global_work_size[1] >= m);
    }

    #[test]
    fn reduction_work_size_nonzero(
        rows in 1usize..=256,
        cols in 1usize..=256,
    ) {
        let result = ReductionWorkSize::optimize(rows, cols);
        prop_assert!(result.global_work_size[0] > 0);
        if let Some(ref local) = result.local_work_size {
            prop_assert!(local[0] > 0);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Quantization fuzzing
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn i2s_pack_unpack_roundtrip(
        values in prop::collection::vec(arb_ternary(), 1..=256)
    ) {
        let packed = I2sPackedFormat::pack(&values);
        let unpacked = I2sPackedFormat::unpack(&packed, values.len());
        prop_assert_eq!(&values, &unpacked, "pack/unpack must round-trip");
    }

    #[test]
    fn i2s_packed_len_correct(count in 1usize..=1024) {
        let packed_len = I2sPackedFormat::packed_len(count);
        prop_assert_eq!(packed_len, count.div_ceil(4));
    }

    #[test]
    fn i2s_unpack_one_matches_unpack(
        values in prop::collection::vec(arb_ternary(), 1..=64)
    ) {
        let packed = I2sPackedFormat::pack(&values);
        for (i, &expected) in values.iter().enumerate() {
            let got = I2sPackedFormat::unpack_one(&packed, i);
            prop_assert_eq!(got, expected, "unpack_one({}) mismatch", i);
        }
    }

    #[test]
    fn dequantize_row_finite(
        cols in 4usize..=64,
        scale in -10.0f32..10.0f32,
    ) {
        // cols must be multiple of 4 for clean packing
        let cols = (cols / 4) * 4;
        if cols == 0 { return Ok(()); }
        let ternary: Vec<i8> = (0..cols).map(|i| [-1, 0, 1][i % 3]).collect();
        let packed = I2sPackedFormat::pack(&ternary);
        let block_size = cols; // single block
        let scales = vec![scale];
        let row = I2sDequantizer::dequantize_row(&packed, &scales, cols, block_size);
        prop_assert_eq!(row.len(), cols);
        for &val in &row {
            prop_assert!(val.is_finite(), "dequantized value must be finite");
        }
    }

    #[test]
    fn dequantize_matrix_finite(
        rows in 1usize..=8,
        block_mult in 1usize..=4,
    ) {
        let block_size = 4;
        let cols = block_size * block_mult;
        let ternary: Vec<i8> = (0..rows * cols).map(|i| [-1, 0, 1][i % 3]).collect();
        let (packed, scales) = build_test_data(&ternary, rows, cols, block_size, 1.0);
        let deq = I2sDequantizer::dequantize_matrix(&packed, &scales, rows, cols, block_size);
        prop_assert_eq!(deq.len(), rows * cols);
        for &val in &deq {
            prop_assert!(val.is_finite());
        }
    }

    #[test]
    fn matvec_ref_agrees_with_block_opt(
        rows in 1usize..=8,
        block_mult in 1usize..=4,
    ) {
        let block_size = 4;
        let cols = block_size * block_mult;
        let ternary: Vec<i8> = (0..rows * cols).map(|i| [-1, 0, 1][i % 3]).collect();
        let (packed, scales) = build_test_data(&ternary, rows, cols, block_size, 1.5);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1).collect();

        let mut y_ref = vec![0.0f32; rows];
        let mut y_opt = vec![0.0f32; rows];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y_ref, rows, cols, block_size);
        QuantizedMatVecCpu::matvec_block_opt(&packed, &scales, &x, &mut y_opt, rows, cols, block_size);

        for i in 0..rows {
            let diff = (y_ref[i] - y_opt[i]).abs();
            prop_assert!(diff < 1e-4, "row {i}: ref={} opt={} diff={diff}", y_ref[i], y_opt[i]);
        }
    }

    #[test]
    fn matvec_zero_weights_produce_zero_output(
        rows in 1usize..=16,
        block_mult in 1usize..=4,
    ) {
        let block_size = 4;
        let cols = block_size * block_mult;
        // All-zero ternary weights
        let ternary = vec![0i8; rows * cols];
        let (packed, scales) = build_test_data(&ternary, rows, cols, block_size, 1.0);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) + 1.0).collect();

        let mut y = vec![99.0f32; rows];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, rows, cols, block_size);
        for (i, &val) in y.iter().enumerate() {
            prop_assert!((val).abs() < 1e-6, "row {i}: expected ~0, got {val}");
        }
    }

    #[test]
    fn matvec_zero_input_produces_zero_output(
        rows in 1usize..=16,
        block_mult in 1usize..=4,
    ) {
        let block_size = 4;
        let cols = block_size * block_mult;
        let ternary: Vec<i8> = (0..rows * cols).map(|i| [-1, 0, 1][i % 3]).collect();
        let (packed, scales) = build_test_data(&ternary, rows, cols, block_size, 2.0);
        let x = vec![0.0f32; cols];

        let mut y = vec![99.0f32; rows];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, rows, cols, block_size);
        for (i, &val) in y.iter().enumerate() {
            prop_assert!((val).abs() < 1e-6, "row {i}: expected ~0, got {val}");
        }
    }

    #[test]
    fn matvec_compare_returns_small_error(
        rows in 1usize..=8,
        block_mult in 1usize..=4,
    ) {
        let block_size = 4;
        let cols = block_size * block_mult;
        let ternary: Vec<i8> = (0..rows * cols).map(|i| [-1, 0, 1][i % 3]).collect();
        let (packed, scales) = build_test_data(&ternary, rows, cols, block_size, 1.0);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.05).collect();

        let max_err = QuantizedMatVecCpu::compare_implementations(
            &packed, &scales, &x, rows, cols, block_size,
        );
        prop_assert!(max_err < 1e-4, "max error between ref and opt: {max_err}");
    }

    #[test]
    fn dequantize_roundtrip_identity(
        values in prop::collection::vec(arb_ternary(), 4..=64)
    ) {
        // Ensure length is multiple of 4
        let len = (values.len() / 4) * 4;
        let values = &values[..len];
        let packed = I2sPackedFormat::pack(values);
        let scale = 1.0f32;
        let deq = I2sDequantizer::dequantize_row(&packed, &[scale], len, len);
        for (i, (&orig, &deq_val)) in values.iter().zip(deq.iter()).enumerate() {
            prop_assert_eq!(
                orig as f32, deq_val,
                "dequantize mismatch at {}: orig={}, deq={}", i, orig, deq_val
            );
        }
    }

    #[test]
    fn dequantize_scale_applied_correctly(
        scale in 0.1f32..10.0f32,
        block_mult in 1usize..=4,
    ) {
        let block_size = 4;
        let cols = block_size * block_mult;
        // All +1 ternary values
        let ternary = vec![1i8; cols];
        let packed = I2sPackedFormat::pack(&ternary);
        let scales = vec![scale; block_mult];
        let deq = I2sDequantizer::dequantize_row(&packed, &scales, cols, block_size);
        for &val in &deq {
            prop_assert!((val - scale).abs() < 1e-5, "expected {scale}, got {val}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Pipeline fuzzing
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn pipeline_valid_config_executes(
        num_layers in 1usize..=4,
        num_heads in 1usize..=8,
        head_dim_mult in 1usize..=4,
    ) {
        let head_dim = head_dim_mult * 2; // keep head_dim even
        let hidden_dim = num_heads * head_dim;
        let intermediate_dim = hidden_dim * 2;
        let config = PipelineConfig {
            num_layers,
            hidden_dim,
            num_heads,
            head_dim,
            intermediate_dim,
            vocab_size: 32,
            max_seq_len: 64,
            use_gpu: false,
            fallback_to_cpu: true,
        };
        let pipeline = InferencePipeline::new(config);
        prop_assert!(pipeline.is_ok(), "valid config should create pipeline");

        let mut pipeline = pipeline.unwrap();
        let result = pipeline.execute_single_token_cpu(&[1], 0);
        prop_assert!(result.is_ok(), "single-token execution should succeed");
    }

    #[test]
    fn pipeline_stages_per_token_formula(num_layers in 1usize..=8) {
        let config = PipelineConfig {
            num_layers,
            hidden_dim: 16,
            num_heads: 4,
            head_dim: 4,
            intermediate_dim: 32,
            vocab_size: 32,
            max_seq_len: 64,
            use_gpu: false,
            fallback_to_cpu: true,
        };
        let pipeline = InferencePipeline::new(config).unwrap();
        let expected = 1 + num_layers * 4 + 2;
        prop_assert_eq!(pipeline.stages_per_token(), expected);
    }

    #[test]
    fn pipeline_stage_order_length(num_layers in 1usize..=8) {
        let config = PipelineConfig {
            num_layers,
            hidden_dim: 16,
            num_heads: 4,
            head_dim: 4,
            intermediate_dim: 32,
            vocab_size: 32,
            max_seq_len: 64,
            use_gpu: false,
            fallback_to_cpu: true,
        };
        let pipeline = InferencePipeline::new(config).unwrap();
        let order = pipeline.stage_order();
        prop_assert_eq!(order.len(), pipeline.stages_per_token());
    }

    #[test]
    fn pipeline_position_within_bounds(
        position in 0usize..=62,
    ) {
        let config = PipelineConfig {
            num_layers: 1,
            hidden_dim: 8,
            num_heads: 2,
            head_dim: 4,
            intermediate_dim: 16,
            vocab_size: 16,
            max_seq_len: 64,
            use_gpu: false,
            fallback_to_cpu: true,
        };
        let mut pipeline = InferencePipeline::new(config).unwrap();
        let result = pipeline.execute_single_token_cpu(&[0], position);
        prop_assert!(result.is_ok(), "position {position} should be valid");
    }

    #[test]
    fn pipeline_execution_count_increments(n_executions in 1usize..=8) {
        let config = PipelineConfig {
            num_layers: 1,
            hidden_dim: 8,
            num_heads: 2,
            head_dim: 4,
            intermediate_dim: 16,
            vocab_size: 16,
            max_seq_len: 64,
            use_gpu: false,
            fallback_to_cpu: true,
        };
        let mut pipeline = InferencePipeline::new(config).unwrap();
        for i in 0..n_executions {
            let _ = pipeline.execute_single_token_cpu(&[0], i % 63);
        }
        prop_assert_eq!(pipeline.execution_count(), n_executions);
    }

    // ── Buffer pool fuzzing ──────────────────────────────────────────────

    #[test]
    fn buffer_pool_allocate_return_cycle(
        size in 1usize..=4096,
        alignment in prop::sample::select(vec![1usize, 4, 8, 16, 64, 256]),
    ) {
        let pool = BufferPool::new();
        let buf = pool.allocate(size, alignment);
        prop_assert!(buf.len() >= size);
        prop_assert!(buf.alignment() == alignment);
        pool.return_buffer(buf);
        let stats = pool.stats();
        prop_assert!(stats.pooled >= 1);
    }

    #[test]
    fn aligned_buffer_initialized_to_zero(
        size in 1usize..=1024,
        alignment in prop::sample::select(vec![1usize, 4, 16, 64]),
    ) {
        let buf: AlignedBuffer<f32> = AlignedBuffer::new(size, alignment);
        for &val in buf.as_slice() {
            prop_assert_eq!(val, 0.0f32, "new buffer should be zero-initialized");
        }
    }

    #[test]
    fn optimal_buffer_size_at_least_requested(
        elements in 1usize..=4096,
        elem_size in prop::sample::select(vec![1usize, 2, 4, 8]),
    ) {
        let opt = optimal_buffer_size(elements, elem_size);
        prop_assert!(opt >= elements * elem_size, "optimal size {opt} < requested {}", elements * elem_size);
    }

    // ── Attention mask fuzzing ───────────────────────────────────────────

    #[test]
    fn causal_mask_diagonal_allowed(seq_len in 1usize..=32) {
        let mask = AttentionMask::causal(seq_len, seq_len, 0);
        for i in 0..seq_len {
            prop_assert!(mask.allows(i, i), "position ({i},{i}) must be allowed");
        }
    }

    #[test]
    fn causal_mask_future_blocked(seq_len in 2usize..=32) {
        let mask = AttentionMask::causal(seq_len, seq_len, 0);
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                prop_assert!(!mask.allows(i, j), "future position ({i},{j}) must be blocked");
            }
        }
    }

    #[test]
    fn none_mask_all_allowed(
        seq_len in 1usize..=16,
        kv_len in 1usize..=16,
    ) {
        let mask = AttentionMask::none(seq_len, kv_len);
        for i in 0..seq_len {
            for j in 0..kv_len {
                prop_assert!(mask.allows(i, j), "none mask must allow ({i},{j})");
            }
        }
    }

    // ── Multi-head attention shape ───────────────────────────────────────

    #[test]
    fn multi_head_attention_output_shape(
        num_heads in 1usize..=4,
        head_dim in 1usize..=8,
        seq_len in 1usize..=4,
    ) {
        let kv_len = seq_len;
        let total_dim = num_heads * head_dim;
        let config = AttentionConfig::new(num_heads, head_dim, seq_len + 1, false).unwrap();

        let q = vec![0.1f32; seq_len * total_dim];
        let k = vec![0.1f32; kv_len * total_dim];
        let v = vec![0.1f32; kv_len * total_dim];
        let mut output = vec![0.0f32; seq_len * total_dim];

        MultiHeadAttentionRef::forward(&config, &q, &k, &v, &mut output, seq_len, kv_len);

        for &val in &output {
            prop_assert!(val.is_finite(), "MHA output must be finite");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. Additional targeted property tests
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn attention_softmax_sums_to_one(
        seq_len in 1usize..=4,
        kv_len in 1usize..=8,
        head_dim in 1usize..=8,
    ) {
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i % 5) as f32 * 0.1).collect();
        let k: Vec<f32> = (0..kv_len * head_dim).map(|i| (i % 7) as f32 * 0.1).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut scores = AttentionScores::compute_raw(&q, &k, seq_len, kv_len, head_dim, scale);
        scores.softmax();

        for i in 0..seq_len {
            let row_sum: f32 = (0..kv_len).map(|j| scores.weights[i * kv_len + j]).sum();
            prop_assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "softmax row {i} sums to {row_sum}, expected ~1.0"
            );
        }
    }

    #[test]
    fn embedding_correct_row_copied(
        vocab_size in 2usize..=32,
        embed_dim in 1usize..=16,
        token_idx in 0usize..=31,
    ) {
        let token_idx = token_idx % vocab_size;
        let weight: Vec<f32> = (0..vocab_size * embed_dim)
            .map(|i| i as f32)
            .collect();
        let token_ids = vec![token_idx as u32];
        let mut output = vec![0.0f32; embed_dim];

        embedding_lookup_ref(
            &token_ids, &weight, &mut output, vocab_size, embed_dim, None,
        ).unwrap();

        let expected_start = token_idx * embed_dim;
        for d in 0..embed_dim {
            prop_assert_eq!(
                output[d], weight[expected_start + d],
                "embedding dim {} mismatch", d
            );
        }
    }

    #[test]
    fn output_projection_linear(
        hidden in 2usize..=8,
        vocab in 1usize..=8,
        scale in 0.1f32..5.0f32,
    ) {
        // Test linearity: proj(a*x) = a * proj(x)
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let w = vec![0.1f32; vocab * hidden];

        let mut out1 = vec![0.0f32; vocab];
        let scaled_x: Vec<f32> = x.iter().map(|&v| v * scale).collect();
        let mut out2 = vec![0.0f32; vocab];

        output_projection_ref(&x, &w, &mut out1, 1, hidden, vocab).unwrap();
        output_projection_ref(&scaled_x, &w, &mut out2, 1, hidden, vocab).unwrap();

        for i in 0..vocab {
            let expected = out1[i] * scale;
            let diff = (out2[i] - expected).abs();
            prop_assert!(
                diff < 1e-3,
                "linearity violated at {i}: {expected} vs {}", out2[i]
            );
        }
    }

    #[test]
    fn quantized_matvec_finite_output(
        rows in 1usize..=16,
        block_mult in 1usize..=4,
        scale_val in -5.0f32..5.0f32,
    ) {
        let block_size = 4;
        let cols = block_size * block_mult;
        let ternary: Vec<i8> = (0..rows * cols).map(|i| [-1, 0, 1][i % 3]).collect();
        let (packed, scales) = build_test_data(&ternary, rows, cols, block_size, scale_val);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1).collect();

        let mut y = vec![0.0f32; rows];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, rows, cols, block_size);

        for &val in &y {
            prop_assert!(val.is_finite(), "quantized matvec output must be finite");
        }
    }

    #[test]
    fn i2s_all_zeros_pack_to_specific_pattern(count in 1usize..=128) {
        // All zeros → encoded as 0b01 per value
        let values = vec![0i8; count];
        let packed = I2sPackedFormat::pack(&values);
        for (byte_idx, &byte) in packed.iter().enumerate() {
            let vals_in_byte = (count - byte_idx * 4).min(4);
            let mut expected = 0u8;
            for j in 0..vals_in_byte {
                expected |= 1u8 << (j * 2); // 0b01 per slot
            }
            prop_assert_eq!(byte, expected, "byte {}: expected {:#04x}, got {:#04x}", byte_idx, expected, byte);
        }
    }

    #[test]
    fn work_size_tiled_matmul_covers_output(
        m in 1usize..=256,
        n in 1usize..=256,
        tile_size in prop::sample::select(vec![8usize, 16, 32]),
    ) {
        let result = MatmulWorkSize::optimize_with_tile(m, n, tile_size);
        prop_assert!(result.global_work_size[0] >= n);
        prop_assert!(result.global_work_size[1] >= m);
    }
}
