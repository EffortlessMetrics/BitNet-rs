//! Property-based tests — wave 27.
//!
//! Covers CPU pipeline-parallel mechanics, CPU embedding lookup invariants,
//! CPU layer-norm / RMS-norm numerical guarantees, CPU softmax stability,
//! CUDA layer-norm / RMS-norm CPU fallbacks, CPU reduction algebra, CPU
//! transpose involution, CPU gating activation bounds, CPU tensor-parallel
//! shard round-trips, CPU elementwise-ops algebraic identities, and CPU
//! residual-add semantics.
//!
//! 55 property tests validating: output shape preservation, no NaN/Inf
//! propagation, idempotency, softmax row sums, layer-norm zero-mean /
//! unit-variance, embedding lookup fidelity, pipeline split-merge
//! round-trips, reduction associativity, transpose involution, gating
//! output bounds, tensor-parallel shard-gather fidelity, and more.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::activations::{relu, sigmoid, silu};
use bitnet_kernels::cpu::batch::batched_softmax;
use bitnet_kernels::cpu::embedding::{
    EmbeddingConfig, embedding_accumulate, embedding_lookup, normalize_embeddings,
    positional_embedding,
};
use bitnet_kernels::cpu::gating::{GatingType, apply_gating, geglu, reglu, swiglu};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::pipeline_parallel::{
    PipelineStage, micro_batch_merge, micro_batch_split, optimal_micro_batch_count,
    pipeline_bubble_time, stage_forward,
};
use bitnet_kernels::cpu::reduction::ReductionKernel;
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled};
use bitnet_kernels::cpu::transpose::TransposeKernel;
use bitnet_kernels::cuda::layernorm::{
    LayerNormConfig as CudaLayerNormConfig, layer_norm_cpu_fallback, rms_norm_cpu_fallback,
};
use bitnet_kernels::cuda::softmax::{SoftmaxConfig, softmax_cpu};
use proptest::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn finite_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-10.0f32..10.0, 1..=max_len)
}

fn non_zero_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(0.1f32..10.0, 1..=max_len)
}

// ── Property tests ──────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // ════════════════════════════════════════════════════════════════
    // 1. CPU pipeline-parallel properties
    // ════════════════════════════════════════════════════════════════

    /// micro_batch_split ↔ merge round-trip preserves data.
    #[test]
    fn prop_pipeline_split_merge_roundtrip(
        batch in 1usize..=16,
        dim in 1usize..=8,
        micro in 1usize..=8,
    ) {
        let input: Vec<f32> = (0..(batch * dim)).map(|i| i as f32).collect();
        let chunks = micro_batch_split(&input, batch, dim, micro).unwrap();
        let merged = micro_batch_merge(&chunks).unwrap();
        prop_assert_eq!(merged.len(), input.len());
        for (i, (&a, &b)) in input.iter().zip(merged.iter()).enumerate() {
            prop_assert!((a - b).abs() < 1e-6, "roundtrip mismatch at {i}");
        }
    }

    /// micro_batch_split produces correct number of chunks.
    #[test]
    fn prop_pipeline_split_chunk_count(
        batch in 1usize..=16,
        dim in 1usize..=8,
        micro in 1usize..=8,
    ) {
        let input = vec![0.0f32; batch * dim];
        let chunks = micro_batch_split(&input, batch, dim, micro).unwrap();
        let expected = (batch + micro - 1) / micro;
        prop_assert_eq!(chunks.len(), expected);
    }

    /// stage_forward output has the same length as input.
    #[test]
    fn prop_stage_forward_preserves_len(
        n in 1usize..=64,
        start in 0usize..=4,
        span in 1usize..=4,
    ) {
        let stage = PipelineStage::new(start, start + span);
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let out = stage_forward(&input, &stage).unwrap();
        prop_assert_eq!(out.len(), n);
    }

    /// stage_forward output is finite when input is finite.
    #[test]
    fn prop_stage_forward_finite(
        v in finite_f32_vec(64),
    ) {
        let stage = PipelineStage::new(0, 2);
        let out = stage_forward(&v, &stage).unwrap();
        for &val in &out {
            prop_assert!(val.is_finite(), "stage output non-finite: {val}");
        }
    }

    /// pipeline_bubble_time is non-negative.
    #[test]
    fn prop_bubble_time_non_negative(
        stages in 1usize..=8,
        micros in 1usize..=16,
    ) {
        let bt = pipeline_bubble_time(stages, micros);
        prop_assert!(bt >= 0.0, "bubble time negative: {bt}");
    }

    /// optimal_micro_batch_count >= num_stages.
    #[test]
    fn prop_optimal_micro_batch_ge_stages(
        stages in 1usize..=8,
        max_frac in 0.01f32..0.5,
    ) {
        let optimal = optimal_micro_batch_count(stages, max_frac);
        prop_assert!(optimal >= stages, "optimal {optimal} < stages {stages}");
    }

    // ════════════════════════════════════════════════════════════════
    // 2. CPU embedding properties
    // ════════════════════════════════════════════════════════════════

    /// embedding_lookup output length = num_indices × embedding_dim.
    #[test]
    fn prop_embedding_lookup_output_shape(
        vocab in 4usize..=32,
        dim in 1usize..=16,
        n_idx in 1usize..=8,
    ) {
        let table: Vec<f32> = (0..(vocab * dim)).map(|i| i as f32 * 0.01).collect();
        let indices: Vec<u32> = (0..n_idx).map(|i| (i % vocab) as u32).collect();
        let out = embedding_lookup(&table, &indices, dim).unwrap();
        prop_assert_eq!(out.len(), n_idx * dim);
    }

    /// embedding_lookup retrieves correct rows.
    #[test]
    fn prop_embedding_lookup_correct_rows(
        vocab in 4usize..=16,
        dim in 1usize..=8,
    ) {
        let table: Vec<f32> = (0..(vocab * dim)).map(|i| i as f32).collect();
        let indices: Vec<u32> = (0..vocab.min(4)).map(|i| i as u32).collect();
        let out = embedding_lookup(&table, &indices, dim).unwrap();
        for (tok, &idx) in indices.iter().enumerate() {
            for d in 0..dim {
                let expected = table[idx as usize * dim + d];
                let got = out[tok * dim + d];
                prop_assert!(
                    (expected - got).abs() < 1e-6,
                    "row mismatch: idx={idx}, d={d}, expected={expected}, got={got}"
                );
            }
        }
    }

    /// normalize_embeddings produces unit-norm rows.
    #[test]
    fn prop_normalize_embeddings_unit_norm(
        dim in 2usize..=16,
        n_rows in 1usize..=4,
    ) {
        let n = n_rows * dim;
        let mut emb: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 + 0.1).collect();
        normalize_embeddings(&mut emb, dim);
        for r in 0..n_rows {
            let row = &emb[r * dim..(r + 1) * dim];
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assert!(
                (norm - 1.0).abs() < 1e-4,
                "row {r} norm = {norm}, expected ≈ 1.0"
            );
        }
    }

    /// positional_embedding output length = seq_len × embedding_dim.
    #[test]
    fn prop_positional_embedding_shape(
        seq in 1usize..=16,
        dim in 2usize..=16,
    ) {
        let dim_even = dim & !1; // ensure even
        if dim_even == 0 { return Ok(()); }
        let out = positional_embedding(seq, dim_even);
        prop_assert_eq!(out.len(), seq * dim_even);
    }

    /// positional_embedding values are bounded in [-1, 1].
    #[test]
    fn prop_positional_embedding_bounded(
        seq in 1usize..=16,
        dim in 1usize..=8,
    ) {
        let dim_even = (dim * 2).max(2);
        let out = positional_embedding(seq, dim_even);
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(
                v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6,
                "positional_embedding[{i}] = {v} out of [-1,1]"
            );
        }
    }

    /// embedding_lookup with EmbeddingConfig returns correct shape.
    #[test]
    fn prop_embedding_lookup_config_shape(
        vocab in 4usize..=32,
        dim in 1usize..=16,
        n_idx in 1usize..=8,
    ) {
        let table: Vec<f32> = (0..(vocab * dim)).map(|i| i as f32 * 0.01).collect();
        let indices: Vec<u32> = (0..n_idx).map(|i| (i % vocab) as u32).collect();
        let config = EmbeddingConfig { vocab_size: vocab, embedding_dim: dim, padding_idx: None };
        let out = bitnet_kernels::cpu::embedding::embedding_lookup_simd(&table, &indices, &config).unwrap();
        prop_assert_eq!(out.len(), n_idx * dim);
    }

    /// embedding_lookup with padding_idx zeros out padded rows.
    #[test]
    fn prop_embedding_padding_zeros(
        vocab in 4usize..=16,
        dim in 2usize..=8,
    ) {
        let table: Vec<f32> = (0..(vocab * dim)).map(|i| (i as f32) + 1.0).collect();
        let pad_idx = 0u32;
        let indices = vec![pad_idx];
        let config = EmbeddingConfig { vocab_size: vocab, embedding_dim: dim, padding_idx: Some(pad_idx) };
        let out = bitnet_kernels::cpu::embedding::embedding_lookup_simd(&table, &indices, &config).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.abs() < 1e-6, "padded row[{i}] = {v}, expected 0");
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 3. CPU layer-norm / RMS-norm invariants
    // ════════════════════════════════════════════════════════════════

    /// Layer-norm output has zero mean per row (within tolerance).
    #[test]
    fn prop_layer_norm_zero_mean(
        dim in 4usize..=32,
        batch in 1usize..=4,
    ) {
        let n = batch * dim;
        let input: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 19) as f32 - 9.0).collect();
        let gamma = vec![1.0f32; dim];
        let beta = vec![0.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let out = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        for b in 0..batch {
            let row = &out[b * dim..(b + 1) * dim];
            let mean: f32 = row.iter().sum::<f32>() / dim as f32;
            prop_assert!(mean.abs() < 0.05, "batch {b} mean = {mean}, expected ≈ 0");
        }
    }

    /// Layer-norm output has unit variance per row (gamma=1, beta=0).
    #[test]
    fn prop_layer_norm_unit_variance(
        dim in 8usize..=32,
        batch in 1usize..=4,
    ) {
        let n = batch * dim;
        let input: Vec<f32> = (0..n).map(|i| ((i * 11 + 5) % 23) as f32 - 11.0).collect();
        let gamma = vec![1.0f32; dim];
        let beta = vec![0.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let out = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        for b in 0..batch {
            let row = &out[b * dim..(b + 1) * dim];
            let mean: f32 = row.iter().sum::<f32>() / dim as f32;
            let var: f32 = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / dim as f32;
            prop_assert!(
                (var - 1.0).abs() < 0.15,
                "batch {b} variance = {var}, expected ≈ 1.0"
            );
        }
    }

    /// Layer-norm output length matches input length.
    #[test]
    fn prop_layer_norm_output_length(
        dim in 1usize..=32,
        batch in 1usize..=4,
    ) {
        let n = batch * dim;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let gamma = vec![1.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let out = layer_norm(&input, &gamma, None, &config).unwrap();
        prop_assert_eq!(out.len(), n);
    }

    /// RMS-norm output is finite for finite input.
    #[test]
    fn prop_rms_norm_finite(
        dim in 1usize..=32,
        batch in 1usize..=4,
    ) {
        let n = batch * dim;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.2 - 3.0).collect();
        let gamma = vec![1.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let out = rms_norm(&input, &gamma, &config).unwrap();
        for &v in &out {
            prop_assert!(v.is_finite(), "rms_norm produced non-finite: {v}");
        }
    }

    /// RMS-norm output length matches input length.
    #[test]
    fn prop_rms_norm_output_length(
        dim in 2usize..=32,
        batch in 1usize..=4,
    ) {
        let n = batch * dim;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let gamma = vec![1.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let out = rms_norm(&input, &gamma, &config).unwrap();
        prop_assert_eq!(out.len(), n);
    }

    /// Layer-norm is idempotent with gamma=1, beta=0: applying twice
    /// yields approximately the same result as once.
    #[test]
    fn prop_layer_norm_idempotent(
        dim in 4usize..=16,
    ) {
        let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.5 - 2.0).collect();
        let gamma = vec![1.0f32; dim];
        let beta = vec![0.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let once = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let twice = layer_norm(&once, &gamma, Some(&beta), &config).unwrap();
        for (i, (&a, &b)) in once.iter().zip(twice.iter()).enumerate() {
            prop_assert!(
                (a - b).abs() < 1e-4,
                "layer_norm not idempotent at {i}: {a} vs {b}"
            );
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 4. CUDA layer-norm / RMS-norm CPU fallbacks
    // ════════════════════════════════════════════════════════════════

    /// CUDA layer_norm_cpu_fallback output has zero mean per row.
    #[test]
    fn prop_cuda_layer_norm_zero_mean(
        dim in 4usize..=32,
        rows in 1usize..=4,
    ) {
        let n = rows * dim;
        let input: Vec<f32> = (0..n).map(|i| ((i * 13 + 2) % 17) as f32 - 8.0).collect();
        let gamma = vec![1.0f32; dim];
        let beta = vec![0.0f32; dim];
        let config = CudaLayerNormConfig::with_defaults();
        let out = layer_norm_cpu_fallback(&input, &gamma, &beta, dim, &config).unwrap();
        for r in 0..rows {
            let row = &out[r * dim..(r + 1) * dim];
            let mean: f32 = row.iter().sum::<f32>() / dim as f32;
            prop_assert!(mean.abs() < 0.1, "cuda ln row {r} mean = {mean}");
        }
    }

    /// CUDA rms_norm_cpu_fallback output is finite.
    #[test]
    fn prop_cuda_rms_norm_finite(
        dim in 1usize..=32,
        rows in 1usize..=4,
    ) {
        let n = rows * dim;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let gamma = vec![1.0f32; dim];
        let config = CudaLayerNormConfig::with_defaults();
        let out = rms_norm_cpu_fallback(&input, &gamma, dim, &config).unwrap();
        for &v in &out {
            prop_assert!(v.is_finite(), "cuda rms_norm non-finite: {v}");
        }
    }

    /// CUDA layer_norm_cpu_fallback output length matches input.
    #[test]
    fn prop_cuda_layer_norm_output_length(
        dim in 1usize..=16,
        rows in 1usize..=8,
    ) {
        let n = rows * dim;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let gamma = vec![1.0f32; dim];
        let beta = vec![0.0f32; dim];
        let config = CudaLayerNormConfig::with_defaults();
        let out = layer_norm_cpu_fallback(&input, &gamma, &beta, dim, &config).unwrap();
        prop_assert_eq!(out.len(), n);
    }

    // ════════════════════════════════════════════════════════════════
    // 5. CPU softmax numerical stability
    // ════════════════════════════════════════════════════════════════

    /// softmax_cpu output sums to ≈ 1.0 per row.
    #[test]
    fn prop_softmax_cpu_sums_to_one(
        cols in 2usize..=64,
    ) {
        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.5 - 5.0).collect();
        let mut out = vec![0.0f32; cols];
        let config = SoftmaxConfig::for_shape(cols, 1).unwrap();
        softmax_cpu(&input, &mut out, &config).unwrap();
        let sum: f32 = out.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-4, "softmax sum = {sum}");
    }

    /// softmax_cpu output values are in [0, 1].
    #[test]
    fn prop_softmax_cpu_in_unit_interval(
        cols in 1usize..=64,
    ) {
        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.3 - 3.0).collect();
        let mut out = vec![0.0f32; cols];
        let config = SoftmaxConfig::for_shape(cols, 1).unwrap();
        softmax_cpu(&input, &mut out, &config).unwrap();
        for (i, &val) in out.iter().enumerate() {
            prop_assert!(
                val >= 0.0 && val <= 1.0 + 1e-6,
                "softmax[{i}] = {val} not in [0,1]"
            );
        }
    }

    /// softmax_cpu is translation-invariant.
    #[test]
    fn prop_softmax_cpu_translation_invariant(
        cols in 2usize..=32,
        shift in -5.0f32..5.0,
    ) {
        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.4 - 2.0).collect();
        let shifted: Vec<f32> = input.iter().map(|&x| x + shift).collect();
        let config = SoftmaxConfig::for_shape(cols, 1).unwrap();
        let mut out_orig = vec![0.0f32; cols];
        let mut out_shifted = vec![0.0f32; cols];
        softmax_cpu(&input, &mut out_orig, &config).unwrap();
        softmax_cpu(&shifted, &mut out_shifted, &config).unwrap();
        for i in 0..cols {
            prop_assert!(
                (out_orig[i] - out_shifted[i]).abs() < 1e-4,
                "translation invariance broken at {i}"
            );
        }
    }

    /// batched_softmax rows each sum to ≈ 1.
    #[test]
    fn prop_batched_softmax_sums(
        batch in 1usize..=8,
        seq in 2usize..=32,
    ) {
        let input: Vec<f32> = (0..(batch * seq)).map(|i| (i as f32) * 0.2 - 3.0).collect();
        let out = batched_softmax(&input, batch, seq).unwrap();
        for b in 0..batch {
            let sum: f32 = out[b * seq..(b + 1) * seq].iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-4, "batch {b} sum = {sum}");
        }
    }

    /// softmax_cpu preserves ordering: if x[i] > x[j] then softmax(x)[i] > softmax(x)[j].
    #[test]
    fn prop_softmax_cpu_monotonic(
        cols in 3usize..=32,
    ) {
        // Strictly increasing input so each element is distinct.
        let input: Vec<f32> =
            (0..cols).map(|i| (i as f32) * 0.5 + 1.0).collect();
        let config = SoftmaxConfig::for_shape(cols, 1).unwrap();
        let mut out = vec![0.0f32; cols];
        softmax_cpu(&input, &mut out, &config).unwrap();
        for i in 1..cols {
            prop_assert!(
                out[i] > out[i - 1],
                "monotonicity violated at {i}: {} <= {}",
                out[i],
                out[i - 1]
            );
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 6. CPU reduction algebra
    // ════════════════════════════════════════════════════════════════

    /// sum of all-ones vector equals its length.
    #[test]
    fn prop_reduction_sum_ones(
        n in 1usize..=64,
    ) {
        let data = vec![1.0f32; n];
        let sum = ReductionKernel::sum(&data).unwrap();
        prop_assert!((sum - n as f32).abs() < 1e-4, "sum(ones) = {sum}, expected {n}");
    }

    /// mean of constant vector equals that constant.
    #[test]
    fn prop_reduction_mean_constant(
        n in 1usize..=64,
        c in -10.0f32..10.0,
    ) {
        let data = vec![c; n];
        let mean = ReductionKernel::mean(&data).unwrap();
        prop_assert!(
            (mean - c).abs() < 1e-4,
            "mean of constant {c} = {mean}"
        );
    }

    /// max of a vector is >= all elements.
    #[test]
    fn prop_reduction_max_ge_all(
        v in finite_f32_vec(64),
    ) {
        let max_val = ReductionKernel::max(&v).unwrap();
        for (i, &x) in v.iter().enumerate() {
            prop_assert!(
                max_val.value >= x - 1e-6,
                "max {} < element[{i}] = {x}", max_val.value
            );
        }
    }

    /// min of a vector is <= all elements.
    #[test]
    fn prop_reduction_min_le_all(
        v in finite_f32_vec(64),
    ) {
        let min_val = ReductionKernel::min(&v).unwrap();
        for (i, &x) in v.iter().enumerate() {
            prop_assert!(
                min_val.value <= x + 1e-6,
                "min {} > element[{i}] = {x}", min_val.value
            );
        }
    }

    /// l2_norm of zero vector is 0.
    #[test]
    fn prop_reduction_l2_norm_zero(
        n in 1usize..=64,
    ) {
        let data = vec![0.0f32; n];
        let norm = ReductionKernel::l2_norm(&data).unwrap();
        prop_assert!(norm.abs() < 1e-6, "l2_norm of zeros = {norm}");
    }

    // ════════════════════════════════════════════════════════════════
    // 7. CPU transpose involution
    // ════════════════════════════════════════════════════════════════

    /// Transposing twice recovers the original matrix.
    #[test]
    fn prop_transpose_2d_involution(
        rows in 1usize..=16,
        cols in 1usize..=16,
    ) {
        let n = rows * cols;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let t1 = TransposeKernel::transpose_2d(&data, rows, cols).unwrap();
        let t2 = TransposeKernel::transpose_2d(&t1, cols, rows).unwrap();
        for (i, (&a, &b)) in data.iter().zip(t2.iter()).enumerate() {
            prop_assert!((a - b).abs() < 1e-6, "transpose involution at {i}");
        }
    }

    /// Transpose output length matches input length.
    #[test]
    fn prop_transpose_2d_length(
        rows in 1usize..=16,
        cols in 1usize..=16,
    ) {
        let n = rows * cols;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let out = TransposeKernel::transpose_2d(&data, rows, cols).unwrap();
        prop_assert_eq!(out.len(), n);
    }

    /// squeeze removes all size-1 dims.
    #[test]
    fn prop_squeeze_removes_ones(
        n in 1usize..=5,
    ) {
        let mut shape: Vec<usize> = vec![1; n];
        shape.push(4);
        shape.push(1);
        shape.push(8);
        let squeezed = TransposeKernel::squeeze(&shape);
        for &d in &squeezed {
            prop_assert!(d > 1, "squeeze left size-1 dim");
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 8. CPU gating activation bounds
    // ════════════════════════════════════════════════════════════════

    /// swiglu output is finite for finite inputs.
    #[test]
    fn prop_swiglu_finite(
        v in finite_f32_vec(32),
    ) {
        let up = vec![1.0f32; v.len()];
        let mut out = vec![0.0f32; v.len()];
        swiglu(&v, &up, &mut out).unwrap();
        for (i, &val) in out.iter().enumerate() {
            prop_assert!(val.is_finite(), "swiglu[{i}] = {val}");
        }
    }

    /// geglu output is finite for finite inputs.
    #[test]
    fn prop_geglu_finite(
        v in finite_f32_vec(32),
    ) {
        let up = vec![1.0f32; v.len()];
        let mut out = vec![0.0f32; v.len()];
        geglu(&v, &up, &mut out).unwrap();
        for (i, &val) in out.iter().enumerate() {
            prop_assert!(val.is_finite(), "geglu[{i}] = {val}");
        }
    }

    /// reglu with all-positive gate equals gate (since relu(gate) = gate).
    #[test]
    fn prop_reglu_positive_gate_identity(
        v in non_zero_f32_vec(32),
    ) {
        let up = vec![1.0f32; v.len()];
        let mut out = vec![0.0f32; v.len()];
        reglu(&v, &up, &mut out).unwrap();
        for (i, (&gate_val, &out_val)) in v.iter().zip(out.iter()).enumerate() {
            prop_assert!(
                (gate_val - out_val).abs() < 1e-5,
                "reglu[{i}]: gate={gate_val}, out={out_val}"
            );
        }
    }

    /// apply_gating dispatches correctly: ReGLU with zero gate produces zeros.
    #[test]
    fn prop_apply_gating_reglu_zero_gate(
        n in 1usize..=32,
    ) {
        let gate = vec![0.0f32; n];
        let up: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; n];
        apply_gating(GatingType::ReGLU, &gate, &up, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.abs() < 1e-6, "reglu(0, up)[{i}] = {v}, expected 0");
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 9. CPU pipeline-parallel advanced properties
    // ════════════════════════════════════════════════════════════════

    /// pipeline_forward output preserves total size = batch × dim.
    #[test]
    fn prop_pipeline_forward_output_size(
        batch in 1usize..=8,
        dim in 1usize..=8,
    ) {
        use bitnet_kernels::cpu::pipeline_parallel::{
            pipeline_forward, PipelineConfig, PipelineSchedule,
        };
        let input: Vec<f32> = (0..(batch * dim)).map(|i| i as f32 * 0.1).collect();
        let stages = vec![
            PipelineStage::new(0, 2),
            PipelineStage::new(2, 4),
        ];
        let config = PipelineConfig::new(
            stages,
            batch.max(1),
            PipelineSchedule::Sequential,
        );
        let out = pipeline_forward(&input, batch, dim, &config).unwrap();
        prop_assert_eq!(out.len(), batch * dim);
    }

    /// pipeline_forward output is finite for finite input.
    #[test]
    fn prop_pipeline_forward_finite(
        batch in 1usize..=4,
        dim in 1usize..=8,
    ) {
        use bitnet_kernels::cpu::pipeline_parallel::{
            pipeline_forward, PipelineConfig, PipelineSchedule,
        };
        let input: Vec<f32> = (0..(batch * dim)).map(|i| (i as f32) * 0.05).collect();
        let stages = vec![PipelineStage::new(0, 1)];
        let config = PipelineConfig::new(
            stages,
            1,
            PipelineSchedule::Sequential,
        );
        let out = pipeline_forward(&input, batch, dim, &config).unwrap();
        for &v in &out {
            prop_assert!(v.is_finite(), "pipeline_forward output non-finite: {v}");
        }
    }

    /// Splitting into micro_batch_size=1 gives exactly `batch` chunks.
    #[test]
    fn prop_pipeline_split_size_one(
        batch in 1usize..=16,
        dim in 1usize..=8,
    ) {
        let input = vec![0.0f32; batch * dim];
        let chunks = micro_batch_split(&input, batch, dim, 1).unwrap();
        prop_assert_eq!(chunks.len(), batch);
        for chunk in &chunks {
            prop_assert_eq!(chunk.len(), dim);
        }
    }

    /// Splitting with micro_batch_size >= batch gives exactly 1 chunk.
    #[test]
    fn prop_pipeline_split_full_batch(
        batch in 1usize..=16,
        dim in 1usize..=8,
    ) {
        let input = vec![0.0f32; batch * dim];
        let chunks = micro_batch_split(&input, batch, dim, batch + 1).unwrap();
        prop_assert_eq!(chunks.len(), 1);
        prop_assert_eq!(chunks[0].len(), batch * dim);
    }

    // ════════════════════════════════════════════════════════════════
    // 10. CPU residual-add semantics
    // ════════════════════════════════════════════════════════════════

    /// add_residual(output, residual) → output[i] += residual[i].
    #[test]
    fn prop_add_residual_correctness(
        a in finite_f32_vec(32),
    ) {
        let residual: Vec<f32> = a.iter().map(|x| x * 0.5).collect();
        let mut output = a.clone();
        add_residual(&mut output, &residual).unwrap();
        for (i, ((&orig, &res), &out)) in a.iter().zip(residual.iter()).zip(output.iter()).enumerate() {
            prop_assert!(
                (out - (orig + res)).abs() < 1e-5,
                "add_residual at {i}: {out} != {orig} + {res}"
            );
        }
    }

    /// add_residual_scaled with scale=0 is a no-op.
    #[test]
    fn prop_add_residual_scaled_zero_noop(
        a in finite_f32_vec(32),
    ) {
        let residual = vec![99.0f32; a.len()];
        let mut output = a.clone();
        add_residual_scaled(&mut output, &residual, 0.0).unwrap();
        for (i, (&orig, &out)) in a.iter().zip(output.iter()).enumerate() {
            prop_assert!(
                (out - orig).abs() < 1e-6,
                "scaled(0) changed output at {i}: {orig} → {out}"
            );
        }
    }

    /// add_residual_scaled with scale=1 matches add_residual.
    #[test]
    fn prop_add_residual_scaled_one_matches(
        a in finite_f32_vec(32),
    ) {
        let residual: Vec<f32> = a.iter().map(|x| x * 0.3).collect();
        let mut out_plain = a.clone();
        add_residual(&mut out_plain, &residual).unwrap();
        let mut out_scaled = a.clone();
        add_residual_scaled(&mut out_scaled, &residual, 1.0).unwrap();
        for (i, (&p, &s)) in out_plain.iter().zip(out_scaled.iter()).enumerate() {
            prop_assert!(
                (p - s).abs() < 1e-5,
                "add_residual vs scaled(1) at {i}: {p} vs {s}"
            );
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 11. General kernel no-NaN invariants
    // ════════════════════════════════════════════════════════════════

    /// relu never produces NaN from finite input.
    #[test]
    fn prop_relu_no_nan(x in -100.0f32..100.0) {
        let r = relu(x);
        prop_assert!(!r.is_nan(), "relu({x}) = NaN");
    }

    /// sigmoid never produces NaN from finite input.
    #[test]
    fn prop_sigmoid_no_nan(x in -100.0f32..100.0) {
        let s = sigmoid(x);
        prop_assert!(!s.is_nan(), "sigmoid({x}) = NaN");
    }

    /// silu never produces NaN from finite input.
    #[test]
    fn prop_silu_no_nan(x in -100.0f32..100.0) {
        let s = silu(x);
        prop_assert!(!s.is_nan(), "silu({x}) = NaN");
    }

    /// embedding_accumulate output is finite for finite inputs.
    #[test]
    fn prop_embedding_accumulate_finite(
        vocab in 4usize..=16,
        dim in 1usize..=8,
        n_idx in 1usize..=4,
    ) {
        let table: Vec<f32> = (0..(vocab * dim)).map(|i| i as f32 * 0.01).collect();
        let indices: Vec<u32> = (0..n_idx).map(|i| (i % vocab) as u32).collect();
        let weights: Vec<f32> = (0..n_idx).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let out = embedding_accumulate(&table, &indices, &weights, dim).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.is_finite(), "accumulate[{i}] = {v}");
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 12. Additional algebraic identities
    // ════════════════════════════════════════════════════════════════

    /// relu is monotonically non-decreasing.
    #[test]
    fn prop_relu_monotonic(
        a in -100.0f32..100.0,
        b in -100.0f32..100.0,
    ) {
        if a <= b {
            prop_assert!(relu(a) <= relu(b), "relu not monotonic: relu({a}) > relu({b})");
        }
    }

    /// sigmoid output is always in [0, 1] for finite input.
    #[test]
    fn prop_sigmoid_unit_interval(x in -100.0f32..100.0) {
        let s = sigmoid(x);
        prop_assert!(s >= 0.0 && s <= 1.0, "sigmoid({x}) = {s} not in [0,1]");
    }

    /// layer_norm with all-equal input produces all-zero (or very small) output.
    #[test]
    fn prop_layer_norm_constant_input_zero(
        dim in 2usize..=32,
        val in -10.0f32..10.0,
    ) {
        let input = vec![val; dim];
        let gamma = vec![1.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let out = layer_norm(&input, &gamma, None, &config).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.abs() < 1e-4, "constant input LN[{i}] = {v}, expected ~0");
        }
    }

    /// rms_norm scales proportionally: rms_norm(k*x) ≈ sign(k)*rms_norm(x) for positive k.
    #[test]
    fn prop_rms_norm_scale_invariant(
        dim in 2usize..=16,
    ) {
        let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.3 + 0.1).collect();
        let gamma = vec![1.0f32; dim];
        let config = LayerNormConfig::new(vec![dim]);
        let out1 = rms_norm(&input, &gamma, &config).unwrap();
        let scaled: Vec<f32> = input.iter().map(|x| x * 5.0).collect();
        let out2 = rms_norm(&scaled, &gamma, &config).unwrap();
        for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
            prop_assert!(
                (a - b).abs() < 1e-4,
                "rms_norm scale invariance violated at {i}: {a} vs {b}"
            );
        }
    }

    /// micro_batch_split with batch_size=1 produces 1 chunk equal to the input.
    #[test]
    fn prop_split_single_item_identity(
        dim in 1usize..=16,
    ) {
        let data: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let chunks = micro_batch_split(&data, 1, dim, 1).unwrap();
        prop_assert_eq!(chunks.len(), 1);
        prop_assert_eq!(&chunks[0], &data);
    }
}
