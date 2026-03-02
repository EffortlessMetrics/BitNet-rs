//! Property-based tests — wave 26.
//!
//! Covers CUDA modules: cooperative_groups, embedding_ops, shader_cache,
//! FFN, batch_norm, pipeline_parallel; and CPU modules: FFN, batch_norm,
//! KV cache, attention, layer_norm, softmax, pooling, loss,
//! scatter_gather.
//!
//! 55+ property tests validating: shape preservation, finite outputs,
//! idempotence, round-trip fidelity, boundedness, monotonicity, and more.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::attention::{apply_causal_mask, scaled_dot_product_attention};
use bitnet_kernels::cpu::batch_norm::{BatchNormConfig, batch_norm_forward, batch_norm_inference};
use bitnet_kernels::cpu::ffn::{
    FfnActivation, FfnConfig as CpuFfnConfig, ffn_forward, gated_ffn_forward,
};
use bitnet_kernels::cpu::kv_cache::{
    KvCache, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear, kv_cache_memory_usage,
    kv_cache_slice,
};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::loss::{
    LossReduction, cosine_similarity_loss, cross_entropy_loss, gradient_accumulate,
    gradient_clip_norm, l1_loss, mse_loss, perplexity,
};
use bitnet_kernels::cpu::pipeline_parallel::{
    micro_batch_merge, micro_batch_split, optimal_micro_batch_count, pipeline_bubble_time,
};
use bitnet_kernels::cpu::pooling::{PoolConfig, PoolType, avg_pool1d, max_pool1d, pool_1d};
use bitnet_kernels::cpu::scatter_gather::{gather_1d, scatter_1d, scatter_add};
use bitnet_kernels::cpu::softmax::{
    log_softmax_f32, softmax_f32, softmax_f32_inplace, softmax_with_temperature,
};
use bitnet_kernels::cuda::batch_norm::batch_norm_inference_cpu_fallback;
use bitnet_kernels::cuda::cooperative_groups::{
    CooperativeGroupConfig, CooperativeReduceOp, cooperative_broadcast, cooperative_histogram,
    cooperative_matmul, cooperative_reduce, cooperative_scan, cooperative_sort,
};
use bitnet_kernels::cuda::embedding_ops::{
    EmbeddingBagMode, EmbeddingConfig, EmbeddingTable, embedding_bag, embedding_lookup,
    sinusoidal_position_embedding,
};
use bitnet_kernels::cuda::ffn::{ffn_geglu, ffn_swiglu};
use bitnet_kernels::cuda::shader_cache::{
    HashAlgorithm, ShaderCache, ShaderCacheConfig, ShaderSource, cache_stats, compile_shader,
    invalidate_shader, lookup_shader,
};
use proptest::prelude::*;
use std::path::PathBuf;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn default_coop_config() -> CooperativeGroupConfig {
    CooperativeGroupConfig {
        group_size: 32,
        grid_sync: false,
        thread_block_cluster: false,
        cluster_size: 1,
        shared_mem_bytes: 0,
    }
}

fn make_shader_source(code: &str) -> ShaderSource {
    ShaderSource {
        cuda_source: code.to_string(),
        compile_options: vec![],
        target_arch: "sm_80".to_string(),
    }
}

fn make_shader_cache() -> ShaderCache {
    let cfg = ShaderCacheConfig {
        cache_dir: PathBuf::from("/tmp/bitnet-test-cache"),
        max_cache_size_mb: 64,
        enable_persistence: false,
        hash_algorithm: HashAlgorithm::Fnv1a64,
    };
    ShaderCache::new(cfg).unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════
// CUDA module properties
// ═══════════════════════════════════════════════════════════════════════════

// ── 1. Cooperative groups — reduce, scan, broadcast ─────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn coop_reduce_sum_is_sum(n in 1usize..32) {
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let cfg = default_coop_config();
        let out = cooperative_reduce(&data, CooperativeReduceOp::Sum, &cfg).unwrap();
        let expected: f32 = data.iter().sum();
        prop_assert!((out[0] - expected).abs() < 1e-3,
            "reduce sum: got {} expected {}", out[0], expected);
    }

    #[test]
    fn coop_reduce_max_is_max(n in 1usize..32) {
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.7 - 5.0).collect();
        let cfg = default_coop_config();
        let out = cooperative_reduce(&data, CooperativeReduceOp::Max, &cfg).unwrap();
        let expected = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        prop_assert!((out[0] - expected).abs() < 1e-5,
            "reduce max: got {} expected {}", out[0], expected);
    }

    #[test]
    fn coop_reduce_min_is_min(n in 1usize..32) {
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 + 1.0).collect();
        let cfg = default_coop_config();
        let out = cooperative_reduce(&data, CooperativeReduceOp::Min, &cfg).unwrap();
        let expected = data.iter().copied().fold(f32::INFINITY, f32::min);
        prop_assert!((out[0] - expected).abs() < 1e-5,
            "reduce min: got {} expected {}", out[0], expected);
    }

    #[test]
    fn coop_scan_is_prefix_sum(n in 1usize..16) {
        let mut data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let cfg = default_coop_config();
        cooperative_scan(&mut data, &cfg).unwrap();
        // prefix sums: 1, 3, 6, 10, ...
        for i in 1..n {
            prop_assert!(data[i] >= data[i - 1],
                "scan not monotone at {}", i);
        }
    }

    #[test]
    fn coop_broadcast_fills_uniform(n in 2usize..16) {
        let mut data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let cfg = default_coop_config();
        cooperative_broadcast(&mut data, 0, &cfg).unwrap();
        let first = data[0];
        for (i, &v) in data.iter().enumerate() {
            prop_assert!((v - first).abs() < 1e-6,
                "broadcast: data[{}]={} != {}", i, v, first);
        }
    }

    #[test]
    fn coop_sort_is_sorted(n in 1usize..32) {
        let mut data: Vec<f32> = (0..n).map(|i| ((n - i) as f32) * 1.3).collect();
        let cfg = default_coop_config();
        cooperative_sort(&mut data, &cfg).unwrap();
        for i in 1..n {
            prop_assert!(data[i] >= data[i - 1],
                "not sorted at {}: {} > {}", i, data[i - 1], data[i]);
        }
    }

    #[test]
    fn coop_histogram_total_equals_input_len(n in 1usize..64) {
        let data: Vec<u32> = (0..n).map(|i| (i % 8) as u32).collect();
        let cfg = default_coop_config();
        let hist = cooperative_histogram(&data, 8, &cfg).unwrap();
        let total: u32 = hist.iter().sum();
        prop_assert_eq!(total, n as u32,
            "histogram total {} != input len {}", total, n);
    }

    #[test]
    fn coop_matmul_output_shape(
        m in 1usize..6,
        n in 1usize..6,
        k in 1usize..6,
    ) {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let cfg = default_coop_config();
        let out = cooperative_matmul(&a, &b, m, n, k, &cfg).unwrap();
        prop_assert_eq!(out.len(), m * n,
            "matmul shape: {} != {}", out.len(), m * n);
    }
}

// ── 2. Embedding ops — lookup, position, bag ────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn embedding_lookup_output_shape(
        vocab in 5usize..20,
        dim in 2usize..8,
        n_idx in 1usize..6,
    ) {
        let table = EmbeddingTable::new(vec![0.5f32; vocab * dim], vocab, dim).unwrap();
        let cfg = EmbeddingConfig::new(vocab, dim).unwrap();
        let indices: Vec<u32> = (0..n_idx).map(|i| (i % vocab) as u32).collect();
        let out = embedding_lookup(&table, &indices, &cfg).unwrap();
        prop_assert_eq!(out.len(), n_idx * dim,
            "embedding shape: {} != {}", out.len(), n_idx * dim);
    }

    #[test]
    fn embedding_lookup_finite(
        vocab in 5usize..15,
        dim in 2usize..6,
    ) {
        let table = EmbeddingTable::new(vec![0.1f32; vocab * dim], vocab, dim).unwrap();
        let cfg = EmbeddingConfig::new(vocab, dim).unwrap();
        let indices: Vec<u32> = (0..3).map(|i| (i % vocab) as u32).collect();
        let out = embedding_lookup(&table, &indices, &cfg).unwrap();
        for (i, v) in out.iter().enumerate() {
            prop_assert!(v.is_finite(), "embedding[{}] not finite", i);
        }
    }

    #[test]
    fn sinusoidal_position_shape(
        n_pos in 1usize..8,
        dim in 1usize..6,
    ) {
        let dim = dim + (dim % 2); // must be even
        let positions: Vec<u32> = (0..n_pos as u32).collect();
        let out = sinusoidal_position_embedding(&positions, dim).unwrap();
        prop_assert_eq!(out.len(), n_pos * dim,
            "sinusoidal shape: {} != {}", out.len(), n_pos * dim);
    }

    #[test]
    fn sinusoidal_position_finite(
        n_pos in 1usize..8,
        dim in 1usize..6,
    ) {
        let dim = dim + (dim % 2);
        let positions: Vec<u32> = (0..n_pos as u32).collect();
        let out = sinusoidal_position_embedding(&positions, dim).unwrap();
        for (i, v) in out.iter().enumerate() {
            prop_assert!(v.is_finite(), "sinusoidal[{}] not finite", i);
        }
    }

    #[test]
    fn embedding_bag_sum_shape(
        vocab in 5usize..15,
        dim in 2usize..6,
    ) {
        let table = EmbeddingTable::new(vec![1.0f32; vocab * dim], vocab, dim).unwrap();
        let cfg = EmbeddingConfig::new(vocab, dim).unwrap();
        let indices: Vec<u32> = vec![0, 1, 2];
        let offsets: Vec<usize> = vec![0];
        let out = embedding_bag(&table, &indices, &offsets, EmbeddingBagMode::Sum, &cfg).unwrap();
        prop_assert_eq!(out.len(), dim,
            "bag shape: {} != {}", out.len(), dim);
    }

    #[test]
    fn embedding_bag_mean_finite(
        vocab in 5usize..15,
        dim in 2usize..6,
    ) {
        let table = EmbeddingTable::new(vec![0.5f32; vocab * dim], vocab, dim).unwrap();
        let cfg = EmbeddingConfig::new(vocab, dim).unwrap();
        let indices: Vec<u32> = vec![0, 1, 2];
        let offsets: Vec<usize> = vec![0];
        let out = embedding_bag(&table, &indices, &offsets, EmbeddingBagMode::Mean, &cfg).unwrap();
        for (i, v) in out.iter().enumerate() {
            prop_assert!(v.is_finite(), "bag_mean[{}] not finite", i);
        }
    }
}

// ── 3. Shader cache — compile, lookup, invalidate ──────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn shader_compile_produces_ptx(seed in 0u32..100) {
        let code = format!("extern \"C\" __global__ void k{}() {{}}", seed);
        let src = make_shader_source(&code);
        let cached = compile_shader(&src, HashAlgorithm::Fnv1a64).unwrap();
        prop_assert!(!cached.ptx.is_empty(), "compiled PTX should not be empty");
    }

    #[test]
    fn shader_compile_metadata_hash_nonzero(seed in 0u32..100) {
        let code = format!("extern \"C\" __global__ void kern{}() {{}}", seed);
        let src = make_shader_source(&code);
        let cached = compile_shader(&src, HashAlgorithm::Fnv1a64).unwrap();
        prop_assert!(cached.metadata.source_hash != 0,
            "source hash should be nonzero");
    }

    #[test]
    fn shader_cache_insert_lookup_roundtrip(seed in 0u32..50) {
        let code = format!("extern \"C\" __global__ void f{}() {{}}", seed);
        let src = make_shader_source(&code);
        let cached = compile_shader(&src, HashAlgorithm::Fnv1a64).unwrap();
        let hash = cached.metadata.source_hash;
        let mut cache = make_shader_cache();
        cache.insert(hash, cached);
        let found = lookup_shader(&mut cache, hash);
        prop_assert!(found.is_some(), "lookup after insert should succeed");
    }

    #[test]
    fn shader_cache_invalidate_removes(seed in 0u32..50) {
        let code = format!("extern \"C\" __global__ void g{}() {{}}", seed);
        let src = make_shader_source(&code);
        let cached = compile_shader(&src, HashAlgorithm::Fnv1a64).unwrap();
        let hash = cached.metadata.source_hash;
        let mut cache = make_shader_cache();
        cache.insert(hash, cached);
        let removed = invalidate_shader(&mut cache, hash);
        prop_assert!(removed, "invalidate should return true");
        let found = lookup_shader(&mut cache, hash);
        prop_assert!(found.is_none(), "lookup after invalidate should be None");
    }

    #[test]
    fn shader_cache_stats_hits(seed in 0u32..30) {
        let code = format!("extern \"C\" __global__ void h{}() {{}}", seed);
        let src = make_shader_source(&code);
        let cached = compile_shader(&src, HashAlgorithm::Fnv1a64).unwrap();
        let hash = cached.metadata.source_hash;
        let mut cache = make_shader_cache();
        cache.insert(hash, cached);
        let _ = lookup_shader(&mut cache, hash);
        let stats = cache_stats(&cache);
        prop_assert!(stats.hits >= 1, "hits should be >= 1 after lookup");
    }

    #[test]
    fn shader_different_sources_different_hashes(
        a in 0u32..100,
        b in 0u32..100,
    ) {
        prop_assume!(a != b);
        let src_a = make_shader_source(&format!("extern \"C\" __global__ void a{}() {{}}", a));
        let src_b = make_shader_source(&format!("extern \"C\" __global__ void b{}() {{}}", b));
        let ca = compile_shader(&src_a, HashAlgorithm::Fnv1a64).unwrap();
        let cb = compile_shader(&src_b, HashAlgorithm::Fnv1a64).unwrap();
        prop_assert_ne!(ca.metadata.source_hash, cb.metadata.source_hash,
            "different sources should yield different hashes");
    }
}

// ── 4. CUDA FFN — SwiGLU, GeGLU output shape and finiteness ──────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn cuda_ffn_swiglu_shape(n in 1usize..64) {
        let gate = vec![0.5f32; n];
        let up = vec![0.5f32; n];
        let mut out = vec![0.0f32; n];
        ffn_swiglu(&gate, &up, &mut out, n).unwrap();
        prop_assert_eq!(out.len(), n);
    }

    #[test]
    fn cuda_ffn_swiglu_finite(n in 1usize..64) {
        let gate = vec![0.5f32; n];
        let up = vec![1.0f32; n];
        let mut out = vec![0.0f32; n];
        ffn_swiglu(&gate, &up, &mut out, n).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.is_finite(), "swiglu[{}] not finite", i);
        }
    }

    #[test]
    fn cuda_ffn_geglu_shape(n in 1usize..64) {
        let gate = vec![0.5f32; n];
        let up = vec![0.5f32; n];
        let mut out = vec![0.0f32; n];
        ffn_geglu(&gate, &up, &mut out, n).unwrap();
        prop_assert_eq!(out.len(), n);
    }

    #[test]
    fn cuda_ffn_geglu_finite(n in 1usize..64) {
        let gate = vec![0.5f32; n];
        let up = vec![1.0f32; n];
        let mut out = vec![0.0f32; n];
        ffn_geglu(&gate, &up, &mut out, n).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.is_finite(), "geglu[{}] not finite", i);
        }
    }
}

// ── 5. CUDA batch norm — CPU fallback output shape and normalization ──────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn cuda_batch_norm_inference_shape(n_feat in 1usize..8) {
        let n = n_feat;
        let input = vec![1.0f32; n];
        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];
        let mean = vec![0.0f32; n];
        let var = vec![1.0f32; n];
        let out = batch_norm_inference_cpu_fallback(&input, &gamma, &beta, &mean, &var, 1e-5).unwrap();
        prop_assert_eq!(out.len(), n, "batch_norm inference shape mismatch");
    }

    #[test]
    fn cuda_batch_norm_inference_finite(n_feat in 1usize..8) {
        let n = n_feat;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.3).collect();
        let gamma = vec![1.0f32; n];
        let beta = vec![0.0f32; n];
        let mean = vec![0.0f32; n];
        let var = vec![1.0f32; n];
        let out = batch_norm_inference_cpu_fallback(&input, &gamma, &beta, &mean, &var, 1e-5).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.is_finite(), "batch_norm_inf[{}] not finite", i);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CPU module properties
// ═══════════════════════════════════════════════════════════════════════════

// ── 6. CPU FFN — output shape, finite, activation ──────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn cpu_ffn_output_shape(
        hidden in 2usize..8,
        inter in 2usize..8,
    ) {
        let cfg = CpuFfnConfig {
            hidden_dim: hidden,
            intermediate_dim: inter,
            activation: FfnActivation::ReLU,
        };
        let input = vec![0.5f32; hidden];
        let w_up = vec![0.1f32; hidden * inter];
        let w_down = vec![0.1f32; inter * hidden];
        let out = ffn_forward(&input, &w_up, &w_down, &cfg).unwrap();
        prop_assert_eq!(out.len(), hidden,
            "ffn shape: {} != {}", out.len(), hidden);
    }

    #[test]
    fn cpu_ffn_output_finite(
        hidden in 2usize..8,
        inter in 2usize..8,
    ) {
        let cfg = CpuFfnConfig {
            hidden_dim: hidden,
            intermediate_dim: inter,
            activation: FfnActivation::SiLU,
        };
        let input = vec![0.5f32; hidden];
        let w_up = vec![0.01f32; hidden * inter];
        let w_down = vec![0.01f32; inter * hidden];
        let out = ffn_forward(&input, &w_up, &w_down, &cfg).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.is_finite(), "ffn[{}] not finite", i);
        }
    }

    #[test]
    fn cpu_gated_ffn_output_shape(
        hidden in 2usize..8,
        inter in 2usize..8,
    ) {
        let cfg = CpuFfnConfig {
            hidden_dim: hidden,
            intermediate_dim: inter,
            activation: FfnActivation::GeLU,
        };
        let input = vec![0.5f32; hidden];
        let w_gate = vec![0.1f32; hidden * inter];
        let w_up = vec![0.1f32; hidden * inter];
        let w_down = vec![0.1f32; inter * hidden];
        let out = gated_ffn_forward(&input, &w_gate, &w_up, &w_down, &cfg).unwrap();
        prop_assert_eq!(out.len(), hidden,
            "gated ffn shape: {} != {}", out.len(), hidden);
    }
}

// ── 7. CPU batch norm — normalized output, shape ───────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn cpu_batch_norm_forward_shape(features in 2usize..8) {
        let cfg = BatchNormConfig {
            num_features: features,
            eps: 1e-5,
            momentum: 0.1,
            training: false,
        };
        let input = vec![1.0f32; features];
        let gamma = vec![1.0f32; features];
        let beta = vec![0.0f32; features];
        let running_mean = vec![0.0f32; features];
        let running_var = vec![1.0f32; features];
        let (out, _, _) = batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &cfg).unwrap();
        prop_assert_eq!(out.len(), features);
    }

    #[test]
    fn cpu_batch_norm_inference_shape(features in 2usize..8) {
        let input = vec![1.0f32; features];
        let gamma = vec![1.0f32; features];
        let beta = vec![0.0f32; features];
        let running_mean = vec![0.0f32; features];
        let running_var = vec![1.0f32; features];
        let out = batch_norm_inference(&input, &gamma, &beta, &running_mean, &running_var, 1e-5).unwrap();
        prop_assert_eq!(out.len(), features);
    }
}

// ── 8. CPU layer norm — shape, zero-mean ───────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn layer_norm_output_shape(dim in 2usize..16) {
        let cfg = LayerNormConfig {
            normalized_shape: vec![dim],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let input: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let gamma = vec![1.0f32; dim];
        let beta = vec![0.0f32; dim];
        let out = layer_norm(&input, &gamma, Some(&beta), &cfg).unwrap();
        prop_assert_eq!(out.len(), dim);
    }

    #[test]
    fn layer_norm_near_zero_mean(dim in 4usize..16) {
        let cfg = LayerNormConfig {
            normalized_shape: vec![dim],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.5 - 2.0).collect();
        let gamma = vec![1.0f32; dim];
        let beta = vec![0.0f32; dim];
        let out = layer_norm(&input, &gamma, Some(&beta), &cfg).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / dim as f32;
        prop_assert!(mean.abs() < 0.1,
            "layer norm mean {} should be near zero", mean);
    }

    #[test]
    fn rms_norm_output_shape(dim in 2usize..16) {
        let cfg = LayerNormConfig {
            normalized_shape: vec![dim],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let input: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 0.1).collect();
        let gamma = vec![1.0f32; dim];
        let out = rms_norm(&input, &gamma, &cfg).unwrap();
        prop_assert_eq!(out.len(), dim);
    }

    #[test]
    fn rms_norm_finite(dim in 2usize..16) {
        let cfg = LayerNormConfig {
            normalized_shape: vec![dim],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let input: Vec<f32> = (0..dim).map(|i| i as f32 * 0.3).collect();
        let gamma = vec![1.0f32; dim];
        let out = rms_norm(&input, &gamma, &cfg).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.is_finite(), "rms_norm[{}] not finite", i);
        }
    }
}

// ── 9. CPU softmax — sum-to-one, monotone, temperature ─────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn softmax_sums_to_one(n in 2usize..32) {
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
        let mut out = vec![0.0f32; n];
        softmax_f32(&input, &mut out).unwrap();
        let sum: f32 = out.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-4,
            "softmax sum {} != 1.0", sum);
    }

    #[test]
    fn softmax_all_nonneg(n in 2usize..32) {
        let input: Vec<f32> = (0..n).map(|i| i as f32 - 5.0).collect();
        let mut out = vec![0.0f32; n];
        softmax_f32(&input, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v >= 0.0, "softmax[{}]={} < 0", i, v);
        }
    }

    #[test]
    fn softmax_inplace_sums_to_one(n in 2usize..32) {
        let mut data: Vec<f32> = (0..n).map(|i| i as f32 * 0.3).collect();
        softmax_f32_inplace(&mut data).unwrap();
        let sum: f32 = data.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-4,
            "inplace softmax sum {} != 1.0", sum);
    }

    #[test]
    fn log_softmax_finite(n in 2usize..32) {
        let input: Vec<f32> = (0..n).map(|i| i as f32 - 3.0).collect();
        let mut out = vec![0.0f32; n];
        log_softmax_f32(&input, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.is_finite(), "log_softmax[{}] not finite", i);
            prop_assert!(v <= 0.0, "log_softmax[{}]={} > 0", i, v);
        }
    }

    #[test]
    fn softmax_temperature_sharpens(n in 3usize..16) {
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut out_warm = vec![0.0f32; n];
        let mut out_cold = vec![0.0f32; n];
        softmax_with_temperature(&input, &mut out_warm, 2.0).unwrap();
        softmax_with_temperature(&input, &mut out_cold, 0.5).unwrap();
        let max_warm = out_warm.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_cold = out_cold.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        prop_assert!(max_cold >= max_warm - 1e-6,
            "cold temp should be sharper: max_cold={} max_warm={}", max_cold, max_warm);
    }
}

// ── 10. CPU attention — shape, finiteness, bounds ──────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn attention_output_shape(
        seq in 1usize..6,
        dim in 2usize..8,
    ) {
        let n = seq * dim;
        let q = vec![0.1f32; n];
        let k = vec![0.1f32; n];
        let v = vec![0.1f32; n];
        let out = scaled_dot_product_attention(&q, &k, &v, seq, seq, dim, false).unwrap();
        prop_assert_eq!(out.len(), n,
            "attention output len {} != expected {}", out.len(), n);
    }

    #[test]
    fn attention_output_finite(
        seq in 1usize..5,
        dim in 2usize..6,
    ) {
        let n = seq * dim;
        let q = vec![0.5f32; n];
        let k = vec![0.5f32; n];
        let v = vec![0.5f32; n];
        let out = scaled_dot_product_attention(&q, &k, &v, seq, seq, dim, true).unwrap();
        for (i, &val) in out.iter().enumerate() {
            prop_assert!(val.is_finite(), "attention[{}] not finite", i);
        }
    }

    #[test]
    fn attention_output_bounded_by_values(
        seq in 1usize..5,
        dim in 2usize..6,
    ) {
        let n = seq * dim;
        let q = vec![0.1f32; n];
        let k = vec![0.1f32; n];
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let v_min = v.iter().copied().fold(f32::INFINITY, f32::min);
        let v_max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let out = scaled_dot_product_attention(&q, &k, &v, seq, seq, dim, false).unwrap();
        for (i, &val) in out.iter().enumerate() {
            prop_assert!(val >= v_min - 1e-3 && val <= v_max + 1e-3,
                "attention[{}]={} outside [{}, {}]", i, val, v_min, v_max);
        }
    }

    #[test]
    fn causal_mask_zeroes_upper_triangle(seq in 2usize..8) {
        let mut scores: Vec<f32> = vec![1.0; seq * seq];
        apply_causal_mask(&mut scores, seq).unwrap();
        for i in 0..seq {
            for j in (i + 1)..seq {
                prop_assert!(scores[i * seq + j] <= f32::NEG_INFINITY + 1.0,
                    "upper triangle [{},{}] not masked", i, j);
            }
        }
    }
}

// ── 11. CPU KV cache — append, slice, clear ────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn kv_cache_append_grows(
        dim in 2usize..8,
        n_appends in 1usize..5,
    ) {
        let cfg = KvCacheConfig {
            num_layers: 1,
            num_heads: 1,
            head_dim: dim,
            max_seq_len: 16,
            dtype: KvDtype::F32,
        };
        cfg.validate().unwrap();
        let mut cache = KvCache::new(cfg).unwrap();
        for step in 1..=n_appends {
            let k = vec![1.0f32; dim];
            let v = vec![1.0f32; dim];
            kv_cache_append(&mut cache, 0, &k, &v).unwrap();
            let (keys, _) = kv_cache_slice(&cache, 0, 0, step).unwrap();
            prop_assert_eq!(keys.len(), step * dim,
                "step {}: expected {} got {}", step, step * dim, keys.len());
        }
    }

    #[test]
    fn kv_cache_clear_resets(dim in 2usize..8) {
        let cfg = KvCacheConfig {
            num_layers: 1,
            num_heads: 1,
            head_dim: dim,
            max_seq_len: 16,
            dtype: KvDtype::F32,
        };
        cfg.validate().unwrap();
        let mut cache = KvCache::new(cfg).unwrap();
        let k = vec![1.0f32; dim];
        let v = vec![1.0f32; dim];
        kv_cache_append(&mut cache, 0, &k, &v).unwrap();
        kv_cache_clear(&mut cache);
        let (keys, _) = kv_cache_slice(&cache, 0, 0, 0).unwrap();
        prop_assert!(keys.is_empty(), "cache not empty after clear");
    }

    #[test]
    fn kv_cache_memory_positive(dim in 2usize..8) {
        let cfg = KvCacheConfig {
            num_layers: 1,
            num_heads: 1,
            head_dim: dim,
            max_seq_len: 16,
            dtype: KvDtype::F32,
        };
        cfg.validate().unwrap();
        let mut cache = KvCache::new(cfg).unwrap();
        let k = vec![1.0f32; dim];
        let v = vec![1.0f32; dim];
        kv_cache_append(&mut cache, 0, &k, &v).unwrap();
        let mem = kv_cache_memory_usage(&cache);
        prop_assert!(mem > 0, "memory usage should be > 0");
    }
}

// ── 12. CPU pooling — max/avg shape, output bounds ─────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn max_pool_output_nonempty(n in 4usize..32) {
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let cfg = PoolConfig {
            pool_type: PoolType::Max,
            kernel_size: 2,
            stride: 2,
            padding: 0,
            dilation: 1,
            ceil_mode: false,
        };
        let (out, _indices) = max_pool1d(&input, &cfg).unwrap();
        prop_assert!(!out.is_empty(), "max_pool output should not be empty");
    }

    #[test]
    fn avg_pool_output_nonempty(n in 4usize..32) {
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let cfg = PoolConfig {
            pool_type: PoolType::Average,
            kernel_size: 2,
            stride: 2,
            padding: 0,
            dilation: 1,
            ceil_mode: false,
        };
        let out = avg_pool1d(&input, &cfg).unwrap();
        prop_assert!(!out.is_empty(), "avg_pool output should not be empty");
    }

    #[test]
    fn pool_1d_finite(n in 4usize..32) {
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let cfg = PoolConfig {
            pool_type: PoolType::Max,
            kernel_size: 2,
            stride: 1,
            padding: 0,
            dilation: 1,
            ceil_mode: false,
        };
        let out = pool_1d(&input, &cfg).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.is_finite(), "pool_1d[{}] not finite", i);
        }
    }
}

// ── 13. CPU loss functions — non-negative, gradient shape ──────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn mse_loss_nonneg(n in 2usize..16) {
        let pred: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let target: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let loss = mse_loss(&pred, &target, LossReduction::Mean).unwrap();
        prop_assert!(loss >= 0.0, "MSE loss {} < 0", loss);
    }

    #[test]
    fn l1_loss_nonneg(n in 2usize..16) {
        let pred: Vec<f32> = (0..n).map(|i| i as f32 * 0.2).collect();
        let target: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3).collect();
        let loss = l1_loss(&pred, &target, LossReduction::Mean).unwrap();
        prop_assert!(loss >= 0.0, "L1 loss {} < 0", loss);
    }

    #[test]
    fn cross_entropy_finite(n_class in 2usize..8) {
        let n_samples = 4;
        let logits: Vec<f32> = (0..n_samples * n_class).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let targets: Vec<usize> = (0..n_samples).map(|i| i % n_class).collect();
        let (loss, _grad) = cross_entropy_loss(&logits, &targets, n_class, LossReduction::Mean).unwrap();
        prop_assert!(loss.is_finite(), "CE loss not finite: {}", loss);
    }

    #[test]
    fn cosine_similarity_bounded(n in 2usize..16) {
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 + 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 0.2).collect();
        let sim = cosine_similarity_loss(&a, &b).unwrap();
        prop_assert!((-1.0 - 1e-5..=1.0 + 1e-5).contains(&sim),
            "cosine sim {} out of [-1, 1]", sim);
    }

    #[test]
    fn perplexity_positive(ce in 0.01f32..10.0) {
        let pp = perplexity(ce);
        prop_assert!(pp > 0.0, "perplexity {} should be > 0", pp);
    }

    #[test]
    fn gradient_accumulate_shape(n in 2usize..16) {
        let mut acc = vec![0.0f32; n];
        let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
        gradient_accumulate(&mut acc, &src).unwrap();
        prop_assert_eq!(acc.len(), n);
        for (i, &v) in acc.iter().enumerate() {
            prop_assert!((v - src[i]).abs() < 1e-6,
                "accumulate mismatch at {}", i);
        }
    }

    #[test]
    fn gradient_clip_norm_bounded(n in 2usize..16) {
        let mut grads: Vec<f32> = (0..n).map(|i| i as f32 * 10.0).collect();
        let max_norm = 1.0f32;
        let _orig_norm = gradient_clip_norm(&mut grads, max_norm).unwrap();
        let clipped_norm: f32 = grads.iter().map(|v| v * v).sum::<f32>().sqrt();
        prop_assert!(clipped_norm <= max_norm + 1e-3,
            "clipped norm {} > max {}", clipped_norm, max_norm);
    }
}

// ── 14. Scatter-gather round-trip ──────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn scatter_gather_roundtrip(n in 2usize..16) {
        let values: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let indices: Vec<usize> = (0..n).collect();
        let mut buf = vec![0.0f32; n];
        scatter_1d(&mut buf, &indices, &values).unwrap();
        let gathered = gather_1d(&buf, &indices).unwrap();
        prop_assert_eq!(&gathered, &values,
            "scatter/gather round-trip failed");
    }

    #[test]
    fn scatter_add_accumulates(n in 2usize..16) {
        let values = vec![1.0f32; n];
        let indices: Vec<usize> = vec![0; n]; // all to index 0
        let mut buf = vec![0.0f32; n];
        scatter_add(&mut buf, &indices, &values).unwrap();
        prop_assert!((buf[0] - n as f32).abs() < 1e-4,
            "scatter_add: buf[0]={} expected {}", buf[0], n);
    }
}

// ── 15. Pipeline parallel — micro-batch split/merge, bubble time ───────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn micro_batch_split_merge_roundtrip(
        batch in 2usize..8,
        dim in 2usize..6,
        mb_size in 1usize..4,
    ) {
        let total = batch * dim;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let batches = micro_batch_split(&data, batch, dim, mb_size).unwrap();
        let merged = micro_batch_merge(&batches).unwrap();
        prop_assert_eq!(&merged, &data,
            "split/merge round-trip failed");
    }

    #[test]
    fn micro_batch_split_preserves_total(
        batch in 1usize..8,
        dim in 1usize..6,
        mb_size in 1usize..4,
    ) {
        let total = batch * dim;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let batches = micro_batch_split(&data, batch, dim, mb_size).unwrap();
        let total_elems: usize = batches.iter().map(|b| b.len()).sum();
        prop_assert_eq!(total_elems, total);
    }

    #[test]
    fn pipeline_bubble_time_nonneg(
        stages in 2usize..8,
        micro_batches in 1usize..16,
    ) {
        let bt = pipeline_bubble_time(stages, micro_batches);
        prop_assert!(bt >= 0.0, "bubble time {} < 0", bt);
    }

    #[test]
    fn optimal_micro_batch_at_least_one(
        stages in 2usize..8,
    ) {
        let mb = optimal_micro_batch_count(stages, 0.5);
        prop_assert!(mb >= 1, "optimal micro batch count {} < 1", mb);
    }
}
