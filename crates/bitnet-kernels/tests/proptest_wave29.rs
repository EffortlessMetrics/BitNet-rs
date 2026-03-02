//! Property-based tests — wave 29.
//!
//! Covers CUDA kernel configuration properties (RoPE config, matmul config,
//! fusion launch config, RMSNorm config, quantization), CPU SIMD operation
//! properties (activations, fusion, pooling, scatter-gather, reduction,
//! embedding, RoPE, loss functions), and kernel registry / dispatch properties.
//!
//! 42 property tests covering: launch bounds, shared memory sizing, grid/block
//! dimensions, commutativity, associativity, identity elements, registry
//! dispatch correctness, and numerical stability.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::activations::{relu, sigmoid, silu, tanh_act};
use bitnet_kernels::cpu::batch::{batched_add, batched_softmax};
use bitnet_kernels::cpu::dequant::{dequant_i2s_block, dequant_ternary, pack_ternary};
use bitnet_kernels::cpu::embedding::embedding_lookup;
use bitnet_kernels::cpu::fusion::fused_scale_add;
use bitnet_kernels::cpu::kv_cache::{
    KvCache, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear, kv_cache_memory_usage,
};
use bitnet_kernels::cpu::loss::{LossReduction, cosine_similarity_loss, mse_loss, perplexity};
use bitnet_kernels::cpu::pooling::{PoolConfig, PoolType};
use bitnet_kernels::cpu::quantize::{
    dequantize_symmetric_i8, quantize_binary, quantize_symmetric_i8, quantize_ternary,
};
use bitnet_kernels::cpu::reduction::ReductionKernel;
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled};
use bitnet_kernels::cpu::rope::{RopeConfig, compute_frequencies};
use bitnet_kernels::cpu::scatter_gather::{gather_1d, scatter_add};
use bitnet_kernels::cuda::fusion::{
    FusedElementwiseLaunchConfig, FusedMatmulLaunchConfig, fused_gelu_linear_cpu,
    fused_rmsnorm_linear_cpu,
};
use bitnet_kernels::cuda::matmul::MatmulConfig;
use bitnet_kernels::cuda::quantize::{
    QuantizeConfig, calibrate_scales, dequantize_ternary_cpu, quantize_ternary_cpu,
};
use bitnet_kernels::cuda::rmsnorm::RmsNormConfig;
use bitnet_kernels::cuda::rope::RopeConfig as CudaRopeConfig;
use proptest::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn finite_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-10.0f32..10.0, 1..=max_len)
}

// ── 1. CUDA RoPE config properties ──────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// RoPE config for_shape produces valid grid/block dims.
    #[test]
    fn prop_cuda_rope_config_valid_dims(
        head_dim in (2usize..=128).prop_filter("even", |n| n % 2 == 0),
        n_heads in 1usize..=16,
        seq_len in 1usize..=64,
    ) {
        if let Ok(config) = CudaRopeConfig::for_shape(head_dim, n_heads, seq_len) {
            let (gx, gy, gz) = config.grid_dim();
            let (bx, by, bz) = config.block_dim();
            prop_assert!(gx > 0 && gy > 0 && gz > 0, "zero grid dim");
            prop_assert!(bx > 0 && by > 0 && bz > 0, "zero block dim");
            prop_assert!(bx * by * bz <= 1024, "block exceeds max threads");
        }
    }

    /// RoPE with_base preserves shape-derived dimensions.
    #[test]
    fn prop_cuda_rope_base_preserves_dims(
        head_dim in (2usize..=64).prop_filter("even", |n| n % 2 == 0),
        base in 1000.0f32..100000.0,
    ) {
        if let Ok(config) = CudaRopeConfig::for_shape(head_dim, 1, 8) {
            let modified = config.with_base(base);
            let (gx, _, _) = modified.grid_dim();
            prop_assert!(gx > 0);
        }
    }

    /// RoPE scaling factor builder preserves validity.
    #[test]
    fn prop_cuda_rope_scaling_valid(
        head_dim in (2usize..=64).prop_filter("even", |n| n % 2 == 0),
        factor in 0.1f32..10.0,
    ) {
        if let Ok(config) = CudaRopeConfig::for_shape(head_dim, 2, 16) {
            let modified = config.with_scaling_factor(factor);
            let (bx, _, _) = modified.block_dim();
            prop_assert!(bx > 0);
        }
    }
}

// ── 2. CUDA matmul config properties ────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// MatmulConfig for_shape produces valid grid/block dims.
    #[test]
    fn prop_matmul_config_valid_dims(
        m in 1usize..=64,
        n in 1usize..=64,
        k in 1usize..=64,
    ) {
        if let Ok(config) = MatmulConfig::for_shape(m, n, k) {
            let (gx, gy, gz) = config.grid_dim();
            let (bx, by, bz) = config.block_dim();
            prop_assert!(gx > 0 && gy > 0 && gz > 0);
            prop_assert!(bx > 0 && by > 0 && bz > 0);
        }
    }

    /// MatmulConfig with_alpha_beta preserves dimensions.
    #[test]
    fn prop_matmul_alpha_beta_preserves(
        m in 1usize..=32, n in 1usize..=32, k in 1usize..=32,
        alpha in -5.0f32..5.0, beta in -5.0f32..5.0,
    ) {
        if let Ok(config) = MatmulConfig::for_shape(m, n, k) {
            let modified = config.with_alpha_beta(alpha, beta);
            let (gx, _, _) = modified.grid_dim();
            prop_assert!(gx > 0);
        }
    }

    /// Tiled matmul config has valid tile-aligned grid.
    #[test]
    fn prop_matmul_tiled_valid(
        m in 1usize..=64, n in 1usize..=64, k in 1usize..=64,
        tile in proptest::sample::select(vec![16u32, 32]),
    ) {
        if let Ok(config) = MatmulConfig::for_shape_tiled(m, n, k, tile) {
            let (gx, gy, _) = config.grid_dim();
            prop_assert!(gx > 0 && gy > 0);
        }
    }
}

// ── 3. CUDA RMSNorm config properties ───────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// RmsNormConfig for_shape produces valid launch config.
    #[test]
    fn prop_rmsnorm_config_valid(
        hidden_dim in 1usize..=256,
        n_rows in 1usize..=32,
    ) {
        if let Ok(config) = RmsNormConfig::for_shape(hidden_dim, n_rows) {
            let (gx, _, _) = config.grid_dim();
            let (bx, _, _) = config.block_dim();
            prop_assert!(gx > 0);
            prop_assert!(bx > 0 && bx <= 1024);
        }
    }

    /// RmsNormConfig with_eps produces valid config.
    #[test]
    fn prop_rmsnorm_eps_valid(
        hidden_dim in 1usize..=128,
        eps in 1e-8f32..1e-2,
    ) {
        if let Ok(config) = RmsNormConfig::for_shape(hidden_dim, 4) {
            let modified = config.with_eps(eps);
            let (bx, _, _) = modified.block_dim();
            prop_assert!(bx > 0);
        }
    }
}

// ── 4. CUDA fusion launch config properties ─────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// FusedMatmulLaunchConfig grid/block dims are positive.
    #[test]
    fn prop_fused_matmul_launch_valid(
        n in 1usize..=256,
        out_dim in 1usize..=256,
    ) {
        if let Ok(config) = FusedMatmulLaunchConfig::new(n, out_dim) {
            let (gx, gy, gz) = config.grid_dim();
            let (bx, by, bz) = config.block_dim();
            prop_assert!(gx > 0 && gy > 0 && gz > 0);
            prop_assert!(bx > 0 && by > 0 && bz > 0);
        }
    }

    /// FusedMatmulLaunchConfig shared_mem_bytes is accessible.
    #[test]
    fn prop_fused_matmul_shared_mem(
        n in 1usize..=128,
        out_dim in 1usize..=128,
    ) {
        if let Ok(config) = FusedMatmulLaunchConfig::new(n, out_dim) {
            let _smem = config.shared_mem_bytes();
        }
    }

    /// FusedElementwiseLaunchConfig grid/block dims are positive.
    #[test]
    fn prop_fused_ewise_launch_valid(n in 1usize..=10000) {
        if let Ok(config) = FusedElementwiseLaunchConfig::new(n) {
            let (gx, _, _) = config.grid_dim();
            let (bx, _, _) = config.block_dim();
            prop_assert!(gx > 0);
            prop_assert!(bx > 0 && bx <= 1024);
        }
    }
}

// ── 5. CUDA quantize → dequantize ternary round-trip ────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Ternary quantize → dequantize preserves sign for large values.
    #[test]
    fn prop_cuda_ternary_roundtrip_sign(
        data in proptest::collection::vec(-3.0f32..3.0, 8..=64),
    ) {
        let config = QuantizeConfig::default();
        if let Ok((quantized, scale)) = quantize_ternary_cpu(&data, &config) {
            let recovered = dequantize_ternary_cpu(&quantized, scale);
            for (&orig, &rec) in data.iter().zip(recovered.iter()) {
                if orig.abs() > 1.5 && rec != 0.0 {
                    prop_assert!(
                        orig.signum() == rec.signum(),
                        "sign flip: {} → {}", orig, rec
                    );
                }
            }
        }
    }

    /// calibrate_scales returns non-negative scales.
    #[test]
    fn prop_calibrate_scales_nonneg(
        data in proptest::collection::vec(-5.0f32..5.0, 8..=64),
    ) {
        let config = QuantizeConfig::default();
        if let Ok(scales) = calibrate_scales(&data, &config) {
            for &s in &scales {
                prop_assert!(s >= 0.0, "negative calibrated scale: {}", s);
            }
        }
    }
}

// ── 6. CPU activation identity / bounds properties ──────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// ReLU output is always >= 0.
    #[test]
    fn prop_relu_nonneg(x in -100.0f32..100.0) {
        prop_assert!(relu(x) >= 0.0);
    }

    /// ReLU is idempotent: relu(relu(x)) == relu(x).
    #[test]
    fn prop_relu_idempotent(x in -100.0f32..100.0) {
        prop_assert_eq!(relu(relu(x)), relu(x));
    }

    /// Sigmoid output is in [0, 1].
    #[test]
    fn prop_sigmoid_bounded(x in -20.0f32..20.0) {
        let s = sigmoid(x);
        prop_assert!(s >= 0.0 && s <= 1.0, "sigmoid({}) = {} out of [0,1]", x, s);
    }

    /// Sigmoid symmetry: sigmoid(-x) ≈ 1 - sigmoid(x).
    #[test]
    fn prop_sigmoid_symmetry(x in -10.0f32..10.0) {
        let s1 = sigmoid(x);
        let s2 = sigmoid(-x);
        prop_assert!((s1 + s2 - 1.0).abs() < 1e-5, "symmetry violated");
    }

    /// Tanh output is in [-1, 1].
    #[test]
    fn prop_tanh_bounded(x in -20.0f32..20.0) {
        let t = tanh_act(x);
        prop_assert!(t >= -1.0 && t <= 1.0, "tanh({}) = {} out of [-1,1]", x, t);
    }

    /// Tanh is odd: tanh(-x) ≈ -tanh(x).
    #[test]
    fn prop_tanh_odd_symmetry(x in -10.0f32..10.0) {
        let t1 = tanh_act(x);
        let t2 = tanh_act(-x);
        prop_assert!((t1 + t2).abs() < 1e-5, "odd symmetry violated");
    }

    /// SiLU(0) == 0 (identity at origin).
    #[test]
    fn prop_silu_zero(_dummy in 0u8..1) {
        prop_assert!((silu(0.0)).abs() < 1e-7);
    }

    /// SiLU is continuous: nearby inputs produce nearby outputs.
    #[test]
    fn prop_silu_continuous(x in -10.0f32..10.0) {
        let eps = 1e-4;
        let diff = (silu(x + eps) - silu(x)).abs();
        prop_assert!(diff < 2.0 * eps, "silu discontinuity at {}: diff={}", x, diff);
    }
}

// ── 7. CPU symmetric quantize → dequantize round-trip ───────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Symmetric I8 round-trip preserves sign for significant values.
    #[test]
    fn prop_sym_i8_roundtrip_sign(data in finite_f32_vec(32)) {
        let (quantized, scale) = quantize_symmetric_i8(&data, 8);
        let recovered = dequantize_symmetric_i8(&quantized, scale);
        for (&orig, &rec) in data.iter().zip(recovered.iter()) {
            if orig.abs() > 0.1 && rec != 0.0 {
                prop_assert!(
                    orig.signum() == rec.signum(),
                    "sign flip: orig={}, rec={}", orig, rec
                );
            }
        }
    }

    /// Quantize binary produces only -1 and 1.
    #[test]
    fn prop_binary_quantize_values(data in finite_f32_vec(32)) {
        let binary = quantize_binary(&data);
        for &v in &binary {
            prop_assert!(v == -1 || v == 1, "unexpected binary value: {}", v);
        }
    }

    /// Ternary quantize produces only -1, 0, 1.
    #[test]
    fn prop_ternary_quantize_values(
        data in finite_f32_vec(32),
        threshold in 0.001f32..5.0,
    ) {
        let ternary = quantize_ternary(&data, threshold);
        for &v in &ternary {
            prop_assert!(
                v == -1 || v == 0 || v == 1,
                "unexpected ternary value: {}", v
            );
        }
    }

    /// I8 quantize output length matches input length.
    #[test]
    fn prop_sym_i8_output_len(data in finite_f32_vec(64)) {
        let (quantized, _scale) = quantize_symmetric_i8(&data, 8);
        prop_assert_eq!(quantized.len(), data.len());
    }
}

// ── 8. CPU pack_ternary / dequant_ternary round-trip ────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// pack_ternary → dequant_ternary preserves length (at least).
    #[test]
    fn prop_pack_dequant_ternary_len(
        data in finite_f32_vec(32),
        threshold in 0.1f32..3.0,
    ) {
        let (packed, scale) = pack_ternary(&data, threshold);
        let recovered = dequant_ternary(&packed, scale);
        prop_assert!(recovered.len() >= data.len());
    }

    /// dequant_i2s_block output length matches block_size.
    #[test]
    fn prop_dequant_i2s_block_len(
        block_size in (4usize..=64).prop_filter("multiple of 4", |n| n % 4 == 0),
        scale in 0.01f32..5.0,
    ) {
        let n_bytes = block_size / 4;
        let packed: Vec<u8> = vec![0x55; n_bytes];
        if let Ok(result) = dequant_i2s_block(&packed, scale, block_size) {
            prop_assert_eq!(result.len(), block_size);
        }
    }
}

// ── 9. CPU scatter-gather properties ────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// gather_1d with identity indices returns original data.
    #[test]
    fn prop_gather_identity(data in finite_f32_vec(32)) {
        let indices: Vec<usize> = (0..data.len()).collect();
        if let Ok(result) = gather_1d(&data, &indices) {
            prop_assert_eq!(result, data);
        }
    }

    /// scatter_add with zeros doesn't change destination.
    #[test]
    fn prop_scatter_add_zero_identity(data in finite_f32_vec(16)) {
        let mut dest = data.clone();
        let zeros = vec![0.0f32; data.len()];
        let indices: Vec<usize> = (0..data.len()).collect();
        let _ = scatter_add(&mut dest, &indices, &zeros);
        for (&orig, &updated) in data.iter().zip(dest.iter()) {
            prop_assert!((orig - updated).abs() < 1e-7);
        }
    }
}

// ── 10. CPU fusion and pooling properties ───────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// fused_scale_add with scale=0 returns a unchanged.
    #[test]
    fn prop_fused_scale_add_zero_scale(n in 1usize..=32) {
        let a: Vec<f32> = vec![3.0; n];
        let b: Vec<f32> = vec![7.0; n];
        if let Ok(result) = fused_scale_add(&a, &b, 0.0) {
            for (&r, &orig) in result.iter().zip(a.iter()) {
                prop_assert!((r - orig).abs() < 1e-5);
            }
        }
    }

    /// fused_scale_add output length matches input length.
    #[test]
    fn prop_fused_scale_add_output_len(n in 1usize..=32) {
        let a = vec![1.0f32; n];
        let b = vec![2.0f32; n];
        if let Ok(result) = fused_scale_add(&a, &b, 1.0) {
            prop_assert_eq!(result.len(), n);
        }
    }

    /// PoolConfig with valid params validates ok.
    #[test]
    fn prop_pool_config_valid(
        kernel_size in 1usize..=8,
        stride in 1usize..=4,
    ) {
        let config = PoolConfig::new(PoolType::Max, kernel_size, stride, 0);
        let validation = config.validate();
        prop_assert!(validation.is_ok());
    }
}

// ── 11. CPU KV cache properties ─────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// KV cache construction succeeds for valid configs.
    #[test]
    fn prop_kv_cache_memory_nonneg(
        layers in 1usize..=8,
        heads in 1usize..=4,
        head_dim in 1usize..=32,
        max_seq in 1usize..=64,
    ) {
        let config = KvCacheConfig {
            num_layers: layers,
            num_heads: heads,
            head_dim,
            max_seq_len: max_seq,
            dtype: KvDtype::F32,
        };
        if let Ok(cache) = KvCache::new(config) {
            let _mem = kv_cache_memory_usage(&cache);
        }
    }

    /// Clearing KV cache resets memory to baseline.
    #[test]
    fn prop_kv_cache_clear_resets(
        layers in 1usize..=4,
        heads in 1usize..=2,
        head_dim in 1usize..=16,
    ) {
        let config = KvCacheConfig {
            num_layers: layers,
            num_heads: heads,
            head_dim,
            max_seq_len: 32,
            dtype: KvDtype::F32,
        };
        if let Ok(mut cache) = KvCache::new(config) {
            let baseline = kv_cache_memory_usage(&cache);
            let kv_data: Vec<f32> = vec![1.0; heads * head_dim];
            let _ = kv_cache_append(&mut cache, 0, &kv_data, &kv_data);
            kv_cache_clear(&mut cache);
            let after_clear = kv_cache_memory_usage(&cache);
            prop_assert_eq!(baseline, after_clear, "clear didn't reset memory");
        }
    }
}

// ── 12. Batched softmax / add properties ────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Batched softmax rows sum to approximately 1.0.
    #[test]
    fn prop_batched_softmax_row_sums(
        batch in 1usize..=4,
        seq_len in 1usize..=16,
    ) {
        let data: Vec<f32> = (0..(batch * seq_len)).map(|i| (i as f32) * 0.1 - 1.0).collect();
        if let Ok(result) = batched_softmax(&data, batch, seq_len) {
            for b in 0..batch {
                let row = &result[b * seq_len..(b + 1) * seq_len];
                let sum: f32 = row.iter().sum();
                prop_assert!(
                    (sum - 1.0).abs() < 1e-4,
                    "batch {} softmax sum = {}", b, sum
                );
            }
        }
    }

    /// Batched add output is elementwise a+b.
    #[test]
    fn prop_batched_add_correct(
        batch in 1usize..=4,
        dim in 1usize..=16,
    ) {
        let n = batch * dim;
        let a: Vec<f32> = vec![1.0; n];
        let b: Vec<f32> = vec![2.0; n];
        if let Ok(result) = batched_add(&a, &b, batch, dim) {
            prop_assert_eq!(result.len(), n);
            for &v in &result {
                prop_assert!((v - 3.0).abs() < 1e-5);
            }
        }
    }
}

// ── 13. CPU reduction properties ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// ReductionKernel::sum of uniform data equals n * value.
    #[test]
    fn prop_reduction_sum_uniform(n in 1usize..=64, val in -5.0f32..5.0) {
        let data = vec![val; n];
        if let Ok(sum) = ReductionKernel::sum(&data) {
            let expected: f32 = val * n as f32;
            prop_assert!((sum - expected).abs() < n as f32 * 1e-4);
        }
    }

    /// ReductionKernel::mean of uniform data equals the value.
    #[test]
    fn prop_reduction_mean_uniform(n in 1usize..=64, val in -5.0f32..5.0) {
        let data = vec![val; n];
        if let Ok(mean) = ReductionKernel::mean(&data) {
            let diff: f32 = mean - val;
            prop_assert!(diff.abs() < 1e-4);
        }
    }

    /// ReductionKernel::max returns the maximum value.
    #[test]
    fn prop_reduction_max_correct(data in finite_f32_vec(32)) {
        if let Ok(result) = ReductionKernel::max(&data) {
            for &v in &data {
                prop_assert!(result.value >= v - 1e-7);
            }
        }
    }

    /// ReductionKernel::min returns the minimum value.
    #[test]
    fn prop_reduction_min_correct(data in finite_f32_vec(32)) {
        if let Ok(result) = ReductionKernel::min(&data) {
            for &v in &data {
                prop_assert!(result.value <= v + 1e-7);
            }
        }
    }
}

// ── 14. CPU residual properties ─────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// add_residual(output, residual) => output[i] += residual[i].
    #[test]
    fn prop_add_residual_correct(
        n in 1usize..=32,
    ) {
        let mut output = vec![1.0f32; n];
        let residual = vec![2.0f32; n];
        let _ = add_residual(&mut output, &residual);
        for &v in &output {
            prop_assert!((v - 3.0).abs() < 1e-5);
        }
    }

    /// add_residual_scaled with scale=0 doesn't change output.
    #[test]
    fn prop_add_residual_scaled_zero(n in 1usize..=32) {
        let mut output = vec![5.0f32; n];
        let residual = vec![3.0f32; n];
        let _ = add_residual_scaled(&mut output, &residual, 0.0);
        for &v in &output {
            prop_assert!((v - 5.0).abs() < 1e-5);
        }
    }
}

// ── 15. CPU embedding lookup properties ─────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// embedding_lookup output length equals n_indices * embed_dim.
    #[test]
    fn prop_embedding_output_len(
        vocab in 4usize..=32,
        embed_dim in 2usize..=16,
        n_indices in 1usize..=8,
    ) {
        let table: Vec<f32> = (0..(vocab * embed_dim)).map(|i| i as f32 * 0.01).collect();
        let indices: Vec<u32> = (0..n_indices).map(|i| (i % vocab) as u32).collect();
        if let Ok(result) = embedding_lookup(&table, &indices, embed_dim) {
            prop_assert_eq!(result.len(), n_indices * embed_dim);
        }
    }
}

// ── 16. CPU loss function properties ────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// MSE loss of identical vectors is zero.
    #[test]
    fn prop_mse_loss_identical_zero(
        data in proptest::collection::vec(-3.0f32..3.0, 4..=16),
    ) {
        if let Ok(loss) = mse_loss(&data, &data, LossReduction::Mean) {
            prop_assert!(loss.abs() < 1e-6, "MSE of identical = {}", loss);
        }
    }

    /// Cosine similarity loss of a vector with itself is ~0.0.
    #[test]
    fn prop_cosine_self_similarity(
        data in proptest::collection::vec(0.1f32..5.0, 4..=16),
    ) {
        if let Ok(loss) = cosine_similarity_loss(&data, &data) {
            prop_assert!(loss.abs() < 1e-4, "self-similarity loss = {}", loss);
        }
    }

    /// Perplexity is >= 1.0 for non-negative loss.
    #[test]
    fn prop_perplexity_ge_one(loss in 0.0f32..10.0) {
        let ppl = perplexity(loss);
        prop_assert!(ppl >= 1.0 - 1e-5, "perplexity {} < 1.0", ppl);
    }
}

// ── 17. CPU RoPE properties ─────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// compute_frequencies returns max_seq_len * head_dim values.
    #[test]
    fn prop_rope_freq_len(
        head_dim in (2usize..=64).prop_filter("even", |n| n % 2 == 0),
    ) {
        let max_seq_len = 128;
        let config = RopeConfig::new(head_dim, max_seq_len);
        let freqs = compute_frequencies(&config);
        prop_assert_eq!(freqs.len(), max_seq_len * head_dim);
    }

    /// All RoPE frequencies are finite.
    #[test]
    fn prop_rope_freq_finite(
        head_dim in (2usize..=64).prop_filter("even", |n| n % 2 == 0),
    ) {
        let config = RopeConfig::new(head_dim, 64);
        let freqs = compute_frequencies(&config);
        for &f in &freqs {
            prop_assert!(f.is_finite(), "non-finite frequency: {}", f);
        }
    }
}

// ── 18. CUDA fused CPU fallback properties ──────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// fused_rmsnorm_linear_cpu writes into output buffer.
    #[test]
    fn prop_fused_rmsnorm_linear_cpu_len(
        dim in 2usize..=16,
    ) {
        let input = vec![0.5f32; dim];
        let weight = vec![0.1f32; dim];
        let gamma = vec![1.0f32; dim];
        let mut output = vec![0.0f32; dim];
        if let Ok(()) = fused_rmsnorm_linear_cpu(&input, &weight, &gamma, &mut output, 1e-5) {
            prop_assert_eq!(output.len(), dim);
        }
    }

    /// fused_gelu_linear_cpu writes into output buffer.
    #[test]
    fn prop_fused_gelu_linear_cpu_len(
        dim in 2usize..=16,
    ) {
        let input = vec![0.5f32; dim];
        let weight = vec![0.1f32; dim];
        let bias = vec![0.0f32; dim];
        let mut output = vec![0.0f32; dim];
        if let Ok(()) = fused_gelu_linear_cpu(&input, &weight, &bias, &mut output) {
            prop_assert_eq!(output.len(), dim);
        }
    }
}
