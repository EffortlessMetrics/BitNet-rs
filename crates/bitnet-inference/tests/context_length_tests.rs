//! Context length scaling tests for 16K (Phi-4), 128K (Qwen2.5), and beyond.
//!
//! Validates RoPE table generation, KV cache sizing, and context boundary
//! handling at context lengths required by modern SLMs.

use bitnet_common::{BitNetTensor, Device, Tensor};
use bitnet_kernels::cpu::rope::{self, RopeConfig};
use bitnet_rope::build_tables as build_rope_tables;
use candle_core::DType;

// ── RoPE scaling tests ──────────────────────────────────────────────────────

mod rope_scaling {
    use super::*;

    #[test]
    fn rope_tables_4096_default_bitnet() {
        let cfg = RopeConfig::new(128, 4096);
        let freqs = rope::compute_frequencies(&cfg);
        assert_eq!(freqs.len(), 4096 * 128, "4096 × 128 frequency table");
        assert!(freqs.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rope_tables_16384_phi4() {
        let cfg = RopeConfig::new(128, 16384);
        let freqs = rope::compute_frequencies(&cfg);
        assert_eq!(freqs.len(), 16384 * 128, "16K context for Phi-4");
        assert!(freqs.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rope_tables_131072_qwen25() {
        let cfg = RopeConfig::new(128, 131072);
        let freqs = rope::compute_frequencies(&cfg);
        assert_eq!(freqs.len(), 131072 * 128, "128K context for Qwen2.5");
        assert!(freqs.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rope_frequency_base_affects_output() {
        let base_10k = RopeConfig::new(128, 32).with_base(10_000.0);
        let base_500k = RopeConfig::new(128, 32).with_base(500_000.0);

        let f1 = rope::compute_frequencies(&base_10k);
        let f2 = rope::compute_frequencies(&base_500k);

        // Position 0 is always cos=1,sin=0 for both bases
        // Position 1+ should differ
        let pos1_offset = 128; // head_dim elements per position
        let any_diff = (0..128).any(|i| (f1[pos1_offset + i] - f2[pos1_offset + i]).abs() > 1e-7);
        assert!(any_diff, "base=10000 vs base=500000 must produce different frequencies");
    }

    #[test]
    fn rope_tables_deterministic() {
        let cfg = RopeConfig::new(128, 16384);
        let run1 = rope::compute_frequencies(&cfg);
        let run2 = rope::compute_frequencies(&cfg);
        assert_eq!(run1.len(), run2.len());
        for (a, b) in run1.iter().zip(run2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "RoPE tables must be bitwise identical");
        }
    }

    #[test]
    fn rope_table_memory_reasonable() {
        let sizes =
            [(4096, 128, "4K default"), (16384, 128, "16K Phi-4"), (131072, 128, "128K Qwen2.5")];
        for (seq_len, head_dim, label) in sizes {
            let cfg = RopeConfig::new(head_dim, seq_len);
            let freqs = rope::compute_frequencies(&cfg);
            let bytes = freqs.len() * std::mem::size_of::<f32>();
            eprintln!("{label}: {bytes} bytes ({:.2} MB)", bytes as f64 / 1024.0 / 1024.0);
            // Interleaved cos/sin: seq_len * head_dim * 4 bytes
            let expected = seq_len * head_dim * 4;
            assert_eq!(bytes, expected, "{label}: memory must be seq_len × head_dim × 4");
        }
    }

    #[test]
    fn rope_values_bounded_sin_cos() {
        // Use bitnet-rope crate (the one used by RotaryEmbedding in attention.rs)
        for seq_len in [4096, 16384, 131072] {
            let tables =
                build_rope_tables(128, seq_len, 10_000.0).expect("build_tables should succeed");
            for &v in tables.sin.iter().chain(tables.cos.iter()) {
                assert!(
                    (-1.0..=1.0).contains(&v),
                    "sin/cos value {v} out of [-1,1] at seq_len={seq_len}"
                );
            }
        }
    }

    #[test]
    fn rope_tables_head_dim_variants() {
        for head_dim in [64, 128, 256] {
            let tables =
                build_rope_tables(head_dim, 16384, 10_000.0).expect("build_tables should succeed");
            assert_eq!(tables.half_dim, head_dim / 2);
            assert_eq!(tables.sin.len(), 16384 * (head_dim / 2));
            assert_eq!(tables.cos.len(), 16384 * (head_dim / 2));
            // sin²+cos² ≈ 1 for a sampled position
            let pos = 8192;
            for i in 0..tables.half_dim {
                let idx = pos * tables.half_dim + i;
                let norm = tables.sin[idx] * tables.sin[idx] + tables.cos[idx] * tables.cos[idx];
                assert!(
                    (norm - 1.0).abs() < 1e-5,
                    "sin²+cos² = {norm} ≠ 1 at head_dim={head_dim}, pos={pos}, i={i}"
                );
            }
        }
    }
}

// ── KV cache sizing tests ───────────────────────────────────────────────────

mod kv_cache_sizing {
    use super::*;
    use bitnet_inference::layers::attention::KVCache;

    /// Helper: compute expected KV cache memory for given parameters.
    fn expected_kv_bytes(seq_len: usize, layers: usize, kv_heads: usize, head_dim: usize) -> usize {
        // Each layer has K + V tensors, each of shape [seq_len, kv_heads, head_dim]
        seq_len * kv_heads * head_dim * std::mem::size_of::<f32>() * 2 * layers
    }

    #[test]
    fn cache_allocates_small_config() {
        // Small config that actually allocates: 256 context × 4 layers × 4 heads × 64 dim
        let device = Device::Cpu;
        let cache = KVCache::new(256, 4, 4, 64, &device);
        assert!(cache.is_ok(), "small cache should allocate");
        let cache = cache.unwrap();
        let stats = cache.memory_usage();
        let expected = expected_kv_bytes(256, 4, 4, 64);
        assert_eq!(stats["tensor_memory_bytes"], expected);
    }

    #[test]
    fn cache_memory_estimation_4096_30_layers() {
        // BitNet default: 4096 context × 30 layers × 32 heads × 128 dim
        let total = expected_kv_bytes(4096, 30, 32, 128);
        eprintln!("4096×30 layers: {total} bytes ({:.2} GB)", total as f64 / 1e9);
        // ~3.75 GB — confirm formula
        assert_eq!(total, 4096 * 32 * 128 * 4 * 2 * 30);

        // Verify with actual small allocation that memory_usage matches formula
        let device = Device::Cpu;
        let cache = KVCache::new(64, 2, 4, 64, &device).unwrap();
        let stats = cache.memory_usage();
        assert_eq!(stats["tensor_memory_bytes"], expected_kv_bytes(64, 2, 4, 64));
    }

    #[test]
    fn cache_memory_estimation_16k_phi4() {
        // Phi-4: 16K context, 40 layers, 10 KV heads, head_dim=128
        let total = expected_kv_bytes(16384, 40, 10, 128);
        eprintln!("Phi-4 KV cache: {total} bytes ({:.2} GB)", total as f64 / 1e9);
        // ~6.71 GB
        assert_eq!(total, 16384 * 10 * 128 * 4 * 2 * 40);

        // Verify formula at small scale matches actual allocation
        let device = Device::Cpu;
        let cache = KVCache::new(128, 2, 10, 128, &device).unwrap();
        let stats = cache.memory_usage();
        assert_eq!(stats["tensor_memory_bytes"], expected_kv_bytes(128, 2, 10, 128));
    }

    #[test]
    fn cache_partial_fill() {
        let device = Device::Cpu;
        let mut cache = KVCache::new(512, 2, 4, 64, &device).unwrap();

        // Fill only first 100 positions for layer 0
        let k = BitNetTensor::zeros(&[100, 4, 64], DType::F32, &device).unwrap();
        let v = BitNetTensor::zeros(&[100, 4, 64], DType::F32, &device).unwrap();
        cache.update(0, k, v, 100).unwrap();

        let (k_out, v_out) = cache.get(0).unwrap();
        assert_eq!(k_out.shape()[0], 100, "K cache should be sliced to 100 positions");
        assert_eq!(v_out.shape()[0], 100, "V cache should be sliced to 100 positions");
    }

    #[test]
    fn cache_position_overflow_returns_error() {
        let device = Device::Cpu;
        let mut cache = KVCache::new(128, 2, 4, 64, &device).unwrap();

        // Layer index out of bounds should error
        let k = BitNetTensor::zeros(&[1, 4, 64], DType::F32, &device).unwrap();
        let v = BitNetTensor::zeros(&[1, 4, 64], DType::F32, &device).unwrap();
        let result = cache.update(99, k, v, 1);
        assert!(result.is_err(), "layer_idx out of bounds should error");
    }

    #[test]
    fn cache_gqa_smaller_than_mha() {
        let device = Device::Cpu;
        let seq_len = 256;
        let layers = 4;
        let head_dim = 64;

        // MHA: 32 KV heads
        let mha = KVCache::new(seq_len, layers, 32, head_dim, &device).unwrap();
        let mha_mem = mha.memory_usage()["tensor_memory_bytes"];

        // GQA: 8 KV heads (4× fewer)
        let gqa = KVCache::new(seq_len, layers, 8, head_dim, &device).unwrap();
        let gqa_mem = gqa.memory_usage()["tensor_memory_bytes"];

        assert!(
            gqa_mem < mha_mem,
            "GQA ({gqa_mem} bytes) must use less memory than MHA ({mha_mem} bytes)"
        );
        let ratio = mha_mem as f64 / gqa_mem as f64;
        assert!((ratio - 4.0).abs() < 0.1, "GQA should be ~4× smaller: ratio={ratio:.2}");

        // Also verify the formula scales to Phi-4 sizes
        let mha_16k = expected_kv_bytes(16384, 40, 32, 128);
        let gqa_16k = expected_kv_bytes(16384, 40, 8, 128);
        assert_eq!(mha_16k / gqa_16k, 4, "GQA at 16K scale is 4× smaller");
    }

    #[test]
    fn cache_incremental_append() {
        let device = Device::Cpu;
        let mut cache = KVCache::new(1024, 1, 4, 64, &device).unwrap();

        for i in 1..=5 {
            let k = BitNetTensor::zeros(&[i, 4, 64], DType::F32, &device).unwrap();
            let v = BitNetTensor::zeros(&[i, 4, 64], DType::F32, &device).unwrap();
            cache.update(0, k, v, i).unwrap();

            let (k_out, _) = cache.get(0).unwrap();
            assert_eq!(k_out.shape()[0], i, "after {i} tokens, cache should have {i} positions");
        }
    }

    #[test]
    fn cache_clear_resets() {
        let device = Device::Cpu;
        let mut cache = KVCache::new(1024, 2, 4, 64, &device).unwrap();

        let k = BitNetTensor::zeros(&[50, 4, 64], DType::F32, &device).unwrap();
        let v = BitNetTensor::zeros(&[50, 4, 64], DType::F32, &device).unwrap();
        cache.update(0, k, v, 50).unwrap();

        cache.clear(&device).unwrap();

        let (k_out, v_out) = cache.get(0).unwrap();
        assert_eq!(k_out.shape()[0], 0, "K cache should be empty after clear");
        assert_eq!(v_out.shape()[0], 0, "V cache should be empty after clear");
    }
}

// ── Context boundary tests ──────────────────────────────────────────────────

mod context_boundaries {
    use super::*;
    use bitnet_inference::layers::attention::RotaryEmbedding;
    use bitnet_kernels::cpu::attention_mask::create_causal_mask;

    #[test]
    fn rope_at_exactly_max_context() {
        let max_ctx = 16384;
        let rope = RotaryEmbedding::new(128, max_ctx, 10_000.0, &Device::Cpu);
        assert!(rope.is_ok(), "RoPE with max_context={max_ctx} should succeed");
        let rope = rope.unwrap();
        eprintln!("RoPE cache memory: {} bytes", rope.cache_memory_usage());
    }

    #[test]
    fn rope_exceeding_max_context_errors() {
        let max_ctx = 128;
        let rope = RotaryEmbedding::new(64, max_ctx, 10_000.0, &Device::Cpu).unwrap();

        // The async apply() checks seq_len > max_seq_len
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let result = rt.block_on(async {
            let tensor =
                BitNetTensor::zeros(&[1, max_ctx + 1, 4, 64], DType::F32, &Device::Cpu).unwrap();
            rope.apply(&tensor, max_ctx + 1).await
        });
        assert!(result.is_err(), "seq_len > max_context should error");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("exceeds"), "error should mention exceeding: {err}");
    }

    #[test]
    fn rope_single_token() {
        let rope = RotaryEmbedding::new(64, 1024, 10_000.0, &Device::Cpu).unwrap();

        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let result = rt.block_on(async {
            let tensor = BitNetTensor::zeros(&[1, 1, 4, 64], DType::F32, &Device::Cpu).unwrap();
            rope.apply(&tensor, 1).await
        });
        assert!(result.is_ok(), "single token should work: {:?}", result.err());
    }

    #[test]
    fn rope_zero_tokens_returns_clone() {
        let rope = RotaryEmbedding::new(64, 1024, 10_000.0, &Device::Cpu).unwrap();

        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let result = rt.block_on(async {
            let tensor = BitNetTensor::zeros(&[1, 0, 4, 64], DType::F32, &Device::Cpu).unwrap();
            rope.apply(&tensor, 0).await
        });
        // seq_len=0 returns Ok(clone) per the implementation
        assert!(result.is_ok(), "zero tokens should return clone, not error");
    }

    #[test]
    fn causal_mask_covers_full_context() {
        // Exhaustive check for small/medium sizes
        for ctx_len in [1, 128, 4096] {
            let mask = create_causal_mask(ctx_len);
            assert_eq!(mask.len(), ctx_len * ctx_len);

            for i in 0..ctx_len {
                for j in 0..ctx_len {
                    let val = mask[i * ctx_len + j];
                    if j <= i {
                        assert_eq!(val, 0.0, "mask[{i},{j}] should be 0 (attend)");
                    } else {
                        assert!(
                            val.is_infinite() && val < 0.0,
                            "mask[{i},{j}] should be -inf (block), got {val}"
                        );
                    }
                }
            }
        }

        // 16K: verify size and spot-check corners (full iteration is ~1GB)
        let ctx_16k = 16384;
        let mask = create_causal_mask(ctx_16k);
        assert_eq!(mask.len(), ctx_16k * ctx_16k);
        // Top-left: position (0,0) should be 0 (attend to self)
        assert_eq!(mask[0], 0.0);
        // Top-right: position (0, last) should be -inf
        assert!(mask[ctx_16k - 1].is_infinite() && mask[ctx_16k - 1] < 0.0);
        // Bottom-left: position (last, 0) should be 0 (attend to past)
        assert_eq!(mask[(ctx_16k - 1) * ctx_16k], 0.0);
        // Bottom-right: position (last, last) should be 0 (attend to self)
        assert_eq!(mask[(ctx_16k - 1) * ctx_16k + ctx_16k - 1], 0.0);
        // One past diagonal: (100, 101) should be -inf
        assert!(mask[100 * ctx_16k + 101].is_infinite());
        // One before diagonal: (101, 100) should be 0
        assert_eq!(mask[101 * ctx_16k + 100], 0.0);
    }
}
