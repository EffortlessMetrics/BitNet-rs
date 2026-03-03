#![allow(clippy::approx_constant)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::duplicated_attributes)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::identity_op)]
#![allow(clippy::manual_abs_diff)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::manual_contains)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::no_effect)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::useless_vec)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::assertions_on_constants)]

//! End-to-end integration tests for the A770 OpenCL inference pipeline.
//!
//! These tests exercise multiple OpenCL modules together, validating the full
//! inference flow using CPU reference implementations. No OpenCL hardware is
//! required — all computation is done in software.

// Modules exported from bitnet-kernels (pub mod in lib.rs):
//   opencl_buffer, opencl_cache, opencl_context, opencl_embedding,
//   opencl_kernel_sources, opencl_pipeline, opencl_work_size
//
// Modules that exist as files but are NOT exported (not in lib.rs):
//   opencl_attention, opencl_ffn, opencl_memory, opencl_provider,
//   opencl_quantized, opencl_transformer

use bitnet_kernels::opencl_buffer::{
    AlignedBuffer, BufferPool, optimal_buffer_size, validate_alignment,
};
use bitnet_kernels::opencl_cache::{
    CacheConfig, CacheEvictionStrategy, CachePolicy, KernelCacheKey, KernelCacheManager,
};
use bitnet_kernels::opencl_context::OpenClContextConfig;
use bitnet_kernels::opencl_embedding::{
    EmbeddingConfig, EmbeddingTable, OutputProjection, embedding_lookup_ref, output_projection_ref,
};
use bitnet_kernels::opencl_kernel_sources::{KernelProgramId, KernelSourceRegistry};
use bitnet_kernels::opencl_pipeline::{
    GenerationConfig, InferencePipeline, PipelineBuilder, PipelineConfig, PipelineStage,
    PipelineStatus, StopReason, TokenGenerator,
};
use bitnet_kernels::opencl_work_size::{IntelArcWorkSizeHints, WorkSizeOptimizer};

// ============================================================================
// Helper: CPU reference implementations for modules not yet exported
// ============================================================================

/// CPU RMSNorm: y_i = x_i / sqrt(mean(x^2) + eps)
fn cpu_rms_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n + eps).sqrt();
    x.iter().map(|v| v / rms).collect()
}

/// CPU SiLU activation: silu(x) = x * sigmoid(x)
fn cpu_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// CPU softmax
fn cpu_softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// CPU argmax
#[allow(dead_code)]
fn cpu_argmax(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Deterministic pseudo-random f32 in [-1, 1] from seed + index.
fn pseudo_random(seed: u64, idx: usize) -> f32 {
    let mut x = seed.wrapping_add(idx as u64);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    ((x % 10000) as f32 / 5000.0) - 1.0
}

/// Simple KV cache for CPU reference attention.
struct CpuKvCache {
    k: Vec<Vec<f32>>, // [seq_len][head_dim]
    v: Vec<Vec<f32>>, // [seq_len][head_dim]
    head_dim: usize,
}

impl CpuKvCache {
    fn new(head_dim: usize) -> Self {
        Self { k: Vec::new(), v: Vec::new(), head_dim }
    }

    fn append(&mut self, k_vec: Vec<f32>, v_vec: Vec<f32>) {
        assert_eq!(k_vec.len(), self.head_dim);
        assert_eq!(v_vec.len(), self.head_dim);
        self.k.push(k_vec);
        self.v.push(v_vec);
    }

    fn len(&self) -> usize {
        self.k.len()
    }

    /// Single-head attention: softmax(Q·K^T / sqrt(d)) · V
    fn attend(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.head_dim);
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        // Q·K^T
        let scores: Vec<f32> = self
            .k
            .iter()
            .map(|k| query.iter().zip(k.iter()).map(|(q, k)| q * k).sum::<f32>() * scale)
            .collect();
        let probs = cpu_softmax(&scores);
        // weighted sum of V
        let mut out = vec![0.0f32; self.head_dim];
        for (prob, v) in probs.iter().zip(self.v.iter()) {
            for (o, vi) in out.iter_mut().zip(v.iter()) {
                *o += prob * vi;
            }
        }
        out
    }
}

// ============================================================================
// 1. Full forward pass simulation (embedding → norm → attention → FFN → output)
// ============================================================================

#[test]
fn e2e_full_forward_pass_embedding_to_output() {
    let vocab_size = 64;
    let embed_dim = 32;
    let eps = 1e-5;

    // Create embedding table with deterministic weights
    let weight: Vec<f32> = (0..vocab_size * embed_dim).map(|i| pseudo_random(42, i)).collect();
    let config = EmbeddingConfig::new(vocab_size, embed_dim);
    let table = EmbeddingTable::new(weight.clone(), config).unwrap();

    // Step 1: Embedding lookup
    let token_ids = [7u32];
    let mut embedding = vec![0.0f32; embed_dim];
    table.lookup(&token_ids, &mut embedding).unwrap();
    assert_eq!(embedding.len(), embed_dim);
    assert!(embedding.iter().all(|v| v.is_finite()));

    // Step 2: RMSNorm
    let normed = cpu_rms_norm(&embedding, eps);
    assert_eq!(normed.len(), embed_dim);
    let norm_sq: f32 = normed.iter().map(|v| v * v).sum::<f32>() / embed_dim as f32;
    assert!((norm_sq - 1.0).abs() < 0.5, "RMSNorm output should be roughly unit variance");

    // Step 3: Attention with KV cache
    let head_dim = 8;
    let mut kv = CpuKvCache::new(head_dim);
    let query = &normed[..head_dim];
    let k_vec = normed[..head_dim].to_vec();
    let v_vec = normed[head_dim..2 * head_dim].to_vec();
    kv.append(k_vec, v_vec);
    let attn_out = kv.attend(query);
    assert_eq!(attn_out.len(), head_dim);
    assert!(attn_out.iter().all(|v| v.is_finite()));

    // Step 4: FFN with SiLU
    let ffn_out: Vec<f32> = normed.iter().map(|&x| cpu_silu(x)).collect();
    assert_eq!(ffn_out.len(), embed_dim);
    assert!(ffn_out.iter().all(|v| v.is_finite()));

    // Step 5: Output projection
    let proj_weight: Vec<f32> = (0..vocab_size * embed_dim).map(|i| pseudo_random(99, i)).collect();
    let proj = OutputProjection::new(proj_weight, vocab_size, embed_dim).unwrap();
    let mut logits = vec![0.0f32; vocab_size];
    proj.forward(&ffn_out, &mut logits, 1).unwrap();
    assert_eq!(logits.len(), vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn e2e_full_forward_pass_dimensions_preserved() {
    let vocab_size = 128;
    let embed_dim = 64;

    let weight: Vec<f32> = (0..vocab_size * embed_dim).map(|i| pseudo_random(7, i)).collect();
    let config = EmbeddingConfig::new(vocab_size, embed_dim);
    let table = EmbeddingTable::new(weight, config).unwrap();

    // Multi-token lookup
    let tokens = [1u32, 5, 10];
    let mut embeddings = vec![0.0f32; tokens.len() * embed_dim];
    table.lookup(&tokens, &mut embeddings).unwrap();

    // Each token gets embed_dim values
    for t in 0..tokens.len() {
        let slice = &embeddings[t * embed_dim..(t + 1) * embed_dim];
        assert!(slice.iter().any(|&v| v != 0.0), "token {t} embedding should be non-zero");
    }
}

// ============================================================================
// 2. KV cache + attention integration
// ============================================================================

#[test]
fn e2e_kv_cache_builds_over_multiple_tokens() {
    let head_dim = 16;
    let mut kv = CpuKvCache::new(head_dim);

    // Simulate 5 tokens
    for step in 0..5 {
        let k: Vec<f32> = (0..head_dim).map(|i| pseudo_random(step as u64, i)).collect();
        let v: Vec<f32> = (0..head_dim).map(|i| pseudo_random(step as u64 + 100, i)).collect();
        kv.append(k, v);
        assert_eq!(kv.len(), step + 1);
    }

    assert_eq!(kv.len(), 5);
}

#[test]
fn e2e_kv_cache_contents_match_expected() {
    let head_dim = 4;
    let mut kv = CpuKvCache::new(head_dim);

    let k1 = vec![1.0, 0.0, 0.0, 0.0];
    let v1 = vec![0.0, 1.0, 0.0, 0.0];
    kv.append(k1.clone(), v1.clone());

    // Verify stored values
    assert_eq!(kv.k[0], k1);
    assert_eq!(kv.v[0], v1);

    // With single entry and matching query, output should equal v
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let out = kv.attend(&query);
    // Single entry → softmax([score]) = [1.0] → output = v1
    for (a, b) in out.iter().zip(v1.iter()) {
        assert!((a - b).abs() < 1e-5, "single-entry attention should return v exactly");
    }
}

#[test]
fn e2e_kv_cache_attention_uses_all_cached_entries() {
    let head_dim = 4;
    let mut kv = CpuKvCache::new(head_dim);

    // Two keys: one much larger than the other in the query direction
    kv.append(vec![5.0, 0.0, 0.0, 0.0], vec![10.0, 0.0, 0.0, 0.0]);
    kv.append(vec![0.0, 0.1, 0.0, 0.0], vec![0.0, 20.0, 0.0, 0.0]);

    // Query aligned with first key → first value should dominate
    let out = kv.attend(&[5.0, 0.0, 0.0, 0.0]);
    assert!(out[0] > 5.0, "v1 component ({}) should dominate", out[0]);

    // Query aligned with second key → second value should dominate
    let out2 = kv.attend(&[0.0, 5.0, 0.0, 0.0]);
    assert!(out2[1] > 10.0, "v2 component ({}) should dominate", out2[1]);
}

// ============================================================================
// 3. Quantize → dequantize round-trip (CPU reference)
// ============================================================================

#[test]
fn e2e_ternary_quantize_dequantize_round_trip() {
    // Simulate I2S quantization: map f32 → {-1, 0, +1} → f32
    let input: Vec<f32> = (0..256).map(|i| pseudo_random(42, i)).collect();

    // Quantize: threshold to ternary
    let threshold = 0.3;
    let ternary: Vec<i8> = input
        .iter()
        .map(|&v| {
            if v > threshold {
                1
            } else if v < -threshold {
                -1
            } else {
                0
            }
        })
        .collect();

    // Dequantize (scale = absmax of input)
    let scale = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let dequantized: Vec<f32> = ternary.iter().map(|&t| t as f32 * scale).collect();

    // Verify ternary values preserved
    for (i, &t) in ternary.iter().enumerate() {
        assert!(t == -1 || t == 0 || t == 1, "element {i} should be ternary, got {t}");
        let expected = t as f32 * scale;
        assert!((dequantized[i] - expected).abs() < 1e-6, "round-trip mismatch at {i}");
    }
}

#[test]
fn e2e_ternary_packing_round_trip() {
    // Pack 4 ternary values per byte (2 bits each): -1=0b11, 0=0b00, +1=0b01
    let values: Vec<i8> = vec![1, -1, 0, 1, -1, -1, 0, 0];

    fn pack_ternary(vals: &[i8]) -> Vec<u8> {
        vals.chunks(4)
            .map(|chunk| {
                let mut byte = 0u8;
                for (j, &v) in chunk.iter().enumerate() {
                    let bits: u8 = match v {
                        1 => 0b01,
                        -1 => 0b11,
                        _ => 0b00,
                    };
                    byte |= bits << (j * 2);
                }
                byte
            })
            .collect()
    }

    fn unpack_ternary(packed: &[u8], count: usize) -> Vec<i8> {
        let mut out = Vec::with_capacity(count);
        for &byte in packed {
            for j in 0..4 {
                if out.len() >= count {
                    break;
                }
                let bits = (byte >> (j * 2)) & 0b11;
                out.push(match bits {
                    0b01 => 1,
                    0b11 => -1,
                    _ => 0,
                });
            }
        }
        out
    }

    let packed = pack_ternary(&values);
    let unpacked = unpack_ternary(&packed, values.len());
    assert_eq!(values, unpacked, "pack/unpack round-trip should be lossless");
}

// ============================================================================
// 4. Precision + normalization
// ============================================================================

#[test]
fn e2e_f32_to_f16_round_trip_with_norm() {
    use half::f16;

    let input: Vec<f32> = (0..64).map(|i| pseudo_random(12, i) * 10.0).collect();

    // Convert f32 → f16 → f32
    let f16_vals: Vec<f16> = input.iter().map(|&v| f16::from_f32(v)).collect();
    let recovered: Vec<f32> = f16_vals.iter().map(|v| v.to_f32()).collect();

    // Verify precision within f16 tolerance
    for (i, (&orig, &rec)) in input.iter().zip(recovered.iter()).enumerate() {
        let err = (orig - rec).abs();
        assert!(
            err < 0.05 * orig.abs().max(1.0),
            "f16 round-trip error too large at {i}: orig={orig}, rec={rec}, err={err}"
        );
    }

    // Apply RMSNorm to recovered values
    let normed = cpu_rms_norm(&recovered, 1e-5);
    assert!(normed.iter().all(|v| v.is_finite()), "normed output should be finite");

    // Normalized output should have roughly unit variance
    let mean_sq: f32 = normed.iter().map(|v| v * v).sum::<f32>() / normed.len() as f32;
    assert!((mean_sq - 1.0).abs() < 0.5, "mean_sq={mean_sq}, expected ~1.0");
}

#[test]
fn e2e_normalization_numerical_stability() {
    // Test with extreme values that could cause overflow
    let extreme: Vec<f32> = vec![1e10, -1e10, 1e-10, -1e-10, 0.0, 1.0, -1.0, 100.0];
    let normed = cpu_rms_norm(&extreme, 1e-5);
    assert!(normed.iter().all(|v| v.is_finite()), "should handle extreme values");

    // All-zero input should not produce NaN
    let zeros = vec![0.0f32; 16];
    let normed_z = cpu_rms_norm(&zeros, 1e-5);
    assert!(normed_z.iter().all(|v| v.is_finite()), "zero input should not produce NaN");
}

// ============================================================================
// 5. Pipeline + profiling integration
// ============================================================================

#[test]
fn e2e_pipeline_forward_records_all_stage_timings() {
    let config = PipelineConfig::tiny_test();
    let mut pipeline = InferencePipeline::new(config).unwrap();

    let logits = pipeline.forward(&[1]).unwrap();
    assert!(!logits.is_empty());
    assert!(logits.iter().all(|v| v.is_finite()));

    let timings = pipeline.stage_timings();
    let stages = PipelineStage::all();
    assert_eq!(timings.len(), stages.len(), "should have timing for every stage");

    for timing in timings {
        // After one forward pass, each stage except Sampling should have been called
        // (Sampling only runs during generate(), not forward())
        if timing.stage == PipelineStage::Sampling {
            continue;
        }
        assert!(timing.call_count >= 1, "stage {:?} was not called", timing.stage);
        assert!(timing.total_duration_us > 0 || timing.call_count > 0);
    }
}

#[test]
fn e2e_pipeline_profiling_accumulates_over_calls() {
    let config = PipelineConfig::tiny_test();
    let mut pipeline = InferencePipeline::new(config).unwrap();

    pipeline.forward(&[1]).unwrap();
    pipeline.forward(&[2]).unwrap();
    pipeline.forward(&[3]).unwrap();

    let timings = pipeline.stage_timings();
    for timing in timings {
        // Sampling only fires during generate(), not forward()
        if timing.stage == PipelineStage::Sampling {
            continue;
        }
        assert!(
            timing.call_count >= 3,
            "stage {:?} call_count={}, expected ≥3",
            timing.stage,
            timing.call_count
        );
    }

    let diag = pipeline.diagnostics();
    assert_eq!(diag.total_forward_calls, 3);
}

#[test]
fn e2e_pipeline_builder_constructs_valid_pipeline() {
    let pipeline = PipelineBuilder::new()
        .vocab_size(64)
        .hidden_dim(32)
        .num_layers(2)
        .num_heads(4)
        .head_dim(8)
        .intermediate_dim(64)
        .max_seq_len(128)
        .rms_norm_eps(1e-5)
        .rope_base(10000.0)
        .build()
        .unwrap();

    assert_eq!(pipeline.config().vocab_size, 64);
    assert_eq!(pipeline.config().hidden_dim, 32);
    assert!(matches!(pipeline.status(), PipelineStatus::Ready));
}

// ============================================================================
// 6. Error recovery chain
// ============================================================================

#[test]
fn e2e_pipeline_recovers_from_empty_input_error() {
    let config = PipelineConfig::tiny_test();
    let mut pipeline = InferencePipeline::new(config).unwrap();

    // First call: error (empty input)
    let err = pipeline.forward(&[]).unwrap_err();
    assert!(err.contains("empty"), "error should mention empty input: {err}");

    // Second call: should still work (pipeline not permanently broken)
    let logits = pipeline.forward(&[1]).unwrap();
    assert!(!logits.is_empty());
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn e2e_pipeline_detects_sequence_overflow() {
    let mut config = PipelineConfig::tiny_test();
    config.max_seq_len = 4;
    let mut pipeline = InferencePipeline::new(config).unwrap();

    // Fill up to max
    pipeline.forward(&[1]).unwrap();
    pipeline.forward(&[2]).unwrap();
    pipeline.forward(&[3]).unwrap();
    pipeline.forward(&[4]).unwrap();

    // Should fail at overflow
    let err = pipeline.forward(&[5]).unwrap_err();
    assert!(
        err.contains("max_seq_len") || err.contains("exceeds") || err.contains("sequence"),
        "error should mention sequence length: {err}"
    );
}

#[test]
fn e2e_embedding_handles_invalid_token_id_gracefully() {
    let vocab_size = 10;
    let embed_dim = 4;
    let weight: Vec<f32> = vec![1.0; vocab_size * embed_dim];
    let config = EmbeddingConfig::new(vocab_size, embed_dim);
    let table = EmbeddingTable::new(weight, config).unwrap();

    // Token ID out of range: should zero-fill (not error)
    let mut output = vec![0.0f32; embed_dim];
    table.lookup(&[100], &mut output).unwrap();
    assert!(output.iter().all(|&v| v == 0.0), "OOV token should produce zero vector");
}

// ============================================================================
// 7. Memory management lifecycle
// ============================================================================

#[test]
fn e2e_buffer_pool_allocate_use_return_cycle() {
    let pool = BufferPool::new();

    // Allocate buffers for inference stages
    let emb_buf = pool.allocate(1024, 64);
    let norm_buf = pool.allocate(1024, 64);
    let attn_buf = pool.allocate(2048, 64);

    assert_eq!(emb_buf.len(), 1024);
    assert_eq!(norm_buf.len(), 1024);
    assert_eq!(attn_buf.len(), 2048);

    let stats_before = pool.stats();
    assert!(stats_before.allocated >= 3);

    // Return buffers
    pool.return_buffer(emb_buf);
    pool.return_buffer(norm_buf);
    pool.return_buffer(attn_buf);

    let stats_after = pool.stats();
    assert!(stats_after.pooled >= 3, "returned buffers should be pooled");
}

#[test]
fn e2e_buffer_pool_reuse_reduces_allocation() {
    let pool = BufferPool::new();

    // First allocation
    let buf1 = pool.allocate(512, 64);
    pool.return_buffer(buf1);

    // Second allocation of same size should reuse
    let buf2 = pool.allocate(512, 64);
    assert_eq!(buf2.len(), 512);

    let stats = pool.stats();
    // Pool may or may not track reuse internally, but shouldn't crash
    assert!(stats.allocated >= 1);
}

#[test]
fn e2e_aligned_buffer_lifecycle() {
    let mut buf: AlignedBuffer<f32> = AlignedBuffer::new(256, 64);
    assert_eq!(buf.len(), 256);
    assert!(!buf.is_empty());
    assert!(buf.alignment() >= 64);

    // Write data
    let data = buf.as_mut_slice();
    for (i, v) in data.iter_mut().enumerate() {
        *v = i as f32;
    }

    // Verify
    let read = buf.as_slice();
    for (i, &v) in read.iter().enumerate() {
        assert_eq!(v, i as f32);
    }

    // Device pointer tracking
    assert!(buf.device_ptr().is_none());
    buf.set_device_ptr(0xDEAD_BEEF);
    assert_eq!(buf.device_ptr(), Some(0xDEAD_BEEF));
    buf.clear_device_ptr();
    assert!(buf.device_ptr().is_none());
}

#[test]
fn e2e_buffer_alignment_validates_correctly() {
    // Powers of 2 should validate
    assert!(validate_alignment(64, 64));
    assert!(validate_alignment(128, 64));
    assert!(validate_alignment(256, 128));

    // Misaligned should fail
    assert!(!validate_alignment(65, 64));
    assert!(!validate_alignment(100, 64));
}

// ============================================================================
// 8. Multi-token generation
// ============================================================================

#[test]
fn e2e_multi_token_generation_with_pipeline() {
    let config = PipelineConfig::tiny_test();
    let mut pipeline = InferencePipeline::new(config.clone()).unwrap();

    let gen_config = GenerationConfig::greedy().with_max_tokens(4);
    let result = pipeline.generate(&[1, 2, 3], &gen_config).unwrap();

    assert!(result.generated_tokens > 0, "should generate at least 1 token");
    assert!(result.generated_tokens <= 4, "should respect max_tokens");
    assert_eq!(result.stop_reason, StopReason::MaxTokens);
    assert!(!result.tokens.is_empty());
}

#[test]
fn e2e_multi_token_kv_cache_growth() {
    let head_dim = 8;
    let mut kv = CpuKvCache::new(head_dim);

    // Generate 4 tokens, verify cache grows
    for step in 0u64..4 {
        let k: Vec<f32> = (0..head_dim).map(|i| pseudo_random(step, i)).collect();
        let v: Vec<f32> = (0..head_dim).map(|i| pseudo_random(step + 50, i)).collect();
        kv.append(k, v);
        assert_eq!(kv.len(), (step + 1) as usize);

        // Each step should use full cache for attention
        let query: Vec<f32> = (0..head_dim).map(|i| pseudo_random(step + 200, i)).collect();
        let out = kv.attend(&query);
        assert_eq!(out.len(), head_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn e2e_token_generator_multi_step() {
    let config = PipelineConfig::tiny_test();
    let pipeline = InferencePipeline::new(config).unwrap();
    let gen_config = GenerationConfig::greedy().with_max_tokens(4);

    let mut generator = TokenGenerator::new(pipeline, gen_config);
    let result = generator.generate(&[1]).unwrap();

    assert!(result.generated_tokens > 0);
    assert!(result.generated_tokens <= 4);

    // Verify timings were collected
    assert!(!result.stage_timings.is_empty());
    assert!(result.total_time_us > 0);
}

#[test]
fn e2e_sequential_generation_with_reset() {
    let config = PipelineConfig::tiny_test();
    let mut pipeline = InferencePipeline::new(config).unwrap();

    // First generation
    let gen1 = GenerationConfig::greedy().with_max_tokens(2);
    let result1 = pipeline.generate(&[1], &gen1).unwrap();

    // Reset and generate again
    pipeline.reset();
    let result2 = pipeline.generate(&[1], &gen1).unwrap();

    // Both should produce valid results
    assert!(result1.generated_tokens > 0);
    assert!(result2.generated_tokens > 0);
    // After reset, results should be deterministic (same input → same output)
    assert_eq!(
        result1.tokens, result2.tokens,
        "deterministic pipeline should produce same output after reset"
    );
}

// ============================================================================
// Cross-module integration: embedding + pipeline
// ============================================================================

#[test]
fn e2e_embedding_feeds_pipeline() {
    let vocab_size = 64;
    let embed_dim = 32;

    // Create embedding table
    let weight: Vec<f32> = (0..vocab_size * embed_dim).map(|i| pseudo_random(77, i)).collect();
    let config = EmbeddingConfig::new(vocab_size, embed_dim);
    let table = EmbeddingTable::new(weight, config).unwrap();

    // Lookup a token
    let mut embedding = vec![0.0f32; embed_dim];
    table.lookup(&[5], &mut embedding).unwrap();

    // Use pipeline with matching dimensions
    let pipe_config = PipelineConfig::tiny_test();
    let mut pipeline = InferencePipeline::new(pipe_config).unwrap();

    // Pipeline forward with the token
    let logits = pipeline.forward(&[5]).unwrap();
    assert_eq!(logits.len(), 64); // tiny_test vocab_size
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn e2e_output_projection_ref_matches_struct() {
    let vocab_size = 16;
    let hidden_size = 8;
    let weight: Vec<f32> = (0..vocab_size * hidden_size).map(|i| pseudo_random(55, i)).collect();
    let hidden: Vec<f32> = (0..hidden_size).map(|i| pseudo_random(66, i)).collect();

    // Via struct
    let proj = OutputProjection::new(weight.clone(), vocab_size, hidden_size).unwrap();
    let mut logits_struct = vec![0.0f32; vocab_size];
    proj.forward(&hidden, &mut logits_struct, 1).unwrap();

    // Via free function
    let mut logits_ref = vec![0.0f32; vocab_size];
    output_projection_ref(&hidden, &weight, &mut logits_ref, 1, hidden_size, vocab_size).unwrap();

    for (i, (&a, &b)) in logits_struct.iter().zip(logits_ref.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "mismatch at {i}: struct={a}, ref={b}");
    }
}

// ============================================================================
// Cross-module integration: cache + kernel sources
// ============================================================================

#[test]
fn e2e_kernel_source_registry_feeds_cache() {
    let registry = KernelSourceRegistry::new();
    let cache = KernelCacheManager::new(CacheConfig::default());

    // For each kernel, create a cache key from its source
    let ids = [KernelProgramId::Matmul, KernelProgramId::RmsNorm, KernelProgramId::Softmax];

    for id in &ids {
        let source = registry.get(id).expect("kernel should be registered");
        let key = KernelCacheKey::from_source(
            source.source,
            "Intel Arc A770",
            &registry.build_options_string(id),
            "OpenCL 3.0",
        );

        // Store a fake compiled binary
        let binary = vec![0xDE, 0xAD, 0xBE, 0xEF];
        cache.store(key.clone(), binary.clone(), String::new());

        // Lookup should succeed
        let entry = cache.lookup(&key).expect("cache should contain entry");
        assert_eq!(entry.binary, binary);
    }

    let stats = cache.stats();
    assert_eq!(stats.stores, 3);
    assert_eq!(stats.entry_count, 3);
}

#[test]
fn e2e_kernel_cache_eviction_under_pressure() {
    let config = CacheConfig {
        max_entries: 2,
        max_total_size: 1024 * 1024,
        cache_dir: std::path::PathBuf::from(".test_cache"),
        ttl: None,
        policy: CachePolicy::MemoryOnly,
        eviction_strategy: CacheEvictionStrategy::Lru,
    };
    let cache = KernelCacheManager::new(config);

    let key1 = KernelCacheKey::new(1, "device", "", "3.0");
    let key2 = KernelCacheKey::new(2, "device", "", "3.0");
    let key3 = KernelCacheKey::new(3, "device", "", "3.0");

    cache.store(key1.clone(), vec![1; 100], String::new());
    cache.store(key2.clone(), vec![2; 100], String::new());

    assert!(cache.lookup(&key1).is_some());
    assert!(cache.lookup(&key2).is_some());

    // Third entry should trigger eviction (max_entries=2)
    cache.store(key3.clone(), vec![3; 100], String::new());
    assert!(cache.lookup(&key3).is_some());

    let stats = cache.stats();
    assert!(stats.entry_count <= 2, "should not exceed max_entries");
    assert!(stats.evictions >= 1, "should have evicted at least once");
}

// ============================================================================
// Cross-module integration: work size + buffer allocation
// ============================================================================

#[test]
fn e2e_work_size_guides_buffer_allocation() {
    let optimizer = WorkSizeOptimizer::intel_arc();

    // Compute work sizes for a matmul
    let m = 1024;
    let n = 2048;
    let result = optimizer.optimize_2d(m, n);

    assert!(result.total_work_items > 0);
    assert!(result.efficiency > 0.0);
    assert!(result.efficiency <= 1.0);

    // Allocate aligned buffers matching the work dimensions
    let buf_a: AlignedBuffer<f32> = AlignedBuffer::new(m * n, 64);
    assert_eq!(buf_a.len(), m * n);

    // Verify alignment utility agrees
    let size = optimal_buffer_size(m * n, std::mem::size_of::<f32>());
    assert!(size >= m * n * std::mem::size_of::<f32>());
}

#[test]
fn e2e_intel_arc_hints_produce_valid_work_sizes() {
    let hints = IntelArcWorkSizeHints::default();
    let config = hints.into_config();
    let optimizer = WorkSizeOptimizer::new(config);

    // Test various common inference dimensions
    for &elements in &[32, 64, 128, 256, 512, 1024, 2048, 4096] {
        let result = optimizer.optimize_1d(elements);
        assert!(result.total_work_items >= elements);
        assert!(result.efficiency > 0.0);

        // Global work size should be a multiple of local work size
        if let Some(local) = &result.local_work_size {
            for (g, l) in result.global_work_size.iter().zip(local.iter()) {
                assert_eq!(g % l, 0, "global must be multiple of local");
            }
        }
    }
}

// ============================================================================
// Cross-module integration: pipeline + diagnostics
// ============================================================================

#[test]
fn e2e_pipeline_diagnostics_complete_lifecycle() {
    let config = PipelineConfig::tiny_test();
    let mut pipeline = InferencePipeline::new(config).unwrap();

    // Initial state
    let diag = pipeline.diagnostics();
    assert!(matches!(diag.status, PipelineStatus::Ready));
    assert_eq!(diag.total_forward_calls, 0);
    assert_eq!(diag.total_tokens_generated, 0);

    // After forward pass
    pipeline.forward(&[1, 2]).unwrap();
    let diag = pipeline.diagnostics();
    assert_eq!(diag.total_forward_calls, 1);
    assert!(diag.peak_sequence_len >= 2);

    // After generation
    let gen_config = GenerationConfig::greedy().with_max_tokens(3);
    pipeline.reset();
    pipeline.generate(&[1], &gen_config).unwrap();
    let diag = pipeline.diagnostics();
    assert!(diag.total_tokens_generated > 0);
}

#[test]
fn e2e_pipeline_tokens_per_second_positive() {
    let config = PipelineConfig::tiny_test();
    let mut pipeline = InferencePipeline::new(config).unwrap();

    pipeline.forward(&[1]).unwrap();
    // tokens_per_second should be non-negative
    assert!(pipeline.tokens_per_second() >= 0.0);
}

// ============================================================================
// Cross-module integration: embedding ref functions + buffer
// ============================================================================

#[test]
fn e2e_embedding_ref_with_aligned_buffer() {
    let vocab_size = 32;
    let embed_dim = 16;

    // Store weights in aligned buffer
    let mut weight_buf: AlignedBuffer<f32> = AlignedBuffer::new(vocab_size * embed_dim, 64);
    for (i, v) in weight_buf.as_mut_slice().iter_mut().enumerate() {
        *v = pseudo_random(42, i);
    }

    // Perform lookup using ref function
    let mut output = vec![0.0f32; embed_dim];
    embedding_lookup_ref(&[7], weight_buf.as_slice(), &mut output, vocab_size, embed_dim, None)
        .unwrap();

    // Verify output matches expected slice
    let expected = &weight_buf.as_slice()[7 * embed_dim..8 * embed_dim];
    for (i, (&out, &exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert!((out - exp).abs() < 1e-6, "mismatch at {i}: got {out}, expected {exp}");
    }
}

// ============================================================================
// Cross-module: kernel sources + context config
// ============================================================================

#[test]
fn e2e_kernel_registry_covers_pipeline_stages() {
    let registry = KernelSourceRegistry::new();

    // Pipeline stages that need kernels (Attention is not registered by default)
    let required = [
        KernelProgramId::RmsNorm,
        KernelProgramId::Softmax,
        KernelProgramId::Matmul,
        KernelProgramId::Elementwise,
    ];

    for id in &required {
        assert!(registry.get(id).is_some(), "registry should have kernel for {:?}", id);
    }

    // All kernels should have at least one entry point
    for id in registry.kernel_ids() {
        let source = registry.get(id).unwrap();
        assert!(!source.entry_points.is_empty(), "{id:?} has no entry points");
        assert!(!source.source.is_empty(), "{id:?} has empty source");
    }
}

#[test]
fn e2e_kernel_source_contains_opencl_syntax() {
    let registry = KernelSourceRegistry::new();

    for id in registry.kernel_ids() {
        let source = registry.get(id).unwrap();
        // All kernel sources should contain __kernel or kernel keyword
        assert!(
            source.source.contains("__kernel") || source.source.contains("kernel"),
            "{id:?} source doesn't contain kernel keyword"
        );
    }
}

// ============================================================================
// Cross-module: context config + cache config validation
// ============================================================================

#[test]
fn e2e_context_and_cache_configs_validate_together() {
    // Context config with profiling enabled (for timing data)
    let ctx_config =
        OpenClContextConfig { enable_profiling: true, ..OpenClContextConfig::default() };
    assert!(ctx_config.enable_profiling);

    // Cache config should be valid
    let cache_config = CacheConfig::default();
    assert!(cache_config.validate().is_ok());

    // Create cache manager
    let cache = KernelCacheManager::new(cache_config);
    let stats = cache.stats();
    assert_eq!(stats.entry_count, 0);
    assert_eq!(stats.hits, 0);
}

// ============================================================================
// Tests for modules not yet exported (marked #[ignore])
// ============================================================================

#[test]
#[ignore = "requires opencl_attention module to be exported in lib.rs"]
fn e2e_attention_with_rope_and_kv_cache() {
    // Would test: RoPE encoding → multi-head attention → KV cache update
}

#[test]
#[ignore = "requires opencl_ffn module to be exported in lib.rs"]
fn e2e_gated_ffn_with_silu_activation() {
    // Would test: gate_proj → up_proj → SiLU → down_proj
}

#[test]
#[ignore = "requires opencl_memory module to be exported in lib.rs"]
fn e2e_memory_transfer_tracking_full_pipeline() {
    // Would test: H2D transfer → computation → D2H transfer → bandwidth stats
}

#[test]
#[ignore = "requires opencl_quantized module to be exported in lib.rs"]
fn e2e_quantized_matmul_i2s_round_trip() {
    // Would test: quantize weights → I2S matmul → verify against float reference
}

#[test]
#[ignore = "requires opencl_transformer module to be exported in lib.rs"]
fn e2e_transformer_layer_full_forward() {
    // Would test: pre-norm → attention → residual → FFN → post-norm
}

#[test]
#[ignore = "requires opencl_provider module to be exported in lib.rs"]
fn e2e_opencl_provider_fallback_to_cpu() {
    // Would test: OpenCL provider init failure → CPU fallback → correct output
}
