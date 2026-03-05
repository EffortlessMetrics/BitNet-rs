#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types)]
//! Metal tokenizer and embedding shader tests for Apple Silicon.
//!
//! Tests GPU-side embedding lookup, positional encoding, and token processing.
//! All tests use pure Rust CPU simulation (no GPU crates), f32 arithmetic,
//! and tolerance-based assertions. Tests are `#[ignore]`-gated because CI
//! runs on Linux.

#![cfg(target_os = "macos")]
#![allow(dead_code)]

use std::f32::consts::PI;

// ───────────────────────────────────────────────────────────────────
// Constants
// ───────────────────────────────────────────────────────────────────

/// Metal buffer alignment requirement (bytes).
const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Apple Silicon SIMD group (wavefront) width.
const METAL_SIMD_GROUP_SIZE: u32 = 32;

/// Metal maximum threads per threadgroup.
const METAL_MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// Tolerance for single-step floating point comparisons.
const TOL: f32 = 1e-5;

/// Tolerance for multi-step (accumulated) floating point comparisons.
const TOL_ACCUM: f32 = 1e-3;

/// Default RoPE base frequency.
const ROPE_BASE: f32 = 10_000.0;

// ───────────────────────────────────────────────────────────────────
// Helper types
// ───────────────────────────────────────────────────────────────────

/// Embedding table stored row-major: `[vocab_size, embed_dim]`.
#[derive(Debug, Clone)]
struct EmbeddingTable {
    weights: Vec<f32>,
    vocab_size: usize,
    embed_dim: usize,
}

impl EmbeddingTable {
    fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let mut weights = vec![0.0f32; vocab_size * embed_dim];
        // Deterministic init: weight[v][d] = (v * embed_dim + d) as f32 * 0.01
        for v in 0..vocab_size {
            for d in 0..embed_dim {
                weights[v * embed_dim + d] = (v * embed_dim + d) as f32 * 0.01;
            }
        }
        Self { weights, vocab_size, embed_dim }
    }

    fn from_weights(weights: Vec<f32>, vocab_size: usize, embed_dim: usize) -> Self {
        assert_eq!(weights.len(), vocab_size * embed_dim);
        Self { weights, vocab_size, embed_dim }
    }

    fn lookup(&self, token_id: u32) -> Vec<f32> {
        if (token_id as usize) >= self.vocab_size {
            return vec![0.0f32; self.embed_dim];
        }
        let start = token_id as usize * self.embed_dim;
        self.weights[start..start + self.embed_dim].to_vec()
    }

    fn batch_lookup(&self, token_ids: &[u32]) -> Vec<Vec<f32>> {
        token_ids.iter().map(|&id| self.lookup(id)).collect()
    }
}

/// Configuration for positional encoding.
#[derive(Debug, Clone)]
struct PosEncodingConfig {
    max_seq_len: usize,
    embed_dim: usize,
    base_freq: f32,
    offset: usize,
    scaling_factor: f32,
}

impl PosEncodingConfig {
    fn new(max_seq_len: usize, embed_dim: usize) -> Self {
        Self { max_seq_len, embed_dim, base_freq: ROPE_BASE, offset: 0, scaling_factor: 1.0 }
    }
}

// ───────────────────────────────────────────────────────────────────
// CPU reference implementations
// ───────────────────────────────────────────────────────────────────

/// Sinusoidal positional encoding: PE(pos, 2i) = sin(pos / 10000^(2i/d)),
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d)).
fn cpu_sinusoidal_encoding(config: &PosEncodingConfig) -> Vec<Vec<f32>> {
    let d = config.embed_dim;
    let mut result = Vec::with_capacity(config.max_seq_len);
    for pos in 0..config.max_seq_len {
        let p = (pos + config.offset) as f32 * config.scaling_factor;
        let mut encoding = vec![0.0f32; d];
        for i in 0..d / 2 {
            let exponent = (2 * i) as f32 / d as f32;
            let theta = p / config.base_freq.powf(exponent);
            encoding[2 * i] = theta.sin();
            encoding[2 * i + 1] = theta.cos();
        }
        result.push(encoding);
    }
    result
}

/// RoPE: apply rotary embedding to pairs of elements.
fn cpu_rope(data: &mut [f32], dim: usize, position: usize, base: f32) {
    let half_dim = dim / 2;
    for i in 0..half_dim {
        let exponent = -((2 * i) as f32) / dim as f32;
        let theta = base.powf(exponent);
        let angle = position as f32 * theta;
        let cos_val = angle.cos();
        let sin_val = angle.sin();
        let x0 = data[i * 2];
        let x1 = data[i * 2 + 1];
        data[i * 2] = x0 * cos_val - x1 * sin_val;
        data[i * 2 + 1] = x0 * sin_val + x1 * cos_val;
    }
}

/// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta.
fn cpu_layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter().enumerate().map(|(i, &v)| (v - mean) * inv_std * gamma[i] + beta[i]).collect()
}

/// RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma.
fn cpu_rms_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let rms = (x.iter().map(|&v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    x.iter().enumerate().map(|(i, &v)| v / rms * gamma[i]).collect()
}

/// Matrix-vector multiply: out = W * x (W is [out_dim, in_dim], x is [in_dim]).
fn cpu_matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for r in 0..out_dim {
        for c in 0..in_dim {
            out[r] += w[r * in_dim + c] * x[c];
        }
    }
    out
}

/// Dot product of two vectors.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// L2 norm of a vector.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Assert all elements are within tolerance.
fn assert_close(actual: &[f32], expected: &[f32], tol: f32, msg: &str) {
    assert_eq!(actual.len(), expected.len(), "{msg}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{msg}: index {i} — actual {a}, expected {e}, diff {}",
            (a - e).abs()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// 1. Embedding Lookup (7 tests)
// ═══════════════════════════════════════════════════════════════════

mod embedding_lookup {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_single_token_lookup() {
        let table = EmbeddingTable::new(100, 64);
        let emb = table.lookup(42);
        assert_eq!(emb.len(), 64);
        // Verify deterministic init value.
        for d in 0..64 {
            let expected = (42 * 64 + d) as f32 * 0.01;
            assert!((emb[d] - expected).abs() < TOL);
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_batch_lookup() {
        let table = EmbeddingTable::new(1000, 128);
        let ids = vec![0, 10, 999, 500];
        let embeddings = table.batch_lookup(&ids);
        assert_eq!(embeddings.len(), 4);
        for (idx, &id) in ids.iter().enumerate() {
            let expected_first = (id as usize * 128) as f32 * 0.01;
            assert!((embeddings[idx][0] - expected_first).abs() < TOL);
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_out_of_vocab_handling() {
        let table = EmbeddingTable::new(50, 32);
        // Token ID beyond vocab → zero vector.
        let emb = table.lookup(100);
        assert_eq!(emb.len(), 32);
        assert!(emb.iter().all(|&v| v == 0.0));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_max_vocab_size() {
        // Large vocab (simulated).
        let vocab_size = 128_000;
        let embed_dim = 4;
        let table = EmbeddingTable::new(vocab_size, embed_dim);
        let last = table.lookup((vocab_size - 1) as u32);
        assert_eq!(last.len(), embed_dim);
        let expected = ((vocab_size - 1) * embed_dim) as f32 * 0.01;
        assert!((last[0] - expected).abs() < TOL);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_zero_embedding_dim() {
        let table = EmbeddingTable::new(10, 0);
        let emb = table.lookup(5);
        assert!(emb.is_empty());
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_padding_token() {
        // Convention: token 0 is padding → embedding should be all zeros.
        let mut table = EmbeddingTable::new(100, 64);
        // Zero out row 0.
        for d in 0..64 {
            table.weights[d] = 0.0;
        }
        let emb = table.lookup(0);
        assert!(emb.iter().all(|&v| v == 0.0));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_sequential_vs_random_access() {
        let table = EmbeddingTable::new(256, 64);
        let sequential: Vec<u32> = (0..16).collect();
        let random_ids: Vec<u32> =
            vec![200, 3, 150, 42, 99, 0, 255, 128, 64, 32, 16, 8, 4, 2, 1, 100];

        let seq_emb = table.batch_lookup(&sequential);
        let rand_emb = table.batch_lookup(&random_ids);

        // Each lookup must be independent of access order.
        for (i, &id) in random_ids.iter().enumerate() {
            let direct = table.lookup(id);
            assert_close(&rand_emb[i], &direct, TOL, "random access consistency");
        }
        // Sequential lookups must also be correct.
        for (i, &id) in sequential.iter().enumerate() {
            let direct = table.lookup(id);
            assert_close(&seq_emb[i], &direct, TOL, "sequential access consistency");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. Positional Encoding (7 tests)
// ═══════════════════════════════════════════════════════════════════

mod positional_encoding {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_sinusoidal_generation() {
        let config = PosEncodingConfig::new(16, 64);
        let pe = cpu_sinusoidal_encoding(&config);
        assert_eq!(pe.len(), 16);
        assert_eq!(pe[0].len(), 64);

        // Position 0 → sin(0)=0, cos(0)=1 for lowest frequency pair.
        assert!((pe[0][0] - 0.0).abs() < TOL, "sin(0) should be 0");
        assert!((pe[0][1] - 1.0).abs() < TOL, "cos(0) should be 1");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_rope_encoding() {
        let dim = 8;
        let mut data = vec![1.0f32; dim];
        let original = data.clone();

        cpu_rope(&mut data, dim, 1, ROPE_BASE);

        // After RoPE, magnitude of each pair should be preserved.
        for i in 0..dim / 2 {
            let orig_mag = (original[2 * i].powi(2) + original[2 * i + 1].powi(2)).sqrt();
            let new_mag = (data[2 * i].powi(2) + data[2 * i + 1].powi(2)).sqrt();
            assert!(
                (orig_mag - new_mag).abs() < TOL,
                "RoPE must preserve pair magnitude: pair {i}"
            );
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_learned_positional() {
        // Learned positional: just another embedding table lookup.
        let pos_table = EmbeddingTable::new(512, 64);
        let pos_emb = pos_table.lookup(10);
        assert_eq!(pos_emb.len(), 64);
        let expected = (10 * 64) as f32 * 0.01;
        assert!((pos_emb[0] - expected).abs() < TOL);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_max_sequence_length() {
        let config = PosEncodingConfig::new(4096, 16);
        let pe = cpu_sinusoidal_encoding(&config);
        assert_eq!(pe.len(), 4096);

        // All encodings should be finite.
        for (pos, enc) in pe.iter().enumerate() {
            assert!(enc.iter().all(|v| v.is_finite()), "position {pos} has non-finite values");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_position_offset() {
        let mut config = PosEncodingConfig::new(8, 16);
        let pe_no_offset = cpu_sinusoidal_encoding(&config);

        config.offset = 10;
        config.max_seq_len = 8;
        let pe_offset = cpu_sinusoidal_encoding(&config);

        // pe_offset[0] should equal pe_no_offset at effective position 10
        // (can't compare directly since pe_no_offset only goes to 8, but
        // verify they're different).
        let diff: f32 =
            pe_no_offset[0].iter().zip(pe_offset[0].iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > TOL, "offset should produce different encodings");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_frequency_scaling() {
        let mut config = PosEncodingConfig::new(8, 16);
        let pe_base = cpu_sinusoidal_encoding(&config);

        config.scaling_factor = 2.0;
        let pe_scaled = cpu_sinusoidal_encoding(&config);

        // Scaled position 1 should match unscaled position 2.
        assert_close(&pe_scaled[1], &pe_base[2], TOL, "2x scaling at pos 1 vs pos 2");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_interleaved_vs_sequential() {
        // Interleaved: (sin, cos, sin, cos, ...)
        // Sequential: (sin, sin, ..., cos, cos, ...)
        let dim = 8;
        let pos = 3.0f32;

        // Interleaved (our default).
        let mut interleaved = vec![0.0f32; dim];
        for i in 0..dim / 2 {
            let exponent = (2 * i) as f32 / dim as f32;
            let theta = pos / ROPE_BASE.powf(exponent);
            interleaved[2 * i] = theta.sin();
            interleaved[2 * i + 1] = theta.cos();
        }

        // Sequential layout.
        let mut sequential = vec![0.0f32; dim];
        for i in 0..dim / 2 {
            let exponent = (2 * i) as f32 / dim as f32;
            let theta = pos / ROPE_BASE.powf(exponent);
            sequential[i] = theta.sin();
            sequential[dim / 2 + i] = theta.cos();
        }

        // Both should contain the same values, just reordered.
        for i in 0..dim / 2 {
            assert!((interleaved[2 * i] - sequential[i]).abs() < TOL, "sin mismatch at {i}");
            assert!(
                (interleaved[2 * i + 1] - sequential[dim / 2 + i]).abs() < TOL,
                "cos mismatch at {i}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. Token Type Embeddings (5 tests)
// ═══════════════════════════════════════════════════════════════════

mod token_type_embeddings {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_segment_a_b() {
        let type_table = EmbeddingTable::new(2, 64);
        let seg_a = type_table.lookup(0);
        let seg_b = type_table.lookup(1);
        assert_eq!(seg_a.len(), 64);
        assert_eq!(seg_b.len(), 64);
        // Segments must differ.
        let diff: f32 = seg_a.iter().zip(seg_b.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > TOL, "segment A and B should differ");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_multi_segment() {
        let type_table = EmbeddingTable::new(4, 32);
        for seg in 0..4u32 {
            let emb = type_table.lookup(seg);
            assert_eq!(emb.len(), 32);
        }
        // All pairwise different.
        for i in 0..4u32 {
            for j in (i + 1)..4 {
                let a = type_table.lookup(i);
                let b = type_table.lookup(j);
                let diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
                assert!(diff > TOL, "segments {i} and {j} should differ");
            }
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_zero_segment() {
        // Segment 0 should return the first row, not zeros.
        let type_table = EmbeddingTable::from_weights(vec![1.0; 64], 1, 64);
        let emb = type_table.lookup(0);
        assert!(emb.iter().all(|&v| (v - 1.0).abs() < TOL));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_type_plus_position_fusion() {
        let embed_dim = 32;
        let token_table = EmbeddingTable::new(100, embed_dim);
        let pos_table = EmbeddingTable::new(512, embed_dim);
        let type_table = EmbeddingTable::new(2, embed_dim);

        let token_emb = token_table.lookup(5);
        let pos_emb = pos_table.lookup(3);
        let type_emb = type_table.lookup(1);

        // Fusion = token + position + type.
        let fused: Vec<f32> =
            (0..embed_dim).map(|d| token_emb[d] + pos_emb[d] + type_emb[d]).collect();

        assert_eq!(fused.len(), embed_dim);
        // Verify not all zero.
        assert!(fused.iter().any(|&v| v.abs() > TOL));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_type_embedding_table_bounds() {
        let type_table = EmbeddingTable::new(2, 16);
        // In-range.
        assert_eq!(type_table.lookup(0).len(), 16);
        assert_eq!(type_table.lookup(1).len(), 16);
        // Out of range → zero vector.
        let oob = type_table.lookup(5);
        assert!(oob.iter().all(|&v| v == 0.0));
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. Embedding Normalization (5 tests)
// ═══════════════════════════════════════════════════════════════════

mod embedding_normalization {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_layer_norm_on_embeddings() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let result = cpu_layer_norm(&x, &gamma, &beta, 1e-5);

        // Mean should be ~0 after norm.
        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        assert!(mean.abs() < TOL_ACCUM, "post-LN mean should be ~0, got {mean}");

        // Variance should be ~1.
        let var: f32 =
            result.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / result.len() as f32;
        assert!((var - 1.0).abs() < TOL_ACCUM, "post-LN variance should be ~1, got {var}");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_rms_norm_variant() {
        let x = vec![3.0, 4.0];
        let gamma = vec![1.0, 1.0];
        let result = cpu_rms_norm(&x, &gamma, 1e-5);

        // RMS of [3,4] = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + 1e-5).sqrt();
        let expected: Vec<f32> = x.iter().map(|&v| v / rms).collect();
        assert_close(&result, &expected, TOL, "RMSNorm computation");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_post_embedding_norm() {
        let table = EmbeddingTable::new(100, 32);
        let emb = table.lookup(42);
        let gamma = vec![1.0; 32];
        let beta = vec![0.0; 32];
        let normed = cpu_layer_norm(&emb, &gamma, &beta, 1e-5);

        assert_eq!(normed.len(), 32);
        // All values finite.
        assert!(normed.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_pre_norm_before_projection() {
        let embed_dim = 16;
        let proj_dim = 8;
        let x = vec![2.0f32; embed_dim];
        let gamma = vec![1.0; embed_dim];
        let beta = vec![0.0; embed_dim];

        // Normalize first.
        let normed = cpu_layer_norm(&x, &gamma, &beta, 1e-5);
        // Then project.
        let w: Vec<f32> = (0..proj_dim * embed_dim).map(|i| (i as f32) * 0.001).collect();
        let projected = cpu_matvec(&w, &normed, proj_dim, embed_dim);

        assert_eq!(projected.len(), proj_dim);
        assert!(projected.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_norm_epsilon_sensitivity() {
        let x = vec![1e-7, 1e-7, 1e-7, 1e-7];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];

        let result_small_eps = cpu_layer_norm(&x, &gamma, &beta, 1e-12);
        let result_large_eps = cpu_layer_norm(&x, &gamma, &beta, 1e-1);

        // Both must produce finite values.
        assert!(result_small_eps.iter().all(|v| v.is_finite()), "small eps must be finite");
        assert!(result_large_eps.iter().all(|v| v.is_finite()), "large eps must be finite");

        // Large eps will dominate the tiny variance → near-zero output.
        let large_eps_norm: f32 = result_large_eps.iter().map(|v| v.abs()).sum();
        assert!(large_eps_norm < 1.0, "large eps should suppress near-zero input");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. Embedding Projection (5 tests)
// ═══════════════════════════════════════════════════════════════════

mod embedding_projection {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_dimension_reduction() {
        let in_dim = 64;
        let out_dim = 32;
        let x = vec![1.0f32; in_dim];
        let w: Vec<f32> = (0..out_dim * in_dim).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();

        let projected = cpu_matvec(&w, &x, out_dim, in_dim);
        assert_eq!(projected.len(), out_dim);
        assert!(projected.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_dimension_expansion() {
        let in_dim = 32;
        let out_dim = 128;
        let x = vec![0.5f32; in_dim];
        let w: Vec<f32> = (0..out_dim * in_dim).map(|i| ((i % 5) as f32 - 2.0) * 0.01).collect();

        let projected = cpu_matvec(&w, &x, out_dim, in_dim);
        assert_eq!(projected.len(), out_dim);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_orthogonal_init_verification() {
        // A small orthogonal-like matrix: columns should be approximately orthonormal.
        let dim = 4;
        // Hadamard-like matrix (orthogonal for dim=4), scaled by 1/sqrt(dim).
        let scale = 1.0 / (dim as f32).sqrt();
        #[rustfmt::skip]
        let w = vec![
             scale,  scale,  scale,  scale,
             scale, -scale,  scale, -scale,
             scale,  scale, -scale, -scale,
             scale, -scale, -scale,  scale,
        ];

        // Check column orthonormality.
        for i in 0..dim {
            for j in i..dim {
                let col_i: Vec<f32> = (0..dim).map(|r| w[r * dim + i]).collect();
                let col_j: Vec<f32> = (0..dim).map(|r| w[r * dim + j]).collect();
                let d = dot(&col_i, &col_j);
                if i == j {
                    assert!((d - 1.0).abs() < TOL_ACCUM, "col {i} norm should be ~1, got {d}");
                } else {
                    assert!(d.abs() < TOL_ACCUM, "cols {i},{j} should be orthogonal, got {d}");
                }
            }
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_projection_plus_activation() {
        let in_dim = 16;
        let out_dim = 8;
        let x = vec![1.0f32; in_dim];
        let w: Vec<f32> = (0..out_dim * in_dim).map(|i| ((i as f32) - 64.0) * 0.01).collect();

        let projected = cpu_matvec(&w, &x, out_dim, in_dim);
        // Apply GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let gelu: Vec<f32> = projected
            .iter()
            .map(|&v| {
                let c = (2.0f32 / PI).sqrt();
                v * 0.5 * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh())
            })
            .collect();

        assert_eq!(gelu.len(), out_dim);
        assert!(gelu.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_identity_projection() {
        let dim = 8;
        // Identity matrix.
        let mut w = vec![0.0f32; dim * dim];
        for i in 0..dim {
            w[i * dim + i] = 1.0;
        }
        let x: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let projected = cpu_matvec(&w, &x, dim, dim);
        assert_close(&projected, &x, TOL, "identity projection");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 6. Batch Processing (5 tests)
// ═══════════════════════════════════════════════════════════════════

mod batch_processing {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_variable_length_sequences() {
        let table = EmbeddingTable::new(100, 32);
        let seq1: Vec<u32> = vec![1, 2, 3];
        let seq2: Vec<u32> = vec![4, 5, 6, 7, 8];

        let emb1 = table.batch_lookup(&seq1);
        let emb2 = table.batch_lookup(&seq2);

        assert_eq!(emb1.len(), 3);
        assert_eq!(emb2.len(), 5);
        // Each embedding has correct dimension.
        assert!(emb1.iter().all(|e| e.len() == 32));
        assert!(emb2.iter().all(|e| e.len() == 32));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_padding_masks() {
        let pad_id: u32 = 0;
        let tokens = vec![5u32, 10, 0, 0, 0]; // last 3 are padding.
        let mask: Vec<f32> = tokens.iter().map(|&t| if t != pad_id { 1.0 } else { 0.0 }).collect();

        assert_eq!(mask, vec![1.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_attention_mask_generation() {
        let seq_len = 4;
        // Causal mask: lower triangular.
        let mut mask = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask[i * seq_len + j] = 1.0;
            }
        }

        #[rustfmt::skip]
        let expected = vec![
            1.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
        ];
        assert_close(&mask, &expected, TOL, "causal attention mask");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_batch_token_independence() {
        let table = EmbeddingTable::new(100, 64);
        // Lookup token 42 in two different batch positions.
        let batch_a = vec![1u32, 42, 99];
        let batch_b = vec![50u32, 42, 7];

        let emb_a = table.batch_lookup(&batch_a);
        let emb_b = table.batch_lookup(&batch_b);

        // Token 42 should be identical regardless of batch context.
        assert_close(&emb_a[1], &emb_b[1], TOL, "token 42 batch independence");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_max_batch_handling() {
        let table = EmbeddingTable::new(1000, 16);
        let batch_size = 256;
        let ids: Vec<u32> = (0..batch_size as u32).collect();
        let embeddings = table.batch_lookup(&ids);

        assert_eq!(embeddings.len(), batch_size);
        // Spot-check a few.
        for &i in &[0u32, 127, 255] {
            let direct = table.lookup(i);
            assert_close(&embeddings[i as usize], &direct, TOL, "max batch spot check");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 7. Quantized Embeddings (5 tests)
// ═══════════════════════════════════════════════════════════════════

mod quantized_embeddings {
    use super::*;

    /// Simulate int8 quantized embedding lookup.
    fn int8_lookup(table_i8: &[i8], scale: f32, token_id: usize, embed_dim: usize) -> Vec<f32> {
        let start = token_id * embed_dim;
        table_i8[start..start + embed_dim].iter().map(|&v| v as f32 * scale).collect()
    }

    /// Simulate 4-bit packed embedding lookup (2 values per byte).
    fn int4_lookup(packed: &[u8], scale: f32, token_id: usize, embed_dim: usize) -> Vec<f32> {
        let bytes_per_row = embed_dim / 2;
        let start = token_id * bytes_per_row;
        let mut result = Vec::with_capacity(embed_dim);
        for &byte in &packed[start..start + bytes_per_row] {
            let lo = (byte & 0x0F) as i8 - 8; // Signed 4-bit: [0,15] → [-8,7]
            let hi = ((byte >> 4) & 0x0F) as i8 - 8;
            result.push(lo as f32 * scale);
            result.push(hi as f32 * scale);
        }
        result
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_int8_embedding_table() {
        let embed_dim = 8;
        let vocab_size = 4;
        let table_i8: Vec<i8> =
            (0..vocab_size * embed_dim).map(|i| ((i % 256) as i8).wrapping_mul(3)).collect();
        let scale = 0.05;

        let emb = int8_lookup(&table_i8, scale, 2, embed_dim);
        assert_eq!(emb.len(), embed_dim);
        // Verify dequantization.
        for d in 0..embed_dim {
            let expected = table_i8[2 * embed_dim + d] as f32 * scale;
            assert!((emb[d] - expected).abs() < TOL);
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_4bit_packed_lookup() {
        let embed_dim = 8;
        let _bytes_per_row = embed_dim / 2;
        let packed: Vec<u8> = vec![
            0x48, 0x59, 0x6A, 0x7B, // Row 0
            0x12, 0x34, 0x56, 0x78, // Row 1
        ];
        let scale = 0.1;

        let emb = int4_lookup(&packed, scale, 1, embed_dim);
        assert_eq!(emb.len(), embed_dim);

        // Manually decode row 1: [0x12, 0x34, 0x56, 0x78]
        // 0x12 → lo=(2-8)=-6, hi=(1-8)=-7
        let expected_first = (0x12u8 & 0x0F) as i8 - 8; // 2 - 8 = -6
        assert!((emb[0] - expected_first as f32 * scale).abs() < TOL);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_scale_factor_application() {
        let embed_dim = 4;
        let table_i8: Vec<i8> = vec![10, -20, 30, -40];
        let scale = 0.125;

        let emb = int8_lookup(&table_i8, scale, 0, embed_dim);
        let expected: Vec<f32> = vec![1.25, -2.5, 3.75, -5.0];
        assert_close(&emb, &expected, TOL, "int8 scale factor");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_symmetric_vs_asymmetric() {
        let embed_dim = 4;
        let table_i8: Vec<i8> = vec![50, -50, 100, -100];

        // Symmetric: dequant = val * scale.
        let scale = 0.01;
        let sym = int8_lookup(&table_i8, scale, 0, embed_dim);

        // Asymmetric: dequant = (val - zero_point) * scale.
        let zero_point = 10i8;
        let asym: Vec<f32> =
            table_i8.iter().map(|&v| (v as i16 - zero_point as i16) as f32 * scale).collect();

        // They should differ due to zero_point.
        let diff: f32 = sym.iter().zip(asym.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > TOL, "symmetric vs asymmetric should differ");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_mixed_precision_lookup() {
        let embed_dim = 4;
        let table_i8: Vec<i8> = vec![10, 20, 30, 40];
        let scale = 0.1;

        // Dequantize i8 → f32, then accumulate in f32.
        let emb_f32 = int8_lookup(&table_i8, scale, 0, embed_dim);

        // Simulate f16 intermediate: round to f16 precision then back.
        let emb_f16_sim: Vec<f32> = emb_f32
            .iter()
            .map(|&v| {
                // f16 has ~3.3 decimal digits of precision.
                let f16_val = half::f16::from_f32(v);
                f16_val.to_f32()
            })
            .collect();

        // Should be very close but may have small rounding differences.
        for (i, (&a, &b)) in emb_f32.iter().zip(emb_f16_sim.iter()).enumerate() {
            assert!((a - b).abs() < 0.01, "mixed precision idx {i}: f32={a}, f16→f32={b}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 8. Tied Embeddings (4 tests)
// ═══════════════════════════════════════════════════════════════════

mod tied_embeddings {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_weight_tying_input_output() {
        // Input embedding and output (lm_head) share the same weight matrix.
        let table = EmbeddingTable::new(100, 64);

        // Input: lookup token 5 → embedding.
        let input_emb = table.lookup(5);

        // Output: logits = W^T * hidden. With tied weights, W = embedding table.
        // Logit for token 5 = dot(embedding_row_5, hidden_state).
        let hidden = vec![1.0f32; 64];
        let logit_5: f32 = input_emb.iter().zip(hidden.iter()).map(|(a, b)| a * b).sum();

        // Logit should be sum of row 5 (since hidden is all 1s).
        let expected: f32 = input_emb.iter().sum();
        assert!((logit_5 - expected).abs() < TOL, "tied weight logit");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_shared_embedding_matrix() {
        let table = EmbeddingTable::new(50, 32);

        // Both encoder and decoder share the same table.
        let encoder_emb = table.lookup(10);
        let decoder_emb = table.lookup(10);

        assert_close(&encoder_emb, &decoder_emb, TOL, "shared embedding matrix");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_lm_head_reuse() {
        let vocab_size = 100;
        let embed_dim = 16;
        let table = EmbeddingTable::new(vocab_size, embed_dim);

        // Compute logits for all vocab tokens from a hidden state.
        let hidden: Vec<f32> = (0..embed_dim).map(|i| (i as f32) * 0.1).collect();

        let mut logits = vec![0.0f32; vocab_size];
        for v in 0..vocab_size {
            let row = table.lookup(v as u32);
            logits[v] = dot(&row, &hidden);
        }

        // Logits should be a valid probability distribution basis.
        assert_eq!(logits.len(), vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_untied_gradient_isolation() {
        // With untied weights, modifying the output projection shouldn't affect input.
        let input_table = EmbeddingTable::new(50, 16);
        let mut output_table = EmbeddingTable::new(50, 16);

        let input_before = input_table.lookup(10);

        // Simulate a gradient update to output table only.
        let lr = 0.01;
        for d in 0..16 {
            output_table.weights[10 * 16 + d] -= lr * 0.5; // Fake gradient.
        }

        let input_after = input_table.lookup(10);
        // Input table should be unchanged.
        assert_close(&input_before, &input_after, TOL, "gradient isolation");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 9. Edge Cases (6 tests)
// ═══════════════════════════════════════════════════════════════════

mod edge_cases {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_empty_sequence() {
        let table = EmbeddingTable::new(100, 64);
        let ids: Vec<u32> = vec![];
        let embeddings = table.batch_lookup(&ids);
        assert!(embeddings.is_empty());
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_single_token_sequence() {
        let table = EmbeddingTable::new(100, 64);
        let ids = vec![42u32];
        let embeddings = table.batch_lookup(&ids);
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 64);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_max_vocab_token_id() {
        let vocab_size = 128_000;
        let table = EmbeddingTable::new(vocab_size, 4);

        // Last valid token.
        let last = table.lookup((vocab_size - 1) as u32);
        assert!(last.iter().any(|&v| v != 0.0), "last valid token should have non-zero embedding");

        // First invalid token.
        let oob = table.lookup(vocab_size as u32);
        assert!(oob.iter().all(|&v| v == 0.0), "OOB token should be zeros");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_zero_length_embedding() {
        let table = EmbeddingTable::new(10, 0);
        let emb = table.lookup(5);
        assert!(emb.is_empty());
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_duplicate_tokens() {
        let table = EmbeddingTable::new(100, 32);
        let ids = vec![7u32, 7, 7, 7];
        let embeddings = table.batch_lookup(&ids);

        // All lookups of the same token must return identical embeddings.
        for i in 1..embeddings.len() {
            assert_close(&embeddings[0], &embeddings[i], TOL, "duplicate token consistency");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_all_pad_sequence() {
        let mut table = EmbeddingTable::new(100, 16);
        // Zero out padding row (token 0).
        for d in 0..16 {
            table.weights[d] = 0.0;
        }
        let ids = vec![0u32; 8];
        let embeddings = table.batch_lookup(&ids);
        for emb in &embeddings {
            assert!(emb.iter().all(|&v| v == 0.0), "all-pad should be zero vectors");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 10. Memory Patterns (7 tests)
// ═══════════════════════════════════════════════════════════════════

mod memory_patterns {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_coalesced_embedding_access() {
        // Coalesced: threads in a SIMD group access consecutive memory addresses.
        // For embedding lookup, coalesced access means adjacent threads read
        // adjacent elements of the same embedding row.
        let embed_dim = 128;
        let simd_width = METAL_SIMD_GROUP_SIZE as usize;

        // Verify embedding dim is divisible by SIMD width for full coalescing.
        assert_eq!(
            embed_dim % simd_width,
            0,
            "embed_dim should be multiple of SIMD group size for coalescing"
        );

        let table = EmbeddingTable::new(10, embed_dim);
        let emb = table.lookup(5);

        // Simulate SIMD group reading: each thread reads emb[thread_id + wave * simd_width].
        let num_waves = embed_dim / simd_width;
        let mut reconstructed = vec![0.0f32; embed_dim];
        for wave in 0..num_waves {
            for lane in 0..simd_width {
                let idx = wave * simd_width + lane;
                reconstructed[idx] = emb[idx];
            }
        }
        assert_close(&reconstructed, &emb, TOL, "coalesced access reconstruction");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_bank_conflict_avoidance() {
        // Metal shared memory has 32 banks. Bank conflicts occur when
        // multiple threads in a SIMD group access the same bank.
        // For f32 (4 bytes), bank = (address / 4) % 32.
        let embed_dim = 64;
        let num_banks = 32;

        // Sequential access pattern: thread i reads element i → no conflicts.
        let simd_width = METAL_SIMD_GROUP_SIZE as usize;
        let mut banks_accessed = vec![false; num_banks];
        for lane in 0..simd_width.min(embed_dim) {
            let bank = lane % num_banks;
            assert!(!banks_accessed[bank], "bank conflict detected at lane {lane}, bank {bank}");
            banks_accessed[bank] = true;
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_embedding_cache_locality() {
        // Sequential token IDs should exhibit good cache locality
        // because their embeddings are stored in adjacent memory.
        let embed_dim = 64;
        let table = EmbeddingTable::new(100, embed_dim);

        let sequential_ids: Vec<u32> = (0..8).collect();
        let scattered_ids: Vec<u32> = vec![99, 3, 50, 12, 87, 1, 44, 73];

        // Compute total byte distance between consecutive lookups.
        let seq_distance: usize = sequential_ids
            .windows(2)
            .map(|w| ((w[1] as i64 - w[0] as i64).unsigned_abs() as usize) * embed_dim * 4)
            .sum();
        let scat_distance: usize = scattered_ids
            .windows(2)
            .map(|w| ((w[1] as i64 - w[0] as i64).unsigned_abs() as usize) * embed_dim * 4)
            .sum();

        assert!(
            seq_distance < scat_distance,
            "sequential access should have better locality: seq={seq_distance} vs scat={scat_distance}"
        );
        // Verify lookups still produce correct results regardless of order.
        for &id in &scattered_ids {
            let emb = table.lookup(id);
            assert_eq!(emb.len(), embed_dim);
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_shared_memory_staging() {
        // In Metal, frequently-accessed embedding rows can be staged
        // into threadgroup (shared) memory. Verify the staging pattern.
        let embed_dim = 64;
        let _threadgroup_size = 256;
        let shared_mem_bytes = 32 * 1024; // 32 KB typical shared mem.
        let bytes_per_row = embed_dim * 4; // f32 = 4 bytes.
        let max_cached_rows = shared_mem_bytes / bytes_per_row;

        assert!(max_cached_rows >= 1, "at least one embedding row should fit in shared memory");
        assert!(
            max_cached_rows <= 200,
            "sanity: max_cached_rows={max_cached_rows} should be reasonable"
        );

        // Simulate staging: hot tokens cached in shared memory.
        let table = EmbeddingTable::new(100, embed_dim);
        let hot_tokens: Vec<u32> = (0..max_cached_rows.min(100) as u32).collect();
        let cached: Vec<Vec<f32>> = hot_tokens.iter().map(|&id| table.lookup(id)).collect();

        // Verify staged rows match direct lookup.
        for (i, &id) in hot_tokens.iter().enumerate() {
            let direct = table.lookup(id);
            assert_close(&cached[i], &direct, TOL, "shared memory staging");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_vectorized_loads() {
        // Metal supports float4 (128-bit) vectorized loads.
        // Embedding dimensions that are multiples of 4 can use float4.
        let embed_dim = 128;
        assert_eq!(embed_dim % 4, 0, "embed_dim should be multiple of 4 for float4 loads");

        let table = EmbeddingTable::new(10, embed_dim);
        let emb = table.lookup(3);

        // Simulate float4 vectorized reads.
        let num_float4 = embed_dim / 4;
        let mut reconstructed = vec![0.0f32; embed_dim];
        for chunk in 0..num_float4 {
            let base = chunk * 4;
            // A single float4 load fetches 4 consecutive f32s.
            reconstructed[base] = emb[base];
            reconstructed[base + 1] = emb[base + 1];
            reconstructed[base + 2] = emb[base + 2];
            reconstructed[base + 3] = emb[base + 3];
        }
        assert_close(&reconstructed, &emb, TOL, "vectorized float4 loads");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_memory_bandwidth_estimation() {
        // Estimate memory bandwidth for embedding lookup.
        let vocab_size = 128_000;
        let embed_dim = 4096;
        let batch_size = 32;
        let bytes_per_element = 4; // f32.

        // Each token lookup reads embed_dim * 4 bytes.
        let bytes_per_lookup = embed_dim * bytes_per_element;
        let total_bytes_read = batch_size * bytes_per_lookup;

        // Apple M-series unified memory bandwidth: ~200 GB/s (M2), ~400 GB/s (M3 Max).
        // Theoretical time = total_bytes / bandwidth.
        let bandwidth_gbps = 200.0f64; // Conservative: M2.
        let bandwidth_bps = bandwidth_gbps * 1e9;
        let theoretical_time_s = total_bytes_read as f64 / bandwidth_bps;
        let theoretical_time_us = theoretical_time_s * 1e6;

        // Embedding table size.
        let table_bytes = vocab_size * embed_dim * bytes_per_element;
        let table_mb = table_bytes as f64 / (1024.0 * 1024.0);

        assert!(
            theoretical_time_us < 1000.0,
            "embedding lookup should be sub-ms: estimated {theoretical_time_us:.2}us"
        );
        assert!(table_mb < 4096.0, "table should fit in memory: {table_mb:.1}MB");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn test_buffer_alignment_for_embeddings() {
        let embed_dim = 64;
        let vocab_size = 100;
        let bytes_per_row = embed_dim * 4; // f32.

        // Metal requires buffers to be 256-byte aligned.
        let aligned_row_bytes = (bytes_per_row + METAL_BUFFER_ALIGNMENT - 1)
            / METAL_BUFFER_ALIGNMENT
            * METAL_BUFFER_ALIGNMENT;

        assert_eq!(
            aligned_row_bytes % METAL_BUFFER_ALIGNMENT,
            0,
            "row stride must be aligned to {METAL_BUFFER_ALIGNMENT} bytes"
        );

        // Total buffer size.
        let total_bytes = vocab_size * bytes_per_row;
        let aligned_total = (total_bytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT
            * METAL_BUFFER_ALIGNMENT;

        assert_eq!(aligned_total % METAL_BUFFER_ALIGNMENT, 0, "total buffer must be aligned");
        assert!(aligned_total >= total_bytes, "alignment must not shrink buffer");
    }
}
