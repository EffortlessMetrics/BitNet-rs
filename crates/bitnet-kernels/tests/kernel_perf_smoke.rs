//! CPU kernel performance smoke tests.
//!
//! These are **not** benchmarks — they verify that core kernel operations
//! complete in reasonable wall-clock time at production-relevant dimensions
//! (Phi-4 hidden size 5120, 100k vocab, 16K context).  A generous 10-second
//! ceiling catches catastrophic regressions without flaking on slow CI boxes.

use std::time::Instant;

use bitnet_kernels::cpu::{
    self,
    attention::scaled_dot_product_attention,
    embedding::embedding_lookup,
    layer_norm::{LayerNormConfig, layer_norm, rms_norm},
    rope::{RopeConfig, compute_frequencies},
    simd_matmul::{SimdMatmulConfig, simd_matmul_f32},
};

const MAX_SECS: u64 = 10;

// ── Helpers ────────────────────────────────────────────────────────────

fn ones(n: usize) -> Vec<f32> {
    vec![1.0f32; n]
}

fn ascending(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 * 0.001).collect()
}

fn assert_fast(label: &str, elapsed: std::time::Duration) {
    eprintln!("{label}: {elapsed:?}");
    assert!(elapsed.as_secs() < MAX_SECS, "{label} took too long: {elapsed:?}");
}

// ── Tests ──────────────────────────────────────────────────────────────

/// LayerNorm on a Phi-4 hidden-size vector (5120 dims).
#[test]
fn perf_smoke_layer_norm_5120() {
    let dim = 5120;
    let input = ascending(dim);
    let gamma = ones(dim);
    let beta = ones(dim);
    let config = LayerNormConfig::new(vec![dim]);

    let start = Instant::now();
    let _out = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
    assert_fast("layer_norm(5120)", start.elapsed());
}

/// RMSNorm on 5120-dim — the normalization actually used in LLaMA-family models.
#[test]
fn perf_smoke_rms_norm_5120() {
    let dim = 5120;
    let input = ascending(dim);
    let gamma = ones(dim);
    let config = LayerNormConfig::new(vec![dim]);

    let start = Instant::now();
    let _out = rms_norm(&input, &gamma, &config).unwrap();
    assert_fast("rms_norm(5120)", start.elapsed());
}

/// Softmax over a 100k-element vector (large vocabulary logits).
#[test]
fn perf_smoke_softmax_100k() {
    let n = 100_000;
    let input = ascending(n);

    let start = Instant::now();
    let _out = cpu::batched_softmax(&input, 1, n).unwrap();
    assert_fast("softmax(100k)", start.elapsed());
}

/// SiLU activation on 1M elements.
#[test]
fn perf_smoke_silu_1m() {
    let n = 1_000_000;
    let input = ascending(n);

    let start = Instant::now();
    let _out = cpu::silu_vec(&input);
    assert_fast("silu(1M)", start.elapsed());
}

/// ReLU activation (in-place) on 1M elements.
#[test]
fn perf_smoke_relu_1m() {
    let n = 1_000_000;
    let mut data = ascending(n);

    let start = Instant::now();
    cpu::relu_inplace(&mut data);
    assert_fast("relu(1M)", start.elapsed());
}

/// GeLU activation on 1M elements.
#[test]
fn perf_smoke_gelu_1m() {
    let n = 1_000_000;
    let input = ascending(n);

    let start = Instant::now();
    let _out = cpu::gelu_vec(&input);
    assert_fast("gelu(1M)", start.elapsed());
}

/// Embedding lookup on a 100k-entry table (dim=256), 512 lookups.
#[test]
fn perf_smoke_embedding_lookup_100k() {
    let vocab_size: usize = 100_000;
    let embedding_dim: usize = 256;
    let table: Vec<f32> =
        (0..vocab_size * embedding_dim).map(|i| (i % 1000) as f32 * 0.001).collect();
    let indices: Vec<u32> = (0..512).map(|i| (i * 193) as u32 % vocab_size as u32).collect();

    let start = Instant::now();
    let _out = embedding_lookup(&table, &indices, embedding_dim).unwrap();
    assert_fast("embedding_lookup(100k×256, 512 queries)", start.elapsed());
}

/// Scaled dot-product attention for seq_len=512, head_dim=128 (single head).
/// Attention is O(seq²), so we use a moderate sequence length that still
/// exercises the hot path without blowing the 10 s budget on slow CI.
#[test]
fn perf_smoke_attention_seq512() {
    let seq_len: usize = 512;
    let head_dim: usize = 128;
    let n = seq_len * head_dim;
    let q = ascending(n);
    let k = ascending(n);
    let v = ones(n);

    let start = Instant::now();
    let _out =
        scaled_dot_product_attention(&q, &k, &v, seq_len, seq_len, head_dim, true).unwrap();
    assert_fast("attention(seq=512, head_dim=128, causal)", start.elapsed());
}

/// RoPE frequency table generation for 16K sequence length, head_dim=128.
#[test]
fn perf_smoke_rope_table_16k() {
    let config = RopeConfig::new(128, 16_384);

    let start = Instant::now();
    let freqs = compute_frequencies(&config);
    assert_fast("rope_table(16K×128)", start.elapsed());
    // Sanity: table should have max_seq_len * head_dim entries.
    assert_eq!(freqs.len(), 16_384 * 128);
}

/// f32 GEMM: 512×512 × 512×512 — a reasonable projection-layer size.
#[test]
fn perf_smoke_matmul_512() {
    let m = 512;
    let k = 512;
    let n = 512;
    let a = ones(m * k);
    let b = ones(k * n);
    let mut c = vec![0.0f32; m * n];
    let cfg = SimdMatmulConfig {
        m,
        n,
        k,
        alpha: 1.0,
        beta: 0.0,
        transpose_a: false,
        transpose_b: false,
    };

    let start = Instant::now();
    simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
    assert_fast("matmul(512×512)", start.elapsed());
    // Sanity: each element should be k (sum of 1.0 * 1.0, k times).
    assert!((c[0] - k as f32).abs() < 1e-3, "matmul result incorrect: {}", c[0]);
}
