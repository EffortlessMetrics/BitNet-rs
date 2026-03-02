//! Criterion benchmarks for dense SLM (Small Language Model) inference
//! operations: SiLU, RMSNorm, dense matmul, GQA attention, RoPE,
//! softmax, dense transformer blocks, and token generation.

use bitnet_cpu_activations::{silu_inplace, silu_vec};
use bitnet_kernels::cpu::attention::{AttentionConfig, AttentionKernel, GqaConfig};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, rms_norm};
use bitnet_kernels::cpu::rope::{apply_rope, compute_frequencies, RopeConfig};
use bitnet_kernels::cuda::matmul::{MatmulConfig, matmul_tiled_cpu};
use bitnet_kernels::cuda::softmax::{SoftmaxConfig, softmax_cpu};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

// ── Helpers ──────────────────────────────────────────────────────────

fn rand_vec(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 * 0.7) % 2.0) - 1.0).collect()
}

fn ones_vec(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

// ── SiLU benchmarks ─────────────────────────────────────────────────

fn bench_silu(c: &mut Criterion) {
    let mut group = c.benchmark_group("silu");

    for &size in &[1024usize, 5120] {
        group.bench_with_input(BenchmarkId::new("silu_vec", size), &size, |b, &n| {
            let input = rand_vec(n);
            b.iter(|| black_box(silu_vec(black_box(&input))));
        });

        group.bench_with_input(BenchmarkId::new("silu_inplace", size), &size, |b, &n| {
            b.iter_batched(
                || rand_vec(n),
                |mut v| {
                    silu_inplace(&mut v);
                    black_box(v)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ── RMSNorm benchmarks ──────────────────────────────────────────────

fn bench_rmsnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmsnorm");

    for &size in &[1024usize, 5120] {
        group.bench_with_input(BenchmarkId::new("rms_norm", size), &size, |b, &n| {
            let input = rand_vec(n);
            let gamma = ones_vec(n);
            let config = LayerNormConfig::new(vec![n]);
            b.iter(|| black_box(rms_norm(black_box(&input), black_box(&gamma), &config).unwrap()));
        });
    }

    group.finish();
}

// ── Dense matmul benchmarks ─────────────────────────────────────────

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_matmul");
    group.sample_size(10);

    for &dim in &[512usize, 5120] {
        group.bench_with_input(BenchmarkId::new("matmul_tiled", dim), &dim, |b, &d| {
            let a = rand_vec(d * d);
            let b_mat = rand_vec(d * d);
            let config = MatmulConfig::for_shape(d, d, d).unwrap();
            b.iter_batched(
                || vec![0.0f32; d * d],
                |mut out| {
                    matmul_tiled_cpu(
                        black_box(&a),
                        black_box(&b_mat),
                        &mut out,
                        &config,
                    )
                    .unwrap();
                    black_box(out)
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

// ── Attention benchmarks ────────────────────────────────────────────

fn bench_attention_single_head(c: &mut Criterion) {
    let seq_len = 128;
    let head_dim = 128;
    let num_heads = 1;
    let total = seq_len * num_heads * head_dim;

    let q = rand_vec(total);
    let k = rand_vec(total);
    let v = rand_vec(total);

    let config =
        AttentionConfig { num_heads, head_dim, seq_len, causal: true, scale: None };

    c.bench_function("attention_single_head_128", |b| {
        b.iter(|| {
            black_box(
                AttentionKernel::multi_head_attention(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    &config,
                )
                .unwrap(),
            )
        });
    });
}

fn bench_attention_gqa_40_10(c: &mut Criterion) {
    let seq_len = 128;
    let head_dim = 128;
    let num_q_heads = 40;
    let num_kv_heads = 10;

    let q = rand_vec(seq_len * num_q_heads * head_dim);
    let k = rand_vec(seq_len * num_kv_heads * head_dim);
    let v = rand_vec(seq_len * num_kv_heads * head_dim);

    let config = GqaConfig {
        num_q_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        causal: true,
        scale: None,
    };

    let mut group = c.benchmark_group("attention_gqa");
    group.sample_size(10);
    group.bench_function("gqa_40_10_seq128", |b| {
        b.iter(|| {
            black_box(
                AttentionKernel::grouped_query_attention(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    &config,
                )
                .unwrap(),
            )
        });
    });
    group.finish();
}

// ── RoPE benchmark ──────────────────────────────────────────────────

fn bench_rope_128(c: &mut Criterion) {
    let head_dim = 128;
    let max_seq_len = 512;
    let config = RopeConfig::new(head_dim, max_seq_len);
    let freqs = compute_frequencies(&config);

    c.bench_function("rope_apply_head_dim_128", |b| {
        b.iter_batched(
            || rand_vec(head_dim),
            |mut data| {
                apply_rope(&mut data, black_box(42), head_dim, black_box(&freqs));
                black_box(data)
            },
            BatchSize::SmallInput,
        );
    });
}

// ── Softmax benchmark ───────────────────────────────────────────────

fn bench_softmax_128(c: &mut Criterion) {
    let n = 128;
    let input = rand_vec(n);
    let config = SoftmaxConfig::for_shape(n, 1).unwrap();

    c.bench_function("softmax_128", |b| {
        b.iter_batched(
            || vec![0.0f32; n],
            |mut out| {
                softmax_cpu(black_box(&input), &mut out, &config).unwrap();
                black_box(out)
            },
            BatchSize::SmallInput,
        );
    });
}

// ── Dense transformer block benchmarks ──────────────────────────────

/// Simulates one dense transformer block: RMSNorm → attention → residual
/// → RMSNorm → FFN (up-proj → SiLU → down-proj) → residual.
fn dense_block_forward(hidden: &[f32], hidden_size: usize) -> Vec<f32> {
    let gamma = vec![1.0f32; hidden_size];
    let norm_cfg = LayerNormConfig::new(vec![hidden_size]);

    // Pre-attention RMSNorm
    let normed = rms_norm(hidden, &gamma, &norm_cfg).unwrap();

    // Simulate attention as an identity-like operation (focus on overhead)
    let attn_out = normed.clone();

    // Residual connection
    let mut residual: Vec<f32> =
        hidden.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

    // Pre-FFN RMSNorm
    let normed2 = rms_norm(&residual, &gamma, &norm_cfg).unwrap();

    // FFN up-projection (simulate with SiLU activation)
    let ffn = silu_vec(&normed2);

    // FFN down-projection (identity for benchmarking the composition)
    // Add residual
    for (r, f) in residual.iter_mut().zip(ffn.iter()) {
        *r += f;
    }

    residual
}

fn bench_dense_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_block");

    for &hidden_size in &[256usize, 5120] {
        let label = if hidden_size == 256 { "small_256" } else { "phi4_5120" };
        group.bench_function(BenchmarkId::new("forward", label), |b| {
            let input = rand_vec(hidden_size);
            b.iter(|| black_box(dense_block_forward(black_box(&input), hidden_size)));
        });
    }

    group.finish();
}

// ── Token generation benchmark ──────────────────────────────────────

/// Simulates a single token generation cycle:
/// embedding lookup → dense block → RMSNorm → vocabulary projection (matmul)
/// → softmax → argmax.
fn bench_token_generation(c: &mut Criterion) {
    let hidden_size = 256;
    let vocab_size = 1024;

    let embedding = rand_vec(hidden_size);
    let gamma = ones_vec(hidden_size);
    let norm_cfg = LayerNormConfig::new(vec![hidden_size]);
    let proj_weights = rand_vec(hidden_size * vocab_size);
    let matmul_cfg = MatmulConfig::for_shape(1, vocab_size, hidden_size).unwrap();
    let softmax_cfg = SoftmaxConfig::for_shape(vocab_size, 1).unwrap();

    c.bench_function("token_generation_cycle", |b| {
        b.iter(|| {
            // Dense block
            let block_out = dense_block_forward(black_box(&embedding), hidden_size);

            // Final RMSNorm
            let normed = rms_norm(&block_out, &gamma, &norm_cfg).unwrap();

            // Vocabulary projection: [1, hidden] × [hidden, vocab] → [1, vocab]
            let mut logits = vec![0.0f32; vocab_size];
            matmul_tiled_cpu(&normed, &proj_weights, &mut logits, &matmul_cfg).unwrap();

            // Softmax
            let mut probs = vec![0.0f32; vocab_size];
            softmax_cpu(&logits, &mut probs, &softmax_cfg).unwrap();

            // Argmax
            let token = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            black_box(token)
        });
    });
}

criterion_group!(
    benches,
    bench_silu,
    bench_rmsnorm,
    bench_matmul,
    bench_attention_single_head,
    bench_attention_gqa_40_10,
    bench_rope_128,
    bench_softmax_128,
    bench_dense_block,
    bench_token_generation,
);
criterion_main!(benches);
