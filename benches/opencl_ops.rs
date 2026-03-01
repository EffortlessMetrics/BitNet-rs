//! Criterion benchmarks for A770 OpenCL kernel CPU-reference operations.
//!
//! Establishes baseline CPU performance for operations that will be
//! accelerated on Intel Arc A770 via OpenCL.  No OpenCL runtime required.
//!
//! Categories: dequantization, embedding lookup, normalization (RMSNorm /
//! LayerNorm), attention (softmax, RoPE), KV cache, and pipeline staging.

use bitnet_kernels::cpu::layer_norm::{self, LayerNormConfig};
use bitnet_kernels::opencl_attention::{AttentionScores, KVCacheEntry};
use bitnet_kernels::opencl_embedding::{EmbeddingConfig, EmbeddingNorm, EmbeddingTable};
use bitnet_kernels::opencl_pipeline::PipelineStage;
use bitnet_kernels::opencl_quantized::{I2sBlockLayout, I2sDequantizer, I2sPackedFormat};
use bitnet_rope::build_tables;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

// ── Constants ──────────────────────────────────────────────────────

const EMBED_DIM: usize = 2560;
const HIDDEN_DIM: usize = 2560;
const VOCAB_SIZE: usize = 32_000;
const HEAD_DIM: usize = 64;
const QK256_BLOCK: usize = 256;

// ── Helpers ────────────────────────────────────────────────────────

/// Deterministic f32 vector via xorshift.
fn make_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut state = if seed == 0 { 1u64 } else { seed };
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32) / (u64::MAX as f32) - 0.5
        })
        .collect()
}

/// Generate packed I2_S bytes for `n` ternary values.
fn make_packed_i2s(n: usize) -> Vec<u8> {
    let values: Vec<i8> = (0..n).map(|i| ((i % 3) as i8) - 1).collect();
    I2sPackedFormat::pack(&values)
}

/// Generate per-block scales for QK256 layout.
fn make_scales(n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements.div_ceil(QK256_BLOCK);
    (0..n_blocks).map(|i| 0.5 + 0.01 * i as f32).collect()
}

/// Create a weight table for embedding benchmarks.
fn make_embedding_weight() -> Vec<f32> {
    make_vec(VOCAB_SIZE * EMBED_DIM, 42)
}

// ── 1. Dequantization benchmarks ───────────────────────────────────

fn bench_dequantize_i2s(c: &mut Criterion) {
    let mut group = c.benchmark_group("dequantize_i2s");

    // 256-element block (single QK256 block)
    {
        let packed = make_packed_i2s(QK256_BLOCK);
        let scales = vec![1.0f32];
        group.bench_function("256", |b| {
            b.iter(|| {
                black_box(I2sDequantizer::dequantize_row(
                    black_box(&packed),
                    black_box(&scales),
                    QK256_BLOCK,
                    QK256_BLOCK,
                ))
            });
        });
    }

    // 4096-element row (full hidden dim)
    {
        let n = 4096usize;
        let packed = make_packed_i2s(n);
        let scales = make_scales(n);
        group.bench_function("4096", |b| {
            b.iter(|| {
                black_box(I2sDequantizer::dequantize_row(
                    black_box(&packed),
                    black_box(&scales),
                    n,
                    QK256_BLOCK,
                ))
            });
        });
    }

    // Batch: 32 rows × 256 columns
    {
        let rows = 32usize;
        let cols = QK256_BLOCK;
        let packed = make_packed_i2s(rows * cols);
        let n_blocks = cols.div_ceil(QK256_BLOCK);
        let scales: Vec<f32> = (0..rows * n_blocks).map(|i| 0.5 + 0.01 * i as f32).collect();
        group.bench_function("batch_32x256", |b| {
            b.iter(|| {
                black_box(I2sDequantizer::dequantize_matrix(
                    black_box(&packed),
                    black_box(&scales),
                    rows,
                    cols,
                    QK256_BLOCK,
                ))
            });
        });
    }

    group.finish();
}

// ── 2. Embedding benchmarks ────────────────────────────────────────

fn bench_embedding_lookup(c: &mut Criterion) {
    let weight = make_embedding_weight();
    let config = EmbeddingConfig::new(VOCAB_SIZE, EMBED_DIM);
    let table = EmbeddingTable::new(weight, config).unwrap();
    let mut group = c.benchmark_group("embedding_lookup");

    for &batch in &[1usize, 32, 128] {
        let label = match batch {
            1 => "single",
            32 => "batch_32",
            _ => "batch_128",
        };
        let tokens: Vec<u32> = (0..batch).map(|i| (i * 97 % VOCAB_SIZE) as u32).collect();

        group.bench_function(BenchmarkId::new("embed", label), |b| {
            b.iter_batched(
                || vec![0.0f32; batch * EMBED_DIM],
                |mut output| {
                    table.lookup(black_box(&tokens), &mut output).unwrap();
                    black_box(output)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ── 3. Normalization benchmarks ────────────────────────────────────

fn bench_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalization");

    let input = make_vec(HIDDEN_DIM, 77);
    let gamma = vec![1.0f32; HIDDEN_DIM];
    let beta = vec![0.0f32; HIDDEN_DIM];

    let config = LayerNormConfig::new(vec![HIDDEN_DIM]);

    // RMSNorm
    group.bench_function("rmsnorm_2560", |b| {
        b.iter(|| {
            black_box(layer_norm::rms_norm(black_box(&input), black_box(&gamma), &config).unwrap())
        });
    });

    // LayerNorm
    group.bench_function("layernorm_2560", |b| {
        b.iter(|| {
            black_box(
                layer_norm::layer_norm(
                    black_box(&input),
                    black_box(&gamma),
                    Some(black_box(&beta)),
                    &config,
                )
                .unwrap(),
            )
        });
    });

    // Embedding-specific RMSNorm via EmbeddingNorm
    {
        let norm = EmbeddingNorm::new(EMBED_DIM, 1e-5);
        group.bench_function("embedding_rmsnorm_2560", |b| {
            b.iter_batched(
                || make_vec(EMBED_DIM, 88),
                |mut data| {
                    norm.normalize(&mut data, 1).unwrap();
                    black_box(data)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ── 4. Attention benchmarks ────────────────────────────────────────

fn bench_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention");

    // Softmax over 128-length row
    group.bench_function("softmax_128", |b| {
        b.iter_batched(
            || {
                // Build raw scores for a 1×128 attention pattern
                let q = make_vec(HEAD_DIM, 10);
                let k = make_vec(128 * HEAD_DIM, 20);
                let scale = 1.0 / (HEAD_DIM as f32).sqrt();
                AttentionScores::compute_raw(&q, &k, 1, 128, HEAD_DIM, scale)
            },
            |mut scores| {
                scores.softmax();
                black_box(scores)
            },
            BatchSize::SmallInput,
        );
    });

    // Softmax over 2048-length row
    group.bench_function("softmax_2048", |b| {
        b.iter_batched(
            || {
                let q = make_vec(HEAD_DIM, 30);
                let k = make_vec(2048 * HEAD_DIM, 40);
                let scale = 1.0 / (HEAD_DIM as f32).sqrt();
                AttentionScores::compute_raw(&q, &k, 1, 2048, HEAD_DIM, scale)
            },
            |mut scores| {
                scores.softmax();
                black_box(scores)
            },
            BatchSize::SmallInput,
        );
    });

    // RoPE application on head_dim=64
    group.bench_function("rope_apply_64", |b| {
        let tables = build_tables(HEAD_DIM, 512, 10_000.0).unwrap();
        let input = make_vec(HEAD_DIM, 50);
        b.iter(|| {
            let mut output = vec![0.0f32; HEAD_DIM];
            let pos = 42usize;
            let half = tables.half_dim;
            // Apply RoPE rotation: (x0·cos − x1·sin, x0·sin + x1·cos)
            for i in 0..half {
                let cos_val = tables.cos[pos * half + i];
                let sin_val = tables.sin[pos * half + i];
                let x0 = input[i];
                let x1 = input[i + half];
                output[i] = x0 * cos_val - x1 * sin_val;
                output[i + half] = x0 * sin_val + x1 * cos_val;
            }
            black_box(output)
        });
    });

    group.finish();
}

// ── 5. KV cache benchmarks ────────────────────────────────────────

fn bench_kv_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache");

    // Append one token's K/V to cache
    group.bench_function("append_single", |b| {
        let k_tok = make_vec(HEAD_DIM, 60);
        let v_tok = make_vec(HEAD_DIM, 70);
        b.iter_batched(
            || KVCacheEntry::new(HEAD_DIM, 2048),
            |mut cache| {
                cache.append(black_box(&k_tok), black_box(&v_tok)).unwrap();
                black_box(cache)
            },
            BatchSize::SmallInput,
        );
    });

    // Gather 128 cached entries (fill then read keys/values)
    group.bench_function("gather_128", |b| {
        b.iter_batched(
            || {
                let mut cache = KVCacheEntry::new(HEAD_DIM, 2048);
                for t in 0..128usize {
                    let k: Vec<f32> = (0..HEAD_DIM).map(|d| (t * HEAD_DIM + d) as f32).collect();
                    let v: Vec<f32> =
                        (0..HEAD_DIM).map(|d| (t * HEAD_DIM + d) as f32 * 0.1).collect();
                    cache.append(&k, &v).unwrap();
                }
                cache
            },
            |cache| {
                let keys = black_box(cache.keys());
                let values = black_box(cache.values());
                black_box((keys.len(), values.len()))
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ── 6. Pipeline benchmarks ─────────────────────────────────────────

fn bench_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline");

    // Stage enumeration overhead
    group.bench_function("stage_ordering", |b| {
        b.iter(|| {
            let stages = PipelineStage::all();
            let mut count = 0usize;
            for stage in stages {
                // Force the compiler to touch each variant
                count += format!("{stage}").len();
            }
            black_box(count)
        });
    });

    // I2sBlockLayout selection
    group.bench_function("block_layout_select", |b| {
        let dims = [256usize, 512, 1024, 2048, 2560, 4096];
        b.iter(|| {
            let mut total = 0usize;
            for &d in black_box(&dims) {
                let layout = I2sBlockLayout::Qk256;
                total += layout.blocks_per_row(d);
            }
            black_box(total)
        });
    });

    group.finish();
}

// ── Criterion harness ──────────────────────────────────────────────

criterion_group!(
    benches,
    bench_dequantize_i2s,
    bench_embedding_lookup,
    bench_normalization,
    bench_attention,
    bench_kv_cache,
    bench_pipeline,
);
criterion_main!(benches);
