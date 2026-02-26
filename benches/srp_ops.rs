//! Criterion benchmarks for SRP microcrate operations:
//! logits pipeline, top-k selection, repetition penalty, argmax, RoPE table
//! generation, and KV-cache append.

use bitnet_logits::{
    apply_repetition_penalty, apply_temperature, apply_top_k, apply_top_p, argmax, softmax_in_place,
};
use bitnet_rope::build_tables;
use bitnet_transformer::LayerKVCache;
use candle_core::{DType, Device, Tensor};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

const N: usize = 1000;

fn make_logits() -> Vec<f32> {
    (0..N).map(|i| (i as f32) / (N as f32) - 0.5).collect()
}

/// End-to-end sampling pipeline: temperature → softmax → top-p
fn bench_logits_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("logits_pipeline");
    group.bench_function("temperature_softmax_top_p", |b| {
        b.iter_batched(
            make_logits,
            |mut logits| {
                apply_temperature(&mut logits, 0.8);
                softmax_in_place(&mut logits);
                apply_top_p(&mut logits, 0.9);
                black_box(logits)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Top-k at two settings: shows O(N) select_nth behaviour
fn bench_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_k");
    for k in [5usize, 50usize] {
        group.bench_with_input(BenchmarkId::new("apply_top_k", k), &k, |b, &k| {
            b.iter_batched(
                make_logits,
                |mut logits| black_box(apply_top_k(&mut logits, k)),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Repetition penalty applied to 1000-float logits with 100-token history
fn bench_repetition_penalty(c: &mut Criterion) {
    let history: Vec<u32> = (0..100_u32).collect();
    let mut group = c.benchmark_group("repetition_penalty");
    group.bench_function("1000_logits_100_tokens", |b| {
        b.iter_batched(
            make_logits,
            |mut logits| {
                apply_repetition_penalty(&mut logits, black_box(&history), 1.3);
                black_box(logits)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Argmax over 1000 floats — fast baseline
fn bench_argmax(c: &mut Criterion) {
    let logits = make_logits();
    c.bench_function("argmax_1000", |b| {
        b.iter(|| black_box(argmax(black_box(&logits))));
    });
}

/// RoPE table generation: dim=128, max_seq_len=512, base=10000.0
fn bench_rope_build_tables(c: &mut Criterion) {
    c.bench_function("build_rope_tables_128_512", |b| {
        b.iter(|| black_box(build_tables(black_box(128), black_box(512), black_box(10_000.0))));
    });
}

/// KV-cache single-token append: 4 layers × 8 heads × 512 capacity × 64 head_dim
fn bench_kv_cache_append(c: &mut Criterion) {
    let device = Device::Cpu;
    let n_kv_heads = 8usize;
    let head_dim = 64usize;

    // Single-token K/V tensors: [batch=1, n_kv_heads=8, seq=1, head_dim=64]
    let k_tok = Tensor::zeros(&[1usize, n_kv_heads, 1, head_dim], DType::F32, &device).unwrap();
    let v_tok = Tensor::zeros(&[1usize, n_kv_heads, 1, head_dim], DType::F32, &device).unwrap();

    c.bench_function("kv_cache_append_single_token", |b| {
        b.iter_batched(
            // Fresh empty cache each iteration (4 layers, 512-token capacity)
            || {
                let mut layers = Vec::with_capacity(4);
                for _ in 0..4 {
                    layers.push(LayerKVCache::new(1, n_kv_heads, 512, head_dim, &device).unwrap());
                }
                layers
            },
            |mut layers| {
                for layer in &mut layers {
                    layer.append(black_box(&k_tok), black_box(&v_tok)).unwrap();
                }
                black_box(layers)
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    bench_logits_pipeline,
    bench_top_k,
    bench_repetition_penalty,
    bench_argmax,
    bench_rope_build_tables,
    bench_kv_cache_append,
);
criterion_main!(benches);
