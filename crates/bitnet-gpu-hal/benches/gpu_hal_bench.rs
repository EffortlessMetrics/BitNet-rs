#![cfg(feature = "bench")]
// Benchmarks intentionally use simple casts and naive math for clarity.
#![allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]

//! Criterion benchmarks for GPU HAL CPU reference implementations.
//!
//! These benchmarks establish baselines for softmax, RMS norm, matmul, `RoPE`,
//! and sampling operations. GPU kernel implementations should beat these.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");
    for size in [64, 256, 1024, 4096, 32000] {
        let logits: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 - 0.5).collect();
        group.bench_with_input(BenchmarkId::from_parameter(size), &logits, |b, logits| {
            b.iter(|| {
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
                let result: Vec<f32> = logits.iter().map(|&x| (x - max).exp() / exp_sum).collect();
                black_box(result)
            });
        });
    }
    group.finish();
}

fn bench_rms_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("rms_norm");
    for size in [64, 256, 1024, 4096] {
        let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let weight: Vec<f32> = vec![1.0; size];
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let ss: f32 = input.iter().map(|&x| x * x).sum::<f32>() / input.len() as f32;
                let rms = (ss + 1e-6).sqrt();
                let result: Vec<f32> =
                    input.iter().zip(weight.iter()).map(|(&x, &w)| (x / rms) * w).collect();
                black_box(result)
            });
        });
    }
    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_cpu");
    for size in [16, 32, 64, 128] {
        let a: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.001).collect();
        let b_mat: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.002).collect();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b_iter, &n| {
            b_iter.iter(|| {
                let mut c = vec![0.0f32; n * n];
                for i in 0..n {
                    for k in 0..n {
                        let a_ik = a[i * n + k];
                        for j in 0..n {
                            c[i * n + j] += a_ik * b_mat[k * n + j];
                        }
                    }
                }
                black_box(c)
            });
        });
    }
    group.finish();
}

fn bench_rope(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope");
    for head_dim in [64, 128] {
        let data: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.01).collect();
        group.bench_with_input(BenchmarkId::from_parameter(head_dim), &head_dim, |b, &dim| {
            let theta = 10000.0f32;
            b.iter(|| {
                let mut output = data.clone();
                let half = dim / 2;
                for i in 0..half {
                    let freq = 1.0 / theta.powf(2.0 * i as f32 / dim as f32);
                    let cos_val = freq.cos();
                    let sin_val = freq.sin();
                    let x0 = output[i];
                    let x1 = output[i + half];
                    output[i] = x0 * cos_val - x1 * sin_val;
                    output[i + half] = x0 * sin_val + x1 * cos_val;
                }
                black_box(output)
            });
        });
    }
    group.finish();
}

fn bench_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling");
    let logits: Vec<f32> = (0..32000).map(|i| (i as f32) * 0.0001 - 1.6).collect();

    group.bench_function("argmax_32k", |b| {
        b.iter(|| {
            let idx = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            black_box(idx)
        });
    });

    group.bench_function("top_k_50", |b| {
        b.iter(|| {
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(50);
            black_box(indexed)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_softmax, bench_rms_norm, bench_matmul, bench_rope, bench_sampling,);
criterion_main!(benches);
