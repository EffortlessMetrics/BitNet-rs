#![cfg(feature = "bench")]

//! Benchmarks for core kernel operations:
//! - I2_S matmul at small / medium / large matrix sizes
//! - Scaled dot-product attention score computation (Q·Kᵀ / √d)
//! - RMSNorm forward pass (via candle-nn)

use bitnet_common::QuantizationType;
use bitnet_kernels::KernelManager;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, RmsNorm};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

// ── helpers ─────────────────────────────────────────────────────────────────

fn make_a(m: usize, k: usize) -> Vec<i8> {
    (0..m * k)
        .map(|i| ((i % 3) as i8) - 1) // values in {-1, 0, 1}
        .collect()
}

fn make_b(k: usize, n: usize) -> Vec<u8> {
    (0..k * n).map(|i| (i % 256) as u8).collect()
}

// ── matmul ──────────────────────────────────────────────────────────────────

fn bench_matmul(c: &mut Criterion) {
    let manager = KernelManager::new();
    let kernel = match manager.select_best() {
        Ok(k) => k,
        Err(_) => {
            eprintln!("No kernel provider available – skipping matmul benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("matmul_i2s_sizes");

    // (M, N, K) – representative of transformer layer shapes
    let sizes: &[(usize, usize, usize)] = &[
        (16, 16, 16),     // small
        (64, 64, 64),     // medium
        (128, 128, 128),  // large
        (256, 256, 256),  // very large
        (1, 256, 4096),   // GEMV-like (single token)
    ];

    for &(m, n, k) in sizes {
        let ops = m * n * k * 2; // 2 ops per multiply-accumulate
        group.throughput(Throughput::Elements(ops as u64));

        let a = make_a(m, k);
        let b = make_b(k, n);
        let mut out = vec![0.0f32; m * n];

        group.bench_with_input(
            BenchmarkId::new(kernel.name(), format!("{m}x{n}x{k}")),
            &(m, n, k),
            |bench, &(m, n, k)| {
                bench.iter(|| {
                    kernel
                        .matmul_i2s(black_box(&a), black_box(&b), black_box(&mut out), m, n, k)
                        .unwrap();
                    black_box(&out);
                });
            },
        );
    }

    group.finish();
}

// ── quantize via kernel provider ────────────────────────────────────────────

fn bench_kernel_quantize(c: &mut Criterion) {
    let manager = KernelManager::new();
    let kernel = match manager.select_best() {
        Ok(k) => k,
        Err(_) => {
            eprintln!("No kernel provider available – skipping quantize benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("kernel_quantize_i2s");

    for &size in &[256usize, 1024, 4096] {
        let input: Vec<f32> = (0..size)
            .map(|i| (i as f32 / size as f32) * 2.0 - 1.0)
            .collect();
        let mut output = vec![0u8; size / 4]; // 2 bits per element
        let mut scales = vec![0.0f32; size / 32]; // one scale per 32-element block

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("quantize", size), &size, |b, _| {
            b.iter(|| {
                kernel
                    .quantize(
                        black_box(&input),
                        black_box(&mut output),
                        black_box(&mut scales),
                        QuantizationType::I2S,
                    )
                    .unwrap();
                black_box(&output);
            });
        });
    }

    group.finish();
}

// ── attention scores (Q·Kᵀ / √d) ──────────────────────────────────────────

fn bench_attention_scores(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("attention_scores");

    // (batch, heads, seq_len, head_dim)
    let configs: &[(usize, usize, usize)] = &[
        (4, 32, 64),   // short context
        (4, 128, 64),  // medium context
        (4, 512, 64),  // long context
    ];

    for &(n_heads, seq_len, head_dim) in configs {
        let elems = n_heads * seq_len * head_dim;
        group.throughput(Throughput::Elements(elems as u64));

        let q = Tensor::randn(0.0f32, 1.0, &[1, n_heads, seq_len, head_dim], &device)
            .unwrap();
        let k = Tensor::randn(0.0f32, 1.0, &[1, n_heads, seq_len, head_dim], &device)
            .unwrap();
        let k_t = k.transpose(2, 3).unwrap();
        let scale = (head_dim as f64).sqrt().recip();

        group.bench_with_input(
            BenchmarkId::new("qk_matmul", format!("h{n_heads}_s{seq_len}_d{head_dim}")),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    let scores = q.matmul(&k_t).unwrap();
                    let scaled = (scores * scale).unwrap();
                    black_box(scaled)
                });
            },
        );
    }

    group.finish();
}

// ── RMSNorm ─────────────────────────────────────────────────────────────────

fn bench_rmsnorm(c: &mut Criterion) {
    let device = Device::Cpu;
    let eps = 1e-5;
    let mut group = c.benchmark_group("rmsnorm");

    for &hidden_dim in &[256usize, 1024, 4096] {
        group.throughput(Throughput::Elements(hidden_dim as u64));

        let gamma = Tensor::ones(&[hidden_dim], DType::F32, &device).unwrap();
        let norm = RmsNorm::new(gamma, eps);

        // Simulate a single-token hidden state [1, hidden_dim]
        let input = Tensor::randn(0.0f32, 1.0, &[1, hidden_dim], &device).unwrap();

        group.bench_with_input(
            BenchmarkId::new("forward", hidden_dim),
            &hidden_dim,
            |b, _| {
                b.iter(|| black_box(norm.forward(black_box(&input)).unwrap()));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_kernel_quantize,
    bench_attention_scores,
    bench_rmsnorm,
);
criterion_main!(benches);
