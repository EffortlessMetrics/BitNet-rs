//! Criterion benchmarks for dense model inference ops (cpu_opt).
//!
//! Benchmarks SiLU, RMSNorm, parallel_matmul, and parallel_attention
//! at representative dimensions for dense transformer architectures.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

use bitnet_inference::cpu_opt::{
    parallel_attention, parallel_matmul, rmsnorm, silu, silu_in_place,
};

// ---------------------------------------------------------------------------
// SiLU
// ---------------------------------------------------------------------------

fn bench_silu(c: &mut Criterion) {
    let mut group = c.benchmark_group("silu");

    for dim in [128, 1024, 5120, 13824] {
        let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001 - 0.5).collect();

        group.bench_with_input(BenchmarkId::new("allocating", dim), &input, |b, input| {
            b.iter(|| silu(black_box(input)))
        });

        let mut data = input.clone();
        group.bench_with_input(BenchmarkId::new("in_place", dim), &dim, |b, _| {
            b.iter(|| {
                data.copy_from_slice(&input);
                silu_in_place(black_box(&mut data));
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

fn bench_rmsnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmsnorm_cpu_opt");

    for (rows, dim) in [(1, 5120), (4, 5120), (16, 5120), (1, 1024), (1, 13824)] {
        let input: Vec<f32> = (0..rows * dim).map(|i| (i as f32) * 0.001).collect();
        let weight: Vec<f32> = vec![1.0; dim];
        let mut output = vec![0.0f32; rows * dim];

        let label = format!("{}x{}", rows, dim);
        group.bench_with_input(BenchmarkId::new("forward", &label), &label, |b, _| {
            b.iter(|| {
                rmsnorm(
                    black_box(&input),
                    black_box(&weight),
                    black_box(&mut output),
                    rows,
                    dim,
                    1e-5,
                )
                .unwrap()
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Matmul
// ---------------------------------------------------------------------------

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_cpu_opt");
    group.sample_size(10); // Large matrices are slow

    // (M, N, K) — representative dense model shapes
    let shapes = [
        (1, 128, 128),    // single token, small
        (1, 1024, 1024),  // single token, medium
        (4, 1024, 1024),  // short batch, medium
        (1, 5120, 5120),  // single token, Phi-4 hidden dim
        (1, 13824, 5120), // single token, Phi-4 FFN up-proj
    ];

    let num_threads = 4;

    for (m, n, k) in shapes {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();
        let mut out = vec![0.0f32; m * n];

        let label = format!("{}x{}x{}", m, n, k);
        group.bench_with_input(BenchmarkId::new("parallel", &label), &label, |bench, _| {
            bench.iter(|| {
                parallel_matmul(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut out),
                    m,
                    n,
                    k,
                    num_threads,
                )
                .unwrap()
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

fn bench_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_cpu_opt");
    group.sample_size(10);

    // (seq_len, head_dim, num_heads)
    let configs = [
        (16, 128, 1),  // short seq, single head
        (64, 128, 1),  // medium seq, single head
        (256, 128, 1), // long seq, single head
        (16, 128, 4),  // short seq, multi-head
        (64, 128, 4),  // medium seq, multi-head
    ];

    for (seq_len, head_dim, num_heads) in configs {
        let total = seq_len * head_dim * num_heads;
        let q: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
        let k = q.clone();
        let v: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01 + 0.5).collect();
        let mut out = vec![0.0f32; total];

        let label = format!("s{}_d{}_h{}", seq_len, head_dim, num_heads);
        group.bench_with_input(BenchmarkId::new("forward", &label), &label, |bench, _| {
            bench.iter(|| {
                parallel_attention(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    black_box(&mut out),
                    seq_len,
                    head_dim,
                    num_heads,
                )
                .unwrap()
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Pipeline: SiLU → RMSNorm (simulates dense FFN normalization)
// ---------------------------------------------------------------------------

fn bench_silu_rmsnorm_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("silu_rmsnorm_pipeline");

    for dim in [1024, 5120, 13824] {
        let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001 - 0.5).collect();
        let weight = vec![1.0f32; dim];
        let mut norm_out = vec![0.0f32; dim];

        group.bench_with_input(BenchmarkId::new("pipeline", dim), &dim, |b, _| {
            b.iter(|| {
                let activated = silu(black_box(&input));
                rmsnorm(
                    black_box(&activated),
                    black_box(&weight),
                    black_box(&mut norm_out),
                    1,
                    dim,
                    1e-5,
                )
                .unwrap();
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_silu,
    bench_rmsnorm,
    bench_matmul,
    bench_attention,
    bench_silu_rmsnorm_pipeline,
);
criterion_main!(benches);
