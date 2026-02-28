//! Criterion benchmarks for OpenCL kernel CPU reference implementations.
//!
//! These benchmarks exercise the pure-Rust CPU reference paths for operations
//! that would run as OpenCL kernels on GPU hardware. This validates the
//! reference implementations' performance characteristics and provides a
//! baseline for GPU kernel comparison.
//!
//! Gated behind the `bench` feature to avoid pulling in criterion for
//! normal builds.
#![cfg(feature = "bench")]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

// ── helpers ────────────────────────────────────────────────────────────────

const SIZES: [usize; 5] = [128, 256, 512, 1024, 2048];

/// Create a deterministic f32 vector of length `n`.
fn make_vec(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.001).sin()).collect()
}

/// Create a deterministic i8 ternary weight vector of length `n` (-1, 0, +1).
fn make_ternary_weights(n: usize) -> Vec<i8> {
    (0..n)
        .map(|i| match i % 3 {
            0 => -1i8,
            1 => 0i8,
            _ => 1i8,
        })
        .collect()
}

/// Pack ternary values (-1, 0, +1) into 2-bit packed bytes.
/// Each byte holds 4 ternary values: mapping -1→0b00, 0→0b01, +1→0b10.
fn ternary_pack(values: &[i8]) -> Vec<u8> {
    values
        .chunks(4)
        .map(|chunk| {
            let mut byte = 0u8;
            for (j, &v) in chunk.iter().enumerate() {
                let bits = match v {
                    -1 => 0b00u8,
                    0 => 0b01u8,
                    _ => 0b10u8,
                };
                byte |= bits << (j * 2);
            }
            byte
        })
        .collect()
}

/// Unpack 2-bit packed bytes back to ternary values.
fn ternary_unpack(packed: &[u8], count: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(count);
    for &byte in packed {
        for j in 0..4 {
            if out.len() >= count {
                break;
            }
            let bits = (byte >> (j * 2)) & 0b11;
            out.push(match bits {
                0b00 => -1i8,
                0b01 => 0i8,
                _ => 1i8,
            });
        }
    }
    out
}

// ── I2_S ternary matmul CPU reference ──────────────────────────────────────

/// CPU reference for I2_S ternary matrix–vector multiply.
///
/// Computes y = W·x where W is an M×K ternary matrix and x is K×1.
fn matmul_i2s_cpu(weights: &[i8], input: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; m];
    for row in 0..m {
        let mut acc = 0.0f32;
        for col in 0..k {
            acc += f32::from(weights[row * k + col]) * input[col];
        }
        output[row] = acc;
    }
    output
}

fn bench_matmul_i2s_cpu_ref(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_i2s_cpu");
    for &n in &SIZES {
        let weights = make_ternary_weights(n * n);
        let input = make_vec(n);
        group.bench_with_input(BenchmarkId::new("MxK", format!("{n}x{n}")), &n, |b, _| {
            b.iter(|| black_box(matmul_i2s_cpu(black_box(&weights), black_box(&input), n, n)));
        });
    }
    group.finish();
}

// ── RMSNorm CPU reference ──────────────────────────────────────────────────

/// CPU reference for RMSNorm: y_i = x_i * w_i / sqrt(mean(x²) + eps)
fn rmsnorm_cpu(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mean_sq: f32 = input.iter().map(|&x| x * x).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    input.iter().zip(weight.iter()).map(|(&x, &w)| x * w * inv_rms).collect()
}

fn bench_rmsnorm_cpu_ref(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmsnorm_cpu");
    for &n in &SIZES {
        let input = make_vec(n);
        let weight = make_vec(n);
        group.bench_with_input(BenchmarkId::new("dim", n), &n, |b, _| {
            b.iter(|| black_box(rmsnorm_cpu(black_box(&input), black_box(&weight), 1e-6)));
        });
    }
    group.finish();
}

// ── RoPE CPU reference ─────────────────────────────────────────────────────

/// CPU reference for Rotary Position Embedding (RoPE).
///
/// For each pair (x[2i], x[2i+1]), applies rotation by angle
/// pos * freq where freq = 1 / base^(2i/dim).
fn rope_cpu(input: &[f32], pos: usize, dim: usize, base: f32) -> Vec<f32> {
    let mut output = input.to_vec();
    for i in 0..(dim / 2) {
        let freq = 1.0 / base.powf(2.0 * i as f32 / dim as f32);
        let angle = pos as f32 * freq;
        let (sin_a, cos_a) = angle.sin_cos();
        let x0 = input[2 * i];
        let x1 = input[2 * i + 1];
        output[2 * i] = x0 * cos_a - x1 * sin_a;
        output[2 * i + 1] = x0 * sin_a + x1 * cos_a;
    }
    output
}

fn bench_rope_cpu_ref(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope_cpu");
    for &n in &SIZES {
        let input = make_vec(n);
        group.bench_with_input(BenchmarkId::new("dim", n), &n, |b, _| {
            b.iter(|| black_box(rope_cpu(black_box(&input), 42, n, 10000.0)));
        });
    }
    group.finish();
}

// ── Softmax CPU reference ──────────────────────────────────────────────────

/// CPU reference for row-wise softmax: softmax(x_i) = exp(x_i - max) / sum
fn softmax_cpu(input: &[f32]) -> Vec<f32> {
    let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn bench_softmax_cpu_ref(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_cpu");
    for &n in &SIZES {
        let input = make_vec(n);
        group.bench_with_input(BenchmarkId::new("len", n), &n, |b, _| {
            b.iter(|| black_box(softmax_cpu(black_box(&input))));
        });
    }
    group.finish();
}

// ── Scaled dot-product attention CPU reference ─────────────────────────────

/// CPU reference for single-head scaled dot-product attention.
///
/// attn = softmax(Q·Kᵀ / sqrt(d_k)) · V
/// Q: [1, d_k], K: [seq_len, d_k], V: [seq_len, d_v]
fn attention_cpu(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    seq_len: usize,
    d_k: usize,
) -> Vec<f32> {
    let scale = 1.0 / (d_k as f32).sqrt();

    // Compute Q·Kᵀ scores
    let mut scores = vec![0.0f32; seq_len];
    for s in 0..seq_len {
        let mut dot = 0.0f32;
        for d in 0..d_k {
            dot += query[d] * keys[s * d_k + d];
        }
        scores[s] = dot * scale;
    }

    // Softmax
    let weights = softmax_cpu(&scores);

    // Weighted sum of values
    let mut output = vec![0.0f32; d_k];
    for s in 0..seq_len {
        for d in 0..d_k {
            output[d] += weights[s] * values[s * d_k + d];
        }
    }
    output
}

fn bench_attention_cpu_ref(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_cpu");
    // Sweep sequence lengths with fixed head dim 64
    let head_dim = 64;
    for &seq_len in &SIZES {
        let query = make_vec(head_dim);
        let keys = make_vec(seq_len * head_dim);
        let values = make_vec(seq_len * head_dim);
        group.bench_with_input(BenchmarkId::new("seq_len", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                black_box(attention_cpu(
                    black_box(&query),
                    black_box(&keys),
                    black_box(&values),
                    seq_len,
                    head_dim,
                ))
            });
        });
    }
    group.finish();
}

// ── Ternary pack/unpack CPU reference ──────────────────────────────────────

fn bench_ternary_pack_unpack(c: &mut Criterion) {
    let mut group = c.benchmark_group("ternary_pack_unpack");
    for &n in &SIZES {
        let weights = make_ternary_weights(n * n);
        group.bench_with_input(BenchmarkId::new("pack", format!("{n}x{n}")), &n, |b, _| {
            b.iter(|| black_box(ternary_pack(black_box(&weights))));
        });
        let packed = ternary_pack(&weights);
        let count = n * n;
        group.bench_with_input(BenchmarkId::new("unpack", format!("{n}x{n}")), &n, |b, _| {
            b.iter(|| black_box(ternary_unpack(black_box(&packed), count)));
        });
    }
    group.finish();
}

criterion_group!(
    opencl_benches,
    bench_matmul_i2s_cpu_ref,
    bench_rmsnorm_cpu_ref,
    bench_rope_cpu_ref,
    bench_softmax_cpu_ref,
    bench_attention_cpu_ref,
    bench_ternary_pack_unpack,
);
criterion_main!(opencl_benches);
