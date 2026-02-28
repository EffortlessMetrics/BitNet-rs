#![cfg(feature = "bench")]

//! CPU reference benchmarks for cross-backend comparison:
//! - Matrix multiplication at multiple sizes
//! - Softmax for various vocabulary sizes
//! - Scaled dot-product attention at different sequence lengths
//! - I2_S quantization roundtrip for various tensor sizes
//! - Complete transformer layer forward pass (RMSNorm → matmul → attention)

use bitnet_common::{BitNetTensor, Device, QuantizationType};
use bitnet_kernels::KernelManager;
use bitnet_quantization::I2SQuantizer;
use candle_core::{DType, Device as CandleDevice, Tensor};
use candle_nn::{Module, RmsNorm};
use criterion::{
    BenchmarkId, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime,
};
use std::hint::black_box;

// ── helpers ─────────────────────────────────────────────────────────────────

fn make_a(m: usize, k: usize) -> Vec<i8> {
    (0..m * k).map(|i| ((i % 3) as i8) - 1).collect()
}

fn make_b(k: usize, n: usize) -> Vec<u8> {
    (0..k * n).map(|i| (i % 256) as u8).collect()
}

fn generate_data(size: usize) -> Vec<f32> {
    (0..size).map(|i| ((i as f32 / size as f32) * 2.0 - 1.0) * 0.95).collect()
}

// ── matmul_cpu_reference ────────────────────────────────────────────────────

fn bench_matmul_cpu_reference(c: &mut Criterion) {
    let manager = KernelManager::new();
    let kernel = match manager.select_best() {
        Ok(k) => k,
        Err(_) => {
            eprintln!("No kernel provider available – skipping matmul benchmarks");
            return;
        }
    };

    let mut group: criterion::BenchmarkGroup<'_, WallTime> =
        c.benchmark_group("matmul_cpu_reference");

    let sizes: &[(usize, usize, usize)] = &[(64, 64, 64), (256, 256, 256), (1024, 1024, 1024)];

    for &(m, n, k) in sizes {
        let ops = m * n * k * 2;
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

// ── softmax_cpu_reference ───────────────────────────────────────────────────

fn bench_softmax_cpu_reference(c: &mut Criterion) {
    let device = CandleDevice::Cpu;
    let mut group: criterion::BenchmarkGroup<'_, WallTime> =
        c.benchmark_group("softmax_cpu_reference");

    let vocab_sizes: &[usize] = &[256, 32_000, 128_000];

    for &vocab in vocab_sizes {
        group.throughput(Throughput::Elements(vocab as u64));

        let logits = Tensor::randn(0.0f32, 1.0, &[1, vocab], &device).unwrap();

        group.bench_with_input(BenchmarkId::new("softmax", vocab), &vocab, |b, _| {
            b.iter(|| {
                let max = logits.max(1).unwrap();
                let shifted = logits.broadcast_sub(&max.unsqueeze(1).unwrap()).unwrap();
                let exp = shifted.exp().unwrap();
                let sum = exp.sum(1).unwrap();
                let result = exp.broadcast_div(&sum.unsqueeze(1).unwrap()).unwrap();
                black_box(result)
            });
        });
    }

    group.finish();
}

// ── attention_cpu_reference ─────────────────────────────────────────────────

fn bench_attention_cpu_reference(c: &mut Criterion) {
    let device = CandleDevice::Cpu;
    let mut group: criterion::BenchmarkGroup<'_, WallTime> =
        c.benchmark_group("attention_cpu_reference");

    // (n_heads, seq_len, head_dim)
    let configs: &[(usize, usize, usize)] = &[(4, 128, 64), (4, 512, 64), (4, 2048, 64)];

    for &(n_heads, seq_len, head_dim) in configs {
        let elems = n_heads * seq_len * head_dim;
        group.throughput(Throughput::Elements(elems as u64));

        let q = Tensor::randn(0.0f32, 1.0, &[1, n_heads, seq_len, head_dim], &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, &[1, n_heads, seq_len, head_dim], &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, &[1, n_heads, seq_len, head_dim], &device).unwrap();
        let k_t = k.transpose(2, 3).unwrap();
        let scale = (head_dim as f64).sqrt().recip();

        group.bench_with_input(
            BenchmarkId::new("scaled_dot_product", format!("h{n_heads}_s{seq_len}_d{head_dim}")),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    let scores = q.matmul(&k_t).unwrap();
                    let scaled = (scores * scale).unwrap();
                    // Softmax over last dimension
                    let max = scaled.max(3).unwrap();
                    let shifted = scaled.broadcast_sub(&max.unsqueeze(3).unwrap()).unwrap();
                    let exp = shifted.exp().unwrap();
                    let sum = exp.sum(3).unwrap();
                    let attn_weights = exp.broadcast_div(&sum.unsqueeze(3).unwrap()).unwrap();
                    let output = attn_weights.matmul(&v).unwrap();
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

// ── quantization_cpu_reference ──────────────────────────────────────────────

fn bench_quantization_cpu_reference(c: &mut Criterion) {
    let mut group: criterion::BenchmarkGroup<'_, WallTime> =
        c.benchmark_group("quantization_cpu_reference");

    let sizes: &[usize] = &[256, 1024, 4096, 16384];

    for &size in sizes {
        let data = generate_data(size);
        let tensor = BitNetTensor::from_slice(&data, &[size], &Device::Cpu).unwrap();
        let quantizer = I2SQuantizer::new();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("i2s_quantize", size), &size, |b, _| {
            b.iter(|| black_box(quantizer.quantize_tensor(&tensor).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("i2s_dequantize", size), &size, |b, _| {
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            b.iter(|| black_box(quantizer.dequantize_tensor(&quantized).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("i2s_roundtrip", size), &size, |b, _| {
            b.iter(|| {
                let q = quantizer.quantize_tensor(&tensor).unwrap();
                black_box(quantizer.dequantize_tensor(&q).unwrap())
            });
        });
    }

    group.finish();
}

// ── full_layer_cpu_reference ────────────────────────────────────────────────

fn bench_full_layer_cpu_reference(c: &mut Criterion) {
    let device = CandleDevice::Cpu;
    let eps = 1e-5;
    let mut group: criterion::BenchmarkGroup<'_, WallTime> =
        c.benchmark_group("full_layer_cpu_reference");

    // (hidden_dim, n_heads, seq_len)
    let configs: &[(usize, usize, usize)] = &[(256, 4, 64), (512, 8, 128), (1024, 8, 256)];

    for &(hidden_dim, n_heads, seq_len) in configs {
        let head_dim = hidden_dim / n_heads;
        let elems = seq_len * hidden_dim;
        group.throughput(Throughput::Elements(elems as u64));

        let gamma = Tensor::ones(&[hidden_dim], DType::F32, &device).unwrap();
        let norm = RmsNorm::new(gamma, eps);

        let input = Tensor::randn(0.0f32, 1.0, &[1, seq_len, hidden_dim], &device).unwrap();
        let wq = Tensor::randn(0.0f32, 0.01, &[hidden_dim, hidden_dim], &device).unwrap();

        group.bench_with_input(
            BenchmarkId::new("rmsnorm_proj_attn", format!("d{hidden_dim}_h{n_heads}_s{seq_len}")),
            &elems,
            |b, _| {
                b.iter(|| {
                    // RMSNorm
                    let normed = norm.forward(&input).unwrap();
                    // Linear projection (simulated Q)
                    let q = normed.matmul(&wq.t().unwrap()).unwrap();
                    // Reshape for multi-head attention
                    let q = q
                        .reshape(&[1, seq_len, n_heads, head_dim])
                        .unwrap()
                        .transpose(1, 2)
                        .unwrap();
                    // Self-attention score (Q·Qᵀ/√d)
                    let q_t = q.transpose(2, 3).unwrap();
                    let scale = (head_dim as f64).sqrt().recip();
                    let scores = q.matmul(&q_t).unwrap();
                    let scaled = (scores * scale).unwrap();
                    black_box(scaled)
                });
            },
        );
    }

    group.finish();
}

// ── kernel setup overhead ───────────────────────────────────────────────────

fn bench_kernel_setup_overhead(c: &mut Criterion) {
    let mut group: criterion::BenchmarkGroup<'_, WallTime> =
        c.benchmark_group("kernel_setup_overhead");

    group.bench_function("kernel_manager_creation", |b| {
        b.iter(|| black_box(KernelManager::new()));
    });

    group.bench_function("kernel_selection", |b| {
        let manager = KernelManager::new();
        b.iter(|| black_box(manager.select_best()));
    });

    group.bench_function("buffer_allocation_4k", |b| {
        b.iter(|| {
            let buf: Vec<f32> = vec![0.0; 4096];
            black_box(buf);
        });
    });

    group.bench_function("buffer_allocation_1m", |b| {
        b.iter(|| {
            let buf: Vec<f32> = vec![0.0; 1_048_576];
            black_box(buf);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_matmul_cpu_reference,
    bench_softmax_cpu_reference,
    bench_attention_cpu_reference,
    bench_quantization_cpu_reference,
    bench_full_layer_cpu_reference,
    bench_kernel_setup_overhead,
);
criterion_main!(benches);
