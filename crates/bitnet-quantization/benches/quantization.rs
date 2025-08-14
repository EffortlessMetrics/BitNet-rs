//! Benchmarks for quantization algorithms
//!
//! These benchmarks compare the performance of different quantization implementations
//! and validate that the Rust implementation meets or exceeds the performance of
//! the Python baseline.

use bitnet_common::{BitNetTensor, QuantizationType};
use bitnet_quantization::{I2SQuantizer, Quantize, QuantizerTrait, TL1Quantizer, TL2Quantizer};
use candle_core::{Device, Tensor as CandleTensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Helper function to create benchmark tensors
fn create_benchmark_tensor(size: usize) -> BitNetTensor {
    let device = Device::Cpu;
    let data: Vec<f32> = (0..size)
        .map(|i| (i as f32 - size as f32 / 2.0) / (size as f32 / 4.0))
        .collect();
    let tensor = CandleTensor::from_vec(data, &[size], &device).unwrap();
    BitNetTensor::new(tensor)
}

/// Helper function to create 2D benchmark tensors
fn create_2d_benchmark_tensor(rows: usize, cols: usize) -> BitNetTensor {
    let device = Device::Cpu;
    let size = rows * cols;
    let data: Vec<f32> = (0..size)
        .map(|i| (i as f32 - size as f32 / 2.0) / (size as f32 / 4.0))
        .collect();
    let tensor = CandleTensor::from_vec(data, &[rows, cols], &device).unwrap();
    BitNetTensor::new(tensor)
}

/// Benchmark quantization performance for different tensor sizes
fn bench_quantization_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_sizes");

    let sizes = vec![1024, 4096, 16384, 65536, 262144];

    for size in sizes {
        let tensor = create_benchmark_tensor(size);

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark I2_S quantization
        group.bench_with_input(BenchmarkId::new("I2S_quantize", size), &size, |b, _| {
            let quantizer = I2SQuantizer::new();
            b.iter(|| black_box(quantizer.quantize_tensor(black_box(&tensor)).unwrap()))
        });

        // Benchmark TL1 quantization
        group.bench_with_input(BenchmarkId::new("TL1_quantize", size), &size, |b, _| {
            let quantizer = TL1Quantizer::new();
            b.iter(|| black_box(quantizer.quantize_tensor(black_box(&tensor)).unwrap()))
        });

        // Benchmark TL2 quantization
        group.bench_with_input(BenchmarkId::new("TL2_quantize", size), &size, |b, _| {
            let quantizer = TL2Quantizer::new();
            b.iter(|| black_box(quantizer.quantize_tensor(black_box(&tensor)).unwrap()))
        });
    }

    group.finish();
}

/// Benchmark dequantization performance
fn bench_dequantization_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("dequantization_sizes");

    let sizes = vec![1024, 4096, 16384, 65536, 262144];

    for size in sizes {
        let tensor = create_benchmark_tensor(size);

        group.throughput(Throughput::Elements(size as u64));

        // Pre-quantize tensors for dequantization benchmarks
        let i2s_quantizer = I2SQuantizer::new();
        let tl1_quantizer = TL1Quantizer::new();
        let tl2_quantizer = TL2Quantizer::new();

        let i2s_quantized = i2s_quantizer.quantize_tensor(&tensor).unwrap();
        let tl1_quantized = tl1_quantizer.quantize_tensor(&tensor).unwrap();
        let tl2_quantized = tl2_quantizer.quantize_tensor(&tensor).unwrap();

        // Benchmark I2_S dequantization
        group.bench_with_input(BenchmarkId::new("I2S_dequantize", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    i2s_quantizer
                        .dequantize_tensor(black_box(&i2s_quantized))
                        .unwrap(),
                )
            })
        });

        // Benchmark TL1 dequantization
        group.bench_with_input(BenchmarkId::new("TL1_dequantize", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    tl1_quantizer
                        .dequantize_tensor(black_box(&tl1_quantized))
                        .unwrap(),
                )
            })
        });

        // Benchmark TL2 dequantization
        group.bench_with_input(BenchmarkId::new("TL2_dequantize", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    tl2_quantizer
                        .dequantize_tensor(black_box(&tl2_quantized))
                        .unwrap(),
                )
            })
        });
    }

    group.finish();
}

/// Benchmark round-trip quantization (quantize + dequantize)
fn bench_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("round_trip");

    let sizes = vec![4096, 16384, 65536];

    for size in sizes {
        let tensor = create_benchmark_tensor(size);

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark I2_S round-trip
        group.bench_with_input(BenchmarkId::new("I2S_round_trip", size), &size, |b, _| {
            let quantizer = I2SQuantizer::new();
            b.iter(|| {
                let quantized = quantizer.quantize_tensor(black_box(&tensor)).unwrap();
                black_box(quantizer.dequantize_tensor(&quantized).unwrap())
            })
        });

        // Benchmark TL1 round-trip
        group.bench_with_input(BenchmarkId::new("TL1_round_trip", size), &size, |b, _| {
            let quantizer = TL1Quantizer::new();
            b.iter(|| {
                let quantized = quantizer.quantize_tensor(black_box(&tensor)).unwrap();
                black_box(quantizer.dequantize_tensor(&quantized).unwrap())
            })
        });

        // Benchmark TL2 round-trip
        group.bench_with_input(BenchmarkId::new("TL2_round_trip", size), &size, |b, _| {
            let quantizer = TL2Quantizer::new();
            b.iter(|| {
                let quantized = quantizer.quantize_tensor(black_box(&tensor)).unwrap();
                black_box(quantizer.dequantize_tensor(&quantized).unwrap())
            })
        });
    }

    group.finish();
}

/// Benchmark different block sizes for I2_S quantization
fn bench_block_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_sizes");

    let tensor = create_benchmark_tensor(65536);
    let block_sizes = vec![16, 32, 64, 128, 256];

    group.throughput(Throughput::Elements(65536));

    for block_size in block_sizes {
        group.bench_with_input(
            BenchmarkId::new("I2S_block_size", block_size),
            &block_size,
            |b, &block_size| {
                let quantizer = I2SQuantizer::with_block_size(block_size);
                b.iter(|| black_box(quantizer.quantize_tensor(black_box(&tensor)).unwrap()))
            },
        );
    }

    group.finish();
}

/// Benchmark 2D tensor quantization (matrix-like workloads)
fn bench_2d_tensors(c: &mut Criterion) {
    let mut group = c.benchmark_group("2d_tensors");

    let tensor_configs = vec![
        (256, 256),   // 64K elements
        (512, 512),   // 256K elements
        (1024, 1024), // 1M elements
    ];

    for (rows, cols) in tensor_configs {
        let tensor = create_2d_benchmark_tensor(rows, cols);
        let size = rows * cols;

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark I2_S on 2D tensors
        group.bench_with_input(
            BenchmarkId::new("I2S_2d", format!("{}x{}", rows, cols)),
            &size,
            |b, _| {
                let quantizer = I2SQuantizer::new();
                b.iter(|| black_box(quantizer.quantize_tensor(black_box(&tensor)).unwrap()))
            },
        );

        // Benchmark TL2 on 2D tensors (optimized for x86)
        group.bench_with_input(
            BenchmarkId::new("TL2_2d", format!("{}x{}", rows, cols)),
            &size,
            |b, _| {
                let quantizer = TL2Quantizer::new();
                b.iter(|| black_box(quantizer.quantize_tensor(black_box(&tensor)).unwrap()))
            },
        );
    }

    group.finish();
}

/// Benchmark trait-based quantization (dynamic dispatch)
fn bench_trait_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("trait_quantization");

    let tensor = create_benchmark_tensor(16384);

    group.throughput(Throughput::Elements(16384));

    // Benchmark using trait objects (dynamic dispatch)
    let quantizers: Vec<Box<dyn QuantizerTrait>> = vec![
        Box::new(I2SQuantizer::new()),
        Box::new(TL1Quantizer::new()),
        Box::new(TL2Quantizer::new()),
    ];

    for (i, quantizer) in quantizers.iter().enumerate() {
        let name = match i {
            0 => "I2S_trait",
            1 => "TL1_trait",
            2 => "TL2_trait",
            _ => unreachable!(),
        };

        group.bench_function(name, |b| {
            b.iter(|| black_box(quantizer.quantize_tensor(black_box(&tensor)).unwrap()))
        });
    }

    // Benchmark using Quantize trait on tensor directly
    group.bench_function("tensor_quantize_I2S", |b| {
        b.iter(|| black_box(tensor.quantize(QuantizationType::I2S).unwrap()))
    });

    group.bench_function("tensor_quantize_TL1", |b| {
        b.iter(|| black_box(tensor.quantize(QuantizationType::TL1).unwrap()))
    });

    group.bench_function("tensor_quantize_TL2", |b| {
        b.iter(|| black_box(tensor.quantize(QuantizationType::TL2).unwrap()))
    });

    group.finish();
}

/// Benchmark memory usage and compression ratios
fn bench_compression_ratios(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_ratios");

    let sizes = vec![4096, 16384, 65536];

    for size in sizes {
        let tensor = create_benchmark_tensor(size);

        // Measure compression ratios (not timing, but useful for analysis)
        group.bench_with_input(
            BenchmarkId::new("compression_analysis", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let i2s_quantized = tensor.quantize(QuantizationType::I2S).unwrap();
                    let tl1_quantized = tensor.quantize(QuantizationType::TL1).unwrap();
                    let tl2_quantized = tensor.quantize(QuantizationType::TL2).unwrap();

                    black_box((
                        i2s_quantized.compression_ratio(),
                        tl1_quantized.compression_ratio(),
                        tl2_quantized.compression_ratio(),
                    ))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark SIMD vs scalar implementations
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn bench_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar");

    let tensor = create_benchmark_tensor(32768);

    group.throughput(Throughput::Elements(32768));

    // Benchmark with SIMD enabled (default)
    group.bench_function("I2S_simd_enabled", |b| {
        let quantizer = I2SQuantizer::new();
        b.iter(|| black_box(quantizer.quantize_tensor(black_box(&tensor)).unwrap()))
    });

    // Note: We can't easily disable SIMD at runtime in this implementation,
    // but the scalar fallback paths are tested in the unit tests

    group.finish();
}

/// Fallback for non-SIMD architectures
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn bench_simd_vs_scalar(_c: &mut Criterion) {
    // No-op for architectures without SIMD support
}

criterion_group!(
    benches,
    bench_quantization_sizes,
    bench_dequantization_sizes,
    bench_round_trip,
    bench_block_sizes,
    bench_2d_tensors,
    bench_trait_quantization,
    bench_compression_ratios,
    bench_simd_vs_scalar
);

criterion_main!(benches);
