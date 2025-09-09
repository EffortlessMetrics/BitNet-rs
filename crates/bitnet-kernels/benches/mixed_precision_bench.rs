//! Mixed precision performance benchmarks
//!
//! This benchmark suite compares the performance of different precision modes
//! (FP32, FP16, BF16) in matrix multiplication operations.

#![cfg(feature = "gpu")]

use bitnet_kernels::gpu::{
    CudaDeviceInfo, MixedPrecisionKernel, PrecisionMode, convert_to_bf16_sim, convert_to_fp16_sim,
    detect_best_precision,
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

/// Helper function to create benchmark matrices
fn create_test_matrices(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32).cos()).collect();
    let c: Vec<f32> = vec![0.0; m * n];
    (a, b, c)
}

/// Create a mock kernel for benchmarking (since GPU may not be available in CI)
fn create_mock_kernel(supports_fp16: bool, supports_bf16: bool) -> Option<MixedPrecisionKernel> {
    let device_info = CudaDeviceInfo {
        device_id: 0,
        name: "Mock Benchmark Device".to_string(),
        compute_capability: if supports_bf16 {
            (8, 0)
        } else if supports_fp16 {
            (6, 0)
        } else {
            (5, 0)
        },
        total_memory: 8 * 1024 * 1024 * 1024, // 8GB
        multiprocessor_count: 68,
        max_threads_per_block: 1024,
        max_shared_memory_per_block: 49152,
        supports_fp16,
        supports_bf16,
    };

    MixedPrecisionKernel::with_device_info(device_info, PrecisionMode::Auto).ok()
}

/// Benchmark matrix multiplication across different precision modes
fn bench_matmul_precision_modes(c: &mut Criterion) {
    let sizes = vec![64, 128, 256];

    for size in sizes {
        let mut group = c.benchmark_group(format!("matmul_{}x{}", size, size));
        group.throughput(Throughput::Elements((size * size * size) as u64));

        let (a, b, _) = create_test_matrices(size, size, size);

        // Test with different device capabilities
        let kernels: Vec<(&str, MixedPrecisionKernel)> = vec![
            ("modern_gpu", create_mock_kernel(true, true)), // Supports FP16 and BF16
            ("older_gpu", create_mock_kernel(true, false)), // Supports FP16 only
            ("legacy_gpu", create_mock_kernel(false, false)), // FP32 only
        ]
        .into_iter()
        .filter_map(|(name, kernel)| kernel.map(|k| (name, k)))
        .collect();

        for (device_name, mut kernel) in kernels {
            let precision_modes =
                vec![PrecisionMode::FP32, PrecisionMode::FP16, PrecisionMode::BF16];

            for precision in precision_modes {
                if !kernel.supports_fp16() && precision == PrecisionMode::FP16 {
                    continue;
                }
                if !kernel.supports_bf16() && precision == PrecisionMode::BF16 {
                    continue;
                }

                let bench_name = format!("{}_{}_{:?}", device_name, size, precision);
                let mut c_result = vec![0.0; size * size];

                group.bench_with_input(
                    BenchmarkId::new("precision_comparison", bench_name),
                    &size,
                    |bench, &_size| {
                        bench.iter(|| {
                            let result = match precision {
                                PrecisionMode::FP32 => {
                                    kernel.matmul_fp32(&a, &b, &mut c_result, size, size, size)
                                }
                                PrecisionMode::FP16 => {
                                    kernel.matmul_fp16(&a, &b, &mut c_result, size, size, size)
                                }
                                PrecisionMode::BF16 => {
                                    kernel.matmul_bf16(&a, &b, &mut c_result, size, size, size)
                                }
                                PrecisionMode::Auto => {
                                    kernel.matmul_auto(&a, &b, &mut c_result, size, size, size)
                                }
                            };
                            result.unwrap();
                            black_box(&c_result);
                        })
                    },
                );
            }
        }
    }
}

/// Benchmark precision conversion utilities
fn bench_precision_conversion(c: &mut Criterion) {
    let sizes = vec![1024, 4096, 16384];

    for size in sizes {
        let mut group = c.benchmark_group(format!("precision_conversion_{}", size));
        group.throughput(Throughput::Elements(size as u64));

        let test_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();

        group.bench_function("fp16_conversion", |b| {
            b.iter(|| {
                let converted = convert_to_fp16_sim(black_box(&test_data));
                black_box(converted);
            })
        });

        group.bench_function("bf16_conversion", |b| {
            b.iter(|| {
                let converted = convert_to_bf16_sim(black_box(&test_data));
                black_box(converted);
            })
        });
    }
}

/// Benchmark optimal precision detection
fn bench_precision_detection(c: &mut Criterion) {
    let device_configs = vec![
        ("ampere", (8, 0), true, true),
        ("turing", (7, 5), true, false),
        ("pascal", (6, 1), true, false),
        ("maxwell", (5, 2), false, false),
    ];

    c.bench_function("precision_detection", |b| {
        b.iter(|| {
            for (name, compute_cap, fp16, bf16) in &device_configs {
                let device_info = CudaDeviceInfo {
                    device_id: 0,
                    name: format!("Benchmark {}", name),
                    compute_capability: *compute_cap,
                    total_memory: 8 * 1024 * 1024 * 1024,
                    multiprocessor_count: 68,
                    max_threads_per_block: 1024,
                    max_shared_memory_per_block: 49152,
                    supports_fp16: *fp16,
                    supports_bf16: *bf16,
                };

                let precision = detect_best_precision(black_box(&device_info));
                black_box(precision);
            }
        })
    });
}

criterion_group!(
    benches,
    bench_matmul_precision_modes,
    bench_precision_conversion,
    bench_precision_detection
);
criterion_main!(benches);
