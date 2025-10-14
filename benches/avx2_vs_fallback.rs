#![cfg(feature = "bench")]

use bitnet_kernels::KernelProvider;
use bitnet_kernels::cpu::{fallback::FallbackKernel, x86::Avx2Kernel};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn matmul_performance(c: &mut Criterion) {
    // Test different matrix sizes to evaluate performance characteristics
    let sizes = vec![
        (8, 8, 8),       // Small, single block
        (16, 16, 16),    // Medium
        (32, 32, 32),    // Aligned with AVX2 block size
        (64, 64, 64),    // Large
        (128, 128, 128), // Very large
        (100, 100, 100), // Non-power-of-2
    ];

    let mut group = c.benchmark_group("matmul_i2s");

    for (m, n, k) in sizes {
        // Generate test data with varied values
        let mut a = vec![0i8; m * k];
        let mut b = vec![0u8; k * n];

        for (i, val) in a.iter_mut().enumerate() {
            *val = ((i % 127) as i8).wrapping_sub(64);
        }
        for (i, val) in b.iter_mut().enumerate() {
            *val = (i % 256) as u8;
        }

        // Benchmark fallback kernel
        group.bench_with_input(
            BenchmarkId::new("fallback", format!("{}x{}x{}", m, n, k)),
            &(m, n, k),
            |bench, &(m, n, k)| {
                let kernel = FallbackKernel;
                let mut c = vec![0.0f32; m * n];
                bench.iter(|| {
                    kernel
                        .matmul_i2s(black_box(&a), black_box(&b), black_box(&mut c), m, n, k)
                        .unwrap();
                    black_box(&c);
                })
            },
        );

        // Benchmark AVX2 kernel if available
        let avx2 = Avx2Kernel;
        if avx2.is_available() {
            group.bench_with_input(
                BenchmarkId::new("avx2", format!("{}x{}x{}", m, n, k)),
                &(m, n, k),
                |bench, &(m, n, k)| {
                    let mut c = vec![0.0f32; m * n];
                    bench.iter(|| {
                        avx2.matmul_i2s(black_box(&a), black_box(&b), black_box(&mut c), m, n, k)
                            .unwrap();
                        black_box(&c);
                    })
                },
            );

            // Verify correctness for each size
            let mut c_avx2 = vec![0.0f32; m * n];
            let mut c_fallback = vec![0.0f32; m * n];

            avx2.matmul_i2s(&a, &b, &mut c_avx2, m, n, k).unwrap();
            FallbackKernel.matmul_i2s(&a, &b, &mut c_fallback, m, n, k).unwrap();

            for i in 0..m * n {
                assert!(
                    (c_avx2[i] - c_fallback[i]).abs() < 1e-4,
                    "Result mismatch at index {} for {}x{}x{} matrix",
                    i,
                    m,
                    n,
                    k
                );
            }
        }
    }

    group.finish();
}

fn matmul_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_edge_cases");

    // Test edge cases with unusual dimensions
    let edge_cases = vec![
        (1, 1, 1000),  // Dot product
        (1000, 1, 1),  // Column vector
        (1, 1000, 1),  // Row vector
        (256, 256, 1), // Outer product
        (1, 1, 32768), // Very long dot product
    ];

    let avx2 = Avx2Kernel;
    if !avx2.is_available() {
        println!("Skipping AVX2 edge case benchmarks - AVX2 not available");
        return;
    }

    for (m, n, k) in edge_cases {
        let a = vec![1i8; m * k];
        let b = vec![1u8; k * n];

        group.bench_with_input(
            BenchmarkId::new("avx2", format!("{}x{}x{}", m, n, k)),
            &(m, n, k),
            |bench, &(m, n, k)| {
                let mut c = vec![0.0f32; m * n];
                bench.iter(|| {
                    avx2.matmul_i2s(black_box(&a), black_box(&b), black_box(&mut c), m, n, k)
                        .unwrap();
                    black_box(&c);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, matmul_performance, matmul_edge_cases);
criterion_main!(benches);
