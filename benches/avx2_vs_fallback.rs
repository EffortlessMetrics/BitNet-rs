#![cfg(feature = "bench")]

use std::time::{Duration, Instant};

use bitnet_kernels::KernelProvider;
use bitnet_kernels::cpu::{fallback::FallbackKernel, x86::Avx2Kernel};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn matmul_avx2_vs_fallback(c: &mut Criterion) {
    let m = 32;
    let n = 32;
    let k = 32;
    let a = vec![1i8; m * k];
    let b = vec![1u8; k * n];
    let mut c_out = vec![0.0f32; m * n];

    // Fallback benchmark
    c.bench_function("matmul_fallback", |bch| {
        let kernel = FallbackKernel;
        bch.iter(|| {
            let mut out = c_out.clone();
            kernel.matmul_i2s(black_box(&a), black_box(&b), black_box(&mut out), m, n, k).unwrap();
            black_box(out)
        })
    });

    // AVX2 benchmark and performance check
    let avx2 = Avx2Kernel;
    if avx2.is_available() {
        c.bench_function("matmul_avx2", |bch| {
            bch.iter(|| {
                let mut out = c_out.clone();
                avx2.matmul_i2s(black_box(&a), black_box(&b), black_box(&mut out), m, n, k)
                    .unwrap();
                black_box(out)
            })
        });

        // Verify AVX2 is faster than fallback with simple timing
        let iterations = 10;
        let mut dur_fallback = Duration::default();
        let mut dur_avx2 = Duration::default();
        for _ in 0..iterations {
            let mut out = c_out.clone();
            let start = Instant::now();
            FallbackKernel.matmul_i2s(&a, &b, &mut out, m, n, k).unwrap();
            dur_fallback += start.elapsed();

            let mut out = c_out.clone();
            let start = Instant::now();
            avx2.matmul_i2s(&a, &b, &mut out, m, n, k).unwrap();
            dur_avx2 += start.elapsed();
        }
        assert!(dur_avx2 < dur_fallback, "AVX2 kernel should be faster than fallback");
    }
}

criterion_group!(benches, matmul_avx2_vs_fallback);
criterion_main!(benches);
