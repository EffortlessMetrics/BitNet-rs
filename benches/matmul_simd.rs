#![cfg(feature = "bench")]

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
use bitnet_kernels::cpu::Avx2Kernel;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use bitnet_kernels::cpu::Avx512Kernel;
use bitnet_kernels::cpu::FallbackKernel;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn matmul_benchmarks(c: &mut Criterion) {
    let m = 64;
    let n = 64;
    let k = 64;
    let a = vec![1i8; m * k];
    let b = vec![1u8; k * n];

    let mut group = c.benchmark_group("matmul_i2s");

    // Scalar fallback
    group.bench_function("scalar", |bch| {
        let kernel = FallbackKernel;
        bch.iter(|| {
            let mut out = vec![0.0f32; m * n];
            kernel.matmul_i2s(black_box(&a), black_box(&b), black_box(&mut out), m, n, k).unwrap();
        });
    });

    // AVX2 implementation
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        let kernel = Avx2Kernel;
        if kernel.is_available() {
            group.bench_function("avx2", |bch| {
                bch.iter(|| {
                    let mut out = vec![0.0f32; m * n];
                    kernel
                        .matmul_i2s(black_box(&a), black_box(&b), black_box(&mut out), m, n, k)
                        .unwrap();
                });
            });
        }
    }

    // AVX-512 implementation
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        let kernel = Avx512Kernel;
        if kernel.is_available() {
            group.bench_function("avx512", |bch| {
                bch.iter(|| {
                    let mut out = vec![0.0f32; m * n];
                    kernel
                        .matmul_i2s(black_box(&a), black_box(&b), black_box(&mut out), m, n, k)
                        .unwrap();
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, matmul_benchmarks);
criterion_main!(benches);
