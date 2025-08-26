#![cfg(feature = "bench")]

use bitnet_kernels::KernelProvider;
use bitnet_kernels::cpu::x86::{Avx2Kernel, Avx512Kernel};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn matmul_performance(c: &mut Criterion) {
    let sizes = vec![(8, 8, 8), (16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128)];

    let mut group = c.benchmark_group("matmul_avx_comparison");

    for (m, n, k) in sizes {
        let mut a = vec![0i8; m * k];
        let mut b = vec![0u8; k * n];

        for i in 0..m * k {
            a[i] = ((i % 127) as i8).wrapping_sub(64);
        }
        for i in 0..k * n {
            b[i] = (i % 256) as u8;
        }

        let avx2 = Avx2Kernel;
        if avx2.is_available() {
            group.bench_with_input(
                BenchmarkId::new("avx2", format!("{}x{}x{}", m, n, k)),
                &(m, n, k),
                |bench, &(m, n, k)| {
                    let mut c_out = vec![0.0f32; m * n];
                    bench.iter(|| {
                        avx2.matmul_i2s(
                            black_box(&a),
                            black_box(&b),
                            black_box(&mut c_out),
                            m,
                            n,
                            k,
                        )
                        .unwrap();
                        black_box(&c_out);
                    })
                },
            );
        }

        let avx512 = Avx512Kernel;
        if avx512.is_available() {
            group.bench_with_input(
                BenchmarkId::new("avx512", format!("{}x{}x{}", m, n, k)),
                &(m, n, k),
                |bench, &(m, n, k)| {
                    let mut c_out = vec![0.0f32; m * n];
                    bench.iter(|| {
                        avx512
                            .matmul_i2s(
                                black_box(&a),
                                black_box(&b),
                                black_box(&mut c_out),
                                m,
                                n,
                                k,
                            )
                            .unwrap();
                        black_box(&c_out);
                    })
                },
            );
        }

        if avx2.is_available() && avx512.is_available() {
            let mut c2 = vec![0.0f32; m * n];
            let mut c512 = vec![0.0f32; m * n];
            avx2.matmul_i2s(&a, &b, &mut c2, m, n, k).unwrap();
            avx512.matmul_i2s(&a, &b, &mut c512, m, n, k).unwrap();
            for i in 0..m * n {
                assert!((c2[i] - c512[i]).abs() < 1e-4);
            }
        }
    }

    group.finish();
}

criterion_group!(benches, matmul_performance);
criterion_main!(benches);
