//! Criterion micro-benchmarks for Apple Silicon NEON kernel operations.
//!
//! Benchmarks cover: matrix multiplication, softmax (via layernorm path),
//! layer normalization, elementwise ops, activations, reductions, batch
//! normalization, transpose, and pooling.
//!
//! On non-aarch64 targets the benchmark group is empty so the file still
//! compiles without errors.

#[cfg(target_arch = "aarch64")]
use criterion::{BatchSize, BenchmarkId};
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(target_arch = "aarch64")]
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Helpers: deterministic mock data
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn make_f32_vec(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32) / (n as f32) - 0.5).collect()
}

#[cfg(target_arch = "aarch64")]
fn make_f32_ones(n: usize) -> Vec<f32> {
    vec![1.0f32; n]
}

#[cfg(target_arch = "aarch64")]
fn make_f32_zeros(n: usize) -> Vec<f32> {
    vec![0.0f32; n]
}

#[cfg(target_arch = "aarch64")]
fn make_f32_matrix(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect()
}

// ===========================================================================
// aarch64 benchmarks — import real NEON kernels from bitnet_kernels
// ===========================================================================

#[cfg(target_arch = "aarch64")]
fn bench_neon_matmul(c: &mut Criterion) {
    use bitnet_kernels::cpu::simd_matmul::{SimdMatmulConfig, simd_matmul_f32};

    let mut group = c.benchmark_group("neon_matmul");
    for &size in &[64usize, 256] {
        let a = make_f32_matrix(size, size);
        let b = make_f32_matrix(size, size);
        let cfg = SimdMatmulConfig {
            m: size,
            n: size,
            k: size,
            alpha: 1.0,
            beta: 0.0,
            transpose_a: false,
            transpose_b: false,
        };
        group.bench_with_input(BenchmarkId::new("square", size), &size, |bench, &_n| {
            bench.iter_batched(
                || vec![0.0f32; size * size],
                |mut out| {
                    simd_matmul_f32(black_box(&a), black_box(&b), &mut out, &cfg).unwrap();
                    black_box(out)
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

#[cfg(target_arch = "aarch64")]
fn bench_neon_softmax(c: &mut Criterion) {
    use bitnet_kernels::cpu::neon_reductions::{neon_max_f32, neon_sum_f32};

    let mut group = c.benchmark_group("neon_softmax");
    for &size in &[128usize, 1024, 4096] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, &n| {
            bench.iter_batched(
                || make_f32_vec(n),
                |mut v| {
                    // NEON-accelerated softmax: max → subtract → exp → sum → normalize
                    let max_val = unsafe { neon_max_f32(&v) };
                    for x in v.iter_mut() {
                        *x = (*x - max_val).exp();
                    }
                    let sum = unsafe { neon_sum_f32(&v) };
                    let inv = 1.0 / sum;
                    for x in v.iter_mut() {
                        *x *= inv;
                    }
                    black_box(v)
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

#[cfg(target_arch = "aarch64")]
fn bench_neon_layernorm(c: &mut Criterion) {
    use bitnet_kernels::cpu::neon_layernorm::{layernorm_neon, rmsnorm_neon};

    let mut group = c.benchmark_group("neon_layernorm");
    for &size in &[128usize, 1024, 4096] {
        let input = make_f32_vec(size);
        let gamma = make_f32_ones(size);
        let beta = make_f32_zeros(size);

        group.bench_with_input(BenchmarkId::new("layernorm", size), &size, |bench, &n| {
            let mut output = vec![0.0f32; n];
            bench.iter(|| {
                unsafe { layernorm_neon(&input, &mut output, &gamma, &beta, 1e-5) };
                black_box(&output);
            });
        });

        group.bench_with_input(BenchmarkId::new("rmsnorm", size), &size, |bench, &n| {
            let mut output = vec![0.0f32; n];
            bench.iter(|| {
                unsafe { rmsnorm_neon(&input, &mut output, &gamma, 1e-5) };
                black_box(&output);
            });
        });
    }
    group.finish();
}

#[cfg(target_arch = "aarch64")]
fn bench_neon_elementwise(c: &mut Criterion) {
    use bitnet_kernels::cpu::neon_elementwise::{
        neon_add_f32, neon_fma_f32, neon_mul_f32, neon_scale_f32,
    };

    let size = 4096usize;
    let a = make_f32_vec(size);
    let b = make_f32_vec(size);
    let cv = make_f32_vec(size);

    let mut group = c.benchmark_group("neon_elementwise");

    group.bench_function("add_4096", |bench| {
        let mut out = vec![0.0f32; size];
        bench.iter(|| {
            unsafe { neon_add_f32(&a, &b, &mut out) };
            black_box(&out);
        });
    });

    group.bench_function("mul_4096", |bench| {
        let mut out = vec![0.0f32; size];
        bench.iter(|| {
            unsafe { neon_mul_f32(&a, &b, &mut out) };
            black_box(&out);
        });
    });

    group.bench_function("scale_4096", |bench| {
        let mut out = vec![0.0f32; size];
        bench.iter(|| {
            unsafe { neon_scale_f32(&a, 0.5, &mut out) };
            black_box(&out);
        });
    });

    group.bench_function("fma_4096", |bench| {
        let mut out = vec![0.0f32; size];
        bench.iter(|| {
            unsafe { neon_fma_f32(&a, &b, &cv, &mut out) };
            black_box(&out);
        });
    });

    group.finish();
}

#[cfg(target_arch = "aarch64")]
fn bench_neon_activations(c: &mut Criterion) {
    use bitnet_kernels::cpu::{gelu_vec, silu_vec};

    let mut group = c.benchmark_group("neon_activations");
    for &size in &[256usize, 1024, 4096] {
        let input = make_f32_vec(size);

        group.bench_with_input(BenchmarkId::new("relu", size), &size, |bench, &n| {
            bench.iter_batched(
                || make_f32_vec(n),
                |mut v| {
                    // In-place relu via NEON max with zero
                    for x in v.iter_mut() {
                        *x = x.max(0.0);
                    }
                    black_box(v)
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("sigmoid", size), &size, |bench, &_n| {
            bench.iter_batched(
                || make_f32_vec(size),
                |mut v| {
                    for x in v.iter_mut() {
                        *x = 1.0 / (1.0 + (-*x).exp());
                    }
                    black_box(v)
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("silu", size), &size, |bench, &_n| {
            bench.iter(|| black_box(silu_vec(black_box(&input))));
        });

        group.bench_with_input(BenchmarkId::new("gelu", size), &size, |bench, &_n| {
            bench.iter(|| black_box(gelu_vec(black_box(&input))));
        });
    }
    group.finish();
}

#[cfg(target_arch = "aarch64")]
fn bench_neon_reductions(c: &mut Criterion) {
    use bitnet_kernels::cpu::neon_reductions::{neon_dot_f32, neon_max_f32, neon_sum_f32};

    let mut group = c.benchmark_group("neon_reductions");
    for &size in &[256usize, 1024, 4096] {
        let data = make_f32_vec(size);
        let data2 = make_f32_vec(size);

        group.bench_with_input(BenchmarkId::new("sum", size), &size, |bench, &_n| {
            bench.iter(|| black_box(unsafe { neon_sum_f32(black_box(&data)) }));
        });

        group.bench_with_input(BenchmarkId::new("max", size), &size, |bench, &_n| {
            bench.iter(|| black_box(unsafe { neon_max_f32(black_box(&data)) }));
        });

        group.bench_with_input(BenchmarkId::new("dot_product", size), &size, |bench, &_n| {
            bench.iter(|| black_box(unsafe { neon_dot_f32(black_box(&data), black_box(&data2)) }));
        });
    }
    group.finish();
}

#[cfg(target_arch = "aarch64")]
fn bench_neon_batch_norm(c: &mut Criterion) {
    use bitnet_kernels::cpu::batch_norm::batch_norm_inference;

    let mut group = c.benchmark_group("neon_batch_norm");
    // batch_norm_inference expects input of shape [batch_size * num_features]
    for &(batch, features) in &[(8usize, 64usize), (32, 128), (64, 256)] {
        let n = batch * features;
        let input = make_f32_vec(n);
        let gamma = make_f32_ones(features);
        let beta = make_f32_zeros(features);
        let running_mean = make_f32_zeros(features);
        let running_var = make_f32_ones(features);

        group.bench_with_input(
            BenchmarkId::new("inference", format!("{batch}x{features}")),
            &n,
            |bench, &_n| {
                bench.iter(|| {
                    black_box(
                        batch_norm_inference(
                            black_box(&input),
                            &gamma,
                            &beta,
                            &running_mean,
                            &running_var,
                            1e-5,
                        )
                        .unwrap(),
                    )
                });
            },
        );
    }
    group.finish();
}

#[cfg(target_arch = "aarch64")]
fn bench_neon_transpose(c: &mut Criterion) {
    use bitnet_kernels::cpu::transpose::TransposeKernel;

    let mut group = c.benchmark_group("neon_transpose");
    for &size in &[4usize, 16, 64] {
        let data = make_f32_matrix(size, size);
        group.bench_with_input(
            BenchmarkId::new("2d_square", format!("{size}x{size}")),
            &size,
            |bench, &n| {
                bench.iter(|| {
                    black_box(TransposeKernel::transpose_2d(black_box(&data), n, n).unwrap())
                });
            },
        );
    }
    group.finish();
}

#[cfg(target_arch = "aarch64")]
fn bench_neon_pooling(c: &mut Criterion) {
    use bitnet_kernels::cpu::pooling::{PoolConfig, PoolType, pool_1d};

    let input = make_f32_vec(4096);
    let mut group = c.benchmark_group("neon_pooling");

    for &(label, pool_type) in &[("max", PoolType::Max), ("avg", PoolType::Average)] {
        for &kernel_size in &[2usize, 4, 8] {
            let config = PoolConfig::new(pool_type, kernel_size, kernel_size, 0);
            group.bench_with_input(
                BenchmarkId::new(label, format!("k{kernel_size}")),
                &kernel_size,
                |bench, &_k| {
                    bench.iter(|| black_box(pool_1d(black_box(&input), &config).unwrap()));
                },
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion wiring
// ---------------------------------------------------------------------------
#[cfg(target_arch = "aarch64")]
criterion_group!(
    benches,
    bench_neon_matmul,
    bench_neon_softmax,
    bench_neon_layernorm,
    bench_neon_elementwise,
    bench_neon_activations,
    bench_neon_reductions,
    bench_neon_batch_norm,
    bench_neon_transpose,
    bench_neon_pooling,
);

// Stub for non-aarch64 targets so the file still compiles.
#[cfg(not(target_arch = "aarch64"))]
fn _stub(_c: &mut Criterion) {}

#[cfg(not(target_arch = "aarch64"))]
criterion_group!(benches, _stub);

criterion_main!(benches);
