//! Canonical no-scale QK256 GEMV benchmark surfaces.
//!
//! CPU-BITNET-005c keeps this benchmark on the authoritative `i2s_qk256`
//! scalar/AVX2 APIs. `cargo bench --no-run` is compile evidence only; it is not
//! a throughput claim or a benchmark receipt.

use bitnet_quantization::i2s_qk256::{
    QK256_AVX2_GEMV_KERNEL_ID, QK256_BLOCK, QK256_PACKED_BYTES, QK256_SCALAR_GEMV_KERNEL_ID,
    gemv_qk256_row, gemv_qk256_with_kernel_selection, qk256_gemv_scalar,
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

/// Typical model dimensions for a 2B parameter model.
const TYPICAL_2B_ROWS: usize = 2048;
const TYPICAL_2B_COLS: usize = 2048;

/// Create canonical no-scale QK256 row-major packed weights.
fn create_qk256_weights(rows: usize, cols: usize) -> (Vec<u8>, usize) {
    let blocks_per_row = cols.div_ceil(QK256_BLOCK);
    let row_stride = blocks_per_row * QK256_PACKED_BYTES;
    let packed: Vec<u8> =
        (0..rows * row_stride).map(|i| ((i * 0x55 + i / 7) & 0xFF) as u8).collect();

    (packed, row_stride)
}

/// Create a deterministic activation vector.
fn create_activation_vector(cols: usize) -> Vec<f32> {
    (0..cols)
        .map(|i| {
            let x = (i as f32 - cols as f32 / 2.0) / (cols as f32 / 6.0);
            x * (-x * x / 2.0).exp()
        })
        .collect()
}

/// Benchmark the canonical scalar QK256 GEMV oracle across different sizes.
fn bench_qk256_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("qk256_gemv_scalar");

    let test_sizes = [
        ("256x256", 256, 256),
        ("1Kx1K", 1024, 1024),
        ("2Kx2K", TYPICAL_2B_ROWS, TYPICAL_2B_COLS),
        ("4Kx4K", 4096, 4096),
    ];

    for (name, rows, cols) in test_sizes {
        let (packed, _) = create_qk256_weights(rows, cols);
        let activations = create_activation_vector(cols);
        let mut output = vec![0.0f32; rows];

        group.throughput(Throughput::Elements((rows * cols) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| {
                qk256_gemv_scalar(
                    black_box(&packed),
                    black_box(&activations),
                    black_box(&mut output),
                    rows,
                    cols,
                )
                .unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark proof-level kernel selection overhead while forcing scalar.
fn bench_qk256_kernel_selection_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("qk256_kernel_selection_overhead");

    let rows = 512;
    let cols = 1024;
    let (packed, row_stride) = create_qk256_weights(rows, cols);
    let activations = create_activation_vector(cols);
    let mut output = vec![0.0f32; rows];

    group.throughput(Throughput::Elements((rows * cols) as u64));

    group.bench_function("forced_scalar_selection", |b| {
        b.iter(|| {
            let selection = gemv_qk256_with_kernel_selection(
                black_box(&packed),
                black_box(&activations),
                black_box(&mut output),
                rows,
                cols,
                row_stride,
                Some(QK256_SCALAR_GEMV_KERNEL_ID),
                true,
            )
            .unwrap();
            black_box(selection);
        });
    });

    group.finish();
}

/// Benchmark sequential row-major memory access through the canonical scalar API.
fn bench_qk256_memory_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("qk256_memory_access");

    let rows = 2048;
    let cols = 2048;
    let (packed, _) = create_qk256_weights(rows, cols);
    let activations = create_activation_vector(cols);

    group.bench_function("sequential_row_major", |b| {
        let mut output = vec![0.0f32; rows];
        b.iter(|| {
            qk256_gemv_scalar(
                black_box(&packed),
                black_box(&activations),
                black_box(&mut output),
                rows,
                cols,
            )
            .unwrap();
        });
    });

    group.finish();
}

/// Benchmark AVX2/FMA QK256 GEMV against the scalar row oracle.
#[cfg(target_arch = "x86_64")]
fn bench_qk256_avx2_gemv(c: &mut Criterion) {
    use bitnet_quantization::i2s_qk256_avx2::gemv_qk256_avx2;

    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 bench: avx2/fma unavailable");
        return;
    }

    let mut group = c.benchmark_group("qk256_gemv_avx2_vs_scalar");

    let test_sizes = [
        ("256x256", 256, 256),
        ("512x2048", 512, 2048),
        ("2Kx2K", TYPICAL_2B_ROWS, TYPICAL_2B_COLS),
    ];

    for (name, rows, cols) in test_sizes {
        let (qs, row_stride) = create_qk256_weights(rows, cols);
        let x = create_activation_vector(cols);
        let mut y = vec![0.0f32; rows];

        group.throughput(Throughput::Elements((rows * cols) as u64));

        group.bench_with_input(BenchmarkId::new("scalar_row", name), &name, |b, _| {
            b.iter(|| {
                for (row, out) in y.iter_mut().enumerate() {
                    let start = row * row_stride;
                    *out = gemv_qk256_row(
                        black_box(&qs[start..start + row_stride]),
                        black_box(&x),
                        cols,
                    );
                }
                black_box(&y);
            });
        });

        group.bench_with_input(BenchmarkId::new("forced_avx2_selection", name), &name, |b, _| {
            b.iter(|| {
                let selection = gemv_qk256_with_kernel_selection(
                    black_box(&qs),
                    black_box(&x),
                    black_box(&mut y),
                    rows,
                    cols,
                    row_stride,
                    Some(QK256_AVX2_GEMV_KERNEL_ID),
                    true,
                )
                .unwrap();
                black_box(selection);
                black_box(&y);
            });
        });

        group.bench_with_input(BenchmarkId::new("direct_avx2", name), &name, |b, _| {
            b.iter(|| {
                gemv_qk256_avx2(
                    black_box(&qs),
                    black_box(&x),
                    black_box(&mut y),
                    rows,
                    cols,
                    row_stride,
                )
                .unwrap();
                black_box(&y);
            });
        });
    }

    group.finish();
}

#[cfg(not(target_arch = "x86_64"))]
fn bench_qk256_avx2_gemv(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_qk256_scalar,
    bench_qk256_avx2_gemv,
    bench_qk256_kernel_selection_overhead,
    bench_qk256_memory_access
);
criterion_main!(benches);
