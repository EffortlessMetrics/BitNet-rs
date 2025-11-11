//! QK256 GEMV Performance Benchmarks
//!
//! This benchmark suite measures the performance of QK256 (GGML-compatible)
//! GEMV (General Matrix-Vector) operations across different SIMD implementations.
//!
//! ## Sprint-2 Track A: QK256 SIMD Optimization (#417)
//!
//! **PR1 Baseline**: This file establishes scalar baseline performance for QK256 GEMV.
//! Future PRs will add:
//! - PR2: Unpack path benchmarks (nibble LUT expansion)
//! - PR3: AVX2 kernel benchmarks (FMA tiling)
//! - PR4: Integration benchmarks (full model inference)
//!
//! **Acceptance Criteria (PR1)**:
//! - Scalar baseline recorded in `docs/baselines/qk256-scalar-baseline.json`
//! - Benchmark compiles on x86_64 and ARM (feature-gated)
//! - Throughput reported in tokens/sec for 2B model size (typical workload)

use bitnet_quantization::qk256_dispatch::{QK256, qk256_gemv_scalar};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

/// Typical model dimensions for 2B parameter model
const TYPICAL_2B_ROWS: usize = 2048; // Hidden dimension
const TYPICAL_2B_COLS: usize = 2048; // Input/output dimension

/// Helper: Create realistic quantized weight data (2-bit signed)
/// In QK256 format: packed 2-bit values + separate f32 scales
fn create_qk256_weights(rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(cols % QK256, 0, "Cols must be multiple of QK256");

    let n_blocks = rows * cols / QK256;
    let packed_size = rows * cols / 4; // 2 bits per element = 4 elements per byte

    // Packed 2-bit values (placeholder pattern)
    let packed: Vec<u8> = (0..packed_size)
        .map(|i| ((i * 0x55) & 0xFF) as u8) // Simple pattern: 01010101
        .collect();

    // Scales (one per QK256 block)
    let scales: Vec<f32> = (0..n_blocks)
        .map(|i| 1.0 / ((i % 100) + 1) as f32) // Realistic scale distribution
        .collect();

    (packed, scales)
}

/// Helper: Create activation vector (input to GEMV)
fn create_activation_vector(cols: usize) -> Vec<f32> {
    (0..cols)
        .map(|i| {
            // Gaussian-like distribution
            let x = (i as f32 - cols as f32 / 2.0) / (cols as f32 / 6.0);
            x * (-x * x / 2.0).exp()
        })
        .collect()
}

// Note: qk256_gemv_scalar() is imported from bitnet_quantization::qk256_dispatch
// We reuse the library implementation to avoid duplication and ensure benchmarks
// measure the exact same code path that will be used in production.

/// Benchmark: Scalar QK256 GEMV across different sizes
fn bench_qk256_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("qk256_gemv_scalar");

    // Test sizes: small (256), medium (1K), typical 2B (2K), large (4K)
    let test_sizes = vec![
        ("256x256", 256, 256),
        ("1Kx1K", 1024, 1024),
        ("2Kx2K", TYPICAL_2B_ROWS, TYPICAL_2B_COLS),
        ("4Kx4K", 4096, 4096),
    ];

    for (name, rows, cols) in test_sizes {
        let (packed, scales) = create_qk256_weights(rows, cols);
        let activations = create_activation_vector(cols);
        let mut output = vec![0.0f32; rows];

        // Throughput: elements processed (rows * cols)
        group.throughput(Throughput::Elements((rows * cols) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| {
                qk256_gemv_scalar(
                    black_box(&mut output),
                    black_box(rows),
                    black_box(cols),
                    black_box(&packed),
                    black_box(&scales),
                    black_box(&activations),
                );
            });
        });
    }

    group.finish();
}

/// Benchmark: QK256 dispatch overhead (future: runtime switch between scalar/AVX2)
///
/// **PR1**: No-op (dispatch not yet implemented)
/// **PR3**: Measures overhead of runtime CPU feature detection and dispatch
fn bench_qk256_dispatch_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("qk256_dispatch_overhead");

    // Placeholder: In PR3, this will benchmark the dispatch layer
    group.bench_function("dispatch_noop", |b| {
        b.iter(|| {
            // PR1: No dispatch yet, just measure empty loop overhead
            black_box(0);
        });
    });

    group.finish();
}

/// Benchmark: Memory access patterns for QK256 blocks
///
/// Measures cache behavior and memory bandwidth utilization.
/// Important for understanding why AVX2 may or may not help.
fn bench_qk256_memory_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("qk256_memory_access");

    let rows = 2048;
    let cols = 2048;
    let (packed, scales) = create_qk256_weights(rows, cols);
    let activations = create_activation_vector(cols);

    // Sequential access pattern (typical for GEMV)
    group.bench_function("sequential_row_major", |b| {
        let mut output = vec![0.0f32; rows];
        b.iter(|| {
            qk256_gemv_scalar(
                black_box(&mut output),
                rows,
                cols,
                black_box(&packed),
                black_box(&scales),
                black_box(&activations),
            );
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_qk256_scalar,
    bench_qk256_dispatch_overhead,
    bench_qk256_memory_access
);
criterion_main!(benches);

// ============================================================================
// PR1 TODO List
// ============================================================================
// [ ] Run baseline benchmarks: `cargo bench --bench qk256_gemv --features cpu`
// [ ] Record results to `docs/baselines/qk256-scalar-baseline.json`
// [ ] Verify benches compile on ARM (--target aarch64-unknown-linux-gnu)
// [ ] Document baseline performance in PR description
//
// PR2 will add:
// - bench_qk256_unpack: Nibble LUT expansion benchmarks
// - Unpack correctness tests (property-based)
//
// PR3 will add:
// - bench_qk256_avx2: AVX2 kernel benchmarks
// - bench_qk256_dispatch_overhead: Real dispatch measurement
// - Parity tests (scalar vs AVX2, cosine ≥ .99999)
//
// PR4 will add:
// - bench_qk256_full_model: End-to-end throughput (2B model)
// - Thread scaling benchmarks (1, 2, 4, 8 threads)
// - Production receipt generation
