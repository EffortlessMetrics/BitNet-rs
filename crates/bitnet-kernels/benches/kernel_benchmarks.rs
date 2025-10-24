//! Criterion benchmarks for kernel performance regression detection
//!
//! This benchmark suite provides detailed performance measurements for all
//! kernel implementations, enabling automated regression detection and
//! performance optimization tracking.

use bitnet_common::QuantizationType;
use bitnet_kernels::KernelManager;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

/// Test data generator for consistent benchmarking
struct BenchmarkData;

impl BenchmarkData {
    fn matrix_a(m: usize, k: usize) -> Vec<i8> {
        (0..m * k).map(|i| ((i % 256) as i16 - 128) as i8).collect()
    }

    fn matrix_b(k: usize, n: usize) -> Vec<u8> {
        (0..k * n).map(|i| (i % 256) as u8).collect()
    }

    fn quantization_input(len: usize) -> Vec<f32> {
        (0..len).map(|i| (i as f32 / len as f32) * 4.0 - 2.0).collect()
    }
}

/// Benchmark matrix multiplication performance
fn bench_matmul(c: &mut Criterion) {
    let manager = KernelManager::new();
    let kernel = manager.select_best().expect("Should have a kernel");

    let mut group = c.benchmark_group("matmul");

    let sizes = vec![(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512)];

    for (m, n, k) in sizes {
        let a_data = BenchmarkData::matrix_a(m, k);
        let b_data = BenchmarkData::matrix_b(k, n);
        let mut c_result = vec![0.0f32; m * n];

        // Set throughput for GFLOPS calculation
        let ops = (m * n * k) as u64;
        group.throughput(Throughput::Elements(ops));

        group.bench_with_input(
            BenchmarkId::new(kernel.name(), format!("{}x{}x{}", m, n, k)),
            &(m, n, k),
            |b, &(m, n, k)| {
                b.iter(|| {
                    kernel
                        .matmul_i2s(
                            black_box(&a_data),
                            black_box(&b_data),
                            black_box(&mut c_result),
                            black_box(m),
                            black_box(n),
                            black_box(k),
                        )
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark quantization performance
fn bench_quantization(c: &mut Criterion) {
    let manager = KernelManager::new();
    let kernel = manager.select_best().expect("Should have a kernel");

    let mut group = c.benchmark_group("quantization");

    let qtypes = vec![QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];

    let sizes = vec![1024, 4096, 16384, 65536];

    for qtype in qtypes {
        for size in &sizes {
            let input = BenchmarkData::quantization_input(*size);
            let mut output = vec![0u8; size / 4];
            let mut scales = vec![0.0f32; size.div_ceil(32)];

            // Set throughput for elements per second
            group.throughput(Throughput::Elements(*size as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("{}-{:?}", kernel.name(), qtype), size),
                size,
                |b, &_size| {
                    b.iter(|| {
                        kernel
                            .quantize(
                                black_box(&input),
                                black_box(&mut output),
                                black_box(&mut scales),
                                black_box(qtype),
                            )
                            .unwrap();
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark kernel selection overhead
fn bench_kernel_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_selection");

    group.bench_function("manager_creation", |b| {
        b.iter(|| {
            let manager = black_box(KernelManager::new());
            black_box(manager);
        });
    });

    group.bench_function("kernel_selection", |b| {
        let manager = KernelManager::new();
        b.iter(|| {
            let kernel = black_box(manager.select_best().unwrap());
            black_box(kernel);
        });
    });

    group.finish();
}

/// Benchmark memory bandwidth for different operations
fn bench_memory_bandwidth(c: &mut Criterion) {
    let manager = KernelManager::new();
    let kernel = manager.select_best().expect("Should have a kernel");

    let mut group = c.benchmark_group("memory_bandwidth");

    // Large matrix multiplication to test memory bandwidth
    let m = 1024;
    let n = 1024;
    let k = 1024;

    let a_data = BenchmarkData::matrix_a(m, k);
    let b_data = BenchmarkData::matrix_b(k, n);
    let mut c_result = vec![0.0f32; m * n];

    // Calculate total memory accessed
    let bytes_accessed = (a_data.len() + b_data.len() + c_result.len() * 4) as u64;
    group.throughput(Throughput::Bytes(bytes_accessed));

    group.bench_function(format!("{}_large_matmul", kernel.name()), |b| {
        b.iter(|| {
            kernel
                .matmul_i2s(
                    black_box(&a_data),
                    black_box(&b_data),
                    black_box(&mut c_result),
                    black_box(m),
                    black_box(n),
                    black_box(k),
                )
                .unwrap();
        });
    });

    group.finish();
}

/// Benchmark different kernel implementations if available
fn bench_kernel_comparison(c: &mut Criterion) {
    let manager = KernelManager::new();
    let available_providers = manager.list_available_providers();

    if available_providers.len() < 2 {
        // Skip comparison if only one kernel is available
        return;
    }

    let mut group = c.benchmark_group("kernel_comparison");

    // Test with a medium-sized problem
    let m = 128;
    let n = 128;
    let k = 128;

    let a_data = BenchmarkData::matrix_a(m, k);
    let b_data = BenchmarkData::matrix_b(k, n);

    let ops = (m * n * k) as u64;
    group.throughput(Throughput::Elements(ops));

    // Benchmark the selected kernel
    let kernel = manager.select_best().expect("Should have a kernel");
    let mut c_result = vec![0.0f32; m * n];

    group.bench_function(format!("selected_{}", kernel.name()), |b| {
        b.iter(|| {
            kernel
                .matmul_i2s(
                    black_box(&a_data),
                    black_box(&b_data),
                    black_box(&mut c_result),
                    black_box(m),
                    black_box(n),
                    black_box(k),
                )
                .unwrap();
        });
    });

    group.finish();
}

/// Benchmark quantization accuracy vs performance trade-offs
fn bench_quantization_accuracy(c: &mut Criterion) {
    let manager = KernelManager::new();
    let kernel = manager.select_best().expect("Should have a kernel");

    let mut group = c.benchmark_group("quantization_accuracy");

    let size = 4096;
    let input = BenchmarkData::quantization_input(size);

    let qtypes = vec![
        ("I2S", QuantizationType::I2S),
        ("TL1", QuantizationType::TL1),
        ("TL2", QuantizationType::TL2),
    ];

    group.throughput(Throughput::Elements(size as u64));

    for (name, qtype) in qtypes {
        let mut output = vec![0u8; size / 4];
        let mut scales = vec![0.0f32; size.div_ceil(32)];

        group.bench_function(format!("{}_{}", kernel.name(), name), |b| {
            b.iter(|| {
                kernel
                    .quantize(
                        black_box(&input),
                        black_box(&mut output),
                        black_box(&mut scales),
                        black_box(qtype),
                    )
                    .unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark cache performance with different data sizes
fn bench_cache_performance(c: &mut Criterion) {
    let manager = KernelManager::new();
    let kernel = manager.select_best().expect("Should have a kernel");

    let mut group = c.benchmark_group("cache_performance");

    // Test different sizes to see cache effects
    let sizes = vec![
        (32, 32, 32),       // L1 cache friendly
        (128, 128, 128),    // L2 cache friendly
        (512, 512, 512),    // L3 cache friendly
        (1024, 1024, 1024), // Memory bound
    ];

    for (m, n, k) in sizes {
        let a_data = BenchmarkData::matrix_a(m, k);
        let b_data = BenchmarkData::matrix_b(k, n);
        let mut c_result = vec![0.0f32; m * n];

        let ops = (m * n * k) as u64;
        group.throughput(Throughput::Elements(ops));

        group.bench_with_input(
            BenchmarkId::new("cache_test", format!("{}x{}x{}", m, n, k)),
            &(m, n, k),
            |b, &(m, n, k)| {
                b.iter(|| {
                    kernel
                        .matmul_i2s(
                            black_box(&a_data),
                            black_box(&b_data),
                            black_box(&mut c_result),
                            black_box(m),
                            black_box(n),
                            black_box(k),
                        )
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark QK256 dequantization performance (scalar vs AVX2)
///
/// This benchmark measures the performance improvement of AVX2-accelerated
/// QK256 dequantization compared to the scalar reference implementation.
///
/// **Target:** ≥3× speedup on AVX2 hardware (planned with nibble-LUT + FMA)
///
/// **Current Baseline (MVP):** ~1.2× speedup with basic AVX2 vectorization
///
/// **Test sizes:** 256, 512, 1024, 4096, 16384 elements
///
/// The benchmark tests both:
/// - Scalar dequantization (baseline)
/// - AVX2 dequantization (when available on x86_64 with AVX2 support)
///
/// Throughput is measured in elements/sec and Gelem/sec to enable comparison
/// across different input sizes and hardware configurations.
///
/// ## Running the Benchmark
///
/// ```bash
/// # Quick test for 256 elements
/// cargo bench --bench kernel_benchmarks --no-default-features --features cpu,avx2 -- qk256_dequant/scalar/256 --quick
///
/// # Full suite (all sizes, scalar + AVX2)
/// cargo bench --bench kernel_benchmarks --no-default-features --features cpu,avx2 -- qk256_dequant
///
/// # With baseline comparison (requires saved baseline)
/// cargo bench --bench kernel_benchmarks --no-default-features --features cpu,avx2 -- qk256_dequant --baseline v0.1
/// ```
fn bench_qk256_dequant(c: &mut Criterion) {
    let mut group = c.benchmark_group("qk256_dequant");

    // Test sizes: powers of 2 from 256 to 16384
    // QK256 block size is 256, so all sizes are multiples
    let sizes = vec![256, 512, 1024, 4096, 16384];

    for size in sizes {
        // Generate test data: packed quantized values and scales
        // QK256 format: 2 bits per value → 4 values per byte
        let packed = vec![0x1Bu8 as i8; size / 4]; // 0x1B = 0b00011011 (mixed 2-bit values)
        let num_blocks = size.div_ceil(256);
        let scales = vec![0.5f32; num_blocks]; // Scale factor for each 256-element block

        // Set throughput for elements per second
        group.throughput(Throughput::Elements(size as u64));

        // Benchmark scalar implementation (always available)
        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, &_size| {
            use bitnet_kernels::cpu::x86::Avx2Kernel;
            let kernel = Avx2Kernel;

            b.iter(|| {
                let output = kernel
                    .dequantize_qk256_scalar(&packed, &scales, 256)
                    .expect("Scalar dequantize should succeed");
                black_box(output);
            });
        });

        // Benchmark AVX2 implementation (if available on x86_64)
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        {
            if is_x86_feature_detected!("avx2") {
                use bitnet_kernels::KernelProvider;
                use bitnet_kernels::cpu::x86::Avx2Kernel;

                group.bench_with_input(BenchmarkId::new("avx2", size), &size, |b, &_size| {
                    let kernel = Avx2Kernel;

                    b.iter(|| {
                        let output = kernel
                            .dequantize_qk256(&packed, &scales, 256)
                            .expect("AVX2 dequantize should succeed");
                        black_box(output);
                    });
                });
            }
        }
    }

    group.finish();
}

/// Benchmark QK256 dequantization breakdown (unpack, LUT, scale separately)
///
/// Detailed performance breakdown to identify optimization opportunities:
/// 1. **Unpack step**: 2-bit extraction from packed bytes → codes
/// 2. **LUT step**: Code → weight lookup ([-2.0, -1.0, 1.0, 2.0])
/// 3. **Scale step**: Weight * scale multiplication
/// 4. **Total**: Combined end-to-end dequantization
///
/// This helps pinpoint which step(s) are bottlenecks for future optimization.
fn bench_qk256_dequant_breakdown(c: &mut Criterion) {
    let mut group = c.benchmark_group("qk256_dequant_breakdown");

    const SIZE: usize = 4096; // Medium size for detailed profiling
    const NUM_BLOCKS: usize = SIZE / 256;

    let packed = vec![0x1Bu8; SIZE / 4];
    let scales = vec![0.5f32; NUM_BLOCKS];

    group.throughput(Throughput::Elements(SIZE as u64));

    // Step 1: Unpack only (no LUT, no scale)
    group.bench_function("unpack_only", |b| {
        b.iter(|| {
            let mut codes = vec![0u8; SIZE];
            for (i, &byte) in packed.iter().enumerate() {
                let base = i * 4;
                codes[base] = byte & 0x03;
                codes[base + 1] = (byte >> 2) & 0x03;
                codes[base + 2] = (byte >> 4) & 0x03;
                codes[base + 3] = (byte >> 6) & 0x03;
            }
            black_box(codes);
        });
    });

    // Step 2: Unpack + LUT (no scale)
    group.bench_function("unpack_lut", |b| {
        const LUT: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];

        b.iter(|| {
            let mut weights = vec![0.0f32; SIZE];
            for (i, &byte) in packed.iter().enumerate() {
                let base = i * 4;
                weights[base] = LUT[(byte & 0x03) as usize];
                weights[base + 1] = LUT[((byte >> 2) & 0x03) as usize];
                weights[base + 2] = LUT[((byte >> 4) & 0x03) as usize];
                weights[base + 3] = LUT[((byte >> 6) & 0x03) as usize];
            }
            black_box(weights);
        });
    });

    // Step 3: Unpack + LUT + Scale (complete pipeline)
    group.bench_function("unpack_lut_scale", |b| {
        const LUT: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];

        b.iter(|| {
            let mut output = vec![0.0f32; SIZE];
            for block_idx in 0..NUM_BLOCKS {
                let block_start = block_idx * 256;
                let packed_start = block_idx * 64;
                let scale = scales[block_idx];

                for (byte_idx, &byte) in packed[packed_start..packed_start + 64].iter().enumerate()
                {
                    let base = block_start + byte_idx * 4;
                    output[base] = LUT[(byte & 0x03) as usize] * scale;
                    output[base + 1] = LUT[((byte >> 2) & 0x03) as usize] * scale;
                    output[base + 2] = LUT[((byte >> 4) & 0x03) as usize] * scale;
                    output[base + 3] = LUT[((byte >> 6) & 0x03) as usize] * scale;
                }
            }
            black_box(output);
        });
    });

    group.finish();
}

/// Benchmark QK256 memory bandwidth utilization
///
/// Measures effective memory bandwidth for QK256 dequantization to understand
/// if the operation is compute-bound or memory-bound.
///
/// **Theoretical peak bandwidth:**
/// - DDR4-3200: ~25.6 GB/s per channel
/// - L1 cache: ~200 GB/s (load) + ~100 GB/s (store)
/// - L2 cache: ~50-100 GB/s
/// - L3 cache: ~20-40 GB/s
fn bench_qk256_memory_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("qk256_memory_bandwidth");

    let sizes = vec![
        (256, "L1_cache"),       // ~1 KB (fits in L1)
        (4096, "L2_cache"),      // ~16 KB (fits in L2)
        (65536, "L3_cache"),     // ~256 KB (fits in L3)
        (1048576, "DRAM_bound"), // ~4 MB (DRAM-bound)
    ];

    for (size, cache_level) in sizes {
        let packed = vec![0x1Bu8 as i8; size / 4];
        let num_blocks = size.div_ceil(256);
        let scales = vec![0.5f32; num_blocks];

        // Calculate total bytes accessed:
        // - Read: packed bytes (size/4) + scales (num_blocks * 4)
        // - Write: output (size * 4)
        let bytes_read = (size / 4) + (num_blocks * 4);
        let bytes_written = size * 4;
        let total_bytes = bytes_read + bytes_written;

        group.throughput(Throughput::Bytes(total_bytes as u64));

        group.bench_with_input(BenchmarkId::new("scalar", cache_level), &size, |b, &_size| {
            use bitnet_kernels::cpu::x86::Avx2Kernel;
            let kernel = Avx2Kernel;

            b.iter(|| {
                let output = kernel
                    .dequantize_qk256_scalar(&packed, &scales, 256)
                    .expect("Scalar dequantize should succeed");
                black_box(output);
            });
        });

        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        {
            if is_x86_feature_detected!("avx2") {
                use bitnet_kernels::KernelProvider;
                use bitnet_kernels::cpu::x86::Avx2Kernel;

                group.bench_with_input(
                    BenchmarkId::new("avx2", cache_level),
                    &size,
                    |b, &_size| {
                        let kernel = Avx2Kernel;

                        b.iter(|| {
                            let output = kernel
                                .dequantize_qk256(&packed, &scales, 256)
                                .expect("AVX2 dequantize should succeed");
                            black_box(output);
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark QK256 speedup vs block count
///
/// Measures how speedup changes with tensor size to understand SIMD
/// amortization costs and identify optimal block sizes.
fn bench_qk256_speedup_analysis(c: &mut Criterion) {
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        use bitnet_kernels::KernelProvider;
        use bitnet_kernels::cpu::x86::Avx2Kernel;

        let mut group = c.benchmark_group("qk256_speedup_analysis");

        // Block counts: 1, 2, 4, 8, 16, 32, 64 (256 to 16384 elements)
        let block_counts = [1, 2, 4, 8, 16, 32, 64];

        for &num_blocks in &block_counts {
            let size = num_blocks * 256;
            let packed = vec![0x1Bu8 as i8; size / 4];
            let scales = vec![0.5f32; num_blocks];

            group.throughput(Throughput::Elements(size as u64));

            let kernel = Avx2Kernel;

            // Scalar baseline
            group.bench_with_input(BenchmarkId::new("scalar", num_blocks), &num_blocks, |b, &_| {
                b.iter(|| {
                    let output = kernel
                        .dequantize_qk256_scalar(&packed, &scales, 256)
                        .expect("Scalar dequantize should succeed");
                    black_box(output);
                });
            });

            // AVX2 optimized
            group.bench_with_input(BenchmarkId::new("avx2", num_blocks), &num_blocks, |b, &_| {
                b.iter(|| {
                    let output = kernel
                        .dequantize_qk256(&packed, &scales, 256)
                        .expect("AVX2 dequantize should succeed");
                    black_box(output);
                });
            });
        }

        group.finish();
    }
}

criterion_group!(
    benches,
    bench_matmul,
    bench_quantization,
    bench_kernel_selection,
    bench_memory_bandwidth,
    bench_kernel_comparison,
    bench_quantization_accuracy,
    bench_cache_performance,
    bench_qk256_dequant,
    bench_qk256_dequant_breakdown,
    bench_qk256_memory_bandwidth,
    bench_qk256_speedup_analysis
);

criterion_main!(benches);
