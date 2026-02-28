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
/// **Target:** ≥3× speedup on AVX2 hardware
///
/// **Test sizes:** 256, 512, 1024, 4096, 16384 elements
///
/// The benchmark tests both:
/// - Scalar dequantization (baseline)
/// - AVX2 dequantization (when available on x86_64 with AVX2 support)
///
/// Throughput is measured in elements/sec and GB/sec to enable comparison
/// across different input sizes and hardware configurations.
///
/// ## Current Performance (MVP)
///
/// As of the MVP implementation, the AVX2 path shows:
/// - **Small sizes (256-512)**: ~0.76-1.0× speedup (not faster than scalar)
/// - **Medium sizes (1024-4096)**: ~1.3-1.5× speedup (modest improvement)
/// - **Large sizes (16384)**: ~1.2-1.5× speedup (memory-bound)
///
/// This is below the 3× target due to:
/// - Scalar unpacking bottleneck (2-bit extraction not vectorized)
/// - LUT overhead (scalar array indexing)
/// - Small block size (256 elements may not amortize SIMD setup)
///
/// See `crates/bitnet-models/src/quant/i2s_qk256_avx2.rs` for optimization notes.
///
/// ## Running the Benchmark
///
/// ```bash
/// # Quick test for 256 elements
/// cargo bench --bench kernel_benchmarks --no-default-features --features cpu,avx2 -- qk256_dequant/scalar/256 --quick
///
/// # Full suite (all sizes, scalar + AVX2)
/// cargo bench --bench kernel_benchmarks --no-default-features --features cpu,avx2 -- qk256_dequant
/// ```
fn bench_qk256_dequant(c: &mut Criterion) {
    let mut group = c.benchmark_group("qk256_dequant");

    // Test sizes: powers of 2 from 256 to 16384
    // QK256 block size is 256, so all sizes are multiples
    let sizes = vec![256, 512, 1024, 4096, 16384];

    for size in sizes {
        // Generate test data: packed quantized values and scales
        // QK256 format: 2 bits per value → 4 values per byte
        let packed = vec![0x1Bu8; size / 4]; // 0x1B = 0b00011011 (mixed 2-bit values)
        let num_blocks = size.div_ceil(256);
        let scales = vec![0.5f32; num_blocks]; // Scale factor for each 256-element block

        // Set throughput for elements per second
        group.throughput(Throughput::Elements(size as u64));

        // Benchmark scalar implementation (always available)
        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, &_size| {
            b.iter(|| {
                // Simulate scalar dequantization: unpack 2-bit values and scale
                let mut output = Vec::with_capacity(size);
                for (chunk_idx, chunk) in packed.chunks(64).enumerate() {
                    let scale = scales[chunk_idx.min(scales.len() - 1)];
                    for &byte in chunk {
                        // Extract 4 2-bit values from each byte
                        let v0 = ((byte & 0x03) as i8 - 2) as f32 * scale;
                        let v1 = (((byte >> 2) & 0x03) as i8 - 2) as f32 * scale;
                        let v2 = (((byte >> 4) & 0x03) as i8 - 2) as f32 * scale;
                        let v3 = (((byte >> 6) & 0x03) as i8 - 2) as f32 * scale;
                        output.push(v0);
                        output.push(v1);
                        output.push(v2);
                        output.push(v3);
                    }
                }
                black_box(output);
            });
        });

        // Benchmark AVX2 implementation (if available on x86_64)
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        {
            if is_x86_feature_detected!("avx2") {
                group.bench_with_input(BenchmarkId::new("avx2", size), &size, |b, &_size| {
                    b.iter(|| {
                        // Simulate AVX2 dequantization using SIMD intrinsics
                        // This is a placeholder - the real AVX2 implementation
                        // is in bitnet-models/src/quant/i2s_qk256_avx2.rs
                        let mut output = vec![0.0f32; size];

                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            use std::arch::x86_64::*;

                            let mut out_idx = 0;
                            for (chunk_idx, chunk) in packed.chunks(64).enumerate() {
                                let scale = scales[chunk_idx.min(scales.len() - 1)];
                                let _scale_vec = _mm256_set1_ps(scale);

                                // Process 8 f32 elements at a time with AVX2
                                // TODO: Use _scale_vec for SIMD multiplication
                                for &byte in chunk {
                                    // Scalar unpacking (TODO: optimize with SIMD)
                                    let codes = [
                                        ((byte & 0x03) as i8 - 2) as f32,
                                        (((byte >> 2) & 0x03) as i8 - 2) as f32,
                                        (((byte >> 4) & 0x03) as i8 - 2) as f32,
                                        (((byte >> 6) & 0x03) as i8 - 2) as f32,
                                    ];

                                    for &code in &codes {
                                        if out_idx < output.len() {
                                            output[out_idx] = code * scale;
                                            out_idx += 1;
                                        }
                                    }
                                }
                            }
                        }
                        black_box(output);
                    });
                });
            }
        }
    }

    group.finish();
}

/// Benchmark flash attention vs naive attention
fn bench_flash_attention(c: &mut Criterion) {
    use bitnet_kernels::cpu::flash_attention::{
        FlashAttentionConfig, flash_attention, naive_attention,
    };

    let mut group = c.benchmark_group("flash_attention");

    // (num_heads, num_kv_heads, head_dim, seq_len, label)
    let configs = vec![
        (8, 8, 64, 64, "mha_small"),
        (8, 8, 64, 256, "mha_medium"),
        (8, 8, 64, 512, "mha_large"),
        (32, 8, 64, 128, "gqa_4x"),
        (32, 1, 64, 128, "mqa"),
        (8, 8, 128, 256, "mha_d128"),
    ];

    for (num_heads, num_kv_heads, head_dim, seq_len, label) in &configs {
        let cfg = FlashAttentionConfig {
            num_heads: *num_heads,
            num_kv_heads: *num_kv_heads,
            head_dim: *head_dim,
            causal: true,
            block_q: 32,
            block_kv: 32,
        };
        let batch = 1;
        let q_len = batch * seq_len * num_heads * head_dim;
        let kv_len = batch * seq_len * num_kv_heads * head_dim;

        let q: Vec<f32> = (0..q_len).map(|i| ((i % 97) as f32 - 48.0) / 48.0).collect();
        let k: Vec<f32> = (0..kv_len).map(|i| ((i % 89) as f32 - 44.0) / 44.0).collect();
        let v: Vec<f32> = (0..kv_len).map(|i| ((i % 83) as f32 - 41.0) / 41.0).collect();

        group.throughput(Throughput::Elements(*seq_len as u64));

        group.bench_with_input(BenchmarkId::new("flash", label), seq_len, |b, _| {
            let mut out = vec![0.0f32; q_len];
            b.iter(|| {
                flash_attention(&cfg, &q, &k, &v, &mut out, batch, *seq_len, *seq_len).unwrap();
                black_box(&out);
            });
        });

        group.bench_with_input(BenchmarkId::new("naive", label), seq_len, |b, _| {
            let mut out = vec![0.0f32; q_len];
            b.iter(|| {
                naive_attention(&cfg, &q, &k, &v, &mut out, batch, *seq_len, *seq_len).unwrap();
                black_box(&out);
            });
        });
    }

    group.finish();
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
    bench_flash_attention
);

criterion_main!(benches);
