//! Criterion benchmarks for kernel performance regression detection
//!
//! This benchmark suite provides detailed performance measurements for all
//! kernel implementations, enabling automated regression detection and
//! performance optimization tracking.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use bitnet_kernels::{KernelManager, KernelProvider};
use bitnet_common::QuantizationType;

/// Test data generator for consistent benchmarking
struct BenchmarkData;

impl BenchmarkData {
    fn matrix_a(m: usize, k: usize) -> Vec<i8> {
        (0..m * k).map(|i| ((i % 256) as i8).wrapping_sub(128)).collect()
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
    
    let sizes = vec![
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ];
    
    for (m, n, k) in sizes {
        let a = BenchmarkData::matrix_a(m, k);
        let b = BenchmarkData::matrix_b(k, n);
        let mut c_result = vec![0.0f32; m * n];
        
        // Set throughput for GFLOPS calculation
        let ops = (m * n * k) as u64;
        group.throughput(Throughput::Elements(ops));
        
        group.bench_with_input(
            BenchmarkId::new(kernel.name(), format!("{}x{}x{}", m, n, k)),
            &(m, n, k),
            |b, &(m, n, k)| {
                b.iter(|| {
                    kernel.matmul_i2s(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut c_result),
                        black_box(m),
                        black_box(n),
                        black_box(k),
                    ).unwrap();
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
    
    let qtypes = vec![
        QuantizationType::I2S,
        QuantizationType::TL1,
        QuantizationType::TL2,
    ];
    
    let sizes = vec![1024, 4096, 16384, 65536];
    
    for qtype in qtypes {
        for size in &sizes {
            let input = BenchmarkData::quantization_input(*size);
            let mut output = vec![0u8; size / 4];
            let mut scales = vec![0.0f32; (size + 31) / 32];
            
            // Set throughput for elements per second
            group.throughput(Throughput::Elements(*size as u64));
            
            group.bench_with_input(
                BenchmarkId::new(
                    format!("{}-{:?}", kernel.name(), qtype),
                    size,
                ),
                size,
                |b, &size| {
                    b.iter(|| {
                        kernel.quantize(
                            black_box(&input),
                            black_box(&mut output),
                            black_box(&mut scales),
                            black_box(qtype),
                        ).unwrap();
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
    
    let a = BenchmarkData::matrix_a(m, k);
    let b = BenchmarkData::matrix_b(k, n);
    let mut c_result = vec![0.0f32; m * n];
    
    // Calculate total memory accessed
    let bytes_accessed = (a.len() + b.len() + c_result.len() * 4) as u64;
    group.throughput(Throughput::Bytes(bytes_accessed));
    
    group.bench_function(
        format!("{}_large_matmul", kernel.name()),
        |b| {
            b.iter(|| {
                kernel.matmul_i2s(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut c_result),
                    black_box(m),
                    black_box(n),
                    black_box(k),
                ).unwrap();
            });
        },
    );
    
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
    
    let a = BenchmarkData::matrix_a(m, k);
    let b = BenchmarkData::matrix_b(k, n);
    
    let ops = (m * n * k) as u64;
    group.throughput(Throughput::Elements(ops));
    
    // Benchmark the selected kernel
    let kernel = manager.select_best().expect("Should have a kernel");
    let mut c_result = vec![0.0f32; m * n];
    
    group.bench_function(
        format!("selected_{}", kernel.name()),
        |b| {
            b.iter(|| {
                kernel.matmul_i2s(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut c_result),
                    black_box(m),
                    black_box(n),
                    black_box(k),
                ).unwrap();
            });
        },
    );
    
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
        let mut scales = vec![0.0f32; (size + 31) / 32];
        
        group.bench_function(
            format!("{}_{}", kernel.name(), name),
            |b| {
                b.iter(|| {
                    kernel.quantize(
                        black_box(&input),
                        black_box(&mut output),
                        black_box(&mut scales),
                        black_box(qtype),
                    ).unwrap();
                });
            },
        );
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
        (32, 32, 32),    // L1 cache friendly
        (128, 128, 128), // L2 cache friendly
        (512, 512, 512), // L3 cache friendly
        (1024, 1024, 1024), // Memory bound
    ];
    
    for (m, n, k) in sizes {
        let a = BenchmarkData::matrix_a(m, k);
        let b = BenchmarkData::matrix_b(k, n);
        let mut c_result = vec![0.0f32; m * n];
        
        let ops = (m * n * k) as u64;
        group.throughput(Throughput::Elements(ops));
        
        group.bench_with_input(
            BenchmarkId::new("cache_test", format!("{}x{}x{}", m, n, k)),
            &(m, n, k),
            |b, &(m, n, k)| {
                b.iter(|| {
                    kernel.matmul_i2s(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut c_result),
                        black_box(m),
                        black_box(n),
                        black_box(k),
                    ).unwrap();
                });
            },
        );
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
    bench_cache_performance
);

criterion_main!(benches);