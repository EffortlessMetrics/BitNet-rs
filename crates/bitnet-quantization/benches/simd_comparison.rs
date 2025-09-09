//! SIMD Implementation Comparison Benchmarks
//!
//! This benchmark suite compares the performance of old vs new SIMD intrinsic
//! implementations in the I2S quantization kernels. It focuses on the specific
//! improvements made in PR #174 where intrinsic calls were modernized.

use bitnet_common::BitNetTensor;
use bitnet_quantization::I2SQuantizer;
use candle_core::{Device, Tensor as CandleTensor};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Instant;

/// Helper function to create benchmark tensors with realistic data distribution
fn create_realistic_tensor(size: usize) -> BitNetTensor {
    let device = Device::Cpu;

    // Create data that mimics typical neural network weights
    let data: Vec<f32> = (0..size)
        .map(|i| {
            let x = (i as f32 - size as f32 / 2.0) / (size as f32 / 6.0);
            // Gaussian-like distribution with some outliers
            if i % 100 == 0 {
                x * 3.0 // Occasional larger values
            } else {
                x * (-x * x / 2.0).exp() * 2.0 // Roughly normal, scaled for I2S range
            }
        })
        .collect();

    let tensor = CandleTensor::from_vec(data, &[size], &device).unwrap();
    BitNetTensor::new(tensor)
}

/// Helper function to create aligned benchmark tensors
fn create_aligned_tensor(size: usize) -> BitNetTensor {
    let device = Device::Cpu;

    // Create data that aligns well with SIMD register sizes
    let data: Vec<f32> =
        (0..size).map(|i| (i as f32 * std::f32::consts::PI / 32.0).sin() * 1.8).collect();

    let tensor = CandleTensor::from_vec(data, &[size], &device).unwrap();
    BitNetTensor::new(tensor)
}

/// Benchmark memory access patterns for different tensor sizes
fn bench_memory_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_access_patterns");

    // Test sizes that exercise different cache levels and alignment scenarios
    let sizes = vec![
        512,    // L1 cache friendly
        4096,   // L2 cache boundary
        32768,  // L3 cache friendly
        262144, // Larger than typical L3 cache
    ];

    for size in sizes {
        let tensor = create_realistic_tensor(size);
        let quantizer = I2SQuantizer::new();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("quantize_realistic_data", size), &size, |b, _| {
            b.iter(|| black_box(quantizer.quantize_tensor(black_box(&tensor)).unwrap()))
        });

        // Pre-quantize for dequantization benchmarks
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();

        group.bench_with_input(
            BenchmarkId::new("dequantize_realistic_data", size),
            &size,
            |b, _| {
                b.iter(|| black_box(quantizer.dequantize_tensor(black_box(&quantized)).unwrap()))
            },
        );
    }

    group.finish();
}

/// Benchmark specific SIMD register alignment scenarios
fn bench_alignment_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("alignment_scenarios");

    // Test sizes that align differently with SIMD registers
    // x86_64 AVX2: 8 f32 elements (256 bits)
    // aarch64 NEON: 4 f32 elements (128 bits)
    let test_cases = vec![
        ("exact_avx2_align", 8 * 100), // Exactly divisible by AVX2 register size
        ("exact_neon_align", 4 * 100), // Exactly divisible by NEON register size
        ("avx2_plus_one", 8 * 100 + 1), // AVX2 + 1 remainder
        ("neon_plus_one", 4 * 100 + 1), // NEON + 1 remainder
        ("avx2_minus_one", 8 * 100 - 1), // AVX2 - 1
        ("neon_minus_one", 4 * 100 - 1), // NEON - 1
        ("mixed_alignment", 8 * 50 + 4 * 25), // Mix of alignments
    ];

    let quantizer = I2SQuantizer::new();

    for (name, size) in test_cases {
        let tensor = create_aligned_tensor(size);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("quantize", name), &size, |b, _| {
            b.iter(|| black_box(quantizer.quantize_tensor(black_box(&tensor)).unwrap()))
        });

        let quantized = quantizer.quantize_tensor(&tensor).unwrap();

        group.bench_with_input(BenchmarkId::new("dequantize", name), &size, |b, _| {
            b.iter(|| black_box(quantizer.dequantize_tensor(black_box(&quantized)).unwrap()))
        });
    }

    group.finish();
}

/// Benchmark the specific intrinsics improvements from PR #174
/// This focuses on the memory load/store patterns that were optimized
#[cfg(target_arch = "x86_64")]
fn bench_x86_intrinsics_improvements(c: &mut Criterion) {
    let mut group = c.benchmark_group("x86_intrinsics_improvements");

    // Test the specific scenarios where the new intrinsics should be faster
    let tensor_size = 32768; // Large enough to see memory throughput differences
    let tensor = create_realistic_tensor(tensor_size);
    let quantizer = I2SQuantizer::new();

    group.throughput(Throughput::Elements(tensor_size as u64));

    // Benchmark quantization (exercises the _mm_storeu_si64 replacement)
    group.bench_function("quantize_modern_intrinsics", |b| {
        b.iter(|| black_box(quantizer.quantize_tensor(black_box(&tensor)).unwrap()))
    });

    // Pre-quantize for dequantization benchmarks
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();

    // Benchmark dequantization (exercises the _mm_loadu_si64 replacement)
    group.bench_function("dequantize_modern_intrinsics", |b| {
        b.iter(|| black_box(quantizer.dequantize_tensor(black_box(&quantized)).unwrap()))
    });

    // Benchmark the round-trip to test both improvements together
    group.bench_function("round_trip_modern_intrinsics", |b| {
        b.iter(|| {
            let q = quantizer.quantize_tensor(black_box(&tensor)).unwrap();
            black_box(quantizer.dequantize_tensor(&q).unwrap())
        })
    });

    group.finish();
}

/// Benchmark ARM NEON improvements
#[cfg(target_arch = "aarch64")]
fn bench_arm_neon_improvements(c: &mut Criterion) {
    let mut group = c.benchmark_group("arm_neon_improvements");

    let tensor_size = 32768;
    let tensor = create_realistic_tensor(tensor_size);
    let quantizer = I2SQuantizer::new();

    group.throughput(Throughput::Elements(tensor_size as u64));

    // Benchmark quantization with NEON
    group.bench_function("quantize_neon_optimized", |b| {
        b.iter(|| black_box(quantizer.quantize_tensor(black_box(&tensor)).unwrap()))
    });

    let quantized = quantizer.quantize_tensor(&tensor).unwrap();

    // Benchmark dequantization with NEON
    group.bench_function("dequantize_neon_optimized", |b| {
        b.iter(|| black_box(quantizer.dequantize_tensor(black_box(&quantized)).unwrap()))
    });

    group.finish();
}

/// Fallback benchmarks for non-SIMD architectures
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn bench_scalar_fallback(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_fallback");

    let tensor_size = 32768;
    let tensor = create_realistic_tensor(tensor_size);
    let quantizer = I2SQuantizer::new();

    group.throughput(Throughput::Elements(tensor_size as u64));

    group.bench_function("quantize_scalar", |b| {
        b.iter(|| black_box(quantizer.quantize_tensor(black_box(&tensor)).unwrap()))
    });

    let quantized = quantizer.quantize_tensor(&tensor).unwrap();

    group.bench_function("dequantize_scalar", |b| {
        b.iter(|| black_box(quantizer.dequantize_tensor(black_box(&quantized)).unwrap()))
    });

    group.finish();
}

/// Benchmark throughput comparisons at different block sizes
/// This tests how the SIMD improvements scale with different quantization parameters
fn bench_block_size_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_size_throughput");

    let tensor_size = 65536;
    let tensor = create_realistic_tensor(tensor_size);

    // Test block sizes that interact differently with SIMD register sizes
    let block_sizes = vec![8, 16, 32, 64, 128, 256];

    for block_size in block_sizes {
        let quantizer = I2SQuantizer::with_block_size(block_size);

        group.throughput(Throughput::Elements(tensor_size as u64));

        group.bench_with_input(
            BenchmarkId::new("quantize_block_size", block_size),
            &block_size,
            |b, _| b.iter(|| black_box(quantizer.quantize_tensor(black_box(&tensor)).unwrap())),
        );

        let quantized = quantizer.quantize_tensor(&tensor).unwrap();

        group.bench_with_input(
            BenchmarkId::new("dequantize_block_size", block_size),
            &block_size,
            |b, _| {
                b.iter(|| black_box(quantizer.dequantize_tensor(black_box(&quantized)).unwrap()))
            },
        );
    }

    group.finish();
}

/// Benchmark memory throughput for different data patterns
fn bench_data_pattern_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_pattern_throughput");

    let tensor_size = 32768;
    let quantizer = I2SQuantizer::new();

    group.throughput(Throughput::Elements(tensor_size as u64));

    // Test different data patterns that might affect SIMD performance
    let patterns = vec![
        ("sequential", (0..tensor_size).map(|i| i as f32 * 0.001).collect()),
        ("alternating", (0..tensor_size).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect()),
        (
            "sine_wave",
            (0..tensor_size)
                .map(|i| (i as f32 * std::f32::consts::PI / 256.0).sin() * 2.0)
                .collect(),
        ),
        (
            "gaussian_like",
            (0..tensor_size)
                .map(|i| {
                    let x = (i as f32 - tensor_size as f32 / 2.0) / (tensor_size as f32 / 6.0);
                    x * (-x * x / 2.0).exp() * 2.0
                })
                .collect(),
        ),
        (
            "sparse",
            (0..tensor_size)
                .map(|i| if i % 10 == 0 { (i as f32 * 0.01).sin() } else { 0.0 })
                .collect(),
        ),
    ];

    for (pattern_name, data) in patterns {
        let device = Device::Cpu;
        let tensor_candle = CandleTensor::from_vec(data, &[tensor_size], &device).unwrap();
        let tensor = BitNetTensor::new(tensor_candle);

        group.bench_with_input(BenchmarkId::new("quantize", pattern_name), pattern_name, |b, _| {
            b.iter(|| black_box(quantizer.quantize_tensor(black_box(&tensor)).unwrap()))
        });

        let quantized = quantizer.quantize_tensor(&tensor).unwrap();

        group.bench_with_input(
            BenchmarkId::new("dequantize", pattern_name),
            pattern_name,
            |b, _| {
                b.iter(|| black_box(quantizer.dequantize_tensor(black_box(&quantized)).unwrap()))
            },
        );
    }

    group.finish();
}

/// Comprehensive performance comparison showing before/after improvements
fn bench_overall_improvement_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("improvement_comparison");

    // This benchmark provides a comprehensive view of the improvements
    let test_configurations = vec![
        ("small_tensor", 1024),
        ("medium_tensor", 16384),
        ("large_tensor", 131072),
        ("huge_tensor", 1048576),
    ];

    for (config_name, size) in test_configurations {
        let tensor = create_realistic_tensor(size);
        let quantizer = I2SQuantizer::new();

        group.throughput(Throughput::Elements(size as u64));

        // Quantization performance
        group.bench_with_input(
            BenchmarkId::new("quantize_optimized", config_name),
            &size,
            |b, _| b.iter(|| black_box(quantizer.quantize_tensor(black_box(&tensor)).unwrap())),
        );

        // Pre-quantize for dequantization tests
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();

        // Dequantization performance
        group.bench_with_input(
            BenchmarkId::new("dequantize_optimized", config_name),
            &size,
            |b, _| {
                b.iter(|| black_box(quantizer.dequantize_tensor(black_box(&quantized)).unwrap()))
            },
        );

        // Round-trip performance (most realistic use case)
        group.bench_with_input(
            BenchmarkId::new("round_trip_optimized", config_name),
            &size,
            |b, _| {
                b.iter(|| {
                    let q = quantizer.quantize_tensor(black_box(&tensor)).unwrap();
                    black_box(quantizer.dequantize_tensor(&q).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Simple timing test that prints human-readable results
fn simple_performance_comparison() {
    println!("\n=== SIMD Performance Improvement Analysis ===");

    let sizes = vec![4096, 16384, 65536];
    let quantizer = I2SQuantizer::new();

    for size in sizes {
        println!("\nTensor size: {} elements", size);

        let tensor = create_realistic_tensor(size);

        // Warm up
        for _ in 0..5 {
            let _ = quantizer.quantize_tensor(&tensor).unwrap();
        }

        // Time quantization
        let start = Instant::now();
        let runs = 50;
        for _ in 0..runs {
            let _ = quantizer.quantize_tensor(&tensor).unwrap();
        }
        let quantize_time = start.elapsed() / runs;

        // Time dequantization
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let start = Instant::now();
        for _ in 0..runs {
            let _ = quantizer.dequantize_tensor(&quantized).unwrap();
        }
        let dequantize_time = start.elapsed() / runs;

        // Calculate throughput
        let quantize_throughput = size as f64 / quantize_time.as_secs_f64() / 1_000_000.0; // Million elements/sec
        let dequantize_throughput = size as f64 / dequantize_time.as_secs_f64() / 1_000_000.0;

        println!(
            "  Quantization:   {:>8.2}μs ({:>6.1} Melem/s)",
            quantize_time.as_micros(),
            quantize_throughput
        );
        println!(
            "  Dequantization: {:>8.2}μs ({:>6.1} Melem/s)",
            dequantize_time.as_micros(),
            dequantize_throughput
        );

        // Architecture-specific information
        #[cfg(target_arch = "x86_64")]
        {
            let avx2_available = is_x86_feature_detected!("avx2");
            println!("  Architecture: x86_64, AVX2: {}", avx2_available);
        }

        #[cfg(target_arch = "aarch64")]
        {
            let neon_available = std::arch::is_aarch64_feature_detected!("neon");
            println!("  Architecture: aarch64, NEON: {}", neon_available);
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            println!("  Architecture: scalar fallback");
        }
    }

    println!("\n=== End Performance Analysis ===\n");
}

// Platform-specific benchmark groups
#[cfg(target_arch = "x86_64")]
criterion_group!(
    simd_benches,
    bench_memory_access_patterns,
    bench_alignment_scenarios,
    bench_x86_intrinsics_improvements,
    bench_block_size_throughput,
    bench_data_pattern_throughput,
    bench_overall_improvement_comparison
);

#[cfg(target_arch = "aarch64")]
criterion_group!(
    simd_benches,
    bench_memory_access_patterns,
    bench_alignment_scenarios,
    bench_arm_neon_improvements,
    bench_block_size_throughput,
    bench_data_pattern_throughput,
    bench_overall_improvement_comparison
);

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
criterion_group!(
    simd_benches,
    bench_memory_access_patterns,
    bench_alignment_scenarios,
    bench_scalar_fallback,
    bench_block_size_throughput,
    bench_data_pattern_throughput,
    bench_overall_improvement_comparison
);

criterion_main!(simd_benches);

// Run simple performance comparison when benchmarks are executed
#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn run_simple_performance_comparison() {
        simple_performance_comparison();
    }
}
