//! Benchmarks for quantization kernels
//!
//! This benchmark compares the performance of scalar vs SIMD implementations
//! for I2S, TL1, and TL2 quantization methods.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use bitnet_quantization::{I2SQuantizer, TL1Quantizer, TL2Quantizer, QuantizerTrait};
use bitnet_common::{BitNetTensor, Device};

fn generate_test_data(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| ((i as f32 / size as f32) * 2.0 - 1.0) * 0.95)
        .collect()
}

fn bench_i2s_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("i2s_quantization");
    
    for size in [1024, 4096, 16384, 65536].iter() {
        let data = generate_test_data(*size);
        let tensor = BitNetTensor::from_slice(&data, &[*size], &Device::Cpu).unwrap();
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("quantize", size), size, |b, _| {
            let quantizer = I2SQuantizer::new();
            b.iter(|| {
                let result = quantizer.quantize_tensor(&tensor).unwrap();
                black_box(result);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("round_trip", size), size, |b, _| {
            let quantizer = I2SQuantizer::new();
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            b.iter(|| {
                let result = quantizer.dequantize_tensor(&quantized).unwrap();
                black_box(result);
            });
        });
    }
    
    group.finish();
}

fn bench_tl1_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("tl1_quantization");
    
    for size in [1024, 4096, 16384, 65536].iter() {
        let data = generate_test_data(*size);
        let tensor = BitNetTensor::from_slice(&data, &[*size], &Device::Cpu).unwrap();
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("quantize", size), size, |b, _| {
            let quantizer = TL1Quantizer::new();
            b.iter(|| {
                let result = quantizer.quantize_tensor(&tensor).unwrap();
                black_box(result);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("round_trip", size), size, |b, _| {
            let quantizer = TL1Quantizer::new();
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            b.iter(|| {
                let result = quantizer.dequantize_tensor(&quantized).unwrap();
                black_box(result);
            });
        });
    }
    
    group.finish();
}

fn bench_tl2_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("tl2_quantization");
    
    for size in [1024, 4096, 16384, 65536].iter() {
        let data = generate_test_data(*size);
        let tensor = BitNetTensor::from_slice(&data, &[*size], &Device::Cpu).unwrap();
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("quantize", size), size, |b, _| {
            let quantizer = TL2Quantizer::new();
            b.iter(|| {
                let result = quantizer.quantize_tensor(&tensor).unwrap();
                black_box(result);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("round_trip", size), size, |b, _| {
            let quantizer = TL2Quantizer::new();
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            b.iter(|| {
                let result = quantizer.dequantize_tensor(&quantized).unwrap();
                black_box(result);
            });
        });
    }
    
    group.finish();
}

fn bench_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar");
    let size = 16384;
    let data = generate_test_data(size);
    let tensor = BitNetTensor::from_slice(&data, &[size], &Device::Cpu).unwrap();
    
    group.throughput(Throughput::Elements(size as u64));
    
    // I2S with SIMD enabled (default)
    group.bench_function("i2s_simd", |b| {
        let quantizer = I2SQuantizer::new();
        b.iter(|| {
            let result = quantizer.quantize_tensor(&tensor).unwrap();
            black_box(result);
        });
    });
    
    // I2S with SIMD disabled (would need feature flag)
    // This would require a way to disable SIMD at runtime
    
    #[cfg(target_arch = "aarch64")]
    group.bench_function("tl1_neon", |b| {
        let quantizer = TL1Quantizer::new();
        b.iter(|| {
            let result = quantizer.quantize_tensor(&tensor).unwrap();
            black_box(result);
        });
    });
    
    #[cfg(target_arch = "x86_64")]
    group.bench_function("tl2_avx2", |b| {
        let quantizer = TL2Quantizer::new();
        b.iter(|| {
            let result = quantizer.quantize_tensor(&tensor).unwrap();
            black_box(result);
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_i2s_quantization,
    bench_tl1_quantization,
    bench_tl2_quantization,
    bench_simd_vs_scalar
);
criterion_main!(benches);