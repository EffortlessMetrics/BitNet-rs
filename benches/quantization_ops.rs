#![cfg(feature = "bench")]

//! Benchmarks for quantization operations across multiple algorithms:
//! - I2_S quantize/dequantize roundtrip (32-element blocks)
//! - QK256 block unpack and GEMV (256-element blocks)
//! - TL1/TL2 lookup-table quantize/dequantize

use bitnet_common::{BitNetTensor, Device};
use bitnet_quantization::i2s_qk256::{
    QK256_BLOCK, QK256_PACKED_BYTES, gemv_qk256, gemv_qk256_row, unpack_qk256_block,
};
use bitnet_quantization::{I2SQuantizer, TL1Quantizer, TL2Quantizer};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

/// Candle device used by TL1/TL2 quantizer APIs.
fn candle_cpu() -> candle_core::Device {
    candle_core::Device::Cpu
}

fn generate_data(size: usize) -> Vec<f32> {
    (0..size).map(|i| ((i as f32 / size as f32) * 2.0 - 1.0) * 0.95).collect()
}

/// Pack random 2-bit codes into QK256 byte layout for benchmarking.
fn make_packed_qk256(rows: usize, cols: usize) -> (Vec<u8>, usize) {
    let blocks_per_row = cols.div_ceil(QK256_BLOCK);
    let row_stride = blocks_per_row * QK256_PACKED_BYTES;
    let mut qs = vec![0u8; rows * row_stride];
    // Fill with a repeating pattern so the data isn't all-zero.
    for (i, b) in qs.iter_mut().enumerate() {
        *b = (i % 256) as u8;
    }
    (qs, row_stride)
}

// ── I2_S roundtrip ──────────────────────────────────────────────────────────

fn bench_i2s_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("i2s_roundtrip");

    for &size in &[256usize, 1024, 4096] {
        let data = generate_data(size);
        let tensor = BitNetTensor::from_slice(&data, &[size], &Device::Cpu).unwrap();
        let quantizer = I2SQuantizer::new();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("quantize", size), &size, |b, _| {
            b.iter(|| black_box(quantizer.quantize_tensor(&tensor).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("dequantize", size), &size, |b, _| {
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            b.iter(|| black_box(quantizer.dequantize_tensor(&quantized).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("roundtrip", size), &size, |b, _| {
            b.iter(|| {
                let q = quantizer.quantize_tensor(&tensor).unwrap();
                black_box(quantizer.dequantize_tensor(&q).unwrap())
            });
        });
    }

    group.finish();
}

// ── QK256 block operations ──────────────────────────────────────────────────

fn bench_qk256_unpack(c: &mut Criterion) {
    let mut group = c.benchmark_group("qk256_unpack");

    // Each block is 256 elements packed into 64 bytes.
    for &num_blocks in &[1usize, 4, 16] {
        let total_elems = num_blocks * QK256_BLOCK;
        group.throughput(Throughput::Elements(total_elems as u64));

        let packed: Vec<[u8; QK256_PACKED_BYTES]> = (0..num_blocks)
            .map(|b| {
                let mut block = [0u8; QK256_PACKED_BYTES];
                for (i, v) in block.iter_mut().enumerate() {
                    *v = ((b * QK256_PACKED_BYTES + i) % 256) as u8;
                }
                block
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("blocks", num_blocks), &num_blocks, |b, _| {
            b.iter(|| {
                let mut out = [0u8; QK256_BLOCK];
                for blk in &packed {
                    unpack_qk256_block(blk, &mut out);
                }
                black_box(out)
            });
        });
    }

    group.finish();
}

fn bench_qk256_gemv(c: &mut Criterion) {
    let mut group = c.benchmark_group("qk256_gemv");

    for &(rows, cols) in &[(1usize, 256usize), (4, 1024), (16, 4096)] {
        let total = rows * cols;
        group.throughput(Throughput::Elements(total as u64));

        let activations = generate_data(cols);
        let (packed, row_stride) = make_packed_qk256(rows, cols);

        if rows == 1 {
            // Single-row variant
            group.bench_with_input(BenchmarkId::new("single_row", cols), &cols, |b, _| {
                b.iter(|| black_box(gemv_qk256_row(&packed[..row_stride], &activations, cols)));
            });
        }

        group.bench_with_input(
            BenchmarkId::new("full", format!("{rows}x{cols}")),
            &total,
            |b, _| {
                b.iter(|| {
                    let mut out = vec![0.0f32; rows];
                    gemv_qk256(&packed, &activations, &mut out, rows, cols, row_stride).unwrap();
                    black_box(out)
                });
            },
        );
    }

    group.finish();
}

// ── TL1 / TL2 lookup operations ────────────────────────────────────────────

fn bench_tl1_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("tl1_lookup");
    let dev = candle_cpu();

    for &size in &[256usize, 1024, 4096] {
        let data = generate_data(size);
        let tensor = BitNetTensor::from_slice(&data, &[size], &Device::Cpu).unwrap();
        let quantizer = TL1Quantizer::new();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("quantize", size), &size, |b, _| {
            b.iter(|| black_box(quantizer.quantize(&tensor, &dev).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("roundtrip", size), &size, |b, _| {
            let quantized = quantizer.quantize(&tensor, &dev).unwrap();
            b.iter(|| black_box(quantizer.dequantize(&quantized, &dev).unwrap()));
        });
    }

    group.finish();
}

fn bench_tl2_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("tl2_lookup");
    let dev = candle_cpu();

    for &size in &[256usize, 1024, 4096] {
        let data = generate_data(size);
        let tensor = BitNetTensor::from_slice(&data, &[size], &Device::Cpu).unwrap();
        let quantizer = TL2Quantizer::new();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("quantize", size), &size, |b, _| {
            b.iter(|| black_box(quantizer.quantize(&tensor, &dev).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("roundtrip", size), &size, |b, _| {
            let quantized = quantizer.quantize(&tensor, &dev).unwrap();
            b.iter(|| black_box(quantizer.dequantize(&quantized, &dev).unwrap()));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_i2s_roundtrip,
    bench_qk256_unpack,
    bench_qk256_gemv,
    bench_tl1_lookup,
    bench_tl2_lookup,
);
criterion_main!(benches);
