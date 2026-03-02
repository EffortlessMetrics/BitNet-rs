//! Benchmarks for quantization and dequantization operations.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_int8_quantize(c: &mut Criterion) {
    let data: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.001 - 2.0).collect();

    c.bench_function("int8_quantize_4096", |b| {
        b.iter(|| {
            let scale = data.iter().fold(0.0f32, |a, &v| a.max(v.abs())) / 127.0;
            let _quantized: Vec<i8> =
                data.iter().map(|&v| (v / scale).round().clamp(-128.0, 127.0) as i8).collect();
            black_box(scale)
        })
    });
}

fn bench_int8_dequantize(c: &mut Criterion) {
    let quantized: Vec<i8> = (0..4096).map(|i| (i % 256) as i8).collect();
    let scale = 0.015625f32;

    c.bench_function("int8_dequantize_4096", |b| {
        b.iter(|| {
            let _output: Vec<f32> = quantized.iter().map(|&v| v as f32 * scale).collect();
            black_box(())
        })
    });
}

fn bench_int4_quantize(c: &mut Criterion) {
    let data: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.001 - 2.0).collect();

    c.bench_function("int4_quantize_4096", |b| {
        b.iter(|| {
            let scale = data.iter().fold(0.0f32, |a, &v| a.max(v.abs())) / 7.0;
            let _quantized: Vec<u8> = data
                .chunks(2)
                .map(|chunk| {
                    let lo = ((chunk[0] / scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
                    let hi = if chunk.len() > 1 {
                        ((chunk[1] / scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8
                    } else {
                        8
                    };
                    (hi << 4) | (lo & 0x0F)
                })
                .collect();
            black_box(scale)
        })
    });
}

fn bench_int4_dequantize(c: &mut Criterion) {
    let packed: Vec<u8> = (0..2048).map(|i| (i & 0xFF) as u8).collect();
    let scale = 0.28571f32;

    c.bench_function("int4_dequantize_4096", |b| {
        b.iter(|| {
            let _output: Vec<f32> = packed
                .iter()
                .flat_map(|&byte| {
                    let lo = (byte & 0x0F) as i8 - 8;
                    let hi = ((byte >> 4) & 0x0F) as i8 - 8;
                    [lo as f32 * scale, hi as f32 * scale]
                })
                .collect();
            black_box(())
        })
    });
}

fn bench_bf16_to_f32(c: &mut Criterion) {
    // Simulate BF16 values stored as u16
    let bf16_data: Vec<u16> = (0..4096).map(|i| (0x3F80 + (i & 0xFF)) as u16).collect();

    c.bench_function("bf16_to_f32_4096", |b| {
        b.iter(|| {
            let _output: Vec<f32> = bf16_data
                .iter()
                .map(|&bits| {
                    let f32_bits = (bits as u32) << 16;
                    f32::from_bits(f32_bits)
                })
                .collect();
            black_box(())
        })
    });
}

fn bench_f32_to_f16(c: &mut Criterion) {
    let data: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.001).collect();

    c.bench_function("f32_to_f16_4096", |b| {
        b.iter(|| {
            let _output: Vec<u16> = data
                .iter()
                .map(|&v| {
                    let bits = v.to_bits();
                    let sign = (bits >> 16) & 0x8000;
                    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
                    let mant = (bits >> 13) & 0x3FF;
                    if exp <= 0 {
                        sign as u16
                    } else if exp >= 31 {
                        (sign | 0x7C00) as u16
                    } else {
                        (sign | ((exp as u32) << 10) | mant) as u16
                    }
                })
                .collect();
            black_box(())
        })
    });
}

fn bench_i2s_encode(c: &mut Criterion) {
    let ternary: Vec<i8> = (0..4096)
        .map(|i| match i % 3 {
            0 => 0,
            1 => 1,
            _ => -1,
        })
        .collect();

    c.bench_function("i2s_encode_4096", |b| {
        b.iter(|| {
            let _packed: Vec<u8> = ternary
                .chunks(4)
                .map(|chunk| {
                    let mut byte = 0u8;
                    for (j, &val) in chunk.iter().enumerate() {
                        let bits: u8 = match val {
                            0 => 0b00,
                            1 => 0b01,
                            -1 => 0b11,
                            _ => 0b00,
                        };
                        byte |= bits << (j * 2);
                    }
                    byte
                })
                .collect();
            black_box(())
        })
    });
}

fn bench_i2s_decode(c: &mut Criterion) {
    let packed: Vec<u8> = (0..1024).map(|i| (i & 0xFF) as u8).collect();

    c.bench_function("i2s_decode_4096", |b| {
        b.iter(|| {
            let _output: Vec<i8> = packed
                .iter()
                .flat_map(|&byte| {
                    (0..4).map(move |j| match (byte >> (j * 2)) & 0b11 {
                        0b00 => 0i8,
                        0b01 => 1,
                        0b11 => -1,
                        _ => 0,
                    })
                })
                .collect();
            black_box(())
        })
    });
}

criterion_group!(
    benches,
    bench_int8_quantize,
    bench_int8_dequantize,
    bench_int4_quantize,
    bench_int4_dequantize,
    bench_bf16_to_f32,
    bench_f32_to_f16,
    bench_i2s_encode,
    bench_i2s_decode,
);
criterion_main!(benches);
