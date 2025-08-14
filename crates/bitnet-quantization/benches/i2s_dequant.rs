use bitnet_common::QuantizationType;
use bitnet_quantization::{I2SLayout, I2SQuantizer, QuantizedTensor};
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn bench_i2s_dequant(c: &mut Criterion) {
    let layout = I2SLayout::default();
    let blocks = 8_192;
    let packed = vec![0u8; blocks * layout.data_bytes_per_block];
    let scales = vec![1.0f32; blocks];
    
    let qt = QuantizedTensor::new_with_params(
        packed,
        scales,
        None,
        vec![layout.block_size * blocks],
        QuantizationType::I2S,
        layout.block_size,
    );
    
    let quantizer = I2SQuantizer::with_block_size(layout.block_size);
    
    c.bench_function("i2s_dequant_8k_blocks", |b| {
        b.iter(|| {
            let tensor = quantizer.dequantize_tensor(black_box(&qt)).unwrap();
            black_box(tensor);
        })
    });
}

criterion_group!(benches, bench_i2s_dequant);
criterion_main!(benches);