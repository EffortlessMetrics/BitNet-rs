use bitnet_common::QuantizationType;
use bitnet_kernels::{KernelManager, KernelProvider};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn benchmark_kernels(c: &mut Criterion) {
    let mut manager = KernelManager::new();
    let kernel = manager.select_best().unwrap();

    let a = vec![1i8; 1024];
    let b = vec![1u8; 1024];
    let mut c_out = vec![0.0f32; 1024];

    c.bench_function("matmul_i2s", |b| {
        b.iter(|| {
            let result =
                kernel.matmul_i2s(black_box(&a), black_box(&b), black_box(&mut c_out), 32, 32, 32);
            black_box(result)
        })
    });

    let input = vec![1.0f32; 1024];
    let mut output = vec![0u8; 256];
    let mut scales = vec![0.0f32; 32];

    c.bench_function("quantize_i2s", |b| {
        b.iter(|| {
            let result = kernel.quantize(
                black_box(&input),
                black_box(&mut output),
                black_box(&mut scales),
                QuantizationType::I2S,
            );
            black_box(result)
        })
    });
}

criterion_group!(benches, benchmark_kernels);
criterion_main!(benches);
