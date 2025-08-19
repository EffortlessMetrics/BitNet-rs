use bitnet_common::BitNetConfig;
use bitnet_inference::InferenceEngine;
use bitnet_models::BitNetModel;
use candle_core::Device;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn benchmark_inference(c: &mut Criterion) {
    let device = Device::Cpu;
    let config = BitNetConfig::default();
    let model = Box::new(BitNetModel::new(config, device));
    let mut engine = InferenceEngine::new(model);

    c.bench_function("inference_basic", |b| {
        b.iter(|| {
            let result = engine.generate(black_box("Hello, world!"));
            black_box(result)
        })
    });
}

criterion_group!(benches, benchmark_inference);
criterion_main!(benches);
