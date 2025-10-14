#![cfg(feature = "bench")]

use bitnet_common::{BitNetConfig, Device};
use bitnet_inference::engine::InferenceEngine;
use bitnet_models::{BitNetModel, Model};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::Arc;

fn benchmark_inference(c: &mut Criterion) {
    let device = Device::Cpu;
    let config = BitNetConfig::default();
    let model_impl = BitNetModel::new(config, device);
    let model: Arc<dyn Model> = Arc::new(model_impl);

    // Create a dummy tokenizer for the benchmark
    // In real usage, you'd load an actual tokenizer
    let tokenizer: Arc<dyn bitnet::prelude::Tokenizer> = Arc::new(DummyTokenizer);

    let engine = InferenceEngine::new(model, tokenizer, device).expect("Failed to create engine");

    c.bench_function("inference_basic", |b| {
        b.iter(|| {
            // Use proper inference parameters
            let params = bitnet_inference::InferenceParams::default();
            let result = engine.generate(black_box("Hello, world!"), params);
            black_box(result)
        })
    });
}

// Dummy tokenizer for benchmarking purposes
struct DummyTokenizer;

impl bitnet::prelude::Tokenizer for DummyTokenizer {
    fn encode(
        &self,
        text: &str,
        _add_special_tokens: bool,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        Ok(text.bytes().map(|b| b as u32).collect())
    }

    fn decode(
        &self,
        tokens: &[u32],
        _skip_special_tokens: bool,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let bytes: Vec<u8> = tokens.iter().map(|&t| t as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        token.bytes().next().map(|b| b as u32)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        Some(String::from_utf8_lossy(&[id as u8]).to_string())
    }

    fn pad_token_id(&self) -> Option<u32> {
        Some(0)
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(0)
    }

    fn bos_token_id(&self) -> Option<u32> {
        Some(1)
    }
}

criterion_group!(benches, benchmark_inference);
criterion_main!(benches);
