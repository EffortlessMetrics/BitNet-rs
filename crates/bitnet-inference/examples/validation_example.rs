//! Example of basic inference with BitNet models

use bitnet_common::{BitNetConfig, Device};
use bitnet_inference::{GenerationConfig, InferenceConfig, InferenceEngine, backends::CpuBackend};
use bitnet_models::{BitNetModel, Model};
use bitnet_tokenizers::TokenizerBuilder;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("BitNet Rust Inference Example");

    // Create model configuration
    let model_config = BitNetConfig::default();
    let device = Device::Cpu;

    // Create model
    let model: Arc<dyn Model> = Arc::new(BitNetModel::new(model_config, device));

    // Create backend
    let _cpu_backend = CpuBackend::new(model.clone())?;

    // Create tokenizer
    let tokenizer = TokenizerBuilder::from_pretrained("gpt2")?;

    // Create inference engine
    let _inference_config = InferenceConfig::default();
    let engine = InferenceEngine::new(model, tokenizer, device)?;

    // Generate text
    let prompt = "Hello, how are you?";
    let generation_config = GenerationConfig::default()
        .with_max_tokens(50)
        .with_temperature(0.7)
        .with_top_p(0.9)
        .with_top_k(50);

    println!("Prompt: {}", prompt);
    println!("Generating response...");

    let response = engine.generate_with_config(prompt, &generation_config).await?;

    println!("Response: {}", response);

    Ok(())
}
