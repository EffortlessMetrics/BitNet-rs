//! Example of basic inference with BitNet models

use bitnet_common::{BitNetConfig, Device};
use bitnet_inference::{backends::CpuBackend, GenerationConfig, InferenceConfig, InferenceEngine};
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
    let model: Arc<dyn Model> = Arc::new(BitNetModel::new(model_config, device.clone()));

    // Create backend
    let _cpu_backend = CpuBackend::new(model.clone())?;

    // Create tokenizer
    let tokenizer = TokenizerBuilder::from_pretrained("gpt2")?;

    // Create inference engine
    let _inference_config = InferenceConfig::default();
    let engine = InferenceEngine::new(model, tokenizer, device)?;

    // Generate text
    let prompt = "Hello, how are you?";
    let generation_config = GenerationConfig {
        max_new_tokens: 50,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        ..Default::default()
    };

    println!("Prompt: {}", prompt);
    println!("Generating response...");

    let response = engine.generate_with_config(prompt, &generation_config).await?;

    println!("Response: {}", response);

    Ok(())
}
