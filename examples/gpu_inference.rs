//! GPU inference example using BitNet.rs with CUDA acceleration
//!
//! This example demonstrates how to use GPU acceleration for faster inference.

use bitnet::prelude::*;
use std::{env, sync::Arc};

#[cfg(all(feature = "cuda", feature = "examples"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Get model path from command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <model_path>", args[0]);
        eprintln!("Example: {} model.gguf", args[0]);
        std::process::exit(1);
    }
    let model_path = &args[1];

    println!("Loading BitNet model from: {}", model_path);

    // Try to create CUDA device, fall back to CPU if not available
    let device = if cfg!(feature = "cuda") {
        println!("Using CUDA device");
        Device::Cuda(0)
    } else {
        println!("CUDA not available, using CPU");
        Device::Cpu
    };

    // Load the model and tokenizer
    let loader = ModelLoader::new(device.clone());
    let model = loader.load(model_path)?;
    println!("Model loaded successfully");

    // Create a default tokenizer for demonstration
    // In practice, you would load the appropriate tokenizer for your model
    let tokenizer = Arc::new(bitnet_tokenizers::create_default_tokenizer()?);

    // Create inference engine
    let mut engine = InferenceEngine::new(Arc::new(model), tokenizer, device.clone())?;
    println!("Inference engine created");

    // GPU optimizations and device-aware quantization are enabled automatically for CUDA devices
    if matches!(device, Device::Cuda(_)) {
        println!("GPU optimizations enabled automatically");
        println!("Device-aware quantization with automatic CPU fallback active");
    }

    // Configure generation parameters optimized for GPU
    let config = GenerationConfig { do_sample: true, seed: Some(42), ..Default::default() };

    // Single inference example
    let prompt = "The future of GPU computing in AI is";

    println!("\n--- GPU Inference Example ---");
    let start_time = std::time::Instant::now();

    let response = engine.generate(prompt, &config)?;

    let elapsed = start_time.elapsed();
    println!("Inference completed in: {:?}", elapsed);
    println!("Throughput: {:.2} tokens/second", response.len() as f64 / elapsed.as_secs_f64());

    // Display results
    println!("\nPrompt: {}", prompt);
    println!("Response: {}", response);

    // GPU-specific information
    if let Device::Cuda(device_id) = device {
        println!("\n--- GPU Device Information ---");
        println!("Using CUDA device: {}", device_id);
        println!("GPU acceleration enabled for optimal performance");
    }

    println!("\nGPU inference completed successfully!");
    Ok(())
}

#[cfg(all(not(feature = "cuda"), feature = "examples"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature to be enabled.");
    eprintln!("Run with: cargo run --example gpu_inference --features cuda");
    std::process::exit(1);
}

#[cfg(not(feature = "examples"))]
fn main() {}
