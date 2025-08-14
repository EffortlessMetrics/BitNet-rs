//! GPU inference example using BitNet.rs with CUDA acceleration
//!
//! This example demonstrates how to use GPU acceleration for faster inference.

use bitnet::prelude::*;
use std::env;

#[cfg(feature = "gpu")]
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
    let device = match Device::new_cuda(0) {
        Ok(device) => {
            println!("Using CUDA device: {:?}", device);
            device
        }
        Err(e) => {
            println!("CUDA not available ({}), falling back to CPU", e);
            Device::Cpu
        }
    };

    // Load the model
    let loader = ModelLoader::new(device.clone());
    let model = loader.load(model_path)?;
    println!("Model loaded successfully");

    // Create inference engine with GPU optimization
    let mut engine = InferenceEngine::new(model)?;

    // Enable GPU-specific optimizations if available
    if device.is_cuda() {
        engine.enable_gpu_optimizations(true)?;
        println!("GPU optimizations enabled");
    }

    // Configure generation parameters optimized for GPU
    let config = GenerationConfig {
        max_new_tokens: 100,
        temperature: 0.8,
        top_p: 0.95,
        top_k: Some(50),
        repetition_penalty: 1.05,
        batch_size: 4, // GPU can handle larger batches efficiently
        ..Default::default()
    };
    engine.set_generation_config(config);

    // Batch inference example
    let prompts = vec![
        "The future of GPU computing in AI is",
        "CUDA acceleration enables",
        "High-performance inference requires",
        "Parallel processing allows us to",
    ];

    println!("\n--- Batch Inference ---");
    let start_time = std::time::Instant::now();

    let responses = engine.generate_batch(&prompts)?;

    let elapsed = start_time.elapsed();
    println!("Batch inference completed in: {:?}", elapsed);
    println!(
        "Throughput: {:.2} tokens/second",
        (responses.iter().map(|r| r.len()).sum::<usize>() as f64) / elapsed.as_secs_f64()
    );

    // Display results
    for (prompt, response) in prompts.iter().zip(responses.iter()) {
        println!("\nPrompt: {}", prompt);
        println!("Response: {}", response);
    }

    // Memory usage statistics
    if device.is_cuda() {
        let memory_info = engine.get_memory_info()?;
        println!("\n--- GPU Memory Usage ---");
        println!("Allocated: {} MB", memory_info.allocated_mb);
        println!("Cached: {} MB", memory_info.cached_mb);
        println!("Reserved: {} MB", memory_info.reserved_mb);
    }

    println!("\nGPU inference completed successfully!");
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("This example requires the 'gpu' feature to be enabled.");
    eprintln!("Run with: cargo run --example gpu_inference --features gpu");
    std::process::exit(1);
}
