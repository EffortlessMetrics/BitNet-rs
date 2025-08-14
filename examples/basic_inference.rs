//! Basic inference example using BitNet.rs
//!
//! This example demonstrates how to load a BitNet model and perform basic text generation.

use bitnet::prelude::*;
use std::env;

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

    // Create device (CPU for this example)
    let device = Device::Cpu;
    println!("Using device: {:?}", device);

    // Load the model
    let loader = ModelLoader::new(device.clone());
    let model = loader.load(model_path)?;
    println!("Model loaded successfully");

    // Create inference engine
    let mut engine = InferenceEngine::new(model)?;
    println!("Inference engine created");

    // Configure generation parameters
    let config = GenerationConfig {
        max_new_tokens: 50,
        temperature: 0.7,
        top_p: 0.9,
        top_k: Some(40),
        repetition_penalty: 1.1,
        ..Default::default()
    };
    engine.set_generation_config(config);

    // Example prompts
    let prompts = vec![
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "The most important lesson I learned today was",
    ];

    // Generate text for each prompt
    for (i, prompt) in prompts.iter().enumerate() {
        println!("\n--- Example {} ---", i + 1);
        println!("Prompt: {}", prompt);
        println!("Response: ");

        let response = engine.generate(prompt)?;
        println!("{}", response);
    }

    println!("\nInference completed successfully!");
    Ok(())
}
