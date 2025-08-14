//! Streaming generation example using BitNet.rs
//!
//! This example demonstrates how to use streaming generation for real-time text output.

use bitnet::prelude::*;
use futures::StreamExt;
use std::env;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    // Create device
    let device = Device::Cpu;
    println!("Using device: {:?}", device);

    // Load the model
    let loader = ModelLoader::new(device.clone());
    let model = loader.load(model_path)?;
    println!("Model loaded successfully");

    // Create inference engine
    let mut engine = InferenceEngine::new(model)?;
    println!("Inference engine created");

    // Configure generation parameters for streaming
    let config = GenerationConfig {
        max_new_tokens: 200,
        temperature: 0.7,
        top_p: 0.9,
        top_k: Some(40),
        repetition_penalty: 1.1,
        stream_tokens: true, // Enable token-by-token streaming
        ..Default::default()
    };
    engine.set_generation_config(config);

    // Interactive loop
    loop {
        print!("\nEnter a prompt (or 'quit' to exit): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let prompt = input.trim();

        if prompt.is_empty() {
            continue;
        }

        if prompt.eq_ignore_ascii_case("quit") {
            break;
        }

        println!("\nPrompt: {}", prompt);
        print!("Response: ");
        io::stdout().flush()?;

        // Create streaming generation
        let mut stream = engine.generate_stream(prompt).await?;
        let mut full_response = String::new();

        // Process tokens as they arrive
        while let Some(token_result) = stream.next().await {
            match token_result {
                Ok(token) => {
                    print!("{}", token);
                    io::stdout().flush()?;
                    full_response.push_str(&token);
                }
                Err(e) => {
                    eprintln!("\nError during generation: {}", e);
                    break;
                }
            }
        }

        println!("\n");
        println!("--- Generation Statistics ---");
        let stats = engine.get_last_generation_stats()?;
        println!("Tokens generated: {}", stats.tokens_generated);
        println!(
            "Generation time: {:.2}s",
            stats.generation_time_ms as f64 / 1000.0
        );
        println!("Tokens per second: {:.2}", stats.tokens_per_second);
        println!("Total response length: {} characters", full_response.len());
    }

    println!("Goodbye!");
    Ok(())
}
