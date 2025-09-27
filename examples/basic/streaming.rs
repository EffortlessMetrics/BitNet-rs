//! Streaming generation example using BitNet.rs
//!
//! This example demonstrates how to use streaming generation for real-time text output.
//!
//! Features demonstrated:
//! - Real-time token streaming with `StreamResponse`
//! - Access to token IDs alongside generated text (added in v0.1.0)
//! - Generation statistics and performance monitoring
//! - Interactive prompt handling

use bitnet::prelude::*;
use futures::StreamExt;
use std::env;
use std::io::{self, Write};
use std::sync::Arc;

#[cfg(feature = "examples")]
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
    let model: Arc<dyn Model> = Arc::new(loader.load(model_path)?);
    println!("Model loaded successfully");

    // Create a tokenizer (you'll need to provide your tokenizer implementation)
    // For this example, we'll use a placeholder
    // In real usage, load the tokenizer from a file or configuration
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(create_tokenizer()?);

    // Create inference engine with Arc-wrapped model and tokenizer
    let engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)?;
    println!("Inference engine created");

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

        // Configure generation parameters for streaming
        let config = GenerationConfig {
            max_new_tokens: 200,
            temperature: 0.7,
            top_p: Some(0.9),
            top_k: Some(40),
            repetition_penalty: 1.1,
            do_sample: true,
            seed: None,
        };

        // Create streaming generation
        let mut stream = engine.generate_stream_with_config(prompt, &config);
        let mut full_response = String::new();
        let start_time = std::time::Instant::now();
        let mut token_count = 0;

        // Process tokens as they arrive
        while let Some(token_result) = stream.next().await {
            match token_result {
                Ok(stream_response) => {
                    // Display the generated text
                    print!("{}", stream_response.text);
                    io::stdout().flush()?;
                    full_response.push_str(&stream_response.text);

                    // Access token IDs for debugging or analysis (new in v0.1.0)
                    if !stream_response.token_ids.is_empty() {
                        eprintln!(
                            "[DEBUG] Generated {} token(s) with IDs: {:?}",
                            stream_response.token_ids.len(),
                            stream_response.token_ids
                        );
                    }

                    token_count += stream_response.token_ids.len();
                }
                Err(e) => {
                    eprintln!("\nError during generation: {}", e);
                    break;
                }
            }
        }

        let elapsed = start_time.elapsed();
        let tokens_per_second = if elapsed.as_secs_f64() > 0.0 {
            token_count as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        println!("\n");
        println!("--- Generation Statistics ---");
        println!("Tokens generated: {}", token_count);
        println!("Generation time: {:.2}s", elapsed.as_secs_f64());
        println!("Tokens per second: {:.2}", tokens_per_second);
        println!("Total response length: {} characters", full_response.len());
    }

    println!("Goodbye!");
    Ok(())
}

// Placeholder function - replace with actual tokenizer loading
fn create_tokenizer() -> Result<impl Tokenizer, Box<dyn std::error::Error>> {
    // This should load your actual tokenizer
    // For now, return a dummy implementation
    struct DummyTokenizer;

    impl Tokenizer for DummyTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
            // Simple character-based encoding for demonstration
            Ok(text.chars().map(|c| c as u32).collect())
        }

        fn decode(&self, tokens: &[u32]) -> Result<String, Box<dyn std::error::Error>> {
            // Simple character-based decoding for demonstration
            Ok(tokens.iter().map(|&t| char::from_u32(t).unwrap_or('?')).collect())
        }

        fn vocab_size(&self) -> usize {
            65536 // Unicode BMP size
        }
    }

    Ok(DummyTokenizer)
}

#[cfg(not(feature = "examples"))]
fn main() {}
