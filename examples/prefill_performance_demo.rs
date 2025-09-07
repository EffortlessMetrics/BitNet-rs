//! Prefill Performance Demonstration Example
//!
//! This example demonstrates the explicit prefill functionality introduced in PR #187,
//! showcasing:
//! - Explicit cache warming with `engine.prefill()`
//! - Structured performance metrics (TimingMetrics, ThroughputMetrics)
//! - Batch processing with prefill optimization
//! - JSON export of comprehensive performance data
//!
//! Usage: cargo run --example prefill_performance_demo --features cpu -- <model_path>

use bitnet::prelude::*;
use futures::StreamExt;
use serde_json;
use std::env;
use std::sync::Arc;
use std::time::Instant;

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

    println!("🚀 BitNet.rs Prefill Performance Demonstration");
    println!("Model: {}", model_path);
    println!("Features: Explicit prefill, structured metrics, batch optimization");
    println!("{}", "=".repeat(60));

    // Load model and create inference engine
    let device = Device::Cpu;
    let loader = ModelLoader::new(device.clone());
    let model: Arc<dyn Model> = Arc::new(loader.load(model_path)?);

    // Create tokenizer (mock for this example)
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(create_mock_tokenizer()?);

    // Create inference engine
    let mut engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)?;

    // Test prompts for different scenarios
    let test_prompts = vec![
        ("Short", "Hello"),
        ("Medium", "The future of artificial intelligence will likely include"),
        (
            "Long",
            "In a comprehensive analysis of modern technological advancement, we must consider the multifaceted implications of artificial intelligence, machine learning, and their profound impact on society, economics, and human interaction patterns across diverse global contexts",
        ),
    ];

    println!("\n📊 PREFILL PERFORMANCE ANALYSIS");
    println!("{}", "-".repeat(60));

    for (category, prompt) in &test_prompts {
        println!(
            "\n🔍 Testing {} prompt: \"{}...\"",
            category,
            &prompt.chars().take(40).collect::<String>()
        );

        // Tokenize the prompt
        let tokenize_start = Instant::now();
        let prompt_tokens = tokenizer.encode(prompt, true, true)?;
        let tokenize_duration = tokenize_start.elapsed();

        println!(
            "   Tokens: {} | Tokenize time: {:.2}ms",
            prompt_tokens.len(),
            tokenize_duration.as_secs_f64() * 1000.0
        );

        // Explicit prefill with timing
        let prefill_start = Instant::now();
        engine.prefill(&prompt_tokens).await?;
        let prefill_duration = prefill_start.elapsed();

        // Calculate prefill throughput
        let prefill_throughput = prompt_tokens.len() as f64 / prefill_duration.as_secs_f64();

        println!(
            "   ⚡ Prefill: {:.2}ms | {:.1} tokens/sec",
            prefill_duration.as_secs_f64() * 1000.0,
            prefill_throughput
        );

        // Generate with performance tracking
        let generation_config = GenerationConfig {
            max_new_tokens: 20,
            temperature: 0.7,
            enable_metrics: true,
            ..Default::default()
        };

        let generation_start = Instant::now();
        let response = engine.generate_with_config(prompt, &generation_config).await?;
        let generation_duration = generation_start.elapsed();

        println!(
            "   🎯 Generation: {:.2}ms | Output: \"{}...\"",
            generation_duration.as_secs_f64() * 1000.0,
            &response.text.chars().take(30).collect::<String>()
        );

        // Display structured performance metrics if available
        if let Some(metrics) = &response.metrics {
            println!("   📈 Detailed Metrics:");
            println!("      - Tokenize: {:.2}ms", metrics.timing.tokenize);
            println!("      - Prefill: {:.2}ms", metrics.timing.prefill);
            println!("      - Decode: {:.2}ms", metrics.timing.decode);
            println!("      - Total: {:.2}ms", metrics.timing.total);
            println!("      - Prefill TPS: {:.1}", metrics.throughput.prefill);
            println!("      - Decode TPS: {:.1}", metrics.throughput.decode);
            println!("      - E2E TPS: {:.1}", metrics.throughput.e2e);
        }
    }

    // Batch processing demonstration
    println!("\n🔄 BATCH PROCESSING WITH PREFILL");
    println!("{}", "-".repeat(60));

    let batch_prompts = vec![
        "Explain quantum computing",
        "Describe machine learning",
        "What is artificial intelligence",
    ];

    let batch_start = Instant::now();
    let mut batch_results = Vec::new();

    for (i, prompt) in batch_prompts.iter().enumerate() {
        println!(
            "\n   Batch item {}: Processing \"{}...\"",
            i + 1,
            &prompt.chars().take(25).collect::<String>()
        );

        // Tokenize and prefill
        let tokens = tokenizer.encode(prompt, true, true)?;
        engine.prefill(&tokens).await?;

        // Generate
        let config = GenerationConfig {
            max_new_tokens: 15,
            temperature: 0.7,
            enable_metrics: true,
            ..Default::default()
        };

        let result = engine.generate_with_config(prompt, &config).await?;
        batch_results.push(result);

        println!(
            "     ✅ Completed: \"{}...\"",
            &batch_results.last().unwrap().text.chars().take(40).collect::<String>()
        );
    }

    let batch_duration = batch_start.elapsed();
    println!("\n   🎯 Batch Summary:");
    println!("      - Total time: {:.2}ms", batch_duration.as_secs_f64() * 1000.0);
    println!("      - Items processed: {}", batch_results.len());
    println!(
        "      - Avg per item: {:.2}ms",
        batch_duration.as_secs_f64() * 1000.0 / batch_results.len() as f64
    );

    // Export comprehensive metrics
    if let Some(metrics) = &batch_results.last().unwrap().metrics {
        let json_metrics = serde_json::to_string_pretty(&metrics)?;
        std::fs::write("prefill_performance_metrics.json", &json_metrics)?;
        println!("\n💾 Metrics exported to: prefill_performance_metrics.json");
        println!("   Sample metrics structure:");
        println!("{}", json_metrics.lines().take(10).collect::<Vec<_>>().join("\n"));
        println!("   ...");
    }

    println!("\n🎉 Prefill performance demonstration completed!");
    println!("Key benefits demonstrated:");
    println!("  ✅ Explicit prefill control for cache warming");
    println!("  ✅ Precise timing measurement at each stage");
    println!("  ✅ Structured performance metrics export");
    println!("  ✅ Batch processing optimization");

    Ok(())
}

#[cfg(feature = "examples")]
fn create_mock_tokenizer() -> Result<impl Tokenizer, Box<dyn std::error::Error>> {
    // Mock tokenizer implementation for demonstration
    // In real usage, use TokenizerBuilder::from_file() or similar
    Ok(MockTokenizer::new())
}

#[cfg(feature = "examples")]
struct MockTokenizer;

#[cfg(feature = "examples")]
impl MockTokenizer {
    fn new() -> Self {
        Self
    }
}

#[cfg(feature = "examples")]
impl Tokenizer for MockTokenizer {
    fn encode(
        &self,
        text: &str,
        _add_special_tokens: bool,
        _add_bos: bool,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        // Simple word-based tokenization for demo
        Ok(text.split_whitespace().enumerate().map(|(i, _)| i as u32 + 1).collect())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, Box<dyn std::error::Error>> {
        Ok(format!("decoded_{}", tokens.len()))
    }

    fn vocab_size(&self) -> usize {
        10000
    }
}

#[cfg(not(feature = "examples"))]
fn main() {
    println!("This example requires the 'examples' feature to be enabled.");
    println!(
        "Run with: cargo run --example prefill_performance_demo --features examples,cpu -- <model_path>"
    );
}
