//! Prefill Performance Demonstration
//!
//! This example demonstrates the enhanced prefill functionality added in PR #187,
//! showing how to use explicit prefill for cache warming and performance measurement
//! in batch inference operations.

#[cfg(feature = "examples")]
use bitnet::prelude::*;
#[cfg(feature = "examples")]
use serde_json;
#[cfg(feature = "examples")]
use std::env;
#[cfg(feature = "examples")]
use std::time::Instant;

#[cfg(feature = "examples")]
pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Get model path from environment or command line
    let model_path = env::var("BITNET_GGUF")
        .or_else(|_| env::args().nth(1).ok_or("No model path provided"))
        .unwrap_or_else(|_| {
            eprintln!("Usage: {} <model_path>", env::args().next().unwrap());
            eprintln!("   or: BITNET_GGUF=model.gguf {}", env::args().next().unwrap());
            std::process::exit(1);
        });

    println!("🚀 BitNet-rs Prefill Performance Demo");
    println!("📁 Model: {}", model_path);
    println!();

    // Load model with device auto-selection
    let device = Device::Auto;
    let loader = ModelLoader::new(device.clone());
    let model = loader.load(&model_path)?;

    // Create tokenizer (uses universal tokenizer with GGUF integration)
    let tokenizer = TokenizerBuilder::from_file(&model_path)?;

    // Create inference engine with prefill support
    let mut engine = InferenceEngine::new(model, tokenizer)?;

    println!("✅ Engine initialized with device: {:?}", device);
    println!();

    // Demo prompts of different lengths to show prefill behavior
    let demo_prompts = vec![
        "Hello world",
        "The future of artificial intelligence involves",
        "In a comprehensive analysis of modern machine learning techniques, we can observe that the field has evolved significantly over the past decade, with breakthrough innovations in",
    ];

    println!("🔬 Demonstrating Explicit Prefill Functionality");
    println!("================================================");

    for (i, prompt) in demo_prompts.iter().enumerate() {
        println!(
            "\n📝 Demo {}: \"{}...\"",
            i + 1,
            if prompt.len() > 40 { &prompt[..40] } else { prompt }
        );

        // Configure generation with metrics enabled
        let config = GenerationConfig {
            max_new_tokens: 20,
            temperature: 0.7,
            top_k: Some(40),
            top_p: 0.9,
            enable_metrics: true, // Enable detailed performance metrics
            ..Default::default()
        };

        // Measure total time
        let start_total = Instant::now();

        // Generate with prefill integration
        let response = engine.generate_with_config(prompt, &config)?;
        let total_time = start_total.elapsed();

        // Display results
        println!("💬 Generated: {}", response.text.trim());

        // Show detailed metrics if available
        if let Some(metrics) = response.metrics {
            println!("⏱️  Performance Metrics:");
            println!("   • Tokenization: {:.2}ms", metrics.timing.tokenize);
            println!("   • Prefill:      {:.2}ms", metrics.timing.prefill);
            println!("   • Decode:       {:.2}ms", metrics.timing.decode);
            println!("   • Total:        {:.2}ms", metrics.timing.total);
            println!();
            println!("🚀 Throughput:");
            println!("   • Prefill:  {:.1} tok/s", metrics.throughput.prefill);
            println!("   • Decode:   {:.1} tok/s", metrics.throughput.decode);
            println!("   • E2E:      {:.1} tok/s", metrics.throughput.e2e);

            // Export metrics as JSON for analysis
            let metrics_json = serde_json::to_string_pretty(&metrics)?;
            println!("\n📊 JSON Metrics:");
            println!("{}", metrics_json);
        }

        println!("⏰ Wall-clock time: {:.2}ms", total_time.as_secs_f64() * 1000.0);
        println!("{}", "-".repeat(50));
    }

    // Demonstrate batch processing with prefill
    println!("\n🚀 Batch Processing with Prefill");
    println!("=================================");

    let batch_prompts =
        vec!["Explain quantum computing", "What is machine learning?", "Describe neural networks"];

    let batch_start = Instant::now();
    let mut batch_results = Vec::new();

    for (i, prompt) in batch_prompts.iter().enumerate() {
        println!("\n🔄 Processing batch item {}/{}", i + 1, batch_prompts.len());

        let config = GenerationConfig {
            max_new_tokens: 15,
            temperature: 0.5,
            enable_metrics: true,
            ..Default::default()
        };

        let response = engine.generate_with_config(prompt, &config)?;
        batch_results.push((prompt, response));

        // Show progress
        if let Some(ref metrics) = batch_results.last().unwrap().1.metrics {
            println!(
                "   Prefill: {:.1}ms, Decode: {:.1}ms",
                metrics.timing.prefill, metrics.timing.decode
            );
        }
    }

    let batch_total_time = batch_start.elapsed();

    println!("\n📈 Batch Processing Summary");
    println!("==========================");

    let mut total_prefill_time = 0.0;
    let mut total_decode_time = 0.0;
    let mut total_tokens = 0;

    for (i, (prompt, response)) in batch_results.iter().enumerate() {
        println!(
            "• Item {}: \"{}\" → \"{}\"",
            i + 1,
            if prompt.len() > 20 { &prompt[..20] } else { prompt },
            if response.text.len() > 30 { &response.text[..30] } else { &response.text }
        );

        if let Some(ref metrics) = response.metrics {
            total_prefill_time += metrics.timing.prefill;
            total_decode_time += metrics.timing.decode;
            total_tokens += response.text.split_whitespace().count(); // Rough token estimate
        }
    }

    println!("\n🏁 Final Statistics:");
    println!("   • Batch size:     {}", batch_results.len());
    println!("   • Total time:     {:.2}ms", batch_total_time.as_secs_f64() * 1000.0);
    println!(
        "   • Avg per item:   {:.2}ms",
        batch_total_time.as_secs_f64() * 1000.0 / batch_results.len() as f64
    );
    println!("   • Total prefill:  {:.2}ms", total_prefill_time);
    println!("   • Total decode:   {:.2}ms", total_decode_time);
    println!("   • Est. tokens:    ~{}", total_tokens);

    if total_tokens > 0 {
        let throughput = total_tokens as f64 / batch_total_time.as_secs_f64();
        println!("   • Batch throughput: {:.1} tok/s", throughput);
    }

    println!("\n✨ Prefill Demo Complete!");
    println!("\n💡 Key Benefits of Explicit Prefill:");
    println!("   • Separate timing measurement for prefill vs generation");
    println!("   • KV cache warming improves subsequent generation quality");
    println!("   • Better performance analysis and debugging capabilities");
    println!("   • Enhanced batch processing with consistent pipeline");

    Ok(())
}

#[cfg(not(feature = "examples"))]
pub fn main() {
    println!("This example requires the 'examples' feature to be enabled.");
    println!("Run with: cargo run --example prefill_performance_demo --features examples");
}
