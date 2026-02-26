//! # GGUF Metadata Inspection Example
//!
//! This example demonstrates how to use BitNet-rs's enhanced GGUF metadata
//! extraction capabilities to inspect model files without loading the full model.
//!
//! ## Key Features Demonstrated
//!
//! - **Lightweight inspection**: Get metadata without loading tensors into memory
//! - **Comprehensive metadata**: Extract KV pairs, quantization hints, and tensor summaries
//! - **Error handling**: Robust parsing with detailed error messages
//! - **Performance**: Fast header-only parsing for quick model analysis
//!
//! ## Usage
//!
//! ```bash
//! # Inspect a GGUF model (human-readable output)
//! cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- path/to/model.gguf
//!
//! # Get JSON output for programmatic processing
//! cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- --json path/to/model.gguf
//!
//! # Example with environment variable
//! BITNET_GGUF=models/bitnet/model.gguf cargo run --example inspect_gguf_metadata --no-default-features --features cpu
//!
//! # JSON output with environment variable
//! BITNET_GGUF=models/bitnet/model.gguf cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- --json
//! ```

use anyhow::{Context, Result};
use bitnet_inference::engine::{ModelInfo, TensorSummary, format_dtype, inspect_model};
use bitnet_inference::gguf::GgufValue;
use std::env;

fn main() -> Result<()> {
    // Initialize tracing for detailed logging
    tracing_subscriber::fmt::init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let json_output = args.contains(&"--json".to_string());

    // Get model path from command line or environment
    let model_path = get_model_path()?;

    // Perform lightweight inspection
    let mut model_info = inspect_model(&model_path)
        .with_context(|| format!("Failed to inspect model: {}", model_path.display()))?;

    if json_output {
        // Force computation of enhanced metadata for JSON output
        let _ = model_info.get_categorized_metadata();
        let _ = model_info.get_tensor_statistics();

        // Output JSON
        let json = model_info.to_json()?;
        println!("{}", json);
    } else {
        // Human-readable output
        println!("🔍 Inspecting GGUF model: {}", model_path.display());
        println!("{}", "=".repeat(60));

        // Display basic header information
        display_header_info(&model_info);

        // Pre-compute enhanced metadata to avoid borrowing conflicts
        let categorized = model_info.get_categorized_metadata().clone();
        let stats = model_info.get_tensor_statistics().clone();

        // Display enhanced key-value metadata with categorization
        display_enhanced_kv_metadata(&model_info, &categorized);

        // Display quantization information
        display_quantization_hints(&model_info);

        // Display tensor summaries with enhanced categorization
        display_enhanced_tensor_summaries(&model_info, &stats);

        println!("\n✅ Inspection completed successfully!");
        println!(
            "💡 Tip: Use --json flag for JSON output or `cargo run -p bitnet-cli -- inspect --model {}` for CLI integration",
            model_path.display()
        );
    }

    Ok(())
}

fn get_model_path() -> Result<std::path::PathBuf> {
    // Try command line argument first (skip --json flag if present)
    let args: Vec<String> = env::args().collect();
    let model_arg = args.iter().skip(1).find(|arg| !arg.starts_with("--"));
    if let Some(path) = model_arg {
        return Ok(path.clone().into());
    }

    // Try environment variable
    if let Ok(path) = env::var("BITNET_GGUF") {
        return Ok(path.into());
    }

    if let Ok(path) = env::var("CROSSVAL_GGUF") {
        return Ok(path.into());
    }

    anyhow::bail!(
        "Please provide a model path as argument or set BITNET_GGUF environment variable\n\n\
         Examples:\n\
         cargo run --example inspect_gguf_metadata -- path/to/model.gguf\n\
         cargo run --example inspect_gguf_metadata -- --json path/to/model.gguf\n\
         BITNET_GGUF=path/to/model.gguf cargo run --example inspect_gguf_metadata\n\n\
         Flags:\n\
         --json    Output results in JSON format for programmatic processing"
    );
}

fn display_header_info(model_info: &ModelInfo) {
    println!("📋 **Basic Header Information**");
    println!("   GGUF Version: {}", model_info.version());
    println!("   Total Tensors: {}", model_info.n_tensors());
    println!("   KV Metadata Entries: {}", model_info.n_kv());
    println!();
}

fn display_enhanced_kv_metadata(
    model_info: &ModelInfo,
    categorized: &bitnet_inference::engine::CategorizedMetadata,
) {
    println!("🔑 **Enhanced Key-Value Metadata** ({} entries)", model_info.kv_specs().len());

    if model_info.kv_specs().is_empty() {
        println!("   (No KV metadata found)");
        println!();
        return;
    }

    // Display each category with enhanced formatting
    display_enhanced_kv_category("📊 Model Parameters", &categorized.model_params);
    display_enhanced_kv_category("🏗️  Architecture", &categorized.architecture);
    display_enhanced_kv_category("🔤 Tokenizer", &categorized.tokenizer);
    display_enhanced_kv_category("🎯 Training", &categorized.training);
    display_enhanced_kv_category("⚖️  Quantization", &categorized.quantization);
    display_enhanced_kv_category("📝 Other", &categorized.other);
    println!();
}

fn display_enhanced_kv_category(title: &str, kvs: &std::collections::HashMap<String, String>) {
    if kvs.is_empty() {
        return;
    }

    println!("   {}:", title);
    for (key, value) in kvs {
        println!("      {}: {}", key, value);
    }
}

fn display_quantization_hints(model_info: &ModelInfo) {
    println!("⚖️  **Quantization Information** ({} hints)", model_info.quantization_hints().len());

    if model_info.quantization_hints().is_empty() {
        println!("   (No quantization metadata found)");
        println!();
        return;
    }

    for hint in model_info.quantization_hints() {
        println!("   🔧 {}: {}", hint.key, format_value(&hint.value));
    }
    println!();
}

fn display_enhanced_tensor_summaries(
    model_info: &ModelInfo,
    stats: &bitnet_inference::engine::TensorStatistics,
) {
    println!("📊 **Enhanced Tensor Summary** ({} tensors)", model_info.tensor_summaries().len());

    if model_info.tensor_summaries().is_empty() {
        println!("   (No tensor information found)");
        println!();
        return;
    }

    // Display statistics overview
    println!("   📈 **Statistics Overview**:");
    println!("      Total Parameters: {:.2}M", stats.total_parameters as f64 / 1_000_000.0);
    println!("      Estimated Memory: {:.2} MB", stats.estimated_memory_bytes as f64 / 1_000_000.0);
    println!(
        "      Unique Data Types: {} ({})",
        stats.unique_dtypes.len(),
        stats.unique_dtypes.iter().map(|dt| format_dtype(*dt)).collect::<Vec<_>>().join(", ")
    );
    if let Some(ref largest) = stats.largest_tensor {
        println!("      Largest Tensor: {}", largest);
    }
    println!();

    // Group tensors by category with enhanced display
    let mut categories: std::collections::HashMap<String, Vec<&TensorSummary>> =
        std::collections::HashMap::new();
    for tensor in model_info.tensor_summaries() {
        let category = tensor.category.as_deref().unwrap_or("other");
        categories.entry(category.to_string()).or_default().push(tensor);
    }

    // Display each category
    for (category, tensors) in categories {
        if !tensors.is_empty() {
            let params_in_category = stats.parameters_by_category.get(&category).unwrap_or(&0);
            println!(
                "   📁 **{}** ({} tensors, {:.2}M params):",
                capitalize(&category),
                tensors.len(),
                *params_in_category as f64 / 1_000_000.0
            );

            for tensor in tensors.iter().take(10) {
                // Limit display for readability
                let shape_str =
                    tensor.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" × ");
                let dtype_name = tensor.dtype_name.as_deref().unwrap_or("Unknown");
                println!(
                    "      📐 {} [{}] ({}, {:.1}K params)",
                    tensor.name,
                    shape_str,
                    dtype_name,
                    tensor.parameter_count as f64 / 1000.0
                );
            }

            if tensors.len() > 10 {
                println!("      ... and {} more tensors", tensors.len() - 10);
            }
            println!();
        }
    }
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

fn format_value(value: &GgufValue) -> String {
    match value {
        GgufValue::U8(v) => v.to_string(),
        GgufValue::I8(v) => v.to_string(),
        GgufValue::U16(v) => v.to_string(),
        GgufValue::I16(v) => v.to_string(),
        GgufValue::U32(v) => v.to_string(),
        GgufValue::I32(v) => v.to_string(),
        GgufValue::F32(v) => format!("{:.6}", v),
        GgufValue::Bool(v) => v.to_string(),
        GgufValue::String(v) => format!("\"{}\"", v),
        GgufValue::Array(arr) => {
            if arr.len() <= 5 {
                format!("[{}]", arr.iter().map(format_value).collect::<Vec<_>>().join(", "))
            } else {
                format!(
                    "[{}, ... {} more items]",
                    arr.iter().take(3).map(format_value).collect::<Vec<_>>().join(", "),
                    arr.len() - 3
                )
            }
        }
        GgufValue::U64(v) => v.to_string(),
        GgufValue::I64(v) => v.to_string(),
        GgufValue::F64(v) => format!("{:.6}", v),
    }
}
