//! # GGUF Metadata Inspection Example
//!
//! This example demonstrates how to use BitNet.rs's enhanced GGUF metadata
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
//! # Inspect a GGUF model
//! cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- path/to/model.gguf
//!
//! # Example with actual model file
//! BITNET_GGUF=models/bitnet/model.gguf cargo run --example inspect_gguf_metadata --no-default-features --features cpu
//! ```

use anyhow::{Context, Result};
use bitnet_inference::engine::{ModelInfo, TensorSummary, inspect_model};
use bitnet_inference::gguf::{GgufKv, GgufValue};
use std::env;

fn main() -> Result<()> {
    // Initialize tracing for detailed logging
    tracing_subscriber::fmt::init();

    // Get model path from command line or environment
    let model_path = get_model_path()?;

    println!("🔍 Inspecting GGUF model: {}", model_path.display());
    println!("{}", "=".repeat(60));

    // Perform lightweight inspection
    let model_info = inspect_model(&model_path)
        .with_context(|| format!("Failed to inspect model: {}", model_path.display()))?;

    // Display basic header information
    display_header_info(&model_info);

    // Display key-value metadata
    display_kv_metadata(&model_info);

    // Display quantization information
    display_quantization_hints(&model_info);

    // Display tensor summaries
    display_tensor_summaries(&model_info);

    println!("\n✅ Inspection completed successfully!");
    println!(
        "💡 Tip: Use `cargo run -p bitnet-cli -- inspect --model {}` for JSON output",
        model_path.display()
    );

    Ok(())
}

fn get_model_path() -> Result<std::path::PathBuf> {
    // Try command line argument first
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        return Ok(args[1].clone().into());
    }

    // Try environment variable
    if let Ok(path) = env::var("BITNET_GGUF") {
        return Ok(path.into());
    }

    if let Ok(path) = env::var("CROSSVAL_GGUF") {
        return Ok(path.into());
    }

    anyhow::bail!(
        "Please provide a model path as argument or set BITNET_GGUF environment variable\n\
         Example: cargo run --example inspect_gguf_metadata -- path/to/model.gguf"
    );
}

fn display_header_info(model_info: &ModelInfo) {
    println!("📋 **Basic Header Information**");
    println!("   GGUF Version: {}", model_info.version());
    println!("   Total Tensors: {}", model_info.n_tensors());
    println!("   KV Metadata Entries: {}", model_info.n_kv());
    println!();
}

fn display_kv_metadata(model_info: &ModelInfo) {
    println!("🔑 **Key-Value Metadata** ({} entries)", model_info.kv_specs().len());

    if model_info.kv_specs().is_empty() {
        println!("   (No KV metadata found)");
        println!();
        return;
    }

    // Group metadata by category for better readability
    let mut model_params = Vec::new();
    let mut architecture_info = Vec::new();
    let mut tokenizer_info = Vec::new();
    let mut other_metadata = Vec::new();

    for kv in model_info.kv_specs() {
        let category = categorize_kv_key(&kv.key);
        match category {
            "model" => model_params.push(kv),
            "architecture" => architecture_info.push(kv),
            "tokenizer" => tokenizer_info.push(kv),
            _ => other_metadata.push(kv),
        }
    }

    display_kv_category("Model Parameters", &model_params);
    display_kv_category("Architecture", &architecture_info);
    display_kv_category("Tokenizer", &tokenizer_info);
    display_kv_category("Other Metadata", &other_metadata);
    println!();
}

fn categorize_kv_key(key: &str) -> &'static str {
    let key_lower = key.to_lowercase();

    if key_lower.contains("vocab_size")
        || key_lower.contains("context_length")
        || key_lower.contains("embedding_length")
        || key_lower.contains("block_count")
        || key_lower.contains("feed_forward_length")
        || key_lower.contains("attention.head_count")
    {
        "model"
    } else if key_lower.contains("architecture")
        || key_lower.contains("attention")
        || key_lower.contains("rope")
        || key_lower.contains("layer")
    {
        "architecture"
    } else if key_lower.contains("tokenizer")
        || key_lower.contains("bos_token")
        || key_lower.contains("eos_token")
        || key_lower.contains("pad_token")
    {
        "tokenizer"
    } else {
        "other"
    }
}

fn display_kv_category(title: &str, kvs: &[&GgufKv]) {
    if kvs.is_empty() {
        return;
    }

    println!("   📁 {title}:");
    for kv in kvs {
        println!("      {}: {}", kv.key, format_value(&kv.value));
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

fn display_tensor_summaries(model_info: &ModelInfo) {
    println!("📊 **Tensor Summary** ({} tensors)", model_info.tensor_summaries().len());

    if model_info.tensor_summaries().is_empty() {
        println!("   (No tensor information found)");
        println!();
        return;
    }

    // Group tensors by type for better organization
    let mut embeddings = Vec::new();
    let mut weights = Vec::new();
    let mut biases = Vec::new();
    let mut other_tensors = Vec::new();

    for tensor in model_info.tensor_summaries() {
        let category = categorize_tensor_name(&tensor.name);
        match category {
            "embedding" => embeddings.push(tensor),
            "weight" => weights.push(tensor),
            "bias" => biases.push(tensor),
            _ => other_tensors.push(tensor),
        }
    }

    display_tensor_category("Embeddings", &embeddings);
    display_tensor_category("Weights", &weights);
    display_tensor_category("Biases", &biases);
    display_tensor_category("Other Tensors", &other_tensors);

    // Display summary statistics
    display_tensor_statistics(model_info.tensor_summaries());
    println!();
}

fn categorize_tensor_name(name: &str) -> &'static str {
    let name_lower = name.to_lowercase();

    if name_lower.contains("embed") || name_lower.contains("token") {
        "embedding"
    } else if name_lower.contains("weight") || name_lower.contains(".w") {
        "weight"
    } else if name_lower.contains("bias") || name_lower.contains(".b") {
        "bias"
    } else {
        "other"
    }
}

fn display_tensor_category(title: &str, tensors: &[&TensorSummary]) {
    if tensors.is_empty() {
        return;
    }

    println!("   📁 {title} ({} tensors):", tensors.len());
    for tensor in tensors.iter().take(10) {
        // Limit display for readability
        let shape_str = tensor.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" × ");
        println!(
            "      📐 {} [{}] (dtype: {})",
            tensor.name,
            shape_str,
            format_dtype(tensor.dtype)
        );
    }

    if tensors.len() > 10 {
        println!("      ... and {} more tensors", tensors.len() - 10);
    }
}

fn display_tensor_statistics(tensors: &[TensorSummary]) {
    if tensors.is_empty() {
        return;
    }

    let total_params: u64 = tensors.iter().map(|t| t.shape.iter().product::<u64>()).sum();

    let unique_dtypes: std::collections::HashSet<u32> = tensors.iter().map(|t| t.dtype).collect();

    println!("   📊 Statistics:");
    println!("      Total Parameters: {:.2}M", total_params as f64 / 1_000_000.0);
    println!(
        "      Unique Data Types: {} ({})",
        unique_dtypes.len(),
        unique_dtypes.iter().map(|dt| format_dtype(*dt)).collect::<Vec<_>>().join(", ")
    );
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
                format!("[{}]", arr.iter().map(|v| format_value(v)).collect::<Vec<_>>().join(", "))
            } else {
                format!(
                    "[{}, ... {} more items]",
                    arr.iter().take(3).map(|v| format_value(v)).collect::<Vec<_>>().join(", "),
                    arr.len() - 3
                )
            }
        }
        GgufValue::U64(v) => v.to_string(),
        GgufValue::I64(v) => v.to_string(),
        GgufValue::F64(v) => format!("{:.6}", v),
    }
}

fn format_dtype(dtype: u32) -> String {
    match dtype {
        0 => "F32".to_string(),
        1 => "F16".to_string(),
        2 => "Q4_0".to_string(),
        3 => "Q4_1".to_string(),
        6 => "Q5_0".to_string(),
        7 => "Q5_1".to_string(),
        8 => "Q8_0".to_string(),
        9 => "Q8_1".to_string(),
        10 => "Q2_K".to_string(),
        11 => "Q3_K".to_string(),
        12 => "Q4_K".to_string(),
        13 => "Q5_K".to_string(),
        14 => "Q6_K".to_string(),
        15 => "Q8_K".to_string(),
        17 => "I2_S".to_string(),  // BitNet native format
        18 => "IQ2_S".to_string(), // BitNet extended format
        19 => "TL1".to_string(),   // Table lookup 1
        20 => "TL2".to_string(),   // Table lookup 2
        _ => format!("Unknown({})", dtype),
    }
}
