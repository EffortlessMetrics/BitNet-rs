# GGUF Metadata Inspection Guide

BitNet-rs provides comprehensive GGUF metadata inspection capabilities with advanced categorization and JSON serialization for detailed analysis without loading tensors into memory.

## Core Features

1. **Lightweight Header Parsing**: Only reads GGUF header for fast analysis
2. **Comprehensive Metadata Extraction**: KV pairs, quantization hints, tensor summaries with categorization
3. **Smart Categorization**: Automatically categorizes metadata into model parameters, architecture, tokenizer, training, quantization, and other categories
4. **Enhanced Tensor Analysis**: Detailed tensor categorization (embeddings, weights, biases, normalization, attention, feed-forward, output heads)
5. **Memory Estimation**: Automatic memory footprint calculation based on tensor dtypes
6. **JSON Serialization**: Full JSON export capability for programmatic processing
7. **Statistical Analysis**: Parameter counts, dtype distribution, and memory usage statistics
8. **Memory Efficient**: No tensor data loading required
9. **Error Resilient**: Handles malformed GGUF files gracefully
10. **Performance Optimized**: Suitable for CI/CD pipelines and automation

## API Usage

```rust
use bitnet_inference::engine::inspect_model;

// Lightweight inspection with enhanced features
let mut model_info = inspect_model("model.gguf")?;

// Access raw metadata (backward compatible)
let kv_specs = model_info.kv_specs();           // All key-value metadata
let quant_hints = model_info.quantization_hints(); // Quantization-related metadata
let tensors = model_info.tensor_summaries();    // Enhanced tensor summaries with categorization

// Access enhanced categorized metadata
let categorized = model_info.get_categorized_metadata();
println!("Model parameters: {:?}", categorized.model_params);
println!("Architecture info: {:?}", categorized.architecture);
println!("Tokenizer config: {:?}", categorized.tokenizer);
println!("Training metadata: {:?}", categorized.training);
println!("Quantization details: {:?}", categorized.quantization);

// Access tensor statistics
let stats = model_info.get_tensor_statistics();
println!("Total parameters: {}", stats.total_parameters);
println!("Memory estimate: {} bytes", stats.estimated_memory_bytes);
println!("Parameters by category: {:?}", stats.parameters_by_category);
println!("Largest tensor: {:?}", stats.largest_tensor);

// JSON serialization
let json_pretty = model_info.to_json()?;        // Pretty-printed JSON
let json_compact = model_info.to_json_compact()?; // Compact JSON
```

## Commands

```bash
# Enhanced CLI inspection (planned integration)
cargo run -p bitnet-cli -- inspect --model model.gguf --json

# Enhanced example with categorized human-readable output
cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- model.gguf

# Enhanced example with JSON output for programmatic processing
cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- --json model.gguf

# Using environment variable
BITNET_GGUF=model.gguf cargo run --example inspect_gguf_metadata --no-default-features --features cpu

# JSON output with environment variable
BITNET_GGUF=model.gguf cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- --json

# Quick header validation (fast path)
cargo test --no-default-features --features cpu -p bitnet-inference --test engine_inspect

# Test enhanced categorization and JSON features
cargo test --no-default-features --features cpu -p bitnet-inference --test engine_inspect -- comprehensive_metadata_categorization
cargo test --no-default-features --features cpu -p bitnet-inference --test engine_inspect -- json_serialization
cargo test --no-default-features --features cpu -p bitnet-inference --test engine_inspect -- categorization_functions
```

## Use Cases

- **CI/CD**: Fast model validation in deployment pipelines with comprehensive metadata extraction
- **Model Analysis**: Understand quantization schemes, architecture, and parameter distribution without full loading
- **Model Cataloging**: Automated metadata extraction for model management systems with JSON export
- **Performance Planning**: Memory usage estimation and parameter analysis for deployment sizing
- **Debugging**: Inspect GGUF structure for compatibility issues with detailed categorization
- **Integration**: Programmatic access to model metadata through JSON API for downstream tools
- **Research**: Analyze model architecture patterns and quantization strategies across model families
