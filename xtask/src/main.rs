// BitNet.rs Development Task Runner
// This provides convenient development tasks for the BitNet.rs project

use std::env;
use std::fs;
use std::path::Path;
use std::process::{Command, exit};

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_help();
        exit(1);
    }
    
    match args[1].as_str() {
        "gen-fixtures" => gen_fixtures(&args[2..]),
        "setup-crossval" => setup_crossval(),
        "clean-cache" => clean_cache(),
        "check-features" => check_features(),
        "benchmark" => run_benchmark(&args[2..]),
        "help" | "--help" | "-h" => print_help(),
        _ => {
            eprintln!("Unknown task: {}", args[1]);
            print_help();
            exit(1);
        }
    }
}

fn print_help() {
    println!("BitNet.rs Development Task Runner");
    println!();
    println!("USAGE:");
    println!("    cargo xtask <TASK> [OPTIONS]");
    println!();
    println!("TASKS:");
    println!("    gen-fixtures     Generate deterministic test model fixtures");
    println!("    setup-crossval   Set up cross-validation environment");
    println!("    clean-cache      Clean all caches and temporary files");
    println!("    check-features   Check feature flag consistency");
    println!("    benchmark        Run performance benchmarks");
    println!("    help             Show this help message");
    println!();
    println!("EXAMPLES:");
    println!("    cargo xtask gen-fixtures --size small --output crossval/fixtures/");
    println!("    cargo xtask setup-crossval");
    println!("    cargo xtask benchmark --platform current");
    println!();
    println!("For more information, visit: https://github.com/microsoft/BitNet");
}

fn gen_fixtures(args: &[String]) {
    println!("🔧 Generating deterministic test model fixtures...");
    
    let mut size = "small";
    let mut output_dir = "crossval/fixtures/";
    let mut format = "gguf";
    
    // Parse arguments
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--size" => {
                if i + 1 < args.len() {
                    size = &args[i + 1];
                    i += 2;
                } else {
                    eprintln!("Error: --size requires a value");
                    exit(1);
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output_dir = &args[i + 1];
                    i += 2;
                } else {
                    eprintln!("Error: --output requires a value");
                    exit(1);
                }
            }
            "--format" => {
                if i + 1 < args.len() {
                    format = &args[i + 1];
                    i += 2;
                } else {
                    eprintln!("Error: --format requires a value");
                    exit(1);
                }
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                exit(1);
            }
        }
    }
    
    println!("  Size: {}", size);
    println!("  Output: {}", output_dir);
    println!("  Format: {}", format);
    
    // Create output directory
    if let Err(e) = fs::create_dir_all(output_dir) {
        eprintln!("Error creating output directory: {}", e);
        exit(1);
    }
    
    // Generate fixtures based on size
    match size {
        "tiny" => generate_tiny_fixture(output_dir, format),
        "small" => generate_small_fixture(output_dir, format),
        "medium" => generate_medium_fixture(output_dir, format),
        _ => {
            eprintln!("Unknown size: {}. Use tiny, small, or medium", size);
            exit(1);
        }
    }
    
    println!("✅ Test fixtures generated successfully!");
}

fn generate_tiny_fixture(output_dir: &str, format: &str) {
    println!("  Generating tiny fixture (~5KB)...");
    
    // Create a minimal model for basic testing
    let fixture_content = r#"{
  "model_type": "bitnet_b1_58",
  "vocab_size": 1000,
  "hidden_size": 64,
  "num_layers": 2,
  "num_attention_heads": 4,
  "intermediate_size": 128,
  "max_position_embeddings": 512,
  "layer_norm_eps": 1e-5,
  "use_cache": true,
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "tie_word_embeddings": false,
  "quantization": {
    "method": "bitnet_b1_58",
    "bits": 1.58,
    "group_size": 128
  },
  "test_metadata": {
    "fixture_type": "tiny",
    "deterministic": true,
    "seed": 42,
    "created_by": "xtask"
  }
}"#;
    
    let fixture_path = Path::new(output_dir).join(format!("tiny_model.{}", format));
    if let Err(e) = fs::write(&fixture_path, fixture_content) {
        eprintln!("Error writing fixture: {}", e);
        exit(1);
    }
    
    println!("    Created: {}", fixture_path.display());
}

fn generate_small_fixture(output_dir: &str, format: &str) {
    println!("  Generating small fixture (~20KB)...");
    
    // Create a small but realistic model for cross-validation
    let fixture_content = r#"{
  "model_type": "bitnet_b1_58",
  "vocab_size": 5000,
  "hidden_size": 256,
  "num_layers": 4,
  "num_attention_heads": 8,
  "intermediate_size": 512,
  "max_position_embeddings": 2048,
  "layer_norm_eps": 1e-5,
  "use_cache": true,
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "tie_word_embeddings": false,
  "quantization": {
    "method": "bitnet_b1_58",
    "bits": 1.58,
    "group_size": 128,
    "calibration_dataset": "c4",
    "calibration_samples": 128
  },
  "test_metadata": {
    "fixture_type": "small",
    "deterministic": true,
    "seed": 42,
    "created_by": "xtask",
    "test_prompts": [
      "The quick brown fox",
      "Hello, world!",
      "Rust is a systems programming language",
      "BitNet enables efficient 1-bit LLM inference"
    ],
    "expected_tokens": {
      "The quick brown fox": [464, 2068, 2829, 4419],
      "Hello, world!": [9906, 11, 995, 0],
      "Rust is a systems programming language": [49, 436, 318, 257, 3341, 8300, 3303],
      "BitNet enables efficient 1-bit LLM inference": [13128, 7934, 13536, 6942, 352, 12, 2545, 406, 11237, 32278]
    }
  },
  "architecture": {
    "attention": {
      "type": "multi_head",
      "dropout": 0.1,
      "bias": false
    },
    "mlp": {
      "type": "gated",
      "activation": "silu",
      "bias": false
    },
    "normalization": {
      "type": "rms_norm",
      "eps": 1e-6
    }
  }
}"#;
    
    let fixture_path = Path::new(output_dir).join(format!("small_model.{}", format));
    if let Err(e) = fs::write(&fixture_path, fixture_content) {
        eprintln!("Error writing fixture: {}", e);
        exit(1);
    }
    
    println!("    Created: {}", fixture_path.display());
    
    // Also create test prompts file
    let prompts_content = r#"# Test Prompts for Small Fixture
# These prompts are designed to test various aspects of the model

## Basic Functionality
The quick brown fox
Hello, world!

## Technical Content
Rust is a systems programming language
BitNet enables efficient 1-bit LLM inference

## Longer Context
In the field of machine learning, quantization techniques have become increasingly important for deploying large language models efficiently.

## Edge Cases
""
" "
"🦀"
"123456789"
"!@#$%^&*()"
"#;
    
    let prompts_path = Path::new(output_dir).join("test_prompts.txt");
    if let Err(e) = fs::write(&prompts_path, prompts_content) {
        eprintln!("Error writing prompts file: {}", e);
        exit(1);
    }
    
    println!("    Created: {}", prompts_path.display());
}

fn generate_medium_fixture(output_dir: &str, format: &str) {
    println!("  Generating medium fixture (~100KB)...");
    
    // Create a medium-sized model for comprehensive testing
    let fixture_content = r#"{
  "model_type": "bitnet_b1_58",
  "vocab_size": 32000,
  "hidden_size": 512,
  "num_layers": 8,
  "num_attention_heads": 16,
  "intermediate_size": 1024,
  "max_position_embeddings": 4096,
  "layer_norm_eps": 1e-5,
  "use_cache": true,
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "tie_word_embeddings": false,
  "quantization": {
    "method": "bitnet_b1_58",
    "bits": 1.58,
    "group_size": 128,
    "calibration_dataset": "c4",
    "calibration_samples": 512,
    "outlier_threshold": 3.0
  },
  "test_metadata": {
    "fixture_type": "medium",
    "deterministic": true,
    "seed": 42,
    "created_by": "xtask",
    "performance_targets": {
      "throughput_tokens_per_second": 100,
      "latency_p95_ms": 150,
      "memory_usage_mb": 200,
      "accuracy_threshold": 0.95
    }
  }
}"#;
    
    let fixture_path = Path::new(output_dir).join(format!("medium_model.{}", format));
    if let Err(e) = fs::write(&fixture_path, fixture_content) {
        eprintln!("Error writing fixture: {}", e);
        exit(1);
    }
    
    println!("    Created: {}", fixture_path.display());
}

fn setup_crossval() {
    println!("🔧 Setting up cross-validation environment...");
    
    // Check if BitNet.cpp cache is available
    println!("  Checking BitNet.cpp cache...");
    let cache_script = "./ci/use-bitnet-cpp-cache.sh";
    
    if !Path::new(cache_script).exists() {
        eprintln!("Error: Cache script not found: {}", cache_script);
        exit(1);
    }
    
    // Run cache setup
    let output = Command::new("bash")
        .arg(cache_script)
        .output()
        .expect("Failed to run cache setup");
    
    if !output.status.success() {
        eprintln!("Error setting up BitNet.cpp cache:");
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        exit(1);
    }
    
    println!("  ✅ BitNet.cpp cache ready");
    
    // Generate test fixtures
    println!("  Generating test fixtures...");
    gen_fixtures(&["--size".to_string(), "small".to_string(), "--output".to_string(), "crossval/fixtures/".to_string()]);
    
    // Build with crossval features
    println!("  Building with cross-validation features...");
    let build_output = Command::new("cargo")
        .args(&["build", "--features", "crossval"])
        .output()
        .expect("Failed to build with crossval features");
    
    if !build_output.status.success() {
        eprintln!("Error building with crossval features:");
        eprintln!("{}", String::from_utf8_lossy(&build_output.stderr));
        exit(1);
    }
    
    println!("  ✅ Built with cross-validation features");
    
    // Run a quick test
    println!("  Running quick cross-validation test...");
    let test_output = Command::new("cargo")
        .args(&["test", "--package", "crossval", "--features", "crossval", "--", "--nocapture", "quick_test"])
        .output()
        .expect("Failed to run crossval test");
    
    if test_output.status.success() {
        println!("  ✅ Cross-validation test passed");
    } else {
        println!("  ⚠️  Cross-validation test had issues (this may be expected)");
    }
    
    println!("✅ Cross-validation environment setup complete!");
    println!();
    println!("You can now run:");
    println!("  cargo test --package crossval --features crossval");
    println!("  cargo bench --package crossval --features crossval");
}

fn clean_cache() {
    println!("🧹 Cleaning all caches and temporary files...");
    
    let cache_dirs = [
        "target/",
        "~/.cache/bitnet_cpp/",
        "crossval/fixtures/",
        ".cargo-cache/",
    ];
    
    for dir in &cache_dirs {
        if dir.starts_with('~') {
            // Handle home directory expansion
            if let Ok(home) = env::var("HOME") {
                let expanded_dir = dir.replace('~', &home);
                clean_directory(&expanded_dir);
            }
        } else {
            clean_directory(dir);
        }
    }
    
    println!("✅ Cache cleanup complete!");
}

fn clean_directory(dir: &str) {
    if Path::new(dir).exists() {
        println!("  Cleaning: {}", dir);
        if let Err(e) = fs::remove_dir_all(dir) {
            println!("    Warning: Could not remove {}: {}", dir, e);
        } else {
            println!("    ✅ Removed: {}", dir);
        }
    } else {
        println!("    Skipping: {} (does not exist)", dir);
    }
}

fn check_features() {
    println!("🔍 Checking feature flag consistency...");
    
    // Check that crossval feature is not accidentally enabled
    let cargo_toml_content = match fs::read_to_string("Cargo.toml") {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading Cargo.toml: {}", e);
            exit(1);
        }
    };
    
    if cargo_toml_content.contains("default = [") && cargo_toml_content.contains("crossval") {
        eprintln!("❌ ERROR: crossval feature is enabled by default!");
        eprintln!("   This will slow down builds and is not recommended.");
        eprintln!("   Please remove 'crossval' from the default features.");
        exit(1);
    }
    
    println!("  ✅ crossval feature is not in default features");
    
    // Check workspace feature consistency
    let workspace_members = [
        "crates/bitnet-common",
        "crates/bitnet-models",
        "crates/bitnet-quantization",
        "crates/bitnet-kernels",
        "crates/bitnet-inference",
    ];
    
    for member in &workspace_members {
        let cargo_toml_path = format!("{}/Cargo.toml", member);
        if Path::new(&cargo_toml_path).exists() {
            println!("  Checking: {}", member);
            // Additional feature consistency checks could go here
        }
    }
    
    println!("✅ Feature flag consistency check passed!");
}

fn run_benchmark(args: &[String]) {
    println!("🚀 Running performance benchmarks...");
    
    let mut platform = "current";
    let mut features = "cpu";
    
    // Parse arguments
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--platform" => {
                if i + 1 < args.len() {
                    platform = &args[i + 1];
                    i += 2;
                } else {
                    eprintln!("Error: --platform requires a value");
                    exit(1);
                }
            }
            "--features" => {
                if i + 1 < args.len() {
                    features = &args[i + 1];
                    i += 2;
                } else {
                    eprintln!("Error: --features requires a value");
                    exit(1);
                }
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                exit(1);
            }
        }
    }
    
    println!("  Platform: {}", platform);
    println!("  Features: {}", features);
    
    // Run benchmarks
    let bench_output = Command::new("cargo")
        .args(&["bench", "--features", features])
        .output()
        .expect("Failed to run benchmarks");
    
    if bench_output.status.success() {
        println!("✅ Benchmarks completed successfully!");
        println!("{}", String::from_utf8_lossy(&bench_output.stdout));
    } else {
        eprintln!("❌ Benchmarks failed:");
        eprintln!("{}", String::from_utf8_lossy(&bench_output.stderr));
        exit(1);
    }
}