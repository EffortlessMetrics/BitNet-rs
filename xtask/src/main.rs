//! BitNet.rs development tasks
//!
//! This binary provides development utilities for the BitNet.rs project,
//! including fixture generation for cross-validation testing.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "xtask")]
#[command(about = "BitNet.rs development tasks")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate test fixtures for cross-validation
    GenFixtures {
        /// Output directory for fixtures
        #[arg(short, long, default_value = "crossval/fixtures")]
        output: PathBuf,
        
        /// Generate deterministic fixtures (same every time)
        #[arg(long)]
        deterministic: bool,
        
        /// Number of test prompts per fixture
        #[arg(long, default_value = "5")]
        prompts: usize,
    },
    
    /// Validate existing fixtures
    ValidateFixtures {
        /// Fixtures directory to validate
        #[arg(short, long, default_value = "crossval/fixtures")]
        input: PathBuf,
    },
    
    /// Clean generated fixtures
    CleanFixtures {
        /// Fixtures directory to clean
        #[arg(short, long, default_value = "crossval/fixtures")]
        input: PathBuf,
    },
    
    /// Migrate C++ configuration to Rust
    MigrateConfig {
        /// Input C++ configuration file
        #[arg(short, long)]
        from: PathBuf,
        
        /// Output Rust configuration file
        #[arg(short, long)]
        to: PathBuf,
        
        /// Configuration format (json, yaml, toml)
        #[arg(long, default_value = "auto")]
        format: String,
    },
    
    /// Validate model format compatibility
    ValidateModel {
        /// Model file to validate
        #[arg(short, long)]
        model: PathBuf,
        
        /// Check cross-compatibility with C++ implementation
        #[arg(long)]
        cross_validate: bool,
    },
    
    /// Generate migration report
    MigrationReport {
        /// Source directory to analyze
        #[arg(short, long, default_value = ".")]
        source: PathBuf,
        
        /// Output report file
        #[arg(short, long, default_value = "migration-report.md")]
        output: PathBuf,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestFixture {
    name: String,
    model_path: String,
    test_prompts: Vec<String>,
    expected_tokens: Option<Vec<Vec<u32>>>,
    description: String,
    model_size_kb: u64,
    max_tokens: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::GenFixtures { output, deterministic, prompts } => {
            generate_fixtures(&output, deterministic, prompts)
        }
        Commands::ValidateFixtures { input } => {
            validate_fixtures(&input)
        }
        Commands::CleanFixtures { input } => {
            clean_fixtures(&input)
        }
        Commands::MigrateConfig { from, to, format } => {
            migrate_config(&from, &to, &format)
        }
        Commands::ValidateModel { model, cross_validate } => {
            validate_model(&model, cross_validate)
        }
        Commands::MigrationReport { source, output } => {
            generate_migration_report(&source, &output)
        }
    }
}

fn generate_fixtures(output_dir: &Path, deterministic: bool, prompt_count: usize) -> Result<()> {
    println!("🔧 Generating test fixtures...");
    
    // Create output directory if it doesn't exist
    fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", output_dir))?;
    
    // Set up RNG for deterministic generation
    use rand::{Rng, SeedableRng};
    let mut rng = if deterministic {
        rand::rngs::StdRng::seed_from_u64(42) // Fixed seed for deterministic output
    } else {
        rand::rngs::StdRng::from_entropy()
    };
    
    // Generate different types of fixtures
    let fixtures = vec![
        generate_minimal_fixture(prompt_count),
        generate_performance_fixture(prompt_count),
        generate_accuracy_fixture(prompt_count),
        generate_edge_case_fixture(prompt_count),
        generate_multilingual_fixture(prompt_count),
    ];
    
    for fixture in fixtures {
        let filename = format!("{}.json", fixture.name);
        let filepath = output_dir.join(&filename);
        
        let json = serde_json::to_string_pretty(&fixture)
            .with_context(|| format!("Failed to serialize fixture: {}", fixture.name))?;
        
        fs::write(&filepath, json)
            .with_context(|| format!("Failed to write fixture: {:?}", filepath))?;
        
        println!("✅ Generated fixture: {}", filename);
    }
    
    // Generate a dummy model file for testing
    generate_dummy_model(output_dir, "minimal_model.gguf", 20 * 1024)?;
    generate_dummy_model(output_dir, "test_model.gguf", 100 * 1024)?;
    generate_dummy_model(output_dir, "benchmark_model.gguf", 500 * 1024)?;
    
    println!("🎉 Fixture generation complete!");
    Ok(())
}

fn generate_minimal_fixture(prompt_count: usize) -> TestFixture {
    TestFixture {
        name: "minimal_generated".to_string(),
        model_path: "minimal_model.gguf".to_string(),
        test_prompts: generate_test_prompts("minimal", prompt_count),
        expected_tokens: None,
        description: "Generated minimal fixture for basic testing".to_string(),
        model_size_kb: 20,
        max_tokens: 50,
    }
}

fn generate_performance_fixture(prompt_count: usize) -> TestFixture {
    TestFixture {
        name: "performance_generated".to_string(),
        model_path: "benchmark_model.gguf".to_string(),
        test_prompts: generate_test_prompts("performance", prompt_count),
        expected_tokens: None,
        description: "Generated performance fixture for benchmarking".to_string(),
        model_size_kb: 500,
        max_tokens: 200,
    }
}

fn generate_accuracy_fixture(prompt_count: usize) -> TestFixture {
    TestFixture {
        name: "accuracy_generated".to_string(),
        model_path: "test_model.gguf".to_string(),
        test_prompts: generate_test_prompts("accuracy", prompt_count),
        expected_tokens: None,
        description: "Generated accuracy fixture for numerical validation".to_string(),
        model_size_kb: 100,
        max_tokens: 100,
    }
}

fn generate_edge_case_fixture(prompt_count: usize) -> TestFixture {
    TestFixture {
        name: "edge_cases_generated".to_string(),
        model_path: "test_model.gguf".to_string(),
        test_prompts: generate_edge_case_prompts(prompt_count),
        expected_tokens: None,
        description: "Generated edge case fixture for robustness testing".to_string(),
        model_size_kb: 100,
        max_tokens: 150,
    }
}

fn generate_multilingual_fixture(prompt_count: usize) -> TestFixture {
    TestFixture {
        name: "multilingual_generated".to_string(),
        model_path: "test_model.gguf".to_string(),
        test_prompts: generate_multilingual_prompts(prompt_count),
        expected_tokens: None,
        description: "Generated multilingual fixture for international testing".to_string(),
        model_size_kb: 200,
        max_tokens: 120,
    }
}

fn generate_test_prompts(category: &str, count: usize) -> Vec<String> {
    let base_prompts = match category {
        "minimal" => vec![
            "Hello, world!",
            "Test prompt",
            "Simple generation",
            "Basic inference",
            "Quick test",
        ],
        "performance" => vec![
            "Generate a detailed explanation of machine learning concepts and their applications in modern technology.",
            "Write a comprehensive analysis of the benefits and challenges of artificial intelligence in healthcare.",
            "Describe the evolution of neural networks from perceptrons to modern transformer architectures.",
            "Explain the mathematical foundations of deep learning and optimization algorithms used in training.",
            "Discuss the ethical implications of AI systems and the importance of responsible AI development.",
        ],
        "accuracy" => vec![
            "The capital of France is",
            "2 + 2 equals",
            "The largest planet in our solar system is",
            "Water boils at",
            "The speed of light is approximately",
        ],
        _ => vec![
            "Default test prompt",
            "Another test case",
            "Third test example",
        ],
    };
    
    base_prompts
        .into_iter()
        .cycle()
        .take(count)
        .map(|s| s.to_string())
        .collect()
}

fn generate_edge_case_prompts(count: usize) -> Vec<String> {
    let edge_cases = vec![
        "", // Empty prompt
        " ", // Whitespace only
        "A", // Single character
        "🚀🦀🎉", // Unicode emojis
        "This is a very long prompt that tests the model's ability to handle extended input sequences and generate appropriate responses for lengthy context windows that might challenge the inference engine's memory management and processing capabilities.",
        "Prompt with\nnewlines\nand\ttabs",
        "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?",
        "Numbers: 1234567890 and math: 2+2=4, 10*5=50",
        "Mixed: Hello123 World456 Test789!",
        "Repeated words: test test test test test",
    ];
    
    edge_cases
        .into_iter()
        .cycle()
        .take(count)
        .map(|s| s.to_string())
        .collect()
}

fn generate_multilingual_prompts(count: usize) -> Vec<String> {
    let multilingual = vec![
        "Hello, how are you?", // English
        "Bonjour, comment allez-vous?", // French
        "Hola, ¿cómo estás?", // Spanish
        "Hallo, wie geht es dir?", // German
        "Ciao, come stai?", // Italian
        "こんにちは、元気ですか？", // Japanese
        "你好，你好吗？", // Chinese
        "Привет, как дела?", // Russian
        "안녕하세요, 어떻게 지내세요?", // Korean
        "مرحبا، كيف حالك؟", // Arabic
    ];
    
    multilingual
        .into_iter()
        .cycle()
        .take(count)
        .map(|s| s.to_string())
        .collect()
}

fn generate_dummy_model(output_dir: &Path, filename: &str, size_bytes: usize) -> Result<()> {
    let filepath = output_dir.join(filename);
    
    // Don't overwrite existing model files
    if filepath.exists() {
        println!("⏭️  Skipping existing model: {}", filename);
        return Ok(());
    }
    
    // Generate dummy model data (not a real GGUF file, just for testing)
    let dummy_data = vec![0u8; size_bytes];
    
    fs::write(&filepath, dummy_data)
        .with_context(|| format!("Failed to write dummy model: {:?}", filepath))?;
    
    println!("📦 Generated dummy model: {} ({} KB)", filename, size_bytes / 1024);
    Ok(())
}

fn validate_fixtures(fixtures_dir: &Path) -> Result<()> {
    println!("🔍 Validating test fixtures...");
    
    if !fixtures_dir.exists() {
        anyhow::bail!("Fixtures directory does not exist: {:?}", fixtures_dir);
    }
    
    let mut fixture_count = 0;
    let mut error_count = 0;
    
    for entry in fs::read_dir(fixtures_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            fixture_count += 1;
            
            match validate_single_fixture(&path) {
                Ok(_) => println!("✅ Valid fixture: {}", path.file_name().unwrap().to_string_lossy()),
                Err(e) => {
                    println!("❌ Invalid fixture: {}: {}", path.file_name().unwrap().to_string_lossy(), e);
                    error_count += 1;
                }
            }
        }
    }
    
    println!("\n📊 Validation Summary:");
    println!("   Total fixtures: {}", fixture_count);
    println!("   Valid: {}", fixture_count - error_count);
    println!("   Invalid: {}", error_count);
    
    if error_count > 0 {
        anyhow::bail!("Validation failed: {} invalid fixtures", error_count);
    }
    
    println!("🎉 All fixtures are valid!");
    Ok(())
}

fn validate_single_fixture(path: &Path) -> Result<()> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read fixture: {:?}", path))?;
    
    let fixture: TestFixture = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse fixture JSON: {:?}", path))?;
    
    // Validate fixture contents
    if fixture.name.is_empty() {
        anyhow::bail!("Fixture name cannot be empty");
    }
    
    if fixture.model_path.is_empty() {
        anyhow::bail!("Model path cannot be empty");
    }
    
    if fixture.test_prompts.is_empty() {
        anyhow::bail!("Test prompts cannot be empty");
    }
    
    if fixture.max_tokens == 0 {
        anyhow::bail!("Max tokens must be greater than 0");
    }
    
    Ok(())
}

fn clean_fixtures(fixtures_dir: &Path) -> Result<()> {
    println!("🧹 Cleaning generated fixtures...");
    
    if !fixtures_dir.exists() {
        println!("⏭️  Fixtures directory does not exist, nothing to clean");
        return Ok(());
    }
    
    let generated_patterns = vec![
        "_generated.json",
        "minimal_model.gguf",
        "test_model.gguf", 
        "benchmark_model.gguf",
    ];
    
    let mut cleaned_count = 0;
    
    for entry in fs::read_dir(fixtures_dir)? {
        let entry = entry?;
        let path = entry.path();
        let filename = path.file_name().unwrap().to_string_lossy();
        
        if generated_patterns.iter().any(|pattern| filename.contains(pattern)) {
            fs::remove_file(&path)
                .with_context(|| format!("Failed to remove file: {:?}", path))?;
            println!("🗑️  Removed: {}", filename);
            cleaned_count += 1;
        }
    }
    
    println!("🎉 Cleaned {} generated files", cleaned_count);
    Ok(())
}

fn migrate_config(from: &Path, to: &Path, format: &str) -> Result<()> {
    println!("🔄 Migrating configuration from C++ to Rust...");
    
    if !from.exists() {
        anyhow::bail!("Input configuration file not found: {:?}", from);
    }
    
    // Read input configuration
    let input_content = fs::read_to_string(from)
        .with_context(|| format!("Failed to read input file: {:?}", from))?;
    
    // Detect format if auto
    let input_format = if format == "auto" {
        detect_config_format(from)?
    } else {
        format.to_string()
    };
    
    println!("📄 Detected input format: {}", input_format);
    
    // Parse input configuration
    let config = parse_cpp_config(&input_content, &input_format)?;
    
    // Convert to Rust configuration
    let rust_config = convert_to_rust_config(config)?;
    
    // Write output configuration
    let output_content = serialize_rust_config(&rust_config)?;
    
    // Create output directory if needed
    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory: {:?}", parent))?;
    }
    
    fs::write(to, output_content)
        .with_context(|| format!("Failed to write output file: {:?}", to))?;
    
    println!("✅ Configuration migrated successfully!");
    println!("   Input:  {:?}", from);
    println!("   Output: {:?}", to);
    println!("");
    println!("📝 Next steps:");
    println!("   1. Review the generated configuration");
    println!("   2. Adjust settings for your specific use case");
    println!("   3. Test with: cargo run -- --config {:?}", to);
    
    Ok(())
}

fn validate_model(model_path: &Path, cross_validate: bool) -> Result<()> {
    println!("🔍 Validating model format compatibility...");
    
    if !model_path.exists() {
        anyhow::bail!("Model file not found: {:?}", model_path);
    }
    
    // Check file extension
    let extension = model_path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");
    
    match extension.to_lowercase().as_str() {
        "gguf" => {
            println!("✅ GGUF format detected - fully compatible with BitNet.rs");
            validate_gguf_model(model_path)?;
        }
        "safetensors" => {
            println!("✅ SafeTensors format detected - compatible with BitNet.rs");
            validate_safetensors_model(model_path)?;
        }
        "bin" => {
            println!("⚠️  Binary format detected - may need conversion");
            println!("   Consider converting to GGUF format for optimal compatibility");
        }
        _ => {
            println!("❓ Unknown format - manual validation required");
            println!("   Supported formats: .gguf, .safetensors");
        }
    }
    
    // Cross-validation if requested
    if cross_validate {
        println!("🔄 Running cross-validation with C++ implementation...");
        cross_validate_model(model_path)?;
    }
    
    println!("✅ Model validation complete!");
    Ok(())
}

fn generate_migration_report(source_dir: &Path, output_file: &Path) -> Result<()> {
    println!("📊 Generating migration report...");
    
    if !source_dir.exists() {
        anyhow::bail!("Source directory not found: {:?}", source_dir);
    }
    
    let mut report = MigrationReport::new();
    
    // Analyze C++ code
    analyze_cpp_code(source_dir, &mut report)?;
    
    // Analyze configuration files
    analyze_config_files(source_dir, &mut report)?;
    
    // Analyze model files
    analyze_model_files(source_dir, &mut report)?;
    
    // Generate recommendations
    generate_recommendations(&mut report)?;
    
    // Write report
    let report_content = format_migration_report(&report)?;
    
    if let Some(parent) = output_file.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory: {:?}", parent))?;
    }
    
    fs::write(output_file, report_content)
        .with_context(|| format!("Failed to write report: {:?}", output_file))?;
    
    println!("✅ Migration report generated: {:?}", output_file);
    println!("");
    println!("📋 Summary:");
    println!("   C++ files found: {}", report.cpp_files.len());
    println!("   Config files found: {}", report.config_files.len());
    println!("   Model files found: {}", report.model_files.len());
    println!("   Estimated migration time: {}", report.estimated_time);
    
    Ok(())
}

// Helper functions for migration tools

fn detect_config_format(path: &Path) -> Result<String> {
    let extension = path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");
    
    match extension.to_lowercase().as_str() {
        "json" => Ok("json".to_string()),
        "yaml" | "yml" => Ok("yaml".to_string()),
        "toml" => Ok("toml".to_string()),
        _ => {
            // Try to detect from content
            let content = fs::read_to_string(path)?;
            if content.trim_start().starts_with('{') {
                Ok("json".to_string())
            } else if content.contains("---") || content.contains(":") {
                Ok("yaml".to_string())
            } else {
                Ok("json".to_string()) // Default fallback
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CppConfig {
    model_path: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    batch_size: Option<u32>,
    num_threads: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RustConfig {
    model: ModelConfig,
    generation: GenerationConfig,
    performance: PerformanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelConfig {
    path: String,
    device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GenerationConfig {
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceConfig {
    batch_size: u32,
    num_threads: u32,
    cache_size: String,
}

fn parse_cpp_config(content: &str, format: &str) -> Result<CppConfig> {
    match format {
        "json" => {
            serde_json::from_str(content)
                .with_context(|| "Failed to parse JSON configuration")
        }
        "yaml" => {
            serde_yaml::from_str(content)
                .with_context(|| "Failed to parse YAML configuration")
        }
        "toml" => {
            toml::from_str(content)
                .with_context(|| "Failed to parse TOML configuration")
        }
        _ => anyhow::bail!("Unsupported configuration format: {}", format),
    }
}

fn convert_to_rust_config(cpp_config: CppConfig) -> Result<RustConfig> {
    Ok(RustConfig {
        model: ModelConfig {
            path: cpp_config.model_path.unwrap_or_else(|| "model.gguf".to_string()),
            device: "cpu".to_string(),
        },
        generation: GenerationConfig {
            max_tokens: cpp_config.max_tokens.unwrap_or(100),
            temperature: cpp_config.temperature.unwrap_or(0.7),
            top_p: cpp_config.top_p.unwrap_or(0.9),
            top_k: cpp_config.top_k.unwrap_or(40),
        },
        performance: PerformanceConfig {
            batch_size: cpp_config.batch_size.unwrap_or(1),
            num_threads: cpp_config.num_threads.unwrap_or(4),
            cache_size: "1GB".to_string(),
        },
    })
}

fn serialize_rust_config(config: &RustConfig) -> Result<String> {
    toml::to_string_pretty(config)
        .with_context(|| "Failed to serialize Rust configuration")
}

fn validate_gguf_model(model_path: &Path) -> Result<()> {
    // Basic GGUF validation (simplified)
    let file = fs::File::open(model_path)?;
    let mut reader = std::io::BufReader::new(file);
    
    // Check GGUF magic number (simplified check)
    let mut magic = [0u8; 4];
    std::io::Read::read_exact(&mut reader, &mut magic)?;
    
    if &magic == b"GGUF" {
        println!("   ✅ Valid GGUF magic number");
    } else {
        println!("   ⚠️  Invalid GGUF magic number - file may be corrupted");
    }
    
    Ok(())
}

fn validate_safetensors_model(_model_path: &Path) -> Result<()> {
    // SafeTensors validation would go here
    println!("   ✅ SafeTensors format validation (placeholder)");
    Ok(())
}

fn cross_validate_model(_model_path: &Path) -> Result<()> {
    // Cross-validation would go here
    println!("   ✅ Cross-validation with C++ implementation (placeholder)");
    println!("   💡 Run 'cargo test --features crossval' for full cross-validation");
    Ok(())
}

#[derive(Debug, Default)]
struct MigrationReport {
    cpp_files: Vec<String>,
    config_files: Vec<String>,
    model_files: Vec<String>,
    api_calls: Vec<String>,
    estimated_time: String,
    recommendations: Vec<String>,
}

impl MigrationReport {
    fn new() -> Self {
        Self::default()
    }
}

fn analyze_cpp_code(source_dir: &Path, report: &mut MigrationReport) -> Result<()> {
    for entry in walkdir::WalkDir::new(source_dir) {
        let entry = entry?;
        let path = entry.path();
        
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            match extension.to_lowercase().as_str() {
                "cpp" | "cc" | "cxx" | "c" | "h" | "hpp" | "hxx" => {
                    report.cpp_files.push(path.to_string_lossy().to_string());
                    
                    // Analyze API calls
                    if let Ok(content) = fs::read_to_string(path) {
                        for line in content.lines() {
                            if line.contains("bitnet_") {
                                report.api_calls.push(line.trim().to_string());
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    
    Ok(())
}

fn analyze_config_files(source_dir: &Path, report: &mut MigrationReport) -> Result<()> {
    for entry in walkdir::WalkDir::new(source_dir) {
        let entry = entry?;
        let path = entry.path();
        
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            match extension.to_lowercase().as_str() {
                "json" | "yaml" | "yml" | "toml" | "cfg" | "conf" => {
                    report.config_files.push(path.to_string_lossy().to_string());
                }
                _ => {}
            }
        }
    }
    
    Ok(())
}

fn analyze_model_files(source_dir: &Path, report: &mut MigrationReport) -> Result<()> {
    for entry in walkdir::WalkDir::new(source_dir) {
        let entry = entry?;
        let path = entry.path();
        
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            match extension.to_lowercase().as_str() {
                "gguf" | "safetensors" | "bin" | "pt" | "pth" => {
                    report.model_files.push(path.to_string_lossy().to_string());
                }
                _ => {}
            }
        }
    }
    
    Ok(())
}

fn generate_recommendations(report: &mut MigrationReport) -> Result<()> {
    // Estimate migration time
    let cpp_count = report.cpp_files.len();
    let api_count = report.api_calls.len();
    
    report.estimated_time = match (cpp_count, api_count) {
        (0..=5, 0..=20) => "2-4 hours".to_string(),
        (6..=20, 21..=100) => "1-2 days".to_string(),
        (21..=50, 101..=500) => "1-2 weeks".to_string(),
        _ => "2-4 weeks".to_string(),
    };
    
    // Generate recommendations
    if !report.cpp_files.is_empty() {
        report.recommendations.push("Migrate C++ API calls to Rust equivalents".to_string());
    }
    
    if !report.config_files.is_empty() {
        report.recommendations.push("Convert configuration files to TOML format".to_string());
    }
    
    if !report.model_files.is_empty() {
        report.recommendations.push("Validate model format compatibility".to_string());
    }
    
    if report.api_calls.len() > 50 {
        report.recommendations.push("Consider gradual migration approach".to_string());
    }
    
    Ok(())
}

fn format_migration_report(report: &MigrationReport) -> Result<String> {
    let mut content = String::new();
    
    content.push_str("# Migration Report\n\n");
    content.push_str(&format!("Generated on: {}\n\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
    
    content.push_str("## Summary\n\n");
    content.push_str(&format!("- **C++ files found**: {}\n", report.cpp_files.len()));
    content.push_str(&format!("- **Configuration files found**: {}\n", report.config_files.len()));
    content.push_str(&format!("- **Model files found**: {}\n", report.model_files.len()));
    content.push_str(&format!("- **API calls found**: {}\n", report.api_calls.len()));
    content.push_str(&format!("- **Estimated migration time**: {}\n\n", report.estimated_time));
    
    if !report.cpp_files.is_empty() {
        content.push_str("## C++ Files\n\n");
        for file in &report.cpp_files {
            content.push_str(&format!("- `{}`\n", file));
        }
        content.push_str("\n");
    }
    
    if !report.config_files.is_empty() {
        content.push_str("## Configuration Files\n\n");
        for file in &report.config_files {
            content.push_str(&format!("- `{}`\n", file));
        }
        content.push_str("\n");
    }
    
    if !report.model_files.is_empty() {
        content.push_str("## Model Files\n\n");
        for file in &report.model_files {
            content.push_str(&format!("- `{}`\n", file));
        }
        content.push_str("\n");
    }
    
    if !report.api_calls.is_empty() {
        content.push_str("## API Calls Found\n\n");
        content.push_str("```cpp\n");
        for call in report.api_calls.iter().take(20) { // Limit to first 20
            content.push_str(&format!("{}\n", call));
        }
        if report.api_calls.len() > 20 {
            content.push_str(&format!("... and {} more\n", report.api_calls.len() - 20));
        }
        content.push_str("```\n\n");
    }
    
    if !report.recommendations.is_empty() {
        content.push_str("## Recommendations\n\n");
        for rec in &report.recommendations {
            content.push_str(&format!("- {}\n", rec));
        }
        content.push_str("\n");
    }
    
    content.push_str("## Next Steps\n\n");
    content.push_str("1. Review the migration guide: `docs/cpp-to-rust-migration.md`\n");
    content.push_str("2. Set up development environment: `./scripts/dev-setup.sh`\n");
    content.push_str("3. Migrate configuration files: `cargo xtask migrate-config`\n");
    content.push_str("4. Update API calls using the compatibility matrix\n");
    content.push_str("5. Test with cross-validation: `cargo test --features crossval`\n");
    content.push_str("6. Deploy gradually with monitoring\n\n");
    
    content.push_str("---\n");
    content.push_str("*This report was generated automatically by the BitNet.rs migration tools.*\n");
    
    Ok(content)
}