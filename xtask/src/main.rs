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