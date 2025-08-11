//! BitNet CLI application
//! 
//! A comprehensive command-line interface for BitNet 1-bit LLM inference.
//! Supports model loading, inference, conversion, benchmarking, and serving.

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, Shell};
use console::style;
use std::io;
use tracing::{error, info};
use candle_core::{DType, IndexOp};

#[cfg(feature = "full-cli")]
mod commands;
mod config;
mod sampling;

#[cfg(feature = "full-cli")]
use commands::{BenchmarkCommand, ConvertCommand, InferenceCommand, ServeCommand};
use config::{CliConfig, ConfigBuilder};

/// BitNet CLI - High-performance 1-bit LLM inference toolkit
#[derive(Parser)]
#[command(name = "bitnet")]
#[command(about = "BitNet 1-bit LLM inference toolkit")]
#[command(long_about = r#"
BitNet is a high-performance inference framework for 1-bit Large Language Models.
This CLI provides comprehensive tools for model inference, conversion, benchmarking,
and serving with support for multiple quantization formats and hardware acceleration.

Examples:
  # Run inference with a model
  bitnet inference --model model.gguf --prompt "Hello, world!"
  
  # Interactive mode
  bitnet inference --model model.gguf --interactive
  
  # Batch processing
  bitnet inference --model model.gguf --input-file prompts.txt
  
  # Convert model formats
  bitnet convert --input model.safetensors --output model.gguf
  
  # Benchmark performance
  bitnet benchmark --model model.gguf --device cuda
  
  # Start inference server
  bitnet serve --model model.gguf --port 8080

For more information, visit: https://github.com/microsoft/BitNet
"#)]
#[command(version)]
#[command(author = "BitNet Contributors")]
struct Cli {
    /// Configuration file path
    #[arg(short, long, value_name = "PATH", global = true)]
    config: Option<std::path::PathBuf>,
    
    /// Device to use (cpu, cuda, auto)
    #[arg(short, long, value_name = "DEVICE", global = true)]
    device: Option<String>,
    
    /// Log level (trace, debug, info, warn, error)
    #[arg(long, value_name = "LEVEL", global = true)]
    log_level: Option<String>,
    
    /// Number of CPU threads
    #[arg(long, value_name = "N", global = true)]
    threads: Option<usize>,
    
    /// Batch size for processing
    #[arg(long, value_name = "SIZE", global = true)]
    batch_size: Option<usize>,
    
    /// Generate shell completions
    #[arg(long, value_name = "SHELL")]
    completions: Option<Shell>,
    
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run simple text generation
    Run {
        /// Model file path
        #[arg(short, long)]
        model: std::path::PathBuf,
        
        /// Tokenizer file path (optional, will look for sibling file if not provided)
        #[arg(long)]
        tokenizer: Option<std::path::PathBuf>,
        
        /// Input prompt
        #[arg(short, long)]
        prompt: String,
        
        /// Maximum new tokens to generate
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
        
        /// Temperature for sampling (0 = greedy)
        #[arg(long, default_value_t = 1.0)]
        temperature: f32,
        
        /// Top-k sampling (0 = disabled)
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        
        /// Top-p (nucleus) sampling
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        
        /// Repetition penalty
        #[arg(long, default_value_t = 1.1)]
        repetition_penalty: f32,
        
        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u64>,
    },
    
    #[cfg(feature = "full-cli")]
    /// Run inference on a model
    #[command(alias = "infer")]
    Inference(InferenceCommand),
    
    #[cfg(feature = "full-cli")]
    /// Convert between model formats
    #[command(alias = "conv")]
    Convert(ConvertCommand),
    
    #[cfg(feature = "full-cli")]
    /// Benchmark model performance
    #[command(alias = "bench")]
    Benchmark(BenchmarkCommand),
    
    #[cfg(feature = "full-cli")]
    /// Start inference server
    #[command(alias = "server")]
    Serve(ServeCommand),
    
    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    
    /// Show system information
    Info,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,
    /// Set configuration value
    Set {
        /// Configuration key
        key: String,
        /// Configuration value
        value: String,
    },
    /// Reset configuration to defaults
    Reset,
    /// Show configuration file path
    Path,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse CLI arguments
    let cli = Cli::parse();
    
    // Handle shell completions
    if let Some(shell) = cli.completions {
        generate_completions(shell);
        return Ok(());
    }
    
    // Load configuration
    let config = load_configuration(&cli).await?;
    
    // Setup logging
    setup_logging(&config, cli.log_level.as_deref())?;
    
    // Handle commands
    let result = match cli.command {
        Some(Commands::Run { model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty, seed }) => {
            run_simple_generation(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty, seed).await
        }
        #[cfg(feature = "full-cli")]
        Some(Commands::Inference(cmd)) => cmd.execute(&config).await,
        #[cfg(feature = "full-cli")]
        Some(Commands::Convert(cmd)) => cmd.execute(&config).await,
        #[cfg(feature = "full-cli")]
        Some(Commands::Benchmark(cmd)) => cmd.execute(&config).await,
        #[cfg(feature = "full-cli")]
        Some(Commands::Serve(cmd)) => cmd.execute(&config).await,
        Some(Commands::Config { action }) => handle_config_command(action, &config).await,
        Some(Commands::Info) => show_system_info().await,
        None => {
            // No command provided, show help
            let mut cmd = Cli::command();
            cmd.print_help()?;
            Ok(())
        }
    };
    
    // Handle errors gracefully
    if let Err(e) = result {
        error!("Command failed: {}", e);
        
        // Show error chain
        let mut source = e.source();
        while let Some(err) = source {
            error!("  Caused by: {}", err);
            source = err.source();
        }
        
        std::process::exit(1);
    }
    
    Ok(())
}

/// Load configuration from file and merge with CLI arguments
async fn load_configuration(cli: &Cli) -> Result<CliConfig> {
    let config_path = if let Some(path) = &cli.config {
        path.clone()
    } else {
        CliConfig::default_config_path()
            .unwrap_or_else(|_| std::path::PathBuf::from("bitnet.toml"))
    };
    
    let config = ConfigBuilder::from_file(&config_path)
        .unwrap_or_else(|_| {
            info!("Using default configuration");
            ConfigBuilder::new()
        })
        .device(cli.device.clone())
        .log_level(cli.log_level.clone())
        .cpu_threads(cli.threads)
        .batch_size(cli.batch_size)
        .build()
        .context("Failed to build configuration")?;
    
    Ok(config)
}

/// Setup logging based on configuration
fn setup_logging(config: &CliConfig, log_level_override: Option<&str>) -> Result<()> {
    let level = log_level_override.unwrap_or(&config.logging.level);
    
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level));
    
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false);
    
    match config.logging.format.as_str() {
        "json" => {
            subscriber
                .json()
                .with_timer(tracing_subscriber::fmt::time::uptime())
                .init();
        }
        "compact" => {
            subscriber
                .compact()
                .init();
        }
        _ => {
            subscriber
                .pretty()
                .init();
        }
    }
    
    Ok(())
}

/// Generate shell completions
fn generate_completions(shell: Shell) {
    let mut cmd = Cli::command();
    let name = cmd.get_name().to_string();
    generate(shell, &mut cmd, name, &mut io::stdout());
}

/// Handle configuration commands
async fn handle_config_command(action: ConfigAction, config: &CliConfig) -> Result<()> {
    match action {
        ConfigAction::Show => {
            let config_str = toml::to_string_pretty(config)
                .context("Failed to serialize configuration")?;
            println!("{}", config_str);
        }
        ConfigAction::Set { key, value } => {
            println!("Setting {} = {}", key, value);
            // In a full implementation, this would update the config file
            println!("{}", style("Configuration setting not yet implemented").yellow());
        }
        ConfigAction::Reset => {
            println!("Resetting configuration to defaults");
            // In a full implementation, this would reset the config file
            println!("{}", style("Configuration reset not yet implemented").yellow());
        }
        ConfigAction::Path => {
            let path = CliConfig::default_config_path()
                .unwrap_or_else(|_| std::path::PathBuf::from("bitnet.toml"));
            println!("{}", path.display());
        }
    }
    Ok(())
}

/// Run text generation with sampling
async fn run_simple_generation(
    model_path: std::path::PathBuf,
    tokenizer_path: Option<std::path::PathBuf>,
    prompt: String,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
) -> Result<()> {
    use bitnet_models::{Model, transformer::KVCache};
    use bitnet_tokenizers::Tokenizer;
    use bitnet_common::Device;
    use std::sync::Arc;
    use crate::sampling::Sampler;
    
    println!("Loading model from: {}", model_path.display());
    
    // Load model with simplified GGUF loader
    let (config, tensors) = bitnet_models::gguf_simple::load_gguf(&model_path, Device::Cpu)
        .context("Failed to load GGUF model")?;
    
    let model = bitnet_models::BitNetModel::from_gguf(config.clone(), tensors, Device::Cpu)
        .context("Failed to build model from GGUF")?;
    let model = Arc::new(model) as Arc<dyn Model>;
    
    // Load tokenizer
    let tokenizer_path = tokenizer_path.or_else(|| {
        // Look for common tokenizer file names
        let base = model_path.with_extension("");
        for ext in &["tokenizer.json", "tokenizer.model", "vocab.json"] {
            let path = base.with_extension(ext);
            if path.exists() {
                return Some(path);
            }
        }
        None
    });
    
    let tokenizer = if let Some(path) = tokenizer_path {
        println!("Loading tokenizer from: {}", path.display());
        // For now just use mock tokenizer
        println!("Warning: Using mock tokenizer (real tokenizer loading not yet implemented)");
        Box::new(bitnet_tokenizers::MockTokenizer::new()) as Box<dyn Tokenizer>
    } else {
        println!("Warning: No tokenizer found, using mock tokenizer");
        Box::new(bitnet_tokenizers::MockTokenizer::new()) as Box<dyn Tokenizer>
    };
    
    // Tokenize prompt
    let mut tokens = tokenizer.encode(&prompt, true)?;
    println!("Input tokens ({}): {:?}", tokens.len(), &tokens[..10.min(tokens.len())]);
    
    // Create KV cache
    let cache = KVCache::new(&config, 1, &candle_core::Device::Cpu)?;
    let mut any_cache: Box<dyn std::any::Any> = Box::new(cache);
    
    // Create sampler
    let mut sampler = Sampler::new(temperature, top_k, top_p, repetition_penalty, seed);
    
    print!("Generating: {}", prompt);
    std::io::Write::flush(&mut std::io::stdout())?;
    
    // Track generated tokens for repetition penalty
    let mut generated_tokens = Vec::new();
    
    // Generation loop
    for _ in 0..max_new_tokens {
        // Embed tokens
        let x = model.embed(&tokens)?;
        
        // Forward pass
        let h = model.forward(&x, any_cache.as_mut())?;
        
        // Get logits
        let logits = model.logits(&h)?;
        
        // Extract last token logits
        let logits_vec = extract_logits(&logits)?;
        
        // Sample next token
        let next_token = sampler.sample(&logits_vec, &generated_tokens);
        tokens.push(next_token);
        generated_tokens.push(next_token);
        
        // Decode and print the new token
        let token_text = tokenizer.decode(&[next_token], false)?;
        print!("{}", token_text);
        std::io::Write::flush(&mut std::io::stdout())?;
        
        // Check for EOS
        if let Some(eos) = tokenizer.eos_token_id() {
            if next_token == eos {
                break;
            }
        }
    }
    
    println!("\n\nGeneration complete!");
    println!("Generated {} tokens", generated_tokens.len());
    Ok(())
}

/// Extract logits vector from tensor
fn extract_logits(tensor: &bitnet_common::ConcreteTensor) -> Result<Vec<f32>> {
    use bitnet_common::{BitNetError, ConcreteTensor, Tensor};
    
    let shape = tensor.shape();
    if shape.len() != 3 {
        return Err(BitNetError::Validation("Expected 3D tensor".into()).into());
    }
    
    let (_batch, seq_len, _vocab) = (shape[0], shape[1], shape[2]);
    
    match tensor {
        ConcreteTensor::BitNet(t) => {
            let candle = t.to_candle()?;
            let last = candle
                .narrow(1, seq_len - 1, 1)?
                .squeeze(1)?
                .i(0)?;
            let last = if last.dtype() != DType::F32 { last.to_dtype(DType::F32)? } else { last };
            Ok(last.to_vec1::<f32>()?)
        }
        ConcreteTensor::Mock(_) => {
            // Return mock logits for testing
            Ok(vec![0.1; 50257])
        }
    }
}


/// Show system information
async fn show_system_info() -> Result<()> {
    println!("{}", style("BitNet System Information").bold().cyan());
    println!();
    
    // Version information
    println!("{}", style("Version:").bold());
    println!("  BitNet CLI: {}", env!("CARGO_PKG_VERSION"));
    println!("  Rust: {}", std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()));
    println!();
    
    // System information
    println!("{}", style("System:").bold());
    println!("  OS: {}", std::env::consts::OS);
    println!("  Architecture: {}", std::env::consts::ARCH);
    println!("  CPU cores: {}", num_cpus::get());
    println!();
    
    // Feature information
    println!("{}", style("Features:").bold());
    #[cfg(feature = "gpu")]
    {
        println!("  GPU support: {}", style("✓ Enabled").green());
        // Check CUDA availability
        match bitnet_kernels::gpu::is_cuda_available() {
            true => println!("  CUDA: {}", style("✓ Available").green()),
            false => println!("  CUDA: {}", style("✗ Not available").red()),
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        println!("  GPU support: {}", style("✗ Disabled").red());
    }
    
    // CPU features
    println!("  CPU features:");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            println!("    AVX2: {}", style("✓").green());
        } else {
            println!("    AVX2: {}", style("✗").red());
        }
        if is_x86_feature_detected!("avx512f") {
            println!("    AVX-512: {}", style("✓").green());
        } else {
            println!("    AVX-512: {}", style("✗").red());
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            println!("    NEON: {}", style("✓").green());
        } else {
            println!("    NEON: {}", style("✗").red());
        }
    }
    
    println!();
    
    // Model formats
    println!("{}", style("Supported formats:").bold());
    println!("  GGUF: {}", style("✓").green());
    println!("  SafeTensors: {}", style("✓").green());
    println!("  HuggingFace: {}", style("✓").green());
    println!();
    
    // Quantization types
    println!("{}", style("Quantization types:").bold());
    println!("  I2_S (2-bit signed): {}", style("✓").green());
    println!("  TL1 (ARM optimized): {}", style("✓").green());
    println!("  TL2 (x86 optimized): {}", style("✓").green());
    
    Ok(())
}