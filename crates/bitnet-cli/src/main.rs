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

mod commands;
mod config;

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
    /// Run inference on a model
    #[command(alias = "infer")]
    Inference(InferenceCommand),
    
    /// Convert between model formats
    #[command(alias = "conv")]
    Convert(ConvertCommand),
    
    /// Benchmark model performance
    #[command(alias = "bench")]
    Benchmark(BenchmarkCommand),
    
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
        Some(Commands::Inference(cmd)) => cmd.execute(&config).await,
        Some(Commands::Convert(cmd)) => cmd.execute(&config).await,
        Some(Commands::Benchmark(cmd)) => cmd.execute(&config).await,
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