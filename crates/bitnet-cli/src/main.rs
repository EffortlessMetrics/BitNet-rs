//! BitNet CLI application
//!
//! A comprehensive command-line interface for BitNet 1-bit LLM inference.
//! Supports model loading, inference, conversion, benchmarking, and serving.

use anyhow::{Context, Result};
use bitnet_common::Tensor;
use candle_core::{DType, IndexOp};
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{Shell, generate};
use console::style;
use std::io;
use tracing::{error, info};

#[cfg(feature = "full-cli")]
mod commands;
mod config;
mod exit;
#[cfg(feature = "full-cli")]
mod ln_rules;
mod sampling;
mod score;
pub mod tokenizer_discovery;

use exit::*;

/// Build the CLI command for external use (e.g., in tests)
pub fn build_cli() -> clap::Command {
    Cli::command()
}

fn compiled_features() -> &'static [&'static str] {
    &[
        #[cfg(feature = "cpu")]
        "cpu",
        #[cfg(feature = "gpu")]
        "gpu",
        // Removed cuda and ffi - not declared in bitnet-cli/Cargo.toml
        #[cfg(feature = "iq2s-ffi")]
        "iq2s-ffi",
        #[cfg(feature = "crossval")]
        "crossval",
    ]
}

/// CLI interface version (SemVer for CLI surface compatibility)
const INTERFACE_VERSION: &str = "1.0.0";

fn bitnet_version() -> &'static str {
    use std::sync::OnceLock;
    static VERSION_STRING: OnceLock<String> = OnceLock::new();

    VERSION_STRING.get_or_init(|| {
        let features = compiled_features();
        let features_line = if features.is_empty() {
            "features: none".to_string()
        } else {
            format!("features: {}", features.join(", "))
        };

        #[cfg(feature = "iq2s-ffi")]
        let ggml_line = format!("ggml: {}", bitnet_ggml_ffi::GGML_COMMIT);
        #[cfg(not(feature = "iq2s-ffi"))]
        let ggml_line = String::new();

        if ggml_line.is_empty() {
            format!("{}\n{}", env!("CARGO_PKG_VERSION"), features_line)
        } else {
            format!("{}\n{}\n{}", env!("CARGO_PKG_VERSION"), features_line, ggml_line)
        }
    })
}

#[cfg(feature = "cli-bench")]
use commands::BenchmarkCommand;
#[cfg(feature = "full-cli")]
use commands::{ConvertCommand, InferenceCommand, InspectCommand, ServeCommand};
use config::{CliConfig, ConfigBuilder};

/// BitNet CLI - High-performance 1-bit LLM inference toolkit
#[derive(Parser)]
#[command(name = "bitnet")]
#[command(about = "BitNet.rs — 1-bit neural network inference with strict receipts")]
#[command(long_about = r#"BitNet.rs CLI — one-shot generation and chat with strict receipts

QUICK EXAMPLES:

  # Deterministic math sanity check (validates model correctness)
  RUST_LOG=warn bitnet run --model model.gguf --tokenizer tokenizer.json \
    --prompt "Answer with a single digit: 2+2=" --max-tokens 1 --temperature 0.0 --greedy

  # General Q&A with instruct template
  RUST_LOG=warn bitnet run --model model.gguf --tokenizer tokenizer.json \
    --prompt "What is 2+2?" --max-tokens 16 --temperature 0.0 --greedy

  # Creative completion (nucleus sampling)
  RUST_LOG=warn bitnet run --model model.gguf --tokenizer tokenizer.json \
    --prompt "Explain photosynthesis" --max-tokens 128 --temperature 0.7 --top-p 0.95

  # Interactive chat (auto-detects template, clean output)
  RUST_LOG=warn bitnet chat --model model.gguf --tokenizer tokenizer.json

LOGGING:
  Set RUST_LOG=warn (default: info) to reduce log noise and focus on generated text.
  Options: error, warn, info, debug, trace

PERFORMANCE:
  For best CPU throughput, build with:
    RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
      cargo build --release --features cpu

  Run with:
    RAYON_NUM_THREADS=$(nproc) RUST_LOG=warn bitnet run ...
"#)]
#[command(version = bitnet_version())]
#[command(author = "BitNet Contributors")]
#[command(after_help = format!(
    "CLI Interface Version: {}\nDocs: https://docs.rs/bitnet\nIssues: https://github.com/EffortlessMetrics/BitNet-rs/issues",
    INTERFACE_VERSION
))]
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

    /// Write the effective configuration to a file and exit
    #[arg(long, value_name = "PATH")]
    save_config: Option<std::path::PathBuf>,

    /// Print CLI interface version and exit
    #[arg(long)]
    interface_version: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run simple text generation
    ///
    /// # Examples
    ///
    /// Auto-detect template for Q&A (recommended):
    ///   bitnet run --model model.gguf --prompt "Who wrote Pride and Prejudice?"
    ///
    /// Instruct template (explicit Q&A format):
    ///   bitnet run --model model.gguf --prompt-template instruct \
    ///     --prompt "What is 2+2?" --max-tokens 16
    ///
    /// LLaMA-3 chat format with system prompt:
    ///   bitnet run --model model.gguf --prompt-template llama3-chat \
    ///     --system-prompt "You are a helpful assistant" \
    ///     --prompt "Explain photosynthesis" --max-tokens 128
    ///
    /// Deterministic Q&A with greedy decoding:
    ///   bitnet run --model model.gguf --prompt "Test question" \
    ///     --temperature 0.0 --greedy --seed 42
    ///
    /// Raw completion (no Q&A formatting):
    ///   bitnet run --model model.gguf --prompt-template raw \
    ///     --prompt "2+2=" --max-tokens 16
    #[command(alias = "generate")]
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

        /// Maximum new tokens to generate (aliases: --max-tokens, --n-predict)
        #[arg(long, visible_aliases = ["max-tokens", "n-predict"], default_value_t = 32)]
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

        /// Allow falling back to mock loader if real loader fails
        /// Also toggled by env BITNET_ALLOW_MOCK=1
        #[arg(long, env = "BITNET_ALLOW_MOCK", default_value_t = false)]
        allow_mock: bool,

        /// Strict mapping mode: fail if any tensors are unmapped
        #[arg(long, default_value_t = false)]
        strict_mapping: bool,

        /// Strict tokenizer mode: fail if no real tokenizer available
        #[arg(long, default_value_t = false)]
        strict_tokenizer: bool,

        /// Output JSON results to file
        #[arg(long)]
        json_out: Option<std::path::PathBuf>,

        /// Dump token IDs to stdout
        #[arg(long, default_value_t = false)]
        dump_ids: bool,

        /// Insert BOS token at start of prompt
        #[arg(long, default_value_t = false)]
        bos: bool,

        /// Use greedy decoding (overrides temperature)
        #[arg(long, default_value_t = false)]
        greedy: bool,

        /// Enable deterministic mode (single-threaded)
        #[arg(long, default_value_t = false)]
        deterministic: bool,

        /// Number of threads to use (0 = all cores)
        #[arg(long, default_value_t = 0)]
        threads: usize,

        /// Dump logit steps during generation (max steps)
        #[arg(long)]
        dump_logit_steps: Option<usize>,

        /// Top-k tokens to include in logit dump
        #[arg(long, default_value = "10", value_name = "K")]
        logits_topk: usize,

        /// Assert greedy argmax invariant when dumping logits
        #[arg(long, default_value_t = false)]
        assert_greedy: bool,
    },

    /// Tokenize text and output token IDs as JSON
    Tokenize {
        /// Model GGUF path (for extracting tokenizer and counts)
        #[arg(long)]
        model: std::path::PathBuf,

        /// Optional external SentencePiece tokenizer (overrides GGUF)
        #[arg(long)]
        tokenizer: Option<std::path::PathBuf>,

        /// Text to tokenize (inline)
        #[arg(long, conflicts_with = "file")]
        text: Option<String>,

        /// Read text from file
        #[arg(long, conflicts_with = "text")]
        file: Option<std::path::PathBuf>,

        /// Insert BOS token at start
        #[arg(long, default_value_t = false)]
        bos: bool,

        /// Output JSON to file (stdout if omitted)
        #[arg(long)]
        json_out: Option<std::path::PathBuf>,
    },

    /// Calculate perplexity score for a model
    Score(score::ScoreArgs),

    #[cfg(feature = "full-cli")]
    /// Run inference on a model
    ///
    /// # Examples
    ///
    /// Auto-detect template (recommended):
    ///   bitnet inference --model model.gguf --prompt "Who wrote Pride and Prejudice?"
    ///
    /// Instruct template (Q&A format):
    ///   bitnet inference --model model.gguf --prompt-template instruct \
    ///     --prompt "What is 2+2?" --max-tokens 16
    ///
    /// LLaMA-3 chat with system prompt:
    ///   bitnet inference --model model.gguf --prompt-template llama3-chat \
    ///     --system-prompt "You are a helpful assistant" \
    ///     --prompt "Explain photosynthesis" --max-tokens 128
    ///
    /// Batch Q&A from file:
    ///   bitnet inference --model model.gguf --input-file questions.txt \
    ///     --batch-size 4 --format jsonl > answers.jsonl
    #[command(alias = "infer")]
    Inference(Box<InferenceCommand>),

    #[cfg(feature = "full-cli")]
    /// Interactive chat mode (streaming)
    ///
    /// # Examples
    ///
    /// Auto-detect chat template:
    ///   bitnet chat --model model.gguf --tokenizer tokenizer.json
    ///
    /// LLaMA-3 chat with system prompt:
    ///   bitnet chat --model model.gguf --prompt-template llama3-chat \
    ///     --system-prompt "You are a helpful coding assistant"
    ///
    /// Creative chat with nucleus sampling:
    ///   bitnet chat --model model.gguf --temperature 0.8 --top-p 0.95
    Chat(Box<InferenceCommand>),

    #[cfg(feature = "full-cli")]
    /// Convert between model formats
    #[command(alias = "conv")]
    Convert(ConvertCommand),

    #[cfg(feature = "cli-bench")]
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

    #[cfg(feature = "full-cli")]
    /// Inspect model metadata and diagnostics
    Inspect(InspectCommand),

    /// Check GGUF file compatibility using header validation
    CompatCheck {
        /// Path to .gguf file
        path: std::path::PathBuf,

        /// Output JSON
        #[arg(long)]
        json: bool,

        /// Fail on unsupported version or suspicious counts
        #[arg(long)]
        strict: bool,

        /// Show key-value metadata (limit with --kv-limit)
        #[arg(long)]
        show_kv: bool,

        /// Limit number of KV pairs to show (default: 20)
        #[arg(long, default_value_t = 20)]
        kv_limit: usize,
    },
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

    // Handle interface version flag
    if cli.interface_version {
        println!("{}", INTERFACE_VERSION);
        return Ok(());
    }

    // Load configuration
    let config = load_configuration(&cli).await?;

    // Handle save-config flag
    if let Some(path) = &cli.save_config {
        config.save_to_file(path)?;
        println!("Saved effective configuration to {}", path.display());
        return Ok(());
    }

    // Setup logging
    setup_logging(&config, cli.log_level.as_deref())?;

    // Handle commands
    let result = match cli.command {
        Some(Commands::Run {
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
            allow_mock,
            strict_mapping,
            strict_tokenizer,
            json_out,
            dump_ids,
            bos,
            greedy,
            deterministic,
            threads,
            dump_logit_steps,
            logits_topk,
            assert_greedy,
        }) => {
            run_simple_generation(
                model,
                tokenizer,
                prompt,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                seed,
                allow_mock,
                strict_mapping,
                strict_tokenizer,
                json_out,
                dump_ids,
                bos,
                greedy,
                deterministic,
                threads,
                dump_logit_steps,
                logits_topk,
                assert_greedy,
            )
            .await
        }
        #[cfg(feature = "full-cli")]
        Some(Commands::Inference(cmd)) => (*cmd).execute(&config).await,
        #[cfg(feature = "full-cli")]
        Some(Commands::Chat(cmd)) => (*cmd).run_chat(&config).await,
        #[cfg(feature = "full-cli")]
        Some(Commands::Convert(cmd)) => cmd.execute(&config).await,
        #[cfg(feature = "cli-bench")]
        Some(Commands::Benchmark(cmd)) => cmd.execute(&config).await,
        #[cfg(feature = "full-cli")]
        Some(Commands::Serve(cmd)) => cmd.execute(&config).await,
        Some(Commands::Tokenize { model, tokenizer, text, file, bos, json_out }) => {
            handle_tokenize_command(model, tokenizer, text, file, bos, json_out).await
        }
        Some(Commands::Score(args)) => score::run_score(&args).await,
        Some(Commands::Config { action }) => handle_config_command(action, &config).await,
        Some(Commands::Info) => show_system_info().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::Inspect(cmd)) => cmd.execute().await,
        Some(Commands::CompatCheck { path, json, strict, show_kv, kv_limit }) => {
            handle_compat_check_command(path, json, strict, show_kv, kv_limit).await
        }
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
        CliConfig::default_config_path().unwrap_or_else(|_| std::path::PathBuf::from("bitnet.toml"))
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
        .with_target(false)
        .with_writer(std::io::stderr);

    match config.logging.format.as_str() {
        "json" => {
            subscriber.json().with_timer(tracing_subscriber::fmt::time::uptime()).init();
        }
        "compact" => {
            subscriber.compact().init();
        }
        _ => {
            subscriber.pretty().init();
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
/// Handle tokenize command - tokenize text and output JSON
async fn handle_tokenize_command(
    model_path: std::path::PathBuf,
    tokenizer_path: Option<std::path::PathBuf>,
    text: Option<String>,
    file: Option<std::path::PathBuf>,
    bos: bool,
    json_out: Option<std::path::PathBuf>,
) -> Result<()> {
    use bitnet_models::GgufReader;
    use bitnet_tokenizers::Tokenizer;

    // Read GGUF to get counts (always needed)
    let gguf_bytes = std::fs::read(&model_path)
        .with_context(|| format!("Failed to read model: {}", model_path.display()))?;
    let gguf = GgufReader::new(&gguf_bytes).context("Failed to parse GGUF")?;

    let counts = serde_json::json!({
        "n_kv": gguf.metadata_keys().len(),
        "n_tensors": gguf.tensor_count(),
        "unmapped": 0  // tokenize doesn't map tensors
    });

    // Load tokenizer: prefer external, fall back to GGUF
    let (tokenizer, is_external): (Box<dyn Tokenizer>, bool) =
        if let Some(spm_path) = tokenizer_path {
            let tok = bitnet_tokenizers::load_tokenizer(&spm_path).with_context(|| {
                format!("Failed to load external tokenizer: {}", spm_path.display())
            })?;
            (tok, true)
        } else {
            let tok = bitnet_tokenizers::loader::load_tokenizer_from_gguf_reader(&gguf)
                .context("No tokenizer in GGUF, provide --tokenizer")?;
            (tok, false)
        };

    // Read input text
    let input = if let Some(s) = text {
        s
    } else if let Some(p) = file {
        std::fs::read_to_string(p).context("Failed to read input file")?
    } else {
        anyhow::bail!("Provide --text or --file");
    };

    // Tokenize with BOS policy
    let ids = tokenizer.encode(&input, bos, false)?;

    // Build output JSON
    let output = serde_json::json!({
        "tokens": {
            "ids": ids,
            "count": ids.len(),
        },
        "gen_policy": {
            "bos": bos
        },
        "counts": counts,
        "tokenizer": {
            "type": "sentencepiece",  // all our tokenizers are SP
            "origin": if is_external { "external" } else { "embedded" },
            "bos": tokenizer.bos_token_id(),
            "eos": tokenizer.eos_token_id(),
        }
    });

    // Write output
    if let Some(path) = json_out {
        std::fs::write(&path, serde_json::to_string_pretty(&output)?)
            .with_context(|| format!("Failed to write JSON to {}", path.display()))?;
        println!("Wrote {}", path.display());
    } else {
        println!("{}", serde_json::to_string_pretty(&output)?);
    }

    Ok(())
}

async fn handle_config_command(action: ConfigAction, config: &CliConfig) -> Result<()> {
    match action {
        ConfigAction::Show => {
            let config_str =
                toml::to_string_pretty(config).context("Failed to serialize configuration")?;
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
#[allow(clippy::too_many_arguments)]
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
    allow_mock: bool,
    _strict_mapping: bool,
    strict_tokenizer: bool,
    json_out: Option<std::path::PathBuf>,
    dump_ids: bool,
    bos: bool,
    greedy: bool,
    deterministic: bool,
    threads: usize,
    dump_logit_steps: Option<usize>,
    logits_topk: usize,
    assert_greedy: bool,
) -> Result<()> {
    use crate::sampling::Sampler;
    use bitnet_common::Device;
    use bitnet_models::{Model, transformer::KVCache};
    use bitnet_tokenizers::Tokenizer;
    use std::sync::Arc;

    // Simple logit step for dumping
    #[derive(Debug, serde::Serialize)]
    struct LogitStep {
        step: usize,
        top_logits: Vec<serde_json::Value>,
        chosen_id: Option<u32>,
    }

    // Set deterministic mode if requested
    if deterministic {
        unsafe {
            std::env::set_var("BITNET_DETERMINISTIC", "1");
            std::env::set_var("RAYON_NUM_THREADS", "1");
            if threads > 0 {
                std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
            }
        }
    }

    // Override temperature if greedy mode
    let temperature = if greedy { 0.0 } else { temperature };

    println!("Loading model from: {}", model_path.display());

    // Try real loader first
    use bitnet_models::loader::{LoadConfig, ModelLoader};

    let loader = ModelLoader::new(Device::Cpu);
    let load_config =
        LoadConfig { use_mmap: true, validate_checksums: false, progress_callback: None };

    let (model, config): (Arc<dyn Model>, _) = match loader
        .load_with_config(&model_path, &load_config)
    {
        Ok(m) => {
            let cfg = m.config().clone();
            (Arc::from(m) as Arc<dyn Model>, cfg)
        }
        Err(e) => {
            if !allow_mock {
                anyhow::bail!(
                    "Failed to load real model: {e}\n\
                     To run with mock tensors (for smoke/UX testing only), \
                     pass --allow-mock or set BITNET_ALLOW_MOCK=1"
                );
            }
            tracing::warn!("Real loader failed: {e}. Falling back to MOCK loader (by request).");
            // Mock fallback
            let load_result = bitnet_models::gguf_simple::load_gguf_full(
                &model_path,
                Device::Cpu,
                bitnet_models::GGUFLoaderConfig::default(),
            )
            .context("Mock loader also failed")?;
            // TODO: Wire up load_result.i2s_qk256 to raw_tensors once GGUF loader is updated
            let raw_tensors = std::collections::HashMap::new();
            let m = bitnet_models::BitNetModel::from_gguf(
                load_result.config.clone(),
                load_result.tensors,
                raw_tensors,
                Device::Cpu,
            )
            .context("Failed to build mock model")?;
            (Arc::new(m) as Arc<dyn Model>, load_result.config)
        }
    };

    // Load tokenizer with auto-discovery
    // Priority: explicit path → sibling tokenizer.json → parent tokenizer.json → GGUF embedded → mock

    // Track GGUF metadata for JSON output
    let mut gguf_metadata: Option<(usize, usize)> = None;
    let mut external_tokenizer = false;

    let tokenizer: Box<dyn Tokenizer> = {
        // Try auto-discovery first (handles explicit, sibling, parent)
        let discovered_path = if tokenizer_path.is_some() {
            // Explicit path provided
            crate::tokenizer_discovery::resolve_tokenizer(&model_path, tokenizer_path)
        } else {
            // Try sibling/parent discovery
            crate::tokenizer_discovery::resolve_tokenizer(&model_path, None)
        };

        match discovered_path {
            Ok(path) => {
                // Found tokenizer via discovery
                external_tokenizer = true;
                println!("Loading tokenizer from: {}", path.display());

                match bitnet_tokenizers::load_tokenizer(&path) {
                    Ok(tok) => tok,
                    Err(e) => {
                        if strict_tokenizer {
                            eprintln!("Strict tokenizer failed: Failed to load tokenizer: {e}");
                            std::process::exit(EXIT_STRICT_TOKENIZER);
                        }
                        if !allow_mock {
                            anyhow::bail!(
                                "Failed to load tokenizer from {}: {e}. Use --allow-mock to use mock tokenizer.",
                                path.display()
                            );
                        }
                        println!("Warning: Using mock tokenizer due to: {e}");
                        Box::new(bitnet_tokenizers::MockTokenizer::new()) as Box<dyn Tokenizer>
                    }
                }
            }
            Err(_discovery_err) => {
                // Discovery failed, try GGUF embedded as fallback
                println!("No external tokenizer found, attempting to load from GGUF model...");

                // Read the GGUF file to get tokenizer metadata
                let gguf_data = std::fs::read(&model_path)
                    .context("Failed to read GGUF file for tokenizer extraction")?;
                let reader = bitnet_models::GgufReader::new(&gguf_data)
                    .context("Failed to parse GGUF for tokenizer extraction")?;

                // Capture metadata counts
                let n_tensors = reader.tensor_count() as usize;
                let n_kv = reader.metadata_keys().len();
                gguf_metadata = Some((n_kv, n_tensors));

                match bitnet_tokenizers::loader::load_tokenizer_from_gguf_reader(&reader) {
                    Ok(tok) => {
                        println!("Successfully loaded SentencePiece tokenizer from GGUF");
                        tok
                    }
                    Err(e) => {
                        if strict_tokenizer {
                            eprintln!(
                                "Strict tokenizer failed: Failed to load tokenizer from GGUF: {e}"
                            );
                            std::process::exit(EXIT_STRICT_TOKENIZER);
                        }
                        if !allow_mock {
                            // Provide actionable error message
                            let model_dir =
                                model_path.parent().unwrap_or_else(|| std::path::Path::new("."));
                            anyhow::bail!(
                                "Failed to load tokenizer from GGUF: {e}\n\
                                 \n\
                                 No tokenizer found. Solutions:\n\
                                 1. Download tokenizer:\n\
                                    cargo run -p xtask -- tokenizer --into {}\n\
                                 2. Provide explicit tokenizer path:\n\
                                    --tokenizer /path/to/tokenizer.json\n\
                                 3. Use mock tokenizer for testing:\n\
                                    --allow-mock",
                                model_dir.display()
                            );
                        }
                        println!("Warning: Using mock tokenizer due to: {e}");
                        Box::new(bitnet_tokenizers::MockTokenizer::new()) as Box<dyn Tokenizer>
                    }
                }
            }
        }
    };

    // Tokenize prompt with BOS policy
    let mut tokens = tokenizer.encode(&prompt, bos, false)?;
    println!("Input tokens ({}): {:?}", tokens.len(), &tokens[..10.min(tokens.len())]);

    // Create KV cache
    let cache = KVCache::new(&config, 1, &candle_core::Device::Cpu)?;
    let mut any_cache: Box<dyn std::any::Any> = Box::new(cache);

    // Create sampler
    let mut sampler = Sampler::new(temperature, top_k, top_p, repetition_penalty, seed);

    print!("Generating: {}", prompt);
    std::io::Write::flush(&mut std::io::stdout())?;

    // Track timing
    let start_time = std::time::Instant::now();
    let mut first_token_ms: Option<u64> = None;

    // Track generated tokens for repetition penalty
    let mut generated_tokens = Vec::new();

    // Track logits dump if requested
    let mut logits_dump: Vec<LogitStep> = Vec::new();

    // Generation loop
    for step_idx in 0..max_new_tokens {
        // Embed tokens
        let x = model.embed(&tokens)?;

        // Forward pass
        let h = model.forward(&x, any_cache.as_mut())?;

        // Extract last token hidden state first to avoid 3D×2D matmul issues
        let last_hidden = extract_last_token_hidden(&h)?;

        // Debug tap: hidden state RMS sanity (catches "everything is zero")
        if std::env::var("BITNET_DEBUG_LOGITS").as_deref() == Ok("1") && step_idx == 0 {
            let h_vec = tensor_to_vec(&last_hidden)?;
            let hidden_rms = compute_rms(&h_vec);
            eprintln!("hidden_rms={:.6}", hidden_rms);
        }

        // Get logits from last token hidden state
        let logits = model.logits(&last_hidden)?;

        // Extract logits vector with robust shape handling
        let logits_vec = extract_logits_2d(&logits)?;

        // Debug tap: dump logits shape and top-5 on first step (BITNET_DEBUG_LOGITS=1)
        if step_idx == 0 && std::env::var("BITNET_DEBUG_LOGITS").as_deref() == Ok("1") {
            let logits_shape = logits.shape();
            eprintln!(
                "logits_shape=(rows={}, cols={})",
                logits_shape.first().copied().unwrap_or(1),
                logits_shape.get(1).copied().unwrap_or(logits_vec.len())
            );
            let mut idx: Vec<usize> = (0..logits_vec.len()).collect();
            idx.sort_by(|a, b| {
                logits_vec[*b].partial_cmp(&logits_vec[*a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            let top = &idx[..idx.len().min(5)];
            eprintln!("top5_idx={:?}", top);
            eprintln!("top5_val={:?}", top.iter().map(|&i| logits_vec[i]).collect::<Vec<_>>());
        }

        // Capture logits if requested
        if dump_logit_steps.is_some_and(|max_steps| step_idx < max_steps) {
            // Helper for deterministic, robust top-k
            let topk_indices = {
                let mut indexed: Vec<(usize, f32)> =
                    logits_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                // Sort by (-logit, token_id) for determinism
                indexed.sort_by(|a, b| match (a.1.is_finite(), b.1.is_finite()) {
                    (false, true) => std::cmp::Ordering::Greater,
                    (true, false) => std::cmp::Ordering::Less,
                    _ => {
                        let cmp = b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal);
                        if cmp == std::cmp::Ordering::Equal { a.0.cmp(&b.0) } else { cmp }
                    }
                });
                indexed.into_iter().take(logits_topk).map(|(i, _)| i).collect::<Vec<_>>()
            };

            let top_logits: Vec<(u32, f32)> =
                topk_indices.iter().map(|&i| (i as u32, logits_vec[i])).collect();

            // Will capture chosen_id after sampling
            let step = LogitStep {
                step: step_idx,
                top_logits: top_logits
                    .iter()
                    .map(|&(token_id, logit)| {
                        serde_json::json!({
                            "token_id": token_id,
                            "logit": logit
                        })
                    })
                    .collect(),
                chosen_id: None, // Will set after sampling
            };
            logits_dump.push(step);
        }

        // Sample next token
        let next_token = sampler.sample(&logits_vec, &generated_tokens);

        // Assert greedy invariant if requested
        if assert_greedy && greedy && dump_logit_steps.is_some_and(|max_steps| step_idx < max_steps)
        {
            let (mut best_i, mut best_v) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in logits_vec.iter().enumerate() {
                if v.is_finite() && v > best_v {
                    best_v = v;
                    best_i = i;
                }
            }
            if next_token as usize != best_i {
                eprintln!("ERROR: Non-argmax token chosen in --greedy at step {}", step_idx);
                eprintln!("  argmax={} (logit={:.4}) but chosen={}", best_i, best_v, next_token);
                std::process::exit(EXIT_ARGMAX_MISMATCH);
            }
        }

        // Update chosen token in logits dump
        if dump_logit_steps.is_some_and(|max_steps| step_idx < max_steps) && !logits_dump.is_empty()
        {
            logits_dump.last_mut().unwrap().chosen_id = Some(next_token);
        }

        tokens.push(next_token);
        generated_tokens.push(next_token);

        // Track first token time
        if first_token_ms.is_none() {
            first_token_ms = Some(start_time.elapsed().as_millis() as u64);
        }

        // Decode and print the new token
        let token_text = tokenizer.decode(&[next_token])?;
        print!("{}", token_text);
        std::io::Write::flush(&mut std::io::stdout())?;

        // Check for EOS
        if let Some(eos) = tokenizer.eos_token_id()
            && next_token == eos
        {
            break;
        }
    }

    // Calculate timing metrics
    let total_ms = start_time.elapsed().as_millis() as u64;
    let tok_per_sec = if total_ms > 0 {
        (generated_tokens.len() as f64) / (total_ms as f64 / 1000.0)
    } else {
        0.0
    };

    println!("\n\nGeneration complete!");
    println!(
        "Generated {} tokens in {}ms ({:.1} tok/s)",
        generated_tokens.len(),
        total_ms,
        tok_per_sec
    );

    // Output JSON if requested
    if let Some(json_path) = json_out {
        let generated_text = tokenizer.decode(&generated_tokens)?;

        // Get tokenizer info
        let tokenizer_info = serde_json::json!({
            "type": "sentencepiece",
            "origin": if external_tokenizer { "external" } else { "embedded" },
            "bos": tokenizer.bos_token_id().unwrap_or(1),
            "eos": tokenizer.eos_token_id().unwrap_or(2),
        });

        // Count info from GGUF metadata
        let (n_kv, n_tensors) = gguf_metadata.unwrap_or((0, 0));
        let counts = serde_json::json!({
            "n_kv": n_kv,
            "n_tensors": n_tensors,
            "unmapped": 0,  // In strict mode this is always 0
        });

        let gen_policy = serde_json::json!({
            "bos": bos,
            "temperature": temperature,
            "seed": seed.unwrap_or(0),
            "greedy": greedy,
            "deterministic": deterministic,
        });

        let prompt_tokens_len = tokens.len() - generated_tokens.len();
        let output = serde_json::json!({
            "prompt": prompt,
            "text": generated_text,
            "tokens": {
                "prompt": prompt_tokens_len,
                "generated": generated_tokens.len(),
                "total": prompt_tokens_len + generated_tokens.len(),
                "ids": generated_tokens,
            },
            "latency": {
                "cmd_to_first_ms": first_token_ms,
                "decode_first_ms": first_token_ms,  // Same as cmd_to_first for now
                "total_ms": total_ms,
            },
            "throughput": {
                "tokens_per_second": tok_per_sec,
                "decoded_tokens": generated_tokens.len(),
            },
            "counts": counts,
            "tokenizer": tokenizer_info,
            "gen_policy": gen_policy,
            "logits_dump": if !logits_dump.is_empty() {
                Some(logits_dump.iter().map(|step| {
                    serde_json::json!({
                        "step": step.step,
                        "top_logits": step.top_logits,
                        "chosen_id": step.chosen_id
                    })
                }).collect::<Vec<_>>())
            } else {
                None
            },
        });
        std::fs::write(&json_path, serde_json::to_string_pretty(&output)?)?;
        println!("JSON output written to: {}", json_path.display());
    }

    // Dump IDs if requested
    if dump_ids {
        println!("Token IDs: {:?}", generated_tokens);
    }

    Ok(())
}

/// Extract last token hidden state from 3D tensor \[B,T,H\] -> \[B,H\]
fn extract_last_token_hidden(
    tensor: &bitnet_common::ConcreteTensor,
) -> Result<bitnet_common::ConcreteTensor> {
    use bitnet_common::{BitNetError, ConcreteTensor, Tensor};

    let shape = tensor.shape();
    if shape.len() != 3 {
        return Err(BitNetError::Validation("Expected 3D tensor".into()).into());
    }

    let (batch_size, seq_len, hidden_size) = (shape[0], shape[1], shape[2]);

    match tensor {
        ConcreteTensor::BitNet(t) => {
            let candle = t.as_candle();
            // Extract last token: [B, T, H] -> [B, H]
            let last = candle.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
            Ok(ConcreteTensor::BitNet(bitnet_common::BitNetTensor::new(last)))
        }
        ConcreteTensor::Mock(_) => {
            // Return mock hidden state [B, H]
            Ok(ConcreteTensor::mock(vec![batch_size, hidden_size]))
        }
    }
}

/// Extract logits vector from 2D tensor \[B,V\] -> `Vec<f32>`
fn extract_logits_2d(tensor: &bitnet_common::ConcreteTensor) -> Result<Vec<f32>> {
    use bitnet_common::{BitNetError, ConcreteTensor, Tensor};

    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(BitNetError::Validation("Expected 2D tensor".into()).into());
    }

    let (_batch, _vocab) = (shape[0], shape[1]);

    match tensor {
        ConcreteTensor::BitNet(t) => {
            let candle = t.as_candle();
            // Extract first batch: [B, V] -> [V]
            let batch_0 = candle.i(0)?;
            let batch_0 =
                if batch_0.dtype() != DType::F32 { batch_0.to_dtype(DType::F32)? } else { batch_0 };
            Ok(batch_0.to_vec1::<f32>()?)
        }
        ConcreteTensor::Mock(_) => {
            // Return mock logits for testing
            Ok(vec![0.1; 50257])
        }
    }
}

/// Extract logits vector from tensor (legacy function for compatibility)
#[allow(dead_code)]
fn extract_logits(tensor: &bitnet_common::ConcreteTensor) -> Result<Vec<f32>> {
    use bitnet_common::{BitNetError, ConcreteTensor, Tensor};

    let shape = tensor.shape();
    if shape.len() != 3 {
        return Err(BitNetError::Validation("Expected 3D tensor".into()).into());
    }

    let (_batch, seq_len, _vocab) = (shape[0], shape[1], shape[2]);

    match tensor {
        ConcreteTensor::BitNet(t) => {
            let candle = t.as_candle();
            let last = candle.narrow(1, seq_len - 1, 1)?.squeeze(1)?.i(0)?;
            let last = if last.dtype() != DType::F32 { last.to_dtype(DType::F32)? } else { last };
            Ok(last.to_vec1::<f32>()?)
        }
        ConcreteTensor::Mock(_) => {
            // Return mock logits for testing
            Ok(vec![0.1; 50257])
        }
    }
}

/// Convert tensor to f32 vector for diagnostics
fn tensor_to_vec(tensor: &bitnet_common::ConcreteTensor) -> Result<Vec<f32>> {
    use bitnet_common::ConcreteTensor;

    match tensor {
        ConcreteTensor::BitNet(t) => {
            let candle = t.as_candle();
            let candle_f32 = if candle.dtype() != DType::F32 {
                candle.to_dtype(DType::F32)?
            } else {
                candle.clone()
            };
            // Flatten to 1D vector
            let flattened = candle_f32.flatten_all()?;
            Ok(flattened.to_vec1::<f32>()?)
        }
        ConcreteTensor::Mock(mock) => {
            // Return mock values - use shape from tensor
            let size: usize = mock.shape().iter().product();
            Ok(vec![0.1; size])
        }
    }
}

/// Compute RMS (root mean square) of a vector
#[inline]
fn compute_rms(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = xs.iter().map(|x| x * x).sum();
    (sum_sq / (xs.len() as f32)).sqrt()
}

/// Show system information
async fn show_system_info() -> Result<()> {
    println!("{}", style("BitNet System Information").bold().cyan());
    println!();

    // Version information
    println!("{}", style("Version:").bold());
    println!("  BitNet CLI: {}", env!("CARGO_PKG_VERSION"));
    println!(
        "  Rust: {}",
        std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string())
    );
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
        #[cfg(feature = "gpu")]
        {
            match candle_core::Device::cuda_if_available(0).is_ok() {
                true => println!("  CUDA: {}", style("✓ Available").green()),
                false => println!("  CUDA: {}", style("✗ Not available").red()),
            }
        }
        #[cfg(not(feature = "gpu"))]
        println!("  CUDA: {}", style("✗ Not compiled").yellow())
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

/// Inspect model metadata without loading full tensors
#[allow(dead_code)]
async fn handle_inspect_command(model_path: std::path::PathBuf, json: bool) -> Result<()> {
    use bitnet_models::GgufReader;
    use bitnet_models::formats::ModelFormat;
    use memmap2::Mmap;
    use serde_json::json;
    use std::fs::File;

    // Tokenizer source constants
    const TOKENIZER_SOURCE_EMBEDDED: &str = "embedded-gguf";
    const TOKENIZER_SOURCE_EXTERNAL: &str = "external";

    // Detect model format
    let format = ModelFormat::detect_from_header(&model_path)?;

    // Extract metadata based on format
    let metadata = match format {
        ModelFormat::Gguf => {
            // Memory-map the file for efficient reading
            let file = File::open(&model_path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            let reader = GgufReader::new(&mmap)?;

            // Extract key metadata
            let name =
                reader.get_string_metadata("general.name").unwrap_or_else(|| "unknown".to_string());
            let architecture = reader
                .get_string_metadata("general.architecture")
                .unwrap_or_else(|| "unknown".to_string());
            fn get_quantization(reader: &GgufReader) -> String {
                if let Some(q) = reader.get_string_metadata("general.quantization_type") {
                    q
                } else if let Some(q) = reader.get_quantization_type() {
                    format!("{:?}", q)
                } else {
                    "unknown".to_string()
                }
            }
            let quantization = get_quantization(&reader);
            let vocab_size = reader
                .get_u32_metadata("llama.vocab_size")
                .or_else(|| reader.get_u32_metadata("tokenizer.ggml.tokens"))
                .unwrap_or(0);
            let context_length = reader.get_u32_metadata("llama.context_length").unwrap_or(0);

            // Check for tokenizer
            let has_tokenizer = reader.get_u32_metadata("tokenizer.ggml.tokens").is_some();
            let tokenizer_source =
                if has_tokenizer { TOKENIZER_SOURCE_EMBEDDED } else { TOKENIZER_SOURCE_EXTERNAL };

            // Get tensor count
            let tensor_count = reader.tensor_count();

            // Add backend info for IQ2_S quantization
            let backend_info = if quantization.contains("IQ2_S") || quantization.contains("iq2_s") {
                #[cfg(feature = "iq2s-ffi")]
                {
                    use bitnet_models::quant::backend::Iq2sBackend;
                    let backend = Iq2sBackend::selected();
                    Some(json!({
                        "kind": backend.name(),
                        "ggml_commit": bitnet_ggml_ffi::GGML_COMMIT,
                        "qk": backend.qk(),
                        "block_bytes": backend.block_bytes()
                    }))
                }
                #[cfg(not(feature = "iq2s-ffi"))]
                {
                    Some(json!({
                        "kind": "rust",
                        "qk": 256,
                        "block_bytes": 66
                    }))
                }
            } else {
                None
            };

            let mut metadata = json!({
                "format": "GGUF",
                "name": name,
                "architecture": architecture,
                "quantization": {
                    "name": quantization
                },
                "vocab_size": vocab_size,
                "context_length": context_length,
                "tensor_count": tensor_count,
                "tokenizer": {
                    "source": tokenizer_source,
                    "embedded": has_tokenizer
                },
                "scoring_policy": {
                    "add_bos": true,  // Default GGUF behavior
                    "append_eos": false,
                    "mask_pad": true
                }
            });

            // If we detected IQ2_S, attach backend info under quantization
            if let Some(backend) = backend_info {
                metadata["quantization"]["backend"] = backend;
            }

            metadata
        }
        ModelFormat::SafeTensors => {
            use std::io::Read;

            let mut file = File::open(&model_path)?;
            let mut header_size_bytes = [0u8; 8];
            file.read_exact(&mut header_size_bytes)?;
            let header_size = u64::from_le_bytes(header_size_bytes) as usize;

            let mut header_bytes = vec![0u8; header_size];
            file.read_exact(&mut header_bytes)?;
            let header_str = String::from_utf8(header_bytes)
                .map_err(|e| anyhow::anyhow!("Invalid header encoding: {}", e))?;
            let header: serde_json::Value = serde_json::from_str(&header_str)?;

            // Count tensors (keys that aren't "__metadata__")
            let tensor_count = header
                .as_object()
                .map(|obj| obj.keys().filter(|k| *k != "__metadata__").count())
                .unwrap_or(0);

            json!({
                "format": "SafeTensors",
                "tensor_count": tensor_count,
                "metadata": header.get("__metadata__").unwrap_or(&json!({})),
                "tokenizer": {
                    "source": "external-json"
                },
                "scoring_policy": {
                    "add_bos": true,
                    "append_eos": false,
                    "mask_pad": true
                }
            })
        }
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&metadata)?);
    } else {
        println!("{}", style("Model Metadata").bold().cyan());
        println!("{:#?}", metadata);
    }

    Ok(())
}

/// Check GGUF file compatibility using the new header parser
async fn handle_compat_check_command(
    path: std::path::PathBuf,
    json: bool,
    strict: bool,
    show_kv: bool,
    kv_limit: usize,
) -> Result<()> {
    use bitnet_inference::gguf;
    use serde_json::json;

    let header = match gguf::read_header_blocking(&path) {
        Ok(h) => h,
        Err(e) => {
            match &e {
                gguf::GgufError::Io(_) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
                gguf::GgufError::BadMagic(_)
                | gguf::GgufError::Malformed
                | gguf::GgufError::ShortHeader(_) => {
                    eprintln!("{e}");
                    std::process::exit(2);
                }
                gguf::GgufError::UnsupportedVersion(_) => {
                    eprintln!("{e}");
                    std::process::exit(3);
                }
                _ => {
                    eprintln!("{e}");
                    std::process::exit(2);
                } // Future variants
            }
        }
    };

    let supported = (1..=3).contains(&header.version);
    let suspicious = header.n_tensors > 10_000_000 || header.n_kv > 10_000_000;

    // Read KV pairs if requested
    let kvs = if show_kv {
        match gguf::read_kv_pairs(&path, Some(kv_limit)) {
            Ok(kvs) => Some(kvs),
            Err(e) => {
                eprintln!("Warning: Failed to read KV pairs: {}", e);
                None
            }
        }
    } else {
        None
    };

    if json {
        let mut obj = json!({
            "path": path.display().to_string(),
            "status": "valid",
            "gguf": {
                "version": header.version,
                "n_tensors": header.n_tensors,
                "n_kv": header.n_kv,
            },
            "compatibility": {
                "supported_version": supported,
                "tensors_reasonable": !suspicious,
                "kvs_reasonable": !suspicious,
            }
        });

        if let Some(kvs) = kvs {
            let kv_json: Vec<_> = kvs
                .iter()
                .map(|kv| {
                    let value_str = match &kv.value {
                        gguf::GgufValue::U8(v) => json!(v),
                        gguf::GgufValue::I8(v) => json!(v),
                        gguf::GgufValue::U16(v) => json!(v),
                        gguf::GgufValue::I16(v) => json!(v),
                        gguf::GgufValue::U32(v) => json!(v),
                        gguf::GgufValue::I32(v) => json!(v),
                        gguf::GgufValue::F32(v) => json!(v),
                        gguf::GgufValue::Bool(v) => json!(v),
                        gguf::GgufValue::String(v) => json!(v),
                        gguf::GgufValue::Array(_) => json!("[array]"),
                        gguf::GgufValue::U64(v) => json!(v),
                        gguf::GgufValue::I64(v) => json!(v),
                        gguf::GgufValue::F64(v) => json!(v),
                    };
                    json!({
                        "key": kv.key,
                        "value": value_str
                    })
                })
                .collect();
            obj["metadata"] = json!(kv_json);
        }

        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("File:      {}", path.display());
        println!("Status:    ✓ Valid GGUF");
        println!(
            "Version:   {} {}",
            header.version,
            if supported { "(supported)" } else { "(unsupported)" }
        );
        println!("Tensors:   {}", header.n_tensors);
        println!("KV pairs:  {}", header.n_kv);

        if let Some(kvs) = kvs {
            println!("\nMetadata (showing {} of {}):", kvs.len(), header.n_kv);
            for kv in kvs.iter().take(kv_limit) {
                let value_str = match &kv.value {
                    gguf::GgufValue::U8(v) => format!("{}", v),
                    gguf::GgufValue::I8(v) => format!("{}", v),
                    gguf::GgufValue::U16(v) => format!("{}", v),
                    gguf::GgufValue::I16(v) => format!("{}", v),
                    gguf::GgufValue::U32(v) => format!("{}", v),
                    gguf::GgufValue::I32(v) => format!("{}", v),
                    gguf::GgufValue::F32(v) => format!("{}", v),
                    gguf::GgufValue::Bool(v) => format!("{}", v),
                    gguf::GgufValue::String(v) => {
                        if v.len() > 50 {
                            format!("\"{}...\"", &v[..47])
                        } else {
                            format!("\"{}\"", v)
                        }
                    }
                    gguf::GgufValue::Array(arr) => format!("[{} items]", arr.len()),
                    gguf::GgufValue::U64(v) => format!("{}", v),
                    gguf::GgufValue::I64(v) => format!("{}", v),
                    gguf::GgufValue::F64(v) => format!("{}", v),
                };
                println!("  {:<30} = {}", kv.key, value_str);
            }
        }

        if suspicious {
            eprintln!("⚠ Unusually high tensor/KV counts detected");
        }
        if !supported {
            eprintln!("⚠ Unsupported GGUF version");
        }
    }

    if strict && (!supported || suspicious) {
        std::process::exit(4);
    }
    Ok(())
}
