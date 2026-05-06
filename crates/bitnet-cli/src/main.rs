//! BitNet CLI application
//!
//! A comprehensive command-line interface for BitNet 1-bit LLM inference.
//! Supports model loading, inference, conversion, benchmarking, and serving.

// COMPILE-TIME FIREWALL: Prevent mock feature in production CLI
#[cfg(feature = "mock")]
compile_error!("The 'mock' feature must never be enabled for the CLI ΓÇô tests only.");

use anyhow::{Context, Result};
use bitnet_common::Tensor;
use bitnet_startup_contract_guard::{
    ContractPolicy, RuntimeComponent, evaluate_and_emit, feature_line,
};
use candle_core::{DType, IndexOp};
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{Shell, generate};
use console::style;
use std::io;
use tracing::{debug, error, info, warn};

#[cfg(feature = "full-cli")]
mod commands;
mod config;
mod exit;
#[cfg(feature = "full-cli")]
mod ln_rules;
mod score;
pub mod tokenizer_discovery;

use exit::*;

/// Build the CLI command for external use (e.g., in tests)
pub fn build_cli() -> clap::Command {
    Cli::command()
}

/// CLI interface version (SemVer for CLI surface compatibility)
const INTERFACE_VERSION: &str = "1.0.0";

fn bitnet_version() -> &'static str {
    use std::sync::OnceLock;
    static VERSION_STRING: OnceLock<String> = OnceLock::new();

    VERSION_STRING.get_or_init(|| {
        let features_line = feature_line();

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
#[command(about = "BitNet-rs ΓÇö 1-bit neural network inference with strict receipts")]
#[command(long_about = r#"BitNet-rs CLI ΓÇö one-shot generation and chat with strict receipts

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

  QK256 Models (I2_S quantization):
    - Without AVX2: ~0.1 tok/s (scalar kernels, ~10s per token)
    - With AVX2: ~1.2├ù faster (optimized kernels)
    - For quick validation: use --max-tokens 4-16
    - SIMD optimizations (ΓëÑ3├ù faster) coming in v0.2.0
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

    /// Device/backend to use (cpu, cuda, nvidia-rtx-5070-ti-cuda, nvidia-rtx-5070-ti-wgpu, oneapi, gpu, metal, mpsgraph, apple-m4-metal, apple-m4-mpsgraph, apple-m4-cpu-neon, npu, auto)
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
        /// Model file or directory path (.gguf file or HuggingFace model directory)
        #[arg(short, long)]
        model: std::path::PathBuf,

        /// Model format: auto (detect from path), gguf, safetensors
        #[arg(long, value_name = "FORMAT", default_value = "auto")]
        model_format: String,

        /// Model architecture override (e.g. bitnet, llama, phi3); auto-detected if omitted
        #[arg(long, value_name = "ARCH")]
        architecture: Option<String>,

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

        /// Strict loader mode: fail-fast with enhanced loader (sets BITNET_DISABLE_MINIMAL_LOADER=1, BITNET_STRICT_MODE=1)
        #[arg(long, default_value_t = false)]
        strict_loader: bool,

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

        /// Prompt template: auto (detect), raw (no formatting), instruct (Q&A format), llama3-chat (LLaMA-3 format)
        #[arg(long, value_name = "TEMPLATE", default_value = "auto")]
        prompt_template: String,

        /// System prompt for chat models
        #[arg(long, value_name = "TEXT")]
        system_prompt: Option<String>,

        /// Stop sequences (can be repeated for multiple sequences)
        #[arg(long = "stop", value_name = "SEQ")]
        stop: Vec<String>,

        /// Stop token IDs (numeric token IDs, can be repeated)
        #[arg(long = "stop-id", value_name = "ID")]
        stop_id: Vec<u32>,

        /// Dump logit steps during generation (max steps)
        #[arg(long)]
        dump_logit_steps: Option<usize>,

        /// Top-k tokens to include in logit dump
        #[arg(long, default_value = "10", value_name = "K")]
        logits_topk: usize,

        /// Assert greedy argmax invariant when dumping logits
        #[arg(long, default_value_t = false)]
        assert_greedy: bool,

        /// Suppress performance warnings
        #[arg(long, default_value_t = false)]
        no_warnings: bool,
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

    /// Probe selected device identity without launching kernels
    DeviceSmoke {
        /// Output JSON probe receipt to file
        #[arg(long)]
        json_out: Option<std::path::PathBuf>,
    },

    /// Compile and launch a tiny CUDA vector-add kernel
    CudaSmoke {
        /// CUDA device index to probe and launch on
        #[arg(long, default_value_t = 0)]
        device_index: usize,

        /// Output JSON smoke receipt to file
        #[arg(long)]
        json_out: Option<std::path::PathBuf>,
    },

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

    /// List all supported model architectures
    ListArchitectures {
        /// Output in JSON format
        #[arg(long)]
        json: bool,
    },

    /// List all available prompt templates
    ListTemplates {
        /// Output in JSON format
        #[arg(long)]
        json: bool,
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
    // RUNTIME GUARD: Forbid test shims in production
    if std::env::var_os("BITNET_GPU_FAKE").is_some() && std::env::var_os("CI").is_none() {
        eprintln!("Error: BITNET_GPU_FAKE is test-only and not allowed outside CI.");
        std::process::exit(8);
    }

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

    let startup_contract_report =
        evaluate_and_emit(RuntimeComponent::Cli, ContractPolicy::Observe)?;
    if !startup_contract_report.is_compatible() {
        warn!(component = ?RuntimeComponent::Cli, "CLI startup contract reported issues");
    }

    let requested_backend_label =
        cli.device.clone().unwrap_or_else(|| config.default_device.clone());

    // Report backend selection at startup so logs and receipts are deterministic.
    {
        use bitnet_common::{BackendRequest, select_backend};
        use bitnet_kernels::device_features::current_kernel_capabilities;

        let caps = current_kernel_capabilities();
        let request =
            BackendRequest::from_label(&requested_backend_label).unwrap_or(BackendRequest::Auto);
        let strict_mode = std::env::var("BITNET_STRICT_MODE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        match select_backend(request, &caps) {
            Ok(result) => info!(backend_selection = %result.identity_summary(), "backend selected"),
            Err(e) if strict_mode => return Err(e.into()),
            Err(e) => warn!(error = %e, "backend selection warning"),
        }
    }

    let result = match cli.command {
        Some(Commands::Run {
            model,
            model_format,
            architecture,
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
            strict_loader,
            json_out,
            dump_ids,
            bos,
            greedy,
            deterministic,
            threads,
            prompt_template,
            system_prompt,
            stop,
            stop_id,
            dump_logit_steps,
            logits_topk,
            assert_greedy,
            no_warnings,
        }) => {
            run_simple_generation(
                &requested_backend_label,
                model,
                model_format,
                architecture,
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
                strict_loader,
                json_out,
                dump_ids,
                bos,
                greedy,
                deterministic,
                threads,
                prompt_template,
                system_prompt,
                stop,
                stop_id,
                dump_logit_steps,
                logits_topk,
                assert_greedy,
                no_warnings,
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
        Some(Commands::DeviceSmoke { json_out }) => {
            handle_device_smoke_command(&requested_backend_label, json_out).await
        }
        Some(Commands::CudaSmoke { device_index, json_out }) => {
            handle_cuda_smoke_command(&requested_backend_label, device_index, json_out).await
        }
        #[cfg(feature = "full-cli")]
        Some(Commands::Inspect(cmd)) => cmd.execute().await,
        Some(Commands::CompatCheck { path, json, strict, show_kv, kv_limit }) => {
            handle_compat_check_command(path, json, strict, show_kv, kv_limit).await
        }
        Some(Commands::ListArchitectures { json }) => {
            use bitnet_common::ArchitectureRegistry;

            if json {
                let archs: Vec<_> = ArchitectureRegistry::known_architectures()
                    .iter()
                    .filter_map(|arch| {
                        ArchitectureRegistry::lookup(arch).map(|defaults| {
                            serde_json::json!({
                                "architecture": arch,
                                "norm_type": format!("{:?}", defaults.norm_type),
                                "activation_type": format!("{:?}", defaults.activation_type),
                                "default_context_length": defaults.default_context_length,
                            })
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&archs).unwrap());
            } else {
                println!("{:<30} {:<12} {:<12} Context", "Architecture", "Norm", "Activation");
                println!("{}", "-".repeat(70));
                for arch in ArchitectureRegistry::known_architectures() {
                    if let Some(defaults) = ArchitectureRegistry::lookup(arch) {
                        println!(
                            "{:<30} {:<12} {:<12} {}",
                            arch,
                            format!("{:?}", defaults.norm_type),
                            format!("{:?}", defaults.activation_type),
                            defaults
                                .default_context_length
                                .map_or("default".to_string(), |v| v.to_string()),
                        );
                    }
                }
            }
            Ok(())
        }
        Some(Commands::ListTemplates { json }) => {
            use bitnet_prompt_templates::TemplateType;

            if json {
                let templates: Vec<_> = TemplateType::all_variants()
                    .iter()
                    .map(|t| {
                        let info = t.info();
                        serde_json::json!({
                            "name": info.name,
                            "stop_sequences": info.stop_sequences,
                            "adds_bos": info.adds_bos,
                            "parses_special": info.parses_special,
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&templates).unwrap());
            } else {
                println!("{:<30} {:<6} {:<8} Stop Sequences", "Template", "BOS", "Special");
                println!("{}", "-".repeat(80));
                for t in TemplateType::all_variants() {
                    let info = t.info();
                    let stops = if info.stop_sequences.is_empty() {
                        "(none)".to_string()
                    } else {
                        info.stop_sequences.join(", ")
                    };
                    println!(
                        "{:<30} {:<6} {:<8} {}",
                        info.name,
                        if info.adds_bos { "yes" } else { "no" },
                        if info.parses_special { "yes" } else { "no" },
                        stops,
                    );
                }
            }
            Ok(())
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
    let (tokenizer, is_external): (std::sync::Arc<dyn Tokenizer + Send + Sync>, bool) =
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

async fn handle_device_smoke_command(
    requested_backend_label: &str,
    json_out: Option<std::path::PathBuf>,
) -> Result<()> {
    use bitnet_common::BackendRequest;

    const MACHINE_ID: &str = "windows-9950x3d-rtx5070ti";
    const RTX_5070_TI_CUDA: &str = "nvidia-rtx-5070-ti-cuda";
    const REFERENCE_BACKEND: &str = "amd-9950x3d-cpu-avx512";

    let request = BackendRequest::from_label(requested_backend_label)
        .with_context(|| format!("unsupported device-smoke backend: {requested_backend_label}"))?;
    let requested_backend = request.to_string();

    if !matches!(request, BackendRequest::Cuda | BackendRequest::NvidiaRtx5070TiCuda) {
        anyhow::bail!(
            "device-smoke currently supports cuda and nvidia-rtx-5070-ti-cuda only, got {requested_backend}"
        );
    }

    let mut cuda_probe = bitnet_device_probe::probe_nvidia_cuda(Some(0));
    let identity_error =
        if matches!(request, BackendRequest::NvidiaRtx5070TiCuda) && cuda_probe.available {
            validate_rtx_5070_ti_identity(&cuda_probe)
        } else {
            None
        };

    if let Some(error) = &identity_error {
        cuda_probe.available = false;
        cuda_probe.failure_reason = Some(error.clone());
    }

    let error = if !cuda_probe.available {
        cuda_probe
            .failure_reason
            .clone()
            .or_else(|| Some("requested CUDA probe device is unavailable".to_string()))
    } else {
        None
    };
    let selected_backend = if error.is_none() {
        Some(if matches!(request, BackendRequest::NvidiaRtx5070TiCuda) {
            RTX_5070_TI_CUDA
        } else {
            "cuda"
        })
    } else {
        None
    };

    let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let artifact_path = json_out.as_ref().map(|path| path.display().to_string());
    let receipt = serde_json::json!({
        "schema": 1,
        "artifact_kind": "cuda_probe",
        "machine_id": MACHINE_ID,
        "hardware_lane": RTX_5070_TI_CUDA,
        "timestamp_utc": timestamp_utc,
        "requested_backend": requested_backend,
        "selected_backend": selected_backend,
        "runtime_api": "cuda",
        "reference_backend": REFERENCE_BACKEND,
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "cuda": cuda_probe,
        "claim": "cuda_runtime_probe_recorded",
        "kernel_execution": false,
        "artifact_path": artifact_path,
        "error": error,
    });

    write_json_output(json_out.as_ref(), &receipt)?;

    if let Some(error) = receipt.get("error").and_then(serde_json::Value::as_str) {
        anyhow::bail!("{error}");
    }

    Ok(())
}

struct CudaSmokeReceiptFields {
    result: &'static str,
    input_len: serde_json::Value,
    max_abs_error: serde_json::Value,
    mean_abs_error: serde_json::Value,
    host_to_device_bytes: serde_json::Value,
    device_to_host_bytes: serde_json::Value,
    invocations: u64,
    fallback_invocations: u64,
    kernel_launches: u64,
}

impl Default for CudaSmokeReceiptFields {
    fn default() -> Self {
        Self {
            result: "fail",
            input_len: serde_json::Value::Null,
            max_abs_error: serde_json::Value::Null,
            mean_abs_error: serde_json::Value::Null,
            host_to_device_bytes: serde_json::Value::Null,
            device_to_host_bytes: serde_json::Value::Null,
            invocations: 0,
            fallback_invocations: 0,
            kernel_launches: 0,
        }
    }
}

fn run_cuda_smoke_kernel_receipt_fields(
    cuda_probe: &mut bitnet_device_probe::NvidiaCudaProbe,
    device_index: usize,
    error: &mut Option<String>,
) -> CudaSmokeReceiptFields {
    #[cfg(feature = "cuda")]
    {
        match bitnet_kernels::gpu::run_cuda_tiny_vector_add_smoke(device_index) {
            Ok(smoke) => {
                cuda_probe.selected_device_index = Some(smoke.device_info.device_id);
                cuda_probe.selected_device_name = Some(smoke.device_info.name);
                cuda_probe.compute_capability = Some(format!(
                    "{}.{}",
                    smoke.device_info.compute_capability.0, smoke.device_info.compute_capability.1
                ));
                cuda_probe.vram_bytes = Some(smoke.device_info.total_memory as u64);

                if !smoke.passed {
                    *error = Some(format!(
                        "tiny CUDA vector add mismatch: max_abs_error={}, mean_abs_error={}",
                        smoke.max_abs_error, smoke.mean_abs_error
                    ));
                }

                CudaSmokeReceiptFields {
                    result: if smoke.passed { "pass" } else { "fail" },
                    input_len: serde_json::json!(smoke.input_len),
                    max_abs_error: serde_json::json!(smoke.max_abs_error),
                    mean_abs_error: serde_json::json!(smoke.mean_abs_error),
                    host_to_device_bytes: serde_json::json!(
                        smoke.kernel_stats.host_to_device_bytes
                    ),
                    device_to_host_bytes: serde_json::json!(
                        smoke.kernel_stats.device_to_host_bytes
                    ),
                    invocations: smoke.kernel_stats.invocations,
                    fallback_invocations: smoke.kernel_stats.fallback_invocations,
                    kernel_launches: smoke.kernel_stats.kernel_launches,
                }
            }
            Err(err) => {
                *error = Some(format!("tiny CUDA vector add smoke failed: {err}"));
                CudaSmokeReceiptFields::default()
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = cuda_probe;
        let _ = device_index;
        *error = Some("compiled without the cuda feature".to_string());
        CudaSmokeReceiptFields::default()
    }
}

async fn handle_cuda_smoke_command(
    requested_backend_label: &str,
    device_index: usize,
    json_out: Option<std::path::PathBuf>,
) -> Result<()> {
    use bitnet_common::BackendRequest;

    const MACHINE_ID: &str = "windows-9950x3d-rtx5070ti";
    const RTX_5070_TI_CUDA: &str = "nvidia-rtx-5070-ti-cuda";
    const REFERENCE_BACKEND: &str = "amd-9950x3d-cpu-avx512";
    const KERNEL_ID: &str = "cuda_tiny_vector_add";

    let request = BackendRequest::from_label(requested_backend_label)
        .with_context(|| format!("unsupported cuda-smoke backend: {requested_backend_label}"))?;
    let requested_backend = request.to_string();

    if !matches!(request, BackendRequest::NvidiaRtx5070TiCuda) {
        anyhow::bail!(
            "cuda-smoke currently supports nvidia-rtx-5070-ti-cuda only, got {requested_backend}"
        );
    }

    let mut cuda_probe = bitnet_device_probe::probe_nvidia_cuda(Some(device_index));
    let identity_error =
        if matches!(request, BackendRequest::NvidiaRtx5070TiCuda) && cuda_probe.available {
            validate_rtx_5070_ti_identity(&cuda_probe)
        } else {
            None
        };

    if let Some(error) = &identity_error {
        cuda_probe.available = false;
        cuda_probe.failure_reason = Some(error.clone());
    }

    let mut error = if !cuda_probe.available {
        cuda_probe
            .failure_reason
            .clone()
            .or_else(|| Some("requested CUDA smoke device is unavailable".to_string()))
    } else {
        None
    };

    let selected_backend = if cuda_probe.available {
        Some(if matches!(request, BackendRequest::NvidiaRtx5070TiCuda) {
            RTX_5070_TI_CUDA
        } else {
            "cuda"
        })
    } else {
        None
    };

    let outcome = if error.is_none() {
        run_cuda_smoke_kernel_receipt_fields(&mut cuda_probe, device_index, &mut error)
    } else {
        CudaSmokeReceiptFields::default()
    };

    let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let artifact_path = json_out.as_ref().map(|path| path.display().to_string());
    let claim = if error.is_none() && outcome.result == "pass" {
        "kernel_smoke_tested"
    } else {
        "cuda_kernel_smoke_attempted"
    };
    let receipt = serde_json::json!({
        "schema": 1,
        "artifact_kind": "cuda_smoke",
        "machine_id": MACHINE_ID,
        "hardware_lane": RTX_5070_TI_CUDA,
        "timestamp_utc": timestamp_utc,
        "requested_backend": requested_backend,
        "selected_backend": selected_backend,
        "runtime_api": "cuda",
        "reference_backend": REFERENCE_BACKEND,
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "cuda": {
            "available": cuda_probe.available,
            "device_count": cuda_probe.device_count,
            "device_index": cuda_probe.selected_device_index,
            "device_name": cuda_probe.selected_device_name,
            "compute_capability": cuda_probe.compute_capability,
            "driver_version": cuda_probe.driver_version,
            "cuda_runtime_version": cuda_probe.cuda_runtime_version,
            "cuda_toolkit_version": cuda_probe.cuda_toolkit_version,
            "nvrtc_version": cuda_probe.nvrtc_version,
            "nvml_available": cuda_probe.nvml_available,
            "vram_bytes": cuda_probe.vram_bytes,
            "power_limit_watts": cuda_probe.power_limit_watts,
            "power_draw_watts": cuda_probe.power_draw_watts,
            "temperature_c": cuda_probe.temperature_c,
        },
        "kernel_stats": [
            {
                "kernel_id": KERNEL_ID,
                "invocations": outcome.invocations,
                "fallback_invocations": outcome.fallback_invocations,
                "host_to_device_bytes": outcome.host_to_device_bytes,
                "device_to_host_bytes": outcome.device_to_host_bytes,
                "kernel_launches": outcome.kernel_launches,
                "kernel_time_ms": null
            }
        ],
        "input_len": outcome.input_len,
        "max_abs_error": outcome.max_abs_error,
        "mean_abs_error": outcome.mean_abs_error,
        "result": outcome.result,
        "claim": claim,
        "artifact_path": artifact_path,
        "error": error,
    });

    write_json_output(json_out.as_ref(), &receipt)?;

    if let Some(error) = receipt.get("error").and_then(serde_json::Value::as_str) {
        anyhow::bail!("{error}");
    }

    Ok(())
}

fn validate_rtx_5070_ti_identity(probe: &bitnet_device_probe::NvidiaCudaProbe) -> Option<String> {
    match probe.selected_device_name.as_deref() {
        Some(name) if is_rtx_5070_ti_device_name(name) => None,
        Some(name) => {
            Some(format!("requested nvidia-rtx-5070-ti-cuda but selected CUDA device is {name:?}"))
        }
        None => Some(
            "requested nvidia-rtx-5070-ti-cuda but selected CUDA device name was not reported"
                .to_string(),
        ),
    }
}

fn is_rtx_5070_ti_device_name(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase();
    normalized.contains("rtx 5070 ti")
}

fn write_json_output(path: Option<&std::path::PathBuf>, value: &serde_json::Value) -> Result<()> {
    let json = serde_json::to_string_pretty(value)?;
    if let Some(path) = path {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create {}", parent.display()))?;
        }
        std::fs::write(path, json)
            .with_context(|| format!("Failed to write JSON to {}", path.display()))?;
        println!("Wrote {}", path.display());
    } else {
        println!("{json}");
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

/// Check if AVX2 is available at runtime
#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn has_avx2() -> bool {
    false
}

/// Check for QK256 quantization and emit performance warnings if using scalar kernels
fn check_and_warn_qk256_performance(model_path: &std::path::Path, max_tokens: usize) -> Result<()> {
    use bitnet_models::GgufReader;

    // Read GGUF file to check for I2_S quantization
    let gguf_data = std::fs::read(model_path)
        .with_context(|| format!("Failed to read model file: {}", model_path.display()))?;

    let reader =
        GgufReader::new(&gguf_data).context("Failed to parse GGUF file for quantization check")?;

    // Check if the model uses I2_S quantization (which could be QK256)
    let has_i2s = reader.tensor_names().iter().any(|name| {
        if let Some(info) = reader.get_tensor_info_by_name(name) {
            matches!(info.tensor_type, bitnet_models::formats::gguf::GgufTensorType::I2_S)
        } else {
            false
        }
    });

    if !has_i2s {
        // No I2_S quantization, no warning needed
        return Ok(());
    }

    // Count I2_S tensors to check if it's a significant portion of the model
    let i2s_count = reader
        .tensor_names()
        .iter()
        .filter(|name| {
            if let Some(info) = reader.get_tensor_info_by_name(name) {
                matches!(info.tensor_type, bitnet_models::formats::gguf::GgufTensorType::I2_S)
            } else {
                false
            }
        })
        .count();

    // Only warn if we have a significant number of I2_S tensors (likely QK256)
    if i2s_count < 5 {
        return Ok(());
    }

    // Check if AVX2 is available
    let avx2_available = has_avx2();

    // If AVX2 is available, QK256 will use optimized kernels, no warning needed
    // (This is conservative - the actual dispatch depends on runtime detection in the kernel)
    if avx2_available {
        // Still show a minimal note about QK256 usage
        eprintln!("{} Using QK256 quantization with AVX2 acceleration", style("Γä╣").cyan().bold());
        return Ok(());
    }

    // Show performance warning for scalar kernels
    eprintln!();
    eprintln!("{}", style("ΓÜá  WARNING: Using QK256 scalar kernels (~0.1 tok/s)").yellow().bold());
    eprintln!();
    eprintln!("For quick validation, use --max-tokens 4-16");
    eprintln!("Performance: ~10 seconds per token (2B models)");
    eprintln!();

    // Estimate time for requested token count
    let estimated_seconds = max_tokens * 10; // ~10 seconds per token
    if estimated_seconds > 60 {
        let minutes = estimated_seconds / 60;
        eprintln!("Estimated time for {} tokens: ~{} minutes", max_tokens, minutes);
    } else {
        eprintln!("Estimated time for {} tokens: ~{} seconds", max_tokens, estimated_seconds);
    }
    eprintln!();
    eprintln!("SIMD optimizations coming in v0.2.0 (ΓëÑ3├ù faster)");
    eprintln!();
    eprintln!("Use --no-warnings to suppress this message");
    eprintln!();

    Ok(())
}

fn detect_loader_mode_for_path(path: &std::path::Path, is_hf_directory: bool) -> &'static str {
    if is_hf_directory {
        return "huggingface";
    }

    match path.extension().and_then(|ext| ext.to_str()).map(str::to_ascii_lowercase) {
        Some(ext) if ext == "gguf" => bitnet_models::GgufLoaderMode::RealGguf.as_str(),
        Some(ext) if ext == "safetensors" => "safetensors",
        _ => "unknown",
    }
}

#[derive(Debug, Clone)]
struct RunBackendIdentity {
    requested_backend: String,
    selected_backend: String,
    runtime_api: String,
    fallback_used: bool,
    fallback_reason: Option<String>,
}

fn resolve_run_backend_identity(
    requested_backend_label: &str,
    strict_backend: bool,
) -> Result<RunBackendIdentity> {
    use bitnet_common::{BackendRequest, select_backend};
    use bitnet_kernels::device_features::current_kernel_capabilities;

    let request =
        BackendRequest::from_label(requested_backend_label).unwrap_or(BackendRequest::Auto);
    let caps = current_kernel_capabilities();

    match select_backend(request, &caps) {
        Ok(result) => Ok(RunBackendIdentity {
            requested_backend: result.requested_backend(),
            selected_backend: result.selected_backend(),
            runtime_api: result.runtime_api().to_string(),
            fallback_used: result.fallback_used(),
            fallback_reason: result.fallback_reason().map(str::to_string),
        }),
        Err(err) if strict_backend => Err(err.into()),
        Err(err) => Ok(RunBackendIdentity {
            requested_backend: request.to_string(),
            selected_backend: "cpu".to_string(),
            runtime_api: "cpu".to_string(),
            fallback_used: request != BackendRequest::Auto && request != BackendRequest::Cpu,
            fallback_reason: Some(err.to_string()),
        }),
    }
}

fn detected_cpu_feature_labels() -> Vec<String> {
    let features = bitnet_common::runtime_diag::CpuFeatures::detect();
    let mut labels = Vec::new();
    if features.neon {
        labels.push("neon".to_string());
    }
    if features.avx512f {
        labels.push("avx512f".to_string());
    }
    if features.avx2 {
        labels.push("avx2".to_string());
    }
    if features.avx {
        labels.push("avx".to_string());
    }
    if features.fma {
        labels.push("fma".to_string());
    }
    if features.sse42 {
        labels.push("sse4.2".to_string());
    }
    if features.sse2 {
        labels.push("sse2".to_string());
    }
    if labels.is_empty() {
        labels.push("scalar".to_string());
    }
    labels
}

fn cpu_kernel_implementation(quantization: bitnet_common::QuantizationType) -> &'static str {
    if matches!(quantization, bitnet_common::QuantizationType::I2S) && cfg!(target_arch = "aarch64")
    {
        // The current Apple CPU proof path has NEON available, but the packed
        // GGUF I2_S reference kernel is still scalar. Keep the receipt honest.
        return "scalar";
    }
    bitnet_common::runtime_diag::CpuFeatures::detect().best_simd()
}

fn effective_thread_count(threads: usize) -> usize {
    if threads > 0 {
        return threads;
    }

    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|threads| *threads > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1))
}

fn kernel_family_for_quantization(quantization: bitnet_common::QuantizationType) -> &'static str {
    match quantization {
        bitnet_common::QuantizationType::I2S => "i2_s",
        bitnet_common::QuantizationType::TL1 => "tl1",
        bitnet_common::QuantizationType::TL2 => "tl2",
    }
}

fn layout_source_for_quantization(quantization: bitnet_common::QuantizationType) -> &'static str {
    match quantization {
        bitnet_common::QuantizationType::I2S => "gguf_packed_i2_s_reference",
        bitnet_common::QuantizationType::TL1 => "tl1_reference",
        bitnet_common::QuantizationType::TL2 => "tl2_reference",
    }
}

fn kernel_layout_for_quantization(quantization: bitnet_common::QuantizationType) -> &'static str {
    match quantization {
        bitnet_common::QuantizationType::I2S => "gguf_packed_i2_s",
        bitnet_common::QuantizationType::TL1 => "tl1",
        bitnet_common::QuantizationType::TL2 => "tl2",
    }
}

fn dequantizes_before_compute(quantization: bitnet_common::QuantizationType) -> bool {
    !matches!(quantization, bitnet_common::QuantizationType::I2S)
}

fn infer_model_repo(path: &std::path::Path) -> String {
    let normalized = path.to_string_lossy().to_ascii_lowercase();
    if normalized.contains("bitnet-b1.58-2b-4t")
        || normalized.contains("microsoft-bitnet-b1.58-2b-4t")
    {
        "microsoft/bitnet-b1.58-2B-4T-gguf".to_string()
    } else {
        "local".to_string()
    }
}

fn infer_model_architecture(path: &std::path::Path) -> String {
    if infer_model_repo(path) == "microsoft/bitnet-b1.58-2B-4T-gguf" {
        "bitnet_b1_58".to_string()
    } else {
        "unknown".to_string()
    }
}

fn receipt_model_format(
    path: &std::path::Path,
    requested_format: &str,
    is_hf_directory: bool,
) -> String {
    if is_hf_directory {
        return "huggingface".to_string();
    }
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        .filter(|ext| ext == "gguf" || ext == "safetensors")
        .unwrap_or_else(|| requested_format.to_string())
}

fn infer_tokenizer_label(
    tokenizer: &dyn bitnet_tokenizers::Tokenizer,
    source: bitnet_tokenizers::auto::TokenizerSource,
) -> String {
    if tokenizer.token_to_id("<|eot_id|>").is_some() {
        "llama3".to_string()
    } else {
        source.as_str().to_string()
    }
}

fn compute_model_sha256(path: &std::path::Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut file =
        std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Run text generation with sampling
#[allow(clippy::too_many_arguments)]
async fn run_simple_generation(
    requested_backend_label: &str,
    model_path: std::path::PathBuf,
    model_format: String,
    _architecture: Option<String>,
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
    strict_loader: bool,
    json_out: Option<std::path::PathBuf>,
    dump_ids: bool,
    bos: bool,
    greedy: bool,
    deterministic: bool,
    threads: usize,
    prompt_template: String,
    system_prompt: Option<String>,
    stop: Vec<String>,
    stop_id: Vec<u32>,
    dump_logit_steps: Option<usize>,
    logits_topk: usize,
    assert_greedy: bool,
    no_warnings: bool,
) -> Result<()> {
    use bitnet_common::Device;
    use bitnet_models::{Model, transformer::KVCache};
    use bitnet_sampling::{SamplingConfig, SamplingStrategy};
    use bitnet_tokenizers::Tokenizer;
    use std::sync::Arc;

    // Validate --model-format
    match model_format.as_str() {
        "auto" | "gguf" | "safetensors" => {}
        other => {
            anyhow::bail!(
                "Invalid --model-format '{}'. Supported values: auto, gguf, safetensors",
                other
            );
        }
    }

    // Resolve model format: auto-detect from path when format is "auto"
    let is_hf_directory = match model_format.as_str() {
        "gguf" => false,
        "safetensors" => true,
        _ => model_path.is_dir(),
    };

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

    // Set strict loader mode if requested (AC1: fail-fast with enhanced loader + strict tolerance)
    if strict_loader {
        unsafe {
            std::env::set_var("BITNET_DISABLE_MINIMAL_LOADER", "1");
            std::env::set_var("BITNET_STRICT_MODE", "1");
        }
        debug!("Strict loader enabled (BITNET_DISABLE_MINIMAL_LOADER=1, BITNET_STRICT_MODE=1)");
    }

    // Override temperature if greedy mode
    let temperature = if greedy { 0.0 } else { temperature };

    let strict_backend = strict_loader
        || std::env::var("BITNET_STRICT_MODE")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
    let backend_identity = resolve_run_backend_identity(requested_backend_label, strict_backend)?;

    // Parse and resolve template type
    use bitnet_inference::TemplateType;
    let template_type: TemplateType = if prompt_template == "auto" {
        // Auto-detect will be done after loading tokenizer
        TemplateType::Instruct // Default fallback
    } else {
        prompt_template.parse().with_context(|| {
            format!(
                "Invalid prompt template '{}'. Supported: raw, instruct, llama3-chat",
                prompt_template
            )
        })?
    };

    if is_hf_directory {
        println!("Loading HuggingFace model from directory: {}", model_path.display());
    } else {
        println!("Loading model from: {}", model_path.display());
    }

    // Check for QK256 scalar kernel usage and emit performance warnings (GGUF only)
    if !no_warnings && !is_hf_directory {
        check_and_warn_qk256_performance(&model_path, max_new_tokens)?;
    }

    // Try real loader first
    use bitnet_models::loader::{LoadConfig, ModelLoader};

    let loader = ModelLoader::new(Device::Cpu);
    let load_config =
        LoadConfig { use_mmap: true, validate_checksums: false, progress_callback: None };
    let loader_mode;

    let (model, config): (Arc<dyn Model>, _) = match loader
        .load_with_config(&model_path, &load_config)
    {
        Ok(m) => {
            let cfg = m.config().clone();
            loader_mode = detect_loader_mode_for_path(&model_path, is_hf_directory);
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
            if !strict_loader {
                unsafe {
                    std::env::set_var("BITNET_ALLOW_MINIMAL_LOADER", "1");
                }
                warn!(
                    "BITNET_ALLOW_MINIMAL_LOADER=1 enabled by --allow-mock for compatibility fallback"
                );
            }
            // Mock fallback
            let load_result = bitnet_models::gguf_simple::load_gguf_full(
                &model_path,
                Device::Cpu,
                bitnet_models::GGUFLoaderConfig::default(),
            )
            .context("Mock loader also failed")?;
            loader_mode = load_result.loader_mode.as_str();
            warn!("GGUF loader mode: {}", loader_mode);
            let mut raw_tensors = std::collections::HashMap::new();
            for (name, qk256) in load_result.i2s_qk256 {
                let expected_bytes = qk256.rows * qk256.row_stride_bytes;
                let mut packed = qk256.qs;
                if packed.len() != expected_bytes {
                    tracing::warn!(
                        "QK256 '{}' byte length {} differs from expected {}; normalizing for runtime tensor",
                        name,
                        packed.len(),
                        expected_bytes
                    );
                    packed.resize(expected_bytes, 0);
                }

                let raw_tensor = candle_core::Tensor::from_raw_buffer(
                    &packed,
                    DType::U8,
                    &[qk256.rows, qk256.row_stride_bytes],
                    &candle_core::Device::Cpu,
                )
                .with_context(|| format!("Failed to build QK256 raw tensor for {name}"))?;

                raw_tensors.insert(format!("{}.qk256_qs", name), raw_tensor);
            }
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

    // Load tokenizer with deterministic CPU-BITNET authority.
    // Priority: explicit path -> GGUF metadata -> sibling tokenizer asset.

    // Track GGUF metadata for JSON output
    let mut gguf_metadata: Option<(usize, usize)> = None;
    let effective_strict_tokenizer = strict_tokenizer || strict_loader;

    let tokenizer_resolution = match bitnet_tokenizers::auto::resolve_tokenizer(
        &model_path,
        tokenizer_path.as_deref(),
        effective_strict_tokenizer,
    ) {
        Ok(resolution) => {
            match resolution.source {
                bitnet_tokenizers::auto::TokenizerSource::Explicit
                | bitnet_tokenizers::auto::TokenizerSource::Sibling => {
                    if let Some(path) = &resolution.path {
                        println!("Loading tokenizer from: {}", path.display());
                    }
                }
                bitnet_tokenizers::auto::TokenizerSource::GgufMetadata => {
                    println!("Successfully loaded tokenizer from GGUF metadata");
                }
                bitnet_tokenizers::auto::TokenizerSource::CompatibilityFallback => {}
            }
            resolution
        }
        Err(e) => {
            if effective_strict_tokenizer {
                eprintln!("Strict tokenizer failed: {e}");
                std::process::exit(EXIT_STRICT_TOKENIZER);
            }
            if !allow_mock {
                let model_dir = if is_hf_directory {
                    model_path.as_path()
                } else {
                    model_path.parent().unwrap_or_else(|| std::path::Path::new("."))
                };
                anyhow::bail!(
                    "{e}\n\
                     \n\
                     No tokenizer found. Solutions:\n\
                     1. Download tokenizer:\n\
                        cargo run -p xtask -- tokenizer --into {}\n\
                     2. Provide explicit tokenizer path:\n\
                        --tokenizer /path/to/tokenizer.json\n\
                     3. Use mock tokenizer for testing only:\n\
                        --allow-mock",
                    model_dir.display()
                );
            }
            println!("Warning: Using mock tokenizer due to: {e}");
            bitnet_tokenizers::auto::TokenizerResolution {
                tokenizer: std::sync::Arc::new(bitnet_tokenizers::MockTokenizer::new()),
                source: bitnet_tokenizers::auto::TokenizerSource::CompatibilityFallback,
                strict: false,
                path: None,
            }
        }
    };
    let tokenizer_source = tokenizer_resolution.source;
    let tokenizer_strict = tokenizer_resolution.strict;
    let tokenizer: std::sync::Arc<dyn Tokenizer + Send + Sync> = tokenizer_resolution.tokenizer;

    if tokenizer_source == bitnet_tokenizers::auto::TokenizerSource::GgufMetadata {
        let gguf_data = std::fs::read(&model_path)
            .context("Failed to read GGUF file for tokenizer metadata")?;
        let reader = bitnet_models::GgufReader::new(&gguf_data)
            .context("Failed to parse GGUF for tokenizer metadata")?;
        gguf_metadata = Some((reader.metadata_keys().len(), reader.tensor_count() as usize));
    }

    // Auto-detect template if needed
    let template_type = if prompt_template == "auto" {
        // Check if tokenizer has special tokens for LLaMA-3
        if tokenizer.token_to_id("<|eot_id|>").is_some() {
            debug!("Auto-detected llama3-chat template (tokenizer has <|eot_id|>)");
            TemplateType::Llama3Chat
        } else {
            debug!("Auto-detected instruct template (fallback)");
            TemplateType::Instruct
        }
    } else {
        template_type
    };

    // Format prompt using the template
    let formatted_prompt = template_type.apply(&prompt, system_prompt.as_deref());

    // Get template's default stop sequences and merge with manual stops
    let template_stops = template_type.default_stop_sequences();
    let mut all_stop_sequences = stop.clone();
    for template_stop in template_stops {
        if !all_stop_sequences.contains(&template_stop) {
            all_stop_sequences.push(template_stop);
        }
    }

    // Resolve template stop sequences to token IDs and merge with manual stop IDs
    let template_stop_ids = template_type.resolve_stop_token_ids(tokenizer.as_ref());
    let mut all_stop_ids = stop_id.clone();
    for template_id in template_stop_ids {
        if !all_stop_ids.contains(&template_id) {
            all_stop_ids.push(template_id);
        }
    }

    debug!(
        "Template: {} | Stop sequences: {:?} | Stop IDs: {:?}",
        template_type, all_stop_sequences, all_stop_ids
    );

    // Determine BOS policy (user flag wins, else template default)
    let bos_policy = if bos {
        true // explicit --bos flag
    } else {
        template_type.should_add_bos() // template default
    };

    // Tokenize formatted prompt with proper BOS policy and special token parsing
    let parse_special = template_type.parse_special();
    let mut tokens = tokenizer.encode(&formatted_prompt, bos_policy, parse_special)?;
    println!("Input tokens ({}): {:?}", tokens.len(), &tokens[..10.min(tokens.len())]);

    // Create KV cache
    let cache = KVCache::new(&config, 1, &candle_core::Device::Cpu)?;
    let mut any_cache: Box<dyn std::any::Any> = Box::new(cache);

    // Create sampler
    let mut sampler = SamplingStrategy::new(SamplingConfig {
        temperature,
        top_k: top_k as u32,
        top_p,
        repetition_penalty,
        seed,
    });

    print!("Generating: {}", formatted_prompt);
    std::io::Write::flush(&mut std::io::stdout())?;

    // Track timing
    let start_time = std::time::Instant::now();
    let mut first_token_ms: Option<u64> = None;

    // Track generated tokens for repetition penalty
    let mut generated_tokens = Vec::new();

    // Track logits dump if requested
    let mut logits_dump: Vec<LogitStep> = Vec::new();

    // Rolling tail for fast string-stop checking (only if we have string stops)
    let max_stop_len = all_stop_sequences.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut tail = if max_stop_len > 0 {
        Some(String::with_capacity(max_stop_len.saturating_add(16)))
    } else {
        None
    };

    // BITNET_TRACE_TIMING=1: Enable timing instrumentation
    let timing_enabled = std::env::var("BITNET_TRACE_TIMING").as_deref() == Ok("1");

    // Generation loop: incremental decoding
    //
    // Each step:
    //   1. Embed ONLY the new token (last in sequence)
    //   2. Forward pass uses KV cache for historical context
    //   3. No need to re-embed previous tokens (O(N) not O(N┬▓))
    //
    // Historical context is maintained via:
    //   - KV cache: stores key/value tensors from previous steps
    //   - `tokens` vector: tracks full sequence for stop detection/logging
    //
    // Performance impact: This changes embedding from O(N┬▓) to O(N), providing
    // ~50├ù speedup for 100-token generation (avoids re-embedding 1+2+...+N tokens).
    for step_idx in 0..max_new_tokens {
        // Embed only the LAST token (incremental)
        // KV cache already maintains historical context
        let last_token = tokens.last().copied().expect("tokens must be non-empty");

        let t0 = if timing_enabled { Some(std::time::Instant::now()) } else { None };
        let x = model.embed(&[last_token])?;
        if let Some(t) = t0 {
            eprintln!("timing: embed_us={}", t.elapsed().as_micros());
        }

        // Forward pass (with KV cache handling history)
        let t1 = if timing_enabled { Some(std::time::Instant::now()) } else { None };
        let h = model.forward(&x, any_cache.as_mut())?;
        if let Some(t) = t1 {
            eprintln!("timing: forward_us={}", t.elapsed().as_micros());
        }

        // Extract last token hidden state first to avoid 3D├ù2D matmul issues
        let last_hidden = extract_last_token_hidden(&h)?;

        // Debug tap: hidden state RMS sanity (catches "everything is zero")
        if std::env::var("BITNET_DEBUG_LOGITS").as_deref() == Ok("1") && step_idx == 0 {
            let h_vec = tensor_to_vec(&last_hidden)?;
            let hidden_rms = compute_rms(&h_vec);
            eprintln!("hidden_rms={:.6}", hidden_rms);
        }

        // Get logits from last token hidden state
        let t2 = if timing_enabled { Some(std::time::Instant::now()) } else { None };
        let logits = model.logits(&last_hidden)?;
        if let Some(t) = t2 {
            eprintln!("timing: logits_us={}", t.elapsed().as_micros());
        }

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
        let t3 = if timing_enabled { Some(std::time::Instant::now()) } else { None };
        let next_token = sampler.sample(&logits_vec, &generated_tokens)?;
        if let Some(t) = t3 {
            eprintln!("timing: sample_us={}", t.elapsed().as_micros());
        }

        // BITNET_PARITY=1: Log chosen token + top-10 logits for greedy decode verification
        if std::env::var("BITNET_PARITY").as_deref() == Ok("1") {
            // Extract top-10 logits with token IDs
            let mut logits_with_idx: Vec<(usize, f32)> =
                logits_vec.iter().copied().enumerate().collect();
            logits_with_idx
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top_k_logits: Vec<(u32, f32)> =
                logits_with_idx.iter().take(10).map(|(idx, logit)| (*idx as u32, *logit)).collect();

            // JSON format for easy parsing
            eprintln!(
                "{{\"step\":{},\"token\":{},\"top_k\":{}}}",
                step_idx,
                next_token,
                serde_json::to_string(&top_k_logits).unwrap_or_default()
            );
        }

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

        // Maintain rolling tail (if present)
        if let Some(t) = &mut tail {
            t.push_str(&token_text);
            if t.len() > max_stop_len {
                let cut = t.len() - max_stop_len;
                // SAFETY: Find char boundary (compatible with MSRV 1.90.0)
                // floor_char_boundary is 1.91.0+, so we implement manually
                let mut safe_cut = cut;
                while safe_cut > 0 && !t.is_char_boundary(safe_cut) {
                    safe_cut -= 1;
                }
                t.drain(..safe_cut);
            }
        }

        // 1) Token-ID stops (includes template-resolved IDs like <|eot_id|>)
        if all_stop_ids.contains(&next_token) {
            debug!("Stopped on token ID: {}", next_token);
            break;
        }

        // 2) EOS fallback
        if let Some(eos) = tokenizer.eos_token_id()
            && next_token == eos
        {
            debug!("Stopped on EOS token");
            break;
        }

        // 3) String-based stops on rolling tail (no full decode)
        if let Some(t) = &tail
            && !all_stop_sequences.is_empty()
            && all_stop_sequences.iter().any(|pat| t.ends_with(pat))
        {
            if let Some(hit) = all_stop_sequences.iter().find(|pat| t.ends_with(*pat)) {
                debug!("Stopped on sequence: {:?}", hit);
            }
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
        let tokenizer_source_str = tokenizer_source.as_str();
        let tokenizer_info = serde_json::json!({
            "type": "sentencepiece",
            "origin": if tokenizer_source == bitnet_tokenizers::auto::TokenizerSource::GgufMetadata {
                "embedded"
            } else {
                "external"
            },
            "source": tokenizer_source_str,
            "strict": tokenizer_strict,
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
        let loader_info = serde_json::json!({
            "mode": loader_mode,
            "minimal_fallback_allowed": std::env::var("BITNET_ALLOW_MINIMAL_LOADER").as_deref() == Ok("1"),
            "minimal_fallback_disabled": std::env::var("BITNET_DISABLE_MINIMAL_LOADER").as_deref() == Ok("1")
                || std::env::var("BITNET_STRICT_MODE").as_deref() == Ok("1"),
            "minimal_loader_fallback_used": loader_mode != bitnet_models::GgufLoaderMode::RealGguf.as_str(),
            "tokenizer_source": tokenizer_source_str,
            "mock_tensors_used": loader_mode != bitnet_models::GgufLoaderMode::RealGguf.as_str(),
        });

        let prompt_tokens_len = tokens.len() - generated_tokens.len();
        let kernel_family = kernel_family_for_quantization(config.quantization.quantization_type);
        let kernel_implementation =
            cpu_kernel_implementation(config.quantization.quantization_type);
        let selected_kernel = format!("{kernel_family}-{kernel_implementation}-reference");
        let layout_source = layout_source_for_quantization(config.quantization.quantization_type);
        let kernel_layout = kernel_layout_for_quantization(config.quantization.quantization_type);
        let dequantizes_before_compute =
            dequantizes_before_compute(config.quantization.quantization_type);
        let model_sha256 = compute_model_sha256(&model_path)?;
        let model_repo = infer_model_repo(&model_path);
        let canonical_bitnet_model = model_repo == "microsoft/bitnet-b1.58-2B-4T-gguf";
        let model_architecture = infer_model_architecture(&model_path);
        let model_format_label = receipt_model_format(&model_path, &model_format, is_hf_directory);
        let model_file =
            model_path.file_name().and_then(|name| name.to_str()).unwrap_or_default().to_string();
        let tokenizer_label = infer_tokenizer_label(tokenizer.as_ref(), tokenizer_source);
        let thread_count = effective_thread_count(threads);
        let cpu_features = detected_cpu_feature_labels();
        let fallback_reason = backend_identity.fallback_reason.clone();
        let requested_backend = backend_identity.requested_backend.as_str();
        let selected_backend = backend_identity.selected_backend.as_str();
        let runtime_api = backend_identity.runtime_api.as_str();
        let apple_machine = apple_machine_receipt_json(requested_backend, selected_backend);
        let strict_cpu_reference_artifact = strict_backend
            && canonical_bitnet_model
            && runtime_api == "cpu"
            && loader_mode == bitnet_models::GgufLoaderMode::RealGguf.as_str();
        let artifact_kind = if strict_cpu_reference_artifact {
            "strict_bitnet_cpu_reference"
        } else {
            "inference_result"
        };
        let mut output = serde_json::json!({
            "schema_version": "1.0.0",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "artifact_kind": artifact_kind,
            "artifact_path": json_path.display().to_string(),
            "requested_backend": requested_backend,
            "selected_backend": selected_backend,
            "runtime_api": runtime_api,
            "fallback_used": backend_identity.fallback_used,
            "fallback_reason": fallback_reason,
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
            "model": {
                "repo": model_repo,
                "file": model_file,
                "path": model_path.display().to_string(),
                "sha256": model_sha256,
                "format": model_format_label,
                "architecture": model_architecture,
                "context_length": config.model.max_position_embeddings,
                "tokenizer": tokenizer_label,
                "vocab_size": tokenizer.vocab_size(),
                "loader_mode": loader_mode,
            },
            "bitnet": {
                "weight_quantization": if canonical_bitnet_model { "W1.58" } else { "unknown" },
                "activation_quantization": if canonical_bitnet_model { "A8" } else { "unknown" },
                "kernel_format": kernel_family,
                "kernel_family": kernel_family,
                "execution_phase": "decode",
                "layout_source": layout_source,
                "fallback_layout": serde_json::Value::Null,
            },
            "execution": {
                "phase": "decode",
                "prompt_tokens": prompt_tokens_len,
                "generated_tokens": generated_tokens.len(),
                "batch_size": 1,
                "thread_count": thread_count,
                "requested_backend": requested_backend,
                "selected_backend": selected_backend,
                "runtime_api": runtime_api,
                "fallback_used": backend_identity.fallback_used,
                "fallback_reason": backend_identity.fallback_reason.as_deref(),
            },
            "kernel": {
                "family": kernel_family,
                "implementation": kernel_implementation,
                "layout": kernel_layout,
                "dequantizes_before_compute": dequantizes_before_compute,
                "kernel_id": selected_kernel.as_str(),
            },
            "cpu": {
                "arch": std::env::consts::ARCH,
                "features": &cpu_features,
                "threads": thread_count,
            },
            "strict_provenance": {
                "requested_backend": requested_backend,
                "selected_backend": selected_backend,
                "requested_kernel": selected_kernel.as_str(),
                "selected_kernel": selected_kernel.as_str(),
                "loader_mode": loader_mode,
                "tokenizer_source": tokenizer_source_str,
                "model_family": "bitnet",
                "quant_format": format!("{}", config.quantization.quantization_type),
                "cpu_features": &cpu_features,
                "thread_count": thread_count,
                "fallback_used": backend_identity.fallback_used,
                "fallback_reason": backend_identity.fallback_reason.as_deref(),
                "prompt_tokens": prompt_tokens_len,
                "decode_tokens": generated_tokens.len(),
                "decode_tps": tok_per_sec,
            },
            "counts": counts,
            "tokenizer": tokenizer_info,
            "loader": loader_info,
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
        if let Some(apple_machine) = apple_machine
            && let Some(object) = output.as_object_mut()
        {
            object.insert("machine_id".to_string(), apple_machine["machine_id"].clone());
            object.insert("resolved_device".to_string(), apple_machine["resolved_device"].clone());
            object.insert("apple".to_string(), apple_machine);
        }
        write_json_output(Some(&json_path), &output)?;
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

fn apple_machine_receipt_json(
    requested_backend: &str,
    selected_backend: &str,
) -> Option<serde_json::Value> {
    if !is_apple_m4_backend_label(requested_backend) && !is_apple_m4_backend_label(selected_backend)
    {
        return None;
    }

    let probe = probe_apple_cli_machine();
    Some(apple_machine_receipt_json_from_probe(&probe))
}

fn is_apple_m4_backend_label(label: &str) -> bool {
    label.trim().to_ascii_lowercase().starts_with("apple-m4-")
}

#[derive(Debug, Clone, Default)]
struct AppleCliMachineProbe {
    chip: Option<String>,
    cpu_cores: Option<usize>,
    gpu_cores: Option<usize>,
    unified_memory: Option<bool>,
    unified_memory_bytes: Option<u64>,
    macos_version: Option<String>,
    macos_build: Option<String>,
    native_or_virtualized: Option<String>,
    metal_visible: bool,
}

fn probe_apple_cli_machine() -> AppleCliMachineProbe {
    if std::env::consts::OS != "macos" {
        return AppleCliMachineProbe {
            native_or_virtualized: Some("not-macos".to_string()),
            ..AppleCliMachineProbe::default()
        };
    }

    let sw_vers = command_stdout_text("sw_vers", &[]).0;
    let hardware = command_stdout_text("system_profiler", &["SPHardwareDataType"]).0;
    let displays = command_stdout_text("system_profiler", &["SPDisplaysDataType"]).0;
    let (metal, metal_success) = command_stdout_text("system_profiler", &["SPMetalDataType"]);
    let memsize = command_stdout_text("sysctl", &["hw.memsize"]).0;
    let virtualization = command_stdout_text("sysctl", &["kern.hv_vmm_present"]).0;

    let chip = parse_receipt_colon_value(&hardware, "Chip")
        .or_else(|| parse_receipt_colon_value(&metal, "Chipset Model"))
        .or_else(|| parse_receipt_colon_value(&displays, "Chipset Model"));
    let unified_memory = if chip.as_deref().is_some_and(|value| value.starts_with("Apple M")) {
        Some(true)
    } else if chip.is_some() {
        Some(false)
    } else {
        None
    };

    AppleCliMachineProbe {
        chip,
        cpu_cores: parse_receipt_colon_value(&hardware, "Total Number of Cores")
            .and_then(|value| parse_receipt_first_usize(&value)),
        gpu_cores: parse_receipt_colon_value(&metal, "Total Number of Cores")
            .or_else(|| parse_receipt_colon_value(&displays, "Total Number of Cores"))
            .and_then(|value| parse_receipt_first_usize(&value)),
        unified_memory,
        unified_memory_bytes: parse_receipt_colon_value(&memsize, "hw.memsize").and_then(|value| {
            value.split_whitespace().next().and_then(|number| number.parse::<u64>().ok())
        }),
        macos_version: parse_receipt_colon_value(&sw_vers, "ProductVersion"),
        macos_build: parse_receipt_colon_value(&sw_vers, "BuildVersion"),
        native_or_virtualized: parse_receipt_virtualization_state(&virtualization),
        metal_visible: (metal_success && receipt_metal_text_reports_visibility(&metal))
            || receipt_metal_text_reports_visibility(&displays),
    }
}

fn command_stdout_text(command: &str, args: &[&str]) -> (String, bool) {
    std::process::Command::new(command)
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .map_or_else(
            |_| (String::new(), false),
            |output| {
                (String::from_utf8_lossy(&output.stdout).into_owned(), output.status.success())
            },
        )
}

fn parse_receipt_colon_value(output: &str, key: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let trimmed = line.trim();
        let value = trimmed.strip_prefix(key)?.trim_start().strip_prefix(':')?.trim();
        (!value.is_empty()).then(|| value.to_owned())
    })
}

fn parse_receipt_first_usize(value: &str) -> Option<usize> {
    let mut digits = String::new();
    for ch in value.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
        } else if !digits.is_empty() {
            break;
        }
    }
    digits.parse().ok()
}

fn parse_receipt_virtualization_state(output: &str) -> Option<String> {
    let value = parse_receipt_colon_value(output, "kern.hv_vmm_present")?;
    match value.split_whitespace().next() {
        Some("0") => Some("native-macos".to_string()),
        Some("1") => Some("virtualized-macos".to_string()),
        _ => Some("unknown".to_string()),
    }
}

fn receipt_metal_text_reports_visibility(output: &str) -> bool {
    let lower = output.to_ascii_lowercase();
    lower.contains("metal")
        && (lower.contains("chipset model")
            || lower.contains("metal support")
            || lower.contains("metal family")
            || lower.contains("gpu"))
}

fn apple_machine_receipt_json_from_probe(probe: &AppleCliMachineProbe) -> serde_json::Value {
    let mut resolved_device = serde_json::Map::new();
    resolved_device.insert(
        "chip".to_string(),
        serde_json::Value::String(probe.chip.clone().unwrap_or_else(|| "unknown".to_string())),
    );
    if let Some(cpu_cores) = probe.cpu_cores {
        resolved_device.insert("cpu_cores".to_string(), serde_json::json!(cpu_cores));
    }
    if let Some(gpu_cores) = probe.gpu_cores {
        resolved_device.insert("gpu_cores".to_string(), serde_json::json!(gpu_cores));
    }
    if let Some(unified_memory) = probe.unified_memory {
        resolved_device.insert("unified_memory".to_string(), serde_json::json!(unified_memory));
    }
    if let Some(unified_memory_bytes) = probe.unified_memory_bytes {
        resolved_device
            .insert("unified_memory_bytes".to_string(), serde_json::json!(unified_memory_bytes));
    }

    serde_json::json!({
        "machine_id": "apple-m4-mac-mini",
        "resolved_device": resolved_device,
        "macos": {
            "version": probe.macos_version,
            "build": probe.macos_build,
            "native_or_virtualized": probe.native_or_virtualized,
        },
        "metal_visible": probe.metal_visible,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apple_cpu_neon_identity_is_preserved_or_visible_fallback() {
        let identity = resolve_run_backend_identity("apple-m4-cpu-neon", false).unwrap();

        assert_eq!(identity.requested_backend, "apple-m4-cpu-neon");
        assert_eq!(identity.runtime_api, "cpu");
        assert!(
            identity.selected_backend == "apple-m4-cpu-neon" || identity.selected_backend == "cpu",
            "unexpected selected backend: {}",
            identity.selected_backend
        );
        if identity.selected_backend == "cpu" {
            assert!(identity.fallback_used);
            assert!(identity.fallback_reason.is_some());
        }
    }

    #[test]
    fn i2s_receipt_kernel_family_is_stable() {
        assert_eq!(kernel_family_for_quantization(bitnet_common::QuantizationType::I2S), "i2_s");
    }

    #[test]
    fn i2s_receipt_records_packed_reference_layout() {
        assert_eq!(
            layout_source_for_quantization(bitnet_common::QuantizationType::I2S),
            "gguf_packed_i2_s_reference"
        );
        assert_eq!(
            kernel_layout_for_quantization(bitnet_common::QuantizationType::I2S),
            "gguf_packed_i2_s"
        );
        assert!(!dequantizes_before_compute(bitnet_common::QuantizationType::I2S));
    }

    #[test]
    fn apple_i2s_receipt_does_not_overclaim_neon_kernel() {
        #[cfg(target_arch = "aarch64")]
        assert_eq!(cpu_kernel_implementation(bitnet_common::QuantizationType::I2S), "scalar");
    }

    #[test]
    fn known_bitnet_model_path_records_canonical_repo() {
        let repo = infer_model_repo(std::path::Path::new(
            "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
        ));

        assert_eq!(repo, "microsoft/bitnet-b1.58-2B-4T-gguf");
    }

    #[test]
    fn apple_m4_receipt_includes_resolved_machine_fields() {
        let probe = AppleCliMachineProbe {
            chip: Some("Apple M4".to_string()),
            cpu_cores: Some(10),
            gpu_cores: Some(10),
            unified_memory: Some(true),
            unified_memory_bytes: Some(17_179_869_184),
            macos_version: Some("15.4".to_string()),
            macos_build: Some("24E248".to_string()),
            native_or_virtualized: Some("native-macos".to_string()),
            metal_visible: true,
        };
        let receipt = apple_machine_receipt_json_from_probe(&probe);

        assert_eq!(receipt["machine_id"], "apple-m4-mac-mini");
        assert_eq!(receipt["resolved_device"]["chip"], "Apple M4");
        assert_eq!(receipt["resolved_device"]["cpu_cores"], 10);
        assert_eq!(receipt["resolved_device"]["gpu_cores"], 10);
        assert_eq!(receipt["resolved_device"]["unified_memory"], true);
        assert_eq!(receipt["resolved_device"]["unified_memory_bytes"], 17_179_869_184_u64);
        assert_eq!(receipt["macos"]["native_or_virtualized"], "native-macos");
        assert_eq!(receipt["metal_visible"], true);
    }

    #[test]
    fn non_apple_backend_does_not_probe_apple_machine_fields() {
        assert!(apple_machine_receipt_json("cpu", "cpu").is_none());
    }

    #[test]
    fn apple_receipt_metal_visibility_accepts_display_profiler_output() {
        assert!(receipt_metal_text_reports_visibility(
            "Graphics/Displays:\n\n    Apple M4:\n      Chipset Model: Apple M4\n      Metal Support: Metal 4\n",
        ));
    }
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
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        println!("  GPU support: {}", style("Γ£ô Enabled").green());
        // Check CUDA availability
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            match candle_core::Device::cuda_if_available(0).is_ok() {
                true => println!("  CUDA: {}", style("Γ£ô Available").green()),
                false => println!("  CUDA: {}", style("Γ£ù Not available").red()),
            }
        }
        #[cfg(not(any(feature = "gpu", feature = "cuda")))]
        println!("  CUDA: {}", style("Γ£ù Not compiled").yellow())
    }
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    {
        println!("  GPU support: {}", style("Γ£ù Disabled").red());
    }

    // CPU features
    println!("  CPU features:");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            println!("    AVX2: {}", style("Γ£ô").green());
        } else {
            println!("    AVX2: {}", style("Γ£ù").red());
        }
        if is_x86_feature_detected!("avx512f") {
            println!("    AVX-512: {}", style("Γ£ô").green());
        } else {
            println!("    AVX-512: {}", style("Γ£ù").red());
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            println!("    NEON: {}", style("Γ£ô").green());
        } else {
            println!("    NEON: {}", style("Γ£ù").red());
        }
    }

    println!();

    // Model formats
    println!("{}", style("Supported formats:").bold());
    println!("  GGUF: {}", style("Γ£ô").green());
    println!("  SafeTensors: {}", style("Γ£ô").green());
    println!("  HuggingFace: {}", style("Γ£ô").green());
    println!();

    // Quantization types
    println!("{}", style("Quantization types:").bold());
    println!("  I2_S (2-bit signed): {}", style("Γ£ô").green());
    println!("  TL1 (ARM optimized): {}", style("Γ£ô").green());
    println!("  TL2 (x86 optimized): {}", style("Γ£ô").green());

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
            fn canonicalize_quantization_name(name: &str) -> Option<&'static str> {
                match bitnet_models::formats::gguf::GgufTensorType::from_quant_string(name) {
                    Some(bitnet_models::formats::gguf::GgufTensorType::I2_S) => Some("I2_S"),
                    Some(bitnet_models::formats::gguf::GgufTensorType::IQ2_S) => Some("IQ2_S"),
                    _ => None,
                }
            }

            fn get_quantization(reader: &GgufReader) -> String {
                if let Some(q) = reader.get_string_metadata("general.quantization_type") {
                    canonicalize_quantization_name(&q).map(str::to_string).unwrap_or(q)
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
        println!("Status:    Γ£ô Valid GGUF");
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
            eprintln!("ΓÜá Unusually high tensor/KV counts detected");
        }
        if !supported {
            eprintln!("ΓÜá Unsupported GGUF version");
        }
    }

    if strict && (!supported || suspicious) {
        std::process::exit(4);
    }
    Ok(())
}
