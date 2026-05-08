//! BitNet CLI application
//!
//! A comprehensive command-line interface for BitNet 1-bit LLM inference.
//! Supports model loading, inference, conversion, benchmarking, and serving.

// COMPILE-TIME FIREWALL: Prevent mock feature in production CLI
#[cfg(feature = "mock")]
compile_error!("The 'mock' feature must never be enabled for the CLI - tests only.");

use anyhow::{Context, Result};
use bitnet_common::Tensor;
use bitnet_startup_contract_guard::{
    ContractPolicy, RuntimeComponent, evaluate_and_emit, feature_line,
};
use candle_core::{DType, IndexOp};
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{Shell, generate};
use console::style;
use std::alloc::{GlobalAlloc, Layout, System};
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tracing::{debug, error, info, warn};

#[global_allocator]
static ALLOCATION_AUDIT_ALLOCATOR: AllocationAuditAllocator = AllocationAuditAllocator;

static ALLOCATION_AUDIT_ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCATION_AUDIT_ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOCATION_AUDIT_ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOCATION_AUDIT_DEALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOCATION_AUDIT_DEALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

struct AllocationAuditAllocator;

unsafe impl GlobalAlloc for AllocationAuditAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && ALLOCATION_AUDIT_ENABLED.load(Ordering::Relaxed) {
            record_allocation_audit_alloc(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() && ALLOCATION_AUDIT_ENABLED.load(Ordering::Relaxed) {
            record_allocation_audit_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if ALLOCATION_AUDIT_ENABLED.load(Ordering::Relaxed) {
            record_allocation_audit_dealloc(layout.size());
        }
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() && ALLOCATION_AUDIT_ENABLED.load(Ordering::Relaxed) {
            record_allocation_audit_dealloc(layout.size());
            record_allocation_audit_alloc(new_size);
        }
        new_ptr
    }
}

fn record_allocation_audit_alloc(size: usize) {
    ALLOCATION_AUDIT_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    ALLOCATION_AUDIT_ALLOC_BYTES.fetch_add(size as u64, Ordering::Relaxed);
}

fn record_allocation_audit_dealloc(size: usize) {
    ALLOCATION_AUDIT_DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    ALLOCATION_AUDIT_DEALLOC_BYTES.fetch_add(size as u64, Ordering::Relaxed);
}

#[cfg(feature = "full-cli")]
mod commands;
mod config;
mod exit;
mod intel_arc;
mod intel_npu;
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
use commands::{
    AnswerCorpusCommand, AnswerParityCommand, ConvertCommand, InferenceCommand, InspectCommand,
    ServeCommand,
};
use config::{CliConfig, ConfigBuilder, DEVICE_HELP};

/// BitNet CLI - High-performance 1-bit LLM inference toolkit
#[derive(Parser)]
#[command(name = "bitnet")]
#[command(about = "BitNet-rs - 1-bit neural network inference with strict receipts")]
#[command(long_about = r#"BitNet-rs CLI - one-shot generation and chat with strict receipts

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

  # Apple M4 local answer path: CPU/NEON is the reliable user-facing route today.
  # The JSON receipt records requested_backend, selected_backend, runtime_api, and fallback_used.
  RUST_LOG=warn bitnet --device apple-m4-cpu-neon run \
    --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    --prompt "What is 2+2? Answer briefly." --max-tokens 32 \
    --temperature 0.0 --greedy --deterministic \
    --strict-loader --strict-tokenizer --json-out local-answer-cpu-neon.json

APPLE M4 ROUTING:
  apple-m4-cpu-neon: reliable local-answer path with strict receipts.
  apple-m4-metal: receipt-backed Metal phase/subgraph proof only unless a strict
    full-model Metal receipt later proves more.
  apple-m4-mpsgraph: graph/reference lane, not native Metal or Neural Engine proof.

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
    - With AVX2: ~1.2x faster (optimized kernels)
    - For quick validation: use --max-tokens 4-16
    - SIMD optimizations (>=3x faster) coming in v0.2.0
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

    #[arg(short, long, value_name = "DEVICE", global = true, help = DEVICE_HELP)]
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
    /// Apple M4 local answer path with strict CPU/NEON receipt:
    ///   bitnet --device apple-m4-cpu-neon run \
    ///     --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    ///     --prompt "What is 2+2? Answer briefly." --max-tokens 32 \
    ///     --temperature 0.0 --greedy --deterministic \
    ///     --strict-loader --strict-tokenizer --json-out local-answer-cpu-neon.json
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

        /// Profile label to record in JSON receipts (for example: smoke_1, prefill_512, decode_128)
        #[arg(long, value_name = "PROFILE")]
        profile_id: Option<String>,

        /// Measure scoped hot-loop allocation counter deltas in profile receipts
        #[arg(long, default_value_t = false)]
        allocation_audit: bool,
    },

    /// Ask one question using the answer-readiness generation path
    Ask {
        /// Model file or directory path (.gguf file or HuggingFace model directory)
        #[arg(short, long)]
        model: std::path::PathBuf,

        /// Optional explicit tokenizer path
        #[arg(long)]
        tokenizer: Option<std::path::PathBuf>,

        /// User question to answer
        #[arg(short, long)]
        question: String,

        /// Optional system prompt
        #[arg(long = "system", value_name = "TEXT")]
        system_prompt: Option<String>,

        /// Maximum new tokens to generate (aliases: --max-tokens, --n-predict)
        #[arg(long, visible_aliases = ["max-tokens", "n-predict"], default_value_t = 96)]
        max_new_tokens: usize,

        /// Temperature for sampling. The default ask path is deterministic greedy.
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,

        /// Top-k sampling (0 = disabled)
        #[arg(long, default_value_t = 0)]
        top_k: usize,

        /// Top-p (nucleus) sampling
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,

        /// Require the selected backend to be the RTX 5070 Ti CUDA proof lane
        #[arg(long, default_value_t = false)]
        strict_cuda: bool,

        /// Require strict real-model CPU execution with no fallback
        #[arg(long, default_value_t = false)]
        strict_cpu: bool,

        /// Output answer-shaped receipt to file
        #[arg(long, value_name = "PATH")]
        receipt_out: Option<std::path::PathBuf>,
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
    /// Run the fixed CPU answer-readiness corpus through the `run` surface
    AnswerCorpus(Box<AnswerCorpusCommand>),

    #[cfg(feature = "full-cli")]
    /// Compare answer-corpus receipts for backend parity diagnostics
    AnswerParity(Box<AnswerParityCommand>),

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

    /// Probe Lunar Lake 258V platform visibility without launching kernels
    LunarLakeProbe {
        /// Output JSON probe receipt to file
        #[arg(long)]
        json_out: Option<std::path::PathBuf>,
    },

    /// Probe Intel NPU OpenVINO runtime visibility without compiling graphs
    IntelNpuProbe {
        /// Require OpenVINO to report an NPU runtime device
        #[arg(long, default_value_t = false)]
        strict: bool,

        /// Output JSON probe receipt to file
        #[arg(long)]
        json_out: Option<std::path::PathBuf>,
    },

    /// Run a tiny static OpenVINO NPU graph smoke without BitNet inference
    IntelNpuSmoke {
        /// Require tiny graph execution to pass
        #[arg(long, default_value_t = false)]
        strict: bool,

        /// Output JSON smoke receipt to file
        #[arg(long)]
        json_out: Option<std::path::PathBuf>,
    },

    /// Run selected static BitNet subgraph parity on OpenVINO NPU
    IntelNpuBitnetSubgraph {
        /// Require selected subgraph parity to pass
        #[arg(long, default_value_t = false)]
        strict: bool,

        /// Output JSON parity receipt to file
        #[arg(long)]
        json_out: Option<std::path::PathBuf>,
    },

    /// Run a tiny static OpenVINO GPU.0 graph smoke for Arc 140V
    #[command(name = "intel-arc-140v-openvino-gpu-smoke")]
    IntelArc140vOpenvinoGpuSmoke {
        /// Require Arc 140V identity and tiny graph execution to pass
        #[arg(long, default_value_t = false)]
        strict: bool,

        /// Output JSON smoke receipt to file
        #[arg(long)]
        json_out: Option<std::path::PathBuf>,
    },

    /// Run a tiny native OpenCL kernel smoke for Arc 140V
    #[command(name = "intel-arc-140v-opencl-smoke")]
    IntelArc140vOpenclSmoke {
        /// Require Arc 140V native OpenCL kernel execution to pass
        #[arg(long, default_value_t = false)]
        strict: bool,

        /// Output JSON smoke receipt to file
        #[arg(long)]
        json_out: Option<std::path::PathBuf>,
    },

    /// Run validation-only preflight checks
    Validate {
        #[command(subcommand)]
        action: ValidateAction,
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
enum ValidateAction {
    /// Emit a CPU BitNet validation preflight receipt without running inference
    CpuBitnet {
        /// Machine label for the validation target
        #[arg(long, default_value = "intel-258v")]
        machine: String,

        /// Canonical BitNet GGUF model path
        #[arg(long)]
        model: std::path::PathBuf,

        /// Optional tokenizer artifact path
        #[arg(long)]
        tokenizer: Option<std::path::PathBuf>,

        /// Requested backend identity
        #[arg(long, default_value = "cpu")]
        backend: String,

        /// Require strict, no-fallback validation semantics
        #[arg(long, default_value_t = false)]
        strict: bool,

        /// Maximum tokens intended for the eventual validation run
        #[arg(long, visible_aliases = ["max-tokens", "n-predict"], default_value_t = 1)]
        max_tokens: usize,

        /// Same-machine platform artifact to cross-link
        #[arg(long)]
        platform_artifact: Option<std::path::PathBuf>,

        /// Output JSON validation receipt to file
        #[arg(long)]
        json_out: Option<std::path::PathBuf>,
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

#[cfg(windows)]
fn main() -> Result<()> {
    // The generated clap command tree is deep enough to overflow the default
    // Windows main-thread stack before subcommands such as `answer-parity`
    // can print help. Run the CLI body on an explicitly larger stack.
    let stack_size = std::env::var("BITNET_CLI_STACK_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64 * 1024 * 1024);

    std::thread::Builder::new()
        .name("bitnet-cli-main".to_string())
        .stack_size(stack_size)
        .spawn(run_main)?
        .join()
        .map_err(|panic| {
            if let Some(message) = panic.downcast_ref::<&str>() {
                anyhow::anyhow!("bitnet CLI worker thread panicked: {message}")
            } else if let Some(message) = panic.downcast_ref::<String>() {
                anyhow::anyhow!("bitnet CLI worker thread panicked: {message}")
            } else {
                anyhow::anyhow!("bitnet CLI worker thread panicked")
            }
        })?
}

#[cfg(not(windows))]
fn main() -> Result<()> {
    run_main()
}

fn run_main() -> Result<()> {
    tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(async_main())
}

async fn async_main() -> Result<()> {
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
            Err(e) if strict_mode => {
                let message = backend_selection_error_message_with_note(
                    &requested_backend_label,
                    &e.to_string(),
                );
                return Err(anyhow::anyhow!(message));
            }
            Err(e) => {
                let message = backend_selection_error_message_with_note(
                    &requested_backend_label,
                    &e.to_string(),
                );
                warn!(error = %message, "backend selection warning");
            }
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
            profile_id,
            allocation_audit,
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
                profile_id,
                allocation_audit,
            )
            .await
        }
        Some(Commands::Ask {
            model,
            tokenizer,
            question,
            system_prompt,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            strict_cuda,
            strict_cpu,
            receipt_out,
        }) => {
            run_ask_generation(
                &requested_backend_label,
                model,
                tokenizer,
                question,
                system_prompt,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                strict_cuda,
                strict_cpu,
                receipt_out,
            )
            .await
        }
        #[cfg(feature = "full-cli")]
        Some(Commands::Inference(cmd)) => (*cmd).execute(&config).await,
        #[cfg(feature = "full-cli")]
        Some(Commands::Chat(cmd)) => (*cmd).run_chat(&config).await,
        #[cfg(feature = "full-cli")]
        Some(Commands::AnswerCorpus(cmd)) => (*cmd).execute(&requested_backend_label).await,
        #[cfg(feature = "full-cli")]
        Some(Commands::AnswerParity(cmd)) => (*cmd).execute().await,
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
        Some(Commands::LunarLakeProbe { json_out }) => {
            handle_lunar_lake_probe_command(json_out).await
        }
        Some(Commands::IntelNpuProbe { strict, json_out }) => {
            intel_npu::handle_probe_command(strict, json_out).await
        }
        Some(Commands::IntelNpuSmoke { strict, json_out }) => {
            intel_npu::handle_smoke_command(strict, json_out).await
        }
        Some(Commands::IntelNpuBitnetSubgraph { strict, json_out }) => {
            intel_npu::handle_bitnet_subgraph_command(strict, json_out).await
        }
        Some(Commands::IntelArc140vOpenvinoGpuSmoke { strict, json_out }) => {
            intel_arc::handle_openvino_gpu_smoke_command(strict, json_out).await
        }
        Some(Commands::IntelArc140vOpenclSmoke { strict, json_out }) => {
            intel_arc::handle_opencl_smoke_command(strict, json_out).await
        }
        Some(Commands::Validate { action }) => handle_validate_command(action).await,
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

fn build_lunar_lake_probe_receipt(
    probe: bitnet_device_probe::Lnl258vPlatformProbe,
    timestamp_utc: String,
    artifact_path: Option<String>,
) -> serde_json::Value {
    serde_json::json!({
        "schema": 1,
        "artifact_kind": "lnl258v_platform_probe",
        "machine_id": probe.machine_id.clone(),
        "hardware_lane": "core-ultra-7-258v",
        "proof_stage": probe.proof_stage.clone(),
        "timestamp_utc": timestamp_utc,
        "requested_backend": "core-ultra-7-258v",
        "selected_backend": "core-ultra-7-258v",
        "runtime_api": "platform_probe",
        "fallback_used": probe.fallback_used,
        "fallback_backend": null,
        "fallback_reason": null,
        "platform": probe,
        "kernel_execution": false,
        "graph_execution": false,
        "bitnet_inference": false,
        "claim": "lunar_lake_runtime_visibility_recorded",
        "must_not_claim": [
            "BitNet inference works on 258V",
            "Arc 140V execution works",
            "Intel NPU execution works",
            "NPU accelerates BitNet"
        ],
        "artifact_path": artifact_path,
    })
}

async fn handle_lunar_lake_probe_command(json_out: Option<std::path::PathBuf>) -> Result<()> {
    let probe = bitnet_device_probe::probe_lnl258v_platform();
    let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let artifact_path = json_out.as_ref().map(|path| path.display().to_string());
    let receipt = build_lunar_lake_probe_receipt(probe, timestamp_utc, artifact_path);

    write_json_output(json_out.as_ref(), &receipt)?;

    Ok(())
}

#[derive(Debug, Clone)]
struct CpuBitnetValidationPreflight {
    status: &'static str,
    proof_stage: &'static str,
    validation_attempted: bool,
    blocked_before_inference: bool,
    blocker_stage: Option<&'static str>,
    blocker_reason: Option<String>,
    tokenizer_source: Option<String>,
    tokenizer_path: Option<std::path::PathBuf>,
    model_sha256: Option<String>,
}

fn sibling_tokenizer_path(model_path: &std::path::Path) -> Option<std::path::PathBuf> {
    let parent = model_path.parent()?;
    for name in ["tokenizer.json", "tokenizer.model"] {
        let candidate = parent.join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn cpu_bitnet_validation_preflight(
    model_path: &std::path::Path,
    tokenizer_path: Option<&std::path::Path>,
    backend: &str,
    strict: bool,
) -> CpuBitnetValidationPreflight {
    if backend != "cpu" && backend != "intel-258v-cpu-avx2" {
        return CpuBitnetValidationPreflight {
            status: "blocked_wrong_backend",
            proof_stage: "blocked_preflight",
            validation_attempted: false,
            blocked_before_inference: true,
            blocker_stage: Some("backend_selection"),
            blocker_reason: Some(format!(
                "CPU258V validation is CPU-only; requested backend was {backend:?}"
            )),
            tokenizer_source: None,
            tokenizer_path: None,
            model_sha256: None,
        };
    }

    if !model_path.exists() {
        return CpuBitnetValidationPreflight {
            status: "blocked_missing_canonical_model",
            proof_stage: "blocked_preflight",
            validation_attempted: false,
            blocked_before_inference: true,
            blocker_stage: Some("load_model"),
            blocker_reason: Some(format!("model path does not exist: {}", model_path.display())),
            tokenizer_source: None,
            tokenizer_path: None,
            model_sha256: None,
        };
    }

    let resolved_tokenizer = if let Some(path) = tokenizer_path {
        if path.exists() { Some(("explicit".to_string(), path.to_path_buf())) } else { None }
    } else {
        sibling_tokenizer_path(model_path).map(|path| {
            let source =
                if path.file_name().and_then(|name| name.to_str()) == Some("tokenizer.json") {
                    "sibling_tokenizer_json"
                } else {
                    "sibling_sentencepiece"
                };
            (source.to_string(), path)
        })
    };

    let Some((tokenizer_source, tokenizer_path)) = resolved_tokenizer else {
        return CpuBitnetValidationPreflight {
            status: "blocked_missing_tokenizer",
            proof_stage: "blocked_preflight",
            validation_attempted: false,
            blocked_before_inference: true,
            blocker_stage: Some("tokenize_prompt"),
            blocker_reason: Some(if strict {
                "strict validation requires an explicit tokenizer or sibling tokenizer asset"
                    .to_string()
            } else {
                "tokenizer artifact was not found; no fallback tokenizer is used by CPU258V validation"
                    .to_string()
            }),
            tokenizer_source: None,
            tokenizer_path: None,
            model_sha256: None,
        };
    };

    let model_sha256 = compute_model_sha256(model_path).ok();

    CpuBitnetValidationPreflight {
        status: "preflight_ready",
        proof_stage: "runtime_detected",
        validation_attempted: false,
        blocked_before_inference: false,
        blocker_stage: None,
        blocker_reason: None,
        tokenizer_source: Some(tokenizer_source),
        tokenizer_path: Some(tokenizer_path),
        model_sha256,
    }
}

struct CpuBitnetValidationReceiptInput {
    machine: String,
    model: std::path::PathBuf,
    tokenizer: Option<std::path::PathBuf>,
    backend: String,
    strict: bool,
    max_tokens: usize,
    platform_artifact: Option<std::path::PathBuf>,
    json_out: Option<std::path::PathBuf>,
    timestamp_utc: String,
}

fn build_cpu_bitnet_validation_receipt(
    input: CpuBitnetValidationReceiptInput,
) -> serde_json::Value {
    let platform = bitnet_device_probe::probe_lnl258v_platform();
    let preflight = cpu_bitnet_validation_preflight(
        &input.model,
        input.tokenizer.as_deref(),
        &input.backend,
        input.strict,
    );
    let cpu_features = detected_cpu_feature_labels();
    let thread_count = platform.cpu.threads.max(1);
    let artifact_path = input.json_out.as_ref().map(|path| path.display().to_string());
    let platform_artifact = input.platform_artifact.as_ref().map(|path| path.display().to_string());
    let tokenizer_path = preflight.tokenizer_path.as_ref().map(|path| path.display().to_string());
    let blocker = match (preflight.blocker_stage, preflight.blocker_reason.as_ref()) {
        (Some(stage), Some(reason)) => serde_json::json!({
            "stage": stage,
            "reason": reason,
        }),
        _ => serde_json::Value::Null,
    };

    serde_json::json!({
        "schema": 1,
        "artifact_kind": "cpu-bitnet-validation",
        "machine_id": input.machine,
        "hardware_lane": "intel-258v-cpu-avx2",
        "timestamp_utc": input.timestamp_utc,
        "proof_stage": preflight.proof_stage,
        "status": preflight.status,
        "validation_attempted": preflight.validation_attempted,
        "blocked_before_inference": preflight.blocked_before_inference,
        "blocker": blocker,
        "strict": input.strict,
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "kernel_execution": false,
        "bitnet_inference": false,
        "hardware": {
            "requested_backend": input.backend,
            "selected_backend": "intel-258v-cpu-avx2",
            "runtime_api": "cpu",
            "cpu": {
                "model": platform.cpu.brand,
                "cores": platform.cpu.cores,
                "threads": platform.cpu.threads,
                "p_core_count": platform.cpu.p_core_count,
                "lp_e_core_count": platform.cpu.lp_e_core_count,
                "avx2_detected": platform.cpu.has_avx2,
                "avx512_detected": platform.cpu.has_avx512,
                "fma_detected": platform.cpu.has_fma,
                "sse42_detected": platform.cpu.has_sse42,
                "features": cpu_features,
            },
            "platform_artifact": platform_artifact,
        },
        "model": {
            "expected_repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
            "expected_file": "ggml-model-i2_s.gguf",
            "path": input.model.display().to_string(),
            "exists": input.model.exists(),
            "sha256": preflight.model_sha256,
            "format": "gguf",
            "architecture": "bitnet_b1_58",
            "context_length": 4096,
            "tokenizer": "llama3",
            "vocab_size": 128256,
            "loader_mode": null,
        },
        "tokenizer": {
            "path": tokenizer_path,
            "source": preflight.tokenizer_source,
            "strict": input.strict,
        },
        "bitnet": {
            "weight_quantization": "W1.58",
            "activation_quantization": "A8",
            "weight_domain": "ternary",
            "kernel_family": "i2_s|tl2|qk256",
            "layout": null,
            "layout_source": null,
            "fallback_layout": null,
        },
        "execution": {
            "phase": "load_model",
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "max_tokens_requested": input.max_tokens,
            "batch_size": 1,
            "thread_count": thread_count,
            "requested_backend": "intel-258v-cpu-avx2",
            "selected_backend": "intel-258v-cpu-avx2",
            "requested_kernel": null,
            "selected_kernel": null,
            "fallback_used": false,
            "fallback_reason": null,
        },
        "claim_allowed": if preflight.status == "preflight_ready" {
            "The 258V CPU lane preflight found the requested model and tokenizer artifacts; no inference was run."
        } else {
            "The 258V CPU lane emitted a structured validation blocker before inference."
        },
        "claims_not_allowed": [
            "Strict BitNet GGUF loaded on 258V",
            "Tokenizer authority resolved through the inference path on 258V",
            "QK256 or TL2 execution ran on 258V",
            "BitNet inference works on 258V",
            "258V CPU benchmark performance"
        ],
        "artifact_path": artifact_path,
    })
}

async fn handle_validate_command(action: ValidateAction) -> Result<()> {
    match action {
        ValidateAction::CpuBitnet {
            machine,
            model,
            tokenizer,
            backend,
            strict,
            max_tokens,
            platform_artifact,
            json_out,
        } => {
            let timestamp_utc =
                chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
            let receipt = build_cpu_bitnet_validation_receipt(CpuBitnetValidationReceiptInput {
                machine,
                model,
                tokenizer,
                backend,
                strict,
                max_tokens,
                platform_artifact,
                json_out: json_out.clone(),
                timestamp_utc,
            });
            write_json_output(json_out.as_ref(), &receipt)?;
            Ok(())
        }
    }
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
        Err(err) if strict_backend => {
            let message =
                backend_selection_error_message_with_note(&request.to_string(), &err.to_string());
            Err(anyhow::anyhow!(message))
        }
        Err(err) => Ok(RunBackendIdentity {
            requested_backend: request.to_string(),
            selected_backend: "cpu".to_string(),
            runtime_api: "cpu".to_string(),
            fallback_used: request != BackendRequest::Auto && request != BackendRequest::Cpu,
            fallback_reason: Some(backend_selection_error_message_with_note(
                &request.to_string(),
                &err.to_string(),
            )),
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
    if std::env::var("BITNET_FORCE_SCALAR").as_deref() == Ok("1")
        || std::env::var("BITNET_CPU_KERNEL").as_deref() == Ok("scalar")
    {
        return "scalar";
    }
    if std::env::var("BITNET_CPU_KERNEL").as_deref() == Ok("avx2")
        && bitnet_common::runtime_diag::CpuFeatures::detect().avx2
    {
        return "avx2";
    }
    if std::env::var("BITNET_CPU_KERNEL").as_deref() == Ok("avx512")
        && bitnet_common::runtime_diag::CpuFeatures::detect().avx512f
    {
        return "avx512";
    }
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

fn detected_cpu_model_label() -> String {
    #[cfg(target_os = "windows")]
    {
        std::env::var("PROCESSOR_IDENTIFIER")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "unknown-windows-cpu".to_string())
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|cpuinfo| {
                cpuinfo.lines().find_map(|line| {
                    line.strip_prefix("model name")
                        .and_then(|rest| {
                            rest.split_once(':').map(|(_, value)| value.trim().to_string())
                        })
                        .filter(|value| !value.is_empty())
                })
            })
            .unwrap_or_else(|| "unknown-cpu".to_string())
    }
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
    } else if normalized.contains("qwen3-0.6b") {
        "Qwen/Qwen3-0.6B-GGUF".to_string()
    } else if normalized.contains("qwen2.5-0.5b") || normalized.contains("qwen2_5_0_5b") {
        "Qwen/Qwen2.5-0.5B-Instruct".to_string()
    } else {
        "local".to_string()
    }
}

fn infer_model_architecture(path: &std::path::Path) -> String {
    let normalized = path.to_string_lossy().to_ascii_lowercase();
    if infer_model_repo(path) == "microsoft/bitnet-b1.58-2B-4T-gguf" {
        "bitnet_b1_58".to_string()
    } else if normalized.contains("qwen3") {
        "qwen3".to_string()
    } else if normalized.contains("qwen2.5") || normalized.contains("qwen2_5") {
        "qwen2".to_string()
    } else {
        "unknown".to_string()
    }
}

fn receipt_model_family(model_architecture: &str) -> &'static str {
    match model_architecture {
        "bitnet_b1_58" => "bitnet",
        "qwen2" | "qwen3" => "qwen",
        _ => "unknown",
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

fn tokenizer_type_for_receipt(
    tokenizer_label: &str,
    source: bitnet_tokenizers::auto::TokenizerSource,
) -> String {
    if tokenizer_label == source.as_str() {
        match source {
            bitnet_tokenizers::auto::TokenizerSource::Explicit
            | bitnet_tokenizers::auto::TokenizerSource::Sibling => {
                "external_tokenizer_file".to_string()
            }
            bitnet_tokenizers::auto::TokenizerSource::GgufMetadata => "gguf_metadata".to_string(),
            bitnet_tokenizers::auto::TokenizerSource::CompatibilityFallback => {
                "compatibility_fallback".to_string()
            }
        }
    } else {
        tokenizer_label.to_string()
    }
}

fn gguf_header_counts_for_receipt(
    path: &std::path::Path,
    is_hf_directory: bool,
) -> Option<(usize, usize)> {
    if is_hf_directory || path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
        return None;
    }

    let header = bitnet_inference::gguf::read_header_blocking(path).ok()?;
    Some((usize::try_from(header.n_kv).ok()?, usize::try_from(header.n_tensors).ok()?))
}

fn compute_model_sha256(path: &std::path::Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut file =
        std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn greedy_top1_token_id(logits: &[f32]) -> Option<u32> {
    logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(left_id, left), (right_id, right)| {
            left.partial_cmp(right)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right_id.cmp(left_id))
        })
        .map(|(token_id, _)| token_id as u32)
}

fn qwen_trace_path() -> Option<std::path::PathBuf> {
    std::env::var("BITNET_QWEN_TRACE_JSONL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(std::path::PathBuf::from)
}

fn qwen_trace_enabled() -> bool {
    qwen_trace_path().is_some() || std::env::var("BITNET_QWEN_TRACE").as_deref() == Ok("1")
}

fn qwen_trace_reset_file() -> Result<()> {
    let Some(path) = qwen_trace_path() else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }
    std::fs::write(&path, b"")
        .with_context(|| format!("Failed to reset Qwen trace {}", path.display()))?;
    Ok(())
}

fn qwen_trace_write(value: serde_json::Value) -> Result<()> {
    if !qwen_trace_enabled() {
        return Ok(());
    }
    let line = serde_json::to_string(&value)?;
    if let Some(path) = qwen_trace_path() {
        use std::io::Write as _;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("Failed to open Qwen trace {}", path.display()))?;
        writeln!(file, "{line}")
            .with_context(|| format!("Failed to append Qwen trace {}", path.display()))?;
    } else {
        eprintln!("{line}");
    }
    Ok(())
}

fn qwen_trace_number(value: f64) -> serde_json::Value {
    if value.is_finite() { serde_json::json!(value) } else { serde_json::Value::Null }
}

fn qwen_trace_tensor(
    stage: &str,
    step: Option<usize>,
    tensor: &bitnet_common::ConcreteTensor,
) -> Result<()> {
    if !qwen_trace_enabled() {
        return Ok(());
    }
    let values = tensor_to_vec(tensor)?;
    let mut finite_count = 0usize;
    let mut nonfinite_count = 0usize;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut checksum = 0.0f64;
    for (idx, value) in values.iter().enumerate() {
        let value = *value as f64;
        if value.is_finite() {
            finite_count += 1;
            sum += value;
            sum_sq += value * value;
            min = min.min(value);
            max = max.max(value);
            if idx < 4096 {
                checksum += value * ((idx % 257) + 1) as f64;
            }
        } else {
            nonfinite_count += 1;
        }
    }
    let denom = finite_count.max(1) as f64;
    qwen_trace_write(serde_json::json!({
        "kind": "qwen_trace_tensor",
        "stage": stage,
        "step": step,
        "dims": tensor.shape(),
        "len": values.len(),
        "finite": finite_count,
        "nonfinite": nonfinite_count,
        "mean": qwen_trace_number(sum / denom),
        "rms": qwen_trace_number((sum_sq / denom).sqrt()),
        "min": qwen_trace_number(min),
        "max": qwen_trace_number(max),
        "checksum": qwen_trace_number(checksum),
        "sample": values
            .iter()
            .take(8)
            .map(|value| qwen_trace_number(*value as f64))
            .collect::<Vec<_>>(),
    }))
}

fn qwen_trace_top_logits_stage(
    stage: &str,
    step: Option<usize>,
    logits_vec: &[f32],
    chosen_id: Option<u32>,
) -> Result<()> {
    if !qwen_trace_enabled() {
        return Ok(());
    }
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| match (a.1.is_finite(), b.1.is_finite()) {
        (false, true) => std::cmp::Ordering::Greater,
        (true, false) => std::cmp::Ordering::Less,
        _ => {
            let cmp = b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal);
            if cmp == std::cmp::Ordering::Equal { a.0.cmp(&b.0) } else { cmp }
        }
    });
    let top_logits = indexed
        .into_iter()
        .take(20)
        .map(|(token_id, logit)| {
            serde_json::json!({
                "token_id": token_id,
                "logit": qwen_trace_number(logit as f64),
            })
        })
        .collect::<Vec<_>>();
    qwen_trace_write(serde_json::json!({
        "kind": "qwen_trace_logits",
        "stage": stage,
        "step": step,
        "chosen_id": chosen_id,
        "top_logits": top_logits,
    }))
}

fn qwen_trace_top_logits(step: usize, logits_vec: &[f32], chosen_id: Option<u32>) -> Result<()> {
    qwen_trace_top_logits_stage("lm_head.top_logits", Some(step), logits_vec, chosen_id)
}

fn qwen_trace_full_prompt_enabled() -> bool {
    qwen_trace_enabled() && std::env::var("BITNET_QWEN_TRACE_FULL_PROMPT").as_deref() == Ok("1")
}

fn qwen_trace_prompt_id_override() -> Result<Option<Vec<u32>>> {
    let Ok(raw) = std::env::var("BITNET_QWEN_TRACE_PROMPT_IDS") else {
        return Ok(None);
    };
    if !qwen_trace_enabled() {
        anyhow::bail!(
            "BITNET_QWEN_TRACE_PROMPT_IDS requires BITNET_QWEN_TRACE_JSONL or BITNET_QWEN_TRACE=1"
        );
    }
    let mut ids = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        ids.push(trimmed.parse::<u32>().with_context(|| {
            format!("invalid token id in BITNET_QWEN_TRACE_PROMPT_IDS: {trimmed}")
        })?);
    }
    if ids.is_empty() {
        anyhow::bail!("BITNET_QWEN_TRACE_PROMPT_IDS did not contain any token ids");
    }
    Ok(Some(ids))
}

fn nvidia_smi_memory_used_bytes(device_index: Option<usize>) -> Option<u64> {
    let mut command = std::process::Command::new("nvidia-smi");
    let index_arg;
    if let Some(index) = device_index {
        index_arg = index.to_string();
        command.args(["-i", index_arg.as_str()]);
    }
    let output =
        command.args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).output().ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    nvidia_smi_memory_used_bytes_from_csv(&stdout)
}

fn nvidia_smi_memory_used_bytes_from_csv(stdout: &str) -> Option<u64> {
    let first = stdout.lines().map(str::trim).find(|line| !line.is_empty())?;
    let mib_text = first.split_whitespace().next()?;
    let mib = mib_text.parse::<u64>().ok()?;
    mib.checked_mul(1024 * 1024)
}

#[derive(Debug, Clone)]
struct StrictReferenceReceipt {
    artifact_path: String,
    generated_token_id: Option<u32>,
    top1_token_id: Option<u32>,
}

fn strict_reference_receipt_path(json_path: &std::path::Path) -> std::path::PathBuf {
    if let Ok(path) = std::env::var("BITNET_CPU_REFERENCE_RECEIPT") {
        return std::path::PathBuf::from(path);
    }

    json_path.with_file_name("strict-bitnet-cpu-reference.json")
}

fn read_strict_reference_receipt(
    reference_path: &std::path::Path,
) -> Result<Option<StrictReferenceReceipt>> {
    if !reference_path.exists() {
        return Ok(None);
    }

    let json = std::fs::read_to_string(reference_path)
        .with_context(|| format!("Failed to read {}", reference_path.display()))?;
    let receipt: serde_json::Value = serde_json::from_str(&json)
        .with_context(|| format!("Failed to parse {}", reference_path.display()))?;
    let generated_token_id = receipt
        .pointer("/tokens/ids/0")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok());
    let top1_token_id = receipt
        .pointer("/logits_dump/0/top_logits/0/token_id")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok());
    let artifact_path = receipt
        .get("artifact_path")
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| reference_path.display().to_string());

    Ok(Some(StrictReferenceReceipt { artifact_path, generated_token_id, top1_token_id }))
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
    profile_id: Option<String>,
    allocation_audit: bool,
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
    bitnet_qk256_dispatch::reset_qk256_dispatch_coverage();
    let strict_cuda_backend_selected = strict_backend
        && backend_identity.selected_backend.as_str() == "nvidia-rtx-5070-ti-cuda"
        && backend_identity.runtime_api.as_str() == "cuda"
        && !backend_identity.fallback_used;
    let cuda_memory_before_bytes =
        strict_cuda_backend_selected.then(|| nvidia_smi_memory_used_bytes(Some(0))).flatten();
    unsafe {
        std::env::set_var("BITNET_REQUESTED_BACKEND", backend_identity.requested_backend.as_str());
        std::env::set_var("BITNET_SELECTED_BACKEND", backend_identity.selected_backend.as_str());
        std::env::set_var("BITNET_RUNTIME_API", backend_identity.runtime_api.as_str());
        if strict_cuda_backend_selected {
            std::env::set_var("BITNET_STRICT_CUDA_BACKEND", "1");
        } else {
            std::env::remove_var("BITNET_STRICT_CUDA_BACKEND");
        }
    }

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
    if qwen_trace_enabled() {
        qwen_trace_reset_file()?;
        unsafe {
            std::env::remove_var("BITNET_QWEN_TRACE_ACTIVE");
            std::env::remove_var("BITNET_QWEN_TRACE_STEP");
        }
        qwen_trace_write(serde_json::json!({
            "kind": "qwen_trace_event",
            "stage": "trace_start",
            "model_path": model_path.display().to_string(),
            "requested_backend": requested_backend_label,
            "prompt_template": prompt_template.clone(),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "greedy": greedy,
            "deterministic": deterministic,
        }))?;
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
    let model_load_start = std::time::Instant::now();

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
    let model_load_ms = elapsed_ms(model_load_start);

    // Load tokenizer with deterministic CPU-BITNET authority.
    // Priority: explicit path -> GGUF metadata -> sibling tokenizer asset.

    // Track GGUF header counts for JSON output independently of tokenizer source.
    let gguf_metadata = gguf_header_counts_for_receipt(&model_path, is_hf_directory);
    let effective_strict_tokenizer = strict_tokenizer || strict_loader;

    let tokenizer_load_start = std::time::Instant::now();
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
    let tokenizer_load_ms = elapsed_ms(tokenizer_load_start);
    let tokenizer_source = tokenizer_resolution.source;
    let tokenizer_strict = tokenizer_resolution.strict;
    let tokenizer: std::sync::Arc<dyn Tokenizer + Send + Sync> = tokenizer_resolution.tokenizer;

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
    let prompt_tokenize_start = std::time::Instant::now();
    let mut tokens = tokenizer.encode(&formatted_prompt, bos_policy, parse_special)?;
    if let Some(override_ids) = qwen_trace_prompt_id_override()? {
        qwen_trace_write(serde_json::json!({
            "kind": "qwen_trace_event",
            "stage": "prompt.ids_override",
            "original_prompt_ids": tokens.clone(),
            "override_prompt_ids": override_ids.clone(),
        }))?;
        tokens = override_ids;
    }
    ensure_non_empty_generation_context(&mut tokens, tokenizer.as_ref())?;
    let prompt_tokenize_ms = elapsed_ms(prompt_tokenize_start);
    println!("Input tokens ({}): {:?}", tokens.len(), &tokens[..10.min(tokens.len())]);
    qwen_trace_write(serde_json::json!({
        "kind": "qwen_trace_prompt",
        "stage": "prompt.ids",
        "template": template_type.to_string(),
        "bos_policy": bos_policy,
        "parse_special": parse_special,
        "formatted_prompt": formatted_prompt.clone(),
        "prompt_ids": tokens.clone(),
    }))?;

    if qwen_trace_full_prompt_enabled() {
        unsafe {
            std::env::set_var("BITNET_QWEN_TRACE_ACTIVE", "1");
            std::env::set_var("BITNET_QWEN_TRACE_STEP", "-1");
        }
        let full_prompt_result: Result<()> = (|| {
            let full_x = model.embed(&tokens)?;
            qwen_trace_tensor("full_prompt.input_embedding", None, &full_x)?;
            let mut no_cache: Box<dyn std::any::Any> = Box::new(());
            let full_h = model.forward(&full_x, no_cache.as_mut())?;
            qwen_trace_tensor("full_prompt.forward_output", None, &full_h)?;
            let full_last_hidden = extract_last_token_hidden(&full_h)?;
            qwen_trace_tensor("full_prompt.last_hidden", None, &full_last_hidden)?;
            let full_logits = model.logits(&full_last_hidden)?;
            let full_logits_vec = extract_logits_2d(&full_logits)?;
            let full_top1 = greedy_top1_token_id(&full_logits_vec);
            qwen_trace_top_logits_stage(
                "full_prompt.lm_head.top_logits",
                None,
                &full_logits_vec,
                full_top1,
            )?;
            qwen_trace_write(serde_json::json!({
                "kind": "qwen_trace_event",
                "stage": "full_prompt.first_generated_token",
                "token_id": full_top1,
            }))?;
            Ok(())
        })();
        unsafe {
            std::env::remove_var("BITNET_QWEN_TRACE_ACTIVE");
            std::env::remove_var("BITNET_QWEN_TRACE_STEP");
        }
        full_prompt_result?;
    }

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
    let mut first_token_decode_ms: Option<f64> = None;

    // Track generated tokens for repetition penalty
    let mut generated_tokens = Vec::new();

    // Always prefill the prompt prefix so generation conditions on the full prompt.
    // Profile requests additionally retain per-step timing/allocation details.
    let profile_requested = profile_id.is_some();
    if allocation_audit && !profile_requested {
        anyhow::bail!(
            "--allocation-audit requires --profile-id so allocation claims are receipt-scoped"
        );
    }
    if allocation_audit && json_out.is_none() {
        anyhow::bail!("--allocation-audit requires --json-out so allocation claims are durable");
    }
    if allocation_audit && !allocation_audit_backend_supported(&backend_identity) {
        anyhow::bail!(
            "--allocation-audit is currently scoped to --device apple-m4-cpu-neon with fallback_used=false; got requested_backend={}, selected_backend={}, runtime_api={}, fallback_used={}",
            backend_identity.requested_backend,
            backend_identity.selected_backend,
            backend_identity.runtime_api,
            backend_identity.fallback_used
        );
    }
    let allocation_audit_enabled = allocation_audit;
    let allocation_audit_guard = AllocationAuditGuard::enable(allocation_audit_enabled);
    let mut prefill_token_count = 0usize;
    let mut prefill_step_ms = Vec::new();
    let mut prefill_step_allocs = Vec::new();
    let prefill_start = std::time::Instant::now();
    if tokens.len() > 1 {
        for token in &tokens[..tokens.len() - 1] {
            let step_start = std::time::Instant::now();
            let step_alloc_start = AllocationAuditSnapshot::current();
            let x = model.embed(&[*token])?;
            let _ = model.forward(&x, any_cache.as_mut())?;
            let step_ms = elapsed_ms(step_start);
            if profile_requested {
                prefill_step_ms.push(step_ms);
            }
            if allocation_audit_enabled {
                prefill_step_allocs.push(AllocationAuditSnapshot::delta_since(step_alloc_start));
            }
            prefill_token_count += 1;
        }
    }
    let prefill_ms = if prefill_token_count > 0 { elapsed_ms(prefill_start) } else { 0.0 };

    // Track logits dump if requested
    let mut logits_dump: Vec<LogitStep> = Vec::new();
    let mut top1_tokens = Vec::new();

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
    let mut decode_step_ms = Vec::with_capacity(max_new_tokens);
    let mut embed_step_ms = Vec::with_capacity(max_new_tokens);
    let mut forward_step_ms = Vec::with_capacity(max_new_tokens);
    let mut logits_step_ms = Vec::with_capacity(max_new_tokens);
    let mut sample_step_ms = Vec::with_capacity(max_new_tokens);
    let mut token_decode_step_ms = Vec::with_capacity(max_new_tokens);
    let mut decode_step_allocs = Vec::with_capacity(max_new_tokens);
    let mut embed_step_allocs = Vec::with_capacity(max_new_tokens);
    let mut forward_step_allocs = Vec::with_capacity(max_new_tokens);
    let mut logits_step_allocs = Vec::with_capacity(max_new_tokens);
    let mut sample_step_allocs = Vec::with_capacity(max_new_tokens);
    let mut token_decode_step_allocs = Vec::with_capacity(max_new_tokens);
    for step_idx in 0..max_new_tokens {
        let qwen_trace_this_step = qwen_trace_enabled() && step_idx == 0;
        if qwen_trace_this_step {
            unsafe {
                std::env::set_var("BITNET_QWEN_TRACE_ACTIVE", "1");
                std::env::set_var("BITNET_QWEN_TRACE_STEP", step_idx.to_string());
            }
        }
        let decode_step_start = std::time::Instant::now();
        let decode_alloc_start = AllocationAuditSnapshot::current();
        // Embed only the LAST token (incremental)
        // KV cache already maintains historical context
        let last_token = tokens.last().copied().expect("tokens must be non-empty");

        let t0 = std::time::Instant::now();
        let embed_alloc_start = AllocationAuditSnapshot::current();
        let x = model.embed(&[last_token])?;
        if qwen_trace_this_step {
            qwen_trace_write(serde_json::json!({
                "kind": "qwen_trace_event",
                "stage": "decode.input_token",
                "step": step_idx,
                "token_id": last_token,
            }))?;
            qwen_trace_tensor("decode.input_embedding", Some(step_idx), &x)?;
        }
        let embed_ms = elapsed_ms(t0);
        if allocation_audit_enabled {
            embed_step_allocs.push(AllocationAuditSnapshot::delta_since(embed_alloc_start));
        }
        embed_step_ms.push(embed_ms);
        if timing_enabled {
            eprintln!("timing: embed_us={}", ms_to_us(embed_ms));
        }

        // Forward pass (with KV cache handling history)
        let t1 = std::time::Instant::now();
        let forward_alloc_start = AllocationAuditSnapshot::current();
        let h = model.forward(&x, any_cache.as_mut())?;
        if qwen_trace_this_step {
            qwen_trace_tensor("decode.forward_output", Some(step_idx), &h)?;
        }
        let forward_ms = elapsed_ms(t1);
        if allocation_audit_enabled {
            forward_step_allocs.push(AllocationAuditSnapshot::delta_since(forward_alloc_start));
        }
        forward_step_ms.push(forward_ms);
        if timing_enabled {
            eprintln!("timing: forward_us={}", ms_to_us(forward_ms));
        }

        // Extract last token hidden state first to avoid 3D├ù2D matmul issues
        let last_hidden = extract_last_token_hidden(&h)?;
        if qwen_trace_this_step {
            qwen_trace_tensor("decode.last_hidden", Some(step_idx), &last_hidden)?;
        }

        // Debug tap: hidden state RMS sanity (catches "everything is zero")
        if std::env::var("BITNET_DEBUG_LOGITS").as_deref() == Ok("1") && step_idx == 0 {
            let h_vec = tensor_to_vec(&last_hidden)?;
            let hidden_rms = compute_rms(&h_vec);
            eprintln!("hidden_rms={:.6}", hidden_rms);
        }

        // Get logits from last token hidden state
        let t2 = std::time::Instant::now();
        let logits_alloc_start = AllocationAuditSnapshot::current();
        let logits = model.logits(&last_hidden)?;
        let logits_ms = elapsed_ms(t2);
        logits_step_ms.push(logits_ms);
        if timing_enabled {
            eprintln!("timing: logits_us={}", ms_to_us(logits_ms));
        }

        // Extract logits vector with robust shape handling
        let logits_vec = extract_logits_2d(&logits)?;
        if allocation_audit_enabled {
            logits_step_allocs.push(AllocationAuditSnapshot::delta_since(logits_alloc_start));
        }
        let greedy_top1_token = greedy_top1_token_id(&logits_vec);
        if let Some(token_id) = greedy_top1_token {
            top1_tokens.push(token_id);
        }

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
        let t3 = std::time::Instant::now();
        let sample_alloc_start = AllocationAuditSnapshot::current();
        let next_token = sampler.sample(&logits_vec, &generated_tokens)?;
        if qwen_trace_this_step {
            qwen_trace_top_logits(step_idx, &logits_vec, Some(next_token))?;
            qwen_trace_write(serde_json::json!({
                "kind": "qwen_trace_event",
                "stage": "decode.first_generated_token",
                "step": step_idx,
                "token_id": next_token,
            }))?;
            unsafe {
                std::env::remove_var("BITNET_QWEN_TRACE_ACTIVE");
                std::env::remove_var("BITNET_QWEN_TRACE_STEP");
            }
        }
        let sample_ms = elapsed_ms(t3);
        if allocation_audit_enabled {
            sample_step_allocs.push(AllocationAuditSnapshot::delta_since(sample_alloc_start));
        }
        sample_step_ms.push(sample_ms);
        if timing_enabled {
            eprintln!("timing: sample_us={}", ms_to_us(sample_ms));
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
            let Some(best_i) = greedy_top1_token else {
                anyhow::bail!("No finite logits found for --assert-greedy at step {step_idx}");
            };
            if next_token != best_i {
                eprintln!("ERROR: Non-argmax token chosen in --greedy at step {}", step_idx);
                eprintln!("  argmax={} but chosen={}", best_i, next_token);
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
        let token_decode_start = std::time::Instant::now();
        let token_decode_alloc_start = AllocationAuditSnapshot::current();
        let token_text = tokenizer.decode(&[next_token])?;
        if allocation_audit_enabled {
            token_decode_step_allocs
                .push(AllocationAuditSnapshot::delta_since(token_decode_alloc_start));
        }
        token_decode_step_ms.push(elapsed_ms(token_decode_start));
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
        let step_ms = elapsed_ms(decode_step_start);
        if first_token_decode_ms.is_none() {
            first_token_decode_ms = Some(step_ms);
        }
        decode_step_ms.push(step_ms);
        if allocation_audit_enabled {
            decode_step_allocs.push(AllocationAuditSnapshot::delta_since(decode_alloc_start));
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
    drop(allocation_audit_guard);

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
        let tokenizer_label = infer_tokenizer_label(tokenizer.as_ref(), tokenizer_source);
        let tokenizer_type = tokenizer_type_for_receipt(&tokenizer_label, tokenizer_source);
        let tokenizer_info = serde_json::json!({
            "type": tokenizer_type,
            "model_family": tokenizer_type,
            "origin": if tokenizer_source == bitnet_tokenizers::auto::TokenizerSource::GgufMetadata {
                "embedded"
            } else {
                "external"
            },
            "source": tokenizer_source_str,
            "strict": tokenizer_strict,
            "pretokenizer_authority": "unknown",
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
        let model_family = receipt_model_family(&model_architecture);
        let model_format_label = receipt_model_format(&model_path, &model_format, is_hf_directory);
        let model_file =
            model_path.file_name().and_then(|name| name.to_str()).unwrap_or_default().to_string();
        let thread_count = effective_thread_count(threads);
        let cpu_features = detected_cpu_feature_labels();
        let cpu_model = detected_cpu_model_label();
        let fallback_reason = backend_identity.fallback_reason.clone();
        let requested_backend = backend_identity.requested_backend.as_str();
        let selected_backend = backend_identity.selected_backend.as_str();
        let runtime_api = backend_identity.runtime_api.as_str();
        let apple_machine = apple_machine_receipt_json(requested_backend, selected_backend);
        let bitnet_linear_coverage = bitnet_qk256_dispatch::qk256_dispatch_coverage();
        let strict_cuda_selected_artifact = strict_backend
            && canonical_bitnet_model
            && selected_backend == "nvidia-rtx-5070-ti-cuda"
            && runtime_api == "cuda"
            && loader_mode == bitnet_models::GgufLoaderMode::RealGguf.as_str()
            && !backend_identity.fallback_used;
        let strict_cuda_proof_artifact =
            strict_cuda_selected_artifact && generated_tokens.len() == 1;
        let strict_cuda_short_decode_artifact =
            strict_cuda_selected_artifact && generated_tokens.len() > 1;
        let cuda_generated_token_id = generated_tokens.first().copied();
        let cuda_top1_token_id = top1_tokens.first().copied();
        let cuda_kernel_invocations = bitnet_linear_coverage.bitnet_linear_layers_on_cuda;
        let cuda_weight_residency = if strict_cuda_selected_artifact {
            bitnet_qk256_dispatch::qk256_cuda_weight_residency()
        } else {
            None
        };
        let weights_uploaded_once = cuda_weight_residency
            .as_ref()
            .map(|residency| residency.weights_uploaded_once)
            .unwrap_or(false);
        let per_token_weight_upload = cuda_weight_residency
            .as_ref()
            .map(|residency| residency.per_token_weight_upload)
            .unwrap_or(strict_cuda_selected_artifact && cuda_kernel_invocations > 0);
        let cuda_memory_after_bytes =
            strict_cuda_selected_artifact.then(|| nvidia_smi_memory_used_bytes(Some(0))).flatten();
        let cuda_memory_hwm_bytes =
            cuda_memory_before_bytes.into_iter().chain(cuda_memory_after_bytes).max();
        let expected_reference_path =
            strict_cuda_proof_artifact.then(|| strict_reference_receipt_path(&json_path));
        let strict_reference_receipt = match expected_reference_path.as_deref() {
            Some(path) => read_strict_reference_receipt(path)?,
            None => None,
        };
        let reference_artifact_path = strict_reference_receipt
            .as_ref()
            .map(|receipt| receipt.artifact_path.clone())
            .or_else(|| expected_reference_path.as_ref().map(|path| path.display().to_string()));
        let cpu_greedy_token_id =
            strict_reference_receipt.as_ref().and_then(|receipt| receipt.generated_token_id);
        let cpu_top1_token_id =
            strict_reference_receipt.as_ref().and_then(|receipt| receipt.top1_token_id);
        let greedy_token_agreement = cpu_greedy_token_id
            .zip(cuda_generated_token_id)
            .map(|(cpu_token, cuda_token)| cpu_token == cuda_token);
        let top1_agreement = cpu_top1_token_id
            .zip(cuda_top1_token_id)
            .map(|(cpu_token, cuda_token)| cpu_token == cuda_token);
        let cuda_probe = if strict_cuda_proof_artifact || strict_cuda_short_decode_artifact {
            Some(bitnet_device_probe::probe_nvidia_cuda(Some(0)))
        } else {
            None
        };
        let strict_cpu_reference_artifact = strict_backend
            && canonical_bitnet_model
            && runtime_api == "cpu"
            && loader_mode == bitnet_models::GgufLoaderMode::RealGguf.as_str();
        let artifact_kind = if strict_cpu_reference_artifact && profile_requested {
            "strict_bitnet_cpu_profile"
        } else if strict_cpu_reference_artifact {
            "strict_bitnet_cpu_reference"
        } else if strict_cuda_proof_artifact {
            "strict_bitnet_cuda_proof"
        } else if strict_cuda_short_decode_artifact {
            "strict_bitnet_cuda_short_decode_proof"
        } else {
            "inference_result"
        };
        let steady_decode_step_ms = decode_step_ms.get(1..).unwrap_or(&[]);
        let steady_decode_step_allocs = decode_step_allocs.get(1..).unwrap_or(&[]);
        let decode_total_ms = decode_step_ms.iter().sum::<f64>();
        let sampling_ms_per_token = if sample_step_ms.is_empty() {
            None
        } else {
            Some(sample_step_ms.iter().sum::<f64>() / sample_step_ms.len() as f64)
        };
        let steady_decode_tps = steady_decode_tps_ms(&decode_step_ms);
        let steady_alloc_count_per_token = mean_alloc_count(steady_decode_step_allocs);
        let steady_alloc_bytes_per_token = mean_alloc_bytes(steady_decode_step_allocs);
        let profile_label = profile_id.as_deref().unwrap_or("default");
        let profile_claim_scope = profile_claim_scope(runtime_api, selected_backend);
        let profile_machine_context_recorded = profile_machine_context_recorded(
            runtime_api,
            selected_backend,
            apple_machine.is_some(),
            &cpu_features,
            !cpu_model.is_empty(),
        ) || cuda_probe.is_some();
        let profile_receipt = serde_json::json!({
            "id": profile_label,
            "requested": profile_requested,
            "kind": "steady_decode_prefill",
            "claim_scope": profile_claim_scope,
            "phase": "decode",
            "machine_context_recorded": profile_machine_context_recorded,
            "backend": {
                "requested_backend": requested_backend,
                "selected_backend": selected_backend,
                "runtime_api": runtime_api,
                "fallback_used": backend_identity.fallback_used,
                "fallback_reason": backend_identity.fallback_reason.as_deref(),
            },
            "prompt_prefill": {
                "exercised": prefill_token_count > 0,
                "tokens": prefill_token_count,
                "ms": rounded_ms(prefill_ms),
                "per_token_ms": timing_samples_json(&prefill_step_ms),
                "kv_cache_behavior": if prefill_token_count > 0 {
                    "prompt_prefix_prefilled_before_decode"
                } else if profile_requested {
                    "single_token_prompt_no_prefix_prefill"
                } else {
                    "not_requested"
                },
            },
            "decode": {
                "generated_tokens": generated_tokens.len(),
                "warmup_tokens": usize::from(!decode_step_ms.is_empty()),
                "steady_state_tokens": decode_step_ms.len().saturating_sub(1),
                "first_token_decode_ms": first_token_decode_ms.map(rounded_ms),
                "steady_state_tok_s": steady_decode_tps.map(|value| (value * 1000.0).round() / 1000.0),
                "per_token_ms": timing_samples_json(&decode_step_ms),
                "steady_per_token_ms": timing_samples_json(steady_decode_step_ms),
                "embed_ms": timing_samples_json(&embed_step_ms),
                "forward_ms": timing_samples_json(&forward_step_ms),
                "logits_ms": timing_samples_json(&logits_step_ms),
                "sample_ms": timing_samples_json(&sample_step_ms),
                "token_decode_ms": timing_samples_json(&token_decode_step_ms),
            },
            "allocation_audit": {
                "enabled": allocation_audit_enabled,
                "method": if allocation_audit_enabled {
                    "process_global_allocator_counter_delta"
                } else {
                    "not_requested"
                },
                "scope": if allocation_audit_enabled {
                    "selected Apple M4 CPU/NEON BitNet prompt-prefill and decode hot loop"
                } else {
                    "not_requested"
                },
                "claim_scope": if allocation_audit_enabled {
                    "allocation counter deltas for the selected Apple M4 CPU/NEON BitNet profile only"
                } else {
                    "not_requested"
                },
                "warmup_tokens": usize::from(!decode_step_allocs.is_empty()),
                "measured_tokens": decode_step_allocs.len().saturating_sub(1),
                "per_token_alloc_count_delta": allocation_count_delta_json(&decode_step_allocs),
                "per_token_alloc_bytes_delta": allocation_bytes_delta_json(&decode_step_allocs),
                "steady_state_alloc_count_per_token": steady_alloc_count_per_token.map(rounded_ms),
                "steady_state_alloc_bytes_per_token": steady_alloc_bytes_per_token.map(rounded_ms),
                "instrumentation_included": [
                    "prompt_prefill_step",
                    "decode_step_total",
                    "model.embed",
                    "model.forward",
                    "model.logits_and_extract",
                    "sampler.sample",
                    "tokenizer.decode",
                    "stdout_text_write",
                    "token_vector_updates",
                    "stop_tail_updates"
                ],
                "instrumentation_excluded": [
                    "model_load",
                    "tokenizer_load",
                    "prompt_tokenize",
                    "json_receipt_serialization",
                    "debug_logit_dump_topk_unless_enabled"
                ],
                "prompt_prefill": allocation_samples_json(&prefill_step_allocs),
                "decode": {
                    "total": allocation_samples_json(&decode_step_allocs),
                    "steady_state": allocation_samples_json(steady_decode_step_allocs),
                    "embed": allocation_samples_json(&embed_step_allocs),
                    "forward": allocation_samples_json(&forward_step_allocs),
                    "logits": allocation_samples_json(&logits_step_allocs),
                    "sample": allocation_samples_json(&sample_step_allocs),
                    "token_decode": allocation_samples_json(&token_decode_step_allocs),
                },
            },
            "model_load_ms": rounded_ms(model_load_ms),
            "tokenizer_load_ms": rounded_ms(tokenizer_load_ms),
            "prompt_tokenize_ms": rounded_ms(prompt_tokenize_ms),
        });
        let decode_steady_state_tok_s =
            steady_decode_tps.map(|value| (value * 1000.0).round() / 1000.0);
        let execution_phase =
            if strict_cuda_short_decode_artifact { "short_decode" } else { "decode" };
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
                "ids": generated_tokens.clone(),
                "prompt_ids": tokens[..prompt_tokens_len].to_vec(),
                "generated_ids": generated_tokens.clone(),
            },
            "latency": {
                "cmd_to_first_ms": first_token_ms,
                "decode_first_ms": first_token_ms,  // Same as cmd_to_first for now
                "total_ms": total_ms,
            },
            "timing": {
                "model_load_ms": rounded_ms(model_load_ms),
                "tokenizer_load_ms": rounded_ms(tokenizer_load_ms),
                "tokenize_ms": rounded_ms(prompt_tokenize_ms),
                "prefill_ms": rounded_ms(prefill_ms),
                "first_token_ms": first_token_ms,
                "first_token_decode_ms": first_token_decode_ms.map(rounded_ms),
                "decode_total_ms": rounded_ms(decode_total_ms),
                "decode_steady_state_tok_s": decode_steady_state_tok_s,
                "sampling_ms_per_token": sampling_ms_per_token.map(rounded_ms),
            },
            "throughput": {
                "tokens_per_second": tok_per_sec,
                "decoded_tokens": generated_tokens.len(),
            },
            "profile": profile_receipt,
            "model": {
                "repo": model_repo,
                "file": model_file,
                "path": model_path.display().to_string(),
                "sha256": model_sha256,
                "format": model_format_label,
                "family": model_family,
                "architecture": model_architecture,
                "context_length": config.model.max_position_embeddings,
                "tokenizer": tokenizer_label,
                "vocab_size": tokenizer.vocab_size(),
                "tie_word_embeddings": serde_json::Value::Null,
                "output_head_tensor": "output.weight",
                "loader_mode": loader_mode,
                "fallback_loader_used": loader_mode != bitnet_models::GgufLoaderMode::RealGguf.as_str(),
            },
            "bitnet": {
                "weight_quantization": if canonical_bitnet_model { "W1.58" } else { "unknown" },
                "activation_quantization": if canonical_bitnet_model { "A8" } else { "unknown" },
                "quantization": if canonical_bitnet_model { "W1.58A8" } else { "unknown" },
                "kernel_format": kernel_family,
                "kernel_family": kernel_family,
                "execution_phase": execution_phase,
                "layout_source": layout_source,
                "fallback_layout": serde_json::Value::Null,
                "weights_uploaded_once": weights_uploaded_once,
                "per_token_weight_upload": per_token_weight_upload,
            },
            "execution": {
                "phase": execution_phase,
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
            "execution_coverage": {
                "bitnet_linear_layers_total": bitnet_linear_coverage.bitnet_linear_layers_total,
                "bitnet_linear_layers_on_cuda": bitnet_linear_coverage.bitnet_linear_layers_on_cuda,
                "bitnet_linear_layers_cpu_fallback": bitnet_linear_coverage.bitnet_linear_layers_cpu_fallback,
                "unsupported_ops": bitnet_linear_coverage.unsupported_ops.clone(),
                "execution_claim": bitnet_linear_coverage.execution_claim,
            },
            "kernel": {
                "family": kernel_family,
                "implementation": kernel_implementation,
                "layout": kernel_layout,
                "dequantizes_before_compute": dequantizes_before_compute,
                "kernel_id": selected_kernel.as_str(),
            },
            "cpu": {
                "model": cpu_model.as_str(),
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
                "tokenizer_strict": tokenizer_strict,
                "model_family": model_family,
                "quant_format": format!("{}", config.quantization.quantization_type),
                "cpu_model": cpu_model.as_str(),
                "cpu_features": &cpu_features,
                "thread_count": thread_count,
                "fallback_used": backend_identity.fallback_used,
                "fallback_reason": backend_identity.fallback_reason.as_deref(),
                "prompt_tokens": prompt_tokens_len,
                "decode_tokens": generated_tokens.len(),
                "phase": execution_phase,
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
        if strict_cuda_proof_artifact && let Some(object) = output.as_object_mut() {
            object.insert("claim".to_string(), serde_json::json!("strict_bitnet_cuda_inference"));
            object.insert("speedup_claim".to_string(), serde_json::json!(false));
            object.insert(
                "reference_backend".to_string(),
                serde_json::json!("amd-9950x3d-cpu-avx512"),
            );
            object.insert("fallback_backend".to_string(), serde_json::Value::Null);
            if let Some(cuda_probe) = &cuda_probe {
                object.insert(
                    "cuda".to_string(),
                    serde_json::json!({
                        "available": cuda_probe.available,
                        "device_count": cuda_probe.device_count,
                        "device_index": cuda_probe.selected_device_index,
                        "device_name": cuda_probe.selected_device_name,
                        "compute_capability": cuda_probe.compute_capability,
                        "driver_version": cuda_probe.driver_version,
                        "cuda_runtime_version": cuda_probe.cuda_runtime_version,
                        "cuda_toolkit_version": cuda_probe.cuda_toolkit_version,
                        "nvrtc_version": cuda_probe.nvrtc_version,
                        "vram_bytes": cuda_probe.vram_bytes,
                        "cuda_kernel_invocations": cuda_kernel_invocations,
                    }),
                );
            }
            object.insert(
                "reference".to_string(),
                serde_json::json!({
                    "cpu_reference_artifact": reference_artifact_path,
                    "cuda_greedy_token_id": cuda_generated_token_id,
                    "cpu_greedy_token_id": cpu_greedy_token_id,
                    "greedy_token_agreement": greedy_token_agreement,
                    "cuda_top1_token_id": cuda_top1_token_id,
                    "cpu_top1_token_id": cpu_top1_token_id,
                    "top1_agreement": top1_agreement,
                    "max_abs_error": serde_json::Value::Null,
                    "mean_abs_error": serde_json::Value::Null,
                }),
            );
            object.insert(
                "kernel_stats".to_string(),
                serde_json::json!([{
                    "kernel_id": bitnet_kernels::cuda::CUDA_QK256_GEMV_KERNEL_ID,
                    "invocations": cuda_kernel_invocations,
                    "fallback_invocations": bitnet_linear_coverage.bitnet_linear_layers_cpu_fallback,
                    "host_to_device_bytes": serde_json::Value::Null,
                    "device_to_host_bytes": serde_json::Value::Null,
                    "kernel_launches": cuda_kernel_invocations,
                    "kernel_time_ms": serde_json::Value::Null,
                }]),
            );
            object.insert(
                "kernel".to_string(),
                serde_json::json!({
                    "family": "qk256",
                    "implementation": "cuda",
                    "layout": kernel_layout,
                    "dequantizes_before_compute": false,
                    "kernel_id": bitnet_kernels::cuda::CUDA_QK256_GEMV_KERNEL_ID,
                }),
            );
            if let Some(strict_provenance) =
                object.get_mut("strict_provenance").and_then(serde_json::Value::as_object_mut)
            {
                strict_provenance.insert(
                    "requested_kernel".to_string(),
                    serde_json::json!(bitnet_kernels::cuda::CUDA_QK256_GEMV_KERNEL_ID),
                );
                strict_provenance.insert(
                    "selected_kernel".to_string(),
                    serde_json::json!(bitnet_kernels::cuda::CUDA_QK256_GEMV_KERNEL_ID),
                );
                strict_provenance.insert(
                    "cuda_kernel_invocations".to_string(),
                    serde_json::json!(cuda_kernel_invocations),
                );
            }
            let cpu_fallback_ops = if bitnet_linear_coverage.bitnet_linear_layers_cpu_fallback == 0
            {
                Vec::<String>::new()
            } else {
                vec!["qk256_cpu_fallback".to_string()]
            };
            object.insert("cpu_fallback_ops".to_string(), serde_json::json!(cpu_fallback_ops));
        }
        if strict_cuda_short_decode_artifact && let Some(object) = output.as_object_mut() {
            object
                .insert("claim".to_string(), serde_json::json!("strict_bitnet_cuda_short_decode"));
            object.insert("speedup_claim".to_string(), serde_json::json!(false));
            object.insert(
                "reference_backend".to_string(),
                serde_json::json!("amd-9950x3d-cpu-avx512"),
            );
            object.insert("fallback_backend".to_string(), serde_json::Value::Null);
            object.insert("execution_phase".to_string(), serde_json::json!("short_decode"));
            object.insert("prompt_tokens".to_string(), serde_json::json!(prompt_tokens_len));
            object
                .insert("generated_tokens".to_string(), serde_json::json!(generated_tokens.len()));
            object.insert("prefill_ms".to_string(), serde_json::json!(first_token_ms.unwrap_or(0)));
            object
                .insert("prompt_prefill_ms".to_string(), serde_json::json!(rounded_ms(prefill_ms)));
            object.insert(
                "prefill_timing_source".to_string(),
                serde_json::json!("time_to_first_token_current_cli_path"),
            );
            object.insert("first_token_ms".to_string(), serde_json::json!(first_token_ms));
            object.insert(
                "decode_steady_state_tok_s".to_string(),
                serde_json::json!(decode_steady_state_tok_s),
            );
            object.insert(
                "cuda_kernel_invocations".to_string(),
                serde_json::json!(cuda_kernel_invocations),
            );
            object.insert(
                "cuda_memory_hwm_bytes".to_string(),
                serde_json::json!(cuda_memory_hwm_bytes),
            );
            object.insert(
                "cuda_memory_hwm_source".to_string(),
                serde_json::json!("nvidia-smi-memory.used-sampled"),
            );
            if let Some(cuda_probe) = &cuda_probe {
                object.insert(
                    "cuda".to_string(),
                    serde_json::json!({
                        "available": cuda_probe.available,
                        "device_count": cuda_probe.device_count,
                        "device_index": cuda_probe.selected_device_index,
                        "device_name": cuda_probe.selected_device_name,
                        "compute_capability": cuda_probe.compute_capability,
                        "driver_version": cuda_probe.driver_version,
                        "cuda_runtime_version": cuda_probe.cuda_runtime_version,
                        "cuda_toolkit_version": cuda_probe.cuda_toolkit_version,
                        "nvrtc_version": cuda_probe.nvrtc_version,
                        "vram_bytes": cuda_probe.vram_bytes,
                        "cuda_kernel_invocations": cuda_kernel_invocations,
                        "memory_used_before_bytes": cuda_memory_before_bytes,
                        "memory_used_after_bytes": cuda_memory_after_bytes,
                        "memory_hwm_bytes": cuda_memory_hwm_bytes,
                        "memory_hwm_source": "nvidia-smi-memory.used-sampled",
                    }),
                );
            }
            object.insert(
                "timing".to_string(),
                serde_json::json!({
                    "model_load_ms": rounded_ms(model_load_ms),
                    "tokenizer_load_ms": rounded_ms(tokenizer_load_ms),
                    "tokenize_ms": rounded_ms(prompt_tokenize_ms),
                    "prefill_ms": first_token_ms.unwrap_or(0),
                    "prompt_prefill_ms": rounded_ms(prefill_ms),
                    "prefill_timing_source": "time_to_first_token_current_cli_path",
                    "first_token_ms": first_token_ms,
                    "first_token_decode_ms": first_token_decode_ms.map(rounded_ms),
                    "decode_total_ms": rounded_ms(decode_total_ms),
                    "decode_steady_state_tok_s": decode_steady_state_tok_s,
                    "sampling_ms_per_token": sampling_ms_per_token.map(rounded_ms),
                    "decode_step_ms": timing_samples_json(&decode_step_ms),
                    "embed_ms": timing_samples_json(&embed_step_ms),
                    "forward_ms": timing_samples_json(&forward_step_ms),
                    "logits_ms": timing_samples_json(&logits_step_ms),
                    "sample_ms": timing_samples_json(&sample_step_ms),
                    "token_decode_ms": timing_samples_json(&token_decode_step_ms),
                    "total_ms": total_ms,
                }),
            );
            object.insert(
                "kv_cache".to_string(),
                serde_json::json!({
                    "enabled": true,
                    "mode": "incremental_decode",
                    "device": "cpu",
                    "batch_size": 1,
                    "prompt_tokens": prompt_tokens_len,
                    "generated_tokens": generated_tokens.len(),
                    "decode_steps": generated_tokens.len(),
                }),
            );
            object.insert(
                "kernel_stats".to_string(),
                serde_json::json!([{
                    "kernel_id": bitnet_kernels::cuda::CUDA_QK256_GEMV_KERNEL_ID,
                    "invocations": cuda_kernel_invocations,
                    "fallback_invocations": bitnet_linear_coverage.bitnet_linear_layers_cpu_fallback,
                    "host_to_device_bytes": serde_json::Value::Null,
                    "device_to_host_bytes": serde_json::Value::Null,
                    "kernel_launches": cuda_kernel_invocations,
                    "kernel_time_ms": serde_json::Value::Null,
                }]),
            );
            object.insert(
                "kernel".to_string(),
                serde_json::json!({
                    "family": "qk256",
                    "implementation": "cuda",
                    "layout": kernel_layout,
                    "dequantizes_before_compute": false,
                    "kernel_id": bitnet_kernels::cuda::CUDA_QK256_GEMV_KERNEL_ID,
                }),
            );
            if let Some(strict_provenance) =
                object.get_mut("strict_provenance").and_then(serde_json::Value::as_object_mut)
            {
                strict_provenance.insert(
                    "requested_kernel".to_string(),
                    serde_json::json!(bitnet_kernels::cuda::CUDA_QK256_GEMV_KERNEL_ID),
                );
                strict_provenance.insert(
                    "selected_kernel".to_string(),
                    serde_json::json!(bitnet_kernels::cuda::CUDA_QK256_GEMV_KERNEL_ID),
                );
                strict_provenance.insert(
                    "cuda_kernel_invocations".to_string(),
                    serde_json::json!(cuda_kernel_invocations),
                );
            }
            let cpu_fallback_ops = if bitnet_linear_coverage.bitnet_linear_layers_cpu_fallback == 0
            {
                Vec::<String>::new()
            } else {
                vec!["qk256_cpu_fallback".to_string()]
            };
            object.insert("cpu_fallback_ops".to_string(), serde_json::json!(cpu_fallback_ops));
        }
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

#[allow(clippy::too_many_arguments)]
async fn run_ask_generation(
    requested_backend_label: &str,
    model: std::path::PathBuf,
    tokenizer: Option<std::path::PathBuf>,
    question: String,
    system_prompt: Option<String>,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    strict_cuda: bool,
    strict_cpu: bool,
    receipt_out: Option<std::path::PathBuf>,
) -> Result<()> {
    const RTX_5070_TI_CUDA: &str = "nvidia-rtx-5070-ti-cuda";
    if strict_cuda && strict_cpu {
        anyhow::bail!("--strict-cuda and --strict-cpu are mutually exclusive");
    }
    if strict_cuda && requested_backend_label != RTX_5070_TI_CUDA {
        anyhow::bail!(
            "--strict-cuda requires --device {RTX_5070_TI_CUDA}; requested backend was {requested_backend_label}"
        );
    }
    if strict_cpu && requested_backend_label != "cpu" {
        anyhow::bail!(
            "--strict-cpu requires --device cpu; requested backend was {requested_backend_label}"
        );
    }
    if strict_cuda && let Some(cuda_bin) = ensure_strict_cuda_ask_runtime_libraries_visible()? {
        debug!(
            "added CUDA Toolkit bin directory to process PATH for strict CUDA ask: {}",
            cuda_bin.display()
        );
    }

    let question_for_receipt = question.clone();
    let system_prompt_for_receipt = system_prompt.clone();
    run_simple_generation(
        requested_backend_label,
        model,
        "auto".to_string(),
        None,
        tokenizer,
        question,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        1.1,
        None,
        false,
        false,
        true,
        true,
        receipt_out.clone(),
        false,
        false,
        true,
        true,
        0,
        "llama3-chat".to_string(),
        system_prompt,
        vec!["<|eot_id|>".to_string(), "<|end_of_text|>".to_string()],
        Vec::new(),
        None,
        10,
        false,
        false,
        Some("ask".to_string()),
        false,
    )
    .await?;

    let Some(receipt_path) = receipt_out else {
        return Ok(());
    };
    let run_receipt: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&receipt_path)
            .with_context(|| format!("failed to read run receipt {}", receipt_path.display()))?,
    )
    .with_context(|| format!("invalid run receipt {}", receipt_path.display()))?;

    if strict_cuda {
        validate_strict_cuda_ask_receipt(&run_receipt)?;
    }
    if strict_cpu {
        validate_strict_cpu_ask_receipt(&run_receipt)?;
    }

    let answer = run_receipt["text"].as_str().unwrap_or_default();
    let artifact_kind = if run_receipt["runtime_api"] == "cuda"
        && run_receipt["selected_backend"] == RTX_5070_TI_CUDA
    {
        "bitnet_cuda_answer"
    } else {
        "bitnet_cpu_answer"
    };
    let quality = answer_quality_receipt(answer, &run_receipt, max_new_tokens);
    let answer_receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": artifact_kind,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "question": question_for_receipt,
        "answer": answer,
        "model": {
            "repo": run_receipt["model"]["repo"].clone(),
            "file": run_receipt["model"]["file"].clone(),
            "path": run_receipt["model"]["path"].clone(),
            "loader_mode": run_receipt["model"]["loader_mode"].clone(),
            "fallback_loader_used": run_receipt["model"]["fallback_loader_used"].clone(),
            "tokenizer": run_receipt["model"]["tokenizer"].clone(),
        },
        "backend": {
            "requested_backend": run_receipt["requested_backend"].clone(),
            "selected_backend": run_receipt["selected_backend"].clone(),
            "runtime_api": run_receipt["runtime_api"].clone(),
            "fallback_used": run_receipt["fallback_used"].clone(),
            "fallback_reason": run_receipt["fallback_reason"].clone(),
        },
        "prompt_template": {
            "family": "llama3-chat",
            "system_prompt_present": system_prompt_for_receipt.as_ref().is_some_and(|value| !value.is_empty()),
            "bos_inserted": true,
            "assistant_prefix_inserted": true,
            "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"],
        },
        "prompt_prefill": {
            "exercised": run_receipt["profile"]["prompt_prefill"]["exercised"].clone(),
            "tokens": run_receipt["profile"]["prompt_prefill"]["tokens"].clone(),
            "kv_cache_behavior": run_receipt["profile"]["prompt_prefill"]["kv_cache_behavior"].clone(),
        },
        "token_ids": {
            "prompt": run_receipt["tokens"]["prompt_ids"].clone(),
            "generated": run_receipt["tokens"]["generated_ids"].clone(),
        },
        "bitnet": {
            "quantization": run_receipt["bitnet"]["quantization"].clone(),
            "kernel_family": run_receipt["bitnet"]["kernel_family"].clone(),
            "kernel_id": run_receipt["kernel"]["kernel_id"].clone(),
            "weights_uploaded_once": run_receipt["bitnet"]["weights_uploaded_once"].clone(),
            "per_token_weight_upload": run_receipt["bitnet"]["per_token_weight_upload"].clone(),
        },
        "execution_coverage": run_receipt["execution_coverage"].clone(),
        "quality": quality,
        "speedup_claim": false,
        "source_receipt": run_receipt,
    });
    write_json_output(Some(&receipt_path), &answer_receipt)?;
    if strict_cuda {
        validate_strict_cuda_answer_quality(&answer_receipt)?;
    }
    if strict_cpu {
        validate_strict_cpu_answer_quality(&answer_receipt)?;
    }
    Ok(())
}

fn validate_strict_cuda_ask_receipt(run_receipt: &serde_json::Value) -> Result<()> {
    const RTX_5070_TI_CUDA: &str = "nvidia-rtx-5070-ti-cuda";
    let selected_backend = run_receipt["selected_backend"].as_str().unwrap_or_default();
    let runtime_api = run_receipt["runtime_api"].as_str().unwrap_or_default();
    let fallback_used = run_receipt["fallback_used"].as_bool().unwrap_or(true);
    if selected_backend != RTX_5070_TI_CUDA || runtime_api != "cuda" || fallback_used {
        anyhow::bail!(
            "strict CUDA ask did not preserve the RTX 5070 Ti CUDA lane: selected_backend={selected_backend}, runtime_api={runtime_api}, fallback_used={fallback_used}"
        );
    }
    let cpu_fallback = run_receipt["execution_coverage"]["bitnet_linear_layers_cpu_fallback"]
        .as_u64()
        .unwrap_or(1);
    if cpu_fallback != 0 {
        anyhow::bail!("strict CUDA ask recorded {cpu_fallback} BitNet linear CPU fallback layers");
    }
    Ok(())
}

fn validate_strict_cuda_answer_quality(answer_receipt: &serde_json::Value) -> Result<()> {
    let quality = &answer_receipt["quality"];
    if quality["garbage_filter_passed"].as_bool().unwrap_or(false) {
        return Ok(());
    }

    let quality_summary = serde_json::to_string(quality)
        .unwrap_or_else(|_| "<unprintable quality receipt>".to_string());
    anyhow::bail!(
        "strict CUDA ask failed answer quality gate after writing receipt: {quality_summary}"
    )
}

fn validate_strict_cpu_ask_receipt(run_receipt: &serde_json::Value) -> Result<()> {
    let selected_backend = run_receipt["selected_backend"].as_str().unwrap_or_default();
    let runtime_api = run_receipt["runtime_api"].as_str().unwrap_or_default();
    let fallback_used = run_receipt["fallback_used"].as_bool().unwrap_or(true);
    if runtime_api != "cpu" || fallback_used {
        anyhow::bail!(
            "strict CPU ask did not preserve the CPU lane: selected_backend={selected_backend}, runtime_api={runtime_api}, fallback_used={fallback_used}"
        );
    }
    if !matches!(selected_backend, "cpu" | "cpu-rust") {
        anyhow::bail!("strict CPU ask selected non-CPU backend `{selected_backend}`");
    }

    let loader_mode = run_receipt["loader"]["mode"]
        .as_str()
        .or_else(|| run_receipt["model"]["loader_mode"].as_str())
        .unwrap_or_default();
    if loader_mode != bitnet_models::GgufLoaderMode::RealGguf.as_str() {
        anyhow::bail!("strict CPU ask requires real_gguf loader mode, got `{loader_mode}`");
    }

    let tokenizer_strict = run_receipt["tokenizer"]["strict"].as_bool().unwrap_or(false);
    let tokenizer_source = run_receipt["tokenizer"]["source"].as_str().unwrap_or_default();
    if !tokenizer_strict || tokenizer_source.is_empty() || tokenizer_source == "unknown" {
        anyhow::bail!(
            "strict CPU ask requires strict tokenizer source, got source=`{tokenizer_source}` strict={tokenizer_strict}"
        );
    }

    let selected_kernel = run_receipt["kernel"]["kernel_id"].as_str().unwrap_or_default();
    if selected_kernel.is_empty()
        || selected_kernel.contains("mock")
        || selected_kernel.contains("diagnostic")
    {
        anyhow::bail!("strict CPU ask selected invalid kernel `{selected_kernel}`");
    }
    Ok(())
}

fn validate_strict_cpu_answer_quality(answer_receipt: &serde_json::Value) -> Result<()> {
    let quality = &answer_receipt["quality"];
    if quality["garbage_filter_passed"].as_bool().unwrap_or(false) {
        return Ok(());
    }

    let quality_summary = serde_json::to_string(quality)
        .unwrap_or_else(|_| "<unprintable quality receipt>".to_string());
    anyhow::bail!(
        "strict CPU ask failed answer quality gate after writing receipt: {quality_summary}"
    )
}

fn ensure_strict_cuda_ask_runtime_libraries_visible() -> Result<Option<std::path::PathBuf>> {
    #[cfg(all(feature = "cuda", target_os = "windows"))]
    {
        ensure_windows_cuda_toolkit_bin_on_path()
    }

    #[cfg(not(all(feature = "cuda", target_os = "windows")))]
    {
        Ok(None)
    }
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn ensure_windows_cuda_toolkit_bin_on_path() -> Result<Option<std::path::PathBuf>> {
    if windows_cuda_runtime_libraries_visible_on_path() {
        return Ok(None);
    }

    let Some(cuda_bin) = discover_windows_cuda_toolkit_bin() else {
        return Ok(None);
    };
    prepend_process_path(&cuda_bin).with_context(|| {
        format!("failed to add CUDA Toolkit bin to PATH: {}", cuda_bin.display())
    })?;
    Ok(Some(cuda_bin))
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn discover_windows_cuda_toolkit_bin() -> Option<std::path::PathBuf> {
    discover_cuda_toolkit_bin_from_roots(windows_cuda_toolkit_search_roots())
}

#[cfg(any(test, all(feature = "cuda", target_os = "windows")))]
fn discover_cuda_toolkit_bin_from_roots<I, P>(roots: I) -> Option<std::path::PathBuf>
where
    I: IntoIterator<Item = P>,
    P: AsRef<std::path::Path>,
{
    let mut candidates = Vec::new();
    for root in roots {
        collect_cuda_toolkit_bin_candidates(root.as_ref(), &mut candidates);
    }
    candidates.sort_by(|left, right| {
        cuda_bin_version_key(right).cmp(&cuda_bin_version_key(left)).then_with(|| left.cmp(right))
    });
    candidates.into_iter().find(|candidate| cuda_toolkit_bin_has_runtime_libraries(candidate))
}

#[cfg(any(test, all(feature = "cuda", target_os = "windows")))]
fn collect_cuda_toolkit_bin_candidates(
    root: &std::path::Path,
    candidates: &mut Vec<std::path::PathBuf>,
) {
    candidates.push(root.to_path_buf());
    candidates.push(root.join("bin"));

    let Ok(children) = std::fs::read_dir(root) else {
        return;
    };
    for child in children.flatten() {
        let path = child.path();
        if path.is_dir()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with('v'))
        {
            candidates.push(path.join("bin"));
        }
    }
}

#[cfg(any(test, all(feature = "cuda", target_os = "windows")))]
fn cuda_toolkit_bin_has_runtime_libraries(bin: &std::path::Path) -> bool {
    cuda_toolkit_bin_has_any(bin, WINDOWS_NVRTC_LIBRARY_NAMES)
        && cuda_toolkit_bin_has_any(bin, WINDOWS_CUDART_LIBRARY_NAMES)
}

#[cfg(any(test, all(feature = "cuda", target_os = "windows")))]
fn cuda_toolkit_bin_has_any(bin: &std::path::Path, names: &[&str]) -> bool {
    names.iter().any(|name| bin.join(name).is_file())
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn windows_cuda_runtime_libraries_visible_on_path() -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|entry| cuda_toolkit_bin_has_runtime_libraries(&entry))
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn windows_cuda_toolkit_search_roots() -> Vec<std::path::PathBuf> {
    let mut roots = Vec::new();
    for (key, value) in std::env::vars_os() {
        if key.to_string_lossy().to_ascii_uppercase().starts_with("CUDA_PATH") && !value.is_empty()
        {
            roots.push(std::path::PathBuf::from(value));
        }
    }

    for key in ["ProgramW6432", "ProgramFiles"] {
        if let Some(program_files) = std::env::var_os(key) {
            roots.push(
                std::path::PathBuf::from(program_files)
                    .join("NVIDIA GPU Computing Toolkit")
                    .join("CUDA"),
            );
        }
    }
    roots.push(std::path::PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"));

    dedupe_paths(roots)
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn dedupe_paths(paths: Vec<std::path::PathBuf>) -> Vec<std::path::PathBuf> {
    let mut deduped = Vec::<std::path::PathBuf>::new();
    for path in paths {
        if !deduped.iter().any(|existing| paths_equal_for_process_path(existing, &path)) {
            deduped.push(path);
        }
    }
    deduped
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn prepend_process_path(path: &std::path::Path) -> Result<()> {
    let current = std::env::var_os("PATH").unwrap_or_default();
    let mut entries = Vec::from([path.to_path_buf()]);
    entries.extend(
        std::env::split_paths(&current).filter(|entry| !paths_equal_for_process_path(entry, path)),
    );
    let updated_path = std::env::join_paths(entries)?;
    // SAFETY: Strict CUDA ask adjusts this process before CUDA/NVRTC loading
    // starts, so cudarc can discover Toolkit DLLs installed in the standard
    // Windows location. The CLI does not read PATH concurrently in this block.
    unsafe {
        std::env::set_var("PATH", updated_path);
    }
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn paths_equal_for_process_path(left: &std::path::Path, right: &std::path::Path) -> bool {
    left.to_string_lossy().eq_ignore_ascii_case(&right.to_string_lossy())
}

#[cfg(any(test, all(feature = "cuda", target_os = "windows")))]
fn cuda_bin_version_key(path: &std::path::Path) -> (u32, u32, u32) {
    let version_name =
        path.parent().and_then(|parent| parent.file_name()).and_then(|name| name.to_str());
    parse_cuda_version_name(version_name.unwrap_or_default())
}

#[cfg(any(test, all(feature = "cuda", target_os = "windows")))]
fn parse_cuda_version_name(name: &str) -> (u32, u32, u32) {
    let Some(rest) = name.strip_prefix('v') else {
        return (0, 0, 0);
    };
    let mut parts = rest.split('.');
    let major = parts.next().and_then(|value| value.parse().ok()).unwrap_or_default();
    let minor = parts.next().and_then(|value| value.parse().ok()).unwrap_or_default();
    let patch = parts.next().and_then(|value| value.parse().ok()).unwrap_or_default();
    (major, minor, patch)
}

#[cfg(any(test, all(feature = "cuda", target_os = "windows")))]
const WINDOWS_NVRTC_LIBRARY_NAMES: &[&str] =
    &["nvrtc64_120_0.dll", "nvrtc64_120.dll", "nvrtc64_12.dll", "nvrtc64.dll", "nvrtc.dll"];

#[cfg(any(test, all(feature = "cuda", target_os = "windows")))]
const WINDOWS_CUDART_LIBRARY_NAMES: &[&str] =
    &["cudart64_120.dll", "cudart64_12.dll", "cudart64.dll", "cudart.dll"];

fn answer_quality_receipt(
    answer: &str,
    run_receipt: &serde_json::Value,
    max_new_tokens: usize,
) -> serde_json::Value {
    let trimmed = strip_answer_special_markers(answer).trim().to_string();
    let non_empty_answer = !trimmed.is_empty();
    let printable_utf8 = trimmed.chars().all(|ch| ch == '\n' || ch == '\t' || !ch.is_control());
    let no_replacement_chars = !trimmed.contains('\u{FFFD}');
    let no_raw_special_tokens = !trimmed.contains("<|") && !trimmed.contains("|>");
    let mostly_text = answer_mostly_text(&trimmed);
    let language_signal = answer_has_language_signal(&trimmed);
    let suspicious_fragment_count = suspicious_answer_fragment_count(&trimmed);
    let fragment_filter_passed = suspicious_fragment_count <= 1;
    let garbage_filter_passed = non_empty_answer
        && printable_utf8
        && no_replacement_chars
        && no_raw_special_tokens
        && mostly_text
        && language_signal
        && fragment_filter_passed;
    let generated = run_receipt["tokens"]["generated"].as_u64().unwrap_or_default() as usize;
    serde_json::json!({
        "printable_utf8": printable_utf8,
        "non_empty_answer": non_empty_answer,
        "stop_reason": if generated >= max_new_tokens { "max_tokens" } else { "eos_or_stop_sequence" },
        "garbage_filter_passed": garbage_filter_passed,
        "no_replacement_chars": no_replacement_chars,
        "no_raw_special_tokens": no_raw_special_tokens,
        "mostly_text": mostly_text,
        "language_signal": language_signal,
        "suspicious_fragment_count": suspicious_fragment_count,
        "fragment_filter_passed": fragment_filter_passed,
    })
}

fn strip_answer_special_markers(answer: &str) -> String {
    answer.replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "").replace("<|eot_id|>", "")
}

fn answer_mostly_text(answer: &str) -> bool {
    let mut meaningful = 0usize;
    let mut punctuation_or_control = 0usize;
    for ch in answer.chars() {
        if ch.is_alphanumeric() || ch.is_whitespace() {
            meaningful += 1;
        } else if ch.is_ascii_punctuation() || ch.is_control() {
            punctuation_or_control += 1;
        }
    }
    meaningful > 0 && punctuation_or_control <= meaningful.saturating_mul(2)
}

fn answer_has_language_signal(answer: &str) -> bool {
    let compact: String = answer.chars().filter(|ch| !ch.is_whitespace()).collect();
    let numeric_short_answer = compact.len() <= 8
        && compact.chars().any(|ch| ch.is_ascii_digit())
        && compact.chars().all(|ch| ch.is_ascii_digit() || matches!(ch, '.' | '-' | '+'));
    if numeric_short_answer {
        return true;
    }

    answer_word_tokens(answer).any(|word| ANSWER_QUALITY_LANGUAGE_WORDS.contains(&word.as_str()))
}

fn suspicious_answer_fragment_count(answer: &str) -> usize {
    answer
        .split_whitespace()
        .filter(|token| {
            let alphabetic = token.chars().filter(|ch| ch.is_alphabetic()).count();
            if alphabetic == 0 {
                return false;
            }
            let apostrophes = token.matches('\'').count();
            let ascii_punctuation = token.chars().filter(|ch| ch.is_ascii_punctuation()).count();
            let internal_period = token.contains('.')
                && !token.ends_with('.')
                && token.chars().any(|ch| ch.is_alphabetic());
            (apostrophes > 1) || internal_period || (alphabetic >= 3 && ascii_punctuation >= 3)
        })
        .count()
}

fn answer_word_tokens(answer: &str) -> impl Iterator<Item = String> + '_ {
    answer
        .split(|ch: char| !ch.is_alphabetic())
        .filter(|word| word.len() >= 2)
        .map(str::to_ascii_lowercase)
}

const ANSWER_QUALITY_LANGUAGE_WORDS: &[&str] = &[
    "a",
    "about",
    "add",
    "adds",
    "an",
    "and",
    "answer",
    "are",
    "architecture",
    "blue",
    "bit",
    "bitnet",
    "black",
    "capital",
    "color",
    "colors",
    "common",
    "compute",
    "data",
    "efficient",
    "explain",
    "for",
    "four",
    "france",
    "function",
    "green",
    "is",
    "language",
    "low",
    "memory",
    "model",
    "number",
    "numbers",
    "of",
    "one",
    "paris",
    "python",
    "red",
    "reduce",
    "sentence",
    "shape",
    "shapes",
    "the",
    "that",
    "three",
    "to",
    "uses",
    "weight",
    "weights",
    "white",
    "with",
    "wet",
    "water",
    "yellow",
    "yes",
    "no",
];

fn ensure_non_empty_generation_context(
    tokens: &mut Vec<u32>,
    tokenizer: &dyn bitnet_tokenizers::Tokenizer,
) -> Result<()> {
    if !tokens.is_empty() {
        return Ok(());
    }

    if let Some(bos_id) = tokenizer.bos_token_id() {
        warn!(
            "Tokenizer produced an empty prompt token sequence; seeding generation with BOS token id {bos_id}"
        );
        tokens.push(bos_id);
        return Ok(());
    }

    anyhow::bail!(
        "Prompt produced zero tokens and tokenizer has no BOS token. Provide a non-empty prompt, set --bos with a tokenizer that defines BOS, or use a template that emits content."
    )
}

#[cfg(test)]
mod empty_generation_context_tests {
    use super::{ensure_non_empty_generation_context, nvidia_smi_memory_used_bytes_from_csv};
    use bitnet_common::Result as TokenizerResult;
    use bitnet_tokenizers::Tokenizer;

    struct EmptyTokenizerWithBos;
    impl Tokenizer for EmptyTokenizerWithBos {
        fn encode(
            &self,
            _text: &str,
            _add_bos: bool,
            _parse_special: bool,
        ) -> TokenizerResult<Vec<u32>> {
            Ok(Vec::new())
        }
        fn decode(&self, _ids: &[u32]) -> TokenizerResult<String> {
            Ok(String::new())
        }
        fn vocab_size(&self) -> usize {
            1
        }
        fn token_to_piece(&self, _token: u32) -> Option<String> {
            None
        }
        fn eos_token_id(&self) -> Option<u32> {
            Some(2)
        }
        fn bos_token_id(&self) -> Option<u32> {
            Some(1)
        }
    }

    struct EmptyTokenizerNoBos;
    impl Tokenizer for EmptyTokenizerNoBos {
        fn encode(
            &self,
            _text: &str,
            _add_bos: bool,
            _parse_special: bool,
        ) -> TokenizerResult<Vec<u32>> {
            Ok(Vec::new())
        }
        fn decode(&self, _ids: &[u32]) -> TokenizerResult<String> {
            Ok(String::new())
        }
        fn vocab_size(&self) -> usize {
            1
        }
        fn token_to_piece(&self, _token: u32) -> Option<String> {
            None
        }
    }

    #[test]
    fn empty_tokens_are_seeded_with_bos_when_available() {
        let mut tokens = Vec::new();
        let tokenizer = EmptyTokenizerWithBos;
        ensure_non_empty_generation_context(&mut tokens, &tokenizer).expect("should seed BOS");
        assert_eq!(tokens, vec![1]);
    }

    #[test]
    fn empty_tokens_error_when_bos_unavailable() {
        let mut tokens = Vec::new();
        let tokenizer = EmptyTokenizerNoBos;
        let err = ensure_non_empty_generation_context(&mut tokens, &tokenizer)
            .expect_err("missing BOS should return error");
        assert!(err.to_string().contains("zero tokens"));
        assert!(tokens.is_empty());
    }

    #[test]
    fn parses_nvidia_smi_memory_used_mib() {
        assert_eq!(nvidia_smi_memory_used_bytes_from_csv("5673 MiB\n"), Some(5_948_571_648));
    }

    #[test]
    fn rejects_nvidia_smi_memory_used_without_number() {
        assert_eq!(nvidia_smi_memory_used_bytes_from_csv("N/A\n"), None);
    }
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

fn elapsed_ms(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn ms_to_us(ms: f64) -> u128 {
    (ms * 1000.0).round() as u128
}

fn rounded_ms(ms: f64) -> f64 {
    (ms * 1000.0).round() / 1000.0
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AllocationAuditSnapshot {
    alloc_count: u64,
    alloc_bytes: u64,
    dealloc_count: u64,
    dealloc_bytes: u64,
}

impl AllocationAuditSnapshot {
    fn current() -> Self {
        Self {
            alloc_count: ALLOCATION_AUDIT_ALLOC_COUNT.load(Ordering::Relaxed),
            alloc_bytes: ALLOCATION_AUDIT_ALLOC_BYTES.load(Ordering::Relaxed),
            dealloc_count: ALLOCATION_AUDIT_DEALLOC_COUNT.load(Ordering::Relaxed),
            dealloc_bytes: ALLOCATION_AUDIT_DEALLOC_BYTES.load(Ordering::Relaxed),
        }
    }

    fn delta_since(start: Self) -> Self {
        let current = Self::current();
        Self {
            alloc_count: current.alloc_count.saturating_sub(start.alloc_count),
            alloc_bytes: current.alloc_bytes.saturating_sub(start.alloc_bytes),
            dealloc_count: current.dealloc_count.saturating_sub(start.dealloc_count),
            dealloc_bytes: current.dealloc_bytes.saturating_sub(start.dealloc_bytes),
        }
    }
}

struct AllocationAuditGuard {
    previous: bool,
}

impl AllocationAuditGuard {
    fn enable(enabled: bool) -> Self {
        let previous = ALLOCATION_AUDIT_ENABLED.swap(enabled, Ordering::Relaxed);
        Self { previous }
    }
}

impl Drop for AllocationAuditGuard {
    fn drop(&mut self) {
        ALLOCATION_AUDIT_ENABLED.store(self.previous, Ordering::Relaxed);
    }
}

fn timing_samples_json(samples: &[f64]) -> serde_json::Value {
    if samples.is_empty() {
        return serde_json::json!({
            "count": 0,
            "total_ms": 0.0,
            "min_ms": serde_json::Value::Null,
            "mean_ms": serde_json::Value::Null,
            "p50_ms": serde_json::Value::Null,
            "p95_ms": serde_json::Value::Null,
            "max_ms": serde_json::Value::Null,
        });
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let total_ms = samples.iter().sum::<f64>();
    let mean_ms = total_ms / samples.len() as f64;

    serde_json::json!({
        "count": samples.len(),
        "total_ms": rounded_ms(total_ms),
        "min_ms": rounded_ms(sorted[0]),
        "mean_ms": rounded_ms(mean_ms),
        "p50_ms": rounded_ms(percentile_nearest(&sorted, 50)),
        "p95_ms": rounded_ms(percentile_nearest(&sorted, 95)),
        "max_ms": rounded_ms(sorted[sorted.len() - 1]),
    })
}

fn allocation_samples_json(samples: &[AllocationAuditSnapshot]) -> serde_json::Value {
    if samples.is_empty() {
        return serde_json::json!({
            "count": 0,
            "alloc_count_total": 0,
            "alloc_bytes_total": 0,
            "dealloc_count_total": 0,
            "dealloc_bytes_total": 0,
            "net_bytes_total": 0,
            "mean_alloc_count_per_token": serde_json::Value::Null,
            "mean_alloc_bytes_per_token": serde_json::Value::Null,
            "max_alloc_count_per_token": serde_json::Value::Null,
            "max_alloc_bytes_per_token": serde_json::Value::Null,
        });
    }

    let total_alloc_count = samples.iter().map(|sample| sample.alloc_count).sum::<u64>();
    let total_alloc_bytes = samples.iter().map(|sample| sample.alloc_bytes).sum::<u64>();
    let total_dealloc_count = samples.iter().map(|sample| sample.dealloc_count).sum::<u64>();
    let total_dealloc_bytes = samples.iter().map(|sample| sample.dealloc_bytes).sum::<u64>();
    let max_alloc_count = samples.iter().map(|sample| sample.alloc_count).max().unwrap_or(0);
    let max_alloc_bytes = samples.iter().map(|sample| sample.alloc_bytes).max().unwrap_or(0);
    let count = samples.len() as f64;

    let net_bytes_total = total_alloc_bytes as i64 - total_dealloc_bytes as i64;

    serde_json::json!({
        "count": samples.len(),
        "alloc_count_total": total_alloc_count,
        "alloc_bytes_total": total_alloc_bytes,
        "dealloc_count_total": total_dealloc_count,
        "dealloc_bytes_total": total_dealloc_bytes,
        "net_bytes_total": net_bytes_total,
        "mean_alloc_count_per_token": rounded_ms(total_alloc_count as f64 / count),
        "mean_alloc_bytes_per_token": rounded_ms(total_alloc_bytes as f64 / count),
        "max_alloc_count_per_token": max_alloc_count,
        "max_alloc_bytes_per_token": max_alloc_bytes,
    })
}

fn allocation_count_delta_json(samples: &[AllocationAuditSnapshot]) -> serde_json::Value {
    if samples.is_empty() {
        return serde_json::json!({
            "count": 0,
            "total": 0,
            "mean_per_token": serde_json::Value::Null,
            "max_per_token": serde_json::Value::Null,
        });
    }

    let total = samples.iter().map(|sample| sample.alloc_count).sum::<u64>();
    let max = samples.iter().map(|sample| sample.alloc_count).max().unwrap_or(0);

    serde_json::json!({
        "count": samples.len(),
        "total": total,
        "mean_per_token": rounded_ms(total as f64 / samples.len() as f64),
        "max_per_token": max,
    })
}

fn allocation_bytes_delta_json(samples: &[AllocationAuditSnapshot]) -> serde_json::Value {
    if samples.is_empty() {
        return serde_json::json!({
            "count": 0,
            "total": 0,
            "mean_per_token": serde_json::Value::Null,
            "max_per_token": serde_json::Value::Null,
        });
    }

    let total = samples.iter().map(|sample| sample.alloc_bytes).sum::<u64>();
    let max = samples.iter().map(|sample| sample.alloc_bytes).max().unwrap_or(0);

    serde_json::json!({
        "count": samples.len(),
        "total": total,
        "mean_per_token": rounded_ms(total as f64 / samples.len() as f64),
        "max_per_token": max,
    })
}

fn mean_alloc_count(samples: &[AllocationAuditSnapshot]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    Some(samples.iter().map(|sample| sample.alloc_count).sum::<u64>() as f64 / samples.len() as f64)
}

fn mean_alloc_bytes(samples: &[AllocationAuditSnapshot]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    Some(samples.iter().map(|sample| sample.alloc_bytes).sum::<u64>() as f64 / samples.len() as f64)
}

fn percentile_nearest(sorted_samples: &[f64], percentile: usize) -> f64 {
    debug_assert!(!sorted_samples.is_empty());
    let rank = (percentile as f64 / 100.0 * sorted_samples.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted_samples.len() - 1);
    sorted_samples[index]
}

fn steady_decode_tps_ms(decode_step_ms: &[f64]) -> Option<f64> {
    let steady = decode_step_ms.get(1..)?;
    if steady.is_empty() {
        return None;
    }
    let steady_ms = steady.iter().sum::<f64>();
    if steady_ms <= 0.0 {
        return None;
    }
    Some(steady.len() as f64 / (steady_ms / 1000.0))
}

fn profile_claim_scope(runtime_api: &str, selected_backend: &str) -> &'static str {
    if runtime_api == "cpu" || selected_backend == "cpu-rust" {
        "selected CPU backend phase timing only"
    } else if selected_backend.starts_with("apple-m4") || runtime_api == "metal" {
        "selected Apple backend phase timing only"
    } else if runtime_api == "cuda" || selected_backend.contains("cuda") {
        "selected CUDA backend phase timing only"
    } else {
        "selected backend phase timing only"
    }
}

fn profile_machine_context_recorded(
    runtime_api: &str,
    selected_backend: &str,
    apple_machine_present: bool,
    cpu_features: &[String],
    cpu_model_present: bool,
) -> bool {
    if runtime_api == "cpu" || selected_backend == "cpu-rust" {
        return cpu_model_present || !cpu_features.is_empty();
    }

    if selected_backend.starts_with("apple-m4") || runtime_api == "metal" {
        return apple_machine_present;
    }

    apple_machine_present || cpu_model_present || !cpu_features.is_empty()
}

fn allocation_audit_backend_supported(identity: &RunBackendIdentity) -> bool {
    identity.requested_backend == "apple-m4-cpu-neon"
        && identity.selected_backend == "apple-m4-cpu-neon"
        && identity.runtime_api == "cpu"
        && !identity.fallback_used
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

fn backend_selection_error_message_with_note(requested_backend_label: &str, error: &str) -> String {
    match apple_backend_failure_note(requested_backend_label) {
        Some(note) => format!("{error}. {note}"),
        None => error.to_string(),
    }
}

fn apple_backend_failure_note(requested_backend_label: &str) -> Option<&'static str> {
    match requested_backend_label.trim().to_ascii_lowercase().as_str() {
        "apple-m4-metal" => Some(
            "apple-m4-metal is the native Metal proof lane; it does not imply MPSGraph or Neural Engine execution and must not silently fall back to CPU in strict mode. Run on native macOS Apple M4 with Metal visible, or request apple-m4-cpu-neon for the CPU/NEON reference lane.",
        ),
        "apple-m4-mpsgraph" => Some(
            "apple-m4-mpsgraph is the graph/reference proof lane; it is not native Metal kernel proof and is not Neural Engine proof unless the resolved target is receipt-backed.",
        ),
        "apple-m4-cpu-neon" => Some(
            "apple-m4-cpu-neon is the Apple ARM64 CPU/NEON fallback and parity lane; it is not Metal acceleration, and scalar fallback must be visible in receipts.",
        ),
        _ => None,
    }
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
    fn answer_quality_rejects_punctuation_noise() {
        let run_receipt = serde_json::json!({
            "tokens": {
                "generated": 8,
            }
        });
        let quality = answer_quality_receipt("!!!,,,!!!", &run_receipt, 16);

        assert_eq!(quality["non_empty_answer"], true);
        assert_eq!(quality["mostly_text"], false);
        assert_eq!(quality["garbage_filter_passed"], false);
    }

    #[test]
    fn answer_quality_marks_max_token_stop() {
        let run_receipt = serde_json::json!({
            "tokens": {
                "generated": 16,
            }
        });
        let quality = answer_quality_receipt("BitNet uses low-bit weights.", &run_receipt, 16);

        assert_eq!(quality["garbage_filter_passed"], true);
        assert_eq!(quality["stop_reason"], "max_tokens");
    }

    #[test]
    fn answer_quality_rejects_observed_cuda_fragment_garbage() {
        let run_receipt = serde_json::json!({
            "tokens": {
                "generated": 16,
            }
        });
        let answer = "-lived'Elicence'E facts-livedConvert!\"\n\n Gab Clock Paperback,SIGNALIR realise.iOS rzd";
        let quality = answer_quality_receipt(answer, &run_receipt, 16);

        assert_eq!(quality["non_empty_answer"], true);
        assert_eq!(quality["mostly_text"], true);
        assert_eq!(quality["language_signal"], false);
        assert_eq!(quality["fragment_filter_passed"], false);
        assert_eq!(quality["garbage_filter_passed"], false);
    }

    #[test]
    fn answer_quality_accepts_short_numeric_answer() {
        let run_receipt = serde_json::json!({
            "tokens": {
                "generated": 1,
            }
        });
        let quality = answer_quality_receipt("4", &run_receipt, 16);

        assert_eq!(quality["language_signal"], true);
        assert_eq!(quality["garbage_filter_passed"], true);
    }

    #[test]
    fn strict_cuda_answer_quality_gate_rejects_failed_receipt() {
        let answer_receipt = serde_json::json!({
            "quality": {
                "garbage_filter_passed": false,
                "language_signal": false,
                "suspicious_fragment_count": 3,
            }
        });

        let err = validate_strict_cuda_answer_quality(&answer_receipt).unwrap_err().to_string();

        assert!(err.contains("strict CUDA ask failed answer quality gate"), "got: {err}");
        assert!(err.contains("\"garbage_filter_passed\":false"), "got: {err}");
    }

    #[test]
    fn strict_cuda_answer_quality_gate_accepts_passed_receipt() {
        let answer_receipt = serde_json::json!({
            "quality": {
                "garbage_filter_passed": true,
            }
        });

        validate_strict_cuda_answer_quality(&answer_receipt).unwrap();
    }

    #[test]
    fn strict_cpu_ask_receipt_accepts_real_cpu_path() {
        let run_receipt = serde_json::json!({
            "selected_backend": "cpu-rust",
            "runtime_api": "cpu",
            "fallback_used": false,
            "loader": { "mode": "real_gguf" },
            "tokenizer": { "source": "gguf_metadata", "strict": true },
            "kernel": { "kernel_id": "i2_s-avx2-reference" }
        });

        validate_strict_cpu_ask_receipt(&run_receipt).unwrap();
    }

    #[test]
    fn strict_cpu_ask_receipt_rejects_fallback() {
        let run_receipt = serde_json::json!({
            "selected_backend": "cpu-rust",
            "runtime_api": "cpu",
            "fallback_used": true,
            "loader": { "mode": "real_gguf" },
            "tokenizer": { "source": "gguf_metadata", "strict": true },
            "kernel": { "kernel_id": "i2_s-avx2-reference" }
        });

        let err = validate_strict_cpu_ask_receipt(&run_receipt).unwrap_err().to_string();

        assert!(err.contains("strict CPU ask did not preserve the CPU lane"), "got: {err}");
    }

    #[test]
    fn strict_cpu_answer_quality_gate_rejects_failed_receipt() {
        let answer_receipt = serde_json::json!({
            "quality": {
                "garbage_filter_passed": false,
            }
        });

        let err = validate_strict_cpu_answer_quality(&answer_receipt).unwrap_err().to_string();

        assert!(err.contains("strict CPU ask failed answer quality gate"), "got: {err}");
    }

    #[test]
    fn cuda_toolkit_bin_discovery_prefers_highest_version_with_runtime_libraries() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cuda_root = temp_dir.path().join("CUDA");
        let older_bin = cuda_root.join("v12.1").join("bin");
        let newer_bin = cuda_root.join("v12.9").join("bin");
        std::fs::create_dir_all(&older_bin).unwrap();
        std::fs::create_dir_all(&newer_bin).unwrap();
        std::fs::write(older_bin.join("nvrtc64_120_0.dll"), b"").unwrap();
        std::fs::write(older_bin.join("cudart64_120.dll"), b"").unwrap();
        std::fs::write(newer_bin.join("nvrtc64_120_0.dll"), b"").unwrap();
        std::fs::write(newer_bin.join("cudart64_120.dll"), b"").unwrap();

        let discovered = discover_cuda_toolkit_bin_from_roots([cuda_root]).unwrap();

        assert_eq!(discovered, newer_bin);
    }

    #[test]
    fn cuda_toolkit_bin_discovery_rejects_partial_toolkit_bin() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cuda_root = temp_dir.path().join("CUDA");
        let bin = cuda_root.join("v12.9").join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(bin.join("nvrtc64_120_0.dll"), b"").unwrap();

        assert!(discover_cuda_toolkit_bin_from_roots([cuda_root]).is_none());
    }

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
    fn strict_apple_metal_error_describes_non_fallback_proof_lane() {
        let err = resolve_run_backend_identity("apple-m4-metal", true).unwrap_err().to_string();

        assert!(err.contains("apple-m4-metal"), "got: {err}");
        assert!(err.contains("native Metal proof lane"), "got: {err}");
        assert!(err.contains("must not silently fall back to CPU"), "got: {err}");
        assert!(err.contains("apple-m4-cpu-neon"), "got: {err}");
    }

    #[test]
    fn non_strict_apple_mpsgraph_fallback_reason_keeps_graph_boundary() {
        let identity = resolve_run_backend_identity("apple-m4-mpsgraph", false).unwrap();

        assert_eq!(identity.requested_backend, "apple-m4-mpsgraph");
        assert!(identity.fallback_used);
        let fallback_reason = identity.fallback_reason.unwrap();
        assert!(fallback_reason.contains("graph/reference proof lane"), "got: {fallback_reason}");
        assert!(
            fallback_reason.contains("not native Metal kernel proof"),
            "got: {fallback_reason}"
        );
        assert!(fallback_reason.contains("not Neural Engine proof"), "got: {fallback_reason}");
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
    fn qwen_receipt_identity_uses_dense_family() {
        let path = std::path::Path::new("models/slm/Qwen3-0.6B-Q8_0.gguf");
        let architecture = infer_model_architecture(path);

        assert_eq!(infer_model_repo(path), "Qwen/Qwen3-0.6B-GGUF");
        assert_eq!(architecture, "qwen3");
        assert_eq!(receipt_model_family(&architecture), "qwen");
    }

    #[test]
    fn receipt_tokenizer_type_uses_inferred_model_label() {
        assert_eq!(
            tokenizer_type_for_receipt(
                "llama3",
                bitnet_tokenizers::auto::TokenizerSource::Explicit
            ),
            "llama3"
        );
        assert_eq!(
            tokenizer_type_for_receipt(
                "explicit",
                bitnet_tokenizers::auto::TokenizerSource::Explicit
            ),
            "external_tokenizer_file"
        );
    }

    #[test]
    fn gguf_header_counts_for_receipt_reads_counts_without_full_metadata_parse() {
        use std::io::Write;

        let mut file = tempfile::Builder::new().suffix(".gguf").tempfile().unwrap();
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3_u32.to_le_bytes()).unwrap();
        file.write_all(&332_u64.to_le_bytes()).unwrap();
        file.write_all(&45_u64.to_le_bytes()).unwrap();

        assert_eq!(gguf_header_counts_for_receipt(file.path(), false), Some((45, 332)));
        assert_eq!(gguf_header_counts_for_receipt(file.path(), true), None);
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

    #[test]
    fn timing_samples_json_records_percentiles_and_total() {
        let summary = timing_samples_json(&[3.0, 1.0, 2.0, 10.0]);

        assert_eq!(summary["count"], 4);
        assert_eq!(summary["total_ms"], 16.0);
        assert_eq!(summary["min_ms"], 1.0);
        assert_eq!(summary["mean_ms"], 4.0);
        assert_eq!(summary["p50_ms"], 2.0);
        assert_eq!(summary["p95_ms"], 10.0);
        assert_eq!(summary["max_ms"], 10.0);
    }

    #[test]
    fn steady_decode_tps_excludes_first_decode_token() {
        let tps = steady_decode_tps_ms(&[100.0, 50.0, 50.0]).unwrap();

        assert_eq!((tps * 1000.0).round() / 1000.0, 20.0);
        assert!(steady_decode_tps_ms(&[100.0]).is_none());
    }

    #[test]
    fn cpu_profile_receipt_uses_cpu_claim_scope() {
        let cpu_features = vec!["avx2".to_string(), "fma".to_string()];

        assert_eq!(
            profile_claim_scope("cpu", "cpu-rust"),
            "selected CPU backend phase timing only"
        );
        assert!(profile_machine_context_recorded("cpu", "cpu-rust", false, &cpu_features, true));
    }

    #[test]
    fn apple_profile_receipt_keeps_apple_claim_scope() {
        let cpu_features = Vec::new();

        assert_eq!(
            profile_claim_scope("metal", "apple-m4-metal"),
            "selected Apple backend phase timing only"
        );
        assert!(profile_machine_context_recorded(
            "metal",
            "apple-m4-metal",
            true,
            &cpu_features,
            false
        ));
    }

    #[test]
    fn allocation_samples_json_records_counter_deltas() {
        let summary = allocation_samples_json(&[
            AllocationAuditSnapshot {
                alloc_count: 3,
                alloc_bytes: 128,
                dealloc_count: 1,
                dealloc_bytes: 32,
            },
            AllocationAuditSnapshot {
                alloc_count: 1,
                alloc_bytes: 64,
                dealloc_count: 1,
                dealloc_bytes: 96,
            },
        ]);

        assert_eq!(summary["count"], 2);
        assert_eq!(summary["alloc_count_total"], 4);
        assert_eq!(summary["alloc_bytes_total"], 192);
        assert_eq!(summary["dealloc_count_total"], 2);
        assert_eq!(summary["dealloc_bytes_total"], 128);
        assert_eq!(summary["net_bytes_total"], 64);
        assert_eq!(summary["mean_alloc_count_per_token"], 2.0);
        assert_eq!(summary["mean_alloc_bytes_per_token"], 96.0);
        assert_eq!(summary["max_alloc_count_per_token"], 3);
        assert_eq!(summary["max_alloc_bytes_per_token"], 128);
    }

    #[test]
    fn allocation_delta_helpers_record_per_token_means() {
        let samples = [
            AllocationAuditSnapshot {
                alloc_count: 2,
                alloc_bytes: 80,
                dealloc_count: 0,
                dealloc_bytes: 0,
            },
            AllocationAuditSnapshot {
                alloc_count: 4,
                alloc_bytes: 160,
                dealloc_count: 0,
                dealloc_bytes: 0,
            },
        ];

        let count = allocation_count_delta_json(&samples);
        let bytes = allocation_bytes_delta_json(&samples);

        assert_eq!(count["total"], 6);
        assert_eq!(count["mean_per_token"], 3.0);
        assert_eq!(count["max_per_token"], 4);
        assert_eq!(bytes["total"], 240);
        assert_eq!(bytes["mean_per_token"], 120.0);
        assert_eq!(bytes["max_per_token"], 160);
    }

    #[test]
    fn allocation_audit_requires_selected_apple_cpu_neon_without_fallback() {
        assert!(allocation_audit_backend_supported(&RunBackendIdentity {
            requested_backend: "apple-m4-cpu-neon".to_string(),
            selected_backend: "apple-m4-cpu-neon".to_string(),
            runtime_api: "cpu".to_string(),
            fallback_used: false,
            fallback_reason: None,
        }));

        assert!(!allocation_audit_backend_supported(&RunBackendIdentity {
            requested_backend: "apple-m4-cpu-neon".to_string(),
            selected_backend: "cpu".to_string(),
            runtime_api: "cpu".to_string(),
            fallback_used: true,
            fallback_reason: Some("Apple CPU/NEON unavailable".to_string()),
        }));
        assert!(!allocation_audit_backend_supported(&RunBackendIdentity {
            requested_backend: "cpu".to_string(),
            selected_backend: "cpu-rust".to_string(),
            runtime_api: "cpu".to_string(),
            fallback_used: false,
            fallback_reason: None,
        }));
    }

    #[test]
    fn lunar_lake_probe_receipt_is_visibility_only() {
        let probe = bitnet_device_probe::probe_lnl258v_platform();
        let receipt = build_lunar_lake_probe_receipt(
            probe,
            "2026-05-06T00:00:00Z".to_string(),
            Some("ci/hardware/intel-258v/2026-05-06/platform-probe.json".to_string()),
        );

        assert_eq!(receipt["artifact_kind"], "lnl258v_platform_probe");
        assert_eq!(receipt["hardware_lane"], "core-ultra-7-258v");
        assert_eq!(receipt["proof_stage"], "runtime_detected");
        assert_eq!(receipt["runtime_api"], "platform_probe");
        assert_eq!(receipt["kernel_execution"], false);
        assert_eq!(receipt["graph_execution"], false);
        assert_eq!(receipt["bitnet_inference"], false);
        assert_eq!(receipt["fallback_used"], false);
        assert!(receipt["platform"]["cpu"]["has_avx512"].is_boolean());
    }

    #[test]
    fn cpu258v_validation_blocks_missing_model_before_inference() {
        let missing_model = std::env::temp_dir()
            .join(format!("bitnet-missing-{}-ggml-model-i2_s.gguf", std::process::id()));
        let receipt = build_cpu_bitnet_validation_receipt(CpuBitnetValidationReceiptInput {
            machine: "intel-258v".to_string(),
            model: missing_model,
            tokenizer: None,
            backend: "cpu".to_string(),
            strict: true,
            max_tokens: 1,
            platform_artifact: None,
            json_out: None,
            timestamp_utc: "2026-05-06T00:00:00Z".to_string(),
        });

        assert_eq!(receipt["artifact_kind"], "cpu-bitnet-validation");
        assert_eq!(receipt["hardware_lane"], "intel-258v-cpu-avx2");
        assert_eq!(receipt["proof_stage"], "blocked_preflight");
        assert_eq!(receipt["status"], "blocked_missing_canonical_model");
        assert_eq!(receipt["blocked_before_inference"], true);
        assert_eq!(receipt["kernel_execution"], false);
        assert_eq!(receipt["bitnet_inference"], false);
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["blocker"]["stage"], "load_model");
    }

    #[test]
    fn cpu258v_validation_rejects_accelerator_backend() {
        let receipt = build_cpu_bitnet_validation_receipt(CpuBitnetValidationReceiptInput {
            machine: "intel-258v".to_string(),
            model: std::path::PathBuf::from("models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"),
            tokenizer: None,
            backend: "intel-npu".to_string(),
            strict: true,
            max_tokens: 1,
            platform_artifact: None,
            json_out: None,
            timestamp_utc: "2026-05-06T00:00:00Z".to_string(),
        });

        assert_eq!(receipt["status"], "blocked_wrong_backend");
        assert_eq!(receipt["blocker"]["stage"], "backend_selection");
        assert_eq!(receipt["hardware"]["requested_backend"], "intel-npu");
        assert_eq!(receipt["bitnet_inference"], false);
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
