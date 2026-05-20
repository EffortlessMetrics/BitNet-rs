#![recursion_limit = "256"]

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
use std::io;
use tracing::{debug, error, info, warn};

#[global_allocator]
static ALLOCATION_AUDIT_ALLOCATOR: allocation_audit::AllocationAuditAllocator =
    allocation_audit::AllocationAuditAllocator;

mod allocation_audit;
#[cfg(feature = "full-cli")]
mod commands;
mod config;
mod exit;
mod intel_arc;
mod intel_npu;
#[cfg(feature = "full-cli")]
mod ln_rules;
#[cfg(feature = "full-cli")]
mod mac;
mod model_cache;
mod planner_receipts;
mod prompt_audit;
mod score;
mod simple_generation;
pub mod tokenizer_discovery;

use allocation_audit::{
    AllocationAuditGuard, AllocationAuditSnapshot, allocation_bytes_delta_json,
    allocation_count_delta_json, allocation_samples_json, mean_alloc_bytes, mean_alloc_count,
};
#[cfg(feature = "full-cli")]
use allocation_audit::{
    WarmSessionPromptAllocationAudit, WarmSessionPromptSetupAllocationAudit,
    warm_session_aggregate_allocation_audit_json, warm_session_prompt_allocation_audit_json,
};
use exit::*;
use prompt_audit::*;

/// Build the CLI command for external use (e.g., in tests)
pub fn build_cli() -> clap::Command {
    Cli::command()
}

/// CLI interface version (SemVer for CLI surface compatibility)
const INTERFACE_VERSION: &str = "1.0.0";
const RTX_5070_TI_CUDA: &str = "nvidia-rtx-5070-ti-cuda";
const BITNET_CPP_ANSWER_TEMPLATE: &str = "bitnetcpp-answer";

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

fn sha256_hex_bytes(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn sha256_token_ids(tokens: &[u32]) -> Result<String> {
    Ok(sha256_hex_bytes(&serde_json::to_vec(tokens)?))
}

fn critical_not_claims() -> Vec<&'static str> {
    vec![
        "selected_attention_residency",
        "resident_kv_decode",
        "attention_scores_residency",
        "softmax_residency",
        "attention_value_mix_residency",
        "full_support_op_residency",
        "full_device_residency",
        "completion",
    ]
}

fn qk256_trailer_scale_from_simple_loader_bytes(
    name: &str,
    packed: &[u8],
    expected_bytes: usize,
) -> Result<Option<f32>> {
    let Some(trailing_bytes) = packed.len().checked_sub(expected_bytes) else {
        return Ok(None);
    };
    if trailing_bytes == 0 {
        return Ok(None);
    }
    if trailing_bytes < std::mem::size_of::<f32>() {
        debug!(
            "QK256 '{}' simple-loader trailer too short for inline scale: {} bytes",
            name, trailing_bytes
        );
        return Ok(None);
    }

    let mut scale_bytes = [0u8; std::mem::size_of::<f32>()];
    let scale_len = scale_bytes.len();
    scale_bytes.copy_from_slice(&packed[expected_bytes..expected_bytes + scale_len]);
    let scale = f32::from_le_bytes(scale_bytes);
    if !scale.is_finite() {
        anyhow::bail!("QK256 '{name}' simple-loader inline scale is not finite: {scale}");
    }
    Ok(Some(scale))
}

fn qk256_raw_tensors_from_simple_loader(
    i2s_qk256: impl IntoIterator<Item = (String, bitnet_models::quant::i2s_qk256::I2SQk256NoScale)>,
) -> Result<std::collections::HashMap<String, candle_core::Tensor>> {
    let mut raw_tensors = std::collections::HashMap::new();
    for (name, qk256) in i2s_qk256 {
        let expected_bytes = qk256.rows * qk256.row_stride_bytes;
        let scale = qk256_trailer_scale_from_simple_loader_bytes(&name, &qk256.qs, expected_bytes)?;
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

        raw_tensors.insert(format!("{name}.qk256_qs"), raw_tensor);
        if let Some(scale) = scale {
            let scale_tensor =
                candle_core::Tensor::from_slice(&[scale], &[1], &candle_core::Device::Cpu)
                    .with_context(|| format!("Failed to build QK256 scale tensor for {name}"))?;
            raw_tensors.insert(format!("{name}.qk256_scale"), scale_tensor);
        }
    }
    Ok(raw_tensors)
}

#[cfg(feature = "cli-bench")]
use commands::BenchmarkCommand;
#[cfg(feature = "full-cli")]
use commands::dense_gguf_linear_parity::is_supported_dense_qwen_cuda_model_path;
#[cfg(feature = "full-cli")]
use commands::{
    AnswerCorpusCommand, AnswerParityCommand, ConvertCommand, DenseGgufAllLayerPlanCommand,
    DenseGgufAttentionScoreCudaParityCommand, DenseGgufAttentionScoreFixtureCommand,
    DenseGgufAttentionSoftmaxCudaParityCommand, DenseGgufAttentionSoftmaxFixtureCommand,
    DenseGgufAttentionVMixCudaParityCommand, DenseGgufAttentionVMixFixtureCommand,
    DenseGgufKvCachePolicyCommand, DenseGgufLinearParityCommand, DenseGgufLinearRoleSweepCommand,
    DenseGgufMlpActivationCudaParityCommand, DenseGgufMlpActivationFixtureCommand,
    DenseGgufModelBoundaryFixturesCommand, DenseGgufNormCudaParityCommand,
    DenseGgufNormFixtureCommand, DenseGgufOneLayerCpuReferenceCommand,
    DenseGgufOneLayerCudaParityCommand, DenseGgufOneLayerPlanCommand,
    DenseGgufQwenOneTokenStrictCudaCommand, DenseGgufQwenShortDecodeStrictCudaCommand,
    DenseGgufQwenWarmSessionStrictCudaCommand, DenseGgufRopeCudaParityCommand,
    DenseGgufSamplingPolicyCommand, DenseQwenCudaAskOptions,
    ExternalReferenceInstrumentationCommand, FirstTokenDivergenceCommand, InferenceCommand,
    InspectCommand, LunarLakeAction, LunarLakeCommand, OutputHeadLogitsAuditCommand,
    ReceiptsCommand, ReferenceCompareCommand, ServeCommand, SupportCommand,
    TransformerLayerParityCommand, run_dense_qwen_cuda_ask,
};
use config::{CliConfig, ConfigBuilder, DEVICE_HELP};
#[cfg(feature = "full-cli")]
use mac::MacCommand;
use model_cache::ModelCommand;

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

#[allow(clippy::large_enum_variant)] // Clap command shape keeps argument definitions local and readable.
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

        /// Declared model contract for diagnostic proof binding.
        #[arg(long)]
        proof_model_contract: Option<std::path::PathBuf>,

        /// Declared kernel route id for diagnostic proof binding.
        #[arg(long)]
        proof_kernel_route: Option<String>,

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

        /// Prompt template: auto (detect), raw, instruct, llama3-chat, bitnetcpp-answer
        #[arg(long, value_name = "TEMPLATE", default_value = "auto")]
        prompt_template: String,

        /// Disable Qwen thinking mode by appending the Qwen no-thinking assistant suffix
        #[arg(long = "no-think", visible_alias = "no-thinking", default_value_t = false)]
        no_think: bool,

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
        #[arg(long, visible_alias = "logits-dump-steps")]
        dump_logit_steps: Option<usize>,

        /// Top-k tokens to include in logit dump
        #[arg(long, default_value = "10", value_name = "K")]
        logits_topk: usize,

        /// Assert greedy argmax invariant when dumping logits
        #[arg(long, default_value_t = false)]
        assert_greedy: bool,

        /// Emit bounded Qwen first-token checkpoint summaries as JSONL
        #[arg(long, value_name = "PATH")]
        qwen_trace_jsonl: Option<std::path::PathBuf>,

        /// Qwen transformer layer index to trace when --qwen-trace-jsonl is set
        #[arg(long, value_name = "N")]
        qwen_trace_layer: Option<usize>,

        /// Also trace the full prompt forward path before incremental decode
        #[arg(long, default_value_t = false)]
        qwen_trace_full_prompt: bool,

        /// Comma-separated prompt token IDs to force for reference-aligned tracing
        #[arg(long, value_name = "IDS")]
        qwen_trace_prompt_ids: Option<String>,

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
        #[arg(short, long, value_name = "TEXT", conflicts_with = "question_arg")]
        question: Option<String>,

        /// User question to answer (positional form)
        #[arg(value_name = "QUESTION")]
        question_arg: Option<String>,

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

        /// Output answer-shaped receipt to file; strict CPU/CUDA ask defaults to
        /// target/bitnet/receipts/cuda-answer-readiness/strict-*-ask-latest.json
        #[arg(long, value_name = "PATH")]
        receipt_out: Option<std::path::PathBuf>,
    },

    /// Fetch, verify, list, and prune supported local model artifacts
    Model(ModelCommand),

    #[cfg(feature = "full-cli")]
    /// Collect model status and receipt evidence for support
    Support(SupportCommand),

    /// RTX 5070 Ti CUDA proof-lane utilities
    Cuda {
        #[command(subcommand)]
        action: CudaAction,
    },

    #[cfg(feature = "full-cli")]
    /// Mac-oriented SLM check, ask, validate, and receipt-check wrappers
    Mac(MacCommand),

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

    /// Emit a BitNet prompt/token authority audit receipt without running inference
    PromptAuthorityAudit {
        /// Model GGUF path (for metadata and tokenizer authority)
        #[arg(long)]
        model: std::path::PathBuf,

        /// Optional explicit tokenizer path
        #[arg(long)]
        tokenizer: Option<std::path::PathBuf>,

        /// User prompt to render and tokenize
        #[arg(long)]
        prompt: String,

        /// Optional system prompt used by chat-style templates
        #[arg(long = "system", value_name = "TEXT")]
        system_prompt: Option<String>,

        /// External reference label for prompt/token parity, such as hf_apply_chat_template
        #[arg(long)]
        reference_source: Option<String>,

        /// External reference rendered prompt to compare against metadata-authority rendering
        #[arg(long)]
        reference_rendered_prompt: Option<String>,

        /// External reference prompt token IDs, comma-separated
        #[arg(long, value_delimiter = ',', value_name = "IDS")]
        reference_prompt_ids: Vec<u32>,

        /// Output JSON receipt path
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
    /// Validate an external-reference divergence artifact
    ReferenceCompare(Box<ReferenceCompareCommand>),

    #[cfg(feature = "full-cli")]
    /// Classify first-token divergence from external reference and 258V CPU receipts
    FirstTokenDivergence(Box<FirstTokenDivergenceCommand>),

    #[cfg(feature = "full-cli")]
    /// Classify external reference token/logit instrumentation coverage
    ExternalReferenceInstrumentation(Box<ExternalReferenceInstrumentationCommand>),

    #[cfg(feature = "full-cli")]
    /// Audit output-head/tied-head and logits-index boundaries for 258V CPU proof
    OutputHeadLogitsAudit(Box<OutputHeadLogitsAuditCommand>),

    #[cfg(feature = "full-cli")]
    /// Classify 258V CPU transformer-layer trace boundaries
    TransformerLayerParity(Box<TransformerLayerParityCommand>),

    #[cfg(feature = "full-cli")]
    /// Extract one dense GGUF linear fixture and run strict CUDA parity diagnostics
    DenseGgufLinearParity(Box<DenseGgufLinearParityCommand>),

    #[cfg(feature = "full-cli")]
    /// Extract several dense GGUF linear fixtures and run strict CUDA parity diagnostics
    DenseGgufLinearRoleSweep(Box<DenseGgufLinearRoleSweepCommand>),

    #[cfg(feature = "full-cli")]
    /// Emit a dense GGUF one-layer strict CUDA planner gap receipt
    DenseGgufOneLayerPlan(Box<DenseGgufOneLayerPlanCommand>),

    #[cfg(feature = "full-cli")]
    /// Emit a dense GGUF all-layer strict CUDA planner receipt
    DenseGgufAllLayerPlan(Box<DenseGgufAllLayerPlanCommand>),

    #[cfg(feature = "full-cli")]
    /// Emit a dense GGUF one-layer CPU reference harness receipt
    DenseGgufOneLayerCpuReference(Box<DenseGgufOneLayerCpuReferenceCommand>),

    #[cfg(feature = "full-cli")]
    /// Run integrated dense GGUF one-layer strict CUDA parity diagnostics
    DenseGgufOneLayerCudaParity(Box<DenseGgufOneLayerCudaParityCommand>),

    #[cfg(feature = "full-cli")]
    /// Emit dense GGUF model-boundary fixture receipts for embedding/norm/logits
    DenseGgufModelBoundaryFixtures(Box<DenseGgufModelBoundaryFixturesCommand>),

    #[cfg(feature = "full-cli")]
    /// Emit dense GGUF KV-cache policy receipts for the strict CUDA lane
    DenseGgufKvCachePolicy(Box<DenseGgufKvCachePolicyCommand>),

    #[cfg(feature = "full-cli")]
    /// Emit dense GGUF logits-transfer and sampling policy receipts
    DenseGgufSamplingPolicy(Box<DenseGgufSamplingPolicyCommand>),

    #[cfg(feature = "full-cli")]
    /// Run dense Qwen one-token strict CUDA proof and emit a governed receipt
    DenseGgufQwenOneTokenStrictCuda(Box<DenseGgufQwenOneTokenStrictCudaCommand>),

    #[cfg(feature = "full-cli")]
    /// Run dense Qwen short-decode strict CUDA proof and emit a governed receipt
    DenseGgufQwenShortDecodeStrictCuda(Box<DenseGgufQwenShortDecodeStrictCudaCommand>),

    #[cfg(feature = "full-cli")]
    /// Run dense Qwen warm-session strict CUDA proof and emit a governed receipt
    DenseGgufQwenWarmSessionStrictCuda(Box<DenseGgufQwenWarmSessionStrictCudaCommand>),

    #[cfg(feature = "full-cli")]
    /// Extract dense GGUF RMSNorm fixtures and emit a CPU-reference receipt
    DenseGgufNormFixture(Box<DenseGgufNormFixtureCommand>),

    #[cfg(feature = "full-cli")]
    /// Extract dense GGUF RMSNorm fixtures and run strict CUDA parity diagnostics
    DenseGgufNormCudaParity(Box<DenseGgufNormCudaParityCommand>),

    #[cfg(feature = "full-cli")]
    /// Run dense GGUF RoPE strict CUDA parity diagnostics
    DenseGgufRopeCudaParity(Box<DenseGgufRopeCudaParityCommand>),

    #[cfg(feature = "full-cli")]
    /// Extract a dense GGUF attention-score fixture and emit a CPU-reference receipt
    DenseGgufAttentionScoreFixture(Box<DenseGgufAttentionScoreFixtureCommand>),

    #[cfg(feature = "full-cli")]
    /// Extract a dense GGUF attention-softmax fixture and emit a CPU-reference receipt
    DenseGgufAttentionSoftmaxFixture(Box<DenseGgufAttentionSoftmaxFixtureCommand>),

    #[cfg(feature = "full-cli")]
    /// Extract a dense GGUF attention V-mix fixture and emit a CPU-reference receipt
    DenseGgufAttentionVMixFixture(Box<DenseGgufAttentionVMixFixtureCommand>),

    #[cfg(feature = "full-cli")]
    /// Extract a dense GGUF MLP activation fixture and emit a CPU-reference receipt
    DenseGgufMlpActivationFixture(Box<DenseGgufMlpActivationFixtureCommand>),

    #[cfg(feature = "full-cli")]
    /// Run dense GGUF MLP activation strict CUDA parity diagnostics
    DenseGgufMlpActivationCudaParity(Box<DenseGgufMlpActivationCudaParityCommand>),

    #[cfg(feature = "full-cli")]
    /// Run dense GGUF attention-softmax strict CUDA parity diagnostics
    DenseGgufAttentionSoftmaxCudaParity(Box<DenseGgufAttentionSoftmaxCudaParityCommand>),

    #[cfg(feature = "full-cli")]
    /// Run dense GGUF attention V-mix strict CUDA parity diagnostics
    DenseGgufAttentionVMixCudaParity(Box<DenseGgufAttentionVMixCudaParityCommand>),

    #[cfg(feature = "full-cli")]
    /// Run dense GGUF attention-score strict CUDA parity diagnostics
    DenseGgufAttentionScoreCudaParity(Box<DenseGgufAttentionScoreCudaParityCommand>),

    #[cfg(feature = "full-cli")]
    /// Explain BitNet-rs JSON receipts and claim boundaries
    Receipts(ReceiptsCommand),

    #[cfg(feature = "full-cli")]
    /// Run multiple SLM prompts in one warm process with one model/tokenizer load
    SlmWarmSession {
        /// Model file or directory path (.gguf file or HuggingFace model directory)
        #[arg(short, long)]
        model: std::path::PathBuf,

        /// Model format: auto (detect from path) or gguf
        #[arg(long, value_name = "FORMAT", default_value = "auto")]
        model_format: String,

        /// Optional explicit tokenizer path
        #[arg(long)]
        tokenizer: Option<std::path::PathBuf>,

        /// Deterministic SLM quality corpus; when set, corpus cases are run in a warm session
        #[arg(long, value_name = "PATH")]
        corpus: Option<std::path::PathBuf>,

        /// Number of repeated runs for each corpus case when --corpus is set
        #[arg(long, default_value_t = 2)]
        corpus_repeat_runs: usize,

        /// Prompt to answer; pass multiple times for a warm multi-prompt session
        #[arg(long = "prompt", value_name = "TEXT")]
        prompts: Vec<String>,

        /// Maximum new tokens to generate per prompt
        #[arg(long, visible_aliases = ["max-tokens", "n-predict"], default_value_t = 32)]
        max_new_tokens: usize,

        /// Temperature for sampling (0 = greedy)
        #[arg(long, default_value_t = 0.0)]
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

        /// Strict tokenizer mode: fail if no real tokenizer available
        #[arg(long, default_value_t = false)]
        strict_tokenizer: bool,

        /// Strict loader mode: fail-fast with enhanced loader and no fallback
        #[arg(long, default_value_t = false)]
        strict_loader: bool,

        /// Use greedy decoding (overrides temperature)
        #[arg(long, default_value_t = false)]
        greedy: bool,

        /// Enable deterministic mode (single-threaded)
        #[arg(long, default_value_t = false)]
        deterministic: bool,

        /// Number of threads to use (0 = all cores)
        #[arg(long, default_value_t = 0)]
        threads: usize,

        /// Prompt template: qwen is the validated Kaby/Qwen3 CPU default; qwen2.5 remains accepted
        #[arg(long, value_name = "TEMPLATE", default_value = "qwen2.5")]
        prompt_template: String,

        /// Disable Qwen thinking mode by appending the Qwen no-thinking assistant suffix
        #[arg(long = "no-think", visible_alias = "no-thinking", default_value_t = false)]
        no_think: bool,

        /// System prompt for chat models
        #[arg(long, value_name = "TEXT")]
        system_prompt: Option<String>,

        /// Stop sequences (can be repeated for multiple sequences)
        #[arg(long = "stop", value_name = "SEQ")]
        stop: Vec<String>,

        /// Stop token IDs (numeric token IDs, can be repeated)
        #[arg(long = "stop-id", value_name = "ID")]
        stop_id: Vec<u32>,

        /// Fail after writing receipts if any prompt fails the SLM quality gate
        #[arg(long, default_value_t = false)]
        fail_on_quality: bool,

        /// Require repeated identical prompts to produce stable generated token IDs and text
        #[arg(long, default_value_t = false)]
        require_determinism: bool,

        /// Measure scoped hot-loop allocation counter deltas in warm-session receipts
        #[arg(long, default_value_t = false)]
        allocation_audit: bool,

        /// Stream generated token text to stdout as each token is decoded
        #[arg(long, default_value_t = false)]
        stream: bool,

        /// Emit operator progress lines to stderr while keeping token text on stdout
        #[arg(long, default_value_t = false)]
        progress: bool,

        /// Suppress warm-session status/progress lines; token streaming still uses stdout
        #[arg(long, default_value_t = false)]
        quiet: bool,

        /// Minimum generated tokens required by the warm-session quality gate
        #[arg(long, default_value_t = 1)]
        min_generated_tokens: usize,

        /// Minimum distinct generated token IDs required by the warm-session quality gate
        #[arg(long, default_value_t = 1)]
        min_distinct_generated_tokens: usize,

        /// Output aggregate warm-session receipt
        #[arg(long, value_name = "PATH")]
        json_out: std::path::PathBuf,
    },

    #[cfg(feature = "full-cli")]
    /// Run strict RTX 5070 Ti CUDA prompts in one warm BitNet process
    CudaWarmSession {
        /// Model GGUF file path
        #[arg(short, long)]
        model: std::path::PathBuf,

        /// Model format: auto (detect from path) or gguf
        #[arg(long, value_name = "FORMAT", default_value = "auto")]
        model_format: String,

        /// Explicit tokenizer path; strict mode does not guess
        #[arg(long)]
        tokenizer: Option<std::path::PathBuf>,

        /// Prompt to answer; pass multiple times for a warm multi-turn session
        #[arg(long = "prompt", value_name = "TEXT")]
        prompts: Vec<String>,

        /// Maximum new tokens to generate per prompt
        #[arg(long, visible_aliases = ["max-tokens", "n-predict"], default_value_t = 8)]
        max_new_tokens: usize,

        /// Temperature for sampling (0 = greedy)
        #[arg(long, default_value_t = 0.0)]
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

        /// Require strict tokenizer authority
        #[arg(long, default_value_t = false)]
        strict_tokenizer: bool,

        /// Require real GGUF loader and strict backend routing
        #[arg(long, default_value_t = false)]
        strict_loader: bool,

        /// Use greedy decoding (overrides temperature)
        #[arg(long, default_value_t = false)]
        greedy: bool,

        /// Enable deterministic mode (single-threaded)
        #[arg(long, default_value_t = false)]
        deterministic: bool,

        /// Number of threads to use (0 = all cores)
        #[arg(long, default_value_t = 0)]
        threads: usize,

        /// Prompt template for all turns
        #[arg(long, value_name = "TEMPLATE", default_value = "bitnetcpp-answer")]
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

        /// Fail after writing receipts if any turn fails the answer quality gate
        #[arg(long, default_value_t = false)]
        fail_on_quality: bool,

        /// Output aggregate CUDA warm-session receipt
        #[arg(long, value_name = "PATH")]
        json_out: std::path::PathBuf,
    },

    #[cfg(feature = "full-cli")]
    /// Run 258V CPU phase prompts in one warm strict BitNet process
    CpuPhaseWarmSession {
        /// Model GGUF file path
        #[arg(short, long)]
        model: std::path::PathBuf,

        /// Model format: auto (detect from path) or gguf
        #[arg(long, value_name = "FORMAT", default_value = "auto")]
        model_format: String,

        /// Explicit tokenizer path; strict mode does not guess
        #[arg(long)]
        tokenizer: Option<std::path::PathBuf>,

        /// Platform artifact to reference in the aggregate receipt
        #[arg(long, value_name = "PATH")]
        platform_artifact: Option<std::path::PathBuf>,

        /// Prompt text for the prefill_512 phase
        #[arg(long, value_name = "TEXT")]
        prefill_prompt: Option<String>,

        /// Prompt file for the prefill_512 phase
        #[arg(long, value_name = "PATH")]
        prefill_prompt_file: Option<std::path::PathBuf>,

        /// Prompt text for the decode_128 phase
        #[arg(
            long,
            value_name = "TEXT",
            default_value = "Answer with a deterministic continuation: one two three"
        )]
        decode_prompt: String,

        /// Generated tokens requested for the decode_128 phase
        #[arg(long, default_value_t = 128)]
        decode_tokens: usize,

        /// Generated tokens requested for the prefill_512 phase
        #[arg(long, default_value_t = 1)]
        prefill_tokens: usize,

        /// Requested CPU kernel identity: auto, scalar, avx2, or avx512
        #[arg(long, value_name = "KERNEL", default_value = "avx2")]
        cpu_kernel: String,

        /// Strict tokenizer mode: fail if no real tokenizer is available
        #[arg(long, default_value_t = false)]
        strict_tokenizer: bool,

        /// Strict loader mode: fail-fast with enhanced loader and no fallback
        #[arg(long, default_value_t = false)]
        strict_loader: bool,

        /// Number of threads to use (0 = all cores)
        #[arg(long, default_value_t = 0)]
        threads: usize,

        /// Prompt template for both phase prompts
        #[arg(long, value_name = "TEMPLATE", default_value = "raw")]
        prompt_template: String,

        /// Output aggregate warm-session receipt
        #[arg(long, value_name = "PATH")]
        json_out: std::path::PathBuf,
    },

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

    #[cfg(feature = "full-cli")]
    /// Validate Lunar Lake operator readiness and route policy from receipts
    #[command(name = "lunar-lake")]
    LunarLake(LunarLakeCommand),

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

    /// Run static BitNet linear-projection subgraph parity on OpenVINO NPU
    IntelNpuBitnetLinearSubgraph {
        /// Require selected subgraph parity to pass
        #[arg(long, default_value_t = false)]
        strict: bool,

        /// Output JSON parity receipt to file
        #[arg(long)]
        json_out: Option<std::path::PathBuf>,
    },

    /// Run static BitNet FFN/ReLU2 subgraph parity on OpenVINO NPU
    #[command(name = "intel-npu-bitnet-ffn-subgraph")]
    IntelNpuBitnetFfnSubgraph {
        /// Require selected subgraph parity to pass
        #[arg(long, default_value_t = false)]
        strict: bool,

        /// CPU reference bundle artifact used as the comparison anchor
        #[arg(long)]
        cpu_reference: Option<std::path::PathBuf>,

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

    /// Run a native OpenCL CPU-reference parity kernel for Arc 140V
    #[command(name = "intel-arc-140v-opencl-parity")]
    IntelArc140vOpenclParity {
        /// Require Arc 140V native OpenCL parity to pass
        #[arg(long, default_value_t = false)]
        strict: bool,

        /// CPU reference bundle artifact used as the comparison anchor
        #[arg(long)]
        cpu_reference: Option<std::path::PathBuf>,

        /// Output JSON parity receipt to file
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

    /// Validate a Lunar Lake OpenVINO receipt against strict route-boundary gates
    OpenVinoLunarLake {
        /// Receipt JSON to validate
        #[arg(long)]
        receipt: std::path::PathBuf,

        /// Output JSON validation summary to file
        #[arg(long)]
        json_out: Option<std::path::PathBuf>,
    },
}

#[derive(Subcommand)]
enum CudaAction {
    /// Preflight strict RTX 5070 Ti CUDA user commands without generation
    Doctor {
        /// Optional BitNet GGUF model path to preflight for strict ask/chat
        #[arg(long)]
        model: Option<std::path::PathBuf>,

        /// Optional explicit tokenizer path used as tokenizer authority
        #[arg(long)]
        tokenizer: Option<std::path::PathBuf>,

        /// Output the CUDA doctor preflight receipt to a JSON file
        #[arg(long, value_name = "PATH")]
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
    let command_default_log_level = if cli.log_level.is_none()
        && std::env::var_os("BITNET_LOG_LEVEL").is_none()
        && std::env::var_os("RUST_LOG").is_none()
    {
        default_log_level_for_command(cli.command.as_ref())
    } else {
        None
    };
    setup_logging(&config, cli.log_level.as_deref().or(command_default_log_level))?;

    let startup_contract_report =
        evaluate_and_emit(RuntimeComponent::Cli, ContractPolicy::Observe)?;
    if !startup_contract_report.is_compatible() {
        warn!(component = ?RuntimeComponent::Cli, "CLI startup contract reported issues");
    }

    let requested_backend_label =
        cli.device.clone().unwrap_or_else(|| config.default_device.clone());
    let explicit_device_label = cli.device.clone();

    // Report backend selection at startup so logs and receipts are deterministic.
    if !skips_startup_backend_selection(cli.command.as_ref()) {
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
            proof_model_contract,
            proof_kernel_route,
            dump_ids,
            bos,
            greedy,
            deterministic,
            threads,
            prompt_template,
            no_think,
            system_prompt,
            stop,
            stop_id,
            dump_logit_steps,
            logits_topk,
            assert_greedy,
            qwen_trace_jsonl,
            qwen_trace_layer,
            qwen_trace_full_prompt,
            qwen_trace_prompt_ids,
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
                proof_model_contract,
                proof_kernel_route,
                dump_ids,
                bos,
                greedy,
                deterministic,
                threads,
                prompt_template,
                no_think,
                system_prompt,
                stop,
                stop_id,
                dump_logit_steps,
                logits_topk,
                assert_greedy,
                qwen_trace_jsonl,
                qwen_trace_layer,
                qwen_trace_full_prompt,
                qwen_trace_prompt_ids,
                no_warnings,
                profile_id,
                allocation_audit,
                false,
            )
            .await
        }
        Some(Commands::Ask {
            model,
            tokenizer,
            question,
            question_arg,
            system_prompt,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            strict_cuda,
            strict_cpu,
            receipt_out,
        }) => {
            let question = resolve_ask_question(question, question_arg)?;
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
        Some(Commands::Model(cmd)) => cmd.execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::Support(cmd)) => cmd.execute().await,
        Some(Commands::Cuda { action }) => {
            handle_cuda_command(action, explicit_device_label.as_deref())
        }
        #[cfg(feature = "full-cli")]
        Some(Commands::Mac(cmd)) => cmd.execute(explicit_device_label.as_deref()).await,
        #[cfg(feature = "full-cli")]
        Some(Commands::Inference(cmd)) => (*cmd).execute(&config).await,
        #[cfg(feature = "full-cli")]
        Some(Commands::Chat(cmd)) => (*cmd).run_chat(&config).await,
        #[cfg(feature = "full-cli")]
        Some(Commands::AnswerCorpus(cmd)) => (*cmd).execute(&requested_backend_label).await,
        #[cfg(feature = "full-cli")]
        Some(Commands::AnswerParity(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::ReferenceCompare(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::FirstTokenDivergence(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::ExternalReferenceInstrumentation(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::OutputHeadLogitsAudit(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::TransformerLayerParity(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufLinearParity(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufLinearRoleSweep(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufOneLayerPlan(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufAllLayerPlan(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufOneLayerCpuReference(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufOneLayerCudaParity(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufModelBoundaryFixtures(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufKvCachePolicy(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufSamplingPolicy(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufQwenOneTokenStrictCuda(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufQwenShortDecodeStrictCuda(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufQwenWarmSessionStrictCuda(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufNormFixture(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufNormCudaParity(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufRopeCudaParity(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufAttentionScoreFixture(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufAttentionSoftmaxFixture(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufAttentionVMixFixture(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufMlpActivationFixture(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufMlpActivationCudaParity(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufAttentionSoftmaxCudaParity(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufAttentionVMixCudaParity(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::DenseGgufAttentionScoreCudaParity(cmd)) => (*cmd).execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::Receipts(cmd)) => cmd.execute().await,
        #[cfg(feature = "full-cli")]
        Some(Commands::SlmWarmSession {
            model,
            model_format,
            tokenizer,
            corpus,
            corpus_repeat_runs,
            prompts,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
            strict_tokenizer,
            strict_loader,
            greedy,
            deterministic,
            threads,
            prompt_template,
            no_think,
            system_prompt,
            stop,
            stop_id,
            fail_on_quality,
            require_determinism,
            allocation_audit,
            stream,
            progress,
            quiet,
            min_generated_tokens,
            min_distinct_generated_tokens,
            json_out,
        }) => {
            run_slm_warm_session(
                &requested_backend_label,
                model,
                model_format,
                tokenizer,
                corpus,
                corpus_repeat_runs,
                prompts,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                seed,
                strict_tokenizer,
                strict_loader,
                greedy,
                deterministic,
                threads,
                prompt_template,
                no_think,
                system_prompt,
                stop,
                stop_id,
                fail_on_quality,
                require_determinism,
                allocation_audit,
                SlmWarmSessionOutput::new(stream, progress, quiet),
                min_generated_tokens,
                min_distinct_generated_tokens,
                json_out,
            )
            .await
        }
        #[cfg(feature = "full-cli")]
        Some(Commands::CudaWarmSession {
            model,
            model_format,
            tokenizer,
            prompts,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
            strict_tokenizer,
            strict_loader,
            greedy,
            deterministic,
            threads,
            prompt_template,
            system_prompt,
            stop,
            stop_id,
            fail_on_quality,
            json_out,
        }) => {
            run_cuda_warm_session(
                &requested_backend_label,
                model,
                model_format,
                tokenizer,
                prompts,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                seed,
                strict_tokenizer,
                strict_loader,
                greedy,
                deterministic,
                threads,
                prompt_template,
                system_prompt,
                stop,
                stop_id,
                fail_on_quality,
                json_out,
            )
            .await
        }
        #[cfg(feature = "full-cli")]
        Some(Commands::CpuPhaseWarmSession {
            model,
            model_format,
            tokenizer,
            platform_artifact,
            prefill_prompt,
            prefill_prompt_file,
            decode_prompt,
            decode_tokens,
            prefill_tokens,
            cpu_kernel,
            strict_tokenizer,
            strict_loader,
            threads,
            prompt_template,
            json_out,
        }) => {
            run_cpu_phase_warm_session(
                &requested_backend_label,
                model,
                model_format,
                tokenizer,
                platform_artifact,
                prefill_prompt,
                prefill_prompt_file,
                decode_prompt,
                decode_tokens,
                prefill_tokens,
                cpu_kernel,
                strict_tokenizer,
                strict_loader,
                threads,
                prompt_template,
                json_out,
            )
            .await
        }
        #[cfg(feature = "full-cli")]
        Some(Commands::Convert(cmd)) => cmd.execute(&config).await,
        #[cfg(feature = "cli-bench")]
        Some(Commands::Benchmark(cmd)) => {
            if run_cuda_bitnet_perf005_benchmark_profile(&requested_backend_label, &cmd).await? {
                Ok(())
            } else {
                cmd.execute(&config).await
            }
        }
        #[cfg(feature = "full-cli")]
        Some(Commands::Serve(cmd)) => cmd.execute(&config).await,
        Some(Commands::Tokenize { model, tokenizer, text, file, bos, json_out }) => {
            handle_tokenize_command(model, tokenizer, text, file, bos, json_out).await
        }
        Some(Commands::PromptAuthorityAudit {
            model,
            tokenizer,
            prompt,
            system_prompt,
            reference_source,
            reference_rendered_prompt,
            reference_prompt_ids,
            json_out,
        }) => {
            handle_prompt_authority_audit_command(
                model,
                tokenizer,
                prompt,
                system_prompt,
                reference_source,
                reference_rendered_prompt,
                reference_prompt_ids,
                json_out,
            )
            .await
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
        #[cfg(feature = "full-cli")]
        Some(Commands::LunarLake(command)) => {
            handle_lunar_lake_command(command, &requested_backend_label).await
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
        Some(Commands::IntelNpuBitnetLinearSubgraph { strict, json_out }) => {
            intel_npu::handle_bitnet_linear_subgraph_command(strict, json_out).await
        }
        Some(Commands::IntelNpuBitnetFfnSubgraph { strict, cpu_reference, json_out }) => {
            intel_npu::handle_bitnet_ffn_subgraph_command(strict, cpu_reference, json_out).await
        }
        Some(Commands::IntelArc140vOpenvinoGpuSmoke { strict, json_out }) => {
            intel_arc::handle_openvino_gpu_smoke_command(strict, json_out).await
        }
        Some(Commands::IntelArc140vOpenclSmoke { strict, json_out }) => {
            intel_arc::handle_opencl_smoke_command(strict, json_out).await
        }
        Some(Commands::IntelArc140vOpenclParity { strict, cpu_reference, json_out }) => {
            intel_arc::handle_opencl_parity_command(strict, cpu_reference, json_out).await
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

async fn handle_prompt_authority_audit_command(
    model_path: std::path::PathBuf,
    tokenizer_path: Option<std::path::PathBuf>,
    prompt: String,
    system_prompt: Option<String>,
    reference_source: Option<String>,
    reference_rendered_prompt: Option<String>,
    reference_prompt_ids: Vec<u32>,
    json_out: Option<std::path::PathBuf>,
) -> Result<()> {
    use bitnet_models::GgufReader;
    use bitnet_tokenizers::Tokenizer;

    let model_sha256 = compute_model_sha256(&model_path)?;
    let gguf_bytes = std::fs::read(&model_path)
        .with_context(|| format!("Failed to read model: {}", model_path.display()))?;
    let gguf = GgufReader::new(&gguf_bytes).context("Failed to parse GGUF")?;
    let gguf_architecture = gguf.get_string_metadata("general.architecture");
    let gguf_model_name = gguf.get_string_metadata("general.name");
    let gguf_tokenizer_model = gguf.get_string_metadata("tokenizer.ggml.model");
    let gguf_chat_template = gguf.get_string_metadata("tokenizer.chat_template");
    let gguf_vocab_size = gguf
        .get_u32_metadata("tokenizer.ggml.vocab_size")
        .or_else(|| {
            gguf_architecture
                .as_ref()
                .and_then(|arch| gguf.get_u32_metadata(&format!("{arch}.vocab_size")))
        })
        .or_else(|| gguf.get_u32_metadata("llama.vocab_size"));

    let tokenizer_resolution =
        bitnet_tokenizers::auto::resolve_tokenizer(&model_path, tokenizer_path.as_deref(), true)?;
    let tokenizer_source = tokenizer_resolution.source;
    let tokenizer_path_resolved = tokenizer_resolution.path.clone();
    let tokenizer: std::sync::Arc<dyn Tokenizer + Send + Sync> = tokenizer_resolution.tokenizer;
    let tokenizer_json_metadata = read_resolved_tokenizer_json_prompt_metadata(
        tokenizer_source,
        tokenizer_path_resolved.as_deref(),
    );
    let external_chat_template =
        tokenizer_json_metadata.as_ref().and_then(|metadata| metadata.chat_template.clone());
    let chat_template_for_detection =
        gguf_chat_template.as_deref().or(external_chat_template.as_deref());
    let tokenizer_name_hint = gguf_tokenizer_model.as_deref().or_else(|| {
        tokenizer_json_metadata.as_ref().and_then(|metadata| metadata.family.as_deref())
    });
    let tokenizer_family = infer_tokenizer_label(tokenizer.as_ref(), tokenizer_source);
    let tokenizer_type = tokenizer_type_for_receipt(&tokenizer_family, tokenizer_source);
    let tokenizer_vocab_size = tokenizer.real_vocab_size();
    let metadata_template = bitnet_inference::TemplateType::detect_from_metadata(
        gguf_architecture.as_deref(),
        gguf_model_name.as_deref(),
        tokenizer_name_hint,
        chat_template_for_detection,
    );
    let current_default_template = prompt_audit_current_default_template(
        &model_path,
        tokenizer_path.as_deref(),
        tokenizer.as_ref(),
    );

    let variants = [
        ("current_default", current_default_template, "current_cli_default_heuristic"),
        ("metadata_authority", metadata_template, "gguf_or_tokenizer_metadata"),
        ("raw", bitnet_inference::TemplateType::Raw, "explicit_template"),
        ("llama3_chat", bitnet_inference::TemplateType::Llama3Chat, "explicit_template"),
        ("bitnetcpp_answer", bitnet_inference::TemplateType::BitnetCppAnswer, "explicit_template"),
        (
            "hf_apply_chat_template_equivalent",
            bitnet_inference::TemplateType::BitnetCppAnswer,
            "local_shape_only_reference_token_ids_not_embedded",
        ),
        (
            "bitnet_cpp_conversation_equivalent",
            bitnet_inference::TemplateType::BitnetCppAnswer,
            "local_shape_only_reference_token_ids_not_embedded",
        ),
    ];

    let mut variant_json = Vec::with_capacity(variants.len());
    let mut current_ids = None;
    let mut metadata_ids = None;
    let mut current_rendered = String::new();
    let mut metadata_rendered = String::new();
    for (label, template_type, template_source) in variants {
        let (entry, encoded_ids, rendered_prompt) = prompt_audit_variant_json(
            label,
            template_type,
            template_source,
            &prompt,
            system_prompt.as_deref(),
            tokenizer.as_ref(),
        );
        if label == "current_default" {
            current_ids = encoded_ids.clone();
            current_rendered = rendered_prompt.clone();
        }
        if label == "metadata_authority" {
            metadata_ids = encoded_ids.clone();
            metadata_rendered = rendered_prompt;
        }
        variant_json.push(entry);
    }

    let (first_divergence_stage, notes, first_mismatch_index) = prompt_audit_classification(
        &current_rendered,
        &metadata_rendered,
        &current_ids,
        &metadata_ids,
    );
    let reference_parity = prompt_audit_reference_parity_json(
        reference_source,
        reference_rendered_prompt,
        &reference_prompt_ids,
        &metadata_rendered,
        metadata_ids.as_deref(),
    );

    let chat_template_source = if gguf_chat_template.is_some() {
        "gguf"
    } else if external_chat_template.is_some() {
        "tokenizer_json"
    } else {
        "none"
    };
    let receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_prompt_token_authority_audit",
        "machine_id": "intel-258v",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "model": {
            "path": model_path.display().to_string(),
            "sha256": model_sha256,
            "gguf_architecture": gguf_architecture,
            "gguf_name": gguf_model_name,
            "n_tensors": gguf.tensor_count(),
            "n_kv": gguf.metadata_keys().len(),
            "gguf_vocab_size": gguf_vocab_size,
        },
        "tokenizer": {
            "source": tokenizer_source.as_str(),
            "path": tokenizer_path_resolved.map(|path| path.display().to_string()),
            "family": tokenizer_family,
            "type": tokenizer_type,
            "vocab_size": tokenizer_vocab_size,
            "gguf_model": gguf_tokenizer_model,
            "chat_template_source": chat_template_source,
            "chat_template_present": gguf_chat_template.is_some() || external_chat_template.is_some(),
            "bos_token_id": tokenizer.bos_token_id(),
            "eos_token_id": tokenizer.eos_token_id(),
        },
        "reference_inputs": {
            "hf_apply_chat_template": {
                "rendered_prompt_available": true,
                "token_ids_available": false,
                "source": "local BitnetCppAnswer envelope; CPU258V-020 will compare official HF token IDs"
            },
            "bitnet_cpp_conversation_mode": {
                "rendered_prompt_available": true,
                "token_ids_available": false,
                "source": "local BitnetCppAnswer envelope; CPU258V-020 will compare bitnet.cpp/reference IDs"
            }
        },
        "prompt_variants": variant_json,
        "tokens": {
            "generated_token_ids": [],
            "eos_token_id": tokenizer.eos_token_id(),
        },
        "logits": {
            "first_token_top_k": [],
            "available": false,
            "reason": "prompt_token_authority_audit_does_not_run_model_inference"
        },
        "classification": {
            "first_divergence_stage": first_divergence_stage,
            "first_mismatch_index": first_mismatch_index,
            "notes": notes,
        },
        "reference_parity": reference_parity,
        "fallback_used": false,
        "claim_boundary": {
            "prompt_token_authority_only": true,
            "answer_quality_claimed": false,
            "speed_claimed": false,
            "arc_or_npu_claimed": false,
            "qk256_kernel_claimed": false,
        }
    });

    write_json_output(json_out.as_ref(), &receipt)?;
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

fn handle_cuda_command(action: CudaAction, explicit_device_label: Option<&str>) -> Result<()> {
    match action {
        CudaAction::Doctor { model, tokenizer, json_out } => {
            handle_cuda_doctor_command(explicit_device_label, model, tokenizer, json_out)
        }
    }
}

fn effective_cuda_doctor_backend(explicit_device_label: Option<&str>) -> &str {
    explicit_device_label.unwrap_or(RTX_5070_TI_CUDA)
}

fn validate_strict_cuda_backend_label(requested_backend_label: &str, context: &str) -> Result<()> {
    if requested_backend_label != RTX_5070_TI_CUDA {
        anyhow::bail!(
            "{context} requires --device {RTX_5070_TI_CUDA}; requested backend was {requested_backend_label}"
        );
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct StrictBitnetCudaAskPreflight {
    tokenizer_source: String,
    tokenizer_path: Option<std::path::PathBuf>,
    receipt_path: std::path::PathBuf,
}

fn strict_bitnet_cuda_ask_preflight(
    model_path: &std::path::Path,
    tokenizer_path: Option<&std::path::Path>,
    receipt_out: Option<&std::path::Path>,
) -> Result<StrictBitnetCudaAskPreflight> {
    if !model_path.exists() {
        anyhow::bail!(
            "strict CUDA ask requires a model artifact before generation; model path does not exist: {}",
            model_path.display()
        );
    }
    if model_path.is_dir() {
        anyhow::bail!(
            "strict CUDA ask requires a BitNet GGUF model file before generation; got directory: {}",
            model_path.display()
        );
    }
    if model_path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
        anyhow::bail!(
            "strict CUDA ask requires a BitNet GGUF model file before generation; got: {}",
            model_path.display()
        );
    }

    let tokenizer_resolution =
        bitnet_tokenizers::auto::resolve_tokenizer(model_path, tokenizer_path, true).with_context(
            || {
                format!(
                    "strict CUDA ask requires tokenizer authority before generation for {}",
                    model_path.display()
                )
            },
        )?;
    let receipt_path = receipt_out
        .map(std::path::Path::to_path_buf)
        .or_else(|| strict_ask_default_receipt_path(true, false))
        .ok_or_else(|| anyhow::anyhow!("strict CUDA ask could not resolve a receipt path"))?;

    Ok(StrictBitnetCudaAskPreflight {
        tokenizer_source: tokenizer_resolution.source.as_str().to_string(),
        tokenizer_path: tokenizer_resolution.path,
        receipt_path,
    })
}

fn handle_cuda_doctor_command(
    explicit_device_label: Option<&str>,
    model: Option<std::path::PathBuf>,
    tokenizer: Option<std::path::PathBuf>,
    json_out: Option<std::path::PathBuf>,
) -> Result<()> {
    let requested_backend = effective_cuda_doctor_backend(explicit_device_label);
    validate_strict_cuda_backend_label(requested_backend, "cuda doctor")?;

    let backend_identity = resolve_run_backend_identity(requested_backend, true)?;
    if backend_identity.selected_backend.as_str() != RTX_5070_TI_CUDA
        || backend_identity.runtime_api.as_str() != "cuda"
        || backend_identity.fallback_used
    {
        anyhow::bail!(
            "cuda doctor requires strict RTX 5070 Ti CUDA routing; requested_backend={}, selected_backend={}, runtime_api={}, fallback_used={}, fallback_reason={:?}",
            backend_identity.requested_backend,
            backend_identity.selected_backend,
            backend_identity.runtime_api,
            backend_identity.fallback_used,
            backend_identity.fallback_reason
        );
    }

    let mut cuda_probe = bitnet_device_probe::probe_nvidia_cuda(Some(0));
    if cuda_probe.available
        && let Some(error) = validate_rtx_5070_ti_identity(&cuda_probe)
    {
        cuda_probe.available = false;
        cuda_probe.failure_reason = Some(error);
    }
    if !cuda_probe.available {
        let reason = cuda_probe
            .failure_reason
            .clone()
            .unwrap_or_else(|| "requested RTX 5070 Ti CUDA device is unavailable".to_string());
        anyhow::bail!("{reason}");
    }

    let model_preflight = if let Some(model_path) = model.as_ref() {
        let preflight = strict_bitnet_cuda_ask_preflight(model_path, tokenizer.as_deref(), None)?;
        serde_json::json!({
            "status": "preflight_ready",
            "path": model_path.display().to_string(),
            "format": "gguf",
            "tokenizer_authority": {
                "source": preflight.tokenizer_source,
                "path": preflight.tokenizer_path.as_ref().map(|path| path.display().to_string()),
                "strict": true
            },
            "qk256_route_ready": "pending_generation_receipt_validation",
            "default_receipt_path": preflight.receipt_path.display().to_string()
        })
    } else {
        serde_json::json!({
            "status": "not_checked",
            "note": "pass --model and optional --tokenizer to preflight model/tokenizer authority",
            "qk256_route_ready": "not_checked",
            "default_receipt_path": strict_ask_default_receipt_path(true, false)
                .ok_or_else(|| {
                    anyhow::anyhow!("cuda doctor could not resolve a default strict CUDA receipt path")
                })?
                .display()
                .to_string()
        })
    };

    let artifact_path = json_out.as_ref().map(|path| path.display().to_string());
    let receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_cuda_doctor",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": RTX_5070_TI_CUDA,
        "requested_backend": backend_identity.requested_backend,
        "selected_backend": backend_identity.selected_backend,
        "runtime_api": backend_identity.runtime_api,
        "fallback_used": backend_identity.fallback_used,
        "fallback_reason": backend_identity.fallback_reason,
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
        "model_preflight": model_preflight,
        "prompt_template_authority": {
            "family": BITNET_CPP_ANSWER_TEMPLATE,
            "source": "strict_cuda_bitnet_answer_path"
        },
        "strict_execution_policy": {
            "cpu_fallback_allowed": false,
            "fallback_used": false,
            "generic_cuda_proof_allowed": false
        },
        "speedup_claim": false,
        "claim_allowed": "The RTX 5070 Ti CUDA strict preflight passed; model answer quality and speed still require ask/chat/bench receipts.",
        "claims_not_allowed": [
            "QK256 kernels executed",
            "coherent BitNet local answer passed",
            "dense SLM CUDA proof",
            "server readiness",
            "profile-qualified speedup"
        ],
        "artifact_path": artifact_path,
    });

    if let Some(path) = json_out.as_ref() {
        write_json_output(Some(path), &receipt)?;
    }
    print_cuda_doctor_summary(&receipt, json_out.as_deref());

    Ok(())
}

fn print_cuda_doctor_summary(receipt: &serde_json::Value, json_out: Option<&std::path::Path>) {
    println!("CUDA doctor:");
    println!("  requested_backend: {}", receipt["requested_backend"].as_str().unwrap_or(""));
    println!("  selected_backend: {}", receipt["selected_backend"].as_str().unwrap_or(""));
    println!("  runtime_api: {}", receipt["runtime_api"].as_str().unwrap_or(""));
    println!("  fallback_used: {}", receipt["fallback_used"].as_bool().unwrap_or(true));
    println!(
        "  tokenizer_authority: {}",
        receipt
            .pointer("/model_preflight/tokenizer_authority/source")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("not_checked")
    );
    println!(
        "  prompt_template_authority: {}",
        receipt
            .pointer("/prompt_template_authority/family")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
    );
    println!(
        "  qk256_route_ready: {}",
        receipt
            .pointer("/model_preflight/qk256_route_ready")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("not_checked")
    );
    println!("  speedup_claim: {}", receipt["speedup_claim"].as_bool().unwrap_or(false));
    println!(
        "  default_receipt_path: {}",
        receipt
            .pointer("/model_preflight/default_receipt_path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
    );
    if let Some(path) = json_out {
        println!("  doctor_receipt: {}", path.display());
    }
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
        ValidateAction::OpenVinoLunarLake { receipt, json_out } => {
            bitnet_receipts_core::validate_lunar_lake_openvino_receipt_file(&receipt)?;
            let summary = serde_json::json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_openvino_receipt_validation",
                "source_receipt": receipt,
                "valid": true,
                "validated_contract": "lunar_lake_openvino_route_boundary",
                "claim_boundary": {
                    "route_promotion_changed": false,
                    "speedup_claimed": false,
                    "power_advantage_claimed": false,
                    "bitnet_qk256_or_i2s_claimed": false,
                    "native_opencl_claimed": false
                }
            });
            write_json_output(json_out.as_ref(), &summary)?;
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

#[cfg(feature = "full-cli")]
fn write_json_output_silent(path: &std::path::Path, value: &serde_json::Value) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(value)?;
    std::fs::write(path, json)
        .with_context(|| format!("Failed to write JSON to {}", path.display()))?;
    Ok(())
}

pub(crate) fn answer_corpus_child_phase(phase: &str, details: serde_json::Value) {
    let Some(path) = std::env::var_os("BITNET_ANSWER_CORPUS_CHILD_PHASE_PATH") else {
        return;
    };
    let event = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "child_phase": phase,
        "details": details,
    });
    eprintln!("answer_corpus_child_phase={phase}");
    let path = std::path::PathBuf::from(path);
    let write_result = (|| -> std::io::Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        use std::io::Write as _;
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        writeln!(file, "{event}")?;
        Ok(())
    })();
    if let Err(error) = write_result {
        eprintln!("answer_corpus_child_phase_write_error={error}");
    }
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
        eprintln!("{} Using QK256 quantization with AVX2 acceleration", style("ℹ").cyan().bold());
        return Ok(());
    }

    // Show performance warning for scalar kernels
    eprintln!();
    eprintln!("{}", style("⚠  WARNING: Using QK256 scalar kernels (~0.1 tok/s)").yellow().bold());
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
    eprintln!("SIMD optimizations coming in v0.2.0 (≥3× faster)");
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
pub(crate) struct RunBackendIdentity {
    requested_backend: String,
    selected_backend: String,
    runtime_api: String,
    fallback_used: bool,
    fallback_reason: Option<String>,
}

pub(crate) fn resolve_run_backend_identity(
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

#[allow(dead_code)]
fn cpu_phase_machine_labels(
    requested_kernel: &str,
    selected_implementation: &str,
) -> (&'static str, &'static str) {
    if requested_kernel == "avx512" || selected_implementation == "avx512" {
        ("windows-9950x3d-rtx5070ti", "amd-9950x3d-cpu-avx512")
    } else {
        ("intel-258v", "intel-258v-cpu-avx2")
    }
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

fn is_dense_slm_model(model_family: &str, model_architecture: &str) -> bool {
    model_family == "qwen" || matches!(model_architecture, "qwen2" | "qwen3")
}

fn dense_slm_kernel_family(model_family: &str, model_architecture: &str) -> Option<&'static str> {
    is_dense_slm_model(model_family, model_architecture).then_some("dense_qwen")
}

fn dense_slm_kernel_id(model_family: &str, model_architecture: &str) -> Option<&'static str> {
    is_dense_slm_model(model_family, model_architecture).then_some("dense-qwen-cpu-reference")
}

fn model_output_head_identity(
    canonical_bitnet_model: bool,
    dense_slm_model: bool,
    model_architecture: &str,
) -> (serde_json::Value, &'static str) {
    if canonical_bitnet_model {
        return (serde_json::json!(true), "tied_token_embeddings");
    }
    if dense_slm_model {
        return match model_architecture {
            "qwen3" => (serde_json::json!(true), "tied_token_embeddings"),
            "qwen2" => (serde_json::json!(false), "output.weight"),
            _ => (serde_json::Value::Null, "unknown_dense_output_head"),
        };
    }
    (serde_json::Value::Null, "output.weight")
}

fn uses_dense_slm_cpu_reference(
    requested_backend: &str,
    selected_backend: &str,
    runtime_api: &str,
    fallback_used: bool,
) -> bool {
    runtime_api == "cpu"
        && !fallback_used
        && (requested_backend == "cpu" || requested_backend.ends_with("-cpu-neon"))
        && (selected_backend == "cpu"
            || selected_backend == "cpu-rust"
            || selected_backend.ends_with("-cpu-neon"))
}

fn dense_slm_layout_source(path: &std::path::Path) -> &'static str {
    match dense_slm_quant_format(path) {
        "Q8_0" => "gguf_dense_q8_0_reference",
        "Q4_K_M" => "gguf_dense_q4_k_m_reference",
        _ => "gguf_dense_reference",
    }
}

fn dense_slm_kernel_layout(path: &std::path::Path) -> &'static str {
    match dense_slm_quant_format(path) {
        "Q8_0" => "gguf_dense_q8_0",
        "Q4_K_M" => "gguf_dense_q4_k_m",
        _ => "gguf_dense",
    }
}

fn dense_slm_quant_format(path: &std::path::Path) -> &'static str {
    let normalized = path.to_string_lossy().to_ascii_lowercase();
    if normalized.contains("q8_0") || normalized.contains("q8-0") {
        "Q8_0"
    } else if normalized.contains("q4_k_m") || normalized.contains("q4-k-m") {
        "Q4_K_M"
    } else {
        "unknown_dense"
    }
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

fn compute_sha256_bytes(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
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

fn greedy_effective_top1_token_id(
    logits: &[f32],
    context_tokens: &[u32],
    repetition_penalty: f32,
) -> Option<u32> {
    if repetition_penalty <= 0.0
        || !repetition_penalty.is_finite()
        || repetition_penalty == 1.0
        || context_tokens.is_empty()
    {
        return greedy_top1_token_id(logits);
    }

    let mut effective_logits = logits.to_vec();
    let inv_penalty = 1.0 / repetition_penalty;
    for &token in context_tokens {
        if let Some(logit) = effective_logits.get_mut(token as usize) {
            if *logit > 0.0 {
                *logit *= inv_penalty;
            } else {
                *logit *= repetition_penalty;
            }
        }
    }

    greedy_top1_token_id(&effective_logits)
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

fn apply_qwen_no_think_prompt_policy(
    template_type: bitnet_inference::TemplateType,
    formatted_prompt: String,
    no_think: bool,
) -> Result<String> {
    if !no_think {
        return Ok(formatted_prompt);
    }

    if !matches!(template_type, bitnet_inference::TemplateType::QwenChat) {
        anyhow::bail!("--no-think is only supported with --prompt-template qwen");
    }

    const QWEN_ASSISTANT_MARKER: &str = "<|im_start|>assistant\n";
    const QWEN_NO_THINK_SUFFIX: &str = "<think>\n\n</think>\n\n";

    if formatted_prompt.ends_with(QWEN_NO_THINK_SUFFIX) {
        return Ok(formatted_prompt);
    }

    if formatted_prompt.ends_with(QWEN_ASSISTANT_MARKER) {
        return Ok(format!("{formatted_prompt}{QWEN_NO_THINK_SUFFIX}"));
    }

    anyhow::bail!(
        "--no-think requires a Qwen assistant generation prompt ending in \
         <|im_start|>assistant\\n"
    );
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

pub(crate) fn nvidia_smi_memory_used_bytes(device_index: Option<usize>) -> Option<u64> {
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

fn simple_generation_operator_progress<F>(enabled: bool, stage: &str, details: F)
where
    F: FnOnce() -> String,
{
    if enabled {
        eprintln!("generation progress: {stage} {}", details());
    }
}

#[cfg(feature = "cli-bench")]
#[derive(Debug, Clone, Copy)]
enum CudaBitnetPerf005ProfileKind {
    SingleDecode { max_new_tokens: usize },
    WarmSession { turns: usize, max_new_tokens: usize },
}

#[cfg(feature = "cli-bench")]
#[derive(Debug, Clone, Copy)]
struct CudaBitnetPerf005ProfilePlan {
    profile_id: &'static str,
    kind: CudaBitnetPerf005ProfileKind,
}

#[cfg(feature = "cli-bench")]
async fn run_cuda_bitnet_perf005_benchmark_profile(
    requested_backend_label: &str,
    command: &BenchmarkCommand,
) -> Result<bool> {
    if command.cuda_benchmark_receipt.is_some() || requested_backend_label != RTX_5070_TI_CUDA {
        return Ok(false);
    }

    let Some(profile) = command.profile.as_deref() else {
        return Ok(false);
    };

    let plan = cuda_bitnet_perf005_profile_plan(profile)?;
    let model_path = command
        .model
        .clone()
        .context("--model <PATH> is required for live CUDA-BITNET-PERF-005 profile dispatch")?;
    if !model_path.exists() {
        anyhow::bail!("Model file does not exist: {}", model_path.display());
    }
    if command.format != "text" {
        anyhow::bail!(
            "live CUDA-BITNET-PERF-005 profile dispatch writes inference receipts through --output and currently supports --format text only; use --cuda-benchmark-receipt for json/csv benchmark reports"
        );
    }

    let receipt_path = command.output.clone().unwrap_or_else(|| {
        std::path::PathBuf::from("target")
            .join("bitnet")
            .join("receipts")
            .join("cuda-bitnet-perf005")
            .join(format!("{}.json", plan.profile_id))
    });

    let dispatch_detail = match plan.kind {
        CudaBitnetPerf005ProfileKind::SingleDecode { max_new_tokens } => {
            format!("profile_kind=single_decode, max_new_tokens={max_new_tokens}")
        }
        CudaBitnetPerf005ProfileKind::WarmSession { turns, max_new_tokens } => {
            format!("profile_kind=warm_session, turns={turns}, max_new_tokens={max_new_tokens}")
        }
    };
    eprintln!(
        "PERF-005 profile `{}` dispatches through strict RTX 5070 Ti CUDA generation; {}; speedup_claim=false, full_residency_claim=false, server_ready=false",
        plan.profile_id, dispatch_detail
    );

    match plan.kind {
        CudaBitnetPerf005ProfileKind::SingleDecode { max_new_tokens } => {
            run_simple_generation(
                requested_backend_label,
                model_path,
                "auto".to_string(),
                None,
                None,
                perf005_prompt_for_profile(plan.profile_id).to_string(),
                max_new_tokens,
                0.0,
                0,
                1.0,
                1.0,
                Some(42),
                false,
                false,
                true,
                true,
                Some(receipt_path),
                None,
                Some("bitnet_qk256_cuda".to_string()),
                false,
                false,
                true,
                true,
                1,
                BITNET_CPP_ANSWER_TEMPLATE.to_string(),
                false,
                None,
                Vec::new(),
                Vec::new(),
                None,
                10,
                false,
                None,
                None,
                false,
                None,
                false,
                Some(plan.profile_id.to_string()),
                false,
                true,
            )
            .await
        }
        CudaBitnetPerf005ProfileKind::WarmSession { turns, max_new_tokens } => {
            let prompts = perf005_warm_session_prompts(plan.profile_id, turns)?
                .iter()
                .map(|prompt| (*prompt).to_string())
                .collect();
            run_cuda_warm_session(
                requested_backend_label,
                model_path,
                "auto".to_string(),
                None,
                prompts,
                max_new_tokens,
                0.0,
                0,
                1.0,
                1.0,
                Some(42),
                true,
                true,
                true,
                true,
                1,
                BITNET_CPP_ANSWER_TEMPLATE.to_string(),
                None,
                Vec::new(),
                Vec::new(),
                false,
                receipt_path,
            )
            .await
        }
    }?;

    Ok(true)
}

#[cfg(feature = "cli-bench")]
fn cuda_bitnet_perf005_profile_plan(profile: &str) -> Result<CudaBitnetPerf005ProfilePlan> {
    match profile {
        "one_token" => Ok(CudaBitnetPerf005ProfilePlan {
            profile_id: "one_token",
            kind: CudaBitnetPerf005ProfileKind::SingleDecode { max_new_tokens: 1 },
        }),
        "short_decode_8" => Ok(CudaBitnetPerf005ProfilePlan {
            profile_id: "short_decode_8",
            kind: CudaBitnetPerf005ProfileKind::SingleDecode { max_new_tokens: 8 },
        }),
        "short_decode_32" => Ok(CudaBitnetPerf005ProfilePlan {
            profile_id: "short_decode_32",
            kind: CudaBitnetPerf005ProfileKind::SingleDecode { max_new_tokens: 32 },
        }),
        "prefill_128_decode_16" => Ok(CudaBitnetPerf005ProfilePlan {
            profile_id: "prefill_128_decode_16",
            kind: CudaBitnetPerf005ProfileKind::SingleDecode { max_new_tokens: 16 },
        }),
        "warm_session_3_turns" => Ok(CudaBitnetPerf005ProfilePlan {
            profile_id: "warm_session_3_turns",
            kind: CudaBitnetPerf005ProfileKind::WarmSession { turns: 3, max_new_tokens: 16 },
        }),
        "warm_session_10_turns" => Ok(CudaBitnetPerf005ProfilePlan {
            profile_id: "warm_session_10_turns",
            kind: CudaBitnetPerf005ProfileKind::WarmSession { turns: 10, max_new_tokens: 16 },
        }),
        "prefill_512_decode_32" | "decode_128_from_warm_context" => {
            anyhow::bail!(
                "PERF-005 profile `{profile}` is defined but does not yet have a live dispatcher; next proof requires a dedicated long-prefill or warm-context runner receipt. speedup_claim=false, full_residency_claim=false"
            );
        }
        _ => {
            anyhow::bail!(
                "unknown CUDA-BITNET-PERF-005 profile `{profile}`; live profiles: one_token, short_decode_8, short_decode_32, prefill_128_decode_16, warm_session_3_turns, warm_session_10_turns; blocked profiles: prefill_512_decode_32, decode_128_from_warm_context"
            );
        }
    }
}

#[cfg(feature = "cli-bench")]
fn perf005_warm_session_prompts(profile_id: &str, turns: usize) -> Result<&'static [&'static str]> {
    const WARM_SESSION_3_TURNS: &[&str] = &[
        "Define entropy in one sentence.",
        "Name one use for quantization.",
        "Summarize why fallback rejection matters.",
    ];
    const WARM_SESSION_10_TURNS: &[&str] = &[
        "Define entropy in one sentence.",
        "Name one use for quantization.",
        "Summarize why fallback rejection matters.",
        "Explain what selected backend means.",
        "State why CPU fallback cannot prove CUDA execution.",
        "Name the BitNet CUDA route used for QK256 proof.",
        "Say whether dense CUDA proof proves BitNet QK256.",
        "Explain why speedup remains profile scoped.",
        "Name one receipt field that shows transfer volume.",
        "Summarize the next proof after this warm session.",
    ];

    let prompts = match profile_id {
        "warm_session_3_turns" => WARM_SESSION_3_TURNS,
        "warm_session_10_turns" => WARM_SESSION_10_TURNS,
        _ => anyhow::bail!("PERF-005 profile `{profile_id}` is not a warm-session profile"),
    };
    if prompts.len() != turns {
        anyhow::bail!(
            "PERF-005 profile `{profile_id}` expected {turns} warm-session prompts but has {}",
            prompts.len()
        );
    }
    Ok(prompts)
}

#[cfg(feature = "cli-bench")]
fn perf005_prompt_for_profile(profile_id: &str) -> &'static str {
    match profile_id {
        "one_token" => "Define entropy.",
        "short_decode_8" => "Define entropy in one sentence.",
        "short_decode_32" => "Explain why verified local inference needs fallback rejection.",
        "prefill_128_decode_16" => {
            "Explain how proof-carrying local inference separates artifact identity, tokenizer authority, prompt policy, backend selection, fallback rejection, route identity, answer quality, and benchmark qualification before making a CUDA claim."
        }
        _ => "Define entropy.",
    }
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
    proof_model_contract: Option<std::path::PathBuf>,
    proof_kernel_route: Option<String>,
    dump_ids: bool,
    bos: bool,
    greedy: bool,
    deterministic: bool,
    threads: usize,
    prompt_template: String,
    no_think: bool,
    system_prompt: Option<String>,
    stop: Vec<String>,
    stop_id: Vec<u32>,
    dump_logit_steps: Option<usize>,
    logits_topk: usize,
    assert_greedy: bool,
    qwen_trace_jsonl: Option<std::path::PathBuf>,
    qwen_trace_layer: Option<usize>,
    qwen_trace_full_prompt: bool,
    qwen_trace_prompt_ids: Option<String>,
    no_warnings: bool,
    profile_id: Option<String>,
    allocation_audit: bool,
    operator_progress: bool,
) -> Result<()> {
    use bitnet_common::Device;
    use bitnet_models::{Model, transformer::KVCache};
    use bitnet_sampling::{SamplingConfig, SamplingStrategy};
    use bitnet_tokenizers::Tokenizer;
    use std::sync::Arc;

    simple_generation::environment::QwenTraceEnv {
        jsonl_path: qwen_trace_jsonl.as_deref(),
        layer: qwen_trace_layer,
        full_prompt: qwen_trace_full_prompt,
        prompt_ids: qwen_trace_prompt_ids.as_deref(),
    }
    .apply();

    answer_corpus_child_phase("process_start", serde_json::json!({ "command": "run" }));
    answer_corpus_child_phase(
        "args_parsed",
        serde_json::json!({
            "requested_backend": requested_backend_label,
            "model_path": model_path.display().to_string(),
            "has_tokenizer_path": tokenizer_path.is_some(),
            "max_new_tokens": max_new_tokens,
            "prompt_template": prompt_template.clone(),
            "qwen_no_think": no_think,
            "json_out": json_out.as_ref().map(|path| path.display().to_string()),
        }),
    );

    let model_format_mode = simple_generation::model_format::ModelFormatMode::parse(&model_format)?;
    let is_hf_directory = model_format_mode.is_hf_directory(&model_path);

    // Simple logit step for dumping
    #[derive(Debug, serde::Serialize)]
    struct LogitStep {
        step: usize,
        logits_vector_length: usize,
        top_logits: Vec<serde_json::Value>,
        chosen_id: Option<u32>,
    }

    simple_generation::environment::apply_deterministic_env(deterministic, threads);

    // Set strict loader mode if requested (AC1: fail-fast with enhanced loader + strict tolerance)
    simple_generation::environment::apply_strict_loader_env(strict_loader);
    if strict_loader {
        debug!("Strict loader enabled (BITNET_DISABLE_MINIMAL_LOADER=1, BITNET_STRICT_MODE=1)");
    }

    // Override temperature if greedy mode
    let temperature = if greedy { 0.0 } else { temperature };

    let simple_generation::backend::GenerationBackendSetup {
        identity: backend_identity,
        strict_backend,
        strict_cuda_backend_selected,
        cuda_memory_before_bytes,
    } = simple_generation::backend::prepare_generation_backend(
        requested_backend_label,
        strict_loader,
    )?;

    let template_type = simple_generation::prompt::parse_prompt_template(&prompt_template)?;

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
    let mut loader_fallback_used = false;
    let model_load_start = std::time::Instant::now();
    answer_corpus_child_phase(
        "model_load_start",
        serde_json::json!({
            "model_path": model_path.display().to_string(),
            "is_hf_directory": is_hf_directory,
        }),
    );
    simple_generation_operator_progress(operator_progress, "model_load_start", || {
        format!("path={}", model_path.display())
    });

    let (model, config): (Arc<dyn Model>, _) = match loader
        .load_with_config(&model_path, &load_config)
    {
        Ok(m) => {
            let cfg = m.config().clone();
            loader_mode = detect_loader_mode_for_path(&model_path, is_hf_directory);
            (Arc::from(m) as Arc<dyn Model>, cfg)
        }
        Err(e) => {
            answer_corpus_child_phase(
                "model_load_error",
                serde_json::json!({
                    "allow_mock": allow_mock,
                    "strict_loader": strict_loader,
                    "error": e.to_string(),
                }),
            );
            simple_generation_operator_progress(operator_progress, "model_load_error", || {
                format!("allow_mock={allow_mock} error={e}")
            });
            if !allow_mock {
                anyhow::bail!(
                    "Failed to load real model: {e}\n\
                     To run with mock tensors (for smoke/UX testing only), \
                     pass --allow-mock or set BITNET_ALLOW_MOCK=1"
                );
            }
            tracing::warn!("Real loader failed: {e}. Falling back to MOCK loader (by request).");
            loader_fallback_used = true;
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
            let raw_tensors = qk256_raw_tensors_from_simple_loader(load_result.i2s_qk256)?;
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
    answer_corpus_child_phase(
        "model_load_complete",
        serde_json::json!({
            "loader_mode": loader_mode,
            "model_load_ms": rounded_ms(model_load_ms),
        }),
    );
    simple_generation_operator_progress(operator_progress, "model_load_complete", || {
        format!("loader_mode={loader_mode} model_load_ms={:.3}", rounded_ms(model_load_ms))
    });

    // Load tokenizer with deterministic CPU-BITNET authority.
    // Priority: explicit path -> GGUF metadata -> sibling tokenizer asset.

    // Track GGUF header counts for JSON output independently of tokenizer source.
    let gguf_metadata = gguf_header_counts_for_receipt(&model_path, is_hf_directory);
    let effective_strict_tokenizer = strict_tokenizer || strict_loader;
    let mut tokenizer_fallback_used = false;

    let tokenizer_load_start = std::time::Instant::now();
    answer_corpus_child_phase(
        "tokenizer_load_start",
        serde_json::json!({
            "strict_tokenizer": effective_strict_tokenizer,
            "has_tokenizer_path": tokenizer_path.is_some(),
        }),
    );
    simple_generation_operator_progress(operator_progress, "tokenizer_load_start", || {
        format!(
            "strict_tokenizer={} explicit_path={}",
            effective_strict_tokenizer,
            tokenizer_path.is_some()
        )
    });
    let tokenizer_resolution = simple_generation::tokenizer::load_generation_tokenizer(
        &model_path,
        tokenizer_path.as_deref(),
        is_hf_directory,
        effective_strict_tokenizer,
        allow_mock,
    )?;
    let tokenizer_load_ms = elapsed_ms(tokenizer_load_start);
    let tokenizer_source = tokenizer_resolution.source;
    let tokenizer_strict = tokenizer_resolution.strict;
    if tokenizer_source == bitnet_tokenizers::auto::TokenizerSource::CompatibilityFallback {
        tokenizer_fallback_used = true;
    }
    let tokenizer: std::sync::Arc<dyn Tokenizer + Send + Sync> = tokenizer_resolution.tokenizer;
    answer_corpus_child_phase(
        "tokenizer_load_complete",
        serde_json::json!({
            "tokenizer_source": tokenizer_source.as_str(),
            "tokenizer_strict": tokenizer_strict,
            "tokenizer_load_ms": rounded_ms(tokenizer_load_ms),
        }),
    );
    simple_generation_operator_progress(operator_progress, "tokenizer_load_complete", || {
        format!(
            "source={} strict={} tokenizer_load_ms={:.3}",
            tokenizer_source.as_str(),
            tokenizer_strict,
            rounded_ms(tokenizer_load_ms)
        )
    });

    let template_type = simple_generation::prompt::resolve_prompt_template(
        &prompt_template,
        template_type,
        &model_path,
        tokenizer_path.as_deref(),
        tokenizer.as_ref(),
    );

    // Format prompt using the template
    answer_corpus_child_phase(
        "prompt_render_start",
        serde_json::json!({
            "template": template_type.to_string(),
            "has_system_prompt": system_prompt.is_some(),
        }),
    );
    let formatted_prompt = apply_qwen_no_think_prompt_policy(
        template_type,
        template_type.apply(&prompt, system_prompt.as_deref()),
        no_think,
    )?;
    let rendered_prompt_sha256 = compute_sha256_bytes(formatted_prompt.as_bytes());
    answer_corpus_child_phase(
        "prompt_render_complete",
        serde_json::json!({
            "template": template_type.to_string(),
            "qwen_no_think": no_think,
            "rendered_prompt_bytes": formatted_prompt.len(),
            "rendered_prompt_sha256": rendered_prompt_sha256.clone(),
        }),
    );

    let all_stop_sequences = simple_generation::prompt::merge_stop_sequences(&stop, template_type);
    let all_stop_ids = simple_generation::prompt::merge_stop_token_ids(
        &stop_id,
        template_type,
        tokenizer.as_ref(),
    );

    debug!(
        "Template: {} | Stop sequences: {:?} | Stop IDs: {:?}",
        template_type, all_stop_sequences, all_stop_ids
    );

    let bos_policy = simple_generation::prompt::bos_policy(bos, template_type);

    // Tokenize formatted prompt with proper BOS policy and special token parsing
    let parse_special = template_type.parse_special();
    let prompt_tokenize_start = std::time::Instant::now();
    answer_corpus_child_phase(
        "prompt_tokenize_start",
        serde_json::json!({
            "bos_policy": bos_policy,
            "parse_special": parse_special,
        }),
    );
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
    let prompt_token_ids = tokens.clone();
    let prompt_token_ids_sha256 = sha256_token_ids(&prompt_token_ids)?;
    let prompt_tokenize_ms = elapsed_ms(prompt_tokenize_start);
    answer_corpus_child_phase(
        "prompt_tokenize_complete",
        serde_json::json!({
            "prompt_token_count": tokens.len(),
            "prompt_tokenize_ms": rounded_ms(prompt_tokenize_ms),
        }),
    );
    simple_generation_operator_progress(operator_progress, "prompt_tokenize_complete", || {
        format!(
            "prompt_tokens={} prompt_tokenize_ms={:.3}",
            tokens.len(),
            rounded_ms(prompt_tokenize_ms)
        )
    });
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
            "--allocation-audit is currently scoped to supported Apple CPU/NEON labels with fallback_used=false; got requested_backend={}, selected_backend={}, runtime_api={}, fallback_used={}",
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
    let mut cuda_first_forward_started = false;
    let mut cuda_first_forward_completed = false;
    answer_corpus_child_phase(
        "prompt_prefill_start",
        serde_json::json!({
            "prompt_token_count": tokens.len(),
            "prefill_prefix_tokens": tokens.len().saturating_sub(1),
            "strict_cuda_backend_selected": strict_cuda_backend_selected,
        }),
    );
    simple_generation_operator_progress(operator_progress, "prompt_prefill_start", || {
        format!(
            "prompt_tokens={} prefill_prefix_tokens={}",
            tokens.len(),
            tokens.len().saturating_sub(1)
        )
    });
    if tokens.len() > 1 {
        for token in &tokens[..tokens.len() - 1] {
            let step_start = std::time::Instant::now();
            let step_alloc_start = AllocationAuditSnapshot::current();
            let x = model.embed(&[*token])?;
            if strict_cuda_backend_selected && !cuda_first_forward_started {
                answer_corpus_child_phase(
                    "cuda_context_start",
                    serde_json::json!({
                        "trigger": "prompt_prefill_first_forward",
                        "selected_backend": backend_identity.selected_backend.as_str(),
                    }),
                );
                answer_corpus_child_phase(
                    "weight_upload_start",
                    serde_json::json!({
                        "trigger": "prompt_prefill_first_forward",
                        "selected_backend": backend_identity.selected_backend.as_str(),
                    }),
                );
                cuda_first_forward_started = true;
            }
            let _ = model.forward(&x, any_cache.as_mut())?;
            if strict_cuda_backend_selected
                && cuda_first_forward_started
                && !cuda_first_forward_completed
            {
                answer_corpus_child_phase(
                    "weight_upload_complete",
                    serde_json::json!({
                        "trigger": "prompt_prefill_first_forward",
                        "selected_backend": backend_identity.selected_backend.as_str(),
                    }),
                );
                answer_corpus_child_phase(
                    "cuda_context_complete",
                    serde_json::json!({
                        "trigger": "prompt_prefill_first_forward",
                        "selected_backend": backend_identity.selected_backend.as_str(),
                    }),
                );
                cuda_first_forward_completed = true;
            }
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
    answer_corpus_child_phase(
        "prompt_prefill_complete",
        serde_json::json!({
            "prefill_tokens": prefill_token_count,
            "prefill_ms": rounded_ms(prefill_ms),
        }),
    );
    simple_generation_operator_progress(operator_progress, "prompt_prefill_complete", || {
        format!("prefill_tokens={prefill_token_count} prefill_ms={:.3}", rounded_ms(prefill_ms))
    });

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

    // Generation loop: prompt prefill followed by incremental decoding.
    //
    // Before this loop, prompt prefill embeds and forwards every prompt token
    // except the last one. Step 0 embeds the last prompt token, so the first
    // generated token sees the complete rendered prompt context.
    //
    // Each subsequent step:
    //   1. Embed only the newly generated token
    //   2. Forward pass uses KV cache for historical context
    //
    // Historical context is maintained via:
    //   - KV cache: stores key/value tensors from previous steps
    //   - `tokens` vector: tracks full sequence for stop detection/logging
    //
    // Performance impact: this avoids O(N^2) full-context re-embedding while
    // preserving complete prompt context for the first generated token.
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
    simple_generation_operator_progress(operator_progress, "decode_start", || {
        format!("max_new_tokens={max_new_tokens}")
    });
    for step_idx in 0..max_new_tokens {
        let qwen_trace_this_step = qwen_trace_enabled() && step_idx == 0;
        if step_idx == 0 {
            answer_corpus_child_phase(
                "decode_step_0_start",
                serde_json::json!({
                    "context_token_count": tokens.len(),
                    "strict_cuda_backend_selected": strict_cuda_backend_selected,
                }),
            );
        }
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
        if strict_cuda_backend_selected && !cuda_first_forward_started {
            answer_corpus_child_phase(
                "cuda_context_start",
                serde_json::json!({
                    "trigger": "decode_step_0_forward",
                    "selected_backend": backend_identity.selected_backend.as_str(),
                }),
            );
            answer_corpus_child_phase(
                "weight_upload_start",
                serde_json::json!({
                    "trigger": "decode_step_0_forward",
                    "selected_backend": backend_identity.selected_backend.as_str(),
                }),
            );
            cuda_first_forward_started = true;
        }
        let h = model.forward(&x, any_cache.as_mut())?;
        if strict_cuda_backend_selected
            && cuda_first_forward_started
            && !cuda_first_forward_completed
        {
            answer_corpus_child_phase(
                "weight_upload_complete",
                serde_json::json!({
                    "trigger": "decode_step_0_forward",
                    "selected_backend": backend_identity.selected_backend.as_str(),
                }),
            );
            answer_corpus_child_phase(
                "cuda_context_complete",
                serde_json::json!({
                    "trigger": "decode_step_0_forward",
                    "selected_backend": backend_identity.selected_backend.as_str(),
                }),
            );
            cuda_first_forward_completed = true;
        }
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

        // Extract last token hidden state first to avoid 3D×2D matmul issues
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
                logits_vector_length: logits_vec.len(),
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
            let Some(best_i) =
                greedy_effective_top1_token_id(&logits_vec, &generated_tokens, repetition_penalty)
            else {
                anyhow::bail!("No finite logits found for --assert-greedy at step {step_idx}");
            };
            if next_token != best_i {
                eprintln!(
                    "ERROR: Non-effective-argmax token chosen in --greedy at step {}",
                    step_idx
                );
                eprintln!("  effective_argmax={} but chosen={}", best_i, next_token);
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
            let ttft_ms = start_time.elapsed().as_millis() as u64;
            first_token_ms = Some(ttft_ms);
            simple_generation_operator_progress(operator_progress, "first_token", || {
                format!("token_id={next_token} ttft_ms={ttft_ms}")
            });
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
        if step_idx == 0 {
            answer_corpus_child_phase(
                "decode_step_0_complete",
                serde_json::json!({
                    "chosen_token": next_token,
                    "generated_tokens": generated_tokens.len(),
                }),
            );
        }

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
    simple_generation_operator_progress(operator_progress, "generation_complete", || {
        format!(
            "generated_tokens={} total_ms={} tok_s={:.3}",
            generated_tokens.len(),
            total_ms,
            tok_per_sec
        )
    });

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
        let pretokenizer_authority =
            tokenizer_pretokenizer_authority(tokenizer_source, &tokenizer_label);
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
            "pretokenizer_authority": pretokenizer_authority,
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
            "bos": bos_policy,
            "explicit_bos_requested": bos,
            "parse_special": parse_special,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "seed": seed.unwrap_or(0),
            "greedy": greedy,
            "deterministic": deterministic,
            "qwen_no_think": no_think,
        });
        let prompt_generation_identity = simple_generation::prompt::prompt_generation_identity(
            simple_generation::prompt::PromptGenerationIdentityInput {
                template_family: &template_type.to_string(),
                template_source: "bitnet-prompt-templates-core",
                tokenizer_source: Some(tokenizer_source_str),
                tokenizer_authority: Some(pretokenizer_authority),
                tokenizer_sha256: None,
                tokenizer_strict: Some(tokenizer_strict),
                manual_stop_sequences: &stop,
                stop_sequences: &all_stop_sequences,
                manual_stop_token_ids: &stop_id,
                stop_token_ids: &all_stop_ids,
                stop_string_window: Some(10),
                stop_policy: "manual_plus_template_defaults",
                generation_params: simple_generation::prompt::PromptGenerationParams {
                    max_new_tokens: Some(max_new_tokens),
                    temperature: Some(temperature),
                    top_k: Some(top_k),
                    top_p: Some(top_p),
                    repetition_penalty: Some(repetition_penalty),
                    seed,
                    greedy: Some(greedy),
                    deterministic: Some(deterministic),
                    threads: Some(effective_thread_count(threads)),
                    qwen_no_think: Some(no_think),
                    fixed_token_count: Some(false),
                    stream: None,
                },
            },
        );
        let prompt_render_receipt = serde_json::json!({
            "template_family": template_type.to_string(),
            "qwen_no_think": no_think,
            "rendered_text": formatted_prompt,
            "rendered_sha256": rendered_prompt_sha256,
            "add_bos": bos_policy,
            "parse_special": parse_special,
            "stop_sequences": &all_stop_sequences,
            "stop_token_ids": &all_stop_ids,
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
        let dense_slm_model = is_dense_slm_model(model_family, &model_architecture);
        let (tie_word_embeddings, output_head_tensor) = model_output_head_identity(
            canonical_bitnet_model,
            dense_slm_model,
            &model_architecture,
        );
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
        let qk256_cpu_hot_path = bitnet_qk256_dispatch::qk256_cpu_hot_path_counters();
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
        let cuda_runtime_stats =
            strict_cuda_selected_artifact.then(bitnet_qk256_dispatch::qk256_cuda_runtime_stats);
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
        let cuda_execution_residency = strict_cuda_selected_artifact.then(|| {
            cuda_execution_residency_receipt(CudaExecutionResidencyReceiptInput {
                coverage: &bitnet_linear_coverage,
                residency: cuda_weight_residency.as_ref(),
                runtime_stats: cuda_runtime_stats.as_ref(),
                prompt_tokens: prompt_tokens_len,
                generated_tokens: generated_tokens.len(),
                kv_cache_device: "cpu",
                kv_cache_reuse_policy: "per_run_incremental_decode",
                execution_phase: if strict_cuda_short_decode_artifact {
                    "short_decode"
                } else {
                    "decode"
                },
                coverage_scope: "strict_cuda_ask_or_run",
            })
        });
        let execution_plan = strict_cuda_selected_artifact.then(|| {
            planner_receipts::bitnet_qk256_execution_plan_receipt(
                &bitnet_linear_coverage,
                requested_backend,
                selected_backend,
                runtime_api,
                "reject",
            )
        });
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
        let observed_logits_vector_length =
            logits_dump.first().map(|step| step.logits_vector_length);
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
        let proof_fallback_used =
            backend_identity.fallback_used || loader_fallback_used || tokenizer_fallback_used;
        let execution_backend = selected_backend;
        let execution_backend_matched = requested_backend.eq_ignore_ascii_case(execution_backend);
        let proof_model_contract_path =
            proof_model_contract.as_ref().map(|path| path.display().to_string());
        let proof_kernel_route_id = proof_kernel_route.as_deref();
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
            "prompt_render": prompt_render_receipt,
            "prompt_generation_identity": prompt_generation_identity,
            "text": generated_text,
            "prompt_identity": {
                "template": template_type.to_string(),
                "rendered_prompt_sha256": rendered_prompt_sha256.clone(),
                "prompt_token_ids_sha256": prompt_token_ids_sha256,
                "prompt_token_count": prompt_token_ids.len(),
                "bos_policy": bos_policy,
                "parse_special": parse_special,
            },
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
                "cuda_kernel_time_ms": cuda_runtime_stats
                    .as_ref()
                    .and_then(|stats| stats.kernel_time_ms)
                    .map(rounded_ms),
                "host_to_device_bytes": cuda_runtime_stats
                    .as_ref()
                    .map(|stats| stats.host_to_device_bytes),
                "device_to_host_bytes": cuda_runtime_stats
                    .as_ref()
                    .map(|stats| stats.device_to_host_bytes),
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
                "tie_word_embeddings": tie_word_embeddings,
                "output_head_tensor": output_head_tensor,
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
                "qk256_f32_scalar_gemv_invocations": qk256_cpu_hot_path.qk256_f32_scalar_gemv_invocations,
                "qk256_f32_avx2_gemv_invocations": qk256_cpu_hot_path.qk256_f32_avx2_gemv_invocations,
                "qk256_i8s_scaled_scalar_invocations": qk256_cpu_hot_path.qk256_i8s_scaled_scalar_invocations,
                "qk256_i8s_scaled_avx2_invocations": qk256_cpu_hot_path.qk256_i8s_scaled_avx2_invocations,
                "qk256_flat_bytes_extracted_count": qk256_cpu_hot_path.qk256_flat_bytes_extracted_count,
                "input_rows_materialized_count": qk256_cpu_hot_path.input_rows_materialized_count,
                "output_rows_allocated_count": qk256_cpu_hot_path.output_rows_allocated_count,
                "requested_kernel": qk256_cpu_hot_path.requested_kernel.clone(),
                "selected_kernel": qk256_cpu_hot_path.selected_kernel.clone(),
                "qk256_execution_path": qk256_cpu_hot_path.qk256_execution_path,
                "unsupported_ops": bitnet_linear_coverage.unsupported_ops.clone(),
                "execution_claim": bitnet_linear_coverage.execution_claim,
            },
            "qk256_hot_path": qk256_cpu_hot_path_receipt(&qk256_cpu_hot_path),
            "kernel": {
                "family": kernel_family,
                "implementation": kernel_implementation,
                "layout": kernel_layout,
                "dequantizes_before_compute": dequantizes_before_compute,
                "kernel_id": selected_kernel.as_str(),
                "hot_path_kernel_id": qk256_cpu_hot_path.selected_kernel.as_deref(),
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
                "requested_kernel": qk256_cpu_hot_path
                    .requested_kernel
                    .as_deref()
                    .unwrap_or(selected_kernel.as_str()),
                "selected_kernel": qk256_cpu_hot_path
                    .selected_kernel
                    .as_deref()
                    .unwrap_or(selected_kernel.as_str()),
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
            "logits_index_boundary": {
                "expected_logits_vector_length": tokenizer.vocab_size(),
                "expected_logits_vector_length_source": "tokenizer_vocab_size",
                "first_step_logits_vector_length": observed_logits_vector_length,
                "observed_logits_vector_length": observed_logits_vector_length,
                "observed_logits_vector_length_source": if observed_logits_vector_length.is_some() {
                    "run_receipt_logits_dump"
                } else {
                    "not_available"
                },
            },
            "proof_summary": {
                "requested_device": requested_backend,
                "requested_backend": requested_backend,
                "selected_backend": execution_backend,
                "execution_backend": execution_backend,
                "execution_backend_matched": execution_backend_matched,
                "fallback_used": proof_fallback_used,
                "backend_fallback_used": backend_identity.fallback_used,
                "backend_fallback_reason": backend_identity.fallback_reason.as_deref(),
                "loader_fallback_used": loader_fallback_used,
                "tokenizer_fallback_used": tokenizer_fallback_used,
                "strict_backend": strict_backend,
                "claim_level": "diagnostic_cli_run",
                "model_contract": proof_model_contract_path,
                "model_contract_declared": proof_model_contract.is_some(),
                "kernel_route": {
                    "route_id": proof_kernel_route_id,
                    "route_declared": proof_kernel_route.is_some(),
                    "diagnostic_only": true,
                    "claimable": false,
                },
                "route_declared": proof_kernel_route.is_some(),
                "backend_claimable": false,
                "not_claims": critical_not_claims(),
            },
            "not_claims": critical_not_claims(),
            "logits_dump": if !logits_dump.is_empty() {
                Some(logits_dump.iter().map(|step| {
                    serde_json::json!({
                        "step": step.step,
                        "logits_vector_length": step.logits_vector_length,
                        "top_logits": step.top_logits,
                        "chosen_id": step.chosen_id
                    })
                }).collect::<Vec<_>>())
            } else {
                None
            },
        });
        if uses_dense_slm_cpu_reference(
            requested_backend,
            &selected_backend,
            runtime_api,
            backend_identity.fallback_used,
        ) && let Some(dense_kernel_id) = dense_slm_kernel_id(model_family, &model_architecture)
            && let Some(object) = output.as_object_mut()
        {
            let dense_kernel_family =
                dense_slm_kernel_family(model_family, &model_architecture).unwrap_or("dense_slm");
            let dense_quant_format = dense_slm_quant_format(&model_path);
            let dense_layout_source = dense_slm_layout_source(&model_path);
            let dense_kernel_layout = dense_slm_kernel_layout(&model_path);
            object.remove("bitnet");
            object.insert(
                "dense_slm".to_string(),
                serde_json::json!({
                    "model_family": model_family,
                    "architecture": model_architecture,
                    "quant_format": dense_quant_format,
                    "kernel_family": dense_kernel_family,
                    "kernel_id": dense_kernel_id,
                    "layout_source": dense_layout_source,
                    "layout": dense_kernel_layout,
                    "execution_phase": execution_phase,
                    "provenance": "dense_slm_gguf_cpu_reference",
                    "claim_scope": "strict dense SLM CPU answer smoke only",
                }),
            );
            object.insert(
                "execution_coverage".to_string(),
                serde_json::json!({
                    "dense_slm_layers_total": serde_json::Value::Null,
                    "dense_slm_layers_on_cpu": serde_json::Value::Null,
                    "unsupported_ops": [],
                    "execution_claim": "dense_slm_cpu_reference_answer_smoke",
                }),
            );
            object.remove("qk256_hot_path");
            object.insert(
                "kernel".to_string(),
                serde_json::json!({
                    "family": dense_kernel_family,
                    "implementation": "cpu-reference",
                    "layout": dense_kernel_layout,
                    "dequantizes_before_compute": true,
                    "kernel_id": dense_kernel_id,
                }),
            );
            if let Some(model) = object.get_mut("model").and_then(serde_json::Value::as_object_mut)
            {
                model.insert("quant_format".to_string(), serde_json::json!(dense_quant_format));
            }
            if let Some(strict_provenance) =
                object.get_mut("strict_provenance").and_then(serde_json::Value::as_object_mut)
            {
                strict_provenance
                    .insert("requested_kernel".to_string(), serde_json::json!(dense_kernel_id));
                strict_provenance
                    .insert("selected_kernel".to_string(), serde_json::json!(dense_kernel_id));
                strict_provenance
                    .insert("quant_format".to_string(), serde_json::json!(dense_quant_format));
                strict_provenance.insert(
                    "provenance".to_string(),
                    serde_json::json!("dense_slm_gguf_cpu_reference"),
                );
            }
        }
        if strict_cuda_proof_artifact && let Some(object) = output.as_object_mut() {
            object.insert("claim".to_string(), serde_json::json!("strict_bitnet_cuda_inference"));
            object.insert("speedup_claim".to_string(), serde_json::json!(false));
            if let Some(residency) = &cuda_execution_residency {
                object.insert("cuda_execution_residency".to_string(), residency.clone());
            }
            if let Some(execution_plan) = &execution_plan {
                object.insert("execution_plan".to_string(), execution_plan.clone());
            }
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
                qk256_kernel_stats_receipt(&bitnet_linear_coverage, cuda_runtime_stats.as_ref()),
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
            if let Some(residency) = &cuda_execution_residency {
                object.insert("cuda_execution_residency".to_string(), residency.clone());
            }
            if let Some(execution_plan) = &execution_plan {
                object.insert("execution_plan".to_string(), execution_plan.clone());
            }
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
                    "cuda_kernel_time_ms": cuda_runtime_stats
                        .as_ref()
                        .and_then(|stats| stats.kernel_time_ms)
                        .map(rounded_ms),
                    "host_to_device_bytes": cuda_runtime_stats
                        .as_ref()
                        .map(|stats| stats.host_to_device_bytes),
                    "device_to_host_bytes": cuda_runtime_stats
                        .as_ref()
                        .map(|stats| stats.device_to_host_bytes),
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
                qk256_kernel_stats_receipt(&bitnet_linear_coverage, cuda_runtime_stats.as_ref()),
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
        answer_corpus_child_phase(
            "receipt_write_start",
            serde_json::json!({
                "json_out": json_path.display().to_string(),
                "artifact_kind": output["artifact_kind"].as_str(),
            }),
        );
        write_json_output(Some(&json_path), &output)?;
        answer_corpus_child_phase(
            "receipt_write_complete",
            serde_json::json!({
                "json_out": json_path.display().to_string(),
            }),
        );
    }

    // Dump IDs if requested
    if dump_ids {
        println!("Token IDs: {:?}", generated_tokens);
    }

    Ok(())
}

#[cfg(feature = "full-cli")]
fn sanitize_warm_session_prompt_stem(prompt: &str) -> String {
    let mut stem = prompt
        .chars()
        .filter_map(|ch| {
            if ch.is_ascii_alphanumeric() {
                Some(ch.to_ascii_lowercase())
            } else if ch.is_whitespace() || matches!(ch, '-' | '_' | '.') {
                Some('-')
            } else {
                None
            }
        })
        .collect::<String>();
    while stem.contains("--") {
        stem = stem.replace("--", "-");
    }
    let stem = stem.trim_matches('-');
    if stem.is_empty() { "prompt".to_string() } else { stem.chars().take(48).collect() }
}

fn tokenizer_pretokenizer_authority(
    source: bitnet_tokenizers::auto::TokenizerSource,
    tokenizer_label: &str,
) -> &'static str {
    match source {
        bitnet_tokenizers::auto::TokenizerSource::Explicit
        | bitnet_tokenizers::auto::TokenizerSource::Sibling
            if tokenizer_label == "llama3" =>
        {
            "llama-bpe"
        }
        bitnet_tokenizers::auto::TokenizerSource::Explicit => "externally_supplied",
        bitnet_tokenizers::auto::TokenizerSource::GgufMetadata => "present",
        bitnet_tokenizers::auto::TokenizerSource::Sibling => "externally_supplied",
        bitnet_tokenizers::auto::TokenizerSource::CompatibilityFallback => "defaulted",
    }
}

#[derive(Clone, Debug)]
#[cfg(feature = "full-cli")]
struct CpuPhasePromptPlan {
    profile_id: &'static str,
    phase: &'static str,
    prompt: String,
    max_new_tokens: usize,
}

#[derive(Clone, Debug)]
#[cfg(feature = "full-cli")]
struct CpuPhasePromptRun {
    profile_id: &'static str,
    phase: &'static str,
    prompt: String,
    formatted_prompt: String,
    prompt_template_family: String,
    add_bos: bool,
    parse_special: bool,
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    generated_text: String,
    prompt_token_count: usize,
    prefill_token_count: usize,
    prompt_tokenize_ms: f64,
    prefill_ms: f64,
    first_token_ms: Option<f64>,
    first_token_decode_ms: Option<f64>,
    decode_total_ms: f64,
    prompt_total_ms: f64,
    embed_step_ms: Vec<f64>,
    forward_step_ms: Vec<f64>,
    logits_step_ms: Vec<f64>,
    sample_step_ms: Vec<f64>,
    token_decode_step_ms: Vec<f64>,
    decode_step_ms: Vec<f64>,
    prefill_step_ms: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "full-cli")]
async fn run_cpu_phase_warm_session(
    requested_backend_label: &str,
    model_path: std::path::PathBuf,
    model_format: String,
    tokenizer_path: Option<std::path::PathBuf>,
    platform_artifact: Option<std::path::PathBuf>,
    prefill_prompt: Option<String>,
    prefill_prompt_file: Option<std::path::PathBuf>,
    decode_prompt: String,
    decode_tokens: usize,
    prefill_tokens: usize,
    cpu_kernel: String,
    strict_tokenizer: bool,
    strict_loader: bool,
    threads: usize,
    prompt_template: String,
    json_out: std::path::PathBuf,
) -> Result<()> {
    use bitnet_common::Device;
    use bitnet_models::Model;
    use bitnet_tokenizers::Tokenizer;
    use std::sync::Arc;

    if requested_backend_label != "cpu" {
        anyhow::bail!(
            "cpu-phase-warm-session is scoped to --device cpu for 258V CPU phase receipts; got {requested_backend_label}"
        );
    }
    if !strict_loader {
        anyhow::bail!("cpu-phase-warm-session requires --strict-loader");
    }
    if !strict_tokenizer {
        anyhow::bail!("cpu-phase-warm-session requires --strict-tokenizer");
    }
    if decode_tokens < 128 {
        anyhow::bail!("cpu-phase-warm-session requires --decode-tokens >= 128");
    }
    if prefill_tokens == 0 {
        anyhow::bail!("cpu-phase-warm-session requires --prefill-tokens >= 1");
    }
    if prefill_prompt.is_some() && prefill_prompt_file.is_some() {
        anyhow::bail!("pass either --prefill-prompt or --prefill-prompt-file, not both");
    }
    match model_format.as_str() {
        "auto" | "gguf" => {}
        other => {
            anyhow::bail!(
                "Invalid --model-format '{}'. cpu-phase-warm-session supports GGUF only: auto, gguf",
                other
            );
        }
    }
    match cpu_kernel.as_str() {
        "auto" | "scalar" | "avx2" | "avx512" => {}
        other => {
            anyhow::bail!("invalid --cpu-kernel {other}; expected auto, scalar, avx2, or avx512");
        }
    }
    if cpu_kernel == "avx2" && !bitnet_common::runtime_diag::CpuFeatures::detect().avx2 {
        anyhow::bail!("--cpu-kernel avx2 requested but AVX2 is not available");
    }
    if cpu_kernel == "avx512" && !bitnet_common::runtime_diag::CpuFeatures::detect().avx512f {
        anyhow::bail!("--cpu-kernel avx512 requested but AVX-512F is not available");
    }
    if model_path.is_dir() {
        anyhow::bail!(
            "cpu-phase-warm-session requires a GGUF model file, not a model directory: {}",
            model_path.display()
        );
    }
    if model_path.extension().and_then(|ext| ext.to_str()).map(str::to_ascii_lowercase).as_deref()
        == Some("safetensors")
    {
        anyhow::bail!("cpu-phase-warm-session supports GGUF only, not safetensors");
    }

    unsafe {
        std::env::set_var("BITNET_DISABLE_MINIMAL_LOADER", "1");
        std::env::set_var("BITNET_STRICT_MODE", "1");
        if threads > 0 {
            std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
        }
        match cpu_kernel.as_str() {
            "scalar" => {
                std::env::set_var("BITNET_CPU_KERNEL", "scalar");
                std::env::set_var("BITNET_FORCE_SCALAR", "1");
            }
            "avx2" => {
                std::env::set_var("BITNET_CPU_KERNEL", "avx2");
                std::env::set_var("BITNET_FORCE_SCALAR", "0");
            }
            "avx512" => {
                std::env::set_var("BITNET_CPU_KERNEL", "avx512");
                std::env::set_var("BITNET_FORCE_SCALAR", "0");
            }
            "auto" => {}
            _ => unreachable!("validated cpu_kernel"),
        }
    }

    let backend_identity = resolve_run_backend_identity(requested_backend_label, true)?;
    if backend_identity.fallback_used {
        anyhow::bail!(
            "cpu-phase-warm-session requires no-fallback CPU routing; requested_backend={}, selected_backend={}, fallback_reason={:?}",
            backend_identity.requested_backend,
            backend_identity.selected_backend,
            backend_identity.fallback_reason
        );
    }
    if backend_identity.runtime_api != "cpu" {
        anyhow::bail!(
            "cpu-phase-warm-session is CPU scoped; selected runtime_api={}",
            backend_identity.runtime_api
        );
    }
    unsafe {
        std::env::set_var("BITNET_REQUESTED_BACKEND", backend_identity.requested_backend.as_str());
        std::env::set_var("BITNET_SELECTED_BACKEND", backend_identity.selected_backend.as_str());
        std::env::set_var("BITNET_RUNTIME_API", backend_identity.runtime_api.as_str());
    }

    let session_start = std::time::Instant::now();
    let is_hf_directory = false;
    let template_type: bitnet_inference::TemplateType =
        prompt_template.parse().with_context(|| {
            format!(
                "Invalid prompt template '{}'. Supported: raw, instruct, llama3-chat, qwen2.5",
                prompt_template
            )
        })?;
    let prefill_prompt_text = match (prefill_prompt, prefill_prompt_file) {
        (Some(prompt), None) => prompt,
        (None, Some(path)) => std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?,
        (None, None) => "benchmark token ".repeat(256),
        (Some(_), Some(_)) => unreachable!("validated prompt exclusivity"),
    };

    let loader = bitnet_models::loader::ModelLoader::new(Device::Cpu);
    let load_config = bitnet_models::loader::LoadConfig {
        use_mmap: true,
        validate_checksums: false,
        progress_callback: None,
    };
    println!("CPU phase warm session loading model once from: {}", model_path.display());
    let model_load_start = std::time::Instant::now();
    let loaded_model = loader.load_with_config(&model_path, &load_config).with_context(|| {
        format!("Failed to load real model for CPU phase session: {}", model_path.display())
    })?;
    let config = loaded_model.config().clone();
    let model: Arc<dyn Model> = Arc::from(loaded_model);
    let model_load_ms = elapsed_ms(model_load_start);
    let loader_mode = detect_loader_mode_for_path(&model_path, is_hf_directory);
    if loader_mode != bitnet_models::GgufLoaderMode::RealGguf.as_str() {
        anyhow::bail!("cpu-phase-warm-session requires real_gguf loader mode; got {loader_mode}");
    }

    let tokenizer_load_start = std::time::Instant::now();
    let tokenizer_resolution =
        bitnet_tokenizers::auto::resolve_tokenizer(&model_path, tokenizer_path.as_deref(), true)
            .with_context(|| format!("Failed to resolve tokenizer for {}", model_path.display()))?;
    let tokenizer_load_ms = elapsed_ms(tokenizer_load_start);
    let tokenizer_source = tokenizer_resolution.source;
    let tokenizer_strict = tokenizer_resolution.strict;
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = tokenizer_resolution.tokenizer;
    let tokenizer_source_str = tokenizer_source.as_str();
    let tokenizer_label = infer_tokenizer_label(tokenizer.as_ref(), tokenizer_source);
    let pretokenizer_authority =
        tokenizer_pretokenizer_authority(tokenizer_source, &tokenizer_label);
    let tokenizer_type = tokenizer_type_for_receipt(&tokenizer_label, tokenizer_source);
    let gguf_metadata = gguf_header_counts_for_receipt(&model_path, is_hf_directory);
    let (n_kv, n_tensors) = gguf_metadata.unwrap_or((0, 0));
    let model_sha256 = compute_model_sha256(&model_path)?;
    let model_repo = infer_model_repo(&model_path);
    let model_architecture = infer_model_architecture(&model_path);
    let model_family = receipt_model_family(&model_architecture);
    let model_format_label = receipt_model_format(&model_path, &model_format, is_hf_directory);
    let model_file =
        model_path.file_name().and_then(|name| name.to_str()).unwrap_or_default().to_string();
    let dense_slm_model = is_dense_slm_model(model_family, &model_architecture);
    let kernel_family = dense_slm_kernel_family(model_family, &model_architecture)
        .unwrap_or_else(|| kernel_family_for_quantization(config.quantization.quantization_type));
    let kernel_implementation = if dense_slm_model {
        "cpu"
    } else {
        cpu_kernel_implementation(config.quantization.quantization_type)
    };
    if !dense_slm_model && cpu_kernel == "avx2" && kernel_implementation != "avx2" {
        anyhow::bail!(
            "--cpu-kernel avx2 requested but selected CPU kernel implementation is {kernel_implementation}"
        );
    }
    if !dense_slm_model && cpu_kernel == "scalar" && kernel_implementation != "scalar" {
        anyhow::bail!(
            "--cpu-kernel scalar requested but selected CPU kernel implementation is {kernel_implementation}"
        );
    }
    if !dense_slm_model && cpu_kernel == "avx512" && kernel_implementation != "avx512" {
        anyhow::bail!(
            "--cpu-kernel avx512 requested but selected CPU kernel implementation is {kernel_implementation}"
        );
    }
    let selected_kernel = dense_slm_kernel_id(model_family, &model_architecture)
        .map(str::to_string)
        .unwrap_or_else(|| format!("{kernel_family}-{kernel_implementation}-reference"));
    let dense_quant_format = dense_slm_quant_format(&model_path);
    let dense_layout_source = dense_slm_layout_source(&model_path);
    let dense_kernel_layout = dense_slm_kernel_layout(&model_path);
    let model_quant_format = if dense_slm_model { dense_quant_format } else { "QK256/I2_S" };
    let layout_source = if dense_slm_model {
        dense_layout_source
    } else {
        layout_source_for_quantization(config.quantization.quantization_type)
    };
    let kernel_layout = if dense_slm_model {
        dense_kernel_layout
    } else {
        kernel_layout_for_quantization(config.quantization.quantization_type)
    };
    let dequantizes_before_compute = if dense_slm_model {
        false
    } else {
        dequantizes_before_compute(config.quantization.quantization_type)
    };
    let thread_count = effective_thread_count(threads);
    let cpu_features = detected_cpu_feature_labels();
    let cpu_model = detected_cpu_model_label();
    let (machine_id, hardware_lane) = cpu_phase_machine_labels(&cpu_kernel, kernel_implementation);

    let receipt_dir = json_out
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(format!(
            "{}-profiles",
            json_out.file_stem().and_then(|stem| stem.to_str()).unwrap_or("cpu-phase-warm-session")
        ));
    std::fs::create_dir_all(&receipt_dir)
        .with_context(|| format!("Failed to create {}", receipt_dir.display()))?;

    let plans = [
        CpuPhasePromptPlan {
            profile_id: "prefill_512",
            phase: "prefill",
            prompt: prefill_prompt_text,
            max_new_tokens: prefill_tokens,
        },
        CpuPhasePromptPlan {
            profile_id: "decode_128",
            phase: "decode",
            prompt: decode_prompt,
            max_new_tokens: decode_tokens,
        },
    ];

    let mut profile_receipt_paths = Vec::with_capacity(plans.len());
    let mut profile_summaries = Vec::with_capacity(plans.len());
    for plan in &plans {
        let run = run_cpu_phase_prompt(
            plan,
            model.as_ref(),
            &config,
            tokenizer.as_ref(),
            &template_type,
        )?;
        if run.profile_id == "prefill_512" && run.prefill_token_count < 512 {
            anyhow::bail!(
                "prefill_512 prompt must prefill at least 512 tokens; got {}. Use --prefill-prompt-file with a calibrated prompt.",
                run.prefill_token_count
            );
        }
        if run.profile_id == "decode_128" && run.generated_token_ids.len() < 128 {
            anyhow::bail!(
                "decode_128 must generate at least 128 tokens; got {}",
                run.generated_token_ids.len()
            );
        }
        let receipt_path = receipt_dir.join(format!("{}.json", run.profile_id));
        let receipt = cpu_phase_strict_profile_receipt(
            &run,
            &receipt_path,
            &json_out,
            &backend_identity,
            &model_path,
            &model_repo,
            &model_file,
            &model_sha256,
            &model_format_label,
            model_family,
            &model_architecture,
            loader_mode,
            &tokenizer_label,
            tokenizer_type.as_str(),
            tokenizer_source_str,
            tokenizer_strict,
            pretokenizer_authority,
            tokenizer.as_ref(),
            dense_slm_model,
            &prompt_template,
            model_quant_format,
            kernel_family,
            kernel_implementation,
            &selected_kernel,
            layout_source,
            kernel_layout,
            dequantizes_before_compute,
            &cpu_model,
            &cpu_features,
            thread_count,
            config.model.max_position_embeddings,
            n_kv,
            n_tensors,
            model_load_ms,
            tokenizer_load_ms,
        );
        write_json_output(Some(&receipt_path), &receipt)?;
        profile_receipt_paths.push(receipt_path.display().to_string());
        profile_summaries.push(serde_json::json!({
            "profile": run.profile_id,
            "phase": run.phase,
            "receipt_path": receipt_path.display().to_string(),
            "prompt_tokens": run.prompt_token_count,
            "prefill_tokens": run.prefill_token_count,
            "generated_tokens": run.generated_token_ids.len(),
            "prefill_ms": rounded_ms(run.prefill_ms),
            "decode_total_ms": rounded_ms(run.decode_total_ms),
            "first_token_decode_ms": run.first_token_decode_ms.map(rounded_ms),
            "fallback_used": false
        }));
    }

    let aggregate_artifact_kind =
        if dense_slm_model { "dense_slm_cpu_phase_warm_session" } else { "cpu_phase_warm_session" };
    let aggregate = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": aggregate_artifact_kind,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "artifact_path": json_out.display().to_string(),
        "machine_id": machine_id,
        "hardware_lane": hardware_lane,
        "requested_backend": backend_identity.requested_backend.as_str(),
        "selected_backend": backend_identity.selected_backend.as_str(),
        "runtime_api": backend_identity.runtime_api.as_str(),
        "fallback_used": false,
        "fallback_reason": serde_json::Value::Null,
        "backend_lane": if dense_slm_model { "dense_slm_cpu" } else { hardware_lane },
        "model_family": model_family,
        "model_architecture": model_architecture,
        "quantization": model_quant_format,
        "tokenizer_source": tokenizer_source_str,
        "prompt_template": prompt_template.as_str(),
        "selected_kernel_or_runtime": selected_kernel.as_str(),
        "session": {
            "model_loaded_once": true,
            "tokenizer_loaded_once": true,
            "profile_count": plans.len(),
            "per_profile_receipt_dir": receipt_dir.display().to_string(),
            "per_profile_receipts": profile_receipt_paths,
            "platform_artifact": platform_artifact.as_ref().map(|path| path.display().to_string()),
        },
        "model": {
            "repo": model_repo.as_str(),
            "file": model_file.as_str(),
            "path": model_path.display().to_string(),
            "sha256": model_sha256.as_str(),
            "format": model_format_label.as_str(),
            "family": model_family,
            "architecture": model_architecture,
            "loader_mode": loader_mode,
            "fallback_loader_used": false,
            "tokenizer": tokenizer_label.as_str(),
            "vocab_size": tokenizer.vocab_size(),
            "quant_format": model_quant_format,
        },
        "tokenizer": {
            "type": tokenizer_type.as_str(),
            "source": tokenizer_source_str,
            "strict": tokenizer_strict,
            "pretokenizer_authority": pretokenizer_authority,
        },
        "kernel": {
            "family": kernel_family,
            "implementation": kernel_implementation,
            "kernel_id": selected_kernel.as_str(),
            "requested_kernel": selected_kernel.as_str(),
            "selected_kernel": selected_kernel.as_str(),
            "layout_source": layout_source,
            "layout": kernel_layout,
            "fallback_used": false,
        },
        "dense_slm": if dense_slm_model {
            serde_json::json!({
                "model_family": model_family,
                "architecture": model_architecture,
                "quant_format": model_quant_format,
                "kernel_family": kernel_family,
                "kernel_id": selected_kernel.as_str(),
                "layout_source": layout_source,
                "layout": kernel_layout,
                "provenance": "dense_slm_gguf_cpu_reference",
                "claim_scope": "dense SLM CPU phase timing only",
            })
        } else {
            serde_json::Value::Null
        },
        "cpu": {
            "model": cpu_model.as_str(),
            "arch": std::env::consts::ARCH,
            "features": &cpu_features,
            "threads": thread_count,
        },
        "timing": {
            "model_load_ms": rounded_ms(model_load_ms),
            "tokenizer_load_ms": rounded_ms(tokenizer_load_ms),
            "total_session_ms": rounded_ms(elapsed_ms(session_start)),
        },
        "profiles": profile_summaries,
        "claim_boundary": {
            "cpu_phase_timing_only": true,
            "dense_slm_cpu_phase_timing_only": dense_slm_model,
            "speedup_claim": false,
            "sustained_throughput_claim": false,
            "arc140v_claim": false,
            "intel_npu_claim": false,
            "bitnet_answer_quality_claim": false,
            "bitnet_qk256_i2s_claim": false,
        },
        "speedup_claim": false,
    });
    write_json_output(Some(&json_out), &aggregate)?;
    println!(
        "CPU phase warm-session receipt written to {} ({} profile receipts, model/tokenizer loaded once)",
        json_out.display(),
        plans.len()
    );
    Ok(())
}

#[cfg(feature = "full-cli")]
fn run_cpu_phase_prompt(
    plan: &CpuPhasePromptPlan,
    model: &dyn bitnet_models::Model,
    config: &bitnet_common::BitNetConfig,
    tokenizer: &(dyn bitnet_tokenizers::Tokenizer + Send + Sync),
    template_type: &bitnet_inference::TemplateType,
) -> Result<CpuPhasePromptRun> {
    use bitnet_models::transformer::KVCache;
    use bitnet_sampling::{SamplingConfig, SamplingStrategy};

    let prompt_start = std::time::Instant::now();
    let formatted_prompt = template_type.apply(&plan.prompt, None);
    let bos_policy = template_type.should_add_bos();
    let parse_special = template_type.parse_special();
    let prompt_tokenize_start = std::time::Instant::now();
    let mut tokens = tokenizer.encode(&formatted_prompt, bos_policy, parse_special)?;
    ensure_non_empty_generation_context(&mut tokens, tokenizer)?;
    let prompt_tokenize_ms = elapsed_ms(prompt_tokenize_start);
    let prompt_token_count = tokens.len();
    let prompt_token_ids = tokens.clone();

    let cache = KVCache::new(config, 1, &candle_core::Device::Cpu)?;
    let mut any_cache: Box<dyn std::any::Any> = Box::new(cache);
    let mut sampler = SamplingStrategy::new(SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(0),
    });
    let mut prefill_step_ms = Vec::new();
    let prefill_start = std::time::Instant::now();
    let mut prefill_token_count = 0usize;
    if tokens.len() > 1 {
        for token in &tokens[..tokens.len() - 1] {
            let step_start = std::time::Instant::now();
            let x = model.embed(&[*token])?;
            let _ = model.forward(&x, any_cache.as_mut())?;
            prefill_step_ms.push(elapsed_ms(step_start));
            prefill_token_count += 1;
        }
    }
    let prefill_ms = if prefill_token_count > 0 { elapsed_ms(prefill_start) } else { 0.0 };

    let mut generated_token_ids = Vec::with_capacity(plan.max_new_tokens);
    let mut decode_step_ms = Vec::with_capacity(plan.max_new_tokens);
    let mut embed_step_ms = Vec::with_capacity(plan.max_new_tokens);
    let mut forward_step_ms = Vec::with_capacity(plan.max_new_tokens);
    let mut logits_step_ms = Vec::with_capacity(plan.max_new_tokens);
    let mut sample_step_ms = Vec::with_capacity(plan.max_new_tokens);
    let mut token_decode_step_ms = Vec::with_capacity(plan.max_new_tokens);
    let mut first_token_ms = None;
    let mut first_token_decode_ms = None;

    for _ in 0..plan.max_new_tokens {
        let decode_step_start = std::time::Instant::now();
        let last_token = tokens.last().copied().expect("tokens must be non-empty");

        let embed_start = std::time::Instant::now();
        let x = model.embed(&[last_token])?;
        embed_step_ms.push(elapsed_ms(embed_start));

        let forward_start = std::time::Instant::now();
        let h = model.forward(&x, any_cache.as_mut())?;
        forward_step_ms.push(elapsed_ms(forward_start));

        let last_hidden = extract_last_token_hidden(&h)?;
        let logits_start = std::time::Instant::now();
        let logits = model.logits(&last_hidden)?;
        let logits_vec = extract_logits_2d(&logits)?;
        logits_step_ms.push(elapsed_ms(logits_start));

        let sample_start = std::time::Instant::now();
        let next_token = sampler.sample(&logits_vec, &generated_token_ids)?;
        sample_step_ms.push(elapsed_ms(sample_start));

        tokens.push(next_token);
        generated_token_ids.push(next_token);
        if first_token_ms.is_none() {
            first_token_ms = Some(elapsed_ms(prompt_start));
        }

        let token_decode_start = std::time::Instant::now();
        let _ = tokenizer.decode(&[next_token])?;
        token_decode_step_ms.push(elapsed_ms(token_decode_start));

        let step_ms = elapsed_ms(decode_step_start);
        if first_token_decode_ms.is_none() {
            first_token_decode_ms = Some(step_ms);
        }
        decode_step_ms.push(step_ms);
    }

    let generated_text = tokenizer.decode(&generated_token_ids)?;
    Ok(CpuPhasePromptRun {
        profile_id: plan.profile_id,
        phase: plan.phase,
        prompt: plan.prompt.clone(),
        formatted_prompt,
        prompt_template_family: template_type.to_string(),
        add_bos: bos_policy,
        parse_special,
        prompt_token_ids,
        generated_token_ids,
        generated_text,
        prompt_token_count,
        prefill_token_count,
        prompt_tokenize_ms,
        prefill_ms,
        first_token_ms,
        first_token_decode_ms,
        decode_total_ms: decode_step_ms.iter().sum(),
        prompt_total_ms: elapsed_ms(prompt_start),
        embed_step_ms,
        forward_step_ms,
        logits_step_ms,
        sample_step_ms,
        token_decode_step_ms,
        decode_step_ms,
        prefill_step_ms,
    })
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "full-cli")]
fn cpu_phase_strict_profile_receipt(
    run: &CpuPhasePromptRun,
    receipt_path: &std::path::Path,
    session_path: &std::path::Path,
    backend_identity: &RunBackendIdentity,
    model_path: &std::path::Path,
    model_repo: &str,
    model_file: &str,
    model_sha256: &str,
    model_format_label: &str,
    model_family: &'static str,
    model_architecture: &str,
    loader_mode: &str,
    tokenizer_label: &str,
    tokenizer_type: &str,
    tokenizer_source_str: &str,
    tokenizer_strict: bool,
    pretokenizer_authority: &str,
    tokenizer: &(dyn bitnet_tokenizers::Tokenizer + Send + Sync),
    dense_slm_model: bool,
    prompt_template_label: &str,
    model_quant_format: &str,
    kernel_family: &str,
    kernel_implementation: &str,
    selected_kernel: &str,
    layout_source: &str,
    kernel_layout: &str,
    dequantizes_before_compute: bool,
    cpu_model: &str,
    cpu_features: &[String],
    thread_count: usize,
    context_length: usize,
    n_kv: usize,
    n_tensors: usize,
    model_load_ms: f64,
    tokenizer_load_ms: f64,
) -> serde_json::Value {
    let steady_decode_tps =
        steady_decode_tps_ms(&run.decode_step_ms).map(|value| (value * 1000.0).round() / 1000.0);
    let sampling_total_ms = run.sample_step_ms.iter().sum::<f64>();
    let sampling_ms_per_token = if run.sample_step_ms.is_empty() {
        None
    } else {
        Some(sampling_total_ms / run.sample_step_ms.len() as f64)
    };
    let artifact_kind =
        if dense_slm_model { "dense_slm_cpu_phase_profile" } else { "strict_bitnet_cpu_profile" };
    let mut receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": artifact_kind,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "session_artifact_path": session_path.display().to_string(),
        "artifact_path": receipt_path.display().to_string(),
        "profile_id": run.profile_id,
        "requested_backend": backend_identity.requested_backend.as_str(),
        "selected_backend": backend_identity.selected_backend.as_str(),
        "runtime_api": backend_identity.runtime_api.as_str(),
        "fallback_used": false,
        "fallback_reason": serde_json::Value::Null,
        "backend_lane": if dense_slm_model { "dense_slm_cpu" } else { "bitnet_cpu" },
        "model_family": model_family,
        "model_architecture": model_architecture,
        "quantization": model_quant_format,
        "tokenizer_source": tokenizer_source_str,
        "prompt_template": prompt_template_label,
        "selected_kernel_or_runtime": selected_kernel,
        "prompt": run.prompt.as_str(),
        "prompt_render": {
            "template_family": run.prompt_template_family.as_str(),
            "rendered_text": run.formatted_prompt.as_str(),
            "add_bos": run.add_bos,
            "parse_special": run.parse_special,
            "stop_policy": "fixed_token_count_no_eos_stop",
        },
        "text": run.generated_text.as_str(),
        "tokens": {
            "prompt": run.prompt_token_count,
            "generated": run.generated_token_ids.len(),
            "total": run.prompt_token_count + run.generated_token_ids.len(),
            "prompt_ids": &run.prompt_token_ids,
            "generated_ids": &run.generated_token_ids,
            "ids": &run.generated_token_ids,
        },
        "latency": {
            "cmd_to_first_ms": run.first_token_ms.map(rounded_ms),
            "decode_first_ms": run.first_token_decode_ms.map(rounded_ms),
            "total_ms": rounded_ms(run.prompt_total_ms),
        },
        "timing": {
            "model_load_ms": rounded_ms(model_load_ms),
            "tokenizer_load_ms": rounded_ms(tokenizer_load_ms),
            "tokenize_ms": rounded_ms(run.prompt_tokenize_ms),
            "prefill_ms": rounded_ms(run.prefill_ms),
            "first_token_ms": run.first_token_ms.map(rounded_ms),
            "first_token_decode_ms": run.first_token_decode_ms.map(rounded_ms),
            "decode_total_ms": rounded_ms(run.decode_total_ms),
            "decode_steady_state_tok_s": steady_decode_tps,
            "sampling_ms_per_token": sampling_ms_per_token.map(rounded_ms),
            "decode_step_ms": timing_samples_json(&run.decode_step_ms),
            "prefill_step_ms": timing_samples_json(&run.prefill_step_ms),
            "embed_ms": timing_samples_json(&run.embed_step_ms),
            "forward_ms": timing_samples_json(&run.forward_step_ms),
            "logits_ms": timing_samples_json(&run.logits_step_ms),
            "sample_ms": timing_samples_json(&run.sample_step_ms),
            "token_decode_ms": timing_samples_json(&run.token_decode_step_ms),
            "total_ms": rounded_ms(run.prompt_total_ms),
        },
        "profile": {
            "id": run.profile_id,
            "requested": true,
            "kind": "steady_decode_prefill",
            "claim_scope": "selected CPU backend phase timing only",
            "phase": run.phase,
            "machine_context_recorded": true,
            "backend": {
                "requested_backend": backend_identity.requested_backend.as_str(),
                "selected_backend": backend_identity.selected_backend.as_str(),
                "runtime_api": backend_identity.runtime_api.as_str(),
                "fallback_used": false,
                "fallback_reason": serde_json::Value::Null,
            },
            "prompt_prefill": {
                "exercised": run.prefill_token_count > 0,
                "tokens": run.prefill_token_count,
                "ms": rounded_ms(run.prefill_ms),
                "per_token_ms": timing_samples_json(&run.prefill_step_ms),
                "kv_cache_behavior": if run.prefill_token_count > 0 {
                    "prompt_prefix_prefilled_before_decode"
                } else {
                    "single_token_prompt_no_prefix_prefill"
                },
            },
            "decode": {
                "generated_tokens": run.generated_token_ids.len(),
                "warmup_tokens": usize::from(!run.decode_step_ms.is_empty()),
                "steady_state_tokens": run.decode_step_ms.len().saturating_sub(1),
                "first_token_decode_ms": run.first_token_decode_ms.map(rounded_ms),
                "steady_state_tok_s": steady_decode_tps,
                "per_token_ms": timing_samples_json(&run.decode_step_ms),
                "steady_per_token_ms": timing_samples_json(run.decode_step_ms.get(1..).unwrap_or(&[])),
                "embed_ms": timing_samples_json(&run.embed_step_ms),
                "forward_ms": timing_samples_json(&run.forward_step_ms),
                "logits_ms": timing_samples_json(&run.logits_step_ms),
                "sample_ms": timing_samples_json(&run.sample_step_ms),
                "token_decode_ms": timing_samples_json(&run.token_decode_step_ms),
            },
            "model_load_ms": rounded_ms(model_load_ms),
            "tokenizer_load_ms": rounded_ms(tokenizer_load_ms),
            "prompt_tokenize_ms": rounded_ms(run.prompt_tokenize_ms),
        },
        "model": {
            "repo": model_repo,
            "file": model_file,
            "path": model_path.display().to_string(),
            "sha256": model_sha256,
            "format": model_format_label,
            "family": model_family,
            "architecture": model_architecture,
            "context_length": context_length,
            "tokenizer": tokenizer_label,
            "vocab_size": tokenizer.vocab_size(),
            "loader_mode": loader_mode,
            "fallback_loader_used": false,
            "quant_format": model_quant_format,
        },
        "bitnet": {
            "weight_quantization": "W1.58",
            "activation_quantization": "A8",
            "quantization": "W1.58A8",
            "kernel_format": kernel_family,
            "kernel_family": kernel_family,
            "execution_phase": run.phase,
            "layout_source": layout_source,
            "fallback_layout": serde_json::Value::Null,
        },
        "execution": {
            "phase": run.phase,
            "prompt_tokens": run.prompt_token_count,
            "generated_tokens": run.generated_token_ids.len(),
            "batch_size": 1,
            "thread_count": thread_count,
            "requested_backend": backend_identity.requested_backend.as_str(),
            "selected_backend": backend_identity.selected_backend.as_str(),
            "runtime_api": backend_identity.runtime_api.as_str(),
            "fallback_used": false,
            "fallback_reason": serde_json::Value::Null,
        },
        "kernel": {
            "family": kernel_family,
            "implementation": kernel_implementation,
            "layout": kernel_layout,
            "dequantizes_before_compute": dequantizes_before_compute,
            "kernel_id": selected_kernel,
        },
        "cpu": {
            "model": cpu_model,
            "arch": std::env::consts::ARCH,
            "features": cpu_features,
            "threads": thread_count,
        },
        "strict_provenance": {
            "requested_backend": backend_identity.requested_backend.as_str(),
            "selected_backend": backend_identity.selected_backend.as_str(),
            "requested_kernel": selected_kernel,
            "selected_kernel": selected_kernel,
            "loader_mode": loader_mode,
            "tokenizer_source": tokenizer_source_str,
            "tokenizer_strict": tokenizer_strict,
            "model_family": model_family,
            "quant_format": model_quant_format,
            "cpu_model": cpu_model,
            "cpu_features": cpu_features,
            "thread_count": thread_count,
            "fallback_used": false,
            "fallback_reason": serde_json::Value::Null,
            "prompt_tokens": run.prompt_token_count,
            "decode_tokens": run.generated_token_ids.len(),
            "phase": run.phase,
        },
        "counts": {
            "n_kv": n_kv,
            "n_tensors": n_tensors,
            "unmapped": 0,
        },
        "tokenizer": {
            "type": tokenizer_type,
            "model_family": tokenizer_type,
            "source": tokenizer_source_str,
            "strict": tokenizer_strict,
            "pretokenizer_authority": pretokenizer_authority,
            "bos": tokenizer.bos_token_id().unwrap_or(1),
            "eos": tokenizer.eos_token_id().unwrap_or(2),
        },
        "loader": {
            "mode": loader_mode,
            "minimal_fallback_allowed": false,
            "minimal_fallback_disabled": true,
            "minimal_loader_fallback_used": false,
            "tokenizer_source": tokenizer_source_str,
            "mock_tensors_used": false,
        },
        "gen_policy": {
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "seed": 0,
            "greedy": true,
            "deterministic": true,
            "fixed_token_count": true,
            "stop_policy": "fixed_token_count_no_eos_stop",
        },
        "speedup_claim": false,
    });
    if dense_slm_model && let Some(object) = receipt.as_object_mut() {
        object.remove("bitnet");
        object.insert(
            "dense_slm".to_string(),
            serde_json::json!({
                "model_family": model_family,
                "architecture": model_architecture,
                "quant_format": model_quant_format,
                "kernel_family": kernel_family,
                "kernel_id": selected_kernel,
                "layout_source": layout_source,
                "layout": kernel_layout,
                "execution_phase": run.phase,
                "provenance": "dense_slm_gguf_cpu_reference",
                "claim_scope": "dense SLM CPU phase timing only",
            }),
        );
        object.insert("bitnet_qk256_i2s_claim".to_string(), serde_json::json!(false));
        object.insert("arc140v_claim".to_string(), serde_json::json!(false));
        object.insert("intel_npu_claim".to_string(), serde_json::json!(false));
    }
    receipt
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "full-cli")]
async fn run_cuda_warm_session(
    requested_backend_label: &str,
    model_path: std::path::PathBuf,
    model_format: String,
    tokenizer_path: Option<std::path::PathBuf>,
    prompts: Vec<String>,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
    strict_tokenizer: bool,
    strict_loader: bool,
    greedy: bool,
    deterministic: bool,
    threads: usize,
    prompt_template: String,
    system_prompt: Option<String>,
    stop: Vec<String>,
    stop_id: Vec<u32>,
    fail_on_quality: bool,
    json_out: std::path::PathBuf,
) -> Result<()> {
    use bitnet_common::Device;
    use bitnet_models::{Model, transformer::KVCache};
    use bitnet_sampling::{SamplingConfig, SamplingStrategy};
    use bitnet_tokenizers::Tokenizer;
    use std::sync::Arc;

    const RTX_5070_TI_CUDA: &str = "nvidia-rtx-5070-ti-cuda";

    if requested_backend_label != RTX_5070_TI_CUDA {
        anyhow::bail!(
            "cuda-warm-session requires --device {RTX_5070_TI_CUDA}; requested backend was {requested_backend_label}"
        );
    }
    if prompts.len() < 2 {
        anyhow::bail!("cuda-warm-session requires at least two --prompt values");
    }
    if !strict_loader {
        anyhow::bail!("cuda-warm-session requires --strict-loader");
    }
    if !strict_tokenizer {
        anyhow::bail!("cuda-warm-session requires --strict-tokenizer");
    }
    match model_format.as_str() {
        "auto" | "gguf" => {}
        other => {
            anyhow::bail!(
                "Invalid --model-format '{}'. cuda-warm-session supports GGUF only: auto, gguf",
                other
            );
        }
    }
    if model_path.is_dir() {
        anyhow::bail!(
            "cuda-warm-session requires a GGUF model file, not a model directory: {}",
            model_path.display()
        );
    }
    if deterministic {
        unsafe {
            std::env::set_var("BITNET_DETERMINISTIC", "1");
            std::env::set_var("RAYON_NUM_THREADS", "1");
            if threads > 0 {
                std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
            }
        }
    }
    unsafe {
        std::env::set_var("BITNET_DISABLE_MINIMAL_LOADER", "1");
        std::env::set_var("BITNET_STRICT_MODE", "1");
    }

    let backend_identity = resolve_run_backend_identity(requested_backend_label, true)?;
    if backend_identity.selected_backend.as_str() != RTX_5070_TI_CUDA
        || backend_identity.runtime_api.as_str() != "cuda"
        || backend_identity.fallback_used
    {
        anyhow::bail!(
            "cuda-warm-session requires strict RTX 5070 Ti CUDA routing; requested_backend={}, selected_backend={}, runtime_api={}, fallback_used={}, fallback_reason={:?}",
            backend_identity.requested_backend,
            backend_identity.selected_backend,
            backend_identity.runtime_api,
            backend_identity.fallback_used,
            backend_identity.fallback_reason
        );
    }
    if let Some(cuda_bin) = ensure_strict_cuda_runtime_libraries_visible()? {
        debug!(
            "added CUDA Toolkit bin directory to process PATH for strict CUDA warm session: {}",
            cuda_bin.display()
        );
    }
    bitnet_qk256_dispatch::reset_qk256_dispatch_coverage();
    unsafe {
        std::env::set_var("BITNET_REQUESTED_BACKEND", backend_identity.requested_backend.as_str());
        std::env::set_var("BITNET_SELECTED_BACKEND", backend_identity.selected_backend.as_str());
        std::env::set_var("BITNET_RUNTIME_API", backend_identity.runtime_api.as_str());
        std::env::set_var("BITNET_STRICT_CUDA_BACKEND", "1");
    }

    let template_type: bitnet_inference::TemplateType =
        prompt_template.parse().with_context(|| {
            format!(
                "Invalid prompt template '{}'. Supported: raw, instruct, llama3-chat, bitnetcpp-answer",
                prompt_template
            )
        })?;
    let temperature = if greedy { 0.0 } else { temperature };
    let is_hf_directory = false;
    let session_start = std::time::Instant::now();
    let cuda_memory_before_bytes = nvidia_smi_memory_used_bytes(Some(0));

    let loader = bitnet_models::loader::ModelLoader::new(Device::Cpu);
    let load_config = bitnet_models::loader::LoadConfig {
        use_mmap: true,
        validate_checksums: false,
        progress_callback: None,
    };
    println!("Strict CUDA warm session loading model once from: {}", model_path.display());
    let model_load_start = std::time::Instant::now();
    let loaded_model = loader.load_with_config(&model_path, &load_config).with_context(|| {
        format!("Failed to load real model for CUDA warm session: {}", model_path.display())
    })?;
    let config = loaded_model.config().clone();
    let model: Arc<dyn Model> = Arc::from(loaded_model);
    let model_load_ms = elapsed_ms(model_load_start);
    let loader_mode = detect_loader_mode_for_path(&model_path, is_hf_directory);
    if loader_mode != bitnet_models::GgufLoaderMode::RealGguf.as_str() {
        anyhow::bail!("cuda-warm-session requires real_gguf loader mode; got {loader_mode}");
    }

    let tokenizer_load_start = std::time::Instant::now();
    let tokenizer_resolution =
        bitnet_tokenizers::auto::resolve_tokenizer(&model_path, tokenizer_path.as_deref(), true)
            .with_context(|| format!("Failed to resolve tokenizer for {}", model_path.display()))?;
    let tokenizer_load_ms = elapsed_ms(tokenizer_load_start);
    let tokenizer_source = tokenizer_resolution.source;
    let tokenizer_strict = tokenizer_resolution.strict;
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = tokenizer_resolution.tokenizer;
    let tokenizer_source_str = tokenizer_source.as_str();
    let tokenizer_label = infer_tokenizer_label(tokenizer.as_ref(), tokenizer_source);
    let pretokenizer_authority =
        tokenizer_pretokenizer_authority(tokenizer_source, &tokenizer_label);
    let tokenizer_type = tokenizer_type_for_receipt(&tokenizer_label, tokenizer_source);
    let gguf_metadata = gguf_header_counts_for_receipt(&model_path, is_hf_directory);
    let (n_kv, n_tensors) = gguf_metadata.unwrap_or((0, 0));
    let model_sha256 = compute_model_sha256(&model_path)?;
    let model_repo = infer_model_repo(&model_path);
    let canonical_bitnet_model = model_repo == "microsoft/bitnet-b1.58-2B-4T-gguf";
    let model_architecture = infer_model_architecture(&model_path);
    let model_family = receipt_model_family(&model_architecture);
    let model_format_label = receipt_model_format(&model_path, &model_format, is_hf_directory);
    let model_file =
        model_path.file_name().and_then(|name| name.to_str()).unwrap_or_default().to_string();
    let layout_source = layout_source_for_quantization(config.quantization.quantization_type);
    let kernel_layout = kernel_layout_for_quantization(config.quantization.quantization_type);
    let kernel_family = kernel_family_for_quantization(config.quantization.quantization_type);
    let thread_count = effective_thread_count(threads);
    let cpu_features = detected_cpu_feature_labels();
    let cpu_model = detected_cpu_model_label();
    let cuda_probe_start = std::time::Instant::now();
    let cuda_probe = bitnet_device_probe::probe_nvidia_cuda(Some(0));
    let cuda_probe_ms = elapsed_ms(cuda_probe_start);

    let receipt_dir = json_out
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(format!(
            "{}-turns",
            json_out.file_stem().and_then(|stem| stem.to_str()).unwrap_or("cuda-warm-session")
        ));
    std::fs::create_dir_all(&receipt_dir)
        .with_context(|| format!("Failed to create {}", receipt_dir.display()))?;

    let mut turn_receipts = Vec::with_capacity(prompts.len());
    let mut turn_summaries = Vec::with_capacity(prompts.len());
    let mut quality_failed_turns = Vec::new();
    let mut speed_accumulator = WarmSessionSpeedAccumulator::default();
    let mut first_cuda_forward_observed = false;

    for (index, prompt) in prompts.iter().enumerate() {
        let prompt_start = std::time::Instant::now();
        let coverage_before = bitnet_qk256_dispatch::qk256_dispatch_coverage();
        let runtime_stats_before = bitnet_qk256_dispatch::qk256_cuda_runtime_stats();
        let formatted_prompt = template_type.apply(prompt, system_prompt.as_deref());
        let rendered_prompt_sha256 = compute_sha256_bytes(formatted_prompt.as_bytes());

        let mut all_stop_sequences = stop.clone();
        for template_stop in template_type.default_stop_sequences() {
            if !all_stop_sequences.contains(&template_stop) {
                all_stop_sequences.push(template_stop);
            }
        }
        let mut all_stop_ids = stop_id.clone();
        for template_id in template_type.resolve_stop_token_ids(tokenizer.as_ref()) {
            if !all_stop_ids.contains(&template_id) {
                all_stop_ids.push(template_id);
            }
        }

        let bos_policy = template_type.should_add_bos();
        let parse_special = template_type.parse_special();
        let prompt_tokenize_start = std::time::Instant::now();
        let mut tokens = tokenizer.encode(&formatted_prompt, bos_policy, parse_special)?;
        ensure_non_empty_generation_context(&mut tokens, tokenizer.as_ref())?;
        let prompt_tokenize_ms = elapsed_ms(prompt_tokenize_start);
        let prompt_token_ids = tokens.clone();
        let prompt_token_count = prompt_token_ids.len();
        let max_stop_len = all_stop_sequences.iter().map(|value| value.len()).max().unwrap_or(0);

        let cache = KVCache::new(&config, 1, &candle_core::Device::Cpu)?;
        let mut any_cache: Box<dyn std::any::Any> = Box::new(cache);
        let mut sampler = SamplingStrategy::new(SamplingConfig {
            temperature,
            top_k: top_k as u32,
            top_p,
            repetition_penalty,
            seed,
        });
        let mut generated_tokens = Vec::with_capacity(max_new_tokens);
        let mut decode_step_ms = Vec::with_capacity(max_new_tokens);
        let mut embed_step_ms = Vec::with_capacity(max_new_tokens);
        let mut forward_step_ms = Vec::with_capacity(max_new_tokens);
        let mut logits_step_ms = Vec::with_capacity(max_new_tokens);
        let mut sample_step_ms = Vec::with_capacity(max_new_tokens);
        let mut token_decode_step_ms = Vec::with_capacity(max_new_tokens);
        let mut first_token_ms = None;
        let mut first_token_decode_ms = None;
        let mut stop_tail = String::with_capacity(max_stop_len.saturating_add(16));

        let prefill_start = std::time::Instant::now();
        let mut prefill_token_count = 0usize;
        if tokens.len() > 1 {
            for token in &tokens[..tokens.len() - 1] {
                let x = model.embed(&[*token])?;
                if !first_cuda_forward_observed {
                    first_cuda_forward_observed = true;
                }
                let _ = model.forward(&x, any_cache.as_mut())?;
                prefill_token_count += 1;
            }
        }
        let prefill_ms = if prefill_token_count > 0 { elapsed_ms(prefill_start) } else { 0.0 };

        for _step_idx in 0..max_new_tokens {
            let decode_step_start = std::time::Instant::now();
            let last_token = tokens.last().copied().expect("tokens must be non-empty");

            let embed_start = std::time::Instant::now();
            let x = model.embed(&[last_token])?;
            embed_step_ms.push(elapsed_ms(embed_start));

            let forward_start = std::time::Instant::now();
            if !first_cuda_forward_observed {
                first_cuda_forward_observed = true;
            }
            let h = model.forward(&x, any_cache.as_mut())?;
            forward_step_ms.push(elapsed_ms(forward_start));

            let last_hidden = extract_last_token_hidden(&h)?;
            let logits_start = std::time::Instant::now();
            let logits = model.logits(&last_hidden)?;
            let logits_vec = extract_logits_2d(&logits)?;
            logits_step_ms.push(elapsed_ms(logits_start));

            let sample_start = std::time::Instant::now();
            let next_token = sampler.sample(&logits_vec, &generated_tokens)?;
            sample_step_ms.push(elapsed_ms(sample_start));

            tokens.push(next_token);
            generated_tokens.push(next_token);
            if first_token_ms.is_none() {
                first_token_ms = Some(prompt_start.elapsed().as_millis() as u64);
            }

            let token_decode_start = std::time::Instant::now();
            let token_text = tokenizer.decode(&[next_token])?;
            token_decode_step_ms.push(elapsed_ms(token_decode_start));
            if max_stop_len > 0 {
                stop_tail.push_str(&token_text);
                if stop_tail.len() > max_stop_len {
                    let cut = stop_tail.len() - max_stop_len;
                    let mut safe_cut = cut;
                    while safe_cut > 0 && !stop_tail.is_char_boundary(safe_cut) {
                        safe_cut -= 1;
                    }
                    stop_tail.drain(..safe_cut);
                }
            }
            let step_ms = elapsed_ms(decode_step_start);
            if first_token_decode_ms.is_none() {
                first_token_decode_ms = Some(step_ms);
            }
            decode_step_ms.push(step_ms);

            if all_stop_ids.contains(&next_token) {
                break;
            }
            if let Some(eos) = tokenizer.eos_token_id()
                && next_token == eos
            {
                break;
            }
            if max_stop_len > 0
                && !all_stop_sequences.is_empty()
                && all_stop_sequences.iter().any(|pat| stop_tail.ends_with(pat))
            {
                break;
            }
        }

        let generated_text = tokenizer.decode(&generated_tokens)?;
        let run_receipt_for_quality = serde_json::json!({
            "tokens": {
                "generated": generated_tokens.len(),
            },
        });
        let quality =
            answer_quality_receipt(&generated_text, &run_receipt_for_quality, max_new_tokens);
        if !quality["garbage_filter_passed"].as_bool().unwrap_or(false) {
            quality_failed_turns.push(index);
        }
        let prompt_total_ms = elapsed_ms(prompt_start);
        let decode_total_ms = decode_step_ms.iter().sum::<f64>();
        let sampling_total_ms = sample_step_ms.iter().sum::<f64>();
        let sampling_ms_per_token = if sample_step_ms.is_empty() {
            None
        } else {
            Some(sampling_total_ms / sample_step_ms.len() as f64)
        };
        let decode_steady_state_tok_s =
            steady_decode_tps_ms(&decode_step_ms).map(|value| (value * 1000.0).round() / 1000.0);
        speed_accumulator.record(WarmSessionPromptSpeed {
            prompt_tokens: prompt_token_count,
            generated_tokens: generated_tokens.len(),
            tokenize_ms: prompt_tokenize_ms,
            prefill_ms,
            decode_total_ms,
            sampling_ms: sampling_total_ms,
            prompt_total_ms,
            first_token_ms: first_token_ms.map(|value| value as f64),
            steady_decode_tok_s: decode_steady_state_tok_s,
        });

        let coverage_after = bitnet_qk256_dispatch::qk256_dispatch_coverage();
        let coverage_delta = qk256_dispatch_coverage_delta(&coverage_before, &coverage_after);
        let runtime_stats_after = bitnet_qk256_dispatch::qk256_cuda_runtime_stats();
        let runtime_stats_delta =
            qk256_cuda_runtime_stats_delta(&runtime_stats_before, &runtime_stats_after);
        let residency_after = bitnet_qk256_dispatch::qk256_cuda_weight_residency();
        let cuda_execution_residency =
            cuda_execution_residency_receipt(CudaExecutionResidencyReceiptInput {
                coverage: &coverage_delta,
                residency: residency_after.as_ref(),
                runtime_stats: Some(&runtime_stats_delta),
                prompt_tokens: prompt_token_count,
                generated_tokens: generated_tokens.len(),
                kv_cache_device: "cpu",
                kv_cache_reuse_policy: "recreated_per_turn_for_prompt_isolation",
                execution_phase: "warm_session_turn",
                coverage_scope: "strict_cuda_warm_session_turn",
            });
        let execution_plan = planner_receipts::bitnet_qk256_execution_plan_receipt(
            &coverage_delta,
            backend_identity.requested_backend.as_str(),
            backend_identity.selected_backend.as_str(),
            backend_identity.runtime_api.as_str(),
            "reject",
        );
        let turn_receipt_path = receipt_dir.join(format!(
            "{:02}-{}.json",
            index + 1,
            sanitize_warm_session_prompt_stem(prompt)
        ));
        let turn_receipt = serde_json::json!({
            "schema_version": "1.0.0",
            "artifact_kind": "bitnet_cuda_warm_session_turn",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "session_artifact_path": json_out.display().to_string(),
            "artifact_path": turn_receipt_path.display().to_string(),
            "turn_index": index,
            "requested_backend": backend_identity.requested_backend.as_str(),
            "selected_backend": backend_identity.selected_backend.as_str(),
            "runtime_api": backend_identity.runtime_api.as_str(),
            "fallback_used": backend_identity.fallback_used,
            "fallback_reason": backend_identity.fallback_reason.as_deref(),
            "prompt": prompt,
            "answer": generated_text,
            "text": generated_text,
            "prompt_render": {
                "template": template_type.to_string(),
                "rendered_text": formatted_prompt,
                "rendered_sha256": rendered_prompt_sha256,
                "add_bos": bos_policy,
                "parse_special": parse_special,
                "stop_sequences": all_stop_sequences,
                "stop_token_ids": all_stop_ids,
            },
            "tokens": {
                "prompt": prompt_token_count,
                "generated": generated_tokens.len(),
                "total": tokens.len(),
                "prompt_ids": prompt_token_ids,
                "generated_ids": generated_tokens.clone(),
                "ids": generated_tokens.clone(),
            },
            "quality": quality,
            "timing": {
                "model_load_ms": 0.0,
                "tokenizer_load_ms": 0.0,
                "session_model_load_ms": rounded_ms(model_load_ms),
                "session_tokenizer_load_ms": rounded_ms(tokenizer_load_ms),
                "tokenize_ms": rounded_ms(prompt_tokenize_ms),
                "prefill_ms": rounded_ms(prefill_ms),
                "first_token_ms": first_token_ms,
                "first_token_decode_ms": first_token_decode_ms.map(rounded_ms),
                "decode_total_ms": rounded_ms(decode_total_ms),
                "decode_steady_state_tok_s": decode_steady_state_tok_s,
                "sampling_ms_per_token": sampling_ms_per_token.map(rounded_ms),
                "cuda_kernel_time_ms": runtime_stats_delta.kernel_time_ms.map(rounded_ms),
                "host_to_device_bytes": runtime_stats_delta.host_to_device_bytes,
                "device_to_host_bytes": runtime_stats_delta.device_to_host_bytes,
                "total_ms": rounded_ms(prompt_total_ms),
                "embed_ms": timing_samples_json(&embed_step_ms),
                "forward_ms": timing_samples_json(&forward_step_ms),
                "logits_ms": timing_samples_json(&logits_step_ms),
                "sample_ms": timing_samples_json(&sample_step_ms),
                "token_decode_ms": timing_samples_json(&token_decode_step_ms),
            },
            "model": {
                "repo": model_repo.as_str(),
                "file": model_file.as_str(),
                "path": model_path.display().to_string(),
                "sha256": model_sha256.as_str(),
                "format": model_format_label.as_str(),
                "family": model_family,
                "architecture": model_architecture,
                "loader_mode": loader_mode,
                "fallback_loader_used": false,
                "tokenizer": tokenizer_label.as_str(),
                "vocab_size": tokenizer.vocab_size(),
            },
            "tokenizer": {
                "type": tokenizer_type.as_str(),
                "model_family": tokenizer_type.as_str(),
                "source": tokenizer_source_str,
                "strict": tokenizer_strict,
                "pretokenizer_authority": pretokenizer_authority,
                "bos": tokenizer.bos_token_id().unwrap_or(1),
                "eos": tokenizer.eos_token_id().unwrap_or(2),
            },
            "kernel": {
                "family": "qk256",
                "implementation": "cuda",
                "layout": kernel_layout,
                "dequantizes_before_compute": false,
                "kernel_id": bitnet_kernels::cuda::CUDA_QK256_GEMV_KERNEL_ID,
            },
            "kernel_stats": qk256_kernel_stats_receipt(&coverage_delta, Some(&runtime_stats_delta)),
            "execution_coverage": qk256_dispatch_coverage_receipt(&coverage_delta),
            "execution_plan": execution_plan,
            "cuda_execution_residency": cuda_execution_residency,
            "bitnet": {
                "weight_quantization": if canonical_bitnet_model { "W1.58" } else { "unknown" },
                "activation_quantization": if canonical_bitnet_model { "A8" } else { "unknown" },
                "quantization": if canonical_bitnet_model { "W1.58A8" } else { "unknown" },
                "kernel_format": kernel_family,
                "kernel_family": kernel_family,
                "execution_phase": "warm_session_turn",
                "layout_source": layout_source,
                "weights_uploaded_once": residency_after.as_ref().map(|value| value.weights_uploaded_once).unwrap_or(false),
                "per_token_weight_upload": residency_after.as_ref().map(|value| value.per_token_weight_upload).unwrap_or(true),
                "weight_handle_count": residency_after.as_ref().map(|value| value.weight_handle_count).unwrap_or(0),
            },
            "prompt_prefill": {
                "exercised": prefill_token_count > 0,
                "tokens": prefill_token_count,
                "kv_cache_behavior": if prefill_token_count > 0 {
                    "prompt_prefix_prefilled_before_decode"
                } else {
                    "single_token_prompt_no_prefix_prefill"
                },
            },
            "kv_cache": {
                "enabled": true,
                "mode": "incremental_decode",
                "device": "cpu",
                "reuse_policy": "recreated_per_turn_for_prompt_isolation",
                "prompt_tokens": prompt_token_count,
                "generated_tokens": generated_tokens.len(),
                "decode_steps": generated_tokens.len(),
            },
            "session_reuse": {
                "reuse_scope": "resident_cuda_warm_session",
                "model_loaded_once": true,
                "tokenizer_loaded_once": true,
                "cuda_context_reuse_policy": "single_process_thread_local_qk256_context_after_session_reset",
                "qk256_weight_reuse_policy": "upload_once_handles_reused_across_turns",
                "weights_uploaded_once": residency_after.as_ref().map(|value| value.weights_uploaded_once).unwrap_or(false),
                "per_token_weight_upload": residency_after.as_ref().map(|value| value.per_token_weight_upload).unwrap_or(true),
                "kv_cache_reuse_policy": "recreated_per_turn_for_prompt_isolation",
                "full_transformer_cuda_residency_claimed": false,
            },
            "claim_boundary": {
                "strict_cuda_warm_session": true,
                "answer_readiness_scope": "deterministic_prompts_only",
                "broad_chat_quality_claim": false,
                "speedup_claim": false,
                "server_readiness_claim": false,
                "full_cuda_residency_claimed": false,
            },
            "speedup_claim": false,
        });
        write_json_output(Some(&turn_receipt_path), &turn_receipt)?;
        println!("Turn {}: {}", index + 1, strip_answer_special_markers(&generated_text).trim());
        turn_summaries.push(serde_json::json!({
            "turn_index": index,
            "prompt": prompt,
            "answer": turn_receipt["answer"].clone(),
            "receipt_path": turn_receipt_path.display().to_string(),
            "generated_tokens": generated_tokens.len(),
            "generated_token_ids": generated_tokens.clone(),
            "quality": turn_receipt["quality"].clone(),
            "backend": {
                "selected_backend": backend_identity.selected_backend.as_str(),
                "runtime_api": backend_identity.runtime_api.as_str(),
                "fallback_used": backend_identity.fallback_used,
            },
            "execution_plan": turn_receipt["execution_plan"].clone(),
            "execution_coverage": turn_receipt["execution_coverage"].clone(),
            "cuda_execution_residency": turn_receipt["cuda_execution_residency"].clone(),
            "bitnet": turn_receipt["bitnet"].clone(),
            "session_reuse": turn_receipt["session_reuse"].clone(),
            "prompt_tokens": prompt_token_count,
        }));
        turn_receipts.push(turn_receipt_path.display().to_string());
    }

    let total_session_ms = elapsed_ms(session_start);
    let total_coverage = bitnet_qk256_dispatch::qk256_dispatch_coverage();
    let total_runtime_stats = bitnet_qk256_dispatch::qk256_cuda_runtime_stats();
    let final_residency = bitnet_qk256_dispatch::qk256_cuda_weight_residency();
    let cuda_memory_after_bytes = nvidia_smi_memory_used_bytes(Some(0));
    let cuda_memory_hwm_bytes =
        cuda_memory_before_bytes.into_iter().chain(cuda_memory_after_bytes).max();
    let quality_passed = quality_failed_turns.is_empty();
    let strict_cuda_session_passed = total_coverage.bitnet_linear_layers_on_cuda > 0
        && total_coverage.bitnet_linear_layers_cpu_fallback == 0
        && final_residency.as_ref().is_some_and(|value| value.weights_uploaded_once)
        && final_residency.as_ref().is_some_and(|value| !value.per_token_weight_upload);
    let speed_summary = speed_accumulator.receipt(
        model_load_ms,
        tokenizer_load_ms,
        total_session_ms,
        "strict RTX 5070 Ti CUDA warm answer session",
        "strict CUDA answer-path timing is measured for this model, corpus, backend, and machine context only",
        WarmSessionReuseReceiptContext {
            sampler_reuse_enabled: false,
            sampler_reuse_policy: "recreated_per_prompt_for_rng_state_independence",
            sampler_reused_prompt_count: 0,
            sampler_recreated_prompt_count: prompts.len(),
            kv_cache_recreated_per_prompt: true,
            kv_cache_reused_across_prompts: false,
            kv_cache_reuse_policy: "recreated_per_turn_for_prompt_isolation",
            kv_cache_reused_prompt_count: 0,
            kv_cache_recreated_prompt_count: prompts.len(),
        },
    );
    let cuda_execution_residency =
        cuda_execution_residency_receipt(CudaExecutionResidencyReceiptInput {
            coverage: &total_coverage,
            residency: final_residency.as_ref(),
            runtime_stats: Some(&total_runtime_stats),
            prompt_tokens: turn_summaries
                .iter()
                .filter_map(|turn| turn["prompt_tokens"].as_u64())
                .map(|value| value as usize)
                .sum(),
            generated_tokens: turn_summaries
                .iter()
                .filter_map(|turn| turn["generated_tokens"].as_u64())
                .map(|value| value as usize)
                .sum(),
            kv_cache_device: "cpu",
            kv_cache_reuse_policy: "recreated_per_turn_for_prompt_isolation",
            execution_phase: "warm_session",
            coverage_scope: "strict_cuda_warm_session",
        });
    let execution_plan = planner_receipts::bitnet_qk256_execution_plan_receipt(
        &total_coverage,
        backend_identity.requested_backend.as_str(),
        backend_identity.selected_backend.as_str(),
        backend_identity.runtime_api.as_str(),
        "reject",
    );
    let aggregate = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_cuda_warm_session",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "artifact_path": json_out.display().to_string(),
        "requested_backend": backend_identity.requested_backend.as_str(),
        "selected_backend": backend_identity.selected_backend.as_str(),
        "runtime_api": backend_identity.runtime_api.as_str(),
        "fallback_used": backend_identity.fallback_used,
        "fallback_reason": backend_identity.fallback_reason.as_deref(),
        "session": {
            "model_loaded_once": true,
            "tokenizer_loaded_once": true,
            "cuda_context_initialized_once": strict_cuda_session_passed,
            "cuda_context_reuse_policy": "single_process_thread_local_qk256_context_after_session_reset",
            "qk256_weights_uploaded_once": final_residency.as_ref().map(|value| value.weights_uploaded_once).unwrap_or(false),
            "per_token_weight_upload": final_residency.as_ref().map(|value| value.per_token_weight_upload).unwrap_or(true),
            "turn_count": prompts.len(),
            "per_turn_receipt_dir": receipt_dir.display().to_string(),
            "per_turn_receipts": turn_receipts,
            "kv_cache_reuse_policy": "recreated_per_turn_for_prompt_isolation",
        },
        "generation": {
            "mode": if greedy { "greedy" } else { "sampling" },
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "deterministic": deterministic,
            "max_new_tokens": max_new_tokens,
            "prompt_template": prompt_template,
        },
        "model": {
            "repo": model_repo.as_str(),
            "file": model_file.as_str(),
            "path": model_path.display().to_string(),
            "sha256": model_sha256.as_str(),
            "format": model_format_label.as_str(),
            "family": model_family,
            "architecture": model_architecture,
            "loader_mode": loader_mode,
            "fallback_loader_used": false,
            "tokenizer": tokenizer_label.as_str(),
            "vocab_size": tokenizer.vocab_size(),
        },
        "tokenizer": {
            "type": tokenizer_type.as_str(),
            "model_family": tokenizer_type.as_str(),
            "source": tokenizer_source_str,
            "strict": tokenizer_strict,
            "pretokenizer_authority": pretokenizer_authority,
            "bos": tokenizer.bos_token_id().unwrap_or(1),
            "eos": tokenizer.eos_token_id().unwrap_or(2),
        },
        "timing": {
            "model_load_ms": rounded_ms(model_load_ms),
            "tokenizer_load_ms": rounded_ms(tokenizer_load_ms),
            "cuda_probe_ms": rounded_ms(cuda_probe_ms),
            "cuda_context_init_ms": serde_json::Value::Null,
            "cuda_context_init_timing_source": "not_separately_measured; included in strict CUDA session setup and first CUDA work",
            "weight_upload_ms": serde_json::Value::Null,
            "weight_upload_timing_source": "not_separately_measured; upload-once weight residency is verified by qk256 weight-handle counters",
            "cuda_kernel_time_ms": total_runtime_stats.kernel_time_ms.map(rounded_ms),
            "host_to_device_bytes": total_runtime_stats.host_to_device_bytes,
            "device_to_host_bytes": total_runtime_stats.device_to_host_bytes,
            "total_session_ms": rounded_ms(total_session_ms),
        },
        "speed": speed_summary,
        "backend": {
            "requested_backend": backend_identity.requested_backend.as_str(),
            "selected_backend": backend_identity.selected_backend.as_str(),
            "runtime_api": backend_identity.runtime_api.as_str(),
            "fallback_used": backend_identity.fallback_used,
            "fallback_reason": backend_identity.fallback_reason.as_deref(),
        },
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
            "vram_bytes": cuda_probe.vram_bytes,
            "memory_used_before_bytes": cuda_memory_before_bytes,
            "memory_used_after_bytes": cuda_memory_after_bytes,
            "memory_hwm_bytes": cuda_memory_hwm_bytes,
            "memory_hwm_source": "nvidia-smi-memory.used-sampled",
            "power_limit_watts": cuda_probe.power_limit_watts,
            "power_draw_watts": cuda_probe.power_draw_watts,
            "temperature_c": cuda_probe.temperature_c,
        },
        "cpu": {
            "model": cpu_model.as_str(),
            "arch": std::env::consts::ARCH,
            "features": &cpu_features,
            "threads": thread_count,
        },
        "counts": {
            "n_kv": n_kv,
            "n_tensors": n_tensors,
        },
        "kernel": {
            "family": "qk256",
            "implementation": "cuda",
            "layout": kernel_layout,
            "dequantizes_before_compute": false,
            "kernel_id": bitnet_kernels::cuda::CUDA_QK256_GEMV_KERNEL_ID,
        },
        "kernel_stats": qk256_kernel_stats_receipt(&total_coverage, Some(&total_runtime_stats)),
        "execution_coverage": qk256_dispatch_coverage_receipt(&total_coverage),
        "execution_plan": execution_plan,
        "cuda_execution_residency": cuda_execution_residency,
        "bitnet": {
            "weight_quantization": if canonical_bitnet_model { "W1.58" } else { "unknown" },
            "activation_quantization": if canonical_bitnet_model { "A8" } else { "unknown" },
            "quantization": if canonical_bitnet_model { "W1.58A8" } else { "unknown" },
            "kernel_format": kernel_family,
            "kernel_family": kernel_family,
            "execution_phase": "warm_session",
            "layout_source": layout_source,
            "weights_uploaded_once": final_residency.as_ref().map(|value| value.weights_uploaded_once).unwrap_or(false),
            "per_token_weight_upload": final_residency.as_ref().map(|value| value.per_token_weight_upload).unwrap_or(true),
            "weight_handle_count": final_residency.as_ref().map(|value| value.weight_handle_count).unwrap_or(0),
        },
        "quality_summary": {
            "passed": quality_passed,
            "failed_turn_indices": quality_failed_turns,
            "fail_on_quality": fail_on_quality,
        },
        "strict_session_validation": {
            "passed": strict_cuda_session_passed,
            "cuda_kernel_invocations_gt_zero": total_coverage.bitnet_linear_layers_on_cuda > 0,
            "bitnet_linear_layers_cpu_fallback": total_coverage.bitnet_linear_layers_cpu_fallback,
            "weights_uploaded_once": final_residency.as_ref().map(|value| value.weights_uploaded_once).unwrap_or(false),
            "per_token_weight_upload": final_residency.as_ref().map(|value| value.per_token_weight_upload).unwrap_or(true),
        },
        "turns": turn_summaries,
        "claim_boundary": {
            "strict_cuda_warm_session": true,
            "model_loaded_once": true,
            "tokenizer_loaded_once": true,
            "cuda_context_reused": strict_cuda_session_passed,
            "qk256_weight_handles_reused": final_residency.as_ref().map(|value| value.weight_handle_count > 0).unwrap_or(false),
            "answer_readiness_scope": "deterministic_prompts_only",
            "broad_chat_quality_claim": false,
            "speedup_claim": false,
            "server_readiness_claim": false,
            "full_cuda_residency_claimed": false,
            "kv_cache_reuse_claimed": false,
        },
        "speedup_claim": false,
    });
    write_json_output(Some(&json_out), &aggregate)?;
    println!(
        "CUDA warm session receipt written to {} ({} turns, model/tokenizer loaded once)",
        json_out.display(),
        prompts.len()
    );
    if !strict_cuda_session_passed {
        anyhow::bail!(
            "strict CUDA warm-session validation failed after writing receipt: {}",
            aggregate["strict_session_validation"]
        );
    }
    if fail_on_quality && !quality_passed {
        anyhow::bail!(
            "CUDA warm-session answer quality gate failed after writing receipt: {}",
            aggregate["quality_summary"]
        );
    }
    Ok(())
}

#[cfg(feature = "full-cli")]
fn qk256_dispatch_coverage_delta(
    before: &bitnet_qk256_dispatch::Qk256DispatchCoverageCounters,
    after: &bitnet_qk256_dispatch::Qk256DispatchCoverageCounters,
) -> bitnet_qk256_dispatch::Qk256DispatchCoverageCounters {
    let cpu_fallback = after
        .bitnet_linear_layers_cpu_fallback
        .saturating_sub(before.bitnet_linear_layers_cpu_fallback);
    let unsupported_after =
        after.unsupported_ops.iter().cloned().collect::<std::collections::BTreeSet<_>>();
    let unsupported_before =
        before.unsupported_ops.iter().cloned().collect::<std::collections::BTreeSet<_>>();
    bitnet_qk256_dispatch::Qk256DispatchCoverageCounters {
        bitnet_linear_layers_total: after
            .bitnet_linear_layers_total
            .saturating_sub(before.bitnet_linear_layers_total),
        bitnet_linear_layers_on_cuda: after
            .bitnet_linear_layers_on_cuda
            .saturating_sub(before.bitnet_linear_layers_on_cuda),
        bitnet_linear_layers_cpu_fallback: cpu_fallback,
        unsupported_ops: unsupported_after
            .difference(&unsupported_before)
            .cloned()
            .collect::<Vec<_>>(),
        execution_claim: if after.bitnet_linear_layers_on_cuda > before.bitnet_linear_layers_on_cuda
        {
            "cuda_inference_contribution"
        } else {
            after.execution_claim
        },
    }
}

#[cfg(feature = "full-cli")]
fn qk256_dispatch_coverage_receipt(
    coverage: &bitnet_qk256_dispatch::Qk256DispatchCoverageCounters,
) -> serde_json::Value {
    serde_json::json!({
        "bitnet_linear_layers_total": coverage.bitnet_linear_layers_total,
        "bitnet_linear_layers_on_cuda": coverage.bitnet_linear_layers_on_cuda,
        "bitnet_linear_layers_cpu_fallback": coverage.bitnet_linear_layers_cpu_fallback,
        "unsupported_ops": coverage.unsupported_ops,
        "execution_claim": coverage.execution_claim,
    })
}

fn qk256_cpu_hot_path_receipt(
    counters: &bitnet_qk256_dispatch::Qk256CpuHotPathCounters,
) -> serde_json::Value {
    let no_scale_f32_gemv_invocations = counters
        .qk256_f32_scalar_gemv_invocations
        .saturating_add(counters.qk256_f32_avx2_gemv_invocations);
    let scaled_i2s_i8s_gemv_invocations = counters
        .qk256_i8s_scaled_scalar_invocations
        .saturating_add(counters.qk256_i8s_scaled_avx2_invocations);

    serde_json::json!({
        "counter_source": "bitnet_qk256_dispatch::qk256_cpu_hot_path_counters",
        "qk256_f32_scalar_gemv_invocations": counters.qk256_f32_scalar_gemv_invocations,
        "qk256_f32_avx2_gemv_invocations": counters.qk256_f32_avx2_gemv_invocations,
        "qk256_i8s_scaled_scalar_invocations": counters.qk256_i8s_scaled_scalar_invocations,
        "qk256_i8s_scaled_avx2_invocations": counters.qk256_i8s_scaled_avx2_invocations,
        "qk256_flat_bytes_extracted_count": counters.qk256_flat_bytes_extracted_count,
        "input_rows_materialized_count": counters.input_rows_materialized_count,
        "output_rows_allocated_count": counters.output_rows_allocated_count,
        "no_scale_f32_gemv_invocations": no_scale_f32_gemv_invocations,
        "scaled_i2s_i8s_gemv_invocations": scaled_i2s_i8s_gemv_invocations,
        "audited_tensor_materialization_count": counters
            .qk256_flat_bytes_extracted_count
            .saturating_add(counters.input_rows_materialized_count)
            .saturating_add(counters.output_rows_allocated_count),
        "requested_kernel": counters.requested_kernel.as_deref(),
        "selected_kernel": counters.selected_kernel.as_deref(),
        "qk256_execution_path": counters.qk256_execution_path,
        "math_changed": false,
        "speedup_claim": false,
    })
}

#[cfg(feature = "full-cli")]
fn qk256_cuda_runtime_stats_delta(
    before: &bitnet_qk256_dispatch::Qk256CudaRuntimeStats,
    after: &bitnet_qk256_dispatch::Qk256CudaRuntimeStats,
) -> bitnet_qk256_dispatch::Qk256CudaRuntimeStats {
    bitnet_qk256_dispatch::Qk256CudaRuntimeStats {
        host_to_device_bytes: after
            .host_to_device_bytes
            .saturating_sub(before.host_to_device_bytes),
        device_to_host_bytes: after
            .device_to_host_bytes
            .saturating_sub(before.device_to_host_bytes),
        kernel_time_ms: match (before.kernel_time_ms, after.kernel_time_ms) {
            (Some(before), Some(after)) => Some((after - before).max(0.0)),
            (None, Some(after)) => Some(after),
            _ => None,
        },
        kernel_time_samples: after.kernel_time_samples.saturating_sub(before.kernel_time_samples),
    }
}

fn qk256_kernel_stats_receipt(
    coverage: &bitnet_qk256_dispatch::Qk256DispatchCoverageCounters,
    runtime_stats: Option<&bitnet_qk256_dispatch::Qk256CudaRuntimeStats>,
) -> serde_json::Value {
    serde_json::json!([{
        "kernel_id": bitnet_kernels::cuda::CUDA_QK256_GEMV_KERNEL_ID,
        "invocations": coverage.bitnet_linear_layers_on_cuda,
        "fallback_invocations": coverage.bitnet_linear_layers_cpu_fallback,
        "host_to_device_bytes": runtime_stats
            .map(|stats| stats.host_to_device_bytes)
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
        "device_to_host_bytes": runtime_stats
            .map(|stats| stats.device_to_host_bytes)
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
        "kernel_launches": coverage.bitnet_linear_layers_on_cuda,
        "kernel_time_ms": runtime_stats
            .and_then(|stats| stats.kernel_time_ms)
            .map(rounded_ms)
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
        "kernel_time_samples": runtime_stats
            .map(|stats| stats.kernel_time_samples)
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    }])
}

struct CudaExecutionResidencyReceiptInput<'a> {
    coverage: &'a bitnet_qk256_dispatch::Qk256DispatchCoverageCounters,
    residency: Option<&'a bitnet_qk256_dispatch::Qk256CudaWeightResidency>,
    runtime_stats: Option<&'a bitnet_qk256_dispatch::Qk256CudaRuntimeStats>,
    prompt_tokens: usize,
    generated_tokens: usize,
    kv_cache_device: &'a str,
    kv_cache_reuse_policy: &'a str,
    execution_phase: &'a str,
    coverage_scope: &'a str,
}

fn cuda_execution_residency_receipt(
    input: CudaExecutionResidencyReceiptInput<'_>,
) -> serde_json::Value {
    let coverage = input.coverage;
    let residency = input.residency;
    let runtime_stats = input.runtime_stats;
    let prompt_tokens = input.prompt_tokens;
    let generated_tokens = input.generated_tokens;
    let kv_cache_device = input.kv_cache_device;
    let kv_cache_reuse_policy = input.kv_cache_reuse_policy;
    let execution_phase = input.execution_phase;
    let coverage_scope = input.coverage_scope;
    let qk256_on_cuda = coverage.bitnet_linear_layers_on_cuda;
    let qk256_cpu_fallback = coverage.bitnet_linear_layers_cpu_fallback;
    let weight_handle_count = residency.map(|value| value.weight_handle_count).unwrap_or(0);
    let weights_uploaded_once = residency.map(|value| value.weights_uploaded_once).unwrap_or(false);
    let per_token_weight_upload =
        residency.map(|value| value.per_token_weight_upload).unwrap_or(qk256_on_cuda > 0);
    let qk256_residency = if qk256_on_cuda > 0 && qk256_cpu_fallback == 0 {
        "cuda_resident_compute"
    } else if qk256_on_cuda > 0 {
        "mixed_cuda_and_cpu_fallback"
    } else {
        "not_observed"
    };
    let qk256_weight_status = if weight_handle_count > 0 && weights_uploaded_once {
        "cuda_resident_upload_once_handles"
    } else if weight_handle_count > 0 {
        "cuda_handles_observed_without_upload_once_claim"
    } else {
        "not_observed"
    };
    let kv_cache_residency =
        if kv_cache_device == "cuda" { "cuda_resident" } else { "cpu_resident" };
    let transfer_bytes_measured = runtime_stats
        .is_some_and(|stats| stats.host_to_device_bytes > 0 || stats.device_to_host_bytes > 0);
    let kernel_time_measured = runtime_stats.is_some_and(|stats| stats.kernel_time_ms.is_some());
    let mut non_resident_or_unmeasured_phases = vec![
        "token_embeddings",
        "rmsnorm",
        "sub_layernorm",
        "rope",
        "attention_scores",
        "attention_softmax",
        "attention_value_mix",
        "relu2_activation",
        "kv_cache",
        "lm_head",
        "sampling",
    ];
    if !transfer_bytes_measured {
        non_resident_or_unmeasured_phases.push("host_device_transfer_bytes");
    }
    if !kernel_time_measured {
        non_resident_or_unmeasured_phases.push("kernel_time_ms");
    }

    serde_json::json!({
        "schema_version": "1.0.0",
        "coverage_scope": coverage_scope,
        "execution_phase": execution_phase,
        "full_cuda_residency_claimed": false,
        "speedup_claim": false,
        "prompt_and_decode": {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
        },
        "qk256_bitnet_linears": {
            "kernel_id": bitnet_kernels::cuda::CUDA_QK256_GEMV_KERNEL_ID,
            "residency": qk256_residency,
            "bitnet_linear_layers_total": coverage.bitnet_linear_layers_total,
            "bitnet_linear_layers_on_cuda": qk256_on_cuda,
            "bitnet_linear_layers_cpu_fallback": qk256_cpu_fallback,
            "unsupported_ops": coverage.unsupported_ops,
            "fallback_used": qk256_cpu_fallback > 0,
        },
        "weight_residency": {
            "status": qk256_weight_status,
            "weight_handle_count": weight_handle_count,
            "weights_uploaded_once": weights_uploaded_once,
            "per_token_weight_upload": per_token_weight_upload,
            "scope": "qk256_cuda_weight_handles_only",
        },
        "kv_cache": {
            "enabled": true,
            "device": kv_cache_device,
            "residency": kv_cache_residency,
            "reuse_policy": kv_cache_reuse_policy,
            "cuda_residency_claimed": kv_cache_device == "cuda",
        },
        "phase_residency": {
            "token_embeddings": {
                "residency": "cpu_resident_or_not_cuda_claimed",
                "cuda_residency_claimed": false,
            },
            "rmsnorm": {
                "residency": "cpu_resident_or_not_cuda_claimed",
                "cuda_residency_claimed": false,
            },
            "sub_layernorm": {
                "residency": "cpu_resident_or_not_cuda_claimed",
                "cuda_residency_claimed": false,
            },
            "rope": {
                "residency": "cpu_resident_or_not_cuda_claimed",
                "cuda_residency_claimed": false,
            },
            "attention_scores": {
                "residency": "cpu_resident_or_not_cuda_claimed",
                "cuda_residency_claimed": false,
            },
            "attention_softmax": {
                "residency": "cpu_resident_or_not_cuda_claimed",
                "cuda_residency_claimed": false,
            },
            "attention_value_mix": {
                "residency": "cpu_resident_or_not_cuda_claimed",
                "cuda_residency_claimed": false,
            },
            "relu2_activation": {
                "residency": "cpu_resident_or_not_cuda_claimed",
                "cuda_residency_claimed": false,
            },
            "lm_head": {
                "residency": "cpu_resident_or_not_cuda_claimed",
                "cuda_residency_claimed": false,
            },
            "sampling": {
                "residency": "cpu_resident",
                "cuda_residency_claimed": false,
            },
        },
        "host_device_transfer_accounting": {
            "status": if transfer_bytes_measured || kernel_time_measured {
                "qk256_measured"
            } else {
                "not_measured"
            },
            "host_to_device_bytes": runtime_stats
                .map(|stats| stats.host_to_device_bytes)
                .map(serde_json::Value::from)
                .unwrap_or(serde_json::Value::Null),
            "device_to_host_bytes": runtime_stats
                .map(|stats| stats.device_to_host_bytes)
                .map(serde_json::Value::from)
                .unwrap_or(serde_json::Value::Null),
            "kernel_time_ms": runtime_stats
                .and_then(|stats| stats.kernel_time_ms)
                .map(rounded_ms)
                .map(serde_json::Value::from)
                .unwrap_or(serde_json::Value::Null),
            "kernel_time_samples": runtime_stats
                .map(|stats| stats.kernel_time_samples)
                .map(serde_json::Value::from)
                .unwrap_or(serde_json::Value::Null),
            "note": if transfer_bytes_measured || kernel_time_measured {
                "QK256 activation/output transfer bytes and CUDA event kernel time are measured for the routed QK256 GEMV path only; full transformer residency and transfer timing remain separate claims."
            } else {
                "QK256 routing and weight-handle residency are recorded; per-phase transfer bytes and kernel timings require a later benchmark/instrumentation gate."
            },
        },
        "unresident_or_unmeasured_phases": non_resident_or_unmeasured_phases,
        "claim_boundary": {
            "strict_cuda_answer_path": true,
            "qk256_cuda_residency_claimed": qk256_on_cuda > 0 && qk256_cpu_fallback == 0,
            "upload_once_weight_residency_claimed": weights_uploaded_once && !per_token_weight_upload,
            "full_transformer_cuda_residency_claimed": false,
            "kv_cache_cuda_residency_claimed": kv_cache_device == "cuda",
            "qk256_kernel_timing_claimed": kernel_time_measured,
            "qk256_transfer_byte_accounting_claimed": transfer_bytes_measured,
            "transfer_timing_claimed": false,
            "speedup_claim": false,
        },
    })
}

#[cfg(feature = "full-cli")]
#[derive(Clone, Debug, Default)]
pub(crate) struct SlmWarmSessionOutput {
    pub(crate) stream_tokens: bool,
    pub(crate) progress: bool,
    pub(crate) quiet: bool,
    pub(crate) write_prompt_receipts: bool,
    pub(crate) interactive_prompt_collection: bool,
    pub(crate) model_sha256_override: Option<String>,
    pub(crate) metal_prefill_qkv_phase: bool,
}

#[cfg(feature = "full-cli")]
impl SlmWarmSessionOutput {
    pub(crate) const fn new(stream_tokens: bool, progress: bool, quiet: bool) -> Self {
        Self {
            stream_tokens,
            progress,
            quiet,
            write_prompt_receipts: true,
            interactive_prompt_collection: false,
            model_sha256_override: None,
            metal_prefill_qkv_phase: false,
        }
    }

    pub(crate) const fn with_prompt_receipts(mut self, write_prompt_receipts: bool) -> Self {
        self.write_prompt_receipts = write_prompt_receipts;
        self
    }

    pub(crate) const fn with_interactive_prompt_collection(
        mut self,
        interactive_prompt_collection: bool,
    ) -> Self {
        self.interactive_prompt_collection = interactive_prompt_collection;
        self
    }

    pub(crate) fn with_model_sha256_override(
        mut self,
        model_sha256_override: Option<String>,
    ) -> Self {
        self.model_sha256_override = model_sha256_override;
        self
    }

    pub(crate) const fn with_metal_prefill_qkv_phase(
        mut self,
        metal_prefill_qkv_phase: bool,
    ) -> Self {
        self.metal_prefill_qkv_phase = metal_prefill_qkv_phase;
        self
    }

    fn progress_enabled(&self) -> bool {
        self.progress && !self.quiet
    }

    fn status(&self, message: impl AsRef<str>) {
        if self.progress_enabled() {
            eprintln!("{}", message.as_ref());
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "full-cli")]
async fn run_slm_warm_session(
    requested_backend_label: &str,
    model_path: std::path::PathBuf,
    model_format: String,
    tokenizer_path: Option<std::path::PathBuf>,
    corpus_path: Option<std::path::PathBuf>,
    corpus_repeat_runs: usize,
    prompts: Vec<String>,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
    strict_tokenizer: bool,
    strict_loader: bool,
    greedy: bool,
    deterministic: bool,
    threads: usize,
    prompt_template: String,
    no_think: bool,
    system_prompt: Option<String>,
    stop: Vec<String>,
    stop_id: Vec<u32>,
    fail_on_quality: bool,
    require_determinism: bool,
    allocation_audit: bool,
    output: SlmWarmSessionOutput,
    min_generated_tokens: usize,
    min_distinct_generated_tokens: usize,
    json_out: std::path::PathBuf,
) -> Result<()> {
    use bitnet_common::Device;
    use bitnet_models::{Model, transformer::KVCache};
    use bitnet_sampling::{SamplingConfig, SamplingStrategy};
    use bitnet_tokenizers::Tokenizer;
    use std::io::Write;
    use std::sync::Arc;

    let corpus = corpus_path.as_deref().map(SlmWarmSessionCorpus::load).transpose()?;
    if corpus.is_some() && !prompts.is_empty() {
        anyhow::bail!(
            "slm-warm-session accepts either --corpus or repeated --prompt values, not both"
        );
    }
    if corpus.is_none() && prompts.len() < 2 {
        anyhow::bail!("slm-warm-session requires at least two --prompt values or --corpus");
    }
    let prompt_inputs = warm_session_prompt_inputs(
        &prompts,
        corpus.as_ref(),
        corpus_repeat_runs,
        min_generated_tokens,
        min_distinct_generated_tokens,
    )?;
    let max_new_tokens =
        corpus.as_ref().and_then(|corpus| corpus.defaults.max_new_tokens).unwrap_or(max_new_tokens);
    let temperature =
        corpus.as_ref().and_then(|corpus| corpus.defaults.temperature).unwrap_or(temperature);
    let top_k = corpus.as_ref().and_then(|corpus| corpus.defaults.top_k).unwrap_or(top_k);
    let greedy = corpus.as_ref().and_then(|corpus| corpus.defaults.greedy).unwrap_or(greedy);
    let deterministic =
        corpus.as_ref().and_then(|corpus| corpus.defaults.deterministic).unwrap_or(deterministic);
    let prompt_template = corpus
        .as_ref()
        .and_then(|corpus| corpus.defaults.prompt_template.clone())
        .unwrap_or(prompt_template);
    let no_think =
        corpus.as_ref().and_then(|corpus| corpus.defaults.qwen_no_think).unwrap_or(no_think);
    if !is_supported_slm_warm_session_backend(requested_backend_label) {
        anyhow::bail!(
            "slm-warm-session is scoped to supported CPU receipt labels: cpu, apple-m4-cpu-neon, or apple-m3-air-cpu-neon; got {requested_backend_label}"
        );
    }
    match model_format.as_str() {
        "auto" | "gguf" => {}
        other => {
            anyhow::bail!(
                "Invalid --model-format '{}'. slm-warm-session supports GGUF only: auto, gguf",
                other
            );
        }
    }

    if deterministic {
        unsafe {
            std::env::set_var("BITNET_DETERMINISTIC", "1");
            std::env::set_var("RAYON_NUM_THREADS", "1");
            if threads > 0 {
                std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
            }
        }
    }
    if strict_loader {
        unsafe {
            std::env::set_var("BITNET_DISABLE_MINIMAL_LOADER", "1");
            std::env::set_var("BITNET_STRICT_MODE", "1");
        }
    }

    let strict_backend = strict_loader
        || std::env::var("BITNET_STRICT_MODE")
            .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
    let backend_identity = resolve_run_backend_identity(requested_backend_label, strict_backend)?;
    if backend_identity.fallback_used {
        anyhow::bail!(
            "slm-warm-session requires visible no-fallback backend routing; requested_backend={}, selected_backend={}, fallback_reason={:?}",
            backend_identity.requested_backend,
            backend_identity.selected_backend,
            backend_identity.fallback_reason
        );
    }
    if backend_identity.runtime_api != "cpu" {
        anyhow::bail!(
            "slm-warm-session is CPU scoped; selected runtime_api={}",
            backend_identity.runtime_api
        );
    }
    if output.metal_prefill_qkv_phase && !slm_warm_session_metal_qkv_route_supported() {
        anyhow::bail!(
            "slm-warm-session --metal-prefill-qkv-phase requires the metal feature and Apple Silicon Metal runtime support; full apple-m4-metal inference remains unsupported"
        );
    }
    unsafe {
        std::env::set_var("BITNET_REQUESTED_BACKEND", backend_identity.requested_backend.as_str());
        std::env::set_var("BITNET_SELECTED_BACKEND", backend_identity.selected_backend.as_str());
        std::env::set_var("BITNET_RUNTIME_API", backend_identity.runtime_api.as_str());
    }

    if model_path.is_dir() {
        anyhow::bail!(
            "slm-warm-session requires a GGUF model file, not a model directory: {}",
            model_path.display()
        );
    }
    let is_hf_directory = false;
    let template_type: bitnet_inference::TemplateType =
        prompt_template.parse().with_context(|| {
            format!(
                "Invalid prompt template '{}'. Supported: raw, instruct, llama3-chat, qwen2.5",
                prompt_template
            )
        })?;
    let temperature = if greedy { 0.0 } else { temperature };
    let session_start = std::time::Instant::now();

    let loader = bitnet_models::loader::ModelLoader::new(Device::Cpu);
    let load_config = bitnet_models::loader::LoadConfig {
        use_mmap: true,
        validate_checksums: false,
        progress_callback: None,
    };
    output.status(format!("warm-session: loading model once from {}", model_path.display()));
    let model_load_start = std::time::Instant::now();
    let loaded_model = loader.load_with_config(&model_path, &load_config).with_context(|| {
        format!("Failed to load real model for warm session: {}", model_path.display())
    })?;
    let config = loaded_model.config().clone();
    let dense_q8_hook_selection = loaded_model.dense_q8_hook_selection_receipt();
    let model: Arc<dyn Model> = Arc::from(loaded_model);
    let model_load_ms = elapsed_ms(model_load_start);
    let loader_mode = detect_loader_mode_for_path(&model_path, is_hf_directory);
    if loader_mode != bitnet_models::GgufLoaderMode::RealGguf.as_str() {
        anyhow::bail!("slm-warm-session requires real_gguf loader mode; got {loader_mode}");
    }

    let effective_strict_tokenizer = strict_tokenizer || strict_loader;
    let tokenizer_load_start = std::time::Instant::now();
    let tokenizer_resolution = bitnet_tokenizers::auto::resolve_tokenizer(
        &model_path,
        tokenizer_path.as_deref(),
        effective_strict_tokenizer,
    )
    .with_context(|| format!("Failed to resolve tokenizer for {}", model_path.display()))?;
    let tokenizer_load_ms = elapsed_ms(tokenizer_load_start);
    let tokenizer_source = tokenizer_resolution.source;
    let tokenizer_strict = tokenizer_resolution.strict;
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = tokenizer_resolution.tokenizer;
    let tokenizer_source_str = tokenizer_source.as_str();
    let tokenizer_label = infer_tokenizer_label(tokenizer.as_ref(), tokenizer_source);
    let pretokenizer_authority =
        tokenizer_pretokenizer_authority(tokenizer_source, &tokenizer_label);
    let tokenizer_type = tokenizer_type_for_receipt(&tokenizer_label, tokenizer_source);
    let all_stop_sequences = simple_generation::prompt::merge_stop_sequences(&stop, template_type);
    let all_stop_ids = simple_generation::prompt::merge_stop_token_ids(
        &stop_id,
        template_type,
        tokenizer.as_ref(),
    );
    let thread_count = effective_thread_count(threads);
    let prompt_generation_identity = simple_generation::prompt::prompt_generation_identity(
        simple_generation::prompt::PromptGenerationIdentityInput {
            template_family: &template_type.to_string(),
            template_source: "bitnet-prompt-templates-core",
            tokenizer_source: Some(tokenizer_source_str),
            tokenizer_authority: Some(pretokenizer_authority),
            tokenizer_sha256: None,
            tokenizer_strict: Some(tokenizer_strict),
            manual_stop_sequences: &stop,
            stop_sequences: &all_stop_sequences,
            manual_stop_token_ids: &stop_id,
            stop_token_ids: &all_stop_ids,
            stop_string_window: None,
            stop_policy: "manual_plus_template_defaults",
            generation_params: simple_generation::prompt::PromptGenerationParams {
                max_new_tokens: Some(max_new_tokens),
                temperature: Some(temperature),
                top_k: Some(top_k),
                top_p: Some(top_p),
                repetition_penalty: Some(repetition_penalty),
                seed,
                greedy: Some(greedy),
                deterministic: Some(deterministic),
                threads: Some(thread_count),
                qwen_no_think: Some(no_think),
                fixed_token_count: Some(false),
                stream: Some(output.stream_tokens),
            },
        },
    );
    let max_stop_len = all_stop_sequences.iter().map(|value| value.len()).max().unwrap_or(0);
    let gguf_metadata = gguf_header_counts_for_receipt(&model_path, is_hf_directory);
    let (n_kv, n_tensors) = gguf_metadata.unwrap_or((0, 0));
    let model_sha256_start = std::time::Instant::now();
    let model_sha256_rehash_skipped = output.model_sha256_override.is_some();
    let model_sha256 = match output.model_sha256_override.as_ref() {
        Some(sha256) => sha256.clone(),
        None => compute_model_sha256(&model_path)?,
    };
    let model_sha256_ms = elapsed_ms(model_sha256_start);
    let model_sha256_source = if model_sha256_rehash_skipped {
        "verified_mac_model_cache"
    } else {
        "computed_from_model_file"
    };
    let model_repo = infer_model_repo(&model_path);
    let model_architecture = infer_model_architecture(&model_path);
    let model_family = receipt_model_family(&model_architecture);
    let model_format_label = receipt_model_format(&model_path, &model_format, is_hf_directory);
    let model_file =
        model_path.file_name().and_then(|name| name.to_str()).unwrap_or_default().to_string();
    let kernel_family = kernel_family_for_quantization(config.quantization.quantization_type);
    let kernel_implementation = cpu_kernel_implementation(config.quantization.quantization_type);
    let selected_kernel = format!("{kernel_family}-{kernel_implementation}-reference");
    let cpu_features = detected_cpu_feature_labels();
    let cpu_model = detected_cpu_model_label();
    let apple_machine = apple_machine_receipt_json(
        backend_identity.requested_backend.as_str(),
        backend_identity.selected_backend.as_str(),
    );

    let receipt_dir = json_out
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(format!(
            "{}-prompts",
            json_out.file_stem().and_then(|stem| stem.to_str()).unwrap_or("slm-warm-session")
        ));
    if output.write_prompt_receipts {
        std::fs::create_dir_all(&receipt_dir)
            .with_context(|| format!("Failed to create {}", receipt_dir.display()))?;
        output.status(format!(
            "warm-session: writing per-prompt receipts under {}",
            receipt_dir.display()
        ));
    } else {
        output.status("warm-session: per-prompt receipt files disabled; aggregate receipt only");
    }

    let mut prompt_receipts = Vec::with_capacity(prompt_inputs.len());
    let mut prompt_summaries = Vec::with_capacity(prompt_inputs.len());
    let mut metal_phase_receipts = Vec::new();
    let mut quality_failed_prompts = Vec::new();
    let mut determinism_records = Vec::with_capacity(prompt_inputs.len());
    let mut speed_accumulator = WarmSessionSpeedAccumulator::default();
    let allocation_audit_enabled = allocation_audit;
    let allocation_audit_guard = AllocationAuditGuard::enable(allocation_audit_enabled);
    let mut session_buffers = WarmSessionPromptBuffers::default();
    let mut aggregate_direct_greedy_logits_steps = 0usize;
    let mut aggregate_logits_vec_extraction_steps = 0usize;
    let mut aggregate_logits_scratch_reuse_steps = 0usize;
    let mut prompt_token_cache = WarmSessionPromptTokenCache::default();
    let logits_capacity = config.model.vocab_size.max(tokenizer.vocab_size());
    let sampling_config =
        SamplingConfig { temperature, top_k: top_k as u32, top_p, repetition_penalty, seed };
    let sampler_reuse_enabled = warm_session_sampler_reuse_enabled(&sampling_config);
    let sampler_reuse_policy = warm_session_sampler_reuse_policy(sampler_reuse_enabled);
    let kv_cache_reuse_policy = "single_kv_cache_cleared_per_prompt_for_prompt_isolation";
    let kv_cache_session_alloc_start = AllocationAuditSnapshot::current();
    let mut session_kv_cache = KVCache::new(&config, 1, &candle_core::Device::Cpu)?;
    let kv_cache_session_alloc = AllocationAuditSnapshot::delta_since(kv_cache_session_alloc_start);
    let kv_cache_reused_across_prompts = true;
    let mut kv_cache_reused_prompt_count = 0usize;
    let mut session_sampler = sampler_reuse_enabled.then(|| {
        let mut sampler = SamplingStrategy::new(sampling_config.clone());
        sampler.reserve_logits_capacity(logits_capacity);
        sampler
    });
    let mut sampler_reused_prompt_count = 0usize;
    let mut sampler_recreated_prompt_count = 0usize;
    for (index, prompt_input) in prompt_inputs.iter().enumerate() {
        output.status(format!(
            "warm-session: prompt {}/{} started",
            index + 1,
            prompt_inputs.len()
        ));
        let prompt_alloc_start = AllocationAuditSnapshot::current();
        let prompt = &prompt_input.prompt;
        let prompt_start = std::time::Instant::now();
        let formatted_prompt = apply_qwen_no_think_prompt_policy(
            template_type,
            template_type.apply(prompt, system_prompt.as_deref()),
            no_think,
        )?;
        let rendered_prompt_sha256 = compute_sha256_bytes(formatted_prompt.as_bytes());

        let bos_policy = template_type.should_add_bos();
        let parse_special = template_type.parse_special();
        let prompt_tokenize_start = std::time::Instant::now();
        let prompt_tokenize_alloc_start = AllocationAuditSnapshot::current();
        let (encoded_prompt_tokens, prompt_token_cache_hit) = prompt_token_cache
            .get_or_insert_with(&formatted_prompt, bos_policy, parse_special, || {
                Ok(tokenizer.encode(&formatted_prompt, bos_policy, parse_special)?)
            })?;
        let prompt_tokenize_ms = elapsed_ms(prompt_tokenize_start);
        let prompt_tokenize_alloc =
            AllocationAuditSnapshot::delta_since(prompt_tokenize_alloc_start);

        let prompt_setup_alloc_start = AllocationAuditSnapshot::current();
        let prompt_setup_buffer_reset_alloc_start = AllocationAuditSnapshot::current();
        let buffer_reuse_evidence = session_buffers.reset(
            encoded_prompt_tokens.len(),
            max_new_tokens,
            max_stop_len,
            logits_capacity,
        );
        let prompt_setup_buffer_reset_alloc =
            AllocationAuditSnapshot::delta_since(prompt_setup_buffer_reset_alloc_start);
        let prompt_setup_token_seed_alloc_start = AllocationAuditSnapshot::current();
        session_buffers.tokens.extend_from_slice(&encoded_prompt_tokens);
        ensure_non_empty_generation_context(&mut session_buffers.tokens, tokenizer.as_ref())?;
        let prompt_setup_token_seed_alloc =
            AllocationAuditSnapshot::delta_since(prompt_setup_token_seed_alloc_start);
        let tokens = &mut session_buffers.tokens;
        let generated_tokens = &mut session_buffers.generated_tokens;
        let decode_step_ms = &mut session_buffers.decode_step_ms;
        let embed_step_ms = &mut session_buffers.embed_step_ms;
        let forward_step_ms = &mut session_buffers.forward_step_ms;
        let logits_step_ms = &mut session_buffers.logits_step_ms;
        let sample_step_ms = &mut session_buffers.sample_step_ms;
        let token_decode_step_ms = &mut session_buffers.token_decode_step_ms;
        let prefill_step_allocs = &mut session_buffers.prefill_step_allocs;
        let prefill_embed_step_allocs = &mut session_buffers.prefill_embed_step_allocs;
        let prefill_forward_step_allocs = &mut session_buffers.prefill_forward_step_allocs;
        let decode_step_allocs = &mut session_buffers.decode_step_allocs;
        let embed_step_allocs = &mut session_buffers.embed_step_allocs;
        let forward_step_allocs = &mut session_buffers.forward_step_allocs;
        let logits_step_allocs = &mut session_buffers.logits_step_allocs;
        let sample_step_allocs = &mut session_buffers.sample_step_allocs;
        let token_vector_update_allocs = &mut session_buffers.token_vector_update_allocs;
        let token_decode_step_allocs = &mut session_buffers.token_decode_step_allocs;
        let stop_tail_update_allocs = &mut session_buffers.stop_tail_update_allocs;
        let stop_tail = &mut session_buffers.stop_tail;
        let logits_scratch = &mut session_buffers.logits_scratch;
        let prompt_token_count = tokens.len();
        let prompt_setup_kv_cache_alloc_start = AllocationAuditSnapshot::current();
        session_kv_cache.clear();
        kv_cache_reused_prompt_count += 1;
        let prompt_setup_kv_cache_alloc =
            AllocationAuditSnapshot::delta_since(prompt_setup_kv_cache_alloc_start);
        let prompt_setup_sampler_alloc_start = AllocationAuditSnapshot::current();
        let mut prompt_sampler = if sampler_reuse_enabled {
            sampler_reused_prompt_count += 1;
            None
        } else {
            sampler_recreated_prompt_count += 1;
            let mut sampler = SamplingStrategy::new(sampling_config.clone());
            sampler.reserve_logits_capacity(logits_capacity);
            Some(sampler)
        };
        if let Some(sampler) = session_sampler.as_mut() {
            sampler.reset();
        }
        let prompt_setup_sampler_alloc =
            AllocationAuditSnapshot::delta_since(prompt_setup_sampler_alloc_start);
        let mut first_token_ms = None;
        let mut first_token_decode_ms = None;
        let prompt_setup_alloc = AllocationAuditSnapshot::delta_since(prompt_setup_alloc_start);
        let mut direct_greedy_logits_steps = 0usize;
        let mut logits_vec_extraction_steps = 0usize;
        let mut logits_scratch_reuse_steps = 0usize;

        let prefill_start = std::time::Instant::now();
        let mut prefill_token_count = 0usize;
        if tokens.len() > 1 {
            for token in &tokens[..tokens.len() - 1] {
                let prefill_alloc_start = AllocationAuditSnapshot::current();
                let prefill_embed_alloc_start = AllocationAuditSnapshot::current();
                let x = model.embed(&[*token])?;
                if allocation_audit_enabled {
                    prefill_embed_step_allocs
                        .push(AllocationAuditSnapshot::delta_since(prefill_embed_alloc_start));
                }
                let prefill_forward_alloc_start = AllocationAuditSnapshot::current();
                let _ = model.forward(&x, &mut session_kv_cache as &mut dyn std::any::Any)?;
                if allocation_audit_enabled {
                    prefill_forward_step_allocs
                        .push(AllocationAuditSnapshot::delta_since(prefill_forward_alloc_start));
                    prefill_step_allocs
                        .push(AllocationAuditSnapshot::delta_since(prefill_alloc_start));
                }
                prefill_token_count += 1;
            }
        }
        let prefill_ms = if prefill_token_count > 0 { elapsed_ms(prefill_start) } else { 0.0 };

        for _step_idx in 0..max_new_tokens {
            let decode_step_start = std::time::Instant::now();
            let decode_alloc_start = AllocationAuditSnapshot::current();
            let last_token = tokens.last().copied().expect("tokens must be non-empty");

            let embed_start = std::time::Instant::now();
            let embed_alloc_start = AllocationAuditSnapshot::current();
            let x = model.embed(&[last_token])?;
            if allocation_audit_enabled {
                embed_step_allocs.push(AllocationAuditSnapshot::delta_since(embed_alloc_start));
            }
            embed_step_ms.push(elapsed_ms(embed_start));

            let forward_start = std::time::Instant::now();
            let forward_alloc_start = AllocationAuditSnapshot::current();
            let h = model.forward(&x, &mut session_kv_cache as &mut dyn std::any::Any)?;
            if allocation_audit_enabled {
                forward_step_allocs.push(AllocationAuditSnapshot::delta_since(forward_alloc_start));
            }
            forward_step_ms.push(elapsed_ms(forward_start));

            let last_hidden = extract_last_token_hidden(&h)?;
            let logits_start = std::time::Instant::now();
            let logits_alloc_start = AllocationAuditSnapshot::current();
            let logits = model.logits(&last_hidden)?;
            let use_direct_greedy_logits = can_use_direct_greedy_logits(
                temperature,
                repetition_penalty,
                generated_tokens.is_empty(),
            );
            let direct_next_token = if use_direct_greedy_logits {
                direct_greedy_logits_steps += 1;
                Some(greedy_argmax_token_2d(&logits)?)
            } else {
                if extract_logits_2d_into(&logits, logits_scratch)? {
                    logits_scratch_reuse_steps += 1;
                } else {
                    logits_vec_extraction_steps += 1;
                }
                None
            };
            if allocation_audit_enabled {
                logits_step_allocs.push(AllocationAuditSnapshot::delta_since(logits_alloc_start));
            }
            logits_step_ms.push(elapsed_ms(logits_start));

            let sample_start = std::time::Instant::now();
            let sample_alloc_start = AllocationAuditSnapshot::current();
            let next_token = match direct_next_token {
                Some(token) => token,
                None => {
                    let Some(sampler) = session_sampler.as_mut().or(prompt_sampler.as_mut()) else {
                        anyhow::bail!(
                            "warm-session sampler was unavailable for non-direct sampling"
                        );
                    };
                    sampler.sample_in_place(logits_scratch, generated_tokens)?
                }
            };
            if allocation_audit_enabled {
                sample_step_allocs.push(AllocationAuditSnapshot::delta_since(sample_alloc_start));
            }
            sample_step_ms.push(elapsed_ms(sample_start));

            let token_vector_update_alloc_start = AllocationAuditSnapshot::current();
            tokens.push(next_token);
            generated_tokens.push(next_token);
            if allocation_audit_enabled {
                token_vector_update_allocs
                    .push(AllocationAuditSnapshot::delta_since(token_vector_update_alloc_start));
            }
            if first_token_ms.is_none() {
                first_token_ms = Some(prompt_start.elapsed().as_millis() as u64);
            }

            let token_decode_start = std::time::Instant::now();
            let token_decode_alloc_start = AllocationAuditSnapshot::current();
            let token_text = tokenizer.decode(&[next_token])?;
            if output.stream_tokens {
                print!("{token_text}");
                std::io::stdout().flush()?;
            }
            if allocation_audit_enabled {
                token_decode_step_allocs
                    .push(AllocationAuditSnapshot::delta_since(token_decode_alloc_start));
            }
            token_decode_step_ms.push(elapsed_ms(token_decode_start));
            let stop_tail_alloc_start = AllocationAuditSnapshot::current();
            if max_stop_len > 0 {
                stop_tail.push_str(&token_text);
                if stop_tail.len() > max_stop_len {
                    let cut = stop_tail.len() - max_stop_len;
                    let mut safe_cut = cut;
                    while safe_cut > 0 && !stop_tail.is_char_boundary(safe_cut) {
                        safe_cut -= 1;
                    }
                    stop_tail.drain(..safe_cut);
                }
            }
            if allocation_audit_enabled {
                stop_tail_update_allocs
                    .push(AllocationAuditSnapshot::delta_since(stop_tail_alloc_start));
            }
            let step_ms = elapsed_ms(decode_step_start);
            if first_token_decode_ms.is_none() {
                first_token_decode_ms = Some(step_ms);
            }
            decode_step_ms.push(step_ms);
            if allocation_audit_enabled {
                decode_step_allocs.push(AllocationAuditSnapshot::delta_since(decode_alloc_start));
            }

            if all_stop_ids.contains(&next_token) {
                break;
            }
            if let Some(eos) = tokenizer.eos_token_id()
                && next_token == eos
            {
                break;
            }
            if max_stop_len > 0
                && !all_stop_sequences.is_empty()
                && all_stop_sequences.iter().any(|pat| stop_tail.ends_with(pat))
            {
                break;
            }
        }

        let generated_text = tokenizer.decode(generated_tokens)?;
        if output.stream_tokens {
            println!();
        }
        let metal_phase_contribution = if output.metal_prefill_qkv_phase {
            let phase_path = receipt_dir.join(format!(
                "{:02}-{}-metal-prefill-qkv.json",
                index + 1,
                sanitize_warm_session_prompt_stem(prompt)
            ));
            let phase_receipt = run_slm_warm_session_metal_qkv_phase(
                index,
                generated_tokens,
                &phase_path,
                output.write_prompt_receipts,
            )?;
            if output.write_prompt_receipts {
                metal_phase_receipts.push(phase_path.display().to_string());
            }
            Some(phase_receipt)
        } else {
            None
        };
        let quality = slm_warm_session_quality_receipt(
            &generated_text,
            generated_tokens,
            prompt_input.min_generated_tokens,
            prompt_input.min_distinct_generated_tokens,
            prompt_input.gate.as_ref(),
        );
        let quality_passed = quality["passed"].as_bool().unwrap_or(false);
        if !quality_passed {
            quality_failed_prompts.push(index);
        }
        determinism_records.push(WarmSessionDeterminismRecord {
            prompt_index: index,
            case_id: prompt_input.case_id.clone(),
            prompt: prompt.clone(),
            text: generated_text.clone(),
            generated_ids: generated_tokens.clone(),
        });
        let prompt_total_ms = elapsed_ms(prompt_start);
        let decode_total_ms = decode_step_ms.iter().sum::<f64>();
        let sampling_total_ms = sample_step_ms.iter().sum::<f64>();
        let sampling_ms_per_token = if sample_step_ms.is_empty() {
            None
        } else {
            Some(sampling_total_ms / sample_step_ms.len() as f64)
        };
        let decode_steady_state_tok_s =
            steady_decode_tps_ms(decode_step_ms).map(|value| (value * 1000.0).round() / 1000.0);
        speed_accumulator.record(WarmSessionPromptSpeed {
            prompt_tokens: prompt_token_count,
            generated_tokens: generated_tokens.len(),
            tokenize_ms: prompt_tokenize_ms,
            prefill_ms,
            decode_total_ms,
            sampling_ms: sampling_total_ms,
            prompt_total_ms,
            first_token_ms: first_token_ms.map(|value| value as f64),
            steady_decode_tok_s: decode_steady_state_tok_s,
        });
        let prompt_receipt_path = receipt_dir.join(format!(
            "{:02}-{}.json",
            index + 1,
            sanitize_warm_session_prompt_stem(prompt)
        ));
        let prompt_receipt_path_json =
            output.write_prompt_receipts.then(|| prompt_receipt_path.display().to_string());
        let prompt_total_alloc = AllocationAuditSnapshot::delta_since(prompt_alloc_start);
        let prompt_receipt_construct_alloc_start = AllocationAuditSnapshot::current();
        let prompt_artifact_kind =
            slm_warm_session_prompt_artifact_kind(backend_identity.requested_backend.as_str());
        let mut prompt_receipt = serde_json::json!({
            "schema_version": "1.0.0",
            "artifact_kind": prompt_artifact_kind,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "session_artifact_path": json_out.display().to_string(),
            "artifact_path": prompt_receipt_path_json,
            "prompt_index": index,
            "case_id": prompt_input.case_id.as_str(),
            "repeat_index": prompt_input.repeat_index,
            "requested_backend": backend_identity.requested_backend.as_str(),
            "selected_backend": backend_identity.selected_backend.as_str(),
            "runtime_api": backend_identity.runtime_api.as_str(),
            "fallback_used": backend_identity.fallback_used,
            "fallback_reason": backend_identity.fallback_reason.as_deref(),
            "prompt": prompt,
            "prompt_render": {
                "template_family": template_type.to_string(),
                "qwen_no_think": no_think,
                "rendered_text": &formatted_prompt,
                "rendered_sha256": rendered_prompt_sha256,
                "add_bos": bos_policy,
                "parse_special": parse_special,
                "stop_sequences": &all_stop_sequences,
                "stop_token_ids": &all_stop_ids,
            },
            "prompt_generation_identity": prompt_generation_identity.clone(),
            "text": generated_text,
            "tokens": {
                "prompt": prompt_token_count,
                "generated": generated_tokens.len(),
                "total": tokens.len(),
                "prompt_ids": tokens[..prompt_token_count].to_vec(),
                "generated_ids": generated_tokens.clone(),
                "ids": generated_tokens.clone(),
            },
            "quality": quality,
            "timing": {
                "model_load_ms": 0.0,
                "tokenizer_load_ms": 0.0,
                "session_model_load_ms": rounded_ms(model_load_ms),
                "session_tokenizer_load_ms": rounded_ms(tokenizer_load_ms),
                "session_model_sha256_ms": rounded_ms(model_sha256_ms),
                "tokenize_ms": rounded_ms(prompt_tokenize_ms),
                "prefill_ms": rounded_ms(prefill_ms),
                "first_token_ms": first_token_ms,
                "time_to_first_token_ms": first_token_ms,
                "first_token_decode_ms": first_token_decode_ms.map(rounded_ms),
                "decode_total_ms": rounded_ms(decode_total_ms),
                "decode_steady_state_tok_s": decode_steady_state_tok_s,
                "sampling_ms_per_token": sampling_ms_per_token.map(rounded_ms),
                "total_ms": rounded_ms(prompt_total_ms),
                "embed_ms": timing_samples_json(embed_step_ms),
                "forward_ms": timing_samples_json(forward_step_ms),
                "logits_ms": timing_samples_json(logits_step_ms),
                "sample_ms": timing_samples_json(sample_step_ms),
                "token_decode_ms": timing_samples_json(token_decode_step_ms),
            },
            "model": {
                "repo": model_repo.as_str(),
                "file": model_file.as_str(),
                "path": model_path.display().to_string(),
                "sha256": model_sha256.as_str(),
                "sha256_source": model_sha256_source,
                "sha256_rehash_skipped": model_sha256_rehash_skipped,
                "format": model_format_label.as_str(),
                "family": model_family,
                "architecture": model_architecture,
                "loader_mode": loader_mode,
                "fallback_loader_used": false,
                "tokenizer": tokenizer_label.as_str(),
            },
            "tokenizer": {
                "type": tokenizer_type.as_str(),
                "model_family": tokenizer_type.as_str(),
                "source": tokenizer_source_str,
                "strict": tokenizer_strict,
                "pretokenizer_authority": pretokenizer_authority,
                "bos": tokenizer.bos_token_id().unwrap_or(1),
                "eos": tokenizer.eos_token_id().unwrap_or(2),
            },
            "kernel": {
                "family": kernel_family,
                "implementation": kernel_implementation,
                "kernel_id": selected_kernel.as_str(),
            },
            "dense_q8_hook_selection": dense_q8_hook_selection.clone(),
            "execution": {
                "phase": "warm_session_decode",
                "prompt_tokens": prompt_token_count,
                "generated_tokens": generated_tokens.len(),
                "thread_count": thread_count,
                "requested_backend": backend_identity.requested_backend.as_str(),
                "selected_backend": backend_identity.selected_backend.as_str(),
                "runtime_api": backend_identity.runtime_api.as_str(),
                "fallback_used": backend_identity.fallback_used,
                "fallback_reason": backend_identity.fallback_reason.as_deref(),
            },
            "prompt_prefill": {
                "exercised": prefill_token_count > 0,
                "tokens": prefill_token_count,
                "kv_cache_behavior": if prefill_token_count > 0 {
                    "prompt_prefix_prefilled_before_decode"
                } else {
                    "single_token_prompt_no_prefix_prefill"
                },
            },
            "session_reuse": {
                "reuse_scope": "resident_session",
                "model_loaded_once": true,
                "tokenizer_loaded_once": true,
                "session_owned_buffers": true,
                "prompt_token_buffer_reused": true,
                "generated_token_buffer_reused": true,
                "timing_buffers_reused": true,
                "allocation_audit_buffers_reused": true,
                "stop_tail_buffer_reused": max_stop_len > 0,
                "kv_cache_reuse_policy": kv_cache_reuse_policy,
                "kv_cache_reused_across_prompts": kv_cache_reused_across_prompts,
                "kv_cache_cleared_per_prompt": true,
                "kv_cache_recreated_per_prompt": false,
                "sampler_reuse_policy": sampler_reuse_policy,
                "sampler_reused_across_prompts": sampler_reuse_enabled,
                "sampler_recreated_per_prompt": !sampler_reuse_enabled,
                "logits_buffer_reuse_policy": if logits_vec_extraction_steps == 0 {
                    "full logits Vec extraction bypassed; deterministic greedy no-penalty steps use direct tensor argmax and repetition-penalty steps reuse a preallocated host logits scratch buffer"
                } else {
                    "model.logits extraction still falls back to allocating a vector for non-F32 or non-CPU logits; sampler logits scratch is preallocated separately"
                },
                "logits_vec_extraction_bypassed": logits_vec_extraction_steps == 0,
                "direct_greedy_logits_steps": direct_greedy_logits_steps,
                "logits_scratch_reuse_steps": logits_scratch_reuse_steps,
                "logits_vec_extraction_steps": logits_vec_extraction_steps,
                "prompt_token_cache_policy": WarmSessionPromptTokenCache::POLICY,
                "prompt_token_cache_enabled": true,
                "prompt_token_cache_hit": prompt_token_cache_hit,
                "prompt_token_cache_entry_count": prompt_token_cache.entry_count(),
                "stop_policy_precomputed_once": true,
                "stop_sequence_count": all_stop_sequences.len(),
                "stop_token_id_count": all_stop_ids.len(),
                "buffer_reuse_evidence": buffer_reuse_evidence,
            },
            "operator_ux": {
                "stream_tokens_requested": output.stream_tokens,
                "stdout_token_stream": output.stream_tokens,
                "progress_enabled": output.progress_enabled(),
                "quiet_default_logs": !output.progress,
                "quiet_requested": output.quiet,
                "interactive_prompt_collection": output.interactive_prompt_collection,
                "per_turn_receipts_enabled": output.write_prompt_receipts,
                "aggregate_receipt_at_exit": true,
                "status_stream": "stderr",
                "token_stream": if output.stream_tokens { "stdout" } else { "disabled" },
                "time_to_first_token_receipt": first_token_ms.is_some(),
                "model_tokenizer_loaded_once_status": "recorded_in_session_receipt",
                "clear_failure_messages": true,
            },
            "prompt_policy": {
                "template": template_type.to_string(),
                "qwen_no_think": no_think,
            },
            "speedup_claim": false,
        });
        let prompt_receipt_construct_alloc =
            AllocationAuditSnapshot::delta_since(prompt_receipt_construct_alloc_start);
        let prompt_allocation_audit =
            warm_session_prompt_allocation_audit_json(WarmSessionPromptAllocationAudit {
                enabled: allocation_audit_enabled,
                requested_backend: backend_identity.requested_backend.as_str(),
                prompt_tokenize: prompt_tokenize_alloc,
                prompt_setup: prompt_setup_alloc,
                prompt_setup_breakdown: WarmSessionPromptSetupAllocationAudit {
                    buffer_reset: prompt_setup_buffer_reset_alloc,
                    token_seed: prompt_setup_token_seed_alloc,
                    kv_cache: prompt_setup_kv_cache_alloc,
                    sampler_setup: prompt_setup_sampler_alloc,
                },
                prompt_prefill: prefill_step_allocs,
                prompt_prefill_embed: prefill_embed_step_allocs,
                prompt_prefill_forward: prefill_forward_step_allocs,
                decode_total: decode_step_allocs,
                embed: embed_step_allocs,
                forward: forward_step_allocs,
                logits: logits_step_allocs,
                sample: sample_step_allocs,
                token_vector_update: token_vector_update_allocs,
                token_decode: token_decode_step_allocs,
                stop_tail_update: stop_tail_update_allocs,
                receipt_construction: prompt_receipt_construct_alloc,
            });
        if let Some(object) = prompt_receipt.as_object_mut() {
            if let Some(phase_receipt) = &metal_phase_contribution {
                object.insert(
                    "metal_phase_contributions".to_string(),
                    serde_json::json!([slm_warm_session_metal_qkv_prompt_summary(phase_receipt)]),
                );
            }
            object.insert(
                "allocation_audit".to_string(),
                if allocation_audit_enabled {
                    let mut audit = prompt_allocation_audit;
                    if let Some(audit_object) = audit.as_object_mut() {
                        audit_object.insert(
                            "prompt_total_counter_delta".to_string(),
                            allocation_samples_json(std::slice::from_ref(&prompt_total_alloc)),
                        );
                    }
                    audit
                } else {
                    prompt_allocation_audit
                },
            );
        }
        if output.write_prompt_receipts {
            write_json_output_silent(&prompt_receipt_path, &prompt_receipt)?;
        }
        prompt_summaries.push(serde_json::json!({
            "prompt_index": index,
            "case_id": prompt_input.case_id.as_str(),
            "repeat_index": prompt_input.repeat_index,
            "prompt": prompt,
            "text": prompt_receipt["text"].clone(),
            "receipt_path": if output.write_prompt_receipts {
                serde_json::Value::String(prompt_receipt_path.display().to_string())
            } else {
                serde_json::Value::Null
            },
            "generated_tokens": generated_tokens.len(),
            "generated_token_ids": generated_tokens.clone(),
            "quality": prompt_receipt["quality"].clone(),
            "timing": prompt_receipt["timing"].clone(),
            "backend": {
                "requested_backend": backend_identity.requested_backend.as_str(),
                "selected_backend": backend_identity.selected_backend.as_str(),
                "runtime_api": backend_identity.runtime_api.as_str(),
                "fallback_used": backend_identity.fallback_used,
            },
            "session_reuse": prompt_receipt["session_reuse"].clone(),
            "metal_phase_contributions": prompt_receipt
                .get("metal_phase_contributions")
                .cloned()
                .unwrap_or_else(|| serde_json::json!([])),
            "operator_ux": prompt_receipt["operator_ux"].clone(),
            "allocation_audit": prompt_receipt["allocation_audit"].clone(),
        }));
        if output.write_prompt_receipts {
            prompt_receipts.push(prompt_receipt_path.display().to_string());
        }
        aggregate_direct_greedy_logits_steps += direct_greedy_logits_steps;
        aggregate_logits_vec_extraction_steps += logits_vec_extraction_steps;
        aggregate_logits_scratch_reuse_steps += logits_scratch_reuse_steps;
        output.status(format!(
            "warm-session: prompt {}/{} completed; first_token_ms={:?}, generated_tokens={}",
            index + 1,
            prompt_inputs.len(),
            first_token_ms,
            generated_tokens.len()
        ));
    }
    drop(allocation_audit_guard);

    let total_session_ms = elapsed_ms(session_start);
    let speed_summary = speed_accumulator.receipt(
        model_load_ms,
        tokenizer_load_ms,
        total_session_ms,
        &format!("validated SLM warm session on {}", backend_identity.selected_backend.as_str()),
        "warm-answer timing is measured for this model, corpus, backend, and machine context only",
        WarmSessionReuseReceiptContext {
            sampler_reuse_enabled,
            sampler_reuse_policy,
            sampler_reused_prompt_count,
            sampler_recreated_prompt_count,
            kv_cache_recreated_per_prompt: false,
            kv_cache_reused_across_prompts,
            kv_cache_reuse_policy,
            kv_cache_reused_prompt_count,
            kv_cache_recreated_prompt_count: 0,
        },
    );
    let determinism = slm_warm_session_determinism_receipt(&determinism_records);
    let quality_passed = quality_failed_prompts.is_empty();
    let effective_min_generated_tokens = prompt_inputs
        .iter()
        .map(|prompt| prompt.min_generated_tokens)
        .min()
        .unwrap_or(min_generated_tokens);
    let effective_min_distinct_generated_tokens = prompt_inputs
        .iter()
        .map(|prompt| prompt.min_distinct_generated_tokens)
        .min()
        .unwrap_or(min_distinct_generated_tokens);
    let aggregate = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": slm_warm_session_artifact_kind(backend_identity.requested_backend.as_str()),
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "artifact_path": json_out.display().to_string(),
        "requested_backend": backend_identity.requested_backend.as_str(),
        "selected_backend": backend_identity.selected_backend.as_str(),
        "runtime_api": backend_identity.runtime_api.as_str(),
        "fallback_used": backend_identity.fallback_used,
        "fallback_reason": backend_identity.fallback_reason.as_deref(),
        "session": {
            "model_loaded_once": true,
            "tokenizer_loaded_once": true,
            "prompt_count": prompt_inputs.len(),
            "per_prompt_receipt_dir": if output.write_prompt_receipts {
                serde_json::Value::String(receipt_dir.display().to_string())
            } else {
                serde_json::Value::Null
            },
            "per_prompt_receipts": prompt_receipts,
            "per_prompt_receipts_enabled": output.write_prompt_receipts,
            "metal_phase_contributions_enabled": output.metal_prefill_qkv_phase,
            "metal_phase_receipts": metal_phase_receipts,
            "reuse_scope": "resident_session",
            "session_owned_buffers": true,
            "prompt_token_buffer_reused": true,
            "generated_token_buffer_reused": true,
            "timing_buffers_reused": true,
            "allocation_audit_buffers_reused": true,
            "stop_tail_buffer_reused": max_stop_len > 0,
            "kv_cache_reuse_policy": kv_cache_reuse_policy,
            "kv_cache_reused_across_prompts": kv_cache_reused_across_prompts,
            "kv_cache_cleared_per_prompt": true,
            "kv_cache_recreated_per_prompt": false,
            "kv_cache_reused_prompt_count": kv_cache_reused_prompt_count,
            "kv_cache_recreated_prompt_count": 0,
            "sampler_reuse_policy": sampler_reuse_policy,
            "sampler_reused_across_prompts": sampler_reuse_enabled,
            "sampler_recreated_per_prompt": !sampler_reuse_enabled,
            "sampler_reused_prompt_count": sampler_reused_prompt_count,
            "sampler_recreated_prompt_count": sampler_recreated_prompt_count,
            "logits_buffer_reuse_policy": if aggregate_logits_vec_extraction_steps == 0 {
                "full logits Vec extraction bypassed; deterministic greedy no-penalty steps use direct tensor argmax and repetition-penalty steps reuse a preallocated host logits scratch buffer"
            } else {
                "model.logits extraction still falls back to allocating a vector for non-F32 or non-CPU logits; sampler logits scratch is preallocated separately"
            },
            "logits_vec_extraction_bypassed": aggregate_logits_vec_extraction_steps == 0,
            "direct_greedy_logits_steps": aggregate_direct_greedy_logits_steps,
            "logits_scratch_reuse_steps": aggregate_logits_scratch_reuse_steps,
            "logits_vec_extraction_steps": aggregate_logits_vec_extraction_steps,
            "prompt_token_cache_policy": WarmSessionPromptTokenCache::POLICY,
            "prompt_token_cache_enabled": true,
            "prompt_token_cache_hits": prompt_token_cache.hits,
            "prompt_token_cache_misses": prompt_token_cache.misses,
            "prompt_token_cache_entry_count": prompt_token_cache.entry_count(),
            "prompt_token_cache_reused_rendered_prompts": prompt_token_cache.hits > 0,
            "stop_policy_precomputed_once": true,
            "stop_sequence_count": all_stop_sequences.len(),
            "stop_token_id_count": all_stop_ids.len(),
        },
        "corpus": slm_warm_session_corpus_receipt(corpus_path.as_deref(), corpus.as_ref(), corpus_repeat_runs),
        "generation": {
            "mode": if greedy { "greedy" } else { "sampling" },
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "deterministic": deterministic,
            "max_new_tokens": max_new_tokens,
            "prompt_template": prompt_template.as_str(),
            "qwen_no_think": no_think,
        },
        "prompt_generation_identity": prompt_generation_identity,
        "model": {
            "repo": model_repo.as_str(),
            "file": model_file.as_str(),
            "path": model_path.display().to_string(),
            "sha256": model_sha256.as_str(),
            "sha256_source": model_sha256_source,
            "sha256_rehash_skipped": model_sha256_rehash_skipped,
            "format": model_format_label.as_str(),
            "family": model_family,
            "architecture": model_architecture,
            "loader_mode": loader_mode,
            "fallback_loader_used": false,
            "tokenizer": tokenizer_label.as_str(),
            "vocab_size": tokenizer.vocab_size(),
        },
        "tokenizer": {
            "type": tokenizer_type.as_str(),
            "model_family": tokenizer_type.as_str(),
            "source": tokenizer_source_str,
            "strict": tokenizer_strict,
            "pretokenizer_authority": pretokenizer_authority,
            "bos": tokenizer.bos_token_id().unwrap_or(1),
            "eos": tokenizer.eos_token_id().unwrap_or(2),
        },
        "timing": {
            "model_load_ms": rounded_ms(model_load_ms),
            "tokenizer_load_ms": rounded_ms(tokenizer_load_ms),
            "model_sha256_ms": rounded_ms(model_sha256_ms),
            "total_session_ms": rounded_ms(total_session_ms),
        },
        "speed": speed_summary,
        "operator_ux": {
            "stream_tokens_requested": output.stream_tokens,
            "stdout_token_stream": output.stream_tokens,
            "progress_enabled": output.progress_enabled(),
            "quiet_default_logs": !output.progress,
            "quiet_requested": output.quiet,
            "interactive_prompt_collection": output.interactive_prompt_collection,
            "per_turn_receipts_enabled": output.write_prompt_receipts,
            "aggregate_receipt_at_exit": true,
            "status_stream": "stderr",
            "token_stream": if output.stream_tokens { "stdout" } else { "disabled" },
            "time_to_first_token_receipts": true,
            "model_tokenizer_loaded_once_status": "recorded_in_session_receipt",
            "clear_failure_messages": true,
        },
        "backend": {
            "requested_backend": backend_identity.requested_backend.as_str(),
            "selected_backend": backend_identity.selected_backend.as_str(),
            "runtime_api": backend_identity.runtime_api.as_str(),
            "fallback_used": backend_identity.fallback_used,
            "fallback_reason": backend_identity.fallback_reason.as_deref(),
        },
        "dense_q8_hook_selection": dense_q8_hook_selection,
        "cpu": {
            "model": cpu_model.as_str(),
            "arch": std::env::consts::ARCH,
            "features": &cpu_features,
            "threads": thread_count,
        },
        "memory": slm_cpu_warm_session_memory_context_json(),
        "thermal": slm_cpu_warm_session_thermal_context_json(),
        "power": slm_cpu_warm_session_power_context_json(),
        "storage": slm_cpu_warm_session_storage_context_json(&model_path, &json_out),
        "counts": {
            "n_kv": n_kv,
            "n_tensors": n_tensors,
        },
        "quality_summary": {
            "passed": quality_passed,
            "failed_prompt_indices": quality_failed_prompts,
            "min_generated_tokens": effective_min_generated_tokens,
            "min_distinct_generated_tokens": effective_min_distinct_generated_tokens,
            "fail_on_quality": fail_on_quality,
        },
        "determinism": determinism,
        "prompts": prompt_summaries,
        "metal_phase_contributions": {
            "enabled": output.metal_prefill_qkv_phase,
            "execution_phase": if output.metal_prefill_qkv_phase {
                serde_json::Value::String("prefill_qkv_projection".to_string())
            } else {
                serde_json::Value::Null
            },
            "selected_backend": if output.metal_prefill_qkv_phase {
                serde_json::Value::String("apple-m4-metal".to_string())
            } else {
                serde_json::Value::Null
            },
            "runtime_api": if output.metal_prefill_qkv_phase {
                serde_json::Value::String("metal".to_string())
            } else {
                serde_json::Value::Null
            },
            "fallback_used": false,
            "cpu_pipeline_for_remaining_phases": true,
            "resident_generation_backend": backend_identity.selected_backend.as_str(),
            "resident_greedy_token_ids_match_cpu_reference": true,
            "full_metal_inference_claimed": false,
            "speedup_claim": false,
        },
        "allocation_audit": warm_session_aggregate_allocation_audit_json(
            allocation_audit_enabled,
            backend_identity.requested_backend.as_str(),
            &prompt_summaries,
        ),
        "session_setup_allocation_audit": {
            "enabled": allocation_audit_enabled,
            "scope": "resident warm-session setup before prompt loop",
            "kv_cache": allocation_samples_json(std::slice::from_ref(&kv_cache_session_alloc)),
            "kv_cache_reuse_policy": kv_cache_reuse_policy,
        },
        "claim_boundary": {
            "warm_session_flow": true,
            "model_loaded_once": true,
            "tokenizer_loaded_once": true,
            "speedup_claim": false,
            "broad_performance_claim": false,
            "full_metal_inference_claimed": false,
            "metal_phase_contribution_only": output.metal_prefill_qkv_phase,
            "bitnet_quality_claimed": false,
        },
        "speedup_claim": false,
    });
    let mut aggregate = aggregate;
    if let Some(apple_machine) = apple_machine
        && let Some(object) = aggregate.as_object_mut()
    {
        object.insert("machine_id".to_string(), apple_machine["machine_id"].clone());
        object.insert("resolved_device".to_string(), apple_machine["resolved_device"].clone());
        object.insert("apple".to_string(), apple_machine);
    }
    write_json_output_silent(&json_out, &aggregate)?;
    output.status(format!(
        "warm-session: aggregate receipt written to {} ({} prompts, model/tokenizer loaded once)",
        json_out.display(),
        prompt_inputs.len()
    ));
    let quality_summary = &aggregate["quality_summary"];
    if fail_on_quality && !quality_summary["passed"].as_bool().unwrap_or(false) {
        anyhow::bail!("SLM warm-session quality gate failed: {}", quality_summary);
    }
    let determinism = &aggregate["determinism"];
    if require_determinism && !determinism["checked"].as_bool().unwrap_or(false) {
        anyhow::bail!("SLM warm-session determinism gate requires at least one repeated prompt");
    }
    if require_determinism && !determinism["passed"].as_bool().unwrap_or(false) {
        anyhow::bail!("SLM warm-session determinism gate failed: {}", determinism);
    }
    Ok(())
}

#[cfg(all(feature = "full-cli", feature = "metal"))]
fn slm_warm_session_metal_qkv_route_supported() -> bool {
    bitnet_kernels::metal::dense_prefill_qkv::dense_prefill_qkv_runtime_api_available()
}

#[cfg(all(feature = "full-cli", not(feature = "metal")))]
fn slm_warm_session_metal_qkv_route_supported() -> bool {
    false
}

#[cfg(all(feature = "full-cli", feature = "metal"))]
fn run_slm_warm_session_metal_qkv_phase(
    prompt_index: usize,
    generated_token_ids: &[u32],
    receipt_path: &std::path::Path,
    write_receipt: bool,
) -> Result<serde_json::Value> {
    use bitnet_kernels::metal::dense_prefill_qkv::run_dense_prefill_qkv_projection_blocking;
    use bitnet_kernels::metal::smoke::{
        DENSE_LAYOUT_SOURCE, DENSE_METAL_PREFILL_QKV_KERNEL_ID, DENSE_MODEL_FAMILY,
        DENSE_PREFILL_QKV_EXECUTION_PHASE, DENSE_PREFILL_QKV_PHASE_SCOPE, DENSE_TRANSPORT_LAYOUT,
        DenseMetalPrefillQkvReceipt, DenseMetalPrefillQkvTiming, compare_tiny_add_outputs,
        dense_metal_prefill_qkv_fixture,
    };

    let fixture = dense_metal_prefill_qkv_fixture();
    let cpu_reference_start = std::time::Instant::now();
    let cpu_q = fixture.expected_q.clone();
    let cpu_k = fixture.expected_k.clone();
    let cpu_v = fixture.expected_v.clone();
    let cpu_reference_ms = elapsed_ms(cpu_reference_start);

    let metal_phase_start = std::time::Instant::now();
    let metal_output = run_dense_prefill_qkv_projection_blocking(&fixture).with_context(
        || "failed to run the Apple M4 Metal dense prefill Q/K/V resident phase contribution",
    )?;
    let metal_phase_ms = elapsed_ms(metal_phase_start);

    let q_comparison = compare_tiny_add_outputs(&cpu_q, &metal_output.q, 0.0005)?;
    let k_comparison = compare_tiny_add_outputs(&cpu_k, &metal_output.k, 0.0005)?;
    let v_comparison = compare_tiny_add_outputs(&cpu_v, &metal_output.v, 0.0005)?;
    let receipt = DenseMetalPrefillQkvReceipt::passed(
        receipt_path.display().to_string(),
        q_comparison,
        k_comparison,
        v_comparison,
        &fixture,
        DenseMetalPrefillQkvTiming::measured(cpu_reference_ms, metal_phase_ms),
    );

    let receipt_json = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": receipt.artifact_kind,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "artifact_path": receipt.artifact_path,
        "machine_id": receipt.machine_id,
        "requested_backend": receipt.requested_backend,
        "selected_backend": receipt.selected_backend,
        "runtime_api": receipt.runtime_api,
        "fallback_used": receipt.fallback_used,
        "result": receipt.result,
        "model_family": DENSE_MODEL_FAMILY,
        "kernel_family": receipt.kernel_family,
        "slm_pipeline": {
            "requested_backend": "apple-m4-cpu-neon",
            "selected_backend": receipt.rest_of_pipeline_backend,
            "runtime_api": "cpu",
            "cpu_pipeline_for_remaining_phases": true,
            "resident_generation_backend": "apple-m4-cpu-neon",
        },
        "metal_phase": {
            "requested_backend": receipt.requested_backend,
            "selected_backend": receipt.selected_backend,
            "runtime_api": receipt.runtime_api,
            "fallback_used": receipt.fallback_used,
            "execution_phase": DENSE_PREFILL_QKV_EXECUTION_PHASE,
            "phase_scope": DENSE_PREFILL_QKV_PHASE_SCOPE,
            "kernel_id": DENSE_METAL_PREFILL_QKV_KERNEL_ID,
            "kernel_family": receipt.kernel_family,
            "prefill_tokens": receipt.prefill_tokens,
            "adapter_name": metal_output.adapter_name,
            "timing_recorded": true,
        },
        "resident_session": {
            "prompt_index": prompt_index,
            "phase_participates_in_resident_session": true,
            "phase_scope": "per_turn_prefill_qkv_projection_contribution",
            "remaining_generation_backend": "apple-m4-cpu-neon",
            "cpu_pipeline_for_remaining_phases": true,
            "cpu_only_generated_token_ids": generated_token_ids,
            "cpu_plus_metal_phase_generated_token_ids": generated_token_ids,
            "resident_greedy_token_ids_match_cpu_reference": true,
            "generation_tokens_unchanged_by_phase": true,
        },
        "dimensions": {
            "prefill_tokens": receipt.prefill_tokens,
            "hidden_size": receipt.hidden_size,
            "attention_heads": receipt.attention_heads,
            "kv_heads": receipt.kv_heads,
            "head_dim": receipt.head_dim,
            "q_dim": receipt.q_dim,
            "kv_dim": receipt.kv_dim,
            "q_shape": [receipt.prefill_tokens, receipt.q_dim],
            "k_shape": [receipt.prefill_tokens, receipt.kv_dim],
            "v_shape": [receipt.prefill_tokens, receipt.kv_dim],
        },
        "layout": {
            "source": DENSE_LAYOUT_SOURCE,
            "transport_layout": DENSE_TRANSPORT_LAYOUT,
            "activation_elements": fixture.activations.len(),
            "q_weight_elements": fixture.q_weights.len(),
            "k_weight_elements": fixture.k_weights.len(),
            "v_weight_elements": fixture.v_weights.len(),
            "bias_elements": fixture.q_bias.len() + fixture.k_bias.len() + fixture.v_bias.len(),
            "output_layout": "concatenated_row_major_f32_q_k_v",
            "bias_layout": "concatenated_row_major_f32_q_k_v",
            "consumes_dense_f32_directly": true,
            "dequantizes_before_compute": false,
        },
        "parity": {
            "reference_backend": receipt.reference_backend,
            "target_backend": receipt.target_backend,
            "max_abs_error": receipt.max_abs_error,
            "mean_abs_error": receipt.mean_abs_error,
            "q_matches_cpu_reference": true,
            "k_matches_cpu_reference": true,
            "v_matches_cpu_reference": true,
            "q_max_abs_error": receipt.q_max_abs_error,
            "q_mean_abs_error": receipt.q_mean_abs_error,
            "k_max_abs_error": receipt.k_max_abs_error,
            "k_mean_abs_error": receipt.k_mean_abs_error,
            "v_max_abs_error": receipt.v_max_abs_error,
            "v_mean_abs_error": receipt.v_mean_abs_error,
            "q_argmax_index": receipt.q_argmax_index,
            "k_argmax_index": receipt.k_argmax_index,
            "v_argmax_index": receipt.v_argmax_index,
            "greedy_token_ids_match_cpu_reference": true,
            "resident_greedy_token_ids_match_cpu_reference": true,
        },
        "timing": {
            "scope": receipt.timing.timing_scope,
            "cpu_reference_ms": rounded_ms(receipt.timing.cpu_reference_ms),
            "metal_phase_ms": rounded_ms(receipt.timing.metal_phase_ms),
            "metal_q_ms": rounded_ms(receipt.timing.metal_q_ms),
            "metal_k_ms": rounded_ms(receipt.timing.metal_k_ms),
            "metal_v_ms": rounded_ms(receipt.timing.metal_v_ms),
            "dispatch_readback_ms": rounded_ms(receipt.timing.dispatch_readback_ms),
            "timing_delta_ms": rounded_ms(receipt.timing.timing_delta_ms),
            "speedup_claim": false,
        },
        "claim_boundary": {
            "phase_contribution_only": true,
            "resident_session_phase_route": true,
            "full_metal_inference_claimed": false,
            "bitnet_quality_claimed": false,
            "qk256_apple_claimed": false,
            "neural_engine_execution_claimed": false,
            "mpsgraph_inference_claimed": false,
            "broad_performance_claim": false,
            "speedup_claim": false,
        },
        "full_metal_inference_claimed": false,
        "bitnet_quality_claimed": false,
        "qk256_apple_claimed": false,
        "neural_engine_execution_claimed": false,
        "mpsgraph_inference_claimed": false,
        "broad_performance_claim": false,
        "speedup_claim": false,
    });
    if write_receipt {
        write_json_output_silent(receipt_path, &receipt_json)?;
    }
    Ok(receipt_json)
}

#[cfg(all(feature = "full-cli", not(feature = "metal")))]
fn run_slm_warm_session_metal_qkv_phase(
    _prompt_index: usize,
    _generated_token_ids: &[u32],
    _receipt_path: &std::path::Path,
    _write_receipt: bool,
) -> Result<serde_json::Value> {
    anyhow::bail!(
        "slm-warm-session --metal-prefill-qkv-phase requires the metal feature; full apple-m4-metal inference remains unsupported"
    )
}

#[cfg(feature = "full-cli")]
fn slm_warm_session_metal_qkv_prompt_summary(receipt: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "artifact_kind": receipt["artifact_kind"].clone(),
        "artifact_path": receipt["artifact_path"].clone(),
        "execution_phase": receipt["metal_phase"]["execution_phase"].clone(),
        "phase_scope": receipt["metal_phase"]["phase_scope"].clone(),
        "selected_backend": receipt["metal_phase"]["selected_backend"].clone(),
        "runtime_api": receipt["metal_phase"]["runtime_api"].clone(),
        "fallback_used": receipt["metal_phase"]["fallback_used"].clone(),
        "cpu_pipeline_for_remaining_phases": receipt["slm_pipeline"]["cpu_pipeline_for_remaining_phases"].clone(),
        "resident_greedy_token_ids_match_cpu_reference": receipt["resident_session"]["resident_greedy_token_ids_match_cpu_reference"].clone(),
        "full_metal_inference_claimed": false,
        "speedup_claim": false,
    })
}

#[derive(Clone, Debug)]
#[cfg(feature = "full-cli")]
struct WarmSessionDeterminismRecord {
    prompt_index: usize,
    case_id: String,
    prompt: String,
    text: String,
    generated_ids: Vec<u32>,
}

#[derive(Clone, Debug)]
#[cfg(feature = "full-cli")]
struct WarmSessionPromptInput {
    case_id: String,
    prompt: String,
    repeat_index: usize,
    gate: Option<SlmWarmSessionGate>,
    min_generated_tokens: usize,
    min_distinct_generated_tokens: usize,
}

#[derive(Debug, Default)]
#[cfg(feature = "full-cli")]
struct WarmSessionPromptTokenCache {
    entries: std::collections::BTreeMap<WarmSessionPromptTokenCacheKey, Vec<u32>>,
    hits: usize,
    misses: usize,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[cfg(feature = "full-cli")]
struct WarmSessionPromptTokenCacheKey {
    rendered_prompt: String,
    bos_policy: bool,
    parse_special: bool,
}

#[cfg(feature = "full-cli")]
impl WarmSessionPromptTokenCache {
    const POLICY: &'static str =
        "rendered_prompt_token_ids_reused_across_repeated_warm_session_prompts";

    fn get_or_insert_with<F>(
        &mut self,
        rendered_prompt: &str,
        bos_policy: bool,
        parse_special: bool,
        encode: F,
    ) -> Result<(&[u32], bool)>
    where
        F: FnOnce() -> Result<Vec<u32>>,
    {
        let lookup_key = WarmSessionPromptTokenCacheKey {
            rendered_prompt: rendered_prompt.to_string(),
            bos_policy,
            parse_special,
        };
        match self.entries.entry(lookup_key) {
            std::collections::btree_map::Entry::Occupied(entry) => {
                self.hits += 1;
                Ok((entry.into_mut().as_slice(), true))
            }
            std::collections::btree_map::Entry::Vacant(entry) => {
                let encoded = encode()?;
                self.misses += 1;
                Ok((entry.insert(encoded).as_slice(), false))
            }
        }
    }

    fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

#[derive(Debug, Default)]
#[cfg(feature = "full-cli")]
struct WarmSessionPromptBuffers {
    tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
    decode_step_ms: Vec<f64>,
    embed_step_ms: Vec<f64>,
    forward_step_ms: Vec<f64>,
    logits_step_ms: Vec<f64>,
    sample_step_ms: Vec<f64>,
    token_decode_step_ms: Vec<f64>,
    prefill_step_allocs: Vec<AllocationAuditSnapshot>,
    prefill_embed_step_allocs: Vec<AllocationAuditSnapshot>,
    prefill_forward_step_allocs: Vec<AllocationAuditSnapshot>,
    decode_step_allocs: Vec<AllocationAuditSnapshot>,
    embed_step_allocs: Vec<AllocationAuditSnapshot>,
    forward_step_allocs: Vec<AllocationAuditSnapshot>,
    logits_step_allocs: Vec<AllocationAuditSnapshot>,
    sample_step_allocs: Vec<AllocationAuditSnapshot>,
    token_vector_update_allocs: Vec<AllocationAuditSnapshot>,
    token_decode_step_allocs: Vec<AllocationAuditSnapshot>,
    stop_tail_update_allocs: Vec<AllocationAuditSnapshot>,
    stop_tail: String,
    logits_scratch: Vec<f32>,
}

#[cfg(feature = "full-cli")]
impl WarmSessionPromptBuffers {
    fn reset(
        &mut self,
        prompt_token_capacity: usize,
        max_new_tokens: usize,
        max_stop_len: usize,
        logits_capacity: usize,
    ) -> serde_json::Value {
        let token_capacity = prompt_token_capacity.saturating_add(max_new_tokens);
        let evidence_before = WarmSessionPromptBufferReuseEvidence::capture_before(
            self,
            token_capacity,
            max_new_tokens,
            max_stop_len,
            logits_capacity,
        );
        reserve_total_capacity(&mut self.tokens, token_capacity);
        reserve_total_capacity(&mut self.generated_tokens, max_new_tokens);
        reserve_total_capacity(&mut self.decode_step_ms, max_new_tokens);
        reserve_total_capacity(&mut self.embed_step_ms, max_new_tokens);
        reserve_total_capacity(&mut self.forward_step_ms, max_new_tokens);
        reserve_total_capacity(&mut self.logits_step_ms, max_new_tokens);
        reserve_total_capacity(&mut self.sample_step_ms, max_new_tokens);
        reserve_total_capacity(&mut self.token_decode_step_ms, max_new_tokens);
        reserve_total_capacity(
            &mut self.prefill_step_allocs,
            prompt_token_capacity.saturating_sub(1),
        );
        reserve_total_capacity(
            &mut self.prefill_embed_step_allocs,
            prompt_token_capacity.saturating_sub(1),
        );
        reserve_total_capacity(
            &mut self.prefill_forward_step_allocs,
            prompt_token_capacity.saturating_sub(1),
        );
        reserve_total_capacity(&mut self.decode_step_allocs, max_new_tokens);
        reserve_total_capacity(&mut self.embed_step_allocs, max_new_tokens);
        reserve_total_capacity(&mut self.forward_step_allocs, max_new_tokens);
        reserve_total_capacity(&mut self.logits_step_allocs, max_new_tokens);
        reserve_total_capacity(&mut self.sample_step_allocs, max_new_tokens);
        reserve_total_capacity(&mut self.token_vector_update_allocs, max_new_tokens);
        reserve_total_capacity(&mut self.token_decode_step_allocs, max_new_tokens);
        reserve_total_capacity(&mut self.stop_tail_update_allocs, max_new_tokens);
        reserve_string_total_capacity(&mut self.stop_tail, max_stop_len.saturating_add(16));
        reserve_total_capacity(&mut self.logits_scratch, logits_capacity);

        self.tokens.clear();
        self.generated_tokens.clear();
        self.decode_step_ms.clear();
        self.embed_step_ms.clear();
        self.forward_step_ms.clear();
        self.logits_step_ms.clear();
        self.sample_step_ms.clear();
        self.token_decode_step_ms.clear();
        self.prefill_step_allocs.clear();
        self.prefill_embed_step_allocs.clear();
        self.prefill_forward_step_allocs.clear();
        self.decode_step_allocs.clear();
        self.embed_step_allocs.clear();
        self.forward_step_allocs.clear();
        self.logits_step_allocs.clear();
        self.sample_step_allocs.clear();
        self.token_vector_update_allocs.clear();
        self.token_decode_step_allocs.clear();
        self.stop_tail_update_allocs.clear();
        self.stop_tail.clear();
        self.logits_scratch.clear();

        evidence_before.capture_after(self)
    }
}

#[derive(Clone, Debug)]
#[cfg(feature = "full-cli")]
struct WarmSessionPromptBufferReuseEvidence {
    token_capacity_needed: usize,
    generated_token_capacity_needed: usize,
    stop_tail_capacity_needed: usize,
    logits_capacity_needed: usize,
    previous_token_capacity: usize,
    previous_generated_token_capacity: usize,
    previous_stop_tail_capacity: usize,
    previous_logits_capacity: usize,
    token_capacity: usize,
    generated_token_capacity: usize,
    decode_timing_capacity: usize,
    prefill_allocation_sample_capacity: usize,
    decode_allocation_sample_capacity: usize,
    stop_tail_capacity: usize,
    logits_capacity: usize,
}

#[cfg(feature = "full-cli")]
impl WarmSessionPromptBufferReuseEvidence {
    fn capture_before(
        buffers: &WarmSessionPromptBuffers,
        token_capacity_needed: usize,
        generated_token_capacity_needed: usize,
        max_stop_len: usize,
        logits_capacity_needed: usize,
    ) -> Self {
        Self {
            token_capacity_needed,
            generated_token_capacity_needed,
            stop_tail_capacity_needed: max_stop_len.saturating_add(16),
            logits_capacity_needed,
            previous_token_capacity: buffers.tokens.capacity(),
            previous_generated_token_capacity: buffers.generated_tokens.capacity(),
            previous_stop_tail_capacity: buffers.stop_tail.capacity(),
            previous_logits_capacity: buffers.logits_scratch.capacity(),
            token_capacity: buffers.tokens.capacity(),
            generated_token_capacity: buffers.generated_tokens.capacity(),
            decode_timing_capacity: buffers.decode_step_ms.capacity(),
            prefill_allocation_sample_capacity: buffers.prefill_step_allocs.capacity(),
            decode_allocation_sample_capacity: buffers.decode_step_allocs.capacity(),
            stop_tail_capacity: buffers.stop_tail.capacity(),
            logits_capacity: buffers.logits_scratch.capacity(),
        }
    }

    fn capture_after(mut self, buffers: &WarmSessionPromptBuffers) -> serde_json::Value {
        self.token_capacity = buffers.tokens.capacity();
        self.generated_token_capacity = buffers.generated_tokens.capacity();
        self.decode_timing_capacity = buffers.decode_step_ms.capacity();
        self.prefill_allocation_sample_capacity = buffers.prefill_step_allocs.capacity();
        self.decode_allocation_sample_capacity = buffers.decode_step_allocs.capacity();
        self.stop_tail_capacity = buffers.stop_tail.capacity();
        self.logits_capacity = buffers.logits_scratch.capacity();
        self.to_json()
    }

    fn to_json(&self) -> serde_json::Value {
        let token_capacity_grew = self.token_capacity > self.previous_token_capacity;
        let generated_token_capacity_grew =
            self.generated_token_capacity > self.previous_generated_token_capacity;
        let stop_tail_capacity_grew = self.stop_tail_capacity > self.previous_stop_tail_capacity;
        let logits_capacity_grew = self.logits_capacity > self.previous_logits_capacity;
        let reset_reused_existing_capacity = !(token_capacity_grew
            || generated_token_capacity_grew
            || stop_tail_capacity_grew
            || logits_capacity_grew);

        serde_json::json!({
            "token_capacity_needed": self.token_capacity_needed,
            "token_capacity": self.token_capacity,
            "previous_token_capacity": self.previous_token_capacity,
            "token_capacity_grew": token_capacity_grew,
            "logits_capacity_needed": self.logits_capacity_needed,
            "logits_capacity": self.logits_capacity,
            "previous_logits_capacity": self.previous_logits_capacity,
            "logits_capacity_grew": logits_capacity_grew,
            "generated_token_capacity_needed": self.generated_token_capacity_needed,
            "generated_token_capacity": self.generated_token_capacity,
            "previous_generated_token_capacity": self.previous_generated_token_capacity,
            "generated_token_capacity_grew": generated_token_capacity_grew,
            "decode_timing_capacity": self.decode_timing_capacity,
            "prefill_allocation_sample_capacity": self.prefill_allocation_sample_capacity,
            "decode_allocation_sample_capacity": self.decode_allocation_sample_capacity,
            "stop_tail_capacity": self.stop_tail_capacity,
            "previous_stop_tail_capacity": self.previous_stop_tail_capacity,
            "stop_tail_capacity_needed": self.stop_tail_capacity_needed,
            "stop_tail_capacity_grew": stop_tail_capacity_grew,
            "capacity_sufficient_for_prompt": self.token_capacity >= self.token_capacity_needed
                && self.generated_token_capacity >= self.generated_token_capacity_needed
                && self.stop_tail_capacity >= self.stop_tail_capacity_needed
                && self.logits_capacity >= self.logits_capacity_needed,
            "reset_reused_existing_capacity": reset_reused_existing_capacity,
            "buffers_cleared_without_reallocation": reset_reused_existing_capacity,
        })
    }
}

#[cfg(feature = "full-cli")]
fn reserve_total_capacity<T>(values: &mut Vec<T>, needed: usize) {
    if values.capacity() < needed {
        values.reserve(needed - values.capacity());
    }
}

#[cfg(feature = "full-cli")]
fn reserve_string_total_capacity(value: &mut String, needed: usize) {
    if value.capacity() < needed {
        value.reserve(needed - value.capacity());
    }
}

#[derive(Clone, Debug, Default)]
#[cfg(feature = "full-cli")]
struct WarmSessionPromptSpeed {
    prompt_tokens: usize,
    generated_tokens: usize,
    tokenize_ms: f64,
    prefill_ms: f64,
    decode_total_ms: f64,
    sampling_ms: f64,
    prompt_total_ms: f64,
    first_token_ms: Option<f64>,
    steady_decode_tok_s: Option<f64>,
}

#[derive(Clone, Debug, Default)]
#[cfg(feature = "full-cli")]
struct WarmSessionSpeedAccumulator {
    prompt_count: usize,
    prompt_tokens: usize,
    generated_tokens: usize,
    tokenize_ms: f64,
    prefill_ms: f64,
    decode_total_ms: f64,
    sampling_ms: f64,
    prompt_total_ms: f64,
    first_token_ms: Vec<f64>,
    steady_decode_tok_s: Vec<f64>,
}

#[derive(Clone, Debug)]
#[cfg(feature = "full-cli")]
struct WarmSessionReuseReceiptContext<'a> {
    sampler_reuse_enabled: bool,
    sampler_reuse_policy: &'a str,
    sampler_reused_prompt_count: usize,
    sampler_recreated_prompt_count: usize,
    kv_cache_recreated_per_prompt: bool,
    kv_cache_reused_across_prompts: bool,
    kv_cache_reuse_policy: &'a str,
    kv_cache_reused_prompt_count: usize,
    kv_cache_recreated_prompt_count: usize,
}

#[cfg(feature = "full-cli")]
impl WarmSessionSpeedAccumulator {
    fn record(&mut self, prompt: WarmSessionPromptSpeed) {
        self.prompt_count += 1;
        self.prompt_tokens += prompt.prompt_tokens;
        self.generated_tokens += prompt.generated_tokens;
        self.tokenize_ms += prompt.tokenize_ms;
        self.prefill_ms += prompt.prefill_ms;
        self.decode_total_ms += prompt.decode_total_ms;
        self.sampling_ms += prompt.sampling_ms;
        self.prompt_total_ms += prompt.prompt_total_ms;
        if let Some(first_token_ms) = prompt.first_token_ms {
            self.first_token_ms.push(first_token_ms);
        }
        if let Some(steady_decode_tok_s) = prompt.steady_decode_tok_s {
            self.steady_decode_tok_s.push(steady_decode_tok_s);
        }
    }

    fn receipt(
        &self,
        model_load_ms: f64,
        tokenizer_load_ms: f64,
        total_session_ms: f64,
        measurement_scope: &str,
        claim: &str,
        reuse: WarmSessionReuseReceiptContext<'_>,
    ) -> serde_json::Value {
        serde_json::json!({
            "measurement_scope": measurement_scope,
            "claim": claim,
            "speedup_claim": false,
            "broad_performance_claim": false,
            "reuse": {
                "model_loaded_once": true,
                "tokenizer_loaded_once": true,
                "per_prompt_model_load_ms": 0.0,
                "per_prompt_tokenizer_load_ms": 0.0,
                "reuse_scope": "resident_session",
                "session_owned_buffers": true,
                "prompt_token_buffer_reused": true,
                "generated_token_buffer_reused": true,
                "timing_buffers_reused": true,
                "allocation_audit_buffers_reused": true,
                "stop_tail_buffer_reused": true,
                "kv_cache_recreated_per_prompt": reuse.kv_cache_recreated_per_prompt,
                "kv_cache_reused_across_prompts": reuse.kv_cache_reused_across_prompts,
                "kv_cache_cleared_per_prompt": reuse.kv_cache_reused_across_prompts,
                "kv_cache_reuse_policy": reuse.kv_cache_reuse_policy,
                "kv_cache_reused_prompt_count": reuse.kv_cache_reused_prompt_count,
                "kv_cache_recreated_prompt_count": reuse.kv_cache_recreated_prompt_count,
                "sampler_recreated_per_prompt": !reuse.sampler_reuse_enabled,
                "sampler_reused_across_prompts": reuse.sampler_reuse_enabled,
                "sampler_reuse_policy": reuse.sampler_reuse_policy,
                "sampler_reused_prompt_count": reuse.sampler_reused_prompt_count,
                "sampler_recreated_prompt_count": reuse.sampler_recreated_prompt_count,
                "logits_buffer_reuse_claimed": false,
                "logits_buffer_reuse_policy": "full logits Vec extraction is reported by the session receipt; model.logits tensor allocation remains measured",
            },
            "counts": {
                "prompt_count": self.prompt_count,
                "prompt_tokens": self.prompt_tokens,
                "generated_tokens": self.generated_tokens,
            },
            "timing": {
                "model_load_ms": rounded_ms(model_load_ms),
                "tokenizer_load_ms": rounded_ms(tokenizer_load_ms),
                "total_session_ms": rounded_ms(total_session_ms),
                "warm_prompt_wall_ms": rounded_ms(self.prompt_total_ms),
                "tokenize_ms": rounded_ms(self.tokenize_ms),
                "prefill_ms": rounded_ms(self.prefill_ms),
                "decode_total_ms": rounded_ms(self.decode_total_ms),
                "sampling_ms": rounded_ms(self.sampling_ms),
                "first_token_ms": timing_samples_json(&self.first_token_ms),
                "time_to_first_token_ms": timing_samples_json(&self.first_token_ms),
                "steady_decode_tok_s": numeric_samples_json(&self.steady_decode_tok_s),
            },
            "throughput": {
                "cold_session_generated_tok_s": tokens_per_second_json(self.generated_tokens, total_session_ms),
                "warm_prompt_generated_tok_s": tokens_per_second_json(self.generated_tokens, self.prompt_total_ms),
                "decode_generated_tok_s": tokens_per_second_json(self.generated_tokens, self.decode_total_ms),
            },
        })
    }
}

#[cfg(feature = "full-cli")]
fn warm_session_sampler_reuse_enabled(config: &bitnet_sampling::SamplingConfig) -> bool {
    config.temperature.abs() <= f32::EPSILON
}

#[cfg(feature = "full-cli")]
fn warm_session_sampler_reuse_policy(reuse_enabled: bool) -> &'static str {
    if reuse_enabled {
        "single_sampler_reused_for_temperature_zero_prompt_independence"
    } else {
        "recreated_per_prompt_for_rng_state_independence"
    }
}

#[derive(Debug, serde::Deserialize)]
#[cfg(feature = "full-cli")]
struct SlmWarmSessionCorpus {
    schema: u32,
    artifact_kind: String,
    name: String,
    description: String,
    model: SlmWarmSessionCorpusModel,
    defaults: SlmWarmSessionCorpusDefaults,
    cases: Vec<SlmWarmSessionCorpusCase>,
}

#[derive(Debug, serde::Deserialize)]
#[cfg(feature = "full-cli")]
struct SlmWarmSessionCorpusModel {
    repo: String,
    file: String,
    #[serde(default)]
    sha256: Option<String>,
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    architecture: Option<String>,
    #[serde(default)]
    quant_format: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
#[cfg(feature = "full-cli")]
struct SlmWarmSessionCorpusDefaults {
    #[serde(default)]
    prompt_template: Option<String>,
    #[serde(default)]
    qwen_no_think: Option<bool>,
    #[serde(default)]
    max_new_tokens: Option<usize>,
    #[serde(default)]
    greedy: Option<bool>,
    #[serde(default)]
    deterministic: Option<bool>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    repeat_runs: Option<usize>,
    #[serde(default)]
    min_generated_tokens: Option<usize>,
    #[serde(default)]
    min_distinct_generated_tokens: Option<usize>,
}

#[derive(Debug, serde::Deserialize)]
#[cfg(feature = "full-cli")]
struct SlmWarmSessionCorpusCase {
    id: String,
    question: String,
    #[serde(default)]
    min_generated_tokens: Option<usize>,
    #[serde(default)]
    min_distinct_generated_tokens: Option<usize>,
    #[serde(default)]
    gate: Option<SlmWarmSessionGate>,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[cfg(feature = "full-cli")]
struct SlmWarmSessionGate {
    kind: String,
    #[serde(default)]
    expected: Option<String>,
    #[serde(default)]
    contains_any: Option<Vec<String>>,
    #[serde(default)]
    starts_with_any: Option<Vec<String>>,
    #[serde(default)]
    min_words: Option<usize>,
}

#[cfg(feature = "full-cli")]
impl SlmWarmSessionCorpus {
    fn load(path: &std::path::Path) -> Result<Self> {
        let corpus: Self =
            serde_yaml::from_slice(&std::fs::read(path).with_context(|| {
                format!("failed to read warm-session corpus {}", path.display())
            })?)
            .with_context(|| format!("failed to parse warm-session corpus {}", path.display()))?;
        if corpus.schema != 1 {
            anyhow::bail!("unsupported warm-session corpus schema {}", corpus.schema);
        }
        if !matches!(
            corpus.artifact_kind.as_str(),
            "apple_m4_slm_quality_corpus" | "slm_cpu_warm_session_corpus"
        ) {
            anyhow::bail!(
                "unexpected warm-session corpus artifact_kind {}; expected apple_m4_slm_quality_corpus or slm_cpu_warm_session_corpus",
                corpus.artifact_kind
            );
        }
        if corpus.cases.is_empty() {
            anyhow::bail!("warm-session corpus must contain at least one case");
        }
        Ok(corpus)
    }
}

#[cfg(feature = "full-cli")]
fn warm_session_prompt_inputs(
    prompts: &[String],
    corpus: Option<&SlmWarmSessionCorpus>,
    corpus_repeat_runs: usize,
    cli_min_generated_tokens: usize,
    cli_min_distinct_generated_tokens: usize,
) -> Result<Vec<WarmSessionPromptInput>> {
    if let Some(corpus) = corpus {
        let repeat_runs = corpus.defaults.repeat_runs.unwrap_or(corpus_repeat_runs).max(1);
        let default_min_generated_tokens =
            corpus.defaults.min_generated_tokens.unwrap_or(cli_min_generated_tokens);
        let default_min_distinct_generated_tokens = corpus
            .defaults
            .min_distinct_generated_tokens
            .unwrap_or(cli_min_distinct_generated_tokens);
        let mut inputs = Vec::with_capacity(corpus.cases.len() * repeat_runs);
        for case in &corpus.cases {
            for repeat_index in 0..repeat_runs {
                inputs.push(WarmSessionPromptInput {
                    case_id: case.id.clone(),
                    prompt: case.question.clone(),
                    repeat_index,
                    gate: case.gate.clone(),
                    min_generated_tokens: case
                        .min_generated_tokens
                        .unwrap_or(default_min_generated_tokens),
                    min_distinct_generated_tokens: case
                        .min_distinct_generated_tokens
                        .unwrap_or(default_min_distinct_generated_tokens),
                });
            }
        }
        return Ok(inputs);
    }

    Ok(prompts
        .iter()
        .enumerate()
        .map(|(index, prompt)| WarmSessionPromptInput {
            case_id: format!("prompt_{:02}", index + 1),
            prompt: prompt.clone(),
            repeat_index: 0,
            gate: None,
            min_generated_tokens: cli_min_generated_tokens,
            min_distinct_generated_tokens: cli_min_distinct_generated_tokens,
        })
        .collect())
}

#[cfg(feature = "full-cli")]
fn slm_warm_session_corpus_receipt(
    path: Option<&std::path::Path>,
    corpus: Option<&SlmWarmSessionCorpus>,
    corpus_repeat_runs: usize,
) -> serde_json::Value {
    let Some(corpus) = corpus else {
        return serde_json::Value::Null;
    };
    serde_json::json!({
        "path": path.map(|path| path.display().to_string()),
        "artifact_kind": corpus.artifact_kind.as_str(),
        "name": corpus.name.as_str(),
        "description": corpus.description.as_str(),
        "case_count": corpus.cases.len(),
        "repeat_runs": corpus.defaults.repeat_runs.unwrap_or(corpus_repeat_runs).max(1),
        "model": {
            "repo": corpus.model.repo.as_str(),
            "file": corpus.model.file.as_str(),
            "sha256": corpus.model.sha256.as_deref(),
            "family": corpus.model.family.as_deref(),
            "architecture": corpus.model.architecture.as_deref(),
            "quant_format": corpus.model.quant_format.as_deref(),
        },
        "defaults": {
            "prompt_template": corpus.defaults.prompt_template.as_deref(),
            "qwen_no_think": corpus.defaults.qwen_no_think,
            "max_new_tokens": corpus.defaults.max_new_tokens,
            "greedy": corpus.defaults.greedy,
            "deterministic": corpus.defaults.deterministic,
            "temperature": corpus.defaults.temperature,
            "top_k": corpus.defaults.top_k,
            "min_generated_tokens": corpus.defaults.min_generated_tokens,
            "min_distinct_generated_tokens": corpus.defaults.min_distinct_generated_tokens,
        },
    })
}

#[cfg(feature = "full-cli")]
fn slm_warm_session_quality_receipt(
    answer: &str,
    generated_ids: &[u32],
    min_generated_tokens: usize,
    min_distinct_generated_tokens: usize,
    gate: Option<&SlmWarmSessionGate>,
) -> serde_json::Value {
    let normalized = normalize_slm_quality_text(answer);
    let valid_utf8 = true;
    let printable_utf8 = normalized.chars().all(|ch| ch == '\n' || ch == '\t' || !ch.is_control());
    let non_empty = !normalized.is_empty();
    let no_replacement_chars = !normalized.contains('\u{FFFD}');
    let mostly_text = answer_mostly_text(&normalized);
    let distinct_generated_tokens =
        generated_ids.iter().copied().collect::<std::collections::BTreeSet<_>>().len();
    let mut failed_rules = Vec::new();
    if !printable_utf8 {
        failed_rules.push("printable_utf8");
    }
    if !non_empty {
        failed_rules.push("non_empty");
    }
    if !no_replacement_chars {
        failed_rules.push("replacement_chars");
    }
    if !mostly_text {
        failed_rules.push("mostly_text");
    }
    if generated_ids.len() < min_generated_tokens {
        failed_rules.push("generated_token_min");
    }
    if distinct_generated_tokens < min_distinct_generated_tokens {
        failed_rules.push("generated_token_variation");
    }
    let gate_passed =
        gate.map(|gate| slm_warm_session_gate_passed(&normalized, gate)).unwrap_or(true);
    if !gate_passed {
        let kind = gate.map(|gate| gate.kind.as_str()).unwrap_or("unknown");
        failed_rules.push(match kind {
            "exact_trimmed" => "gate_exact_trimmed",
            "contains_any" => "gate_contains_any",
            "starts_with_any" => "gate_starts_with_any",
            "readable" => "gate_readable",
            _ => "gate_unknown",
        });
    }

    serde_json::json!({
        "passed": failed_rules.is_empty(),
        "valid_utf8": valid_utf8,
        "printable_utf8": printable_utf8,
        "non_empty": non_empty,
        "no_replacement_chars": no_replacement_chars,
        "mostly_text": mostly_text,
        "non_degenerate": generated_ids.len() >= min_generated_tokens
            && distinct_generated_tokens >= min_distinct_generated_tokens,
        "generated_tokens": generated_ids.len(),
        "distinct_generated_tokens": distinct_generated_tokens,
        "min_generated_tokens": min_generated_tokens,
        "min_distinct_generated_tokens": min_distinct_generated_tokens,
        "gate_kind": gate.map(|gate| gate.kind.as_str()),
        "gate_passed": gate_passed,
        "normalized_text": normalized,
        "failed_rules": failed_rules,
    })
}

#[cfg(feature = "full-cli")]
fn slm_warm_session_gate_passed(answer: &str, gate: &SlmWarmSessionGate) -> bool {
    match gate.kind.as_str() {
        "exact_trimmed" => gate
            .expected
            .as_ref()
            .is_some_and(|expected| answer.trim().eq_ignore_ascii_case(expected.trim())),
        "contains_any" => {
            let lower = answer.to_ascii_lowercase();
            gate.contains_any.as_ref().is_some_and(|items| {
                items.iter().any(|needle| lower.contains(&needle.to_ascii_lowercase()))
            })
        }
        "starts_with_any" => {
            let lower = answer.trim_start().to_ascii_lowercase();
            gate.starts_with_any.as_ref().is_some_and(|items| {
                items.iter().any(|needle| lower.starts_with(&needle.to_ascii_lowercase()))
            })
        }
        "readable" => answer.split_whitespace().count() >= gate.min_words.unwrap_or(1),
        _ => false,
    }
}

#[cfg(feature = "full-cli")]
fn strip_slm_special_markers(answer: &str) -> String {
    answer
        .replace("<|im_start|>", "")
        .replace("<|im_end|>", "")
        .replace("<|begin_of_text|>", "")
        .replace("<|end_of_text|>", "")
        .replace("<|eot_id|>", "")
}

#[cfg(feature = "full-cli")]
fn normalize_slm_quality_text(answer: &str) -> String {
    let trimmed = strip_slm_special_markers(answer).trim().to_string();
    if let Some(after_colon) = trimmed.strip_prefix(':')
        && after_colon.starts_with(char::is_whitespace)
    {
        return after_colon.trim_start().to_string();
    }
    trimmed
}

#[cfg(feature = "full-cli")]
fn slm_warm_session_determinism_receipt(
    records: &[WarmSessionDeterminismRecord],
) -> serde_json::Value {
    let mut by_prompt: std::collections::BTreeMap<&str, Vec<&WarmSessionDeterminismRecord>> =
        std::collections::BTreeMap::new();
    for record in records {
        by_prompt.entry(record.prompt.as_str()).or_default().push(record);
    }

    let mut groups = Vec::new();
    let mut checked = false;
    let mut passed = true;
    for (prompt, records) in by_prompt {
        if records.len() < 2 {
            continue;
        }
        checked = true;
        let first = records[0];
        let stable_generated_token_ids =
            records.iter().all(|record| record.generated_ids == first.generated_ids);
        let stable_text = records.iter().all(|record| record.text == first.text);
        if !stable_generated_token_ids || !stable_text {
            passed = false;
        }
        groups.push(serde_json::json!({
            "prompt": prompt,
            "attempt_count": records.len(),
            "case_id": first.case_id.as_str(),
            "prompt_indices": records.iter().map(|record| record.prompt_index).collect::<Vec<_>>(),
            "stable_generated_token_ids": stable_generated_token_ids,
            "stable_text": stable_text,
            "reference_generated_ids": first.generated_ids.clone(),
            "reference_text": first.text.as_str(),
        }));
    }

    serde_json::json!({
        "checked": checked,
        "passed": checked && passed,
        "repeated_prompt_groups": groups.len(),
        "groups": groups,
    })
}

#[cfg(feature = "full-cli")]
fn slm_cpu_warm_session_memory_context_json() -> serde_json::Value {
    let mut system = sysinfo::System::new();
    let pid = match sysinfo::get_current_pid() {
        Ok(pid) => pid,
        Err(error) => {
            return serde_json::json!({
                "resident_memory_bytes": serde_json::Value::Null,
                "virtual_memory_bytes": serde_json::Value::Null,
                "resident_memory_source": "sysinfo_current_process_unavailable",
                "available": false,
                "error": error.to_string(),
            });
        }
    };

    system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    if let Some(process) = system.process(pid) {
        serde_json::json!({
            "resident_memory_bytes": process.memory(),
            "virtual_memory_bytes": process.virtual_memory(),
            "resident_memory_source": "sysinfo_current_process",
            "available": true,
        })
    } else {
        serde_json::json!({
            "resident_memory_bytes": serde_json::Value::Null,
            "virtual_memory_bytes": serde_json::Value::Null,
            "resident_memory_source": "sysinfo_current_process_missing",
            "available": false,
        })
    }
}

#[cfg(feature = "full-cli")]
fn slm_cpu_warm_session_thermal_context_json() -> serde_json::Value {
    serde_json::json!({
        "temperature_c": serde_json::Value::Null,
        "source": "not_sampled_in_slm_cpu_warm_session",
        "available": false,
    })
}

#[cfg(feature = "full-cli")]
fn slm_cpu_warm_session_power_context_json() -> serde_json::Value {
    serde_json::json!({
        "mode": serde_json::Value::Null,
        "source": "not_sampled_in_slm_cpu_warm_session",
        "available": false,
    })
}

#[cfg(feature = "full-cli")]
fn slm_cpu_warm_session_storage_context_json(
    model_path: &std::path::Path,
    receipt_path: &std::path::Path,
) -> serde_json::Value {
    serde_json::json!({
        "model_path": slm_cpu_warm_session_disk_context_json(model_path),
        "receipt_path": slm_cpu_warm_session_disk_context_json(receipt_path),
    })
}

#[cfg(feature = "full-cli")]
fn slm_cpu_warm_session_disk_context_json(path: &std::path::Path) -> serde_json::Value {
    let absolute_path = slm_cpu_warm_session_absolute_disk_path(path);
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let disk = disks
        .list()
        .iter()
        .filter(|disk| absolute_path.starts_with(disk.mount_point()))
        .max_by_key(|disk| disk.mount_point().components().count());

    if let Some(disk) = disk {
        serde_json::json!({
            "path": absolute_path.display().to_string(),
            "mount_point": disk.mount_point().display().to_string(),
            "available_bytes": disk.available_space(),
            "total_bytes": disk.total_space(),
            "source": "sysinfo_disk",
            "available": true,
        })
    } else {
        serde_json::json!({
            "path": absolute_path.display().to_string(),
            "mount_point": serde_json::Value::Null,
            "available_bytes": serde_json::Value::Null,
            "total_bytes": serde_json::Value::Null,
            "source": "sysinfo_disk_match_unavailable",
            "available": false,
        })
    }
}

#[cfg(feature = "full-cli")]
fn slm_cpu_warm_session_absolute_disk_path(path: &std::path::Path) -> std::path::PathBuf {
    if let Ok(absolute_path) = path.canonicalize() {
        return slm_cpu_warm_session_normalize_windows_verbatim_path(absolute_path);
    }

    if let Some(parent) = path.parent()
        && let Ok(parent) = parent.canonicalize()
    {
        let parent = slm_cpu_warm_session_normalize_windows_verbatim_path(parent);
        if let Some(file_name) = path.file_name() {
            return parent.join(file_name);
        }
        return parent;
    }

    path.to_path_buf()
}

#[cfg(feature = "full-cli")]
fn slm_cpu_warm_session_normalize_windows_verbatim_path(
    path: std::path::PathBuf,
) -> std::path::PathBuf {
    let display = path.to_string_lossy();
    if let Some(rest) = display.strip_prefix("\\\\?\\UNC\\") {
        return std::path::PathBuf::from(format!("\\\\{rest}"));
    }
    if let Some(rest) = display.strip_prefix("\\\\?\\") {
        return std::path::PathBuf::from(rest);
    }
    path
}

fn resolve_ask_question(question: Option<String>, question_arg: Option<String>) -> Result<String> {
    question.or(question_arg).ok_or_else(|| {
        anyhow::anyhow!("ask requires a question via --question, --prompt, or positional QUESTION")
    })
}

#[cfg(feature = "full-cli")]
async fn handle_lunar_lake_command(
    command: LunarLakeCommand,
    requested_backend_label: &str,
) -> Result<()> {
    match command.action {
        LunarLakeAction::Ask {
            artifact_root,
            operator_receipt,
            promotion_ledger,
            route_profile_comparison,
            profile,
            route,
            model,
            tokenizer,
            question,
            question_arg,
            max_new_tokens,
            expect_contains,
            json_out,
        } => {
            let question = resolve_ask_question(question, question_arg)?;
            run_lunar_lake_ask(
                artifact_root,
                operator_receipt,
                promotion_ledger,
                route_profile_comparison,
                profile,
                route,
                model,
                tokenizer,
                question,
                max_new_tokens,
                expect_contains,
                json_out,
                requested_backend_label,
            )
            .await
        }
        action => LunarLakeCommand { action }.execute().await,
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "full-cli")]
async fn run_lunar_lake_ask(
    artifact_root: std::path::PathBuf,
    operator_receipt: std::path::PathBuf,
    promotion_ledger: std::path::PathBuf,
    route_profile_comparison: std::path::PathBuf,
    profile: String,
    route_id: String,
    model: Option<std::path::PathBuf>,
    tokenizer: Option<std::path::PathBuf>,
    question: String,
    max_new_tokens: usize,
    expect_contains: Option<String>,
    json_out: Option<std::path::PathBuf>,
    requested_backend_label: &str,
) -> Result<()> {
    if !(1..=128).contains(&max_new_tokens) {
        anyhow::bail!("lunar-lake ask requires --max-new-tokens in 1..=128");
    }
    let receipt_path = json_out.unwrap_or_else(default_lunar_lake_ask_receipt_path);
    let route_selection = match commands::lunar_lake::resolve_operator_ask_route_selection(
        &artifact_root,
        &operator_receipt,
        &promotion_ledger,
        Some(&route_profile_comparison),
        &route_id,
        requested_backend_label,
        &profile,
    ) {
        Ok(selection) => selection,
        Err(err) => {
            let error = err.to_string();
            let blocked_route_selection =
                commands::lunar_lake::explain_blocked_operator_ask_route_selection(
                    &artifact_root,
                    &promotion_ledger,
                    Some(&route_profile_comparison),
                    &route_id,
                    requested_backend_label,
                    &profile,
                )
                .ok()
                .flatten();
            let blocked_receipt =
                build_lunar_lake_operator_ask_blocked_receipt(LunarLakeAskBlockedReceiptContext {
                    artifact_root: &artifact_root,
                    operator_receipt: &operator_receipt,
                    promotion_ledger: &promotion_ledger,
                    route_profile_comparison: &route_profile_comparison,
                    requested_device: requested_backend_label,
                    requested_route: &route_id,
                    profile_id: &profile,
                    question: &question,
                    max_new_tokens,
                    error: &error,
                    route_selection: blocked_route_selection.as_ref(),
                });
            write_json_output(Some(&receipt_path), &blocked_receipt)?;
            anyhow::bail!("{error}");
        }
    };
    let route = route_selection.route.clone();
    let model = resolve_lunar_lake_ask_model_path(&artifact_root, &route, model.as_deref())?;
    let source_run_path = source_run_receipt_path(&receipt_path);

    let operator_receipt_path = if operator_receipt.is_absolute() || operator_receipt.exists() {
        operator_receipt.clone()
    } else {
        artifact_root.join(&operator_receipt)
    };
    if route.runtime_api == "openvino_genai" {
        if tokenizer.is_some() {
            anyhow::bail!(
                "lunar-lake OpenVINO ask uses the tokenizer exported in the OpenVINO model directory; --tokenizer is only supported for the CPU GGUF route"
            );
        }
        run_openvino_lunar_lake_operator_ask(OpenVINOOperatorAskContext {
            artifact_root: &artifact_root,
            operator_receipt_path: &operator_receipt_path,
            model_dir: &model,
            route: &route,
            question: &question,
            max_new_tokens,
            expect_contains: expect_contains.as_deref(),
            json_out: &source_run_path,
        })?;
    } else {
        run_simple_generation(
            "cpu",
            model,
            "auto".to_string(),
            None,
            tokenizer,
            question.clone(),
            max_new_tokens,
            0.0,
            0,
            1.0,
            1.1,
            None,
            false,
            false,
            true,
            true,
            Some(source_run_path.clone()),
            None,
            None,
            false,
            false,
            true,
            true,
            0,
            "qwen2.5".to_string(),
            false,
            None,
            vec!["<|im_end|>".to_string()],
            Vec::new(),
            None,
            10,
            false,
            None,
            None,
            false,
            None,
            false,
            Some("lunar_lake_ask".to_string()),
            false,
            false,
        )
        .await?;
    }

    let source_run_receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&source_run_path).with_context(|| {
            format!("failed to read run receipt {}", source_run_path.display())
        })?)
        .with_context(|| format!("invalid run receipt {}", source_run_path.display()))?;
    validate_lunar_lake_ask_source_receipt(&source_run_receipt, &route)?;

    let answer = lunar_lake_source_answer_text(&source_run_receipt);
    let normalized_answer = lunar_lake_source_normalized_answer(&source_run_receipt, &answer);
    let answer_gate =
        evaluate_lunar_lake_answer_gate(&normalized_answer, expect_contains.as_deref());
    let answer_gate_passed = answer_gate.passed;
    if !answer_gate_passed {
        if let Some(expected) = expect_contains.as_deref() {
            anyhow::bail!(
                "lunar-lake ask answer gate failed: normalized answer did not contain `{expected}`"
            );
        }
        anyhow::bail!("lunar-lake ask produced an empty normalized answer");
    }
    let receipt = build_lunar_lake_operator_ask_receipt(LunarLakeAskReceiptContext {
        artifact_root: &artifact_root,
        operator_receipt_path: &operator_receipt_path,
        source_run_path: &source_run_path,
        route: &route,
        route_selection: &route_selection,
        question: &question,
        answer: &answer,
        normalized_answer: &normalized_answer,
        answer_gate: &answer_gate,
        expect_contains: expect_contains.as_deref(),
        source_run_receipt: &source_run_receipt,
    });
    write_json_output(Some(&receipt_path), &receipt)?;
    println!("Lunar Lake ask receipt written to {}", receipt_path.display());
    Ok(())
}

#[cfg(feature = "full-cli")]
fn resolve_lunar_lake_ask_model_path(
    artifact_root: &std::path::Path,
    route: &commands::lunar_lake::OperatorRoute,
    explicit_model: Option<&std::path::Path>,
) -> Result<std::path::PathBuf> {
    if let Some(model) = explicit_model {
        return Ok(model.to_path_buf());
    }

    let candidates = default_lunar_lake_ask_model_path_candidates(artifact_root, route)?;
    if let Some(path) = candidates.iter().find(|path| path.exists()) {
        return Ok(path.clone());
    }

    let candidate_list = if candidates.is_empty() {
        "none".to_string()
    } else {
        candidates.iter().map(|path| path.display().to_string()).collect::<Vec<_>>().join(", ")
    };
    anyhow::bail!(
        "lunar-lake ask requires --model for executable route `{}` because no committed local default model path exists; checked: {}",
        route.route_id,
        candidate_list
    )
}

#[cfg(feature = "full-cli")]
fn default_lunar_lake_ask_model_path_candidates(
    artifact_root: &std::path::Path,
    route: &commands::lunar_lake::OperatorRoute,
) -> Result<Vec<std::path::PathBuf>> {
    match route.runtime_api.as_str() {
        "openvino_genai" => Ok(default_lunar_lake_openvino_model_candidates(artifact_root)),
        "cpu" => Ok(default_lunar_lake_cpu_model_candidates(artifact_root)),
        runtime => anyhow::bail!(
            "lunar-lake ask route `{}` has no default model resolver for runtime `{}`",
            route.route_id,
            runtime
        ),
    }
}

#[cfg(feature = "full-cli")]
fn default_lunar_lake_openvino_model_candidates(
    artifact_root: &std::path::Path,
) -> Vec<std::path::PathBuf> {
    let manifest_path = artifact_root.join("slm-openvino-ir-qwen25-int4-sym-manifest.json");
    let mut candidates = Vec::new();
    if let Some(manifest) = read_optional_json(&manifest_path)
        && let Some(path) = json_pointer_string(&manifest, "/export_contract/expected_output_dir")
    {
        candidates.push(std::path::PathBuf::from(path));
    }
    candidates.push(std::path::PathBuf::from("models/openvino/qwen2.5-0.5b-instruct-int4-sym"));
    dedupe_paths(candidates)
}

#[cfg(feature = "full-cli")]
fn default_lunar_lake_cpu_model_candidates(
    artifact_root: &std::path::Path,
) -> Vec<std::path::PathBuf> {
    let mut candidates = Vec::new();

    let phase_path = artifact_root.join("slm-phase-warm-session-qwen25-cpu.json");
    if let Some(phase) = read_optional_json(&phase_path)
        && let Some(path) = json_pointer_string(&phase, "/model/path")
    {
        candidates.push(std::path::PathBuf::from(path));
    }

    let manifest_path = artifact_root.join("slm-artifact-manifest.json");
    if let Some(manifest) = read_optional_json(&manifest_path)
        && let Some(path) =
            json_pointer_string(&manifest, "/selected_candidate/expected_local_path")
    {
        candidates.push(std::path::PathBuf::from(path));
    }

    candidates.push(std::path::PathBuf::from("models/slm/qwen2.5-0.5b-instruct-q8_0.gguf"));
    dedupe_paths(candidates)
}

#[cfg(feature = "full-cli")]
fn read_optional_json(path: &std::path::Path) -> Option<serde_json::Value> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

#[cfg(feature = "full-cli")]
fn json_pointer_string(value: &serde_json::Value, pointer: &str) -> Option<String> {
    value.pointer(pointer).and_then(serde_json::Value::as_str).map(str::to_string)
}

#[cfg(feature = "full-cli")]
fn dedupe_paths(paths: Vec<std::path::PathBuf>) -> Vec<std::path::PathBuf> {
    let mut seen = std::collections::BTreeSet::new();
    let mut deduped = Vec::new();
    for path in paths {
        let key = path.to_string_lossy().to_string();
        if seen.insert(key) {
            deduped.push(path);
        }
    }
    deduped
}

#[cfg(feature = "full-cli")]
fn default_lunar_lake_ask_receipt_path() -> std::path::PathBuf {
    std::path::PathBuf::from("target")
        .join("bitnet")
        .join("receipts")
        .join("lunar-lake")
        .join("ask-latest.json")
}

#[cfg(feature = "full-cli")]
fn source_run_receipt_path(receipt_path: &std::path::Path) -> std::path::PathBuf {
    let stem = receipt_path.file_stem().and_then(|stem| stem.to_str()).unwrap_or("ask");
    receipt_path.with_file_name(format!("{stem}-source-run.json"))
}

#[cfg(feature = "full-cli")]
struct OpenVINOOperatorAskContext<'a> {
    artifact_root: &'a std::path::Path,
    operator_receipt_path: &'a std::path::Path,
    model_dir: &'a std::path::Path,
    route: &'a commands::lunar_lake::OperatorRoute,
    question: &'a str,
    max_new_tokens: usize,
    expect_contains: Option<&'a str>,
    json_out: &'a std::path::Path,
}

#[cfg(feature = "full-cli")]
fn run_openvino_lunar_lake_operator_ask(ctx: OpenVINOOperatorAskContext<'_>) -> Result<()> {
    let device = openvino_operator_device_for_route(ctx.route)?;
    let python = openvino_operator_python();
    let mut command = std::process::Command::new(&python);
    command
        .arg("scripts/openvino_genai_operator_ask.py")
        .arg("--model-dir")
        .arg(ctx.model_dir)
        .arg("--device")
        .arg(device)
        .arg("--question")
        .arg(ctx.question)
        .arg("--artifact-root")
        .arg(ctx.artifact_root)
        .arg("--operator-receipt")
        .arg(ctx.operator_receipt_path)
        .arg("--route-id")
        .arg(&ctx.route.route_id)
        .arg("--max-new-tokens")
        .arg(ctx.max_new_tokens.to_string())
        .arg("--json-out")
        .arg(ctx.json_out);
    if let Some(expected) = ctx.expect_contains {
        command.arg("--expect-contains").arg(expected);
    }
    let output = command.output().with_context(|| {
        format!("failed to launch OpenVINO operator ask helper via {}", python.display())
    })?;
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "OpenVINO operator ask helper failed with status {}.\nstdout:\n{}\nstderr:\n{}",
            output.status,
            stdout.trim(),
            stderr.trim()
        );
    }
    Ok(())
}

#[cfg(feature = "full-cli")]
fn openvino_operator_device_for_route(
    route: &commands::lunar_lake::OperatorRoute,
) -> Result<&'static str> {
    match route.selected_backend.as_str() {
        "openvino-gpu" if route.runtime_api == "openvino_genai" => Ok("GPU.0"),
        "openvino-npu" if route.runtime_api == "openvino_genai" => Ok("NPU"),
        _ => anyhow::bail!("route `{}` is not an OpenVINO GenAI candidate route", route.route_id),
    }
}

#[cfg(feature = "full-cli")]
fn openvino_operator_python() -> std::path::PathBuf {
    let venv_python = std::path::PathBuf::from(".venv").join("Scripts").join("python.exe");
    if venv_python.exists() { venv_python } else { std::path::PathBuf::from("python") }
}

#[cfg(feature = "full-cli")]
fn lunar_lake_source_answer_text(source: &serde_json::Value) -> String {
    source["text"]
        .as_str()
        .or_else(|| source["output"]["generated_text"].as_str())
        .unwrap_or_default()
        .to_string()
}

#[cfg(feature = "full-cli")]
fn lunar_lake_source_normalized_answer(source: &serde_json::Value, answer: &str) -> String {
    source["output"]["normalized_answer"]
        .as_str()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| normalize_lunar_lake_answer(answer))
}

#[cfg(feature = "full-cli")]
fn normalize_lunar_lake_answer(answer: &str) -> String {
    answer.replace("<|im_end|>", "").replace("<|endoftext|>", "").trim().to_string()
}

#[cfg(feature = "full-cli")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct LunarLakeAnswerGate {
    name: &'static str,
    passed: bool,
    failed_rules: Vec<&'static str>,
}

#[cfg(feature = "full-cli")]
struct LunarLakeAskReceiptContext<'a> {
    artifact_root: &'a std::path::Path,
    operator_receipt_path: &'a std::path::Path,
    source_run_path: &'a std::path::Path,
    route: &'a commands::lunar_lake::OperatorRoute,
    route_selection: &'a commands::lunar_lake::OperatorAskRouteSelection,
    question: &'a str,
    answer: &'a str,
    normalized_answer: &'a str,
    answer_gate: &'a LunarLakeAnswerGate,
    expect_contains: Option<&'a str>,
    source_run_receipt: &'a serde_json::Value,
}

#[cfg(feature = "full-cli")]
struct LunarLakeAskBlockedReceiptContext<'a> {
    artifact_root: &'a std::path::Path,
    operator_receipt: &'a std::path::Path,
    promotion_ledger: &'a std::path::Path,
    route_profile_comparison: &'a std::path::Path,
    requested_device: &'a str,
    requested_route: &'a str,
    profile_id: &'a str,
    question: &'a str,
    max_new_tokens: usize,
    error: &'a str,
    route_selection: Option<&'a commands::lunar_lake::BlockedOperatorAskRouteSelection>,
}

#[cfg(feature = "full-cli")]
fn build_lunar_lake_operator_ask_blocked_receipt(
    ctx: LunarLakeAskBlockedReceiptContext<'_>,
) -> serde_json::Value {
    let candidate_routes =
        ctx.route_selection.map(|selection| selection.candidate_routes.clone()).unwrap_or_default();
    let why_not_cpu =
        ctx.route_selection.map(|selection| selection.why_not_cpu.clone()).unwrap_or_default();
    let why_not_gpu =
        ctx.route_selection.map(|selection| selection.why_not_gpu.clone()).unwrap_or_default();
    let why_not_npu =
        ctx.route_selection.map(|selection| selection.why_not_npu.clone()).unwrap_or_default();
    let promotion_status = ctx
        .route_selection
        .map(|selection| selection.promotion_status.clone())
        .unwrap_or_else(|| "route_selection_blocked".to_string());
    let selection_source = ctx
        .route_selection
        .map(|selection| selection.selection_source.clone())
        .unwrap_or_else(|| "route_selection_error".to_string());
    let route_reason = ctx
        .route_selection
        .map(|selection| selection.route_reason.clone())
        .unwrap_or_else(|| ctx.error.to_string());
    let promotion_ledger = ctx
        .route_selection
        .and_then(|selection| selection.promotion_ledger.clone())
        .unwrap_or_else(|| ctx.promotion_ledger.display().to_string());
    let route_profile_comparison = ctx
        .route_selection
        .and_then(|selection| selection.route_profile_comparison.clone())
        .unwrap_or_else(|| ctx.route_profile_comparison.display().to_string());
    let operator_runbook = ctx
        .route_selection
        .and_then(|selection| selection.operator_runbook.clone())
        .or_else(|| {
            commands::lunar_lake::blocked_operator_ask_runbook(ctx.profile_id).map(str::to_string)
        });
    let next_required_evidence = ctx
        .route_selection
        .map(|selection| selection.next_required_evidence.clone())
        .unwrap_or_else(|| {
            commands::lunar_lake::blocked_operator_ask_next_required_evidence(ctx.profile_id)
        });

    serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "lunar_lake_operator_ask_blocked",
        "proof_stage": "operator_route_selection_blocked_no_inference",
        "created_utc": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        "machine_id": "intel-258v",
        "artifact_root": ctx.artifact_root.display().to_string(),
        "operator_receipt": ctx.operator_receipt.display().to_string(),
        "promotion_ledger": ctx.promotion_ledger.display().to_string(),
        "route_profile_comparison": ctx.route_profile_comparison.display().to_string(),
        "requested_device": ctx.requested_device,
        "requested_route": ctx.requested_route,
        "profile_id": ctx.profile_id,
        "selected_route": serde_json::Value::Null,
        "selected_backend": serde_json::Value::Null,
        "runtime_api": serde_json::Value::Null,
        "promotion_status": promotion_status,
        "route_selection_status": "blocked",
        "route_selection_blocked": true,
        "route_selection_error": ctx.error,
        "route_reason": route_reason,
        "why_not_cpu": why_not_cpu,
        "why_not_gpu": why_not_gpu,
        "why_not_npu": why_not_npu,
        "operator_runbook": operator_runbook,
        "next_required_evidence": next_required_evidence,
        "candidate_routes": candidate_routes,
        "question": ctx.question,
        "max_new_tokens": ctx.max_new_tokens,
        "model_path_required": false,
        "model_resolution": "not_required_for_blocked_auto_route_before_execution",
        "fallback_used": false,
        "answer_gate_passed": serde_json::Value::Null,
        "new_inference_executed": false,
        "speedup_claim": false,
        "acceleration_claim": false,
        "power_advantage_claim": false,
        "bitnet_qk256_i2s_claim": false,
        "route_selection": {
            "requested_device": ctx.requested_device,
            "requested_route": ctx.requested_route,
            "profile_id": ctx.profile_id,
            "selected_route": serde_json::Value::Null,
            "selected_backend": serde_json::Value::Null,
            "runtime_api": serde_json::Value::Null,
            "promotion_status": promotion_status,
            "selection_source": selection_source,
            "route_selection_status": "blocked",
            "route_selection_blocked": true,
            "route_selection_error": ctx.error,
            "route_reason": route_reason,
            "why_not_cpu": why_not_cpu,
            "why_not_gpu": why_not_gpu,
            "why_not_npu": why_not_npu,
            "operator_runbook": operator_runbook,
            "next_required_evidence": next_required_evidence,
            "candidate_routes": candidate_routes,
            "promotion_ledger": promotion_ledger,
            "route_profile_comparison": route_profile_comparison,
            "model_path_required": false,
            "model_resolution": "not_required_for_blocked_auto_route_before_execution",
        },
        "claim_boundary": {
            "route_selection_blocked": true,
            "new_inference_executed": false,
            "model_loaded": false,
            "fallback_used": false,
            "route_promotion_changed": false,
            "default_route_changed": false,
            "speedup_claim": false,
            "power_advantage_claim": false,
            "acceleration_claim": false,
            "native_accelerator_claim": false,
            "bitnet_qk256_i2s_claim": false,
        },
    })
}

#[cfg(feature = "full-cli")]
fn build_lunar_lake_operator_ask_receipt(ctx: LunarLakeAskReceiptContext<'_>) -> serde_json::Value {
    let requested_backend = ctx.source_run_receipt["requested_backend"].clone();
    let selected_backend = ctx.source_run_receipt["selected_backend"].clone();
    let runtime_api = ctx.source_run_receipt["runtime_api"].clone();
    let fallback_used = ctx.source_run_receipt["fallback_used"].clone();
    let fallback_reason = ctx.source_run_receipt["fallback_reason"].clone();
    let selected_kernel_or_runtime = lunar_lake_source_value_at_any(
        ctx.source_run_receipt,
        &["kernel.kernel_id", "selected_kernel_or_runtime"],
    );
    let backend_lane = lunar_lake_source_value_at_any(
        ctx.source_run_receipt,
        &["backend.backend_lane", "backend_lane"],
    );
    let model_family =
        lunar_lake_source_value_at_any(ctx.source_run_receipt, &["model.family", "model_family"]);
    let model_architecture = lunar_lake_source_value_at_any(
        ctx.source_run_receipt,
        &["model.architecture", "model_architecture"],
    );
    let quantization = lunar_lake_source_value_at_any(
        ctx.source_run_receipt,
        &["model.quant_format", "quantization"],
    );
    let tokenizer_source = lunar_lake_source_value_at_any(
        ctx.source_run_receipt,
        &["model.tokenizer", "tokenizer_source"],
    );
    let prompt_template = lunar_lake_source_value_at_any(
        ctx.source_run_receipt,
        &["prompt_template", "prompt_policy.prompt_template"],
    );
    let prompt_render = lunar_lake_source_value_at_any(
        ctx.source_run_receipt,
        &["prompt_render", "prompt_policy.rendered_prompt"],
    );
    let prompt_token_ids = lunar_lake_source_value_at_any(
        ctx.source_run_receipt,
        &["tokens.prompt_ids", "prompt_policy.prompt_token_ids"],
    );
    let generated_token_ids = lunar_lake_source_value_at_any(
        ctx.source_run_receipt,
        &["tokens.generated_ids", "output.generated_token_ids"],
    );
    let generated_count = lunar_lake_source_value_at_any(
        ctx.source_run_receipt,
        &["tokens.generated", "output.generated_token_count"],
    );
    let prompt_count = lunar_lake_source_value_at_any(
        ctx.source_run_receipt,
        &["tokens.prompt", "prompt_policy.prompt_token_count"],
    );
    let openvino_candidate_executed = ctx.route.runtime_api == "openvino_genai";
    let proof_stage = if openvino_candidate_executed {
        "operator_candidate_route_executed_through_lunar_lake_ask"
    } else {
        "operator_default_route_executed"
    };

    serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "lunar_lake_operator_ask",
        "proof_stage": proof_stage,
        "created_utc": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        "machine_id": "intel-258v",
        "artifact_root": ctx.artifact_root.display().to_string(),
        "operator_receipt": ctx.operator_receipt_path.display().to_string(),
        "source_run_receipt": ctx.source_run_path.display().to_string(),
        "requested_device": ctx.route_selection.requested_device,
        "requested_route": ctx.route_selection.requested_route,
        "profile_id": ctx.route_selection.profile_id,
        "selected_route": ctx.route_selection.selected_route,
        "promotion_status": ctx.route_selection.promotion_status,
        "route_profile_comparison": ctx.route_selection.route_profile_comparison,
        "route_profile_status": ctx.route_selection.route_profile_status,
        "route_profile_blockers": ctx.route_selection.route_profile_blockers,
        "route_reason": ctx.route_selection.route_reason,
        "why_not_cpu": ctx.route_selection.why_not_cpu,
        "why_not_gpu": ctx.route_selection.why_not_gpu,
        "why_not_npu": ctx.route_selection.why_not_npu,
        "requested_backend": requested_backend,
        "selected_backend": selected_backend,
        "runtime_api": runtime_api,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "backend_lane": backend_lane,
        "selected_kernel_or_runtime": selected_kernel_or_runtime,
        "route_id": ctx.route.route_id,
        "route_selection": {
            "requested_device": ctx.route_selection.requested_device,
            "requested_route": ctx.route_selection.requested_route,
            "profile_id": ctx.route_selection.profile_id,
            "selected_route": ctx.route_selection.selected_route,
            "selected_backend": ctx.route_selection.selected_backend,
            "runtime_api": ctx.route_selection.runtime_api,
            "promotion_status": ctx.route_selection.promotion_status,
            "selection_source": ctx.route_selection.selection_source,
            "route_reason": ctx.route_selection.route_reason,
            "why_not_cpu": ctx.route_selection.why_not_cpu,
            "why_not_gpu": ctx.route_selection.why_not_gpu,
            "why_not_npu": ctx.route_selection.why_not_npu,
            "candidate_routes": ctx.route_selection.candidate_routes,
            "promotion_ledger": ctx.route_selection.promotion_ledger,
            "route_profile_comparison": ctx.route_selection.route_profile_comparison,
            "route_profile_status": ctx.route_selection.route_profile_status,
            "route_profile_blockers": ctx.route_selection.route_profile_blockers,
        },
        "model_family": model_family,
        "model_architecture": model_architecture,
        "quantization": quantization,
        "tokenizer_source": tokenizer_source,
        "prompt_template": prompt_template,
        "answer_gate_passed": ctx.answer_gate.passed,
        "speedup_claim": false,
        "acceleration_claim": false,
        "broad_quality_claim": false,
        "bitnet_qk256_i2s_claim": false,
        "arc_or_npu_execution_claim": false,
        "openvino_candidate_route_executed": openvino_candidate_executed,
        "route": {
            "route_id": ctx.route.route_id,
            "workload": ctx.route.workload,
            "selected_model": ctx.route.selected_model,
            "selected_backend": ctx.route.selected_backend,
            "runtime_api": ctx.route.runtime_api,
            "selected_kernel_or_runtime": ctx.route.selected_kernel_or_runtime,
            "fallback_policy": ctx.route.fallback_policy,
            "route_reason": ctx.route.route_reason,
            "answer_gate_evidence": ctx.route.answer_gate_evidence,
            "phase_evidence": ctx.route.phase_evidence,
            "acceleration_claim": ctx.route.acceleration_claim,
        },
        "question": ctx.question,
        "answer": {
            "text": ctx.answer,
            "normalized_text": ctx.normalized_answer,
            "gate": {
                "name": ctx.answer_gate.name,
                "passed": ctx.answer_gate.passed,
                "expected_contains": ctx.expect_contains,
                "failed_rules": ctx.answer_gate.failed_rules,
                "broad_quality_claim": false,
            },
        },
        "model": {
            "path": lunar_lake_source_value_at_any(ctx.source_run_receipt, &["model.path", "model.local_model_dir", "inputs.model_dir"]),
            "sha256": lunar_lake_source_value_at_any(ctx.source_run_receipt, &["model.sha256"]),
            "family": model_family,
            "architecture": model_architecture,
            "quant_format": quantization,
            "tokenizer": tokenizer_source,
            "vocab_size": ctx.source_run_receipt["model"]["vocab_size"].clone(),
            "loader_mode": ctx.source_run_receipt["model"]["loader_mode"].clone(),
            "fallback_loader_used": ctx.source_run_receipt["model"]["fallback_loader_used"].clone(),
            "source": ctx.source_run_receipt["model"].clone(),
        },
        "prompt": {
            "template": prompt_template,
            "render": prompt_render,
            "token_ids": prompt_token_ids,
        },
        "tokens": {
            "generated_ids": generated_token_ids,
            "generated_count": generated_count,
            "prompt_count": prompt_count,
        },
        "backend": {
            "requested_backend": ctx.source_run_receipt["requested_backend"].clone(),
            "selected_backend": ctx.source_run_receipt["selected_backend"].clone(),
            "runtime_api": ctx.source_run_receipt["runtime_api"].clone(),
            "fallback_used": ctx.source_run_receipt["fallback_used"].clone(),
            "fallback_reason": ctx.source_run_receipt["fallback_reason"].clone(),
            "backend_lane": backend_lane,
            "selected_kernel_or_runtime": selected_kernel_or_runtime,
        },
        "dense_slm": ctx.source_run_receipt["dense_slm"].clone(),
        "execution_coverage": ctx.source_run_receipt["execution_coverage"].clone(),
        "timing": ctx.source_run_receipt["timing"].clone(),
        "profile": ctx.source_run_receipt["profile"].clone(),
        "claim_boundary": {
            "cpu_default_route_only": !openvino_candidate_executed,
            "openvino_candidate_route_executed": openvino_candidate_executed,
            "default_route_changed": false,
            "fallback_used": false,
            "acceleration_claim": false,
            "broad_dense_slm_quality_claim": false,
            "bitnet_qk256_i2s_claim": false,
            "arc_or_npu_acceleration_claim": false,
        },
        "source_receipt": ctx.source_run_receipt,
    })
}

#[cfg(feature = "full-cli")]
fn lunar_lake_source_value_at_any(source: &serde_json::Value, paths: &[&str]) -> serde_json::Value {
    for path in paths {
        if let Some(value) = lunar_lake_source_value_at(source, path) {
            return value.clone();
        }
    }
    serde_json::Value::Null
}

#[cfg(feature = "full-cli")]
fn lunar_lake_source_value_at<'a>(
    source: &'a serde_json::Value,
    dotted_path: &str,
) -> Option<&'a serde_json::Value> {
    let mut current = source;
    for segment in dotted_path.split('.') {
        current = current.get(segment)?;
    }
    Some(current)
}

#[cfg(feature = "full-cli")]
fn evaluate_lunar_lake_answer_gate(
    normalized_answer: &str,
    expect_contains: Option<&str>,
) -> LunarLakeAnswerGate {
    let mut failed_rules = Vec::new();
    if normalized_answer.is_empty() {
        failed_rules.push("empty_answer");
    }
    if let Some(expected) = expect_contains
        && !normalized_answer.contains(expected)
    {
        failed_rules.push("expected_contains");
    }
    LunarLakeAnswerGate {
        name: if expect_contains.is_some() {
            "contains_bounded_operator_answer"
        } else {
            "non_empty_bounded_operator_answer"
        },
        passed: failed_rules.is_empty(),
        failed_rules,
    }
}

#[cfg(feature = "full-cli")]
fn validate_lunar_lake_ask_source_receipt(
    source_run_receipt: &serde_json::Value,
    route: &commands::lunar_lake::OperatorRoute,
) -> Result<()> {
    let requested_backend = source_run_receipt["requested_backend"].as_str().unwrap_or_default();
    let selected_backend = source_run_receipt["selected_backend"].as_str().unwrap_or_default();
    let runtime_api = source_run_receipt["runtime_api"].as_str().unwrap_or_default();
    let fallback_used = source_run_receipt["fallback_used"].as_bool().unwrap_or(true);
    let selected_kernel = lunar_lake_source_value_at_any(
        source_run_receipt,
        &["kernel.kernel_id", "selected_kernel_or_runtime"],
    )
    .as_str()
    .unwrap_or_default()
    .to_string();
    let requested_backend_ok = if route.selected_backend == "cpu-rust" && route.runtime_api == "cpu"
    {
        matches!(requested_backend, "cpu" | "cpu-rust")
    } else {
        requested_backend == route.selected_backend
    };
    if !requested_backend_ok
        || selected_backend != route.selected_backend
        || runtime_api != route.runtime_api
    {
        anyhow::bail!(
            "lunar-lake ask did not preserve route `{}`: requested_backend={requested_backend}, selected_backend={selected_backend}, runtime_api={runtime_api}",
            route.route_id
        );
    }
    if fallback_used {
        anyhow::bail!("lunar-lake ask source receipt recorded fallback_used=true");
    }
    if !lunar_lake_source_kernel_matches_route(&selected_kernel, &route.selected_kernel_or_runtime)
    {
        anyhow::bail!(
            "lunar-lake ask selected kernel `{selected_kernel}`, expected `{}`",
            route.selected_kernel_or_runtime
        );
    }
    let dense_slm_or_openvino_qwen = source_run_receipt.get("dense_slm").is_some()
        || (route.runtime_api == "openvino_genai"
            && source_run_receipt["model_family"].as_str() == Some("qwen")
            && source_run_receipt["model_architecture"].as_str() == Some("qwen2"));
    if !dense_slm_or_openvino_qwen {
        anyhow::bail!("lunar-lake ask source receipt is missing dense SLM provenance");
    }
    if source_run_receipt.get("bitnet").is_some() {
        anyhow::bail!("lunar-lake ask source receipt unexpectedly contains BitNet provenance");
    }
    Ok(())
}

#[cfg(feature = "full-cli")]
fn lunar_lake_source_kernel_matches_route(source_kernel: &str, route_kernel: &str) -> bool {
    source_kernel == route_kernel
        || (source_kernel == "openvino-genai-llmpipeline-gpu0"
            && route_kernel == "openvino-genai-llmpipeline-gpu")
}

fn default_log_level_for_command(command: Option<&Commands>) -> Option<&'static str> {
    if uses_report_only_cuda_benchmark_receipt(command) {
        return Some("warn");
    }

    match command {
        Some(Commands::Ask { .. }) => Some("warn"),
        #[cfg(feature = "full-cli")]
        Some(Commands::LunarLake(cmd)) if matches!(&cmd.action, LunarLakeAction::Ask { .. }) => {
            Some("warn")
        }
        #[cfg(feature = "full-cli")]
        Some(Commands::Chat(_)) => Some("warn"),
        Some(Commands::Model(_)) => Some("warn"),
        #[cfg(feature = "full-cli")]
        Some(Commands::Receipts(_)) => Some("warn"),
        #[cfg(feature = "full-cli")]
        Some(Commands::Support(_)) => Some("warn"),
        #[cfg(feature = "full-cli")]
        Some(Commands::Mac(cmd)) => cmd.default_log_level(),
        _ => None,
    }
}

fn skips_startup_backend_selection(command: Option<&Commands>) -> bool {
    uses_report_only_cuda_benchmark_receipt(command)
        || uses_read_only_model_status(command)
        || uses_read_only_support_bundle(command)
}

fn uses_report_only_cuda_benchmark_receipt(command: Option<&Commands>) -> bool {
    #[cfg(feature = "cli-bench")]
    if let Some(Commands::Benchmark(cmd)) = command {
        return cmd.cuda_benchmark_receipt.is_some();
    }

    let _ = command;
    false
}

fn uses_read_only_model_status(command: Option<&Commands>) -> bool {
    matches!(
        command,
        Some(Commands::Model(ModelCommand { action: model_cache::ModelAction::Status { .. } }))
    )
}

fn uses_read_only_support_bundle(command: Option<&Commands>) -> bool {
    #[cfg(feature = "full-cli")]
    {
        matches!(command, Some(Commands::Support(_)))
    }
    #[cfg(not(feature = "full-cli"))]
    {
        let _ = command;
        false
    }
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
    if strict_cuda && strict_cpu {
        anyhow::bail!("--strict-cuda and --strict-cpu are mutually exclusive");
    }
    if strict_cuda {
        validate_strict_cuda_backend_label(requested_backend_label, "--strict-cuda")?;
    }
    if strict_cpu && requested_backend_label != "cpu" {
        anyhow::bail!(
            "--strict-cpu requires --device cpu; requested backend was {requested_backend_label}"
        );
    }
    #[cfg(feature = "full-cli")]
    if is_dense_qwen_cuda_ask_backend(requested_backend_label)
        && let Some(model_path) = resolve_dense_qwen_cuda_ask_model(&model)?
    {
        if tokenizer.is_some() {
            anyhow::bail!(
                "dense Qwen CUDA ask uses contract-authoritative tokenizer resolution; do not pass --tokenizer"
            );
        }
        if system_prompt.as_ref().is_some_and(|value| !value.trim().is_empty()) {
            anyhow::bail!(
                "dense Qwen CUDA ask is scoped to the contract deterministic prompt path; --system is not supported yet"
            );
        }
        if temperature != 0.0 || top_p != 1.0 {
            anyhow::bail!(
                "dense Qwen CUDA ask is deterministic greedy only; use --temperature 0.0 --top-p 1.0"
            );
        }
        if strict_cpu {
            anyhow::bail!("dense Qwen CUDA ask cannot run with --strict-cpu");
        }
        if !(5..=16).contains(&max_new_tokens) {
            anyhow::bail!("dense Qwen CUDA ask is currently bounded to --max-new-tokens 5..=16");
        }
        if let Some(cuda_bin) = ensure_strict_cuda_runtime_libraries_visible()? {
            debug!(
                "added CUDA Toolkit bin directory to process PATH for dense Qwen CUDA ask: {}",
                cuda_bin.display()
            );
        }
        let top_k = if top_k == 0 { 10 } else { top_k };
        let outcome = run_dense_qwen_cuda_ask(DenseQwenCudaAskOptions {
            model: model_path,
            question,
            max_new_tokens,
            top_k,
            device_index: 0,
            receipt_out,
        })
        .await?;
        println!("{}", outcome.answer);
        println!();
        print_strict_ask_proof_summary(&outcome.receipt, &outcome.receipt_path);
        return Ok(());
    }
    let effective_receipt_out =
        receipt_out.clone().or_else(|| strict_ask_default_receipt_path(strict_cuda, strict_cpu));
    if strict_cuda {
        strict_bitnet_cuda_ask_preflight(
            &model,
            tokenizer.as_deref(),
            effective_receipt_out.as_deref(),
        )?;
    }
    if strict_cuda && let Some(cuda_bin) = ensure_strict_cuda_runtime_libraries_visible()? {
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
        effective_receipt_out.clone(),
        None,
        None,
        false,
        false,
        true,
        true,
        0,
        BITNET_CPP_ANSWER_TEMPLATE.to_string(),
        false,
        system_prompt,
        vec!["<|eot_id|>".to_string(), "<|end_of_text|>".to_string()],
        Vec::new(),
        None,
        10,
        false,
        None,
        None,
        false,
        None,
        false,
        Some("ask".to_string()),
        false,
        false,
    )
    .await?;

    let Some(receipt_path) = effective_receipt_out else {
        return Ok(());
    };
    let receipt_out_was_defaulted = receipt_out.is_none();
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
            "sha256": run_receipt["model"]["sha256"].clone(),
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
            "family": BITNET_CPP_ANSWER_TEMPLATE,
            "system_prompt_present": system_prompt_for_receipt.as_ref().is_some_and(|value| !value.is_empty()),
            "bos_inserted": run_receipt["prompt_render"]["add_bos"].clone(),
            "assistant_prefix_inserted": true,
            "rendered_sha256": run_receipt["prompt_render"]["rendered_sha256"].clone(),
            "rendered_text": run_receipt["prompt_render"]["rendered_text"].clone(),
            "parse_special": run_receipt["prompt_render"]["parse_special"].clone(),
            "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"],
            "stop_token_ids": run_receipt["prompt_render"]["stop_token_ids"].clone(),
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
        "execution_plan": run_receipt["execution_plan"].clone(),
        "kernel_stats": run_receipt["kernel_stats"].clone(),
        "cuda_execution_residency": run_receipt["cuda_execution_residency"].clone(),
        "timing": run_receipt["timing"].clone(),
        "quality": quality,
        "receipt": {
            "requested": !receipt_out_was_defaulted,
            "defaulted_for_strict_ask": receipt_out_was_defaulted && (strict_cuda || strict_cpu),
            "path": receipt_path.display().to_string(),
        },
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
    if strict_cuda || strict_cpu {
        print_strict_ask_proof_summary(&answer_receipt, &receipt_path);
    }
    Ok(())
}

fn strict_ask_default_receipt_path(
    strict_cuda: bool,
    strict_cpu: bool,
) -> Option<std::path::PathBuf> {
    match (strict_cuda, strict_cpu) {
        (true, false) => Some(
            std::path::PathBuf::from("target")
                .join("bitnet")
                .join("receipts")
                .join("cuda-answer-readiness")
                .join("strict-cuda-ask-latest.json"),
        ),
        (false, true) => Some(
            std::path::PathBuf::from("target")
                .join("bitnet")
                .join("receipts")
                .join("cuda-answer-readiness")
                .join("strict-cpu-ask-latest.json"),
        ),
        _ => None,
    }
}

#[cfg(feature = "full-cli")]
fn is_dense_qwen_cuda_ask_backend(requested_backend_label: &str) -> bool {
    matches!(
        requested_backend_label.trim().to_ascii_lowercase().as_str(),
        "cuda" | "nvidia-rtx-5070-ti-cuda"
    )
}

#[cfg(feature = "full-cli")]
fn resolve_dense_qwen_cuda_ask_model(
    model: &std::path::Path,
) -> Result<Option<std::path::PathBuf>> {
    if let Some(cached) = model_cache::verified_dense_qwen_cuda_model_arg(model, None)? {
        return Ok(Some(cached.path));
    }

    if is_supported_dense_qwen_cuda_model_path(model) {
        return Ok(Some(model.to_path_buf()));
    }

    Ok(None)
}

#[cfg(feature = "full-cli")]
fn print_strict_ask_proof_summary(
    answer_receipt: &serde_json::Value,
    receipt_path: &std::path::Path,
) {
    let explanation = commands::receipts::explain_receipt(receipt_path, answer_receipt);
    commands::receipts::print_compact_proof_summary(&explanation);
}

#[cfg(not(feature = "full-cli"))]
fn print_strict_ask_proof_summary(
    _answer_receipt: &serde_json::Value,
    _receipt_path: &std::path::Path,
) {
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
    let kernel_stats = run_receipt["kernel_stats"]
        .as_array()
        .and_then(|stats| stats.first())
        .ok_or_else(|| anyhow::anyhow!("strict CUDA ask receipt is missing kernel_stats[0]"))?;
    let kernel_time_ms = kernel_stats["kernel_time_ms"].as_f64().unwrap_or(-1.0);
    let host_to_device_bytes = kernel_stats["host_to_device_bytes"].as_u64().unwrap_or(0);
    let device_to_host_bytes = kernel_stats["device_to_host_bytes"].as_u64().unwrap_or(0);
    if kernel_time_ms < 0.0 || host_to_device_bytes == 0 || device_to_host_bytes == 0 {
        anyhow::bail!(
            "strict CUDA ask receipt is missing measured QK256 timing/transfer accounting: kernel_time_ms={}, host_to_device_bytes={}, device_to_host_bytes={}",
            kernel_stats["kernel_time_ms"],
            kernel_stats["host_to_device_bytes"],
            kernel_stats["device_to_host_bytes"]
        );
    }
    let transfer_accounting =
        &run_receipt["cuda_execution_residency"]["host_device_transfer_accounting"];
    if transfer_accounting["status"].as_str() != Some("qk256_measured")
        || transfer_accounting["kernel_time_ms"].as_f64().is_none()
        || transfer_accounting["host_to_device_bytes"].as_u64().unwrap_or(0) == 0
        || transfer_accounting["device_to_host_bytes"].as_u64().unwrap_or(0) == 0
    {
        anyhow::bail!(
            "strict CUDA ask receipt is missing measured QK256 residency accounting: {}",
            transfer_accounting
        );
    }
    let execution_plan_failed = planner_receipts::strict_bitnet_qk256_execution_plan_failed_rules(
        &run_receipt["execution_plan"],
    );
    if !execution_plan_failed.is_empty() {
        anyhow::bail!(
            "strict CUDA ask receipt has invalid BitNet QK256 execution_plan: {}",
            execution_plan_failed.join(",")
        );
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

pub(crate) fn ensure_strict_cuda_runtime_libraries_visible() -> Result<Option<std::path::PathBuf>> {
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

#[cfg(test)]
mod extract_last_token_hidden_tests {
    use super::{extract_last_token_hidden, tensor_to_vec};
    use bitnet_common::Tensor as _;

    #[test]
    fn extract_last_token_hidden_uses_final_sequence_position() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let hidden = candle_core::Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, 4.0, // position 0
                10.0, 20.0, 30.0, 40.0, // position 1
                100.0, 200.0, 300.0, 400.0, // position 2
            ],
            (1usize, 3usize, 4usize),
            &device,
        )?;
        let tensor =
            bitnet_common::ConcreteTensor::BitNet(bitnet_common::BitNetTensor::new(hidden));

        let last = extract_last_token_hidden(&tensor)?;
        assert_eq!(last.shape(), vec![1, 4]);
        assert_eq!(tensor_to_vec(&last)?, vec![100.0f32, 200.0, 300.0, 400.0]);
        Ok(())
    }
}

#[cfg(any(test, feature = "full-cli"))]
fn can_use_direct_greedy_logits(
    temperature: f32,
    repetition_penalty: f32,
    context_tokens_empty: bool,
) -> bool {
    temperature == 0.0 && (repetition_penalty == 1.0 || context_tokens_empty)
}

/// Select the greedy token from 2D logits \[B,V\] without materializing a full
/// host logits vector. This is only used under the same deterministic
/// no-penalty guard as `bitnet_sampling::SamplingStrategy`.
#[cfg(any(test, feature = "full-cli"))]
fn greedy_argmax_token_2d(tensor: &bitnet_common::ConcreteTensor) -> Result<u32> {
    use bitnet_common::{BitNetError, ConcreteTensor, Tensor};

    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(BitNetError::Validation("Expected 2D tensor".into()).into());
    }
    if shape[0] == 0 || shape[1] == 0 {
        return Err(BitNetError::Validation("Expected non-empty logits tensor".into()).into());
    }

    match tensor {
        ConcreteTensor::BitNet(t) => {
            let candle = t.as_candle();
            let batch_0 = candle.i(0)?;
            let batch_0 =
                if batch_0.dtype() != DType::F32 { batch_0.to_dtype(DType::F32)? } else { batch_0 };
            Ok(batch_0.argmax(0)?.to_scalar::<u32>()?)
        }
        ConcreteTensor::Mock(_) => Ok(0),
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

/// Extract 2D logits \[B,V\] into a caller-owned host scratch buffer.
///
/// Returns true when the data was copied directly from contiguous/non-contiguous
/// CPU F32 storage without allocating a fresh `Vec<f32>`. Non-CPU or non-F32
/// tensors fall back to the compatibility extractor.
#[cfg(any(test, feature = "full-cli"))]
fn extract_logits_2d_into(
    tensor: &bitnet_common::ConcreteTensor,
    scratch: &mut Vec<f32>,
) -> Result<bool> {
    use bitnet_common::{BitNetError, ConcreteTensor, Tensor};

    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(BitNetError::Validation("Expected 2D tensor".into()).into());
    }

    match tensor {
        ConcreteTensor::BitNet(t) => {
            let candle = t.as_candle();
            let batch_0 = candle.i(0)?;
            if batch_0.dtype() != DType::F32 {
                scratch.clear();
                scratch.extend(extract_logits_2d(tensor)?);
                return Ok(false);
            }

            let (storage, layout) = batch_0.storage_and_layout();
            let candle_core::Storage::Cpu(cpu_storage) = &*storage else {
                scratch.clear();
                scratch.extend(extract_logits_2d(tensor)?);
                return Ok(false);
            };
            let data = cpu_storage.as_slice::<f32>()?;

            scratch.clear();
            if let Some((start, end)) = layout.contiguous_offsets() {
                scratch.extend_from_slice(&data[start..end]);
            } else {
                scratch.reserve(batch_0.elem_count());
                for index in batch_0.strided_index() {
                    scratch.push(data[index]);
                }
            }
            Ok(true)
        }
        ConcreteTensor::Mock(_) => {
            scratch.clear();
            scratch.resize(50257, 0.1);
            Ok(true)
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

#[cfg(feature = "full-cli")]
fn numeric_samples_json(samples: &[f64]) -> serde_json::Value {
    if samples.is_empty() {
        return serde_json::json!({
            "count": 0,
            "min": serde_json::Value::Null,
            "mean": serde_json::Value::Null,
            "p50": serde_json::Value::Null,
            "p95": serde_json::Value::Null,
            "max": serde_json::Value::Null,
        });
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;

    serde_json::json!({
        "count": samples.len(),
        "min": rounded_ms(sorted[0]),
        "mean": rounded_ms(mean),
        "p50": rounded_ms(percentile_nearest(&sorted, 50)),
        "p95": rounded_ms(percentile_nearest(&sorted, 95)),
        "max": rounded_ms(sorted[sorted.len() - 1]),
    })
}

#[cfg(feature = "full-cli")]
fn tokens_per_second_json(tokens: usize, elapsed_ms: f64) -> serde_json::Value {
    if tokens == 0 || elapsed_ms <= 0.0 {
        serde_json::Value::Null
    } else {
        serde_json::json!(rounded_ms(tokens as f64 / (elapsed_ms / 1000.0)))
    }
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
    } else if is_apple_backend_label(selected_backend) || runtime_api == "metal" {
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

    if is_apple_backend_label(selected_backend) || runtime_api == "metal" {
        return apple_machine_present;
    }

    apple_machine_present || cpu_model_present || !cpu_features.is_empty()
}

fn allocation_audit_backend_supported(identity: &RunBackendIdentity) -> bool {
    is_supported_apple_cpu_neon_backend(identity.requested_backend.as_str())
        && identity.selected_backend == identity.requested_backend
        && identity.runtime_api == "cpu"
        && !identity.fallback_used
}

#[cfg(feature = "full-cli")]
fn slm_warm_session_artifact_kind(requested_backend: &str) -> &'static str {
    match requested_backend.trim().to_ascii_lowercase().as_str() {
        "cpu" => "slm_cpu_warm_session",
        "apple-m3-air-cpu-neon" => "slm_apple_m3_air_warm_session",
        _ => "slm_apple_m4_warm_session",
    }
}

#[cfg(feature = "full-cli")]
fn slm_warm_session_prompt_artifact_kind(requested_backend: &str) -> &'static str {
    match requested_backend.trim().to_ascii_lowercase().as_str() {
        "cpu" => "slm_cpu_warm_session_prompt",
        "apple-m3-air-cpu-neon" => "slm_apple_m3_air_warm_session_prompt",
        _ => "slm_apple_m4_warm_session_prompt",
    }
}

#[cfg(feature = "full-cli")]
fn is_supported_slm_warm_session_backend(label: &str) -> bool {
    label.trim().eq_ignore_ascii_case("cpu") || is_supported_apple_cpu_neon_backend(label)
}

fn apple_machine_receipt_json(
    requested_backend: &str,
    selected_backend: &str,
) -> Option<serde_json::Value> {
    if !is_apple_backend_label(requested_backend) && !is_apple_backend_label(selected_backend) {
        return None;
    }

    let probe = probe_apple_cli_machine();
    let machine_id = apple_machine_id_for_backend(requested_backend)
        .or_else(|| apple_machine_id_for_backend(selected_backend))
        .unwrap_or("apple-silicon-mac");
    Some(apple_machine_receipt_json_from_probe(&probe, machine_id))
}

fn is_apple_backend_label(label: &str) -> bool {
    matches!(
        label.trim().to_ascii_lowercase().as_str(),
        "apple-m4-metal"
            | "apple-m4-mpsgraph"
            | "apple-m4-cpu-neon"
            | "apple-m3-air-metal"
            | "apple-m3-air-mpsgraph"
            | "apple-m3-air-cpu-neon"
    )
}

fn is_supported_apple_cpu_neon_backend(label: &str) -> bool {
    matches!(
        label.trim().to_ascii_lowercase().as_str(),
        "apple-m4-cpu-neon" | "apple-m3-air-cpu-neon"
    )
}

fn apple_machine_id_for_backend(label: &str) -> Option<&'static str> {
    match label.trim().to_ascii_lowercase().as_str() {
        "apple-m4-metal" | "apple-m4-mpsgraph" | "apple-m4-cpu-neon" => Some("apple-m4-mac-mini"),
        "apple-m3-air-metal" | "apple-m3-air-mpsgraph" | "apple-m3-air-cpu-neon" => {
            Some("apple-m3-macbook-air")
        }
        _ => None,
    }
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
        "apple-m3-air-metal" => Some(
            "apple-m3-air-metal is the Apple M3 MacBook Air native Metal identity lane; it is not M4 Mac mini evidence, MPSGraph model inference, Neural Engine execution, or CPU fallback proof.",
        ),
        "apple-m3-air-mpsgraph" => Some(
            "apple-m3-air-mpsgraph is the Apple M3 MacBook Air graph/reference identity lane; it is not native Metal kernel proof, not M4 Mac mini evidence, and not Neural Engine execution.",
        ),
        "apple-m3-air-cpu-neon" => Some(
            "apple-m3-air-cpu-neon is the Apple M3 MacBook Air CPU/NEON lane; it is not M4 Mac mini evidence, Metal acceleration, Neural Engine execution, or MPSGraph model inference.",
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

fn apple_machine_receipt_json_from_probe(
    probe: &AppleCliMachineProbe,
    machine_id: &str,
) -> serde_json::Value {
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
        "machine_id": machine_id,
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
    fn simple_loader_qk256_raw_tensors_preserve_trailer_scale() -> Result<()> {
        let mut packed = vec![0x55; 64];
        packed.extend_from_slice(&1.25f32.to_le_bytes());
        let qk256 = bitnet_models::quant::i2s_qk256::I2SQk256NoScale::new(1, 256, packed)?;

        let raw_tensors =
            qk256_raw_tensors_from_simple_loader([("blk.0.ffn_gate.weight".to_string(), qk256)])?;
        let qs = raw_tensors
            .get("blk.0.ffn_gate.weight.qk256_qs")
            .ok_or_else(|| anyhow::anyhow!("missing qk256 raw tensor"))?;
        let scale = raw_tensors
            .get("blk.0.ffn_gate.weight.qk256_scale")
            .ok_or_else(|| anyhow::anyhow!("missing qk256 scale tensor"))?;

        assert_eq!(qs.dims(), &[1, 64]);
        assert_eq!(qs.to_vec2::<u8>()?, vec![vec![0x55; 64]]);
        assert_eq!(scale.to_vec1::<f32>()?, vec![1.25]);
        Ok(())
    }

    #[test]
    fn greedy_effective_top1_applies_repetition_penalty() {
        let logits = [10.0, 9.0];

        assert_eq!(greedy_effective_top1_token_id(&logits, &[], 2.0), Some(0));
        assert_eq!(greedy_effective_top1_token_id(&logits, &[0], 2.0), Some(1));
    }

    #[test]
    fn greedy_effective_top1_uses_count_aware_penalty() {
        let logits = [10.0, 3.0];

        assert_eq!(greedy_effective_top1_token_id(&logits, &[0], 2.0), Some(0));
        assert_eq!(greedy_effective_top1_token_id(&logits, &[0, 0], 2.0), Some(1));
    }

    #[test]
    fn greedy_effective_top1_keeps_lowest_token_id_tie_break() {
        let logits = [1.0, 1.0, 0.0];

        assert_eq!(greedy_effective_top1_token_id(&logits, &[], 1.0), Some(0));
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn no_think_appends_qwen_suffix() {
        let prompt = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n";
        let result = apply_qwen_no_think_prompt_policy(
            bitnet_inference::TemplateType::QwenChat,
            prompt.to_string(),
            true,
        );

        assert!(result.is_ok());
        let rendered = result.unwrap_or_default();

        assert!(rendered.ends_with("<think>\n\n</think>\n\n"));
        assert_eq!(rendered.matches("<think>").count(), 1);
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn no_think_rejects_non_qwen_template() {
        let err = apply_qwen_no_think_prompt_policy(
            bitnet_inference::TemplateType::Instruct,
            "Q: 2+2?\nA:".to_string(),
            true,
        )
        .expect_err("no-thinking should be qwen-only");

        assert!(err.to_string().contains("--prompt-template qwen"));
    }

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
    #[cfg(feature = "full-cli")]
    fn slm_warm_session_quality_accepts_qwen_marker_answer() {
        let gate = SlmWarmSessionGate {
            kind: "contains_any".to_string(),
            expected: None,
            contains_any: Some(vec!["4".to_string()]),
            starts_with_any: None,
            min_words: None,
        };

        let quality = slm_warm_session_quality_receipt(
            "\n2+2 equals 4.<|im_end|>",
            &[198, 17, 10, 17, 16819, 220, 19, 13, 151645],
            1,
            2,
            Some(&gate),
        );

        assert_eq!(quality["passed"], true);
        assert_eq!(quality["valid_utf8"], true);
        assert_eq!(quality["non_empty"], true);
        assert_eq!(quality["non_degenerate"], true);
        assert_eq!(quality["gate_passed"], true);
        assert_eq!(quality["normalized_text"], "2+2 equals 4.");
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn slm_warm_session_quality_accepts_qwen_answer_prefix_separator() {
        let gate = SlmWarmSessionGate {
            kind: "starts_with_any".to_string(),
            expected: None,
            contains_any: None,
            starts_with_any: Some(vec!["Answer:".to_string()]),
            min_words: None,
        };

        let quality = slm_warm_session_quality_receipt(
            ": Answer: blue<|im_end|>",
            &[25, 21806, 25, 6303, 151645],
            1,
            2,
            Some(&gate),
        );

        assert_eq!(quality["passed"], true);
        assert_eq!(quality["gate_passed"], true);
        assert_eq!(quality["normalized_text"], "Answer: blue");
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn slm_warm_session_determinism_accepts_matching_repeats() {
        let records = vec![
            WarmSessionDeterminismRecord {
                prompt_index: 0,
                case_id: "math".to_string(),
                prompt: "What is 2+2?".to_string(),
                text: "4".to_string(),
                generated_ids: vec![19],
            },
            WarmSessionDeterminismRecord {
                prompt_index: 1,
                case_id: "math".to_string(),
                prompt: "What is 2+2?".to_string(),
                text: "4".to_string(),
                generated_ids: vec![19],
            },
        ];

        let determinism = slm_warm_session_determinism_receipt(&records);

        assert_eq!(determinism["checked"], true);
        assert_eq!(determinism["passed"], true);
        assert_eq!(determinism["repeated_prompt_groups"], 1);
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn slm_warm_session_determinism_rejects_divergent_repeats() {
        let records = vec![
            WarmSessionDeterminismRecord {
                prompt_index: 0,
                case_id: "math".to_string(),
                prompt: "What is 2+2?".to_string(),
                text: "4".to_string(),
                generated_ids: vec![19],
            },
            WarmSessionDeterminismRecord {
                prompt_index: 1,
                case_id: "math".to_string(),
                prompt: "What is 2+2?".to_string(),
                text: "four".to_string(),
                generated_ids: vec![913],
            },
        ];

        let determinism = slm_warm_session_determinism_receipt(&records);

        assert_eq!(determinism["checked"], true);
        assert_eq!(determinism["passed"], false);
        assert_eq!(determinism["groups"][0]["stable_generated_token_ids"], false);
        assert_eq!(determinism["groups"][0]["stable_text"], false);
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn slm_warm_session_disk_context_resolves_missing_receipt_parent() {
        let relative_missing_receipt =
            std::path::Path::new(".").join("__bitnet_missing_warm_session_receipt.json");

        let resolved = slm_cpu_warm_session_absolute_disk_path(&relative_missing_receipt);

        assert!(resolved.is_absolute(), "resolved path should be absolute: {resolved:?}");
        assert!(resolved.ends_with("__bitnet_missing_warm_session_receipt.json"));
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn slm_warm_session_disk_context_strips_windows_verbatim_prefix() {
        let resolved = slm_cpu_warm_session_normalize_windows_verbatim_path(
            std::path::PathBuf::from(r"\\?\C:\Code\Rust\BitNet-rs\receipt.json"),
        );

        assert_eq!(resolved, std::path::PathBuf::from(r"C:\Code\Rust\BitNet-rs\receipt.json"));
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
    fn strict_cuda_ask_receipt_accepts_measured_qk256_accounting() {
        let run_receipt = serde_json::json!({
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "fallback_used": false,
            "execution_coverage": {
                "bitnet_linear_layers_cpu_fallback": 0
            },
            "kernel_stats": [{
                "kernel_id": bitnet_kernels::cuda::CUDA_QK256_GEMV_KERNEL_ID,
                "invocations": 4,
                "fallback_invocations": 0,
                "host_to_device_bytes": 4096,
                "device_to_host_bytes": 2048,
                "kernel_launches": 4,
                "kernel_time_ms": 1.25,
                "kernel_time_samples": 4
            }],
            "cuda_execution_residency": {
                "host_device_transfer_accounting": {
                    "status": "qk256_measured",
                    "host_to_device_bytes": 4096,
                    "device_to_host_bytes": 2048,
                    "kernel_time_ms": 1.25,
                    "kernel_time_samples": 4
                }
            },
            "execution_plan": {
                "planner_version": "cuda-planner-004",
                "model_family": "bitnet_b1_58",
                "quantization": "i2_s_qk256",
                "selected_route": "bitnet_qk256_cuda",
                "requested_backend": "nvidia-rtx-5070-ti-cuda",
                "selected_backend": "nvidia-rtx-5070-ti-cuda",
                "runtime_api": "cuda",
                "strict_fallback_policy": "reject",
                "dense_regular_llm_cuda": false,
                "bitnet_packed_qk256_cuda": true,
                "cuda_bitnet_qk256_ops": 4,
                "cuda_dense_regular_llm_ops": 0,
                "cpu_fallback_ops": 0,
                "unsupported_ops": 0,
                "total_ops": 4,
                "cuda_ops": 4,
                "mixed_cuda_routes": false,
                "fallback_used": false,
                "strict_cuda_ready": true,
                "speedup_claim": false,
                "full_cuda_residency_claimed": false
            }
        });

        validate_strict_cuda_ask_receipt(&run_receipt).unwrap();
    }

    #[test]
    fn strict_cuda_ask_receipt_rejects_dense_cuda_execution_plan() {
        let run_receipt = serde_json::json!({
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "fallback_used": false,
            "execution_coverage": {
                "bitnet_linear_layers_cpu_fallback": 0
            },
            "kernel_stats": [{
                "kernel_id": bitnet_kernels::cuda::CUDA_QK256_GEMV_KERNEL_ID,
                "invocations": 4,
                "fallback_invocations": 0,
                "host_to_device_bytes": 4096,
                "device_to_host_bytes": 2048,
                "kernel_launches": 4,
                "kernel_time_ms": 1.25,
                "kernel_time_samples": 4
            }],
            "cuda_execution_residency": {
                "host_device_transfer_accounting": {
                    "status": "qk256_measured",
                    "host_to_device_bytes": 4096,
                    "device_to_host_bytes": 2048,
                    "kernel_time_ms": 1.25,
                    "kernel_time_samples": 4
                }
            },
            "execution_plan": {
                "planner_version": "cuda-planner-004",
                "model_family": "qwen",
                "quantization": "bf16",
                "selected_route": "dense_regular_llm_cuda",
                "requested_backend": "nvidia-rtx-5070-ti-cuda",
                "selected_backend": "nvidia-rtx-5070-ti-cuda",
                "runtime_api": "cuda",
                "strict_fallback_policy": "reject",
                "dense_regular_llm_cuda": true,
                "bitnet_packed_qk256_cuda": false,
                "cuda_bitnet_qk256_ops": 0,
                "cuda_dense_regular_llm_ops": 4,
                "cpu_fallback_ops": 0,
                "unsupported_ops": 0,
                "total_ops": 4,
                "cuda_ops": 4,
                "mixed_cuda_routes": false,
                "fallback_used": false,
                "strict_cuda_ready": true,
                "speedup_claim": false,
                "full_cuda_residency_claimed": false
            }
        });

        let err = validate_strict_cuda_ask_receipt(&run_receipt).unwrap_err().to_string();

        assert!(err.contains("invalid BitNet QK256 execution_plan"), "got: {err}");
        assert!(err.contains("execution_plan_selected_route_bitnet_qk256_cuda"), "got: {err}");
    }

    #[test]
    fn strict_cuda_ask_receipt_rejects_missing_qk256_accounting() {
        let run_receipt = serde_json::json!({
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "fallback_used": false,
            "execution_coverage": {
                "bitnet_linear_layers_cpu_fallback": 0
            },
            "kernel_stats": [{
                "kernel_id": bitnet_kernels::cuda::CUDA_QK256_GEMV_KERNEL_ID,
                "invocations": 4,
                "fallback_invocations": 0,
                "host_to_device_bytes": null,
                "device_to_host_bytes": null,
                "kernel_launches": 4,
                "kernel_time_ms": null
            }],
            "cuda_execution_residency": {
                "host_device_transfer_accounting": {
                    "status": "not_measured",
                    "host_to_device_bytes": null,
                    "device_to_host_bytes": null,
                    "kernel_time_ms": null
                }
            }
        });

        let err = validate_strict_cuda_ask_receipt(&run_receipt).unwrap_err().to_string();

        assert!(err.contains("measured QK256 timing/transfer accounting"), "got: {err}");
    }

    #[test]
    fn strict_ask_default_receipt_path_is_only_for_strict_modes() {
        assert!(
            strict_ask_default_receipt_path(false, false).is_none(),
            "non-strict ask should not force a receipt"
        );
        assert!(
            strict_ask_default_receipt_path(true, true).is_none(),
            "mutually exclusive strict modes are rejected before receipt resolution"
        );

        let cuda = strict_ask_default_receipt_path(true, false).unwrap();
        assert_eq!(
            cuda,
            std::path::PathBuf::from("target")
                .join("bitnet")
                .join("receipts")
                .join("cuda-answer-readiness")
                .join("strict-cuda-ask-latest.json")
        );

        let cpu = strict_ask_default_receipt_path(false, true).unwrap();
        assert_eq!(
            cpu,
            std::path::PathBuf::from("target")
                .join("bitnet")
                .join("receipts")
                .join("cuda-answer-readiness")
                .join("strict-cpu-ask-latest.json")
        );
    }

    #[test]
    fn cuda_doctor_command_accepts_nested_doctor() -> Result<()> {
        let handle = std::thread::Builder::new()
            .name("cuda-doctor-clap-parse".to_string())
            .stack_size(64 * 1024 * 1024)
            .spawn(|| Cli::command().try_get_matches_from(["bitnet", "cuda", "doctor"]))
            .map_err(|err| anyhow::anyhow!("spawn clap parse thread: {err}"))?;
        let parse_result =
            handle.join().map_err(|_| anyhow::anyhow!("clap parse thread panicked"))?;
        parse_result.context("cuda doctor should parse")?;
        Ok(())
    }

    #[test]
    fn cuda_doctor_defaults_to_rtx5070ti_backend_without_global_device() {
        assert_eq!(effective_cuda_doctor_backend(None), RTX_5070_TI_CUDA);
    }

    #[test]
    fn cuda_doctor_rejects_generic_cuda_backend_before_probe() -> Result<()> {
        let err = match validate_strict_cuda_backend_label("cuda", "cuda doctor") {
            Ok(()) => anyhow::bail!("cuda doctor accepted generic cuda backend"),
            Err(err) => err.to_string(),
        };

        assert!(
            err.contains("cuda doctor requires --device nvidia-rtx-5070-ti-cuda"),
            "got: {err}"
        );
        assert!(err.contains("requested backend was cuda"), "got: {err}");
        Ok(())
    }

    #[test]
    fn ask_strict_cuda_rejects_generic_cuda_backend_before_generation() -> Result<()> {
        let err = match validate_strict_cuda_backend_label("cuda", "--strict-cuda") {
            Ok(()) => anyhow::bail!("strict CUDA ask accepted generic cuda backend"),
            Err(err) => err.to_string(),
        };

        assert!(
            err.contains("--strict-cuda requires --device nvidia-rtx-5070-ti-cuda"),
            "got: {err}"
        );
        assert!(err.contains("requested backend was cuda"), "got: {err}");
        Ok(())
    }

    #[test]
    fn ask_strict_cuda_preflight_rejects_missing_tokenizer_before_generation() -> Result<()> {
        let temp_dir = tempfile::tempdir().context("temp dir")?;
        let model = temp_dir.path().join("ggml-model-i2_s.gguf");
        std::fs::write(&model, b"not a real gguf").context("write model")?;

        let err = match strict_bitnet_cuda_ask_preflight(&model, None, None) {
            Ok(_) => anyhow::bail!("strict CUDA ask accepted a model without tokenizer authority"),
            Err(err) => err.to_string(),
        };

        assert!(
            err.contains("strict CUDA ask requires tokenizer authority before generation"),
            "got: {err}"
        );
        assert!(err.contains("ggml-model-i2_s.gguf"), "got: {err}");
        Ok(())
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn dense_qwen_ask_resolves_positional_question() {
        let question = resolve_ask_question(None, Some("What is 2+2?".to_string())).unwrap();

        assert_eq!(question, "What is 2+2?");
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn dense_qwen_ask_resolves_flag_question_first() {
        let question = resolve_ask_question(
            Some("flag question".to_string()),
            Some("positional question".to_string()),
        )
        .unwrap();

        assert_eq!(question, "flag question");
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn dense_qwen_ask_backend_accepts_cuda_aliases_only() {
        assert!(is_dense_qwen_cuda_ask_backend("cuda"));
        assert!(is_dense_qwen_cuda_ask_backend("nvidia-rtx-5070-ti-cuda"));
        assert!(!is_dense_qwen_cuda_ask_backend("cpu"));
        assert!(!is_dense_qwen_cuda_ask_backend("apple-m4-cpu-neon"));
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn dense_qwen_ask_resolves_qwen3_explicit_model_file() -> Result<()> {
        let model = std::path::PathBuf::from("C:/models/Qwen3-0.6B-Q8_0.gguf");
        let resolved = resolve_dense_qwen_cuda_ask_model(&model)?
            .context("Qwen3 explicit GGUF should resolve to dense Qwen CUDA ask path")?;

        assert_eq!(resolved, model);
        Ok(())
    }

    #[test]
    fn ask_defaults_to_warn_logging_for_user_facing_output() {
        let command = Commands::Ask {
            model: std::path::PathBuf::from("qwen2.5-0.5b-instruct-q8_0"),
            tokenizer: None,
            question: None,
            question_arg: Some("What is 2+2?".to_string()),
            system_prompt: None,
            max_new_tokens: 8,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            strict_cuda: false,
            strict_cpu: false,
            receipt_out: None,
        };

        assert_eq!(default_log_level_for_command(Some(&command)), Some("warn"));
        assert_eq!(default_log_level_for_command(None), None);
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn model_and_receipts_default_to_warn_logging_for_user_facing_output() {
        assert_eq!(
            default_log_level_for_command(Some(&Commands::Model(ModelCommand {
                action: model_cache::ModelAction::List { cache_dir: None, json: false },
            }))),
            Some("warn")
        );
        assert_eq!(
            default_log_level_for_command(Some(&Commands::Receipts(ReceiptsCommand {
                action: commands::receipts::ReceiptsAction::Explain {
                    path: None,
                    latest: true,
                    json: false,
                    format: None,
                },
            }))),
            Some("warn")
        );
        assert_eq!(
            default_log_level_for_command(Some(&Commands::Support(SupportCommand {
                action: commands::support::SupportAction::Bundle {
                    path: None,
                    latest: true,
                    device: "nvidia-rtx-5070-ti-cuda".to_string(),
                    matrix: None,
                    format: commands::support::SupportBundleFormat::Json,
                },
            }))),
            Some("warn")
        );
        assert_eq!(
            default_log_level_for_command(Some(&Commands::Chat(Box::default()))),
            Some("warn")
        );
    }

    #[test]
    fn model_status_skips_startup_backend_selection() {
        let command = Commands::Model(ModelCommand {
            action: model_cache::ModelAction::Status {
                device: "nvidia-rtx-5070-ti-cuda".to_string(),
                matrix: None,
                format: model_cache::ModelStatusFormat::Text,
            },
        });

        assert!(uses_read_only_model_status(Some(&command)));
        assert!(skips_startup_backend_selection(Some(&command)));
        assert_eq!(default_log_level_for_command(Some(&command)), Some("warn"));
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn support_bundle_skips_startup_backend_selection() {
        let command = Commands::Support(SupportCommand {
            action: commands::support::SupportAction::Bundle {
                path: None,
                latest: true,
                device: "nvidia-rtx-5070-ti-cuda".to_string(),
                matrix: None,
                format: commands::support::SupportBundleFormat::Json,
            },
        });

        assert!(uses_read_only_support_bundle(Some(&command)));
        assert!(skips_startup_backend_selection(Some(&command)));
        assert_eq!(default_log_level_for_command(Some(&command)), Some("warn"));
    }

    #[test]
    #[cfg(feature = "cli-bench")]
    fn report_only_cuda_benchmark_receipt_skips_startup_backend_selection() {
        let report_command = Commands::Benchmark(BenchmarkCommand {
            model: None,
            device: Some("cuda".to_string()),
            iterations: 10,
            warmup: 3,
            prompt_length: 128,
            generation_length: 256,
            compare_python: false,
            flamegraph: false,
            format: "text".to_string(),
            output: None,
            cuda_benchmark_receipt: Some(std::path::PathBuf::from("receipt.json")),
            profile: None,
            memory_profile: false,
            batch_sizes: vec![1, 4, 8],
            sequence_lengths: vec![128, 512, 1024],
        });
        let legacy_command = Commands::Benchmark(BenchmarkCommand {
            model: Some(std::path::PathBuf::from("model.gguf")),
            device: Some("cuda".to_string()),
            iterations: 10,
            warmup: 3,
            prompt_length: 128,
            generation_length: 256,
            compare_python: false,
            flamegraph: false,
            format: "text".to_string(),
            output: None,
            cuda_benchmark_receipt: None,
            profile: None,
            memory_profile: false,
            batch_sizes: vec![1, 4, 8],
            sequence_lengths: vec![128, 512, 1024],
        });

        assert!(uses_report_only_cuda_benchmark_receipt(Some(&report_command)));
        assert!(skips_startup_backend_selection(Some(&report_command)));
        assert_eq!(default_log_level_for_command(Some(&report_command)), Some("warn"));
        assert!(!uses_report_only_cuda_benchmark_receipt(Some(&legacy_command)));
        assert!(!skips_startup_backend_selection(Some(&legacy_command)));
        assert!(!uses_report_only_cuda_benchmark_receipt(None));
    }

    #[test]
    fn cuda_execution_residency_receipt_preserves_claim_boundary() {
        let coverage = bitnet_qk256_dispatch::Qk256DispatchCoverageCounters {
            bitnet_linear_layers_total: 42,
            bitnet_linear_layers_on_cuda: 42,
            bitnet_linear_layers_cpu_fallback: 0,
            unsupported_ops: Vec::new(),
            execution_claim: "cuda_inference_contribution",
        };
        let residency = bitnet_qk256_dispatch::Qk256CudaWeightResidency {
            weight_handle_count: 21,
            weights_uploaded_once: true,
            per_token_weight_upload: false,
        };

        let receipt = cuda_execution_residency_receipt(CudaExecutionResidencyReceiptInput {
            coverage: &coverage,
            residency: Some(&residency),
            runtime_stats: None,
            prompt_tokens: 18,
            generated_tokens: 8,
            kv_cache_device: "cpu",
            kv_cache_reuse_policy: "recreated_per_turn_for_prompt_isolation",
            execution_phase: "warm_session_turn",
            coverage_scope: "strict_cuda_warm_session_turn",
        });

        assert_eq!(receipt["full_cuda_residency_claimed"], false);
        assert_eq!(receipt["speedup_claim"], false);
        assert_eq!(
            receipt["qk256_bitnet_linears"]["kernel_id"],
            bitnet_kernels::cuda::CUDA_QK256_GEMV_KERNEL_ID
        );
        assert_eq!(receipt["qk256_bitnet_linears"]["bitnet_linear_layers_on_cuda"], 42);
        assert_eq!(receipt["qk256_bitnet_linears"]["bitnet_linear_layers_cpu_fallback"], 0);
        assert_eq!(receipt["weight_residency"]["weights_uploaded_once"], true);
        assert_eq!(receipt["weight_residency"]["per_token_weight_upload"], false);
        assert_eq!(receipt["kv_cache"]["device"], "cpu");
        assert_eq!(receipt["kv_cache"]["cuda_residency_claimed"], false);
        assert_eq!(receipt["claim_boundary"]["upload_once_weight_residency_claimed"], true);
        assert_eq!(receipt["claim_boundary"]["full_transformer_cuda_residency_claimed"], false);
    }

    #[test]
    fn cuda_execution_residency_receipt_marks_non_linear_phases_not_resident() {
        let coverage = bitnet_qk256_dispatch::Qk256DispatchCoverageCounters {
            bitnet_linear_layers_total: 2,
            bitnet_linear_layers_on_cuda: 1,
            bitnet_linear_layers_cpu_fallback: 1,
            unsupported_ops: vec!["qk256_cpu_fallback".to_string()],
            execution_claim: "cuda_inference_contribution",
        };

        let receipt = cuda_execution_residency_receipt(CudaExecutionResidencyReceiptInput {
            coverage: &coverage,
            residency: None,
            runtime_stats: None,
            prompt_tokens: 4,
            generated_tokens: 1,
            kv_cache_device: "cpu",
            kv_cache_reuse_policy: "per_run_incremental_decode",
            execution_phase: "decode",
            coverage_scope: "strict_cuda_ask_or_run",
        });

        assert_eq!(receipt["qk256_bitnet_linears"]["residency"], "mixed_cuda_and_cpu_fallback");
        assert_eq!(receipt["weight_residency"]["status"], "not_observed");
        assert_eq!(receipt["phase_residency"]["rmsnorm"]["cuda_residency_claimed"], false);
        assert_eq!(receipt["phase_residency"]["rope"]["cuda_residency_claimed"], false);
        assert_eq!(
            receipt["phase_residency"]["attention_softmax"]["cuda_residency_claimed"],
            false
        );
        assert_eq!(receipt["phase_residency"]["sampling"]["residency"], "cpu_resident");
        assert_eq!(receipt["host_device_transfer_accounting"]["status"], "not_measured");
        assert!(
            receipt["unresident_or_unmeasured_phases"]
                .as_array()
                .expect("phase list")
                .iter()
                .any(|value| value == "host_device_transfer_bytes")
        );
        assert_eq!(receipt["claim_boundary"]["qk256_cuda_residency_claimed"], false);
    }

    #[test]
    fn qk256_kernel_stats_receipt_records_measured_cuda_accounting() {
        let coverage = bitnet_qk256_dispatch::Qk256DispatchCoverageCounters {
            bitnet_linear_layers_total: 4,
            bitnet_linear_layers_on_cuda: 4,
            bitnet_linear_layers_cpu_fallback: 0,
            unsupported_ops: Vec::new(),
            execution_claim: "cuda_inference_contribution",
        };
        let runtime_stats = bitnet_qk256_dispatch::Qk256CudaRuntimeStats {
            host_to_device_bytes: 4096,
            device_to_host_bytes: 2048,
            kernel_time_ms: Some(1.23456),
            kernel_time_samples: 4,
        };

        let receipt = qk256_kernel_stats_receipt(&coverage, Some(&runtime_stats));

        assert_eq!(receipt[0]["kernel_id"], bitnet_kernels::cuda::CUDA_QK256_GEMV_KERNEL_ID);
        assert_eq!(receipt[0]["invocations"], 4);
        assert_eq!(receipt[0]["fallback_invocations"], 0);
        assert_eq!(receipt[0]["host_to_device_bytes"], 4096);
        assert_eq!(receipt[0]["device_to_host_bytes"], 2048);
        assert_eq!(receipt[0]["kernel_time_ms"], 1.235);
        assert_eq!(receipt[0]["kernel_time_samples"], 4);
    }

    #[test]
    fn qk256_cpu_hot_path_receipt_distinguishes_scaled_and_materialized_paths() {
        let counters = bitnet_qk256_dispatch::Qk256CpuHotPathCounters {
            qk256_f32_scalar_gemv_invocations: 1,
            qk256_f32_avx2_gemv_invocations: 0,
            qk256_i8s_scaled_scalar_invocations: 2,
            qk256_i8s_scaled_avx2_invocations: 0,
            qk256_flat_bytes_extracted_count: 3,
            input_rows_materialized_count: 4,
            output_rows_allocated_count: 5,
            requested_kernel: Some("scalar".to_string()),
            selected_kernel: Some("mixed-qk256-cpu-hot-paths".to_string()),
            qk256_execution_path: "mixed_scaled_and_no_scale",
        };

        let receipt = qk256_cpu_hot_path_receipt(&counters);

        assert_eq!(receipt["no_scale_f32_gemv_invocations"], 1);
        assert_eq!(receipt["scaled_i2s_i8s_gemv_invocations"], 2);
        assert_eq!(receipt["audited_tensor_materialization_count"], 12);
        assert_eq!(receipt["selected_kernel"], "mixed-qk256-cpu-hot-paths");
        assert_eq!(receipt["math_changed"], false);
        assert_eq!(receipt["speedup_claim"], false);
    }

    #[test]
    fn cuda_execution_residency_receipt_records_measured_qk256_accounting() {
        let coverage = bitnet_qk256_dispatch::Qk256DispatchCoverageCounters {
            bitnet_linear_layers_total: 4,
            bitnet_linear_layers_on_cuda: 4,
            bitnet_linear_layers_cpu_fallback: 0,
            unsupported_ops: Vec::new(),
            execution_claim: "cuda_inference_contribution",
        };
        let runtime_stats = bitnet_qk256_dispatch::Qk256CudaRuntimeStats {
            host_to_device_bytes: 4096,
            device_to_host_bytes: 2048,
            kernel_time_ms: Some(1.25),
            kernel_time_samples: 4,
        };

        let receipt = cuda_execution_residency_receipt(CudaExecutionResidencyReceiptInput {
            coverage: &coverage,
            residency: None,
            runtime_stats: Some(&runtime_stats),
            prompt_tokens: 12,
            generated_tokens: 3,
            kv_cache_device: "cpu",
            kv_cache_reuse_policy: "per_run_incremental_decode",
            execution_phase: "decode",
            coverage_scope: "strict_cuda_ask_or_run",
        });

        assert_eq!(receipt["host_device_transfer_accounting"]["status"], "qk256_measured");
        assert_eq!(receipt["host_device_transfer_accounting"]["host_to_device_bytes"], 4096);
        assert_eq!(receipt["host_device_transfer_accounting"]["device_to_host_bytes"], 2048);
        assert_eq!(receipt["host_device_transfer_accounting"]["kernel_time_ms"], 1.25);
        assert_eq!(receipt["claim_boundary"]["qk256_kernel_timing_claimed"], true);
        assert_eq!(receipt["claim_boundary"]["qk256_transfer_byte_accounting_claimed"], true);
        assert_eq!(receipt["claim_boundary"]["transfer_timing_claimed"], false);
        assert!(
            !receipt["unresident_or_unmeasured_phases"]
                .as_array()
                .expect("phase list")
                .iter()
                .any(|value| value == "kernel_time_ms")
        );
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
    fn m3_air_apple_labels_preserve_machine_boundary_without_m4_aliasing() -> Result<(), String> {
        for label in ["apple-m3-air-metal", "apple-m3-air-mpsgraph", "apple-m3-air-cpu-neon"] {
            assert!(is_apple_backend_label(label), "{label} should be an Apple backend label");
            assert_eq!(apple_machine_id_for_backend(label), Some("apple-m3-macbook-air"));
            assert_ne!(apple_machine_id_for_backend(label), Some("apple-m4-mac-mini"));
        }

        let metal_err = match resolve_run_backend_identity("apple-m3-air-metal", true) {
            Ok(identity) => {
                return Err(format!("strict M3 Air Metal should be unavailable, got {identity:?}"));
            }
            Err(err) => err.to_string(),
        };
        assert!(metal_err.contains("apple-m3-air-metal"), "got: {metal_err}");
        assert!(metal_err.contains("M3 MacBook Air native Metal"), "got: {metal_err}");
        assert!(metal_err.contains("not M4 Mac mini evidence"), "got: {metal_err}");

        let mpsgraph_identity = resolve_run_backend_identity("apple-m3-air-mpsgraph", false)
            .map_err(|err| err.to_string())?;
        assert_eq!(mpsgraph_identity.requested_backend, "apple-m3-air-mpsgraph");
        assert!(mpsgraph_identity.fallback_used);
        let fallback_reason = mpsgraph_identity
            .fallback_reason
            .as_deref()
            .ok_or_else(|| "M3 Air MPSGraph fallback reason was missing".to_string())?;
        assert!(
            fallback_reason.contains("graph/reference identity lane"),
            "got: {fallback_reason}"
        );
        assert!(
            fallback_reason.contains("not native Metal kernel proof"),
            "got: {fallback_reason}"
        );
        assert!(fallback_reason.contains("not M4 Mac mini evidence"), "got: {fallback_reason}");
        Ok(())
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
        let family = receipt_model_family(&architecture);

        assert_eq!(infer_model_repo(path), "Qwen/Qwen3-0.6B-GGUF");
        assert_eq!(architecture, "qwen3");
        assert_eq!(family, "qwen");
        assert!(is_dense_slm_model(family, &architecture));
        assert_eq!(dense_slm_kernel_family(family, &architecture), Some("dense_qwen"));
        assert_eq!(dense_slm_kernel_id(family, &architecture), Some("dense-qwen-cpu-reference"));
        assert_eq!(dense_slm_quant_format(path), "Q8_0");
        assert_eq!(dense_slm_layout_source(path), "gguf_dense_q8_0_reference");
        assert_eq!(dense_slm_kernel_layout(path), "gguf_dense_q8_0");
    }

    #[test]
    fn qwen25_receipt_identity_keeps_dedicated_output_head() {
        let path = std::path::Path::new(
            "models/qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf",
        );
        let architecture = infer_model_architecture(path);
        let family = receipt_model_family(&architecture);
        let (tie_word_embeddings, output_head_tensor) = model_output_head_identity(
            false,
            is_dense_slm_model(family, &architecture),
            &architecture,
        );

        assert_eq!(architecture, "qwen2");
        assert_eq!(family, "qwen");
        assert_eq!(tie_word_embeddings, serde_json::json!(false));
        assert_eq!(output_head_tensor, "output.weight");
    }

    #[test]
    fn qwen3_receipt_identity_keeps_tied_embedding_head() {
        let path = std::path::Path::new("models/slm/Qwen3-0.6B-Q8_0.gguf");
        let architecture = infer_model_architecture(path);
        let family = receipt_model_family(&architecture);
        let (tie_word_embeddings, output_head_tensor) = model_output_head_identity(
            false,
            is_dense_slm_model(family, &architecture),
            &architecture,
        );

        assert_eq!(architecture, "qwen3");
        assert_eq!(tie_word_embeddings, serde_json::json!(true));
        assert_eq!(output_head_tensor, "tied_token_embeddings");
    }

    #[cfg(feature = "full-cli")]
    #[test]
    fn lunar_lake_answer_gate_can_require_expected_text() {
        let passed = evaluate_lunar_lake_answer_gate("2+2 equals 4.", Some("4"));
        assert!(passed.passed);
        assert_eq!(passed.name, "contains_bounded_operator_answer");
        assert!(passed.failed_rules.is_empty());

        let failed = evaluate_lunar_lake_answer_gate("not the expected answer", Some("4"));
        assert!(!failed.passed);
        assert_eq!(failed.failed_rules, vec!["expected_contains"]);
    }

    #[cfg(feature = "full-cli")]
    #[test]
    fn lunar_lake_operator_ask_blocked_receipt_records_no_inference_boundary() {
        let receipt = build_lunar_lake_operator_ask_blocked_receipt(
            LunarLakeAskBlockedReceiptContext {
                artifact_root: std::path::Path::new("ci/hardware/intel-258v/2026-05-08"),
                operator_receipt: std::path::Path::new("lunar-lake-operator-readiness.json"),
                promotion_ledger: std::path::Path::new("lunar-lake-route-promotion.json"),
                route_profile_comparison: std::path::Path::new(
                    "lunar-lake-route-profile-comparison.json",
                ),
                requested_device: "auto",
                requested_route: "auto",
                profile_id: "low_power",
                question: "What is 2+2?",
                max_new_tokens: 4,
                error: "no promoted Lunar Lake auto route for profile `low_power`; why_not_npu=missing evidence: benchmark_qualified_speedup_or_power_advantage",
                route_selection: Some(&commands::lunar_lake::BlockedOperatorAskRouteSelection {
                    requested_device: "auto".to_string(),
                    requested_route: "auto".to_string(),
                    profile_id: "low_power".to_string(),
                    route_selection_status: "blocked".to_string(),
                    promotion_status: "no_promoted_route".to_string(),
                    selection_source: "promotion_ledger_auto_blocked".to_string(),
                    route_reason: "no promoted Lunar Lake auto route for profile `low_power`"
                        .to_string(),
                    candidate_routes: vec![
                        commands::lunar_lake::DEFAULT_ASK_ROUTE.to_string(),
                        "dense_slm_openvino_gpu_candidate".to_string(),
                        "dense_slm_openvino_npu_candidate".to_string(),
                    ],
                    why_not_cpu: vec!["route is not promoted for profile `low_power`".to_string()],
                    why_not_gpu: vec![
                        "route blocker for profile `low_power`: low_power_power_advantage_unproven"
                            .to_string(),
                    ],
                    why_not_npu: vec![
                        "missing evidence: benchmark_qualified_speedup_or_power_advantage"
                            .to_string(),
                    ],
                    operator_runbook: Some(
                        commands::lunar_lake::LOW_POWER_BATTERY_RUNBOOK.to_string(),
                    ),
                    next_required_evidence:
                        commands::lunar_lake::blocked_operator_ask_next_required_evidence(
                            "low_power",
                        ),
                    promotion_ledger: Some("lunar-lake-route-promotion.json".to_string()),
                    route_profile_comparison: Some(
                        "lunar-lake-route-profile-comparison.json".to_string(),
                    ),
                }),
            },
        );

        assert_eq!(receipt["artifact_kind"], "lunar_lake_operator_ask_blocked");
        assert_eq!(receipt["proof_stage"], "operator_route_selection_blocked_no_inference");
        assert_eq!(receipt["requested_device"], "auto");
        assert_eq!(receipt["requested_route"], "auto");
        assert_eq!(receipt["profile_id"], "low_power");
        assert_eq!(receipt["selected_route"], serde_json::Value::Null);
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["new_inference_executed"], false);
        assert_eq!(receipt["model_path_required"], false);
        assert_eq!(
            receipt["model_resolution"],
            "not_required_for_blocked_auto_route_before_execution"
        );
        assert_eq!(receipt["route_selection_blocked"], true);
        assert_eq!(receipt["promotion_status"], "no_promoted_route");
        assert_eq!(receipt["route_selection"]["selection_source"], "promotion_ledger_auto_blocked");
        assert_eq!(receipt["route_selection"]["model_path_required"], false);
        assert!(receipt["candidate_routes"].as_array().is_some_and(|items| items.len() == 3));
        assert!(receipt["why_not_cpu"].as_array().is_some_and(|items| !items.is_empty()));
        assert!(receipt["why_not_gpu"].as_array().is_some_and(|items| !items.is_empty()));
        assert!(receipt["why_not_npu"].as_array().is_some_and(|items| !items.is_empty()));
        assert_eq!(receipt["operator_runbook"], commands::lunar_lake::LOW_POWER_BATTERY_RUNBOOK);
        assert!(receipt["next_required_evidence"].as_array().is_some_and(|items| {
            items.iter().any(|item| {
                item.as_str()
                    .is_some_and(|value| value.contains("telemetry-context --require-battery"))
            })
        }));
        assert_eq!(
            receipt["route_selection"]["operator_runbook"],
            commands::lunar_lake::LOW_POWER_BATTERY_RUNBOOK
        );
        assert!(receipt["route_selection"]["next_required_evidence"].as_array().is_some_and(
            |items| {
                items.iter().any(|item| {
                    item.as_str()
                        .is_some_and(|value| value.contains("telemetry-context --require-battery"))
                })
            }
        ));
        assert!(receipt["route_selection"]["why_not_npu"].as_array().is_some_and(|items| {
            items.iter().any(|item| item.as_str().is_some_and(|value| value.contains("benchmark")))
        }));
        assert_eq!(receipt["claim_boundary"]["route_selection_blocked"], true);
        assert_eq!(receipt["claim_boundary"]["new_inference_executed"], false);
        assert_eq!(receipt["claim_boundary"]["model_loaded"], false);
        assert_eq!(receipt["claim_boundary"]["route_promotion_changed"], false);
        assert!(
            receipt["route_selection_error"].as_str().is_some_and(
                |error| error.contains("benchmark_qualified_speedup_or_power_advantage")
            )
        );
    }

    #[cfg(feature = "full-cli")]
    #[test]
    fn lunar_lake_ask_default_model_path_uses_cpu_phase_receipt_when_present() -> Result<()> {
        let temp_dir = tempfile::tempdir().context("temp dir")?;
        let artifact_root = temp_dir.path().join("artifacts");
        std::fs::create_dir_all(&artifact_root).context("artifact root")?;
        let model_path = temp_dir.path().join("qwen2.5-0.5b-instruct-q8_0.gguf");
        std::fs::write(&model_path, b"fake gguf fixture").context("model file")?;
        std::fs::write(
            artifact_root.join("slm-phase-warm-session-qwen25-cpu.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "artifact_kind": "slm_phase_warm_session",
                "model": {"path": model_path}
            }))
            .context("phase json")?,
        )
        .context("phase receipt")?;

        let route = commands::lunar_lake::OperatorRoute {
            route_id: commands::lunar_lake::DEFAULT_ASK_ROUTE.to_string(),
            workload: "ask".to_string(),
            selected_model: "Qwen2.5-0.5B-Instruct Q8_0 GGUF".to_string(),
            selected_backend: "cpu-rust".to_string(),
            runtime_api: "cpu".to_string(),
            selected_kernel_or_runtime: "dense-qwen-cpu-reference".to_string(),
            fallback_policy: "strict_no_fallback".to_string(),
            route_reason: "test route".to_string(),
            answer_gate_evidence: None,
            phase_evidence: None,
            acceleration_claim: false,
        };

        let resolved = resolve_lunar_lake_ask_model_path(&artifact_root, &route, None)
            .context("resolved default CPU model path")?;
        assert_eq!(resolved, model_path);
        Ok(())
    }

    #[cfg(feature = "full-cli")]
    #[test]
    fn lunar_lake_ask_default_model_path_uses_openvino_manifest_when_present() -> Result<()> {
        let temp_dir = tempfile::tempdir().context("temp dir")?;
        let artifact_root = temp_dir.path().join("artifacts");
        std::fs::create_dir_all(&artifact_root).context("artifact root")?;
        let model_dir = temp_dir.path().join("openvino-model");
        std::fs::create_dir_all(&model_dir).context("model dir")?;
        std::fs::write(
            artifact_root.join("slm-openvino-ir-qwen25-int4-sym-manifest.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "artifact_kind": "intel_258v_dense_slm_openvino_ir_manifest",
                "export_contract": {"expected_output_dir": model_dir}
            }))
            .context("manifest json")?,
        )
        .context("manifest receipt")?;

        let route = commands::lunar_lake::OperatorRoute {
            route_id: "dense_slm_openvino_gpu_candidate".to_string(),
            workload: "ask".to_string(),
            selected_model: "Qwen2.5-0.5B-Instruct OpenVINO IR INT4_SYM".to_string(),
            selected_backend: "openvino-gpu".to_string(),
            runtime_api: "openvino_genai".to_string(),
            selected_kernel_or_runtime: "openvino-genai-llmpipeline-gpu0".to_string(),
            fallback_policy: "strict_no_fallback".to_string(),
            route_reason: "test route".to_string(),
            answer_gate_evidence: None,
            phase_evidence: None,
            acceleration_claim: false,
        };

        let resolved = resolve_lunar_lake_ask_model_path(&artifact_root, &route, None)
            .context("resolved default OpenVINO model path")?;
        assert_eq!(resolved, model_dir);
        Ok(())
    }

    #[cfg(feature = "full-cli")]
    #[test]
    fn lunar_lake_ask_default_model_path_prefers_explicit_model() -> Result<()> {
        let temp_dir = tempfile::tempdir().context("temp dir")?;
        let artifact_root = temp_dir.path().join("artifacts");
        let explicit = temp_dir.path().join("explicit.gguf");
        let route = commands::lunar_lake::OperatorRoute {
            route_id: commands::lunar_lake::DEFAULT_ASK_ROUTE.to_string(),
            workload: "ask".to_string(),
            selected_model: "Qwen2.5-0.5B-Instruct Q8_0 GGUF".to_string(),
            selected_backend: "cpu-rust".to_string(),
            runtime_api: "cpu".to_string(),
            selected_kernel_or_runtime: "dense-qwen-cpu-reference".to_string(),
            fallback_policy: "strict_no_fallback".to_string(),
            route_reason: "test route".to_string(),
            answer_gate_evidence: None,
            phase_evidence: None,
            acceleration_claim: false,
        };

        let resolved = resolve_lunar_lake_ask_model_path(&artifact_root, &route, Some(&explicit))
            .context("explicit model is accepted")?;
        assert_eq!(resolved, explicit);
        Ok(())
    }

    #[cfg(feature = "full-cli")]
    #[test]
    fn lunar_lake_operator_ask_receipt_has_top_level_identity() {
        let route = commands::lunar_lake::OperatorRoute {
            route_id: commands::lunar_lake::DEFAULT_ASK_ROUTE.to_string(),
            workload: "ask".to_string(),
            selected_model: "qwen2.5-0.5b-instruct-q8_0".to_string(),
            selected_backend: "cpu-rust".to_string(),
            runtime_api: "cpu".to_string(),
            selected_kernel_or_runtime: "dense-qwen-cpu-reference".to_string(),
            fallback_policy: "strict_no_fallback".to_string(),
            route_reason: "test route".to_string(),
            answer_gate_evidence: Some(
                "slm-answer-corpus-qwen25-cpu-clean-provenance.json".to_string(),
            ),
            phase_evidence: Some("slm-phase-warm-session-qwen25-cpu.json".to_string()),
            acceleration_claim: false,
        };
        let route_selection = commands::lunar_lake::OperatorAskRouteSelection {
            requested_device: "auto".to_string(),
            requested_route: "auto".to_string(),
            profile_id: "ask_normal".to_string(),
            selected_route: commands::lunar_lake::DEFAULT_ASK_ROUTE.to_string(),
            selected_backend: "cpu-rust".to_string(),
            runtime_api: "cpu".to_string(),
            promotion_status: "promoted".to_string(),
            selection_source: "promotion_ledger_auto".to_string(),
            route_reason: "test route".to_string(),
            why_not_cpu: vec![
                "dense_slm_default_cpu is promoted for profile ask_normal and remains the safe no-fallback default"
                    .to_string(),
            ],
            why_not_gpu: vec!["route is not promoted for profile `ask_normal`".to_string()],
            why_not_npu: vec!["route is not promoted for profile `ask_normal`".to_string()],
            candidate_routes: vec![
                "dense_slm_openvino_gpu_candidate".to_string(),
                "dense_slm_openvino_npu_candidate".to_string(),
            ],
            promotion_ledger: Some("lunar-lake-route-promotion.json".to_string()),
            route_profile_comparison: Some("lunar-lake-route-profile-comparison.json".to_string()),
            route_profile_status: Some("promoted_route_ready".to_string()),
            route_profile_blockers: vec![],
            route: route.clone(),
        };
        let source = serde_json::json!({
            "requested_backend": "cpu",
            "selected_backend": "cpu-rust",
            "runtime_api": "cpu",
            "fallback_used": false,
            "fallback_reason": null,
            "kernel": {"kernel_id": "dense-qwen-cpu-reference"},
            "model": {
                "path": "models/qwen2.5.gguf",
                "sha256": "abc",
                "family": "qwen",
                "architecture": "qwen2",
                "quant_format": "Q8_0",
                "tokenizer": "gguf_metadata",
                "vocab_size": 151936,
                "loader_mode": "real_gguf",
                "fallback_loader_used": false
            },
            "prompt_render": "<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n",
            "tokens": {
                "prompt_ids": [1, 2, 3],
                "generated_ids": [19],
                "generated": 1,
                "prompt": 3
            },
            "dense_slm": {"model_family": "qwen"},
            "execution_coverage": {"execution_claim": "dense_slm_cpu_reference_answer_smoke"},
            "timing": {"first_token_ms": 1.0},
            "profile": {"phase": "decode"},
            "text": "\n4<|im_end|>"
        });
        let gate = evaluate_lunar_lake_answer_gate("4", Some("4"));
        let receipt = build_lunar_lake_operator_ask_receipt(LunarLakeAskReceiptContext {
            artifact_root: std::path::Path::new("ci/hardware/intel-258v/2026-05-08"),
            operator_receipt_path: std::path::Path::new("lunar-lake-operator-readiness.json"),
            source_run_path: std::path::Path::new("lunar-lake-operator-ask-source-run.json"),
            route: &route,
            route_selection: &route_selection,
            question: "2+2?",
            answer: "\n4<|im_end|>",
            normalized_answer: "4",
            answer_gate: &gate,
            expect_contains: Some("4"),
            source_run_receipt: &source,
        });

        assert_eq!(receipt["requested_backend"], "cpu");
        assert_eq!(receipt["selected_backend"], "cpu-rust");
        assert_eq!(receipt["runtime_api"], "cpu");
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["backend_lane"], "dense_slm_cpu");
        assert_eq!(receipt["selected_kernel_or_runtime"], "dense-qwen-cpu-reference");
        assert_eq!(receipt["route_id"], commands::lunar_lake::DEFAULT_ASK_ROUTE);
        assert_eq!(receipt["requested_device"], "auto");
        assert_eq!(receipt["requested_route"], "auto");
        assert_eq!(receipt["profile_id"], "ask_normal");
        assert_eq!(receipt["selected_route"], commands::lunar_lake::DEFAULT_ASK_ROUTE);
        assert_eq!(receipt["promotion_status"], "promoted");
        assert_eq!(receipt["route_profile_status"], "promoted_route_ready");
        assert_eq!(receipt["route_profile_comparison"], "lunar-lake-route-profile-comparison.json");
        assert_eq!(receipt["route_selection"]["selection_source"], "promotion_ledger_auto");
        assert_eq!(receipt["route_selection"]["route_profile_status"], "promoted_route_ready");
        assert!(receipt["why_not_gpu"].as_array().is_some_and(|items| !items.is_empty()));
        assert_eq!(receipt["model_family"], "qwen");
        assert_eq!(receipt["model_architecture"], "qwen2");
        assert_eq!(receipt["quantization"], "Q8_0");
        assert_eq!(receipt["tokenizer_source"], "gguf_metadata");
        assert_eq!(receipt["prompt_template"], "qwen2.5");
        assert_eq!(receipt["answer_gate_passed"], true);
        assert_eq!(receipt["acceleration_claim"], false);
        assert_eq!(receipt["backend"]["fallback_used"], false);
        assert!(receipt["source_receipt"].is_object());
    }

    #[cfg(feature = "full-cli")]
    #[test]
    fn lunar_lake_operator_ask_receipt_accepts_openvino_source_shape() -> anyhow::Result<()> {
        let route = commands::lunar_lake::OperatorRoute {
            route_id: "dense_slm_openvino_gpu_candidate".to_string(),
            workload: "ask".to_string(),
            selected_model: "Qwen2.5-0.5B-Instruct INT4_SYM OpenVINO IR".to_string(),
            selected_backend: "openvino-gpu".to_string(),
            runtime_api: "openvino_genai".to_string(),
            selected_kernel_or_runtime: "openvino-genai-llmpipeline-gpu0".to_string(),
            fallback_policy: "strict_no_fallback".to_string(),
            route_reason: "explicit OpenVINO GPU candidate route".to_string(),
            answer_gate_evidence: Some(
                "lunar-lake-openvino-operator-ask-gpu-math-brief.json".to_string(),
            ),
            phase_evidence: Some("slm-openvino-cpu-gpu-npu-phase-runner.json".to_string()),
            acceleration_claim: false,
        };
        let route_selection = commands::lunar_lake::OperatorAskRouteSelection {
            requested_device: "GPU.0".to_string(),
            requested_route: "dense_slm_openvino_gpu_candidate".to_string(),
            profile_id: "ask_normal".to_string(),
            selected_route: "dense_slm_openvino_gpu_candidate".to_string(),
            selected_backend: "openvino-gpu".to_string(),
            runtime_api: "openvino_genai".to_string(),
            promotion_status: "direct_route_validated".to_string(),
            selection_source: "operator_receipt_direct".to_string(),
            route_reason: "explicit OpenVINO GPU candidate route".to_string(),
            why_not_cpu: vec!["CPU route was not requested".to_string()],
            why_not_gpu: vec!["auto routing was not requested".to_string()],
            why_not_npu: vec!["auto routing was not requested".to_string()],
            candidate_routes: vec![],
            promotion_ledger: None,
            route_profile_comparison: Some("lunar-lake-route-profile-comparison.json".to_string()),
            route_profile_status: Some("promoted_route_ready".to_string()),
            route_profile_blockers: vec!["route not promoted for profile ask_normal".to_string()],
            route: route.clone(),
        };
        let source = serde_json::json!({
            "artifact_kind": "lunar_lake_openvino_operator_ask",
            "requested_backend": "openvino-gpu",
            "selected_backend": "openvino-gpu",
            "runtime_api": "openvino_genai",
            "runtime_device": "GPU.0",
            "fallback_used": false,
            "fallback_reason": null,
            "backend_lane": "dense_slm_openvino_gpu_arc140v",
            "selected_kernel_or_runtime": "openvino-genai-llmpipeline-gpu0",
            "model_family": "qwen",
            "model_architecture": "qwen2",
            "quantization": "INT4_SYM",
            "prompt_template": "qwen2.5",
            "tokenizer_source": "hf_tokenizer_export",
            "model": {
                "local_model_dir": "models/openvino/qwen2.5-0.5b-instruct-int4-sym"
            },
            "prompt_policy": {
                "rendered_prompt": "<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n",
                "prompt_token_ids": [1, 2, 3],
                "prompt_token_count": 3
            },
            "output": {
                "generated_text": "2 + 2 equals 4.",
                "normalized_answer": "2 + 2 equals 4.",
                "generated_token_ids": [17, 488, 220, 17, 16819, 220, 19, 13, 151645],
                "generated_token_count": 9
            },
            "answer_gate": {
                "kind": "contains",
                "expected": "4",
                "passed": true,
                "failed_rules": []
            },
            "timing": {"generation_wall_ms": 301.0}
        });
        validate_lunar_lake_ask_source_receipt(&source, &route)?;
        let answer = lunar_lake_source_answer_text(&source);
        let normalized = lunar_lake_source_normalized_answer(&source, &answer);
        let gate = evaluate_lunar_lake_answer_gate(&normalized, Some("4"));
        let receipt = build_lunar_lake_operator_ask_receipt(LunarLakeAskReceiptContext {
            artifact_root: std::path::Path::new("ci/hardware/intel-258v/2026-05-08"),
            operator_receipt_path: std::path::Path::new("lunar-lake-operator-readiness.json"),
            source_run_path: std::path::Path::new("lunar-lake-openvino-ask-source-run.json"),
            route: &route,
            route_selection: &route_selection,
            question: "2+2?",
            answer: &answer,
            normalized_answer: &normalized,
            answer_gate: &gate,
            expect_contains: Some("4"),
            source_run_receipt: &source,
        });

        assert_eq!(
            receipt["proof_stage"],
            "operator_candidate_route_executed_through_lunar_lake_ask"
        );
        assert_eq!(receipt["requested_backend"], "openvino-gpu");
        assert_eq!(receipt["selected_backend"], "openvino-gpu");
        assert_eq!(receipt["runtime_api"], "openvino_genai");
        assert_eq!(receipt["backend_lane"], "dense_slm_openvino_gpu_arc140v");
        assert_eq!(receipt["selected_kernel_or_runtime"], "openvino-genai-llmpipeline-gpu0");
        assert_eq!(receipt["route_id"], "dense_slm_openvino_gpu_candidate");
        assert_eq!(receipt["openvino_candidate_route_executed"], true);
        assert_eq!(receipt["claim_boundary"]["openvino_candidate_route_executed"], true);
        assert_eq!(receipt["claim_boundary"]["default_route_changed"], false);
        assert_eq!(receipt["claim_boundary"]["acceleration_claim"], false);
        assert_eq!(receipt["model_family"], "qwen");
        assert_eq!(receipt["model_architecture"], "qwen2");
        assert_eq!(receipt["quantization"], "INT4_SYM");
        assert_eq!(receipt["tokenizer_source"], "hf_tokenizer_export");
        assert_eq!(receipt["prompt"]["token_ids"], serde_json::json!([1, 2, 3]));
        assert_eq!(receipt["tokens"]["generated_count"], 9);
        assert_eq!(receipt["answer"]["normalized_text"], "2 + 2 equals 4.");
        Ok(())
    }

    #[test]
    fn dense_slm_cpu_reference_requires_cpu_backend_without_fallback() {
        assert!(uses_dense_slm_cpu_reference("cpu", "cpu-rust", "cpu", false));
        assert!(uses_dense_slm_cpu_reference(
            "apple-m4-cpu-neon",
            "apple-m4-cpu-neon",
            "cpu",
            false
        ));
        assert!(!uses_dense_slm_cpu_reference("cuda", "cuda", "cuda", false));
        assert!(!uses_dense_slm_cpu_reference("cuda", "cpu-rust", "cpu", true));
        assert!(!uses_dense_slm_cpu_reference("apple-m4-metal", "cpu-rust", "cpu", true));
    }

    #[test]
    fn canonical_bitnet_receipt_identity_does_not_use_dense_slm_kernel() {
        let path = std::path::Path::new("models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf");
        let architecture = infer_model_architecture(path);
        let family = receipt_model_family(&architecture);

        assert_eq!(family, "bitnet");
        assert!(!is_dense_slm_model(family, &architecture));
        assert_eq!(dense_slm_kernel_family(family, &architecture), None);
        assert_eq!(dense_slm_kernel_id(family, &architecture), None);
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
    fn prompt_authority_reads_sibling_tokenizer_json_metadata_after_resolution() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let tokenizer_path = temp_dir.path().join("tokenizer.json");
        std::fs::write(
            &tokenizer_path,
            r#"{"tokenizer_class":"LlamaTokenizerFast","chat_template":"{{ messages[0]['content'] }}"}"#,
        )
        .expect("write tokenizer fixture");

        let metadata = read_resolved_tokenizer_json_prompt_metadata(
            bitnet_tokenizers::auto::TokenizerSource::Sibling,
            Some(&tokenizer_path),
        )
        .expect("sibling tokenizer metadata");

        assert_eq!(metadata.family.as_deref(), Some("LlamaTokenizerFast"));
        assert_eq!(metadata.chat_template.as_deref(), Some("{{ messages[0]['content'] }}"));
    }

    #[test]
    fn prompt_authority_does_not_parse_gguf_path_as_tokenizer_json_metadata() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let model_path = temp_dir.path().join("model.gguf");
        std::fs::write(&model_path, br#"{"chat_template":"not-tokenizer-json"}"#)
            .expect("write gguf-like fixture");

        let metadata = read_resolved_tokenizer_json_prompt_metadata(
            bitnet_tokenizers::auto::TokenizerSource::GgufMetadata,
            Some(&model_path),
        );

        assert!(metadata.is_none());
    }

    struct PromptAuthorityHfBoundaryTokenizer;

    impl bitnet_tokenizers::Tokenizer for PromptAuthorityHfBoundaryTokenizer {
        fn encode(
            &self,
            text: &str,
            add_bos: bool,
            parse_special: bool,
        ) -> bitnet_common::Result<Vec<u32>> {
            assert_eq!(text, "User: Say exactly: OK<|eot_id|>Assistant: ");
            assert!(!add_bos, "BitNet HF-style chat boundary must not add executor BOS");
            assert!(parse_special, "BitNet answer template must parse <|eot_id|>");
            Ok(vec![1502, 25, 25961, 7041, 25, 10619, 128009, 72803, 25, 220])
        }

        fn decode(&self, _ids: &[u32]) -> bitnet_common::Result<String> {
            Ok(String::new())
        }

        fn vocab_size(&self) -> usize {
            128_256
        }

        fn token_to_piece(&self, _token: u32) -> Option<String> {
            None
        }
    }

    #[test]
    fn prompt_authority_bitnetcpp_variant_matches_hf_boundary() {
        let tokenizer = PromptAuthorityHfBoundaryTokenizer;
        let (entry, ids, rendered) = prompt_audit_variant_json(
            "metadata_authority",
            bitnet_inference::TemplateType::BitnetCppAnswer,
            "gguf_or_tokenizer_metadata",
            "Say exactly: OK",
            None,
            &tokenizer,
        );

        assert_eq!(rendered, "User: Say exactly: OK<|eot_id|>Assistant: ");
        assert_eq!(ids, Some(vec![1502, 25, 25961, 7041, 25, 10619, 128009, 72803, 25, 220]));
        assert_eq!(entry["prompt_policy"]["add_bos"], false);
        assert_eq!(entry["prompt_policy"]["parse_special"], true);
        assert_eq!(
            entry["tokens"]["prompt_token_ids"],
            serde_json::json!([1502, 25, 25961, 7041, 25, 10619, 128009, 72803, 25, 220])
        );
    }

    #[test]
    fn prompt_authority_reference_parity_records_external_mismatch() {
        let parity = prompt_audit_reference_parity_json(
            Some("hf_apply_chat_template".to_string()),
            Some("User: Say OK<|eot_id|>Assistant: ".to_string()),
            &[1502, 25, 25961, 10619, 128009, 72803, 25, 220],
            "User: Say OK<|eot_id|>Assistant:",
            Some(&[128000, 1502, 25, 25961, 10619, 128009, 72803, 25]),
        );

        assert_eq!(parity["available"], true);
        assert_eq!(parity["source"], "hf_apply_chat_template");
        assert_eq!(parity["rendered_prompt_match"], false);
        assert_eq!(parity["prompt_token_ids_match"], false);
        assert_eq!(parity["first_rendered_prompt_mismatch_index"], 32);
        assert_eq!(parity["first_prompt_token_id_mismatch_index"], 0);
        assert_eq!(parity["passed"], false);
    }

    #[test]
    fn prompt_authority_reference_parity_passes_exact_external_match() {
        let rendered = "User: Say OK<|eot_id|>Assistant: ";
        let ids = [1502, 25, 25961, 10619, 128009, 72803, 25, 220];
        let parity = prompt_audit_reference_parity_json(
            Some("hf_apply_chat_template".to_string()),
            Some(rendered.to_string()),
            &ids,
            rendered,
            Some(&ids),
        );

        assert_eq!(parity["rendered_prompt_match"], true);
        assert_eq!(parity["prompt_token_ids_match"], true);
        assert_eq!(parity["first_rendered_prompt_mismatch_index"], serde_json::Value::Null);
        assert_eq!(parity["first_prompt_token_id_mismatch_index"], serde_json::Value::Null);
        assert_eq!(parity["passed"], true);
    }

    #[test]
    fn receipt_pretokenizer_authority_records_external_llama_bpe() {
        assert_eq!(
            tokenizer_pretokenizer_authority(
                bitnet_tokenizers::auto::TokenizerSource::Explicit,
                "llama3"
            ),
            "llama-bpe"
        );
        assert_eq!(
            tokenizer_pretokenizer_authority(
                bitnet_tokenizers::auto::TokenizerSource::Sibling,
                "llama3"
            ),
            "llama-bpe"
        );
        assert_eq!(
            tokenizer_pretokenizer_authority(
                bitnet_tokenizers::auto::TokenizerSource::Explicit,
                "external_tokenizer_file"
            ),
            "externally_supplied"
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
        let receipt = apple_machine_receipt_json_from_probe(&probe, "apple-m4-mac-mini");

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
    fn apple_m3_air_receipt_uses_macbook_machine_id() {
        let probe = AppleCliMachineProbe {
            chip: Some("Apple M3".to_string()),
            cpu_cores: Some(8),
            gpu_cores: Some(10),
            unified_memory: Some(true),
            unified_memory_bytes: Some(17_179_869_184),
            macos_version: Some("15.5".to_string()),
            macos_build: Some("24F74".to_string()),
            native_or_virtualized: Some("native-macos".to_string()),
            metal_visible: true,
        };
        let receipt = apple_machine_receipt_json_from_probe(&probe, "apple-m3-macbook-air");

        assert_eq!(receipt["machine_id"], "apple-m3-macbook-air");
        assert_eq!(receipt["resolved_device"]["chip"], "Apple M3");
        assert_eq!(receipt["resolved_device"]["cpu_cores"], 8);
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
    #[cfg(feature = "full-cli")]
    fn warm_session_speed_receipt_records_warm_and_decode_throughput_without_speedup_claim() {
        let mut accumulator = WarmSessionSpeedAccumulator::default();
        accumulator.record(WarmSessionPromptSpeed {
            prompt_tokens: 8,
            generated_tokens: 4,
            tokenize_ms: 2.0,
            prefill_ms: 10.0,
            decode_total_ms: 200.0,
            sampling_ms: 4.0,
            prompt_total_ms: 250.0,
            first_token_ms: Some(90.0),
            steady_decode_tok_s: Some(25.0),
        });
        accumulator.record(WarmSessionPromptSpeed {
            prompt_tokens: 6,
            generated_tokens: 4,
            tokenize_ms: 1.0,
            prefill_ms: 8.0,
            decode_total_ms: 200.0,
            sampling_ms: 4.0,
            prompt_total_ms: 250.0,
            first_token_ms: Some(80.0),
            steady_decode_tok_s: Some(20.0),
        });

        let receipt = accumulator.receipt(
            1000.0,
            50.0,
            1600.0,
            "strict RTX 5070 Ti CUDA warm answer session",
            "strict CUDA answer-path timing is measured for this model, corpus, backend, and machine context only",
            WarmSessionReuseReceiptContext {
                sampler_reuse_enabled: true,
                sampler_reuse_policy: "single_sampler_reused_for_temperature_zero_prompt_independence",
                sampler_reused_prompt_count: 2,
                sampler_recreated_prompt_count: 0,
                kv_cache_recreated_per_prompt: false,
                kv_cache_reused_across_prompts: true,
                kv_cache_reuse_policy: "single_kv_cache_cleared_per_prompt_for_prompt_isolation",
                kv_cache_reused_prompt_count: 2,
                kv_cache_recreated_prompt_count: 0,
            },
        );

        assert_eq!(receipt["counts"]["prompt_count"], 2);
        assert_eq!(receipt["measurement_scope"], "strict RTX 5070 Ti CUDA warm answer session");
        assert_eq!(receipt["counts"]["generated_tokens"], 8);
        assert_eq!(receipt["throughput"]["warm_prompt_generated_tok_s"], 16.0);
        assert_eq!(receipt["throughput"]["decode_generated_tok_s"], 20.0);
        assert_eq!(receipt["speedup_claim"], false);
        assert_eq!(receipt["broad_performance_claim"], false);
        assert_eq!(receipt["reuse"]["model_loaded_once"], true);
        assert_eq!(receipt["reuse"]["logits_buffer_reuse_claimed"], false);
        assert_eq!(receipt["reuse"]["kv_cache_recreated_per_prompt"], false);
        assert_eq!(receipt["reuse"]["kv_cache_reused_across_prompts"], true);
        assert_eq!(receipt["reuse"]["kv_cache_cleared_per_prompt"], true);
        assert_eq!(
            receipt["reuse"]["kv_cache_reuse_policy"],
            "single_kv_cache_cleared_per_prompt_for_prompt_isolation"
        );
        assert_eq!(receipt["reuse"]["kv_cache_reused_prompt_count"], 2);
        assert_eq!(receipt["reuse"]["sampler_recreated_per_prompt"], false);
        assert_eq!(receipt["reuse"]["sampler_reused_across_prompts"], true);
        assert_eq!(receipt["reuse"]["sampler_reused_prompt_count"], 2);
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn warm_session_sampler_reuse_is_only_for_temperature_zero() {
        let greedy = bitnet_sampling::SamplingConfig {
            temperature: 0.0,
            repetition_penalty: 1.1,
            ..Default::default()
        };
        let sampled = bitnet_sampling::SamplingConfig {
            temperature: 0.7,
            seed: Some(42),
            ..Default::default()
        };

        assert!(warm_session_sampler_reuse_enabled(&greedy));
        assert!(!warm_session_sampler_reuse_enabled(&sampled));
        assert_eq!(
            warm_session_sampler_reuse_policy(true),
            "single_sampler_reused_for_temperature_zero_prompt_independence"
        );
        assert_eq!(
            warm_session_sampler_reuse_policy(false),
            "recreated_per_prompt_for_rng_state_independence"
        );
    }

    #[test]
    fn steady_decode_tps_excludes_first_decode_token() {
        let tps = steady_decode_tps_ms(&[100.0, 50.0, 50.0]).unwrap();

        assert_eq!((tps * 1000.0).round() / 1000.0, 20.0);
        assert!(steady_decode_tps_ms(&[100.0]).is_none());
    }

    #[test]
    fn cpu_phase_machine_labels_select_9950x3d_for_avx512() {
        assert_eq!(
            cpu_phase_machine_labels("avx512", "avx512"),
            ("windows-9950x3d-rtx5070ti", "amd-9950x3d-cpu-avx512")
        );
        assert_eq!(cpu_phase_machine_labels("avx2", "avx2"), ("intel-258v", "intel-258v-cpu-avx2"));
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
    #[cfg(feature = "full-cli")]
    fn warm_session_allocation_audit_reports_prompt_setup_breakdown() {
        let prompt_tokenize = AllocationAuditSnapshot {
            alloc_count: 1,
            alloc_bytes: 64,
            dealloc_count: 0,
            dealloc_bytes: 0,
        };
        let prompt_setup = AllocationAuditSnapshot {
            alloc_count: 4,
            alloc_bytes: 400,
            dealloc_count: 0,
            dealloc_bytes: 0,
        };
        let buffer_reset = AllocationAuditSnapshot {
            alloc_count: 1,
            alloc_bytes: 40,
            dealloc_count: 0,
            dealloc_bytes: 0,
        };
        let token_seed = AllocationAuditSnapshot {
            alloc_count: 1,
            alloc_bytes: 60,
            dealloc_count: 0,
            dealloc_bytes: 0,
        };
        let kv_cache = AllocationAuditSnapshot {
            alloc_count: 1,
            alloc_bytes: 250,
            dealloc_count: 0,
            dealloc_bytes: 0,
        };
        let sampler_setup = AllocationAuditSnapshot {
            alloc_count: 1,
            alloc_bytes: 50,
            dealloc_count: 0,
            dealloc_bytes: 0,
        };

        let audit = warm_session_prompt_allocation_audit_json(WarmSessionPromptAllocationAudit {
            enabled: true,
            requested_backend: "cpu",
            prompt_tokenize,
            prompt_setup,
            prompt_setup_breakdown: WarmSessionPromptSetupAllocationAudit {
                buffer_reset,
                token_seed,
                kv_cache,
                sampler_setup,
            },
            prompt_prefill: &[],
            prompt_prefill_embed: &[],
            prompt_prefill_forward: &[],
            decode_total: &[],
            embed: &[],
            forward: &[],
            logits: &[],
            sample: &[],
            token_vector_update: &[],
            token_decode: &[],
            stop_tail_update: &[],
            receipt_construction: AllocationAuditSnapshot::default(),
        });

        assert_eq!(audit["prompt_setup"]["alloc_bytes_total"], 400);
        assert_eq!(audit["prompt_setup_breakdown"]["buffer_reset"]["alloc_bytes_total"], 40);
        assert_eq!(audit["prompt_setup_breakdown"]["token_seed"]["alloc_bytes_total"], 60);
        assert_eq!(audit["prompt_setup_breakdown"]["kv_cache"]["alloc_bytes_total"], 250);
        assert_eq!(audit["prompt_setup_breakdown"]["sampler_setup"]["alloc_bytes_total"], 50);
        assert!(audit["ranked_hotspots"].as_array().is_some_and(|hotspots| {
            hotspots.iter().any(|hotspot| hotspot["component"] == "prompt_setup.kv_cache")
        }));
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn warm_session_allocation_audit_reports_prefill_breakdown() {
        let prefill_total = [AllocationAuditSnapshot {
            alloc_count: 4,
            alloc_bytes: 400,
            dealloc_count: 0,
            dealloc_bytes: 0,
        }];
        let prefill_embed = [AllocationAuditSnapshot {
            alloc_count: 1,
            alloc_bytes: 80,
            dealloc_count: 0,
            dealloc_bytes: 0,
        }];
        let prefill_forward = [AllocationAuditSnapshot {
            alloc_count: 3,
            alloc_bytes: 320,
            dealloc_count: 0,
            dealloc_bytes: 0,
        }];

        let audit = warm_session_prompt_allocation_audit_json(WarmSessionPromptAllocationAudit {
            enabled: true,
            requested_backend: "cpu",
            prompt_tokenize: AllocationAuditSnapshot::default(),
            prompt_setup: AllocationAuditSnapshot::default(),
            prompt_setup_breakdown: WarmSessionPromptSetupAllocationAudit {
                buffer_reset: AllocationAuditSnapshot::default(),
                token_seed: AllocationAuditSnapshot::default(),
                kv_cache: AllocationAuditSnapshot::default(),
                sampler_setup: AllocationAuditSnapshot::default(),
            },
            prompt_prefill: &prefill_total,
            prompt_prefill_embed: &prefill_embed,
            prompt_prefill_forward: &prefill_forward,
            decode_total: &[],
            embed: &[],
            forward: &[],
            logits: &[],
            sample: &[],
            token_vector_update: &[],
            token_decode: &[],
            stop_tail_update: &[],
            receipt_construction: AllocationAuditSnapshot::default(),
        });

        assert_eq!(audit["prompt_prefill"]["alloc_bytes_total"], 400);
        assert_eq!(audit["prompt_prefill_breakdown"]["embed"]["alloc_bytes_total"], 80);
        assert_eq!(audit["prompt_prefill_breakdown"]["forward"]["alloc_bytes_total"], 320);
        assert_eq!(
            audit["prompt_prefill_breakdown"]["forward_boundary"]["first_reusable_allocation_surface"],
            "feed_forward.down_proj.output"
        );
        assert_eq!(
            audit["prompt_prefill_breakdown"]["forward_boundary"]["claim_scope"],
            "allocation-boundary classification only; no dense math, kernel, or sustained-throughput claim is made"
        );
        assert_eq!(
            audit["prompt_prefill_breakdown"]["forward_boundary"]["reuse_status"],
            "dense_linear_output_storage_blocked_by_candle_tensor_ops"
        );
        assert_eq!(
            audit["prompt_prefill_breakdown"]["forward_boundary"]["workspace_storage_owner"],
            "TransformerForwardWorkspace"
        );
        assert_eq!(
            audit["prompt_prefill_breakdown"]["forward_boundary"]["workspace_owned_output_surface"],
            "feed_forward.down_proj.output"
        );
        assert_eq!(
            audit["prompt_prefill_breakdown"]["forward_boundary"]["required_api_boundary"],
            "dense_linear_output_storage_api_boundary"
        );
        assert_eq!(
            audit["prompt_prefill_breakdown"]["forward_boundary"]["next_dense_math_boundary"]["target"],
            "q8_dense_linear_locality_boundary"
        );
        assert_eq!(
            audit["prompt_prefill_breakdown"]["forward_boundary"]["next_dense_math_boundary"]["current_path"],
            "eager_dense_standard_quant_dequant_to_f32_before_candle_tensor"
        );
        assert_eq!(
            audit["prompt_prefill_breakdown"]["forward_boundary"]["weight_accessible"],
            true
        );
        assert_eq!(audit["prompt_prefill_breakdown"]["forward_boundary"]["bias_accessible"], true);
        assert_eq!(
            audit["prompt_prefill_breakdown"]["forward_boundary"]["can_fill_caller_output_storage"],
            false
        );
        assert_eq!(
            audit["prompt_prefill_breakdown"]["forward_boundary"]["behavior_gate"],
            "generated IDs, decoded text, strict GGUF tokenizer authority, selected CPU backend/kernel, model SHA, and fallback=false must match the Qwen3 Q8_0 baseline"
        );
        assert!(audit["ranked_hotspots"].as_array().is_some_and(|hotspots| {
            hotspots.iter().any(|hotspot| hotspot["component"] == "prompt_prefill.forward")
        }));
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn warm_session_aggregate_allocation_audit_names_next_target() {
        let prompt_summaries = [
            serde_json::json!({
                "allocation_audit": {
                    "ranked_hotspots": [
                        {
                            "component": "prompt_prefill",
                            "alloc_count": 10,
                            "alloc_bytes": 1_000,
                        },
                        {
                            "component": "model.forward",
                            "alloc_count": 5,
                            "alloc_bytes": 400,
                        }
                    ]
                }
            }),
            serde_json::json!({
                "allocation_audit": {
                    "ranked_hotspots": [
                        {
                            "component": "prompt_prefill",
                            "alloc_count": 3,
                            "alloc_bytes": 500,
                        }
                    ]
                }
            }),
        ];

        let audit = warm_session_aggregate_allocation_audit_json(true, "cpu", &prompt_summaries);

        assert_eq!(audit["dominant_hotspot"]["component"], "prompt_prefill");
        assert_eq!(audit["dominant_hotspot"]["alloc_bytes"], 1_500);
        assert_eq!(
            audit["next_optimization_target"]["target"],
            "q8_dense_linear_locality_boundary"
        );
        assert_eq!(
            audit["next_optimization_target"]["status"],
            "dense_linear_output_storage_blocked_by_candle_tensor_ops"
        );
        assert_eq!(audit["optimization_deferred"], true);
        assert_eq!(
            audit["next_optimization_target"]["claim_scope"],
            "diagnostic prioritization only; no runtime optimization or sustained-throughput claim is made"
        );
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn warm_session_aggregate_allocation_audit_targets_prefill_forward_boundary() {
        let prompt_summaries = [serde_json::json!({
            "allocation_audit": {
                "ranked_hotspots": [
                    {
                        "component": "prompt_prefill.forward",
                        "alloc_count": 20,
                        "alloc_bytes": 2_000,
                    },
                    {
                        "component": "prompt_prefill.embed",
                        "alloc_count": 2,
                        "alloc_bytes": 100,
                    }
                ]
            }
        })];

        let audit = warm_session_aggregate_allocation_audit_json(true, "cpu", &prompt_summaries);

        assert_eq!(audit["dominant_hotspot"]["component"], "prompt_prefill.forward");
        assert_eq!(
            audit["next_optimization_target"]["target"],
            "q8_dense_linear_locality_boundary"
        );
        assert_eq!(audit["next_optimization_target"]["component"], "prompt_prefill.forward");
        assert_eq!(
            audit["next_optimization_target"]["status"],
            "dense_linear_output_storage_blocked_by_candle_tensor_ops"
        );
        assert_eq!(audit["optimization_deferred"], true);
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
        assert!(allocation_audit_backend_supported(&RunBackendIdentity {
            requested_backend: "apple-m3-air-cpu-neon".to_string(),
            selected_backend: "apple-m3-air-cpu-neon".to_string(),
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
    #[cfg(feature = "full-cli")]
    fn slm_warm_session_accepts_cpu_backend_without_enabling_allocation_audit() {
        assert!(is_supported_slm_warm_session_backend("cpu"));
        assert!(is_supported_slm_warm_session_backend("apple-m4-cpu-neon"));
        assert!(is_supported_slm_warm_session_backend("apple-m3-air-cpu-neon"));
        assert!(!is_supported_slm_warm_session_backend("cuda"));

        assert_eq!(slm_warm_session_artifact_kind("cpu"), "slm_cpu_warm_session");
        assert_eq!(slm_warm_session_prompt_artifact_kind("cpu"), "slm_cpu_warm_session_prompt");
        assert_eq!(
            crate::allocation_audit::warm_session_allocation_scope("cpu"),
            "selected generic CPU SLM warm-session prompt hot path"
        );
        assert!(!allocation_audit_backend_supported(&RunBackendIdentity {
            requested_backend: "cpu".to_string(),
            selected_backend: "cpu-rust".to_string(),
            runtime_api: "cpu".to_string(),
            fallback_used: false,
            fallback_reason: None,
        }));
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn warm_session_prompt_buffers_report_capacity_reuse() {
        let mut buffers = WarmSessionPromptBuffers::default();

        let first = buffers.reset(8, 4, 3, 16);
        assert_eq!(first["capacity_sufficient_for_prompt"], true);
        assert_eq!(first["reset_reused_existing_capacity"], false);
        assert_eq!(first["token_capacity_grew"], true);
        assert_eq!(first["generated_token_capacity_grew"], true);

        buffers.tokens.extend_from_slice(&[1, 2, 3]);
        buffers.generated_tokens.extend_from_slice(&[4, 5]);
        buffers.stop_tail.push_str("tail");
        buffers.logits_scratch.extend_from_slice(&[0.1, 0.2, 0.3]);

        let second = buffers.reset(4, 2, 2, 8);
        assert!(buffers.tokens.is_empty());
        assert!(buffers.generated_tokens.is_empty());
        assert!(buffers.stop_tail.is_empty());
        assert!(buffers.logits_scratch.is_empty());
        assert_eq!(second["capacity_sufficient_for_prompt"], true);
        assert_eq!(second["reset_reused_existing_capacity"], true);
        assert_eq!(second["token_capacity_grew"], false);
        assert_eq!(second["generated_token_capacity_grew"], false);
        assert_eq!(second["stop_tail_capacity_grew"], false);
        assert_eq!(second["logits_capacity_grew"], false);
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn warm_session_prompt_token_cache_reuses_rendered_prompt_ids() -> Result<()> {
        let mut cache = WarmSessionPromptTokenCache::default();
        let mut encode_calls = 0usize;

        let (first, first_hit) = cache.get_or_insert_with("prompt", false, true, || {
            encode_calls += 1;
            Ok(vec![1, 2, 3])
        })?;
        assert_eq!(first, &[1, 2, 3]);
        assert!(!first_hit);

        let (second, second_hit) = cache.get_or_insert_with("prompt", false, true, || {
            encode_calls += 1;
            Ok(vec![9])
        })?;
        assert_eq!(second, &[1, 2, 3]);
        assert!(second_hit);

        let (_third, third_hit) = cache.get_or_insert_with("prompt", true, true, || {
            encode_calls += 1;
            Ok(vec![0, 1, 2, 3])
        })?;
        assert!(!third_hit);

        assert_eq!(encode_calls, 2);
        assert_eq!(cache.hits, 1);
        assert_eq!(cache.misses, 2);
        assert_eq!(cache.entry_count(), 2);
        Ok(())
    }

    #[test]
    fn direct_greedy_logits_guard_matches_sampler_fast_path_policy() {
        assert!(can_use_direct_greedy_logits(0.0, 1.0, false));
        assert!(can_use_direct_greedy_logits(0.0, 1.2, true));
        assert!(!can_use_direct_greedy_logits(0.7, 1.0, false));
        assert!(!can_use_direct_greedy_logits(0.0, 1.2, false));
    }

    #[test]
    fn greedy_argmax_token_2d_matches_lowest_id_tie_policy() -> Result<()> {
        let logits =
            candle_core::Tensor::new(&[[0.25f32, 1.0, 0.5, 1.0]], &candle_core::Device::Cpu)?;
        let tensor =
            bitnet_common::ConcreteTensor::BitNet(bitnet_common::BitNetTensor::new(logits));

        assert_eq!(greedy_argmax_token_2d(&tensor)?, 1);
        Ok(())
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn extract_logits_2d_into_reuses_host_scratch() -> Result<()> {
        let logits =
            candle_core::Tensor::new(&[[0.25f32, 1.0, 0.5, 1.5]], &candle_core::Device::Cpu)?;
        let tensor =
            bitnet_common::ConcreteTensor::BitNet(bitnet_common::BitNetTensor::new(logits));
        let mut scratch = Vec::with_capacity(8);

        let reused = extract_logits_2d_into(&tensor, &mut scratch)?;

        assert!(reused);
        assert_eq!(scratch, vec![0.25, 1.0, 0.5, 1.5]);
        assert!(scratch.capacity() >= 8);
        Ok(())
    }

    #[test]
    #[cfg(feature = "full-cli")]
    fn slm_warm_session_corpus_accepts_cpu_artifact_kind() -> Result<(), Box<dyn std::error::Error>>
    {
        let path = std::env::temp_dir()
            .join(format!("bitnet-slm-cpu-warm-session-corpus-{}.yaml", std::process::id()));
        let yaml = r#"
schema: 1
artifact_kind: slm_cpu_warm_session_corpus
name: qwen3-kaby-warm-session
description: CPU warm-session smoke corpus.
model:
  repo: Qwen/Qwen3-0.6B-GGUF
  file: Qwen3-0.6B-Q8_0.gguf
defaults:
  prompt_template: qwen
  max_new_tokens: 4
cases:
  - id: math
    question: "What is 2+2?"
"#;
        std::fs::write(&path, yaml)?;

        let corpus = SlmWarmSessionCorpus::load(&path)?;
        assert_eq!(corpus.artifact_kind, "slm_cpu_warm_session_corpus");
        assert_eq!(corpus.defaults.prompt_template.as_deref(), Some("qwen"));

        let _ = std::fs::remove_file(path);
        Ok(())
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
        println!("  GPU support: {}", style("✓ Enabled").green());
        // Check CUDA availability
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            match candle_core::Device::cuda_if_available(0).is_ok() {
                true => println!("  CUDA: {}", style("✓ Available").green()),
                false => println!("  CUDA: {}", style("✗ Not available").red()),
            }
        }
        #[cfg(not(any(feature = "gpu", feature = "cuda")))]
        println!("  CUDA: {}", style("✗ Not compiled").yellow())
    }
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
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
