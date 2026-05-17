//! Mac-oriented operator wrappers for the supported Apple M4 SLM path.

use anyhow::{Context, Result, anyhow};
use bitnet_repl_core::{ReplInput, parse_repl_input};
use clap::{ArgAction, Args, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{self, BufRead, IsTerminal, Read, Write};
#[cfg(unix)]
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::process::Command;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::model_cache::{self, VerifiedCachedModel};

const APPLE_M4_CPU_NEON: &str = "apple-m4-cpu-neon";
const APPLE_M3_AIR_CPU_NEON: &str = "apple-m3-air-cpu-neon";
const APPLE_M4_METAL: &str = "apple-m4-metal";
const APPLE_M3_AIR_METAL: &str = "apple-m3-air-metal";
const APPLE_M3_AIR_MPSGRAPH: &str = "apple-m3-air-mpsgraph";
const MAC_ASK_DEFAULT_RECEIPT: &str = "target/apple-m4-productization/mac-ask.json";
const MAC_CHAT_DEFAULT_RECEIPT: &str = "target/apple-m4-continuity/mac-chat.json";
const MAC_SMOKE_DEFAULT_RECEIPT: &str = "target/apple-m4-continuity/mac-smoke.json";
const MAC_DOCTOR_DEFAULT_RECEIPT: &str = "target/apple-m4-slm-excellence/mac-doctor.json";
const MAC_STATUS_DEFAULT_RECEIPT: &str = "target/apple-m4-inference-ops/mac-status.json";
const MAC_REPORT_REFRESH_DEFAULT_RECEIPT: &str =
    "target/apple-m4-inference-ops/report-refresh-manifest.json";
const MAC_REGRESSION_DASHBOARD_DEFAULT_RECEIPT: &str =
    "target/apple-m4-inference-ops/regression-dashboard.json";
const MAC_REGRESSION_DASHBOARD_DEFAULT_MARKDOWN: &str =
    "target/apple-m4-inference-ops/regression-dashboard.md";
const APPLE_M4_REPORT_ROOT: &str = "ci/hardware/apple-m4-mac-mini";
const MAC_BITNET_WARM_DEFAULT_RECEIPT: &str = "target/apple-m4-local-answer/mac-bitnet-warm.json";
const MAC_BITNET_BENCHMARK_DEFAULT_RECEIPT: &str =
    "target/apple-m4-bitnet-eval-and-benchmark/bitnet-benchmark/summary.json";
const MAC_BITNET_CHAT_GATE_DEFAULT_RECEIPT: &str =
    "target/apple-m4-bitnet-productization/bitnet-chat-gate.json";
const MAC_SERVE_DEFAULT_RECEIPT_DIR: &str = "target/apple-m4-local-server/receipts";
const MAC_SERVE_CHECK_DEFAULT_RECEIPT: &str = "target/apple-m4-local-server/mac-serve-check.json";
const MAC_SERVE_DEFAULT_HOST: &str = "127.0.0.1";
const MAC_SERVE_DEFAULT_PORT: u16 = 8080;
const MAC_SERVE_DEFAULT_MAX_NEW_TOKENS: usize = 64;
const MAC_BITNET_PROOF_DEFAULT_RECEIPT: &str =
    "target/apple-m4-continuity/mac-bitnet-proof-preflight.json";
const MAC_VALIDATE_DEFAULT_RECEIPT: &str = "target/apple-m4-productization/mac-validate.json";
const MAC_BENCHMARK_DEFAULT_RECEIPT: &str = "target/apple-m4-slm-eval-v2/mac-benchmark.json";
const MAC_VALIDATE_DEFAULT_CORPUS: &str = "ci/quality/apple-m4-slm-quality-corpus.yaml";
const MAC_SMOKE_PROMPT: &str = "Answer with a single digit: 2+2=";
const MAC_SMOKE_EXPECTED_FRAGMENT: &str = "4";
const QWEN_PROMPT_TEMPLATE: &str = "qwen2.5";
const BITNET_M4_EXPECTED_MODEL_SHA256: &str =
    "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162";
const BITNET_M4_EXPECTED_TOKENIZER_SHA256: &str =
    "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7";
const BITNET_M4_PROMPT_TEMPLATE: &str = "bitnetcpp-answer";
const BITNET_M4_MODEL_ID: &str = "microsoft-bitnet-b1.58-2B-4T-i2s";
const BITNET_M4_DEFAULT_TOKENIZER_PATH: &str = "models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json";
const LOW_DISK_HEADROOM_BYTES: u64 = 1_073_741_824;
const OPERATOR_PROFILE_TOKENS: &[usize] = &[16, 32, 64];
const PERFORMANCE_PROFILE_TOKENS: &[usize] = &[16, 32, 64, 128];
const OPERATOR_PROFILE_PROMPTS: &[&str] = &[
    "What is 2+2? Answer briefly.",
    "Name the capital of France.",
    "Write one short sentence about Rust.",
];
const BITNET_WARM_PROMPTS: &[&str] = &[
    "Answer with a single digit: 2+2=",
    "Name the capital of France. Answer with one word.",
    "Answer with a single digit: 2+2=",
];
const BITNET_WARM_PROFILE_PROMPTS: &[&str] = &[
    "Answer with a single digit: 2+2=",
    "Name the capital of France. Answer with one word.",
    "Return exactly READY.",
    "Answer with a single digit: 3+1=",
    "Write exactly the word blue.",
    "Answer yes or no: is fire hot?",
    "Answer with a single digit: 5-2=",
    "Write exactly OK.",
    "Answer with one word: what color is the sky on a clear day?",
    "Answer with a single digit: 1+1=",
    "Write exactly local.",
    "Answer yes or no: is ice warm?",
];
const M4_SLM_BENCHMARK_V2_PROFILES: &[&str] = &[
    "short_prompt_16_out",
    "short_prompt_64_out",
    "long_prompt_16_out",
    "long_prompt_128_out",
    "context_1k",
    "context_4k",
    "resident_25",
    "resident_50",
    "resident_100",
];
const M4_SLM_BENCHMARK_V2_TIMING_METRICS: &[&str] = &[
    "cold_load_ms",
    "tokenizer_load_ms",
    "prompt_tokenize_ms",
    "prefill_ms",
    "time_to_first_token_ms",
    "decode_total_ms",
    "sampling_ms_per_token",
    "total_wall_ms",
];
const M4_SLM_BENCHMARK_V2_THROUGHPUT_METRICS: &[&str] =
    &["input_tokens_per_second", "output_tokens_per_second", "decode_tokens_per_second"];
const M4_SLM_BENCHMARK_V2_MEMORY_METRICS: &[&str] = &["peak_memory_mb", "memory_drift_mb"];
const M4_SLM_BENCHMARK_V2_AGGREGATE_SPEED_METRICS: &[&str] = &[
    "cold_load_ms",
    "tokenizer_load_ms",
    "prompt_tokenize_ms",
    "prefill_ms",
    "ttft_ms",
    "sampling_ms_per_token",
    "input_tok_s",
    "output_tok_s",
    "decode_tok_s",
    "total_wall_ms",
];
const M4_SLM_BENCHMARK_V2_LEGACY_AGGREGATE_SPEED_METRICS: &[&str] = &[
    "cold_load_ms",
    "tokenizer_load_ms",
    "prompt_tokenize_ms",
    "prefill_ms",
    "ttft_ms",
    "input_tok_s",
    "output_tok_s",
    "decode_tok_s",
    "total_wall_ms",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MacValidateProfileSet {
    /// Run the deterministic smoke corpus once.
    Smoke,
    /// Run bounded 16/32/64 warm-answer timing profiles and write an aggregate summary.
    Operator,
    /// Run release-mode 16/32/64/128 warm-answer timing profiles.
    Performance,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MacSmokeModelFamily {
    /// Run the dense Qwen SLM golden smoke path.
    DenseSlm,
    /// Run the explicit BitNet one-shot ask smoke path.
    Bitnet,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MacChatModelFamily {
    /// Run the supported dense Qwen SLM resident chat path.
    DenseSlm,
    /// Reserved BitNet chat route. Currently gate-checked and disabled.
    Bitnet,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, ValueEnum)]
enum MacBenchmarkProfile {
    /// Short prompts with a 16-token output budget.
    #[value(name = "short_prompt_16_out")]
    ShortPrompt16Out,
    /// Short prompts with a 64-token output budget.
    #[value(name = "short_prompt_64_out")]
    ShortPrompt64Out,
    /// Long prompts with a 16-token output budget.
    #[value(name = "long_prompt_16_out")]
    LongPrompt16Out,
    /// Long prompts with a 128-token output budget.
    #[value(name = "long_prompt_128_out")]
    LongPrompt128Out,
    /// Synthetic context prompt targeting roughly 1k input tokens.
    #[value(name = "context_1k")]
    Context1k,
    /// Synthetic context prompt targeting roughly 4k input tokens.
    #[value(name = "context_4k")]
    Context4k,
    /// Resident 25-prompt warm-session profile.
    #[value(name = "resident_25")]
    Resident25,
    /// Resident 50-prompt warm-session profile.
    #[value(name = "resident_50")]
    Resident50,
    /// Resident 100-prompt warm-session profile.
    #[value(name = "resident_100")]
    Resident100,
}

impl MacBenchmarkProfile {
    fn id(self) -> &'static str {
        match self {
            Self::ShortPrompt16Out => "short_prompt_16_out",
            Self::ShortPrompt64Out => "short_prompt_64_out",
            Self::LongPrompt16Out => "long_prompt_16_out",
            Self::LongPrompt128Out => "long_prompt_128_out",
            Self::Context1k => "context_1k",
            Self::Context4k => "context_4k",
            Self::Resident25 => "resident_25",
            Self::Resident50 => "resident_50",
            Self::Resident100 => "resident_100",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, ValueEnum)]
enum BitnetWarmProfile {
    /// Resident 25-prompt variable warm-session profile.
    #[value(name = "resident_25")]
    Resident25,
    /// Resident 50-prompt variable warm-session profile.
    #[value(name = "resident_50")]
    Resident50,
    /// Resident 100-prompt variable warm-session profile.
    #[value(name = "resident_100")]
    Resident100,
}

impl BitnetWarmProfile {
    const fn id(self) -> &'static str {
        match self {
            Self::Resident25 => "resident_25",
            Self::Resident50 => "resident_50",
            Self::Resident100 => "resident_100",
        }
    }

    const fn prompt_count(self) -> usize {
        match self {
            Self::Resident25 => 25,
            Self::Resident50 => 50,
            Self::Resident100 => 100,
        }
    }
}

/// Run Apple M4 local operator flows with strict receipts.
#[derive(Debug, Args)]
pub struct MacCommand {
    #[command(subcommand)]
    action: MacAction,
}

#[derive(Debug, Subcommand)]
enum MacAction {
    /// List Apple M4 model states, cache status, and selection guidance.
    Models {
        /// Override model cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Emit JSON instead of text.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Summarize Apple M4 dense SLM, BitNet, disk/cache, receipt, and command readiness.
    Status {
        /// Override model cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Emit JSON to stdout after writing --json-out.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// Output strict Mac status receipt.
        #[arg(long, value_name = "PATH", default_value = MAC_STATUS_DEFAULT_RECEIPT)]
        json_out: PathBuf,
    },

    /// Generate a model-free advisory/nightly Apple M4 report refresh manifest.
    ReportRefresh {
        /// Committed Apple M4 report root to inventory.
        #[arg(long, value_name = "PATH", default_value = APPLE_M4_REPORT_ROOT)]
        root: PathBuf,

        /// Output strict report-refresh manifest receipt.
        #[arg(long, value_name = "PATH", default_value = MAC_REPORT_REFRESH_DEFAULT_RECEIPT)]
        json_out: PathBuf,

        /// Emit JSON to stdout after writing --json-out.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Build a model-free dense SLM and BitNet regression dashboard from committed reports.
    RegressionDashboard {
        /// Committed Apple M4 report root to inventory.
        #[arg(long, value_name = "PATH", default_value = APPLE_M4_REPORT_ROOT)]
        root: PathBuf,

        /// Output strict regression dashboard receipt.
        #[arg(long, value_name = "PATH", default_value = MAC_REGRESSION_DASHBOARD_DEFAULT_RECEIPT)]
        json_out: PathBuf,

        /// Output compact Markdown dashboard.
        #[arg(
            long,
            value_name = "PATH",
            default_value = MAC_REGRESSION_DASHBOARD_DEFAULT_MARKDOWN
        )]
        markdown_out: PathBuf,

        /// Emit JSON to stdout after writing --json-out and --markdown-out.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Check the cached Apple M4 SLM model artifact and routing boundary.
    Check {
        /// Supported model id. Defaults to the validated Apple M4 SLM runtime artifact.
        #[arg(long, default_value = model_cache::M4_SLM_RUNTIME_MODEL_ID)]
        model_id: String,

        /// Override model cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Emit JSON instead of text.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Ask one question through the Rust-native Apple M4 CPU/NEON SLM path.
    Ask {
        /// Question to answer. This is the shortest Mac path: `bitnet mac ask "What is 2+2?"`.
        #[arg(value_name = "QUESTION")]
        question: Option<String>,

        /// Question to answer. Kept for scripts that already use `--question`.
        #[arg(short, long = "question", value_name = "QUESTION")]
        question_flag: Option<String>,

        /// Supported model id. Defaults to the validated Apple M4 SLM runtime artifact.
        #[arg(long, default_value = model_cache::M4_SLM_RUNTIME_MODEL_ID)]
        model_id: String,

        /// Override model cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Explicit BitNet GGUF path for the one-shot BitNet ask route.
        #[arg(long = "model-path", value_name = "PATH")]
        model_path: Option<PathBuf>,

        /// Explicit tokenizer.json path for the one-shot BitNet ask route.
        #[arg(long, value_name = "PATH")]
        tokenizer: Option<PathBuf>,

        /// Optional system prompt.
        #[arg(long = "system", value_name = "TEXT")]
        system_prompt: Option<String>,

        /// Maximum new tokens to generate.
        #[arg(long, visible_aliases = ["max-tokens", "n-predict"], default_value_t = 32)]
        max_new_tokens: usize,

        /// Temperature for sampling. The Mac wrapper defaults to deterministic greedy.
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,

        /// Top-k sampling. The Mac wrapper defaults to greedy top-1 behavior.
        #[arg(long, default_value_t = 1)]
        top_k: usize,

        /// Top-p sampling.
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,

        /// Repetition penalty.
        #[arg(long, default_value_t = 1.1)]
        repetition_penalty: f32,

        /// Random seed for reproducibility.
        #[arg(long)]
        seed: Option<u64>,

        /// Number of CPU threads to use (0 = all cores; deterministic mode may override).
        #[arg(long, default_value_t = 0)]
        threads: usize,

        /// Optional wall-clock timeout for the explicit BitNet one-shot ask route.
        #[arg(long, value_name = "SECONDS")]
        timeout_seconds: Option<u64>,

        /// Output strict Mac answer receipt.
        #[arg(long, value_name = "PATH", default_value = MAC_ASK_DEFAULT_RECEIPT)]
        json_out: PathBuf,

        /// Emit operator progress lines to stderr while keeping generated text on stdout.
        #[arg(long, default_value_t = false)]
        progress: bool,

        /// Suppress Mac ask status/progress lines; generated text still uses stdout.
        #[arg(long, default_value_t = false)]
        quiet: bool,
    },

    /// Run a compact Apple M4 model-family health smoke with cache and receipt checks.
    Smoke {
        /// Model family to smoke. BitNet smoke uses one-shot ask; fixed warm proof is `mac bitnet-warm`.
        #[arg(long, value_enum, default_value_t = MacSmokeModelFamily::DenseSlm)]
        model_family: MacSmokeModelFamily,

        /// Supported model id. Defaults to the validated Apple M4 SLM runtime artifact.
        #[arg(long, default_value = model_cache::M4_SLM_RUNTIME_MODEL_ID)]
        model_id: String,

        /// Override model cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Explicit BitNet GGUF path. Only accepted with --model-family bitnet.
        #[arg(long, value_name = "PATH")]
        model_path: Option<PathBuf>,

        /// Explicit BitNet tokenizer path. Only accepted with --model-family bitnet.
        #[arg(long, value_name = "PATH")]
        tokenizer: Option<PathBuf>,

        /// Maximum new tokens for the fixed smoke prompt.
        #[arg(long, visible_aliases = ["max-tokens", "n-predict"], default_value_t = 4)]
        max_new_tokens: usize,

        /// Number of CPU threads to use (0 = all cores; deterministic mode may override).
        #[arg(long, default_value_t = 0)]
        threads: usize,

        /// Output aggregate golden smoke receipt.
        #[arg(long, value_name = "PATH", default_value = MAC_SMOKE_DEFAULT_RECEIPT)]
        json_out: PathBuf,
    },

    /// Run BitNet prompts in one warm Apple M4 CPU/NEON process without enabling chat.
    BitnetWarm {
        /// Accepted BitNet model id. Only microsoft-bitnet-b1.58-2B-4T-i2s is supported.
        #[arg(long, default_value = BITNET_M4_MODEL_ID)]
        model_id: String,

        /// Override model cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Explicit accepted BitNet GGUF path.
        #[arg(long = "model-path", value_name = "PATH")]
        model_path: Option<PathBuf>,

        /// Explicit accepted external tokenizer path. Defaults to the Microsoft tokenizer.
        #[arg(long, value_name = "PATH")]
        tokenizer: Option<PathBuf>,

        /// Operator prompt to run in the warm session. Repeat the flag and include at least one exact repeated prompt for determinism. Defaults to the fixed proof prompt set when omitted.
        #[arg(long = "prompt", value_name = "TEXT")]
        prompts: Vec<String>,

        /// Resident variable warm-session profile to run. Repeat for resident_25, resident_50, and resident_100 checkpoints.
        #[arg(long = "profile", value_enum, value_name = "PROFILE")]
        profiles: Vec<BitnetWarmProfile>,

        /// Maximum new tokens per warm prompt.
        #[arg(long, visible_aliases = ["max-tokens", "n-predict"], default_value_t = 8)]
        max_new_tokens: usize,

        /// Number of CPU threads to use (0 = all cores; deterministic mode may override).
        #[arg(long, default_value_t = 0)]
        threads: usize,

        /// Optional wall-clock timeout for the whole warm session.
        #[arg(long, value_name = "SECONDS")]
        timeout_seconds: Option<u64>,

        /// Emit operator progress lines to stderr.
        #[arg(long, default_value_t = false)]
        progress: bool,

        /// Suppress final status line; receipts are still written.
        #[arg(long, default_value_t = false)]
        quiet: bool,

        /// Output aggregate BitNet warm-session receipt.
        #[arg(long, value_name = "PATH", default_value = MAC_BITNET_WARM_DEFAULT_RECEIPT)]
        json_out: PathBuf,
    },

    /// Evaluate the receipt-backed gate for future BitNet Mac chat without enabling chat.
    BitnetChatGate {
        /// Variable BitNet warm-session receipt with operator prompts and repeated-prompt determinism.
        #[arg(long = "warm-receipt", value_name = "PATH")]
        warm_receipt: PathBuf,

        /// Warm-session failure or timeout receipt proving partial-failure diagnostics.
        #[arg(long = "failure-receipt", value_name = "PATH")]
        failure_receipt: Option<PathBuf>,

        /// Future streaming-semantics receipt. Missing evidence keeps the gate blocked.
        #[arg(long = "streaming-receipt", value_name = "PATH")]
        streaming_receipt: Option<PathBuf>,

        /// Emit the gate receipt JSON to stdout after writing --json-out.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// Output BitNet chat gate receipt.
        #[arg(long, value_name = "PATH", default_value = MAC_BITNET_CHAT_GATE_DEFAULT_RECEIPT)]
        json_out: PathBuf,
    },

    /// Benchmark BitNet one-shot ask and fixed warm paths without enabling chat or serve.
    BitnetBenchmark {
        /// Accepted BitNet model id. Only microsoft-bitnet-b1.58-2B-4T-i2s is supported.
        #[arg(long, default_value = BITNET_M4_MODEL_ID)]
        model_id: String,

        /// Override model cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Explicit accepted BitNet GGUF path.
        #[arg(long = "model-path", value_name = "PATH")]
        model_path: Option<PathBuf>,

        /// Explicit accepted external tokenizer path. Defaults to the Microsoft tokenizer.
        #[arg(long, value_name = "PATH")]
        tokenizer: Option<PathBuf>,

        /// One-shot prompt for the ask benchmark path.
        #[arg(long, default_value = "What is 2+2? Answer with only the number.")]
        one_shot_prompt: String,

        /// Maximum new tokens for the one-shot and fixed warm prompts.
        #[arg(long, visible_aliases = ["max-tokens", "n-predict"], default_value_t = 8)]
        max_new_tokens: usize,

        /// Number of CPU threads to use (0 = all cores; deterministic mode may override).
        #[arg(long, default_value_t = 0)]
        threads: usize,

        /// Emit operator progress lines to stderr.
        #[arg(long, default_value_t = false)]
        progress: bool,

        /// Suppress final status line; receipts are still written.
        #[arg(long, default_value_t = false)]
        quiet: bool,

        /// Output aggregate BitNet benchmark receipt.
        #[arg(long, value_name = "PATH", default_value = MAC_BITNET_BENCHMARK_DEFAULT_RECEIPT)]
        json_out: PathBuf,
    },

    /// Run one local health verdict for the supported Apple M4 dense-SLM path.
    Doctor {
        /// Supported model id. Defaults to the validated Apple M4 SLM runtime artifact.
        #[arg(long, default_value = model_cache::M4_SLM_RUNTIME_MODEL_ID)]
        model_id: String,

        /// Override model cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Maximum new tokens for the fixed smoke prompt.
        #[arg(long, visible_aliases = ["max-tokens", "n-predict"], default_value_t = 4)]
        max_new_tokens: usize,

        /// Number of CPU threads to use (0 = all cores; deterministic mode may override).
        #[arg(long, default_value_t = 0)]
        threads: usize,

        /// Output aggregate Mac doctor receipt.
        #[arg(long, value_name = "PATH", default_value = MAC_DOCTOR_DEFAULT_RECEIPT)]
        json_out: PathBuf,
    },

    /// Serve local M4 dense-SLM health and readiness endpoints without generation.
    Serve {
        /// Supported model id. Defaults to the validated Apple M4 SLM runtime artifact.
        #[arg(long, default_value = model_cache::M4_SLM_RUNTIME_MODEL_ID)]
        model_id: String,

        /// Override model cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// M4 server device route. Only apple-m4-cpu-neon is supported for full answers.
        #[arg(long, default_value = APPLE_M4_CPU_NEON)]
        device: String,

        /// Host to bind. Defaults to loopback for local appliance use.
        #[arg(long, default_value = MAC_SERVE_DEFAULT_HOST)]
        host: String,

        /// Port to bind.
        #[arg(long, default_value_t = MAC_SERVE_DEFAULT_PORT)]
        port: u16,

        /// Require strict cache, tokenizer, backend, and fallback behavior.
        #[arg(long, default_value_t = true)]
        strict: bool,

        /// Stream by default for later completion endpoints; health/ready do not generate.
        #[arg(long, default_value_t = true)]
        stream: bool,

        /// Default maximum new tokens for completion requests that omit max_tokens.
        #[arg(long, visible_aliases = ["max-tokens", "n-predict"], default_value_t = MAC_SERVE_DEFAULT_MAX_NEW_TOKENS)]
        max_new_tokens: usize,

        /// Default completion temperature.
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,

        /// Default top-k setting.
        #[arg(long, default_value_t = 1)]
        top_k: usize,

        /// Default nucleus sampling top-p setting.
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,

        /// Default repetition penalty.
        #[arg(long, default_value_t = 1.1)]
        repetition_penalty: f32,

        /// Optional deterministic sampling seed.
        #[arg(long)]
        seed: Option<u64>,

        /// Directory where later request receipts will be exported.
        #[arg(long, value_name = "PATH", default_value = MAC_SERVE_DEFAULT_RECEIPT_DIR)]
        receipt_dir: PathBuf,
    },

    /// Check a running M4 local server readiness and optional completion/receipt export.
    ServeCheck {
        /// Base URL for the running local M4 server.
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,

        /// Run a tiny completion and verify its receipt export endpoint.
        #[arg(long, default_value_t = false)]
        completion: bool,

        /// Prompt for the optional completion probe.
        #[arg(long, default_value = "What is 2+2? Answer briefly.")]
        prompt: String,

        /// Maximum new tokens for the optional completion probe.
        #[arg(long, visible_aliases = ["max-tokens", "n-predict"], default_value_t = 1)]
        max_new_tokens: usize,

        /// Output the server readiness check receipt.
        #[arg(long, value_name = "PATH", default_value = MAC_SERVE_CHECK_DEFAULT_RECEIPT)]
        json_out: PathBuf,
    },

    /// Run multiple prompts in one resident Apple M4 CPU/NEON SLM session.
    Chat {
        /// Model family for the resident chat route. BitNet is gate-checked and disabled.
        #[arg(long, value_enum, default_value_t = MacChatModelFamily::DenseSlm)]
        model_family: MacChatModelFamily,

        /// Prompt to answer. Repeat for each turn in the resident Mac session.
        #[arg(long = "prompt", value_name = "TEXT")]
        prompts: Vec<String>,

        /// Read additional prompts from stdin, one non-empty line per turn.
        #[arg(long, default_value_t = false)]
        stdin: bool,

        /// Collect prompts from an interactive line loop until /exit, /quit, or EOF.
        #[arg(long, default_value_t = false)]
        interactive: bool,

        /// Supported model id. Defaults to the validated Apple M4 SLM runtime artifact.
        #[arg(long, default_value = model_cache::M4_SLM_RUNTIME_MODEL_ID)]
        model_id: String,

        /// Override model cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Optional system prompt.
        #[arg(long = "system", value_name = "TEXT")]
        system_prompt: Option<String>,

        /// Maximum new tokens to generate per prompt.
        #[arg(long, visible_aliases = ["max-tokens", "n-predict"], default_value_t = 64)]
        max_new_tokens: usize,

        /// Temperature for sampling. The Mac wrapper defaults to deterministic greedy.
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,

        /// Top-k sampling. The Mac wrapper defaults to greedy top-1 behavior.
        #[arg(long, default_value_t = 1)]
        top_k: usize,

        /// Top-p sampling.
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,

        /// Repetition penalty.
        #[arg(long, default_value_t = 1.1)]
        repetition_penalty: f32,

        /// Random seed for reproducibility.
        #[arg(long)]
        seed: Option<u64>,

        /// Number of CPU threads to use (0 = all cores; deterministic mode may override).
        #[arg(long, default_value_t = 0)]
        threads: usize,

        /// Do not stream generated token text to stdout.
        #[arg(long = "no-stream", action = ArgAction::SetFalse, default_value_t = true)]
        stream: bool,

        /// Emit operator progress lines to stderr while keeping token text on stdout.
        #[arg(long, default_value_t = false)]
        progress: bool,

        /// Suppress warm-session status/progress lines; token streaming still uses stdout.
        #[arg(long, default_value_t = false)]
        quiet: bool,

        /// Disable per-turn receipt files; the aggregate session receipt still records each turn.
        #[arg(long = "no-turn-receipts", action = ArgAction::SetFalse, default_value_t = true)]
        turn_receipts: bool,

        /// Include scoped hot-loop allocation counter deltas in session receipts.
        #[arg(long, default_value_t = false)]
        allocation_audit: bool,

        /// Route the validated Q/K/V prefill Metal phase as an opt-in contribution; answers remain CPU/NEON.
        #[arg(long = "metal-prefill-qkv-phase", default_value_t = false)]
        metal_prefill_qkv_phase: bool,

        /// Output aggregate Mac chat session receipt.
        #[arg(long, value_name = "PATH", default_value = MAC_CHAT_DEFAULT_RECEIPT)]
        json_out: PathBuf,
    },

    /// Run the supported SLM quality corpus in one warm Apple M4 CPU/NEON session.
    Validate {
        /// Supported model id. Defaults to the validated Apple M4 SLM runtime artifact.
        #[arg(long, default_value = model_cache::M4_SLM_RUNTIME_MODEL_ID)]
        model_id: String,

        /// Override model cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Deterministic SLM quality corpus.
        #[arg(long, value_name = "PATH", default_value = MAC_VALIDATE_DEFAULT_CORPUS)]
        corpus: PathBuf,

        /// Number of repeated runs for each corpus case.
        #[arg(long, default_value_t = 2)]
        corpus_repeat_runs: usize,

        /// Validation profile set. Use operator for 16/32/64 profiles or performance for release-mode 16/32/64/128 profiles.
        #[arg(long, value_enum, default_value_t = MacValidateProfileSet::Smoke)]
        profile_set: MacValidateProfileSet,

        /// Maximum new tokens per prompt when the corpus does not override it.
        #[arg(long, visible_aliases = ["max-tokens", "n-predict"], default_value_t = 32)]
        max_new_tokens: usize,

        /// Number of CPU threads to use (0 = all cores; deterministic mode may override).
        #[arg(long, default_value_t = 0)]
        threads: usize,

        /// Include scoped hot-loop allocation counter deltas in warm-session profile receipts.
        #[arg(long, default_value_t = false)]
        allocation_audit: bool,

        /// Route the validated Q/K/V prefill Metal phase during the smoke quality corpus.
        #[arg(long = "metal-prefill-qkv-phase", default_value_t = false)]
        metal_prefill_qkv_phase: bool,

        /// Emit operator progress lines to stderr while validation runs.
        #[arg(long, default_value_t = false)]
        progress: bool,

        /// Suppress validation status/progress lines; receipt artifacts are still written.
        #[arg(long, default_value_t = false)]
        quiet: bool,

        /// Output aggregate warm-session receipt.
        #[arg(long, value_name = "PATH", default_value = MAC_VALIDATE_DEFAULT_RECEIPT)]
        json_out: PathBuf,
    },

    /// Run release-mode dense SLM benchmark profiles for v2 M4 receipts.
    Benchmark {
        /// Supported dense SLM model id. Defaults to the validated Apple M4 SLM runtime artifact.
        #[arg(long, default_value = model_cache::M4_SLM_RUNTIME_MODEL_ID)]
        model_id: String,

        /// Override model cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

        /// Benchmark profile to run. Repeat to build the v2 profile matrix.
        #[arg(long, value_enum, required = true)]
        profile: Vec<MacBenchmarkProfile>,

        /// Number of CPU threads to use (0 = all cores; deterministic mode may override).
        #[arg(long, default_value_t = 0)]
        threads: usize,

        /// Include scoped hot-loop allocation counter deltas in profile receipts.
        #[arg(long, default_value_t = false)]
        allocation_audit: bool,

        /// Emit operator progress lines to stderr while benchmark profiles run.
        #[arg(long, default_value_t = false)]
        progress: bool,

        /// Suppress benchmark status/progress lines; receipt artifacts are still written.
        #[arg(long, default_value_t = false)]
        quiet: bool,

        /// Output aggregate v2 benchmark receipt.
        #[arg(long, value_name = "PATH", default_value = MAC_BENCHMARK_DEFAULT_RECEIPT)]
        json_out: PathBuf,
    },

    /// Validate M4 BitNet proof inputs without enabling BitNet chat/serve routing.
    BitnetProof {
        /// Accepted BitNet GGUF path. This command never downloads model artifacts.
        #[arg(long, value_name = "PATH")]
        model: PathBuf,

        /// Tokenizer/pre-tokenizer authority for the accepted BitNet artifact.
        #[arg(long = "tokenizer-authority", value_name = "AUTHORITY")]
        tokenizer_authority: Option<String>,

        /// Receipt from the artifact sweep proving this BitNet artifact is accepted.
        #[arg(long = "accepted-artifact", value_name = "PATH")]
        accepted_artifact: Option<PathBuf>,

        /// Strict Apple M4 answer-corpus proof receipt to validate.
        #[arg(long = "proof-receipt", value_name = "PATH")]
        proof_receipt: Option<PathBuf>,

        /// Deterministic proof prompt to be used by the later strict proof item.
        #[arg(long, default_value = "What is 2+2? Answer briefly.")]
        prompt: String,

        /// Maximum new tokens for the later strict proof item.
        #[arg(long, visible_aliases = ["max-tokens", "n-predict"], default_value_t = 16)]
        max_new_tokens: usize,

        /// Require strict loader/tokenizer behavior for the later proof.
        #[arg(long, default_value_t = false)]
        strict: bool,

        /// Output the M4 BitNet proof preflight receipt.
        #[arg(long, value_name = "PATH", default_value = MAC_BITNET_PROOF_DEFAULT_RECEIPT)]
        json_out: PathBuf,
    },

    /// Check Apple M4 SLM answer/warm-session receipts for hidden fallback or overclaims.
    ReceiptsCheck {
        /// Receipt file or directory containing JSON receipts.
        path: PathBuf,

        /// Compare Apple M4 dense-SLM or BitNet receipts against a published baseline receipt.
        #[arg(long = "regression-baseline", value_name = "PATH")]
        regression_baseline: Option<PathBuf>,

        /// Emit JSON instead of text.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Compare Apple M4 dense-SLM or BitNet receipts against a stored local envelope.
    Regression {
        /// Current receipt file or directory containing JSON receipts.
        path: PathBuf,

        /// Stored M4 dense-SLM or BitNet envelope receipt to compare against.
        #[arg(long = "baseline", value_name = "PATH")]
        baseline: PathBuf,

        /// Fail when drift warnings are detected. Default mode is advisory.
        #[arg(long = "fail-on-drift", default_value_t = false)]
        fail_on_drift: bool,

        /// Emit JSON instead of text.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
}

#[derive(Debug, Serialize)]
struct ReceiptCheckSummary {
    path: PathBuf,
    artifact_kind: String,
    requested_backend: String,
    selected_backend: String,
    runtime_api: String,
    fallback_used: bool,
    prompt_count: Option<usize>,
    generated_tokens: Option<usize>,
    passed: bool,
    regression: Option<RegressionCheckSummary>,
}

#[derive(Debug, Serialize)]
struct RegressionCheckSummary {
    baseline_path: PathBuf,
    advisory: bool,
    matched_context: bool,
    warning_count: usize,
    warnings: Vec<RegressionWarning>,
}

#[derive(Debug, Serialize)]
struct RegressionWarning {
    profile_id: String,
    field: String,
    baseline: f64,
    observed: f64,
    threshold_percent: f64,
    direction: String,
}

impl MacCommand {
    pub(crate) fn default_log_level(&self) -> Option<&'static str> {
        match self.action {
            MacAction::Chat { .. } => Some("warn"),
            _ => None,
        }
    }

    pub async fn execute(self, explicit_device_label: Option<&str>) -> Result<()> {
        match self.action {
            MacAction::Models { cache_dir, json } => {
                ensure_supported_mac_device(explicit_device_label, "mac models")?;
                model_cache::list_apple_m4_models(cache_dir, json)
            }
            MacAction::Status { cache_dir, json, json_out } => {
                ensure_supported_mac_device(explicit_device_label, "mac status")?;
                run_status(cache_dir, json_out, json)
            }
            MacAction::ReportRefresh { root, json_out, json } => {
                ensure_supported_mac_device(explicit_device_label, "mac report-refresh")?;
                run_report_refresh_manifest(root, json_out, json)
            }
            MacAction::RegressionDashboard { root, json_out, markdown_out, json } => {
                ensure_supported_mac_device(explicit_device_label, "mac regression-dashboard")?;
                run_regression_dashboard(root, json_out, markdown_out, json)
            }
            MacAction::Check { model_id, cache_dir, json } => {
                ensure_supported_mac_device(explicit_device_label, "mac check")?;
                run_check(&model_id, cache_dir, json)
            }
            MacAction::Ask {
                question,
                question_flag,
                model_id,
                cache_dir,
                model_path,
                tokenizer,
                system_prompt,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                seed,
                threads,
                timeout_seconds,
                json_out,
                progress,
                quiet,
            } => {
                ensure_supported_mac_device(explicit_device_label, "mac ask")?;
                let question = resolve_mac_question(question, question_flag)?;
                run_ask(
                    &model_id,
                    cache_dir,
                    model_path,
                    tokenizer,
                    question,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    seed,
                    threads,
                    timeout_seconds,
                    json_out,
                    progress,
                    quiet,
                )
                .await
            }
            MacAction::Smoke {
                model_family,
                model_id,
                cache_dir,
                model_path,
                tokenizer,
                max_new_tokens,
                threads,
                json_out,
            } => {
                ensure_supported_mac_device(explicit_device_label, "mac smoke")?;
                run_smoke(
                    model_family,
                    &model_id,
                    cache_dir,
                    model_path,
                    tokenizer,
                    max_new_tokens,
                    threads,
                    json_out,
                )
                .await
            }
            MacAction::BitnetWarm {
                model_id,
                cache_dir,
                model_path,
                tokenizer,
                prompts,
                profiles,
                max_new_tokens,
                threads,
                timeout_seconds,
                progress,
                quiet,
                json_out,
            } => {
                ensure_supported_mac_device(explicit_device_label, "mac bitnet-warm")?;
                run_bitnet_warm(BitnetWarmRun {
                    model_id: &model_id,
                    cache_dir,
                    model_path,
                    tokenizer,
                    prompts,
                    profiles,
                    max_new_tokens,
                    threads,
                    timeout_seconds,
                    progress,
                    quiet,
                    json_out,
                })
                .await
            }
            MacAction::BitnetChatGate {
                warm_receipt,
                failure_receipt,
                streaming_receipt,
                json,
                json_out,
            } => {
                ensure_supported_mac_device(explicit_device_label, "mac bitnet-chat-gate")?;
                run_bitnet_chat_gate(
                    &warm_receipt,
                    failure_receipt.as_deref(),
                    streaming_receipt.as_deref(),
                    json_out,
                    json,
                )
            }
            MacAction::BitnetBenchmark {
                model_id,
                cache_dir,
                model_path,
                tokenizer,
                one_shot_prompt,
                max_new_tokens,
                threads,
                progress,
                quiet,
                json_out,
            } => {
                ensure_supported_mac_device(explicit_device_label, "mac bitnet-benchmark")?;
                run_bitnet_benchmark(BitnetBenchmarkRun {
                    model_id: &model_id,
                    cache_dir,
                    model_path,
                    tokenizer,
                    one_shot_prompt,
                    max_new_tokens,
                    threads,
                    progress,
                    quiet,
                    json_out,
                })
                .await
            }
            MacAction::Doctor { model_id, cache_dir, max_new_tokens, threads, json_out } => {
                ensure_supported_mac_device(explicit_device_label, "mac doctor")?;
                run_doctor(&model_id, cache_dir, max_new_tokens, threads, json_out).await
            }
            MacAction::Serve {
                model_id,
                cache_dir,
                device,
                host,
                port,
                strict,
                stream,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                seed,
                receipt_dir,
            } => {
                ensure_supported_mac_serve_device(explicit_device_label)?;
                let defaults = MacServeGenerationDefaults {
                    max_new_tokens,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    seed,
                };
                let endpoint = MacServeEndpoint { host, port };
                run_mac_serve(
                    model_id,
                    cache_dir,
                    device,
                    endpoint,
                    strict,
                    stream,
                    defaults,
                    receipt_dir,
                )
                .await
            }
            MacAction::ServeCheck { url, completion, prompt, max_new_tokens, json_out } => {
                run_mac_serve_check(&url, completion, &prompt, max_new_tokens, json_out)
            }
            MacAction::Chat {
                model_family,
                prompts,
                stdin,
                interactive,
                model_id,
                cache_dir,
                system_prompt,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                seed,
                threads,
                stream,
                progress,
                quiet,
                turn_receipts,
                allocation_audit,
                metal_prefill_qkv_phase,
                json_out,
            } => {
                ensure_supported_mac_device(explicit_device_label, "mac chat")?;
                ensure_mac_chat_family_gate(model_family, &model_id)?;
                let prompt_input = resolve_mac_chat_prompts(prompts, stdin, interactive, quiet)?;
                let chat = run_chat_session(
                    &model_id,
                    cache_dir,
                    prompt_input.prompts,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    seed,
                    threads,
                    stream,
                    progress,
                    quiet,
                    turn_receipts,
                    prompt_input.interactive,
                    allocation_audit,
                    metal_prefill_qkv_phase,
                    json_out,
                );
                tokio::select! {
                    result = chat => result,
                    signal = tokio::signal::ctrl_c() => {
                        signal.context("failed to listen for Ctrl-C while running mac chat")?;
                        anyhow::bail!(
                            "mac chat interrupted by Ctrl-C before the aggregate session receipt completed; use /exit or EOF between prompts for a clean aggregate receipt"
                        );
                    }
                }
            }
            MacAction::Validate {
                model_id,
                cache_dir,
                corpus,
                corpus_repeat_runs,
                profile_set,
                max_new_tokens,
                threads,
                allocation_audit,
                metal_prefill_qkv_phase,
                progress,
                quiet,
                json_out,
            } => {
                let requested_backend =
                    ensure_supported_mac_validate_device(explicit_device_label)?;
                run_validate(MacValidateRun {
                    model_id: &model_id,
                    cache_dir,
                    requested_backend,
                    corpus,
                    corpus_repeat_runs,
                    profile_set,
                    max_new_tokens,
                    threads,
                    allocation_audit,
                    metal_prefill_qkv_phase,
                    progress,
                    quiet,
                    json_out,
                })
                .await
            }
            MacAction::Benchmark {
                model_id,
                cache_dir,
                profile,
                threads,
                allocation_audit,
                progress,
                quiet,
                json_out,
            } => {
                let requested_backend =
                    ensure_supported_mac_benchmark_device(explicit_device_label)?;
                run_benchmark(MacBenchmarkRun {
                    model_id: &model_id,
                    cache_dir,
                    requested_backend,
                    profiles: profile,
                    threads,
                    allocation_audit,
                    progress,
                    quiet,
                    json_out,
                })
                .await
            }
            MacAction::BitnetProof {
                model,
                tokenizer_authority,
                accepted_artifact,
                proof_receipt,
                prompt,
                max_new_tokens,
                strict,
                json_out,
            } => {
                ensure_supported_mac_device(explicit_device_label, "mac bitnet-proof")?;
                run_bitnet_proof_preflight(
                    model,
                    tokenizer_authority,
                    accepted_artifact,
                    proof_receipt,
                    prompt,
                    max_new_tokens,
                    strict,
                    json_out,
                )
            }
            MacAction::ReceiptsCheck { path, regression_baseline, json } => {
                run_receipts_check(&path, regression_baseline.as_deref(), json)
            }
            MacAction::Regression { path, baseline, fail_on_drift, json } => {
                run_regression_check(&path, &baseline, fail_on_drift, json)
            }
        }
    }
}

fn resolve_mac_question(positional: Option<String>, flag: Option<String>) -> Result<String> {
    match (positional, flag) {
        (Some(_), Some(_)) => anyhow::bail!(
            "provide the Mac question either positionally, e.g. `bitnet mac ask \"What is 2+2?\"`, or with --question, not both"
        ),
        (Some(question), None) | (None, Some(question)) if !question.trim().is_empty() => {
            Ok(question)
        }
        _ => anyhow::bail!(
            "missing Mac question; pass it positionally, e.g. `bitnet mac ask \"What is 2+2?\"`, or with --question"
        ),
    }
}

fn run_status(cache_dir: Option<PathBuf>, json_out: PathBuf, json: bool) -> Result<()> {
    let catalog = model_cache::apple_m4_models_catalog_json(cache_dir.clone())
        .context("failed to build Apple M4 model catalog for mac status")?;
    let bitnet = bitnet_mac_ask_readiness_json(cache_dir);
    let receipt = apple_m4_inference_status_receipt(&json_out, catalog, bitnet);
    validate_mac_receipt_value(&json_out, &receipt)?;
    write_json_receipt(&json_out, &receipt)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&receipt)?);
    } else {
        print_mac_status_summary(&receipt, &json_out);
    }
    Ok(())
}

fn apple_m4_inference_status_receipt(
    json_out: &Path,
    catalog: serde_json::Value,
    bitnet: serde_json::Value,
) -> serde_json::Value {
    let rows = catalog["rows"].as_array().cloned().unwrap_or_default();
    let dense_rows = rows
        .iter()
        .filter(|row| matches!(row["state"].as_str(), Some("default" | "supported")))
        .cloned()
        .collect::<Vec<_>>();
    let dense_ready =
        dense_rows.iter().filter(|row| row["cache_state"].as_str() == Some("ready")).count();
    let supported_dense_model_ids = dense_rows
        .iter()
        .filter_map(|row| row["id"].as_str().map(ToOwned::to_owned))
        .collect::<Vec<_>>();
    let default_model_id = catalog["default_model_id"]
        .as_str()
        .unwrap_or(model_cache::M4_SLM_RUNTIME_MODEL_ID)
        .to_string();
    let default_row = rows
        .iter()
        .find(|row| row["id"].as_str() == Some(default_model_id.as_str()))
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let default_cache_ready = default_row["cache_state"].as_str() == Some("ready");
    let disk_available = catalog["disk"]["available"].clone();
    let disk_low = catalog["disk"]["low_disk"].clone();
    let recommended_first_model_id = catalog["disk"]["recommended_first_model_id"].clone();
    let commands = serde_json::json!({
        "models": "bitnet mac models",
        "status": "bitnet mac status",
        "report_refresh": "bitnet mac report-refresh",
        "regression_dashboard": "bitnet mac regression-dashboard",
        "fetch_default": format!("bitnet model fetch {default_model_id}"),
        "verify_default": format!("bitnet model verify {default_model_id}"),
        "ask_default": "bitnet mac ask \"What is 2+2?\"",
        "chat_dense": "bitnet mac chat --prompt \"What is 2+2?\" --prompt \"Name the capital of France.\"",
        "serve_dense": "bitnet mac serve --host 127.0.0.1 --port 8080",
        "doctor": "bitnet mac doctor",
        "smoke_dense": "bitnet mac smoke",
        "smoke_bitnet": "bitnet mac smoke --model-family bitnet",
        "regression": "bitnet mac regression <receipt.json> --baseline <baseline.json>",
        "bitnet_ask": bitnet["commands"]["ask_cached_model"].clone(),
        "bitnet_warm": bitnet["commands"]["warm_cached_model"].clone(),
        "bitnet_chat_gate": "bitnet mac bitnet-chat-gate --warm-receipt <warm.json> --failure-receipt <failure.json> --streaming-receipt <streaming.json>",
    });
    serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "apple_m4_inference_status",
        "generated_at": chrono::Utc::now().to_rfc3339(),
        "operator_command": "mac status",
        "status": "ok",
        "receipt_path": json_out,
        "requested_backend": APPLE_M4_CPU_NEON,
        "selected_backend": APPLE_M4_CPU_NEON,
        "runtime_api": "cpu",
        "fallback_used": false,
        "machine": {
            "id": "apple-m4-mac-mini",
            "scope": "local operator readiness summary",
        },
        "disk": {
            "available": disk_available,
            "low_disk": disk_low,
            "recommendation": catalog["disk"]["recommendation"].clone(),
            "guidance": catalog["disk"]["guidance"].clone(),
            "recommended_first_model_id": recommended_first_model_id,
        },
        "dense_slm": {
            "default_model_id": default_model_id,
            "supported_model_ids": supported_dense_model_ids,
            "supported_model_count": dense_rows.len(),
            "ready_model_count": dense_ready,
            "default_cache_ready": default_cache_ready,
            "default_row": default_row,
            "ask_enabled": true,
            "chat_enabled": true,
            "serve_enabled": true,
            "claim_boundary": {
                "dense_slm_only": true,
                "supported_qwen_models_only": true,
                "broad_model_quality_claim": false,
                "broad_performance_claim": false,
            },
        },
        "bitnet": {
            "readiness": bitnet,
            "ask_enabled": true,
            "warm_enabled": true,
            "chat_enabled": false,
            "serve_enabled": false,
            "claim_boundary": bitnet_mac_ask_readiness_claim_boundary(),
        },
        "report_inventory": apple_m4_report_inventory_json(),
        "commands": commands,
        "claim_boundary": {
            "status_only": true,
            "no_live_model_run": true,
            "fallback_used": false,
            "dense_slm_and_bitnet_evidence_separated": true,
            "bitnet_chat_enabled": false,
            "bitnet_serve_enabled": false,
            "full_metal_inference_claimed": false,
            "qk256_apple_claimed": false,
            "neural_engine_execution_claimed": false,
            "mpsgraph_inference_claimed": false,
            "macbook_evidence": false,
            "broad_apple_silicon_claim": false,
            "broad_performance_claim": false,
            "speedup_claim": false,
        },
    })
}

fn print_mac_status_summary(receipt: &serde_json::Value, json_out: &Path) {
    let dense = &receipt["dense_slm"];
    let disk = &receipt["disk"];
    let bitnet = &receipt["bitnet"];
    println!("Apple M4 inference status: {}", receipt["status"].as_str().unwrap_or("unknown"));
    println!(
        "Disk: available={}, low_disk={}, recommendation={}",
        disk["available"].as_str().unwrap_or("unknown"),
        disk["low_disk"]
            .as_bool()
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        disk["recommendation"].as_str().unwrap_or("unknown")
    );
    println!(
        "Dense SLM: default={}, ready={}/{}, ask=true, chat=true, serve=true",
        dense["default_model_id"].as_str().unwrap_or("<unknown>"),
        dense["ready_model_count"].as_u64().unwrap_or(0),
        dense["supported_model_count"].as_u64().unwrap_or(0)
    );
    println!(
        "BitNet: ask={}, warm={}, chat=false, serve=false, ready={}",
        bitnet["ask_enabled"].as_bool().unwrap_or(false),
        bitnet["warm_enabled"].as_bool().unwrap_or(false),
        bitnet["readiness"]["ready"].as_bool().unwrap_or(false)
    );
    println!("Next: {}", receipt["commands"]["models"].as_str().unwrap_or("bitnet mac models"));
    println!("Receipt: {}", json_out.display());
    println!(
        "Claim boundary: status only; no live model run, no BitNet chat/serve, no full Metal/QK256/Neural Engine/MPSGraph/MacBook/broad performance claim."
    );
}

fn apple_m4_report_inventory_json() -> serde_json::Value {
    let root = Path::new(APPLE_M4_REPORT_ROOT);
    serde_json::json!({
        "root": root,
        "dense_slm_eval_v2": latest_matching_report(root, "slm-eval-v2", "summary.json"),
        "dense_slm_benchmark_v2": latest_matching_report(root, "slm-benchmark-v2", "summary.json"),
        "bitnet_eval": latest_matching_report(root, "bitnet-eval", "answer-corpus.json"),
        "bitnet_benchmark": latest_matching_report(root, "bitnet-benchmark", "summary.json"),
        "bitnet_variable_warm": latest_matching_report(root, "bitnet-productization", "variable-warm-session.json"),
    })
}

fn latest_matching_report(root: &Path, segment: &str, filename: &str) -> Option<String> {
    matching_reports(root, segment, filename).pop().map(|path| path.to_string_lossy().to_string())
}

fn matching_reports(root: &Path, segment: &str, filename: &str) -> Vec<PathBuf> {
    let mut matches = Vec::new();
    collect_matching_reports(root, segment, filename, &mut matches);
    matches.sort();
    matches
}

fn collect_matching_reports(root: &Path, segment: &str, filename: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_matching_reports(&path, segment, filename, out);
        } else if path.file_name().and_then(|name| name.to_str()) == Some(filename)
            && path.components().any(|component| component.as_os_str() == segment)
        {
            out.push(path);
        }
    }
}

fn run_report_refresh_manifest(root: PathBuf, json_out: PathBuf, json: bool) -> Result<()> {
    let receipt = apple_m4_report_refresh_manifest_receipt(&root, &json_out);
    validate_mac_receipt_value(&json_out, &receipt)?;
    write_json_receipt(&json_out, &receipt)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&receipt)?);
    } else {
        print_report_refresh_manifest_summary(&receipt, &json_out);
    }
    Ok(())
}

fn apple_m4_report_refresh_manifest_receipt(root: &Path, json_out: &Path) -> serde_json::Value {
    let families = apple_m4_report_refresh_families(root);
    let report_count =
        families.iter().filter_map(|family| family["report_count"].as_u64()).sum::<u64>();
    let complete =
        families.iter().all(|family| family["report_count"].as_u64().unwrap_or_default() > 0);
    serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "apple_m4_report_refresh_manifest",
        "generated_at": chrono::Utc::now().to_rfc3339(),
        "operator_command": "mac report-refresh",
        "status": if complete { "ok" } else { "missing_reports" },
        "receipt_path": json_out,
        "requested_backend": APPLE_M4_CPU_NEON,
        "selected_backend": APPLE_M4_CPU_NEON,
        "runtime_api": "cpu",
        "fallback_used": false,
        "machine": {
            "id": "apple-m4-mac-mini",
            "scope": "committed report refresh manifest",
        },
        "report_root": root,
        "family_count": families.len(),
        "report_count": report_count,
        "refresh_modes": {
            "advisory_manifest": true,
            "nightly_manifest": true,
            "release_manifest": true,
            "generic_pr_ci_model_free": true,
            "generic_pr_ci_live_model_run": false,
            "model_downloads": false,
            "long_resident_soaks": false,
        },
        "families": families,
        "validation": {
            "manifest_command": "bitnet mac report-refresh --json",
            "manifest_receipt_check": "bitnet mac receipts-check target/apple-m4-inference-ops/report-refresh-manifest.json --json",
            "report_receipt_check": "bitnet mac receipts-check <report.json> --json",
            "regression_check": "bitnet mac regression <current-report.json> --baseline <baseline-report.json>",
        },
        "claim_boundary": {
            "manifest_only": true,
            "no_live_model_run": true,
            "no_model_download": true,
            "dense_slm_and_bitnet_evidence_separated": true,
            "bitnet_chat_enabled": false,
            "bitnet_serve_enabled": false,
            "full_metal_inference_claimed": false,
            "qk256_apple_claimed": false,
            "neural_engine_execution_claimed": false,
            "mpsgraph_inference_claimed": false,
            "macbook_evidence": false,
            "broad_apple_silicon_claim": false,
            "broad_model_quality_claim": false,
            "broad_performance_claim": false,
            "speedup_claim": false,
        },
    })
}

fn apple_m4_report_refresh_families(root: &Path) -> Vec<serde_json::Value> {
    [
        (
            "dense_slm_eval_v2",
            "dense_slm",
            "slm-eval-v2",
            "summary.json",
            "apple_m4_slm_eval_summary",
            "Seeded dense SLM quality reports.",
            "local/advisory/nightly: regenerate v2 dense eval reports with the seeded corpus, then validate committed summaries.",
        ),
        (
            "dense_slm_benchmark_v2",
            "dense_slm",
            "slm-benchmark-v2",
            "summary.json",
            "apple_m4_slm_benchmark_v2",
            "Dense SLM TTFT, throughput, memory, and resident-session benchmark reports.",
            "local/advisory/nightly: regenerate v2 dense benchmark summaries in release mode, then validate committed summaries.",
        ),
        (
            "bitnet_eval",
            "bitnet",
            "bitnet-eval",
            "answer-corpus.json",
            "bitnet_apple_m4_local_answer_corpus",
            "BitNet seeded quality and reference-vs-Rust reports.",
            "local/advisory/nightly: regenerate the BitNet answer corpus with the accepted GGUF/tokenizer, then validate the committed corpus receipt.",
        ),
        (
            "bitnet_benchmark",
            "bitnet",
            "bitnet-benchmark",
            "summary.json",
            "bitnet_apple_m4_benchmark_v1",
            "BitNet one-shot ask and fixed-warm benchmark reports.",
            "local/advisory/nightly: regenerate BitNet one-shot/fixed-warm benchmark summaries, then validate the committed benchmark receipt.",
        ),
        (
            "bitnet_variable_warm",
            "bitnet",
            "bitnet-productization",
            "variable-warm-session.json",
            "bitnet_apple_m4_warm_session",
            "BitNet variable-prompt warm-session productization receipts.",
            "local/advisory/nightly: rerun the variable warm-session proof before any BitNet chat or serve gate changes.",
        ),
    ]
    .into_iter()
    .map(
        |(
            id,
            evidence_family,
            path_segment,
            summary_filename,
            expected_artifact_kind,
            description,
            refresh_command_template,
        )| {
            apple_m4_report_refresh_family_json(
                root,
                id,
                evidence_family,
                path_segment,
                summary_filename,
                expected_artifact_kind,
                description,
                refresh_command_template,
            )
        },
    )
    .collect()
}

#[allow(clippy::too_many_arguments)]
fn apple_m4_report_refresh_family_json(
    root: &Path,
    id: &str,
    evidence_family: &str,
    path_segment: &str,
    summary_filename: &str,
    expected_artifact_kind: &str,
    description: &str,
    refresh_command_template: &str,
) -> serde_json::Value {
    let report_paths = matching_reports(root, path_segment, summary_filename);
    let reports = report_paths
        .iter()
        .map(|path| apple_m4_report_refresh_report_json(root, path, expected_artifact_kind))
        .collect::<Vec<_>>();
    let mut dates = std::collections::BTreeSet::new();
    let mut model_ids = std::collections::BTreeSet::new();
    let mut artifact_kinds = std::collections::BTreeSet::new();
    let mut fallback_free_count = 0_u64;
    let mut strict_cpu_neon_count = 0_u64;
    for report in &reports {
        if let Some(date) = report["date"].as_str() {
            dates.insert(date.to_string());
        }
        if let Some(model_id) = report["model_id"].as_str() {
            model_ids.insert(model_id.to_string());
        }
        if let Some(artifact_kind) = report["artifact_kind"].as_str() {
            artifact_kinds.insert(artifact_kind.to_string());
        }
        if report["fallback_used"].as_bool() == Some(false) {
            fallback_free_count = fallback_free_count.saturating_add(1);
        }
        if report["selected_backend"].as_str() == Some(APPLE_M4_CPU_NEON) {
            strict_cpu_neon_count = strict_cpu_neon_count.saturating_add(1);
        }
    }
    serde_json::json!({
        "id": id,
        "evidence_family": evidence_family,
        "description": description,
        "path_segment": path_segment,
        "summary_filename": summary_filename,
        "expected_artifact_kind": expected_artifact_kind,
        "refresh_tiers": ["advisory", "nightly", "release"],
        "generic_pr_ci": {
            "validate_committed_reports_only": true,
            "live_model_run": false,
            "model_downloads": false,
        },
        "refresh_command_template": refresh_command_template,
        "validation_commands": [
            "bitnet mac receipts-check <report.json> --json",
            "bitnet mac regression <current-report.json> --baseline <baseline-report.json>",
        ],
        "report_count": reports.len(),
        "latest_report": report_paths.last().map(|path| path.to_string_lossy().to_string()),
        "dates": dates.into_iter().collect::<Vec<_>>(),
        "model_ids": model_ids.into_iter().collect::<Vec<_>>(),
        "artifact_kinds": artifact_kinds.into_iter().collect::<Vec<_>>(),
        "fallback_free_count": fallback_free_count,
        "strict_cpu_neon_count": strict_cpu_neon_count,
        "reports": reports,
        "claim_boundary": {
            "dense_slm_evidence": evidence_family == "dense_slm",
            "bitnet_evidence": evidence_family == "bitnet",
            "evidence_families_mixed": false,
            "generic_pr_ci_live_model_run": false,
            "broad_model_quality_claim": false,
            "broad_performance_claim": false,
            "speedup_claim": false,
        },
    })
}

fn apple_m4_report_refresh_report_json(
    root: &Path,
    path: &Path,
    expected_artifact_kind: &str,
) -> serde_json::Value {
    let parsed = std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let date = apple_m4_report_date(root, path);
    match parsed {
        Some(receipt) => serde_json::json!({
            "path": path,
            "date": date,
            "parse_status": "ok",
            "artifact_kind": receipt["artifact_kind"].as_str(),
            "expected_artifact_kind": expected_artifact_kind,
            "artifact_kind_matches": receipt["artifact_kind"].as_str() == Some(expected_artifact_kind),
            "model_id": apple_m4_report_model_id(&receipt),
            "selected_backend": receipt_string(&receipt, "selected_backend"),
            "runtime_api": receipt_string(&receipt, "runtime_api"),
            "fallback_used": receipt_bool(&receipt, "fallback_used"),
            "prompt_count": apple_m4_report_prompt_count(&receipt),
            "generated_tokens": apple_m4_report_generated_tokens(&receipt),
        }),
        None => serde_json::json!({
            "path": path,
            "date": date,
            "parse_status": "unreadable",
            "artifact_kind": serde_json::Value::Null,
            "expected_artifact_kind": expected_artifact_kind,
            "artifact_kind_matches": false,
            "model_id": serde_json::Value::Null,
            "selected_backend": serde_json::Value::Null,
            "runtime_api": serde_json::Value::Null,
            "fallback_used": serde_json::Value::Null,
            "prompt_count": serde_json::Value::Null,
            "generated_tokens": serde_json::Value::Null,
        }),
    }
}

fn apple_m4_report_date(root: &Path, path: &Path) -> Option<String> {
    path.strip_prefix(root)
        .ok()
        .and_then(|relative| relative.components().next())
        .and_then(|component| component.as_os_str().to_str())
        .map(ToOwned::to_owned)
}

fn apple_m4_report_model_id(receipt: &serde_json::Value) -> Option<String> {
    for segments in [
        &["model_id"][..],
        &["model", "id"][..],
        &["model_cache", "id"][..],
        &["model", "model_id"][..],
    ] {
        if let Some(model_id) = json_value_at(receipt, segments).as_str() {
            return Some(model_id.to_string());
        }
    }
    if receipt["model"]["family"].as_str() == Some("bitnet")
        && receipt["model"]["sha256"].as_str() == Some(BITNET_M4_EXPECTED_MODEL_SHA256)
    {
        return Some(BITNET_M4_MODEL_ID.to_string());
    }
    None
}

fn apple_m4_report_prompt_count(receipt: &serde_json::Value) -> Option<u64> {
    receipt["prompt_count"].as_u64().or_else(|| {
        receipt["corpus"]["case_count"]
            .as_u64()
            .or_else(|| receipt["quality_summary"]["total"].as_u64())
    })
}

fn apple_m4_report_generated_tokens(receipt: &serde_json::Value) -> Option<u64> {
    receipt["generated_tokens"]
        .as_u64()
        .or_else(|| receipt["evidence"]["generated_tokens_total"].as_u64())
}

fn print_report_refresh_manifest_summary(receipt: &serde_json::Value, json_out: &Path) {
    println!(
        "Apple M4 report refresh manifest: {}",
        receipt["status"].as_str().unwrap_or("unknown")
    );
    println!("Report root: {}", receipt["report_root"].as_str().unwrap_or(APPLE_M4_REPORT_ROOT));
    println!(
        "Families: {}, reports: {}",
        receipt["family_count"].as_u64().unwrap_or(0),
        receipt["report_count"].as_u64().unwrap_or(0)
    );
    if let Some(families) = receipt["families"].as_array() {
        for family in families {
            println!(
                "- {}: reports={}, latest={}",
                family["id"].as_str().unwrap_or("<unknown>"),
                family["report_count"].as_u64().unwrap_or(0),
                family["latest_report"].as_str().unwrap_or("<missing>")
            );
        }
    }
    println!("Receipt: {}", json_out.display());
    println!(
        "Claim boundary: manifest only; no live model run, no model downloads, dense SLM and BitNet evidence stay separate."
    );
}

fn run_regression_dashboard(
    root: PathBuf,
    json_out: PathBuf,
    markdown_out: PathBuf,
    json: bool,
) -> Result<()> {
    let receipt = apple_m4_regression_dashboard_receipt(&root, &json_out, &markdown_out);
    validate_mac_receipt_value(&json_out, &receipt)?;
    let markdown = apple_m4_regression_dashboard_markdown(&receipt);
    write_json_receipt(&json_out, &receipt)?;
    if let Some(parent) = markdown_out.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("failed to create dashboard directory {}", parent.display())
        })?;
    }
    std::fs::write(&markdown_out, markdown).with_context(|| {
        format!("failed to write regression dashboard {}", markdown_out.display())
    })?;
    if json {
        println!("{}", serde_json::to_string_pretty(&receipt)?);
    } else {
        print_regression_dashboard_summary(&receipt, &json_out, &markdown_out);
    }
    Ok(())
}

fn apple_m4_regression_dashboard_receipt(
    root: &Path,
    json_out: &Path,
    markdown_out: &Path,
) -> serde_json::Value {
    let manifest_families = apple_m4_report_refresh_families(root);
    let families =
        manifest_families.iter().map(apple_m4_regression_dashboard_family_json).collect::<Vec<_>>();
    let group_count =
        families.iter().filter_map(|family| family["group_count"].as_u64()).sum::<u64>();
    let comparable_group_count =
        families.iter().filter_map(|family| family["comparable_group_count"].as_u64()).sum::<u64>();
    let report_count =
        families.iter().filter_map(|family| family["report_count"].as_u64()).sum::<u64>();
    serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "apple_m4_regression_dashboard",
        "generated_at": chrono::Utc::now().to_rfc3339(),
        "operator_command": "mac regression-dashboard",
        "status": "ok",
        "receipt_path": json_out,
        "markdown_path": markdown_out,
        "requested_backend": APPLE_M4_CPU_NEON,
        "selected_backend": APPLE_M4_CPU_NEON,
        "runtime_api": "cpu",
        "fallback_used": false,
        "machine": {
            "id": "apple-m4-mac-mini",
            "scope": "committed report regression dashboard",
        },
        "report_root": root,
        "report_count": report_count,
        "family_count": families.len(),
        "group_count": group_count,
        "comparable_group_count": comparable_group_count,
        "dashboard_contract": {
            "model_free": true,
            "committed_reports_only": true,
            "matching_requires_same_evidence_family": true,
            "matching_requires_same_model_id": true,
            "matching_requires_same_model_sha256": true,
            "matching_requires_same_tokenizer_authority": true,
            "matching_requires_same_artifact_kind": true,
            "matching_requires_same_backend": true,
            "matching_requires_fallback_false": true,
        },
        "families": families,
        "claim_boundary": {
            "dashboard_only": true,
            "no_live_model_run": true,
            "no_model_download": true,
            "dense_slm_and_bitnet_evidence_separated": true,
            "bitnet_chat_enabled": false,
            "bitnet_serve_enabled": false,
            "full_metal_inference_claimed": false,
            "qk256_apple_claimed": false,
            "neural_engine_execution_claimed": false,
            "mpsgraph_inference_claimed": false,
            "macbook_evidence": false,
            "broad_apple_silicon_claim": false,
            "broad_model_quality_claim": false,
            "broad_performance_claim": false,
            "speedup_claim": false,
        },
    })
}

fn apple_m4_regression_dashboard_family_json(family: &serde_json::Value) -> serde_json::Value {
    let family_id = family["id"].as_str().unwrap_or("<unknown>");
    let evidence_family = family["evidence_family"].as_str().unwrap_or("<unknown>");
    let expected_artifact_kind = family["expected_artifact_kind"].as_str().unwrap_or("<unknown>");
    let mut groups: std::collections::BTreeMap<String, Vec<serde_json::Value>> =
        std::collections::BTreeMap::new();
    if let Some(reports) = family["reports"].as_array() {
        for report in reports {
            let Some(path_text) = report["path"].as_str() else {
                continue;
            };
            let path = Path::new(path_text);
            let Ok(bytes) = std::fs::read(path) else {
                continue;
            };
            let Ok(receipt) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
                continue;
            };
            let identity = apple_m4_dashboard_report_identity_json(&receipt, evidence_family);
            let group_key = apple_m4_dashboard_group_key(family_id, &identity);
            groups.entry(group_key).or_default().push(serde_json::json!({
                "path": path,
                "date": report["date"].clone(),
                "artifact_kind": receipt["artifact_kind"].as_str(),
                "identity": identity,
                "metrics": apple_m4_dashboard_metrics_json(&receipt),
            }));
        }
    }
    let mut dashboard_groups = Vec::new();
    let mut comparable_group_count = 0_u64;
    for (group_key, mut reports) in groups {
        reports.sort_by_key(|report| report["path"].as_str().unwrap_or_default().to_string());
        let report_count = reports.len();
        let latest = reports.last().cloned().unwrap_or(serde_json::Value::Null);
        let baseline = if report_count > 1 {
            comparable_group_count = comparable_group_count.saturating_add(1);
            reports.get(report_count - 2).cloned().unwrap_or(serde_json::Value::Null)
        } else {
            serde_json::Value::Null
        };
        let comparison_status = if report_count > 1 { "ready" } else { "insufficient_history" };
        let latest_path = latest["path"].as_str().unwrap_or("<missing>");
        let baseline_path = baseline["path"].as_str().unwrap_or(latest_path);
        dashboard_groups.push(serde_json::json!({
            "group_key": group_key,
            "evidence_family": evidence_family,
            "expected_artifact_kind": expected_artifact_kind,
            "model_id": latest["identity"]["model_id"].clone(),
            "model_sha256": latest["identity"]["model_sha256"].clone(),
            "tokenizer_authority": latest["identity"]["tokenizer_authority"].clone(),
            "selected_backend": latest["identity"]["selected_backend"].clone(),
            "runtime_api": latest["identity"]["runtime_api"].clone(),
            "fallback_used": latest["identity"]["fallback_used"].clone(),
            "report_count": report_count,
            "comparison_status": comparison_status,
            "latest_report": latest_path,
            "baseline_report": if report_count > 1 { serde_json::Value::String(baseline_path.to_string()) } else { serde_json::Value::Null },
            "regression_command": format!("bitnet mac regression {latest_path} --baseline {baseline_path}"),
            "latest_metrics": latest["metrics"].clone(),
            "reports": reports,
            "claim_boundary": {
                "dense_slm_evidence": evidence_family == "dense_slm",
                "bitnet_evidence": evidence_family == "bitnet",
                "evidence_families_mixed": false,
                "broad_model_quality_claim": false,
                "broad_performance_claim": false,
                "speedup_claim": false,
            },
        }));
    }
    let report_count = family["report_count"].as_u64().unwrap_or_default();
    serde_json::json!({
        "id": family_id,
        "evidence_family": evidence_family,
        "expected_artifact_kind": expected_artifact_kind,
        "report_count": report_count,
        "group_count": dashboard_groups.len(),
        "comparable_group_count": comparable_group_count,
        "groups": dashboard_groups,
        "claim_boundary": {
            "dense_slm_evidence": evidence_family == "dense_slm",
            "bitnet_evidence": evidence_family == "bitnet",
            "evidence_families_mixed": false,
            "generic_pr_ci_live_model_run": false,
            "broad_model_quality_claim": false,
            "broad_performance_claim": false,
            "speedup_claim": false,
        },
    })
}

fn apple_m4_dashboard_group_key(family_id: &str, identity: &serde_json::Value) -> String {
    format!(
        "{}::{}::{}::{}::{}",
        family_id,
        identity["model_id"].as_str().unwrap_or("<unknown>"),
        identity["model_sha256"].as_str().unwrap_or("<unknown>"),
        identity["tokenizer_authority"].as_str().unwrap_or("<unknown>"),
        identity["selected_backend"].as_str().unwrap_or("<unknown>")
    )
}

fn apple_m4_dashboard_report_identity_json(
    receipt: &serde_json::Value,
    evidence_family: &str,
) -> serde_json::Value {
    serde_json::json!({
        "evidence_family": evidence_family,
        "artifact_kind": receipt["artifact_kind"].as_str(),
        "model_id": apple_m4_report_model_id(receipt).unwrap_or_else(|| "<unknown>".to_string()),
        "model_sha256": apple_m4_report_model_sha256(receipt).unwrap_or_else(|| "<unknown>".to_string()),
        "tokenizer_authority": apple_m4_report_tokenizer_authority(receipt).unwrap_or_else(|| "<unknown>".to_string()),
        "prompt_template": receipt["prompt_template"].as_str().unwrap_or("<not-recorded>"),
        "selected_backend": receipt_string(receipt, "selected_backend").unwrap_or_else(|| "<unknown>".to_string()),
        "runtime_api": receipt_string(receipt, "runtime_api").unwrap_or_else(|| "<unknown>".to_string()),
        "fallback_used": receipt_bool(receipt, "fallback_used").unwrap_or(true),
    })
}

fn apple_m4_report_model_sha256(receipt: &serde_json::Value) -> Option<String> {
    for segments in [
        &["model", "sha256"][..],
        &["model_cache", "sha256"][..],
        &["model", "answer_ready", "sha256"][..],
    ] {
        if let Some(sha256) = json_value_at(receipt, segments).as_str() {
            return Some(sha256.to_string());
        }
    }
    None
}

fn apple_m4_report_tokenizer_authority(receipt: &serde_json::Value) -> Option<String> {
    if let Some(authority) = receipt["tokenizer"]["authority"].as_str() {
        return Some(authority.to_string());
    }
    if let Some(pre) = receipt["tokenizer"]["pretokenizer_authority"].as_str() {
        return Some(pre.to_string());
    }
    if let Some(pre) = receipt["model_cache"]["tokenizer_pre"].as_str() {
        return Some(pre.to_string());
    }
    if let Some(sha256) = receipt["tokenizer"]["sha256"].as_str() {
        let source = receipt["tokenizer"]["source"].as_str().unwrap_or("tokenizer");
        return Some(format!("{source}:{sha256}"));
    }
    if let Some(sha256) = receipt["tokenizer"]["authority"]["sha256"].as_str() {
        let source = receipt["tokenizer"]["authority"]["source"].as_str().unwrap_or("tokenizer");
        let pre = receipt["tokenizer"]["authority"]["ggml_pre"].as_str().unwrap_or("unknown-pre");
        return Some(format!("{source}:{pre}:{sha256}"));
    }
    None
}

fn apple_m4_dashboard_metrics_json(receipt: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "prompt_count": apple_m4_report_prompt_count(receipt),
        "generated_tokens": apple_m4_report_generated_tokens(receipt),
        "quality": {
            "cases_total": receipt["accuracy"]["cases_total"].as_u64()
                .or_else(|| receipt["quality_summary"]["total"].as_u64())
                .or_else(|| receipt["speed"]["counts"]["prompt_count"].as_u64()),
            "cases_passed": receipt["accuracy"]["cases_passed"].as_u64()
                .or_else(|| receipt["quality_summary"]["passed"].as_u64()),
            "quality_passed": receipt["quality_summary"]["passed"].as_bool()
                .or_else(|| receipt["stability"]["quality_passed"].as_bool()),
            "failed": receipt["quality_summary"]["failed"].as_u64(),
        },
        "speed": {
            "ttft_ms_p50": receipt["speed"]["ttft_ms_p50"].as_f64()
                .or_else(|| receipt["speed"]["ttft_ms"]["p50"].as_f64())
                .or_else(|| receipt["speed"]["timing"]["time_to_first_token_ms"]["p50_ms"].as_f64()),
            "input_tok_s_p50": receipt["speed"]["input_tok_s_p50"].as_f64()
                .or_else(|| receipt["speed"]["input_tok_s"]["p50"].as_f64()),
            "output_tok_s_p50": receipt["speed"]["output_tok_s_p50"].as_f64()
                .or_else(|| receipt["speed"]["output_tok_s"]["p50"].as_f64()),
            "decode_tok_s_p50": receipt["speed"]["decode_tok_s_p50"].as_f64()
                .or_else(|| receipt["speed"]["decode_tok_s"]["p50"].as_f64())
                .or_else(|| receipt["speed"]["timing"]["steady_decode_tok_s"]["p50"].as_f64()),
            "total_wall_ms_p50": receipt["speed"]["total_wall_ms_p50"].as_f64()
                .or_else(|| receipt["speed"]["total_wall_ms"]["p50"].as_f64())
                .or_else(|| receipt["speed"]["timing"]["total_session_ms"].as_f64()),
        },
        "memory": {
            "peak_memory_mb_p50": receipt["memory"]["peak_memory_mb"]["p50"].as_f64()
                .or_else(|| receipt["memory"]["peak_memory_mb"].as_f64()),
            "resident_memory_bytes": receipt["memory"]["resident_memory_bytes"].as_u64(),
            "memory_drift_mb_p50": receipt["memory"]["memory_drift_mb"]["p50"].as_f64()
                .or_else(|| receipt["stability"]["memory_drift_mb"].as_f64()),
        },
    })
}

fn apple_m4_regression_dashboard_markdown(receipt: &serde_json::Value) -> String {
    let mut out = String::new();
    out.push_str("# Apple M4 Inference Regression Dashboard\n\n");
    out.push_str("Model-free dashboard generated from committed Apple M4 receipts only.\n\n");
    out.push_str("| Family | Evidence | Model | Reports | Status | Latest | Baseline |\n");
    out.push_str("|---|---|---|---:|---|---|---|\n");
    if let Some(families) = receipt["families"].as_array() {
        for family in families {
            if let Some(groups) = family["groups"].as_array() {
                for group in groups {
                    out.push_str(&format!(
                        "| `{}` | `{}` | `{}` | {} | `{}` | `{}` | `{}` |\n",
                        family["id"].as_str().unwrap_or("<unknown>"),
                        group["evidence_family"].as_str().unwrap_or("<unknown>"),
                        group["model_id"].as_str().unwrap_or("<unknown>"),
                        group["report_count"].as_u64().unwrap_or(0),
                        group["comparison_status"].as_str().unwrap_or("<unknown>"),
                        group["latest_report"].as_str().unwrap_or("<missing>"),
                        group["baseline_report"].as_str().unwrap_or("<none>")
                    ));
                }
            }
        }
    }
    out.push_str("\nClaim boundary: dashboard only; no live model run, no model download, no BitNet chat/serve, no full Metal, QK256, Neural Engine, MPSGraph, MacBook, broad quality, broad performance, or speedup claim.\n");
    out
}

fn print_regression_dashboard_summary(
    receipt: &serde_json::Value,
    json_out: &Path,
    markdown_out: &Path,
) {
    println!("Apple M4 regression dashboard: {}", receipt["status"].as_str().unwrap_or("unknown"));
    println!(
        "Families: {}, groups: {}, comparable: {}, reports: {}",
        receipt["family_count"].as_u64().unwrap_or(0),
        receipt["group_count"].as_u64().unwrap_or(0),
        receipt["comparable_group_count"].as_u64().unwrap_or(0),
        receipt["report_count"].as_u64().unwrap_or(0)
    );
    println!("Receipt: {}", json_out.display());
    println!("Markdown: {}", markdown_out.display());
    println!(
        "Claim boundary: dashboard only; no live model run, no model downloads, dense SLM and BitNet evidence stay separate."
    );
}

struct MacChatPrompts {
    prompts: Vec<String>,
    interactive: bool,
}

fn resolve_mac_chat_prompts(
    mut prompts: Vec<String>,
    read_stdin: bool,
    interactive: bool,
    quiet: bool,
) -> Result<MacChatPrompts> {
    if read_stdin && interactive {
        anyhow::bail!("mac chat --stdin and --interactive cannot be used together");
    }
    let stdin_is_terminal = io::stdin().is_terminal();
    let should_collect_interactively =
        interactive || (stdin_is_terminal && prompts.is_empty() && !read_stdin);
    if should_collect_interactively {
        let stdin = io::stdin();
        let mut reader = stdin.lock();
        prompts.extend(collect_mac_chat_interactive_prompts(
            &mut reader,
            stdin_is_terminal && !quiet,
        )?);
    }
    let should_read_stdin =
        !should_collect_interactively && (read_stdin || (!stdin_is_terminal && prompts.is_empty()));
    if should_read_stdin {
        let mut input = String::new();
        io::stdin()
            .read_to_string(&mut input)
            .context("failed to read Mac chat prompts from stdin")?;
        prompts.extend(
            input.lines().map(str::trim).filter(|line| !line.is_empty()).map(ToOwned::to_owned),
        );
    }
    prompts.retain(|prompt| !prompt.trim().is_empty());
    if prompts.len() < 2 {
        anyhow::bail!(
            "mac chat requires at least two prompts for a resident session; pass --prompt multiple times, pipe one prompt per line with --stdin, or use --interactive then finish with /exit or EOF. For one question use `bitnet mac ask \"What is 2+2?\"`."
        );
    }
    Ok(MacChatPrompts { prompts, interactive: should_collect_interactively })
}

fn collect_mac_chat_interactive_prompts<R: BufRead>(
    reader: &mut R,
    show_prompt: bool,
) -> Result<Vec<String>> {
    let mut prompts = Vec::new();
    if show_prompt {
        eprintln!(
            "mac chat: enter prompts, then /exit, /quit, or EOF to run the resident session. Ctrl-C cancels."
        );
    }
    loop {
        if show_prompt {
            eprint!("mac> ");
            io::stderr().flush().context("failed to flush mac chat prompt")?;
        }
        let mut line = String::new();
        let bytes = reader.read_line(&mut line).context("failed to read Mac chat prompt")?;
        if bytes == 0 {
            break;
        }
        match parse_repl_input(&line) {
            None => {}
            Some(ReplInput::Exit) => break,
            Some(ReplInput::Help) => {
                if show_prompt {
                    eprintln!(
                        "mac chat commands: /exit or /quit runs collected prompts; /clear clears prompts; /metrics shows prompt count."
                    );
                }
            }
            Some(ReplInput::Clear) => {
                prompts.clear();
                if show_prompt {
                    eprintln!("mac chat: cleared collected prompts.");
                }
            }
            Some(ReplInput::Metrics) => {
                if show_prompt {
                    eprintln!("mac chat: collected {} prompt(s).", prompts.len());
                }
            }
            Some(ReplInput::Message(prompt)) => prompts.push(prompt),
        }
    }
    Ok(prompts)
}

fn ensure_mac_chat_family_gate(model_family: MacChatModelFamily, model_id: &str) -> Result<()> {
    if model_family == MacChatModelFamily::Bitnet
        || model_cache::is_apple_m4_bitnet_artifact_id(model_id)
    {
        anyhow::bail!(
            "BitNet Mac chat is disabled by M4-BITNET-PROD-004 until the receipt-backed chat gate passes. Required evidence: variable `bitnet mac bitnet-warm` receipt with repeated-prompt determinism, timeout/partial-failure receipt, streaming semantics receipt, and preserved chat/serve=false claim boundaries. Use `bitnet mac bitnet-chat-gate --warm-receipt <PATH> --failure-receipt <PATH> --streaming-receipt <PATH>` to write the gate receipt. Current allowed BitNet routes remain `bitnet mac ask` and `bitnet mac bitnet-warm`; dense SLM chat remains available with --model-family dense-slm."
        );
    }
    Ok(())
}

fn run_bitnet_chat_gate(
    warm_receipt: &Path,
    failure_receipt: Option<&Path>,
    streaming_receipt: Option<&Path>,
    json_out: PathBuf,
    json: bool,
) -> Result<()> {
    let warm = inspect_bitnet_chat_gate_warm_receipt(warm_receipt);
    let failure = inspect_bitnet_chat_gate_failure_receipt(failure_receipt);
    let streaming = inspect_bitnet_chat_gate_streaming_receipt(streaming_receipt);
    let requirements_passed = warm["passed"].as_bool() == Some(true)
        && warm["repeated_prompt_determinism_passed"].as_bool() == Some(true)
        && failure["passed"].as_bool() == Some(true)
        && failure["timeout_boundary_recorded"].as_bool() == Some(true)
        && streaming["passed"].as_bool() == Some(true);
    let status = if requirements_passed { "ready_to_enable" } else { "blocked" };
    let receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_apple_m4_chat_gate",
        "generated_at": chrono::Utc::now().to_rfc3339(),
        "operator_command": "mac bitnet-chat-gate",
        "status": status,
        "model_id": BITNET_M4_MODEL_ID,
        "requested_backend": APPLE_M4_CPU_NEON,
        "selected_backend": APPLE_M4_CPU_NEON,
        "runtime_api": "cpu",
        "fallback_used": false,
        "model": {
            "family": "bitnet",
            "id": BITNET_M4_MODEL_ID,
            "expected_sha256": BITNET_M4_EXPECTED_MODEL_SHA256,
            "quant_format": "I2_S",
        },
        "tokenizer": {
            "expected_sha256": BITNET_M4_EXPECTED_TOKENIZER_SHA256,
            "authority": "external_tokenizer_json",
            "pretokenizer_authority": "llama-bpe",
            "strict": true,
        },
        "prompt": {
            "template_family": BITNET_M4_PROMPT_TEMPLATE,
            "authority": "bitnetcpp-answer",
        },
        "requirements": {
            "variable_warm_session_receipt": warm,
            "timeout_failure_receipt": failure,
            "streaming_semantics_receipt": streaming,
        },
        "chat_enablement": {
            "gate_passed": requirements_passed,
            "chat_enabled": false,
            "serve_enabled": false,
            "next_step": if requirements_passed {
                "Open a separate route-enablement PR that consumes this gate receipt and proves BitNet chat receipts."
            } else {
                "Collect the missing BitNet warm, timeout/failure, and streaming-semantics receipts before enabling chat."
            },
        },
        "mac_bitnet_claim_boundary": {
            "bitnet_chat_gate": true,
            "bitnet_chat_enabled": false,
            "chat_enabled": false,
            "serve_enabled": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false,
            "speedup_claim": false,
        },
        "bitnet_quality_claimed": false,
        "broad_performance_claim": false,
        "speedup_claim": false,
    });
    validate_mac_receipt_value(&json_out, &receipt)?;
    write_json_receipt(&json_out, &receipt)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&receipt)?);
    } else if requirements_passed {
        println!("BitNet chat gate ready-to-enable receipt written: {}", json_out.display());
    }
    if !requirements_passed {
        anyhow::bail!(
            "BitNet chat gate is blocked; receipt written to {}. Missing evidence must be collected before `bitnet mac chat --model-family bitnet` can be enabled.",
            json_out.display()
        );
    }
    Ok(())
}

fn inspect_bitnet_chat_gate_warm_receipt(path: &Path) -> serde_json::Value {
    match read_json_receipt(path).and_then(|receipt| {
        let summary = validate_mac_receipt_value(path, &receipt)?;
        if summary.artifact_kind != "bitnet_apple_m4_warm_session" {
            anyhow::bail!("{} must be a bitnet_apple_m4_warm_session receipt", path.display());
        }
        let variable_prompts =
            receipt["bitnet_warm_prompt_source"]["variable_prompts"].as_bool() == Some(true);
        let determinism = receipt["determinism"]["checked"].as_bool() == Some(true)
            && receipt["determinism"]["passed"].as_bool() == Some(true);
        let chat_disabled =
            receipt["mac_bitnet_claim_boundary"]["chat_enabled"].as_bool() == Some(false);
        if !variable_prompts {
            anyhow::bail!("{} must record operator variable warm prompts", path.display());
        }
        if !determinism {
            anyhow::bail!("{} must record passing repeated-prompt determinism", path.display());
        }
        if !chat_disabled {
            anyhow::bail!("{} must preserve chat_enabled=false", path.display());
        }
        Ok(serde_json::json!({
            "required": true,
            "passed": true,
            "path": path,
            "artifact_kind": summary.artifact_kind,
            "variable_prompts": true,
            "repeated_prompt_determinism_passed": true,
            "chat_enabled": false,
        }))
    }) {
        Ok(value) => value,
        Err(error) => serde_json::json!({
            "required": true,
            "passed": false,
            "path": path,
            "repeated_prompt_determinism_passed": false,
            "error": error.to_string(),
        }),
    }
}

fn inspect_bitnet_chat_gate_failure_receipt(path: Option<&Path>) -> serde_json::Value {
    let Some(path) = path else {
        return serde_json::json!({
            "required": true,
            "passed": false,
            "path": null,
            "timeout_boundary_recorded": false,
            "error": "missing --failure-receipt",
        });
    };
    match read_json_receipt(path).and_then(|receipt| {
        let summary = validate_mac_receipt_value(path, &receipt)?;
        if summary.artifact_kind != "bitnet_apple_m4_warm_session_failure" {
            anyhow::bail!(
                "{} must be a bitnet_apple_m4_warm_session_failure receipt",
                path.display()
            );
        }
        if receipt["timeout_boundary"]["enforced"].as_bool().is_none()
            || receipt["timeout_boundary"]["reached"].as_bool().is_none()
            || receipt["failure"]["stage"].as_str().is_none_or(str::is_empty)
        {
            anyhow::bail!("{} must record timeout boundary and failure stage", path.display());
        }
        Ok(serde_json::json!({
            "required": true,
            "passed": true,
            "path": path,
            "artifact_kind": summary.artifact_kind,
            "failure_stage": receipt["failure"]["stage"],
            "timeout_boundary_recorded": true,
            "chat_enabled": false,
        }))
    }) {
        Ok(value) => value,
        Err(error) => serde_json::json!({
            "required": true,
            "passed": false,
            "path": path,
            "timeout_boundary_recorded": false,
            "error": error.to_string(),
        }),
    }
}

fn inspect_bitnet_chat_gate_streaming_receipt(path: Option<&Path>) -> serde_json::Value {
    let Some(path) = path else {
        return serde_json::json!({
            "required": true,
            "passed": false,
            "path": null,
            "error": "missing --streaming-receipt",
        });
    };
    match read_json_receipt(path).and_then(|receipt| {
        if receipt["artifact_kind"].as_str() != Some("bitnet_apple_m4_chat_streaming_semantics") {
            anyhow::bail!(
                "{} must be a bitnet_apple_m4_chat_streaming_semantics receipt",
                path.display()
            );
        }
        if receipt["requested_backend"].as_str() != Some(APPLE_M4_CPU_NEON)
            || receipt["selected_backend"].as_str() != Some(APPLE_M4_CPU_NEON)
            || receipt["runtime_api"].as_str() != Some("cpu")
            || receipt["fallback_used"].as_bool() != Some(false)
            || receipt["streaming_semantics"]["token_order_preserved"].as_bool() != Some(true)
            || receipt["streaming_semantics"]["final_receipt_exported"].as_bool() != Some(true)
        {
            anyhow::bail!(
                "{} must prove strict backend, fallback=false, token order, and final receipt export",
                path.display()
            );
        }
        Ok(serde_json::json!({
            "required": true,
            "passed": true,
            "path": path,
            "artifact_kind": "bitnet_apple_m4_chat_streaming_semantics",
        }))
    }) {
        Ok(value) => value,
        Err(error) => serde_json::json!({
            "required": true,
            "passed": false,
            "path": path,
            "error": error.to_string(),
        }),
    }
}

fn ensure_supported_mac_device(explicit_device_label: Option<&str>, command: &str) -> Result<()> {
    let Some(label) = explicit_device_label else {
        return Ok(());
    };
    if label == APPLE_M4_CPU_NEON {
        return Ok(());
    }
    anyhow::bail!(
        "{command} routes the supported Mac local-answer path through --device {APPLE_M4_CPU_NEON}; requested --device {label}. Full apple-m4-metal inference, MPSGraph inference, and hidden CPU fallback are not supported by this wrapper."
    )
}

fn ensure_supported_mac_validate_device(
    explicit_device_label: Option<&str>,
) -> Result<&'static str> {
    let Some(label) = explicit_device_label else {
        return Ok(APPLE_M4_CPU_NEON);
    };
    match label {
        APPLE_M4_CPU_NEON => Ok(APPLE_M4_CPU_NEON),
        APPLE_M3_AIR_CPU_NEON => Ok(APPLE_M3_AIR_CPU_NEON),
        _ => anyhow::bail!(
            "mac validate routes supported Apple CPU/NEON validation through --device {APPLE_M4_CPU_NEON} or --device {APPLE_M3_AIR_CPU_NEON}; requested --device {label}. Full {APPLE_M4_METAL}/{APPLE_M3_AIR_METAL} inference, {APPLE_M3_AIR_MPSGRAPH}/MPSGraph model inference, and hidden CPU fallback are not supported by this wrapper."
        ),
    }
}

fn ensure_supported_mac_benchmark_device(
    explicit_device_label: Option<&str>,
) -> Result<&'static str> {
    let Some(label) = explicit_device_label else {
        return Ok(APPLE_M4_CPU_NEON);
    };
    if label == APPLE_M4_CPU_NEON {
        return Ok(APPLE_M4_CPU_NEON);
    }
    anyhow::bail!(
        "mac benchmark routes the dense SLM benchmark path through --device {APPLE_M4_CPU_NEON}; requested --device {label}. Full apple-m4-metal inference, MPSGraph inference, Neural Engine execution, and hidden CPU fallback are not supported by this benchmark wrapper."
    )
}

fn ensure_supported_mac_serve_device(explicit_device_label: Option<&str>) -> Result<()> {
    let Some(label) = explicit_device_label else {
        return Ok(());
    };
    if label == APPLE_M4_CPU_NEON {
        return Ok(());
    }
    anyhow::bail!(
        "mac serve routes the supported Mac local service path through --device {APPLE_M4_CPU_NEON}; requested --device {label}. Full apple-m4-metal inference, MPSGraph inference, and hidden CPU fallback are not supported by this wrapper."
    )
}

fn run_bitnet_proof_preflight(
    model: PathBuf,
    tokenizer_authority: Option<String>,
    accepted_artifact: Option<PathBuf>,
    proof_receipt: Option<PathBuf>,
    prompt: String,
    max_new_tokens: usize,
    strict: bool,
    json_out: PathBuf,
) -> Result<()> {
    let tokenizer_authority = tokenizer_authority.map(|value| value.trim().to_string());
    let tokenizer_authority = tokenizer_authority.filter(|value| !value.is_empty());
    let proof_summary = match proof_receipt.as_deref() {
        Some(path) if path.exists() => {
            validate_bitnet_local_answer_proof_receipt(path, tokenizer_authority.as_deref())
                .map(Some)
                .unwrap_or_else(|error| {
                    Some(serde_json::json!({
                        "path": path,
                        "valid": false,
                        "error": error.to_string(),
                    }))
                })
        }
        Some(path) => Some(serde_json::json!({
            "path": path,
            "valid": false,
            "error": "proof receipt is missing",
        })),
        None => None,
    };
    let proof_valid =
        proof_summary.as_ref().and_then(|summary| summary["valid"].as_bool()) == Some(true);
    let artifact_summary = match accepted_artifact.as_deref() {
        Some(path) if path.exists() => {
            validate_bitnet_accepted_artifact_receipt(path, tokenizer_authority.as_deref())
                .map(Some)
                .unwrap_or_else(|error| {
                    Some(serde_json::json!({
                        "path": path,
                        "valid": false,
                        "error": error.to_string(),
                    }))
                })
        }
        Some(path) => Some(serde_json::json!({
            "path": path,
            "valid": false,
            "error": "accepted artifact receipt is missing",
        })),
        None => None,
    };

    let mut blockers = Vec::new();
    if !strict {
        blockers.push(
            "pass --strict so hidden loader/tokenizer fallback cannot count as proof".to_string(),
        );
    }
    if tokenizer_authority.is_none() && !proof_valid {
        blockers.push(
            "pass --tokenizer-authority from the accepted BitNet artifact sweep receipt or pass --proof-receipt with tokenizer authority"
                .to_string(),
        );
    }
    let local_model_exists = model.exists();
    if !local_model_exists && !proof_valid {
        blockers.push(format!(
            "accepted BitNet GGUF is missing at {}; this M4 command never downloads artifacts",
            model.display()
        ));
    }
    if !proof_valid {
        match (&accepted_artifact, &artifact_summary) {
            (None, _) => blockers.push(
                "pass --accepted-artifact <receipt.json> from the MacBook/artifact sweep or --proof-receipt <answer-corpus.json>"
                    .to_string(),
            ),
            (Some(_), Some(summary))
                if summary.get("valid").and_then(|value| value.as_bool()) != Some(true) =>
            {
                let reason = summary
                    .get("error")
                    .and_then(|value| value.as_str())
                    .unwrap_or("accepted artifact receipt did not validate");
                blockers.push(format!("accepted artifact receipt is not usable: {reason}"));
            }
            _ => {}
        }
    }
    if let Some(summary) = &proof_summary
        && summary.get("valid").and_then(|value| value.as_bool()) != Some(true)
    {
        let reason = summary
            .get("error")
            .and_then(|value| value.as_str())
            .unwrap_or("proof receipt did not validate");
        blockers.push(format!("proof receipt is not usable: {reason}"));
    }
    if prompt.trim().is_empty() {
        blockers.push("proof prompt must not be empty".to_string());
    }
    if max_new_tokens == 0 {
        blockers.push("max-new-tokens must be greater than zero".to_string());
    }

    let result = if !blockers.is_empty() {
        "blocked"
    } else if proof_valid {
        "verified"
    } else {
        "ready"
    };
    let receipt_tokenizer_authority =
        tokenizer_authority.as_ref().map(|authority| serde_json::json!(authority)).or_else(|| {
            proof_summary.as_ref().and_then(|summary| summary.get("tokenizer_authority")).cloned()
        });
    let receipt = serde_json::json!({
        "artifact_kind": "apple_m4_bitnet_proof_preflight",
        "schema_version": 1,
        "result": result,
        "proof_executed": proof_valid,
        "requested_backend": APPLE_M4_CPU_NEON,
        "selected_backend": APPLE_M4_CPU_NEON,
        "runtime_api": "cpu",
        "fallback_used": false,
        "model": {
            "path": model,
            "exists": local_model_exists,
            "required": "accepted BitNet GGUF from apple-bitnet-artifact-sweep",
        },
        "tokenizer": {
            "authority": receipt_tokenizer_authority,
            "required": true,
        },
        "accepted_artifact": {
            "path": accepted_artifact,
            "summary": artifact_summary,
        },
        "proof_receipt": {
            "path": proof_receipt,
            "summary": proof_summary,
        },
        "generation": {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "greedy": true,
            "deterministic": true,
        },
        "required_receipt_fields": [
            "model.source",
            "model.sha256",
            "tokenizer.authority",
            "kernel_family",
            "requested_backend",
            "selected_backend",
            "runtime_api",
            "fallback_used",
            "generation.text",
            "generation.generated_token_ids",
            "timing",
            "claim_boundary"
        ],
        "claim_boundary": {
            "m4_bitnet_proof_prepared": true,
            "m4_bitnet_answer_corpus_proof_verified": proof_valid,
            "bitnet_answer_corpus_quality_verified": proof_valid,
            "bitnet_answer_quality_claimed": false,
            "bitnet_mac_ask_chat_enabled": false,
            "bitnet_mac_serve_enabled": false,
            "artifact_accepted_by_this_item": false,
            "full_metal_inference_claimed": false,
            "qk256_apple_claimed": false,
            "neural_engine_execution_claimed": false,
            "broad_performance_claim": false
        },
        "blockers": blockers,
    });
    write_json_receipt(&json_out, &receipt)?;

    if result == "blocked" {
        let blockers = receipt
            .get("blockers")
            .and_then(|value| value.as_array())
            .map(|items| {
                items.iter().filter_map(|item| item.as_str()).collect::<Vec<_>>().join("; ")
            })
            .unwrap_or_else(|| "unknown blocker".to_string());
        anyhow::bail!("M4 BitNet proof is blocked: {blockers}");
    }

    if proof_valid {
        println!(
            "M4 BitNet proof receipt verified; BitNet remains limited to explicit one-shot `bitnet mac ask` plus fixed-prompt `bitnet mac bitnet-warm` and does not enable `bitnet mac chat` or `bitnet mac serve`. Receipt: {}",
            json_out.display()
        );
    } else {
        println!(
            "M4 BitNet proof preflight passed; BitNet remains limited to explicit one-shot `bitnet mac ask` plus fixed-prompt `bitnet mac bitnet-warm` and does not enable `bitnet mac chat` or `bitnet mac serve`. Receipt: {}",
            json_out.display()
        );
    }
    Ok(())
}

fn validate_bitnet_local_answer_proof_receipt(
    path: &Path,
    expected_tokenizer_authority: Option<&str>,
) -> Result<serde_json::Value> {
    let receipt = read_json_receipt(path)?;
    if receipt["artifact_kind"].as_str() != Some("bitnet_apple_m4_local_answer_corpus") {
        anyhow::bail!("proof receipt artifact_kind must be bitnet_apple_m4_local_answer_corpus");
    }
    if receipt["model"]["sha256"].as_str() != Some(BITNET_M4_EXPECTED_MODEL_SHA256) {
        anyhow::bail!("proof receipt model.sha256 does not match accepted Microsoft I2_S artifact");
    }
    if receipt["model"]["answer_ready_artifact_available"].as_bool() != Some(true)
        || receipt["model"]["answer_ready"]["state"].as_str() != Some("answer_ready")
    {
        anyhow::bail!("proof receipt does not record an answer_ready model artifact");
    }
    let tokenizer_authority = &receipt["tokenizer"]["authority"];
    if tokenizer_authority["source"].as_str() != Some("external_tokenizer_json")
        || tokenizer_authority["sha256"].as_str() != Some(BITNET_M4_EXPECTED_TOKENIZER_SHA256)
        || tokenizer_authority["ggml_pre"].as_str() != Some("llama-bpe")
        || receipt["tokenizer"]["strict"].as_bool() != Some(true)
    {
        anyhow::bail!(
            "proof receipt tokenizer authority is not the strict external llama-bpe authority"
        );
    }
    if let Some(expected) = expected_tokenizer_authority {
        let accepted_aliases = [
            "external_tokenizer_json",
            "externally_supplied_llama_bpe",
            "llama-bpe",
            "llama-bpe-external",
        ];
        if expected != tokenizer_authority["source"].as_str().unwrap_or_default()
            && expected != receipt["tokenizer"]["source"].as_str().unwrap_or_default()
            && expected != tokenizer_authority["ggml_pre"].as_str().unwrap_or_default()
            && !accepted_aliases.contains(&expected)
        {
            anyhow::bail!(
                "tokenizer authority mismatch: proof receipt does not match `{expected}`"
            );
        }
    }

    let quality_summary = &receipt["quality_summary"];
    let total = quality_summary["total"].as_u64().unwrap_or(0);
    let passed = quality_summary["passed"].as_u64().unwrap_or(0);
    if total == 0
        || passed != total
        || quality_summary["failed"].as_u64().unwrap_or(1) != 0
        || quality_summary["timeout"].as_u64().unwrap_or(1) != 0
        || quality_summary["not_run"].as_u64().unwrap_or(1) != 0
    {
        anyhow::bail!("proof receipt quality_summary is not an all-passed answer corpus");
    }

    let claim_boundary = &receipt["claim_boundary"];
    for (path_name, expected) in [
        ("answer_ready_artifact_available", true),
        ("backend_quality_gate_passed", true),
        ("coherent_output_observed", true),
        ("coherent_answer_claimed", true),
    ] {
        if claim_boundary[path_name].as_bool() != Some(expected) {
            anyhow::bail!("proof receipt claim_boundary.{path_name} must be {expected}");
        }
    }
    for path_name in [
        "diagnostic_only_until_answer_ready_artifact",
        "full_metal_inference_claimed",
        "neural_engine_claimed",
        "qk256_apple_claimed",
        "broad_performance_claimed",
    ] {
        if claim_boundary[path_name].as_bool() != Some(false) {
            anyhow::bail!("proof receipt claim_boundary.{path_name} must be false");
        }
    }

    let cases = receipt["cases"]
        .as_array()
        .ok_or_else(|| anyhow!("proof receipt cases must be an array"))?;
    if cases.is_empty() {
        anyhow::bail!("proof receipt must contain at least one case");
    }
    let mut generated_tokens = 0_u64;
    for (index, case) in cases.iter().enumerate() {
        let case_label = case["id"].as_str().unwrap_or("<unknown>");
        if case["status"].as_str() != Some("passed")
            || case["quality"]["passed"].as_bool() != Some(true)
            || case["quality"]["non_empty_answer"].as_bool() != Some(true)
        {
            anyhow::bail!("proof receipt case {index} ({case_label}) did not pass quality");
        }
        if case["answer"].as_str().unwrap_or_default().trim().is_empty() {
            anyhow::bail!("proof receipt case {index} ({case_label}) has empty answer text");
        }
        if case["backend"]["requested_backend"].as_str() != Some(APPLE_M4_CPU_NEON)
            || case["backend"]["selected_backend"].as_str() != Some(APPLE_M4_CPU_NEON)
            || case["backend"]["runtime_api"].as_str() != Some("cpu")
            || case["backend"]["fallback_used"].as_bool() != Some(false)
        {
            anyhow::bail!(
                "proof receipt case {index} ({case_label}) backend/fallback fields are not strict apple-m4-cpu-neon"
            );
        }
        if case["model"]["sha256"].as_str() != Some(BITNET_M4_EXPECTED_MODEL_SHA256)
            || case["model"]["family"].as_str() != Some("bitnet")
            || case["loader"]["mode"].as_str() != Some("real_gguf")
        {
            anyhow::bail!(
                "proof receipt case {index} ({case_label}) model/loader identity is not strict BitNet real GGUF"
            );
        }
        if case["tokenizer"]["strict"].as_bool() != Some(true)
            || case["tokenizer"]["pretokenizer_authority"].as_str() != Some("llama-bpe")
        {
            anyhow::bail!(
                "proof receipt case {index} ({case_label}) tokenizer authority is not strict llama-bpe"
            );
        }
        if case["prompt_template"].as_str() != Some(BITNET_M4_PROMPT_TEMPLATE)
            || case["prompt"]["template_family"].as_str() != Some(BITNET_M4_PROMPT_TEMPLATE)
            || case["prompt_prefill"]["exercised"].as_bool() != Some(true)
        {
            anyhow::bail!(
                "proof receipt case {index} ({case_label}) prompt/template/prefill evidence is incomplete"
            );
        }
        let token_ids = case["token_ids"]["generated"].as_array().ok_or_else(|| {
            anyhow!("proof receipt case {index} ({case_label}) is missing generated token IDs")
        })?;
        if token_ids.is_empty() {
            anyhow::bail!("proof receipt case {index} ({case_label}) has no generated token IDs");
        }
        let case_generated_tokens = case["tokens"]["generated"].as_u64().unwrap_or(0);
        if case_generated_tokens == 0 || case_generated_tokens as usize != token_ids.len() {
            anyhow::bail!(
                "proof receipt case {index} ({case_label}) token count does not match generated token IDs"
            );
        }
        generated_tokens = generated_tokens.saturating_add(case_generated_tokens);
        for path_name in [&["timing", "decode_total_ms"][..], &["latency", "total_ms"][..]] {
            if !json_number_at(case, path_name) {
                anyhow::bail!(
                    "proof receipt case {index} ({case_label}) is missing timing/latency fields"
                );
            }
        }
    }

    Ok(serde_json::json!({
        "path": path,
        "valid": true,
        "artifact_kind": receipt["artifact_kind"],
        "case_count": cases.len(),
        "generated_tokens": generated_tokens,
        "quality_summary": quality_summary,
        "model_sha256": BITNET_M4_EXPECTED_MODEL_SHA256,
        "tokenizer_sha256": BITNET_M4_EXPECTED_TOKENIZER_SHA256,
        "tokenizer_authority": tokenizer_authority,
        "prompt_template": BITNET_M4_PROMPT_TEMPLATE,
        "requested_backend": APPLE_M4_CPU_NEON,
        "selected_backend": APPLE_M4_CPU_NEON,
        "runtime_api": "cpu",
        "fallback_used": false,
        "mac_route_enabled": false,
    }))
}

fn json_number_at(value: &serde_json::Value, path: &[&str]) -> bool {
    let mut current = value;
    for key in path {
        current = &current[*key];
    }
    current.is_number()
}

fn validate_bitnet_accepted_artifact_receipt(
    path: &Path,
    expected_tokenizer_authority: Option<&str>,
) -> Result<serde_json::Value> {
    let receipt = read_json_receipt(path)?;
    let accepted = receipt.get("accepted").and_then(|value| value.as_bool()) == Some(true)
        || receipt.pointer("/artifact/accepted").and_then(|value| value.as_bool()) == Some(true)
        || receipt.get("result").and_then(|value| value.as_str()) == Some("accepted");
    if !accepted {
        anyhow::bail!("receipt must record accepted=true or result=accepted");
    }
    let tokenizer_authority = receipt
        .pointer("/tokenizer/authority")
        .or_else(|| receipt.get("tokenizer_authority"))
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow!("receipt is missing tokenizer authority"))?;
    if let Some(expected) = expected_tokenizer_authority
        && tokenizer_authority != expected
    {
        anyhow::bail!(
            "tokenizer authority mismatch: receipt has `{tokenizer_authority}`, command requested `{expected}`"
        );
    }
    let model_sha256 = receipt
        .pointer("/model/sha256")
        .or_else(|| receipt.get("sha256"))
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow!("receipt is missing model SHA256"))?;
    let kernel_family = receipt
        .get("kernel_family")
        .or_else(|| receipt.pointer("/model/kernel_family"))
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow!("receipt is missing kernel_family"))?;
    Ok(serde_json::json!({
        "path": path,
        "valid": true,
        "accepted": true,
        "model_sha256": model_sha256,
        "tokenizer_authority": tokenizer_authority,
        "kernel_family": kernel_family,
    }))
}

fn write_json_receipt(path: &Path, receipt: &serde_json::Value) -> Result<()> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(receipt)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn run_check(model_id: &str, cache_dir: Option<PathBuf>, json: bool) -> Result<()> {
    let status = model_cache::apple_m4_slm_cache_status_json(model_id, cache_dir, true)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&status)?);
    } else {
        println!("Apple M4 SLM model: {}", status["id"].as_str().unwrap_or(model_id));
        println!("Cache path: {}", status["cache_path"].as_str().unwrap_or("<unknown>"));
        println!("State: {}", status["state"].as_str().unwrap_or("<unknown>"));
        println!("Runtime backend: {APPLE_M4_CPU_NEON}");
        println!(
            "Claim boundary: SLM CPU/NEON local answers only; no BitNet, full Metal, Neural Engine, QK256, or broad performance claim."
        );
    }
    if status["ready"].as_bool().unwrap_or(false) {
        Ok(())
    } else {
        let next_step = status["next_step"].as_str().unwrap_or("run `bitnet model fetch`");
        anyhow::bail!("Apple M4 SLM model cache is not ready: {next_step}")
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_ask(
    model_id: &str,
    cache_dir: Option<PathBuf>,
    model_path: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    question: String,
    system_prompt: Option<String>,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
    threads: usize,
    timeout_seconds: Option<u64>,
    json_out: PathBuf,
    progress: bool,
    quiet: bool,
) -> Result<()> {
    if model_cache::is_apple_m4_bitnet_artifact_id(model_id) {
        return run_bitnet_ask(
            model_id,
            cache_dir,
            model_path,
            tokenizer,
            question,
            system_prompt,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
            threads,
            timeout_seconds,
            json_out,
            progress,
            quiet,
        )
        .await;
    }
    if timeout_seconds.is_some() {
        anyhow::bail!(
            "`bitnet mac ask --timeout-seconds` is currently scoped to the explicit BitNet one-shot route; dense SLM ask does not use this timeout flag yet"
        );
    }
    if model_path.is_some() || tokenizer.is_some() {
        anyhow::bail!(
            "`bitnet mac ask` accepts --model-path/--tokenizer only for the explicit BitNet one-shot route; dense SLM models use --model-id and the verified model cache"
        );
    }
    let model = model_cache::verified_apple_m4_slm_model(model_id, cache_dir)?;
    let progress_enabled = progress && !quiet;
    if !quiet {
        eprintln!("{}", mac_ask_operator_summary_line(&model, &json_out));
    }
    run_one_shot_mac_answer(
        &model,
        question,
        system_prompt,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        seed,
        threads,
        json_out,
        "mac ask",
        progress_enabled,
    )
    .await?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_bitnet_ask(
    model_id: &str,
    cache_dir: Option<PathBuf>,
    model_path: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    question: String,
    system_prompt: Option<String>,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
    threads: usize,
    timeout_seconds: Option<u64>,
    json_out: PathBuf,
    progress: bool,
    quiet: bool,
) -> Result<()> {
    let started_at = std::time::Instant::now();
    let progress_enabled = progress && !quiet;
    if temperature != 0.0 || top_p != 1.0 {
        anyhow::bail!(
            "BitNet Mac ask is currently scoped to deterministic greedy proof settings; use --temperature 0.0 --top-p 1.0"
        );
    }
    if !matches!(top_k, 0 | 1) {
        anyhow::bail!("BitNet Mac ask is currently scoped to greedy top-k 0 or 1");
    }
    let failure_context = BitNetMacAskFailureContext {
        model_id: model_id.to_string(),
        cache_dir: cache_dir.clone(),
        model_path: model_path.clone(),
        tokenizer_path: tokenizer.clone(),
        question_bytes: question.len(),
        question_sha256: sha256_hex(question.as_bytes()),
        max_new_tokens,
        timeout_seconds,
        progress_enabled,
        started_at,
    };
    let tokenizer = match tokenizer {
        Some(tokenizer) => tokenizer,
        None => {
            return fail_bitnet_mac_ask_with_receipt(
                &json_out,
                failure_context,
                "tokenizer_authority_missing",
                "BitNet Mac ask requires explicit tokenizer authority; pass --tokenizer models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json",
                false,
            );
        }
    };
    mac_ask_progress(progress_enabled, "tokenizer_verify_start", || {
        format!("path={}", tokenizer.display())
    });
    if !tokenizer.exists() {
        return fail_bitnet_mac_ask_with_receipt(
            &json_out,
            BitNetMacAskFailureContext {
                tokenizer_path: Some(tokenizer.clone()),
                ..failure_context.clone()
            },
            "tokenizer_missing",
            &format!(
                "BitNet Mac ask tokenizer is missing at {}; pass the accepted external tokenizer.json",
                tokenizer.display()
            ),
            false,
        );
    }
    let tokenizer_sha256 = match verify_bitnet_m4_tokenizer(&tokenizer) {
        Ok(sha256) => sha256,
        Err(error) => {
            return fail_bitnet_mac_ask_with_receipt(
                &json_out,
                BitNetMacAskFailureContext {
                    tokenizer_path: Some(tokenizer.clone()),
                    ..failure_context.clone()
                },
                "tokenizer_verify_failed",
                &error.to_string(),
                false,
            );
        }
    };
    mac_ask_progress(progress_enabled, "tokenizer_verify_complete", || {
        format!("sha256={}...", short_sha(&tokenizer_sha256))
    });
    mac_ask_progress(progress_enabled, "model_verify_start", || format!("model_id={model_id}"));
    let model = match model_cache::verified_apple_m4_bitnet_model(model_id, cache_dir, model_path) {
        Ok(model) => model,
        Err(error) => {
            return fail_bitnet_mac_ask_with_receipt(
                &json_out,
                BitNetMacAskFailureContext {
                    tokenizer_path: Some(tokenizer.clone()),
                    ..failure_context.clone()
                },
                "model_verify_failed",
                &error.to_string(),
                false,
            );
        }
    };
    mac_ask_progress(progress_enabled, "model_verify_complete", || {
        format!("path={} sha256={}...", model.path.display(), short_sha(&model.sha256))
    });
    if !quiet {
        eprintln!(
            "{}",
            mac_bitnet_ask_operator_summary_line(&model, &tokenizer, &tokenizer_sha256, &json_out)
        );
    }
    mac_ask_progress(progress_enabled, "generation_start", || {
        format!(
            "max_new_tokens={max_new_tokens} receipt={} timing_fields=model_load_ms,tokenizer_load_ms,prefill_ms,first_token_ms,decode_total_ms",
            json_out.display()
        )
    });
    let ask_generation = crate::run_simple_generation(
        APPLE_M4_CPU_NEON,
        model.path.clone(),
        "auto".to_string(),
        None,
        Some(tokenizer.clone()),
        question,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        seed,
        false,
        false,
        true,
        true,
        Some(json_out.clone()),
        None,
        None,
        false,
        false,
        true,
        true,
        threads,
        BITNET_M4_PROMPT_TEMPLATE.to_string(),
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
        true,
        Some("mac_bitnet_ask".to_string()),
        false,
        progress_enabled,
    );
    let ask_result = if let Some(seconds) = timeout_seconds {
        match tokio::time::timeout(std::time::Duration::from_secs(seconds), ask_generation).await {
            Ok(result) => result,
            Err(_) => {
                return fail_bitnet_mac_ask_with_receipt(
                    &json_out,
                    BitNetMacAskFailureContext {
                        model_path: Some(model.path.clone()),
                        tokenizer_path: Some(tokenizer.clone()),
                        ..failure_context.clone()
                    },
                    "generation_timeout",
                    &format!("BitNet Mac ask exceeded --timeout-seconds {seconds}"),
                    true,
                );
            }
        }
    } else {
        ask_generation.await
    };
    if let Err(error) = ask_result {
        let message = error.to_string();
        let stage = bitnet_mac_ask_failure_stage(&message);
        return fail_bitnet_mac_ask_with_receipt(
            &json_out,
            BitNetMacAskFailureContext {
                model_path: Some(model.path.clone()),
                tokenizer_path: Some(tokenizer.clone()),
                ..failure_context.clone()
            },
            stage,
            &message,
            false,
        );
    }
    annotate_and_validate_bitnet_mac_ask_receipt(&json_out, &model, &tokenizer, &tokenizer_sha256)?;
    mac_ask_progress(progress_enabled, "receipt_validated", || {
        format!("path={} chat=false serve=false", json_out.display())
    });
    Ok(())
}

#[derive(Clone)]
struct BitNetMacAskFailureContext {
    model_id: String,
    cache_dir: Option<PathBuf>,
    model_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    question_bytes: usize,
    question_sha256: String,
    max_new_tokens: usize,
    timeout_seconds: Option<u64>,
    progress_enabled: bool,
    started_at: std::time::Instant,
}

fn fail_bitnet_mac_ask_with_receipt(
    json_out: &Path,
    context: BitNetMacAskFailureContext,
    stage: &str,
    message: &str,
    timeout_reached: bool,
) -> Result<()> {
    let repair_guidance = bitnet_mac_ask_failure_repair_guidance(stage, &context, timeout_reached);
    let repair_text = bitnet_mac_ask_failure_repair_text(&repair_guidance);
    if let Err(receipt_error) = write_bitnet_mac_ask_failure_receipt(
        json_out,
        context,
        stage,
        message,
        timeout_reached,
        &repair_guidance,
    ) {
        anyhow::bail!(
            "{message}{repair_text}; additionally failed to write BitNet Mac ask failure receipt {}: {receipt_error}",
            json_out.display()
        );
    }
    anyhow::bail!("{message}; failure receipt written to {}{repair_text}", json_out.display())
}

fn write_bitnet_mac_ask_failure_receipt(
    path: &Path,
    context: BitNetMacAskFailureContext,
    stage: &str,
    message: &str,
    timeout_reached: bool,
    repair_guidance: &[String],
) -> Result<()> {
    let elapsed_ms = context.started_at.elapsed().as_secs_f64() * 1000.0;
    let timeout_enforced = context.timeout_seconds.is_some();
    let receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "artifact_kind": "bitnet_apple_m4_mac_ask_failure",
        "artifact_path": path.display().to_string(),
        "operator_command": "mac ask",
        "status": "failed",
        "requested_backend": APPLE_M4_CPU_NEON,
        "selected_backend": APPLE_M4_CPU_NEON,
        "runtime_api": "cpu",
        "fallback_used": false,
        "fallback_reason": serde_json::Value::Null,
        "model_id": context.model_id,
        "model": {
            "expected_sha256": BITNET_M4_EXPECTED_MODEL_SHA256,
            "path": context.model_path.as_ref().map(|path| path.display().to_string()),
            "family": "bitnet",
        },
        "tokenizer": {
            "expected_sha256": BITNET_M4_EXPECTED_TOKENIZER_SHA256,
            "path": context.tokenizer_path.as_ref().map(|path| path.display().to_string()),
            "strict": true,
            "pretokenizer_authority": "llama-bpe",
        },
        "cache": {
            "cache_dir": context.cache_dir.as_ref().map(|path| path.display().to_string()),
        },
        "prompt": {
            "bytes": context.question_bytes,
            "sha256": context.question_sha256,
            "template_family": BITNET_M4_PROMPT_TEMPLATE,
        },
        "generation": {
            "max_new_tokens": context.max_new_tokens,
            "generated_text": "",
            "generated_token_ids": [],
            "generated_tokens": 0,
            "partial_text": "",
            "partial_token_ids": [],
            "partial_generation_available": false,
        },
        "failure": {
            "stage": stage,
            "message": message,
            "elapsed_ms": (elapsed_ms * 1000.0).round() / 1000.0,
        },
        "progress": {
            "enabled": context.progress_enabled,
            "status_stream": "stderr",
            "last_stage": stage,
            "stage_taxonomy": bitnet_mac_ask_stage_taxonomy(),
            "diagnostic_note": "BitNet one-shot ask exposes tokenizer/model verification plus generation timing stages; partial generation may be unavailable for cancelled timeouts",
        },
        "timeout_boundary": {
            "configured_seconds": context.timeout_seconds,
            "reached": timeout_reached,
            "enforced": timeout_enforced,
            "status": if timeout_reached { "reached" } else { "not_reached" },
            "stage": stage,
            "note": "failure occurred before a complete BitNet one-shot answer receipt was produced",
        },
        "repair_guidance": repair_guidance,
        "mac_bitnet_claim_boundary": {
            "bitnet_one_shot_mac_ask": true,
            "partial_failure_receipt": true,
            "chat_enabled": false,
            "serve_enabled": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false,
            "speedup_claim": false,
        },
        "bitnet_quality_claimed": false,
        "memory": memory_receipt_json(),
    });
    write_json_receipt(path, &receipt)
}

fn bitnet_mac_ask_failure_repair_guidance(
    stage: &str,
    context: &BitNetMacAskFailureContext,
    timeout_reached: bool,
) -> Vec<String> {
    let mut guidance = Vec::new();
    let cache_dir_arg = context
        .cache_dir
        .as_ref()
        .map(|path| format!(" --cache-dir {}", path.display()))
        .unwrap_or_default();
    if stage == "tokenizer_authority_missing" {
        guidance.push(
            "BitNet ask does not infer tokenizer authority from the GGUF or dense SLM cache; pass the accepted external tokenizer explicitly.".to_string(),
        );
    }
    if stage.contains("tokenizer") {
        guidance.push(format!(
            "pass --tokenizer models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json with SHA256 {BITNET_M4_EXPECTED_TOKENIZER_SHA256}"
        ));
        if let Some(tokenizer_path) = context.tokenizer_path.as_ref() {
            guidance.push(format!(
                "verify the tokenizer path with `shasum -a 256 {}` before retrying",
                tokenizer_path.display()
            ));
        }
    }
    if stage.contains("model") || stage.contains("generation") || stage == "decode" {
        let model_repair = if context.model_path.is_some() {
            format!(
                "replace --model-path with the accepted Microsoft I2_S GGUF with SHA256 {BITNET_M4_EXPECTED_MODEL_SHA256}"
            )
        } else {
            format!(
                "run `bitnet model fetch {}`{} or pass --model-path <accepted-bitnet-gguf>",
                context.model_id, cache_dir_arg
            )
        };
        guidance.push(model_repair);
        if let Some(model_path) = context.model_path.as_ref() {
            guidance.push(format!(
                "verify the explicit model with `bitnet model verify {} --path {}` before retrying",
                context.model_id,
                model_path.display()
            ));
        } else {
            guidance.push(format!(
                "inspect cache state with `bitnet mac models{cache_dir_arg}` and `bitnet model verify {}`{}",
                context.model_id, cache_dir_arg
            ));
        }
    }
    if timeout_reached || stage.contains("timeout") {
        guidance.push(
            "rerun with --progress to see the last completed one-shot stage, then increase --timeout-seconds or reduce --max-new-tokens".to_string(),
        );
    }
    guidance.push(
        "keep BitNet chat and serve disabled; this receipt is a failed one-shot ask attempt"
            .to_string(),
    );
    guidance
}

fn bitnet_mac_ask_failure_repair_text(guidance: &[String]) -> String {
    if guidance.is_empty() {
        return String::new();
    }
    let mut text = String::from("\nRepair guidance:");
    for (index, step) in guidance.iter().enumerate() {
        text.push_str(&format!("\n  {}. {step}", index + 1));
    }
    text
}

fn bitnet_mac_ask_failure_stage(message: &str) -> &'static str {
    let lower = message.to_ascii_lowercase();
    if lower.contains("failed to load real model") || lower.contains("model load") {
        "model_load"
    } else if lower.contains("failed to resolve tokenizer") || lower.contains("tokenizer load") {
        "tokenizer_load"
    } else if lower.contains("tokenize") || lower.contains("encode") {
        "prompt_tokenize"
    } else if lower.contains("prefill") {
        "prefill"
    } else if lower.contains("first token") || lower.contains("time_to_first") {
        "first_token"
    } else if lower.contains("decode")
        || lower.contains("forward")
        || lower.contains("logits")
        || lower.contains("sample")
    {
        "decode"
    } else if lower.contains("receipt") || lower.contains("write") || lower.contains("create") {
        "receipt_write"
    } else {
        "generation_failed"
    }
}

fn bitnet_mac_ask_stage_taxonomy() -> Vec<&'static str> {
    vec![
        "tokenizer_verify",
        "model_verify",
        "model_load",
        "tokenizer_load",
        "prompt_tokenize",
        "prefill",
        "first_token",
        "decode",
        "receipt_write",
        "receipt_validation",
    ]
}

fn mac_ask_progress<F>(enabled: bool, stage: &str, details: F)
where
    F: FnOnce() -> String,
{
    if enabled {
        eprintln!("mac ask progress: {stage} {}", details());
    }
}

fn short_sha(sha256: &str) -> &str {
    sha256.get(..12).unwrap_or(sha256)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn mac_ask_operator_summary_line(model: &VerifiedCachedModel, json_out: &Path) -> String {
    let sha = model.sha256.get(..12).unwrap_or(&model.sha256);
    format!(
        "mac ask: model={} quant={} cache=verified cache_root={} backend={} fallback=false receipt={} sha256={}...",
        model.id,
        model.quantization,
        model.cache_root.display(),
        APPLE_M4_CPU_NEON,
        json_out.display(),
        sha
    )
}

fn mac_bitnet_ask_operator_summary_line(
    model: &VerifiedCachedModel,
    tokenizer: &Path,
    tokenizer_sha256: &str,
    json_out: &Path,
) -> String {
    let sha = model.sha256.get(..12).unwrap_or(&model.sha256);
    let tokenizer_sha = tokenizer_sha256.get(..12).unwrap_or(tokenizer_sha256);
    format!(
        "mac ask bitnet: model={} quant={} model_path={} tokenizer={} tokenizer_sha256={}... backend={} fallback=false receipt={} sha256={}... chat=false serve=false",
        model.id,
        model.quantization,
        model.path.display(),
        tokenizer.display(),
        tokenizer_sha,
        APPLE_M4_CPU_NEON,
        json_out.display(),
        sha
    )
}

fn verify_bitnet_m4_tokenizer(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to open BitNet tokenizer {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read BitNet tokenizer {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let sha256 = format!("{:x}", hasher.finalize());
    if sha256 != BITNET_M4_EXPECTED_TOKENIZER_SHA256 {
        anyhow::bail!(
            "BitNet Mac ask requires tokenizer SHA256 {}; got {} for {}",
            BITNET_M4_EXPECTED_TOKENIZER_SHA256,
            sha256,
            path.display()
        );
    }
    Ok(sha256)
}

#[allow(clippy::too_many_arguments)]
async fn run_one_shot_mac_answer(
    model: &VerifiedCachedModel,
    question: String,
    system_prompt: Option<String>,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
    threads: usize,
    json_out: PathBuf,
    operator_command: &str,
    progress: bool,
) -> Result<ReceiptCheckSummary> {
    crate::run_simple_generation(
        APPLE_M4_CPU_NEON,
        model.path.clone(),
        "auto".to_string(),
        None,
        None,
        question,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        seed,
        false,
        false,
        true,
        true,
        Some(json_out.clone()),
        None,
        None,
        false,
        false,
        true,
        true,
        threads,
        QWEN_PROMPT_TEMPLATE.to_string(),
        false,
        system_prompt,
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
        Some("mac_ask".to_string()),
        false,
        progress,
    )
    .await?;
    annotate_and_validate_mac_receipt(&json_out, model, operator_command)?;
    let receipt = read_json_receipt(&json_out)?;
    validate_mac_receipt_value(&json_out, &receipt)
}

async fn run_smoke(
    model_family: MacSmokeModelFamily,
    model_id: &str,
    cache_dir: Option<PathBuf>,
    model_path: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    max_new_tokens: usize,
    threads: usize,
    json_out: PathBuf,
) -> Result<()> {
    if model_family == MacSmokeModelFamily::Bitnet {
        return run_bitnet_smoke(
            model_id,
            cache_dir,
            model_path,
            tokenizer,
            max_new_tokens,
            threads,
            json_out,
        )
        .await;
    }
    if model_path.is_some() || tokenizer.is_some() {
        anyhow::bail!(
            "`bitnet mac smoke` accepts --model-path/--tokenizer only with --model-family bitnet; dense SLM smoke uses --model-id and the verified model cache"
        );
    }
    let cache_status =
        model_cache::apple_m4_slm_cache_status_json(model_id, cache_dir.clone(), false)?;
    if !cache_status["ready"].as_bool().unwrap_or(false) {
        let next_step = cache_status["next_step"]
            .as_str()
            .unwrap_or("run `bitnet model fetch qwen2.5-0.5b-instruct-q8_0`");
        anyhow::bail!("Apple M4 dense-SLM golden smoke cannot run: {next_step}");
    }

    let model = model_cache::verified_apple_m4_slm_model(model_id, cache_dir)?;
    let cache_status = verified_cache_status_json(&model);
    if let Some(parent) = json_out.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let answer_receipt_path = sibling_receipt_path(&json_out, "answer");
    let answer_summary = run_one_shot_mac_answer(
        &model,
        MAC_SMOKE_PROMPT.to_string(),
        None,
        max_new_tokens,
        0.0,
        1,
        1.0,
        1.1,
        None,
        threads,
        answer_receipt_path.clone(),
        "mac smoke",
        false,
    )
    .await?;
    let answer_receipt = read_json_receipt(&answer_receipt_path)?;
    let text = answer_receipt["text"].as_str().unwrap_or_default().trim().to_string();
    let answer_contains_expected = text.contains(MAC_SMOKE_EXPECTED_FRAGMENT);
    if !answer_contains_expected {
        anyhow::bail!(
            "Apple M4 dense-SLM golden smoke expected the fixed prompt to contain `{MAC_SMOKE_EXPECTED_FRAGMENT}`, got {text:?}"
        );
    }

    let aggregate = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "apple_m4_slm_golden_smoke",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "artifact_path": json_out.display().to_string(),
        "requested_backend": APPLE_M4_CPU_NEON,
        "selected_backend": APPLE_M4_CPU_NEON,
        "runtime_api": "cpu",
        "fallback_used": false,
        "fallback_reason": serde_json::Value::Null,
        "prompt": MAC_SMOKE_PROMPT,
        "expected_text_fragment": MAC_SMOKE_EXPECTED_FRAGMENT,
        "expected_text_fragment_found": answer_contains_expected,
        "text": text,
        "tokens": answer_receipt["tokens"].clone(),
        "model": answer_receipt["model"].clone(),
        "tokenizer": answer_receipt["tokenizer"].clone(),
        "quality": answer_receipt["quality"].clone(),
        "timing": answer_receipt["timing"].clone(),
        "answer_receipt": {
            "path": answer_receipt_path.display().to_string(),
            "artifact_kind": answer_summary.artifact_kind,
            "prompt_count": answer_summary.prompt_count,
            "generated_tokens": answer_summary.generated_tokens,
            "validated": answer_summary.passed,
        },
        "cache_health": {
            "checked": true,
            "ready": true,
            "state": cache_status["state"].clone(),
            "cache_root": cache_status["cache_root"].clone(),
            "cache_path": cache_status["cache_path"].clone(),
            "metadata_path": cache_status["metadata_path"].clone(),
            "present": cache_status["present"].clone(),
            "size_matches": cache_status["size_matches"].clone(),
            "metadata_present": cache_status["metadata_present"].clone(),
            "verified": cache_status["verified"].clone(),
            "verification_passes": cache_status["verification_passes"].clone(),
            "disk": disk_health_json(&model.cache_root, model.bytes),
        },
        "model_cache": {
            "id": model.id,
            "display_name": model.display_name,
            "cache_root": model.cache_root,
            "path": model.path,
            "sha256": model.sha256,
            "bytes": model.bytes,
            "architecture": model.architecture,
            "quantization": model.quantization,
            "tokenizer_model": model.tokenizer_model,
            "tokenizer_pre": model.tokenizer_pre,
            "chat_template": model.chat_template,
            "support_note": model.support_note,
        },
        "mac_claim_boundary": {
            "slm_local_answer": true,
            "golden_smoke": true,
            "requested_backend": APPLE_M4_CPU_NEON,
            "bitnet_quality_claimed": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false,
            "speedup_claim": false
        },
        "broad_performance_claim": false,
        "speedup_claim": false,
    });
    validate_mac_receipt_value(&json_out, &aggregate)?;
    std::fs::write(&json_out, serde_json::to_vec_pretty(&aggregate)?)
        .with_context(|| format!("failed to write {}", json_out.display()))?;
    println!(
        "Mac golden smoke passed: {} (answer receipt: {})",
        json_out.display(),
        answer_receipt_path.display()
    );
    Ok(())
}

async fn run_bitnet_smoke(
    model_id: &str,
    cache_dir: Option<PathBuf>,
    model_path: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    max_new_tokens: usize,
    threads: usize,
    json_out: PathBuf,
) -> Result<()> {
    let model_id = if model_id == model_cache::M4_SLM_RUNTIME_MODEL_ID {
        BITNET_M4_MODEL_ID
    } else {
        model_id
    };
    if !model_cache::is_apple_m4_bitnet_artifact_id(model_id) {
        anyhow::bail!(
            "`bitnet mac smoke --model-family bitnet` only supports {BITNET_M4_MODEL_ID}; got `{model_id}`"
        );
    }
    let tokenizer = tokenizer.unwrap_or_else(|| PathBuf::from(BITNET_M4_DEFAULT_TOKENIZER_PATH));
    if let Some(parent) = json_out.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let answer_receipt_path = sibling_receipt_path(&json_out, "answer");
    run_bitnet_ask(
        model_id,
        cache_dir.clone(),
        model_path,
        Some(tokenizer.clone()),
        MAC_SMOKE_PROMPT.to_string(),
        None,
        max_new_tokens,
        0.0,
        1,
        1.0,
        1.1,
        None,
        threads,
        None,
        answer_receipt_path.clone(),
        false,
        true,
    )
    .await?;

    let answer_receipt = read_json_receipt(&answer_receipt_path)?;
    let answer_summary = validate_mac_receipt_value(&answer_receipt_path, &answer_receipt)?;
    let text = answer_receipt["text"].as_str().unwrap_or_default().trim().to_string();
    let answer_contains_expected = text.contains(MAC_SMOKE_EXPECTED_FRAGMENT);
    if !answer_contains_expected {
        anyhow::bail!(
            "Apple M4 BitNet one-shot smoke expected the fixed prompt to contain `{MAC_SMOKE_EXPECTED_FRAGMENT}`, got {text:?}"
        );
    }

    let aggregate = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_apple_m4_mac_smoke",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "artifact_path": json_out.display().to_string(),
        "operator_command": "mac smoke --model-family bitnet",
        "requested_backend": APPLE_M4_CPU_NEON,
        "selected_backend": APPLE_M4_CPU_NEON,
        "runtime_api": "cpu",
        "fallback_used": false,
        "fallback_reason": serde_json::Value::Null,
        "model_family": "bitnet",
        "model_id": model_id,
        "prompt": MAC_SMOKE_PROMPT,
        "expected_text_fragment": MAC_SMOKE_EXPECTED_FRAGMENT,
        "expected_text_fragment_found": answer_contains_expected,
        "text": text,
        "tokens": answer_receipt["tokens"].clone(),
        "model": answer_receipt["model"].clone(),
        "tokenizer": answer_receipt["tokenizer"].clone(),
        "quality": answer_receipt["quality"].clone(),
        "timing": answer_receipt["timing"].clone(),
        "answer_receipt": {
            "path": answer_receipt_path.display().to_string(),
            "artifact_kind": answer_summary.artifact_kind,
            "prompt_count": answer_summary.prompt_count,
            "generated_tokens": answer_summary.generated_tokens,
            "validated": answer_summary.passed,
        },
        "cache_health": bitnet_mac_ask_readiness_json(cache_dir),
        "mac_bitnet_claim_boundary": {
            "bitnet_one_shot_mac_ask": true,
            "bitnet_mac_smoke": true,
            "chat_enabled": false,
            "serve_enabled": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false,
            "speedup_claim": false
        },
        "bitnet_quality_claimed": false,
        "broad_performance_claim": false,
        "speedup_claim": false,
        "memory": memory_receipt_json(),
    });
    validate_mac_receipt_value(&json_out, &aggregate)?;
    std::fs::write(&json_out, serde_json::to_vec_pretty(&aggregate)?)
        .with_context(|| format!("failed to write {}", json_out.display()))?;
    println!(
        "Mac BitNet one-shot smoke passed: {} (answer receipt: {}, chat=false, serve=false)",
        json_out.display(),
        answer_receipt_path.display()
    );
    Ok(())
}

struct BitnetWarmRun<'a> {
    model_id: &'a str,
    cache_dir: Option<PathBuf>,
    model_path: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    prompts: Vec<String>,
    profiles: Vec<BitnetWarmProfile>,
    max_new_tokens: usize,
    threads: usize,
    timeout_seconds: Option<u64>,
    progress: bool,
    quiet: bool,
    json_out: PathBuf,
}

async fn run_bitnet_warm(request: BitnetWarmRun<'_>) -> Result<()> {
    let BitnetWarmRun {
        model_id,
        cache_dir,
        model_path,
        tokenizer,
        prompts,
        profiles,
        max_new_tokens,
        threads,
        timeout_seconds,
        progress,
        quiet,
        json_out,
    } = request;
    let started_at = std::time::Instant::now();
    let progress_enabled = progress && !quiet;
    if max_new_tokens == 0 {
        anyhow::bail!("`bitnet mac bitnet-warm` requires --max-new-tokens greater than zero");
    }
    if !model_cache::is_apple_m4_bitnet_artifact_id(model_id) {
        anyhow::bail!(
            "`bitnet mac bitnet-warm` only supports {BITNET_M4_MODEL_ID}; got `{model_id}`"
        );
    }
    let (prompts, prompt_source, profile_plan) =
        resolve_bitnet_warm_prompt_plan(prompts, profiles)?;
    let prompt_count = prompts.len();
    let mut failure_context = BitNetWarmFailureContext {
        model_id: model_id.to_string(),
        cache_dir: cache_dir.clone(),
        model_path: model_path.clone(),
        tokenizer_path: tokenizer.clone(),
        prompt_count,
        prompt_source,
        prompt_sha256s: prompts.iter().map(|prompt| sha256_hex(prompt.as_bytes())).collect(),
        profile_plan: profile_plan.clone(),
        max_new_tokens,
        timeout_seconds,
        progress_enabled,
        started_at,
    };
    let tokenizer = tokenizer.unwrap_or_else(|| PathBuf::from(BITNET_M4_DEFAULT_TOKENIZER_PATH));
    failure_context.tokenizer_path = Some(tokenizer.clone());
    bitnet_warm_progress(progress_enabled, "tokenizer_verify_start", || {
        format!("path={}", tokenizer.display())
    });
    if !tokenizer.exists() {
        return fail_bitnet_warm_with_receipt(
            &json_out,
            failure_context,
            "tokenizer_missing",
            &format!(
                "BitNet warm session requires the accepted external tokenizer at {}; pass --tokenizer {}",
                tokenizer.display(),
                BITNET_M4_DEFAULT_TOKENIZER_PATH
            ),
            false,
        );
    }
    let tokenizer_sha256 = match verify_bitnet_m4_tokenizer(&tokenizer) {
        Ok(sha256) => sha256,
        Err(error) => {
            return fail_bitnet_warm_with_receipt(
                &json_out,
                failure_context,
                "tokenizer_verify_failed",
                &error.to_string(),
                false,
            );
        }
    };
    bitnet_warm_progress(progress_enabled, "tokenizer_verify_complete", || {
        format!("sha256={}...", short_sha(&tokenizer_sha256))
    });
    bitnet_warm_progress(progress_enabled, "model_verify_start", || format!("model_id={model_id}"));
    let model = match model_cache::verified_apple_m4_bitnet_model(model_id, cache_dir, model_path) {
        Ok(model) => model,
        Err(error) => {
            return fail_bitnet_warm_with_receipt(
                &json_out,
                failure_context,
                "model_verify_failed",
                &error.to_string(),
                false,
            );
        }
    };
    failure_context.model_path = Some(model.path.clone());
    bitnet_warm_progress(progress_enabled, "model_verify_complete", || {
        format!("path={} sha256={}...", model.path.display(), short_sha(&model.sha256))
    });
    bitnet_warm_progress(progress_enabled, "warm_session_start", || {
        format!(
            "prompts={prompt_count} max_new_tokens={max_new_tokens} timeout_seconds={} stages=model_load,tokenizer_load,prefill,first_token,decode,receipt_write receipt={}",
            timeout_seconds
                .map(|seconds| seconds.to_string())
                .unwrap_or_else(|| "none".to_string()),
            json_out.display()
        )
    });
    let warm_session = crate::run_slm_warm_session(
        APPLE_M4_CPU_NEON,
        model.path.clone(),
        "gguf".to_string(),
        Some(tokenizer.clone()),
        None,
        1,
        prompts,
        max_new_tokens,
        0.0,
        1,
        1.0,
        1.1,
        None,
        true,
        true,
        true,
        true,
        threads,
        BITNET_M4_PROMPT_TEMPLATE.to_string(),
        false,
        None,
        vec!["<|eot_id|>".to_string(), "<|end_of_text|>".to_string()],
        Vec::new(),
        true,
        true,
        false,
        crate::SlmWarmSessionOutput::new(false, progress, quiet)
            .with_model_sha256_override(Some(model.sha256.clone())),
        1,
        1,
        json_out.clone(),
    );
    let warm_result = if let Some(seconds) = timeout_seconds {
        match tokio::time::timeout(std::time::Duration::from_secs(seconds), warm_session).await {
            Ok(result) => result,
            Err(_) => {
                return fail_bitnet_warm_with_receipt(
                    &json_out,
                    failure_context,
                    "timeout",
                    &format!("BitNet warm session exceeded --timeout-seconds {seconds}"),
                    true,
                );
            }
        }
    } else {
        warm_session.await
    };
    if let Err(error) = warm_result {
        let message = error.to_string();
        let stage = bitnet_warm_failure_stage(&message);
        return fail_bitnet_warm_with_receipt(&json_out, failure_context, stage, &message, false);
    }
    bitnet_warm_progress(progress_enabled, "receipt_write_complete", || {
        format!("path={}", json_out.display())
    });
    if let Err(error) = annotate_and_validate_bitnet_warm_session_receipt(
        &json_out,
        &model,
        &tokenizer,
        &tokenizer_sha256,
        prompt_source,
        profile_plan.as_ref(),
        timeout_seconds,
    ) {
        return fail_bitnet_warm_with_receipt(
            &json_out,
            failure_context,
            "receipt_validation_failed",
            &error.to_string(),
            false,
        );
    }
    bitnet_warm_progress(progress_enabled, "receipt_validated", || {
        format!("path={} chat=false serve=false", json_out.display())
    });
    if !quiet {
        println!(
            "Mac BitNet warm session passed: {} (prompts={}, prompt_source={}, chat=false, serve=false)",
            json_out.display(),
            prompt_count,
            prompt_source.as_str()
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BitnetWarmPromptSource {
    FixedProof,
    OperatorPrompts,
    ProfilePrompts,
}

impl BitnetWarmPromptSource {
    const fn as_str(self) -> &'static str {
        match self {
            Self::FixedProof => "fixed_proof_prompts",
            Self::OperatorPrompts => "operator_prompts",
            Self::ProfilePrompts => "profile_prompts",
        }
    }

    const fn variable_prompts(self) -> bool {
        matches!(self, Self::OperatorPrompts | Self::ProfilePrompts)
    }
}

#[derive(Clone, Debug)]
struct BitnetWarmProfilePlan {
    profiles: Vec<BitnetWarmProfile>,
    max_prompt_count: usize,
}

impl BitnetWarmProfilePlan {
    fn profile_ids(&self) -> Vec<&'static str> {
        self.profiles.iter().map(|profile| profile.id()).collect()
    }
}

fn resolve_bitnet_warm_prompt_plan(
    prompts: Vec<String>,
    profiles: Vec<BitnetWarmProfile>,
) -> Result<(Vec<String>, BitnetWarmPromptSource, Option<BitnetWarmProfilePlan>)> {
    if profiles.is_empty() {
        let (prompts, source) = resolve_bitnet_warm_prompts(prompts)?;
        return Ok((prompts, source, None));
    }
    if !prompts.is_empty() {
        anyhow::bail!(
            "`bitnet mac bitnet-warm --profile` cannot be combined with --prompt; use named resident profiles or explicit prompts, not both"
        );
    }
    let mut unique = std::collections::BTreeSet::new();
    for profile in profiles {
        unique.insert(profile);
    }
    let profiles = unique.into_iter().collect::<Vec<_>>();
    let max_prompt_count =
        profiles.iter().map(|profile| profile.prompt_count()).max().unwrap_or_default();
    if max_prompt_count == 0 {
        anyhow::bail!("`bitnet mac bitnet-warm --profile` requires at least one resident profile");
    }
    let plan = BitnetWarmProfilePlan { profiles, max_prompt_count };
    Ok((
        bitnet_warm_profile_prompts(max_prompt_count),
        BitnetWarmPromptSource::ProfilePrompts,
        Some(plan),
    ))
}

fn resolve_bitnet_warm_prompts(
    prompts: Vec<String>,
) -> Result<(Vec<String>, BitnetWarmPromptSource)> {
    if prompts.is_empty() {
        return Ok((
            BITNET_WARM_PROMPTS.iter().map(|prompt| (*prompt).to_string()).collect(),
            BitnetWarmPromptSource::FixedProof,
        ));
    }
    if prompts.len() < 2 {
        anyhow::bail!(
            "`bitnet mac bitnet-warm --prompt` requires at least two prompt values for a warm session"
        );
    }
    for (index, prompt) in prompts.iter().enumerate() {
        if prompt.trim().is_empty() {
            anyhow::bail!(
                "`bitnet mac bitnet-warm --prompt` value {} must not be empty",
                index + 1
            );
        }
    }
    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    for prompt in &prompts {
        *counts.entry(prompt.as_str()).or_default() += 1;
    }
    if !counts.values().any(|count| *count >= 2) {
        anyhow::bail!(
            "`bitnet mac bitnet-warm --prompt` requires at least one exact repeated prompt so deterministic warm reuse can be checked before chat is enabled"
        );
    }
    Ok((prompts, BitnetWarmPromptSource::OperatorPrompts))
}

fn bitnet_warm_profile_prompts(count: usize) -> Vec<String> {
    let mut prompts = (0..count)
        .map(|index| {
            BITNET_WARM_PROFILE_PROMPTS[index % BITNET_WARM_PROFILE_PROMPTS.len()].to_string()
        })
        .collect::<Vec<_>>();
    if prompts.len() >= 2 {
        let first = prompts[0].clone();
        if let Some(last) = prompts.last_mut() {
            *last = first;
        }
    }
    prompts
}

#[derive(Clone)]
struct BitNetWarmFailureContext {
    model_id: String,
    cache_dir: Option<PathBuf>,
    model_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    prompt_count: usize,
    prompt_source: BitnetWarmPromptSource,
    prompt_sha256s: Vec<String>,
    profile_plan: Option<BitnetWarmProfilePlan>,
    max_new_tokens: usize,
    timeout_seconds: Option<u64>,
    progress_enabled: bool,
    started_at: std::time::Instant,
}

fn fail_bitnet_warm_with_receipt(
    json_out: &Path,
    context: BitNetWarmFailureContext,
    stage: &str,
    message: &str,
    timeout_reached: bool,
) -> Result<()> {
    let repair_guidance = bitnet_warm_failure_repair_guidance(stage, &context, timeout_reached);
    let repair_text = bitnet_mac_ask_failure_repair_text(&repair_guidance);
    if let Err(receipt_error) = write_bitnet_warm_failure_receipt(
        json_out,
        context,
        stage,
        message,
        timeout_reached,
        &repair_guidance,
    ) {
        anyhow::bail!(
            "{message}{repair_text}; additionally failed to write BitNet warm failure receipt {}: {receipt_error}",
            json_out.display()
        );
    }
    anyhow::bail!("{message}; failure receipt written to {}{repair_text}", json_out.display())
}

fn write_bitnet_warm_failure_receipt(
    path: &Path,
    context: BitNetWarmFailureContext,
    stage: &str,
    message: &str,
    timeout_reached: bool,
    repair_guidance: &[String],
) -> Result<()> {
    let elapsed_ms = context.started_at.elapsed().as_secs_f64() * 1000.0;
    let timeout_enforced = context.timeout_seconds.is_some();
    let receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "artifact_kind": "bitnet_apple_m4_warm_session_failure",
        "artifact_path": path.display().to_string(),
        "operator_command": "mac bitnet-warm",
        "status": "failed",
        "requested_backend": APPLE_M4_CPU_NEON,
        "selected_backend": APPLE_M4_CPU_NEON,
        "runtime_api": "cpu",
        "fallback_used": false,
        "fallback_reason": serde_json::Value::Null,
        "model_id": context.model_id,
        "model": {
            "expected_sha256": BITNET_M4_EXPECTED_MODEL_SHA256,
            "path": context.model_path.as_ref().map(|path| path.display().to_string()),
            "family": "bitnet",
        },
        "tokenizer": {
            "expected_sha256": BITNET_M4_EXPECTED_TOKENIZER_SHA256,
            "path": context.tokenizer_path.as_ref().map(|path| path.display().to_string()),
            "strict": true,
            "pretokenizer_authority": "llama-bpe",
        },
        "cache": {
            "cache_dir": context.cache_dir.as_ref().map(|path| path.display().to_string()),
        },
        "prompt": {
            "count": context.prompt_count,
            "source": context.prompt_source.as_str(),
            "sha256s": context.prompt_sha256s,
            "template_family": BITNET_M4_PROMPT_TEMPLATE,
        },
        "profile_set": bitnet_warm_profile_plan_metadata_json(context.profile_plan.as_ref()),
        "generation": {
            "max_new_tokens": context.max_new_tokens,
            "generated_text": "",
            "generated_token_ids": [],
            "generated_tokens": 0,
            "partial_text": "",
            "partial_token_ids": [],
            "partial_generation_available": false,
        },
        "failure": {
            "stage": stage,
            "message": message,
            "elapsed_ms": (elapsed_ms * 1000.0).round() / 1000.0,
        },
        "progress": {
            "enabled": context.progress_enabled,
            "status_stream": "stderr",
            "last_stage": stage,
            "stage_taxonomy": bitnet_warm_stage_taxonomy(),
            "diagnostic_note": "model_load, tokenizer_load, prefill, first_token, decode, and receipt_write are explicit warm-session diagnostic stages",
        },
        "timeout_boundary": {
            "configured_seconds": context.timeout_seconds,
            "reached": timeout_reached,
            "enforced": timeout_enforced,
            "status": if timeout_reached { "reached" } else { "not_reached" },
            "stage": stage,
            "note": "failure occurred before a complete BitNet warm-session aggregate receipt was produced",
        },
        "repair_guidance": repair_guidance,
        "mac_bitnet_claim_boundary": {
            "bitnet_warm_session": true,
            "partial_failure_receipt": true,
            "chat_enabled": false,
            "serve_enabled": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false,
            "speedup_claim": false,
        },
        "bitnet_quality_claimed": false,
        "broad_performance_claim": false,
        "speedup_claim": false,
        "memory": memory_receipt_json(),
    });
    write_json_receipt(path, &receipt)
}

fn bitnet_warm_failure_repair_guidance(
    stage: &str,
    context: &BitNetWarmFailureContext,
    timeout_reached: bool,
) -> Vec<String> {
    let mut guidance = Vec::new();
    let cache_dir_arg = context
        .cache_dir
        .as_ref()
        .map(|path| format!(" --cache-dir {}", path.display()))
        .unwrap_or_default();
    if stage.contains("tokenizer") {
        guidance.push(format!(
            "pass --tokenizer {BITNET_M4_DEFAULT_TOKENIZER_PATH} with SHA256 {BITNET_M4_EXPECTED_TOKENIZER_SHA256}"
        ));
        if let Some(tokenizer_path) = context.tokenizer_path.as_ref() {
            guidance.push(format!(
                "verify the tokenizer path with `shasum -a 256 {}` before retrying",
                tokenizer_path.display()
            ));
        }
    }
    if stage.contains("model") || stage == "warm_session_failed" {
        if context.model_path.is_some() {
            guidance.push(format!(
                "replace --model-path with the accepted Microsoft I2_S GGUF with SHA256 {BITNET_M4_EXPECTED_MODEL_SHA256}"
            ));
        } else {
            guidance.push(format!(
                "run `bitnet model fetch {}`{} or pass --model-path <accepted-bitnet-gguf>",
                context.model_id, cache_dir_arg
            ));
        }
        guidance.push(format!(
            "inspect cache state with `bitnet mac models{cache_dir_arg}` and `bitnet model verify {}`{}",
            context.model_id, cache_dir_arg
        ));
    }
    if timeout_reached || stage == "timeout" {
        guidance.push(
            "rerun with --progress to see the last completed warm-session stage, then increase --timeout-seconds or reduce the prompt/max-token set".to_string(),
        );
    }
    if matches!(
        stage,
        "prompt_tokenize"
            | "prefill"
            | "first_token"
            | "decode"
            | "receipt_write"
            | "receipt_validation_failed"
    ) {
        guidance.push(
            "retry the same prompt set with --progress and inspect any per-prompt receipts beside the aggregate receipt".to_string(),
        );
    }
    guidance.push(
        "keep BitNet chat and serve disabled; this receipt is a failed warm-session attempt"
            .to_string(),
    );
    guidance
}

fn bitnet_warm_failure_stage(message: &str) -> &'static str {
    let lower = message.to_ascii_lowercase();
    if lower.contains("failed to load real model") || lower.contains("model load") {
        "model_load"
    } else if lower.contains("failed to resolve tokenizer") || lower.contains("tokenizer load") {
        "tokenizer_load"
    } else if lower.contains("tokenize") || lower.contains("encode") {
        "prompt_tokenize"
    } else if lower.contains("prefill") {
        "prefill"
    } else if lower.contains("first token") || lower.contains("time_to_first") {
        "first_token"
    } else if lower.contains("decode")
        || lower.contains("forward")
        || lower.contains("logits")
        || lower.contains("sample")
    {
        "decode"
    } else if lower.contains("receipt") || lower.contains("write") || lower.contains("create") {
        "receipt_write"
    } else {
        "warm_session_failed"
    }
}

fn bitnet_warm_stage_taxonomy() -> Vec<&'static str> {
    vec![
        "tokenizer_verify",
        "model_verify",
        "model_load",
        "tokenizer_load",
        "prompt_tokenize",
        "prefill",
        "first_token",
        "decode",
        "receipt_write",
        "receipt_validation",
    ]
}

fn bitnet_warm_progress<F>(enabled: bool, stage: &str, details: F)
where
    F: FnOnce() -> String,
{
    if enabled {
        eprintln!("mac bitnet-warm progress: {stage} {}", details());
    }
}

struct BitnetBenchmarkRun<'a> {
    model_id: &'a str,
    cache_dir: Option<PathBuf>,
    model_path: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    one_shot_prompt: String,
    max_new_tokens: usize,
    threads: usize,
    progress: bool,
    quiet: bool,
    json_out: PathBuf,
}

#[derive(Default)]
struct BitnetBenchmarkMetricSamples {
    prompt_tokens: Vec<f64>,
    output_tokens: Vec<f64>,
    model_load_ms: Vec<f64>,
    tokenizer_load_ms: Vec<f64>,
    prompt_tokenize_ms: Vec<f64>,
    prefill_ms: Vec<f64>,
    ttft_ms: Vec<f64>,
    decode_total_ms: Vec<f64>,
    sampling_ms_per_token: Vec<f64>,
    total_wall_ms: Vec<f64>,
    input_tokens_per_second: Vec<f64>,
    output_tokens_per_second: Vec<f64>,
    decode_tokens_per_second: Vec<f64>,
    peak_memory_mb: Vec<f64>,
    memory_drift_mb: Vec<f64>,
}

impl BitnetBenchmarkMetricSamples {
    fn extend_from_summary(&mut self, summary: &serde_json::Value) {
        self.prompt_tokens.extend(benchmark_stat_samples(&summary["prompt_tokens"]));
        self.output_tokens.extend(benchmark_stat_samples(&summary["output_tokens"]));
        self.model_load_ms.extend(benchmark_stat_samples(&summary["timing"]["model_load_ms"]));
        self.tokenizer_load_ms
            .extend(benchmark_stat_samples(&summary["timing"]["tokenizer_load_ms"]));
        self.prompt_tokenize_ms
            .extend(benchmark_stat_samples(&summary["timing"]["prompt_tokenize_ms"]));
        self.prefill_ms.extend(benchmark_stat_samples(&summary["timing"]["prefill_ms"]));
        self.ttft_ms.extend(benchmark_stat_samples(&summary["timing"]["time_to_first_token_ms"]));
        self.decode_total_ms.extend(benchmark_stat_samples(&summary["timing"]["decode_total_ms"]));
        self.sampling_ms_per_token
            .extend(benchmark_stat_samples(&summary["timing"]["sampling_ms_per_token"]));
        self.total_wall_ms.extend(benchmark_stat_samples(&summary["timing"]["total_wall_ms"]));
        self.input_tokens_per_second
            .extend(benchmark_stat_samples(&summary["throughput"]["input_tokens_per_second"]));
        self.output_tokens_per_second
            .extend(benchmark_stat_samples(&summary["throughput"]["output_tokens_per_second"]));
        self.decode_tokens_per_second
            .extend(benchmark_stat_samples(&summary["throughput"]["decode_tokens_per_second"]));
        self.peak_memory_mb.extend(benchmark_stat_samples(&summary["memory"]["peak_memory_mb"]));
        self.memory_drift_mb.extend(benchmark_stat_samples(&summary["memory"]["memory_drift_mb"]));
    }

    fn speed_json(&self) -> serde_json::Value {
        benchmark_flat_metric_json(&[
            ("cold_load_ms", &self.model_load_ms),
            ("model_load_ms", &self.model_load_ms),
            ("tokenizer_load_ms", &self.tokenizer_load_ms),
            ("prompt_tokenize_ms", &self.prompt_tokenize_ms),
            ("prefill_ms", &self.prefill_ms),
            ("ttft_ms", &self.ttft_ms),
            ("decode_total_ms", &self.decode_total_ms),
            ("sampling_ms_per_token", &self.sampling_ms_per_token),
            ("input_tok_s", &self.input_tokens_per_second),
            ("output_tok_s", &self.output_tokens_per_second),
            ("decode_tok_s", &self.decode_tokens_per_second),
            ("total_wall_ms", &self.total_wall_ms),
        ])
    }

    fn memory_json(&self) -> serde_json::Value {
        let mut memory = benchmark_flat_metric_json(&[
            ("peak_memory_mb", &self.peak_memory_mb),
            ("memory_drift_mb", &self.memory_drift_mb),
        ]);
        if let Some(object) = memory.as_object_mut() {
            object.insert(
                "source".to_string(),
                serde_json::json!("getrusage.ru_maxrss process peak delta"),
            );
        }
        memory
    }
}

async fn run_bitnet_benchmark(request: BitnetBenchmarkRun<'_>) -> Result<()> {
    if cfg!(debug_assertions) {
        anyhow::bail!(
            "mac bitnet-benchmark must be run from a release build; use `cargo build --release --locked -p bitnet-cli --no-default-features --features cpu,full-cli --bin bitnet` and then `target/release/bitnet mac bitnet-benchmark ...`"
        );
    }
    let BitnetBenchmarkRun {
        model_id,
        cache_dir,
        model_path,
        tokenizer,
        one_shot_prompt,
        max_new_tokens,
        threads,
        progress,
        quiet,
        json_out,
    } = request;
    if max_new_tokens == 0 {
        anyhow::bail!("`bitnet mac bitnet-benchmark` requires --max-new-tokens greater than zero");
    }
    if one_shot_prompt.trim().is_empty() {
        anyhow::bail!("`bitnet mac bitnet-benchmark` requires a non-empty --one-shot-prompt");
    }

    let output_dir = json_out.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
    let receipt_dir = output_dir.join("receipts");
    std::fs::create_dir_all(&receipt_dir)
        .with_context(|| format!("failed to create {}", receipt_dir.display()))?;
    let ask_receipt = receipt_dir.join("bitnet-mac-ask-benchmark.json");
    let warm_receipt = receipt_dir.join("bitnet-mac-bitnet-warm-benchmark.json");

    let benchmark_start_peak_mb = peak_memory_mb();
    if progress && !quiet {
        eprintln!("mac bitnet-benchmark: running one-shot ask path");
    }
    run_ask(
        model_id,
        cache_dir.clone(),
        model_path.clone(),
        tokenizer.clone(),
        one_shot_prompt,
        None,
        max_new_tokens,
        0.0,
        1,
        1.0,
        1.1,
        None,
        threads,
        None,
        ask_receipt.clone(),
        progress,
        quiet,
    )
    .await?;
    if progress && !quiet {
        eprintln!("mac bitnet-benchmark: running fixed warm path");
    }
    run_bitnet_warm(BitnetWarmRun {
        model_id,
        cache_dir,
        model_path,
        tokenizer,
        prompts: Vec::new(),
        profiles: Vec::new(),
        max_new_tokens,
        threads,
        timeout_seconds: None,
        progress,
        quiet,
        json_out: warm_receipt.clone(),
    })
    .await?;

    let ask = read_json_receipt(&ask_receipt)?;
    let warm = read_json_receipt(&warm_receipt)?;
    let summary = bitnet_benchmark_summary(
        &json_out,
        &ask_receipt,
        &ask,
        &warm_receipt,
        &warm,
        benchmark_start_peak_mb,
    )?;
    std::fs::write(&json_out, serde_json::to_vec_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", json_out.display()))?;
    validate_mac_receipt_value(&json_out, &summary)?;
    if !quiet {
        println!(
            "Mac BitNet benchmark summary written to {} (one-shot + fixed warm, chat=false, serve=false)",
            json_out.display()
        );
    }
    Ok(())
}

fn bitnet_benchmark_summary(
    json_out: &Path,
    ask_path: &Path,
    ask: &serde_json::Value,
    warm_path: &Path,
    warm: &serde_json::Value,
    benchmark_start_peak_mb: Option<f64>,
) -> Result<serde_json::Value> {
    let ask_summary = bitnet_one_shot_benchmark_path_summary(ask_path, ask)?;
    let warm_summary = bitnet_warm_benchmark_path_summary(warm_path, warm)?;
    let mut all_samples = BitnetBenchmarkMetricSamples::default();
    all_samples.extend_from_summary(&ask_summary);
    all_samples.extend_from_summary(&warm_summary);

    let prompt_count = ask_summary["prompt_count"].as_u64().unwrap_or_default()
        + warm_summary["prompt_count"].as_u64().unwrap_or_default();
    let generated_tokens = ask_summary["generated_tokens"].as_u64().unwrap_or_default()
        + warm_summary["generated_tokens"].as_u64().unwrap_or_default();
    let memory_end_peak_mb = peak_memory_mb();
    let mut memory = all_samples.memory_json();
    if let Some(object) = memory.as_object_mut() {
        object
            .insert("start_peak_memory_mb".to_string(), optional_f64_json(benchmark_start_peak_mb));
        object.insert("end_peak_memory_mb".to_string(), optional_f64_json(memory_end_peak_mb));
        object.insert(
            "process_peak_drift_mb".to_string(),
            optional_f64_json(memory_delta_mb(benchmark_start_peak_mb, memory_end_peak_mb)),
        );
    }
    let model_path = ask["model"]["path"]
        .as_str()
        .or_else(|| warm["model"]["path"].as_str())
        .unwrap_or_default();
    let tokenizer_path = ask["mac_bitnet_claim_boundary"]["tokenizer_path"]
        .as_str()
        .or_else(|| warm["mac_bitnet_claim_boundary"]["tokenizer_path"].as_str())
        .unwrap_or(BITNET_M4_DEFAULT_TOKENIZER_PATH);

    Ok(serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_apple_m4_benchmark_v1",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "artifact_path": json_out.display().to_string(),
        "requested_backend": APPLE_M4_CPU_NEON,
        "selected_backend": APPLE_M4_CPU_NEON,
        "runtime_api": "cpu",
        "fallback_used": false,
        "fallback_reason": serde_json::Value::Null,
        "benchmark_set": "bitnet-one-shot-fixed-warm-v1",
        "paths": {
            "one_shot": ask_summary,
            "fixed_warm": warm_summary,
        },
        "prompt_count": prompt_count,
        "generated_tokens": generated_tokens,
        "speed": all_samples.speed_json(),
        "memory": memory,
        "timeout_boundary": {
            "enforced": false,
            "reached": false,
            "status": "not_reached",
            "timeout_seconds": serde_json::Value::Null,
            "partial_failure_receipt": false,
        },
        "build": {
            "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "release_mode": !cfg!(debug_assertions),
        },
        "model": {
            "id": BITNET_M4_MODEL_ID,
            "family": "bitnet",
            "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
            "file": "ggml-model-i2_s.gguf",
            "path": model_path,
            "sha256": BITNET_M4_EXPECTED_MODEL_SHA256,
            "bytes": ask["model_cache"]["bytes"].as_u64().or_else(|| warm["model_cache"]["bytes"].as_u64()),
            "architecture": "bitnet_b1_58",
            "quantization": "I2_S",
            "answer_ready_artifact_available": true,
            "answer_ready": {
                "state": "answer_ready"
            }
        },
        "tokenizer": {
            "source": "external_tokenizer_json",
            "path": tokenizer_path,
            "sha256": BITNET_M4_EXPECTED_TOKENIZER_SHA256,
            "authority": "llama-bpe",
            "pretokenizer_authority": "llama-bpe",
            "strict": true
        },
        "prompt_template": BITNET_M4_PROMPT_TEMPLATE,
        "benchmark_contract": {
            "scope": "Apple M4 Mac mini BitNet one-shot and fixed-warm benchmark v1",
            "path_execution_model": "one mac ask run plus one fixed-prompt resident warm-session run",
            "one_shot_loaded_independently": true,
            "fixed_warm_model_tokenizer_reuse_visible": true,
            "cold_load_separated": true,
            "percentiles": ["p50", "p90", "p99"],
            "input_tok_s_definition": "prompt_tokens / (prompt_tokenize_ms + prefill_ms)",
            "output_tok_s_definition": "generated_tokens / total_prompt_wall_ms",
            "decode_tok_s_definition": "generated_tokens / decode_total_ms",
            "memory_drift_definition": "ru_maxrss process peak delta when available; monotonic peak, not live RSS",
            "timeout_boundary_recorded": true,
            "thresholds_are_claim_bounds_not_speed_guarantees": true
        },
        "evidence": {
            "one_shot_receipt": ask_path.display().to_string(),
            "warm_session_receipt": warm_path.display().to_string(),
            "warm_prompt_receipts": warm["session"]["per_prompt_receipts"].clone(),
            "generated_text_recorded": true,
            "generated_token_ids_recorded": true,
            "operator_commands": ["mac ask", "mac bitnet-warm"],
        },
        "mac_claim_boundary": {
            "bitnet_benchmark": true,
            "one_shot_mac_ask": true,
            "fixed_warm_session": true,
            "accepted_i2s_artifact_only": true,
            "dense_slm_evidence_used": false,
            "chat_enabled": false,
            "serve_enabled": false,
            "bitnet_quality_claimed": false,
            "broad_model_quality_claim": false,
            "broad_performance_claim": false,
            "speedup_claim": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "macbook_evidence": false,
            "broad_apple_silicon_claimed": false
        },
        "bitnet_quality_claimed": false,
        "broad_performance_claim": false,
        "speedup_claim": false,
    }))
}

fn bitnet_one_shot_benchmark_path_summary(
    path: &Path,
    receipt: &serde_json::Value,
) -> Result<serde_json::Value> {
    let summary = validate_mac_receipt_value(path, receipt)?;
    if summary.artifact_kind != "strict_bitnet_cpu_profile" {
        anyhow::bail!(
            "{} BitNet benchmark one-shot receipt must be strict_bitnet_cpu_profile",
            path.display()
        );
    }
    require_exact_string_at(path, receipt, &["operator_command"], "mac ask")?;
    require_exact_string_at(path, receipt, &["model", "family"], "bitnet")?;
    require_exact_string_at(path, receipt, &["model", "sha256"], BITNET_M4_EXPECTED_MODEL_SHA256)?;
    require_exact_string_at(path, receipt, &["loader", "mode"], "real_gguf")?;
    require_bool_at(path, receipt, &["tokenizer", "strict"], true)?;
    require_exact_string_at(path, receipt, &["tokenizer", "pretokenizer_authority"], "llama-bpe")?;
    require_bool_at(path, receipt, &["mac_bitnet_claim_boundary", "chat_enabled"], false)?;
    require_bool_at(path, receipt, &["mac_bitnet_claim_boundary", "serve_enabled"], false)?;

    let prompt_tokens = required_json_f64(&receipt["tokens"]["prompt"], "tokens.prompt")?;
    let generated_tokens = required_json_f64(&receipt["tokens"]["generated"], "tokens.generated")?;
    let model_load_ms =
        required_json_f64(&receipt["timing"]["model_load_ms"], "timing.model_load_ms")?;
    let tokenizer_load_ms =
        required_json_f64(&receipt["timing"]["tokenizer_load_ms"], "timing.tokenizer_load_ms")?;
    let tokenize_ms = required_json_f64(&receipt["timing"]["tokenize_ms"], "timing.tokenize_ms")?;
    let prefill_ms = required_json_f64(&receipt["timing"]["prefill_ms"], "timing.prefill_ms")?;
    let ttft_ms = receipt["timing"]["time_to_first_token_ms"]
        .as_f64()
        .or_else(|| receipt["timing"]["first_token_ms"].as_f64())
        .or_else(|| receipt["latency"]["cmd_to_first_ms"].as_f64())
        .ok_or_else(|| anyhow!("{} one-shot benchmark receipt is missing TTFT", path.display()))?;
    let decode_total_ms =
        required_json_f64(&receipt["timing"]["decode_total_ms"], "timing.decode_total_ms")?;
    let total_wall_ms = required_json_f64(&receipt["latency"]["total_ms"], "latency.total_ms")?;
    let sampling_ms_per_token =
        optional_positive_sample(&receipt["timing"]["sampling_ms_per_token"]);
    let mut input_tok_s = Vec::new();
    let mut output_tok_s = Vec::new();
    let mut decode_tok_s = Vec::new();
    push_positive_rate(&mut input_tok_s, prompt_tokens, tokenize_ms + prefill_ms);
    push_positive_rate(&mut output_tok_s, generated_tokens, total_wall_ms);
    push_positive_rate(&mut decode_tok_s, generated_tokens, decode_total_ms);
    let peak_memory_mb = optional_peak_memory_samples(&receipt["memory"]["peak_memory_mb"]);

    Ok(serde_json::json!({
        "path_id": "one_shot_mac_ask",
        "operator_command": "mac ask",
        "receipt_path": path.display().to_string(),
        "prompt_count": 1,
        "generated_tokens": generated_tokens as u64,
        "model_loaded_once": true,
        "tokenizer_loaded_once": true,
        "quality_passed": true,
        "prompt_tokens": benchmark_stat_json(&[prompt_tokens]),
        "output_tokens": benchmark_stat_json(&[generated_tokens]),
        "timing": {
            "model_load_ms": benchmark_stat_json(&[model_load_ms]),
            "tokenizer_load_ms": benchmark_stat_json(&[tokenizer_load_ms]),
            "prompt_tokenize_ms": benchmark_stat_json(&[tokenize_ms]),
            "prefill_ms": benchmark_stat_json(&[prefill_ms]),
            "time_to_first_token_ms": benchmark_stat_json(&[ttft_ms]),
            "decode_total_ms": benchmark_stat_json(&[decode_total_ms]),
            "sampling_ms_per_token": benchmark_stat_json(&sampling_ms_per_token),
            "total_wall_ms": benchmark_stat_json(&[total_wall_ms]),
        },
        "throughput": {
            "input_tokens_per_second": benchmark_stat_json(&input_tok_s),
            "output_tokens_per_second": benchmark_stat_json(&output_tok_s),
            "decode_tokens_per_second": benchmark_stat_json(&decode_tok_s),
        },
        "memory": {
            "peak_memory_mb": benchmark_stat_json(&peak_memory_mb),
            "memory_drift_mb": benchmark_stat_json(&[0.0]),
            "source": "getrusage.ru_maxrss process peak",
        },
        "timeout_boundary": {
            "enforced": false,
            "reached": false,
            "status": "not_reached",
        },
        "claim_boundary": {
            "scope": "this one-shot prompt, model, backend, and machine receipt only",
            "chat_enabled": false,
            "serve_enabled": false,
            "broad_performance_claim": false,
            "speedup_claim": false,
        },
    }))
}

fn bitnet_warm_benchmark_path_summary(
    path: &Path,
    receipt: &serde_json::Value,
) -> Result<serde_json::Value> {
    let summary = validate_mac_receipt_value(path, receipt)?;
    if summary.artifact_kind != "bitnet_apple_m4_warm_session" {
        anyhow::bail!(
            "{} BitNet benchmark warm receipt must be bitnet_apple_m4_warm_session",
            path.display()
        );
    }
    require_exact_string_at(path, receipt, &["operator_command"], "mac bitnet-warm")?;
    require_exact_string_at(path, receipt, &["model", "family"], "bitnet")?;
    require_exact_string_at(path, receipt, &["model", "sha256"], BITNET_M4_EXPECTED_MODEL_SHA256)?;
    require_exact_string_at(path, receipt, &["model", "loader_mode"], "real_gguf")?;
    require_bool_at(path, receipt, &["tokenizer", "strict"], true)?;
    require_exact_string_at(path, receipt, &["tokenizer", "pretokenizer_authority"], "llama-bpe")?;
    require_bool_at(path, receipt, &["mac_bitnet_claim_boundary", "chat_enabled"], false)?;
    require_bool_at(path, receipt, &["mac_bitnet_claim_boundary", "serve_enabled"], false)?;

    let prompt_receipts =
        receipt["session"]["per_prompt_receipts"].as_array().ok_or_else(|| {
            anyhow!(
                "{} BitNet warm benchmark receipt is missing per-prompt receipts",
                path.display()
            )
        })?;
    let prompt_count = receipt["session"]["prompt_count"].as_u64().unwrap_or_default();
    if prompt_receipts.len() != prompt_count as usize {
        anyhow::bail!("{} BitNet warm benchmark prompt receipt count mismatch", path.display());
    }

    let mut prompt_tokens = Vec::with_capacity(prompt_receipts.len());
    let mut generated_tokens = Vec::with_capacity(prompt_receipts.len());
    let mut tokenization_ms = Vec::with_capacity(prompt_receipts.len());
    let mut prefill_ms = Vec::with_capacity(prompt_receipts.len());
    let mut ttft_ms = Vec::with_capacity(prompt_receipts.len());
    let mut decode_total_ms = Vec::with_capacity(prompt_receipts.len());
    let mut sampling_ms_per_token = Vec::with_capacity(prompt_receipts.len());
    let mut total_wall_ms = Vec::with_capacity(prompt_receipts.len());
    let mut input_tok_s = Vec::with_capacity(prompt_receipts.len());
    let mut output_tok_s = Vec::with_capacity(prompt_receipts.len());
    let mut decode_tok_s = Vec::with_capacity(prompt_receipts.len());
    let mut generated_total = 0_u64;
    for prompt_receipt in prompt_receipts {
        let prompt_path = PathBuf::from(prompt_receipt.as_str().ok_or_else(|| {
            anyhow!("{} BitNet warm benchmark has non-string prompt receipt path", path.display())
        })?);
        let prompt = read_json_receipt(&prompt_path)?;
        let prompt_token_count = required_json_f64(&prompt["tokens"]["prompt"], "tokens.prompt")?;
        let generated_token_count =
            required_json_f64(&prompt["tokens"]["generated"], "tokens.generated")?;
        let tokenize = required_json_f64(&prompt["timing"]["tokenize_ms"], "timing.tokenize_ms")?;
        let prefill = required_json_f64(&prompt["timing"]["prefill_ms"], "timing.prefill_ms")?;
        let ttft = required_json_f64(
            &prompt["timing"]["time_to_first_token_ms"],
            "timing.time_to_first_token_ms",
        )
        .or_else(|_| {
            required_json_f64(&prompt["timing"]["first_token_ms"], "timing.first_token_ms")
        })?;
        let decode =
            required_json_f64(&prompt["timing"]["decode_total_ms"], "timing.decode_total_ms")?;
        let total = required_json_f64(&prompt["timing"]["total_ms"], "timing.total_ms")?;
        prompt_tokens.push(prompt_token_count);
        generated_tokens.push(generated_token_count);
        tokenization_ms.push(tokenize);
        prefill_ms.push(prefill);
        ttft_ms.push(ttft);
        decode_total_ms.push(decode);
        if let Some(sample_ms) = prompt["timing"]["sampling_ms_per_token"].as_f64() {
            sampling_ms_per_token.push(sample_ms);
        }
        total_wall_ms.push(total);
        push_positive_rate(&mut input_tok_s, prompt_token_count, tokenize + prefill);
        push_positive_rate(&mut output_tok_s, generated_token_count, total);
        push_positive_rate(&mut decode_tok_s, generated_token_count, decode);
        generated_total += generated_token_count as u64;
    }
    let peak_memory_mb = optional_peak_memory_samples(&receipt["memory"]["peak_memory_mb"]);

    Ok(serde_json::json!({
        "path_id": "fixed_warm_session",
        "operator_command": "mac bitnet-warm",
        "receipt_path": path.display().to_string(),
        "prompt_count": prompt_count,
        "generated_tokens": generated_total,
        "model_loaded_once": receipt["session"]["model_loaded_once"].as_bool().unwrap_or(false),
        "tokenizer_loaded_once": receipt["session"]["tokenizer_loaded_once"].as_bool().unwrap_or(false),
        "reuse_scope": receipt["session"]["reuse_scope"].clone(),
        "quality_passed": receipt["quality_summary"]["passed"].as_bool().unwrap_or(false),
        "prompt_tokens": benchmark_stat_json(&prompt_tokens),
        "output_tokens": benchmark_stat_json(&generated_tokens),
        "timing": {
            "model_load_ms": benchmark_stat_json(&[required_json_f64(&receipt["timing"]["model_load_ms"], "timing.model_load_ms")?]),
            "tokenizer_load_ms": benchmark_stat_json(&[required_json_f64(&receipt["timing"]["tokenizer_load_ms"], "timing.tokenizer_load_ms")?]),
            "prompt_tokenize_ms": benchmark_stat_json(&tokenization_ms),
            "prefill_ms": benchmark_stat_json(&prefill_ms),
            "time_to_first_token_ms": benchmark_stat_json(&ttft_ms),
            "decode_total_ms": benchmark_stat_json(&decode_total_ms),
            "sampling_ms_per_token": benchmark_stat_json(&sampling_ms_per_token),
            "total_wall_ms": benchmark_stat_json(&total_wall_ms),
        },
        "throughput": {
            "input_tokens_per_second": benchmark_stat_json(&input_tok_s),
            "output_tokens_per_second": benchmark_stat_json(&output_tok_s),
            "decode_tokens_per_second": benchmark_stat_json(&decode_tok_s),
        },
        "memory": {
            "peak_memory_mb": benchmark_stat_json(&peak_memory_mb),
            "memory_drift_mb": benchmark_stat_json(&[0.0]),
            "source": "getrusage.ru_maxrss process peak",
        },
        "timeout_boundary": {
            "enforced": false,
            "reached": false,
            "status": "not_reached",
        },
        "claim_boundary": {
            "scope": "this fixed warm prompt set, model, backend, and machine receipt only",
            "chat_enabled": false,
            "serve_enabled": false,
            "broad_performance_claim": false,
            "speedup_claim": false,
        },
    }))
}

fn optional_positive_sample(value: &serde_json::Value) -> Vec<f64> {
    value.as_f64().filter(|number| *number > 0.0).into_iter().collect()
}

fn optional_peak_memory_samples(value: &serde_json::Value) -> Vec<f64> {
    let mut samples = optional_positive_sample(value);
    if samples.is_empty()
        && let Some(sample) = peak_memory_mb().filter(|number| *number > 0.0)
    {
        samples.push(round3(sample));
    }
    samples
}

async fn run_doctor(
    model_id: &str,
    cache_dir: Option<PathBuf>,
    max_new_tokens: usize,
    threads: usize,
    json_out: PathBuf,
) -> Result<()> {
    let bitnet_ask_readiness = bitnet_mac_ask_readiness_json(cache_dir.clone());
    let cache_status =
        model_cache::apple_m4_slm_cache_status_json(model_id, cache_dir.clone(), true)?;
    let unsupported_backend_rejected =
        ensure_supported_mac_device(Some(APPLE_M4_METAL), "mac doctor unsupported-backend probe")
            .is_err();
    if !cache_status["ready"].as_bool().unwrap_or(false) {
        let next_step = cache_status["next_step"]
            .as_str()
            .unwrap_or("run `bitnet model fetch qwen2.5-0.5b-instruct-q8_0`");
        let failed = mac_doctor_base_receipt(
            &json_out,
            "fail",
            cache_status.clone(),
            bitnet_ask_readiness,
            unsupported_backend_rejected,
            max_new_tokens,
        );
        write_json_receipt(&json_out, &failed)?;
        anyhow::bail!("Mac doctor cannot pass because the model cache is not ready: {next_step}");
    }

    let model = model_cache::verified_apple_m4_slm_model(model_id, cache_dir)?;
    let smoke_receipt_path = sibling_receipt_path(&json_out, "smoke");
    run_smoke(
        MacSmokeModelFamily::DenseSlm,
        &model.id,
        Some(model.cache_root.clone()),
        None,
        None,
        max_new_tokens,
        threads,
        smoke_receipt_path.clone(),
    )
    .await?;
    let smoke_receipt = read_json_receipt(&smoke_receipt_path)?;
    let smoke_summary = validate_mac_receipt_value(&smoke_receipt_path, &smoke_receipt)?;
    let expected_fragment_found =
        smoke_receipt["expected_text_fragment_found"].as_bool().unwrap_or(false);
    if !expected_fragment_found {
        anyhow::bail!(
            "Mac doctor smoke check did not find expected fragment `{MAC_SMOKE_EXPECTED_FRAGMENT}`"
        );
    }
    if !unsupported_backend_rejected {
        anyhow::bail!("Mac doctor failed: unsupported apple-m4-metal request was not rejected");
    }

    let cache_status = verified_cache_status_json(&model);
    let mut receipt = mac_doctor_base_receipt(
        &json_out,
        "pass",
        cache_status,
        bitnet_ask_readiness,
        unsupported_backend_rejected,
        max_new_tokens,
    );
    let Some(object) = receipt.as_object_mut() else {
        anyhow::bail!("Mac doctor receipt is not an object");
    };
    object.insert("text".to_string(), smoke_receipt["text"].clone());
    object.insert("tokens".to_string(), smoke_receipt["tokens"].clone());
    object.insert("model".to_string(), smoke_receipt["model"].clone());
    object.insert("tokenizer".to_string(), smoke_receipt["tokenizer"].clone());
    object.insert("quality".to_string(), smoke_receipt["quality"].clone());
    object.insert("timing".to_string(), smoke_receipt["timing"].clone());
    object.insert("memory".to_string(), memory_receipt_json());
    object.insert(
        "smoke_receipt".to_string(),
        serde_json::json!({
            "path": smoke_receipt_path,
            "artifact_kind": smoke_summary.artifact_kind,
            "prompt_count": smoke_summary.prompt_count,
            "generated_tokens": smoke_summary.generated_tokens,
            "validated": smoke_summary.passed,
            "expected_text_fragment_found": expected_fragment_found,
        }),
    );
    object.insert(
        "model_cache".to_string(),
        serde_json::json!({
            "id": model.id,
            "display_name": model.display_name,
            "cache_root": model.cache_root,
            "path": model.path,
            "sha256": model.sha256,
            "bytes": model.bytes,
            "architecture": model.architecture,
            "quantization": model.quantization,
            "tokenizer_model": model.tokenizer_model,
            "tokenizer_pre": model.tokenizer_pre,
            "chat_template": model.chat_template,
            "support_note": model.support_note,
        }),
    );
    validate_mac_receipt_value(&json_out, &receipt)?;
    write_json_receipt(&json_out, &receipt)?;
    println!(
        "Mac doctor passed: {} (smoke receipt: {})",
        json_out.display(),
        smoke_receipt_path.display()
    );
    Ok(())
}

#[derive(Clone, Debug)]
struct MacServeGenerationDefaults {
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
}

#[derive(Clone, Debug)]
struct MacServeEndpoint {
    host: String,
    port: u16,
}

async fn run_mac_serve(
    model_id: String,
    cache_dir: Option<PathBuf>,
    device: String,
    endpoint: MacServeEndpoint,
    strict: bool,
    stream: bool,
    defaults: MacServeGenerationDefaults,
    receipt_dir: PathBuf,
) -> Result<()> {
    let MacServeEndpoint { host, port } = endpoint;
    if !strict {
        anyhow::bail!("mac serve requires strict mode; hidden fallback is not allowed");
    }
    if device != APPLE_M4_CPU_NEON {
        anyhow::bail!(
            "mac serve routes the supported Mac local service path through --device {APPLE_M4_CPU_NEON}; requested --device {device}. Full apple-m4-metal inference, MPSGraph inference, and hidden CPU fallback are not supported by this wrapper."
        );
    }
    let cache_status =
        model_cache::apple_m4_slm_cache_status_json(&model_id, cache_dir.clone(), true)?;
    if !cache_status["ready"].as_bool().unwrap_or(false) {
        let next_step = cache_status["next_step"]
            .as_str()
            .unwrap_or("run `bitnet model fetch qwen2.5-0.5b-instruct-q8_0`");
        anyhow::bail!("mac serve cannot start because the model cache is not ready: {next_step}");
    }
    std::fs::create_dir_all(&receipt_dir)
        .with_context(|| format!("failed to create receipt directory {}", receipt_dir.display()))?;
    ensure_mac_serve_receipt_dir_ready(&receipt_dir)?;
    let model = model_cache::verified_apple_m4_slm_model(&model_id, cache_dir)?;
    let generator = MacServeGenerator::load(&model)?;
    let state = Arc::new(MacServeState::new(
        model,
        host.clone(),
        port,
        stream,
        defaults,
        receipt_dir,
        Some(generator),
    ));
    let address = format!("{host}:{port}");
    if !mac_serve_host_is_loopback(&host) {
        eprintln!(
            "warning: bitnet mac serve is a local-service wrapper; binding to non-loopback host {host} may expose health/readiness state outside this machine"
        );
    }
    let listener = TcpListener::bind(&address)
        .await
        .with_context(|| format!("failed to bind M4 local server to {address}"))?;
    let local_addr =
        listener.local_addr().map(|addr| addr.to_string()).unwrap_or_else(|_| address.clone());
    eprintln!("{}", mac_serve_listening_line(&local_addr));
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (stream, _) = accepted.context("failed to accept M4 local server connection")?;
                let state = Arc::clone(&state);
                tokio::spawn(async move {
                    if let Err(error) = handle_mac_serve_connection(stream, state).await {
                        tracing::warn!(error = %error, "M4 local server connection failed");
                    }
                });
            }
            signal = tokio::signal::ctrl_c() => {
                signal.context("failed to listen for Ctrl-C while running mac serve")?;
                eprintln!("bitnet mac serve shutting down");
                return Ok(());
            }
        }
    }
}

struct MacServeState {
    started_at: std::time::Instant,
    started_at_utc: String,
    model: VerifiedCachedModel,
    cache_status: serde_json::Value,
    disk: serde_json::Value,
    host: String,
    port: u16,
    stream: bool,
    defaults: MacServeGenerationDefaults,
    receipt_dir: PathBuf,
    generator: Option<tokio::sync::Mutex<MacServeGenerator>>,
}

impl MacServeState {
    fn new(
        model: VerifiedCachedModel,
        host: String,
        port: u16,
        stream: bool,
        defaults: MacServeGenerationDefaults,
        receipt_dir: PathBuf,
        generator: Option<MacServeGenerator>,
    ) -> Self {
        let cache_status = verified_cache_status_json(&model);
        let disk = disk_health_json(&model.cache_root, model.bytes);
        Self {
            started_at: std::time::Instant::now(),
            started_at_utc: chrono::Utc::now().to_rfc3339(),
            model,
            cache_status,
            disk,
            host,
            port,
            stream,
            defaults,
            receipt_dir,
            generator: generator.map(tokio::sync::Mutex::new),
        }
    }

    fn uptime_seconds(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }
}

fn mac_serve_listening_line(local_addr: &str) -> String {
    format!(
        "bitnet mac serve listening on http://{local_addr} (health: /health, models: /models, ready: /ready)"
    )
}

struct MacServeGenerator {
    model: Arc<dyn bitnet_models::Model>,
    config: bitnet_common::BitNetConfig,
    tokenizer: Arc<dyn bitnet_tokenizers::Tokenizer + Send + Sync>,
    prompt_template: bitnet_inference::TemplateType,
    tokenizer_source: String,
    tokenizer_type: String,
    pretokenizer_authority: String,
    tokenizer_strict: bool,
}

impl MacServeGenerator {
    fn load(model: &VerifiedCachedModel) -> Result<Self> {
        use bitnet_common::Device;
        use bitnet_models::loader::{LoadConfig, ModelLoader};

        let loader = ModelLoader::new(Device::Cpu);
        let load_config =
            LoadConfig { use_mmap: true, validate_checksums: false, progress_callback: None };
        let loaded_model =
            loader.load_with_config(&model.path, &load_config).with_context(|| {
                format!("failed to load supported M4 dense SLM model {}", model.path.display())
            })?;
        let config = loaded_model.config().clone();
        let model_impl: Arc<dyn bitnet_models::Model> = Arc::from(loaded_model);
        let tokenizer_resolution =
            bitnet_tokenizers::auto::resolve_tokenizer(&model.path, None, true).with_context(
                || format!("failed to resolve strict tokenizer for {}", model.path.display()),
            )?;
        let tokenizer_source = tokenizer_resolution.source.as_str().to_string();
        let tokenizer_strict = tokenizer_resolution.strict;
        let tokenizer = tokenizer_resolution.tokenizer;
        let tokenizer_label =
            crate::infer_tokenizer_label(tokenizer.as_ref(), tokenizer_resolution.source);
        let pretokenizer_authority =
            crate::tokenizer_pretokenizer_authority(tokenizer_resolution.source, &tokenizer_label);
        let tokenizer_type =
            crate::tokenizer_type_for_receipt(&tokenizer_label, tokenizer_resolution.source);
        let prompt_template: bitnet_inference::TemplateType = QWEN_PROMPT_TEMPLATE
            .parse()
            .with_context(|| format!("invalid M4 server prompt template {QWEN_PROMPT_TEMPLATE}"))?;
        Ok(Self {
            model: model_impl,
            config,
            tokenizer,
            prompt_template,
            tokenizer_source,
            tokenizer_type,
            pretokenizer_authority: pretokenizer_authority.to_string(),
            tokenizer_strict,
        })
    }

    fn complete(
        &self,
        state: &MacServeState,
        request: MacServeCompletionRequest,
    ) -> Result<MacServeCompletion> {
        use bitnet_models::transformer::KVCache;
        use bitnet_sampling::{SamplingConfig, SamplingStrategy};

        let request_id = mac_serve_request_id();
        let prompt = request.prompt_text()?;
        let system_prompt = request.system_prompt();
        let max_new_tokens =
            request.max_new_tokens.or(request.max_tokens).unwrap_or(state.defaults.max_new_tokens);
        if max_new_tokens == 0 {
            anyhow::bail!("completion max_tokens/max_new_tokens must be greater than zero");
        }
        if max_new_tokens > 512 {
            anyhow::bail!(
                "completion max_tokens/max_new_tokens is capped at 512 for the M4 local server; got {max_new_tokens}"
            );
        }
        let temperature = request.temperature.unwrap_or(state.defaults.temperature);
        let top_k = request.top_k.unwrap_or(state.defaults.top_k);
        let top_p = request.top_p.unwrap_or(state.defaults.top_p);
        let repetition_penalty =
            request.repetition_penalty.unwrap_or(state.defaults.repetition_penalty);
        let seed = request.seed.or(state.defaults.seed);
        let stream = request.stream.unwrap_or(state.stream);
        let request_started = std::time::Instant::now();
        let formatted_prompt = self.prompt_template.apply(&prompt, system_prompt.as_deref());

        let mut all_stop_sequences = vec!["<|im_end|>".to_string()];
        for template_stop in self.prompt_template.default_stop_sequences() {
            if !all_stop_sequences.contains(&template_stop) {
                all_stop_sequences.push(template_stop);
            }
        }
        let mut all_stop_ids = Vec::new();
        for template_id in self.prompt_template.resolve_stop_token_ids(self.tokenizer.as_ref()) {
            if !all_stop_ids.contains(&template_id) {
                all_stop_ids.push(template_id);
            }
        }
        let max_stop_len = all_stop_sequences.iter().map(|value| value.len()).max().unwrap_or(0);

        let tokenize_start = std::time::Instant::now();
        let mut tokens = self.tokenizer.encode(
            &formatted_prompt,
            self.prompt_template.should_add_bos(),
            self.prompt_template.parse_special(),
        )?;
        let tokenize_ms = crate::elapsed_ms(tokenize_start);
        crate::ensure_non_empty_generation_context(&mut tokens, self.tokenizer.as_ref())?;
        let prompt_token_count = tokens.len();
        let prompt_token_ids = tokens.clone();
        let cache = KVCache::new(&self.config, 1, &candle_core::Device::Cpu)?;
        let mut any_cache: Box<dyn std::any::Any> = Box::new(cache);
        let mut sampler = SamplingStrategy::new(SamplingConfig {
            temperature,
            top_k: top_k as u32,
            top_p,
            repetition_penalty,
            seed,
        });
        sampler
            .reserve_logits_capacity(self.config.model.vocab_size.max(self.tokenizer.vocab_size()));

        let prefill_start = std::time::Instant::now();
        let mut prefill_token_count = 0usize;
        if tokens.len() > 1 {
            for token in &tokens[..tokens.len() - 1] {
                let x = self.model.embed(&[*token])?;
                let _ = self.model.forward(&x, any_cache.as_mut())?;
                prefill_token_count += 1;
            }
        }
        let prefill_ms =
            if prefill_token_count > 0 { crate::elapsed_ms(prefill_start) } else { 0.0 };

        let mut generated_token_ids = Vec::with_capacity(max_new_tokens);
        let mut token_texts = Vec::with_capacity(max_new_tokens);
        let mut decode_step_ms = Vec::with_capacity(max_new_tokens);
        let mut sample_step_ms = Vec::with_capacity(max_new_tokens);
        let mut stop_tail = String::with_capacity(max_stop_len);
        let mut first_token_ms = None;
        let mut finish_reason = "length";
        for _ in 0..max_new_tokens {
            let decode_step_start = std::time::Instant::now();
            let last_token = tokens.last().copied().expect("tokens must be non-empty");
            let x = self.model.embed(&[last_token])?;
            let h = self.model.forward(&x, any_cache.as_mut())?;
            let last_hidden = crate::extract_last_token_hidden(&h)?;
            let logits = self.model.logits(&last_hidden)?;
            let logits_vec = crate::extract_logits_2d(&logits)?;
            let sample_start = std::time::Instant::now();
            let next_token = sampler.sample(&logits_vec, &generated_token_ids)?;
            sample_step_ms.push(crate::elapsed_ms(sample_start));
            tokens.push(next_token);
            generated_token_ids.push(next_token);
            if first_token_ms.is_none() {
                first_token_ms = Some(request_started.elapsed().as_millis() as u64);
            }
            let token_text = self.tokenizer.decode(&[next_token])?;
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
            token_texts.push(token_text);
            decode_step_ms.push(crate::elapsed_ms(decode_step_start));

            if all_stop_ids.contains(&next_token) {
                finish_reason = "stop";
                break;
            }
            if let Some(eos) = self.tokenizer.eos_token_id()
                && next_token == eos
            {
                finish_reason = "stop";
                break;
            }
            if max_stop_len > 0
                && !all_stop_sequences.is_empty()
                && all_stop_sequences.iter().any(|pat| stop_tail.ends_with(pat))
            {
                finish_reason = "stop";
                break;
            }
        }

        let text = self.tokenizer.decode(&generated_token_ids)?;
        let decode_ms = decode_step_ms.iter().sum::<f64>();
        let sampling_ms = sample_step_ms.iter().sum::<f64>();
        let total_ms = crate::elapsed_ms(request_started);
        let receipt_path = state.receipt_dir.join(format!("{request_id}.json"));
        let receipt = serde_json::json!({
            "schema_version": "1.0.0",
            "artifact_kind": "bitnet_apple_m4_local_server_completion",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "request_id": request_id,
            "artifact_path": receipt_path.display().to_string(),
            "requested_backend": APPLE_M4_CPU_NEON,
            "selected_backend": APPLE_M4_CPU_NEON,
            "runtime_api": "cpu",
            "fallback_used": false,
            "fallback_reason": serde_json::Value::Null,
            "server": mac_serve_server_json(state),
            "model": {
                "id": &state.model.id,
                "display_name": &state.model.display_name,
                "path": &state.model.path,
                "sha256": &state.model.sha256,
                "sha256_source": "verified_cache_metadata_and_startup_check",
                "bytes": state.model.bytes,
                "architecture": &state.model.architecture,
                "quantization": &state.model.quantization,
            },
            "tokenizer": {
                "type": self.tokenizer_type,
                "source": self.tokenizer_source,
                "strict": self.tokenizer_strict,
                "pretokenizer_authority": self.pretokenizer_authority,
                "prompt_template": QWEN_PROMPT_TEMPLATE,
                "bos": self.tokenizer.bos_token_id().unwrap_or(1),
                "eos": self.tokenizer.eos_token_id().unwrap_or(2),
            },
            "request": {
                "model": request.model,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "stream": stream,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "seed": seed,
            },
            "generation": {
                "mode": if temperature == 0.0 && top_k == 1 { "greedy" } else { "sampling" },
                "text": text,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_token_count,
                "generated_tokens": generated_token_ids.len(),
                "prompt_token_ids": prompt_token_ids,
                "generated_token_ids": generated_token_ids,
                "token_texts": token_texts,
            },
            "timing": {
                "model_load_ms": 0.0,
                "tokenizer_load_ms": 0.0,
                "tokenize_ms": crate::rounded_ms(tokenize_ms),
                "prefill_ms": crate::rounded_ms(prefill_ms),
                "first_token_ms": first_token_ms,
                "time_to_first_token_ms": first_token_ms,
                "decode_ms": crate::rounded_ms(decode_ms),
                "sampling_ms": crate::rounded_ms(sampling_ms),
                "total_ms": crate::rounded_ms(total_ms),
                "decode_step_ms": crate::timing_samples_json(&decode_step_ms),
                "sample_step_ms": crate::timing_samples_json(&sample_step_ms),
            },
            "session_reuse": {
                "reuse_scope": "resident_server",
                "model_loaded_at_startup": true,
                "tokenizer_loaded_at_startup": true,
                "request_serialized": true,
                "kv_cache_reuse_policy": "recreated_per_request_for_prompt_isolation",
            },
            "claim_boundary": {
                "local_server_completion_endpoint": true,
                "streaming_transport": stream,
                "openai_compatibility_claimed": false,
                "production_readiness_claimed": false,
                "bitnet_quality_claimed": false,
                "full_metal_inference_claimed": false,
                "mpsgraph_inference_claimed": false,
                "neural_engine_execution_claimed": false,
                "qk256_apple_claimed": false,
                "broad_performance_claim": false,
            },
        });
        write_json_receipt(&receipt_path, &receipt)?;
        Ok(MacServeCompletion {
            request_id,
            model_id: state.model.id.clone(),
            text: receipt["generation"]["text"].as_str().unwrap_or_default().to_string(),
            token_texts: receipt["generation"]["token_texts"]
                .as_array()
                .map(|items| {
                    items.iter().filter_map(|item| item.as_str().map(ToOwned::to_owned)).collect()
                })
                .unwrap_or_default(),
            generated_token_ids: receipt["generation"]["generated_token_ids"]
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_u64().map(|value| value as u32))
                        .collect()
                })
                .unwrap_or_default(),
            finish_reason: receipt["generation"]["finish_reason"]
                .as_str()
                .unwrap_or("length")
                .to_string(),
            stream,
            receipt_path,
            receipt,
        })
    }
}

#[derive(Debug)]
struct MacServeCompletion {
    request_id: String,
    model_id: String,
    text: String,
    token_texts: Vec<String>,
    generated_token_ids: Vec<u32>,
    finish_reason: String,
    stream: bool,
    receipt_path: PathBuf,
    receipt: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize)]
struct MacServeCompletionRequest {
    model: Option<String>,
    prompt: Option<String>,
    messages: Option<Vec<MacServeChatMessage>>,
    max_tokens: Option<usize>,
    max_new_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    seed: Option<u64>,
    stream: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
struct MacServeChatMessage {
    role: String,
    content: String,
}

impl MacServeCompletionRequest {
    fn prompt_text(&self) -> Result<String> {
        if let Some(prompt) = self.prompt.as_deref().map(str::trim)
            && !prompt.is_empty()
        {
            return Ok(prompt.to_string());
        }
        let messages = self
            .messages
            .as_ref()
            .ok_or_else(|| anyhow!("completion request requires `prompt` or `messages`"))?;
        messages
            .iter()
            .rev()
            .find(|message| message.role.eq_ignore_ascii_case("user"))
            .map(|message| message.content.trim().to_string())
            .filter(|content| !content.is_empty())
            .ok_or_else(|| {
                anyhow!("completion request messages must include a non-empty user message")
            })
    }

    fn system_prompt(&self) -> Option<String> {
        self.messages.as_ref().and_then(|messages| {
            let text = messages
                .iter()
                .filter(|message| message.role.eq_ignore_ascii_case("system"))
                .map(|message| message.content.trim())
                .filter(|content| !content.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");
            (!text.is_empty()).then_some(text)
        })
    }
}

async fn handle_mac_serve_connection(
    mut stream: TcpStream,
    state: Arc<MacServeState>,
) -> Result<()> {
    let request = read_mac_serve_http_request(&mut stream).await?;
    if request.is_empty() {
        return Ok(());
    }
    let response = mac_serve_http_reply(&request, &state).await?;
    let header = format!(
        "HTTP/1.1 {} {}\r\ncontent-type: {}\r\ncache-control: no-store\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        response.status,
        response.reason,
        response.content_type,
        response.body.len()
    );
    stream.write_all(header.as_bytes()).await.context("failed to write HTTP header")?;
    stream.write_all(&response.body).await.context("failed to write HTTP body")?;
    stream.shutdown().await.context("failed to close HTTP stream")?;
    Ok(())
}

struct MacServeHttpReply {
    status: u16,
    reason: &'static str,
    content_type: &'static str,
    body: Vec<u8>,
}

impl MacServeHttpReply {
    fn json(status: u16, reason: &'static str, body: serde_json::Value) -> Result<Self> {
        Ok(Self {
            status,
            reason,
            content_type: "application/json",
            body: serde_json::to_vec_pretty(&body)?,
        })
    }

    fn sse(body: String) -> Self {
        Self {
            status: 200,
            reason: "OK",
            content_type: "text/event-stream",
            body: body.into_bytes(),
        }
    }
}

async fn read_mac_serve_http_request(stream: &mut TcpStream) -> Result<String> {
    let mut buffer = Vec::with_capacity(8192);
    let mut chunk = [0u8; 4096];
    loop {
        let read = stream.read(&mut chunk).await.context("failed to read HTTP request")?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if buffer.len() > 1_048_576 {
            anyhow::bail!("M4 local server request exceeded 1 MiB limit");
        }
        if mac_serve_http_request_complete(&buffer) {
            break;
        }
    }
    Ok(String::from_utf8_lossy(&buffer).to_string())
}

fn mac_serve_http_request_complete(buffer: &[u8]) -> bool {
    let Some(header_end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") else {
        return false;
    };
    let header_len = header_end + 4;
    let headers = String::from_utf8_lossy(&buffer[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    buffer.len() >= header_len.saturating_add(content_length)
}

async fn mac_serve_http_reply(request: &str, state: &MacServeState) -> Result<MacServeHttpReply> {
    let (method, path) = mac_serve_request_method_path(request);
    if method == "POST" && path == "/v1/chat/completions" {
        return mac_serve_completion_http_reply(request, state).await;
    }
    if method == "GET" && path.starts_with("/receipts/") {
        return mac_serve_receipt_http_reply(path, state);
    }
    let (status, reason, body) = mac_serve_http_response(request, state);
    MacServeHttpReply::json(status, reason, body)
}

fn mac_serve_http_response(
    request: &str,
    state: &MacServeState,
) -> (u16, &'static str, serde_json::Value) {
    let (method, path) = mac_serve_request_method_path(request);
    if method != "GET" {
        return (
            405,
            "Method Not Allowed",
            serde_json::json!({
                "status": "error",
                "error": "method_not_allowed",
                "allowed_methods": ["GET"],
            }),
        );
    }
    match path {
        "/" | "/health" | "/health/live" => (200, "OK", mac_serve_health_json(state)),
        "/models" => match mac_serve_models_json(state) {
            Ok(body) => (200, "OK", body),
            Err(error) => (
                500,
                "Internal Server Error",
                serde_json::json!({
                    "status": "error",
                    "error": "models_catalog_failed",
                    "message": error.to_string(),
                }),
            ),
        },
        "/ready" | "/health/ready" => {
            let ready = mac_serve_ready_json(state);
            let status = if ready["ready"].as_bool() == Some(true) { 200 } else { 503 };
            let reason = if status == 200 { "OK" } else { "Service Unavailable" };
            (status, reason, ready)
        }
        _ => (
            404,
            "Not Found",
            serde_json::json!({
                "status": "error",
                "error": "not_found",
                "available_endpoints": ["/health", "/health/live", "/models", "/ready", "/health/ready", "/v1/chat/completions", "/receipts/{id}"],
            }),
        ),
    }
}

fn mac_serve_receipt_http_reply(path: &str, state: &MacServeState) -> Result<MacServeHttpReply> {
    let receipt_id = path.trim_start_matches("/receipts/").trim();
    let Some(receipt_stem) = mac_serve_normalize_receipt_id(receipt_id) else {
        return MacServeHttpReply::json(
            400,
            "Bad Request",
            serde_json::json!({
                "status": "error",
                "error": "invalid_receipt_id",
                "message": "receipt id must be a single safe file stem, for example m4srv-123",
            }),
        );
    };
    let receipt_path = state.receipt_dir.join(format!("{receipt_stem}.json"));
    if !receipt_path.exists() {
        return MacServeHttpReply::json(
            404,
            "Not Found",
            serde_json::json!({
                "status": "error",
                "error": "receipt_not_found",
                "receipt_id": receipt_stem,
                "receipt_dir": &state.receipt_dir,
            }),
        );
    }
    let canonical_dir = std::fs::canonicalize(&state.receipt_dir).with_context(|| {
        format!("failed to canonicalize receipt directory {}", state.receipt_dir.display())
    })?;
    let canonical_receipt = std::fs::canonicalize(&receipt_path)
        .with_context(|| format!("failed to canonicalize receipt {}", receipt_path.display()))?;
    if !canonical_receipt.starts_with(&canonical_dir) {
        return MacServeHttpReply::json(
            400,
            "Bad Request",
            serde_json::json!({
                "status": "error",
                "error": "invalid_receipt_id",
                "message": "receipt path escapes the configured receipt directory",
            }),
        );
    }
    let receipt_text = std::fs::read_to_string(&canonical_receipt)
        .with_context(|| format!("failed to read receipt {}", canonical_receipt.display()))?;
    let receipt: serde_json::Value = serde_json::from_str(&receipt_text)
        .with_context(|| format!("receipt is not valid JSON: {}", canonical_receipt.display()))?;
    MacServeHttpReply::json(200, "OK", receipt)
}

fn mac_serve_normalize_receipt_id(receipt_id: &str) -> Option<String> {
    let receipt_id = receipt_id.strip_suffix(".json").unwrap_or(receipt_id);
    if receipt_id.is_empty()
        || receipt_id.contains('/')
        || receipt_id.contains('\\')
        || receipt_id.contains("..")
    {
        return None;
    }
    let safe = receipt_id.chars().all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'));
    safe.then(|| receipt_id.to_string())
}

fn run_mac_serve_check(
    url: &str,
    completion: bool,
    prompt: &str,
    max_new_tokens: usize,
    json_out: PathBuf,
) -> Result<()> {
    let base = MacServeCheckEndpoint::parse(url)?;
    let ready_response = mac_serve_check_http_json(&base, "GET", "/ready", None)?;
    let ready_json = ready_response.body.clone();
    let ready_pass = ready_response.status == 200
        && ready_json["ready"].as_bool() == Some(true)
        && ready_json["backend"]["selected_backend"].as_str() == Some(APPLE_M4_CPU_NEON)
        && ready_json["backend"]["fallback_used"].as_bool() == Some(false);
    let models_response = mac_serve_check_http_json(&base, "GET", "/models", None)?;
    let models_json = models_response.body.clone();
    let models_pass = mac_serve_check_models_catalog_pass(models_response.status, &models_json);
    let models_result = mac_serve_check_models_result(models_response.status, &models_json);

    let mut completion_result = serde_json::json!({
        "executed": false,
        "passed": serde_json::Value::Null,
    });
    let mut receipt_export_result = serde_json::json!({
        "executed": false,
        "passed": serde_json::Value::Null,
    });

    if completion {
        let request = serde_json::json!({
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "stream": false,
        });
        let completion_response =
            mac_serve_check_http_json(&base, "POST", "/v1/chat/completions", Some(&request))?;
        let completion_json = completion_response.body.clone();
        let receipt_id = completion_json["receipt_path"]
            .as_str()
            .and_then(|path| path.rsplit('/').next())
            .and_then(mac_serve_normalize_receipt_id);
        let completion_pass = completion_response.status == 200
            && completion_json["receipt"]["selected_backend"].as_str() == Some(APPLE_M4_CPU_NEON)
            && completion_json["receipt"]["fallback_used"].as_bool() == Some(false)
            && receipt_id.is_some();
        completion_result = serde_json::json!({
            "executed": true,
            "status": completion_response.status,
            "passed": completion_pass,
            "request_id": completion_json["id"],
            "receipt_id": receipt_id,
            "generated_tokens": completion_json["usage"]["completion_tokens"],
            "finish_reason": completion_json["choices"][0]["finish_reason"],
        });

        if let Some(receipt_id) = receipt_id {
            let receipt_path = format!("/receipts/{receipt_id}");
            let receipt_response = mac_serve_check_http_json(&base, "GET", &receipt_path, None)?;
            let receipt_json = receipt_response.body.clone();
            let export_pass = receipt_response.status == 200
                && receipt_json["request_id"] == completion_json["id"]
                && receipt_json["selected_backend"].as_str() == Some(APPLE_M4_CPU_NEON)
                && receipt_json["fallback_used"].as_bool() == Some(false);
            receipt_export_result = serde_json::json!({
                "executed": true,
                "status": receipt_response.status,
                "passed": export_pass,
                "request_id": receipt_json["request_id"],
                "artifact_kind": receipt_json["artifact_kind"],
                "selected_backend": receipt_json["selected_backend"],
                "fallback_used": receipt_json["fallback_used"],
            });
        }
    }

    let completion_pass = completion_result["passed"].as_bool().unwrap_or(!completion);
    let receipt_export_pass = receipt_export_result["passed"].as_bool().unwrap_or(!completion);
    let passed = ready_pass && models_pass && completion_pass && receipt_export_pass;
    let receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_apple_m4_local_server_check",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "artifact_path": json_out.display().to_string(),
        "result": if passed { "pass" } else { "fail" },
        "server": {
            "url": url,
            "ready_endpoint": "/ready",
            "models_endpoint": "/models",
            "completion_endpoint": "/v1/chat/completions",
            "receipt_export_endpoint": "/receipts/{id}",
        },
        "checks": {
            "ready": {
                "executed": true,
                "status": ready_response.status,
                "passed": ready_pass,
                "ready": ready_json["ready"],
                "selected_backend": ready_json["backend"]["selected_backend"],
                "fallback_used": ready_json["backend"]["fallback_used"],
            },
            "models": models_result,
            "completion": completion_result,
            "receipt_export": receipt_export_result,
        },
        "claim_boundary": {
            "server_readiness_checked": true,
            "model_catalog_checked": true,
            "completion_probe_executed": completion,
            "receipt_export_checked": completion,
            "production_readiness_claimed": false,
            "openai_compatibility_claimed": false,
            "bitnet_quality_claimed": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false,
        },
    });
    write_json_receipt(&json_out, &receipt)?;
    if passed {
        println!("mac serve-check passed: {}", json_out.display());
        Ok(())
    } else {
        anyhow::bail!("mac serve-check failed; receipt written to {}", json_out.display())
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct MacServeCheckEndpoint {
    host: String,
    port: u16,
}

impl MacServeCheckEndpoint {
    fn parse(url: &str) -> Result<Self> {
        let rest = url
            .trim()
            .strip_prefix("http://")
            .ok_or_else(|| anyhow!("mac serve-check currently supports http://HOST:PORT only"))?;
        let authority = rest.trim_end_matches('/');
        if authority.is_empty() || authority.contains('/') {
            anyhow::bail!("mac serve-check URL must be a bare http://HOST:PORT base URL");
        }
        let (host, port) = authority
            .rsplit_once(':')
            .ok_or_else(|| anyhow!("mac serve-check URL must include an explicit port"))?;
        let port = port
            .parse::<u16>()
            .with_context(|| format!("invalid mac serve-check port in {url}"))?;
        if host.trim().is_empty() {
            anyhow::bail!("mac serve-check URL host must not be empty");
        }
        Ok(Self { host: host.trim_start_matches('[').trim_end_matches(']').to_string(), port })
    }
}

struct MacServeCheckHttpResponse {
    status: u16,
    body: serde_json::Value,
}

fn mac_serve_check_models_result(status: u16, body: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "executed": true,
        "status": status,
        "passed": mac_serve_check_models_catalog_pass(status, body),
        "artifact_kind": body["artifact_kind"],
        "default_model_id": body["catalog"]["default_model_id"],
        "resident_model_id": body["resident_model_id"],
        "generation_executed": body["generation_executed"],
        "bitnet_ask_only": mac_serve_models_catalog_has_bitnet_ask_only(body),
        "bitnet_warm_session": mac_serve_models_catalog_has_bitnet_warm(body),
        "disk_available_bytes": body["catalog"]["disk"]["available_bytes"],
        "recommended_first_model_id": body["catalog"]["disk"]["recommended_first_model_id"],
        "recommended_fetch_command": mac_serve_models_catalog_recommended_command(body, "fetch_command"),
        "recommended_verify_command": mac_serve_models_catalog_recommended_command(body, "verify_command"),
    })
}

fn mac_serve_check_models_catalog_pass(status: u16, body: &serde_json::Value) -> bool {
    status == 200
        && body["artifact_kind"].as_str() == Some("bitnet_apple_m4_local_server_models")
        && body["status"].as_str() == Some("ok")
        && body["resident_model_id"].as_str().is_some()
        && body["generation_executed"].as_bool() == Some(false)
        && body["claim_boundary"]["bitnet_quality_claimed"].as_bool() == Some(false)
        && body["claim_boundary"]["full_metal_inference_claimed"].as_bool() == Some(false)
        && body["catalog"]["default_model_id"].as_str()
            == Some(model_cache::M4_SLM_RUNTIME_MODEL_ID)
        && body["catalog"]["disk"]["default_model_headroom_bytes"].as_u64().is_some()
        && mac_serve_models_catalog_has_supported_model(body, model_cache::M4_SLM_RUNTIME_MODEL_ID)
        && mac_serve_models_catalog_has_supported_model(body, "qwen2.5-1.5b-instruct-q4_k_m")
        && mac_serve_models_catalog_has_bitnet_ask_only(body)
        && mac_serve_models_catalog_has_bitnet_warm(body)
        && mac_serve_models_catalog_recommended_commands_are_coherent(body)
}

fn mac_serve_models_catalog_has_supported_model(body: &serde_json::Value, model_id: &str) -> bool {
    body["catalog"]["rows"].as_array().is_some_and(|rows| {
        rows.iter().any(|row| {
            row["id"].as_str() == Some(model_id)
                && matches!(row["state"].as_str(), Some("default" | "supported"))
                && row["fetch_command"].as_str().is_some()
        })
    })
}

fn mac_serve_models_catalog_has_bitnet_ask_only(body: &serde_json::Value) -> bool {
    body["catalog"]["rows"].as_array().is_some_and(|rows| {
        rows.iter().any(|row| {
            row["id"].as_str() == Some("microsoft-bitnet-b1.58-2B-4T-i2s")
                && row["state"].as_str() == Some("supported-ask")
                && row["mac_ask_enabled"].as_bool() == Some(true)
                && row["mac_bitnet_warm_enabled"].as_bool() == Some(true)
                && row["mac_chat_enabled"].as_bool() == Some(false)
                && row["mac_ask_chat_enabled"].as_bool() == Some(false)
                && row["mac_serve_enabled"].as_bool() == Some(false)
                && row["proof_status"].as_str()
                    == Some("answer-corpus-and-warm-session-proof-passed-explicit-artifact")
                && row["proof_command"].as_str().is_some_and(|command| {
                    command.contains("mac bitnet-proof")
                        && command.contains("--proof-receipt")
                        && command.contains("bitnet-answer-corpus-full-release.json")
                })
                && row["fetch_command"]
                    .as_str()
                    .is_some_and(|command| command.contains("bitnet model fetch microsoft-bitnet"))
                && row["recommended_fetch_headroom_bytes"].as_u64().is_some()
        })
    })
}

fn mac_serve_models_catalog_has_bitnet_warm(body: &serde_json::Value) -> bool {
    body["catalog"]["rows"].as_array().is_some_and(|rows| {
        rows.iter().any(|row| {
            row["id"].as_str() == Some("microsoft-bitnet-b1.58-2B-4T-i2s")
                && row["mac_bitnet_warm_enabled"].as_bool() == Some(true)
                && row["warm_command"].as_str().is_some_and(|command| {
                    command.contains("mac bitnet-warm")
                        && command.contains("--model-path")
                        && command.contains("--tokenizer")
                })
                && row["warm_receipt_path"].as_str().is_some_and(|path| {
                    path.contains("ci/hardware/apple-m4-mac-mini/2026-05-14/bitnet-warm")
                        && path.ends_with("bitnet-mac-bitnet-warm-runtime-receipt.json")
                })
                && row["mac_chat_enabled"].as_bool() == Some(false)
                && row["mac_serve_enabled"].as_bool() == Some(false)
        })
    })
}

fn mac_serve_models_catalog_recommended_commands_are_coherent(body: &serde_json::Value) -> bool {
    let Some(model_id) = body["catalog"]["disk"]["recommended_first_model_id"].as_str() else {
        return true;
    };
    body["catalog"]["rows"].as_array().is_some_and(|rows| {
        rows.iter().any(|row| {
            row["id"].as_str() == Some(model_id)
                && matches!(row["state"].as_str(), Some("default" | "supported"))
                && row["fetch_command"].as_str().is_some()
                && row["verify_command"].as_str().is_some()
        })
    })
}

fn mac_serve_models_catalog_recommended_command(
    body: &serde_json::Value,
    command_key: &str,
) -> serde_json::Value {
    let Some(model_id) = body["catalog"]["disk"]["recommended_first_model_id"].as_str() else {
        return serde_json::Value::Null;
    };
    body["catalog"]["rows"]
        .as_array()
        .and_then(|rows| rows.iter().find(|row| row["id"].as_str() == Some(model_id)))
        .and_then(|row| row[command_key].as_str())
        .map(serde_json::Value::from)
        .unwrap_or(serde_json::Value::Null)
}

fn mac_serve_check_http_json(
    endpoint: &MacServeCheckEndpoint,
    method: &str,
    path: &str,
    body: Option<&serde_json::Value>,
) -> Result<MacServeCheckHttpResponse> {
    let host_authority = if endpoint.host.contains(':') {
        format!("[{}]", endpoint.host)
    } else {
        endpoint.host.clone()
    };
    let addr = format!("{}:{}", host_authority, endpoint.port);
    let mut stream = std::net::TcpStream::connect(&addr)
        .with_context(|| format!("failed to connect to M4 local server at {addr}"))?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(std::time::Duration::from_secs(30)))?;
    let body_text = match body {
        Some(value) => serde_json::to_string(value)?,
        None => String::new(),
    };
    let request = format!(
        "{method} {path} HTTP/1.1\r\nhost: {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        host_authority,
        body_text.len(),
        body_text
    );
    stream.write_all(request.as_bytes()).context("failed to write server check request")?;
    let mut response = String::new();
    stream.read_to_string(&mut response).context("failed to read server check response")?;
    let (head, body) = response
        .split_once("\r\n\r\n")
        .ok_or_else(|| anyhow!("M4 local server returned a malformed HTTP response"))?;
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| anyhow!("M4 local server response did not include a status code"))?;
    let body = serde_json::from_str(body).context("M4 local server response body was not JSON")?;
    Ok(MacServeCheckHttpResponse { status, body })
}

async fn mac_serve_completion_http_reply(
    request: &str,
    state: &MacServeState,
) -> Result<MacServeHttpReply> {
    let body = mac_serve_http_body(request);
    let completion_request: MacServeCompletionRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(error) => {
            return MacServeHttpReply::json(
                400,
                "Bad Request",
                serde_json::json!({
                    "status": "error",
                    "error": "invalid_completion_request",
                    "message": error.to_string(),
                }),
            );
        }
    };
    if let Some(requested_model) = completion_request.model.as_deref().map(str::trim)
        && !requested_model.is_empty()
        && requested_model != state.model.id
    {
        return MacServeHttpReply::json(
            400,
            "Bad Request",
            serde_json::json!({
                "status": "error",
                "error": "unsupported_model",
                "message": format!(
                    "mac serve has resident model {}; request asked for {requested_model}",
                    state.model.id
                ),
            }),
        );
    }
    let Some(generator) = state.generator.as_ref() else {
        return MacServeHttpReply::json(
            503,
            "Service Unavailable",
            serde_json::json!({
                "status": "error",
                "error": "generation_unavailable",
                "reason": "M4 local server state was created without a resident generator",
            }),
        );
    };
    let completion = {
        let generator = generator.lock().await;
        match generator.complete(state, completion_request) {
            Ok(completion) => completion,
            Err(error) => {
                return MacServeHttpReply::json(
                    500,
                    "Internal Server Error",
                    serde_json::json!({
                        "status": "error",
                        "error": "completion_failed",
                        "message": error.to_string(),
                        "claim_boundary": {
                            "openai_compatibility_claimed": false,
                            "production_readiness_claimed": false,
                            "bitnet_quality_claimed": false,
                            "full_metal_inference_claimed": false,
                        },
                    }),
                );
            }
        }
    };
    if completion.stream {
        Ok(MacServeHttpReply::sse(mac_serve_completion_sse(&completion)?))
    } else {
        MacServeHttpReply::json(200, "OK", mac_serve_completion_json(&completion))
    }
}

fn mac_serve_request_method_path(request: &str) -> (&str, &str) {
    let mut parts = request.lines().next().unwrap_or_default().split_whitespace();
    let method = parts.next().unwrap_or_default();
    let path = parts.next().unwrap_or_default().split('?').next().unwrap_or_default();
    (method, path)
}

fn mac_serve_http_body(request: &str) -> &str {
    request
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .or_else(|| request.split_once("\n\n").map(|(_, body)| body))
        .unwrap_or_default()
}

fn mac_serve_completion_json(completion: &MacServeCompletion) -> serde_json::Value {
    serde_json::json!({
        "id": completion.request_id,
        "object": "chat.completion",
        "model": completion.model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion.text,
                },
                "finish_reason": completion.finish_reason,
            }
        ],
        "receipt_path": completion.receipt_path,
        "receipt": completion.receipt,
        "usage": {
            "completion_tokens": completion.generated_token_ids.len(),
        },
        "claim_boundary": {
            "openai_compatibility_claimed": false,
            "production_readiness_claimed": false,
        },
    })
}

fn mac_serve_completion_sse(completion: &MacServeCompletion) -> Result<String> {
    let mut output = String::new();
    output.push_str("event: metadata\n");
    output.push_str("data: ");
    output.push_str(&serde_json::to_string(&serde_json::json!({
        "id": completion.request_id,
        "object": "chat.completion.chunk",
        "model": completion.model_id,
        "receipt_path": completion.receipt_path,
        "generated_token_ids": completion.generated_token_ids,
        "claim_boundary": {
            "openai_compatibility_claimed": false,
            "production_readiness_claimed": false,
        },
    }))?);
    output.push_str("\n\n");
    for token_text in &completion.token_texts {
        output.push_str("data: ");
        output.push_str(&serde_json::to_string(&serde_json::json!({
            "id": completion.request_id,
            "object": "chat.completion.chunk",
            "model": completion.model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": token_text,
                    },
                    "finish_reason": serde_json::Value::Null,
                }
            ],
        }))?);
        output.push_str("\n\n");
    }
    output.push_str("data: ");
    output.push_str(&serde_json::to_string(&serde_json::json!({
        "id": completion.request_id,
        "object": "chat.completion.chunk",
        "model": completion.model_id,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": completion.finish_reason,
            }
        ],
    }))?);
    output.push_str("\n\ndata: [DONE]\n\n");
    Ok(output)
}

fn mac_serve_health_json(state: &MacServeState) -> serde_json::Value {
    serde_json::json!({
        "artifact_kind": "bitnet_apple_m4_local_server_health",
        "status": "healthy",
        "server": mac_serve_server_json(state),
        "uptime_seconds": state.uptime_seconds(),
        "generation_executed": false,
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "ready": "/ready",
            "completions": "/v1/chat/completions",
            "receipts": "/receipts/{id}",
        },
        "claim_boundary": mac_serve_claim_boundary_json(),
    })
}

fn mac_serve_models_json(state: &MacServeState) -> Result<serde_json::Value> {
    let catalog = model_cache::apple_m4_models_catalog_json(Some(state.model.cache_root.clone()))?;
    Ok(serde_json::json!({
        "artifact_kind": "bitnet_apple_m4_local_server_models",
        "status": "ok",
        "server": mac_serve_server_json(state),
        "resident_model_id": &state.model.id,
        "generation_executed": false,
        "catalog": catalog,
        "claim_boundary": mac_serve_claim_boundary_json(),
    }))
}

fn mac_serve_ready_json(state: &MacServeState) -> serde_json::Value {
    let cache_ready = state.cache_status["ready"].as_bool().unwrap_or(false);
    let backend_ready = true;
    let tokenizer_ready = true;
    let receipt_ready = mac_serve_receipt_dir_ready(&state.receipt_dir);
    let ready = cache_ready && backend_ready && tokenizer_ready && receipt_ready;
    serde_json::json!({
        "artifact_kind": "bitnet_apple_m4_local_server_ready",
        "status": if ready { "ready" } else { "not_ready" },
        "ready": ready,
        "server": mac_serve_server_json(state),
        "model": {
            "id": &state.model.id,
            "display_name": &state.model.display_name,
            "path": &state.model.path,
            "sha256": &state.model.sha256,
            "bytes": state.model.bytes,
            "architecture": &state.model.architecture,
            "quantization": &state.model.quantization,
            "sha256_source": "verified_cache_metadata_and_startup_check",
        },
        "tokenizer": {
            "model": &state.model.tokenizer_model,
            "pretokenizer_authority": &state.model.tokenizer_pre,
            "chat_template": state.model.chat_template,
            "prompt_template": QWEN_PROMPT_TEMPLATE,
        },
        "backend": {
            "requested_backend": APPLE_M4_CPU_NEON,
            "selected_backend": APPLE_M4_CPU_NEON,
            "runtime_api": "cpu",
            "fallback_used": false,
        },
        "checks": {
            "cache": &state.cache_status,
            "disk": &state.disk,
            "tokenizer": {
                "checked": true,
                "ready": tokenizer_ready,
                "authority": &state.model.tokenizer_pre,
            },
            "backend": {
                "checked": true,
                "ready": backend_ready,
                "unsupported_full_metal_rejected": true,
            },
            "receipts": {
                "checked": true,
                "ready": receipt_ready,
                "dir": &state.receipt_dir,
                "mode": "per_request_http_export",
                "endpoint": "/receipts/{id}",
            },
            "generation": {
                "checked": false,
                "executed": false,
                "reason": "health and ready endpoints do not run generation",
            },
        },
        "claim_boundary": mac_serve_claim_boundary_json(),
    })
}

fn mac_serve_server_json(state: &MacServeState) -> serde_json::Value {
    serde_json::json!({
        "host": &state.host,
        "port": state.port,
        "started_at": &state.started_at_utc,
        "streaming_default": state.stream,
        "receipt_dir": &state.receipt_dir,
    })
}

fn mac_serve_claim_boundary_json() -> serde_json::Value {
    serde_json::json!({
        "dense_slm_local_server_health_ready": true,
        "generation_endpoint_implemented": true,
        "streaming_completions_work": true,
        "receipt_export_endpoint_implemented": true,
        "openai_compatibility_claimed": false,
        "production_readiness_claimed": false,
        "bitnet_quality_claimed": false,
        "full_metal_inference_claimed": false,
        "mpsgraph_inference_claimed": false,
        "neural_engine_execution_claimed": false,
        "qk256_apple_claimed": false,
        "broad_performance_claim": false,
    })
}

fn ensure_mac_serve_receipt_dir_ready(receipt_dir: &Path) -> Result<()> {
    if !receipt_dir.is_dir() {
        anyhow::bail!("mac serve receipt path is not a directory: {}", receipt_dir.display());
    }
    mac_serve_write_receipt_probe(receipt_dir).with_context(|| {
        format!("mac serve receipt directory is not writable: {}", receipt_dir.display())
    })
}

fn mac_serve_receipt_dir_ready(receipt_dir: &Path) -> bool {
    receipt_dir.is_dir() && mac_serve_write_receipt_probe(receipt_dir).is_ok()
}

fn mac_serve_write_receipt_probe(receipt_dir: &Path) -> Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    for attempt in 0..8 {
        let probe = receipt_dir.join(format!(
            ".bitnet-mac-serve-receipt-probe-{}-{now}-{attempt}",
            std::process::id()
        ));
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&probe) {
            Ok(_) => {
                std::fs::remove_file(&probe).with_context(|| {
                    format!("failed to remove receipt probe {}", probe.display())
                })?;
                return Ok(());
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to create receipt probe {}", probe.display())
                });
            }
        }
    }
    anyhow::bail!("failed to choose a unique receipt probe path in {}", receipt_dir.display())
}

fn mac_serve_host_is_loopback(host: &str) -> bool {
    let host = host.trim().trim_start_matches('[').trim_end_matches(']');
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>().map(|addr| addr.is_loopback()).unwrap_or(false)
}

fn mac_serve_request_id() -> String {
    let nanos = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default();
    format!("m4srv-{nanos}")
}

fn mac_doctor_base_receipt(
    json_out: &Path,
    result: &str,
    cache_status: serde_json::Value,
    bitnet_ask_readiness: serde_json::Value,
    unsupported_backend_rejected: bool,
    max_new_tokens: usize,
) -> serde_json::Value {
    let expected_bytes = cache_status["expected"]["bytes"].as_u64().unwrap_or_default();
    let cache_root = cache_status["cache_root"]
        .as_str()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": "apple_m4_slm_doctor",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "artifact_path": json_out.display().to_string(),
        "result": result,
        "requested_backend": APPLE_M4_CPU_NEON,
        "selected_backend": APPLE_M4_CPU_NEON,
        "runtime_api": "cpu",
        "fallback_used": false,
        "fallback_reason": serde_json::Value::Null,
        "prompt": MAC_SMOKE_PROMPT,
        "expected_text_fragment": MAC_SMOKE_EXPECTED_FRAGMENT,
        "checks": {
            "cache": {
                "checked": true,
                "ready": cache_status["ready"].clone(),
                "state": cache_status["state"].clone(),
                "cache_root": cache_status["cache_root"].clone(),
                "cache_path": cache_status["cache_path"].clone(),
                "metadata_path": cache_status["metadata_path"].clone(),
                "present": cache_status["present"].clone(),
                "size_matches": cache_status["size_matches"].clone(),
                "metadata_present": cache_status["metadata_present"].clone(),
                "verified": cache_status["verified"].clone(),
                "next_step": cache_status["next_step"].clone(),
            },
            "disk": disk_health_json(&cache_root, expected_bytes),
            "smoke": {
                "checked": result == "pass",
                "prompt": MAC_SMOKE_PROMPT,
                "max_new_tokens": max_new_tokens,
            },
            "receipt_validation": {
                "checked": result == "pass",
            },
            "backend": {
                "checked": true,
                "requested_backend": APPLE_M4_CPU_NEON,
                "selected_backend": APPLE_M4_CPU_NEON,
                "runtime_api": "cpu",
                "fallback_used": false,
            },
            "unsupported_backend": {
                "checked": true,
                "requested_backend": APPLE_M4_METAL,
                "rejected": unsupported_backend_rejected,
                "note": "mac doctor verifies that full apple-m4-metal inference remains blocked for the dense SLM wrapper",
            },
            "bitnet_ask": bitnet_ask_readiness,
        },
        "mac_claim_boundary": {
            "slm_local_answer": true,
            "doctor": true,
            "bitnet_one_shot_ask_readiness_checked": true,
            "bitnet_fixed_prompt_warm_readiness_checked": true,
            "requested_backend": APPLE_M4_CPU_NEON,
            "bitnet_quality_claimed": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false,
            "speedup_claim": false
        },
        "broad_performance_claim": false,
        "speedup_claim": false,
    })
}

fn bitnet_mac_ask_readiness_json(cache_dir: Option<PathBuf>) -> serde_json::Value {
    let catalog = match model_cache::apple_m4_models_catalog_json(cache_dir) {
        Ok(catalog) => catalog,
        Err(error) => {
            return serde_json::json!({
                "checked": true,
                "ready": false,
                "advisory": true,
                "blocks_doctor": false,
                "error": error.to_string(),
                "claim_boundary": bitnet_mac_ask_readiness_claim_boundary(),
            });
        }
    };
    let Some(row) = catalog["rows"]
        .as_array()
        .and_then(|rows| rows.iter().find(|row| row["id"].as_str() == Some(BITNET_M4_MODEL_ID)))
    else {
        return serde_json::json!({
            "checked": true,
            "ready": false,
            "advisory": true,
            "blocks_doctor": false,
            "model": {
                "id": BITNET_M4_MODEL_ID,
                "catalog_state": "missing",
            },
            "commands": {
                "models": bitnet_mac_models_command(catalog["cache_root"].as_str()),
            },
            "claim_boundary": bitnet_mac_ask_readiness_claim_boundary(),
        });
    };

    let cached_model_ready = row["cache_state"].as_str() == Some("ready");
    let tokenizer_path = PathBuf::from(BITNET_M4_DEFAULT_TOKENIZER_PATH);
    let (tokenizer_present, tokenizer_verified, tokenizer_sha256, tokenizer_error) =
        if tokenizer_path.is_file() {
            match verify_bitnet_m4_tokenizer(&tokenizer_path) {
                Ok(sha256) => (true, true, Some(sha256), None),
                Err(error) => (true, false, None, Some(error.to_string())),
            }
        } else {
            (false, false, None, None)
        };
    let bitnet_cached_route_ready = cached_model_ready && tokenizer_verified;

    serde_json::json!({
        "checked": true,
        "ready": bitnet_cached_route_ready,
        "advisory": true,
        "blocks_doctor": false,
        "model": {
            "id": BITNET_M4_MODEL_ID,
            "catalog_state": row["state"].clone(),
            "cache_state": row["cache_state"].clone(),
            "cache_path": row["cache_path"].clone(),
            "expected_sha256": BITNET_M4_EXPECTED_MODEL_SHA256,
            "cached_model_ready": cached_model_ready,
            "fetch_command": row["fetch_command"].clone(),
            "verify_command": row["verify_command"].clone(),
            "proof_status": row["proof_status"].clone(),
            "proof_command": row["proof_command"].clone(),
            "mac_ask_enabled": row["mac_ask_enabled"].clone(),
            "mac_bitnet_warm_enabled": row["mac_bitnet_warm_enabled"].clone(),
            "mac_chat_enabled": row["mac_chat_enabled"].clone(),
            "mac_serve_enabled": row["mac_serve_enabled"].clone(),
            "warm_command": row["warm_command"].clone(),
            "warm_receipt_path": row["warm_receipt_path"].clone(),
        },
        "tokenizer": {
            "path": tokenizer_path,
            "expected_sha256": BITNET_M4_EXPECTED_TOKENIZER_SHA256,
            "present": tokenizer_present,
            "verified": tokenizer_verified,
            "sha256": tokenizer_sha256,
            "error": tokenizer_error,
            "required_explicit_argument": "--tokenizer",
        },
        "commands": {
            "models": bitnet_mac_models_command(catalog["cache_root"].as_str()),
            "ask_cached_model": bitnet_cached_ask_command(catalog["cache_root"].as_str()),
            "warm_cached_model": bitnet_cached_warm_command(catalog["cache_root"].as_str()),
            "fetch": row["fetch_command"].clone(),
            "verify": row["verify_command"].clone(),
        },
        "claim_boundary": bitnet_mac_ask_readiness_claim_boundary(),
    })
}

fn bitnet_mac_models_command(cache_root: Option<&str>) -> String {
    match cache_root {
        Some(cache_root) => format!("bitnet mac models --cache-dir {cache_root}"),
        None => "bitnet mac models".to_string(),
    }
}

fn bitnet_cached_ask_command(cache_root: Option<&str>) -> String {
    let cache_arg =
        cache_root.map(|cache_root| format!(" --cache-dir {cache_root}")).unwrap_or_default();
    format!(
        "bitnet mac ask --model-id {BITNET_M4_MODEL_ID}{cache_arg} --tokenizer {BITNET_M4_DEFAULT_TOKENIZER_PATH} \"What is 2+2? Answer briefly.\""
    )
}

fn bitnet_cached_warm_command(cache_root: Option<&str>) -> String {
    let cache_arg =
        cache_root.map(|cache_root| format!(" --cache-dir {cache_root}")).unwrap_or_default();
    format!(
        "bitnet mac bitnet-warm --model-id {BITNET_M4_MODEL_ID}{cache_arg} --tokenizer {BITNET_M4_DEFAULT_TOKENIZER_PATH}"
    )
}

fn bitnet_mac_ask_readiness_claim_boundary() -> serde_json::Value {
    serde_json::json!({
        "bitnet_one_shot_mac_ask": true,
        "bitnet_fixed_prompt_warm_session": true,
        "readiness_only": true,
        "chat_enabled": false,
        "serve_enabled": false,
        "full_metal_inference_claimed": false,
        "mpsgraph_inference_claimed": false,
        "neural_engine_execution_claimed": false,
        "qk256_apple_claimed": false,
        "broad_performance_claim": false,
        "speedup_claim": false,
        "bitnet_quality_claimed": false,
    })
}

fn verified_cache_status_json(model: &VerifiedCachedModel) -> serde_json::Value {
    let metadata_path = model
        .path
        .parent()
        .map(|parent| parent.join("bitnet-model-cache.json"))
        .unwrap_or_else(|| PathBuf::from("bitnet-model-cache.json"));
    serde_json::json!({
        "state": "ready",
        "ready": true,
        "cache_root": model.cache_root.clone(),
        "cache_path": model.path.clone(),
        "metadata_path": metadata_path,
        "present": true,
        "size_matches": true,
        "metadata_present": true,
        "verified": true,
        "verification_passes": 1,
        "note": "mac smoke performs one strict model-cache verification pass before generation and records the resulting ready state",
    })
}

#[allow(clippy::too_many_arguments)]
async fn run_chat_session(
    model_id: &str,
    cache_dir: Option<PathBuf>,
    prompts: Vec<String>,
    system_prompt: Option<String>,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
    threads: usize,
    stream: bool,
    progress: bool,
    quiet: bool,
    turn_receipts: bool,
    interactive_prompt_collection: bool,
    allocation_audit: bool,
    metal_prefill_qkv_phase: bool,
    json_out: PathBuf,
) -> Result<()> {
    let model = model_cache::verified_apple_m4_slm_model(model_id, cache_dir)?;
    if progress && !quiet {
        eprintln!(
            "mac chat: loading verified model/tokenizer once for {} prompt(s); streaming={}, turn_receipts={}",
            prompts.len(),
            stream,
            turn_receipts
        );
    }
    crate::run_slm_warm_session(
        APPLE_M4_CPU_NEON,
        model.path.clone(),
        "auto".to_string(),
        None,
        None,
        1,
        prompts,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        seed,
        true,
        true,
        true,
        true,
        threads,
        QWEN_PROMPT_TEMPLATE.to_string(),
        false,
        system_prompt,
        vec!["<|im_end|>".to_string()],
        Vec::new(),
        true,
        false,
        allocation_audit,
        crate::SlmWarmSessionOutput::new(stream, progress, quiet)
            .with_prompt_receipts(turn_receipts)
            .with_interactive_prompt_collection(interactive_prompt_collection)
            .with_model_sha256_override(Some(model.sha256.clone()))
            .with_metal_prefill_qkv_phase(metal_prefill_qkv_phase),
        1,
        1,
        json_out.clone(),
    )
    .await?;
    let summary = annotate_and_validate_mac_receipt_silent(&json_out, &model, "mac chat")?;
    if progress && !quiet {
        eprintln!(
            "mac chat: aggregate receipt checked: {} ({}, generated_tokens={:?}, model/tokenizer loaded once)",
            json_out.display(),
            summary.artifact_kind,
            summary.generated_tokens
        );
    }
    Ok(())
}

struct MacValidateRun<'a> {
    model_id: &'a str,
    cache_dir: Option<PathBuf>,
    requested_backend: &'static str,
    corpus: PathBuf,
    corpus_repeat_runs: usize,
    profile_set: MacValidateProfileSet,
    max_new_tokens: usize,
    threads: usize,
    allocation_audit: bool,
    metal_prefill_qkv_phase: bool,
    progress: bool,
    quiet: bool,
    json_out: PathBuf,
}

async fn run_validate(request: MacValidateRun<'_>) -> Result<()> {
    let MacValidateRun {
        model_id,
        cache_dir,
        requested_backend,
        corpus,
        corpus_repeat_runs,
        profile_set,
        max_new_tokens,
        threads,
        allocation_audit,
        metal_prefill_qkv_phase,
        progress,
        quiet,
        json_out,
    } = request;

    if profile_set == MacValidateProfileSet::Performance && cfg!(debug_assertions) {
        anyhow::bail!(
            "mac validate --profile-set performance must be run from a release build; use `cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- mac validate --profile-set performance ...`"
        );
    }
    let model = model_cache::verified_apple_m4_slm_model(model_id, cache_dir)?;
    if profile_set == MacValidateProfileSet::Operator {
        if metal_prefill_qkv_phase {
            anyhow::bail!(
                "mac validate --metal-prefill-qkv-phase is scoped to the smoke quality corpus; profile timing remains CPU/NEON until M4-METAL-007"
            );
        }
        return run_operator_profiles(
            model,
            requested_backend,
            json_out,
            threads,
            allocation_audit,
            progress,
            quiet,
        )
        .await;
    }
    if profile_set == MacValidateProfileSet::Performance {
        if metal_prefill_qkv_phase {
            anyhow::bail!(
                "mac validate --metal-prefill-qkv-phase is scoped to the smoke quality corpus; profile timing remains CPU/NEON until M4-METAL-007"
            );
        }
        return run_performance_profiles(
            model,
            requested_backend,
            json_out,
            threads,
            allocation_audit,
            progress,
            quiet,
        )
        .await;
    }
    crate::run_slm_warm_session(
        requested_backend,
        model.path.clone(),
        "auto".to_string(),
        None,
        Some(corpus),
        corpus_repeat_runs,
        Vec::new(),
        max_new_tokens,
        0.0,
        1,
        1.0,
        1.1,
        None,
        true,
        true,
        true,
        true,
        threads,
        QWEN_PROMPT_TEMPLATE.to_string(),
        false,
        None,
        vec!["<|im_end|>".to_string()],
        Vec::new(),
        true,
        true,
        allocation_audit,
        crate::SlmWarmSessionOutput::new(false, progress, quiet)
            .with_model_sha256_override(Some(model.sha256.clone()))
            .with_metal_prefill_qkv_phase(metal_prefill_qkv_phase),
        1,
        1,
        json_out.clone(),
    )
    .await?;
    annotate_and_validate_mac_receipt(&json_out, &model, "mac validate")?;
    Ok(())
}

async fn run_operator_profiles(
    model: VerifiedCachedModel,
    requested_backend: &'static str,
    json_out: PathBuf,
    threads: usize,
    allocation_audit: bool,
    progress: bool,
    quiet: bool,
) -> Result<()> {
    run_warm_profile_set(
        model,
        requested_backend,
        json_out,
        threads,
        WarmProfileSetSpec {
            name: "operator",
            artifact_kind: "apple_m4_slm_operator_profiles",
            tokens: OPERATOR_PROFILE_TOKENS,
            command: "mac validate --profile-set operator",
            required_release: false,
            allocation_audit,
            progress,
            quiet,
        },
    )
    .await
}

async fn run_performance_profiles(
    model: VerifiedCachedModel,
    requested_backend: &'static str,
    json_out: PathBuf,
    threads: usize,
    allocation_audit: bool,
    progress: bool,
    quiet: bool,
) -> Result<()> {
    if cfg!(debug_assertions) {
        anyhow::bail!(
            "mac validate --profile-set performance must be run from a release build; use `cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- mac validate --profile-set performance ...`"
        );
    }
    run_warm_profile_set(
        model,
        requested_backend,
        json_out,
        threads,
        WarmProfileSetSpec {
            name: "performance",
            artifact_kind: "apple_m4_slm_performance_profiles",
            tokens: PERFORMANCE_PROFILE_TOKENS,
            command: "mac validate --profile-set performance",
            required_release: true,
            allocation_audit,
            progress,
            quiet,
        },
    )
    .await
}

struct MacBenchmarkRun<'a> {
    model_id: &'a str,
    cache_dir: Option<PathBuf>,
    requested_backend: &'static str,
    profiles: Vec<MacBenchmarkProfile>,
    threads: usize,
    allocation_audit: bool,
    progress: bool,
    quiet: bool,
    json_out: PathBuf,
}

struct BenchmarkProfileSpec {
    profile: MacBenchmarkProfile,
    max_new_tokens: usize,
    prompts: Vec<String>,
    target_context_tokens: Option<usize>,
    scenario: &'static str,
}

#[derive(Default)]
struct BenchmarkMetricSamples {
    cold_load_ms: Vec<f64>,
    tokenizer_load_ms: Vec<f64>,
    prompt_tokenize_ms: Vec<f64>,
    prefill_ms: Vec<f64>,
    time_to_first_token_ms: Vec<f64>,
    sampling_ms_per_token: Vec<f64>,
    input_tokens_per_second: Vec<f64>,
    output_tokens_per_second: Vec<f64>,
    decode_tokens_per_second: Vec<f64>,
    total_wall_ms: Vec<f64>,
    peak_memory_mb: Vec<f64>,
    memory_drift_mb: Vec<f64>,
}

impl BenchmarkMetricSamples {
    fn extend_from_profile(&mut self, profile: &serde_json::Value) {
        self.cold_load_ms.extend(benchmark_stat_samples(&profile["timing"]["cold_load_ms"]));
        self.tokenizer_load_ms
            .extend(benchmark_stat_samples(&profile["timing"]["tokenizer_load_ms"]));
        self.prompt_tokenize_ms
            .extend(benchmark_stat_samples(&profile["timing"]["prompt_tokenize_ms"]));
        self.prefill_ms.extend(benchmark_stat_samples(&profile["timing"]["prefill_ms"]));
        self.time_to_first_token_ms
            .extend(benchmark_stat_samples(&profile["timing"]["time_to_first_token_ms"]));
        self.sampling_ms_per_token
            .extend(benchmark_stat_samples(&profile["timing"]["sampling_ms_per_token"]));
        self.input_tokens_per_second
            .extend(benchmark_stat_samples(&profile["throughput"]["input_tokens_per_second"]));
        self.output_tokens_per_second
            .extend(benchmark_stat_samples(&profile["throughput"]["output_tokens_per_second"]));
        self.decode_tokens_per_second
            .extend(benchmark_stat_samples(&profile["throughput"]["decode_tokens_per_second"]));
        self.total_wall_ms.extend(benchmark_stat_samples(&profile["timing"]["total_wall_ms"]));
        self.peak_memory_mb.extend(benchmark_stat_samples(&profile["memory"]["peak_memory_mb"]));
        self.memory_drift_mb.extend(benchmark_stat_samples(&profile["memory"]["memory_drift_mb"]));
    }

    fn speed_json(&self) -> serde_json::Value {
        benchmark_flat_metric_json(&[
            ("cold_load_ms", &self.cold_load_ms),
            ("tokenizer_load_ms", &self.tokenizer_load_ms),
            ("prompt_tokenize_ms", &self.prompt_tokenize_ms),
            ("prefill_ms", &self.prefill_ms),
            ("ttft_ms", &self.time_to_first_token_ms),
            ("sampling_ms_per_token", &self.sampling_ms_per_token),
            ("input_tok_s", &self.input_tokens_per_second),
            ("output_tok_s", &self.output_tokens_per_second),
            ("decode_tok_s", &self.decode_tokens_per_second),
            ("total_wall_ms", &self.total_wall_ms),
        ])
    }

    fn memory_json(&self) -> serde_json::Value {
        let mut memory = benchmark_flat_metric_json(&[
            ("peak_memory_mb", &self.peak_memory_mb),
            ("memory_drift_mb", &self.memory_drift_mb),
        ]);
        if let Some(object) = memory.as_object_mut() {
            object.insert(
                "source".to_string(),
                serde_json::json!("getrusage.ru_maxrss process peak delta"),
            );
        }
        memory
    }
}

async fn run_benchmark(request: MacBenchmarkRun<'_>) -> Result<()> {
    if cfg!(debug_assertions) {
        anyhow::bail!(
            "mac benchmark must be run from a release build; use `cargo build --release --locked -p bitnet-cli --no-default-features --features cpu,full-cli --bin bitnet` and then `target/release/bitnet mac benchmark ...`"
        );
    }
    let MacBenchmarkRun {
        model_id,
        cache_dir,
        requested_backend,
        profiles,
        threads,
        allocation_audit,
        progress,
        quiet,
        json_out,
    } = request;
    let profiles = dedupe_benchmark_profiles(profiles)?;
    let model = model_cache::verified_apple_m4_slm_model(model_id, cache_dir)?;
    let receipt_dir =
        json_out.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from(".")).join(
            format!(
                "{}-profiles",
                json_out.file_stem().and_then(|stem| stem.to_str()).unwrap_or("mac-benchmark")
            ),
        );
    std::fs::create_dir_all(&receipt_dir)
        .with_context(|| format!("failed to create {}", receipt_dir.display()))?;

    let benchmark_start_peak_mb = peak_memory_mb();
    let mut summaries = Vec::with_capacity(profiles.len());
    let mut all_samples = BenchmarkMetricSamples::default();
    for profile in profiles {
        let spec = benchmark_profile_spec(profile);
        let profile_id = spec.profile.id();
        let receipt_path = receipt_dir.join(format!("{profile_id}.json"));
        if progress && !quiet {
            eprintln!("mac benchmark: running {profile_id}");
        }
        let profile_start_peak_mb = peak_memory_mb();
        crate::run_slm_warm_session(
            requested_backend,
            model.path.clone(),
            "auto".to_string(),
            None,
            None,
            1,
            spec.prompts.clone(),
            spec.max_new_tokens,
            0.0,
            1,
            1.0,
            1.1,
            None,
            true,
            true,
            true,
            true,
            threads,
            QWEN_PROMPT_TEMPLATE.to_string(),
            false,
            None,
            vec!["<|im_end|>".to_string()],
            Vec::new(),
            true,
            false,
            allocation_audit,
            crate::SlmWarmSessionOutput::new(false, progress, quiet)
                .with_model_sha256_override(Some(model.sha256.clone())),
            1,
            1,
            receipt_path.clone(),
        )
        .await?;
        annotate_and_validate_mac_receipt(&receipt_path, &model, "mac benchmark")?;
        let receipt = read_json_receipt(&receipt_path)?;
        let summary =
            benchmark_profile_summary(&spec, &receipt_path, &receipt, profile_start_peak_mb)?;
        all_samples.extend_from_profile(&summary);
        summaries.push(summary);
    }

    let profile_ids =
        summaries.iter().filter_map(|profile| profile["profile_id"].as_str()).collect::<Vec<_>>();
    let profile_ids_display = profile_ids.join(", ");
    let generated_tokens_total = summaries
        .iter()
        .map(|profile| profile["generated_tokens"].as_u64().unwrap_or_default())
        .sum::<u64>();
    let prompt_count_total = summaries
        .iter()
        .map(|profile| profile["prompt_count"].as_u64().unwrap_or_default())
        .sum::<u64>();
    let memory_end_peak_mb = peak_memory_mb();
    let mut memory = all_samples.memory_json();
    if let Some(object) = memory.as_object_mut() {
        object
            .insert("start_peak_memory_mb".to_string(), optional_f64_json(benchmark_start_peak_mb));
        object.insert("end_peak_memory_mb".to_string(), optional_f64_json(memory_end_peak_mb));
        object.insert(
            "process_peak_drift_mb".to_string(),
            optional_f64_json(memory_delta_mb(benchmark_start_peak_mb, memory_end_peak_mb)),
        );
    }

    let aggregate = serde_json::json!({
        "schema_version": "1.1.0",
        "artifact_kind": "apple_m4_slm_benchmark_v2",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "artifact_path": json_out.display().to_string(),
        "requested_backend": requested_backend,
        "selected_backend": requested_backend,
        "runtime_api": "cpu",
        "fallback_used": false,
        "fallback_reason": serde_json::Value::Null,
        "profile_set": "slm-benchmark-v2",
        "profiles_required": profile_ids,
        "profiles": summaries,
        "prompt_count": prompt_count_total,
        "generated_tokens": generated_tokens_total,
        "speed": all_samples.speed_json(),
        "memory": memory,
        "build": {
            "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "release_mode": !cfg!(debug_assertions),
        },
        "model_cache": {
            "id": model.id,
            "display_name": model.display_name,
            "cache_root": model.cache_root,
            "path": model.path,
            "sha256": model.sha256,
            "bytes": model.bytes,
            "architecture": model.architecture,
            "quantization": model.quantization,
            "tokenizer_model": model.tokenizer_model,
            "tokenizer_pre": model.tokenizer_pre,
            "chat_template": model.chat_template,
            "support_note": model.support_note,
        },
        "benchmark_contract": {
            "contract_version": "1.1.0",
            "scope": "Apple M4 Mac mini dense SLM benchmark v2",
            "profile_execution_model": "one resident warm-session run per named profile",
            "supported_profiles": M4_SLM_BENCHMARK_V2_PROFILES,
            "required_metrics": {
                "timing": M4_SLM_BENCHMARK_V2_TIMING_METRICS,
                "throughput": M4_SLM_BENCHMARK_V2_THROUGHPUT_METRICS,
                "memory": M4_SLM_BENCHMARK_V2_MEMORY_METRICS,
                "aggregate_speed": M4_SLM_BENCHMARK_V2_AGGREGATE_SPEED_METRICS,
            },
            "profiles_loaded_independently": true,
            "profile_set_model_loads": summaries.len(),
            "cold_load_separated": true,
            "model_tokenizer_reuse_visible_per_profile": true,
            "percentiles": ["p50", "p90", "p99"],
            "input_tok_s_definition": "prompt_tokens / (prompt_tokenize_ms + prefill_ms)",
            "output_tok_s_definition": "generated_tokens / total_prompt_wall_ms",
            "decode_tok_s_definition": "generated_tokens / decode_total_ms",
            "memory_drift_definition": "per-profile ru_maxrss process peak delta; monotonic peak, not live RSS",
            "thresholds_are_claim_bounds_not_speed_guarantees": true
        },
        "evidence": {
            "profile_receipts": summaries
                .iter()
                .filter_map(|profile| profile["receipt_path"].as_str())
                .collect::<Vec<_>>(),
            "generated_text_recorded": true,
            "generated_token_ids_recorded": true,
            "operator_command": "mac benchmark"
        },
        "mac_claim_boundary": {
            "dense_slm_only": true,
            "timing_profile": true,
            "bounded_benchmark_profiles_only": true,
            "broad_model_quality_claim": false,
            "broad_performance_claim": false,
            "speedup_claim": false,
            "bitnet_quality_claimed": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "macbook_evidence": false
        },
        "speedup_claim": false,
    });
    std::fs::write(&json_out, serde_json::to_vec_pretty(&aggregate)?)
        .with_context(|| format!("failed to write {}", json_out.display()))?;
    validate_mac_receipt_value(&json_out, &aggregate)?;
    println!(
        "Mac benchmark v2 summary written to {} (profiles: {})",
        json_out.display(),
        profile_ids_display
    );
    Ok(())
}

fn dedupe_benchmark_profiles(
    profiles: Vec<MacBenchmarkProfile>,
) -> Result<Vec<MacBenchmarkProfile>> {
    if profiles.is_empty() {
        anyhow::bail!("mac benchmark requires at least one --profile");
    }
    let mut deduped = Vec::with_capacity(profiles.len());
    for profile in profiles {
        if !deduped.contains(&profile) {
            deduped.push(profile);
        }
    }
    Ok(deduped)
}

fn benchmark_profile_spec(profile: MacBenchmarkProfile) -> BenchmarkProfileSpec {
    match profile {
        MacBenchmarkProfile::ShortPrompt16Out => BenchmarkProfileSpec {
            profile,
            max_new_tokens: 16,
            prompts: short_benchmark_prompts(),
            target_context_tokens: None,
            scenario: "short_prompt",
        },
        MacBenchmarkProfile::ShortPrompt64Out => BenchmarkProfileSpec {
            profile,
            max_new_tokens: 64,
            prompts: short_benchmark_prompts(),
            target_context_tokens: None,
            scenario: "short_prompt",
        },
        MacBenchmarkProfile::LongPrompt16Out => BenchmarkProfileSpec {
            profile,
            max_new_tokens: 16,
            prompts: long_benchmark_prompts(),
            target_context_tokens: None,
            scenario: "long_prompt",
        },
        MacBenchmarkProfile::LongPrompt128Out => BenchmarkProfileSpec {
            profile,
            max_new_tokens: 128,
            prompts: long_benchmark_prompts(),
            target_context_tokens: None,
            scenario: "long_prompt",
        },
        MacBenchmarkProfile::Context1k => BenchmarkProfileSpec {
            profile,
            max_new_tokens: 16,
            prompts: context_benchmark_prompts(1_000, 3),
            target_context_tokens: Some(1_000),
            scenario: "synthetic_context",
        },
        MacBenchmarkProfile::Context4k => BenchmarkProfileSpec {
            profile,
            max_new_tokens: 16,
            prompts: context_benchmark_prompts(4_000, 3),
            target_context_tokens: Some(4_000),
            scenario: "synthetic_context",
        },
        MacBenchmarkProfile::Resident25 => BenchmarkProfileSpec {
            profile,
            max_new_tokens: 16,
            prompts: resident_benchmark_prompts(25),
            target_context_tokens: None,
            scenario: "resident_session",
        },
        MacBenchmarkProfile::Resident50 => BenchmarkProfileSpec {
            profile,
            max_new_tokens: 16,
            prompts: resident_benchmark_prompts(50),
            target_context_tokens: None,
            scenario: "resident_session",
        },
        MacBenchmarkProfile::Resident100 => BenchmarkProfileSpec {
            profile,
            max_new_tokens: 16,
            prompts: resident_benchmark_prompts(100),
            target_context_tokens: None,
            scenario: "resident_session",
        },
    }
}

fn short_benchmark_prompts() -> Vec<String> {
    [
        "What is 7+5? Answer with the number only.",
        "Name the capital of France. Answer with one word.",
        "Write one short sentence about Rust.",
        "Classify this sentiment as positive or negative: the tool is reliable.",
        "Rewrite in five words or fewer: local inference keeps private prompts on the machine.",
    ]
    .iter()
    .map(|prompt| (*prompt).to_string())
    .collect()
}

fn long_benchmark_prompts() -> Vec<String> {
    [
        "A field operator is preparing an offline laptop for a week of local document triage. The machine has limited disk space, stable power, and no network access after departure. The operator needs short deterministic answers, clear failure messages, and receipts that identify the exact model and tokenizer. Summarize the operational priority in one sentence.",
        "A finance team is comparing two internal notes. The first note says revenue is recognized when the service is delivered. The second note says cash collection may happen later and should not by itself decide revenue timing. Explain the accounting idea in plain language.",
        "A developer is checking whether a local model runner should make broad platform claims from one receipt. The runner recorded a model hash, tokenizer source, backend, fallback flag, token IDs, timing, and memory. State the safe claim boundary.",
        "A support engineer found that a prompt sometimes returns fenced JSON and sometimes plain JSON. The scoring harness requires a strict schema and records both raw generated text and normalized text. Explain what should be tracked before changing the scoring rule.",
        "A small local service exposes health, readiness, and completion routes. It is intended for local apps on loopback, not production hosting. Describe the most important receipt fields for each completion request.",
    ]
    .iter()
    .map(|prompt| (*prompt).to_string())
    .collect()
}

fn context_benchmark_prompts(target_context_tokens: usize, count: usize) -> Vec<String> {
    (0..count).map(|variant| synthetic_context_prompt(target_context_tokens, variant)).collect()
}

fn synthetic_context_prompt(target_context_tokens: usize, variant: usize) -> String {
    // The synthetic log has numeric IDs and repeated labels that tokenize denser
    // than prose; this keeps the benchmark near the named 1k/4k context scale.
    let target_words = target_context_tokens.saturating_mul(7) / 12;
    let mut prompt = String::from(
        "Use the synthetic operations log below. Answer the final question briefly.\n\n",
    );
    let mut words = prompt.split_whitespace().count();
    let mut index = 0usize;
    while words < target_words {
        let line = format!(
            "Record {index:04}: team alpha handled queue {}, team beta verified checksum {}, and ticket M4-{} stayed inside the local CPU receipt boundary.\n",
            (index + variant) % 17,
            (index * 13 + variant) % 97,
            (index + variant) % 31
        );
        words += line.split_whitespace().count();
        prompt.push_str(&line);
        index += 1;
    }
    prompt.push_str("\nQuestion: which boundary should the operator preserve? Answer briefly.");
    prompt
}

fn resident_benchmark_prompts(count: usize) -> Vec<String> {
    let base = [
        "Answer with a single digit: 2+2=",
        "Name the capital of France. Answer with one word.",
        "Write one short sentence about local inference.",
        "Classify as yes or no: receipts should record fallback status.",
        "Give two words that describe deterministic scoring.",
    ];
    (0..count).map(|index| format!("{} [turn {}]", base[index % base.len()], index + 1)).collect()
}

fn benchmark_profile_summary(
    spec: &BenchmarkProfileSpec,
    path: &Path,
    receipt: &serde_json::Value,
    profile_start_peak_mb: Option<f64>,
) -> Result<serde_json::Value> {
    let profile_id = spec.profile.id();
    let prompt_count = receipt["session"]["prompt_count"].as_u64().unwrap_or_default();
    let quality_passed = receipt["quality_summary"]["passed"].as_bool().unwrap_or(false);
    if prompt_count == 0 || !quality_passed {
        anyhow::bail!(
            "benchmark profile {profile_id} did not produce a passing warm-session receipt"
        );
    }
    let prompt_receipts =
        receipt["session"]["per_prompt_receipts"].as_array().ok_or_else(|| {
            anyhow!("benchmark profile {profile_id} is missing per-prompt receipt paths")
        })?;
    if prompt_receipts.len() != prompt_count as usize {
        anyhow::bail!("benchmark profile {profile_id} prompt receipt count mismatch");
    }

    let mut prompt_tokens = Vec::with_capacity(prompt_receipts.len());
    let mut generated_tokens = Vec::with_capacity(prompt_receipts.len());
    let mut tokenization_ms = Vec::with_capacity(prompt_receipts.len());
    let mut prefill_ms = Vec::with_capacity(prompt_receipts.len());
    let mut ttft_ms = Vec::with_capacity(prompt_receipts.len());
    let mut decode_total_ms = Vec::with_capacity(prompt_receipts.len());
    let mut sampling_ms_per_token = Vec::with_capacity(prompt_receipts.len());
    let mut total_wall_ms = Vec::with_capacity(prompt_receipts.len());
    let mut input_tok_s = Vec::with_capacity(prompt_receipts.len());
    let mut output_tok_s = Vec::with_capacity(prompt_receipts.len());
    let mut decode_tok_s = Vec::with_capacity(prompt_receipts.len());
    let mut generated_total = 0u64;
    for prompt_receipt in prompt_receipts {
        let prompt_path = PathBuf::from(prompt_receipt.as_str().ok_or_else(|| {
            anyhow!("benchmark profile {profile_id} has non-string prompt receipt path")
        })?);
        let prompt = read_json_receipt(&prompt_path)?;
        let prompt_token_count = required_json_f64(&prompt["tokens"]["prompt"], "tokens.prompt")?;
        let generated_token_count =
            required_json_f64(&prompt["tokens"]["generated"], "tokens.generated")?;
        let tokenize = required_json_f64(&prompt["timing"]["tokenize_ms"], "timing.tokenize_ms")?;
        let prefill = required_json_f64(&prompt["timing"]["prefill_ms"], "timing.prefill_ms")?;
        let ttft = required_json_f64(
            &prompt["timing"]["time_to_first_token_ms"],
            "timing.time_to_first_token_ms",
        )?;
        let decode =
            required_json_f64(&prompt["timing"]["decode_total_ms"], "timing.decode_total_ms")?;
        let total = required_json_f64(&prompt["timing"]["total_ms"], "timing.total_ms")?;
        prompt_tokens.push(prompt_token_count);
        generated_tokens.push(generated_token_count);
        tokenization_ms.push(tokenize);
        prefill_ms.push(prefill);
        ttft_ms.push(ttft);
        decode_total_ms.push(decode);
        if let Some(sample_ms) = prompt["timing"]["sampling_ms_per_token"].as_f64() {
            sampling_ms_per_token.push(sample_ms);
        }
        total_wall_ms.push(total);
        push_positive_rate(&mut input_tok_s, prompt_token_count, tokenize + prefill);
        push_positive_rate(&mut output_tok_s, generated_token_count, total);
        push_positive_rate(&mut decode_tok_s, generated_token_count, decode);
        generated_total += generated_token_count as u64;
    }

    let profile_end_peak_mb = peak_memory_mb();
    let memory_drift_mb = memory_delta_mb(profile_start_peak_mb, profile_end_peak_mb);
    Ok(serde_json::json!({
        "profile_id": profile_id,
        "receipt_path": path.display().to_string(),
        "scenario": spec.scenario,
        "requested_max_new_tokens": spec.max_new_tokens,
        "prompt_count": prompt_count,
        "target_context_tokens": spec.target_context_tokens,
        "generated_tokens": generated_total,
        "quality_passed": quality_passed,
        "model_loaded_once": receipt["session"]["model_loaded_once"].as_bool().unwrap_or(false),
        "tokenizer_loaded_once": receipt["session"]["tokenizer_loaded_once"].as_bool().unwrap_or(false),
        "reuse_scope": receipt["session"]["reuse_scope"].clone(),
        "prompt_tokens": benchmark_stat_json(&prompt_tokens),
        "output_tokens": benchmark_stat_json(&generated_tokens),
        "timing": {
            "cold_load_ms": benchmark_stat_json(&[required_json_f64(&receipt["timing"]["model_load_ms"], "timing.model_load_ms")?]),
            "tokenizer_load_ms": benchmark_stat_json(&[required_json_f64(&receipt["timing"]["tokenizer_load_ms"], "timing.tokenizer_load_ms")?]),
            "prompt_tokenize_ms": benchmark_stat_json(&tokenization_ms),
            "prefill_ms": benchmark_stat_json(&prefill_ms),
            "time_to_first_token_ms": benchmark_stat_json(&ttft_ms),
            "decode_total_ms": benchmark_stat_json(&decode_total_ms),
            "sampling_ms_per_token": benchmark_stat_json(&sampling_ms_per_token),
            "total_wall_ms": benchmark_stat_json(&total_wall_ms),
        },
        "throughput": {
            "input_tokens_per_second": benchmark_stat_json(&input_tok_s),
            "output_tokens_per_second": benchmark_stat_json(&output_tok_s),
            "decode_tokens_per_second": benchmark_stat_json(&decode_tok_s),
        },
        "memory": {
            "peak_memory_mb": benchmark_stat_json(&optional_sample(profile_end_peak_mb)),
            "memory_drift_mb": benchmark_stat_json(&optional_sample(memory_drift_mb)),
            "source": "getrusage.ru_maxrss process peak delta",
        },
        "claim_boundary": {
            "scope": "this profile, model, backend, and machine receipt only",
            "broad_performance_claim": false,
            "speedup_claim": false,
        },
        "allocation_audit": receipt["allocation_audit"].clone(),
    }))
}

fn push_positive_rate(out: &mut Vec<f64>, numerator: f64, denominator_ms: f64) {
    if numerator > 0.0 && denominator_ms > 0.0 {
        out.push(round3(numerator * 1000.0 / denominator_ms));
    }
}

fn required_json_f64(value: &serde_json::Value, label: &str) -> Result<f64> {
    let Some(number) = value.as_f64() else {
        anyhow::bail!("benchmark receipt is missing numeric {label}");
    };
    Ok(number)
}

fn optional_sample(value: Option<f64>) -> Vec<f64> {
    value.into_iter().collect()
}

fn optional_f64_json(value: Option<f64>) -> serde_json::Value {
    value.map(serde_json::Value::from).unwrap_or(serde_json::Value::Null)
}

fn memory_delta_mb(start: Option<f64>, end: Option<f64>) -> Option<f64> {
    Some(round3((end? - start?).max(0.0)))
}

fn benchmark_flat_metric_json(metrics: &[(&str, &[f64])]) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    for (name, samples) in metrics {
        let stats = benchmark_stats(samples);
        object.insert(format!("{name}_p50"), optional_f64_json(stats.p50));
        object.insert(format!("{name}_p90"), optional_f64_json(stats.p90));
        object.insert(format!("{name}_p99"), optional_f64_json(stats.p99));
    }
    serde_json::Value::Object(object)
}

fn benchmark_stat_json(samples: &[f64]) -> serde_json::Value {
    let stats = benchmark_stats(samples);
    serde_json::json!({
        "count": stats.count,
        "p50": optional_f64_json(stats.p50),
        "p90": optional_f64_json(stats.p90),
        "p99": optional_f64_json(stats.p99),
        "min": optional_f64_json(stats.min),
        "max": optional_f64_json(stats.max),
        "samples": samples.iter().map(|sample| round3(*sample)).collect::<Vec<_>>(),
    })
}

fn benchmark_stat_samples(value: &serde_json::Value) -> Vec<f64> {
    value["samples"]
        .as_array()
        .map(|samples| samples.iter().filter_map(|sample| sample.as_f64()).collect())
        .unwrap_or_default()
}

struct BenchmarkStats {
    count: usize,
    p50: Option<f64>,
    p90: Option<f64>,
    p99: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
}

fn benchmark_stats(samples: &[f64]) -> BenchmarkStats {
    let mut sorted = samples.iter().copied().filter(|value| value.is_finite()).collect::<Vec<_>>();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.is_empty() {
        return BenchmarkStats { count: 0, p50: None, p90: None, p99: None, min: None, max: None };
    }
    BenchmarkStats {
        count: sorted.len(),
        p50: Some(round3(percentile_nearest_rank(&sorted, 50.0))),
        p90: Some(round3(percentile_nearest_rank(&sorted, 90.0))),
        p99: Some(round3(percentile_nearest_rank(&sorted, 99.0))),
        min: sorted.first().map(|value| round3(*value)),
        max: sorted.last().map(|value| round3(*value)),
    }
}

fn percentile_nearest_rank(sorted: &[f64], percentile: f64) -> f64 {
    let rank = ((percentile / 100.0) * sorted.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted.len().saturating_sub(1));
    sorted[index]
}

struct WarmProfileSetSpec {
    name: &'static str,
    artifact_kind: &'static str,
    tokens: &'static [usize],
    command: &'static str,
    required_release: bool,
    allocation_audit: bool,
    progress: bool,
    quiet: bool,
}

async fn run_warm_profile_set(
    model: VerifiedCachedModel,
    requested_backend: &'static str,
    json_out: PathBuf,
    threads: usize,
    spec: WarmProfileSetSpec,
) -> Result<()> {
    let receipt_dir =
        json_out.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from(".")).join(
            format!(
                "{}-profiles",
                json_out.file_stem().and_then(|stem| stem.to_str()).unwrap_or("mac-validate")
            ),
        );
    std::fs::create_dir_all(&receipt_dir)
        .with_context(|| format!("failed to create {}", receipt_dir.display()))?;

    let mut summaries = Vec::with_capacity(spec.tokens.len());
    for tokens in spec.tokens {
        let profile_id = format!("warm_{tokens}");
        let receipt_path = receipt_dir.join(format!("{profile_id}.json"));
        crate::run_slm_warm_session(
            requested_backend,
            model.path.clone(),
            "auto".to_string(),
            None,
            None,
            1,
            OPERATOR_PROFILE_PROMPTS.iter().map(|prompt| (*prompt).to_string()).collect(),
            *tokens,
            0.0,
            1,
            1.0,
            1.1,
            None,
            true,
            true,
            true,
            true,
            threads,
            QWEN_PROMPT_TEMPLATE.to_string(),
            false,
            None,
            vec!["<|im_end|>".to_string()],
            Vec::new(),
            true,
            false,
            spec.allocation_audit,
            crate::SlmWarmSessionOutput::new(false, spec.progress, spec.quiet)
                .with_model_sha256_override(Some(model.sha256.clone())),
            1,
            1,
            receipt_path.clone(),
        )
        .await?;
        annotate_and_validate_mac_receipt(&receipt_path, &model, spec.command)?;
        let receipt: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&receipt_path)
                .with_context(|| format!("failed to read {}", receipt_path.display()))?,
        )?;
        summaries.push(operator_profile_summary(&profile_id, *tokens, &receipt_path, &receipt)?);
    }
    let profile_ids = profile_ids_json(spec.tokens);
    let profile_set_model_loads = spec.tokens.len();
    let build_profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    let release_mode = !cfg!(debug_assertions);

    let aggregate = serde_json::json!({
        "schema_version": "1.0.0",
        "artifact_kind": mac_profile_set_artifact_kind(requested_backend, spec.artifact_kind),
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "artifact_path": json_out.display().to_string(),
        "requested_backend": requested_backend,
        "selected_backend": requested_backend,
        "runtime_api": "cpu",
        "fallback_used": false,
        "fallback_reason": serde_json::Value::Null,
        "profile_set": spec.name,
        "profiles": summaries,
        "build": {
            "profile": build_profile,
            "release_mode": release_mode,
        },
        "operator_thresholds": {
            "scope": supported_apple_slm_scope(requested_backend),
            "profile_execution_model": "one warm-session run per token budget",
            "profiles_loaded_independently": true,
            "profile_set_model_loads": profile_set_model_loads,
            "profiles_required": profile_ids,
            "cold_load_separated": true,
            "model_tokenizer_reuse_visible": true,
            "model_tokenizer_reuse_visible_per_profile": true,
            "reuse_scope": "within_each_profile",
            "initial_targets": initial_targets_json(spec.tokens),
            "hard_latency_thresholds": serde_json::Value::Null,
            "thresholds_are_claim_bounds_not_speed_guarantees": true
        },
        "performance_baseline": {
            "release_mode_required": spec.required_release,
            "release_mode_observed": release_mode,
            "warm_128_included": spec.tokens.contains(&128),
            "baseline_scope": "release-mode warm-session timing for this model, backend, machine, and profile set only",
            "cold_load_separated": true,
            "broad_performance_claim": false,
            "speedup_claim": false
        },
        "operator_ux": {
            "stream_tokens_requested": false,
            "progress_enabled": spec.progress && !spec.quiet,
            "quiet_default_logs": !spec.progress,
            "quiet_requested": spec.quiet,
            "status_stream": "stderr",
            "time_to_first_token_receipts": true,
            "clear_failure_messages": true
        },
        "model_cache": {
            "id": model.id,
            "display_name": model.display_name,
            "cache_root": model.cache_root,
            "path": model.path,
            "sha256": model.sha256,
            "bytes": model.bytes,
            "architecture": model.architecture,
            "quantization": model.quantization,
            "tokenizer_model": model.tokenizer_model,
            "tokenizer_pre": model.tokenizer_pre,
            "chat_template": model.chat_template,
            "support_note": model.support_note,
        },
        "mac_claim_boundary": {
            "slm_local_answer": true,
            "timing_profile": true,
            "bitnet_quality_claimed": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false,
            "speedup_claim": false
        },
        "allocation_audit": profile_set_allocation_audit_json(
            &summaries,
            spec.allocation_audit,
            requested_backend,
        ),
        "speedup_claim": false,
    });
    std::fs::write(&json_out, serde_json::to_vec_pretty(&aggregate)?)
        .with_context(|| format!("failed to write {}", json_out.display()))?;
    validate_mac_receipt_value(&json_out, &aggregate)?;
    println!(
        "Mac {} profile summary written to {} (profiles: {})",
        spec.name,
        json_out.display(),
        profile_ids_display(spec.tokens)
    );
    Ok(())
}

fn mac_profile_set_artifact_kind(
    requested_backend: &str,
    default_artifact_kind: &'static str,
) -> &'static str {
    match (requested_backend, default_artifact_kind) {
        (APPLE_M3_AIR_CPU_NEON, "apple_m4_slm_operator_profiles") => {
            "apple_m3_air_slm_operator_profiles"
        }
        (APPLE_M3_AIR_CPU_NEON, "apple_m4_slm_performance_profiles") => {
            "apple_m3_air_slm_performance_profiles"
        }
        _ => default_artifact_kind,
    }
}

fn supported_apple_slm_scope(requested_backend: &str) -> &'static str {
    match requested_backend {
        APPLE_M3_AIR_CPU_NEON => "supported Apple M3 Air SLM warm-answer timing only",
        _ => "supported Apple M4 SLM warm-answer timing only",
    }
}

fn operator_profile_summary(
    profile_id: &str,
    requested_max_new_tokens: usize,
    path: &Path,
    receipt: &serde_json::Value,
) -> Result<serde_json::Value> {
    let speed = &receipt["speed"];
    let generated_tokens = speed["counts"]["generated_tokens"]
        .as_u64()
        .or_else(|| {
            receipt["prompts"].as_array().map(|prompts| {
                prompts
                    .iter()
                    .map(|prompt| prompt["generated_tokens"].as_u64().unwrap_or_default())
                    .sum::<u64>()
            })
        })
        .unwrap_or_default();
    let prompt_count = receipt["session"]["prompt_count"].as_u64().unwrap_or_default();
    let quality_passed = receipt["quality_summary"]["passed"].as_bool().unwrap_or(false);
    let cold_load_separated = !receipt["timing"]["model_load_ms"].is_null()
        && !receipt["timing"]["tokenizer_load_ms"].is_null()
        && !receipt["speed"]["timing"]["warm_prompt_wall_ms"].is_null();
    if prompt_count == 0 || generated_tokens == 0 || !quality_passed {
        anyhow::bail!("operator profile {profile_id} did not produce a valid warm-session receipt");
    }
    Ok(serde_json::json!({
        "profile_id": profile_id,
        "receipt_path": path.display().to_string(),
        "requested_max_new_tokens": requested_max_new_tokens,
        "prompt_count": prompt_count,
        "generated_tokens": generated_tokens,
        "quality_passed": quality_passed,
        "cold_load_separated": cold_load_separated,
        "model_loaded_once": receipt["session"]["model_loaded_once"].as_bool().unwrap_or(false),
        "tokenizer_loaded_once": receipt["session"]["tokenizer_loaded_once"].as_bool().unwrap_or(false),
        "reuse_scope": "within_profile",
        "resident_session": {
            "reuse_scope": receipt["session"]["reuse_scope"].clone(),
            "session_owned_buffers": receipt["session"]["session_owned_buffers"].clone(),
            "prompt_token_buffer_reused": receipt["session"]["prompt_token_buffer_reused"].clone(),
            "generated_token_buffer_reused": receipt["session"]["generated_token_buffer_reused"].clone(),
            "timing_buffers_reused": receipt["session"]["timing_buffers_reused"].clone(),
            "allocation_audit_buffers_reused": receipt["session"]["allocation_audit_buffers_reused"].clone(),
            "stop_tail_buffer_reused": receipt["session"]["stop_tail_buffer_reused"].clone(),
            "kv_cache_reuse_policy": receipt["session"]["kv_cache_reuse_policy"].clone(),
            "sampler_reuse_policy": receipt["session"]["sampler_reuse_policy"].clone(),
            "logits_buffer_reuse_policy": receipt["session"]["logits_buffer_reuse_policy"].clone(),
        },
        "timing": {
            "model_load_ms": receipt["timing"]["model_load_ms"].clone(),
            "tokenizer_load_ms": receipt["timing"]["tokenizer_load_ms"].clone(),
            "total_session_ms": receipt["speed"]["timing"]["total_session_ms"].clone(),
            "tokenize_ms": receipt["speed"]["timing"]["tokenize_ms"].clone(),
            "prefill_ms": receipt["speed"]["timing"]["prefill_ms"].clone(),
            "warm_prompt_wall_ms": receipt["speed"]["timing"]["warm_prompt_wall_ms"].clone(),
            "first_token_ms": receipt["speed"]["timing"]["first_token_ms"].clone(),
            "time_to_first_token_ms": receipt["speed"]["timing"]["time_to_first_token_ms"].clone(),
            "decode_total_ms": receipt["speed"]["timing"]["decode_total_ms"].clone(),
            "sampling_ms": receipt["speed"]["timing"]["sampling_ms"].clone(),
            "warm_prompt_generated_tok_s": receipt["speed"]["throughput"]["warm_prompt_generated_tok_s"].clone(),
            "decode_generated_tok_s": receipt["speed"]["throughput"]["decode_generated_tok_s"].clone(),
        },
        "operator_ux": receipt["operator_ux"].clone(),
        "memory": {
            "peak_memory_mb": peak_memory_mb(),
            "peak_memory_source": "getrusage.ru_maxrss",
        },
        "claim_boundary": {
            "speedup_claim": false,
            "broad_performance_claim": false,
            "scope": "this profile, model, backend, and machine receipt only",
        },
        "allocation_audit": receipt["allocation_audit"].clone(),
    }))
}

fn profile_set_allocation_audit_json(
    summaries: &[serde_json::Value],
    enabled: bool,
    requested_backend: &str,
) -> serde_json::Value {
    if !enabled {
        return serde_json::json!({
            "enabled": false,
            "method": "not_requested",
            "scope": "not_requested",
        });
    }

    let mut totals = std::collections::BTreeMap::<String, (u64, u64)>::new();
    for summary in summaries {
        let Some(hotspots) = summary["allocation_audit"]["ranked_hotspots"].as_array() else {
            continue;
        };
        for hotspot in hotspots {
            let Some(component) = hotspot["component"].as_str() else {
                continue;
            };
            let entry = totals.entry(component.to_string()).or_default();
            entry.0 += hotspot["alloc_count"].as_u64().unwrap_or_default();
            entry.1 += hotspot["alloc_bytes"].as_u64().unwrap_or_default();
        }
    }
    let mut ranked = totals
        .into_iter()
        .map(|(component, (alloc_count, alloc_bytes))| {
            serde_json::json!({
                "component": component,
                "alloc_count": alloc_count,
                "alloc_bytes": alloc_bytes,
            })
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right["alloc_bytes"]
            .as_u64()
            .unwrap_or_default()
            .cmp(&left["alloc_bytes"].as_u64().unwrap_or_default())
            .then_with(|| {
                right["alloc_count"]
                    .as_u64()
                    .unwrap_or_default()
                    .cmp(&left["alloc_count"].as_u64().unwrap_or_default())
            })
            .then_with(|| {
                left["component"]
                    .as_str()
                    .unwrap_or_default()
                    .cmp(right["component"].as_str().unwrap_or_default())
            })
    });

    serde_json::json!({
        "enabled": true,
        "method": "process_global_allocator_counter_delta",
        "scope": profile_set_allocation_scope(requested_backend),
        "claim_scope": "aggregate of prompt-level allocation counter deltas; no optimization or performance improvement claimed",
        "profile_count": summaries.len(),
        "ranked_hotspots": ranked,
        "optimization_deferred": true,
    })
}

fn profile_set_allocation_scope(requested_backend: &str) -> &'static str {
    match requested_backend {
        APPLE_M3_AIR_CPU_NEON => "selected Apple M3 Air CPU/NEON SLM warm-session profile set",
        _ => "selected Apple M4 CPU/NEON SLM warm-session profile set",
    }
}

fn memory_receipt_json() -> serde_json::Value {
    serde_json::json!({
        "peak_memory_mb": peak_memory_mb(),
        "peak_memory_source": "getrusage.ru_maxrss",
    })
}

fn profile_ids_json(tokens: &[usize]) -> serde_json::Value {
    serde_json::Value::Array(
        tokens.iter().map(|tokens| serde_json::Value::String(format!("warm_{tokens}"))).collect(),
    )
}

fn profile_ids_display(tokens: &[usize]) -> String {
    tokens.iter().map(|tokens| format!("warm_{tokens}")).collect::<Vec<_>>().join(", ")
}

fn initial_targets_json(tokens: &[usize]) -> serde_json::Value {
    let mut targets = serde_json::Map::new();
    for tokens in tokens {
        let target = match *tokens {
            16 => "complete reliably",
            32 => "complete without timeout",
            64 => "measured and bounded",
            128 => "release-mode baseline only; no latency guarantee",
            _ => "measured and bounded",
        };
        targets.insert(format!("warm_{tokens}"), serde_json::Value::String(target.to_string()));
    }
    serde_json::Value::Object(targets)
}

fn read_json_receipt(path: &Path) -> Result<serde_json::Value> {
    serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("invalid JSON receipt {}", path.display()))
}

fn sibling_receipt_path(path: &Path, suffix: &str) -> PathBuf {
    let parent = path.parent().filter(|parent| !parent.as_os_str().is_empty());
    let stem = path.file_stem().and_then(|stem| stem.to_str()).unwrap_or("mac-smoke");
    let filename = format!("{stem}-{suffix}.json");
    parent.map(|parent| parent.join(&filename)).unwrap_or_else(|| PathBuf::from(filename))
}

fn disk_health_json(cache_root: &Path, expected_bytes: u64) -> serde_json::Value {
    let probe_path =
        cache_root.ancestors().find(|path| path.exists()).unwrap_or_else(|| Path::new("."));
    let available_bytes = available_bytes(probe_path);
    let recommended_bytes =
        expected_bytes.saturating_mul(2).saturating_add(LOW_DISK_HEADROOM_BYTES);
    serde_json::json!({
        "checked": true,
        "probe_path": probe_path,
        "available_bytes": available_bytes,
        "recommended_headroom_bytes": recommended_bytes,
        "low_disk": available_bytes.is_some_and(|available| available < recommended_bytes),
        "guidance": "run `bitnet model prune --all` or set BITNET_MODEL_CACHE_DIR / --cache-dir when low_disk=true"
    })
}

#[cfg(unix)]
fn available_bytes(path: &Path) -> Option<u64> {
    let output = Command::new("df").arg("-k").arg(path).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().nth(1)?;
    let available_kib = line.split_whitespace().nth(3)?.parse::<u64>().ok()?;
    Some(available_kib.saturating_mul(1024))
}

#[cfg(not(unix))]
fn available_bytes(_path: &Path) -> Option<u64> {
    None
}

#[cfg(unix)]
fn peak_memory_mb() -> Option<f64> {
    let mut usage = MaybeUninit::<libc::rusage>::uninit();
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    let usage = unsafe { usage.assume_init() };
    let raw = usage.ru_maxrss as f64;
    #[cfg(target_os = "macos")]
    let bytes = raw;
    #[cfg(not(target_os = "macos"))]
    let bytes = raw * 1024.0;
    Some(round3(bytes / (1024.0 * 1024.0)))
}

#[cfg(not(unix))]
fn peak_memory_mb() -> Option<f64> {
    None
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

fn annotate_and_validate_mac_receipt(
    path: &Path,
    model: &VerifiedCachedModel,
    operator_command: &str,
) -> Result<()> {
    let summary = annotate_and_validate_mac_receipt_silent(path, model, operator_command)?;
    println!(
        "Mac receipt checked: {} ({}, generated_tokens={:?})",
        path.display(),
        summary.artifact_kind,
        summary.generated_tokens
    );
    Ok(())
}

fn annotate_and_validate_mac_receipt_silent(
    path: &Path,
    model: &VerifiedCachedModel,
    operator_command: &str,
) -> Result<ReceiptCheckSummary> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read Mac receipt {}", path.display()))?;
    let mut receipt: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid Mac receipt {}", path.display()))?;
    let summary = validate_mac_receipt_value(path, &receipt)?;
    let Some(object) = receipt.as_object_mut() else {
        anyhow::bail!("Mac receipt {} is not a JSON object", path.display());
    };
    object.insert("operator_command".to_string(), serde_json::json!(operator_command));
    object.insert(
        "model_cache".to_string(),
        serde_json::json!({
            "id": model.id,
            "display_name": model.display_name,
            "cache_root": model.cache_root,
            "path": model.path,
            "sha256": model.sha256,
            "bytes": model.bytes,
            "architecture": model.architecture,
            "quantization": model.quantization,
            "tokenizer_model": model.tokenizer_model,
            "tokenizer_pre": model.tokenizer_pre,
            "chat_template": model.chat_template,
            "support_note": model.support_note,
        }),
    );
    object.insert(
        "mac_claim_boundary".to_string(),
        serde_json::json!({
            "slm_local_answer": true,
            "requested_backend": summary.requested_backend.as_str(),
            "bitnet_quality_claimed": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false,
        }),
    );
    object.entry("memory".to_string()).or_insert_with(memory_receipt_json);
    std::fs::write(path, serde_json::to_vec_pretty(&receipt)?)
        .with_context(|| format!("failed to update Mac receipt {}", path.display()))?;
    Ok(summary)
}

fn annotate_and_validate_bitnet_mac_ask_receipt(
    path: &Path,
    model: &VerifiedCachedModel,
    tokenizer: &Path,
    tokenizer_sha256: &str,
) -> Result<()> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read BitNet Mac ask receipt {}", path.display()))?;
    let mut receipt: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid BitNet Mac ask receipt {}", path.display()))?;
    let summary = validate_mac_receipt_value(path, &receipt)?;
    if receipt["model"]["sha256"].as_str() != Some(BITNET_M4_EXPECTED_MODEL_SHA256) {
        anyhow::bail!("{} does not use the accepted Microsoft BitNet I2_S GGUF", path.display());
    }
    if receipt["model"]["family"].as_str() != Some("bitnet")
        || receipt["loader"]["mode"].as_str()
            != Some(bitnet_models::GgufLoaderMode::RealGguf.as_str())
    {
        anyhow::bail!("{} does not record strict real-GGUF BitNet loading", path.display());
    }
    if receipt["tokenizer"]["strict"].as_bool() != Some(true)
        || receipt["tokenizer"]["pretokenizer_authority"].as_str() != Some("llama-bpe")
    {
        anyhow::bail!(
            "{} does not record strict external llama-bpe tokenizer authority",
            path.display()
        );
    }
    if receipt["prompt_render"]["template_family"].as_str() != Some(BITNET_M4_PROMPT_TEMPLATE) {
        anyhow::bail!("{} does not use the BitNet.cpp answer prompt template", path.display());
    }
    let Some(object) = receipt.as_object_mut() else {
        anyhow::bail!("BitNet Mac ask receipt {} is not a JSON object", path.display());
    };
    object.insert("operator_command".to_string(), serde_json::json!("mac ask"));
    object.insert(
        "model_cache".to_string(),
        serde_json::json!({
            "id": model.id,
            "display_name": model.display_name,
            "cache_root": model.cache_root,
            "path": model.path,
            "sha256": model.sha256,
            "bytes": model.bytes,
            "architecture": model.architecture,
            "quantization": model.quantization,
            "tokenizer_model": model.tokenizer_model,
            "tokenizer_pre": model.tokenizer_pre,
            "chat_template": model.chat_template,
            "support_note": model.support_note,
        }),
    );
    object.insert(
        "mac_bitnet_claim_boundary".to_string(),
        serde_json::json!({
            "bitnet_one_shot_mac_ask": true,
            "answer_corpus_proof_gate": "MODEL-ARTIFACT-007/M4-QA-001",
            "requested_backend": APPLE_M4_CPU_NEON,
            "tokenizer_path": tokenizer,
            "tokenizer_sha256": tokenizer_sha256,
            "chat_enabled": false,
            "serve_enabled": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false,
        }),
    );
    object.entry("bitnet_quality_claimed".to_string()).or_insert(serde_json::json!(false));
    object.entry("memory".to_string()).or_insert_with(memory_receipt_json);
    std::fs::write(path, serde_json::to_vec_pretty(&receipt)?)
        .with_context(|| format!("failed to update BitNet Mac ask receipt {}", path.display()))?;
    println!(
        "Mac BitNet receipt checked: {} ({}, generated_tokens={:?}, chat=false, serve=false)",
        path.display(),
        summary.artifact_kind,
        summary.generated_tokens
    );
    Ok(())
}

fn annotate_and_validate_bitnet_warm_session_receipt(
    path: &Path,
    model: &VerifiedCachedModel,
    tokenizer: &Path,
    tokenizer_sha256: &str,
    prompt_source: BitnetWarmPromptSource,
    profile_plan: Option<&BitnetWarmProfilePlan>,
    timeout_seconds: Option<u64>,
) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| {
        format!("failed to read BitNet warm-session receipt {}", path.display())
    })?;
    let mut receipt: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid BitNet warm-session receipt {}", path.display()))?;
    if receipt["artifact_kind"].as_str() != Some("slm_apple_m4_warm_session") {
        anyhow::bail!("{} was not produced by the warm-session receipt engine", path.display());
    }
    validate_mac_receipt_value(path, &receipt)?;
    if receipt["model"]["sha256"].as_str() != Some(BITNET_M4_EXPECTED_MODEL_SHA256) {
        anyhow::bail!("{} does not use the accepted Microsoft BitNet I2_S GGUF", path.display());
    }
    if receipt["model"]["family"].as_str() != Some("bitnet")
        || receipt["model"]["loader_mode"].as_str()
            != Some(bitnet_models::GgufLoaderMode::RealGguf.as_str())
    {
        anyhow::bail!(
            "{} does not record strict real-GGUF BitNet warm-session loading",
            path.display()
        );
    }
    if receipt["tokenizer"]["strict"].as_bool() != Some(true)
        || receipt["tokenizer"]["pretokenizer_authority"].as_str() != Some("llama-bpe")
    {
        anyhow::bail!(
            "{} does not record strict external llama-bpe tokenizer authority",
            path.display()
        );
    }
    if receipt["generation"]["prompt_template"].as_str() != Some(BITNET_M4_PROMPT_TEMPLATE) {
        anyhow::bail!("{} does not use the BitNet.cpp answer prompt template", path.display());
    }
    let prompt_count = receipt["session"]["prompt_count"].as_u64().unwrap_or_default();
    if prompt_source == BitnetWarmPromptSource::FixedProof
        && prompt_count != BITNET_WARM_PROMPTS.len() as u64
    {
        anyhow::bail!("{} must record the fixed BitNet warm proof prompt count", path.display());
    }
    if prompt_source == BitnetWarmPromptSource::OperatorPrompts && prompt_count < 2 {
        anyhow::bail!("{} must record at least two operator BitNet warm prompts", path.display());
    }
    if let Some(profile_plan) = profile_plan
        && prompt_count != profile_plan.max_prompt_count as u64
    {
        anyhow::bail!(
            "{} must record the requested BitNet warm profile prompt count",
            path.display()
        );
    }
    if receipt["session"]["model_loaded_once"].as_bool() != Some(true)
        || receipt["session"]["tokenizer_loaded_once"].as_bool() != Some(true)
    {
        anyhow::bail!(
            "{} must record one BitNet warm-session model/tokenizer load",
            path.display()
        );
    }
    if receipt["determinism"]["checked"].as_bool() != Some(true)
        || receipt["determinism"]["passed"].as_bool() != Some(true)
    {
        anyhow::bail!("{} must record passing repeated-prompt determinism", path.display());
    }
    let profile_set_receipt = profile_plan
        .map(|plan| bitnet_warm_profile_set_receipt_json(&receipt, plan))
        .transpose()?;
    let Some(object) = receipt.as_object_mut() else {
        anyhow::bail!("BitNet warm-session receipt {} is not a JSON object", path.display());
    };
    object.insert("artifact_kind".to_string(), serde_json::json!("bitnet_apple_m4_warm_session"));
    object.insert("operator_command".to_string(), serde_json::json!("mac bitnet-warm"));
    object.insert("model_id".to_string(), serde_json::json!(model.id));
    object.insert(
        "bitnet_warm_prompt_source".to_string(),
        serde_json::json!({
            "source": prompt_source.as_str(),
            "variable_prompts": prompt_source.variable_prompts(),
            "fixed_proof_prompt_count": BITNET_WARM_PROMPTS.len(),
            "session_prompt_count": prompt_count,
            "determinism_requires_repeated_prompt": true,
        }),
    );
    if let Some(profile_set_receipt) = profile_set_receipt {
        object.insert("bitnet_warm_profile_set".to_string(), profile_set_receipt);
    }
    object.insert(
        "timeout_boundary".to_string(),
        serde_json::json!({
            "configured_seconds": timeout_seconds,
            "reached": false,
            "enforced": timeout_seconds.is_some(),
            "status": "not_reached",
            "stage": serde_json::Value::Null,
        }),
    );
    object.insert(
        "model_cache".to_string(),
        serde_json::json!({
            "id": model.id,
            "display_name": model.display_name,
            "cache_root": model.cache_root,
            "path": model.path,
            "sha256": model.sha256,
            "bytes": model.bytes,
            "architecture": model.architecture,
            "quantization": model.quantization,
            "tokenizer_model": model.tokenizer_model,
            "tokenizer_pre": model.tokenizer_pre,
            "chat_template": model.chat_template,
            "support_note": model.support_note,
        }),
    );
    object.insert(
        "mac_bitnet_claim_boundary".to_string(),
        serde_json::json!({
            "bitnet_warm_session": true,
            "answer_corpus_proof_gate": "MODEL-ARTIFACT-007/M4-QA-001",
            "requested_backend": APPLE_M4_CPU_NEON,
            "tokenizer_path": tokenizer,
            "tokenizer_sha256": tokenizer_sha256,
            "chat_enabled": false,
            "serve_enabled": false,
            "full_metal_inference_claimed": false,
            "mpsgraph_inference_claimed": false,
            "neural_engine_execution_claimed": false,
            "qk256_apple_claimed": false,
            "broad_performance_claim": false,
            "speedup_claim": false,
        }),
    );
    object.entry("bitnet_quality_claimed".to_string()).or_insert(serde_json::json!(false));
    object.entry("broad_performance_claim".to_string()).or_insert(serde_json::json!(false));
    object.entry("speedup_claim".to_string()).or_insert(serde_json::json!(false));
    object.entry("memory".to_string()).or_insert_with(memory_receipt_json);
    if let Some(claim_boundary) =
        object.get_mut("claim_boundary").and_then(|value| value.as_object_mut())
    {
        claim_boundary.insert("bitnet_warm_session".to_string(), serde_json::json!(true));
        claim_boundary.insert("chat_enabled".to_string(), serde_json::json!(false));
        claim_boundary.insert("serve_enabled".to_string(), serde_json::json!(false));
        claim_boundary.insert("qk256_apple_claimed".to_string(), serde_json::json!(false));
        claim_boundary
            .insert("neural_engine_execution_claimed".to_string(), serde_json::json!(false));
        claim_boundary.insert("mpsgraph_inference_claimed".to_string(), serde_json::json!(false));
    }
    validate_mac_receipt_value(path, &receipt)?;
    std::fs::write(path, serde_json::to_vec_pretty(&receipt)?).with_context(|| {
        format!("failed to update BitNet warm-session receipt {}", path.display())
    })?;
    Ok(())
}

fn bitnet_warm_profile_plan_metadata_json(
    plan: Option<&BitnetWarmProfilePlan>,
) -> serde_json::Value {
    let Some(plan) = plan else {
        return serde_json::Value::Null;
    };
    serde_json::json!({
        "profiles_requested": plan.profile_ids(),
        "max_prompt_count": plan.max_prompt_count,
        "profile_execution_model": "single resident warm-session run with prefix checkpoints",
    })
}

fn bitnet_warm_profile_set_receipt_json(
    receipt: &serde_json::Value,
    plan: &BitnetWarmProfilePlan,
) -> Result<serde_json::Value> {
    let prompts = receipt["prompts"]
        .as_array()
        .ok_or_else(|| anyhow!("BitNet warm profile receipt is missing prompt summaries"))?;
    let prompt_receipts =
        receipt["session"]["per_prompt_receipts"].as_array().ok_or_else(|| {
            anyhow!("BitNet warm profile receipt is missing per-prompt receipt paths")
        })?;
    let mut profiles = Vec::with_capacity(plan.profiles.len());
    for profile in &plan.profiles {
        let prompt_count = profile.prompt_count();
        if prompts.len() < prompt_count || prompt_receipts.len() < prompt_count {
            anyhow::bail!(
                "BitNet warm profile {} requires {prompt_count} prompts, got {} summaries and {} receipt paths",
                profile.id(),
                prompts.len(),
                prompt_receipts.len()
            );
        }
        let prompt_slice = &prompts[..prompt_count];
        let generated_tokens = prompt_slice
            .iter()
            .map(|prompt| prompt["generated_tokens"].as_u64().unwrap_or_default())
            .sum::<u64>();
        let quality_passed = prompt_slice
            .iter()
            .all(|prompt| prompt["quality"]["passed"].as_bool().unwrap_or(false));
        let mut ttft_ms = Vec::with_capacity(prompt_count);
        let mut total_wall_ms = Vec::with_capacity(prompt_count);
        let mut decode_total_ms = Vec::with_capacity(prompt_count);
        for prompt in prompt_slice {
            if let Some(value) = prompt["timing"]["time_to_first_token_ms"].as_f64() {
                ttft_ms.push(value);
            }
            if let Some(value) = prompt["timing"]["total_ms"].as_f64() {
                total_wall_ms.push(value);
            }
            if let Some(value) = prompt["timing"]["decode_total_ms"].as_f64() {
                decode_total_ms.push(value);
            }
        }
        profiles.push(serde_json::json!({
            "profile_id": profile.id(),
            "prompt_count": prompt_count,
            "receipt_scope": "prefix_checkpoint",
            "profile_execution_model": "same resident session as the aggregate receipt",
            "per_prompt_receipts": prompt_receipts[..prompt_count].to_vec(),
            "generated_tokens": generated_tokens,
            "quality_passed": quality_passed,
            "determinism_checked": receipt["determinism"]["checked"].as_bool().unwrap_or(false),
            "determinism_passed": receipt["determinism"]["passed"].as_bool().unwrap_or(false),
            "timing": {
                "time_to_first_token_ms": benchmark_stat_json(&ttft_ms),
                "total_wall_ms": benchmark_stat_json(&total_wall_ms),
                "decode_total_ms": benchmark_stat_json(&decode_total_ms),
            },
            "memory": receipt["memory"].clone(),
            "claim_boundary": {
                "scope": "this profile checkpoint, model, backend, and machine receipt only",
                "chat_enabled": false,
                "serve_enabled": false,
                "broad_performance_claim": false,
                "speedup_claim": false,
            },
        }));
    }
    Ok(serde_json::json!({
        "profiles_requested": plan.profile_ids(),
        "max_prompt_count": plan.max_prompt_count,
        "resident_session_count": 1,
        "profile_execution_model": "single resident warm-session run with prefix checkpoints",
        "profile_summaries": profiles,
        "per_turn_receipts": receipt["session"]["per_prompt_receipts"].clone(),
        "aggregate_timing": receipt["speed"]["timing"].clone(),
        "aggregate_memory": receipt["memory"].clone(),
        "determinism": receipt["determinism"].clone(),
        "chat_enabled": false,
        "serve_enabled": false,
    }))
}

fn run_receipts_check(path: &Path, regression_baseline: Option<&Path>, json: bool) -> Result<()> {
    let receipt_paths = collect_receipt_paths(path)?;
    if receipt_paths.is_empty() {
        anyhow::bail!("no JSON receipts found under {}", path.display());
    }
    let baseline = regression_baseline.map(load_regression_baseline).transpose()?;
    let mut summaries = Vec::with_capacity(receipt_paths.len());
    let mut regression_compared = 0usize;
    for receipt_path in receipt_paths {
        let receipt: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&receipt_path)
                .with_context(|| format!("failed to read {}", receipt_path.display()))?,
        )
        .with_context(|| format!("invalid JSON receipt {}", receipt_path.display()))?;
        let mut summary = validate_mac_receipt_value(&receipt_path, &receipt)?;
        if let Some(baseline) = &baseline
            && receipt["artifact_kind"].as_str() == baseline.receipt["artifact_kind"].as_str()
        {
            summary.regression = Some(compare_mac_regression(&receipt_path, &receipt, baseline)?);
            regression_compared += 1;
        }
        summaries.push(summary);
    }
    if baseline.is_some() && regression_compared == 0 {
        anyhow::bail!(
            "regression baseline was provided, but no matching Apple M4 dense-SLM or BitNet receipts were found under {}",
            path.display()
        );
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&summaries)?);
    } else {
        for summary in &summaries {
            println!(
                "ok: {} ({}, prompts={:?}, generated_tokens={:?})",
                summary.path.display(),
                summary.artifact_kind,
                summary.prompt_count,
                summary.generated_tokens
            );
            if let Some(regression) = &summary.regression {
                println!(
                    "regression: baseline={}, advisory=true, warnings={}",
                    regression.baseline_path.display(),
                    regression.warning_count
                );
                for warning in &regression.warnings {
                    println!(
                        "warning: {} {} baseline={} observed={} threshold={}%",
                        warning.profile_id,
                        warning.field,
                        warning.baseline,
                        warning.observed,
                        warning.threshold_percent
                    );
                }
            }
        }
    }
    Ok(())
}

fn run_regression_check(
    path: &Path,
    baseline_path: &Path,
    fail_on_drift: bool,
    json: bool,
) -> Result<()> {
    let receipt_paths = collect_receipt_paths(path)?;
    if receipt_paths.is_empty() {
        anyhow::bail!("no JSON receipts found under {}", path.display());
    }
    let baseline = load_regression_baseline(baseline_path)?;
    let single_input = path.is_file();
    let mut summaries = Vec::with_capacity(receipt_paths.len());
    let mut regression_compared = 0usize;
    let mut warning_count = 0usize;
    for receipt_path in receipt_paths {
        let receipt: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&receipt_path)
                .with_context(|| format!("failed to read {}", receipt_path.display()))?,
        )
        .with_context(|| format!("invalid JSON receipt {}", receipt_path.display()))?;
        let mut summary = validate_mac_receipt_value(&receipt_path, &receipt)?;
        if receipt["artifact_kind"].as_str() == baseline.receipt["artifact_kind"].as_str() {
            match compare_mac_regression(&receipt_path, &receipt, &baseline) {
                Ok(regression) => {
                    warning_count += regression.warning_count;
                    summary.regression = Some(regression);
                    regression_compared += 1;
                }
                Err(error) if single_input => return Err(error),
                Err(_) => {}
            }
        }
        summaries.push(summary);
    }
    if regression_compared == 0 {
        anyhow::bail!(
            "no matching Apple M4 dense-SLM or BitNet receipts under {} could be compared to baseline {}",
            path.display(),
            baseline_path.display()
        );
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&summaries)?);
    } else {
        for summary in &summaries {
            println!(
                "ok: {} ({}, prompts={:?}, generated_tokens={:?})",
                summary.path.display(),
                summary.artifact_kind,
                summary.prompt_count,
                summary.generated_tokens
            );
            if let Some(regression) = &summary.regression {
                println!(
                    "regression: baseline={}, advisory={}, fail_on_drift={}, warnings={}",
                    regression.baseline_path.display(),
                    regression.advisory,
                    fail_on_drift,
                    regression.warning_count
                );
                for warning in &regression.warnings {
                    println!(
                        "warning: {} {} baseline={} observed={} threshold={}%",
                        warning.profile_id,
                        warning.field,
                        warning.baseline,
                        warning.observed,
                        warning.threshold_percent
                    );
                }
            }
        }
    }
    if fail_on_drift && warning_count > 0 {
        anyhow::bail!(
            "Mac regression drift exceeded advisory thresholds: {warning_count} warning(s)"
        );
    }
    Ok(())
}

fn collect_receipt_paths(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if !path.is_dir() {
        anyhow::bail!("receipt path does not exist: {}", path.display());
    }
    let mut out = Vec::new();
    collect_receipt_paths_recursive(path, &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_receipt_paths_recursive(path: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in
        std::fs::read_dir(path).with_context(|| format!("failed to read {}", path.display()))?
    {
        let entry = entry?;
        let entry_path = entry.path();
        if entry_path.is_dir() {
            collect_receipt_paths_recursive(&entry_path, out)?;
        } else if entry_path.extension().and_then(|ext| ext.to_str()) == Some("json") {
            out.push(entry_path);
        }
    }
    Ok(())
}

struct RegressionBaseline {
    path: PathBuf,
    receipt: serde_json::Value,
}

fn load_regression_baseline(path: &Path) -> Result<RegressionBaseline> {
    let receipt: serde_json::Value = serde_json::from_slice(
        &std::fs::read(path)
            .with_context(|| format!("failed to read regression baseline {}", path.display()))?,
    )
    .with_context(|| format!("invalid JSON regression baseline {}", path.display()))?;
    validate_mac_receipt_value(path, &receipt)?;
    if !matches!(
        receipt["artifact_kind"].as_str(),
        Some("apple_m4_slm_performance_profiles")
            | Some("slm_apple_m4_warm_session")
            | Some("apple_m4_slm_eval_summary")
            | Some("apple_m4_slm_benchmark_v2")
            | Some("bitnet_apple_m4_local_answer_corpus")
            | Some("bitnet_apple_m4_warm_session")
            | Some("bitnet_apple_m4_benchmark_v1")
    ) {
        anyhow::bail!(
            "regression baseline {} must be an apple_m4_slm_performance_profiles, slm_apple_m4_warm_session, apple_m4_slm_eval_summary, apple_m4_slm_benchmark_v2, bitnet_apple_m4_local_answer_corpus, bitnet_apple_m4_warm_session, or bitnet_apple_m4_benchmark_v1 receipt",
            path.display()
        );
    }
    Ok(RegressionBaseline { path: path.to_path_buf(), receipt })
}

fn compare_mac_regression(
    path: &Path,
    receipt: &serde_json::Value,
    baseline: &RegressionBaseline,
) -> Result<RegressionCheckSummary> {
    match receipt["artifact_kind"].as_str() {
        Some("apple_m4_slm_performance_profiles") => {
            compare_dense_slm_performance_regression(path, receipt, baseline)
        }
        Some("slm_apple_m4_warm_session") => {
            compare_dense_slm_warm_session_regression(path, receipt, baseline)
        }
        Some("apple_m4_slm_eval_summary") => {
            compare_dense_slm_eval_summary_regression(path, receipt, baseline)
        }
        Some("apple_m4_slm_benchmark_v2") => {
            compare_dense_slm_benchmark_v2_regression(path, receipt, baseline)
        }
        Some("bitnet_apple_m4_local_answer_corpus") => {
            compare_bitnet_eval_answer_corpus_regression(path, receipt, baseline)
        }
        Some("bitnet_apple_m4_warm_session") => {
            compare_bitnet_warm_session_regression(path, receipt, baseline)
        }
        Some("bitnet_apple_m4_benchmark_v1") => {
            compare_bitnet_benchmark_v1_regression(path, receipt, baseline)
        }
        _ => anyhow::bail!("{} is not an Apple M4 regression envelope receipt", path.display()),
    }
}

fn compare_dense_slm_performance_regression(
    path: &Path,
    receipt: &serde_json::Value,
    baseline: &RegressionBaseline,
) -> Result<RegressionCheckSummary> {
    const DECODE_TOK_S_LOWER_PCT: f64 = 12.5;
    const WARM_PROMPT_TOK_S_LOWER_PCT: f64 = 15.0;
    const TIME_TO_FIRST_TOKEN_HIGHER_PCT: f64 = 15.0;
    const TOTAL_SESSION_MS_HIGHER_PCT: f64 = 15.0;
    const PEAK_MEMORY_MB_HIGHER_PCT: f64 = 10.0;

    ensure_regression_context_matches(path, receipt, &baseline.path, &baseline.receipt)?;

    let profiles = receipt["profiles"].as_array().ok_or_else(|| {
        anyhow!("{} performance regression receipt is missing profiles", path.display())
    })?;
    let mut warnings = Vec::new();
    for profile in profiles {
        let Some(profile_id) = profile["profile_id"].as_str() else {
            anyhow::bail!("{} performance profile is missing profile_id", path.display());
        };
        let baseline_profile = find_profile(&baseline.receipt, profile_id).ok_or_else(|| {
            anyhow!(
                "regression baseline {} is missing profile {profile_id}",
                baseline.path.display()
            )
        })?;

        compare_lower_is_worse(
            &mut warnings,
            profile_id,
            "timing.decode_generated_tok_s",
            regression_metric(baseline_profile, &["timing", "decode_generated_tok_s"])?,
            regression_metric(profile, &["timing", "decode_generated_tok_s"])?,
            DECODE_TOK_S_LOWER_PCT,
        );
        compare_lower_is_worse(
            &mut warnings,
            profile_id,
            "timing.warm_prompt_generated_tok_s",
            regression_metric(baseline_profile, &["timing", "warm_prompt_generated_tok_s"])?,
            regression_metric(profile, &["timing", "warm_prompt_generated_tok_s"])?,
            WARM_PROMPT_TOK_S_LOWER_PCT,
        );
        compare_higher_is_worse(
            &mut warnings,
            profile_id,
            "timing.time_to_first_token_ms",
            regression_metric(baseline_profile, &["timing", "time_to_first_token_ms"])?,
            regression_metric(profile, &["timing", "time_to_first_token_ms"])?,
            TIME_TO_FIRST_TOKEN_HIGHER_PCT,
        );
        compare_higher_is_worse(
            &mut warnings,
            profile_id,
            "timing.total_session_ms",
            regression_metric(baseline_profile, &["timing", "total_session_ms"])?,
            regression_metric(profile, &["timing", "total_session_ms"])?,
            TOTAL_SESSION_MS_HIGHER_PCT,
        );
        compare_higher_is_worse(
            &mut warnings,
            profile_id,
            "memory.peak_memory_mb",
            regression_metric(baseline_profile, &["memory", "peak_memory_mb"])?,
            regression_metric(profile, &["memory", "peak_memory_mb"])?,
            PEAK_MEMORY_MB_HIGHER_PCT,
        );
    }

    Ok(RegressionCheckSummary {
        baseline_path: baseline.path.clone(),
        advisory: true,
        matched_context: true,
        warning_count: warnings.len(),
        warnings,
    })
}

fn compare_dense_slm_warm_session_regression(
    path: &Path,
    receipt: &serde_json::Value,
    baseline: &RegressionBaseline,
) -> Result<RegressionCheckSummary> {
    const DECODE_TOK_S_LOWER_PCT: f64 = 12.5;
    const WARM_PROMPT_TOK_S_LOWER_PCT: f64 = 15.0;
    const TIME_TO_FIRST_TOKEN_HIGHER_PCT: f64 = 15.0;
    const TOTAL_SESSION_MS_HIGHER_PCT: f64 = 15.0;
    const PEAK_MEMORY_MB_HIGHER_PCT: f64 = 10.0;

    ensure_warm_session_regression_context_matches(
        path,
        receipt,
        &baseline.path,
        &baseline.receipt,
    )?;

    let mut warnings = Vec::new();
    compare_lower_is_worse(
        &mut warnings,
        "warm_session",
        "speed.throughput.decode_generated_tok_s",
        regression_metric(&baseline.receipt, &["speed", "throughput", "decode_generated_tok_s"])?,
        regression_metric(receipt, &["speed", "throughput", "decode_generated_tok_s"])?,
        DECODE_TOK_S_LOWER_PCT,
    );
    compare_lower_is_worse(
        &mut warnings,
        "warm_session",
        "speed.throughput.warm_prompt_generated_tok_s",
        regression_metric(
            &baseline.receipt,
            &["speed", "throughput", "warm_prompt_generated_tok_s"],
        )?,
        regression_metric(receipt, &["speed", "throughput", "warm_prompt_generated_tok_s"])?,
        WARM_PROMPT_TOK_S_LOWER_PCT,
    );
    compare_higher_is_worse(
        &mut warnings,
        "warm_session",
        "speed.timing.time_to_first_token_ms",
        regression_metric(&baseline.receipt, &["speed", "timing", "time_to_first_token_ms"])?,
        regression_metric(receipt, &["speed", "timing", "time_to_first_token_ms"])?,
        TIME_TO_FIRST_TOKEN_HIGHER_PCT,
    );
    compare_higher_is_worse(
        &mut warnings,
        "warm_session",
        "timing.total_session_ms",
        regression_metric(&baseline.receipt, &["timing", "total_session_ms"])?,
        regression_metric(receipt, &["timing", "total_session_ms"])?,
        TOTAL_SESSION_MS_HIGHER_PCT,
    );
    compare_higher_is_worse(
        &mut warnings,
        "warm_session",
        "memory.peak_memory_mb",
        regression_metric(&baseline.receipt, &["memory", "peak_memory_mb"])?,
        regression_metric(receipt, &["memory", "peak_memory_mb"])?,
        PEAK_MEMORY_MB_HIGHER_PCT,
    );

    Ok(RegressionCheckSummary {
        baseline_path: baseline.path.clone(),
        advisory: true,
        matched_context: true,
        warning_count: warnings.len(),
        warnings,
    })
}

fn compare_bitnet_warm_session_regression(
    path: &Path,
    receipt: &serde_json::Value,
    baseline: &RegressionBaseline,
) -> Result<RegressionCheckSummary> {
    const DECODE_TOK_S_LOWER_PCT: f64 = 12.5;
    const WARM_PROMPT_TOK_S_LOWER_PCT: f64 = 15.0;
    const TIME_TO_FIRST_TOKEN_HIGHER_PCT: f64 = 15.0;
    const LOAD_HIGHER_PCT: f64 = 20.0;
    const PREFILL_HIGHER_PCT: f64 = 15.0;
    const SAMPLING_HIGHER_PCT: f64 = 20.0;
    const TOTAL_SESSION_MS_HIGHER_PCT: f64 = 15.0;
    const RESIDENT_MEMORY_HIGHER_PCT: f64 = 10.0;

    ensure_bitnet_warm_session_regression_context_matches(
        path,
        receipt,
        &baseline.path,
        &baseline.receipt,
    )?;

    let mut warnings = Vec::new();
    compare_lower_is_worse(
        &mut warnings,
        "bitnet_warm_session",
        "speed.throughput.decode_generated_tok_s",
        regression_metric(&baseline.receipt, &["speed", "throughput", "decode_generated_tok_s"])?,
        regression_metric(receipt, &["speed", "throughput", "decode_generated_tok_s"])?,
        DECODE_TOK_S_LOWER_PCT,
    );
    compare_lower_is_worse(
        &mut warnings,
        "bitnet_warm_session",
        "speed.throughput.warm_prompt_generated_tok_s",
        regression_metric(
            &baseline.receipt,
            &["speed", "throughput", "warm_prompt_generated_tok_s"],
        )?,
        regression_metric(receipt, &["speed", "throughput", "warm_prompt_generated_tok_s"])?,
        WARM_PROMPT_TOK_S_LOWER_PCT,
    );
    compare_higher_is_worse(
        &mut warnings,
        "bitnet_warm_session",
        "speed.timing.time_to_first_token_ms",
        regression_metric(&baseline.receipt, &["speed", "timing", "time_to_first_token_ms"])?,
        regression_metric(receipt, &["speed", "timing", "time_to_first_token_ms"])?,
        TIME_TO_FIRST_TOKEN_HIGHER_PCT,
    );
    compare_higher_is_worse(
        &mut warnings,
        "bitnet_warm_session",
        "speed.timing.model_load_ms",
        regression_metric(&baseline.receipt, &["speed", "timing", "model_load_ms"])?,
        regression_metric(receipt, &["speed", "timing", "model_load_ms"])?,
        LOAD_HIGHER_PCT,
    );
    compare_higher_is_worse(
        &mut warnings,
        "bitnet_warm_session",
        "speed.timing.tokenizer_load_ms",
        regression_metric(&baseline.receipt, &["speed", "timing", "tokenizer_load_ms"])?,
        regression_metric(receipt, &["speed", "timing", "tokenizer_load_ms"])?,
        LOAD_HIGHER_PCT,
    );
    compare_higher_is_worse(
        &mut warnings,
        "bitnet_warm_session",
        "speed.timing.prefill_ms",
        regression_metric(&baseline.receipt, &["speed", "timing", "prefill_ms"])?,
        regression_metric(receipt, &["speed", "timing", "prefill_ms"])?,
        PREFILL_HIGHER_PCT,
    );
    compare_higher_is_worse(
        &mut warnings,
        "bitnet_warm_session",
        "speed.timing.sampling_ms",
        regression_metric(&baseline.receipt, &["speed", "timing", "sampling_ms"])?,
        regression_metric(receipt, &["speed", "timing", "sampling_ms"])?,
        SAMPLING_HIGHER_PCT,
    );
    compare_higher_is_worse(
        &mut warnings,
        "bitnet_warm_session",
        "timing.total_session_ms",
        regression_metric(&baseline.receipt, &["timing", "total_session_ms"])?,
        regression_metric(receipt, &["timing", "total_session_ms"])?,
        TOTAL_SESSION_MS_HIGHER_PCT,
    );
    compare_higher_is_worse(
        &mut warnings,
        "bitnet_warm_session",
        "memory.resident_memory_bytes",
        regression_metric(&baseline.receipt, &["memory", "resident_memory_bytes"])?,
        regression_metric(receipt, &["memory", "resident_memory_bytes"])?,
        RESIDENT_MEMORY_HIGHER_PCT,
    );

    Ok(RegressionCheckSummary {
        baseline_path: baseline.path.clone(),
        advisory: true,
        matched_context: true,
        warning_count: warnings.len(),
        warnings,
    })
}

fn compare_dense_slm_eval_summary_regression(
    path: &Path,
    receipt: &serde_json::Value,
    baseline: &RegressionBaseline,
) -> Result<RegressionCheckSummary> {
    const ACCURACY_LOWER_PCT: f64 = 0.0;
    const DECODE_TOK_S_LOWER_PCT: f64 = 12.5;
    const THROUGHPUT_LOWER_PCT: f64 = 15.0;
    const LATENCY_HIGHER_PCT: f64 = 15.0;
    const LOAD_HIGHER_PCT: f64 = 20.0;
    const SAMPLING_HIGHER_PCT: f64 = 20.0;
    const PEAK_MEMORY_MB_HIGHER_PCT: f64 = 10.0;

    ensure_slm_eval_summary_regression_context_matches(
        path,
        receipt,
        &baseline.path,
        &baseline.receipt,
    )?;

    let mut warnings = Vec::new();
    for field in [
        "cases_passed",
        "exact_match",
        "normalized_match",
        "json_schema_pass",
        "numeric_tolerance_pass",
        "required_keywords_pass",
        "forbidden_tokens_pass",
    ] {
        compare_lower_is_worse(
            &mut warnings,
            "seeded_corpus",
            &format!("accuracy.{field}"),
            regression_metric(&baseline.receipt, &["accuracy", field])?,
            regression_metric(receipt, &["accuracy", field])?,
            ACCURACY_LOWER_PCT,
        );
    }
    compare_slm_eval_scoring_summary_regression(
        &mut warnings,
        path,
        receipt,
        &baseline.path,
        &baseline.receipt,
        ACCURACY_LOWER_PCT,
    )?;
    compare_slm_eval_task_family_regression(
        &mut warnings,
        path,
        receipt,
        &baseline.path,
        &baseline.receipt,
        ACCURACY_LOWER_PCT,
    )?;
    for (field, threshold) in [
        ("input_tok_s_p50", THROUGHPUT_LOWER_PCT),
        ("output_tok_s_p50", THROUGHPUT_LOWER_PCT),
        ("decode_tok_s_p50", DECODE_TOK_S_LOWER_PCT),
    ] {
        compare_lower_is_worse(
            &mut warnings,
            "seeded_corpus",
            &format!("speed.{field}"),
            regression_metric(&baseline.receipt, &["speed", field])?,
            regression_metric(receipt, &["speed", field])?,
            threshold,
        );
    }
    for (field, threshold) in [
        ("cold_load_ms_p50", LOAD_HIGHER_PCT),
        ("tokenizer_load_ms_p50", LOAD_HIGHER_PCT),
        ("prompt_tokenize_ms_p50", LOAD_HIGHER_PCT),
        ("prefill_ms_p50", LATENCY_HIGHER_PCT),
        ("ttft_ms_p50", LATENCY_HIGHER_PCT),
        ("ttft_ms_p90", LATENCY_HIGHER_PCT),
        ("sampling_ms_per_token_p50", SAMPLING_HIGHER_PCT),
        ("total_wall_ms_p50", LATENCY_HIGHER_PCT),
    ] {
        compare_higher_is_worse(
            &mut warnings,
            "seeded_corpus",
            &format!("speed.{field}"),
            regression_metric(&baseline.receipt, &["speed", field])?,
            regression_metric(receipt, &["speed", field])?,
            threshold,
        );
    }
    compare_higher_is_worse(
        &mut warnings,
        "resident_stability",
        "memory.peak_memory_mb",
        regression_metric(&baseline.receipt, &["memory", "peak_memory_mb"])?,
        regression_metric(receipt, &["memory", "peak_memory_mb"])?,
        PEAK_MEMORY_MB_HIGHER_PCT,
    );

    Ok(RegressionCheckSummary {
        baseline_path: baseline.path.clone(),
        advisory: true,
        matched_context: true,
        warning_count: warnings.len(),
        warnings,
    })
}

fn compare_dense_slm_benchmark_v2_regression(
    path: &Path,
    receipt: &serde_json::Value,
    baseline: &RegressionBaseline,
) -> Result<RegressionCheckSummary> {
    const DECODE_TOK_S_LOWER_PCT: f64 = 12.5;
    const THROUGHPUT_LOWER_PCT: f64 = 15.0;
    const LATENCY_HIGHER_PCT: f64 = 15.0;
    const LOAD_HIGHER_PCT: f64 = 20.0;
    const SAMPLING_HIGHER_PCT: f64 = 20.0;
    const PEAK_MEMORY_MB_HIGHER_PCT: f64 = 10.0;
    const MEMORY_DRIFT_MB_HIGHER_PCT: f64 = 15.0;

    ensure_slm_benchmark_v2_regression_context_matches(
        path,
        receipt,
        &baseline.path,
        &baseline.receipt,
    )?;

    let mut warnings = Vec::new();
    for percentile in ["p50", "p90", "p99"] {
        for (metric, threshold) in [
            ("cold_load_ms", LOAD_HIGHER_PCT),
            ("tokenizer_load_ms", LOAD_HIGHER_PCT),
            ("prompt_tokenize_ms", LATENCY_HIGHER_PCT),
            ("prefill_ms", LATENCY_HIGHER_PCT),
            ("ttft_ms", LATENCY_HIGHER_PCT),
            ("total_wall_ms", LATENCY_HIGHER_PCT),
        ] {
            let field = format!("{metric}_{percentile}");
            compare_higher_is_worse(
                &mut warnings,
                "benchmark_summary",
                &format!("speed.{field}"),
                regression_metric(&baseline.receipt, &["speed", field.as_str()])?,
                regression_metric(receipt, &["speed", field.as_str()])?,
                threshold,
            );
        }
        for (metric, threshold) in [
            ("input_tok_s", THROUGHPUT_LOWER_PCT),
            ("output_tok_s", THROUGHPUT_LOWER_PCT),
            ("decode_tok_s", DECODE_TOK_S_LOWER_PCT),
        ] {
            let field = format!("{metric}_{percentile}");
            compare_lower_is_worse(
                &mut warnings,
                "benchmark_summary",
                &format!("speed.{field}"),
                regression_metric(&baseline.receipt, &["speed", field.as_str()])?,
                regression_metric(receipt, &["speed", field.as_str()])?,
                threshold,
            );
        }
        for (metric, threshold) in [
            ("peak_memory_mb", PEAK_MEMORY_MB_HIGHER_PCT),
            ("memory_drift_mb", MEMORY_DRIFT_MB_HIGHER_PCT),
        ] {
            let field = format!("{metric}_{percentile}");
            compare_higher_is_worse(
                &mut warnings,
                "benchmark_summary",
                &format!("memory.{field}"),
                regression_metric(&baseline.receipt, &["memory", field.as_str()])?,
                regression_metric(receipt, &["memory", field.as_str()])?,
                threshold,
            );
        }
    }

    let profiles = receipt["profiles"].as_array().ok_or_else(|| {
        anyhow!("{} SLM benchmark v2 summary is missing profiles", path.display())
    })?;
    for profile in profiles {
        let Some(profile_id) = profile["profile_id"].as_str() else {
            anyhow::bail!("{} SLM benchmark v2 profile is missing profile_id", path.display());
        };
        let baseline_profile = find_profile(&baseline.receipt, profile_id).ok_or_else(|| {
            anyhow!(
                "regression baseline {} is missing SLM benchmark v2 profile {profile_id}",
                baseline.path.display()
            )
        })?;

        for percentile in ["p50", "p90", "p99"] {
            for (metric, threshold) in [
                ("cold_load_ms", LOAD_HIGHER_PCT),
                ("tokenizer_load_ms", LOAD_HIGHER_PCT),
                ("prompt_tokenize_ms", LATENCY_HIGHER_PCT),
                ("prefill_ms", LATENCY_HIGHER_PCT),
                ("time_to_first_token_ms", LATENCY_HIGHER_PCT),
                ("decode_total_ms", LATENCY_HIGHER_PCT),
                ("sampling_ms_per_token", SAMPLING_HIGHER_PCT),
                ("total_wall_ms", LATENCY_HIGHER_PCT),
            ] {
                compare_higher_is_worse(
                    &mut warnings,
                    profile_id,
                    &format!("timing.{metric}.{percentile}"),
                    regression_metric(baseline_profile, &["timing", metric, percentile])?,
                    regression_metric(profile, &["timing", metric, percentile])?,
                    threshold,
                );
            }
            for (metric, threshold) in [
                ("input_tokens_per_second", THROUGHPUT_LOWER_PCT),
                ("output_tokens_per_second", THROUGHPUT_LOWER_PCT),
                ("decode_tokens_per_second", DECODE_TOK_S_LOWER_PCT),
            ] {
                compare_lower_is_worse(
                    &mut warnings,
                    profile_id,
                    &format!("throughput.{metric}.{percentile}"),
                    regression_metric(baseline_profile, &["throughput", metric, percentile])?,
                    regression_metric(profile, &["throughput", metric, percentile])?,
                    threshold,
                );
            }
            for (metric, threshold) in [
                ("peak_memory_mb", PEAK_MEMORY_MB_HIGHER_PCT),
                ("memory_drift_mb", MEMORY_DRIFT_MB_HIGHER_PCT),
            ] {
                compare_higher_is_worse(
                    &mut warnings,
                    profile_id,
                    &format!("memory.{metric}.{percentile}"),
                    regression_metric(baseline_profile, &["memory", metric, percentile])?,
                    regression_metric(profile, &["memory", metric, percentile])?,
                    threshold,
                );
            }
        }
    }

    Ok(RegressionCheckSummary {
        baseline_path: baseline.path.clone(),
        advisory: true,
        matched_context: true,
        warning_count: warnings.len(),
        warnings,
    })
}

fn compare_bitnet_eval_answer_corpus_regression(
    path: &Path,
    receipt: &serde_json::Value,
    baseline: &RegressionBaseline,
) -> Result<RegressionCheckSummary> {
    const STRICT_QUALITY_PCT: f64 = 0.0;

    ensure_bitnet_eval_answer_corpus_regression_context_matches(
        path,
        receipt,
        &baseline.path,
        &baseline.receipt,
    )?;

    let mut warnings = Vec::new();
    for field in ["passed"] {
        compare_lower_is_worse(
            &mut warnings,
            "bitnet_eval:quality_summary",
            &format!("quality_summary.{field}"),
            regression_metric(&baseline.receipt, &["quality_summary", field])?,
            regression_metric(receipt, &["quality_summary", field])?,
            STRICT_QUALITY_PCT,
        );
        compare_lower_is_worse(
            &mut warnings,
            "bitnet_eval:scoring_summary",
            &format!("scoring_summary.{field}"),
            regression_metric(&baseline.receipt, &["scoring_summary", field])?,
            regression_metric(receipt, &["scoring_summary", field])?,
            STRICT_QUALITY_PCT,
        );
    }
    for field in ["failed", "timeout", "not_run"] {
        compare_higher_is_worse(
            &mut warnings,
            "bitnet_eval:quality_summary",
            &format!("quality_summary.{field}"),
            regression_metric(&baseline.receipt, &["quality_summary", field])?,
            regression_metric(receipt, &["quality_summary", field])?,
            STRICT_QUALITY_PCT,
        );
    }
    for field in ["failed", "not_run"] {
        compare_higher_is_worse(
            &mut warnings,
            "bitnet_eval:scoring_summary",
            &format!("scoring_summary.{field}"),
            regression_metric(&baseline.receipt, &["scoring_summary", field])?,
            regression_metric(receipt, &["scoring_summary", field])?,
            STRICT_QUALITY_PCT,
        );
    }
    compare_bitnet_eval_task_family_regression(
        &mut warnings,
        path,
        receipt,
        &baseline.path,
        &baseline.receipt,
        STRICT_QUALITY_PCT,
    )?;
    compare_bitnet_eval_reference_regression(
        &mut warnings,
        path,
        receipt,
        &baseline.path,
        &baseline.receipt,
        STRICT_QUALITY_PCT,
    )?;

    Ok(RegressionCheckSummary {
        baseline_path: baseline.path.clone(),
        advisory: true,
        matched_context: true,
        warning_count: warnings.len(),
        warnings,
    })
}

fn compare_bitnet_benchmark_v1_regression(
    path: &Path,
    receipt: &serde_json::Value,
    baseline: &RegressionBaseline,
) -> Result<RegressionCheckSummary> {
    const DECODE_TOK_S_LOWER_PCT: f64 = 12.5;
    const THROUGHPUT_LOWER_PCT: f64 = 15.0;
    const LATENCY_HIGHER_PCT: f64 = 15.0;
    const LOAD_HIGHER_PCT: f64 = 20.0;
    const SAMPLING_HIGHER_PCT: f64 = 20.0;
    const PEAK_MEMORY_MB_HIGHER_PCT: f64 = 10.0;
    const MEMORY_DRIFT_MB_HIGHER_PCT: f64 = 15.0;

    ensure_bitnet_benchmark_v1_regression_context_matches(
        path,
        receipt,
        &baseline.path,
        &baseline.receipt,
    )?;

    let mut warnings = Vec::new();
    for percentile in ["p50", "p90", "p99"] {
        for (metric, threshold) in [
            ("cold_load_ms", LOAD_HIGHER_PCT),
            ("model_load_ms", LOAD_HIGHER_PCT),
            ("tokenizer_load_ms", LOAD_HIGHER_PCT),
            ("prompt_tokenize_ms", LATENCY_HIGHER_PCT),
            ("prefill_ms", LATENCY_HIGHER_PCT),
            ("ttft_ms", LATENCY_HIGHER_PCT),
            ("decode_total_ms", LATENCY_HIGHER_PCT),
            ("sampling_ms_per_token", SAMPLING_HIGHER_PCT),
            ("total_wall_ms", LATENCY_HIGHER_PCT),
        ] {
            let field = format!("{metric}_{percentile}");
            compare_higher_is_worse(
                &mut warnings,
                "bitnet_benchmark:summary",
                &format!("speed.{field}"),
                regression_metric(&baseline.receipt, &["speed", field.as_str()])?,
                regression_metric(receipt, &["speed", field.as_str()])?,
                threshold,
            );
        }
        for (metric, threshold) in [
            ("input_tok_s", THROUGHPUT_LOWER_PCT),
            ("output_tok_s", THROUGHPUT_LOWER_PCT),
            ("decode_tok_s", DECODE_TOK_S_LOWER_PCT),
        ] {
            let field = format!("{metric}_{percentile}");
            compare_lower_is_worse(
                &mut warnings,
                "bitnet_benchmark:summary",
                &format!("speed.{field}"),
                regression_metric(&baseline.receipt, &["speed", field.as_str()])?,
                regression_metric(receipt, &["speed", field.as_str()])?,
                threshold,
            );
        }
        for (metric, threshold) in [
            ("peak_memory_mb", PEAK_MEMORY_MB_HIGHER_PCT),
            ("memory_drift_mb", MEMORY_DRIFT_MB_HIGHER_PCT),
        ] {
            let field = format!("{metric}_{percentile}");
            compare_higher_is_worse(
                &mut warnings,
                "bitnet_benchmark:summary",
                &format!("memory.{field}"),
                regression_metric(&baseline.receipt, &["memory", field.as_str()])?,
                regression_metric(receipt, &["memory", field.as_str()])?,
                threshold,
            );
        }
    }
    compare_higher_is_worse(
        &mut warnings,
        "bitnet_benchmark:summary",
        "memory.process_peak_drift_mb",
        regression_metric(&baseline.receipt, &["memory", "process_peak_drift_mb"])?,
        regression_metric(receipt, &["memory", "process_peak_drift_mb"])?,
        MEMORY_DRIFT_MB_HIGHER_PCT,
    );
    for (path_key, profile_id) in
        [("one_shot", "bitnet_path:one_shot"), ("fixed_warm", "bitnet_path:fixed_warm")]
    {
        compare_bitnet_benchmark_path_regression(
            &mut warnings,
            profile_id,
            &receipt["paths"][path_key],
            &baseline.receipt["paths"][path_key],
        )?;
    }

    Ok(RegressionCheckSummary {
        baseline_path: baseline.path.clone(),
        advisory: true,
        matched_context: true,
        warning_count: warnings.len(),
        warnings,
    })
}

fn compare_bitnet_benchmark_path_regression(
    warnings: &mut Vec<RegressionWarning>,
    profile_id: &str,
    path_summary: &serde_json::Value,
    baseline_summary: &serde_json::Value,
) -> Result<()> {
    const DECODE_TOK_S_LOWER_PCT: f64 = 12.5;
    const THROUGHPUT_LOWER_PCT: f64 = 15.0;
    const LATENCY_HIGHER_PCT: f64 = 15.0;
    const LOAD_HIGHER_PCT: f64 = 20.0;
    const SAMPLING_HIGHER_PCT: f64 = 20.0;
    const PEAK_MEMORY_MB_HIGHER_PCT: f64 = 10.0;
    const MEMORY_DRIFT_MB_HIGHER_PCT: f64 = 15.0;

    for percentile in ["p50", "p90", "p99"] {
        for (metric, threshold) in [
            ("model_load_ms", LOAD_HIGHER_PCT),
            ("tokenizer_load_ms", LOAD_HIGHER_PCT),
            ("prompt_tokenize_ms", LATENCY_HIGHER_PCT),
            ("prefill_ms", LATENCY_HIGHER_PCT),
            ("time_to_first_token_ms", LATENCY_HIGHER_PCT),
            ("decode_total_ms", LATENCY_HIGHER_PCT),
            ("sampling_ms_per_token", SAMPLING_HIGHER_PCT),
            ("total_wall_ms", LATENCY_HIGHER_PCT),
        ] {
            compare_higher_is_worse(
                warnings,
                profile_id,
                &format!("timing.{metric}.{percentile}"),
                regression_metric(baseline_summary, &["timing", metric, percentile])?,
                regression_metric(path_summary, &["timing", metric, percentile])?,
                threshold,
            );
        }
        for (metric, threshold) in [
            ("input_tokens_per_second", THROUGHPUT_LOWER_PCT),
            ("output_tokens_per_second", THROUGHPUT_LOWER_PCT),
            ("decode_tokens_per_second", DECODE_TOK_S_LOWER_PCT),
        ] {
            compare_lower_is_worse(
                warnings,
                profile_id,
                &format!("throughput.{metric}.{percentile}"),
                regression_metric(baseline_summary, &["throughput", metric, percentile])?,
                regression_metric(path_summary, &["throughput", metric, percentile])?,
                threshold,
            );
        }
        for (metric, threshold) in [
            ("peak_memory_mb", PEAK_MEMORY_MB_HIGHER_PCT),
            ("memory_drift_mb", MEMORY_DRIFT_MB_HIGHER_PCT),
        ] {
            compare_higher_is_worse(
                warnings,
                profile_id,
                &format!("memory.{metric}.{percentile}"),
                regression_metric(baseline_summary, &["memory", metric, percentile])?,
                regression_metric(path_summary, &["memory", metric, percentile])?,
                threshold,
            );
        }
    }
    Ok(())
}

fn ensure_bitnet_eval_answer_corpus_regression_context_matches(
    path: &Path,
    receipt: &serde_json::Value,
    baseline_path: &Path,
    baseline: &serde_json::Value,
) -> Result<()> {
    for (label, observed, expected) in [
        ("artifact_kind", receipt["artifact_kind"].as_str(), baseline["artifact_kind"].as_str()),
        ("model_family", receipt["model_family"].as_str(), baseline["model_family"].as_str()),
        (
            "requested_backend",
            receipt["requested_backend"].as_str(),
            baseline["requested_backend"].as_str(),
        ),
        (
            "selected_backend",
            receipt["selected_backend"].as_str(),
            baseline["selected_backend"].as_str(),
        ),
        ("runtime_api", receipt["runtime_api"].as_str(), baseline["runtime_api"].as_str()),
        (
            "prompt_template",
            receipt["prompt_template"].as_str(),
            baseline["prompt_template"].as_str(),
        ),
        ("model.family", receipt["model"]["family"].as_str(), baseline["model"]["family"].as_str()),
        ("model.repo", receipt["model"]["repo"].as_str(), baseline["model"]["repo"].as_str()),
        ("model.file", receipt["model"]["file"].as_str(), baseline["model"]["file"].as_str()),
        ("model.path", receipt["model"]["path"].as_str(), baseline["model"]["path"].as_str()),
        (
            "model.revision",
            receipt["model"]["revision"].as_str(),
            baseline["model"]["revision"].as_str(),
        ),
        ("model.sha256", receipt["model"]["sha256"].as_str(), baseline["model"]["sha256"].as_str()),
        (
            "model.architecture",
            receipt["model"]["architecture"].as_str(),
            baseline["model"]["architecture"].as_str(),
        ),
        (
            "model.quant_format",
            receipt["model"]["quant_format"].as_str(),
            baseline["model"]["quant_format"].as_str(),
        ),
        (
            "model.loader_mode",
            receipt["model"]["loader_mode"].as_str(),
            baseline["model"]["loader_mode"].as_str(),
        ),
        (
            "model.answer_ready.state",
            receipt["model"]["answer_ready"]["state"].as_str(),
            baseline["model"]["answer_ready"]["state"].as_str(),
        ),
        (
            "tokenizer.source",
            receipt["tokenizer"]["source"].as_str(),
            baseline["tokenizer"]["source"].as_str(),
        ),
        (
            "tokenizer.path",
            receipt["tokenizer"]["path"].as_str(),
            baseline["tokenizer"]["path"].as_str(),
        ),
        (
            "tokenizer.authority.source",
            receipt["tokenizer"]["authority"]["source"].as_str(),
            baseline["tokenizer"]["authority"]["source"].as_str(),
        ),
        (
            "tokenizer.authority.repo",
            receipt["tokenizer"]["authority"]["repo"].as_str(),
            baseline["tokenizer"]["authority"]["repo"].as_str(),
        ),
        (
            "tokenizer.authority.revision",
            receipt["tokenizer"]["authority"]["revision"].as_str(),
            baseline["tokenizer"]["authority"]["revision"].as_str(),
        ),
        (
            "tokenizer.authority.sha256",
            receipt["tokenizer"]["authority"]["sha256"].as_str(),
            baseline["tokenizer"]["authority"]["sha256"].as_str(),
        ),
        (
            "tokenizer.authority.ggml_pre",
            receipt["tokenizer"]["authority"]["ggml_pre"].as_str(),
            baseline["tokenizer"]["authority"]["ggml_pre"].as_str(),
        ),
        ("corpus.name", receipt["corpus"]["name"].as_str(), baseline["corpus"]["name"].as_str()),
        ("corpus.path", receipt["corpus"]["path"].as_str(), baseline["corpus"]["path"].as_str()),
        (
            "reference_comparison.schema",
            receipt["reference_comparison"]["schema"].as_str(),
            baseline["reference_comparison"]["schema"].as_str(),
        ),
        (
            "reference_comparison.rust_runner.selected_backend",
            receipt["reference_comparison"]["rust_runner"]["selected_backend"].as_str(),
            baseline["reference_comparison"]["rust_runner"]["selected_backend"].as_str(),
        ),
        (
            "reference_comparison.rust_runner.runtime_api",
            receipt["reference_comparison"]["rust_runner"]["runtime_api"].as_str(),
            baseline["reference_comparison"]["rust_runner"]["runtime_api"].as_str(),
        ),
        (
            "reference_comparison.rust_runner.prompt_template",
            receipt["reference_comparison"]["rust_runner"]["prompt_template"].as_str(),
            baseline["reference_comparison"]["rust_runner"]["prompt_template"].as_str(),
        ),
        (
            "reference_comparison.rust_runner.tokenizer_authority.source",
            receipt["reference_comparison"]["rust_runner"]["tokenizer_authority"]["source"]
                .as_str(),
            baseline["reference_comparison"]["rust_runner"]["tokenizer_authority"]["source"]
                .as_str(),
        ),
        (
            "reference_comparison.rust_runner.tokenizer_authority.sha256",
            receipt["reference_comparison"]["rust_runner"]["tokenizer_authority"]["sha256"]
                .as_str(),
            baseline["reference_comparison"]["rust_runner"]["tokenizer_authority"]["sha256"]
                .as_str(),
        ),
        (
            "reference_comparison.rust_runner.tokenizer_authority.ggml_pre",
            receipt["reference_comparison"]["rust_runner"]["tokenizer_authority"]["ggml_pre"]
                .as_str(),
            baseline["reference_comparison"]["rust_runner"]["tokenizer_authority"]["ggml_pre"]
                .as_str(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch (observed={observed:?}, baseline={expected:?})",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for (label, observed, expected) in [
        (
            "corpus.case_count",
            receipt["corpus"]["case_count"].as_u64(),
            baseline["corpus"]["case_count"].as_u64(),
        ),
        (
            "corpus.selected_case_count",
            receipt["corpus"]["selected_case_count"].as_u64(),
            baseline["corpus"]["selected_case_count"].as_u64(),
        ),
        (
            "quality_summary.total",
            receipt["quality_summary"]["total"].as_u64(),
            baseline["quality_summary"]["total"].as_u64(),
        ),
        (
            "scoring_summary.total",
            receipt["scoring_summary"]["total"].as_u64(),
            baseline["scoring_summary"]["total"].as_u64(),
        ),
        (
            "reference_comparison.summary.total",
            receipt["reference_comparison"]["summary"]["total"].as_u64(),
            baseline["reference_comparison"]["summary"]["total"].as_u64(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch (observed={observed:?}, baseline={expected:?})",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for (label, observed, expected) in [
        ("fallback_used", receipt["fallback_used"].as_bool(), baseline["fallback_used"].as_bool()),
        (
            "model.answer_ready_artifact_available",
            receipt["model"]["answer_ready_artifact_available"].as_bool(),
            baseline["model"]["answer_ready_artifact_available"].as_bool(),
        ),
        (
            "tokenizer.strict",
            receipt["tokenizer"]["strict"].as_bool(),
            baseline["tokenizer"]["strict"].as_bool(),
        ),
        (
            "reference_comparison.enabled",
            receipt["reference_comparison"]["enabled"].as_bool(),
            baseline["reference_comparison"]["enabled"].as_bool(),
        ),
        (
            "reference_comparison.reference_runner_required",
            receipt["reference_comparison"]["reference_runner_required"].as_bool(),
            baseline["reference_comparison"]["reference_runner_required"].as_bool(),
        ),
        (
            "reference_comparison.rust_runner.fallback_used",
            receipt["reference_comparison"]["rust_runner"]["fallback_used"].as_bool(),
            baseline["reference_comparison"]["rust_runner"]["fallback_used"].as_bool(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for (label, observed, expected) in [
        (
            "corpus.selected_case_ids",
            &receipt["corpus"]["selected_case_ids"],
            &baseline["corpus"]["selected_case_ids"],
        ),
        (
            "scoring_summary.kinds",
            &receipt["scoring_summary"]["kinds"],
            &baseline["scoring_summary"]["kinds"],
        ),
    ] {
        if observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for flag in [
        "dense_slm_evidence_used",
        "chat_enabled",
        "serve_enabled",
        "performance_claimed",
        "full_metal_inference_claimed",
        "qk256_apple_claimed",
        "neural_engine_claimed",
        "mpsgraph_claimed",
        "broad_apple_silicon_claimed",
        "runtime_accuracy_claimed",
    ] {
        if receipt["reference_comparison"]["claim_boundary"][flag].as_bool() != Some(false)
            || baseline["reference_comparison"]["claim_boundary"][flag].as_bool() != Some(false)
        {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: reference_comparison.claim_boundary.{flag} must remain false",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for flag in [
        "full_metal_inference_claimed",
        "neural_engine_claimed",
        "qk256_apple_claimed",
        "broad_performance_claimed",
    ] {
        if receipt["claim_boundary"][flag].as_bool() != Some(false)
            || baseline["claim_boundary"][flag].as_bool() != Some(false)
        {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: claim_boundary.{flag} must remain false",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for flag in ["local_answer_path", "answer_ready_artifact_available"] {
        if receipt["claim_boundary"][flag].as_bool() != Some(true)
            || baseline["claim_boundary"][flag].as_bool() != Some(true)
        {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: claim_boundary.{flag} must remain true",
                path.display(),
                baseline_path.display()
            );
        }
    }
    ensure_bitnet_eval_task_family_context_matches(path, receipt, baseline_path, baseline)?;
    Ok(())
}

fn ensure_bitnet_eval_task_family_context_matches(
    path: &Path,
    receipt: &serde_json::Value,
    baseline_path: &Path,
    baseline: &serde_json::Value,
) -> Result<()> {
    let observed_families = receipt["task_family_summary"].as_object().ok_or_else(|| {
        anyhow!("{} BitNet eval task_family_summary must be an object", path.display())
    })?;
    let baseline_families = baseline["task_family_summary"].as_object().ok_or_else(|| {
        anyhow!(
            "regression baseline {} BitNet eval task_family_summary must be an object",
            baseline_path.display()
        )
    })?;
    for (family, baseline_family) in baseline_families {
        let observed_family = observed_families.get(family).ok_or_else(|| {
            anyhow!(
                "{} cannot be compared to baseline {}: task_family_summary.{family} is missing",
                path.display(),
                baseline_path.display()
            )
        })?;
        for field in ["total", "scoring.total"] {
            let (observed, expected) = if field == "total" {
                (observed_family["total"].as_u64(), baseline_family["total"].as_u64())
            } else {
                (
                    observed_family["scoring"]["total"].as_u64(),
                    baseline_family["scoring"]["total"].as_u64(),
                )
            };
            if observed.is_none() || expected.is_none() || observed != expected {
                anyhow::bail!(
                    "{} cannot be compared to baseline {}: task_family_summary.{family}.{field} mismatch",
                    path.display(),
                    baseline_path.display()
                );
            }
        }
        if observed_family["scoring"]["enabled"].as_bool()
            != baseline_family["scoring"]["enabled"].as_bool()
        {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: task_family_summary.{family}.scoring.enabled mismatch",
                path.display(),
                baseline_path.display()
            );
        }
        if observed_family["scoring"]["kinds"] != baseline_family["scoring"]["kinds"] {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: task_family_summary.{family}.scoring.kinds mismatch",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for family in observed_families.keys() {
        if !baseline_families.contains_key(family) {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: task_family_summary.{family} is not present in baseline",
                path.display(),
                baseline_path.display()
            );
        }
    }
    Ok(())
}

fn compare_bitnet_eval_task_family_regression(
    warnings: &mut Vec<RegressionWarning>,
    path: &Path,
    receipt: &serde_json::Value,
    baseline_path: &Path,
    baseline: &serde_json::Value,
    threshold_percent: f64,
) -> Result<()> {
    let observed_families = receipt["task_family_summary"].as_object().ok_or_else(|| {
        anyhow!("{} BitNet eval task_family_summary must be an object", path.display())
    })?;
    let baseline_families = baseline["task_family_summary"].as_object().ok_or_else(|| {
        anyhow!(
            "regression baseline {} BitNet eval task_family_summary must be an object",
            baseline_path.display()
        )
    })?;
    for (family, baseline_family) in baseline_families {
        let observed_family = observed_families.get(family).ok_or_else(|| {
            anyhow!(
                "{} cannot be compared to baseline {}: task_family_summary.{family} is missing",
                path.display(),
                baseline_path.display()
            )
        })?;
        for field in ["passed"] {
            compare_lower_is_worse(
                warnings,
                &format!("bitnet_task_family:{family}"),
                &format!("task_family_summary.{family}.{field}"),
                metric_value(&baseline_family[field]).ok_or_else(|| {
                    anyhow!(
                        "regression baseline {} is missing numeric regression metric task_family_summary.{family}.{field}",
                        baseline_path.display()
                    )
                })?,
                metric_value(&observed_family[field]).ok_or_else(|| {
                    anyhow!(
                        "{} is missing numeric regression metric task_family_summary.{family}.{field}",
                        path.display()
                    )
                })?,
                threshold_percent,
            );
            compare_lower_is_worse(
                warnings,
                &format!("bitnet_task_family:{family}"),
                &format!("task_family_summary.{family}.scoring.{field}"),
                metric_value(&baseline_family["scoring"][field]).ok_or_else(|| {
                    anyhow!(
                        "regression baseline {} is missing numeric regression metric task_family_summary.{family}.scoring.{field}",
                        baseline_path.display()
                    )
                })?,
                metric_value(&observed_family["scoring"][field]).ok_or_else(|| {
                    anyhow!(
                        "{} is missing numeric regression metric task_family_summary.{family}.scoring.{field}",
                        path.display()
                    )
                })?,
                threshold_percent,
            );
        }
        for field in ["failed", "timeout", "not_run"] {
            compare_higher_is_worse(
                warnings,
                &format!("bitnet_task_family:{family}"),
                &format!("task_family_summary.{family}.{field}"),
                metric_value(&baseline_family[field]).ok_or_else(|| {
                    anyhow!(
                        "regression baseline {} is missing numeric regression metric task_family_summary.{family}.{field}",
                        baseline_path.display()
                    )
                })?,
                metric_value(&observed_family[field]).ok_or_else(|| {
                    anyhow!(
                        "{} is missing numeric regression metric task_family_summary.{family}.{field}",
                        path.display()
                    )
                })?,
                threshold_percent,
            );
        }
        for field in ["failed", "not_run"] {
            compare_higher_is_worse(
                warnings,
                &format!("bitnet_task_family:{family}"),
                &format!("task_family_summary.{family}.scoring.{field}"),
                metric_value(&baseline_family["scoring"][field]).ok_or_else(|| {
                    anyhow!(
                        "regression baseline {} is missing numeric regression metric task_family_summary.{family}.scoring.{field}",
                        baseline_path.display()
                    )
                })?,
                metric_value(&observed_family["scoring"][field]).ok_or_else(|| {
                    anyhow!(
                        "{} is missing numeric regression metric task_family_summary.{family}.scoring.{field}",
                        path.display()
                    )
                })?,
                threshold_percent,
            );
        }
    }
    Ok(())
}

fn compare_bitnet_eval_reference_regression(
    warnings: &mut Vec<RegressionWarning>,
    path: &Path,
    receipt: &serde_json::Value,
    baseline_path: &Path,
    baseline: &serde_json::Value,
    threshold_percent: f64,
) -> Result<()> {
    for field in ["comparable_cases", "matched", "text_matches", "generated_token_id_matches"] {
        compare_lower_is_worse(
            warnings,
            "bitnet_eval:reference_comparison",
            &format!("reference_comparison.summary.{field}"),
            regression_metric(baseline, &["reference_comparison", "summary", field])?,
            regression_metric(receipt, &["reference_comparison", "summary", field])?,
            threshold_percent,
        );
    }
    for field in ["mismatched", "not_run", "partially_compared", "reference_not_supplied"] {
        compare_higher_is_worse(
            warnings,
            "bitnet_eval:reference_comparison",
            &format!("reference_comparison.summary.{field}"),
            regression_metric(baseline, &["reference_comparison", "summary", field]).with_context(
                || {
                    format!(
                        "regression baseline {} is missing reference_comparison.summary.{field}",
                        baseline_path.display()
                    )
                },
            )?,
            regression_metric(receipt, &["reference_comparison", "summary", field]).with_context(
                || format!("{} is missing reference_comparison.summary.{field}", path.display()),
            )?,
            threshold_percent,
        );
    }
    Ok(())
}

fn ensure_bitnet_benchmark_v1_regression_context_matches(
    path: &Path,
    receipt: &serde_json::Value,
    baseline_path: &Path,
    baseline: &serde_json::Value,
) -> Result<()> {
    for (label, observed, expected) in [
        ("artifact_kind", receipt["artifact_kind"].as_str(), baseline["artifact_kind"].as_str()),
        ("benchmark_set", receipt["benchmark_set"].as_str(), baseline["benchmark_set"].as_str()),
        (
            "requested_backend",
            receipt["requested_backend"].as_str(),
            baseline["requested_backend"].as_str(),
        ),
        (
            "selected_backend",
            receipt["selected_backend"].as_str(),
            baseline["selected_backend"].as_str(),
        ),
        ("runtime_api", receipt["runtime_api"].as_str(), baseline["runtime_api"].as_str()),
        (
            "prompt_template",
            receipt["prompt_template"].as_str(),
            baseline["prompt_template"].as_str(),
        ),
        ("model.id", receipt["model"]["id"].as_str(), baseline["model"]["id"].as_str()),
        ("model.family", receipt["model"]["family"].as_str(), baseline["model"]["family"].as_str()),
        ("model.repo", receipt["model"]["repo"].as_str(), baseline["model"]["repo"].as_str()),
        ("model.file", receipt["model"]["file"].as_str(), baseline["model"]["file"].as_str()),
        ("model.path", receipt["model"]["path"].as_str(), baseline["model"]["path"].as_str()),
        ("model.sha256", receipt["model"]["sha256"].as_str(), baseline["model"]["sha256"].as_str()),
        (
            "model.architecture",
            receipt["model"]["architecture"].as_str(),
            baseline["model"]["architecture"].as_str(),
        ),
        (
            "model.quantization",
            receipt["model"]["quantization"].as_str(),
            baseline["model"]["quantization"].as_str(),
        ),
        (
            "model.answer_ready.state",
            receipt["model"]["answer_ready"]["state"].as_str(),
            baseline["model"]["answer_ready"]["state"].as_str(),
        ),
        (
            "tokenizer.source",
            receipt["tokenizer"]["source"].as_str(),
            baseline["tokenizer"]["source"].as_str(),
        ),
        (
            "tokenizer.path",
            receipt["tokenizer"]["path"].as_str(),
            baseline["tokenizer"]["path"].as_str(),
        ),
        (
            "tokenizer.sha256",
            receipt["tokenizer"]["sha256"].as_str(),
            baseline["tokenizer"]["sha256"].as_str(),
        ),
        (
            "tokenizer.authority",
            receipt["tokenizer"]["authority"].as_str(),
            baseline["tokenizer"]["authority"].as_str(),
        ),
        (
            "tokenizer.pretokenizer_authority",
            receipt["tokenizer"]["pretokenizer_authority"].as_str(),
            baseline["tokenizer"]["pretokenizer_authority"].as_str(),
        ),
        (
            "build.profile",
            receipt["build"]["profile"].as_str(),
            baseline["build"]["profile"].as_str(),
        ),
        (
            "timeout_boundary.status",
            receipt["timeout_boundary"]["status"].as_str(),
            baseline["timeout_boundary"]["status"].as_str(),
        ),
        (
            "memory.source",
            receipt["memory"]["source"].as_str(),
            baseline["memory"]["source"].as_str(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch (observed={observed:?}, baseline={expected:?})",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for (label, observed, expected) in [
        ("prompt_count", receipt["prompt_count"].as_u64(), baseline["prompt_count"].as_u64()),
        (
            "generated_tokens",
            receipt["generated_tokens"].as_u64(),
            baseline["generated_tokens"].as_u64(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch (observed={observed:?}, baseline={expected:?})",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for (label, observed, expected) in [
        ("fallback_used", receipt["fallback_used"].as_bool(), baseline["fallback_used"].as_bool()),
        (
            "build.release_mode",
            receipt["build"]["release_mode"].as_bool(),
            baseline["build"]["release_mode"].as_bool(),
        ),
        (
            "model.answer_ready_artifact_available",
            receipt["model"]["answer_ready_artifact_available"].as_bool(),
            baseline["model"]["answer_ready_artifact_available"].as_bool(),
        ),
        (
            "tokenizer.strict",
            receipt["tokenizer"]["strict"].as_bool(),
            baseline["tokenizer"]["strict"].as_bool(),
        ),
        (
            "timeout_boundary.enforced",
            receipt["timeout_boundary"]["enforced"].as_bool(),
            baseline["timeout_boundary"]["enforced"].as_bool(),
        ),
        (
            "timeout_boundary.reached",
            receipt["timeout_boundary"]["reached"].as_bool(),
            baseline["timeout_boundary"]["reached"].as_bool(),
        ),
        (
            "evidence.generated_text_recorded",
            receipt["evidence"]["generated_text_recorded"].as_bool(),
            baseline["evidence"]["generated_text_recorded"].as_bool(),
        ),
        (
            "evidence.generated_token_ids_recorded",
            receipt["evidence"]["generated_token_ids_recorded"].as_bool(),
            baseline["evidence"]["generated_token_ids_recorded"].as_bool(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch",
                path.display(),
                baseline_path.display()
            );
        }
    }
    if receipt["evidence"]["operator_commands"] != baseline["evidence"]["operator_commands"] {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: evidence.operator_commands mismatch",
            path.display(),
            baseline_path.display()
        );
    }
    for flag in
        ["bitnet_benchmark", "one_shot_mac_ask", "fixed_warm_session", "accepted_i2s_artifact_only"]
    {
        if receipt["mac_claim_boundary"][flag].as_bool() != Some(true)
            || baseline["mac_claim_boundary"][flag].as_bool() != Some(true)
        {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: mac_claim_boundary.{flag} must remain true",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for flag in [
        "dense_slm_evidence_used",
        "chat_enabled",
        "serve_enabled",
        "bitnet_quality_claimed",
        "broad_model_quality_claim",
        "broad_performance_claim",
        "speedup_claim",
        "full_metal_inference_claimed",
        "mpsgraph_inference_claimed",
        "neural_engine_execution_claimed",
        "qk256_apple_claimed",
        "macbook_evidence",
        "broad_apple_silicon_claimed",
    ] {
        if receipt["mac_claim_boundary"][flag].as_bool() != Some(false)
            || baseline["mac_claim_boundary"][flag].as_bool() != Some(false)
        {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: mac_claim_boundary.{flag} must remain false",
                path.display(),
                baseline_path.display()
            );
        }
    }
    ensure_bitnet_benchmark_path_context_matches(
        path,
        "one_shot",
        &receipt["paths"]["one_shot"],
        baseline_path,
        &baseline["paths"]["one_shot"],
    )?;
    ensure_bitnet_benchmark_path_context_matches(
        path,
        "fixed_warm",
        &receipt["paths"]["fixed_warm"],
        baseline_path,
        &baseline["paths"]["fixed_warm"],
    )?;
    Ok(())
}

fn ensure_bitnet_benchmark_path_context_matches(
    path: &Path,
    path_key: &str,
    summary: &serde_json::Value,
    baseline_path: &Path,
    baseline_summary: &serde_json::Value,
) -> Result<()> {
    for (label, observed, expected) in [
        ("path_id", summary["path_id"].as_str(), baseline_summary["path_id"].as_str()),
        (
            "operator_command",
            summary["operator_command"].as_str(),
            baseline_summary["operator_command"].as_str(),
        ),
        (
            "timeout_boundary.status",
            summary["timeout_boundary"]["status"].as_str(),
            baseline_summary["timeout_boundary"]["status"].as_str(),
        ),
        (
            "memory.source",
            summary["memory"]["source"].as_str(),
            baseline_summary["memory"]["source"].as_str(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: paths.{path_key}.{label} mismatch (observed={observed:?}, baseline={expected:?})",
                path.display(),
                baseline_path.display()
            );
        }
    }
    if summary["reuse_scope"].is_string() || baseline_summary["reuse_scope"].is_string() {
        let observed = summary["reuse_scope"].as_str();
        let expected = baseline_summary["reuse_scope"].as_str();
        if observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: paths.{path_key}.reuse_scope mismatch",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for (label, observed, expected) in [
        (
            "prompt_count",
            summary["prompt_count"].as_u64(),
            baseline_summary["prompt_count"].as_u64(),
        ),
        (
            "generated_tokens",
            summary["generated_tokens"].as_u64(),
            baseline_summary["generated_tokens"].as_u64(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: paths.{path_key}.{label} mismatch (observed={observed:?}, baseline={expected:?})",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for (label, observed, expected) in [
        (
            "model_loaded_once",
            summary["model_loaded_once"].as_bool(),
            baseline_summary["model_loaded_once"].as_bool(),
        ),
        (
            "tokenizer_loaded_once",
            summary["tokenizer_loaded_once"].as_bool(),
            baseline_summary["tokenizer_loaded_once"].as_bool(),
        ),
        (
            "quality_passed",
            summary["quality_passed"].as_bool(),
            baseline_summary["quality_passed"].as_bool(),
        ),
        (
            "timeout_boundary.enforced",
            summary["timeout_boundary"]["enforced"].as_bool(),
            baseline_summary["timeout_boundary"]["enforced"].as_bool(),
        ),
        (
            "timeout_boundary.reached",
            summary["timeout_boundary"]["reached"].as_bool(),
            baseline_summary["timeout_boundary"]["reached"].as_bool(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: paths.{path_key}.{label} mismatch",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for flag in ["chat_enabled", "serve_enabled", "broad_performance_claim", "speedup_claim"] {
        if summary["claim_boundary"][flag].as_bool() != Some(false)
            || baseline_summary["claim_boundary"][flag].as_bool() != Some(false)
        {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: paths.{path_key}.claim_boundary.{flag} must remain false",
                path.display(),
                baseline_path.display()
            );
        }
    }
    Ok(())
}

fn ensure_regression_context_matches(
    path: &Path,
    receipt: &serde_json::Value,
    baseline_path: &Path,
    baseline: &serde_json::Value,
) -> Result<()> {
    for (label, observed, expected) in [
        ("artifact_kind", receipt["artifact_kind"].as_str(), baseline["artifact_kind"].as_str()),
        ("profile_set", receipt["profile_set"].as_str(), baseline["profile_set"].as_str()),
        (
            "requested_backend",
            receipt["requested_backend"].as_str(),
            baseline["requested_backend"].as_str(),
        ),
        (
            "selected_backend",
            receipt["selected_backend"].as_str(),
            baseline["selected_backend"].as_str(),
        ),
        ("runtime_api", receipt["runtime_api"].as_str(), baseline["runtime_api"].as_str()),
        (
            "model_cache.id",
            receipt["model_cache"]["id"].as_str(),
            baseline["model_cache"]["id"].as_str(),
        ),
        (
            "model_cache.sha256",
            receipt["model_cache"]["sha256"].as_str(),
            baseline["model_cache"]["sha256"].as_str(),
        ),
        (
            "model_cache.quantization",
            receipt["model_cache"]["quantization"].as_str(),
            baseline["model_cache"]["quantization"].as_str(),
        ),
        (
            "model_cache.tokenizer_model",
            receipt["model_cache"]["tokenizer_model"].as_str(),
            baseline["model_cache"]["tokenizer_model"].as_str(),
        ),
        (
            "model_cache.tokenizer_pre",
            receipt["model_cache"]["tokenizer_pre"].as_str(),
            baseline["model_cache"]["tokenizer_pre"].as_str(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch (observed={observed:?}, baseline={expected:?})",
                path.display(),
                baseline_path.display()
            );
        }
    }
    if receipt["fallback_used"].as_bool() != baseline["fallback_used"].as_bool() {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: fallback_used mismatch",
            path.display(),
            baseline_path.display()
        );
    }
    Ok(())
}

fn compare_slm_eval_task_family_regression(
    warnings: &mut Vec<RegressionWarning>,
    path: &Path,
    receipt: &serde_json::Value,
    baseline_path: &Path,
    baseline: &serde_json::Value,
    threshold_percent: f64,
) -> Result<()> {
    let baseline_families = &baseline["task_families"];
    let observed_families = &receipt["task_families"];
    if baseline_families.is_null() && observed_families.is_null() {
        return Ok(());
    }
    let baseline_families = baseline_families.as_object().ok_or_else(|| {
        anyhow!(
            "regression baseline {} task_families must be an object when present",
            baseline_path.display()
        )
    })?;
    let observed_families = observed_families.as_object().ok_or_else(|| {
        anyhow!("{} task_families must be an object when present", path.display())
    })?;
    for family in baseline_families.keys() {
        let observed_family = observed_families.get(family).ok_or_else(|| {
            anyhow!(
                "{} cannot be compared to baseline {}: task_families.{family} is missing",
                path.display(),
                baseline_path.display()
            )
        })?;
        let baseline_family = &baseline_families[family];
        for field in ["cases_total", "cases_scored"] {
            if baseline_family[field].as_u64() != observed_family[field].as_u64() {
                anyhow::bail!(
                    "{} cannot be compared to baseline {}: task_families.{family}.{field} mismatch",
                    path.display(),
                    baseline_path.display()
                );
            }
        }
        if baseline_family["scoring_kinds"] != observed_family["scoring_kinds"] {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: task_families.{family}.scoring_kinds mismatch",
                path.display(),
                baseline_path.display()
            );
        }
        for field in
            ["cases_passed", "pass_rate", "quality_gate_cases_passed", "quality_gate_pass_rate"]
        {
            compare_lower_is_worse(
                warnings,
                &format!("task_family:{family}"),
                &format!("task_families.{family}.{field}"),
                metric_value(&baseline_family[field]).ok_or_else(|| {
                    anyhow!(
                        "regression baseline {} is missing numeric regression metric task_families.{family}.{field}",
                        baseline_path.display()
                    )
                })?,
                metric_value(&observed_family[field]).ok_or_else(|| {
                    anyhow!(
                        "{} is missing numeric regression metric task_families.{family}.{field}",
                        path.display()
                    )
                })?,
                threshold_percent,
            );
        }
    }
    for family in observed_families.keys() {
        if !baseline_families.contains_key(family) {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: task_families.{family} is not present in baseline",
                path.display(),
                baseline_path.display()
            );
        }
    }
    Ok(())
}

fn compare_slm_eval_scoring_summary_regression(
    warnings: &mut Vec<RegressionWarning>,
    path: &Path,
    receipt: &serde_json::Value,
    baseline_path: &Path,
    baseline: &serde_json::Value,
    threshold_percent: f64,
) -> Result<()> {
    let baseline_summary = &baseline["scoring_summary"];
    let observed_summary = &receipt["scoring_summary"];
    if baseline_summary.is_null() && observed_summary.is_null() {
        return Ok(());
    }
    let baseline_summary = baseline_summary.as_object().ok_or_else(|| {
        anyhow!(
            "regression baseline {} scoring_summary must be an object when present",
            baseline_path.display()
        )
    })?;
    let observed_summary = observed_summary.as_object().ok_or_else(|| {
        anyhow!("{} scoring_summary must be an object when present", path.display())
    })?;
    if baseline_summary.get("enabled").and_then(|value| value.as_bool())
        != observed_summary.get("enabled").and_then(|value| value.as_bool())
    {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: scoring_summary.enabled mismatch",
            path.display(),
            baseline_path.display()
        );
    }
    if baseline_summary.get("total").and_then(|value| value.as_u64())
        != observed_summary.get("total").and_then(|value| value.as_u64())
    {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: scoring_summary.total mismatch",
            path.display(),
            baseline_path.display()
        );
    }
    if baseline_summary.get("kinds") != observed_summary.get("kinds") {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: scoring_summary.kinds mismatch",
            path.display(),
            baseline_path.display()
        );
    }
    for field in ["passed"] {
        compare_lower_is_worse(
            warnings,
            "seeded_corpus",
            &format!("scoring_summary.{field}"),
            metric_value(&baseline["scoring_summary"][field]).ok_or_else(|| {
                anyhow!(
                    "regression baseline {} is missing numeric regression metric scoring_summary.{field}",
                    baseline_path.display()
                )
            })?,
            metric_value(&receipt["scoring_summary"][field]).ok_or_else(|| {
                anyhow!("{} is missing numeric regression metric scoring_summary.{field}", path.display())
            })?,
            threshold_percent,
        );
    }
    for field in ["failed", "not_run"] {
        compare_higher_is_worse(
            warnings,
            "seeded_corpus",
            &format!("scoring_summary.{field}"),
            metric_value(&baseline["scoring_summary"][field]).ok_or_else(|| {
                anyhow!(
                    "regression baseline {} is missing numeric regression metric scoring_summary.{field}",
                    baseline_path.display()
                )
            })?,
            metric_value(&receipt["scoring_summary"][field]).ok_or_else(|| {
                anyhow!("{} is missing numeric regression metric scoring_summary.{field}", path.display())
            })?,
            threshold_percent,
        );
    }
    Ok(())
}

fn ensure_slm_eval_summary_regression_context_matches(
    path: &Path,
    receipt: &serde_json::Value,
    baseline_path: &Path,
    baseline: &serde_json::Value,
) -> Result<()> {
    for (label, observed, expected) in [
        ("artifact_kind", receipt["artifact_kind"].as_str(), baseline["artifact_kind"].as_str()),
        ("machine_id", receipt["machine_id"].as_str(), baseline["machine_id"].as_str()),
        ("model_id", receipt["model_id"].as_str(), baseline["model_id"].as_str()),
        (
            "requested_backend",
            receipt["requested_backend"].as_str(),
            baseline["requested_backend"].as_str(),
        ),
        (
            "selected_backend",
            receipt["selected_backend"].as_str(),
            baseline["selected_backend"].as_str(),
        ),
        ("runtime_api", receipt["runtime_api"].as_str(), baseline["runtime_api"].as_str()),
        ("model.repo", receipt["model"]["repo"].as_str(), baseline["model"]["repo"].as_str()),
        ("model.file", receipt["model"]["file"].as_str(), baseline["model"]["file"].as_str()),
        ("model.sha256", receipt["model"]["sha256"].as_str(), baseline["model"]["sha256"].as_str()),
        ("model.family", receipt["model"]["family"].as_str(), baseline["model"]["family"].as_str()),
        (
            "model.architecture",
            receipt["model"]["architecture"].as_str(),
            baseline["model"]["architecture"].as_str(),
        ),
        (
            "model.quantization",
            receipt["model"]["quantization"].as_str(),
            baseline["model"]["quantization"].as_str(),
        ),
        (
            "tokenizer.source",
            receipt["tokenizer"]["source"].as_str(),
            baseline["tokenizer"]["source"].as_str(),
        ),
        (
            "tokenizer.authority",
            receipt["tokenizer"]["authority"].as_str(),
            baseline["tokenizer"]["authority"].as_str(),
        ),
        (
            "tokenizer.pretokenizer_authority",
            receipt["tokenizer"]["pretokenizer_authority"].as_str(),
            baseline["tokenizer"]["pretokenizer_authority"].as_str(),
        ),
        (
            "prompt_template",
            receipt["prompt_template"].as_str(),
            baseline["prompt_template"].as_str(),
        ),
        ("corpus.name", receipt["corpus"]["name"].as_str(), baseline["corpus"]["name"].as_str()),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch (observed={observed:?}, baseline={expected:?})",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for (label, observed, expected) in [
        ("corpus.seed", receipt["corpus"]["seed"].as_u64(), baseline["corpus"]["seed"].as_u64()),
        (
            "corpus.case_count",
            receipt["corpus"]["case_count"].as_u64(),
            baseline["corpus"]["case_count"].as_u64(),
        ),
        (
            "accuracy.cases_total",
            receipt["accuracy"]["cases_total"].as_u64(),
            baseline["accuracy"]["cases_total"].as_u64(),
        ),
        (
            "accuracy.cases_scored",
            receipt["accuracy"]["cases_scored"].as_u64(),
            baseline["accuracy"]["cases_scored"].as_u64(),
        ),
        (
            "stability.resident_prompts",
            receipt["stability"]["resident_prompts"].as_u64(),
            baseline["stability"]["resident_prompts"].as_u64(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch (observed={observed:?}, baseline={expected:?})",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for (label, observed, expected) in [
        ("fallback_used", receipt["fallback_used"].as_bool(), baseline["fallback_used"].as_bool()),
        (
            "tokenizer.strict",
            receipt["tokenizer"]["strict"].as_bool(),
            baseline["tokenizer"]["strict"].as_bool(),
        ),
        (
            "stability.quality_passed",
            receipt["stability"]["quality_passed"].as_bool(),
            baseline["stability"]["quality_passed"].as_bool(),
        ),
        (
            "claim_boundary.dense_slm_only",
            receipt["claim_boundary"]["dense_slm_only"].as_bool(),
            baseline["claim_boundary"]["dense_slm_only"].as_bool(),
        ),
        (
            "claim_boundary.bounded_seeded_corpus_only",
            receipt["claim_boundary"]["bounded_seeded_corpus_only"].as_bool(),
            baseline["claim_boundary"]["bounded_seeded_corpus_only"].as_bool(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch (observed={observed:?}, baseline={expected:?})",
                path.display(),
                baseline_path.display()
            );
        }
    }
    if receipt["stability"]["quality_passed"].as_bool() != Some(true)
        || baseline["stability"]["quality_passed"].as_bool() != Some(true)
    {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: both SLM eval summaries must pass resident stability quality",
            path.display(),
            baseline_path.display()
        );
    }
    for flag in [
        "broad_model_quality_claim",
        "broad_performance_claim",
        "bitnet_evidence",
        "bitnet_quality_claimed",
        "full_metal_inference_claimed",
        "qk256_apple_claimed",
        "neural_engine_claimed",
        "neural_engine_execution_claimed",
        "mpsgraph_inference_claimed",
        "macbook_evidence",
        "speedup_claim",
    ] {
        if receipt["claim_boundary"][flag].as_bool() != Some(false)
            || baseline["claim_boundary"][flag].as_bool() != Some(false)
        {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: claim_boundary.{flag} must remain false",
                path.display(),
                baseline_path.display()
            );
        }
    }
    Ok(())
}

fn ensure_slm_benchmark_v2_regression_context_matches(
    path: &Path,
    receipt: &serde_json::Value,
    baseline_path: &Path,
    baseline: &serde_json::Value,
) -> Result<()> {
    for (label, observed, expected) in [
        ("artifact_kind", receipt["artifact_kind"].as_str(), baseline["artifact_kind"].as_str()),
        ("profile_set", receipt["profile_set"].as_str(), baseline["profile_set"].as_str()),
        (
            "requested_backend",
            receipt["requested_backend"].as_str(),
            baseline["requested_backend"].as_str(),
        ),
        (
            "selected_backend",
            receipt["selected_backend"].as_str(),
            baseline["selected_backend"].as_str(),
        ),
        ("runtime_api", receipt["runtime_api"].as_str(), baseline["runtime_api"].as_str()),
        (
            "model_cache.id",
            receipt["model_cache"]["id"].as_str(),
            baseline["model_cache"]["id"].as_str(),
        ),
        (
            "model_cache.sha256",
            receipt["model_cache"]["sha256"].as_str(),
            baseline["model_cache"]["sha256"].as_str(),
        ),
        (
            "model_cache.architecture",
            receipt["model_cache"]["architecture"].as_str(),
            baseline["model_cache"]["architecture"].as_str(),
        ),
        (
            "model_cache.quantization",
            receipt["model_cache"]["quantization"].as_str(),
            baseline["model_cache"]["quantization"].as_str(),
        ),
        (
            "model_cache.tokenizer_pre",
            receipt["model_cache"]["tokenizer_pre"].as_str(),
            baseline["model_cache"]["tokenizer_pre"].as_str(),
        ),
        (
            "evidence.operator_command",
            receipt["evidence"]["operator_command"].as_str(),
            baseline["evidence"]["operator_command"].as_str(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch (observed={observed:?}, baseline={expected:?})",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for (label, observed, expected) in [
        ("fallback_used", receipt["fallback_used"].as_bool(), baseline["fallback_used"].as_bool()),
        (
            "build.release_mode",
            receipt["build"]["release_mode"].as_bool(),
            baseline["build"]["release_mode"].as_bool(),
        ),
        (
            "evidence.generated_text_recorded",
            receipt["evidence"]["generated_text_recorded"].as_bool(),
            baseline["evidence"]["generated_text_recorded"].as_bool(),
        ),
        (
            "evidence.generated_token_ids_recorded",
            receipt["evidence"]["generated_token_ids_recorded"].as_bool(),
            baseline["evidence"]["generated_token_ids_recorded"].as_bool(),
        ),
        (
            "mac_claim_boundary.dense_slm_only",
            receipt["mac_claim_boundary"]["dense_slm_only"].as_bool(),
            baseline["mac_claim_boundary"]["dense_slm_only"].as_bool(),
        ),
        (
            "mac_claim_boundary.bounded_benchmark_profiles_only",
            receipt["mac_claim_boundary"]["bounded_benchmark_profiles_only"].as_bool(),
            baseline["mac_claim_boundary"]["bounded_benchmark_profiles_only"].as_bool(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for flag in [
        "broad_model_quality_claim",
        "broad_performance_claim",
        "speedup_claim",
        "bitnet_quality_claimed",
        "full_metal_inference_claimed",
        "mpsgraph_inference_claimed",
        "neural_engine_execution_claimed",
        "qk256_apple_claimed",
        "macbook_evidence",
    ] {
        if receipt["mac_claim_boundary"][flag].as_bool() != Some(false)
            || baseline["mac_claim_boundary"][flag].as_bool() != Some(false)
        {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: mac_claim_boundary.{flag} must remain false",
                path.display(),
                baseline_path.display()
            );
        }
    }
    if receipt["profiles_required"] != baseline["profiles_required"] {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: profiles_required mismatch",
            path.display(),
            baseline_path.display()
        );
    }

    let profiles = receipt["profiles"].as_array().ok_or_else(|| {
        anyhow!("{} SLM benchmark v2 summary is missing profiles", path.display())
    })?;
    for profile in profiles {
        let Some(profile_id) = profile["profile_id"].as_str() else {
            anyhow::bail!("{} SLM benchmark v2 profile is missing profile_id", path.display());
        };
        let baseline_profile = find_profile(baseline, profile_id).ok_or_else(|| {
            anyhow!(
                "regression baseline {} is missing SLM benchmark v2 profile {profile_id}",
                baseline_path.display()
            )
        })?;
        for (label, observed, expected) in [
            ("scenario", profile["scenario"].as_str(), baseline_profile["scenario"].as_str()),
            (
                "reuse_scope",
                profile["reuse_scope"].as_str(),
                baseline_profile["reuse_scope"].as_str(),
            ),
        ] {
            if observed.is_none() || expected.is_none() || observed != expected {
                anyhow::bail!(
                    "{} cannot be compared to baseline {}: profile {profile_id} {label} mismatch",
                    path.display(),
                    baseline_path.display()
                );
            }
        }
        for (label, observed, expected) in [
            (
                "prompt_count",
                profile["prompt_count"].as_u64(),
                baseline_profile["prompt_count"].as_u64(),
            ),
            (
                "requested_max_new_tokens",
                profile["requested_max_new_tokens"].as_u64(),
                baseline_profile["requested_max_new_tokens"].as_u64(),
            ),
        ] {
            if observed.is_none() || expected.is_none() || observed != expected {
                anyhow::bail!(
                    "{} cannot be compared to baseline {}: profile {profile_id} {label} mismatch",
                    path.display(),
                    baseline_path.display()
                );
            }
        }
        if profile["target_context_tokens"] != baseline_profile["target_context_tokens"] {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: profile {profile_id} target_context_tokens mismatch",
                path.display(),
                baseline_path.display()
            );
        }
    }
    let baseline_profiles = baseline["profiles"].as_array().ok_or_else(|| {
        anyhow!("regression baseline {} is missing profiles", baseline_path.display())
    })?;
    for baseline_profile in baseline_profiles {
        let Some(profile_id) = baseline_profile["profile_id"].as_str() else {
            anyhow::bail!(
                "regression baseline {} SLM benchmark v2 profile is missing profile_id",
                baseline_path.display()
            );
        };
        if find_profile(receipt, profile_id).is_none() {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: profile {profile_id} is missing",
                path.display(),
                baseline_path.display()
            );
        }
    }
    Ok(())
}

fn ensure_warm_session_regression_context_matches(
    path: &Path,
    receipt: &serde_json::Value,
    baseline_path: &Path,
    baseline: &serde_json::Value,
) -> Result<()> {
    for (label, observed, expected) in [
        ("artifact_kind", receipt["artifact_kind"].as_str(), baseline["artifact_kind"].as_str()),
        (
            "requested_backend",
            receipt["requested_backend"].as_str(),
            baseline["requested_backend"].as_str(),
        ),
        (
            "selected_backend",
            receipt["selected_backend"].as_str(),
            baseline["selected_backend"].as_str(),
        ),
        ("runtime_api", receipt["runtime_api"].as_str(), baseline["runtime_api"].as_str()),
        (
            "model_cache.id",
            receipt["model_cache"]["id"].as_str(),
            baseline["model_cache"]["id"].as_str(),
        ),
        (
            "model_cache.sha256",
            receipt["model_cache"]["sha256"].as_str(),
            baseline["model_cache"]["sha256"].as_str(),
        ),
        (
            "model_cache.quantization",
            receipt["model_cache"]["quantization"].as_str(),
            baseline["model_cache"]["quantization"].as_str(),
        ),
        (
            "model_cache.tokenizer_model",
            receipt["model_cache"]["tokenizer_model"].as_str(),
            baseline["model_cache"]["tokenizer_model"].as_str(),
        ),
        (
            "model_cache.tokenizer_pre",
            receipt["model_cache"]["tokenizer_pre"].as_str(),
            baseline["model_cache"]["tokenizer_pre"].as_str(),
        ),
        (
            "generation.prompt_template",
            receipt["generation"]["prompt_template"].as_str(),
            baseline["generation"]["prompt_template"].as_str(),
        ),
        (
            "generation.mode",
            receipt["generation"]["mode"].as_str(),
            baseline["generation"]["mode"].as_str(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch (observed={observed:?}, baseline={expected:?})",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for (label, observed, expected) in [
        (
            "session.prompt_count",
            receipt["session"]["prompt_count"].as_u64(),
            baseline["session"]["prompt_count"].as_u64(),
        ),
        (
            "generation.max_new_tokens",
            receipt["generation"]["max_new_tokens"].as_u64(),
            baseline["generation"]["max_new_tokens"].as_u64(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch (observed={observed:?}, baseline={expected:?})",
                path.display(),
                baseline_path.display()
            );
        }
    }
    if receipt["fallback_used"].as_bool() != baseline["fallback_used"].as_bool() {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: fallback_used mismatch",
            path.display(),
            baseline_path.display()
        );
    }
    if receipt["quality_summary"]["passed"].as_bool() != Some(true)
        || baseline["quality_summary"]["passed"].as_bool() != Some(true)
    {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: both warm-session receipts must pass quality",
            path.display(),
            baseline_path.display()
        );
    }
    if receipt["determinism"].is_object()
        && baseline["determinism"].is_object()
        && (receipt["determinism"]["passed"].as_bool() != Some(true)
            || baseline["determinism"]["passed"].as_bool() != Some(true))
    {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: both warm-session receipts must pass determinism",
            path.display(),
            baseline_path.display()
        );
    }
    Ok(())
}

fn ensure_bitnet_warm_session_regression_context_matches(
    path: &Path,
    receipt: &serde_json::Value,
    baseline_path: &Path,
    baseline: &serde_json::Value,
) -> Result<()> {
    if receipt["model"]["sha256"].as_str() != Some(BITNET_M4_EXPECTED_MODEL_SHA256)
        || baseline["model"]["sha256"].as_str() != Some(BITNET_M4_EXPECTED_MODEL_SHA256)
    {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: both BitNet warm-session receipts must use the accepted Microsoft I2_S GGUF",
            path.display(),
            baseline_path.display()
        );
    }
    if receipt["model_id"].as_str() != Some(BITNET_M4_MODEL_ID)
        || baseline["model_id"].as_str() != Some(BITNET_M4_MODEL_ID)
    {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: both BitNet warm-session receipts must record model_id={BITNET_M4_MODEL_ID}",
            path.display(),
            baseline_path.display()
        );
    }

    for (label, observed, expected) in [
        ("schema_version", receipt["schema_version"].as_str(), baseline["schema_version"].as_str()),
        ("artifact_kind", receipt["artifact_kind"].as_str(), baseline["artifact_kind"].as_str()),
        (
            "operator_command",
            receipt["operator_command"].as_str(),
            baseline["operator_command"].as_str(),
        ),
        ("machine_id", receipt["machine_id"].as_str(), baseline["machine_id"].as_str()),
        (
            "requested_backend",
            receipt["requested_backend"].as_str(),
            baseline["requested_backend"].as_str(),
        ),
        (
            "selected_backend",
            receipt["selected_backend"].as_str(),
            baseline["selected_backend"].as_str(),
        ),
        ("runtime_api", receipt["runtime_api"].as_str(), baseline["runtime_api"].as_str()),
        (
            "backend.requested_backend",
            receipt["backend"]["requested_backend"].as_str(),
            baseline["backend"]["requested_backend"].as_str(),
        ),
        (
            "backend.selected_backend",
            receipt["backend"]["selected_backend"].as_str(),
            baseline["backend"]["selected_backend"].as_str(),
        ),
        (
            "backend.runtime_api",
            receipt["backend"]["runtime_api"].as_str(),
            baseline["backend"]["runtime_api"].as_str(),
        ),
        ("model_id", receipt["model_id"].as_str(), baseline["model_id"].as_str()),
        ("model.family", receipt["model"]["family"].as_str(), baseline["model"]["family"].as_str()),
        ("model.repo", receipt["model"]["repo"].as_str(), baseline["model"]["repo"].as_str()),
        ("model.file", receipt["model"]["file"].as_str(), baseline["model"]["file"].as_str()),
        ("model.path", receipt["model"]["path"].as_str(), baseline["model"]["path"].as_str()),
        ("model.sha256", receipt["model"]["sha256"].as_str(), baseline["model"]["sha256"].as_str()),
        (
            "model.architecture",
            receipt["model"]["architecture"].as_str(),
            baseline["model"]["architecture"].as_str(),
        ),
        ("model.format", receipt["model"]["format"].as_str(), baseline["model"]["format"].as_str()),
        (
            "model.loader_mode",
            receipt["model"]["loader_mode"].as_str(),
            baseline["model"]["loader_mode"].as_str(),
        ),
        (
            "model.tokenizer",
            receipt["model"]["tokenizer"].as_str(),
            baseline["model"]["tokenizer"].as_str(),
        ),
        (
            "model_cache.id",
            receipt["model_cache"]["id"].as_str(),
            baseline["model_cache"]["id"].as_str(),
        ),
        (
            "model_cache.path",
            receipt["model_cache"]["path"].as_str(),
            baseline["model_cache"]["path"].as_str(),
        ),
        (
            "model_cache.sha256",
            receipt["model_cache"]["sha256"].as_str(),
            baseline["model_cache"]["sha256"].as_str(),
        ),
        (
            "model_cache.architecture",
            receipt["model_cache"]["architecture"].as_str(),
            baseline["model_cache"]["architecture"].as_str(),
        ),
        (
            "model_cache.quantization",
            receipt["model_cache"]["quantization"].as_str(),
            baseline["model_cache"]["quantization"].as_str(),
        ),
        (
            "model_cache.tokenizer_model",
            receipt["model_cache"]["tokenizer_model"].as_str(),
            baseline["model_cache"]["tokenizer_model"].as_str(),
        ),
        (
            "model_cache.tokenizer_pre",
            receipt["model_cache"]["tokenizer_pre"].as_str(),
            baseline["model_cache"]["tokenizer_pre"].as_str(),
        ),
        (
            "tokenizer.source",
            receipt["tokenizer"]["source"].as_str(),
            baseline["tokenizer"]["source"].as_str(),
        ),
        (
            "tokenizer.type",
            receipt["tokenizer"]["type"].as_str(),
            baseline["tokenizer"]["type"].as_str(),
        ),
        (
            "tokenizer.model_family",
            receipt["tokenizer"]["model_family"].as_str(),
            baseline["tokenizer"]["model_family"].as_str(),
        ),
        (
            "tokenizer.pretokenizer_authority",
            receipt["tokenizer"]["pretokenizer_authority"].as_str(),
            baseline["tokenizer"]["pretokenizer_authority"].as_str(),
        ),
        (
            "generation.prompt_template",
            receipt["generation"]["prompt_template"].as_str(),
            baseline["generation"]["prompt_template"].as_str(),
        ),
        (
            "generation.mode",
            receipt["generation"]["mode"].as_str(),
            baseline["generation"]["mode"].as_str(),
        ),
        (
            "session.reuse_scope",
            receipt["session"]["reuse_scope"].as_str(),
            baseline["session"]["reuse_scope"].as_str(),
        ),
        (
            "session.kv_cache_reuse_policy",
            receipt["session"]["kv_cache_reuse_policy"].as_str(),
            baseline["session"]["kv_cache_reuse_policy"].as_str(),
        ),
        (
            "session.sampler_reuse_policy",
            receipt["session"]["sampler_reuse_policy"].as_str(),
            baseline["session"]["sampler_reuse_policy"].as_str(),
        ),
        (
            "bitnet_warm_prompt_source.source",
            receipt["bitnet_warm_prompt_source"]["source"].as_str(),
            baseline["bitnet_warm_prompt_source"]["source"].as_str(),
        ),
        (
            "mac_bitnet_claim_boundary.tokenizer_path",
            receipt["mac_bitnet_claim_boundary"]["tokenizer_path"].as_str(),
            baseline["mac_bitnet_claim_boundary"]["tokenizer_path"].as_str(),
        ),
        (
            "mac_bitnet_claim_boundary.tokenizer_sha256",
            receipt["mac_bitnet_claim_boundary"]["tokenizer_sha256"].as_str(),
            baseline["mac_bitnet_claim_boundary"]["tokenizer_sha256"].as_str(),
        ),
        (
            "mac_bitnet_claim_boundary.requested_backend",
            receipt["mac_bitnet_claim_boundary"]["requested_backend"].as_str(),
            baseline["mac_bitnet_claim_boundary"]["requested_backend"].as_str(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch (observed={observed:?}, baseline={expected:?})",
                path.display(),
                baseline_path.display()
            );
        }
    }

    for (label, observed, expected) in [
        (
            "tokenizer.bos",
            receipt["tokenizer"]["bos"].as_u64(),
            baseline["tokenizer"]["bos"].as_u64(),
        ),
        (
            "tokenizer.eos",
            receipt["tokenizer"]["eos"].as_u64(),
            baseline["tokenizer"]["eos"].as_u64(),
        ),
        (
            "model.vocab_size",
            receipt["model"]["vocab_size"].as_u64(),
            baseline["model"]["vocab_size"].as_u64(),
        ),
        (
            "generation.max_new_tokens",
            receipt["generation"]["max_new_tokens"].as_u64(),
            baseline["generation"]["max_new_tokens"].as_u64(),
        ),
        (
            "generation.top_k",
            receipt["generation"]["top_k"].as_u64(),
            baseline["generation"]["top_k"].as_u64(),
        ),
        (
            "session.prompt_count",
            receipt["session"]["prompt_count"].as_u64(),
            baseline["session"]["prompt_count"].as_u64(),
        ),
        (
            "session.stop_sequence_count",
            receipt["session"]["stop_sequence_count"].as_u64(),
            baseline["session"]["stop_sequence_count"].as_u64(),
        ),
        (
            "session.stop_token_id_count",
            receipt["session"]["stop_token_id_count"].as_u64(),
            baseline["session"]["stop_token_id_count"].as_u64(),
        ),
        (
            "speed.counts.prompt_count",
            receipt["speed"]["counts"]["prompt_count"].as_u64(),
            baseline["speed"]["counts"]["prompt_count"].as_u64(),
        ),
        (
            "speed.counts.prompt_tokens",
            receipt["speed"]["counts"]["prompt_tokens"].as_u64(),
            baseline["speed"]["counts"]["prompt_tokens"].as_u64(),
        ),
        (
            "speed.counts.generated_tokens",
            receipt["speed"]["counts"]["generated_tokens"].as_u64(),
            baseline["speed"]["counts"]["generated_tokens"].as_u64(),
        ),
        (
            "bitnet_warm_prompt_source.fixed_proof_prompt_count",
            receipt["bitnet_warm_prompt_source"]["fixed_proof_prompt_count"].as_u64(),
            baseline["bitnet_warm_prompt_source"]["fixed_proof_prompt_count"].as_u64(),
        ),
        (
            "bitnet_warm_prompt_source.session_prompt_count",
            receipt["bitnet_warm_prompt_source"]["session_prompt_count"].as_u64(),
            baseline["bitnet_warm_prompt_source"]["session_prompt_count"].as_u64(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch (observed={observed:?}, baseline={expected:?})",
                path.display(),
                baseline_path.display()
            );
        }
    }

    for (label, observed, expected) in [
        ("fallback_used", receipt["fallback_used"].as_bool(), baseline["fallback_used"].as_bool()),
        (
            "backend.fallback_used",
            receipt["backend"]["fallback_used"].as_bool(),
            baseline["backend"]["fallback_used"].as_bool(),
        ),
        (
            "model.fallback_loader_used",
            receipt["model"]["fallback_loader_used"].as_bool(),
            baseline["model"]["fallback_loader_used"].as_bool(),
        ),
        (
            "tokenizer.strict",
            receipt["tokenizer"]["strict"].as_bool(),
            baseline["tokenizer"]["strict"].as_bool(),
        ),
        (
            "generation.deterministic",
            receipt["generation"]["deterministic"].as_bool(),
            baseline["generation"]["deterministic"].as_bool(),
        ),
        (
            "session.model_loaded_once",
            receipt["session"]["model_loaded_once"].as_bool(),
            baseline["session"]["model_loaded_once"].as_bool(),
        ),
        (
            "session.tokenizer_loaded_once",
            receipt["session"]["tokenizer_loaded_once"].as_bool(),
            baseline["session"]["tokenizer_loaded_once"].as_bool(),
        ),
        (
            "session.per_prompt_receipts_enabled",
            receipt["session"]["per_prompt_receipts_enabled"].as_bool(),
            baseline["session"]["per_prompt_receipts_enabled"].as_bool(),
        ),
        (
            "bitnet_warm_prompt_source.variable_prompts",
            receipt["bitnet_warm_prompt_source"]["variable_prompts"].as_bool(),
            baseline["bitnet_warm_prompt_source"]["variable_prompts"].as_bool(),
        ),
        (
            "bitnet_warm_prompt_source.determinism_requires_repeated_prompt",
            receipt["bitnet_warm_prompt_source"]["determinism_requires_repeated_prompt"].as_bool(),
            baseline["bitnet_warm_prompt_source"]["determinism_requires_repeated_prompt"].as_bool(),
        ),
        (
            "quality_summary.fail_on_quality",
            receipt["quality_summary"]["fail_on_quality"].as_bool(),
            baseline["quality_summary"]["fail_on_quality"].as_bool(),
        ),
    ] {
        if observed.is_none() || expected.is_none() || observed != expected {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch",
                path.display(),
                baseline_path.display()
            );
        }
    }

    if receipt["backend"]["fallback_reason"] != baseline["backend"]["fallback_reason"] {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: backend.fallback_reason mismatch",
            path.display(),
            baseline_path.display()
        );
    }
    if receipt["timeout_policy"] != baseline["timeout_policy"]
        || receipt["timeout_seconds"] != baseline["timeout_seconds"]
    {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: timeout policy mismatch",
            path.display(),
            baseline_path.display()
        );
    }
    for (label, observed, expected) in [
        (
            "generation.temperature",
            receipt["generation"]["temperature"].as_f64(),
            baseline["generation"]["temperature"].as_f64(),
        ),
        (
            "generation.top_p",
            receipt["generation"]["top_p"].as_f64(),
            baseline["generation"]["top_p"].as_f64(),
        ),
        (
            "generation.repetition_penalty",
            receipt["generation"]["repetition_penalty"].as_f64(),
            baseline["generation"]["repetition_penalty"].as_f64(),
        ),
    ] {
        let values_match = match (observed, expected) {
            (Some(observed), Some(expected)) => (observed - expected).abs() <= f64::EPSILON,
            _ => false,
        };
        if !values_match {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: {label} mismatch",
                path.display(),
                baseline_path.display()
            );
        }
    }
    if receipt["quality_summary"]["passed"].as_bool() != Some(true)
        || baseline["quality_summary"]["passed"].as_bool() != Some(true)
    {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: both BitNet warm-session receipts must pass quality",
            path.display(),
            baseline_path.display()
        );
    }
    if receipt["determinism"]["checked"].as_bool() != Some(true)
        || baseline["determinism"]["checked"].as_bool() != Some(true)
        || receipt["determinism"]["passed"].as_bool() != Some(true)
        || baseline["determinism"]["passed"].as_bool() != Some(true)
    {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: both BitNet warm-session receipts must pass repeated-prompt determinism",
            path.display(),
            baseline_path.display()
        );
    }
    ensure_bitnet_warm_session_claim_boundaries_match(path, receipt, baseline_path, baseline)?;
    ensure_bitnet_warm_session_prompts_match(path, receipt, baseline_path, baseline)?;
    Ok(())
}

fn ensure_bitnet_warm_session_claim_boundaries_match(
    path: &Path,
    receipt: &serde_json::Value,
    baseline_path: &Path,
    baseline: &serde_json::Value,
) -> Result<()> {
    for flag in
        ["bitnet_warm_session", "warm_session_flow", "model_loaded_once", "tokenizer_loaded_once"]
    {
        if receipt["claim_boundary"][flag].as_bool() != Some(true)
            || baseline["claim_boundary"][flag].as_bool() != Some(true)
        {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: claim_boundary.{flag} must remain true",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for flag in [
        "bitnet_quality_claimed",
        "broad_performance_claim",
        "chat_enabled",
        "full_metal_inference_claimed",
        "metal_phase_contribution_only",
        "mpsgraph_inference_claimed",
        "neural_engine_execution_claimed",
        "qk256_apple_claimed",
        "serve_enabled",
        "speedup_claim",
    ] {
        if receipt["claim_boundary"][flag].as_bool() != Some(false)
            || baseline["claim_boundary"][flag].as_bool() != Some(false)
        {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: claim_boundary.{flag} must remain false",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for flag in ["bitnet_warm_session"] {
        if receipt["mac_bitnet_claim_boundary"][flag].as_bool() != Some(true)
            || baseline["mac_bitnet_claim_boundary"][flag].as_bool() != Some(true)
        {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: mac_bitnet_claim_boundary.{flag} must remain true",
                path.display(),
                baseline_path.display()
            );
        }
    }
    for flag in [
        "broad_performance_claim",
        "chat_enabled",
        "full_metal_inference_claimed",
        "mpsgraph_inference_claimed",
        "neural_engine_execution_claimed",
        "qk256_apple_claimed",
        "serve_enabled",
        "speedup_claim",
    ] {
        if receipt["mac_bitnet_claim_boundary"][flag].as_bool() != Some(false)
            || baseline["mac_bitnet_claim_boundary"][flag].as_bool() != Some(false)
        {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: mac_bitnet_claim_boundary.{flag} must remain false",
                path.display(),
                baseline_path.display()
            );
        }
    }
    Ok(())
}

fn ensure_bitnet_warm_session_prompts_match(
    path: &Path,
    receipt: &serde_json::Value,
    baseline_path: &Path,
    baseline: &serde_json::Value,
) -> Result<()> {
    let prompts = receipt["prompts"].as_array().ok_or_else(|| {
        anyhow!("{} BitNet warm-session receipt is missing prompts", path.display())
    })?;
    let baseline_prompts = baseline["prompts"].as_array().ok_or_else(|| {
        anyhow!(
            "regression baseline {} BitNet warm-session receipt is missing prompts",
            baseline_path.display()
        )
    })?;
    if prompts.len() != baseline_prompts.len() {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: prompt set length mismatch (observed={}, baseline={})",
            path.display(),
            baseline_path.display(),
            prompts.len(),
            baseline_prompts.len()
        );
    }
    let session_prompt_count = receipt["session"]["prompt_count"].as_u64().unwrap_or_default();
    if prompts.len() as u64 != session_prompt_count {
        anyhow::bail!(
            "{} cannot be compared to baseline {}: prompts length does not match session.prompt_count",
            path.display(),
            baseline_path.display()
        );
    }
    for (index, (prompt, baseline_prompt)) in prompts.iter().zip(baseline_prompts).enumerate() {
        for (label, observed, expected) in [
            ("case_id", prompt["case_id"].as_str(), baseline_prompt["case_id"].as_str()),
            ("prompt", prompt["prompt"].as_str(), baseline_prompt["prompt"].as_str()),
            (
                "backend.requested_backend",
                prompt["backend"]["requested_backend"].as_str(),
                baseline_prompt["backend"]["requested_backend"].as_str(),
            ),
            (
                "backend.selected_backend",
                prompt["backend"]["selected_backend"].as_str(),
                baseline_prompt["backend"]["selected_backend"].as_str(),
            ),
            (
                "backend.runtime_api",
                prompt["backend"]["runtime_api"].as_str(),
                baseline_prompt["backend"]["runtime_api"].as_str(),
            ),
        ] {
            if observed.is_none() || expected.is_none() || observed != expected {
                anyhow::bail!(
                    "{} cannot be compared to baseline {}: prompts[{index}].{label} mismatch (observed={observed:?}, baseline={expected:?})",
                    path.display(),
                    baseline_path.display()
                );
            }
        }
        for (label, observed, expected) in [
            (
                "prompt_index",
                prompt["prompt_index"].as_u64(),
                baseline_prompt["prompt_index"].as_u64(),
            ),
            (
                "repeat_index",
                prompt["repeat_index"].as_u64(),
                baseline_prompt["repeat_index"].as_u64(),
            ),
            (
                "generated_tokens",
                prompt["generated_tokens"].as_u64(),
                baseline_prompt["generated_tokens"].as_u64(),
            ),
        ] {
            if observed.is_none() || expected.is_none() || observed != expected {
                anyhow::bail!(
                    "{} cannot be compared to baseline {}: prompts[{index}].{label} mismatch (observed={observed:?}, baseline={expected:?})",
                    path.display(),
                    baseline_path.display()
                );
            }
        }
        if prompt["backend"]["fallback_used"].as_bool() != Some(false)
            || baseline_prompt["backend"]["fallback_used"].as_bool() != Some(false)
        {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: prompts[{index}].backend.fallback_used must remain false",
                path.display(),
                baseline_path.display()
            );
        }
        if prompt["quality"]["passed"].as_bool() != Some(true)
            || baseline_prompt["quality"]["passed"].as_bool() != Some(true)
        {
            anyhow::bail!(
                "{} cannot be compared to baseline {}: prompts[{index}].quality.passed must remain true",
                path.display(),
                baseline_path.display()
            );
        }
    }
    Ok(())
}

fn find_profile<'a>(
    receipt: &'a serde_json::Value,
    profile_id: &str,
) -> Option<&'a serde_json::Value> {
    receipt["profiles"]
        .as_array()?
        .iter()
        .find(|profile| profile["profile_id"].as_str() == Some(profile_id))
}

fn regression_metric(value: &serde_json::Value, path: &[&str]) -> Result<f64> {
    let mut current = value;
    for segment in path {
        current = &current[*segment];
    }
    metric_value(current)
        .ok_or_else(|| anyhow!("missing numeric regression metric {}", path.join(".")))
}

fn metric_value(value: &serde_json::Value) -> Option<f64> {
    if let Some(number) = value.as_f64() {
        return Some(number);
    }
    if let Some(mean) = value["mean_ms"].as_f64() {
        return Some(mean);
    }
    let values = value.as_array()?;
    let mut total = 0.0;
    let mut count = 0usize;
    for value in values {
        if let Some(number) = value.as_f64() {
            total += number;
            count += 1;
        }
    }
    (count > 0).then_some(total / count as f64)
}

fn compare_lower_is_worse(
    warnings: &mut Vec<RegressionWarning>,
    profile_id: &str,
    field: &str,
    baseline: f64,
    observed: f64,
    threshold_percent: f64,
) {
    if baseline > 0.0 && observed < baseline * (1.0 - threshold_percent / 100.0) {
        warnings.push(RegressionWarning {
            profile_id: profile_id.to_string(),
            field: field.to_string(),
            baseline: round3(baseline),
            observed: round3(observed),
            threshold_percent,
            direction: "lower_is_worse".to_string(),
        });
    }
}

fn compare_higher_is_worse(
    warnings: &mut Vec<RegressionWarning>,
    profile_id: &str,
    field: &str,
    baseline: f64,
    observed: f64,
    threshold_percent: f64,
) {
    if observed > baseline * (1.0 + threshold_percent / 100.0) {
        warnings.push(RegressionWarning {
            profile_id: profile_id.to_string(),
            field: field.to_string(),
            baseline: round3(baseline),
            observed: round3(observed),
            threshold_percent,
            direction: "higher_is_worse".to_string(),
        });
    }
}

fn validate_mac_receipt_value(
    path: &Path,
    receipt: &serde_json::Value,
) -> Result<ReceiptCheckSummary> {
    let artifact_kind = receipt["artifact_kind"].as_str().unwrap_or("<missing>").to_string();
    if is_metal_phase_receipt(receipt) {
        return validate_metal_phase_receipt(path, receipt, artifact_kind);
    }

    let requested_backend = receipt_string(receipt, "requested_backend").unwrap_or_default();
    let selected_backend = receipt_string(receipt, "selected_backend").unwrap_or_default();
    let runtime_api = receipt_string(receipt, "runtime_api").unwrap_or_default();
    let fallback_used = receipt_bool(receipt, "fallback_used").unwrap_or(true);

    if !is_supported_mac_cpu_neon_receipt_backend(requested_backend.as_str()) {
        anyhow::bail!(
            "{} requested_backend must be {APPLE_M4_CPU_NEON} or {APPLE_M3_AIR_CPU_NEON}, got {requested_backend:?}",
            path.display()
        );
    }
    if selected_backend != requested_backend {
        anyhow::bail!(
            "{} selected_backend must match requested_backend {requested_backend:?}, got {selected_backend:?}",
            path.display()
        );
    }
    if runtime_api != "cpu" {
        anyhow::bail!("{} runtime_api must be cpu, got {runtime_api:?}", path.display());
    }
    if fallback_used {
        anyhow::bail!(
            "{} records fallback_used=true; hidden fallback is not allowed",
            path.display()
        );
    }
    if receipt_flag_true(receipt, "full_metal_inference_claimed") {
        anyhow::bail!("{} claims full apple-m4-metal inference", path.display());
    }
    if receipt_flag_true(receipt, "neural_engine_execution_claimed") {
        anyhow::bail!("{} claims Neural Engine execution", path.display());
    }
    if receipt_flag_true(receipt, "mpsgraph_inference_claimed") {
        anyhow::bail!("{} claims MPSGraph model inference", path.display());
    }
    if receipt_flag_true(receipt, "qk256_apple_claimed") {
        anyhow::bail!("{} claims QK256 on Apple Silicon", path.display());
    }
    if receipt_flag_true(receipt, "bitnet_quality_claimed") {
        anyhow::bail!("{} claims BitNet local-answer quality", path.display());
    }
    if receipt_flag_true(receipt, "broad_performance_claim")
        || receipt_flag_true(receipt, "speedup_claim")
    {
        anyhow::bail!("{} claims broad Mac performance or speedup", path.display());
    }

    let (prompt_count, generated_tokens) = if artifact_kind == "slm_apple_m4_warm_session"
        || artifact_kind == "slm_apple_m3_air_warm_session"
        || artifact_kind == "bitnet_apple_m4_warm_session"
    {
        validate_warm_session_receipt(path, receipt, requested_backend.as_str())?
    } else if artifact_kind == "apple_m4_slm_operator_profiles"
        || artifact_kind == "apple_m4_slm_performance_profiles"
        || artifact_kind == "apple_m3_air_slm_operator_profiles"
        || artifact_kind == "apple_m3_air_slm_performance_profiles"
    {
        validate_profile_set_receipt(
            path,
            receipt,
            requested_backend.as_str(),
            artifact_kind.as_str(),
        )?
    } else if artifact_kind == "apple_m4_slm_eval_summary" {
        validate_slm_eval_summary_receipt(path, receipt)?
    } else if artifact_kind == "bitnet_apple_m4_local_answer_corpus" {
        validate_bitnet_eval_answer_corpus_receipt(path, receipt)?
    } else if artifact_kind == "apple_m4_slm_benchmark_v2" {
        validate_slm_benchmark_v2_receipt(path, receipt)?
    } else if artifact_kind == "bitnet_apple_m4_benchmark_v1" {
        validate_bitnet_benchmark_v1_receipt(path, receipt)?
    } else if artifact_kind == "bitnet_apple_m4_mac_ask_failure" {
        validate_bitnet_mac_ask_failure_receipt(path, receipt)?
    } else if artifact_kind == "bitnet_apple_m4_warm_session_failure" {
        validate_bitnet_warm_session_failure_receipt(path, receipt)?
    } else if artifact_kind == "bitnet_apple_m4_chat_gate" {
        validate_bitnet_chat_gate_receipt(path, receipt)?
    } else if artifact_kind == "apple_m4_inference_status" {
        validate_apple_m4_inference_status_receipt(path, receipt)?
    } else if artifact_kind == "apple_m4_report_refresh_manifest" {
        validate_apple_m4_report_refresh_manifest_receipt(path, receipt)?
    } else if artifact_kind == "apple_m4_regression_dashboard" {
        validate_apple_m4_regression_dashboard_receipt(path, receipt)?
    } else {
        validate_one_shot_receipt(path, receipt)?
    };

    Ok(ReceiptCheckSummary {
        path: path.to_path_buf(),
        artifact_kind,
        requested_backend,
        selected_backend,
        runtime_api,
        fallback_used,
        prompt_count,
        generated_tokens,
        passed: true,
        regression: None,
    })
}

fn is_supported_mac_cpu_neon_receipt_backend(label: &str) -> bool {
    matches!(label, APPLE_M4_CPU_NEON | APPLE_M3_AIR_CPU_NEON)
}

fn validate_bitnet_mac_ask_failure_receipt(
    path: &Path,
    receipt: &serde_json::Value,
) -> Result<(Option<usize>, Option<usize>)> {
    if receipt["schema_version"].as_str() != Some("1.0.0") {
        anyhow::bail!(
            "{} BitNet Mac ask failure receipt schema_version must be 1.0.0",
            path.display()
        );
    }
    if receipt["status"].as_str() != Some("failed") {
        anyhow::bail!("{} BitNet Mac ask failure receipt status must be failed", path.display());
    }
    if receipt["operator_command"].as_str() != Some("mac ask") {
        anyhow::bail!(
            "{} BitNet Mac ask failure receipt must record operator_command=mac ask",
            path.display()
        );
    }
    if receipt["model_id"].as_str() != Some("microsoft-bitnet-b1.58-2B-4T-i2s") {
        anyhow::bail!(
            "{} BitNet Mac ask failure receipt must record the accepted BitNet model id",
            path.display()
        );
    }
    if receipt["model"]["expected_sha256"].as_str() != Some(BITNET_M4_EXPECTED_MODEL_SHA256) {
        anyhow::bail!(
            "{} BitNet Mac ask failure receipt must record the accepted model SHA256",
            path.display()
        );
    }
    if receipt["tokenizer"]["expected_sha256"].as_str() != Some(BITNET_M4_EXPECTED_TOKENIZER_SHA256)
        || receipt["tokenizer"]["strict"].as_bool() != Some(true)
        || receipt["tokenizer"]["pretokenizer_authority"].as_str() != Some("llama-bpe")
    {
        anyhow::bail!(
            "{} BitNet Mac ask failure receipt must record strict external llama-bpe tokenizer authority",
            path.display()
        );
    }
    if receipt["prompt"]["template_family"].as_str() != Some(BITNET_M4_PROMPT_TEMPLATE) {
        anyhow::bail!(
            "{} BitNet Mac ask failure receipt must record the BitNet prompt template",
            path.display()
        );
    }
    if receipt["generation"]["generated_text"].as_str() != Some("")
        || receipt["generation"]["generated_token_ids"].as_array().is_none_or(|ids| !ids.is_empty())
        || receipt["generation"]["generated_tokens"].as_u64() != Some(0)
        || receipt["generation"]["partial_text"].as_str() != Some("")
        || receipt["generation"]["partial_token_ids"].as_array().is_none_or(|ids| !ids.is_empty())
        || receipt["generation"]["partial_generation_available"].as_bool() != Some(false)
    {
        anyhow::bail!(
            "{} BitNet Mac ask failure receipt must record empty partial generation",
            path.display()
        );
    }
    if receipt["failure"]["stage"].as_str().is_none_or(str::is_empty)
        || receipt["failure"]["message"].as_str().is_none_or(str::is_empty)
        || receipt["failure"]["elapsed_ms"].as_f64().is_none()
    {
        anyhow::bail!(
            "{} BitNet Mac ask failure receipt must record failure stage, message, and elapsed_ms",
            path.display()
        );
    }
    let taxonomy = receipt["progress"]["stage_taxonomy"].as_array().ok_or_else(|| {
        anyhow!(
            "{} BitNet Mac ask failure receipt is missing progress.stage_taxonomy",
            path.display()
        )
    })?;
    for required in
        ["model_load", "tokenizer_load", "prefill", "first_token", "decode", "receipt_write"]
    {
        if !taxonomy.iter().any(|stage| stage.as_str() == Some(required)) {
            anyhow::bail!(
                "{} BitNet Mac ask failure receipt progress taxonomy must include {required}",
                path.display()
            );
        }
    }
    if receipt["progress"]["last_stage"].as_str().is_none_or(str::is_empty) {
        anyhow::bail!(
            "{} BitNet Mac ask failure receipt must record progress.last_stage",
            path.display()
        );
    }
    if receipt["timeout_boundary"]["reached"].as_bool().is_none()
        || receipt["timeout_boundary"]["enforced"].as_bool().is_none()
        || receipt["timeout_boundary"]["status"].as_str().is_none_or(str::is_empty)
        || receipt["timeout_boundary"]["stage"].as_str().is_none_or(str::is_empty)
    {
        anyhow::bail!(
            "{} BitNet Mac ask failure receipt must record an explicit timeout boundary",
            path.display()
        );
    }
    if receipt["repair_guidance"].as_array().is_none_or(|guidance| guidance.is_empty()) {
        anyhow::bail!(
            "{} BitNet Mac ask failure receipt must include repair guidance",
            path.display()
        );
    }
    if receipt["mac_bitnet_claim_boundary"]["bitnet_one_shot_mac_ask"].as_bool() != Some(true)
        || receipt["mac_bitnet_claim_boundary"]["partial_failure_receipt"].as_bool() != Some(true)
        || receipt["mac_bitnet_claim_boundary"]["chat_enabled"].as_bool() != Some(false)
        || receipt["mac_bitnet_claim_boundary"]["serve_enabled"].as_bool() != Some(false)
        || receipt["mac_bitnet_claim_boundary"]["full_metal_inference_claimed"].as_bool()
            != Some(false)
        || receipt["mac_bitnet_claim_boundary"]["qk256_apple_claimed"].as_bool() != Some(false)
        || receipt["mac_bitnet_claim_boundary"]["broad_performance_claim"].as_bool() != Some(false)
        || receipt["mac_bitnet_claim_boundary"]["speedup_claim"].as_bool() != Some(false)
        || receipt["bitnet_quality_claimed"].as_bool() != Some(false)
    {
        anyhow::bail!(
            "{} BitNet Mac ask failure receipt must preserve BitNet ask claim boundaries",
            path.display()
        );
    }
    Ok((Some(1), Some(0)))
}

fn validate_bitnet_warm_session_failure_receipt(
    path: &Path,
    receipt: &serde_json::Value,
) -> Result<(Option<usize>, Option<usize>)> {
    if receipt["schema_version"].as_str() != Some("1.0.0") {
        anyhow::bail!(
            "{} BitNet warm failure receipt schema_version must be 1.0.0",
            path.display()
        );
    }
    if receipt["status"].as_str() != Some("failed") {
        anyhow::bail!("{} BitNet warm failure receipt status must be failed", path.display());
    }
    if receipt["operator_command"].as_str() != Some("mac bitnet-warm") {
        anyhow::bail!(
            "{} BitNet warm failure receipt must record operator_command=mac bitnet-warm",
            path.display()
        );
    }
    if receipt["model_id"].as_str() != Some(BITNET_M4_MODEL_ID) {
        anyhow::bail!(
            "{} BitNet warm failure receipt must record the accepted BitNet model id",
            path.display()
        );
    }
    if receipt["model"]["expected_sha256"].as_str() != Some(BITNET_M4_EXPECTED_MODEL_SHA256) {
        anyhow::bail!(
            "{} BitNet warm failure receipt must record the accepted model SHA256",
            path.display()
        );
    }
    if receipt["tokenizer"]["expected_sha256"].as_str() != Some(BITNET_M4_EXPECTED_TOKENIZER_SHA256)
        || receipt["tokenizer"]["strict"].as_bool() != Some(true)
        || receipt["tokenizer"]["pretokenizer_authority"].as_str() != Some("llama-bpe")
    {
        anyhow::bail!(
            "{} BitNet warm failure receipt must record strict external llama-bpe tokenizer authority",
            path.display()
        );
    }
    if receipt["prompt"]["template_family"].as_str() != Some(BITNET_M4_PROMPT_TEMPLATE)
        || receipt["prompt"]["count"].as_u64().unwrap_or_default() < 2
        || receipt["prompt"]["source"].as_str().is_none_or(str::is_empty)
    {
        anyhow::bail!(
            "{} BitNet warm failure receipt must record prompt count, source, and template",
            path.display()
        );
    }
    if receipt["generation"]["generated_text"].as_str() != Some("")
        || receipt["generation"]["generated_token_ids"].as_array().is_none_or(|ids| !ids.is_empty())
        || receipt["generation"]["generated_tokens"].as_u64() != Some(0)
        || receipt["generation"]["partial_text"].as_str() != Some("")
        || receipt["generation"]["partial_token_ids"].as_array().is_none_or(|ids| !ids.is_empty())
        || receipt["generation"]["partial_generation_available"].as_bool() != Some(false)
    {
        anyhow::bail!(
            "{} BitNet warm failure receipt must record empty partial generation",
            path.display()
        );
    }
    if receipt["failure"]["stage"].as_str().is_none_or(str::is_empty)
        || receipt["failure"]["message"].as_str().is_none_or(str::is_empty)
        || receipt["failure"]["elapsed_ms"].as_f64().is_none()
    {
        anyhow::bail!(
            "{} BitNet warm failure receipt must record failure stage, message, and elapsed_ms",
            path.display()
        );
    }
    let taxonomy = receipt["progress"]["stage_taxonomy"].as_array().ok_or_else(|| {
        anyhow!("{} BitNet warm failure receipt is missing progress.stage_taxonomy", path.display())
    })?;
    for required in
        ["model_load", "tokenizer_load", "prefill", "first_token", "decode", "receipt_write"]
    {
        if !taxonomy.iter().any(|stage| stage.as_str() == Some(required)) {
            anyhow::bail!(
                "{} BitNet warm failure receipt progress taxonomy must include {required}",
                path.display()
            );
        }
    }
    if receipt["timeout_boundary"]["reached"].as_bool().is_none()
        || receipt["timeout_boundary"]["enforced"].as_bool().is_none()
        || receipt["timeout_boundary"]["status"].as_str().is_none_or(str::is_empty)
    {
        anyhow::bail!(
            "{} BitNet warm failure receipt must record an explicit timeout boundary",
            path.display()
        );
    }
    if receipt["repair_guidance"].as_array().is_none_or(|guidance| guidance.is_empty()) {
        anyhow::bail!(
            "{} BitNet warm failure receipt must include repair guidance",
            path.display()
        );
    }
    if receipt["mac_bitnet_claim_boundary"]["bitnet_warm_session"].as_bool() != Some(true)
        || receipt["mac_bitnet_claim_boundary"]["partial_failure_receipt"].as_bool() != Some(true)
        || receipt["mac_bitnet_claim_boundary"]["chat_enabled"].as_bool() != Some(false)
        || receipt["mac_bitnet_claim_boundary"]["serve_enabled"].as_bool() != Some(false)
        || receipt["mac_bitnet_claim_boundary"]["full_metal_inference_claimed"].as_bool()
            != Some(false)
        || receipt["mac_bitnet_claim_boundary"]["qk256_apple_claimed"].as_bool() != Some(false)
        || receipt["mac_bitnet_claim_boundary"]["broad_performance_claim"].as_bool() != Some(false)
        || receipt["mac_bitnet_claim_boundary"]["speedup_claim"].as_bool() != Some(false)
        || receipt["bitnet_quality_claimed"].as_bool() != Some(false)
    {
        anyhow::bail!(
            "{} BitNet warm failure receipt must preserve BitNet warm claim boundaries",
            path.display()
        );
    }
    Ok((Some(receipt["prompt"]["count"].as_u64().unwrap_or_default() as usize), Some(0)))
}

fn validate_bitnet_chat_gate_receipt(
    path: &Path,
    receipt: &serde_json::Value,
) -> Result<(Option<usize>, Option<usize>)> {
    require_exact_string_at(path, receipt, &["schema_version"], "1.0.0")?;
    require_exact_string_at(path, receipt, &["artifact_kind"], "bitnet_apple_m4_chat_gate")?;
    require_exact_string_at(path, receipt, &["operator_command"], "mac bitnet-chat-gate")?;
    let status = require_non_empty_string_at(path, receipt, &["status"])?;
    if !matches!(status, "blocked" | "ready_to_enable") {
        anyhow::bail!(
            "{} BitNet chat gate status must be blocked or ready_to_enable",
            path.display()
        );
    }
    require_exact_string_at(path, receipt, &["model_id"], BITNET_M4_MODEL_ID)?;
    require_exact_string_at(path, receipt, &["model", "family"], "bitnet")?;
    require_exact_string_at(
        path,
        receipt,
        &["model", "expected_sha256"],
        BITNET_M4_EXPECTED_MODEL_SHA256,
    )?;
    require_exact_string_at(path, receipt, &["model", "quant_format"], "I2_S")?;
    require_exact_string_at(
        path,
        receipt,
        &["tokenizer", "expected_sha256"],
        BITNET_M4_EXPECTED_TOKENIZER_SHA256,
    )?;
    require_exact_string_at(path, receipt, &["tokenizer", "pretokenizer_authority"], "llama-bpe")?;
    require_bool_at(path, receipt, &["tokenizer", "strict"], true)?;
    require_exact_string_at(
        path,
        receipt,
        &["prompt", "template_family"],
        BITNET_M4_PROMPT_TEMPLATE,
    )?;

    for key in
        ["variable_warm_session_receipt", "timeout_failure_receipt", "streaming_semantics_receipt"]
    {
        require_bool_at(path, receipt, &["requirements", key, "required"], true)?;
        if json_value_at(receipt, &["requirements", key, "passed"]).as_bool().is_none() {
            anyhow::bail!(
                "{} BitNet chat gate requirement {key}.passed must be recorded",
                path.display()
            );
        }
    }
    require_bool_at(
        path,
        receipt,
        &["requirements", "variable_warm_session_receipt", "required"],
        true,
    )?;
    if json_value_at(
        receipt,
        &["requirements", "variable_warm_session_receipt", "repeated_prompt_determinism_passed"],
    )
    .as_bool()
    .is_none()
    {
        anyhow::bail!(
            "{} BitNet chat gate must record repeated-prompt determinism evidence",
            path.display()
        );
    }
    if json_value_at(
        receipt,
        &["requirements", "timeout_failure_receipt", "timeout_boundary_recorded"],
    )
    .as_bool()
    .is_none()
    {
        anyhow::bail!(
            "{} BitNet chat gate must record timeout/failure boundary evidence",
            path.display()
        );
    }
    let gate_passed = receipt["chat_enablement"]["gate_passed"].as_bool().ok_or_else(|| {
        anyhow!("{} BitNet chat gate must record chat_enablement.gate_passed", path.display())
    })?;
    let all_requirements_passed =
        ["variable_warm_session_receipt", "timeout_failure_receipt", "streaming_semantics_receipt"]
            .iter()
            .all(|key| {
                json_value_at(receipt, &["requirements", key, "passed"]).as_bool() == Some(true)
            });
    if gate_passed != all_requirements_passed {
        anyhow::bail!(
            "{} BitNet chat gate gate_passed must match requirement results",
            path.display()
        );
    }
    if (status == "ready_to_enable") != gate_passed {
        anyhow::bail!("{} BitNet chat gate status must reflect gate_passed", path.display());
    }
    require_bool_at(path, receipt, &["chat_enablement", "chat_enabled"], false)?;
    require_bool_at(path, receipt, &["chat_enablement", "serve_enabled"], false)?;
    require_bool_at(path, receipt, &["mac_bitnet_claim_boundary", "bitnet_chat_gate"], true)?;
    require_bool_at(path, receipt, &["mac_bitnet_claim_boundary", "chat_enabled"], false)?;
    require_bool_at(path, receipt, &["mac_bitnet_claim_boundary", "serve_enabled"], false)?;
    require_bool_at(
        path,
        receipt,
        &["mac_bitnet_claim_boundary", "full_metal_inference_claimed"],
        false,
    )?;
    require_bool_at(path, receipt, &["mac_bitnet_claim_boundary", "qk256_apple_claimed"], false)?;
    require_bool_at(
        path,
        receipt,
        &["mac_bitnet_claim_boundary", "broad_performance_claim"],
        false,
    )?;
    require_bool_at(path, receipt, &["mac_bitnet_claim_boundary", "speedup_claim"], false)?;
    require_bool_at(path, receipt, &["bitnet_quality_claimed"], false)?;
    Ok((Some(0), Some(0)))
}

fn validate_apple_m4_inference_status_receipt(
    path: &Path,
    receipt: &serde_json::Value,
) -> Result<(Option<usize>, Option<usize>)> {
    require_exact_string_at(path, receipt, &["schema_version"], "1.0.0")?;
    require_exact_string_at(path, receipt, &["artifact_kind"], "apple_m4_inference_status")?;
    require_exact_string_at(path, receipt, &["operator_command"], "mac status")?;
    require_exact_string_at(path, receipt, &["machine", "id"], "apple-m4-mac-mini")?;
    require_bool_at(path, receipt, &["claim_boundary", "status_only"], true)?;
    require_bool_at(path, receipt, &["claim_boundary", "no_live_model_run"], true)?;
    require_bool_at(
        path,
        receipt,
        &["claim_boundary", "dense_slm_and_bitnet_evidence_separated"],
        true,
    )?;
    require_bool_at(path, receipt, &["claim_boundary", "bitnet_chat_enabled"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "bitnet_serve_enabled"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "full_metal_inference_claimed"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "qk256_apple_claimed"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "neural_engine_execution_claimed"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "mpsgraph_inference_claimed"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "macbook_evidence"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "broad_apple_silicon_claim"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "broad_performance_claim"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "speedup_claim"], false)?;

    require_non_empty_string_at(path, receipt, &["dense_slm", "default_model_id"])?;
    let supported_count =
        require_u64_at(path, receipt, &["dense_slm", "supported_model_count"], true)?;
    let ready_count = require_u64_at(path, receipt, &["dense_slm", "ready_model_count"], false)?;
    if ready_count > supported_count {
        anyhow::bail!(
            "{} M4 inference status ready_model_count must not exceed supported_model_count",
            path.display()
        );
    }
    require_bool_at(path, receipt, &["dense_slm", "ask_enabled"], true)?;
    require_bool_at(path, receipt, &["dense_slm", "chat_enabled"], true)?;
    require_bool_at(path, receipt, &["dense_slm", "serve_enabled"], true)?;
    require_bool_at(
        path,
        receipt,
        &["dense_slm", "claim_boundary", "broad_model_quality_claim"],
        false,
    )?;
    require_bool_at(
        path,
        receipt,
        &["dense_slm", "claim_boundary", "broad_performance_claim"],
        false,
    )?;

    require_bool_at(path, receipt, &["bitnet", "ask_enabled"], true)?;
    require_bool_at(path, receipt, &["bitnet", "warm_enabled"], true)?;
    require_bool_at(path, receipt, &["bitnet", "chat_enabled"], false)?;
    require_bool_at(path, receipt, &["bitnet", "serve_enabled"], false)?;
    require_bool_at(path, receipt, &["bitnet", "claim_boundary", "chat_enabled"], false)?;
    require_bool_at(path, receipt, &["bitnet", "claim_boundary", "serve_enabled"], false)?;
    require_bool_at(path, receipt, &["bitnet", "claim_boundary", "bitnet_quality_claimed"], false)?;

    for field in [
        "models",
        "status",
        "report_refresh",
        "regression_dashboard",
        "ask_default",
        "chat_dense",
        "serve_dense",
        "doctor",
        "smoke_dense",
        "regression",
        "bitnet_chat_gate",
    ] {
        require_non_empty_string_at(path, receipt, &["commands", field])?;
    }
    require_non_empty_string_at(path, receipt, &["report_inventory", "root"])?;
    Ok((Some(0), Some(0)))
}

fn validate_apple_m4_report_refresh_manifest_receipt(
    path: &Path,
    receipt: &serde_json::Value,
) -> Result<(Option<usize>, Option<usize>)> {
    require_exact_string_at(path, receipt, &["schema_version"], "1.0.0")?;
    require_exact_string_at(path, receipt, &["artifact_kind"], "apple_m4_report_refresh_manifest")?;
    require_exact_string_at(path, receipt, &["operator_command"], "mac report-refresh")?;
    require_exact_string_at(path, receipt, &["machine", "id"], "apple-m4-mac-mini")?;
    require_non_empty_string_at(path, receipt, &["report_root"])?;
    require_bool_at(path, receipt, &["refresh_modes", "advisory_manifest"], true)?;
    require_bool_at(path, receipt, &["refresh_modes", "nightly_manifest"], true)?;
    require_bool_at(path, receipt, &["refresh_modes", "release_manifest"], true)?;
    require_bool_at(path, receipt, &["refresh_modes", "generic_pr_ci_model_free"], true)?;
    require_bool_at(path, receipt, &["refresh_modes", "generic_pr_ci_live_model_run"], false)?;
    require_bool_at(path, receipt, &["refresh_modes", "model_downloads"], false)?;
    require_bool_at(path, receipt, &["refresh_modes", "long_resident_soaks"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "manifest_only"], true)?;
    require_bool_at(path, receipt, &["claim_boundary", "no_live_model_run"], true)?;
    require_bool_at(path, receipt, &["claim_boundary", "no_model_download"], true)?;
    require_bool_at(
        path,
        receipt,
        &["claim_boundary", "dense_slm_and_bitnet_evidence_separated"],
        true,
    )?;
    require_bool_at(path, receipt, &["claim_boundary", "bitnet_chat_enabled"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "bitnet_serve_enabled"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "full_metal_inference_claimed"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "qk256_apple_claimed"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "neural_engine_execution_claimed"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "mpsgraph_inference_claimed"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "macbook_evidence"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "broad_apple_silicon_claim"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "broad_model_quality_claim"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "broad_performance_claim"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "speedup_claim"], false)?;

    for field in
        ["manifest_command", "manifest_receipt_check", "report_receipt_check", "regression_check"]
    {
        require_non_empty_string_at(path, receipt, &["validation", field])?;
    }

    let families = receipt["families"].as_array().ok_or_else(|| {
        anyhow!("{} report refresh manifest is missing families array", path.display())
    })?;
    if families.len() < 5 {
        anyhow::bail!(
            "{} report refresh manifest must cover dense SLM and BitNet report families",
            path.display()
        );
    }
    let family_count = require_u64_at(path, receipt, &["family_count"], true)?;
    if family_count != families.len() as u64 {
        anyhow::bail!(
            "{} report refresh manifest family_count must equal families length",
            path.display()
        );
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut total_reports = 0_u64;
    for family in families {
        let id = require_non_empty_string_at(path, family, &["id"])?;
        seen.insert(id.to_string());
        let evidence_family = require_non_empty_string_at(path, family, &["evidence_family"])?;
        if evidence_family != "dense_slm" && evidence_family != "bitnet" {
            anyhow::bail!(
                "{} report refresh manifest family {id} has unsupported evidence_family {evidence_family:?}",
                path.display()
            );
        }
        require_non_empty_string_at(path, family, &["path_segment"])?;
        require_non_empty_string_at(path, family, &["summary_filename"])?;
        let expected_artifact_kind =
            require_non_empty_string_at(path, family, &["expected_artifact_kind"])?;
        require_non_empty_string_at(path, family, &["refresh_command_template"])?;
        require_non_empty_string_array_at(path, family, &["refresh_tiers"])?;
        require_non_empty_string_array_at(path, family, &["validation_commands"])?;
        require_bool_at(path, family, &["generic_pr_ci", "validate_committed_reports_only"], true)?;
        require_bool_at(path, family, &["generic_pr_ci", "live_model_run"], false)?;
        require_bool_at(path, family, &["generic_pr_ci", "model_downloads"], false)?;
        require_bool_at(path, family, &["claim_boundary", "evidence_families_mixed"], false)?;
        require_bool_at(path, family, &["claim_boundary", "generic_pr_ci_live_model_run"], false)?;
        require_bool_at(path, family, &["claim_boundary", "broad_model_quality_claim"], false)?;
        require_bool_at(path, family, &["claim_boundary", "broad_performance_claim"], false)?;
        require_bool_at(path, family, &["claim_boundary", "speedup_claim"], false)?;
        require_bool_at(
            path,
            family,
            &["claim_boundary", "dense_slm_evidence"],
            evidence_family == "dense_slm",
        )?;
        require_bool_at(
            path,
            family,
            &["claim_boundary", "bitnet_evidence"],
            evidence_family == "bitnet",
        )?;

        let report_count = require_u64_at(path, family, &["report_count"], true)?;
        total_reports = total_reports.saturating_add(report_count);
        require_non_empty_string_at(path, family, &["latest_report"])?;
        let reports = family["reports"].as_array().ok_or_else(|| {
            anyhow!(
                "{} report refresh manifest family {id} is missing reports array",
                path.display()
            )
        })?;
        if reports.len() as u64 != report_count {
            anyhow::bail!(
                "{} report refresh manifest family {id} report_count does not match reports length",
                path.display()
            );
        }
        let fallback_free_count = require_u64_at(path, family, &["fallback_free_count"], true)?;
        let strict_cpu_neon_count = require_u64_at(path, family, &["strict_cpu_neon_count"], true)?;
        if fallback_free_count != report_count || strict_cpu_neon_count != report_count {
            anyhow::bail!(
                "{} report refresh manifest family {id} must be fallback-free and strict apple-m4-cpu-neon for every committed report",
                path.display()
            );
        }
        for report in reports {
            require_non_empty_string_at(path, report, &["path"])?;
            require_non_empty_string_at(path, report, &["date"])?;
            require_exact_string_at(path, report, &["parse_status"], "ok")?;
            require_exact_string_at(path, report, &["artifact_kind"], expected_artifact_kind)?;
            require_exact_string_at(
                path,
                report,
                &["expected_artifact_kind"],
                expected_artifact_kind,
            )?;
            require_bool_at(path, report, &["artifact_kind_matches"], true)?;
            require_non_empty_string_at(path, report, &["selected_backend"])?;
            require_exact_string_at(path, report, &["selected_backend"], APPLE_M4_CPU_NEON)?;
            require_exact_string_at(path, report, &["runtime_api"], "cpu")?;
            require_bool_at(path, report, &["fallback_used"], false)?;
        }
    }
    for required in [
        "dense_slm_eval_v2",
        "dense_slm_benchmark_v2",
        "bitnet_eval",
        "bitnet_benchmark",
        "bitnet_variable_warm",
    ] {
        if !seen.contains(required) {
            anyhow::bail!(
                "{} report refresh manifest is missing required family {required}",
                path.display()
            );
        }
    }
    let receipt_report_count = require_u64_at(path, receipt, &["report_count"], true)?;
    if receipt_report_count != total_reports {
        anyhow::bail!(
            "{} report refresh manifest report_count must equal family report totals",
            path.display()
        );
    }
    Ok((Some(0), Some(0)))
}

fn validate_apple_m4_regression_dashboard_receipt(
    path: &Path,
    receipt: &serde_json::Value,
) -> Result<(Option<usize>, Option<usize>)> {
    require_exact_string_at(path, receipt, &["schema_version"], "1.0.0")?;
    require_exact_string_at(path, receipt, &["artifact_kind"], "apple_m4_regression_dashboard")?;
    require_exact_string_at(path, receipt, &["operator_command"], "mac regression-dashboard")?;
    require_exact_string_at(path, receipt, &["machine", "id"], "apple-m4-mac-mini")?;
    require_non_empty_string_at(path, receipt, &["report_root"])?;
    require_non_empty_string_at(path, receipt, &["markdown_path"])?;
    require_bool_at(path, receipt, &["dashboard_contract", "model_free"], true)?;
    require_bool_at(path, receipt, &["dashboard_contract", "committed_reports_only"], true)?;
    require_bool_at(
        path,
        receipt,
        &["dashboard_contract", "matching_requires_same_evidence_family"],
        true,
    )?;
    require_bool_at(
        path,
        receipt,
        &["dashboard_contract", "matching_requires_same_model_id"],
        true,
    )?;
    require_bool_at(
        path,
        receipt,
        &["dashboard_contract", "matching_requires_same_model_sha256"],
        true,
    )?;
    require_bool_at(
        path,
        receipt,
        &["dashboard_contract", "matching_requires_same_tokenizer_authority"],
        true,
    )?;
    require_bool_at(
        path,
        receipt,
        &["dashboard_contract", "matching_requires_same_artifact_kind"],
        true,
    )?;
    require_bool_at(
        path,
        receipt,
        &["dashboard_contract", "matching_requires_same_backend"],
        true,
    )?;
    require_bool_at(
        path,
        receipt,
        &["dashboard_contract", "matching_requires_fallback_false"],
        true,
    )?;
    require_bool_at(path, receipt, &["claim_boundary", "dashboard_only"], true)?;
    require_bool_at(path, receipt, &["claim_boundary", "no_live_model_run"], true)?;
    require_bool_at(path, receipt, &["claim_boundary", "no_model_download"], true)?;
    require_bool_at(
        path,
        receipt,
        &["claim_boundary", "dense_slm_and_bitnet_evidence_separated"],
        true,
    )?;
    require_bool_at(path, receipt, &["claim_boundary", "bitnet_chat_enabled"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "bitnet_serve_enabled"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "full_metal_inference_claimed"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "qk256_apple_claimed"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "neural_engine_execution_claimed"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "mpsgraph_inference_claimed"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "macbook_evidence"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "broad_apple_silicon_claim"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "broad_model_quality_claim"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "broad_performance_claim"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "speedup_claim"], false)?;

    let families = receipt["families"].as_array().ok_or_else(|| {
        anyhow!("{} regression dashboard is missing families array", path.display())
    })?;
    if families.len() < 5 {
        anyhow::bail!(
            "{} regression dashboard must cover dense SLM and BitNet report families",
            path.display()
        );
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut total_reports = 0_u64;
    let mut total_groups = 0_u64;
    for family in families {
        let id = require_non_empty_string_at(path, family, &["id"])?;
        seen.insert(id.to_string());
        let evidence_family = require_non_empty_string_at(path, family, &["evidence_family"])?;
        if evidence_family != "dense_slm" && evidence_family != "bitnet" {
            anyhow::bail!(
                "{} regression dashboard family {id} has unsupported evidence_family {evidence_family:?}",
                path.display()
            );
        }
        require_non_empty_string_at(path, family, &["expected_artifact_kind"])?;
        require_bool_at(path, family, &["claim_boundary", "evidence_families_mixed"], false)?;
        require_bool_at(path, family, &["claim_boundary", "generic_pr_ci_live_model_run"], false)?;
        require_bool_at(path, family, &["claim_boundary", "broad_model_quality_claim"], false)?;
        require_bool_at(path, family, &["claim_boundary", "broad_performance_claim"], false)?;
        require_bool_at(path, family, &["claim_boundary", "speedup_claim"], false)?;
        require_bool_at(
            path,
            family,
            &["claim_boundary", "dense_slm_evidence"],
            evidence_family == "dense_slm",
        )?;
        require_bool_at(
            path,
            family,
            &["claim_boundary", "bitnet_evidence"],
            evidence_family == "bitnet",
        )?;
        let report_count = require_u64_at(path, family, &["report_count"], true)?;
        let group_count = require_u64_at(path, family, &["group_count"], true)?;
        total_reports = total_reports.saturating_add(report_count);
        total_groups = total_groups.saturating_add(group_count);
        let groups = family["groups"].as_array().ok_or_else(|| {
            anyhow!("{} regression dashboard family {id} is missing groups array", path.display())
        })?;
        if groups.len() as u64 != group_count {
            anyhow::bail!(
                "{} regression dashboard family {id} group_count does not match groups length",
                path.display()
            );
        }
        let mut family_group_reports = 0_u64;
        for group in groups {
            require_non_empty_string_at(path, group, &["group_key"])?;
            require_exact_string_at(path, group, &["evidence_family"], evidence_family)?;
            require_non_empty_string_at(path, group, &["expected_artifact_kind"])?;
            require_non_empty_string_at(path, group, &["model_id"])?;
            require_non_empty_string_at(path, group, &["model_sha256"])?;
            require_non_empty_string_at(path, group, &["tokenizer_authority"])?;
            require_exact_string_at(path, group, &["selected_backend"], APPLE_M4_CPU_NEON)?;
            require_exact_string_at(path, group, &["runtime_api"], "cpu")?;
            require_bool_at(path, group, &["fallback_used"], false)?;
            let group_report_count = require_u64_at(path, group, &["report_count"], true)?;
            family_group_reports = family_group_reports.saturating_add(group_report_count);
            let status = require_non_empty_string_at(path, group, &["comparison_status"])?;
            if status != "ready" && status != "insufficient_history" {
                anyhow::bail!(
                    "{} regression dashboard group comparison_status must be ready or insufficient_history",
                    path.display()
                );
            }
            require_non_empty_string_at(path, group, &["latest_report"])?;
            require_non_empty_string_at(path, group, &["regression_command"])?;
            require_bool_at(path, group, &["claim_boundary", "evidence_families_mixed"], false)?;
            require_bool_at(path, group, &["claim_boundary", "broad_model_quality_claim"], false)?;
            require_bool_at(path, group, &["claim_boundary", "broad_performance_claim"], false)?;
            require_bool_at(path, group, &["claim_boundary", "speedup_claim"], false)?;
            require_bool_at(
                path,
                group,
                &["claim_boundary", "dense_slm_evidence"],
                evidence_family == "dense_slm",
            )?;
            require_bool_at(
                path,
                group,
                &["claim_boundary", "bitnet_evidence"],
                evidence_family == "bitnet",
            )?;
        }
        if family_group_reports != report_count {
            anyhow::bail!(
                "{} regression dashboard family {id} group report totals must match report_count",
                path.display()
            );
        }
    }
    for required in [
        "dense_slm_eval_v2",
        "dense_slm_benchmark_v2",
        "bitnet_eval",
        "bitnet_benchmark",
        "bitnet_variable_warm",
    ] {
        if !seen.contains(required) {
            anyhow::bail!(
                "{} regression dashboard is missing required family {required}",
                path.display()
            );
        }
    }
    let receipt_report_count = require_u64_at(path, receipt, &["report_count"], true)?;
    if receipt_report_count != total_reports {
        anyhow::bail!(
            "{} regression dashboard report_count must equal family report totals",
            path.display()
        );
    }
    let receipt_group_count = require_u64_at(path, receipt, &["group_count"], true)?;
    if receipt_group_count != total_groups {
        anyhow::bail!(
            "{} regression dashboard group_count must equal family group totals",
            path.display()
        );
    }
    Ok((Some(0), Some(0)))
}

fn validate_slm_eval_summary_receipt(
    path: &Path,
    receipt: &serde_json::Value,
) -> Result<(Option<usize>, Option<usize>)> {
    require_exact_string_at(path, receipt, &["schema_version"], "1.0.0")?;
    require_exact_string_at(path, receipt, &["artifact_kind"], "apple_m4_slm_eval_summary")?;
    require_exact_string_at(path, receipt, &["machine_id"], "apple-m4-mac-mini")?;
    require_non_empty_string_at(path, receipt, &["model_id"])?;

    require_non_empty_string_at(path, receipt, &["model", "repo"])?;
    require_non_empty_string_at(path, receipt, &["model", "file"])?;
    let model_sha = require_non_empty_string_at(path, receipt, &["model", "sha256"])?;
    if !is_sha256_hex(model_sha) {
        anyhow::bail!(
            "{} SLM eval summary model.sha256 must be a 64-character SHA256 hex digest",
            path.display()
        );
    }
    let model_family = require_non_empty_string_at(path, receipt, &["model", "family"])?;
    if model_family.eq_ignore_ascii_case("bitnet") {
        anyhow::bail!(
            "{} SLM eval summary is dense-SLM only and must not record a BitNet model family",
            path.display()
        );
    }
    require_non_empty_string_at(path, receipt, &["model", "architecture"])?;
    require_non_empty_string_at(path, receipt, &["model", "quantization"])?;

    require_non_empty_string_at(path, receipt, &["tokenizer", "source"])?;
    require_non_empty_string_at(path, receipt, &["tokenizer", "authority"])?;
    require_non_empty_string_at(path, receipt, &["tokenizer", "pretokenizer_authority"])?;
    require_bool_at(path, receipt, &["tokenizer", "strict"], true)?;
    require_non_empty_string_at(path, receipt, &["prompt_template"])?;

    require_non_empty_string_at(path, receipt, &["corpus", "name"])?;
    require_u64_at(path, receipt, &["corpus", "seed"], false)?;
    let corpus_case_count = require_u64_at(path, receipt, &["corpus", "case_count"], true)?;

    let cases_total = require_u64_at(path, receipt, &["accuracy", "cases_total"], true)?;
    let cases_scored = require_u64_at(path, receipt, &["accuracy", "cases_scored"], true)?;
    let cases_passed = require_u64_at(path, receipt, &["accuracy", "cases_passed"], false)?;
    if cases_total != corpus_case_count {
        anyhow::bail!(
            "{} SLM eval summary accuracy.cases_total must match corpus.case_count",
            path.display()
        );
    }
    if cases_scored > cases_total {
        anyhow::bail!(
            "{} SLM eval summary accuracy.cases_scored must not exceed cases_total",
            path.display()
        );
    }
    if cases_passed > cases_scored {
        anyhow::bail!(
            "{} SLM eval summary accuracy.cases_passed must not exceed cases_scored",
            path.display()
        );
    }
    for field in [
        "exact_match",
        "normalized_match",
        "json_schema_pass",
        "numeric_tolerance_pass",
        "required_keywords_pass",
        "forbidden_tokens_pass",
    ] {
        require_unit_rate_at(path, receipt, &["accuracy", field])?;
    }

    require_bool_at(path, receipt, &["evidence", "generated_text_recorded"], true)?;
    require_bool_at(path, receipt, &["evidence", "generated_token_ids_recorded"], true)?;
    require_non_empty_string_array_at(path, receipt, &["evidence", "case_receipts"])?;
    require_non_empty_string_at(path, receipt, &["evidence", "source_answer_corpus_receipt"])?;
    let generated_tokens_total =
        require_u64_at(path, receipt, &["evidence", "generated_tokens_total"], true)?;

    for field in [
        "cold_load_ms_p50",
        "tokenizer_load_ms_p50",
        "prompt_tokenize_ms_p50",
        "prefill_ms_p50",
        "ttft_ms_p50",
        "ttft_ms_p90",
        "input_tok_s_p50",
        "output_tok_s_p50",
        "decode_tok_s_p50",
        "sampling_ms_per_token_p50",
        "total_wall_ms_p50",
    ] {
        require_positive_number_at(path, receipt, &["speed", field])?;
    }
    let ttft_p50 = require_positive_number_at(path, receipt, &["speed", "ttft_ms_p50"])?;
    let ttft_p90 = require_positive_number_at(path, receipt, &["speed", "ttft_ms_p90"])?;
    if ttft_p90 < ttft_p50 {
        anyhow::bail!(
            "{} SLM eval summary speed.ttft_ms_p90 must be greater than or equal to speed.ttft_ms_p50",
            path.display()
        );
    }

    require_positive_number_at(path, receipt, &["memory", "peak_memory_mb"])?;
    require_non_empty_string_at(path, receipt, &["memory", "source"])?;

    require_u64_at(path, receipt, &["stability", "resident_prompts"], true)?;
    require_bool_at(path, receipt, &["stability", "quality_passed"], true)?;
    require_number_at(path, receipt, &["stability", "memory_drift_mb"], true)?;

    require_bool_at(path, receipt, &["claim_boundary", "dense_slm_only"], true)?;
    require_bool_at(path, receipt, &["claim_boundary", "bounded_seeded_corpus_only"], true)?;
    require_bool_at(path, receipt, &["claim_boundary", "broad_model_quality_claim"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "broad_performance_claim"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "bitnet_evidence"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "full_metal_inference_claimed"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "qk256_apple_claimed"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "neural_engine_claimed"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "mpsgraph_inference_claimed"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "macbook_evidence"], false)?;
    require_bool_at(path, receipt, &["claim_boundary", "speedup_claim"], false)?;

    Ok((Some(cases_total as usize), Some(generated_tokens_total as usize)))
}

fn validate_bitnet_eval_answer_corpus_receipt(
    path: &Path,
    receipt: &serde_json::Value,
) -> Result<(Option<usize>, Option<usize>)> {
    require_exact_string_at(path, receipt, &["schema_version"], "1.0.0")?;
    require_exact_string_at(
        path,
        receipt,
        &["artifact_kind"],
        "bitnet_apple_m4_local_answer_corpus",
    )?;
    require_exact_string_at(path, receipt, &["model_family"], "bitnet")?;
    require_exact_string_at(path, receipt, &["prompt_template"], BITNET_M4_PROMPT_TEMPLATE)?;

    require_exact_string_at(path, receipt, &["model", "family"], "bitnet")?;
    require_exact_string_at(path, receipt, &["model", "sha256"], BITNET_M4_EXPECTED_MODEL_SHA256)?;
    require_exact_string_at(path, receipt, &["model", "quant_format"], "I2_S")?;
    require_bool_at(path, receipt, &["model", "answer_ready_artifact_available"], true)?;
    require_exact_string_at(path, receipt, &["model", "answer_ready", "state"], "answer_ready")?;

    require_exact_string_at(
        path,
        receipt,
        &["tokenizer", "authority", "source"],
        "external_tokenizer_json",
    )?;
    require_exact_string_at(
        path,
        receipt,
        &["tokenizer", "authority", "sha256"],
        BITNET_M4_EXPECTED_TOKENIZER_SHA256,
    )?;
    require_exact_string_at(path, receipt, &["tokenizer", "authority", "ggml_pre"], "llama-bpe")?;
    require_bool_at(path, receipt, &["tokenizer", "strict"], true)?;

    require_exact_string_at(
        path,
        receipt,
        &["corpus", "name"],
        "apple-m4-bitnet-eval-seeded-corpus",
    )?;
    let corpus_case_count = require_u64_at(path, receipt, &["corpus", "case_count"], true)?;
    let quality_total = require_u64_at(path, receipt, &["quality_summary", "total"], true)?;
    let quality_passed = require_u64_at(path, receipt, &["quality_summary", "passed"], false)?;
    let quality_failed = require_u64_at(path, receipt, &["quality_summary", "failed"], false)?;
    let quality_timeout = require_u64_at(path, receipt, &["quality_summary", "timeout"], false)?;
    let quality_not_run = require_u64_at(path, receipt, &["quality_summary", "not_run"], false)?;
    if quality_total != corpus_case_count {
        anyhow::bail!(
            "{} BitNet eval receipt quality_summary.total must match corpus.case_count",
            path.display()
        );
    }
    if quality_passed + quality_failed + quality_timeout + quality_not_run != quality_total {
        anyhow::bail!(
            "{} BitNet eval receipt quality_summary counts must sum to total",
            path.display()
        );
    }
    if quality_timeout != 0 || quality_not_run != 0 {
        anyhow::bail!(
            "{} BitNet eval receipt must complete every case without timeout or not_run rows",
            path.display()
        );
    }

    require_bool_at(path, receipt, &["scoring_summary", "enabled"], true)?;
    let scoring_total = require_u64_at(path, receipt, &["scoring_summary", "total"], true)?;
    if scoring_total != quality_total {
        anyhow::bail!(
            "{} BitNet eval receipt scoring_summary.total must match quality_summary.total",
            path.display()
        );
    }

    let task_family_summary =
        json_value_at(receipt, &["task_family_summary"]).as_object().ok_or_else(|| {
            anyhow!("{} BitNet eval receipt is missing task_family_summary object", path.display())
        })?;
    if task_family_summary.is_empty() {
        anyhow::bail!(
            "{} BitNet eval receipt task_family_summary must not be empty",
            path.display()
        );
    }
    let mut task_family_total = 0_u64;
    for (family, summary) in task_family_summary {
        let total = summary["total"].as_u64().ok_or_else(|| {
            anyhow!(
                "{} BitNet eval receipt task_family_summary.{family}.total is missing",
                path.display()
            )
        })?;
        let passed = summary["passed"].as_u64().unwrap_or(0);
        let failed = summary["failed"].as_u64().unwrap_or(0);
        let timeout = summary["timeout"].as_u64().unwrap_or(0);
        let not_run = summary["not_run"].as_u64().unwrap_or(0);
        if passed + failed + timeout + not_run != total {
            anyhow::bail!(
                "{} BitNet eval receipt task_family_summary.{family} counts must sum to total",
                path.display()
            );
        }
        if summary["scoring"]["enabled"].as_bool() != Some(true) {
            anyhow::bail!(
                "{} BitNet eval receipt task_family_summary.{family}.scoring.enabled must be true",
                path.display()
            );
        }
        task_family_total = task_family_total.saturating_add(total);
    }
    if task_family_total != quality_total {
        anyhow::bail!(
            "{} BitNet eval receipt task-family totals must match quality_summary.total",
            path.display()
        );
    }

    require_exact_string_at(
        path,
        receipt,
        &["reference_comparison", "schema"],
        "bitnet_reference_vs_rust_v1",
    )?;
    let reference_total =
        require_u64_at(path, receipt, &["reference_comparison", "summary", "total"], true)?;
    if reference_total != quality_total {
        anyhow::bail!(
            "{} BitNet eval receipt reference_comparison.summary.total must match quality_summary.total",
            path.display()
        );
    }
    for claim in [
        "dense_slm_evidence_used",
        "chat_enabled",
        "serve_enabled",
        "performance_claimed",
        "full_metal_inference_claimed",
        "qk256_apple_claimed",
        "neural_engine_claimed",
        "mpsgraph_claimed",
        "broad_apple_silicon_claimed",
        "runtime_accuracy_claimed",
    ] {
        require_bool_at(path, receipt, &["reference_comparison", "claim_boundary", claim], false)?;
    }
    for claim in [
        "full_metal_inference_claimed",
        "neural_engine_claimed",
        "qk256_apple_claimed",
        "broad_performance_claimed",
    ] {
        require_bool_at(path, receipt, &["claim_boundary", claim], false)?;
    }

    let cases = receipt["cases"]
        .as_array()
        .ok_or_else(|| anyhow!("{} BitNet eval receipt cases must be an array", path.display()))?;
    if cases.len() as u64 != quality_total {
        anyhow::bail!(
            "{} BitNet eval receipt cases length must match quality_summary.total",
            path.display()
        );
    }

    let mut generated_tokens = 0_u64;
    for (index, case) in cases.iter().enumerate() {
        let case_label = case["id"].as_str().unwrap_or("<unknown>");
        match case["status"].as_str() {
            Some("passed" | "quality_failed") => {}
            observed => {
                anyhow::bail!(
                    "{} BitNet eval receipt case {index} ({case_label}) must be passed or quality_failed, got {observed:?}",
                    path.display()
                );
            }
        }
        if case["answer"].as_str().unwrap_or_default().trim().is_empty()
            || case["quality"]["non_empty_answer"].as_bool() != Some(true)
            || case["quality"]["printable_utf8"].as_bool() != Some(true)
            || case["quality"]["no_replacement_chars"].as_bool() != Some(true)
        {
            anyhow::bail!(
                "{} BitNet eval receipt case {index} ({case_label}) must record valid non-empty UTF-8 answer text",
                path.display()
            );
        }
        if case["backend"]["requested_backend"].as_str() != Some(APPLE_M4_CPU_NEON)
            || case["backend"]["selected_backend"].as_str() != Some(APPLE_M4_CPU_NEON)
            || case["backend"]["runtime_api"].as_str() != Some("cpu")
            || case["backend"]["fallback_used"].as_bool() != Some(false)
        {
            anyhow::bail!(
                "{} BitNet eval receipt case {index} ({case_label}) backend/fallback fields are not strict apple-m4-cpu-neon",
                path.display()
            );
        }
        if case["model"]["sha256"].as_str() != Some(BITNET_M4_EXPECTED_MODEL_SHA256)
            || case["model"]["family"].as_str() != Some("bitnet")
            || case["loader"]["mode"].as_str() != Some("real_gguf")
        {
            anyhow::bail!(
                "{} BitNet eval receipt case {index} ({case_label}) model/loader identity is not strict BitNet real GGUF",
                path.display()
            );
        }
        if case["tokenizer"]["strict"].as_bool() != Some(true)
            || case["tokenizer"]["pretokenizer_authority"].as_str() != Some("llama-bpe")
        {
            anyhow::bail!(
                "{} BitNet eval receipt case {index} ({case_label}) tokenizer authority is not strict llama-bpe",
                path.display()
            );
        }
        if case["prompt_template"].as_str() != Some(BITNET_M4_PROMPT_TEMPLATE)
            || case["prompt"]["template_family"].as_str() != Some(BITNET_M4_PROMPT_TEMPLATE)
            || case["prompt_prefill"]["exercised"].as_bool() != Some(true)
        {
            anyhow::bail!(
                "{} BitNet eval receipt case {index} ({case_label}) prompt/template/prefill evidence is incomplete",
                path.display()
            );
        }
        let token_ids = case["token_ids"]["generated"].as_array().ok_or_else(|| {
            anyhow!(
                "{} BitNet eval receipt case {index} ({case_label}) is missing generated token IDs",
                path.display()
            )
        })?;
        let case_generated_tokens = case["tokens"]["generated"].as_u64().unwrap_or(0);
        if token_ids.is_empty() || case_generated_tokens == 0 {
            anyhow::bail!(
                "{} BitNet eval receipt case {index} ({case_label}) has no generated token IDs",
                path.display()
            );
        }
        if case_generated_tokens as usize != token_ids.len() {
            anyhow::bail!(
                "{} BitNet eval receipt case {index} ({case_label}) token count does not match generated token IDs",
                path.display()
            );
        }
        if case["reference_comparison"]["schema"].as_str() != Some("bitnet_reference_vs_rust_v1") {
            anyhow::bail!(
                "{} BitNet eval receipt case {index} ({case_label}) is missing reference comparison schema",
                path.display()
            );
        }
        for path_name in [&["timing", "decode_total_ms"][..], &["latency", "total_ms"][..]] {
            if !json_number_at(case, path_name) {
                anyhow::bail!(
                    "{} BitNet eval receipt case {index} ({case_label}) is missing timing/latency fields",
                    path.display()
                );
            }
        }
        generated_tokens = generated_tokens.saturating_add(case_generated_tokens);
    }

    Ok((Some(quality_total as usize), Some(generated_tokens as usize)))
}

fn validate_bitnet_benchmark_v1_receipt(
    path: &Path,
    receipt: &serde_json::Value,
) -> Result<(Option<usize>, Option<usize>)> {
    require_exact_string_at(path, receipt, &["schema_version"], "1.0.0")?;
    require_exact_string_at(path, receipt, &["artifact_kind"], "bitnet_apple_m4_benchmark_v1")?;
    require_exact_string_at(path, receipt, &["benchmark_set"], "bitnet-one-shot-fixed-warm-v1")?;
    require_bool_at(path, receipt, &["build", "release_mode"], true)?;

    require_exact_string_at(path, receipt, &["model", "id"], BITNET_M4_MODEL_ID)?;
    require_exact_string_at(path, receipt, &["model", "family"], "bitnet")?;
    require_exact_string_at(path, receipt, &["model", "sha256"], BITNET_M4_EXPECTED_MODEL_SHA256)?;
    require_exact_string_at(path, receipt, &["model", "quantization"], "I2_S")?;
    require_bool_at(path, receipt, &["model", "answer_ready_artifact_available"], true)?;
    require_exact_string_at(path, receipt, &["model", "answer_ready", "state"], "answer_ready")?;

    require_exact_string_at(path, receipt, &["tokenizer", "source"], "external_tokenizer_json")?;
    require_exact_string_at(
        path,
        receipt,
        &["tokenizer", "sha256"],
        BITNET_M4_EXPECTED_TOKENIZER_SHA256,
    )?;
    require_exact_string_at(path, receipt, &["tokenizer", "authority"], "llama-bpe")?;
    require_exact_string_at(path, receipt, &["tokenizer", "pretokenizer_authority"], "llama-bpe")?;
    require_bool_at(path, receipt, &["tokenizer", "strict"], true)?;
    require_exact_string_at(path, receipt, &["prompt_template"], BITNET_M4_PROMPT_TEMPLATE)?;

    for field in [
        "cold_load_ms",
        "model_load_ms",
        "tokenizer_load_ms",
        "prompt_tokenize_ms",
        "prefill_ms",
        "ttft_ms",
        "decode_total_ms",
        "sampling_ms_per_token",
        "input_tok_s",
        "output_tok_s",
        "decode_tok_s",
        "total_wall_ms",
    ] {
        validate_benchmark_percentiles(path, receipt, &["speed"], field, true)?;
    }
    validate_benchmark_percentiles(path, receipt, &["memory"], "peak_memory_mb", true)?;
    validate_benchmark_percentiles(path, receipt, &["memory"], "memory_drift_mb", false)?;
    require_non_empty_string_at(path, receipt, &["memory", "source"])?;

    require_bool_at(path, receipt, &["timeout_boundary", "enforced"], false)?;
    require_bool_at(path, receipt, &["timeout_boundary", "reached"], false)?;
    require_exact_string_at(path, receipt, &["timeout_boundary", "status"], "not_reached")?;

    require_bool_at(path, receipt, &["evidence", "generated_text_recorded"], true)?;
    require_bool_at(path, receipt, &["evidence", "generated_token_ids_recorded"], true)?;
    require_non_empty_string_at(path, receipt, &["evidence", "one_shot_receipt"])?;
    require_non_empty_string_at(path, receipt, &["evidence", "warm_session_receipt"])?;
    require_non_empty_string_array_at(path, receipt, &["evidence", "warm_prompt_receipts"])?;
    require_non_empty_string_array_at(path, receipt, &["evidence", "operator_commands"])?;

    require_bool_at(path, receipt, &["mac_claim_boundary", "bitnet_benchmark"], true)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "one_shot_mac_ask"], true)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "fixed_warm_session"], true)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "accepted_i2s_artifact_only"], true)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "dense_slm_evidence_used"], false)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "chat_enabled"], false)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "serve_enabled"], false)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "bitnet_quality_claimed"], false)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "broad_model_quality_claim"], false)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "broad_performance_claim"], false)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "speedup_claim"], false)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "full_metal_inference_claimed"], false)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "mpsgraph_inference_claimed"], false)?;
    require_bool_at(
        path,
        receipt,
        &["mac_claim_boundary", "neural_engine_execution_claimed"],
        false,
    )?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "qk256_apple_claimed"], false)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "macbook_evidence"], false)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "broad_apple_silicon_claimed"], false)?;

    let paths = json_value_at(receipt, &["paths"]).as_object().ok_or_else(|| {
        anyhow!("{} BitNet benchmark v1 summary is missing paths object", path.display())
    })?;
    let one_shot = paths.get("one_shot").ok_or_else(|| {
        anyhow!("{} BitNet benchmark v1 summary is missing one_shot path", path.display())
    })?;
    let fixed_warm = paths.get("fixed_warm").ok_or_else(|| {
        anyhow!("{} BitNet benchmark v1 summary is missing fixed_warm path", path.display())
    })?;
    validate_bitnet_benchmark_path_summary(path, one_shot, "one_shot_mac_ask", "mac ask", false)?;
    validate_bitnet_benchmark_path_summary(
        path,
        fixed_warm,
        "fixed_warm_session",
        "mac bitnet-warm",
        true,
    )?;

    let prompt_count_total = one_shot["prompt_count"].as_u64().unwrap_or_default()
        + fixed_warm["prompt_count"].as_u64().unwrap_or_default();
    let generated_tokens_total = one_shot["generated_tokens"].as_u64().unwrap_or_default()
        + fixed_warm["generated_tokens"].as_u64().unwrap_or_default();
    if receipt["prompt_count"].as_u64() != Some(prompt_count_total) {
        anyhow::bail!(
            "{} BitNet benchmark v1 prompt_count must equal one-shot plus fixed-warm prompt counts",
            path.display()
        );
    }
    if receipt["generated_tokens"].as_u64() != Some(generated_tokens_total) {
        anyhow::bail!(
            "{} BitNet benchmark v1 generated_tokens must equal one-shot plus fixed-warm generated tokens",
            path.display()
        );
    }

    Ok((Some(prompt_count_total as usize), Some(generated_tokens_total as usize)))
}

fn validate_bitnet_benchmark_path_summary(
    receipt_path: &Path,
    summary: &serde_json::Value,
    expected_path_id: &str,
    expected_operator_command: &str,
    require_resident_reuse: bool,
) -> Result<()> {
    require_exact_string_at(receipt_path, summary, &["path_id"], expected_path_id)?;
    require_exact_string_at(
        receipt_path,
        summary,
        &["operator_command"],
        expected_operator_command,
    )?;
    require_non_empty_string_at(receipt_path, summary, &["receipt_path"])?;
    require_u64_at(receipt_path, summary, &["prompt_count"], true)?;
    require_u64_at(receipt_path, summary, &["generated_tokens"], true)?;
    require_bool_at(receipt_path, summary, &["model_loaded_once"], true)?;
    require_bool_at(receipt_path, summary, &["tokenizer_loaded_once"], true)?;
    require_bool_at(receipt_path, summary, &["quality_passed"], true)?;
    if require_resident_reuse {
        require_exact_string_at(receipt_path, summary, &["reuse_scope"], "resident_session")?;
    }

    validate_benchmark_stat_object(receipt_path, summary, &["prompt_tokens"], true)?;
    validate_benchmark_stat_object(receipt_path, summary, &["output_tokens"], true)?;
    for field in [
        "model_load_ms",
        "tokenizer_load_ms",
        "prompt_tokenize_ms",
        "prefill_ms",
        "time_to_first_token_ms",
        "decode_total_ms",
        "sampling_ms_per_token",
        "total_wall_ms",
    ] {
        validate_benchmark_stat_object(receipt_path, summary, &["timing", field], true)?;
    }
    for field in ["input_tokens_per_second", "output_tokens_per_second", "decode_tokens_per_second"]
    {
        validate_benchmark_stat_object(receipt_path, summary, &["throughput", field], true)?;
    }
    validate_benchmark_stat_object(receipt_path, summary, &["memory", "peak_memory_mb"], true)?;
    validate_benchmark_stat_object(receipt_path, summary, &["memory", "memory_drift_mb"], false)?;
    require_non_empty_string_at(receipt_path, summary, &["memory", "source"])?;
    require_bool_at(receipt_path, summary, &["timeout_boundary", "enforced"], false)?;
    require_bool_at(receipt_path, summary, &["timeout_boundary", "reached"], false)?;
    require_exact_string_at(receipt_path, summary, &["timeout_boundary", "status"], "not_reached")?;
    require_bool_at(receipt_path, summary, &["claim_boundary", "chat_enabled"], false)?;
    require_bool_at(receipt_path, summary, &["claim_boundary", "serve_enabled"], false)?;
    require_bool_at(receipt_path, summary, &["claim_boundary", "broad_performance_claim"], false)?;
    require_bool_at(receipt_path, summary, &["claim_boundary", "speedup_claim"], false)?;
    Ok(())
}

fn validate_slm_benchmark_v2_receipt(
    path: &Path,
    receipt: &serde_json::Value,
) -> Result<(Option<usize>, Option<usize>)> {
    let schema_version = require_non_empty_string_at(path, receipt, &["schema_version"])?;
    if schema_version != "1.0.0" && schema_version != "1.1.0" {
        anyhow::bail!(
            "{} SLM benchmark v2 schema_version must be \"1.0.0\" or \"1.1.0\", got {schema_version:?}",
            path.display()
        );
    }
    let requires_explicit_contract = schema_version == "1.1.0";
    require_exact_string_at(path, receipt, &["artifact_kind"], "apple_m4_slm_benchmark_v2")?;
    require_exact_string_at(path, receipt, &["profile_set"], "slm-benchmark-v2")?;
    require_bool_at(path, receipt, &["build", "release_mode"], true)?;

    require_non_empty_string_at(path, receipt, &["model_cache", "id"])?;
    require_non_empty_string_at(path, receipt, &["model_cache", "sha256"])?;
    require_non_empty_string_at(path, receipt, &["model_cache", "architecture"])?;
    require_non_empty_string_at(path, receipt, &["model_cache", "quantization"])?;
    require_non_empty_string_at(path, receipt, &["model_cache", "tokenizer_pre"])?;

    let profiles = receipt["profiles"].as_array().ok_or_else(|| {
        anyhow!("{} SLM benchmark v2 summary is missing profiles", path.display())
    })?;
    if profiles.is_empty() {
        anyhow::bail!(
            "{} SLM benchmark v2 summary must include at least one profile",
            path.display()
        );
    }

    let aggregate_speed_metrics = if requires_explicit_contract {
        M4_SLM_BENCHMARK_V2_AGGREGATE_SPEED_METRICS
    } else {
        M4_SLM_BENCHMARK_V2_LEGACY_AGGREGATE_SPEED_METRICS
    };
    for &field in aggregate_speed_metrics {
        validate_benchmark_percentiles(path, receipt, &["speed"], field, true)?;
    }
    validate_benchmark_percentiles(path, receipt, &["memory"], "peak_memory_mb", true)?;
    validate_benchmark_percentiles(path, receipt, &["memory"], "memory_drift_mb", false)?;
    require_non_empty_string_at(path, receipt, &["memory", "source"])?;
    if requires_explicit_contract {
        require_exact_string_at(
            path,
            receipt,
            &["benchmark_contract", "contract_version"],
            "1.1.0",
        )?;
        require_string_array_equals(
            path,
            receipt,
            &["benchmark_contract", "supported_profiles"],
            M4_SLM_BENCHMARK_V2_PROFILES,
        )?;
        require_string_array_equals(
            path,
            receipt,
            &["benchmark_contract", "required_metrics", "timing"],
            M4_SLM_BENCHMARK_V2_TIMING_METRICS,
        )?;
        require_string_array_equals(
            path,
            receipt,
            &["benchmark_contract", "required_metrics", "throughput"],
            M4_SLM_BENCHMARK_V2_THROUGHPUT_METRICS,
        )?;
        require_string_array_equals(
            path,
            receipt,
            &["benchmark_contract", "required_metrics", "memory"],
            M4_SLM_BENCHMARK_V2_MEMORY_METRICS,
        )?;
        require_string_array_equals(
            path,
            receipt,
            &["benchmark_contract", "required_metrics", "aggregate_speed"],
            M4_SLM_BENCHMARK_V2_AGGREGATE_SPEED_METRICS,
        )?;
    }

    require_bool_at(path, receipt, &["evidence", "generated_text_recorded"], true)?;
    require_bool_at(path, receipt, &["evidence", "generated_token_ids_recorded"], true)?;
    require_non_empty_string_array_at(path, receipt, &["evidence", "profile_receipts"])?;
    require_exact_string_at(path, receipt, &["evidence", "operator_command"], "mac benchmark")?;

    require_bool_at(path, receipt, &["mac_claim_boundary", "dense_slm_only"], true)?;
    require_bool_at(
        path,
        receipt,
        &["mac_claim_boundary", "bounded_benchmark_profiles_only"],
        true,
    )?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "broad_model_quality_claim"], false)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "broad_performance_claim"], false)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "speedup_claim"], false)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "bitnet_quality_claimed"], false)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "full_metal_inference_claimed"], false)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "mpsgraph_inference_claimed"], false)?;
    require_bool_at(
        path,
        receipt,
        &["mac_claim_boundary", "neural_engine_execution_claimed"],
        false,
    )?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "qk256_apple_claimed"], false)?;
    require_bool_at(path, receipt, &["mac_claim_boundary", "macbook_evidence"], false)?;

    let mut prompt_count_total = 0u64;
    let mut generated_tokens_total = 0u64;
    let mut seen_profiles = std::collections::BTreeSet::new();
    let mut observed_profile_ids = Vec::with_capacity(profiles.len());
    for profile in profiles {
        let profile_id = require_non_empty_string_at(path, profile, &["profile_id"])?;
        if !M4_SLM_BENCHMARK_V2_PROFILES.contains(&profile_id) {
            anyhow::bail!(
                "{} SLM benchmark v2 profile_id {profile_id:?} is not part of the v2 profile contract",
                path.display()
            );
        }
        if !seen_profiles.insert(profile_id.to_string()) {
            anyhow::bail!(
                "{} SLM benchmark v2 profile {profile_id:?} is duplicated",
                path.display()
            );
        }
        observed_profile_ids.push(profile_id.to_string());
        require_non_empty_string_at(path, profile, &["receipt_path"])?;
        let prompt_count = require_u64_at(path, profile, &["prompt_count"], true)?;
        let generated_tokens = require_u64_at(path, profile, &["generated_tokens"], true)?;
        require_bool_at(path, profile, &["quality_passed"], true)?;
        require_bool_at(path, profile, &["model_loaded_once"], true)?;
        require_bool_at(path, profile, &["tokenizer_loaded_once"], true)?;
        require_exact_string_at(path, profile, &["reuse_scope"], "resident_session")?;

        validate_benchmark_stat_object(path, profile, &["prompt_tokens"], true)?;
        validate_benchmark_stat_object(path, profile, &["output_tokens"], true)?;
        for &field in M4_SLM_BENCHMARK_V2_TIMING_METRICS {
            validate_benchmark_stat_object(path, profile, &["timing", field], true)?;
        }
        for &field in M4_SLM_BENCHMARK_V2_THROUGHPUT_METRICS {
            validate_benchmark_stat_object(path, profile, &["throughput", field], true)?;
        }
        validate_benchmark_stat_object(path, profile, &["memory", "peak_memory_mb"], true)?;
        validate_benchmark_stat_object(path, profile, &["memory", "memory_drift_mb"], false)?;
        require_non_empty_string_at(path, profile, &["memory", "source"])?;

        prompt_count_total += prompt_count;
        generated_tokens_total += generated_tokens;
    }
    if requires_explicit_contract {
        let profiles_required =
            json_value_at(receipt, &["profiles_required"]).as_array().ok_or_else(|| {
                anyhow!("{} SLM benchmark v2 summary is missing profiles_required", path.display())
            })?;
        let required_profile_ids = profiles_required
            .iter()
            .map(|value| {
                value.as_str().map(str::to_string).ok_or_else(|| {
                    anyhow!(
                        "{} SLM benchmark v2 profiles_required must contain only strings",
                        path.display()
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if required_profile_ids != observed_profile_ids {
            anyhow::bail!(
                "{} SLM benchmark v2 profiles_required must match profiles order {:?}, got {:?}",
                path.display(),
                observed_profile_ids,
                required_profile_ids
            );
        }
    }

    if receipt["prompt_count"].as_u64() != Some(prompt_count_total) {
        anyhow::bail!(
            "{} SLM benchmark v2 prompt_count must equal the sum of profile prompt counts",
            path.display()
        );
    }
    if receipt["generated_tokens"].as_u64() != Some(generated_tokens_total) {
        anyhow::bail!(
            "{} SLM benchmark v2 generated_tokens must equal the sum of profile generated tokens",
            path.display()
        );
    }

    Ok((Some(prompt_count_total as usize), Some(generated_tokens_total as usize)))
}

fn validate_benchmark_percentiles(
    path: &Path,
    receipt: &serde_json::Value,
    base: &[&str],
    metric: &str,
    positive: bool,
) -> Result<()> {
    let p50_key = format!("{metric}_p50");
    let p90_key = format!("{metric}_p90");
    let p99_key = format!("{metric}_p99");
    let mut p50_path = base.to_vec();
    p50_path.push(p50_key.as_str());
    let mut p90_path = base.to_vec();
    p90_path.push(p90_key.as_str());
    let mut p99_path = base.to_vec();
    p99_path.push(p99_key.as_str());
    let p50 = if positive {
        require_positive_number_at(path, receipt, &p50_path)?
    } else {
        require_number_at(path, receipt, &p50_path, true)?
    };
    let p90 = if positive {
        require_positive_number_at(path, receipt, &p90_path)?
    } else {
        require_number_at(path, receipt, &p90_path, true)?
    };
    let p99 = if positive {
        require_positive_number_at(path, receipt, &p99_path)?
    } else {
        require_number_at(path, receipt, &p99_path, true)?
    };
    if p90 < p50 || p99 < p90 {
        anyhow::bail!(
            "{} SLM benchmark v2 metric {metric} must have p50 <= p90 <= p99",
            path.display()
        );
    }
    Ok(())
}

fn validate_benchmark_stat_object(
    path: &Path,
    receipt: &serde_json::Value,
    segments: &[&str],
    positive: bool,
) -> Result<()> {
    let value = json_value_at(receipt, segments);
    if !value.is_object() {
        anyhow::bail!(
            "{} SLM benchmark v2 {} must be a stats object",
            path.display(),
            json_path_label(segments)
        );
    }
    require_u64_at(path, receipt, &[segments, &["count"]].concat(), true)?;
    let p50 = if positive {
        require_positive_number_at(path, receipt, &[segments, &["p50"]].concat())?
    } else {
        require_number_at(path, receipt, &[segments, &["p50"]].concat(), true)?
    };
    let p90 = if positive {
        require_positive_number_at(path, receipt, &[segments, &["p90"]].concat())?
    } else {
        require_number_at(path, receipt, &[segments, &["p90"]].concat(), true)?
    };
    let p99 = if positive {
        require_positive_number_at(path, receipt, &[segments, &["p99"]].concat())?
    } else {
        require_number_at(path, receipt, &[segments, &["p99"]].concat(), true)?
    };
    if p90 < p50 || p99 < p90 {
        anyhow::bail!(
            "{} SLM benchmark v2 {} must have p50 <= p90 <= p99",
            path.display(),
            json_path_label(segments)
        );
    }
    Ok(())
}

fn validate_profile_set_receipt(
    path: &Path,
    receipt: &serde_json::Value,
    requested_backend: &str,
    artifact_kind: &str,
) -> Result<(Option<usize>, Option<usize>)> {
    let required = match artifact_kind {
        "apple_m4_slm_operator_profiles" | "apple_m3_air_slm_operator_profiles" => {
            &[("warm_16", 16_u64), ("warm_32", 32_u64), ("warm_64", 64_u64)][..]
        }
        "apple_m4_slm_performance_profiles" | "apple_m3_air_slm_performance_profiles" => {
            &[("warm_16", 16_u64), ("warm_32", 32_u64), ("warm_64", 64_u64), ("warm_128", 128_u64)]
                [..]
        }
        _ => {
            anyhow::bail!("{} has unsupported profile receipt kind {artifact_kind}", path.display())
        }
    };
    if receipt["operator_thresholds"]["cold_load_separated"] != true {
        anyhow::bail!("{} profile summary must separate cold load timing", path.display());
    }
    if receipt["operator_thresholds"]["model_tokenizer_reuse_visible"] != true {
        anyhow::bail!("{} profile summary must record model/tokenizer reuse", path.display());
    }
    if receipt["operator_thresholds"]["model_tokenizer_reuse_visible_per_profile"] != true {
        anyhow::bail!(
            "{} profile summary must scope model/tokenizer reuse visibility per profile",
            path.display()
        );
    }
    if receipt["operator_thresholds"]["thresholds_are_claim_bounds_not_speed_guarantees"] != true {
        anyhow::bail!(
            "{} profile summary must record that thresholds are claim bounds, not speed guarantees",
            path.display()
        );
    }
    if receipt["operator_thresholds"]["profiles_loaded_independently"] != true {
        anyhow::bail!(
            "{} profile summary must disclose independent per-token-budget warm-session runs",
            path.display()
        );
    }
    let profiles = receipt["profiles"]
        .as_array()
        .ok_or_else(|| anyhow!("{} profile summary is missing profiles", path.display()))?;
    if profiles.len() != required.len() {
        anyhow::bail!(
            "{} profile summary must contain exactly {}",
            path.display(),
            required.iter().map(|(profile, _)| *profile).collect::<Vec<_>>().join(", ")
        );
    }
    if receipt["operator_thresholds"]["profile_set_model_loads"].as_u64()
        != Some(required.len() as u64)
    {
        anyhow::bail!(
            "{} profile summary must record profile_set_model_loads={}",
            path.display(),
            required.len()
        );
    }
    if artifact_kind == "apple_m4_slm_performance_profiles"
        || artifact_kind == "apple_m3_air_slm_performance_profiles"
    {
        if receipt["profile_set"].as_str() != Some("performance") {
            anyhow::bail!(
                "{} performance summary must record profile_set=performance",
                path.display()
            );
        }
        if receipt["build"]["release_mode"].as_bool() != Some(true)
            || receipt["performance_baseline"]["release_mode_observed"].as_bool() != Some(true)
        {
            anyhow::bail!(
                "{} performance summary must be recorded from a release build",
                path.display()
            );
        }
        if receipt["performance_baseline"]["warm_128_included"].as_bool() != Some(true) {
            anyhow::bail!("{} performance summary must include warm_128", path.display());
        }
    }
    let allocation_audit_enabled =
        receipt["allocation_audit"]["enabled"].as_bool().unwrap_or(false);
    if allocation_audit_enabled {
        if receipt["allocation_audit"]["method"].as_str()
            != Some("process_global_allocator_counter_delta")
        {
            anyhow::bail!(
                "{} allocation audit must record process-global allocator counter deltas",
                path.display()
            );
        }
        if receipt["allocation_audit"]["optimization_deferred"].as_bool() != Some(true) {
            anyhow::bail!(
                "{} allocation audit must record that optimization is deferred",
                path.display()
            );
        }
        if receipt["allocation_audit"]["ranked_hotspots"]
            .as_array()
            .is_none_or(|hotspots| hotspots.is_empty())
        {
            anyhow::bail!("{} allocation audit must rank hotspots", path.display());
        }
        let expected_scope = profile_set_allocation_scope(requested_backend);
        if receipt["allocation_audit"]["scope"].as_str() != Some(expected_scope) {
            anyhow::bail!("{} allocation audit scope must be {expected_scope:?}", path.display());
        }
    }
    for (profile_id, requested_tokens) in required {
        if !profiles.iter().any(|profile| {
            profile["profile_id"] == *profile_id
                && profile["requested_max_new_tokens"].as_u64() == Some(*requested_tokens)
        }) {
            anyhow::bail!("{} profile summary is missing {profile_id}", path.display());
        }
    }
    let mut generated_total = 0usize;
    for profile in profiles {
        if profile["quality_passed"].as_bool() != Some(true) {
            anyhow::bail!("{} profile quality failed", path.display());
        }
        if profile["prompt_count"].as_u64().unwrap_or_default() == 0 {
            anyhow::bail!("{} profile records zero prompts", path.display());
        }
        let generated = profile["generated_tokens"].as_u64().unwrap_or_default();
        if generated == 0 {
            anyhow::bail!("{} profile records zero generated tokens", path.display());
        }
        if profile["model_loaded_once"].as_bool() != Some(true)
            || profile["tokenizer_loaded_once"].as_bool() != Some(true)
        {
            anyhow::bail!("{} profile does not record model/tokenizer reuse", path.display());
        }
        if profile["cold_load_separated"].as_bool() != Some(true) {
            anyhow::bail!("{} profile must record cold_load_separated=true", path.display());
        }
        if profile["reuse_scope"].as_str() != Some("within_profile") {
            anyhow::bail!("{} profile must record reuse_scope=within_profile", path.display());
        }
        if profile["resident_session"]["reuse_scope"].as_str() != Some("resident_session")
            || profile["resident_session"]["session_owned_buffers"].as_bool() != Some(true)
            || profile["resident_session"]["prompt_token_buffer_reused"].as_bool() != Some(true)
            || profile["resident_session"]["generated_token_buffer_reused"].as_bool() != Some(true)
            || profile["resident_session"]["timing_buffers_reused"].as_bool() != Some(true)
        {
            anyhow::bail!(
                "{} profile must record resident-session owned buffer reuse",
                path.display()
            );
        }
        if profile["resident_session"]["kv_cache_reuse_policy"].as_str()
            != Some("recreated_per_prompt_for_prompt_isolation")
            || profile["resident_session"]["sampler_reuse_policy"].as_str()
                != Some("recreated_per_prompt_for_deterministic_prompt_independence")
        {
            anyhow::bail!("{} profile must record prompt runtime reset policies", path.display());
        }
        if allocation_audit_enabled {
            if profile["allocation_audit"]["enabled"].as_bool() != Some(true) {
                anyhow::bail!(
                    "{} profile summary must include enabled allocation audit details",
                    path.display()
                );
            }
            let expected_scope = warm_session_allocation_scope(requested_backend);
            if profile["allocation_audit"]["scope"].as_str() != Some(expected_scope) {
                anyhow::bail!(
                    "{} profile allocation audit scope must be {expected_scope:?}",
                    path.display()
                );
            }
            if profile["allocation_audit"]["ranked_hotspots"]
                .as_array()
                .is_none_or(|hotspots| hotspots.is_empty())
            {
                anyhow::bail!("{} profile allocation audit must rank hotspots", path.display());
            }
        }
        let timing = &profile["timing"];
        for field in [
            "model_load_ms",
            "tokenizer_load_ms",
            "warm_prompt_wall_ms",
            "decode_total_ms",
            "sampling_ms",
            "warm_prompt_generated_tok_s",
            "decode_generated_tok_s",
        ] {
            if timing[field].is_null() {
                anyhow::bail!(
                    "{} profile {} is missing timing.{field}",
                    path.display(),
                    profile["profile_id"].as_str().unwrap_or("<unknown>")
                );
            }
        }
        if artifact_kind == "apple_m4_slm_performance_profiles"
            || artifact_kind == "apple_m3_air_slm_performance_profiles"
        {
            for field in ["total_session_ms", "tokenize_ms", "prefill_ms", "first_token_ms"] {
                if timing[field].is_null() {
                    anyhow::bail!(
                        "{} performance profile {} is missing timing.{field}",
                        path.display(),
                        profile["profile_id"].as_str().unwrap_or("<unknown>")
                    );
                }
            }
            if profile["memory"]["peak_memory_mb"].is_null() {
                anyhow::bail!(
                    "{} performance profile {} is missing memory.peak_memory_mb",
                    path.display(),
                    profile["profile_id"].as_str().unwrap_or("<unknown>")
                );
            }
        }
        generated_total += generated as usize;
    }
    Ok((Some(profiles.len()), Some(generated_total)))
}

fn warm_session_allocation_scope(requested_backend: &str) -> &'static str {
    match requested_backend {
        APPLE_M3_AIR_CPU_NEON => "selected Apple M3 Air CPU/NEON SLM warm-session prompt hot path",
        _ => "selected Apple M4 CPU/NEON SLM warm-session prompt hot path",
    }
}

fn is_metal_phase_receipt(receipt: &serde_json::Value) -> bool {
    receipt["artifact_kind"].as_str() == Some("phase_contribution")
        && receipt["metal_phase"].is_object()
}

fn validate_metal_phase_receipt(
    path: &Path,
    receipt: &serde_json::Value,
    artifact_kind: String,
) -> Result<ReceiptCheckSummary> {
    let requested_backend = receipt_string(receipt, "requested_backend").unwrap_or_default();
    let selected_backend = receipt_string(receipt, "selected_backend").unwrap_or_default();
    let runtime_api = receipt_string(receipt, "runtime_api").unwrap_or_default();
    let fallback_used = receipt_bool(receipt, "fallback_used").unwrap_or(true);

    if requested_backend != APPLE_M4_METAL {
        anyhow::bail!(
            "{} Metal phase requested_backend must be {APPLE_M4_METAL}, got {requested_backend:?}",
            path.display()
        );
    }
    if selected_backend != APPLE_M4_METAL {
        anyhow::bail!(
            "{} Metal phase selected_backend must be {APPLE_M4_METAL}, got {selected_backend:?}",
            path.display()
        );
    }
    if runtime_api != "metal" {
        anyhow::bail!(
            "{} Metal phase runtime_api must be metal, got {runtime_api:?}",
            path.display()
        );
    }
    if fallback_used {
        anyhow::bail!("{} Metal phase records fallback_used=true", path.display());
    }
    if receipt_flag_true(receipt, "full_metal_inference_claimed")
        || receipt_flag_true(receipt, "full_metal_inference")
    {
        anyhow::bail!("{} claims full apple-m4-metal inference", path.display());
    }
    if receipt_flag_true(receipt, "neural_engine_execution_claimed") {
        anyhow::bail!("{} claims Neural Engine execution", path.display());
    }
    if receipt_flag_true(receipt, "mpsgraph_inference_claimed") {
        anyhow::bail!("{} claims MPSGraph model inference", path.display());
    }
    if receipt_flag_true(receipt, "qk256_apple_claimed") {
        anyhow::bail!("{} claims QK256 on Apple Silicon", path.display());
    }
    if receipt_flag_true(receipt, "bitnet_quality_claimed") {
        anyhow::bail!("{} claims BitNet local-answer quality", path.display());
    }
    if receipt_flag_true(receipt, "broad_performance_claim")
        || receipt_flag_true(receipt, "speedup_claim")
    {
        anyhow::bail!("{} claims broad Mac performance or speedup", path.display());
    }

    let slm_pipeline = &receipt["slm_pipeline"];
    if slm_pipeline["selected_backend"].as_str() != Some(APPLE_M4_CPU_NEON)
        || slm_pipeline["runtime_api"].as_str() != Some("cpu")
        || slm_pipeline["cpu_pipeline_for_remaining_phases"].as_bool() != Some(true)
    {
        anyhow::bail!(
            "{} Metal phase receipt must record the remaining SLM pipeline as {APPLE_M4_CPU_NEON}",
            path.display()
        );
    }

    let metal_phase = &receipt["metal_phase"];
    if metal_phase["selected_backend"].as_str() != Some(APPLE_M4_METAL)
        || metal_phase["runtime_api"].as_str() != Some("metal")
        || metal_phase["fallback_used"].as_bool() != Some(false)
    {
        anyhow::bail!(
            "{} Metal phase details must record selected_backend={APPLE_M4_METAL}, runtime_api=metal, fallback_used=false",
            path.display()
        );
    }
    let execution_phase = metal_phase["execution_phase"].as_str().unwrap_or_default();
    if !matches!(execution_phase, "prefill_linear_projection" | "prefill_qkv_projection") {
        anyhow::bail!(
            "{} Metal phase has unsupported execution_phase={execution_phase:?}",
            path.display()
        );
    }
    if metal_phase["kernel_id"].as_str().is_none() {
        anyhow::bail!("{} Metal phase receipt is missing kernel_id", path.display());
    }
    if metal_phase["timing_recorded"].as_bool() != Some(true) {
        anyhow::bail!("{} Metal phase receipt must record phase timing", path.display());
    }

    let layout = &receipt["layout"];
    if layout["consumes_dense_f32_directly"].as_bool() != Some(true)
        || layout["dequantizes_before_compute"].as_bool() != Some(false)
    {
        anyhow::bail!(
            "{} Metal phase layout must record direct dense f32 consumption without dequantization",
            path.display()
        );
    }
    let parity = &receipt["parity"];
    if parity["reference_backend"].as_str() != Some(APPLE_M4_CPU_NEON)
        || parity["target_backend"].as_str() != Some(APPLE_M4_METAL)
    {
        anyhow::bail!(
            "{} Metal phase parity must compare {APPLE_M4_CPU_NEON} against {APPLE_M4_METAL}",
            path.display()
        );
    }
    if parity["max_abs_error"].is_null() || parity["mean_abs_error"].is_null() {
        anyhow::bail!("{} Metal phase parity is missing error metrics", path.display());
    }
    match execution_phase {
        "prefill_linear_projection" => {
            validate_prefill_linear_phase_receipt(path, layout, parity)?;
        }
        "prefill_qkv_projection" => {
            validate_prefill_qkv_phase_receipt(path, receipt, layout, parity)?;
        }
        _ => unreachable!("execution_phase was matched above"),
    }

    let timing = &receipt["timing"];
    if !timing.is_object() {
        anyhow::bail!("{} Metal phase receipt is missing timing delta metrics", path.display());
    }
    if timing["scope"].as_str().is_none() {
        anyhow::bail!("{} Metal phase timing is missing scope", path.display());
    }
    for field in ["cpu_reference_ms", "metal_phase_ms", "timing_delta_ms"] {
        if timing[field].as_f64().is_none() {
            anyhow::bail!("{} Metal phase timing is missing {field}", path.display());
        }
    }
    if timing["cpu_reference_ms"].as_f64().unwrap_or_default() < 0.0
        || timing["metal_phase_ms"].as_f64().unwrap_or_default() < 0.0
    {
        anyhow::bail!("{} Metal phase timing must be non-negative", path.display());
    }
    if timing["speedup_claim"].as_bool() != Some(false) {
        anyhow::bail!("{} Metal phase timing must not claim speedup", path.display());
    }

    Ok(ReceiptCheckSummary {
        path: path.to_path_buf(),
        artifact_kind,
        requested_backend,
        selected_backend,
        runtime_api,
        fallback_used,
        prompt_count: None,
        generated_tokens: None,
        passed: true,
        regression: None,
    })
}

fn validate_prefill_linear_phase_receipt(
    path: &Path,
    layout: &serde_json::Value,
    parity: &serde_json::Value,
) -> Result<()> {
    for field in ["batch_size", "in_features", "out_features"] {
        if layout[field].as_u64().unwrap_or_default() == 0 {
            anyhow::bail!("{} Metal phase layout is missing {field}", path.display());
        }
    }
    if parity["greedy_token_ids_match_cpu_reference"].as_bool() != Some(true) {
        anyhow::bail!(
            "{} Metal phase parity must match CPU/NEON reference greedy token IDs",
            path.display()
        );
    }
    Ok(())
}

fn validate_prefill_qkv_phase_receipt(
    path: &Path,
    receipt: &serde_json::Value,
    layout: &serde_json::Value,
    parity: &serde_json::Value,
) -> Result<()> {
    let dimensions = &receipt["dimensions"];
    for field in ["hidden_size", "attention_heads", "kv_heads", "head_dim", "q_dim", "kv_dim"] {
        if dimensions[field].as_u64().unwrap_or_default() == 0 {
            anyhow::bail!("{} Metal Q/K/V phase dimensions are missing {field}", path.display());
        }
    }
    let prefill_tokens = receipt["metal_phase"]["prefill_tokens"].as_u64().unwrap_or_default();
    let q_dim = dimensions["q_dim"].as_u64().unwrap_or_default();
    let kv_dim = dimensions["kv_dim"].as_u64().unwrap_or_default();
    for (field, expected_width) in [("q_shape", q_dim), ("k_shape", kv_dim), ("v_shape", kv_dim)] {
        let shape = dimensions[field].as_array();
        if shape.is_none_or(|shape| {
            shape.len() != 2
                || shape[0].as_u64() != Some(prefill_tokens)
                || shape[1].as_u64() != Some(expected_width)
        }) {
            anyhow::bail!("{} Metal Q/K/V phase dimensions have invalid {field}", path.display());
        }
    }
    for field in
        ["activation_elements", "q_weight_elements", "k_weight_elements", "v_weight_elements"]
    {
        if layout[field].as_u64().unwrap_or_default() == 0 {
            anyhow::bail!("{} Metal Q/K/V phase layout is missing {field}", path.display());
        }
    }
    if layout["output_layout"].as_str() != Some("concatenated_row_major_f32_q_k_v")
        || layout["bias_layout"].as_str() != Some("concatenated_row_major_f32_q_k_v")
    {
        anyhow::bail!(
            "{} Metal Q/K/V phase layout must record concatenated Q/K/V output and bias layouts",
            path.display()
        );
    }
    for prefix in ["q", "k", "v"] {
        let matches_field = format!("{prefix}_matches_cpu_reference");
        if parity[matches_field.as_str()].as_bool() != Some(true) {
            anyhow::bail!(
                "{} Metal Q/K/V phase parity must record {prefix}_matches_cpu_reference=true",
                path.display()
            );
        }
        for suffix in ["max_abs_error", "mean_abs_error"] {
            let field = format!("{prefix}_{suffix}");
            if parity[&field].as_f64().is_none() {
                anyhow::bail!("{} Metal Q/K/V phase parity is missing {field}", path.display());
            }
        }
    }
    Ok(())
}

fn validate_one_shot_receipt(
    path: &Path,
    receipt: &serde_json::Value,
) -> Result<(Option<usize>, Option<usize>)> {
    let text = receipt["text"].as_str().unwrap_or_default();
    if text.trim().is_empty() {
        anyhow::bail!("{} one-shot Mac receipt has empty generated text", path.display());
    }
    let generated = receipt["tokens"]["generated"].as_u64().unwrap_or_default() as usize;
    let generated_ids = receipt["tokens"]["generated_ids"]
        .as_array()
        .or_else(|| receipt["tokens"]["ids"].as_array());
    if generated == 0 || generated_ids.is_none_or(|ids| ids.is_empty()) {
        anyhow::bail!("{} one-shot Mac receipt is missing generated token IDs", path.display());
    }
    if receipt["model"]["sha256"].as_str().is_none() {
        anyhow::bail!("{} one-shot Mac receipt is missing model sha256", path.display());
    }
    if receipt["tokenizer"]["source"].as_str().is_none() {
        anyhow::bail!("{} one-shot Mac receipt is missing tokenizer source", path.display());
    }
    Ok((Some(1), Some(generated)))
}

fn validate_warm_session_receipt(
    path: &Path,
    receipt: &serde_json::Value,
    expected_backend: &str,
) -> Result<(Option<usize>, Option<usize>)> {
    if receipt["session"]["model_loaded_once"] != true {
        anyhow::bail!(
            "{} warm-session receipt does not record model_loaded_once=true",
            path.display()
        );
    }
    if receipt["session"]["tokenizer_loaded_once"] != true {
        anyhow::bail!(
            "{} warm-session receipt does not record tokenizer_loaded_once=true",
            path.display()
        );
    }
    if receipt["quality_summary"]["passed"].as_bool().is_some_and(|passed| !passed) {
        anyhow::bail!("{} warm-session quality summary failed", path.display());
    }
    if receipt["operator_ux"].is_object()
        && receipt["operator_ux"]["time_to_first_token_receipts"].as_bool() != Some(true)
    {
        anyhow::bail!(
            "{} warm-session receipt must expose time-to-first-token UX receipts",
            path.display()
        );
    }
    let prompts = receipt["prompts"].as_array().ok_or_else(|| {
        anyhow!("{} warm-session receipt is missing prompt summaries", path.display())
    })?;
    if prompts.is_empty() {
        anyhow::bail!("{} warm-session receipt has no prompts", path.display());
    }
    let metal_phase_contributions_enabled =
        receipt["metal_phase_contributions"]["enabled"].as_bool().unwrap_or(false);
    if metal_phase_contributions_enabled {
        validate_warm_session_metal_phase_header(path, receipt)?;
    }
    let dense_quality_corpus =
        receipt["corpus"]["artifact_kind"].as_str() == Some("apple_m4_slm_quality_corpus");
    if dense_quality_corpus {
        validate_dense_slm_quality_corpus_header(path, receipt, prompts.len())?;
    }
    let mut generated_total = 0usize;
    for prompt in prompts {
        if prompt["backend"]["requested_backend"].as_str() != Some(expected_backend) {
            anyhow::bail!(
                "{} warm-session prompt requested backend must match aggregate backend {expected_backend}",
                path.display()
            );
        }
        if prompt["backend"]["selected_backend"].as_str() != Some(expected_backend) {
            anyhow::bail!(
                "{} warm-session prompt selected backend must match aggregate backend {expected_backend}",
                path.display()
            );
        }
        if prompt["backend"]["runtime_api"].as_str() != Some("cpu") {
            anyhow::bail!("{} warm-session prompt runtime_api must be cpu", path.display());
        }
        if prompt["backend"]["fallback_used"].as_bool().unwrap_or(true) {
            anyhow::bail!("{} warm-session prompt records fallback_used=true", path.display());
        }
        if prompt["text"].as_str().unwrap_or_default().trim().is_empty() {
            anyhow::bail!("{} warm-session prompt has empty generated text", path.display());
        }
        let generated = prompt["generated_tokens"].as_u64().unwrap_or_default() as usize;
        if generated == 0 {
            anyhow::bail!("{} warm-session prompt generated zero tokens", path.display());
        }
        let generated_ids = prompt["generated_token_ids"]
            .as_array()
            .or_else(|| prompt["tokens"]["generated_ids"].as_array())
            .ok_or_else(|| {
                anyhow!("{} warm-session prompt is missing generated token IDs", path.display())
            })?;
        if generated_ids.is_empty() {
            anyhow::bail!("{} warm-session prompt generated token IDs are empty", path.display());
        }
        if generated_ids.len() != generated {
            anyhow::bail!(
                "{} warm-session prompt generated token ID count does not match generated_tokens",
                path.display()
            );
        }
        validate_warm_session_prompt_quality(path, prompt)?;
        if prompt["operator_ux"].is_object()
            && prompt["operator_ux"]["time_to_first_token_receipt"].as_bool() != Some(true)
        {
            anyhow::bail!(
                "{} warm-session prompt is missing time-to-first-token UX receipt",
                path.display()
            );
        }
        if prompt["timing"]["time_to_first_token_ms"].is_null() && prompt["operator_ux"].is_object()
        {
            anyhow::bail!(
                "{} warm-session prompt is missing timing.time_to_first_token_ms",
                path.display()
            );
        }
        if metal_phase_contributions_enabled {
            validate_warm_session_prompt_metal_phase(path, prompt)?;
        }
        generated_total += generated;
    }
    if dense_quality_corpus {
        validate_dense_slm_quality_corpus_determinism(path, receipt, prompts)?;
    }
    Ok((Some(prompts.len()), Some(generated_total)))
}

fn validate_dense_slm_quality_corpus_header(
    path: &Path,
    receipt: &serde_json::Value,
    prompt_count: usize,
) -> Result<()> {
    let corpus = &receipt["corpus"];
    let expected_case_count = expected_dense_slm_quality_corpus_cases(path, receipt)?;
    let case_count = corpus["case_count"].as_u64().unwrap_or_default() as usize;
    let repeat_runs = corpus["repeat_runs"].as_u64().unwrap_or_default() as usize;
    if case_count != expected_case_count {
        anyhow::bail!(
            "{} dense SLM corpus must contain {expected_case_count} prompt cases",
            path.display()
        );
    }
    if repeat_runs < 2 {
        anyhow::bail!("{} dense SLM corpus must repeat prompts for determinism", path.display());
    }
    if prompt_count != case_count.saturating_mul(repeat_runs) {
        anyhow::bail!(
            "{} dense SLM corpus prompt count must equal case_count * repeat_runs",
            path.display()
        );
    }
    if receipt["generation"]["mode"].as_str() != Some("greedy")
        || receipt["generation"]["deterministic"].as_bool() != Some(true)
        || receipt["generation"]["temperature"].as_f64() != Some(0.0)
        || receipt["generation"]["top_k"].as_u64() != Some(1)
    {
        anyhow::bail!(
            "{} dense SLM corpus must be deterministic greedy top-1 generation",
            path.display()
        );
    }
    if receipt["model"]["sha256"].as_str().is_none() {
        anyhow::bail!("{} dense SLM corpus receipt is missing model sha256", path.display());
    }
    if receipt["tokenizer"]["source"].as_str().is_none()
        || receipt["tokenizer"]["pretokenizer_authority"].as_str().is_none()
    {
        anyhow::bail!("{} dense SLM corpus receipt is missing tokenizer authority", path.display());
    }
    Ok(())
}

fn expected_dense_slm_quality_corpus_cases(
    path: &Path,
    receipt: &serde_json::Value,
) -> Result<usize> {
    match receipt["corpus"]["name"].as_str() {
        Some("apple-m4-slm-quality-determinism-v1") => Ok(5),
        Some("apple-m4-slm-quality-determinism-v2") => Ok(7),
        _ => {
            anyhow::bail!("{} dense SLM corpus receipt has unexpected corpus name", path.display())
        }
    }
}

fn validate_warm_session_prompt_quality(path: &Path, prompt: &serde_json::Value) -> Result<()> {
    let quality = &prompt["quality"];
    if quality["passed"].as_bool() != Some(true) {
        anyhow::bail!("{} warm-session prompt quality failed", path.display());
    }
    for field in ["valid_utf8", "non_empty", "non_degenerate"] {
        if quality[field].as_bool() != Some(true) {
            anyhow::bail!(
                "{} warm-session prompt quality must record {field}=true",
                path.display()
            );
        }
    }
    if quality["failed_rules"].as_array().is_none_or(|rules| !rules.is_empty()) {
        anyhow::bail!("{} warm-session prompt quality failed_rules must be empty", path.display());
    }
    if quality["distinct_generated_tokens"].as_u64().unwrap_or_default() == 0 {
        anyhow::bail!(
            "{} warm-session prompt quality must record distinct generated tokens",
            path.display()
        );
    }
    Ok(())
}

fn validate_dense_slm_quality_corpus_determinism(
    path: &Path,
    receipt: &serde_json::Value,
    prompts: &[serde_json::Value],
) -> Result<()> {
    let expected_case_count = expected_dense_slm_quality_corpus_cases(path, receipt)?;
    let determinism = &receipt["determinism"];
    if determinism["checked"].as_bool() != Some(true)
        || determinism["passed"].as_bool() != Some(true)
    {
        anyhow::bail!("{} dense SLM corpus determinism failed", path.display());
    }
    if determinism["repeated_prompt_groups"].as_u64() != Some(expected_case_count as u64) {
        anyhow::bail!(
            "{} dense SLM corpus must record {expected_case_count} repeated prompt groups",
            path.display(),
        );
    }
    let groups = determinism["groups"].as_array().ok_or_else(|| {
        anyhow!("{} dense SLM corpus determinism is missing groups", path.display())
    })?;
    if groups.len() != expected_case_count {
        anyhow::bail!(
            "{} dense SLM corpus determinism groups must have length {expected_case_count}",
            path.display(),
        );
    }
    for group in groups {
        if group["attempt_count"].as_u64().unwrap_or_default() < 2
            || group["stable_generated_token_ids"].as_bool() != Some(true)
            || group["stable_text"].as_bool() != Some(true)
        {
            anyhow::bail!(
                "{} dense SLM corpus has unstable deterministic greedy output",
                path.display()
            );
        }
        if group["reference_generated_ids"].as_array().is_none_or(|ids| ids.is_empty()) {
            anyhow::bail!(
                "{} dense SLM corpus determinism group is missing reference generated token IDs",
                path.display()
            );
        }
    }

    let mut by_case: std::collections::BTreeMap<String, Vec<&serde_json::Value>> =
        std::collections::BTreeMap::new();
    for prompt in prompts {
        let case_id = prompt["case_id"].as_str().ok_or_else(|| {
            anyhow!("{} dense SLM corpus prompt is missing case_id", path.display())
        })?;
        if prompt["repeat_index"].as_u64().is_none() {
            anyhow::bail!("{} dense SLM corpus prompt is missing repeat_index", path.display());
        }
        by_case.entry(case_id.to_string()).or_default().push(prompt);
    }
    if by_case.len() != expected_case_count {
        anyhow::bail!(
            "{} dense SLM corpus must cover {expected_case_count} case IDs",
            path.display()
        );
    }
    for (case_id, prompts) in by_case {
        if prompts.len() < 2 {
            anyhow::bail!(
                "{} dense SLM corpus case {case_id} must have repeated prompts",
                path.display()
            );
        }
        let first_ids = &prompts[0]["generated_token_ids"];
        let first_text = prompts[0]["text"].as_str().unwrap_or_default();
        if first_ids.as_array().is_none_or(|ids| ids.is_empty()) || first_text.trim().is_empty() {
            anyhow::bail!(
                "{} dense SLM corpus case {case_id} is missing reference text or token IDs",
                path.display()
            );
        }
        for prompt in prompts.iter().skip(1) {
            if &prompt["generated_token_ids"] != first_ids
                || prompt["text"].as_str().unwrap_or_default() != first_text
            {
                anyhow::bail!(
                    "{} dense SLM corpus case {case_id} changed deterministic greedy output",
                    path.display()
                );
            }
        }
    }
    Ok(())
}

fn validate_warm_session_metal_phase_header(
    path: &Path,
    receipt: &serde_json::Value,
) -> Result<()> {
    let phase = &receipt["metal_phase_contributions"];
    if phase["execution_phase"].as_str() != Some("prefill_qkv_projection")
        || phase["selected_backend"].as_str() != Some(APPLE_M4_METAL)
        || phase["runtime_api"].as_str() != Some("metal")
        || phase["fallback_used"].as_bool() != Some(false)
        || phase["cpu_pipeline_for_remaining_phases"].as_bool() != Some(true)
        || phase["resident_generation_backend"].as_str() != Some(APPLE_M4_CPU_NEON)
        || phase["resident_greedy_token_ids_match_cpu_reference"].as_bool() != Some(true)
    {
        anyhow::bail!(
            "{} warm-session Metal phase header must record phase-scoped apple-m4-metal contribution with CPU/NEON resident generation",
            path.display()
        );
    }
    if receipt_flag_true(phase, "full_metal_inference_claimed")
        || receipt_flag_true(phase, "speedup_claim")
    {
        anyhow::bail!(
            "{} warm-session Metal phase header must not claim full Metal inference or speedup",
            path.display()
        );
    }
    Ok(())
}

fn validate_warm_session_prompt_metal_phase(path: &Path, prompt: &serde_json::Value) -> Result<()> {
    let phases = prompt["metal_phase_contributions"].as_array().ok_or_else(|| {
        anyhow!("{} warm-session prompt is missing Metal phase contributions", path.display())
    })?;
    if phases.is_empty() {
        anyhow::bail!("{} warm-session prompt has no Metal phase contributions", path.display());
    }
    for phase in phases {
        if phase["execution_phase"].as_str() != Some("prefill_qkv_projection")
            || phase["selected_backend"].as_str() != Some(APPLE_M4_METAL)
            || phase["runtime_api"].as_str() != Some("metal")
            || phase["fallback_used"].as_bool() != Some(false)
            || phase["cpu_pipeline_for_remaining_phases"].as_bool() != Some(true)
            || phase["resident_greedy_token_ids_match_cpu_reference"].as_bool() != Some(true)
        {
            anyhow::bail!(
                "{} warm-session prompt Metal phase contribution must record Q/K/V Metal parity and CPU/NEON remainder",
                path.display()
            );
        }
        if receipt_flag_true(phase, "full_metal_inference_claimed")
            || receipt_flag_true(phase, "speedup_claim")
        {
            anyhow::bail!(
                "{} warm-session prompt Metal phase contribution must not claim full Metal inference or speedup",
                path.display()
            );
        }
    }
    Ok(())
}

fn json_value_at<'a>(value: &'a serde_json::Value, segments: &[&str]) -> &'a serde_json::Value {
    let mut current = value;
    for segment in segments {
        current = &current[*segment];
    }
    current
}

fn json_path_label(segments: &[&str]) -> String {
    segments.join(".")
}

fn require_non_empty_string_at<'a>(
    receipt_path: &Path,
    value: &'a serde_json::Value,
    segments: &[&str],
) -> Result<&'a str> {
    let label = json_path_label(segments);
    let text = json_value_at(value, segments)
        .as_str()
        .ok_or_else(|| anyhow!("{} SLM eval summary is missing {label}", receipt_path.display()))?;
    if text.trim().is_empty() {
        anyhow::bail!("{} SLM eval summary {label} must not be empty", receipt_path.display());
    }
    Ok(text)
}

fn require_exact_string_at(
    receipt_path: &Path,
    value: &serde_json::Value,
    segments: &[&str],
    expected: &str,
) -> Result<()> {
    let label = json_path_label(segments);
    let observed = json_value_at(value, segments).as_str();
    if observed != Some(expected) {
        anyhow::bail!(
            "{} SLM eval summary {label} must be {expected:?}, got {observed:?}",
            receipt_path.display()
        );
    }
    Ok(())
}

fn require_bool_at(
    receipt_path: &Path,
    value: &serde_json::Value,
    segments: &[&str],
    expected: bool,
) -> Result<()> {
    let label = json_path_label(segments);
    let observed = json_value_at(value, segments).as_bool();
    if observed != Some(expected) {
        anyhow::bail!("{} SLM eval summary {label} must be {expected}", receipt_path.display());
    }
    Ok(())
}

fn require_u64_at(
    receipt_path: &Path,
    value: &serde_json::Value,
    segments: &[&str],
    positive: bool,
) -> Result<u64> {
    let label = json_path_label(segments);
    let number = json_value_at(value, segments).as_u64().ok_or_else(|| {
        anyhow!("{} SLM eval summary is missing numeric {label}", receipt_path.display())
    })?;
    if positive && number == 0 {
        anyhow::bail!(
            "{} SLM eval summary {label} must be greater than zero",
            receipt_path.display()
        );
    }
    Ok(number)
}

fn require_number_at(
    receipt_path: &Path,
    value: &serde_json::Value,
    segments: &[&str],
    non_negative: bool,
) -> Result<f64> {
    let label = json_path_label(segments);
    let number = json_value_at(value, segments).as_f64().ok_or_else(|| {
        anyhow!("{} SLM eval summary is missing numeric {label}", receipt_path.display())
    })?;
    if non_negative && number < 0.0 {
        anyhow::bail!("{} SLM eval summary {label} must be non-negative", receipt_path.display());
    }
    Ok(number)
}

fn require_positive_number_at(
    receipt_path: &Path,
    value: &serde_json::Value,
    segments: &[&str],
) -> Result<f64> {
    let label = json_path_label(segments);
    let number = require_number_at(receipt_path, value, segments, false)?;
    if number <= 0.0 {
        anyhow::bail!(
            "{} SLM eval summary {label} must be greater than zero",
            receipt_path.display()
        );
    }
    Ok(number)
}

fn require_unit_rate_at(
    receipt_path: &Path,
    value: &serde_json::Value,
    segments: &[&str],
) -> Result<()> {
    let label = json_path_label(segments);
    let rate = require_number_at(receipt_path, value, segments, false)?;
    if !(0.0..=1.0).contains(&rate) {
        anyhow::bail!(
            "{} SLM eval summary {label} must be a rate between 0 and 1",
            receipt_path.display()
        );
    }
    Ok(())
}

fn require_non_empty_string_array_at(
    receipt_path: &Path,
    value: &serde_json::Value,
    segments: &[&str],
) -> Result<()> {
    let label = json_path_label(segments);
    let values = json_value_at(value, segments).as_array().ok_or_else(|| {
        anyhow!("{} SLM eval summary is missing array {label}", receipt_path.display())
    })?;
    if values.is_empty()
        || values.iter().any(|value| value.as_str().is_none_or(|text| text.trim().is_empty()))
    {
        anyhow::bail!(
            "{} SLM eval summary {label} must contain non-empty strings",
            receipt_path.display()
        );
    }
    Ok(())
}

fn require_string_array_equals(
    receipt_path: &Path,
    value: &serde_json::Value,
    segments: &[&str],
    expected: &[&str],
) -> Result<()> {
    let label = json_path_label(segments);
    let values = json_value_at(value, segments).as_array().ok_or_else(|| {
        anyhow!("{} SLM eval summary is missing array {label}", receipt_path.display())
    })?;
    let observed = values
        .iter()
        .map(|value| {
            value.as_str().ok_or_else(|| {
                anyhow!(
                    "{} SLM eval summary {label} must contain only strings",
                    receipt_path.display()
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if observed != expected {
        anyhow::bail!(
            "{} SLM eval summary {label} must be {:?}, got {:?}",
            receipt_path.display(),
            expected,
            observed
        );
    }
    Ok(())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn receipt_string(receipt: &serde_json::Value, key: &str) -> Option<String> {
    receipt[key].as_str().or_else(|| receipt["backend"][key].as_str()).map(ToOwned::to_owned)
}

fn receipt_bool(receipt: &serde_json::Value, key: &str) -> Option<bool> {
    receipt[key].as_bool().or_else(|| receipt["backend"][key].as_bool())
}

fn receipt_flag_true(value: &serde_json::Value, key: &str) -> bool {
    match value {
        serde_json::Value::Object(map) => {
            map.get(key).and_then(serde_json::Value::as_bool).unwrap_or(false)
                || map.values().any(|child| receipt_flag_true(child, key))
        }
        serde_json::Value::Array(values) => {
            values.iter().any(|child| receipt_flag_true(child, key))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mac_validate_rejects_m3_metal_identity_without_cpu_fallback_claim() -> Result<(), String> {
        let err = match ensure_supported_mac_validate_device(Some(APPLE_M3_AIR_METAL)) {
            Ok(label) => return Err(format!("M3 Air Metal should be rejected, got {label}")),
            Err(err) => err.to_string(),
        };

        assert!(err.contains(APPLE_M3_AIR_METAL), "got: {err}");
        assert!(err.contains(APPLE_M3_AIR_MPSGRAPH), "got: {err}");
        assert!(err.contains(APPLE_M3_AIR_CPU_NEON), "got: {err}");
        assert!(err.contains("hidden CPU fallback"), "got: {err}");
        Ok(())
    }

    fn test_verified_model(cache_root: &Path) -> VerifiedCachedModel {
        VerifiedCachedModel {
            id: "qwen2.5-0.5b-instruct-q8_0".to_string(),
            display_name: "Qwen2.5 0.5B Instruct Q8_0".to_string(),
            path: cache_root.join("qwen2.5-0.5b-instruct-q8_0.gguf"),
            cache_root: cache_root.to_path_buf(),
            sha256: "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e".to_string(),
            bytes: 675_710_816,
            architecture: "qwen2".to_string(),
            quantization: "Q8_0".to_string(),
            tokenizer_model: "gpt2".to_string(),
            tokenizer_pre: "qwen2".to_string(),
            chat_template: true,
            support_note: "test model".to_string(),
        }
    }

    fn test_state() -> Result<(tempfile::TempDir, MacServeState), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let receipt_dir = temp.path().join("receipts");
        std::fs::create_dir_all(&receipt_dir)?;
        let state = MacServeState::new(
            test_verified_model(temp.path()),
            "127.0.0.1".to_string(),
            8080,
            true,
            MacServeGenerationDefaults {
                max_new_tokens: 8,
                temperature: 0.0,
                top_k: 1,
                top_p: 1.0,
                repetition_penalty: 1.1,
                seed: None,
            },
            receipt_dir,
            None,
        );
        Ok((temp, state))
    }

    #[test]
    fn bitnet_warm_profile_prompt_plan_builds_resident_100_checkpoints()
    -> Result<(), Box<dyn std::error::Error>> {
        let (prompts, source, plan) = resolve_bitnet_warm_prompt_plan(
            Vec::new(),
            vec![
                BitnetWarmProfile::Resident100,
                BitnetWarmProfile::Resident25,
                BitnetWarmProfile::Resident50,
            ],
        )?;
        let plan = plan.ok_or_else(|| std::io::Error::other("missing profile plan"))?;

        assert_eq!(source, BitnetWarmPromptSource::ProfilePrompts);
        assert_eq!(plan.profile_ids(), vec!["resident_25", "resident_50", "resident_100"]);
        assert_eq!(plan.max_prompt_count, 100);
        assert_eq!(prompts.len(), 100);
        assert_eq!(prompts.first(), prompts.last());

        let mut counts = std::collections::BTreeMap::new();
        for prompt in &prompts {
            *counts.entry(prompt).or_insert(0usize) += 1;
        }
        assert!(counts.values().any(|count| *count >= 2));
        Ok(())
    }

    #[test]
    fn bitnet_warm_profile_prompt_plan_rejects_prompt_mix() {
        let err = resolve_bitnet_warm_prompt_plan(
            vec![
                "Answer with a single digit: 2+2=".to_string(),
                "Answer with a single digit: 2+2=".to_string(),
            ],
            vec![BitnetWarmProfile::Resident25],
        )
        .expect_err("profile and prompt mix should fail")
        .to_string();

        assert!(err.contains("cannot be combined with --prompt"), "got: {err}");
    }

    #[test]
    fn mac_receipts_check_accepts_m3_warm_session_prompt_backend()
    -> Result<(), Box<dyn std::error::Error>> {
        let receipt = test_m3_warm_session_receipt();

        let summary = validate_mac_receipt_value(Path::new("m3-warm-session.json"), &receipt)?;

        assert_eq!(summary.artifact_kind, "slm_apple_m3_air_warm_session");
        assert_eq!(summary.requested_backend, APPLE_M3_AIR_CPU_NEON);
        assert_eq!(summary.selected_backend, APPLE_M3_AIR_CPU_NEON);
        assert_eq!(summary.prompt_count, Some(1));
        assert_eq!(summary.generated_tokens, Some(2));
        Ok(())
    }

    #[test]
    fn mac_receipt_annotation_preserves_m3_claim_boundary_backend()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let receipt_path = temp.path().join("m3-warm-session.json");
        std::fs::write(&receipt_path, serde_json::to_vec_pretty(&test_m3_warm_session_receipt())?)?;

        annotate_and_validate_mac_receipt_silent(
            &receipt_path,
            &test_verified_model(temp.path()),
            "mac validate",
        )?;

        let annotated: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt_path)?)?;
        assert_eq!(annotated["mac_claim_boundary"]["requested_backend"], APPLE_M3_AIR_CPU_NEON);
        Ok(())
    }

    #[test]
    fn mac_receipts_check_accepts_m3_operator_profile_summary()
    -> Result<(), Box<dyn std::error::Error>> {
        let receipt = test_m3_operator_profile_summary();

        let summary = validate_mac_receipt_value(Path::new("m3-operator.json"), &receipt)?;

        assert_eq!(summary.artifact_kind, "apple_m3_air_slm_operator_profiles");
        assert_eq!(summary.requested_backend, APPLE_M3_AIR_CPU_NEON);
        assert_eq!(summary.selected_backend, APPLE_M3_AIR_CPU_NEON);
        assert_eq!(summary.prompt_count, Some(3));
        assert_eq!(summary.generated_tokens, Some(112));
        Ok(())
    }

    #[test]
    fn mac_receipts_check_accepts_bitnet_eval_answer_corpus()
    -> Result<(), Box<dyn std::error::Error>> {
        let receipt = test_bitnet_eval_answer_corpus_receipt();

        let summary = validate_mac_receipt_value(Path::new("bitnet-eval.json"), &receipt)?;

        assert_eq!(summary.artifact_kind, "bitnet_apple_m4_local_answer_corpus");
        assert_eq!(summary.requested_backend, APPLE_M4_CPU_NEON);
        assert_eq!(summary.selected_backend, APPLE_M4_CPU_NEON);
        assert_eq!(summary.prompt_count, Some(2));
        assert_eq!(summary.generated_tokens, Some(3));
        Ok(())
    }

    #[test]
    fn mac_receipts_check_accepts_bitnet_benchmark_v1() -> Result<(), Box<dyn std::error::Error>> {
        let receipt = test_bitnet_benchmark_v1_receipt();

        let summary = validate_mac_receipt_value(Path::new("bitnet-benchmark.json"), &receipt)?;

        assert_eq!(summary.artifact_kind, "bitnet_apple_m4_benchmark_v1");
        assert_eq!(summary.requested_backend, APPLE_M4_CPU_NEON);
        assert_eq!(summary.selected_backend, APPLE_M4_CPU_NEON);
        assert_eq!(summary.prompt_count, Some(4));
        assert_eq!(summary.generated_tokens, Some(8));
        Ok(())
    }

    #[test]
    fn mac_profile_set_allocation_scope_preserves_m3_backend_label() {
        assert_eq!(
            profile_set_allocation_scope(APPLE_M3_AIR_CPU_NEON),
            "selected Apple M3 Air CPU/NEON SLM warm-session profile set"
        );
    }

    fn test_m3_warm_session_receipt() -> serde_json::Value {
        serde_json::json!({
            "artifact_kind": "slm_apple_m3_air_warm_session",
            "backend": {
                "requested_backend": APPLE_M3_AIR_CPU_NEON,
                "selected_backend": APPLE_M3_AIR_CPU_NEON,
                "runtime_api": "cpu",
                "fallback_used": false
            },
            "session": {
                "model_loaded_once": true,
                "tokenizer_loaded_once": true
            },
            "corpus": {
                "artifact_kind": "test_warm_session"
            },
            "quality_summary": {
                "passed": true
            },
            "prompts": [
                {
                    "backend": {
                        "requested_backend": APPLE_M3_AIR_CPU_NEON,
                        "selected_backend": APPLE_M3_AIR_CPU_NEON,
                        "runtime_api": "cpu",
                        "fallback_used": false
                    },
                    "text": "ready",
                    "generated_tokens": 2,
                    "generated_token_ids": [1, 2],
                    "quality": {
                        "passed": true,
                        "valid_utf8": true,
                        "non_empty": true,
                        "non_degenerate": true,
                        "failed_rules": [],
                        "distinct_generated_tokens": 2
                    }
                }
            ]
        })
    }

    fn test_bitnet_eval_answer_corpus_receipt() -> serde_json::Value {
        let case_backend = serde_json::json!({
            "requested_backend": APPLE_M4_CPU_NEON,
            "selected_backend": APPLE_M4_CPU_NEON,
            "runtime_api": "cpu",
            "fallback_used": false
        });
        let case_model = serde_json::json!({
            "family": "bitnet",
            "sha256": BITNET_M4_EXPECTED_MODEL_SHA256
        });
        let case_tokenizer = serde_json::json!({
            "strict": true,
            "pretokenizer_authority": "llama-bpe"
        });
        serde_json::json!({
            "schema_version": "1.0.0",
            "artifact_kind": "bitnet_apple_m4_local_answer_corpus",
            "model_family": "bitnet",
            "prompt_template": BITNET_M4_PROMPT_TEMPLATE,
            "requested_backend": APPLE_M4_CPU_NEON,
            "selected_backend": APPLE_M4_CPU_NEON,
            "runtime_api": "cpu",
            "fallback_used": false,
            "model": {
                "family": "bitnet",
                "sha256": BITNET_M4_EXPECTED_MODEL_SHA256,
                "quant_format": "I2_S",
                "answer_ready_artifact_available": true,
                "answer_ready": {
                    "state": "answer_ready"
                }
            },
            "tokenizer": {
                "strict": true,
                "authority": {
                    "source": "external_tokenizer_json",
                    "sha256": BITNET_M4_EXPECTED_TOKENIZER_SHA256,
                    "ggml_pre": "llama-bpe"
                }
            },
            "corpus": {
                "name": "apple-m4-bitnet-eval-seeded-corpus",
                "case_count": 2
            },
            "quality_summary": {
                "total": 2,
                "passed": 1,
                "failed": 1,
                "timeout": 0,
                "not_run": 0
            },
            "scoring_summary": {
                "enabled": true,
                "total": 2,
                "passed": 1,
                "failed": 1,
                "not_run": 0
            },
            "task_family_summary": {
                "arithmetic_exact": {
                    "total": 1,
                    "passed": 1,
                    "failed": 0,
                    "timeout": 0,
                    "not_run": 0,
                    "scoring": {
                        "enabled": true
                    }
                },
                "numeric_tolerance": {
                    "total": 1,
                    "passed": 0,
                    "failed": 1,
                    "timeout": 0,
                    "not_run": 0,
                    "scoring": {
                        "enabled": true
                    }
                }
            },
            "reference_comparison": {
                "schema": "bitnet_reference_vs_rust_v1",
                "summary": {
                    "total": 2
                },
                "claim_boundary": {
                    "dense_slm_evidence_used": false,
                    "chat_enabled": false,
                    "serve_enabled": false,
                    "performance_claimed": false,
                    "full_metal_inference_claimed": false,
                    "qk256_apple_claimed": false,
                    "neural_engine_claimed": false,
                    "mpsgraph_claimed": false,
                    "broad_apple_silicon_claimed": false,
                    "runtime_accuracy_claimed": false
                }
            },
            "claim_boundary": {
                "full_metal_inference_claimed": false,
                "neural_engine_claimed": false,
                "qk256_apple_claimed": false,
                "broad_performance_claimed": false
            },
            "cases": [
                {
                    "id": "case-1",
                    "status": "passed",
                    "answer": "4",
                    "quality": {
                        "passed": true,
                        "non_empty_answer": true,
                        "printable_utf8": true,
                        "no_replacement_chars": true
                    },
                    "backend": case_backend.clone(),
                    "model": case_model.clone(),
                    "loader": {
                        "mode": "real_gguf"
                    },
                    "tokenizer": case_tokenizer.clone(),
                    "prompt_template": BITNET_M4_PROMPT_TEMPLATE,
                    "prompt": {
                        "template_family": BITNET_M4_PROMPT_TEMPLATE
                    },
                    "prompt_prefill": {
                        "exercised": true
                    },
                    "token_ids": {
                        "generated": [4]
                    },
                    "tokens": {
                        "generated": 1
                    },
                    "reference_comparison": {
                        "schema": "bitnet_reference_vs_rust_v1"
                    },
                    "timing": {
                        "decode_total_ms": 1.0
                    },
                    "latency": {
                        "total_ms": 2.0
                    }
                },
                {
                    "id": "case-2",
                    "status": "quality_failed",
                    "answer": "wrong",
                    "quality": {
                        "passed": false,
                        "non_empty_answer": true,
                        "printable_utf8": true,
                        "no_replacement_chars": true
                    },
                    "backend": case_backend,
                    "model": case_model,
                    "loader": {
                        "mode": "real_gguf"
                    },
                    "tokenizer": case_tokenizer,
                    "prompt_template": BITNET_M4_PROMPT_TEMPLATE,
                    "prompt": {
                        "template_family": BITNET_M4_PROMPT_TEMPLATE
                    },
                    "prompt_prefill": {
                        "exercised": true
                    },
                    "token_ids": {
                        "generated": [1, 2]
                    },
                    "tokens": {
                        "generated": 2
                    },
                    "reference_comparison": {
                        "schema": "bitnet_reference_vs_rust_v1"
                    },
                    "timing": {
                        "decode_total_ms": 1.0
                    },
                    "latency": {
                        "total_ms": 2.0
                    }
                }
            ]
        })
    }

    fn test_bitnet_benchmark_v1_receipt() -> serde_json::Value {
        let one_shot = test_bitnet_benchmark_path_summary("one_shot_mac_ask", "mac ask", 1, 2);
        let fixed_warm =
            test_bitnet_benchmark_path_summary("fixed_warm_session", "mac bitnet-warm", 3, 6);
        let mut samples = BitnetBenchmarkMetricSamples::default();
        samples.extend_from_summary(&one_shot);
        samples.extend_from_summary(&fixed_warm);
        serde_json::json!({
            "schema_version": "1.0.0",
            "artifact_kind": "bitnet_apple_m4_benchmark_v1",
            "requested_backend": APPLE_M4_CPU_NEON,
            "selected_backend": APPLE_M4_CPU_NEON,
            "runtime_api": "cpu",
            "fallback_used": false,
            "benchmark_set": "bitnet-one-shot-fixed-warm-v1",
            "build": {
                "release_mode": true
            },
            "model": {
                "id": BITNET_M4_MODEL_ID,
                "family": "bitnet",
                "sha256": BITNET_M4_EXPECTED_MODEL_SHA256,
                "quantization": "I2_S",
                "answer_ready_artifact_available": true,
                "answer_ready": {
                    "state": "answer_ready"
                }
            },
            "tokenizer": {
                "source": "external_tokenizer_json",
                "sha256": BITNET_M4_EXPECTED_TOKENIZER_SHA256,
                "authority": "llama-bpe",
                "pretokenizer_authority": "llama-bpe",
                "strict": true
            },
            "prompt_template": BITNET_M4_PROMPT_TEMPLATE,
            "paths": {
                "one_shot": one_shot,
                "fixed_warm": fixed_warm
            },
            "prompt_count": 4,
            "generated_tokens": 8,
            "speed": samples.speed_json(),
            "memory": samples.memory_json(),
            "timeout_boundary": {
                "enforced": false,
                "reached": false,
                "status": "not_reached"
            },
            "evidence": {
                "generated_text_recorded": true,
                "generated_token_ids_recorded": true,
                "one_shot_receipt": "receipts/ask.json",
                "warm_session_receipt": "receipts/warm.json",
                "warm_prompt_receipts": ["receipts/warm-1.json"],
                "operator_commands": ["mac ask", "mac bitnet-warm"]
            },
            "mac_claim_boundary": {
                "bitnet_benchmark": true,
                "one_shot_mac_ask": true,
                "fixed_warm_session": true,
                "accepted_i2s_artifact_only": true,
                "dense_slm_evidence_used": false,
                "chat_enabled": false,
                "serve_enabled": false,
                "bitnet_quality_claimed": false,
                "broad_model_quality_claim": false,
                "broad_performance_claim": false,
                "speedup_claim": false,
                "full_metal_inference_claimed": false,
                "mpsgraph_inference_claimed": false,
                "neural_engine_execution_claimed": false,
                "qk256_apple_claimed": false,
                "macbook_evidence": false,
                "broad_apple_silicon_claimed": false
            },
            "bitnet_quality_claimed": false,
            "broad_performance_claim": false,
            "speedup_claim": false
        })
    }

    fn test_bitnet_benchmark_path_summary(
        path_id: &str,
        operator_command: &str,
        prompt_count: u64,
        generated_tokens: u64,
    ) -> serde_json::Value {
        serde_json::json!({
            "path_id": path_id,
            "operator_command": operator_command,
            "receipt_path": "receipt.json",
            "prompt_count": prompt_count,
            "generated_tokens": generated_tokens,
            "model_loaded_once": true,
            "tokenizer_loaded_once": true,
            "reuse_scope": "resident_session",
            "quality_passed": true,
            "prompt_tokens": benchmark_stat_json(&[20.0]),
            "output_tokens": benchmark_stat_json(&[generated_tokens as f64]),
            "timing": {
                "model_load_ms": benchmark_stat_json(&[4000.0]),
                "tokenizer_load_ms": benchmark_stat_json(&[100.0]),
                "prompt_tokenize_ms": benchmark_stat_json(&[1.0]),
                "prefill_ms": benchmark_stat_json(&[7000.0]),
                "time_to_first_token_ms": benchmark_stat_json(&[7500.0]),
                "decode_total_ms": benchmark_stat_json(&[900.0]),
                "sampling_ms_per_token": benchmark_stat_json(&[0.1]),
                "total_wall_ms": benchmark_stat_json(&[8500.0])
            },
            "throughput": {
                "input_tokens_per_second": benchmark_stat_json(&[2.857]),
                "output_tokens_per_second": benchmark_stat_json(&[0.235]),
                "decode_tokens_per_second": benchmark_stat_json(&[2.222])
            },
            "memory": {
                "peak_memory_mb": benchmark_stat_json(&[4200.0]),
                "memory_drift_mb": benchmark_stat_json(&[0.0]),
                "source": "getrusage.ru_maxrss process peak"
            },
            "timeout_boundary": {
                "enforced": false,
                "reached": false,
                "status": "not_reached"
            },
            "claim_boundary": {
                "chat_enabled": false,
                "serve_enabled": false,
                "broad_performance_claim": false,
                "speedup_claim": false
            }
        })
    }

    fn test_m3_operator_profile_summary() -> serde_json::Value {
        serde_json::json!({
            "artifact_kind": "apple_m3_air_slm_operator_profiles",
            "requested_backend": APPLE_M3_AIR_CPU_NEON,
            "selected_backend": APPLE_M3_AIR_CPU_NEON,
            "runtime_api": "cpu",
            "fallback_used": false,
            "profile_set": "operator",
            "profiles": [
                test_operator_profile("warm_16", 16, 16),
                test_operator_profile("warm_32", 32, 32),
                test_operator_profile("warm_64", 64, 64)
            ],
            "build": {
                "profile": "release",
                "release_mode": true
            },
            "operator_thresholds": {
                "scope": "supported Apple M3 Air SLM warm-answer timing only",
                "profiles_loaded_independently": true,
                "profile_set_model_loads": 3,
                "cold_load_separated": true,
                "model_tokenizer_reuse_visible": true,
                "model_tokenizer_reuse_visible_per_profile": true,
                "thresholds_are_claim_bounds_not_speed_guarantees": true
            },
            "performance_baseline": {
                "release_mode_observed": true,
                "warm_128_included": false
            },
            "allocation_audit": {
                "enabled": true,
                "method": "process_global_allocator_counter_delta",
                "scope": "selected Apple M3 Air CPU/NEON SLM warm-session profile set",
                "optimization_deferred": true,
                "ranked_hotspots": [
                    {
                        "component": "decode_total",
                        "alloc_count": 1,
                        "alloc_bytes": 2
                    }
                ]
            }
        })
    }

    fn test_operator_profile(
        profile_id: &str,
        requested_tokens: u64,
        generated_tokens: u64,
    ) -> serde_json::Value {
        serde_json::json!({
            "profile_id": profile_id,
            "requested_max_new_tokens": requested_tokens,
            "quality_passed": true,
            "prompt_count": 1,
            "generated_tokens": generated_tokens,
            "model_loaded_once": true,
            "tokenizer_loaded_once": true,
            "cold_load_separated": true,
            "reuse_scope": "within_profile",
            "resident_session": {
                "reuse_scope": "resident_session",
                "session_owned_buffers": true,
                "prompt_token_buffer_reused": true,
                "generated_token_buffer_reused": true,
                "timing_buffers_reused": true,
                "kv_cache_reuse_policy": "recreated_per_prompt_for_prompt_isolation",
                "sampler_reuse_policy": "recreated_per_prompt_for_deterministic_prompt_independence"
            },
            "allocation_audit": {
                "enabled": true,
                "scope": "selected Apple M3 Air CPU/NEON SLM warm-session prompt hot path",
                "ranked_hotspots": [
                    {
                        "component": "decode_total",
                        "alloc_count": 1,
                        "alloc_bytes": 2
                    }
                ]
            },
            "timing": {
                "model_load_ms": 1.0,
                "tokenizer_load_ms": 1.0,
                "warm_prompt_wall_ms": 1.0,
                "decode_total_ms": 1.0,
                "sampling_ms": 1.0,
                "warm_prompt_generated_tok_s": 1.0,
                "decode_generated_tok_s": 1.0
            }
        })
    }

    #[test]
    fn bitnet_mac_ask_model_failure_guidance_includes_cache_repair_commands()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let context = BitNetMacAskFailureContext {
            model_id: "microsoft-bitnet-b1.58-2B-4T-i2s".to_string(),
            cache_dir: Some(temp.path().join("models")),
            model_path: None,
            tokenizer_path: Some(temp.path().join("tokenizer.json")),
            question_bytes: 12,
            question_sha256: "prompt-sha".to_string(),
            max_new_tokens: 16,
            timeout_seconds: None,
            progress_enabled: false,
            started_at: std::time::Instant::now(),
        };

        let guidance =
            bitnet_mac_ask_failure_repair_guidance("model_verify_failed", &context, false);
        let joined = guidance.join("\n");

        assert!(joined.contains("bitnet model fetch microsoft-bitnet-b1.58-2B-4T-i2s"));
        assert!(joined.contains("--cache-dir"));
        assert!(joined.contains("bitnet mac models"));
        assert!(joined.contains("bitnet model verify microsoft-bitnet-b1.58-2B-4T-i2s"));
        assert!(joined.contains("chat and serve disabled"));
        assert!(bitnet_mac_ask_failure_repair_text(&guidance).contains("Repair guidance:"));
        Ok(())
    }

    #[test]
    fn bitnet_mac_ask_model_failure_guidance_includes_explicit_model_verify_command()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let model_path = temp.path().join("wrong-bitnet.gguf");
        let context = BitNetMacAskFailureContext {
            model_id: "microsoft-bitnet-b1.58-2B-4T-i2s".to_string(),
            cache_dir: None,
            model_path: Some(model_path.clone()),
            tokenizer_path: Some(temp.path().join("tokenizer.json")),
            question_bytes: 12,
            question_sha256: "prompt-sha".to_string(),
            max_new_tokens: 16,
            timeout_seconds: None,
            progress_enabled: false,
            started_at: std::time::Instant::now(),
        };

        let guidance =
            bitnet_mac_ask_failure_repair_guidance("model_verify_failed", &context, false);
        let joined = guidance.join("\n");

        assert!(joined.contains("replace --model-path with the accepted Microsoft I2_S GGUF"));
        assert!(joined.contains("bitnet model verify microsoft-bitnet-b1.58-2B-4T-i2s --path"));
        assert!(joined.contains(&model_path.display().to_string()));
        Ok(())
    }

    #[test]
    fn bitnet_mac_ask_timeout_failure_receipt_preserves_boundary()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let receipt_path = temp.path().join("bitnet-ask-timeout-failure.json");
        let context = BitNetMacAskFailureContext {
            model_id: BITNET_M4_MODEL_ID.to_string(),
            cache_dir: Some(temp.path().join("models")),
            model_path: Some(temp.path().join("ggml-model-i2_s.gguf")),
            tokenizer_path: Some(temp.path().join("tokenizer.json")),
            question_bytes: 12,
            question_sha256: "prompt-sha".to_string(),
            max_new_tokens: 16,
            timeout_seconds: Some(1),
            progress_enabled: true,
            started_at: std::time::Instant::now(),
        };
        let guidance = bitnet_mac_ask_failure_repair_guidance("generation_timeout", &context, true);

        write_bitnet_mac_ask_failure_receipt(
            &receipt_path,
            context,
            "generation_timeout",
            "BitNet Mac ask exceeded --timeout-seconds 1",
            true,
            &guidance,
        )?;
        let receipt = read_json_receipt(&receipt_path)?;
        validate_mac_receipt_value(&receipt_path, &receipt)?;

        assert_eq!(receipt["timeout_boundary"]["configured_seconds"], 1);
        assert_eq!(receipt["timeout_boundary"]["enforced"], true);
        assert_eq!(receipt["timeout_boundary"]["reached"], true);
        assert_eq!(receipt["timeout_boundary"]["stage"], "generation_timeout");
        assert_eq!(receipt["generation"]["partial_generation_available"], false);
        assert_eq!(receipt["progress"]["enabled"], true);
        assert!(guidance.join("\n").contains("increase --timeout-seconds"));
        assert_eq!(receipt["mac_bitnet_claim_boundary"]["chat_enabled"], false);
        assert_eq!(receipt["mac_bitnet_claim_boundary"]["serve_enabled"], false);
        Ok(())
    }

    #[test]
    fn bitnet_mac_ask_readiness_is_advisory_and_preserves_claim_boundary()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let readiness = bitnet_mac_ask_readiness_json(Some(temp.path().join("models")));

        assert_eq!(readiness["checked"], true);
        assert_eq!(readiness["advisory"], true);
        assert_eq!(readiness["blocks_doctor"], false);
        assert_eq!(readiness["model"]["id"], BITNET_M4_MODEL_ID);
        assert_eq!(readiness["model"]["catalog_state"], "supported-ask");
        assert_eq!(readiness["claim_boundary"]["chat_enabled"], false);
        assert_eq!(readiness["claim_boundary"]["serve_enabled"], false);
        assert_eq!(readiness["claim_boundary"]["bitnet_fixed_prompt_warm_session"], true);
        assert_eq!(readiness["model"]["mac_bitnet_warm_enabled"], true);
        assert_eq!(readiness["claim_boundary"]["full_metal_inference_claimed"], false);
        assert!(
            readiness["commands"]["models"]
                .as_str()
                .ok_or_else(|| std::io::Error::other("models command"))?
                .contains("bitnet mac models")
        );
        assert!(
            readiness["commands"]["ask_cached_model"]
                .as_str()
                .ok_or_else(|| std::io::Error::other("ask command"))?
                .contains("--tokenizer models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json")
        );
        assert!(
            readiness["commands"]["warm_cached_model"]
                .as_str()
                .ok_or_else(|| std::io::Error::other("warm command"))?
                .contains("bitnet mac bitnet-warm")
        );
        Ok(())
    }

    #[test]
    fn mac_serve_ready_json_records_cache_backend_and_no_generation()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_temp, state) = test_state()?;
        let ready = mac_serve_ready_json(&state);

        assert_eq!(ready["artifact_kind"], "bitnet_apple_m4_local_server_ready");
        assert_eq!(ready["ready"], true);
        assert_eq!(ready["checks"]["cache"]["ready"], true);
        assert_eq!(ready["backend"]["requested_backend"], APPLE_M4_CPU_NEON);
        assert_eq!(ready["backend"]["selected_backend"], APPLE_M4_CPU_NEON);
        assert_eq!(ready["backend"]["fallback_used"], false);
        assert_eq!(ready["checks"]["generation"]["executed"], false);
        assert_eq!(ready["claim_boundary"]["generation_endpoint_implemented"], true);
        assert_eq!(ready["claim_boundary"]["bitnet_quality_claimed"], false);
        assert_eq!(ready["claim_boundary"]["full_metal_inference_claimed"], false);
        Ok(())
    }

    #[test]
    fn mac_serve_http_ready_and_health_routes_are_json_no_generation()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_temp, state) = test_state()?;

        let (status, reason, health) =
            mac_serve_http_response("GET /health HTTP/1.1\r\n\r\n", &state);
        assert_eq!((status, reason), (200, "OK"));
        assert_eq!(health["artifact_kind"], "bitnet_apple_m4_local_server_health");
        assert_eq!(health["generation_executed"], false);

        let (status, reason, ready) =
            mac_serve_http_response("GET /ready HTTP/1.1\r\n\r\n", &state);
        assert_eq!((status, reason), (200, "OK"));
        assert_eq!(ready["artifact_kind"], "bitnet_apple_m4_local_server_ready");
        assert_eq!(ready["ready"], true);
        assert_eq!(ready["checks"]["generation"]["checked"], false);
        Ok(())
    }

    #[test]
    fn mac_serve_http_models_route_reports_catalog_no_generation()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_temp, state) = test_state()?;

        let (status, reason, models) =
            mac_serve_http_response("GET /models HTTP/1.1\r\n\r\n", &state);

        assert_eq!((status, reason), (200, "OK"));
        assert_eq!(models["artifact_kind"], "bitnet_apple_m4_local_server_models");
        assert_eq!(models["generation_executed"], false);
        assert_eq!(models["resident_model_id"], "qwen2.5-0.5b-instruct-q8_0");
        assert_eq!(models["catalog"]["default_model_id"], "qwen2.5-0.5b-instruct-q8_0");
        assert!(models["catalog"]["disk"]["default_model_headroom_bytes"].as_u64().is_some());
        let rows = models["catalog"]["rows"]
            .as_array()
            .ok_or_else(|| std::io::Error::other("model rows"))?;
        assert!(rows.iter().any(|row| {
            row["id"] == "qwen2.5-1.5b-instruct-q4_k_m" && row["state"] == "supported"
        }));
        assert!(rows.iter().any(|row| {
            row["id"] == "microsoft-bitnet-b1.58-2B-4T-i2s"
                && row["state"] == "supported-ask"
                && row["mac_ask_enabled"] == true
                && row["mac_bitnet_warm_enabled"] == true
                && row["mac_chat_enabled"] == false
                && row["mac_ask_chat_enabled"] == false
                && row["mac_serve_enabled"] == false
                && row["proof_status"]
                    == "answer-corpus-and-warm-session-proof-passed-explicit-artifact"
                && row["proof_command"].as_str().is_some_and(|command| {
                    command.contains("mac bitnet-proof") && command.contains("--proof-receipt")
                })
                && row["warm_command"].as_str().is_some_and(|command| {
                    command.contains("mac bitnet-warm") && command.contains("--tokenizer")
                })
                && row["warm_receipt_path"]
                    == "ci/hardware/apple-m4-mac-mini/2026-05-14/bitnet-warm/bitnet-mac-bitnet-warm-runtime-receipt.json"
                && row["fetch_command"]
                    .as_str()
                    .is_some_and(|command| command.contains("bitnet model fetch microsoft-bitnet"))
        }));
        assert_eq!(models["claim_boundary"]["bitnet_quality_claimed"], false);
        Ok(())
    }

    #[test]
    fn mac_serve_check_models_catalog_pass_requires_operator_catalog_shape()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_temp, state) = test_state()?;
        let models = mac_serve_models_json(&state)?;

        assert!(mac_serve_check_models_catalog_pass(200, &models));
        assert!(!mac_serve_check_models_catalog_pass(503, &models));

        let mut generated = models.clone();
        generated["generation_executed"] = serde_json::Value::Bool(true);
        assert!(!mac_serve_check_models_catalog_pass(200, &generated));

        let mut missing_bitnet_gate = models.clone();
        missing_bitnet_gate["catalog"]["rows"]
            .as_array_mut()
            .ok_or_else(|| std::io::Error::other("model rows"))?
            .retain(|row| row["id"].as_str() != Some("microsoft-bitnet-b1.58-2B-4T-i2s"));
        assert!(!mac_serve_check_models_catalog_pass(200, &missing_bitnet_gate));
        Ok(())
    }

    #[test]
    fn mac_serve_check_models_result_records_recommended_commands()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_temp, state) = test_state()?;
        let models = mac_serve_models_json(&state)?;

        let result = mac_serve_check_models_result(200, &models);

        assert_eq!(result["passed"], true);
        assert_eq!(result["recommended_first_model_id"], "qwen2.5-0.5b-instruct-q8_0");
        assert!(
            result["recommended_fetch_command"]
                .as_str()
                .ok_or_else(|| std::io::Error::other("fetch command"))?
                .contains("bitnet model fetch qwen2.5-0.5b-instruct-q8_0")
        );
        assert!(
            result["recommended_verify_command"]
                .as_str()
                .ok_or_else(|| std::io::Error::other("verify command"))?
                .contains("bitnet model verify qwen2.5-0.5b-instruct-q8_0")
        );
        Ok(())
    }

    #[test]
    fn mac_ask_operator_summary_line_reports_model_cache_backend_and_receipt() {
        let temp = tempfile::tempdir().expect("tempdir");
        let model = test_verified_model(temp.path());
        let receipt = temp.path().join("mac-ask.json");
        let summary = mac_ask_operator_summary_line(&model, &receipt);

        assert!(summary.contains("mac ask: model=qwen2.5-0.5b-instruct-q8_0"));
        assert!(summary.contains("quant=Q8_0"));
        assert!(summary.contains("cache=verified"));
        assert!(summary.contains(&format!("cache_root={}", temp.path().display())));
        assert!(summary.contains("backend=apple-m4-cpu-neon"));
        assert!(summary.contains("fallback=false"));
        assert!(summary.contains(&format!("receipt={}", receipt.display())));
        assert!(summary.contains("sha256=ca59ca7f13d0"));
    }

    #[test]
    fn mac_serve_listening_line_advertises_model_catalog_endpoint() {
        let line = mac_serve_listening_line("127.0.0.1:8080");

        assert_eq!(
            line,
            "bitnet mac serve listening on http://127.0.0.1:8080 (health: /health, models: /models, ready: /ready)"
        );
    }

    #[test]
    fn mac_serve_ready_json_requires_receipt_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let receipt_path = temp.path().join("receipt-file");
        std::fs::write(&receipt_path, b"not a directory").expect("receipt file");
        let state = MacServeState::new(
            test_verified_model(temp.path()),
            "127.0.0.1".to_string(),
            8080,
            true,
            MacServeGenerationDefaults {
                max_new_tokens: 8,
                temperature: 0.0,
                top_k: 1,
                top_p: 1.0,
                repetition_penalty: 1.1,
                seed: None,
            },
            receipt_path,
            None,
        );

        let ready = mac_serve_ready_json(&state);

        assert_eq!(ready["ready"], false);
        assert_eq!(ready["status"], "not_ready");
        assert_eq!(ready["checks"]["receipts"]["ready"], false);
    }

    #[test]
    fn mac_serve_receipt_dir_probe_rejects_non_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let receipt_path = temp.path().join("receipt-file");
        std::fs::write(&receipt_path, b"not a directory").expect("receipt file");

        assert!(!mac_serve_receipt_dir_ready(&receipt_path));
        assert!(ensure_mac_serve_receipt_dir_ready(&receipt_path).is_err());
    }

    #[test]
    fn mac_serve_http_rejects_unknown_paths_and_methods() -> Result<(), Box<dyn std::error::Error>>
    {
        let (_temp, state) = test_state()?;

        let (status, reason, body) =
            mac_serve_http_response("POST /ready HTTP/1.1\r\n\r\n", &state);
        assert_eq!((status, reason), (405, "Method Not Allowed"));
        assert_eq!(body["error"], "method_not_allowed");

        let (status, reason, body) =
            mac_serve_http_response("GET /v1/chat/completions HTTP/1.1\r\n\r\n", &state);
        assert_eq!((status, reason), (404, "Not Found"));
        assert_eq!(body["error"], "not_found");
        Ok(())
    }

    #[tokio::test]
    async fn mac_serve_receipt_endpoint_exports_existing_receipt()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_temp, state) = test_state()?;
        let receipt_path = state.receipt_dir.join("m4srv-test.json");
        std::fs::write(
            &receipt_path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "artifact_kind": "bitnet_apple_m4_local_server_completion",
                "request_id": "m4srv-test",
                "selected_backend": APPLE_M4_CPU_NEON,
                "fallback_used": false,
            }))
            .expect("receipt json"),
        )
        .expect("write receipt");

        let reply = mac_serve_http_reply("GET /receipts/m4srv-test HTTP/1.1\r\n\r\n", &state)
            .await
            .expect("reply");
        assert_eq!(reply.status, 200);
        assert_eq!(reply.content_type, "application/json");
        let body: serde_json::Value = serde_json::from_slice(&reply.body).expect("json body");
        assert_eq!(body["artifact_kind"], "bitnet_apple_m4_local_server_completion");
        assert_eq!(body["request_id"], "m4srv-test");
        assert_eq!(body["selected_backend"], APPLE_M4_CPU_NEON);
        assert_eq!(body["fallback_used"], false);
        Ok(())
    }

    #[tokio::test]
    async fn mac_serve_receipt_endpoint_rejects_unsafe_or_missing_receipts()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_temp, state) = test_state()?;

        let unsafe_reply = mac_serve_http_reply("GET /receipts/../secret HTTP/1.1\r\n\r\n", &state)
            .await
            .expect("unsafe reply");
        assert_eq!(unsafe_reply.status, 400);
        let unsafe_body: serde_json::Value =
            serde_json::from_slice(&unsafe_reply.body).expect("json body");
        assert_eq!(unsafe_body["error"], "invalid_receipt_id");

        let missing_reply =
            mac_serve_http_reply("GET /receipts/m4srv-missing HTTP/1.1\r\n\r\n", &state)
                .await
                .expect("missing reply");
        assert_eq!(missing_reply.status, 404);
        let missing_body: serde_json::Value =
            serde_json::from_slice(&missing_reply.body).expect("json body");
        assert_eq!(missing_body["error"], "receipt_not_found");
        Ok(())
    }

    #[tokio::test]
    async fn mac_serve_completion_endpoint_reports_unavailable_without_generator()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_temp, state) = test_state()?;
        let request = concat!(
            "POST /v1/chat/completions HTTP/1.1\r\n",
            "content-type: application/json\r\n",
            "content-length: 39\r\n",
            "\r\n",
            "{\"prompt\":\"What is 2+2?\",\"stream\":false}"
        );
        let reply = mac_serve_http_reply(request, &state).await.expect("reply");
        assert_eq!(reply.status, 503);
        assert_eq!(reply.content_type, "application/json");
        let body: serde_json::Value = serde_json::from_slice(&reply.body).expect("json body");
        assert_eq!(body["error"], "generation_unavailable");
        Ok(())
    }

    #[tokio::test]
    async fn mac_serve_completion_endpoint_rejects_invalid_json()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_temp, state) = test_state()?;
        let request = concat!(
            "POST /v1/chat/completions HTTP/1.1\r\n",
            "content-type: application/json\r\n",
            "content-length: 5\r\n",
            "\r\n",
            "{bad}"
        );
        let state = MacServeState { generator: None, ..state };
        let reply = mac_serve_http_reply(request, &state).await.expect("reply");
        assert_eq!(reply.status, 400);
        let body: serde_json::Value = serde_json::from_slice(&reply.body).expect("json body");
        assert_eq!(body["error"], "invalid_completion_request");
        Ok(())
    }

    #[tokio::test]
    async fn mac_serve_completion_endpoint_rejects_wrong_model_id()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_temp, state) = test_state()?;
        let request = concat!(
            "POST /v1/chat/completions HTTP/1.1\r\n",
            "content-type: application/json\r\n",
            "content-length: 64\r\n",
            "\r\n",
            "{\"model\":\"other-model\",\"prompt\":\"What is 2+2?\",\"stream\":false}"
        );
        let reply = mac_serve_http_reply(request, &state).await.expect("reply");
        assert_eq!(reply.status, 400);
        let body: serde_json::Value = serde_json::from_slice(&reply.body).expect("json body");
        assert_eq!(body["error"], "unsupported_model");
        Ok(())
    }

    #[test]
    fn mac_serve_completion_request_accepts_prompt_or_chat_messages() {
        let prompt_request: MacServeCompletionRequest =
            serde_json::from_str(r#"{"prompt":"What is 2+2?"}"#).expect("prompt request");
        assert_eq!(prompt_request.prompt_text().expect("prompt"), "What is 2+2?");

        let chat_request: MacServeCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Name the capital of France."}]}"#,
        )
        .expect("chat request");
        assert_eq!(chat_request.prompt_text().expect("chat prompt"), "Name the capital of France.");
        assert_eq!(chat_request.system_prompt().as_deref(), Some("Be brief."));
    }

    #[test]
    fn mac_serve_host_loopback_detection_warns_only_for_non_loopback() {
        assert!(mac_serve_host_is_loopback("127.0.0.1"));
        assert!(mac_serve_host_is_loopback("localhost"));
        assert!(mac_serve_host_is_loopback("::1"));
        assert!(mac_serve_host_is_loopback("[::1]"));
        assert!(!mac_serve_host_is_loopback("0.0.0.0"));
        assert!(!mac_serve_host_is_loopback("192.168.1.5"));
    }

    #[test]
    fn mac_serve_check_url_parser_requires_http_host_port() {
        assert_eq!(
            MacServeCheckEndpoint::parse("http://127.0.0.1:8080").expect("url"),
            MacServeCheckEndpoint { host: "127.0.0.1".to_string(), port: 8080 }
        );
        assert_eq!(
            MacServeCheckEndpoint::parse("http://[::1]:8080").expect("ipv6 url"),
            MacServeCheckEndpoint { host: "::1".to_string(), port: 8080 }
        );
        assert!(MacServeCheckEndpoint::parse("https://127.0.0.1:8080").is_err());
        assert!(MacServeCheckEndpoint::parse("http://127.0.0.1").is_err());
        assert!(MacServeCheckEndpoint::parse("http://127.0.0.1:8080/path").is_err());
    }
}
