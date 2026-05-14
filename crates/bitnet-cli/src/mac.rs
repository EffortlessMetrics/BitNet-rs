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
const APPLE_M4_METAL: &str = "apple-m4-metal";
const MAC_ASK_DEFAULT_RECEIPT: &str = "target/apple-m4-productization/mac-ask.json";
const MAC_CHAT_DEFAULT_RECEIPT: &str = "target/apple-m4-continuity/mac-chat.json";
const MAC_SMOKE_DEFAULT_RECEIPT: &str = "target/apple-m4-continuity/mac-smoke.json";
const MAC_DOCTOR_DEFAULT_RECEIPT: &str = "target/apple-m4-slm-excellence/mac-doctor.json";
const MAC_SERVE_DEFAULT_RECEIPT_DIR: &str = "target/apple-m4-local-server/receipts";
const MAC_SERVE_CHECK_DEFAULT_RECEIPT: &str = "target/apple-m4-local-server/mac-serve-check.json";
const MAC_SERVE_DEFAULT_HOST: &str = "127.0.0.1";
const MAC_SERVE_DEFAULT_PORT: u16 = 8080;
const MAC_SERVE_DEFAULT_MAX_NEW_TOKENS: usize = 64;
const MAC_BITNET_PROOF_DEFAULT_RECEIPT: &str =
    "target/apple-m4-continuity/mac-bitnet-proof-preflight.json";
const MAC_VALIDATE_DEFAULT_RECEIPT: &str = "target/apple-m4-productization/mac-validate.json";
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MacValidateProfileSet {
    /// Run the deterministic smoke corpus once.
    Smoke,
    /// Run bounded 16/32/64 warm-answer timing profiles and write an aggregate summary.
    Operator,
    /// Run release-mode 16/32/64/128 warm-answer timing profiles.
    Performance,
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

    /// Run a compact Apple M4 dense-SLM health smoke with cache and receipt checks.
    Smoke {
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

        /// Output aggregate golden smoke receipt.
        #[arg(long, value_name = "PATH", default_value = MAC_SMOKE_DEFAULT_RECEIPT)]
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

        /// Compare Apple M4 dense SLM receipts against a published baseline receipt.
        #[arg(long = "regression-baseline", value_name = "PATH")]
        regression_baseline: Option<PathBuf>,

        /// Emit JSON instead of text.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Compare Apple M4 dense-SLM receipts against a stored local envelope.
    Regression {
        /// Current receipt file or directory containing JSON receipts.
        path: PathBuf,

        /// Stored M4 dense-SLM envelope receipt to compare against.
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
                    json_out,
                    progress,
                    quiet,
                )
                .await
            }
            MacAction::Smoke { model_id, cache_dir, max_new_tokens, threads, json_out } => {
                ensure_supported_mac_device(explicit_device_label, "mac smoke")?;
                run_smoke(&model_id, cache_dir, max_new_tokens, threads, json_out).await
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
                ensure_supported_mac_device(explicit_device_label, "mac validate")?;
                run_validate(MacValidateRun {
                    model_id: &model_id,
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
            "M4 BitNet proof receipt verified; BitNet remains limited to explicit one-shot `bitnet mac ask` and does not enable `bitnet mac chat` or `bitnet mac serve`. Receipt: {}",
            json_out.display()
        );
    } else {
        println!(
            "M4 BitNet proof preflight passed; BitNet remains limited to explicit one-shot `bitnet mac ask` and does not enable `bitnet mac chat` or `bitnet mac serve`. Receipt: {}",
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
            json_out,
            progress,
            quiet,
        )
        .await;
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
    if let Err(error) = crate::run_simple_generation(
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
    )
    .await
    {
        let message = error.to_string();
        return fail_bitnet_mac_ask_with_receipt(
            &json_out,
            BitNetMacAskFailureContext {
                model_path: Some(model.path.clone()),
                tokenizer_path: Some(tokenizer.clone()),
                ..failure_context.clone()
            },
            "generation_failed",
            &message,
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
    started_at: std::time::Instant,
}

fn fail_bitnet_mac_ask_with_receipt(
    json_out: &Path,
    context: BitNetMacAskFailureContext,
    stage: &str,
    message: &str,
) -> Result<()> {
    let repair_guidance = bitnet_mac_ask_failure_repair_guidance(stage, &context);
    let repair_text = bitnet_mac_ask_failure_repair_text(&repair_guidance);
    if let Err(receipt_error) =
        write_bitnet_mac_ask_failure_receipt(json_out, context, stage, message, &repair_guidance)
    {
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
    repair_guidance: &[String],
) -> Result<()> {
    let elapsed_ms = context.started_at.elapsed().as_secs_f64() * 1000.0;
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
        },
        "failure": {
            "stage": stage,
            "message": message,
            "elapsed_ms": (elapsed_ms * 1000.0).round() / 1000.0,
        },
        "timeout_boundary": {
            "configured_seconds": serde_json::Value::Null,
            "reached": false,
            "enforced": false,
            "status": "not_reached",
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
    if stage.contains("model") || stage == "generation_failed" {
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
    model_id: &str,
    cache_dir: Option<PathBuf>,
    max_new_tokens: usize,
    threads: usize,
    json_out: PathBuf,
) -> Result<()> {
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
        &model.id,
        Some(model.cache_root.clone()),
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
                && row["mac_chat_enabled"].as_bool() == Some(false)
                && row["mac_ask_chat_enabled"].as_bool() == Some(false)
                && row["mac_serve_enabled"].as_bool() == Some(false)
                && row["proof_status"].as_str()
                    == Some("answer-corpus-proof-passed-one-shot-ask-explicit-artifact")
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
    let cached_ask_ready = cached_model_ready && tokenizer_verified;

    serde_json::json!({
        "checked": true,
        "ready": cached_ask_ready,
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

fn bitnet_mac_ask_readiness_claim_boundary() -> serde_json::Value {
    serde_json::json!({
        "bitnet_one_shot_mac_ask": true,
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
        return run_operator_profiles(model, json_out, threads, allocation_audit, progress, quiet)
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
            json_out,
            threads,
            allocation_audit,
            progress,
            quiet,
        )
        .await;
    }
    crate::run_slm_warm_session(
        APPLE_M4_CPU_NEON,
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
    json_out: PathBuf,
    threads: usize,
    allocation_audit: bool,
    progress: bool,
    quiet: bool,
) -> Result<()> {
    run_warm_profile_set(
        model,
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
            APPLE_M4_CPU_NEON,
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
        "artifact_kind": spec.artifact_kind,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "artifact_path": json_out.display().to_string(),
        "requested_backend": APPLE_M4_CPU_NEON,
        "selected_backend": APPLE_M4_CPU_NEON,
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
            "scope": "supported Apple M4 SLM warm-answer timing only",
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
        "allocation_audit": profile_set_allocation_audit_json(&summaries, spec.allocation_audit),
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
        "scope": "selected Apple M4 CPU/NEON SLM warm-session profile set",
        "claim_scope": "aggregate of prompt-level allocation counter deltas; no optimization or performance improvement claimed",
        "profile_count": summaries.len(),
        "ranked_hotspots": ranked,
        "optimization_deferred": true,
    })
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
            "requested_backend": APPLE_M4_CPU_NEON,
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
            summary.regression =
                Some(compare_dense_slm_regression(&receipt_path, &receipt, baseline)?);
            regression_compared += 1;
        }
        summaries.push(summary);
    }
    if baseline.is_some() && regression_compared == 0 {
        anyhow::bail!(
            "regression baseline was provided, but no matching Apple M4 dense SLM receipts were found under {}",
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
            match compare_dense_slm_regression(&receipt_path, &receipt, &baseline) {
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
            "no matching Apple M4 dense SLM receipts under {} could be compared to baseline {}",
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
    ) {
        anyhow::bail!(
            "regression baseline {} must be an apple_m4_slm_performance_profiles, slm_apple_m4_warm_session, or apple_m4_slm_eval_summary receipt",
            path.display()
        );
    }
    Ok(RegressionBaseline { path: path.to_path_buf(), receipt })
}

fn compare_dense_slm_regression(
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
        _ => anyhow::bail!("{} is not an Apple M4 dense SLM envelope receipt", path.display()),
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

    if requested_backend != APPLE_M4_CPU_NEON {
        anyhow::bail!(
            "{} requested_backend must be {APPLE_M4_CPU_NEON}, got {requested_backend:?}",
            path.display()
        );
    }
    if selected_backend != APPLE_M4_CPU_NEON {
        anyhow::bail!(
            "{} selected_backend must be {APPLE_M4_CPU_NEON}, got {selected_backend:?}",
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

    let (prompt_count, generated_tokens) = if artifact_kind == "slm_apple_m4_warm_session" {
        validate_warm_session_receipt(path, receipt)?
    } else if artifact_kind == "apple_m4_slm_operator_profiles"
        || artifact_kind == "apple_m4_slm_performance_profiles"
    {
        validate_profile_set_receipt(path, receipt, artifact_kind.as_str())?
    } else if artifact_kind == "apple_m4_slm_eval_summary" {
        validate_slm_eval_summary_receipt(path, receipt)?
    } else if artifact_kind == "bitnet_apple_m4_mac_ask_failure" {
        validate_bitnet_mac_ask_failure_receipt(path, receipt)?
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
    if receipt["timeout_boundary"]["reached"].as_bool() != Some(false)
        || receipt["timeout_boundary"]["enforced"].as_bool() != Some(false)
        || receipt["timeout_boundary"]["status"].as_str() != Some("not_reached")
    {
        anyhow::bail!(
            "{} BitNet Mac ask failure receipt must record an explicit non-reached timeout boundary",
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

fn validate_profile_set_receipt(
    path: &Path,
    receipt: &serde_json::Value,
    artifact_kind: &str,
) -> Result<(Option<usize>, Option<usize>)> {
    let required = match artifact_kind {
        "apple_m4_slm_operator_profiles" => {
            &[("warm_16", 16_u64), ("warm_32", 32_u64), ("warm_64", 64_u64)][..]
        }
        "apple_m4_slm_performance_profiles" => {
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
    if artifact_kind == "apple_m4_slm_performance_profiles" {
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
        if artifact_kind == "apple_m4_slm_performance_profiles" {
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
        if prompt["backend"]["requested_backend"].as_str() != Some(APPLE_M4_CPU_NEON) {
            anyhow::bail!("{} warm-session prompt requested a non-Mac CPU backend", path.display());
        }
        if prompt["backend"]["selected_backend"] != APPLE_M4_CPU_NEON {
            anyhow::bail!("{} warm-session prompt selected a non-Mac CPU backend", path.display());
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
            started_at: std::time::Instant::now(),
        };

        let guidance = bitnet_mac_ask_failure_repair_guidance("model_verify_failed", &context);
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
            started_at: std::time::Instant::now(),
        };

        let guidance = bitnet_mac_ask_failure_repair_guidance("model_verify_failed", &context);
        let joined = guidance.join("\n");

        assert!(joined.contains("replace --model-path with the accepted Microsoft I2_S GGUF"));
        assert!(joined.contains("bitnet model verify microsoft-bitnet-b1.58-2B-4T-i2s --path"));
        assert!(joined.contains(&model_path.display().to_string()));
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
                && row["mac_chat_enabled"] == false
                && row["mac_ask_chat_enabled"] == false
                && row["mac_serve_enabled"] == false
                && row["proof_status"]
                    == "answer-corpus-proof-passed-one-shot-ask-explicit-artifact"
                && row["proof_command"].as_str().is_some_and(|command| {
                    command.contains("mac bitnet-proof") && command.contains("--proof-receipt")
                })
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
