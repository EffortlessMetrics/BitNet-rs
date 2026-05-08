//! Mac-oriented operator wrappers for the supported Apple M4 SLM path.

use anyhow::{Context, Result, anyhow};
use clap::{Args, Subcommand, ValueEnum};
use serde::Serialize;
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};

use crate::model_cache::{self, VerifiedCachedModel};

const APPLE_M4_CPU_NEON: &str = "apple-m4-cpu-neon";
const APPLE_M4_METAL: &str = "apple-m4-metal";
const MAC_ASK_DEFAULT_RECEIPT: &str = "target/apple-m4-productization/mac-ask.json";
const MAC_VALIDATE_DEFAULT_RECEIPT: &str = "target/apple-m4-productization/mac-validate.json";
const MAC_VALIDATE_DEFAULT_CORPUS: &str = "ci/quality/apple-m4-slm-quality-corpus.yaml";
const QWEN_PROMPT_TEMPLATE: &str = "qwen2.5";
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

/// Run the supported Apple M4 local-answer flow with strict receipts.
#[derive(Debug, Args)]
pub struct MacCommand {
    #[command(subcommand)]
    action: MacAction,
}

#[derive(Debug, Subcommand)]
enum MacAction {
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
        /// Question to answer.
        #[arg(short, long)]
        question: String,

        /// Supported model id. Defaults to the validated Apple M4 SLM runtime artifact.
        #[arg(long, default_value = model_cache::M4_SLM_RUNTIME_MODEL_ID)]
        model_id: String,

        /// Override model cache root. Defaults to ~/.cache/bitnet-rs/models.
        #[arg(long, value_name = "PATH")]
        cache_dir: Option<PathBuf>,

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

        /// Output aggregate warm-session receipt.
        #[arg(long, value_name = "PATH", default_value = MAC_VALIDATE_DEFAULT_RECEIPT)]
        json_out: PathBuf,
    },

    /// Check Apple M4 SLM answer/warm-session receipts for hidden fallback or overclaims.
    ReceiptsCheck {
        /// Receipt file or directory containing JSON receipts.
        path: PathBuf,

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
}

impl MacCommand {
    pub async fn execute(self, explicit_device_label: Option<&str>) -> Result<()> {
        match self.action {
            MacAction::Check { model_id, cache_dir, json } => {
                ensure_supported_mac_device(explicit_device_label, "mac check")?;
                run_check(&model_id, cache_dir, json)
            }
            MacAction::Ask {
                question,
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
                json_out,
            } => {
                ensure_supported_mac_device(explicit_device_label, "mac ask")?;
                run_ask(
                    &model_id,
                    cache_dir,
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
                )
                .await
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
                json_out,
            } => {
                ensure_supported_mac_device(explicit_device_label, "mac validate")?;
                run_validate(
                    &model_id,
                    cache_dir,
                    corpus,
                    corpus_repeat_runs,
                    profile_set,
                    max_new_tokens,
                    threads,
                    allocation_audit,
                    json_out,
                )
                .await
            }
            MacAction::ReceiptsCheck { path, json } => run_receipts_check(&path, json),
        }
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
) -> Result<()> {
    let model = model_cache::verified_apple_m4_slm_model(model_id, cache_dir)?;
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
        system_prompt,
        vec!["<|im_end|>".to_string()],
        Vec::new(),
        None,
        10,
        false,
        false,
        Some("mac_ask".to_string()),
        false,
    )
    .await?;
    annotate_and_validate_mac_receipt(&json_out, &model, "mac ask")?;
    Ok(())
}

async fn run_validate(
    model_id: &str,
    cache_dir: Option<PathBuf>,
    corpus: PathBuf,
    corpus_repeat_runs: usize,
    profile_set: MacValidateProfileSet,
    max_new_tokens: usize,
    threads: usize,
    allocation_audit: bool,
    json_out: PathBuf,
) -> Result<()> {
    if profile_set == MacValidateProfileSet::Performance && cfg!(debug_assertions) {
        anyhow::bail!(
            "mac validate --profile-set performance must be run from a release build; use `cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- mac validate --profile-set performance ...`"
        );
    }
    let model = model_cache::verified_apple_m4_slm_model(model_id, cache_dir)?;
    if profile_set == MacValidateProfileSet::Operator {
        return run_operator_profiles(model, json_out, threads, allocation_audit).await;
    }
    if profile_set == MacValidateProfileSet::Performance {
        return run_performance_profiles(model, json_out, threads, allocation_audit).await;
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
        },
    )
    .await
}

async fn run_performance_profiles(
    model: VerifiedCachedModel,
    json_out: PathBuf,
    threads: usize,
    allocation_audit: bool,
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
            "decode_total_ms": receipt["speed"]["timing"]["decode_total_ms"].clone(),
            "sampling_ms": receipt["speed"]["timing"]["sampling_ms"].clone(),
            "warm_prompt_generated_tok_s": receipt["speed"]["throughput"]["warm_prompt_generated_tok_s"].clone(),
            "decode_generated_tok_s": receipt["speed"]["throughput"]["decode_generated_tok_s"].clone(),
        },
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
    std::fs::write(path, serde_json::to_vec_pretty(&receipt)?)
        .with_context(|| format!("failed to update Mac receipt {}", path.display()))?;
    println!(
        "Mac receipt checked: {} ({}, generated_tokens={:?})",
        path.display(),
        summary.artifact_kind,
        summary.generated_tokens
    );
    Ok(())
}

fn run_receipts_check(path: &Path, json: bool) -> Result<()> {
    let receipt_paths = collect_receipt_paths(path)?;
    if receipt_paths.is_empty() {
        anyhow::bail!("no JSON receipts found under {}", path.display());
    }
    let mut summaries = Vec::with_capacity(receipt_paths.len());
    for receipt_path in receipt_paths {
        let receipt: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&receipt_path)
                .with_context(|| format!("failed to read {}", receipt_path.display()))?,
        )
        .with_context(|| format!("invalid JSON receipt {}", receipt_path.display()))?;
        summaries.push(validate_mac_receipt_value(&receipt_path, &receipt)?);
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
        }
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
    })
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
    if metal_phase["execution_phase"].as_str() != Some("prefill_linear_projection") {
        anyhow::bail!(
            "{} Metal phase must be execution_phase=prefill_linear_projection",
            path.display()
        );
    }
    if metal_phase["kernel_id"].as_str().is_none() {
        anyhow::bail!("{} Metal phase receipt is missing kernel_id", path.display());
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
    for field in ["batch_size", "in_features", "out_features"] {
        if layout[field].as_u64().unwrap_or_default() == 0 {
            anyhow::bail!("{} Metal phase layout is missing {field}", path.display());
        }
    }

    let parity = &receipt["parity"];
    if parity["reference_backend"].as_str() != Some(APPLE_M4_CPU_NEON)
        || parity["target_backend"].as_str() != Some(APPLE_M4_METAL)
        || parity["greedy_token_ids_match_cpu_reference"].as_bool() != Some(true)
    {
        anyhow::bail!(
            "{} Metal phase parity must match CPU/NEON reference greedy token IDs",
            path.display()
        );
    }
    if parity["max_abs_error"].is_null() || parity["mean_abs_error"].is_null() {
        anyhow::bail!("{} Metal phase parity is missing error metrics", path.display());
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
    })
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
    let prompts = receipt["prompts"].as_array().ok_or_else(|| {
        anyhow!("{} warm-session receipt is missing prompt summaries", path.display())
    })?;
    if prompts.is_empty() {
        anyhow::bail!("{} warm-session receipt has no prompts", path.display());
    }
    let mut generated_total = 0usize;
    for prompt in prompts {
        if prompt["backend"]["selected_backend"] != APPLE_M4_CPU_NEON {
            anyhow::bail!("{} warm-session prompt selected a non-Mac CPU backend", path.display());
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
        generated_total += generated;
    }
    Ok((Some(prompts.len()), Some(generated_total)))
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
