//! Mac-oriented operator wrappers for the supported Apple M4 SLM path.

use anyhow::{Context, Result, anyhow};
use clap::{Args, Subcommand};
use serde::Serialize;
use std::path::{Path, PathBuf};

use crate::model_cache::{self, VerifiedCachedModel};

const APPLE_M4_CPU_NEON: &str = "apple-m4-cpu-neon";
const MAC_ASK_DEFAULT_RECEIPT: &str = "target/apple-m4-productization/mac-ask.json";
const MAC_VALIDATE_DEFAULT_RECEIPT: &str = "target/apple-m4-productization/mac-validate.json";
const MAC_VALIDATE_DEFAULT_CORPUS: &str = "ci/quality/apple-m4-slm-quality-corpus.yaml";
const QWEN_PROMPT_TEMPLATE: &str = "qwen2.5";

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

        /// Maximum new tokens per prompt when the corpus does not override it.
        #[arg(long, visible_aliases = ["max-tokens", "n-predict"], default_value_t = 32)]
        max_new_tokens: usize,

        /// Number of CPU threads to use (0 = all cores; deterministic mode may override).
        #[arg(long, default_value_t = 0)]
        threads: usize,

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
                max_new_tokens,
                threads,
                json_out,
            } => {
                ensure_supported_mac_device(explicit_device_label, "mac validate")?;
                run_validate(
                    &model_id,
                    cache_dir,
                    corpus,
                    corpus_repeat_runs,
                    max_new_tokens,
                    threads,
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
    max_new_tokens: usize,
    threads: usize,
    json_out: PathBuf,
) -> Result<()> {
    let model = model_cache::verified_apple_m4_slm_model(model_id, cache_dir)?;
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
        1,
        1,
        json_out.clone(),
    )
    .await?;
    annotate_and_validate_mac_receipt(&json_out, &model, "mac validate")?;
    Ok(())
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
