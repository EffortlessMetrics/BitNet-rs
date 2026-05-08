//! Answer corpus runner for CPU-first and Apple M4 local-answer baselines.

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    ffi::OsString,
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const RTX_5070_TI_CUDA: &str = "nvidia-rtx-5070-ti-cuda";

/// Run the fixed answer corpus through the existing `bitnet run` surface.
#[derive(Args, Debug)]
pub struct AnswerCorpusCommand {
    /// Path to the answer corpus YAML.
    #[arg(long, default_value = "ci/quality/bitnet-answer-corpus.yaml", value_name = "PATH")]
    pub corpus: PathBuf,

    /// Official BitNet GGUF model path.
    #[arg(long, value_name = "PATH")]
    pub model: PathBuf,

    /// Explicit tokenizer path. If omitted, the run path must resolve one strictly.
    #[arg(long, value_name = "PATH")]
    pub tokenizer: Option<PathBuf>,

    /// Backend label for this baseline.
    #[arg(long, value_name = "BACKEND")]
    pub device: Option<String>,

    /// Output aggregate answer-corpus receipt.
    #[arg(
        long,
        value_name = "PATH",
        default_value = "target/bitnet/receipts/cpu-answer-corpus.json"
    )]
    pub json_out: PathBuf,

    /// Do not invoke model generation; validate corpus shape and emit not_run rows.
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,

    /// Per-prompt timeout for child `bitnet run` invocations.
    #[arg(long, value_name = "SECONDS")]
    pub per_prompt_timeout_seconds: Option<u64>,

    /// Fail the command if any executed prompt fails its quality gate.
    #[arg(long, default_value_t = false)]
    pub fail_on_quality: bool,

    /// Dump this many per-step logit records into each child run receipt.
    #[arg(long, value_name = "N")]
    pub dump_logit_steps: Option<usize>,

    /// Number of top logits to include when --dump-logit-steps is used.
    #[arg(long, default_value_t = 10, value_name = "K")]
    pub logits_topk: usize,

    /// CPU kernel lane to request for child strict CPU runs.
    #[arg(long, value_enum, value_name = "KERNEL")]
    pub cpu_kernel: Option<AnswerCpuKernel>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum AnswerCpuKernel {
    Scalar,
    Avx2,
    Avx512,
}

impl AnswerCpuKernel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
        }
    }

    fn child_env(self) -> Vec<(&'static str, &'static str)> {
        match self {
            Self::Scalar => vec![("BITNET_CPU_KERNEL", "scalar"), ("BITNET_FORCE_SCALAR", "1")],
            Self::Avx2 => vec![("BITNET_CPU_KERNEL", "avx2"), ("BITNET_FORCE_SCALAR", "0")],
            Self::Avx512 => vec![("BITNET_CPU_KERNEL", "avx512"), ("BITNET_FORCE_SCALAR", "0")],
        }
    }
}

impl AnswerCorpusCommand {
    /// Execute the answer corpus runner.
    pub async fn execute(&self, default_device: &str) -> Result<()> {
        let corpus = AnswerCorpus::load(&self.corpus)?;
        let device =
            normalize_answer_corpus_device(self.device.as_deref().unwrap_or(default_device));
        if !matches!(device.as_str(), "cpu" | "apple-m4-cpu-neon" | "cuda" | RTX_5070_TI_CUDA) {
            anyhow::bail!(
                "answer-corpus only accepts --device cpu, --device apple-m4-cpu-neon, --device cuda, or --device {RTX_5070_TI_CUDA}; got {device}"
            );
        }
        if self.cpu_kernel.is_some() && device != "cpu" {
            anyhow::bail!("--cpu-kernel is only valid with --device cpu");
        }
        if self.cpu_kernel == Some(AnswerCpuKernel::Avx2) && !cpu_avx2_available() {
            anyhow::bail!("--cpu-kernel avx2 requested but AVX2 is unavailable on this host");
        }
        if self.cpu_kernel == Some(AnswerCpuKernel::Avx512) && !cpu_avx512_available() {
            anyhow::bail!("--cpu-kernel avx512 requested but AVX512 is unavailable on this host");
        }
        let artifact_kind = answer_corpus_artifact_kind(&device, &corpus.artifact_kind);
        let default_timeout_seconds = effective_default_timeout_seconds(
            self.per_prompt_timeout_seconds,
            corpus.defaults.per_prompt_timeout_seconds,
        );

        let receipt_dir = self
            .json_out
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
            .join(format!(
                "{}-runs",
                self.json_out.file_stem().and_then(|stem| stem.to_str()).unwrap_or("answer-corpus")
            ));
        fs::create_dir_all(&receipt_dir)?;

        let exe = std::env::current_exe().context("failed to resolve current bitnet executable")?;
        let mut rows = Vec::with_capacity(corpus.cases.len());
        for case in &corpus.cases {
            let row = if self.dry_run {
                self.not_run_row(case, "dry_run_requested")
            } else {
                self.run_case(&exe, &receipt_dir, &corpus, case, &device, default_timeout_seconds)?
            };
            rows.push(row);
        }

        let total = rows.len();
        let passed = rows.iter().filter(|row| row["quality"]["passed"] == true).count();
        let failed = rows
            .iter()
            .filter(|row| row["status"] == "quality_failed" || row["status"] == "command_failed")
            .count();
        let timed_out = rows.iter().filter(|row| row["status"] == "timeout").count();
        let not_run = rows.iter().filter(|row| row["status"] == "not_run").count();
        let aggregate_tokenizer =
            match (corpus.model.family.as_deref(), corpus.defaults.prompt_template.as_str()) {
                (_, "bitnetcpp-answer") => "externally_supplied_llama_bpe",
                (Some("qwen"), _) => "gguf_metadata",
                _ => "llama3",
            };

        let receipt = json!({
            "schema_version": "1.0.0",
            "artifact_kind": artifact_kind,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "corpus": {
                "path": self.corpus.display().to_string(),
                "name": corpus.name,
                "description": corpus.description,
                "case_count": corpus.cases.len(),
            },
            "model": {
                "repo": corpus.model.repo,
                "file": corpus.model.file,
                "sha256": corpus.model.sha256,
                "family": corpus.model.family,
                "architecture": corpus.model.architecture,
                "quant_format": corpus.model.quant_format,
                "path": self.model.display().to_string(),
                "loader_mode": "real_gguf",
                "fallback_loader_used": false,
                "tokenizer": aggregate_tokenizer,
                "tokenizer_path": self.tokenizer.as_ref().map(|path| path.display().to_string()),
            },
            "backend": {
                "requested_backend": device.as_str(),
                "selected_backend": device.as_str(),
                "runtime_api": answer_corpus_runtime_api(&device),
                "fallback_used": false,
            },
            "prompt_template": {
                "family": corpus.defaults.prompt_template,
            },
            "generation": {
                "mode": if corpus.defaults.greedy { "greedy" } else { "sampling" },
                "temperature": corpus.defaults.temperature,
                "deterministic": corpus.defaults.deterministic,
                "strict_loader": corpus.defaults.strict_loader,
                "default_max_new_tokens": corpus.defaults.max_new_tokens,
                "per_prompt_timeout_seconds": default_timeout_seconds,
                "logits_dump_steps": self.dump_logit_steps,
                "logits_topk": if self.dump_logit_steps.is_some() {
                    Some(self.logits_topk)
                } else {
                    None
                },
                "requested_cpu_kernel": self.cpu_kernel.map(AnswerCpuKernel::as_str),
            },
            "quality_summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "timeout": timed_out,
                "not_run": not_run,
            },
            "claim_boundary": {
                "slm_answer_path": corpus.artifact_kind == "slm_answer_corpus",
                "local_answer_path": device.as_str() == "apple-m4-cpu-neon",
                "diagnostic_only_until_answer_ready_artifact": true,
                "coherent_answer_claimed": false,
                "cuda_answer_corpus": is_cuda_answer_corpus_device(&device),
                "strict_cuda_answer_claimed": false,
                "full_metal_inference_claimed": false,
                "qk256_apple_claimed": false,
                "neural_engine_claimed": false,
                "broad_performance_claimed": false,
            },
            "cases": rows,
            "speedup_claim": false,
        });

        if let Some(parent) = self.json_out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&self.json_out, serde_json::to_vec_pretty(&receipt)?)?;
        println!("answer corpus receipt written to {}", self.json_out.display());

        if self.fail_on_quality && (failed > 0 || timed_out > 0) {
            anyhow::bail!("answer corpus quality failed: {failed} failed, {timed_out} timed out");
        }
        Ok(())
    }

    fn run_case(
        &self,
        exe: &Path,
        receipt_dir: &Path,
        corpus: &AnswerCorpus,
        case: &AnswerCase,
        device: &str,
        default_timeout_seconds: u64,
    ) -> Result<Value> {
        let case_receipt = receipt_dir.join(format!("{}.json", sanitize_file_stem(&case.id)));
        let max_new_tokens = case.max_new_tokens.unwrap_or(corpus.defaults.max_new_tokens);
        let timeout_seconds = case.timeout_seconds.unwrap_or(default_timeout_seconds).max(1);

        let mut args: Vec<OsString> = vec![
            "--device".into(),
            device.into(),
            "run".into(),
            "--model".into(),
            self.model.as_os_str().to_owned(),
            "--prompt".into(),
            case.question.clone().into(),
            "--max-new-tokens".into(),
            max_new_tokens.to_string().into(),
            "--temperature".into(),
            corpus.defaults.temperature.to_string().into(),
            "--prompt-template".into(),
            corpus.defaults.prompt_template.clone().into(),
            "--json-out".into(),
            case_receipt.as_os_str().to_owned(),
        ];
        if let Some(tokenizer) = &self.tokenizer {
            args.push("--tokenizer".into());
            args.push(tokenizer.as_os_str().to_owned());
        }
        if corpus.defaults.greedy {
            args.push("--greedy".into());
        }
        if corpus.defaults.deterministic {
            args.push("--deterministic".into());
        }
        if corpus.defaults.strict_loader {
            args.push("--strict-loader".into());
            args.push("--strict-tokenizer".into());
        }
        if let Some(steps) = self.dump_logit_steps {
            args.push("--dump-logit-steps".into());
            args.push(steps.to_string().into());
            args.push("--logits-topk".into());
            args.push(self.logits_topk.to_string().into());
            if corpus.defaults.greedy {
                args.push("--assert-greedy".into());
            }
        }
        let child_env = self.cpu_kernel.map(AnswerCpuKernel::child_env).unwrap_or_default();
        let run =
            run_child_with_timeout(exe, &args, &child_env, Duration::from_secs(timeout_seconds))?;
        if run.timed_out {
            return Ok(child_failure_row(ChildFailureRowInput {
                case,
                status: "timeout",
                failed_rule: "timeout",
                exe,
                args: &args,
                child_env: &child_env,
                run: &run,
                case_receipt: &case_receipt,
                device,
                timeout_seconds,
                cpu_kernel: self.cpu_kernel,
            }));
        }
        if !run.success {
            return Ok(child_failure_row(ChildFailureRowInput {
                case,
                status: "command_failed",
                failed_rule: "command_failed",
                exe,
                args: &args,
                child_env: &child_env,
                run: &run,
                case_receipt: &case_receipt,
                device,
                timeout_seconds,
                cpu_kernel: self.cpu_kernel,
            }));
        }

        let run_receipt: Value = serde_json::from_slice(
            &fs::read(&case_receipt)
                .with_context(|| format!("missing run receipt {}", case_receipt.display()))?,
        )
        .with_context(|| format!("invalid run receipt {}", case_receipt.display()))?;
        let answer = run_receipt["text"].as_str().unwrap_or_default().to_string();
        let token_ids = generated_token_ids(&run_receipt);
        let prompt_prefill = prompt_prefill_receipt(&run_receipt);
        let generated_token_count = run_receipt["tokens"]["generated"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(token_ids.len());
        let min_generated_tokens =
            case.min_generated_tokens.or(corpus.defaults.min_generated_tokens);
        let min_distinct_generated_tokens =
            case.min_distinct_generated_tokens.or(corpus.defaults.min_distinct_generated_tokens);
        let mut quality = evaluate_quality(
            &answer,
            &case.gate,
            Some(&token_ids),
            min_generated_tokens,
            min_distinct_generated_tokens,
        );
        quality.failed_rules.extend(answer_receipt_failed_rules(&run_receipt, device));
        quality.passed = quality.failed_rules.is_empty();
        let status = if quality.passed { "passed" } else { "quality_failed" };

        Ok(json!({
            "id": case.id,
            "question": case.question,
            "status": status,
            "run_receipt_path": case_receipt.display().to_string(),
            "answer": answer,
            "tokens": run_receipt.get("tokens").cloned().unwrap_or(Value::Null),
            "token_ids": {
                "prompt": run_receipt["tokens"]["prompt_ids"].clone(),
                "generated": run_receipt["tokens"]["generated_ids"].clone(),
            },
            "logits_dump": run_receipt.get("logits_dump").cloned().unwrap_or(Value::Null),
            "prompt": {
                "rendered_text": run_receipt["prompt_render"]["rendered_text"]
                    .as_str()
                    .map(Value::from)
                    .unwrap_or_else(|| run_receipt["prompt"].clone()),
                "rendered_sha256": run_receipt["prompt_render"]["rendered_sha256"].clone(),
                "template_family": corpus.defaults.prompt_template,
                "add_bos": run_receipt["prompt_render"]["add_bos"]
                    .as_bool()
                    .map(Value::from)
                    .unwrap_or_else(|| run_receipt["gen_policy"]["bos"].clone()),
                "add_special": run_receipt["prompt_render"]["parse_special"]
                    .as_bool()
                    .map(Value::from)
                    .unwrap_or_else(|| {
                        Value::from(
                            run_receipt["tokenizer"]["bos"].is_number()
                                || run_receipt["tokenizer"]["eos"].is_number(),
                        )
                    }),
            },
            "prompt_template": corpus.defaults.prompt_template,
            "prompt_prefill": prompt_prefill,
            "position": {
                "next_decode_position": run_receipt["tokens"]["prompt"].clone(),
            },
            "quality": {
                "passed": quality.passed,
                "printable_utf8": quality.printable_utf8,
                "non_empty_answer": quality.non_empty_answer,
                "no_replacement_chars": quality.no_replacement_chars,
                "no_raw_special_tokens": quality.no_raw_special_tokens,
                "mostly_text": quality.mostly_text,
                "generated_tokens": generated_token_count,
                "distinct_generated_tokens": quality.distinct_generated_tokens,
                "min_generated_tokens": min_generated_tokens,
                "min_distinct_generated_tokens": min_distinct_generated_tokens,
                "gate_kind": case.gate.kind,
                "failed_rules": quality.failed_rules,
            },
            "backend": {
                "requested_backend": run_receipt["requested_backend"].clone(),
                "selected_backend": run_receipt["selected_backend"].clone(),
                "runtime_api": run_receipt["runtime_api"].clone(),
                "fallback_used": run_receipt["fallback_used"].clone(),
            },
            "kernel": {
                "selected_kernel": run_receipt["kernel"]["kernel_id"].clone(),
                "family": run_receipt["kernel"]["family"].clone(),
            },
            "loader": {
                "mode": run_receipt["loader"]["mode"].clone(),
            },
            "tokenizer": {
                "source": run_receipt["tokenizer"]["source"].clone(),
                "strict": run_receipt["tokenizer"]["strict"].clone(),
                "model_family": run_receipt["tokenizer"]["type"].clone(),
                "pretokenizer_authority": run_receipt["tokenizer"]["pretokenizer_authority"]
                    .as_str()
                    .map(Value::from)
                    .unwrap_or_else(|| Value::from("unknown")),
            },
            "model": {
                "repo": run_receipt["model"]["repo"].clone(),
                "file": run_receipt["model"]["file"].clone(),
                "sha256": run_receipt["model"]["sha256"].clone(),
                "family": run_receipt["model"]["family"].clone(),
                "architecture": run_receipt["model"]["architecture"].clone(),
                "quant_format": run_receipt["model"]["quant_format"]
                    .as_str()
                    .map(Value::from)
                    .or_else(|| corpus.model.quant_format.as_ref().map(|value| Value::from(value.clone())))
                    .unwrap_or_else(|| run_receipt["strict_provenance"]["quant_format"].clone()),
                "vocab_size": run_receipt["model"]["vocab_size"].clone(),
                "tie_word_embeddings": run_receipt["model"]["tie_word_embeddings"].clone(),
                "output_head_tensor": run_receipt["model"]["output_head_tensor"].clone(),
            }
        }))
    }

    fn not_run_row(&self, case: &AnswerCase, reason: &str) -> Value {
        json!({
            "id": case.id,
            "question": case.question,
            "status": "not_run",
            "reason": reason,
            "quality": {
                "passed": false,
                "failed_rules": ["not_run"],
            }
        })
    }
}

#[derive(Debug, Deserialize)]
struct AnswerCorpus {
    schema: u32,
    artifact_kind: String,
    name: String,
    description: String,
    model: CorpusModel,
    defaults: CorpusDefaults,
    cases: Vec<AnswerCase>,
}

impl AnswerCorpus {
    fn load(path: &Path) -> Result<Self> {
        let corpus: Self = serde_yaml::from_slice(
            &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", path.display()))?;
        if corpus.schema != 1 {
            anyhow::bail!("unsupported answer corpus schema {}", corpus.schema);
        }
        if !matches!(corpus.artifact_kind.as_str(), "bitnet_answer_corpus" | "slm_answer_corpus") {
            anyhow::bail!("unexpected answer corpus artifact_kind {}", corpus.artifact_kind);
        }
        if corpus.cases.is_empty() {
            anyhow::bail!("answer corpus must contain at least one case");
        }
        Ok(corpus)
    }
}

#[derive(Debug, Deserialize)]
struct CorpusModel {
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

#[derive(Debug, Deserialize)]
struct CorpusDefaults {
    prompt_template: String,
    max_new_tokens: usize,
    greedy: bool,
    deterministic: bool,
    strict_loader: bool,
    temperature: f32,
    per_prompt_timeout_seconds: Option<u64>,
    min_generated_tokens: Option<usize>,
    min_distinct_generated_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct AnswerCase {
    id: String,
    question: String,
    max_new_tokens: Option<usize>,
    timeout_seconds: Option<u64>,
    min_generated_tokens: Option<usize>,
    min_distinct_generated_tokens: Option<usize>,
    gate: AnswerGate,
}

#[derive(Debug, Deserialize)]
struct AnswerGate {
    kind: String,
    expected: Option<String>,
    contains_any: Option<Vec<String>>,
    starts_with_any: Option<Vec<String>>,
    min_words: Option<usize>,
}

struct QualityResult {
    passed: bool,
    printable_utf8: bool,
    non_empty_answer: bool,
    no_replacement_chars: bool,
    no_raw_special_tokens: bool,
    mostly_text: bool,
    distinct_generated_tokens: usize,
    failed_rules: Vec<String>,
}

fn normalize_answer_corpus_device(device: &str) -> String {
    match device.trim() {
        "auto" => "cpu".to_string(),
        other => other.to_string(),
    }
}

fn answer_corpus_artifact_kind(device: &str, corpus_artifact_kind: &str) -> &'static str {
    match (device, corpus_artifact_kind) {
        ("apple-m4-cpu-neon", _) => "bitnet_apple_m4_local_answer_corpus",
        ("cuda" | RTX_5070_TI_CUDA, _) => "bitnet_cuda_answer_diagnostic_corpus",
        (_, "slm_answer_corpus") => "slm_cpu_answer_corpus",
        _ => "bitnet_cpu_answer_corpus",
    }
}

fn answer_corpus_runtime_api(device: &str) -> &'static str {
    if is_cuda_answer_corpus_device(device) { "cuda" } else { "cpu" }
}

fn is_cuda_answer_corpus_device(device: &str) -> bool {
    matches!(device, "cuda" | RTX_5070_TI_CUDA)
}

fn prompt_prefill_receipt(run_receipt: &Value) -> Value {
    let prompt_token_count = run_receipt["tokens"]["prompt"].as_u64().unwrap_or_else(|| {
        run_receipt["tokens"]["prompt_ids"]
            .as_array()
            .map(|tokens| tokens.len() as u64)
            .unwrap_or_default()
    });
    let profile_prefill = &run_receipt["profile"]["prompt_prefill"];
    let exercised = profile_prefill["exercised"].as_bool().unwrap_or(prompt_token_count > 0);
    json!({
        "executed": exercised,
        "exercised": exercised,
        "prompt_token_count": prompt_token_count,
        "decode_start_position": prompt_token_count,
        "kv_cache_behavior": profile_prefill["kv_cache_behavior"].clone(),
        "source": if profile_prefill.is_object() {
            "run_receipt_profile"
        } else {
            "tokens_prompt_count"
        },
    })
}

fn effective_default_timeout_seconds(cli: Option<u64>, corpus: Option<u64>) -> u64 {
    cli.or(corpus).unwrap_or(300).max(1)
}

fn evaluate_quality(
    answer: &str,
    gate: &AnswerGate,
    generated_token_ids: Option<&[u32]>,
    min_generated_tokens: Option<usize>,
    min_distinct_generated_tokens: Option<usize>,
) -> QualityResult {
    let normalized = strip_special_markers(answer).trim().to_string();
    let non_empty_answer = !normalized.is_empty();
    let no_replacement_chars = !normalized.contains('\u{FFFD}');
    let no_raw_special_tokens = !contains_raw_special_token(&normalized);
    let mostly_text = mostly_text(&normalized);
    let printable_utf8 = normalized.chars().all(|ch| ch == '\n' || ch == '\t' || !ch.is_control());
    let generated_token_count = generated_token_ids.map(|tokens| tokens.len()).unwrap_or(0);
    let distinct_generated_tokens = generated_token_ids
        .map(|tokens| tokens.iter().copied().collect::<std::collections::BTreeSet<_>>().len())
        .unwrap_or(0);

    let mut failed_rules = Vec::new();
    if !non_empty_answer {
        failed_rules.push("empty_answer".to_string());
    }
    if !no_replacement_chars {
        failed_rules.push("replacement_chars".to_string());
    }
    if !no_raw_special_tokens {
        failed_rules.push("raw_special_tokens".to_string());
    }
    if !mostly_text {
        failed_rules.push("mostly_text".to_string());
    }
    if !printable_utf8 {
        failed_rules.push("printable_utf8".to_string());
    }

    if !gate_passed(&normalized, gate) {
        failed_rules.push(format!("gate_{}", gate.kind));
    }
    if let Some(minimum) = min_generated_tokens
        && generated_token_count < minimum
    {
        failed_rules.push("generated_token_min".to_string());
    }
    if let Some(minimum) = min_distinct_generated_tokens
        && distinct_generated_tokens < minimum
    {
        failed_rules.push("generated_token_variation".to_string());
    }

    QualityResult {
        passed: failed_rules.is_empty(),
        printable_utf8,
        non_empty_answer,
        no_replacement_chars,
        no_raw_special_tokens,
        mostly_text,
        distinct_generated_tokens,
        failed_rules,
    }
}

fn answer_receipt_failed_rules(run_receipt: &Value, expected_backend: &str) -> Vec<String> {
    let mut failed = Vec::new();
    let requested_backend = run_receipt["requested_backend"].as_str().unwrap_or_default();
    let selected_backend = run_receipt["selected_backend"].as_str().unwrap_or_default();
    let runtime_api = run_receipt["runtime_api"].as_str().unwrap_or_default();
    let fallback_used = run_receipt["fallback_used"].as_bool().unwrap_or(true);
    if requested_backend != expected_backend {
        failed.push(format!("requested_backend_{expected_backend}"));
    }
    let selected_backend_valid = match expected_backend {
        "cpu" => matches!(selected_backend, "cpu" | "cpu-rust"),
        "apple-m4-cpu-neon" => selected_backend == "apple-m4-cpu-neon",
        "cuda" => selected_backend.contains("cuda"),
        RTX_5070_TI_CUDA => selected_backend == RTX_5070_TI_CUDA,
        _ => false,
    };
    if !selected_backend_valid {
        failed.push(format!("selected_backend_{expected_backend}"));
    }
    let expected_runtime_api = answer_corpus_runtime_api(expected_backend);
    if runtime_api != expected_runtime_api {
        failed.push(format!("runtime_api_{expected_runtime_api}"));
    }
    if fallback_used {
        failed.push("fallback_false".to_string());
    }

    let loader_mode = run_receipt["loader"]["mode"]
        .as_str()
        .or_else(|| run_receipt["model"]["loader_mode"].as_str())
        .unwrap_or_default();
    if loader_mode != "real_gguf" {
        failed.push("loader_real_gguf".to_string());
    }

    let tokenizer_source = run_receipt["tokenizer"]["source"].as_str().unwrap_or_default();
    let tokenizer_strict = run_receipt["tokenizer"]["strict"].as_bool().unwrap_or(false);
    if tokenizer_source.is_empty() || tokenizer_source == "unknown" {
        failed.push("tokenizer_source_recorded".to_string());
    }
    if !tokenizer_strict {
        failed.push("tokenizer_strict".to_string());
    }

    let selected_kernel = run_receipt["kernel"]["kernel_id"].as_str().unwrap_or_default();
    if selected_kernel.is_empty() {
        failed.push("selected_kernel_recorded".to_string());
    }
    if selected_kernel.contains("mock") || selected_kernel.contains("diagnostic") {
        failed.push("selected_kernel_production".to_string());
    }
    if is_cuda_answer_corpus_device(expected_backend) {
        let cuda_kernel_recorded = selected_kernel.contains("cuda")
            || run_receipt["kernel_stats"].as_array().is_some_and(|stats| {
                stats.iter().any(|stat| {
                    stat["kernel_id"].as_str().is_some_and(|id| id.contains("cuda"))
                        && stat["invocations"].as_u64().unwrap_or_default() > 0
                })
            });
        if !cuda_kernel_recorded {
            failed.push("cuda_kernel_recorded".to_string());
        }
        let cpu_fallback = run_receipt["execution_coverage"]["bitnet_linear_layers_cpu_fallback"]
            .as_u64()
            .unwrap_or(1);
        if cpu_fallback != 0 {
            failed.push("cuda_bitnet_linear_cpu_fallback_zero".to_string());
        }
    }

    if !run_receipt["tokens"]["prompt_ids"].is_array() {
        failed.push("prompt_token_ids_recorded".to_string());
    }
    if generated_token_ids(run_receipt).is_empty() {
        failed.push("generated_token_ids_recorded".to_string());
    }
    failed
}

fn generated_token_ids(receipt: &Value) -> Vec<u32> {
    receipt["tokens"]["generated_ids"]
        .as_array()
        .or_else(|| receipt["tokens"]["ids"].as_array())
        .map(|ids| {
            ids.iter()
                .filter_map(|value| value.as_u64().and_then(|id| u32::try_from(id).ok()))
                .collect()
        })
        .unwrap_or_default()
}

fn gate_passed(answer: &str, gate: &AnswerGate) -> bool {
    match gate.kind.as_str() {
        "exact_trimmed" => {
            let Some(expected) = &gate.expected else {
                return false;
            };
            answer.trim().eq_ignore_ascii_case(expected.trim())
        }
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
        "readable" => word_count(answer) >= gate.min_words.unwrap_or(1),
        _ => false,
    }
}

fn strip_special_markers(answer: &str) -> String {
    answer.replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "").replace("<|eot_id|>", "")
}

fn contains_raw_special_token(answer: &str) -> bool {
    answer.contains("<|") || answer.contains("|>")
}

fn mostly_text(answer: &str) -> bool {
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

fn word_count(answer: &str) -> usize {
    answer.split_whitespace().filter(|word| word.chars().any(char::is_alphanumeric)).count()
}

struct ChildRun {
    success: bool,
    timed_out: bool,
    exit_code: Option<i32>,
    stdout_path: PathBuf,
    stderr_path: PathBuf,
    phase_path: PathBuf,
    stdout: String,
    stderr: String,
    child_phases: Vec<Value>,
    last_observed_phase: Option<String>,
}

struct ChildFailureRowInput<'a> {
    case: &'a AnswerCase,
    status: &'static str,
    failed_rule: &'static str,
    exe: &'a Path,
    args: &'a [OsString],
    child_env: &'a [(&'static str, &'static str)],
    run: &'a ChildRun,
    case_receipt: &'a Path,
    device: &'a str,
    timeout_seconds: u64,
    cpu_kernel: Option<AnswerCpuKernel>,
}

fn child_failure_row(input: ChildFailureRowInput<'_>) -> Value {
    json!({
        "id": input.case.id,
        "question": input.case.question,
        "status": input.status,
        "exit_code": input.run.exit_code,
        "timeout_seconds": input.timeout_seconds,
        "run_receipt_path": input.case_receipt.display().to_string(),
        "quality": {
            "passed": false,
            "failed_rules": [input.failed_rule],
        },
        "backend": {
            "requested_backend": input.device,
            "selected_backend": input.device,
            "runtime_api": answer_corpus_runtime_api(input.device),
            "fallback_used": false,
            "source": "answer_corpus_launcher",
        },
        "kernel": {
            "requested_cpu_kernel": input.cpu_kernel.map(AnswerCpuKernel::as_str),
            "selected_kernel": Value::Null,
            "family": Value::Null,
            "source": "missing_child_receipt",
        },
        "child_invocation": {
            "executable": input.exe.display().to_string(),
            "args": os_args_json(input.args),
            "environment_overrides": child_environment_json(input.child_env),
            "timeout_seconds": input.timeout_seconds,
            "expected_receipt_path": input.case_receipt.display().to_string(),
            "phase_path": input.run.phase_path.display().to_string(),
        },
        "child_process": {
            "success": input.run.success,
            "timed_out": input.run.timed_out,
            "exit_code": input.run.exit_code,
            "exit_code_hex": input.run.exit_code.map(exit_code_hex),
            "crash_class": classify_child_exit(input.run),
            "receipt_observed": input.case_receipt.exists(),
            "last_observed_phase": input.run.last_observed_phase,
            "phase_events": input.run.child_phases,
            "stdout_path": input.run.stdout_path.display().to_string(),
            "stderr_path": input.run.stderr_path.display().to_string(),
            "phase_path": input.run.phase_path.display().to_string(),
        },
        "stdout_tail": tail_string(&input.run.stdout, 4096),
        "stderr_tail": tail_string(&input.run.stderr, 4096),
    })
}

fn os_args_json(args: &[OsString]) -> Value {
    Value::Array(args.iter().map(|arg| Value::String(arg.to_string_lossy().into_owned())).collect())
}

fn child_environment_json(child_env: &[(&'static str, &'static str)]) -> Value {
    let mut env = serde_json::Map::new();
    for (key, value) in child_env {
        env.insert((*key).to_string(), Value::String((*value).to_string()));
    }
    env.insert("RUST_LOG".to_string(), Value::String(child_rust_log_value()));
    Value::Object(env)
}

fn classify_child_exit(run: &ChildRun) -> &'static str {
    if run.timed_out {
        return "timeout";
    }
    match run.exit_code {
        None => "terminated_without_exit_code",
        Some(0) if run.success => "success",
        Some(code) if is_windows_native_status(code) => classify_windows_status(code),
        Some(_) => "nonzero_exit",
    }
}

fn is_windows_native_status(code: i32) -> bool {
    (code as u32) & 0xC000_0000 == 0xC000_0000
}

fn classify_windows_status(code: i32) -> &'static str {
    match code as u32 {
        0xC000_0005 => "windows_access_violation",
        0xC000_001D => "windows_illegal_instruction",
        0xC000_00FD => "windows_stack_overflow",
        0xC000_0374 => "windows_heap_corruption",
        0xC000_0409 => "windows_stack_buffer_overrun_or_fast_fail",
        _ => "windows_native_status",
    }
}

fn exit_code_hex(code: i32) -> String {
    format!("0x{:08X}", code as u32)
}

fn child_rust_log_value() -> String {
    std::env::var("BITNET_ANSWER_CORPUS_CHILD_RUST_LOG").unwrap_or_else(|_| "warn".into())
}

fn run_child_with_timeout(
    exe: &Path,
    args: &[OsString],
    envs: &[(&'static str, &'static str)],
    timeout: Duration,
) -> Result<ChildRun> {
    let child_rust_log = child_rust_log_value();
    let stdout_path = child_capture_path("stdout");
    let stderr_path = child_capture_path("stderr");
    let phase_path = child_capture_path("phases");
    let stdout_file = File::create(&stdout_path)
        .with_context(|| format!("failed to create {}", stdout_path.display()))?;
    let stderr_file = File::create(&stderr_path)
        .with_context(|| format!("failed to create {}", stderr_path.display()))?;
    let mut child = Command::new(exe)
        .args(args)
        .envs(envs.iter().copied())
        .env("RUST_LOG", child_rust_log)
        .env("BITNET_ANSWER_CORPUS_CHILD_PHASE_PATH", &phase_path)
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .spawn()
        .with_context(|| format!("failed to spawn {}", exe.display()))?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            let stdout = read_child_capture(&stdout_path);
            let stderr = read_child_capture(&stderr_path);
            let child_phases = read_child_phase_events(&phase_path);
            let last_observed_phase = last_observed_child_phase(&child_phases, &stderr);
            if status.success() {
                remove_child_capture(&stdout_path);
                remove_child_capture(&stderr_path);
                remove_child_capture(&phase_path);
            }
            return Ok(ChildRun {
                success: status.success(),
                timed_out: false,
                exit_code: status.code(),
                stdout_path,
                stderr_path,
                phase_path,
                stdout,
                stderr,
                child_phases,
                last_observed_phase,
            });
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let status = child.wait()?;
            let stdout = read_child_capture(&stdout_path);
            let stderr = read_child_capture(&stderr_path);
            let child_phases = read_child_phase_events(&phase_path);
            let last_observed_phase = last_observed_child_phase(&child_phases, &stderr);
            return Ok(ChildRun {
                success: false,
                timed_out: true,
                exit_code: status.code(),
                stdout_path,
                stderr_path,
                phase_path,
                stdout,
                stderr,
                child_phases,
                last_observed_phase,
            });
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn read_child_phase_events(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .map(|contents| {
            contents.lines().filter_map(|line| serde_json::from_str::<Value>(line).ok()).collect()
        })
        .unwrap_or_default()
}

fn last_observed_child_phase(events: &[Value], stderr: &str) -> Option<String> {
    events
        .iter()
        .rev()
        .find_map(|event| {
            event["child_phase"].as_str().or_else(|| event["phase"].as_str()).map(str::to_string)
        })
        .or_else(|| {
            stderr.lines().rev().find_map(|line| {
                line.split_once("answer_corpus_child_phase=")
                    .map(|(_, phase)| phase.trim().to_string())
                    .filter(|phase| !phase.is_empty())
            })
        })
}

fn cpu_avx2_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx2")
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

fn cpu_avx512_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx512f")
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

fn child_capture_path(kind: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir()
        .join(format!("bitnet-answer-corpus-{}-{nanos}-{sequence}-{kind}.log", std::process::id()))
}

fn read_child_capture(path: &Path) -> String {
    fs::read(path).map(|bytes| String::from_utf8_lossy(&bytes).into_owned()).unwrap_or_default()
}

fn remove_child_capture(path: &Path) {
    let _ = fs::remove_file(path);
}

fn tail_string(value: &str, max_chars: usize) -> String {
    let len = value.chars().count();
    if len <= max_chars { value.to_string() } else { value.chars().skip(len - max_chars).collect() }
}

fn sanitize_file_stem(id: &str) -> String {
    id.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' { ch } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate(kind: &str) -> AnswerGate {
        AnswerGate {
            kind: kind.to_string(),
            expected: None,
            contains_any: None,
            starts_with_any: None,
            min_words: None,
        }
    }

    #[test]
    fn exact_gate_accepts_trimmed_answer() {
        let gate = AnswerGate { expected: Some("4".to_string()), ..gate("exact_trimmed") };
        let quality = evaluate_quality(" 4\n", &gate, None, None, None);
        assert!(quality.passed);
    }

    #[test]
    fn quality_rejects_raw_special_tokens() {
        let quality =
            evaluate_quality("<|start_header_id|>assistant", &gate("readable"), None, None, None);
        assert!(!quality.passed);
        assert!(quality.failed_rules.contains(&"raw_special_tokens".to_string()));
    }

    #[test]
    fn quality_rejects_punctuation_noise() {
        let quality = evaluate_quality("!!!,,,!!!", &gate("readable"), None, None, None);
        assert!(!quality.passed);
        assert!(quality.failed_rules.contains(&"mostly_text".to_string()));
    }

    #[test]
    fn apple_m4_quality_rejects_short_or_degenerate_token_output() {
        let gate = AnswerGate { min_words: Some(2), ..gate("readable") };
        let quality = evaluate_quality("short answer", &gate, Some(&[7, 7, 7]), Some(4), Some(2));
        assert!(!quality.passed);
        assert!(quality.failed_rules.contains(&"generated_token_min".to_string()));
        assert!(quality.failed_rules.contains(&"generated_token_variation".to_string()));
        assert_eq!(quality.distinct_generated_tokens, 1);
    }

    #[test]
    fn cli_timeout_overrides_corpus_default() {
        assert_eq!(effective_default_timeout_seconds(Some(1), Some(300)), 1);
        assert_eq!(effective_default_timeout_seconds(None, Some(120)), 120);
        assert_eq!(effective_default_timeout_seconds(None, None), 300);
        assert_eq!(effective_default_timeout_seconds(Some(0), Some(300)), 1);
    }

    #[test]
    fn cpu_answer_receipt_accepts_strict_cpu_truth() {
        let receipt = json!({
            "requested_backend": "cpu",
            "selected_backend": "cpu-rust",
            "runtime_api": "cpu",
            "fallback_used": false,
            "loader": { "mode": "real_gguf" },
            "tokenizer": { "source": "gguf_metadata", "strict": true },
            "kernel": { "kernel_id": "i2_s-avx2-reference" },
            "tokens": {
                "prompt_ids": [1, 2, 3],
                "generated_ids": [4]
            }
        });

        assert!(answer_receipt_failed_rules(&receipt, "cpu").is_empty());
    }

    #[test]
    fn avx512_cpu_kernel_selector_sets_child_env() {
        assert_eq!(AnswerCpuKernel::Avx512.as_str(), "avx512");
        assert_eq!(
            AnswerCpuKernel::Avx512.child_env(),
            vec![("BITNET_CPU_KERNEL", "avx512"), ("BITNET_FORCE_SCALAR", "0")]
        );
    }

    #[test]
    fn slm_corpus_uses_slm_cpu_artifact_kind() {
        assert_eq!(
            answer_corpus_artifact_kind("cpu", "slm_answer_corpus"),
            "slm_cpu_answer_corpus"
        );
        assert_eq!(
            answer_corpus_artifact_kind("cpu", "bitnet_answer_corpus"),
            "bitnet_cpu_answer_corpus"
        );
    }

    #[test]
    fn prompt_prefill_receipt_prefers_profile_data() {
        let receipt = json!({
            "tokens": {
                "prompt": 7,
                "prompt_ids": [1, 2, 3, 4, 5, 6, 7]
            },
            "profile": {
                "prompt_prefill": {
                    "exercised": true,
                    "kv_cache_behavior": "prompt_prefix_prefilled_before_decode"
                }
            }
        });

        let prefill = prompt_prefill_receipt(&receipt);
        assert_eq!(prefill["executed"], true);
        assert_eq!(prefill["prompt_token_count"], 7);
        assert_eq!(prefill["decode_start_position"], 7);
        assert_eq!(prefill["source"], "run_receipt_profile");
    }

    #[test]
    fn child_failure_row_records_cuda_crash_diagnostics() {
        let case = AnswerCase {
            id: "math_2_plus_2".to_string(),
            question: "What is 2+2? Answer with only the number.".to_string(),
            max_new_tokens: Some(4),
            timeout_seconds: None,
            min_generated_tokens: None,
            min_distinct_generated_tokens: None,
            gate: gate("exact_trimmed"),
        };
        let run = ChildRun {
            success: false,
            timed_out: false,
            exit_code: Some(-1_073_740_791),
            stdout_path: PathBuf::from("target/bitnet/receipts/math.stdout.log"),
            stderr_path: PathBuf::from("target/bitnet/receipts/math.stderr.log"),
            phase_path: PathBuf::from("target/bitnet/receipts/math.phases.jsonl"),
            stdout: "selected_backend=nvidia-rtx-5070-ti-cuda".to_string(),
            stderr:
                "answer_corpus_child_phase=prompt_prefill_start\nchild terminated before receipt"
                    .to_string(),
            child_phases: vec![json!({
                "child_phase": "backend_select_complete",
                "details": {
                    "selected_backend": RTX_5070_TI_CUDA,
                    "runtime_api": "cuda"
                }
            })],
            last_observed_phase: Some("backend_select_complete".to_string()),
        };
        let args = vec![
            "--device".into(),
            RTX_5070_TI_CUDA.into(),
            "run".into(),
            "--json-out".into(),
            "target/bitnet/receipts/math.json".into(),
        ];
        let row = child_failure_row(ChildFailureRowInput {
            case: &case,
            status: "command_failed",
            failed_rule: "command_failed",
            exe: Path::new("bitnet.exe"),
            args: &args,
            child_env: &[],
            run: &run,
            case_receipt: Path::new("target/bitnet/receipts/math.json"),
            device: RTX_5070_TI_CUDA,
            timeout_seconds: 120,
            cpu_kernel: None,
        });

        assert_eq!(row["status"], "command_failed");
        assert_eq!(row["backend"]["runtime_api"], "cuda");
        assert_eq!(row["child_process"]["exit_code_hex"], "0xC0000409");
        assert_eq!(
            row["child_process"]["crash_class"],
            "windows_stack_buffer_overrun_or_fast_fail"
        );
        assert_eq!(row["child_process"]["receipt_observed"], false);
        assert_eq!(row["child_process"]["last_observed_phase"], "backend_select_complete");
        assert_eq!(
            row["child_process"]["phase_events"][0]["child_phase"],
            "backend_select_complete"
        );
        assert_eq!(row["child_process"]["stdout_path"], "target/bitnet/receipts/math.stdout.log");
        assert_eq!(
            row["child_invocation"]["phase_path"],
            "target/bitnet/receipts/math.phases.jsonl"
        );
        assert_eq!(row["child_invocation"]["timeout_seconds"], 120);
        assert_eq!(row["quality"]["failed_rules"], json!(["command_failed"]));
    }

    #[test]
    fn child_failure_row_records_requested_cpu_kernel_env() {
        let case = AnswerCase {
            id: "math_2_plus_2".to_string(),
            question: "What is 2+2? Answer with only the number.".to_string(),
            max_new_tokens: Some(4),
            timeout_seconds: None,
            min_generated_tokens: None,
            min_distinct_generated_tokens: None,
            gate: gate("exact_trimmed"),
        };
        let run = ChildRun {
            success: false,
            timed_out: true,
            exit_code: None,
            stdout_path: PathBuf::from("target/bitnet/receipts/math.stdout.log"),
            stderr_path: PathBuf::from("target/bitnet/receipts/math.stderr.log"),
            phase_path: PathBuf::from("target/bitnet/receipts/math.phases.jsonl"),
            stdout: String::new(),
            stderr: "timeout".to_string(),
            child_phases: Vec::new(),
            last_observed_phase: None,
        };
        let env = AnswerCpuKernel::Avx512.child_env();
        let args = vec!["--device".into(), "cpu".into(), "run".into()];
        let row = child_failure_row(ChildFailureRowInput {
            case: &case,
            status: "timeout",
            failed_rule: "timeout",
            exe: Path::new("bitnet.exe"),
            args: &args,
            child_env: &env,
            run: &run,
            case_receipt: Path::new("target/bitnet/receipts/math.json"),
            device: "cpu",
            timeout_seconds: 1,
            cpu_kernel: Some(AnswerCpuKernel::Avx512),
        });

        assert_eq!(row["status"], "timeout");
        assert_eq!(row["child_process"]["crash_class"], "timeout");
        assert_eq!(row["kernel"]["requested_cpu_kernel"], "avx512");
        assert_eq!(row["child_invocation"]["environment_overrides"]["BITNET_CPU_KERNEL"], "avx512");
        assert_eq!(row["child_invocation"]["environment_overrides"]["BITNET_FORCE_SCALAR"], "0");
    }

    #[test]
    fn last_child_phase_prefers_phase_jsonl_over_stderr_tail() {
        let events = vec![
            json!({ "child_phase": "model_load_start" }),
            json!({ "child_phase": "tokenizer_load_complete" }),
        ];
        let stderr = "answer_corpus_child_phase=prompt_render_start";

        assert_eq!(
            last_observed_child_phase(&events, stderr),
            Some("tokenizer_load_complete".to_string())
        );
    }

    #[test]
    fn last_child_phase_falls_back_to_stderr_marker() {
        let stderr = "line one\nanswer_corpus_child_phase=decode_step_0_start\n";

        assert_eq!(last_observed_child_phase(&[], stderr), Some("decode_step_0_start".to_string()));
    }

    #[test]
    fn answer_receipt_accepts_strict_apple_m4_cpu_neon_truth() {
        let receipt = json!({
            "requested_backend": "apple-m4-cpu-neon",
            "selected_backend": "apple-m4-cpu-neon",
            "runtime_api": "cpu",
            "fallback_used": false,
            "loader": { "mode": "real_gguf" },
            "tokenizer": { "source": "gguf_metadata", "strict": true },
            "kernel": { "kernel_id": "i2_s-scalar-reference" },
            "tokens": {
                "prompt_ids": [1, 2, 3],
                "ids": [4]
            }
        });

        assert!(answer_receipt_failed_rules(&receipt, "apple-m4-cpu-neon").is_empty());
    }

    #[test]
    fn answer_receipt_accepts_strict_cuda_truth() {
        let receipt = json!({
            "requested_backend": RTX_5070_TI_CUDA,
            "selected_backend": RTX_5070_TI_CUDA,
            "runtime_api": "cuda",
            "fallback_used": false,
            "loader": { "mode": "real_gguf" },
            "tokenizer": { "source": "gguf_metadata", "strict": true },
            "kernel": { "kernel_id": "qk256_gemv_cuda" },
            "kernel_stats": [{ "kernel_id": "qk256_gemv_cuda", "invocations": 8 }],
            "execution_coverage": { "bitnet_linear_layers_cpu_fallback": 0 },
            "tokens": { "prompt_ids": [1, 2], "generated_ids": [3] },
        });

        assert!(answer_receipt_failed_rules(&receipt, RTX_5070_TI_CUDA).is_empty());
    }

    #[test]
    fn cpu_answer_receipt_rejects_hidden_fallback_and_missing_ids() {
        let receipt = json!({
            "requested_backend": "cpu",
            "selected_backend": "cpu-rust",
            "runtime_api": "cpu",
            "fallback_used": true,
            "loader": { "mode": "minimal_compatibility" },
            "tokenizer": { "source": "unknown", "strict": false },
            "kernel": { "kernel_id": "mock-diagnostic" },
            "tokens": {}
        });

        let failed = answer_receipt_failed_rules(&receipt, "cpu");
        assert!(failed.contains(&"fallback_false".to_string()));
        assert!(failed.contains(&"loader_real_gguf".to_string()));
        assert!(failed.contains(&"tokenizer_source_recorded".to_string()));
        assert!(failed.contains(&"tokenizer_strict".to_string()));
        assert!(failed.contains(&"selected_kernel_production".to_string()));
        assert!(failed.contains(&"prompt_token_ids_recorded".to_string()));
        assert!(failed.contains(&"generated_token_ids_recorded".to_string()));
    }
}
