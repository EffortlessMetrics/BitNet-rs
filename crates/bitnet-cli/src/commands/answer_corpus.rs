//! Answer corpus runner for CPU-first and Apple M4 local-answer baselines.

use crate::{model_cache, planner_receipts};
use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const RTX_5070_TI_CUDA: &str = "nvidia-rtx-5070-ti-cuda";
const INTEL_A770_OPENCL: &str = "intel-a770-opencl";
const INTEL_ARC_A770_OPENCL: &str = "intel-arc-a770-opencl";
const A770_BITNET_MODEL_CONTRACT: &str = "docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml";
const A770_BITNET_QK256_ROUTE_ID: &str = "a770.bitnet.i2s.qk256";
const ANSWER_RECEIPT_REQUIRED_CASE_FIELDS: &[&str] = &[
    "text",
    "tokens.prompt",
    "tokens.generated",
    "tokens.total",
    "tokens.prompt_ids",
    "tokens.generated_ids",
    "model.repo",
    "model.file",
    "model.sha256",
    "model.family",
    "model.architecture",
    "tokenizer.source",
    "tokenizer.strict",
    "tokenizer.pretokenizer_authority",
    "requested_backend",
    "selected_backend",
    "runtime_api",
    "fallback_used",
    "loader.mode",
    "kernel.kernel_id",
    "timing.model_load_ms",
    "timing.tokenizer_load_ms",
    "timing.tokenize_ms",
    "timing.prefill_ms",
    "timing.first_token_ms",
    "timing.decode_total_ms",
    "latency.total_ms",
];
const ANSWER_RECEIPT_CHECKED_RULES: &[&str] = &[
    "generated_text_recorded",
    "prompt_token_count_recorded",
    "generated_token_count_recorded",
    "total_token_count_recorded",
    "prompt_token_ids_recorded",
    "generated_token_ids_recorded",
    "prompt_token_count_matches_ids",
    "generated_token_count_matches_ids",
    "total_token_count_matches",
    "model_repo_recorded",
    "model_file_recorded",
    "model_sha256_recorded",
    "model_family_recorded",
    "model_architecture_recorded",
    "tokenizer_source_recorded",
    "tokenizer_strict",
    "tokenizer_pretokenizer_authority_recorded",
    "requested_backend",
    "selected_backend",
    "runtime_api",
    "fallback_false",
    "speedup_claim_false",
    "loader_real_gguf",
    "selected_kernel_recorded",
    "selected_kernel_production",
    "timing_model_load_ms_recorded",
    "timing_tokenizer_load_ms_recorded",
    "timing_tokenize_ms_recorded",
    "timing_prefill_ms_recorded",
    "timing_first_token_ms_recorded",
    "timing_decode_total_ms_recorded",
    "latency_total_ms_recorded",
    "qk256_hot_path_recorded",
    "qk256_hot_path_invocations_positive",
    "qk256_hot_path_materialization_audited",
    "a770_model_contract_declared",
    "a770_model_contract_path",
    "a770_kernel_route_declared",
    "a770_kernel_route_id",
    "a770_kernel_route_diagnostic_only",
    "a770_kernel_route_claimable_false",
    "a770_backend_claimable_false",
];
const ANSWER_CORPUS_SCORING_KINDS: &[&str] = &[
    "exact_match",
    "normalized_match",
    "contains_expected",
    "json_schema",
    "numeric_tolerance",
    "required_keywords",
    "forbidden_tokens",
    "required_forbidden_tokens",
];
const ANSWER_CORPUS_FAILURE_CATEGORIES: &[&str] =
    &["formatting", "factual_table", "extraction", "refusal", "timeout", "schema", "normalization"];

/// Run the fixed answer corpus through the existing `bitnet run` surface.
#[derive(Args, Debug)]
pub struct AnswerCorpusCommand {
    /// Path to the answer corpus YAML.
    #[arg(long, default_value = "ci/quality/bitnet-answer-corpus.yaml", value_name = "PATH")]
    pub corpus: PathBuf,

    /// Official BitNet GGUF model path.
    #[arg(long, value_name = "PATH")]
    pub model: PathBuf,

    /// Supported dense Apple M4 model ID used to stamp SLM aggregate receipt identity.
    #[arg(long, value_name = "MODEL_ID")]
    pub model_id: Option<String>,

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

    /// Run only matching corpus case IDs. Repeat to run multiple cases.
    #[arg(long = "case-id", value_name = "ID")]
    pub case_ids: Vec<String>,
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
        if !matches!(
            device.as_str(),
            "cpu" | "apple-m4-cpu-neon" | "cuda" | RTX_5070_TI_CUDA | INTEL_A770_OPENCL
        ) {
            anyhow::bail!(
                "answer-corpus only accepts --device cpu, --device apple-m4-cpu-neon, --device cuda, --device {RTX_5070_TI_CUDA}, or --device {INTEL_A770_OPENCL}; got {device}"
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
        let model_identity = AnswerCorpusModelIdentity::resolve(self.model_id.as_deref(), &corpus)?;
        let default_timeout_seconds = effective_default_timeout_seconds(
            self.per_prompt_timeout_seconds,
            corpus.defaults.per_prompt_timeout_seconds,
        );
        if !self.dry_run {
            validate_answer_corpus_inputs(self, &corpus)?;
        }

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

        let selected_cases = self.selected_cases(&corpus)?;
        let selected_case_ids: Vec<String> =
            selected_cases.iter().map(|case| case.id.clone()).collect();

        let exe = std::env::current_exe().context("failed to resolve current bitnet executable")?;
        let mut rows = Vec::with_capacity(selected_cases.len());
        for case in selected_cases {
            let row = if self.dry_run {
                self.not_run_row(case, "dry_run_requested")
            } else {
                self.run_case(&exe, &receipt_dir, &corpus, case, &device, default_timeout_seconds)?
            };
            rows.push(row);
        }
        ensure_rows_match_model_identity(&rows, &model_identity)?;

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
        let aggregate_execution_plan = aggregate_execution_plan(&rows, &device);
        let slm_answer_path = corpus.artifact_kind == "slm_answer_corpus";
        let bounded_slm_answer_smoke_passed =
            slm_answer_path && passed == total && failed == 0 && timed_out == 0 && not_run == 0;
        let dense_slm_clean_provenance = slm_answer_path
            && rows.iter().all(|row| {
                row["kernel"]["selected_kernel"] == "dense-qwen-cpu-reference"
                    && row["kernel"]["family"] == "dense_qwen"
            });
        let top_level_selected_backend =
            aggregate_case_str(&rows, &["backend", "selected_backend"])
                .unwrap_or(device.as_str())
                .to_string();
        let top_level_runtime_api = aggregate_case_str(&rows, &["backend", "runtime_api"])
            .unwrap_or_else(|| answer_corpus_runtime_api(&device))
            .to_string();
        let top_level_fallback_used =
            rows.iter().any(|row| row["backend"]["fallback_used"].as_bool().unwrap_or(false));
        let model_id_pinned = model_identity.id.is_some();
        let top_level_model_family = if model_id_pinned {
            model_identity.family.clone().unwrap_or_else(|| "unknown".to_string())
        } else {
            aggregate_case_str(&rows, &["model", "family"])
                .map(str::to_string)
                .or_else(|| model_identity.family.clone())
                .unwrap_or_else(|| "unknown".to_string())
        };
        let top_level_model_architecture = if model_id_pinned {
            model_identity.architecture.clone().unwrap_or_else(|| "unknown".to_string())
        } else {
            aggregate_case_str(&rows, &["model", "architecture"])
                .map(str::to_string)
                .or_else(|| model_identity.architecture.clone())
                .unwrap_or_else(|| "unknown".to_string())
        };
        let top_level_quantization = if model_id_pinned {
            model_identity.quant_format.clone().unwrap_or_else(|| "unknown".to_string())
        } else {
            aggregate_case_str(&rows, &["model", "quant_format"])
                .map(str::to_string)
                .or_else(|| model_identity.quant_format.clone())
                .unwrap_or_else(|| "unknown".to_string())
        };
        let top_level_model_repo = if model_id_pinned {
            model_identity.repo.clone()
        } else {
            aggregate_case_str(&rows, &["model", "repo"])
                .map(str::to_string)
                .unwrap_or_else(|| model_identity.repo.clone())
        };
        let top_level_model_file = if model_id_pinned {
            model_identity.file.clone()
        } else {
            aggregate_case_str(&rows, &["model", "file"])
                .map(str::to_string)
                .unwrap_or_else(|| model_identity.file.clone())
        };
        let top_level_model_sha256 = if model_id_pinned {
            model_identity.sha256.clone()
        } else {
            aggregate_case_str(&rows, &["model", "sha256"])
                .map(str::to_string)
                .or_else(|| model_identity.sha256.clone())
        };
        let top_level_tokenizer_source = aggregate_case_str(&rows, &["tokenizer", "source"])
            .unwrap_or(aggregate_tokenizer)
            .to_string();
        let top_level_selected_kernel_or_runtime =
            aggregate_case_str(&rows, &["kernel", "selected_kernel"])
                .unwrap_or(&top_level_runtime_api)
                .to_string();
        let prompt_generation_identity =
            aggregate_case_value(&rows, &["prompt_generation_identity"]).unwrap_or(Value::Null);
        let proof_route_contract = answer_corpus_proof_route_contract(&device);
        let top_level_backend_lane =
            answer_corpus_backend_lane(&device, slm_answer_path, &top_level_model_family);
        let answer_ready_artifact_available = corpus_answer_ready_artifact_available(&corpus.model);
        let backend_quality_gate_passed =
            total > 0 && passed == total && failed == 0 && timed_out == 0 && not_run == 0;
        let bitnet_answer_path = corpus.artifact_kind == "bitnet_answer_corpus";
        let a770_opencl_answer_corpus = is_a770_opencl_answer_corpus_device(&device);
        let coherent_answer_claimed = bitnet_answer_path
            && answer_ready_artifact_available
            && backend_quality_gate_passed
            && !a770_opencl_answer_corpus;

        let receipt = json!({
            "schema_version": "1.0.0",
            "artifact_kind": artifact_kind,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "requested_backend": device.as_str(),
            "selected_backend": top_level_selected_backend,
            "runtime_api": top_level_runtime_api,
            "fallback_used": top_level_fallback_used,
            "backend_lane": top_level_backend_lane,
            "model_family": top_level_model_family.clone(),
            "model_architecture": top_level_model_architecture.clone(),
            "quantization": top_level_quantization.clone(),
            "tokenizer_source": top_level_tokenizer_source,
            "prompt_template": corpus.defaults.prompt_template.as_str(),
            "prompt_generation_identity": &prompt_generation_identity,
            "selected_kernel_or_runtime": top_level_selected_kernel_or_runtime,
            "corpus": {
                "id": corpus.corpus_id(),
                "path": self.corpus.display().to_string(),
                "name": corpus.name.clone(),
                "description": corpus.description.clone(),
                "case_count": corpus.cases.len(),
                "selected_case_count": selected_case_ids.len(),
                "selected_case_ids": selected_case_ids,
                "metadata": answer_corpus_metadata_receipt(&corpus),
                "contract": answer_corpus_contract_receipt(&corpus),
            },
            "model": {
                "id": model_identity.id,
                "repo": top_level_model_repo,
                "revision": model_identity.revision,
                "file": top_level_model_file,
                "sha256": top_level_model_sha256,
                "bytes": model_identity.bytes,
                "family": top_level_model_family,
                "architecture": top_level_model_architecture,
                "quant_format": top_level_quantization,
                "path": self.model.display().to_string(),
                "loader_mode": "real_gguf",
                "fallback_loader_used": false,
                "tokenizer": aggregate_tokenizer,
                "tokenizer_authority": model_identity.tokenizer_authority,
                "tokenizer_path": self.tokenizer.as_ref().map(|path| path.display().to_string()),
                "answer_ready_artifact_available": answer_ready_artifact_available,
                "answer_ready": corpus.model.answer_ready,
            },
            "tokenizer": {
                "source": aggregate_tokenizer,
                "path": self.tokenizer.as_ref().map(|path| path.display().to_string()),
                "strict": corpus.defaults.strict_loader,
                "authority": corpus.model.tokenizer_authority,
            },
            "backend": {
                "requested_backend": device.as_str(),
                "selected_backend": top_level_selected_backend,
                "runtime_api": top_level_runtime_api,
                "fallback_used": top_level_fallback_used,
            },
            "execution_plan": aggregate_execution_plan,
            "proof_route_contract": proof_route_contract,
            "prompt_template_policy": {
                "family": corpus.defaults.prompt_template.as_str(),
                "identity_sha256": prompt_generation_identity["identity_sha256"].as_str(),
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
                "failure_categories": failure_category_summary(&rows),
            },
            "scoring_summary": scoring_summary(&rows),
            "scoring_contract": answer_corpus_scoring_contract_receipt(&corpus),
            "task_family_summary": task_family_summary(&rows),
            "profile_summary": profile_summary(&rows),
            "reference_comparison": reference_comparison_summary(
                &rows,
                bitnet_answer_path,
                &top_level_selected_backend,
                &top_level_runtime_api,
                top_level_fallback_used,
                corpus.defaults.prompt_template.as_str(),
                &corpus.model.tokenizer_authority,
                &corpus.metadata.reference_comparison_plan,
            ),
            "receipt_quality": {
                "case_receipt_checker": "answer_receipt_failed_rules",
                "checked": !self.dry_run,
                "required_case_fields": answer_receipt_required_case_fields(),
                "checked_rules": answer_receipt_checked_rules(),
            },
            "claim_boundary": {
                "slm_answer_path": slm_answer_path,
                "bounded_slm_answer_smoke_passed": bounded_slm_answer_smoke_passed,
                "dense_slm_clean_provenance": dense_slm_clean_provenance,
                "local_answer_path": device.as_str() == "apple-m4-cpu-neon",
                "answer_ready_artifact_available": answer_ready_artifact_available,
                "backend_quality_gate_passed": backend_quality_gate_passed,
                "diagnostic_only_until_answer_ready_artifact": a770_opencl_answer_corpus
                    || (bitnet_answer_path && !answer_ready_artifact_available)
                    || (slm_answer_path && !bounded_slm_answer_smoke_passed),
                "coherent_output_observed": coherent_answer_claimed,
                "coherent_answer_claimed": coherent_answer_claimed,
                "cuda_answer_corpus": is_cuda_answer_corpus_device(&device),
                "a770_opencl_answer_corpus": a770_opencl_answer_corpus,
                "a770_opencl_route_diagnostic": a770_opencl_answer_corpus,
                "strict_cuda_answer_claimed": false,
                "strict_a770_answer_claimed": false,
                "full_metal_inference_claimed": false,
                "full_a770_residency_claimed": false,
                "trusted_partial_acceleration_claimed": false,
                "a770_speedup_claimed": false,
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

    fn selected_cases<'a>(&self, corpus: &'a AnswerCorpus) -> Result<Vec<&'a AnswerCase>> {
        if self.case_ids.is_empty() {
            return Ok(corpus.cases.iter().collect());
        }

        let requested: BTreeSet<&str> = self.case_ids.iter().map(String::as_str).collect();
        let selected: Vec<&AnswerCase> =
            corpus.cases.iter().filter(|case| requested.contains(case.id.as_str())).collect();
        let found: BTreeSet<&str> = selected.iter().map(|case| case.id.as_str()).collect();
        let missing: Vec<&str> = requested.difference(&found).copied().collect();
        if !missing.is_empty() {
            anyhow::bail!("answer-corpus --case-id not found: {}", missing.join(", "));
        }
        Ok(selected)
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
        if corpus.defaults.qwen_no_think {
            args.push("--no-think".into());
        }
        if corpus.defaults.strict_loader {
            args.push("--strict-loader".into());
            args.push("--strict-tokenizer".into());
        }
        args.extend(answer_corpus_child_proof_args(device));
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
            case.scoring.as_ref(),
            Some(&token_ids),
            min_generated_tokens,
            min_distinct_generated_tokens,
        );
        quality.failed_rules.extend(answer_receipt_failed_rules(&run_receipt, device));
        quality.passed = quality.failed_rules.is_empty();
        let status = if quality.passed { "passed" } else { "quality_failed" };
        let failure_category_labels = failure_categories_for_case(
            case.task_family(),
            status,
            &quality.failed_rules,
            &quality.failure_taxonomy,
            quality.scoring.as_ref().map(|scoring| scoring.kind.as_str()),
        );
        let failure_categories = failure_category_fields(&failure_category_labels);

        Ok(json!({
            "id": case.id,
            "task_family": case.task_family(),
            "category": case.category.as_deref().unwrap_or_else(|| case.task_family()),
            "profile": case.profile(),
            "seed_material": case.seed_material,
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
            "logits_index_boundary": run_receipt.get("logits_index_boundary").cloned().unwrap_or(Value::Null),
            "prompt": {
                "rendered_text": run_receipt["prompt_render"]["rendered_text"]
                    .as_str()
                    .map(Value::from)
                    .unwrap_or_else(|| run_receipt["prompt"].clone()),
                "rendered_sha256": run_receipt["prompt_render"]["rendered_sha256"].clone(),
                "template_family": corpus.defaults.prompt_template,
                "qwen_no_think": corpus.defaults.qwen_no_think,
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
            "prompt_generation_identity": run_receipt["prompt_generation_identity"].clone(),
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
                "scoring": scoring_result_json(quality.scoring.as_ref()),
                "failed_rules": quality.failed_rules,
                "failure_taxonomy": quality.failure_taxonomy,
                "failure_category_labels": failure_category_labels,
                "failure_categories": failure_categories,
            },
            "backend": {
                "requested_backend": run_receipt["requested_backend"].clone(),
                "selected_backend": run_receipt["selected_backend"].clone(),
                "runtime_api": run_receipt["runtime_api"].clone(),
                "fallback_used": run_receipt["fallback_used"].clone(),
            },
            "timing": run_receipt.get("timing").cloned().unwrap_or(Value::Null),
            "latency": run_receipt.get("latency").cloned().unwrap_or(Value::Null),
            "throughput": run_receipt.get("throughput").cloned().unwrap_or(Value::Null),
            "execution_coverage": run_receipt.get("execution_coverage").cloned().unwrap_or(Value::Null),
            "qk256_hot_path": run_receipt.get("qk256_hot_path").cloned().unwrap_or(Value::Null),
            "execution_plan": run_receipt.get("execution_plan").cloned().unwrap_or(Value::Null),
            "proof_summary": run_receipt.get("proof_summary").cloned().unwrap_or(Value::Null),
            "kernel": {
                "selected_kernel": run_receipt["kernel"]["kernel_id"].clone(),
                "family": run_receipt["kernel"]["family"].clone(),
                "hot_path_kernel": run_receipt["kernel"]["hot_path_kernel_id"].clone(),
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
            "reference_comparison": reference_comparison_json(
                case,
                Some(&answer),
                Some(&token_ids),
                &run_receipt["selected_backend"],
                &run_receipt["runtime_api"],
                &run_receipt["fallback_used"],
            ),
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
            "task_family": case.task_family(),
            "category": case.category.as_deref().unwrap_or_else(|| case.task_family()),
            "profile": case.profile(),
            "seed_material": case.seed_material,
            "question": case.question,
            "status": "not_run",
            "reason": reason,
            "quality": {
                "passed": false,
                "failed_rules": ["not_run"],
                "failure_taxonomy": [],
                "failure_category_labels": [],
                "failure_categories": failure_category_fields(&[]),
                "scoring": scoring_not_run_json(case.scoring.as_ref()),
            },
            "backend": {
                "requested_backend": Value::Null,
                "selected_backend": Value::Null,
                "runtime_api": Value::Null,
                "fallback_used": Value::Null,
            },
            "tokens": Value::Null,
            "token_ids": {
                "prompt": Value::Null,
                "generated": Value::Null,
            },
            "reference_comparison": reference_comparison_json(
                case,
                None,
                None,
                &Value::Null,
                &Value::Null,
                &Value::Null,
            ),
        })
    }
}

#[derive(Debug, Deserialize)]
struct AnswerCorpus {
    schema: u32,
    artifact_kind: String,
    name: String,
    description: String,
    #[serde(default)]
    metadata: AnswerCorpusMetadata,
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
        let mut case_ids = BTreeSet::new();
        for case in &corpus.cases {
            if !case_ids.insert(case.id.as_str()) {
                anyhow::bail!("answer corpus case id `{}` is duplicated", case.id);
            }
            validate_answer_scoring(case)?;
        }
        validate_answer_corpus_contract(&corpus)?;
        Ok(corpus)
    }

    fn corpus_id(&self) -> &str {
        self.metadata
            .corpus_contract
            .as_ref()
            .map(|contract| contract.corpus_id.as_str())
            .filter(|value| !value.is_empty())
            .unwrap_or(self.name.as_str())
    }
}

#[derive(Debug, Default, Deserialize)]
struct AnswerCorpusMetadata {
    #[serde(default)]
    campaign: Option<String>,
    #[serde(default)]
    work_item: Option<String>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    generator_policy: Option<String>,
    #[serde(default)]
    case_count_target: Option<usize>,
    #[serde(default)]
    prompt_template: Option<String>,
    #[serde(default)]
    scoring_status: Option<String>,
    #[serde(default)]
    claim_boundary: Option<Value>,
    #[serde(default)]
    corpus_contract: Option<CorpusContract>,
    #[serde(default)]
    expected_answer_authority: Option<Value>,
    #[serde(default)]
    reference_comparison_plan: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct CorpusContract {
    #[serde(default)]
    contract_version: String,
    #[serde(default)]
    corpus_id: String,
    #[serde(default)]
    corpus_version: String,
    #[serde(default)]
    seed_generation_rules: String,
    #[serde(default)]
    expected_output_provenance: String,
    #[serde(default)]
    normalization_rules: String,
    #[serde(default)]
    scoring_schema: String,
    #[serde(default)]
    scorer_self_tests: Vec<String>,
    #[serde(default)]
    receipt_contract: String,
}

#[derive(Debug, Deserialize)]
struct CorpusModel {
    repo: String,
    #[serde(default, alias = "revision")]
    repo_revision: Option<String>,
    file: String,
    #[serde(default)]
    sha256: Option<String>,
    #[serde(default)]
    bytes: Option<u64>,
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    architecture: Option<String>,
    #[serde(default)]
    quant_format: Option<String>,
    #[serde(default)]
    answer_ready: Option<CorpusAnswerReady>,
    #[serde(default)]
    tokenizer_authority: Option<CorpusTokenizerAuthority>,
}

#[derive(Debug, Clone)]
struct AnswerCorpusModelIdentity {
    id: Option<String>,
    repo: String,
    revision: Option<String>,
    file: String,
    sha256: Option<String>,
    bytes: Option<u64>,
    family: Option<String>,
    architecture: Option<String>,
    quant_format: Option<String>,
    tokenizer_authority: Option<String>,
}

impl AnswerCorpusModelIdentity {
    fn resolve(model_id: Option<&str>, corpus: &AnswerCorpus) -> Result<Self> {
        let Some(model_id) = model_id else {
            return Ok(Self::from_corpus(&corpus.model));
        };
        if corpus.artifact_kind != "slm_answer_corpus" {
            anyhow::bail!(
                "answer-corpus --model-id is only supported for slm_answer_corpus receipts; corpus `{}` has artifact_kind `{}`",
                corpus.name,
                corpus.artifact_kind
            );
        }
        let metadata = model_cache::apple_m4_slm_model_receipt_metadata(model_id)?;
        Ok(Self::from_apple_m4_slm_model(metadata))
    }

    fn from_corpus(model: &CorpusModel) -> Self {
        Self {
            id: None,
            repo: model.repo.clone(),
            revision: model.repo_revision.clone(),
            file: model.file.clone(),
            sha256: model.sha256.clone(),
            bytes: model.bytes,
            family: model.family.clone(),
            architecture: model.architecture.clone(),
            quant_format: model.quant_format.clone(),
            tokenizer_authority: model
                .tokenizer_authority
                .as_ref()
                .map(|authority| authority.source.clone()),
        }
    }

    fn from_apple_m4_slm_model(metadata: model_cache::AppleM4SlmModelReceiptMetadata) -> Self {
        Self {
            id: Some(metadata.id.to_string()),
            repo: metadata.repo.to_string(),
            revision: Some(metadata.revision.to_string()),
            file: metadata.file.to_string(),
            sha256: Some(metadata.sha256.to_string()),
            bytes: Some(metadata.bytes),
            family: Some(metadata.family.to_string()),
            architecture: Some(metadata.architecture.to_string()),
            quant_format: Some(metadata.quantization.to_string()),
            tokenizer_authority: Some(metadata.tokenizer_authority.to_string()),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct CorpusAnswerReady {
    state: String,
    gate: Option<String>,
    manifest: Option<String>,
    evidence: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct CorpusTokenizerAuthority {
    source: String,
    repo: Option<String>,
    revision: Option<String>,
    sha256: Option<String>,
    ggml_pre: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CorpusDefaults {
    prompt_template: String,
    #[serde(default)]
    qwen_no_think: bool,
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
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    profile: Option<String>,
    #[serde(default)]
    seed_material: Option<String>,
    question: String,
    max_new_tokens: Option<usize>,
    timeout_seconds: Option<u64>,
    min_generated_tokens: Option<usize>,
    min_distinct_generated_tokens: Option<usize>,
    #[serde(default)]
    reference: Option<AnswerReference>,
    #[serde(default)]
    scoring: Option<AnswerScoring>,
    gate: AnswerGate,
}

impl AnswerCase {
    fn task_family(&self) -> &str {
        self.category.as_deref().unwrap_or("uncategorized")
    }

    fn profile(&self) -> &str {
        self.profile.as_deref().unwrap_or("unprofiled")
    }
}

#[derive(Debug, Deserialize)]
struct AnswerReference {
    #[serde(default)]
    runner: Option<String>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    receipt: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    generated_token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Deserialize)]
struct AnswerScoring {
    #[serde(default, alias = "planned_kind")]
    kind: Option<String>,
    #[serde(default)]
    expected: Option<String>,
    #[serde(default)]
    expected_normalized: Option<String>,
    #[serde(default)]
    schema: Option<Value>,
    #[serde(default)]
    expected_number: Option<f64>,
    #[serde(default, alias = "tolerance")]
    numeric_tolerance: Option<f64>,
    #[serde(default)]
    required_keywords: Option<Vec<String>>,
    #[serde(default)]
    forbidden_tokens: Option<Vec<String>>,
    #[serde(default)]
    expected_answer_authority: Option<Value>,
}

impl AnswerScoring {
    fn kind(&self) -> &str {
        self.kind.as_deref().unwrap_or("unspecified")
    }
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
    failure_taxonomy: Vec<String>,
    scoring: Option<ScoringResult>,
}

struct ScoringResult {
    kind: String,
    passed: bool,
    normalized_answer: String,
    failed_rules: Vec<String>,
    failure_taxonomy: Vec<String>,
    failure_categories: Vec<String>,
    details: Value,
}

fn normalize_answer_corpus_device(device: &str) -> String {
    match device.trim() {
        "auto" => "cpu".to_string(),
        INTEL_ARC_A770_OPENCL | "a770-opencl" => INTEL_A770_OPENCL.to_string(),
        other => other.to_string(),
    }
}

fn answer_corpus_artifact_kind(device: &str, corpus_artifact_kind: &str) -> &'static str {
    match (device, corpus_artifact_kind) {
        ("apple-m4-cpu-neon", _) => "bitnet_apple_m4_local_answer_corpus",
        ("cuda" | RTX_5070_TI_CUDA, _) => "bitnet_cuda_answer_diagnostic_corpus",
        (INTEL_A770_OPENCL, _) => "bitnet_a770_opencl_answer_diagnostic_corpus",
        (_, "slm_answer_corpus") => "slm_cpu_answer_corpus",
        _ => "bitnet_cpu_answer_corpus",
    }
}

fn answer_corpus_runtime_api(device: &str) -> &'static str {
    if is_cuda_answer_corpus_device(device) {
        "cuda"
    } else if is_a770_opencl_answer_corpus_device(device) {
        "opencl"
    } else {
        "cpu"
    }
}

fn is_cuda_answer_corpus_device(device: &str) -> bool {
    matches!(device, "cuda" | RTX_5070_TI_CUDA)
}

fn is_a770_opencl_answer_corpus_device(device: &str) -> bool {
    device == INTEL_A770_OPENCL
}

fn answer_corpus_child_proof_args(device: &str) -> Vec<OsString> {
    if !is_a770_opencl_answer_corpus_device(device) {
        return Vec::new();
    }
    vec![
        "--proof-model-contract".into(),
        A770_BITNET_MODEL_CONTRACT.into(),
        "--proof-kernel-route".into(),
        A770_BITNET_QK256_ROUTE_ID.into(),
    ]
}

fn answer_corpus_proof_route_contract(device: &str) -> Value {
    if !is_a770_opencl_answer_corpus_device(device) {
        return Value::Null;
    }
    json!({
        "enabled": true,
        "diagnostic_only": true,
        "claimable": false,
        "model_contract": A770_BITNET_MODEL_CONTRACT,
        "kernel_route": {
            "route_id": A770_BITNET_QK256_ROUTE_ID,
            "route_declared_in_child_runs": true,
            "diagnostic_only": true,
            "claimable": false,
        },
        "not_claims": [
            "answer_quality",
            "full_bitnet_inference",
            "trusted_partial_acceleration",
            "full_a770_residency",
            "speedup",
        ],
    })
}

fn answer_corpus_backend_lane(
    device: &str,
    slm_answer_path: bool,
    model_family: &str,
) -> &'static str {
    if slm_answer_path && device == "cpu" && model_family == "qwen" {
        "dense_slm_cpu"
    } else if is_cuda_answer_corpus_device(device) {
        "bitnet_cuda"
    } else if is_a770_opencl_answer_corpus_device(device) {
        "bitnet_a770_opencl"
    } else if device == "apple-m4-cpu-neon" {
        "apple_m4_cpu_neon"
    } else {
        "bitnet_cpu"
    }
}

fn aggregate_case_str<'a>(rows: &'a [Value], path: &[&str]) -> Option<&'a str> {
    rows.iter().find_map(|row| {
        let mut cursor = row;
        for key in path {
            cursor = cursor.get(*key)?;
        }
        cursor.as_str()
    })
}

fn aggregate_case_value(rows: &[Value], path: &[&str]) -> Option<Value> {
    rows.iter().find_map(|row| {
        let mut cursor = row;
        for key in path {
            cursor = cursor.get(*key)?;
        }
        Some(cursor.clone())
    })
}

fn ensure_rows_match_model_identity(
    rows: &[Value],
    identity: &AnswerCorpusModelIdentity,
) -> Result<()> {
    let Some(model_id) = identity.id.as_deref() else {
        return Ok(());
    };

    let mut mismatches = Vec::new();
    for row in rows {
        let Some(case_id) = row.get("id").and_then(Value::as_str) else {
            continue;
        };
        let Some(model) = row.get("model") else {
            continue;
        };
        for (field, expected) in [
            ("file", Some(identity.file.as_str())),
            ("sha256", identity.sha256.as_deref()),
            ("family", identity.family.as_deref()),
            ("architecture", identity.architecture.as_deref()),
            ("quant_format", identity.quant_format.as_deref()),
        ] {
            let Some(expected) = expected else {
                continue;
            };
            let observed = model.get(field).and_then(Value::as_str);
            if observed != Some(expected) {
                mismatches.push(format!(
                    "{case_id}.model.{field}: expected `{expected}`, observed `{}`",
                    observed.unwrap_or("<missing>")
                ));
            }
        }
    }

    if !mismatches.is_empty() {
        anyhow::bail!(
            "answer-corpus --model-id `{model_id}` does not match child run model metadata: {}",
            mismatches.join("; ")
        );
    }
    Ok(())
}

fn validate_answer_corpus_inputs(
    command: &AnswerCorpusCommand,
    corpus: &AnswerCorpus,
) -> Result<()> {
    if !command.model.exists() {
        anyhow::bail!(
            "answer-corpus model not found: {}. Strict local-answer proof requires a real model path; hidden fallback is not allowed. Use --dry-run to validate corpus shape without loading a model.",
            command.model.display()
        );
    }

    let requires_external_tokenizer = corpus
        .model
        .tokenizer_authority
        .as_ref()
        .is_some_and(|authority| authority.source == "external_tokenizer_json");

    if let Some(tokenizer) = &command.tokenizer {
        if !tokenizer.exists() {
            anyhow::bail!(
                "answer-corpus tokenizer not found: {}. Corpus `{}` requires strict tokenizer authority; hidden tokenizer fallback is not allowed.",
                tokenizer.display(),
                corpus.name
            );
        }
    } else if requires_external_tokenizer {
        anyhow::bail!(
            "answer-corpus requires --tokenizer for corpus `{}` because tokenizer authority source is external_tokenizer_json; hidden tokenizer fallback is not allowed.",
            corpus.name
        );
    }

    Ok(())
}

fn validate_answer_corpus_contract(corpus: &AnswerCorpus) -> Result<()> {
    let Some(contract) = corpus.metadata.corpus_contract.as_ref() else {
        return Ok(());
    };

    let required_fields = [
        ("contract_version", contract.contract_version.as_str()),
        ("corpus_id", contract.corpus_id.as_str()),
        ("corpus_version", contract.corpus_version.as_str()),
        ("seed_generation_rules", contract.seed_generation_rules.as_str()),
        ("expected_output_provenance", contract.expected_output_provenance.as_str()),
        ("normalization_rules", contract.normalization_rules.as_str()),
        ("scoring_schema", contract.scoring_schema.as_str()),
        ("receipt_contract", contract.receipt_contract.as_str()),
    ];
    let missing: Vec<&str> = required_fields
        .into_iter()
        .filter_map(|(field, value)| value.trim().is_empty().then_some(field))
        .collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "answer corpus `{}` corpus_contract is missing required fields: {}",
            corpus.name,
            missing.join(", ")
        );
    }
    if contract.scorer_self_tests.is_empty()
        || contract.scorer_self_tests.iter().any(|value| value.trim().is_empty())
    {
        anyhow::bail!(
            "answer corpus `{}` corpus_contract.scorer_self_tests must list non-empty scorer self-tests",
            corpus.name
        );
    }
    if contract.corpus_id != corpus.name {
        anyhow::bail!(
            "answer corpus `{}` corpus_contract.corpus_id must match corpus name `{}`",
            contract.corpus_id,
            corpus.name
        );
    }
    if corpus.metadata.seed.is_none() {
        anyhow::bail!("answer corpus `{}` corpus_contract requires metadata.seed", corpus.name);
    }
    if corpus.metadata.generator_policy.as_deref().is_none_or(str::is_empty) {
        anyhow::bail!(
            "answer corpus `{}` corpus_contract requires metadata.generator_policy",
            corpus.name
        );
    }
    if corpus.metadata.case_count_target != Some(corpus.cases.len()) {
        anyhow::bail!(
            "answer corpus `{}` metadata.case_count_target must equal case count {}",
            corpus.name,
            corpus.cases.len()
        );
    }
    if corpus.metadata.prompt_template.as_deref() != Some(corpus.defaults.prompt_template.as_str())
    {
        anyhow::bail!(
            "answer corpus `{}` metadata.prompt_template must match defaults.prompt_template `{}`",
            corpus.name,
            corpus.defaults.prompt_template
        );
    }
    Ok(())
}

fn answer_corpus_metadata_receipt(corpus: &AnswerCorpus) -> Value {
    json!({
        "campaign": corpus.metadata.campaign,
        "work_item": corpus.metadata.work_item,
        "seed": corpus.metadata.seed,
        "generator_policy": corpus.metadata.generator_policy,
        "case_count_target": corpus.metadata.case_count_target,
        "prompt_template": corpus.metadata.prompt_template,
        "scoring_status": corpus.metadata.scoring_status,
        "expected_answer_authority": corpus.metadata.expected_answer_authority,
        "reference_comparison_plan": corpus.metadata.reference_comparison_plan,
        "claim_boundary": corpus.metadata.claim_boundary,
    })
}

fn answer_corpus_contract_receipt(corpus: &AnswerCorpus) -> Value {
    corpus
        .metadata
        .corpus_contract
        .as_ref()
        .map(|contract| {
            json!({
                "contract_version": contract.contract_version,
                "corpus_id": contract.corpus_id,
                "corpus_version": contract.corpus_version,
                "seed_generation_rules": contract.seed_generation_rules,
                "expected_output_provenance": contract.expected_output_provenance,
                "normalization_rules": contract.normalization_rules,
                "scoring_schema": contract.scoring_schema,
                "scorer_self_tests": contract.scorer_self_tests,
                "receipt_contract": contract.receipt_contract,
            })
        })
        .unwrap_or(Value::Null)
}

fn answer_corpus_scoring_contract_receipt(corpus: &AnswerCorpus) -> Value {
    let contract = corpus.metadata.corpus_contract.as_ref();
    json!({
        "contract_version": contract.map(|contract| contract.contract_version.as_str()),
        "scoring_schema": contract.map(|contract| contract.scoring_schema.as_str()),
        "normalization_rules": contract.map(|contract| contract.normalization_rules.as_str()),
        "expected_output_provenance": contract
            .map(|contract| contract.expected_output_provenance.as_str()),
        "expected_answer_authority": corpus.metadata.expected_answer_authority,
        "reference_comparison_plan": corpus.metadata.reference_comparison_plan,
        "receipt_contract": contract.map(|contract| contract.receipt_contract.as_str()),
        "scorer_self_tests": contract.map(|contract| contract.scorer_self_tests.as_slice()),
        "supported_scoring_kinds": ANSWER_CORPUS_SCORING_KINDS,
        "supported_failure_categories": ANSWER_CORPUS_FAILURE_CATEGORIES,
    })
}

fn corpus_answer_ready_artifact_available(model: &CorpusModel) -> bool {
    model.answer_ready.as_ref().is_some_and(|authority| authority.state == "answer_ready")
}

fn validate_answer_scoring(case: &AnswerCase) -> Result<()> {
    let Some(scoring) = &case.scoring else {
        return Ok(());
    };
    match scoring.kind() {
        "exact_match" | "normalized_match" | "contains_expected" => {
            if scoring.expected_answer().is_none() {
                anyhow::bail!(
                    "answer corpus case `{}` scoring `{}` requires expected or expected_normalized",
                    case.id,
                    scoring.kind()
                );
            }
        }
        "json_schema" => {
            if scoring.schema.is_none() {
                anyhow::bail!(
                    "answer corpus case `{}` scoring json_schema requires schema",
                    case.id
                );
            }
        }
        "numeric_tolerance" => {
            let has_expected_number = scoring.expected_number.is_some()
                || scoring.expected_answer().and_then(first_number).is_some();
            if !has_expected_number {
                anyhow::bail!(
                    "answer corpus case `{}` scoring numeric_tolerance requires expected_number or numeric expected text",
                    case.id
                );
            }
            if scoring.numeric_tolerance.unwrap_or(0.0) < 0.0 {
                anyhow::bail!(
                    "answer corpus case `{}` scoring numeric_tolerance cannot be negative",
                    case.id
                );
            }
        }
        "required_keywords" => {
            if scoring.required_keywords.as_ref().is_none_or(Vec::is_empty) {
                anyhow::bail!(
                    "answer corpus case `{}` scoring required_keywords requires required_keywords",
                    case.id
                );
            }
        }
        "forbidden_tokens" => {
            if scoring.forbidden_tokens.as_ref().is_none_or(Vec::is_empty) {
                anyhow::bail!(
                    "answer corpus case `{}` scoring forbidden_tokens requires forbidden_tokens",
                    case.id
                );
            }
        }
        "required_forbidden_tokens" => {
            if scoring.required_keywords.as_ref().is_none_or(Vec::is_empty)
                && scoring.forbidden_tokens.as_ref().is_none_or(Vec::is_empty)
            {
                anyhow::bail!(
                    "answer corpus case `{}` scoring required_forbidden_tokens requires required_keywords or forbidden_tokens",
                    case.id
                );
            }
        }
        other => {
            anyhow::bail!(
                "answer corpus case `{}` has unsupported scoring kind `{other}`",
                case.id
            );
        }
    }
    Ok(())
}

impl AnswerScoring {
    fn expected_answer(&self) -> Option<&str> {
        self.expected_normalized.as_deref().or(self.expected.as_deref())
    }
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
    scoring: Option<&AnswerScoring>,
    generated_token_ids: Option<&[u32]>,
    min_generated_tokens: Option<usize>,
    min_distinct_generated_tokens: Option<usize>,
) -> QualityResult {
    let normalized = normalize_scoring_text(answer);
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

    let gate_answer = gate_evaluation_text(&normalized, scoring);
    if !gate_passed(&gate_answer, gate) {
        failed_rules.push(format!("gate_{}", gate.kind));
    }
    let scoring_result = scoring.map(|scoring| evaluate_scoring(&normalized, scoring));
    if let Some(scoring_result) = &scoring_result
        && !scoring_result.passed
    {
        failed_rules
            .extend(scoring_result.failed_rules.iter().map(|rule| format!("scoring_{rule}")));
    }
    let exact_scoring_passed = scoring
        .map(|scoring| {
            matches!(scoring.kind(), "exact_match" | "normalized_match" | "numeric_tolerance")
        })
        .unwrap_or(false)
        && scoring_result.as_ref().is_some_and(|result| result.passed);
    if let Some(minimum) = min_generated_tokens
        && generated_token_count < minimum
    {
        failed_rules.push("generated_token_min".to_string());
    }
    if let Some(minimum) = min_distinct_generated_tokens
        && distinct_generated_tokens < minimum
        && !exact_scoring_passed
    {
        failed_rules.push("generated_token_variation".to_string());
    }
    let failure_taxonomy =
        quality_failure_taxonomy(&failed_rules, scoring_result.as_ref(), &normalized);

    QualityResult {
        passed: failed_rules.is_empty(),
        printable_utf8,
        non_empty_answer,
        no_replacement_chars,
        no_raw_special_tokens,
        mostly_text,
        distinct_generated_tokens,
        failed_rules,
        failure_taxonomy,
        scoring: scoring_result,
    }
}

fn evaluate_scoring(answer: &str, scoring: &AnswerScoring) -> ScoringResult {
    let kind = scoring.kind().to_string();
    let normalized_answer = normalize_scoring_text(answer);
    let mut failed_rules = Vec::new();
    let mut details = Map::new();
    details.insert("kind".to_string(), Value::String(kind.clone()));
    insert_expected_answer_authority(&mut details, scoring);

    match kind.as_str() {
        "exact_match" => {
            let expected = scoring.expected_answer().unwrap_or_default();
            let observed = normalized_answer.trim();
            details.insert("expected".to_string(), Value::String(expected.to_string()));
            details.insert("observed".to_string(), Value::String(observed.to_string()));
            if observed != expected.trim() {
                failed_rules.push("exact_match".to_string());
            }
        }
        "normalized_match" => {
            let expected = scoring.expected_answer().unwrap_or_default();
            let observed = normalize_match_text(&normalized_answer);
            let expected_normalized = normalize_match_text(expected);
            let observed_compact = compact_match_text(&observed);
            let expected_compact = compact_match_text(&expected_normalized);
            details.insert(
                "expected_normalized".to_string(),
                Value::String(expected_normalized.clone()),
            );
            details.insert("observed_normalized".to_string(), Value::String(observed.clone()));
            details.insert("expected_compact".to_string(), Value::String(expected_compact.clone()));
            details.insert("observed_compact".to_string(), Value::String(observed_compact.clone()));
            if observed != expected_normalized && observed_compact != expected_compact {
                failed_rules.push("normalized_match".to_string());
            }
        }
        "contains_expected" => {
            let expected = scoring.expected_answer().unwrap_or_default();
            let observed = normalize_match_text(&normalized_answer);
            let expected_normalized = normalize_match_text(expected);
            let expected_present = !expected_normalized.is_empty()
                && contains_keyword_form(&observed, &expected_normalized);
            details.insert("expected_normalized".to_string(), Value::String(expected_normalized));
            details.insert("observed_normalized".to_string(), Value::String(observed));
            details.insert("expected_present".to_string(), Value::Bool(expected_present));
            if !expected_present {
                failed_rules.push("contains_expected".to_string());
            }
        }
        "json_schema" => {
            let schema = scoring.schema.as_ref();
            let schema_result = schema
                .and_then(normalized_json_schema)
                .map(|schema| validate_schema_style_json(&normalized_answer, &schema))
                .unwrap_or_else(|| SchemaStyleResult {
                    parsed: false,
                    source: "raw",
                    failures: vec!["schema_missing_or_invalid".to_string()],
                });
            details.insert("parsed_json".to_string(), Value::Bool(schema_result.parsed));
            details
                .insert("json_source".to_string(), Value::String(schema_result.source.to_string()));
            if !schema_result.failures.is_empty() {
                failed_rules.extend(schema_result.failures);
            }
        }
        "numeric_tolerance" => {
            let expected = scoring
                .expected_number
                .or_else(|| scoring.expected_answer().and_then(first_number));
            let tolerance = scoring.numeric_tolerance.unwrap_or(0.0);
            let observed_candidates = numeric_answer_candidates(&normalized_answer);
            let matching = expected.and_then(|expected| {
                observed_candidates
                    .iter()
                    .copied()
                    .find(|observed| (*observed - expected).abs() <= tolerance)
            });
            let observed = matching.or_else(|| observed_candidates.first().copied());
            details.insert("expected_number".to_string(), json!(expected));
            details.insert("observed_number".to_string(), json!(observed));
            details.insert("observed_number_candidates".to_string(), json!(observed_candidates));
            details.insert("numeric_tolerance".to_string(), json!(tolerance));
            match (expected, observed) {
                (Some(_), Some(_)) if matching.is_some() => {}
                (Some(_), Some(_)) => failed_rules.push("numeric_tolerance".to_string()),
                (Some(_), None) => failed_rules.push("numeric_observed_missing".to_string()),
                (None, _) => failed_rules.push("numeric_expected_missing".to_string()),
            }
        }
        "required_keywords" => {
            let missing =
                missing_keywords(&normalized_answer, scoring.required_keywords.as_deref());
            let has_missing = !missing.is_empty();
            details.insert("required_keywords_missing".to_string(), json!(missing));
            if has_missing {
                failed_rules.push("required_keywords".to_string());
            }
        }
        "forbidden_tokens" => {
            let observed =
                observed_forbidden_tokens(&normalized_answer, scoring.forbidden_tokens.as_deref());
            let has_observed = !observed.is_empty();
            details.insert("forbidden_tokens_observed".to_string(), json!(observed));
            if has_observed {
                failed_rules.push("forbidden_tokens".to_string());
            }
        }
        "required_forbidden_tokens" => {
            let missing =
                missing_keywords(&normalized_answer, scoring.required_keywords.as_deref());
            let observed =
                observed_forbidden_tokens(&normalized_answer, scoring.forbidden_tokens.as_deref());
            let has_missing = !missing.is_empty();
            let has_observed = !observed.is_empty();
            details.insert("required_keywords_missing".to_string(), json!(missing));
            details.insert("forbidden_tokens_observed".to_string(), json!(observed));
            if has_missing {
                failed_rules.push("required_keywords".to_string());
            }
            if has_observed {
                failed_rules.push("forbidden_tokens".to_string());
            }
        }
        _ => failed_rules.push("unsupported_kind".to_string()),
    }

    let failure_taxonomy = scoring_failure_taxonomy(answer, scoring, &kind, &failed_rules);
    let failure_categories =
        failure_categories_for_case("", "", &failed_rules, &failure_taxonomy, Some(&kind));

    ScoringResult {
        kind,
        passed: failed_rules.is_empty(),
        normalized_answer,
        failure_taxonomy,
        failure_categories,
        failed_rules,
        details: Value::Object(details),
    }
}

fn scoring_result_json(result: Option<&ScoringResult>) -> Value {
    result.map_or(Value::Null, |result| {
        json!({
            "kind": &result.kind,
            "passed": result.passed,
            "normalized_answer": &result.normalized_answer,
            "failed_rules": &result.failed_rules,
            "failure_taxonomy": &result.failure_taxonomy,
            "failure_category_labels": &result.failure_categories,
            "failure_categories": failure_category_fields(&result.failure_categories),
            "details": &result.details,
        })
    })
}

fn insert_expected_answer_authority(details: &mut Map<String, Value>, scoring: &AnswerScoring) {
    if let Some(authority) = &scoring.expected_answer_authority {
        details.insert("expected_answer_authority".to_string(), authority.clone());
    }
}

fn scoring_not_run_json(scoring: Option<&AnswerScoring>) -> Value {
    scoring.map_or(Value::Null, |scoring| {
        json!({
            "kind": scoring.kind(),
            "status": "not_run",
            "passed": Value::Null,
            "expected": &scoring.expected,
            "expected_normalized": &scoring.expected_normalized,
            "schema": &scoring.schema,
            "expected_number": scoring.expected_number,
            "numeric_tolerance": scoring.numeric_tolerance,
            "required_keywords": &scoring.required_keywords,
            "forbidden_tokens": &scoring.forbidden_tokens,
            "expected_answer_authority": &scoring.expected_answer_authority,
            "failure_taxonomy": [],
            "failure_category_labels": [],
            "failure_categories": failure_category_fields(&[]),
        })
    })
}

fn scoring_summary(rows: &[Value]) -> Value {
    let mut total = 0usize;
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut not_run = 0usize;
    let mut kinds = BTreeSet::new();
    let mut failure_taxonomy = BTreeMap::<String, usize>::new();
    let mut failure_categories = BTreeMap::<String, usize>::new();
    for row in rows {
        let scoring = &row["quality"]["scoring"];
        let Some(kind) = scoring["kind"].as_str() else {
            continue;
        };
        total += 1;
        kinds.insert(kind.to_string());
        match scoring["passed"].as_bool() {
            Some(true) => passed += 1,
            Some(false) => failed += 1,
            None => not_run += 1,
        }
        for taxonomy in
            scoring["failure_taxonomy"].as_array().into_iter().flatten().filter_map(Value::as_str)
        {
            *failure_taxonomy.entry(taxonomy.to_string()).or_default() += 1;
        }
        let mut scoring_categories = failure_categories_from_value(scoring);
        if scoring_categories.is_empty() {
            let failed_rules = scoring["failed_rules"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>();
            let taxonomy = scoring["failure_taxonomy"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>();
            scoring_categories.extend(failure_categories_for_case(
                "",
                "",
                &failed_rules,
                &taxonomy,
                Some(kind),
            ));
        }
        for category in scoring_categories {
            *failure_categories.entry(category).or_default() += 1;
        }
    }
    json!({
        "enabled": total > 0,
        "total": total,
        "passed": passed,
        "failed": failed,
        "not_run": not_run,
        "kinds": kinds.into_iter().collect::<Vec<_>>(),
        "failure_taxonomy": failure_taxonomy,
        "failure_categories": failure_categories,
    })
}

fn failure_category_summary(rows: &[Value]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::<String, usize>::new();
    for row in rows {
        for category in row_failure_categories(row) {
            *counts.entry(category.to_string()).or_default() += 1;
        }
    }
    counts
}

fn task_family_summary(rows: &[Value]) -> Value {
    let mut families = BTreeMap::<String, TaskFamilyStats>::new();
    for row in rows {
        let family = row["task_family"]
            .as_str()
            .or_else(|| row["category"].as_str())
            .unwrap_or("uncategorized");
        let stats = families.entry(family.to_string()).or_default();
        stats.total += 1;
        match row["status"].as_str().unwrap_or("unknown") {
            "passed" => stats.passed += 1,
            "quality_failed" | "command_failed" => stats.failed += 1,
            "timeout" => stats.timeout += 1,
            "not_run" => stats.not_run += 1,
            status => *stats.status_counts.entry(status.to_string()).or_default() += 1,
        }

        let scoring = &row["quality"]["scoring"];
        if let Some(kind) = scoring["kind"].as_str() {
            stats.scoring_total += 1;
            stats.scoring_kinds.insert(kind.to_string());
            match scoring["passed"].as_bool() {
                Some(true) => stats.scoring_passed += 1,
                Some(false) => stats.scoring_failed += 1,
                None => stats.scoring_not_run += 1,
            }
        }

        for taxonomy in row_failure_taxonomy(row) {
            *stats.failure_taxonomy.entry(taxonomy.to_string()).or_default() += 1;
        }
        for category in row_failure_categories(row) {
            *stats.failure_categories.entry(category.to_string()).or_default() += 1;
        }
    }

    Value::Object(
        families
            .into_iter()
            .map(|(family, stats)| (family, stats.into_json()))
            .collect::<Map<_, _>>(),
    )
}

fn profile_summary(rows: &[Value]) -> Value {
    let mut profiles = BTreeMap::<String, TaskFamilyStats>::new();
    for row in rows {
        let profile = row["profile"].as_str().unwrap_or("unprofiled");
        let stats = profiles.entry(profile.to_string()).or_default();
        stats.total += 1;
        match row["status"].as_str().unwrap_or("unknown") {
            "passed" => stats.passed += 1,
            "quality_failed" | "command_failed" => stats.failed += 1,
            "timeout" => stats.timeout += 1,
            "not_run" => stats.not_run += 1,
            status => *stats.status_counts.entry(status.to_string()).or_default() += 1,
        }

        let scoring = &row["quality"]["scoring"];
        if let Some(kind) = scoring["kind"].as_str() {
            stats.scoring_total += 1;
            stats.scoring_kinds.insert(kind.to_string());
            match scoring["passed"].as_bool() {
                Some(true) => stats.scoring_passed += 1,
                Some(false) => stats.scoring_failed += 1,
                None => stats.scoring_not_run += 1,
            }
        }

        for taxonomy in row_failure_taxonomy(row) {
            *stats.failure_taxonomy.entry(taxonomy.to_string()).or_default() += 1;
        }
        for category in row_failure_categories(row) {
            *stats.failure_categories.entry(category.to_string()).or_default() += 1;
        }
    }

    Value::Object(
        profiles
            .into_iter()
            .map(|(profile, stats)| (profile, stats.into_json()))
            .collect::<Map<_, _>>(),
    )
}

fn row_failure_taxonomy(row: &Value) -> BTreeSet<&str> {
    let mut taxonomy = BTreeSet::new();
    for value in row["quality"]["failure_taxonomy"].as_array().into_iter().flatten() {
        if let Some(value) = value.as_str() {
            taxonomy.insert(value);
        }
    }
    for value in row["quality"]["scoring"]["failure_taxonomy"].as_array().into_iter().flatten() {
        if let Some(value) = value.as_str() {
            taxonomy.insert(value);
        }
    }
    taxonomy
}

fn row_failure_categories(row: &Value) -> BTreeSet<String> {
    let mut categories = BTreeSet::new();
    categories.extend(failure_categories_from_value(&row["quality"]));
    categories.extend(failure_categories_from_value(&row["quality"]["scoring"]));
    if categories.is_empty() {
        let failed_rules = row["quality"]["failed_rules"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect::<Vec<_>>();
        let taxonomy =
            row_failure_taxonomy(row).into_iter().map(str::to_string).collect::<Vec<_>>();
        let task_family =
            row["task_family"].as_str().or_else(|| row["category"].as_str()).unwrap_or_default();
        let status = row["status"].as_str().unwrap_or_default();
        categories.extend(failure_categories_for_case(
            task_family,
            status,
            &failed_rules,
            &taxonomy,
            None,
        ));
    }
    categories
}

fn failure_categories_from_value(value: &Value) -> BTreeSet<String> {
    let mut categories = BTreeSet::new();
    if let Some(labels) = value["failure_category_labels"].as_array() {
        for label in labels.iter().filter_map(Value::as_str) {
            categories.insert(label.to_string());
        }
    }
    if let Some(fields) = value["failure_categories"].as_object() {
        for category in ANSWER_CORPUS_FAILURE_CATEGORIES {
            if fields.get(*category).and_then(Value::as_bool) == Some(true) {
                categories.insert((*category).to_string());
            }
        }
    }
    categories
}

#[derive(Default)]
struct TaskFamilyStats {
    total: usize,
    passed: usize,
    failed: usize,
    timeout: usize,
    not_run: usize,
    status_counts: BTreeMap<String, usize>,
    scoring_total: usize,
    scoring_passed: usize,
    scoring_failed: usize,
    scoring_not_run: usize,
    scoring_kinds: BTreeSet<String>,
    failure_taxonomy: BTreeMap<String, usize>,
    failure_categories: BTreeMap<String, usize>,
}

impl TaskFamilyStats {
    fn into_json(self) -> Value {
        json!({
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "timeout": self.timeout,
            "not_run": self.not_run,
            "other_statuses": self.status_counts,
            "scoring": {
                "enabled": self.scoring_total > 0,
                "total": self.scoring_total,
                "passed": self.scoring_passed,
                "failed": self.scoring_failed,
                "not_run": self.scoring_not_run,
                "kinds": self.scoring_kinds.into_iter().collect::<Vec<_>>(),
            },
            "failure_taxonomy": self.failure_taxonomy,
            "failure_categories": self.failure_categories,
        })
    }
}

fn reference_comparison_json(
    case: &AnswerCase,
    answer: Option<&str>,
    generated_token_ids: Option<&[u32]>,
    selected_backend: &Value,
    runtime_api: &Value,
    fallback_used: &Value,
) -> Value {
    let reference = case.reference.as_ref();
    let reference_text = reference.and_then(|reference| reference.text.as_deref());
    let reference_token_ids =
        reference.and_then(|reference| reference.generated_token_ids.as_ref());
    let rust_answer_available = answer.is_some();
    let mut comparable_fields = Vec::new();
    let mut missing_fields = Vec::new();
    let mut mismatched_fields = Vec::new();

    let text_match = match (reference_text, answer) {
        (Some(reference), Some(answer)) => {
            comparable_fields.push("text");
            let matched = reference.trim() == answer.trim();
            if !matched {
                mismatched_fields.push("text");
            }
            Some(matched)
        }
        (Some(_), None) => {
            missing_fields.push("rust.text");
            None
        }
        (None, Some(_)) => {
            missing_fields.push("reference.text");
            None
        }
        (None, None) => None,
    };
    let generated_token_ids_match = match (reference_token_ids, generated_token_ids) {
        (Some(reference), Some(generated)) => {
            comparable_fields.push("generated_token_ids");
            let matched = reference.as_slice() == generated;
            if !matched {
                mismatched_fields.push("generated_token_ids");
            }
            Some(matched)
        }
        (Some(_), None) => {
            missing_fields.push("rust.generated_token_ids");
            None
        }
        (None, Some(_)) => {
            missing_fields.push("reference.generated_token_ids");
            None
        }
        (None, None) => None,
    };

    let status =
        if reference.is_none() || (reference_text.is_none() && reference_token_ids.is_none()) {
            "reference_not_supplied"
        } else if !rust_answer_available {
            "not_run"
        } else if !mismatched_fields.is_empty() {
            "mismatched"
        } else if comparable_fields.len() < 2 || !missing_fields.is_empty() {
            "partially_compared"
        } else {
            "matched"
        };

    json!({
        "schema": "bitnet_reference_vs_rust_v1",
        "enabled": reference.is_some(),
        "reference": {
            "runner": reference.and_then(|reference| reference.runner.as_deref()),
            "source": reference.and_then(|reference| reference.source.as_deref()),
            "receipt": reference.and_then(|reference| reference.receipt.as_deref()),
            "text": reference_text,
            "generated_token_ids": reference_token_ids,
        },
        "rust": {
            "status": if rust_answer_available { "available" } else { "not_run" },
            "selected_backend": selected_backend,
            "runtime_api": runtime_api,
            "fallback_used": fallback_used,
            "text": answer,
            "generated_token_ids": generated_token_ids,
        },
        "comparison": {
            "status": status,
            "text_match": text_match,
            "generated_token_ids_match": generated_token_ids_match,
            "comparable_fields": comparable_fields,
            "missing_fields": missing_fields,
            "mismatched_fields": mismatched_fields,
        }
    })
}

fn reference_comparison_summary(
    rows: &[Value],
    bitnet_answer_path: bool,
    selected_backend: &str,
    runtime_api: &str,
    fallback_used: bool,
    prompt_template: &str,
    tokenizer_authority: &Option<CorpusTokenizerAuthority>,
    reference_comparison_plan: &Option<Value>,
) -> Value {
    let mut status_counts = BTreeMap::<String, usize>::new();
    let mut reference_supplied = 0usize;
    let mut reference_not_supplied = 0usize;
    let mut comparable_cases = 0usize;
    let mut text_matches = 0usize;
    let mut generated_token_id_matches = 0usize;
    let mut mismatched_fields = BTreeMap::<String, usize>::new();

    for row in rows {
        let comparison = &row["reference_comparison"]["comparison"];
        let status = comparison["status"].as_str().unwrap_or("missing");
        *status_counts.entry(status.to_string()).or_default() += 1;
        if row["reference_comparison"]["enabled"].as_bool() == Some(true) {
            reference_supplied += 1;
        } else {
            reference_not_supplied += 1;
        }
        if comparison["comparable_fields"].as_array().is_some_and(|fields| !fields.is_empty()) {
            comparable_cases += 1;
        }
        if comparison["text_match"].as_bool() == Some(true) {
            text_matches += 1;
        }
        if comparison["generated_token_ids_match"].as_bool() == Some(true) {
            generated_token_id_matches += 1;
        }
        for field in comparison["mismatched_fields"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            *mismatched_fields.entry(field.to_string()).or_default() += 1;
        }
    }

    json!({
        "schema": "bitnet_reference_vs_rust_v1",
        "enabled": bitnet_answer_path,
        "reference_runner_required": bitnet_answer_path,
        "reference_comparison_plan": reference_comparison_plan,
        "rust_runner": {
            "selected_backend": selected_backend,
            "runtime_api": runtime_api,
            "fallback_used": fallback_used,
            "prompt_template": prompt_template,
            "tokenizer_authority": tokenizer_authority,
        },
        "summary": {
            "total": rows.len(),
            "reference_supplied": reference_supplied,
            "reference_not_supplied": reference_not_supplied,
            "comparable_cases": comparable_cases,
            "matched": status_counts.get("matched").copied().unwrap_or_default(),
            "mismatched": status_counts.get("mismatched").copied().unwrap_or_default(),
            "partially_compared": status_counts
                .get("partially_compared")
                .copied()
                .unwrap_or_default(),
            "not_run": status_counts.get("not_run").copied().unwrap_or_default(),
            "status_counts": status_counts,
            "text_matches": text_matches,
            "generated_token_id_matches": generated_token_id_matches,
            "mismatched_fields": mismatched_fields,
        },
        "claim_boundary": {
            "dense_slm_evidence_used": false,
            "runtime_accuracy_claimed": false,
            "performance_claimed": false,
            "chat_enabled": false,
            "serve_enabled": false,
            "full_metal_inference_claimed": false,
            "qk256_apple_claimed": false,
            "neural_engine_claimed": false,
            "mpsgraph_claimed": false,
            "broad_apple_silicon_claimed": false,
        },
    })
}

fn quality_failure_taxonomy(
    failed_rules: &[String],
    scoring: Option<&ScoringResult>,
    answer: &str,
) -> Vec<String> {
    let mut taxonomy = BTreeSet::new();
    if failed_rules.iter().any(|rule| rule == "raw_special_tokens")
        || contains_raw_special_token(answer)
    {
        taxonomy.insert("raw_special_token_tail".to_string());
        taxonomy.insert("template_or_stop".to_string());
    }
    for rule in failed_rules {
        if matches!(rule.as_str(), "replacement_chars" | "mostly_text" | "printable_utf8") {
            taxonomy.insert("format_only".to_string());
        } else if rule == "empty_answer"
            || rule.starts_with("gate_")
            || matches!(rule.as_str(), "generated_token_min" | "generated_token_variation")
        {
            taxonomy.insert("answer_content".to_string());
        }
    }
    if let Some(scoring) = scoring {
        taxonomy.extend(scoring.failure_taxonomy.iter().cloned());
    }
    if !failed_rules.is_empty() && looks_like_refusal(answer) {
        taxonomy.insert("refusal".to_string());
    }
    taxonomy.into_iter().collect()
}

fn scoring_failure_taxonomy(
    answer: &str,
    scoring: &AnswerScoring,
    kind: &str,
    failed_rules: &[String],
) -> Vec<String> {
    if failed_rules.is_empty() {
        return Vec::new();
    }
    let mut taxonomy = BTreeSet::new();
    if contains_raw_special_token(answer) {
        taxonomy.insert("raw_special_token_tail".to_string());
        taxonomy.insert("template_or_stop".to_string());
    }
    if looks_like_refusal(answer) {
        taxonomy.insert("refusal".to_string());
    }
    match kind {
        "exact_match" => {
            let expected = scoring.expected_answer().unwrap_or_default();
            if normalize_match_text(answer) == normalize_match_text(expected) {
                taxonomy.insert("punctuation_casing_normalization".to_string());
            } else {
                taxonomy.insert("answer_content".to_string());
            }
        }
        "normalized_match"
        | "contains_expected"
        | "required_keywords"
        | "forbidden_tokens"
        | "required_forbidden_tokens" => {
            taxonomy.insert("answer_content".to_string());
        }
        "numeric_tolerance" => {
            if failed_rules.iter().any(|rule| {
                matches!(rule.as_str(), "numeric_observed_missing" | "numeric_expected_missing")
            }) {
                taxonomy.insert("format_only".to_string());
            } else {
                taxonomy.insert("answer_content".to_string());
            }
        }
        "json_schema" => {
            if contains_fenced_json(answer) {
                taxonomy.insert("fenced_json".to_string());
            }
            for rule in failed_rules {
                if rule == "json_parse"
                    || rule.starts_with("json_required_")
                    || rule.starts_with("json_additional_")
                    || rule.starts_with("json_type")
                {
                    taxonomy.insert("format_only".to_string());
                } else if rule.starts_with("json_const_") || rule.starts_with("json_enum_") {
                    taxonomy.insert("answer_content".to_string());
                }
            }
        }
        _ => {
            taxonomy.insert("answer_content".to_string());
        }
    }
    if taxonomy.is_empty() {
        taxonomy.insert("answer_content".to_string());
    }
    taxonomy.into_iter().collect()
}

fn failure_categories_for_case(
    task_family: &str,
    status: &str,
    failed_rules: &[String],
    failure_taxonomy: &[String],
    scoring_kind: Option<&str>,
) -> Vec<String> {
    let mut categories = BTreeSet::<String>::new();
    let has_failure =
        status == "timeout" || !failed_rules.is_empty() || !failure_taxonomy.is_empty();
    if !has_failure {
        return Vec::new();
    }

    let has_taxonomy = |label: &str| failure_taxonomy.iter().any(|value| value == label);
    let has_rule = |rule: &str| failed_rules.iter().any(|value| value == rule);

    if status == "timeout" || has_rule("timeout") || has_taxonomy("timeout") {
        categories.insert("timeout".to_string());
    }
    if has_taxonomy("refusal") {
        categories.insert("refusal".to_string());
    }
    if has_taxonomy("punctuation_casing_normalization") {
        categories.insert("normalization".to_string());
    }
    if has_taxonomy("raw_special_token_tail")
        || has_taxonomy("template_or_stop")
        || has_taxonomy("fenced_json")
        || has_taxonomy("format_only")
        || has_rule("replacement_chars")
        || has_rule("mostly_text")
        || has_rule("printable_utf8")
        || has_rule("raw_special_tokens")
    {
        categories.insert("formatting".to_string());
    }
    if scoring_kind == Some("json_schema")
        || failed_rules.iter().any(|rule| {
            rule == "json_parse"
                || rule == "schema_missing_or_invalid"
                || rule.starts_with("json_")
                || rule.starts_with("scoring_json_")
        })
    {
        categories.insert("schema".to_string());
    }

    let task_family = task_family.to_ascii_lowercase();
    if task_family.contains("fixed_table") || task_family.contains("factual") {
        categories.insert("factual_table".to_string());
    }
    if task_family.contains("extraction") {
        categories.insert("extraction".to_string());
    }

    categories.into_iter().collect()
}

fn failure_category_fields(categories: &[String]) -> Value {
    let set = categories.iter().map(String::as_str).collect::<BTreeSet<_>>();
    Value::Object(
        ANSWER_CORPUS_FAILURE_CATEGORIES
            .iter()
            .map(|category| ((*category).to_string(), Value::Bool(set.contains(*category))))
            .collect::<Map<_, _>>(),
    )
}

fn looks_like_refusal(answer: &str) -> bool {
    let normalized = normalize_match_text(answer);
    [
        "i cannot",
        "i can not",
        "i can't",
        "cannot answer",
        "can't answer",
        "unable to answer",
        "not able to answer",
        "sorry",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

fn contains_fenced_json(answer: &str) -> bool {
    let trimmed = answer.trim_start();
    trimmed.starts_with("```") || trimmed.contains("\n```")
}

fn normalize_scoring_text(value: &str) -> String {
    let stripped = strip_special_markers(value);
    let collapsed = stripped.split_whitespace().collect::<Vec<_>>().join(" ");
    strip_leading_assistant_separator(&collapsed).to_string()
}

fn normalize_match_text(value: &str) -> String {
    normalize_scoring_text(value)
        .chars()
        .map(
            |ch| {
                if ch.is_alphanumeric() || ch.is_whitespace() {
                    ch.to_ascii_lowercase()
                } else {
                    ' '
                }
            },
        )
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn compact_match_text(value: &str) -> String {
    value.chars().filter(|ch| ch.is_alphanumeric()).collect()
}

fn gate_evaluation_text<'a>(answer: &'a str, scoring: Option<&AnswerScoring>) -> Cow<'a, str> {
    if scoring.is_some_and(|scoring| scoring.kind() == "json_schema") {
        json_scoring_candidate(answer).text
    } else {
        Cow::Borrowed(answer)
    }
}

fn first_number(value: &str) -> Option<f64> {
    all_numbers(value).into_iter().next()
}

fn numeric_answer_candidates(value: &str) -> Vec<f64> {
    let normalized = normalize_scoring_text(value);
    for marker in ["answer is", "answer:", "result is", "equals", "=", "therefore", " is "] {
        if let Some(index) = normalized.to_ascii_lowercase().rfind(marker) {
            let start = index + marker.len();
            let candidates = all_numbers(&normalized[start..]);
            if !candidates.is_empty() {
                return candidates;
            }
        }
    }

    let candidates = all_numbers(&normalized);
    if normalized
        .trim_start()
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_digit() || matches!(ch, '-' | '+' | '.'))
        || candidates.len() <= 1
    {
        return candidates;
    }
    Vec::new()
}

fn all_numbers(value: &str) -> Vec<f64> {
    value
        .split(|ch: char| !(ch.is_ascii_digit() || matches!(ch, '.' | '-' | '+')))
        .filter(|token| !token.is_empty() && !matches!(*token, "." | "-" | "+" | "-." | "+."))
        .filter_map(parse_number_token)
        .collect()
}

fn parse_number_token(token: &str) -> Option<f64> {
    let trimmed = token.trim_matches(|ch| matches!(ch, '.' | '+' | '-'));
    let start = token.find(trimmed)?;
    let end = start + trimmed.len();
    token[..end].parse::<f64>().ok()
}

fn missing_keywords(answer: &str, keywords: Option<&[String]>) -> Vec<String> {
    keywords
        .unwrap_or_default()
        .iter()
        .filter(|keyword| !contains_keyword_boundary(answer, keyword))
        .cloned()
        .collect()
}

fn observed_forbidden_tokens(answer: &str, tokens: Option<&[String]>) -> Vec<String> {
    tokens
        .unwrap_or_default()
        .iter()
        .filter(|token| contains_keyword_boundary(answer, token))
        .cloned()
        .collect()
}

fn contains_keyword_boundary(answer: &str, keyword: &str) -> bool {
    let haystack = answer.to_ascii_lowercase();
    let needle = keyword.trim().to_ascii_lowercase();
    if needle.is_empty() {
        return false;
    }
    keyword_boundary_forms(&needle).iter().any(|needle| contains_keyword_form(&haystack, needle))
}

fn keyword_boundary_forms(needle: &str) -> Vec<String> {
    let mut forms = vec![needle.to_string()];
    if let Some(plural) = regular_plural_form(needle)
        && plural != needle
    {
        forms.push(plural);
    }
    forms
}

fn regular_plural_form(needle: &str) -> Option<String> {
    let last = needle.chars().next_back()?;
    if !last.is_ascii_alphabetic() || needle.ends_with('s') {
        return None;
    }
    let plural = if needle.ends_with('x')
        || needle.ends_with('z')
        || needle.ends_with("ch")
        || needle.ends_with("sh")
    {
        format!("{needle}es")
    } else if let Some(stem) = needle.strip_suffix('y') {
        let preceding = stem.chars().next_back();
        if preceding.is_some_and(|ch| {
            ch.is_ascii_alphabetic()
                && !matches!(ch.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u')
        }) {
            format!("{stem}ies")
        } else {
            format!("{needle}s")
        }
    } else {
        format!("{needle}s")
    };
    Some(plural)
}

fn contains_keyword_form(haystack: &str, needle: &str) -> bool {
    let needle_starts_alnum = needle.chars().next().is_some_and(char::is_alphanumeric);
    let needle_ends_alnum = needle.chars().next_back().is_some_and(char::is_alphanumeric);
    let mut search_from = 0usize;
    while let Some(relative_index) = haystack[search_from..].find(needle) {
        let start = search_from + relative_index;
        let end = start + needle.len();
        let before = haystack[..start].chars().next_back();
        let after = haystack[end..].chars().next();
        let left_ok = !needle_starts_alnum || before.is_none_or(|ch| !ch.is_alphanumeric());
        let right_ok = !needle_ends_alnum || after.is_none_or(|ch| !ch.is_alphanumeric());
        if left_ok && right_ok {
            return true;
        }
        search_from = haystack[start..]
            .chars()
            .next()
            .map(|ch| start + ch.len_utf8())
            .unwrap_or(haystack.len());
    }
    false
}

fn normalized_json_schema(schema: &Value) -> Option<Value> {
    match schema {
        Value::String(schema) => serde_json::from_str(schema).ok(),
        Value::Object(_) => Some(schema.clone()),
        _ => None,
    }
}

struct SchemaStyleResult {
    parsed: bool,
    source: &'static str,
    failures: Vec<String>,
}

fn validate_schema_style_json(answer: &str, schema: &Value) -> SchemaStyleResult {
    let candidate = json_scoring_candidate(answer);
    let Ok(value) = serde_json::from_str::<Value>(candidate.text.trim()) else {
        return SchemaStyleResult {
            parsed: false,
            source: candidate.source,
            failures: vec!["json_parse".to_string()],
        };
    };
    let mut failures = Vec::new();
    if schema["type"].as_str() == Some("object") && !value.is_object() {
        failures.push("json_type_object".to_string());
    }
    if let Some(required) = schema["required"].as_array()
        && let Some(object) = value.as_object()
    {
        for field in required.iter().filter_map(Value::as_str) {
            if !object.contains_key(field) {
                failures.push(format!("json_required_{field}"));
            }
        }
    }
    if schema["additionalProperties"].as_bool() == Some(false)
        && let (Some(object), Some(properties)) =
            (value.as_object(), schema["properties"].as_object())
    {
        for field in object.keys() {
            if !properties.contains_key(field) {
                failures.push(format!("json_additional_{field}"));
            }
        }
    }
    if let (Some(object), Some(properties)) = (value.as_object(), schema["properties"].as_object())
    {
        for (field, field_schema) in properties {
            let Some(observed) = object.get(field) else {
                continue;
            };
            if let Some(expected_const) = field_schema.get("const")
                && observed != expected_const
            {
                failures.push(format!("json_const_{field}"));
            }
            if let Some(expected_type) = field_schema["type"].as_str()
                && !json_type_matches(observed, expected_type)
            {
                failures.push(format!("json_type_{field}_{expected_type}"));
            }
            if let Some(enum_values) = field_schema["enum"].as_array()
                && !enum_values.iter().any(|candidate| candidate == observed)
            {
                failures.push(format!("json_enum_{field}"));
            }
        }
    }
    SchemaStyleResult { parsed: true, source: candidate.source, failures }
}

struct JsonScoringCandidate<'a> {
    text: Cow<'a, str>,
    source: &'static str,
}

fn json_scoring_candidate(answer: &str) -> JsonScoringCandidate<'_> {
    let trimmed = answer.trim();
    if serde_json::from_str::<Value>(trimmed).is_ok() {
        return JsonScoringCandidate { text: Cow::Borrowed(trimmed), source: "raw" };
    }
    if let Some(payload) = fenced_json_payload(trimmed) {
        return JsonScoringCandidate { text: Cow::Owned(payload), source: "fenced_json" };
    }
    if let Some(payload) = embedded_json_object(trimmed) {
        return JsonScoringCandidate { text: Cow::Owned(payload), source: "embedded_json" };
    }
    JsonScoringCandidate { text: Cow::Borrowed(trimmed), source: "raw" }
}

fn fenced_json_payload(answer: &str) -> Option<String> {
    let fence_start = answer.find("```")?;
    let after_open = &answer[fence_start + 3..];
    let after_payload_start = if let Some(index) = after_open.find('\n') {
        &after_open[index + 1..]
    } else {
        let trimmed = after_open.trim_start();
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            trimmed
        } else {
            let info_len = trimmed.find(char::is_whitespace)?;
            trimmed[info_len..].trim_start()
        }
    };
    let fence_end =
        after_payload_start.find("\n```").or_else(|| after_payload_start.find("```"))?;
    Some(after_payload_start[..fence_end].trim().to_string())
}

fn embedded_json_object(answer: &str) -> Option<String> {
    let start = answer.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (relative_index, ch) in answer[start..].char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let end = start + relative_index + ch.len_utf8();
                    return Some(answer[start..end].trim().to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn json_type_matches(value: &Value, expected_type: &str) -> bool {
    match expected_type {
        "array" => value.is_array(),
        "boolean" => value.is_boolean(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "object" => value.is_object(),
        "string" => value.is_string(),
        "null" => value.is_null(),
        _ => false,
    }
}

fn answer_receipt_failed_rules(run_receipt: &Value, expected_backend: &str) -> Vec<String> {
    let mut failed = Vec::new();
    let generated_text = run_receipt["text"].as_str().unwrap_or_default();
    if generated_text.is_empty() {
        failed.push("generated_text_recorded".to_string());
    }

    if !non_empty_str_at(run_receipt, &["model", "repo"]) {
        failed.push("model_repo_recorded".to_string());
    }
    if !non_empty_str_at(run_receipt, &["model", "file"]) {
        failed.push("model_file_recorded".to_string());
    }
    let model_sha256 = run_receipt["model"]["sha256"].as_str().unwrap_or_default();
    if !is_sha256_hex(model_sha256) {
        failed.push("model_sha256_recorded".to_string());
    }
    if !non_empty_str_at(run_receipt, &["model", "family"]) {
        failed.push("model_family_recorded".to_string());
    }
    if !non_empty_str_at(run_receipt, &["model", "architecture"]) {
        failed.push("model_architecture_recorded".to_string());
    }

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
        INTEL_A770_OPENCL => selected_backend == INTEL_A770_OPENCL,
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
    if truthy_bool_at_any(
        run_receipt,
        &[
            &["speedup_claim"][..],
            &["claim_boundary", "speedup_claim"][..],
            &["claim_boundary", "full_metal_inference_claimed"][..],
            &["claim_boundary", "full_a770_residency_claimed"][..],
            &["claim_boundary", "trusted_partial_acceleration_claimed"][..],
            &["claim_boundary", "a770_speedup_claimed"][..],
            &["claim_boundary", "broad_performance_claimed"][..],
        ],
    ) {
        failed.push("speedup_claim_false".to_string());
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
    let pretokenizer_authority =
        run_receipt["tokenizer"]["pretokenizer_authority"].as_str().unwrap_or_default();
    if pretokenizer_authority.is_empty()
        || matches!(pretokenizer_authority, "unknown" | "defaulted")
    {
        failed.push("tokenizer_pretokenizer_authority_recorded".to_string());
    }

    let selected_kernel = run_receipt["kernel"]["kernel_id"].as_str().unwrap_or_default();
    if selected_kernel.is_empty() {
        failed.push("selected_kernel_recorded".to_string());
    }
    if selected_kernel.contains("mock") || selected_kernel.contains("diagnostic") {
        failed.push("selected_kernel_production".to_string());
    }
    let model_family = run_receipt["model"]["family"].as_str().unwrap_or_default();
    if model_family == "qwen" {
        if run_receipt.get("dense_slm").is_none() {
            failed.push("dense_slm_provenance_recorded".to_string());
        }
        if run_receipt.get("bitnet").is_some() {
            failed.push("dense_slm_no_bitnet_provenance".to_string());
        }
        if contains_bitnet_dense_forbidden(selected_kernel) {
            failed.push("dense_slm_kernel_not_bitnet_qk256".to_string());
        }
        let kernel_layout = run_receipt["kernel"]["layout"].as_str().unwrap_or_default();
        if contains_bitnet_dense_forbidden(kernel_layout) {
            failed.push("dense_slm_layout_not_bitnet_qk256".to_string());
        }
        let kernel_family = run_receipt["kernel"]["family"].as_str().unwrap_or_default();
        if contains_bitnet_dense_forbidden(kernel_family) {
            failed.push("dense_slm_kernel_family_not_bitnet_qk256".to_string());
        }
        if run_receipt.get("dense_slm").is_some_and(json_contains_bitnet_dense_forbidden) {
            failed.push("dense_slm_fields_not_bitnet_qk256".to_string());
        }
        if run_receipt.get("strict_provenance").is_some_and(json_contains_bitnet_dense_forbidden) {
            failed.push("dense_slm_strict_provenance_not_bitnet_qk256".to_string());
        }
        if run_receipt["execution_coverage"]
            .as_object()
            .is_some_and(|coverage| coverage.keys().any(|key| key.starts_with("bitnet_")))
        {
            failed.push("dense_slm_execution_coverage_not_bitnet_qk256".to_string());
        }
    } else if model_family == "bitnet" && expected_backend == "cpu" {
        failed.extend(qk256_hot_path_failed_rules(run_receipt));
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
        failed.extend(
            planner_receipts::strict_bitnet_qk256_execution_plan_failed_rules(
                &run_receipt["execution_plan"],
            )
            .into_iter()
            .map(str::to_string),
        );
    }
    if is_a770_opencl_answer_corpus_device(expected_backend) {
        let proof_summary = &run_receipt["proof_summary"];
        if proof_summary["model_contract_declared"].as_bool() != Some(true) {
            failed.push("a770_model_contract_declared".to_string());
        }
        let model_contract = proof_summary["model_contract"].as_str().unwrap_or_default();
        if !model_contract.ends_with(A770_BITNET_MODEL_CONTRACT) {
            failed.push("a770_model_contract_path".to_string());
        }
        if proof_summary["route_declared"].as_bool() != Some(true) {
            failed.push("a770_kernel_route_declared".to_string());
        }
        let kernel_route = &proof_summary["kernel_route"];
        if kernel_route["route_id"].as_str() != Some(A770_BITNET_QK256_ROUTE_ID) {
            failed.push("a770_kernel_route_id".to_string());
        }
        if kernel_route["diagnostic_only"].as_bool() != Some(true) {
            failed.push("a770_kernel_route_diagnostic_only".to_string());
        }
        if kernel_route["claimable"].as_bool() != Some(false) {
            failed.push("a770_kernel_route_claimable_false".to_string());
        }
        if proof_summary["backend_claimable"].as_bool() != Some(false) {
            failed.push("a770_backend_claimable_false".to_string());
        }
    }

    let prompt_count = u64_at(run_receipt, &["tokens", "prompt"]);
    let generated_count = u64_at(run_receipt, &["tokens", "generated"]);
    let total_count = u64_at(run_receipt, &["tokens", "total"]);
    let prompt_ids = run_receipt["tokens"]["prompt_ids"].as_array();
    let generated_ids = run_receipt["tokens"]["generated_ids"].as_array();
    if prompt_count.is_none() {
        failed.push("prompt_token_count_recorded".to_string());
    }
    if generated_count.is_none() {
        failed.push("generated_token_count_recorded".to_string());
    }
    if total_count.is_none() {
        failed.push("total_token_count_recorded".to_string());
    }
    if prompt_ids.is_none() {
        failed.push("prompt_token_ids_recorded".to_string());
    }
    if generated_ids.is_none_or(Vec::is_empty) {
        failed.push("generated_token_ids_recorded".to_string());
    }
    if let (Some(count), Some(ids)) = (prompt_count, prompt_ids)
        && count != ids.len() as u64
    {
        failed.push("prompt_token_count_matches_ids".to_string());
    }
    if let (Some(count), Some(ids)) = (generated_count, generated_ids)
        && count != ids.len() as u64
    {
        failed.push("generated_token_count_matches_ids".to_string());
    }
    if let (Some(total), Some(prompt), Some(generated)) =
        (total_count, prompt_count, generated_count)
        && total != prompt + generated
    {
        failed.push("total_token_count_matches".to_string());
    }

    for (path, rule) in [
        (&["timing", "model_load_ms"][..], "timing_model_load_ms_recorded"),
        (&["timing", "tokenizer_load_ms"][..], "timing_tokenizer_load_ms_recorded"),
        (&["timing", "tokenize_ms"][..], "timing_tokenize_ms_recorded"),
        (&["timing", "prefill_ms"][..], "timing_prefill_ms_recorded"),
        (&["timing", "first_token_ms"][..], "timing_first_token_ms_recorded"),
        (&["timing", "decode_total_ms"][..], "timing_decode_total_ms_recorded"),
        (&["latency", "total_ms"][..], "latency_total_ms_recorded"),
    ] {
        if !number_at(run_receipt, path) {
            failed.push(rule.to_string());
        }
    }
    failed
}

fn contains_bitnet_dense_forbidden(value: &str) -> bool {
    let normalized = value.to_ascii_lowercase();
    normalized.contains("i2_s") || normalized.contains("qk256") || normalized.contains("bitnet")
}

fn json_contains_bitnet_dense_forbidden(value: &Value) -> bool {
    match value {
        Value::String(value) => contains_bitnet_dense_forbidden(value),
        Value::Array(values) => values.iter().any(json_contains_bitnet_dense_forbidden),
        Value::Object(values) => values.iter().any(|(key, value)| {
            contains_bitnet_dense_forbidden(key) || json_contains_bitnet_dense_forbidden(value)
        }),
        _ => false,
    }
}

fn qk256_hot_path_failed_rules(run_receipt: &Value) -> Vec<String> {
    let mut failed = Vec::new();
    let hot_path = &run_receipt["qk256_hot_path"];
    if !hot_path.is_object() {
        failed.push("qk256_hot_path_recorded".to_string());
        return failed;
    }

    for field in [
        "qk256_f32_scalar_gemv_invocations",
        "qk256_f32_avx2_gemv_invocations",
        "qk256_i8s_scaled_scalar_invocations",
        "qk256_i8s_scaled_avx2_invocations",
        "qk256_flat_bytes_extracted_count",
        "input_rows_materialized_count",
        "output_rows_allocated_count",
        "no_scale_f32_gemv_invocations",
        "scaled_i2s_i8s_gemv_invocations",
        "audited_tensor_materialization_count",
    ] {
        if hot_path[field].as_u64().is_none() {
            failed.push(format!("qk256_hot_path_{field}_recorded"));
        }
    }

    let total_invocations = hot_path["no_scale_f32_gemv_invocations"]
        .as_u64()
        .unwrap_or(0)
        .saturating_add(hot_path["scaled_i2s_i8s_gemv_invocations"].as_u64().unwrap_or(0));
    if total_invocations == 0 {
        failed.push("qk256_hot_path_invocations_positive".to_string());
    }

    if hot_path["audited_tensor_materialization_count"].as_u64().unwrap_or(0) == 0 {
        failed.push("qk256_hot_path_materialization_audited".to_string());
    }

    if hot_path["qk256_execution_path"].as_str().unwrap_or_default().is_empty() {
        failed.push("qk256_hot_path_execution_path_recorded".to_string());
    }
    if hot_path["selected_kernel"].as_str().unwrap_or_default().is_empty() {
        failed.push("qk256_hot_path_selected_kernel_recorded".to_string());
    }
    if truthy_bool_at_any(hot_path, &[&["speedup_claim"][..], &["math_changed"][..]]) {
        failed.push("qk256_hot_path_no_speedup_or_math_change_claim".to_string());
    }

    failed
}

fn answer_receipt_required_case_fields() -> &'static [&'static str] {
    ANSWER_RECEIPT_REQUIRED_CASE_FIELDS
}

fn answer_receipt_checked_rules() -> &'static [&'static str] {
    ANSWER_RECEIPT_CHECKED_RULES
}

fn non_empty_str_at(value: &Value, path: &[&str]) -> bool {
    value_at(value, path).and_then(Value::as_str).is_some_and(|text| !text.is_empty())
}

fn number_at(value: &Value, path: &[&str]) -> bool {
    value_at(value, path).is_some_and(Value::is_number)
}

fn u64_at(value: &Value, path: &[&str]) -> Option<u64> {
    value_at(value, path).and_then(Value::as_u64)
}

fn truthy_bool_at_any(value: &Value, paths: &[&[&str]]) -> bool {
    paths.iter().any(|path| value_at(value, path).and_then(Value::as_bool) == Some(true))
}

fn value_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn aggregate_execution_plan(rows: &[Value], device: &str) -> Value {
    if !is_cuda_answer_corpus_device(device) && !is_a770_opencl_answer_corpus_device(device) {
        return Value::Null;
    }

    let mut plan = rows
        .iter()
        .find_map(|row| row.get("execution_plan").filter(|plan| plan.is_object()).cloned())
        .unwrap_or(Value::Null);
    if let Some(plan) = plan.as_object_mut() {
        plan.insert("scope".to_string(), Value::from("answer_corpus_aggregate"));
        plan.insert("case_count".to_string(), Value::from(rows.len() as u64));
    }
    plan
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
    let mut cleaned = strip_leading_chatml_assistant(answer.trim_start()).to_string();
    cleaned = cleaned.replace("<|begin_of_text|>", "");
    if let Some(index) = earliest_known_stop_marker(&cleaned) {
        cleaned.truncate(index);
    }
    cleaned
}

fn strip_leading_chatml_assistant(answer: &str) -> &str {
    let mut rest = answer;
    loop {
        let Some(after_marker) = rest.strip_prefix("<|im_start|>assistant") else {
            return rest;
        };
        rest = after_marker
            .strip_prefix("\r\n")
            .or_else(|| after_marker.strip_prefix('\n'))
            .or_else(|| after_marker.strip_prefix(' '))
            .unwrap_or(after_marker)
            .trim_start();
    }
}

fn earliest_known_stop_marker(answer: &str) -> Option<usize> {
    ["<|im_end|>", "<|end_of_text|>", "<|endoftext|>", "<|eot_id|>", "<|im_start|>"]
        .iter()
        .filter_map(|marker| answer.find(marker))
        .min()
}

fn strip_leading_assistant_separator(answer: &str) -> &str {
    if let Some(after_colon) = answer.strip_prefix(':')
        && after_colon.starts_with(char::is_whitespace)
    {
        after_colon.trim_start()
    } else {
        answer
    }
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
    let failed_rules = vec![input.failed_rule.to_string()];
    let failure_taxonomy = vec![input.failed_rule.to_string()];
    let failure_category_labels = failure_categories_for_case(
        input.case.task_family(),
        input.status,
        &failed_rules,
        &failure_taxonomy,
        input.case.scoring.as_ref().map(AnswerScoring::kind),
    );
    let failure_categories = failure_category_fields(&failure_category_labels);
    json!({
        "id": input.case.id,
        "task_family": input.case.task_family(),
        "category": input.case.category.as_deref().unwrap_or_else(|| input.case.task_family()),
        "profile": input.case.profile(),
        "seed_material": input.case.seed_material,
        "question": input.case.question,
        "status": input.status,
        "exit_code": input.run.exit_code,
        "timeout_seconds": input.timeout_seconds,
        "run_receipt_path": input.case_receipt.display().to_string(),
        "quality": {
            "passed": false,
            "failed_rules": failed_rules,
            "failure_taxonomy": failure_taxonomy,
            "failure_category_labels": failure_category_labels,
            "failure_categories": failure_categories,
            "scoring": scoring_not_run_json(input.case.scoring.as_ref()),
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
        "tokens": Value::Null,
        "token_ids": {
            "prompt": Value::Null,
            "generated": Value::Null,
        },
        "reference_comparison": reference_comparison_json(
            input.case,
            None,
            None,
            &Value::from(input.device),
            &Value::from(answer_corpus_runtime_api(input.device)),
            &Value::from(false),
        ),
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

    fn scoring(kind: &str) -> AnswerScoring {
        AnswerScoring {
            kind: Some(kind.to_string()),
            expected: None,
            expected_normalized: None,
            schema: None,
            expected_number: None,
            numeric_tolerance: None,
            required_keywords: None,
            forbidden_tokens: None,
            expected_answer_authority: None,
        }
    }

    #[test]
    fn exact_gate_accepts_trimmed_answer() {
        let gate = AnswerGate { expected: Some("4".to_string()), ..gate("exact_trimmed") };
        let quality = evaluate_quality(" 4\n", &gate, None, None, None, None);
        assert!(quality.passed);
    }

    #[test]
    fn quality_rejects_raw_special_tokens() {
        let quality = evaluate_quality(
            "<|start_header_id|>assistant",
            &gate("readable"),
            None,
            None,
            None,
            None,
        );
        assert!(!quality.passed);
        assert!(quality.failed_rules.contains(&"raw_special_tokens".to_string()));
        assert!(quality.failure_taxonomy.contains(&"raw_special_token_tail".to_string()));
        assert!(quality.failure_taxonomy.contains(&"template_or_stop".to_string()));
    }

    #[test]
    fn quality_treats_qwen_im_end_as_stop_marker() {
        let gate = AnswerGate { expected: Some("4".to_string()), ..gate("exact_trimmed") };
        let quality =
            evaluate_quality("\n4<|im_end|>", &gate, None, Some(&[198, 19, 151645]), None, None);

        assert!(quality.passed);
        assert!(!quality.failed_rules.contains(&"raw_special_tokens".to_string()));
    }

    #[test]
    fn slm_eval_scoring_quality_treats_qwen_im_start_tail_as_stop_marker() {
        let gate = AnswerGate { expected: Some("4".to_string()), ..gate("exact_trimmed") };
        let scoring =
            AnswerScoring { expected_normalized: Some("4".to_string()), ..scoring("exact_match") };
        let quality = evaluate_quality(
            "4<|im_start|>user\nignored",
            &gate,
            Some(&scoring),
            Some(&[19, 151644, 872]),
            None,
            None,
        );

        assert!(quality.passed);
        assert_eq!(
            quality.scoring.as_ref().map(|scoring| scoring.normalized_answer.as_str()),
            Some("4")
        );
        assert!(!quality.failed_rules.contains(&"raw_special_tokens".to_string()));
    }

    #[test]
    fn slm_eval_scoring_quality_accepts_qwen_assistant_prefix_separator() {
        let gate = AnswerGate { expected: Some("4".to_string()), ..gate("exact_trimmed") };
        let scoring =
            AnswerScoring { expected_normalized: Some("4".to_string()), ..scoring("exact_match") };
        let quality = evaluate_quality(
            ": 4<|im_end|>",
            &gate,
            Some(&scoring),
            Some(&[25, 220, 19, 151645]),
            None,
            None,
        );

        assert!(quality.passed);
        assert_eq!(
            quality.scoring.as_ref().map(|scoring| scoring.normalized_answer.as_str()),
            Some("4")
        );
    }

    #[test]
    fn answer_corpus_quality_normalizes_leading_assistant_colon_for_starts_with() {
        let gate = AnswerGate {
            starts_with_any: Some(vec!["yes".to_string()]),
            ..gate("starts_with_any")
        };
        let quality = evaluate_quality(
            ": Yes, the sky is usually blue.<|im_end|>",
            &gate,
            None,
            Some(&[25, 9454, 11, 279, 12765]),
            None,
            None,
        );

        assert!(quality.passed);
        assert!(!quality.failed_rules.contains(&"gate_starts_with_any".to_string()));
    }

    #[test]
    fn answer_corpus_scoring_normalizes_leading_assistant_colon_for_normalized_match() {
        let scoring = AnswerScoring {
            expected_normalized: Some("done".to_string()),
            ..scoring("normalized_match")
        };
        let result = evaluate_scoring(": Done<|im_end|>", &scoring);

        assert!(result.passed);
    }

    #[test]
    fn quality_rejects_punctuation_noise() {
        let quality = evaluate_quality("!!!,,,!!!", &gate("readable"), None, None, None, None);
        assert!(!quality.passed);
        assert!(quality.failed_rules.contains(&"mostly_text".to_string()));
    }

    #[test]
    fn apple_m4_quality_rejects_short_or_degenerate_token_output() {
        let gate = AnswerGate { min_words: Some(2), ..gate("readable") };
        let quality =
            evaluate_quality("short answer", &gate, None, Some(&[7, 7, 7]), Some(4), Some(2));
        assert!(!quality.passed);
        assert!(quality.failed_rules.contains(&"generated_token_min".to_string()));
        assert!(quality.failed_rules.contains(&"generated_token_variation".to_string()));
        assert_eq!(quality.distinct_generated_tokens, 1);
    }

    #[test]
    fn exact_scored_single_token_answer_does_not_need_variation() {
        let gate = AnswerGate {
            starts_with_any: Some(vec!["yes".to_string()]),
            ..gate("starts_with_any")
        };
        let scoring = AnswerScoring {
            expected_normalized: Some("yes".to_string()),
            ..scoring("normalized_match")
        };
        let quality =
            evaluate_quality("Yes", &gate, Some(&scoring), Some(&[9454]), Some(1), Some(2));

        assert!(quality.passed, "{:?}", quality.failed_rules);
        assert_eq!(quality.distinct_generated_tokens, 1);
    }

    #[test]
    fn slm_eval_scoring_exact_and_normalized_match_are_deterministic() {
        let exact =
            AnswerScoring { expected_normalized: Some("15".to_string()), ..scoring("exact_match") };
        assert!(evaluate_scoring("15", &exact).passed);
        let punctuation_only = evaluate_scoring("15.", &exact);
        assert!(!punctuation_only.passed);
        assert!(
            punctuation_only
                .failure_taxonomy
                .contains(&"punctuation_casing_normalization".to_string())
        );

        let normalized = AnswerScoring {
            expected_normalized: Some("ant, cat, dog".to_string()),
            ..scoring("normalized_match")
        };
        assert!(evaluate_scoring(" Ant,   cat, dog. ", &normalized).passed);
    }

    #[test]
    fn slm_eval_scoring_validates_json_schema_style_outputs() {
        let scoring = AnswerScoring {
            schema: Some(json!({
                "type": "object",
                "required": ["status"],
                "additionalProperties": false,
                "properties": {
                    "status": { "const": "ready", "type": "string" }
                }
            })),
            ..scoring("json_schema")
        };

        assert!(evaluate_scoring(r#"{"status":"ready"}"#, &scoring).passed);
        let result = evaluate_scoring(r#"{"status":"ready","extra":true}"#, &scoring);
        assert!(!result.passed);
        assert!(result.failed_rules.contains(&"json_additional_extra".to_string()));
        assert!(result.failure_taxonomy.contains(&"format_only".to_string()));

        let fenced = evaluate_scoring("```json\n{\"status\":\"ready\"}\n```", &scoring);
        assert!(fenced.passed);
        assert_eq!(fenced.details["json_source"], "fenced_json");

        let embedded = evaluate_scoring("Here is the JSON: {\"status\":\"ready\"}", &scoring);
        assert!(embedded.passed);
        assert_eq!(embedded.details["json_source"], "embedded_json");

        let malformed_fenced = evaluate_scoring("```json\n{\"status\":\"ready\"\n```", &scoring);
        assert!(!malformed_fenced.passed);
        assert!(malformed_fenced.failed_rules.contains(&"json_parse".to_string()));
        assert!(malformed_fenced.failure_taxonomy.contains(&"fenced_json".to_string()));
        assert!(malformed_fenced.failure_taxonomy.contains(&"format_only".to_string()));
    }

    #[test]
    fn slm_eval_failure_categories_cover_mechanical_triage_fields() {
        let exact =
            AnswerScoring { expected_normalized: Some("15".to_string()), ..scoring("exact_match") };
        let punctuation_only = evaluate_scoring("15.", &exact);
        assert!(!punctuation_only.passed);
        assert!(punctuation_only.failure_categories.contains(&"normalization".to_string()));

        let schema = AnswerScoring {
            schema: Some(json!({
                "type": "object",
                "required": ["status"],
                "additionalProperties": false,
                "properties": {
                    "status": { "const": "ready", "type": "string" }
                }
            })),
            ..scoring("json_schema")
        };
        let schema_result = evaluate_scoring(r#"{"status":"ready","extra":true}"#, &schema);
        assert!(schema_result.failure_categories.contains(&"schema".to_string()));
        assert!(schema_result.failure_categories.contains(&"formatting".to_string()));

        let refused = evaluate_quality(
            "Sorry, I cannot answer that.",
            &AnswerGate { expected: Some("Paris".to_string()), ..gate("exact_trimmed") },
            None,
            Some(&[1, 2, 3]),
            None,
            None,
        );
        let refusal_categories = failure_categories_for_case(
            "fixed_table_qa",
            "quality_failed",
            &refused.failed_rules,
            &refused.failure_taxonomy,
            None,
        );
        assert!(refusal_categories.contains(&"refusal".to_string()));
        assert!(refusal_categories.contains(&"factual_table".to_string()));

        let timeout_rule = vec!["timeout".to_string()];
        let timeout_categories = failure_categories_for_case(
            "synthetic_extraction",
            "timeout",
            &timeout_rule,
            &timeout_rule,
            Some("exact_match"),
        );
        assert!(timeout_categories.contains(&"timeout".to_string()));
        assert!(timeout_categories.contains(&"extraction".to_string()));

        let fields = failure_category_fields(&timeout_categories);
        assert_eq!(fields["timeout"], true);
        assert_eq!(fields["extraction"], true);
        assert_eq!(fields["schema"], false);
    }

    #[test]
    fn slm_eval_scoring_supports_numeric_tolerance() {
        let scoring = AnswerScoring {
            expected_number: Some(std::f64::consts::PI),
            numeric_tolerance: Some(0.01),
            ..scoring("numeric_tolerance")
        };

        assert!(evaluate_scoring("approximately 3.141", &scoring).passed);
        assert!(evaluate_scoring("The answer is approximately 3.141.", &scoring).passed);
        assert!(!evaluate_scoring("To find 3.141 percent of 20", &scoring).passed);
        let result = evaluate_scoring("3.20", &scoring);
        assert!(!result.passed);
        assert!(result.failed_rules.contains(&"numeric_tolerance".to_string()));
        assert!(result.failure_taxonomy.contains(&"answer_content".to_string()));
    }

    #[test]
    fn bitnet_250_repair_scoring_normalizes_table_and_rewrite_answers() {
        let table = AnswerScoring {
            expected_normalized: Some("cyd".to_string()),
            expected_answer_authority: Some(json!({
                "source": "closed_form_yaml_fixture",
                "case_family": "fixed_table_qa"
            })),
            ..scoring("contains_expected")
        };
        let table_result = evaluate_scoring("Cyd has green.", &table);
        assert!(table_result.passed);
        assert_eq!(
            table_result.details["expected_answer_authority"]["source"],
            "closed_form_yaml_fixture"
        );

        let rewrite = AnswerScoring {
            expected_normalized: Some("local fast verified".to_string()),
            ..scoring("normalized_match")
        };
        assert!(evaluate_scoring("local, fast, verified", &rewrite).passed);

        let compact_rewrite = AnswerScoring {
            expected_normalized: Some("receipt has tokens".to_string()),
            ..scoring("normalized_match")
        };
        assert!(evaluate_scoring("receipthastokens", &compact_rewrite).passed);
    }

    #[test]
    fn slm_eval_scoring_required_and_forbidden_tokens_affect_quality() {
        let scoring = AnswerScoring {
            required_keywords: Some(vec!["ready".to_string()]),
            forbidden_tokens: Some(vec!["maybe".to_string()]),
            ..scoring("required_forbidden_tokens")
        };
        let gate =
            AnswerGate { contains_any: Some(vec!["maybe".to_string()]), ..gate("contains_any") };
        let quality = evaluate_quality("maybe later", &gate, Some(&scoring), None, None, None);

        assert!(!quality.passed);
        assert!(quality.failed_rules.contains(&"scoring_required_keywords".to_string()));
        assert!(quality.failed_rules.contains(&"scoring_forbidden_tokens".to_string()));
        assert!(quality.failure_taxonomy.contains(&"answer_content".to_string()));
        assert_eq!(
            quality.scoring.as_ref().map(|result| result.kind.as_str()),
            Some("required_forbidden_tokens")
        );
    }

    #[test]
    fn slm_eval_scoring_keyword_checks_use_boundaries() {
        let required = AnswerScoring {
            required_keywords: Some(vec!["red".to_string()]),
            ..scoring("required_keywords")
        };
        let missing = evaluate_scoring("ready", &required);
        assert!(!missing.passed);
        assert!(missing.failed_rules.contains(&"required_keywords".to_string()));

        let forbidden = AnswerScoring {
            forbidden_tokens: Some(vec!["red".to_string()]),
            ..scoring("forbidden_tokens")
        };
        let clean = evaluate_scoring("ready", &forbidden);
        assert!(clean.passed);
        let observed = evaluate_scoring("red alert", &forbidden);
        assert!(!observed.passed);
        assert!(observed.failed_rules.contains(&"forbidden_tokens".to_string()));

        let phrase = AnswerScoring {
            required_keywords: Some(vec!["model cache".to_string()]),
            ..scoring("required_keywords")
        };
        assert!(evaluate_scoring("The model cache is ready.", &phrase).passed);

        let plural_required = AnswerScoring {
            required_keywords: Some(vec!["fallback".to_string()]),
            ..scoring("required_keywords")
        };
        assert!(evaluate_scoring("Check local route fallbacks.", &plural_required).passed);

        let plural_forbidden = AnswerScoring {
            forbidden_tokens: Some(vec!["warning".to_string()]),
            ..scoring("forbidden_tokens")
        };
        let observed_plural = evaluate_scoring("Warnings were emitted.", &plural_forbidden);
        assert!(!observed_plural.passed);
        assert!(observed_plural.failed_rules.contains(&"forbidden_tokens".to_string()));
    }

    #[test]
    fn slm_eval_scoring_summary_counts_failure_taxonomy() {
        let rows = vec![
            json!({
                "quality": {
                    "scoring": {
                        "kind": "exact_match",
                        "passed": false,
                        "failure_taxonomy": ["punctuation_casing_normalization"]
                    }
                }
            }),
            json!({
                "quality": {
                    "scoring": {
                        "kind": "json_schema",
                        "passed": false,
                        "failure_taxonomy": ["fenced_json", "format_only"]
                    }
                }
            }),
        ];

        let summary = scoring_summary(&rows);
        assert_eq!(summary["failed"], 2);
        assert_eq!(summary["failure_taxonomy"]["punctuation_casing_normalization"], 1);
        assert_eq!(summary["failure_taxonomy"]["fenced_json"], 1);
        assert_eq!(summary["failure_taxonomy"]["format_only"], 1);
        assert_eq!(summary["failure_categories"]["normalization"], 1);
        assert_eq!(summary["failure_categories"]["schema"], 1);
        assert_eq!(summary["failure_categories"]["formatting"], 1);
    }

    #[test]
    fn bitnet_reference_comparison_records_matched_text_and_tokens() {
        let case = AnswerCase {
            id: "math_2_plus_2".to_string(),
            category: Some("arithmetic_exact".to_string()),
            profile: Some("regression_tiny".to_string()),
            seed_material: Some("seed=912587 family=arithmetic".to_string()),
            question: "What is 2+2?".to_string(),
            max_new_tokens: Some(4),
            timeout_seconds: None,
            min_generated_tokens: None,
            min_distinct_generated_tokens: None,
            reference: Some(AnswerReference {
                runner: Some("microsoft_bitnet_cpp".to_string()),
                source: Some("fixture".to_string()),
                receipt: Some("reference.json".to_string()),
                text: Some("4".to_string()),
                generated_token_ids: Some(vec![19]),
            }),
            scoring: Some(scoring("exact_match")),
            gate: gate("exact_trimmed"),
        };
        let selected_backend = Value::from("apple-m4-cpu-neon");
        let runtime_api = Value::from("cpu");
        let fallback_used = Value::from(false);
        let comparison = reference_comparison_json(
            &case,
            Some("4"),
            Some(&[19]),
            &selected_backend,
            &runtime_api,
            &fallback_used,
        );

        assert_eq!(comparison["schema"], "bitnet_reference_vs_rust_v1");
        assert_eq!(comparison["enabled"], true);
        assert_eq!(comparison["reference"]["runner"], "microsoft_bitnet_cpp");
        assert_eq!(comparison["rust"]["selected_backend"], "apple-m4-cpu-neon");
        assert_eq!(comparison["rust"]["fallback_used"], false);
        assert_eq!(comparison["comparison"]["status"], "matched");
        assert_eq!(comparison["comparison"]["text_match"], true);
        assert_eq!(comparison["comparison"]["generated_token_ids_match"], true);
    }

    #[test]
    fn bitnet_task_family_summary_counts_statuses_and_failure_taxonomy() {
        let rows = vec![
            json!({
                "task_family": "arithmetic_exact",
                "status": "passed",
                "quality": {
                    "failure_taxonomy": [],
                    "scoring": {
                        "kind": "exact_match",
                        "passed": true,
                        "failure_taxonomy": []
                    }
                }
            }),
            json!({
                "task_family": "arithmetic_exact",
                "status": "timeout",
                "quality": {
                    "failure_taxonomy": ["timeout"],
                    "failure_category_labels": ["timeout"],
                    "failure_categories": { "timeout": true },
                    "scoring": {
                        "kind": "exact_match",
                        "passed": null,
                        "failure_taxonomy": []
                    }
                }
            }),
            json!({
                "task_family": "format_constrained_json",
                "status": "quality_failed",
                "quality": {
                    "failure_taxonomy": ["format_only"],
                    "failure_category_labels": ["formatting", "schema"],
                    "failure_categories": { "formatting": true, "schema": true },
                    "scoring": {
                        "kind": "json_schema",
                        "passed": false,
                        "failure_taxonomy": ["fenced_json"],
                        "failure_category_labels": ["formatting", "schema"],
                        "failure_categories": { "formatting": true, "schema": true }
                    }
                }
            }),
            json!({
                "task_family": "numeric_tolerance",
                "status": "quality_failed",
                "quality": {
                    "failure_taxonomy": ["answer_content"],
                    "scoring": {
                        "kind": "numeric_tolerance",
                        "passed": false,
                        "failure_taxonomy": ["answer_content"]
                    }
                }
            }),
        ];

        let summary = task_family_summary(&rows);
        assert_eq!(summary["arithmetic_exact"]["total"], 2);
        assert_eq!(summary["arithmetic_exact"]["passed"], 1);
        assert_eq!(summary["arithmetic_exact"]["timeout"], 1);
        assert_eq!(summary["arithmetic_exact"]["scoring"]["not_run"], 1);
        assert_eq!(summary["arithmetic_exact"]["failure_taxonomy"]["timeout"], 1);
        assert_eq!(summary["format_constrained_json"]["failed"], 1);
        assert_eq!(summary["format_constrained_json"]["failure_taxonomy"]["fenced_json"], 1);
        assert_eq!(summary["format_constrained_json"]["failure_taxonomy"]["format_only"], 1);
        assert_eq!(summary["format_constrained_json"]["failure_categories"]["formatting"], 1);
        assert_eq!(summary["format_constrained_json"]["failure_categories"]["schema"], 1);
        assert_eq!(summary["numeric_tolerance"]["failed"], 1);
        assert_eq!(summary["numeric_tolerance"]["failure_taxonomy"]["answer_content"], 1);
    }

    #[test]
    fn slm_answer_profile_summary_counts_profile_statuses() {
        let rows = vec![
            json!({
                "profile": "regression_tiny",
                "status": "passed",
                "quality": {
                    "failure_taxonomy": [],
                    "scoring": {
                        "kind": "required_keywords",
                        "passed": true,
                        "failure_taxonomy": []
                    }
                }
            }),
            json!({
                "profile": "regression_tiny",
                "status": "quality_failed",
                "quality": {
                    "failure_taxonomy": ["answer_content"],
                    "scoring": {
                        "kind": "required_keywords",
                        "passed": false,
                        "failure_taxonomy": ["answer_content"]
                    }
                }
            }),
            json!({
                "profile": "ask_normal",
                "status": "timeout",
                "quality": {
                    "failure_taxonomy": ["timeout"],
                    "scoring": {
                        "kind": "required_keywords",
                        "passed": null,
                        "failure_taxonomy": []
                    }
                }
            }),
        ];

        let summary = profile_summary(&rows);
        assert_eq!(summary["regression_tiny"]["total"], 2);
        assert_eq!(summary["regression_tiny"]["passed"], 1);
        assert_eq!(summary["regression_tiny"]["failed"], 1);
        assert_eq!(summary["regression_tiny"]["failure_taxonomy"]["answer_content"], 1);
        assert_eq!(summary["ask_normal"]["timeout"], 1);
        assert_eq!(summary["ask_normal"]["scoring"]["not_run"], 1);
    }

    #[test]
    fn cli_timeout_overrides_corpus_default() {
        assert_eq!(effective_default_timeout_seconds(Some(1), Some(300)), 1);
        assert_eq!(effective_default_timeout_seconds(None, Some(120)), 120);
        assert_eq!(effective_default_timeout_seconds(None, None), 300);
        assert_eq!(effective_default_timeout_seconds(Some(0), Some(300)), 1);
    }

    #[test]
    fn qwen_no_think_defaults_to_false_and_can_be_set() {
        let corpus = serde_yaml::from_str::<AnswerCorpus>(
            r#"
schema: 1
artifact_kind: slm_answer_corpus
name: qwen-no-think-fixture
description: qwen no-thinking fixture
model:
  repo: Qwen/Qwen3-0.6B-GGUF
  file: Qwen3-0.6B-Q8_0.gguf
  sha256: 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
  family: qwen
defaults:
  prompt_template: qwen
  qwen_no_think: true
  max_new_tokens: 1
  greedy: true
  deterministic: true
  strict_loader: true
  temperature: 0.0
cases:
  - id: say_ok
    question: "Say exactly: OK"
    gate:
      kind: exact_trimmed
      expected: "OK"
"#,
        );
        assert!(corpus.is_ok());
        let Some(corpus) = corpus.ok() else { return };
        assert!(corpus.defaults.qwen_no_think);

        let corpus = serde_yaml::from_str::<AnswerCorpus>(
            r#"
schema: 1
artifact_kind: slm_answer_corpus
name: qwen-default-fixture
description: qwen default fixture
model:
  repo: Qwen/Qwen3-0.6B-GGUF
  file: Qwen3-0.6B-Q8_0.gguf
  sha256: 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
  family: qwen
defaults:
  prompt_template: qwen
  max_new_tokens: 1
  greedy: true
  deterministic: true
  strict_loader: true
  temperature: 0.0
cases:
  - id: say_ok
    question: "Say exactly: OK"
    gate:
      kind: exact_trimmed
      expected: "OK"
"#,
        );
        assert!(corpus.is_ok());
        let Some(corpus) = corpus.ok() else { return };
        assert!(!corpus.defaults.qwen_no_think);
    }

    #[test]
    fn slm_answer_aggregate_identity_uses_dense_cpu_lane() {
        let rows = vec![json!({
            "backend": {
                "selected_backend": "cpu-rust",
                "runtime_api": "cpu",
                "fallback_used": false,
            },
            "kernel": {
                "selected_kernel": "dense-qwen-cpu-reference",
            },
            "tokenizer": {
                "source": "gguf_metadata",
            },
        })];

        assert_eq!(aggregate_case_str(&rows, &["backend", "selected_backend"]), Some("cpu-rust"));
        assert_eq!(
            aggregate_case_str(&rows, &["kernel", "selected_kernel"]),
            Some("dense-qwen-cpu-reference")
        );
        assert_eq!(aggregate_case_str(&rows, &["tokenizer", "source"]), Some("gguf_metadata"));
        assert_eq!(answer_corpus_backend_lane("cpu", true, "qwen"), "dense_slm_cpu");
    }

    fn strict_answer_receipt_fixture(
        requested_backend: &str,
        selected_backend: &str,
        runtime_api: &str,
        kernel_id: &str,
    ) -> Value {
        json!({
            "text": " 4",
            "requested_backend": requested_backend,
            "selected_backend": selected_backend,
            "runtime_api": runtime_api,
            "fallback_used": false,
            "loader": { "mode": "real_gguf" },
            "tokenizer": {
                "source": "explicit",
                "strict": true,
                "pretokenizer_authority": "llama-bpe"
            },
            "kernel": { "kernel_id": kernel_id },
            "qk256_hot_path": {
                "qk256_f32_scalar_gemv_invocations": 0,
                "qk256_f32_avx2_gemv_invocations": 0,
                "qk256_i8s_scaled_scalar_invocations": 1,
                "qk256_i8s_scaled_avx2_invocations": 0,
                "qk256_flat_bytes_extracted_count": 1,
                "input_rows_materialized_count": 1,
                "output_rows_allocated_count": 1,
                "no_scale_f32_gemv_invocations": 0,
                "scaled_i2s_i8s_gemv_invocations": 1,
                "audited_tensor_materialization_count": 3,
                "selected_kernel": "qk256-i2s-i8s-scaled-scalar-gemv",
                "qk256_execution_path": "scaled_i2s_i8s",
                "math_changed": false,
                "speedup_claim": false
            },
            "tokens": {
                "prompt": 3,
                "generated": 1,
                "total": 4,
                "prompt_ids": [1, 2, 3],
                "generated_ids": [4]
            },
            "model": {
                "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
                "file": "ggml-model-i2_s.gguf",
                "sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
                "family": "bitnet",
                "architecture": "bitnet_b1_58"
            },
            "timing": {
                "model_load_ms": 1.0,
                "tokenizer_load_ms": 1.0,
                "tokenize_ms": 1.0,
                "prefill_ms": 1.0,
                "first_token_ms": 1,
                "decode_total_ms": 1.0
            },
            "latency": {
                "total_ms": 2
            }
        })
    }

    #[test]
    fn cpu_answer_receipt_accepts_strict_cpu_truth() {
        let receipt =
            strict_answer_receipt_fixture("cpu", "cpu-rust", "cpu", "i2_s-avx2-reference");

        assert!(answer_receipt_failed_rules(&receipt, "cpu").is_empty());
    }

    #[test]
    fn cpu_answer_receipt_rejects_missing_qk256_hot_path_counters() {
        let mut receipt =
            strict_answer_receipt_fixture("cpu", "cpu-rust", "cpu", "i2_s-avx2-reference");
        if let Some(receipt) = receipt.as_object_mut() {
            receipt.remove("qk256_hot_path");
        }

        let failed = answer_receipt_failed_rules(&receipt, "cpu");

        assert!(failed.contains(&"qk256_hot_path_recorded".to_string()));
    }

    #[test]
    fn cpu_answer_receipt_rejects_zero_qk256_hot_path_invocations() {
        let mut receipt =
            strict_answer_receipt_fixture("cpu", "cpu-rust", "cpu", "i2_s-avx2-reference");
        receipt["qk256_hot_path"]["qk256_i8s_scaled_scalar_invocations"] = json!(0);
        receipt["qk256_hot_path"]["scaled_i2s_i8s_gemv_invocations"] = json!(0);
        receipt["qk256_hot_path"]["audited_tensor_materialization_count"] = json!(0);

        let failed = answer_receipt_failed_rules(&receipt, "cpu");

        assert!(failed.contains(&"qk256_hot_path_invocations_positive".to_string()));
        assert!(failed.contains(&"qk256_hot_path_materialization_audited".to_string()));
    }

    #[test]
    fn slm_answer_receipt_accepts_dense_qwen_cpu_provenance() {
        let receipt = json!({
            "text": "4",
            "requested_backend": "cpu",
            "selected_backend": "cpu-rust",
            "runtime_api": "cpu",
            "fallback_used": false,
            "loader": { "mode": "real_gguf" },
            "tokenizer": {
                "source": "explicit",
                "strict": true,
                "pretokenizer_authority": "gguf_metadata"
            },
            "model": {
                "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "file": "qwen2.5-0.5b-instruct-q8_0.gguf",
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "family": "qwen",
                "architecture": "qwen2"
            },
            "dense_slm": {
                "model_family": "qwen",
                "kernel_id": "dense-qwen-cpu-reference",
                "layout": "gguf_dense_q8_0"
            },
            "execution_coverage": {
                "dense_slm_layers_total": null,
                "dense_slm_layers_on_cpu": null,
                "unsupported_ops": [],
                "execution_claim": "dense_slm_cpu_reference_answer_smoke"
            },
            "kernel": {
                "kernel_id": "dense-qwen-cpu-reference",
                "family": "dense_qwen",
                "layout": "gguf_dense_q8_0"
            },
            "tokens": {
                "prompt": 3,
                "generated": 1,
                "total": 4,
                "prompt_ids": [1, 2, 3],
                "generated_ids": [4]
            },
            "timing": {
                "model_load_ms": 1.0,
                "tokenizer_load_ms": 1.0,
                "tokenize_ms": 1.0,
                "prefill_ms": 1.0,
                "first_token_ms": 1,
                "decode_total_ms": 1.0
            },
            "latency": {
                "total_ms": 2
            }
        });

        assert!(answer_receipt_failed_rules(&receipt, "cpu").is_empty());
    }

    #[test]
    fn slm_answer_receipt_rejects_bitnet_provenance_for_qwen() {
        let receipt = json!({
            "requested_backend": "cpu",
            "selected_backend": "cpu-rust",
            "runtime_api": "cpu",
            "fallback_used": false,
            "loader": { "mode": "real_gguf" },
            "tokenizer": { "source": "explicit", "strict": true },
            "model": { "family": "qwen" },
            "bitnet": { "kernel_family": "i2_s" },
            "execution_coverage": { "bitnet_linear_layers_cpu_fallback": 0 },
            "kernel": {
                "kernel_id": "i2_s-avx2-reference",
                "family": "i2_s",
                "layout": "gguf_packed_i2_s"
            },
            "tokens": {
                "prompt_ids": [1, 2, 3],
                "generated_ids": [4]
            }
        });

        let failed = answer_receipt_failed_rules(&receipt, "cpu");
        assert!(failed.contains(&"dense_slm_provenance_recorded".to_string()));
        assert!(failed.contains(&"dense_slm_no_bitnet_provenance".to_string()));
        assert!(failed.contains(&"dense_slm_kernel_not_bitnet_qk256".to_string()));
        assert!(failed.contains(&"dense_slm_layout_not_bitnet_qk256".to_string()));
        assert!(failed.contains(&"dense_slm_execution_coverage_not_bitnet_qk256".to_string()));
    }

    #[test]
    fn slm_answer_receipt_rejects_nested_bitnet_provenance_for_qwen() {
        let receipt = json!({
            "requested_backend": "cpu",
            "selected_backend": "cpu-rust",
            "runtime_api": "cpu",
            "fallback_used": false,
            "loader": { "mode": "real_gguf" },
            "tokenizer": { "source": "explicit", "strict": true },
            "model": { "family": "qwen" },
            "dense_slm": {
                "model_family": "qwen",
                "bitnet_kernel_family": "dense_qwen",
                "kernel_id": "dense-qwen-cpu-reference",
                "layout": "gguf_dense_q8_0"
            },
            "execution_coverage": {
                "dense_slm_layers_total": null,
                "dense_slm_layers_on_cpu": null,
                "unsupported_ops": [],
                "execution_claim": "dense_slm_cpu_reference_answer_smoke"
            },
            "kernel": {
                "kernel_id": "dense-qwen-cpu-reference",
                "family": "qk256",
                "layout": "gguf_dense_q8_0"
            },
            "strict_provenance": {
                "requested_kernel": "i2_s-avx2-reference",
                "selected_kernel": "dense-qwen-cpu-reference",
                "quant_format": "BitNet packed kernel"
            },
            "tokens": {
                "prompt_ids": [1, 2, 3],
                "generated_ids": [4]
            }
        });

        let failed = answer_receipt_failed_rules(&receipt, "cpu");
        assert!(failed.contains(&"dense_slm_kernel_family_not_bitnet_qk256".to_string()));
        assert!(failed.contains(&"dense_slm_fields_not_bitnet_qk256".to_string()));
        assert!(failed.contains(&"dense_slm_strict_provenance_not_bitnet_qk256".to_string()));
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
    fn a770_answer_corpus_route_is_diagnostic_opencl() {
        assert_eq!(normalize_answer_corpus_device(INTEL_ARC_A770_OPENCL), INTEL_A770_OPENCL);
        assert_eq!(normalize_answer_corpus_device("a770-opencl"), INTEL_A770_OPENCL);
        assert_eq!(
            answer_corpus_artifact_kind(INTEL_A770_OPENCL, "bitnet_answer_corpus"),
            "bitnet_a770_opencl_answer_diagnostic_corpus"
        );
        assert_eq!(answer_corpus_runtime_api(INTEL_A770_OPENCL), "opencl");
        assert_eq!(
            answer_corpus_backend_lane(INTEL_A770_OPENCL, false, "bitnet"),
            "bitnet_a770_opencl"
        );
        let proof_args: Vec<String> = answer_corpus_child_proof_args(INTEL_A770_OPENCL)
            .into_iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            proof_args,
            vec![
                "--proof-model-contract",
                A770_BITNET_MODEL_CONTRACT,
                "--proof-kernel-route",
                A770_BITNET_QK256_ROUTE_ID,
            ]
        );
        assert!(answer_corpus_child_proof_args("cpu").is_empty());
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
            category: None,
            profile: None,
            seed_material: None,
            question: "What is 2+2? Answer with only the number.".to_string(),
            max_new_tokens: Some(4),
            timeout_seconds: None,
            min_generated_tokens: None,
            min_distinct_generated_tokens: None,
            reference: None,
            scoring: None,
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
            category: None,
            profile: None,
            seed_material: None,
            question: "What is 2+2? Answer with only the number.".to_string(),
            max_new_tokens: Some(4),
            timeout_seconds: None,
            min_generated_tokens: None,
            min_distinct_generated_tokens: None,
            reference: None,
            scoring: None,
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
        let receipt = strict_answer_receipt_fixture(
            "apple-m4-cpu-neon",
            "apple-m4-cpu-neon",
            "cpu",
            "i2_s-scalar-reference",
        );

        assert!(answer_receipt_failed_rules(&receipt, "apple-m4-cpu-neon").is_empty());
    }

    #[test]
    fn answer_receipt_accepts_strict_cuda_truth() {
        let mut receipt = strict_answer_receipt_fixture(
            RTX_5070_TI_CUDA,
            RTX_5070_TI_CUDA,
            "cuda",
            "qk256_gemv_cuda",
        );
        receipt["kernel_stats"] = json!([{ "kernel_id": "qk256_gemv_cuda", "invocations": 8 }]);
        receipt["execution_coverage"] = json!({ "bitnet_linear_layers_cpu_fallback": 0 });
        receipt["execution_plan"] = json!({
            "planner_version": "cuda-planner-004",
            "model_family": "bitnet_b1_58",
            "quantization": "i2_s_qk256",
            "selected_route": "bitnet_qk256_cuda",
            "requested_backend": RTX_5070_TI_CUDA,
            "selected_backend": RTX_5070_TI_CUDA,
            "runtime_api": "cuda",
            "strict_fallback_policy": "reject",
            "dense_regular_llm_cuda": false,
            "bitnet_packed_qk256_cuda": true,
            "cuda_bitnet_qk256_ops": 8,
            "cuda_dense_regular_llm_ops": 0,
            "cpu_fallback_ops": 0,
            "unsupported_ops": 0,
            "total_ops": 8,
            "cuda_ops": 8,
            "mixed_cuda_routes": false,
            "fallback_used": false,
            "strict_cuda_ready": true,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        });

        assert!(answer_receipt_failed_rules(&receipt, RTX_5070_TI_CUDA).is_empty());
    }

    #[test]
    fn answer_receipt_accepts_strict_a770_opencl_truth() {
        let mut receipt = strict_answer_receipt_fixture(
            INTEL_A770_OPENCL,
            INTEL_A770_OPENCL,
            "opencl",
            "a770_opencl_qk256_i2s_i8s_scaled_dispatch_candidate",
        );
        receipt["claim_boundary"] = json!({
            "speedup_claim": false,
            "full_a770_residency_claimed": false,
            "trusted_partial_acceleration_claimed": false,
            "a770_speedup_claimed": false,
            "broad_performance_claimed": false
        });
        receipt["proof_summary"] = json!({
            "model_contract": A770_BITNET_MODEL_CONTRACT,
            "model_contract_declared": true,
            "kernel_route": {
                "route_id": A770_BITNET_QK256_ROUTE_ID,
                "route_declared": true,
                "diagnostic_only": true,
                "claimable": false
            },
            "route_declared": true,
            "backend_claimable": false
        });

        assert!(answer_receipt_failed_rules(&receipt, INTEL_A770_OPENCL).is_empty());
    }

    #[test]
    fn a770_answer_receipt_requires_model_contract_and_kernel_route() {
        let receipt = strict_answer_receipt_fixture(
            INTEL_A770_OPENCL,
            INTEL_A770_OPENCL,
            "opencl",
            "a770_opencl_qk256_i2s_i8s_scaled_dispatch_candidate",
        );

        let failed = answer_receipt_failed_rules(&receipt, INTEL_A770_OPENCL);

        assert!(failed.contains(&"a770_model_contract_declared".to_string()));
        assert!(failed.contains(&"a770_model_contract_path".to_string()));
        assert!(failed.contains(&"a770_kernel_route_declared".to_string()));
        assert!(failed.contains(&"a770_kernel_route_id".to_string()));
        assert!(failed.contains(&"a770_kernel_route_diagnostic_only".to_string()));
        assert!(failed.contains(&"a770_kernel_route_claimable_false".to_string()));
        assert!(failed.contains(&"a770_backend_claimable_false".to_string()));
    }

    #[test]
    fn cuda_answer_receipt_rejects_missing_execution_plan() {
        let mut receipt = strict_answer_receipt_fixture(
            RTX_5070_TI_CUDA,
            RTX_5070_TI_CUDA,
            "cuda",
            "qk256_gemv_cuda",
        );
        receipt["kernel_stats"] = json!([{ "kernel_id": "qk256_gemv_cuda", "invocations": 8 }]);
        receipt["execution_coverage"] = json!({ "bitnet_linear_layers_cpu_fallback": 0 });

        let failed = answer_receipt_failed_rules(&receipt, RTX_5070_TI_CUDA);

        assert!(failed.contains(&"execution_plan_planner_version".to_string()));
        assert!(failed.contains(&"execution_plan_selected_route_bitnet_qk256_cuda".to_string()));
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
        assert!(failed.contains(&"generated_text_recorded".to_string()));
        assert!(failed.contains(&"fallback_false".to_string()));
        assert!(failed.contains(&"loader_real_gguf".to_string()));
        assert!(failed.contains(&"model_sha256_recorded".to_string()));
        assert!(failed.contains(&"tokenizer_source_recorded".to_string()));
        assert!(failed.contains(&"tokenizer_strict".to_string()));
        assert!(failed.contains(&"tokenizer_pretokenizer_authority_recorded".to_string()));
        assert!(failed.contains(&"selected_kernel_production".to_string()));
        assert!(failed.contains(&"prompt_token_count_recorded".to_string()));
        assert!(failed.contains(&"generated_token_count_recorded".to_string()));
        assert!(failed.contains(&"total_token_count_recorded".to_string()));
        assert!(failed.contains(&"prompt_token_ids_recorded".to_string()));
        assert!(failed.contains(&"generated_token_ids_recorded".to_string()));
        assert!(failed.contains(&"timing_decode_total_ms_recorded".to_string()));
        assert!(failed.contains(&"latency_total_ms_recorded".to_string()));
    }

    #[test]
    fn answer_receipt_rejects_speedup_or_full_inference_claims() {
        let mut receipt =
            strict_answer_receipt_fixture("cpu", "cpu-rust", "cpu", "i2_s-avx2-reference");
        receipt["speedup_claim"] = json!(true);
        receipt["claim_boundary"] = json!({
            "speedup_claim": true,
            "full_metal_inference_claimed": true,
            "broad_performance_claimed": true
        });

        let failed = answer_receipt_failed_rules(&receipt, "cpu");

        assert!(failed.contains(&"speedup_claim_false".to_string()));
        assert!(!failed.contains(&"fallback_false".to_string()));
    }

    #[test]
    fn answer_receipt_quality_contract_covers_m4_qa_003_fields() {
        let required_fields = answer_receipt_required_case_fields();
        for field in [
            "text",
            "tokens.prompt",
            "tokens.generated",
            "tokens.total",
            "tokens.prompt_ids",
            "tokens.generated_ids",
            "model.repo",
            "model.file",
            "model.sha256",
            "tokenizer.pretokenizer_authority",
            "requested_backend",
            "selected_backend",
            "fallback_used",
            "timing.decode_total_ms",
            "latency.total_ms",
        ] {
            assert!(required_fields.contains(&field), "missing required receipt field `{field}`");
        }
        let checked_rules = answer_receipt_checked_rules();
        for rule in [
            "generated_text_recorded",
            "model_sha256_recorded",
            "tokenizer_pretokenizer_authority_recorded",
            "generated_token_count_matches_ids",
            "timing_decode_total_ms_recorded",
            "fallback_false",
            "speedup_claim_false",
            "a770_kernel_route_declared",
        ] {
            assert!(checked_rules.contains(&rule), "missing receipt quality rule `{rule}`");
        }
    }
}
