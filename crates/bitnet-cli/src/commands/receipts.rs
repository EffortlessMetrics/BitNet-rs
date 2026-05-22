//! Receipt explanation helpers for operator-facing proof summaries.
//!
//! This command intentionally does not validate a receipt against one narrow
//! schema. It extracts the common proof fields shared by BitNet CUDA, dense CUDA,
//! answer-corpus, warm-session, and benchmark receipts so users can inspect what
//! actually ran without needing to know every receipt variant.

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const DEFAULT_RECEIPTS_DIR: &str = "target/bitnet/receipts";
const MODEL_COVERAGE_MATRIX_RELATIVE: &[&str] =
    &["ci", "model-artifacts", "model-coverage-matrix.toml"];

/// Inspect and explain BitNet-rs receipt JSON.
#[derive(Args, Debug, Clone)]
pub struct ReceiptsCommand {
    #[command(subcommand)]
    pub action: ReceiptsAction,
}

#[derive(Subcommand, Debug, Clone)]
pub enum ReceiptsAction {
    /// Explain a receipt file, or the newest receipt under target/bitnet/receipts.
    Explain {
        /// Receipt file to explain. With --latest, this may be a directory to search.
        #[arg(value_name = "PATH")]
        path: Option<PathBuf>,

        /// Explain the newest JSON receipt under the path or default receipt directory.
        #[arg(long, default_value_t = false)]
        latest: bool,

        /// Emit normalized JSON instead of text.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// Emit text or normalized JSON. Equivalent to --json when set to json.
        #[arg(long, value_enum)]
        format: Option<ReceiptExplainFormat>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ReceiptExplainFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ReceiptExplanation {
    pub schema_version: u32,
    pub path: String,
    pub model_coverage_row: Option<String>,
    pub current_tier: Option<String>,
    pub selected_backend: Option<String>,
    pub selected_route: Option<String>,
    pub fallback_used: Option<bool>,
    pub product_cli_ready: Option<bool>,
    pub server_ready: Option<bool>,
    pub server_ready_scope: Option<String>,
    pub speedup_claim: Option<bool>,
    pub full_residency_claim: Option<bool>,
    pub bitnet_packed_i2s_qk256_proof: Option<bool>,
    pub dense_regular_llm_cuda_proof: Option<bool>,
    pub artifact_kind: Option<String>,
    pub claim: Option<String>,
    pub model: Option<String>,
    pub model_coverage: ModelCoverageExplanation,
    pub backend: BackendExplanation,
    pub execution_plan: ExecutionPlanExplanation,
    pub kernels: Vec<String>,
    pub quality: QualityExplanation,
    pub timing: TimingExplanation,
    pub residency: ResidencyExplanation,
    pub benchmark_qualification: BenchmarkQualificationExplanation,
    pub openvino: OpenVinoExplanation,
    pub claim_limits: ClaimLimitsExplanation,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct ModelCoverageExplanation {
    pub source: Option<String>,
    pub row: Option<String>,
    pub current_tier: Option<String>,
    pub status: Option<String>,
    pub route: Option<String>,
    pub product_cli_ready: Option<bool>,
    pub speedup_claim: Option<bool>,
    pub benchmark_qualified: Option<bool>,
    pub server_ready: Option<bool>,
    pub server_ready_scope: Option<String>,
    pub full_residency_claim: Option<bool>,
    pub bitnet_packed_i2s_qk256_proof: Option<bool>,
    pub dense_regular_llm_cuda_proof: Option<bool>,
    pub claim_boundary: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct BackendExplanation {
    pub requested_backend: Option<String>,
    pub selected_backend: Option<String>,
    pub runtime_api: Option<String>,
    pub fallback_used: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct ExecutionPlanExplanation {
    pub selected_route: Option<String>,
    pub model_family: Option<String>,
    pub quantization: Option<String>,
    pub strict_cuda_ready: Option<bool>,
    pub speedup_claim: Option<bool>,
    pub full_cuda_residency_claimed: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct QualityExplanation {
    pub answer_quality_passed: Option<bool>,
    pub benchmark_quality_passed: Option<bool>,
    pub parity_passed: Option<bool>,
    pub first_divergence: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct TimingExplanation {
    pub total_ms: Option<f64>,
    pub first_token_ms: Option<f64>,
    pub decode_total_ms: Option<f64>,
    pub steady_decode_tok_s: Option<f64>,
    pub kernel_time_ms: Option<f64>,
    pub host_to_device_bytes: Option<u64>,
    pub device_to_host_bytes: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct ResidencyExplanation {
    pub qk256_cuda_residency_claimed: Option<bool>,
    pub model_loaded_once: Option<bool>,
    pub cuda_context_once: Option<bool>,
    pub weights_uploaded_once: Option<bool>,
    pub per_request_model_load: Option<bool>,
    pub per_token_weight_upload: Option<bool>,
    pub workspace_reused: Option<bool>,
    pub kv_cache_residency: Option<String>,
    pub full_cuda_residency_claimed: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct ClaimLimitsExplanation {
    pub speedup_claim: Option<bool>,
    pub benchmark_qualified_speedup: Option<bool>,
    pub full_cuda_residency_claimed: Option<bool>,
    pub dense_gguf_inference_claimed: Option<bool>,
    pub bitnet_packed_i2s_qk256_proof: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct BenchmarkQualificationExplanation {
    pub status: Option<String>,
    pub benchmark_qualified_speedup: Option<bool>,
    pub accepted_profiles: Vec<String>,
    pub blocked_profiles: Vec<String>,
    pub speedup_claim_allowed: Option<bool>,
    pub transfer_timing_status: Option<String>,
    pub host_to_device_source: Option<String>,
    pub host_to_device_scope: Option<String>,
    pub host_to_device_includes_non_transfer_overhead: Option<bool>,
    pub pure_host_to_device_timing_recorded: Option<bool>,
    pub device_to_host_timing_recorded: Option<bool>,
    pub profile_reviews: Vec<BenchmarkProfileExplanation>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct BenchmarkProfileExplanation {
    pub profile: String,
    pub decision: Option<String>,
    pub benchmark_qualified_speedup: Option<bool>,
    pub speedup_claim_allowed: Option<bool>,
    pub fallback_free: Option<bool>,
    pub quality_passed: Option<bool>,
    pub generated_token_ids_match: Option<bool>,
    pub cpu_total_ms_mean: Option<f64>,
    pub cuda_total_ms_mean: Option<f64>,
    pub observed_cpu_total_ms_div_cuda_total_ms: Option<f64>,
    pub host_to_device_ms: Option<f64>,
    pub host_to_device_ms_source: Option<String>,
    pub host_to_device_ms_scope: Option<String>,
    pub host_to_device_ms_includes_non_transfer_overhead: Option<bool>,
    pub pure_host_to_device_ms_source: Option<String>,
    pub device_to_host_ms: Option<f64>,
    pub device_to_host_ms_source: Option<String>,
    pub blockers: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct OpenVinoExplanation {
    pub route_id: Option<String>,
    pub route_reason: Option<String>,
    pub requested_backend: Option<String>,
    pub selected_backend: Option<String>,
    pub runtime_api: Option<String>,
    pub runtime_device: Option<String>,
    pub resolved_device: Option<String>,
    pub proof_family: Option<String>,
    pub proof_stage: Option<String>,
    pub backend_lane: Option<String>,
    pub selected_kernel_or_runtime: Option<String>,
    pub quality_status: Option<String>,
    pub timing_scope: Option<String>,
    pub promotion_status: Option<String>,
    pub blockers: Vec<String>,
    pub does_not_prove: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelCoverageMatrix {
    schema: u32,
    artifact_kind: String,
    #[serde(default)]
    entry: Vec<ModelCoverageEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelCoverageEntry {
    id: String,
    status: String,
    current_tier: String,
    #[serde(default)]
    accelerator_routes: Vec<String>,
    claim_boundary: String,
    claims: ModelCoverageClaims,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelCoverageClaims {
    benchmark_qualified: bool,
    product_cli_ready: bool,
    server_ready: bool,
    speedup_claim: bool,
    full_residency_claim: bool,
    bitnet_packed_i2s_qk256_proof: bool,
    dense_regular_llm_cuda_proof: bool,
}

impl ReceiptsCommand {
    pub async fn execute(&self) -> Result<()> {
        match &self.action {
            ReceiptsAction::Explain { path, latest, json, format } => {
                let receipt_path = resolve_receipt_path(path.as_deref(), *latest)?;
                let receipt = read_receipt_json(&receipt_path)?;
                let explanation = explain_receipt(&receipt_path, &receipt);
                let output_json = *json || matches!(format, Some(ReceiptExplainFormat::Json));
                if output_json {
                    println!("{}", serde_json::to_string_pretty(&explanation)?);
                } else {
                    print_receipt_explanation(&explanation);
                }
                Ok(())
            }
        }
    }
}

pub(crate) fn resolve_receipt_path(path: Option<&Path>, latest: bool) -> Result<PathBuf> {
    if latest {
        let search_root = path.unwrap_or_else(|| Path::new(DEFAULT_RECEIPTS_DIR));
        return latest_receipt_under(search_root);
    }

    let path = path.ok_or_else(|| anyhow!("pass a receipt path or use --latest"))?;
    if path.is_dir() {
        bail!("{} is a directory; pass --latest to search it", path.display());
    }
    Ok(path.to_path_buf())
}

pub(crate) fn read_receipt_json(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse receipt JSON {}", path.display()))
}

fn latest_receipt_under(root: &Path) -> Result<PathBuf> {
    let mut latest: Option<(SystemTime, PathBuf)> = None;
    collect_latest_json(root, &mut latest)?;
    latest
        .map(|(_, path)| path)
        .ok_or_else(|| anyhow!("no JSON receipts found under {}", root.display()))
}

fn collect_latest_json(root: &Path, latest: &mut Option<(SystemTime, PathBuf)>) -> Result<()> {
    if root.is_file() {
        consider_latest_file(root, latest)?;
        return Ok(());
    }

    for entry in fs::read_dir(root).with_context(|| format!("failed to list {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_latest_json(&path, latest)?;
        } else if file_type.is_file() {
            consider_latest_file(&path, latest)?;
        }
    }
    Ok(())
}

fn consider_latest_file(path: &Path, latest: &mut Option<(SystemTime, PathBuf)>) -> Result<()> {
    if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
        return Ok(());
    }
    let modified = fs::metadata(path)
        .with_context(|| format!("failed to stat {}", path.display()))?
        .modified()
        .unwrap_or(SystemTime::UNIX_EPOCH);
    let should_replace = latest.as_ref().is_none_or(|(current_time, current_path)| {
        modified > *current_time || (modified == *current_time && path < current_path.as_path())
    });
    if should_replace {
        *latest = Some((modified, path.to_path_buf()));
    }
    Ok(())
}

pub fn explain_receipt(path: &Path, receipt: &Value) -> ReceiptExplanation {
    let mut explanation = ReceiptExplanation {
        schema_version: 1,
        path: path.display().to_string(),
        model_coverage_row: None,
        current_tier: None,
        selected_backend: None,
        selected_route: None,
        fallback_used: None,
        product_cli_ready: None,
        server_ready: None,
        server_ready_scope: None,
        speedup_claim: None,
        full_residency_claim: None,
        bitnet_packed_i2s_qk256_proof: None,
        dense_regular_llm_cuda_proof: None,
        artifact_kind: string_at(receipt, &["artifact_kind"]),
        claim: string_at(receipt, &["claim"]),
        model: model_summary(receipt),
        model_coverage: ModelCoverageExplanation::default(),
        backend: backend_explanation(receipt),
        execution_plan: execution_plan_explanation(receipt),
        kernels: kernel_ids(receipt),
        quality: quality_explanation(receipt),
        timing: timing_explanation(receipt),
        residency: residency_explanation(receipt),
        benchmark_qualification: benchmark_qualification_explanation(receipt),
        openvino: openvino_explanation(receipt),
        claim_limits: claim_limits_explanation(receipt),
    };
    explanation.model_coverage = model_coverage_explanation(&explanation, receipt);
    apply_receipt_json_contract_aliases(&mut explanation);
    explanation
}

fn apply_receipt_json_contract_aliases(explanation: &mut ReceiptExplanation) {
    explanation.model_coverage_row = explanation.model_coverage.row.clone();
    explanation.current_tier = explanation.model_coverage.current_tier.clone();
    explanation.selected_backend = explanation.backend.selected_backend.clone();
    explanation.selected_route = explanation
        .execution_plan
        .selected_route
        .clone()
        .or_else(|| explanation.model_coverage.route.clone());
    explanation.fallback_used = explanation.backend.fallback_used;
    explanation.product_cli_ready = explanation.model_coverage.product_cli_ready;
    explanation.server_ready = explanation.model_coverage.server_ready;
    explanation.server_ready_scope = explanation.model_coverage.server_ready_scope.clone();
    explanation.speedup_claim =
        explanation.model_coverage.speedup_claim.or(explanation.claim_limits.speedup_claim);
    explanation.full_residency_claim = explanation
        .model_coverage
        .full_residency_claim
        .or(explanation.residency.full_cuda_residency_claimed)
        .or(explanation.claim_limits.full_cuda_residency_claimed);
    explanation.bitnet_packed_i2s_qk256_proof = explanation
        .model_coverage
        .bitnet_packed_i2s_qk256_proof
        .or(explanation.claim_limits.bitnet_packed_i2s_qk256_proof);
    explanation.dense_regular_llm_cuda_proof = explanation
        .model_coverage
        .dense_regular_llm_cuda_proof
        .or(explanation.claim_limits.dense_gguf_inference_claimed);
}

fn backend_explanation(receipt: &Value) -> BackendExplanation {
    BackendExplanation {
        requested_backend: string_at(receipt, &["requested_backend"])
            .or_else(|| string_at(receipt, &["backend", "requested_backend"]))
            .or_else(|| string_at(receipt, &["execution_plan", "requested_backend"])),
        selected_backend: string_at(receipt, &["selected_backend"])
            .or_else(|| string_at(receipt, &["backend", "selected_backend"]))
            .or_else(|| string_at(receipt, &["execution_plan", "selected_backend"])),
        runtime_api: string_at(receipt, &["runtime_api"])
            .or_else(|| string_at(receipt, &["backend", "runtime_api"]))
            .or_else(|| string_at(receipt, &["execution_plan", "runtime_api"])),
        fallback_used: bool_at(receipt, &["fallback_used"])
            .or_else(|| bool_at(receipt, &["backend", "fallback_used"]))
            .or_else(|| bool_at(receipt, &["execution_plan", "fallback_used"])),
    }
}

fn execution_plan_explanation(receipt: &Value) -> ExecutionPlanExplanation {
    ExecutionPlanExplanation {
        selected_route: string_at(receipt, &["execution_plan", "selected_route"])
            .or_else(|| string_at(receipt, &["route_id"]))
            .or_else(|| string_at(receipt, &["route", "route_id"]))
            .or_else(|| string_at(receipt, &["selected_route"])),
        model_family: string_at(receipt, &["execution_plan", "model_family"]),
        quantization: string_at(receipt, &["execution_plan", "quantization"]),
        strict_cuda_ready: bool_at(receipt, &["execution_plan", "strict_cuda_ready"]),
        speedup_claim: bool_at(receipt, &["execution_plan", "speedup_claim"]),
        full_cuda_residency_claimed: bool_at(
            receipt,
            &["execution_plan", "full_cuda_residency_claimed"],
        ),
    }
}

fn quality_explanation(receipt: &Value) -> QualityExplanation {
    QualityExplanation {
        answer_quality_passed: bool_at(receipt, &["answer_quality", "passed"])
            .or_else(|| bool_at(receipt, &["quality", "passed"]))
            .or_else(|| bool_at(receipt, &["quality_gate", "passed"]))
            .or_else(|| bool_at(receipt, &["answer_gate", "passed"]))
            .or_else(|| bool_at(receipt, &["generation", "all_answer_gates_passed"]))
            .or_else(|| bool_at(receipt, &["quality", "garbage_filter_passed"]))
            .or_else(|| bool_at(receipt, &["benchmark", "quality_passed"])),
        benchmark_quality_passed: bool_at(receipt, &["benchmark", "quality_passed"]),
        parity_passed: bool_at(receipt, &["parity", "passed"]),
        first_divergence: string_at(receipt, &["first_divergence", "kind"])
            .or_else(|| string_at(receipt, &["first_divergence", "classification"]))
            .or_else(|| string_at(receipt, &["parity", "first_divergence"])),
    }
}

fn timing_explanation(receipt: &Value) -> TimingExplanation {
    TimingExplanation {
        total_ms: f64_at(receipt, &["timing", "total_ms"])
            .or_else(|| f64_at(receipt, &["benchmark", "cuda_median_total_ms"]))
            .or_else(|| f64_at(receipt, &["benchmark", "cuda_median_total_session_ms"])),
        first_token_ms: f64_at(receipt, &["timing", "first_token_ms"])
            .or_else(|| f64_at(receipt, &["benchmark", "cuda_median_first_token_ms"])),
        decode_total_ms: f64_at(receipt, &["timing", "decode_total_ms"])
            .or_else(|| f64_at(receipt, &["benchmark", "cuda_median_decode_total_ms"])),
        steady_decode_tok_s: f64_at(receipt, &["timing", "steady_decode_tok_s"])
            .or_else(|| f64_at(receipt, &["benchmark", "cuda_median_generated_tokens_per_second"])),
        kernel_time_ms: f64_at(receipt, &["timing", "kernel_time_ms"])
            .or_else(|| f64_at(receipt, &["timing", "cuda_kernel_time_ms"]))
            .or_else(|| f64_at(receipt, &["benchmark", "cuda_median_kernel_time_ms"]))
            .or_else(|| {
                f64_at(
                    receipt,
                    &[
                        "cuda_execution_residency",
                        "host_device_transfer_accounting",
                        "kernel_time_ms",
                    ],
                )
            })
            .or_else(|| sum_kernel_f64(receipt, "kernel_time_ms")),
        host_to_device_bytes: u64_at(receipt, &["timing", "host_to_device_bytes"])
            .or_else(|| u64_at(receipt, &["benchmark", "cuda_median_host_to_device_bytes"]))
            .or_else(|| {
                u64_at(
                    receipt,
                    &[
                        "cuda_execution_residency",
                        "host_device_transfer_accounting",
                        "host_to_device_bytes",
                    ],
                )
            })
            .or_else(|| sum_kernel_u64(receipt, "host_to_device_bytes")),
        device_to_host_bytes: u64_at(receipt, &["timing", "device_to_host_bytes"])
            .or_else(|| u64_at(receipt, &["benchmark", "cuda_median_device_to_host_bytes"]))
            .or_else(|| {
                u64_at(
                    receipt,
                    &[
                        "cuda_execution_residency",
                        "host_device_transfer_accounting",
                        "device_to_host_bytes",
                    ],
                )
            })
            .or_else(|| sum_kernel_u64(receipt, "device_to_host_bytes")),
    }
}

fn residency_explanation(receipt: &Value) -> ResidencyExplanation {
    ResidencyExplanation {
        qk256_cuda_residency_claimed: bool_at(
            receipt,
            &["cuda_execution_residency", "claim_boundary", "qk256_cuda_residency_claimed"],
        ),
        model_loaded_once: bool_at(receipt, &["session_lifecycle", "model_loaded_once"])
            .or_else(|| bool_at(receipt, &["tensor_residency", "model_loaded_once"]))
            .or_else(|| bool_at(receipt, &["residency", "model_loaded_once"])),
        cuda_context_once: bool_at(receipt, &["session_lifecycle", "cuda_context_once"])
            .or_else(|| bool_at(receipt, &["session_lifecycle", "cuda_context_initialized_once"]))
            .or_else(|| bool_at(receipt, &["tensor_residency", "cuda_context_once"]))
            .or_else(|| bool_at(receipt, &["tensor_residency", "cuda_context_initialized_once"]))
            .or_else(|| bool_at(receipt, &["residency", "cuda_context_once"])),
        weights_uploaded_once: bool_at(
            receipt,
            &["cuda_execution_residency", "weights", "uploaded_once"],
        )
        .or_else(|| {
            bool_at(
                receipt,
                &["cuda_execution_residency", "weight_residency", "weights_uploaded_once"],
            )
        })
        .or_else(|| bool_at(receipt, &["bitnet", "weights_uploaded_once"]))
        .or_else(|| bool_at(receipt, &["session_lifecycle", "weights_uploaded_once"]))
        .or_else(|| bool_at(receipt, &["tensor_residency", "weights_uploaded_once"]))
        .or_else(|| bool_at(receipt, &["residency", "weights_uploaded_once"]))
        .or_else(|| bool_at(receipt, &["proof", "weights_uploaded_once"])),
        per_request_model_load: bool_at(receipt, &["session_lifecycle", "per_request_model_load"])
            .or_else(|| bool_at(receipt, &["tensor_residency", "per_request_model_load"]))
            .or_else(|| bool_at(receipt, &["residency", "per_request_model_load"])),
        per_token_weight_upload: bool_at(
            receipt,
            &["cuda_execution_residency", "weights", "per_token_weight_upload"],
        )
        .or_else(|| {
            bool_at(
                receipt,
                &["cuda_execution_residency", "weight_residency", "per_token_weight_upload"],
            )
        })
        .or_else(|| bool_at(receipt, &["bitnet", "per_token_weight_upload"]))
        .or_else(|| bool_at(receipt, &["tensor_residency", "per_token_weight_upload"]))
        .or_else(|| bool_at(receipt, &["residency", "per_token_weight_upload"]))
        .or_else(|| bool_at(receipt, &["proof", "per_token_weight_upload"])),
        workspace_reused: bool_at(receipt, &["session_lifecycle", "workspace_reused"])
            .or_else(|| bool_at(receipt, &["session_lifecycle", "runtime_buffers_reused"]))
            .or_else(|| bool_at(receipt, &["tensor_residency", "workspace_reused"]))
            .or_else(|| bool_at(receipt, &["tensor_residency", "runtime_buffers_reused"]))
            .or_else(|| bool_at(receipt, &["residency", "workspace_reused"]))
            .or_else(|| bool_at(receipt, &["residency", "runtime_buffers_reused"])),
        kv_cache_residency: string_at(
            receipt,
            &["cuda_execution_residency", "kv_cache", "residency"],
        )
        .or_else(|| string_at(receipt, &["cuda_execution_residency", "kv_cache", "device"]))
        .or_else(|| string_at(receipt, &["kv_cache", "device"])),
        full_cuda_residency_claimed: bool_at(
            receipt,
            &["cuda_execution_residency", "full_cuda_residency_claimed"],
        )
        .or_else(|| bool_at(receipt, &["tensor_residency", "full_cuda_residency_claimed"]))
        .or_else(|| bool_at(receipt, &["execution_plan", "full_cuda_residency_claimed"])),
    }
}

fn claim_limits_explanation(receipt: &Value) -> ClaimLimitsExplanation {
    ClaimLimitsExplanation {
        speedup_claim: bool_at(receipt, &["speedup_claim"])
            .or_else(|| bool_at(receipt, &["claim_boundary", "speedup_claim"]))
            .or_else(|| bool_at(receipt, &["execution_plan", "speedup_claim"])),
        benchmark_qualified_speedup: bool_at(receipt, &["benchmark_qualified_speedup"])
            .or_else(|| bool_at(receipt, &["benchmark", "benchmark_qualified_speedup"])),
        full_cuda_residency_claimed: bool_at(
            receipt,
            &["claim_boundary", "full_cuda_residency_claimed"],
        )
        .or_else(|| bool_at(receipt, &["execution_plan", "full_cuda_residency_claimed"]))
        .or_else(|| bool_at(receipt, &["cuda_execution_residency", "full_cuda_residency_claimed"])),
        dense_gguf_inference_claimed: bool_at(
            receipt,
            &["claim_boundary", "dense_gguf_inference_claimed"],
        )
        .or_else(|| bool_at(receipt, &["fixture", "dense_gguf_inference_claimed"]))
        .or_else(|| bool_at(receipt, &["tensor_residency", "dense_gguf_inference_claimed"])),
        bitnet_packed_i2s_qk256_proof: bool_at(
            receipt,
            &["claim_boundary", "bitnet_packed_i2s_qk256_proof"],
        )
        .or_else(|| bool_at(receipt, &["execution_path", "bitnet_packed_kernel_proof"])),
    }
}

fn benchmark_qualification_explanation(receipt: &Value) -> BenchmarkQualificationExplanation {
    let mut accepted_profiles =
        string_array_at(receipt, &["qualification_decision", "accepted_profiles"]);
    if accepted_profiles.is_empty() {
        accepted_profiles =
            string_array_at(receipt, &["comparator_summary", "accepted_speedup_profiles"]);
    }
    let mut blocked_profiles =
        string_array_at(receipt, &["qualification_decision", "blocked_profiles"]);
    if blocked_profiles.is_empty() {
        blocked_profiles = blocked_comparator_profiles(receipt);
    }

    BenchmarkQualificationExplanation {
        status: string_at(receipt, &["qualification_decision", "status"])
            .or_else(|| string_at(receipt, &["comparator_summary", "status"])),
        benchmark_qualified_speedup: bool_at(
            receipt,
            &["qualification_decision", "benchmark_qualified_speedup"],
        )
        .or_else(|| bool_at(receipt, &["benchmark_qualified_speedup"]))
        .or_else(|| bool_at(receipt, &["claim_boundary", "benchmark_qualified_speedup"]))
        .or_else(|| bool_at(receipt, &["comparator_summary", "benchmark_qualified_speedup"])),
        accepted_profiles,
        blocked_profiles,
        speedup_claim_allowed: bool_at(
            receipt,
            &["qualification_decision", "speedup_claim_allowed"],
        )
        .or_else(|| bool_at(receipt, &["comparator_summary", "speedup_claim_allowed"])),
        transfer_timing_status: string_at(receipt, &["transfer_timing_review", "status"])
            .or_else(|| string_at(receipt, &["transfer_timing", "status"])),
        host_to_device_source: string_at(
            receipt,
            &["transfer_timing_review", "host_to_device_source"],
        ),
        host_to_device_scope: string_at(
            receipt,
            &["transfer_timing_review", "host_to_device_scope"],
        ),
        host_to_device_includes_non_transfer_overhead: bool_at(
            receipt,
            &["transfer_timing_review", "host_to_device_ms_includes_non_transfer_overhead"],
        ),
        pure_host_to_device_timing_recorded: bool_at(
            receipt,
            &["transfer_timing_review", "host_to_device_pure_transfer_timing_recorded"],
        )
        .or_else(|| bool_at(receipt, &["transfer_timing", "pure_host_to_device_timing_recorded"])),
        device_to_host_timing_recorded: bool_at(
            receipt,
            &["transfer_timing_review", "device_to_host_timing_recorded"],
        )
        .or_else(|| bool_at(receipt, &["transfer_timing", "device_to_host_timing_recorded"])),
        profile_reviews: benchmark_profile_reviews(receipt),
    }
}

fn blocked_comparator_profiles(receipt: &Value) -> Vec<String> {
    if !is_repeated_comparator_receipt(receipt) {
        return Vec::new();
    }
    if bool_at(receipt, &["comparator_summary", "benchmark_qualified_speedup"]) == Some(true) {
        return Vec::new();
    }
    receipt
        .get("profiles")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|profile| string_at(profile, &["profile"]))
        .collect()
}

fn benchmark_profile_reviews(receipt: &Value) -> Vec<BenchmarkProfileExplanation> {
    if let Some(reviews) = receipt.get("profile_reviews").and_then(Value::as_array) {
        return reviews
            .iter()
            .filter_map(|review| {
                let profile = string_at(review, &["profile"])?;
                Some(BenchmarkProfileExplanation {
                    profile,
                    decision: string_at(review, &["decision"]),
                    benchmark_qualified_speedup: bool_at(review, &["benchmark_qualified_speedup"]),
                    speedup_claim_allowed: bool_at(review, &["speedup_claim_allowed"]),
                    fallback_free: bool_at(review, &["fallback_free"]),
                    quality_passed: bool_at(review, &["quality_passed"]),
                    generated_token_ids_match: bool_at(review, &["generated_token_ids_match"]),
                    cpu_total_ms_mean: f64_at(review, &["cpu_total_ms_mean"]),
                    cuda_total_ms_mean: f64_at(review, &["cuda_total_ms_mean"]),
                    observed_cpu_total_ms_div_cuda_total_ms: f64_at(
                        review,
                        &["observed_cpu_total_ms_div_cuda_total_ms"],
                    )
                    .or_else(|| {
                        f64_at(review, &["observed_median_cpu_total_ms_div_cuda_total_ms"])
                    }),
                    host_to_device_ms: f64_at(review, &["host_to_device_ms"]),
                    host_to_device_ms_source: string_at(review, &["host_to_device_ms_source"]),
                    host_to_device_ms_scope: string_at(review, &["host_to_device_ms_scope"]),
                    host_to_device_ms_includes_non_transfer_overhead: bool_at(
                        review,
                        &["host_to_device_ms_includes_non_transfer_overhead"],
                    ),
                    pure_host_to_device_ms_source: string_at(
                        review,
                        &["pure_host_to_device_ms_source"],
                    ),
                    device_to_host_ms: f64_at(review, &["device_to_host_ms"]),
                    device_to_host_ms_source: string_at(review, &["device_to_host_ms_source"]),
                    blockers: string_array_at(review, &["blockers"]),
                })
            })
            .collect();
    }

    if !is_repeated_comparator_receipt(receipt) {
        return Vec::new();
    }

    receipt
        .get("profiles")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|profile_receipt| {
            let profile = string_at(profile_receipt, &["profile"])?;
            Some(BenchmarkProfileExplanation {
                profile,
                decision: string_at(profile_receipt, &["status"]),
                benchmark_qualified_speedup: bool_at(
                    profile_receipt,
                    &["benchmark_qualified_speedup"],
                ),
                speedup_claim_allowed: bool_at(profile_receipt, &["speedup_claim"]),
                fallback_free: bool_at(profile_receipt, &["fallback_free"]),
                quality_passed: bool_at(profile_receipt, &["generated_token_ids_match"]),
                generated_token_ids_match: bool_at(profile_receipt, &["generated_token_ids_match"]),
                cpu_total_ms_mean: f64_at(profile_receipt, &["cpu_total_ms", "mean"]),
                cuda_total_ms_mean: f64_at(profile_receipt, &["cuda_total_ms", "mean"]),
                observed_cpu_total_ms_div_cuda_total_ms: comparator_speed_ratio(profile_receipt),
                host_to_device_ms: f64_at(profile_receipt, &["host_to_device_ms", "mean"]),
                host_to_device_ms_source: None,
                host_to_device_ms_scope: None,
                host_to_device_ms_includes_non_transfer_overhead: bool_at(
                    profile_receipt,
                    &["host_to_device_ms_includes_non_transfer_overhead"],
                ),
                pure_host_to_device_ms_source: None,
                device_to_host_ms: f64_at(profile_receipt, &["device_to_host_ms", "mean"]),
                device_to_host_ms_source: None,
                blockers: Vec::new(),
            })
        })
        .collect()
}

fn is_repeated_comparator_receipt(receipt: &Value) -> bool {
    receipt.get("comparator_summary").is_some()
        || string_at(receipt, &["artifact_kind"]).is_some_and(|kind| {
            kind == "qwen3_cuda_repeated_comparator"
                || kind == "dense_gguf_qwen_repeated_comparator"
        })
}

fn comparator_speed_ratio(profile: &Value) -> Option<f64> {
    let cpu = f64_at(profile, &["cpu_total_ms", "mean"])?;
    let cuda = f64_at(profile, &["cuda_total_ms", "mean"])?;
    (cuda > 0.0).then_some(cpu / cuda)
}

fn openvino_explanation(receipt: &Value) -> OpenVinoExplanation {
    let requested_backend = string_at(receipt, &["requested_backend"])
        .or_else(|| string_at(receipt, &["backend", "requested_backend"]))
        .or_else(|| string_at(receipt, &["execution_plan", "requested_backend"]));
    let selected_backend = string_at(receipt, &["selected_backend"])
        .or_else(|| string_at(receipt, &["backend", "selected_backend"]))
        .or_else(|| string_at(receipt, &["execution_plan", "selected_backend"]));
    let runtime_api = string_at(receipt, &["runtime_api"])
        .or_else(|| string_at(receipt, &["backend", "runtime_api"]))
        .or_else(|| string_at(receipt, &["execution_plan", "runtime_api"]));
    let route_id = string_at(receipt, &["route_id"])
        .or_else(|| string_at(receipt, &["route", "route_id"]))
        .or_else(|| string_at(receipt, &["execution_plan", "route_id"]))
        .or_else(|| string_at(receipt, &["execution_plan", "selected_route"]))
        .or_else(|| string_at(receipt, &["selected_route"]));
    let artifact_kind = string_at(receipt, &["artifact_kind"]);
    let backend_lane = string_at(receipt, &["backend_lane"]);

    if !is_openvino_receipt(
        artifact_kind.as_deref(),
        route_id.as_deref(),
        requested_backend.as_deref(),
        selected_backend.as_deref(),
        runtime_api.as_deref(),
        backend_lane.as_deref(),
    ) {
        return OpenVinoExplanation::default();
    }

    let runtime_device = string_at(receipt, &["runtime_device"])
        .or_else(|| string_at(receipt, &["device"]))
        .or_else(|| string_at(receipt, &["backend", "runtime_device"]));
    let resolved_device = string_at(receipt, &["resolved_device"])
        .or_else(|| string_at(receipt, &["device_name"]))
        .or_else(|| string_at(receipt, &["backend", "resolved_device"]));
    let proof_family = string_at(receipt, &["proof_family"])
        .or_else(|| backend_lane.clone())
        .or_else(|| route_id.clone());
    let proof_stage = string_at(receipt, &["proof_stage"]);
    let selected_kernel_or_runtime = string_at(receipt, &["selected_kernel_or_runtime"])
        .or_else(|| string_at(receipt, &["selected_kernel"]))
        .or_else(|| string_at(receipt, &["runtime", "selected_kernel_or_runtime"]));
    let route_reason = string_at(receipt, &["route", "route_reason"])
        .or_else(|| string_at(receipt, &["route_reason"]));
    let timing_scope = string_at(receipt, &["timing", "timing_scope"])
        .or_else(|| string_at(receipt, &["timing_scope"]))
        .or_else(|| string_at(receipt, &["comparison_scope"]))
        .or_else(|| infer_openvino_timing_scope(receipt));
    let promotion_status = string_at(receipt, &["promotion_status"])
        .or_else(|| string_at(receipt, &["route", "promotion_status"]))
        .or_else(|| string_at(receipt, &["route_status"]))
        .or_else(|| infer_openvino_promotion_status(route_id.as_deref()));
    let quality_status = openvino_quality_status(receipt);
    let blockers = openvino_blockers(receipt, route_id.as_deref(), selected_backend.as_deref());
    let does_not_prove = openvino_does_not_prove(
        route_id.as_deref(),
        selected_backend.as_deref(),
        promotion_status.as_deref(),
        receipt,
    );

    OpenVinoExplanation {
        route_id,
        route_reason,
        requested_backend,
        selected_backend,
        runtime_api,
        runtime_device,
        resolved_device,
        proof_family,
        proof_stage,
        backend_lane,
        selected_kernel_or_runtime,
        quality_status,
        timing_scope,
        promotion_status,
        blockers,
        does_not_prove,
    }
}

fn is_openvino_receipt(
    artifact_kind: Option<&str>,
    route_id: Option<&str>,
    requested_backend: Option<&str>,
    selected_backend: Option<&str>,
    runtime_api: Option<&str>,
    backend_lane: Option<&str>,
) -> bool {
    [artifact_kind, route_id, requested_backend, selected_backend, runtime_api, backend_lane]
        .into_iter()
        .flatten()
        .any(|value| value.to_ascii_lowercase().contains("openvino"))
}

fn infer_openvino_timing_scope(receipt: &Value) -> Option<String> {
    if value_at(receipt, &["timing", "openvino_perf_metrics"]).is_some()
        || f64_at(receipt, &["timing", "pipeline_construct_wall_ms"]).is_some()
        || f64_at(receipt, &["timing", "generation_wall_ms"]).is_some()
    {
        return Some("openvino_pipeline_construct_and_generation_wall_time".to_string());
    }
    if value_at(receipt, &["generation", "devices"]).is_some() {
        return Some("openvino_multi_device_generation_summary".to_string());
    }
    None
}

fn infer_openvino_promotion_status(route_id: Option<&str>) -> Option<String> {
    let route_id = route_id?;
    if route_id.contains("candidate") {
        Some("candidate".to_string())
    } else if route_id.contains("promoted") || route_id == "dense_slm_default_cpu" {
        Some("promoted".to_string())
    } else {
        None
    }
}

fn openvino_quality_status(receipt: &Value) -> Option<String> {
    if let Some(passed) = bool_at(receipt, &["answer_gate", "passed"]) {
        return Some(if passed { "answer_gate_passed" } else { "answer_gate_failed" }.to_string());
    }
    if let Some(passed) = bool_at(receipt, &["quality_gate", "passed"]) {
        return Some(
            if passed { "quality_gate_passed" } else { "quality_gate_failed" }.to_string(),
        );
    }
    if let Some(status) = string_at(receipt, &["profile_quality", "status"]) {
        return Some(status);
    }
    if let Some(passed) = bool_at(receipt, &["generation", "all_answer_gates_passed"]) {
        return Some(
            if passed { "all_answer_gates_passed" } else { "answer_gate_failures_present" }
                .to_string(),
        );
    }
    None
}

fn openvino_blockers(
    receipt: &Value,
    route_id: Option<&str>,
    selected_backend: Option<&str>,
) -> Vec<String> {
    let mut blockers = BTreeSet::new();
    extend_string_set(&mut blockers, string_array_at(receipt, &["blockers"]));
    extend_string_set(&mut blockers, string_array_at(receipt, &["route", "blockers"]));
    extend_string_set(&mut blockers, string_array_at(receipt, &["known_gaps"]));
    extend_string_set(&mut blockers, string_array_at(receipt, &["timing", "known_gaps"]));
    extend_string_set(&mut blockers, string_array_at(receipt, &["profile_quality", "notes"]));

    if route_id.is_some_and(|route| route.contains("candidate")) {
        blockers.insert(
            "route remains candidate until exact-profile promotion evidence exists".to_string(),
        );
    }
    if bool_at(receipt, &["output", "generated_token_ids_available_from_pipeline"]) == Some(false)
        || bool_at(
            receipt,
            &["environment", "transformers", "generated_token_ids_available_from_pipeline"],
        ) == Some(false)
    {
        blockers
            .insert("direct generated token IDs are unavailable from OpenVINO GenAI".to_string());
    }
    if selected_backend == Some("openvino-npu") {
        blockers.insert("NPU promotion requires cache plus warm/resident evidence".to_string());
    }
    blockers.into_iter().collect()
}

fn openvino_does_not_prove(
    route_id: Option<&str>,
    selected_backend: Option<&str>,
    promotion_status: Option<&str>,
    receipt: &Value,
) -> Vec<String> {
    let mut limits = BTreeSet::new();
    limits.insert("BitNet packed I2_S/QK256 proof".to_string());
    limits.insert("full BitNet accelerator inference".to_string());
    limits.insert("QK256 accelerator decode".to_string());

    if selected_backend == Some("openvino-gpu") {
        limits.insert("native OpenCL execution proof".to_string());
    }
    if selected_backend == Some("openvino-npu") {
        limits.insert("native NPU kernel execution".to_string());
        limits.insert("NPU cold one-off usability".to_string());
        limits.insert("dynamic decode, beam search, or parallel sampling on NPU".to_string());
    }
    if promotion_status != Some("promoted")
        || route_id.is_some_and(|route| route.contains("candidate"))
    {
        limits.insert("route promotion".to_string());
    }
    if bool_at(receipt, &["route", "acceleration_claim"]) == Some(false)
        || bool_at(receipt, &["acceleration_claim"]) == Some(false)
    {
        limits.insert("acceleration claim".to_string());
    }
    if bool_at(receipt, &["speedup_claim"]) == Some(false)
        || bool_at(receipt, &["route", "speedup_claim"]) == Some(false)
        || bool_at(receipt, &["claim_boundary", "speedup_claim"]) == Some(false)
    {
        limits.insert("speedup claim".to_string());
    }
    limits.into_iter().collect()
}

fn extend_string_set(values: &mut BTreeSet<String>, entries: Vec<String>) {
    values.extend(entries.into_iter().filter(|entry| !entry.trim().is_empty()));
}

fn model_coverage_explanation(
    explanation: &ReceiptExplanation,
    receipt: &Value,
) -> ModelCoverageExplanation {
    let mut coverage = ModelCoverageExplanation::default();
    let Some(matrix_path) = find_model_coverage_matrix() else {
        coverage.warnings.push(format!(
            "model coverage matrix not found; run from the BitNet-rs repo or set BITNET_MODEL_COVERAGE_MATRIX to {}",
            MODEL_COVERAGE_MATRIX_RELATIVE.join("/")
        ));
        return coverage;
    };
    coverage.source = Some(matrix_path.display().to_string());

    let matrix = match read_model_coverage_matrix(&matrix_path) {
        Ok(matrix) => matrix,
        Err(err) => {
            coverage.warnings.push(format!("model coverage matrix unavailable: {err}"));
            return coverage;
        }
    };

    if let Some(entry) = match_model_coverage_entry(&matrix, explanation, receipt) {
        coverage.row = Some(entry.id.clone());
        coverage.current_tier = Some(entry.current_tier.clone());
        coverage.status = Some(entry.status.clone());
        coverage.route = explanation
            .execution_plan
            .selected_route
            .clone()
            .or_else(|| entry.accelerator_routes.first().cloned());
        coverage.product_cli_ready = Some(entry.claims.product_cli_ready);
        coverage.speedup_claim = Some(entry.claims.speedup_claim);
        coverage.benchmark_qualified = Some(entry.claims.benchmark_qualified);
        coverage.server_ready = Some(entry.claims.server_ready);
        coverage.server_ready_scope = model_coverage_server_ready_scope(entry);
        coverage.full_residency_claim = Some(entry.claims.full_residency_claim);
        coverage.bitnet_packed_i2s_qk256_proof = Some(entry.claims.bitnet_packed_i2s_qk256_proof);
        coverage.dense_regular_llm_cuda_proof = Some(entry.claims.dense_regular_llm_cuda_proof);
        coverage.claim_boundary = Some(entry.claim_boundary.clone());
        add_model_coverage_warnings(&mut coverage, entry, explanation, receipt);
    } else {
        coverage.warnings.push("no model coverage row matched this receipt".to_string());
    }

    coverage
}

fn model_coverage_server_ready_scope(entry: &ModelCoverageEntry) -> Option<String> {
    entry.claims.server_ready.then(|| "exact_profile".to_string())
}

fn find_model_coverage_matrix() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("BITNET_MODEL_COVERAGE_MATRIX").map(PathBuf::from)
        && path.exists()
    {
        return Some(path);
    }
    if let Ok(current_dir) = std::env::current_dir()
        && let Some(path) = find_model_coverage_matrix_from(&current_dir)
    {
        return Some(path);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
        && let Some(path) = find_model_coverage_matrix_from(parent)
    {
        return Some(path);
    }
    None
}

fn find_model_coverage_matrix_from(start: &Path) -> Option<PathBuf> {
    for ancestor in start.ancestors() {
        let mut candidate = ancestor.to_path_buf();
        for segment in MODEL_COVERAGE_MATRIX_RELATIVE {
            candidate.push(segment);
        }
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn read_model_coverage_matrix(path: &Path) -> Result<ModelCoverageMatrix> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let matrix: ModelCoverageMatrix = toml::from_str(&text)
        .with_context(|| format!("failed to parse model coverage matrix {}", path.display()))?;
    if matrix.schema != 1 {
        bail!("unsupported model coverage schema {}", matrix.schema);
    }
    if matrix.artifact_kind != "model_coverage_matrix" {
        bail!("expected artifact_kind=model_coverage_matrix, got {}", matrix.artifact_kind);
    }
    Ok(matrix)
}

fn match_model_coverage_entry<'a>(
    matrix: &'a ModelCoverageMatrix,
    explanation: &ReceiptExplanation,
    receipt: &Value,
) -> Option<&'a ModelCoverageEntry> {
    if let Some(explicit_row) = explicit_model_coverage_row(receipt)
        && let Some(entry) =
            matrix.entry.iter().find(|entry| entry.id.eq_ignore_ascii_case(&explicit_row))
    {
        return Some(entry);
    }

    let search_text = receipt_search_text(receipt, explanation);
    let route = explanation.execution_plan.selected_route.as_deref();
    if route == Some("dense_regular_llm_cuda")
        && (search_text.contains("qwen3")
            || search_text.contains("qwen3-0.6b-instruct-q8_0")
            || search_text.contains("qwen3-0.6b-q8_0.gguf"))
    {
        return matrix.entry.iter().find(|entry| entry.id == "dense_qwen3_06b_q8_candidate");
    }

    if route == Some("dense_regular_llm_cuda")
        && (search_text.contains("qwen")
            || search_text.contains("qwen25")
            || search_text.contains("qwen2.5"))
    {
        return matrix.entry.iter().find(|entry| entry.id == "dense_qwen25_05b_q8_cuda");
    }

    if route == Some("bitnet_qk256_cuda")
        || explanation.claim_limits.bitnet_packed_i2s_qk256_proof == Some(true)
        || (search_text.contains("bitnet")
            && (search_text.contains("ggml-model-i2_s.gguf")
                || search_text.contains("i2_s")
                || search_text.contains("qk256")))
    {
        return matrix.entry.iter().find(|entry| {
            entry.id == "bitnet_official_2b_i2s_qk256"
                || (entry.claims.bitnet_packed_i2s_qk256_proof
                    && entry
                        .accelerator_routes
                        .iter()
                        .any(|candidate| candidate == "bitnet_qk256_cuda"))
        });
    }

    None
}

fn explicit_model_coverage_row(receipt: &Value) -> Option<String> {
    string_at(receipt, &["model_coverage_row"])
        .or_else(|| string_at(receipt, &["model_coverage", "row"]))
        .or_else(|| string_at(receipt, &["model_coverage", "id"]))
}

fn receipt_search_text(receipt: &Value, explanation: &ReceiptExplanation) -> String {
    let mut parts = Vec::new();
    collect_json_strings(receipt, &mut parts);
    if let Some(model) = &explanation.model {
        parts.push(model.clone());
    }
    if let Some(route) = &explanation.execution_plan.selected_route {
        parts.push(route.clone());
    }
    if let Some(kind) = &explanation.artifact_kind {
        parts.push(kind.clone());
    }
    if let Some(claim) = &explanation.claim {
        parts.push(claim.clone());
    }
    parts.join(" ").to_ascii_lowercase()
}

fn collect_json_strings(value: &Value, parts: &mut Vec<String>) {
    match value {
        Value::String(value) => parts.push(value.clone()),
        Value::Array(values) => {
            for value in values {
                collect_json_strings(value, parts);
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                collect_json_strings(value, parts);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn add_model_coverage_warnings(
    coverage: &mut ModelCoverageExplanation,
    entry: &ModelCoverageEntry,
    explanation: &ReceiptExplanation,
    receipt: &Value,
) {
    if !entry.claims.speedup_claim {
        coverage.warnings.push("speedup is not qualified by the model coverage row".to_string());
    }
    if !entry.claims.server_ready {
        coverage
            .warnings
            .push("server readiness is not claimed by the model coverage row".to_string());
    }
    if entry.claims.dense_regular_llm_cuda_proof && !entry.claims.bitnet_packed_i2s_qk256_proof {
        coverage
            .warnings
            .push("dense regular-LLM CUDA proof is not BitNet packed I2_S/QK256 proof".to_string());
    }
    if entry.claims.bitnet_packed_i2s_qk256_proof && !entry.claims.dense_regular_llm_cuda_proof {
        coverage
            .warnings
            .push("BitNet packed I2_S/QK256 proof is not dense SLM CUDA proof".to_string());
    }
    if let Some(route) = &explanation.execution_plan.selected_route
        && !entry.accelerator_routes.is_empty()
        && !entry.accelerator_routes.iter().any(|candidate| candidate == route)
    {
        coverage.warnings.push(format!(
            "receipt route `{route}` does not match model coverage routes: {}",
            entry.accelerator_routes.join(", ")
        ));
    }
    add_server_shared_engine_warnings(coverage, entry, receipt);
}

fn add_server_shared_engine_warnings(
    coverage: &mut ModelCoverageExplanation,
    entry: &ModelCoverageEntry,
    receipt: &Value,
) {
    if string_at(receipt, &["receipt_kind"]).as_deref()
        != Some("server_shared_engine_chat_completion")
    {
        return;
    }

    let model_identity_sha256 = string_at(receipt, &["model_identity", "model_sha256"]);
    let top_level_sha256 = string_at(receipt, &["model_sha256"]);
    let checksum_identity_matches =
        match (model_identity_sha256.as_deref(), top_level_sha256.as_deref()) {
            (Some(identity), Some(top_level)) => {
                is_sha256_hex(identity) && is_sha256_hex(top_level) && identity == top_level
            }
            _ => false,
        };
    if !checksum_identity_matches {
        coverage.warnings.push(
            "server shared-engine receipt is missing exact artifact checksum identity".to_string(),
        );
    }

    if string_at(receipt, &["endpoint_profile", "endpoint"]).as_deref()
        != Some("/v1/chat/completions")
        || string_at(receipt, &["endpoint_profile", "method"]).as_deref() != Some("POST")
        || string_at(receipt, &["endpoint_profile", "request_profile"]).is_none()
    {
        coverage.warnings.push(
            "server shared-engine receipt is missing endpoint/request profile scope".to_string(),
        );
    }

    if u64_at(receipt, &["generation_policy", "max_tokens"]).is_none()
        || f64_at(receipt, &["generation_policy", "temperature"]).is_none()
        || f64_at(receipt, &["generation_policy", "top_p"]).is_none()
        || string_at(receipt, &["generation_policy", "decoding"]).is_none()
    {
        coverage
            .warnings
            .push("server shared-engine receipt is missing generation-policy scope".to_string());
    }

    if bool_at(receipt, &["server_ready_claimed"]) == Some(true) && !entry.claims.server_ready {
        coverage.warnings.push(
            "receipt claims server readiness, but the model coverage row does not promote server_ready"
                .to_string(),
        );
    }
}

fn model_summary(receipt: &Value) -> Option<String> {
    let model = receipt.get("model")?;
    if let Some(repo) = model.get("repo").and_then(Value::as_str) {
        let file = model
            .get("file")
            .and_then(Value::as_str)
            .or_else(|| model.get("filename").and_then(Value::as_str));
        return Some(match file {
            Some(file) => format!("{repo} / {file}"),
            None => repo.to_string(),
        });
    }
    if let Some(file) = model.get("file").and_then(Value::as_str) {
        return Some(file.to_string());
    }
    if let Some(id) = model.get("id").and_then(Value::as_str) {
        return Some(id.to_string());
    }
    None
}

fn kernel_ids(receipt: &Value) -> Vec<String> {
    let mut ids = BTreeSet::new();
    if let Some(id) = string_at(receipt, &["kernel", "selected_kernel"]) {
        ids.insert(id);
    }
    if let Some(id) = string_at(receipt, &["selected_kernel"]) {
        ids.insert(id);
    }
    collect_kernel_ids_from_array(receipt.get("kernel_stats"), &mut ids);
    collect_kernel_ids_from_array(receipt.get("kernels"), &mut ids);
    ids.into_iter().collect()
}

fn collect_kernel_ids_from_array(value: Option<&Value>, ids: &mut BTreeSet<String>) {
    let Some(entries) = value.and_then(Value::as_array) else {
        return;
    };
    for entry in entries {
        if let Some(id) = entry
            .get("kernel_id")
            .or_else(|| entry.get("selected_kernel"))
            .or_else(|| entry.get("name"))
            .and_then(Value::as_str)
        {
            ids.insert(id.to_string());
        }
    }
}

pub fn compact_proof_lines(explanation: &ReceiptExplanation) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push("Proof:".to_string());

    if let Some(model) = &explanation.model {
        lines.push(format!("  model: {model}"));
    }
    if let Some(row) = &explanation.model_coverage.row {
        lines.push(format!("  model coverage row: {row}"));
    }
    if let Some(route) = &explanation.execution_plan.selected_route {
        lines.push(format!("  route: {route}"));
    }
    if let Some(backend) = &explanation.backend.selected_backend {
        lines.push(format!("  backend: {backend}"));
    }
    if let Some(runtime) = &explanation.backend.runtime_api {
        lines.push(format!("  runtime: {runtime}"));
    }
    if let Some(device) = &explanation.openvino.runtime_device {
        lines.push(format!("  device: {device}"));
    }
    if !explanation.kernels.is_empty() {
        lines.push(format!("  kernel: {}", explanation.kernels.join(", ")));
    } else if let Some(runtime) = &explanation.openvino.selected_kernel_or_runtime {
        lines.push(format!("  runtime id: {runtime}"));
    }
    if let Some(fallback) = explanation.backend.fallback_used {
        lines.push(format!("  fallback: {fallback}"));
    }
    if let Some(quality) = explanation
        .quality
        .answer_quality_passed
        .or(explanation.quality.benchmark_quality_passed)
        .or(explanation.quality.parity_passed)
    {
        lines.push(format!("  quality: {quality}"));
    }
    if let Some(weights_uploaded_once) = explanation.residency.weights_uploaded_once {
        let weight_text =
            if weights_uploaded_once { "uploaded once" } else { "not upload-once proven" };
        lines.push(format!("  weights: {weight_text}"));
    }
    if let Some(kernel_time_ms) = explanation.timing.kernel_time_ms {
        lines.push(format!("  kernel time: {kernel_time_ms:.3} ms"));
    }
    if explanation.timing.host_to_device_bytes.is_some()
        || explanation.timing.device_to_host_bytes.is_some()
    {
        let h2d = explanation
            .timing
            .host_to_device_bytes
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let d2h = explanation
            .timing
            .device_to_host_bytes
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        lines.push(format!("  transfers: h2d={h2d} bytes d2h={d2h} bytes"));
    }
    if let Some(full_residency) = explanation.residency.full_cuda_residency_claimed {
        lines.push(format!("  full cuda residency: {full_residency}"));
    }
    let speedup_claim =
        explanation.claim_limits.speedup_claim.or(explanation.execution_plan.speedup_claim);
    if let Some(speedup_claim) = speedup_claim {
        lines.push(format!("  speed claim: {speedup_claim}"));
    }
    if let Some(status) = &explanation.openvino.promotion_status {
        lines.push(format!("  OpenVINO promotion: {status}"));
    }
    if !explanation.openvino.does_not_prove.is_empty() {
        lines.push(format!("  does not prove: {}", explanation.openvino.does_not_prove.join("; ")));
    }
    lines.push(format!("  receipt: {}", explanation.path));

    lines
}

pub fn print_compact_proof_summary(explanation: &ReceiptExplanation) {
    for line in compact_proof_lines(explanation) {
        println!("{line}");
    }
}

fn print_receipt_explanation(explanation: &ReceiptExplanation) {
    println!("Receipt: {}", explanation.path);
    print_option("Artifact", explanation.artifact_kind.as_deref());
    print_option("Claim", explanation.claim.as_deref());
    print_option("Model", explanation.model.as_deref());

    if has_model_coverage(&explanation.model_coverage) {
        println!();
        println!("Model Coverage:");
        print_option_indented("source", explanation.model_coverage.source.as_deref());
        print_option_indented("row", explanation.model_coverage.row.as_deref());
        print_option_indented("current_tier", explanation.model_coverage.current_tier.as_deref());
        print_option_indented("status", explanation.model_coverage.status.as_deref());
        print_option_indented("route", explanation.model_coverage.route.as_deref());
        print_bool_indented("product_cli_ready", explanation.model_coverage.product_cli_ready);
        print_bool_indented("speedup_claim", explanation.model_coverage.speedup_claim);
        print_bool_indented("benchmark_qualified", explanation.model_coverage.benchmark_qualified);
        print_bool_indented("server_ready", explanation.model_coverage.server_ready);
        print_option_indented(
            "server_ready_scope",
            explanation.model_coverage.server_ready_scope.as_deref(),
        );
        print_bool_indented(
            "bitnet_packed_i2s_qk256_proof",
            explanation.model_coverage.bitnet_packed_i2s_qk256_proof,
        );
        print_bool_indented(
            "dense_regular_llm_cuda_proof",
            explanation.model_coverage.dense_regular_llm_cuda_proof,
        );
        print_option_indented(
            "claim_boundary",
            explanation.model_coverage.claim_boundary.as_deref(),
        );
        print_string_list_indented("warnings", &explanation.model_coverage.warnings);
    }

    println!();
    println!("Backend:");
    print_option_indented("requested", explanation.backend.requested_backend.as_deref());
    print_option_indented("selected", explanation.backend.selected_backend.as_deref());
    print_option_indented("runtime", explanation.backend.runtime_api.as_deref());
    print_bool_indented("fallback", explanation.backend.fallback_used);

    if has_execution_plan(&explanation.execution_plan) {
        println!();
        println!("Execution Plan:");
        print_option_indented("route", explanation.execution_plan.selected_route.as_deref());
        print_option_indented("model_family", explanation.execution_plan.model_family.as_deref());
        print_option_indented("quantization", explanation.execution_plan.quantization.as_deref());
        print_bool_indented("strict_cuda_ready", explanation.execution_plan.strict_cuda_ready);
        print_bool_indented("speedup_claim", explanation.execution_plan.speedup_claim);
        print_bool_indented(
            "full_cuda_residency_claimed",
            explanation.execution_plan.full_cuda_residency_claimed,
        );
    }

    if !explanation.kernels.is_empty() {
        println!();
        println!("Kernels:");
        for kernel in &explanation.kernels {
            println!("  - {kernel}");
        }
    }

    if has_quality(&explanation.quality) {
        println!();
        println!("Quality:");
        print_bool_indented("answer_quality_passed", explanation.quality.answer_quality_passed);
        print_bool_indented(
            "benchmark_quality_passed",
            explanation.quality.benchmark_quality_passed,
        );
        print_bool_indented("parity_passed", explanation.quality.parity_passed);
        print_option_indented("first_divergence", explanation.quality.first_divergence.as_deref());
    }

    if has_timing(&explanation.timing) {
        println!();
        println!("Timing:");
        print_f64_indented("total_ms", explanation.timing.total_ms);
        print_f64_indented("first_token_ms", explanation.timing.first_token_ms);
        print_f64_indented("decode_total_ms", explanation.timing.decode_total_ms);
        print_f64_indented("steady_decode_tok_s", explanation.timing.steady_decode_tok_s);
        print_f64_indented("kernel_time_ms", explanation.timing.kernel_time_ms);
        print_u64_indented("host_to_device_bytes", explanation.timing.host_to_device_bytes);
        print_u64_indented("device_to_host_bytes", explanation.timing.device_to_host_bytes);
    }

    if has_residency(&explanation.residency) {
        println!();
        println!("Residency:");
        print_bool_indented(
            "qk256_cuda_residency_claimed",
            explanation.residency.qk256_cuda_residency_claimed,
        );
        print_bool_indented("weights_uploaded_once", explanation.residency.weights_uploaded_once);
        print_bool_indented("model_loaded_once", explanation.residency.model_loaded_once);
        print_bool_indented("cuda_context_once", explanation.residency.cuda_context_once);
        print_bool_indented("per_request_model_load", explanation.residency.per_request_model_load);
        print_bool_indented(
            "per_token_weight_upload",
            explanation.residency.per_token_weight_upload,
        );
        print_bool_indented("workspace_reused", explanation.residency.workspace_reused);
        print_option_indented("kv_cache", explanation.residency.kv_cache_residency.as_deref());
        print_bool_indented(
            "full_cuda_residency_claimed",
            explanation.residency.full_cuda_residency_claimed,
        );
    }

    if has_benchmark_qualification(&explanation.benchmark_qualification) {
        println!();
        println!("Benchmark Qualification:");
        print_option_indented("status", explanation.benchmark_qualification.status.as_deref());
        print_bool_indented(
            "benchmark_qualified_speedup",
            explanation.benchmark_qualification.benchmark_qualified_speedup,
        );
        print_bool_indented(
            "speedup_claim_allowed",
            explanation.benchmark_qualification.speedup_claim_allowed,
        );
        print_string_list_indented(
            "accepted_profiles",
            &explanation.benchmark_qualification.accepted_profiles,
        );
        print_string_list_indented(
            "blocked_profiles",
            &explanation.benchmark_qualification.blocked_profiles,
        );
        print_option_indented(
            "transfer_timing_status",
            explanation.benchmark_qualification.transfer_timing_status.as_deref(),
        );
        print_option_indented(
            "host_to_device_source",
            explanation.benchmark_qualification.host_to_device_source.as_deref(),
        );
        print_option_indented(
            "host_to_device_scope",
            explanation.benchmark_qualification.host_to_device_scope.as_deref(),
        );
        print_bool_indented(
            "host_to_device_includes_non_transfer_overhead",
            explanation.benchmark_qualification.host_to_device_includes_non_transfer_overhead,
        );
        print_bool_indented(
            "pure_host_to_device_timing_recorded",
            explanation.benchmark_qualification.pure_host_to_device_timing_recorded,
        );
        print_bool_indented(
            "device_to_host_timing_recorded",
            explanation.benchmark_qualification.device_to_host_timing_recorded,
        );
        if !explanation.benchmark_qualification.profile_reviews.is_empty() {
            println!("  profiles:");
            for profile in &explanation.benchmark_qualification.profile_reviews {
                print_benchmark_profile(profile);
            }
        }
    }

    if has_openvino(&explanation.openvino) {
        println!();
        println!("OpenVINO:");
        print_option_indented("route_id", explanation.openvino.route_id.as_deref());
        print_option_indented("route_reason", explanation.openvino.route_reason.as_deref());
        print_option_indented(
            "requested_backend",
            explanation.openvino.requested_backend.as_deref(),
        );
        print_option_indented("selected_backend", explanation.openvino.selected_backend.as_deref());
        print_option_indented("runtime_api", explanation.openvino.runtime_api.as_deref());
        print_option_indented("runtime_device", explanation.openvino.runtime_device.as_deref());
        print_option_indented("resolved_device", explanation.openvino.resolved_device.as_deref());
        print_option_indented("proof_family", explanation.openvino.proof_family.as_deref());
        print_option_indented("proof_stage", explanation.openvino.proof_stage.as_deref());
        print_option_indented("backend_lane", explanation.openvino.backend_lane.as_deref());
        print_option_indented(
            "selected_kernel_or_runtime",
            explanation.openvino.selected_kernel_or_runtime.as_deref(),
        );
        print_option_indented("quality_status", explanation.openvino.quality_status.as_deref());
        print_option_indented("timing_scope", explanation.openvino.timing_scope.as_deref());
        print_option_indented("promotion_status", explanation.openvino.promotion_status.as_deref());
        print_string_list_indented("blockers", &explanation.openvino.blockers);
        print_string_list_indented("does_not_prove", &explanation.openvino.does_not_prove);
    }

    println!();
    println!("Claim Limits:");
    print_bool_indented("speedup_claim", explanation.claim_limits.speedup_claim);
    print_bool_indented(
        "benchmark_qualified_speedup",
        explanation.claim_limits.benchmark_qualified_speedup,
    );
    print_bool_indented(
        "full_cuda_residency_claimed",
        explanation.claim_limits.full_cuda_residency_claimed,
    );
    print_bool_indented(
        "dense_gguf_inference_claimed",
        explanation.claim_limits.dense_gguf_inference_claimed,
    );
    print_bool_indented(
        "bitnet_packed_i2s_qk256_proof",
        explanation.claim_limits.bitnet_packed_i2s_qk256_proof,
    );
}

fn print_option(label: &str, value: Option<&str>) {
    if let Some(value) = value {
        println!("{label}: {value}");
    }
}

fn print_option_indented(label: &str, value: Option<&str>) {
    if let Some(value) = value {
        println!("  {label}: {value}");
    }
}

fn print_bool_indented(label: &str, value: Option<bool>) {
    if let Some(value) = value {
        println!("  {label}: {value}");
    }
}

fn print_f64_indented(label: &str, value: Option<f64>) {
    if let Some(value) = value {
        println!("  {label}: {value:.3}");
    }
}

fn print_u64_indented(label: &str, value: Option<u64>) {
    if let Some(value) = value {
        println!("  {label}: {value}");
    }
}

fn print_string_list_indented(label: &str, values: &[String]) {
    if !values.is_empty() {
        println!("  {label}: {}", values.join(", "));
    }
}

fn print_benchmark_profile(profile: &BenchmarkProfileExplanation) {
    println!("    - profile: {}", profile.profile);
    print_option_profile("decision", profile.decision.as_deref());
    print_bool_profile("benchmark_qualified_speedup", profile.benchmark_qualified_speedup);
    print_bool_profile("speedup_claim_allowed", profile.speedup_claim_allowed);
    print_bool_profile("fallback_free", profile.fallback_free);
    print_bool_profile("quality_passed", profile.quality_passed);
    print_bool_profile("generated_token_ids_match", profile.generated_token_ids_match);
    print_f64_profile("cpu_total_ms_mean", profile.cpu_total_ms_mean);
    print_f64_profile("cuda_total_ms_mean", profile.cuda_total_ms_mean);
    print_f64_profile(
        "observed_cpu_total_ms_div_cuda_total_ms",
        profile.observed_cpu_total_ms_div_cuda_total_ms,
    );
    print_f64_profile("host_to_device_ms", profile.host_to_device_ms);
    print_option_profile("host_to_device_ms_source", profile.host_to_device_ms_source.as_deref());
    print_option_profile("host_to_device_ms_scope", profile.host_to_device_ms_scope.as_deref());
    print_bool_profile(
        "host_to_device_ms_includes_non_transfer_overhead",
        profile.host_to_device_ms_includes_non_transfer_overhead,
    );
    print_option_profile(
        "pure_host_to_device_ms_source",
        profile.pure_host_to_device_ms_source.as_deref(),
    );
    print_f64_profile("device_to_host_ms", profile.device_to_host_ms);
    print_option_profile("device_to_host_ms_source", profile.device_to_host_ms_source.as_deref());
    if !profile.blockers.is_empty() {
        println!("      blockers: {}", profile.blockers.join("; "));
    }
}

fn print_option_profile(label: &str, value: Option<&str>) {
    if let Some(value) = value {
        println!("      {label}: {value}");
    }
}

fn print_bool_profile(label: &str, value: Option<bool>) {
    if let Some(value) = value {
        println!("      {label}: {value}");
    }
}

fn print_f64_profile(label: &str, value: Option<f64>) {
    if let Some(value) = value {
        println!("      {label}: {value:.3}");
    }
}

fn has_execution_plan(plan: &ExecutionPlanExplanation) -> bool {
    plan.selected_route.is_some()
        || plan.model_family.is_some()
        || plan.quantization.is_some()
        || plan.strict_cuda_ready.is_some()
        || plan.speedup_claim.is_some()
        || plan.full_cuda_residency_claimed.is_some()
}

fn has_quality(quality: &QualityExplanation) -> bool {
    quality.answer_quality_passed.is_some()
        || quality.benchmark_quality_passed.is_some()
        || quality.parity_passed.is_some()
        || quality.first_divergence.is_some()
}

fn has_timing(timing: &TimingExplanation) -> bool {
    timing.total_ms.is_some()
        || timing.first_token_ms.is_some()
        || timing.decode_total_ms.is_some()
        || timing.steady_decode_tok_s.is_some()
        || timing.kernel_time_ms.is_some()
        || timing.host_to_device_bytes.is_some()
        || timing.device_to_host_bytes.is_some()
}

fn has_residency(residency: &ResidencyExplanation) -> bool {
    residency.qk256_cuda_residency_claimed.is_some()
        || residency.model_loaded_once.is_some()
        || residency.cuda_context_once.is_some()
        || residency.weights_uploaded_once.is_some()
        || residency.per_request_model_load.is_some()
        || residency.per_token_weight_upload.is_some()
        || residency.workspace_reused.is_some()
        || residency.kv_cache_residency.is_some()
        || residency.full_cuda_residency_claimed.is_some()
}

fn has_benchmark_qualification(qualification: &BenchmarkQualificationExplanation) -> bool {
    qualification.status.is_some()
        || qualification.benchmark_qualified_speedup.is_some()
        || !qualification.accepted_profiles.is_empty()
        || !qualification.blocked_profiles.is_empty()
        || qualification.speedup_claim_allowed.is_some()
        || qualification.transfer_timing_status.is_some()
        || qualification.host_to_device_source.is_some()
        || qualification.host_to_device_scope.is_some()
        || qualification.host_to_device_includes_non_transfer_overhead.is_some()
        || qualification.pure_host_to_device_timing_recorded.is_some()
        || qualification.device_to_host_timing_recorded.is_some()
        || !qualification.profile_reviews.is_empty()
}

fn has_openvino(openvino: &OpenVinoExplanation) -> bool {
    openvino.route_id.is_some()
        || openvino.route_reason.is_some()
        || openvino.requested_backend.is_some()
        || openvino.selected_backend.is_some()
        || openvino.runtime_api.is_some()
        || openvino.runtime_device.is_some()
        || openvino.resolved_device.is_some()
        || openvino.proof_family.is_some()
        || openvino.proof_stage.is_some()
        || openvino.backend_lane.is_some()
        || openvino.selected_kernel_or_runtime.is_some()
        || openvino.quality_status.is_some()
        || openvino.timing_scope.is_some()
        || openvino.promotion_status.is_some()
        || !openvino.blockers.is_empty()
        || !openvino.does_not_prove.is_empty()
}

fn has_model_coverage(coverage: &ModelCoverageExplanation) -> bool {
    coverage.source.is_some()
        || coverage.row.is_some()
        || coverage.current_tier.is_some()
        || coverage.status.is_some()
        || coverage.route.is_some()
        || coverage.product_cli_ready.is_some()
        || coverage.speedup_claim.is_some()
        || coverage.benchmark_qualified.is_some()
        || coverage.server_ready.is_some()
        || coverage.server_ready_scope.is_some()
        || coverage.bitnet_packed_i2s_qk256_proof.is_some()
        || coverage.dense_regular_llm_cuda_proof.is_some()
        || coverage.claim_boundary.is_some()
        || !coverage.warnings.is_empty()
}

fn string_at(value: &Value, path: &[&str]) -> Option<String> {
    value_at(value, path).and_then(Value::as_str).map(str::to_string)
}

fn bool_at(value: &Value, path: &[&str]) -> Option<bool> {
    value_at(value, path).and_then(Value::as_bool)
}

fn f64_at(value: &Value, path: &[&str]) -> Option<f64> {
    value_at(value, path).and_then(Value::as_f64)
}

fn u64_at(value: &Value, path: &[&str]) -> Option<u64> {
    value_at(value, path).and_then(Value::as_u64)
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn value_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
}

fn string_array_at(value: &Value, path: &[&str]) -> Vec<String> {
    let Some(entries) = value_at(value, path).and_then(Value::as_array) else {
        return Vec::new();
    };
    entries.iter().filter_map(Value::as_str).map(str::to_string).collect()
}

fn sum_kernel_u64(receipt: &Value, field: &str) -> Option<u64> {
    let entries = receipt.get("kernel_stats")?.as_array()?;
    let mut total = 0u64;
    let mut found = false;
    for entry in entries {
        if let Some(value) = entry.get(field).and_then(Value::as_u64) {
            total = total.saturating_add(value);
            found = true;
        }
    }
    found.then_some(total)
}

fn sum_kernel_f64(receipt: &Value, field: &str) -> Option<f64> {
    let entries = receipt.get("kernel_stats")?.as_array()?;
    let mut total = 0.0f64;
    let mut found = false;
    for entry in entries {
        if let Some(value) = entry.get(field).and_then(Value::as_f64) {
            total += value;
            found = true;
        }
    }
    found.then_some(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use serde_json::{Value, json};

    #[derive(Parser, Debug)]
    struct TestReceiptsCli {
        #[command(subcommand)]
        action: ReceiptsAction,
    }

    struct ExpectedReceiptJson<'a> {
        model_coverage_row: &'a str,
        current_tier: &'a str,
        selected_backend: &'a str,
        selected_route: Option<&'a str>,
        fallback_used: bool,
        product_cli_ready: bool,
        server_ready: bool,
        server_ready_scope: Option<&'a str>,
        speedup_claim: bool,
        full_residency_claim: bool,
        bitnet_packed_i2s_qk256_proof: bool,
        dense_regular_llm_cuda_proof: bool,
    }

    fn assert_receipt_json_contract(
        receipt: &Value,
        expected: ExpectedReceiptJson<'_>,
    ) -> Result<()> {
        let explanation = explain_receipt(Path::new("receipt.json"), receipt);
        let value = serde_json::to_value(&explanation)?;

        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["model_coverage_row"], expected.model_coverage_row);
        assert_eq!(value["current_tier"], expected.current_tier);
        assert_eq!(value["selected_backend"], expected.selected_backend);
        if let Some(route) = expected.selected_route {
            assert_eq!(value["selected_route"], route);
        } else {
            assert!(value["selected_route"].is_null());
        }
        assert_eq!(value["fallback_used"], expected.fallback_used);
        assert_eq!(value["product_cli_ready"], expected.product_cli_ready);
        assert_eq!(value["server_ready"], expected.server_ready);
        if let Some(scope) = expected.server_ready_scope {
            assert_eq!(value["server_ready_scope"], scope);
        } else {
            assert!(value["server_ready_scope"].is_null());
        }
        assert_eq!(value["speedup_claim"], expected.speedup_claim);
        assert_eq!(value["full_residency_claim"], expected.full_residency_claim);
        assert_eq!(value["bitnet_packed_i2s_qk256_proof"], expected.bitnet_packed_i2s_qk256_proof);
        assert_eq!(value["dense_regular_llm_cuda_proof"], expected.dense_regular_llm_cuda_proof);
        Ok(())
    }

    #[test]
    fn receipts_explain_accepts_format_json_alias() {
        let cli =
            TestReceiptsCli::parse_from(["receipts", "explain", "--latest", "--format", "json"]);

        let ReceiptsAction::Explain { latest, json, format, .. } = cli.action;
        assert!(latest);
        assert!(!json);
        assert_eq!(format, Some(ReceiptExplainFormat::Json));
    }

    #[test]
    fn receipts_explain_json_contract_locks_cuda_model_rows() -> Result<()> {
        assert_receipt_json_contract(
            &json!({
                "artifact_kind": "bitnet_cuda_answer",
                "model": {
                    "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
                    "file": "ggml-model-i2_s.gguf"
                },
                "backend": {
                    "selected_backend": "nvidia-rtx-5070-ti-cuda",
                    "runtime_api": "cuda",
                    "fallback_used": false
                },
                "execution_plan": {
                    "selected_route": "bitnet_qk256_cuda",
                    "model_family": "bitnet_b1_58",
                    "speedup_claim": false
                },
                "claim_boundary": {
                    "bitnet_packed_i2s_qk256_proof": true,
                    "dense_gguf_inference_claimed": false
                }
            }),
            ExpectedReceiptJson {
                model_coverage_row: "bitnet_official_2b_i2s_qk256",
                current_tier: "product_cli_ready",
                selected_backend: "nvidia-rtx-5070-ti-cuda",
                selected_route: Some("bitnet_qk256_cuda"),
                fallback_used: false,
                product_cli_ready: true,
                server_ready: false,
                server_ready_scope: None,
                speedup_claim: false,
                full_residency_claim: false,
                bitnet_packed_i2s_qk256_proof: true,
                dense_regular_llm_cuda_proof: false,
            },
        )?;

        assert_receipt_json_contract(
            &json!({
                "artifact_kind": "dense_gguf_qwen_warm_session_strict_cuda_proof",
                "model": {
                    "id": "qwen2.5-0.5b-instruct-q8_0"
                },
                "execution_plan": {
                    "selected_route": "dense_regular_llm_cuda",
                    "model_family": "qwen",
                    "selected_backend": "nvidia-rtx-5070-ti-cuda",
                    "fallback_used": false,
                    "speedup_claim": false
                },
                "claim_boundary": {
                    "bitnet_packed_i2s_qk256_proof": false,
                    "dense_regular_llm_cuda_claimed": true,
                    "server_ready_claimed": false,
                    "speedup_claim": false
                }
            }),
            ExpectedReceiptJson {
                model_coverage_row: "dense_qwen25_05b_q8_cuda",
                current_tier: "product_cli_ready",
                selected_backend: "nvidia-rtx-5070-ti-cuda",
                selected_route: Some("dense_regular_llm_cuda"),
                fallback_used: false,
                product_cli_ready: true,
                server_ready: true,
                server_ready_scope: Some("exact_profile"),
                speedup_claim: false,
                full_residency_claim: false,
                bitnet_packed_i2s_qk256_proof: false,
                dense_regular_llm_cuda_proof: true,
            },
        )?;

        assert_receipt_json_contract(
            &json!({
                "artifact_kind": "dense_gguf_qwen_ask_strict_cuda_proof",
                "model": {
                    "id": "qwen3-0.6b-instruct-q8_0",
                    "file": "Qwen3-0.6B-Q8_0.gguf",
                    "architecture": "qwen3"
                },
                "execution_plan": {
                    "selected_route": "dense_regular_llm_cuda",
                    "model_family": "qwen",
                    "selected_backend": "nvidia-rtx-5070-ti-cuda",
                    "fallback_used": false,
                    "speedup_claim": false
                },
                "claim_boundary": {
                    "bitnet_packed_i2s_qk256_proof": false,
                    "dense_regular_llm_cuda_claimed": true,
                    "server_ready_claimed": false,
                    "speedup_claim": false,
                    "full_cuda_residency_claimed": false
                }
            }),
            ExpectedReceiptJson {
                model_coverage_row: "dense_qwen3_06b_q8_candidate",
                current_tier: "product_cli_ready",
                selected_backend: "nvidia-rtx-5070-ti-cuda",
                selected_route: Some("dense_regular_llm_cuda"),
                fallback_used: false,
                product_cli_ready: true,
                server_ready: true,
                server_ready_scope: Some("exact_profile"),
                speedup_claim: false,
                full_residency_claim: false,
                bitnet_packed_i2s_qk256_proof: false,
                dense_regular_llm_cuda_proof: true,
            },
        )?;

        assert_receipt_json_contract(
            &json!({
                "artifact_kind": "smollm2_same_prompt_comparator_blocker",
                "model_coverage_row": "dense_smollm2_360m_candidate",
                "model": {
                    "id": "smollm2-360m-instruct",
                    "architecture": "smollm2"
                },
                "selected_backend": "nvidia-rtx-5070-ti-cuda",
                "fallback_used": false,
                "quality_gate": {
                    "passed": false,
                    "blocker": "same-prompt comparator required"
                },
                "claim_boundary": {
                    "bitnet_packed_i2s_qk256_proof": false,
                    "dense_regular_llm_cuda_claimed": false,
                    "server_ready_claimed": false,
                    "speedup_claim": false,
                    "full_cuda_residency_claimed": false
                }
            }),
            ExpectedReceiptJson {
                model_coverage_row: "dense_smollm2_360m_candidate",
                current_tier: "structurally_valid",
                selected_backend: "nvidia-rtx-5070-ti-cuda",
                selected_route: None,
                fallback_used: false,
                product_cli_ready: false,
                server_ready: false,
                server_ready_scope: None,
                speedup_claim: false,
                full_residency_claim: false,
                bitnet_packed_i2s_qk256_proof: false,
                dense_regular_llm_cuda_proof: false,
            },
        )?;

        Ok(())
    }

    #[test]
    fn explain_receipt_extracts_cuda_plan_and_claim_limits() {
        let receipt = json!({
            "artifact_kind": "dense_regular_llm_cuda",
            "claim": "dense_regular_llm_cuda_tensor_residency_tested",
            "model": {
                "artifact_kind": "dense_gguf",
                "file": "qwen-fixture.gguf",
                "model_family": "qwen"
            },
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "fallback_used": false,
            "execution_plan": {
                "selected_route": "dense_regular_llm_cuda",
                "model_family": "qwen",
                "quantization": "dense_fp16",
                "strict_cuda_ready": true,
                "speedup_claim": false,
                "full_cuda_residency_claimed": false
            },
            "kernel_stats": [
                {
                    "kernel_id": "dense_f16_gemm_cuda",
                    "kernel_time_ms": 1.25,
                    "host_to_device_bytes": 40,
                    "device_to_host_bytes": 24
                }
            ],
            "parity": {
                "passed": true
            },
            "claim_boundary": {
                "speedup_claim": false,
                "full_cuda_residency_claimed": false,
                "dense_gguf_inference_claimed": false,
                "bitnet_packed_i2s_qk256_proof": false
            }
        });

        let explanation = explain_receipt(Path::new("receipt.json"), &receipt);

        assert_eq!(explanation.artifact_kind.as_deref(), Some("dense_regular_llm_cuda"));
        assert_eq!(explanation.model.as_deref(), Some("qwen-fixture.gguf"));
        assert_eq!(
            explanation.backend.selected_backend.as_deref(),
            Some("nvidia-rtx-5070-ti-cuda")
        );
        assert_eq!(
            explanation.execution_plan.selected_route.as_deref(),
            Some("dense_regular_llm_cuda")
        );
        assert_eq!(explanation.kernels, vec!["dense_f16_gemm_cuda"]);
        assert_eq!(explanation.quality.parity_passed, Some(true));
        assert_eq!(explanation.timing.kernel_time_ms, Some(1.25));
        assert_eq!(explanation.timing.host_to_device_bytes, Some(40));
        assert_eq!(explanation.claim_limits.speedup_claim, Some(false));
        assert_eq!(explanation.claim_limits.dense_gguf_inference_claimed, Some(false));
        assert_eq!(explanation.claim_limits.bitnet_packed_i2s_qk256_proof, Some(false));
        assert_eq!(explanation.schema_version, 1);
        assert_eq!(explanation.selected_backend.as_deref(), Some("nvidia-rtx-5070-ti-cuda"));
        assert_eq!(explanation.selected_route.as_deref(), Some("dense_regular_llm_cuda"));
        assert_eq!(explanation.fallback_used, Some(false));
    }

    #[test]
    fn compact_summary_extracts_strict_ask_receipt_shape() {
        let receipt = json!({
            "artifact_kind": "bitnet_cuda_answer",
            "model": {
                "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
                "file": "ggml-model-i2_s.gguf"
            },
            "backend": {
                "selected_backend": "nvidia-rtx-5070-ti-cuda",
                "runtime_api": "cuda",
                "fallback_used": false
            },
            "execution_plan": {
                "selected_route": "bitnet_qk256_cuda",
                "model_family": "bitnet_b1_58",
                "quantization": "i2_s_qk256",
                "speedup_claim": false,
                "full_cuda_residency_claimed": false
            },
            "kernel_stats": [
                {
                    "kernel_id": "qk256_gemv_cuda"
                }
            ],
            "quality": {
                "garbage_filter_passed": true
            },
            "timing": {
                "cuda_kernel_time_ms": 2.5,
                "host_to_device_bytes": 4096,
                "device_to_host_bytes": 2048
            },
            "cuda_execution_residency": {
                "weight_residency": {
                    "weights_uploaded_once": true,
                    "per_token_weight_upload": false
                },
                "kv_cache": {
                    "device": "cpu"
                },
                "full_cuda_residency_claimed": false
            },
            "speedup_claim": false
        });

        let explanation = explain_receipt(
            Path::new("target/bitnet/receipts/cuda-answer-readiness/strict-cuda-ask-latest.json"),
            &receipt,
        );
        let lines = compact_proof_lines(&explanation);

        assert_eq!(explanation.quality.answer_quality_passed, Some(true));
        assert_eq!(explanation.residency.weights_uploaded_once, Some(true));
        assert_eq!(explanation.timing.kernel_time_ms, Some(2.5));
        assert!(lines.contains(&"  route: bitnet_qk256_cuda".to_string()));
        assert!(lines.contains(&"  backend: nvidia-rtx-5070-ti-cuda".to_string()));
        assert!(lines.contains(&"  kernel: qk256_gemv_cuda".to_string()));
        assert!(lines.contains(&"  fallback: false".to_string()));
        assert!(lines.contains(&"  quality: true".to_string()));
        assert!(lines.contains(&"  weights: uploaded once".to_string()));
        assert!(lines.contains(&"  speed claim: false".to_string()));
    }

    #[test]
    fn receipts_explain_extracts_lunar_lake_openvino_route_summary() -> Result<()> {
        let receipt = json!({
            "artifact_kind": "lunar_lake_openvino_operator_ask",
            "proof_stage": "operator_candidate_route_executed",
            "requested_backend": "openvino-gpu",
            "selected_backend": "openvino-gpu",
            "runtime_api": "openvino_genai",
            "runtime_device": "GPU.0",
            "resolved_device": "Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)",
            "fallback_used": false,
            "backend_lane": "dense_slm_openvino_gpu_arc140v",
            "selected_kernel_or_runtime": "openvino-genai-llmpipeline-gpu0",
            "model_family": "qwen",
            "route_id": "dense_slm_openvino_gpu_candidate",
            "route": {
                "route_id": "dense_slm_openvino_gpu_candidate",
                "route_reason": "Candidate route because answer gates and phase metrics exist, but no benchmark-qualified speedup claim is recorded.",
                "acceleration_claim": false
            },
            "model": {
                "repo": "Qwen/Qwen2.5-0.5B-Instruct"
            },
            "output": {
                "generated_token_ids_available_from_pipeline": false
            },
            "answer_gate": {
                "passed": true
            },
            "timing": {
                "pipeline_construct_wall_ms": 6089.31,
                "generation_wall_ms": 2458.34,
                "openvino_perf_metrics": {
                    "time_to_first_token": {
                        "mean_ms": 1455.4
                    }
                }
            },
            "claim_boundary": {
                "speedup_claim": false,
                "bitnet_packed_i2s_qk256_proof": false
            }
        });

        let explanation = explain_receipt(Path::new("openvino-gpu.json"), &receipt);

        assert_eq!(explanation.selected_route.as_deref(), Some("dense_slm_openvino_gpu_candidate"));
        assert_eq!(explanation.quality.answer_quality_passed, Some(true));
        assert_eq!(
            explanation.openvino.route_id.as_deref(),
            Some("dense_slm_openvino_gpu_candidate")
        );
        assert_eq!(explanation.openvino.selected_backend.as_deref(), Some("openvino-gpu"));
        assert_eq!(explanation.openvino.runtime_api.as_deref(), Some("openvino_genai"));
        assert_eq!(explanation.openvino.runtime_device.as_deref(), Some("GPU.0"));
        assert_eq!(
            explanation.openvino.proof_family.as_deref(),
            Some("dense_slm_openvino_gpu_arc140v")
        );
        assert_eq!(
            explanation.openvino.selected_kernel_or_runtime.as_deref(),
            Some("openvino-genai-llmpipeline-gpu0")
        );
        assert_eq!(explanation.openvino.quality_status.as_deref(), Some("answer_gate_passed"));
        assert_eq!(
            explanation.openvino.timing_scope.as_deref(),
            Some("openvino_pipeline_construct_and_generation_wall_time")
        );
        assert_eq!(explanation.openvino.promotion_status.as_deref(), Some("candidate"));
        assert!(
            explanation
                .openvino
                .blockers
                .iter()
                .any(|blocker| blocker.contains("direct generated token IDs"))
        );
        assert!(
            explanation
                .openvino
                .does_not_prove
                .iter()
                .any(|limit| limit == "native OpenCL execution proof")
        );
        assert!(explanation.openvino.does_not_prove.iter().any(|limit| limit == "route promotion"));

        let value = serde_json::to_value(&explanation)?;
        assert_eq!(value["openvino"]["runtime_device"], "GPU.0");
        assert_eq!(value["openvino"]["quality_status"], "answer_gate_passed");

        let lines = compact_proof_lines(&explanation);
        assert!(lines.contains(&"  route: dense_slm_openvino_gpu_candidate".to_string()));
        assert!(lines.contains(&"  device: GPU.0".to_string()));
        assert!(lines.contains(&"  OpenVINO promotion: candidate".to_string()));
        assert!(lines.iter().any(|line| line.contains("does not prove")));
        Ok(())
    }

    #[test]
    fn receipts_explain_links_bitnet_receipt_to_model_coverage() -> Result<()> {
        let receipt = json!({
            "artifact_kind": "bitnet_cuda_answer",
            "model": {
                "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
                "file": "ggml-model-i2_s.gguf"
            },
            "backend": {
                "selected_backend": "nvidia-rtx-5070-ti-cuda",
                "runtime_api": "cuda",
                "fallback_used": false
            },
            "execution_plan": {
                "selected_route": "bitnet_qk256_cuda",
                "model_family": "bitnet_b1_58",
                "speedup_claim": false
            },
            "claim_boundary": {
                "bitnet_packed_i2s_qk256_proof": true,
                "dense_gguf_inference_claimed": false
            }
        });

        let explanation = explain_receipt(Path::new("strict-bitnet.json"), &receipt);

        assert_eq!(explanation.model_coverage.row.as_deref(), Some("bitnet_official_2b_i2s_qk256"));
        assert_eq!(explanation.model_coverage.current_tier.as_deref(), Some("product_cli_ready"));
        assert_eq!(explanation.model_coverage.route.as_deref(), Some("bitnet_qk256_cuda"));
        assert_eq!(explanation.model_coverage.product_cli_ready, Some(true));
        assert_eq!(explanation.model_coverage.speedup_claim, Some(false));
        assert_eq!(explanation.model_coverage.server_ready, Some(false));
        assert_eq!(explanation.model_coverage.server_ready_scope, None);
        assert_eq!(explanation.server_ready, Some(false));
        assert_eq!(explanation.model_coverage.bitnet_packed_i2s_qk256_proof, Some(true));
        assert_eq!(explanation.model_coverage.dense_regular_llm_cuda_proof, Some(false));
        assert_eq!(explanation.model_coverage_row.as_deref(), Some("bitnet_official_2b_i2s_qk256"));
        assert_eq!(explanation.current_tier.as_deref(), Some("product_cli_ready"));
        assert_eq!(explanation.selected_backend.as_deref(), Some("nvidia-rtx-5070-ti-cuda"));
        assert_eq!(explanation.selected_route.as_deref(), Some("bitnet_qk256_cuda"));
        assert_eq!(explanation.fallback_used, Some(false));
        assert_eq!(explanation.product_cli_ready, Some(true));
        assert_eq!(explanation.server_ready_scope, None);
        assert_eq!(explanation.speedup_claim, Some(false));
        assert_eq!(explanation.full_residency_claim, Some(false));
        assert_eq!(explanation.bitnet_packed_i2s_qk256_proof, Some(true));
        assert_eq!(explanation.dense_regular_llm_cuda_proof, Some(false));
        assert!(
            explanation
                .model_coverage
                .warnings
                .iter()
                .any(|warning| warning.contains("not dense SLM CUDA proof"))
        );

        let value = serde_json::to_value(&explanation)?;
        assert_eq!(value["model_coverage_row"], "bitnet_official_2b_i2s_qk256");
        assert_eq!(value["current_tier"], "product_cli_ready");
        assert_eq!(value["selected_backend"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(value["selected_route"], "bitnet_qk256_cuda");
        assert_eq!(value["fallback_used"], false);
        assert_eq!(value["product_cli_ready"], true);
        assert_eq!(value["server_ready"], false);
        assert!(value["server_ready_scope"].is_null());
        assert_eq!(value["speedup_claim"], false);
        assert_eq!(value["full_residency_claim"], false);
        assert_eq!(value["bitnet_packed_i2s_qk256_proof"], true);
        assert_eq!(value["dense_regular_llm_cuda_proof"], false);
        Ok(())
    }

    #[test]
    fn receipts_explain_links_dense_qwen_receipt_to_model_coverage() -> Result<()> {
        let receipt = json!({
            "artifact_kind": "dense_gguf_qwen_warm_session_strict_cuda_proof",
            "claim": "dense_gguf_qwen_warm_session_strict_cuda_proof_recorded",
            "model": {
                "id": "qwen2.5-0.5b-instruct-q8_0"
            },
            "execution_plan": {
                "selected_route": "dense_regular_llm_cuda",
                "model_family": "qwen",
                "requested_backend": "nvidia-rtx-5070-ti-cuda",
                "selected_backend": "nvidia-rtx-5070-ti-cuda",
                "runtime_api": "cuda",
                "fallback_used": false,
                "speedup_claim": false
            },
            "claim_boundary": {
                "bitnet_packed_i2s_qk256_proof": false,
                "dense_regular_llm_cuda_claimed": true,
                "server_ready_claimed": false,
                "speedup_claim": false
            },
            "session_lifecycle": {
                "model_loaded_once": true,
                "cuda_context_once": true,
                "weights_uploaded_once": true,
                "per_request_model_load": false,
                "workspace_reused": true,
                "fallback_used": false
            },
            "tensor_residency": {
                "model_loaded_once": true,
                "cuda_context_once": true,
                "weights_uploaded_once": true,
                "per_request_model_load": false,
                "per_token_weight_upload": false,
                "workspace_reused": true,
                "full_cuda_residency_claimed": false
            }
        });

        let explanation = explain_receipt(Path::new("dense-qwen.json"), &receipt);

        assert_eq!(explanation.model_coverage.row.as_deref(), Some("dense_qwen25_05b_q8_cuda"));
        assert_eq!(explanation.model_coverage.current_tier.as_deref(), Some("product_cli_ready"));
        assert_eq!(explanation.model_coverage.route.as_deref(), Some("dense_regular_llm_cuda"));
        assert_eq!(explanation.model_coverage.product_cli_ready, Some(true));
        assert_eq!(explanation.model_coverage.speedup_claim, Some(false));
        assert_eq!(explanation.model_coverage.server_ready, Some(true));
        assert_eq!(explanation.model_coverage.server_ready_scope.as_deref(), Some("exact_profile"));
        assert_eq!(explanation.model_coverage.bitnet_packed_i2s_qk256_proof, Some(false));
        assert_eq!(explanation.model_coverage.dense_regular_llm_cuda_proof, Some(true));
        assert_eq!(explanation.model_coverage_row.as_deref(), Some("dense_qwen25_05b_q8_cuda"));
        assert_eq!(explanation.current_tier.as_deref(), Some("product_cli_ready"));
        assert_eq!(explanation.selected_backend.as_deref(), Some("nvidia-rtx-5070-ti-cuda"));
        assert_eq!(explanation.selected_route.as_deref(), Some("dense_regular_llm_cuda"));
        assert_eq!(explanation.fallback_used, Some(false));
        assert_eq!(explanation.product_cli_ready, Some(true));
        assert_eq!(explanation.server_ready, Some(true));
        assert_eq!(explanation.server_ready_scope.as_deref(), Some("exact_profile"));
        assert_eq!(explanation.speedup_claim, Some(false));
        assert_eq!(explanation.full_residency_claim, Some(false));
        assert_eq!(explanation.bitnet_packed_i2s_qk256_proof, Some(false));
        assert_eq!(explanation.dense_regular_llm_cuda_proof, Some(true));
        assert_eq!(explanation.residency.model_loaded_once, Some(true));
        assert_eq!(explanation.residency.cuda_context_once, Some(true));
        assert_eq!(explanation.residency.weights_uploaded_once, Some(true));
        assert_eq!(explanation.residency.per_request_model_load, Some(false));
        assert_eq!(explanation.residency.per_token_weight_upload, Some(false));
        assert_eq!(explanation.residency.workspace_reused, Some(true));
        assert!(
            explanation
                .model_coverage
                .warnings
                .iter()
                .any(|warning| warning.contains("not BitNet packed I2_S/QK256 proof"))
        );

        let value = serde_json::to_value(&explanation)?;
        assert_eq!(value["model_coverage_row"], "dense_qwen25_05b_q8_cuda");
        assert_eq!(value["current_tier"], "product_cli_ready");
        assert_eq!(value["selected_backend"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(value["selected_route"], "dense_regular_llm_cuda");
        assert_eq!(value["fallback_used"], false);
        assert_eq!(value["product_cli_ready"], true);
        assert_eq!(value["server_ready"], true);
        assert_eq!(value["server_ready_scope"], "exact_profile");
        assert_eq!(value["speedup_claim"], false);
        assert_eq!(value["full_residency_claim"], false);
        assert_eq!(value["bitnet_packed_i2s_qk256_proof"], false);
        assert_eq!(value["dense_regular_llm_cuda_proof"], true);
        assert_eq!(value["residency"]["model_loaded_once"], true);
        assert_eq!(value["residency"]["cuda_context_once"], true);
        assert_eq!(value["residency"]["weights_uploaded_once"], true);
        assert_eq!(value["residency"]["per_request_model_load"], false);
        assert_eq!(value["residency"]["per_token_weight_upload"], false);
        assert_eq!(value["residency"]["workspace_reused"], true);
        Ok(())
    }

    #[test]
    fn receipts_explain_links_qwen3_dense_receipt_to_product_cli_coverage() -> Result<()> {
        let receipt = json!({
            "artifact_kind": "dense_gguf_qwen_ask_strict_cuda_proof",
            "claim": "dense_gguf_qwen_ask_strict_cuda_proof_recorded",
            "model": {
                "id": "qwen3-0.6b-instruct-q8_0",
                "file": "Qwen3-0.6B-Q8_0.gguf",
                "architecture": "qwen3"
            },
            "execution_plan": {
                "selected_route": "dense_regular_llm_cuda",
                "model_family": "qwen",
                "requested_backend": "nvidia-rtx-5070-ti-cuda",
                "selected_backend": "nvidia-rtx-5070-ti-cuda",
                "runtime_api": "cuda",
                "fallback_used": false,
                "speedup_claim": false
            },
            "claim_boundary": {
                "bitnet_packed_i2s_qk256_proof": false,
                "dense_regular_llm_cuda_claimed": true,
                "server_ready_claimed": false,
                "speedup_claim": false,
                "full_cuda_residency_claimed": false
            }
        });

        let explanation = explain_receipt(Path::new("dense-qwen3.json"), &receipt);

        assert_eq!(explanation.model_coverage.row.as_deref(), Some("dense_qwen3_06b_q8_candidate"));
        assert_eq!(explanation.model_coverage.current_tier.as_deref(), Some("product_cli_ready"));
        assert_eq!(explanation.model_coverage.route.as_deref(), Some("dense_regular_llm_cuda"));
        assert_eq!(explanation.model_coverage.product_cli_ready, Some(true));
        assert_eq!(explanation.model_coverage.server_ready, Some(true));
        assert_eq!(explanation.model_coverage.server_ready_scope.as_deref(), Some("exact_profile"));
        assert_eq!(explanation.model_coverage.speedup_claim, Some(false));
        assert_eq!(explanation.model_coverage.full_residency_claim, Some(false));
        assert_eq!(explanation.model_coverage.bitnet_packed_i2s_qk256_proof, Some(false));
        assert_eq!(explanation.model_coverage.dense_regular_llm_cuda_proof, Some(true));
        assert_eq!(explanation.model_coverage_row.as_deref(), Some("dense_qwen3_06b_q8_candidate"));
        assert_eq!(explanation.current_tier.as_deref(), Some("product_cli_ready"));
        assert_eq!(explanation.selected_backend.as_deref(), Some("nvidia-rtx-5070-ti-cuda"));
        assert_eq!(explanation.selected_route.as_deref(), Some("dense_regular_llm_cuda"));
        assert_eq!(explanation.fallback_used, Some(false));
        assert_eq!(explanation.product_cli_ready, Some(true));
        assert_eq!(explanation.server_ready, Some(true));
        assert_eq!(explanation.server_ready_scope.as_deref(), Some("exact_profile"));
        assert_eq!(explanation.speedup_claim, Some(false));
        assert_eq!(explanation.full_residency_claim, Some(false));
        assert_eq!(explanation.bitnet_packed_i2s_qk256_proof, Some(false));
        assert_eq!(explanation.dense_regular_llm_cuda_proof, Some(true));

        let value = serde_json::to_value(&explanation)?;
        assert_eq!(value["model_coverage_row"], "dense_qwen3_06b_q8_candidate");
        assert_eq!(value["current_tier"], "product_cli_ready");
        assert_eq!(value["selected_backend"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(value["selected_route"], "dense_regular_llm_cuda");
        assert_eq!(value["fallback_used"], false);
        assert_eq!(value["product_cli_ready"], true);
        assert_eq!(value["server_ready"], true);
        assert_eq!(value["server_ready_scope"], "exact_profile");
        assert_eq!(value["speedup_claim"], false);
        assert_eq!(value["full_residency_claim"], false);
        assert_eq!(value["bitnet_packed_i2s_qk256_proof"], false);
        assert_eq!(value["dense_regular_llm_cuda_proof"], true);
        Ok(())
    }

    #[test]
    fn receipts_explain_surfaces_qwen3_repeated_comparator_profiles() -> Result<()> {
        let profiles = vec![
            qwen3_repeated_comparator_profile("one_token", 1200.0, 1800.0),
            qwen3_repeated_comparator_profile("short_decode_8", 8200.0, 9700.0),
            qwen3_repeated_comparator_profile("short_decode_32", 31200.0, 35600.0),
            qwen3_repeated_comparator_profile("warm_session_3_turns", 24500.0, 29200.0),
            qwen3_repeated_comparator_profile("decode_128_from_warm_context", 105000.0, 117000.0),
        ];
        let receipt = json!({
            "schema": 1,
            "artifact_kind": "qwen3_cuda_repeated_comparator",
            "machine_id": "windows-9950x3d-rtx5070ti",
            "hardware_lane": "nvidia_rtx_5070_ti_cuda",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "reference_backend": "amd-9950x3d-cpu-avx512",
            "runtime_api": "cuda",
            "selected_route": "dense_regular_llm_cuda",
            "claim": "qwen3_cuda_repeated_comparator",
            "fallback_used": false,
            "speedup_claim": false,
            "benchmark_qualified_speedup": false,
            "full_cuda_residency_claimed": false,
            "dense_gguf_inference_claimed": false,
            "broad_dense_gguf_ready_claimed": false,
            "qwen25_proof_inherited": false,
            "server_ready_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "claim_boundary": {
                "qwen3_cuda_repeated_comparator_claimed": true,
                "qwen_one_token_cuda_claimed": true,
                "qwen_short_decode_cuda_claimed": true,
                "qwen_warm_session_cuda_claimed": true,
                "qwen_chat_cuda_claimed": true,
                "server_ready_claimed": false,
                "speedup_claim": false,
                "benchmark_qualified_speedup": false,
                "full_cuda_residency_claimed": false,
                "broad_dense_gguf_ready_claimed": false,
                "qwen25_proof_inherited": false,
                "bitnet_packed_i2s_qk256_proof": false
            },
            "model": {
                "id": "qwen3-0.6b-instruct-q8_0",
                "architecture": "qwen3",
                "model_family": "qwen",
                "artifact_kind": "dense_gguf",
                "file": "Qwen3-0.6B-Q8_0.gguf",
                "sha256": "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"
            },
            "execution_plan": {
                "model_family": "qwen",
                "quantization": "dense_gguf_q8_0_qwen3_product_contract",
                "selected_route": "dense_regular_llm_cuda",
                "requested_backend": "nvidia-rtx-5070-ti-cuda",
                "selected_backend": "nvidia-rtx-5070-ti-cuda",
                "runtime_api": "cuda",
                "strict_fallback_policy": "reject",
                "dense_regular_llm_cuda": true,
                "bitnet_packed_qk256_cuda": false,
                "fallback_used": false,
                "strict_cuda_ready": true,
                "speedup_claim": false,
                "full_cuda_residency_claimed": false,
                "cuda_dense_regular_llm_ops": 8112,
                "cuda_bitnet_qk256_ops": 0,
                "cpu_fallback_ops": 0,
                "unsupported_ops": 0
            },
            "profiles": profiles,
            "comparator_summary": {
                "status": "repeated_comparator_only",
                "profiles_recorded": 5,
                "min_runs_per_backend": 3,
                "total_cpu_runs": 15,
                "total_cuda_runs": 15,
                "fallback_free": true,
                "same_artifact_sha": true,
                "same_tokenizer_prompt_policy": true,
                "deterministic_generation_policy": true,
                "generated_tokens_compared": true,
                "speedup_claim_allowed": false,
                "benchmark_qualified_speedup": false,
                "accepted_speedup_profiles": [],
                "remaining_qualification_blockers": [
                    "profile-specific speedup thresholds remain unreviewed",
                    "pure host-to-device timing remains separated from the model-load envelope"
                ]
            },
            "transfer_timing": {
                "status": "host_to_device_model_load_envelope_device_to_host_measured",
                "host_to_device_bytes_recorded": true,
                "device_to_host_bytes_recorded": true,
                "host_to_device_timing_recorded": true,
                "device_to_host_timing_recorded": true,
                "pure_host_to_device_timing_recorded": false
            }
        });

        let explanation = explain_receipt(Path::new("qwen3-repeated-comparator.json"), &receipt);

        assert_eq!(explanation.model_coverage_row.as_deref(), Some("dense_qwen3_06b_q8_candidate"));
        assert_eq!(explanation.current_tier.as_deref(), Some("product_cli_ready"));
        assert_eq!(explanation.selected_backend.as_deref(), Some("nvidia-rtx-5070-ti-cuda"));
        assert_eq!(explanation.selected_route.as_deref(), Some("dense_regular_llm_cuda"));
        assert_eq!(explanation.fallback_used, Some(false));
        assert_eq!(explanation.product_cli_ready, Some(true));
        assert_eq!(explanation.server_ready, Some(true));
        assert_eq!(explanation.server_ready_scope.as_deref(), Some("exact_profile"));
        assert_eq!(explanation.speedup_claim, Some(false));
        assert_eq!(explanation.full_residency_claim, Some(false));
        assert_eq!(explanation.bitnet_packed_i2s_qk256_proof, Some(false));
        assert_eq!(explanation.dense_regular_llm_cuda_proof, Some(true));

        let qualification = &explanation.benchmark_qualification;
        assert_eq!(qualification.status.as_deref(), Some("repeated_comparator_only"));
        assert_eq!(qualification.benchmark_qualified_speedup, Some(false));
        assert_eq!(qualification.speedup_claim_allowed, Some(false));
        assert!(qualification.accepted_profiles.is_empty());
        assert_eq!(
            qualification.blocked_profiles,
            vec![
                "one_token",
                "short_decode_8",
                "short_decode_32",
                "warm_session_3_turns",
                "decode_128_from_warm_context"
            ]
        );
        assert_eq!(
            qualification.transfer_timing_status.as_deref(),
            Some("host_to_device_model_load_envelope_device_to_host_measured")
        );
        assert_eq!(qualification.pure_host_to_device_timing_recorded, Some(false));
        assert_eq!(qualification.device_to_host_timing_recorded, Some(true));
        assert_eq!(qualification.profile_reviews.len(), 5);

        let short_decode = qualification
            .profile_reviews
            .iter()
            .find(|profile| profile.profile == "short_decode_32")
            .ok_or_else(|| anyhow!("short_decode_32 comparator profile missing"))?;
        assert_eq!(
            short_decode.decision.as_deref(),
            Some("repeated_same_artifact_cpu_cuda_comparator")
        );
        assert_eq!(short_decode.benchmark_qualified_speedup, Some(false));
        assert_eq!(short_decode.speedup_claim_allowed, Some(false));
        assert_eq!(short_decode.fallback_free, Some(true));
        assert_eq!(short_decode.quality_passed, Some(true));
        assert_eq!(short_decode.generated_token_ids_match, Some(true));
        assert_eq!(short_decode.cpu_total_ms_mean, Some(31200.0));
        assert_eq!(short_decode.cuda_total_ms_mean, Some(35600.0));
        assert_eq!(short_decode.host_to_device_ms, Some(8.0));
        assert_eq!(short_decode.device_to_host_ms, Some(0.5));
        Ok(())
    }

    fn qwen3_repeated_comparator_profile(
        profile: &str,
        cpu_total_ms_mean: f64,
        cuda_total_ms_mean: f64,
    ) -> Value {
        json!({
            "profile": profile,
            "status": "repeated_same_artifact_cpu_cuda_comparator",
            "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
            "cuda_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "selected_route": "dense_regular_llm_cuda",
            "run_count": 3,
            "cpu_runs": 3,
            "cuda_runs": 3,
            "min_runs_per_backend": 3,
            "fallback_free": true,
            "same_artifact_sha": true,
            "same_tokenizer_prompt_policy": true,
            "deterministic_generation_policy": true,
            "generated_token_ids_match": true,
            "speedup_claim": false,
            "benchmark_qualified_speedup": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "full_cuda_residency_claimed": false,
            "server_ready_claimed": false,
            "transfer_timing_status": "host_to_device_model_load_envelope_device_to_host_measured",
            "cpu_total_ms": {
                "count": 3,
                "min": cpu_total_ms_mean,
                "mean": cpu_total_ms_mean,
                "max": cpu_total_ms_mean
            },
            "cuda_total_ms": {
                "count": 3,
                "min": cuda_total_ms_mean,
                "mean": cuda_total_ms_mean,
                "max": cuda_total_ms_mean
            },
            "host_to_device_ms": {
                "count": 3,
                "min": 8.0,
                "mean": 8.0,
                "max": 8.0
            },
            "device_to_host_ms": {
                "count": 3,
                "min": 0.5,
                "mean": 0.5,
                "max": 0.5
            }
        })
    }

    #[test]
    fn receipts_explain_links_server_smoke_receipt_to_dense_qwen_coverage() {
        let receipt = json!({
            "receipt_kind": "server_shared_engine_chat_completion",
            "runtime_path": "shared_local_inference_engine",
            "runtime_api": "cuda",
            "model_identity": {
                "model_id": "qwen2.5-0.5b-instruct-q8_0",
                "requested_model": "qwen2.5-0.5b-instruct-q8_0",
                "active_model_id": "model-1",
                "active_model_path": "models/qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf",
                "model_sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"
            },
            "endpoint_profile": {
                "endpoint": "/v1/chat/completions",
                "method": "POST",
                "request_profile": "non_streaming_chat_completion",
                "streaming": false,
                "message_count": 1
            },
            "generation_policy": {
                "max_tokens": 16,
                "temperature": 0.0,
                "top_p": 1.0,
                "decoding": "greedy"
            },
            "requested_model": "qwen2.5-0.5b-instruct-q8_0",
            "active_model_id": "model-1",
            "active_model_path": "models/qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf",
            "model_sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
            "model_coverage_row": "dense_qwen25_05b_q8_cuda",
            "model_coverage_tier": "product_cli_ready",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_route": "dense_regular_llm_cuda",
            "fallback_used": false,
            "simulated_inference": false,
            "generated_text_non_empty": true,
            "quality_gate": {
                "gate": "server_non_empty_utf8_response",
                "passed": true,
                "generated_text_non_empty": true,
                "utf8_valid": true,
                "broad_chat_quality_claimed": false
            },
            "server_smoke_response_claimed": true,
            "server_ready_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false,
            "dense_regular_llm_cuda_inference_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false
        });

        let explanation = explain_receipt(Path::new("server-smoke.json"), &receipt);

        assert_eq!(explanation.model_coverage.row.as_deref(), Some("dense_qwen25_05b_q8_cuda"));
        assert_eq!(explanation.model_coverage.current_tier.as_deref(), Some("product_cli_ready"));
        assert_eq!(explanation.model_coverage.route.as_deref(), Some("dense_regular_llm_cuda"));
        assert_eq!(
            explanation.backend.requested_backend.as_deref(),
            Some("nvidia-rtx-5070-ti-cuda")
        );
        assert_eq!(
            explanation.backend.selected_backend.as_deref(),
            Some("nvidia-rtx-5070-ti-cuda")
        );
        assert_eq!(explanation.backend.runtime_api.as_deref(), Some("cuda"));
        assert_eq!(explanation.backend.fallback_used, Some(false));
        assert_eq!(explanation.quality.answer_quality_passed, Some(true));
        assert_eq!(explanation.model_coverage.server_ready, Some(true));
        assert_eq!(explanation.model_coverage.speedup_claim, Some(false));
        assert_eq!(explanation.model_coverage.dense_regular_llm_cuda_proof, Some(true));
        assert_eq!(explanation.model_coverage.bitnet_packed_i2s_qk256_proof, Some(false));
        assert_eq!(explanation.model_coverage_row.as_deref(), Some("dense_qwen25_05b_q8_cuda"));
        assert_eq!(explanation.current_tier.as_deref(), Some("product_cli_ready"));
        assert_eq!(explanation.selected_backend.as_deref(), Some("nvidia-rtx-5070-ti-cuda"));
        assert_eq!(explanation.selected_route.as_deref(), Some("dense_regular_llm_cuda"));
        assert_eq!(explanation.fallback_used, Some(false));
        assert_eq!(explanation.server_ready, Some(true));
        assert_eq!(explanation.speedup_claim, Some(false));
        assert_eq!(explanation.full_residency_claim, Some(false));
        assert_eq!(explanation.bitnet_packed_i2s_qk256_proof, Some(false));
        assert_eq!(explanation.dense_regular_llm_cuda_proof, Some(true));
        assert!(
            !explanation
                .model_coverage
                .warnings
                .iter()
                .any(|warning| warning.contains("missing exact artifact checksum"))
        );
        assert!(
            !explanation
                .model_coverage
                .warnings
                .iter()
                .any(|warning| warning.contains("missing endpoint/request profile"))
        );
        assert!(
            !explanation
                .model_coverage
                .warnings
                .iter()
                .any(|warning| warning.contains("missing generation-policy"))
        );
    }

    #[test]
    fn receipts_explain_warns_on_inconsistent_server_checksum_identity() {
        let receipt = json!({
            "receipt_kind": "server_shared_engine_chat_completion",
            "runtime_path": "shared_local_inference_engine",
            "runtime_api": "cuda",
            "model_identity": {
                "model_id": "qwen2.5-0.5b-instruct-q8_0",
                "requested_model": "qwen2.5-0.5b-instruct-q8_0",
                "active_model_id": "model-1",
                "active_model_path": "models/qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf",
                "model_sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"
            },
            "endpoint_profile": {
                "endpoint": "/v1/chat/completions",
                "method": "POST",
                "request_profile": "non_streaming_chat_completion",
                "streaming": false,
                "message_count": 1
            },
            "generation_policy": {
                "max_tokens": 16,
                "temperature": 0.0,
                "top_p": 1.0,
                "decoding": "greedy"
            },
            "requested_model": "qwen2.5-0.5b-instruct-q8_0",
            "active_model_id": "model-1",
            "active_model_path": "models/qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf",
            "model_sha256": "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
            "model_coverage_row": "dense_qwen25_05b_q8_cuda",
            "model_coverage_tier": "product_cli_ready",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_route": "dense_regular_llm_cuda",
            "fallback_used": false,
            "quality_gate": {
                "passed": true
            },
            "server_smoke_response_claimed": true,
            "server_ready_claimed": false,
            "speedup_claim": false,
            "dense_regular_llm_cuda_inference_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false
        });

        let explanation = explain_receipt(Path::new("server-smoke-mismatched-sha.json"), &receipt);
        let warnings = &explanation.model_coverage.warnings;

        assert!(warnings.iter().any(|warning| warning.contains("missing exact artifact checksum")));
        assert!(
            !warnings.iter().any(|warning| warning.contains("missing endpoint/request profile"))
        );
        assert!(!warnings.iter().any(|warning| warning.contains("missing generation-policy")));
    }

    #[test]
    fn receipts_explain_warns_on_unpromoted_server_ready_claim() {
        let receipt = json!({
            "receipt_kind": "server_shared_engine_chat_completion",
            "runtime_path": "shared_local_inference_engine",
            "runtime_api": "cuda",
            "requested_model": "microsoft-bitnet-b1.58-2B-4T-i2s",
            "active_model_id": "bitnet-model-1",
            "active_model_path": "models/microsoft-bitnet-b1.58-2B-4T-i2s/ggml-model-i2_s.gguf",
            "model_coverage_row": "bitnet_official_2b_i2s_qk256",
            "model_coverage_tier": "product_cli_ready",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_route": "bitnet_qk256_cuda",
            "fallback_used": false,
            "quality_gate": {
                "passed": true
            },
            "server_smoke_response_claimed": true,
            "server_ready_claimed": true,
            "speedup_claim": false,
            "dense_regular_llm_cuda_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": true
        });

        let explanation = explain_receipt(Path::new("server-smoke-stale.json"), &receipt);
        let warnings = &explanation.model_coverage.warnings;

        assert_eq!(explanation.model_coverage.row.as_deref(), Some("bitnet_official_2b_i2s_qk256"));
        assert_eq!(explanation.server_ready, Some(false));
        assert_eq!(explanation.selected_route.as_deref(), Some("bitnet_qk256_cuda"));
        assert!(warnings.iter().any(|warning| warning.contains("missing exact artifact checksum")));
        assert!(
            warnings.iter().any(|warning| warning.contains("missing endpoint/request profile"))
        );
        assert!(warnings.iter().any(|warning| warning.contains("missing generation-policy")));
        assert!(warnings.iter().any(|warning| warning.contains("does not promote server_ready")));
    }

    #[test]
    fn explain_receipt_extracts_benchmark_qualification_profiles() {
        let receipt = json!({
            "artifact_kind": "dense_gguf_qwen_benchmark_qualification_review",
            "claim": "dense_gguf_qwen_benchmark_qualification_review",
            "benchmark_qualified_speedup": false,
            "speedup_claim": false,
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "fallback_used": false,
            "execution_plan": {
                "selected_route": "dense_regular_llm_cuda",
                "selected_backend": "nvidia-rtx-5070-ti-cuda",
                "runtime_api": "cuda",
                "fallback_used": false,
                "speedup_claim": false,
                "full_cuda_residency_claimed": false
            },
            "qualification_decision": {
                "accepted_profiles": [],
                "benchmark_qualified_speedup": false,
                "blocked_profiles": ["one_token", "warm_session_3_turns"],
                "speedup_claim_allowed": false,
                "status": "not_accepted"
            },
            "transfer_timing_review": {
                "device_to_host_timing_recorded": true,
                "host_to_device_ms_includes_non_transfer_overhead": true,
                "host_to_device_pure_transfer_timing_recorded": false,
                "host_to_device_scope": "model_load_wall_clock_envelope",
                "host_to_device_source": "wall_clock_model_load_with_cuda_weight_upload",
                "status": "host_to_device_model_load_envelope_device_to_host_measured"
            },
            "profile_reviews": [
                {
                    "benchmark_qualified_speedup": false,
                    "blockers": [
                        "CUDA mean total time is slower than CPU mean total time",
                        "pure host-to-device transfer timing is unmeasured"
                    ],
                    "cpu_total_ms_mean": 2872.8427,
                    "cuda_total_ms_mean": 3978.571,
                    "decision": "not_accepted",
                    "device_to_host_ms": 0.8953,
                    "device_to_host_ms_source": "wall_clock_extract_logits_2d_local",
                    "fallback_free": true,
                    "generated_token_ids_match": true,
                    "host_to_device_ms": 3513.8495,
                    "host_to_device_ms_includes_non_transfer_overhead": true,
                    "host_to_device_ms_scope": "model_load_wall_clock_envelope",
                    "host_to_device_ms_source": "wall_clock_model_load_with_cuda_weight_upload",
                    "observed_cpu_total_ms_div_cuda_total_ms": 0.722,
                    "profile": "one_token",
                    "pure_host_to_device_ms_source": "not_measured_by_dense_qwen_runtime",
                    "quality_passed": true,
                    "speedup_claim_allowed": false
                }
            ],
            "claim_boundary": {
                "benchmark_qualified_speedup": false,
                "full_cuda_residency_claimed": false,
                "bitnet_packed_i2s_qk256_proof": false
            }
        });

        let explanation = explain_receipt(Path::new("benchmark.json"), &receipt);

        assert_eq!(explanation.benchmark_qualification.status.as_deref(), Some("not_accepted"));
        assert_eq!(explanation.benchmark_qualification.benchmark_qualified_speedup, Some(false));
        assert_eq!(
            explanation.benchmark_qualification.blocked_profiles,
            vec!["one_token", "warm_session_3_turns"]
        );
        assert_eq!(
            explanation.benchmark_qualification.host_to_device_includes_non_transfer_overhead,
            Some(true)
        );
        assert_eq!(
            explanation.benchmark_qualification.pure_host_to_device_timing_recorded,
            Some(false)
        );
        assert_eq!(explanation.benchmark_qualification.profile_reviews.len(), 1);
        let profile = &explanation.benchmark_qualification.profile_reviews[0];
        assert_eq!(profile.profile, "one_token");
        assert_eq!(profile.decision.as_deref(), Some("not_accepted"));
        assert_eq!(profile.cpu_total_ms_mean, Some(2872.8427));
        assert_eq!(profile.cuda_total_ms_mean, Some(3978.571));
        assert_eq!(profile.host_to_device_ms, Some(3513.8495));
        assert_eq!(
            profile.host_to_device_ms_source.as_deref(),
            Some("wall_clock_model_load_with_cuda_weight_upload")
        );
        assert_eq!(
            profile.pure_host_to_device_ms_source.as_deref(),
            Some("not_measured_by_dense_qwen_runtime")
        );
        assert!(
            profile
                .blockers
                .iter()
                .any(|blocker| { blocker == "pure host-to-device transfer timing is unmeasured" })
        );
    }

    #[test]
    fn latest_receipt_prefers_newest_json_recursively() {
        let temp = tempfile::tempdir().unwrap();
        let old = temp.path().join("old.json");
        let nested = temp.path().join("nested");
        let newest = nested.join("new.json");
        fs::create_dir_all(&nested).unwrap();
        fs::write(&old, "{}").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        fs::write(&newest, "{}").unwrap();

        assert_eq!(latest_receipt_under(temp.path()).unwrap(), newest);
    }

    #[test]
    fn resolve_receipt_requires_path_without_latest() {
        let err = resolve_receipt_path(None, false).unwrap_err().to_string();
        assert!(err.contains("pass a receipt path or use --latest"));
    }
}
