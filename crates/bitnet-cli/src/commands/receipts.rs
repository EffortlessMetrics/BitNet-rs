//! Receipt explanation helpers for operator-facing proof summaries.
//!
//! This command intentionally does not validate a receipt against one narrow
//! schema. It extracts the common proof fields shared by BitNet CUDA, dense CUDA,
//! answer-corpus, warm-session, and benchmark receipts so users can inspect what
//! actually ran without needing to know every receipt variant.

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Subcommand};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const DEFAULT_RECEIPTS_DIR: &str = "target/bitnet/receipts";

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
    },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ReceiptExplanation {
    pub path: String,
    pub artifact_kind: Option<String>,
    pub claim: Option<String>,
    pub model: Option<String>,
    pub backend: BackendExplanation,
    pub execution_plan: ExecutionPlanExplanation,
    pub kernels: Vec<String>,
    pub quality: QualityExplanation,
    pub timing: TimingExplanation,
    pub residency: ResidencyExplanation,
    pub claim_limits: ClaimLimitsExplanation,
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
    pub weights_uploaded_once: Option<bool>,
    pub per_token_weight_upload: Option<bool>,
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

impl ReceiptsCommand {
    pub async fn execute(&self) -> Result<()> {
        match &self.action {
            ReceiptsAction::Explain { path, latest, json } => {
                let receipt_path = resolve_receipt_path(path.as_deref(), *latest)?;
                let receipt = read_receipt_json(&receipt_path)?;
                let explanation = explain_receipt(&receipt_path, &receipt);
                if *json {
                    println!("{}", serde_json::to_string_pretty(&explanation)?);
                } else {
                    print_receipt_explanation(&explanation);
                }
                Ok(())
            }
        }
    }
}

fn resolve_receipt_path(path: Option<&Path>, latest: bool) -> Result<PathBuf> {
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

fn read_receipt_json(path: &Path) -> Result<Value> {
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
    ReceiptExplanation {
        path: path.display().to_string(),
        artifact_kind: string_at(receipt, &["artifact_kind"]),
        claim: string_at(receipt, &["claim"]),
        model: model_summary(receipt),
        backend: backend_explanation(receipt),
        execution_plan: execution_plan_explanation(receipt),
        kernels: kernel_ids(receipt),
        quality: quality_explanation(receipt),
        timing: timing_explanation(receipt),
        residency: residency_explanation(receipt),
        claim_limits: claim_limits_explanation(receipt),
    }
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
        selected_route: string_at(receipt, &["execution_plan", "selected_route"]),
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
        weights_uploaded_once: bool_at(
            receipt,
            &["cuda_execution_residency", "weights", "uploaded_once"],
        )
        .or_else(|| bool_at(receipt, &["residency", "weights_uploaded_once"]))
        .or_else(|| bool_at(receipt, &["proof", "weights_uploaded_once"])),
        per_token_weight_upload: bool_at(
            receipt,
            &["cuda_execution_residency", "weights", "per_token_weight_upload"],
        )
        .or_else(|| bool_at(receipt, &["residency", "per_token_weight_upload"]))
        .or_else(|| bool_at(receipt, &["proof", "per_token_weight_upload"])),
        kv_cache_residency: string_at(
            receipt,
            &["cuda_execution_residency", "kv_cache", "residency"],
        ),
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

fn print_receipt_explanation(explanation: &ReceiptExplanation) {
    println!("Receipt: {}", explanation.path);
    print_option("Artifact", explanation.artifact_kind.as_deref());
    print_option("Claim", explanation.claim.as_deref());
    print_option("Model", explanation.model.as_deref());

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
        print_bool_indented(
            "per_token_weight_upload",
            explanation.residency.per_token_weight_upload,
        );
        print_option_indented("kv_cache", explanation.residency.kv_cache_residency.as_deref());
        print_bool_indented(
            "full_cuda_residency_claimed",
            explanation.residency.full_cuda_residency_claimed,
        );
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
        || residency.weights_uploaded_once.is_some()
        || residency.per_token_weight_upload.is_some()
        || residency.kv_cache_residency.is_some()
        || residency.full_cuda_residency_claimed.is_some()
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

fn value_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
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
    use serde_json::json;

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
