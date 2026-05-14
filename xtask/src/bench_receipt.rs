use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::path::Path;

const CRITICAL_NOT_CLAIMS: &[&str] = &[
    "selected_attention_residency",
    "resident_kv_decode",
    "attention_scores_residency",
    "softmax_residency",
    "attention_value_mix_residency",
    "full_support_op_residency",
    "full_device_residency",
    "completion",
];

const REQUIRED_POINTERS: &[&str] = &[
    "/schema_version",
    "/receipt_type",
    "/run_id",
    "/repo/commit",
    "/repo/tree",
    "/repo/dirty",
    "/repo/cargo_lock_hash",
    "/repo/rustc",
    "/repo/features",
    "/device/device_slug",
    "/device/device_instance_hash",
    "/model/contract",
    "/model/model_id",
    "/model/weights_sha256",
    "/model/tokenizer_sha256",
    "/model/chat_template_hash",
    "/backend/selected_backend",
    "/backend/backend_family",
    "/backend/fallback_used",
    "/kernel_route/route_verified",
    "/kernel_route/device_slug",
    "/kernel_route/selected_backend",
    "/kernel_route/kernel_variants",
    "/benchmark_profile/id",
    "/benchmark_profile/profile_hash",
    "/quality_gate/required",
    "/quality_gate/quality_passed",
    "/quality_gate/quality_receipt",
    "/claim_gate/classification",
    "/measurements",
    "/not_claims",
];

#[derive(Debug, Serialize)]
struct BenchReceiptVerifyReport {
    diagnostic: &'static str,
    producer: &'static str,
    receipt_path: String,
    passed: bool,
    claimable: bool,
    classification: String,
    quality_required: bool,
    quality_passed: bool,
    repo_dirty: bool,
    fallback_used: bool,
    route_verified: bool,
    model_contract_matched: Option<bool>,
    resource_envelope_complete: Option<bool>,
    run_id: Option<String>,
    benchmark_profile: Option<String>,
    selected_backend: Option<String>,
    failures: Vec<String>,
    blocked_reasons: Vec<String>,
    not_claims: Vec<String>,
}

pub fn verify_receipt(receipt_path: &Path, format: &str, require_claimable: bool) -> Result<()> {
    let report = build_verify_report(receipt_path, require_claimable)?;
    emit_report(&report, format)?;
    if !report.passed {
        bail!("bench receipt verification failed: {}", report.failures.join(", "));
    }
    Ok(())
}

fn build_verify_report(
    receipt_path: &Path,
    require_claimable: bool,
) -> Result<BenchReceiptVerifyReport> {
    let raw = fs::read_to_string(receipt_path)
        .with_context(|| format!("reading {}", receipt_path.display()))?;
    let value: Value = serde_json::from_str(&raw)
        .with_context(|| format!("parsing {}", receipt_path.display()))?;

    let mut failures = Vec::new();
    let mut blocked_reasons = Vec::new();
    for pointer in REQUIRED_POINTERS {
        if value.pointer(pointer).is_none_or(Value::is_null) {
            failures.push(format!("missing required field {pointer}"));
        }
    }

    if str_at(&value, "/receipt_type") != Some("bench_run") {
        failures.push("receipt_type must be bench_run".to_string());
    }

    let quality_required = bool_at(&value, "/quality_gate/required").unwrap_or(false);
    let quality_passed = bool_at(&value, "/quality_gate/quality_passed").unwrap_or(false);
    let repo_dirty = bool_at(&value, "/repo/dirty").unwrap_or(true);
    let fallback_used = bool_at(&value, "/backend/fallback_used").unwrap_or(true);
    let route_verified = bool_at(&value, "/kernel_route/route_verified").unwrap_or(false);
    let model_contract_matched = bool_at(&value, "/claim_gate/model_contract_matched");
    let resource_envelope_complete = bool_at(&value, "/claim_gate/resource_envelope_complete");
    let claimable = bool_at(&value, "/claim_gate/benchmark_claim_allowed")
        .or_else(|| bool_at(&value, "/claim_gate/claim_allowed"))
        .unwrap_or(false);
    let classification =
        str_at(&value, "/claim_gate/classification").unwrap_or("diagnostic_only").to_string();

    let not_claims = array_strings(&value, "/not_claims");
    for not_claim in CRITICAL_NOT_CLAIMS {
        if !not_claims.iter().any(|value| value == not_claim) {
            failures.push(format!("missing critical not-claim {not_claim}"));
        }
    }

    if quality_required && str_at(&value, "/quality_gate/quality_receipt").unwrap_or("").is_empty()
    {
        failures.push("quality is required but quality_receipt is empty".to_string());
    }
    if claimable && !quality_passed {
        failures.push("benchmark claim allowed while quality_passed=false".to_string());
    }
    if claimable && repo_dirty {
        failures.push("benchmark claim allowed from dirty repo".to_string());
    }
    if claimable && fallback_used {
        failures.push("benchmark claim allowed while fallback_used=true".to_string());
    }
    if claimable && !route_verified {
        failures.push("benchmark claim allowed while route_verified=false".to_string());
    }
    if claimable && model_contract_matched == Some(false) {
        failures.push("benchmark claim allowed while model_contract_matched=false".to_string());
    }
    if claimable && resource_envelope_complete == Some(false) {
        failures.push("benchmark claim allowed while resource_envelope_complete=false".to_string());
    }
    if claimable && classification == "diagnostic_only" {
        failures.push("benchmark claim allowed with diagnostic_only classification".to_string());
    }

    if !quality_passed {
        blocked_reasons.push("quality_not_passed".to_string());
    }
    if repo_dirty {
        blocked_reasons.push("repo_dirty".to_string());
    }
    if fallback_used {
        blocked_reasons.push("fallback_used".to_string());
    }
    if !route_verified {
        blocked_reasons.push("route_not_verified".to_string());
    }
    if model_contract_matched == Some(false) {
        blocked_reasons.push("model_contract_not_matched".to_string());
    }
    if resource_envelope_complete == Some(false) {
        blocked_reasons.push("resource_envelope_incomplete".to_string());
    }
    if require_claimable && !claimable {
        failures.push("receipt is not claimable but --require-claimable was set".to_string());
    }

    Ok(BenchReceiptVerifyReport {
        diagnostic: "bench_receipt_verify",
        producer: "cargo xtask bench verify-receipt",
        receipt_path: receipt_path.display().to_string(),
        passed: failures.is_empty(),
        claimable,
        classification,
        quality_required,
        quality_passed,
        repo_dirty,
        fallback_used,
        route_verified,
        model_contract_matched,
        resource_envelope_complete,
        run_id: str_at(&value, "/run_id").map(ToOwned::to_owned),
        benchmark_profile: str_at(&value, "/benchmark_profile/id").map(ToOwned::to_owned),
        selected_backend: str_at(&value, "/backend/selected_backend").map(ToOwned::to_owned),
        failures,
        blocked_reasons,
        not_claims,
    })
}

fn emit_report(report: &BenchReceiptVerifyReport, format: &str) -> Result<()> {
    match format {
        "json" => println!("{}", serde_json::to_string_pretty(report)?),
        "human" => {
            println!("bench receipt verify: passed={}", report.passed);
            println!("claimable: {}", report.claimable);
            println!("classification: {}", report.classification);
            if !report.blocked_reasons.is_empty() {
                println!("blocked_reasons: {}", report.blocked_reasons.join(", "));
            }
            if !report.failures.is_empty() {
                println!("failures: {}", report.failures.join(", "));
            }
            println!("not_claims: {}", report.not_claims.join(", "));
        }
        other => bail!("unsupported bench receipt output format: {other}"),
    }
    Ok(())
}

fn bool_at(value: &Value, pointer: &str) -> Option<bool> {
    value.pointer(pointer).and_then(Value::as_bool)
}

fn str_at<'a>(value: &'a Value, pointer: &str) -> Option<&'a str> {
    value.pointer(pointer).and_then(Value::as_str)
}

fn array_strings(value: &Value, pointer: &str) -> Vec<String> {
    value
        .pointer(pointer)
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_str).map(ToOwned::to_owned).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt(claim_allowed: bool, quality_passed: bool, dirty: bool) -> Value {
        serde_json::json!({
            "schema_version": 1,
            "receipt_type": "bench_run",
            "run_id": "run-1",
            "repo": {
                "commit": "abc",
                "tree": "def",
                "dirty": dirty,
                "cargo_lock_hash": "sha256:lock",
                "rustc": "rustc 1.95.0",
                "features": ["cpu"]
            },
            "device": {
                "device_slug": "amd-5700x-intel-a770",
                "device_instance_hash": "sha256:device"
            },
            "model": {
                "contract": "docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml",
                "model_id": "microsoft/bitnet-b1.58-2B-4T-gguf",
                "weights_sha256": "sha256:weights",
                "tokenizer_sha256": "sha256:tokenizer",
                "chat_template_hash": "sha256:template"
            },
            "backend": {
                "selected_backend": "intel-arc-a770-opencl",
                "backend_family": "intel-opencl",
                "fallback_used": false
            },
            "kernel_route": {
                "route_verified": true,
                "device_slug": "amd-5700x-intel-a770",
                "selected_backend": "intel-arc-a770-opencl",
                "kernel_variants": [
                    { "op": "qk256_i2s_gemv", "kernel_variant_id": "a770_opencl_qk256_i2s_route_pending_claim_receipts" }
                ]
            },
            "benchmark_profile": {
                "id": "prefill_512_decode_64",
                "profile_hash": "sha256:profile"
            },
            "quality_gate": {
                "required": true,
                "quality_passed": quality_passed,
                "quality_receipt": "quality.json"
            },
            "claim_gate": {
                "benchmark_claim_allowed": claim_allowed,
                "classification": if claim_allowed { "performance_proven" } else { "diagnostic_only" },
                "model_contract_matched": true,
                "resource_envelope_complete": true
            },
            "measurements": { "summary": {} },
            "not_claims": [
                "selected_attention_residency",
                "resident_kv_decode",
                "attention_scores_residency",
                "softmax_residency",
                "attention_value_mix_residency",
                "full_support_op_residency",
                "full_device_residency",
                "completion"
            ]
        })
    }

    fn write_receipt(value: &Value) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("receipt.json");
        fs::write(&path, serde_json::to_string_pretty(value).unwrap()).unwrap();
        (dir, path)
    }

    #[test]
    fn diagnostic_receipt_can_verify_without_claim() {
        let (_dir, path) = write_receipt(&receipt(false, false, true));
        let report = build_verify_report(&path, false).unwrap();
        assert!(report.passed);
        assert!(!report.claimable);
        assert!(report.blocked_reasons.iter().any(|reason| reason == "quality_not_passed"));
    }

    #[test]
    fn rejects_quality_failed_claim() {
        let (_dir, path) = write_receipt(&receipt(true, false, false));
        let report = build_verify_report(&path, false).unwrap();
        assert!(!report.passed);
        assert!(report.failures.iter().any(|failure| failure.contains("quality_passed=false")));
    }

    #[test]
    fn rejects_dirty_claim() {
        let (_dir, path) = write_receipt(&receipt(true, true, true));
        let report = build_verify_report(&path, false).unwrap();
        assert!(!report.passed);
        assert!(report.failures.iter().any(|failure| failure.contains("dirty repo")));
    }
}
