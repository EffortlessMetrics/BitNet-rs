//! Support bundle command for receipt-backed issue reports.

use crate::model_cache::{self, ModelStatusDashboard};

use super::receipts::{self, ReceiptExplanation};
use anyhow::Result;
use clap::{Args, Subcommand, ValueEnum};
use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Collect model status and latest receipt evidence into one pasteable artifact.
#[derive(Args, Debug, Clone)]
pub struct SupportCommand {
    #[command(subcommand)]
    pub action: SupportAction,
}

#[derive(Subcommand, Debug, Clone)]
pub enum SupportAction {
    /// Emit a support bundle from model status and a receipt explanation.
    Bundle {
        /// Receipt file to explain. With --latest, this may be a directory to search.
        #[arg(value_name = "PATH")]
        path: Option<PathBuf>,

        /// Use the newest JSON receipt under the path or default receipt directory.
        #[arg(long, default_value_t = false)]
        latest: bool,

        /// Device label to summarize, for example nvidia-rtx-5070-ti-cuda.
        #[arg(long, value_name = "DEVICE")]
        device: String,

        /// Override the coverage matrix path. Defaults to ci/model-artifacts/model-coverage-matrix.toml when run from this repo.
        #[arg(long, value_name = "PATH", env = "BITNET_MODEL_COVERAGE_MATRIX")]
        matrix: Option<PathBuf>,

        /// Output format.
        #[arg(long, value_enum, default_value_t = SupportBundleFormat::Json)]
        format: SupportBundleFormat,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SupportBundleFormat {
    Text,
    Json,
}

#[derive(Debug, Serialize)]
struct SupportBundle {
    schema_version: u32,
    kind: &'static str,
    created_utc: String,
    device: String,
    summary: SupportBundleSummary,
    binary: SupportBinaryIdentity,
    runtime: SupportRuntimeInfo,
    model_status: ModelStatusDashboard,
    latest_receipt: ReceiptExplanation,
}

#[derive(Debug, Serialize)]
struct SupportBundleSummary {
    model_coverage_row: Option<String>,
    current_tier: Option<String>,
    selected_backend: Option<String>,
    selected_route: Option<String>,
    fallback_used: Option<bool>,
    quality_gate: String,
    server_ready: Option<bool>,
    server_ready_scope: Option<String>,
    speedup_claim: Option<bool>,
    full_residency_claim: Option<bool>,
    bitnet_packed_i2s_qk256_proof: Option<bool>,
    dense_regular_llm_cuda_proof: Option<bool>,
    next_proof: Option<String>,
    claim_boundary: Option<String>,
    receipt_path: String,
}

#[derive(Debug, Serialize)]
struct SupportBinaryIdentity {
    name: &'static str,
    crate_version: &'static str,
    git_commit: Option<String>,
    git_commit_source: Option<&'static str>,
    build_timestamp: Option<&'static str>,
    rustc_version: Option<&'static str>,
    target_triple: Option<&'static str>,
}

#[derive(Debug, Default, Serialize)]
struct SupportRuntimeInfo {
    selected_backend: Option<String>,
    runtime_api: Option<String>,
    device_name: Option<String>,
    driver_version: Option<String>,
    cuda_runtime_version: Option<String>,
    cuda_driver_version: Option<String>,
    source: Option<&'static str>,
}

impl SupportCommand {
    pub async fn execute(&self) -> Result<()> {
        match &self.action {
            SupportAction::Bundle { path, latest, device, matrix, format } => {
                let bundle = support_bundle(
                    path.as_deref(),
                    *latest,
                    device,
                    matrix.clone(),
                    current_timestamp_utc(),
                )?;
                match format {
                    SupportBundleFormat::Json => {
                        println!("{}", serde_json::to_string_pretty(&bundle)?);
                    }
                    SupportBundleFormat::Text => print_support_bundle_text(&bundle),
                }
                Ok(())
            }
        }
    }
}

fn support_bundle(
    receipt_path: Option<&Path>,
    latest: bool,
    device: &str,
    matrix: Option<PathBuf>,
    created_utc: String,
) -> Result<SupportBundle> {
    let model_status = model_cache::model_status_dashboard_for_device(device, matrix)?;
    let receipt_path = receipts::resolve_receipt_path(receipt_path, latest)?;
    let receipt = receipts::read_receipt_json(&receipt_path)?;
    let latest_receipt = receipts::explain_receipt(&receipt_path, &receipt);
    let summary = support_summary(&latest_receipt, &model_status);
    let runtime = runtime_info(&latest_receipt, &receipt);

    Ok(SupportBundle {
        schema_version: 1,
        kind: "bitnet_support_bundle",
        created_utc,
        device: device.to_string(),
        summary,
        binary: binary_identity(),
        runtime,
        model_status,
        latest_receipt,
    })
}

fn support_summary(
    receipt: &ReceiptExplanation,
    model_status: &ModelStatusDashboard,
) -> SupportBundleSummary {
    let next_proof = receipt
        .model_coverage_row
        .as_deref()
        .and_then(|row| model_status.next_proof_for_row(row))
        .map(ToOwned::to_owned)
        .or_else(|| receipt.model_coverage.claim_boundary.clone());

    SupportBundleSummary {
        model_coverage_row: receipt.model_coverage_row.clone(),
        current_tier: receipt.current_tier.clone(),
        selected_backend: receipt.selected_backend.clone(),
        selected_route: receipt.selected_route.clone(),
        fallback_used: receipt.fallback_used,
        quality_gate: quality_gate_status(receipt).to_string(),
        server_ready: receipt.server_ready,
        server_ready_scope: receipt.server_ready_scope.clone(),
        speedup_claim: receipt.speedup_claim,
        full_residency_claim: receipt.full_residency_claim,
        bitnet_packed_i2s_qk256_proof: receipt.bitnet_packed_i2s_qk256_proof,
        dense_regular_llm_cuda_proof: receipt.dense_regular_llm_cuda_proof,
        next_proof,
        claim_boundary: receipt.model_coverage.claim_boundary.clone(),
        receipt_path: receipt.path.clone(),
    }
}

fn quality_gate_status(receipt: &ReceiptExplanation) -> &'static str {
    let gates = [
        receipt.quality.answer_quality_passed,
        receipt.quality.benchmark_quality_passed,
        receipt.quality.parity_passed,
    ];
    if gates.contains(&Some(false)) {
        "blocked"
    } else if gates.contains(&Some(true)) {
        "passed"
    } else {
        "not_available"
    }
}

fn runtime_info(explanation: &ReceiptExplanation, receipt: &Value) -> SupportRuntimeInfo {
    let info = SupportRuntimeInfo {
        selected_backend: explanation.selected_backend.clone(),
        runtime_api: explanation.backend.runtime_api.clone(),
        device_name: string_at_any(
            receipt,
            &[
                &["backend", "device_name"],
                &["backend_runtime", "device_name"],
                &["hardware", "device_name"],
                &["device", "name"],
            ],
        ),
        driver_version: string_at_any(
            receipt,
            &[
                &["backend", "driver_version"],
                &["backend_runtime", "driver_version"],
                &["hardware", "driver_version"],
                &["driver_version"],
            ],
        ),
        cuda_runtime_version: string_at_any(
            receipt,
            &[
                &["backend", "cuda_runtime_version"],
                &["backend_runtime", "cuda_runtime_version"],
                &["hardware", "cuda_runtime_version"],
                &["cuda_runtime_version"],
            ],
        ),
        cuda_driver_version: string_at_any(
            receipt,
            &[
                &["backend", "cuda_driver_version"],
                &["backend_runtime", "cuda_driver_version"],
                &["hardware", "cuda_driver_version"],
                &["cuda_driver_version"],
            ],
        ),
        source: None,
    };

    if info.selected_backend.is_some()
        || info.runtime_api.is_some()
        || info.device_name.is_some()
        || info.driver_version.is_some()
        || info.cuda_runtime_version.is_some()
        || info.cuda_driver_version.is_some()
    {
        SupportRuntimeInfo { source: Some("latest_receipt"), ..info }
    } else {
        info
    }
}

fn binary_identity() -> SupportBinaryIdentity {
    let git_commit = option_env!("GITHUB_SHA")
        .filter(|sha| !sha.trim().is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            option_env!("VERGEN_GIT_SHA")
                .filter(|sha| !sha.trim().is_empty())
                .map(ToOwned::to_owned)
        });
    let git_commit_source = if option_env!("GITHUB_SHA").is_some_and(|sha| !sha.trim().is_empty()) {
        Some("GITHUB_SHA")
    } else if option_env!("VERGEN_GIT_SHA").is_some_and(|sha| !sha.trim().is_empty()) {
        Some("VERGEN_GIT_SHA")
    } else {
        None
    };

    SupportBinaryIdentity {
        name: env!("CARGO_PKG_NAME"),
        crate_version: env!("CARGO_PKG_VERSION"),
        git_commit,
        git_commit_source,
        build_timestamp: option_env!("BITNET_BUILD_TS").or(option_env!("VERGEN_BUILD_TIMESTAMP")),
        rustc_version: option_env!("VERGEN_RUSTC_SEMVER"),
        target_triple: option_env!("VERGEN_CARGO_TARGET_TRIPLE"),
    }
}

fn current_timestamp_utc() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn print_support_bundle_text(bundle: &SupportBundle) {
    println!("BitNet support bundle");
    println!("  device: {}", bundle.device);
    println!("  receipt: {}", bundle.summary.receipt_path);
    if let Some(row) = &bundle.summary.model_coverage_row {
        println!("  model_coverage_row: {row}");
    }
    if let Some(route) = &bundle.summary.selected_route {
        println!("  selected_route: {route}");
    }
    if let Some(backend) = &bundle.summary.selected_backend {
        println!("  selected_backend: {backend}");
    }
    if let Some(fallback) = bundle.summary.fallback_used {
        println!("  fallback_used: {fallback}");
    }
    println!("  quality_gate: {}", bundle.summary.quality_gate);
    println!("  binary: {} {}", bundle.binary.name, bundle.binary.crate_version);
}

fn string_at_any(value: &Value, paths: &[&[&str]]) -> Option<String> {
    paths.iter().find_map(|path| string_at(value, path))
}

fn string_at(value: &Value, path: &[&str]) -> Option<String> {
    let mut current = value;
    for segment in path {
        current = current.get(*segment)?;
    }
    current.as_str().map(ToOwned::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;
    use clap::Parser;
    use serde_json::json;
    use std::fs;

    #[derive(Parser, Debug)]
    struct SupportActionParser {
        #[command(subcommand)]
        action: SupportAction,
    }

    fn model_matrix_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("ci")
            .join("model-artifacts")
            .join("model-coverage-matrix.toml")
    }

    fn support_bundle_value_for_receipt(receipt_name: &str, receipt: Value) -> Result<Value> {
        let dir = tempfile::tempdir()?;
        let receipt_path = dir.path().join(receipt_name);
        fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;

        let bundle = support_bundle(
            Some(&receipt_path),
            false,
            "nvidia-rtx-5070-ti-cuda",
            Some(model_matrix_path()),
            "2026-05-20T00:00:00Z".to_string(),
        )?;
        serde_json::to_value(&bundle).context("support bundle must serialize to JSON")
    }

    #[test]
    fn support_bundle_accepts_latest_device_and_format_json() -> Result<()> {
        let parsed = SupportActionParser::try_parse_from([
            "support",
            "bundle",
            "--latest",
            "--device",
            "nvidia-rtx-5070-ti-cuda",
            "--format",
            "json",
        ])?;

        let SupportAction::Bundle { latest, device, format, .. } = parsed.action;
        assert!(latest);
        assert_eq!(device, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(format, SupportBundleFormat::Json);
        Ok(())
    }

    #[test]
    fn support_bundle_combines_model_status_latest_receipt_and_summary() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let receipt_path = dir.path().join("qwen25-receipt.json");
        let receipt = json!({
            "artifact_kind": "dense_gguf_qwen_warm_session_strict_cuda_proof",
            "claim": "dense_gguf_qwen_warm_session_strict_cuda_proof_recorded",
            "model": {
                "id": "qwen2.5-0.5b-instruct-q8_0"
            },
            "backend": {
                "selected_backend": "nvidia-rtx-5070-ti-cuda",
                "runtime_api": "cuda",
                "fallback_used": false,
                "device_name": "NVIDIA GeForce RTX 5070 Ti",
                "driver_version": "test-driver",
                "cuda_runtime_version": "test-cuda-runtime"
            },
            "execution_plan": {
                "selected_route": "dense_regular_llm_cuda",
                "model_family": "qwen",
                "requested_backend": "nvidia-rtx-5070-ti-cuda",
                "speedup_claim": false
            },
            "quality_gate": {
                "passed": true
            },
            "claim_boundary": {
                "bitnet_packed_i2s_qk256_proof": false,
                "dense_regular_llm_cuda_claimed": true,
                "speedup_claim": false,
                "full_cuda_residency_claimed": false
            }
        });
        fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;

        let bundle = support_bundle(
            Some(dir.path()),
            true,
            "nvidia-rtx-5070-ti-cuda",
            Some(model_matrix_path()),
            "2026-05-18T00:00:00Z".to_string(),
        )?;
        let value = serde_json::to_value(&bundle)?;

        assert_eq!(value["kind"], "bitnet_support_bundle");
        assert_eq!(value["device"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(value["summary"]["model_coverage_row"], "dense_qwen25_05b_q8_cuda");
        assert_eq!(value["summary"]["current_tier"], "product_cli_ready");
        assert_eq!(value["summary"]["selected_backend"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(value["summary"]["selected_route"], "dense_regular_llm_cuda");
        assert_eq!(value["summary"]["fallback_used"], false);
        assert_eq!(value["summary"]["quality_gate"], "passed");
        assert_eq!(value["summary"]["server_ready"], true);
        assert_eq!(value["summary"]["server_ready_scope"], "exact_profile");
        assert_eq!(value["summary"]["speedup_claim"], false);
        assert_eq!(value["summary"]["full_residency_claim"], false);
        assert_eq!(value["summary"]["bitnet_packed_i2s_qk256_proof"], false);
        assert_eq!(value["summary"]["dense_regular_llm_cuda_proof"], true);
        assert!(
            value["summary"]["next_proof"]
                .as_str()
                .context("support summary next_proof must be a string")?
                .contains("device-side top-k or greedy sampler receipt")
        );
        assert!(
            value["summary"]["claim_boundary"]
                .as_str()
                .context("support summary claim_boundary must be a string")?
                .contains("BitNet packed I2_S/QK256 proof")
        );
        assert_eq!(value["runtime"]["runtime_api"], "cuda");
        assert_eq!(value["runtime"]["driver_version"], "test-driver");
        assert_eq!(value["runtime"]["cuda_runtime_version"], "test-cuda-runtime");
        assert_eq!(value["runtime"]["source"], "latest_receipt");
        assert_eq!(value["latest_receipt"]["model_coverage_row"], "dense_qwen25_05b_q8_cuda");
        assert_eq!(value["latest_receipt"]["server_ready_scope"], "exact_profile");

        let models = value["model_status"]["models"]
            .as_array()
            .context("support bundle model_status.models must be an array")?;
        assert!(models.iter().any(|model| {
            model["model_coverage_row"] == "bitnet_official_2b_i2s_qk256"
                && model["bitnet_packed_i2s_qk256_proof"] == true
                && model["dense_regular_llm_cuda_proof"] == false
        }));
        assert!(models.iter().any(|model| {
            model["model_coverage_row"] == "dense_qwen25_05b_q8_cuda"
                && model["dense_regular_llm_cuda_proof"] == true
                && model["bitnet_packed_i2s_qk256_proof"] == false
        }));
        Ok(())
    }

    #[test]
    fn support_bundle_preserves_official_bitnet_qk256_boundaries() -> Result<()> {
        let value = support_bundle_value_for_receipt(
            "bitnet-i2s-qk256-receipt.json",
            json!({
                "artifact_kind": "bitnet_cuda_answer",
                "claim": "bitnet_cuda_answer_recorded",
                "model": {
                    "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
                    "file": "ggml-model-i2_s.gguf"
                },
                "backend": {
                    "selected_backend": "nvidia-rtx-5070-ti-cuda",
                    "runtime_api": "cuda",
                    "fallback_used": false,
                    "device_name": "NVIDIA GeForce RTX 5070 Ti"
                },
                "execution_plan": {
                    "selected_route": "bitnet_qk256_cuda",
                    "model_family": "bitnet_b1_58",
                    "requested_backend": "nvidia-rtx-5070-ti-cuda",
                    "speedup_claim": false
                },
                "quality_gate": {
                    "passed": true
                },
                "claim_boundary": {
                    "bitnet_packed_i2s_qk256_proof": true,
                    "dense_regular_llm_cuda_claimed": false,
                    "speedup_claim": false,
                    "full_cuda_residency_claimed": false
                }
            }),
        )?;

        assert_eq!(value["summary"]["model_coverage_row"], "bitnet_official_2b_i2s_qk256");
        assert_eq!(value["summary"]["current_tier"], "product_cli_ready");
        assert_eq!(value["summary"]["selected_backend"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(value["summary"]["selected_route"], "bitnet_qk256_cuda");
        assert_eq!(value["summary"]["fallback_used"], false);
        assert_eq!(value["summary"]["quality_gate"], "passed");
        assert_eq!(value["summary"]["server_ready"], false);
        assert!(value["summary"]["server_ready_scope"].is_null());
        assert_eq!(value["summary"]["speedup_claim"], false);
        assert_eq!(value["summary"]["full_residency_claim"], false);
        assert_eq!(value["summary"]["bitnet_packed_i2s_qk256_proof"], true);
        assert_eq!(value["summary"]["dense_regular_llm_cuda_proof"], false);
        assert!(
            value["summary"]["next_proof"]
                .as_str()
                .context("BitNet support summary next_proof must be a string")?
                .contains("profile-specific speedup qualification")
        );
        assert!(
            value["summary"]["claim_boundary"]
                .as_str()
                .context("BitNet support summary claim_boundary must be a string")?
                .contains("does not prove dense regular-LLM CUDA")
        );
        assert_eq!(value["runtime"]["runtime_api"], "cuda");
        assert_eq!(value["runtime"]["device_name"], "NVIDIA GeForce RTX 5070 Ti");
        assert_eq!(value["latest_receipt"]["model_coverage_row"], "bitnet_official_2b_i2s_qk256");
        assert_eq!(value["latest_receipt"]["bitnet_packed_i2s_qk256_proof"], true);
        assert_eq!(value["latest_receipt"]["dense_regular_llm_cuda_proof"], false);
        Ok(())
    }

    #[test]
    fn support_bundle_preserves_qwen3_dense_product_boundaries() -> Result<()> {
        let value = support_bundle_value_for_receipt(
            "qwen3-dense-cuda-receipt.json",
            json!({
                "artifact_kind": "dense_gguf_qwen_chat_strict_cuda_proof",
                "claim": "dense_gguf_qwen_chat_strict_cuda_proof_recorded",
                "model": {
                    "id": "qwen3-0.6b-instruct-q8_0",
                    "file": "Qwen3-0.6B-Q8_0.gguf",
                    "architecture": "qwen3"
                },
                "backend": {
                    "selected_backend": "nvidia-rtx-5070-ti-cuda",
                    "runtime_api": "cuda",
                    "fallback_used": false,
                    "device_name": "NVIDIA GeForce RTX 5070 Ti"
                },
                "execution_plan": {
                    "selected_route": "dense_regular_llm_cuda",
                    "model_family": "qwen",
                    "requested_backend": "nvidia-rtx-5070-ti-cuda",
                    "speedup_claim": false
                },
                "quality_gate": {
                    "passed": true
                },
                "claim_boundary": {
                    "bitnet_packed_i2s_qk256_proof": false,
                    "dense_regular_llm_cuda_claimed": true,
                    "speedup_claim": false,
                    "full_cuda_residency_claimed": false
                }
            }),
        )?;

        assert_eq!(value["summary"]["model_coverage_row"], "dense_qwen3_06b_q8_candidate");
        assert_eq!(value["summary"]["current_tier"], "product_cli_ready");
        assert_eq!(value["summary"]["selected_backend"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(value["summary"]["selected_route"], "dense_regular_llm_cuda");
        assert_eq!(value["summary"]["fallback_used"], false);
        assert_eq!(value["summary"]["quality_gate"], "passed");
        assert_eq!(value["summary"]["server_ready"], true);
        assert_eq!(value["summary"]["server_ready_scope"], "exact_profile");
        assert_eq!(value["summary"]["speedup_claim"], false);
        assert_eq!(value["summary"]["full_residency_claim"], false);
        assert_eq!(value["summary"]["bitnet_packed_i2s_qk256_proof"], false);
        assert_eq!(value["summary"]["dense_regular_llm_cuda_proof"], true);
        assert!(
            value["summary"]["next_proof"]
                .as_str()
                .context("Qwen3 support summary next_proof must be a string")?
                .contains("hardware aggregate receipt")
        );
        let claim_boundary = value["summary"]["claim_boundary"]
            .as_str()
            .context("Qwen3 support summary claim_boundary must be a string")?;
        assert!(claim_boundary.contains("does not inherit Qwen2.5 CUDA receipts"));
        assert!(claim_boundary.contains("BitNet QK256 behavior"));
        assert_eq!(value["latest_receipt"]["model_coverage_row"], "dense_qwen3_06b_q8_candidate");
        assert_eq!(value["latest_receipt"]["bitnet_packed_i2s_qk256_proof"], false);
        assert_eq!(value["latest_receipt"]["dense_regular_llm_cuda_proof"], true);
        Ok(())
    }

    #[test]
    fn cuda_support_issue_template_requires_bundle_json_and_claim_boundaries() -> Result<()> {
        let template_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".github")
            .join("ISSUE_TEMPLATE")
            .join("cuda-support.yml");
        let template: serde_yaml::Value = serde_yaml::from_slice(
            &fs::read(&template_path)
                .with_context(|| format!("read {}", template_path.display()))?,
        )?;

        assert_eq!(template["name"].as_str(), Some("CUDA Support"));
        let labels = template["labels"]
            .as_sequence()
            .context("cuda support issue labels must be a sequence")?;
        assert!(labels.iter().any(|label| label.as_str() == Some("support")));
        assert!(labels.iter().any(|label| label.as_str() == Some("cuda")));

        let body =
            template["body"].as_sequence().context("cuda support issue body must be a sequence")?;
        let support_bundle_index = body
            .iter()
            .position(|item| item["id"].as_str() == Some("support-bundle"))
            .context("cuda support issue template must include support-bundle field")?;
        let issue_index = body
            .iter()
            .position(|item| item["id"].as_str() == Some("issue"))
            .context("cuda support issue template must include issue field")?;
        assert!(
            support_bundle_index < issue_index,
            "support bundle must be requested before free-form issue prose"
        );

        let support_bundle = &body[support_bundle_index];
        assert_eq!(support_bundle["type"].as_str(), Some("textarea"));
        assert_eq!(support_bundle["attributes"]["render"].as_str(), Some("json"));
        assert_eq!(support_bundle["validations"]["required"].as_bool(), Some(true));
        let description = support_bundle["attributes"]["description"]
            .as_str()
            .context("support-bundle description must be a string")?;
        assert!(description.contains(
            "bitnet support bundle --latest --device nvidia-rtx-5070-ti-cuda --format json"
        ));
        let placeholder = support_bundle["attributes"]["placeholder"]
            .as_str()
            .context("support-bundle placeholder must be a string")?;
        for required_fragment in [
            "\"kind\": \"bitnet_support_bundle\"",
            "\"selected_backend\": \"nvidia-rtx-5070-ti-cuda\"",
            "\"selected_route\":",
            "\"fallback_used\": false",
            "\"speedup_claim\": false",
            "\"claim_boundary\":",
        ] {
            assert!(
                placeholder.contains(required_fragment),
                "support-bundle placeholder missing {required_fragment}"
            );
        }

        let claim_boundaries = body
            .iter()
            .find(|item| item["id"].as_str() == Some("claim-boundaries"))
            .context("cuda support issue template must include claim-boundaries")?;
        let options = claim_boundaries["attributes"]["options"]
            .as_sequence()
            .context("claim boundary options must be a sequence")?;
        let option_labels = options
            .iter()
            .filter_map(|option| option["label"].as_str())
            .collect::<Vec<_>>()
            .join("\n");
        for required_boundary in [
            "selected backend is `nvidia-rtx-5070-ti-cuda`, not generic `cuda`",
            "`fallback_used=false`",
            "`speedup_claim=false`",
            "Qwen2.5 exact-profile server readiness is not being treated as broad dense GGUF server readiness",
            "Dense CUDA proof is not being treated as BitNet I2_S/QK256 proof",
            "Qwen2.5 evidence is not being treated as Qwen3 evidence",
        ] {
            assert!(
                option_labels.contains(required_boundary),
                "claim boundary checklist missing {required_boundary}"
            );
        }

        Ok(())
    }
}
