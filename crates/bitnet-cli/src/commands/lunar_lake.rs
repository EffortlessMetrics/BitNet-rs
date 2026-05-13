//! Lunar Lake operator readiness helpers.
//!
//! These commands do not run inference. They turn the existing 258V proof bundle
//! into an operator-facing route/readiness artifact so users can see which path
//! is the safe default and which accelerator paths remain bounded candidates.

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_ARTIFACT_ROOT: &str = "ci/hardware/intel-258v/2026-05-08";

const DENSE_CPU_ANSWER: &str = "slm-answer-corpus-qwen25-cpu-clean-provenance.json";
const DENSE_CPU_PHASE: &str = "slm-phase-warm-session-qwen25-cpu.json";
const DENSE_OV_PHASE: &str = "slm-openvino-cpu-gpu-npu-phase-runner.json";
const DENSE_OV_CPU: &str = "slm-openvino-cpu-llmpipeline-smoke.json";
const DENSE_OV_GPU: &str = "slm-openvino-gpu-arc140v-llmpipeline-smoke.json";
const DENSE_OV_NPU: &str = "slm-openvino-npu-llmpipeline-smoke.json";
const BITNET_CPU_BUNDLE: &str = "cpu-reference-bundle-after-semantic-fix.json";
const BITNET_REFERENCE: &str = "cpu-bitnet-ref-001-external-boundary.json";
const BITNET_PERF_MICRO: &str = "cpu-bitnet-perf-001-i2s-microbench.json";
const BITNET_PERF_TILING: &str = "cpu-bitnet-perf-002-i2s-tiling-matrix.json";
const ARC_OPENCL_PARITY: &str = "arc-140v-opencl-parity.json";
const NPU_RMSNORM: &str = "npu-bitnet-rmsnorm-subgraph-parity.json";
const NPU_LINEAR: &str = "npu-bitnet-linear-projection-subgraph-parity.json";
const NPU_FFN: &str = "npu-bitnet-ffn-subgraph-parity.json";
const OPERATOR_READINESS: &str = "lunar-lake-operator-readiness.json";
const REGRESSION_BUNDLE: &str = "lunar-lake-regression-bundle.json";

/// Lunar Lake operator commands.
#[derive(Args, Debug, Clone)]
pub struct LunarLakeCommand {
    #[command(subcommand)]
    pub action: LunarLakeAction,
}

#[derive(Subcommand, Debug, Clone)]
pub enum LunarLakeAction {
    /// Validate the committed Lunar Lake artifact bundle and emit route policy.
    Validate {
        /// Artifact root containing the 258V receipts to index.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// Output JSON readiness receipt to file.
        #[arg(long)]
        json_out: Option<PathBuf>,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when required operator evidence is missing or fallback is observed.
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Check the Lunar Lake operator receipt for drift and emit a regression bundle.
    Regress {
        /// Artifact root containing the 258V receipts to index.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// Operator readiness receipt to verify. Relative paths are resolved under artifact-root.
        #[arg(long, default_value = OPERATOR_READINESS)]
        operator_receipt: PathBuf,

        /// Output JSON regression bundle to file.
        #[arg(long)]
        json_out: Option<PathBuf>,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when the regression bundle reports drift.
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Compare Lunar Lake operator routes and bounded evidence.
    Compare {
        /// Artifact root containing the 258V receipts to index.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// Operator readiness receipt to compare. Relative paths are resolved under artifact-root.
        #[arg(long, default_value = OPERATOR_READINESS)]
        operator_receipt: PathBuf,

        /// Regression bundle to compare. Relative paths are resolved under artifact-root.
        #[arg(long, default_value = REGRESSION_BUNDLE)]
        regression_bundle: PathBuf,

        /// Output JSON comparison receipt to file.
        #[arg(long)]
        json_out: Option<PathBuf>,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when the comparison receipt reports drift.
        #[arg(long, default_value_t = false)]
        strict: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LunarLakeOperatorReceipt {
    pub schema_version: String,
    pub artifact_kind: String,
    pub proof_stage: String,
    pub created_utc: String,
    pub machine_id: String,
    pub artifact_root: String,
    pub operator_ready: bool,
    pub default_route: OperatorRoute,
    pub routes: Vec<OperatorRoute>,
    pub evidence: Vec<EvidenceStatus>,
    pub gaps: Vec<String>,
    pub claim_boundary: ClaimBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OperatorRoute {
    pub route_id: String,
    pub workload: String,
    pub selected_model: String,
    pub selected_backend: String,
    pub runtime_api: String,
    pub selected_kernel_or_runtime: String,
    pub fallback_policy: String,
    pub route_reason: String,
    pub answer_gate_evidence: Option<String>,
    pub phase_evidence: Option<String>,
    pub acceleration_claim: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvidenceStatus {
    pub evidence_id: String,
    pub path: String,
    pub present: bool,
    pub artifact_kind: Option<String>,
    pub requested_backend: Option<String>,
    pub selected_backend: Option<String>,
    pub runtime_api: Option<String>,
    pub fallback_used: Option<bool>,
    pub answer_gate_passed: Option<bool>,
    pub phase_timing_present: Option<bool>,
    pub speedup_claim: Option<bool>,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClaimBoundary {
    pub cpu_is_truth_path: bool,
    pub dense_slm_default_is_cpu_until_speedup_qualified: bool,
    pub openvino_gpu_npu_are_candidates_not_speedup_claims: bool,
    pub arc_bitnet_full_inference_claimed: bool,
    pub npu_bitnet_full_inference_claimed: bool,
    pub qk256_accelerator_decode_claimed: bool,
    pub hidden_fallback_allowed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LunarLakeRegressionBundle {
    pub schema_version: String,
    pub artifact_kind: String,
    pub proof_stage: String,
    pub created_utc: String,
    pub machine_id: String,
    pub artifact_root: String,
    pub operator_receipt: String,
    pub regression_passed: bool,
    pub checks: Vec<RegressionCheck>,
    pub gaps: Vec<String>,
    pub claim_boundary: ClaimBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegressionCheck {
    pub check_id: String,
    pub status: String,
    pub evidence: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LunarLakeComparisonReceipt {
    pub schema_version: String,
    pub artifact_kind: String,
    pub proof_stage: String,
    pub created_utc: String,
    pub machine_id: String,
    pub artifact_root: String,
    pub operator_receipt: String,
    pub regression_bundle: String,
    pub comparison_ready: bool,
    pub operator_ready: bool,
    pub regression_passed: bool,
    pub default_route_id: String,
    pub routes: Vec<RouteComparison>,
    pub evidence: Vec<EvidenceStatus>,
    pub checks: Vec<RegressionCheck>,
    pub gaps: Vec<String>,
    pub claim_boundary: ClaimBoundary,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RouteComparison {
    pub route_id: String,
    pub role: String,
    pub workload: String,
    pub selected_model: String,
    pub selected_backend: String,
    pub runtime_api: String,
    pub selected_kernel_or_runtime: String,
    pub fallback_policy: String,
    pub answer_gate_evidence: Option<String>,
    pub phase_evidence: Option<String>,
    pub evidence_ready: bool,
    pub acceleration_claim: bool,
    pub route_reason: String,
    pub notes: Vec<String>,
}

impl LunarLakeCommand {
    pub async fn execute(&self) -> Result<()> {
        match &self.action {
            LunarLakeAction::Validate { artifact_root, json_out, created_utc, strict } => {
                let receipt = match created_utc {
                    Some(created_utc) => {
                        let created_utc = normalize_created_utc(created_utc)?;
                        build_operator_readiness_receipt_with_created_utc(
                            artifact_root,
                            created_utc,
                        )?
                    }
                    None => build_operator_readiness_receipt(artifact_root)?,
                };
                write_or_print_receipt(&receipt, json_out.as_deref())?;
                if *strict && !receipt.operator_ready {
                    bail!("Lunar Lake operator readiness failed: {}", receipt.gaps.join("; "));
                }
                Ok(())
            }
            LunarLakeAction::Regress {
                artifact_root,
                operator_receipt,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_regression_bundle_with_created_utc(
                    artifact_root,
                    operator_receipt,
                    created_utc,
                )?;
                write_or_print_regression_bundle(&receipt, json_out.as_deref())?;
                if *strict && !receipt.regression_passed {
                    bail!("Lunar Lake regression bundle failed: {}", receipt.gaps.join("; "));
                }
                Ok(())
            }
            LunarLakeAction::Compare {
                artifact_root,
                operator_receipt,
                regression_bundle,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_comparison_receipt_with_created_utc(
                    artifact_root,
                    operator_receipt,
                    regression_bundle,
                    created_utc,
                )?;
                write_or_print_comparison_receipt(&receipt, json_out.as_deref())?;
                if *strict && !receipt.comparison_ready {
                    bail!("Lunar Lake comparison failed: {}", receipt.gaps.join("; "));
                }
                Ok(())
            }
        }
    }
}

pub fn build_operator_readiness_receipt(root: &Path) -> Result<LunarLakeOperatorReceipt> {
    build_operator_readiness_receipt_with_created_utc(
        root,
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
    )
}

pub fn build_operator_readiness_receipt_with_created_utc(
    root: &Path,
    created_utc: String,
) -> Result<LunarLakeOperatorReceipt> {
    let evidence = vec![
        inspect_receipt(
            root,
            "dense_slm_cpu_answer",
            DENSE_CPU_ANSWER,
            EvidenceExpectation::Answer,
        )?,
        inspect_receipt(root, "dense_slm_cpu_phase", DENSE_CPU_PHASE, EvidenceExpectation::Phase)?,
        inspect_receipt(root, "dense_slm_openvino_cpu", DENSE_OV_CPU, EvidenceExpectation::Answer)?,
        inspect_receipt(
            root,
            "dense_slm_openvino_gpu_arc140v",
            DENSE_OV_GPU,
            EvidenceExpectation::Answer,
        )?,
        inspect_receipt(root, "dense_slm_openvino_npu", DENSE_OV_NPU, EvidenceExpectation::Answer)?,
        inspect_receipt(
            root,
            "dense_slm_openvino_phase_runner",
            DENSE_OV_PHASE,
            EvidenceExpectation::AnswerAndPhase,
        )?,
        inspect_receipt(
            root,
            "bitnet_cpu_reference_bundle",
            BITNET_CPU_BUNDLE,
            EvidenceExpectation::Present,
        )?,
        inspect_receipt(
            root,
            "bitnet_external_reference_boundary",
            BITNET_REFERENCE,
            EvidenceExpectation::Present,
        )?,
        inspect_receipt(
            root,
            "bitnet_i2s_gemv_gemm_microbench",
            BITNET_PERF_MICRO,
            EvidenceExpectation::NoSpeedupClaim,
        )?,
        inspect_receipt(
            root,
            "bitnet_i2s_tiling_thread_matrix",
            BITNET_PERF_TILING,
            EvidenceExpectation::NoSpeedupClaim,
        )?,
        inspect_receipt(
            root,
            "arc140v_native_opencl_parity",
            ARC_OPENCL_PARITY,
            EvidenceExpectation::Present,
        )?,
        inspect_receipt(
            root,
            "npu_rmsnorm_static_subgraph",
            NPU_RMSNORM,
            EvidenceExpectation::Present,
        )?,
        inspect_receipt(
            root,
            "npu_linear_static_subgraph",
            NPU_LINEAR,
            EvidenceExpectation::Present,
        )?,
        inspect_receipt(root, "npu_ffn_static_subgraph", NPU_FFN, EvidenceExpectation::Present)?,
    ];

    let dense_cpu_ready = evidence_ok(&evidence, "dense_slm_cpu_answer")
        && evidence_ok(&evidence, "dense_slm_cpu_phase");
    let dense_openvino_ready = evidence_ok(&evidence, "dense_slm_openvino_cpu")
        && evidence_ok(&evidence, "dense_slm_openvino_gpu_arc140v")
        && evidence_ok(&evidence, "dense_slm_openvino_npu")
        && evidence_ok(&evidence, "dense_slm_openvino_phase_runner");
    let bitnet_cpu_ready = evidence_ok(&evidence, "bitnet_cpu_reference_bundle")
        && evidence_ok(&evidence, "bitnet_external_reference_boundary")
        && evidence_ok(&evidence, "bitnet_i2s_gemv_gemm_microbench")
        && evidence_ok(&evidence, "bitnet_i2s_tiling_thread_matrix");
    let arc_npu_bounded_ready = evidence_ok(&evidence, "arc140v_native_opencl_parity")
        && evidence_ok(&evidence, "npu_rmsnorm_static_subgraph")
        && evidence_ok(&evidence, "npu_linear_static_subgraph")
        && evidence_ok(&evidence, "npu_ffn_static_subgraph");

    let mut gaps = Vec::new();
    for item in &evidence {
        if !item.issues.is_empty() {
            gaps.push(format!("{}: {}", item.evidence_id, item.issues.join(", ")));
        }
    }
    if !dense_cpu_ready {
        gaps.push("dense SLM CPU answer/phase baseline is not operator-ready".to_string());
    }
    if !dense_openvino_ready {
        gaps.push("dense SLM OpenVINO CPU/GPU/NPU candidate evidence is incomplete".to_string());
    }
    if !bitnet_cpu_ready {
        gaps.push("BitNet CPU reference/performance evidence is incomplete".to_string());
    }
    if !arc_npu_bounded_ready {
        gaps.push("Arc/NPU bounded parity evidence is incomplete".to_string());
    }

    let default_route = dense_slm_cpu_route();
    let routes = vec![
        default_route.clone(),
        bitnet_cpu_route(),
        openvino_gpu_candidate_route(),
        openvino_npu_candidate_route(),
    ];

    Ok(LunarLakeOperatorReceipt {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_operator_readiness".to_string(),
        proof_stage: "operator_routes_indexed".to_string(),
        created_utc,
        machine_id: "intel-258v".to_string(),
        artifact_root: path_string(root),
        operator_ready: gaps.is_empty(),
        default_route,
        routes,
        evidence,
        gaps,
        claim_boundary: ClaimBoundary {
            cpu_is_truth_path: true,
            dense_slm_default_is_cpu_until_speedup_qualified: true,
            openvino_gpu_npu_are_candidates_not_speedup_claims: true,
            arc_bitnet_full_inference_claimed: false,
            npu_bitnet_full_inference_claimed: false,
            qk256_accelerator_decode_claimed: false,
            hidden_fallback_allowed: false,
        },
    })
}

pub fn build_regression_bundle_with_created_utc(
    root: &Path,
    operator_receipt: &Path,
    created_utc: String,
) -> Result<LunarLakeRegressionBundle> {
    let operator_receipt_path = resolve_receipt_path(root, operator_receipt);
    let bytes = fs::read(&operator_receipt_path)
        .with_context(|| format!("failed to read {}", operator_receipt_path.display()))?;
    let operator: LunarLakeOperatorReceipt = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", operator_receipt_path.display()))?;

    let checks = vec![
        regression_check(
            "operator_receipt_ready",
            operator.operator_ready,
            vec![OPERATOR_READINESS],
            if operator.operator_ready {
                vec!["operator readiness receipt reports operator_ready=true".to_string()]
            } else {
                operator.gaps.clone()
            },
        ),
        regression_check(
            "dense_slm_default_cpu_route",
            operator.default_route.route_id == "dense_slm_default_cpu"
                && operator.default_route.selected_backend == "cpu-rust"
                && operator.default_route.runtime_api == "cpu"
                && !operator.default_route.acceleration_claim
                && evidence_ok(&operator.evidence, "dense_slm_cpu_answer")
                && evidence_ok(&operator.evidence, "dense_slm_cpu_phase"),
            vec![DENSE_CPU_ANSWER, DENSE_CPU_PHASE],
            vec![
                format!("default_route={}", operator.default_route.route_id),
                format!("selected_backend={}", operator.default_route.selected_backend),
            ],
        ),
        regression_check(
            "bitnet_cpu_reference_route",
            route_ok(&operator, "bitnet_reference_cpu")
                && evidence_ok(&operator.evidence, "bitnet_cpu_reference_bundle")
                && evidence_ok(&operator.evidence, "bitnet_external_reference_boundary")
                && evidence_ok(&operator.evidence, "bitnet_i2s_gemv_gemm_microbench")
                && evidence_ok(&operator.evidence, "bitnet_i2s_tiling_thread_matrix"),
            vec![BITNET_CPU_BUNDLE, BITNET_REFERENCE, BITNET_PERF_MICRO, BITNET_PERF_TILING],
            vec!["BitNet remains CPU reference-only in the operator route policy".to_string()],
        ),
        regression_check(
            "openvino_dense_slm_candidates_bounded",
            route_ok(&operator, "dense_slm_openvino_gpu_candidate")
                && route_ok(&operator, "dense_slm_openvino_npu_candidate")
                && evidence_ok(&operator.evidence, "dense_slm_openvino_gpu_arc140v")
                && evidence_ok(&operator.evidence, "dense_slm_openvino_npu")
                && evidence_ok(&operator.evidence, "dense_slm_openvino_phase_runner"),
            vec![DENSE_OV_GPU, DENSE_OV_NPU, DENSE_OV_PHASE],
            vec!["OpenVINO GPU and NPU remain candidate routes without speedup claims".to_string()],
        ),
        regression_check(
            "arc_npu_bitnet_claim_boundaries",
            !operator.claim_boundary.arc_bitnet_full_inference_claimed
                && !operator.claim_boundary.npu_bitnet_full_inference_claimed
                && !operator.claim_boundary.qk256_accelerator_decode_claimed
                && evidence_ok(&operator.evidence, "arc140v_native_opencl_parity")
                && evidence_ok(&operator.evidence, "npu_rmsnorm_static_subgraph")
                && evidence_ok(&operator.evidence, "npu_linear_static_subgraph")
                && evidence_ok(&operator.evidence, "npu_ffn_static_subgraph"),
            vec![ARC_OPENCL_PARITY, NPU_RMSNORM, NPU_LINEAR, NPU_FFN],
            vec!["Arc and NPU evidence remains bounded to parity/subgraph receipts".to_string()],
        ),
        regression_check(
            "no_hidden_fallback_or_acceleration_claim",
            !operator.claim_boundary.hidden_fallback_allowed
                && operator.evidence.iter().all(|item| item.fallback_used == Some(false))
                && operator.routes.iter().all(|route| !route.acceleration_claim)
                && operator.evidence.iter().all(|item| item.speedup_claim != Some(true)),
            vec![OPERATOR_READINESS],
            vec![
                "all indexed evidence reports fallback_used=false".to_string(),
                "all operator routes keep acceleration_claim=false".to_string(),
            ],
        ),
    ];
    let gaps = checks
        .iter()
        .filter(|check| check.status != "passed")
        .map(|check| format!("{}: {}", check.check_id, check.notes.join(", ")))
        .collect::<Vec<_>>();

    Ok(LunarLakeRegressionBundle {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_regression_bundle".to_string(),
        proof_stage: "operator_regression_indexed".to_string(),
        created_utc,
        machine_id: "intel-258v".to_string(),
        artifact_root: path_string(root),
        operator_receipt: path_string(&operator_receipt_path),
        regression_passed: gaps.is_empty(),
        checks,
        gaps,
        claim_boundary: operator.claim_boundary,
    })
}

pub fn build_comparison_receipt_with_created_utc(
    root: &Path,
    operator_receipt: &Path,
    regression_bundle: &Path,
    created_utc: String,
) -> Result<LunarLakeComparisonReceipt> {
    let operator_receipt_path = resolve_receipt_path(root, operator_receipt);
    let regression_bundle_path = resolve_receipt_path(root, regression_bundle);
    let operator: LunarLakeOperatorReceipt = read_json_receipt(&operator_receipt_path)?;
    let regression: LunarLakeRegressionBundle = read_json_receipt(&regression_bundle_path)?;

    let mut gaps = Vec::new();
    if !operator.operator_ready {
        gaps.push(format!("operator receipt not ready: {}", operator.gaps.join("; ")));
    }
    if !regression.regression_passed {
        gaps.push(format!("regression bundle failed: {}", regression.gaps.join("; ")));
    }
    if operator.machine_id != regression.machine_id {
        gaps.push(format!(
            "machine_id mismatch: operator={} regression={}",
            operator.machine_id, regression.machine_id
        ));
    }
    if operator.claim_boundary != regression.claim_boundary {
        gaps.push("claim boundary mismatch between operator and regression receipts".to_string());
    }

    let routes = operator
        .routes
        .iter()
        .map(|route| compare_route(route, &operator.evidence))
        .collect::<Vec<_>>();
    for route in &routes {
        if !route.evidence_ready {
            gaps.push(format!("route {} has incomplete evidence", route.route_id));
        }
        if route.acceleration_claim {
            gaps.push(format!("route {} claims acceleration", route.route_id));
        }
    }

    let comparison_ready = gaps.is_empty();
    Ok(LunarLakeComparisonReceipt {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_operator_comparison".to_string(),
        proof_stage: "operator_routes_compared".to_string(),
        created_utc,
        machine_id: operator.machine_id.clone(),
        artifact_root: path_string(root),
        operator_receipt: path_string(&operator_receipt_path),
        regression_bundle: path_string(&regression_bundle_path),
        comparison_ready,
        operator_ready: operator.operator_ready,
        regression_passed: regression.regression_passed,
        default_route_id: operator.default_route.route_id.clone(),
        routes,
        evidence: operator.evidence,
        checks: regression.checks,
        gaps,
        claim_boundary: operator.claim_boundary,
    })
}

fn normalize_created_utc(created_utc: &str) -> Result<String> {
    let timestamp = chrono::DateTime::parse_from_rfc3339(created_utc)
        .with_context(|| format!("invalid --created-utc timestamp `{created_utc}`"))?;
    Ok(timestamp.with_timezone(&chrono::Utc).to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
}

#[derive(Debug, Clone, Copy)]
enum EvidenceExpectation {
    Present,
    Answer,
    Phase,
    AnswerAndPhase,
    NoSpeedupClaim,
}

fn inspect_receipt(
    root: &Path,
    evidence_id: &str,
    file_name: &str,
    expectation: EvidenceExpectation,
) -> Result<EvidenceStatus> {
    let path = root.join(file_name);
    if !path.exists() {
        return Ok(EvidenceStatus {
            evidence_id: evidence_id.to_string(),
            path: path_string(&path),
            present: false,
            artifact_kind: None,
            requested_backend: None,
            selected_backend: None,
            runtime_api: None,
            fallback_used: None,
            answer_gate_passed: None,
            phase_timing_present: None,
            speedup_claim: None,
            issues: vec!["missing required receipt".to_string()],
        });
    }

    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let json: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;

    let fallback_used = fallback_used(&json);
    let answer_gate_passed = answer_gate_passed(&json);
    let phase_timing_present = phase_timing_present(&json);
    let speedup_claim = bool_at_any(&json, &["speedup_claim", "claim_boundary.speedup_claim"]);

    let mut issues = Vec::new();
    match fallback_used {
        Some(false) => {}
        Some(true) => issues.push("fallback_used=true".to_string()),
        None => issues.push("fallback status missing".to_string()),
    }
    match expectation {
        EvidenceExpectation::Present => {}
        EvidenceExpectation::Answer => {
            if answer_gate_passed != Some(true) {
                issues.push("answer gate did not pass or is missing".to_string());
            }
        }
        EvidenceExpectation::Phase => {
            if phase_timing_present != Some(true) {
                issues.push("phase timing evidence missing".to_string());
            }
        }
        EvidenceExpectation::AnswerAndPhase => {
            if answer_gate_passed != Some(true) {
                issues.push("answer gate did not pass or is missing".to_string());
            }
            if phase_timing_present != Some(true) {
                issues.push("phase timing evidence missing".to_string());
            }
        }
        EvidenceExpectation::NoSpeedupClaim => {
            if speedup_claim != Some(false) {
                issues.push("speedup_claim=false missing".to_string());
            }
        }
    }

    Ok(EvidenceStatus {
        evidence_id: evidence_id.to_string(),
        path: path_string(&path),
        present: true,
        artifact_kind: string_at(&json, "artifact_kind"),
        requested_backend: string_at_any(
            &json,
            &["requested_backend", "backend.requested_backend"],
        ),
        selected_backend: string_at_any(&json, &["selected_backend", "backend.selected_backend"]),
        runtime_api: string_at_any(&json, &["runtime_api", "backend.runtime_api"]),
        fallback_used,
        answer_gate_passed,
        phase_timing_present,
        speedup_claim,
        issues,
    })
}

fn evidence_ok(evidence: &[EvidenceStatus], id: &str) -> bool {
    evidence
        .iter()
        .find(|item| item.evidence_id == id)
        .is_some_and(|item| item.present && item.issues.is_empty())
}

fn dense_slm_cpu_route() -> OperatorRoute {
    OperatorRoute {
        route_id: "dense_slm_default_cpu".to_string(),
        workload: "ask".to_string(),
        selected_model: "Qwen2.5-0.5B-Instruct Q8_0 GGUF".to_string(),
        selected_backend: "cpu-rust".to_string(),
        runtime_api: "cpu".to_string(),
        selected_kernel_or_runtime: "dense-qwen-cpu-reference".to_string(),
        fallback_policy: "strict_no_fallback".to_string(),
        route_reason: "Default user-facing route because the dense Qwen CPU path has strict answer gates, generated-token evidence, phase receipts, and fallback_used=false; accelerator paths are candidates until speedup is benchmark-qualified.".to_string(),
        answer_gate_evidence: Some(DENSE_CPU_ANSWER.to_string()),
        phase_evidence: Some(DENSE_CPU_PHASE.to_string()),
        acceleration_claim: false,
    }
}

fn bitnet_cpu_route() -> OperatorRoute {
    OperatorRoute {
        route_id: "bitnet_reference_cpu".to_string(),
        workload: "bitnet_strict".to_string(),
        selected_model: "microsoft/bitnet-b1.58-2B-4T GGUF I2_S".to_string(),
        selected_backend: "intel-258v-cpu-avx2".to_string(),
        runtime_api: "cpu".to_string(),
        selected_kernel_or_runtime: "qk256/i2_s-cpu".to_string(),
        fallback_policy: "strict_no_fallback".to_string(),
        route_reason: "BitNet remains on CPU because the 258V CPU has the corrected reference bundle, bitnet.cpp boundary evidence, scalar/AVX2 parity, and I2_S GEMV/GEMM tuning receipts; Arc/NPU BitNet evidence is still selected kernel or static subgraph only.".to_string(),
        answer_gate_evidence: Some(BITNET_CPU_BUNDLE.to_string()),
        phase_evidence: Some(BITNET_PERF_TILING.to_string()),
        acceleration_claim: false,
    }
}

fn openvino_gpu_candidate_route() -> OperatorRoute {
    OperatorRoute {
        route_id: "dense_slm_openvino_gpu_candidate".to_string(),
        workload: "dense_slm_acceleration_candidate".to_string(),
        selected_model: "Qwen2.5-0.5B-Instruct OpenVINO IR INT4_SYM".to_string(),
        selected_backend: "openvino-gpu".to_string(),
        runtime_api: "openvino_genai".to_string(),
        selected_kernel_or_runtime: "openvino-genai-llmpipeline-gpu".to_string(),
        fallback_policy: "strict_no_fallback".to_string(),
        route_reason: "Candidate route because Arc 140V OpenVINO GenAI bounded answer gates and phase metrics exist with fallback_used=false, but no benchmark-qualified speedup claim is recorded.".to_string(),
        answer_gate_evidence: Some(DENSE_OV_GPU.to_string()),
        phase_evidence: Some(DENSE_OV_PHASE.to_string()),
        acceleration_claim: false,
    }
}

fn openvino_npu_candidate_route() -> OperatorRoute {
    OperatorRoute {
        route_id: "dense_slm_openvino_npu_candidate".to_string(),
        workload: "dense_slm_static_graph_candidate".to_string(),
        selected_model: "Qwen2.5-0.5B-Instruct OpenVINO IR INT4_SYM".to_string(),
        selected_backend: "openvino-npu".to_string(),
        runtime_api: "openvino_genai".to_string(),
        selected_kernel_or_runtime: "openvino-genai-llmpipeline-npu".to_string(),
        fallback_policy: "strict_no_fallback".to_string(),
        route_reason: "Candidate route because Intel NPU OpenVINO GenAI bounded answer gates and phase metrics exist with fallback_used=false under INT4 symmetric constraints; no dynamic decode, beam, parallel sampling, packed QK256, or acceleration claim is made.".to_string(),
        answer_gate_evidence: Some(DENSE_OV_NPU.to_string()),
        phase_evidence: Some(DENSE_OV_PHASE.to_string()),
        acceleration_claim: false,
    }
}

fn write_or_print_receipt(receipt: &LunarLakeOperatorReceipt, path: Option<&Path>) -> Result<()> {
    let json = serde_json::to_vec_pretty(receipt)?;
    if let Some(path) = path {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, json)?;
        println!("Lunar Lake operator readiness receipt written to {}", path.display());
    } else {
        println!("{}", String::from_utf8_lossy(&json));
    }
    Ok(())
}

fn write_or_print_regression_bundle(
    receipt: &LunarLakeRegressionBundle,
    path: Option<&Path>,
) -> Result<()> {
    let json = serde_json::to_vec_pretty(receipt)?;
    if let Some(path) = path {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, json)?;
        println!("Lunar Lake regression bundle written to {}", path.display());
    } else {
        println!("{}", String::from_utf8_lossy(&json));
    }
    Ok(())
}

fn write_or_print_comparison_receipt(
    receipt: &LunarLakeComparisonReceipt,
    path: Option<&Path>,
) -> Result<()> {
    let json = serde_json::to_vec_pretty(receipt)?;
    if let Some(path) = path {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, json)?;
        println!("Lunar Lake comparison receipt written to {}", path.display());
    } else {
        println!("{}", String::from_utf8_lossy(&json));
    }
    Ok(())
}

fn read_json_receipt<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("failed to parse {}", path.display()))
}

fn resolve_receipt_path(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() || path.exists() { path.to_path_buf() } else { root.join(path) }
}

fn compare_route(route: &OperatorRoute, evidence: &[EvidenceStatus]) -> RouteComparison {
    let attached = [&route.answer_gate_evidence, &route.phase_evidence]
        .into_iter()
        .flatten()
        .filter_map(|file_name| evidence_for_file(evidence, file_name))
        .collect::<Vec<_>>();
    let missing = [&route.answer_gate_evidence, &route.phase_evidence]
        .into_iter()
        .flatten()
        .filter(|file_name| evidence_for_file(evidence, file_name).is_none())
        .cloned()
        .collect::<Vec<_>>();

    let evidence_ready =
        missing.is_empty() && attached.iter().all(|item| item.present && item.issues.is_empty());
    let mut notes = vec![format!("role={}", route_role(route))];
    if !missing.is_empty() {
        notes.push(format!("missing attached evidence: {}", missing.join(", ")));
    }
    for item in &attached {
        notes.push(format!(
            "{} present={} fallback_used={:?} answer_gate={:?} phase_timing={:?}",
            item.evidence_id,
            item.present,
            item.fallback_used,
            item.answer_gate_passed,
            item.phase_timing_present
        ));
        if !item.issues.is_empty() {
            notes.push(format!("{} issues: {}", item.evidence_id, item.issues.join(", ")));
        }
    }

    RouteComparison {
        route_id: route.route_id.clone(),
        role: route_role(route).to_string(),
        workload: route.workload.clone(),
        selected_model: route.selected_model.clone(),
        selected_backend: route.selected_backend.clone(),
        runtime_api: route.runtime_api.clone(),
        selected_kernel_or_runtime: route.selected_kernel_or_runtime.clone(),
        fallback_policy: route.fallback_policy.clone(),
        answer_gate_evidence: route.answer_gate_evidence.clone(),
        phase_evidence: route.phase_evidence.clone(),
        evidence_ready,
        acceleration_claim: route.acceleration_claim,
        route_reason: route.route_reason.clone(),
        notes,
    }
}

fn evidence_for_file<'a>(
    evidence: &'a [EvidenceStatus],
    file_name: &str,
) -> Option<&'a EvidenceStatus> {
    evidence.iter().find(|item| {
        item.path == file_name
            || item.path.replace('\\', "/").ends_with(&format!("/{file_name}"))
            || item.path.replace('\\', "/").ends_with(file_name)
    })
}

fn route_role(route: &OperatorRoute) -> &'static str {
    match route.route_id.as_str() {
        "dense_slm_default_cpu" => "default_cpu_answer_path",
        "bitnet_reference_cpu" => "bitnet_cpu_reference_path",
        "dense_slm_openvino_gpu_candidate" => "dense_slm_gpu_candidate",
        "dense_slm_openvino_npu_candidate" => "dense_slm_npu_candidate",
        _ => "additional_route",
    }
}

fn regression_check(
    check_id: &str,
    passed: bool,
    evidence: Vec<&str>,
    notes: Vec<String>,
) -> RegressionCheck {
    RegressionCheck {
        check_id: check_id.to_string(),
        status: if passed { "passed" } else { "failed" }.to_string(),
        evidence: evidence.into_iter().map(ToString::to_string).collect(),
        notes,
    }
}

fn route_ok(operator: &LunarLakeOperatorReceipt, route_id: &str) -> bool {
    operator.routes.iter().any(|route| route.route_id == route_id && !route.acceleration_claim)
}

fn fallback_used(json: &Value) -> Option<bool> {
    if let Some(value) = bool_at_any(json, &["fallback_used", "backend.fallback_used"]) {
        return Some(value);
    }

    let device_fallbacks =
        json.pointer("/generation/devices").and_then(Value::as_array).map(|devices| {
            devices
                .iter()
                .filter_map(|device| device.get("fallback_used").and_then(Value::as_bool))
                .collect::<Vec<_>>()
        });
    if let Some(values) = device_fallbacks
        && !values.is_empty()
    {
        return Some(values.iter().any(|value| *value));
    }

    let profile_fallbacks = json.get("profiles").and_then(Value::as_array).map(|profiles| {
        profiles
            .iter()
            .filter_map(|profile| profile.get("fallback_used").and_then(Value::as_bool))
            .collect::<Vec<_>>()
    });
    if let Some(values) = profile_fallbacks
        && !values.is_empty()
    {
        return Some(values.iter().any(|value| *value));
    }

    None
}

fn answer_gate_passed(json: &Value) -> Option<bool> {
    if let Some(value) = bool_at_any(
        json,
        &[
            "answer_gate_passed",
            "quality.passed",
            "generation.all_answer_gates_passed",
            "summary.all_passed",
        ],
    ) {
        return Some(value);
    }

    if let Some(failed) = json.pointer("/summary/failed").and_then(Value::as_u64) {
        return Some(failed == 0);
    }

    if let Some(failed) = json.pointer("/generation/failed").and_then(Value::as_u64) {
        let passed = json.pointer("/generation/passed").and_then(Value::as_u64).unwrap_or(0);
        return Some(failed == 0 && passed > 0);
    }

    if let Some(cases) = json.get("cases").and_then(Value::as_array)
        && !cases.is_empty()
    {
        return Some(cases.iter().all(case_passed));
    }

    if let Some(devices) = json.pointer("/generation/devices").and_then(Value::as_array)
        && !devices.is_empty()
    {
        return Some(devices.iter().all(|device| {
            device.get("failed").and_then(Value::as_u64).unwrap_or(1) == 0
                && device.get("passed").and_then(Value::as_u64).unwrap_or(0) > 0
        }));
    }

    None
}

fn case_passed(case: &Value) -> bool {
    case.get("status").and_then(Value::as_str) == Some("passed")
        || case.pointer("/quality/passed").and_then(Value::as_bool) == Some(true)
}

fn phase_timing_present(json: &Value) -> Option<bool> {
    if let Some(profiles) = json.get("profiles").and_then(Value::as_array) {
        return Some(!profiles.is_empty() && profiles.iter().any(profile_has_timing));
    }

    if let Some(devices) = json.pointer("/generation/devices").and_then(Value::as_array) {
        return Some(!devices.is_empty() && devices.iter().any(device_has_timing));
    }

    None
}

fn profile_has_timing(profile: &Value) -> bool {
    ["prefill_ms", "first_token_decode_ms", "decode_total_ms", "total_ms"]
        .iter()
        .any(|key| profile.get(*key).and_then(Value::as_f64).is_some())
}

fn device_has_timing(device: &Value) -> bool {
    device.get("pipeline_construct_wall_ms").and_then(Value::as_f64).is_some()
        || device.pointer("/perf_metrics").is_some()
        || device.pointer("/streaming/first_text_chunk_ms").is_some()
}

fn string_at_any(json: &Value, paths: &[&str]) -> Option<String> {
    paths.iter().find_map(|path| string_at(json, path))
}

fn string_at(json: &Value, path: &str) -> Option<String> {
    value_at(json, path).and_then(Value::as_str).map(ToString::to_string)
}

fn bool_at_any(json: &Value, paths: &[&str]) -> Option<bool> {
    paths.iter().find_map(|path| value_at(json, path).and_then(Value::as_bool))
}

fn value_at<'a>(json: &'a Value, dotted_path: &str) -> Option<&'a Value> {
    let mut current = json;
    for segment in dotted_path.split('.') {
        current = current.get(segment)?;
    }
    Some(current)
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn operator_readiness_passes_with_required_receipts() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;

        let receipt = build_operator_readiness_receipt(temp.path())?;

        assert!(receipt.operator_ready, "{:?}", receipt.gaps);
        assert_eq!(receipt.default_route.route_id, "dense_slm_default_cpu");
        assert_eq!(receipt.default_route.selected_backend, "cpu-rust");
        assert!(
            receipt.routes.iter().any(|route| route.route_id == "dense_slm_openvino_gpu_candidate")
        );
        assert!(receipt.routes.iter().all(|route| !route.acceleration_claim));
        assert!(receipt.claim_boundary.cpu_is_truth_path);
        assert!(!receipt.claim_boundary.hidden_fallback_allowed);
        Ok(())
    }

    #[test]
    fn operator_readiness_reports_missing_receipts() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_json(
            temp.path(),
            DENSE_CPU_ANSWER,
            json!({
                "artifact_kind": "slm_cpu_answer_corpus",
                "fallback_used": false,
                "cases": [{"status": "passed"}]
            }),
        )?;

        let receipt = build_operator_readiness_receipt(temp.path())?;

        assert!(!receipt.operator_ready);
        assert!(receipt.gaps.iter().any(|gap| gap.contains("missing required receipt")));
        Ok(())
    }

    #[test]
    fn operator_readiness_rejects_fallback() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), true)?;

        let receipt = build_operator_readiness_receipt(temp.path())?;

        assert!(!receipt.operator_ready);
        assert!(receipt.gaps.iter().any(|gap| gap.contains("fallback_used=true")));
        Ok(())
    }

    #[test]
    fn operator_readiness_accepts_reproducible_timestamp() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;

        let receipt = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            normalize_created_utc("2026-05-13T11:36:09-04:00")?,
        )?;

        assert_eq!(receipt.created_utc, "2026-05-13T15:36:09Z");
        Ok(())
    }

    #[test]
    fn regression_bundle_passes_with_operator_ready_receipt() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-13T15:36:09Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;

        let bundle = build_regression_bundle_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "2026-05-13T16:59:00Z".to_string(),
        )?;

        assert!(bundle.regression_passed, "{:?}", bundle.gaps);
        assert_eq!(bundle.artifact_kind, "lunar_lake_regression_bundle");
        assert!(bundle.checks.iter().any(|check| check.check_id == "dense_slm_default_cpu_route"));
        assert!(bundle.checks.iter().all(|check| check.status == "passed"));
        assert!(!bundle.claim_boundary.hidden_fallback_allowed);
        Ok(())
    }

    #[test]
    fn regression_bundle_rejects_operator_fallback() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), true)?;
        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-13T15:36:09Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;

        let bundle = build_regression_bundle_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "2026-05-13T16:59:00Z".to_string(),
        )?;

        assert!(!bundle.regression_passed);
        assert!(
            bundle.gaps.iter().any(|gap| gap.contains("no_hidden_fallback_or_acceleration_claim"))
        );
        Ok(())
    }

    #[test]
    fn comparison_receipt_indexes_operator_routes_and_regression_checks() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-13T15:36:09Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;
        let regression = build_regression_bundle_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "2026-05-13T17:05:00Z".to_string(),
        )?;
        fs::write(temp.path().join(REGRESSION_BUNDLE), serde_json::to_vec_pretty(&regression)?)?;

        let comparison = build_comparison_receipt_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(REGRESSION_BUNDLE),
            "2026-05-13T18:30:00Z".to_string(),
        )?;

        assert!(comparison.comparison_ready, "{:?}", comparison.gaps);
        assert_eq!(comparison.artifact_kind, "lunar_lake_operator_comparison");
        assert_eq!(comparison.default_route_id, "dense_slm_default_cpu");
        assert!(comparison.routes.iter().any(|route| {
            route.route_id == "dense_slm_default_cpu"
                && route.role == "default_cpu_answer_path"
                && route.evidence_ready
        }));
        assert!(comparison.routes.iter().all(|route| !route.acceleration_claim));
        assert!(comparison.checks.iter().all(|check| check.status == "passed"));
        assert!(!comparison.claim_boundary.hidden_fallback_allowed);
        Ok(())
    }

    #[test]
    fn comparison_receipt_rejects_failed_regression_bundle() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), true)?;
        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-13T15:36:09Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;
        let regression = build_regression_bundle_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "2026-05-13T17:05:00Z".to_string(),
        )?;
        fs::write(temp.path().join(REGRESSION_BUNDLE), serde_json::to_vec_pretty(&regression)?)?;

        let comparison = build_comparison_receipt_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(REGRESSION_BUNDLE),
            "2026-05-13T18:30:00Z".to_string(),
        )?;

        assert!(!comparison.comparison_ready);
        assert!(comparison.gaps.iter().any(|gap| gap.contains("regression bundle failed")));
        Ok(())
    }

    fn write_minimal_receipts(root: &Path, fallback: bool) -> Result<()> {
        let answer = json!({
            "artifact_kind": "answer",
            "fallback_used": fallback,
            "requested_backend": "cpu",
            "selected_backend": "cpu-rust",
            "runtime_api": "cpu",
            "cases": [{"status": "passed"}]
        });
        let phase = json!({
            "artifact_kind": "phase",
            "fallback_used": fallback,
            "requested_backend": "cpu",
            "selected_backend": "cpu-rust",
            "runtime_api": "cpu",
            "profiles": [{"prefill_ms": 1.0}]
        });
        let openvino = json!({
            "artifact_kind": "openvino",
            "fallback_used": fallback,
            "requested_backend": "openvino-cpu-gpu-npu",
            "selected_backend": "openvino-cpu-gpu-npu",
            "runtime_api": "openvino_genai",
            "generation": {
                "all_answer_gates_passed": true,
                "devices": [{"passed": 1, "failed": 0, "pipeline_construct_wall_ms": 1.0, "fallback_used": fallback}]
            }
        });
        let present = json!({
            "artifact_kind": "present",
            "fallback_used": fallback
        });
        let no_speedup = json!({
            "artifact_kind": "perf",
            "fallback_used": fallback,
            "speedup_claim": false
        });

        for file in [DENSE_CPU_ANSWER, DENSE_OV_CPU, DENSE_OV_GPU, DENSE_OV_NPU] {
            write_json(root, file, answer.clone())?;
        }
        write_json(root, DENSE_CPU_PHASE, phase)?;
        write_json(root, DENSE_OV_PHASE, openvino)?;
        for file in [
            BITNET_CPU_BUNDLE,
            BITNET_REFERENCE,
            ARC_OPENCL_PARITY,
            NPU_RMSNORM,
            NPU_LINEAR,
            NPU_FFN,
        ] {
            write_json(root, file, present.clone())?;
        }
        for file in [BITNET_PERF_MICRO, BITNET_PERF_TILING] {
            write_json(root, file, no_speedup.clone())?;
        }
        Ok(())
    }

    fn write_json(root: &Path, file: &str, value: Value) -> Result<()> {
        fs::create_dir_all(root)?;
        fs::write(root.join(file), serde_json::to_vec_pretty(&value)?)?;
        Ok(())
    }
}
