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
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_ARTIFACT_ROOT: &str = "ci/hardware/intel-258v/2026-05-08";

const DENSE_CPU_ANSWER: &str = "slm-answer-corpus-qwen25-cpu-clean-provenance.json";
const DENSE_CPU_PHASE: &str = "slm-phase-warm-session-qwen25-cpu.json";
const DENSE_OV_PHASE: &str = "slm-openvino-cpu-gpu-npu-phase-runner.json";
const DENSE_OV_CPU: &str = "slm-openvino-cpu-llmpipeline-smoke.json";
const DENSE_OV_GPU: &str = "slm-openvino-gpu-arc140v-llmpipeline-smoke.json";
const DENSE_OV_NPU: &str = "slm-openvino-npu-llmpipeline-smoke.json";
const DENSE_OV_GPU_OPERATOR_ASK: &str = "lunar-lake-openvino-operator-ask-gpu-math-brief.json";
const DENSE_OV_NPU_OPERATOR_ASK: &str = "lunar-lake-openvino-operator-ask-npu-math-brief.json";
const DENSE_CPU_CORPUS_V2: &str = "slm-answer-corpus-qwen25-cpu-corpus-v2.json";
const DENSE_OV_CORPUS_V2: &str = "slm-openvino-cpu-gpu-npu-corpus-v2.json";
const BITNET_CPU_BUNDLE: &str = "cpu-reference-bundle-after-semantic-fix.json";
const BITNET_REFERENCE: &str = "cpu-bitnet-ref-001-external-boundary.json";
const BITNET_REFERENCE_DIRECT: &str = "external-first-token-reference-direct.json";
const BITNET_DIVERGENCE_DIRECT: &str = "first-token-divergence-classification-direct.json";
const BITNET_PERF_MICRO: &str = "cpu-bitnet-perf-001-i2s-microbench.json";
const BITNET_PERF_TILING: &str = "cpu-bitnet-perf-002-i2s-tiling-matrix.json";
const BITNET_PERF_APPLIED: &str = "cpu-bitnet-perf-003-i2s-applied-thread-matrix.json";
const BITNET_EMBEDDING_EVIDENCE: &str = "cpu-bitnet-embd-001-q6k-embedding-evidence.json";
const ARC_OPENCL_PARITY: &str = "arc-140v-opencl-parity.json";
const NPU_RMSNORM: &str = "npu-bitnet-rmsnorm-subgraph-parity.json";
const NPU_LINEAR: &str = "npu-bitnet-linear-projection-subgraph-parity.json";
const NPU_FFN: &str = "npu-bitnet-ffn-subgraph-parity.json";
const OPERATOR_READINESS: &str = "lunar-lake-operator-readiness.json";
const REGRESSION_BUNDLE: &str = "lunar-lake-regression-bundle.json";
const OPERATOR_COMPARISON: &str = "lunar-lake-operator-comparison.json";
const ROUTE_PROMOTION_LEDGER: &str = "lunar-lake-route-promotion.json";
const ROUTE_PROFILE_COMPARISON: &str = "lunar-lake-route-profile-comparison.json";
const DENSE_PHASE_COMPARISON: &str = "slm-openvino-cpu-gpu-npu-phase-comparison.json";
const DENSE_CPU_OPERATOR_ASK: &str = "lunar-lake-operator-ask-math-brief.json";
const ANSWER_CORPUS_V2: &str = "ci/quality/lunar-lake-answer-corpus-v2.yaml";
const REGRESSION_V2_SURFACE_ID: &str = "lunar_lake_regression_v2";
pub const DEFAULT_ASK_ROUTE: &str = "dense_slm_default_cpu";

const REQUIRED_CORPUS_V2_PROFILES: &[&str] =
    &["regression_tiny", "ask_short", "ask_normal", "structured", "prefill_heavy", "decode_heavy"];
const REQUIRED_CORPUS_V2_CATEGORIES: &[&str] = &[
    "math",
    "copy_exact",
    "yes_no",
    "short_factual",
    "instruction_following",
    "stop_and_eos",
    "prompt_history_sensitivity",
    "structured_output",
    "long_prompt_summarization",
    "short_reasoning",
    "decode_heavy",
];
const REQUIRED_ROUTE_PROFILES: &[&str] = &[
    "regression_tiny",
    "ask_short",
    "ask_normal",
    "prefill_heavy",
    "decode_heavy",
    "structured",
    "low_power",
    "bitnet_strict_reference",
];

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

        /// Optional expanded Lunar Lake answer corpus v2 fixture to index.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = ANSWER_CORPUS_V2)]
        answer_corpus_v2: Option<PathBuf>,

        /// Optional route profile comparison receipt to index.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = ROUTE_PROFILE_COMPARISON)]
        route_profile_comparison: Option<PathBuf>,

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

    /// Build a profile-aware route promotion ledger from the operator evidence.
    Promote {
        /// Artifact root containing the 258V receipts to index.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// Operator readiness receipt to evaluate. Relative paths are resolved under artifact-root.
        #[arg(long, default_value = OPERATOR_READINESS)]
        operator_receipt: PathBuf,

        /// Operator comparison receipt to evaluate. Relative paths are resolved under artifact-root.
        #[arg(long, default_value = OPERATOR_COMPARISON)]
        comparison_receipt: PathBuf,

        /// Output JSON promotion ledger to file.
        #[arg(long)]
        json_out: Option<PathBuf>,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when the promotion ledger cannot safely preserve CPU as the default route.
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Compare promoted and candidate routes against fixed workload profiles.
    ProfileCompare {
        /// Artifact root containing the 258V receipts to index.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// Route promotion ledger to evaluate. Relative paths are resolved under artifact-root.
        #[arg(long, default_value = ROUTE_PROMOTION_LEDGER)]
        promotion_ledger: PathBuf,

        /// Dense SLM phase comparison receipt to index. Relative paths are resolved under artifact-root.
        #[arg(long, default_value = DENSE_PHASE_COMPARISON)]
        phase_comparison: PathBuf,

        /// Dense Qwen CPU corpus-v2 execution receipt to classify promoted CPU profile quality.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = DENSE_CPU_CORPUS_V2)]
        cpu_corpus_v2: Option<PathBuf>,

        /// OpenVINO CPU/GPU/NPU corpus-v2 execution receipt to classify candidate profile quality.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = DENSE_OV_CORPUS_V2)]
        openvino_corpus_v2: Option<PathBuf>,

        /// Output JSON profile comparison to file.
        #[arg(long)]
        json_out: Option<PathBuf>,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when the profile comparison cannot safely preserve CPU as default.
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Diagnose bounded dense Qwen CPU corpus-v2 profile blockers without running inference.
    QualityDiagnose {
        /// Artifact root containing the 258V receipts to inspect.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// Dense Qwen CPU corpus-v2 execution receipt to diagnose.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = DENSE_CPU_CORPUS_V2)]
        cpu_corpus_v2: PathBuf,

        /// Optional route-profile comparison receipt to attach route blockers.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = ROUTE_PROFILE_COMPARISON)]
        route_profile_comparison: Option<PathBuf>,

        /// Output JSON diagnosis receipt to file.
        #[arg(
            long,
            default_value = "ci/hardware/intel-258v/2026-05-08/slm-qwen25-cpu-corpus-v2-diagnosis.json"
        )]
        json_out: PathBuf,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when the diagnosis cannot safely classify the committed corpus receipt.
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Ask through the evidence-backed Lunar Lake default route.
    Ask {
        /// Artifact root containing the 258V receipts to index.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// Operator readiness receipt to enforce before generation.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = OPERATOR_READINESS)]
        operator_receipt: PathBuf,

        /// Route promotion ledger to use when --route auto or --device auto is requested.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = ROUTE_PROMOTION_LEDGER)]
        promotion_ledger: PathBuf,

        /// Workload profile to resolve when auto-routing is requested.
        #[arg(long, default_value = "ask_normal")]
        profile: String,

        /// Operator route to execute, or auto to select from the promotion ledger.
        /// Only ledger-promoted dense_slm_default_cpu can execute today.
        #[arg(long, default_value = DEFAULT_ASK_ROUTE)]
        route: String,

        /// Dense Qwen GGUF model path.
        #[arg(long)]
        model: PathBuf,

        /// Optional explicit tokenizer path.
        #[arg(long)]
        tokenizer: Option<PathBuf>,

        /// User question to answer.
        #[arg(long, value_name = "TEXT", conflicts_with = "question_arg")]
        question: Option<String>,

        /// User question to answer (positional form).
        #[arg(value_name = "QUESTION")]
        question_arg: Option<String>,

        /// Maximum new tokens to generate. The Lunar Lake default ask path is bounded.
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,

        /// Optional bounded-answer gate: normalized output must contain this text.
        #[arg(long, value_name = "TEXT")]
        expect_contains: Option<String>,

        /// Output JSON operator ask receipt to file.
        #[arg(long)]
        json_out: Option<PathBuf>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer_corpus_v2: Option<AnswerCorpusV2Summary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_profile_comparison: Option<RouteProfileRegressionSummary>,
    #[serde(default)]
    pub regression_surface: RegressionSurfaceSummary,
    pub regression_passed: bool,
    pub checks: Vec<RegressionCheck>,
    pub gaps: Vec<String>,
    pub claim_boundary: ClaimBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegressionSurfaceSummary {
    pub surface_id: String,
    pub strict_default: bool,
    pub answer_corpus_v2_indexed: bool,
    pub route_profile_comparison_indexed: bool,
    pub required_answer_profiles: Vec<String>,
    pub required_answer_categories: Vec<String>,
    pub required_route_profiles: Vec<String>,
    pub fallback_observed: bool,
    pub candidate_routes_remain_unpromoted: bool,
    pub benchmark_qualified_advantage_claimed: bool,
    pub strict_ready: bool,
    pub gaps: Vec<String>,
}

impl Default for RegressionSurfaceSummary {
    fn default() -> Self {
        Self {
            surface_id: REGRESSION_V2_SURFACE_ID.to_string(),
            strict_default: true,
            answer_corpus_v2_indexed: false,
            route_profile_comparison_indexed: false,
            required_answer_profiles: REQUIRED_CORPUS_V2_PROFILES
                .iter()
                .map(|profile| (*profile).to_string())
                .collect(),
            required_answer_categories: REQUIRED_CORPUS_V2_CATEGORIES
                .iter()
                .map(|category| (*category).to_string())
                .collect(),
            required_route_profiles: REQUIRED_ROUTE_PROFILES
                .iter()
                .map(|profile| (*profile).to_string())
                .collect(),
            fallback_observed: false,
            candidate_routes_remain_unpromoted: false,
            benchmark_qualified_advantage_claimed: false,
            strict_ready: false,
            gaps: vec![
                "answer corpus v2 is not indexed".to_string(),
                "route profile comparison is not indexed".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegressionCheck {
    pub check_id: String,
    pub status: String,
    pub evidence: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnswerCorpusV2Summary {
    pub path: String,
    pub schema: u64,
    pub name: String,
    pub route_scope: Option<String>,
    pub model_family: Option<String>,
    pub model_architecture: Option<String>,
    pub quantization: Option<String>,
    pub prompt_template: Option<String>,
    pub case_count: usize,
    pub profiles: Vec<String>,
    pub categories: Vec<String>,
    pub claim_boundary_preserved: bool,
    pub fixture_ready: bool,
    pub gaps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouteProfileRegressionSummary {
    pub path: String,
    pub profile_comparison_ready: bool,
    pub default_route_id: String,
    pub profiles: Vec<String>,
    pub candidate_routes_remain_unpromoted: bool,
    pub benchmark_qualified_advantage_claimed: bool,
    pub fallback_observed: bool,
    pub gpu_npu_promotion_blockers: Vec<String>,
    pub regression_ready: bool,
    pub gaps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LunarLakeRoutePromotionLedger {
    pub schema_version: String,
    pub artifact_kind: String,
    pub proof_stage: String,
    pub created_utc: String,
    pub machine_id: String,
    pub artifact_root: String,
    pub operator_receipt: String,
    pub comparison_receipt: String,
    pub promotion_ready: bool,
    pub default_route_id: String,
    pub auto_route_policy: AutoRoutePolicy,
    pub workload_profiles: Vec<WorkloadProfile>,
    pub routes: Vec<RoutePromotion>,
    pub gaps: Vec<String>,
    pub claim_boundary: ClaimBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AutoRoutePolicy {
    pub policy_stage: String,
    pub default_route: String,
    pub hidden_fallback_allowed: bool,
    pub cpu_default_until_profile_promoted: bool,
    pub candidate_routes_require_profile_promotion: bool,
    pub route_reason_required: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkloadProfile {
    pub profile_id: String,
    pub prompt_tokens: String,
    pub output_tokens: String,
    pub purpose: String,
    pub promoted_route: Option<String>,
    pub candidate_routes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoutePromotion {
    pub route_id: String,
    pub status: String,
    pub promoted_for: Vec<String>,
    pub blocked_for: Vec<String>,
    pub required_evidence: Vec<String>,
    pub present_evidence: Vec<String>,
    pub missing_evidence: Vec<String>,
    pub selected_backend: String,
    pub runtime_api: String,
    pub fallback_policy: String,
    pub answer_gate_evidence: Option<String>,
    pub phase_evidence: Option<String>,
    pub fallback_used: Option<bool>,
    pub answer_gate_passed: Option<bool>,
    pub phase_timing_present: Option<bool>,
    pub speedup_claim: bool,
    pub acceleration_claim: bool,
    pub last_evidence_utc: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LunarLakeRouteProfileComparison {
    pub schema_version: String,
    pub artifact_kind: String,
    pub proof_stage: String,
    pub created_utc: String,
    pub machine_id: String,
    pub artifact_root: String,
    pub promotion_ledger: String,
    pub phase_comparison_receipt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_corpus_v2_receipt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openvino_corpus_v2_receipt: Option<String>,
    pub profile_comparison_ready: bool,
    pub default_route_id: String,
    pub profiles: Vec<WorkloadProfileEvaluation>,
    pub gaps: Vec<String>,
    pub claim_boundary: ClaimBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkloadProfileEvaluation {
    pub profile_id: String,
    pub prompt_tokens: String,
    pub output_tokens: String,
    pub purpose: String,
    pub promoted_route: Option<String>,
    pub candidate_routes: Vec<String>,
    pub profile_status: String,
    pub route_evidence: Vec<ProfileRouteEvidence>,
    pub promotion_decision: String,
    pub gaps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProfileRouteEvidence {
    pub route_id: String,
    pub route_status: String,
    pub selected_backend: String,
    pub runtime_api: String,
    pub fallback_used: Option<bool>,
    pub answer_gate_passed: Option<bool>,
    pub phase_timing_present: Option<bool>,
    pub timing: ProfileTimingSummary,
    pub benchmark_qualified_advantage: bool,
    pub promotion_eligible_for_profile: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_quality: Option<ProfileQualityEvidence>,
    pub evidence: Vec<String>,
    pub blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProfileQualityEvidence {
    pub source_receipt: String,
    pub route_id: String,
    pub profile_id: String,
    pub profile_present: bool,
    pub cases_total: u64,
    pub passed: u64,
    pub failed: u64,
    pub fallback_used: Option<bool>,
    pub status: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QwenCpuCorpusV2Diagnosis {
    pub schema_version: String,
    pub artifact_kind: String,
    pub proof_stage: String,
    pub created_utc: String,
    pub machine_id: String,
    pub artifact_root: String,
    pub cpu_corpus_v2_receipt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_profile_comparison_receipt: Option<String>,
    pub route_id: String,
    pub model_family: Option<String>,
    pub model_architecture: Option<String>,
    pub quantization: Option<String>,
    pub requested_backend: Option<String>,
    pub selected_backend: Option<String>,
    pub runtime_api: Option<String>,
    pub fallback_used: Option<bool>,
    pub quality_summary: CorpusV2QualitySummary,
    pub profile_diagnoses: Vec<CorpusV2ProfileDiagnosis>,
    pub failed_cases: Vec<CorpusV2FailedCaseDiagnosis>,
    pub route_blocked: bool,
    pub blocker_summary: Vec<String>,
    pub recommended_next_actions: Vec<String>,
    pub diagnosis_ready: bool,
    pub gaps: Vec<String>,
    pub claim_boundary: CorpusV2DiagnosisClaimBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CorpusV2QualitySummary {
    pub total: u64,
    pub passed: u64,
    pub failed: u64,
    pub timeout: u64,
    pub not_run: u64,
    pub failed_profiles: Vec<String>,
    pub failed_categories: Vec<String>,
    pub failure_classes: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CorpusV2ProfileDiagnosis {
    pub profile_id: String,
    pub total: u64,
    pub passed: u64,
    pub failed: u64,
    pub blocked: bool,
    pub failed_case_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_profile_status: Option<String>,
    pub route_blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CorpusV2FailedCaseDiagnosis {
    pub id: String,
    pub profile: String,
    pub category: String,
    pub task_family: Option<String>,
    pub status: String,
    pub gate_kind: Option<String>,
    pub scoring_kind: Option<String>,
    pub failed_rules: Vec<String>,
    pub failure_taxonomy: Vec<String>,
    pub missing_required_keywords: Vec<String>,
    pub forbidden_tokens_observed: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_normalized: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_normalized: Option<String>,
    pub answer_preview: String,
    pub generated_tokens: Option<u64>,
    pub prompt_tokens: Option<u64>,
    pub run_receipt_path: Option<String>,
    pub fallback_used: Option<bool>,
    pub classification: String,
    pub recommended_fix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CorpusV2DiagnosisClaimBoundary {
    pub diagnostic_only: bool,
    pub new_inference_executed: bool,
    pub broad_quality_claim: bool,
    pub speedup_claim: bool,
    pub route_promotion_changed: bool,
    pub arc_or_npu_execution_claim: bool,
    pub bitnet_qk256_i2s_behavior_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProfileTimingSummary {
    pub timing_scope: String,
    pub source_receipts: Vec<String>,
    pub cold_load_ms: Option<f64>,
    pub tokenize_ms: Option<f64>,
    pub prefill_ms: Option<f64>,
    pub first_token_ms: Option<f64>,
    pub decode_total_ms: Option<f64>,
    pub generation_total_ms: Option<f64>,
    pub total_response_ms: Option<f64>,
    pub output_tokens: Option<u64>,
    pub throughput_tokens_per_s: Option<f64>,
    pub phase_coverage: Vec<String>,
    pub known_gaps: Vec<String>,
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
                answer_corpus_v2,
                route_profile_comparison,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_regression_bundle_with_created_utc_and_inputs(
                    artifact_root,
                    operator_receipt,
                    answer_corpus_v2.as_deref(),
                    route_profile_comparison.as_deref(),
                    created_utc,
                )?;
                write_or_print_regression_bundle(&receipt, json_out.as_deref())?;
                if *strict {
                    let strict_gaps = strict_regression_v2_gaps(&receipt);
                    if !strict_gaps.is_empty() {
                        bail!("Lunar Lake regression bundle failed: {}", strict_gaps.join("; "));
                    }
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
            LunarLakeAction::Promote {
                artifact_root,
                operator_receipt,
                comparison_receipt,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_route_promotion_ledger_with_created_utc(
                    artifact_root,
                    operator_receipt,
                    comparison_receipt,
                    created_utc,
                )?;
                write_or_print_route_promotion_ledger(&receipt, json_out.as_deref())?;
                if *strict && !receipt.promotion_ready {
                    bail!("Lunar Lake route promotion ledger failed: {}", receipt.gaps.join("; "));
                }
                Ok(())
            }
            LunarLakeAction::ProfileCompare {
                artifact_root,
                promotion_ledger,
                phase_comparison,
                cpu_corpus_v2,
                openvino_corpus_v2,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_route_profile_comparison_with_created_utc_and_inputs(
                    artifact_root,
                    promotion_ledger,
                    phase_comparison,
                    cpu_corpus_v2.as_deref(),
                    openvino_corpus_v2.as_deref(),
                    created_utc,
                )?;
                write_or_print_route_profile_comparison(&receipt, json_out.as_deref())?;
                if *strict && !receipt.profile_comparison_ready {
                    bail!(
                        "Lunar Lake route profile comparison failed: {}",
                        receipt.gaps.join("; ")
                    );
                }
                Ok(())
            }
            LunarLakeAction::QualityDiagnose {
                artifact_root,
                cpu_corpus_v2,
                route_profile_comparison,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_qwen_cpu_corpus_v2_diagnosis_with_created_utc(
                    artifact_root,
                    cpu_corpus_v2,
                    route_profile_comparison.as_deref(),
                    created_utc,
                )?;
                write_or_print_qwen_cpu_corpus_v2_diagnosis(&receipt, Some(json_out))?;
                if *strict && !receipt.diagnosis_ready {
                    bail!(
                        "Lunar Lake dense Qwen CPU corpus-v2 diagnosis failed: {}",
                        receipt.gaps.join("; ")
                    );
                }
                Ok(())
            }
            LunarLakeAction::Ask { .. } => {
                bail!("lunar-lake ask must be handled by the CLI runtime")
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
        inspect_receipt(
            root,
            "dense_slm_openvino_gpu_operator_ask",
            DENSE_OV_GPU_OPERATOR_ASK,
            EvidenceExpectation::Answer,
        )?,
        inspect_receipt(root, "dense_slm_openvino_npu", DENSE_OV_NPU, EvidenceExpectation::Answer)?,
        inspect_receipt(
            root,
            "dense_slm_openvino_npu_operator_ask",
            DENSE_OV_NPU_OPERATOR_ASK,
            EvidenceExpectation::Answer,
        )?,
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
            "bitnet_external_direct_token_boundary",
            BITNET_REFERENCE_DIRECT,
            EvidenceExpectation::Present,
        )?,
        inspect_receipt(
            root,
            "bitnet_first_token_direct_classifier",
            BITNET_DIVERGENCE_DIRECT,
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
            "bitnet_i2s_applied_thread_matrix",
            BITNET_PERF_APPLIED,
            EvidenceExpectation::NoSpeedupClaim,
        )?,
        inspect_receipt(
            root,
            "bitnet_embedding_quantization_evidence",
            BITNET_EMBEDDING_EVIDENCE,
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
        && evidence_ok(&evidence, "dense_slm_openvino_gpu_operator_ask")
        && evidence_ok(&evidence, "dense_slm_openvino_npu")
        && evidence_ok(&evidence, "dense_slm_openvino_npu_operator_ask")
        && evidence_ok(&evidence, "dense_slm_openvino_phase_runner");
    let bitnet_cpu_ready = evidence_ok(&evidence, "bitnet_cpu_reference_bundle")
        && evidence_ok(&evidence, "bitnet_external_reference_boundary")
        && evidence_ok(&evidence, "bitnet_external_direct_token_boundary")
        && evidence_ok(&evidence, "bitnet_first_token_direct_classifier")
        && evidence_ok(&evidence, "bitnet_i2s_gemv_gemm_microbench")
        && evidence_ok(&evidence, "bitnet_i2s_tiling_thread_matrix")
        && evidence_ok(&evidence, "bitnet_i2s_applied_thread_matrix")
        && evidence_ok(&evidence, "bitnet_embedding_quantization_evidence");
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

#[cfg(test)]
pub fn build_regression_bundle_with_created_utc(
    root: &Path,
    operator_receipt: &Path,
    created_utc: String,
) -> Result<LunarLakeRegressionBundle> {
    build_regression_bundle_with_created_utc_and_inputs(
        root,
        operator_receipt,
        None,
        None,
        created_utc,
    )
}

pub fn build_regression_bundle_with_created_utc_and_inputs(
    root: &Path,
    operator_receipt: &Path,
    answer_corpus_v2: Option<&Path>,
    route_profile_comparison: Option<&Path>,
    created_utc: String,
) -> Result<LunarLakeRegressionBundle> {
    let operator_receipt_path = resolve_receipt_path(root, operator_receipt);
    let bytes = fs::read(&operator_receipt_path)
        .with_context(|| format!("failed to read {}", operator_receipt_path.display()))?;
    let operator: LunarLakeOperatorReceipt = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", operator_receipt_path.display()))?;

    let mut checks = vec![
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
                && evidence_ok(&operator.evidence, "bitnet_external_direct_token_boundary")
                && evidence_ok(&operator.evidence, "bitnet_first_token_direct_classifier")
                && evidence_ok(&operator.evidence, "bitnet_i2s_gemv_gemm_microbench")
                && evidence_ok(&operator.evidence, "bitnet_i2s_tiling_thread_matrix")
                && evidence_ok(&operator.evidence, "bitnet_i2s_applied_thread_matrix")
                && evidence_ok(&operator.evidence, "bitnet_embedding_quantization_evidence"),
            vec![
                BITNET_CPU_BUNDLE,
                BITNET_REFERENCE,
                BITNET_REFERENCE_DIRECT,
                BITNET_DIVERGENCE_DIRECT,
                BITNET_PERF_MICRO,
                BITNET_PERF_TILING,
                BITNET_PERF_APPLIED,
                BITNET_EMBEDDING_EVIDENCE,
            ],
            vec!["BitNet remains CPU reference-only in the operator route policy".to_string()],
        ),
        regression_check(
            "openvino_dense_slm_candidates_bounded",
            route_ok(&operator, "dense_slm_openvino_gpu_candidate")
                && route_ok(&operator, "dense_slm_openvino_npu_candidate")
                && evidence_ok(&operator.evidence, "dense_slm_openvino_gpu_arc140v")
                && evidence_ok(&operator.evidence, "dense_slm_openvino_gpu_operator_ask")
                && evidence_ok(&operator.evidence, "dense_slm_openvino_npu")
                && evidence_ok(&operator.evidence, "dense_slm_openvino_npu_operator_ask")
                && evidence_ok(&operator.evidence, "dense_slm_openvino_phase_runner"),
            vec![
                DENSE_OV_GPU,
                DENSE_OV_GPU_OPERATOR_ASK,
                DENSE_OV_NPU,
                DENSE_OV_NPU_OPERATOR_ASK,
                DENSE_OV_PHASE,
            ],
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
    let answer_corpus_v2 = if let Some(path) = answer_corpus_v2 {
        let path = resolve_receipt_path(root, path);
        let summary = inspect_answer_corpus_v2(&path)?;
        checks.push(regression_check_owned(
            "dense_slm_answer_corpus_v2_fixture",
            summary.fixture_ready,
            vec![summary.path.clone()],
            corpus_v2_notes(&summary),
        ));
        Some(summary)
    } else {
        None
    };
    let route_profile_comparison = if let Some(path) = route_profile_comparison {
        let path = resolve_receipt_path(root, path);
        let summary = inspect_route_profile_regression(&path)?;
        checks.push(regression_check_owned(
            "route_profile_comparison_regression_ready",
            summary.regression_ready,
            vec![summary.path.clone()],
            route_profile_regression_notes(&summary),
        ));
        Some(summary)
    } else {
        None
    };
    let gaps = checks
        .iter()
        .filter(|check| check.status != "passed")
        .map(|check| format!("{}: {}", check.check_id, check.notes.join(", ")))
        .collect::<Vec<_>>();
    let regression_surface = build_regression_surface_summary(
        answer_corpus_v2.as_ref(),
        route_profile_comparison.as_ref(),
    );

    Ok(LunarLakeRegressionBundle {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_regression_bundle".to_string(),
        proof_stage: "operator_regression_indexed".to_string(),
        created_utc,
        machine_id: "intel-258v".to_string(),
        artifact_root: path_string(root),
        operator_receipt: path_string(&operator_receipt_path),
        answer_corpus_v2,
        route_profile_comparison,
        regression_surface,
        regression_passed: gaps.is_empty(),
        checks,
        gaps,
        claim_boundary: operator.claim_boundary,
    })
}

fn build_regression_surface_summary(
    answer_corpus_v2: Option<&AnswerCorpusV2Summary>,
    route_profile_comparison: Option<&RouteProfileRegressionSummary>,
) -> RegressionSurfaceSummary {
    let mut summary = RegressionSurfaceSummary {
        answer_corpus_v2_indexed: answer_corpus_v2.is_some(),
        route_profile_comparison_indexed: route_profile_comparison.is_some(),
        candidate_routes_remain_unpromoted: route_profile_comparison
            .map(|summary| summary.candidate_routes_remain_unpromoted)
            .unwrap_or(false),
        benchmark_qualified_advantage_claimed: route_profile_comparison
            .map(|summary| summary.benchmark_qualified_advantage_claimed)
            .unwrap_or(false),
        fallback_observed: route_profile_comparison
            .map(|summary| summary.fallback_observed)
            .unwrap_or(false),
        gaps: Vec::new(),
        ..RegressionSurfaceSummary::default()
    };

    if let Some(corpus) = answer_corpus_v2 {
        if !corpus.fixture_ready {
            summary
                .gaps
                .push(format!("answer corpus v2 fixture is not ready: {}", corpus.gaps.join("; ")));
        }
    } else {
        summary.gaps.push("answer corpus v2 is not indexed".to_string());
    }

    if let Some(route_profiles) = route_profile_comparison {
        if !route_profiles.regression_ready {
            summary.gaps.push(format!(
                "route profile comparison is not regression-ready: {}",
                route_profiles.gaps.join("; ")
            ));
        }
        if route_profiles.fallback_observed {
            summary.gaps.push("route profile comparison observed fallback_used=true".to_string());
        }
        if route_profiles.benchmark_qualified_advantage_claimed {
            summary.gaps.push("benchmark-qualified route advantage was claimed".to_string());
        }
        if !route_profiles.candidate_routes_remain_unpromoted {
            summary
                .gaps
                .push("OpenVINO GPU/NPU candidate route became promotion-eligible".to_string());
        }
    } else {
        summary.gaps.push("route profile comparison is not indexed".to_string());
    }

    summary.gaps.sort();
    summary.gaps.dedup();
    summary.strict_ready = summary.gaps.is_empty();
    summary
}

fn strict_regression_v2_gaps(receipt: &LunarLakeRegressionBundle) -> Vec<String> {
    let mut gaps = Vec::new();
    if !receipt.regression_passed {
        gaps.extend(receipt.gaps.iter().cloned());
    }
    if !receipt.regression_surface.strict_ready {
        gaps.extend(
            receipt.regression_surface.gaps.iter().map(|gap| format!("regression_surface: {gap}")),
        );
    }
    gaps.sort();
    gaps.dedup();
    gaps
}

#[derive(Debug, Deserialize)]
struct AnswerCorpusV2Fixture {
    schema: u64,
    artifact_kind: String,
    name: String,
    #[serde(default)]
    metadata: AnswerCorpusV2Metadata,
    #[serde(default)]
    model: AnswerCorpusV2Model,
    #[serde(default)]
    cases: Vec<AnswerCorpusV2Case>,
}

#[derive(Debug, Default, Deserialize)]
struct AnswerCorpusV2Metadata {
    route_scope: Option<String>,
    prompt_template: Option<String>,
    #[serde(default)]
    claim_boundary: AnswerCorpusV2ClaimBoundary,
}

#[derive(Debug, Default, Deserialize)]
struct AnswerCorpusV2ClaimBoundary {
    broad_quality_claim: Option<bool>,
    speedup_claim: Option<bool>,
    arc_execution_claim: Option<bool>,
    npu_execution_claim: Option<bool>,
    bitnet_qk256_claim: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct AnswerCorpusV2Model {
    family: Option<String>,
    architecture: Option<String>,
    quant_format: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnswerCorpusV2Case {
    id: String,
    category: String,
    profile: String,
    #[serde(default)]
    gate: Option<serde_yaml::Value>,
}

fn inspect_answer_corpus_v2(path: &Path) -> Result<AnswerCorpusV2Summary> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let corpus: AnswerCorpusV2Fixture = serde_yaml::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;

    let profiles = sorted_unique(corpus.cases.iter().map(|case| case.profile.as_str()));
    let categories = sorted_unique(corpus.cases.iter().map(|case| case.category.as_str()));
    let mut gaps = Vec::new();
    if corpus.schema != 1 {
        gaps.push(format!("expected schema=1, got {}", corpus.schema));
    }
    if corpus.artifact_kind != "slm_answer_corpus" {
        gaps.push(format!(
            "expected artifact_kind=slm_answer_corpus, got {}",
            corpus.artifact_kind
        ));
    }
    if corpus.name != "lunar-lake-qwen25-answer-corpus-v2" {
        gaps.push(format!("unexpected corpus name {}", corpus.name));
    }
    if corpus.metadata.route_scope.as_deref() != Some(DEFAULT_ASK_ROUTE) {
        gaps.push(format!(
            "route_scope must be {DEFAULT_ASK_ROUTE}; got {:?}",
            corpus.metadata.route_scope
        ));
    }
    if corpus.model.family.as_deref() != Some("qwen")
        || corpus.model.architecture.as_deref() != Some("qwen2")
        || corpus.model.quant_format.as_deref() != Some("Q8_0")
    {
        gaps.push("model identity must remain Qwen/Qwen2 Q8_0".to_string());
    }
    if corpus.cases.len() < 10 {
        gaps.push(format!("expected at least 10 bounded cases, got {}", corpus.cases.len()));
    }
    if let Some(case) = corpus.cases.iter().find(|case| case.gate.is_none()) {
        gaps.push(format!("case {} is missing a gate", case.id));
    }

    if let Some(missing) = first_missing(&profiles, REQUIRED_CORPUS_V2_PROFILES) {
        gaps.push(format!("missing required profile {missing}"));
    }
    if let Some(missing) = first_missing(&categories, REQUIRED_CORPUS_V2_CATEGORIES) {
        gaps.push(format!("missing required category {missing}"));
    }

    let claim_boundary = &corpus.metadata.claim_boundary;
    let claim_boundary_preserved = claim_boundary.broad_quality_claim == Some(false)
        && claim_boundary.speedup_claim == Some(false)
        && claim_boundary.arc_execution_claim == Some(false)
        && claim_boundary.npu_execution_claim == Some(false)
        && claim_boundary.bitnet_qk256_claim == Some(false);
    if !claim_boundary_preserved {
        gaps.push(
            "corpus v2 claim boundary must keep quality/speedup/Arc/NPU/BitNet-QK256 claims false"
                .to_string(),
        );
    }

    Ok(AnswerCorpusV2Summary {
        path: path_string(path),
        schema: corpus.schema,
        name: corpus.name,
        route_scope: corpus.metadata.route_scope,
        model_family: corpus.model.family,
        model_architecture: corpus.model.architecture,
        quantization: corpus.model.quant_format,
        prompt_template: corpus.metadata.prompt_template,
        case_count: corpus.cases.len(),
        profiles,
        categories,
        claim_boundary_preserved,
        fixture_ready: gaps.is_empty(),
        gaps,
    })
}

fn inspect_route_profile_regression(path: &Path) -> Result<RouteProfileRegressionSummary> {
    let comparison: LunarLakeRouteProfileComparison = read_json_receipt(path)?;
    let profiles =
        comparison.profiles.iter().map(|profile| profile.profile_id.clone()).collect::<Vec<_>>();
    let mut gaps = Vec::new();
    if !comparison.profile_comparison_ready {
        gaps.push(format!("route profile comparison not ready: {}", comparison.gaps.join("; ")));
    }
    if comparison.default_route_id != DEFAULT_ASK_ROUTE {
        gaps.push(format!(
            "default route changed from {DEFAULT_ASK_ROUTE} to {}",
            comparison.default_route_id
        ));
    }
    if let Some(missing) = first_missing(&profiles, REQUIRED_ROUTE_PROFILES) {
        gaps.push(format!("route profile comparison missing profile {missing}"));
    }

    let mut fallback_observed = false;
    let mut benchmark_qualified_advantage_claimed = false;
    let mut candidate_promotion_eligible = false;
    let mut blockers = BTreeSet::new();
    for profile in &comparison.profiles {
        for route in &profile.route_evidence {
            if route.fallback_used == Some(true) {
                fallback_observed = true;
            }
            if route.benchmark_qualified_advantage {
                benchmark_qualified_advantage_claimed = true;
            }
            if is_openvino_candidate_route(&route.route_id) {
                if route.promotion_eligible_for_profile {
                    candidate_promotion_eligible = true;
                }
                for blocker in &route.blockers {
                    blockers.insert(blocker.clone());
                }
            }
        }
    }
    if fallback_observed {
        gaps.push("route profile comparison observed fallback_used=true".to_string());
    }
    if benchmark_qualified_advantage_claimed {
        gaps.push("benchmark-qualified route advantage was claimed".to_string());
    }
    if candidate_promotion_eligible {
        gaps.push("OpenVINO GPU/NPU candidate route became promotion-eligible".to_string());
    }
    if blockers.is_empty() {
        gaps.push("OpenVINO GPU/NPU candidate blockers are missing".to_string());
    }

    Ok(RouteProfileRegressionSummary {
        path: path_string(path),
        profile_comparison_ready: comparison.profile_comparison_ready,
        default_route_id: comparison.default_route_id,
        profiles,
        candidate_routes_remain_unpromoted: !candidate_promotion_eligible,
        benchmark_qualified_advantage_claimed,
        fallback_observed,
        gpu_npu_promotion_blockers: blockers.into_iter().collect(),
        regression_ready: gaps.is_empty(),
        gaps,
    })
}

fn corpus_v2_notes(summary: &AnswerCorpusV2Summary) -> Vec<String> {
    let mut notes = vec![
        format!("case_count={}", summary.case_count),
        format!("profiles={}", summary.profiles.join(",")),
        format!("categories={}", summary.categories.join(",")),
        format!("claim_boundary_preserved={}", summary.claim_boundary_preserved),
    ];
    notes.extend(summary.gaps.iter().cloned());
    notes
}

fn route_profile_regression_notes(summary: &RouteProfileRegressionSummary) -> Vec<String> {
    let mut notes = vec![
        format!("profiles={}", summary.profiles.join(",")),
        format!(
            "candidate_routes_remain_unpromoted={}",
            summary.candidate_routes_remain_unpromoted
        ),
        format!(
            "benchmark_qualified_advantage_claimed={}",
            summary.benchmark_qualified_advantage_claimed
        ),
        format!("fallback_observed={}", summary.fallback_observed),
    ];
    notes.extend(summary.gaps.iter().cloned());
    notes
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

pub fn build_route_promotion_ledger_with_created_utc(
    root: &Path,
    operator_receipt: &Path,
    comparison_receipt: &Path,
    created_utc: String,
) -> Result<LunarLakeRoutePromotionLedger> {
    let operator_receipt_path = resolve_receipt_path(root, operator_receipt);
    let comparison_receipt_path = resolve_receipt_path(root, comparison_receipt);
    let operator: LunarLakeOperatorReceipt = read_json_receipt(&operator_receipt_path)?;
    let comparison: LunarLakeComparisonReceipt = read_json_receipt(&comparison_receipt_path)?;

    let mut gaps = Vec::new();
    if !operator.operator_ready {
        gaps.push(format!("operator receipt not ready: {}", operator.gaps.join("; ")));
    }
    if !comparison.comparison_ready {
        gaps.push(format!("comparison receipt not ready: {}", comparison.gaps.join("; ")));
    }
    if operator.machine_id != comparison.machine_id {
        gaps.push(format!(
            "machine_id mismatch: operator={} comparison={}",
            operator.machine_id, comparison.machine_id
        ));
    }
    if operator.default_route.route_id != DEFAULT_ASK_ROUTE {
        gaps.push(format!(
            "default route changed from {DEFAULT_ASK_ROUTE} to {}",
            operator.default_route.route_id
        ));
    }
    if operator.claim_boundary.hidden_fallback_allowed {
        gaps.push("operator claim boundary allows hidden fallback".to_string());
    }

    let routes = operator
        .routes
        .iter()
        .map(|route| promote_route(route, &operator, &comparison))
        .collect::<Vec<_>>();

    let default_promoted = routes
        .iter()
        .any(|route| route.route_id == DEFAULT_ASK_ROUTE && route.status == "promoted");
    if !default_promoted {
        gaps.push("dense Qwen CPU default route is not promoted".to_string());
    }
    for route in &routes {
        if route.acceleration_claim {
            gaps.push(format!("route {} claims acceleration", route.route_id));
        }
        if route.speedup_claim {
            gaps.push(format!("route {} claims speedup before profile promotion", route.route_id));
        }
    }

    let promotion_ready = gaps.is_empty();
    Ok(LunarLakeRoutePromotionLedger {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_route_promotion_ledger".to_string(),
        proof_stage: "route_promotion_policy_recorded".to_string(),
        created_utc,
        machine_id: operator.machine_id.clone(),
        artifact_root: path_string(root),
        operator_receipt: path_string(&operator_receipt_path),
        comparison_receipt: path_string(&comparison_receipt_path),
        promotion_ready,
        default_route_id: operator.default_route.route_id.clone(),
        auto_route_policy: AutoRoutePolicy {
            policy_stage: "ledger_driven_auto_route_enabled".to_string(),
            default_route: DEFAULT_ASK_ROUTE.to_string(),
            hidden_fallback_allowed: false,
            cpu_default_until_profile_promoted: true,
            candidate_routes_require_profile_promotion: true,
            route_reason_required: true,
            notes: vec![
                "ledger-driven auto routing may select only routes promoted for the requested profile".to_string(),
                "dense Qwen CPU remains the user-facing auto/default route for ask profiles".to_string(),
                "OpenVINO GPU and NPU routes require profile-specific answer, fallback, phase, regression, and speedup-or-power evidence before promotion".to_string(),
                "BitNet remains a CPU reference route until accelerator BitNet parity and timing evidence exists".to_string(),
            ],
        },
        workload_profiles: workload_profiles(),
        routes,
        gaps,
        claim_boundary: operator.claim_boundary,
    })
}

#[cfg(test)]
pub fn build_route_profile_comparison_with_created_utc(
    root: &Path,
    promotion_ledger: &Path,
    phase_comparison: &Path,
    created_utc: String,
) -> Result<LunarLakeRouteProfileComparison> {
    build_route_profile_comparison_with_created_utc_and_inputs(
        root,
        promotion_ledger,
        phase_comparison,
        None,
        None,
        created_utc,
    )
}

pub fn build_route_profile_comparison_with_created_utc_and_inputs(
    root: &Path,
    promotion_ledger: &Path,
    phase_comparison: &Path,
    cpu_corpus_v2: Option<&Path>,
    openvino_corpus_v2: Option<&Path>,
    created_utc: String,
) -> Result<LunarLakeRouteProfileComparison> {
    let promotion_ledger_path = resolve_receipt_path(root, promotion_ledger);
    let phase_comparison_path = resolve_receipt_path(root, phase_comparison);
    let ledger: LunarLakeRoutePromotionLedger = read_json_receipt(&promotion_ledger_path)?;
    let phase_comparison_json: Value = read_json_receipt(&phase_comparison_path)?;
    let quality_index = load_profile_quality_index(root, cpu_corpus_v2, openvino_corpus_v2)?;

    let mut gaps = Vec::new();
    if !ledger.promotion_ready {
        gaps.push(format!("promotion ledger not ready: {}", ledger.gaps.join("; ")));
    }
    if ledger.default_route_id != DEFAULT_ASK_ROUTE {
        gaps.push(format!(
            "default route changed from {DEFAULT_ASK_ROUTE} to {}",
            ledger.default_route_id
        ));
    }
    if ledger.claim_boundary.hidden_fallback_allowed {
        gaps.push("route profile comparison refuses hidden fallback".to_string());
    }
    if !ledger.claim_boundary.openvino_gpu_npu_are_candidates_not_speedup_claims {
        gaps.push("OpenVINO GPU/NPU candidate boundary is not preserved".to_string());
    }

    let profiles = ledger
        .workload_profiles
        .iter()
        .map(|profile| {
            evaluate_workload_profile(
                root,
                profile,
                &ledger,
                &phase_comparison_json,
                &quality_index,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let default_profile_indexed = profiles.iter().any(|profile| {
        profile.promoted_route.as_deref() == Some(DEFAULT_ASK_ROUTE)
            && profile.route_evidence.iter().any(|route| route.route_id == DEFAULT_ASK_ROUTE)
    });
    if !default_profile_indexed {
        gaps.push("dense Qwen CPU default route is not indexed in route profiles".to_string());
    }
    for profile in &profiles {
        for route in &profile.route_evidence {
            if route.fallback_used == Some(true) {
                gaps.push(format!(
                    "{} route {} observed fallback_used=true",
                    profile.profile_id, route.route_id
                ));
            }
            if route.benchmark_qualified_advantage {
                gaps.push(format!(
                    "{} route {} unexpectedly records benchmark-qualified advantage",
                    profile.profile_id, route.route_id
                ));
            }
        }
    }

    let profile_comparison_ready = gaps.is_empty();
    Ok(LunarLakeRouteProfileComparison {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_route_profile_comparison".to_string(),
        proof_stage: "route_profiles_indexed_no_promotion_change".to_string(),
        created_utc,
        machine_id: ledger.machine_id.clone(),
        artifact_root: path_string(root),
        promotion_ledger: path_string(&promotion_ledger_path),
        phase_comparison_receipt: path_string(&phase_comparison_path),
        cpu_corpus_v2_receipt: quality_index.cpu_source.clone(),
        openvino_corpus_v2_receipt: quality_index.openvino_source.clone(),
        profile_comparison_ready,
        default_route_id: ledger.default_route_id,
        profiles,
        gaps,
        claim_boundary: ledger.claim_boundary,
    })
}

pub fn build_qwen_cpu_corpus_v2_diagnosis_with_created_utc(
    root: &Path,
    cpu_corpus_v2: &Path,
    route_profile_comparison: Option<&Path>,
    created_utc: String,
) -> Result<QwenCpuCorpusV2Diagnosis> {
    let cpu_corpus_v2_path = resolve_receipt_path(root, cpu_corpus_v2);
    let corpus: Value = read_json_receipt(&cpu_corpus_v2_path)?;
    let route_profile_comparison_path =
        route_profile_comparison.map(|path| resolve_receipt_path(root, path));
    let route_profile_comparison_json = route_profile_comparison_path
        .as_ref()
        .filter(|path| path.exists())
        .map(|path| read_json_receipt::<Value>(path))
        .transpose()?;
    let route_profile_statuses = cpu_route_profile_statuses(route_profile_comparison_json.as_ref());

    let mut gaps = Vec::new();
    if string_at(&corpus, "artifact_kind").as_deref() != Some("slm_cpu_answer_corpus") {
        gaps.push(
            "CPU corpus-v2 receipt must have artifact_kind=slm_cpu_answer_corpus".to_string(),
        );
    }
    if string_at(&corpus, "selected_backend").as_deref() != Some("cpu-rust") {
        gaps.push("CPU corpus-v2 receipt must select backend cpu-rust".to_string());
    }
    if string_at(&corpus, "runtime_api").as_deref() != Some("cpu") {
        gaps.push("CPU corpus-v2 receipt must record runtime_api=cpu".to_string());
    }
    let fallback_used = fallback_used(&corpus);
    if fallback_used != Some(false) {
        gaps.push("CPU corpus-v2 receipt must record fallback_used=false".to_string());
    }
    if bool_at_any(&corpus, &["speedup_claim", "claim_boundary.broad_performance_claimed"])
        == Some(true)
    {
        gaps.push(
            "CPU corpus-v2 diagnosis refuses speedup or broad performance claims".to_string(),
        );
    }
    if bool_at_any(
        &corpus,
        &[
            "claim_boundary.neural_engine_claimed",
            "claim_boundary.full_metal_inference_claimed",
            "claim_boundary.qk256_apple_claimed",
        ],
    ) == Some(true)
    {
        gaps.push("CPU corpus-v2 diagnosis refuses accelerator or BitNet QK256 claims".to_string());
    }

    let cases = corpus.get("cases").and_then(Value::as_array).cloned().unwrap_or_default();
    if cases.is_empty() {
        gaps.push("CPU corpus-v2 receipt has no cases".to_string());
    }

    let failed_cases = cases
        .iter()
        .filter(|case| !case_passed(case))
        .map(diagnose_corpus_v2_failed_case)
        .collect::<Vec<_>>();

    let profile_diagnoses =
        diagnose_corpus_v2_profiles(&corpus, &failed_cases, &route_profile_statuses);
    let quality_summary = summarize_corpus_v2_quality(&corpus, &failed_cases);
    let route_blocked = quality_summary.failed > 0 || fallback_used != Some(false);
    let blocker_summary = corpus_v2_blocker_summary(&quality_summary, fallback_used);
    let recommended_next_actions = corpus_v2_recommended_actions(&failed_cases, route_blocked);
    let diagnosis_ready = gaps.is_empty();

    Ok(QwenCpuCorpusV2Diagnosis {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_qwen_cpu_corpus_v2_diagnosis".to_string(),
        proof_stage: "corpus_v2_failures_classified_no_inference".to_string(),
        created_utc,
        machine_id: "intel-258v".to_string(),
        artifact_root: path_string(root),
        cpu_corpus_v2_receipt: path_string(&cpu_corpus_v2_path),
        route_profile_comparison_receipt: route_profile_comparison_path
            .as_ref()
            .map(|path| path_string(path)),
        route_id: DEFAULT_ASK_ROUTE.to_string(),
        model_family: string_at(&corpus, "model_family")
            .or_else(|| string_at(&corpus, "model.family")),
        model_architecture: string_at(&corpus, "model_architecture")
            .or_else(|| string_at(&corpus, "model.architecture")),
        quantization: string_at(&corpus, "quantization")
            .or_else(|| string_at(&corpus, "model.quant_format")),
        requested_backend: string_at(&corpus, "requested_backend")
            .or_else(|| string_at(&corpus, "backend.requested_backend")),
        selected_backend: string_at(&corpus, "selected_backend")
            .or_else(|| string_at(&corpus, "backend.selected_backend")),
        runtime_api: string_at(&corpus, "runtime_api")
            .or_else(|| string_at(&corpus, "backend.runtime_api")),
        fallback_used,
        quality_summary,
        profile_diagnoses,
        failed_cases,
        route_blocked,
        blocker_summary,
        recommended_next_actions,
        diagnosis_ready,
        gaps,
        claim_boundary: CorpusV2DiagnosisClaimBoundary {
            diagnostic_only: true,
            new_inference_executed: false,
            broad_quality_claim: false,
            speedup_claim: false,
            route_promotion_changed: false,
            arc_or_npu_execution_claim: false,
            bitnet_qk256_i2s_behavior_changed: false,
        },
    })
}

pub fn load_operator_ask_route(
    root: &Path,
    operator_receipt: &Path,
    route_id: &str,
) -> Result<OperatorRoute> {
    let operator_receipt_path = resolve_receipt_path(root, operator_receipt);
    let operator: LunarLakeOperatorReceipt = read_json_receipt(&operator_receipt_path)?;
    if !operator.operator_ready {
        bail!("Lunar Lake operator receipt is not ready: {}", operator.gaps.join("; "));
    }
    if operator.machine_id != "intel-258v" {
        bail!("Lunar Lake ask requires machine_id=intel-258v; got {}", operator.machine_id);
    }
    if operator.claim_boundary.hidden_fallback_allowed {
        bail!("Lunar Lake ask refuses receipts that allow hidden fallback");
    }
    if operator.claim_boundary.arc_bitnet_full_inference_claimed
        || operator.claim_boundary.npu_bitnet_full_inference_claimed
        || operator.claim_boundary.qk256_accelerator_decode_claimed
    {
        bail!("Lunar Lake ask refuses receipts with accelerator BitNet/QK256 claims");
    }

    let route = operator
        .routes
        .iter()
        .find(|route| route.route_id == route_id)
        .with_context(|| format!("operator route `{route_id}` not found"))?;
    if route.route_id != DEFAULT_ASK_ROUTE {
        bail!(
            "Lunar Lake ask currently supports only route `{DEFAULT_ASK_ROUTE}`; got `{}`",
            route.route_id
        );
    }
    if route.workload != "ask" {
        bail!("Lunar Lake ask route has unexpected workload `{}`", route.workload);
    }
    if route.selected_backend != "cpu-rust" || route.runtime_api != "cpu" {
        bail!(
            "Lunar Lake ask default route must select cpu-rust/cpu; got {}/{}",
            route.selected_backend,
            route.runtime_api
        );
    }
    if route.selected_kernel_or_runtime != "dense-qwen-cpu-reference" {
        bail!(
            "Lunar Lake ask default route must use dense-qwen-cpu-reference; got {}",
            route.selected_kernel_or_runtime
        );
    }
    if route.fallback_policy != "strict_no_fallback" {
        bail!(
            "Lunar Lake ask default route must be strict_no_fallback; got {}",
            route.fallback_policy
        );
    }
    if route.acceleration_claim {
        bail!("Lunar Lake ask default route must not claim acceleration");
    }
    for evidence_file in [&route.answer_gate_evidence, &route.phase_evidence].into_iter().flatten()
    {
        let evidence = evidence_for_file(&operator.evidence, evidence_file)
            .with_context(|| format!("route evidence `{evidence_file}` not indexed"))?;
        if !evidence.present || !evidence.issues.is_empty() {
            bail!("route evidence `{evidence_file}` is not ready: {}", evidence.issues.join("; "));
        }
        if evidence.fallback_used != Some(false) {
            bail!("route evidence `{evidence_file}` does not prove fallback_used=false");
        }
    }

    Ok(route.clone())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OperatorAskRouteSelection {
    pub requested_device: String,
    pub requested_route: String,
    pub profile_id: String,
    pub selected_route: String,
    pub selected_backend: String,
    pub runtime_api: String,
    pub promotion_status: String,
    pub selection_source: String,
    pub route_reason: String,
    pub why_not_cpu: Vec<String>,
    pub why_not_gpu: Vec<String>,
    pub why_not_npu: Vec<String>,
    pub candidate_routes: Vec<String>,
    pub promotion_ledger: Option<String>,
    pub route: OperatorRoute,
}

pub fn resolve_operator_ask_route_selection(
    root: &Path,
    operator_receipt: &Path,
    promotion_ledger: &Path,
    requested_route: &str,
    requested_device: &str,
    profile_id: &str,
) -> Result<OperatorAskRouteSelection> {
    let requested_route = normalize_auto_selector(requested_route, DEFAULT_ASK_ROUTE);
    let requested_device = normalize_auto_selector(requested_device, "auto");
    let route_auto = requested_route.eq_ignore_ascii_case("auto");
    let device_auto = requested_device.eq_ignore_ascii_case("auto");

    if !route_auto && !device_auto {
        let route = load_operator_ask_route(root, operator_receipt, &requested_route)?;
        validate_operator_ask_requested_device(&requested_device, &route)?;
        return Ok(OperatorAskRouteSelection {
            requested_device,
            requested_route,
            profile_id: profile_id.to_string(),
            selected_route: route.route_id.clone(),
            selected_backend: route.selected_backend.clone(),
            runtime_api: route.runtime_api.clone(),
            promotion_status: "direct_route_validated".to_string(),
            selection_source: "operator_receipt_direct".to_string(),
            route_reason: route.route_reason.clone(),
            why_not_cpu: if route.route_id == DEFAULT_ASK_ROUTE {
                vec!["CPU route was explicitly requested and validated".to_string()]
            } else {
                vec!["CPU route was not requested".to_string()]
            },
            why_not_gpu: vec!["auto routing was not requested".to_string()],
            why_not_npu: vec!["auto routing was not requested".to_string()],
            candidate_routes: vec![],
            promotion_ledger: None,
            route,
        });
    }

    let ledger_path = resolve_receipt_path(root, promotion_ledger);
    let ledger: LunarLakeRoutePromotionLedger = read_json_receipt(&ledger_path)?;
    validate_auto_route_ledger(&ledger)?;
    let profile = ledger
        .workload_profiles
        .iter()
        .find(|profile| profile.profile_id == profile_id)
        .with_context(|| format!("auto route profile `{profile_id}` not found in ledger"))?;
    let selected_route_id =
        profile.promoted_route.as_deref().unwrap_or(ledger.default_route_id.as_str());
    let promotion = route_promotion(&ledger, selected_route_id)?;
    validate_auto_selected_promotion(promotion, profile_id)?;
    let route = load_operator_ask_route(root, operator_receipt, selected_route_id)?;
    validate_operator_ask_requested_device(&requested_device, &route)?;
    let (why_not_cpu, why_not_gpu, why_not_npu) =
        route_selection_explanations(&ledger, profile, selected_route_id);

    Ok(OperatorAskRouteSelection {
        requested_device,
        requested_route,
        profile_id: profile.profile_id.clone(),
        selected_route: route.route_id.clone(),
        selected_backend: route.selected_backend.clone(),
        runtime_api: route.runtime_api.clone(),
        promotion_status: promotion.status.clone(),
        selection_source: "promotion_ledger_auto".to_string(),
        route_reason: promotion.reason.clone(),
        why_not_cpu,
        why_not_gpu,
        why_not_npu,
        candidate_routes: profile.candidate_routes.clone(),
        promotion_ledger: Some(path_string(&ledger_path)),
        route,
    })
}

fn normalize_auto_selector(value: &str, default_value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() { default_value.to_string() } else { trimmed.to_string() }
}

fn validate_operator_ask_requested_device(
    requested_device: &str,
    route: &OperatorRoute,
) -> Result<()> {
    if requested_device.eq_ignore_ascii_case("auto") {
        return Ok(());
    }

    let normalized = requested_device.to_ascii_lowercase();
    let route_is_cpu = route.selected_backend == "cpu-rust" && route.runtime_api == "cpu";
    if route_is_cpu && matches!(normalized.as_str(), "cpu" | "cpu-rust" | DEFAULT_ASK_ROUTE) {
        return Ok(());
    }

    bail!(
        "Lunar Lake ask route `{}` selects {}/{} but requested --device `{requested_device}`; explicit accelerator devices are not auto-routed until their routes are promoted",
        route.route_id,
        route.selected_backend,
        route.runtime_api
    )
}

fn validate_auto_route_ledger(ledger: &LunarLakeRoutePromotionLedger) -> Result<()> {
    if !ledger.promotion_ready {
        bail!("Lunar Lake route promotion ledger is not ready: {}", ledger.gaps.join("; "));
    }
    if ledger.machine_id != "intel-258v" {
        bail!("Lunar Lake auto route requires machine_id=intel-258v; got {}", ledger.machine_id);
    }
    if ledger.default_route_id != DEFAULT_ASK_ROUTE
        || ledger.auto_route_policy.default_route != DEFAULT_ASK_ROUTE
    {
        bail!(
            "Lunar Lake auto route requires default route {DEFAULT_ASK_ROUTE}; got ledger default {} policy default {}",
            ledger.default_route_id,
            ledger.auto_route_policy.default_route
        );
    }
    if ledger.auto_route_policy.hidden_fallback_allowed
        || ledger.claim_boundary.hidden_fallback_allowed
    {
        bail!("Lunar Lake auto route refuses ledgers that allow hidden fallback");
    }
    if !ledger.auto_route_policy.cpu_default_until_profile_promoted
        || !ledger.auto_route_policy.candidate_routes_require_profile_promotion
        || !ledger.auto_route_policy.route_reason_required
    {
        bail!("Lunar Lake auto route requires fail-closed route promotion policy flags");
    }
    if ledger.claim_boundary.arc_bitnet_full_inference_claimed
        || ledger.claim_boundary.npu_bitnet_full_inference_claimed
        || ledger.claim_boundary.qk256_accelerator_decode_claimed
    {
        bail!("Lunar Lake auto route refuses ledgers with accelerator BitNet/QK256 claims");
    }
    Ok(())
}

fn route_promotion<'a>(
    ledger: &'a LunarLakeRoutePromotionLedger,
    route_id: &str,
) -> Result<&'a RoutePromotion> {
    ledger
        .routes
        .iter()
        .find(|route| route.route_id == route_id)
        .with_context(|| format!("route `{route_id}` not found in promotion ledger"))
}

fn validate_auto_selected_promotion(route: &RoutePromotion, profile_id: &str) -> Result<()> {
    if route.status != "promoted" || !route.promoted_for.iter().any(|profile| profile == profile_id)
    {
        bail!(
            "route `{}` is not promoted for profile `{profile_id}`; status={} promoted_for={}",
            route.route_id,
            route.status,
            route.promoted_for.join(",")
        );
    }
    if route.fallback_used != Some(false) {
        bail!("route `{}` does not prove fallback_used=false", route.route_id);
    }
    if route.speedup_claim || route.acceleration_claim {
        bail!(
            "route `{}` cannot be auto-selected with speedup or acceleration claims",
            route.route_id
        );
    }
    if route.reason.trim().is_empty() {
        bail!("route `{}` is missing a route reason", route.route_id);
    }
    Ok(())
}

fn route_selection_explanations(
    ledger: &LunarLakeRoutePromotionLedger,
    profile: &WorkloadProfile,
    selected_route_id: &str,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let why_not_cpu = if selected_route_id == DEFAULT_ASK_ROUTE {
        vec![format!(
            "{DEFAULT_ASK_ROUTE} is promoted for profile {} and remains the safe no-fallback default",
            profile.profile_id
        )]
    } else {
        route_not_selected_reasons(ledger, DEFAULT_ASK_ROUTE, &profile.profile_id)
    };
    let why_not_gpu =
        route_not_selected_reasons(ledger, "dense_slm_openvino_gpu_candidate", &profile.profile_id);
    let why_not_npu =
        route_not_selected_reasons(ledger, "dense_slm_openvino_npu_candidate", &profile.profile_id);
    (why_not_cpu, why_not_gpu, why_not_npu)
}

fn route_not_selected_reasons(
    ledger: &LunarLakeRoutePromotionLedger,
    route_id: &str,
    profile_id: &str,
) -> Vec<String> {
    let Some(route) = ledger.routes.iter().find(|route| route.route_id == route_id) else {
        return vec![format!("route `{route_id}` is not present in the promotion ledger")];
    };
    let mut reasons = Vec::new();
    if route.status != "promoted" {
        reasons.push(format!("route status is `{}`", route.status));
    }
    if !route.promoted_for.iter().any(|profile| profile == profile_id) {
        reasons.push(format!("route is not promoted for profile `{profile_id}`"));
    }
    if route.fallback_used != Some(false) {
        reasons.push("route does not prove fallback_used=false".to_string());
    }
    if route.speedup_claim {
        reasons.push("route source claims speedup before profile promotion".to_string());
    }
    if route.acceleration_claim {
        reasons.push("route source claims acceleration before profile promotion".to_string());
    }
    for item in &route.missing_evidence {
        reasons.push(format!("missing evidence: {item}"));
    }
    if reasons.is_empty() {
        reasons.push(format!("route was not selected for profile `{profile_id}`"));
    }
    reasons.sort();
    reasons.dedup();
    reasons
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
        route_reason: "BitNet remains on CPU because the 258V CPU has the corrected reference bundle, direct bitnet.cpp generated-token/logit boundary evidence, scalar/AVX2 parity, I2_S GEMV/GEMM tuning receipts, applied-thread microbench evidence, and explicit embedding-quantization status; Arc/NPU BitNet evidence is still selected kernel or static subgraph only.".to_string(),
        answer_gate_evidence: Some(BITNET_CPU_BUNDLE.to_string()),
        phase_evidence: Some(BITNET_PERF_APPLIED.to_string()),
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
        answer_gate_evidence: Some(DENSE_OV_GPU_OPERATOR_ASK.to_string()),
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
        answer_gate_evidence: Some(DENSE_OV_NPU_OPERATOR_ASK.to_string()),
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

fn write_or_print_route_promotion_ledger(
    receipt: &LunarLakeRoutePromotionLedger,
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
        println!("Lunar Lake route promotion ledger written to {}", path.display());
    } else {
        println!("{}", String::from_utf8_lossy(&json));
    }
    Ok(())
}

fn write_or_print_route_profile_comparison(
    receipt: &LunarLakeRouteProfileComparison,
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
        println!("Lunar Lake route profile comparison written to {}", path.display());
    } else {
        println!("{}", String::from_utf8_lossy(&json));
    }
    Ok(())
}

fn write_or_print_qwen_cpu_corpus_v2_diagnosis(
    receipt: &QwenCpuCorpusV2Diagnosis,
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
        println!("Lunar Lake Qwen CPU corpus-v2 diagnosis written to {}", path.display());
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

#[derive(Debug, Clone, Default)]
struct ProfileQualityIndex {
    by_route: BTreeMap<String, BTreeMap<String, ProfileQualityEvidence>>,
    cpu_source: Option<String>,
    openvino_source: Option<String>,
}

impl ProfileQualityIndex {
    fn insert(&mut self, quality: ProfileQualityEvidence) {
        self.by_route
            .entry(quality.route_id.clone())
            .or_default()
            .insert(quality.profile_id.clone(), quality);
    }

    fn get(&self, route_id: &str, profile_id: &str) -> Option<&ProfileQualityEvidence> {
        self.by_route.get(route_id)?.get(profile_id)
    }

    fn has_route(&self, route_id: &str) -> bool {
        self.by_route.contains_key(route_id)
    }
}

fn load_profile_quality_index(
    root: &Path,
    cpu_corpus_v2: Option<&Path>,
    openvino_corpus_v2: Option<&Path>,
) -> Result<ProfileQualityIndex> {
    let mut index = ProfileQualityIndex::default();
    if let Some(path) = cpu_corpus_v2 {
        let path = resolve_receipt_path(root, path);
        let json: Value = read_json_receipt(&path)?;
        let source = path_string(&path);
        index.cpu_source = Some(source.clone());
        insert_profile_summary(
            &mut index,
            DEFAULT_ASK_ROUTE,
            &source,
            value_at(&json, "profile_summary"),
            bool_at_any(&json, &["fallback_used", "backend.fallback_used"]),
        );
    }
    if let Some(path) = openvino_corpus_v2 {
        let path = resolve_receipt_path(root, path);
        let json: Value = read_json_receipt(&path)?;
        let source = path_string(&path);
        index.openvino_source = Some(source.clone());
        if let Some(devices) = value_at(&json, "generation.devices").and_then(Value::as_array) {
            for device in devices {
                let Some(route_id) = openvino_device_route_id(device) else {
                    continue;
                };
                insert_profile_summary(
                    &mut index,
                    route_id,
                    &source,
                    value_at(device, "quality_summary.profile_summary"),
                    bool_at_any(device, &["fallback_used"]),
                );
            }
        }
    }
    Ok(index)
}

fn openvino_device_route_id(device: &Value) -> Option<&'static str> {
    match string_at_any(device, &["runtime_device", "device"]).as_deref()? {
        "GPU.0" => Some("dense_slm_openvino_gpu_candidate"),
        "NPU" => Some("dense_slm_openvino_npu_candidate"),
        _ => None,
    }
}

fn insert_profile_summary(
    index: &mut ProfileQualityIndex,
    route_id: &str,
    source_receipt: &str,
    profile_summary: Option<&Value>,
    fallback_used: Option<bool>,
) {
    let Some(summary) = profile_summary.and_then(Value::as_object) else {
        return;
    };
    for (profile_id, profile) in summary {
        let cases_total = u64_at(profile, "total").unwrap_or(0);
        let passed = u64_at(profile, "passed").unwrap_or(0);
        let failed = u64_at(profile, "failed").unwrap_or(0);
        let status = if failed == 0 && cases_total > 0 {
            "passed"
        } else if cases_total == 0 {
            "missing"
        } else {
            "quality_failed"
        };
        let mut notes = Vec::new();
        if failed > 0 {
            notes.push(format!("{failed} corpus-v2 cases failed for profile {profile_id}"));
        }
        if fallback_used == Some(true) {
            notes.push("fallback_used=true observed in corpus-v2 receipt".to_string());
        }
        index.insert(ProfileQualityEvidence {
            source_receipt: source_receipt.to_string(),
            route_id: route_id.to_string(),
            profile_id: profile_id.clone(),
            profile_present: cases_total > 0,
            cases_total,
            passed,
            failed,
            fallback_used,
            status: status.to_string(),
            notes,
        });
    }
}

#[derive(Debug, Clone, Default)]
struct CpuRouteProfileStatus {
    profile_status: Option<String>,
    promotion_decision: Option<String>,
    blockers: Vec<String>,
}

fn cpu_route_profile_statuses(
    route_profile_comparison: Option<&Value>,
) -> BTreeMap<String, CpuRouteProfileStatus> {
    let mut statuses = BTreeMap::new();
    let Some(profiles) =
        route_profile_comparison.and_then(|value| value.get("profiles")).and_then(Value::as_array)
    else {
        return statuses;
    };
    for profile in profiles {
        let Some(profile_id) = string_at(profile, "profile_id") else {
            continue;
        };
        let route = profile.get("route_evidence").and_then(Value::as_array).and_then(|routes| {
            routes
                .iter()
                .find(|route| string_at(route, "route_id").as_deref() == Some(DEFAULT_ASK_ROUTE))
        });
        let Some(route) = route else {
            continue;
        };
        statuses.insert(
            profile_id,
            CpuRouteProfileStatus {
                profile_status: string_at(profile, "profile_status"),
                promotion_decision: string_at(profile, "promotion_decision"),
                blockers: string_array_at(route, "blockers"),
            },
        );
    }
    statuses
}

fn summarize_corpus_v2_quality(
    corpus: &Value,
    failed_cases: &[CorpusV2FailedCaseDiagnosis],
) -> CorpusV2QualitySummary {
    let quality = value_at(corpus, "quality_summary");
    let failed_profiles = failed_cases
        .iter()
        .map(|case| case.profile.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let failed_categories = failed_cases
        .iter()
        .map(|case| case.category.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let mut failure_classes = BTreeMap::<String, u64>::new();
    for case in failed_cases {
        *failure_classes.entry(case.classification.clone()).or_default() += 1;
    }

    CorpusV2QualitySummary {
        total: quality.and_then(|value| u64_at(value, "total")).unwrap_or(0),
        passed: quality.and_then(|value| u64_at(value, "passed")).unwrap_or(0),
        failed: quality
            .and_then(|value| u64_at(value, "failed"))
            .unwrap_or(failed_cases.len() as u64),
        timeout: quality.and_then(|value| u64_at(value, "timeout")).unwrap_or(0),
        not_run: quality.and_then(|value| u64_at(value, "not_run")).unwrap_or(0),
        failed_profiles,
        failed_categories,
        failure_classes,
    }
}

fn diagnose_corpus_v2_profiles(
    corpus: &Value,
    failed_cases: &[CorpusV2FailedCaseDiagnosis],
    route_profile_statuses: &BTreeMap<String, CpuRouteProfileStatus>,
) -> Vec<CorpusV2ProfileDiagnosis> {
    let mut profile_ids = BTreeSet::<String>::new();
    if let Some(summary) = value_at(corpus, "profile_summary").and_then(Value::as_object) {
        profile_ids.extend(summary.keys().cloned());
    }
    profile_ids.extend(failed_cases.iter().map(|case| case.profile.clone()));
    profile_ids.extend(route_profile_statuses.keys().cloned());

    profile_ids
        .into_iter()
        .map(|profile_id| {
            let summary = value_at(corpus, "profile_summary")
                .and_then(|value| value.get(&profile_id))
                .unwrap_or(&Value::Null);
            let failed_case_ids = failed_cases
                .iter()
                .filter(|case| case.profile == profile_id)
                .map(|case| case.id.clone())
                .collect::<Vec<_>>();
            let route_status = route_profile_statuses.get(&profile_id);
            let mut route_blockers =
                route_status.map(|status| status.blockers.clone()).unwrap_or_default();
            if let Some(decision) =
                route_status.and_then(|status| status.promotion_decision.clone())
            {
                route_blockers.push(decision);
            }
            route_blockers.sort();
            route_blockers.dedup();

            let failed = u64_at(summary, "failed").unwrap_or(failed_case_ids.len() as u64);
            CorpusV2ProfileDiagnosis {
                profile_id,
                total: u64_at(summary, "total").unwrap_or(0),
                passed: u64_at(summary, "passed").unwrap_or(0),
                failed,
                blocked: failed > 0 || !route_blockers.is_empty(),
                failed_case_ids,
                route_profile_status: route_status.and_then(|status| status.profile_status.clone()),
                route_blockers,
            }
        })
        .collect()
}

fn diagnose_corpus_v2_failed_case(case: &Value) -> CorpusV2FailedCaseDiagnosis {
    let scoring = value_at(case, "quality.scoring");
    let details = scoring.and_then(|value| value.get("details"));
    let missing_required_keywords = details
        .map(|value| string_array_at(value, "required_keywords_missing"))
        .unwrap_or_default();
    let forbidden_tokens_observed = details
        .map(|value| string_array_at(value, "forbidden_tokens_observed"))
        .unwrap_or_default();
    let answer = string_at(case, "answer").unwrap_or_default();
    let failed_rules = string_array_at(case, "quality.failed_rules");
    let scoring_passed = scoring.and_then(|value| value.get("passed")).and_then(Value::as_bool);
    let gate_kind = string_at(case, "quality.gate_kind");
    let generated_tokens =
        u64_at(case, "quality.generated_tokens").or_else(|| u64_at(case, "tokens.generated"));
    let classification = classify_corpus_v2_failure(
        &answer,
        gate_kind.as_deref(),
        scoring_passed,
        &failed_rules,
        &missing_required_keywords,
        generated_tokens,
    );
    let recommended_fix =
        recommended_corpus_v2_case_fix(&classification, gate_kind.as_deref(), scoring_passed);

    CorpusV2FailedCaseDiagnosis {
        id: string_at(case, "id").unwrap_or_else(|| "unknown_case".to_string()),
        profile: string_at(case, "profile").unwrap_or_else(|| "unknown_profile".to_string()),
        category: string_at(case, "category").unwrap_or_else(|| "unknown_category".to_string()),
        task_family: string_at(case, "task_family"),
        status: string_at(case, "status").unwrap_or_else(|| "quality_failed".to_string()),
        gate_kind,
        scoring_kind: scoring.and_then(|value| string_at(value, "kind")),
        failed_rules,
        failure_taxonomy: string_array_at(case, "quality.failure_taxonomy"),
        missing_required_keywords,
        forbidden_tokens_observed,
        expected_normalized: details.and_then(|value| string_at(value, "expected_normalized")),
        observed_normalized: details.and_then(|value| string_at(value, "observed_normalized")),
        answer_preview: answer_preview(&answer),
        generated_tokens,
        prompt_tokens: u64_at(case, "tokens.prompt"),
        run_receipt_path: string_at(case, "run_receipt_path").map(|path| path.replace('\\', "/")),
        fallback_used: bool_at_any(case, &["fallback_used", "backend.fallback_used"]),
        classification,
        recommended_fix,
    }
}

fn classify_corpus_v2_failure(
    answer: &str,
    gate_kind: Option<&str>,
    scoring_passed: Option<bool>,
    failed_rules: &[String],
    missing_required_keywords: &[String],
    generated_tokens: Option<u64>,
) -> String {
    let trimmed = answer.trim_start();
    if trimmed.starts_with(':')
        && matches!(gate_kind, Some("starts_with_any"))
        && failed_rules.iter().any(|rule| rule == "gate_starts_with_any")
        && failed_rules.iter().any(|rule| rule.contains("normalized_match"))
    {
        return "assistant_prefix_gate_mismatch".to_string();
    }
    if trimmed.starts_with(':') && scoring_passed == Some(true) {
        return "gate_stricter_than_scoring_after_prefix".to_string();
    }
    if !missing_required_keywords.is_empty()
        && generated_tokens.is_some_and(|tokens| tokens <= 8)
        && (trimmed.ends_with('+') || trimmed.split_whitespace().count() < 6)
    {
        return "generation_budget_or_truncation".to_string();
    }
    if !missing_required_keywords.is_empty() {
        return "answer_content_missing_required_terms".to_string();
    }
    if failed_rules.iter().any(|rule| rule.starts_with("gate_")) {
        return "answer_gate_mismatch".to_string();
    }
    "answer_content_failed".to_string()
}

fn recommended_corpus_v2_case_fix(
    classification: &str,
    gate_kind: Option<&str>,
    scoring_passed: Option<bool>,
) -> String {
    match classification {
        "assistant_prefix_gate_mismatch" => {
            "Normalize or suppress leading assistant-role punctuation before exact starts-with/normalized-match gates, then rerun the same case without changing route promotion.".to_string()
        }
        "gate_stricter_than_scoring_after_prefix" => {
            "Review the gate versus scoring contract: scoring passed, but the bounded gate failed after role-prefix punctuation or wording drift.".to_string()
        }
        "generation_budget_or_truncation" => {
            "Rerun this bounded case with either a tighter prompt or a slightly larger max_new_tokens budget; classify as output-budget drift before changing route policy.".to_string()
        }
        "answer_content_missing_required_terms" => {
            "Treat this as a bounded answer-content failure for the dense Qwen CPU control route; adjust prompt/gate only if the expected answer contract is too narrow.".to_string()
        }
        _ if gate_kind == Some("readable") && scoring_passed == Some(false) => {
            "Readable output was produced but missed required route-policy terms; tune the prompt or expected keywords before route promotion.".to_string()
        }
        _ => "Keep this profile blocked until the case has a clean rerun or the corpus gate is intentionally revised.".to_string(),
    }
}

fn corpus_v2_blocker_summary(
    quality: &CorpusV2QualitySummary,
    fallback_used: Option<bool>,
) -> Vec<String> {
    let mut blockers = Vec::new();
    if fallback_used != Some(false) {
        blockers
            .push("fallback_used is not false in the dense Qwen CPU corpus-v2 receipt".to_string());
    }
    if quality.failed > 0 {
        blockers.push(format!(
            "{} of {} corpus-v2 cases failed across profiles [{}]",
            quality.failed,
            quality.total,
            quality.failed_profiles.join(", ")
        ));
    }
    for (classification, count) in &quality.failure_classes {
        blockers.push(format!("{count} failed case(s) classified as {classification}"));
    }
    blockers
}

fn corpus_v2_recommended_actions(
    failed_cases: &[CorpusV2FailedCaseDiagnosis],
    route_blocked: bool,
) -> Vec<String> {
    let mut actions = Vec::new();
    if route_blocked {
        actions.push(
            "Keep dense_slm_default_cpu blocked for affected corpus-v2 profiles until failed cases rerun cleanly."
                .to_string(),
        );
    }
    let classes =
        failed_cases.iter().map(|case| case.classification.as_str()).collect::<BTreeSet<_>>();
    if classes.contains("assistant_prefix_gate_mismatch") {
        actions.push(
            "Fix or normalize leading assistant-prefix punctuation for exact yes/no and one-word gates."
                .to_string(),
        );
    }
    if classes.contains("generation_budget_or_truncation") {
        actions.push(
            "Check short-answer max_new_tokens budgets before treating truncated math output as model incapability."
                .to_string(),
        );
    }
    if classes.contains("answer_content_missing_required_terms") {
        actions.push(
            "Review prompt wording and required-keyword gates for answer-content misses, then rerun corpus v2."
                .to_string(),
        );
    }
    if actions.is_empty() {
        actions.push(
            "No corpus-v2 failures were found; keep regression v2 as the guardrail.".to_string(),
        );
    }
    actions
}

fn promote_route(
    route: &OperatorRoute,
    operator: &LunarLakeOperatorReceipt,
    comparison: &LunarLakeComparisonReceipt,
) -> RoutePromotion {
    let attached = attached_route_evidence(route, &operator.evidence);
    let comparison_route = comparison.routes.iter().find(|item| item.route_id == route.route_id);
    let evidence_ready = comparison_route.is_some_and(|item| item.evidence_ready)
        && attached.iter().all(|item| item.present && item.issues.is_empty());
    let fallback_used = if attached.is_empty() {
        None
    } else {
        Some(attached.iter().any(|item| item.fallback_used == Some(true)))
    };
    let answer_gate_passed = attached.iter().filter_map(|item| item.answer_gate_passed).next();
    let phase_timing_present = attached.iter().filter_map(|item| item.phase_timing_present).next();
    let speedup_claim = attached.iter().any(|item| item.speedup_claim == Some(true));
    let mut present_evidence = Vec::new();
    let mut missing_evidence = Vec::new();
    for file_name in [&route.answer_gate_evidence, &route.phase_evidence].into_iter().flatten() {
        match evidence_for_file(&operator.evidence, file_name) {
            Some(item) if item.present && item.issues.is_empty() => {
                present_evidence.push(file_name.clone());
            }
            Some(item) => {
                missing_evidence.push(format!("{file_name}: {}", item.issues.join(", ")));
            }
            None => missing_evidence.push(format!("{file_name}: not indexed")),
        }
    }

    let mut required_evidence = vec![
        "fallback_used=false".to_string(),
        "operator_regression_or_comparison_ready".to_string(),
    ];
    let (status, promoted_for, blocked_for, reason) = match route.route_id.as_str() {
        DEFAULT_ASK_ROUTE => {
            required_evidence.push("answer_gate".to_string());
            required_evidence.push("phase_timing".to_string());
            if evidence_ready
                && fallback_used == Some(false)
                && answer_gate_passed == Some(true)
                && phase_timing_present == Some(true)
                && !route.acceleration_claim
                && !speedup_claim
            {
                (
                    "promoted".to_string(),
                    vec![
                        "regression_tiny".to_string(),
                        "ask_short".to_string(),
                        "ask_normal".to_string(),
                    ],
                    vec!["accelerator_required".to_string(), "bitnet_strict_reference".to_string()],
                    "Dense Qwen CPU is promoted as the default route because answer gates, phase evidence, strict no-fallback identity, and comparison readiness are present.".to_string(),
                )
            } else {
                (
                    "blocked".to_string(),
                    vec![],
                    vec!["all_profiles".to_string()],
                    "Dense Qwen CPU cannot be promoted until answer, phase, fallback, and comparison evidence are clean.".to_string(),
                )
            }
        }
        "bitnet_reference_cpu" => {
            required_evidence.push("corrected_cpu_reference_bundle".to_string());
            required_evidence.push("direct_bitnetcpp_boundary".to_string());
            required_evidence.push("first_token_classifier".to_string());
            required_evidence.push("bitnet_external_reference_boundary".to_string());
            required_evidence.push("bitnet_i2s_perf_evidence".to_string());
            if evidence_ready
                && fallback_used == Some(false)
                && !route.acceleration_claim
                && !speedup_claim
            {
                (
                    "promoted".to_string(),
                    vec!["bitnet_strict_reference".to_string()],
                    vec!["general_dense_slm_ask".to_string(), "auto_default".to_string()],
                    "BitNet CPU is promoted only as the strict BitNet reference route; dense Qwen CPU remains the default user-facing route.".to_string(),
                )
            } else {
                (
                    "blocked".to_string(),
                    vec![],
                    vec!["bitnet_strict_reference".to_string()],
                    "BitNet CPU reference route lacks clean no-fallback reference/perf evidence."
                        .to_string(),
                )
            }
        }
        "dense_slm_openvino_gpu_candidate" => {
            required_evidence.push("answer_gate".to_string());
            required_evidence.push("phase_timing".to_string());
            required_evidence.push("benchmark_qualified_speedup_or_power_advantage".to_string());
            required_evidence.push("profile_regression_bundle".to_string());
            if evidence_ready
                && fallback_used == Some(false)
                && answer_gate_passed == Some(true)
                && phase_timing_present == Some(true)
                && !route.acceleration_claim
                && !speedup_claim
            {
                missing_evidence.push("benchmark_qualified_speedup_or_power_advantage".to_string());
                missing_evidence.push("profile_regression_bundle".to_string());
                (
                    "candidate".to_string(),
                    vec![],
                    vec!["auto_default".to_string(), "cold_start".to_string()],
                    "OpenVINO GPU has bounded answer and phase evidence with fallback=false, but remains a candidate until a workload-profile speedup or power advantage is recorded.".to_string(),
                )
            } else {
                (
                    "blocked".to_string(),
                    vec![],
                    vec!["all_profiles".to_string()],
                    "OpenVINO GPU route cannot be considered for promotion until candidate evidence is clean.".to_string(),
                )
            }
        }
        "dense_slm_openvino_npu_candidate" => {
            required_evidence.push("answer_gate".to_string());
            required_evidence.push("phase_timing".to_string());
            required_evidence.push("benchmark_qualified_speedup_or_power_advantage".to_string());
            required_evidence.push("profile_regression_bundle".to_string());
            required_evidence.push("npu_int4_static_greedy_constraints".to_string());
            if evidence_ready
                && fallback_used == Some(false)
                && answer_gate_passed == Some(true)
                && phase_timing_present == Some(true)
                && !route.acceleration_claim
                && !speedup_claim
            {
                missing_evidence.push("benchmark_qualified_speedup_or_power_advantage".to_string());
                missing_evidence.push("profile_regression_bundle".to_string());
                (
                    "candidate".to_string(),
                    vec![],
                    vec![
                        "auto_default".to_string(),
                        "dynamic_decode".to_string(),
                        "beam_search".to_string(),
                        "parallel_sampling".to_string(),
                    ],
                    "OpenVINO NPU has bounded INT4 dense SLM answer and phase evidence with fallback=false, but remains a candidate until profile-specific advantage and constraints are recorded.".to_string(),
                )
            } else {
                (
                    "blocked".to_string(),
                    vec![],
                    vec!["all_profiles".to_string()],
                    "OpenVINO NPU route cannot be considered for promotion until candidate evidence is clean.".to_string(),
                )
            }
        }
        _ => (
            if evidence_ready { "candidate" } else { "blocked" }.to_string(),
            vec![],
            vec!["auto_default".to_string()],
            "Additional route is not promoted by the Lunar Lake route policy.".to_string(),
        ),
    };

    RoutePromotion {
        route_id: route.route_id.clone(),
        status,
        promoted_for,
        blocked_for,
        required_evidence,
        present_evidence,
        missing_evidence,
        selected_backend: route.selected_backend.clone(),
        runtime_api: route.runtime_api.clone(),
        fallback_policy: route.fallback_policy.clone(),
        answer_gate_evidence: route.answer_gate_evidence.clone(),
        phase_evidence: route.phase_evidence.clone(),
        fallback_used,
        answer_gate_passed,
        phase_timing_present,
        speedup_claim,
        acceleration_claim: route.acceleration_claim,
        last_evidence_utc: operator.created_utc.clone(),
        reason,
    }
}

fn evaluate_workload_profile(
    root: &Path,
    profile: &WorkloadProfile,
    ledger: &LunarLakeRoutePromotionLedger,
    phase_comparison: &Value,
    quality_index: &ProfileQualityIndex,
) -> Result<WorkloadProfileEvaluation> {
    let mut route_ids = Vec::new();
    if let Some(route_id) = &profile.promoted_route {
        route_ids.push(route_id.clone());
    }
    for route_id in &profile.candidate_routes {
        if !route_ids.contains(route_id) {
            route_ids.push(route_id.clone());
        }
    }

    let route_evidence = route_ids
        .iter()
        .map(|route_id| {
            evaluate_profile_route(root, profile, route_id, ledger, phase_comparison, quality_index)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut gaps = Vec::new();
    if route_evidence.is_empty() {
        gaps.push("profile has no promoted or candidate route".to_string());
    }
    for route in &route_evidence {
        if route.fallback_used == Some(true) {
            gaps.push(format!("{} fallback_used=true", route.route_id));
        }
        if route.route_status == "candidate" && route.benchmark_qualified_advantage {
            gaps.push(format!(
                "{} records benchmark advantage while still candidate",
                route.route_id
            ));
        }
    }

    let promoted_ready = route_evidence.iter().any(|route| route.promotion_eligible_for_profile);
    let promoted_route_blocked = profile.promoted_route.as_ref().is_some_and(|route_id| {
        route_evidence
            .iter()
            .any(|route| route.route_id == *route_id && !route.promotion_eligible_for_profile)
    });
    let profile_status = if promoted_ready {
        "promoted_route_ready"
    } else if promoted_route_blocked {
        "promoted_route_blocked"
    } else if route_evidence.iter().any(|route| route.route_status == "candidate") {
        "candidate_only"
    } else {
        "unqualified_gap"
    }
    .to_string();
    let promotion_decision = match &profile.promoted_route {
        Some(route_id) if promoted_ready => {
            format!("{route_id} remains promoted for {}", profile.profile_id)
        }
        Some(route_id) => format!(
            "{route_id} is listed as promoted for {}, but profile evidence is incomplete",
            profile.profile_id
        ),
        None => format!(
            "{} has no promoted route; candidate evidence is indexed without promotion",
            profile.profile_id
        ),
    };

    Ok(WorkloadProfileEvaluation {
        profile_id: profile.profile_id.clone(),
        prompt_tokens: profile.prompt_tokens.clone(),
        output_tokens: profile.output_tokens.clone(),
        purpose: profile.purpose.clone(),
        promoted_route: profile.promoted_route.clone(),
        candidate_routes: profile.candidate_routes.clone(),
        profile_status,
        route_evidence,
        promotion_decision,
        gaps,
    })
}

fn evaluate_profile_route(
    root: &Path,
    profile: &WorkloadProfile,
    route_id: &str,
    ledger: &LunarLakeRoutePromotionLedger,
    phase_comparison: &Value,
    quality_index: &ProfileQualityIndex,
) -> Result<ProfileRouteEvidence> {
    let route = ledger
        .routes
        .iter()
        .find(|route| route.route_id == route_id)
        .with_context(|| format!("route `{route_id}` not found in promotion ledger"))?;
    let timing = profile_timing_for_route(root, route_id, phase_comparison)?;
    let mut blockers = route.missing_evidence.clone();
    let profile_quality = quality_index.get(route_id, &profile.profile_id).cloned();
    if route.status != "promoted" || !route.promoted_for.contains(&profile.profile_id) {
        blockers.push(format!("route not promoted for profile {}", profile.profile_id));
    }
    if route.status == "candidate" {
        blockers.push("candidate route requires benchmark-qualified profile evidence".to_string());
    }
    if timing.known_gaps.iter().any(|gap| gap.contains("missing")) {
        blockers.push("timing coverage has missing profile fields".to_string());
    }
    if profile.profile_id == "low_power" {
        blockers.push("power telemetry receipt missing for low_power promotion".to_string());
    }
    if route.speedup_claim {
        blockers.push("route source claims speedup before profile promotion".to_string());
    }
    if let Some(quality) = &profile_quality {
        if !quality.profile_present {
            blockers.push("corpus_v2 profile quality evidence missing".to_string());
        }
        if quality.fallback_used == Some(true) {
            blockers.push("corpus_v2 profile observed fallback_used=true".to_string());
        }
        if quality.failed > 0 {
            blockers.push(format!(
                "corpus_v2 profile {} has {} quality failures",
                profile.profile_id, quality.failed
            ));
        }
    } else if quality_index.has_route(route_id) {
        blockers.push("corpus_v2 profile quality evidence missing".to_string());
    }
    blockers.sort();
    blockers.dedup();

    let promotion_eligible_for_profile = route.status == "promoted"
        && route.promoted_for.contains(&profile.profile_id)
        && route.fallback_used == Some(false)
        && blockers.is_empty();

    let mut evidence = route.present_evidence.clone();
    if let Some(quality) = &profile_quality
        && !evidence.contains(&quality.source_receipt)
    {
        evidence.push(quality.source_receipt.clone());
    }

    Ok(ProfileRouteEvidence {
        route_id: route.route_id.clone(),
        route_status: route.status.clone(),
        selected_backend: route.selected_backend.clone(),
        runtime_api: route.runtime_api.clone(),
        fallback_used: route.fallback_used,
        answer_gate_passed: route.answer_gate_passed,
        phase_timing_present: route.phase_timing_present,
        timing,
        benchmark_qualified_advantage: false,
        promotion_eligible_for_profile,
        profile_quality,
        evidence,
        blockers,
    })
}

fn profile_timing_for_route(
    root: &Path,
    route_id: &str,
    phase_comparison: &Value,
) -> Result<ProfileTimingSummary> {
    match route_id {
        DEFAULT_ASK_ROUTE => dense_cpu_profile_timing(root, phase_comparison),
        "dense_slm_openvino_gpu_candidate" => {
            openvino_profile_timing(root, DENSE_OV_GPU_OPERATOR_ASK, "openvino_gpu_operator_ask")
        }
        "dense_slm_openvino_npu_candidate" => {
            openvino_profile_timing(root, DENSE_OV_NPU_OPERATOR_ASK, "openvino_npu_operator_ask")
        }
        "bitnet_reference_cpu" => Ok(ProfileTimingSummary {
            timing_scope: "bitnet_reference_cpu_not_dense_slm_profile".to_string(),
            source_receipts: vec![BITNET_PERF_APPLIED.to_string(), BITNET_CPU_BUNDLE.to_string()],
            cold_load_ms: None,
            tokenize_ms: None,
            prefill_ms: None,
            first_token_ms: None,
            decode_total_ms: None,
            generation_total_ms: None,
            total_response_ms: None,
            output_tokens: None,
            throughput_tokens_per_s: None,
            phase_coverage: vec![
                "BitNet I2_S applied-thread evidence is indexed separately".to_string(),
                "Not comparable to dense Qwen OpenVINO route profiles".to_string(),
            ],
            known_gaps: vec![
                "BitNet route remains a strict reference route, not a general dense SLM ask route"
                    .to_string(),
            ],
        }),
        _ => Ok(ProfileTimingSummary {
            timing_scope: "unknown_route".to_string(),
            source_receipts: vec![],
            cold_load_ms: None,
            tokenize_ms: None,
            prefill_ms: None,
            first_token_ms: None,
            decode_total_ms: None,
            generation_total_ms: None,
            total_response_ms: None,
            output_tokens: None,
            throughput_tokens_per_s: None,
            phase_coverage: vec![],
            known_gaps: vec![format!("no timing extractor for route `{route_id}`")],
        }),
    }
}

fn dense_cpu_profile_timing(root: &Path, phase_comparison: &Value) -> Result<ProfileTimingSummary> {
    let ask_path = root.join(DENSE_CPU_OPERATOR_ASK);
    let ask: Value = read_json_receipt(&ask_path)?;
    let output_tokens = number_at_any(&ask, &["tokens.generated_count", "timing.decode_tokens"])
        .map(|value| value as u64);
    let generation_total_ms = number_at_any(&ask, &["timing.decode_total_ms"]);
    let throughput_tokens_per_s = number_at_any(&ask, &["timing.decode_steady_state_tok_s"])
        .or_else(|| throughput_from_tokens(output_tokens, generation_total_ms));

    let mut phase_coverage = vec![
        "operator_ask_math_brief".to_string(),
        "cpu_timing_model_load_tokenize_prefill_first_token_decode".to_string(),
    ];
    let prefill_512 = value_at(phase_comparison, "gguf_cpu_reference.timing.prefill_512").is_some();
    let decode_128 = value_at(phase_comparison, "gguf_cpu_reference.timing.decode_128").is_some();
    if prefill_512 {
        phase_coverage.push("warm_prefill_512".to_string());
    }
    if decode_128 {
        phase_coverage.push("warm_decode_128".to_string());
    }

    Ok(ProfileTimingSummary {
        timing_scope: "dense_qwen_cpu_operator_ask_plus_warm_phase_receipts".to_string(),
        source_receipts: vec![
            DENSE_CPU_OPERATOR_ASK.to_string(),
            DENSE_CPU_PHASE.to_string(),
            DENSE_PHASE_COMPARISON.to_string(),
        ],
        cold_load_ms: number_at_any(&ask, &["timing.model_load_ms"]),
        tokenize_ms: number_at_any(&ask, &["timing.tokenize_ms"]),
        prefill_ms: number_at_any(&ask, &["timing.prefill_ms"]),
        first_token_ms: number_at_any(&ask, &["timing.first_token_ms"]),
        decode_total_ms: generation_total_ms,
        generation_total_ms,
        total_response_ms: number_at_any(&ask, &["latency.total_ms"]),
        output_tokens,
        throughput_tokens_per_s,
        phase_coverage,
        known_gaps: vec![
            "bounded math ask only; not expanded profile regression corpus".to_string(),
            "power and thermal context not normalized in this comparison".to_string(),
        ],
    })
}

fn openvino_profile_timing(
    root: &Path,
    receipt_name: &str,
    timing_scope: &str,
) -> Result<ProfileTimingSummary> {
    let path = root.join(receipt_name);
    let ask: Value = read_json_receipt(&path)?;
    let output_tokens = number_at_any(&ask, &["timing.openvino_perf_metrics.num_generated_tokens"])
        .map(|value| value as u64);
    let generation_total_ms = number_at_any(
        &ask,
        &["timing.generation_wall_ms", "timing.openvino_perf_metrics.generate.mean_ms"],
    );

    Ok(ProfileTimingSummary {
        timing_scope: timing_scope.to_string(),
        source_receipts: vec![receipt_name.to_string(), DENSE_OV_PHASE.to_string()],
        cold_load_ms: number_at_any(
            &ask,
            &["timing.openvino_perf_metrics.load_time_ms", "timing.pipeline_construct_wall_ms"],
        ),
        tokenize_ms: number_at_any(&ask, &["timing.openvino_perf_metrics.tokenization.mean_ms"]),
        prefill_ms: None,
        first_token_ms: number_at_any(
            &ask,
            &[
                "timing.openvino_perf_metrics.time_to_first_token.mean_ms",
                "timing.first_streamed_text_chunk_ms",
            ],
        ),
        decode_total_ms: generation_total_ms,
        generation_total_ms,
        total_response_ms: sum_optional(
            number_at_any(&ask, &["timing.pipeline_construct_wall_ms"]),
            generation_total_ms,
        ),
        output_tokens,
        throughput_tokens_per_s: throughput_from_tokens(output_tokens, generation_total_ms),
        phase_coverage: vec![
            "bounded_operator_ask_math_brief".to_string(),
            "openvino_genai_perf_metrics".to_string(),
            "pipeline_construct_and_generation_wall_time".to_string(),
        ],
        known_gaps: vec![
            "profile regression bundle missing".to_string(),
            "benchmark-qualified speedup or power advantage missing".to_string(),
            "OpenVINO receipts do not expose prefill_512/decode_128 splits for every profile"
                .to_string(),
            "generated token IDs are not available directly from OpenVINO GenAI internals"
                .to_string(),
        ],
    })
}

fn throughput_from_tokens(tokens: Option<u64>, total_ms: Option<f64>) -> Option<f64> {
    let tokens = tokens?;
    let total_ms = total_ms?;
    if tokens == 0 || total_ms <= 0.0 {
        return None;
    }
    Some(tokens as f64 / (total_ms / 1000.0))
}

fn sum_optional(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left + right),
        _ => None,
    }
}

fn attached_route_evidence<'a>(
    route: &OperatorRoute,
    evidence: &'a [EvidenceStatus],
) -> Vec<&'a EvidenceStatus> {
    [&route.answer_gate_evidence, &route.phase_evidence]
        .into_iter()
        .flatten()
        .filter_map(|file_name| evidence_for_file(evidence, file_name))
        .collect()
}

fn workload_profiles() -> Vec<WorkloadProfile> {
    vec![
        WorkloadProfile {
            profile_id: "regression_tiny".to_string(),
            prompt_tokens: "<=64".to_string(),
            output_tokens: "<=32".to_string(),
            purpose: "cheap strict regression smoke for local runs".to_string(),
            promoted_route: Some(DEFAULT_ASK_ROUTE.to_string()),
            candidate_routes: vec![],
        },
        WorkloadProfile {
            profile_id: "ask_short".to_string(),
            prompt_tokens: "<=64".to_string(),
            output_tokens: "<=32".to_string(),
            purpose: "one-off short prompt and short answer".to_string(),
            promoted_route: Some(DEFAULT_ASK_ROUTE.to_string()),
            candidate_routes: vec![
                "dense_slm_openvino_gpu_candidate".to_string(),
                "dense_slm_openvino_npu_candidate".to_string(),
            ],
        },
        WorkloadProfile {
            profile_id: "ask_normal".to_string(),
            prompt_tokens: "<=512".to_string(),
            output_tokens: "<=128".to_string(),
            purpose: "default local assistant question profile".to_string(),
            promoted_route: Some(DEFAULT_ASK_ROUTE.to_string()),
            candidate_routes: vec![
                "dense_slm_openvino_gpu_candidate".to_string(),
                "dense_slm_openvino_npu_candidate".to_string(),
            ],
        },
        WorkloadProfile {
            profile_id: "prefill_heavy".to_string(),
            prompt_tokens: ">=2048".to_string(),
            output_tokens: "<=64".to_string(),
            purpose: "long prompt with short answer where GPU/NPU prefill may earn promotion"
                .to_string(),
            promoted_route: None,
            candidate_routes: vec![
                DEFAULT_ASK_ROUTE.to_string(),
                "dense_slm_openvino_gpu_candidate".to_string(),
                "dense_slm_openvino_npu_candidate".to_string(),
            ],
        },
        WorkloadProfile {
            profile_id: "decode_heavy".to_string(),
            prompt_tokens: "<=256".to_string(),
            output_tokens: ">=512".to_string(),
            purpose: "long answer where steady decode throughput must be measured".to_string(),
            promoted_route: None,
            candidate_routes: vec![
                DEFAULT_ASK_ROUTE.to_string(),
                "dense_slm_openvino_gpu_candidate".to_string(),
                "dense_slm_openvino_npu_candidate".to_string(),
            ],
        },
        WorkloadProfile {
            profile_id: "structured".to_string(),
            prompt_tokens: "<=512".to_string(),
            output_tokens: "<=256".to_string(),
            purpose: "bounded JSON or tool-style output with deterministic answer gates"
                .to_string(),
            promoted_route: None,
            candidate_routes: vec![DEFAULT_ASK_ROUTE.to_string()],
        },
        WorkloadProfile {
            profile_id: "low_power".to_string(),
            prompt_tokens: "<=512".to_string(),
            output_tokens: "<=128".to_string(),
            purpose:
                "battery or quiet-mode ask where NPU/GPU must prove power or stability advantage"
                    .to_string(),
            promoted_route: None,
            candidate_routes: vec![
                DEFAULT_ASK_ROUTE.to_string(),
                "dense_slm_openvino_npu_candidate".to_string(),
                "dense_slm_openvino_gpu_candidate".to_string(),
            ],
        },
        WorkloadProfile {
            profile_id: "bitnet_strict_reference".to_string(),
            prompt_tokens: "fixed BitNet corpus".to_string(),
            output_tokens: "bounded".to_string(),
            purpose: "BitNet CPU semantic/performance reference, not general dense SLM ask"
                .to_string(),
            promoted_route: Some("bitnet_reference_cpu".to_string()),
            candidate_routes: vec![],
        },
    ]
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

fn regression_check_owned(
    check_id: &str,
    passed: bool,
    evidence: Vec<String>,
    notes: Vec<String>,
) -> RegressionCheck {
    RegressionCheck {
        check_id: check_id.to_string(),
        status: if passed { "passed" } else { "failed" }.to_string(),
        evidence,
        notes,
    }
}

fn sorted_unique<'a>(items: impl Iterator<Item = &'a str>) -> Vec<String> {
    items.map(ToString::to_string).collect::<BTreeSet<_>>().into_iter().collect()
}

fn first_missing<'a>(actual: &[String], required: &'a [&str]) -> Option<&'a str> {
    required.iter().copied().find(|item| !actual.iter().any(|actual| actual == item))
}

fn is_openvino_candidate_route(route_id: &str) -> bool {
    matches!(route_id, "dense_slm_openvino_gpu_candidate" | "dense_slm_openvino_npu_candidate")
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
            "answer_gate.passed",
            "execution.answer_gate_passed",
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

fn string_array_at(json: &Value, path: &str) -> Vec<String> {
    value_at(json, path)
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_str).map(ToString::to_string).collect())
        .unwrap_or_default()
}

fn bool_at_any(json: &Value, paths: &[&str]) -> Option<bool> {
    paths.iter().find_map(|path| value_at(json, path).and_then(Value::as_bool))
}

fn number_at_any(json: &Value, paths: &[&str]) -> Option<f64> {
    paths.iter().find_map(|path| value_at(json, path).and_then(Value::as_f64))
}

fn u64_at(json: &Value, path: &str) -> Option<u64> {
    value_at(json, path).and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_f64().filter(|value| *value >= 0.0).map(|value| value as u64))
    })
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

fn answer_preview(answer: &str) -> String {
    const MAX_PREVIEW_CHARS: usize = 240;
    let mut preview = answer.chars().take(MAX_PREVIEW_CHARS).collect::<String>();
    if answer.chars().count() > MAX_PREVIEW_CHARS {
        preview.push_str("...");
    }
    preview
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn answer_gate_accepts_openvino_operator_ask_shape() {
        let receipt = json!({
            "artifact_kind": "lunar_lake_openvino_operator_ask",
            "fallback_used": false,
            "answer_gate": {
                "kind": "contains",
                "expected": "4",
                "passed": true,
                "failed_rules": []
            },
            "execution": {
                "answer_gate_passed": true
            }
        });

        assert_eq!(answer_gate_passed(&receipt), Some(true));
    }

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
        assert!(!bundle.regression_surface.strict_ready);
        assert!(
            strict_regression_v2_gaps(&bundle)
                .iter()
                .any(|gap| gap.contains("answer corpus v2 is not indexed"))
        );
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

    #[test]
    fn route_promotion_promotes_cpu_default_and_keeps_accelerators_candidate() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-14T17:00:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;
        let regression = build_regression_bundle_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "2026-05-14T17:05:00Z".to_string(),
        )?;
        fs::write(temp.path().join(REGRESSION_BUNDLE), serde_json::to_vec_pretty(&regression)?)?;
        let comparison = build_comparison_receipt_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(REGRESSION_BUNDLE),
            "2026-05-14T17:10:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_COMPARISON), serde_json::to_vec_pretty(&comparison)?)?;

        let ledger = build_route_promotion_ledger_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(OPERATOR_COMPARISON),
            "2026-05-14T17:15:00Z".to_string(),
        )?;

        assert!(ledger.promotion_ready, "{:?}", ledger.gaps);
        let Some(cpu) = ledger.routes.iter().find(|route| route.route_id == DEFAULT_ASK_ROUTE)
        else {
            bail!("missing cpu route");
        };
        assert_eq!(cpu.status, "promoted");
        assert!(cpu.promoted_for.contains(&"ask_normal".to_string()));
        let Some(gpu) =
            ledger.routes.iter().find(|route| route.route_id == "dense_slm_openvino_gpu_candidate")
        else {
            bail!("missing gpu route");
        };
        assert_eq!(gpu.status, "candidate");
        assert!(
            gpu.missing_evidence
                .contains(&"benchmark_qualified_speedup_or_power_advantage".to_string())
        );
        assert_eq!(ledger.auto_route_policy.default_route, DEFAULT_ASK_ROUTE);
        assert!(ledger.auto_route_policy.candidate_routes_require_profile_promotion);
        Ok(())
    }

    #[test]
    fn route_promotion_blocks_when_operator_comparison_failed() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), true)?;
        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-14T17:00:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;
        let regression = build_regression_bundle_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "2026-05-14T17:05:00Z".to_string(),
        )?;
        fs::write(temp.path().join(REGRESSION_BUNDLE), serde_json::to_vec_pretty(&regression)?)?;
        let comparison = build_comparison_receipt_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(REGRESSION_BUNDLE),
            "2026-05-14T17:10:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_COMPARISON), serde_json::to_vec_pretty(&comparison)?)?;

        let ledger = build_route_promotion_ledger_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(OPERATOR_COMPARISON),
            "2026-05-14T17:15:00Z".to_string(),
        )?;

        assert!(!ledger.promotion_ready);
        assert!(ledger.gaps.iter().any(|gap| gap.contains("operator receipt not ready")));
        assert!(
            ledger
                .routes
                .iter()
                .any(|route| route.route_id == DEFAULT_ASK_ROUTE && route.status == "blocked")
        );
        Ok(())
    }

    #[test]
    fn route_profile_comparison_indexes_profiles_without_promoting_accelerators() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        write_json(
            temp.path(),
            DENSE_CPU_OPERATOR_ASK,
            json!({
                "artifact_kind": "lunar_lake_operator_ask",
                "fallback_used": false,
                "answer_gate_passed": true,
                "timing": {
                    "model_load_ms": 100.0,
                    "tokenize_ms": 2.0,
                    "prefill_ms": 20.0,
                    "first_token_ms": 30.0,
                    "decode_total_ms": 90.0,
                    "decode_steady_state_tok_s": 10.0
                },
                "latency": {
                    "total_ms": 150.0
                },
                "tokens": {
                    "generated_count": 8
                }
            }),
        )?;
        write_json(
            temp.path(),
            DENSE_PHASE_COMPARISON,
            json!({
                "artifact_kind": "intel_258v_dense_slm_openvino_phase_comparison",
                "fallback_used": false,
                "gguf_cpu_reference": {
                    "timing": {
                        "prefill_512": {},
                        "decode_128": {}
                    }
                }
            }),
        )?;

        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-14T17:00:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;
        let regression = build_regression_bundle_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "2026-05-14T17:05:00Z".to_string(),
        )?;
        fs::write(temp.path().join(REGRESSION_BUNDLE), serde_json::to_vec_pretty(&regression)?)?;
        let comparison = build_comparison_receipt_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(REGRESSION_BUNDLE),
            "2026-05-14T17:10:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_COMPARISON), serde_json::to_vec_pretty(&comparison)?)?;
        let ledger = build_route_promotion_ledger_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(OPERATOR_COMPARISON),
            "2026-05-14T17:15:00Z".to_string(),
        )?;
        fs::write(temp.path().join(ROUTE_PROMOTION_LEDGER), serde_json::to_vec_pretty(&ledger)?)?;

        let profiles = build_route_profile_comparison_with_created_utc(
            temp.path(),
            Path::new(ROUTE_PROMOTION_LEDGER),
            Path::new(DENSE_PHASE_COMPARISON),
            "2026-05-14T17:30:00Z".to_string(),
        )?;

        assert!(profiles.profile_comparison_ready, "{:?}", profiles.gaps);
        assert_eq!(profiles.artifact_kind, "lunar_lake_route_profile_comparison");
        let Some(ask_normal) =
            profiles.profiles.iter().find(|profile| profile.profile_id == "ask_normal")
        else {
            bail!("missing ask_normal profile");
        };
        assert!(ask_normal.route_evidence.iter().any(|route| {
            route.route_id == DEFAULT_ASK_ROUTE && route.promotion_eligible_for_profile
        }));
        assert!(ask_normal.route_evidence.iter().any(|route| {
            route.route_id == "dense_slm_openvino_gpu_candidate"
                && route.route_status == "candidate"
                && !route.benchmark_qualified_advantage
        }));
        let Some(low_power) =
            profiles.profiles.iter().find(|profile| profile.profile_id == "low_power")
        else {
            bail!("missing low_power profile");
        };
        assert!(low_power.route_evidence.iter().any(|route| {
            route.route_id == "dense_slm_openvino_npu_candidate"
                && route.blockers.contains(
                    &"power telemetry receipt missing for low_power promotion".to_string(),
                )
        }));
        Ok(())
    }

    #[test]
    fn route_profile_comparison_indexes_corpus_v2_profile_quality_blockers() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        write_route_corpus_v2_receipts(temp.path())?;
        write_json(
            temp.path(),
            DENSE_CPU_OPERATOR_ASK,
            json!({
                "artifact_kind": "lunar_lake_operator_ask",
                "fallback_used": false,
                "answer_gate_passed": true,
                "timing": {
                    "model_load_ms": 100.0,
                    "tokenize_ms": 2.0,
                    "prefill_ms": 20.0,
                    "first_token_ms": 30.0,
                    "decode_total_ms": 90.0,
                    "decode_steady_state_tok_s": 10.0
                },
                "latency": {"total_ms": 150.0},
                "tokens": {"generated_count": 8}
            }),
        )?;
        write_json(
            temp.path(),
            DENSE_PHASE_COMPARISON,
            json!({
                "artifact_kind": "intel_258v_dense_slm_openvino_phase_comparison",
                "fallback_used": false,
                "gguf_cpu_reference": {"timing": {"prefill_512": {}, "decode_128": {}}}
            }),
        )?;

        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-14T17:00:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;
        let regression = build_regression_bundle_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "2026-05-14T17:05:00Z".to_string(),
        )?;
        fs::write(temp.path().join(REGRESSION_BUNDLE), serde_json::to_vec_pretty(&regression)?)?;
        let comparison = build_comparison_receipt_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(REGRESSION_BUNDLE),
            "2026-05-14T17:10:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_COMPARISON), serde_json::to_vec_pretty(&comparison)?)?;
        let ledger = build_route_promotion_ledger_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(OPERATOR_COMPARISON),
            "2026-05-14T17:15:00Z".to_string(),
        )?;
        fs::write(temp.path().join(ROUTE_PROMOTION_LEDGER), serde_json::to_vec_pretty(&ledger)?)?;

        let profiles = build_route_profile_comparison_with_created_utc_and_inputs(
            temp.path(),
            Path::new(ROUTE_PROMOTION_LEDGER),
            Path::new(DENSE_PHASE_COMPARISON),
            Some(Path::new(DENSE_CPU_CORPUS_V2)),
            Some(Path::new(DENSE_OV_CORPUS_V2)),
            "2026-05-16T07:30:00Z".to_string(),
        )?;

        assert!(profiles.profile_comparison_ready, "{:?}", profiles.gaps);
        let cpu_corpus_path = path_string(&temp.path().join(DENSE_CPU_CORPUS_V2));
        assert_eq!(profiles.cpu_corpus_v2_receipt.as_deref(), Some(cpu_corpus_path.as_str()));
        let Some(ask_short) =
            profiles.profiles.iter().find(|profile| profile.profile_id == "ask_short")
        else {
            bail!("missing ask_short profile");
        };
        assert_eq!(ask_short.profile_status, "promoted_route_blocked");
        let Some(cpu_route) =
            ask_short.route_evidence.iter().find(|route| route.route_id == DEFAULT_ASK_ROUTE)
        else {
            bail!("missing CPU route evidence");
        };
        assert!(!cpu_route.promotion_eligible_for_profile);
        assert_eq!(cpu_route.profile_quality.as_ref().map(|quality| quality.failed), Some(1));
        assert!(cpu_route.blockers.iter().any(|blocker| {
            blocker.contains("corpus_v2 profile ask_short has 1 quality failures")
        }));
        let Some(gpu_route) = ask_short
            .route_evidence
            .iter()
            .find(|route| route.route_id == "dense_slm_openvino_gpu_candidate")
        else {
            bail!("missing GPU route evidence");
        };
        assert_eq!(gpu_route.profile_quality.as_ref().map(|quality| quality.passed), Some(1));
        assert!(gpu_route.blockers.iter().any(|blocker| {
            blocker.contains("corpus_v2 profile ask_short has 1 quality failures")
        }));
        Ok(())
    }

    #[test]
    fn regression_bundle_v2_indexes_corpus_fixture_and_profile_comparison() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        write_answer_corpus_v2(temp.path(), "corpus-v2.yaml")?;
        write_json(
            temp.path(),
            DENSE_CPU_OPERATOR_ASK,
            json!({
                "artifact_kind": "lunar_lake_operator_ask",
                "fallback_used": false,
                "answer_gate_passed": true,
                "timing": {
                    "model_load_ms": 100.0,
                    "tokenize_ms": 2.0,
                    "prefill_ms": 20.0,
                    "first_token_ms": 30.0,
                    "decode_total_ms": 90.0,
                    "decode_steady_state_tok_s": 10.0
                },
                "latency": {"total_ms": 150.0},
                "tokens": {"generated_count": 8}
            }),
        )?;
        write_json(
            temp.path(),
            DENSE_PHASE_COMPARISON,
            json!({
                "artifact_kind": "intel_258v_dense_slm_openvino_phase_comparison",
                "fallback_used": false,
                "gguf_cpu_reference": {"timing": {"prefill_512": {}, "decode_128": {}}}
            }),
        )?;

        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-14T17:00:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;
        let regression = build_regression_bundle_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "2026-05-14T17:05:00Z".to_string(),
        )?;
        fs::write(temp.path().join(REGRESSION_BUNDLE), serde_json::to_vec_pretty(&regression)?)?;
        let comparison = build_comparison_receipt_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(REGRESSION_BUNDLE),
            "2026-05-14T17:10:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_COMPARISON), serde_json::to_vec_pretty(&comparison)?)?;
        let ledger = build_route_promotion_ledger_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(OPERATOR_COMPARISON),
            "2026-05-14T17:15:00Z".to_string(),
        )?;
        fs::write(temp.path().join(ROUTE_PROMOTION_LEDGER), serde_json::to_vec_pretty(&ledger)?)?;
        let profiles = build_route_profile_comparison_with_created_utc(
            temp.path(),
            Path::new(ROUTE_PROMOTION_LEDGER),
            Path::new(DENSE_PHASE_COMPARISON),
            "2026-05-14T17:30:00Z".to_string(),
        )?;
        fs::write(
            temp.path().join(ROUTE_PROFILE_COMPARISON),
            serde_json::to_vec_pretty(&profiles)?,
        )?;

        let bundle = build_regression_bundle_with_created_utc_and_inputs(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Some(Path::new("corpus-v2.yaml")),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            "2026-05-14T23:55:00Z".to_string(),
        )?;

        assert!(bundle.regression_passed, "{:?}", bundle.gaps);
        assert!(bundle.checks.iter().any(|check| {
            check.check_id == "dense_slm_answer_corpus_v2_fixture" && check.status == "passed"
        }));
        assert!(bundle.checks.iter().any(|check| {
            check.check_id == "route_profile_comparison_regression_ready"
                && check.status == "passed"
        }));
        let Some(corpus) = bundle.answer_corpus_v2.as_ref() else {
            bail!("missing answer_corpus_v2 summary");
        };
        assert_eq!(corpus.case_count, 11);
        assert!(corpus.profiles.contains(&"prefill_heavy".to_string()));
        let Some(route_profiles) = bundle.route_profile_comparison.as_ref() else {
            bail!("missing route_profile_comparison summary");
        };
        assert!(route_profiles.candidate_routes_remain_unpromoted);
        assert!(!route_profiles.benchmark_qualified_advantage_claimed);
        assert!(bundle.regression_surface.strict_default);
        assert!(bundle.regression_surface.strict_ready, "{:?}", bundle.regression_surface.gaps);
        assert!(bundle.regression_surface.answer_corpus_v2_indexed);
        assert!(bundle.regression_surface.route_profile_comparison_indexed);
        assert!(strict_regression_v2_gaps(&bundle).is_empty());
        Ok(())
    }

    #[test]
    fn regression_bundle_v2_fails_when_profile_comparison_reports_fallback() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        write_answer_corpus_v2(temp.path(), "corpus-v2.yaml")?;
        write_json(
            temp.path(),
            DENSE_CPU_OPERATOR_ASK,
            json!({
                "artifact_kind": "lunar_lake_operator_ask",
                "fallback_used": false,
                "answer_gate_passed": true,
                "timing": {
                    "model_load_ms": 100.0,
                    "tokenize_ms": 2.0,
                    "prefill_ms": 20.0,
                    "first_token_ms": 30.0,
                    "decode_total_ms": 90.0,
                    "decode_steady_state_tok_s": 10.0
                },
                "latency": {"total_ms": 150.0},
                "tokens": {"generated_count": 8}
            }),
        )?;
        write_json(
            temp.path(),
            DENSE_PHASE_COMPARISON,
            json!({
                "artifact_kind": "intel_258v_dense_slm_openvino_phase_comparison",
                "fallback_used": false,
                "gguf_cpu_reference": {"timing": {"prefill_512": {}, "decode_128": {}}}
            }),
        )?;

        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-14T17:00:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;
        let regression = build_regression_bundle_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "2026-05-14T17:05:00Z".to_string(),
        )?;
        fs::write(temp.path().join(REGRESSION_BUNDLE), serde_json::to_vec_pretty(&regression)?)?;
        let comparison = build_comparison_receipt_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(REGRESSION_BUNDLE),
            "2026-05-14T17:10:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_COMPARISON), serde_json::to_vec_pretty(&comparison)?)?;
        let ledger = build_route_promotion_ledger_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(OPERATOR_COMPARISON),
            "2026-05-14T17:15:00Z".to_string(),
        )?;
        fs::write(temp.path().join(ROUTE_PROMOTION_LEDGER), serde_json::to_vec_pretty(&ledger)?)?;
        let mut profiles = build_route_profile_comparison_with_created_utc(
            temp.path(),
            Path::new(ROUTE_PROMOTION_LEDGER),
            Path::new(DENSE_PHASE_COMPARISON),
            "2026-05-14T17:30:00Z".to_string(),
        )?;
        let Some(route) =
            profiles.profiles.iter_mut().flat_map(|profile| &mut profile.route_evidence).next()
        else {
            bail!("missing route profile evidence");
        };
        route.fallback_used = Some(true);
        fs::write(
            temp.path().join(ROUTE_PROFILE_COMPARISON),
            serde_json::to_vec_pretty(&profiles)?,
        )?;

        let bundle = build_regression_bundle_with_created_utc_and_inputs(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Some(Path::new("corpus-v2.yaml")),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            "2026-05-14T23:55:00Z".to_string(),
        )?;

        assert!(!bundle.regression_passed);
        assert!(!bundle.regression_surface.strict_ready);
        assert!(
            strict_regression_v2_gaps(&bundle).iter().any(|gap| gap.contains("fallback_used=true")),
            "{:?}",
            strict_regression_v2_gaps(&bundle)
        );
        Ok(())
    }

    #[test]
    fn quality_diagnosis_classifies_qwen_cpu_corpus_v2_blockers() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_json(
            temp.path(),
            DENSE_CPU_CORPUS_V2,
            json!({
                "artifact_kind": "slm_cpu_answer_corpus",
                "requested_backend": "cpu",
                "selected_backend": "cpu-rust",
                "runtime_api": "cpu",
                "fallback_used": false,
                "model_family": "qwen",
                "model_architecture": "qwen2",
                "quantization": "Q8_0",
                "speedup_claim": false,
                "quality_summary": {"total": 3, "passed": 1, "failed": 2, "timeout": 0, "not_run": 0},
                "profile_summary": {
                    "regression_tiny": {"total": 2, "passed": 1, "failed": 1},
                    "ask_short": {"total": 1, "passed": 0, "failed": 1}
                },
                "cases": [
                    {
                        "id": "math_2_plus_2_brief",
                        "profile": "regression_tiny",
                        "category": "math",
                        "status": "passed",
                        "answer": "4",
                        "quality": {"passed": true}
                    },
                    {
                        "id": "arithmetic_add_7_8",
                        "profile": "regression_tiny",
                        "category": "math",
                        "status": "quality_failed",
                        "answer": "\nThe result of 7 + ",
                        "tokens": {"prompt": 40, "generated": 8},
                        "quality": {
                            "passed": false,
                            "gate_kind": "contains_any",
                            "generated_tokens": 8,
                            "failed_rules": ["gate_contains_any", "scoring_required_keywords"],
                            "failure_taxonomy": ["answer_content"],
                            "scoring": {
                                "kind": "required_forbidden_tokens",
                                "passed": false,
                                "details": {
                                    "required_keywords_missing": ["15"],
                                    "forbidden_tokens_observed": []
                                }
                            }
                        }
                    },
                    {
                        "id": "yes_no_clear_sky",
                        "profile": "ask_short",
                        "category": "yes_no",
                        "status": "quality_failed",
                        "answer": ": Yes. The sky is usually blue",
                        "backend": {"fallback_used": false},
                        "tokens": {"prompt": 43, "generated": 8},
                        "quality": {
                            "passed": false,
                            "gate_kind": "starts_with_any",
                            "generated_tokens": 8,
                            "failed_rules": ["scoring_normalized_match"],
                            "failure_taxonomy": ["answer_content"],
                            "scoring": {
                                "kind": "normalized_match",
                                "passed": false,
                                "details": {
                                    "expected_normalized": "yes",
                                    "observed_normalized": "yes. the sky is usually blue"
                                }
                            }
                        }
                    }
                ]
            }),
        )?;
        write_json(
            temp.path(),
            ROUTE_PROFILE_COMPARISON,
            json!({
                "profiles": [
                    {
                        "profile_id": "regression_tiny",
                        "profile_status": "promoted_route_blocked",
                        "promotion_decision": "dense_slm_default_cpu is listed as promoted but blocked",
                        "route_evidence": [
                            {
                                "route_id": DEFAULT_ASK_ROUTE,
                                "blockers": ["corpus_v2 profile regression_tiny has 1 quality failures"]
                            }
                        ]
                    },
                    {
                        "profile_id": "ask_short",
                        "profile_status": "promoted_route_blocked",
                        "route_evidence": [
                            {
                                "route_id": DEFAULT_ASK_ROUTE,
                                "blockers": ["corpus_v2 profile ask_short has 1 quality failures"]
                            }
                        ]
                    }
                ]
            }),
        )?;

        let receipt = build_qwen_cpu_corpus_v2_diagnosis_with_created_utc(
            temp.path(),
            Path::new(DENSE_CPU_CORPUS_V2),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            "2026-05-16T09:30:00Z".to_string(),
        )?;

        assert!(receipt.diagnosis_ready, "{:?}", receipt.gaps);
        assert!(receipt.route_blocked);
        assert_eq!(receipt.quality_summary.failed, 2);
        assert_eq!(receipt.failed_cases.len(), 2);
        assert!(
            receipt.quality_summary.failure_classes.contains_key("generation_budget_or_truncation")
        );
        assert!(receipt.quality_summary.failure_classes.contains_key("answer_content_failed"));
        assert!(
            !receipt.quality_summary.failure_classes.contains_key("assistant_prefix_gate_mismatch")
        );
        assert!(receipt.profile_diagnoses.iter().any(|profile| profile.profile_id == "ask_short"
            && profile.route_profile_status.as_deref() == Some("promoted_route_blocked")));
        assert!(!receipt.claim_boundary.new_inference_executed);
        assert!(!receipt.claim_boundary.route_promotion_changed);
        Ok(())
    }

    #[test]
    fn ask_route_loads_dense_cpu_default_only() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-13T15:36:09Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;

        let route =
            load_operator_ask_route(temp.path(), Path::new(OPERATOR_READINESS), DEFAULT_ASK_ROUTE)?;

        assert_eq!(route.route_id, DEFAULT_ASK_ROUTE);
        assert_eq!(route.selected_backend, "cpu-rust");
        assert_eq!(route.runtime_api, "cpu");
        assert!(!route.acceleration_claim);
        Ok(())
    }

    #[test]
    fn ask_route_rejects_openvino_candidate() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-13T15:36:09Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;

        let err = load_operator_ask_route(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "dense_slm_openvino_gpu_candidate",
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("supports only route"), "got: {err}");
        Ok(())
    }

    #[test]
    fn ask_route_rejects_fallback_evidence() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), true)?;
        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-13T15:36:09Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;

        let err =
            load_operator_ask_route(temp.path(), Path::new(OPERATOR_READINESS), DEFAULT_ASK_ROUTE)
                .unwrap_err()
                .to_string();

        assert!(err.contains("not ready") || err.contains("fallback"), "got: {err}");
        Ok(())
    }

    #[test]
    fn auto_ask_selects_promoted_cpu_route_from_ledger() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-16T10:00:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;
        let regression = build_regression_bundle_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "2026-05-16T10:05:00Z".to_string(),
        )?;
        fs::write(temp.path().join(REGRESSION_BUNDLE), serde_json::to_vec_pretty(&regression)?)?;
        let comparison = build_comparison_receipt_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(REGRESSION_BUNDLE),
            "2026-05-16T10:10:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_COMPARISON), serde_json::to_vec_pretty(&comparison)?)?;
        let ledger = build_route_promotion_ledger_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(OPERATOR_COMPARISON),
            "2026-05-16T10:15:00Z".to_string(),
        )?;
        fs::write(temp.path().join(ROUTE_PROMOTION_LEDGER), serde_json::to_vec_pretty(&ledger)?)?;

        let selection = resolve_operator_ask_route_selection(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(ROUTE_PROMOTION_LEDGER),
            "auto",
            "auto",
            "ask_normal",
        )?;

        assert_eq!(selection.selection_source, "promotion_ledger_auto");
        assert_eq!(selection.selected_route, DEFAULT_ASK_ROUTE);
        assert_eq!(selection.promotion_status, "promoted");
        assert_eq!(selection.selected_backend, "cpu-rust");
        assert_eq!(selection.runtime_api, "cpu");
        assert!(
            selection.candidate_routes.contains(&"dense_slm_openvino_gpu_candidate".to_string())
        );
        assert!(selection.why_not_gpu.iter().any(|reason| {
            reason.contains("route status is `candidate`")
                || reason.contains("route is not promoted for profile")
        }));
        assert!(selection.why_not_npu.iter().any(|reason| {
            reason.contains("route status is `candidate`")
                || reason.contains("route is not promoted for profile")
        }));
        Ok(())
    }

    #[test]
    fn auto_ask_rejects_unknown_profile() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-16T10:00:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;
        let regression = build_regression_bundle_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "2026-05-16T10:05:00Z".to_string(),
        )?;
        fs::write(temp.path().join(REGRESSION_BUNDLE), serde_json::to_vec_pretty(&regression)?)?;
        let comparison = build_comparison_receipt_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(REGRESSION_BUNDLE),
            "2026-05-16T10:10:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_COMPARISON), serde_json::to_vec_pretty(&comparison)?)?;
        let ledger = build_route_promotion_ledger_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(OPERATOR_COMPARISON),
            "2026-05-16T10:15:00Z".to_string(),
        )?;
        fs::write(temp.path().join(ROUTE_PROMOTION_LEDGER), serde_json::to_vec_pretty(&ledger)?)?;

        let err = resolve_operator_ask_route_selection(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(ROUTE_PROMOTION_LEDGER),
            "auto",
            "auto",
            "unlisted_profile",
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("profile `unlisted_profile` not found"), "got: {err}");
        Ok(())
    }

    #[test]
    fn auto_ask_rejects_explicit_accelerator_device_mismatch() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-16T10:00:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;
        let regression = build_regression_bundle_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "2026-05-16T10:05:00Z".to_string(),
        )?;
        fs::write(temp.path().join(REGRESSION_BUNDLE), serde_json::to_vec_pretty(&regression)?)?;
        let comparison = build_comparison_receipt_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(REGRESSION_BUNDLE),
            "2026-05-16T10:10:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_COMPARISON), serde_json::to_vec_pretty(&comparison)?)?;
        let ledger = build_route_promotion_ledger_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(OPERATOR_COMPARISON),
            "2026-05-16T10:15:00Z".to_string(),
        )?;
        fs::write(temp.path().join(ROUTE_PROMOTION_LEDGER), serde_json::to_vec_pretty(&ledger)?)?;

        let err = resolve_operator_ask_route_selection(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(ROUTE_PROMOTION_LEDGER),
            "auto",
            "openvino-npu",
            "ask_normal",
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("requested --device `openvino-npu`"), "got: {err}");
        assert!(err.contains("explicit accelerator devices are not auto-routed"), "got: {err}");
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

        for file in [
            DENSE_CPU_ANSWER,
            DENSE_OV_CPU,
            DENSE_OV_GPU,
            DENSE_OV_NPU,
            DENSE_OV_GPU_OPERATOR_ASK,
            DENSE_OV_NPU_OPERATOR_ASK,
        ] {
            write_json(root, file, answer.clone())?;
        }
        write_json(root, DENSE_CPU_PHASE, phase)?;
        write_json(root, DENSE_OV_PHASE, openvino)?;
        for file in [
            BITNET_CPU_BUNDLE,
            BITNET_REFERENCE,
            BITNET_REFERENCE_DIRECT,
            BITNET_DIVERGENCE_DIRECT,
            ARC_OPENCL_PARITY,
            NPU_RMSNORM,
            NPU_LINEAR,
            NPU_FFN,
        ] {
            write_json(root, file, present.clone())?;
        }
        for file in
            [BITNET_PERF_MICRO, BITNET_PERF_TILING, BITNET_PERF_APPLIED, BITNET_EMBEDDING_EVIDENCE]
        {
            write_json(root, file, no_speedup.clone())?;
        }
        Ok(())
    }

    fn write_json(root: &Path, file: &str, value: Value) -> Result<()> {
        fs::create_dir_all(root)?;
        fs::write(root.join(file), serde_json::to_vec_pretty(&value)?)?;
        Ok(())
    }

    fn write_answer_corpus_v2(root: &Path, file: &str) -> Result<()> {
        fs::create_dir_all(root)?;
        fs::write(
            root.join(file),
            r#"schema: 1
artifact_kind: slm_answer_corpus
name: lunar-lake-qwen25-answer-corpus-v2
metadata:
  route_scope: dense_slm_default_cpu
  prompt_template: qwen2.5
  claim_boundary:
    broad_quality_claim: false
    speedup_claim: false
    arc_execution_claim: false
    npu_execution_claim: false
    bitnet_qk256_claim: false
model:
  family: qwen
  architecture: qwen2
  quant_format: Q8_0
cases:
  - id: math_2_plus_2_brief
    category: math
    profile: regression_tiny
    gate: {kind: contains_any}
  - id: copy_exact_color_triplet
    category: copy_exact
    profile: regression_tiny
    gate: {kind: contains_any}
  - id: yes_no_clear_sky
    category: yes_no
    profile: ask_short
    gate: {kind: starts_with_any}
  - id: short_factual_capital_france
    category: short_factual
    profile: ask_short
    gate: {kind: contains_any}
  - id: instruction_single_sentence_rust
    category: instruction_following
    profile: ask_normal
    gate: {kind: contains_any}
  - id: stop_token_one_word_done
    category: stop_and_eos
    profile: regression_tiny
    gate: {kind: starts_with_any}
  - id: transcript_context_code_word
    category: prompt_history_sensitivity
    profile: ask_normal
    gate: {kind: contains_any}
  - id: structured_json_city_country
    category: structured_output
    profile: structured
    gate: {kind: contains_any}
  - id: long_prompt_summary_route_policy
    category: long_prompt_summarization
    profile: prefill_heavy
    gate: {kind: contains_any}
  - id: short_reasoning_heavier_object
    category: short_reasoning
    profile: ask_normal
    gate: {kind: contains_any}
  - id: decode_heavy_short_list
    category: decode_heavy
    profile: decode_heavy
    gate: {kind: readable}
"#,
        )?;
        Ok(())
    }

    fn write_route_corpus_v2_receipts(root: &Path) -> Result<()> {
        write_json(
            root,
            DENSE_CPU_CORPUS_V2,
            json!({
                "artifact_kind": "slm_cpu_answer_corpus",
                "fallback_used": false,
                "profile_summary": {
                    "regression_tiny": {"total": 4, "passed": 4, "failed": 0},
                    "ask_short": {"total": 2, "passed": 1, "failed": 1},
                    "ask_normal": {"total": 3, "passed": 3, "failed": 0}
                }
            }),
        )?;
        write_json(
            root,
            DENSE_OV_CORPUS_V2,
            json!({
                "artifact_kind": "intel_258v_dense_slm_openvino_corpus_v2",
                "fallback_used": false,
                "generation": {
                    "devices": [
                        {
                            "runtime_device": "CPU",
                            "fallback_used": false,
                            "quality_summary": {
                                "profile_summary": {
                                    "ask_short": {"total": 2, "passed": 2, "failed": 0}
                                }
                            }
                        },
                        {
                            "runtime_device": "GPU.0",
                            "fallback_used": false,
                            "quality_summary": {
                                "profile_summary": {
                                    "ask_short": {"total": 2, "passed": 1, "failed": 1},
                                    "ask_normal": {"total": 3, "passed": 3, "failed": 0}
                                }
                            }
                        },
                        {
                            "runtime_device": "NPU",
                            "fallback_used": false,
                            "quality_summary": {
                                "profile_summary": {
                                    "ask_short": {"total": 2, "passed": 2, "failed": 0}
                                }
                            }
                        }
                    ]
                }
            }),
        )?;
        Ok(())
    }
}
