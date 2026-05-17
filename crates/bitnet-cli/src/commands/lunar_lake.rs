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
#[cfg(target_os = "windows")]
use std::process::Command;

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
#[cfg(test)]
const REGRESSION_BUNDLE: &str = "lunar-lake-regression-bundle.json";
const OPERATOR_COMPARISON: &str = "lunar-lake-operator-comparison.json";
const ROUTE_PROMOTION_LEDGER: &str = "lunar-lake-route-promotion.json";
const ROUTE_PROFILE_COMPARISON: &str = "lunar-lake-route-profile-comparison.json";
const REGRESSION_BUNDLE_V2: &str = "lunar-lake-regression-bundle-v2.json";
const COLD_WARM_PROFILE_BENCHMARK: &str =
    "ci/hardware/intel-258v/2026-05-08/lunar-lake-cold-warm-profile-benchmark.json";
const COLD_WARM_PROFILE_BENCHMARK_FILE: &str = "lunar-lake-cold-warm-profile-benchmark.json";
const POWER_THERMAL_CONTEXT_FILE: &str = "lunar-lake-power-thermal-context.json";
const DURABILITY_BUNDLE: &str =
    "ci/hardware/intel-258v/2026-05-08/lunar-lake-durability-bundle.json";
const DURABLE_QWEN_CPU_WARM_SESSION: &str = "lunar-lake-durable-qwen25-cpu-warm-session.json";
const CPU_SLM_PHASE_ATTRIBUTION: &str = "lunar-lake-cpu-slm-phase-attribution.json";
const CPU_SLM_RESIDENT_SESSION: &str = "lunar-lake-cpu-slm-resident-session.json";
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
const DURABILITY_REQUIRED_PROFILES: &[&str] = &["regression_tiny", "ask_short", "ask_normal"];

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

        /// Optional cold/warm profile benchmark qualification receipt to index.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = COLD_WARM_PROFILE_BENCHMARK_FILE)]
        cold_warm_benchmark: Option<PathBuf>,

        /// Optional durability bundle to index.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = DURABILITY_BUNDLE)]
        durability_bundle: Option<PathBuf>,

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

        /// Strict regression-v2 bundle to compare. Relative paths are resolved under artifact-root.
        #[arg(long, default_value = REGRESSION_BUNDLE_V2)]
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

        /// Optional power/thermal context receipt to normalize profile telemetry evidence.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long)]
        telemetry_context: Option<PathBuf>,

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

    /// Qualify cold/warm profile timing evidence without running inference or changing routes.
    #[command(alias = "bench")]
    Benchmark {
        /// Artifact root containing the 258V receipts to inspect.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// Route profile comparison receipt to inspect. Relative paths are resolved under artifact-root.
        #[arg(long, default_value = ROUTE_PROFILE_COMPARISON)]
        route_profile_comparison: PathBuf,

        /// Dense SLM phase comparison receipt to inspect. Relative paths are resolved under artifact-root.
        #[arg(long, default_value = DENSE_PHASE_COMPARISON)]
        phase_comparison: PathBuf,

        /// Optional power/thermal context receipt to attach to route timing evidence.
        #[arg(long)]
        telemetry_context: Option<PathBuf>,

        /// Output JSON cold/warm benchmark qualification receipt to file.
        #[arg(long, default_value = COLD_WARM_PROFILE_BENCHMARK)]
        json_out: PathBuf,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when the benchmark qualification surface cannot safely gate route promotion.
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Attribute the promoted dense Qwen CPU route timing from existing receipts.
    CpuSlmPhaseAttribution {
        /// Artifact root containing the 258V receipts to inspect.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// Dense Qwen CPU warm-session phase receipt to inspect.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = DENSE_CPU_PHASE)]
        cpu_phase: PathBuf,

        /// Cold/warm profile benchmark qualification receipt to inspect.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = COLD_WARM_PROFILE_BENCHMARK_FILE)]
        cold_warm_benchmark: PathBuf,

        /// Dense SLM phase comparison receipt to inspect for OpenVINO CPU context.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = DENSE_PHASE_COMPARISON)]
        phase_comparison: PathBuf,

        /// Output JSON CPU dense-SLM attribution receipt to file.
        #[arg(long, default_value = CPU_SLM_PHASE_ATTRIBUTION)]
        json_out: PathBuf,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when the attribution cannot classify the CPU timing evidence.
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Summarize resident dense Qwen CPU no-reload timing from repeated warm-session receipts.
    CpuSlmResidentSession {
        /// Artifact root containing the 258V receipts to inspect.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// CPU dense-SLM phase attribution receipt to inspect.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = CPU_SLM_PHASE_ATTRIBUTION)]
        phase_attribution: PathBuf,

        /// Repeated dense Qwen CPU warm-session receipt to inspect.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = DURABLE_QWEN_CPU_WARM_SESSION)]
        repeated_warm_session: PathBuf,

        /// Repeated executions required before the resident session can be treated as covered.
        #[arg(long, default_value_t = 10)]
        required_repeats: u64,

        /// Output JSON resident-session receipt to file.
        #[arg(long, default_value = CPU_SLM_RESIDENT_SESSION)]
        json_out: PathBuf,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when the resident-session artifact cannot classify the no-reload evidence.
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Capture current machine memory/power/thermal context for route benchmark receipts.
    #[command(alias = "telemetry")]
    TelemetryContext {
        /// Artifact root for relative output paths.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// Output JSON telemetry context receipt to file.
        #[arg(long, default_value = POWER_THERMAL_CONTEXT_FILE)]
        json_out: PathBuf,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when memory and power context cannot be captured.
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Index repeated-run durability requirements without running inference or changing routes.
    #[command(alias = "durable")]
    Durability {
        /// Artifact root containing the 258V receipts to inspect.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// Route profile comparison receipt to inspect. Relative paths are resolved under artifact-root.
        #[arg(long, default_value = ROUTE_PROFILE_COMPARISON)]
        route_profile_comparison: PathBuf,

        /// Cold/warm benchmark qualification receipt to inspect. Relative paths are resolved under artifact-root.
        #[arg(long, default_value = COLD_WARM_PROFILE_BENCHMARK_FILE)]
        cold_warm_benchmark: PathBuf,

        /// Dense Qwen CPU corpus-v2 receipt to inspect. Relative paths are resolved under artifact-root.
        #[arg(long, default_value = DENSE_CPU_CORPUS_V2)]
        cpu_corpus_v2: PathBuf,

        /// Strict regression-v2 bundle to inspect. Relative paths are resolved under artifact-root.
        #[arg(long, default_value = REGRESSION_BUNDLE_V2)]
        regression_bundle: PathBuf,

        /// Optional repeated dense Qwen CPU warm-session receipt to index.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = DURABLE_QWEN_CPU_WARM_SESSION)]
        repeated_warm_session: Option<PathBuf>,

        /// Repeated executions required before a profile can be called durable.
        #[arg(long, default_value_t = 10)]
        required_repeats: u64,

        /// Output JSON durability bundle to file.
        #[arg(long, default_value = DURABILITY_BUNDLE)]
        json_out: PathBuf,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when the durability index violates routing or claim boundaries.
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cold_warm_benchmark: Option<ColdWarmRegressionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub durability_bundle: Option<DurabilityRegressionSummary>,
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
    pub cold_warm_benchmark_indexed: bool,
    #[serde(default)]
    pub durability_bundle_indexed: bool,
    pub required_answer_profiles: Vec<String>,
    pub required_answer_categories: Vec<String>,
    pub required_route_profiles: Vec<String>,
    #[serde(default = "default_durability_required_profiles")]
    pub required_durability_profiles: Vec<String>,
    pub fallback_observed: bool,
    pub candidate_routes_remain_unpromoted: bool,
    pub benchmark_qualified_advantage_claimed: bool,
    pub cold_warm_benchmark_ready: bool,
    #[serde(default)]
    pub durability_stability_proven: bool,
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
            cold_warm_benchmark_indexed: false,
            durability_bundle_indexed: false,
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
            required_durability_profiles: default_durability_required_profiles(),
            fallback_observed: false,
            candidate_routes_remain_unpromoted: false,
            benchmark_qualified_advantage_claimed: false,
            cold_warm_benchmark_ready: false,
            durability_stability_proven: false,
            strict_ready: false,
            gaps: vec![
                "answer corpus v2 is not indexed".to_string(),
                "route profile comparison is not indexed".to_string(),
                "cold/warm benchmark qualification is not indexed".to_string(),
                "durability bundle is not indexed".to_string(),
            ],
        }
    }
}

fn default_durability_required_profiles() -> Vec<String> {
    DURABILITY_REQUIRED_PROFILES.iter().map(|profile| (*profile).to_string()).collect()
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
pub struct ColdWarmRegressionSummary {
    pub path: String,
    pub benchmark_gate_ready: bool,
    pub profiles: Vec<String>,
    pub promoted_routes_have_critical_timing: bool,
    pub candidate_routes_remain_unpromoted: bool,
    pub fallback_observed: bool,
    pub benchmark_qualified_advantage_claimed: bool,
    pub telemetry_gaps: Vec<String>,
    pub regression_ready: bool,
    pub gaps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DurabilityRegressionSummary {
    pub path: String,
    pub durability_index_ready: bool,
    pub stability_proven: bool,
    pub profiles: Vec<String>,
    pub required_repeat_count: u64,
    pub stable_profile_count: usize,
    pub fallback_observed: bool,
    pub answer_drift_detected: bool,
    pub route_drift_detected: bool,
    pub repeated_run_stability_claim: bool,
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
    #[serde(default)]
    pub regression_surface: RegressionSurfaceSummary,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry_context_receipt: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<BenchmarkTelemetry>,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LunarLakeColdWarmBenchmark {
    pub schema_version: String,
    pub artifact_kind: String,
    pub proof_stage: String,
    pub created_utc: String,
    pub machine_id: String,
    pub artifact_root: String,
    pub route_profile_comparison_receipt: String,
    pub phase_comparison_receipt: String,
    pub benchmark_gate_ready: bool,
    pub profiles: Vec<ColdWarmProfileBenchmark>,
    pub gaps: Vec<String>,
    pub claim_boundary: BenchmarkClaimBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColdWarmProfileBenchmark {
    pub profile_id: String,
    pub promoted_route: Option<String>,
    pub candidate_routes: Vec<String>,
    pub routes: Vec<ColdWarmRouteBenchmark>,
    pub profile_gaps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColdWarmRouteBenchmark {
    pub route_id: String,
    pub route_status: String,
    pub selected_backend: String,
    pub runtime_api: String,
    pub fallback_used: Option<bool>,
    pub answer_gate_passed: Option<bool>,
    pub phase_timing_present: Option<bool>,
    pub timing: ProfileTimingSummary,
    pub telemetry: BenchmarkTelemetry,
    pub critical_timing_present: bool,
    pub benchmark_qualified_advantage: bool,
    pub promotion_blocked: bool,
    pub blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkTelemetry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry_receipt: Option<String>,
    pub memory_context: String,
    pub power_context: String,
    pub thermal_context: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub telemetry_gaps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkClaimBoundary {
    pub new_inference_executed: bool,
    pub route_promotion_changed: bool,
    pub broad_quality_claim: bool,
    pub speedup_claim: bool,
    pub acceleration_claim: bool,
    pub hidden_fallback_allowed: bool,
    pub dense_slm_as_bitnet_proof: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LunarLakeCpuSlmPhaseAttribution {
    pub schema_version: String,
    pub artifact_kind: String,
    pub proof_stage: String,
    pub created_utc: String,
    pub machine_id: String,
    pub artifact_root: String,
    pub source_receipts: CpuSlmAttributionSources,
    pub model: CpuSlmAttributionModel,
    pub backend: CpuSlmAttributionBackend,
    pub cold_one_off: CpuSlmColdAttribution,
    pub warm_session: CpuSlmWarmAttribution,
    pub openvino_cpu_context: Option<CpuSlmOpenVinoCpuContext>,
    pub attribution_ready: bool,
    pub findings: Vec<String>,
    pub recommended_next_items: Vec<String>,
    pub gaps: Vec<String>,
    pub claim_boundary: CpuSlmPerfClaimBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CpuSlmAttributionSources {
    pub cpu_phase_receipt: String,
    pub cold_warm_benchmark_receipt: String,
    pub phase_comparison_receipt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CpuSlmAttributionModel {
    pub model_family: Option<String>,
    pub model_architecture: Option<String>,
    pub quantization: Option<String>,
    pub tokenizer_source: Option<String>,
    pub prompt_template: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CpuSlmAttributionBackend {
    pub route_id: String,
    pub selected_backend: String,
    pub runtime_api: String,
    pub selected_kernel_or_runtime: Option<String>,
    pub fallback_used: Option<bool>,
    pub answer_gate_passed: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CpuSlmColdAttribution {
    pub profile_id: String,
    pub timing: ProfileTimingSummary,
    pub model_load_share_of_total: Option<f64>,
    pub tokenize_share_of_total: Option<f64>,
    pub first_token_share_of_total: Option<f64>,
    pub decode_share_of_total: Option<f64>,
    pub reported_prefill_share_of_total: Option<f64>,
    pub non_decode_ms: Option<f64>,
    pub timing_notes: Vec<String>,
    pub blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CpuSlmWarmAttribution {
    pub model_loaded_once: Option<bool>,
    pub tokenizer_loaded_once: Option<bool>,
    pub model_load_ms: Option<f64>,
    pub tokenizer_load_ms: Option<f64>,
    pub total_session_ms: Option<f64>,
    pub profiles: Vec<CpuSlmWarmProfileAttribution>,
    pub timing_notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CpuSlmWarmProfileAttribution {
    pub profile: String,
    pub prompt_tokens: Option<u64>,
    pub generated_tokens: Option<u64>,
    pub prefill_ms: Option<f64>,
    pub first_token_decode_ms: Option<f64>,
    pub decode_total_ms: Option<f64>,
    pub prefill_ms_per_prompt_token: Option<f64>,
    pub decode_tokens_per_s: Option<f64>,
    pub fallback_used: Option<bool>,
    pub receipt_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CpuSlmOpenVinoCpuContext {
    pub source_receipt: Option<String>,
    pub selected_backend: Option<String>,
    pub runtime_api: Option<String>,
    pub fallback_used: Option<bool>,
    pub answer_gate_passed: Option<bool>,
    pub pipeline_load_ms: Option<f64>,
    pub case_elapsed_ms_sum: Option<f64>,
    pub timing_scope: String,
    pub comparison_notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CpuSlmPerfClaimBoundary {
    pub new_inference_executed: bool,
    pub route_promotion_changed: bool,
    pub broad_quality_claim: bool,
    pub speedup_claim: bool,
    pub power_advantage_claim: bool,
    pub acceleration_claim: bool,
    pub arc_npu_execution_claim: bool,
    pub bitnet_qk256_i2s_claim: bool,
    pub hidden_fallback_allowed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LunarLakeCpuSlmResidentSession {
    pub schema_version: String,
    pub artifact_kind: String,
    pub proof_stage: String,
    pub created_utc: String,
    pub machine_id: String,
    pub artifact_root: String,
    pub source_receipts: CpuSlmResidentSessionSources,
    pub model: CpuSlmAttributionModel,
    pub backend: CpuSlmAttributionBackend,
    pub resident_session: CpuSlmResidentSessionEvidence,
    pub cold_reference: CpuSlmResidentColdReference,
    pub profiles: Vec<CpuSlmResidentProfileSummary>,
    pub resident_ready: bool,
    pub findings: Vec<String>,
    pub recommended_next_items: Vec<String>,
    pub gaps: Vec<String>,
    pub claim_boundary: CpuSlmPerfClaimBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CpuSlmResidentSessionSources {
    pub phase_attribution_receipt: String,
    pub repeated_warm_session_receipt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CpuSlmResidentSessionEvidence {
    pub reuse_scope: Option<String>,
    pub model_loaded_once: Option<bool>,
    pub tokenizer_loaded_once: Option<bool>,
    pub model_load_ms: Option<f64>,
    pub model_sha256_ms: Option<f64>,
    pub tokenizer_load_ms: Option<f64>,
    pub total_session_ms: Option<f64>,
    pub prompt_count: Option<u64>,
    pub per_prompt_receipts_enabled: Option<bool>,
    pub session_owned_buffers: Option<bool>,
    pub prompt_token_buffer_reused: Option<bool>,
    pub generated_token_buffer_reused: Option<bool>,
    pub timing_buffers_reused: Option<bool>,
    pub stop_policy_precomputed_once: Option<bool>,
    pub resident_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CpuSlmResidentColdReference {
    pub profile_id: Option<String>,
    pub total_response_ms: Option<f64>,
    pub cold_load_ms: Option<f64>,
    pub tokenize_ms: Option<f64>,
    pub prefill_ms: Option<f64>,
    pub first_token_ms: Option<f64>,
    pub decode_total_ms: Option<f64>,
    pub timing_scope: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CpuSlmResidentProfileSummary {
    pub profile_id: String,
    pub case_ids: Vec<String>,
    pub observed_execution_count: u64,
    pub required_execution_count: u64,
    pub model_reload_observed: bool,
    pub tokenizer_reload_observed: bool,
    pub fallback_observed: bool,
    pub answer_gate_passed: bool,
    pub deterministic_generated_ids: Option<bool>,
    pub deterministic_text: Option<bool>,
    pub total_ms: CpuSlmResidentMetricSummary,
    pub time_to_first_token_ms: CpuSlmResidentMetricSummary,
    pub prefill_ms: CpuSlmResidentMetricSummary,
    pub decode_total_ms: CpuSlmResidentMetricSummary,
    pub tokenize_ms: CpuSlmResidentMetricSummary,
    pub generated_tokens: CpuSlmResidentMetricSummary,
    pub decode_tokens_per_s_mean: Option<f64>,
    pub cold_to_resident_total_ratio: Option<f64>,
    pub blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CpuSlmResidentMetricSummary {
    pub sample_count: u64,
    pub min: Option<f64>,
    pub mean: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LunarLakeTelemetryContext {
    pub schema_version: String,
    pub artifact_kind: String,
    pub proof_stage: String,
    pub created_utc: String,
    pub machine_id: String,
    pub telemetry_scope: String,
    pub memory_context: String,
    pub power_context: String,
    pub thermal_context: String,
    pub availability: TelemetryAvailability,
    pub memory: TelemetryMemoryContext,
    pub power: TelemetryPowerContext,
    pub thermal: TelemetryThermalContext,
    pub sources: Vec<TelemetrySourceStatus>,
    pub gaps: Vec<String>,
    pub claim_boundary: TelemetryClaimBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TelemetryAvailability {
    pub memory_context_recorded: bool,
    pub power_context_recorded: bool,
    pub thermal_context_recorded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TelemetryMemoryContext {
    pub source: String,
    pub total_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
    pub used_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TelemetryPowerContext {
    pub source: String,
    pub active_scheme: Option<String>,
    pub battery_status: Option<String>,
    pub ac_power_inferred: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TelemetryThermalContext {
    pub source: String,
    pub thermal_zones_visible: Option<u64>,
    pub temperatures_celsius: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TelemetrySourceStatus {
    pub source: String,
    pub available: bool,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TelemetryClaimBoundary {
    pub new_inference_executed: bool,
    pub telemetry_measurement_executed: bool,
    pub route_promotion_changed: bool,
    pub speedup_claim: bool,
    pub power_advantage_claim: bool,
    pub acceleration_claim: bool,
    pub hidden_fallback_allowed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LunarLakeDurabilityBundle {
    pub schema_version: String,
    pub artifact_kind: String,
    pub proof_stage: String,
    pub created_utc: String,
    pub machine_id: String,
    pub artifact_root: String,
    pub route_profile_comparison_receipt: String,
    pub cold_warm_benchmark_receipt: String,
    pub cpu_corpus_v2_receipt: String,
    pub regression_bundle_receipt: String,
    pub repeated_warm_session_receipt: Option<String>,
    pub required_repeat_count: u64,
    pub durability_index_ready: bool,
    pub stability_proven: bool,
    pub profiles: Vec<DurabilityProfileSummary>,
    pub gaps: Vec<String>,
    pub next_required_evidence: Vec<String>,
    pub claim_boundary: DurabilityClaimBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DurabilityProfileSummary {
    pub profile_id: String,
    pub route_id: String,
    pub route_status: String,
    pub promoted_route: Option<String>,
    pub baseline_case_count: u64,
    pub baseline_cases_passed: u64,
    pub baseline_cases_failed: u64,
    pub observed_execution_count: u64,
    pub required_execution_count: u64,
    pub answer_drift_detected: Option<bool>,
    pub route_drift_detected: bool,
    pub fallback_drift_detected: Option<bool>,
    pub latency_variance_status: String,
    pub stability_status: String,
    pub blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DurabilityClaimBoundary {
    pub new_inference_executed: bool,
    pub route_promotion_changed: bool,
    pub broad_quality_claim: bool,
    pub speedup_claim: bool,
    pub acceleration_claim: bool,
    pub hidden_fallback_allowed: bool,
    pub dense_slm_as_bitnet_proof: bool,
    pub repeated_run_stability_claim: bool,
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
                cold_warm_benchmark,
                durability_bundle,
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
                    cold_warm_benchmark.as_deref(),
                    durability_bundle.as_deref(),
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
                telemetry_context,
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
                    telemetry_context.as_deref(),
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
            LunarLakeAction::Benchmark {
                artifact_root,
                route_profile_comparison,
                phase_comparison,
                telemetry_context,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_cold_warm_benchmark_with_created_utc(
                    artifact_root,
                    route_profile_comparison,
                    phase_comparison,
                    telemetry_context.as_deref(),
                    created_utc,
                )?;
                write_or_print_cold_warm_benchmark(&receipt, Some(json_out))?;
                if *strict && !receipt.benchmark_gate_ready {
                    bail!(
                        "Lunar Lake cold/warm benchmark qualification failed: {}",
                        receipt.gaps.join("; ")
                    );
                }
                Ok(())
            }
            LunarLakeAction::CpuSlmPhaseAttribution {
                artifact_root,
                cpu_phase,
                cold_warm_benchmark,
                phase_comparison,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_cpu_slm_phase_attribution_with_created_utc(
                    artifact_root,
                    cpu_phase,
                    cold_warm_benchmark,
                    phase_comparison,
                    created_utc,
                )?;
                let json_out = resolve_receipt_path(artifact_root, json_out);
                write_or_print_cpu_slm_phase_attribution(&receipt, Some(&json_out))?;
                if *strict && !receipt.attribution_ready {
                    bail!(
                        "Lunar Lake CPU dense SLM phase attribution failed: {}",
                        receipt.gaps.join("; ")
                    );
                }
                Ok(())
            }
            LunarLakeAction::CpuSlmResidentSession {
                artifact_root,
                phase_attribution,
                repeated_warm_session,
                required_repeats,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_cpu_slm_resident_session_with_created_utc(
                    artifact_root,
                    phase_attribution,
                    repeated_warm_session,
                    *required_repeats,
                    created_utc,
                )?;
                let json_out = resolve_receipt_path(artifact_root, json_out);
                write_or_print_cpu_slm_resident_session(&receipt, Some(&json_out))?;
                if *strict && !receipt.resident_ready {
                    bail!(
                        "Lunar Lake CPU dense SLM resident-session check failed: {}",
                        receipt.gaps.join("; ")
                    );
                }
                Ok(())
            }
            LunarLakeAction::TelemetryContext { artifact_root, json_out, created_utc, strict } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_telemetry_context_with_created_utc(artifact_root, created_utc);
                let json_out = resolve_receipt_path(artifact_root, json_out);
                write_or_print_telemetry_context(&receipt, Some(&json_out))?;
                if *strict
                    && (!receipt.availability.memory_context_recorded
                        || !receipt.availability.power_context_recorded)
                {
                    bail!(
                        "Lunar Lake telemetry context capture failed required memory/power context: {}",
                        receipt.gaps.join("; ")
                    );
                }
                Ok(())
            }
            LunarLakeAction::Durability {
                artifact_root,
                route_profile_comparison,
                cold_warm_benchmark,
                cpu_corpus_v2,
                regression_bundle,
                repeated_warm_session,
                required_repeats,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_durability_bundle_with_created_utc(
                    artifact_root,
                    route_profile_comparison,
                    cold_warm_benchmark,
                    cpu_corpus_v2,
                    regression_bundle,
                    repeated_warm_session.as_deref(),
                    *required_repeats,
                    created_utc,
                )?;
                write_or_print_durability_bundle(&receipt, Some(json_out))?;
                if *strict && !receipt.durability_index_ready {
                    bail!("Lunar Lake durability index failed: {}", receipt.gaps.join("; "));
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
    cold_warm_benchmark: Option<&Path>,
    durability_bundle: Option<&Path>,
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
    let cold_warm_benchmark = if let Some(path) = cold_warm_benchmark {
        let path = resolve_receipt_path(root, path);
        let summary = inspect_cold_warm_regression(&path)?;
        checks.push(regression_check_owned(
            "cold_warm_benchmark_regression_ready",
            summary.regression_ready,
            vec![summary.path.clone()],
            cold_warm_regression_notes(&summary),
        ));
        Some(summary)
    } else {
        None
    };
    let durability_bundle = if let Some(path) = durability_bundle {
        let path = resolve_receipt_path(root, path);
        let summary = inspect_durability_regression(&path)?;
        checks.push(regression_check_owned(
            "durability_bundle_regression_ready",
            summary.regression_ready,
            vec![summary.path.clone()],
            durability_regression_notes(&summary),
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
        cold_warm_benchmark.as_ref(),
        durability_bundle.as_ref(),
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
        cold_warm_benchmark,
        durability_bundle,
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
    cold_warm_benchmark: Option<&ColdWarmRegressionSummary>,
    durability_bundle: Option<&DurabilityRegressionSummary>,
) -> RegressionSurfaceSummary {
    let mut summary = RegressionSurfaceSummary {
        answer_corpus_v2_indexed: answer_corpus_v2.is_some(),
        route_profile_comparison_indexed: route_profile_comparison.is_some(),
        cold_warm_benchmark_indexed: cold_warm_benchmark.is_some(),
        durability_bundle_indexed: durability_bundle.is_some(),
        candidate_routes_remain_unpromoted: route_profile_comparison
            .map(|summary| summary.candidate_routes_remain_unpromoted)
            .unwrap_or(false),
        benchmark_qualified_advantage_claimed: route_profile_comparison
            .map(|summary| summary.benchmark_qualified_advantage_claimed)
            .unwrap_or(false)
            || cold_warm_benchmark
                .map(|summary| summary.benchmark_qualified_advantage_claimed)
                .unwrap_or(false),
        fallback_observed: route_profile_comparison
            .map(|summary| summary.fallback_observed)
            .unwrap_or(false)
            || cold_warm_benchmark.map(|summary| summary.fallback_observed).unwrap_or(false),
        cold_warm_benchmark_ready: cold_warm_benchmark
            .map(|summary| summary.regression_ready)
            .unwrap_or(false),
        durability_stability_proven: durability_bundle
            .map(|summary| summary.stability_proven)
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

    if let Some(benchmark) = cold_warm_benchmark {
        if !benchmark.regression_ready {
            summary.gaps.push(format!(
                "cold/warm benchmark qualification is not regression-ready: {}",
                benchmark.gaps.join("; ")
            ));
        }
        if benchmark.fallback_observed {
            summary.gaps.push("cold/warm benchmark observed fallback_used=true".to_string());
        }
        if benchmark.benchmark_qualified_advantage_claimed {
            summary.gaps.push(
                "cold/warm benchmark recorded benchmark-qualified route advantage".to_string(),
            );
        }
        if !benchmark.candidate_routes_remain_unpromoted {
            summary
                .gaps
                .push("cold/warm benchmark shows OpenVINO candidate route promotion".to_string());
        }
        if !benchmark.promoted_routes_have_critical_timing {
            summary.gaps.push("promoted routes are missing critical cold/warm timing".to_string());
        }
    } else {
        summary.gaps.push("cold/warm benchmark qualification is not indexed".to_string());
    }

    if let Some(durability) = durability_bundle {
        if !durability.regression_ready {
            summary.gaps.push(format!(
                "durability bundle is not regression-ready: {}",
                durability.gaps.join("; ")
            ));
        }
        if !durability.stability_proven {
            summary
                .gaps
                .push("durability bundle has not proven repeated-run stability".to_string());
        }
        if durability.fallback_observed {
            summary.gaps.push("durability bundle observed fallback_used=true".to_string());
        }
        if durability.answer_drift_detected {
            summary.gaps.push("durability bundle observed answer drift".to_string());
        }
        if durability.route_drift_detected {
            summary.gaps.push("durability bundle observed route drift".to_string());
        }
    } else {
        summary.gaps.push("durability bundle is not indexed".to_string());
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

fn inspect_cold_warm_regression(path: &Path) -> Result<ColdWarmRegressionSummary> {
    let benchmark: LunarLakeColdWarmBenchmark = read_json_receipt(path)?;
    let profiles =
        benchmark.profiles.iter().map(|profile| profile.profile_id.clone()).collect::<Vec<_>>();
    let mut gaps = Vec::new();
    if !benchmark.benchmark_gate_ready {
        gaps.push(format!("cold/warm benchmark gate not ready: {}", benchmark.gaps.join("; ")));
    }
    if let Some(missing) = first_missing(&profiles, REQUIRED_ROUTE_PROFILES) {
        gaps.push(format!("cold/warm benchmark missing profile {missing}"));
    }
    if benchmark.claim_boundary.new_inference_executed {
        gaps.push("cold/warm benchmark executed new inference".to_string());
    }
    if benchmark.claim_boundary.route_promotion_changed {
        gaps.push("cold/warm benchmark changed route promotion".to_string());
    }
    if benchmark.claim_boundary.speedup_claim || benchmark.claim_boundary.acceleration_claim {
        gaps.push("cold/warm benchmark claimed speedup or acceleration".to_string());
    }
    if benchmark.claim_boundary.hidden_fallback_allowed {
        gaps.push("cold/warm benchmark allows hidden fallback".to_string());
    }
    if benchmark.claim_boundary.dense_slm_as_bitnet_proof {
        gaps.push("cold/warm benchmark treats dense SLM evidence as BitNet proof".to_string());
    }

    let mut fallback_observed = false;
    let mut benchmark_qualified_advantage_claimed = false;
    let mut promoted_routes_have_critical_timing = true;
    let mut candidate_routes_remain_unpromoted = true;
    let mut telemetry_gaps = BTreeSet::new();
    for profile in &benchmark.profiles {
        for route in &profile.routes {
            if route.fallback_used == Some(true) {
                fallback_observed = true;
            }
            if route.benchmark_qualified_advantage {
                benchmark_qualified_advantage_claimed = true;
            }
            if route.route_status == "promoted" && !route.critical_timing_present {
                promoted_routes_have_critical_timing = false;
            }
            if is_openvino_candidate_route(&route.route_id) && route.route_status == "promoted" {
                candidate_routes_remain_unpromoted = false;
            }
            for value in [
                &route.telemetry.memory_context,
                &route.telemetry.power_context,
                &route.telemetry.thermal_context,
            ] {
                if value.contains("not_normalized")
                    || value.contains("not_recorded")
                    || value.contains("missing")
                    || value.contains("unavailable")
                {
                    telemetry_gaps
                        .insert(format!("{}:{}={}", profile.profile_id, route.route_id, value));
                }
            }
            for gap in &route.telemetry.telemetry_gaps {
                telemetry_gaps.insert(format!("{}:{}={}", profile.profile_id, route.route_id, gap));
            }
        }
    }
    if fallback_observed {
        gaps.push("cold/warm benchmark observed fallback_used=true".to_string());
    }
    if benchmark_qualified_advantage_claimed {
        gaps.push("cold/warm benchmark recorded benchmark-qualified route advantage".to_string());
    }
    if !promoted_routes_have_critical_timing {
        gaps.push("promoted routes are missing critical cold/warm timing".to_string());
    }
    if !candidate_routes_remain_unpromoted {
        gaps.push(
            "OpenVINO GPU/NPU candidate route was promoted in cold/warm benchmark".to_string(),
        );
    }

    Ok(ColdWarmRegressionSummary {
        path: path_string(path),
        benchmark_gate_ready: benchmark.benchmark_gate_ready,
        profiles,
        promoted_routes_have_critical_timing,
        candidate_routes_remain_unpromoted,
        fallback_observed,
        benchmark_qualified_advantage_claimed,
        telemetry_gaps: telemetry_gaps.into_iter().collect(),
        regression_ready: gaps.is_empty(),
        gaps,
    })
}

fn inspect_durability_regression(path: &Path) -> Result<DurabilityRegressionSummary> {
    let bundle: LunarLakeDurabilityBundle = read_json_receipt(path)?;
    let profiles =
        bundle.profiles.iter().map(|profile| profile.profile_id.clone()).collect::<Vec<_>>();
    let mut gaps = Vec::new();
    if !bundle.durability_index_ready {
        gaps.push(format!("durability bundle is not ready: {}", bundle.gaps.join("; ")));
    }
    if !bundle.stability_proven {
        gaps.push("durability bundle has stability_proven=false".to_string());
    }
    if let Some(missing) = first_missing(&profiles, DURABILITY_REQUIRED_PROFILES) {
        gaps.push(format!("durability bundle missing profile {missing}"));
    }
    if !bundle.next_required_evidence.is_empty() {
        gaps.push(format!(
            "durability bundle still requires evidence: {}",
            bundle.next_required_evidence.join("; ")
        ));
    }
    if bundle.claim_boundary.new_inference_executed {
        gaps.push("durability bundle executed new inference".to_string());
    }
    if bundle.claim_boundary.route_promotion_changed {
        gaps.push("durability bundle changed route promotion".to_string());
    }
    if bundle.claim_boundary.broad_quality_claim {
        gaps.push("durability bundle made a broad quality claim".to_string());
    }
    if bundle.claim_boundary.speedup_claim || bundle.claim_boundary.acceleration_claim {
        gaps.push("durability bundle claimed speedup or acceleration".to_string());
    }
    if bundle.claim_boundary.hidden_fallback_allowed {
        gaps.push("durability bundle allows hidden fallback".to_string());
    }
    if bundle.claim_boundary.dense_slm_as_bitnet_proof {
        gaps.push("durability bundle treats dense SLM evidence as BitNet proof".to_string());
    }
    if !bundle.claim_boundary.repeated_run_stability_claim {
        gaps.push(
            "durability bundle must carry the bounded repeated-run stability claim".to_string(),
        );
    }

    let mut fallback_observed = false;
    let mut answer_drift_detected = false;
    let mut route_drift_detected = false;
    let mut stable_profile_count = 0usize;
    for profile_id in DURABILITY_REQUIRED_PROFILES {
        let Some(profile) =
            bundle.profiles.iter().find(|profile| profile.profile_id == *profile_id)
        else {
            continue;
        };
        if profile.route_id != DEFAULT_ASK_ROUTE {
            gaps.push(format!(
                "durability profile {profile_id} route changed to {}",
                profile.route_id
            ));
        }
        if profile.observed_execution_count < profile.required_execution_count {
            gaps.push(format!(
                "durability profile {profile_id} observed {}/{} executions",
                profile.observed_execution_count, profile.required_execution_count
            ));
        }
        if profile.observed_execution_count < bundle.required_repeat_count {
            gaps.push(format!(
                "durability profile {profile_id} is below bundle required_repeat_count {}",
                bundle.required_repeat_count
            ));
        }
        if profile.answer_drift_detected != Some(false) {
            answer_drift_detected = true;
        }
        if profile.route_drift_detected {
            route_drift_detected = true;
        }
        if profile.fallback_drift_detected != Some(false) {
            fallback_observed = true;
        }
        if profile.stability_status != "stable" {
            gaps.push(format!(
                "durability profile {profile_id} stability_status={}",
                profile.stability_status
            ));
        }
        if !profile.blockers.is_empty() {
            gaps.push(format!(
                "durability profile {profile_id} blockers: {}",
                profile.blockers.join("; ")
            ));
        }
        if profile.stability_status == "stable" && profile.blockers.is_empty() {
            stable_profile_count += 1;
        }
    }
    if fallback_observed {
        gaps.push("durability bundle observed fallback drift".to_string());
    }
    if answer_drift_detected {
        gaps.push("durability bundle observed answer drift".to_string());
    }
    if route_drift_detected {
        gaps.push("durability bundle observed route drift".to_string());
    }

    gaps.sort();
    gaps.dedup();
    Ok(DurabilityRegressionSummary {
        path: path_string(path),
        durability_index_ready: bundle.durability_index_ready,
        stability_proven: bundle.stability_proven,
        profiles,
        required_repeat_count: bundle.required_repeat_count,
        stable_profile_count,
        fallback_observed,
        answer_drift_detected,
        route_drift_detected,
        repeated_run_stability_claim: bundle.claim_boundary.repeated_run_stability_claim,
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

fn cold_warm_regression_notes(summary: &ColdWarmRegressionSummary) -> Vec<String> {
    let mut notes = vec![
        format!("profiles={}", summary.profiles.join(",")),
        format!("benchmark_gate_ready={}", summary.benchmark_gate_ready),
        format!(
            "promoted_routes_have_critical_timing={}",
            summary.promoted_routes_have_critical_timing
        ),
        format!(
            "candidate_routes_remain_unpromoted={}",
            summary.candidate_routes_remain_unpromoted
        ),
        format!(
            "benchmark_qualified_advantage_claimed={}",
            summary.benchmark_qualified_advantage_claimed
        ),
        format!("fallback_observed={}", summary.fallback_observed),
        format!("telemetry_gap_count={}", summary.telemetry_gaps.len()),
    ];
    notes.extend(summary.gaps.iter().cloned());
    notes
}

fn durability_regression_notes(summary: &DurabilityRegressionSummary) -> Vec<String> {
    let mut notes = vec![
        format!("profiles={}", summary.profiles.join(",")),
        format!("durability_index_ready={}", summary.durability_index_ready),
        format!("stability_proven={}", summary.stability_proven),
        format!("required_repeat_count={}", summary.required_repeat_count),
        format!("stable_profile_count={}", summary.stable_profile_count),
        format!("fallback_observed={}", summary.fallback_observed),
        format!("answer_drift_detected={}", summary.answer_drift_detected),
        format!("route_drift_detected={}", summary.route_drift_detected),
        format!("repeated_run_stability_claim={}", summary.repeated_run_stability_claim),
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
        regression_surface: regression.regression_surface,
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
    telemetry_context: Option<&Path>,
    created_utc: String,
) -> Result<LunarLakeRouteProfileComparison> {
    let promotion_ledger_path = resolve_receipt_path(root, promotion_ledger);
    let phase_comparison_path = resolve_receipt_path(root, phase_comparison);
    let ledger: LunarLakeRoutePromotionLedger = read_json_receipt(&promotion_ledger_path)?;
    let phase_comparison_json: Value = read_json_receipt(&phase_comparison_path)?;
    let quality_index = load_profile_quality_index(root, cpu_corpus_v2, openvino_corpus_v2)?;

    let mut gaps = Vec::new();
    let telemetry_context = load_benchmark_telemetry_context(root, telemetry_context, &mut gaps)?;
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
                telemetry_context.as_ref(),
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
        telemetry_context_receipt: telemetry_context
            .as_ref()
            .map(|context| context.receipt.clone()),
        profile_comparison_ready,
        default_route_id: ledger.default_route_id,
        profiles,
        gaps,
        claim_boundary: ledger.claim_boundary,
    })
}

pub fn build_cold_warm_benchmark_with_created_utc(
    root: &Path,
    route_profile_comparison: &Path,
    phase_comparison: &Path,
    telemetry_context: Option<&Path>,
    created_utc: String,
) -> Result<LunarLakeColdWarmBenchmark> {
    let route_profile_comparison_path = resolve_receipt_path(root, route_profile_comparison);
    let phase_comparison_path = resolve_receipt_path(root, phase_comparison);
    let comparison: LunarLakeRouteProfileComparison =
        read_json_receipt(&route_profile_comparison_path)?;
    let phase_comparison_json: Value = read_json_receipt(&phase_comparison_path)?;

    let mut gaps = Vec::new();
    let telemetry_context = load_benchmark_telemetry_context(root, telemetry_context, &mut gaps)?;
    if !comparison.profile_comparison_ready {
        gaps.push(format!("route profile comparison is not ready: {}", comparison.gaps.join("; ")));
    }
    if comparison.claim_boundary.hidden_fallback_allowed {
        gaps.push("benchmark qualification refuses hidden fallback".to_string());
    }
    if comparison.claim_boundary.arc_bitnet_full_inference_claimed
        || comparison.claim_boundary.npu_bitnet_full_inference_claimed
        || comparison.claim_boundary.qk256_accelerator_decode_claimed
    {
        gaps.push("benchmark qualification refuses accelerator BitNet/QK256 claims".to_string());
    }
    if string_at(&phase_comparison_json, "artifact_kind").is_none() {
        gaps.push("phase comparison receipt is missing artifact_kind".to_string());
    }
    if fallback_used(&phase_comparison_json) == Some(true) {
        gaps.push("phase comparison receipt observed fallback_used=true".to_string());
    }

    let profiles = comparison
        .profiles
        .iter()
        .map(|profile| cold_warm_profile_benchmark(profile, telemetry_context.as_ref(), &mut gaps))
        .collect::<Vec<_>>();

    let benchmark_gate_ready = gaps.is_empty();
    Ok(LunarLakeColdWarmBenchmark {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_cold_warm_profile_benchmark".to_string(),
        proof_stage: "profile_timing_qualification_no_promotion_change".to_string(),
        created_utc,
        machine_id: comparison.machine_id,
        artifact_root: path_string(root),
        route_profile_comparison_receipt: path_string(&route_profile_comparison_path),
        phase_comparison_receipt: path_string(&phase_comparison_path),
        benchmark_gate_ready,
        profiles,
        gaps,
        claim_boundary: BenchmarkClaimBoundary {
            new_inference_executed: false,
            route_promotion_changed: false,
            broad_quality_claim: false,
            speedup_claim: false,
            acceleration_claim: false,
            hidden_fallback_allowed: false,
            dense_slm_as_bitnet_proof: false,
        },
    })
}

pub fn build_cpu_slm_phase_attribution_with_created_utc(
    root: &Path,
    cpu_phase: &Path,
    cold_warm_benchmark: &Path,
    phase_comparison: &Path,
    created_utc: String,
) -> Result<LunarLakeCpuSlmPhaseAttribution> {
    let cpu_phase_path = resolve_receipt_path(root, cpu_phase);
    let cold_warm_path = resolve_receipt_path(root, cold_warm_benchmark);
    let phase_comparison_path = resolve_receipt_path(root, phase_comparison);
    let cpu_phase_json: Value = read_json_receipt(&cpu_phase_path)?;
    let cold_warm: LunarLakeColdWarmBenchmark = read_json_receipt(&cold_warm_path)?;
    let phase_comparison_json: Value = read_json_receipt(&phase_comparison_path)?;

    let mut gaps = Vec::new();
    let mut findings = Vec::new();
    if fallback_used(&cpu_phase_json) == Some(true) {
        gaps.push("dense Qwen CPU phase receipt observed fallback_used=true".to_string());
    }
    if !cold_warm.benchmark_gate_ready {
        gaps.push(format!("cold/warm benchmark is not ready: {}", cold_warm.gaps.join("; ")));
    }
    if cold_warm.claim_boundary.new_inference_executed {
        gaps.push("cold/warm benchmark executed new inference".to_string());
    }
    if cold_warm.claim_boundary.route_promotion_changed {
        gaps.push("cold/warm benchmark changed route promotion".to_string());
    }
    if cold_warm.claim_boundary.speedup_claim || cold_warm.claim_boundary.acceleration_claim {
        gaps.push("cold/warm benchmark made speedup or acceleration claim".to_string());
    }
    if cold_warm.claim_boundary.hidden_fallback_allowed {
        gaps.push("cold/warm benchmark allows hidden fallback".to_string());
    }
    if string_at(&phase_comparison_json, "artifact_kind").as_deref()
        != Some("intel_258v_dense_slm_openvino_phase_comparison")
    {
        gaps.push("phase comparison receipt is not the dense SLM OpenVINO comparison".to_string());
    }
    if fallback_used(&phase_comparison_json) == Some(true) {
        gaps.push("phase comparison observed fallback_used=true".to_string());
    }

    let cold_route = find_cpu_cold_route(&cold_warm).with_context(|| {
        format!("{} does not contain dense_slm_default_cpu timing", cold_warm_path.display())
    })?;
    let cold_one_off = cpu_slm_cold_attribution(cold_route.profile_id, cold_route.route)?;
    let warm_session = cpu_slm_warm_attribution(&cpu_phase_json, &mut gaps);
    let openvino_cpu_context = cpu_slm_openvino_cpu_context(&phase_comparison_json);

    if let Some(total) = cold_one_off.timing.total_response_ms {
        findings.push(format!("cpu_one_off_total_response_ms={total:.3}"));
    }
    if let Some(load_share) = cold_one_off.model_load_share_of_total {
        findings.push(format!("cpu_one_off_model_load_share={load_share:.3}"));
    }
    if let Some(prefill_share) = cold_one_off.reported_prefill_share_of_total {
        findings.push(format!("cpu_one_off_prefill_share={prefill_share:.3}"));
    }
    if let Some(profile) =
        warm_session.profiles.iter().find(|profile| profile.profile == "decode_128")
        && let Some(tokens_per_s) = profile.decode_tokens_per_s
    {
        findings.push(format!("warm_decode_128_tokens_per_s={tokens_per_s:.3}"));
    }
    if let Some(context) = &openvino_cpu_context {
        if context.pipeline_load_ms.is_some() || context.case_elapsed_ms_sum.is_some() {
            findings.push("openvino_cpu_smoke_context_indexed_without_speedup_claim".to_string());
        }
    } else {
        gaps.push("OpenVINO CPU comparison context is missing".to_string());
    }

    let recommended_next_items = vec![
        "LNL258V-CPU-SLM-PERF-002: add resident CPU session/no-reload timing".to_string(),
        "LNL258V-CPU-SLM-PERF-003: compare Rust GGUF CPU against OpenVINO CPU for the same Qwen profiles".to_string(),
        "LNL258V-GPU-QUAL-001: keep GPU promotion blocked until corpus-v2 quality failures are classified".to_string(),
        "LNL258V-NPU-COLD-001: decompose NPU cold load separately from hot decode".to_string(),
    ];
    let attribution_ready = gaps.is_empty();

    Ok(LunarLakeCpuSlmPhaseAttribution {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_cpu_slm_phase_attribution".to_string(),
        proof_stage: "cpu_dense_slm_phase_attribution_no_new_inference".to_string(),
        created_utc,
        machine_id: "intel-258v".to_string(),
        artifact_root: path_string(root),
        source_receipts: CpuSlmAttributionSources {
            cpu_phase_receipt: path_string(&cpu_phase_path),
            cold_warm_benchmark_receipt: path_string(&cold_warm_path),
            phase_comparison_receipt: path_string(&phase_comparison_path),
        },
        model: CpuSlmAttributionModel {
            model_family: string_at_any(&cpu_phase_json, &["model_family", "model.family"]),
            model_architecture: string_at_any(
                &cpu_phase_json,
                &["model_architecture", "model.architecture"],
            ),
            quantization: string_at_any(&cpu_phase_json, &["quantization", "model.quant_format"]),
            tokenizer_source: string_at_any(
                &cpu_phase_json,
                &["tokenizer_source", "tokenizer.source"],
            ),
            prompt_template: string_at(&cpu_phase_json, "prompt_template"),
        },
        backend: CpuSlmAttributionBackend {
            route_id: DEFAULT_ASK_ROUTE.to_string(),
            selected_backend: cold_route.route.selected_backend.clone(),
            runtime_api: cold_route.route.runtime_api.clone(),
            selected_kernel_or_runtime: string_at(&cpu_phase_json, "selected_kernel_or_runtime"),
            fallback_used: cold_route.route.fallback_used,
            answer_gate_passed: cold_route.route.answer_gate_passed,
        },
        cold_one_off,
        warm_session,
        openvino_cpu_context,
        attribution_ready,
        findings,
        recommended_next_items,
        gaps,
        claim_boundary: CpuSlmPerfClaimBoundary {
            new_inference_executed: false,
            route_promotion_changed: false,
            broad_quality_claim: false,
            speedup_claim: false,
            power_advantage_claim: false,
            acceleration_claim: false,
            arc_npu_execution_claim: false,
            bitnet_qk256_i2s_claim: false,
            hidden_fallback_allowed: false,
        },
    })
}

struct CpuColdRouteRef<'a> {
    profile_id: &'a str,
    route: &'a ColdWarmRouteBenchmark,
}

fn find_cpu_cold_route(benchmark: &LunarLakeColdWarmBenchmark) -> Option<CpuColdRouteRef<'_>> {
    for wanted in ["ask_short", "ask_normal", "regression_tiny"] {
        if let Some(found) = benchmark.profiles.iter().find_map(|profile| {
            (profile.profile_id == wanted)
                .then(|| {
                    profile.routes.iter().find(|route| route.route_id == DEFAULT_ASK_ROUTE).map(
                        |route| CpuColdRouteRef { profile_id: profile.profile_id.as_str(), route },
                    )
                })
                .flatten()
        }) {
            return Some(found);
        }
    }
    benchmark.profiles.iter().find_map(|profile| {
        profile
            .routes
            .iter()
            .find(|route| route.route_id == DEFAULT_ASK_ROUTE)
            .map(|route| CpuColdRouteRef { profile_id: profile.profile_id.as_str(), route })
    })
}

fn cpu_slm_cold_attribution(
    profile_id: &str,
    route: &ColdWarmRouteBenchmark,
) -> Result<CpuSlmColdAttribution> {
    let timing = route.timing.clone();
    let total = timing.total_response_ms;
    let share = |value: Option<f64>| -> Option<f64> {
        let total = total?;
        let value = value?;
        (total > 0.0).then(|| value / total)
    };
    let non_decode_ms = match (timing.total_response_ms, timing.decode_total_ms) {
        (Some(total), Some(decode)) => Some((total - decode).max(0.0)),
        _ => None,
    };
    let mut timing_notes = Vec::new();
    if timing.prefill_ms.is_some() && timing.first_token_ms.is_some() {
        timing_notes.push(
            "cold one-off receipt reports both prefill_ms and first_token_ms; treat shares as diagnostic attribution, not additive benchmark accounting".to_string(),
        );
    }
    if timing.known_gaps.iter().any(|gap| gap.contains("bounded math ask only")) {
        timing_notes.push("cold one-off attribution is from bounded math ask, not expanded corpus-v2 profile execution".to_string());
    }
    if route.benchmark_qualified_advantage {
        bail!("CPU attribution refuses benchmark-qualified advantage claims");
    }
    Ok(CpuSlmColdAttribution {
        profile_id: profile_id.to_string(),
        timing,
        model_load_share_of_total: share(route.timing.cold_load_ms),
        tokenize_share_of_total: share(route.timing.tokenize_ms),
        first_token_share_of_total: share(route.timing.first_token_ms),
        decode_share_of_total: share(route.timing.decode_total_ms),
        reported_prefill_share_of_total: share(route.timing.prefill_ms),
        non_decode_ms,
        timing_notes,
        blockers: route.blockers.clone(),
    })
}

fn cpu_slm_warm_attribution(json: &Value, gaps: &mut Vec<String>) -> CpuSlmWarmAttribution {
    let profiles = json
        .get("profiles")
        .and_then(Value::as_array)
        .map(|profiles| profiles.iter().map(cpu_slm_warm_profile_attribution).collect::<Vec<_>>())
        .unwrap_or_default();
    if profiles.is_empty() {
        gaps.push("dense Qwen CPU warm phase receipt has no profiles".to_string());
    }
    let mut timing_notes = Vec::new();
    if bool_at_any(json, &["session.model_loaded_once"]) == Some(true) {
        timing_notes
            .push("warm-session receipt loaded the model once across phase profiles".to_string());
    }
    if bool_at_any(json, &["session.tokenizer_loaded_once"]) == Some(true) {
        timing_notes.push(
            "warm-session receipt loaded the tokenizer once across phase profiles".to_string(),
        );
    }
    CpuSlmWarmAttribution {
        model_loaded_once: bool_at_any(json, &["session.model_loaded_once"]),
        tokenizer_loaded_once: bool_at_any(json, &["session.tokenizer_loaded_once"]),
        model_load_ms: number_at_any(json, &["timing.model_load_ms"]),
        tokenizer_load_ms: number_at_any(json, &["timing.tokenizer_load_ms"]),
        total_session_ms: number_at_any(json, &["timing.total_session_ms"]),
        profiles,
        timing_notes,
    }
}

fn cpu_slm_warm_profile_attribution(profile: &Value) -> CpuSlmWarmProfileAttribution {
    let prompt_tokens = u64_at(profile, "prompt_tokens");
    let generated_tokens = u64_at(profile, "generated_tokens");
    let prefill_ms = number_at_any(profile, &["prefill_ms"]);
    let decode_total_ms = number_at_any(profile, &["decode_total_ms"]);
    let prefill_ms_per_prompt_token = match (prefill_ms, prompt_tokens) {
        (Some(ms), Some(tokens)) if tokens > 0 => Some(ms / tokens as f64),
        _ => None,
    };
    let decode_tokens_per_s = match (decode_total_ms, generated_tokens) {
        (Some(ms), Some(tokens)) if ms > 0.0 => Some(tokens as f64 / (ms / 1000.0)),
        _ => None,
    };
    CpuSlmWarmProfileAttribution {
        profile: string_at(profile, "profile").unwrap_or_else(|| "unknown".to_string()),
        prompt_tokens,
        generated_tokens,
        prefill_ms,
        first_token_decode_ms: number_at_any(profile, &["first_token_decode_ms"]),
        decode_total_ms,
        prefill_ms_per_prompt_token,
        decode_tokens_per_s,
        fallback_used: bool_at_any(profile, &["fallback_used"]),
        receipt_path: string_at(profile, "receipt_path"),
    }
}

fn cpu_slm_openvino_cpu_context(json: &Value) -> Option<CpuSlmOpenVinoCpuContext> {
    let cpu = value_at(json, "openvino_paths.cpu")?;
    Some(CpuSlmOpenVinoCpuContext {
        source_receipt: string_at(cpu, "source_receipt"),
        selected_backend: string_at(cpu, "selected_backend"),
        runtime_api: string_at(cpu, "runtime_api"),
        fallback_used: bool_at_any(cpu, &["fallback_used"]),
        answer_gate_passed: bool_at_any(cpu, &["answer_gate.passed"]).or_else(|| {
            let passed = u64_at(cpu, "answer_gate.passed")?;
            let failed = u64_at(cpu, "answer_gate.failed").unwrap_or(0);
            Some(passed > 0 && failed == 0)
        }),
        pipeline_load_ms: number_at_any(cpu, &["timing.pipeline_load_ms"]),
        case_elapsed_ms_sum: number_at_any(cpu, &["timing.case_elapsed_ms_sum"]),
        timing_scope: "openvino_cpu_smoke_level_context_only".to_string(),
        comparison_notes: vec![
            "OpenVINO CPU timing is smoke-level context from existing receipts, not a speedup claim".to_string(),
            "OpenVINO GenAI receipt does not expose tokenize/prefill/first-token/decode splits for this comparison".to_string(),
        ],
    })
}

pub fn build_cpu_slm_resident_session_with_created_utc(
    root: &Path,
    phase_attribution: &Path,
    repeated_warm_session: &Path,
    required_repeats: u64,
    created_utc: String,
) -> Result<LunarLakeCpuSlmResidentSession> {
    let phase_attribution_path = resolve_receipt_path(root, phase_attribution);
    let repeated_warm_session_path = resolve_receipt_path(root, repeated_warm_session);
    let phase_attribution_json: Value = read_json_receipt(&phase_attribution_path)?;
    let repeated_json: Value = read_json_receipt(&repeated_warm_session_path)?;

    let mut gaps = Vec::new();
    if string_at(&phase_attribution_json, "artifact_kind").as_deref()
        != Some("lunar_lake_cpu_slm_phase_attribution")
    {
        gaps.push(
            "phase attribution receipt must have artifact_kind=lunar_lake_cpu_slm_phase_attribution"
                .to_string(),
        );
    }
    if bool_at_any(&phase_attribution_json, &["attribution_ready"]) != Some(true) {
        gaps.push("phase attribution receipt is not attribution_ready=true".to_string());
    }
    if string_at(&repeated_json, "artifact_kind").as_deref() != Some("slm_cpu_warm_session") {
        gaps.push(
            "repeated warm-session receipt must have artifact_kind=slm_cpu_warm_session"
                .to_string(),
        );
    }
    if string_at_any(&repeated_json, &["selected_backend", "backend.selected_backend"]).as_deref()
        != Some("cpu-rust")
    {
        gaps.push("resident session must select backend cpu-rust".to_string());
    }
    if string_at_any(&repeated_json, &["runtime_api", "backend.runtime_api"]).as_deref()
        != Some("cpu")
    {
        gaps.push("resident session must record runtime_api=cpu".to_string());
    }
    if fallback_used(&repeated_json) != Some(false) {
        gaps.push("resident session must record fallback_used=false".to_string());
    }
    if bool_at_any(&repeated_json, &["quality_summary.passed"]) != Some(true) {
        gaps.push("resident session must record passing answer gates".to_string());
    }
    if bool_at_any(&repeated_json, &["determinism.passed"]) != Some(true) {
        gaps.push("resident session must record determinism.passed=true".to_string());
    }
    if bool_at_any(
        &repeated_json,
        &[
            "speedup_claim",
            "claim_boundary.speedup_claim",
            "claim_boundary.broad_performance_claim",
            "claim_boundary.full_metal_inference_claimed",
            "claim_boundary.bitnet_quality_claimed",
        ],
    ) == Some(true)
    {
        gaps.push("resident session refuses speedup, accelerator, or BitNet claims".to_string());
    }

    let resident_session = cpu_slm_resident_session_evidence(&repeated_json);
    if resident_session.model_loaded_once != Some(true) {
        gaps.push("resident session did not prove model_loaded_once=true".to_string());
    }
    if resident_session.tokenizer_loaded_once != Some(true) {
        gaps.push("resident session did not prove tokenizer_loaded_once=true".to_string());
    }

    let cold_reference = cpu_slm_resident_cold_reference(&phase_attribution_json);
    let profiles = cpu_slm_resident_profiles(
        &repeated_json,
        required_repeats,
        cold_reference.total_response_ms,
        &mut gaps,
    );
    if profiles.is_empty() {
        gaps.push("resident session has no repeated profile timing summaries".to_string());
    }

    let mut findings = Vec::new();
    if let Some(total) = cold_reference.total_response_ms {
        findings.push(format!("cold_reference_total_response_ms={total:.3}"));
    }
    if let Some(load) = resident_session.model_load_ms {
        findings.push(format!("resident_session_model_load_ms={load:.3}"));
    }
    for profile in &profiles {
        if let Some(mean) = profile.total_ms.mean {
            findings.push(format!("resident_{}_mean_total_ms={mean:.3}", profile.profile_id));
        }
        if let Some(ratio) = profile.cold_to_resident_total_ratio {
            findings
                .push(format!("cold_to_resident_total_ratio_{}={ratio:.3}", profile.profile_id));
        }
    }

    let recommended_next_items = vec![
        "LNL258V-CPU-SLM-PERF-003: compare Rust GGUF CPU against OpenVINO CPU for the same Qwen profiles".to_string(),
        "LNL258V-GPU-QUAL-001: classify OpenVINO GPU corpus-v2 quality failures before promotion".to_string(),
        "LNL258V-NPU-COLD-001: decompose NPU cold load separately from hot decode".to_string(),
    ];
    let resident_ready = gaps.is_empty()
        && !profiles.is_empty()
        && profiles.iter().all(|profile| profile.blockers.is_empty());

    Ok(LunarLakeCpuSlmResidentSession {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_cpu_slm_resident_session".to_string(),
        proof_stage: "resident_cpu_no_reload_timing_no_new_inference".to_string(),
        created_utc,
        machine_id: "intel-258v".to_string(),
        artifact_root: path_string(root),
        source_receipts: CpuSlmResidentSessionSources {
            phase_attribution_receipt: path_string(&phase_attribution_path),
            repeated_warm_session_receipt: path_string(&repeated_warm_session_path),
        },
        model: CpuSlmAttributionModel {
            model_family: string_at_any(&repeated_json, &["model.family", "corpus.model.family"]),
            model_architecture: string_at_any(
                &repeated_json,
                &["model.architecture", "corpus.model.architecture"],
            ),
            quantization: string_at_any(
                &repeated_json,
                &["model.quant_format", "corpus.model.quant_format"],
            ),
            tokenizer_source: string_at_any(&repeated_json, &["model.tokenizer"]),
            prompt_template: string_at_any(
                &repeated_json,
                &["generation.prompt_template", "corpus.defaults.prompt_template"],
            ),
        },
        backend: CpuSlmAttributionBackend {
            route_id: DEFAULT_ASK_ROUTE.to_string(),
            selected_backend: string_at_any(
                &repeated_json,
                &["selected_backend", "backend.selected_backend"],
            )
            .unwrap_or_else(|| "unknown".to_string()),
            runtime_api: string_at_any(&repeated_json, &["runtime_api", "backend.runtime_api"])
                .unwrap_or_else(|| "unknown".to_string()),
            selected_kernel_or_runtime: Some("resident_cpu_rust_gguf".to_string()),
            fallback_used: fallback_used(&repeated_json),
            answer_gate_passed: bool_at_any(&repeated_json, &["quality_summary.passed"]),
        },
        resident_session,
        cold_reference,
        profiles,
        resident_ready,
        findings,
        recommended_next_items,
        gaps,
        claim_boundary: CpuSlmPerfClaimBoundary {
            new_inference_executed: false,
            route_promotion_changed: false,
            broad_quality_claim: false,
            speedup_claim: false,
            power_advantage_claim: false,
            acceleration_claim: false,
            arc_npu_execution_claim: false,
            bitnet_qk256_i2s_claim: false,
            hidden_fallback_allowed: false,
        },
    })
}

fn cpu_slm_resident_session_evidence(json: &Value) -> CpuSlmResidentSessionEvidence {
    CpuSlmResidentSessionEvidence {
        reuse_scope: string_at(json, "session.reuse_scope"),
        model_loaded_once: bool_at_any(json, &["session.model_loaded_once"]),
        tokenizer_loaded_once: bool_at_any(json, &["session.tokenizer_loaded_once"]),
        model_load_ms: number_at_any(json, &["timing.model_load_ms"]),
        model_sha256_ms: number_at_any(json, &["timing.model_sha256_ms"]),
        tokenizer_load_ms: number_at_any(json, &["timing.tokenizer_load_ms"]),
        total_session_ms: number_at_any(json, &["timing.total_session_ms"]),
        prompt_count: u64_at(json, "session.prompt_count"),
        per_prompt_receipts_enabled: bool_at_any(json, &["session.per_prompt_receipts_enabled"]),
        session_owned_buffers: bool_at_any(json, &["session.session_owned_buffers"]),
        prompt_token_buffer_reused: bool_at_any(json, &["session.prompt_token_buffer_reused"]),
        generated_token_buffer_reused: bool_at_any(
            json,
            &["session.generated_token_buffer_reused"],
        ),
        timing_buffers_reused: bool_at_any(json, &["session.timing_buffers_reused"]),
        stop_policy_precomputed_once: bool_at_any(json, &["session.stop_policy_precomputed_once"]),
        resident_memory_bytes: u64_at(json, "memory.resident_memory_bytes"),
    }
}

fn cpu_slm_resident_cold_reference(json: &Value) -> CpuSlmResidentColdReference {
    CpuSlmResidentColdReference {
        profile_id: string_at(json, "cold_one_off.profile_id"),
        total_response_ms: number_at_any(json, &["cold_one_off.timing.total_response_ms"]),
        cold_load_ms: number_at_any(json, &["cold_one_off.timing.cold_load_ms"]),
        tokenize_ms: number_at_any(json, &["cold_one_off.timing.tokenize_ms"]),
        prefill_ms: number_at_any(json, &["cold_one_off.timing.prefill_ms"]),
        first_token_ms: number_at_any(json, &["cold_one_off.timing.first_token_ms"]),
        decode_total_ms: number_at_any(json, &["cold_one_off.timing.decode_total_ms"]),
        timing_scope: "cold_one_off_reference_from_cpu_phase_attribution".to_string(),
    }
}

#[derive(Default)]
struct ResidentProfileAccumulator {
    case_ids: BTreeSet<String>,
    observed_execution_count: u64,
    model_reload_observed: bool,
    tokenizer_reload_observed: bool,
    fallback_observed: bool,
    answer_gate_seen: bool,
    answer_gate_passed: bool,
    deterministic_generated_ids: Option<bool>,
    deterministic_text: Option<bool>,
    total_ms: Vec<f64>,
    time_to_first_token_ms: Vec<f64>,
    prefill_ms: Vec<f64>,
    decode_total_ms: Vec<f64>,
    tokenize_ms: Vec<f64>,
    generated_tokens: Vec<f64>,
}

fn cpu_slm_resident_profiles(
    json: &Value,
    required_repeats: u64,
    cold_reference_total_ms: Option<f64>,
    gaps: &mut Vec<String>,
) -> Vec<CpuSlmResidentProfileSummary> {
    let mut by_index = BTreeMap::<u64, &Value>::new();
    for prompt in json.get("prompts").and_then(Value::as_array).into_iter().flatten() {
        if let Some(index) = u64_at(prompt, "prompt_index") {
            by_index.insert(index, prompt);
        }
    }
    if by_index.is_empty() {
        gaps.push("resident warm-session receipt has no prompt receipts".to_string());
    }

    let mut profiles = BTreeMap::<String, ResidentProfileAccumulator>::new();
    for group in json.pointer("/determinism/groups").and_then(Value::as_array).into_iter().flatten()
    {
        let Some(case_id) = group.get("case_id").and_then(Value::as_str) else {
            gaps.push("resident determinism group is missing case_id".to_string());
            continue;
        };
        let Some(profile_id) = durability_profile_for_case_id(case_id) else {
            continue;
        };
        let prompt_indices = group
            .get("prompt_indices")
            .and_then(Value::as_array)
            .map(|indices| indices.iter().filter_map(Value::as_u64).collect::<Vec<_>>())
            .unwrap_or_default();
        let entry = profiles.entry(profile_id.to_string()).or_default();
        entry.case_ids.insert(case_id.to_string());
        entry.observed_execution_count =
            entry.observed_execution_count.max(u64_at(group, "attempt_count").unwrap_or(0));
        if !entry.answer_gate_seen {
            entry.answer_gate_passed = true;
            entry.answer_gate_seen = true;
        }
        entry.deterministic_generated_ids = Some(
            entry.deterministic_generated_ids.unwrap_or(true)
                && bool_at_any(group, &["stable_generated_token_ids"]) == Some(true),
        );
        entry.deterministic_text = Some(
            entry.deterministic_text.unwrap_or(true)
                && bool_at_any(group, &["stable_text"]) == Some(true),
        );
        for index in prompt_indices {
            let Some(prompt) = by_index.get(&index) else {
                gaps.push(format!(
                    "resident determinism group {case_id} references missing prompt_index {index}"
                ));
                continue;
            };
            entry.fallback_observed |= fallback_used(prompt) != Some(false);
            entry.answer_gate_passed &= answer_gate_passed(prompt) == Some(true);
            entry.model_reload_observed |=
                number_at_any(prompt, &["timing.model_load_ms"]).is_some_and(|value| value > 0.0);
            entry.tokenizer_reload_observed |= number_at_any(prompt, &["timing.tokenizer_load_ms"])
                .is_some_and(|value| value > 0.0);
            push_number(prompt, "timing.total_ms", &mut entry.total_ms);
            push_first_number(
                prompt,
                &["timing.time_to_first_token_ms", "timing.first_token_ms"],
                &mut entry.time_to_first_token_ms,
            );
            push_number(prompt, "timing.prefill_ms", &mut entry.prefill_ms);
            push_number(prompt, "timing.decode_total_ms", &mut entry.decode_total_ms);
            push_number(prompt, "timing.tokenize_ms", &mut entry.tokenize_ms);
            if let Some(tokens) = u64_at(prompt, "generated_tokens") {
                entry.generated_tokens.push(tokens as f64);
            }
        }
    }

    profiles
        .into_iter()
        .map(|(profile_id, entry)| {
            let mut blockers = Vec::new();
            if entry.observed_execution_count < required_repeats {
                blockers.push(format!(
                    "resident profile observed {}/{} required executions",
                    entry.observed_execution_count, required_repeats
                ));
            }
            if entry.model_reload_observed {
                blockers.push("model reload observed inside resident prompt loop".to_string());
            }
            if entry.tokenizer_reload_observed {
                blockers.push("tokenizer reload observed inside resident prompt loop".to_string());
            }
            if entry.fallback_observed {
                blockers.push("fallback observed inside resident prompt loop".to_string());
            }
            if !entry.answer_gate_passed {
                blockers
                    .push("answer gate failure observed inside resident prompt loop".to_string());
            }
            if entry.deterministic_generated_ids != Some(true) {
                blockers.push("generated token IDs drifted in resident prompt loop".to_string());
            }
            if entry.deterministic_text != Some(true) {
                blockers.push("decoded text drifted in resident prompt loop".to_string());
            }
            if entry.total_ms.is_empty() {
                blockers.push("resident profile has no total_ms timing samples".to_string());
            }
            blockers.sort();
            blockers.dedup();

            let total_ms = resident_metric_summary(&entry.total_ms);
            let decode_total_ms = resident_metric_summary(&entry.decode_total_ms);
            let generated_tokens = resident_metric_summary(&entry.generated_tokens);
            let decode_tokens_per_s_mean =
                match (sum_f64(&entry.generated_tokens), sum_f64(&entry.decode_total_ms)) {
                    (Some(tokens), Some(ms)) if ms > 0.0 => Some(tokens / (ms / 1000.0)),
                    _ => None,
                };
            let cold_to_resident_total_ratio = match (cold_reference_total_ms, total_ms.mean) {
                (Some(cold), Some(warm)) if warm > 0.0 => Some(cold / warm),
                _ => None,
            };

            CpuSlmResidentProfileSummary {
                profile_id,
                case_ids: entry.case_ids.into_iter().collect(),
                observed_execution_count: entry.observed_execution_count,
                required_execution_count: required_repeats,
                model_reload_observed: entry.model_reload_observed,
                tokenizer_reload_observed: entry.tokenizer_reload_observed,
                fallback_observed: entry.fallback_observed,
                answer_gate_passed: entry.answer_gate_passed,
                deterministic_generated_ids: entry.deterministic_generated_ids,
                deterministic_text: entry.deterministic_text,
                total_ms,
                time_to_first_token_ms: resident_metric_summary(&entry.time_to_first_token_ms),
                prefill_ms: resident_metric_summary(&entry.prefill_ms),
                decode_total_ms,
                tokenize_ms: resident_metric_summary(&entry.tokenize_ms),
                generated_tokens,
                decode_tokens_per_s_mean,
                cold_to_resident_total_ratio,
                blockers,
            }
        })
        .collect()
}

fn push_number(json: &Value, path: &str, out: &mut Vec<f64>) {
    if let Some(value) = number_at_any(json, &[path]) {
        out.push(value);
    }
}

fn push_first_number(json: &Value, paths: &[&str], out: &mut Vec<f64>) {
    if let Some(value) = number_at_any(json, paths) {
        out.push(value);
    }
}

fn resident_metric_summary(values: &[f64]) -> CpuSlmResidentMetricSummary {
    let sample_count = values.len() as u64;
    if values.is_empty() {
        return CpuSlmResidentMetricSummary { sample_count, min: None, mean: None, max: None };
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    for value in values {
        min = min.min(*value);
        max = max.max(*value);
        sum += value;
    }
    CpuSlmResidentMetricSummary {
        sample_count,
        min: Some(min),
        mean: Some(sum / values.len() as f64),
        max: Some(max),
    }
}

fn sum_f64(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum())
}

pub fn build_telemetry_context_with_created_utc(
    _root: &Path,
    created_utc: String,
) -> LunarLakeTelemetryContext {
    let memory = collect_telemetry_memory_context();
    let power = collect_telemetry_power_context();
    let thermal = collect_telemetry_thermal_context();

    let memory_context_recorded = memory.total_bytes.is_some() || memory.available_bytes.is_some();
    let power_context_recorded =
        power.active_scheme.as_ref().is_some_and(|value| !value.is_empty())
            || power.battery_status.as_ref().is_some_and(|value| !value.is_empty())
            || power.ac_power_inferred.is_some();
    let thermal_context_recorded =
        thermal.thermal_zones_visible.unwrap_or(0) > 0 || !thermal.temperatures_celsius.is_empty();

    let mut gaps = Vec::new();
    if !memory_context_recorded {
        gaps.push(
            "memory context is not available from the current OS telemetry probe".to_string(),
        );
    }
    if !power_context_recorded {
        gaps.push("power context is not available from the current OS telemetry probe".to_string());
    }
    if !thermal_context_recorded {
        gaps.push(
            "thermal sensor context is not available from the current OS telemetry probe"
                .to_string(),
        );
    }
    gaps.push(
        "power context is recorded for routing evidence, but no speedup or power-advantage claim is made"
            .to_string(),
    );

    let availability = TelemetryAvailability {
        memory_context_recorded,
        power_context_recorded,
        thermal_context_recorded,
    };
    let memory_context = format_memory_context(&memory);
    let power_context = format_power_context(&power);
    let thermal_context = format_thermal_context(&thermal);
    let sources = vec![
        TelemetrySourceStatus {
            source: memory.source.clone(),
            available: memory_context_recorded,
            status: if memory_context_recorded {
                "captured".to_string()
            } else {
                "unavailable".to_string()
            },
        },
        TelemetrySourceStatus {
            source: power.source.clone(),
            available: power_context_recorded,
            status: if power_context_recorded {
                "captured".to_string()
            } else {
                "unavailable".to_string()
            },
        },
        TelemetrySourceStatus {
            source: thermal.source.clone(),
            available: thermal_context_recorded,
            status: if thermal_context_recorded {
                "captured".to_string()
            } else {
                "unavailable".to_string()
            },
        },
    ];

    LunarLakeTelemetryContext {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_power_thermal_context".to_string(),
        proof_stage: "live_telemetry_context_captured_no_promotion_change".to_string(),
        created_utc,
        machine_id: "intel-258v".to_string(),
        telemetry_scope: "current_machine_runtime_telemetry".to_string(),
        memory_context,
        power_context,
        thermal_context,
        availability,
        memory,
        power,
        thermal,
        sources,
        gaps,
        claim_boundary: TelemetryClaimBoundary {
            new_inference_executed: false,
            telemetry_measurement_executed: true,
            route_promotion_changed: false,
            speedup_claim: false,
            power_advantage_claim: false,
            acceleration_claim: false,
            hidden_fallback_allowed: false,
        },
    }
}

fn collect_telemetry_memory_context() -> TelemetryMemoryContext {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let total_bytes = nonzero_u64(system.total_memory());
    let available_bytes = nonzero_u64(system.available_memory());
    let used_bytes = match (total_bytes, available_bytes) {
        (Some(total), Some(available)) => Some(total.saturating_sub(available)),
        _ => nonzero_u64(system.used_memory()),
    };
    TelemetryMemoryContext {
        source: "sysinfo".to_string(),
        total_bytes,
        available_bytes,
        used_bytes,
    }
}

fn collect_telemetry_power_context() -> TelemetryPowerContext {
    let active_scheme = platform_power_mode();
    let battery_status = platform_battery_status();
    let ac_power_inferred = battery_status.as_deref().and_then(infer_ac_power_from_battery_status);
    TelemetryPowerContext {
        source: "os_power_probe".to_string(),
        active_scheme,
        battery_status,
        ac_power_inferred,
    }
}

fn collect_telemetry_thermal_context() -> TelemetryThermalContext {
    #[cfg(target_os = "windows")]
    {
        if let Some(temperatures_celsius) = windows_thermal_temperatures_celsius()
            && !temperatures_celsius.is_empty()
        {
            return TelemetryThermalContext {
                source: "windows_msa_cpi_thermal_zone".to_string(),
                thermal_zones_visible: Some(temperatures_celsius.len() as u64),
                temperatures_celsius,
            };
        }
        TelemetryThermalContext {
            source: "windows_msa_cpi_thermal_zone".to_string(),
            thermal_zones_visible: None,
            temperatures_celsius: Vec::new(),
        }
    }

    #[cfg(target_os = "linux")]
    {
        let temperatures = linux_thermal_temperatures_celsius();
        if !temperatures.is_empty() {
            return TelemetryThermalContext {
                source: "linux_sysfs_thermal".to_string(),
                thermal_zones_visible: Some(temperatures.len() as u64),
                temperatures_celsius: temperatures,
            };
        }
        let visible = fs::read_dir("/sys/class/thermal").ok().map(|entries| {
            entries
                .flatten()
                .filter(|entry| entry.file_name().to_string_lossy().starts_with("thermal_zone"))
                .count() as u64
        });
        return TelemetryThermalContext {
            source: "linux_sysfs_thermal".to_string(),
            thermal_zones_visible: visible,
            temperatures_celsius: Vec::new(),
        };
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        TelemetryThermalContext {
            source: "thermal_probe_unavailable".to_string(),
            thermal_zones_visible: None,
            temperatures_celsius: Vec::new(),
        }
    }
}

fn format_memory_context(memory: &TelemetryMemoryContext) -> String {
    match (memory.total_bytes, memory.available_bytes) {
        (Some(total), Some(available)) => {
            let used = memory.used_bytes.unwrap_or_else(|| total.saturating_sub(available));
            format!(
                "source={};total_bytes={total};available_bytes={available};used_bytes={used}",
                memory.source
            )
        }
        (Some(total), None) => {
            format!("source={};total_bytes={total};available_bytes=unavailable", memory.source)
        }
        _ => "memory_context_unavailable".to_string(),
    }
}

fn format_power_context(power: &TelemetryPowerContext) -> String {
    if power.active_scheme.is_none()
        && power.battery_status.is_none()
        && power.ac_power_inferred.is_none()
    {
        return "power_context_unavailable".to_string();
    }
    let active_scheme = power.active_scheme.as_deref().unwrap_or("unavailable");
    let battery_status = power.battery_status.as_deref().unwrap_or("unavailable");
    let ac_power = power
        .ac_power_inferred
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unavailable".to_string());
    format!(
        "source={};active_scheme={active_scheme};battery_status={battery_status};ac_power_inferred={ac_power}",
        power.source
    )
}

fn format_thermal_context(thermal: &TelemetryThermalContext) -> String {
    if !thermal.temperatures_celsius.is_empty() {
        let values = thermal
            .temperatures_celsius
            .iter()
            .map(|value| format!("{value:.1}"))
            .collect::<Vec<_>>()
            .join(",");
        return format!("source={};temperatures_celsius={values}", thermal.source);
    }
    match thermal.thermal_zones_visible {
        Some(count) if count > 0 => {
            format!(
                "source={};thermal_zones_visible={count};temperatures_celsius=unavailable",
                thermal.source
            )
        }
        _ => "thermal_context_unavailable".to_string(),
    }
}

fn nonzero_u64(value: u64) -> Option<u64> {
    (value > 0).then_some(value)
}

#[cfg(target_os = "windows")]
fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    Command::new(command)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(target_os = "windows")]
fn platform_power_mode() -> Option<String> {
    command_stdout("powercfg", &["/GETACTIVESCHEME"])
}

#[cfg(target_os = "linux")]
fn platform_power_mode() -> Option<String> {
    let governors = fs::read_dir("/sys/devices/system/cpu")
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let path = entry.path().join("cpufreq/scaling_governor");
            fs::read_to_string(path).ok().map(|value| value.trim().to_string())
        })
        .filter(|value| !value.is_empty())
        .collect::<BTreeSet<_>>();
    (!governors.is_empty()).then(|| governors.into_iter().collect::<Vec<_>>().join(","))
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn platform_power_mode() -> Option<String> {
    None
}

#[cfg(target_os = "windows")]
fn platform_battery_status() -> Option<String> {
    command_stdout(
        "powershell",
        &[
            "-NoProfile",
            "-Command",
            "$b = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1; if ($null -eq $b) { '' } else { \"BatteryStatus=$($b.BatteryStatus);EstimatedChargeRemaining=$($b.EstimatedChargeRemaining)\" }",
        ],
    )
}

#[cfg(target_os = "linux")]
fn platform_battery_status() -> Option<String> {
    let supplies = fs::read_dir("/sys/class/power_supply").ok()?;
    for entry in supplies.flatten() {
        let status_path = entry.path().join("status");
        if let Ok(value) = fs::read_to_string(status_path) {
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn platform_battery_status() -> Option<String> {
    None
}

fn infer_ac_power_from_battery_status(status: &str) -> Option<bool> {
    let lower = status.to_ascii_lowercase();
    if lower.contains("charging")
        || lower.contains("full")
        || lower.contains("ac")
        || lower.contains("batterystatus=2")
        || lower.contains("batterystatus=6")
        || lower.contains("batterystatus=7")
        || lower.contains("batterystatus=8")
        || lower.contains("batterystatus=9")
        || lower.contains("batterystatus=11")
    {
        return Some(true);
    }
    if lower.contains("discharging")
        || lower.contains("batterystatus=1")
        || lower.contains("batterystatus=4")
        || lower.contains("batterystatus=5")
    {
        return Some(false);
    }
    None
}

#[cfg(target_os = "windows")]
fn windows_thermal_temperatures_celsius() -> Option<Vec<f64>> {
    let json = command_stdout(
        "powershell",
        &[
            "-NoProfile",
            "-Command",
            "Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction SilentlyContinue | Select-Object -ExpandProperty CurrentTemperature | ConvertTo-Json -Compress",
        ],
    )?;
    let value: Value = serde_json::from_str(&json).ok()?;
    let raw_values = match value {
        Value::Number(number) => number.as_f64().into_iter().collect::<Vec<_>>(),
        Value::Array(values) => values.into_iter().filter_map(|value| value.as_f64()).collect(),
        _ => Vec::new(),
    };
    let temperatures = raw_values
        .into_iter()
        .filter_map(|value| {
            let celsius = (value / 10.0) - 273.15;
            celsius.is_finite().then_some(celsius)
        })
        .filter(|value| *value > -50.0 && *value < 150.0)
        .collect::<Vec<_>>();
    (!temperatures.is_empty()).then_some(temperatures)
}

#[cfg(target_os = "linux")]
fn linux_thermal_temperatures_celsius() -> Vec<f64> {
    fs::read_dir("/sys/class/thermal")
        .ok()
        .into_iter()
        .flat_map(|entries| entries.flatten())
        .filter_map(|entry| fs::read_to_string(entry.path().join("temp")).ok())
        .filter_map(|value| value.trim().parse::<f64>().ok())
        .map(|value| value / 1000.0)
        .filter(|value| value.is_finite() && *value > -50.0 && *value < 150.0)
        .collect()
}

fn cold_warm_profile_benchmark(
    profile: &WorkloadProfileEvaluation,
    telemetry_context: Option<&BenchmarkTelemetryContext>,
    global_gaps: &mut Vec<String>,
) -> ColdWarmProfileBenchmark {
    let mut profile_gaps = Vec::new();
    let routes = profile
        .route_evidence
        .iter()
        .map(|route| {
            cold_warm_route_benchmark(
                profile,
                route,
                telemetry_context,
                global_gaps,
                &mut profile_gaps,
            )
        })
        .collect::<Vec<_>>();
    if profile.promoted_route.is_none() && routes.iter().all(|route| route.promotion_blocked) {
        profile_gaps.push(format!(
            "{} has no benchmark-qualified promoted route; candidate evidence remains indexed only",
            profile.profile_id
        ));
    }
    ColdWarmProfileBenchmark {
        profile_id: profile.profile_id.clone(),
        promoted_route: profile.promoted_route.clone(),
        candidate_routes: profile.candidate_routes.clone(),
        routes,
        profile_gaps,
    }
}

fn cold_warm_route_benchmark(
    profile: &WorkloadProfileEvaluation,
    route: &ProfileRouteEvidence,
    telemetry_context: Option<&BenchmarkTelemetryContext>,
    global_gaps: &mut Vec<String>,
    profile_gaps: &mut Vec<String>,
) -> ColdWarmRouteBenchmark {
    let mut blockers = route
        .blockers
        .iter()
        .filter_map(|blocker| {
            if telemetry_context.is_some()
                && blocker == "power telemetry receipt missing for low_power promotion"
            {
                None
            } else {
                Some(blocker.clone())
            }
        })
        .collect::<Vec<_>>();
    blockers.extend(route.timing.known_gaps.iter().filter_map(|gap| {
        if telemetry_context.is_some()
            && gap == "power and thermal context not normalized in this comparison"
        {
            None
        } else {
            Some(gap.clone())
        }
    }));

    let timing_required = route.route_id != "bitnet_reference_cpu";
    let critical_timing_present = !timing_required
        || (route.timing.cold_load_ms.is_some()
            && route.timing.first_token_ms.is_some()
            && route.timing.decode_total_ms.is_some()
            && route.timing.throughput_tokens_per_s.is_some());
    if !critical_timing_present {
        blockers.push("cold/warm critical timing is incomplete".to_string());
    }
    if timing_required && route.timing.total_response_ms.is_none() {
        blockers.push("total response latency is missing".to_string());
    }
    if !timing_required {
        blockers.push(
            "BitNet route uses separate CPU reference and I2_S performance receipts".to_string(),
        );
    }
    let telemetry = benchmark_telemetry_for_route(profile, telemetry_context);
    if profile.profile_id == "low_power" && !power_context_is_promotion_evidence(&telemetry) {
        blockers.push(if telemetry.telemetry_receipt.is_some() {
            "power telemetry receipt does not provide low_power promotion evidence".to_string()
        } else {
            "power telemetry receipt missing for low_power promotion".to_string()
        });
    }
    blockers.sort();
    blockers.dedup();

    if route.fallback_used == Some(true) {
        global_gaps.push(format!(
            "{} route {} observed fallback_used=true",
            profile.profile_id, route.route_id
        ));
    }
    if route.route_status == "promoted" && !critical_timing_present {
        global_gaps.push(format!(
            "{} promoted route {} is missing critical cold/warm timing",
            profile.profile_id, route.route_id
        ));
    }
    if route.route_status != "promoted" && route.benchmark_qualified_advantage {
        global_gaps.push(format!(
            "{} candidate route {} claims benchmark-qualified advantage outside route promotion",
            profile.profile_id, route.route_id
        ));
    }
    if route.route_status != "promoted" && !blockers.is_empty() {
        profile_gaps.push(format!(
            "{} route {} remains blocked: {}",
            profile.profile_id,
            route.route_id,
            blockers.join("; ")
        ));
    }

    let benchmark_qualified_advantage =
        route.benchmark_qualified_advantage && critical_timing_present && blockers.is_empty();
    let promotion_blocked = route.route_status != "promoted" && !benchmark_qualified_advantage;
    ColdWarmRouteBenchmark {
        route_id: route.route_id.clone(),
        route_status: route.route_status.clone(),
        selected_backend: route.selected_backend.clone(),
        runtime_api: route.runtime_api.clone(),
        fallback_used: route.fallback_used,
        answer_gate_passed: route.answer_gate_passed,
        phase_timing_present: route.phase_timing_present,
        timing: route.timing.clone(),
        telemetry,
        critical_timing_present,
        benchmark_qualified_advantage,
        promotion_blocked,
        blockers,
    }
}

#[derive(Debug, Clone, PartialEq)]
struct BenchmarkTelemetryContext {
    receipt: String,
    memory_context: String,
    power_context: String,
    thermal_context: String,
    telemetry_gaps: Vec<String>,
}

fn load_benchmark_telemetry_context(
    root: &Path,
    telemetry_context: Option<&Path>,
    global_gaps: &mut Vec<String>,
) -> Result<Option<BenchmarkTelemetryContext>> {
    let Some(path) = telemetry_context else {
        return Ok(None);
    };
    let telemetry_path = resolve_receipt_path(root, path);
    let telemetry: Value = read_json_receipt(&telemetry_path)?;
    match string_at(&telemetry, "artifact_kind").as_deref() {
        Some("lunar_lake_power_thermal_context") => {}
        Some(other) => global_gaps
            .push(format!("power/thermal context receipt has unexpected artifact_kind `{other}`")),
        None => {
            global_gaps.push("power/thermal context receipt is missing artifact_kind".to_string())
        }
    }
    if bool_at_any(&telemetry, &["claim_boundary.route_promotion_changed"]).unwrap_or(false) {
        global_gaps.push("power/thermal context receipt changed route promotion".to_string());
    }
    if bool_at_any(&telemetry, &["claim_boundary.speedup_claim"]).unwrap_or(false) {
        global_gaps.push("power/thermal context receipt claims speedup".to_string());
    }
    if bool_at_any(&telemetry, &["claim_boundary.power_advantage_claim"]).unwrap_or(false) {
        global_gaps.push("power/thermal context receipt claims power advantage".to_string());
    }
    if bool_at_any(&telemetry, &["claim_boundary.acceleration_claim"]).unwrap_or(false) {
        global_gaps.push("power/thermal context receipt claims acceleration".to_string());
    }
    Ok(Some(BenchmarkTelemetryContext {
        receipt: path_string(&telemetry_path),
        memory_context: string_at(&telemetry, "memory_context")
            .unwrap_or_else(|| "memory_context_not_recorded".to_string()),
        power_context: string_at(&telemetry, "power_context")
            .unwrap_or_else(|| "power_context_not_recorded".to_string()),
        thermal_context: string_at(&telemetry, "thermal_context")
            .unwrap_or_else(|| "thermal_context_not_recorded".to_string()),
        telemetry_gaps: string_array_at(&telemetry, "gaps"),
    }))
}

fn benchmark_telemetry_for_route(
    profile: &WorkloadProfileEvaluation,
    telemetry_context: Option<&BenchmarkTelemetryContext>,
) -> BenchmarkTelemetry {
    if let Some(context) = telemetry_context {
        return BenchmarkTelemetry {
            telemetry_receipt: Some(context.receipt.clone()),
            memory_context: context.memory_context.clone(),
            power_context: context.power_context.clone(),
            thermal_context: context.thermal_context.clone(),
            telemetry_gaps: context.telemetry_gaps.clone(),
        };
    }
    BenchmarkTelemetry {
        telemetry_receipt: None,
        memory_context: "not_normalized_in_current_profile_benchmark".to_string(),
        power_context: if profile.profile_id == "low_power" {
            "required_for_promotion_but_not_recorded".to_string()
        } else {
            "not_normalized_in_current_profile_benchmark".to_string()
        },
        thermal_context: "not_normalized_in_current_profile_benchmark".to_string(),
        telemetry_gaps: Vec::new(),
    }
}

fn telemetry_for_profile_route(context: &BenchmarkTelemetryContext) -> BenchmarkTelemetry {
    BenchmarkTelemetry {
        telemetry_receipt: Some(context.receipt.clone()),
        memory_context: context.memory_context.clone(),
        power_context: context.power_context.clone(),
        thermal_context: context.thermal_context.clone(),
        telemetry_gaps: context.telemetry_gaps.clone(),
    }
}

fn power_context_is_recorded(context: &BenchmarkTelemetryContext) -> bool {
    let value = context.power_context.to_ascii_lowercase();
    !(value.contains("not_recorded")
        || value.contains("not_normalized")
        || value.contains("missing")
        || value.contains("unavailable")
        || value.contains("required_for_promotion_but_not_recorded"))
}

fn power_context_is_promotion_evidence(telemetry: &BenchmarkTelemetry) -> bool {
    let value = telemetry.power_context.to_ascii_lowercase();
    !(value.contains("not_recorded")
        || value.contains("not_normalized")
        || value.contains("missing")
        || value.contains("unavailable")
        || value.contains("required_for_promotion_but_not_recorded"))
}

pub fn build_durability_bundle_with_created_utc(
    root: &Path,
    route_profile_comparison: &Path,
    cold_warm_benchmark: &Path,
    cpu_corpus_v2: &Path,
    regression_bundle: &Path,
    repeated_warm_session: Option<&Path>,
    required_repeat_count: u64,
    created_utc: String,
) -> Result<LunarLakeDurabilityBundle> {
    let route_profile_comparison_path = resolve_receipt_path(root, route_profile_comparison);
    let cold_warm_benchmark_path = resolve_receipt_path(root, cold_warm_benchmark);
    let cpu_corpus_v2_path = resolve_receipt_path(root, cpu_corpus_v2);
    let regression_bundle_path = resolve_receipt_path(root, regression_bundle);
    let repeated_warm_session_path =
        repeated_warm_session.map(|path| resolve_receipt_path(root, path));

    let comparison: LunarLakeRouteProfileComparison =
        read_json_receipt(&route_profile_comparison_path)?;
    let benchmark: LunarLakeColdWarmBenchmark = read_json_receipt(&cold_warm_benchmark_path)?;
    let corpus: Value = read_json_receipt(&cpu_corpus_v2_path)?;
    let regression: LunarLakeRegressionBundle = read_json_receipt(&regression_bundle_path)?;
    let repeated_warm_session_json = repeated_warm_session_path
        .as_ref()
        .map(|path| read_json_receipt::<Value>(path))
        .transpose()?;

    let mut gaps = Vec::new();
    if !comparison.profile_comparison_ready {
        gaps.push(format!("route profile comparison is not ready: {}", comparison.gaps.join("; ")));
    }
    if !benchmark.benchmark_gate_ready {
        gaps.push(format!("cold/warm benchmark is not ready: {}", benchmark.gaps.join("; ")));
    }
    if !regression.regression_passed || !regression.regression_surface.strict_ready {
        gaps.push("strict regression-v2 bundle is not ready".to_string());
    }
    if comparison.claim_boundary.hidden_fallback_allowed
        || benchmark.claim_boundary.hidden_fallback_allowed
        || regression.claim_boundary.hidden_fallback_allowed
    {
        gaps.push("durability index refuses hidden fallback".to_string());
    }
    if benchmark.claim_boundary.new_inference_executed {
        gaps.push(
            "durability index refuses benchmark receipts that executed new inference".to_string(),
        );
    }
    if benchmark.claim_boundary.route_promotion_changed {
        gaps.push("durability index refuses route-promotion changes".to_string());
    }
    if benchmark.claim_boundary.speedup_claim || benchmark.claim_boundary.acceleration_claim {
        gaps.push("durability index refuses speedup or acceleration claims".to_string());
    }
    if benchmark.claim_boundary.dense_slm_as_bitnet_proof {
        gaps.push("durability index refuses dense SLM evidence as BitNet proof".to_string());
    }
    if fallback_used(&corpus) == Some(true) {
        gaps.push("CPU corpus-v2 receipt observed fallback_used=true".to_string());
    }
    let repeated_profile_evidence = repeated_warm_session_json
        .as_ref()
        .map(|receipt| repeated_warm_session_profile_evidence(receipt, &mut gaps))
        .unwrap_or_default();

    let corpus_profiles = corpus_profile_counts(&corpus);
    let benchmark_profiles = benchmark
        .profiles
        .iter()
        .map(|profile| (profile.profile_id.as_str(), profile))
        .collect::<BTreeMap<_, _>>();
    let mut next_required_evidence = Vec::new();
    let mut profiles = Vec::new();

    for profile_id in DURABILITY_REQUIRED_PROFILES {
        let Some(profile) =
            comparison.profiles.iter().find(|profile| profile.profile_id == *profile_id)
        else {
            gaps.push(format!("durability profile {profile_id} is missing from route comparison"));
            continue;
        };
        let counts = corpus_profiles.get(*profile_id).cloned().unwrap_or_default();
        let route = profile.route_evidence.iter().find(|route| route.route_id == DEFAULT_ASK_ROUTE);
        let benchmark_route = benchmark_profiles.get(profile_id).and_then(|profile| {
            profile.routes.iter().find(|route| route.route_id == DEFAULT_ASK_ROUTE)
        });
        let Some(route) = route else {
            gaps.push(format!(
                "durability profile {profile_id} is missing dense Qwen CPU route evidence"
            ));
            continue;
        };

        let repeated_evidence = repeated_profile_evidence.get(*profile_id);
        let observed_execution_count = repeated_evidence
            .map(|evidence| evidence.observed_execution_count)
            .unwrap_or(if counts.total > 0 { 1 } else { 0 });
        let mut blockers = route.blockers.clone();
        if counts.total == 0 {
            blockers.push("no CPU corpus-v2 baseline cases for profile".to_string());
        }
        if counts.failed > 0 {
            blockers.push(format!("CPU corpus-v2 profile has {} quality failures", counts.failed));
        }
        let repeated_fallback_observed =
            repeated_evidence.map(|evidence| evidence.fallback_drift_detected).unwrap_or(false);
        let fallback_observed = route.fallback_used == Some(true)
            || counts.fallback_observed
            || repeated_fallback_observed;
        if fallback_observed {
            blockers.push("fallback_used=true observed in indexed profile evidence".to_string());
        }
        if let Some(evidence) = repeated_evidence {
            blockers.extend(evidence.blockers.iter().cloned());
            if evidence.answer_drift_detected {
                blockers
                    .push("answer drift detected in repeated warm-session evidence".to_string());
            }
            if !evidence.quality_passed {
                blockers.push(
                    "answer gate failure detected in repeated warm-session evidence".to_string(),
                );
            }
        }
        if observed_execution_count < required_repeat_count {
            blockers.push(format!(
                "repeated-run evidence missing: observed {observed_execution_count}/{required_repeat_count} executions"
            ));
            next_required_evidence.push(format!(
                "run {profile_id} on {DEFAULT_ASK_ROUTE} {required_repeat_count} times and record answer, route, fallback, and latency variance"
            ));
        }
        if benchmark_route.map(|route| !route.critical_timing_present).unwrap_or(true) {
            blockers.push("critical cold/warm timing missing for durability profile".to_string());
        }
        blockers.sort();
        blockers.dedup();

        let stability_status = if blockers.iter().any(|blocker| blocker.contains("repeated-run")) {
            "awaiting_repeated_run_evidence"
        } else if blockers.is_empty() {
            "stable"
        } else {
            "blocked"
        };

        profiles.push(DurabilityProfileSummary {
            profile_id: (*profile_id).to_string(),
            route_id: DEFAULT_ASK_ROUTE.to_string(),
            route_status: route.route_status.clone(),
            promoted_route: profile.promoted_route.clone(),
            baseline_case_count: counts.total,
            baseline_cases_passed: counts.passed,
            baseline_cases_failed: counts.failed,
            observed_execution_count,
            required_execution_count: required_repeat_count,
            answer_drift_detected: repeated_evidence
                .map(|evidence| evidence.answer_drift_detected)
                .or(if observed_execution_count >= 2 { Some(false) } else { None }),
            route_drift_detected: profile.promoted_route.as_deref() != Some(DEFAULT_ASK_ROUTE),
            fallback_drift_detected: Some(fallback_observed),
            latency_variance_status: repeated_evidence
                .map(RepeatedWarmSessionProfileEvidence::latency_variance_status)
                .unwrap_or_else(|| {
                    if observed_execution_count >= 2 {
                        "variance_window_available".to_string()
                    } else {
                        "not_evaluated_single_execution".to_string()
                    }
                }),
            stability_status: stability_status.to_string(),
            blockers,
        });
    }

    next_required_evidence.sort();
    next_required_evidence.dedup();

    let stability_proven = !profiles.is_empty()
        && profiles.iter().all(|profile| {
            profile.observed_execution_count >= profile.required_execution_count
                && profile.baseline_cases_failed == 0
                && profile.answer_drift_detected == Some(false)
                && !profile.route_drift_detected
                && profile.fallback_drift_detected == Some(false)
                && profile.blockers.is_empty()
        });
    if !stability_proven {
        next_required_evidence.push(
            "collect repeated-run receipts before promoting durability or latency-variance claims"
                .to_string(),
        );
        next_required_evidence.sort();
        next_required_evidence.dedup();
    }

    let durability_index_ready = gaps.is_empty();
    Ok(LunarLakeDurabilityBundle {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_durability_bundle".to_string(),
        proof_stage: "repeated_run_requirements_indexed_no_new_inference".to_string(),
        created_utc,
        machine_id: comparison.machine_id,
        artifact_root: path_string(root),
        route_profile_comparison_receipt: path_string(&route_profile_comparison_path),
        cold_warm_benchmark_receipt: path_string(&cold_warm_benchmark_path),
        cpu_corpus_v2_receipt: path_string(&cpu_corpus_v2_path),
        regression_bundle_receipt: path_string(&regression_bundle_path),
        repeated_warm_session_receipt: repeated_warm_session_path
            .as_ref()
            .map(|path| path_string(path)),
        required_repeat_count,
        durability_index_ready,
        stability_proven,
        profiles,
        gaps,
        next_required_evidence,
        claim_boundary: DurabilityClaimBoundary {
            new_inference_executed: false,
            route_promotion_changed: false,
            broad_quality_claim: false,
            speedup_claim: false,
            acceleration_claim: false,
            hidden_fallback_allowed: false,
            dense_slm_as_bitnet_proof: false,
            repeated_run_stability_claim: stability_proven,
        },
    })
}

#[derive(Default, Clone)]
struct CorpusProfileCounts {
    total: u64,
    passed: u64,
    failed: u64,
    fallback_observed: bool,
}

fn corpus_profile_counts(corpus: &Value) -> BTreeMap<String, CorpusProfileCounts> {
    let mut counts = BTreeMap::<String, CorpusProfileCounts>::new();
    let top_level_fallback = fallback_used(corpus) == Some(true);
    for case in corpus.get("cases").and_then(Value::as_array).into_iter().flatten() {
        let Some(profile) = case.get("profile").and_then(Value::as_str) else {
            continue;
        };
        let entry = counts.entry(profile.to_string()).or_default();
        entry.total += 1;
        if case.get("status").and_then(Value::as_str) == Some("passed") {
            entry.passed += 1;
        } else {
            entry.failed += 1;
        }
        entry.fallback_observed |= top_level_fallback || fallback_used(case) == Some(true);
    }
    counts
}

#[derive(Default, Clone)]
struct RepeatedWarmSessionProfileEvidence {
    observed_execution_count: u64,
    groups_seen: u64,
    answer_drift_detected: bool,
    fallback_drift_detected: bool,
    quality_passed: bool,
    timing_sample_count: usize,
    blockers: Vec<String>,
}

impl RepeatedWarmSessionProfileEvidence {
    fn merge_group(&mut self, group: RepeatedWarmSessionGroupEvidence) {
        if self.groups_seen == 0 {
            self.observed_execution_count = group.attempt_count;
            self.quality_passed = group.quality_passed;
        } else {
            self.observed_execution_count = self.observed_execution_count.min(group.attempt_count);
            self.quality_passed &= group.quality_passed;
        }
        self.groups_seen += 1;
        self.answer_drift_detected |= group.answer_drift_detected;
        self.fallback_drift_detected |= group.fallback_drift_detected;
        self.timing_sample_count += group.timing_sample_count;
        self.blockers.extend(group.blockers);
        self.blockers.sort();
        self.blockers.dedup();
    }

    fn latency_variance_status(&self) -> String {
        if self.timing_sample_count >= 2 {
            "variance_window_available".to_string()
        } else {
            "not_evaluated_missing_timing_samples".to_string()
        }
    }
}

struct RepeatedWarmSessionGroupEvidence {
    attempt_count: u64,
    answer_drift_detected: bool,
    fallback_drift_detected: bool,
    quality_passed: bool,
    timing_sample_count: usize,
    blockers: Vec<String>,
}

fn repeated_warm_session_profile_evidence(
    receipt: &Value,
    gaps: &mut Vec<String>,
) -> BTreeMap<String, RepeatedWarmSessionProfileEvidence> {
    if string_at(receipt, "artifact_kind").as_deref() != Some("slm_cpu_warm_session") {
        gaps.push(
            "repeated warm-session receipt must have artifact_kind=slm_cpu_warm_session"
                .to_string(),
        );
    }
    if string_at_any(receipt, &["selected_backend", "backend.selected_backend"]).as_deref()
        != Some("cpu-rust")
    {
        gaps.push("repeated warm-session receipt must select backend cpu-rust".to_string());
    }
    if string_at_any(receipt, &["runtime_api", "backend.runtime_api"]).as_deref() != Some("cpu") {
        gaps.push("repeated warm-session receipt must record runtime_api=cpu".to_string());
    }
    if fallback_used(receipt) != Some(false) {
        gaps.push("repeated warm-session receipt must record fallback_used=false".to_string());
    }
    if bool_at_any(receipt, &["quality_summary.passed"]) != Some(true) {
        gaps.push("repeated warm-session receipt must record passing quality gates".to_string());
    }
    if bool_at_any(receipt, &["determinism.passed"]) != Some(true) {
        gaps.push("repeated warm-session receipt must record determinism.passed=true".to_string());
    }
    if bool_at_any(
        receipt,
        &[
            "speedup_claim",
            "claim_boundary.speedup_claim",
            "claim_boundary.broad_performance_claim",
            "claim_boundary.full_metal_inference_claimed",
            "claim_boundary.bitnet_quality_claimed",
        ],
    ) == Some(true)
    {
        gaps.push(
            "durability index refuses speedup, accelerator, or BitNet claims from repeated receipt"
                .to_string(),
        );
    }

    let prompt_by_index = receipt
        .get("prompts")
        .and_then(Value::as_array)
        .map(|prompts| {
            prompts
                .iter()
                .filter_map(|prompt| {
                    let index = u64_at(prompt, "prompt_index")?;
                    Some((index, prompt))
                })
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();

    let mut profiles = BTreeMap::<String, RepeatedWarmSessionProfileEvidence>::new();
    let groups = receipt.pointer("/determinism/groups").and_then(Value::as_array);
    let Some(groups) = groups else {
        gaps.push("repeated warm-session receipt has no determinism.groups".to_string());
        return profiles;
    };

    for group in groups {
        let Some(case_id) = group.get("case_id").and_then(Value::as_str) else {
            gaps.push("repeated warm-session determinism group is missing case_id".to_string());
            continue;
        };
        let Some(profile_id) = durability_profile_for_case_id(case_id) else {
            continue;
        };
        let evidence = repeated_warm_session_group_evidence(group, &prompt_by_index);
        profiles.entry(profile_id.to_string()).or_default().merge_group(evidence);
    }

    profiles
}

fn repeated_warm_session_group_evidence(
    group: &Value,
    prompt_by_index: &BTreeMap<u64, &Value>,
) -> RepeatedWarmSessionGroupEvidence {
    let case_id = group.get("case_id").and_then(Value::as_str).unwrap_or("unknown_case");
    let attempt_count = u64_at(group, "attempt_count").unwrap_or(0);
    let stable_ids = bool_at_any(group, &["stable_generated_token_ids"]) == Some(true);
    let stable_text = bool_at_any(group, &["stable_text"]) == Some(true);
    let mut blockers = Vec::new();
    if attempt_count == 0 {
        blockers.push(format!("repeated warm-session group {case_id} is missing attempt_count"));
    }
    if !stable_ids {
        blockers.push(format!("repeated warm-session group {case_id} generated token IDs drifted"));
    }
    if !stable_text {
        blockers.push(format!("repeated warm-session group {case_id} decoded text drifted"));
    }

    let prompt_indices = group
        .get("prompt_indices")
        .and_then(Value::as_array)
        .map(|indices| indices.iter().filter_map(Value::as_u64).collect::<Vec<_>>())
        .unwrap_or_default();
    if prompt_indices.len() < attempt_count as usize {
        blockers.push(format!(
            "repeated warm-session group {case_id} has {}/{} prompt receipts",
            prompt_indices.len(),
            attempt_count
        ));
    }

    let mut quality_passed = true;
    let mut fallback_drift_detected = false;
    let mut timing_sample_count = 0usize;
    for index in prompt_indices {
        let Some(prompt) = prompt_by_index.get(&index) else {
            blockers.push(format!(
                "repeated warm-session group {case_id} references missing prompt_index {index}"
            ));
            quality_passed = false;
            continue;
        };
        if fallback_used(prompt) != Some(false) {
            fallback_drift_detected = true;
        }
        if answer_gate_passed(prompt) != Some(true) {
            quality_passed = false;
        }
        if durability_prompt_has_timing(prompt) {
            timing_sample_count += 1;
        }
    }
    if !quality_passed {
        blockers.push(format!("repeated warm-session group {case_id} has answer-gate failures"));
    }
    if fallback_drift_detected {
        blockers.push(format!("repeated warm-session group {case_id} observed fallback"));
    }
    if timing_sample_count < 2 {
        blockers.push(format!("repeated warm-session group {case_id} lacks enough timing samples"));
    }

    RepeatedWarmSessionGroupEvidence {
        attempt_count,
        answer_drift_detected: !stable_ids || !stable_text || !quality_passed,
        fallback_drift_detected,
        quality_passed,
        timing_sample_count,
        blockers,
    }
}

fn durability_profile_for_case_id(case_id: &str) -> Option<&'static str> {
    if case_id.starts_with("regression_tiny") {
        Some("regression_tiny")
    } else if case_id.starts_with("ask_short") {
        Some("ask_short")
    } else if case_id.starts_with("ask_normal") {
        Some("ask_normal")
    } else {
        None
    }
}

fn durability_prompt_has_timing(prompt: &Value) -> bool {
    number_at_any(
        prompt,
        &[
            "timing.total_ms",
            "timing.first_token_ms",
            "timing.first_token_decode_ms",
            "timing.time_to_first_token_ms",
            "timing.decode_total_ms",
        ],
    )
    .is_some()
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

fn write_or_print_cold_warm_benchmark(
    receipt: &LunarLakeColdWarmBenchmark,
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
        println!("Lunar Lake cold/warm profile benchmark written to {}", path.display());
    } else {
        println!("{}", String::from_utf8_lossy(&json));
    }
    Ok(())
}

fn write_or_print_cpu_slm_phase_attribution(
    receipt: &LunarLakeCpuSlmPhaseAttribution,
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
        println!("Lunar Lake CPU dense SLM phase attribution written to {}", path.display());
    } else {
        println!("{}", String::from_utf8_lossy(&json));
    }
    Ok(())
}

fn write_or_print_cpu_slm_resident_session(
    receipt: &LunarLakeCpuSlmResidentSession,
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
        println!("Lunar Lake CPU dense SLM resident-session receipt written to {}", path.display());
    } else {
        println!("{}", String::from_utf8_lossy(&json));
    }
    Ok(())
}

fn write_or_print_telemetry_context(
    receipt: &LunarLakeTelemetryContext,
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
        println!("Lunar Lake telemetry context receipt written to {}", path.display());
    } else {
        println!("{}", String::from_utf8_lossy(&json));
    }
    Ok(())
}

fn write_or_print_durability_bundle(
    receipt: &LunarLakeDurabilityBundle,
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
        println!("Lunar Lake durability bundle written to {}", path.display());
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
    telemetry_context: Option<&BenchmarkTelemetryContext>,
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
            evaluate_profile_route(
                root,
                profile,
                route_id,
                ledger,
                phase_comparison,
                quality_index,
                telemetry_context,
            )
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
    telemetry_context: Option<&BenchmarkTelemetryContext>,
) -> Result<ProfileRouteEvidence> {
    let route = ledger
        .routes
        .iter()
        .find(|route| route.route_id == route_id)
        .with_context(|| format!("route `{route_id}` not found in promotion ledger"))?;
    let timing = profile_timing_for_route(root, route_id, phase_comparison, telemetry_context)?;
    let mut blockers = route.missing_evidence.clone();
    let profile_quality = quality_index.get(route_id, &profile.profile_id).cloned();
    let telemetry = telemetry_context.map(telemetry_for_profile_route);
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
        if telemetry_context.is_some_and(|context| power_context_is_recorded(context)) {
            blockers.push("power advantage evidence missing for low_power promotion".to_string());
        } else {
            blockers.push("power telemetry receipt missing for low_power promotion".to_string());
        }
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
        telemetry,
        evidence,
        blockers,
    })
}

fn profile_timing_for_route(
    root: &Path,
    route_id: &str,
    phase_comparison: &Value,
    telemetry_context: Option<&BenchmarkTelemetryContext>,
) -> Result<ProfileTimingSummary> {
    match route_id {
        DEFAULT_ASK_ROUTE => dense_cpu_profile_timing(root, phase_comparison, telemetry_context),
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

fn dense_cpu_profile_timing(
    root: &Path,
    phase_comparison: &Value,
    telemetry_context: Option<&BenchmarkTelemetryContext>,
) -> Result<ProfileTimingSummary> {
    let ask_path = root.join(DENSE_CPU_OPERATOR_ASK);
    let ask: Value = read_json_receipt(&ask_path)?;
    let cold_load_ms = number_at_any(&ask, &["timing.model_load_ms"]);
    let tokenizer_load_ms = number_at_any(&ask, &["timing.tokenizer_load_ms"]);
    let tokenize_ms = number_at_any(&ask, &["timing.tokenize_ms"]);
    let prefill_ms = number_at_any(&ask, &["timing.prefill_ms"]);
    let output_tokens = number_at_any(&ask, &["tokens.generated_count", "timing.decode_tokens"])
        .map(|value| value as u64);
    let generation_total_ms = number_at_any(&ask, &["timing.decode_total_ms"]);
    let throughput_tokens_per_s = number_at_any(&ask, &["timing.decode_steady_state_tok_s"])
        .or_else(|| throughput_from_tokens(output_tokens, generation_total_ms));
    let total_response_ms = number_at_any(&ask, &["latency.total_ms", "timing.total_response_ms"])
        .or_else(|| {
            sum_all_optional([
                cold_load_ms,
                tokenizer_load_ms,
                tokenize_ms,
                prefill_ms,
                generation_total_ms,
            ])
        });

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
    let mut known_gaps =
        vec!["bounded math ask only; not expanded profile regression corpus".to_string()];
    if let Some(context) = telemetry_context {
        phase_coverage.push("telemetry_context_indexed".to_string());
        if context.thermal_context.to_ascii_lowercase().contains("unavailable") {
            known_gaps.push("thermal sensor context unavailable in telemetry receipt".to_string());
        }
        if !power_context_is_recorded(context) {
            known_gaps.push("power context unavailable in telemetry receipt".to_string());
        }
    } else {
        known_gaps.push("power and thermal context not normalized in this comparison".to_string());
    }

    Ok(ProfileTimingSummary {
        timing_scope: "dense_qwen_cpu_operator_ask_plus_warm_phase_receipts".to_string(),
        source_receipts: vec![
            DENSE_CPU_OPERATOR_ASK.to_string(),
            DENSE_CPU_PHASE.to_string(),
            DENSE_PHASE_COMPARISON.to_string(),
        ],
        cold_load_ms,
        tokenize_ms,
        prefill_ms,
        first_token_ms: number_at_any(&ask, &["timing.first_token_ms"]),
        decode_total_ms: generation_total_ms,
        generation_total_ms,
        total_response_ms,
        output_tokens,
        throughput_tokens_per_s,
        phase_coverage,
        known_gaps,
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

fn sum_all_optional<const N: usize>(values: [Option<f64>; N]) -> Option<f64> {
    values.into_iter().try_fold(0.0, |sum, value| value.map(|value| sum + value))
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
    fn comparison_receipt_carries_strict_regression_v2_surface() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-17T02:00:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;
        let mut regression = build_regression_bundle_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "2026-05-17T02:05:00Z".to_string(),
        )?;
        regression.regression_surface.answer_corpus_v2_indexed = true;
        regression.regression_surface.route_profile_comparison_indexed = true;
        regression.regression_surface.cold_warm_benchmark_indexed = true;
        regression.regression_surface.durability_bundle_indexed = true;
        regression.regression_surface.cold_warm_benchmark_ready = true;
        regression.regression_surface.durability_stability_proven = true;
        regression.regression_surface.candidate_routes_remain_unpromoted = true;
        regression.regression_surface.strict_ready = true;
        regression.regression_surface.gaps.clear();
        fs::write(temp.path().join(REGRESSION_BUNDLE_V2), serde_json::to_vec_pretty(&regression)?)?;

        let comparison = build_comparison_receipt_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(REGRESSION_BUNDLE_V2),
            "2026-05-17T02:10:00Z".to_string(),
        )?;

        assert!(comparison.comparison_ready, "{:?}", comparison.gaps);
        assert!(comparison.regression_bundle.ends_with(REGRESSION_BUNDLE_V2));
        assert!(comparison.regression_surface.strict_ready);
        assert!(comparison.regression_surface.answer_corpus_v2_indexed);
        assert!(comparison.regression_surface.route_profile_comparison_indexed);
        assert!(comparison.regression_surface.cold_warm_benchmark_indexed);
        assert!(comparison.regression_surface.durability_bundle_indexed);
        assert!(comparison.regression_surface.durability_stability_proven);
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

        write_json(
            temp.path(),
            POWER_THERMAL_CONTEXT_FILE,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_power_thermal_context",
                "proof_stage": "live_telemetry_context_captured_no_promotion_change",
                "created_utc": "2026-05-17T05:45:00Z",
                "machine_id": "intel-258v",
                "memory_context": "source=sysinfo;total_bytes=33873780736;available_bytes=10407493632;used_bytes=23466287104",
                "power_context": "source=os_power_probe;active_scheme=Balanced;battery_status=BatteryStatus=2;EstimatedChargeRemaining=100;ac_power_inferred=true",
                "thermal_context": "thermal_context_unavailable",
                "gaps": [
                    "thermal sensor context is not available from the current OS telemetry probe",
                    "power context is recorded for routing evidence, but no speedup or power-advantage claim is made"
                ],
                "claim_boundary": {
                    "new_inference_executed": false,
                    "telemetry_measurement_executed": true,
                    "route_promotion_changed": false,
                    "speedup_claim": false,
                    "power_advantage_claim": false,
                    "acceleration_claim": false,
                    "hidden_fallback_allowed": false
                }
            }),
        )?;
        let profiles_with_telemetry = build_route_profile_comparison_with_created_utc_and_inputs(
            temp.path(),
            Path::new(ROUTE_PROMOTION_LEDGER),
            Path::new(DENSE_PHASE_COMPARISON),
            None,
            None,
            Some(Path::new(POWER_THERMAL_CONTEXT_FILE)),
            "2026-05-17T06:55:00Z".to_string(),
        )?;
        assert!(
            profiles_with_telemetry.profile_comparison_ready,
            "{:?}",
            profiles_with_telemetry.gaps
        );
        assert_eq!(
            profiles_with_telemetry.telemetry_context_receipt.as_deref(),
            Some(path_string(&temp.path().join(POWER_THERMAL_CONTEXT_FILE)).as_str())
        );
        let Some(ask_normal_with_telemetry) = profiles_with_telemetry
            .profiles
            .iter()
            .find(|profile| profile.profile_id == "ask_normal")
        else {
            bail!("missing ask_normal profile with telemetry");
        };
        let Some(cpu_route_with_telemetry) = ask_normal_with_telemetry
            .route_evidence
            .iter()
            .find(|route| route.route_id == DEFAULT_ASK_ROUTE)
        else {
            bail!("missing CPU route evidence with telemetry");
        };
        assert!(cpu_route_with_telemetry.telemetry.is_some());
        assert!(
            !cpu_route_with_telemetry.timing.known_gaps.contains(
                &"power and thermal context not normalized in this comparison".to_string()
            )
        );
        assert!(
            cpu_route_with_telemetry
                .timing
                .known_gaps
                .contains(&"thermal sensor context unavailable in telemetry receipt".to_string())
        );
        let Some(low_power_with_telemetry) = profiles_with_telemetry
            .profiles
            .iter()
            .find(|profile| profile.profile_id == "low_power")
        else {
            bail!("missing low_power profile with telemetry");
        };
        assert!(low_power_with_telemetry.route_evidence.iter().any(|route| {
            route.route_id == "dense_slm_openvino_npu_candidate"
                && route.blockers.contains(
                    &"power advantage evidence missing for low_power promotion".to_string(),
                )
                && !route.blockers.contains(
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
                    "tokenizer_load_ms": 5.0,
                    "tokenize_ms": 2.0,
                    "prefill_ms": 20.0,
                    "first_token_ms": 30.0,
                    "decode_total_ms": 90.0,
                    "decode_steady_state_tok_s": 10.0
                },
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
        write_json(
            temp.path(),
            POWER_THERMAL_CONTEXT_FILE,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_power_thermal_context",
                "proof_stage": "telemetry_availability_recorded",
                "created_utc": "2026-05-16T17:50:00Z",
                "machine_id": "intel-258v",
                "memory_context": "not_recorded_in_committed_receipts",
                "power_context": "not_recorded_in_committed_receipts",
                "thermal_context": "not_recorded_in_committed_receipts",
                "gaps": ["power telemetry records absence only"],
                "claim_boundary": {
                    "new_measurement_executed": false,
                    "route_promotion_changed": false,
                    "speedup_claim": false,
                    "acceleration_claim": false,
                    "hidden_fallback_allowed": false
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

        let profiles = build_route_profile_comparison_with_created_utc_and_inputs(
            temp.path(),
            Path::new(ROUTE_PROMOTION_LEDGER),
            Path::new(DENSE_PHASE_COMPARISON),
            Some(Path::new(DENSE_CPU_CORPUS_V2)),
            Some(Path::new(DENSE_OV_CORPUS_V2)),
            None,
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
    fn cold_warm_benchmark_indexes_profile_timing_without_promoting_candidates() -> Result<()> {
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
                    "tokenizer_load_ms": 5.0,
                    "tokenize_ms": 2.0,
                    "prefill_ms": 20.0,
                    "first_token_ms": 30.0,
                    "decode_total_ms": 90.0,
                    "decode_steady_state_tok_s": 10.0
                },
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
        write_json(
            temp.path(),
            POWER_THERMAL_CONTEXT_FILE,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_power_thermal_context",
                "proof_stage": "telemetry_availability_recorded",
                "created_utc": "2026-05-16T17:50:00Z",
                "machine_id": "intel-258v",
                "memory_context": "not_recorded_in_committed_receipts",
                "power_context": "not_recorded_in_committed_receipts",
                "thermal_context": "not_recorded_in_committed_receipts",
                "gaps": ["power telemetry records absence only"],
                "claim_boundary": {
                    "new_measurement_executed": false,
                    "route_promotion_changed": false,
                    "speedup_claim": false,
                    "acceleration_claim": false,
                    "hidden_fallback_allowed": false
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
        let profiles = build_route_profile_comparison_with_created_utc_and_inputs(
            temp.path(),
            Path::new(ROUTE_PROMOTION_LEDGER),
            Path::new(DENSE_PHASE_COMPARISON),
            Some(Path::new(DENSE_CPU_CORPUS_V2)),
            Some(Path::new(DENSE_OV_CORPUS_V2)),
            None,
            "2026-05-16T07:30:00Z".to_string(),
        )?;
        fs::write(
            temp.path().join(ROUTE_PROFILE_COMPARISON),
            serde_json::to_vec_pretty(&profiles)?,
        )?;

        let benchmark = build_cold_warm_benchmark_with_created_utc(
            temp.path(),
            Path::new(ROUTE_PROFILE_COMPARISON),
            Path::new(DENSE_PHASE_COMPARISON),
            Some(Path::new(POWER_THERMAL_CONTEXT_FILE)),
            "2026-05-16T18:00:00Z".to_string(),
        )?;

        assert!(benchmark.benchmark_gate_ready, "{:?}", benchmark.gaps);
        assert_eq!(benchmark.artifact_kind, "lunar_lake_cold_warm_profile_benchmark");
        let Some(ask_normal) =
            benchmark.profiles.iter().find(|profile| profile.profile_id == "ask_normal")
        else {
            bail!("missing ask_normal benchmark");
        };
        let cpu = ask_normal
            .routes
            .iter()
            .find(|route| route.route_id == DEFAULT_ASK_ROUTE)
            .context("missing CPU route benchmark")?;
        assert!(cpu.critical_timing_present);
        assert!(!cpu.promotion_blocked);
        assert_eq!(cpu.timing.total_response_ms, Some(217.0));
        assert!(cpu.telemetry.telemetry_receipt.is_some());
        assert_eq!(cpu.telemetry.memory_context, "not_recorded_in_committed_receipts");
        assert!(!cpu.blockers.iter().any(|blocker| blocker == "total response latency is missing"));
        let gpu = ask_normal
            .routes
            .iter()
            .find(|route| route.route_id == "dense_slm_openvino_gpu_candidate")
            .context("missing GPU route benchmark")?;
        assert!(gpu.promotion_blocked);
        assert!(!gpu.benchmark_qualified_advantage);
        let Some(low_power) =
            benchmark.profiles.iter().find(|profile| profile.profile_id == "low_power")
        else {
            bail!("missing low_power benchmark");
        };
        assert!(low_power.routes.iter().any(|route| {
            route.route_id == "dense_slm_openvino_npu_candidate"
                && route.blockers.contains(
                    &"power telemetry receipt does not provide low_power promotion evidence"
                        .to_string(),
                )
        }));
        assert!(!benchmark.claim_boundary.speedup_claim);
        assert!(!benchmark.claim_boundary.route_promotion_changed);
        Ok(())
    }

    #[test]
    fn cpu_slm_phase_attribution_indexes_cold_and_warm_cpu_timing() -> Result<()> {
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
                    "tokenizer_load_ms": 5.0,
                    "tokenize_ms": 2.0,
                    "prefill_ms": 20.0,
                    "first_token_ms": 30.0,
                    "decode_total_ms": 90.0,
                    "decode_steady_state_tok_s": 10.0
                },
                "latency": {"total_ms": 217.0},
                "tokens": {"generated_count": 8}
            }),
        )?;
        write_json(
            temp.path(),
            DENSE_CPU_PHASE,
            json!({
                "artifact_kind": "dense_slm_cpu_phase_warm_session",
                "fallback_used": false,
                "model_family": "qwen",
                "model_architecture": "qwen2",
                "quantization": "Q8_0",
                "tokenizer_source": "gguf_metadata",
                "prompt_template": "qwen2.5",
                "selected_kernel_or_runtime": "dense-qwen-cpu-reference",
                "session": {
                    "model_loaded_once": true,
                    "tokenizer_loaded_once": true
                },
                "timing": {
                    "model_load_ms": 40.0,
                    "tokenizer_load_ms": 5.0,
                    "total_session_ms": 1000.0
                },
                "profiles": [
                    {
                        "profile": "prefill_512",
                        "prompt_tokens": 512,
                        "prefill_ms": 1024.0,
                        "generated_tokens": 1,
                        "first_token_decode_ms": 20.0,
                        "decode_total_ms": 20.0,
                        "fallback_used": false,
                        "receipt_path": "prefill.json"
                    },
                    {
                        "profile": "decode_128",
                        "prompt_tokens": 32,
                        "prefill_ms": 64.0,
                        "generated_tokens": 128,
                        "first_token_decode_ms": 12.0,
                        "decode_total_ms": 640.0,
                        "fallback_used": false,
                        "receipt_path": "decode.json"
                    }
                ]
            }),
        )?;
        write_json(
            temp.path(),
            DENSE_PHASE_COMPARISON,
            json!({
                "artifact_kind": "intel_258v_dense_slm_openvino_phase_comparison",
                "fallback_used": false,
                "gguf_cpu_reference": {"timing": {"prefill_512": {}, "decode_128": {}}},
                "openvino_paths": {
                    "cpu": {
                        "source_receipt": "openvino-cpu.json",
                        "selected_backend": "openvino-cpu",
                        "runtime_api": "openvino_genai",
                        "fallback_used": false,
                        "answer_gate": {"passed": true},
                        "timing": {
                            "pipeline_load_ms": 10.0,
                            "case_elapsed_ms_sum": 20.0
                        }
                    }
                }
            }),
        )?;

        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-17T08:00:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;
        let regression = build_regression_bundle_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "2026-05-17T08:01:00Z".to_string(),
        )?;
        fs::write(temp.path().join(REGRESSION_BUNDLE), serde_json::to_vec_pretty(&regression)?)?;
        let comparison = build_comparison_receipt_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(REGRESSION_BUNDLE),
            "2026-05-17T08:02:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_COMPARISON), serde_json::to_vec_pretty(&comparison)?)?;
        let ledger = build_route_promotion_ledger_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(OPERATOR_COMPARISON),
            "2026-05-17T08:03:00Z".to_string(),
        )?;
        fs::write(temp.path().join(ROUTE_PROMOTION_LEDGER), serde_json::to_vec_pretty(&ledger)?)?;
        let profiles = build_route_profile_comparison_with_created_utc_and_inputs(
            temp.path(),
            Path::new(ROUTE_PROMOTION_LEDGER),
            Path::new(DENSE_PHASE_COMPARISON),
            Some(Path::new(DENSE_CPU_CORPUS_V2)),
            Some(Path::new(DENSE_OV_CORPUS_V2)),
            None,
            "2026-05-17T08:04:00Z".to_string(),
        )?;
        fs::write(
            temp.path().join(ROUTE_PROFILE_COMPARISON),
            serde_json::to_vec_pretty(&profiles)?,
        )?;
        let cold_warm = build_cold_warm_benchmark_with_created_utc(
            temp.path(),
            Path::new(ROUTE_PROFILE_COMPARISON),
            Path::new(DENSE_PHASE_COMPARISON),
            None,
            "2026-05-17T08:05:00Z".to_string(),
        )?;
        fs::write(temp.path().join("cold-warm.json"), serde_json::to_vec_pretty(&cold_warm)?)?;

        let receipt = build_cpu_slm_phase_attribution_with_created_utc(
            temp.path(),
            Path::new(DENSE_CPU_PHASE),
            Path::new("cold-warm.json"),
            Path::new(DENSE_PHASE_COMPARISON),
            "2026-05-17T08:06:00Z".to_string(),
        )?;

        assert!(receipt.attribution_ready, "{:?}", receipt.gaps);
        assert_eq!(receipt.artifact_kind, "lunar_lake_cpu_slm_phase_attribution");
        assert_eq!(receipt.backend.selected_backend, "cpu-rust");
        assert_eq!(receipt.backend.runtime_api, "cpu");
        assert_eq!(receipt.cold_one_off.timing.total_response_ms, Some(217.0));
        assert_eq!(receipt.cold_one_off.model_load_share_of_total, Some(100.0 / 217.0));
        assert!(receipt.warm_session.model_loaded_once == Some(true));
        let decode = receipt
            .warm_session
            .profiles
            .iter()
            .find(|profile| profile.profile == "decode_128")
            .context("missing decode_128")?;
        assert_eq!(decode.decode_tokens_per_s, Some(200.0));
        let openvino = receipt.openvino_cpu_context.as_ref().context("missing openvino cpu")?;
        assert_eq!(openvino.pipeline_load_ms, Some(10.0));
        assert!(!receipt.claim_boundary.new_inference_executed);
        assert!(!receipt.claim_boundary.route_promotion_changed);
        assert!(!receipt.claim_boundary.speedup_claim);
        assert!(!receipt.claim_boundary.arc_npu_execution_claim);
        assert!(!receipt.claim_boundary.bitnet_qk256_i2s_claim);
        Ok(())
    }

    #[test]
    fn cpu_slm_resident_session_summarizes_no_reload_warm_loop() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_json(
            temp.path(),
            "phase-attribution.json",
            json!({
                "artifact_kind": "lunar_lake_cpu_slm_phase_attribution",
                "attribution_ready": true,
                "cold_one_off": {
                    "profile_id": "ask_short",
                    "timing": {
                        "cold_load_ms": 100.0,
                        "tokenize_ms": 2.0,
                        "prefill_ms": 20.0,
                        "first_token_ms": 30.0,
                        "decode_total_ms": 40.0,
                        "total_response_ms": 200.0
                    }
                }
            }),
        )?;
        write_json(
            temp.path(),
            "resident.json",
            json!({
                "artifact_kind": "slm_cpu_warm_session",
                "selected_backend": "cpu-rust",
                "runtime_api": "cpu",
                "fallback_used": false,
                "speedup_claim": false,
                "quality_summary": {"passed": true},
                "determinism": {
                    "passed": true,
                    "groups": [
                        {
                            "case_id": "ask_short_math",
                            "attempt_count": 2,
                            "stable_generated_token_ids": true,
                            "stable_text": true,
                            "prompt_indices": [0, 1]
                        }
                    ]
                },
                "claim_boundary": {
                    "speedup_claim": false,
                    "broad_performance_claim": false,
                    "full_metal_inference_claimed": false,
                    "bitnet_quality_claimed": false
                },
                "model": {
                    "family": "qwen",
                    "architecture": "qwen2",
                    "quant_format": "Q8_0",
                    "tokenizer": "tokenizer.json"
                },
                "generation": {"prompt_template": "qwen2.5"},
                "session": {
                    "reuse_scope": "resident_session",
                    "model_loaded_once": true,
                    "tokenizer_loaded_once": true,
                    "prompt_count": 2,
                    "per_prompt_receipts_enabled": true,
                    "session_owned_buffers": true,
                    "prompt_token_buffer_reused": true,
                    "generated_token_buffer_reused": true,
                    "timing_buffers_reused": true,
                    "stop_policy_precomputed_once": true
                },
                "memory": {"resident_memory_bytes": 1000},
                "timing": {
                    "model_load_ms": 100.0,
                    "model_sha256_ms": 5.0,
                    "tokenizer_load_ms": 10.0,
                    "total_session_ms": 260.0
                },
                "prompts": [
                    {
                        "prompt_index": 0,
                        "case_id": "ask_short_math",
                        "fallback_used": false,
                        "generated_tokens": 4,
                        "quality": {"passed": true},
                        "timing": {
                            "model_load_ms": 0.0,
                            "tokenizer_load_ms": 0.0,
                            "total_ms": 80.0,
                            "time_to_first_token_ms": 30.0,
                            "prefill_ms": 20.0,
                            "decode_total_ms": 40.0,
                            "tokenize_ms": 2.0
                        }
                    },
                    {
                        "prompt_index": 1,
                        "case_id": "ask_short_math",
                        "backend": {"fallback_used": false},
                        "generated_tokens": 4,
                        "quality": {"passed": true},
                        "timing": {
                            "model_load_ms": 0.0,
                            "tokenizer_load_ms": 0.0,
                            "total_ms": 100.0,
                            "first_token_ms": 40.0,
                            "prefill_ms": 22.0,
                            "decode_total_ms": 44.0,
                            "tokenize_ms": 3.0
                        }
                    }
                ]
            }),
        )?;

        let receipt = build_cpu_slm_resident_session_with_created_utc(
            temp.path(),
            Path::new("phase-attribution.json"),
            Path::new("resident.json"),
            2,
            "2026-05-17T09:15:00Z".to_string(),
        )?;

        assert!(receipt.resident_ready, "{:?}", receipt.gaps);
        assert_eq!(receipt.artifact_kind, "lunar_lake_cpu_slm_resident_session");
        assert_eq!(receipt.backend.selected_backend, "cpu-rust");
        assert_eq!(receipt.resident_session.model_loaded_once, Some(true));
        assert_eq!(receipt.resident_session.tokenizer_loaded_once, Some(true));
        let profile = receipt
            .profiles
            .iter()
            .find(|profile| profile.profile_id == "ask_short")
            .context("missing ask_short profile")?;
        assert_eq!(profile.observed_execution_count, 2);
        assert_eq!(profile.total_ms.mean, Some(90.0));
        assert_eq!(profile.decode_tokens_per_s_mean, Some(8.0 / 0.084));
        assert_eq!(profile.cold_to_resident_total_ratio, Some(200.0 / 90.0));
        assert!(profile.blockers.is_empty());
        assert!(!receipt.claim_boundary.new_inference_executed);
        assert!(!receipt.claim_boundary.speedup_claim);
        assert!(!receipt.claim_boundary.route_promotion_changed);
        assert!(!receipt.claim_boundary.arc_npu_execution_claim);
        assert!(!receipt.claim_boundary.bitnet_qk256_i2s_claim);
        Ok(())
    }

    #[test]
    fn telemetry_context_records_live_context_without_route_claims() -> Result<()> {
        let temp = tempfile::tempdir()?;

        let receipt = build_telemetry_context_with_created_utc(
            temp.path(),
            "2026-05-17T05:45:00Z".to_string(),
        );

        assert_eq!(receipt.artifact_kind, "lunar_lake_power_thermal_context");
        assert_eq!(receipt.proof_stage, "live_telemetry_context_captured_no_promotion_change");
        assert!(receipt.claim_boundary.telemetry_measurement_executed);
        assert!(!receipt.claim_boundary.new_inference_executed);
        assert!(!receipt.claim_boundary.route_promotion_changed);
        assert!(!receipt.claim_boundary.speedup_claim);
        assert!(!receipt.claim_boundary.power_advantage_claim);
        assert!(!receipt.claim_boundary.acceleration_claim);
        assert_eq!(receipt.memory.source, "sysinfo");
        assert!(receipt.sources.iter().any(|source| source.source == "sysinfo"));
        Ok(())
    }

    #[test]
    fn durability_bundle_indexes_repeat_gap_and_repeated_stability() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        write_answer_corpus_v2(temp.path(), "corpus-v2.yaml")?;
        write_json(
            temp.path(),
            DENSE_CPU_CORPUS_V2,
            json!({
                "artifact_kind": "slm_cpu_answer_corpus",
                "fallback_used": false,
                "profile_summary": {
                    "regression_tiny": {"total": 4, "passed": 4, "failed": 0},
                    "ask_short": {"total": 2, "passed": 2, "failed": 0},
                    "ask_normal": {"total": 3, "passed": 3, "failed": 0}
                },
                "cases": [
                    {"id": "math_2_plus_2_brief", "profile": "regression_tiny", "status": "passed"},
                    {"id": "copy_exact_color_triplet", "profile": "regression_tiny", "status": "passed"},
                    {"id": "stop_token_one_word_done", "profile": "regression_tiny", "status": "passed"},
                    {"id": "arithmetic_add_7_8", "profile": "regression_tiny", "status": "passed"},
                    {"id": "yes_no_clear_sky", "profile": "ask_short", "status": "passed"},
                    {"id": "short_factual_capital_france", "profile": "ask_short", "status": "passed"},
                    {"id": "instruction_single_sentence_rust", "profile": "ask_normal", "status": "passed"},
                    {"id": "transcript_context_code_word", "profile": "ask_normal", "status": "passed"},
                    {"id": "short_reasoning_apples_left", "profile": "ask_normal", "status": "passed"}
                ]
            }),
        )?;
        write_json(
            temp.path(),
            DENSE_OV_CORPUS_V2,
            json!({
                "artifact_kind": "intel_258v_dense_slm_openvino_corpus_v2",
                "fallback_used": false,
                "generation": {
                    "devices": [
                        {
                            "runtime_device": "GPU.0",
                            "fallback_used": false,
                            "quality_summary": {
                                "profile_summary": {
                                    "ask_short": {"total": 2, "passed": 2, "failed": 0},
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
            None,
            "2026-05-16T07:30:00Z".to_string(),
        )?;
        fs::write(
            temp.path().join(ROUTE_PROFILE_COMPARISON),
            serde_json::to_vec_pretty(&profiles)?,
        )?;
        let cold_warm = build_cold_warm_benchmark_with_created_utc(
            temp.path(),
            Path::new(ROUTE_PROFILE_COMPARISON),
            Path::new(DENSE_PHASE_COMPARISON),
            None,
            "2026-05-16T18:00:00Z".to_string(),
        )?;
        fs::write(temp.path().join("cold-warm.json"), serde_json::to_vec_pretty(&cold_warm)?)?;
        let mut regression_v2 = build_regression_bundle_with_created_utc_and_inputs(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Some(Path::new("corpus-v2.yaml")),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            Some(Path::new("cold-warm.json")),
            None,
            "2026-05-16T19:05:00Z".to_string(),
        )?;
        // Seed the durability builder with the pre-REG-005 strict surface it
        // originally consumed; REG-005 adds durability back into regression.
        regression_v2.regression_passed = true;
        regression_v2.gaps.clear();
        regression_v2.regression_surface.strict_ready = true;
        regression_v2.regression_surface.gaps.clear();
        fs::write(
            temp.path().join(REGRESSION_BUNDLE_V2),
            serde_json::to_vec_pretty(&regression_v2)?,
        )?;

        let durability = build_durability_bundle_with_created_utc(
            temp.path(),
            Path::new(ROUTE_PROFILE_COMPARISON),
            Path::new("cold-warm.json"),
            Path::new(DENSE_CPU_CORPUS_V2),
            Path::new(REGRESSION_BUNDLE_V2),
            None,
            10,
            "2026-05-16T20:20:00Z".to_string(),
        )?;

        assert!(durability.durability_index_ready, "{:?}", durability.gaps);
        assert!(!durability.stability_proven);
        assert!(!durability.claim_boundary.repeated_run_stability_claim);
        assert!(!durability.claim_boundary.new_inference_executed);
        let ask_short = durability
            .profiles
            .iter()
            .find(|profile| profile.profile_id == "ask_short")
            .context("missing ask_short durability profile")?;
        assert_eq!(ask_short.observed_execution_count, 1);
        assert_eq!(ask_short.required_execution_count, 10);
        assert_eq!(ask_short.baseline_cases_failed, 0);
        assert!(ask_short.answer_drift_detected.is_none());
        assert!(ask_short.blockers.iter().any(|blocker| blocker.contains("repeated-run")));
        assert!(
            durability
                .next_required_evidence
                .iter()
                .any(|evidence| { evidence.contains("collect repeated-run receipts") })
        );

        write_repeated_warm_session_receipt(temp.path(), "durable-warm.json")?;
        let durability = build_durability_bundle_with_created_utc(
            temp.path(),
            Path::new(ROUTE_PROFILE_COMPARISON),
            Path::new("cold-warm.json"),
            Path::new(DENSE_CPU_CORPUS_V2),
            Path::new(REGRESSION_BUNDLE_V2),
            Some(Path::new("durable-warm.json")),
            10,
            "2026-05-16T20:40:00Z".to_string(),
        )?;

        assert!(durability.durability_index_ready, "{:?}", durability.gaps);
        assert!(durability.stability_proven, "{:?}", durability.profiles);
        assert!(durability.claim_boundary.repeated_run_stability_claim);
        let expected_repeated_receipt = path_string(&temp.path().join("durable-warm.json"));
        assert_eq!(durability.repeated_warm_session_receipt, Some(expected_repeated_receipt));
        assert!(
            durability
                .next_required_evidence
                .iter()
                .all(|evidence| !evidence.contains("repeated-run"))
        );
        for profile in &durability.profiles {
            assert_eq!(profile.observed_execution_count, 10);
            assert_eq!(profile.required_execution_count, 10);
            assert_eq!(profile.answer_drift_detected, Some(false));
            assert_eq!(profile.fallback_drift_detected, Some(false));
            assert_eq!(profile.latency_variance_status, "variance_window_available");
            assert_eq!(profile.stability_status, "stable");
            assert!(profile.blockers.is_empty(), "{profile:?}");
        }
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
        let cold_warm = build_cold_warm_benchmark_with_created_utc(
            temp.path(),
            Path::new(ROUTE_PROFILE_COMPARISON),
            Path::new(DENSE_PHASE_COMPARISON),
            None,
            "2026-05-14T17:45:00Z".to_string(),
        )?;
        fs::write(temp.path().join("cold-warm.json"), serde_json::to_vec_pretty(&cold_warm)?)?;
        write_json(
            temp.path(),
            DENSE_CPU_CORPUS_V2,
            json!({
                "artifact_kind": "slm_cpu_answer_corpus",
                "fallback_used": false,
                "cases": [
                    {"id": "math_2_plus_2_brief", "profile": "regression_tiny", "status": "passed"},
                    {"id": "copy_exact_color_triplet", "profile": "regression_tiny", "status": "passed"},
                    {"id": "stop_token_one_word_done", "profile": "regression_tiny", "status": "passed"},
                    {"id": "arithmetic_add_7_8", "profile": "regression_tiny", "status": "passed"},
                    {"id": "yes_no_clear_sky", "profile": "ask_short", "status": "passed"},
                    {"id": "short_factual_capital_france", "profile": "ask_short", "status": "passed"},
                    {"id": "instruction_single_sentence_rust", "profile": "ask_normal", "status": "passed"},
                    {"id": "transcript_context_code_word", "profile": "ask_normal", "status": "passed"},
                    {"id": "short_reasoning_apples_left", "profile": "ask_normal", "status": "passed"}
                ]
            }),
        )?;

        write_json(
            temp.path(),
            "durability.json",
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_durability_bundle",
                "proof_stage": "repeated_run_requirements_indexed_no_new_inference",
                "created_utc": "2026-05-14T23:45:00Z",
                "machine_id": "intel-258v",
                "artifact_root": path_string(temp.path()),
                "route_profile_comparison_receipt": path_string(&temp.path().join(ROUTE_PROFILE_COMPARISON)),
                "cold_warm_benchmark_receipt": path_string(&temp.path().join("cold-warm.json")),
                "cpu_corpus_v2_receipt": path_string(&temp.path().join(DENSE_CPU_CORPUS_V2)),
                "regression_bundle_receipt": path_string(&temp.path().join(REGRESSION_BUNDLE_V2)),
                "repeated_warm_session_receipt": path_string(&temp.path().join("durable-warm.json")),
                "required_repeat_count": 10,
                "durability_index_ready": true,
                "stability_proven": true,
                "profiles": [
                    stable_durability_profile("regression_tiny", 4, 4),
                    stable_durability_profile("ask_short", 2, 2),
                    stable_durability_profile("ask_normal", 3, 3)
                ],
                "gaps": [],
                "next_required_evidence": [],
                "claim_boundary": {
                    "new_inference_executed": false,
                    "route_promotion_changed": false,
                    "broad_quality_claim": false,
                    "speedup_claim": false,
                    "acceleration_claim": false,
                    "hidden_fallback_allowed": false,
                    "dense_slm_as_bitnet_proof": false,
                    "repeated_run_stability_claim": true
                }
            }),
        )?;

        let bundle = build_regression_bundle_with_created_utc_and_inputs(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Some(Path::new("corpus-v2.yaml")),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            Some(Path::new("cold-warm.json")),
            Some(Path::new("durability.json")),
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
        assert!(bundle.regression_surface.cold_warm_benchmark_indexed);
        assert!(bundle.regression_surface.cold_warm_benchmark_ready);
        assert!(bundle.regression_surface.durability_bundle_indexed);
        assert!(bundle.regression_surface.durability_stability_proven);
        let Some(cold_warm) = bundle.cold_warm_benchmark.as_ref() else {
            bail!("missing cold_warm_benchmark summary");
        };
        assert!(cold_warm.promoted_routes_have_critical_timing);
        assert!(cold_warm.candidate_routes_remain_unpromoted);
        let Some(durability) = bundle.durability_bundle.as_ref() else {
            bail!("missing durability bundle summary");
        };
        assert!(durability.regression_ready, "{:?}", durability.gaps);
        assert!(durability.stability_proven);
        assert_eq!(durability.stable_profile_count, 3);
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
        let cold_warm = build_cold_warm_benchmark_with_created_utc(
            temp.path(),
            Path::new(ROUTE_PROFILE_COMPARISON),
            Path::new(DENSE_PHASE_COMPARISON),
            None,
            "2026-05-14T17:45:00Z".to_string(),
        )?;
        fs::write(temp.path().join("cold-warm.json"), serde_json::to_vec_pretty(&cold_warm)?)?;

        let bundle = build_regression_bundle_with_created_utc_and_inputs(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Some(Path::new("corpus-v2.yaml")),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            Some(Path::new("cold-warm.json")),
            None,
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

    fn stable_durability_profile(profile_id: &str, total: u64, passed: u64) -> Value {
        json!({
            "profile_id": profile_id,
            "route_id": DEFAULT_ASK_ROUTE,
            "route_status": "promoted",
            "promoted_route": DEFAULT_ASK_ROUTE,
            "baseline_case_count": total,
            "baseline_cases_passed": passed,
            "baseline_cases_failed": total.saturating_sub(passed),
            "observed_execution_count": 10,
            "required_execution_count": 10,
            "answer_drift_detected": false,
            "route_drift_detected": false,
            "fallback_drift_detected": false,
            "latency_variance_status": "variance_window_available",
            "stability_status": "stable",
            "blockers": []
        })
    }

    fn write_repeated_warm_session_receipt(root: &Path, file: &str) -> Result<()> {
        let cases = [
            ("regression_tiny_math_2_plus_2_brief", 0u64),
            ("ask_short_capital_france", 10u64),
            ("ask_normal_instruction_rust", 20u64),
        ];
        let groups = cases
            .iter()
            .map(|(case_id, start)| {
                json!({
                    "case_id": case_id,
                    "attempt_count": 10,
                    "prompt_indices": (*start..*start + 10).collect::<Vec<_>>(),
                    "stable_generated_token_ids": true,
                    "stable_text": true
                })
            })
            .collect::<Vec<_>>();
        let prompts = cases
            .iter()
            .flat_map(|(case_id, start)| {
                (*start..*start + 10).map(move |prompt_index| {
                    json!({
                        "case_id": case_id,
                        "prompt_index": prompt_index,
                        "repeat_index": prompt_index - *start,
                        "fallback_used": false,
                        "backend": {
                            "fallback_used": false,
                            "runtime_api": "cpu",
                            "selected_backend": "cpu-rust"
                        },
                        "quality": {
                            "passed": true
                        },
                        "timing": {
                            "total_ms": 1.0,
                            "first_token_ms": 1.0,
                            "decode_total_ms": 1.0
                        }
                    })
                })
            })
            .collect::<Vec<_>>();

        write_json(
            root,
            file,
            json!({
                "artifact_kind": "slm_cpu_warm_session",
                "selected_backend": "cpu-rust",
                "runtime_api": "cpu",
                "fallback_used": false,
                "backend": {
                    "selected_backend": "cpu-rust",
                    "runtime_api": "cpu",
                    "fallback_used": false
                },
                "quality_summary": {
                    "passed": true,
                    "failed_prompt_indices": []
                },
                "determinism": {
                    "passed": true,
                    "repeated_prompt_groups": 3,
                    "groups": groups
                },
                "claim_boundary": {
                    "speedup_claim": false,
                    "broad_performance_claim": false,
                    "full_metal_inference_claimed": false,
                    "bitnet_quality_claimed": false
                },
                "prompts": prompts
            }),
        )
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
