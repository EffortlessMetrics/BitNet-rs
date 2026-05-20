//! Lunar Lake operator readiness helpers.
//!
//! These commands do not run inference. They turn the existing 258V proof bundle
//! into an operator-facing route/readiness artifact so users can see which path
//! is the safe default, which profiles have earned accelerator routes, and
//! which profiles remain blocked.

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
const DENSE_SLM_ARTIFACT_MANIFEST: &str = "slm-artifact-manifest.json";
const DENSE_SLM_OPENVINO_IR_MANIFEST: &str = "slm-openvino-ir-qwen25-int4-sym-manifest.json";
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
const BITNET_SEMANTIC_SOURCE_CHANGES: &str = "lunar-lake-bitnet-semantic-source-changes.json";
const BITNET_SEMANTIC_INTAKE: &str = "lunar-lake-bitnet-semantic-intake.json";
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
pub const LOW_POWER_BATTERY_RUNBOOK: &str = "docs/hardware/intel-258v-low-power-battery-runbook.md";
const COLD_WARM_PROFILE_BENCHMARK: &str =
    "ci/hardware/intel-258v/2026-05-08/lunar-lake-cold-warm-profile-benchmark.json";
const COLD_WARM_PROFILE_BENCHMARK_FILE: &str = "lunar-lake-cold-warm-profile-benchmark.json";
const POWER_THERMAL_CONTEXT_FILE: &str = "lunar-lake-power-thermal-context.json";
const POWER_PROFILE_EVIDENCE_FILE: &str = "lunar-lake-power-profile-evidence.json";
const THERMAL_TEMPERATURE_AVAILABILITY_FILE: &str =
    "lunar-lake-thermal-temperature-availability.json";
const LOW_POWER_ENERGY_PROXY_FILE: &str = "lunar-lake-low-power-energy-proxy.json";
const DURABILITY_BUNDLE: &str =
    "ci/hardware/intel-258v/2026-05-08/lunar-lake-durability-bundle.json";
const DURABLE_QWEN_CPU_WARM_SESSION: &str = "lunar-lake-durable-qwen25-cpu-warm-session.json";
const CPU_SLM_PHASE_ATTRIBUTION: &str = "lunar-lake-cpu-slm-phase-attribution.json";
const CPU_SLM_RESIDENT_SESSION: &str = "lunar-lake-cpu-slm-resident-session.json";
const CPU_SLM_RUNTIME_COMPARISON: &str = "lunar-lake-cpu-slm-runtime-comparison.json";
const OPENVINO_GPU_CORPUS_V2_DIAGNOSIS: &str = "lunar-lake-openvino-gpu-corpus-v2-diagnosis.json";
const OPENVINO_NPU_COLD_START_DIAGNOSIS: &str = "lunar-lake-openvino-npu-cold-start-diagnosis.json";
const OPENVINO_NPU_RESIDENT_SESSION: &str = "lunar-lake-openvino-npu-resident-session.json";
const OPENVINO_NPU_CACHE_EXPERIMENT: &str = "lunar-lake-openvino-npu-cache-experiment.json";
const OPENVINO_GENERATION_BUDGET_SENSITIVITY: &str =
    "lunar-lake-openvino-generation-budget-sensitivity.json";
const OPENVINO_PROFILE_RUN: &str = "lunar-lake-openvino-profile-run.json";
const OPENVINO_GPU_PROFILE_PROMOTION_TARGETS: &[&str] =
    &["ask_short", "ask_normal", "prefill_heavy", "decode_heavy"];
const OPENVINO_NPU_PROFILE_PROMOTION_TARGETS: &[&str] = &["warm_resident"];
const DENSE_PHASE_COMPARISON: &str = "slm-openvino-cpu-gpu-npu-phase-comparison.json";
const DENSE_CPU_OPERATOR_ASK: &str = "lunar-lake-operator-ask-math-brief.json";
const BLOCKED_AUTO_ASK_RECEIPT: &str = "lunar-lake-operator-ask-auto-low-power-blocked.json";
const AUTO_GPU_ASK_SHORT_ASK_RECEIPT: &str =
    "lunar-lake-operator-ask-auto-gpu-ask-short-math-brief.json";
const AUTO_GPU_ASK_NORMAL_ASK_RECEIPT: &str =
    "lunar-lake-operator-ask-auto-gpu-ask-normal-math-brief.json";
const AUTO_NPU_WARM_RESIDENT_ASK_RECEIPT: &str =
    "lunar-lake-operator-ask-auto-npu-warm-resident-math-brief.json";
const ANSWER_CORPUS_V2: &str = "ci/quality/lunar-lake-answer-corpus-v2.yaml";
const REGRESSION_V2_SURFACE_ID: &str = "lunar_lake_regression_v2";
pub const DEFAULT_ASK_ROUTE: &str = "dense_slm_default_cpu";

const REQUIRED_CORPUS_V2_PROFILES: &[&str] = &[
    "regression_tiny",
    "ask_short",
    "ask_normal",
    "structured",
    "prefill_heavy",
    "decode_heavy",
    "low_power",
    "warm_resident",
];
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
    "resident_session",
];
const REQUIRED_ROUTE_PROFILES: &[&str] = &[
    "regression_tiny",
    "ask_short",
    "ask_normal",
    "prefill_heavy",
    "decode_heavy",
    "structured",
    "low_power",
    "warm_resident",
    "bitnet_strict_reference",
];
const BENCHMARK_QUALIFIED_LATENCY_RATIO_MAX: f64 = 0.90;
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

        /// Route promotion ledger to index for profile-aware operator readiness.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = ROUTE_PROMOTION_LEDGER)]
        route_promotion_ledger: Option<PathBuf>,

        /// Route profile comparison receipt to index for profile-aware operator readiness.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = ROUTE_PROFILE_COMPARISON)]
        route_profile_comparison: Option<PathBuf>,

        /// Low-power route power-profile evidence receipt to index in readiness.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = POWER_PROFILE_EVIDENCE_FILE)]
        power_profile_evidence: Option<PathBuf>,

        /// Thermal temperature availability diagnosis receipt to index in readiness.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = THERMAL_TEMPERATURE_AVAILABILITY_FILE)]
        thermal_temperature_availability: Option<PathBuf>,

        /// Blocked low_power auto ask receipt to index in readiness.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = BLOCKED_AUTO_ASK_RECEIPT)]
        blocked_ask_receipt: Option<PathBuf>,

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

        /// Optional BitNet semantic-intake receipt to index.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = BITNET_SEMANTIC_INTAKE)]
        bitnet_semantic_intake: Option<PathBuf>,

        /// Optional low-power route power-profile evidence receipt to index.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = POWER_PROFILE_EVIDENCE_FILE)]
        power_profile_evidence: Option<PathBuf>,

        /// Optional thermal temperature availability diagnosis receipt to index.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = THERMAL_TEMPERATURE_AVAILABILITY_FILE)]
        thermal_temperature_availability: Option<PathBuf>,

        /// Optional successful auto ask_short ask receipt to index.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = AUTO_GPU_ASK_SHORT_ASK_RECEIPT)]
        ask_short_ask_receipt: Option<PathBuf>,

        /// Optional successful auto ask_normal ask receipt to index.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = AUTO_GPU_ASK_NORMAL_ASK_RECEIPT)]
        ask_normal_ask_receipt: Option<PathBuf>,

        /// Optional successful auto warm-resident ask receipt to index.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = AUTO_NPU_WARM_RESIDENT_ASK_RECEIPT)]
        warm_resident_ask_receipt: Option<PathBuf>,

        /// Optional blocked auto-route ask receipt to index.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = BLOCKED_AUTO_ASK_RECEIPT)]
        blocked_ask_receipt: Option<PathBuf>,

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

        /// Optional route-profile comparison receipt proving benchmark-qualified exact-profile promotions.
        /// Relative paths are resolved under artifact-root.
        #[arg(long)]
        route_profile_comparison: Option<PathBuf>,

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

        /// Active answer corpus v2 fixture used to verify corpus receipt case alignment.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = ANSWER_CORPUS_V2)]
        answer_corpus_v2: Option<PathBuf>,

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

        /// Optional OpenVINO GPU corpus-v2 diagnosis receipt to attach candidate-route blockers.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = OPENVINO_GPU_CORPUS_V2_DIAGNOSIS)]
        gpu_quality_diagnosis: Option<PathBuf>,

        /// Optional OpenVINO NPU corpus-v2 diagnosis receipt to attach candidate-route blockers.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = "lunar-lake-openvino-npu-corpus-v2-diagnosis.json")]
        npu_quality_diagnosis: Option<PathBuf>,

        /// Optional OpenVINO NPU cold-start diagnosis receipt to attach cold-route blockers.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = OPENVINO_NPU_COLD_START_DIAGNOSIS)]
        npu_cold_start_diagnosis: Option<PathBuf>,

        /// Optional OpenVINO NPU resident-session receipt to clear warm-route proof gaps.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = OPENVINO_NPU_RESIDENT_SESSION)]
        npu_resident_session: Option<PathBuf>,

        /// Optional OpenVINO NPU cache experiment receipt to attach cached-cold blockers.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = OPENVINO_NPU_CACHE_EXPERIMENT)]
        npu_cache_experiment: Option<PathBuf>,

        /// Optional OpenVINO generation-budget sensitivity receipt to attach exact-answer blockers.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = OPENVINO_GENERATION_BUDGET_SENSITIVITY)]
        openvino_budget_sensitivity: Option<PathBuf>,

        /// Optional Rust GGUF CPU profile-run receipt for profile-specific default-route timing.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long)]
        cpu_profile_run: Option<PathBuf>,

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

    /// Compare Rust GGUF CPU and OpenVINO CPU evidence for dense Qwen without changing routes.
    CpuSlmRuntimeComparison {
        /// Artifact root containing the 258V receipts to inspect.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// CPU dense-SLM resident-session receipt to inspect.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = CPU_SLM_RESIDENT_SESSION)]
        resident_session: PathBuf,

        /// OpenVINO CPU/GPU/NPU corpus-v2 receipt to inspect for OpenVINO CPU profile quality.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = DENSE_OV_CORPUS_V2)]
        openvino_corpus_v2: PathBuf,

        /// OpenVINO CPU/GPU/NPU phase-runner receipt to inspect for OpenVINO CPU PerfMetrics.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = DENSE_OV_PHASE)]
        openvino_phase_runner: PathBuf,

        /// Output JSON runtime-comparison receipt to file.
        #[arg(long, default_value = CPU_SLM_RUNTIME_COMPARISON)]
        json_out: PathBuf,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when the runtime comparison cannot classify Rust CPU vs OpenVINO CPU evidence.
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Diagnose OpenVINO dense-SLM corpus-v2 failures for one candidate route.
    OpenVinoQualityDiagnose {
        /// Artifact root containing the 258V receipts to inspect.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// OpenVINO CPU/GPU/NPU corpus-v2 receipt to diagnose.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = DENSE_OV_CORPUS_V2)]
        openvino_corpus_v2: PathBuf,

        /// Active answer corpus v2 fixture used to verify corpus receipt case alignment.
        /// Relative paths are resolved under artifact-root unless they exist from the current dir.
        #[arg(long, default_value = ANSWER_CORPUS_V2)]
        answer_corpus_v2: Option<PathBuf>,

        /// Runtime device to diagnose from the corpus receipt.
        #[arg(long, default_value = "GPU.0")]
        runtime_device: String,

        /// Output JSON diagnosis receipt to file.
        #[arg(long, default_value = OPENVINO_GPU_CORPUS_V2_DIAGNOSIS)]
        json_out: PathBuf,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when the diagnosis cannot classify the requested OpenVINO route.
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Decompose OpenVINO NPU cold-start evidence from committed receipts.
    NpuColdStartDiagnosis {
        /// Artifact root containing the 258V receipts to inspect.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// OpenVINO CPU/GPU/NPU phase-runner receipt to inspect for NPU load and hot-path metrics.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = DENSE_OV_PHASE)]
        openvino_phase_runner: PathBuf,

        /// OpenVINO CPU/GPU/NPU phase-comparison receipt to inspect for indexed NPU timing.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = DENSE_PHASE_COMPARISON)]
        phase_comparison: PathBuf,

        /// OpenVINO NPU operator-ask receipt to inspect.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = DENSE_OV_NPU_OPERATOR_ASK)]
        operator_ask: PathBuf,

        /// OpenVINO CPU/GPU/NPU corpus-v2 receipt to inspect for NPU quality/profile context.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = DENSE_OV_CORPUS_V2)]
        openvino_corpus_v2: PathBuf,

        /// Output JSON cold-start diagnosis receipt to file.
        #[arg(long, default_value = OPENVINO_NPU_COLD_START_DIAGNOSIS)]
        json_out: PathBuf,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when the NPU cold-start evidence cannot be classified.
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

        /// Require the captured sample to be a battery-mode sample for low_power evidence.
        #[arg(long, default_value_t = false)]
        require_battery: bool,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when memory and power context cannot be captured.
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Index low-power route evidence from telemetry and benchmark receipts without promotion.
    PowerProfile {
        /// Artifact root containing the 258V receipts to inspect.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// Route profile comparison receipt to inspect. Relative paths are resolved under artifact-root.
        #[arg(long, default_value = ROUTE_PROFILE_COMPARISON)]
        route_profile_comparison: PathBuf,

        /// Cold/warm benchmark qualification receipt to inspect. Relative paths are resolved under artifact-root.
        #[arg(long, default_value = COLD_WARM_PROFILE_BENCHMARK_FILE)]
        cold_warm_benchmark: PathBuf,

        /// Power/thermal context receipt to inspect. Relative paths are resolved under artifact-root.
        #[arg(long, default_value = POWER_THERMAL_CONTEXT_FILE)]
        telemetry_context: PathBuf,

        /// Optional battery-mode telemetry receipt captured for the same low_power route/profile matrix.
        /// Relative paths are resolved under artifact-root.
        #[arg(long)]
        battery_telemetry_context: Option<PathBuf>,

        /// Optional repeated-run battery-drain or energy proxy receipt for low_power route evidence.
        /// Relative paths are resolved under artifact-root.
        #[arg(long)]
        energy_proxy: Option<PathBuf>,

        /// Output JSON power-profile evidence receipt to file.
        #[arg(long, default_value = POWER_PROFILE_EVIDENCE_FILE)]
        json_out: PathBuf,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when the power-profile evidence cannot be indexed.
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Build a no-inference low-power battery-drain energy-proxy receipt.
    EnergyProxy {
        /// Artifact root for relative input and output paths.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// Telemetry receipt captured before the repeated low_power run.
        /// Relative paths are resolved under artifact-root.
        #[arg(long)]
        before_telemetry: PathBuf,

        /// Telemetry receipt captured after the repeated low_power run.
        /// Relative paths are resolved under artifact-root.
        #[arg(long)]
        after_telemetry: PathBuf,

        /// Route ID sampled by the repeated low_power run.
        #[arg(long, default_value = "dense_slm_openvino_npu_candidate")]
        route_id: String,

        /// Profile ID sampled by the repeated low_power run.
        #[arg(long, default_value = "low_power")]
        profile_id: String,

        /// Number of repeated asks or iterations covered by the sample.
        #[arg(long)]
        sample_count: u64,

        /// Output JSON low-power energy-proxy receipt to file.
        #[arg(long, default_value = LOW_POWER_ENERGY_PROXY_FILE)]
        json_out: PathBuf,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when battery-mode and energy-proxy evidence cannot be recorded.
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

    /// Index shared BitNet semantic fixes that require Lunar Lake reruns after merge.
    BitnetIntake {
        /// Artifact root containing the 258V receipts to inspect.
        #[arg(long, default_value = DEFAULT_ARTIFACT_ROOT)]
        artifact_root: PathBuf,

        /// Source-change ledger for shared BitNet semantic fixes.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = BITNET_SEMANTIC_SOURCE_CHANGES)]
        source_changes: PathBuf,

        /// Corrected Lunar Lake BitNet CPU reference bundle.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = BITNET_CPU_BUNDLE)]
        cpu_reference_bundle: PathBuf,

        /// Operator comparison receipt to check for route/readiness freshness.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = OPERATOR_COMPARISON)]
        operator_comparison: PathBuf,

        /// Output JSON BitNet semantic-intake receipt to file.
        #[arg(long, default_value = BITNET_SEMANTIC_INTAKE)]
        json_out: PathBuf,

        /// Override the receipt creation timestamp for reproducible committed receipts.
        #[arg(long)]
        created_utc: Option<String>,

        /// Fail when merged shared semantic fixes require Lunar Lake BitNet reruns.
        #[arg(long, default_value_t = false)]
        strict: bool,
    },

    /// Ask through an evidence-backed Lunar Lake route.
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

        /// Route profile comparison receipt used to fail closed on stale profile promotion.
        /// Relative paths are resolved under artifact-root.
        #[arg(long, default_value = ROUTE_PROFILE_COMPARISON)]
        route_profile_comparison: PathBuf,

        /// Workload profile to resolve when auto-routing is requested.
        #[arg(long, default_value = "ask_normal")]
        profile: String,

        /// Operator route to execute, or auto to select from the promotion ledger.
        /// Auto selection only uses ledger-promoted routes; OpenVINO candidate routes require
        /// an explicit --route plus matching --device request.
        #[arg(long, default_value = DEFAULT_ASK_ROUTE)]
        route: String,

        /// Dense Qwen model path. When omitted, the ask path resolves the local model path from
        /// committed Lunar Lake artifact manifests or phase receipts after route selection.
        #[arg(long)]
        model: Option<PathBuf>,

        /// Optional explicit tokenizer path.
        #[arg(long)]
        tokenizer: Option<PathBuf>,

        /// User question to answer.
        #[arg(
            long,
            visible_alias = "prompt",
            value_name = "TEXT",
            conflicts_with = "question_arg"
        )]
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_policy: Option<OperatorRoutePolicySummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_profile_evidence: Option<PowerProfileRegressionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal_temperature_availability: Option<ThermalTemperatureAvailabilityRegressionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked_ask_receipt: Option<BlockedAskRegressionSummary>,
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
pub struct OperatorRoutePolicySummary {
    pub route_promotion_ledger: String,
    pub route_profile_comparison: String,
    pub policy_ready: bool,
    pub promotion_ready: bool,
    pub profile_comparison_ready: bool,
    #[serde(default)]
    pub route_model_identity_ready: bool,
    #[serde(default)]
    pub route_model_identity_coverage: RouteModelIdentityCoverage,
    pub default_route_id: String,
    pub auto_route_policy_stage: String,
    pub hidden_fallback_allowed: bool,
    pub profile_scoped_promotion_only: bool,
    pub openvino_gpu_promoted_profiles: Vec<String>,
    pub openvino_npu_promoted_profiles: Vec<String>,
    pub profile_promotions: Vec<OperatorProfilePromotionSummary>,
    pub blocked_profiles: Vec<String>,
    pub gaps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OperatorProfilePromotionSummary {
    pub profile_id: String,
    pub promoted_route: Option<String>,
    pub profile_status: Option<String>,
    pub promotion_decision: Option<String>,
    pub route_blockers: Vec<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bitnet_semantic_intake: Option<BitnetSemanticIntakeRegressionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_profile_evidence: Option<PowerProfileRegressionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal_temperature_availability: Option<ThermalTemperatureAvailabilityRegressionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ask_short_ask_receipt: Option<OperatorAskRegressionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ask_normal_ask_receipt: Option<OperatorAskRegressionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warm_resident_ask_receipt: Option<OperatorAskRegressionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked_ask_receipt: Option<BlockedAskRegressionSummary>,
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
    #[serde(default)]
    pub bitnet_semantic_intake_indexed: bool,
    #[serde(default)]
    pub bitnet_cpu_reference_evidence_indexed: bool,
    #[serde(default)]
    pub bitnet_cpu_reference_evidence_ready: bool,
    #[serde(default)]
    pub power_profile_evidence_indexed: bool,
    #[serde(default)]
    pub thermal_temperature_availability_indexed: bool,
    #[serde(default)]
    pub thermal_temperature_available: bool,
    #[serde(default)]
    pub thermal_usable_temperature_reading_count: usize,
    #[serde(default)]
    pub arc_npu_bounded_evidence_indexed: bool,
    #[serde(default)]
    pub arc_npu_bounded_evidence_ready: bool,
    #[serde(default)]
    pub ask_short_ask_receipt_indexed: bool,
    #[serde(default)]
    pub ask_short_auto_ask_ready: bool,
    #[serde(default)]
    pub ask_normal_ask_receipt_indexed: bool,
    #[serde(default)]
    pub ask_normal_auto_ask_ready: bool,
    #[serde(default)]
    pub warm_resident_ask_receipt_indexed: bool,
    #[serde(default)]
    pub warm_resident_auto_ask_ready: bool,
    #[serde(default)]
    pub blocked_ask_receipt_indexed: bool,
    #[serde(default)]
    pub route_profile_model_identity_ready: bool,
    #[serde(default)]
    pub cold_warm_model_identity_ready: bool,
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
    pub timing_coverage: TimingApplicabilityCoverageSummary,
    #[serde(default)]
    pub durability_stability_proven: bool,
    #[serde(default)]
    pub route_promotion_scope: RoutePromotionScopeSummary,
    #[serde(default)]
    pub low_power_promotion_ready: bool,
    #[serde(default)]
    pub power_advantage_proven: bool,
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
            bitnet_semantic_intake_indexed: false,
            bitnet_cpu_reference_evidence_indexed: false,
            bitnet_cpu_reference_evidence_ready: false,
            power_profile_evidence_indexed: false,
            thermal_temperature_availability_indexed: false,
            thermal_temperature_available: false,
            thermal_usable_temperature_reading_count: 0,
            arc_npu_bounded_evidence_indexed: false,
            arc_npu_bounded_evidence_ready: false,
            ask_short_ask_receipt_indexed: false,
            ask_short_auto_ask_ready: false,
            ask_normal_ask_receipt_indexed: false,
            ask_normal_auto_ask_ready: false,
            warm_resident_ask_receipt_indexed: false,
            warm_resident_auto_ask_ready: false,
            blocked_ask_receipt_indexed: false,
            route_profile_model_identity_ready: false,
            cold_warm_model_identity_ready: false,
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
            timing_coverage: TimingApplicabilityCoverageSummary::default(),
            durability_stability_proven: false,
            route_promotion_scope: RoutePromotionScopeSummary::default(),
            low_power_promotion_ready: false,
            power_advantage_proven: false,
            strict_ready: false,
            gaps: vec![
                "answer corpus v2 is not indexed".to_string(),
                "route profile comparison is not indexed".to_string(),
                "cold/warm benchmark qualification is not indexed".to_string(),
                "durability bundle is not indexed".to_string(),
                "BitNet semantic intake is not indexed".to_string(),
                "low_power power-profile evidence is not indexed".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct RoutePromotionScopeSummary {
    pub openvino_gpu_promoted_profiles: Vec<String>,
    pub openvino_npu_promoted_profiles: Vec<String>,
    pub profile_scoped_promotion_only: bool,
    pub openvino_npu_remains_candidate: bool,
    pub unexpected_openvino_profile_promotions: Vec<String>,
    pub notes: Vec<String>,
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
    #[serde(default)]
    pub timing_coverage: TimingApplicabilityCoverageSummary,
    #[serde(default)]
    pub route_model_identity_coverage: RouteModelIdentityCoverage,
    #[serde(default)]
    pub route_model_identity_ready: bool,
    pub candidate_routes_remain_unpromoted: bool,
    pub benchmark_qualified_advantage_claimed: bool,
    pub fallback_observed: bool,
    pub gpu_npu_promotion_blockers: Vec<String>,
    #[serde(default)]
    pub gpu_npu_promotion_blocker_summary: Vec<PromotionBlockerSummary>,
    #[serde(default)]
    pub route_promotion_scope: RoutePromotionScopeSummary,
    pub regression_ready: bool,
    pub gaps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColdWarmRegressionSummary {
    pub path: String,
    pub benchmark_gate_ready: bool,
    pub profiles: Vec<String>,
    #[serde(default)]
    pub timing_coverage: TimingApplicabilityCoverageSummary,
    #[serde(default)]
    pub route_model_identity_coverage: RouteModelIdentityCoverage,
    #[serde(default)]
    pub route_model_identity_ready: bool,
    pub promoted_routes_have_critical_timing: bool,
    pub candidate_routes_remain_unpromoted: bool,
    pub fallback_observed: bool,
    pub benchmark_qualified_advantage_claimed: bool,
    pub telemetry_gaps: Vec<String>,
    #[serde(default)]
    pub route_promotion_scope: RoutePromotionScopeSummary,
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
pub struct BitnetSemanticIntakeRegressionSummary {
    pub path: String,
    pub intake_ready: bool,
    pub rerun_required: bool,
    pub pending_shared_change_count: usize,
    #[serde(default)]
    pub closed_shared_change_count: usize,
    pub merged_to_main_count: usize,
    pub stale_after_merged_count: usize,
    pub source_lanes: Vec<String>,
    pub pending_changes: Vec<String>,
    #[serde(default)]
    pub closed_changes: Vec<String>,
    pub required_reruns: Vec<String>,
    pub claim_boundary_preserved: bool,
    pub regression_ready: bool,
    pub gaps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PowerProfileRegressionSummary {
    pub path: String,
    pub power_profile_index_ready: bool,
    pub low_power_promotion_ready: bool,
    pub power_advantage_proven: bool,
    pub low_power_route_count: usize,
    pub low_power_routes_remain_unpromoted: bool,
    pub current_context_is_ac_only: bool,
    pub battery_mode_sample_recorded: bool,
    pub battery_sample_source: Option<String>,
    pub energy_proxy_recorded: bool,
    pub energy_proxy_source: Option<String>,
    pub thermal_context_recorded: bool,
    #[serde(default)]
    pub operator_runbook: Option<String>,
    #[serde(default)]
    pub next_required_evidence: Vec<String>,
    pub claim_boundary_preserved: bool,
    pub regression_ready: bool,
    pub gaps: Vec<String>,
    pub blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThermalTemperatureAvailabilityRegressionSummary {
    pub path: String,
    pub thermal_zone_visibility_available: bool,
    pub thermal_temperature_available: bool,
    pub usable_temperature_reading_count: usize,
    pub measured_temperature_claim: bool,
    pub telemetry_probe_executed: bool,
    pub claim_boundary_preserved: bool,
    pub regression_ready: bool,
    pub gaps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OperatorAskRegressionSummary {
    pub path: String,
    pub ask_receipt_ready: bool,
    pub profile_id: String,
    pub requested_device: String,
    pub requested_route: String,
    pub selected_route: String,
    pub selected_backend: String,
    pub runtime_api: String,
    pub promotion_status: String,
    pub route_profile_status: Option<String>,
    pub route_profile_blockers: Vec<String>,
    pub fallback_used: bool,
    pub answer_gate_passed: bool,
    pub openvino_candidate_route_executed: bool,
    pub new_inference_executed: bool,
    pub speedup_claim: bool,
    pub power_advantage_claim: bool,
    pub acceleration_claim: bool,
    pub bitnet_qk256_i2s_claim: bool,
    pub generated_token_ids_available: bool,
    pub source_run_receipt: Option<String>,
    pub regression_ready: bool,
    pub gaps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BlockedAskRegressionSummary {
    pub path: String,
    pub blocked_receipt_ready: bool,
    pub profile_id: String,
    pub requested_device: String,
    pub requested_route: String,
    pub route_selection_blocked: bool,
    pub model_path_required: bool,
    pub model_loaded: bool,
    pub model_resolution: String,
    pub candidate_routes: Vec<String>,
    pub why_not_cpu: Vec<String>,
    pub why_not_gpu: Vec<String>,
    pub why_not_npu: Vec<String>,
    #[serde(default)]
    pub operator_runbook: Option<String>,
    #[serde(default)]
    pub next_required_evidence: Vec<String>,
    pub new_inference_executed: bool,
    pub fallback_used: bool,
    pub route_promotion_changed: bool,
    pub speedup_claim: bool,
    pub power_advantage_claim: bool,
    pub acceleration_claim: bool,
    pub bitnet_qk256_i2s_claim: bool,
    pub route_selection_error: String,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ask_short_ask_receipt: Option<OperatorAskRegressionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ask_normal_ask_receipt: Option<OperatorAskRegressionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warm_resident_ask_receipt: Option<OperatorAskRegressionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked_ask_receipt: Option<BlockedAskRegressionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_policy: Option<OperatorRoutePolicySummary>,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct RouteModelIdentityCoverage {
    pub route_row_count: usize,
    pub route_rows_with_identity: usize,
    pub route_rows_with_model_hash: usize,
    pub route_rows_with_tokenizer_template: usize,
    #[serde(default)]
    pub route_rows_without_model_hash_with_known_gap: usize,
    pub all_route_rows_have_identity: bool,
    #[serde(default)]
    pub all_route_rows_have_tokenizer_template: bool,
    #[serde(default)]
    pub model_hash_or_explicit_gap_for_all_route_rows: bool,
    pub routes_without_model_hash: Vec<String>,
    #[serde(default)]
    pub routes_without_model_hash_missing_known_gap: Vec<String>,
    pub routes_without_tokenizer_template: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct RouteModelIdentity {
    pub identity_source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_receipt: Option<String>,
    pub selected_model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_artifact: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_token_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub known_gaps: Vec<String>,
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
    pub answer_corpus_v2_fixture: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_corpus_v2_receipt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openvino_corpus_v2_receipt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry_context_receipt: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub route_diagnosis_receipts: Vec<String>,
    pub profile_comparison_ready: bool,
    pub default_route_id: String,
    pub profiles: Vec<WorkloadProfileEvaluation>,
    #[serde(default)]
    pub timing_coverage: TimingApplicabilityCoverageSummary,
    #[serde(default)]
    pub route_model_identity_coverage: RouteModelIdentityCoverage,
    #[serde(default)]
    pub route_promotion_scope: RoutePromotionScopeSummary,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub promotion_blocker_summary: Vec<PromotionBlockerSummary>,
    pub gaps: Vec<String>,
    pub claim_boundary: ClaimBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct PromotionBlockerSummary {
    pub blocker: String,
    pub occurrence_count: u64,
    pub route_ids: Vec<String>,
    pub profile_ids: Vec<String>,
    pub next_action: String,
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
    #[serde(default)]
    pub ledger_route_status: String,
    #[serde(default)]
    pub selected_model: String,
    pub selected_backend: String,
    pub runtime_api: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_identity: Option<RouteModelIdentity>,
    pub fallback_used: Option<bool>,
    pub answer_gate_passed: Option<bool>,
    pub phase_timing_present: Option<bool>,
    pub timing: ProfileTimingSummary,
    pub timing_applicability: ProfileTimingApplicability,
    pub benchmark_qualified_advantage: bool,
    pub promotion_eligible_for_profile: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_quality: Option<ProfileQualityEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<BenchmarkTelemetry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_advantage_context: Option<ProfileRouteAdvantageContext>,
    pub evidence: Vec<String>,
    pub blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ProfileRouteAdvantageContext {
    pub baseline_route_id: String,
    pub baseline_route_status: String,
    pub baseline_total_response_ms: Option<f64>,
    pub route_total_response_ms: Option<f64>,
    pub route_to_baseline_total_response_ratio: Option<f64>,
    pub observed_total_response_lower_than_baseline: Option<bool>,
    pub benchmark_qualified: bool,
    pub qualification_status: String,
    pub qualification_blockers: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ProfileTimingApplicability {
    pub profile_id: String,
    pub required_prompt_tokens: String,
    pub required_output_tokens: String,
    pub measured_prompt_tokens: Option<u64>,
    pub measured_output_tokens: Option<u64>,
    pub timing_matches_profile: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TimingApplicabilityCoverageSummary {
    pub route_count: usize,
    pub profile_specific_route_count: usize,
    pub proxy_or_missing_route_count: usize,
    pub promotion_eligible_route_count: usize,
    pub promotion_eligible_profile_specific_route_count: usize,
    pub candidate_route_count: usize,
    pub candidate_proxy_or_missing_route_count: usize,
    pub promotion_eligible_routes_have_profile_specific_timing: bool,
    pub proxy_or_missing_timing_routes_blocked: bool,
    pub proxy_or_missing_routes: Vec<String>,
    pub promotion_eligible_proxy_or_missing_routes: Vec<String>,
    pub unblocked_proxy_or_missing_routes: Vec<String>,
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
    pub prompt_tokens: Option<u64>,
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
    #[serde(default)]
    pub timing_coverage: TimingApplicabilityCoverageSummary,
    #[serde(default)]
    pub route_model_identity_coverage: RouteModelIdentityCoverage,
    #[serde(default)]
    pub route_promotion_scope: RoutePromotionScopeSummary,
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
    #[serde(default)]
    pub ledger_route_status: String,
    #[serde(default)]
    pub selected_model: String,
    pub selected_backend: String,
    pub runtime_api: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_identity: Option<RouteModelIdentity>,
    pub fallback_used: Option<bool>,
    pub answer_gate_passed: Option<bool>,
    pub phase_timing_present: Option<bool>,
    pub timing: ProfileTimingSummary,
    #[serde(default)]
    pub timing_applicability: ProfileTimingApplicability,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_advantage_context: Option<ProfileRouteAdvantageContext>,
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
pub struct LunarLakeCpuSlmRuntimeComparison {
    pub schema_version: String,
    pub artifact_kind: String,
    pub proof_stage: String,
    pub created_utc: String,
    pub machine_id: String,
    pub artifact_root: String,
    pub source_receipts: CpuSlmRuntimeComparisonSources,
    pub model: CpuSlmAttributionModel,
    pub rust_gguf_cpu: CpuSlmRuntimeRouteSummary,
    pub openvino_cpu: CpuSlmRuntimeRouteSummary,
    pub profiles: Vec<CpuSlmRuntimeProfileComparison>,
    pub comparison_ready: bool,
    pub findings: Vec<String>,
    pub recommended_next_items: Vec<String>,
    pub gaps: Vec<String>,
    pub claim_boundary: CpuSlmPerfClaimBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CpuSlmRuntimeComparisonSources {
    pub resident_session_receipt: String,
    pub openvino_corpus_v2_receipt: String,
    pub openvino_phase_runner_receipt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CpuSlmRuntimeRouteSummary {
    pub route_id: String,
    pub selected_backend: String,
    pub runtime_api: String,
    pub selected_kernel_or_runtime: Option<String>,
    pub fallback_used: Option<bool>,
    pub answer_gate_passed: Option<bool>,
    pub quality_scope: String,
    pub timing_scope: String,
    pub load_or_construct_ms: Option<f64>,
    pub known_gaps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CpuSlmRuntimeProfileComparison {
    pub profile_id: String,
    pub rust_cpu: CpuSlmRuntimeProfileEvidence,
    pub openvino_cpu: CpuSlmRuntimeProfileEvidence,
    pub openvino_to_rust_total_ratio: Option<f64>,
    pub status: String,
    pub blockers: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CpuSlmRuntimeProfileEvidence {
    pub route_id: String,
    pub selected_backend: String,
    pub runtime_api: String,
    pub fallback_used: Option<bool>,
    pub answer_gate_passed: Option<bool>,
    pub cases_total: Option<u64>,
    pub cases_passed: Option<u64>,
    pub cases_failed: Option<u64>,
    pub timing_ms: CpuSlmResidentMetricSummary,
    pub time_to_first_token_ms: CpuSlmResidentMetricSummary,
    pub tokenize_ms: CpuSlmResidentMetricSummary,
    pub generated_tokens: CpuSlmResidentMetricSummary,
    pub throughput_tokens_per_s_mean: Option<f64>,
    pub timing_source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LunarLakeOpenVinoCorpusV2Diagnosis {
    pub schema_version: String,
    pub artifact_kind: String,
    pub proof_stage: String,
    pub created_utc: String,
    pub machine_id: String,
    pub artifact_root: String,
    pub openvino_corpus_v2_receipt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer_corpus_v2_fixture: Option<String>,
    pub requested_runtime_device: String,
    pub selected_backend: Option<String>,
    pub runtime_api: Option<String>,
    pub runtime_device: Option<String>,
    pub backend_lane: Option<String>,
    pub selected_kernel_or_runtime: Option<String>,
    pub fallback_used: Option<bool>,
    pub promotion_status: Option<String>,
    pub quality_summary: CorpusV2QualitySummary,
    pub profile_diagnoses: Vec<CorpusV2ProfileDiagnosis>,
    pub failed_cases: Vec<CorpusV2FailedCaseDiagnosis>,
    pub case_alignment: CorpusV2CaseAlignmentDiagnosis,
    pub generated_token_visibility: OpenVinoGeneratedTokenVisibility,
    pub route_blocked: bool,
    pub blocker_summary: Vec<String>,
    pub recommended_next_actions: Vec<String>,
    pub diagnosis_ready: bool,
    pub gaps: Vec<String>,
    pub claim_boundary: CorpusV2DiagnosisClaimBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorpusV2CaseAlignmentDiagnosis {
    pub fixture_verified: bool,
    pub expected_case_count: Option<u64>,
    pub observed_case_count: u64,
    pub missing_case_ids: Vec<String>,
    pub stale_or_unexpected_case_ids: Vec<String>,
    pub aligned_with_active_fixture: Option<bool>,
    pub blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenVinoGeneratedTokenVisibility {
    pub direct_generated_token_ids_available: bool,
    pub retokenized_generated_ids_used: bool,
    pub sources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LunarLakeNpuColdStartDiagnosis {
    pub schema_version: String,
    pub artifact_kind: String,
    pub proof_stage: String,
    pub created_utc: String,
    pub machine_id: String,
    pub artifact_root: String,
    pub source_receipts: NpuColdStartSources,
    pub route: NpuColdStartRouteIdentity,
    pub cold_start: NpuColdStartEvidence,
    pub hot_path: NpuHotPathEvidence,
    pub corpus_v2_context: NpuCorpusV2Context,
    pub diagnosis_ready: bool,
    pub findings: Vec<String>,
    pub recommended_next_items: Vec<String>,
    pub gaps: Vec<String>,
    pub claim_boundary: NpuColdStartClaimBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NpuColdStartSources {
    pub openvino_phase_runner_receipt: String,
    pub phase_comparison_receipt: String,
    pub operator_ask_receipt: String,
    pub openvino_corpus_v2_receipt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NpuColdStartRouteIdentity {
    pub route_id: String,
    pub requested_backend: Option<String>,
    pub selected_backend: Option<String>,
    pub runtime_api: Option<String>,
    pub runtime_device: Option<String>,
    pub resolved_device: Option<String>,
    pub backend_lane: Option<String>,
    pub selected_kernel_or_runtime: Option<String>,
    pub fallback_used: Option<bool>,
    pub promotion_status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NpuColdStartEvidence {
    pub classification: String,
    pub cold_load_dominant: bool,
    pub samples: Vec<NpuTimingSample>,
    pub pipeline_or_load_ms: CpuSlmResidentMetricSummary,
    pub generation_wall_ms: CpuSlmResidentMetricSummary,
    pub first_token_or_text_chunk_ms: CpuSlmResidentMetricSummary,
    pub throughput_tokens_per_s: CpuSlmResidentMetricSummary,
    pub operator_load_to_generation_ratio: Option<f64>,
    pub phase_runner_load_to_generation_ratio: Option<f64>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NpuTimingSample {
    pub source: String,
    pub evidence_scope: String,
    pub pipeline_construct_wall_ms: Option<f64>,
    pub openvino_load_time_ms: Option<f64>,
    pub generation_wall_ms: Option<f64>,
    pub case_elapsed_ms_sum: Option<f64>,
    pub first_streamed_text_chunk_ms: Option<f64>,
    pub openvino_time_to_first_token_ms: Option<f64>,
    pub openvino_generate_ms: Option<f64>,
    pub openvino_inference_ms: Option<f64>,
    pub openvino_tokenization_ms: Option<f64>,
    pub throughput_tokens_per_s: Option<f64>,
    pub generated_tokens: Option<u64>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NpuHotPathEvidence {
    pub bounded_answer_gate_passed: Option<bool>,
    pub fallback_used: Option<bool>,
    pub generation_wall_ms: CpuSlmResidentMetricSummary,
    pub first_text_chunk_ms: CpuSlmResidentMetricSummary,
    pub openvino_time_to_first_token_ms: CpuSlmResidentMetricSummary,
    pub throughput_tokens_per_s: CpuSlmResidentMetricSummary,
    pub generated_tokens: CpuSlmResidentMetricSummary,
    pub hot_path_interesting: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NpuCorpusV2Context {
    pub cases_total: Option<u64>,
    pub passed: Option<u64>,
    pub failed: Option<u64>,
    pub route_blocked_by_quality: bool,
    pub failed_profiles: Vec<String>,
    pub failed_categories: Vec<String>,
    pub direct_generated_token_ids_available: bool,
    pub generated_token_id_source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NpuColdStartClaimBoundary {
    pub diagnostic_only: bool,
    pub new_inference_executed: bool,
    pub route_promotion_changed: bool,
    pub broad_quality_claim: bool,
    pub speedup_claim: bool,
    pub power_advantage_claim: bool,
    pub acceleration_claim: bool,
    pub native_npu_inference_claim: bool,
    pub npu_dynamic_decode_claim: bool,
    pub beam_or_parallel_sampling_claim: bool,
    pub bitnet_qk256_i2s_behavior_changed: bool,
    pub dense_slm_as_bitnet_proof: bool,
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
    pub capture_requirements: TelemetryCaptureRequirements,
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
pub struct TelemetryCaptureRequirements {
    pub battery_mode_required: bool,
    pub battery_mode_sample_recorded: bool,
    pub requirement_satisfied: bool,
    pub status: String,
    pub gaps: Vec<String>,
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
pub struct LunarLakePowerProfileEvidence {
    pub schema_version: String,
    pub artifact_kind: String,
    pub proof_stage: String,
    pub created_utc: String,
    pub machine_id: String,
    pub artifact_root: String,
    pub route_profile_comparison_receipt: String,
    pub cold_warm_benchmark_receipt: String,
    pub telemetry_context_receipt: String,
    pub battery_telemetry_context_receipt: Option<String>,
    pub energy_proxy_receipt: Option<String>,
    pub telemetry: PowerProfileTelemetrySummary,
    pub low_power_routes: Vec<PowerProfileRouteEvidence>,
    pub power_profile_index_ready: bool,
    pub low_power_promotion_ready: bool,
    pub power_advantage_proven: bool,
    pub gaps: Vec<String>,
    #[serde(default)]
    pub operator_runbook: Option<String>,
    pub next_required_evidence: Vec<String>,
    pub claim_boundary: PowerProfileClaimBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PowerProfileTelemetrySummary {
    pub memory_context_recorded: bool,
    pub power_context_recorded: bool,
    pub thermal_context_recorded: bool,
    pub active_scheme: Option<String>,
    pub battery_status: Option<String>,
    pub ac_power_inferred: Option<bool>,
    pub thermal_zones_visible: Option<u64>,
    pub thermal_temperature_count: usize,
    pub current_context_is_ac_only: bool,
    pub battery_mode_sample_recorded: bool,
    pub battery_sample_source: Option<String>,
    pub energy_proxy_recorded: bool,
    pub energy_proxy_source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PowerProfileRouteEvidence {
    pub route_id: String,
    pub route_status: String,
    pub ledger_route_status: String,
    pub selected_backend: String,
    pub runtime_api: String,
    pub fallback_used: Option<bool>,
    pub answer_gate_passed: Option<bool>,
    pub total_response_ms: Option<f64>,
    pub throughput_tokens_per_s: Option<f64>,
    pub benchmark_qualified_advantage: bool,
    pub power_related_blockers: Vec<String>,
    pub all_blockers: Vec<String>,
    pub power_promotion_ready: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PowerProfileClaimBoundary {
    pub new_inference_executed: bool,
    pub route_promotion_changed: bool,
    pub speedup_claim: bool,
    pub power_advantage_claim: bool,
    pub acceleration_claim: bool,
    pub native_npu_inference_claim: bool,
    pub bitnet_qk256_i2s_behavior_changed: bool,
    pub hidden_fallback_allowed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LunarLakeLowPowerEnergyProxy {
    pub schema_version: String,
    pub artifact_kind: String,
    pub proof_stage: String,
    pub created_utc: String,
    pub machine_id: String,
    pub artifact_root: String,
    pub before_telemetry_context_receipt: String,
    pub after_telemetry_context_receipt: String,
    pub route_id: String,
    pub profile_id: String,
    pub sample_count: u64,
    pub before_battery_status: Option<String>,
    pub after_battery_status: Option<String>,
    pub before_charge_percent: Option<i64>,
    pub after_charge_percent: Option<i64>,
    pub charge_delta_percent: Option<i64>,
    pub before_ac_power_inferred: Option<bool>,
    pub after_ac_power_inferred: Option<bool>,
    pub battery_mode_sample_recorded: bool,
    pub energy_proxy_recorded: bool,
    pub gaps: Vec<String>,
    pub claim_boundary: PowerProfileClaimBoundary,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BitnetSemanticSourceChanges {
    pub schema_version: String,
    pub artifact_kind: String,
    pub created_utc: String,
    pub machine_id: String,
    pub changes: Vec<BitnetSemanticSourceChange>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BitnetSemanticSourceChange {
    pub source_lane: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_pr: Option<u64>,
    pub title: String,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_sha: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge_sha: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merged_at_utc: Option<String>,
    #[serde(default)]
    pub semantic_scope: Vec<String>,
    #[serde(default)]
    pub requires_lunar_lake_rerun_when_merged_to_main: bool,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LunarLakeBitnetSemanticIntake {
    pub schema_version: String,
    pub artifact_kind: String,
    pub proof_stage: String,
    pub created_utc: String,
    pub machine_id: String,
    pub artifact_root: String,
    pub source_changes_receipt: String,
    pub cpu_reference_bundle: String,
    pub operator_comparison: String,
    pub source_change_summary: BitnetSemanticSourceChangeSummary,
    pub lunar_lake_evidence: BitnetSemanticLunarLakeEvidence,
    pub changes: Vec<BitnetSemanticChangeIntake>,
    pub rerun_required: bool,
    pub required_reruns: Vec<String>,
    pub intake_ready: bool,
    pub gaps: Vec<String>,
    pub claim_boundary: BitnetSemanticIntakeClaimBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BitnetSemanticSourceChangeSummary {
    pub total_change_count: usize,
    pub pending_shared_change_count: usize,
    #[serde(default)]
    pub closed_shared_change_count: usize,
    pub merged_to_main_count: usize,
    pub stale_after_merged_count: usize,
    pub source_lanes: Vec<String>,
    pub pending_changes: Vec<String>,
    #[serde(default)]
    pub closed_changes: Vec<String>,
    pub merged_changes: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BitnetSemanticLunarLakeEvidence {
    pub cpu_reference_bundle_created_utc: Option<String>,
    pub operator_comparison_created_utc: Option<String>,
    pub evidence_cutoff_utc: Option<String>,
    pub cpu_reference_bundle_path: String,
    pub operator_comparison_path: String,
    pub evidence_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BitnetSemanticChangeIntake {
    pub source_lane: String,
    pub source_pr: Option<u64>,
    pub title: String,
    pub status: String,
    pub semantic_scope: Vec<String>,
    pub requires_lunar_lake_rerun_when_merged_to_main: bool,
    pub merged_at_utc: Option<String>,
    pub stale_after_cpu_reference: bool,
    pub stale_after_operator_comparison: bool,
    pub lunar_lake_rerun_required: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BitnetSemanticIntakeClaimBoundary {
    pub new_inference_executed: bool,
    pub route_promotion_changed: bool,
    pub answer_quality_claim: bool,
    pub speedup_claim: bool,
    pub acceleration_claim: bool,
    pub arc_or_npu_bitnet_claim: bool,
    pub qk256_behavior_changed: bool,
    pub dense_slm_as_bitnet_proof: bool,
    pub hidden_fallback_allowed: bool,
}

impl LunarLakeCommand {
    pub async fn execute(&self) -> Result<()> {
        match &self.action {
            LunarLakeAction::Validate {
                artifact_root,
                route_promotion_ledger,
                route_profile_comparison,
                power_profile_evidence,
                thermal_temperature_availability,
                blocked_ask_receipt,
                json_out,
                created_utc,
                strict,
            } => {
                let receipt = match created_utc {
                    Some(created_utc) => {
                        let created_utc = normalize_created_utc(created_utc)?;
                        build_operator_readiness_receipt_with_created_utc_and_route_policy(
                            artifact_root,
                            created_utc,
                            route_promotion_ledger.as_deref(),
                            route_profile_comparison.as_deref(),
                            power_profile_evidence.as_deref(),
                            thermal_temperature_availability.as_deref(),
                            blocked_ask_receipt.as_deref(),
                        )?
                    }
                    None => build_operator_readiness_receipt_with_route_policy(
                        artifact_root,
                        route_promotion_ledger.as_deref(),
                        route_profile_comparison.as_deref(),
                        power_profile_evidence.as_deref(),
                        thermal_temperature_availability.as_deref(),
                        blocked_ask_receipt.as_deref(),
                    )?,
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
                bitnet_semantic_intake,
                power_profile_evidence,
                thermal_temperature_availability,
                ask_short_ask_receipt,
                ask_normal_ask_receipt,
                warm_resident_ask_receipt,
                blocked_ask_receipt,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt =
                    build_regression_bundle_with_created_utc_and_inputs_and_power_profile_and_warm_ask(
                        artifact_root,
                        operator_receipt,
                        answer_corpus_v2.as_deref(),
                        route_profile_comparison.as_deref(),
                        cold_warm_benchmark.as_deref(),
                        durability_bundle.as_deref(),
                        bitnet_semantic_intake.as_deref(),
                        power_profile_evidence.as_deref(),
                        thermal_temperature_availability.as_deref(),
                        ask_short_ask_receipt.as_deref(),
                        ask_normal_ask_receipt.as_deref(),
                        warm_resident_ask_receipt.as_deref(),
                        blocked_ask_receipt.as_deref(),
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
                route_profile_comparison,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_route_promotion_ledger_with_created_utc_and_profile_evidence(
                    artifact_root,
                    operator_receipt,
                    comparison_receipt,
                    route_profile_comparison.as_deref(),
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
                answer_corpus_v2,
                cpu_corpus_v2,
                openvino_corpus_v2,
                telemetry_context,
                gpu_quality_diagnosis,
                npu_quality_diagnosis,
                npu_cold_start_diagnosis,
                npu_resident_session,
                npu_cache_experiment,
                openvino_budget_sensitivity,
                cpu_profile_run,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt =
                    build_route_profile_comparison_with_created_utc_and_budget_diagnostics(
                        artifact_root,
                        promotion_ledger,
                        phase_comparison,
                        answer_corpus_v2.as_deref(),
                        cpu_corpus_v2.as_deref(),
                        openvino_corpus_v2.as_deref(),
                        telemetry_context.as_deref(),
                        gpu_quality_diagnosis.as_deref(),
                        npu_quality_diagnosis.as_deref(),
                        npu_cold_start_diagnosis.as_deref(),
                        npu_resident_session.as_deref(),
                        npu_cache_experiment.as_deref(),
                        openvino_budget_sensitivity.as_deref(),
                        cpu_profile_run.as_deref(),
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
            LunarLakeAction::CpuSlmRuntimeComparison {
                artifact_root,
                resident_session,
                openvino_corpus_v2,
                openvino_phase_runner,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_cpu_slm_runtime_comparison_with_created_utc(
                    artifact_root,
                    resident_session,
                    openvino_corpus_v2,
                    openvino_phase_runner,
                    created_utc,
                )?;
                let json_out = resolve_receipt_path(artifact_root, json_out);
                write_or_print_cpu_slm_runtime_comparison(&receipt, Some(&json_out))?;
                if *strict && !receipt.comparison_ready {
                    bail!(
                        "Lunar Lake CPU dense SLM runtime comparison failed: {}",
                        receipt.gaps.join("; ")
                    );
                }
                Ok(())
            }
            LunarLakeAction::OpenVinoQualityDiagnose {
                artifact_root,
                openvino_corpus_v2,
                answer_corpus_v2,
                runtime_device,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_openvino_corpus_v2_diagnosis_with_created_utc(
                    artifact_root,
                    openvino_corpus_v2,
                    answer_corpus_v2.as_deref(),
                    runtime_device,
                    created_utc,
                )?;
                let json_out = resolve_receipt_path(artifact_root, json_out);
                write_or_print_openvino_corpus_v2_diagnosis(&receipt, Some(&json_out))?;
                if *strict && !receipt.diagnosis_ready {
                    bail!(
                        "Lunar Lake OpenVINO corpus-v2 diagnosis failed: {}",
                        receipt.gaps.join("; ")
                    );
                }
                Ok(())
            }
            LunarLakeAction::NpuColdStartDiagnosis {
                artifact_root,
                openvino_phase_runner,
                phase_comparison,
                operator_ask,
                openvino_corpus_v2,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_npu_cold_start_diagnosis_with_created_utc(
                    artifact_root,
                    openvino_phase_runner,
                    phase_comparison,
                    operator_ask,
                    openvino_corpus_v2,
                    created_utc,
                )?;
                let json_out = resolve_receipt_path(artifact_root, json_out);
                write_or_print_npu_cold_start_diagnosis(&receipt, Some(&json_out))?;
                if *strict && !receipt.diagnosis_ready {
                    bail!(
                        "Lunar Lake OpenVINO NPU cold-start diagnosis failed: {}",
                        receipt.gaps.join("; ")
                    );
                }
                Ok(())
            }
            LunarLakeAction::TelemetryContext {
                artifact_root,
                json_out,
                require_battery,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_telemetry_context_with_created_utc_and_requirements(
                    artifact_root,
                    created_utc,
                    *require_battery,
                );
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
                if *strict
                    && *require_battery
                    && !receipt.capture_requirements.requirement_satisfied
                {
                    bail!(
                        "Lunar Lake telemetry context capture failed battery-mode requirement: {}",
                        receipt.capture_requirements.gaps.join("; ")
                    );
                }
                Ok(())
            }
            LunarLakeAction::PowerProfile {
                artifact_root,
                route_profile_comparison,
                cold_warm_benchmark,
                telemetry_context,
                battery_telemetry_context,
                energy_proxy,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_power_profile_evidence_with_created_utc(
                    artifact_root,
                    route_profile_comparison,
                    cold_warm_benchmark,
                    telemetry_context,
                    battery_telemetry_context.as_deref(),
                    energy_proxy.as_deref(),
                    created_utc,
                )?;
                let json_out = resolve_receipt_path(artifact_root, json_out);
                write_or_print_power_profile_evidence(&receipt, Some(&json_out))?;
                if *strict && !receipt.power_profile_index_ready {
                    bail!("Lunar Lake power-profile evidence failed: {}", receipt.gaps.join("; "));
                }
                Ok(())
            }
            LunarLakeAction::EnergyProxy {
                artifact_root,
                before_telemetry,
                after_telemetry,
                route_id,
                profile_id,
                sample_count,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_low_power_energy_proxy_with_created_utc(
                    artifact_root,
                    before_telemetry,
                    after_telemetry,
                    route_id.clone(),
                    profile_id.clone(),
                    *sample_count,
                    created_utc,
                )?;
                let json_out = resolve_receipt_path(artifact_root, json_out);
                write_or_print_low_power_energy_proxy(&receipt, Some(&json_out))?;
                if *strict && !receipt.gaps.is_empty() {
                    bail!("Lunar Lake low-power energy proxy failed: {}", receipt.gaps.join("; "));
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
            LunarLakeAction::BitnetIntake {
                artifact_root,
                source_changes,
                cpu_reference_bundle,
                operator_comparison,
                json_out,
                created_utc,
                strict,
            } => {
                let created_utc = match created_utc {
                    Some(created_utc) => normalize_created_utc(created_utc)?,
                    None => chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                };
                let receipt = build_bitnet_semantic_intake_with_created_utc(
                    artifact_root,
                    source_changes,
                    cpu_reference_bundle,
                    operator_comparison,
                    created_utc,
                )?;
                let json_out = resolve_receipt_path(artifact_root, json_out);
                write_or_print_bitnet_semantic_intake(&receipt, Some(&json_out))?;
                if *strict && !receipt.intake_ready {
                    bail!("Lunar Lake BitNet semantic intake failed: {}", receipt.gaps.join("; "));
                }
                Ok(())
            }
            LunarLakeAction::Ask { .. } => {
                bail!("lunar-lake ask must be handled by the CLI runtime")
            }
        }
    }
}

#[cfg(test)]
pub fn build_operator_readiness_receipt(root: &Path) -> Result<LunarLakeOperatorReceipt> {
    build_operator_readiness_receipt_with_created_utc(
        root,
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
    )
}

pub fn build_operator_readiness_receipt_with_route_policy(
    root: &Path,
    route_promotion_ledger: Option<&Path>,
    route_profile_comparison: Option<&Path>,
    power_profile_evidence: Option<&Path>,
    thermal_temperature_availability: Option<&Path>,
    blocked_ask_receipt: Option<&Path>,
) -> Result<LunarLakeOperatorReceipt> {
    build_operator_readiness_receipt_with_created_utc_and_route_policy(
        root,
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        route_promotion_ledger,
        route_profile_comparison,
        power_profile_evidence,
        thermal_temperature_availability,
        blocked_ask_receipt,
    )
}

#[cfg(test)]
pub fn build_operator_readiness_receipt_with_created_utc(
    root: &Path,
    created_utc: String,
) -> Result<LunarLakeOperatorReceipt> {
    build_operator_readiness_receipt_with_created_utc_and_route_policy(
        root,
        created_utc,
        None,
        None,
        None,
        None,
        None,
    )
}

pub fn build_operator_readiness_receipt_with_created_utc_and_route_policy(
    root: &Path,
    created_utc: String,
    route_promotion_ledger: Option<&Path>,
    route_profile_comparison: Option<&Path>,
    power_profile_evidence: Option<&Path>,
    thermal_temperature_availability: Option<&Path>,
    blocked_ask_receipt: Option<&Path>,
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
    let route_policy = match (route_promotion_ledger, route_profile_comparison) {
        (Some(ledger_path), Some(comparison_path)) => {
            let summary = inspect_operator_route_policy(root, ledger_path, comparison_path)?;
            if !summary.policy_ready {
                gaps.push(format!(
                    "route policy evidence is not operator-ready: {}",
                    summary.gaps.join(", ")
                ));
            }
            Some(summary)
        }
        (Some(_), None) | (None, Some(_)) => {
            gaps.push(
                "route policy evidence requires both route promotion ledger and route profile comparison"
                    .to_string(),
            );
            None
        }
        (None, None) => None,
    };
    let power_profile_evidence = if let Some(path) = power_profile_evidence {
        let path = resolve_receipt_path(root, path);
        if path.exists() {
            let summary = inspect_power_profile_regression(&path)?;
            if !summary.regression_ready {
                gaps.push(format!(
                    "power-profile evidence is not readiness-ready: {}",
                    summary.gaps.join(", ")
                ));
            }
            if summary.power_advantage_proven || summary.low_power_promotion_ready {
                gaps.push(
                    "power-profile evidence cannot claim low_power promotion or power advantage in readiness"
                        .to_string(),
                );
            }
            Some(summary)
        } else {
            gaps.push(format!("missing low-power power-profile evidence: {}", path.display()));
            None
        }
    } else {
        None
    };
    let thermal_temperature_availability = if let Some(path) = thermal_temperature_availability {
        let path = resolve_receipt_path(root, path);
        if path.exists() {
            let summary = inspect_thermal_temperature_availability_regression(&path)?;
            if !summary.regression_ready {
                gaps.push(format!(
                    "thermal temperature availability is not readiness-ready: {}",
                    summary.gaps.join(", ")
                ));
            }
            if summary.measured_temperature_claim && summary.usable_temperature_reading_count == 0 {
                gaps.push(
                        "thermal temperature availability claims measured temperatures without usable readings"
                            .to_string(),
                    );
            }
            Some(summary)
        } else {
            gaps.push(format!(
                "missing thermal temperature availability evidence: {}",
                path.display()
            ));
            None
        }
    } else {
        None
    };
    let blocked_ask_receipt = if let Some(path) = blocked_ask_receipt {
        let path = resolve_receipt_path(root, path);
        if path.exists() {
            let summary = inspect_blocked_ask_regression(&path)?;
            if !summary.regression_ready {
                gaps.push(format!(
                    "blocked low_power ask receipt is not readiness-ready: {}",
                    summary.gaps.join(", ")
                ));
            }
            Some(summary)
        } else {
            gaps.push(format!("missing blocked low_power ask receipt: {}", path.display()));
            None
        }
    } else {
        None
    };

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
        route_policy,
        power_profile_evidence,
        thermal_temperature_availability,
        blocked_ask_receipt,
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

fn inspect_operator_route_policy(
    root: &Path,
    ledger_path: &Path,
    comparison_path: &Path,
) -> Result<OperatorRoutePolicySummary> {
    let ledger_path = resolve_receipt_path(root, ledger_path);
    let comparison_path = resolve_receipt_path(root, comparison_path);
    let ledger_path_string = path_string(&ledger_path);
    let comparison_path_string = path_string(&comparison_path);
    let mut gaps = Vec::new();

    let ledger = if ledger_path.exists() {
        Some(read_json_receipt::<LunarLakeRoutePromotionLedger>(&ledger_path)?)
    } else {
        gaps.push(format!("missing route promotion ledger: {}", ledger_path.display()));
        None
    };
    let comparison = if comparison_path.exists() {
        Some(read_json_receipt::<LunarLakeRouteProfileComparison>(&comparison_path)?)
    } else {
        gaps.push(format!("missing route profile comparison: {}", comparison_path.display()));
        None
    };

    let promotion_ready = ledger.as_ref().is_some_and(|ledger| ledger.promotion_ready);
    if let Some(ledger) = &ledger {
        if !ledger.promotion_ready {
            gaps.push("route promotion ledger is not promotion_ready".to_string());
        }
        if ledger.auto_route_policy.hidden_fallback_allowed {
            gaps.push("route promotion ledger allows hidden fallback".to_string());
        }
        if !ledger.auto_route_policy.candidate_routes_require_profile_promotion {
            gaps.push(
                "route promotion ledger does not require profile promotion for candidates"
                    .to_string(),
            );
        }
        if !ledger.auto_route_policy.route_reason_required {
            gaps.push("route promotion ledger does not require route reasons".to_string());
        }
    }

    let profile_comparison_ready =
        comparison.as_ref().is_some_and(|comparison| comparison.profile_comparison_ready);
    if let Some(comparison) = &comparison {
        if !comparison.profile_comparison_ready {
            gaps.push("route profile comparison is not profile_comparison_ready".to_string());
        }
        if comparison.claim_boundary.hidden_fallback_allowed {
            gaps.push("route profile comparison allows hidden fallback".to_string());
        }
        if !comparison.route_promotion_scope.profile_scoped_promotion_only {
            gaps.push("route profile comparison is not profile-scoped".to_string());
        }
        if !comparison.route_promotion_scope.unexpected_openvino_profile_promotions.is_empty() {
            gaps.push(format!(
                "unexpected OpenVINO profile promotions: {}",
                comparison.route_promotion_scope.unexpected_openvino_profile_promotions.join(", ")
            ));
        }
        let unexpected_npu_profiles = comparison
            .route_promotion_scope
            .openvino_npu_promoted_profiles
            .iter()
            .filter(|profile| !OPENVINO_NPU_PROFILE_PROMOTION_TARGETS.contains(&profile.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        if !unexpected_npu_profiles.is_empty() {
            gaps.push(format!(
                "OpenVINO NPU unexpectedly promoted for profiles: {}",
                unexpected_npu_profiles.join(", ")
            ));
        }
    }
    let route_model_identity_coverage = comparison
        .as_ref()
        .map(|comparison| route_model_identity_coverage(&comparison.profiles))
        .unwrap_or_default();
    let route_model_identity_ready =
        route_model_identity_coverage_ready(&route_model_identity_coverage);
    if comparison.is_some() {
        append_route_model_identity_coverage_gaps(
            "operator route policy",
            &route_model_identity_coverage,
            &mut gaps,
        );
    }

    let default_route_id = ledger
        .as_ref()
        .map(|ledger| ledger.default_route_id.clone())
        .or_else(|| comparison.as_ref().map(|comparison| comparison.default_route_id.clone()))
        .unwrap_or_else(|| DEFAULT_ASK_ROUTE.to_string());
    let auto_route_policy_stage = ledger
        .as_ref()
        .map(|ledger| ledger.auto_route_policy.policy_stage.clone())
        .unwrap_or_else(|| "missing_route_promotion_ledger".to_string());
    let hidden_fallback_allowed = ledger
        .as_ref()
        .map(|ledger| ledger.auto_route_policy.hidden_fallback_allowed)
        .unwrap_or(true)
        || comparison
            .as_ref()
            .map(|comparison| comparison.claim_boundary.hidden_fallback_allowed)
            .unwrap_or(true);
    let profile_scoped_promotion_only = comparison
        .as_ref()
        .map(|comparison| comparison.route_promotion_scope.profile_scoped_promotion_only)
        .unwrap_or(false);
    let openvino_gpu_promoted_profiles = comparison
        .as_ref()
        .map(|comparison| comparison.route_promotion_scope.openvino_gpu_promoted_profiles.clone())
        .or_else(|| {
            ledger.as_ref().map(|ledger| {
                promoted_profiles_for_route(ledger, "dense_slm_openvino_gpu_candidate")
            })
        })
        .unwrap_or_default();
    let openvino_npu_promoted_profiles = comparison
        .as_ref()
        .map(|comparison| comparison.route_promotion_scope.openvino_npu_promoted_profiles.clone())
        .or_else(|| {
            ledger.as_ref().map(|ledger| {
                promoted_profiles_for_route(ledger, "dense_slm_openvino_npu_candidate")
            })
        })
        .unwrap_or_default();

    let profile_promotions: Vec<OperatorProfilePromotionSummary> = comparison
        .as_ref()
        .map(|comparison| {
            comparison
                .profiles
                .iter()
                .map(|profile| {
                    let route_blockers = profile
                        .route_evidence
                        .iter()
                        .flat_map(|route| {
                            route
                                .blockers
                                .iter()
                                .map(move |blocker| format!("{}: {}", route.route_id, blocker))
                        })
                        .collect();
                    OperatorProfilePromotionSummary {
                        profile_id: profile.profile_id.clone(),
                        promoted_route: profile.promoted_route.clone(),
                        profile_status: Some(profile.profile_status.clone()),
                        promotion_decision: Some(profile.promotion_decision.clone()),
                        route_blockers,
                    }
                })
                .collect()
        })
        .or_else(|| {
            ledger.as_ref().map(|ledger| {
                ledger
                    .workload_profiles
                    .iter()
                    .map(|profile| OperatorProfilePromotionSummary {
                        profile_id: profile.profile_id.clone(),
                        promoted_route: profile.promoted_route.clone(),
                        profile_status: None,
                        promotion_decision: None,
                        route_blockers: Vec::new(),
                    })
                    .collect()
            })
        })
        .unwrap_or_default();
    let blocked_profiles = profile_promotions
        .iter()
        .filter(|profile| profile.promoted_route.is_none())
        .map(|profile| profile.profile_id.clone())
        .collect();

    Ok(OperatorRoutePolicySummary {
        route_promotion_ledger: ledger_path_string,
        route_profile_comparison: comparison_path_string,
        policy_ready: gaps.is_empty() && promotion_ready && profile_comparison_ready,
        promotion_ready,
        profile_comparison_ready,
        route_model_identity_ready,
        route_model_identity_coverage,
        default_route_id,
        auto_route_policy_stage,
        hidden_fallback_allowed,
        profile_scoped_promotion_only,
        openvino_gpu_promoted_profiles,
        openvino_npu_promoted_profiles,
        profile_promotions,
        blocked_profiles,
        gaps,
    })
}

fn promoted_profiles_for_route(
    ledger: &LunarLakeRoutePromotionLedger,
    route_id: &str,
) -> Vec<String> {
    ledger
        .routes
        .iter()
        .find(|route| route.route_id == route_id)
        .map(|route| route.promoted_for.clone())
        .unwrap_or_default()
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
        None,
        created_utc,
    )
}

#[cfg(test)]
pub fn build_regression_bundle_with_created_utc_and_inputs(
    root: &Path,
    operator_receipt: &Path,
    answer_corpus_v2: Option<&Path>,
    route_profile_comparison: Option<&Path>,
    cold_warm_benchmark: Option<&Path>,
    durability_bundle: Option<&Path>,
    bitnet_semantic_intake: Option<&Path>,
    created_utc: String,
) -> Result<LunarLakeRegressionBundle> {
    build_regression_bundle_with_created_utc_and_inputs_and_power_profile(
        root,
        operator_receipt,
        answer_corpus_v2,
        route_profile_comparison,
        cold_warm_benchmark,
        durability_bundle,
        bitnet_semantic_intake,
        None,
        None,
        created_utc,
    )
}

#[cfg(test)]
pub fn build_regression_bundle_with_created_utc_and_inputs_and_power_profile(
    root: &Path,
    operator_receipt: &Path,
    answer_corpus_v2: Option<&Path>,
    route_profile_comparison: Option<&Path>,
    cold_warm_benchmark: Option<&Path>,
    durability_bundle: Option<&Path>,
    bitnet_semantic_intake: Option<&Path>,
    power_profile_evidence: Option<&Path>,
    blocked_ask_receipt: Option<&Path>,
    created_utc: String,
) -> Result<LunarLakeRegressionBundle> {
    build_regression_bundle_with_created_utc_and_inputs_and_power_profile_and_warm_ask(
        root,
        operator_receipt,
        answer_corpus_v2,
        route_profile_comparison,
        cold_warm_benchmark,
        durability_bundle,
        bitnet_semantic_intake,
        power_profile_evidence,
        None,
        None,
        None,
        None,
        blocked_ask_receipt,
        created_utc,
    )
}

pub fn build_regression_bundle_with_created_utc_and_inputs_and_power_profile_and_warm_ask(
    root: &Path,
    operator_receipt: &Path,
    answer_corpus_v2: Option<&Path>,
    route_profile_comparison: Option<&Path>,
    cold_warm_benchmark: Option<&Path>,
    durability_bundle: Option<&Path>,
    bitnet_semantic_intake: Option<&Path>,
    power_profile_evidence: Option<&Path>,
    thermal_temperature_availability: Option<&Path>,
    ask_short_ask_receipt: Option<&Path>,
    ask_normal_ask_receipt: Option<&Path>,
    warm_resident_ask_receipt: Option<&Path>,
    blocked_ask_receipt: Option<&Path>,
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
            vec![
                "OpenVINO GPU/NPU operator routes start as candidates; any GPU promotion must remain profile-scoped and benchmark-qualified, while NPU remains candidate-only"
                    .to_string(),
            ],
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
    let bitnet_semantic_intake = if let Some(path) = bitnet_semantic_intake {
        let path = resolve_receipt_path(root, path);
        let summary = inspect_bitnet_semantic_intake_regression(&path)?;
        checks.push(regression_check_owned(
            "bitnet_semantic_intake_regression_ready",
            summary.regression_ready,
            vec![summary.path.clone()],
            bitnet_semantic_intake_regression_notes(&summary),
        ));
        Some(summary)
    } else {
        None
    };
    let power_profile_evidence = if let Some(path) = power_profile_evidence {
        let path = resolve_receipt_path(root, path);
        let summary = inspect_power_profile_regression(&path)?;
        checks.push(regression_check_owned(
            "low_power_profile_evidence_regression_ready",
            summary.regression_ready,
            vec![summary.path.clone()],
            power_profile_regression_notes(&summary),
        ));
        Some(summary)
    } else {
        None
    };
    let thermal_temperature_availability = if let Some(path) = thermal_temperature_availability {
        let path = resolve_receipt_path(root, path);
        let summary = inspect_thermal_temperature_availability_regression(&path)?;
        checks.push(regression_check_owned(
            "thermal_temperature_availability_regression_ready",
            summary.regression_ready,
            vec![summary.path.clone()],
            thermal_temperature_availability_regression_notes(&summary),
        ));
        Some(summary)
    } else {
        None
    };
    let npu_warm_resident_promoted = npu_warm_resident_is_promoted(
        cold_warm_benchmark.as_ref(),
        route_profile_comparison.as_ref(),
    );
    let gpu_ask_normal_promoted = openvino_gpu_profile_is_promoted(
        "ask_normal",
        cold_warm_benchmark.as_ref(),
        route_profile_comparison.as_ref(),
    );
    let gpu_ask_short_promoted = openvino_gpu_profile_is_promoted(
        "ask_short",
        cold_warm_benchmark.as_ref(),
        route_profile_comparison.as_ref(),
    );
    let ask_short_ask_receipt = if let Some(path) = ask_short_ask_receipt {
        let path = resolve_receipt_path(root, path);
        let summary = inspect_operator_ask_regression(
            &path,
            OperatorAskRegressionExpectation {
                label: "ask_short",
                profile_id: "ask_short",
                selected_route: "dense_slm_openvino_gpu_candidate",
                selected_backend: "openvino-gpu",
            },
        )?;
        checks.push(regression_check_owned(
            "ask_short_auto_ask_receipt_regression_ready",
            summary.regression_ready,
            vec![summary.path.clone()],
            operator_ask_regression_notes(&summary),
        ));
        Some(summary)
    } else {
        if gpu_ask_short_promoted {
            checks.push(regression_check_owned(
                "ask_short_auto_ask_receipt_regression_ready",
                false,
                vec![AUTO_GPU_ASK_SHORT_ASK_RECEIPT.to_string()],
                vec![
                    "OpenVINO GPU is promoted for ask_short but no successful auto ask receipt was indexed"
                        .to_string(),
                ],
            ));
        }
        None
    };
    let ask_normal_ask_receipt = if let Some(path) = ask_normal_ask_receipt {
        let path = resolve_receipt_path(root, path);
        let summary = inspect_operator_ask_regression(
            &path,
            OperatorAskRegressionExpectation {
                label: "ask_normal",
                profile_id: "ask_normal",
                selected_route: "dense_slm_openvino_gpu_candidate",
                selected_backend: "openvino-gpu",
            },
        )?;
        checks.push(regression_check_owned(
            "ask_normal_auto_ask_receipt_regression_ready",
            summary.regression_ready,
            vec![summary.path.clone()],
            operator_ask_regression_notes(&summary),
        ));
        Some(summary)
    } else {
        if gpu_ask_normal_promoted {
            checks.push(regression_check_owned(
                "ask_normal_auto_ask_receipt_regression_ready",
                false,
                vec![AUTO_GPU_ASK_NORMAL_ASK_RECEIPT.to_string()],
                vec![
                    "OpenVINO GPU is promoted for ask_normal but no successful auto ask receipt was indexed"
                        .to_string(),
                ],
            ));
        }
        None
    };
    let warm_resident_ask_receipt = if let Some(path) = warm_resident_ask_receipt {
        let path = resolve_receipt_path(root, path);
        let summary = inspect_operator_ask_regression(
            &path,
            OperatorAskRegressionExpectation {
                label: "warm_resident",
                profile_id: "warm_resident",
                selected_route: "dense_slm_openvino_npu_candidate",
                selected_backend: "openvino-npu",
            },
        )?;
        checks.push(regression_check_owned(
            "warm_resident_auto_ask_receipt_regression_ready",
            summary.regression_ready,
            vec![summary.path.clone()],
            operator_ask_regression_notes(&summary),
        ));
        Some(summary)
    } else {
        if npu_warm_resident_promoted {
            checks.push(regression_check_owned(
                "warm_resident_auto_ask_receipt_regression_ready",
                false,
                vec![AUTO_NPU_WARM_RESIDENT_ASK_RECEIPT.to_string()],
                vec![
                    "OpenVINO NPU is promoted for warm_resident but no successful auto ask receipt was indexed"
                        .to_string(),
                ],
            ));
        }
        None
    };
    let blocked_ask_receipt = if let Some(path) = blocked_ask_receipt {
        let path = resolve_receipt_path(root, path);
        let summary = inspect_blocked_ask_regression(&path)?;
        checks.push(regression_check_owned(
            "blocked_auto_ask_receipt_regression_ready",
            summary.regression_ready,
            vec![summary.path.clone()],
            blocked_ask_regression_notes(&summary),
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
        bitnet_semantic_intake.as_ref(),
        power_profile_evidence.as_ref(),
        thermal_temperature_availability.as_ref(),
        ask_short_ask_receipt.as_ref(),
        ask_normal_ask_receipt.as_ref(),
        warm_resident_ask_receipt.as_ref(),
        blocked_ask_receipt.as_ref(),
        &operator,
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
        bitnet_semantic_intake,
        power_profile_evidence,
        thermal_temperature_availability,
        ask_short_ask_receipt,
        ask_normal_ask_receipt,
        warm_resident_ask_receipt,
        blocked_ask_receipt,
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
    bitnet_semantic_intake: Option<&BitnetSemanticIntakeRegressionSummary>,
    power_profile_evidence: Option<&PowerProfileRegressionSummary>,
    thermal_temperature_availability: Option<&ThermalTemperatureAvailabilityRegressionSummary>,
    ask_short_ask_receipt: Option<&OperatorAskRegressionSummary>,
    ask_normal_ask_receipt: Option<&OperatorAskRegressionSummary>,
    warm_resident_ask_receipt: Option<&OperatorAskRegressionSummary>,
    blocked_ask_receipt: Option<&BlockedAskRegressionSummary>,
    operator: &LunarLakeOperatorReceipt,
) -> RegressionSurfaceSummary {
    let bitnet_cpu_reference_evidence_ids = [
        "bitnet_cpu_reference_bundle",
        "bitnet_external_reference_boundary",
        "bitnet_external_direct_token_boundary",
        "bitnet_first_token_direct_classifier",
        "bitnet_i2s_gemv_gemm_microbench",
        "bitnet_i2s_tiling_thread_matrix",
        "bitnet_i2s_applied_thread_matrix",
        "bitnet_embedding_quantization_evidence",
    ];
    let bitnet_cpu_reference_route_indexed =
        operator.routes.iter().any(|route| route.route_id == "bitnet_reference_cpu");
    let bitnet_cpu_reference_evidence_indexed = bitnet_cpu_reference_route_indexed
        && bitnet_cpu_reference_evidence_ids
            .iter()
            .all(|id| operator.evidence.iter().any(|item| item.evidence_id == *id && item.present));
    let bitnet_cpu_reference_evidence_ready = route_ok(operator, "bitnet_reference_cpu")
        && bitnet_cpu_reference_evidence_ids.iter().all(|id| evidence_ok(&operator.evidence, id));
    let arc_npu_bounded_evidence_ids = [
        "arc140v_native_opencl_parity",
        "npu_rmsnorm_static_subgraph",
        "npu_linear_static_subgraph",
        "npu_ffn_static_subgraph",
    ];
    let arc_npu_bounded_evidence_indexed = arc_npu_bounded_evidence_ids
        .iter()
        .all(|id| operator.evidence.iter().any(|item| item.evidence_id == *id && item.present));
    let arc_npu_bounded_evidence_ready =
        arc_npu_bounded_evidence_ids.iter().all(|id| evidence_ok(&operator.evidence, id));
    let mut summary = RegressionSurfaceSummary {
        answer_corpus_v2_indexed: answer_corpus_v2.is_some(),
        route_profile_comparison_indexed: route_profile_comparison.is_some(),
        cold_warm_benchmark_indexed: cold_warm_benchmark.is_some(),
        durability_bundle_indexed: durability_bundle.is_some(),
        bitnet_semantic_intake_indexed: bitnet_semantic_intake.is_some(),
        bitnet_cpu_reference_evidence_indexed,
        bitnet_cpu_reference_evidence_ready,
        power_profile_evidence_indexed: power_profile_evidence.is_some(),
        thermal_temperature_availability_indexed: thermal_temperature_availability.is_some(),
        thermal_temperature_available: thermal_temperature_availability
            .map(|summary| summary.thermal_temperature_available)
            .unwrap_or(false),
        thermal_usable_temperature_reading_count: thermal_temperature_availability
            .map(|summary| summary.usable_temperature_reading_count)
            .unwrap_or(0),
        arc_npu_bounded_evidence_indexed,
        arc_npu_bounded_evidence_ready,
        ask_short_ask_receipt_indexed: ask_short_ask_receipt.is_some(),
        ask_short_auto_ask_ready: ask_short_ask_receipt
            .map(|summary| summary.regression_ready)
            .unwrap_or(false),
        ask_normal_ask_receipt_indexed: ask_normal_ask_receipt.is_some(),
        ask_normal_auto_ask_ready: ask_normal_ask_receipt
            .map(|summary| summary.regression_ready)
            .unwrap_or(false),
        warm_resident_ask_receipt_indexed: warm_resident_ask_receipt.is_some(),
        warm_resident_auto_ask_ready: warm_resident_ask_receipt
            .map(|summary| summary.regression_ready)
            .unwrap_or(false),
        blocked_ask_receipt_indexed: blocked_ask_receipt.is_some(),
        route_profile_model_identity_ready: route_profile_comparison
            .map(|summary| summary.route_model_identity_ready)
            .unwrap_or(false),
        cold_warm_model_identity_ready: cold_warm_benchmark
            .map(|summary| summary.route_model_identity_ready)
            .unwrap_or(false),
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
        timing_coverage: cold_warm_benchmark
            .map(|summary| summary.timing_coverage.clone())
            .or_else(|| route_profile_comparison.map(|summary| summary.timing_coverage.clone()))
            .unwrap_or_default(),
        durability_stability_proven: durability_bundle
            .map(|summary| summary.stability_proven)
            .unwrap_or(false),
        low_power_promotion_ready: power_profile_evidence
            .map(|summary| summary.low_power_promotion_ready)
            .unwrap_or(false),
        power_advantage_proven: power_profile_evidence
            .map(|summary| summary.power_advantage_proven)
            .unwrap_or(false),
        route_promotion_scope: cold_warm_benchmark
            .map(|summary| summary.route_promotion_scope.clone())
            .or_else(|| {
                route_profile_comparison.map(|summary| summary.route_promotion_scope.clone())
            })
            .unwrap_or_default(),
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
        if !route_profiles.candidate_routes_remain_unpromoted {
            summary
                .gaps
                .push("OpenVINO GPU/NPU candidate route became promotion-eligible".to_string());
        }
        if !route_profiles.timing_coverage.promotion_eligible_routes_have_profile_specific_timing {
            summary.gaps.push(format!(
                "route profile comparison has promotion-eligible proxy timing: {}",
                route_profiles.timing_coverage.promotion_eligible_proxy_or_missing_routes.join(",")
            ));
        }
        if !route_profiles.timing_coverage.proxy_or_missing_timing_routes_blocked {
            summary.gaps.push(format!(
                "route profile comparison has unblocked proxy timing: {}",
                route_profiles.timing_coverage.unblocked_proxy_or_missing_routes.join(",")
            ));
        }
        if !route_profiles.route_promotion_scope.profile_scoped_promotion_only {
            summary.gaps.push(format!(
                "route profile comparison has unexpected OpenVINO promotions: {}",
                route_profiles
                    .route_promotion_scope
                    .unexpected_openvino_profile_promotions
                    .join(",")
            ));
        }
        if !route_profiles.route_model_identity_ready {
            summary.gaps.push(format!(
                "route profile comparison lacks route/model identity coverage: {}",
                route_profiles.gaps.join("; ")
            ));
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
        if !benchmark.candidate_routes_remain_unpromoted {
            summary
                .gaps
                .push("cold/warm benchmark shows OpenVINO candidate route promotion".to_string());
        }
        if !benchmark.promoted_routes_have_critical_timing {
            summary.gaps.push("promoted routes are missing critical cold/warm timing".to_string());
        }
        if !benchmark.timing_coverage.promotion_eligible_routes_have_profile_specific_timing {
            summary.gaps.push(format!(
                "cold/warm benchmark has promotion-eligible proxy timing: {}",
                benchmark.timing_coverage.promotion_eligible_proxy_or_missing_routes.join(",")
            ));
        }
        if !benchmark.timing_coverage.proxy_or_missing_timing_routes_blocked {
            summary.gaps.push(format!(
                "cold/warm benchmark has unblocked proxy timing: {}",
                benchmark.timing_coverage.unblocked_proxy_or_missing_routes.join(",")
            ));
        }
        if !benchmark.route_promotion_scope.profile_scoped_promotion_only {
            summary.gaps.push(format!(
                "cold/warm benchmark has unexpected OpenVINO promotions: {}",
                benchmark.route_promotion_scope.unexpected_openvino_profile_promotions.join(",")
            ));
        }
        if !benchmark.route_model_identity_ready {
            summary.gaps.push(format!(
                "cold/warm benchmark lacks route/model identity coverage: {}",
                benchmark.gaps.join("; ")
            ));
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

    if let Some(intake) = bitnet_semantic_intake {
        if !intake.regression_ready {
            summary.gaps.push(format!(
                "BitNet semantic intake is not regression-ready: {}",
                intake.gaps.join("; ")
            ));
        }
        if intake.rerun_required {
            summary.gaps.push(format!(
                "BitNet semantic intake requires Lunar Lake reruns: {}",
                intake.required_reruns.join("; ")
            ));
        }
        if !intake.claim_boundary_preserved {
            summary.gaps.push("BitNet semantic intake claim boundary is not preserved".to_string());
        }
    } else {
        summary.gaps.push("BitNet semantic intake is not indexed".to_string());
    }

    if !summary.bitnet_cpu_reference_evidence_indexed {
        summary.gaps.push(
            "BitNet CPU reference route evidence is not indexed in operator readiness".to_string(),
        );
    } else if !summary.bitnet_cpu_reference_evidence_ready {
        summary
            .gaps
            .push("BitNet CPU reference route evidence is not regression-ready".to_string());
    }

    if let Some(power) = power_profile_evidence {
        if !power.regression_ready {
            summary.gaps.push(format!(
                "low_power power-profile evidence is not regression-ready: {}",
                power.gaps.join("; ")
            ));
        }
        if !power.low_power_routes_remain_unpromoted {
            summary.gaps.push(
                "low_power power-profile evidence shows route promotion without promotion-lane proof"
                    .to_string(),
            );
        }
        if !power.claim_boundary_preserved {
            summary
                .gaps
                .push("low_power power-profile claim boundary is not preserved".to_string());
        }
    }

    if let Some(thermal) = thermal_temperature_availability {
        if !thermal.regression_ready {
            summary.gaps.push(format!(
                "thermal temperature availability is not regression-ready: {}",
                thermal.gaps.join("; ")
            ));
        }
        if !thermal.claim_boundary_preserved {
            summary.gaps.push(
                "thermal temperature availability claim boundary is not preserved".to_string(),
            );
        }
        if thermal.measured_temperature_claim && thermal.usable_temperature_reading_count == 0 {
            summary.gaps.push(
                "thermal temperature availability claims measured temperatures without usable readings"
                    .to_string(),
            );
        }
    }

    if !summary.arc_npu_bounded_evidence_indexed {
        summary.gaps.push(
            "Arc/NPU bounded proof evidence is not indexed in operator readiness".to_string(),
        );
    } else if !summary.arc_npu_bounded_evidence_ready {
        summary.gaps.push("Arc/NPU bounded proof evidence is not regression-ready".to_string());
    }

    let npu_warm_resident_promoted = summary
        .route_promotion_scope
        .openvino_npu_promoted_profiles
        .iter()
        .any(|profile| profile == "warm_resident");
    let gpu_ask_normal_promoted = summary
        .route_promotion_scope
        .openvino_gpu_promoted_profiles
        .iter()
        .any(|profile| profile == "ask_normal");
    let gpu_ask_short_promoted = summary
        .route_promotion_scope
        .openvino_gpu_promoted_profiles
        .iter()
        .any(|profile| profile == "ask_short");
    if gpu_ask_short_promoted {
        if let Some(ask) = ask_short_ask_receipt {
            if !ask.regression_ready {
                summary.gaps.push(format!(
                    "ask_short auto GPU ask receipt is not regression-ready: {}",
                    ask.gaps.join("; ")
                ));
            }
            if ask.profile_id != "ask_short"
                || ask.requested_device != "auto"
                || ask.requested_route != "auto"
                || ask.selected_route != "dense_slm_openvino_gpu_candidate"
            {
                summary.gaps.push(
                    "ask_short ask receipt does not prove auto selected the promoted GPU route"
                        .to_string(),
                );
            }
        } else {
            summary.gaps.push(
                "OpenVINO GPU is promoted for ask_short but no successful auto ask receipt is indexed"
                    .to_string(),
            );
        }
    }
    if gpu_ask_normal_promoted {
        if let Some(ask) = ask_normal_ask_receipt {
            if !ask.regression_ready {
                summary.gaps.push(format!(
                    "ask_normal auto GPU ask receipt is not regression-ready: {}",
                    ask.gaps.join("; ")
                ));
            }
            if ask.profile_id != "ask_normal"
                || ask.requested_device != "auto"
                || ask.requested_route != "auto"
                || ask.selected_route != "dense_slm_openvino_gpu_candidate"
            {
                summary.gaps.push(
                    "ask_normal ask receipt does not prove auto selected the promoted GPU route"
                        .to_string(),
                );
            }
        } else {
            summary.gaps.push(
                "OpenVINO GPU is promoted for ask_normal but no successful auto ask receipt is indexed"
                    .to_string(),
            );
        }
    }

    if npu_warm_resident_promoted {
        if let Some(ask) = warm_resident_ask_receipt {
            if !ask.regression_ready {
                summary.gaps.push(format!(
                    "warm_resident auto NPU ask receipt is not regression-ready: {}",
                    ask.gaps.join("; ")
                ));
            }
            if ask.profile_id != "warm_resident"
                || ask.requested_device != "auto"
                || ask.requested_route != "auto"
                || ask.selected_route != "dense_slm_openvino_npu_candidate"
            {
                summary.gaps.push(
                    "warm_resident ask receipt does not prove auto selected the promoted NPU route"
                        .to_string(),
                );
            }
        } else {
            summary.gaps.push(
                "OpenVINO NPU is promoted for warm_resident but no successful auto ask receipt is indexed"
                    .to_string(),
            );
        }
    }

    if let Some(blocked) = blocked_ask_receipt {
        if !blocked.regression_ready {
            summary.gaps.push(format!(
                "blocked auto-route ask receipt is not regression-ready: {}",
                blocked.gaps.join("; ")
            ));
        }
        if !blocked.route_selection_blocked {
            summary
                .gaps
                .push("blocked ask receipt does not prove route_selection_blocked".to_string());
        }
        if blocked.new_inference_executed {
            summary.gaps.push("blocked ask receipt unexpectedly ran inference".to_string());
        }
        if blocked.fallback_used {
            summary.gaps.push("blocked ask receipt unexpectedly observed fallback".to_string());
        }
        if blocked.route_promotion_changed
            || blocked.speedup_claim
            || blocked.power_advantage_claim
            || blocked.acceleration_claim
            || blocked.bitnet_qk256_i2s_claim
        {
            summary.gaps.push(
                "blocked ask receipt violates route-promotion/speedup/power/acceleration claim boundary"
                    .to_string(),
            );
        }
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
    let mut unexpected_candidate_promotion_eligible = false;
    let mut blockers = BTreeSet::new();
    let gpu_npu_promotion_blocker_summary = comparison
        .promotion_blocker_summary
        .iter()
        .filter(|summary| {
            summary.route_ids.iter().any(|route_id| is_openvino_candidate_route(route_id))
        })
        .cloned()
        .collect::<Vec<_>>();
    let timing_coverage = if comparison.timing_coverage.route_count > 0 {
        comparison.timing_coverage.clone()
    } else {
        timing_applicability_coverage(&comparison.profiles)
    };
    let route_model_identity_coverage = route_model_identity_coverage(&comparison.profiles);
    append_route_model_identity_coverage_gaps(
        "route profile comparison",
        &route_model_identity_coverage,
        &mut gaps,
    );
    let route_promotion_scope = route_promotion_scope_from_profile_comparison(&comparison.profiles);
    for profile in &comparison.profiles {
        for route in &profile.route_evidence {
            if route.fallback_used == Some(true) {
                fallback_observed = true;
            }
            if route.benchmark_qualified_advantage {
                benchmark_qualified_advantage_claimed = true;
            }
            if is_openvino_candidate_route(&route.route_id) {
                if route.promotion_eligible_for_profile
                    && !allowed_openvino_profile_promotion(
                        &profile.profile_id,
                        &route.route_id,
                        &route.route_status,
                        profile.promoted_route.as_deref(),
                    )
                {
                    unexpected_candidate_promotion_eligible = true;
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
    if unexpected_candidate_promotion_eligible {
        gaps.push(
            "unexpected OpenVINO GPU/NPU candidate route became promotion-eligible".to_string(),
        );
    }
    if !route_promotion_scope.profile_scoped_promotion_only {
        gaps.push(format!(
            "unexpected OpenVINO profile promotions: {}",
            route_promotion_scope.unexpected_openvino_profile_promotions.join(",")
        ));
    }
    if blockers.is_empty() {
        gaps.push("OpenVINO GPU/NPU candidate blockers are missing".to_string());
    }
    if !timing_coverage.promotion_eligible_routes_have_profile_specific_timing {
        gaps.push(format!(
            "promotion-eligible routes lack profile-specific timing: {}",
            timing_coverage.promotion_eligible_proxy_or_missing_routes.join(",")
        ));
    }
    if !timing_coverage.proxy_or_missing_timing_routes_blocked {
        gaps.push(format!(
            "proxy or missing timing routes lack promotion blockers: {}",
            timing_coverage.unblocked_proxy_or_missing_routes.join(",")
        ));
    }

    Ok(RouteProfileRegressionSummary {
        path: path_string(path),
        profile_comparison_ready: comparison.profile_comparison_ready,
        default_route_id: comparison.default_route_id,
        profiles,
        timing_coverage,
        route_model_identity_ready: route_model_identity_coverage_ready(
            &route_model_identity_coverage,
        ),
        route_model_identity_coverage,
        candidate_routes_remain_unpromoted: !unexpected_candidate_promotion_eligible,
        benchmark_qualified_advantage_claimed,
        fallback_observed,
        gpu_npu_promotion_blockers: blockers.into_iter().collect(),
        gpu_npu_promotion_blocker_summary,
        route_promotion_scope,
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
    let mut unexpected_candidate_routes_remain_unpromoted = true;
    let mut telemetry_gaps = BTreeSet::new();
    let timing_coverage = benchmark.timing_coverage.clone();
    let route_model_identity_coverage =
        cold_warm_route_model_identity_coverage(&benchmark.profiles);
    append_route_model_identity_coverage_gaps(
        "cold/warm benchmark",
        &route_model_identity_coverage,
        &mut gaps,
    );
    let route_promotion_scope = route_promotion_scope_from_cold_warm(&benchmark.profiles);
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
            if is_openvino_candidate_route(&route.route_id)
                && !route.promotion_blocked
                && !allowed_openvino_profile_promotion(
                    &profile.profile_id,
                    &route.route_id,
                    &route.route_status,
                    profile.promoted_route.as_deref(),
                )
            {
                unexpected_candidate_routes_remain_unpromoted = false;
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
    if !promoted_routes_have_critical_timing {
        gaps.push("promoted routes are missing critical cold/warm timing".to_string());
    }
    if !unexpected_candidate_routes_remain_unpromoted {
        gaps.push(
            "unexpected OpenVINO GPU/NPU candidate route was promoted in cold/warm benchmark"
                .to_string(),
        );
    }
    if !route_promotion_scope.profile_scoped_promotion_only {
        gaps.push(format!(
            "unexpected OpenVINO profile promotions in cold/warm benchmark: {}",
            route_promotion_scope.unexpected_openvino_profile_promotions.join(",")
        ));
    }
    if !timing_coverage.promotion_eligible_routes_have_profile_specific_timing {
        gaps.push(format!(
            "promotion-eligible routes lack profile-specific timing: {}",
            timing_coverage.promotion_eligible_proxy_or_missing_routes.join(",")
        ));
    }
    if !timing_coverage.proxy_or_missing_timing_routes_blocked {
        gaps.push(format!(
            "proxy or missing timing routes lack promotion blockers: {}",
            timing_coverage.unblocked_proxy_or_missing_routes.join(",")
        ));
    }

    Ok(ColdWarmRegressionSummary {
        path: path_string(path),
        benchmark_gate_ready: benchmark.benchmark_gate_ready,
        profiles,
        timing_coverage,
        route_model_identity_ready: route_model_identity_coverage_ready(
            &route_model_identity_coverage,
        ),
        route_model_identity_coverage,
        promoted_routes_have_critical_timing,
        candidate_routes_remain_unpromoted: unexpected_candidate_routes_remain_unpromoted,
        fallback_observed,
        benchmark_qualified_advantage_claimed,
        telemetry_gaps: telemetry_gaps.into_iter().collect(),
        route_promotion_scope,
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

fn inspect_bitnet_semantic_intake_regression(
    path: &Path,
) -> Result<BitnetSemanticIntakeRegressionSummary> {
    let intake: LunarLakeBitnetSemanticIntake = read_json_receipt(path)?;
    let mut gaps = Vec::new();
    if intake.artifact_kind != "lunar_lake_bitnet_semantic_intake" {
        gaps.push(format!("unexpected artifact_kind={}", intake.artifact_kind));
    }
    if !intake.intake_ready {
        gaps.push(format!("BitNet semantic intake is not ready: {}", intake.gaps.join("; ")));
    }
    if intake.rerun_required {
        gaps.push(format!(
            "shared BitNet semantic fixes require Lunar Lake reruns: {}",
            intake.required_reruns.join("; ")
        ));
    }
    if intake.source_change_summary.stale_after_merged_count > 0 {
        gaps.push(format!(
            "{} merged shared BitNet semantic changes are newer than Lunar Lake evidence",
            intake.source_change_summary.stale_after_merged_count
        ));
    }

    let claim = &intake.claim_boundary;
    let claim_boundary_preserved = !claim.new_inference_executed
        && !claim.route_promotion_changed
        && !claim.answer_quality_claim
        && !claim.speedup_claim
        && !claim.acceleration_claim
        && !claim.arc_or_npu_bitnet_claim
        && !claim.qk256_behavior_changed
        && !claim.dense_slm_as_bitnet_proof
        && !claim.hidden_fallback_allowed;
    if !claim_boundary_preserved {
        gaps.push(
            "BitNet semantic intake must preserve no-inference/no-promotion/no-speedup/no-acceleration/no-QK256-change claim boundary"
                .to_string(),
        );
    }

    Ok(BitnetSemanticIntakeRegressionSummary {
        path: path_string(path),
        intake_ready: intake.intake_ready,
        rerun_required: intake.rerun_required,
        pending_shared_change_count: intake.source_change_summary.pending_shared_change_count,
        closed_shared_change_count: intake.source_change_summary.closed_shared_change_count,
        merged_to_main_count: intake.source_change_summary.merged_to_main_count,
        stale_after_merged_count: intake.source_change_summary.stale_after_merged_count,
        source_lanes: intake.source_change_summary.source_lanes,
        pending_changes: intake.source_change_summary.pending_changes,
        closed_changes: intake.source_change_summary.closed_changes,
        required_reruns: intake.required_reruns,
        claim_boundary_preserved,
        regression_ready: gaps.is_empty(),
        gaps,
    })
}

fn inspect_power_profile_regression(path: &Path) -> Result<PowerProfileRegressionSummary> {
    let power: LunarLakePowerProfileEvidence = read_json_receipt(path)?;
    let mut gaps = Vec::new();
    if power.artifact_kind != "lunar_lake_power_profile_evidence" {
        gaps.push(format!("unexpected artifact_kind={}", power.artifact_kind));
    }
    if !power.power_profile_index_ready {
        gaps.push(format!(
            "low_power power-profile evidence is not index-ready: {}",
            power.gaps.join("; ")
        ));
    }
    let low_power_routes_remain_unpromoted = power
        .low_power_routes
        .iter()
        .all(|route| route.route_status != "promoted" && !route.power_promotion_ready);
    if !low_power_routes_remain_unpromoted {
        gaps.push(
            "low_power power-profile evidence promoted a route or marked a route promotion-ready"
                .to_string(),
        );
    }
    if power.low_power_routes.iter().any(|route| route.fallback_used == Some(true)) {
        gaps.push("low_power power-profile evidence observed fallback_used=true".to_string());
    }

    let claim = &power.claim_boundary;
    let claim_boundary_preserved = !claim.new_inference_executed
        && !claim.route_promotion_changed
        && !claim.speedup_claim
        && !claim.power_advantage_claim
        && !claim.acceleration_claim
        && !claim.native_npu_inference_claim
        && !claim.bitnet_qk256_i2s_behavior_changed
        && !claim.hidden_fallback_allowed;
    if !claim_boundary_preserved {
        gaps.push(
            "low_power power-profile evidence must preserve no-inference/no-promotion/no-speedup/no-power-advantage/no-acceleration/no-QK256-change claim boundary"
                .to_string(),
        );
    }
    if !power.telemetry.battery_mode_sample_recorded {
        if power.operator_runbook.as_deref() != Some(LOW_POWER_BATTERY_RUNBOOK) {
            gaps.push(format!(
                "low_power power-profile evidence must point to {LOW_POWER_BATTERY_RUNBOOK}"
            ));
        }
        if !power
            .next_required_evidence
            .iter()
            .any(|item| item.contains("telemetry-context --require-battery"))
        {
            gaps.push(
                "low_power power-profile evidence must name telemetry-context --require-battery as next evidence".to_string(),
            );
        }
    }

    let mut blockers = power.gaps.clone();
    for route in &power.low_power_routes {
        blockers.extend(route.power_related_blockers.iter().cloned());
    }
    blockers.sort();
    blockers.dedup();

    Ok(PowerProfileRegressionSummary {
        path: path_string(path),
        power_profile_index_ready: power.power_profile_index_ready,
        low_power_promotion_ready: power.low_power_promotion_ready,
        power_advantage_proven: power.power_advantage_proven,
        low_power_route_count: power.low_power_routes.len(),
        low_power_routes_remain_unpromoted,
        current_context_is_ac_only: power.telemetry.current_context_is_ac_only,
        battery_mode_sample_recorded: power.telemetry.battery_mode_sample_recorded,
        battery_sample_source: power.telemetry.battery_sample_source,
        energy_proxy_recorded: power.telemetry.energy_proxy_recorded,
        energy_proxy_source: power.telemetry.energy_proxy_source,
        thermal_context_recorded: power.telemetry.thermal_context_recorded,
        operator_runbook: power.operator_runbook,
        next_required_evidence: power.next_required_evidence,
        claim_boundary_preserved,
        regression_ready: gaps.is_empty(),
        gaps,
        blockers,
    })
}

fn inspect_thermal_temperature_availability_regression(
    path: &Path,
) -> Result<ThermalTemperatureAvailabilityRegressionSummary> {
    let thermal: Value = read_json_receipt(path)?;
    let mut gaps = Vec::new();

    let artifact_kind = string_at(&thermal, "artifact_kind").unwrap_or_default();
    if artifact_kind != "lunar_lake_thermal_temperature_availability" {
        gaps.push(format!("unexpected artifact_kind={artifact_kind}"));
    }

    let thermal_zone_visibility_available =
        bool_at_any(&thermal, &["decision.thermal_zone_visibility_available"]).unwrap_or(false);
    let thermal_temperature_available =
        bool_at_any(&thermal, &["decision.thermal_temperature_available"]).unwrap_or(false);
    let usable_temperature_reading_count =
        u64_at(&thermal, "decision.usable_temperature_reading_count").unwrap_or(0) as usize;

    if thermal_temperature_available && usable_temperature_reading_count == 0 {
        gaps.push(
            "thermal temperature availability claims temperature availability without usable readings"
                .to_string(),
        );
    }

    let measured_temperature_claim =
        bool_at_any(&thermal, &["claim_boundary.measured_temperature_claim"]).unwrap_or(false);
    let telemetry_probe_executed =
        bool_at_any(&thermal, &["claim_boundary.telemetry_probe_executed"]).unwrap_or(false);

    let claim_boundary_preserved =
        !bool_at_any(&thermal, &["claim_boundary.new_inference_executed"]).unwrap_or(false)
            && !bool_at_any(&thermal, &["claim_boundary.route_promotion_changed"]).unwrap_or(false)
            && !bool_at_any(&thermal, &["claim_boundary.speedup_claim"]).unwrap_or(false)
            && !bool_at_any(&thermal, &["claim_boundary.power_advantage_claim"]).unwrap_or(false)
            && !bool_at_any(&thermal, &["claim_boundary.acceleration_claim"]).unwrap_or(false)
            && !bool_at_any(&thermal, &["claim_boundary.native_opencl_or_native_npu_claim"])
                .unwrap_or(false)
            && !bool_at_any(&thermal, &["claim_boundary.bitnet_qk256_or_i2s_behavior_changed"])
                .unwrap_or(false)
            && (!measured_temperature_claim || usable_temperature_reading_count > 0);
    if !claim_boundary_preserved {
        gaps.push(
            "thermal temperature availability must preserve no-inference/no-promotion/no-speedup/no-power-advantage/no-acceleration/no-QK256-change claim boundary"
                .to_string(),
        );
    }

    Ok(ThermalTemperatureAvailabilityRegressionSummary {
        path: path_string(path),
        thermal_zone_visibility_available,
        thermal_temperature_available,
        usable_temperature_reading_count,
        measured_temperature_claim,
        telemetry_probe_executed,
        claim_boundary_preserved,
        regression_ready: gaps.is_empty(),
        gaps,
    })
}

fn npu_warm_resident_is_promoted(
    cold_warm_benchmark: Option<&ColdWarmRegressionSummary>,
    route_profile_comparison: Option<&RouteProfileRegressionSummary>,
) -> bool {
    cold_warm_benchmark
        .map(|summary| {
            summary
                .route_promotion_scope
                .openvino_npu_promoted_profiles
                .iter()
                .any(|profile| profile == "warm_resident")
        })
        .or_else(|| {
            route_profile_comparison.map(|summary| {
                summary
                    .route_promotion_scope
                    .openvino_npu_promoted_profiles
                    .iter()
                    .any(|profile| profile == "warm_resident")
            })
        })
        .unwrap_or(false)
}

fn openvino_gpu_profile_is_promoted(
    profile_id: &str,
    cold_warm_benchmark: Option<&ColdWarmRegressionSummary>,
    route_profile_comparison: Option<&RouteProfileRegressionSummary>,
) -> bool {
    cold_warm_benchmark
        .map(|summary| {
            summary
                .route_promotion_scope
                .openvino_gpu_promoted_profiles
                .iter()
                .any(|profile| profile == profile_id)
        })
        .or_else(|| {
            route_profile_comparison.map(|summary| {
                summary
                    .route_promotion_scope
                    .openvino_gpu_promoted_profiles
                    .iter()
                    .any(|profile| profile == profile_id)
            })
        })
        .unwrap_or(false)
}

struct OperatorAskRegressionExpectation<'a> {
    label: &'a str,
    profile_id: &'a str,
    selected_route: &'a str,
    selected_backend: &'a str,
}

fn inspect_operator_ask_regression(
    path: &Path,
    expected: OperatorAskRegressionExpectation<'_>,
) -> Result<OperatorAskRegressionSummary> {
    let receipt: Value = read_json_receipt(path)?;
    let mut gaps = Vec::new();
    let artifact_kind = string_at(&receipt, "artifact_kind").unwrap_or_default();
    if artifact_kind != "lunar_lake_operator_ask" {
        gaps.push(format!("unexpected artifact_kind={artifact_kind}"));
    }
    let proof_stage = string_at(&receipt, "proof_stage").unwrap_or_default();
    if proof_stage != "operator_candidate_route_executed_through_lunar_lake_ask" {
        gaps.push(format!("unexpected proof_stage={proof_stage}"));
    }

    let profile_id = string_at(&receipt, "profile_id")
        .or_else(|| string_at(&receipt, "route_selection.profile_id"))
        .unwrap_or_default();
    let requested_device = string_at(&receipt, "requested_device")
        .or_else(|| string_at(&receipt, "route_selection.requested_device"))
        .unwrap_or_default();
    let requested_route = string_at(&receipt, "requested_route")
        .or_else(|| string_at(&receipt, "route_selection.requested_route"))
        .unwrap_or_default();
    let selected_route = string_at(&receipt, "selected_route")
        .or_else(|| string_at(&receipt, "route_selection.selected_route"))
        .or_else(|| string_at(&receipt, "route_id"))
        .unwrap_or_default();
    let selected_backend = string_at(&receipt, "selected_backend")
        .or_else(|| string_at(&receipt, "route_selection.selected_backend"))
        .or_else(|| string_at(&receipt, "backend.selected_backend"))
        .unwrap_or_default();
    let runtime_api = string_at(&receipt, "runtime_api")
        .or_else(|| string_at(&receipt, "route_selection.runtime_api"))
        .or_else(|| string_at(&receipt, "backend.runtime_api"))
        .unwrap_or_default();
    let promotion_status = string_at(&receipt, "promotion_status")
        .or_else(|| string_at(&receipt, "route_selection.promotion_status"))
        .unwrap_or_default();
    let route_profile_status = string_at(&receipt, "route_profile_status")
        .or_else(|| string_at(&receipt, "route_selection.route_profile_status"));
    let route_profile_blockers = string_array_at(&receipt, "route_profile_blockers")
        .into_iter()
        .chain(string_array_at(&receipt, "route_selection.route_profile_blockers"))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let fallback_used = fallback_used(&receipt).unwrap_or(true);
    let answer_gate_passed = answer_gate_passed(&receipt).unwrap_or(false);
    let openvino_candidate_route_executed = bool_at_any(
        &receipt,
        &["openvino_candidate_route_executed", "claim_boundary.openvino_candidate_route_executed"],
    )
    .unwrap_or(false);
    let new_inference_executed = proof_stage.contains("executed");
    let speedup_claim =
        bool_at_any(&receipt, &["speedup_claim", "claim_boundary.speedup_claim"]).unwrap_or(false);
    let power_advantage_claim =
        bool_at_any(&receipt, &["power_advantage_claim", "claim_boundary.power_advantage_claim"])
            .unwrap_or(false);
    let acceleration_claim = bool_at_any(
        &receipt,
        &[
            "acceleration_claim",
            "route.acceleration_claim",
            "claim_boundary.acceleration_claim",
            "claim_boundary.arc_or_npu_acceleration_claim",
        ],
    )
    .unwrap_or(false);
    let bitnet_qk256_i2s_claim =
        bool_at_any(&receipt, &["bitnet_qk256_i2s_claim", "claim_boundary.bitnet_qk256_i2s_claim"])
            .unwrap_or(false);
    let generated_token_ids_available = value_at(&receipt, "tokens.generated_ids")
        .and_then(Value::as_array)
        .is_some_and(|ids| !ids.is_empty())
        || value_at(&receipt, "source_receipt.output.generated_token_ids")
            .and_then(Value::as_array)
            .is_some_and(|ids| !ids.is_empty())
        || bool_at_any(
            &receipt,
            &[
                "source_receipt.output.generated_token_ids_available_from_pipeline",
                "source_receipt.verification.generated_token_ids_available_from_pipeline",
            ],
        ) == Some(true);
    let source_run_receipt = string_at(&receipt, "source_run_receipt");

    if profile_id != expected.profile_id {
        gaps.push(format!(
            "operator ask receipt should cover {}; got {profile_id}",
            expected.profile_id
        ));
    }
    if requested_device != "auto" || requested_route != "auto" {
        gaps.push(format!(
            "operator ask receipt should cover auto/auto request; got device={requested_device} route={requested_route}"
        ));
    }
    if selected_route != expected.selected_route {
        gaps.push(format!(
            "expected selected_route={}; got {selected_route}",
            expected.selected_route
        ));
    }
    if selected_backend != expected.selected_backend {
        gaps.push(format!(
            "expected selected_backend={}; got {selected_backend}",
            expected.selected_backend
        ));
    }
    if runtime_api != "openvino_genai" {
        gaps.push(format!("expected runtime_api=openvino_genai; got {runtime_api}"));
    }
    if promotion_status != "promoted" {
        gaps.push(format!("expected promotion_status=promoted; got {promotion_status}"));
    }
    if route_profile_status.as_deref() != Some("promoted_route_ready") {
        gaps.push(format!(
            "expected route_profile_status=promoted_route_ready; got {}",
            route_profile_status.as_deref().unwrap_or("none")
        ));
    }
    if !route_profile_blockers.is_empty() {
        gaps.push(format!(
            "{} ask receipt has route profile blockers: {}",
            expected.label,
            route_profile_blockers.join("; ")
        ));
    }
    if fallback_used {
        gaps.push(format!("{} ask receipt observed fallback_used=true", expected.label));
    }
    if !answer_gate_passed {
        gaps.push(format!("{} ask receipt did not pass answer gate", expected.label));
    }
    if !openvino_candidate_route_executed {
        gaps.push(format!(
            "{} ask receipt does not prove OpenVINO candidate route execution",
            expected.label
        ));
    }
    if !new_inference_executed {
        gaps.push(format!(
            "{} ask receipt did not record an executed operator ask",
            expected.label
        ));
    }
    if speedup_claim || power_advantage_claim || acceleration_claim || bitnet_qk256_i2s_claim {
        gaps.push(format!(
            "{} ask receipt violates speedup/power/acceleration/BitNet QK256 claim boundary",
            expected.label
        ));
    }
    if !generated_token_ids_available {
        gaps.push(format!("{} ask receipt has no generated token ID evidence", expected.label));
    }

    gaps.sort();
    gaps.dedup();
    Ok(OperatorAskRegressionSummary {
        path: path_string(path),
        ask_receipt_ready: gaps.is_empty(),
        profile_id,
        requested_device,
        requested_route,
        selected_route,
        selected_backend,
        runtime_api,
        promotion_status,
        route_profile_status,
        route_profile_blockers,
        fallback_used,
        answer_gate_passed,
        openvino_candidate_route_executed,
        new_inference_executed,
        speedup_claim,
        power_advantage_claim,
        acceleration_claim,
        bitnet_qk256_i2s_claim,
        generated_token_ids_available,
        source_run_receipt,
        regression_ready: gaps.is_empty(),
        gaps,
    })
}

fn inspect_blocked_ask_regression(path: &Path) -> Result<BlockedAskRegressionSummary> {
    let receipt: Value = read_json_receipt(path)?;
    let mut gaps = Vec::new();
    let artifact_kind = string_at(&receipt, "artifact_kind").unwrap_or_default();
    if artifact_kind != "lunar_lake_operator_ask_blocked" {
        gaps.push(format!("unexpected artifact_kind={artifact_kind}"));
    }
    let proof_stage = string_at(&receipt, "proof_stage").unwrap_or_default();
    if proof_stage != "operator_route_selection_blocked_no_inference" {
        gaps.push(format!("unexpected proof_stage={proof_stage}"));
    }
    let profile_id = string_at(&receipt, "profile_id").unwrap_or_default();
    if profile_id != "low_power" {
        gaps.push(format!("blocked ask receipt should cover low_power; got {profile_id}"));
    }
    let requested_device = string_at(&receipt, "requested_device").unwrap_or_default();
    let requested_route = string_at(&receipt, "requested_route").unwrap_or_default();
    if requested_device != "auto" || requested_route != "auto" {
        gaps.push(format!(
            "blocked ask receipt should cover auto/auto request; got device={requested_device} route={requested_route}"
        ));
    }
    let route_selection_blocked = bool_at_any(
        &receipt,
        &["route_selection_blocked", "claim_boundary.route_selection_blocked"],
    )
    .unwrap_or(false);
    let model_path_required =
        bool_at_any(&receipt, &["model_path_required", "route_selection.model_path_required"])
            .unwrap_or(true);
    let model_loaded = bool_at_any(&receipt, &["claim_boundary.model_loaded"]).unwrap_or(true);
    let model_resolution =
        string_at_any(&receipt, &["model_resolution", "route_selection.model_resolution"])
            .unwrap_or_default();
    let candidate_routes = non_empty_string_array_at_any(
        &receipt,
        &["candidate_routes", "route_selection.candidate_routes"],
    );
    let why_not_cpu =
        non_empty_string_array_at_any(&receipt, &["why_not_cpu", "route_selection.why_not_cpu"]);
    let why_not_gpu =
        non_empty_string_array_at_any(&receipt, &["why_not_gpu", "route_selection.why_not_gpu"]);
    let why_not_npu =
        non_empty_string_array_at_any(&receipt, &["why_not_npu", "route_selection.why_not_npu"]);
    let operator_runbook =
        string_at_any(&receipt, &["operator_runbook", "route_selection.operator_runbook"]);
    let next_required_evidence = non_empty_string_array_at_any(
        &receipt,
        &["next_required_evidence", "route_selection.next_required_evidence"],
    );
    let new_inference_executed =
        bool_at_any(&receipt, &["new_inference_executed", "claim_boundary.new_inference_executed"])
            .unwrap_or(true);
    let fallback_used =
        bool_at_any(&receipt, &["fallback_used", "claim_boundary.fallback_used"]).unwrap_or(true);
    let route_promotion_changed =
        bool_at_any(&receipt, &["claim_boundary.route_promotion_changed"]).unwrap_or(true);
    let speedup_claim =
        bool_at_any(&receipt, &["speedup_claim", "claim_boundary.speedup_claim"]).unwrap_or(true);
    let power_advantage_claim =
        bool_at_any(&receipt, &["power_advantage_claim", "claim_boundary.power_advantage_claim"])
            .unwrap_or(true);
    let acceleration_claim =
        bool_at_any(&receipt, &["acceleration_claim", "claim_boundary.acceleration_claim"])
            .unwrap_or(true);
    let bitnet_qk256_i2s_claim =
        bool_at_any(&receipt, &["bitnet_qk256_i2s_claim", "claim_boundary.bitnet_qk256_i2s_claim"])
            .unwrap_or(true);
    if !route_selection_blocked {
        gaps.push("blocked ask receipt does not prove route_selection_blocked=true".to_string());
    }
    if model_path_required {
        gaps.push(
            "blocked ask receipt should not require model path before blocked auto-route selection"
                .to_string(),
        );
    }
    if model_loaded {
        gaps.push("blocked ask receipt must record model_loaded=false".to_string());
    }
    if model_resolution != "not_required_for_blocked_auto_route_before_execution" {
        gaps.push(format!(
            "blocked ask receipt has unexpected model_resolution={model_resolution}"
        ));
    }
    if new_inference_executed {
        gaps.push("blocked ask receipt must record new_inference_executed=false".to_string());
    }
    if fallback_used {
        gaps.push("blocked ask receipt must record fallback_used=false".to_string());
    }
    if route_promotion_changed || speedup_claim || power_advantage_claim || acceleration_claim {
        gaps.push(
            "blocked ask receipt must preserve no route-promotion/speedup/power/acceleration claims"
                .to_string(),
        );
    }
    if bitnet_qk256_i2s_claim {
        gaps.push("blocked ask receipt must not claim BitNet QK256/I2_S behavior".to_string());
    }
    let route_selection_error = string_at(&receipt, "route_selection_error").unwrap_or_default();
    if route_selection_error.trim().is_empty() {
        gaps.push("blocked ask receipt is missing route_selection_error".to_string());
    }
    for required in
        ["no promoted Lunar Lake auto route", "why_not_cpu=", "why_not_gpu=", "why_not_npu="]
    {
        if !route_selection_error.contains(required) {
            gaps.push(format!("route_selection_error is missing `{required}`"));
        }
    }
    if profile_id == "low_power" {
        if operator_runbook.as_deref() != Some(LOW_POWER_BATTERY_RUNBOOK) {
            gaps.push(format!(
                "low_power blocked ask receipt must point to {LOW_POWER_BATTERY_RUNBOOK}"
            ));
        }
        if !next_required_evidence
            .iter()
            .any(|item| item.contains("telemetry-context --require-battery"))
        {
            gaps.push(
                "low_power blocked ask receipt must name telemetry-context --require-battery as next evidence".to_string(),
            );
        }
        if !route_selection_error.contains(LOW_POWER_BATTERY_RUNBOOK) {
            gaps.push(
                "low_power blocked ask route_selection_error must include the battery runbook path"
                    .to_string(),
            );
        }
    }
    if candidate_routes.is_empty() {
        gaps.push("blocked ask receipt is missing structured candidate_routes".to_string());
    }
    if why_not_cpu.is_empty() {
        gaps.push("blocked ask receipt is missing structured why_not_cpu".to_string());
    }
    if why_not_gpu.is_empty() {
        gaps.push("blocked ask receipt is missing structured why_not_gpu".to_string());
    }
    if why_not_npu.is_empty() {
        gaps.push("blocked ask receipt is missing structured why_not_npu".to_string());
    }

    gaps.sort();
    gaps.dedup();
    Ok(BlockedAskRegressionSummary {
        path: path_string(path),
        blocked_receipt_ready: gaps.is_empty(),
        profile_id,
        requested_device,
        requested_route,
        route_selection_blocked,
        model_path_required,
        model_loaded,
        model_resolution,
        candidate_routes,
        why_not_cpu,
        why_not_gpu,
        why_not_npu,
        operator_runbook,
        next_required_evidence,
        new_inference_executed,
        fallback_used,
        route_promotion_changed,
        speedup_claim,
        power_advantage_claim,
        acceleration_claim,
        bitnet_qk256_i2s_claim,
        route_selection_error,
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
        format!(
            "profile_specific_timing={}/{}",
            summary.timing_coverage.profile_specific_route_count,
            summary.timing_coverage.route_count
        ),
        format!(
            "proxy_or_missing_timing_routes_blocked={}",
            summary.timing_coverage.proxy_or_missing_timing_routes_blocked
        ),
        format!("route_model_identity_ready={}", summary.route_model_identity_ready),
        format!(
            "route_model_identity_rows={}/{}",
            summary.route_model_identity_coverage.route_rows_with_identity,
            summary.route_model_identity_coverage.route_row_count
        ),
        format!(
            "route_model_tokenizer_template_rows={}/{}",
            summary.route_model_identity_coverage.route_rows_with_tokenizer_template,
            summary.route_model_identity_coverage.route_row_count
        ),
        format!(
            "promotion_blocker_summary_count={}",
            summary.gpu_npu_promotion_blocker_summary.len()
        ),
        format!(
            "openvino_gpu_promoted_profiles={}",
            summary.route_promotion_scope.openvino_gpu_promoted_profiles.join(",")
        ),
        format!(
            "openvino_npu_promoted_profiles={}",
            summary.route_promotion_scope.openvino_npu_promoted_profiles.join(",")
        ),
        format!(
            "profile_scoped_promotion_only={}",
            summary.route_promotion_scope.profile_scoped_promotion_only
        ),
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
        format!(
            "profile_specific_timing={}/{}",
            summary.timing_coverage.profile_specific_route_count,
            summary.timing_coverage.route_count
        ),
        format!(
            "proxy_or_missing_timing_routes_blocked={}",
            summary.timing_coverage.proxy_or_missing_timing_routes_blocked
        ),
        format!("route_model_identity_ready={}", summary.route_model_identity_ready),
        format!(
            "route_model_identity_rows={}/{}",
            summary.route_model_identity_coverage.route_rows_with_identity,
            summary.route_model_identity_coverage.route_row_count
        ),
        format!(
            "route_model_tokenizer_template_rows={}/{}",
            summary.route_model_identity_coverage.route_rows_with_tokenizer_template,
            summary.route_model_identity_coverage.route_row_count
        ),
        format!(
            "openvino_gpu_promoted_profiles={}",
            summary.route_promotion_scope.openvino_gpu_promoted_profiles.join(",")
        ),
        format!(
            "openvino_npu_promoted_profiles={}",
            summary.route_promotion_scope.openvino_npu_promoted_profiles.join(",")
        ),
        format!(
            "profile_scoped_promotion_only={}",
            summary.route_promotion_scope.profile_scoped_promotion_only
        ),
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

fn bitnet_semantic_intake_regression_notes(
    summary: &BitnetSemanticIntakeRegressionSummary,
) -> Vec<String> {
    let mut notes = vec![
        format!("intake_ready={}", summary.intake_ready),
        format!("rerun_required={}", summary.rerun_required),
        format!("pending_shared_change_count={}", summary.pending_shared_change_count),
        format!("closed_shared_change_count={}", summary.closed_shared_change_count),
        format!("merged_to_main_count={}", summary.merged_to_main_count),
        format!("stale_after_merged_count={}", summary.stale_after_merged_count),
        format!("source_lanes={}", summary.source_lanes.join(",")),
        format!("claim_boundary_preserved={}", summary.claim_boundary_preserved),
    ];
    notes.extend(summary.gaps.iter().cloned());
    notes
}

fn power_profile_regression_notes(summary: &PowerProfileRegressionSummary) -> Vec<String> {
    let mut notes = vec![
        format!("power_profile_index_ready={}", summary.power_profile_index_ready),
        format!("low_power_promotion_ready={}", summary.low_power_promotion_ready),
        format!("power_advantage_proven={}", summary.power_advantage_proven),
        format!("low_power_route_count={}", summary.low_power_route_count),
        format!(
            "low_power_routes_remain_unpromoted={}",
            summary.low_power_routes_remain_unpromoted
        ),
        format!("current_context_is_ac_only={}", summary.current_context_is_ac_only),
        format!("battery_mode_sample_recorded={}", summary.battery_mode_sample_recorded),
        format!(
            "battery_sample_source={}",
            summary.battery_sample_source.as_deref().unwrap_or("none")
        ),
        format!("energy_proxy_recorded={}", summary.energy_proxy_recorded),
        format!("energy_proxy_source={}", summary.energy_proxy_source.as_deref().unwrap_or("none")),
        format!("thermal_context_recorded={}", summary.thermal_context_recorded),
        format!("operator_runbook={}", summary.operator_runbook.as_deref().unwrap_or("none")),
        format!("next_required_evidence={}", join_or_none(&summary.next_required_evidence)),
        format!("claim_boundary_preserved={}", summary.claim_boundary_preserved),
        format!("blocker_count={}", summary.blockers.len()),
    ];
    notes.extend(summary.gaps.iter().cloned());
    notes
}

fn thermal_temperature_availability_regression_notes(
    summary: &ThermalTemperatureAvailabilityRegressionSummary,
) -> Vec<String> {
    let mut notes = vec![
        format!("thermal_zone_visibility_available={}", summary.thermal_zone_visibility_available),
        format!("thermal_temperature_available={}", summary.thermal_temperature_available),
        format!("usable_temperature_reading_count={}", summary.usable_temperature_reading_count),
        format!("measured_temperature_claim={}", summary.measured_temperature_claim),
        format!("telemetry_probe_executed={}", summary.telemetry_probe_executed),
        format!("claim_boundary_preserved={}", summary.claim_boundary_preserved),
    ];
    notes.extend(summary.gaps.iter().cloned());
    notes
}

fn operator_ask_regression_notes(summary: &OperatorAskRegressionSummary) -> Vec<String> {
    let mut notes = vec![
        format!("profile_id={}", summary.profile_id),
        format!("requested_device={}", summary.requested_device),
        format!("requested_route={}", summary.requested_route),
        format!("selected_route={}", summary.selected_route),
        format!("selected_backend={}", summary.selected_backend),
        format!("runtime_api={}", summary.runtime_api),
        format!("promotion_status={}", summary.promotion_status),
        format!(
            "route_profile_status={}",
            summary.route_profile_status.as_deref().unwrap_or("none")
        ),
        format!("fallback_used={}", summary.fallback_used),
        format!("answer_gate_passed={}", summary.answer_gate_passed),
        format!("openvino_candidate_route_executed={}", summary.openvino_candidate_route_executed),
        format!("new_inference_executed={}", summary.new_inference_executed),
        format!("generated_token_ids_available={}", summary.generated_token_ids_available),
        format!("ask_receipt_ready={}", summary.ask_receipt_ready),
    ];
    notes.extend(summary.gaps.iter().cloned());
    notes
}

fn blocked_ask_regression_notes(summary: &BlockedAskRegressionSummary) -> Vec<String> {
    let mut notes = vec![
        format!("profile_id={}", summary.profile_id),
        format!("requested_device={}", summary.requested_device),
        format!("requested_route={}", summary.requested_route),
        format!("route_selection_blocked={}", summary.route_selection_blocked),
        format!("model_path_required={}", summary.model_path_required),
        format!("model_loaded={}", summary.model_loaded),
        format!("model_resolution={}", summary.model_resolution),
        format!("candidate_routes={}", join_or_none(&summary.candidate_routes)),
        format!("why_not_cpu={}", join_or_none(&summary.why_not_cpu)),
        format!("why_not_gpu={}", join_or_none(&summary.why_not_gpu)),
        format!("why_not_npu={}", join_or_none(&summary.why_not_npu)),
        format!("operator_runbook={}", summary.operator_runbook.as_deref().unwrap_or("none")),
        format!("next_required_evidence={}", join_or_none(&summary.next_required_evidence)),
        format!("new_inference_executed={}", summary.new_inference_executed),
        format!("fallback_used={}", summary.fallback_used),
        format!("blocked_receipt_ready={}", summary.blocked_receipt_ready),
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
    if let Some(route_policy) = operator.route_policy.as_ref()
        && !route_policy.policy_ready
    {
        gaps.push(format!("operator route policy not ready: {}", route_policy.gaps.join("; ")));
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
        ask_short_ask_receipt: regression.ask_short_ask_receipt,
        ask_normal_ask_receipt: regression.ask_normal_ask_receipt,
        warm_resident_ask_receipt: regression.warm_resident_ask_receipt,
        blocked_ask_receipt: regression.blocked_ask_receipt,
        route_policy: operator.route_policy,
        default_route_id: operator.default_route.route_id.clone(),
        routes,
        evidence: operator.evidence,
        checks: regression.checks,
        gaps,
        claim_boundary: operator.claim_boundary,
    })
}

#[cfg(test)]
pub fn build_route_promotion_ledger_with_created_utc(
    root: &Path,
    operator_receipt: &Path,
    comparison_receipt: &Path,
    created_utc: String,
) -> Result<LunarLakeRoutePromotionLedger> {
    build_route_promotion_ledger_with_created_utc_and_profile_evidence(
        root,
        operator_receipt,
        comparison_receipt,
        None,
        created_utc,
    )
}

pub fn build_route_promotion_ledger_with_created_utc_and_profile_evidence(
    root: &Path,
    operator_receipt: &Path,
    comparison_receipt: &Path,
    route_profile_comparison: Option<&Path>,
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
    let profile_promotion_evidence = route_profile_comparison
        .map(|path| {
            openvino_profile_promotions_from_comparison(root, path, &operator.machine_id, &mut gaps)
        })
        .transpose()?;
    let (
        openvino_gpu_promoted_profiles,
        openvino_npu_promoted_profiles,
        profile_promotion_evidence_path,
    ) = profile_promotion_evidence.unwrap_or_default();

    let routes = operator
        .routes
        .iter()
        .map(|route| {
            promote_route(
                route,
                &operator,
                &comparison,
                &openvino_gpu_promoted_profiles,
                &openvino_npu_promoted_profiles,
                profile_promotion_evidence_path.as_deref(),
            )
        })
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
                if openvino_gpu_promoted_profiles.is_empty() {
                    "dense Qwen CPU remains the user-facing auto/default route for ask profiles".to_string()
                } else {
                    format!(
                        "OpenVINO GPU is promoted for benchmark-qualified profiles [{}]; dense Qwen CPU remains the default route id and regression baseline",
                        openvino_gpu_promoted_profiles.iter().cloned().collect::<Vec<_>>().join(",")
                    )
                },
                if openvino_npu_promoted_profiles.is_empty() {
                    "OpenVINO NPU remains candidate-only until warm resident or low-power evidence is profile-qualified".to_string()
                } else {
                    format!(
                        "OpenVINO NPU is promoted only for profile-qualified warm resident profiles [{}]; cold and low-power routes remain blocked unless separately qualified",
                        openvino_npu_promoted_profiles.iter().cloned().collect::<Vec<_>>().join(",")
                    )
                },
                "OpenVINO GPU and NPU routes require profile-specific answer, fallback, phase, regression, and latency-or-power evidence before promotion".to_string(),
                "BitNet remains a CPU reference route until accelerator BitNet parity and timing evidence exists".to_string(),
            ],
        },
        workload_profiles: workload_profiles_with_openvino_promotions(
            &openvino_gpu_promoted_profiles,
            &openvino_npu_promoted_profiles,
        ),
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
        None,
        created_utc,
    )
}

#[cfg(test)]
pub fn build_route_profile_comparison_with_created_utc_and_inputs(
    root: &Path,
    promotion_ledger: &Path,
    phase_comparison: &Path,
    answer_corpus_v2: Option<&Path>,
    cpu_corpus_v2: Option<&Path>,
    openvino_corpus_v2: Option<&Path>,
    telemetry_context: Option<&Path>,
    created_utc: String,
) -> Result<LunarLakeRouteProfileComparison> {
    build_route_profile_comparison_with_created_utc_and_diagnostics(
        root,
        promotion_ledger,
        phase_comparison,
        answer_corpus_v2,
        cpu_corpus_v2,
        openvino_corpus_v2,
        telemetry_context,
        None,
        None,
        None,
        None,
        None,
        created_utc,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn build_route_profile_comparison_with_created_utc_and_diagnostics(
    root: &Path,
    promotion_ledger: &Path,
    phase_comparison: &Path,
    answer_corpus_v2: Option<&Path>,
    cpu_corpus_v2: Option<&Path>,
    openvino_corpus_v2: Option<&Path>,
    telemetry_context: Option<&Path>,
    gpu_quality_diagnosis: Option<&Path>,
    npu_quality_diagnosis: Option<&Path>,
    npu_cold_start_diagnosis: Option<&Path>,
    npu_resident_session: Option<&Path>,
    npu_cache_experiment: Option<&Path>,
    created_utc: String,
) -> Result<LunarLakeRouteProfileComparison> {
    build_route_profile_comparison_with_created_utc_and_budget_diagnostics(
        root,
        promotion_ledger,
        phase_comparison,
        answer_corpus_v2,
        cpu_corpus_v2,
        openvino_corpus_v2,
        telemetry_context,
        gpu_quality_diagnosis,
        npu_quality_diagnosis,
        npu_cold_start_diagnosis,
        npu_resident_session,
        npu_cache_experiment,
        None,
        None,
        created_utc,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn build_route_profile_comparison_with_created_utc_and_budget_diagnostics(
    root: &Path,
    promotion_ledger: &Path,
    phase_comparison: &Path,
    answer_corpus_v2: Option<&Path>,
    cpu_corpus_v2: Option<&Path>,
    openvino_corpus_v2: Option<&Path>,
    telemetry_context: Option<&Path>,
    gpu_quality_diagnosis: Option<&Path>,
    npu_quality_diagnosis: Option<&Path>,
    npu_cold_start_diagnosis: Option<&Path>,
    npu_resident_session: Option<&Path>,
    npu_cache_experiment: Option<&Path>,
    openvino_budget_sensitivity: Option<&Path>,
    cpu_profile_run: Option<&Path>,
    created_utc: String,
) -> Result<LunarLakeRouteProfileComparison> {
    let promotion_ledger_path = resolve_receipt_path(root, promotion_ledger);
    let phase_comparison_path = resolve_receipt_path(root, phase_comparison);
    let ledger: LunarLakeRoutePromotionLedger = read_json_receipt(&promotion_ledger_path)?;
    let phase_comparison_json: Value = read_json_receipt(&phase_comparison_path)?;
    let quality_index = load_profile_quality_index(root, cpu_corpus_v2, openvino_corpus_v2)?;
    let route_identity_index = load_route_model_identity_index(root)?;

    let mut gaps = Vec::new();
    let corpus_alignment =
        load_corpus_case_alignment_index(root, answer_corpus_v2, &quality_index, &mut gaps)?;
    let telemetry_context = load_benchmark_telemetry_context(root, telemetry_context, &mut gaps)?;
    let route_diagnostics = load_route_diagnostics_index(
        root,
        gpu_quality_diagnosis,
        npu_quality_diagnosis,
        npu_cold_start_diagnosis,
        npu_resident_session,
        npu_cache_experiment,
        openvino_budget_sensitivity,
        &mut gaps,
    )?;
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
                &corpus_alignment,
                telemetry_context.as_ref(),
                &route_diagnostics,
                &route_identity_index,
                cpu_profile_run,
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
        }
    }

    let timing_coverage = timing_applicability_coverage(&profiles);
    let route_model_identity_coverage = route_model_identity_coverage(&profiles);
    append_route_model_identity_coverage_gaps(
        "route profile comparison",
        &route_model_identity_coverage,
        &mut gaps,
    );
    if !timing_coverage.promotion_eligible_routes_have_profile_specific_timing {
        gaps.push(format!(
            "promotion-eligible routes lack profile-specific timing: {}",
            timing_coverage.promotion_eligible_proxy_or_missing_routes.join(",")
        ));
    }
    if !timing_coverage.proxy_or_missing_timing_routes_blocked {
        gaps.push(format!(
            "proxy or missing timing routes lack promotion blockers: {}",
            timing_coverage.unblocked_proxy_or_missing_routes.join(",")
        ));
    }

    let profile_comparison_ready = gaps.is_empty();
    let promotion_blocker_summary = promotion_blocker_summary(&profiles);
    let route_promotion_scope = route_promotion_scope_from_profile_comparison(&profiles);
    Ok(LunarLakeRouteProfileComparison {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_route_profile_comparison".to_string(),
        proof_stage: "route_profiles_indexed_no_promotion_change".to_string(),
        created_utc,
        machine_id: ledger.machine_id.clone(),
        artifact_root: path_string(root),
        promotion_ledger: path_string(&promotion_ledger_path),
        phase_comparison_receipt: path_string(&phase_comparison_path),
        answer_corpus_v2_fixture: corpus_alignment.fixture_source.clone(),
        cpu_corpus_v2_receipt: quality_index.cpu_source.clone(),
        openvino_corpus_v2_receipt: quality_index.openvino_source.clone(),
        telemetry_context_receipt: telemetry_context
            .as_ref()
            .map(|context| context.receipt.clone()),
        route_diagnosis_receipts: route_diagnostics.source_receipts(),
        profile_comparison_ready,
        default_route_id: ledger.default_route_id,
        profiles,
        timing_coverage,
        route_model_identity_coverage,
        route_promotion_scope,
        promotion_blocker_summary,
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
    let timing_coverage = if comparison.timing_coverage.route_count > 0 {
        comparison.timing_coverage.clone()
    } else {
        timing_applicability_coverage(&comparison.profiles)
    };
    let route_model_identity_coverage = cold_warm_route_model_identity_coverage(&profiles);
    append_route_model_identity_coverage_gaps(
        "cold/warm benchmark",
        &route_model_identity_coverage,
        &mut gaps,
    );
    let route_promotion_scope = route_promotion_scope_from_cold_warm(&profiles);
    if !timing_coverage.promotion_eligible_routes_have_profile_specific_timing {
        gaps.push(format!(
            "promotion-eligible routes lack profile-specific timing: {}",
            timing_coverage.promotion_eligible_proxy_or_missing_routes.join(",")
        ));
    }
    if !timing_coverage.proxy_or_missing_timing_routes_blocked {
        gaps.push(format!(
            "proxy or missing timing routes lack promotion blockers: {}",
            timing_coverage.unblocked_proxy_or_missing_routes.join(",")
        ));
    }

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
        timing_coverage,
        route_model_identity_coverage,
        route_promotion_scope,
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

pub fn build_cpu_slm_runtime_comparison_with_created_utc(
    root: &Path,
    resident_session: &Path,
    openvino_corpus_v2: &Path,
    openvino_phase_runner: &Path,
    created_utc: String,
) -> Result<LunarLakeCpuSlmRuntimeComparison> {
    let resident_path = resolve_receipt_path(root, resident_session);
    let openvino_corpus_path = resolve_receipt_path(root, openvino_corpus_v2);
    let openvino_phase_path = resolve_receipt_path(root, openvino_phase_runner);
    let resident: LunarLakeCpuSlmResidentSession = read_json_receipt(&resident_path)?;
    let openvino_corpus: Value = read_json_receipt(&openvino_corpus_path)?;
    let openvino_phase: Value = read_json_receipt(&openvino_phase_path)?;

    let mut gaps = Vec::new();
    if resident.artifact_kind != "lunar_lake_cpu_slm_resident_session" {
        gaps.push(
            "resident receipt must have artifact_kind=lunar_lake_cpu_slm_resident_session"
                .to_string(),
        );
    }
    if !resident.resident_ready {
        gaps.push("resident CPU receipt is not resident_ready=true".to_string());
    }
    if resident.claim_boundary.new_inference_executed
        || resident.claim_boundary.speedup_claim
        || resident.claim_boundary.route_promotion_changed
    {
        gaps.push("runtime comparison refuses resident receipts with inference, speedup, or route-promotion claims".to_string());
    }
    if string_at(&openvino_corpus, "artifact_kind").as_deref()
        != Some("intel_258v_dense_slm_openvino_corpus_v2")
    {
        gaps.push("OpenVINO corpus receipt must be corpus-v2".to_string());
    }
    if fallback_used(&openvino_corpus) != Some(false) {
        gaps.push("OpenVINO corpus-v2 receipt must record fallback_used=false".to_string());
    }
    if string_at(&openvino_phase, "artifact_kind").as_deref()
        != Some("intel_258v_dense_slm_openvino_phase_runner")
    {
        gaps.push(
            "OpenVINO phase-runner receipt is missing or has wrong artifact kind".to_string(),
        );
    }
    if fallback_used(&openvino_phase) != Some(false) {
        gaps.push("OpenVINO phase-runner receipt must record fallback_used=false".to_string());
    }

    let openvino_corpus_cpu = openvino_cpu_device(&openvino_corpus)
        .context("OpenVINO corpus-v2 receipt does not contain CPU device evidence")?;
    let openvino_phase_cpu = openvino_cpu_device(&openvino_phase)
        .context("OpenVINO phase-runner receipt does not contain CPU device evidence")?;

    let rust_gguf_cpu = CpuSlmRuntimeRouteSummary {
        route_id: DEFAULT_ASK_ROUTE.to_string(),
        selected_backend: resident.backend.selected_backend.clone(),
        runtime_api: resident.backend.runtime_api.clone(),
        selected_kernel_or_runtime: resident.backend.selected_kernel_or_runtime.clone(),
        fallback_used: resident.backend.fallback_used,
        answer_gate_passed: resident.backend.answer_gate_passed,
        quality_scope: "resident repeated warm-session and CPU corpus-v2 route evidence"
            .to_string(),
        timing_scope: "resident Rust GGUF CPU prompt-loop timing".to_string(),
        load_or_construct_ms: resident.resident_session.model_load_ms,
        known_gaps: vec![
            "Rust GGUF CPU timing is from resident prompt-loop evidence, not OpenVINO PerfMetrics"
                .to_string(),
        ],
    };
    let openvino_cpu = CpuSlmRuntimeRouteSummary {
        route_id: "dense_slm_openvino_cpu_candidate".to_string(),
        selected_backend: string_at(openvino_corpus_cpu, "selected_backend")
            .unwrap_or_else(|| "openvino-cpu".to_string()),
        runtime_api: string_at(openvino_corpus_cpu, "runtime_api")
            .unwrap_or_else(|| "openvino_genai".to_string()),
        selected_kernel_or_runtime: string_at(openvino_corpus_cpu, "selected_kernel_or_runtime"),
        fallback_used: fallback_used(openvino_corpus_cpu),
        answer_gate_passed: openvino_device_answer_gate_passed(openvino_corpus_cpu),
        quality_scope: "OpenVINO CPU corpus-v2 candidate evidence".to_string(),
        timing_scope: "OpenVINO GenAI generation wall time and PerfMetrics context".to_string(),
        load_or_construct_ms: number_at_any(
            openvino_corpus_cpu,
            &["pipeline_construct_wall_ms", "timing.pipeline_load_ms"],
        )
        .or_else(|| number_at_any(openvino_phase_cpu, &["pipeline_construct_wall_ms"])),
        known_gaps: vec![
            "OpenVINO GenAI generated token IDs are retokenized from text when direct IDs are unavailable".to_string(),
            "OpenVINO CPU prefill/decode_128 splits are not equivalent to GGUF CPU phase receipts".to_string(),
        ],
    };

    let mut profile_ids = BTreeSet::new();
    profile_ids.extend(resident.profiles.iter().map(|profile| profile.profile_id.clone()));
    if let Some(summary) =
        openvino_corpus_cpu.pointer("/quality_summary/profile_summary").and_then(Value::as_object)
    {
        profile_ids.extend(summary.keys().cloned());
    }

    let mut profiles = Vec::new();
    for profile_id in profile_ids {
        let rust_profile =
            resident.profiles.iter().find(|profile| profile.profile_id == profile_id);
        let rust_evidence = rust_runtime_profile_evidence(rust_profile, &resident.backend);
        let openvino_evidence = openvino_runtime_profile_evidence(openvino_corpus_cpu, &profile_id);
        let openvino_to_rust_total_ratio =
            match (openvino_evidence.timing_ms.mean, rust_evidence.timing_ms.mean) {
                (Some(openvino), Some(rust)) if rust > 0.0 => Some(openvino / rust),
                _ => None,
            };
        let mut blockers = Vec::new();
        if rust_profile.is_none() {
            blockers.push("Rust GGUF CPU resident profile evidence missing".to_string());
        }
        if openvino_evidence.cases_total.unwrap_or(0) == 0 {
            blockers.push("OpenVINO CPU corpus-v2 profile evidence missing".to_string());
        }
        if openvino_evidence.fallback_used != Some(false) {
            blockers.push("OpenVINO CPU fallback status is not fallback_used=false".to_string());
        }
        if openvino_evidence.cases_failed.unwrap_or(0) > 0 {
            blockers.push(format!(
                "OpenVINO CPU corpus-v2 profile has {} answer-gate failure(s)",
                openvino_evidence.cases_failed.unwrap_or(0)
            ));
        }
        if openvino_evidence.timing_ms.sample_count == 0 {
            blockers.push("OpenVINO CPU profile has no generation wall timing samples".to_string());
        }
        blockers.sort();
        blockers.dedup();

        let status = if blockers.is_empty() {
            "candidate_timing_context_only_no_promotion".to_string()
        } else {
            "blocked_candidate_context_only".to_string()
        };
        let mut notes = vec![
            "Runtime comparison records diagnostic timing context only; it does not promote OpenVINO CPU or claim speedup".to_string(),
        ];
        if let Some(ratio) = openvino_to_rust_total_ratio {
            notes.push(format!(
                "openvino_to_rust_total_ratio={ratio:.3}; ratio is not benchmark-qualified speedup"
            ));
        }

        profiles.push(CpuSlmRuntimeProfileComparison {
            profile_id,
            rust_cpu: rust_evidence,
            openvino_cpu: openvino_evidence,
            openvino_to_rust_total_ratio,
            status,
            blockers,
            notes,
        });
    }

    let mut findings = Vec::new();
    if let Some(load) = rust_gguf_cpu.load_or_construct_ms {
        findings.push(format!("rust_gguf_cpu_model_load_ms={load:.3}"));
    }
    if let Some(load) = openvino_cpu.load_or_construct_ms {
        findings.push(format!("openvino_cpu_pipeline_construct_ms={load:.3}"));
    }
    for profile in &profiles {
        if let Some(ratio) = profile.openvino_to_rust_total_ratio {
            findings.push(format!(
                "profile_{}_openvino_to_rust_total_ratio={ratio:.3}",
                profile.profile_id
            ));
        }
        if !profile.blockers.is_empty() {
            findings.push(format!(
                "profile_{}_openvino_candidate_blocked_by={}",
                profile.profile_id,
                profile.blockers.join("|")
            ));
        }
    }

    let recommended_next_items = vec![
        "LNL258V-GPU-QUAL-001: classify OpenVINO GPU corpus-v2 quality failures before promotion".to_string(),
        "LNL258V-NPU-COLD-001: decompose NPU cold load separately from hot decode".to_string(),
        "LNL258V-ROUTE-005: keep route promotion blocked until profile quality and timing evidence are benchmark-qualified".to_string(),
    ];
    let comparison_ready = gaps.is_empty();

    Ok(LunarLakeCpuSlmRuntimeComparison {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_cpu_slm_runtime_comparison".to_string(),
        proof_stage: "rust_gguf_cpu_vs_openvino_cpu_context_no_promotion_change".to_string(),
        created_utc,
        machine_id: "intel-258v".to_string(),
        artifact_root: path_string(root),
        source_receipts: CpuSlmRuntimeComparisonSources {
            resident_session_receipt: path_string(&resident_path),
            openvino_corpus_v2_receipt: path_string(&openvino_corpus_path),
            openvino_phase_runner_receipt: path_string(&openvino_phase_path),
        },
        model: resident.model.clone(),
        rust_gguf_cpu,
        openvino_cpu,
        profiles,
        comparison_ready,
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

fn openvino_cpu_device(json: &Value) -> Option<&Value> {
    json.pointer("/generation/devices").and_then(Value::as_array)?.iter().find(|device| {
        string_at(device, "runtime_device").as_deref() == Some("CPU")
            || string_at(device, "selected_backend").as_deref() == Some("openvino-cpu")
    })
}

fn openvino_device_answer_gate_passed(device: &Value) -> Option<bool> {
    if let Some(failed) = u64_at(device, "quality_summary.failed") {
        let passed = u64_at(device, "quality_summary.passed").unwrap_or(0);
        return Some(failed == 0 && passed > 0);
    }
    let failed = u64_at(device, "failed")?;
    let passed = u64_at(device, "passed").unwrap_or(0);
    Some(failed == 0 && passed > 0)
}

fn rust_runtime_profile_evidence(
    profile: Option<&CpuSlmResidentProfileSummary>,
    backend: &CpuSlmAttributionBackend,
) -> CpuSlmRuntimeProfileEvidence {
    if let Some(profile) = profile {
        CpuSlmRuntimeProfileEvidence {
            route_id: DEFAULT_ASK_ROUTE.to_string(),
            selected_backend: backend.selected_backend.clone(),
            runtime_api: backend.runtime_api.clone(),
            fallback_used: Some(profile.fallback_observed),
            answer_gate_passed: Some(profile.answer_gate_passed),
            cases_total: Some(profile.case_ids.len() as u64),
            cases_passed: Some(if profile.answer_gate_passed {
                profile.case_ids.len() as u64
            } else {
                0
            }),
            cases_failed: Some(if profile.answer_gate_passed {
                0
            } else {
                profile.case_ids.len() as u64
            }),
            timing_ms: profile.total_ms.clone(),
            time_to_first_token_ms: profile.time_to_first_token_ms.clone(),
            tokenize_ms: profile.tokenize_ms.clone(),
            generated_tokens: profile.generated_tokens.clone(),
            throughput_tokens_per_s_mean: profile.decode_tokens_per_s_mean,
            timing_source: "rust_gguf_cpu_resident_prompt_loop".to_string(),
        }
    } else {
        CpuSlmRuntimeProfileEvidence {
            route_id: DEFAULT_ASK_ROUTE.to_string(),
            selected_backend: backend.selected_backend.clone(),
            runtime_api: backend.runtime_api.clone(),
            fallback_used: None,
            answer_gate_passed: None,
            cases_total: None,
            cases_passed: None,
            cases_failed: None,
            timing_ms: resident_metric_summary(&[]),
            time_to_first_token_ms: resident_metric_summary(&[]),
            tokenize_ms: resident_metric_summary(&[]),
            generated_tokens: resident_metric_summary(&[]),
            throughput_tokens_per_s_mean: None,
            timing_source: "missing_rust_gguf_cpu_profile".to_string(),
        }
    }
}

fn openvino_runtime_profile_evidence(
    device: &Value,
    profile_id: &str,
) -> CpuSlmRuntimeProfileEvidence {
    let summary = device
        .pointer("/quality_summary/profile_summary")
        .and_then(|summary| summary.as_object()?.get(profile_id));
    let cases = device
        .get("cases")
        .and_then(Value::as_array)
        .map(|cases| {
            cases
                .iter()
                .filter(|case| case.get("profile").and_then(Value::as_str) == Some(profile_id))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let mut timing = Vec::new();
    let mut ttft = Vec::new();
    let mut tokenize = Vec::new();
    let mut generated_tokens = Vec::new();
    let mut throughput = Vec::new();
    for case in &cases {
        push_number(case, "timing.generation_wall_ms", &mut timing);
        push_first_number(
            case,
            &[
                "timing.first_streamed_text_chunk_ms",
                "timing.openvino_perf_metrics.time_to_first_token.mean_ms",
            ],
            &mut ttft,
        );
        push_number(case, "timing.openvino_perf_metrics.tokenization.mean_ms", &mut tokenize);
        push_number(
            case,
            "timing.openvino_perf_metrics.num_generated_tokens",
            &mut generated_tokens,
        );
        push_number(case, "timing.openvino_perf_metrics.throughput.mean_ms", &mut throughput);
    }

    CpuSlmRuntimeProfileEvidence {
        route_id: "dense_slm_openvino_cpu_candidate".to_string(),
        selected_backend: string_at(device, "selected_backend")
            .unwrap_or_else(|| "openvino-cpu".to_string()),
        runtime_api: string_at(device, "runtime_api")
            .unwrap_or_else(|| "openvino_genai".to_string()),
        fallback_used: fallback_used(device),
        answer_gate_passed: summary
            .and_then(|summary| u64_at(summary, "failed").map(|failed| failed == 0))
            .or_else(|| {
                if cases.is_empty() {
                    None
                } else {
                    Some(cases.iter().all(|case| case_passed(case)))
                }
            }),
        cases_total: summary
            .and_then(|summary| u64_at(summary, "total"))
            .or_else(|| (!cases.is_empty()).then_some(cases.len() as u64)),
        cases_passed: summary.and_then(|summary| u64_at(summary, "passed")).or_else(|| {
            (!cases.is_empty())
                .then_some(cases.iter().filter(|case| case_passed(case)).count() as u64)
        }),
        cases_failed: summary.and_then(|summary| u64_at(summary, "failed")).or_else(|| {
            (!cases.is_empty())
                .then_some(cases.iter().filter(|case| !case_passed(case)).count() as u64)
        }),
        timing_ms: resident_metric_summary(&timing),
        time_to_first_token_ms: resident_metric_summary(&ttft),
        tokenize_ms: resident_metric_summary(&tokenize),
        generated_tokens: resident_metric_summary(&generated_tokens),
        throughput_tokens_per_s_mean: mean_f64(&throughput),
        timing_source: "openvino_cpu_corpus_v2_generation_wall_and_perfmetrics".to_string(),
    }
}

fn mean_f64(values: &[f64]) -> Option<f64> {
    let sum = sum_f64(values)?;
    Some(sum / values.len() as f64)
}

pub fn build_openvino_corpus_v2_diagnosis_with_created_utc(
    root: &Path,
    openvino_corpus_v2: &Path,
    answer_corpus_v2: Option<&Path>,
    runtime_device: &str,
    created_utc: String,
) -> Result<LunarLakeOpenVinoCorpusV2Diagnosis> {
    let openvino_corpus_path = resolve_receipt_path(root, openvino_corpus_v2);
    let corpus: Value = read_json_receipt(&openvino_corpus_path)?;
    let mut gaps = Vec::new();
    if string_at(&corpus, "artifact_kind").as_deref()
        != Some("intel_258v_dense_slm_openvino_corpus_v2")
    {
        gaps.push(
            "OpenVINO diagnosis requires the dense SLM OpenVINO corpus-v2 receipt".to_string(),
        );
    }
    if fallback_used(&corpus) != Some(false) {
        gaps.push("OpenVINO corpus-v2 top-level fallback_used must be false".to_string());
    }
    let device = openvino_device_by_runtime(&corpus, runtime_device)
        .with_context(|| format!("OpenVINO corpus-v2 device `{runtime_device}` is missing"))?;
    if fallback_used(device) != Some(false) {
        gaps.push(format!("OpenVINO device `{runtime_device}` did not record fallback_used=false"));
    }
    if bool_at_any(device, &["route_promotion_changed"]) == Some(true) {
        gaps.push(format!("OpenVINO device `{runtime_device}` changed route promotion"));
    }

    let answer_corpus_v2_fixture =
        answer_corpus_v2.map(|path| path_string(&resolve_receipt_path(root, path)));
    let route_id = openvino_device_route_id(device).unwrap_or("dense_slm_openvino_candidate");
    let case_alignment = diagnose_openvino_corpus_case_alignment(
        root,
        answer_corpus_v2,
        route_id,
        device,
        &mut gaps,
    )?;

    let failed_cases = device
        .get("cases")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|case| !case_passed(case))
        .map(diagnose_corpus_v2_failed_case)
        .collect::<Vec<_>>();
    let quality_summary = summarize_openvino_device_quality(device, &failed_cases);
    let profile_diagnoses = diagnose_openvino_device_profiles(device, &failed_cases);
    let generated_token_visibility = openvino_generated_token_visibility(device);
    let route_blocked = quality_summary.failed > 0
        || fallback_used(device) != Some(false)
        || !case_alignment.blockers.is_empty();
    let mut blocker_summary = corpus_v2_blocker_summary(&quality_summary, fallback_used(device));
    blocker_summary.extend(case_alignment.blockers.iter().cloned());
    if !generated_token_visibility.direct_generated_token_ids_available {
        blocker_summary.push(
            "generated token IDs are not directly available from OpenVINO GenAI pipeline internals"
                .to_string(),
        );
    }
    blocker_summary.sort();
    blocker_summary.dedup();
    let recommended_next_actions = vec![
        "Keep OpenVINO GPU/NPU routes unpromoted until failed corpus-v2 cases are cleanly rerun or intentionally re-gated".to_string(),
        "Classify yes/no, stop/EOS, long-prefill, short-reasoning, and decode-heavy failures separately before profile promotion".to_string(),
        "Preserve direct-vs-retokenized generated-token visibility in every OpenVINO candidate receipt".to_string(),
    ];
    let diagnosis_ready = gaps.is_empty();

    Ok(LunarLakeOpenVinoCorpusV2Diagnosis {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_openvino_corpus_v2_diagnosis".to_string(),
        proof_stage: "openvino_candidate_quality_diagnosis_no_promotion_change".to_string(),
        created_utc,
        machine_id: "intel-258v".to_string(),
        artifact_root: path_string(root),
        openvino_corpus_v2_receipt: path_string(&openvino_corpus_path),
        answer_corpus_v2_fixture,
        requested_runtime_device: runtime_device.to_string(),
        selected_backend: string_at(device, "selected_backend"),
        runtime_api: string_at(device, "runtime_api"),
        runtime_device: string_at(device, "runtime_device"),
        backend_lane: string_at(device, "backend_lane"),
        selected_kernel_or_runtime: string_at(device, "selected_kernel_or_runtime"),
        fallback_used: fallback_used(device),
        promotion_status: string_at(device, "promotion_status"),
        quality_summary,
        profile_diagnoses,
        failed_cases,
        case_alignment,
        generated_token_visibility,
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

fn diagnose_openvino_corpus_case_alignment(
    root: &Path,
    answer_corpus_v2: Option<&Path>,
    route_id: &str,
    device: &Value,
    gaps: &mut Vec<String>,
) -> Result<CorpusV2CaseAlignmentDiagnosis> {
    let observed = case_ids_from_json_cases(device.get("cases"));
    let observed_case_count = observed.len() as u64;
    let Some(fixture_path) = answer_corpus_v2 else {
        return Ok(CorpusV2CaseAlignmentDiagnosis {
            fixture_verified: false,
            expected_case_count: None,
            observed_case_count,
            missing_case_ids: Vec::new(),
            stale_or_unexpected_case_ids: Vec::new(),
            aligned_with_active_fixture: None,
            blockers: vec![
                "active answer corpus v2 fixture was not provided; case alignment was not verified"
                    .to_string(),
            ],
        });
    };

    let fixture_path = resolve_receipt_path(root, fixture_path);
    if !fixture_path.exists() {
        let message = format!("answer corpus v2 fixture missing: {}", path_string(&fixture_path));
        gaps.push(message.clone());
        return Ok(CorpusV2CaseAlignmentDiagnosis {
            fixture_verified: false,
            expected_case_count: None,
            observed_case_count,
            missing_case_ids: Vec::new(),
            stale_or_unexpected_case_ids: Vec::new(),
            aligned_with_active_fixture: None,
            blockers: vec![message],
        });
    }

    let (_, expected) = load_answer_corpus_v2_case_ids(&fixture_path)?;
    let missing_case_ids = expected.difference(&observed).cloned().collect::<Vec<_>>();
    let stale_or_unexpected_case_ids = observed.difference(&expected).cloned().collect::<Vec<_>>();
    let blockers = corpus_case_alignment_blockers(route_id, &expected, &observed);

    Ok(CorpusV2CaseAlignmentDiagnosis {
        fixture_verified: true,
        expected_case_count: Some(expected.len() as u64),
        observed_case_count,
        missing_case_ids,
        stale_or_unexpected_case_ids,
        aligned_with_active_fixture: Some(blockers.is_empty()),
        blockers,
    })
}

fn openvino_device_by_runtime<'a>(json: &'a Value, runtime_device: &str) -> Option<&'a Value> {
    json.pointer("/generation/devices").and_then(Value::as_array)?.iter().find(|device| {
        string_at(device, "runtime_device").as_deref() == Some(runtime_device)
            || string_at(device, "selected_backend").as_deref() == Some(runtime_device)
    })
}

pub fn build_npu_cold_start_diagnosis_with_created_utc(
    root: &Path,
    openvino_phase_runner: &Path,
    phase_comparison: &Path,
    operator_ask: &Path,
    openvino_corpus_v2: &Path,
    created_utc: String,
) -> Result<LunarLakeNpuColdStartDiagnosis> {
    let phase_runner_path = resolve_receipt_path(root, openvino_phase_runner);
    let phase_comparison_path = resolve_receipt_path(root, phase_comparison);
    let operator_ask_path = resolve_receipt_path(root, operator_ask);
    let corpus_v2_path = resolve_receipt_path(root, openvino_corpus_v2);
    let phase_runner: Value = read_json_receipt(&phase_runner_path)?;
    let phase_comparison: Value = read_json_receipt(&phase_comparison_path)?;
    let operator_ask: Value = read_json_receipt(&operator_ask_path)?;
    let corpus_v2: Value = read_json_receipt(&corpus_v2_path)?;

    let mut gaps = Vec::new();
    if string_at(&phase_runner, "artifact_kind").as_deref()
        != Some("intel_258v_dense_slm_openvino_phase_runner")
    {
        gaps.push("OpenVINO phase-runner receipt has unexpected artifact_kind".to_string());
    }
    if string_at(&phase_comparison, "artifact_kind").as_deref()
        != Some("intel_258v_dense_slm_openvino_phase_comparison")
    {
        gaps.push("OpenVINO phase-comparison receipt has unexpected artifact_kind".to_string());
    }
    if string_at(&operator_ask, "artifact_kind").as_deref()
        != Some("lunar_lake_openvino_operator_ask")
    {
        gaps.push("OpenVINO NPU operator-ask receipt has unexpected artifact_kind".to_string());
    }
    if string_at(&corpus_v2, "artifact_kind").as_deref()
        != Some("intel_258v_dense_slm_openvino_corpus_v2")
    {
        gaps.push("OpenVINO corpus-v2 receipt has unexpected artifact_kind".to_string());
    }

    let phase_npu = openvino_device_by_runtime(&phase_runner, "NPU")
        .context("OpenVINO phase-runner receipt is missing NPU device evidence")?;
    let corpus_npu = openvino_device_by_runtime(&corpus_v2, "NPU")
        .context("OpenVINO corpus-v2 receipt is missing NPU device evidence")?;
    if fallback_used(&phase_runner) != Some(false) || fallback_used(phase_npu) != Some(false) {
        gaps.push("OpenVINO phase-runner NPU evidence must record fallback_used=false".to_string());
    }
    if fallback_used(&operator_ask) != Some(false) {
        gaps.push("OpenVINO NPU operator ask must record fallback_used=false".to_string());
    }
    if fallback_used(&corpus_v2) != Some(false) || fallback_used(corpus_npu) != Some(false) {
        gaps.push("OpenVINO corpus-v2 NPU evidence must record fallback_used=false".to_string());
    }
    if bool_at_any(corpus_npu, &["route_promotion_changed"]) == Some(true) {
        gaps.push("OpenVINO NPU corpus-v2 evidence changed route promotion".to_string());
    }
    if bool_at_any(&operator_ask, &["route.acceleration_claim", "acceleration_claim"]) == Some(true)
    {
        gaps.push("OpenVINO NPU operator ask must not claim acceleration".to_string());
    }

    let route = NpuColdStartRouteIdentity {
        route_id: string_at(&operator_ask, "route_id")
            .unwrap_or_else(|| "dense_slm_openvino_npu_candidate".to_string()),
        requested_backend: string_at(&operator_ask, "requested_backend"),
        selected_backend: string_at(&operator_ask, "selected_backend"),
        runtime_api: string_at(&operator_ask, "runtime_api"),
        runtime_device: string_at(&operator_ask, "runtime_device"),
        resolved_device: string_at(&operator_ask, "resolved_device"),
        backend_lane: string_at(&operator_ask, "backend_lane"),
        selected_kernel_or_runtime: string_at(&operator_ask, "selected_kernel_or_runtime"),
        fallback_used: fallback_used(&operator_ask),
        promotion_status: string_at(corpus_npu, "promotion_status"),
    };

    let samples = vec![
        npu_operator_timing_sample(&operator_ask, path_string(&operator_ask_path)),
        npu_phase_runner_timing_sample(phase_npu, path_string(&phase_runner_path)),
        npu_phase_comparison_timing_sample(&phase_comparison, path_string(&phase_comparison_path)),
        npu_corpus_v2_timing_sample(corpus_npu, path_string(&corpus_v2_path)),
    ];
    let cold_loads = samples
        .iter()
        .filter_map(|sample| sample.openvino_load_time_ms.or(sample.pipeline_construct_wall_ms))
        .collect::<Vec<_>>();
    let generations = samples
        .iter()
        .filter_map(|sample| sample.generation_wall_ms.or(sample.case_elapsed_ms_sum))
        .collect::<Vec<_>>();
    let first_tokens = samples
        .iter()
        .filter_map(|sample| {
            sample.first_streamed_text_chunk_ms.or(sample.openvino_time_to_first_token_ms)
        })
        .collect::<Vec<_>>();
    let throughputs =
        samples.iter().filter_map(|sample| sample.throughput_tokens_per_s).collect::<Vec<_>>();
    let operator_ratio = npu_load_to_generation_ratio(&samples[0]);
    let phase_runner_ratio = npu_load_to_generation_ratio(&samples[1]);
    if cold_loads.is_empty() {
        gaps.push(
            "NPU cold-start diagnosis requires at least one load or pipeline construct timing"
                .to_string(),
        );
    }
    if generations.is_empty() {
        gaps.push(
            "NPU cold-start diagnosis requires at least one generation/hot-path timing".to_string(),
        );
    }
    let cold_load_dominant =
        [operator_ratio, phase_runner_ratio].into_iter().flatten().any(|ratio| ratio >= 10.0)
            || cold_loads.iter().any(|value| *value >= 30_000.0);
    let classification = if cold_load_dominant {
        "openvino_pipeline_load_or_device_compile_dominated".to_string()
    } else if cold_loads.is_empty() || generations.is_empty() {
        "insufficient_cold_hot_timing_evidence".to_string()
    } else {
        "not_load_dominated_from_current_receipts".to_string()
    };
    let cold_notes = vec![
        "Diagnosis is derived from committed receipts only; it does not run OpenVINO or inference"
            .to_string(),
        "OpenVINO NPU load_time_ms/pipeline_construct_wall_ms is treated as pipeline load, device compile, model transfer, or cache-miss time until cache/resident experiments split it further"
            .to_string(),
        "Hot-path timings remain candidate-route evidence and do not promote NPU".to_string(),
    ];
    let cold_start = NpuColdStartEvidence {
        classification,
        cold_load_dominant,
        samples,
        pipeline_or_load_ms: resident_metric_summary(&cold_loads),
        generation_wall_ms: resident_metric_summary(&generations),
        first_token_or_text_chunk_ms: resident_metric_summary(&first_tokens),
        throughput_tokens_per_s: resident_metric_summary(&throughputs),
        operator_load_to_generation_ratio: operator_ratio,
        phase_runner_load_to_generation_ratio: phase_runner_ratio,
        notes: cold_notes,
    };

    let hot_path = npu_hot_path_evidence(&cold_start.samples, phase_npu, &operator_ask);
    let corpus_v2_context = npu_corpus_v2_context(corpus_npu);
    let mut findings = Vec::new();
    if let Some(mean) = cold_start.pipeline_or_load_ms.mean {
        findings.push(format!("npu_pipeline_or_load_mean_ms={mean:.3}"));
    }
    if let Some(max) = cold_start.pipeline_or_load_ms.max {
        findings.push(format!("npu_pipeline_or_load_max_ms={max:.3}"));
    }
    if let Some(mean) = cold_start.generation_wall_ms.mean {
        findings.push(format!("npu_generation_wall_mean_ms={mean:.3}"));
    }
    if let Some(mean) = hot_path.throughput_tokens_per_s.mean {
        findings.push(format!("npu_hot_path_throughput_mean_tokens_per_s={mean:.3}"));
    }
    if let Some(ratio) = cold_start.operator_load_to_generation_ratio {
        findings.push(format!("operator_load_to_generation_ratio={ratio:.3}"));
    }
    if corpus_v2_context.route_blocked_by_quality {
        findings.push(format!(
            "npu_corpus_v2_blocked_by_{}_failed_cases",
            corpus_v2_context.failed.unwrap_or(0)
        ));
    }

    let recommended_next_items = vec![
        "LNL258V-NPU-CACHE-001: run OpenVINO cache hit/miss experiment with a stable cache directory".to_string(),
        "LNL258V-NPU-RESIDENT-001: measure same-process resident NPU warm asks and drift".to_string(),
        "LNL258V-NPU-QUAL-001: classify NPU corpus-v2 failures before any profile promotion".to_string(),
        "LNL258V-POWER-001: collect AC/battery or energy-proxy evidence before low-power promotion".to_string(),
    ];
    let diagnosis_ready = gaps.is_empty();

    Ok(LunarLakeNpuColdStartDiagnosis {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_openvino_npu_cold_start_diagnosis".to_string(),
        proof_stage: "candidate_route_cold_start_diagnosis_no_inference_no_promotion".to_string(),
        created_utc,
        machine_id: "intel-258v".to_string(),
        artifact_root: path_string(root),
        source_receipts: NpuColdStartSources {
            openvino_phase_runner_receipt: path_string(&phase_runner_path),
            phase_comparison_receipt: path_string(&phase_comparison_path),
            operator_ask_receipt: path_string(&operator_ask_path),
            openvino_corpus_v2_receipt: path_string(&corpus_v2_path),
        },
        route,
        cold_start,
        hot_path,
        corpus_v2_context,
        diagnosis_ready,
        findings,
        recommended_next_items,
        gaps,
        claim_boundary: NpuColdStartClaimBoundary {
            diagnostic_only: true,
            new_inference_executed: false,
            route_promotion_changed: false,
            broad_quality_claim: false,
            speedup_claim: false,
            power_advantage_claim: false,
            acceleration_claim: false,
            native_npu_inference_claim: false,
            npu_dynamic_decode_claim: false,
            beam_or_parallel_sampling_claim: false,
            bitnet_qk256_i2s_behavior_changed: false,
            dense_slm_as_bitnet_proof: false,
        },
    })
}

fn npu_operator_timing_sample(json: &Value, source: String) -> NpuTimingSample {
    let timing = value_at(json, "timing").unwrap_or(json);
    NpuTimingSample {
        source,
        evidence_scope: "single_operator_ask".to_string(),
        pipeline_construct_wall_ms: number_at_any(timing, &["pipeline_construct_wall_ms"]),
        openvino_load_time_ms: number_at_any(timing, &["openvino_perf_metrics.load_time_ms"]),
        generation_wall_ms: number_at_any(timing, &["generation_wall_ms"]),
        case_elapsed_ms_sum: None,
        first_streamed_text_chunk_ms: number_at_any(
            timing,
            &["first_streamed_text_chunk_ms", "streaming.first_text_chunk_ms"],
        ),
        openvino_time_to_first_token_ms: number_at_any(
            timing,
            &["openvino_perf_metrics.time_to_first_token.mean_ms"],
        ),
        openvino_generate_ms: number_at_any(timing, &["openvino_perf_metrics.generate.mean_ms"]),
        openvino_inference_ms: number_at_any(timing, &["openvino_perf_metrics.inference.mean_ms"]),
        openvino_tokenization_ms: number_at_any(
            timing,
            &["openvino_perf_metrics.tokenization.mean_ms"],
        ),
        throughput_tokens_per_s: number_at_any(
            timing,
            &["openvino_perf_metrics.throughput.mean_ms"],
        ),
        generated_tokens: u64_at(timing, "openvino_perf_metrics.num_generated_tokens"),
        notes: vec![
            "Operator ask includes OpenVINO PerfMetrics and wall timings for one bounded prompt"
                .to_string(),
        ],
    }
}

fn npu_phase_runner_timing_sample(device: &Value, source: String) -> NpuTimingSample {
    let cases = device.get("cases").and_then(Value::as_array).cloned().unwrap_or_default();
    let load_times = collect_case_numbers(
        &cases,
        &["openvino_perf_metrics.load_time_ms", "timing.openvino_perf_metrics.load_time_ms"],
    );
    let generation =
        collect_case_numbers(&cases, &["generation_wall_ms", "timing.generation_wall_ms"]);
    let first_chunks = collect_case_numbers(
        &cases,
        &["first_streamed_text_chunk_ms", "timing.first_streamed_text_chunk_ms"],
    );
    let ttft = collect_case_numbers(
        &cases,
        &[
            "openvino_perf_metrics.time_to_first_token.mean_ms",
            "timing.openvino_perf_metrics.time_to_first_token.mean_ms",
        ],
    );
    let generate = collect_case_numbers(
        &cases,
        &[
            "openvino_perf_metrics.generate.mean_ms",
            "timing.openvino_perf_metrics.generate.mean_ms",
        ],
    );
    let inference = collect_case_numbers(
        &cases,
        &[
            "openvino_perf_metrics.inference.mean_ms",
            "timing.openvino_perf_metrics.inference.mean_ms",
        ],
    );
    let tokenization = collect_case_numbers(
        &cases,
        &[
            "openvino_perf_metrics.tokenization.mean_ms",
            "timing.openvino_perf_metrics.tokenization.mean_ms",
        ],
    );
    let throughput = collect_case_numbers(
        &cases,
        &[
            "openvino_perf_metrics.throughput.mean_ms",
            "timing.openvino_perf_metrics.throughput.mean_ms",
        ],
    );
    let generated_tokens = collect_case_numbers(
        &cases,
        &[
            "openvino_perf_metrics.num_generated_tokens",
            "timing.openvino_perf_metrics.num_generated_tokens",
        ],
    );
    NpuTimingSample {
        source,
        evidence_scope: "three_case_phase_runner".to_string(),
        pipeline_construct_wall_ms: number_at_any(device, &["pipeline_construct_wall_ms"]),
        openvino_load_time_ms: mean_f64(&load_times),
        generation_wall_ms: mean_f64(&generation),
        case_elapsed_ms_sum: sum_f64(&generation),
        first_streamed_text_chunk_ms: mean_f64(&first_chunks),
        openvino_time_to_first_token_ms: mean_f64(&ttft),
        openvino_generate_ms: mean_f64(&generate),
        openvino_inference_ms: mean_f64(&inference),
        openvino_tokenization_ms: mean_f64(&tokenization),
        throughput_tokens_per_s: mean_f64(&throughput),
        generated_tokens: sum_f64(&generated_tokens).map(|value| value as u64),
        notes: vec![
            "Phase runner averages per-case OpenVINO PerfMetrics for the NPU device".to_string(),
        ],
    }
}

fn npu_phase_comparison_timing_sample(json: &Value, source: String) -> NpuTimingSample {
    let npu = value_at(json, "openvino_paths.npu").unwrap_or(json);
    let timing = value_at(npu, "timing").unwrap_or(npu);
    NpuTimingSample {
        source,
        evidence_scope: "indexed_phase_comparison".to_string(),
        pipeline_construct_wall_ms: number_at_any(timing, &["pipeline_load_ms"]),
        openvino_load_time_ms: number_at_any(timing, &["pipeline_load_ms"]),
        generation_wall_ms: None,
        case_elapsed_ms_sum: number_at_any(timing, &["case_elapsed_ms_sum"]),
        first_streamed_text_chunk_ms: None,
        openvino_time_to_first_token_ms: None,
        openvino_generate_ms: None,
        openvino_inference_ms: None,
        openvino_tokenization_ms: None,
        throughput_tokens_per_s: None,
        generated_tokens: None,
        notes: vec![
            "Phase comparison indexes smoke-level pipeline load and per-case elapsed sums only"
                .to_string(),
        ],
    }
}

fn npu_corpus_v2_timing_sample(device: &Value, source: String) -> NpuTimingSample {
    let cases = device.get("cases").and_then(Value::as_array).cloned().unwrap_or_default();
    let generation = collect_case_numbers(&cases, &["timing.generation_wall_ms"]);
    let first_chunks = collect_case_numbers(&cases, &["timing.first_streamed_text_chunk_ms"]);
    let load_times = collect_case_numbers(&cases, &["timing.openvino_perf_metrics.load_time_ms"]);
    let ttft =
        collect_case_numbers(&cases, &["timing.openvino_perf_metrics.time_to_first_token.mean_ms"]);
    let generate = collect_case_numbers(&cases, &["timing.openvino_perf_metrics.generate.mean_ms"]);
    let inference =
        collect_case_numbers(&cases, &["timing.openvino_perf_metrics.inference.mean_ms"]);
    let tokenization =
        collect_case_numbers(&cases, &["timing.openvino_perf_metrics.tokenization.mean_ms"]);
    let throughput =
        collect_case_numbers(&cases, &["timing.openvino_perf_metrics.throughput.mean_ms"]);
    let generated_tokens =
        collect_case_numbers(&cases, &["timing.openvino_perf_metrics.num_generated_tokens"]);
    NpuTimingSample {
        source,
        evidence_scope: "corpus_v2_profile_quality_receipt".to_string(),
        pipeline_construct_wall_ms: number_at_any(device, &["pipeline_construct_wall_ms"]),
        openvino_load_time_ms: mean_f64(&load_times),
        generation_wall_ms: mean_f64(&generation),
        case_elapsed_ms_sum: sum_f64(&generation),
        first_streamed_text_chunk_ms: mean_f64(&first_chunks),
        openvino_time_to_first_token_ms: mean_f64(&ttft),
        openvino_generate_ms: mean_f64(&generate),
        openvino_inference_ms: mean_f64(&inference),
        openvino_tokenization_ms: mean_f64(&tokenization),
        throughput_tokens_per_s: mean_f64(&throughput),
        generated_tokens: sum_f64(&generated_tokens).map(|value| value as u64),
        notes: vec![
            "Corpus-v2 timing is profile-quality context; failed cases still block promotion"
                .to_string(),
        ],
    }
}

fn collect_case_numbers(cases: &[Value], paths: &[&str]) -> Vec<f64> {
    cases.iter().filter_map(|case| number_at_any(case, paths)).collect()
}

fn npu_load_to_generation_ratio(sample: &NpuTimingSample) -> Option<f64> {
    let load = sample.openvino_load_time_ms.or(sample.pipeline_construct_wall_ms)?;
    let generation = sample.generation_wall_ms.or(sample.case_elapsed_ms_sum)?;
    (generation > 0.0).then_some(load / generation)
}

fn npu_hot_path_evidence(
    samples: &[NpuTimingSample],
    phase_npu: &Value,
    operator_ask: &Value,
) -> NpuHotPathEvidence {
    let generation =
        samples.iter().filter_map(|sample| sample.generation_wall_ms).collect::<Vec<_>>();
    let first_chunks =
        samples.iter().filter_map(|sample| sample.first_streamed_text_chunk_ms).collect::<Vec<_>>();
    let ttft = samples
        .iter()
        .filter_map(|sample| sample.openvino_time_to_first_token_ms)
        .collect::<Vec<_>>();
    let throughput =
        samples.iter().filter_map(|sample| sample.throughput_tokens_per_s).collect::<Vec<_>>();
    let generated_tokens = samples
        .iter()
        .filter_map(|sample| sample.generated_tokens.map(|value| value as f64))
        .collect::<Vec<_>>();
    let hot_path_interesting = throughput.iter().any(|value| *value >= 8.0)
        || generation.iter().any(|value| *value <= 1500.0);
    NpuHotPathEvidence {
        bounded_answer_gate_passed: answer_gate_passed(phase_npu)
            .or_else(|| answer_gate_passed(operator_ask)),
        fallback_used: fallback_used(phase_npu).or_else(|| fallback_used(operator_ask)),
        generation_wall_ms: resident_metric_summary(&generation),
        first_text_chunk_ms: resident_metric_summary(&first_chunks),
        openvino_time_to_first_token_ms: resident_metric_summary(&ttft),
        throughput_tokens_per_s: resident_metric_summary(&throughput),
        generated_tokens: resident_metric_summary(&generated_tokens),
        hot_path_interesting,
        notes: vec![
            "Hot-path evidence is bounded and candidate-only until resident/corpus/power gates pass"
                .to_string(),
            "Cold-start policy remains unresolved for one-off interactive asks".to_string(),
        ],
    }
}

fn npu_corpus_v2_context(device: &Value) -> NpuCorpusV2Context {
    let quality = value_at(device, "quality_summary");
    let cases_total = quality.and_then(|value| u64_at(value, "cases_total"));
    let passed = quality.and_then(|value| u64_at(value, "passed"));
    let failed = quality.and_then(|value| u64_at(value, "failed"));
    let mut failed_profiles = Vec::new();
    if let Some(summary) =
        value_at(device, "quality_summary.profile_summary").and_then(Value::as_object)
    {
        for (profile, value) in summary {
            if u64_at(value, "failed").unwrap_or(0) > 0 {
                failed_profiles.push(profile.clone());
            }
        }
    }
    let mut failed_categories = Vec::new();
    if let Some(summary) =
        value_at(device, "quality_summary.category_summary").and_then(Value::as_object)
    {
        for (category, value) in summary {
            if u64_at(value, "failed").unwrap_or(0) > 0 {
                failed_categories.push(category.clone());
            }
        }
    }
    let cases = device.get("cases").and_then(Value::as_array).cloned().unwrap_or_default();
    let direct_generated_token_ids_available = cases.iter().any(|case| {
        bool_at_any(case, &["generated_token_ids_available_from_pipeline"]) == Some(true)
    });
    let generated_token_id_source = cases
        .iter()
        .filter_map(|case| string_at(case, "generated_token_ids_source"))
        .next()
        .unwrap_or_else(|| "not_reported".to_string());
    NpuCorpusV2Context {
        cases_total,
        passed,
        failed,
        route_blocked_by_quality: failed.unwrap_or(0) > 0,
        failed_profiles,
        failed_categories,
        direct_generated_token_ids_available,
        generated_token_id_source,
    }
}

fn summarize_openvino_device_quality(
    device: &Value,
    failed_cases: &[CorpusV2FailedCaseDiagnosis],
) -> CorpusV2QualitySummary {
    let quality = value_at(device, "quality_summary");
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
        total: quality.and_then(|value| u64_at(value, "cases_total")).unwrap_or(0),
        passed: quality.and_then(|value| u64_at(value, "passed")).unwrap_or(0),
        failed: quality
            .and_then(|value| u64_at(value, "failed"))
            .unwrap_or(failed_cases.len() as u64),
        timeout: 0,
        not_run: 0,
        failed_profiles,
        failed_categories,
        failure_classes,
    }
}

fn diagnose_openvino_device_profiles(
    device: &Value,
    failed_cases: &[CorpusV2FailedCaseDiagnosis],
) -> Vec<CorpusV2ProfileDiagnosis> {
    let mut profile_ids = BTreeSet::<String>::new();
    if let Some(summary) =
        value_at(device, "quality_summary.profile_summary").and_then(Value::as_object)
    {
        profile_ids.extend(summary.keys().cloned());
    }
    profile_ids.extend(failed_cases.iter().map(|case| case.profile.clone()));
    profile_ids
        .into_iter()
        .map(|profile_id| {
            let summary = value_at(device, "quality_summary.profile_summary")
                .and_then(|value| value.get(&profile_id))
                .unwrap_or(&Value::Null);
            let failed_case_ids = failed_cases
                .iter()
                .filter(|case| case.profile == profile_id)
                .map(|case| case.id.clone())
                .collect::<Vec<_>>();
            let failed = u64_at(summary, "failed").unwrap_or(failed_case_ids.len() as u64);
            let mut route_blockers = Vec::new();
            if failed > 0 {
                route_blockers.push(format!(
                    "OpenVINO candidate profile has {failed} answer-gate failure(s)"
                ));
            }
            CorpusV2ProfileDiagnosis {
                profile_id,
                total: u64_at(summary, "total").unwrap_or(0),
                passed: u64_at(summary, "passed").unwrap_or(0),
                failed,
                blocked: failed > 0,
                failed_case_ids,
                route_profile_status: Some("candidate_blocked_by_quality".to_string()),
                route_blockers,
            }
        })
        .collect()
}

fn openvino_generated_token_visibility(device: &Value) -> OpenVinoGeneratedTokenVisibility {
    let mut direct = false;
    let mut retokenized = false;
    let mut sources = BTreeSet::<String>::new();
    for case in device.get("cases").and_then(Value::as_array).into_iter().flatten() {
        direct |= bool_at_any(case, &["generated_token_ids_available_from_pipeline"]) == Some(true);
        if let Some(source) = string_at(case, "generated_token_ids_source") {
            retokenized |= source.contains("retokenized");
            sources.insert(source);
        }
    }
    OpenVinoGeneratedTokenVisibility {
        direct_generated_token_ids_available: direct,
        retokenized_generated_ids_used: retokenized,
        sources: sources.into_iter().collect(),
    }
}

#[cfg(test)]
pub fn build_telemetry_context_with_created_utc(
    _root: &Path,
    created_utc: String,
) -> LunarLakeTelemetryContext {
    build_telemetry_context_with_created_utc_and_requirements(_root, created_utc, false)
}

pub fn build_telemetry_context_with_created_utc_and_requirements(
    _root: &Path,
    created_utc: String,
    require_battery: bool,
) -> LunarLakeTelemetryContext {
    let memory = collect_telemetry_memory_context();
    let power = collect_telemetry_power_context();
    let thermal = collect_telemetry_thermal_context();
    build_telemetry_context_from_parts(created_utc, memory, power, thermal, require_battery)
}

fn build_telemetry_context_from_parts(
    created_utc: String,
    memory: TelemetryMemoryContext,
    power: TelemetryPowerContext,
    thermal: TelemetryThermalContext,
    require_battery: bool,
) -> LunarLakeTelemetryContext {
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
    let capture_requirements = telemetry_capture_requirements(&power, require_battery);
    gaps.extend(capture_requirements.gaps.iter().cloned());

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

    let proof_stage = if require_battery {
        if capture_requirements.requirement_satisfied {
            "battery_mode_telemetry_context_captured_no_promotion_change"
        } else {
            "battery_mode_telemetry_context_blocked_no_promotion_change"
        }
    } else {
        "live_telemetry_context_captured_no_promotion_change"
    };
    let telemetry_scope = if require_battery {
        "low_power_battery_mode_telemetry"
    } else {
        "current_machine_runtime_telemetry"
    };

    LunarLakeTelemetryContext {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_power_thermal_context".to_string(),
        proof_stage: proof_stage.to_string(),
        created_utc,
        machine_id: "intel-258v".to_string(),
        telemetry_scope: telemetry_scope.to_string(),
        memory_context,
        power_context,
        thermal_context,
        availability,
        memory,
        power,
        thermal,
        capture_requirements,
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

fn telemetry_capture_requirements(
    power: &TelemetryPowerContext,
    require_battery: bool,
) -> TelemetryCaptureRequirements {
    let battery_mode_sample_recorded = power.ac_power_inferred == Some(false);
    let mut gaps = Vec::new();
    let (requirement_satisfied, status) = if !require_battery {
        (true, "not_required")
    } else if battery_mode_sample_recorded {
        (true, "battery_mode_sample_recorded")
    } else {
        if power.ac_power_inferred == Some(true) {
            gaps.push(
                "battery-mode telemetry sample required but current power context indicates AC power"
                    .to_string(),
            );
        } else {
            gaps.push(
                "battery-mode telemetry sample required but current power context cannot identify AC/battery state"
                    .to_string(),
            );
        }
        (false, "blocked")
    };

    TelemetryCaptureRequirements {
        battery_mode_required: require_battery,
        battery_mode_sample_recorded,
        requirement_satisfied,
        status: status.to_string(),
        gaps,
    }
}

pub fn build_power_profile_evidence_with_created_utc(
    root: &Path,
    route_profile_comparison: &Path,
    cold_warm_benchmark: &Path,
    telemetry_context: &Path,
    battery_telemetry_context: Option<&Path>,
    energy_proxy: Option<&Path>,
    created_utc: String,
) -> Result<LunarLakePowerProfileEvidence> {
    let route_profile_path = resolve_receipt_path(root, route_profile_comparison);
    let benchmark_path = resolve_receipt_path(root, cold_warm_benchmark);
    let telemetry_path = resolve_receipt_path(root, telemetry_context);
    let battery_telemetry_path =
        battery_telemetry_context.map(|path| resolve_receipt_path(root, path));
    let energy_proxy_path = energy_proxy.map(|path| resolve_receipt_path(root, path));
    let route_profile_json: Value = read_json_receipt(&route_profile_path)?;
    let benchmark_json: Value = read_json_receipt(&benchmark_path)?;
    let telemetry_json: Value = read_json_receipt(&telemetry_path)?;
    let battery_telemetry_json =
        battery_telemetry_path.as_ref().map(|path| read_json_receipt(path)).transpose()?;
    let energy_proxy_json =
        energy_proxy_path.as_ref().map(|path| read_json_receipt(path)).transpose()?;

    let mut gaps = Vec::new();
    if value_at(&route_profile_json, "profile_comparison_ready").and_then(Value::as_bool)
        != Some(true)
    {
        gaps.push("route profile comparison is not ready".to_string());
    }
    if value_at(&benchmark_json, "benchmark_gate_ready").and_then(Value::as_bool) != Some(true) {
        gaps.push("cold/warm benchmark is not ready".to_string());
    }
    if value_at(&route_profile_json, "claim_boundary.hidden_fallback_allowed")
        .and_then(Value::as_bool)
        == Some(true)
        || value_at(&benchmark_json, "claim_boundary.hidden_fallback_allowed")
            .and_then(Value::as_bool)
            == Some(true)
    {
        gaps.push("power-profile evidence refuses hidden fallback".to_string());
    }

    let telemetry = power_profile_telemetry_summary(
        &telemetry_json,
        battery_telemetry_json.as_ref(),
        energy_proxy_json.as_ref(),
    );
    let input_claim_boundary_preserved =
        low_power_input_claim_boundary_preserved(battery_telemetry_json.as_ref())
            && low_power_input_claim_boundary_preserved(energy_proxy_json.as_ref());
    if !input_claim_boundary_preserved {
        gaps.push(
            "low_power battery or energy proxy evidence violates no-inference/no-promotion/no-speedup/no-power-advantage/no-acceleration claim boundary"
                .to_string(),
        );
    }
    if !telemetry.power_context_recorded {
        gaps.push("power context is not recorded".to_string());
    }
    if telemetry.current_context_is_ac_only && !telemetry.battery_mode_sample_recorded {
        gaps.push(
            "current telemetry is AC-only; battery comparison evidence is missing".to_string(),
        );
    }
    if !telemetry.battery_mode_sample_recorded {
        gaps.push("battery-mode sample is missing for low_power promotion".to_string());
    }
    if !telemetry.energy_proxy_recorded {
        gaps.push("energy proxy evidence is missing for low_power promotion".to_string());
    }
    if !telemetry.thermal_context_recorded {
        gaps.push("thermal sensor context remains unavailable".to_string());
    }

    let low_power_routes = power_profile_low_power_routes(&route_profile_json, &benchmark_json);
    if low_power_routes.is_empty() {
        gaps.push("low_power route evidence is missing".to_string());
    }
    let power_advantage_proven = low_power_routes.iter().any(|route| route.power_promotion_ready);
    if !power_advantage_proven {
        gaps.push("no low_power route has benchmark-qualified power evidence".to_string());
    }
    let low_power_promotion_ready = power_advantage_proven
        && telemetry.battery_mode_sample_recorded
        && telemetry.energy_proxy_recorded
        && telemetry.power_context_recorded
        && low_power_routes
            .iter()
            .any(|route| route.route_status == "promoted" && route.power_promotion_ready);
    let power_profile_index_ready = value_at(&route_profile_json, "profiles").is_some()
        && value_at(&benchmark_json, "profiles").is_some()
        && telemetry.power_context_recorded
        && input_claim_boundary_preserved
        && !low_power_routes.is_empty();

    let operator_runbook = Some(LOW_POWER_BATTERY_RUNBOOK.to_string());
    let mut next_required_evidence = Vec::new();
    if !telemetry.battery_mode_sample_recorded {
        next_required_evidence.extend(blocked_operator_ask_next_required_evidence("low_power"));
    }
    if !telemetry.energy_proxy_recorded {
        next_required_evidence.push(
            "record an energy or battery-drain proxy across repeated low_power runs".to_string(),
        );
    }
    if !telemetry.thermal_context_recorded {
        next_required_evidence.push(
            "capture thermal context or keep thermal unavailable as an explicit blocker"
                .to_string(),
        );
    } else if telemetry.thermal_temperature_count == 0 {
        next_required_evidence.push(
            "record thermal temperatures if available; current thermal evidence is zone visibility only"
                .to_string(),
        );
    }
    next_required_evidence.push(
        "only promote low_power after answer gates, fallback=false, stable timing, and power advantage all pass"
            .to_string(),
    );
    let mut deduped_next_required_evidence = Vec::new();
    for item in next_required_evidence {
        if !deduped_next_required_evidence.contains(&item) {
            deduped_next_required_evidence.push(item);
        }
    }
    let next_required_evidence = deduped_next_required_evidence;

    Ok(LunarLakePowerProfileEvidence {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_power_profile_evidence".to_string(),
        proof_stage: "low_power_evidence_indexed_no_promotion_change".to_string(),
        created_utc,
        machine_id: "intel-258v".to_string(),
        artifact_root: path_string(root),
        route_profile_comparison_receipt: path_string(&route_profile_path),
        cold_warm_benchmark_receipt: path_string(&benchmark_path),
        telemetry_context_receipt: path_string(&telemetry_path),
        battery_telemetry_context_receipt: battery_telemetry_path
            .as_ref()
            .map(|path| path_string(path)),
        energy_proxy_receipt: energy_proxy_path.as_ref().map(|path| path_string(path)),
        telemetry,
        low_power_routes,
        power_profile_index_ready,
        low_power_promotion_ready,
        power_advantage_proven,
        gaps,
        operator_runbook,
        next_required_evidence,
        claim_boundary: PowerProfileClaimBoundary {
            new_inference_executed: false,
            route_promotion_changed: false,
            speedup_claim: false,
            power_advantage_claim: false,
            acceleration_claim: false,
            native_npu_inference_claim: false,
            bitnet_qk256_i2s_behavior_changed: false,
            hidden_fallback_allowed: false,
        },
    })
}

pub fn build_low_power_energy_proxy_with_created_utc(
    root: &Path,
    before_telemetry: &Path,
    after_telemetry: &Path,
    route_id: String,
    profile_id: String,
    sample_count: u64,
    created_utc: String,
) -> Result<LunarLakeLowPowerEnergyProxy> {
    let before_path = resolve_receipt_path(root, before_telemetry);
    let after_path = resolve_receipt_path(root, after_telemetry);
    let before_json: Value = read_json_receipt(&before_path)?;
    let after_json: Value = read_json_receipt(&after_path)?;

    let before_battery_status = string_at(&before_json, "power.battery_status");
    let after_battery_status = string_at(&after_json, "power.battery_status");
    let before_charge_percent = before_battery_status.as_deref().and_then(battery_charge_percent);
    let after_charge_percent = after_battery_status.as_deref().and_then(battery_charge_percent);
    let charge_delta_percent =
        before_charge_percent.zip(after_charge_percent).map(|(before, after)| after - before);
    let before_ac_power_inferred =
        value_at(&before_json, "power.ac_power_inferred").and_then(Value::as_bool);
    let after_ac_power_inferred =
        value_at(&after_json, "power.ac_power_inferred").and_then(Value::as_bool);
    let battery_mode_sample_recorded =
        before_ac_power_inferred == Some(false) && after_ac_power_inferred == Some(false);
    let energy_proxy_recorded = sample_count > 0 && charge_delta_percent.is_some();

    let mut gaps = Vec::new();
    if route_id.trim().is_empty() {
        gaps.push("route_id is empty".to_string());
    }
    if profile_id != "low_power" {
        gaps.push(format!("energy proxy is for profile `{profile_id}`, expected low_power"));
    }
    if sample_count == 0 {
        gaps.push("sample_count must be greater than zero".to_string());
    }
    if before_charge_percent.is_none() {
        gaps.push("before telemetry is missing EstimatedChargeRemaining".to_string());
    }
    if after_charge_percent.is_none() {
        gaps.push("after telemetry is missing EstimatedChargeRemaining".to_string());
    }
    if !battery_mode_sample_recorded {
        gaps.push(
            "before and after telemetry must both be battery-mode samples for low_power evidence"
                .to_string(),
        );
    }

    Ok(LunarLakeLowPowerEnergyProxy {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_low_power_energy_proxy".to_string(),
        proof_stage: "battery_drain_proxy_indexed_no_promotion_change".to_string(),
        created_utc,
        machine_id: "intel-258v".to_string(),
        artifact_root: path_string(root),
        before_telemetry_context_receipt: path_string(&before_path),
        after_telemetry_context_receipt: path_string(&after_path),
        route_id,
        profile_id,
        sample_count,
        before_battery_status,
        after_battery_status,
        before_charge_percent,
        after_charge_percent,
        charge_delta_percent,
        before_ac_power_inferred,
        after_ac_power_inferred,
        battery_mode_sample_recorded,
        energy_proxy_recorded,
        gaps,
        claim_boundary: PowerProfileClaimBoundary {
            new_inference_executed: false,
            route_promotion_changed: false,
            speedup_claim: false,
            power_advantage_claim: false,
            acceleration_claim: false,
            native_npu_inference_claim: false,
            bitnet_qk256_i2s_behavior_changed: false,
            hidden_fallback_allowed: false,
        },
    })
}

fn battery_charge_percent(status: &str) -> Option<i64> {
    status.split(';').find_map(|field| {
        let (key, value) = field.split_once('=')?;
        (key.trim() == "EstimatedChargeRemaining").then(|| value.trim().parse().ok()).flatten()
    })
}

fn power_profile_telemetry_summary(
    telemetry_json: &Value,
    battery_telemetry_json: Option<&Value>,
    energy_proxy_json: Option<&Value>,
) -> PowerProfileTelemetrySummary {
    let memory_context_recorded = value_at(telemetry_json, "availability.memory_context_recorded")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let power_context_recorded = value_at(telemetry_json, "availability.power_context_recorded")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let thermal_context_recorded =
        value_at(telemetry_json, "availability.thermal_context_recorded")
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let active_scheme = string_at(telemetry_json, "power.active_scheme");
    let battery_status = string_at(telemetry_json, "power.battery_status");
    let ac_power_inferred =
        value_at(telemetry_json, "power.ac_power_inferred").and_then(Value::as_bool);
    let thermal_zones_visible = u64_at(telemetry_json, "thermal.thermal_zones_visible");
    let thermal_temperature_count = value_at(telemetry_json, "thermal.temperatures_celsius")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let current_context_is_ac_only = ac_power_inferred == Some(true);
    let battery_sample_source = if ac_power_inferred == Some(false) {
        Some("primary_telemetry_context".to_string())
    } else if battery_telemetry_json.is_some_and(|json| {
        value_at(json, "power.ac_power_inferred").and_then(Value::as_bool) == Some(false)
    }) {
        Some("battery_telemetry_context".to_string())
    } else {
        None
    };
    let battery_mode_sample_recorded = battery_sample_source.is_some();
    let energy_proxy_source = if low_power_energy_proxy_present(telemetry_json) {
        Some("primary_telemetry_context".to_string())
    } else if battery_telemetry_json.is_some_and(low_power_energy_proxy_present) {
        Some("battery_telemetry_context".to_string())
    } else if energy_proxy_json.is_some_and(low_power_energy_proxy_present) {
        Some("energy_proxy_receipt".to_string())
    } else {
        None
    };
    let energy_proxy_recorded = energy_proxy_source.is_some();

    PowerProfileTelemetrySummary {
        memory_context_recorded,
        power_context_recorded,
        thermal_context_recorded,
        active_scheme,
        battery_status,
        ac_power_inferred,
        thermal_zones_visible,
        thermal_temperature_count,
        current_context_is_ac_only,
        battery_mode_sample_recorded,
        battery_sample_source,
        energy_proxy_recorded,
        energy_proxy_source,
    }
}

fn low_power_energy_proxy_present(json: &Value) -> bool {
    value_at(json, "energy_proxy").is_some()
        || value_at(json, "power.energy_proxy").is_some()
        || value_at(json, "battery_delta").is_some()
        || value_at(json, "battery_delta_percent").is_some()
        || value_at(json, "charge_delta_percent").is_some()
        || value_at(json, "estimated_charge_delta_percent").is_some()
        || value_at(json, "energy_proxy_recorded").and_then(Value::as_bool) == Some(true)
}

fn low_power_input_claim_boundary_preserved(json: Option<&Value>) -> bool {
    let Some(json) = json else {
        return true;
    };
    bool_at_any(json, &["new_inference_executed", "claim_boundary.new_inference_executed"])
        != Some(true)
        && bool_at_any(json, &["route_promotion_changed", "claim_boundary.route_promotion_changed"])
            != Some(true)
        && bool_at_any(json, &["speedup_claim", "claim_boundary.speedup_claim"]) != Some(true)
        && bool_at_any(json, &["power_advantage_claim", "claim_boundary.power_advantage_claim"])
            != Some(true)
        && bool_at_any(json, &["acceleration_claim", "claim_boundary.acceleration_claim"])
            != Some(true)
        && bool_at_any(
            json,
            &["native_npu_inference_claim", "claim_boundary.native_npu_inference_claim"],
        ) != Some(true)
        && bool_at_any(
            json,
            &[
                "bitnet_qk256_i2s_behavior_changed",
                "claim_boundary.bitnet_qk256_i2s_behavior_changed",
            ],
        ) != Some(true)
        && bool_at_any(json, &["hidden_fallback_allowed", "claim_boundary.hidden_fallback_allowed"])
            != Some(true)
}

fn power_profile_low_power_routes(
    route_profile_json: &Value,
    benchmark_json: &Value,
) -> Vec<PowerProfileRouteEvidence> {
    let Some(profile) = find_profile_value(route_profile_json, "low_power") else {
        return Vec::new();
    };
    let benchmark_profile = find_profile_value(benchmark_json, "low_power");
    value_at(profile, "route_evidence")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|route| {
            let route_id = string_at(route, "route_id")?;
            let benchmark_route = benchmark_profile.and_then(|profile| {
                value_at(profile, "routes").and_then(Value::as_array).and_then(|routes| {
                    routes.iter().find(|candidate| {
                        string_at(candidate, "route_id").as_deref() == Some(route_id.as_str())
                    })
                })
            });
            let mut all_blockers = string_array_at(route, "blockers");
            if let Some(benchmark_route) = benchmark_route {
                all_blockers.extend(string_array_at(benchmark_route, "blockers"));
            }
            all_blockers.sort();
            all_blockers.dedup();
            let power_related_blockers = all_blockers
                .iter()
                .filter(|blocker| {
                    blocker.contains("power")
                        || blocker.contains("battery")
                        || blocker.contains("energy")
                        || blocker.contains("low_power")
                })
                .cloned()
                .collect::<Vec<_>>();
            let benchmark_qualified_advantage = value_at(route, "benchmark_qualified_advantage")
                .and_then(Value::as_bool)
                == Some(true)
                || benchmark_route.and_then(|route| {
                    value_at(route, "benchmark_qualified_advantage").and_then(Value::as_bool)
                }) == Some(true);
            let power_promotion_ready = benchmark_qualified_advantage
                && power_related_blockers.is_empty()
                && value_at(route, "fallback_used").and_then(Value::as_bool) == Some(false)
                && value_at(route, "answer_gate_passed").and_then(Value::as_bool) == Some(true);
            Some(PowerProfileRouteEvidence {
                route_id,
                route_status: string_at(route, "route_status")
                    .unwrap_or_else(|| "unknown".to_string()),
                ledger_route_status: string_at(route, "ledger_route_status")
                    .unwrap_or_else(|| "unknown".to_string()),
                selected_backend: string_at(route, "selected_backend")
                    .unwrap_or_else(|| "unknown".to_string()),
                runtime_api: string_at(route, "runtime_api")
                    .unwrap_or_else(|| "unknown".to_string()),
                fallback_used: value_at(route, "fallback_used").and_then(Value::as_bool),
                answer_gate_passed: value_at(route, "answer_gate_passed").and_then(Value::as_bool),
                total_response_ms: benchmark_route
                    .and_then(|route| number_at_any(route, &["timing.total_response_ms"]))
                    .or_else(|| number_at_any(route, &["timing.total_response_ms"])),
                throughput_tokens_per_s: benchmark_route
                    .and_then(|route| number_at_any(route, &["timing.throughput_tokens_per_s"]))
                    .or_else(|| number_at_any(route, &["timing.throughput_tokens_per_s"])),
                benchmark_qualified_advantage,
                power_related_blockers,
                all_blockers,
                power_promotion_ready,
            })
        })
        .collect()
}

fn find_profile_value<'a>(json: &'a Value, profile_id: &str) -> Option<&'a Value> {
    value_at(json, "profiles").and_then(Value::as_array).and_then(|profiles| {
        profiles
            .iter()
            .find(|profile| string_at(profile, "profile_id").as_deref() == Some(profile_id))
    })
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
        if let Some(thermal_zones_visible) = windows_thermal_zone_count()
            && thermal_zones_visible > 0
        {
            return TelemetryThermalContext {
                source: "windows_perf_thermal_zone".to_string(),
                thermal_zones_visible: Some(thermal_zones_visible),
                temperatures_celsius: Vec::new(),
            };
        }
        TelemetryThermalContext {
            source: "windows_thermal_probe_unavailable".to_string(),
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

#[cfg(target_os = "windows")]
fn windows_thermal_zone_count() -> Option<u64> {
    let json = command_stdout(
        "powershell",
        &[
            "-NoProfile",
            "-Command",
            "Get-CimInstance -ClassName Win32_PerfFormattedData_Counters_ThermalZoneInformation -ErrorAction SilentlyContinue | Where-Object { $_.Name } | Select-Object -ExpandProperty Name | ConvertTo-Json -Compress",
        ],
    )?;
    let value: Value = serde_json::from_str(&json).ok()?;
    let count = match value {
        Value::String(value) => u64::from(!value.trim().is_empty()),
        Value::Array(values) => values
            .into_iter()
            .filter_map(|value| value.as_str().map(str::trim).map(str::to_string))
            .filter(|value| !value.is_empty())
            .count() as u64,
        _ => 0,
    };
    nonzero_u64(count)
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
        || if profile.profile_id == "warm_resident" {
            route.timing.first_token_ms.is_some()
                && route.timing.decode_total_ms.is_some()
                && route.timing.total_response_ms.is_some()
                && route.timing.throughput_tokens_per_s.is_some()
        } else {
            route.timing.cold_load_ms.is_some()
                && route.timing.first_token_ms.is_some()
                && route.timing.decode_total_ms.is_some()
                && route.timing.throughput_tokens_per_s.is_some()
        };
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
    if route.promotion_eligible_for_profile && !critical_timing_present {
        global_gaps.push(format!(
            "{} promoted route {} is missing critical cold/warm timing",
            profile.profile_id, route.route_id
        ));
    }
    if !route.promotion_eligible_for_profile && !blockers.is_empty() {
        profile_gaps.push(format!(
            "{} route {} remains blocked: {}",
            profile.profile_id,
            route.route_id,
            blockers.join("; ")
        ));
    }

    let benchmark_qualified_advantage =
        route.benchmark_qualified_advantage && critical_timing_present;
    let promotion_blocked = !route.promotion_eligible_for_profile;
    ColdWarmRouteBenchmark {
        route_id: route.route_id.clone(),
        route_status: route.route_status.clone(),
        ledger_route_status: route.ledger_route_status.clone(),
        selected_model: route.selected_model.clone(),
        selected_backend: route.selected_backend.clone(),
        runtime_api: route.runtime_api.clone(),
        model_identity: route.model_identity.clone(),
        fallback_used: route.fallback_used,
        answer_gate_passed: route.answer_gate_passed,
        phase_timing_present: route.phase_timing_present,
        timing: route.timing.clone(),
        timing_applicability: route.timing_applicability.clone(),
        route_advantage_context: route.route_advantage_context.clone(),
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

fn thermal_context_is_unavailable(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    value == "thermal_context_unavailable"
        || value.contains("thermal_probe_unavailable")
        || value.contains("not_recorded")
        || value.contains("not_normalized")
        || value.contains("missing")
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

pub fn build_bitnet_semantic_intake_with_created_utc(
    root: &Path,
    source_changes: &Path,
    cpu_reference_bundle: &Path,
    operator_comparison: &Path,
    created_utc: String,
) -> Result<LunarLakeBitnetSemanticIntake> {
    let source_changes_path = resolve_receipt_path(root, source_changes);
    let cpu_reference_bundle_path = resolve_receipt_path(root, cpu_reference_bundle);
    let operator_comparison_path = resolve_receipt_path(root, operator_comparison);

    let source_changes_receipt: BitnetSemanticSourceChanges =
        read_json_receipt(&source_changes_path)?;
    let cpu_reference_bundle_json: Value = read_json_receipt(&cpu_reference_bundle_path)?;
    let operator_comparison_json: Value = read_json_receipt(&operator_comparison_path)?;

    let mut gaps = Vec::new();
    if source_changes_receipt.artifact_kind != "lunar_lake_bitnet_semantic_source_changes" {
        gaps.push(
            "source changes receipt must have artifact_kind=lunar_lake_bitnet_semantic_source_changes"
                .to_string(),
        );
    }
    if source_changes_receipt.changes.is_empty() {
        gaps.push("source changes receipt must list at least one shared BitNet change".to_string());
    }
    if string_at(&cpu_reference_bundle_json, "artifact_kind").as_deref()
        != Some("intel_258v_cpu_reference_bundle")
    {
        gaps.push(
            "CPU reference bundle must have artifact_kind=intel_258v_cpu_reference_bundle"
                .to_string(),
        );
    }
    if string_at(&operator_comparison_json, "artifact_kind").as_deref()
        != Some("lunar_lake_operator_comparison")
    {
        gaps.push(
            "operator comparison must have artifact_kind=lunar_lake_operator_comparison"
                .to_string(),
        );
    }
    if bool_at_any(&cpu_reference_bundle_json, &["cpu_reference.fallback_used"]) != Some(false) {
        gaps.push("CPU reference bundle must record cpu_reference.fallback_used=false".to_string());
    }
    if bool_at_any(&operator_comparison_json, &["comparison_ready"]) != Some(true) {
        gaps.push("operator comparison must be ready before semantic intake".to_string());
    }
    if bool_at_any(&operator_comparison_json, &["claim_boundary.hidden_fallback_allowed"])
        == Some(true)
    {
        gaps.push("operator comparison must not allow hidden fallback".to_string());
    }

    let cpu_created = timestamp_at_any(
        &cpu_reference_bundle_json,
        &["captured_at_utc", "created_utc"],
        "CPU reference bundle timestamp",
        &mut gaps,
    );
    let operator_created = timestamp_at_any(
        &operator_comparison_json,
        &["created_utc", "captured_at_utc"],
        "operator comparison timestamp",
        &mut gaps,
    );
    let evidence_cutoff = match (cpu_created, operator_created) {
        (Some(cpu), Some(operator)) => Some(cpu.min(operator)),
        (Some(cpu), None) => Some(cpu),
        (None, Some(operator)) => Some(operator),
        (None, None) => None,
    };

    let mut source_lanes = BTreeSet::new();
    let mut pending_changes = Vec::new();
    let mut closed_changes = Vec::new();
    let mut merged_changes = Vec::new();
    let mut changes = Vec::new();
    let mut stale_after_merged_count = 0usize;

    for change in &source_changes_receipt.changes {
        source_lanes.insert(change.source_lane.clone());
        let change_label = bitnet_semantic_change_label(change);
        let status = change.status.to_ascii_lowercase();
        let merged_to_main = bitnet_semantic_status_is_merged_to_main(&status);
        let closed_without_main_merge =
            bitnet_semantic_status_is_closed_without_main_merge(&status);
        if merged_to_main {
            merged_changes.push(change_label.clone());
        } else if closed_without_main_merge {
            closed_changes.push(change_label.clone());
        } else {
            pending_changes.push(change_label.clone());
        }
        if change.semantic_scope.is_empty() {
            gaps.push(format!("{change_label} must record at least one semantic_scope"));
        }
        if change.claim_boundary.trim().is_empty() {
            gaps.push(format!("{change_label} must record a claim boundary"));
        }

        let mut notes = Vec::new();
        let merged_at = match change.merged_at_utc.as_deref() {
            Some(timestamp) => match parse_utc_timestamp(timestamp) {
                Ok(timestamp) => Some(timestamp),
                Err(error) => {
                    gaps.push(format!("{change_label} has invalid merged_at_utc: {error:#}"));
                    None
                }
            },
            None if merged_to_main && change.requires_lunar_lake_rerun_when_merged_to_main => {
                gaps.push(format!("{change_label} is merged but missing merged_at_utc"));
                None
            }
            None => None,
        };

        let stale_after_cpu_reference = merged_at
            .zip(cpu_created)
            .is_some_and(|(merged_at, cpu_created)| merged_at > cpu_created);
        let stale_after_operator_comparison = merged_at
            .zip(operator_created)
            .is_some_and(|(merged_at, operator_created)| merged_at > operator_created);
        let lunar_lake_rerun_required = merged_to_main
            && change.requires_lunar_lake_rerun_when_merged_to_main
            && (stale_after_cpu_reference || stale_after_operator_comparison);

        if lunar_lake_rerun_required {
            stale_after_merged_count += 1;
            notes.push(
                "merged shared semantic change is newer than Lunar Lake BitNet evidence"
                    .to_string(),
            );
        } else if merged_to_main && change.requires_lunar_lake_rerun_when_merged_to_main {
            notes.push(
                "merged shared semantic change is covered by current Lunar Lake evidence timestamps"
                    .to_string(),
            );
        } else if closed_without_main_merge {
            notes.push(
                "shared semantic change is closed or superseded without main merge; no Lunar Lake rerun is required"
                    .to_string(),
            );
        } else if change.requires_lunar_lake_rerun_when_merged_to_main {
            notes.push(
                "pending shared semantic change will require Lunar Lake reruns after main merge"
                    .to_string(),
            );
        } else {
            notes.push(
                "change is tracked for visibility and does not currently require rerun".to_string(),
            );
        }

        changes.push(BitnetSemanticChangeIntake {
            source_lane: change.source_lane.clone(),
            source_pr: change.source_pr,
            title: change.title.clone(),
            status: change.status.clone(),
            semantic_scope: change.semantic_scope.clone(),
            requires_lunar_lake_rerun_when_merged_to_main: change
                .requires_lunar_lake_rerun_when_merged_to_main,
            merged_at_utc: change.merged_at_utc.clone(),
            stale_after_cpu_reference,
            stale_after_operator_comparison,
            lunar_lake_rerun_required,
            notes,
        });
    }

    pending_changes.sort();
    closed_changes.sort();
    merged_changes.sort();
    let mut source_lanes = source_lanes.into_iter().collect::<Vec<_>>();
    source_lanes.sort();

    let rerun_required = changes.iter().any(|change| change.lunar_lake_rerun_required);
    let mut required_reruns = Vec::new();
    if rerun_required {
        required_reruns.extend([
            "rerun Lunar Lake BitNet CPU answer corpus".to_string(),
            "rerun scalar-vs-AVX2 BitNet answer parity".to_string(),
            "rerun first-token divergence classifier".to_string(),
            "rerun BitNet CPU phase receipts if kernel-affecting".to_string(),
            "refresh Lunar Lake operator readiness".to_string(),
            "refresh route comparison and regression surfaces".to_string(),
        ]);
    }

    let intake_ready = gaps.is_empty() && !rerun_required;
    if rerun_required {
        gaps.push(
            "merged shared BitNet semantic changes require refreshed Lunar Lake BitNet evidence"
                .to_string(),
        );
    }

    let mut notes = Vec::new();
    if !pending_changes.is_empty() {
        notes.push(
            "pending shared changes are indexed but do not invalidate Lunar Lake receipts until they merge to main"
                .to_string(),
        );
    }
    if !closed_changes.is_empty() {
        notes.push(
            "closed or superseded shared changes are indexed for audit but do not require Lunar Lake reruns"
                .to_string(),
        );
    }
    if stale_after_merged_count == 0 {
        notes.push(
            "no merged-to-main shared semantic change currently stales Lunar Lake evidence"
                .to_string(),
        );
    }

    Ok(LunarLakeBitnetSemanticIntake {
        schema_version: "1.0.0".to_string(),
        artifact_kind: "lunar_lake_bitnet_semantic_intake".to_string(),
        proof_stage: "shared_bitnet_semantic_intake_no_new_inference".to_string(),
        created_utc,
        machine_id: source_changes_receipt.machine_id,
        artifact_root: path_string(root),
        source_changes_receipt: path_string(&source_changes_path),
        cpu_reference_bundle: path_string(&cpu_reference_bundle_path),
        operator_comparison: path_string(&operator_comparison_path),
        source_change_summary: BitnetSemanticSourceChangeSummary {
            total_change_count: source_changes_receipt.changes.len(),
            pending_shared_change_count: pending_changes.len(),
            closed_shared_change_count: closed_changes.len(),
            merged_to_main_count: merged_changes.len(),
            stale_after_merged_count,
            source_lanes,
            pending_changes,
            closed_changes,
            merged_changes,
            notes,
        },
        lunar_lake_evidence: BitnetSemanticLunarLakeEvidence {
            cpu_reference_bundle_created_utc: cpu_created.map(timestamp_string),
            operator_comparison_created_utc: operator_created.map(timestamp_string),
            evidence_cutoff_utc: evidence_cutoff.map(timestamp_string),
            cpu_reference_bundle_path: path_string(&cpu_reference_bundle_path),
            operator_comparison_path: path_string(&operator_comparison_path),
            evidence_paths: vec![
                BITNET_CPU_BUNDLE.to_string(),
                BITNET_REFERENCE.to_string(),
                BITNET_REFERENCE_DIRECT.to_string(),
                BITNET_DIVERGENCE_DIRECT.to_string(),
                BITNET_PERF_APPLIED.to_string(),
                OPERATOR_COMPARISON.to_string(),
            ],
        },
        changes,
        rerun_required,
        required_reruns,
        intake_ready,
        gaps,
        claim_boundary: BitnetSemanticIntakeClaimBoundary {
            new_inference_executed: false,
            route_promotion_changed: false,
            answer_quality_claim: false,
            speedup_claim: false,
            acceleration_claim: false,
            arc_or_npu_bitnet_claim: false,
            qk256_behavior_changed: false,
            dense_slm_as_bitnet_proof: false,
            hidden_fallback_allowed: false,
        },
    })
}

fn bitnet_semantic_change_label(change: &BitnetSemanticSourceChange) -> String {
    match change.source_pr {
        Some(source_pr) => format!("{}#{} {}", change.source_lane, source_pr, change.title),
        None => format!("{} {}", change.source_lane, change.title),
    }
}

fn bitnet_semantic_status_is_merged_to_main(status: &str) -> bool {
    matches!(status, "merged" | "merged_to_main" | "main_merged")
}

fn bitnet_semantic_status_is_closed_without_main_merge(status: &str) -> bool {
    matches!(status, "closed" | "closed_unmerged" | "abandoned" | "superseded" | "withdrawn")
}

fn parse_utc_timestamp(timestamp: &str) -> Result<chrono::DateTime<chrono::Utc>> {
    Ok(chrono::DateTime::parse_from_rfc3339(timestamp)
        .with_context(|| format!("invalid UTC timestamp `{timestamp}`"))?
        .with_timezone(&chrono::Utc))
}

fn timestamp_string(timestamp: chrono::DateTime<chrono::Utc>) -> String {
    timestamp.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn timestamp_at_any(
    json: &Value,
    paths: &[&str],
    label: &str,
    gaps: &mut Vec<String>,
) -> Option<chrono::DateTime<chrono::Utc>> {
    for path in paths {
        if let Some(value) = string_at(json, path) {
            match parse_utc_timestamp(&value) {
                Ok(timestamp) => return Some(timestamp),
                Err(error) => {
                    gaps.push(format!("{label} has invalid {path}: {error:#}"));
                    return None;
                }
            }
        }
    }
    gaps.push(format!("{label} is missing"));
    None
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
    if !matches!(
        route.workload.as_str(),
        "ask" | "dense_slm_acceleration_candidate" | "dense_slm_static_graph_candidate"
    ) {
        bail!("Lunar Lake ask route has unexpected workload `{}`", route.workload);
    }
    validate_lunar_lake_ask_route_runtime(route)?;
    if route.fallback_policy != "strict_no_fallback" {
        bail!("Lunar Lake ask route must be strict_no_fallback; got {}", route.fallback_policy);
    }
    if route.acceleration_claim {
        bail!("Lunar Lake ask route must not claim acceleration");
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

fn validate_lunar_lake_ask_route_runtime(route: &OperatorRoute) -> Result<()> {
    match (
        route.route_id.as_str(),
        route.selected_backend.as_str(),
        route.runtime_api.as_str(),
        route.selected_kernel_or_runtime.as_str(),
    ) {
        (DEFAULT_ASK_ROUTE, "cpu-rust", "cpu", "dense-qwen-cpu-reference") => Ok(()),
        (
            "dense_slm_openvino_gpu_candidate",
            "openvino-gpu",
            "openvino_genai",
            "openvino-genai-llmpipeline-gpu" | "openvino-genai-llmpipeline-gpu0",
        ) => Ok(()),
        (
            "dense_slm_openvino_npu_candidate",
            "openvino-npu",
            "openvino_genai",
            "openvino-genai-llmpipeline-npu",
        ) => Ok(()),
        _ => bail!(
            "Lunar Lake ask route `{}` has unsupported runtime identity {}/{}/{}",
            route.route_id,
            route.selected_backend,
            route.runtime_api,
            route.selected_kernel_or_runtime
        ),
    }
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
    pub route_profile_comparison: Option<String>,
    pub route_profile_status: Option<String>,
    pub route_profile_blockers: Vec<String>,
    pub route: OperatorRoute,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BlockedOperatorAskRouteSelection {
    pub requested_device: String,
    pub requested_route: String,
    pub profile_id: String,
    pub route_selection_status: String,
    pub promotion_status: String,
    pub selection_source: String,
    pub route_reason: String,
    pub candidate_routes: Vec<String>,
    pub why_not_cpu: Vec<String>,
    pub why_not_gpu: Vec<String>,
    pub why_not_npu: Vec<String>,
    pub operator_runbook: Option<String>,
    pub next_required_evidence: Vec<String>,
    pub promotion_ledger: Option<String>,
    pub route_profile_comparison: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AutoRouteProfileGuard {
    comparison_path: String,
    profile_status: String,
    blockers: Vec<String>,
}

pub fn resolve_operator_ask_route_selection(
    root: &Path,
    operator_receipt: &Path,
    promotion_ledger: &Path,
    route_profile_comparison: Option<&Path>,
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
        let profile_guard = if let Some(route_profile_comparison) = route_profile_comparison {
            Some(validate_ask_route_profile_guard(
                root,
                route_profile_comparison,
                &route.route_id,
                profile_id,
                route.route_id == DEFAULT_ASK_ROUTE,
            )?)
        } else {
            None
        };
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
            route_profile_comparison: profile_guard
                .as_ref()
                .map(|guard| guard.comparison_path.clone()),
            route_profile_status: profile_guard.as_ref().map(|guard| guard.profile_status.clone()),
            route_profile_blockers: profile_guard.map(|guard| guard.blockers).unwrap_or_default(),
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
    let Some(selected_route_id) = profile.promoted_route.as_deref() else {
        let (why_not_cpu, why_not_gpu, why_not_npu) =
            route_selection_explanations(&ledger, profile, "");
        let guidance = blocked_operator_ask_error_guidance(&profile.profile_id);
        bail!(
            "no promoted Lunar Lake auto route for profile `{profile_id}`; candidates={}; why_not_cpu={}; why_not_gpu={}; why_not_npu={}{}",
            join_or_none(&profile.candidate_routes),
            join_or_none(&why_not_cpu),
            join_or_none(&why_not_gpu),
            join_or_none(&why_not_npu),
            guidance
        );
    };
    let promotion = route_promotion(&ledger, selected_route_id)?;
    validate_auto_selected_promotion(promotion, profile_id)?;
    let profile_guard = if let Some(route_profile_comparison) = route_profile_comparison {
        Some(validate_ask_route_profile_guard(
            root,
            route_profile_comparison,
            selected_route_id,
            profile_id,
            true,
        )?)
    } else {
        None
    };
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
        route_profile_comparison: profile_guard.as_ref().map(|guard| guard.comparison_path.clone()),
        route_profile_status: profile_guard.as_ref().map(|guard| guard.profile_status.clone()),
        route_profile_blockers: profile_guard.map(|guard| guard.blockers).unwrap_or_default(),
        route,
    })
}

pub fn explain_blocked_operator_ask_route_selection(
    root: &Path,
    promotion_ledger: &Path,
    route_profile_comparison: Option<&Path>,
    requested_route: &str,
    requested_device: &str,
    profile_id: &str,
) -> Result<Option<BlockedOperatorAskRouteSelection>> {
    let requested_route = normalize_auto_selector(requested_route, DEFAULT_ASK_ROUTE);
    let requested_device = normalize_auto_selector(requested_device, "auto");
    let route_auto = requested_route.eq_ignore_ascii_case("auto");
    let device_auto = requested_device.eq_ignore_ascii_case("auto");
    if !route_auto && !device_auto {
        return Ok(None);
    }

    let ledger_path = resolve_receipt_path(root, promotion_ledger);
    let ledger: LunarLakeRoutePromotionLedger = read_json_receipt(&ledger_path)?;
    validate_auto_route_ledger(&ledger)?;
    let profile = ledger
        .workload_profiles
        .iter()
        .find(|profile| profile.profile_id == profile_id)
        .with_context(|| format!("auto route profile `{profile_id}` not found in ledger"))?;
    if profile.promoted_route.is_some() {
        return Ok(None);
    }

    let (why_not_cpu, why_not_gpu, why_not_npu) =
        route_selection_explanations(&ledger, profile, "");
    let route_profile_comparison =
        route_profile_comparison.map(|path| path_string(&resolve_receipt_path(root, path)));

    Ok(Some(BlockedOperatorAskRouteSelection {
        requested_device,
        requested_route,
        profile_id: profile.profile_id.clone(),
        route_selection_status: "blocked".to_string(),
        promotion_status: "no_promoted_route".to_string(),
        selection_source: "promotion_ledger_auto_blocked".to_string(),
        route_reason: format!(
            "no promoted Lunar Lake auto route for profile `{}`",
            profile.profile_id
        ),
        candidate_routes: profile.candidate_routes.clone(),
        why_not_cpu,
        why_not_gpu,
        why_not_npu,
        operator_runbook: blocked_operator_ask_runbook(&profile.profile_id).map(str::to_string),
        next_required_evidence: blocked_operator_ask_next_required_evidence(&profile.profile_id),
        promotion_ledger: Some(path_string(&ledger_path)),
        route_profile_comparison,
    }))
}

pub fn blocked_operator_ask_runbook(profile_id: &str) -> Option<&'static str> {
    match profile_id {
        "low_power" => Some(LOW_POWER_BATTERY_RUNBOOK),
        _ => None,
    }
}

pub fn blocked_operator_ask_next_required_evidence(profile_id: &str) -> Vec<String> {
    match profile_id {
        "low_power" => vec![
            "rerun telemetry-context --require-battery on battery power before collecting low_power route samples".to_string(),
            "collect before/after battery-mode telemetry around the CPU/GPU/NPU low_power route matrix".to_string(),
            "rebuild the low_power energy proxy, power-profile evidence, strict regression, and operator comparison before any promotion decision".to_string(),
        ],
        _ => Vec::new(),
    }
}

fn blocked_operator_ask_error_guidance(profile_id: &str) -> String {
    let next_required_evidence = blocked_operator_ask_next_required_evidence(profile_id);
    let operator_runbook = blocked_operator_ask_runbook(profile_id);
    if next_required_evidence.is_empty() && operator_runbook.is_none() {
        return String::new();
    }
    let mut parts = Vec::new();
    if !next_required_evidence.is_empty() {
        parts.push(format!("next_required_evidence={}", join_or_none(&next_required_evidence)));
    }
    if let Some(runbook) = operator_runbook {
        parts.push(format!("operator_runbook={runbook}"));
    }
    format!("; {}", parts.join("; "))
}

fn normalize_auto_selector(value: &str, default_value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() { default_value.to_string() } else { trimmed.to_string() }
}

fn join_or_none(values: &[String]) -> String {
    if values.is_empty() { "none".to_string() } else { values.join(" | ") }
}

fn join_set_or_none(values: &BTreeSet<String>) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values.iter().cloned().collect::<Vec<_>>().join(",")
    }
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
    if route.selected_backend == "openvino-gpu"
        && route.runtime_api == "openvino_genai"
        && matches!(
            normalized.as_str(),
            "gpu" | "gpu.0" | "openvino-gpu" | "dense_slm_openvino_gpu_candidate"
        )
    {
        return Ok(());
    }
    if route.selected_backend == "openvino-npu"
        && route.runtime_api == "openvino_genai"
        && matches!(
            normalized.as_str(),
            "npu" | "openvino-npu" | "dense_slm_openvino_npu_candidate"
        )
    {
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

fn validate_ask_route_profile_guard(
    root: &Path,
    route_profile_comparison: &Path,
    selected_route_id: &str,
    profile_id: &str,
    require_promotion_ready: bool,
) -> Result<AutoRouteProfileGuard> {
    let comparison_path = resolve_receipt_path(root, route_profile_comparison);
    let comparison: LunarLakeRouteProfileComparison = read_json_receipt(&comparison_path)?;
    if !comparison.profile_comparison_ready {
        bail!("route-profile comparison is not ready: {}", comparison.gaps.join("; "));
    }
    if comparison.machine_id != "intel-258v" {
        bail!(
            "Lunar Lake ask route requires route-profile machine_id=intel-258v; got {}",
            comparison.machine_id
        );
    }
    if comparison.claim_boundary.hidden_fallback_allowed {
        bail!("Lunar Lake ask route refuses route-profile receipts that allow hidden fallback");
    }
    if comparison.claim_boundary.arc_bitnet_full_inference_claimed
        || comparison.claim_boundary.npu_bitnet_full_inference_claimed
        || comparison.claim_boundary.qk256_accelerator_decode_claimed
    {
        bail!(
            "Lunar Lake ask route refuses route-profile receipts with accelerator BitNet/QK256 claims"
        );
    }

    let profile =
        comparison.profiles.iter().find(|profile| profile.profile_id == profile_id).with_context(
            || {
                format!(
                    "ask route profile `{profile_id}` not found in route-profile comparison {}",
                    comparison_path.display()
                )
            },
        )?;
    let route = profile
        .route_evidence
        .iter()
        .find(|route| route.route_id == selected_route_id)
        .with_context(|| {
            format!(
                "route-profile comparison does not include route `{selected_route_id}` for profile `{profile_id}`"
            )
        })?;

    let mut blockers = route.blockers.clone();
    blockers.extend(profile.gaps.iter().cloned());
    if require_promotion_ready && !route.promotion_eligible_for_profile {
        blockers.push(format!(
            "route `{selected_route_id}` is not promotion-eligible for profile `{profile_id}` in route-profile comparison"
        ));
    }
    if require_promotion_ready && profile.profile_status != "promoted_route_ready" {
        blockers.push(format!(
            "profile `{profile_id}` route-profile status is `{}`",
            profile.profile_status
        ));
    }
    let mut fatal_blockers = Vec::new();
    if route.fallback_used != Some(false) {
        let blocker = format!(
            "route `{selected_route_id}` does not prove fallback_used=false in route-profile comparison"
        );
        blockers.push(blocker.clone());
        fatal_blockers.push(blocker);
    }
    if require_promotion_ready {
        fatal_blockers.extend(blockers.iter().cloned());
    }
    blockers.sort();
    blockers.dedup();
    fatal_blockers.sort();
    fatal_blockers.dedup();
    if !fatal_blockers.is_empty() {
        bail!(
            "ask route `{selected_route_id}` for profile `{profile_id}` is blocked by route-profile comparison: {}",
            fatal_blockers.join("; ")
        );
    }

    Ok(AutoRouteProfileGuard {
        comparison_path: path_string(&comparison_path),
        profile_status: profile.profile_status.clone(),
        blockers,
    })
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
    let why_not_gpu = if selected_route_id == "dense_slm_openvino_gpu_candidate" {
        vec![format!(
            "dense_slm_openvino_gpu_candidate is promoted for profile {} by benchmark-qualified route evidence",
            profile.profile_id
        )]
    } else {
        route_not_selected_reasons(ledger, "dense_slm_openvino_gpu_candidate", &profile.profile_id)
    };
    let why_not_npu = if selected_route_id == "dense_slm_openvino_npu_candidate" {
        vec![format!(
            "dense_slm_openvino_npu_candidate is promoted for profile {} by route evidence",
            profile.profile_id
        )]
    } else {
        route_not_selected_reasons(ledger, "dense_slm_openvino_npu_candidate", &profile.profile_id)
    };
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
        reasons.push(describe_route_missing_evidence(item));
    }
    for blocker in &route.blocked_for {
        if route_blocker_applies_to_profile(blocker, profile_id) {
            reasons.push(describe_route_blocker(blocker, profile_id));
        }
    }
    if reasons.is_empty() {
        reasons.push(format!("route was not selected for profile `{profile_id}`"));
    }
    reasons.sort();
    reasons.dedup();
    reasons
}

fn describe_route_missing_evidence(item: &str) -> String {
    match item {
        "benchmark_qualified_speedup_or_power_advantage" => format!(
            "missing evidence: {item} (benchmark-qualified latency or power advantage is not proven)"
        ),
        _ => format!("missing evidence: {item}"),
    }
}

fn describe_route_blocker(blocker: &str, profile_id: &str) -> String {
    match blocker {
        "auto_default" => format!(
            "route blocker for profile `{profile_id}`: {blocker} (auto routing only selects routes explicitly promoted for this profile)"
        ),
        "low_power_power_advantage_unproven" => format!(
            "route blocker for profile `{profile_id}`: {blocker} (battery-mode or energy-proxy power advantage has not been benchmark-qualified)"
        ),
        _ => format!("route blocker for profile `{profile_id}`: {blocker}"),
    }
}

fn route_blocker_applies_to_profile(blocker: &str, profile_id: &str) -> bool {
    if blocker == "all_profiles" || blocker == "auto_default" || blocker.contains(profile_id) {
        return true;
    }
    match profile_id {
        "low_power" => blocker.contains("power") || blocker.contains("low_power"),
        "warm_resident" => {
            blocker.contains("warm_resident")
                || blocker.contains("resident")
                || blocker.contains("cold_start")
        }
        "ask_short" | "ask_normal" => {
            blocker.contains("dynamic_decode")
                || blocker.contains("beam_search")
                || blocker.contains("parallel_sampling")
        }
        _ => false,
    }
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
        route_reason: "Default route id and dense Qwen CPU regression baseline because strict answer gates, generated-token evidence, phase receipts, and fallback_used=false are present; profile-scoped auto routing may select promoted OpenVINO GPU/NPU routes for specific workload profiles, while low_power remains blocked until battery-mode or energy-proxy power evidence is benchmark-qualified.".to_string(),
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
        route_reason: "OpenVINO GPU dense Qwen route is profile-scoped: it may be selected only for workload profiles promoted by the route ledger from fallback-free answer, token-visibility, profile timing, and benchmark-qualified latency evidence; low_power, structured, BitNet, and any unqualified profile remain blocked without separate evidence, and this receipt makes no native OpenCL or acceleration claim.".to_string(),
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
        route_reason: "OpenVINO NPU dense Qwen route is profile-scoped: warm_resident may be selected only when the route ledger promotes the resident-session path from fallback-free quality and timing evidence; cold one-off and low_power profiles remain blocked until separately qualified, with INT4 symmetric greedy constraints and no dynamic decode, beam, parallel sampling, packed QK256, native NPU, or acceleration claim.".to_string(),
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

fn write_or_print_cpu_slm_runtime_comparison(
    receipt: &LunarLakeCpuSlmRuntimeComparison,
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
        println!("Lunar Lake CPU dense SLM runtime comparison written to {}", path.display());
    } else {
        println!("{}", String::from_utf8_lossy(&json));
    }
    Ok(())
}

fn write_or_print_openvino_corpus_v2_diagnosis(
    receipt: &LunarLakeOpenVinoCorpusV2Diagnosis,
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
        println!("Lunar Lake OpenVINO corpus-v2 diagnosis written to {}", path.display());
    } else {
        println!("{}", String::from_utf8_lossy(&json));
    }
    Ok(())
}

fn write_or_print_npu_cold_start_diagnosis(
    receipt: &LunarLakeNpuColdStartDiagnosis,
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
        println!("Lunar Lake OpenVINO NPU cold-start diagnosis written to {}", path.display());
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

fn write_or_print_power_profile_evidence(
    receipt: &LunarLakePowerProfileEvidence,
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
        println!("Lunar Lake power-profile evidence written to {}", path.display());
    } else {
        println!("{}", String::from_utf8_lossy(&json));
    }
    Ok(())
}

fn write_or_print_low_power_energy_proxy(
    receipt: &LunarLakeLowPowerEnergyProxy,
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
        println!("Lunar Lake low-power energy proxy written to {}", path.display());
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

fn write_or_print_bitnet_semantic_intake(
    receipt: &LunarLakeBitnetSemanticIntake,
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
        println!("Lunar Lake BitNet semantic-intake receipt written to {}", path.display());
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

#[derive(Debug, Clone, Default)]
struct CorpusCaseAlignmentIndex {
    fixture_source: Option<String>,
    by_route: BTreeMap<String, CorpusCaseAlignmentEvidence>,
}

#[derive(Debug, Clone, Default)]
struct CorpusCaseAlignmentEvidence {
    source_receipts: Vec<String>,
    blockers: Vec<String>,
}

impl CorpusCaseAlignmentIndex {
    fn add(
        &mut self,
        route_id: &str,
        fixture_source: String,
        receipt_source: String,
        blockers: Vec<String>,
    ) {
        let entry = self.by_route.entry(route_id.to_string()).or_default();
        for source in [fixture_source, receipt_source] {
            if !entry.source_receipts.contains(&source) {
                entry.source_receipts.push(source);
            }
        }
        entry.blockers.extend(blockers);
        entry.blockers.sort();
        entry.blockers.dedup();
    }

    fn get(&self, route_id: &str) -> Option<&CorpusCaseAlignmentEvidence> {
        self.by_route.get(route_id)
    }
}

fn load_corpus_case_alignment_index(
    root: &Path,
    answer_corpus_v2: Option<&Path>,
    quality_index: &ProfileQualityIndex,
    gaps: &mut Vec<String>,
) -> Result<CorpusCaseAlignmentIndex> {
    let mut index = CorpusCaseAlignmentIndex::default();
    let Some(fixture_path) = answer_corpus_v2 else {
        return Ok(index);
    };
    let fixture_path = resolve_receipt_path(root, fixture_path);
    if !fixture_path.exists() {
        gaps.push(format!("answer corpus v2 fixture missing: {}", path_string(&fixture_path)));
        return Ok(index);
    }
    let (fixture_source, expected_case_ids) = load_answer_corpus_v2_case_ids(&fixture_path)?;
    index.fixture_source = Some(fixture_source.clone());

    if let Some(source) = &quality_index.cpu_source {
        let path = PathBuf::from(source);
        let json: Value = read_json_receipt(&path)?;
        let observed = case_ids_from_json_cases(value_at(&json, "cases"));
        let blockers =
            corpus_case_alignment_blockers(DEFAULT_ASK_ROUTE, &expected_case_ids, &observed);
        index.add(DEFAULT_ASK_ROUTE, fixture_source.clone(), source.clone(), blockers);
    }

    if let Some(source) = &quality_index.openvino_source {
        let path = PathBuf::from(source);
        let json: Value = read_json_receipt(&path)?;
        if let Some(devices) = value_at(&json, "generation.devices").and_then(Value::as_array) {
            for device in devices {
                let Some(route_id) = openvino_device_route_id(device) else {
                    continue;
                };
                let observed = case_ids_from_json_cases(device.get("cases"));
                let blockers =
                    corpus_case_alignment_blockers(route_id, &expected_case_ids, &observed);
                index.add(route_id, fixture_source.clone(), source.clone(), blockers);
            }
        }
    }

    Ok(index)
}

fn load_answer_corpus_v2_case_ids(path: &Path) -> Result<(String, BTreeSet<String>)> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let corpus: AnswerCorpusV2Fixture = serde_yaml::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    let case_ids = corpus.cases.into_iter().map(|case| case.id).collect::<BTreeSet<_>>();
    Ok((path_string(path), case_ids))
}

fn case_ids_from_json_cases(cases: Option<&Value>) -> BTreeSet<String> {
    cases
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|case| string_at(case, "id").or_else(|| string_at(case, "case_id")))
        .collect()
}

fn corpus_case_alignment_blockers(
    route_id: &str,
    expected: &BTreeSet<String>,
    observed: &BTreeSet<String>,
) -> Vec<String> {
    let mut blockers = Vec::new();
    if observed.is_empty() {
        blockers.push(format!("{route_id} corpus-v2 receipt has no case IDs to align"));
        return blockers;
    }
    let missing = expected.difference(observed).cloned().collect::<Vec<_>>();
    if !missing.is_empty() {
        blockers.push(format!(
            "{route_id} corpus-v2 receipt is missing active fixture cases [{}]",
            missing.join(", ")
        ));
    }
    let unexpected = observed.difference(expected).cloned().collect::<Vec<_>>();
    if !unexpected.is_empty() {
        blockers.push(format!(
            "{route_id} corpus-v2 receipt has stale or unexpected cases [{}]",
            unexpected.join(", ")
        ));
    }
    blockers
}

#[derive(Debug, Clone, Default)]
struct RouteModelIdentityIndex {
    dense_gguf_cpu: RouteModelIdentity,
    dense_openvino_ir: RouteModelIdentity,
    bitnet_cpu: RouteModelIdentity,
}

impl RouteModelIdentityIndex {
    fn identity_for(&self, route: &RoutePromotion) -> RouteModelIdentity {
        match route.route_id.as_str() {
            DEFAULT_ASK_ROUTE => self.with_selected_model(&self.dense_gguf_cpu, route),
            "dense_slm_openvino_gpu_candidate" | "dense_slm_openvino_npu_candidate" => {
                self.with_selected_model(&self.dense_openvino_ir, route)
            }
            "bitnet_reference_cpu" => self.with_selected_model(&self.bitnet_cpu, route),
            _ => fallback_route_model_identity(route),
        }
    }

    fn with_selected_model(
        &self,
        identity: &RouteModelIdentity,
        route: &RoutePromotion,
    ) -> RouteModelIdentity {
        let mut identity = identity.clone();
        if identity.selected_model.is_empty() {
            identity.selected_model = route_model_label(route);
        }
        identity
    }
}

fn load_route_model_identity_index(root: &Path) -> Result<RouteModelIdentityIndex> {
    Ok(RouteModelIdentityIndex {
        dense_gguf_cpu: load_dense_gguf_route_identity(root)?,
        dense_openvino_ir: load_dense_openvino_route_identity(root)?,
        bitnet_cpu: load_bitnet_cpu_route_identity(root)?,
    })
}

fn load_dense_gguf_route_identity(root: &Path) -> Result<RouteModelIdentity> {
    let manifest_path = resolve_receipt_path(root, Path::new(DENSE_SLM_ARTIFACT_MANIFEST));
    if !manifest_path.exists() {
        return Ok(fallback_manifest_missing_identity(
            "dense_slm_gguf_manifest_missing",
            "Qwen2.5-0.5B-Instruct Q8_0 GGUF",
            &manifest_path,
        ));
    }
    let manifest: Value = read_json_receipt(&manifest_path)?;
    Ok(RouteModelIdentity {
        identity_source: "dense_slm_gguf_manifest".to_string(),
        manifest_receipt: Some(path_string(&manifest_path)),
        selected_model: "Qwen2.5-0.5B-Instruct Q8_0 GGUF".to_string(),
        model_name: string_at(&manifest, "selected_candidate.model_name"),
        model_family: string_at(&manifest, "selected_candidate.family"),
        model_format: string_at(&manifest, "selected_candidate.format"),
        model_artifact: string_at(&manifest, "selected_candidate.file"),
        model_sha256: string_at(&manifest, "selected_candidate.sha256"),
        repo: string_at(&manifest, "selected_candidate.repo"),
        repo_revision: string_at(&manifest, "selected_candidate.repo_revision"),
        quantization: string_at(&manifest, "selected_candidate.quantization"),
        tokenizer_source: string_at(&manifest, "tokenizer.source"),
        tokenizer_family: string_at(&manifest, "tokenizer.pretokenizer")
            .or_else(|| string_at(&manifest, "tokenizer.tokenizer_model")),
        prompt_template: string_at(&manifest, "tokenizer.prompt_template"),
        stop_token_policy: string_at(&manifest, "tokenizer.stop_token_policy"),
        known_gaps: Vec::new(),
    })
}

fn load_dense_openvino_route_identity(root: &Path) -> Result<RouteModelIdentity> {
    let manifest_path = resolve_receipt_path(root, Path::new(DENSE_SLM_OPENVINO_IR_MANIFEST));
    if !manifest_path.exists() {
        return Ok(fallback_manifest_missing_identity(
            "dense_slm_openvino_manifest_missing",
            "Qwen2.5-0.5B-Instruct OpenVINO IR INT4_SYM",
            &manifest_path,
        ));
    }
    let manifest: Value = read_json_receipt(&manifest_path)?;
    Ok(RouteModelIdentity {
        identity_source: "dense_slm_openvino_ir_manifest".to_string(),
        manifest_receipt: Some(path_string(&manifest_path)),
        selected_model: "Qwen2.5-0.5B-Instruct OpenVINO IR INT4_SYM".to_string(),
        model_name: string_at(&manifest, "source_model.model_name"),
        model_family: string_at(&manifest, "source_model.model_family"),
        model_format: string_at(&manifest, "export_contract.format"),
        model_artifact: string_at(&manifest, "export_contract.expected_output_dir"),
        model_sha256: None,
        repo: string_at(&manifest, "source_model.repo"),
        repo_revision: string_at(&manifest, "source_model.revision"),
        quantization: string_at(&manifest, "export_contract.weight_format").map(|format| {
            if bool_at_any(&manifest, &["export_contract.symmetric"]).unwrap_or(false) {
                format!("{format}_symmetric")
            } else {
                format
            }
        }),
        tokenizer_source: string_at(&manifest, "tokenizer.source"),
        tokenizer_family: string_at(&manifest, "tokenizer.tokenizer_family"),
        prompt_template: string_at(&manifest, "tokenizer.prompt_template"),
        stop_token_policy: string_at(&manifest, "tokenizer.stop_token_policy"),
        known_gaps: vec![
            "OpenVINO IR model binaries are not committed; manifest pins source model revision and export contract instead of a local binary SHA256".to_string(),
        ],
    })
}

fn load_bitnet_cpu_route_identity(root: &Path) -> Result<RouteModelIdentity> {
    let manifest_path = resolve_receipt_path(root, Path::new(BITNET_CPU_BUNDLE));
    if !manifest_path.exists() {
        return Ok(fallback_manifest_missing_identity(
            "bitnet_cpu_reference_bundle_missing",
            "microsoft/bitnet-b1.58-2B-4T GGUF I2_S",
            &manifest_path,
        ));
    }
    let manifest: Value = read_json_receipt(&manifest_path)?;
    Ok(RouteModelIdentity {
        identity_source: "bitnet_cpu_reference_bundle".to_string(),
        manifest_receipt: Some(path_string(&manifest_path)),
        selected_model: "microsoft/bitnet-b1.58-2B-4T GGUF I2_S".to_string(),
        model_name: string_at(&manifest, "model.file"),
        model_family: string_at(&manifest, "model.architecture"),
        model_format: string_at(&manifest, "model.format"),
        model_artifact: string_at(&manifest, "model.file"),
        model_sha256: string_at(&manifest, "model.sha256"),
        repo: string_at(&manifest, "model.repo"),
        repo_revision: None,
        quantization: Some("I2_S".to_string()),
        tokenizer_source: string_at(&manifest, "tokenizer.source"),
        tokenizer_family: string_at(&manifest, "tokenizer.type")
            .or_else(|| string_at(&manifest, "tokenizer.pretokenizer_authority")),
        prompt_template: string_at(&manifest, "cpu_reference.prompt_policy"),
        stop_token_policy: None,
        known_gaps: Vec::new(),
    })
}

fn fallback_manifest_missing_identity(
    identity_source: &str,
    selected_model: &str,
    manifest_path: &Path,
) -> RouteModelIdentity {
    RouteModelIdentity {
        identity_source: identity_source.to_string(),
        manifest_receipt: Some(path_string(manifest_path)),
        selected_model: selected_model.to_string(),
        known_gaps: vec![format!(
            "route model identity manifest missing: {}",
            manifest_path.display()
        )],
        ..RouteModelIdentity::default()
    }
}

fn fallback_route_model_identity(route: &RoutePromotion) -> RouteModelIdentity {
    RouteModelIdentity {
        identity_source: "route_promotion_ledger".to_string(),
        selected_model: route_model_label(route),
        known_gaps: vec![
            "route has no specialized model manifest mapping; identity is limited to route ledger fields"
                .to_string(),
        ],
        ..RouteModelIdentity::default()
    }
}

fn route_model_label(route: &RoutePromotion) -> String {
    match route.route_id.as_str() {
        DEFAULT_ASK_ROUTE => "Qwen2.5-0.5B-Instruct Q8_0 GGUF",
        "dense_slm_openvino_gpu_candidate" | "dense_slm_openvino_npu_candidate" => {
            "Qwen2.5-0.5B-Instruct OpenVINO IR INT4_SYM"
        }
        "bitnet_reference_cpu" => "microsoft/bitnet-b1.58-2B-4T GGUF I2_S",
        _ => route.route_id.as_str(),
    }
    .to_string()
}

fn route_model_identity_coverage(
    profiles: &[WorkloadProfileEvaluation],
) -> RouteModelIdentityCoverage {
    route_model_identity_coverage_from_entries(profiles.iter().flat_map(|profile| {
        profile.route_evidence.iter().map(|route| {
            (format!("{}:{}", profile.profile_id, route.route_id), route.model_identity.as_ref())
        })
    }))
}

fn cold_warm_route_model_identity_coverage(
    profiles: &[ColdWarmProfileBenchmark],
) -> RouteModelIdentityCoverage {
    route_model_identity_coverage_from_entries(profiles.iter().flat_map(|profile| {
        profile.routes.iter().map(|route| {
            (format!("{}:{}", profile.profile_id, route.route_id), route.model_identity.as_ref())
        })
    }))
}

fn route_model_identity_coverage_from_entries<'a>(
    entries: impl Iterator<Item = (String, Option<&'a RouteModelIdentity>)>,
) -> RouteModelIdentityCoverage {
    let mut route_row_count = 0usize;
    let mut route_rows_with_identity = 0usize;
    let mut route_rows_with_model_hash = 0usize;
    let mut route_rows_with_tokenizer_template = 0usize;
    let mut route_rows_without_model_hash_with_known_gap = 0usize;
    let mut routes_without_model_hash = BTreeSet::new();
    let mut routes_without_model_hash_missing_known_gap = BTreeSet::new();
    let mut routes_without_tokenizer_template = BTreeSet::new();

    for (route_label, identity) in entries {
        route_row_count += 1;
        if let Some(identity) = identity {
            route_rows_with_identity += 1;
            if identity.model_sha256.is_some() {
                route_rows_with_model_hash += 1;
            } else {
                routes_without_model_hash.insert(route_label.clone());
                if route_model_identity_has_explicit_no_hash_gap(identity) {
                    route_rows_without_model_hash_with_known_gap += 1;
                } else {
                    routes_without_model_hash_missing_known_gap.insert(route_label.clone());
                }
            }
            if identity.tokenizer_source.is_some() && identity.prompt_template.is_some() {
                route_rows_with_tokenizer_template += 1;
            } else {
                routes_without_tokenizer_template.insert(route_label);
            }
        } else {
            routes_without_model_hash.insert(route_label.clone());
            routes_without_model_hash_missing_known_gap.insert(route_label.clone());
            routes_without_tokenizer_template.insert(route_label);
        }
    }

    let all_route_rows_have_tokenizer_template =
        route_row_count == route_rows_with_tokenizer_template;
    let model_hash_or_explicit_gap_for_all_route_rows = route_row_count
        == route_rows_with_model_hash + route_rows_without_model_hash_with_known_gap;

    RouteModelIdentityCoverage {
        route_row_count,
        route_rows_with_identity,
        route_rows_with_model_hash,
        route_rows_with_tokenizer_template,
        route_rows_without_model_hash_with_known_gap,
        all_route_rows_have_identity: route_row_count == route_rows_with_identity,
        all_route_rows_have_tokenizer_template,
        model_hash_or_explicit_gap_for_all_route_rows,
        routes_without_model_hash: routes_without_model_hash.into_iter().collect(),
        routes_without_model_hash_missing_known_gap: routes_without_model_hash_missing_known_gap
            .into_iter()
            .collect(),
        routes_without_tokenizer_template: routes_without_tokenizer_template.into_iter().collect(),
    }
}

fn route_model_identity_coverage_ready(coverage: &RouteModelIdentityCoverage) -> bool {
    coverage.route_row_count > 0
        && coverage.all_route_rows_have_identity
        && coverage.all_route_rows_have_tokenizer_template
        && coverage.model_hash_or_explicit_gap_for_all_route_rows
}

fn route_model_identity_has_explicit_no_hash_gap(identity: &RouteModelIdentity) -> bool {
    identity.known_gaps.iter().any(|gap| {
        let gap = gap.to_ascii_lowercase();
        (gap.contains("sha256") || gap.contains("model hash"))
            && (gap.contains("no-local")
                || gap.contains("no local")
                || gap.contains("not committed")
                || gap.contains("instead of a local"))
    })
}

fn append_route_model_identity_coverage_gaps(
    surface: &str,
    coverage: &RouteModelIdentityCoverage,
    gaps: &mut Vec<String>,
) {
    if coverage.route_row_count == 0 {
        gaps.push(format!("{surface} has no route/model identity rows"));
    }
    if !coverage.all_route_rows_have_identity {
        gaps.push(format!(
            "{surface} has route rows without model identity; missing hash rows: {}",
            coverage.routes_without_model_hash.join(",")
        ));
    }
    if !coverage.all_route_rows_have_tokenizer_template {
        gaps.push(format!(
            "{surface} has route rows without tokenizer/template identity: {}",
            coverage.routes_without_tokenizer_template.join(",")
        ));
    }
    if !coverage.model_hash_or_explicit_gap_for_all_route_rows {
        gaps.push(format!(
            "{surface} has route rows without model hash or explicit no-hash gap: {}",
            coverage.routes_without_model_hash_missing_known_gap.join(",")
        ));
    }
}

#[derive(Debug, Clone, Default)]
struct RouteDiagnosticsIndex {
    by_route: BTreeMap<String, RouteDiagnosticEvidence>,
    by_route_profile: BTreeMap<(String, String), RouteDiagnosticEvidence>,
}

#[derive(Debug, Clone, Default)]
struct RouteDiagnosticEvidence {
    source_receipts: Vec<String>,
    blockers: Vec<String>,
}

impl RouteDiagnosticsIndex {
    fn add(&mut self, route_id: &str, source_receipt: String, blockers: Vec<String>) {
        let entry = self.by_route.entry(route_id.to_string()).or_default();
        add_route_diagnostic_evidence(entry, source_receipt, blockers);
    }

    fn add_for_profile(
        &mut self,
        route_id: &str,
        profile_id: &str,
        source_receipt: String,
        blockers: Vec<String>,
    ) {
        let entry = self
            .by_route_profile
            .entry((route_id.to_string(), profile_id.to_string()))
            .or_default();
        add_route_diagnostic_evidence(entry, source_receipt, blockers);
    }

    fn get(&self, route_id: &str, profile_id: &str) -> RouteDiagnosticEvidence {
        let mut evidence = self.by_route.get(route_id).cloned().unwrap_or_default();
        if let Some(profile_evidence) =
            self.by_route_profile.get(&(route_id.to_string(), profile_id.to_string()))
        {
            for source in &profile_evidence.source_receipts {
                if !evidence.source_receipts.contains(source) {
                    evidence.source_receipts.push(source.clone());
                }
            }
            evidence.blockers.extend(profile_evidence.blockers.iter().cloned());
            evidence.blockers.sort();
            evidence.blockers.dedup();
        }
        evidence
    }

    fn source_receipts(&self) -> Vec<String> {
        self.by_route
            .values()
            .chain(self.by_route_profile.values())
            .flat_map(|evidence| evidence.source_receipts.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

fn add_route_diagnostic_evidence(
    entry: &mut RouteDiagnosticEvidence,
    source_receipt: String,
    blockers: Vec<String>,
) {
    if !entry.source_receipts.contains(&source_receipt) {
        entry.source_receipts.push(source_receipt);
    }
    entry.blockers.extend(blockers);
    entry.blockers.sort();
    entry.blockers.dedup();
}

impl RouteDiagnosticsIndex {
    fn add_case_blocker(
        &mut self,
        route_id: &str,
        profile_id: Option<String>,
        source_receipt: &str,
        blocker: String,
    ) {
        if let Some(profile_id) = profile_id.filter(|profile| !profile.trim().is_empty()) {
            self.add_for_profile(route_id, &profile_id, source_receipt.to_string(), vec![blocker]);
        } else {
            self.add(route_id, source_receipt.to_string(), vec![blocker]);
        }
    }
}

fn load_route_diagnostics_index(
    root: &Path,
    gpu_quality_diagnosis: Option<&Path>,
    npu_quality_diagnosis: Option<&Path>,
    npu_cold_start_diagnosis: Option<&Path>,
    npu_resident_session: Option<&Path>,
    npu_cache_experiment: Option<&Path>,
    openvino_budget_sensitivity: Option<&Path>,
    gaps: &mut Vec<String>,
) -> Result<RouteDiagnosticsIndex> {
    let mut index = RouteDiagnosticsIndex::default();
    if let Some(path) = gpu_quality_diagnosis {
        load_openvino_quality_diagnostic(
            root,
            path,
            "dense_slm_openvino_gpu_candidate",
            &mut index,
            gaps,
        )?;
    }
    if let Some(path) = npu_quality_diagnosis {
        load_openvino_quality_diagnostic(
            root,
            path,
            "dense_slm_openvino_npu_candidate",
            &mut index,
            gaps,
        )?;
    }
    let npu_resident_session_ready = if let Some(path) = npu_resident_session {
        load_npu_resident_session(root, path, &mut index, gaps)?
    } else {
        false
    };
    if let Some(path) = npu_cold_start_diagnosis {
        load_npu_cold_start_diagnostic(root, path, npu_resident_session_ready, &mut index, gaps)?;
    }
    if let Some(path) = npu_cache_experiment {
        load_npu_cache_experiment(root, path, &mut index, gaps)?;
    }
    if let Some(path) = openvino_budget_sensitivity {
        load_openvino_generation_budget_sensitivity(root, path, &mut index, gaps)?;
    }
    Ok(index)
}

fn load_openvino_quality_diagnostic(
    root: &Path,
    path: &Path,
    route_id: &str,
    index: &mut RouteDiagnosticsIndex,
    gaps: &mut Vec<String>,
) -> Result<()> {
    let path = resolve_receipt_path(root, path);
    if !path.exists() {
        gaps.push(format!("route diagnosis receipt missing: {}", path_string(&path)));
        return Ok(());
    }
    let json: Value = read_json_receipt(&path)?;
    if string_at(&json, "artifact_kind").as_deref()
        != Some("lunar_lake_openvino_corpus_v2_diagnosis")
    {
        gaps.push(format!(
            "route diagnosis receipt has unexpected artifact_kind: {}",
            path_string(&path)
        ));
    }
    let source_receipt = path_string(&path);
    let mut global_blockers = Vec::new();
    let mut profile_blockers_indexed = false;
    if let Some(profile_diagnoses) = value_at(&json, "profile_diagnoses").and_then(Value::as_array)
    {
        for profile_diagnosis in profile_diagnoses {
            let Some(profile_id) = string_at(profile_diagnosis, "profile_id") else {
                continue;
            };
            let mut blockers = string_array_at(profile_diagnosis, "route_blockers");
            if let Some(failed) = u64_at(profile_diagnosis, "failed")
                && failed > 0
            {
                blockers.push(format!(
                    "OpenVINO diagnosis profile {profile_id} has {failed} corpus-v2 quality failure(s)"
                ));
            }
            blockers.sort();
            blockers.dedup();
            if !blockers.is_empty() {
                index.add_for_profile(route_id, &profile_id, source_receipt.clone(), blockers);
                profile_blockers_indexed = true;
            }
        }
    }
    if let Some(failed_cases) = value_at(&json, "failed_cases").and_then(Value::as_array) {
        for failed_case in failed_cases {
            let case_id =
                string_at(failed_case, "id").unwrap_or_else(|| "unknown_case".to_string());
            let classification = string_at(failed_case, "classification")
                .unwrap_or_else(|| "quality_failed".to_string());
            let profile_id = string_at(failed_case, "profile");
            index.add_case_blocker(
                route_id,
                profile_id,
                &source_receipt,
                format!("{case_id} failed corpus-v2 diagnosis: {classification}"),
            );
            profile_blockers_indexed = true;
        }
    }
    if !profile_blockers_indexed {
        if bool_at_any(&json, &["route_blocked"]) == Some(true) {
            global_blockers.push("OpenVINO corpus-v2 diagnosis keeps route blocked".to_string());
        }
        if let Some(failed) = u64_at(&json, "quality_summary.failed")
            && failed > 0
        {
            global_blockers
                .push(format!("OpenVINO diagnosis reports {failed} corpus-v2 quality failures"));
        }
        global_blockers.extend(string_array_at(&json, "blocker_summary"));
    }
    if bool_at_any(&json, &["generated_token_visibility.direct_generated_token_ids_available"])
        == Some(false)
    {
        global_blockers.push(
            "OpenVINO generated token IDs are retokenized, not direct pipeline internals"
                .to_string(),
        );
    }
    if bool_at_any(&json, &["claim_boundary.route_promotion_changed"]) == Some(true)
        || bool_at_any(&json, &["claim_boundary.speedup_claim"]) == Some(true)
        || bool_at_any(&json, &["claim_boundary.arc_or_npu_execution_claim"]) == Some(true)
    {
        gaps.push(format!(
            "route diagnosis receipt violates claim boundary: {}",
            path_string(&path)
        ));
    }
    global_blockers.sort();
    global_blockers.dedup();
    if !global_blockers.is_empty() {
        index.add(route_id, source_receipt, global_blockers);
    }
    Ok(())
}

fn load_npu_resident_session(
    root: &Path,
    path: &Path,
    index: &mut RouteDiagnosticsIndex,
    gaps: &mut Vec<String>,
) -> Result<bool> {
    let path = resolve_receipt_path(root, path);
    if !path.exists() {
        gaps.push(format!("NPU resident-session receipt missing: {}", path_string(&path)));
        return Ok(false);
    }
    let json: Value = read_json_receipt(&path)?;
    if string_at(&json, "artifact_kind").as_deref()
        != Some("lunar_lake_openvino_npu_resident_session")
    {
        gaps.push(format!(
            "NPU resident-session receipt has unexpected artifact_kind: {}",
            path_string(&path)
        ));
    }

    let source_receipt = path_string(&path);
    let mut blockers = Vec::new();
    if string_at(&json, "selected_backend").as_deref() != Some("openvino-npu") {
        blockers.push("NPU resident-session selected_backend is not openvino-npu".to_string());
    }
    if string_at(&json, "runtime_device").as_deref() != Some("NPU") {
        blockers.push("NPU resident-session runtime_device is not NPU".to_string());
    }
    if bool_at_any(&json, &["fallback_used"]) != Some(false) {
        blockers.push("NPU resident-session fallback_used=false is not proven".to_string());
    }
    let ready = bool_at_any(&json, &["resident_session.resident_session_ready"]) == Some(true);
    if !ready {
        blockers.push("NPU resident warm-route proof is not ready".to_string());
    }
    let warm_ask_count =
        u64_at(&json, "resident_session.warm_resident_asks.ask_count").unwrap_or(0);
    if warm_ask_count < 10 {
        blockers
            .push(format!("NPU resident warm-route proof has only {warm_ask_count} warm ask(s)"));
    }
    let warm_failed = u64_at(&json, "resident_session.warm_resident_asks.failed").unwrap_or(0);
    if warm_failed > 0 {
        blockers
            .push(format!("NPU resident warm-route proof has {warm_failed} failed warm ask(s)"));
    }
    if bool_at_any(&json, &["resident_session.warm_resident_asks.fallback_used"]) == Some(true) {
        blockers.push("NPU resident warm-route proof observed fallback_used=true".to_string());
    }
    if bool_at_any(&json, &["claim_boundary.route_promotion_changed"]) == Some(true)
        || bool_at_any(&json, &["claim_boundary.speedup_claim"]) == Some(true)
        || bool_at_any(&json, &["claim_boundary.power_advantage_claim"]) == Some(true)
        || bool_at_any(&json, &["claim_boundary.acceleration_claim"]) == Some(true)
        || bool_at_any(&json, &["claim_boundary.native_npu_inference_claim"]) == Some(true)
        || bool_at_any(&json, &["claim_boundary.bitnet_qk256_i2s_behavior_changed"]) == Some(true)
    {
        gaps.push(format!(
            "NPU resident-session receipt violates claim boundary: {}",
            path_string(&path)
        ));
    }

    blockers.sort();
    blockers.dedup();
    let resident_session_ready = ready
        && warm_ask_count >= 10
        && warm_failed == 0
        && bool_at_any(&json, &["fallback_used"]) == Some(false)
        && bool_at_any(&json, &["resident_session.warm_resident_asks.fallback_used"]) != Some(true);
    index.add("dense_slm_openvino_npu_candidate", source_receipt, blockers);
    let mut warm_resident_blockers = Vec::new();
    if warm_ask_count < 30 {
        warm_resident_blockers
            .push(format!("NPU resident stability proof has only {warm_ask_count}/30 warm ask(s)"));
    }
    if bool_at_any(&json, &["stability.answer_drift_detected"]) == Some(true) {
        warm_resident_blockers
            .push("NPU resident stability proof observed answer drift".to_string());
    }
    if bool_at_any(&json, &["stability.fallback_drift_detected"]) == Some(true) {
        warm_resident_blockers
            .push("NPU resident stability proof observed fallback drift".to_string());
    }
    if bool_at_any(&json, &["stability.route_drift_detected"]) == Some(true) {
        warm_resident_blockers
            .push("NPU resident stability proof observed route drift".to_string());
    }
    if value_at(&json, "stability.resident_memory_growth_bytes").is_none() {
        warm_resident_blockers
            .push("NPU resident stability proof lacks memory-growth context".to_string());
    }
    warm_resident_blockers.sort();
    warm_resident_blockers.dedup();
    index.add_for_profile(
        "dense_slm_openvino_npu_candidate",
        "warm_resident",
        path_string(&path),
        warm_resident_blockers,
    );
    Ok(resident_session_ready)
}

fn load_npu_cache_experiment(
    root: &Path,
    path: &Path,
    index: &mut RouteDiagnosticsIndex,
    gaps: &mut Vec<String>,
) -> Result<()> {
    let path = resolve_receipt_path(root, path);
    if !path.exists() {
        gaps.push(format!("NPU cache experiment receipt missing: {}", path_string(&path)));
        return Ok(());
    }
    let json: Value = read_json_receipt(&path)?;
    if string_at(&json, "artifact_kind").as_deref()
        != Some("lunar_lake_openvino_npu_cache_experiment")
    {
        gaps.push(format!(
            "NPU cache experiment receipt has unexpected artifact_kind: {}",
            path_string(&path)
        ));
    }

    let source_receipt = path_string(&path);
    let mut blockers = Vec::new();
    if string_at(&json, "selected_backend").as_deref() != Some("openvino-npu") {
        blockers.push("NPU cache experiment selected_backend is not openvino-npu".to_string());
    }
    if string_at(&json, "runtime_device").as_deref() != Some("NPU") {
        blockers.push("NPU cache experiment runtime_device is not NPU".to_string());
    }
    if bool_at_any(&json, &["fallback_used"]) != Some(false) {
        blockers.push("NPU cache experiment fallback_used=false is not proven".to_string());
    }
    if bool_at_any(&json, &["comparison.cache_experiment_ready"]) != Some(true) {
        blockers.push("NPU cache experiment is not ready".to_string());
    }
    if bool_at_any(&json, &["comparison.first_answer_gate_passed"]) != Some(true)
        || bool_at_any(&json, &["comparison.second_answer_gate_passed"]) != Some(true)
    {
        blockers.push("NPU cache experiment answer gates did not both pass".to_string());
    }
    let cache_effective_by_timing =
        bool_at_any(&json, &["cache.cache_effective_by_timing"]) == Some(true);
    let cache_files_created = bool_at_any(&json, &["cache.cache_files_created"]) == Some(true);
    let cache_files_reused_or_stable =
        bool_at_any(&json, &["cache.cache_files_reused_or_stable"]) == Some(true);
    let runtime_metric_available =
        bool_at_any(&json, &["cache.cache_hit_runtime_metric_available"]) == Some(true);
    if !(runtime_metric_available
        || cache_effective_by_timing && cache_files_created && cache_files_reused_or_stable)
    {
        blockers.push(
            "NPU cache hit evidence is inferred from cache files/timing, not an OpenVINO runtime metric"
                .to_string(),
        );
    }
    if !cache_effective_by_timing {
        let classification = string_at(&json, "comparison.classification")
            .unwrap_or_else(|| "cache_not_materially_proven_for_pipeline_construct".to_string());
        blockers.push(format!(
            "NPU cached cold process does not materially reduce pipeline construction: {classification}"
        ));
    }
    if bool_at_any(&json, &["generated_token_visibility.direct_generated_token_ids_available"])
        == Some(false)
    {
        blockers.push(
            "OpenVINO generated token IDs are retokenized, not direct pipeline internals"
                .to_string(),
        );
    }
    if bool_at_any(&json, &["claim_boundary.route_promotion_changed"]) == Some(true)
        || bool_at_any(&json, &["claim_boundary.speedup_claim"]) == Some(true)
        || bool_at_any(&json, &["claim_boundary.power_advantage_claim"]) == Some(true)
        || bool_at_any(&json, &["claim_boundary.acceleration_claim"]) == Some(true)
        || bool_at_any(&json, &["claim_boundary.native_npu_inference_claim"]) == Some(true)
        || bool_at_any(&json, &["claim_boundary.bitnet_qk256_i2s_behavior_changed"]) == Some(true)
    {
        gaps.push(format!(
            "NPU cache experiment receipt violates claim boundary: {}",
            path_string(&path)
        ));
    }

    blockers.sort();
    blockers.dedup();
    index.add("dense_slm_openvino_npu_candidate", source_receipt, blockers);
    Ok(())
}

fn load_npu_cold_start_diagnostic(
    root: &Path,
    path: &Path,
    npu_resident_session_ready: bool,
    index: &mut RouteDiagnosticsIndex,
    gaps: &mut Vec<String>,
) -> Result<()> {
    let path = resolve_receipt_path(root, path);
    if !path.exists() {
        gaps.push(format!("NPU cold-start diagnosis receipt missing: {}", path_string(&path)));
        return Ok(());
    }
    let json: Value = read_json_receipt(&path)?;
    if string_at(&json, "artifact_kind").as_deref()
        != Some("lunar_lake_openvino_npu_cold_start_diagnosis")
    {
        gaps.push(format!(
            "NPU cold-start diagnosis receipt has unexpected artifact_kind: {}",
            path_string(&path)
        ));
    }
    let mut blockers = Vec::new();
    if bool_at_any(&json, &["cold_start.cold_load_dominant"]) == Some(true) {
        let classification = string_at(&json, "cold_start.classification")
            .unwrap_or_else(|| "cold_load_dominated".to_string());
        blockers.push(format!("NPU cold start is {classification}"));
        if !npu_resident_session_ready {
            blockers.push("NPU cache or resident warm-route proof is missing".to_string());
        }
    }
    if bool_at_any(&json, &["corpus_v2_context.route_blocked_by_quality"]) == Some(true)
        && let Some(failed) = u64_at(&json, "corpus_v2_context.failed")
    {
        blockers.push(format!("NPU corpus-v2 context has {failed} failed cases"));
    }
    if bool_at_any(&json, &["claim_boundary.route_promotion_changed"]) == Some(true)
        || bool_at_any(&json, &["claim_boundary.speedup_claim"]) == Some(true)
        || bool_at_any(&json, &["claim_boundary.power_advantage_claim"]) == Some(true)
        || bool_at_any(&json, &["claim_boundary.acceleration_claim"]) == Some(true)
    {
        gaps.push(format!(
            "NPU cold-start diagnosis violates claim boundary: {}",
            path_string(&path)
        ));
    }
    blockers.sort();
    blockers.dedup();
    let source_receipt = path_string(&path);
    for profile_id in REQUIRED_ROUTE_PROFILES
        .iter()
        .copied()
        .filter(|profile_id| !matches!(*profile_id, "warm_resident" | "bitnet_strict_reference"))
    {
        index.add_for_profile(
            "dense_slm_openvino_npu_candidate",
            profile_id,
            source_receipt.clone(),
            blockers.clone(),
        );
    }
    Ok(())
}

fn load_openvino_generation_budget_sensitivity(
    root: &Path,
    path: &Path,
    index: &mut RouteDiagnosticsIndex,
    gaps: &mut Vec<String>,
) -> Result<()> {
    let path = resolve_receipt_path(root, path);
    if !path.exists() {
        gaps.push(format!(
            "OpenVINO generation-budget sensitivity receipt missing: {}",
            path_string(&path)
        ));
        return Ok(());
    }
    let json: Value = read_json_receipt(&path)?;
    if string_at(&json, "artifact_kind").as_deref()
        != Some("intel_258v_dense_slm_openvino_generation_budget_sensitivity")
    {
        gaps.push(format!(
            "OpenVINO generation-budget sensitivity receipt has unexpected artifact_kind: {}",
            path_string(&path)
        ));
    }
    if bool_at_any(&json, &["fallback_used"]) == Some(true) {
        gaps.push(format!(
            "OpenVINO generation-budget sensitivity receipt observed fallback_used=true: {}",
            path_string(&path)
        ));
    }
    if bool_at_any(
        &json,
        &[
            "route_promotion_changed",
            "claim_boundary.route_promotion_changed",
            "speedup_claim",
            "claim_boundary.speedup_claim",
            "power_advantage_claim",
            "claim_boundary.power_advantage_claim",
            "acceleration_claim",
            "claim_boundary.acceleration_claim",
        ],
    ) == Some(true)
    {
        gaps.push(format!(
            "OpenVINO generation-budget sensitivity receipt violates claim boundary: {}",
            path_string(&path)
        ));
    }

    let Some(devices) = value_at(&json, "devices").and_then(Value::as_array) else {
        gaps.push(format!(
            "OpenVINO generation-budget sensitivity receipt has no device entries: {}",
            path_string(&path)
        ));
        return Ok(());
    };
    let source_receipt = path_string(&path);
    for device in devices {
        if bool_at_any(device, &["fallback_used"]) == Some(true) {
            gaps.push(format!(
                "OpenVINO generation-budget sensitivity device observed fallback_used=true: {}",
                path_string(&path)
            ));
        }
        let Some(route_id) = openvino_device_route_id(device) else {
            continue;
        };
        let mut blockers = Vec::new();
        let mut case_blockers_indexed = false;
        if let Some(summary) = value_at(device, "summary") {
            let overgeneration_count = u64_at(
                summary,
                "blocker_classes.fixture_budget_overgenerates_but_smaller_budget_passes",
            )
            .unwrap_or(0);
            if overgeneration_count > 0 {
                blockers.push(format!(
                    "OpenVINO budget sensitivity reports {overgeneration_count} exact-answer case(s) where a smaller max_new_tokens budget passes but the fixture budget overgenerates"
                ));
            }
            let no_budget_count =
                u64_at(summary, "blocker_classes.no_budget_variant_passes").unwrap_or(0);
            if no_budget_count > 0 {
                blockers.push(format!(
                    "OpenVINO budget sensitivity reports {no_budget_count} exact-answer case(s) with no passing tested generation budget"
                ));
            }
        }
        if let Some(cases) = value_at(device, "cases").and_then(Value::as_array) {
            for case in cases {
                let case_id = string_at(case, "id").unwrap_or_else(|| "unknown_case".to_string());
                let profile_id = string_at(case, "profile");
                let blocker = match string_at(case, "blocker_class").as_deref() {
                    Some("fixture_budget_overgenerates_but_smaller_budget_passes") => {
                        if let Some(budget) = u64_at(case, "first_passing_budget") {
                            Some(format!(
                                "{case_id} overgenerates at the fixture budget but passes with max_new_tokens={budget}"
                            ))
                        } else {
                            Some(format!(
                                "{case_id} overgenerates at the fixture budget and needs a tighter generation budget rerun"
                            ))
                        }
                    }
                    Some("no_budget_variant_passes") => {
                        Some(format!("{case_id} has no passing tested generation budget"))
                    }
                    Some("fixture_budget_passes") => None,
                    Some(classification) => Some(format!(
                        "{case_id} has generation-budget sensitivity class {classification}"
                    )),
                    None => None,
                };
                if let Some(blocker) = blocker {
                    if profile_id.is_some() {
                        index.add_case_blocker(route_id, profile_id, &source_receipt, blocker);
                        case_blockers_indexed = true;
                    } else {
                        blockers.push(blocker);
                    }
                }
            }
        }
        if case_blockers_indexed {
            blockers.retain(|blocker| !blocker.starts_with("OpenVINO budget sensitivity reports "));
        }
        blockers.sort();
        blockers.dedup();
        if !blockers.is_empty() {
            index.add(route_id, source_receipt.clone(), blockers);
        }
    }
    Ok(())
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
    let answer = string_at(case, "answer")
        .or_else(|| string_at(case, "generated_text"))
        .or_else(|| string_at(case, "decoded_preview"))
        .or_else(|| string_at(case, "normalized_output"))
        .unwrap_or_default();
    let failed_rules = string_array_at(case, "quality.failed_rules");
    let scoring_passed = scoring.and_then(|value| value.get("passed")).and_then(Value::as_bool);
    let gate_passed = bool_at_any(case, &["quality.answer_gate.passed", "answer_gate.passed"]);
    let gate_kind = string_at(case, "quality.gate_kind");
    let generated_tokens = u64_at(case, "quality.generated_tokens")
        .or_else(|| u64_at(case, "tokens.generated"))
        .or_else(|| u64_at(case, "generated_token_count"));
    let expected_normalized = details.and_then(|value| string_at(value, "expected_normalized"));
    let observed_normalized = details.and_then(|value| string_at(value, "observed_normalized"));
    let classification = classify_corpus_v2_failure(
        &answer,
        gate_kind.as_deref(),
        gate_passed,
        scoring_passed,
        &failed_rules,
        &missing_required_keywords,
        generated_tokens,
        expected_normalized.as_deref(),
        observed_normalized.as_deref(),
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
        expected_normalized,
        observed_normalized,
        answer_preview: answer_preview(&answer),
        generated_tokens,
        prompt_tokens: u64_at(case, "tokens.prompt").or_else(|| u64_at(case, "prompt_token_count")),
        run_receipt_path: string_at(case, "run_receipt_path").map(|path| path.replace('\\', "/")),
        fallback_used: bool_at_any(case, &["fallback_used", "backend.fallback_used"]),
        classification,
        recommended_fix,
    }
}

#[allow(clippy::too_many_arguments)]
fn classify_corpus_v2_failure(
    answer: &str,
    gate_kind: Option<&str>,
    gate_passed: Option<bool>,
    scoring_passed: Option<bool>,
    failed_rules: &[String],
    missing_required_keywords: &[String],
    generated_tokens: Option<u64>,
    expected_normalized: Option<&str>,
    observed_normalized: Option<&str>,
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
    if failed_rules.iter().any(|rule| rule.contains("normalized_match")) {
        if normalized_answer_overgenerated(expected_normalized, observed_normalized) {
            return "exact_answer_overgenerated".to_string();
        }
        if matches!(gate_kind, Some("starts_with_any")) && gate_passed == Some(false) {
            return "exact_answer_instruction_not_followed".to_string();
        }
        if gate_passed == Some(true) && scoring_passed == Some(false) {
            return "exact_answer_scoring_mismatch_after_gate_pass".to_string();
        }
    }
    if !missing_required_keywords.is_empty()
        && generated_tokens.is_some_and(|tokens| tokens <= 8)
        && (trimmed.ends_with('+') || trimmed.split_whitespace().count() < 6)
    {
        return "generation_budget_or_truncation".to_string();
    }
    if !missing_required_keywords.is_empty() {
        if missing_keywords_present_case_insensitive(answer, missing_required_keywords) {
            return "case_sensitive_required_keyword_mismatch".to_string();
        }
        if any_missing_keyword_present_case_insensitive(answer, missing_required_keywords) {
            return "required_terms_missing_or_case_mismatch".to_string();
        }
        if gate_kind == Some("readable") {
            return "readable_output_missing_required_terms".to_string();
        }
        return "answer_content_missing_required_terms".to_string();
    }
    if failed_rules.iter().any(|rule| rule.starts_with("gate_") || rule == "answer_gate") {
        return "answer_gate_mismatch".to_string();
    }
    "answer_content_failed".to_string()
}

fn normalized_answer_overgenerated(
    expected_normalized: Option<&str>,
    observed_normalized: Option<&str>,
) -> bool {
    let Some(expected) = expected_normalized.map(str::trim).filter(|value| !value.is_empty())
    else {
        return false;
    };
    let Some(observed) = observed_normalized.map(str::trim).filter(|value| !value.is_empty())
    else {
        return false;
    };
    if observed == expected {
        return false;
    }
    let Some(remainder) = observed.strip_prefix(expected) else {
        return false;
    };
    remainder
        .chars()
        .next()
        .is_some_and(|ch| ch.is_whitespace() || matches!(ch, ',' | '.' | ':' | ';' | '!' | '?'))
}

fn missing_keywords_present_case_insensitive(
    answer: &str,
    missing_required_keywords: &[String],
) -> bool {
    !missing_required_keywords.is_empty()
        && missing_required_keywords
            .iter()
            .all(|keyword| contains_case_insensitive(answer, keyword))
}

fn any_missing_keyword_present_case_insensitive(
    answer: &str,
    missing_required_keywords: &[String],
) -> bool {
    missing_required_keywords.iter().any(|keyword| contains_case_insensitive(answer, keyword))
}

fn contains_case_insensitive(answer: &str, keyword: &str) -> bool {
    answer.to_lowercase().contains(&keyword.to_lowercase())
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
        "exact_answer_overgenerated" => {
            "The answer began with the expected exact token but continued; tighten stop/max-token policy or the prompt before treating this as a model-quality failure.".to_string()
        }
        "exact_answer_instruction_not_followed" => {
            "The route ignored a one-word exact-answer instruction; rerun with stricter generation/stop policy before profile promotion.".to_string()
        }
        "exact_answer_scoring_mismatch_after_gate_pass" => {
            "The loose answer gate passed but exact scoring failed; keep the profile blocked until the exact-answer contract is rerun cleanly or intentionally revised.".to_string()
        }
        "case_sensitive_required_keyword_mismatch" => {
            "The output contains the required term only with different casing; review whether the corpus gate should normalize case before rerunning.".to_string()
        }
        "required_terms_missing_or_case_mismatch" => {
            "The output contains some required terms only by case-insensitive match and still misses others; keep the profile blocked and tune prompt/gate wording before promotion.".to_string()
        }
        "readable_output_missing_required_terms" => {
            "Readable output was produced but missed required route-policy terms; tune the prompt or expected keywords before route promotion.".to_string()
        }
        "answer_content_missing_required_terms" => {
            "Treat this as a bounded answer-content failure for the dense Qwen route; adjust prompt/gate only if the expected answer contract is too narrow.".to_string()
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
        blockers.push("fallback_used is not false in the dense Qwen corpus-v2 receipt".to_string());
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
    if classes.contains("exact_answer_overgenerated")
        || classes.contains("exact_answer_instruction_not_followed")
    {
        actions.push(
            "Tighten exact-answer prompts, max-token budgets, or stop/EOS handling before promoting affected routes."
                .to_string(),
        );
    }
    if classes.contains("case_sensitive_required_keyword_mismatch")
        || classes.contains("required_terms_missing_or_case_mismatch")
    {
        actions.push(
            "Review corpus case-sensitivity versus model output casing before interpreting required-keyword misses as true answer failures."
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

fn openvino_profile_promotions_from_comparison(
    root: &Path,
    route_profile_comparison: &Path,
    expected_machine_id: &str,
    gaps: &mut Vec<String>,
) -> Result<(BTreeSet<String>, BTreeSet<String>, Option<String>)> {
    let comparison_path = resolve_receipt_path(root, route_profile_comparison);
    let comparison: Value = read_json_receipt(&comparison_path)?;
    let comparison_path_string = path_string(&comparison_path);
    let mut gpu_promoted_profiles = BTreeSet::new();
    let mut npu_promoted_profiles = BTreeSet::new();
    if string_at(&comparison, "artifact_kind").as_deref()
        != Some("lunar_lake_route_profile_comparison")
    {
        gaps.push(format!(
            "{} is not a Lunar Lake route-profile comparison receipt",
            comparison_path_string
        ));
        return Ok((gpu_promoted_profiles, npu_promoted_profiles, Some(comparison_path_string)));
    }
    if string_at(&comparison, "machine_id").as_deref() != Some(expected_machine_id) {
        gaps.push(format!(
            "{} machine_id does not match operator machine_id {}",
            comparison_path_string, expected_machine_id
        ));
        return Ok((gpu_promoted_profiles, npu_promoted_profiles, Some(comparison_path_string)));
    }
    if bool_at_any(&comparison, &["profile_comparison_ready"]) != Some(true) {
        gaps.push(format!("{} is not ready for profile promotion", comparison_path_string));
        return Ok((gpu_promoted_profiles, npu_promoted_profiles, Some(comparison_path_string)));
    }

    let profiles =
        comparison.get("profiles").and_then(Value::as_array).cloned().unwrap_or_default();
    collect_openvino_profile_promotions(
        &profiles,
        "dense_slm_openvino_gpu_candidate",
        OPENVINO_GPU_PROFILE_PROMOTION_TARGETS,
        &comparison_path_string,
        "OpenVINO GPU",
        &mut gpu_promoted_profiles,
        gaps,
    );
    collect_openvino_profile_promotions(
        &profiles,
        "dense_slm_openvino_npu_candidate",
        OPENVINO_NPU_PROFILE_PROMOTION_TARGETS,
        &comparison_path_string,
        "OpenVINO NPU",
        &mut npu_promoted_profiles,
        gaps,
    );

    Ok((gpu_promoted_profiles, npu_promoted_profiles, Some(comparison_path_string)))
}

fn collect_openvino_profile_promotions(
    profiles: &[Value],
    route_id: &str,
    profile_targets: &[&str],
    comparison_path_string: &str,
    route_label: &str,
    promoted_profiles: &mut BTreeSet<String>,
    gaps: &mut Vec<String>,
) {
    for profile_id in profile_targets {
        let profile = profiles
            .iter()
            .find(|profile| string_at(profile, "profile_id").as_deref() == Some(*profile_id));
        let Some(profile) = profile else {
            gaps.push(format!(
                "{} is missing route-profile evidence for {}",
                comparison_path_string, profile_id
            ));
            continue;
        };
        let route = profile.get("route_evidence").and_then(Value::as_array).and_then(|routes| {
            routes.iter().find(|route| string_at(route, "route_id").as_deref() == Some(route_id))
        });
        let Some(route) = route else {
            gaps.push(format!(
                "{} is missing {} route evidence for {}",
                comparison_path_string, route_label, profile_id
            ));
            continue;
        };
        if openvino_route_profile_is_benchmark_qualified(route, profile_id)
            || openvino_route_profile_is_already_promoted(profile, route, profile_id, route_id)
        {
            promoted_profiles.insert((*profile_id).to_string());
        } else {
            gaps.push(format!(
                "{} route evidence for {} is not benchmark-qualified in {}",
                route_label, profile_id, comparison_path_string
            ));
        }
    }
}

fn openvino_route_profile_is_benchmark_qualified(route: &Value, profile_id: &str) -> bool {
    if bool_at_any(route, &["benchmark_qualified_advantage"]) != Some(true)
        || bool_at_any(route, &["fallback_used"]) != Some(false)
        || bool_at_any(route, &["answer_gate_passed"]) != Some(true)
        || bool_at_any(route, &["phase_timing_present"]) != Some(true)
        || bool_at_any(route, &["timing_applicability.timing_matches_profile"]) != Some(true)
        || bool_at_any(route, &["profile_quality.profile_present"]) != Some(true)
        || bool_at_any(route, &["profile_quality.fallback_used"]) != Some(false)
        || u64_at(route, "profile_quality.failed") != Some(0)
        || bool_at_any(route, &["route_advantage_context.benchmark_qualified"]) != Some(true)
    {
        return false;
    }
    let blockers = string_array_at(route, "blockers");
    blockers.iter().all(|blocker| blocker_is_route_promotion_only(blocker, profile_id))
}

fn openvino_route_profile_is_already_promoted(
    profile: &Value,
    route: &Value,
    profile_id: &str,
    route_id: &str,
) -> bool {
    string_at(profile, "promoted_route").as_deref() == Some(route_id)
        && string_at(profile, "profile_status").as_deref() == Some("promoted_route_ready")
        && string_at(route, "route_status").as_deref() == Some("promoted")
        && bool_at_any(route, &["promotion_eligible_for_profile"]) == Some(true)
        && bool_at_any(route, &["fallback_used"]) == Some(false)
        && bool_at_any(route, &["answer_gate_passed"]) == Some(true)
        && bool_at_any(route, &["phase_timing_present"]) == Some(true)
        && bool_at_any(route, &["timing_applicability.timing_matches_profile"]) == Some(true)
        && bool_at_any(route, &["profile_quality.profile_present"]) == Some(true)
        && bool_at_any(route, &["profile_quality.fallback_used"]) == Some(false)
        && string_at(route, "profile_quality.profile_id").as_deref() == Some(profile_id)
        && u64_at(route, "profile_quality.failed") == Some(0)
        && string_array_at(route, "blockers").is_empty()
}

fn promote_route(
    route: &OperatorRoute,
    operator: &LunarLakeOperatorReceipt,
    comparison: &LunarLakeComparisonReceipt,
    openvino_gpu_promoted_profiles: &BTreeSet<String>,
    openvino_npu_promoted_profiles: &BTreeSet<String>,
    profile_promotion_evidence_path: Option<&str>,
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
                let mut promoted_for = vec![
                    "regression_tiny".to_string(),
                    "ask_short".to_string(),
                    "ask_normal".to_string(),
                    "structured".to_string(),
                ];
                promoted_for.retain(|profile| !openvino_gpu_promoted_profiles.contains(profile));
                promoted_for.retain(|profile| !openvino_npu_promoted_profiles.contains(profile));
                let mut blocked_for =
                    vec!["accelerator_required".to_string(), "bitnet_strict_reference".to_string()];
                blocked_for.extend(
                    openvino_gpu_promoted_profiles
                        .iter()
                        .map(|profile| format!("openvino_gpu_promoted_for_{profile}")),
                );
                blocked_for.extend(
                    openvino_npu_promoted_profiles
                        .iter()
                        .map(|profile| format!("openvino_npu_promoted_for_{profile}")),
                );
                (
                    "promoted".to_string(),
                    promoted_for,
                    blocked_for,
                    if openvino_gpu_promoted_profiles.is_empty()
                        && openvino_npu_promoted_profiles.is_empty()
                    {
                        "Dense Qwen CPU is promoted as the default route because answer gates, phase evidence, strict no-fallback identity, and comparison readiness are present.".to_string()
                    } else {
                        format!(
                            "Dense Qwen CPU remains the default route id and regression baseline, but OpenVINO routes supersede it for profile-qualified profiles [gpu:{}; npu:{}].",
                            join_set_or_none(openvino_gpu_promoted_profiles),
                            join_set_or_none(openvino_npu_promoted_profiles),
                        )
                    },
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
                if openvino_gpu_promoted_profiles.is_empty() {
                    missing_evidence
                        .push("benchmark_qualified_speedup_or_power_advantage".to_string());
                    missing_evidence.push("profile_regression_bundle".to_string());
                    (
                        "candidate".to_string(),
                        vec![],
                        vec!["auto_default".to_string(), "cold_start".to_string()],
                        "OpenVINO GPU has bounded answer and phase evidence with fallback=false, but remains a candidate until a workload-profile speedup or power advantage is recorded.".to_string(),
                    )
                } else {
                    if let Some(path) = profile_promotion_evidence_path
                        && !present_evidence.iter().any(|item| item == path)
                    {
                        present_evidence.push(path.to_string());
                    }
                    let mut blocked_for = vec![
                        "regression_tiny_cpu_baseline".to_string(),
                        "low_power_power_advantage_unproven".to_string(),
                        "structured_profile_unqualified".to_string(),
                        "bitnet_strict_reference".to_string(),
                    ];
                    if !openvino_gpu_promoted_profiles.contains("prefill_heavy") {
                        blocked_for.push("prefill_heavy_profile_unqualified".to_string());
                    }
                    if !openvino_gpu_promoted_profiles.contains("decode_heavy") {
                        blocked_for.push("decode_heavy_profile_unqualified".to_string());
                    }
                    (
                        "promoted".to_string(),
                        openvino_gpu_promoted_profiles.iter().cloned().collect(),
                        blocked_for,
                        format!(
                            "OpenVINO GPU is promoted only for benchmark-qualified dense Qwen profiles [{}] with fallback=false, passing corpus-v2 evidence, direct token visibility, profile-matched timing, and lower total response than the CPU baseline.",
                            openvino_gpu_promoted_profiles
                                .iter()
                                .cloned()
                                .collect::<Vec<_>>()
                                .join(",")
                        ),
                    )
                }
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
                if let Some(path) = profile_promotion_evidence_path
                    && !present_evidence.iter().any(|item| item == path)
                {
                    present_evidence.push(path.to_string());
                }
                if openvino_npu_promoted_profiles.is_empty() {
                    missing_evidence
                        .push("benchmark_qualified_speedup_or_power_advantage".to_string());
                    if profile_promotion_evidence_path.is_none() {
                        missing_evidence.push("profile_regression_bundle".to_string());
                    }
                    (
                        "candidate".to_string(),
                        vec![],
                        vec![
                            "auto_default".to_string(),
                            "cold_start_compile_load_blocker".to_string(),
                            "dynamic_decode".to_string(),
                            "beam_search".to_string(),
                            "parallel_sampling".to_string(),
                            "low_power_power_advantage_unproven".to_string(),
                            "warm_resident_profile_unqualified".to_string(),
                        ],
                        "OpenVINO NPU has bounded INT4 dense SLM answer and phase evidence with fallback=false, but remains a candidate until profile-specific advantage and constraints are recorded.".to_string(),
                    )
                } else {
                    (
                        "promoted".to_string(),
                        openvino_npu_promoted_profiles.iter().cloned().collect(),
                        vec![
                            "auto_default".to_string(),
                            "cold_start_compile_load_blocker".to_string(),
                            "dynamic_decode".to_string(),
                            "beam_search".to_string(),
                            "parallel_sampling".to_string(),
                            "low_power_power_advantage_unproven".to_string(),
                            "ask_short_cold_start_blocked".to_string(),
                            "ask_normal_cold_start_blocked".to_string(),
                            "prefill_heavy_profile_unqualified".to_string(),
                            "decode_heavy_profile_unqualified".to_string(),
                            "structured_profile_unqualified".to_string(),
                            "bitnet_strict_reference".to_string(),
                        ],
                        format!(
                            "OpenVINO NPU is promoted only for profile-qualified warm resident dense Qwen profiles [{}] with fallback=false, passing corpus-v2 evidence, resident-session timing, direct token visibility, and profile-matched latency evidence; cold and low_power profiles remain blocked.",
                            join_set_or_none(openvino_npu_promoted_profiles)
                        ),
                    )
                }
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
    corpus_alignment: &CorpusCaseAlignmentIndex,
    telemetry_context: Option<&BenchmarkTelemetryContext>,
    route_diagnostics: &RouteDiagnosticsIndex,
    route_identity_index: &RouteModelIdentityIndex,
    cpu_profile_run: Option<&Path>,
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

    let mut route_evidence = route_ids
        .iter()
        .map(|route_id| {
            evaluate_profile_route(
                root,
                profile,
                route_id,
                ledger,
                phase_comparison,
                quality_index,
                corpus_alignment,
                telemetry_context,
                route_diagnostics,
                route_identity_index,
                cpu_profile_run,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    attach_route_advantage_context(profile, &mut route_evidence);

    let mut gaps = Vec::new();
    if route_evidence.is_empty() {
        gaps.push("profile has no promoted or candidate route".to_string());
    }
    for route in &route_evidence {
        if route.fallback_used == Some(true) {
            gaps.push(format!("{} fallback_used=true", route.route_id));
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

#[allow(clippy::too_many_arguments)]
fn evaluate_profile_route(
    root: &Path,
    profile: &WorkloadProfile,
    route_id: &str,
    ledger: &LunarLakeRoutePromotionLedger,
    phase_comparison: &Value,
    quality_index: &ProfileQualityIndex,
    corpus_alignment: &CorpusCaseAlignmentIndex,
    telemetry_context: Option<&BenchmarkTelemetryContext>,
    route_diagnostics: &RouteDiagnosticsIndex,
    route_identity_index: &RouteModelIdentityIndex,
    cpu_profile_run: Option<&Path>,
) -> Result<ProfileRouteEvidence> {
    let route = ledger
        .routes
        .iter()
        .find(|route| route.route_id == route_id)
        .with_context(|| format!("route `{route_id}` not found in promotion ledger"))?;
    let timing = profile_timing_for_route(
        root,
        profile,
        route_id,
        phase_comparison,
        telemetry_context,
        quality_index,
        cpu_profile_run,
    )?;
    let timing_applicability = timing_applicability_for_profile(profile, &timing);
    let mut blockers = route.missing_evidence.clone();
    let profile_quality = quality_index.get(route_id, &profile.profile_id).cloned();
    let telemetry = telemetry_context.map(telemetry_for_profile_route);
    if let Some(alignment) = corpus_alignment.get(route_id) {
        blockers.extend(alignment.blockers.iter().cloned());
    }
    let route_diagnostic = route_diagnostics.get(route_id, &profile.profile_id);
    blockers.extend(route_diagnostic.blockers.iter().cloned());
    if route.status != "promoted" || !route.promoted_for.contains(&profile.profile_id) {
        blockers.push(format!("route not promoted for profile {}", profile.profile_id));
    }
    if route.status == "candidate" {
        blockers.push("candidate route requires benchmark-qualified profile evidence".to_string());
    }
    if lunar_lake_ask_runtime_requires_cpu(route_id) {
        blockers.push("lunar-lake ask runtime does not execute OpenVINO routes yet".to_string());
    }
    if timing.known_gaps.iter().any(|gap| timing_gap_is_missing_profile_field(gap)) {
        blockers.push("timing coverage has missing profile fields".to_string());
    }
    if !timing_applicability.timing_matches_profile {
        blockers.push(format!(
            "timing evidence is not profile-specific for profile {}",
            profile.profile_id
        ));
    }
    if profile.profile_id == "low_power" {
        if telemetry_context.is_some_and(power_context_is_recorded) {
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
    if profile_regression_bundle_evidence_satisfied(profile_quality.as_ref()) {
        blockers.retain(|blocker| blocker != "profile_regression_bundle");
    }
    blockers.sort();
    blockers.dedup();

    let promotion_eligible_for_profile = route.status == "promoted"
        && route.promoted_for.contains(&profile.profile_id)
        && route.fallback_used == Some(false)
        && blockers.is_empty();
    let route_status = profile_scoped_route_status(profile, route, promotion_eligible_for_profile);

    let mut evidence = route.present_evidence.clone();
    if let Some(quality) = &profile_quality
        && !evidence.contains(&quality.source_receipt)
    {
        evidence.push(quality.source_receipt.clone());
    }
    if let Some(alignment) = corpus_alignment.get(route_id) {
        for source in &alignment.source_receipts {
            if !evidence.contains(source) {
                evidence.push(source.clone());
            }
        }
    }
    for source in &route_diagnostic.source_receipts {
        if !evidence.contains(source) {
            evidence.push(source.clone());
        }
    }

    let model_identity = route_identity_index.identity_for(route);

    Ok(ProfileRouteEvidence {
        route_id: route.route_id.clone(),
        route_status,
        ledger_route_status: route.status.clone(),
        selected_model: model_identity.selected_model.clone(),
        selected_backend: route.selected_backend.clone(),
        runtime_api: route.runtime_api.clone(),
        model_identity: Some(model_identity),
        fallback_used: route.fallback_used,
        answer_gate_passed: route.answer_gate_passed,
        phase_timing_present: route.phase_timing_present,
        timing,
        timing_applicability,
        benchmark_qualified_advantage: false,
        promotion_eligible_for_profile,
        profile_quality,
        telemetry,
        route_advantage_context: None,
        evidence,
        blockers,
    })
}

fn profile_scoped_route_status(
    profile: &WorkloadProfile,
    route: &RoutePromotion,
    promotion_eligible_for_profile: bool,
) -> String {
    if promotion_eligible_for_profile {
        "promoted".to_string()
    } else if profile.promoted_route.as_deref() == Some(route.route_id.as_str())
        || (route.status == "promoted" && route.promoted_for.contains(&profile.profile_id))
    {
        "blocked".to_string()
    } else {
        "candidate".to_string()
    }
}

fn profile_regression_bundle_evidence_satisfied(
    profile_quality: Option<&ProfileQualityEvidence>,
) -> bool {
    profile_quality.is_some_and(|quality| {
        quality.profile_present && quality.fallback_used == Some(false) && quality.failed == 0
    })
}

fn lunar_lake_ask_runtime_requires_cpu(route_id: &str) -> bool {
    let _ = route_id;
    false
}

fn promotion_blocker_summary(
    profiles: &[WorkloadProfileEvaluation],
) -> Vec<PromotionBlockerSummary> {
    let mut grouped: BTreeMap<String, (BTreeSet<String>, BTreeSet<String>, u64)> = BTreeMap::new();
    for profile in profiles {
        for route in &profile.route_evidence {
            if route.route_status != "candidate" {
                continue;
            }
            for blocker in &route.blockers {
                let (route_ids, profile_ids, occurrence_count) =
                    grouped.entry(blocker.clone()).or_default();
                route_ids.insert(route.route_id.clone());
                profile_ids.insert(profile.profile_id.clone());
                *occurrence_count += 1;
            }
        }
    }

    grouped
        .into_iter()
        .map(|(blocker, (route_ids, profile_ids, occurrence_count))| PromotionBlockerSummary {
            next_action: promotion_blocker_next_action(&blocker),
            blocker,
            occurrence_count,
            route_ids: route_ids.into_iter().collect(),
            profile_ids: profile_ids.into_iter().collect(),
        })
        .collect()
}

fn promotion_blocker_next_action(blocker: &str) -> String {
    if blocker.contains("benchmark_qualified_speedup_or_power_advantage")
        || blocker.contains("benchmark-qualified")
    {
        "produce benchmark-qualified latency, throughput, power, or stability advantage evidence before promotion"
            .to_string()
    } else if blocker.contains("generated token IDs")
        || blocker.contains("retokenized")
        || blocker.contains("direct pipeline internals")
    {
        "add direct OpenVINO generated-token visibility, or keep the route blocked with re-tokenized output identity"
            .to_string()
    } else if blocker.contains("NPU cache or resident") || blocker.contains("resident") {
        "run NPU cache/resident warm-route proof and keep cold one-off routing blocked until classified"
            .to_string()
    } else if blocker.contains("lunar-lake ask runtime") {
        "add an OpenVINO execution path to bitnet lunar-lake ask before promoting this route"
            .to_string()
    } else if blocker.contains("NPU cold start") {
        "keep cold one-off NPU routing blocked; use resident/cache evidence only for warm-route evaluation"
            .to_string()
    } else if blocker.contains("power advantage") || blocker.contains("low_power") {
        "collect profile-specific low-power telemetry and compare against the promoted CPU baseline"
            .to_string()
    } else if blocker.contains("profile_regression_bundle")
        || blocker.contains("corpus_v2 profile quality evidence missing")
    {
        "run corpus-v2 profile evidence for this route/profile before evaluating promotion"
            .to_string()
    } else if blocker.contains("timing evidence is not profile-specific")
        || blocker.contains("timing coverage has missing profile fields")
    {
        "record profile-specific prefill/decode timing that satisfies the workload profile bounds"
            .to_string()
    } else if blocker.contains("route not promoted")
        || blocker.contains("candidate route requires benchmark-qualified profile evidence")
    {
        "keep route candidate-only until all profile-specific promotion evidence clears".to_string()
    } else {
        "classify and clear this blocker in the relevant route-quality or route-performance lane"
            .to_string()
    }
}

fn attach_route_advantage_context(
    profile: &WorkloadProfile,
    route_evidence: &mut [ProfileRouteEvidence],
) {
    let promoted_baseline_route_id = profile
        .promoted_route
        .as_deref()
        .filter(|route_id| route_evidence.iter().any(|route| route.route_id == **route_id))
        .or_else(|| {
            route_evidence
                .iter()
                .any(|route| route.route_id == DEFAULT_ASK_ROUTE)
                .then_some(DEFAULT_ASK_ROUTE)
        });
    let promoted_baseline = promoted_baseline_route_id
        .and_then(|route_id| route_evidence.iter().find(|route| route.route_id == route_id))
        .cloned();
    let cpu_baseline =
        route_evidence.iter().find(|route| route.route_id == DEFAULT_ASK_ROUTE).cloned();
    if promoted_baseline.is_none() && cpu_baseline.is_none() {
        return;
    }

    for route in route_evidence.iter_mut() {
        let baseline = if is_openvino_candidate_route(&route.route_id) {
            cpu_baseline.as_ref().or(promoted_baseline.as_ref())
        } else {
            promoted_baseline.as_ref().or(cpu_baseline.as_ref())
        };
        let Some(baseline) = baseline else {
            continue;
        };
        if route.route_id == baseline.route_id {
            continue;
        }
        if route_has_benchmark_qualified_latency_advantage(profile, &baseline, route) {
            route.benchmark_qualified_advantage = true;
            route.blockers.retain(|blocker| {
                blocker != "benchmark_qualified_speedup_or_power_advantage"
                    && blocker != "candidate route requires benchmark-qualified profile evidence"
            });
            route
                .timing
                .known_gaps
                .retain(|gap| gap != "benchmark-qualified speedup or power advantage missing");
            route.timing.phase_coverage.push("benchmark_qualified_latency_advantage".to_string());
        }
        route.route_advantage_context =
            Some(profile_route_advantage_context(profile, &baseline, route));
    }
}

fn route_is_benchmark_reference(profile: &WorkloadProfile, route: &ProfileRouteEvidence) -> bool {
    if route.promotion_eligible_for_profile {
        return true;
    }
    route.route_id == DEFAULT_ASK_ROUTE
        && route.fallback_used == Some(false)
        && route.answer_gate_passed == Some(true)
        && route.phase_timing_present == Some(true)
        && route.timing_applicability.timing_matches_profile
        && route.timing.total_response_ms.is_some()
        && route.profile_quality.as_ref().is_some_and(|quality| {
            quality.profile_present
                && quality.failed == 0
                && quality.fallback_used == Some(false)
                && quality.profile_id == profile.profile_id
        })
}

fn route_has_benchmark_qualified_latency_advantage(
    profile: &WorkloadProfile,
    baseline: &ProfileRouteEvidence,
    route: &ProfileRouteEvidence,
) -> bool {
    if !is_openvino_candidate_route(&route.route_id) || profile.profile_id == "low_power" {
        return false;
    }
    if !route_is_benchmark_reference(profile, baseline) {
        return false;
    }
    if route.fallback_used != Some(false)
        || route.answer_gate_passed != Some(true)
        || route.phase_timing_present != Some(true)
        || !route.timing_applicability.timing_matches_profile
    {
        return false;
    }
    let Some(quality) = route.profile_quality.as_ref() else {
        return false;
    };
    if !quality.profile_present || quality.failed > 0 || quality.fallback_used != Some(false) {
        return false;
    }
    let (Some(route_total), Some(baseline_total)) =
        (route.timing.total_response_ms, baseline.timing.total_response_ms)
    else {
        return false;
    };
    if baseline_total <= 0.0 || route_total / baseline_total > BENCHMARK_QUALIFIED_LATENCY_RATIO_MAX
    {
        return false;
    }
    route
        .blockers
        .iter()
        .all(|blocker| blocker_allows_latency_advantage_qualification(blocker, &profile.profile_id))
}

fn blocker_allows_latency_advantage_qualification(blocker: &str, profile_id: &str) -> bool {
    blocker == "benchmark_qualified_speedup_or_power_advantage"
        || blocker == "candidate route requires benchmark-qualified profile evidence"
        || blocker == "lunar-lake ask runtime does not execute OpenVINO routes yet"
        || blocker == format!("route not promoted for profile {profile_id}")
}

fn profile_route_advantage_context(
    profile: &WorkloadProfile,
    baseline: &ProfileRouteEvidence,
    route: &ProfileRouteEvidence,
) -> ProfileRouteAdvantageContext {
    let baseline_total_response_ms = baseline.timing.total_response_ms;
    let route_total_response_ms = route.timing.total_response_ms;
    let route_to_baseline_total_response_ratio =
        match (route_total_response_ms, baseline_total_response_ms) {
            (Some(route_total), Some(baseline_total)) if baseline_total > 0.0 => {
                Some(route_total / baseline_total)
            }
            _ => None,
        };
    let observed_total_response_lower_than_baseline =
        match (route_total_response_ms, baseline_total_response_ms) {
            (Some(route_total), Some(baseline_total)) => Some(route_total < baseline_total),
            _ => None,
        };

    let mut qualification_blockers = route
        .blockers
        .iter()
        .filter(|blocker| {
            if route.benchmark_qualified_advantage {
                !blocker_allows_latency_advantage_qualification(blocker, &profile.profile_id)
            } else {
                !blocker_is_route_promotion_only(blocker, &profile.profile_id)
            }
        })
        .cloned()
        .collect::<Vec<_>>();
    if baseline_total_response_ms.is_none() {
        qualification_blockers
            .push(format!("baseline route {} has no total response timing", baseline.route_id));
    }
    if !route_is_benchmark_reference(profile, baseline) {
        qualification_blockers.push(format!(
            "baseline route {} is not benchmark-reference-ready for profile {}",
            baseline.route_id, profile.profile_id
        ));
    }
    if route_total_response_ms.is_none() {
        qualification_blockers.push("route has no total response timing".to_string());
    }
    if !route.timing_applicability.timing_matches_profile {
        qualification_blockers.push(format!(
            "route timing is not profile-specific for profile {}",
            profile.profile_id
        ));
    }
    if !route.benchmark_qualified_advantage {
        qualification_blockers.push(
            "benchmark-qualified advantage is false; comparison is diagnostic only".to_string(),
        );
    }
    qualification_blockers.sort();
    qualification_blockers.dedup();

    let benchmark_qualified =
        route.benchmark_qualified_advantage && qualification_blockers.is_empty();
    let qualification_status = if benchmark_qualified {
        "benchmark_qualified".to_string()
    } else {
        "diagnostic_only_not_benchmark_qualified".to_string()
    };
    let mut notes = vec![
        "observed total response comparison is route-policy evidence, not a speedup claim"
            .to_string(),
        format!(
            "baseline route {} remains the comparison reference for profile {}",
            baseline.route_id, profile.profile_id
        ),
    ];
    if observed_total_response_lower_than_baseline == Some(true) {
        notes.push(
            "lower observed total response still requires route-promotion evidence before use"
                .to_string(),
        );
    }
    if route.route_status != "promoted" {
        notes.push(
            "route remains unpromoted; benchmark qualification does not change route selection"
                .to_string(),
        );
    }

    ProfileRouteAdvantageContext {
        baseline_route_id: baseline.route_id.clone(),
        baseline_route_status: baseline.route_status.clone(),
        baseline_total_response_ms,
        route_total_response_ms,
        route_to_baseline_total_response_ratio,
        observed_total_response_lower_than_baseline,
        benchmark_qualified,
        qualification_status,
        qualification_blockers,
        notes,
    }
}

fn blocker_is_route_promotion_only(blocker: &str, profile_id: &str) -> bool {
    blocker == format!("route not promoted for profile {profile_id}")
        || blocker == "candidate route requires benchmark-qualified profile evidence"
}

fn timing_applicability_for_profile(
    profile: &WorkloadProfile,
    timing: &ProfileTimingSummary,
) -> ProfileTimingApplicability {
    let mut notes = Vec::new();
    let prompt_match = token_count_matches_requirement(
        timing.prompt_tokens,
        &profile.prompt_tokens,
        "prompt",
        &mut notes,
    );
    let output_match = token_count_matches_requirement(
        timing.output_tokens,
        &profile.output_tokens,
        "output",
        &mut notes,
    );
    let statuses = [prompt_match, output_match];
    if statuses.iter().any(Option::is_none) {
        notes
            .push("profile token requirement is descriptive, not mechanically checked".to_string());
    }
    let timing_matches_profile = statuses.into_iter().all(|status| status.unwrap_or(true));

    ProfileTimingApplicability {
        profile_id: profile.profile_id.clone(),
        required_prompt_tokens: profile.prompt_tokens.clone(),
        required_output_tokens: profile.output_tokens.clone(),
        measured_prompt_tokens: timing.prompt_tokens,
        measured_output_tokens: timing.output_tokens,
        timing_matches_profile,
        notes,
    }
}

fn timing_gap_is_missing_profile_field(gap: &str) -> bool {
    gap.contains("operator ask prompt token count missing")
        || gap.contains("prompt token count missing")
        || gap.contains("output token count missing")
        || gap.contains("profile timing borrowed from corpus_v2 profile")
        || gap.starts_with("no timing extractor for route")
}

fn timing_applicability_coverage(
    profiles: &[WorkloadProfileEvaluation],
) -> TimingApplicabilityCoverageSummary {
    let mut summary = TimingApplicabilityCoverageSummary {
        promotion_eligible_routes_have_profile_specific_timing: true,
        proxy_or_missing_timing_routes_blocked: true,
        ..TimingApplicabilityCoverageSummary::default()
    };

    for profile in profiles {
        for route in &profile.route_evidence {
            summary.route_count += 1;
            let route_key = format!("{}:{}", profile.profile_id, route.route_id);
            let timing_matches_profile = route.timing_applicability.timing_matches_profile;
            if timing_matches_profile {
                summary.profile_specific_route_count += 1;
            } else {
                summary.proxy_or_missing_route_count += 1;
                summary.proxy_or_missing_routes.push(route_key.clone());
                let timing_blocker_present = route
                    .blockers
                    .iter()
                    .any(|blocker| blocker.contains("timing evidence is not profile-specific"));
                if !timing_blocker_present {
                    summary.proxy_or_missing_timing_routes_blocked = false;
                    summary.unblocked_proxy_or_missing_routes.push(route_key.clone());
                }
            }

            if route.promotion_eligible_for_profile {
                summary.promotion_eligible_route_count += 1;
                if timing_matches_profile {
                    summary.promotion_eligible_profile_specific_route_count += 1;
                } else {
                    summary.promotion_eligible_routes_have_profile_specific_timing = false;
                    summary.promotion_eligible_proxy_or_missing_routes.push(route_key.clone());
                }
            }

            if is_openvino_candidate_route(&route.route_id) {
                summary.candidate_route_count += 1;
                if !timing_matches_profile {
                    summary.candidate_proxy_or_missing_route_count += 1;
                }
            }
        }
    }

    summary.proxy_or_missing_routes.sort();
    summary.proxy_or_missing_routes.dedup();
    summary.promotion_eligible_proxy_or_missing_routes.sort();
    summary.promotion_eligible_proxy_or_missing_routes.dedup();
    summary.unblocked_proxy_or_missing_routes.sort();
    summary.unblocked_proxy_or_missing_routes.dedup();
    summary
}

fn token_count_matches_requirement(
    measured: Option<u64>,
    requirement: &str,
    label: &str,
    notes: &mut Vec<String>,
) -> Option<bool> {
    let requirement = requirement.trim();
    let Some((operator, threshold)) = parse_token_requirement(requirement) else {
        notes.push(format!("{label} requirement `{requirement}` is not a numeric promotion gate"));
        return None;
    };
    let Some(measured) = measured else {
        notes.push(format!("{label} token count missing for numeric requirement `{requirement}`"));
        return Some(false);
    };
    let matches = match operator {
        "<=" => measured <= threshold,
        ">=" => measured >= threshold,
        _ => false,
    };
    if matches {
        notes.push(format!("{label} timing count {measured} satisfies `{requirement}`"));
    } else {
        notes.push(format!("{label} timing count {measured} does not satisfy `{requirement}`"));
    }
    Some(matches)
}

fn parse_token_requirement(requirement: &str) -> Option<(&'static str, u64)> {
    let requirement = requirement.trim();
    for operator in ["<=", ">="] {
        let Some(rest) = requirement.strip_prefix(operator) else {
            continue;
        };
        let value = rest.trim().parse::<u64>().ok()?;
        return Some((operator, value));
    }
    None
}

fn profile_timing_for_route(
    root: &Path,
    profile: &WorkloadProfile,
    route_id: &str,
    phase_comparison: &Value,
    telemetry_context: Option<&BenchmarkTelemetryContext>,
    quality_index: &ProfileQualityIndex,
    cpu_profile_run: Option<&Path>,
) -> Result<ProfileTimingSummary> {
    match route_id {
        DEFAULT_ASK_ROUTE => dense_cpu_profile_timing(
            root,
            phase_comparison,
            telemetry_context,
            &profile.profile_id,
            cpu_profile_run,
        ),
        "dense_slm_openvino_gpu_candidate" => openvino_profile_timing(
            root,
            DENSE_OV_GPU_OPERATOR_ASK,
            "openvino_gpu_operator_ask",
            route_id,
            &profile.profile_id,
            quality_index,
        ),
        "dense_slm_openvino_npu_candidate" => openvino_profile_timing(
            root,
            DENSE_OV_NPU_OPERATOR_ASK,
            "openvino_npu_operator_ask",
            route_id,
            &profile.profile_id,
            quality_index,
        )
        .and_then(|timing| {
            if profile.profile_id == "warm_resident" {
                npu_resident_profile_timing(root).or_else(|_| Ok(timing))
            } else {
                Ok(timing)
            }
        }),
        "bitnet_reference_cpu" => Ok(ProfileTimingSummary {
            timing_scope: "bitnet_reference_cpu_not_dense_slm_profile".to_string(),
            source_receipts: vec![BITNET_PERF_APPLIED.to_string(), BITNET_CPU_BUNDLE.to_string()],
            prompt_tokens: None,
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
            prompt_tokens: None,
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
    profile_id: &str,
    cpu_profile_run: Option<&Path>,
) -> Result<ProfileTimingSummary> {
    let ask_path = root.join(DENSE_CPU_OPERATOR_ASK);
    let ask: Value = read_json_receipt(&ask_path)?;
    let profile_context = match cpu_profile_run {
        Some(path) => cpu_profile_run_timing_context(root, path, profile_id)?,
        None => None,
    };
    let operator_cold_load_ms = number_at_any(&ask, &["timing.model_load_ms"]);
    let tokenizer_load_ms = number_at_any(&ask, &["timing.tokenizer_load_ms"]);
    let operator_tokenize_ms = number_at_any(&ask, &["timing.tokenize_ms"]);
    let prefill_ms = number_at_any(&ask, &["timing.prefill_ms"]);
    let operator_output_tokens =
        number_at_any(&ask, &["tokens.generated_count", "timing.decode_tokens"])
            .map(|value| value as u64);
    let operator_prompt_tokens = number_at_any(
        &ask,
        &[
            "tokens.prompt_count",
            "tokens.prompt",
            "source_receipt.execution.prompt_tokens",
            "source_receipt.strict_provenance.prompt_tokens",
        ],
    )
    .map(|value| value as u64);
    let generation_total_ms = number_at_any(&ask, &["timing.decode_total_ms"]);
    let prompt_tokens = profile_context
        .as_ref()
        .and_then(|context| context.prompt_tokens)
        .or(operator_prompt_tokens);
    let output_tokens = profile_context
        .as_ref()
        .and_then(|context| context.output_tokens)
        .or(operator_output_tokens);
    let generation_total_ms = profile_context
        .as_ref()
        .and_then(|context| context.generation_total_ms)
        .or(generation_total_ms);
    let cold_load_ms =
        profile_context.as_ref().and_then(|context| context.load_time_ms).or(operator_cold_load_ms);
    let tokenize_ms =
        profile_context.as_ref().and_then(|context| context.tokenize_ms).or(operator_tokenize_ms);
    let throughput_tokens_per_s = profile_context
        .as_ref()
        .and_then(|context| context.throughput_tokens_per_s)
        .or_else(|| number_at_any(&ask, &["timing.decode_steady_state_tok_s"]))
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
    let total_response_ms = profile_context
        .as_ref()
        .and_then(|context| context.total_response_ms)
        .or(total_response_ms);

    let mut phase_coverage = vec![
        "operator_ask_math_brief".to_string(),
        "cpu_timing_model_load_tokenize_prefill_first_token_decode".to_string(),
    ];
    let mut source_receipts = vec![
        DENSE_CPU_OPERATOR_ASK.to_string(),
        DENSE_CPU_PHASE.to_string(),
        DENSE_PHASE_COMPARISON.to_string(),
    ];
    let prefill_512 = value_at(phase_comparison, "gguf_cpu_reference.timing.prefill_512").is_some();
    let decode_128 = value_at(phase_comparison, "gguf_cpu_reference.timing.decode_128").is_some();
    if prefill_512 {
        phase_coverage.push("warm_prefill_512".to_string());
    }
    if decode_128 {
        phase_coverage.push("warm_decode_128".to_string());
    }
    let mut known_gaps = Vec::new();
    if let Some(context) = profile_context.as_ref() {
        if !source_receipts.contains(&context.source_receipt) {
            source_receipts.push(context.source_receipt.clone());
        }
        phase_coverage.push(format!(
            "profile_timing_from_rust_gguf_cpu_profile_run_case_{}",
            context.case_id
        ));
        if context.profile_id != profile_id {
            known_gaps.push(format!(
                "profile timing borrowed from Rust GGUF CPU profile-run profile {} for profile {}",
                context.profile_id, profile_id
            ));
        }
    } else {
        known_gaps
            .push("bounded math ask only; not expanded profile regression corpus".to_string());
        if cpu_profile_run.is_some() {
            known_gaps.push(format!(
                "Rust GGUF CPU profile-run receipt did not contain fallback-free timing for profile {profile_id}"
            ));
        }
    }
    if let Some(context) = telemetry_context {
        phase_coverage.push("telemetry_context_indexed".to_string());
        if thermal_context_is_unavailable(&context.thermal_context) {
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
        source_receipts,
        prompt_tokens,
        cold_load_ms,
        tokenize_ms,
        prefill_ms,
        first_token_ms: profile_context
            .as_ref()
            .and_then(|context| context.first_token_ms)
            .or_else(|| number_at_any(&ask, &["timing.first_token_ms"])),
        decode_total_ms: generation_total_ms,
        generation_total_ms,
        total_response_ms,
        output_tokens,
        throughput_tokens_per_s,
        phase_coverage,
        known_gaps,
    })
}

fn npu_resident_profile_timing(root: &Path) -> Result<ProfileTimingSummary> {
    let path = root.join(OPENVINO_NPU_RESIDENT_SESSION);
    let json: Value = read_json_receipt(&path)?;
    let warm = value_at(&json, "resident_session.warm_resident_asks")
        .context("NPU resident-session receipt missing warm_resident_asks")?;
    let warm_asks = json
        .get("asks")
        .and_then(Value::as_array)
        .map(|asks| {
            asks.iter()
                .filter(|ask| string_at(ask, "phase").as_deref() == Some("warm_resident_ask"))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let prompt_tokens = warm_asks.iter().filter_map(|ask| u64_at(ask, "prompt_token_count")).max();
    let output_tokens =
        warm_asks.iter().filter_map(|ask| u64_at(ask, "generated_token_count")).max();
    let direct_generated_token_ids_available = warm_asks.iter().any(|ask| {
        bool_at_any(ask, &["generated_token_ids_available_from_pipeline"]) == Some(true)
    });
    let generation_total_ms = number_at_any(warm, &["generation_wall_ms.mean"]);
    let throughput_tokens_per_s = number_at_any(warm, &["throughput_tokens_per_s.mean"])
        .or_else(|| throughput_from_tokens(output_tokens, generation_total_ms));

    let mut phase_coverage = vec![
        "npu_resident_same_process_warm_asks".to_string(),
        "openvino_genai_perf_metrics".to_string(),
        "pipeline_construct_excluded_from_warm_resident_timing".to_string(),
    ];
    if direct_generated_token_ids_available {
        phase_coverage.push("direct_openvino_generated_token_ids".to_string());
    }

    Ok(ProfileTimingSummary {
        timing_scope: "openvino_npu_resident_warm_session".to_string(),
        source_receipts: vec![OPENVINO_NPU_RESIDENT_SESSION.to_string()],
        prompt_tokens,
        cold_load_ms: None,
        tokenize_ms: None,
        prefill_ms: None,
        first_token_ms: number_at_any(warm, &["openvino_time_to_first_token_ms.mean"])
            .or_else(|| number_at_any(warm, &["first_streamed_text_chunk_ms.mean"])),
        decode_total_ms: generation_total_ms,
        generation_total_ms,
        total_response_ms: generation_total_ms,
        output_tokens,
        throughput_tokens_per_s,
        phase_coverage,
        known_gaps: vec![
            "warm_resident timing excludes one-off NPU pipeline construction".to_string(),
            "resident evidence is not a power-advantage claim".to_string(),
        ],
    })
}

fn openvino_profile_timing(
    root: &Path,
    receipt_name: &str,
    timing_scope: &str,
    route_id: &str,
    profile_id: &str,
    quality_index: &ProfileQualityIndex,
) -> Result<ProfileTimingSummary> {
    let path = root.join(receipt_name);
    let ask: Value = read_json_receipt(&path)?;
    let operator_output_tokens =
        number_at_any(&ask, &["timing.openvino_perf_metrics.num_generated_tokens"])
            .map(|value| value as u64);
    let operator_generation_total_ms = number_at_any(
        &ask,
        &["timing.generation_wall_ms", "timing.openvino_perf_metrics.generate.mean_ms"],
    );
    let direct_prompt_tokens = number_at_any(
        &ask,
        &[
            "tokens.prompt_count",
            "tokens.prompt",
            "prompt.prompt_token_count",
            "prompt_token_count",
        ],
    )
    .map(|value| value as u64);
    let profile_run_context = openvino_profile_run_timing_context(root, route_id, profile_id)?;
    let corpus_context =
        openvino_corpus_operator_timing_context(route_id, profile_id, quality_index)?;
    let timing_context = profile_run_context.as_ref().or(corpus_context.as_ref());
    let prompt_tokens =
        timing_context.as_ref().and_then(|context| context.prompt_tokens).or(direct_prompt_tokens);
    let output_tokens = timing_context
        .as_ref()
        .and_then(|context| context.output_tokens)
        .or(operator_output_tokens);
    let generation_total_ms = timing_context
        .as_ref()
        .and_then(|context| context.generation_total_ms)
        .or(operator_generation_total_ms);
    let mut source_receipts = vec![receipt_name.to_string(), DENSE_OV_PHASE.to_string()];
    let mut phase_coverage = vec![
        "bounded_operator_ask_math_brief".to_string(),
        "openvino_genai_perf_metrics".to_string(),
        "pipeline_construct_and_generation_wall_time".to_string(),
    ];
    let direct_generated_token_ids_available =
        bool_at_any(&ask, &["output.generated_token_ids_available_from_pipeline"]) == Some(true)
            || corpus_context
                .as_ref()
                .is_some_and(|context| context.direct_generated_token_ids_available);
    let mut known_gaps = vec![
        "benchmark-qualified speedup or power advantage missing".to_string(),
        "OpenVINO receipts do not expose prefill_512/decode_128 splits for every profile"
            .to_string(),
    ];
    if direct_generated_token_ids_available {
        phase_coverage.push("direct_openvino_generated_token_ids".to_string());
    } else {
        known_gaps.push(
            "generated token IDs are not available directly from OpenVINO GenAI internals"
                .to_string(),
        );
    }
    if let Some(context) = profile_run_context.as_ref() {
        if !source_receipts.contains(&context.source_receipt) {
            source_receipts.push(context.source_receipt.clone());
        }
        phase_coverage
            .push(format!("profile_timing_from_openvino_profile_run_case_{}", context.case_id));
        if context.profile_id != profile_id {
            known_gaps.push(format!(
                "profile timing borrowed from OpenVINO profile-run profile {} for profile {}",
                context.profile_id, profile_id
            ));
        }
    }
    if let Some(context) = corpus_context.as_ref() {
        if !source_receipts.contains(&context.source_receipt) {
            source_receipts.push(context.source_receipt.clone());
        }
        if profile_run_context.is_none() {
            phase_coverage.push(format!(
                "profile_timing_supplemented_from_corpus_v2_case_{}",
                context.case_id
            ));
        }
        if profile_run_context.is_none() && context.profile_id != profile_id {
            known_gaps.push(format!(
                "profile timing borrowed from corpus_v2 profile {} for profile {}",
                context.profile_id, profile_id
            ));
        }
    } else if direct_prompt_tokens.is_none() {
        known_gaps.push("OpenVINO operator ask prompt token count missing".to_string());
    }
    if quality_index.has_route(route_id) {
        phase_coverage.push("corpus_v2_profile_regression_evidence_indexed".to_string());
    } else {
        known_gaps.push("profile regression bundle missing".to_string());
    }

    Ok(ProfileTimingSummary {
        timing_scope: timing_scope.to_string(),
        source_receipts,
        prompt_tokens,
        cold_load_ms: timing_context.as_ref().and_then(|context| context.load_time_ms).or_else(
            || {
                number_at_any(
                    &ask,
                    &[
                        "timing.openvino_perf_metrics.load_time_ms",
                        "timing.pipeline_construct_wall_ms",
                    ],
                )
            },
        ),
        tokenize_ms: timing_context.as_ref().and_then(|context| context.tokenize_ms).or_else(
            || number_at_any(&ask, &["timing.openvino_perf_metrics.tokenization.mean_ms"]),
        ),
        prefill_ms: None,
        first_token_ms: timing_context.as_ref().and_then(|context| context.first_token_ms).or_else(
            || {
                number_at_any(
                    &ask,
                    &[
                        "timing.openvino_perf_metrics.time_to_first_token.mean_ms",
                        "timing.first_streamed_text_chunk_ms",
                    ],
                )
            },
        ),
        decode_total_ms: generation_total_ms,
        generation_total_ms,
        total_response_ms: timing_context
            .as_ref()
            .and_then(|context| context.total_response_ms)
            .or_else(|| {
                sum_optional(
                    number_at_any(&ask, &["timing.pipeline_construct_wall_ms"]),
                    generation_total_ms,
                )
            }),
        output_tokens,
        throughput_tokens_per_s: timing_context
            .as_ref()
            .and_then(|context| context.throughput_tokens_per_s)
            .or_else(|| throughput_from_tokens(output_tokens, generation_total_ms)),
        phase_coverage,
        known_gaps,
    })
}

#[derive(Debug, Clone)]
struct OpenVinoCorpusOperatorTimingContext {
    source_receipt: String,
    case_id: String,
    profile_id: String,
    prompt_tokens: Option<u64>,
    output_tokens: Option<u64>,
    load_time_ms: Option<f64>,
    tokenize_ms: Option<f64>,
    first_token_ms: Option<f64>,
    generation_total_ms: Option<f64>,
    total_response_ms: Option<f64>,
    throughput_tokens_per_s: Option<f64>,
    direct_generated_token_ids_available: bool,
}

fn cpu_profile_run_timing_context(
    root: &Path,
    cpu_profile_run: &Path,
    profile_id: &str,
) -> Result<Option<OpenVinoCorpusOperatorTimingContext>> {
    let path = resolve_receipt_path(root, cpu_profile_run);
    if !path.exists() {
        return Ok(None);
    }
    let json: Value = read_json_receipt(&path)?;
    let artifact_kind = string_at(&json, "artifact_kind");
    if !matches!(
        artifact_kind.as_deref(),
        Some("intel_258v_dense_slm_cpu_profile_run" | "lunar_lake_cpu_profile_run")
    ) {
        return Ok(None);
    }

    let case = cpu_profile_run_case(&json, profile_id);
    let Some(case) = case else {
        return Ok(None);
    };
    if fallback_used(case).or_else(|| fallback_used(&json)) != Some(false) {
        return Ok(None);
    }

    let prompt_tokens = u64_at(case, "prompt_token_count")
        .or_else(|| u64_at(case, "prompt.prompt_token_count"))
        .or_else(|| u64_at(case, "tokens.prompt"))
        .or_else(|| u64_at(case, "tokens.prompt_count"));
    let output_tokens = u64_at(case, "generated_token_count")
        .or_else(|| u64_at(case, "tokens.generated"))
        .or_else(|| u64_at(case, "tokens.generated_count"))
        .or_else(|| u64_at(case, "tokens.output"))
        .or_else(|| u64_at(case, "timing.decode_tokens"));
    if prompt_tokens.is_none() && output_tokens.is_none() {
        return Ok(None);
    }

    let generation_total_ms = number_at_any(
        case,
        &["timing.generation_wall_ms", "timing.decode_total_ms", "timing.generation_total_ms"],
    );
    let load_time_ms =
        number_at_any(case, &["timing.model_load_ms", "timing.pipeline_construct_wall_ms"]);
    let tokenize_ms = number_at_any(case, &["timing.tokenize_ms"]);
    let first_token_ms =
        number_at_any(case, &["timing.first_token_ms", "timing.time_to_first_token_ms"]);
    let total_response_ms = number_at_any(
        case,
        &["timing.total_response_ms", "latency.total_ms", "total_response_ms", "total_ms"],
    )
    .or_else(|| {
        sum_all_optional([
            load_time_ms,
            number_at_any(case, &["timing.tokenizer_load_ms"]),
            tokenize_ms,
            number_at_any(case, &["timing.prefill_ms"]),
            generation_total_ms,
        ])
    });
    let throughput_tokens_per_s = number_at_any(
        case,
        &[
            "throughput.tokens_per_second",
            "timing.decode_steady_state_tok_s",
            "throughput_tokens_per_s",
        ],
    )
    .or_else(|| throughput_from_tokens(output_tokens, generation_total_ms));

    Ok(Some(OpenVinoCorpusOperatorTimingContext {
        source_receipt: path_string(&path),
        case_id: string_at(case, "id").unwrap_or_else(|| "unknown_case".to_string()),
        profile_id: string_at(case, "profile").unwrap_or_else(|| profile_id.to_string()),
        prompt_tokens,
        output_tokens,
        load_time_ms,
        tokenize_ms,
        first_token_ms,
        generation_total_ms,
        total_response_ms,
        throughput_tokens_per_s,
        direct_generated_token_ids_available: bool_at_any(
            case,
            &["tokens.direct_generated_token_ids_available", "generated_token_ids_available"],
        )
        .unwrap_or(false),
    }))
}

fn cpu_profile_run_case<'a>(json: &'a Value, profile_id: &str) -> Option<&'a Value> {
    fn matching_case<'a>(cases: &'a [Value], profile_id: &str) -> Option<&'a Value> {
        cases.iter().find(|case| {
            string_at(case, "profile").as_deref() == Some(profile_id)
                && string_at(case, "route_id")
                    .as_deref()
                    .map_or(true, |route_id| route_id == DEFAULT_ASK_ROUTE)
                && string_at(case, "selected_backend")
                    .as_deref()
                    .map_or(true, |backend| backend == "cpu-rust")
        })
    }

    if let Some(cases) = json.get("cases").and_then(Value::as_array)
        && let Some(case) = matching_case(cases, profile_id)
    {
        return Some(case);
    }
    if let Some(cases) = value_at(json, "generation.cases").and_then(Value::as_array)
        && let Some(case) = matching_case(cases, profile_id)
    {
        return Some(case);
    }
    let devices = value_at(json, "generation.devices").and_then(Value::as_array)?;
    let device = devices.iter().find(|device| {
        string_at(device, "route_id").as_deref() == Some(DEFAULT_ASK_ROUTE)
            || string_at(device, "selected_backend").as_deref() == Some("cpu-rust")
            || string_at(device, "runtime_api").as_deref() == Some("cpu")
    })?;
    let cases = device.get("cases").and_then(Value::as_array)?;
    matching_case(cases, profile_id)
}

fn openvino_profile_run_timing_context(
    root: &Path,
    route_id: &str,
    profile_id: &str,
) -> Result<Option<OpenVinoCorpusOperatorTimingContext>> {
    let path = root.join(OPENVINO_PROFILE_RUN);
    if !path.exists() {
        return Ok(None);
    }
    let json: Value = read_json_receipt(&path)?;
    if string_at(&json, "artifact_kind").as_deref()
        != Some("intel_258v_dense_slm_openvino_profile_run")
    {
        return Ok(None);
    }
    let Some(devices) = value_at(&json, "generation.devices").and_then(Value::as_array) else {
        return Ok(None);
    };
    let Some(device) =
        devices.iter().find(|device| openvino_device_route_id(device) == Some(route_id))
    else {
        return Ok(None);
    };
    let Some(cases) = device.get("cases").and_then(Value::as_array) else {
        return Ok(None);
    };
    let Some(case) =
        cases.iter().find(|case| string_at(case, "profile").as_deref() == Some(profile_id))
    else {
        return Ok(None);
    };
    let prompt_tokens = u64_at(case, "prompt_token_count")
        .or_else(|| u64_at(case, "prompt.prompt_token_count"))
        .or_else(|| u64_at(case, "tokens.prompt"));
    let output_tokens = u64_at(case, "generated_token_count")
        .or_else(|| u64_at(case, "tokens.generated"))
        .or_else(|| u64_at(case, "tokens.output"));
    if prompt_tokens.is_none() && output_tokens.is_none() {
        return Ok(None);
    }
    let generation_total_ms = number_at_any(
        case,
        &["timing.generation_wall_ms", "timing.openvino_perf_metrics.generate.mean_ms"],
    );
    let load_time_ms = number_at_any(
        case,
        &["timing.openvino_perf_metrics.load_time_ms", "timing.pipeline_construct_wall_ms"],
    );
    let total_response_ms = number_at_any(case, &["timing.total_response_ms"])
        .or_else(|| sum_optional(load_time_ms, generation_total_ms).or(generation_total_ms));
    Ok(Some(OpenVinoCorpusOperatorTimingContext {
        source_receipt: path_string(&path),
        case_id: string_at(case, "id").unwrap_or_else(|| "unknown_case".to_string()),
        profile_id: string_at(case, "profile").unwrap_or_else(|| "unknown_profile".to_string()),
        prompt_tokens,
        output_tokens,
        load_time_ms,
        tokenize_ms: number_at_any(case, &["timing.openvino_perf_metrics.tokenization.mean_ms"]),
        first_token_ms: number_at_any(
            case,
            &[
                "timing.openvino_perf_metrics.time_to_first_token.mean_ms",
                "timing.first_streamed_text_chunk_ms",
            ],
        ),
        generation_total_ms,
        total_response_ms,
        throughput_tokens_per_s: number_at_any(
            case,
            &["timing.openvino_perf_metrics.throughput.mean_ms"],
        )
        .filter(|value| *value > 0.0)
        .or_else(|| throughput_from_tokens(output_tokens, generation_total_ms)),
        direct_generated_token_ids_available: bool_at_any(
            case,
            &["generated_token_ids_available_from_pipeline"],
        ) == Some(true),
    }))
}

fn openvino_corpus_operator_timing_context(
    route_id: &str,
    profile_id: &str,
    quality_index: &ProfileQualityIndex,
) -> Result<Option<OpenVinoCorpusOperatorTimingContext>> {
    let Some(source) = &quality_index.openvino_source else {
        return Ok(None);
    };
    let path = PathBuf::from(source);
    let json: Value = read_json_receipt(&path)?;
    let Some(devices) = value_at(&json, "generation.devices").and_then(Value::as_array) else {
        return Ok(None);
    };
    let Some(device) =
        devices.iter().find(|device| openvino_device_route_id(device) == Some(route_id))
    else {
        return Ok(None);
    };
    let Some(cases) = device.get("cases").and_then(Value::as_array) else {
        return Ok(None);
    };
    let Some(case) = cases
        .iter()
        .find(|case| {
            string_at(case, "profile").as_deref() == Some(profile_id)
                && openvino_corpus_case_has_token_context(case)
        })
        .or_else(|| {
            cases.iter().find(|case| {
                string_at(case, "id").as_deref() == Some("math_2_plus_2_brief")
                    && openvino_corpus_case_has_token_context(case)
            })
        })
        .or_else(|| {
            cases.iter().find(|case| {
                string_at(case, "profile").as_deref() == Some("regression_tiny")
                    && openvino_corpus_case_has_token_context(case)
            })
        })
        .or_else(|| cases.iter().find(|case| openvino_corpus_case_has_token_context(case)))
        .or_else(|| {
            cases.iter().find(|case| string_at(case, "profile").as_deref() == Some(profile_id))
        })
        .or_else(|| {
            cases
                .iter()
                .find(|case| string_at(case, "id").as_deref() == Some("math_2_plus_2_brief"))
        })
        .or_else(|| {
            cases
                .iter()
                .find(|case| string_at(case, "profile").as_deref() == Some("regression_tiny"))
        })
        .or_else(|| cases.first())
    else {
        return Ok(None);
    };
    let prompt_tokens = u64_at(case, "prompt_token_count")
        .or_else(|| u64_at(case, "prompt.prompt_token_count"))
        .or_else(|| u64_at(case, "tokens.prompt"));
    let output_tokens = u64_at(case, "generated_token_count")
        .or_else(|| u64_at(case, "tokens.generated"))
        .or_else(|| u64_at(case, "tokens.output"));
    if prompt_tokens.is_none() && output_tokens.is_none() {
        return Ok(None);
    }
    let generation_total_ms = number_at_any(
        case,
        &["timing.generation_wall_ms", "timing.openvino_perf_metrics.generate.mean_ms"],
    );
    let throughput_tokens_per_s =
        number_at_any(case, &["timing.openvino_perf_metrics.throughput.mean_ms"])
            .filter(|value| *value > 0.0)
            .or_else(|| throughput_from_tokens(output_tokens, generation_total_ms));
    let total_response_ms = sum_optional(
        number_at_any(case, &["timing.openvino_perf_metrics.load_time_ms"]),
        generation_total_ms,
    )
    .or(generation_total_ms);
    Ok(Some(OpenVinoCorpusOperatorTimingContext {
        source_receipt: source.clone(),
        case_id: string_at(case, "id").unwrap_or_else(|| "unknown_case".to_string()),
        profile_id: string_at(case, "profile").unwrap_or_else(|| "unknown_profile".to_string()),
        prompt_tokens,
        output_tokens,
        load_time_ms: number_at_any(case, &["timing.openvino_perf_metrics.load_time_ms"]),
        tokenize_ms: number_at_any(case, &["timing.openvino_perf_metrics.tokenization.mean_ms"]),
        first_token_ms: number_at_any(
            case,
            &[
                "timing.openvino_perf_metrics.time_to_first_token.mean_ms",
                "timing.first_streamed_text_chunk_ms",
            ],
        ),
        generation_total_ms,
        total_response_ms,
        throughput_tokens_per_s,
        direct_generated_token_ids_available: bool_at_any(
            case,
            &["generated_token_ids_available_from_pipeline"],
        ) == Some(true),
    }))
}

fn openvino_corpus_case_has_token_context(case: &Value) -> bool {
    u64_at(case, "prompt_token_count")
        .or_else(|| u64_at(case, "prompt.prompt_token_count"))
        .or_else(|| u64_at(case, "tokens.prompt"))
        .is_some()
        || u64_at(case, "generated_token_count")
            .or_else(|| u64_at(case, "timing.openvino_perf_metrics.num_generated_tokens"))
            .or_else(|| u64_at(case, "tokens.generated"))
            .or_else(|| u64_at(case, "tokens.output"))
            .is_some()
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

fn workload_profiles_with_openvino_promotions(
    openvino_gpu_promoted_profiles: &BTreeSet<String>,
    openvino_npu_promoted_profiles: &BTreeSet<String>,
) -> Vec<WorkloadProfile> {
    let ask_short_gpu_promoted = openvino_gpu_promoted_profiles.contains("ask_short");
    let ask_normal_gpu_promoted = openvino_gpu_promoted_profiles.contains("ask_normal");
    let prefill_heavy_gpu_promoted = openvino_gpu_promoted_profiles.contains("prefill_heavy");
    let decode_heavy_gpu_promoted = openvino_gpu_promoted_profiles.contains("decode_heavy");
    let warm_resident_gpu_promoted = openvino_gpu_promoted_profiles.contains("warm_resident");
    let warm_resident_npu_promoted = openvino_npu_promoted_profiles.contains("warm_resident");
    vec![
        WorkloadProfile {
            profile_id: "regression_tiny".to_string(),
            prompt_tokens: "<=64".to_string(),
            output_tokens: "<=32".to_string(),
            purpose: "cheap strict regression smoke for local runs".to_string(),
            promoted_route: Some(DEFAULT_ASK_ROUTE.to_string()),
            candidate_routes: vec![
                "dense_slm_openvino_gpu_candidate".to_string(),
                "dense_slm_openvino_npu_candidate".to_string(),
            ],
        },
        WorkloadProfile {
            profile_id: "ask_short".to_string(),
            prompt_tokens: "<=64".to_string(),
            output_tokens: "<=32".to_string(),
            purpose: "one-off short prompt and short answer".to_string(),
            promoted_route: Some(if ask_short_gpu_promoted {
                "dense_slm_openvino_gpu_candidate".to_string()
            } else {
                DEFAULT_ASK_ROUTE.to_string()
            }),
            candidate_routes: if ask_short_gpu_promoted {
                vec![DEFAULT_ASK_ROUTE.to_string(), "dense_slm_openvino_npu_candidate".to_string()]
            } else {
                vec![
                    "dense_slm_openvino_gpu_candidate".to_string(),
                    "dense_slm_openvino_npu_candidate".to_string(),
                ]
            },
        },
        WorkloadProfile {
            profile_id: "ask_normal".to_string(),
            prompt_tokens: "<=512".to_string(),
            output_tokens: "<=128".to_string(),
            purpose: "default local assistant question profile".to_string(),
            promoted_route: Some(if ask_normal_gpu_promoted {
                "dense_slm_openvino_gpu_candidate".to_string()
            } else {
                DEFAULT_ASK_ROUTE.to_string()
            }),
            candidate_routes: if ask_normal_gpu_promoted {
                vec![DEFAULT_ASK_ROUTE.to_string(), "dense_slm_openvino_npu_candidate".to_string()]
            } else {
                vec![
                    "dense_slm_openvino_gpu_candidate".to_string(),
                    "dense_slm_openvino_npu_candidate".to_string(),
                ]
            },
        },
        WorkloadProfile {
            profile_id: "prefill_heavy".to_string(),
            prompt_tokens: ">=2048".to_string(),
            output_tokens: "<=64".to_string(),
            purpose: "long prompt with short answer where GPU/NPU prefill may earn promotion"
                .to_string(),
            promoted_route: prefill_heavy_gpu_promoted
                .then(|| "dense_slm_openvino_gpu_candidate".to_string()),
            candidate_routes: if prefill_heavy_gpu_promoted {
                vec![DEFAULT_ASK_ROUTE.to_string(), "dense_slm_openvino_npu_candidate".to_string()]
            } else {
                vec![
                    DEFAULT_ASK_ROUTE.to_string(),
                    "dense_slm_openvino_gpu_candidate".to_string(),
                    "dense_slm_openvino_npu_candidate".to_string(),
                ]
            },
        },
        WorkloadProfile {
            profile_id: "decode_heavy".to_string(),
            prompt_tokens: "<=256".to_string(),
            output_tokens: ">=512".to_string(),
            purpose: "long answer where steady decode throughput must be measured".to_string(),
            promoted_route: decode_heavy_gpu_promoted
                .then(|| "dense_slm_openvino_gpu_candidate".to_string()),
            candidate_routes: if decode_heavy_gpu_promoted {
                vec![DEFAULT_ASK_ROUTE.to_string(), "dense_slm_openvino_npu_candidate".to_string()]
            } else {
                vec![
                    DEFAULT_ASK_ROUTE.to_string(),
                    "dense_slm_openvino_gpu_candidate".to_string(),
                    "dense_slm_openvino_npu_candidate".to_string(),
                ]
            },
        },
        WorkloadProfile {
            profile_id: "structured".to_string(),
            prompt_tokens: "<=512".to_string(),
            output_tokens: "<=256".to_string(),
            purpose: "bounded JSON or tool-style output with deterministic answer gates"
                .to_string(),
            promoted_route: Some(DEFAULT_ASK_ROUTE.to_string()),
            candidate_routes: vec![],
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
            profile_id: "warm_resident".to_string(),
            prompt_tokens: "<=512".to_string(),
            output_tokens: "<=128".to_string(),
            purpose: "same-process warm or resident ask where GPU/NPU routes must prove stable reuse without cold-start promotion".to_string(),
            promoted_route: if warm_resident_npu_promoted {
                Some("dense_slm_openvino_npu_candidate".to_string())
            } else if warm_resident_gpu_promoted {
                Some("dense_slm_openvino_gpu_candidate".to_string())
            } else {
                None
            },
            candidate_routes: if warm_resident_npu_promoted {
                vec![DEFAULT_ASK_ROUTE.to_string(), "dense_slm_openvino_gpu_candidate".to_string()]
            } else if warm_resident_gpu_promoted {
                vec![DEFAULT_ASK_ROUTE.to_string(), "dense_slm_openvino_npu_candidate".to_string()]
            } else {
                vec![
                    DEFAULT_ASK_ROUTE.to_string(),
                    "dense_slm_openvino_npu_candidate".to_string(),
                    "dense_slm_openvino_gpu_candidate".to_string(),
                ]
            },
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

fn route_promotion_scope_from_profile_comparison(
    profiles: &[WorkloadProfileEvaluation],
) -> RoutePromotionScopeSummary {
    let mut summary = RoutePromotionScopeSummary {
        profile_scoped_promotion_only: true,
        openvino_npu_remains_candidate: true,
        ..RoutePromotionScopeSummary::default()
    };
    for profile in profiles {
        for route in &profile.route_evidence {
            record_openvino_profile_promotion_scope(
                &mut summary,
                &profile.profile_id,
                &route.route_id,
                &route.route_status,
                profile.promoted_route.as_deref(),
                route.promotion_eligible_for_profile,
            );
        }
    }
    finalize_openvino_profile_promotion_scope(summary)
}

fn route_promotion_scope_from_cold_warm(
    profiles: &[ColdWarmProfileBenchmark],
) -> RoutePromotionScopeSummary {
    let mut summary = RoutePromotionScopeSummary {
        profile_scoped_promotion_only: true,
        openvino_npu_remains_candidate: true,
        ..RoutePromotionScopeSummary::default()
    };
    for profile in profiles {
        for route in &profile.routes {
            record_openvino_profile_promotion_scope(
                &mut summary,
                &profile.profile_id,
                &route.route_id,
                &route.route_status,
                profile.promoted_route.as_deref(),
                !route.promotion_blocked,
            );
        }
    }
    finalize_openvino_profile_promotion_scope(summary)
}

fn record_openvino_profile_promotion_scope(
    summary: &mut RoutePromotionScopeSummary,
    profile_id: &str,
    route_id: &str,
    route_status: &str,
    promoted_route: Option<&str>,
    promotion_eligible_for_profile: bool,
) {
    if !is_openvino_candidate_route(route_id)
        || route_status != "promoted"
        || promoted_route != Some(route_id)
        || !promotion_eligible_for_profile
    {
        return;
    }

    match route_id {
        "dense_slm_openvino_gpu_candidate" => {
            summary.openvino_gpu_promoted_profiles.push(profile_id.to_string());
            if !OPENVINO_GPU_PROFILE_PROMOTION_TARGETS.contains(&profile_id) {
                summary
                    .unexpected_openvino_profile_promotions
                    .push(format!("{profile_id}:{route_id}"));
            }
        }
        "dense_slm_openvino_npu_candidate" => {
            summary.openvino_npu_promoted_profiles.push(profile_id.to_string());
            summary.openvino_npu_remains_candidate = false;
            if !OPENVINO_NPU_PROFILE_PROMOTION_TARGETS.contains(&profile_id) {
                summary
                    .unexpected_openvino_profile_promotions
                    .push(format!("{profile_id}:{route_id}"));
            }
        }
        _ => {}
    }
}

fn finalize_openvino_profile_promotion_scope(
    mut summary: RoutePromotionScopeSummary,
) -> RoutePromotionScopeSummary {
    summary.openvino_gpu_promoted_profiles.sort();
    summary.openvino_gpu_promoted_profiles.dedup();
    summary.openvino_npu_promoted_profiles.sort();
    summary.openvino_npu_promoted_profiles.dedup();
    summary.unexpected_openvino_profile_promotions.sort();
    summary.unexpected_openvino_profile_promotions.dedup();
    summary.profile_scoped_promotion_only =
        summary.unexpected_openvino_profile_promotions.is_empty();
    if summary.openvino_gpu_promoted_profiles.is_empty() {
        summary.notes.push("OpenVINO GPU has no profile promotions in this receipt".to_string());
    } else {
        summary.notes.push(format!(
            "OpenVINO GPU is profile-promoted only for {}",
            summary.openvino_gpu_promoted_profiles.join(",")
        ));
    }
    if summary.openvino_npu_promoted_profiles.is_empty() {
        summary.notes.push("OpenVINO NPU remains candidate-only".to_string());
    } else {
        summary.notes.push(format!(
            "OpenVINO NPU is profile-promoted only for {}",
            summary.openvino_npu_promoted_profiles.join(",")
        ));
    }
    summary.notes.push(
        "Profile promotion is not a broad acceleration, power-advantage, or all-profile claim"
            .to_string(),
    );
    summary
}

fn allowed_openvino_profile_promotion(
    profile_id: &str,
    route_id: &str,
    route_status: &str,
    promoted_route: Option<&str>,
) -> bool {
    route_status == "promoted"
        && promoted_route == Some(route_id)
        && match route_id {
            "dense_slm_openvino_gpu_candidate" => {
                OPENVINO_GPU_PROFILE_PROMOTION_TARGETS.contains(&profile_id)
            }
            "dense_slm_openvino_npu_candidate" => {
                OPENVINO_NPU_PROFILE_PROMOTION_TARGETS.contains(&profile_id)
            }
            _ => false,
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

fn non_empty_string_array_at_any(json: &Value, paths: &[&str]) -> Vec<String> {
    paths
        .iter()
        .map(|path| string_array_at(json, path))
        .find(|values| !values.is_empty())
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
    fn profile_scoped_route_status_keeps_global_promotion_separate() {
        let route = RoutePromotion {
            route_id: "dense_slm_openvino_gpu_candidate".to_string(),
            status: "promoted".to_string(),
            promoted_for: vec!["ask_short".to_string()],
            blocked_for: vec!["regression_tiny_cpu_baseline".to_string()],
            required_evidence: vec![],
            present_evidence: vec![],
            missing_evidence: vec![],
            selected_backend: "openvino-gpu".to_string(),
            runtime_api: "openvino_genai".to_string(),
            fallback_policy: "strict_no_fallback".to_string(),
            answer_gate_evidence: None,
            phase_evidence: None,
            fallback_used: Some(false),
            answer_gate_passed: Some(true),
            phase_timing_present: Some(true),
            speedup_claim: false,
            acceleration_claim: false,
            last_evidence_utc: "2026-05-19T04:30:00Z".to_string(),
            reason: "test route".to_string(),
        };
        let ask_short = WorkloadProfile {
            profile_id: "ask_short".to_string(),
            prompt_tokens: "<=64".to_string(),
            output_tokens: "<=32".to_string(),
            purpose: "short ask".to_string(),
            promoted_route: Some(route.route_id.clone()),
            candidate_routes: vec![DEFAULT_ASK_ROUTE.to_string()],
        };
        let regression_tiny = WorkloadProfile {
            profile_id: "regression_tiny".to_string(),
            prompt_tokens: "<=64".to_string(),
            output_tokens: "<=32".to_string(),
            purpose: "strict smoke".to_string(),
            promoted_route: Some(DEFAULT_ASK_ROUTE.to_string()),
            candidate_routes: vec![route.route_id.clone()],
        };

        assert_eq!(profile_scoped_route_status(&ask_short, &route, true), "promoted");
        assert_eq!(profile_scoped_route_status(&ask_short, &route, false), "blocked");
        assert_eq!(profile_scoped_route_status(&regression_tiny, &route, false), "candidate");
    }

    #[test]
    fn route_promotion_scope_records_profile_scoped_gpu_promotion() {
        let timing = ProfileTimingSummary {
            timing_scope: "profile_specific".to_string(),
            source_receipts: vec!["phase.json".to_string()],
            prompt_tokens: Some(32),
            cold_load_ms: Some(100.0),
            tokenize_ms: Some(1.0),
            prefill_ms: Some(10.0),
            first_token_ms: Some(20.0),
            decode_total_ms: Some(30.0),
            generation_total_ms: Some(50.0),
            total_response_ms: Some(151.0),
            output_tokens: Some(8),
            throughput_tokens_per_s: Some(10.0),
            phase_coverage: vec!["first_token".to_string(), "decode".to_string()],
            known_gaps: vec![],
        };
        let gpu_route = ProfileRouteEvidence {
            route_id: "dense_slm_openvino_gpu_candidate".to_string(),
            route_status: "promoted".to_string(),
            ledger_route_status: "promoted".to_string(),
            selected_model: "Qwen2.5-0.5B-Instruct OpenVINO IR INT4_SYM".to_string(),
            selected_backend: "openvino-gpu".to_string(),
            runtime_api: "openvino_genai".to_string(),
            model_identity: None,
            fallback_used: Some(false),
            answer_gate_passed: Some(true),
            phase_timing_present: Some(true),
            timing: timing.clone(),
            timing_applicability: ProfileTimingApplicability {
                profile_id: "ask_short".to_string(),
                required_prompt_tokens: "<=64".to_string(),
                required_output_tokens: "<=32".to_string(),
                measured_prompt_tokens: Some(32),
                measured_output_tokens: Some(8),
                timing_matches_profile: true,
                notes: vec![],
            },
            benchmark_qualified_advantage: true,
            promotion_eligible_for_profile: true,
            profile_quality: None,
            telemetry: None,
            route_advantage_context: None,
            evidence: vec!["route-profile.json".to_string()],
            blockers: vec![],
        };
        let npu_route = ProfileRouteEvidence {
            route_id: "dense_slm_openvino_npu_candidate".to_string(),
            route_status: "candidate".to_string(),
            ledger_route_status: "candidate".to_string(),
            selected_model: "Qwen2.5-0.5B-Instruct OpenVINO IR INT4_SYM".to_string(),
            selected_backend: "openvino-npu".to_string(),
            runtime_api: "openvino_genai".to_string(),
            model_identity: None,
            fallback_used: Some(false),
            answer_gate_passed: Some(true),
            phase_timing_present: Some(true),
            timing,
            timing_applicability: ProfileTimingApplicability::default(),
            benchmark_qualified_advantage: false,
            promotion_eligible_for_profile: false,
            profile_quality: None,
            telemetry: None,
            route_advantage_context: None,
            evidence: vec!["route-profile.json".to_string()],
            blockers: vec!["power advantage evidence missing for low_power promotion".to_string()],
        };
        let profiles = vec![WorkloadProfileEvaluation {
            profile_id: "ask_short".to_string(),
            prompt_tokens: "<=64".to_string(),
            output_tokens: "<=32".to_string(),
            purpose: "short ask".to_string(),
            promoted_route: Some("dense_slm_openvino_gpu_candidate".to_string()),
            candidate_routes: vec![DEFAULT_ASK_ROUTE.to_string()],
            profile_status: "promoted_route_ready".to_string(),
            route_evidence: vec![gpu_route, npu_route],
            promotion_decision: "gpu promoted".to_string(),
            gaps: vec![],
        }];

        let scope = route_promotion_scope_from_profile_comparison(&profiles);

        assert_eq!(scope.openvino_gpu_promoted_profiles, vec!["ask_short".to_string()]);
        assert!(scope.openvino_npu_promoted_profiles.is_empty());
        assert!(scope.openvino_npu_remains_candidate);
        assert!(scope.profile_scoped_promotion_only);
        assert!(scope.unexpected_openvino_profile_promotions.is_empty());
        assert!(scope.notes.iter().any(|note| note.contains("profile-promoted only")));
    }

    #[test]
    fn bitnet_semantic_intake_records_pending_stack_without_rerun() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_bitnet_semantic_intake_inputs(
            temp.path(),
            "stack_open",
            None,
            "2026-05-12T18:43:14Z",
            "2026-05-19T05:30:00Z",
        )?;

        let receipt = build_bitnet_semantic_intake_with_created_utc(
            temp.path(),
            Path::new(BITNET_SEMANTIC_SOURCE_CHANGES),
            Path::new(BITNET_CPU_BUNDLE),
            Path::new(OPERATOR_COMPARISON),
            "2026-05-19T05:45:00Z".to_string(),
        )?;

        assert!(receipt.intake_ready, "{:?}", receipt.gaps);
        assert!(!receipt.rerun_required);
        assert_eq!(receipt.source_change_summary.pending_shared_change_count, 1);
        assert_eq!(receipt.source_change_summary.merged_to_main_count, 0);
        assert!(receipt.required_reruns.is_empty());
        assert!(!receipt.claim_boundary.new_inference_executed);
        assert!(!receipt.claim_boundary.dense_slm_as_bitnet_proof);
        Ok(())
    }

    #[test]
    fn bitnet_semantic_intake_records_closed_unmerged_without_pending_or_rerun() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_bitnet_semantic_intake_inputs(
            temp.path(),
            "closed_unmerged",
            None,
            "2026-05-12T18:43:14Z",
            "2026-05-19T05:30:00Z",
        )?;

        let receipt = build_bitnet_semantic_intake_with_created_utc(
            temp.path(),
            Path::new(BITNET_SEMANTIC_SOURCE_CHANGES),
            Path::new(BITNET_CPU_BUNDLE),
            Path::new(OPERATOR_COMPARISON),
            "2026-05-19T06:58:20Z".to_string(),
        )?;

        assert!(receipt.intake_ready, "{:?}", receipt.gaps);
        assert!(!receipt.rerun_required);
        assert_eq!(receipt.source_change_summary.pending_shared_change_count, 0);
        assert_eq!(receipt.source_change_summary.closed_shared_change_count, 1);
        assert_eq!(receipt.source_change_summary.merged_to_main_count, 0);
        assert!(receipt.source_change_summary.pending_changes.is_empty());
        assert_eq!(receipt.source_change_summary.closed_changes.len(), 1);
        assert!(receipt.required_reruns.is_empty());
        assert!(receipt.changes[0].notes.iter().any(|note| note.contains("closed")));
        Ok(())
    }

    #[test]
    fn bitnet_semantic_intake_requires_rerun_for_newer_merged_shared_fix() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_bitnet_semantic_intake_inputs(
            temp.path(),
            "merged_to_main",
            Some("2026-05-19T06:00:00Z"),
            "2026-05-12T18:43:14Z",
            "2026-05-19T05:30:00Z",
        )?;

        let receipt = build_bitnet_semantic_intake_with_created_utc(
            temp.path(),
            Path::new(BITNET_SEMANTIC_SOURCE_CHANGES),
            Path::new(BITNET_CPU_BUNDLE),
            Path::new(OPERATOR_COMPARISON),
            "2026-05-19T06:05:00Z".to_string(),
        )?;

        assert!(!receipt.intake_ready);
        assert!(receipt.rerun_required);
        assert_eq!(receipt.source_change_summary.stale_after_merged_count, 1);
        assert!(receipt.required_reruns.iter().any(|rerun| rerun.contains("answer corpus")));
        assert!(receipt.gaps.iter().any(|gap| gap.contains("refreshed Lunar Lake BitNet")));
        Ok(())
    }

    #[test]
    fn regression_bundle_v2_fails_when_bitnet_semantic_intake_requires_rerun() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-19T05:30:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;
        write_stale_bitnet_semantic_intake(temp.path(), BITNET_SEMANTIC_INTAKE)?;

        let bundle = build_regression_bundle_with_created_utc_and_inputs(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            None,
            None,
            None,
            None,
            Some(Path::new(BITNET_SEMANTIC_INTAKE)),
            "2026-05-19T06:05:00Z".to_string(),
        )?;

        assert!(!bundle.regression_passed);
        assert!(!bundle.regression_surface.strict_ready);
        assert!(
            strict_regression_v2_gaps(&bundle)
                .iter()
                .any(|gap| gap.contains("BitNet semantic intake requires Lunar Lake reruns")),
            "{:?}",
            strict_regression_v2_gaps(&bundle)
        );
        let Some(intake) = bundle.bitnet_semantic_intake.as_ref() else {
            bail!("missing bitnet_semantic_intake summary");
        };
        assert!(intake.rerun_required);
        assert_eq!(intake.stale_after_merged_count, 1);
        Ok(())
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
        assert!(receipt.default_route.route_reason.contains("profile-scoped auto routing"));
        assert!(receipt.default_route.route_reason.contains("low_power remains blocked"));
        assert!(
            !receipt
                .default_route
                .route_reason
                .contains("accelerator paths are candidates until speedup")
        );
        let gpu_route = receipt
            .routes
            .iter()
            .find(|route| route.route_id == "dense_slm_openvino_gpu_candidate")
            .context("missing OpenVINO GPU route")?;
        assert!(gpu_route.route_reason.contains("profile-scoped"));
        assert!(gpu_route.route_reason.contains("no native OpenCL or acceleration claim"));
        let npu_route = receipt
            .routes
            .iter()
            .find(|route| route.route_id == "dense_slm_openvino_npu_candidate")
            .context("missing OpenVINO NPU route")?;
        assert!(npu_route.route_reason.contains("warm_resident"));
        assert!(
            npu_route.route_reason.contains("cold one-off and low_power profiles remain blocked")
        );
        assert!(npu_route.route_reason.contains("no dynamic decode"));
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
    fn operator_readiness_indexes_profile_route_policy() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        write_minimal_route_policy(temp.path())?;

        let receipt = build_operator_readiness_receipt_with_created_utc_and_route_policy(
            temp.path(),
            "2026-05-19T14:30:00Z".to_string(),
            Some(Path::new(ROUTE_PROMOTION_LEDGER)),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            None,
            None,
            None,
        )?;

        assert!(receipt.operator_ready, "{:?}", receipt.gaps);
        let policy = receipt.route_policy.as_ref().context("missing route policy")?;
        assert!(policy.policy_ready, "{:?}", policy.gaps);
        assert_eq!(policy.default_route_id, "dense_slm_default_cpu");
        assert_eq!(
            policy.openvino_gpu_promoted_profiles,
            vec!["ask_normal".to_string(), "ask_short".to_string()]
        );
        assert!(policy.openvino_npu_promoted_profiles.is_empty());
        assert!(policy.profile_scoped_promotion_only);
        assert!(!policy.hidden_fallback_allowed);
        assert!(policy.blocked_profiles.contains(&"low_power".to_string()));
        assert!(policy.profile_promotions.iter().any(|profile| profile.profile_id == "ask_normal"
            && profile.promoted_route.as_deref() == Some("dense_slm_openvino_gpu_candidate")));
        Ok(())
    }

    #[test]
    fn operator_readiness_indexes_power_and_thermal_context() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        write_minimal_route_policy(temp.path())?;
        write_json(
            temp.path(),
            POWER_PROFILE_EVIDENCE_FILE,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_power_profile_evidence",
                "proof_stage": "low_power_profile_evidence_indexed",
                "created_utc": "2026-05-20T01:05:00Z",
                "machine_id": "intel-258v",
                "artifact_root": path_string(temp.path()),
                "route_profile_comparison_receipt": ROUTE_PROFILE_COMPARISON,
                "cold_warm_benchmark_receipt": COLD_WARM_PROFILE_BENCHMARK_FILE,
                "telemetry_context_receipt": POWER_THERMAL_CONTEXT_FILE,
                "battery_telemetry_context_receipt": null,
                "energy_proxy_receipt": null,
                "telemetry": {
                    "memory_context_recorded": true,
                    "power_context_recorded": true,
                    "thermal_context_recorded": true,
                    "active_scheme": "Balanced",
                    "battery_status": "BatteryStatus=2;EstimatedChargeRemaining=100",
                    "ac_power_inferred": true,
                    "thermal_zones_visible": 1,
                    "thermal_temperature_count": 0,
                    "current_context_is_ac_only": true,
                    "battery_mode_sample_recorded": false,
                    "battery_sample_source": null,
                    "energy_proxy_recorded": false,
                    "energy_proxy_source": null
                },
                "low_power_routes": [
                    {
                        "route_id": "dense_slm_openvino_npu_candidate",
                        "route_status": "candidate",
                        "ledger_route_status": "candidate",
                        "selected_backend": "openvino-npu",
                        "runtime_api": "openvino_genai",
                        "fallback_used": false,
                        "answer_gate_passed": true,
                        "total_response_ms": null,
                        "throughput_tokens_per_s": null,
                        "benchmark_qualified_advantage": false,
                        "power_related_blockers": [
                            "battery-mode sample is missing for low_power promotion"
                        ],
                        "all_blockers": [
                            "battery-mode sample is missing for low_power promotion"
                        ],
                        "power_promotion_ready": false
                    }
                ],
                "power_profile_index_ready": true,
                "low_power_promotion_ready": false,
                "power_advantage_proven": false,
                "gaps": [
                    "battery-mode sample is missing for low_power promotion"
                ],
                "next_required_evidence": [
                    "battery-mode low_power telemetry"
                ],
                "claim_boundary": {
                    "new_inference_executed": false,
                    "route_promotion_changed": false,
                    "speedup_claim": false,
                    "power_advantage_claim": false,
                    "acceleration_claim": false,
                    "native_npu_inference_claim": false,
                    "bitnet_qk256_i2s_behavior_changed": false,
                    "hidden_fallback_allowed": false
                }
            }),
        )?;
        write_json(
            temp.path(),
            THERMAL_TEMPERATURE_AVAILABILITY_FILE,
            json!({
                "schema_version": 1,
                "artifact_kind": "lunar_lake_thermal_temperature_availability",
                "proof_stage": "thermal_temperature_sources_probed_no_claim_change",
                "machine_id": "intel-258v",
                "decision": {
                    "thermal_zone_visibility_available": true,
                    "thermal_temperature_available": false,
                    "usable_temperature_reading_count": 0
                },
                "claim_boundary": {
                    "new_inference_executed": false,
                    "telemetry_probe_executed": true,
                    "measured_temperature_claim": false,
                    "route_promotion_changed": false,
                    "speedup_claim": false,
                    "power_advantage_claim": false,
                    "acceleration_claim": false,
                    "native_opencl_or_native_npu_claim": false,
                    "bitnet_qk256_or_i2s_behavior_changed": false
                }
            }),
        )?;

        let receipt = build_operator_readiness_receipt_with_created_utc_and_route_policy(
            temp.path(),
            "2026-05-20T01:10:00Z".to_string(),
            Some(Path::new(ROUTE_PROMOTION_LEDGER)),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            Some(Path::new(POWER_PROFILE_EVIDENCE_FILE)),
            Some(Path::new(THERMAL_TEMPERATURE_AVAILABILITY_FILE)),
            None,
        )?;

        assert!(receipt.operator_ready, "{:?}", receipt.gaps);
        let power = receipt.power_profile_evidence.as_ref().context("missing power summary")?;
        assert!(power.power_profile_index_ready);
        assert!(!power.low_power_promotion_ready);
        assert!(!power.power_advantage_proven);
        assert!(power.current_context_is_ac_only);
        assert!(!power.battery_mode_sample_recorded);
        assert!(power.thermal_context_recorded);
        let thermal =
            receipt.thermal_temperature_availability.as_ref().context("missing thermal summary")?;
        assert!(thermal.thermal_zone_visibility_available);
        assert!(!thermal.thermal_temperature_available);
        assert_eq!(thermal.usable_temperature_reading_count, 0);
        assert!(!thermal.measured_temperature_claim);
        assert!(thermal.claim_boundary_preserved);
        Ok(())
    }

    #[test]
    fn operator_readiness_indexes_blocked_ask_guidance() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        write_minimal_route_policy(temp.path())?;
        write_json(
            temp.path(),
            BLOCKED_AUTO_ASK_RECEIPT,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_operator_ask_blocked",
                "proof_stage": "operator_route_selection_blocked_no_inference",
                "machine_id": "intel-258v",
                "requested_device": "auto",
                "requested_route": "auto",
                "profile_id": "low_power",
                "selected_route": null,
                "selected_backend": null,
                "runtime_api": null,
                "model_path_required": false,
                "model_loaded": false,
                "model_resolution": "not_required_for_blocked_auto_route_before_execution",
                "promotion_status": "no_promoted_route",
                "route_selection_status": "blocked",
                "route_selection_blocked": true,
                "route_selection_error": format!(
                    "no promoted Lunar Lake auto route for profile `low_power`; why_not_cpu=route is not promoted for profile `low_power`; why_not_gpu=route blocker for profile `low_power`: low_power_power_advantage_unproven; why_not_npu=missing evidence: benchmark_qualified_speedup_or_power_advantage; operator_runbook={LOW_POWER_BATTERY_RUNBOOK}"
                ),
                "candidate_routes": [
                    "dense_slm_default_cpu",
                    "dense_slm_openvino_gpu_candidate",
                    "dense_slm_openvino_npu_candidate"
                ],
                "why_not_cpu": ["route is not promoted for profile `low_power`"],
                "why_not_gpu": [
                    "route blocker for profile `low_power`: low_power_power_advantage_unproven"
                ],
                "why_not_npu": [
                    "missing evidence: benchmark_qualified_speedup_or_power_advantage"
                ],
                "operator_runbook": LOW_POWER_BATTERY_RUNBOOK,
                "next_required_evidence": blocked_operator_ask_next_required_evidence("low_power"),
                "fallback_used": false,
                "new_inference_executed": false,
                "speedup_claim": false,
                "acceleration_claim": false,
                "power_advantage_claim": false,
                "bitnet_qk256_i2s_claim": false,
                "claim_boundary": {
                    "route_selection_blocked": true,
                    "new_inference_executed": false,
                    "fallback_used": false,
                    "model_loaded": false,
                    "route_promotion_changed": false,
                    "speedup_claim": false,
                    "power_advantage_claim": false,
                    "acceleration_claim": false,
                    "native_accelerator_claim": false,
                    "bitnet_qk256_i2s_claim": false
                }
            }),
        )?;

        let receipt = build_operator_readiness_receipt_with_created_utc_and_route_policy(
            temp.path(),
            "2026-05-20T05:10:00Z".to_string(),
            Some(Path::new(ROUTE_PROMOTION_LEDGER)),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            None,
            None,
            Some(Path::new(BLOCKED_AUTO_ASK_RECEIPT)),
        )?;

        assert!(receipt.operator_ready, "{:?}", receipt.gaps);
        let blocked =
            receipt.blocked_ask_receipt.as_ref().context("missing blocked ask summary")?;
        assert!(blocked.regression_ready, "{:?}", blocked.gaps);
        assert_eq!(blocked.profile_id, "low_power");
        assert!(blocked.route_selection_blocked);
        assert!(!blocked.model_path_required);
        assert!(!blocked.model_loaded);
        assert_eq!(blocked.operator_runbook.as_deref(), Some(LOW_POWER_BATTERY_RUNBOOK));
        assert!(
            blocked
                .next_required_evidence
                .iter()
                .any(|item| item.contains("telemetry-context --require-battery"))
        );
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
        regression.regression_surface.bitnet_cpu_reference_evidence_indexed = true;
        regression.regression_surface.bitnet_cpu_reference_evidence_ready = true;
        regression.regression_surface.ask_short_ask_receipt_indexed = true;
        regression.regression_surface.ask_short_auto_ask_ready = true;
        regression.regression_surface.warm_resident_ask_receipt_indexed = true;
        regression.regression_surface.warm_resident_auto_ask_ready = true;
        regression.regression_surface.blocked_ask_receipt_indexed = true;
        regression.regression_surface.arc_npu_bounded_evidence_indexed = true;
        regression.regression_surface.arc_npu_bounded_evidence_ready = true;
        regression.regression_surface.candidate_routes_remain_unpromoted = true;
        regression.regression_surface.strict_ready = true;
        regression.regression_surface.gaps.clear();
        regression.ask_short_ask_receipt =
            Some(ready_gpu_operator_ask_summary("ask_short", AUTO_GPU_ASK_SHORT_ASK_RECEIPT));
        regression.warm_resident_ask_receipt = Some(ready_operator_ask_summary());
        regression.blocked_ask_receipt = Some(BlockedAskRegressionSummary {
            path: "lunar-lake-operator-ask-auto-low-power-blocked.json".to_string(),
            blocked_receipt_ready: true,
            profile_id: "low_power".to_string(),
            requested_device: "auto".to_string(),
            requested_route: "auto".to_string(),
            route_selection_blocked: true,
            model_path_required: false,
            model_loaded: false,
            model_resolution: "not_required_for_blocked_auto_route_before_execution".to_string(),
            candidate_routes: vec![
                DEFAULT_ASK_ROUTE.to_string(),
                "dense_slm_openvino_gpu_candidate".to_string(),
                "dense_slm_openvino_npu_candidate".to_string(),
            ],
            why_not_cpu: vec!["route is not promoted for profile `low_power`".to_string()],
            why_not_gpu: vec![
                "route blocker for profile `low_power`: low_power_power_advantage_unproven"
                    .to_string(),
            ],
            why_not_npu: vec![
                "missing evidence: benchmark_qualified_speedup_or_power_advantage".to_string(),
            ],
            operator_runbook: Some(LOW_POWER_BATTERY_RUNBOOK.to_string()),
            next_required_evidence: blocked_operator_ask_next_required_evidence("low_power"),
            new_inference_executed: false,
            fallback_used: false,
            route_promotion_changed: false,
            speedup_claim: false,
            power_advantage_claim: false,
            acceleration_claim: false,
            bitnet_qk256_i2s_claim: false,
            route_selection_error: format!(
                "no promoted Lunar Lake auto route for profile `low_power`; why_not_npu=missing evidence: benchmark_qualified_speedup_or_power_advantage; operator_runbook={LOW_POWER_BATTERY_RUNBOOK}"
            ),
            regression_ready: true,
            gaps: Vec::new(),
        });
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
        assert!(comparison.regression_surface.bitnet_cpu_reference_evidence_indexed);
        assert!(comparison.regression_surface.bitnet_cpu_reference_evidence_ready);
        assert!(comparison.regression_surface.ask_short_ask_receipt_indexed);
        assert!(comparison.regression_surface.ask_short_auto_ask_ready);
        assert!(comparison.regression_surface.warm_resident_ask_receipt_indexed);
        assert!(comparison.regression_surface.warm_resident_auto_ask_ready);
        assert!(comparison.regression_surface.blocked_ask_receipt_indexed);
        assert!(comparison.regression_surface.arc_npu_bounded_evidence_indexed);
        assert!(comparison.regression_surface.arc_npu_bounded_evidence_ready);
        let Some(ask_short) = comparison.ask_short_ask_receipt.as_ref() else {
            bail!("comparison did not carry ask_short ask receipt summary");
        };
        assert_eq!(ask_short.profile_id, "ask_short");
        assert_eq!(ask_short.selected_route, "dense_slm_openvino_gpu_candidate");
        let Some(warm_ask) = comparison.warm_resident_ask_receipt.as_ref() else {
            bail!("comparison did not carry warm resident ask receipt summary");
        };
        assert_eq!(warm_ask.profile_id, "warm_resident");
        assert_eq!(warm_ask.selected_route, "dense_slm_openvino_npu_candidate");
        assert!(warm_ask.new_inference_executed);
        assert!(warm_ask.generated_token_ids_available);
        let Some(blocked) = comparison.blocked_ask_receipt.as_ref() else {
            bail!("comparison did not carry blocked ask receipt summary");
        };
        assert_eq!(blocked.profile_id, "low_power");
        assert!(blocked.route_selection_blocked);
        assert!(!blocked.model_path_required);
        assert!(!blocked.model_loaded);
        assert_eq!(
            blocked.model_resolution,
            "not_required_for_blocked_auto_route_before_execution"
        );
        assert!(!blocked.new_inference_executed);
        assert!(blocked.route_selection_error.contains("why_not_npu="));
        Ok(())
    }

    #[test]
    fn comparison_receipt_carries_operator_route_policy() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        write_minimal_route_policy(temp.path())?;
        let operator = build_operator_readiness_receipt_with_created_utc_and_route_policy(
            temp.path(),
            "2026-05-19T14:30:00Z".to_string(),
            Some(Path::new(ROUTE_PROMOTION_LEDGER)),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            None,
            None,
            None,
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;
        let regression = build_regression_bundle_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "2026-05-19T14:35:00Z".to_string(),
        )?;
        fs::write(temp.path().join(REGRESSION_BUNDLE), serde_json::to_vec_pretty(&regression)?)?;

        let comparison = build_comparison_receipt_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(REGRESSION_BUNDLE),
            "2026-05-19T14:40:00Z".to_string(),
        )?;

        assert!(comparison.comparison_ready, "{:?}", comparison.gaps);
        let Some(policy) = comparison.route_policy.as_ref() else {
            bail!("comparison did not carry operator route policy");
        };
        assert!(policy.policy_ready, "{:?}", policy.gaps);
        assert_eq!(
            policy.openvino_gpu_promoted_profiles,
            vec!["ask_normal".to_string(), "ask_short".to_string()]
        );
        assert!(policy.openvino_npu_promoted_profiles.is_empty());
        assert!(policy.blocked_profiles.contains(&"low_power".to_string()));
        assert!(!policy.hidden_fallback_allowed);
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
    fn route_promotion_promotes_openvino_routes_for_benchmark_qualified_profiles() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-19T04:30:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;
        let regression = build_regression_bundle_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "2026-05-19T04:35:00Z".to_string(),
        )?;
        fs::write(temp.path().join(REGRESSION_BUNDLE), serde_json::to_vec_pretty(&regression)?)?;
        let comparison = build_comparison_receipt_with_created_utc(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(REGRESSION_BUNDLE),
            "2026-05-19T04:40:00Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_COMPARISON), serde_json::to_vec_pretty(&comparison)?)?;
        write_json(
            temp.path(),
            "gpu-route-profile-ready.json",
            json!({
                "artifact_kind": "lunar_lake_route_profile_comparison",
                "machine_id": "intel-258v",
                "profile_comparison_ready": true,
                "profiles": [
                    benchmark_qualified_openvino_profile("ask_short", "dense_slm_openvino_gpu_candidate"),
                    benchmark_qualified_openvino_profile("ask_normal", "dense_slm_openvino_gpu_candidate"),
                    benchmark_qualified_openvino_profile("prefill_heavy", "dense_slm_openvino_gpu_candidate"),
                    benchmark_qualified_openvino_profile("decode_heavy", "dense_slm_openvino_gpu_candidate"),
                    benchmark_qualified_openvino_profile("warm_resident", "dense_slm_openvino_npu_candidate")
                ]
            }),
        )?;

        let ledger = build_route_promotion_ledger_with_created_utc_and_profile_evidence(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(OPERATOR_COMPARISON),
            Some(Path::new("gpu-route-profile-ready.json")),
            "2026-05-19T04:45:00Z".to_string(),
        )?;

        assert!(ledger.promotion_ready, "{:?}", ledger.gaps);
        let cpu = ledger
            .routes
            .iter()
            .find(|route| route.route_id == DEFAULT_ASK_ROUTE)
            .context("missing CPU route")?;
        assert_eq!(cpu.status, "promoted");
        assert!(cpu.promoted_for.contains(&"regression_tiny".to_string()));
        assert!(cpu.promoted_for.contains(&"structured".to_string()));
        assert!(!cpu.promoted_for.contains(&"ask_short".to_string()));
        assert!(!cpu.promoted_for.contains(&"ask_normal".to_string()));
        let gpu = ledger
            .routes
            .iter()
            .find(|route| route.route_id == "dense_slm_openvino_gpu_candidate")
            .context("missing GPU route")?;
        assert_eq!(gpu.status, "promoted");
        assert_eq!(
            gpu.promoted_for,
            vec![
                "ask_normal".to_string(),
                "ask_short".to_string(),
                "decode_heavy".to_string(),
                "prefill_heavy".to_string()
            ]
        );
        assert!(gpu.missing_evidence.is_empty(), "{:?}", gpu.missing_evidence);
        assert!(
            gpu.present_evidence.iter().any(|item| item.ends_with("gpu-route-profile-ready.json"))
        );
        let npu = ledger
            .routes
            .iter()
            .find(|route| route.route_id == "dense_slm_openvino_npu_candidate")
            .context("missing NPU route")?;
        assert_eq!(npu.status, "promoted");
        assert_eq!(npu.promoted_for, vec!["warm_resident".to_string()]);
        assert!(
            npu.present_evidence.iter().any(|item| item.ends_with("gpu-route-profile-ready.json"))
        );
        assert!(npu.missing_evidence.is_empty(), "{:?}", npu.missing_evidence);
        assert!(
            npu.blocked_for.contains(&"low_power_power_advantage_unproven".to_string()),
            "{:?}",
            npu.blocked_for
        );
        let why_not_npu =
            route_not_selected_reasons(&ledger, "dense_slm_openvino_npu_candidate", "low_power");
        assert!(
            why_not_npu.iter().any(|reason| {
                reason.contains("low_power_power_advantage_unproven")
                    || reason.contains("benchmark_qualified_speedup_or_power_advantage")
            }),
            "{why_not_npu:?}"
        );
        let ask_short = ledger
            .workload_profiles
            .iter()
            .find(|profile| profile.profile_id == "ask_short")
            .context("missing ask_short profile")?;
        assert_eq!(ask_short.promoted_route.as_deref(), Some("dense_slm_openvino_gpu_candidate"));
        assert!(ask_short.candidate_routes.contains(&DEFAULT_ASK_ROUTE.to_string()));
        let warm_resident = ledger
            .workload_profiles
            .iter()
            .find(|profile| profile.profile_id == "warm_resident")
            .context("missing warm_resident profile")?;
        assert_eq!(
            warm_resident.promoted_route.as_deref(),
            Some("dense_slm_openvino_npu_candidate")
        );
        assert!(
            warm_resident
                .candidate_routes
                .contains(&"dense_slm_openvino_gpu_candidate".to_string())
        );
        assert!(
            ledger
                .auto_route_policy
                .notes
                .iter()
                .any(|note| note.contains("OpenVINO GPU is promoted"))
        );
        assert!(
            ledger
                .auto_route_policy
                .notes
                .iter()
                .any(|note| note.contains("OpenVINO NPU is promoted"))
        );
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
                    "prompt_count": 38,
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
        write_route_model_identity_manifests(temp.path())?;

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
        assert!(profiles.route_model_identity_coverage.all_route_rows_have_identity);
        assert!(profiles.route_model_identity_coverage.route_rows_with_model_hash > 0);
        assert!(
            profiles
                .route_model_identity_coverage
                .routes_without_model_hash
                .iter()
                .any(|route| route.contains("dense_slm_openvino_gpu_candidate"))
        );
        assert!(profiles.timing_coverage.route_count > 0);
        assert!(profiles.timing_coverage.promotion_eligible_routes_have_profile_specific_timing);
        assert!(profiles.timing_coverage.proxy_or_missing_timing_routes_blocked);
        assert!(profiles.timing_coverage.candidate_proxy_or_missing_route_count > 0);
        let Some(ask_normal) =
            profiles.profiles.iter().find(|profile| profile.profile_id == "ask_normal")
        else {
            bail!("missing ask_normal profile");
        };
        assert!(ask_normal.route_evidence.iter().any(|route| {
            route.route_id == DEFAULT_ASK_ROUTE && route.promotion_eligible_for_profile
        }));
        let cpu_ask_normal = ask_normal
            .route_evidence
            .iter()
            .find(|route| route.route_id == DEFAULT_ASK_ROUTE)
            .context("missing ask_normal CPU route")?;
        assert_eq!(cpu_ask_normal.selected_model, "Qwen2.5-0.5B-Instruct Q8_0 GGUF");
        let cpu_identity =
            cpu_ask_normal.model_identity.as_ref().context("missing CPU model identity")?;
        assert_eq!(cpu_identity.identity_source, "dense_slm_gguf_manifest");
        assert_eq!(
            cpu_identity.model_sha256.as_deref(),
            Some("ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e")
        );
        assert_eq!(cpu_identity.prompt_template.as_deref(), Some("qwen2.5-instruct-chatml"));
        assert_eq!(cpu_ask_normal.timing_applicability.measured_prompt_tokens, Some(38));
        assert!(cpu_ask_normal.timing_applicability.timing_matches_profile);
        assert!(ask_normal.route_evidence.iter().any(|route| {
            route.route_id == "dense_slm_openvino_gpu_candidate"
                && route.route_status == "candidate"
                && !route.benchmark_qualified_advantage
        }));
        let gpu_ask_normal = ask_normal
            .route_evidence
            .iter()
            .find(|route| route.route_id == "dense_slm_openvino_gpu_candidate")
            .context("missing ask_normal GPU route")?;
        let gpu_identity =
            gpu_ask_normal.model_identity.as_ref().context("missing GPU model identity")?;
        assert_eq!(gpu_identity.identity_source, "dense_slm_openvino_ir_manifest");
        assert_eq!(gpu_identity.model_sha256, None);
        assert!(
            gpu_identity
                .known_gaps
                .iter()
                .any(|gap| gap.contains("OpenVINO IR model binaries are not committed"))
        );
        assert_eq!(gpu_ask_normal.timing_applicability.measured_output_tokens, None);
        assert!(!gpu_ask_normal.timing_applicability.timing_matches_profile);
        assert!(gpu_ask_normal.blockers.contains(
            &"timing evidence is not profile-specific for profile ask_normal".to_string()
        ));
        assert!(
            !gpu_ask_normal.blockers.contains(
                &"lunar-lake ask runtime does not execute OpenVINO routes yet".to_string()
            )
        );
        let structured = profiles
            .profiles
            .iter()
            .find(|profile| profile.profile_id == "structured")
            .context("missing structured profile")?;
        assert_eq!(structured.promoted_route.as_deref(), Some(DEFAULT_ASK_ROUTE));
        assert_eq!(structured.profile_status, "promoted_route_ready");
        let cpu_structured = structured
            .route_evidence
            .iter()
            .find(|route| route.route_id == DEFAULT_ASK_ROUTE)
            .context("missing structured CPU route")?;
        assert!(cpu_structured.promotion_eligible_for_profile);
        assert!(cpu_structured.timing_applicability.timing_matches_profile);
        assert!(cpu_structured.blockers.is_empty(), "{:?}", cpu_structured.blockers);
        let prefill_heavy = profiles
            .profiles
            .iter()
            .find(|profile| profile.profile_id == "prefill_heavy")
            .context("missing prefill_heavy profile")?;
        let cpu_prefill_heavy = prefill_heavy
            .route_evidence
            .iter()
            .find(|route| route.route_id == DEFAULT_ASK_ROUTE)
            .context("missing prefill_heavy CPU route")?;
        assert!(!cpu_prefill_heavy.timing_applicability.timing_matches_profile);
        assert!(
            cpu_prefill_heavy
                .timing_applicability
                .notes
                .iter()
                .any(|note| { note.contains("prompt timing count 38 does not satisfy `>=2048`") })
        );
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
        let bitnet_profile = profiles
            .profiles
            .iter()
            .find(|profile| profile.profile_id == "bitnet_strict_reference")
            .context("missing bitnet_strict_reference profile")?;
        let bitnet_route = bitnet_profile
            .route_evidence
            .iter()
            .find(|route| route.route_id == "bitnet_reference_cpu")
            .context("missing BitNet CPU route")?;
        assert_eq!(
            bitnet_route
                .model_identity
                .as_ref()
                .and_then(|identity| identity.model_sha256.as_deref()),
            Some("4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162")
        );

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

        fs::write(
            temp.path().join("route-profile-ready.json"),
            serde_json::to_vec_pretty(&profiles_with_telemetry)?,
        )?;
        let route_profile_summary =
            inspect_route_profile_regression(&temp.path().join("route-profile-ready.json"))?;
        assert!(route_profile_summary.route_model_identity_ready);
        assert!(route_profile_summary.regression_ready, "{:?}", route_profile_summary.gaps);
        let mut missing_route_identity = profiles_with_telemetry.clone();
        missing_route_identity.profiles[0].route_evidence[0].model_identity = None;
        fs::write(
            temp.path().join("route-profile-missing-identity.json"),
            serde_json::to_vec_pretty(&missing_route_identity)?,
        )?;
        let missing_route_identity_summary = inspect_route_profile_regression(
            &temp.path().join("route-profile-missing-identity.json"),
        )?;
        assert!(!missing_route_identity_summary.route_model_identity_ready);
        assert!(missing_route_identity_summary.gaps.iter().any(|gap| {
            gap.contains("route profile comparison has route rows without model identity")
        }));

        let mut unrelated_hash_gap = profiles_with_telemetry.clone();
        let route_without_hash = unrelated_hash_gap
            .profiles
            .iter_mut()
            .flat_map(|profile| profile.route_evidence.iter_mut())
            .find(|route| {
                route
                    .model_identity
                    .as_ref()
                    .map(|identity| identity.model_sha256.is_none())
                    .unwrap_or(false)
            })
            .context("missing route evidence without a model hash")?;
        route_without_hash
            .model_identity
            .as_mut()
            .context("missing route model identity")?
            .known_gaps = vec!["OpenVINO timing is not GGUF CPU phase timing".to_string()];
        fs::write(
            temp.path().join("route-profile-unrelated-hash-gap.json"),
            serde_json::to_vec_pretty(&unrelated_hash_gap)?,
        )?;
        let unrelated_hash_gap_summary = inspect_route_profile_regression(
            &temp.path().join("route-profile-unrelated-hash-gap.json"),
        )?;
        assert!(!unrelated_hash_gap_summary.route_model_identity_ready);
        assert!(unrelated_hash_gap_summary.gaps.iter().any(|gap| {
            gap.contains("route profile comparison has route rows without model hash or explicit no-hash gap")
        }));

        let cold_warm = build_cold_warm_benchmark_with_created_utc(
            temp.path(),
            Path::new("route-profile-ready.json"),
            Path::new(DENSE_PHASE_COMPARISON),
            Some(Path::new(POWER_THERMAL_CONTEXT_FILE)),
            "2026-05-17T07:05:00Z".to_string(),
        )?;
        assert!(cold_warm.route_model_identity_coverage.all_route_rows_have_identity);
        fs::write(
            temp.path().join("cold-warm-ready.json"),
            serde_json::to_vec_pretty(&cold_warm)?,
        )?;
        let cold_warm_summary =
            inspect_cold_warm_regression(&temp.path().join("cold-warm-ready.json"))?;
        assert!(cold_warm_summary.route_model_identity_ready);
        assert!(cold_warm_summary.regression_ready, "{:?}", cold_warm_summary.gaps);
        let mut missing_benchmark_identity = cold_warm.clone();
        missing_benchmark_identity.profiles[0].routes[0].model_identity = None;
        fs::write(
            temp.path().join("cold-warm-missing-identity.json"),
            serde_json::to_vec_pretty(&missing_benchmark_identity)?,
        )?;
        let missing_benchmark_identity_summary =
            inspect_cold_warm_regression(&temp.path().join("cold-warm-missing-identity.json"))?;
        assert!(!missing_benchmark_identity_summary.route_model_identity_ready);
        assert!(missing_benchmark_identity_summary.gaps.iter().any(|gap| {
            gap.contains("cold/warm benchmark has route rows without model identity")
        }));
        Ok(())
    }

    #[test]
    fn dense_cpu_profile_timing_uses_matching_cpu_profile_run() -> Result<()> {
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
                    "prompt_count": 38,
                    "generated_count": 8
                }
            }),
        )?;
        write_json(
            temp.path(),
            "cpu-profile-run.json",
            json!({
                "artifact_kind": "intel_258v_dense_slm_cpu_profile_run",
                "fallback_used": false,
                "cases": [
                    {
                        "id": "prefill_heavy_cpu_baseline",
                        "profile": "prefill_heavy",
                        "route_id": DEFAULT_ASK_ROUTE,
                        "selected_backend": "cpu-rust",
                        "fallback_used": false,
                        "prompt_token_count": 2300,
                        "generated_token_count": 64,
                        "timing": {
                            "model_load_ms": 1000.0,
                            "tokenize_ms": 20.0,
                            "generation_wall_ms": 44000.0,
                            "first_token_ms": 1200.0,
                            "total_response_ms": 45020.0
                        },
                        "quality": {
                            "passed": true
                        }
                    }
                ]
            }),
        )?;

        let timing = dense_cpu_profile_timing(
            temp.path(),
            &json!({}),
            None,
            "prefill_heavy",
            Some(Path::new("cpu-profile-run.json")),
        )?;

        assert_eq!(timing.prompt_tokens, Some(2300));
        assert_eq!(timing.output_tokens, Some(64));
        assert_eq!(timing.total_response_ms, Some(45020.0));
        assert!(timing.source_receipts.iter().any(|path| path.ends_with("cpu-profile-run.json")));
        assert!(timing.phase_coverage.iter().any(|item| {
            item == "profile_timing_from_rust_gguf_cpu_profile_run_case_prefill_heavy_cpu_baseline"
        }));
        let profile = WorkloadProfile {
            profile_id: "prefill_heavy".to_string(),
            prompt_tokens: ">=2048".to_string(),
            output_tokens: "<=64".to_string(),
            purpose: "test profile".to_string(),
            promoted_route: None,
            candidate_routes: vec![],
        };
        assert!(timing_applicability_for_profile(&profile, &timing).timing_matches_profile);
        Ok(())
    }

    #[test]
    fn route_profile_comparison_indexes_corpus_v2_profile_quality_blockers() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        write_route_model_identity_manifests(temp.path())?;
        write_answer_corpus_v2(temp.path(), "corpus-v2.yaml")?;
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
                "tokens": {"prompt_count": 38, "generated_count": 8}
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
        write_json(
            temp.path(),
            OPENVINO_GPU_CORPUS_V2_DIAGNOSIS,
            json!({
                "artifact_kind": "lunar_lake_openvino_corpus_v2_diagnosis",
                "route_blocked": true,
                "quality_summary": {"failed": 5},
                "profile_diagnoses": [
                    {
                        "profile_id": "ask_short",
                        "failed": 1,
                        "blocked": true,
                        "route_blockers": ["GPU ask_short diagnosis blocker"]
                    },
                    {
                        "profile_id": "regression_tiny",
                        "failed": 1,
                        "blocked": true,
                        "route_blockers": ["GPU regression_tiny diagnosis blocker"]
                    }
                ],
                "failed_cases": [
                    {
                        "id": "yes_no_clear_sky",
                        "profile": "ask_short",
                        "classification": "exact_answer_overgenerated"
                    },
                    {
                        "id": "stop_token_one_word_done",
                        "profile": "regression_tiny",
                        "classification": "exact_answer_instruction_not_followed"
                    }
                ],
                "generated_token_visibility": {
                    "direct_generated_token_ids_available": false
                },
                "blocker_summary": ["GPU diagnosis blocker"],
                "claim_boundary": {
                    "route_promotion_changed": false,
                    "speedup_claim": false,
                    "arc_or_npu_execution_claim": false
                }
            }),
        )?;
        write_json(
            temp.path(),
            "lunar-lake-openvino-npu-corpus-v2-diagnosis.json",
            json!({
                "artifact_kind": "lunar_lake_openvino_corpus_v2_diagnosis",
                "route_blocked": true,
                "quality_summary": {"failed": 4},
                "profile_diagnoses": [
                    {
                        "profile_id": "ask_short",
                        "failed": 1,
                        "blocked": true,
                        "route_blockers": ["NPU ask_short diagnosis blocker"]
                    },
                    {
                        "profile_id": "regression_tiny",
                        "failed": 1,
                        "blocked": true,
                        "route_blockers": ["NPU regression_tiny diagnosis blocker"]
                    }
                ],
                "failed_cases": [
                    {
                        "id": "yes_no_clear_sky",
                        "profile": "ask_short",
                        "classification": "exact_answer_overgenerated"
                    },
                    {
                        "id": "stop_token_one_word_done",
                        "profile": "regression_tiny",
                        "classification": "exact_answer_instruction_not_followed"
                    }
                ],
                "generated_token_visibility": {
                    "direct_generated_token_ids_available": false
                },
                "blocker_summary": ["NPU diagnosis blocker"],
                "claim_boundary": {
                    "route_promotion_changed": false,
                    "speedup_claim": false,
                    "arc_or_npu_execution_claim": false
                }
            }),
        )?;
        write_json(
            temp.path(),
            OPENVINO_NPU_COLD_START_DIAGNOSIS,
            json!({
                "artifact_kind": "lunar_lake_openvino_npu_cold_start_diagnosis",
                "cold_start": {
                    "cold_load_dominant": true,
                    "classification": "openvino_pipeline_load_or_device_compile_dominated"
                },
                "corpus_v2_context": {
                    "route_blocked_by_quality": true,
                    "failed": 4
                },
                "claim_boundary": {
                    "route_promotion_changed": false,
                    "speedup_claim": false,
                    "power_advantage_claim": false,
                    "acceleration_claim": false
                }
            }),
        )?;
        write_json(
            temp.path(),
            OPENVINO_GENERATION_BUDGET_SENSITIVITY,
            json!({
                "artifact_kind": "intel_258v_dense_slm_openvino_generation_budget_sensitivity",
                "fallback_used": false,
                "route_promotion_changed": false,
                "devices": [
                    {
                        "runtime_device": "GPU.0",
                        "fallback_used": false,
                        "summary": {
                            "cases_total": 2,
                            "fixture_budget_passed": 0,
                            "any_budget_passed": 1,
                            "blocker_classes": {
                                "fixture_budget_overgenerates_but_smaller_budget_passes": 1,
                                "no_budget_variant_passes": 1
                            }
                        },
                        "cases": [
                            {
                                "id": "yes_no_clear_sky",
                                "profile": "ask_short",
                                "fixture_budget_passed": false,
                                "any_budget_passed": true,
                                "first_passing_budget": 1,
                                "blocker_class": "fixture_budget_overgenerates_but_smaller_budget_passes"
                            },
                            {
                                "id": "stop_token_one_word_done",
                                "profile": "regression_tiny",
                                "fixture_budget_passed": false,
                                "any_budget_passed": false,
                                "first_passing_budget": null,
                                "blocker_class": "no_budget_variant_passes"
                            },
                            {
                                "id": "copy_exact_color_triplet",
                                "profile": "regression_tiny",
                                "fixture_budget_passed": true,
                                "any_budget_passed": true,
                                "first_passing_budget": 4,
                                "blocker_class": "fixture_budget_passes"
                            }
                        ]
                    },
                    {
                        "runtime_device": "NPU",
                        "fallback_used": false,
                        "summary": {
                            "cases_total": 2,
                            "fixture_budget_passed": 0,
                            "any_budget_passed": 1,
                            "blocker_classes": {
                                "fixture_budget_overgenerates_but_smaller_budget_passes": 1,
                                "no_budget_variant_passes": 1
                            }
                        },
                        "cases": [
                            {
                                "id": "yes_no_clear_sky",
                                "profile": "ask_short",
                                "fixture_budget_passed": false,
                                "any_budget_passed": true,
                                "first_passing_budget": 1,
                                "blocker_class": "fixture_budget_overgenerates_but_smaller_budget_passes"
                            },
                            {
                                "id": "stop_token_one_word_done",
                                "profile": "regression_tiny",
                                "fixture_budget_passed": false,
                                "any_budget_passed": false,
                                "first_passing_budget": null,
                                "blocker_class": "no_budget_variant_passes"
                            },
                            {
                                "id": "copy_exact_color_triplet",
                                "profile": "regression_tiny",
                                "fixture_budget_passed": true,
                                "any_budget_passed": true,
                                "first_passing_budget": 4,
                                "blocker_class": "fixture_budget_passes"
                            }
                        ]
                    }
                ]
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

        let profiles = build_route_profile_comparison_with_created_utc_and_budget_diagnostics(
            temp.path(),
            Path::new(ROUTE_PROMOTION_LEDGER),
            Path::new(DENSE_PHASE_COMPARISON),
            Some(Path::new("corpus-v2.yaml")),
            Some(Path::new(DENSE_CPU_CORPUS_V2)),
            Some(Path::new(DENSE_OV_CORPUS_V2)),
            None,
            Some(Path::new(OPENVINO_GPU_CORPUS_V2_DIAGNOSIS)),
            Some(Path::new("lunar-lake-openvino-npu-corpus-v2-diagnosis.json")),
            Some(Path::new(OPENVINO_NPU_COLD_START_DIAGNOSIS)),
            None,
            None,
            Some(Path::new(OPENVINO_GENERATION_BUDGET_SENSITIVITY)),
            None,
            "2026-05-16T07:30:00Z".to_string(),
        )?;

        assert!(profiles.profile_comparison_ready, "{:?}", profiles.gaps);
        assert_eq!(
            profiles.answer_corpus_v2_fixture.as_deref(),
            Some(path_string(&temp.path().join("corpus-v2.yaml")).as_str())
        );
        assert_eq!(profiles.route_diagnosis_receipts.len(), 4);
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
        assert_eq!(gpu_route.timing_applicability.measured_prompt_tokens, Some(41));
        assert_eq!(gpu_route.timing_applicability.measured_output_tokens, Some(2));
        assert!(gpu_route.timing_applicability.timing_matches_profile);
        assert!(gpu_route.timing.throughput_tokens_per_s.is_some_and(|value| value > 0.0));
        let gpu_advantage = gpu_route
            .route_advantage_context
            .as_ref()
            .context("missing GPU route advantage context")?;
        assert_eq!(gpu_advantage.baseline_route_id, DEFAULT_ASK_ROUTE);
        assert!(!gpu_advantage.benchmark_qualified);
        assert_eq!(gpu_advantage.qualification_status, "diagnostic_only_not_benchmark_qualified");
        assert!(gpu_advantage.route_total_response_ms.is_some());
        assert!(gpu_advantage.baseline_total_response_ms.is_some());
        assert!(gpu_advantage.route_to_baseline_total_response_ratio.is_some());
        assert!(
            gpu_advantage
                .qualification_blockers
                .iter()
                .any(|blocker| { blocker.contains("benchmark-qualified advantage is false") })
        );
        assert!(gpu_advantage.qualification_blockers.iter().any(|blocker| {
            blocker
                .contains("baseline route dense_slm_default_cpu is not benchmark-reference-ready")
        }));
        assert!(
            gpu_route.timing.phase_coverage.iter().any(|coverage| {
                coverage == "profile_timing_supplemented_from_corpus_v2_case_yes_no_clear_sky"
            }),
            "{:?}",
            gpu_route.timing.phase_coverage
        );
        assert!(
            !gpu_route.timing.known_gaps.contains(&"profile regression bundle missing".to_string())
        );
        assert!(!gpu_route.blockers.contains(
            &"timing evidence is not profile-specific for profile ask_short".to_string()
        ));
        assert!(gpu_route.blockers.iter().any(|blocker| {
            blocker.contains("corpus_v2 profile ask_short has 1 quality failures")
        }));
        assert!(gpu_route.blockers.contains(&"GPU ask_short diagnosis blocker".to_string()));
        assert!(
            gpu_route.blockers.contains(
                &"yes_no_clear_sky failed corpus-v2 diagnosis: exact_answer_overgenerated"
                    .to_string()
            ),
            "{:?}",
            gpu_route.blockers
        );
        assert!(
            gpu_route.blockers.contains(
                &"OpenVINO generated token IDs are retokenized, not direct pipeline internals"
                    .to_string()
            )
        );
        assert!(gpu_route.blockers.contains(
            &"yes_no_clear_sky overgenerates at the fixture budget but passes with max_new_tokens=1"
                .to_string()
        ));
        assert!(
            !gpu_route.blockers.contains(&"GPU regression_tiny diagnosis blocker".to_string()),
            "{:?}",
            gpu_route.blockers
        );
        assert!(
            !gpu_route.blockers.contains(
                &"stop_token_one_word_done has no passing tested generation budget".to_string()
            ),
            "{:?}",
            gpu_route.blockers
        );
        assert!(
            gpu_route.blockers.iter().any(|blocker| blocker.contains(
                "dense_slm_openvino_gpu_candidate corpus-v2 receipt is missing active fixture cases [arithmetic_add_7_8, short_reasoning_apples_left]"
            )),
            "{:?}",
            gpu_route.blockers
        );
        assert!(
            gpu_route.blockers.iter().any(|blocker| blocker.contains(
                "dense_slm_openvino_gpu_candidate corpus-v2 receipt has stale or unexpected cases [short_reasoning_heavier_object]"
            )),
            "{:?}",
            gpu_route.blockers
        );
        let Some(prefill_heavy) =
            profiles.profiles.iter().find(|profile| profile.profile_id == "prefill_heavy")
        else {
            bail!("missing prefill_heavy profile");
        };
        let prefill_gpu = prefill_heavy
            .route_evidence
            .iter()
            .find(|route| route.route_id == "dense_slm_openvino_gpu_candidate")
            .context("missing prefill_heavy GPU route evidence")?;
        assert_eq!(prefill_gpu.timing_applicability.measured_prompt_tokens, Some(97));
        assert_eq!(prefill_gpu.timing_applicability.measured_output_tokens, Some(22));
        assert!(!prefill_gpu.timing_applicability.timing_matches_profile);
        assert!(prefill_gpu.timing.phase_coverage.iter().any(|coverage| {
            coverage
                == "profile_timing_supplemented_from_corpus_v2_case_long_prompt_summary_route_policy"
        }));
        assert!(
            prefill_gpu
                .timing_applicability
                .notes
                .iter()
                .any(|note| { note.contains("prompt timing count 97 does not satisfy `>=2048`") })
        );
        let Some(low_power) =
            profiles.profiles.iter().find(|profile| profile.profile_id == "low_power")
        else {
            bail!("missing low_power profile");
        };
        let low_power_gpu = low_power
            .route_evidence
            .iter()
            .find(|route| route.route_id == "dense_slm_openvino_gpu_candidate")
            .context("missing low_power GPU route evidence")?;
        assert!(low_power_gpu.timing.phase_coverage.iter().any(|coverage| {
            coverage
                == "profile_timing_supplemented_from_corpus_v2_case_low_power_route_evidence_copy"
        }));
        assert!(
            low_power_gpu
                .blockers
                .contains(&"power telemetry receipt missing for low_power promotion".to_string())
        );
        let Some(npu_route) = ask_short
            .route_evidence
            .iter()
            .find(|route| route.route_id == "dense_slm_openvino_npu_candidate")
        else {
            bail!("missing NPU route evidence");
        };
        assert!(npu_route.blockers.contains(&"NPU ask_short diagnosis blocker".to_string()));
        assert!(npu_route.blockers.contains(
            &"NPU cold start is openvino_pipeline_load_or_device_compile_dominated".to_string()
        ));
        assert!(
            npu_route
                .blockers
                .contains(&"NPU cache or resident warm-route proof is missing".to_string())
        );
        assert!(npu_route.blockers.contains(
            &"yes_no_clear_sky overgenerates at the fixture budget but passes with max_new_tokens=1"
                .to_string()
        ));
        assert!(
            !npu_route.blockers.contains(&"NPU regression_tiny diagnosis blocker".to_string()),
            "{:?}",
            npu_route.blockers
        );
        assert!(
            !npu_route.blockers.contains(
                &"stop_token_one_word_done has no passing tested generation budget".to_string()
            ),
            "{:?}",
            npu_route.blockers
        );
        assert!(
            profiles.promotion_blocker_summary.iter().any(|summary| {
                summary.blocker.contains("generated token IDs")
                    && summary.route_ids.contains(&"dense_slm_openvino_gpu_candidate".to_string())
                    && summary.route_ids.contains(&"dense_slm_openvino_npu_candidate".to_string())
                    && summary.next_action.contains("direct OpenVINO generated-token visibility")
            }),
            "{:?}",
            profiles.promotion_blocker_summary
        );
        assert!(
            profiles.promotion_blocker_summary.iter().any(|summary| {
                summary.blocker == "benchmark_qualified_speedup_or_power_advantage"
                    && summary.profile_ids.contains(&"ask_short".to_string())
                    && summary.next_action.contains("benchmark-qualified latency")
            }),
            "{:?}",
            profiles.promotion_blocker_summary
        );
        let Some(regression_tiny) =
            profiles.profiles.iter().find(|profile| profile.profile_id == "regression_tiny")
        else {
            bail!("missing regression_tiny profile");
        };
        let regression_gpu = regression_tiny
            .route_evidence
            .iter()
            .find(|route| route.route_id == "dense_slm_openvino_gpu_candidate")
            .context("missing regression_tiny GPU route evidence")?;
        assert!(
            regression_gpu.blockers.contains(&"GPU regression_tiny diagnosis blocker".to_string())
        );
        assert!(regression_gpu.blockers.contains(
            &"stop_token_one_word_done has no passing tested generation budget".to_string()
        ));
        assert!(
            !regression_gpu.blockers.iter().any(|blocker| blocker.contains(
                "copy_exact_color_triplet has generation-budget sensitivity class fixture_budget_passes"
            )),
            "{:?}",
            regression_gpu.blockers
        );
        Ok(())
    }

    #[test]
    fn profile_regression_bundle_blocker_requires_clean_profile_quality() {
        let clean = ProfileQualityEvidence {
            source_receipt: "clean.json".to_string(),
            route_id: "dense_slm_openvino_gpu_candidate".to_string(),
            profile_id: "ask_short".to_string(),
            profile_present: true,
            cases_total: 4,
            passed: 4,
            failed: 0,
            fallback_used: Some(false),
            status: "passed".to_string(),
            notes: Vec::new(),
        };
        assert!(profile_regression_bundle_evidence_satisfied(Some(&clean)));

        let missing_profile = ProfileQualityEvidence { profile_present: false, ..clean.clone() };
        assert!(!profile_regression_bundle_evidence_satisfied(Some(&missing_profile)));

        let failed = ProfileQualityEvidence { failed: 1, passed: 3, ..clean.clone() };
        assert!(!profile_regression_bundle_evidence_satisfied(Some(&failed)));

        let fallback = ProfileQualityEvidence { fallback_used: Some(true), ..clean };
        assert!(!profile_regression_bundle_evidence_satisfied(Some(&fallback)));
        assert!(!profile_regression_bundle_evidence_satisfied(None));
    }

    #[test]
    fn route_profile_comparison_uses_openvino_profile_run_for_heavy_profiles() -> Result<()> {
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
                "tokens": {"prompt_count": 38, "generated_count": 8}
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
            OPENVINO_PROFILE_RUN,
            json!({
                "artifact_kind": "intel_258v_dense_slm_openvino_profile_run",
                "fallback_used": false,
                "generation": {
                    "devices": [
                        {
                            "runtime_device": "GPU.0",
                            "fallback_used": false,
                            "cases": [
                                {
                                    "id": "prefill_heavy_route_policy_long_context",
                                    "profile": "prefill_heavy",
                                    "prompt_token_count": 2731,
                                    "generated_token_count": 64,
                                    "generated_token_ids_available_from_pipeline": true,
                                    "timing": {
                                        "pipeline_construct_wall_ms": 1000.0,
                                        "generation_wall_ms": 1500.0,
                                        "first_streamed_text_chunk_ms": 600.0,
                                        "openvino_perf_metrics": {
                                            "load_time_ms": 900.0,
                                            "tokenization": {"mean_ms": 2.0},
                                            "time_to_first_token": {"mean_ms": 610.0},
                                            "num_generated_tokens": 64,
                                            "throughput": {"mean_ms": 40.0}
                                        }
                                    }
                                },
                                {
                                    "id": "decode_heavy_route_policy_long_generation",
                                    "profile": "decode_heavy",
                                    "prompt_token_count": 66,
                                    "generated_token_count": 512,
                                    "generated_token_ids_available_from_pipeline": true,
                                    "timing": {
                                        "pipeline_construct_wall_ms": 1000.0,
                                        "generation_wall_ms": 9000.0,
                                        "first_streamed_text_chunk_ms": 400.0,
                                        "openvino_perf_metrics": {
                                            "load_time_ms": 900.0,
                                            "tokenization": {"mean_ms": 2.0},
                                            "time_to_first_token": {"mean_ms": 410.0},
                                            "num_generated_tokens": 512,
                                            "throughput": {"mean_ms": 55.0}
                                        }
                                    }
                                }
                            ]
                        },
                        {
                            "runtime_device": "NPU",
                            "fallback_used": false,
                            "cases": [
                                {
                                    "id": "prefill_heavy_route_policy_long_context",
                                    "profile": "prefill_heavy",
                                    "prompt_token_count": 2731,
                                    "generated_token_count": 64,
                                    "generated_token_ids_available_from_pipeline": true,
                                    "timing": {
                                        "pipeline_construct_wall_ms": 1000.0,
                                        "generation_wall_ms": 4200.0,
                                        "first_streamed_text_chunk_ms": 1400.0
                                    }
                                },
                                {
                                    "id": "decode_heavy_route_policy_long_generation",
                                    "profile": "decode_heavy",
                                    "prompt_token_count": 66,
                                    "generated_token_count": 512,
                                    "generated_token_ids_available_from_pipeline": true,
                                    "timing": {
                                        "pipeline_construct_wall_ms": 1000.0,
                                        "generation_wall_ms": 24000.0,
                                        "first_streamed_text_chunk_ms": 400.0
                                    }
                                }
                            ]
                        }
                    ]
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
            None,
            Some(Path::new(DENSE_CPU_CORPUS_V2)),
            Some(Path::new(DENSE_OV_CORPUS_V2)),
            None,
            "2026-05-19T07:20:00Z".to_string(),
        )?;

        let prefill = profiles
            .profiles
            .iter()
            .find(|profile| profile.profile_id == "prefill_heavy")
            .context("missing prefill_heavy profile")?;
        let gpu_prefill = prefill
            .route_evidence
            .iter()
            .find(|route| route.route_id == "dense_slm_openvino_gpu_candidate")
            .context("missing GPU prefill route")?;
        assert_eq!(gpu_prefill.timing_applicability.measured_prompt_tokens, Some(2731));
        assert_eq!(gpu_prefill.timing_applicability.measured_output_tokens, Some(64));
        assert!(gpu_prefill.timing_applicability.timing_matches_profile);
        assert!(gpu_prefill.timing.phase_coverage.iter().any(|coverage| {
            coverage
                == "profile_timing_from_openvino_profile_run_case_prefill_heavy_route_policy_long_context"
        }));

        let decode = profiles
            .profiles
            .iter()
            .find(|profile| profile.profile_id == "decode_heavy")
            .context("missing decode_heavy profile")?;
        let npu_decode = decode
            .route_evidence
            .iter()
            .find(|route| route.route_id == "dense_slm_openvino_npu_candidate")
            .context("missing NPU decode route")?;
        assert_eq!(npu_decode.timing_applicability.measured_prompt_tokens, Some(66));
        assert_eq!(npu_decode.timing_applicability.measured_output_tokens, Some(512));
        assert!(npu_decode.timing_applicability.timing_matches_profile);
        assert_eq!(profiles.timing_coverage.candidate_proxy_or_missing_route_count, 0);
        assert!(
            !profiles
                .timing_coverage
                .proxy_or_missing_routes
                .iter()
                .any(|route| route.contains("dense_slm_openvino"))
        );
        Ok(())
    }

    #[test]
    fn npu_resident_session_clears_missing_warm_route_proof_blocker() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_json(
            temp.path(),
            OPENVINO_NPU_RESIDENT_SESSION,
            json!({
                "artifact_kind": "lunar_lake_openvino_npu_resident_session",
                "selected_backend": "openvino-npu",
                "runtime_device": "NPU",
                "fallback_used": false,
                "resident_session": {
                    "resident_session_ready": true,
                    "warm_resident_asks": {
                        "ask_count": 10,
                        "passed": 10,
                        "failed": 0,
                        "fallback_used": false
                    }
                },
                "claim_boundary": {
                    "route_promotion_changed": false,
                    "speedup_claim": false,
                    "power_advantage_claim": false,
                    "acceleration_claim": false,
                    "native_npu_inference_claim": false,
                    "bitnet_qk256_i2s_behavior_changed": false
                }
            }),
        )?;
        write_json(
            temp.path(),
            OPENVINO_NPU_COLD_START_DIAGNOSIS,
            json!({
                "artifact_kind": "lunar_lake_openvino_npu_cold_start_diagnosis",
                "cold_start": {
                    "cold_load_dominant": true,
                    "classification": "openvino_pipeline_load_or_device_compile_dominated"
                },
                "claim_boundary": {
                    "route_promotion_changed": false,
                    "speedup_claim": false,
                    "power_advantage_claim": false,
                    "acceleration_claim": false
                }
            }),
        )?;

        let mut gaps = Vec::new();
        let diagnostics = load_route_diagnostics_index(
            temp.path(),
            None,
            None,
            Some(Path::new(OPENVINO_NPU_COLD_START_DIAGNOSIS)),
            Some(Path::new(OPENVINO_NPU_RESIDENT_SESSION)),
            None,
            None,
            &mut gaps,
        )?;
        assert!(gaps.is_empty(), "{gaps:?}");
        let evidence = diagnostics.get("dense_slm_openvino_npu_candidate", "ask_short");
        assert!(
            evidence
                .source_receipts
                .iter()
                .any(|source| { source.ends_with(OPENVINO_NPU_RESIDENT_SESSION) })
        );
        assert!(evidence.blockers.contains(
            &"NPU cold start is openvino_pipeline_load_or_device_compile_dominated".to_string()
        ));
        assert!(
            !evidence
                .blockers
                .contains(&"NPU cache or resident warm-route proof is missing".to_string()),
            "{:?}",
            evidence.blockers
        );
        let warm_evidence = diagnostics.get("dense_slm_openvino_npu_candidate", "warm_resident");
        assert!(
            !warm_evidence.blockers.iter().any(|blocker| blocker.contains("NPU cold start")),
            "{:?}",
            warm_evidence.blockers
        );
        assert!(
            warm_evidence
                .source_receipts
                .iter()
                .any(|source| { source.ends_with(OPENVINO_NPU_RESIDENT_SESSION) })
        );
        Ok(())
    }

    #[test]
    fn npu_cache_experiment_adds_cache_context_without_promoting_npu() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_json(
            temp.path(),
            OPENVINO_NPU_CACHE_EXPERIMENT,
            json!({
                "artifact_kind": "lunar_lake_openvino_npu_cache_experiment",
                "selected_backend": "openvino-npu",
                "runtime_device": "NPU",
                "fallback_used": false,
                "cache": {
                    "cache_hit_runtime_metric_available": false,
                    "cache_effective_by_timing": false
                },
                "comparison": {
                    "cache_experiment_ready": true,
                    "first_answer_gate_passed": true,
                    "second_answer_gate_passed": true,
                    "classification": "cache_not_materially_proven_for_pipeline_construct"
                },
                "generated_token_visibility": {
                    "direct_generated_token_ids_available": false
                },
                "claim_boundary": {
                    "route_promotion_changed": false,
                    "speedup_claim": false,
                    "power_advantage_claim": false,
                    "acceleration_claim": false,
                    "native_npu_inference_claim": false,
                    "bitnet_qk256_i2s_behavior_changed": false
                }
            }),
        )?;

        let mut gaps = Vec::new();
        let diagnostics = load_route_diagnostics_index(
            temp.path(),
            None,
            None,
            None,
            None,
            Some(Path::new(OPENVINO_NPU_CACHE_EXPERIMENT)),
            None,
            &mut gaps,
        )?;
        assert!(gaps.is_empty(), "{gaps:?}");
        let evidence = diagnostics.get("dense_slm_openvino_npu_candidate", "low_power");
        assert!(
            evidence
                .source_receipts
                .iter()
                .any(|source| source.ends_with(OPENVINO_NPU_CACHE_EXPERIMENT)),
            "{:?}",
            evidence.source_receipts
        );
        assert!(evidence.blockers.contains(
            &"NPU cache hit evidence is inferred from cache files/timing, not an OpenVINO runtime metric"
                .to_string()
        ));
        assert!(evidence.blockers.iter().any(|blocker| {
            blocker.contains(
                "NPU cached cold process does not materially reduce pipeline construction",
            )
        }));
        Ok(())
    }

    #[test]
    fn npu_cache_experiment_accepts_file_and_timing_cache_evidence() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_json(
            temp.path(),
            OPENVINO_NPU_CACHE_EXPERIMENT,
            json!({
                "artifact_kind": "lunar_lake_openvino_npu_cache_experiment",
                "selected_backend": "openvino-npu",
                "runtime_device": "NPU",
                "fallback_used": false,
                "cache": {
                    "cache_hit_runtime_metric_available": false,
                    "cache_files_created": true,
                    "cache_files_reused_or_stable": true,
                    "cache_effective_by_timing": true
                },
                "comparison": {
                    "cache_experiment_ready": true,
                    "first_answer_gate_passed": true,
                    "second_answer_gate_passed": true,
                    "classification": "cache_materially_reduces_pipeline_construct"
                },
                "generated_token_visibility": {
                    "direct_generated_token_ids_available": false
                },
                "claim_boundary": {
                    "route_promotion_changed": false,
                    "speedup_claim": false,
                    "power_advantage_claim": false,
                    "acceleration_claim": false,
                    "native_npu_inference_claim": false,
                    "bitnet_qk256_i2s_behavior_changed": false
                }
            }),
        )?;

        let mut gaps = Vec::new();
        let diagnostics = load_route_diagnostics_index(
            temp.path(),
            None,
            None,
            None,
            None,
            Some(Path::new(OPENVINO_NPU_CACHE_EXPERIMENT)),
            None,
            &mut gaps,
        )?;
        assert!(gaps.is_empty(), "{gaps:?}");
        let evidence = diagnostics.get("dense_slm_openvino_npu_candidate", "low_power");
        assert!(
            evidence
                .source_receipts
                .iter()
                .any(|source| source.ends_with(OPENVINO_NPU_CACHE_EXPERIMENT)),
            "{:?}",
            evidence.source_receipts
        );
        assert!(
            !evidence.blockers.iter().any(|blocker| blocker.contains("NPU cache hit evidence")),
            "{:?}",
            evidence.blockers
        );
        assert!(
            !evidence.blockers.iter().any(|blocker| blocker.contains("NPU cached cold process")),
            "{:?}",
            evidence.blockers
        );
        assert!(
            evidence.blockers.contains(
                &"OpenVINO generated token IDs are retokenized, not direct pipeline internals"
                    .to_string()
            )
        );
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
                "tokens": {"prompt_count": 38, "generated_count": 8}
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
            None,
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
        assert!(benchmark.timing_coverage.route_count > 0);
        assert!(benchmark.timing_coverage.promotion_eligible_routes_have_profile_specific_timing);
        assert!(benchmark.timing_coverage.proxy_or_missing_timing_routes_blocked);
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
        assert!(cpu.timing_applicability.timing_matches_profile);
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
        let gpu_advantage = gpu
            .route_advantage_context
            .as_ref()
            .context("missing benchmark GPU route advantage context")?;
        assert_eq!(gpu_advantage.baseline_route_id, DEFAULT_ASK_ROUTE);
        assert_eq!(gpu_advantage.qualification_status, "diagnostic_only_not_benchmark_qualified");
        assert!(!gpu_advantage.benchmark_qualified);
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
                "tokens": {"prompt_count": 38, "generated_count": 8}
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
            None,
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
    fn cpu_slm_runtime_comparison_blocks_openvino_cpu_on_profile_quality() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_json(
            temp.path(),
            "phase-attribution.json",
            json!({
                "artifact_kind": "lunar_lake_cpu_slm_phase_attribution",
                "attribution_ready": true,
                "cold_one_off": {
                    "profile_id": "ask_short",
                    "timing": {"total_response_ms": 200.0}
                }
            }),
        )?;
        write_json(
            temp.path(),
            "resident-source.json",
            json!({
                "artifact_kind": "slm_cpu_warm_session",
                "selected_backend": "cpu-rust",
                "runtime_api": "cpu",
                "fallback_used": false,
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
                "model": {"family": "qwen", "architecture": "qwen2", "quant_format": "Q8_0", "tokenizer": "tokenizer.json"},
                "generation": {"prompt_template": "qwen2.5"},
                "session": {"model_loaded_once": true, "tokenizer_loaded_once": true, "prompt_count": 2},
                "timing": {"model_load_ms": 100.0, "tokenizer_load_ms": 10.0},
                "prompts": [
                    {
                        "prompt_index": 0,
                        "case_id": "ask_short_math",
                        "fallback_used": false,
                        "generated_tokens": 4,
                        "quality": {"passed": true},
                        "timing": {"model_load_ms": 0.0, "tokenizer_load_ms": 0.0, "total_ms": 80.0, "time_to_first_token_ms": 30.0, "decode_total_ms": 40.0, "tokenize_ms": 2.0}
                    },
                    {
                        "prompt_index": 1,
                        "case_id": "ask_short_math",
                        "fallback_used": false,
                        "generated_tokens": 4,
                        "quality": {"passed": true},
                        "timing": {"model_load_ms": 0.0, "tokenizer_load_ms": 0.0, "total_ms": 100.0, "time_to_first_token_ms": 40.0, "decode_total_ms": 44.0, "tokenize_ms": 3.0}
                    }
                ]
            }),
        )?;
        let resident = build_cpu_slm_resident_session_with_created_utc(
            temp.path(),
            Path::new("phase-attribution.json"),
            Path::new("resident-source.json"),
            2,
            "2026-05-17T09:35:00Z".to_string(),
        )?;
        fs::write(temp.path().join("resident.json"), serde_json::to_vec_pretty(&resident)?)?;
        write_json(
            temp.path(),
            "ov-corpus.json",
            json!({
                "artifact_kind": "intel_258v_dense_slm_openvino_corpus_v2",
                "fallback_used": false,
                "generation": {
                    "devices": [
                        {
                            "selected_backend": "openvino-cpu",
                            "runtime_api": "openvino_genai",
                            "runtime_device": "CPU",
                            "fallback_used": false,
                            "selected_kernel_or_runtime": "openvino-genai-llmpipeline-cpu",
                            "pipeline_construct_wall_ms": 20.0,
                            "quality_summary": {
                                "profile_summary": {
                                    "ask_short": {"total": 2, "passed": 1, "failed": 1}
                                }
                            },
                            "cases": [
                                {
                                    "id": "ask_short_pass",
                                    "profile": "ask_short",
                                    "status": "passed",
                                    "fallback_used": false,
                                    "timing": {
                                        "generation_wall_ms": 10.0,
                                        "first_streamed_text_chunk_ms": 5.0,
                                        "openvino_perf_metrics": {
                                            "tokenization": {"mean_ms": 1.0},
                                            "num_generated_tokens": 4,
                                            "throughput": {"mean_ms": 20.0}
                                        }
                                    }
                                },
                                {
                                    "id": "ask_short_fail",
                                    "profile": "ask_short",
                                    "status": "failed",
                                    "fallback_used": false,
                                    "timing": {
                                        "generation_wall_ms": 12.0,
                                        "first_streamed_text_chunk_ms": 6.0,
                                        "openvino_perf_metrics": {
                                            "tokenization": {"mean_ms": 2.0},
                                            "num_generated_tokens": 4,
                                            "throughput": {"mean_ms": 18.0}
                                        }
                                    }
                                }
                            ]
                        }
                    ]
                }
            }),
        )?;
        write_json(
            temp.path(),
            "ov-phase.json",
            json!({
                "artifact_kind": "intel_258v_dense_slm_openvino_phase_runner",
                "fallback_used": false,
                "generation": {
                    "devices": [
                        {
                            "selected_backend": "openvino-cpu",
                            "runtime_api": "openvino_genai",
                            "runtime_device": "CPU",
                            "fallback_used": false,
                            "pipeline_construct_wall_ms": 25.0
                        }
                    ]
                }
            }),
        )?;

        let receipt = build_cpu_slm_runtime_comparison_with_created_utc(
            temp.path(),
            Path::new("resident.json"),
            Path::new("ov-corpus.json"),
            Path::new("ov-phase.json"),
            "2026-05-17T09:40:00Z".to_string(),
        )?;

        assert!(receipt.comparison_ready, "{:?}", receipt.gaps);
        assert_eq!(receipt.artifact_kind, "lunar_lake_cpu_slm_runtime_comparison");
        assert_eq!(receipt.rust_gguf_cpu.selected_backend, "cpu-rust");
        assert_eq!(receipt.openvino_cpu.selected_backend, "openvino-cpu");
        let profile = receipt
            .profiles
            .iter()
            .find(|profile| profile.profile_id == "ask_short")
            .context("missing ask_short profile")?;
        assert_eq!(profile.openvino_cpu.cases_failed, Some(1));
        assert_eq!(profile.openvino_cpu.timing_ms.mean, Some(11.0));
        assert_eq!(profile.openvino_to_rust_total_ratio, Some(11.0 / 90.0));
        assert_eq!(profile.status, "blocked_candidate_context_only");
        assert!(profile.blockers.iter().any(|blocker| blocker.contains("answer-gate failure")));
        assert!(!receipt.claim_boundary.new_inference_executed);
        assert!(!receipt.claim_boundary.route_promotion_changed);
        assert!(!receipt.claim_boundary.speedup_claim);
        assert!(!receipt.claim_boundary.arc_npu_execution_claim);
        assert!(!receipt.claim_boundary.bitnet_qk256_i2s_claim);
        Ok(())
    }

    #[test]
    fn openvino_quality_diagnosis_classifies_gpu_corpus_failures() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_answer_corpus_v2(temp.path(), "corpus-v2.yaml")?;
        write_json(
            temp.path(),
            "ov-corpus.json",
            json!({
                "artifact_kind": "intel_258v_dense_slm_openvino_corpus_v2",
                "fallback_used": false,
                "generation": {
                    "devices": [
                        {
                            "selected_backend": "openvino-gpu",
                            "runtime_api": "openvino_genai",
                            "runtime_device": "GPU.0",
                            "backend_lane": "dense_slm_openvino_gpu",
                            "selected_kernel_or_runtime": "openvino-genai-llmpipeline-gpu",
                            "fallback_used": false,
                            "promotion_status": "candidate_only_not_promoted",
                            "quality_summary": {
                                "cases_total": 2,
                                "passed": 1,
                                "failed": 1,
                                "profile_summary": {
                                    "ask_short": {"total": 2, "passed": 1, "failed": 1}
                                },
                                "category_summary": {
                                    "yes_no": {"total": 1, "passed": 0, "failed": 1}
                                }
                            },
                            "cases": [
                                {
                                    "id": "yes_no_clear_sky",
                                    "profile": "ask_short",
                                    "category": "yes_no",
                                    "status": "failed",
                                    "prompt_token_count": 40,
                                    "generated_token_count": 6,
                                    "generated_text": "Yes, the sky on a clear day",
                                    "generated_token_ids_available_from_pipeline": false,
                                    "generated_token_ids_source": "retokenized_decoded_text",
                                    "fallback_used": false,
                                    "quality": {
                                        "passed": false,
                                        "gate_kind": "normalized_match",
                                        "failed_rules": ["normalized_match_failed"],
                                        "failure_taxonomy": ["normalized_match_failed"],
                                        "scoring": {
                                            "kind": "normalized_match",
                                            "passed": false,
                                            "details": {
                                                "expected_normalized": "yes",
                                                "observed_normalized": "yes the sky on a clear day"
                                            }
                                        }
                                    }
                                },
                                {
                                    "id": "math_2_plus_2_brief",
                                    "profile": "ask_short",
                                    "category": "math",
                                    "status": "passed",
                                    "generated_token_ids_available_from_pipeline": false,
                                    "generated_token_ids_source": "retokenized_decoded_text",
                                    "fallback_used": false,
                                    "quality": {"passed": true}
                                }
                            ]
                        }
                    ]
                }
            }),
        )?;

        let receipt = build_openvino_corpus_v2_diagnosis_with_created_utc(
            temp.path(),
            Path::new("ov-corpus.json"),
            Some(Path::new("corpus-v2.yaml")),
            "GPU.0",
            "2026-05-17T09:55:00Z".to_string(),
        )?;

        assert!(receipt.diagnosis_ready, "{:?}", receipt.gaps);
        assert_eq!(receipt.artifact_kind, "lunar_lake_openvino_corpus_v2_diagnosis");
        assert_eq!(receipt.runtime_device.as_deref(), Some("GPU.0"));
        assert!(receipt.route_blocked);
        assert_eq!(receipt.quality_summary.failed, 1);
        assert_eq!(receipt.failed_cases.len(), 1);
        assert_eq!(receipt.failed_cases[0].answer_preview, "Yes, the sky on a clear day");
        assert_eq!(receipt.failed_cases[0].classification, "exact_answer_overgenerated");
        assert_eq!(
            receipt.answer_corpus_v2_fixture.as_deref(),
            Some(path_string(&temp.path().join("corpus-v2.yaml")).as_str())
        );
        assert!(receipt.case_alignment.fixture_verified);
        assert_eq!(receipt.case_alignment.observed_case_count, 2);
        assert_eq!(receipt.case_alignment.expected_case_count, Some(14));
        assert_eq!(receipt.case_alignment.aligned_with_active_fixture, Some(false));
        assert!(
            receipt
                .case_alignment
                .missing_case_ids
                .iter()
                .any(|case_id| case_id == "short_reasoning_apples_left")
        );
        assert!(
            receipt.blocker_summary.iter().any(
                |blocker| blocker.contains("corpus-v2 receipt is missing active fixture cases")
            )
        );
        assert!(!receipt.generated_token_visibility.direct_generated_token_ids_available);
        assert!(receipt.generated_token_visibility.retokenized_generated_ids_used);
        assert!(
            receipt
                .blocker_summary
                .iter()
                .any(|blocker| blocker.contains("generated token IDs are not directly available"))
        );
        assert!(!receipt.claim_boundary.new_inference_executed);
        assert!(!receipt.claim_boundary.route_promotion_changed);
        assert!(!receipt.claim_boundary.speedup_claim);
        assert!(!receipt.claim_boundary.arc_or_npu_execution_claim);
        assert!(!receipt.claim_boundary.bitnet_qk256_i2s_behavior_changed);
        Ok(())
    }

    #[test]
    fn corpus_v2_failure_classifier_separates_exact_and_keyword_contracts() {
        let empty = Vec::<String>::new();
        assert_eq!(
            classify_corpus_v2_failure(
                "Yes, it is usually blue",
                Some("starts_with_any"),
                Some(true),
                Some(false),
                &["normalized_match_failed".to_string()],
                &empty,
                Some(8),
                Some("yes"),
                Some("yes, it is usually blue"),
            ),
            "exact_answer_overgenerated"
        );
        assert_eq!(
            classify_corpus_v2_failure(
                "Yes, Tom still has one apple",
                Some("contains_any"),
                Some(true),
                Some(false),
                &["required_keywords_missing".to_string()],
                &["yes".to_string()],
                Some(8),
                None,
                None,
            ),
            "case_sensitive_required_keyword_mismatch"
        );
        assert_eq!(
            classify_corpus_v2_failure(
                "Fallback route check",
                Some("readable"),
                Some(true),
                Some(false),
                &["required_keywords_missing".to_string()],
                &["fallback".to_string(), "model".to_string()],
                Some(16),
                None,
                None,
            ),
            "required_terms_missing_or_case_mismatch"
        );
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
    fn telemetry_context_blocks_ac_sample_when_battery_required() {
        let receipt = build_telemetry_context_from_parts(
            "2026-05-19T22:30:00Z".to_string(),
            TelemetryMemoryContext {
                source: "test_memory".to_string(),
                total_bytes: Some(16),
                available_bytes: Some(8),
                used_bytes: Some(8),
            },
            TelemetryPowerContext {
                source: "test_power".to_string(),
                active_scheme: Some("Balanced".to_string()),
                battery_status: Some("BatteryStatus=2;EstimatedChargeRemaining=100".to_string()),
                ac_power_inferred: Some(true),
            },
            TelemetryThermalContext {
                source: "test_thermal".to_string(),
                thermal_zones_visible: Some(1),
                temperatures_celsius: Vec::new(),
            },
            true,
        );

        assert_eq!(
            receipt.proof_stage,
            "battery_mode_telemetry_context_blocked_no_promotion_change"
        );
        assert_eq!(receipt.telemetry_scope, "low_power_battery_mode_telemetry");
        assert!(receipt.capture_requirements.battery_mode_required);
        assert!(!receipt.capture_requirements.battery_mode_sample_recorded);
        assert!(!receipt.capture_requirements.requirement_satisfied);
        assert_eq!(receipt.capture_requirements.status, "blocked");
        assert!(receipt.capture_requirements.gaps.iter().any(|gap| {
            gap == "battery-mode telemetry sample required but current power context indicates AC power"
        }));
        assert!(!receipt.claim_boundary.new_inference_executed);
        assert!(!receipt.claim_boundary.power_advantage_claim);
        assert!(!receipt.claim_boundary.acceleration_claim);
    }

    #[test]
    fn telemetry_context_accepts_battery_sample_when_required() {
        let receipt = build_telemetry_context_from_parts(
            "2026-05-19T22:31:00Z".to_string(),
            TelemetryMemoryContext {
                source: "test_memory".to_string(),
                total_bytes: Some(16),
                available_bytes: Some(8),
                used_bytes: Some(8),
            },
            TelemetryPowerContext {
                source: "test_power".to_string(),
                active_scheme: Some("Balanced".to_string()),
                battery_status: Some("BatteryStatus=1;EstimatedChargeRemaining=96".to_string()),
                ac_power_inferred: Some(false),
            },
            TelemetryThermalContext {
                source: "test_thermal".to_string(),
                thermal_zones_visible: Some(1),
                temperatures_celsius: Vec::new(),
            },
            true,
        );

        assert_eq!(
            receipt.proof_stage,
            "battery_mode_telemetry_context_captured_no_promotion_change"
        );
        assert!(receipt.capture_requirements.battery_mode_required);
        assert!(receipt.capture_requirements.battery_mode_sample_recorded);
        assert!(receipt.capture_requirements.requirement_satisfied);
        assert_eq!(receipt.capture_requirements.status, "battery_mode_sample_recorded");
        assert!(receipt.capture_requirements.gaps.is_empty());
    }

    #[test]
    fn thermal_context_can_record_zone_visibility_without_temperatures() {
        let thermal = TelemetryThermalContext {
            source: "windows_perf_thermal_zone".to_string(),
            thermal_zones_visible: Some(1),
            temperatures_celsius: Vec::new(),
        };

        let formatted = format_thermal_context(&thermal);

        assert_eq!(
            formatted,
            "source=windows_perf_thermal_zone;thermal_zones_visible=1;temperatures_celsius=unavailable"
        );
    }

    #[test]
    fn thermal_zone_visibility_is_not_a_missing_thermal_context() {
        assert!(!thermal_context_is_unavailable(
            "source=windows_perf_thermal_zone;thermal_zones_visible=1;temperatures_celsius=unavailable"
        ));
        assert!(thermal_context_is_unavailable("thermal_context_unavailable"));
    }

    #[test]
    fn power_profile_evidence_indexes_low_power_blockers_without_claims() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_json(
            temp.path(),
            ROUTE_PROFILE_COMPARISON,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_route_profile_comparison",
                "profile_comparison_ready": true,
                "profiles": [
                    {
                        "profile_id": "low_power",
                        "route_evidence": [
                            {
                                "route_id": "dense_slm_openvino_npu_candidate",
                                "route_status": "candidate",
                                "ledger_route_status": "candidate",
                                "selected_backend": "openvino-npu",
                                "runtime_api": "openvino_genai",
                                "fallback_used": false,
                                "answer_gate_passed": true,
                                "benchmark_qualified_advantage": false,
                                "blockers": [
                                    "power advantage evidence missing for low_power promotion"
                                ]
                            }
                        ]
                    }
                ],
                "claim_boundary": {
                    "hidden_fallback_allowed": false
                }
            }),
        )?;
        write_json(
            temp.path(),
            COLD_WARM_PROFILE_BENCHMARK_FILE,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_cold_warm_profile_benchmark",
                "benchmark_gate_ready": true,
                "profiles": [
                    {
                        "profile_id": "low_power",
                        "routes": [
                            {
                                "route_id": "dense_slm_openvino_npu_candidate",
                                "timing": {
                                    "total_response_ms": 950.0,
                                    "throughput_tokens_per_s": 9.5
                                },
                                "blockers": [
                                    "power telemetry receipt does not provide low_power promotion evidence"
                                ]
                            }
                        ]
                    }
                ],
                "claim_boundary": {
                    "hidden_fallback_allowed": false
                }
            }),
        )?;
        write_json(
            temp.path(),
            POWER_THERMAL_CONTEXT_FILE,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_power_thermal_context",
                "availability": {
                    "memory_context_recorded": true,
                    "power_context_recorded": true,
                    "thermal_context_recorded": false
                },
                "power": {
                    "active_scheme": "Balanced",
                    "battery_status": "BatteryStatus=2;EstimatedChargeRemaining=100",
                    "ac_power_inferred": true
                },
                "thermal": {
                    "thermal_zones_visible": null,
                    "temperatures_celsius": []
                }
            }),
        )?;

        let receipt = build_power_profile_evidence_with_created_utc(
            temp.path(),
            Path::new(ROUTE_PROFILE_COMPARISON),
            Path::new(COLD_WARM_PROFILE_BENCHMARK_FILE),
            Path::new(POWER_THERMAL_CONTEXT_FILE),
            None,
            None,
            "2026-05-19T08:30:00Z".to_string(),
        )?;

        assert!(receipt.power_profile_index_ready, "{:?}", receipt.gaps);
        assert!(!receipt.low_power_promotion_ready);
        assert!(!receipt.power_advantage_proven);
        assert!(!receipt.claim_boundary.new_inference_executed);
        assert!(!receipt.claim_boundary.power_advantage_claim);
        assert!(
            receipt
                .gaps
                .iter()
                .any(|gap| gap.contains("AC-only; battery comparison evidence is missing"))
        );
        assert!(receipt.gaps.iter().any(|gap| gap.contains("energy proxy evidence is missing")));
        assert_eq!(receipt.operator_runbook.as_deref(), Some(LOW_POWER_BATTERY_RUNBOOK));
        assert!(
            receipt
                .next_required_evidence
                .iter()
                .any(|item| item.contains("telemetry-context --require-battery"))
        );
        let power_profile_path = temp.path().join("power-profile.json");
        fs::write(&power_profile_path, serde_json::to_vec_pretty(&receipt)?)?;
        let summary = inspect_power_profile_regression(&power_profile_path)?;
        assert!(summary.regression_ready, "{:?}", summary.gaps);
        assert_eq!(summary.operator_runbook.as_deref(), Some(LOW_POWER_BATTERY_RUNBOOK));
        assert!(
            summary
                .next_required_evidence
                .iter()
                .any(|item| item.contains("telemetry-context --require-battery"))
        );
        let npu = receipt
            .low_power_routes
            .iter()
            .find(|route| route.route_id == "dense_slm_openvino_npu_candidate")
            .context("missing NPU low_power route")?;
        assert_eq!(npu.total_response_ms, Some(950.0));
        assert!(!npu.power_promotion_ready);
        assert!(npu.power_related_blockers.iter().any(|blocker| blocker.contains("power")));
        Ok(())
    }

    #[test]
    fn power_profile_evidence_indexes_battery_and_energy_proxy_without_promotion() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        write_json(
            temp.path(),
            ROUTE_PROFILE_COMPARISON,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_route_profile_comparison",
                "profile_comparison_ready": true,
                "profiles": [
                    {
                        "profile_id": "low_power",
                        "route_evidence": [
                            {
                                "route_id": "dense_slm_openvino_npu_candidate",
                                "route_status": "candidate",
                                "ledger_route_status": "candidate",
                                "selected_backend": "openvino-npu",
                                "runtime_api": "openvino_genai",
                                "fallback_used": false,
                                "answer_gate_passed": true,
                                "benchmark_qualified_advantage": false,
                                "blockers": [
                                    "benchmark_qualified_speedup_or_power_advantage"
                                ]
                            }
                        ]
                    }
                ],
                "claim_boundary": {
                    "hidden_fallback_allowed": false
                }
            }),
        )?;
        write_json(
            temp.path(),
            COLD_WARM_PROFILE_BENCHMARK_FILE,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_cold_warm_profile_benchmark",
                "benchmark_gate_ready": true,
                "profiles": [
                    {
                        "profile_id": "low_power",
                        "routes": [
                            {
                                "route_id": "dense_slm_openvino_npu_candidate",
                                "timing": {
                                    "total_response_ms": 950.0,
                                    "throughput_tokens_per_s": 9.5
                                },
                                "blockers": [
                                    "benchmark_qualified_speedup_or_power_advantage"
                                ]
                            }
                        ]
                    }
                ],
                "claim_boundary": {
                    "hidden_fallback_allowed": false
                }
            }),
        )?;
        write_json(
            temp.path(),
            POWER_THERMAL_CONTEXT_FILE,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_power_thermal_context",
                "availability": {
                    "memory_context_recorded": true,
                    "power_context_recorded": true,
                    "thermal_context_recorded": true
                },
                "power": {
                    "active_scheme": "Balanced",
                    "battery_status": "BatteryStatus=2;EstimatedChargeRemaining=100",
                    "ac_power_inferred": true
                },
                "thermal": {
                    "thermal_zones_visible": 1,
                    "temperatures_celsius": []
                }
            }),
        )?;
        write_json(
            temp.path(),
            "battery-telemetry.json",
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_power_thermal_context",
                "availability": {
                    "memory_context_recorded": true,
                    "power_context_recorded": true,
                    "thermal_context_recorded": true
                },
                "power": {
                    "active_scheme": "Balanced",
                    "battery_status": "BatteryStatus=1;EstimatedChargeRemaining=96",
                    "ac_power_inferred": false
                },
                "thermal": {
                    "thermal_zones_visible": 1,
                    "temperatures_celsius": []
                }
            }),
        )?;
        write_json(
            temp.path(),
            "energy-proxy.json",
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_low_power_energy_proxy",
                "energy_proxy_recorded": true,
                "charge_delta_percent": -1.0,
                "sample_count": 10,
                "claim_boundary": {
                    "power_advantage_claim": false
                }
            }),
        )?;

        let receipt = build_power_profile_evidence_with_created_utc(
            temp.path(),
            Path::new(ROUTE_PROFILE_COMPARISON),
            Path::new(COLD_WARM_PROFILE_BENCHMARK_FILE),
            Path::new(POWER_THERMAL_CONTEXT_FILE),
            Some(Path::new("battery-telemetry.json")),
            Some(Path::new("energy-proxy.json")),
            "2026-05-19T10:15:00Z".to_string(),
        )?;

        assert!(receipt.power_profile_index_ready, "{:?}", receipt.gaps);
        assert!(receipt.telemetry.battery_mode_sample_recorded);
        assert_eq!(
            receipt.telemetry.battery_sample_source.as_deref(),
            Some("battery_telemetry_context")
        );
        assert!(receipt.telemetry.energy_proxy_recorded);
        assert_eq!(receipt.telemetry.energy_proxy_source.as_deref(), Some("energy_proxy_receipt"));
        assert!(!receipt.low_power_promotion_ready);
        assert!(!receipt.power_advantage_proven);
        assert!(!receipt.gaps.iter().any(|gap| gap.contains("battery-mode sample is missing")));
        assert!(!receipt.gaps.iter().any(|gap| gap.contains("energy proxy evidence is missing")));
        assert!(
            receipt.gaps.iter().any(
                |gap| gap.contains("no low_power route has benchmark-qualified power evidence")
            )
        );
        Ok(())
    }

    #[test]
    fn low_power_energy_proxy_indexes_battery_drain_without_claims() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_json(
            temp.path(),
            "before-telemetry.json",
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_power_thermal_context",
                "availability": {
                    "memory_context_recorded": true,
                    "power_context_recorded": true,
                    "thermal_context_recorded": true
                },
                "power": {
                    "active_scheme": "Balanced",
                    "battery_status": "BatteryStatus=1;EstimatedChargeRemaining=96",
                    "ac_power_inferred": false
                },
                "thermal": {
                    "thermal_zones_visible": 1,
                    "temperatures_celsius": []
                }
            }),
        )?;
        write_json(
            temp.path(),
            "after-telemetry.json",
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_power_thermal_context",
                "availability": {
                    "memory_context_recorded": true,
                    "power_context_recorded": true,
                    "thermal_context_recorded": true
                },
                "power": {
                    "active_scheme": "Balanced",
                    "battery_status": "BatteryStatus=1;EstimatedChargeRemaining=94",
                    "ac_power_inferred": false
                },
                "thermal": {
                    "thermal_zones_visible": 1,
                    "temperatures_celsius": []
                }
            }),
        )?;

        let receipt = build_low_power_energy_proxy_with_created_utc(
            temp.path(),
            Path::new("before-telemetry.json"),
            Path::new("after-telemetry.json"),
            "dense_slm_openvino_npu_candidate".to_string(),
            "low_power".to_string(),
            10,
            "2026-05-19T10:30:00Z".to_string(),
        )?;

        assert!(receipt.battery_mode_sample_recorded, "{:?}", receipt.gaps);
        assert!(receipt.energy_proxy_recorded, "{:?}", receipt.gaps);
        assert_eq!(receipt.before_charge_percent, Some(96));
        assert_eq!(receipt.after_charge_percent, Some(94));
        assert_eq!(receipt.charge_delta_percent, Some(-2));
        assert!(receipt.gaps.is_empty(), "{:?}", receipt.gaps);
        assert!(!receipt.claim_boundary.new_inference_executed);
        assert!(!receipt.claim_boundary.route_promotion_changed);
        assert!(!receipt.claim_boundary.power_advantage_claim);
        assert!(!receipt.claim_boundary.acceleration_claim);
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
                "tokens": {"prompt_count": 38, "generated_count": 8}
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
            None,
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
        write_route_model_identity_manifests(temp.path())?;
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
                "tokens": {"prompt_count": 38, "generated_count": 8}
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
        write_ready_bitnet_semantic_intake(temp.path(), BITNET_SEMANTIC_INTAKE)?;
        write_json(
            temp.path(),
            POWER_THERMAL_CONTEXT_FILE,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_power_thermal_context",
                "availability": {
                    "memory_context_recorded": true,
                    "power_context_recorded": true,
                    "thermal_context_recorded": false
                },
                "power": {
                    "active_scheme": "Balanced",
                    "battery_status": "BatteryStatus=2;EstimatedChargeRemaining=100",
                    "ac_power_inferred": true
                },
                "thermal": {
                    "thermal_zones_visible": null,
                    "temperatures_celsius": []
                }
            }),
        )?;
        let power_profile = build_power_profile_evidence_with_created_utc(
            temp.path(),
            Path::new(ROUTE_PROFILE_COMPARISON),
            Path::new("cold-warm.json"),
            Path::new(POWER_THERMAL_CONTEXT_FILE),
            None,
            None,
            "2026-05-14T23:50:00Z".to_string(),
        )?;
        fs::write(
            temp.path().join(POWER_PROFILE_EVIDENCE_FILE),
            serde_json::to_vec_pretty(&power_profile)?,
        )?;
        write_json(
            temp.path(),
            BLOCKED_AUTO_ASK_RECEIPT,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_operator_ask_blocked",
                "proof_stage": "operator_route_selection_blocked_no_inference",
                "machine_id": "intel-258v",
                "requested_device": "auto",
                "requested_route": "auto",
                "profile_id": "low_power",
                "selected_route": null,
                "selected_backend": null,
                "runtime_api": null,
                "model_path_required": false,
                "model_resolution": "not_required_for_blocked_auto_route_before_execution",
                "promotion_status": "no_promoted_route",
                "route_selection_status": "blocked",
                "route_selection_blocked": true,
                "route_selection_error": format!(
                    "no promoted Lunar Lake auto route for profile `low_power`; why_not_cpu=route is not promoted for profile `low_power`; why_not_gpu=route blocker for profile `low_power`: low_power_power_advantage_unproven; why_not_npu=missing evidence: benchmark_qualified_speedup_or_power_advantage; operator_runbook={LOW_POWER_BATTERY_RUNBOOK}"
                ),
                "candidate_routes": [
                    "dense_slm_default_cpu",
                    "dense_slm_openvino_gpu_candidate",
                    "dense_slm_openvino_npu_candidate"
                ],
                "why_not_cpu": ["route is not promoted for profile `low_power`"],
                "why_not_gpu": [
                    "route blocker for profile `low_power`: low_power_power_advantage_unproven"
                ],
                "why_not_npu": [
                    "missing evidence: benchmark_qualified_speedup_or_power_advantage"
                ],
                "operator_runbook": LOW_POWER_BATTERY_RUNBOOK,
                "next_required_evidence": blocked_operator_ask_next_required_evidence("low_power"),
                "route_selection": {
                    "requested_device": "auto",
                    "requested_route": "auto",
                    "profile_id": "low_power",
                    "selected_route": null,
                    "selected_backend": null,
                    "runtime_api": null,
                    "model_path_required": false,
                    "model_resolution": "not_required_for_blocked_auto_route_before_execution",
                    "promotion_status": "no_promoted_route",
                    "selection_source": "promotion_ledger_auto_blocked",
                    "route_selection_status": "blocked",
                    "route_selection_blocked": true,
                    "route_selection_error": format!(
                        "no promoted Lunar Lake auto route for profile `low_power`; why_not_cpu=route is not promoted for profile `low_power`; why_not_gpu=route blocker for profile `low_power`: low_power_power_advantage_unproven; why_not_npu=missing evidence: benchmark_qualified_speedup_or_power_advantage; operator_runbook={LOW_POWER_BATTERY_RUNBOOK}"
                    ),
                    "candidate_routes": [
                        "dense_slm_default_cpu",
                        "dense_slm_openvino_gpu_candidate",
                        "dense_slm_openvino_npu_candidate"
                    ],
                    "why_not_cpu": ["route is not promoted for profile `low_power`"],
                    "why_not_gpu": [
                        "route blocker for profile `low_power`: low_power_power_advantage_unproven"
                    ],
                    "why_not_npu": [
                        "missing evidence: benchmark_qualified_speedup_or_power_advantage"
                    ],
                    "operator_runbook": LOW_POWER_BATTERY_RUNBOOK,
                    "next_required_evidence": blocked_operator_ask_next_required_evidence("low_power")
                },
                "fallback_used": false,
                "new_inference_executed": false,
                "speedup_claim": false,
                "acceleration_claim": false,
                "power_advantage_claim": false,
                "bitnet_qk256_i2s_claim": false,
                "claim_boundary": {
                    "route_selection_blocked": true,
                    "new_inference_executed": false,
                    "fallback_used": false,
                    "model_loaded": false,
                    "route_promotion_changed": false,
                    "speedup_claim": false,
                    "power_advantage_claim": false,
                    "acceleration_claim": false,
                    "native_accelerator_claim": false,
                    "bitnet_qk256_i2s_claim": false
                }
            }),
        )?;

        let bundle = build_regression_bundle_with_created_utc_and_inputs_and_power_profile(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Some(Path::new("corpus-v2.yaml")),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            Some(Path::new("cold-warm.json")),
            Some(Path::new("durability.json")),
            Some(Path::new(BITNET_SEMANTIC_INTAKE)),
            Some(Path::new(POWER_PROFILE_EVIDENCE_FILE)),
            Some(Path::new(BLOCKED_AUTO_ASK_RECEIPT)),
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
        assert_eq!(corpus.case_count, 14);
        assert!(corpus.profiles.contains(&"prefill_heavy".to_string()));
        assert!(corpus.profiles.contains(&"warm_resident".to_string()));
        let Some(route_profiles) = bundle.route_profile_comparison.as_ref() else {
            bail!("missing route_profile_comparison summary");
        };
        assert!(route_profiles.candidate_routes_remain_unpromoted);
        assert!(!route_profiles.benchmark_qualified_advantage_claimed);
        assert!(
            route_profiles.timing_coverage.promotion_eligible_routes_have_profile_specific_timing
        );
        assert!(route_profiles.timing_coverage.proxy_or_missing_timing_routes_blocked);
        assert!(
            route_profiles.gpu_npu_promotion_blocker_summary.iter().any(|summary| {
                summary.blocker == "benchmark_qualified_speedup_or_power_advantage"
                    && summary.route_ids.contains(&"dense_slm_openvino_gpu_candidate".to_string())
                    && summary.route_ids.contains(&"dense_slm_openvino_npu_candidate".to_string())
                    && summary.next_action.contains("benchmark-qualified latency")
            }),
            "{:?}",
            route_profiles.gpu_npu_promotion_blocker_summary
        );
        assert!(bundle.regression_surface.strict_default);
        assert!(bundle.regression_surface.strict_ready, "{:?}", bundle.regression_surface.gaps);
        assert!(
            bundle
                .regression_surface
                .timing_coverage
                .promotion_eligible_routes_have_profile_specific_timing
        );
        assert!(bundle.regression_surface.answer_corpus_v2_indexed);
        assert!(bundle.regression_surface.route_profile_comparison_indexed);
        assert!(bundle.regression_surface.cold_warm_benchmark_indexed);
        assert!(bundle.regression_surface.cold_warm_benchmark_ready);
        assert!(bundle.regression_surface.durability_bundle_indexed);
        assert!(bundle.regression_surface.durability_stability_proven);
        assert!(bundle.regression_surface.bitnet_semantic_intake_indexed);
        assert!(bundle.regression_surface.bitnet_cpu_reference_evidence_indexed);
        assert!(bundle.regression_surface.bitnet_cpu_reference_evidence_ready);
        assert!(bundle.regression_surface.power_profile_evidence_indexed);
        assert!(bundle.regression_surface.arc_npu_bounded_evidence_indexed);
        assert!(bundle.regression_surface.arc_npu_bounded_evidence_ready);
        assert!(bundle.regression_surface.blocked_ask_receipt_indexed);
        assert!(!bundle.regression_surface.low_power_promotion_ready);
        assert!(!bundle.regression_surface.power_advantage_proven);
        let Some(cold_warm) = bundle.cold_warm_benchmark.as_ref() else {
            bail!("missing cold_warm_benchmark summary");
        };
        assert!(cold_warm.promoted_routes_have_critical_timing);
        assert!(cold_warm.candidate_routes_remain_unpromoted);
        assert!(cold_warm.timing_coverage.promotion_eligible_routes_have_profile_specific_timing);
        assert!(cold_warm.timing_coverage.proxy_or_missing_timing_routes_blocked);
        let Some(durability) = bundle.durability_bundle.as_ref() else {
            bail!("missing durability bundle summary");
        };
        assert!(durability.regression_ready, "{:?}", durability.gaps);
        assert!(durability.stability_proven);
        assert_eq!(durability.stable_profile_count, 3);
        let Some(intake) = bundle.bitnet_semantic_intake.as_ref() else {
            bail!("missing bitnet_semantic_intake summary");
        };
        assert!(intake.regression_ready, "{:?}", intake.gaps);
        assert!(!intake.rerun_required);
        assert_eq!(intake.pending_shared_change_count, 1);
        let Some(power) = bundle.power_profile_evidence.as_ref() else {
            bail!("missing power_profile_evidence summary");
        };
        assert!(power.regression_ready, "{:?}", power.gaps);
        assert!(power.power_profile_index_ready);
        assert!(power.low_power_routes_remain_unpromoted);
        assert!(power.current_context_is_ac_only);
        assert!(!power.battery_mode_sample_recorded);
        assert!(!power.energy_proxy_recorded);
        assert!(!power.thermal_context_recorded);
        assert!(
            power
                .blockers
                .iter()
                .any(|blocker| blocker.contains("battery comparison evidence is missing"))
        );
        let Some(blocked) = bundle.blocked_ask_receipt.as_ref() else {
            bail!("missing blocked_ask_receipt summary");
        };
        assert!(blocked.regression_ready, "{:?}", blocked.gaps);
        assert_eq!(blocked.profile_id, "low_power");
        assert!(blocked.route_selection_blocked);
        assert!(!blocked.new_inference_executed);
        assert!(!blocked.fallback_used);
        assert!(strict_regression_v2_gaps(&bundle).is_empty());
        Ok(())
    }

    #[test]
    fn thermal_temperature_availability_receipt_is_regression_ready_without_temperatures()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_json(
            temp.path(),
            THERMAL_TEMPERATURE_AVAILABILITY_FILE,
            json!({
                "schema_version": 1,
                "artifact_kind": "lunar_lake_thermal_temperature_availability",
                "proof_stage": "thermal_temperature_sources_probed_no_claim_change",
                "machine_id": "intel-258v",
                "decision": {
                    "thermal_zone_visibility_available": true,
                    "thermal_temperature_available": false,
                    "usable_temperature_reading_count": 0
                },
                "claim_boundary": {
                    "new_inference_executed": false,
                    "telemetry_probe_executed": true,
                    "measured_temperature_claim": false,
                    "route_promotion_changed": false,
                    "speedup_claim": false,
                    "power_advantage_claim": false,
                    "acceleration_claim": false,
                    "native_opencl_or_native_npu_claim": false,
                    "bitnet_qk256_or_i2s_behavior_changed": false
                }
            }),
        )?;

        let summary = inspect_thermal_temperature_availability_regression(
            &temp.path().join(THERMAL_TEMPERATURE_AVAILABILITY_FILE),
        )?;

        assert!(summary.regression_ready, "{:?}", summary.gaps);
        assert!(summary.thermal_zone_visibility_available);
        assert!(!summary.thermal_temperature_available);
        assert_eq!(summary.usable_temperature_reading_count, 0);
        assert!(!summary.measured_temperature_claim);
        assert!(summary.telemetry_probe_executed);
        assert!(summary.claim_boundary_preserved);
        let notes = thermal_temperature_availability_regression_notes(&summary);
        assert!(notes.iter().any(|note| note == "usable_temperature_reading_count=0"));
        Ok(())
    }

    #[test]
    fn thermal_temperature_availability_fails_false_measured_temperature_claim() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_json(
            temp.path(),
            THERMAL_TEMPERATURE_AVAILABILITY_FILE,
            json!({
                "schema_version": 1,
                "artifact_kind": "lunar_lake_thermal_temperature_availability",
                "proof_stage": "thermal_temperature_sources_probed_no_claim_change",
                "machine_id": "intel-258v",
                "decision": {
                    "thermal_zone_visibility_available": true,
                    "thermal_temperature_available": true,
                    "usable_temperature_reading_count": 0
                },
                "claim_boundary": {
                    "new_inference_executed": false,
                    "telemetry_probe_executed": true,
                    "measured_temperature_claim": true,
                    "route_promotion_changed": false,
                    "speedup_claim": false,
                    "power_advantage_claim": false,
                    "acceleration_claim": false,
                    "native_opencl_or_native_npu_claim": false,
                    "bitnet_qk256_or_i2s_behavior_changed": false
                }
            }),
        )?;

        let summary = inspect_thermal_temperature_availability_regression(
            &temp.path().join(THERMAL_TEMPERATURE_AVAILABILITY_FILE),
        )?;

        assert!(!summary.regression_ready);
        assert!(summary.gaps.iter().any(|gap| {
            gap.contains("claims temperature availability without usable readings")
        }));
        assert!(summary.gaps.iter().any(|gap| gap.contains("claim boundary")));
        Ok(())
    }

    #[test]
    fn warm_resident_auto_ask_receipt_is_regression_ready() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_warm_resident_auto_npu_ask(temp.path(), AUTO_NPU_WARM_RESIDENT_ASK_RECEIPT)?;

        let summary = inspect_operator_ask_regression(
            &temp.path().join(AUTO_NPU_WARM_RESIDENT_ASK_RECEIPT),
            OperatorAskRegressionExpectation {
                label: "warm_resident",
                profile_id: "warm_resident",
                selected_route: "dense_slm_openvino_npu_candidate",
                selected_backend: "openvino-npu",
            },
        )?;

        assert!(summary.regression_ready, "{:?}", summary.gaps);
        assert_eq!(summary.profile_id, "warm_resident");
        assert_eq!(summary.requested_device, "auto");
        assert_eq!(summary.requested_route, "auto");
        assert_eq!(summary.selected_route, "dense_slm_openvino_npu_candidate");
        assert_eq!(summary.selected_backend, "openvino-npu");
        assert!(summary.new_inference_executed);
        assert!(summary.generated_token_ids_available);
        assert!(!summary.speedup_claim);
        assert!(!summary.power_advantage_claim);
        assert!(!summary.acceleration_claim);
        assert!(!summary.bitnet_qk256_i2s_claim);
        Ok(())
    }

    #[test]
    fn ask_short_auto_gpu_ask_receipt_is_regression_ready() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ask_short_auto_gpu_ask(temp.path(), AUTO_GPU_ASK_SHORT_ASK_RECEIPT)?;

        let summary = inspect_operator_ask_regression(
            &temp.path().join(AUTO_GPU_ASK_SHORT_ASK_RECEIPT),
            OperatorAskRegressionExpectation {
                label: "ask_short",
                profile_id: "ask_short",
                selected_route: "dense_slm_openvino_gpu_candidate",
                selected_backend: "openvino-gpu",
            },
        )?;

        assert!(summary.regression_ready, "{:?}", summary.gaps);
        assert_eq!(summary.profile_id, "ask_short");
        assert_eq!(summary.requested_device, "auto");
        assert_eq!(summary.requested_route, "auto");
        assert_eq!(summary.selected_route, "dense_slm_openvino_gpu_candidate");
        assert_eq!(summary.selected_backend, "openvino-gpu");
        assert!(summary.new_inference_executed);
        assert!(summary.generated_token_ids_available);
        assert!(!summary.speedup_claim);
        assert!(!summary.power_advantage_claim);
        assert!(!summary.acceleration_claim);
        assert!(!summary.bitnet_qk256_i2s_claim);
        Ok(())
    }

    #[test]
    fn ask_normal_auto_gpu_ask_receipt_is_regression_ready() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ask_normal_auto_gpu_ask(temp.path(), AUTO_GPU_ASK_NORMAL_ASK_RECEIPT)?;

        let summary = inspect_operator_ask_regression(
            &temp.path().join(AUTO_GPU_ASK_NORMAL_ASK_RECEIPT),
            OperatorAskRegressionExpectation {
                label: "ask_normal",
                profile_id: "ask_normal",
                selected_route: "dense_slm_openvino_gpu_candidate",
                selected_backend: "openvino-gpu",
            },
        )?;

        assert!(summary.regression_ready, "{:?}", summary.gaps);
        assert_eq!(summary.profile_id, "ask_normal");
        assert_eq!(summary.requested_device, "auto");
        assert_eq!(summary.requested_route, "auto");
        assert_eq!(summary.selected_route, "dense_slm_openvino_gpu_candidate");
        assert_eq!(summary.selected_backend, "openvino-gpu");
        assert!(summary.new_inference_executed);
        assert!(summary.generated_token_ids_available);
        assert!(!summary.speedup_claim);
        assert!(!summary.power_advantage_claim);
        assert!(!summary.acceleration_claim);
        assert!(!summary.bitnet_qk256_i2s_claim);
        Ok(())
    }

    #[test]
    fn regression_surface_requires_auto_ask_when_gpu_ask_short_is_promoted() -> Result<()> {
        let mut route_profiles = ready_route_profile_regression_with_npu_warm_resident();
        let mut cold_warm = ready_cold_warm_regression_with_npu_warm_resident();
        for scope in
            [&mut route_profiles.route_promotion_scope, &mut cold_warm.route_promotion_scope]
        {
            scope.openvino_gpu_promoted_profiles = vec!["ask_short".to_string()];
            scope.openvino_npu_promoted_profiles.clear();
            scope.openvino_npu_remains_candidate = true;
            scope.notes = vec!["OpenVINO GPU is profile-promoted only for ask_short".to_string()];
        }
        let corpus = ready_answer_corpus_v2_summary();
        let durability = ready_durability_summary();
        let intake = ready_bitnet_semantic_intake_summary();
        let power = ready_power_profile_summary();
        let operator = ready_operator_receipt_with_arc_npu_bounded_evidence();

        let missing = build_regression_surface_summary(
            Some(&corpus),
            Some(&route_profiles),
            Some(&cold_warm),
            Some(&durability),
            Some(&intake),
            Some(&power),
            None,
            None,
            None,
            None,
            None,
            &operator,
        );
        assert!(!missing.strict_ready);
        assert!(missing.gaps.iter().any(|gap| {
            gap.contains("OpenVINO GPU is promoted for ask_short")
                && gap.contains("no successful auto ask receipt")
        }));

        let ask = ready_gpu_operator_ask_summary("ask_short", AUTO_GPU_ASK_SHORT_ASK_RECEIPT);
        let ready = build_regression_surface_summary(
            Some(&corpus),
            Some(&route_profiles),
            Some(&cold_warm),
            Some(&durability),
            Some(&intake),
            Some(&power),
            None,
            Some(&ask),
            None,
            None,
            None,
            &operator,
        );
        assert!(ready.ask_short_ask_receipt_indexed);
        assert!(ready.ask_short_auto_ask_ready);
        assert!(ready.strict_ready, "{:?}", ready.gaps);
        Ok(())
    }

    #[test]
    fn regression_surface_requires_auto_ask_when_npu_warm_resident_is_promoted() -> Result<()> {
        let route_profiles = ready_route_profile_regression_with_npu_warm_resident();
        let cold_warm = ready_cold_warm_regression_with_npu_warm_resident();
        let corpus = ready_answer_corpus_v2_summary();
        let durability = ready_durability_summary();
        let intake = ready_bitnet_semantic_intake_summary();
        let power = ready_power_profile_summary();
        let operator = ready_operator_receipt_with_arc_npu_bounded_evidence();

        let missing = build_regression_surface_summary(
            Some(&corpus),
            Some(&route_profiles),
            Some(&cold_warm),
            Some(&durability),
            Some(&intake),
            Some(&power),
            None,
            None,
            None,
            None,
            None,
            &operator,
        );
        assert!(!missing.strict_ready);
        assert!(missing.gaps.iter().any(|gap| {
            gap.contains("OpenVINO NPU is promoted for warm_resident")
                && gap.contains("no successful auto ask receipt")
        }));

        let ask = ready_operator_ask_summary();
        let ready = build_regression_surface_summary(
            Some(&corpus),
            Some(&route_profiles),
            Some(&cold_warm),
            Some(&durability),
            Some(&intake),
            Some(&power),
            None,
            None,
            None,
            Some(&ask),
            None,
            &operator,
        );
        assert!(ready.warm_resident_ask_receipt_indexed);
        assert!(ready.warm_resident_auto_ask_ready);
        assert!(ready.strict_ready, "{:?}", ready.gaps);
        Ok(())
    }

    #[test]
    fn regression_surface_requires_bitnet_cpu_reference_evidence() -> Result<()> {
        let route_profiles = ready_route_profile_regression_with_npu_warm_resident();
        let cold_warm = ready_cold_warm_regression_with_npu_warm_resident();
        let corpus = ready_answer_corpus_v2_summary();
        let durability = ready_durability_summary();
        let intake = ready_bitnet_semantic_intake_summary();
        let power = ready_power_profile_summary();
        let ask = ready_operator_ask_summary();
        let mut missing_operator = ready_operator_receipt_with_arc_npu_bounded_evidence();
        missing_operator.routes.retain(|route| route.route_id != "bitnet_reference_cpu");
        missing_operator.evidence.retain(|item| !item.evidence_id.starts_with("bitnet_"));

        let missing = build_regression_surface_summary(
            Some(&corpus),
            Some(&route_profiles),
            Some(&cold_warm),
            Some(&durability),
            Some(&intake),
            Some(&power),
            None,
            None,
            None,
            Some(&ask),
            None,
            &missing_operator,
        );
        assert!(!missing.strict_ready);
        assert!(!missing.bitnet_cpu_reference_evidence_indexed);
        assert!(!missing.bitnet_cpu_reference_evidence_ready);
        assert!(
            missing
                .gaps
                .iter()
                .any(|gap| { gap.contains("BitNet CPU reference route evidence is not indexed") })
        );

        let operator = ready_operator_receipt_with_arc_npu_bounded_evidence();
        let ready = build_regression_surface_summary(
            Some(&corpus),
            Some(&route_profiles),
            Some(&cold_warm),
            Some(&durability),
            Some(&intake),
            Some(&power),
            None,
            None,
            None,
            Some(&ask),
            None,
            &operator,
        );
        assert!(ready.strict_ready, "{:?}", ready.gaps);
        assert!(ready.bitnet_cpu_reference_evidence_indexed);
        assert!(ready.bitnet_cpu_reference_evidence_ready);
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
                "tokens": {"prompt_count": 38, "generated_count": 8}
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
        write_ready_bitnet_semantic_intake(temp.path(), BITNET_SEMANTIC_INTAKE)?;

        let bundle = build_regression_bundle_with_created_utc_and_inputs(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Some(Path::new("corpus-v2.yaml")),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            Some(Path::new("cold-warm.json")),
            None,
            Some(Path::new(BITNET_SEMANTIC_INTAKE)),
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
        assert!(receipt.quality_summary.failure_classes.contains_key("exact_answer_overgenerated"));
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
    fn ask_route_loads_explicit_openvino_candidate() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_receipts(temp.path(), false)?;
        let operator = build_operator_readiness_receipt_with_created_utc(
            temp.path(),
            "2026-05-13T15:36:09Z".to_string(),
        )?;
        fs::write(temp.path().join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;

        let route = load_operator_ask_route(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            "dense_slm_openvino_gpu_candidate",
        )?;

        assert_eq!(route.route_id, "dense_slm_openvino_gpu_candidate");
        assert_eq!(route.selected_backend, "openvino-gpu");
        assert_eq!(route.runtime_api, "openvino_genai");
        assert_eq!(route.selected_kernel_or_runtime, "openvino-genai-llmpipeline-gpu");
        assert!(!route.acceleration_claim);
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

    fn write_auto_ask_selection_artifacts(root: &Path) -> Result<LunarLakeRouteProfileComparison> {
        write_minimal_receipts(root, false)?;
        write_json(
            root,
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
                "tokens": {
                    "prompt_count": 38,
                    "generated_count": 8
                }
            }),
        )?;
        write_json(
            root,
            DENSE_PHASE_COMPARISON,
            json!({
                "artifact_kind": "intel_258v_dense_slm_openvino_phase_comparison",
                "fallback_used": false,
                "gguf_cpu_reference": {"timing": {"prefill_512": {}, "decode_128": {}}}
            }),
        )?;
        let operator = build_operator_readiness_receipt_with_created_utc(
            root,
            "2026-05-16T10:00:00Z".to_string(),
        )?;
        fs::write(root.join(OPERATOR_READINESS), serde_json::to_vec_pretty(&operator)?)?;
        let regression = build_regression_bundle_with_created_utc(
            root,
            Path::new(OPERATOR_READINESS),
            "2026-05-16T10:05:00Z".to_string(),
        )?;
        fs::write(root.join(REGRESSION_BUNDLE), serde_json::to_vec_pretty(&regression)?)?;
        let comparison = build_comparison_receipt_with_created_utc(
            root,
            Path::new(OPERATOR_READINESS),
            Path::new(REGRESSION_BUNDLE),
            "2026-05-16T10:10:00Z".to_string(),
        )?;
        fs::write(root.join(OPERATOR_COMPARISON), serde_json::to_vec_pretty(&comparison)?)?;
        let ledger = build_route_promotion_ledger_with_created_utc(
            root,
            Path::new(OPERATOR_READINESS),
            Path::new(OPERATOR_COMPARISON),
            "2026-05-16T10:15:00Z".to_string(),
        )?;
        fs::write(root.join(ROUTE_PROMOTION_LEDGER), serde_json::to_vec_pretty(&ledger)?)?;
        let profiles = build_route_profile_comparison_with_created_utc(
            root,
            Path::new(ROUTE_PROMOTION_LEDGER),
            Path::new(DENSE_PHASE_COMPARISON),
            "2026-05-16T10:20:00Z".to_string(),
        )?;
        fs::write(root.join(ROUTE_PROFILE_COMPARISON), serde_json::to_vec_pretty(&profiles)?)?;
        Ok(profiles)
    }

    fn block_regression_tiny_cpu_profile(
        profiles: &mut LunarLakeRouteProfileComparison,
    ) -> Result<()> {
        let profile = profiles
            .profiles
            .iter_mut()
            .find(|profile| profile.profile_id == "regression_tiny")
            .context("missing regression_tiny profile")?;
        profile.profile_status = "promoted_route_blocked".to_string();
        profile.promotion_decision =
            "dense_slm_default_cpu is listed as promoted for regression_tiny, but profile evidence is incomplete"
                .to_string();
        let route = profile
            .route_evidence
            .iter_mut()
            .find(|route| route.route_id == DEFAULT_ASK_ROUTE)
            .context("missing regression_tiny CPU route")?;
        route.promotion_eligible_for_profile = false;
        route.blockers.push("corpus_v2 profile regression_tiny has 1 quality failures".to_string());
        Ok(())
    }

    #[test]
    fn auto_ask_selects_promoted_cpu_route_from_ledger() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_auto_ask_selection_artifacts(temp.path())?;

        let selection = resolve_operator_ask_route_selection(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(ROUTE_PROMOTION_LEDGER),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            "auto",
            "auto",
            "ask_normal",
        )?;

        assert_eq!(selection.selection_source, "promotion_ledger_auto");
        assert_eq!(selection.selected_route, DEFAULT_ASK_ROUTE);
        assert_eq!(selection.promotion_status, "promoted");
        assert_eq!(selection.route_profile_status.as_deref(), Some("promoted_route_ready"));
        assert!(selection.route_profile_blockers.is_empty());
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
    fn auto_ask_rejects_route_profile_blocked_promoted_route() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut profiles = write_auto_ask_selection_artifacts(temp.path())?;
        block_regression_tiny_cpu_profile(&mut profiles)?;
        fs::write(
            temp.path().join(ROUTE_PROFILE_COMPARISON),
            serde_json::to_vec_pretty(&profiles)?,
        )?;

        let err = resolve_operator_ask_route_selection(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(ROUTE_PROMOTION_LEDGER),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            "auto",
            "auto",
            "regression_tiny",
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("blocked by route-profile comparison"), "got: {err}");
        assert!(
            err.contains("corpus_v2 profile regression_tiny has 1 quality failures"),
            "got: {err}"
        );
        Ok(())
    }

    #[test]
    fn direct_ask_rejects_route_profile_blocked_promoted_route() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut profiles = write_auto_ask_selection_artifacts(temp.path())?;
        block_regression_tiny_cpu_profile(&mut profiles)?;
        fs::write(
            temp.path().join(ROUTE_PROFILE_COMPARISON),
            serde_json::to_vec_pretty(&profiles)?,
        )?;

        let err = resolve_operator_ask_route_selection(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(ROUTE_PROMOTION_LEDGER),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            DEFAULT_ASK_ROUTE,
            "cpu",
            "regression_tiny",
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("blocked by route-profile comparison"), "got: {err}");
        assert!(err.contains("profile `regression_tiny`"), "got: {err}");
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
            None,
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
    fn auto_ask_no_promoted_profile_reports_route_blockers() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_auto_ask_selection_artifacts(temp.path())?;

        let err = resolve_operator_ask_route_selection(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(ROUTE_PROMOTION_LEDGER),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            "auto",
            "auto",
            "low_power",
        )
        .unwrap_err()
        .to_string();

        assert!(
            err.contains("no promoted Lunar Lake auto route for profile `low_power`"),
            "got: {err}"
        );
        assert!(err.contains("why_not_cpu="), "got: {err}");
        assert!(err.contains("why_not_gpu="), "got: {err}");
        assert!(err.contains("why_not_npu="), "got: {err}");
        assert!(err.contains("low_power_power_advantage_unproven"), "got: {err}");
        assert!(err.contains("battery-mode or energy-proxy power advantage"), "got: {err}");
        assert!(err.contains("auto routing only selects routes explicitly promoted"), "got: {err}");
        assert!(err.contains("benchmark_qualified_speedup_or_power_advantage"), "got: {err}");
        assert!(err.contains("benchmark-qualified latency or power advantage"), "got: {err}");
        assert!(err.contains("telemetry-context --require-battery"), "got: {err}");
        assert!(err.contains(LOW_POWER_BATTERY_RUNBOOK), "got: {err}");

        let blocked = explain_blocked_operator_ask_route_selection(
            temp.path(),
            Path::new(ROUTE_PROMOTION_LEDGER),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            "auto",
            "auto",
            "low_power",
        )?
        .context("missing blocked auto-route explanation")?;
        assert_eq!(blocked.route_selection_status, "blocked");
        assert_eq!(blocked.promotion_status, "no_promoted_route");
        assert_eq!(blocked.selection_source, "promotion_ledger_auto_blocked");
        assert!(blocked.candidate_routes.contains(&DEFAULT_ASK_ROUTE.to_string()));
        assert!(
            blocked
                .why_not_cpu
                .iter()
                .any(|reason| { reason.contains("route is not promoted for profile `low_power`") })
        );
        assert!(blocked.why_not_gpu.iter().any(|reason| {
            reason.contains("route is not promoted for profile `low_power`")
                || reason.contains("low_power")
        }));
        assert!(
            blocked
                .why_not_npu
                .iter()
                .any(|reason| { reason.contains("low_power_power_advantage_unproven") })
        );
        assert!(blocked.why_not_npu.iter().any(|reason| {
            reason.contains("auto_default")
                && reason.contains("auto routing only selects routes explicitly promoted")
        }));
        assert_eq!(blocked.operator_runbook.as_deref(), Some(LOW_POWER_BATTERY_RUNBOOK));
        assert!(
            blocked
                .next_required_evidence
                .iter()
                .any(|item| item.contains("telemetry-context --require-battery"))
        );
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
            None,
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

    #[test]
    fn direct_openvino_ask_route_records_profile_blockers_without_promoting() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_auto_ask_selection_artifacts(temp.path())?;

        let selection = resolve_operator_ask_route_selection(
            temp.path(),
            Path::new(OPERATOR_READINESS),
            Path::new(ROUTE_PROMOTION_LEDGER),
            Some(Path::new(ROUTE_PROFILE_COMPARISON)),
            "dense_slm_openvino_gpu_candidate",
            "GPU.0",
            "ask_normal",
        )?;

        assert_eq!(selection.selection_source, "operator_receipt_direct");
        assert_eq!(selection.selected_route, "dense_slm_openvino_gpu_candidate");
        assert_eq!(selection.selected_backend, "openvino-gpu");
        assert_eq!(selection.runtime_api, "openvino_genai");
        assert_eq!(selection.promotion_status, "direct_route_validated");
        assert_eq!(selection.route_profile_status.as_deref(), Some("promoted_route_ready"));
        assert!(selection.route_profile_blockers.iter().any(|blocker| {
            blocker.contains("route not promoted for profile")
                || blocker.contains("candidate route requires benchmark-qualified profile evidence")
        }));
        Ok(())
    }

    #[test]
    fn npu_cold_start_diagnosis_classifies_load_dominated_startup() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_json(
            temp.path(),
            DENSE_OV_PHASE,
            json!({
                "artifact_kind": "intel_258v_dense_slm_openvino_phase_runner",
                "fallback_used": false,
                "generation": {
                    "devices": [{
                        "runtime_device": "NPU",
                        "selected_backend": "openvino-npu",
                        "runtime_api": "openvino_genai",
                        "fallback_used": false,
                        "pipeline_construct_wall_ms": 37652.0,
                        "passed": 3,
                        "failed": 0,
                        "cases": [{
                            "generation_wall_ms": 1000.0,
                            "first_streamed_text_chunk_ms": 300.0,
                            "openvino_perf_metrics": {
                                "load_time_ms": 37651.0,
                                "time_to_first_token": {"mean_ms": 320.0},
                                "generate": {"mean_ms": 990.0},
                                "inference": {"mean_ms": 980.0},
                                "tokenization": {"mean_ms": 15.0},
                                "throughput": {"mean_ms": 12.5},
                                "num_generated_tokens": 9
                            }
                        }]
                    }]
                }
            }),
        )?;
        write_json(
            temp.path(),
            DENSE_PHASE_COMPARISON,
            json!({
                "artifact_kind": "intel_258v_dense_slm_openvino_phase_comparison",
                "openvino_paths": {
                    "npu": {
                        "timing": {
                            "pipeline_load_ms": 35469.0,
                            "case_elapsed_ms_sum": 2116.0
                        }
                    }
                }
            }),
        )?;
        write_json(
            temp.path(),
            DENSE_OV_NPU_OPERATOR_ASK,
            json!({
                "artifact_kind": "lunar_lake_openvino_operator_ask",
                "route_id": "dense_slm_openvino_npu_candidate",
                "requested_backend": "openvino-npu",
                "selected_backend": "openvino-npu",
                "runtime_api": "openvino_genai",
                "runtime_device": "NPU",
                "resolved_device": "Intel(R) AI Boost",
                "backend_lane": "dense_slm_openvino_npu",
                "selected_kernel_or_runtime": "openvino-genai-llmpipeline-npu",
                "fallback_used": false,
                "route": {"acceleration_claim": false},
                "answer_gate": {"passed": true},
                "timing": {
                    "pipeline_construct_wall_ms": 40312.0,
                    "generation_wall_ms": 943.0,
                    "openvino_perf_metrics": {
                        "load_time_ms": 40263.0,
                        "time_to_first_token": {"mean_ms": 323.0},
                        "generate": {"mean_ms": 942.0},
                        "inference": {"mean_ms": 919.0},
                        "tokenization": {"mean_ms": 17.0},
                        "throughput": {"mean_ms": 12.9},
                        "num_generated_tokens": 9
                    }
                }
            }),
        )?;
        write_json(
            temp.path(),
            DENSE_OV_CORPUS_V2,
            json!({
                "artifact_kind": "intel_258v_dense_slm_openvino_corpus_v2",
                "fallback_used": false,
                "generation": {
                    "devices": [{
                        "runtime_device": "NPU",
                        "selected_backend": "openvino-npu",
                        "runtime_api": "openvino_genai",
                        "fallback_used": false,
                        "promotion_status": "candidate",
                        "route_promotion_changed": false,
                        "pipeline_construct_wall_ms": 32127.0,
                        "quality_summary": {
                            "cases_total": 12,
                            "passed": 7,
                            "failed": 5,
                            "profile_summary": {
                                "ask_short": {"total": 2, "passed": 1, "failed": 1}
                            },
                            "category_summary": {
                                "yes_no": {"total": 1, "passed": 0, "failed": 1}
                            }
                        },
                        "cases": [{
                            "generated_token_ids_available_from_pipeline": false,
                            "generated_token_ids_source": "retokenized_generated_text_not_pipeline_internal_ids",
                            "timing": {
                                "generation_wall_ms": 876.0,
                                "first_streamed_text_chunk_ms": 680.0,
                                "openvino_perf_metrics": {
                                    "load_time_ms": 32127.0,
                                    "time_to_first_token": {"mean_ms": 406.0},
                                    "generate": {"mean_ms": 874.0},
                                    "inference": {"mean_ms": 857.0},
                                    "tokenization": {"mean_ms": 13.0},
                                    "throughput": {"mean_ms": 14.9},
                                    "num_generated_tokens": 8
                                }
                            }
                        }]
                    }]
                }
            }),
        )?;

        let receipt = build_npu_cold_start_diagnosis_with_created_utc(
            temp.path(),
            Path::new(DENSE_OV_PHASE),
            Path::new(DENSE_PHASE_COMPARISON),
            Path::new(DENSE_OV_NPU_OPERATOR_ASK),
            Path::new(DENSE_OV_CORPUS_V2),
            "2026-05-17T10:00:00Z".to_string(),
        )?;

        assert!(receipt.diagnosis_ready, "{:?}", receipt.gaps);
        assert!(receipt.cold_start.cold_load_dominant);
        assert_eq!(
            receipt.cold_start.classification,
            "openvino_pipeline_load_or_device_compile_dominated"
        );
        assert!(receipt.cold_start.operator_load_to_generation_ratio.unwrap() > 40.0);
        assert!(receipt.hot_path.hot_path_interesting);
        assert!(receipt.corpus_v2_context.route_blocked_by_quality);
        assert!(!receipt.claim_boundary.route_promotion_changed);
        assert!(!receipt.claim_boundary.speedup_claim);
        assert!(!receipt.claim_boundary.acceleration_claim);
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
        if !fallback {
            write_route_model_identity_manifests(root)?;
        }
        Ok(())
    }

    fn write_route_model_identity_manifests(root: &Path) -> Result<()> {
        write_json(
            root,
            DENSE_SLM_ARTIFACT_MANIFEST,
            json!({
                "selected_candidate": {
                    "model_name": "Qwen2.5-0.5B-Instruct",
                    "family": "qwen2.5",
                    "format": "GGUF",
                    "file": "Qwen2.5-0.5B-Instruct-Q8_0.gguf",
                    "sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
                    "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                    "repo_revision": "main",
                    "quantization": "Q8_0"
                },
                "tokenizer": {
                    "source": "Qwen/Qwen2.5-0.5B-Instruct",
                    "pretokenizer": "qwen2",
                    "prompt_template": "qwen2.5-instruct-chatml",
                    "stop_token_policy": "eos_or_stop_sequence"
                }
            }),
        )?;
        write_json(
            root,
            DENSE_SLM_OPENVINO_IR_MANIFEST,
            json!({
                "source_model": {
                    "model_name": "Qwen2.5-0.5B-Instruct",
                    "model_family": "qwen2.5",
                    "repo": "Qwen/Qwen2.5-0.5B-Instruct",
                    "revision": "main"
                },
                "export_contract": {
                    "format": "OpenVINO IR",
                    "expected_output_dir": "openvino/qwen25-0.5b-int4-sym",
                    "weight_format": "INT4",
                    "symmetric": true
                },
                "tokenizer": {
                    "source": "Qwen/Qwen2.5-0.5B-Instruct",
                    "tokenizer_family": "qwen2",
                    "prompt_template": "qwen2.5-instruct-chatml",
                    "stop_token_policy": "eos_or_stop_sequence"
                }
            }),
        )?;
        write_json(
            root,
            BITNET_CPU_BUNDLE,
            json!({
                "artifact_kind": "intel_258v_cpu_reference_bundle",
                "captured_at_utc": "2026-05-12T18:43:14Z",
                "fallback_used": false,
                "model": {
                    "file": "bitnet-b1.58-2B-4T.I2_S.gguf",
                    "architecture": "bitnet",
                    "format": "GGUF",
                    "sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
                    "repo": "microsoft/bitnet-b1.58-2B-4T"
                },
                "tokenizer": {
                    "source": "microsoft/bitnet-b1.58-2B-4T",
                    "type": "sentencepiece"
                },
                "cpu_reference": {
                    "fallback_used": false,
                    "prompt_policy": "bitnet_strict_reference_prompt"
                }
            }),
        )?;
        Ok(())
    }

    fn write_warm_resident_auto_npu_ask(root: &Path, file: &str) -> Result<()> {
        write_json(
            root,
            file,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_operator_ask",
                "proof_stage": "operator_candidate_route_executed_through_lunar_lake_ask",
                "machine_id": "intel-258v",
                "requested_device": "auto",
                "requested_route": "auto",
                "profile_id": "warm_resident",
                "selected_route": "dense_slm_openvino_npu_candidate",
                "selected_backend": "openvino-npu",
                "runtime_api": "openvino_genai",
                "route_id": "dense_slm_openvino_npu_candidate",
                "promotion_status": "promoted",
                "route_profile_status": "promoted_route_ready",
                "route_profile_blockers": [],
                "fallback_used": false,
                "answer_gate_passed": true,
                "openvino_candidate_route_executed": true,
                "speedup_claim": false,
                "power_advantage_claim": false,
                "acceleration_claim": false,
                "bitnet_qk256_i2s_claim": false,
                "tokens": {
                    "prompt_count": 39,
                    "generated_count": 9,
                    "generated_ids": [17, 488, 220, 17, 16819, 220, 19, 13, 151645]
                },
                "route_selection": {
                    "requested_device": "auto",
                    "requested_route": "auto",
                    "profile_id": "warm_resident",
                    "selected_route": "dense_slm_openvino_npu_candidate",
                    "selected_backend": "openvino-npu",
                    "runtime_api": "openvino_genai",
                    "promotion_status": "promoted",
                    "route_profile_status": "promoted_route_ready",
                    "route_profile_blockers": [],
                    "selection_source": "promotion_ledger_auto"
                },
                "source_run_receipt": "source-run.json",
                "source_receipt": {
                    "artifact_kind": "lunar_lake_openvino_operator_ask",
                    "output": {
                        "generated_token_ids_available_from_pipeline": true,
                        "generated_token_ids_source": "openvino_genai_encoded_results_tokens",
                        "generated_token_ids": [17, 488, 220, 17, 16819, 220, 19, 13, 151645]
                    },
                    "verification": {
                        "answer_gate_passed": true,
                        "fallback_used": false,
                        "generated_token_ids_available_from_pipeline": true,
                        "acceleration_claim": false
                    }
                },
                "claim_boundary": {
                    "openvino_candidate_route_executed": true,
                    "fallback_used": false,
                    "speedup_claim": false,
                    "power_advantage_claim": false,
                    "acceleration_claim": false,
                    "arc_or_npu_acceleration_claim": false,
                    "bitnet_qk256_i2s_claim": false
                }
            }),
        )
    }

    fn write_ask_short_auto_gpu_ask(root: &Path, file: &str) -> Result<()> {
        write_auto_gpu_ask(root, file, "ask_short")
    }

    fn write_ask_normal_auto_gpu_ask(root: &Path, file: &str) -> Result<()> {
        write_auto_gpu_ask(root, file, "ask_normal")
    }

    fn write_auto_gpu_ask(root: &Path, file: &str, profile_id: &str) -> Result<()> {
        write_json(
            root,
            file,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_operator_ask",
                "proof_stage": "operator_candidate_route_executed_through_lunar_lake_ask",
                "machine_id": "intel-258v",
                "requested_device": "auto",
                "requested_route": "auto",
                "profile_id": profile_id,
                "selected_route": "dense_slm_openvino_gpu_candidate",
                "selected_backend": "openvino-gpu",
                "runtime_api": "openvino_genai",
                "route_id": "dense_slm_openvino_gpu_candidate",
                "promotion_status": "promoted",
                "route_profile_status": "promoted_route_ready",
                "route_profile_blockers": [],
                "fallback_used": false,
                "answer_gate_passed": true,
                "openvino_candidate_route_executed": true,
                "speedup_claim": false,
                "power_advantage_claim": false,
                "acceleration_claim": false,
                "bitnet_qk256_i2s_claim": false,
                "tokens": {
                    "prompt_count": 39,
                    "generated_count": 8,
                    "generated_ids": [17, 488, 220, 17, 16819, 220, 19, 13]
                },
                "route_selection": {
                    "requested_device": "auto",
                    "requested_route": "auto",
                    "profile_id": profile_id,
                    "selected_route": "dense_slm_openvino_gpu_candidate",
                    "selected_backend": "openvino-gpu",
                    "runtime_api": "openvino_genai",
                    "promotion_status": "promoted",
                    "route_profile_status": "promoted_route_ready",
                    "route_profile_blockers": [],
                    "selection_source": "promotion_ledger_auto"
                },
                "source_run_receipt": "source-run.json",
                "source_receipt": {
                    "artifact_kind": "lunar_lake_openvino_operator_ask",
                    "output": {
                        "generated_token_ids_available_from_pipeline": true,
                        "generated_token_ids_source": "openvino_genai_encoded_results_tokens",
                        "generated_token_ids": [17, 488, 220, 17, 16819, 220, 19, 13]
                    },
                    "verification": {
                        "answer_gate_passed": true,
                        "fallback_used": false,
                        "generated_token_ids_available_from_pipeline": true,
                        "acceleration_claim": false
                    }
                },
                "claim_boundary": {
                    "openvino_candidate_route_executed": true,
                    "fallback_used": false,
                    "speedup_claim": false,
                    "power_advantage_claim": false,
                    "acceleration_claim": false,
                    "arc_or_npu_acceleration_claim": false,
                    "bitnet_qk256_i2s_claim": false
                }
            }),
        )
    }

    fn ready_timing_coverage() -> TimingApplicabilityCoverageSummary {
        TimingApplicabilityCoverageSummary {
            route_count: 2,
            profile_specific_route_count: 2,
            proxy_or_missing_route_count: 0,
            promotion_eligible_route_count: 1,
            promotion_eligible_profile_specific_route_count: 1,
            candidate_route_count: 1,
            candidate_proxy_or_missing_route_count: 0,
            promotion_eligible_routes_have_profile_specific_timing: true,
            proxy_or_missing_timing_routes_blocked: true,
            proxy_or_missing_routes: vec![],
            promotion_eligible_proxy_or_missing_routes: vec![],
            unblocked_proxy_or_missing_routes: vec![],
        }
    }

    fn ready_route_model_identity_coverage() -> RouteModelIdentityCoverage {
        RouteModelIdentityCoverage {
            route_row_count: 2,
            route_rows_with_identity: 2,
            route_rows_with_model_hash: 1,
            route_rows_with_tokenizer_template: 2,
            route_rows_without_model_hash_with_known_gap: 1,
            all_route_rows_have_identity: true,
            all_route_rows_have_tokenizer_template: true,
            model_hash_or_explicit_gap_for_all_route_rows: true,
            routes_without_model_hash: vec![
                "warm_resident:dense_slm_openvino_npu_candidate".to_string(),
            ],
            routes_without_model_hash_missing_known_gap: vec![],
            routes_without_tokenizer_template: vec![],
        }
    }

    fn npu_warm_route_promotion_scope() -> RoutePromotionScopeSummary {
        RoutePromotionScopeSummary {
            openvino_gpu_promoted_profiles: vec![],
            openvino_npu_promoted_profiles: vec!["warm_resident".to_string()],
            profile_scoped_promotion_only: true,
            openvino_npu_remains_candidate: false,
            unexpected_openvino_profile_promotions: vec![],
            notes: vec!["OpenVINO NPU is profile-promoted only for warm_resident".to_string()],
        }
    }

    fn ready_route_profile_regression_with_npu_warm_resident() -> RouteProfileRegressionSummary {
        RouteProfileRegressionSummary {
            path: "route-profile.json".to_string(),
            profile_comparison_ready: true,
            default_route_id: DEFAULT_ASK_ROUTE.to_string(),
            profiles: REQUIRED_ROUTE_PROFILES
                .iter()
                .map(|profile| (*profile).to_string())
                .collect(),
            timing_coverage: ready_timing_coverage(),
            route_model_identity_coverage: ready_route_model_identity_coverage(),
            route_model_identity_ready: true,
            candidate_routes_remain_unpromoted: true,
            benchmark_qualified_advantage_claimed: false,
            fallback_observed: false,
            gpu_npu_promotion_blockers: vec!["low_power_power_advantage_unproven".to_string()],
            gpu_npu_promotion_blocker_summary: vec![],
            route_promotion_scope: npu_warm_route_promotion_scope(),
            regression_ready: true,
            gaps: vec![],
        }
    }

    fn ready_cold_warm_regression_with_npu_warm_resident() -> ColdWarmRegressionSummary {
        ColdWarmRegressionSummary {
            path: "cold-warm.json".to_string(),
            benchmark_gate_ready: true,
            profiles: REQUIRED_ROUTE_PROFILES
                .iter()
                .map(|profile| (*profile).to_string())
                .collect(),
            timing_coverage: ready_timing_coverage(),
            route_model_identity_coverage: ready_route_model_identity_coverage(),
            route_model_identity_ready: true,
            promoted_routes_have_critical_timing: true,
            candidate_routes_remain_unpromoted: true,
            fallback_observed: false,
            benchmark_qualified_advantage_claimed: false,
            telemetry_gaps: vec![],
            route_promotion_scope: npu_warm_route_promotion_scope(),
            regression_ready: true,
            gaps: vec![],
        }
    }

    fn ready_answer_corpus_v2_summary() -> AnswerCorpusV2Summary {
        AnswerCorpusV2Summary {
            path: "corpus-v2.yaml".to_string(),
            schema: 1,
            name: "lunar-lake-qwen25-answer-corpus-v2".to_string(),
            route_scope: Some(DEFAULT_ASK_ROUTE.to_string()),
            model_family: Some("qwen".to_string()),
            model_architecture: Some("qwen2".to_string()),
            quantization: Some("Q8_0".to_string()),
            prompt_template: Some("qwen2.5".to_string()),
            case_count: 14,
            profiles: REQUIRED_CORPUS_V2_PROFILES
                .iter()
                .map(|profile| (*profile).to_string())
                .collect(),
            categories: REQUIRED_CORPUS_V2_CATEGORIES
                .iter()
                .map(|category| (*category).to_string())
                .collect(),
            claim_boundary_preserved: true,
            fixture_ready: true,
            gaps: vec![],
        }
    }

    fn ready_durability_summary() -> DurabilityRegressionSummary {
        DurabilityRegressionSummary {
            path: "durability.json".to_string(),
            durability_index_ready: true,
            stability_proven: true,
            profiles: default_durability_required_profiles(),
            required_repeat_count: 10,
            stable_profile_count: 3,
            fallback_observed: false,
            answer_drift_detected: false,
            route_drift_detected: false,
            repeated_run_stability_claim: true,
            regression_ready: true,
            gaps: vec![],
        }
    }

    fn ready_bitnet_semantic_intake_summary() -> BitnetSemanticIntakeRegressionSummary {
        BitnetSemanticIntakeRegressionSummary {
            path: BITNET_SEMANTIC_INTAKE.to_string(),
            intake_ready: true,
            rerun_required: false,
            pending_shared_change_count: 1,
            closed_shared_change_count: 0,
            merged_to_main_count: 0,
            stale_after_merged_count: 0,
            source_lanes: vec!["a770".to_string()],
            pending_changes: vec!["shared BitNet semantic fix pending".to_string()],
            closed_changes: vec![],
            required_reruns: vec![],
            claim_boundary_preserved: true,
            regression_ready: true,
            gaps: vec![],
        }
    }

    fn ready_power_profile_summary() -> PowerProfileRegressionSummary {
        PowerProfileRegressionSummary {
            path: POWER_PROFILE_EVIDENCE_FILE.to_string(),
            power_profile_index_ready: true,
            low_power_promotion_ready: false,
            power_advantage_proven: false,
            low_power_route_count: 3,
            low_power_routes_remain_unpromoted: true,
            current_context_is_ac_only: true,
            battery_mode_sample_recorded: false,
            battery_sample_source: None,
            energy_proxy_recorded: false,
            energy_proxy_source: None,
            thermal_context_recorded: false,
            operator_runbook: Some(LOW_POWER_BATTERY_RUNBOOK.to_string()),
            next_required_evidence: blocked_operator_ask_next_required_evidence("low_power"),
            claim_boundary_preserved: true,
            regression_ready: true,
            gaps: vec![],
            blockers: vec!["battery comparison evidence is missing".to_string()],
        }
    }

    fn ready_operator_receipt_with_arc_npu_bounded_evidence() -> LunarLakeOperatorReceipt {
        LunarLakeOperatorReceipt {
            schema_version: "1.0.0".to_string(),
            artifact_kind: "lunar_lake_operator_readiness".to_string(),
            proof_stage: "test_ready".to_string(),
            created_utc: "2026-05-20T00:00:00Z".to_string(),
            machine_id: "intel-258v".to_string(),
            artifact_root: DEFAULT_ARTIFACT_ROOT.to_string(),
            operator_ready: true,
            default_route: dense_slm_cpu_route(),
            routes: vec![dense_slm_cpu_route(), bitnet_cpu_route()],
            route_policy: None,
            power_profile_evidence: None,
            thermal_temperature_availability: None,
            blocked_ask_receipt: None,
            evidence: vec![
                ready_bitnet_cpu_reference_evidence("bitnet_cpu_reference_bundle"),
                ready_bitnet_cpu_reference_evidence("bitnet_external_reference_boundary"),
                ready_bitnet_cpu_reference_evidence("bitnet_external_direct_token_boundary"),
                ready_bitnet_cpu_reference_evidence("bitnet_first_token_direct_classifier"),
                ready_bitnet_cpu_reference_evidence("bitnet_i2s_gemv_gemm_microbench"),
                ready_bitnet_cpu_reference_evidence("bitnet_i2s_tiling_thread_matrix"),
                ready_bitnet_cpu_reference_evidence("bitnet_i2s_applied_thread_matrix"),
                ready_bitnet_cpu_reference_evidence("bitnet_embedding_quantization_evidence"),
                ready_arc_npu_bounded_evidence("arc140v_native_opencl_parity"),
                ready_arc_npu_bounded_evidence("npu_rmsnorm_static_subgraph"),
                ready_arc_npu_bounded_evidence("npu_linear_static_subgraph"),
                ready_arc_npu_bounded_evidence("npu_ffn_static_subgraph"),
            ],
            gaps: Vec::new(),
            claim_boundary: ClaimBoundary {
                cpu_is_truth_path: true,
                dense_slm_default_is_cpu_until_speedup_qualified: true,
                openvino_gpu_npu_are_candidates_not_speedup_claims: true,
                arc_bitnet_full_inference_claimed: false,
                npu_bitnet_full_inference_claimed: false,
                qk256_accelerator_decode_claimed: false,
                hidden_fallback_allowed: false,
            },
        }
    }

    fn ready_bitnet_cpu_reference_evidence(evidence_id: &str) -> EvidenceStatus {
        EvidenceStatus {
            evidence_id: evidence_id.to_string(),
            path: format!("{evidence_id}.json"),
            present: true,
            artifact_kind: Some("bitnet_cpu_reference_receipt".to_string()),
            requested_backend: None,
            selected_backend: Some("intel-258v-cpu-avx2".to_string()),
            runtime_api: Some("cpu".to_string()),
            fallback_used: Some(false),
            answer_gate_passed: Some(true),
            phase_timing_present: Some(true),
            speedup_claim: Some(false),
            issues: Vec::new(),
        }
    }

    fn ready_arc_npu_bounded_evidence(evidence_id: &str) -> EvidenceStatus {
        EvidenceStatus {
            evidence_id: evidence_id.to_string(),
            path: format!("{evidence_id}.json"),
            present: true,
            artifact_kind: Some("bounded_parity_receipt".to_string()),
            requested_backend: None,
            selected_backend: None,
            runtime_api: None,
            fallback_used: Some(false),
            answer_gate_passed: None,
            phase_timing_present: None,
            speedup_claim: Some(false),
            issues: Vec::new(),
        }
    }

    fn ready_operator_ask_summary() -> OperatorAskRegressionSummary {
        OperatorAskRegressionSummary {
            path: AUTO_NPU_WARM_RESIDENT_ASK_RECEIPT.to_string(),
            ask_receipt_ready: true,
            profile_id: "warm_resident".to_string(),
            requested_device: "auto".to_string(),
            requested_route: "auto".to_string(),
            selected_route: "dense_slm_openvino_npu_candidate".to_string(),
            selected_backend: "openvino-npu".to_string(),
            runtime_api: "openvino_genai".to_string(),
            promotion_status: "promoted".to_string(),
            route_profile_status: Some("promoted_route_ready".to_string()),
            route_profile_blockers: vec![],
            fallback_used: false,
            answer_gate_passed: true,
            openvino_candidate_route_executed: true,
            new_inference_executed: true,
            speedup_claim: false,
            power_advantage_claim: false,
            acceleration_claim: false,
            bitnet_qk256_i2s_claim: false,
            generated_token_ids_available: true,
            source_run_receipt: Some(
                "lunar-lake-operator-ask-auto-npu-warm-resident-math-brief-source-run.json"
                    .to_string(),
            ),
            regression_ready: true,
            gaps: vec![],
        }
    }

    fn ready_gpu_operator_ask_summary(
        profile_id: &str,
        path: &str,
    ) -> OperatorAskRegressionSummary {
        OperatorAskRegressionSummary {
            path: path.to_string(),
            ask_receipt_ready: true,
            profile_id: profile_id.to_string(),
            requested_device: "auto".to_string(),
            requested_route: "auto".to_string(),
            selected_route: "dense_slm_openvino_gpu_candidate".to_string(),
            selected_backend: "openvino-gpu".to_string(),
            runtime_api: "openvino_genai".to_string(),
            promotion_status: "promoted".to_string(),
            route_profile_status: Some("promoted_route_ready".to_string()),
            route_profile_blockers: vec![],
            fallback_used: false,
            answer_gate_passed: true,
            openvino_candidate_route_executed: true,
            new_inference_executed: true,
            speedup_claim: false,
            power_advantage_claim: false,
            acceleration_claim: false,
            bitnet_qk256_i2s_claim: false,
            generated_token_ids_available: true,
            source_run_receipt: Some(path.replace(".json", "-source-run.json")),
            regression_ready: true,
            gaps: vec![],
        }
    }

    fn write_minimal_route_policy(root: &Path) -> Result<()> {
        write_json(
            root,
            ROUTE_PROMOTION_LEDGER,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_route_promotion_ledger",
                "proof_stage": "route_promotion_policy_recorded",
                "created_utc": "2026-05-19T14:30:00Z",
                "machine_id": "intel-258v",
                "artifact_root": path_string(root),
                "operator_receipt": "lunar-lake-operator-readiness.json",
                "comparison_receipt": "lunar-lake-operator-comparison.json",
                "promotion_ready": true,
                "default_route_id": "dense_slm_default_cpu",
                "auto_route_policy": {
                    "policy_stage": "ledger_driven_auto_route_enabled",
                    "default_route": "dense_slm_default_cpu",
                    "hidden_fallback_allowed": false,
                    "cpu_default_until_profile_promoted": true,
                    "candidate_routes_require_profile_promotion": true,
                    "route_reason_required": true,
                    "notes": []
                },
                "workload_profiles": [
                    {
                        "profile_id": "ask_normal",
                        "prompt_tokens": "<=512",
                        "output_tokens": "<=128",
                        "purpose": "normal ask",
                        "promoted_route": "dense_slm_openvino_gpu_candidate",
                        "candidate_routes": ["dense_slm_default_cpu"]
                    },
                    {
                        "profile_id": "ask_short",
                        "prompt_tokens": "<=64",
                        "output_tokens": "<=32",
                        "purpose": "short ask",
                        "promoted_route": "dense_slm_openvino_gpu_candidate",
                        "candidate_routes": ["dense_slm_default_cpu"]
                    },
                    {
                        "profile_id": "low_power",
                        "prompt_tokens": "<=512",
                        "output_tokens": "<=128",
                        "purpose": "low power",
                        "promoted_route": null,
                        "candidate_routes": [
                            "dense_slm_default_cpu",
                            "dense_slm_openvino_npu_candidate"
                        ]
                    }
                ],
                "routes": [
                    {
                        "route_id": "dense_slm_default_cpu",
                        "status": "promoted",
                        "promoted_for": ["regression_tiny"],
                        "blocked_for": ["openvino_gpu_promoted_for_ask_normal"],
                        "required_evidence": ["fallback_used=false"],
                        "present_evidence": ["cpu"],
                        "missing_evidence": [],
                        "selected_backend": "cpu-rust",
                        "runtime_api": "cpu",
                        "fallback_policy": "strict_no_fallback",
                        "answer_gate_evidence": "cpu.json",
                        "phase_evidence": "phase.json",
                        "fallback_used": false,
                        "answer_gate_passed": true,
                        "phase_timing_present": true,
                        "speedup_claim": false,
                        "acceleration_claim": false,
                        "last_evidence_utc": "2026-05-19T14:30:00Z",
                        "reason": "CPU baseline"
                    },
                    {
                        "route_id": "dense_slm_openvino_gpu_candidate",
                        "status": "promoted",
                        "promoted_for": ["ask_normal", "ask_short"],
                        "blocked_for": ["low_power_power_advantage_unproven"],
                        "required_evidence": ["fallback_used=false"],
                        "present_evidence": ["gpu"],
                        "missing_evidence": [],
                        "selected_backend": "openvino-gpu",
                        "runtime_api": "openvino_genai",
                        "fallback_policy": "strict_no_fallback",
                        "answer_gate_evidence": "gpu.json",
                        "phase_evidence": "phase.json",
                        "fallback_used": false,
                        "answer_gate_passed": true,
                        "phase_timing_present": true,
                        "speedup_claim": false,
                        "acceleration_claim": false,
                        "last_evidence_utc": "2026-05-19T14:30:00Z",
                        "reason": "GPU profile promotion"
                    },
                    {
                        "route_id": "dense_slm_openvino_npu_candidate",
                        "status": "candidate",
                        "promoted_for": [],
                        "blocked_for": ["low_power_power_advantage_unproven"],
                        "required_evidence": ["benchmark_qualified_speedup_or_power_advantage"],
                        "present_evidence": ["npu"],
                        "missing_evidence": ["benchmark_qualified_speedup_or_power_advantage"],
                        "selected_backend": "openvino-npu",
                        "runtime_api": "openvino_genai",
                        "fallback_policy": "strict_no_fallback",
                        "answer_gate_evidence": "npu.json",
                        "phase_evidence": "phase.json",
                        "fallback_used": false,
                        "answer_gate_passed": true,
                        "phase_timing_present": true,
                        "speedup_claim": false,
                        "acceleration_claim": false,
                        "last_evidence_utc": "2026-05-19T14:30:00Z",
                        "reason": "NPU candidate"
                    }
                ],
                "gaps": [],
                "claim_boundary": {
                    "cpu_is_truth_path": true,
                    "dense_slm_default_is_cpu_until_speedup_qualified": true,
                    "openvino_gpu_npu_are_candidates_not_speedup_claims": true,
                    "arc_bitnet_full_inference_claimed": false,
                    "npu_bitnet_full_inference_claimed": false,
                    "qk256_accelerator_decode_claimed": false,
                    "hidden_fallback_allowed": false
                }
            }),
        )?;
        write_json(
            root,
            ROUTE_PROFILE_COMPARISON,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_route_profile_comparison",
                "proof_stage": "route_profiles_indexed_no_promotion_change",
                "created_utc": "2026-05-19T14:30:00Z",
                "machine_id": "intel-258v",
                "artifact_root": path_string(root),
                "promotion_ledger": "lunar-lake-route-promotion.json",
                "phase_comparison_receipt": "phase.json",
                "profile_comparison_ready": true,
                "default_route_id": "dense_slm_default_cpu",
                "profiles": [
                    {
                        "profile_id": "ask_normal",
                        "prompt_tokens": "<=512",
                        "output_tokens": "<=128",
                        "purpose": "normal ask",
                        "promoted_route": "dense_slm_openvino_gpu_candidate",
                        "candidate_routes": ["dense_slm_default_cpu"],
                        "profile_status": "promoted_route_ready",
                        "route_evidence": [],
                        "promotion_decision": "promoted",
                        "gaps": []
                    },
                    {
                        "profile_id": "ask_short",
                        "prompt_tokens": "<=64",
                        "output_tokens": "<=32",
                        "purpose": "short ask",
                        "promoted_route": "dense_slm_openvino_gpu_candidate",
                        "candidate_routes": ["dense_slm_default_cpu"],
                        "profile_status": "promoted_route_ready",
                        "route_evidence": [],
                        "promotion_decision": "promoted",
                        "gaps": []
                    },
                    {
                        "profile_id": "low_power",
                        "prompt_tokens": "<=512",
                        "output_tokens": "<=128",
                        "purpose": "low power",
                        "promoted_route": null,
                        "candidate_routes": [
                            "dense_slm_default_cpu",
                            "dense_slm_openvino_npu_candidate"
                        ],
                        "profile_status": "no_promoted_route",
                        "route_evidence": [
                            {
                                "route_id": "dense_slm_openvino_npu_candidate",
                                "route_status": "candidate",
                                "ledger_route_status": "candidate",
                                "selected_model": "Qwen2.5-0.5B-Instruct OpenVINO IR INT4_SYM",
                                "selected_backend": "openvino-npu",
                                "runtime_api": "openvino_genai",
                                "model_identity": {
                                    "identity_source": "dense_slm_openvino_ir_manifest",
                                    "manifest_receipt": "slm-openvino-ir-qwen25-int4-sym-manifest.json",
                                    "selected_model": "Qwen2.5-0.5B-Instruct OpenVINO IR INT4_SYM",
                                    "model_name": "Qwen2.5-0.5B-Instruct",
                                    "model_family": "qwen2.5",
                                    "model_format": "OpenVINO IR",
                                    "model_artifact": "openvino/qwen25-0.5b-int4-sym",
                                    "model_sha256": null,
                                    "repo": "Qwen/Qwen2.5-0.5B-Instruct",
                                    "repo_revision": "main",
                                    "quantization": "INT4_symmetric",
                                    "tokenizer_source": "Qwen/Qwen2.5-0.5B-Instruct",
                                    "tokenizer_family": "qwen2",
                                    "prompt_template": "qwen2.5-instruct-chatml",
                                    "stop_token_policy": "eos_or_stop_sequence",
                                    "known_gaps": [
                                        "OpenVINO IR model binaries are not committed; manifest pins source model revision and export contract instead of a local binary SHA256"
                                    ]
                                },
                                "fallback_used": false,
                                "answer_gate_passed": true,
                                "phase_timing_present": true,
                                "timing": {
                                    "timing_scope": "minimal",
                                    "source_receipts": [],
                                    "prompt_tokens": 12,
                                    "cold_load_ms": null,
                                    "tokenize_ms": null,
                                    "prefill_ms": null,
                                    "first_token_ms": null,
                                    "decode_total_ms": null,
                                    "generation_total_ms": null,
                                    "total_response_ms": null,
                                    "output_tokens": 4,
                                    "throughput_tokens_per_s": null,
                                    "phase_coverage": [],
                                    "known_gaps": []
                                },
                                "timing_applicability": {
                                    "profile_id": "low_power",
                                    "required_prompt_tokens": "<=512",
                                    "required_output_tokens": "<=128",
                                    "measured_prompt_tokens": 12,
                                    "measured_output_tokens": 4,
                                    "timing_matches_profile": true,
                                    "notes": []
                                },
                                "benchmark_qualified_advantage": false,
                                "promotion_eligible_for_profile": false,
                                "evidence": [],
                                "blockers": ["benchmark_qualified_speedup_or_power_advantage"]
                            }
                        ],
                        "promotion_decision": "blocked",
                        "gaps": ["benchmark_qualified_speedup_or_power_advantage"]
                    }
                ],
                "route_promotion_scope": {
                    "openvino_gpu_promoted_profiles": ["ask_normal", "ask_short"],
                    "openvino_npu_promoted_profiles": [],
                    "profile_scoped_promotion_only": true,
                    "openvino_npu_remains_candidate": true,
                    "unexpected_openvino_profile_promotions": [],
                    "notes": []
                },
                "gaps": [],
                "claim_boundary": {
                    "cpu_is_truth_path": true,
                    "dense_slm_default_is_cpu_until_speedup_qualified": true,
                    "openvino_gpu_npu_are_candidates_not_speedup_claims": true,
                    "arc_bitnet_full_inference_claimed": false,
                    "npu_bitnet_full_inference_claimed": false,
                    "qk256_accelerator_decode_claimed": false,
                    "hidden_fallback_allowed": false
                }
            }),
        )?;
        Ok(())
    }

    fn write_bitnet_semantic_intake_inputs(
        root: &Path,
        status: &str,
        merged_at_utc: Option<&str>,
        cpu_captured_at_utc: &str,
        operator_created_utc: &str,
    ) -> Result<()> {
        write_json(
            root,
            BITNET_SEMANTIC_SOURCE_CHANGES,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_bitnet_semantic_source_changes",
                "created_utc": "2026-05-19T05:45:00Z",
                "machine_id": "intel-258v",
                "changes": [
                    {
                        "source_lane": "a770",
                        "source_pr": 5020,
                        "title": "fix(bitnet): preserve K precision for attention scores",
                        "status": status,
                        "base_ref": "a770/diag-score-input-attribution",
                        "head_sha": "dc4a8ac77750e781c99e2e02af1279a40d476ac7",
                        "merged_at_utc": merged_at_utc,
                        "semantic_scope": ["attention_score_k_precision"],
                        "requires_lunar_lake_rerun_when_merged_to_main": true,
                        "claim_boundary": "shared CPU/A770 runtime correctness fix candidate; no Lunar Lake claim until rerun"
                    }
                ]
            }),
        )?;
        write_json(
            root,
            BITNET_CPU_BUNDLE,
            json!({
                "artifact_kind": "intel_258v_cpu_reference_bundle",
                "captured_at_utc": cpu_captured_at_utc,
                "machine_id": "intel-258v",
                "cpu_reference": {
                    "fallback_used": false
                }
            }),
        )?;
        write_json(
            root,
            OPERATOR_COMPARISON,
            json!({
                "artifact_kind": "lunar_lake_operator_comparison",
                "created_utc": operator_created_utc,
                "machine_id": "intel-258v",
                "comparison_ready": true,
                "claim_boundary": {
                    "hidden_fallback_allowed": false
                }
            }),
        )?;
        Ok(())
    }

    fn write_ready_bitnet_semantic_intake(root: &Path, file: &str) -> Result<()> {
        write_json(
            root,
            file,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_bitnet_semantic_intake",
                "proof_stage": "shared_bitnet_semantic_intake_no_new_inference",
                "created_utc": "2026-05-19T05:45:00Z",
                "machine_id": "intel-258v",
                "artifact_root": path_string(root),
                "source_changes_receipt": path_string(&root.join(BITNET_SEMANTIC_SOURCE_CHANGES)),
                "cpu_reference_bundle": path_string(&root.join(BITNET_CPU_BUNDLE)),
                "operator_comparison": path_string(&root.join(OPERATOR_COMPARISON)),
                "source_change_summary": {
                    "total_change_count": 1,
                    "pending_shared_change_count": 1,
                    "merged_to_main_count": 0,
                    "stale_after_merged_count": 0,
                    "source_lanes": ["a770"],
                    "pending_changes": ["a770#5020 fix(bitnet): preserve K precision for attention scores"],
                    "merged_changes": [],
                    "notes": ["pending shared changes are indexed but do not invalidate Lunar Lake receipts until they merge to main"]
                },
                "lunar_lake_evidence": {
                    "cpu_reference_bundle_created_utc": "2026-05-12T18:43:14Z",
                    "operator_comparison_created_utc": "2026-05-19T05:30:00Z",
                    "evidence_cutoff_utc": "2026-05-12T18:43:14Z",
                    "cpu_reference_bundle_path": path_string(&root.join(BITNET_CPU_BUNDLE)),
                    "operator_comparison_path": path_string(&root.join(OPERATOR_COMPARISON)),
                    "evidence_paths": [BITNET_CPU_BUNDLE, OPERATOR_COMPARISON]
                },
                "changes": [
                    {
                        "source_lane": "a770",
                        "source_pr": 5020,
                        "title": "fix(bitnet): preserve K precision for attention scores",
                        "status": "stack_open",
                        "semantic_scope": ["attention_score_k_precision"],
                        "requires_lunar_lake_rerun_when_merged_to_main": true,
                        "merged_at_utc": null,
                        "stale_after_cpu_reference": false,
                        "stale_after_operator_comparison": false,
                        "lunar_lake_rerun_required": false,
                        "notes": ["pending shared semantic change will require Lunar Lake reruns after main merge"]
                    }
                ],
                "rerun_required": false,
                "required_reruns": [],
                "intake_ready": true,
                "gaps": [],
                "claim_boundary": {
                    "new_inference_executed": false,
                    "route_promotion_changed": false,
                    "answer_quality_claim": false,
                    "speedup_claim": false,
                    "acceleration_claim": false,
                    "arc_or_npu_bitnet_claim": false,
                    "qk256_behavior_changed": false,
                    "dense_slm_as_bitnet_proof": false,
                    "hidden_fallback_allowed": false
                }
            }),
        )
    }

    fn write_stale_bitnet_semantic_intake(root: &Path, file: &str) -> Result<()> {
        write_json(
            root,
            file,
            json!({
                "schema_version": "1.0.0",
                "artifact_kind": "lunar_lake_bitnet_semantic_intake",
                "proof_stage": "shared_bitnet_semantic_intake_no_new_inference",
                "created_utc": "2026-05-19T06:05:00Z",
                "machine_id": "intel-258v",
                "artifact_root": path_string(root),
                "source_changes_receipt": path_string(&root.join(BITNET_SEMANTIC_SOURCE_CHANGES)),
                "cpu_reference_bundle": path_string(&root.join(BITNET_CPU_BUNDLE)),
                "operator_comparison": path_string(&root.join(OPERATOR_COMPARISON)),
                "source_change_summary": {
                    "total_change_count": 1,
                    "pending_shared_change_count": 0,
                    "merged_to_main_count": 1,
                    "stale_after_merged_count": 1,
                    "source_lanes": ["a770"],
                    "pending_changes": [],
                    "merged_changes": ["a770#5020 fix(bitnet): preserve K precision for attention scores"],
                    "notes": ["merged shared semantic change is newer than Lunar Lake evidence"]
                },
                "lunar_lake_evidence": {
                    "cpu_reference_bundle_created_utc": "2026-05-12T18:43:14Z",
                    "operator_comparison_created_utc": "2026-05-19T05:30:00Z",
                    "evidence_cutoff_utc": "2026-05-12T18:43:14Z",
                    "cpu_reference_bundle_path": path_string(&root.join(BITNET_CPU_BUNDLE)),
                    "operator_comparison_path": path_string(&root.join(OPERATOR_COMPARISON)),
                    "evidence_paths": [BITNET_CPU_BUNDLE, OPERATOR_COMPARISON]
                },
                "changes": [
                    {
                        "source_lane": "a770",
                        "source_pr": 5020,
                        "title": "fix(bitnet): preserve K precision for attention scores",
                        "status": "merged_to_main",
                        "semantic_scope": ["attention_score_k_precision"],
                        "requires_lunar_lake_rerun_when_merged_to_main": true,
                        "merged_at_utc": "2026-05-19T06:00:00Z",
                        "stale_after_cpu_reference": true,
                        "stale_after_operator_comparison": true,
                        "lunar_lake_rerun_required": true,
                        "notes": ["merged shared semantic change is newer than Lunar Lake BitNet evidence"]
                    }
                ],
                "rerun_required": true,
                "required_reruns": [
                    "rerun Lunar Lake BitNet CPU answer corpus",
                    "rerun scalar-vs-AVX2 BitNet answer parity"
                ],
                "intake_ready": false,
                "gaps": [
                    "merged shared BitNet semantic changes require refreshed Lunar Lake BitNet evidence"
                ],
                "claim_boundary": {
                    "new_inference_executed": false,
                    "route_promotion_changed": false,
                    "answer_quality_claim": false,
                    "speedup_claim": false,
                    "acceleration_claim": false,
                    "arc_or_npu_bitnet_claim": false,
                    "qk256_behavior_changed": false,
                    "dense_slm_as_bitnet_proof": false,
                    "hidden_fallback_allowed": false
                }
            }),
        )
    }

    fn write_json(root: &Path, file: &str, value: Value) -> Result<()> {
        fs::create_dir_all(root)?;
        fs::write(root.join(file), serde_json::to_vec_pretty(&value)?)?;
        Ok(())
    }

    fn benchmark_qualified_openvino_profile(profile_id: &str, route_id: &str) -> Value {
        json!({
            "profile_id": profile_id,
            "route_evidence": [
                {
                    "route_id": route_id,
                    "benchmark_qualified_advantage": true,
                    "fallback_used": false,
                    "answer_gate_passed": true,
                    "phase_timing_present": true,
                    "timing_applicability": {
                        "timing_matches_profile": true
                    },
                    "profile_quality": {
                        "profile_present": true,
                        "fallback_used": false,
                        "failed": 0
                    },
                    "route_advantage_context": {
                        "benchmark_qualified": true
                    },
                    "blockers": [
                        format!("route not promoted for profile {profile_id}")
                    ]
                }
            ]
        })
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
  - id: arithmetic_add_7_8
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
  - id: short_reasoning_apples_left
    category: short_reasoning
    profile: ask_normal
    gate: {kind: contains_any}
  - id: decode_heavy_short_list
    category: decode_heavy
    profile: decode_heavy
    gate: {kind: readable}
  - id: low_power_route_evidence_copy
    category: copy_exact
    profile: low_power
    gate: {kind: contains_any}
  - id: warm_resident_route_copy
    category: resident_session
    profile: warm_resident
    gate: {kind: readable}
"#,
        )?;
        Ok(())
    }

    fn write_route_corpus_v2_receipts(root: &Path) -> Result<()> {
        let current_case_ids = json!([
            {"id": "math_2_plus_2_brief"},
            {"id": "arithmetic_add_7_8"},
            {"id": "copy_exact_color_triplet"},
            {"id": "yes_no_clear_sky"},
            {"id": "short_factual_capital_france"},
            {"id": "instruction_single_sentence_rust"},
            {"id": "stop_token_one_word_done"},
            {"id": "transcript_context_code_word"},
            {"id": "structured_json_city_country"},
            {"id": "long_prompt_summary_route_policy"},
            {"id": "short_reasoning_apples_left"},
            {"id": "decode_heavy_short_list"},
            {"id": "low_power_route_evidence_copy"},
            {"id": "warm_resident_route_copy"}
        ]);
        let stale_openvino_case_ids = json!([
            {
                "id": "math_2_plus_2_brief",
                "profile": "regression_tiny",
                "prompt_token_count": 39,
                "generated_token_count": 8,
                "timing": {
                    "generation_wall_ms": 534.9,
                    "first_streamed_text_chunk_ms": 470.1,
                    "openvino_perf_metrics": {
                        "load_time_ms": 4390.0,
                        "tokenization": {"mean_ms": 17.1},
                        "time_to_first_token": {"mean_ms": 306.3},
                        "num_generated_tokens": 8,
                        "throughput": {"mean_ms": 30.8}
                    }
                }
            },
            {
                "id": "copy_exact_color_triplet",
                "profile": "regression_tiny",
                "prompt_token_count": 41,
                "generated_token_count": 2,
                "timing": {"generation_wall_ms": 404.3}
            },
            {
                "id": "yes_no_clear_sky",
                "profile": "ask_short",
                "prompt_token_count": 41,
                "generated_token_count": 2,
                "timing": {
                    "generation_wall_ms": 175.1,
                    "first_streamed_text_chunk_ms": 104.0,
                    "openvino_perf_metrics": {
                        "load_time_ms": 4390.0,
                        "tokenization": {"mean_ms": 16.8},
                        "time_to_first_token": {"mean_ms": 106.0},
                        "num_generated_tokens": 2,
                        "throughput": {"mean_ms": 24.0}
                    }
                }
            },
            {"id": "short_factual_capital_france", "profile": "ask_short", "prompt_token_count": 49, "generated_token_count": 1, "timing": {"generation_wall_ms": 86.1}},
            {"id": "instruction_single_sentence_rust", "profile": "ask_normal", "prompt_token_count": 35, "generated_token_count": 7, "timing": {"generation_wall_ms": 340.6}},
            {"id": "stop_token_one_word_done", "profile": "regression_tiny", "prompt_token_count": 38, "generated_token_count": 1, "timing": {"generation_wall_ms": 105.7}},
            {"id": "transcript_context_code_word", "profile": "ask_normal", "prompt_token_count": 58, "generated_token_count": 1, "timing": {"generation_wall_ms": 113.9}},
            {"id": "structured_json_city_country", "profile": "structured", "prompt_token_count": 44, "generated_token_count": 16, "timing": {"generation_wall_ms": 350.2}},
            {"id": "long_prompt_summary_route_policy", "profile": "prefill_heavy", "prompt_token_count": 97, "generated_token_count": 22, "timing": {"generation_wall_ms": 699.9}},
            {"id": "short_reasoning_heavier_object", "profile": "ask_normal", "prompt_token_count": 57, "generated_token_count": 19, "timing": {"generation_wall_ms": 369.7}},
            {"id": "decode_heavy_short_list", "profile": "decode_heavy", "prompt_token_count": 57, "generated_token_count": 19, "timing": {"generation_wall_ms": 369.7}},
            {"id": "low_power_route_evidence_copy", "profile": "low_power", "prompt_token_count": 39, "generated_token_count": 2, "timing": {"generation_wall_ms": 120.0}},
            {"id": "warm_resident_route_copy", "profile": "warm_resident", "prompt_token_count": 39, "generated_token_count": 2, "timing": {"generation_wall_ms": 118.0}}
        ]);
        write_json(
            root,
            DENSE_CPU_CORPUS_V2,
            json!({
                "artifact_kind": "slm_cpu_answer_corpus",
                "fallback_used": false,
                "cases": current_case_ids,
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
                            "cases": stale_openvino_case_ids.clone(),
                            "quality_summary": {
                                "profile_summary": {
                                    "ask_short": {"total": 2, "passed": 2, "failed": 0}
                                }
                            }
                        },
                        {
                            "runtime_device": "GPU.0",
                            "fallback_used": false,
                            "cases": stale_openvino_case_ids.clone(),
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
                            "cases": stale_openvino_case_ids.clone(),
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
