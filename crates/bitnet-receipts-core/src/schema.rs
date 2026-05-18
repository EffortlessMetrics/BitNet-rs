//! Serializable receipt schema fragments.
//!
//! These data-transfer structs are isolated from validator implementations so
//! serde contract changes remain focused and reviewable.

use serde::{Deserialize, Serialize};

/// Model information in receipt
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hidden_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_attention_heads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_key_value_heads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vocab_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_correction_digest: Option<String>,
}

/// Test execution results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TestResults {
    pub total_tests: usize,
    pub passed: usize,
    pub failed: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy_tests: Option<AccuracyTestResults>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub determinism_tests: Option<DeterminismTestResults>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_cache_tests: Option<KVCacheTestResults>,
}

/// Accuracy test results (AC5)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyTestResults {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub i2s_accuracy: Option<AccuracyMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tl1_accuracy: Option<AccuracyMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tl2_accuracy: Option<AccuracyMetric>,
}

/// Individual accuracy metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetric {
    pub mse: f64,
    pub tolerance: f64,
    pub passed: bool,
}

/// Determinism test results (AC3, AC6)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterminismTestResults {
    pub identical_sequences: bool,
    pub runs: usize,
    pub tokens_per_run: usize,
}

/// KV-cache test results (AC7)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheTestResults {
    pub prefill_decode_parity: bool,
    pub cache_hit_rate: f64,
}

/// Performance baseline metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceBaseline {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_generated: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_time_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_token_latency_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub average_token_latency_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_usage_mb: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_efficiency: Option<CacheEfficiency>,
}

/// Cache efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEfficiency {
    pub kv_cache_hit_rate: f64,
    pub tensor_cache_hits: usize,
    pub tensor_cache_misses: usize,
}

/// Cross-validation metrics (deprecated - use ParityMetadata instead)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CrossValidation {
    pub cpp_reference_available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parity_tests_passed: Option<bool>,
}

/// Parity validation metadata (AC4)
///
/// Captures C++ reference comparison metrics for reproducibility and CI validation.
///
/// # Schema Version: 1.0.0
///
/// Status values:
/// - "ok": Rust and C++ outputs match (cosine ≥ 0.99, exact_match_rate = 1.0)
/// - "rust_only": C++ reference not available
/// - "divergence": Outputs differ (cosine < 0.99 or exact_match_rate < 1.0)
/// - "timeout": Parity test exceeded timeout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityMetadata {
    /// C++ reference available for comparison
    pub cpp_available: bool,

    /// Cosine similarity between Rust and C++ logits (0.0 to 1.0)
    /// Present only when cpp_available=true
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cosine_similarity: Option<f32>,

    /// Exact match rate for generated tokens (0.0 to 1.0)
    /// Present only when cpp_available=true
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exact_match_rate: Option<f32>,

    /// Parity status: "ok" | "rust_only" | "divergence" | "timeout"
    pub status: String,
}

/// Strict CPU inference provenance required for end-to-end proof receipts.
///
/// These fields make the strict CPU lane auditable: a receipt can state which
/// backend/kernel were requested, which were actually selected, which loader and
/// tokenizer authorities were used, and whether any fallback was taken.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StrictInferenceProvenance {
    /// Backend requested by the caller (for strict CPU proofs this must be a CPU proof label).
    pub requested_backend: String,
    /// Backend selected by runtime dispatch (for strict CPU proofs this must be a CPU proof label).
    pub selected_backend: String,
    /// Kernel requested by the caller, for example `qk256-avx2-gemv`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_kernel: Option<String>,
    /// Kernel selected by runtime dispatch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_kernel: Option<String>,
    /// Loader authority, for example `real_gguf`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loader_mode: Option<String>,
    /// Tokenizer authority, for example `explicit`, `embedded_gguf`, or `sibling_file`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_source: Option<String>,
    /// True when tokenizer resolution ran under strict proof policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_strict: Option<bool>,
    /// Model family normalized from GGUF metadata, for example `llama` or `bitnet`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_family: Option<String>,
    /// Quantization format normalized from metadata, for example `I2_S` or `QK256/I2_S`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quant_format: Option<String>,
    /// CPU model string reported by the proof host.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_model: Option<String>,
    /// Runtime CPU feature list used for dispatch decisions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cpu_features: Vec<String>,
    /// Thread count used by the decode lane.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_count: Option<usize>,
    /// True if any compatibility, mock, diagnostic, scalar-substitution, or dequant fallback was used.
    pub fallback_used: bool,
    /// Human-readable fallback reason; must be absent when `fallback_used=false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    /// Prompt token count seen by the proof run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<usize>,
    /// Decode token count generated by the proof run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_tokens: Option<usize>,
    /// Strict proof phase: `prefill` or `decode`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    /// p50 per-token decode latency in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_p50_ms: Option<f64>,
    /// p95 per-token decode latency in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_p95_ms: Option<f64>,
    /// Decode throughput in tokens per second.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_tps: Option<f64>,
}

/// Reusable Apple M4 run identity carried by receipts that participate in
/// matching-history comparison.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct M4RunIdentity {
    pub contract_version: String,
    pub machine_id: String,
    pub soc: String,
    pub artifact_kind: String,
    pub evidence_family: String,
    pub os: M4RunIdentityOs,
    pub git: M4RunIdentityGit,
    pub binary: M4RunIdentityBinary,
    pub command: M4RunIdentityCommand,
    pub model: M4RunIdentityModel,
    pub tokenizer: M4RunIdentityTokenizer,
    pub prompt_template: M4RunIdentityPromptTemplate,
    pub backend: M4RunIdentityBackend,
    pub evidence_identity: M4RunIdentityEvidence,
    pub timing: M4RunIdentityTiming,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct M4RunIdentityOs {
    pub name: String,
    pub version: String,
    pub version_source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct M4RunIdentityGit {
    pub commit: String,
    pub commit_source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct M4RunIdentityBinary {
    pub crate_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub build_profile: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct M4RunIdentityCommand {
    pub class: String,
    pub live_model_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct M4RunIdentityModel {
    pub id: String,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loader_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity_scope: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct M4RunIdentityTokenizer {
    pub authority: String,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pretokenizer_authority: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity_scope: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct M4RunIdentityPromptTemplate {
    pub id: String,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity_scope: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct M4RunIdentityBackend {
    pub requested_backend: String,
    pub selected_backend: String,
    pub runtime_api: String,
    pub fallback_used: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct M4RunIdentityEvidence {
    pub scope: String,
    pub seed: String,
    pub corpus_id: String,
    pub profile_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct M4RunIdentityTiming {
    pub source: String,
}
