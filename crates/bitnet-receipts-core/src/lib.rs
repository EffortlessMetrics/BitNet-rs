//! # Inference Receipt Generation (AC4)
//!
//! Generates receipt artifacts documenting real inference execution.
//! Implements schema version 1.0.0 as specified in issue-254-real-inference-spec.md.
//!
//! # Schema Requirements (AC4)
//! - `compute_path`: Must be "real" (not "mock")
//! - `backend`: "cpu" | "cuda" | "metal"
//! - `kernels`: List of executed kernels (e.g., ["i2s_gemv", "rope_apply"])
//! - `deterministic`: Boolean indicating BITNET_DETERMINISTIC=1
//! - `environment`: Environment variables used
//! - `model_info`: Model configuration details
//! - `test_results`: Test execution summary
//! - `performance_baseline`: Performance metrics

use anyhow::{Result, anyhow};
use bitnet_atomic_file_core::atomic_write;
use bitnet_common::CorrectionRecord;
use bitnet_honest_compute::{
    classify_compute_path, validate_compute_path as validate_honest_compute_path,
    validate_kernel_ids as validate_honest_kernel_ids,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// Schema version for receipt format
pub const RECEIPT_SCHEMA_VERSION: &str = "1.0.0";

/// Alias for schema version (for consistency)
pub const RECEIPT_SCHEMA: &str = RECEIPT_SCHEMA_VERSION;

/// Artifact kind for the dense regular-LLM CUDA reference lane.
///
/// This is deliberately separate from BitNet packed I2_S/QK256 CUDA receipt
/// kinds. Dense CUDA evidence may share CUDA runtime plumbing, but it must not
/// satisfy BitNet packed-kernel proof gates.
pub const DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND: &str = "dense_regular_llm_cuda";

/// Model class label for CUDA receipts that exercise dense regular LLM kernels.
pub const DENSE_REGULAR_LLM_MODEL_CLASS: &str = "dense_regular_llm";

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

/// Validate a receipt for the RTX 5070 Ti CUDA tiny-kernel smoke proof.
///
/// This validator is intentionally scoped to strict, fallback-free CUDA proof
/// receipts. It does not validate full BitNet inference and does not treat a
/// probe-only receipt as kernel execution.
pub fn validate_cuda_smoke_receipt_json(receipt: &Value) -> Result<()> {
    let stats = validate_cuda_receipt_common(receipt, "cuda_smoke", "kernel_smoke_tested")?;
    require_string_eq(stats, "kernel_id", "cuda_tiny_vector_add")?;
    require_string_eq(receipt, "result", "pass")?;
    require_positive_u64(receipt, "input_len")?;
    require_non_negative_number(receipt, "max_abs_error")?;
    require_non_negative_number(receipt, "mean_abs_error")?;
    Ok(())
}

/// Validate a receipt for the RTX 5070 Ti CUDA CPU/CUDA parity proof.
///
/// The receipt must prove one deterministic fixture matched the CPU reference,
/// with CUDA invocation counters greater than zero and zero fallback
/// invocations. It is not a benchmark or end-to-end inference validator.
pub fn validate_cuda_parity_receipt_json(receipt: &Value) -> Result<()> {
    let stats = validate_cuda_receipt_common(receipt, "cuda_parity", "cuda_cpu_parity_tested")?;
    require_string_eq(receipt, "result", "pass")?;
    require_positive_u64(receipt, "input_len")?;
    require_non_negative_number(receipt, "max_abs_error")?;
    require_non_negative_number(receipt, "mean_abs_error")?;

    let parity = object_field(receipt, "parity")?;
    require_string_eq(parity, "reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(parity, "target_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(parity, "kernel_id", required_string(stats, "kernel_id")?)?;
    require_string_non_empty(parity, "fixture_id")?;
    require_bool_eq(parity, "passed", true)?;
    require_non_negative_number(parity, "max_abs_error")?;
    require_non_negative_number(parity, "mean_abs_error")?;
    require_non_negative_number(parity, "tolerance")?;
    require_string_non_empty(parity, "tolerance_source")?;

    Ok(())
}

/// Validate a receipt for the dense regular-LLM CUDA reference lane.
///
/// This contract is intentionally a lane boundary, not a BitNet packed-kernel
/// validator. A valid dense CUDA receipt must identify itself as
/// `dense_regular_llm_cuda`, record fallback-free RTX 5070 Ti CUDA execution,
/// name a dense model class, and explicitly keep BitNet packed I2_S/QK256 proof
/// claims false.
pub fn validate_dense_regular_llm_cuda_receipt_json(receipt: &Value) -> Result<()> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND)?;
    require_string_eq(receipt, "hardware_lane", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "runtime_api", "cuda")?;
    require_bool_eq(receipt, "fallback_used", false)?;
    require_null(receipt, "fallback_backend")?;
    require_null(receipt, "fallback_reason")?;
    require_bool_eq(receipt, "speedup_claim", false)?;
    require_null(receipt, "error")?;

    let cuda = object_field(receipt, "cuda")?;
    require_bool_eq(cuda, "available", true)?;
    require_positive_u64(cuda, "device_count")?;
    require_cuda_device_index(cuda)?;
    require_rtx_5070_ti_name(cuda, "device_name")?;
    require_string_eq(cuda, "compute_capability", "12.0")?;
    require_string_non_empty_not_tbd(cuda, "driver_version")?;
    require_string_non_empty_not_tbd(cuda, "cuda_runtime_version")?;
    require_string_non_empty_not_tbd(cuda, "cuda_toolkit_version")?;
    require_string_non_empty_not_tbd(cuda, "nvrtc_version")?;
    require_positive_u64(cuda, "vram_bytes")?;

    let model = object_field(receipt, "model")?;
    require_string_non_empty(model, "model_family")?;
    reject_bitnet_packed_marker(required_string(model, "model_family")?, "model.model_family")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_non_empty(execution_path, "kernel_family")?;
    reject_bitnet_packed_marker(
        required_string(execution_path, "kernel_family")?,
        "execution_path.kernel_family",
    )?;
    require_string_non_empty(execution_path, "quantization_family")?;
    reject_bitnet_packed_marker(
        required_string(execution_path, "quantization_family")?,
        "execution_path.quantization_family",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    let stats = first_kernel_stats(receipt)?;
    require_string_non_empty(stats, "kernel_id")?;
    reject_bitnet_packed_marker(required_string(stats, "kernel_id")?, "kernel_stats[0].kernel_id")?;
    require_positive_u64(stats, "invocations")?;
    require_u64_eq(stats, "fallback_invocations", 0)?;
    require_optional_positive_u64(stats, "host_to_device_bytes")?;
    require_optional_positive_u64(stats, "device_to_host_bytes")?;
    require_optional_non_negative_number(stats, "kernel_time_ms")?;

    let parity = object_field(receipt, "parity")?;
    require_string_non_empty(parity, "reference_backend")?;
    require_string_eq(parity, "target_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_non_empty(parity, "kernel_id")?;
    reject_bitnet_packed_marker(required_string(parity, "kernel_id")?, "parity.kernel_id")?;
    require_string_non_empty(parity, "fixture_id")?;
    reject_bitnet_packed_marker(required_string(parity, "fixture_id")?, "parity.fixture_id")?;
    require_bool_eq(parity, "passed", true)?;
    require_non_negative_number(parity, "max_abs_error")?;
    require_non_negative_number(parity, "mean_abs_error")?;
    require_non_negative_number(parity, "tolerance")?;
    require_string_non_empty(parity, "tolerance_source")?;

    Ok(())
}

/// Validate dense regular-LLM CUDA tensor-residency evidence.
///
/// This builds on the dense CUDA boundary validator and requires a
/// `tensor_residency` section proving the deterministic dense fixture placed
/// its input and output tensors in CUDA device buffers for the kernel launch.
/// It is still a fixture-level residency receipt, not a dense GGUF inference,
/// speedup, server, or full CUDA residency claim.
pub fn validate_dense_regular_llm_cuda_tensor_residency_receipt_json(
    receipt: &Value,
) -> Result<()> {
    validate_dense_regular_llm_cuda_receipt_json(receipt)?;
    require_string_eq(receipt, "claim", "dense_regular_llm_cuda_tensor_residency_tested")?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;

    let stats = first_kernel_stats(receipt)?;
    let residency = object_field(receipt, "tensor_residency")?;
    require_string_eq(residency, "scope", "single_dense_f16_gemm_fixture")?;
    require_string_eq(residency, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(
        residency,
        "fixture_id",
        required_string(object_field(receipt, "parity")?, "fixture_id")?,
    )?;
    require_bool_eq(residency, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(residency, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(residency, "persistent_session_residency_claimed", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    require_bool_eq(residency, "input_tensors_uploaded_once", true)?;
    require_bool_eq(residency, "output_tensor_cuda_resident_during_kernel", true)?;
    require_bool_eq(residency, "host_device_transfer_accounting_matches_kernel_stats", true)?;

    let inputs = array_field(residency, "inputs")?;
    if inputs.len() < 2 {
        return Err(anyhow!("tensor_residency.inputs must contain A and B tensors"));
    }
    for input in inputs {
        require_string_non_empty(input, "name")?;
        require_string_eq(input, "device_residency", "cuda_device_buffer")?;
        require_string_eq(input, "reuse_scope", "single_fixture_launch")?;
        require_u64_eq(input, "upload_count", 1)?;
        require_positive_u64(input, "host_bytes")?;
        reject_bitnet_packed_marker(
            required_string(input, "dtype")?,
            "tensor_residency.inputs.dtype",
        )?;
    }

    let outputs = array_field(residency, "outputs")?;
    if outputs.is_empty() {
        return Err(anyhow!("tensor_residency.outputs must contain an output tensor"));
    }
    for output in outputs {
        require_string_non_empty(output, "name")?;
        require_string_eq(output, "device_residency", "cuda_device_buffer")?;
        require_string_eq(output, "download_scope", "parity_check_only")?;
        require_positive_u64(output, "device_to_host_bytes")?;
        reject_bitnet_packed_marker(
            required_string(output, "dtype")?,
            "tensor_residency.outputs.dtype",
        )?;
    }

    let allocation = object_field(residency, "allocation")?;
    require_positive_u64(allocation, "device_buffer_count")?;
    require_u64_eq(allocation, "persistent_handle_count", 0)?;
    require_bool_eq(allocation, "persistent_handles_claimed", false)?;

    let transfer = object_field(residency, "transfer_accounting")?;
    require_string_eq(transfer, "status", "measured")?;
    require_u64_eq(
        transfer,
        "host_to_device_bytes",
        object_field(stats, "host_to_device_bytes")?.as_u64().ok_or_else(|| {
            anyhow!("kernel_stats[0].host_to_device_bytes must be an unsigned integer")
        })?,
    )?;
    require_u64_eq(
        transfer,
        "device_to_host_bytes",
        object_field(stats, "device_to_host_bytes")?.as_u64().ok_or_else(|| {
            anyhow!("kernel_stats[0].device_to_host_bytes must be an unsigned integer")
        })?,
    )?;

    Ok(())
}

/// Validate dense regular-LLM CUDA persistent fixture residency evidence.
///
/// This is still fixture-scoped evidence: it proves repeated dense FP16 GEMM
/// launches reused one CUDA context/module and persistent device buffers for
/// the deterministic fixture. It does not validate dense GGUF inference,
/// BitNet packed proof, speedup, server readiness, or full CUDA residency.
pub fn validate_dense_regular_llm_cuda_persistent_residency_receipt_json(
    receipt: &Value,
) -> Result<()> {
    validate_dense_regular_llm_cuda_receipt_json(receipt)?;
    require_string_eq(
        receipt,
        "claim",
        "dense_regular_llm_cuda_persistent_fixture_residency_tested",
    )?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;

    let stats = first_kernel_stats(receipt)?;
    let invocations = object_field(stats, "invocations")?
        .as_u64()
        .ok_or_else(|| anyhow!("kernel_stats[0].invocations must be an unsigned integer"))?;
    if invocations < 2 {
        return Err(anyhow!("persistent dense CUDA fixture must record at least two invocations"));
    }
    require_u64_eq(stats, "kernel_launches", invocations)?;
    require_u64_eq(stats, "fallback_invocations", 0)?;

    let parity = object_field(receipt, "parity")?;
    require_u64_eq(parity, "runs", invocations)?;

    let persistent = object_field(receipt, "persistent_session")?;
    require_string_eq(persistent, "scope", "persistent_dense_f16_gemm_fixture_session")?;
    require_u64_eq(persistent, "repeated_runs", invocations)?;
    require_u64_eq(persistent, "context_creations", 1)?;
    require_u64_eq(persistent, "module_loads", 1)?;
    require_u64_eq(persistent, "kernel_launches", invocations)?;
    require_u64_eq(persistent, "input_uploads", 2)?;
    require_u64_eq(persistent, "output_allocations", 1)?;
    require_positive_u64(persistent, "persistent_handle_count")?;
    require_u64_eq(persistent, "per_run_host_to_device_bytes", 0)?;
    require_bool_eq(persistent, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(persistent, "full_cuda_residency_claimed", false)?;
    require_bool_eq(persistent, "speedup_claim", false)?;

    let residency = object_field(receipt, "tensor_residency")?;
    require_string_eq(residency, "scope", "persistent_dense_f16_gemm_fixture_session")?;
    require_string_eq(residency, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(residency, "fixture_id", required_string(parity, "fixture_id")?)?;
    require_bool_eq(residency, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(residency, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(residency, "persistent_session_residency_claimed", true)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    require_bool_eq(residency, "input_tensors_uploaded_once", true)?;
    require_bool_eq(residency, "output_tensor_cuda_resident_during_kernel", true)?;
    require_bool_eq(residency, "host_device_transfer_accounting_matches_kernel_stats", true)?;
    require_u64_eq(residency, "per_run_host_to_device_bytes", 0)?;

    let inputs = array_field(residency, "inputs")?;
    if inputs.len() < 2 {
        return Err(anyhow!("tensor_residency.inputs must contain A and B tensors"));
    }
    for input in inputs {
        require_string_non_empty(input, "name")?;
        require_string_eq(input, "device_residency", "cuda_device_buffer")?;
        require_string_eq(input, "reuse_scope", "persistent_fixture_session")?;
        require_u64_eq(input, "upload_count", 1)?;
        require_positive_u64(input, "host_bytes")?;
        reject_bitnet_packed_marker(
            required_string(input, "dtype")?,
            "tensor_residency.inputs.dtype",
        )?;
    }

    let outputs = array_field(residency, "outputs")?;
    if outputs.is_empty() {
        return Err(anyhow!("tensor_residency.outputs must contain an output tensor"));
    }
    for output in outputs {
        require_string_non_empty(output, "name")?;
        require_string_eq(output, "device_residency", "cuda_device_buffer")?;
        require_string_eq(output, "download_scope", "parity_check_each_run")?;
        require_positive_u64(output, "device_to_host_bytes")?;
        reject_bitnet_packed_marker(
            required_string(output, "dtype")?,
            "tensor_residency.outputs.dtype",
        )?;
    }

    let allocation = object_field(residency, "allocation")?;
    require_positive_u64(allocation, "device_buffer_count")?;
    require_u64_eq(
        allocation,
        "persistent_handle_count",
        object_field(persistent, "persistent_handle_count")?.as_u64().ok_or_else(|| {
            anyhow!("persistent_session.persistent_handle_count must be an unsigned integer")
        })?,
    )?;
    require_bool_eq(allocation, "persistent_handles_claimed", true)?;

    let transfer = object_field(residency, "transfer_accounting")?;
    require_string_eq(transfer, "status", "measured")?;
    require_u64_eq(
        transfer,
        "host_to_device_bytes",
        object_field(stats, "host_to_device_bytes")?.as_u64().ok_or_else(|| {
            anyhow!("kernel_stats[0].host_to_device_bytes must be an unsigned integer")
        })?,
    )?;
    require_u64_eq(
        transfer,
        "device_to_host_bytes",
        object_field(stats, "device_to_host_bytes")?.as_u64().ok_or_else(|| {
            anyhow!("kernel_stats[0].device_to_host_bytes must be an unsigned integer")
        })?,
    )?;

    Ok(())
}

/// Reject dense regular-LLM CUDA receipts at BitNet packed-kernel proof gates.
///
/// BitNet QK256/I2_S validators can call this before evaluating their own proof
/// contract. It gives dense CUDA work a clear receipt label while preventing
/// dense FP/BF/INT kernels from being counted as packed BitNet evidence.
pub fn reject_dense_regular_llm_as_bitnet_packed_cuda_proof(receipt: &Value) -> Result<()> {
    let artifact_kind = receipt.get("artifact_kind").and_then(Value::as_str);
    let model_class = receipt
        .get("execution_path")
        .and_then(|execution_path| execution_path.get("model_class"))
        .and_then(Value::as_str);
    let dense_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| claim_boundary.get("dense_regular_llm_cuda_claimed"))
        .and_then(Value::as_bool);

    if artifact_kind == Some(DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND)
        || model_class == Some(DENSE_REGULAR_LLM_MODEL_CLASS)
        || dense_claim == Some(true)
    {
        return Err(anyhow!(
            "dense_regular_llm CUDA receipt cannot satisfy BitNet packed I2_S/QK256 proof"
        ));
    }

    Ok(())
}

/// Load and validate an RTX 5070 Ti CUDA smoke receipt from disk.
pub fn validate_cuda_smoke_receipt_file(path: &Path) -> Result<()> {
    let receipt = load_json_receipt(path)?;
    validate_cuda_smoke_receipt_json(&receipt)
}

/// Load and validate an RTX 5070 Ti CUDA parity receipt from disk.
pub fn validate_cuda_parity_receipt_file(path: &Path) -> Result<()> {
    let receipt = load_json_receipt(path)?;
    validate_cuda_parity_receipt_json(&receipt)
}

fn load_json_receipt(path: &Path) -> Result<Value> {
    let content = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}

fn validate_cuda_receipt_common<'a>(
    receipt: &'a Value,
    artifact_kind: &str,
    claim: &str,
) -> Result<&'a Value> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", artifact_kind)?;
    require_string_eq(receipt, "machine_id", "windows-9950x3d-rtx5070ti")?;
    require_string_eq(receipt, "hardware_lane", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "runtime_api", "cuda")?;
    require_string_eq(receipt, "claim", claim)?;
    require_bool_eq(receipt, "fallback_used", false)?;
    require_null(receipt, "fallback_backend")?;
    require_null(receipt, "fallback_reason")?;
    require_null(receipt, "error")?;

    let cuda = object_field(receipt, "cuda")?;
    require_bool_eq(cuda, "available", true)?;
    require_positive_u64(cuda, "device_count")?;
    require_cuda_device_index(cuda)?;
    require_rtx_5070_ti_name(cuda, "device_name")?;
    require_string_eq(cuda, "compute_capability", "12.0")?;
    require_string_non_empty_not_tbd(cuda, "driver_version")?;
    require_string_non_empty_not_tbd(cuda, "cuda_runtime_version")?;
    require_string_non_empty_not_tbd(cuda, "cuda_toolkit_version")?;
    require_string_non_empty_not_tbd(cuda, "nvrtc_version")?;
    require_positive_u64(cuda, "vram_bytes")?;

    let stats = first_kernel_stats(receipt)?;
    require_string_non_empty(stats, "kernel_id")?;
    require_positive_u64(stats, "invocations")?;
    require_u64_eq(stats, "fallback_invocations", 0)?;
    require_positive_u64(stats, "host_to_device_bytes")?;
    require_positive_u64(stats, "device_to_host_bytes")?;
    require_positive_u64(stats, "kernel_launches")?;
    require_optional_non_negative_number(stats, "kernel_time_ms")?;

    Ok(stats)
}

fn first_kernel_stats(receipt: &Value) -> Result<&Value> {
    let stats = object_field(receipt, "kernel_stats")?;
    let stats = stats.as_array().ok_or_else(|| anyhow!("kernel_stats must be an array"))?;
    stats.first().ok_or_else(|| anyhow!("kernel_stats must contain at least one entry"))
}

fn object_field<'a>(object: &'a Value, field: &str) -> Result<&'a Value> {
    object.get(field).ok_or_else(|| anyhow!("missing required field `{field}`"))
}

fn array_field<'a>(object: &'a Value, field: &str) -> Result<&'a Vec<Value>> {
    object_field(object, field)?
        .as_array()
        .ok_or_else(|| anyhow!("field `{field}` must be an array"))
}

fn required_string<'a>(object: &'a Value, field: &str) -> Result<&'a str> {
    object_field(object, field)?.as_str().ok_or_else(|| anyhow!("field `{field}` must be a string"))
}

fn require_string_eq(object: &Value, field: &str, expected: &str) -> Result<()> {
    let actual = required_string(object, field)?;
    if actual != expected {
        return Err(anyhow!("field `{field}` must be `{expected}`, got `{actual}`"));
    }
    Ok(())
}

fn require_string_non_empty(object: &Value, field: &str) -> Result<()> {
    let value = required_string(object, field)?;
    if value.trim().is_empty() {
        return Err(anyhow!("field `{field}` must not be empty"));
    }
    Ok(())
}

fn require_string_non_empty_not_tbd(object: &Value, field: &str) -> Result<()> {
    let value = required_string(object, field)?;
    if value.trim().is_empty() || value == "TBD" {
        return Err(anyhow!("field `{field}` must record a concrete value"));
    }
    Ok(())
}

fn require_rtx_5070_ti_name(object: &Value, field: &str) -> Result<()> {
    let value = required_string(object, field)?;
    let compact = value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    if !(compact.contains("nvidia") && compact.contains("rtx5070ti")) {
        return Err(anyhow!("field `{field}` must identify NVIDIA GeForce RTX 5070 Ti"));
    }
    Ok(())
}

fn require_bool_eq(object: &Value, field: &str, expected: bool) -> Result<()> {
    let actual = object_field(object, field)?
        .as_bool()
        .ok_or_else(|| anyhow!("field `{field}` must be a bool"))?;
    if actual != expected {
        return Err(anyhow!("field `{field}` must be `{expected}`, got `{actual}`"));
    }
    Ok(())
}

fn require_null(object: &Value, field: &str) -> Result<()> {
    if !object_field(object, field)?.is_null() {
        return Err(anyhow!("field `{field}` must be null"));
    }
    Ok(())
}

fn require_u64_eq(object: &Value, field: &str, expected: u64) -> Result<()> {
    let actual = object_field(object, field)?
        .as_u64()
        .ok_or_else(|| anyhow!("field `{field}` must be an unsigned integer"))?;
    if actual != expected {
        return Err(anyhow!("field `{field}` must be `{expected}`, got `{actual}`"));
    }
    Ok(())
}

fn require_positive_u64(object: &Value, field: &str) -> Result<()> {
    let actual = object_field(object, field)?
        .as_u64()
        .ok_or_else(|| anyhow!("field `{field}` must be an unsigned integer"))?;
    if actual == 0 {
        return Err(anyhow!("field `{field}` must be greater than zero"));
    }
    Ok(())
}

fn require_optional_positive_u64(object: &Value, field: &str) -> Result<()> {
    let value = object_field(object, field)?;
    if value.is_null() {
        return Ok(());
    }
    let actual = value
        .as_u64()
        .ok_or_else(|| anyhow!("field `{field}` must be null or an unsigned integer"))?;
    if actual == 0 {
        return Err(anyhow!("field `{field}` must be greater than zero when measured"));
    }
    Ok(())
}

fn reject_bitnet_packed_marker(value: &str, field: &str) -> Result<()> {
    let normalized = value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    const BITNET_PACKED_MARKERS: &[&str] = &["bitnet", "i2s", "qk256", "w158a8"];
    if BITNET_PACKED_MARKERS.iter().any(|marker| normalized.contains(marker)) {
        return Err(anyhow!(
            "field `{field}` must not identify BitNet packed I2_S/QK256 proof, got `{value}`"
        ));
    }
    Ok(())
}

fn require_cuda_device_index(cuda: &Value) -> Result<()> {
    if object_field(cuda, "device_index")
        .and_then(|value| {
            value
                .as_u64()
                .ok_or_else(|| anyhow!("field `device_index` must be an unsigned integer"))
        })
        .is_ok()
        || object_field(cuda, "selected_device_index")
            .and_then(|value| {
                value.as_u64().ok_or_else(|| {
                    anyhow!("field `selected_device_index` must be an unsigned integer")
                })
            })
            .is_ok()
    {
        return Ok(());
    }

    Err(anyhow!("cuda receipt must record `device_index` or `selected_device_index`"))
}

fn require_non_negative_number(object: &Value, field: &str) -> Result<()> {
    let actual = object_field(object, field)?
        .as_f64()
        .ok_or_else(|| anyhow!("field `{field}` must be a number"))?;
    if actual < 0.0 {
        return Err(anyhow!("field `{field}` must be non-negative"));
    }
    Ok(())
}

fn require_optional_non_negative_number(object: &Value, field: &str) -> Result<()> {
    let value = object_field(object, field)?;
    if value.is_null() {
        return Ok(());
    }
    let actual =
        value.as_f64().ok_or_else(|| anyhow!("field `{field}` must be null or a number"))?;
    if actual < 0.0 {
        return Err(anyhow!("field `{field}` must be non-negative"));
    }
    Ok(())
}

/// Main inference receipt structure (AC4)
///
/// # Schema Version: 1.0.0
///
/// Provides comprehensive documentation of inference execution including:
/// - Compute path verification (real vs mock)
/// - Backend selection (CPU/GPU)
/// - Kernel execution tracking
/// - Determinism validation
/// - Performance baselines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceReceipt {
    /// Schema version (always "1.0.0")
    pub schema_version: String,

    /// ISO 8601 timestamp of receipt generation
    pub timestamp: String,

    /// Compute path: "real" (required) or "mock" (fails validation)
    pub compute_path: String,

    /// Backend used: "cpu" | "cuda" | "metal"
    pub backend: String,

    /// Backend selection summary: "requested=X detected=\[Y\] selected=Z"
    /// Populated from BackendSelectionResult::summary() at receipt generation time.
    #[serde(default)]
    pub backend_summary: String,

    /// Kernels executed during inference
    /// Examples: ["i2s_gemv", "rope_apply", "attention_real"]
    pub kernels: Vec<String>,

    /// Deterministic mode enabled (BITNET_DETERMINISTIC=1)
    pub deterministic: bool,

    /// Environment variables
    pub environment: HashMap<String, String>,

    /// Model configuration
    pub model_info: ModelInfo,

    /// Test execution results
    pub test_results: TestResults,

    /// Performance metrics baseline
    pub performance_baseline: PerformanceBaseline,

    /// Cross-validation results (optional, deprecated - use parity instead)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cross_validation: Option<CrossValidation>,

    /// Parity validation results (AC4)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parity: Option<ParityMetadata>,

    /// Model corrections applied (LayerNorm rescaling, etc.)
    /// Empty if no corrections applied
    pub corrections: Vec<CorrectionRecord>,

    /// Strict CPU proof provenance (optional for legacy receipts).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict_provenance: Option<StrictInferenceProvenance>,
}

impl InferenceReceipt {
    /// Generate receipt from inference execution
    ///
    /// # AC4 Contract
    /// - Sets `compute_path="real"` if no mock kernels detected
    /// - Sets `compute_path="mock"` if any mock kernels detected
    /// - Collects environment variables (BITNET_*, RAYON_*)
    /// - Records kernel execution list
    ///
    /// # Example
    /// ```no_run
    /// use bitnet_receipts_core::InferenceReceipt;
    ///
    /// let receipt = InferenceReceipt::generate(
    ///     "cpu",
    ///     vec!["i2s_gemv".to_string(), "rope_apply".to_string()],
    ///     None,
    /// ).unwrap();
    ///
    /// assert_eq!(receipt.compute_path, "real");
    /// ```
    pub fn generate(
        backend: &str,
        kernels: Vec<String>,
        backend_summary: Option<String>,
    ) -> Result<Self> {
        // AC4: Detect mock kernels (case-insensitive)
        let compute_path = classify_compute_path(kernels.iter().map(String::as_str));

        Ok(Self {
            schema_version: RECEIPT_SCHEMA_VERSION.to_string(),
            timestamp: Utc::now().to_rfc3339(),
            compute_path: compute_path.to_string(),
            backend: backend.to_string(),
            backend_summary: backend_summary.unwrap_or_default(),
            kernels,
            deterministic: std::env::var("BITNET_DETERMINISTIC").is_ok(),
            environment: Self::collect_env_vars(),
            model_info: ModelInfo::default(),
            test_results: TestResults::default(),
            performance_baseline: PerformanceBaseline::default(),
            cross_validation: None,
            parity: None,
            corrections: Vec::new(),
            strict_provenance: None,
        })
    }

    /// Backward-compatible alias for [`Self::generate`] with no backend summary.
    ///
    /// Equivalent to `generate(backend, kernels, None)`. Prefer `generate()` for new code.
    #[deprecated(since = "0.1.1", note = "use generate(backend, kernels, None) instead")]
    pub fn generate_basic(backend: &str, kernels: Vec<String>) -> Result<Self> {
        Self::generate(backend, kernels, None)
    }

    /// Collect relevant environment variables
    fn collect_env_vars() -> HashMap<String, String> {
        let mut env_vars = HashMap::new();

        // Determinism variables
        if let Ok(val) = std::env::var("BITNET_DETERMINISTIC") {
            env_vars.insert("BITNET_DETERMINISTIC".to_string(), val);
        }
        if let Ok(val) = std::env::var("BITNET_SEED") {
            env_vars.insert("BITNET_SEED".to_string(), val);
        }
        if let Ok(val) = std::env::var("RAYON_NUM_THREADS") {
            env_vars.insert("RAYON_NUM_THREADS".to_string(), val);
        }

        // Model path
        if let Ok(val) = std::env::var("BITNET_GGUF") {
            env_vars.insert("BITNET_GGUF".to_string(), val);
        }

        // System info
        env_vars.insert("RUST_VERSION".to_string(), rustc_version_runtime::version().to_string());
        env_vars.insert("BITNET_VERSION".to_string(), env!("CARGO_PKG_VERSION").to_string());
        env_vars.insert(
            "OS".to_string(),
            format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        );

        // Add CPU and GPU fingerprints (best-effort)
        env_vars.insert("CPU_BRAND".to_string(), detect_cpu_brand());
        if let Some(gpu_info) = detect_gpu_info() {
            env_vars.insert("GPU_INFO".to_string(), gpu_info);
        }

        env_vars
    }

    /// Load receipt from JSON file
    ///
    /// # Example
    /// ```no_run
    /// use std::path::Path;
    /// use bitnet_receipts_core::InferenceReceipt;
    ///
    /// let receipt = InferenceReceipt::load(Path::new("ci/inference.json")).unwrap();
    /// assert_eq!(receipt.schema_version, "1.0.0");
    /// ```
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let receipt: InferenceReceipt = serde_json::from_str(&content)?;
        Ok(receipt)
    }

    /// Serialize this receipt to a pretty-printed JSON string.
    ///
    /// Useful for display, logging, or snapshot testing without writing to disk.
    ///
    /// # Example
    /// ```
    /// use bitnet_receipts_core::InferenceReceipt;
    ///
    /// let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    /// let json = receipt.to_json_string().unwrap();
    /// assert!(json.contains("\"schema_version\""));
    /// ```
    pub fn to_json_string(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Save receipt to JSON file
    ///
    /// # AC4 Contract
    /// - Serializes to pretty JSON
    /// - Creates parent directory if it doesn't exist
    /// - Writes atomically (temp file + rename)
    /// - Typically saved to `ci/inference.json`
    ///
    /// # Example
    /// ```no_run
    /// use std::path::Path;
    /// use bitnet_receipts_core::InferenceReceipt;
    ///
    /// let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    /// receipt.save(Path::new("ci/inference.json")).unwrap();
    /// ```
    pub fn save(&self, path: &Path) -> Result<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Serialize to pretty JSON
        let json = serde_json::to_string_pretty(self)?;

        // Atomic write: write to temp file, then rename
        atomic_write(path, json.as_bytes())?;

        Ok(())
    }

    /// Validate receipt against AC9 requirements
    ///
    /// # AC9 Contract
    /// - MUST have `compute_path="real"` (fail if "mock")
    /// - MUST NOT have mock kernels (case-insensitive check)
    /// - MUST have zero failed tests
    /// - MUST pass accuracy tests (if present)
    /// - MUST pass determinism tests (if deterministic mode enabled)
    /// - MUST have valid kernel IDs (hygiene checks)
    ///
    /// # Example
    /// ```no_run
    /// use bitnet_receipts_core::InferenceReceipt;
    ///
    /// let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    /// assert!(receipt.validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<()> {
        // Validate schema version
        self.validate_schema()?;

        // AC9: Check compute path
        self.validate_compute_path()?;

        // AC9: Check for mock kernels and validate kernel ID hygiene
        self.validate_kernel_ids()?;

        // AC9: Check test results
        if self.test_results.failed > 0 {
            return Err(anyhow!("Failed tests detected: {}", self.test_results.failed));
        }

        // AC9: Validate accuracy tests (if present)
        if let Some(ref accuracy) = self.test_results.accuracy_tests {
            if let Some(ref i2s) = accuracy.i2s_accuracy
                && !i2s.passed
            {
                return Err(anyhow!(
                    "I2S accuracy test failed: MSE {} > tolerance {}",
                    i2s.mse,
                    i2s.tolerance
                ));
            }
            if let Some(ref tl1) = accuracy.tl1_accuracy
                && !tl1.passed
            {
                return Err(anyhow!(
                    "TL1 accuracy test failed: MSE {} > tolerance {}",
                    tl1.mse,
                    tl1.tolerance
                ));
            }
            if let Some(ref tl2) = accuracy.tl2_accuracy
                && !tl2.passed
            {
                return Err(anyhow!(
                    "TL2 accuracy test failed: MSE {} > tolerance {}",
                    tl2.mse,
                    tl2.tolerance
                ));
            }
        }

        // AC9: Validate determinism tests (if deterministic mode)
        if self.deterministic
            && let Some(ref det_tests) = self.test_results.determinism_tests
            && !det_tests.identical_sequences
        {
            return Err(anyhow!("Determinism test failed: sequences not identical"));
        }

        // Soft gate: if backend_summary is non-empty, verify it has the expected format.
        if !self.backend_summary.is_empty() && !self.backend_summary.contains("selected=") {
            return Err(anyhow!(
                "backend_summary format invalid: expected to contain \"selected=\", got: {:?}",
                self.backend_summary
            ));
        }

        Ok(())
    }

    /// Validate this receipt as a strict CPU proof.
    ///
    /// This is intentionally stronger than the legacy receipt validator: it
    /// rejects hidden fallbacks, missing selected kernels, mock/diagnostic/dequant
    /// steady-state kernels, non-authoritative loader/tokenizer paths, and any
    /// requested-vs-selected backend/kernel mismatch.
    pub fn validate_strict_cpu_proof(&self) -> Result<()> {
        self.validate()?;

        let provenance = self
            .strict_provenance
            .as_ref()
            .ok_or_else(|| anyhow!("strict CPU proof missing strict_provenance"))?;

        if !is_strict_cpu_backend_label(&provenance.requested_backend) {
            return Err(anyhow!(
                "strict CPU proof requested backend must be a CPU proof label, got {:?}",
                provenance.requested_backend
            ));
        }
        if !is_strict_cpu_backend_label(&provenance.selected_backend) || self.backend != "cpu" {
            return Err(anyhow!(
                "strict CPU proof selected backend mismatch: receipt backend={:?}, selected={:?}",
                self.backend,
                provenance.selected_backend
            ));
        }
        if provenance.fallback_used {
            return Err(anyhow!(
                "strict CPU proof used fallback: {}",
                provenance.fallback_reason.as_deref().unwrap_or("fallback_reason missing")
            ));
        }
        if provenance.fallback_reason.is_some() {
            return Err(anyhow!(
                "strict CPU proof fallback_reason must be absent when fallback_used=false"
            ));
        }

        let loader_mode = provenance
            .loader_mode
            .as_deref()
            .ok_or_else(|| anyhow!("strict CPU proof missing loader_mode"))?;
        if loader_mode != "real_gguf" {
            return Err(anyhow!(
                "strict CPU proof requires loader_mode=real_gguf, got {:?}",
                loader_mode
            ));
        }

        let tokenizer_source = provenance
            .tokenizer_source
            .as_deref()
            .ok_or_else(|| anyhow!("strict CPU proof missing tokenizer_source"))?;
        let tokenizer_source_lc = tokenizer_source.to_ascii_lowercase();
        const DISALLOWED_TOKENIZER_MARKERS: &[&str] =
            &["mock", "fallback", "compat", "guess", "gpt2"];
        if DISALLOWED_TOKENIZER_MARKERS.iter().any(|marker| tokenizer_source_lc.contains(marker)) {
            return Err(anyhow!(
                "strict CPU proof tokenizer_source is not authoritative: {:?}",
                tokenizer_source
            ));
        }
        match provenance.tokenizer_strict {
            Some(true) => {}
            Some(false) => {
                return Err(anyhow!("strict CPU proof requires tokenizer_strict=true"));
            }
            None => {
                return Err(anyhow!("strict CPU proof missing tokenizer_strict"));
            }
        }

        let quant_format = provenance
            .quant_format
            .as_deref()
            .ok_or_else(|| anyhow!("strict CPU proof missing quant_format"))?;
        let quant_lc = quant_format.to_ascii_lowercase();
        if !(quant_lc.contains("i2_s") || quant_lc.contains("qk256")) {
            return Err(anyhow!(
                "strict CPU proof requires QK256/I2_S quant format, got {:?}",
                quant_format
            ));
        }

        let model_hash = self
            .model_info
            .sha256
            .as_deref()
            .ok_or_else(|| anyhow!("strict CPU proof missing model sha256"))?;
        if model_hash.trim().is_empty() {
            return Err(anyhow!("strict CPU proof model sha256 must not be empty"));
        }

        let cpu_model = provenance
            .cpu_model
            .as_deref()
            .ok_or_else(|| anyhow!("strict CPU proof missing cpu_model"))?;
        if cpu_model.trim().is_empty() {
            return Err(anyhow!("strict CPU proof cpu_model must not be empty"));
        }
        if provenance.cpu_features.is_empty() {
            return Err(anyhow!("strict CPU proof missing cpu_features"));
        }

        let phase =
            provenance.phase.as_deref().ok_or_else(|| anyhow!("strict CPU proof missing phase"))?;
        if phase != "prefill" && phase != "decode" {
            return Err(anyhow!(
                "strict CPU proof phase must be prefill or decode, got {:?}",
                phase
            ));
        }
        let prompt_tokens = provenance
            .prompt_tokens
            .ok_or_else(|| anyhow!("strict CPU proof missing prompt_tokens"))?;
        if prompt_tokens == 0 {
            return Err(anyhow!("strict CPU proof prompt_tokens must be greater than zero"));
        }
        let decode_tokens = provenance
            .decode_tokens
            .ok_or_else(|| anyhow!("strict CPU proof missing decode_tokens"))?;
        if phase == "decode" && decode_tokens == 0 {
            return Err(anyhow!("strict CPU proof decode phase requires decode_tokens > 0"));
        }

        let selected_kernel = provenance
            .selected_kernel
            .as_deref()
            .ok_or_else(|| anyhow!("strict CPU proof missing selected_kernel"))?;
        let cpu_features_lc: Vec<String> =
            provenance.cpu_features.iter().map(|feature| feature.to_ascii_lowercase()).collect();
        if selected_kernel.to_ascii_lowercase().contains("avx2")
            && !(cpu_features_lc.iter().any(|feature| feature == "avx2")
                && cpu_features_lc.iter().any(|feature| feature == "fma"))
        {
            return Err(anyhow!(
                "strict CPU proof selected AVX2 kernel without avx2/fma CPU features"
            ));
        }
        if selected_kernel.to_ascii_lowercase().contains("avx512")
            && !cpu_features_lc.iter().any(|feature| feature == "avx512")
        {
            return Err(anyhow!(
                "strict CPU proof selected AVX-512 kernel without avx512 CPU feature"
            ));
        }
        if selected_kernel.to_ascii_lowercase().contains("neon")
            && !cpu_features_lc.iter().any(|feature| feature == "neon")
        {
            return Err(anyhow!("strict CPU proof selected NEON kernel without neon CPU feature"));
        }
        if let Some(requested_kernel) = provenance.requested_kernel.as_deref()
            && requested_kernel != selected_kernel
        {
            return Err(anyhow!(
                "strict CPU proof kernel mismatch: requested={:?}, selected={:?}",
                requested_kernel,
                selected_kernel
            ));
        }
        if !self.kernels.iter().any(|kernel| kernel == selected_kernel) {
            return Err(anyhow!(
                "strict CPU proof selected_kernel {:?} not present in kernels {:?}",
                selected_kernel,
                self.kernels
            ));
        }

        const DISALLOWED_KERNEL_MARKERS: &[&str] =
            &["mock", "diagnostic", "compat", "fallback", "dense_dequant", "full_dequant"];
        for kernel in &self.kernels {
            let kernel_lc = kernel.to_ascii_lowercase();
            if DISALLOWED_KERNEL_MARKERS.iter().any(|marker| kernel_lc.contains(marker)) {
                return Err(anyhow!("strict CPU proof contains disallowed kernel {:?}", kernel));
            }
        }

        Ok(())
    }

    /// Validate schema version
    ///
    /// # Requirements
    /// - Schema version must be "1.0.0"
    ///
    /// # Example
    /// ```no_run
    /// use bitnet_receipts_core::InferenceReceipt;
    ///
    /// let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    /// assert!(receipt.validate_schema().is_ok());
    /// ```
    pub fn validate_schema(&self) -> Result<()> {
        if self.schema_version != "1.0.0" {
            return Err(anyhow!(
                "Invalid schema version: {} (expected '1.0.0')",
                self.schema_version
            ));
        }
        Ok(())
    }

    /// Validate compute path
    ///
    /// # Requirements
    /// - Compute path must be "real" (not "mock")
    ///
    /// # Example
    /// ```no_run
    /// use bitnet_receipts_core::InferenceReceipt;
    ///
    /// let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    /// assert!(receipt.validate_compute_path().is_ok());
    /// ```
    pub fn validate_compute_path(&self) -> Result<()> {
        validate_honest_compute_path(&self.compute_path).map_err(Into::into)
    }

    /// Validate kernel IDs
    ///
    /// # Requirements
    /// - Kernel array must be non-empty
    /// - No kernel ID can be empty string
    /// - No kernel ID can be whitespace-only
    /// - Each kernel ID must be ≤ 128 characters
    /// - Total kernel count must be ≤ 10,000
    /// - No kernel ID can contain "mock" (case-insensitive)
    ///
    /// # Example
    /// ```no_run
    /// use bitnet_receipts_core::InferenceReceipt;
    ///
    /// let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    /// assert!(receipt.validate_kernel_ids().is_ok());
    /// ```
    pub fn validate_kernel_ids(&self) -> Result<()> {
        validate_honest_kernel_ids(self.kernels.iter().map(String::as_str)).map_err(Into::into)
    }

    /// Builder for test results
    pub fn with_test_results(mut self, test_results: TestResults) -> Self {
        self.test_results = test_results;
        self
    }

    /// Builder for model info
    pub fn with_model_info(mut self, model_info: ModelInfo) -> Self {
        self.model_info = model_info;
        self
    }

    /// Builder for performance baseline
    pub fn with_performance_baseline(mut self, performance: PerformanceBaseline) -> Self {
        self.performance_baseline = performance;
        self
    }

    /// Builder for cross-validation
    pub fn with_cross_validation(mut self, cross_val: CrossValidation) -> Self {
        self.cross_validation = Some(cross_val);
        self
    }

    /// Builder for parity metadata (AC4)
    pub fn with_parity(mut self, parity: ParityMetadata) -> Self {
        self.parity = Some(parity);
        self
    }

    /// Builder for strict CPU proof provenance.
    pub fn with_strict_provenance(mut self, provenance: StrictInferenceProvenance) -> Self {
        self.strict_provenance = Some(provenance);
        self
    }

    /// Builder for corrections
    pub fn with_corrections(mut self, corrections: Vec<CorrectionRecord>) -> Self {
        self.corrections = corrections;
        self
    }

    /// Add a single correction record
    pub fn add_correction(&mut self, correction: CorrectionRecord) {
        self.corrections.push(correction);
    }
}

fn is_strict_cpu_backend_label(label: &str) -> bool {
    matches!(label, "cpu" | "apple-m4-cpu-neon")
}

/// Detect CPU brand string (best-effort).
/// Linux: reads `/proc/cpuinfo` model name; otherwise returns arch.
fn detect_cpu_brand() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in content.lines() {
                if line.starts_with("model name")
                    && let Some(brand) = line.split(':').nth(1)
                {
                    return brand.trim().to_string();
                }
            }
        }
    }
    std::env::consts::ARCH.to_string()
}

/// Detect GPU information (best-effort)
///
/// Uses bitnet-kernels GPU utilities to detect available GPUs.
/// Returns GPU name and compute capability if available.
fn detect_gpu_info() -> Option<String> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        use bitnet_kernels::gpu;
        // Try to get first CUDA device info if available
        if let Ok(devices) = gpu::list_cuda_devices() {
            if let Some(device) = devices.first() {
                return Some(format!(
                    "{} (CC: {}.{})",
                    device.name, device.compute_capability.0, device.compute_capability.1
                ));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_receipt_generation_real_path() {
        let receipt = InferenceReceipt::generate(
            "cpu",
            vec!["i2s_gemv".to_string(), "rope_apply".to_string()],
            None,
        )
        .unwrap();

        assert_eq!(receipt.schema_version, "1.0.0");
        assert_eq!(receipt.compute_path, "real");
        assert_eq!(receipt.backend, "cpu");
        assert!(receipt.kernels.contains(&"i2s_gemv".to_string()));
    }

    #[test]
    fn test_receipt_generation_mock_detected() {
        let receipt = InferenceReceipt::generate(
            "cpu",
            vec!["mock_gemv".to_string(), "i2s_gemv".to_string()],
            None,
        )
        .unwrap();

        assert_eq!(receipt.compute_path, "mock");
    }

    #[test]
    fn test_receipt_validation_passes() {
        let receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();

        assert!(receipt.validate().is_ok());
    }

    #[test]
    fn test_receipt_validation_fails_mock_path() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();

        receipt.compute_path = "mock".to_string();
        assert!(receipt.validate().is_err());
    }

    #[test]
    fn test_receipt_validation_fails_mock_kernels() {
        let receipt =
            InferenceReceipt::generate("cpu", vec!["mock_gemv".to_string()], None).unwrap();

        assert!(receipt.validate().is_err());
    }

    #[test]
    fn test_receipt_validation_fails_failed_tests() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();

        receipt.test_results.failed = 1;
        assert!(receipt.validate().is_err());
    }

    #[test]
    fn test_receipt_with_corrections() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();

        // Add a correction record
        let correction = CorrectionRecord {
            layer: "model.layers.0.input_layernorm.weight".to_string(),
            correction_type: "ln_gamma_rescale_rms".to_string(),
            rms_before: Some(0.5),
            rms_after: Some(1.0),
            factor: Some(2.0),
            policy_fingerprint: "BITNET_FIX_LN_SCALE=1".to_string(),
            metadata: None,
        };
        receipt.add_correction(correction.clone());

        // Verify correction is present
        assert_eq!(receipt.corrections.len(), 1);
        assert_eq!(receipt.corrections[0].layer, "model.layers.0.input_layernorm.weight");
        assert_eq!(receipt.corrections[0].correction_type, "ln_gamma_rescale_rms");
        assert_eq!(receipt.corrections[0].rms_before, Some(0.5));
        assert_eq!(receipt.corrections[0].rms_after, Some(1.0));
        assert_eq!(receipt.corrections[0].factor, Some(2.0));
        assert_eq!(receipt.corrections[0].policy_fingerprint, "BITNET_FIX_LN_SCALE=1");
    }

    #[test]
    fn test_receipt_empty_corrections_by_default() {
        let receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        assert!(receipt.corrections.is_empty(), "Corrections should be empty by default");
    }

    #[test]
    fn test_receipt_serialization_with_corrections() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();

        let correction = CorrectionRecord {
            layer: "test.layer".to_string(),
            correction_type: "ln_gamma_rescale_rms".to_string(),
            rms_before: Some(0.75),
            rms_after: Some(1.0),
            factor: Some(1.33),
            policy_fingerprint: "BITNET_FIX_LN_SCALE=1".to_string(),
            metadata: None,
        };
        receipt.add_correction(correction);

        // Serialize to JSON
        let json = serde_json::to_string_pretty(&receipt).unwrap();

        // Verify JSON contains corrections
        assert!(json.contains("corrections"));
        assert!(json.contains("test.layer"));
        assert!(json.contains("ln_gamma_rescale_rms"));
        assert!(json.contains("BITNET_FIX_LN_SCALE=1"));

        // Deserialize and verify
        let deserialized: InferenceReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.corrections.len(), 1);
        assert_eq!(deserialized.corrections[0].layer, "test.layer");
    }

    #[test]
    fn test_receipt_with_model_metadata() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();

        // Add model SHA256 and correction digest
        receipt.model_info.sha256 = Some("abc123def456".to_string());
        receipt.model_info.effective_correction_digest = Some("digest789".to_string());

        // Serialize and verify
        let json = serde_json::to_string_pretty(&receipt).unwrap();
        assert!(json.contains("sha256"));
        assert!(json.contains("abc123def456"));
        assert!(json.contains("effective_correction_digest"));
        assert!(json.contains("digest789"));
    }

    /// Test validate_schema with invalid version
    #[test]
    fn test_validate_schema_invalid_version() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.schema_version = "2.0.0".to_string();

        let result = receipt.validate_schema();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid schema version"));
    }

    /// Test validate_schema with valid version
    #[test]
    fn test_validate_schema_valid() {
        let receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        assert!(receipt.validate_schema().is_ok());
    }

    /// Test validate_compute_path with invalid path
    #[test]
    fn test_validate_compute_path_invalid() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.compute_path = "mock".to_string();

        let result = receipt.validate_compute_path();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid compute_path"));
    }

    /// Test validate_compute_path with valid path
    #[test]
    fn test_validate_compute_path_valid() {
        let receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        assert!(receipt.validate_compute_path().is_ok());
    }

    /// Test validate_kernel_ids with empty array
    #[test]
    fn test_validate_kernel_ids_empty_array() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = vec![];

        let result = receipt.validate_kernel_ids();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Kernel array is empty"));
    }

    /// Test validate_kernel_ids with empty string
    #[test]
    fn test_validate_kernel_ids_empty_string() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = vec!["".to_string()];

        let result = receipt.validate_kernel_ids();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Empty kernel ID"));
    }

    /// Test validate_kernel_ids with whitespace-only string
    #[test]
    fn test_validate_kernel_ids_whitespace_only() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = vec!["   ".to_string()];

        let result = receipt.validate_kernel_ids();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Whitespace-only kernel ID"));
    }

    /// Test validate_kernel_ids with excessive length
    #[test]
    fn test_validate_kernel_ids_excessive_length() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = vec!["a".repeat(129)];

        let result = receipt.validate_kernel_ids();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds 128 characters"));
    }

    /// Test validate_kernel_ids at exact 128 character boundary (should pass)
    #[test]
    fn test_validate_kernel_ids_exact_128_chars() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = vec!["a".repeat(128)];

        assert!(receipt.validate_kernel_ids().is_ok());
    }

    /// Test validate_kernel_ids with excessive count
    #[test]
    fn test_validate_kernel_ids_excessive_count() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = vec!["kernel".to_string(); 10_001];

        let result = receipt.validate_kernel_ids();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds 10,000 limit"));
    }

    /// Test validate_kernel_ids at exact 10,000 count boundary (should pass)
    #[test]
    fn test_validate_kernel_ids_exact_10k_count() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = vec!["kernel".to_string(); 10_000];

        assert!(receipt.validate_kernel_ids().is_ok());
    }

    /// Test validate_kernel_ids with mock kernel (case-insensitive)
    #[test]
    fn test_validate_kernel_ids_mock_kernel() {
        let test_cases = vec!["mock_kernel", "MOCK_kernel", "kernel_mock", "kernel_MOCK_suffix"];

        for kernel_id in test_cases {
            let mut receipt =
                InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
            receipt.kernels = vec![kernel_id.to_string()];

            let result = receipt.validate_kernel_ids();
            assert!(result.is_err(), "Kernel ID '{}' should be rejected as mock", kernel_id);
            assert!(result.unwrap_err().to_string().contains("Mock kernel detected"));
        }
    }

    /// Test validate_kernel_ids with mixed valid and invalid kernels
    #[test]
    fn test_validate_kernel_ids_mixed_kernels() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = vec!["valid_kernel".to_string(), "".to_string()];

        let result = receipt.validate_kernel_ids();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Empty kernel ID at index 1"));
    }

    /// Test validate_kernel_ids with valid realistic CPU kernels
    #[test]
    fn test_validate_kernel_ids_valid_cpu_kernels() {
        let receipt = InferenceReceipt::generate(
            "cpu",
            vec![
                "i2s_cpu_quantized_matmul".to_string(),
                "tl1_lut_dequant_forward".to_string(),
                "tl2_lut_backward".to_string(),
                "cpu_attention_qkvo".to_string(),
            ],
            None,
        )
        .unwrap();

        assert!(receipt.validate_kernel_ids().is_ok());
    }

    /// Test validate_kernel_ids with valid realistic GPU kernels
    #[test]
    fn test_validate_kernel_ids_valid_gpu_kernels() {
        let receipt = InferenceReceipt::generate(
            "cuda",
            vec![
                "gemm_gpu_fp16".to_string(),
                "cuda_i2s_quantize".to_string(),
                "gpu_attention_flash".to_string(),
            ],
            None,
        )
        .unwrap();

        assert!(receipt.validate_kernel_ids().is_ok());
    }

    fn strict_cpu_proof_receipt() -> InferenceReceipt {
        let mut receipt = InferenceReceipt::generate(
            "cpu",
            vec!["qk256-avx2-gemv".to_string(), "rope_apply".to_string()],
            Some("requested=cpu detected=[cpu] selected=cpu".to_string()),
        )
        .unwrap();
        receipt.model_info.sha256 = Some("abc123def4567890".to_string());
        receipt.with_strict_provenance(StrictInferenceProvenance {
            requested_backend: "cpu".to_string(),
            selected_backend: "cpu".to_string(),
            requested_kernel: Some("qk256-avx2-gemv".to_string()),
            selected_kernel: Some("qk256-avx2-gemv".to_string()),
            loader_mode: Some("real_gguf".to_string()),
            tokenizer_source: Some("embedded_gguf".to_string()),
            tokenizer_strict: Some(true),
            model_family: Some("bitnet".to_string()),
            quant_format: Some("QK256/I2_S".to_string()),
            cpu_model: Some("Intel Core i5-8250U".to_string()),
            cpu_features: vec!["avx2".to_string(), "fma".to_string()],
            thread_count: Some(1),
            fallback_used: false,
            fallback_reason: None,
            prompt_tokens: Some(512),
            decode_tokens: Some(128),
            phase: Some("decode".to_string()),
            latency_p50_ms: Some(12.0),
            latency_p95_ms: Some(15.0),
            decode_tps: Some(80.0),
        })
    }

    #[test]
    fn test_validate_strict_cpu_proof_accepts_authoritative_lane() {
        let receipt = strict_cpu_proof_receipt();

        assert!(receipt.validate_strict_cpu_proof().is_ok());
    }

    #[test]
    fn test_validate_strict_cpu_proof_accepts_apple_cpu_neon_label() {
        let mut receipt = strict_cpu_proof_receipt();
        let provenance = receipt.strict_provenance.as_mut().unwrap();
        provenance.requested_backend = "apple-m4-cpu-neon".to_string();
        provenance.selected_backend = "apple-m4-cpu-neon".to_string();
        provenance.selected_kernel = Some("i2_s-scalar-reference".to_string());
        provenance.requested_kernel = Some("i2_s-scalar-reference".to_string());
        provenance.quant_format = Some("I2_S".to_string());
        provenance.cpu_features = vec!["neon".to_string()];
        receipt.kernels = vec!["i2_s-scalar-reference".to_string()];

        assert!(receipt.validate_strict_cpu_proof().is_ok());
    }

    #[test]
    fn test_validate_strict_cpu_proof_rejects_hidden_fallback() {
        let mut receipt = strict_cpu_proof_receipt();
        let provenance = receipt.strict_provenance.as_mut().unwrap();
        provenance.fallback_used = true;
        provenance.fallback_reason = Some("requested AVX2 but selected scalar".to_string());

        let err = receipt.validate_strict_cpu_proof().unwrap_err().to_string();
        assert!(err.contains("used fallback"), "unexpected error: {err}");
    }

    #[test]
    fn test_validate_strict_cpu_proof_rejects_kernel_mismatch() {
        let mut receipt = strict_cpu_proof_receipt();
        receipt.strict_provenance.as_mut().unwrap().selected_kernel =
            Some("qk256-scalar-gemv".to_string());

        let err = receipt.validate_strict_cpu_proof().unwrap_err().to_string();
        assert!(err.contains("kernel mismatch"), "unexpected error: {err}");
    }

    #[test]
    fn test_validate_strict_cpu_proof_rejects_non_authoritative_tokenizer() {
        let mut receipt = strict_cpu_proof_receipt();
        receipt.strict_provenance.as_mut().unwrap().tokenizer_source =
            Some("gpt2_compat_fallback".to_string());

        let err = receipt.validate_strict_cpu_proof().unwrap_err().to_string();
        assert!(err.contains("tokenizer_source"), "unexpected error: {err}");
    }

    #[test]
    fn test_validate_strict_cpu_proof_rejects_non_real_loader() {
        let mut receipt = strict_cpu_proof_receipt();
        receipt.strict_provenance.as_mut().unwrap().loader_mode =
            Some("compatibility_fallback".to_string());

        let err = receipt.validate_strict_cpu_proof().unwrap_err().to_string();
        assert!(err.contains("loader_mode=real_gguf"), "unexpected error: {err}");
    }

    #[test]
    fn test_validate_strict_cpu_proof_rejects_non_strict_tokenizer_resolution() {
        let mut receipt = strict_cpu_proof_receipt();
        receipt.strict_provenance.as_mut().unwrap().tokenizer_strict = Some(false);

        let err = receipt.validate_strict_cpu_proof().unwrap_err().to_string();
        assert!(err.contains("tokenizer_strict=true"), "unexpected error: {err}");
    }

    #[test]
    fn test_validate_strict_cpu_proof_rejects_missing_model_hash() {
        let mut receipt = strict_cpu_proof_receipt();
        receipt.model_info.sha256 = None;

        let err = receipt.validate_strict_cpu_proof().unwrap_err().to_string();
        assert!(err.contains("model sha256"), "unexpected error: {err}");
    }

    #[test]
    fn test_validate_strict_cpu_proof_rejects_missing_phase() {
        let mut receipt = strict_cpu_proof_receipt();
        receipt.strict_provenance.as_mut().unwrap().phase = None;

        let err = receipt.validate_strict_cpu_proof().unwrap_err().to_string();
        assert!(err.contains("missing phase"), "unexpected error: {err}");
    }

    #[test]
    fn test_validate_strict_cpu_proof_rejects_decode_phase_without_generated_tokens() {
        let mut receipt = strict_cpu_proof_receipt();
        receipt.strict_provenance.as_mut().unwrap().decode_tokens = Some(0);

        let err = receipt.validate_strict_cpu_proof().unwrap_err().to_string();
        assert!(err.contains("decode_tokens > 0"), "unexpected error: {err}");
    }

    #[test]
    fn test_validate_strict_cpu_proof_rejects_avx2_kernel_without_fma_feature() {
        let mut receipt = strict_cpu_proof_receipt();
        receipt.strict_provenance.as_mut().unwrap().cpu_features = vec!["avx2".to_string()];

        let err = receipt.validate_strict_cpu_proof().unwrap_err().to_string();
        assert!(err.contains("avx2/fma"), "unexpected error: {err}");
    }

    /// Test that environment variable collection returns non-empty HashMap with valid content
    /// Kills 3 mutation survivors in receipts.rs:221 (empty HashMap, single empty entry, dummy values)
    #[test]
    fn test_receipt_env_vars_content_validation() {
        // Set test environment variables to ensure we have predictable content
        // SAFETY: This is test code running in isolation. We clean up at the end.
        unsafe {
            std::env::set_var("BITNET_DETERMINISTIC", "1");
            std::env::set_var("BITNET_SEED", "42");
        }

        let vars = InferenceReceipt::collect_env_vars();

        // Kill survivor 1: empty HashMap return
        assert!(!vars.is_empty(), "Environment variables should not be empty");

        // Kill survivor 2 & 3: single empty entry or dummy values
        for (key, value) in &vars {
            assert!(!key.is_empty(), "Environment variable key should not be empty");
            assert!(!value.is_empty(), "Environment variable value should not be empty");

            // Validate actual content - keys should be recognizable environment variables
            assert!(
                key.starts_with("BITNET_")
                    || key.starts_with("RAYON_")
                    || key == "RUST_VERSION"
                    || key == "OS"
                    || key == "CPU_BRAND"
                    || key == "GPU_INFO",
                "Key '{}' should be a valid BitNet/Rayon/Rust environment variable",
                key
            );
        }

        // Verify specific expected variables are present with correct values
        assert!(vars.contains_key("BITNET_DETERMINISTIC"), "Should contain BITNET_DETERMINISTIC");
        assert_eq!(
            vars.get("BITNET_DETERMINISTIC"),
            Some(&"1".to_string()),
            "BITNET_DETERMINISTIC should have value '1'"
        );

        assert!(vars.contains_key("BITNET_SEED"), "Should contain BITNET_SEED when set");
        assert_eq!(
            vars.get("BITNET_SEED"),
            Some(&"42".to_string()),
            "BITNET_SEED should have value '42'"
        );

        assert!(vars.contains_key("RUST_VERSION"), "Should always contain RUST_VERSION");
        let rust_version = vars.get("RUST_VERSION").unwrap();
        assert!(
            rust_version.contains('.'),
            "RUST_VERSION should be a valid version string with dots"
        );

        assert!(vars.contains_key("BITNET_VERSION"), "Should always contain BITNET_VERSION");
        assert!(vars.contains_key("OS"), "Should always contain OS");

        // Clean up test environment variables
        // SAFETY: This is test cleanup code running in isolation.
        unsafe {
            std::env::remove_var("BITNET_DETERMINISTIC");
            std::env::remove_var("BITNET_SEED");
        }
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // generate_basic always produces a receipt that passes schema validation.
    proptest! {
        #[test]
        fn generate_basic_passes_schema_validation(
            backend in prop_oneof![Just("cpu"), Just("cuda"), Just("gpu")],
            kernel_count in 0usize..=8,
        ) {
            let kernels: Vec<String> = (0..kernel_count)
                .map(|i| format!("kernel_{i}"))
                .collect();
            let receipt = InferenceReceipt::generate(backend, kernels, None).unwrap();
            prop_assert!(
                receipt.validate_schema().is_ok(),
                "schema validation failed for backend={:?}",
                backend
            );
        }
    }

    // generate_basic with compute_path "real" always passes validate_compute_path.
    proptest! {
        #[test]
        fn generate_basic_has_real_compute_path(
            backend in "[a-z]{1,8}",
        ) {
            let receipt = InferenceReceipt::generate(&backend, vec!["k".to_string()], None)
                .unwrap();
            prop_assert_eq!(receipt.compute_path.as_str(), "real");
            prop_assert!(receipt.validate_compute_path().is_ok());
        }
    }

    // validate_kernel_ids accepts any kernel IDs that are non-empty, ≤128 chars, and
    // do not contain the "mock" substring (which the honest-compute policy forbids).
    proptest! {
        #[test]
        fn validate_kernel_ids_accepts_valid_ids(
            ids in prop::collection::vec(
                "[a-z_]{1,32}".prop_filter("must not contain 'mock'", |s| !s.contains("mock")),
                1..=16
            ),
        ) {
            let mut receipt =
                InferenceReceipt::generate("cpu", ids.clone(), None).unwrap();
            receipt.kernels = ids;
            prop_assert!(
                receipt.validate_kernel_ids().is_ok(),
                "expected Ok for valid kernel IDs"
            );
        }
    }

    // validate_kernel_ids rejects any slice that contains an empty string.
    proptest! {
        #[test]
        fn validate_kernel_ids_rejects_empty_id(
            prefix in prop::collection::vec("[a-z_]{1,16}", 0..=8),
            suffix in prop::collection::vec("[a-z_]{1,16}", 0..=8),
        ) {
            let mut kernels = prefix;
            kernels.push(String::new()); // inject empty id
            kernels.extend(suffix);
            let mut receipt =
                InferenceReceipt::generate("cpu", vec!["ok".to_string()], None).unwrap();
            receipt.kernels = kernels;
            prop_assert!(
                receipt.validate_kernel_ids().is_err(),
                "expected Err when empty kernel ID present"
            );
        }
    }
}
