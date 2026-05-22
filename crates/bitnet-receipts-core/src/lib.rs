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
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

mod artifact_kinds;
mod schema;

pub use artifact_kinds::{
    CUDA_PLANNER_RECEIPT_VERSION, DENSE_GGUF_ALL_LAYER_EXECUTION_PLAN_ARTIFACT_KIND,
    DENSE_GGUF_ATTENTION_SCORE_CUDA_PARITY_ARTIFACT_KIND,
    DENSE_GGUF_ATTENTION_SCORE_FIXTURE_ARTIFACT_KIND,
    DENSE_GGUF_ATTENTION_SOFTMAX_CUDA_PARITY_ARTIFACT_KIND,
    DENSE_GGUF_ATTENTION_SOFTMAX_FIXTURE_ARTIFACT_KIND,
    DENSE_GGUF_ATTENTION_V_MIX_CUDA_PARITY_ARTIFACT_KIND,
    DENSE_GGUF_ATTENTION_V_MIX_FIXTURE_ARTIFACT_KIND,
    DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND, DENSE_GGUF_KV_CACHE_POLICY_ARTIFACT_KIND,
    DENSE_GGUF_LINEAR_CUDA_PARITY_ARTIFACT_KIND, DENSE_GGUF_LINEAR_FIXTURE_ARTIFACT_KIND,
    DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND,
    DENSE_GGUF_MLP_ACTIVATION_CUDA_PARITY_ARTIFACT_KIND,
    DENSE_GGUF_MLP_ACTIVATION_FIXTURE_ARTIFACT_KIND,
    DENSE_GGUF_MODEL_BOUNDARY_FIXTURES_ARTIFACT_KIND, DENSE_GGUF_NORM_CUDA_PARITY_ARTIFACT_KIND,
    DENSE_GGUF_NORM_FIXTURE_ARTIFACT_KIND, DENSE_GGUF_ONE_LAYER_CPU_REFERENCE_ARTIFACT_KIND,
    DENSE_GGUF_ONE_LAYER_CUDA_INTEGRATED_PARITY_ARTIFACT_KIND,
    DENSE_GGUF_ONE_LAYER_EXECUTION_PLAN_ARTIFACT_KIND,
    DENSE_GGUF_QWEN_ASK_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    DENSE_GGUF_QWEN_CHAT_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    DENSE_GGUF_QWEN_ONE_TOKEN_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    DENSE_GGUF_QWEN_SHORT_DECODE_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    DENSE_GGUF_QWEN_WARM_DECODE_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    DENSE_GGUF_QWEN_WARM_SESSION_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    DENSE_GGUF_ROPE_CUDA_PARITY_ARTIFACT_KIND, DENSE_GGUF_SAMPLING_POLICY_ARTIFACT_KIND,
    DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND, DENSE_REGULAR_LLM_MODEL_CLASS,
    M4_RUN_IDENTITY_CONTRACT_VERSION, RECEIPT_SCHEMA, RECEIPT_SCHEMA_VERSION,
    SERVER_SHARED_ENGINE_CHAT_COMPLETION_RECEIPT_KIND,
};
pub use schema::{
    AccuracyMetric, AccuracyTestResults, CacheEfficiency, CrossValidation, DeterminismTestResults,
    KVCacheTestResults, M4RunIdentity, M4RunIdentityBackend, M4RunIdentityBinary,
    M4RunIdentityCommand, M4RunIdentityEvidence, M4RunIdentityGit, M4RunIdentityModel,
    M4RunIdentityOs, M4RunIdentityPromptTemplate, M4RunIdentityTiming, M4RunIdentityTokenizer,
    ModelInfo, ParityMetadata, PerformanceBaseline, StrictInferenceProvenance, TestResults,
};

use artifact_kinds::{
    BITNET_B158_2B_4T_I2S_MODEL_FILE, BITNET_B158_2B_4T_I2S_MODEL_ID,
    BITNET_B158_2B_4T_I2S_MODEL_SHA256,
    DENSE_ONE_LAYER_ATTENTION_V_MIX_FIXTURE_GAP_CANDIDATE_ORDER,
    DENSE_ONE_LAYER_GAP_CANDIDATE_ORDER, DENSE_ONE_LAYER_NO_REMAINING_GAP_CANDIDATE_ORDER,
    DENSE_ONE_LAYER_REMAINING_GAP_CANDIDATE_ORDER, QWEN3_06B_INSTRUCT_Q8_0_MODEL_FILE,
    QWEN3_06B_INSTRUCT_Q8_0_MODEL_ID, QWEN3_06B_INSTRUCT_Q8_0_MODEL_SHA256,
    QWEN25_05B_INSTRUCT_Q8_0_MODEL_FILE, QWEN25_05B_INSTRUCT_Q8_0_MODEL_ID,
    QWEN25_05B_INSTRUCT_Q8_0_MODEL_SHA256,
};

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

    validate_dense_regular_llm_execution_plan(receipt)?;

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

/// Validate a strict RTX 5070 Ti server shared-engine chat-completion receipt.
///
/// This validates the exact-profile fields required before a server smoke can
/// be considered for promotion. It deliberately keeps `server_ready_claimed`
/// false; a later promotion validator can narrow this further when model
/// coverage actually sets `server_ready=true` for an exact profile.
pub fn validate_server_shared_engine_chat_completion_receipt_json(receipt: &Value) -> Result<()> {
    require_string_eq(receipt, "receipt_kind", SERVER_SHARED_ENGINE_CHAT_COMPLETION_RECEIPT_KIND)?;
    require_string_eq(receipt, "runtime_path", "shared_local_inference_engine")?;
    require_string_eq(receipt, "runtime_api", "cuda")?;
    require_string_eq(receipt, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_bool_eq(receipt, "fallback_used", false)?;
    require_bool_eq(receipt, "simulated_inference", false)?;
    require_bool_eq(receipt, "generated_text_non_empty", true)?;
    require_bool_eq(receipt, "server_smoke_response_claimed", true)?;
    require_bool_eq(receipt, "server_ready_claimed", false)?;
    require_bool_eq(receipt, "speedup_claim", false)?;
    require_bool_eq(receipt, "full_cuda_residency_claimed", false)?;
    require_string_non_empty(receipt, "request_id")?;
    require_string_non_empty(receipt, "prompt_template")?;
    require_string_non_empty(receipt, "tokenizer_authority")?;
    require_string_non_empty(receipt, "prompt_authority")?;
    require_positive_u64(receipt, "prompt_tokens")?;
    require_positive_u64(receipt, "completion_tokens")?;
    required_u64(receipt, "total_ms")?;

    let model_identity = object_field(receipt, "model_identity")?;
    require_string_non_empty(model_identity, "model_id")?;
    require_string_non_empty(model_identity, "requested_model")?;
    require_string_non_empty(model_identity, "active_model_id")?;
    require_string_non_empty(model_identity, "active_model_path")?;
    require_sha256(model_identity, "model_sha256")?;
    require_sha256(receipt, "model_sha256")?;
    require_same_string(receipt, "model_sha256", model_identity, "model_sha256", "model_sha256")?;
    require_same_string(
        receipt,
        "requested_model",
        model_identity,
        "requested_model",
        "requested_model",
    )?;
    require_same_string(
        model_identity,
        "model_id",
        model_identity,
        "requested_model",
        "model_identity.model_id",
    )?;
    require_same_string(
        receipt,
        "active_model_id",
        model_identity,
        "active_model_id",
        "active_model_id",
    )?;
    require_same_string(
        receipt,
        "active_model_path",
        model_identity,
        "active_model_path",
        "active_model_path",
    )?;

    let endpoint = object_field(receipt, "endpoint_profile")?;
    require_string_eq(endpoint, "endpoint", "/v1/chat/completions")?;
    require_string_eq(endpoint, "method", "POST")?;
    require_string_non_empty(endpoint, "request_profile")?;
    let endpoint_streaming = object_field(endpoint, "streaming")?
        .as_bool()
        .ok_or_else(|| anyhow!("field `streaming` must be a bool"))?;
    let receipt_streaming = object_field(receipt, "streaming")?
        .as_bool()
        .ok_or_else(|| anyhow!("field `streaming` must be a bool"))?;
    if endpoint_streaming != receipt_streaming {
        return Err(anyhow!(
            "`streaming` must match between `endpoint_profile` and top-level receipt"
        ));
    }
    require_positive_u64(endpoint, "message_count")?;

    let generation_policy = object_field(receipt, "generation_policy")?;
    require_positive_u64(generation_policy, "max_tokens")?;
    require_non_negative_number(generation_policy, "temperature")?;
    require_positive_number(generation_policy, "top_p")?;
    require_string_non_empty(generation_policy, "decoding")?;

    match required_string(receipt, "selected_route")? {
        "dense_regular_llm_cuda" => {
            validate_dense_qwen_server_shared_engine_receipt(receipt, model_identity)?
        }
        "bitnet_qk256_cuda" => {
            validate_bitnet_qk256_server_shared_engine_receipt(receipt, model_identity)?
        }
        route => {
            return Err(anyhow!(
                "server shared-engine receipt selected_route `{route}` is not an accepted exact-profile server-smoke route"
            ));
        }
    }

    let quality = object_field(receipt, "quality_gate")?;
    require_string_non_empty(quality, "gate")?;
    require_bool_eq(quality, "passed", true)?;
    require_bool_eq(quality, "generated_text_non_empty", true)?;
    require_bool_eq(quality, "utf8_valid", true)?;
    require_bool_eq(quality, "broad_chat_quality_claimed", false)?;

    Ok(())
}

fn validate_dense_qwen_server_shared_engine_receipt(
    receipt: &Value,
    model_identity: &Value,
) -> Result<()> {
    let (model_id, model_sha256, model_coverage_row) = match required_string(
        model_identity,
        "model_id",
    )? {
        QWEN25_05B_INSTRUCT_Q8_0_MODEL_ID => (
            QWEN25_05B_INSTRUCT_Q8_0_MODEL_ID,
            QWEN25_05B_INSTRUCT_Q8_0_MODEL_SHA256,
            "dense_qwen25_05b_q8_cuda",
        ),
        QWEN3_06B_INSTRUCT_Q8_0_MODEL_ID => (
            QWEN3_06B_INSTRUCT_Q8_0_MODEL_ID,
            QWEN3_06B_INSTRUCT_Q8_0_MODEL_SHA256,
            "dense_qwen3_06b_q8_candidate",
        ),
        model_id => {
            return Err(anyhow!(
                "dense server shared-engine receipt model_id `{model_id}` is not an accepted exact-profile dense Qwen model"
            ));
        }
    };

    require_string_eq(model_identity, "model_id", model_id)?;
    require_string_eq(model_identity, "model_sha256", model_sha256)?;
    require_string_eq(receipt, "model_sha256", model_sha256)?;
    require_string_eq(receipt, "model_coverage_row", model_coverage_row)?;
    require_string_eq(receipt, "model_coverage_tier", "product_cli_ready")?;
    require_bool_eq(receipt, "dense_regular_llm_cuda_inference_claimed", true)?;
    require_bool_eq(receipt, "bitnet_packed_i2s_qk256_proof", false)?;
    Ok(())
}

fn validate_bitnet_qk256_server_shared_engine_receipt(
    receipt: &Value,
    model_identity: &Value,
) -> Result<()> {
    require_string_eq(model_identity, "model_id", BITNET_B158_2B_4T_I2S_MODEL_ID)?;
    require_string_eq(model_identity, "model_sha256", BITNET_B158_2B_4T_I2S_MODEL_SHA256)?;
    require_string_eq(receipt, "model_sha256", BITNET_B158_2B_4T_I2S_MODEL_SHA256)?;
    let active_model_path = required_string(model_identity, "active_model_path")?;
    let normalized_path = active_model_path.replace('\\', "/");
    if !normalized_path.ends_with(BITNET_B158_2B_4T_I2S_MODEL_FILE) {
        return Err(anyhow!(
            "model_identity.active_model_path must end with `{}` for BitNet QK256 server smoke",
            BITNET_B158_2B_4T_I2S_MODEL_FILE
        ));
    }
    require_string_eq(receipt, "model_coverage_row", "bitnet_official_2b_i2s_qk256")?;
    require_string_eq(receipt, "model_coverage_tier", "product_cli_ready")?;
    require_bool_eq(receipt, "dense_regular_llm_cuda_inference_claimed", false)?;
    require_bool_eq(receipt, "bitnet_packed_i2s_qk256_proof", true)?;
    validate_bitnet_qk256_server_execution_plan(receipt)?;
    validate_bitnet_qk256_server_execution_coverage(receipt)?;
    validate_bitnet_qk256_server_kernel_stats(receipt)?;
    Ok(())
}

fn validate_bitnet_qk256_server_execution_plan(receipt: &Value) -> Result<()> {
    let plan = object_field(receipt, "execution_plan")?;
    require_string_eq(plan, "planner_version", CUDA_PLANNER_RECEIPT_VERSION)?;
    require_string_eq(plan, "model_family", "bitnet_b1_58")?;
    require_string_eq(plan, "quantization", "i2_s_qk256")?;
    require_string_eq(plan, "selected_route", "bitnet_qk256_cuda")?;
    require_string_eq(plan, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(plan, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(plan, "runtime_api", "cuda")?;
    require_string_eq(plan, "strict_fallback_policy", "reject")?;
    require_bool_eq(plan, "dense_regular_llm_cuda", false)?;
    require_bool_eq(plan, "bitnet_packed_qk256_cuda", true)?;
    require_positive_u64(plan, "cuda_bitnet_qk256_ops")?;
    let cuda_bitnet_ops = required_u64(plan, "cuda_bitnet_qk256_ops")?;
    require_u64_eq(plan, "cuda_dense_regular_llm_ops", 0)?;
    require_u64_eq(plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(plan, "unsupported_ops", 0)?;
    require_u64_eq(plan, "total_ops", cuda_bitnet_ops)?;
    require_u64_eq(plan, "cuda_ops", cuda_bitnet_ops)?;
    require_bool_eq(plan, "mixed_cuda_routes", false)?;
    require_bool_eq(plan, "fallback_used", false)?;
    require_bool_eq(plan, "strict_cuda_ready", true)?;
    require_bool_eq(plan, "speedup_claim", false)?;
    require_bool_eq(plan, "full_cuda_residency_claimed", false)?;
    Ok(())
}

fn validate_bitnet_qk256_server_execution_coverage(receipt: &Value) -> Result<()> {
    let coverage = object_field(receipt, "execution_coverage")?;
    require_string_eq(coverage, "execution_claim", "cuda_inference_contribution")?;
    require_positive_u64(coverage, "bitnet_linear_layers_total")?;
    require_positive_u64(coverage, "bitnet_linear_layers_on_cuda")?;
    let total = required_u64(coverage, "bitnet_linear_layers_total")?;
    let on_cuda = required_u64(coverage, "bitnet_linear_layers_on_cuda")?;
    if total != on_cuda {
        return Err(anyhow!(
            "execution_coverage bitnet_linear_layers_total must match bitnet_linear_layers_on_cuda for zero-fallback BitNet QK256 server smoke"
        ));
    }
    require_u64_eq(coverage, "bitnet_linear_layers_cpu_fallback", 0)?;
    require_bool_eq(coverage, "fallback_used", false)?;
    let unsupported_ops = array_field(coverage, "unsupported_ops")?;
    if !unsupported_ops.is_empty() {
        return Err(anyhow!("execution_coverage.unsupported_ops must be empty"));
    }
    Ok(())
}

fn validate_bitnet_qk256_server_kernel_stats(receipt: &Value) -> Result<()> {
    let stats = array_field(receipt, "kernel_stats")?;
    if stats.is_empty() {
        return Err(anyhow!("kernel_stats must contain QK256 CUDA server-smoke entries"));
    }
    let mut total_invocations = 0_u64;
    for (index, stat) in stats.iter().enumerate() {
        require_string_eq(stat, "kernel_id", "qk256_gemv_cuda")?;
        require_positive_u64(stat, "invocations")?;
        let invocations = required_u64(stat, "invocations")?;
        total_invocations += invocations;
        require_u64_eq(stat, "fallback_invocations", 0)?;
        require_u64_eq(stat, "cpu_fallback_invocations", 0)?;
        require_optional_u64_field(stat, "host_to_device_bytes")?;
        require_optional_u64_field(stat, "device_to_host_bytes")?;
        require_positive_u64(stat, "kernel_launches")?;
        require_optional_non_negative_number(stat, "kernel_time_ms")?;
        require_optional_u64_field(stat, "kernel_time_samples")?;
        if required_u64(stat, "kernel_launches")? != invocations {
            return Err(anyhow!(
                "kernel_stats[{index}].kernel_launches must match invocations for QK256 server smoke"
            ));
        }
    }
    let coverage = object_field(receipt, "execution_coverage")?;
    require_u64_eq(coverage, "bitnet_linear_layers_on_cuda", total_invocations)?;
    Ok(())
}

fn validate_dense_regular_llm_execution_plan(receipt: &Value) -> Result<()> {
    let plan = object_field(receipt, "execution_plan")?;
    require_string_eq(plan, "planner_version", CUDA_PLANNER_RECEIPT_VERSION)?;
    require_string_non_empty(plan, "model_family")?;
    reject_bitnet_packed_marker(
        required_string(plan, "model_family")?,
        "execution_plan.model_family",
    )?;
    require_string_non_empty(plan, "quantization")?;
    reject_bitnet_packed_marker(
        required_string(plan, "quantization")?,
        "execution_plan.quantization",
    )?;
    require_string_eq(plan, "selected_route", DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND)?;
    require_string_eq(plan, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(plan, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(plan, "runtime_api", "cuda")?;
    require_string_eq(plan, "strict_fallback_policy", "reject")?;
    require_bool_eq(plan, "dense_regular_llm_cuda", true)?;
    require_bool_eq(plan, "bitnet_packed_qk256_cuda", false)?;
    require_u64_eq(plan, "cuda_bitnet_qk256_ops", 0)?;
    require_positive_u64(plan, "cuda_dense_regular_llm_ops")?;
    require_u64_eq(plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(plan, "unsupported_ops", 0)?;
    require_positive_u64(plan, "total_ops")?;
    require_positive_u64(plan, "cuda_ops")?;
    require_bool_eq(plan, "mixed_cuda_routes", false)?;
    require_bool_eq(plan, "fallback_used", false)?;
    require_bool_eq(plan, "strict_cuda_ready", true)?;
    require_bool_eq(plan, "speedup_claim", false)?;
    require_bool_eq(plan, "full_cuda_residency_claimed", false)?;

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

/// Validate a descriptor-only dense GGUF tensor inspection receipt.
///
/// This validates model/tensor metadata coverage before any dense GGUF CUDA
/// execution claim exists. A valid receipt may say the GGUF reader can classify
/// Qwen/Llama-style tensor roles, but it must keep dense CUDA execution,
/// dense GGUF inference, speedup, full residency, and BitNet packed proof
/// claims false.
pub fn validate_dense_gguf_tensor_descriptor_inspection_receipt_json(
    receipt: &Value,
) -> Result<()> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND)?;
    require_string_eq(receipt, "claim", "dense_gguf_tensor_descriptors_inspected")?;
    require_string_eq(receipt, "hardware_lane", "nvidia-rtx-5070-ti-cuda")?;
    require_string_non_empty(receipt, "inspection_source")?;
    require_null(receipt, "error")?;

    let model = object_field(receipt, "model")?;
    require_string_non_empty(model, "model_family")?;
    reject_bitnet_packed_marker(required_string(model, "model_family")?, "model.model_family")?;
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;

    let inspection = object_field(receipt, "descriptor_inspection")?;
    require_u64_eq(inspection, "schema", 1)?;
    require_string_eq(inspection, "artifact_kind", DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND)?;
    require_string_eq(inspection, "model_family", required_string(model, "model_family")?)?;
    require_string_eq(inspection, "architecture", required_string(model, "architecture")?)?;
    require_positive_u64(inspection, "tensor_count")?;
    require_positive_u64(inspection, "metadata_count")?;
    require_bool_eq(inspection, "required_roles_present", true)?;
    require_bool_eq(inspection, "strict_descriptor_complete", true)?;
    require_bool_eq(inspection, "bitnet_packed_marker_found", false)?;
    require_bool_eq(inspection, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(inspection, "dense_regular_llm_cuda_claimed", false)?;
    require_bool_eq(inspection, "speedup_claim", false)?;
    require_bool_eq(inspection, "full_cuda_residency_claimed", false)?;

    let route_status = required_string(inspection, "dense_cuda_route_status")?;
    match route_status {
        "dense_float_descriptor_candidate" | "descriptor_only_quant_bridge_required" => {}
        other => {
            return Err(anyhow!(
                "descriptor_inspection.dense_cuda_route_status must be descriptor-only or float-candidate, got `{other}`"
            ));
        }
    }

    let quantization_families = array_field(inspection, "quantization_families")?;
    if quantization_families.is_empty() {
        return Err(anyhow!("descriptor_inspection.quantization_families must not be empty"));
    }
    for family in quantization_families {
        let family = family
            .as_str()
            .ok_or_else(|| anyhow!("quantization_families entries must be strings"))?;
        reject_bitnet_packed_marker(family, "descriptor_inspection.quantization_families")?;
    }

    let descriptors = array_field(inspection, "descriptors")?;
    if descriptors.is_empty() {
        return Err(anyhow!("descriptor_inspection.descriptors must not be empty"));
    }
    let mut roles = BTreeSet::new();
    for descriptor in descriptors {
        require_string_non_empty(descriptor, "name")?;
        reject_bitnet_packed_marker(required_string(descriptor, "name")?, "descriptors.name")?;
        let role = required_string(descriptor, "role")?;
        roles.insert(role.to_string());
        require_string_non_empty(descriptor, "tensor_type")?;
        reject_bitnet_packed_marker(
            required_string(descriptor, "tensor_type")?,
            "descriptors.tensor_type",
        )?;
        require_string_non_empty(descriptor, "descriptor_status")?;
        reject_bitnet_packed_marker(
            required_string(descriptor, "descriptor_status")?,
            "descriptors.descriptor_status",
        )?;
        require_positive_u64(descriptor, "size_bytes")?;
        object_field(descriptor, "shape")?
            .as_array()
            .ok_or_else(|| anyhow!("descriptors.shape must be an array"))?;
        object_field(descriptor, "quantized")?
            .as_bool()
            .ok_or_else(|| anyhow!("descriptors.quantized must be a bool"))?;
    }
    for role in REQUIRED_DENSE_DESCRIPTOR_ROLES {
        if !roles.contains(*role) {
            return Err(anyhow!("descriptor receipt missing required dense tensor role `{role}`"));
        }
    }

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

/// Validate a dense GGUF linear fixture extraction receipt.
///
/// This receipt sits between descriptor inspection and dense CUDA execution. It
/// proves one dense linear tensor can be selected, materialized as F32, and run
/// through a CPU reference matvec. It must keep dense CUDA parity, dense GGUF
/// inference, speedup, full residency, and BitNet packed proof claims false.
pub fn validate_dense_gguf_linear_fixture_extraction_receipt_json(receipt: &Value) -> Result<()> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", DENSE_GGUF_LINEAR_FIXTURE_ARTIFACT_KIND)?;
    require_string_eq(receipt, "claim", "dense_gguf_linear_fixture_extracted")?;
    require_string_eq(receipt, "hardware_lane", "nvidia-rtx-5070-ti-cuda")?;
    require_string_non_empty(receipt, "inspection_source")?;
    require_null(receipt, "error")?;

    let model = object_field(receipt, "model")?;
    require_string_non_empty(model, "model_family")?;
    reject_bitnet_packed_marker(required_string(model, "model_family")?, "model.model_family")?;
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;

    let fixture = object_field(receipt, "linear_fixture")?;
    require_u64_eq(fixture, "schema", 1)?;
    require_string_eq(fixture, "artifact_kind", DENSE_GGUF_LINEAR_FIXTURE_ARTIFACT_KIND)?;
    require_string_eq(fixture, "model_family", required_string(model, "model_family")?)?;
    require_string_eq(fixture, "architecture", required_string(model, "architecture")?)?;
    require_string_non_empty(fixture, "tensor_name")?;
    reject_bitnet_packed_marker(
        required_string(fixture, "tensor_name")?,
        "linear_fixture.tensor_name",
    )?;
    require_extractable_dense_linear_role(required_string(fixture, "role")?)?;
    require_string_non_empty(fixture, "tensor_type")?;
    reject_bitnet_packed_marker(
        required_string(fixture, "tensor_type")?,
        "linear_fixture.tensor_type",
    )?;
    let source_shape = array_field(fixture, "source_shape")?;
    if source_shape.len() != 2 {
        return Err(anyhow!("linear_fixture.source_shape must contain [matrix_cols, matrix_rows]"));
    }
    let source_cols = source_shape[0]
        .as_u64()
        .ok_or_else(|| anyhow!("linear_fixture.source_shape[0] must be an unsigned integer"))?;
    let source_rows = source_shape[1]
        .as_u64()
        .ok_or_else(|| anyhow!("linear_fixture.source_shape[1] must be an unsigned integer"))?;
    require_positive_u64(fixture, "source_size_bytes")?;
    require_positive_u64(fixture, "matrix_rows")?;
    require_positive_u64(fixture, "matrix_cols")?;
    require_positive_u64(fixture, "value_count")?;
    let matrix_rows = object_field(fixture, "matrix_rows")?
        .as_u64()
        .ok_or_else(|| anyhow!("linear_fixture.matrix_rows must be an unsigned integer"))?;
    let matrix_cols = object_field(fixture, "matrix_cols")?
        .as_u64()
        .ok_or_else(|| anyhow!("linear_fixture.matrix_cols must be an unsigned integer"))?;
    let expected_values = matrix_rows.checked_mul(matrix_cols).ok_or_else(|| {
        anyhow!("linear_fixture matrix_rows * matrix_cols overflows receipt validation")
    })?;
    if source_cols != matrix_cols || source_rows != matrix_rows {
        return Err(anyhow!(
            "linear_fixture.source_shape must match GGUF [matrix_cols, matrix_rows]"
        ));
    }
    require_u64_eq(fixture, "value_count", expected_values)?;
    require_string_eq(fixture, "logical_layout", "gguf_in_out_reinterpreted_as_out_in")?;
    require_bool_eq(fixture, "values_materialized_as_f32", true)?;
    require_sha256(fixture, "weight_values_sha256")?;
    require_u64_eq(fixture, "cpu_reference_input_len", matrix_cols)?;
    require_u64_eq(fixture, "cpu_reference_output_len", matrix_rows)?;
    require_sha256(fixture, "cpu_reference_input_sha256")?;
    require_sha256(fixture, "cpu_reference_output_sha256")?;
    require_bool_eq(fixture, "cpu_reference_computed", true)?;
    require_bool_eq(fixture, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(fixture, "dense_regular_llm_cuda_claimed", false)?;
    require_bool_eq(fixture, "cpu_cuda_parity_claimed", false)?;
    require_bool_eq(fixture, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(fixture, "speedup_claim", false)?;
    require_bool_eq(fixture, "full_cuda_residency_claimed", false)?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "cpu_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

/// Validate a dense GGUF norm fixture extraction receipt.
///
/// This receipt records that dense GGUF RMSNorm weight tensors can be selected,
/// materialized as F32, and run through a deterministic CPU RMSNorm reference.
/// It deliberately records the CUDA norm kernel as missing and must keep dense
/// CUDA parity, dense GGUF inference, speedup, full residency, and BitNet packed
/// proof claims false.
pub fn validate_dense_gguf_norm_fixture_extraction_receipt_json(receipt: &Value) -> Result<()> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", DENSE_GGUF_NORM_FIXTURE_ARTIFACT_KIND)?;
    require_string_eq(receipt, "claim", "dense_gguf_norm_fixture_extracted")?;
    require_string_eq(receipt, "hardware_lane", "nvidia-rtx-5070-ti-cuda")?;
    require_string_non_empty(receipt, "inspection_source")?;
    require_null(receipt, "error")?;

    let model = object_field(receipt, "model")?;
    require_string_non_empty(model, "model_family")?;
    reject_bitnet_packed_marker(required_string(model, "model_family")?, "model.model_family")?;
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    if model.get("sha256").is_some() {
        require_sha256(model, "sha256")?;
    }

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_bool_eq(descriptor, "required_roles_present", true)?;
    require_bool_eq(descriptor, "strict_descriptor_complete", true)?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;

    let audit = object_field(receipt, "norm_fixture_audit")?;
    require_u64_eq(audit, "schema", 1)?;
    require_string_eq(audit, "source_artifact_kind", DENSE_GGUF_NORM_FIXTURE_ARTIFACT_KIND)?;
    let roles_total = object_field(audit, "roles_total")?
        .as_u64()
        .ok_or_else(|| anyhow!("norm_fixture_audit.roles_total must be an unsigned integer"))?;
    if roles_total < 2 {
        return Err(anyhow!(
            "norm_fixture_audit.roles_total must cover attention_norm and ffn_norm"
        ));
    }
    require_u64_eq(audit, "roles_extracted", roles_total)?;
    require_u64_eq(audit, "roles_failed", 0)?;
    require_bool_eq(audit, "all_cpu_reference_computed", true)?;
    require_string_eq(audit, "cuda_kernel_status", "missing_cuda_kernel")?;
    require_bool_eq(audit, "strict_cuda_ready", false)?;
    require_bool_eq(audit, "cpu_fallback_allowed", false)?;
    require_string_eq(audit, "transfer_timing_status", "not_measured_no_kernel")?;
    require_bool_eq(audit, "dense_gguf_norm_fixture_extraction_claimed", true)?;
    require_bool_eq(audit, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(audit, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(audit, "speedup_claim", false)?;
    require_bool_eq(audit, "full_cuda_residency_claimed", false)?;

    let covered_roles = array_field(audit, "covered_roles")?;
    if covered_roles.len() != roles_total as usize {
        return Err(anyhow!("norm_fixture_audit.covered_roles length must match roles_total"));
    }
    let mut role_set = BTreeSet::new();
    for role in covered_roles {
        let role = role
            .as_str()
            .ok_or_else(|| anyhow!("norm_fixture_audit.covered_roles entries must be strings"))?;
        require_extractable_dense_norm_role(role)?;
        role_set.insert(role.to_string());
    }
    for role in ["attention_norm", "ffn_norm"] {
        if !role_set.contains(role) {
            return Err(anyhow!("norm_fixture_audit missing required norm role `{role}`"));
        }
    }

    let fixtures = array_field(receipt, "norm_fixtures")?;
    if fixtures.len() != roles_total as usize {
        return Err(anyhow!("norm_fixtures length must match roles_total"));
    }
    for fixture in fixtures {
        require_u64_eq(fixture, "schema", 1)?;
        require_string_eq(fixture, "artifact_kind", DENSE_GGUF_NORM_FIXTURE_ARTIFACT_KIND)?;
        require_string_eq(fixture, "model_family", required_string(model, "model_family")?)?;
        require_string_eq(fixture, "architecture", required_string(model, "architecture")?)?;
        require_string_non_empty(fixture, "tensor_name")?;
        reject_bitnet_packed_marker(
            required_string(fixture, "tensor_name")?,
            "norm_fixtures.tensor_name",
        )?;
        let role = required_string(fixture, "role")?;
        require_extractable_dense_norm_role(role)?;
        if !role_set.contains(role) {
            return Err(anyhow!("norm_fixtures role `{role}` is not listed in covered_roles"));
        }
        let tensor_type = required_string(fixture, "tensor_type")?;
        match tensor_type {
            "f32" | "f16" => {}
            other => {
                return Err(anyhow!("norm_fixtures.tensor_type must be f32 or f16, got `{other}`"));
            }
        }
        let source_shape = array_field(fixture, "source_shape")?;
        if source_shape.len() != 1 {
            return Err(anyhow!("norm_fixtures.source_shape must contain [hidden_dim]"));
        }
        let source_hidden = source_shape[0]
            .as_u64()
            .ok_or_else(|| anyhow!("norm_fixtures.source_shape[0] must be an unsigned integer"))?;
        require_positive_u64(fixture, "source_size_bytes")?;
        require_positive_u64(fixture, "hidden_dim")?;
        require_positive_u64(fixture, "value_count")?;
        require_u64_eq(fixture, "hidden_dim", source_hidden)?;
        require_u64_eq(fixture, "value_count", source_hidden)?;
        require_bool_eq(fixture, "values_materialized_as_f32", true)?;
        require_sha256(fixture, "weight_values_sha256")?;
        require_positive_number(fixture, "rmsnorm_eps")?;
        require_string_non_empty(fixture, "epsilon_source")?;
        require_u64_eq(fixture, "cpu_reference_input_len", source_hidden)?;
        require_u64_eq(fixture, "cpu_reference_output_len", source_hidden)?;
        require_sha256(fixture, "cpu_reference_input_sha256")?;
        require_sha256(fixture, "cpu_reference_output_sha256")?;
        require_bool_eq(fixture, "cpu_reference_computed", true)?;
        require_string_eq(fixture, "cuda_kernel_status", "missing_cuda_kernel")?;
        require_bool_eq(fixture, "dense_gguf_inference_claimed", false)?;
        require_bool_eq(fixture, "dense_regular_llm_cuda_claimed", false)?;
        require_bool_eq(fixture, "cpu_cuda_parity_claimed", false)?;
        require_bool_eq(fixture, "bitnet_packed_i2s_qk256_proof", false)?;
        require_bool_eq(fixture, "speedup_claim", false)?;
        require_bool_eq(fixture, "full_cuda_residency_claimed", false)?;
    }

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_norm_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "cpu_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

/// Validate dense GGUF RMSNorm CUDA parity evidence.
///
/// This receipt bridges descriptor-extracted dense GGUF norm fixtures into the
/// dense CUDA RMSNorm path. It must reject dense GGUF inference, Qwen token or
/// decode claims, speedup, full residency, and BitNet packed I2_S/QK256 proof.
pub fn validate_dense_gguf_norm_cuda_parity_receipt_json(receipt: &Value) -> Result<()> {
    validate_cuda_receipt_common(
        receipt,
        DENSE_GGUF_NORM_CUDA_PARITY_ARTIFACT_KIND,
        "dense_gguf_norm_cuda_parity_tested",
    )?;
    require_bool_eq(receipt, "speedup_claim", false)?;

    let cuda = object_field(receipt, "cuda")?;
    require_bool_eq(cuda, "available", true)?;

    let model = object_field(receipt, "model")?;
    require_string_non_empty(model, "model_family")?;
    reject_bitnet_packed_marker(required_string(model, "model_family")?, "model.model_family")?;
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_sha256(model, "sha256")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "dense_f32_rmsnorm")?;
    require_string_eq(execution_path, "quantization_family", "f32_norm_weights")?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    validate_dense_regular_llm_execution_plan(receipt)?;

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_bool_eq(descriptor, "required_roles_present", true)?;
    require_bool_eq(descriptor, "strict_descriptor_complete", true)?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;

    let parity = object_field(receipt, "parity")?;
    require_bool_eq(parity, "passed", true)?;
    let roles_total = object_field(parity, "roles_total")?
        .as_u64()
        .ok_or_else(|| anyhow!("parity.roles_total must be an unsigned integer"))?;
    if roles_total < 2 {
        return Err(anyhow!("parity.roles_total must cover attention_norm and ffn_norm"));
    }
    require_null(parity, "first_divergence")?;
    let covered_roles = array_field(parity, "covered_roles")?;
    if covered_roles.len() != roles_total as usize {
        return Err(anyhow!("parity.covered_roles length must match roles_total"));
    }
    let mut role_set = BTreeSet::new();
    for role in covered_roles {
        let role =
            role.as_str().ok_or_else(|| anyhow!("parity.covered_roles entries must be strings"))?;
        require_extractable_dense_norm_role(role)?;
        reject_bitnet_packed_marker(role, "parity.covered_roles")?;
        if !role_set.insert(role.to_string()) {
            return Err(anyhow!("parity.covered_roles contains duplicate `{role}`"));
        }
    }
    for required in ["attention_norm", "ffn_norm"] {
        if !role_set.contains(required) {
            return Err(anyhow!("parity.covered_roles missing required `{required}`"));
        }
    }

    let plan = object_field(receipt, "execution_plan")?;
    require_u64_eq(plan, "total_ops", roles_total)?;
    require_u64_eq(plan, "cuda_ops", roles_total)?;
    require_u64_eq(plan, "cuda_dense_regular_llm_ops", roles_total)?;
    require_u64_eq(plan, "cuda_bitnet_qk256_ops", 0)?;
    require_u64_eq(plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(plan, "unsupported_ops", 0)?;
    require_bool_eq(plan, "strict_cuda_ready", true)?;

    let fixtures = array_field(receipt, "norm_fixtures")?;
    if fixtures.len() != roles_total as usize {
        return Err(anyhow!("norm_fixtures length must match roles_total"));
    }
    let stats = array_field(receipt, "kernel_stats")?;
    if stats.len() != roles_total as usize {
        return Err(anyhow!("kernel_stats length must match roles_total"));
    }
    let parity_results = array_field(receipt, "parity_results")?;
    if parity_results.len() != roles_total as usize {
        return Err(anyhow!("parity_results length must match roles_total"));
    }

    let mut stats_h2d = 0_u64;
    let mut stats_d2h = 0_u64;
    let mut stats_launches = 0_u64;
    for (idx, fixture) in fixtures.iter().enumerate() {
        require_u64_eq(fixture, "schema", 1)?;
        require_string_eq(fixture, "source_artifact_kind", DENSE_GGUF_NORM_FIXTURE_ARTIFACT_KIND)?;
        require_string_non_empty(fixture, "fixture_id")?;
        reject_bitnet_packed_marker(
            required_string(fixture, "fixture_id")?,
            "norm_fixtures.fixture_id",
        )?;
        require_string_eq(fixture, "model_family", required_string(model, "model_family")?)?;
        require_string_eq(fixture, "architecture", required_string(model, "architecture")?)?;
        require_string_non_empty(fixture, "tensor_name")?;
        reject_bitnet_packed_marker(
            required_string(fixture, "tensor_name")?,
            "norm_fixtures.tensor_name",
        )?;
        let role = required_string(fixture, "role")?;
        require_extractable_dense_norm_role(role)?;
        if !role_set.contains(role) {
            return Err(anyhow!("norm_fixtures role `{role}` is not listed in covered_roles"));
        }
        require_string_non_empty(fixture, "tensor_type")?;
        reject_bitnet_packed_marker(
            required_string(fixture, "tensor_type")?,
            "norm_fixtures.tensor_type",
        )?;
        let source_shape = array_field(fixture, "source_shape")?;
        if source_shape.len() != 1 {
            return Err(anyhow!("norm_fixtures.source_shape must be one-dimensional"));
        }
        require_positive_u64(fixture, "hidden_dim")?;
        require_positive_u64(fixture, "value_count")?;
        require_bool_eq(fixture, "values_materialized_as_f32", true)?;
        require_sha256(fixture, "weight_values_sha256")?;
        require_positive_number(fixture, "rmsnorm_eps")?;
        require_string_non_empty(fixture, "epsilon_source")?;
        require_string_eq(fixture, "cuda_input_dtype", "f32")?;
        require_string_eq(fixture, "cuda_gamma_dtype", "f32")?;
        require_string_eq(fixture, "cuda_output_dtype", "f32")?;
        require_bool_eq(fixture, "dense_gguf_inference_claimed", false)?;
        require_bool_eq(fixture, "dense_regular_llm_cuda_claimed", true)?;
        require_bool_eq(fixture, "cpu_cuda_parity_claimed", true)?;
        require_bool_eq(fixture, "bitnet_packed_i2s_qk256_proof", false)?;
        require_bool_eq(fixture, "speedup_claim", false)?;
        require_bool_eq(fixture, "full_cuda_residency_claimed", false)?;

        let stat = &stats[idx];
        require_string_eq(stat, "role", role)?;
        require_string_eq(stat, "tensor_name", required_string(fixture, "tensor_name")?)?;
        require_string_eq(stat, "fixture_id", required_string(fixture, "fixture_id")?)?;
        require_string_eq(stat, "kernel_id", "dense_rmsnorm_f32_cuda")?;
        require_positive_u64(stat, "invocations")?;
        require_u64_eq(stat, "fallback_invocations", 0)?;
        require_positive_u64(stat, "host_to_device_bytes")?;
        require_positive_u64(stat, "device_to_host_bytes")?;
        require_positive_u64(stat, "kernel_launches")?;
        require_optional_non_negative_number(stat, "kernel_time_ms")?;
        stats_h2d += object_field(stat, "host_to_device_bytes")?.as_u64().ok_or_else(|| {
            anyhow!("kernel_stats.host_to_device_bytes must be an unsigned integer")
        })?;
        stats_d2h += object_field(stat, "device_to_host_bytes")?.as_u64().ok_or_else(|| {
            anyhow!("kernel_stats.device_to_host_bytes must be an unsigned integer")
        })?;
        stats_launches += object_field(stat, "kernel_launches")?
            .as_u64()
            .ok_or_else(|| anyhow!("kernel_stats.kernel_launches must be an unsigned integer"))?;

        let result = &parity_results[idx];
        require_string_eq(result, "reference_backend", "amd-9950x3d-cpu-avx512")?;
        require_string_eq(result, "target_backend", "nvidia-rtx-5070-ti-cuda")?;
        require_string_eq(result, "kernel_id", "dense_rmsnorm_f32_cuda")?;
        require_string_eq(result, "fixture_id", required_string(fixture, "fixture_id")?)?;
        require_string_eq(result, "role", role)?;
        require_u64_eq(
            result,
            "hidden_dim",
            object_field(fixture, "hidden_dim")?
                .as_u64()
                .ok_or_else(|| anyhow!("norm_fixtures.hidden_dim must be an unsigned integer"))?,
        )?;
        require_bool_eq(result, "passed", true)?;
        require_non_negative_number(result, "max_abs_error")?;
        require_non_negative_number(result, "mean_abs_error")?;
        require_non_negative_number(result, "tolerance")?;
        require_string_non_empty(result, "tolerance_source")?;
    }

    let timing = object_field(receipt, "timing")?;
    require_optional_non_negative_number(timing, "kernel_time_ms")?;
    require_u64_eq(timing, "host_to_device_bytes", stats_h2d)?;
    require_u64_eq(timing, "device_to_host_bytes", stats_d2h)?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_norm_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_norm_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "cpu_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    let residency = object_field(receipt, "tensor_residency")?;
    require_string_eq(residency, "scope", "single_dense_gguf_rmsnorm_fixture")?;
    require_string_eq(residency, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_u64_eq(residency, "roles_total", roles_total)?;
    require_bool_eq(residency, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(residency, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(residency, "persistent_session_residency_claimed", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    require_bool_eq(residency, "input_tensors_uploaded_once", true)?;
    require_bool_eq(residency, "output_tensor_cuda_resident_during_kernel", true)?;
    require_bool_eq(residency, "host_device_transfer_accounting_matches_kernel_stats", true)?;
    let allocation = object_field(residency, "allocation")?;
    require_u64_eq(allocation, "device_buffer_count_per_role", 3)?;
    require_u64_eq(allocation, "persistent_handle_count", 0)?;
    require_bool_eq(allocation, "persistent_handles_claimed", false)?;
    let transfer = object_field(residency, "transfer_accounting")?;
    require_string_eq(transfer, "status", "measured")?;
    require_u64_eq(transfer, "host_to_device_bytes", stats_h2d)?;
    require_u64_eq(transfer, "device_to_host_bytes", stats_d2h)?;
    require_u64_eq(residency, "kernel_launches", stats_launches)?;

    Ok(())
}

/// Validate dense GGUF RoPE CUDA parity evidence.
///
/// This receipt bridges metadata-derived dense GGUF Q/K RoPE fixtures into the
/// dense CUDA RoPE path. It must reject dense GGUF inference, Qwen token or
/// decode claims, speedup, full residency, and BitNet packed I2_S/QK256 proof.
pub fn validate_dense_gguf_rope_cuda_parity_receipt_json(receipt: &Value) -> Result<()> {
    let stats = validate_cuda_receipt_common(
        receipt,
        DENSE_GGUF_ROPE_CUDA_PARITY_ARTIFACT_KIND,
        "dense_gguf_rope_cuda_parity_tested",
    )?;
    require_string_eq(stats, "kernel_id", "dense_rope_f32_cuda")?;
    require_u64_eq(stats, "invocations", 2)?;
    require_u64_eq(stats, "fallback_invocations", 0)?;
    require_u64_eq(stats, "kernel_launches", 2)?;
    require_bool_eq(receipt, "speedup_claim", false)?;

    let model = object_field(receipt, "model")?;
    require_string_non_empty(model, "model_family")?;
    reject_bitnet_packed_marker(required_string(model, "model_family")?, "model.model_family")?;
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_sha256(model, "sha256")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "dense_f32_rope")?;
    require_string_eq(
        execution_path,
        "quantization_family",
        "metadata_derived_rope_qk_f32_fixture",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    validate_dense_regular_llm_execution_plan(receipt)?;
    let plan = object_field(receipt, "execution_plan")?;
    require_u64_eq(plan, "total_ops", 1)?;
    require_u64_eq(plan, "cuda_ops", 1)?;
    require_u64_eq(plan, "cuda_dense_regular_llm_ops", 1)?;
    require_u64_eq(plan, "cuda_bitnet_qk256_ops", 0)?;
    require_u64_eq(plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(plan, "unsupported_ops", 0)?;
    require_bool_eq(plan, "strict_cuda_ready", true)?;

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_bool_eq(descriptor, "required_roles_present", true)?;
    require_bool_eq(descriptor, "strict_descriptor_complete", true)?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;

    let fixture = object_field(receipt, "rope_fixture")?;
    require_u64_eq(fixture, "schema", 1)?;
    require_string_eq(
        fixture,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_string_non_empty(fixture, "fixture_id")?;
    reject_bitnet_packed_marker(
        required_string(fixture, "fixture_id")?,
        "rope_fixture.fixture_id",
    )?;
    require_string_eq(fixture, "model_family", required_string(model, "model_family")?)?;
    require_string_eq(fixture, "architecture", required_string(model, "architecture")?)?;
    require_positive_u64(fixture, "head_dim")?;
    let head_dim = object_field(fixture, "head_dim")?
        .as_u64()
        .ok_or_else(|| anyhow!("rope_fixture.head_dim must be an unsigned integer"))?;
    if head_dim % 2 != 0 {
        return Err(anyhow!("rope_fixture.head_dim must be even"));
    }
    require_positive_u64(fixture, "q_heads")?;
    require_positive_u64(fixture, "kv_heads")?;
    require_positive_u64(fixture, "seq_len")?;
    require_positive_number(fixture, "rope_base")?;
    require_positive_number(fixture, "scaling_factor")?;
    require_bool_eq(fixture, "interleaved", false)?;
    require_string_non_empty(fixture, "head_dim_source")?;
    require_string_non_empty(fixture, "q_heads_source")?;
    require_string_non_empty(fixture, "kv_heads_source")?;
    require_string_non_empty(fixture, "rope_base_source")?;
    require_sha256(fixture, "q_input_sha256")?;
    require_sha256(fixture, "k_input_sha256")?;
    require_sha256(fixture, "cpu_reference_q_output_sha256")?;
    require_sha256(fixture, "cpu_reference_k_output_sha256")?;
    require_string_eq(fixture, "cuda_input_dtype", "f32")?;
    require_string_eq(fixture, "cuda_output_dtype", "f32")?;
    require_bool_eq(fixture, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(fixture, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(fixture, "cpu_cuda_parity_claimed", true)?;
    require_bool_eq(fixture, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(fixture, "speedup_claim", false)?;
    require_bool_eq(fixture, "full_cuda_residency_claimed", false)?;

    let parity = object_field(receipt, "parity")?;
    require_string_eq(parity, "reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(parity, "target_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(parity, "kernel_id", "dense_rope_f32_cuda")?;
    require_string_eq(parity, "fixture_id", required_string(fixture, "fixture_id")?)?;
    require_bool_eq(parity, "passed", true)?;
    require_non_negative_number(parity, "max_abs_error")?;
    require_non_negative_number(parity, "mean_abs_error")?;
    require_non_negative_number(parity, "tolerance")?;
    require_string_non_empty(parity, "tolerance_source")?;
    require_null(parity, "first_divergence")?;

    let timing = object_field(receipt, "timing")?;
    require_optional_non_negative_number(timing, "kernel_time_ms")?;
    require_u64_eq(
        timing,
        "host_to_device_bytes",
        object_field(stats, "host_to_device_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("kernel_stats.host_to_device_bytes must be an integer"))?,
    )?;
    require_u64_eq(
        timing,
        "device_to_host_bytes",
        object_field(stats, "device_to_host_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("kernel_stats.device_to_host_bytes must be an integer"))?,
    )?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_rope_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_norm_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "cpu_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    let residency = object_field(receipt, "tensor_residency")?;
    require_string_eq(residency, "scope", "single_dense_gguf_rope_fixture")?;
    require_string_eq(residency, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(residency, "fixture_id", required_string(fixture, "fixture_id")?)?;
    require_bool_eq(residency, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(residency, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(residency, "persistent_session_residency_claimed", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    require_bool_eq(residency, "input_tensors_uploaded_once", true)?;
    require_bool_eq(residency, "output_tensor_cuda_resident_during_kernel", true)?;
    require_bool_eq(residency, "host_device_transfer_accounting_matches_kernel_stats", true)?;

    let inputs = array_field(residency, "inputs")?;
    if inputs.len() != 2 {
        return Err(anyhow!("RoPE tensor_residency.inputs must contain Q and K inputs"));
    }
    let mut h2d = 0_u64;
    for input in inputs {
        require_string_non_empty(input, "name")?;
        require_string_eq(input, "dtype", "f32")?;
        require_string_eq(input, "device_residency", "cuda_device_buffer")?;
        require_u64_eq(input, "upload_count", 1)?;
        require_string_eq(input, "reuse_scope", "single_fixture_launch")?;
        require_positive_u64(input, "host_bytes")?;
        h2d += object_field(input, "host_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("input.host_bytes must be an unsigned integer"))?;
    }

    let outputs = array_field(residency, "outputs")?;
    if outputs.len() != 2 {
        return Err(anyhow!("RoPE tensor_residency.outputs must contain Q and K outputs"));
    }
    let mut d2h = 0_u64;
    for output in outputs {
        require_string_non_empty(output, "name")?;
        require_string_eq(output, "dtype", "f32")?;
        require_string_eq(output, "device_residency", "cuda_device_buffer")?;
        require_string_eq(output, "download_scope", "parity_check_only")?;
        require_positive_u64(output, "device_to_host_bytes")?;
        d2h += object_field(output, "device_to_host_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("output.device_to_host_bytes must be an unsigned integer"))?;
    }

    let allocation = object_field(residency, "allocation")?;
    require_u64_eq(allocation, "device_buffer_count", 4)?;
    require_u64_eq(allocation, "persistent_handle_count", 0)?;
    require_bool_eq(allocation, "persistent_handles_claimed", false)?;

    let transfer = object_field(residency, "transfer_accounting")?;
    require_string_eq(transfer, "status", "measured")?;
    require_u64_eq(transfer, "host_to_device_bytes", h2d)?;
    require_u64_eq(transfer, "device_to_host_bytes", d2h)?;
    require_u64_eq(transfer, "kernel_invocations", 2)?;
    require_u64_eq(transfer, "kernel_launches", 2)?;
    require_u64_eq(
        stats,
        "host_to_device_bytes",
        object_field(transfer, "host_to_device_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("transfer.host_to_device_bytes must be an integer"))?,
    )?;
    require_u64_eq(
        stats,
        "device_to_host_bytes",
        object_field(transfer, "device_to_host_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("transfer.device_to_host_bytes must be an integer"))?,
    )?;

    Ok(())
}

/// Validate a dense GGUF attention-score fixture extraction receipt.
///
/// This receipt is a CPU-reference bridge after RoPE parity. It records
/// metadata-derived Q/K RoPE outputs, causal masking, and scaled QK scores for
/// the next dense one-layer gap, but it must not claim CUDA parity, dense GGUF
/// inference, speedup, full residency, or BitNet packed I2_S/QK256 proof.
pub fn validate_dense_gguf_attention_score_fixture_receipt_json(receipt: &Value) -> Result<()> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", DENSE_GGUF_ATTENTION_SCORE_FIXTURE_ARTIFACT_KIND)?;
    require_string_eq(receipt, "claim", "dense_gguf_attention_score_fixture_extracted")?;
    require_string_eq(receipt, "hardware_lane", "nvidia-rtx-5070-ti-cuda")?;
    require_string_non_empty(receipt, "inspection_source")?;
    require_null(receipt, "error")?;

    let model = object_field(receipt, "model")?;
    require_string_non_empty(model, "model_family")?;
    reject_bitnet_packed_marker(required_string(model, "model_family")?, "model.model_family")?;
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_sha256(model, "sha256")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(
        execution_path,
        "kernel_family",
        "cpu_reference_attention_scores_after_rope",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    let plan = object_field(receipt, "execution_plan")?;
    require_string_eq(plan, "model_family", required_string(model, "model_family")?)?;
    require_string_eq(plan, "selected_route", "unsupported")?;
    require_string_eq(plan, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(plan, "selected_backend", "unsupported_strict_cuda")?;
    require_string_eq(plan, "runtime_api", "none")?;
    require_string_eq(plan, "strict_fallback_policy", "reject")?;
    require_bool_eq(plan, "dense_regular_llm_cuda", false)?;
    require_bool_eq(plan, "bitnet_packed_qk256_cuda", false)?;
    require_u64_eq(plan, "cuda_bitnet_qk256_ops", 0)?;
    require_u64_eq(plan, "cuda_dense_regular_llm_ops", 0)?;
    require_u64_eq(plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(plan, "unsupported_ops", 1)?;
    require_u64_eq(plan, "total_ops", 1)?;
    require_u64_eq(plan, "cuda_ops", 0)?;
    require_bool_eq(plan, "fallback_used", false)?;
    require_bool_eq(plan, "strict_cuda_ready", false)?;
    require_bool_eq(plan, "speedup_claim", false)?;
    require_bool_eq(plan, "full_cuda_residency_claimed", false)?;

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_bool_eq(descriptor, "required_roles_present", true)?;
    require_bool_eq(descriptor, "strict_descriptor_complete", true)?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;

    let fixture = object_field(receipt, "attention_score_fixture")?;
    require_u64_eq(fixture, "schema", 1)?;
    require_string_eq(
        fixture,
        "source_artifact_kind",
        DENSE_GGUF_ATTENTION_SCORE_FIXTURE_ARTIFACT_KIND,
    )?;
    require_string_eq(
        fixture,
        "source_rope_artifact_kind",
        DENSE_GGUF_ROPE_CUDA_PARITY_ARTIFACT_KIND,
    )?;
    require_string_non_empty(fixture, "source_rope_fixture_id")?;
    require_string_non_empty(fixture, "fixture_id")?;
    require_string_eq(fixture, "model_family", required_string(model, "model_family")?)?;
    require_string_eq(fixture, "architecture", required_string(model, "architecture")?)?;
    require_positive_u64(fixture, "head_dim")?;
    require_positive_u64(fixture, "q_heads")?;
    require_positive_u64(fixture, "kv_heads")?;
    require_positive_u64(fixture, "heads_per_kv_group")?;
    require_positive_u64(fixture, "seq_len")?;
    require_positive_number(fixture, "rope_base")?;
    require_positive_number(fixture, "scaling_factor")?;
    require_positive_number(fixture, "attention_scale")?;
    require_bool_eq(fixture, "causal_mask_applied", true)?;
    require_string_non_empty(fixture, "head_dim_source")?;
    require_string_non_empty(fixture, "q_heads_source")?;
    require_string_non_empty(fixture, "kv_heads_source")?;
    require_string_non_empty(fixture, "rope_base_source")?;
    require_sha256(fixture, "q_rope_output_sha256")?;
    require_sha256(fixture, "k_rope_output_sha256")?;
    require_sha256(fixture, "cpu_reference_scores_sha256")?;
    let shape = array_field(fixture, "score_shape")?;
    if shape.len() != 3 {
        return Err(anyhow!(
            "attention_score_fixture.score_shape must contain [q_heads, seq_len, seq_len]"
        ));
    }
    let q_heads = object_field(fixture, "q_heads")?
        .as_u64()
        .ok_or_else(|| anyhow!("attention_score_fixture.q_heads must be an unsigned integer"))?;
    let seq_len = object_field(fixture, "seq_len")?
        .as_u64()
        .ok_or_else(|| anyhow!("attention_score_fixture.seq_len must be an unsigned integer"))?;
    let expected_score_count = q_heads * seq_len * seq_len;
    require_u64_eq(fixture, "score_count", expected_score_count)?;
    let finite_scores = object_field(fixture, "finite_scores")?.as_u64().ok_or_else(|| {
        anyhow!("attention_score_fixture.finite_scores must be an unsigned integer")
    })?;
    let masked_scores =
        object_field(fixture, "causal_masked_scores")?.as_u64().ok_or_else(|| {
            anyhow!("attention_score_fixture.causal_masked_scores must be an unsigned integer")
        })?;
    if finite_scores == 0 || finite_scores + masked_scores != expected_score_count {
        return Err(anyhow!(
            "attention_score_fixture finite and causal-masked counts must sum to score_count"
        ));
    }
    require_bool_eq(fixture, "cpu_reference_computed", true)?;
    require_string_eq(fixture, "cuda_kernel_status", "missing_cuda_kernel")?;
    require_bool_eq(fixture, "strict_cuda_ready", false)?;
    require_bool_eq(fixture, "cpu_fallback_allowed", false)?;
    require_string_eq(fixture, "transfer_timing_status", "not_measured_no_kernel")?;
    require_bool_eq(fixture, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(fixture, "dense_regular_llm_cuda_claimed", false)?;
    require_bool_eq(fixture, "cpu_cuda_parity_claimed", false)?;
    require_bool_eq(fixture, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(fixture, "speedup_claim", false)?;
    require_bool_eq(fixture, "full_cuda_residency_claimed", false)?;

    let audit = object_field(receipt, "attention_score_gap_audit")?;
    require_u64_eq(audit, "schema", 1)?;
    require_string_eq(
        audit,
        "source_artifact_kind",
        DENSE_GGUF_ATTENTION_SCORE_FIXTURE_ARTIFACT_KIND,
    )?;
    require_string_eq(audit, "gap_role", "attention_scores")?;
    require_bool_eq(audit, "source_rope_cuda_parity_required", true)?;
    require_bool_eq(audit, "source_rope_cuda_parity_available", true)?;
    require_bool_eq(audit, "cpu_reference_available", true)?;
    require_string_eq(audit, "cuda_kernel_status", "missing_cuda_kernel")?;
    require_bool_eq(audit, "strict_cuda_ready", false)?;
    require_bool_eq(audit, "cpu_fallback_allowed", false)?;
    require_bool_eq(audit, "blocks_strict_cuda_one_layer", true)?;
    require_string_eq(audit, "next_required_proof", "cuda_attention_score_kernel_parity")?;
    let deps = array_field(audit, "input_dependencies")?;
    let deps = deps
        .iter()
        .map(|dep| {
            dep.as_str().ok_or_else(|| {
                anyhow!("attention_score_gap_audit.input_dependencies entries must be strings")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if deps != ["rope_q", "rope_k", "causal_mask"] {
        return Err(anyhow!(
            "attention_score_gap_audit.input_dependencies must identify RoPE Q/K and causal mask"
        ));
    }
    let candidate_order = array_field(audit, "candidate_order")?;
    let candidate_order = candidate_order
        .iter()
        .map(|role| {
            role.as_str().ok_or_else(|| {
                anyhow!("attention_score_gap_audit.candidate_order entries must be strings")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if candidate_order != DENSE_ONE_LAYER_GAP_CANDIDATE_ORDER {
        return Err(anyhow!(
            "attention_score_gap_audit.candidate_order must preserve the governed gap order"
        ));
    }
    require_bool_eq(audit, "dense_gguf_attention_score_fixture_extraction_claimed", true)?;
    require_bool_eq(audit, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(audit, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(audit, "speedup_claim", false)?;
    require_bool_eq(audit, "full_cuda_residency_claimed", false)?;

    let timing = object_field(receipt, "timing")?;
    require_null(timing, "kernel_time_ms")?;
    require_u64_eq(timing, "host_to_device_bytes", 0)?;
    require_u64_eq(timing, "device_to_host_bytes", 0)?;
    require_string_eq(timing, "transfer_timing_status", "not_measured_no_kernel")?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_score_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_rope_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_norm_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "cpu_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

/// Validate a dense GGUF attention-softmax fixture extraction receipt.
///
/// This receipt is a CPU-reference bridge after attention-score parity. It
/// records stable row-wise softmax probabilities for the next dense one-layer
/// gap, but it must not claim CUDA parity, dense GGUF inference, speedup, full
/// residency, or BitNet packed I2_S/QK256 proof.
pub fn validate_dense_gguf_attention_softmax_fixture_receipt_json(receipt: &Value) -> Result<()> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(
        receipt,
        "artifact_kind",
        DENSE_GGUF_ATTENTION_SOFTMAX_FIXTURE_ARTIFACT_KIND,
    )?;
    require_string_eq(receipt, "claim", "dense_gguf_attention_softmax_fixture_extracted")?;
    require_string_eq(receipt, "hardware_lane", "nvidia-rtx-5070-ti-cuda")?;
    require_string_non_empty(receipt, "inspection_source")?;
    require_null(receipt, "error")?;

    let model = object_field(receipt, "model")?;
    require_string_non_empty(model, "model_family")?;
    reject_bitnet_packed_marker(required_string(model, "model_family")?, "model.model_family")?;
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_sha256(model, "sha256")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(
        execution_path,
        "kernel_family",
        "cpu_reference_attention_softmax_after_scores",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    let plan = object_field(receipt, "execution_plan")?;
    require_string_eq(plan, "model_family", required_string(model, "model_family")?)?;
    require_string_eq(plan, "selected_route", "unsupported")?;
    require_string_eq(plan, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(plan, "selected_backend", "unsupported_strict_cuda")?;
    require_string_eq(plan, "runtime_api", "none")?;
    require_string_eq(plan, "strict_fallback_policy", "reject")?;
    require_bool_eq(plan, "dense_regular_llm_cuda", false)?;
    require_bool_eq(plan, "bitnet_packed_qk256_cuda", false)?;
    require_u64_eq(plan, "cuda_bitnet_qk256_ops", 0)?;
    require_u64_eq(plan, "cuda_dense_regular_llm_ops", 0)?;
    require_u64_eq(plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(plan, "unsupported_ops", 1)?;
    require_u64_eq(plan, "total_ops", 1)?;
    require_u64_eq(plan, "cuda_ops", 0)?;
    require_bool_eq(plan, "fallback_used", false)?;
    require_bool_eq(plan, "strict_cuda_ready", false)?;
    require_bool_eq(plan, "speedup_claim", false)?;
    require_bool_eq(plan, "full_cuda_residency_claimed", false)?;

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_bool_eq(descriptor, "required_roles_present", true)?;
    require_bool_eq(descriptor, "strict_descriptor_complete", true)?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;

    let fixture = object_field(receipt, "attention_softmax_fixture")?;
    require_u64_eq(fixture, "schema", 1)?;
    require_string_eq(
        fixture,
        "source_artifact_kind",
        DENSE_GGUF_ATTENTION_SOFTMAX_FIXTURE_ARTIFACT_KIND,
    )?;
    require_string_eq(
        fixture,
        "source_attention_score_artifact_kind",
        DENSE_GGUF_ATTENTION_SCORE_CUDA_PARITY_ARTIFACT_KIND,
    )?;
    require_string_non_empty(fixture, "source_attention_score_fixture_id")?;
    require_string_non_empty(fixture, "fixture_id")?;
    require_string_eq(fixture, "model_family", required_string(model, "model_family")?)?;
    require_string_eq(fixture, "architecture", required_string(model, "architecture")?)?;
    require_positive_u64(fixture, "q_heads")?;
    require_positive_u64(fixture, "kv_heads")?;
    require_positive_u64(fixture, "seq_len")?;
    require_positive_u64(fixture, "row_count")?;
    require_sha256(fixture, "attention_scores_sha256")?;
    require_sha256(fixture, "cpu_reference_probabilities_sha256")?;
    let q_heads = object_field(fixture, "q_heads")?
        .as_u64()
        .ok_or_else(|| anyhow!("attention_softmax_fixture.q_heads must be an unsigned integer"))?;
    let seq_len = object_field(fixture, "seq_len")?
        .as_u64()
        .ok_or_else(|| anyhow!("attention_softmax_fixture.seq_len must be an unsigned integer"))?;
    require_u64_eq(fixture, "row_count", q_heads * seq_len)?;
    let expected_probability_count = q_heads * seq_len * seq_len;
    require_u64_eq(fixture, "probability_count", expected_probability_count)?;
    let zero_probs =
        object_field(fixture, "causal_zero_probabilities")?.as_u64().ok_or_else(|| {
            anyhow!("attention_softmax_fixture.causal_zero_probabilities must be unsigned")
        })?;
    if zero_probs >= expected_probability_count {
        return Err(anyhow!(
            "attention_softmax_fixture causal_zero_probabilities must leave finite probabilities"
        ));
    }
    require_non_negative_number(fixture, "max_row_sum_abs_error")?;
    require_bool_eq(fixture, "cpu_reference_computed", true)?;
    require_string_eq(fixture, "cuda_kernel_status", "missing_cuda_kernel")?;
    require_bool_eq(fixture, "strict_cuda_ready", false)?;
    require_bool_eq(fixture, "cpu_fallback_allowed", false)?;
    require_string_eq(fixture, "transfer_timing_status", "not_measured_no_kernel")?;
    require_bool_eq(fixture, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(fixture, "dense_regular_llm_cuda_claimed", false)?;
    require_bool_eq(fixture, "cpu_cuda_parity_claimed", false)?;
    require_bool_eq(fixture, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(fixture, "speedup_claim", false)?;
    require_bool_eq(fixture, "full_cuda_residency_claimed", false)?;

    let audit = object_field(receipt, "attention_softmax_gap_audit")?;
    require_u64_eq(audit, "schema", 1)?;
    require_string_eq(
        audit,
        "source_artifact_kind",
        DENSE_GGUF_ATTENTION_SOFTMAX_FIXTURE_ARTIFACT_KIND,
    )?;
    require_string_eq(audit, "gap_role", "attention_softmax")?;
    require_bool_eq(audit, "source_attention_score_cuda_parity_required", true)?;
    require_bool_eq(audit, "source_attention_score_cuda_parity_available", true)?;
    require_bool_eq(audit, "cpu_reference_available", true)?;
    require_string_eq(audit, "cuda_kernel_status", "missing_cuda_kernel")?;
    require_bool_eq(audit, "strict_cuda_ready", false)?;
    require_bool_eq(audit, "cpu_fallback_allowed", false)?;
    require_bool_eq(audit, "blocks_strict_cuda_one_layer", true)?;
    require_string_eq(audit, "next_required_proof", "cuda_attention_softmax_kernel_parity")?;
    let deps = array_field(audit, "input_dependencies")?;
    let deps = deps
        .iter()
        .map(|dep| {
            dep.as_str().ok_or_else(|| {
                anyhow!("attention_softmax_gap_audit.input_dependencies entries must be strings")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if deps != ["attention_scores"] {
        return Err(anyhow!(
            "attention_softmax_gap_audit.input_dependencies must identify attention_scores"
        ));
    }
    let candidate_order = array_field(audit, "candidate_order")?;
    let candidate_order = candidate_order
        .iter()
        .map(|role| {
            role.as_str().ok_or_else(|| {
                anyhow!("attention_softmax_gap_audit.candidate_order entries must be strings")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if candidate_order != DENSE_ONE_LAYER_GAP_CANDIDATE_ORDER {
        return Err(anyhow!(
            "attention_softmax_gap_audit.candidate_order must preserve the governed gap order"
        ));
    }
    require_bool_eq(audit, "dense_gguf_attention_softmax_fixture_extraction_claimed", true)?;
    require_bool_eq(audit, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(audit, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(audit, "speedup_claim", false)?;
    require_bool_eq(audit, "full_cuda_residency_claimed", false)?;

    let timing = object_field(receipt, "timing")?;
    require_null(timing, "kernel_time_ms")?;
    require_u64_eq(timing, "host_to_device_bytes", 0)?;
    require_u64_eq(timing, "device_to_host_bytes", 0)?;
    require_string_eq(timing, "transfer_timing_status", "not_measured_no_kernel")?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_score_cuda_parity_claimed", false)?;
    require_bool_eq(
        claim_boundary,
        "dense_gguf_attention_softmax_fixture_extraction_claimed",
        true,
    )?;
    require_bool_eq(claim_boundary, "dense_gguf_rope_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_norm_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "cpu_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

/// Validate a dense GGUF attention V-mix fixture extraction receipt.
///
/// This receipt records a CPU-reference `softmax(scores) x V` context fixture
/// after the attention-softmax CUDA parity boundary. It must not claim CUDA
/// V-mix parity, dense GGUF inference, speedup, full residency, or BitNet
/// packed I2_S/QK256 proof.
pub fn validate_dense_gguf_attention_v_mix_fixture_receipt_json(receipt: &Value) -> Result<()> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", DENSE_GGUF_ATTENTION_V_MIX_FIXTURE_ARTIFACT_KIND)?;
    require_string_eq(receipt, "claim", "dense_gguf_attention_v_mix_fixture_extracted")?;
    require_string_eq(receipt, "hardware_lane", "nvidia-rtx-5070-ti-cuda")?;
    require_string_non_empty(receipt, "inspection_source")?;
    require_null(receipt, "error")?;

    let model = object_field(receipt, "model")?;
    require_string_non_empty(model, "model_family")?;
    reject_bitnet_packed_marker(required_string(model, "model_family")?, "model.model_family")?;
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_sha256(model, "sha256")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(
        execution_path,
        "kernel_family",
        "cpu_reference_attention_v_mix_after_softmax",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    let plan = object_field(receipt, "execution_plan")?;
    require_string_eq(plan, "model_family", required_string(model, "model_family")?)?;
    require_string_eq(plan, "selected_route", "unsupported")?;
    require_string_eq(plan, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(plan, "selected_backend", "unsupported_strict_cuda")?;
    require_string_eq(plan, "runtime_api", "none")?;
    require_string_eq(plan, "strict_fallback_policy", "reject")?;
    require_bool_eq(plan, "dense_regular_llm_cuda", false)?;
    require_bool_eq(plan, "bitnet_packed_qk256_cuda", false)?;
    require_u64_eq(plan, "cuda_bitnet_qk256_ops", 0)?;
    require_u64_eq(plan, "cuda_dense_regular_llm_ops", 0)?;
    require_u64_eq(plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(plan, "unsupported_ops", 1)?;
    require_u64_eq(plan, "total_ops", 1)?;
    require_u64_eq(plan, "cuda_ops", 0)?;
    require_bool_eq(plan, "fallback_used", false)?;
    require_bool_eq(plan, "strict_cuda_ready", false)?;
    require_bool_eq(plan, "speedup_claim", false)?;
    require_bool_eq(plan, "full_cuda_residency_claimed", false)?;

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_bool_eq(descriptor, "required_roles_present", true)?;
    require_bool_eq(descriptor, "strict_descriptor_complete", true)?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;

    let fixture = object_field(receipt, "attention_v_mix_fixture")?;
    require_u64_eq(fixture, "schema", 1)?;
    require_string_eq(
        fixture,
        "source_artifact_kind",
        DENSE_GGUF_ATTENTION_V_MIX_FIXTURE_ARTIFACT_KIND,
    )?;
    require_string_eq(
        fixture,
        "source_attention_softmax_artifact_kind",
        DENSE_GGUF_ATTENTION_SOFTMAX_CUDA_PARITY_ARTIFACT_KIND,
    )?;
    require_string_non_empty(fixture, "source_attention_softmax_fixture_id")?;
    require_string_eq(
        fixture,
        "source_attention_v_artifact_kind",
        DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND,
    )?;
    require_string_eq(fixture, "source_attention_v_role", "attention_v")?;
    require_string_non_empty(fixture, "fixture_id")?;
    require_string_eq(fixture, "model_family", required_string(model, "model_family")?)?;
    require_string_eq(fixture, "architecture", required_string(model, "architecture")?)?;
    require_positive_u64(fixture, "q_heads")?;
    require_positive_u64(fixture, "kv_heads")?;
    require_positive_u64(fixture, "heads_per_kv_group")?;
    require_positive_u64(fixture, "head_dim")?;
    require_positive_u64(fixture, "seq_len")?;
    require_positive_u64(fixture, "row_count")?;
    require_positive_u64(fixture, "probability_count")?;
    require_positive_u64(fixture, "value_count")?;
    require_positive_u64(fixture, "context_count")?;
    require_sha256(fixture, "attention_probabilities_sha256")?;
    require_sha256(fixture, "value_states_sha256")?;
    require_sha256(fixture, "cpu_reference_context_sha256")?;
    let q_heads = required_u64(fixture, "q_heads")?;
    let kv_heads = required_u64(fixture, "kv_heads")?;
    let heads_per_kv_group = required_u64(fixture, "heads_per_kv_group")?;
    let seq_len = required_u64(fixture, "seq_len")?;
    let head_dim = required_u64(fixture, "head_dim")?;
    if q_heads % kv_heads != 0 || heads_per_kv_group != q_heads / kv_heads {
        return Err(anyhow!(
            "attention_v_mix_fixture heads_per_kv_group must match q_heads / kv_heads"
        ));
    }
    require_u64_eq(fixture, "row_count", q_heads * seq_len)?;
    require_u64_eq(fixture, "probability_count", q_heads * seq_len * seq_len)?;
    require_u64_eq(fixture, "value_count", kv_heads * seq_len * head_dim)?;
    require_u64_eq(fixture, "context_count", q_heads * seq_len * head_dim)?;
    let zero_probs =
        object_field(fixture, "causal_zero_probabilities")?.as_u64().ok_or_else(|| {
            anyhow!("attention_v_mix_fixture.causal_zero_probabilities must be unsigned")
        })?;
    if zero_probs >= q_heads * seq_len * seq_len {
        return Err(anyhow!(
            "attention_v_mix_fixture causal_zero_probabilities must leave finite probabilities"
        ));
    }
    require_non_negative_number(fixture, "max_context_abs")?;
    require_bool_eq(fixture, "cpu_reference_computed", true)?;
    require_string_eq(fixture, "cuda_kernel_status", "missing_cuda_kernel")?;
    require_bool_eq(fixture, "strict_cuda_ready", false)?;
    require_bool_eq(fixture, "cpu_fallback_allowed", false)?;
    require_string_eq(fixture, "transfer_timing_status", "not_measured_no_kernel")?;
    require_bool_eq(fixture, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(fixture, "dense_regular_llm_cuda_claimed", false)?;
    require_bool_eq(fixture, "cpu_cuda_parity_claimed", false)?;
    require_bool_eq(fixture, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(fixture, "speedup_claim", false)?;
    require_bool_eq(fixture, "full_cuda_residency_claimed", false)?;

    let audit = object_field(receipt, "attention_v_mix_gap_audit")?;
    require_u64_eq(audit, "schema", 1)?;
    require_string_eq(
        audit,
        "source_artifact_kind",
        DENSE_GGUF_ATTENTION_V_MIX_FIXTURE_ARTIFACT_KIND,
    )?;
    require_string_eq(audit, "gap_role", "attention_v_mix")?;
    require_bool_eq(audit, "source_attention_softmax_cuda_parity_required", true)?;
    require_bool_eq(audit, "source_attention_softmax_cuda_parity_available", true)?;
    require_bool_eq(audit, "source_attention_v_cuda_parity_required", true)?;
    require_bool_eq(audit, "source_attention_v_cuda_parity_available", true)?;
    require_bool_eq(audit, "cpu_reference_available", true)?;
    require_string_eq(audit, "cuda_kernel_status", "missing_cuda_kernel")?;
    require_bool_eq(audit, "strict_cuda_ready", false)?;
    require_bool_eq(audit, "cpu_fallback_allowed", false)?;
    require_bool_eq(audit, "blocks_strict_cuda_one_layer", true)?;
    require_string_eq(audit, "next_required_proof", "cuda_attention_v_mix_kernel_parity")?;
    let deps = array_field(audit, "input_dependencies")?;
    let deps = deps
        .iter()
        .map(|dep| {
            dep.as_str().ok_or_else(|| {
                anyhow!("attention_v_mix_gap_audit.input_dependencies entries must be strings")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if deps != ["attention_softmax", "attention_v"] {
        return Err(anyhow!(
            "attention_v_mix_gap_audit.input_dependencies must identify attention_softmax and attention_v"
        ));
    }
    let candidate_order = array_field(audit, "candidate_order")?;
    let candidate_order = candidate_order
        .iter()
        .map(|role| {
            role.as_str().ok_or_else(|| {
                anyhow!("attention_v_mix_gap_audit.candidate_order entries must be strings")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if candidate_order != DENSE_ONE_LAYER_ATTENTION_V_MIX_FIXTURE_GAP_CANDIDATE_ORDER
        && candidate_order != DENSE_ONE_LAYER_REMAINING_GAP_CANDIDATE_ORDER
    {
        return Err(anyhow!(
            "attention_v_mix_gap_audit.candidate_order must preserve the remaining gap order"
        ));
    }
    require_bool_eq(audit, "dense_gguf_attention_v_mix_fixture_extraction_claimed", true)?;
    require_bool_eq(audit, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(audit, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(audit, "speedup_claim", false)?;
    require_bool_eq(audit, "full_cuda_residency_claimed", false)?;

    let timing = object_field(receipt, "timing")?;
    require_null(timing, "kernel_time_ms")?;
    require_u64_eq(timing, "host_to_device_bytes", 0)?;
    require_u64_eq(timing, "device_to_host_bytes", 0)?;
    require_string_eq(timing, "transfer_timing_status", "not_measured_no_kernel")?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_score_cuda_parity_claimed", false)?;
    require_bool_eq(
        claim_boundary,
        "dense_gguf_attention_softmax_fixture_extraction_claimed",
        false,
    )?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_softmax_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_v_mix_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_rope_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_norm_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "cpu_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

/// Validate a dense GGUF MLP activation fixture extraction receipt.
///
/// This receipt records CPU-reference `SiLU(mlp_gate) * mlp_up` activation
/// values after the MLP gate/up CUDA parity boundary. It must not claim CUDA
/// MLP activation parity, dense GGUF inference, speedup, full residency, or
/// BitNet packed I2_S/QK256 proof.
pub fn validate_dense_gguf_mlp_activation_fixture_receipt_json(receipt: &Value) -> Result<()> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", DENSE_GGUF_MLP_ACTIVATION_FIXTURE_ARTIFACT_KIND)?;
    require_string_eq(receipt, "claim", "dense_gguf_mlp_activation_fixture_extracted")?;
    require_string_eq(receipt, "hardware_lane", "nvidia-rtx-5070-ti-cuda")?;
    require_string_non_empty(receipt, "inspection_source")?;
    require_null(receipt, "error")?;

    let model = object_field(receipt, "model")?;
    require_string_non_empty(model, "model_family")?;
    reject_bitnet_packed_marker(required_string(model, "model_family")?, "model.model_family")?;
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_sha256(model, "sha256")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "cpu_reference_mlp_activation")?;
    require_string_eq(
        execution_path,
        "quantization_family",
        "metadata_derived_mlp_activation_fixture",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    let plan = object_field(receipt, "execution_plan")?;
    require_string_eq(plan, "model_family", required_string(model, "model_family")?)?;
    require_string_eq(plan, "selected_route", "unsupported")?;
    require_string_eq(plan, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(plan, "selected_backend", "unsupported_strict_cuda")?;
    require_string_eq(plan, "runtime_api", "none")?;
    require_string_eq(plan, "strict_fallback_policy", "reject")?;
    require_bool_eq(plan, "dense_regular_llm_cuda", false)?;
    require_bool_eq(plan, "bitnet_packed_qk256_cuda", false)?;
    require_u64_eq(plan, "cuda_bitnet_qk256_ops", 0)?;
    require_u64_eq(plan, "cuda_dense_regular_llm_ops", 0)?;
    require_u64_eq(plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(plan, "unsupported_ops", 1)?;
    require_u64_eq(plan, "total_ops", 1)?;
    require_u64_eq(plan, "cuda_ops", 0)?;
    require_bool_eq(plan, "fallback_used", false)?;
    require_bool_eq(plan, "strict_cuda_ready", false)?;
    require_bool_eq(plan, "speedup_claim", false)?;
    require_bool_eq(plan, "full_cuda_residency_claimed", false)?;

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_bool_eq(descriptor, "required_roles_present", true)?;
    require_bool_eq(descriptor, "strict_descriptor_complete", true)?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;

    let fixture = object_field(receipt, "mlp_activation_fixture")?;
    require_u64_eq(fixture, "schema", 1)?;
    require_string_eq(
        fixture,
        "source_artifact_kind",
        DENSE_GGUF_MLP_ACTIVATION_FIXTURE_ARTIFACT_KIND,
    )?;
    require_string_eq(
        fixture,
        "source_mlp_gate_artifact_kind",
        DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND,
    )?;
    require_string_eq(
        fixture,
        "source_mlp_up_artifact_kind",
        DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND,
    )?;
    require_string_eq(fixture, "source_mlp_gate_role", "mlp_gate")?;
    require_string_eq(fixture, "source_mlp_up_role", "mlp_up")?;
    require_string_non_empty(fixture, "source_mlp_gate_fixture_id")?;
    require_string_non_empty(fixture, "source_mlp_up_fixture_id")?;
    require_string_non_empty(fixture, "source_mlp_gate_tensor")?;
    require_string_non_empty(fixture, "source_mlp_up_tensor")?;
    require_string_non_empty(fixture, "fixture_id")?;
    require_string_eq(fixture, "model_family", required_string(model, "model_family")?)?;
    require_string_eq(fixture, "architecture", required_string(model, "architecture")?)?;
    require_string_eq(fixture, "activation_kind", "silu_gate_times_up")?;
    require_positive_u64(fixture, "activation_count")?;
    require_positive_u64(fixture, "gate_output_count")?;
    require_positive_u64(fixture, "up_output_count")?;
    let activation_count = required_u64(fixture, "activation_count")?;
    require_u64_eq(fixture, "gate_output_count", activation_count)?;
    require_u64_eq(fixture, "up_output_count", activation_count)?;
    require_sha256(fixture, "gate_output_sha256")?;
    require_sha256(fixture, "up_output_sha256")?;
    require_sha256(fixture, "cpu_reference_activation_sha256")?;
    require_non_negative_number(fixture, "max_activation_abs")?;
    require_bool_eq(fixture, "cpu_reference_computed", true)?;
    require_string_eq(fixture, "cuda_kernel_status", "missing_cuda_kernel")?;
    require_bool_eq(fixture, "strict_cuda_ready", false)?;
    require_bool_eq(fixture, "cpu_fallback_allowed", false)?;
    require_string_eq(fixture, "transfer_timing_status", "not_measured_no_kernel")?;
    require_bool_eq(fixture, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(fixture, "dense_regular_llm_cuda_claimed", false)?;
    require_bool_eq(fixture, "cpu_cuda_parity_claimed", false)?;
    require_bool_eq(fixture, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(fixture, "speedup_claim", false)?;
    require_bool_eq(fixture, "full_cuda_residency_claimed", false)?;

    let audit = object_field(receipt, "mlp_activation_gap_audit")?;
    require_u64_eq(audit, "schema", 1)?;
    require_string_eq(
        audit,
        "source_artifact_kind",
        DENSE_GGUF_MLP_ACTIVATION_FIXTURE_ARTIFACT_KIND,
    )?;
    require_string_eq(audit, "gap_role", "mlp_activation")?;
    require_bool_eq(audit, "source_mlp_gate_cuda_parity_required", true)?;
    require_bool_eq(audit, "source_mlp_gate_cuda_parity_available", true)?;
    require_bool_eq(audit, "source_mlp_up_cuda_parity_required", true)?;
    require_bool_eq(audit, "source_mlp_up_cuda_parity_available", true)?;
    require_bool_eq(audit, "cpu_reference_available", true)?;
    require_string_eq(audit, "cuda_kernel_status", "missing_cuda_kernel")?;
    require_bool_eq(audit, "strict_cuda_ready", false)?;
    require_bool_eq(audit, "cpu_fallback_allowed", false)?;
    require_bool_eq(audit, "blocks_strict_cuda_one_layer", true)?;
    require_string_eq(audit, "next_required_proof", "cuda_mlp_activation_kernel_parity")?;
    let deps = array_field(audit, "input_dependencies")?;
    let deps = deps
        .iter()
        .map(|dep| {
            dep.as_str().ok_or_else(|| {
                anyhow!("mlp_activation_gap_audit.input_dependencies entries must be strings")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if deps != ["mlp_gate", "mlp_up"] {
        return Err(anyhow!(
            "mlp_activation_gap_audit.input_dependencies must identify mlp_gate and mlp_up"
        ));
    }
    let candidate_order = array_field(audit, "candidate_order")?;
    let candidate_order = candidate_order
        .iter()
        .map(|role| {
            role.as_str().ok_or_else(|| {
                anyhow!("mlp_activation_gap_audit.candidate_order entries must be strings")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if candidate_order != DENSE_ONE_LAYER_REMAINING_GAP_CANDIDATE_ORDER {
        return Err(anyhow!(
            "mlp_activation_gap_audit.candidate_order must preserve the remaining gap order"
        ));
    }
    require_bool_eq(audit, "dense_gguf_mlp_activation_fixture_extraction_claimed", true)?;
    require_bool_eq(audit, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(audit, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(audit, "speedup_claim", false)?;
    require_bool_eq(audit, "full_cuda_residency_claimed", false)?;

    let timing = object_field(receipt, "timing")?;
    require_null(timing, "kernel_time_ms")?;
    require_u64_eq(timing, "host_to_device_bytes", 0)?;
    require_u64_eq(timing, "device_to_host_bytes", 0)?;
    require_string_eq(timing, "transfer_timing_status", "not_measured_no_kernel")?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_score_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_softmax_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_v_mix_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_mlp_activation_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_rope_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_norm_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "cpu_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

/// Validate dense GGUF MLP activation CUDA parity evidence.
///
/// This receipt bridges the MLP activation fixture into a strict CUDA F32
/// SiLU(gate) * up kernel. It remains fixture-level evidence and must not
/// claim dense GGUF inference, Qwen token/decode/chat, speedup, full residency,
/// or BitNet packed I2_S/QK256 proof.
pub fn validate_dense_gguf_mlp_activation_cuda_parity_receipt_json(receipt: &Value) -> Result<()> {
    let stats = validate_cuda_receipt_common(
        receipt,
        DENSE_GGUF_MLP_ACTIVATION_CUDA_PARITY_ARTIFACT_KIND,
        "dense_gguf_mlp_activation_cuda_parity_tested",
    )?;
    require_string_eq(stats, "kernel_id", "dense_mlp_activation_f32_cuda")?;
    require_u64_eq(stats, "invocations", 1)?;
    require_u64_eq(stats, "fallback_invocations", 0)?;
    require_u64_eq(stats, "kernel_launches", 1)?;
    require_bool_eq(receipt, "speedup_claim", false)?;

    let model = object_field(receipt, "model")?;
    require_string_non_empty(model, "model_family")?;
    reject_bitnet_packed_marker(required_string(model, "model_family")?, "model.model_family")?;
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_sha256(model, "sha256")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "dense_f32_mlp_activation")?;
    require_string_eq(
        execution_path,
        "quantization_family",
        "metadata_derived_mlp_activation_fixture",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    validate_dense_regular_llm_execution_plan(receipt)?;
    let plan = object_field(receipt, "execution_plan")?;
    require_u64_eq(plan, "total_ops", 1)?;
    require_u64_eq(plan, "cuda_ops", 1)?;
    require_u64_eq(plan, "cuda_dense_regular_llm_ops", 1)?;
    require_u64_eq(plan, "cuda_bitnet_qk256_ops", 0)?;
    require_u64_eq(plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(plan, "unsupported_ops", 0)?;
    require_bool_eq(plan, "strict_cuda_ready", true)?;

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_bool_eq(descriptor, "required_roles_present", true)?;
    require_bool_eq(descriptor, "strict_descriptor_complete", true)?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;

    let fixture = object_field(receipt, "mlp_activation_fixture")?;
    require_u64_eq(fixture, "schema", 1)?;
    require_string_eq(
        fixture,
        "source_artifact_kind",
        DENSE_GGUF_MLP_ACTIVATION_FIXTURE_ARTIFACT_KIND,
    )?;
    require_string_eq(
        fixture,
        "source_mlp_gate_artifact_kind",
        DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND,
    )?;
    require_string_eq(
        fixture,
        "source_mlp_up_artifact_kind",
        DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND,
    )?;
    require_string_eq(fixture, "source_mlp_gate_role", "mlp_gate")?;
    require_string_eq(fixture, "source_mlp_up_role", "mlp_up")?;
    require_string_non_empty(fixture, "source_mlp_gate_fixture_id")?;
    require_string_non_empty(fixture, "source_mlp_up_fixture_id")?;
    require_string_non_empty(fixture, "source_mlp_gate_tensor")?;
    require_string_non_empty(fixture, "source_mlp_up_tensor")?;
    require_string_non_empty(fixture, "fixture_id")?;
    require_string_eq(fixture, "model_family", required_string(model, "model_family")?)?;
    require_string_eq(fixture, "architecture", required_string(model, "architecture")?)?;
    require_string_eq(fixture, "activation_kind", "silu_gate_times_up")?;
    require_positive_u64(fixture, "activation_count")?;
    require_positive_u64(fixture, "gate_output_count")?;
    require_positive_u64(fixture, "up_output_count")?;
    let activation_count = required_u64(fixture, "activation_count")?;
    require_u64_eq(fixture, "gate_output_count", activation_count)?;
    require_u64_eq(fixture, "up_output_count", activation_count)?;
    require_u64_eq(fixture, "compared_activations", activation_count)?;
    require_sha256(fixture, "gate_output_sha256")?;
    require_sha256(fixture, "up_output_sha256")?;
    require_sha256(fixture, "cpu_reference_activation_sha256")?;
    require_non_negative_number(fixture, "max_activation_abs")?;
    require_string_eq(fixture, "cuda_input_dtype", "f32")?;
    require_string_eq(fixture, "cuda_output_dtype", "f32")?;
    require_string_eq(fixture, "cuda_kernel_status", "parity_passed")?;
    require_bool_eq(fixture, "strict_cuda_ready", true)?;
    require_bool_eq(fixture, "cpu_fallback_allowed", false)?;
    require_string_eq(fixture, "transfer_timing_status", "bytes_measured_time_unmeasured")?;
    require_bool_eq(fixture, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(fixture, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(fixture, "cpu_cuda_parity_claimed", true)?;
    require_bool_eq(fixture, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(fixture, "speedup_claim", false)?;
    require_bool_eq(fixture, "full_cuda_residency_claimed", false)?;

    let audit = object_field(receipt, "mlp_activation_gap_audit")?;
    require_u64_eq(audit, "schema", 1)?;
    require_string_eq(
        audit,
        "source_artifact_kind",
        DENSE_GGUF_MLP_ACTIVATION_FIXTURE_ARTIFACT_KIND,
    )?;
    require_string_eq(audit, "gap_role", "mlp_activation")?;
    require_bool_eq(audit, "source_mlp_gate_cuda_parity_required", true)?;
    require_bool_eq(audit, "source_mlp_gate_cuda_parity_available", true)?;
    require_bool_eq(audit, "source_mlp_up_cuda_parity_required", true)?;
    require_bool_eq(audit, "source_mlp_up_cuda_parity_available", true)?;
    require_bool_eq(audit, "cpu_reference_available", true)?;
    require_string_eq(audit, "cuda_kernel_status", "parity_passed")?;
    require_bool_eq(audit, "strict_cuda_ready", true)?;
    require_bool_eq(audit, "cpu_fallback_allowed", false)?;
    require_bool_eq(audit, "blocks_strict_cuda_one_layer", false)?;
    require_string_eq(audit, "next_required_proof", "one_layer_route_promotion")?;
    let deps = array_field(audit, "input_dependencies")?;
    let deps = deps
        .iter()
        .map(|dep| {
            dep.as_str().ok_or_else(|| {
                anyhow!("mlp_activation_gap_audit.input_dependencies entries must be strings")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if deps != ["mlp_gate", "mlp_up"] {
        return Err(anyhow!(
            "mlp_activation_gap_audit.input_dependencies must identify mlp_gate and mlp_up"
        ));
    }
    let candidate_order = array_field(audit, "candidate_order")?;
    let candidate_order = candidate_order
        .iter()
        .map(|role| {
            role.as_str().ok_or_else(|| {
                anyhow!("mlp_activation_gap_audit.candidate_order entries must be strings")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if candidate_order != DENSE_ONE_LAYER_REMAINING_GAP_CANDIDATE_ORDER {
        return Err(anyhow!(
            "mlp_activation_gap_audit.candidate_order must preserve the remaining gap order"
        ));
    }
    require_bool_eq(audit, "dense_gguf_mlp_activation_fixture_extraction_claimed", true)?;
    require_bool_eq(audit, "dense_gguf_mlp_activation_cuda_parity_claimed", true)?;
    require_bool_eq(audit, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(audit, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(audit, "speedup_claim", false)?;
    require_bool_eq(audit, "full_cuda_residency_claimed", false)?;

    let parity = object_field(receipt, "parity")?;
    require_string_eq(parity, "reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(parity, "target_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(parity, "kernel_id", "dense_mlp_activation_f32_cuda")?;
    require_string_eq(parity, "fixture_id", required_string(fixture, "fixture_id")?)?;
    require_bool_eq(parity, "passed", true)?;
    require_non_negative_number(parity, "max_abs_error")?;
    require_non_negative_number(parity, "mean_abs_error")?;
    require_non_negative_number(parity, "tolerance")?;
    require_string_non_empty(parity, "tolerance_source")?;
    require_u64_eq(parity, "compared_activations", activation_count)?;
    require_null(parity, "first_divergence")?;

    let timing = object_field(receipt, "timing")?;
    require_optional_non_negative_number(timing, "kernel_time_ms")?;
    require_string_eq(timing, "transfer_timing_status", "bytes_measured_time_unmeasured")?;
    require_u64_eq(
        timing,
        "host_to_device_bytes",
        object_field(stats, "host_to_device_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("kernel_stats.host_to_device_bytes must be an integer"))?,
    )?;
    require_u64_eq(
        timing,
        "device_to_host_bytes",
        object_field(stats, "device_to_host_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("kernel_stats.device_to_host_bytes must be an integer"))?,
    )?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_score_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_softmax_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_v_mix_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_mlp_activation_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_mlp_activation_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_norm_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "cpu_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    let residency = object_field(receipt, "tensor_residency")?;
    require_string_eq(residency, "scope", "single_dense_gguf_mlp_activation_fixture")?;
    require_string_eq(residency, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(residency, "fixture_id", required_string(fixture, "fixture_id")?)?;
    require_bool_eq(residency, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(residency, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(residency, "persistent_session_residency_claimed", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    require_bool_eq(residency, "input_tensors_uploaded_once", true)?;
    require_bool_eq(residency, "output_tensor_cuda_resident_during_kernel", true)?;
    require_bool_eq(residency, "host_device_transfer_accounting_matches_kernel_stats", true)?;

    let inputs = array_field(residency, "inputs")?;
    if inputs.len() != 2 {
        return Err(anyhow!(
            "MLP activation tensor_residency.inputs must contain gate and up outputs"
        ));
    }
    let gate = &inputs[0];
    require_string_eq(gate, "name", "dense_gguf_mlp_gate_output")?;
    require_string_eq(gate, "dtype", "f32")?;
    require_string_eq(gate, "device_residency", "cuda_device_buffer")?;
    require_u64_eq(gate, "upload_count", 1)?;
    require_string_eq(gate, "reuse_scope", "single_fixture_launch")?;
    require_u64_eq(gate, "host_bytes", activation_count * 4)?;

    let up = &inputs[1];
    require_string_eq(up, "name", "dense_gguf_mlp_up_output")?;
    require_string_eq(up, "dtype", "f32")?;
    require_string_eq(up, "device_residency", "cuda_device_buffer")?;
    require_u64_eq(up, "upload_count", 1)?;
    require_string_eq(up, "reuse_scope", "single_fixture_launch")?;
    require_u64_eq(up, "host_bytes", activation_count * 4)?;

    let outputs = array_field(residency, "outputs")?;
    if outputs.len() != 1 {
        return Err(anyhow!("MLP activation tensor_residency.outputs must contain activation"));
    }
    let output = &outputs[0];
    require_string_eq(output, "name", "dense_gguf_mlp_activation")?;
    require_string_eq(output, "dtype", "f32")?;
    require_string_eq(output, "device_residency", "cuda_device_buffer")?;
    require_string_eq(output, "download_scope", "parity_check_only")?;
    require_u64_eq(output, "device_to_host_bytes", activation_count * 4)?;

    let allocation = object_field(residency, "allocation")?;
    require_u64_eq(allocation, "device_buffer_count", 3)?;
    require_u64_eq(allocation, "persistent_handle_count", 0)?;
    require_bool_eq(allocation, "persistent_handles_claimed", false)?;

    let transfer = object_field(residency, "transfer_accounting")?;
    require_string_eq(transfer, "status", "measured")?;
    require_u64_eq(transfer, "host_to_device_bytes", activation_count * 2 * 4)?;
    require_u64_eq(transfer, "device_to_host_bytes", activation_count * 4)?;
    require_u64_eq(transfer, "kernel_invocations", 1)?;
    require_u64_eq(transfer, "kernel_launches", 1)?;
    require_u64_eq(
        stats,
        "host_to_device_bytes",
        object_field(transfer, "host_to_device_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("transfer.host_to_device_bytes must be an integer"))?,
    )?;
    require_u64_eq(
        stats,
        "device_to_host_bytes",
        object_field(transfer, "device_to_host_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("transfer.device_to_host_bytes must be an integer"))?,
    )?;

    Ok(())
}

/// Validate dense GGUF attention V-mix CUDA parity evidence.
///
/// This receipt bridges the attention V-mix fixture into a strict CUDA context
/// kernel. It remains fixture-level evidence and must not claim dense GGUF
/// inference, Qwen token/decode/chat, speedup, full residency, or BitNet
/// packed I2_S/QK256 proof.
pub fn validate_dense_gguf_attention_v_mix_cuda_parity_receipt_json(receipt: &Value) -> Result<()> {
    let stats = validate_cuda_receipt_common(
        receipt,
        DENSE_GGUF_ATTENTION_V_MIX_CUDA_PARITY_ARTIFACT_KIND,
        "dense_gguf_attention_v_mix_cuda_parity_tested",
    )?;
    require_string_eq(stats, "kernel_id", "dense_attention_v_mix_f32_cuda")?;
    require_u64_eq(stats, "invocations", 1)?;
    require_u64_eq(stats, "fallback_invocations", 0)?;
    require_u64_eq(stats, "kernel_launches", 1)?;
    require_bool_eq(receipt, "speedup_claim", false)?;

    let model = object_field(receipt, "model")?;
    require_string_non_empty(model, "model_family")?;
    reject_bitnet_packed_marker(required_string(model, "model_family")?, "model.model_family")?;
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_sha256(model, "sha256")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "dense_f32_attention_v_mix")?;
    require_string_eq(
        execution_path,
        "quantization_family",
        "metadata_derived_attention_v_mix_fixture",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    validate_dense_regular_llm_execution_plan(receipt)?;
    let plan = object_field(receipt, "execution_plan")?;
    require_u64_eq(plan, "total_ops", 1)?;
    require_u64_eq(plan, "cuda_ops", 1)?;
    require_u64_eq(plan, "cuda_dense_regular_llm_ops", 1)?;
    require_u64_eq(plan, "cuda_bitnet_qk256_ops", 0)?;
    require_u64_eq(plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(plan, "unsupported_ops", 0)?;
    require_bool_eq(plan, "strict_cuda_ready", true)?;

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_bool_eq(descriptor, "required_roles_present", true)?;
    require_bool_eq(descriptor, "strict_descriptor_complete", true)?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;

    let fixture = object_field(receipt, "attention_v_mix_fixture")?;
    require_u64_eq(fixture, "schema", 1)?;
    require_string_eq(
        fixture,
        "source_artifact_kind",
        DENSE_GGUF_ATTENTION_V_MIX_FIXTURE_ARTIFACT_KIND,
    )?;
    require_string_eq(
        fixture,
        "source_attention_softmax_artifact_kind",
        DENSE_GGUF_ATTENTION_SOFTMAX_CUDA_PARITY_ARTIFACT_KIND,
    )?;
    require_string_non_empty(fixture, "source_attention_softmax_fixture_id")?;
    require_string_eq(
        fixture,
        "source_attention_v_artifact_kind",
        DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND,
    )?;
    require_string_eq(fixture, "source_attention_v_role", "attention_v")?;
    require_string_non_empty(fixture, "fixture_id")?;
    require_string_eq(fixture, "model_family", required_string(model, "model_family")?)?;
    require_string_eq(fixture, "architecture", required_string(model, "architecture")?)?;
    require_positive_u64(fixture, "q_heads")?;
    require_positive_u64(fixture, "kv_heads")?;
    require_positive_u64(fixture, "heads_per_kv_group")?;
    require_positive_u64(fixture, "head_dim")?;
    require_positive_u64(fixture, "seq_len")?;
    require_positive_u64(fixture, "row_count")?;
    require_positive_u64(fixture, "probability_count")?;
    require_positive_u64(fixture, "value_count")?;
    require_positive_u64(fixture, "context_count")?;
    require_positive_u64(fixture, "compared_context_values")?;
    require_sha256(fixture, "attention_probabilities_sha256")?;
    require_sha256(fixture, "value_states_sha256")?;
    require_sha256(fixture, "cpu_reference_context_sha256")?;
    let q_heads = required_u64(fixture, "q_heads")?;
    let kv_heads = required_u64(fixture, "kv_heads")?;
    let heads_per_kv_group = required_u64(fixture, "heads_per_kv_group")?;
    let seq_len = required_u64(fixture, "seq_len")?;
    let head_dim = required_u64(fixture, "head_dim")?;
    if q_heads % kv_heads != 0 || heads_per_kv_group != q_heads / kv_heads {
        return Err(anyhow!(
            "attention_v_mix_fixture heads_per_kv_group must match q_heads / kv_heads"
        ));
    }
    let probability_count = q_heads * seq_len * seq_len;
    let value_count = kv_heads * seq_len * head_dim;
    let context_count = q_heads * seq_len * head_dim;
    require_u64_eq(fixture, "row_count", q_heads * seq_len)?;
    require_u64_eq(fixture, "probability_count", probability_count)?;
    require_u64_eq(fixture, "value_count", value_count)?;
    require_u64_eq(fixture, "context_count", context_count)?;
    require_u64_eq(fixture, "compared_context_values", context_count)?;
    let zero_probs =
        object_field(fixture, "causal_zero_probabilities")?.as_u64().ok_or_else(|| {
            anyhow!("attention_v_mix_fixture.causal_zero_probabilities must be unsigned")
        })?;
    if zero_probs >= probability_count {
        return Err(anyhow!(
            "attention_v_mix_fixture causal_zero_probabilities must leave finite probabilities"
        ));
    }
    require_non_negative_number(fixture, "max_context_abs")?;
    require_string_eq(fixture, "cuda_input_dtype", "f32")?;
    require_string_eq(fixture, "cuda_output_dtype", "f32")?;
    require_string_eq(fixture, "cuda_kernel_status", "parity_passed")?;
    require_bool_eq(fixture, "strict_cuda_ready", true)?;
    require_bool_eq(fixture, "cpu_fallback_allowed", false)?;
    require_string_eq(fixture, "transfer_timing_status", "bytes_measured_time_unmeasured")?;
    require_bool_eq(fixture, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(fixture, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(fixture, "cpu_cuda_parity_claimed", true)?;
    require_bool_eq(fixture, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(fixture, "speedup_claim", false)?;
    require_bool_eq(fixture, "full_cuda_residency_claimed", false)?;

    let parity = object_field(receipt, "parity")?;
    require_string_eq(parity, "reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(parity, "target_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(parity, "kernel_id", "dense_attention_v_mix_f32_cuda")?;
    require_string_eq(parity, "fixture_id", required_string(fixture, "fixture_id")?)?;
    require_bool_eq(parity, "passed", true)?;
    require_non_negative_number(parity, "max_abs_error")?;
    require_non_negative_number(parity, "mean_abs_error")?;
    require_non_negative_number(parity, "tolerance")?;
    require_string_non_empty(parity, "tolerance_source")?;
    require_u64_eq(parity, "compared_context_values", context_count)?;
    require_u64_eq(parity, "causal_zero_probabilities", zero_probs)?;
    require_null(parity, "first_divergence")?;

    let timing = object_field(receipt, "timing")?;
    require_optional_non_negative_number(timing, "kernel_time_ms")?;
    require_u64_eq(
        timing,
        "host_to_device_bytes",
        object_field(stats, "host_to_device_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("kernel_stats.host_to_device_bytes must be an integer"))?,
    )?;
    require_u64_eq(
        timing,
        "device_to_host_bytes",
        object_field(stats, "device_to_host_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("kernel_stats.device_to_host_bytes must be an integer"))?,
    )?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_score_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_softmax_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_v_mix_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_v_mix_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_norm_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "cpu_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    let residency = object_field(receipt, "tensor_residency")?;
    require_string_eq(residency, "scope", "single_dense_gguf_attention_v_mix_fixture")?;
    require_string_eq(residency, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(residency, "fixture_id", required_string(fixture, "fixture_id")?)?;
    require_bool_eq(residency, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(residency, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(residency, "persistent_session_residency_claimed", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    require_bool_eq(residency, "input_tensors_uploaded_once", true)?;
    require_bool_eq(residency, "output_tensor_cuda_resident_during_kernel", true)?;
    require_bool_eq(residency, "host_device_transfer_accounting_matches_kernel_stats", true)?;

    let inputs = array_field(residency, "inputs")?;
    if inputs.len() != 2 {
        return Err(anyhow!(
            "attention V-mix tensor_residency.inputs must contain probabilities and values"
        ));
    }
    let probabilities = &inputs[0];
    require_string_eq(probabilities, "name", "dense_gguf_attention_probabilities")?;
    require_string_eq(probabilities, "dtype", "f32")?;
    require_string_eq(probabilities, "device_residency", "cuda_device_buffer")?;
    require_u64_eq(probabilities, "upload_count", 1)?;
    require_string_eq(probabilities, "reuse_scope", "single_fixture_launch")?;
    require_u64_eq(probabilities, "host_bytes", probability_count * 4)?;

    let values = &inputs[1];
    require_string_eq(values, "name", "dense_gguf_attention_values")?;
    require_string_eq(values, "dtype", "f32")?;
    require_string_eq(values, "device_residency", "cuda_device_buffer")?;
    require_u64_eq(values, "upload_count", 1)?;
    require_string_eq(values, "reuse_scope", "single_fixture_launch")?;
    require_u64_eq(values, "host_bytes", value_count * 4)?;

    let outputs = array_field(residency, "outputs")?;
    if outputs.len() != 1 {
        return Err(anyhow!("attention V-mix tensor_residency.outputs must contain context"));
    }
    let output = &outputs[0];
    require_string_eq(output, "name", "dense_gguf_attention_context")?;
    require_string_eq(output, "dtype", "f32")?;
    require_string_eq(output, "device_residency", "cuda_device_buffer")?;
    require_string_eq(output, "download_scope", "parity_check_only")?;
    require_u64_eq(output, "device_to_host_bytes", context_count * 4)?;

    let allocation = object_field(residency, "allocation")?;
    require_u64_eq(allocation, "device_buffer_count", 3)?;
    require_u64_eq(allocation, "persistent_handle_count", 0)?;
    require_bool_eq(allocation, "persistent_handles_claimed", false)?;

    let transfer = object_field(residency, "transfer_accounting")?;
    require_string_eq(transfer, "status", "measured")?;
    require_u64_eq(transfer, "host_to_device_bytes", (probability_count + value_count) * 4)?;
    require_u64_eq(transfer, "device_to_host_bytes", context_count * 4)?;
    require_u64_eq(transfer, "kernel_invocations", 1)?;
    require_u64_eq(transfer, "kernel_launches", 1)?;
    require_u64_eq(
        stats,
        "host_to_device_bytes",
        object_field(transfer, "host_to_device_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("transfer.host_to_device_bytes must be an integer"))?,
    )?;
    require_u64_eq(
        stats,
        "device_to_host_bytes",
        object_field(transfer, "device_to_host_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("transfer.device_to_host_bytes must be an integer"))?,
    )?;

    Ok(())
}

/// Validate dense GGUF attention-softmax CUDA parity evidence.
///
/// This receipt bridges the attention-softmax fixture into a strict CUDA
/// softmax kernel. It must still reject dense GGUF inference, Qwen
/// token/decode/chat, speedup, persistent-session, full CUDA residency, and
/// BitNet packed I2_S/QK256 proof claims.
pub fn validate_dense_gguf_attention_softmax_cuda_parity_receipt_json(
    receipt: &Value,
) -> Result<()> {
    let stats = validate_cuda_receipt_common(
        receipt,
        DENSE_GGUF_ATTENTION_SOFTMAX_CUDA_PARITY_ARTIFACT_KIND,
        "dense_gguf_attention_softmax_cuda_parity_tested",
    )?;
    require_string_eq(stats, "kernel_id", "dense_attention_softmax_f32_cuda")?;
    require_u64_eq(stats, "invocations", 1)?;
    require_u64_eq(stats, "fallback_invocations", 0)?;
    require_u64_eq(stats, "kernel_launches", 1)?;
    require_bool_eq(receipt, "speedup_claim", false)?;

    let model = object_field(receipt, "model")?;
    require_string_non_empty(model, "model_family")?;
    reject_bitnet_packed_marker(required_string(model, "model_family")?, "model.model_family")?;
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_sha256(model, "sha256")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "dense_f32_attention_softmax")?;
    require_string_eq(
        execution_path,
        "quantization_family",
        "metadata_derived_attention_softmax_fixture",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    validate_dense_regular_llm_execution_plan(receipt)?;
    let plan = object_field(receipt, "execution_plan")?;
    require_u64_eq(plan, "total_ops", 1)?;
    require_u64_eq(plan, "cuda_ops", 1)?;
    require_u64_eq(plan, "cuda_dense_regular_llm_ops", 1)?;
    require_u64_eq(plan, "cuda_bitnet_qk256_ops", 0)?;
    require_u64_eq(plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(plan, "unsupported_ops", 0)?;
    require_bool_eq(plan, "strict_cuda_ready", true)?;

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_bool_eq(descriptor, "required_roles_present", true)?;
    require_bool_eq(descriptor, "strict_descriptor_complete", true)?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;

    let fixture = object_field(receipt, "attention_softmax_fixture")?;
    require_u64_eq(fixture, "schema", 1)?;
    require_string_eq(
        fixture,
        "source_artifact_kind",
        DENSE_GGUF_ATTENTION_SOFTMAX_FIXTURE_ARTIFACT_KIND,
    )?;
    require_string_eq(
        fixture,
        "source_attention_score_artifact_kind",
        DENSE_GGUF_ATTENTION_SCORE_CUDA_PARITY_ARTIFACT_KIND,
    )?;
    require_string_non_empty(fixture, "source_attention_score_fixture_id")?;
    require_string_non_empty(fixture, "fixture_id")?;
    require_string_eq(fixture, "model_family", required_string(model, "model_family")?)?;
    require_string_eq(fixture, "architecture", required_string(model, "architecture")?)?;
    require_positive_u64(fixture, "q_heads")?;
    require_positive_u64(fixture, "kv_heads")?;
    require_positive_u64(fixture, "seq_len")?;
    require_positive_u64(fixture, "row_count")?;
    require_sha256(fixture, "attention_scores_sha256")?;
    require_sha256(fixture, "cpu_reference_probabilities_sha256")?;
    let q_heads = object_field(fixture, "q_heads")?
        .as_u64()
        .ok_or_else(|| anyhow!("attention_softmax_fixture.q_heads must be an unsigned integer"))?;
    let seq_len = object_field(fixture, "seq_len")?
        .as_u64()
        .ok_or_else(|| anyhow!("attention_softmax_fixture.seq_len must be an unsigned integer"))?;
    require_u64_eq(fixture, "row_count", q_heads * seq_len)?;
    let expected_probability_count = q_heads * seq_len * seq_len;
    require_u64_eq(fixture, "probability_count", expected_probability_count)?;
    require_positive_u64(fixture, "compared_probabilities")?;
    require_u64_eq(fixture, "compared_probabilities", expected_probability_count)?;
    require_non_negative_number(fixture, "max_row_sum_abs_error")?;
    require_string_eq(fixture, "cuda_input_dtype", "f32")?;
    require_string_eq(fixture, "cuda_output_dtype", "f32")?;
    require_string_eq(fixture, "cuda_kernel_status", "parity_passed")?;
    require_bool_eq(fixture, "strict_cuda_ready", true)?;
    require_bool_eq(fixture, "cpu_fallback_allowed", false)?;
    require_string_eq(fixture, "transfer_timing_status", "bytes_measured_time_unmeasured")?;
    require_bool_eq(fixture, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(fixture, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(fixture, "cpu_cuda_parity_claimed", true)?;
    require_bool_eq(fixture, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(fixture, "speedup_claim", false)?;
    require_bool_eq(fixture, "full_cuda_residency_claimed", false)?;

    let parity = object_field(receipt, "parity")?;
    require_string_eq(parity, "reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(parity, "target_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(parity, "kernel_id", "dense_attention_softmax_f32_cuda")?;
    require_string_eq(parity, "fixture_id", required_string(fixture, "fixture_id")?)?;
    require_bool_eq(parity, "passed", true)?;
    require_non_negative_number(parity, "max_abs_error")?;
    require_non_negative_number(parity, "mean_abs_error")?;
    require_non_negative_number(parity, "tolerance")?;
    require_string_non_empty(parity, "tolerance_source")?;
    require_u64_eq(parity, "compared_probabilities", expected_probability_count)?;
    require_null(parity, "first_divergence")?;

    let timing = object_field(receipt, "timing")?;
    require_optional_non_negative_number(timing, "kernel_time_ms")?;
    require_u64_eq(
        timing,
        "host_to_device_bytes",
        object_field(stats, "host_to_device_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("kernel_stats.host_to_device_bytes must be an integer"))?,
    )?;
    require_u64_eq(
        timing,
        "device_to_host_bytes",
        object_field(stats, "device_to_host_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("kernel_stats.device_to_host_bytes must be an integer"))?,
    )?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_score_cuda_parity_claimed", true)?;
    require_bool_eq(
        claim_boundary,
        "dense_gguf_attention_softmax_fixture_extraction_claimed",
        true,
    )?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_softmax_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_norm_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "cpu_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    let residency = object_field(receipt, "tensor_residency")?;
    require_string_eq(residency, "scope", "single_dense_gguf_attention_softmax_fixture")?;
    require_string_eq(residency, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(residency, "fixture_id", required_string(fixture, "fixture_id")?)?;
    require_bool_eq(residency, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(residency, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(residency, "persistent_session_residency_claimed", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    require_bool_eq(residency, "input_tensors_uploaded_once", true)?;
    require_bool_eq(residency, "output_tensor_cuda_resident_during_kernel", true)?;
    require_bool_eq(residency, "host_device_transfer_accounting_matches_kernel_stats", true)?;

    let inputs = array_field(residency, "inputs")?;
    if inputs.len() != 1 {
        return Err(anyhow!(
            "attention-softmax tensor_residency.inputs must contain the score input"
        ));
    }
    let input = &inputs[0];
    require_string_eq(input, "name", "dense_gguf_attention_scores")?;
    require_string_eq(input, "dtype", "f32")?;
    require_string_eq(input, "device_residency", "cuda_device_buffer")?;
    require_u64_eq(input, "upload_count", 1)?;
    require_string_eq(input, "reuse_scope", "single_fixture_launch")?;
    require_positive_u64(input, "host_bytes")?;
    let h2d = object_field(input, "host_bytes")?
        .as_u64()
        .ok_or_else(|| anyhow!("input.host_bytes must be an unsigned integer"))?;

    let outputs = array_field(residency, "outputs")?;
    if outputs.len() != 1 {
        return Err(anyhow!(
            "attention-softmax tensor_residency.outputs must contain probabilities"
        ));
    }
    let output = &outputs[0];
    require_string_eq(output, "name", "dense_gguf_attention_probabilities")?;
    require_string_eq(output, "dtype", "f32")?;
    require_string_eq(output, "device_residency", "cuda_device_buffer")?;
    require_string_eq(output, "download_scope", "parity_check_only")?;
    require_positive_u64(output, "device_to_host_bytes")?;
    let d2h = object_field(output, "device_to_host_bytes")?
        .as_u64()
        .ok_or_else(|| anyhow!("output.device_to_host_bytes must be an unsigned integer"))?;

    let allocation = object_field(residency, "allocation")?;
    require_u64_eq(allocation, "device_buffer_count", 2)?;
    require_u64_eq(allocation, "persistent_handle_count", 0)?;
    require_bool_eq(allocation, "persistent_handles_claimed", false)?;

    let transfer = object_field(residency, "transfer_accounting")?;
    require_string_eq(transfer, "status", "measured")?;
    require_u64_eq(transfer, "host_to_device_bytes", h2d)?;
    require_u64_eq(transfer, "device_to_host_bytes", d2h)?;
    require_u64_eq(transfer, "kernel_invocations", 1)?;
    require_u64_eq(transfer, "kernel_launches", 1)?;
    require_u64_eq(
        stats,
        "host_to_device_bytes",
        object_field(transfer, "host_to_device_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("transfer.host_to_device_bytes must be an integer"))?,
    )?;
    require_u64_eq(
        stats,
        "device_to_host_bytes",
        object_field(transfer, "device_to_host_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("transfer.device_to_host_bytes must be an integer"))?,
    )?;

    Ok(())
}

/// Validate dense GGUF attention-score CUDA parity evidence.
///
/// This receipt bridges metadata-derived RoPE Q/K score fixtures into a strict
/// CUDA attention-score kernel. It must still reject dense GGUF inference,
/// Qwen token/decode/chat, speedup, persistent-session, full CUDA residency,
/// and BitNet packed I2_S/QK256 proof claims.
pub fn validate_dense_gguf_attention_score_cuda_parity_receipt_json(receipt: &Value) -> Result<()> {
    let stats = validate_cuda_receipt_common(
        receipt,
        DENSE_GGUF_ATTENTION_SCORE_CUDA_PARITY_ARTIFACT_KIND,
        "dense_gguf_attention_score_cuda_parity_tested",
    )?;
    require_string_eq(stats, "kernel_id", "dense_attention_scores_f32_cuda")?;
    require_u64_eq(stats, "invocations", 1)?;
    require_u64_eq(stats, "fallback_invocations", 0)?;
    require_u64_eq(stats, "kernel_launches", 1)?;
    require_bool_eq(receipt, "speedup_claim", false)?;

    let model = object_field(receipt, "model")?;
    require_string_non_empty(model, "model_family")?;
    reject_bitnet_packed_marker(required_string(model, "model_family")?, "model.model_family")?;
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_sha256(model, "sha256")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "dense_f32_attention_scores")?;
    require_string_eq(
        execution_path,
        "quantization_family",
        "metadata_derived_rope_qk_attention_scores_fixture",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    validate_dense_regular_llm_execution_plan(receipt)?;
    let plan = object_field(receipt, "execution_plan")?;
    require_u64_eq(plan, "total_ops", 1)?;
    require_u64_eq(plan, "cuda_ops", 1)?;
    require_u64_eq(plan, "cuda_dense_regular_llm_ops", 1)?;
    require_u64_eq(plan, "cuda_bitnet_qk256_ops", 0)?;
    require_u64_eq(plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(plan, "unsupported_ops", 0)?;
    require_bool_eq(plan, "strict_cuda_ready", true)?;

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_bool_eq(descriptor, "required_roles_present", true)?;
    require_bool_eq(descriptor, "strict_descriptor_complete", true)?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;

    let fixture = object_field(receipt, "attention_score_fixture")?;
    require_u64_eq(fixture, "schema", 1)?;
    require_string_eq(
        fixture,
        "source_artifact_kind",
        DENSE_GGUF_ATTENTION_SCORE_FIXTURE_ARTIFACT_KIND,
    )?;
    require_string_eq(
        fixture,
        "source_rope_artifact_kind",
        DENSE_GGUF_ROPE_CUDA_PARITY_ARTIFACT_KIND,
    )?;
    require_string_non_empty(fixture, "source_rope_fixture_id")?;
    require_string_non_empty(fixture, "fixture_id")?;
    require_string_eq(fixture, "model_family", required_string(model, "model_family")?)?;
    require_string_eq(fixture, "architecture", required_string(model, "architecture")?)?;
    require_positive_u64(fixture, "head_dim")?;
    require_positive_u64(fixture, "q_heads")?;
    require_positive_u64(fixture, "kv_heads")?;
    require_positive_u64(fixture, "heads_per_kv_group")?;
    require_positive_u64(fixture, "seq_len")?;
    require_positive_number(fixture, "rope_base")?;
    require_positive_number(fixture, "scaling_factor")?;
    require_positive_number(fixture, "attention_scale")?;
    require_bool_eq(fixture, "causal_mask_applied", true)?;
    require_string_non_empty(fixture, "head_dim_source")?;
    require_string_non_empty(fixture, "q_heads_source")?;
    require_string_non_empty(fixture, "kv_heads_source")?;
    require_string_non_empty(fixture, "rope_base_source")?;
    require_sha256(fixture, "q_rope_output_sha256")?;
    require_sha256(fixture, "k_rope_output_sha256")?;
    require_sha256(fixture, "cpu_reference_scores_sha256")?;
    require_string_eq(fixture, "cuda_input_dtype", "f32")?;
    require_string_eq(fixture, "cuda_output_dtype", "f32")?;
    require_string_eq(fixture, "cuda_kernel_status", "parity_passed")?;
    require_bool_eq(fixture, "strict_cuda_ready", true)?;
    require_bool_eq(fixture, "cpu_fallback_allowed", false)?;
    require_string_eq(fixture, "transfer_timing_status", "bytes_measured_time_unmeasured")?;
    require_bool_eq(fixture, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(fixture, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(fixture, "cpu_cuda_parity_claimed", true)?;
    require_bool_eq(fixture, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(fixture, "speedup_claim", false)?;
    require_bool_eq(fixture, "full_cuda_residency_claimed", false)?;

    let shape = array_field(fixture, "score_shape")?;
    if shape.len() != 3 {
        return Err(anyhow!(
            "attention_score_fixture.score_shape must contain [q_heads, seq_len, seq_len]"
        ));
    }
    let q_heads = object_field(fixture, "q_heads")?
        .as_u64()
        .ok_or_else(|| anyhow!("attention_score_fixture.q_heads must be an unsigned integer"))?;
    let seq_len = object_field(fixture, "seq_len")?
        .as_u64()
        .ok_or_else(|| anyhow!("attention_score_fixture.seq_len must be an unsigned integer"))?;
    require_u64_eq(
        fixture,
        "score_count",
        q_heads
            .checked_mul(seq_len)
            .and_then(|value| value.checked_mul(seq_len))
            .ok_or_else(|| anyhow!("attention_score_fixture.score_count overflow"))?,
    )?;
    let score_count = object_field(fixture, "score_count")?
        .as_u64()
        .ok_or_else(|| anyhow!("attention_score_fixture.score_count must be an integer"))?;
    let finite_scores = object_field(fixture, "finite_scores")?
        .as_u64()
        .ok_or_else(|| anyhow!("attention_score_fixture.finite_scores must be an integer"))?;
    let masked_scores =
        object_field(fixture, "causal_masked_scores")?.as_u64().ok_or_else(|| {
            anyhow!("attention_score_fixture.causal_masked_scores must be an integer")
        })?;
    if finite_scores == 0 || finite_scores + masked_scores != score_count {
        return Err(anyhow!(
            "attention_score_fixture finite and causal-masked counts must sum to score_count"
        ));
    }

    let parity = object_field(receipt, "parity")?;
    require_string_eq(parity, "reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(parity, "target_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(parity, "kernel_id", "dense_attention_scores_f32_cuda")?;
    require_string_eq(parity, "fixture_id", required_string(fixture, "fixture_id")?)?;
    require_non_negative_number(parity, "max_abs_error")?;
    require_non_negative_number(parity, "mean_abs_error")?;
    require_non_negative_number(parity, "tolerance")?;
    require_bool_eq(parity, "passed", true)?;
    require_string_non_empty(parity, "tolerance_source")?;
    require_u64_eq(parity, "compared_scores", score_count)?;
    require_u64_eq(parity, "finite_scores", finite_scores)?;
    require_u64_eq(parity, "causal_masked_scores", masked_scores)?;
    require_null(parity, "first_divergence")?;
    let max_abs = object_field(parity, "max_abs_error")?
        .as_f64()
        .ok_or_else(|| anyhow!("parity.max_abs_error must be a number"))?;
    let tolerance = object_field(parity, "tolerance")?
        .as_f64()
        .ok_or_else(|| anyhow!("parity.tolerance must be a number"))?;
    if max_abs > tolerance {
        return Err(anyhow!("attention-score CUDA parity max_abs_error exceeds tolerance"));
    }

    let timing = object_field(receipt, "timing")?;
    require_optional_non_negative_number(timing, "kernel_time_ms")?;
    require_u64_eq(
        timing,
        "host_to_device_bytes",
        object_field(stats, "host_to_device_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("kernel_stats.host_to_device_bytes must be an integer"))?,
    )?;
    require_u64_eq(
        timing,
        "device_to_host_bytes",
        object_field(stats, "device_to_host_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("kernel_stats.device_to_host_bytes must be an integer"))?,
    )?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_score_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_score_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_rope_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_norm_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "cpu_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    let residency = object_field(receipt, "tensor_residency")?;
    require_string_eq(residency, "scope", "single_dense_gguf_attention_score_fixture")?;
    require_string_eq(residency, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(residency, "fixture_id", required_string(fixture, "fixture_id")?)?;
    require_bool_eq(residency, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(residency, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(residency, "persistent_session_residency_claimed", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    require_bool_eq(residency, "input_tensors_uploaded_once", true)?;
    require_bool_eq(residency, "output_tensor_cuda_resident_during_kernel", true)?;
    require_bool_eq(residency, "host_device_transfer_accounting_matches_kernel_stats", true)?;

    let inputs = array_field(residency, "inputs")?;
    if inputs.len() != 2 {
        return Err(anyhow!("attention-score tensor_residency.inputs must contain Q and K"));
    }
    let mut h2d = 0_u64;
    for input in inputs {
        require_string_non_empty(input, "name")?;
        require_string_eq(input, "dtype", "f32")?;
        require_string_eq(input, "device_residency", "cuda_device_buffer")?;
        require_u64_eq(input, "upload_count", 1)?;
        require_positive_u64(input, "host_bytes")?;
        h2d += object_field(input, "host_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("tensor_residency.inputs.host_bytes must be an integer"))?;
    }
    let outputs = array_field(residency, "outputs")?;
    if outputs.len() != 1 {
        return Err(anyhow!("attention-score tensor_residency.outputs must contain scores"));
    }
    let output = &outputs[0];
    require_string_eq(output, "name", "dense_gguf_attention_scores")?;
    require_string_eq(output, "dtype", "f32")?;
    require_string_eq(output, "device_residency", "cuda_device_buffer")?;
    require_string_eq(output, "download_scope", "parity_check_only")?;
    require_positive_u64(output, "device_to_host_bytes")?;
    let d2h = object_field(output, "device_to_host_bytes")?.as_u64().ok_or_else(|| {
        anyhow!("tensor_residency.output.device_to_host_bytes must be an integer")
    })?;

    let allocation = object_field(residency, "allocation")?;
    require_u64_eq(allocation, "device_buffer_count", 3)?;
    require_u64_eq(allocation, "persistent_handle_count", 0)?;
    require_bool_eq(allocation, "persistent_handles_claimed", false)?;
    let transfer = object_field(residency, "transfer_accounting")?;
    require_string_eq(transfer, "status", "measured")?;
    require_u64_eq(transfer, "host_to_device_bytes", h2d)?;
    require_u64_eq(transfer, "device_to_host_bytes", d2h)?;
    require_u64_eq(transfer, "kernel_invocations", 1)?;
    require_u64_eq(transfer, "kernel_launches", 1)?;
    require_u64_eq(
        stats,
        "host_to_device_bytes",
        object_field(transfer, "host_to_device_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("transfer.host_to_device_bytes must be an integer"))?,
    )?;
    require_u64_eq(
        stats,
        "device_to_host_bytes",
        object_field(transfer, "device_to_host_bytes")?
            .as_u64()
            .ok_or_else(|| anyhow!("transfer.device_to_host_bytes must be an integer"))?,
    )?;

    Ok(())
}

/// Validate dense GGUF single-linear CUDA parity evidence.
///
/// This is the first bridge from descriptor-extracted dense GGUF linear
/// fixtures into the dense CUDA GEMM lane. It must still reject dense GGUF
/// inference, speedup, full CUDA residency, and BitNet packed I2_S/QK256 proof
/// claims.
pub fn validate_dense_gguf_linear_cuda_parity_receipt_json(receipt: &Value) -> Result<()> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", DENSE_GGUF_LINEAR_CUDA_PARITY_ARTIFACT_KIND)?;
    require_string_eq(receipt, "claim", "dense_gguf_linear_cuda_parity_tested")?;
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
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "dense_fp16_gemm")?;
    require_string_non_empty(execution_path, "quantization_family")?;
    reject_bitnet_packed_marker(
        required_string(execution_path, "quantization_family")?,
        "execution_path.quantization_family",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    validate_dense_regular_llm_execution_plan(receipt)?;

    let fixture = object_field(receipt, "linear_fixture")?;
    require_u64_eq(fixture, "schema", 1)?;
    require_string_eq(fixture, "source_artifact_kind", DENSE_GGUF_LINEAR_FIXTURE_ARTIFACT_KIND)?;
    require_string_non_empty(fixture, "fixture_id")?;
    reject_bitnet_packed_marker(
        required_string(fixture, "fixture_id")?,
        "linear_fixture.fixture_id",
    )?;
    require_string_eq(fixture, "model_family", required_string(model, "model_family")?)?;
    require_string_eq(fixture, "architecture", required_string(model, "architecture")?)?;
    require_string_non_empty(fixture, "tensor_name")?;
    reject_bitnet_packed_marker(
        required_string(fixture, "tensor_name")?,
        "linear_fixture.tensor_name",
    )?;
    require_extractable_dense_linear_role(required_string(fixture, "role")?)?;
    require_string_non_empty(fixture, "tensor_type")?;
    reject_bitnet_packed_marker(
        required_string(fixture, "tensor_type")?,
        "linear_fixture.tensor_type",
    )?;
    require_positive_u64(fixture, "matrix_rows")?;
    require_positive_u64(fixture, "matrix_cols")?;
    require_string_eq(fixture, "logical_layout", "gguf_in_out_reinterpreted_as_out_in")?;
    require_string_eq(fixture, "gemm_layout", "input_1_by_in_times_weight_in_by_out")?;
    require_bool_eq(fixture, "values_materialized_as_f32", true)?;
    require_string_eq(fixture, "gemm_input_dtype", "f16")?;
    require_string_eq(fixture, "gemm_weight_dtype", "f16")?;
    require_string_eq(fixture, "gemm_output_dtype", "f32")?;
    require_sha256(fixture, "weight_values_sha256")?;
    require_bool_eq(fixture, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(fixture, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(fixture, "cpu_cuda_parity_claimed", true)?;
    require_bool_eq(fixture, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(fixture, "speedup_claim", false)?;
    require_bool_eq(fixture, "full_cuda_residency_claimed", false)?;

    let stats = first_kernel_stats(receipt)?;
    require_string_eq(stats, "kernel_id", "dense_f16_gemm_cuda")?;
    require_positive_u64(stats, "invocations")?;
    require_u64_eq(stats, "fallback_invocations", 0)?;
    require_optional_positive_u64(stats, "host_to_device_bytes")?;
    require_optional_positive_u64(stats, "device_to_host_bytes")?;
    require_optional_non_negative_number(stats, "kernel_time_ms")?;

    let parity = object_field(receipt, "parity")?;
    require_string_non_empty(parity, "reference_backend")?;
    require_string_eq(parity, "target_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(parity, "kernel_id", "dense_f16_gemm_cuda")?;
    require_string_eq(parity, "fixture_id", required_string(fixture, "fixture_id")?)?;
    require_bool_eq(parity, "passed", true)?;
    require_non_negative_number(parity, "max_abs_error")?;
    require_non_negative_number(parity, "mean_abs_error")?;
    require_non_negative_number(parity, "tolerance")?;
    require_string_non_empty(parity, "tolerance_source")?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    let residency = object_field(receipt, "tensor_residency")?;
    require_string_eq(residency, "scope", "single_dense_gguf_linear_fixture")?;
    require_string_eq(residency, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(residency, "fixture_id", required_string(fixture, "fixture_id")?)?;
    require_bool_eq(residency, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(residency, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(residency, "persistent_session_residency_claimed", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    require_bool_eq(residency, "input_tensors_uploaded_once", true)?;
    require_bool_eq(residency, "output_tensor_cuda_resident_during_kernel", true)?;
    require_bool_eq(residency, "host_device_transfer_accounting_matches_kernel_stats", true)?;

    let inputs = array_field(residency, "inputs")?;
    if inputs.len() < 2 {
        return Err(anyhow!("tensor_residency.inputs must contain input and weight tensors"));
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

/// Validate dense GGUF multi-linear role-sweep CUDA parity evidence.
///
/// This is an aggregate planner/receipt bridge over several extracted dense
/// GGUF linear fixtures. It proves dense CUDA route accounting across multiple
/// roles, while still rejecting dense GGUF inference, speedup, full CUDA
/// residency, and BitNet packed I2_S/QK256 proof claims.
pub fn validate_dense_gguf_linear_role_sweep_cuda_parity_receipt_json(
    receipt: &Value,
) -> Result<()> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(
        receipt,
        "artifact_kind",
        DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND,
    )?;
    require_string_eq(receipt, "claim", "dense_gguf_linear_role_sweep_cuda_parity_tested")?;
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
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "dense_fp16_gemm")?;
    require_string_non_empty(execution_path, "quantization_family")?;
    reject_bitnet_packed_marker(
        required_string(execution_path, "quantization_family")?,
        "execution_path.quantization_family",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    validate_dense_regular_llm_execution_plan(receipt)?;

    let sweep = object_field(receipt, "linear_role_sweep")?;
    require_u64_eq(sweep, "schema", 1)?;
    let roles_total = object_field(sweep, "roles_total")?
        .as_u64()
        .ok_or_else(|| anyhow!("linear_role_sweep.roles_total must be an unsigned integer"))?;
    if roles_total < 2 {
        return Err(anyhow!(
            "linear_role_sweep.roles_total must cover at least two dense linear roles"
        ));
    }
    require_u64_eq(sweep, "roles_passed", roles_total)?;
    require_u64_eq(sweep, "roles_failed", 0)?;
    require_bool_eq(sweep, "all_parity_passed", true)?;
    require_non_negative_number(sweep, "max_abs_error")?;
    require_non_negative_number(sweep, "max_mean_abs_error")?;
    require_bool_eq(sweep, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(sweep, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(sweep, "speedup_claim", false)?;
    require_bool_eq(sweep, "full_cuda_residency_claimed", false)?;

    let covered_roles = array_field(sweep, "covered_roles")?;
    if covered_roles.len() != roles_total as usize {
        return Err(anyhow!("linear_role_sweep.covered_roles length must match roles_total"));
    }
    let mut role_set = BTreeSet::new();
    for role in covered_roles {
        let role = role
            .as_str()
            .ok_or_else(|| anyhow!("linear_role_sweep.covered_roles entries must be strings"))?;
        require_extractable_dense_linear_role(role)?;
        reject_bitnet_packed_marker(role, "linear_role_sweep.covered_roles")?;
        if !role_set.insert(role.to_string()) {
            return Err(anyhow!("linear_role_sweep.covered_roles contains duplicate `{role}`"));
        }
    }

    let plan = object_field(receipt, "execution_plan")?;
    require_u64_eq(plan, "total_ops", roles_total)?;
    require_u64_eq(plan, "cuda_ops", roles_total)?;
    require_u64_eq(plan, "cuda_dense_regular_llm_ops", roles_total)?;
    require_u64_eq(plan, "cuda_bitnet_qk256_ops", 0)?;
    require_u64_eq(plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(plan, "unsupported_ops", 0)?;

    let fixtures = array_field(receipt, "linear_fixtures")?;
    if fixtures.len() != roles_total as usize {
        return Err(anyhow!("linear_fixtures length must match roles_total"));
    }
    let stats = array_field(receipt, "kernel_stats")?;
    if stats.len() != roles_total as usize {
        return Err(anyhow!("kernel_stats length must match roles_total"));
    }

    let mut stats_h2d = 0_u64;
    let mut stats_d2h = 0_u64;
    let mut stats_invocations = 0_u64;
    let mut stats_launches = 0_u64;
    for (idx, fixture) in fixtures.iter().enumerate() {
        require_u64_eq(fixture, "schema", 1)?;
        require_string_eq(
            fixture,
            "source_artifact_kind",
            DENSE_GGUF_LINEAR_FIXTURE_ARTIFACT_KIND,
        )?;
        require_string_non_empty(fixture, "fixture_id")?;
        reject_bitnet_packed_marker(
            required_string(fixture, "fixture_id")?,
            "linear_fixtures.fixture_id",
        )?;
        require_string_eq(fixture, "model_family", required_string(model, "model_family")?)?;
        require_string_eq(fixture, "architecture", required_string(model, "architecture")?)?;
        require_string_non_empty(fixture, "tensor_name")?;
        reject_bitnet_packed_marker(
            required_string(fixture, "tensor_name")?,
            "linear_fixtures.tensor_name",
        )?;
        let role = required_string(fixture, "role")?;
        require_extractable_dense_linear_role(role)?;
        if !role_set.contains(role) {
            return Err(anyhow!("linear_fixtures role `{role}` is not listed in covered_roles"));
        }
        require_string_non_empty(fixture, "tensor_type")?;
        reject_bitnet_packed_marker(
            required_string(fixture, "tensor_type")?,
            "linear_fixtures.tensor_type",
        )?;
        require_positive_u64(fixture, "matrix_rows")?;
        require_positive_u64(fixture, "matrix_cols")?;
        require_string_eq(fixture, "logical_layout", "gguf_in_out_reinterpreted_as_out_in")?;
        require_string_eq(fixture, "gemm_layout", "input_1_by_in_times_weight_in_by_out")?;
        require_bool_eq(fixture, "values_materialized_as_f32", true)?;
        require_string_eq(fixture, "gemm_input_dtype", "f16")?;
        require_string_eq(fixture, "gemm_weight_dtype", "f16")?;
        require_string_eq(fixture, "gemm_output_dtype", "f32")?;
        require_sha256(fixture, "weight_values_sha256")?;
        require_bool_eq(fixture, "dense_gguf_inference_claimed", false)?;
        require_bool_eq(fixture, "dense_regular_llm_cuda_claimed", true)?;
        require_bool_eq(fixture, "cpu_cuda_parity_claimed", true)?;
        require_bool_eq(fixture, "bitnet_packed_i2s_qk256_proof", false)?;
        require_bool_eq(fixture, "speedup_claim", false)?;
        require_bool_eq(fixture, "full_cuda_residency_claimed", false)?;

        let stat = &stats[idx];
        require_string_eq(stat, "role", role)?;
        require_string_eq(stat, "tensor_name", required_string(fixture, "tensor_name")?)?;
        require_string_eq(stat, "fixture_id", required_string(fixture, "fixture_id")?)?;
        require_string_eq(stat, "kernel_id", "dense_f16_gemm_cuda")?;
        require_positive_u64(stat, "invocations")?;
        require_u64_eq(stat, "fallback_invocations", 0)?;
        require_positive_u64(stat, "host_to_device_bytes")?;
        require_positive_u64(stat, "device_to_host_bytes")?;
        require_positive_u64(stat, "kernel_launches")?;
        require_optional_non_negative_number(stat, "kernel_time_ms")?;

        stats_h2d += object_field(stat, "host_to_device_bytes")?.as_u64().ok_or_else(|| {
            anyhow!("kernel_stats.host_to_device_bytes must be an unsigned integer")
        })?;
        stats_d2h += object_field(stat, "device_to_host_bytes")?.as_u64().ok_or_else(|| {
            anyhow!("kernel_stats.device_to_host_bytes must be an unsigned integer")
        })?;
        stats_invocations += object_field(stat, "invocations")?
            .as_u64()
            .ok_or_else(|| anyhow!("kernel_stats.invocations must be an unsigned integer"))?;
        stats_launches += object_field(stat, "kernel_launches")?
            .as_u64()
            .ok_or_else(|| anyhow!("kernel_stats.kernel_launches must be an unsigned integer"))?;
    }

    let parity = object_field(receipt, "parity")?;
    require_string_non_empty(parity, "reference_backend")?;
    require_string_eq(parity, "target_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(parity, "kernel_id", "dense_f16_gemm_cuda")?;
    require_bool_eq(parity, "passed", true)?;
    require_u64_eq(parity, "roles_total", roles_total)?;
    require_u64_eq(parity, "roles_passed", roles_total)?;
    require_u64_eq(parity, "roles_failed", 0)?;
    require_non_negative_number(parity, "max_abs_error")?;
    require_non_negative_number(parity, "max_mean_abs_error")?;
    require_non_negative_number(parity, "tolerance")?;
    require_string_non_empty(parity, "tolerance_source")?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    let residency = object_field(receipt, "tensor_residency")?;
    require_string_eq(residency, "scope", "dense_gguf_linear_role_sweep_fixture")?;
    require_string_eq(residency, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_u64_eq(residency, "roles_total", roles_total)?;
    require_bool_eq(residency, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(residency, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(residency, "persistent_session_residency_claimed", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    require_bool_eq(residency, "input_tensors_uploaded_once_per_role", true)?;
    require_bool_eq(residency, "output_tensor_cuda_resident_during_kernel", true)?;
    require_bool_eq(residency, "host_device_transfer_accounting_matches_kernel_stats", true)?;

    let allocation = object_field(residency, "allocation")?;
    require_u64_eq(allocation, "device_buffer_count", roles_total * 3)?;
    require_u64_eq(allocation, "persistent_handle_count", 0)?;
    require_bool_eq(allocation, "persistent_handles_claimed", false)?;

    let transfer = object_field(residency, "transfer_accounting")?;
    require_string_eq(transfer, "status", "measured")?;
    require_u64_eq(transfer, "host_to_device_bytes", stats_h2d)?;
    require_u64_eq(transfer, "device_to_host_bytes", stats_d2h)?;
    require_u64_eq(transfer, "kernel_invocations", stats_invocations)?;
    require_u64_eq(transfer, "kernel_launches", stats_launches)?;

    Ok(())
}

/// Validate dense GGUF one-layer execution-plan gap evidence.
///
/// This artifact records the dense GGUF one-layer planner route. It is a
/// fail-closed planner receipt, not dense GGUF inference.
pub fn validate_dense_gguf_one_layer_execution_plan_receipt_json(receipt: &Value) -> Result<()> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", DENSE_GGUF_ONE_LAYER_EXECUTION_PLAN_ARTIFACT_KIND)?;
    require_string_eq(receipt, "claim", "dense_gguf_one_layer_execution_plan_gap_recorded")?;
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
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_sha256(model, "sha256")?;

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

    validate_dense_one_layer_gap_execution_plan(receipt)?;

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_positive_u64(descriptor, "tensor_count")?;
    require_positive_u64(descriptor, "metadata_count")?;
    require_bool_eq(descriptor, "required_roles_present", true)?;
    require_bool_eq(descriptor, "strict_descriptor_complete", true)?;
    require_string_non_empty(descriptor, "dense_cuda_route_status")?;
    reject_bitnet_packed_marker(
        required_string(descriptor, "dense_cuda_route_status")?,
        "descriptor_coverage.dense_cuda_route_status",
    )?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;
    let quantization_families = array_field(descriptor, "quantization_families")?;
    if quantization_families.is_empty() {
        return Err(anyhow!("descriptor_coverage.quantization_families must not be empty"));
    }
    for family in quantization_families {
        let family = family.as_str().ok_or_else(|| {
            anyhow!("descriptor_coverage.quantization_families entries must be strings")
        })?;
        reject_bitnet_packed_marker(family, "descriptor_coverage.quantization_families")?;
    }

    let one_layer = object_field(receipt, "one_layer_plan")?;
    require_u64_eq(one_layer, "schema", 1)?;
    let total_ops = object_field(one_layer, "total_ops")?
        .as_u64()
        .ok_or_else(|| anyhow!("one_layer_plan.total_ops must be an unsigned integer"))?;
    let cuda_routable_ops =
        object_field(one_layer, "cuda_routable_ops_total")?.as_u64().ok_or_else(|| {
            anyhow!("one_layer_plan.cuda_routable_ops_total must be an unsigned integer")
        })?;
    let linear_cuda_ops =
        object_field(one_layer, "linear_cuda_ops_total")?.as_u64().ok_or_else(|| {
            anyhow!("one_layer_plan.linear_cuda_ops_total must be an unsigned integer")
        })?;
    let norm_cuda_ops = object_field(one_layer, "norm_cuda_ops_total")?
        .as_u64()
        .ok_or_else(|| anyhow!("one_layer_plan.norm_cuda_ops_total must be an unsigned integer"))?;
    let rope_cuda_ops = object_field(one_layer, "rope_cuda_ops_total")?
        .as_u64()
        .ok_or_else(|| anyhow!("one_layer_plan.rope_cuda_ops_total must be an unsigned integer"))?;
    let attention_score_cuda_ops =
        object_field(one_layer, "attention_score_cuda_ops_total")?.as_u64().ok_or_else(|| {
            anyhow!("one_layer_plan.attention_score_cuda_ops_total must be an unsigned integer")
        })?;
    let attention_softmax_cuda_ops =
        object_field(one_layer, "attention_softmax_cuda_ops_total")?.as_u64().ok_or_else(|| {
            anyhow!("one_layer_plan.attention_softmax_cuda_ops_total must be an unsigned integer")
        })?;
    let attention_v_mix_cuda_ops =
        object_field(one_layer, "attention_v_mix_cuda_ops_total")?.as_u64().ok_or_else(|| {
            anyhow!("one_layer_plan.attention_v_mix_cuda_ops_total must be an unsigned integer")
        })?;
    let mlp_activation_cuda_ops =
        object_field(one_layer, "mlp_activation_cuda_ops_total")?.as_u64().ok_or_else(|| {
            anyhow!("one_layer_plan.mlp_activation_cuda_ops_total must be an unsigned integer")
        })?;
    let unsupported_ops = object_field(one_layer, "unsupported_strict_cuda_ops_total")?
        .as_u64()
        .ok_or_else(|| {
            anyhow!("one_layer_plan.unsupported_strict_cuda_ops_total must be an unsigned integer")
        })?;
    if cuda_routable_ops == 0
        || linear_cuda_ops == 0
        || norm_cuda_ops == 0
        || rope_cuda_ops == 0
        || attention_score_cuda_ops == 0
        || attention_softmax_cuda_ops == 0
        || attention_v_mix_cuda_ops == 0
        || mlp_activation_cuda_ops == 0
        || cuda_routable_ops
            != linear_cuda_ops
                + norm_cuda_ops
                + rope_cuda_ops
                + attention_score_cuda_ops
                + attention_softmax_cuda_ops
                + attention_v_mix_cuda_ops
                + mlp_activation_cuda_ops
        || unsupported_ops != 0
        || total_ops != cuda_routable_ops + unsupported_ops
    {
        return Err(anyhow!(
            "one_layer_plan must route dense CUDA linears, RMSNorm, RoPE, attention scores, attention softmax, attention V-mix, and MLP activation with no unsupported strict CUDA ops"
        ));
    }
    require_u64_eq(one_layer, "cpu_fallback_ops_total", 0)?;
    require_bool_eq(one_layer, "strict_cuda_ready", true)?;
    require_bool_eq(one_layer, "unsupported_ops_explicitly_listed", true)?;
    require_bool_eq(one_layer, "dense_gguf_one_layer_execution_plan_claimed", true)?;
    require_bool_eq(one_layer, "one_layer_inference_claimed", false)?;
    require_bool_eq(one_layer, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(one_layer, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(one_layer, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(one_layer, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(one_layer, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(one_layer, "speedup_claim", false)?;
    require_bool_eq(one_layer, "full_cuda_residency_claimed", false)?;

    let operations = array_field(one_layer, "operations")?;
    if operations.len() != total_ops as usize {
        return Err(anyhow!("one_layer_plan.operations length must match total_ops"));
    }
    let mut seen_cuda_ops = 0_u64;
    let mut seen_linear_cuda_ops = 0_u64;
    let mut seen_norm_cuda_ops = 0_u64;
    let mut seen_rope_cuda_ops = 0_u64;
    let mut seen_attention_score_cuda_ops = 0_u64;
    let mut seen_attention_softmax_cuda_ops = 0_u64;
    let mut seen_attention_v_mix_cuda_ops = 0_u64;
    let mut seen_mlp_activation_cuda_ops = 0_u64;
    let mut seen_unsupported_ops = 0_u64;
    let mut seen_unsupported_roles = BTreeSet::new();
    for (idx, op) in operations.iter().enumerate() {
        require_u64_eq(op, "index", idx as u64)?;
        require_string_non_empty(op, "name")?;
        reject_bitnet_packed_marker(
            required_string(op, "name")?,
            "one_layer_plan.operations.name",
        )?;
        require_string_non_empty(op, "role")?;
        reject_bitnet_packed_marker(
            required_string(op, "role")?,
            "one_layer_plan.operations.role",
        )?;
        require_string_non_empty(op, "op_type")?;
        require_positive_u64(op, "size")?;
        require_string_non_empty(op, "source")?;
        require_bool_eq(op, "fallback_used", false)?;
        require_string_non_empty(op, "reason")?;

        match required_string(op, "route")? {
            DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND => {
                require_string_eq(op, "status", "cuda_routable")?;
                let op_type = required_string(op, "op_type")?;
                if !matches!(
                    op_type,
                    "matmul" | "rmsnorm" | "rope" | "attention" | "softmax" | "activation"
                ) {
                    return Err(anyhow!(
                        "CUDA-routable dense op_type must be matmul, rmsnorm, rope, governed attention, governed softmax, or governed activation, got `{op_type}`"
                    ));
                }
                require_bool_eq(op, "is_quantized", false)?;
                match op_type {
                    "matmul" => {
                        let tensor = required_string(op, "source_tensor")?;
                        reject_bitnet_packed_marker(
                            tensor,
                            "one_layer_plan.operations.source_tensor",
                        )?;
                        let tensor_type = required_string(op, "source_tensor_type")?;
                        reject_bitnet_packed_marker(
                            tensor_type,
                            "one_layer_plan.operations.source_tensor_type",
                        )?;
                        if tensor_type == "f32" {
                            return Err(anyhow!(
                                "CUDA-routable dense matmul op must not use f32 norm tensor type"
                            ));
                        }
                        seen_linear_cuda_ops += 1;
                    }
                    "rmsnorm" => {
                        let tensor = required_string(op, "source_tensor")?;
                        reject_bitnet_packed_marker(
                            tensor,
                            "one_layer_plan.operations.source_tensor",
                        )?;
                        let tensor_type = required_string(op, "source_tensor_type")?;
                        reject_bitnet_packed_marker(
                            tensor_type,
                            "one_layer_plan.operations.source_tensor_type",
                        )?;
                        if tensor_type != "f32" {
                            return Err(anyhow!(
                                "CUDA-routable dense rmsnorm op must use f32 source_tensor_type"
                            ));
                        }
                        seen_norm_cuda_ops += 1;
                    }
                    "rope" => {
                        require_string_eq(op, "source", "derived_transformer_op")?;
                        require_null(op, "source_tensor")?;
                        require_null(op, "source_tensor_type")?;
                        require_null(op, "source_shape")?;
                        seen_rope_cuda_ops += 1;
                    }
                    "attention" => {
                        let role = required_string(op, "role")?;
                        require_string_eq(op, "source", "derived_transformer_op")?;
                        require_null(op, "source_tensor")?;
                        require_null(op, "source_tensor_type")?;
                        require_null(op, "source_shape")?;
                        match role {
                            "attention_scores" => seen_attention_score_cuda_ops += 1,
                            "attention_v_mix" => seen_attention_v_mix_cuda_ops += 1,
                            other => {
                                return Err(anyhow!(
                                    "CUDA-routable dense attention op role must be attention_scores or attention_v_mix, got `{other}`"
                                ));
                            }
                        }
                    }
                    "softmax" => {
                        require_string_eq(op, "role", "attention_softmax")?;
                        require_string_eq(op, "source", "derived_transformer_op")?;
                        require_null(op, "source_tensor")?;
                        require_null(op, "source_tensor_type")?;
                        require_null(op, "source_shape")?;
                        seen_attention_softmax_cuda_ops += 1;
                    }
                    "activation" => {
                        require_string_eq(op, "role", "mlp_activation")?;
                        require_string_eq(op, "source", "derived_transformer_op")?;
                        require_null(op, "source_tensor")?;
                        require_null(op, "source_tensor_type")?;
                        require_null(op, "source_shape")?;
                        seen_mlp_activation_cuda_ops += 1;
                    }
                    _ => unreachable!("op_type checked above"),
                }
                seen_cuda_ops += 1;
            }
            "unsupported" => {
                require_string_eq(op, "status", "unsupported_strict_cuda")?;
                if required_string(op, "op_type")? == "matmul" {
                    return Err(anyhow!(
                        "dense one-layer plan must not mark dense matmul ops unsupported"
                    ));
                }
                seen_unsupported_roles.insert(required_string(op, "role")?.to_string());
                seen_unsupported_ops += 1;
            }
            other => {
                return Err(anyhow!(
                    "one_layer_plan.operations route must be dense_regular_llm_cuda or unsupported, got `{other}`"
                ));
            }
        }
    }
    if seen_cuda_ops != cuda_routable_ops
        || seen_linear_cuda_ops != linear_cuda_ops
        || seen_norm_cuda_ops != norm_cuda_ops
        || seen_rope_cuda_ops != rope_cuda_ops
        || seen_attention_score_cuda_ops != attention_score_cuda_ops
        || seen_attention_softmax_cuda_ops != attention_softmax_cuda_ops
        || seen_attention_v_mix_cuda_ops != attention_v_mix_cuda_ops
        || seen_mlp_activation_cuda_ops != mlp_activation_cuda_ops
        || seen_unsupported_ops != unsupported_ops
    {
        return Err(anyhow!("one_layer_plan operation route counts do not match summary"));
    }
    let counts = DenseOneLayerGapCounts {
        cuda_routable_ops,
        linear_cuda_ops,
        norm_cuda_ops,
        rope_cuda_ops,
        attention_score_cuda_ops,
        attention_softmax_cuda_ops,
        attention_v_mix_cuda_ops,
        mlp_activation_cuda_ops,
        unsupported_ops,
    };
    validate_dense_one_layer_gap_audit(receipt, &counts, &seen_unsupported_roles)?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

/// Validate dense GGUF all-layer execution-plan evidence.
///
/// This artifact proves that every inspected transformer block has a governed
/// dense CUDA block plan. It keeps model-boundary, inference, speedup, full
/// residency, server, and BitNet packed-kernel claims false.
pub fn validate_dense_gguf_all_layer_execution_plan_receipt_json(receipt: &Value) -> Result<()> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", DENSE_GGUF_ALL_LAYER_EXECUTION_PLAN_ARTIFACT_KIND)?;
    require_string_eq(receipt, "claim", "dense_gguf_all_layer_execution_plan_recorded")?;
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
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_sha256(model, "sha256")?;

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

    validate_dense_one_layer_gap_execution_plan(receipt)?;

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_positive_u64(descriptor, "tensor_count")?;
    require_positive_u64(descriptor, "metadata_count")?;
    let required_roles_present = object_field(descriptor, "required_roles_present")?
        .as_bool()
        .ok_or_else(|| anyhow!("field `required_roles_present` must be a bool"))?;
    let strict_descriptor_complete = object_field(descriptor, "strict_descriptor_complete")?
        .as_bool()
        .ok_or_else(|| anyhow!("field `strict_descriptor_complete` must be a bool"))?;
    let transformer_block_required_roles_present =
        match descriptor.get("transformer_block_required_roles_present") {
            Some(value) => value.as_bool().ok_or_else(|| {
                anyhow!("field `transformer_block_required_roles_present` must be a bool")
            })?,
            None => required_roles_present,
        };
    if !transformer_block_required_roles_present {
        return Err(anyhow!(
            "descriptor_coverage.transformer_block_required_roles_present must be true"
        ));
    }
    if let Some(missing) = descriptor.get("missing_transformer_block_roles") {
        let missing = missing
            .as_array()
            .ok_or_else(|| anyhow!("field `missing_transformer_block_roles` must be an array"))?;
        if !missing.is_empty() {
            return Err(anyhow!(
                "descriptor_coverage.missing_transformer_block_roles must be empty"
            ));
        }
    }
    if !(required_roles_present && strict_descriptor_complete)
        && descriptor.get("missing_model_boundary_roles").is_none()
    {
        return Err(anyhow!(
            "incomplete all-layer descriptor coverage must list missing_model_boundary_roles"
        ));
    }
    require_string_non_empty(descriptor, "dense_cuda_route_status")?;
    reject_bitnet_packed_marker(
        required_string(descriptor, "dense_cuda_route_status")?,
        "descriptor_coverage.dense_cuda_route_status",
    )?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;

    let all_layer = object_field(receipt, "all_layer_plan")?;
    require_u64_eq(all_layer, "schema", 1)?;
    let layer_total =
        object_field(all_layer, "transformer_layers_total")?.as_u64().ok_or_else(|| {
            anyhow!("all_layer_plan.transformer_layers_total must be an unsigned integer")
        })?;
    if layer_total == 0 {
        return Err(anyhow!("all_layer_plan.transformer_layers_total must be positive"));
    }
    require_u64_eq(all_layer, "layers_with_complete_cuda_block_plan", layer_total)?;
    require_bool_eq(all_layer, "layer_plan_matches_layer0", true)?;
    if !array_field(all_layer, "layer_differences")?.is_empty() {
        return Err(anyhow!(
            "all_layer_plan.layer_differences must be empty for strict CUDA ready receipts"
        ));
    }
    if !array_field(all_layer, "missing_layer_indices")?.is_empty() {
        return Err(anyhow!("all_layer_plan.missing_layer_indices must be empty"));
    }

    let total_ops = object_field(all_layer, "total_ops")?
        .as_u64()
        .ok_or_else(|| anyhow!("all_layer_plan.total_ops must be an unsigned integer"))?;
    let cuda_ops =
        object_field(all_layer, "cuda_routable_ops_total")?.as_u64().ok_or_else(|| {
            anyhow!("all_layer_plan.cuda_routable_ops_total must be an unsigned integer")
        })?;
    let linear_ops =
        object_field(all_layer, "linear_cuda_ops_total")?.as_u64().ok_or_else(|| {
            anyhow!("all_layer_plan.linear_cuda_ops_total must be an unsigned integer")
        })?;
    let norm_ops = object_field(all_layer, "norm_cuda_ops_total")?
        .as_u64()
        .ok_or_else(|| anyhow!("all_layer_plan.norm_cuda_ops_total must be an unsigned integer"))?;
    let rope_ops = object_field(all_layer, "rope_cuda_ops_total")?
        .as_u64()
        .ok_or_else(|| anyhow!("all_layer_plan.rope_cuda_ops_total must be an unsigned integer"))?;
    let attention_score_ops =
        object_field(all_layer, "attention_score_cuda_ops_total")?.as_u64().ok_or_else(|| {
            anyhow!("all_layer_plan.attention_score_cuda_ops_total must be an unsigned integer")
        })?;
    let attention_softmax_ops =
        object_field(all_layer, "attention_softmax_cuda_ops_total")?.as_u64().ok_or_else(|| {
            anyhow!("all_layer_plan.attention_softmax_cuda_ops_total must be an unsigned integer")
        })?;
    let attention_v_mix_ops =
        object_field(all_layer, "attention_v_mix_cuda_ops_total")?.as_u64().ok_or_else(|| {
            anyhow!("all_layer_plan.attention_v_mix_cuda_ops_total must be an unsigned integer")
        })?;
    let mlp_activation_ops =
        object_field(all_layer, "mlp_activation_cuda_ops_total")?.as_u64().ok_or_else(|| {
            anyhow!("all_layer_plan.mlp_activation_cuda_ops_total must be an unsigned integer")
        })?;
    require_u64_eq(all_layer, "unsupported_strict_cuda_ops_total", 0)?;
    require_u64_eq(all_layer, "cpu_fallback_ops_total", 0)?;
    require_bool_eq(all_layer, "strict_cuda_ready", true)?;
    require_string_eq(all_layer, "strict_cuda_ready_scope", "transformer_blocks_only")?;
    require_bool_eq(all_layer, "all_layers_inspected", true)?;
    require_u64_eq(all_layer, "operations_per_layer", 14)?;
    if total_ops != layer_total * 14
        || cuda_ops != total_ops
        || linear_ops != layer_total * 7
        || norm_ops != layer_total * 2
        || rope_ops != layer_total
        || attention_score_ops != layer_total
        || attention_softmax_ops != layer_total
        || attention_v_mix_ops != layer_total
        || mlp_activation_ops != layer_total
        || cuda_ops
            != linear_ops
                + norm_ops
                + rope_ops
                + attention_score_ops
                + attention_softmax_ops
                + attention_v_mix_ops
                + mlp_activation_ops
    {
        return Err(anyhow!(
            "all_layer_plan counts must equal 14 governed dense CUDA block ops per transformer layer"
        ));
    }
    require_bool_eq(all_layer, "dense_gguf_all_layer_execution_plan_claimed", true)?;
    require_bool_eq(all_layer, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(all_layer, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(all_layer, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(all_layer, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(all_layer, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(all_layer, "speedup_claim", false)?;
    require_bool_eq(all_layer, "persistent_session_residency_claimed", false)?;
    require_bool_eq(all_layer, "full_cuda_residency_claimed", false)?;

    let layers = array_field(all_layer, "layers")?;
    if layers.len() != layer_total as usize {
        return Err(anyhow!("all_layer_plan.layers length must match transformer_layers_total"));
    }
    let mut layer0_operation_signature_sha256: Option<String> = None;
    for (expected_index, layer) in layers.iter().enumerate() {
        require_u64_eq(layer, "layer_index", expected_index as u64)?;
        require_u64_eq(layer, "total_ops", 14)?;
        require_u64_eq(layer, "cuda_routable_ops_total", 14)?;
        require_u64_eq(layer, "linear_cuda_ops_total", 7)?;
        require_u64_eq(layer, "norm_cuda_ops_total", 2)?;
        require_u64_eq(layer, "rope_cuda_ops_total", 1)?;
        require_u64_eq(layer, "attention_score_cuda_ops_total", 1)?;
        require_u64_eq(layer, "attention_softmax_cuda_ops_total", 1)?;
        require_u64_eq(layer, "attention_v_mix_cuda_ops_total", 1)?;
        require_u64_eq(layer, "mlp_activation_cuda_ops_total", 1)?;
        require_u64_eq(layer, "unsupported_strict_cuda_ops_total", 0)?;
        require_u64_eq(layer, "cpu_fallback_ops_total", 0)?;
        require_bool_eq(layer, "strict_cuda_ready", true)?;
        require_bool_eq(layer, "matches_layer0", true)?;
        require_sha256(layer, "operation_signature_sha256")?;
        let operations = array_field(layer, "operations")?;
        if operations.len() != 14 {
            return Err(anyhow!("all_layer_plan.layers.operations must contain 14 governed ops"));
        }
        let computed_signature =
            dense_all_layer_operation_signature_sha256(operations).map_err(|err| {
                anyhow!(
                    "all_layer_plan.layers[{expected_index}].operation_signature_sha256 could not be recomputed: {err}"
                )
            })?;
        if required_string(layer, "operation_signature_sha256")? != computed_signature {
            return Err(anyhow!(
                "all_layer_plan.layers[{expected_index}].operation_signature_sha256 must match operations"
            ));
        }
        match &layer0_operation_signature_sha256 {
            Some(layer0_signature) if layer0_signature != &computed_signature => {
                return Err(anyhow!(
                    "all_layer_plan.layers[{expected_index}].operation_signature_sha256 must match layer 0"
                ));
            }
            Some(_) => {}
            None => layer0_operation_signature_sha256 = Some(computed_signature),
        }
        for (op_index, op) in operations.iter().enumerate() {
            require_u64_eq(op, "index", op_index as u64)?;
            require_string_non_empty(op, "name")?;
            reject_bitnet_packed_marker(
                required_string(op, "name")?,
                "all_layer_plan.layers.operations.name",
            )?;
            require_string_non_empty(op, "role")?;
            let (expected_role, expected_op_type) = DENSE_ALL_LAYER_OPERATION_SEQUENCE[op_index];
            require_string_eq(op, "role", expected_role)?;
            reject_bitnet_packed_marker(
                required_string(op, "role")?,
                "all_layer_plan.layers.operations.role",
            )?;
            require_string_non_empty(op, "op_type")?;
            require_string_eq(op, "op_type", expected_op_type)?;
            require_positive_u64(op, "size")?;
            require_string_eq(op, "route", DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND)?;
            require_string_eq(op, "status", "cuda_routable")?;
            require_bool_eq(op, "fallback_used", false)?;
            require_string_non_empty(op, "reason")?;
        }
    }

    let gaps = object_field(receipt, "model_boundary_gaps")?;
    require_u64_eq(gaps, "schema", 1)?;
    require_bool_eq(gaps, "all_boundary_gaps_explicit", true)?;
    require_bool_eq(gaps, "qwen_one_token_cuda_blocked", true)?;
    require_bool_eq(gaps, "qwen_short_decode_cuda_blocked", true)?;
    require_bool_eq(gaps, "qwen_chat_cuda_blocked", true)?;
    require_string_eq(gaps, "next_required_proof", "dense_gguf_model_boundary_fixtures")?;
    require_bool_eq(gaps, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(gaps, "speedup_claim", false)?;
    require_bool_eq(gaps, "full_cuda_residency_claimed", false)?;
    let gap_entries = array_field(gaps, "gaps")?;
    let mut required_gaps = BTreeSet::from([
        "token_embedding",
        "final_norm",
        "lm_head_logits",
        "kv_cache_policy",
        "sampling",
    ]);
    for gap in gap_entries {
        let name = required_string(gap, "gap")?;
        required_gaps.remove(name);
        require_string_eq(gap, "status", "not_governed_by_all_layer_block_plan")?;
        require_string_non_empty(gap, "disposition")?;
        require_bool_eq(gap, "blocks_qwen_one_token", true)?;
        require_bool_eq(gap, "blocks_qwen_short_decode", true)?;
        require_bool_eq(gap, "blocks_qwen_chat", true)?;
        require_string_non_empty(gap, "required_next_proof")?;
    }
    if !required_gaps.is_empty() {
        return Err(anyhow!("model_boundary_gaps missing required gaps: {required_gaps:?}"));
    }

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_cpu_reference_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_cuda_integrated_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_all_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

/// Validate dense GGUF model-boundary fixture evidence.
///
/// This artifact records token embedding lookup, final norm, and LM-head/logit
/// diagnostics under the dense CUDA route boundary. It keeps KV cache,
/// sampling, one-token/decode/chat, speedup, full residency, and BitNet packed
/// proof claims false.
pub fn validate_dense_gguf_model_boundary_fixtures_receipt_json(receipt: &Value) -> Result<()> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", DENSE_GGUF_MODEL_BOUNDARY_FIXTURES_ARTIFACT_KIND)?;
    require_string_eq(receipt, "claim", "dense_gguf_model_boundary_fixtures_recorded")?;
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
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_sha256(model, "sha256")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "dense_cuda_model_boundary_fixture_route")?;
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

    validate_dense_one_layer_gap_execution_plan(receipt)?;
    let plan = object_field(receipt, "execution_plan")?;
    require_u64_eq(plan, "total_ops", 3)?;
    require_u64_eq(plan, "cuda_ops", 3)?;
    require_u64_eq(plan, "cuda_dense_regular_llm_ops", 3)?;

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_positive_u64(descriptor, "tensor_count")?;
    require_positive_u64(descriptor, "metadata_count")?;
    require_bool_eq(descriptor, "required_roles_present", true)?;
    require_bool_eq(descriptor, "strict_descriptor_complete", true)?;
    require_string_non_empty(descriptor, "dense_cuda_route_status")?;
    reject_bitnet_packed_marker(
        required_string(descriptor, "dense_cuda_route_status")?,
        "descriptor_coverage.dense_cuda_route_status",
    )?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;

    let fixtures = object_field(receipt, "model_boundary_fixtures")?;
    require_u64_eq(fixtures, "schema", 1)?;
    reject_bitnet_packed_marker(
        required_string(fixtures, "fixture_id")?,
        "model_boundary_fixtures.fixture_id",
    )?;
    let seq_len = required_u64(fixtures, "seq_len")?;
    let hidden_size = required_u64(fixtures, "hidden_size")?;
    let vocab_size = required_u64(fixtures, "vocab_size")?;
    if seq_len == 0 || hidden_size == 0 || vocab_size == 0 {
        return Err(anyhow!("model_boundary_fixtures dimensions must be positive"));
    }
    let token_ids = array_field(fixtures, "token_ids")?;
    if token_ids.len() != seq_len as usize {
        return Err(anyhow!("model_boundary_fixtures.token_ids length must match seq_len"));
    }
    for token_id in token_ids {
        let token_id = token_id
            .as_u64()
            .ok_or_else(|| anyhow!("model_boundary_fixtures.token_ids entries must be integers"))?;
        if token_id >= vocab_size {
            return Err(anyhow!("model_boundary_fixtures token id must be inside vocab_size"));
        }
    }
    require_sha256(fixtures, "token_ids_sha256")?;
    require_u64_eq(fixtures, "fixtures_total", 3)?;
    let token_embedding_fixture = object_field(fixtures, "token_embedding")?;
    validate_dense_boundary_tensor_fixture(token_embedding_fixture, "token_embedding")?;
    let expected_embedding_len = seq_len
        .checked_mul(hidden_size)
        .ok_or_else(|| anyhow!("model_boundary_fixtures token_embedding output_len overflows"))?;
    require_u64_eq(token_embedding_fixture, "output_len", expected_embedding_len)?;

    let final_norm = object_field(fixtures, "final_norm")?;
    require_positive_number(final_norm, "rmsnorm_eps")?;
    require_string_non_empty(final_norm, "epsilon_source")?;
    require_sha256(final_norm, "input_sha256")?;
    require_sha256(final_norm, "output_sha256")?;
    let final_norm_fixture = object_field(final_norm, "fixture")?;
    validate_dense_boundary_tensor_fixture(final_norm_fixture, "final_norm")?;
    require_u64_eq(final_norm_fixture, "output_len", hidden_size)?;
    require_string_eq(
        final_norm_fixture,
        "output_sha256",
        required_string(final_norm, "output_sha256")?,
    )?;

    let lm_head = object_field(fixtures, "lm_head_logits")?;
    let logits_len = required_u64(lm_head, "logits_len")?;
    if logits_len == 0 {
        return Err(anyhow!("model_boundary_fixtures.lm_head_logits.logits_len must be positive"));
    }
    if logits_len != vocab_size {
        return Err(anyhow!(
            "model_boundary_fixtures.lm_head_logits.logits_len must match vocab_size"
        ));
    }
    require_sha256(lm_head, "logits_sha256")?;
    let top_k = required_u64(lm_head, "top_k")?;
    if top_k == 0 || top_k > logits_len {
        return Err(anyhow!(
            "model_boundary_fixtures.lm_head_logits.top_k must be in 1..=logits_len"
        ));
    }
    let top_k_entries = array_field(lm_head, "top_k_entries")?;
    if top_k_entries.len() != top_k as usize {
        return Err(anyhow!(
            "model_boundary_fixtures.lm_head_logits.top_k_entries length must match top_k"
        ));
    }
    for (idx, entry) in top_k_entries.iter().enumerate() {
        require_u64_eq(entry, "rank", idx as u64)?;
        let token_id = required_u64(entry, "token_id")?;
        if token_id >= logits_len {
            return Err(anyhow!(
                "model_boundary_fixtures top-k token_id must be inside logits_len"
            ));
        }
        require_number(entry, "value")?;
    }
    let lm_head_fixture = object_field(lm_head, "fixture")?;
    validate_dense_boundary_tensor_fixture(lm_head_fixture, "lm_head_logits")?;
    require_u64_eq(lm_head_fixture, "output_len", logits_len)?;
    require_string_eq(
        lm_head_fixture,
        "output_sha256",
        required_string(lm_head, "logits_sha256")?,
    )?;

    require_bool_eq(fixtures, "boundary_fixtures_claimed", true)?;
    require_bool_eq(fixtures, "token_embedding_fixture_claimed", true)?;
    require_bool_eq(fixtures, "final_norm_fixture_claimed", true)?;
    require_bool_eq(fixtures, "lm_head_logits_fixture_claimed", true)?;
    require_bool_eq(fixtures, "fixture_route_only", true)?;
    require_bool_eq(fixtures, "cuda_kernel_execution_claimed", false)?;
    require_u64_eq(fixtures, "kernel_invocations", 0)?;
    require_bool_eq(fixtures, "fallback_used", false)?;
    require_bool_eq(fixtures, "kv_cache_policy_claimed", false)?;
    require_bool_eq(fixtures, "sampling_integration_claimed", false)?;
    require_bool_eq(fixtures, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(fixtures, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(fixtures, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(fixtures, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(fixtures, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(fixtures, "speedup_claim", false)?;
    require_bool_eq(fixtures, "persistent_session_residency_claimed", false)?;
    require_bool_eq(fixtures, "full_cuda_residency_claimed", false)?;

    let remaining = object_field(receipt, "remaining_model_boundary_gaps")?;
    require_u64_eq(remaining, "schema", 1)?;
    let mut required_gaps = BTreeSet::from(["kv_cache_policy", "sampling"]);
    for gap in array_field(remaining, "gaps")? {
        let name = required_string(gap, "gap")?;
        required_gaps.remove(name);
        require_string_eq(gap, "status", "not_governed_by_model_boundary_fixtures")?;
        require_string_non_empty(gap, "required_next_proof")?;
        require_bool_eq(gap, "blocks_qwen_one_token", true)?;
        require_bool_eq(gap, "blocks_qwen_short_decode", true)?;
        require_bool_eq(gap, "blocks_qwen_chat", true)?;
    }
    if !required_gaps.is_empty() {
        return Err(anyhow!(
            "remaining_model_boundary_gaps missing required gaps: {required_gaps:?}"
        ));
    }
    require_bool_eq(remaining, "qwen_one_token_cuda_blocked", true)?;
    require_bool_eq(remaining, "qwen_short_decode_cuda_blocked", true)?;
    require_bool_eq(remaining, "qwen_chat_cuda_blocked", true)?;
    require_string_eq(remaining, "next_required_proof", "dense_gguf_kv_cache_policy_receipt")?;
    require_bool_eq(remaining, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(remaining, "speedup_claim", false)?;
    require_bool_eq(remaining, "full_cuda_residency_claimed", false)?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_cpu_reference_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_cuda_integrated_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_all_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_model_boundary_fixtures_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "kv_cache_policy_claimed", false)?;
    require_bool_eq(claim_boundary, "sampling_integration_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

/// Validate dense GGUF KV-cache policy evidence.
///
/// This artifact records model-derived KV-cache dimensions, the strict CUDA
/// residency policy, and estimated prefill/decode bytes. It does not allocate a
/// runtime KV cache, generate tokens, integrate sampling, claim speedup, claim
/// full CUDA residency, or prove BitNet packed I2_S/QK256 execution.
pub fn validate_dense_gguf_kv_cache_policy_receipt_json(receipt: &Value) -> Result<()> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", DENSE_GGUF_KV_CACHE_POLICY_ARTIFACT_KIND)?;
    require_string_eq(receipt, "claim", "dense_gguf_kv_cache_policy_recorded")?;
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
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_sha256(model, "sha256")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "dense_cuda_kv_cache_policy_route")?;
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

    validate_dense_one_layer_gap_execution_plan(receipt)?;
    let plan = object_field(receipt, "execution_plan")?;
    require_u64_eq(plan, "total_ops", 1)?;
    require_u64_eq(plan, "cuda_ops", 1)?;
    require_u64_eq(plan, "cuda_dense_regular_llm_ops", 1)?;

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_positive_u64(descriptor, "tensor_count")?;
    require_positive_u64(descriptor, "metadata_count")?;
    require_bool_eq(descriptor, "required_roles_present", true)?;
    require_bool_eq(descriptor, "strict_descriptor_complete", true)?;
    require_string_non_empty(descriptor, "dense_cuda_route_status")?;
    reject_bitnet_packed_marker(
        required_string(descriptor, "dense_cuda_route_status")?,
        "descriptor_coverage.dense_cuda_route_status",
    )?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;

    let policy = object_field(receipt, "kv_cache_policy")?;
    require_u64_eq(policy, "schema", 1)?;
    reject_bitnet_packed_marker(
        required_string(policy, "policy_id")?,
        "kv_cache_policy.policy_id",
    )?;
    require_string_eq(policy, "policy_scope", "dense_qwen_prefill_decode_boundary")?;
    require_string_eq(policy, "planned_residency", "cuda_required_for_strict_dense_cuda")?;
    require_string_eq(policy, "observed_residency", "not_allocated_policy_only")?;
    require_string_eq(policy, "kv_element_dtype", "f16")?;
    let bytes_per_element = required_u64(policy, "kv_element_bytes")?;
    if bytes_per_element != 2 {
        return Err(anyhow!("kv_cache_policy.kv_element_bytes must be 2 for f16 policy"));
    }

    let transformer_layers = required_u64(policy, "transformer_layers_total")?;
    let context_length = required_u64(policy, "context_length")?;
    let seq_len = required_u64(policy, "seq_len")?;
    let decode_steps = required_u64(policy, "decode_steps")?;
    let q_heads = required_u64(policy, "q_heads")?;
    let kv_heads = required_u64(policy, "kv_heads")?;
    let key_head_dim = required_u64(policy, "key_head_dim")?;
    let value_head_dim = required_u64(policy, "value_head_dim")?;
    let heads_per_kv_group = required_u64(policy, "heads_per_kv_group")?;
    if transformer_layers == 0
        || context_length == 0
        || seq_len == 0
        || decode_steps == 0
        || q_heads == 0
        || kv_heads == 0
        || key_head_dim == 0
        || value_head_dim == 0
        || heads_per_kv_group == 0
    {
        return Err(anyhow!("kv_cache_policy dimensions must be positive"));
    }
    if q_heads % kv_heads != 0 || q_heads / kv_heads != heads_per_kv_group {
        return Err(anyhow!(
            "kv_cache_policy q_heads must be divisible by kv_heads and match heads_per_kv_group"
        ));
    }
    if context_length < seq_len {
        return Err(anyhow!("kv_cache_policy context_length must cover seq_len"));
    }

    let values_per_token_per_layer = required_u64(policy, "kv_values_per_token_per_layer")?;
    let expected_values = kv_heads
        .checked_mul(
            key_head_dim
                .checked_add(value_head_dim)
                .ok_or_else(|| anyhow!("kv_cache_policy key/value dimension sum overflowed"))?,
        )
        .ok_or_else(|| anyhow!("kv_cache_policy values per token overflowed"))?;
    if values_per_token_per_layer != expected_values {
        return Err(anyhow!(
            "kv_cache_policy kv_values_per_token_per_layer must equal kv_heads * (key_head_dim + value_head_dim)"
        ));
    }
    let bytes_per_token_per_layer = required_u64(policy, "kv_bytes_per_token_per_layer")?;
    let expected_bytes_per_token_per_layer = values_per_token_per_layer
        .checked_mul(bytes_per_element)
        .ok_or_else(|| anyhow!("kv_cache_policy bytes per token per layer overflowed"))?;
    if bytes_per_token_per_layer != expected_bytes_per_token_per_layer {
        return Err(anyhow!(
            "kv_cache_policy kv_bytes_per_token_per_layer must equal kv_values_per_token_per_layer * kv_element_bytes"
        ));
    }
    let bytes_per_token_all_layers = required_u64(policy, "kv_bytes_per_token_all_layers")?;
    let expected_bytes_all_layers = bytes_per_token_per_layer
        .checked_mul(transformer_layers)
        .ok_or_else(|| anyhow!("kv_cache_policy bytes per token all layers overflowed"))?;
    if bytes_per_token_all_layers != expected_bytes_all_layers {
        return Err(anyhow!(
            "kv_cache_policy kv_bytes_per_token_all_layers must equal per-layer bytes times layer count"
        ));
    }

    let metadata = object_field(policy, "metadata_sources")?;
    require_string_non_empty(metadata, "transformer_layers")?;
    require_string_non_empty(metadata, "context_length")?;
    require_string_non_empty(metadata, "q_heads")?;
    require_string_non_empty(metadata, "kv_heads")?;
    require_string_non_empty(metadata, "key_head_dim")?;
    require_string_non_empty(metadata, "value_head_dim")?;

    let prefill = object_field(policy, "prefill")?;
    require_u64_eq(prefill, "write_tokens", seq_len)?;
    require_bool_eq(prefill, "writes_keys", true)?;
    require_bool_eq(prefill, "writes_values", true)?;
    require_u64_eq(
        prefill,
        "write_bytes_estimate",
        bytes_per_token_all_layers
            .checked_mul(seq_len)
            .ok_or_else(|| anyhow!("kv_cache_policy prefill bytes overflowed"))?,
    )?;
    require_string_eq(prefill, "write_path", "qkv_projection_to_cuda_kv_cache")?;
    require_bool_eq(prefill, "measured", false)?;

    let decode = object_field(policy, "decode")?;
    require_u64_eq(decode, "decode_steps", decode_steps)?;
    require_u64_eq(decode, "read_tokens_per_step", seq_len)?;
    require_u64_eq(
        decode,
        "read_bytes_per_step_estimate",
        bytes_per_token_all_layers
            .checked_mul(seq_len)
            .ok_or_else(|| anyhow!("kv_cache_policy decode read bytes overflowed"))?,
    )?;
    require_u64_eq(decode, "write_tokens_per_step", 1)?;
    require_u64_eq(decode, "write_bytes_per_step_estimate", bytes_per_token_all_layers)?;
    require_string_eq(decode, "read_path", "cuda_kv_cache_to_attention")?;
    require_string_eq(decode, "write_path", "qkv_projection_to_cuda_kv_cache")?;
    require_bool_eq(decode, "measured", false)?;

    let max_context = object_field(policy, "max_context")?;
    require_u64_eq(max_context, "tokens", context_length)?;
    require_u64_eq(
        max_context,
        "bytes_estimate",
        bytes_per_token_all_layers
            .checked_mul(context_length)
            .ok_or_else(|| anyhow!("kv_cache_policy max context bytes overflowed"))?,
    )?;

    require_bool_eq(policy, "kv_cache_policy_claimed", true)?;
    require_bool_eq(policy, "runtime_kv_cache_allocated", false)?;
    require_bool_eq(policy, "kv_cache_cuda_residency_claimed", false)?;
    require_bool_eq(policy, "estimated_bytes_only", true)?;
    require_bool_eq(policy, "transfer_bytes_measured", false)?;
    require_bool_eq(policy, "transfer_timing_measured", false)?;
    require_bool_eq(policy, "fallback_used", false)?;
    require_bool_eq(policy, "sampling_integration_claimed", false)?;
    require_bool_eq(policy, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(policy, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(policy, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(policy, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(policy, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(policy, "speedup_claim", false)?;
    require_bool_eq(policy, "persistent_session_residency_claimed", false)?;
    require_bool_eq(policy, "full_cuda_residency_claimed", false)?;

    let remaining = object_field(receipt, "remaining_model_boundary_gaps")?;
    require_u64_eq(remaining, "schema", 1)?;
    let gaps = array_field(remaining, "gaps")?;
    if gaps.len() != 1 {
        return Err(anyhow!("kv cache policy receipt must leave exactly the sampling gap"));
    }
    let sampling_gap = &gaps[0];
    require_string_eq(sampling_gap, "gap", "sampling")?;
    require_string_eq(sampling_gap, "status", "not_governed_by_kv_cache_policy")?;
    require_string_eq(sampling_gap, "required_next_proof", "dense_gguf_sampling_policy_receipt")?;
    require_bool_eq(sampling_gap, "blocks_qwen_one_token", true)?;
    require_bool_eq(sampling_gap, "blocks_qwen_short_decode", true)?;
    require_bool_eq(sampling_gap, "blocks_qwen_chat", true)?;
    require_bool_eq(remaining, "kv_cache_policy_claimed", true)?;
    require_bool_eq(remaining, "sampling_integration_claimed", false)?;
    require_bool_eq(remaining, "qwen_one_token_cuda_blocked", true)?;
    require_bool_eq(remaining, "qwen_short_decode_cuda_blocked", true)?;
    require_bool_eq(remaining, "qwen_chat_cuda_blocked", true)?;
    require_string_eq(remaining, "next_required_proof", "dense_gguf_sampling_policy_receipt")?;
    require_bool_eq(remaining, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(remaining, "speedup_claim", false)?;
    require_bool_eq(remaining, "full_cuda_residency_claimed", false)?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_cpu_reference_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_cuda_integrated_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_all_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_model_boundary_fixtures_claimed", true)?;
    require_bool_eq(claim_boundary, "kv_cache_policy_claimed", true)?;
    require_bool_eq(claim_boundary, "kv_cache_cuda_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "sampling_integration_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

/// Validate dense GGUF logits-transfer and sampling-policy evidence.
///
/// This artifact records the governed logits boundary and deterministic CPU
/// sampler policy needed before Qwen one-token CUDA proof. It does not execute
/// runtime sampling, generate tokens, claim dense GGUF inference, claim speedup,
/// claim full CUDA residency, or prove BitNet packed I2_S/QK256 execution.
pub fn validate_dense_gguf_sampling_policy_receipt_json(receipt: &Value) -> Result<()> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", DENSE_GGUF_SAMPLING_POLICY_ARTIFACT_KIND)?;
    require_string_eq(receipt, "claim", "dense_gguf_sampling_policy_recorded")?;
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
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_sha256(model, "sha256")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "dense_cuda_sampling_policy_route")?;
    reject_bitnet_packed_marker(
        required_string(execution_path, "kernel_family")?,
        "execution_path.kernel_family",
    )?;
    require_string_eq(
        execution_path,
        "quantization_family",
        "dense_gguf_q8_0_f16_logits_sampling_policy_contract",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    validate_dense_one_layer_gap_execution_plan(receipt)?;
    let plan = object_field(receipt, "execution_plan")?;
    require_u64_eq(plan, "total_ops", 1)?;
    require_u64_eq(plan, "cuda_ops", 1)?;
    require_u64_eq(plan, "cuda_dense_regular_llm_ops", 1)?;

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_positive_u64(descriptor, "tensor_count")?;
    require_positive_u64(descriptor, "metadata_count")?;
    require_bool_eq(descriptor, "required_roles_present", true)?;
    require_bool_eq(descriptor, "strict_descriptor_complete", true)?;
    require_string_non_empty(descriptor, "dense_cuda_route_status")?;
    reject_bitnet_packed_marker(
        required_string(descriptor, "dense_cuda_route_status")?,
        "descriptor_coverage.dense_cuda_route_status",
    )?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;
    let quantization_families = array_field(descriptor, "quantization_families")?;
    if quantization_families.is_empty() {
        return Err(anyhow!("descriptor_coverage.quantization_families must not be empty"));
    }
    for family in quantization_families {
        let family = family.as_str().ok_or_else(|| {
            anyhow!("descriptor_coverage.quantization_families entries must be strings")
        })?;
        reject_bitnet_packed_marker(family, "descriptor_coverage.quantization_families")?;
    }

    let policy = object_field(receipt, "sampling_policy")?;
    require_u64_eq(policy, "schema", 1)?;
    reject_bitnet_packed_marker(
        required_string(policy, "policy_id")?,
        "sampling_policy.policy_id",
    )?;
    require_string_eq(policy, "policy_scope", "dense_qwen_logits_to_sampler_boundary")?;
    require_string_eq(policy, "logits_source", "dense_gguf_model_boundary_lm_head_logits")?;
    require_sha256(policy, "logits_sha256")?;
    let logits_len = required_u64(policy, "logits_len")?;
    let vocab_size = required_u64(policy, "vocab_size")?;
    let seq_len = required_u64(policy, "seq_len")?;
    if logits_len == 0 || vocab_size == 0 || seq_len == 0 {
        return Err(anyhow!(
            "sampling_policy logits_len, vocab_size, and seq_len must be positive"
        ));
    }
    if logits_len != vocab_size {
        return Err(anyhow!("sampling_policy logits_len must equal vocab_size"));
    }
    require_string_eq(policy, "logits_dtype", "f32")?;
    let logits_element_bytes = required_u64(policy, "logits_element_bytes")?;
    if logits_element_bytes != 4 {
        return Err(anyhow!("sampling_policy.logits_element_bytes must be 4 for f32 logits"));
    }
    require_u64_eq(
        policy,
        "logits_transfer_bytes_per_step_estimate",
        logits_len
            .checked_mul(logits_element_bytes)
            .ok_or_else(|| anyhow!("sampling_policy logits transfer byte estimate overflowed"))?,
    )?;
    require_string_eq(policy, "logits_transfer_path", "cuda_lm_head_logits_to_cpu_sampler")?;
    require_bool_eq(policy, "logits_transfer_required_for_cpu_sampling", true)?;
    require_bool_eq(policy, "logits_transfer_bytes_measured", false)?;
    require_bool_eq(policy, "logits_transfer_timing_measured", false)?;
    require_string_eq(policy, "sampler_backend", "bitnet-sampling")?;
    require_string_eq(policy, "sampler_location", "cpu")?;
    require_string_eq(policy, "sampler_mode", "greedy")?;
    let temperature = object_field(policy, "temperature")?
        .as_f64()
        .ok_or_else(|| anyhow!("field `temperature` must be a number"))?;
    if temperature != 0.0 {
        return Err(anyhow!("sampling_policy.temperature must be 0.0 for greedy policy"));
    }
    require_u64_eq(policy, "top_k_filter", 0)?;
    let top_p = object_field(policy, "top_p")?
        .as_f64()
        .ok_or_else(|| anyhow!("field `top_p` must be a number"))?;
    if top_p != 1.0 {
        return Err(anyhow!("sampling_policy.top_p must be 1.0 for greedy policy"));
    }
    let repetition_penalty = object_field(policy, "repetition_penalty")?
        .as_f64()
        .ok_or_else(|| anyhow!("field `repetition_penalty` must be a number"))?;
    if repetition_penalty != 1.0 {
        return Err(anyhow!("sampling_policy.repetition_penalty must be 1.0 for fixture policy"));
    }
    require_bool_eq(policy, "deterministic", true)?;
    require_string_eq(policy, "tie_break_policy", "lowest_token_id")?;
    require_bool_eq(policy, "rng_required", false)?;
    let selected_token = required_u64(policy, "selected_token_id_from_fixture_logits")?;
    if selected_token >= logits_len {
        return Err(anyhow!(
            "sampling_policy.selected_token_id_from_fixture_logits must be inside logits range"
        ));
    }
    require_string_eq(policy, "selected_token_scope", "fixture_logits_only_not_generation")?;
    let top_k = required_u64(policy, "top_k")?;
    let top_k_entries = array_field(policy, "top_k_entries")?;
    if top_k == 0 || top_k_entries.is_empty() {
        return Err(anyhow!("sampling_policy must record non-empty top_k_entries"));
    }
    if top_k_entries.len() as u64 != top_k {
        return Err(anyhow!("sampling_policy.top_k must match top_k_entries length"));
    }
    if top_k > logits_len {
        return Err(anyhow!("sampling_policy.top_k cannot exceed logits_len"));
    }
    for (idx, entry) in top_k_entries.iter().enumerate() {
        require_u64_eq(entry, "rank", idx as u64)?;
        let token_id = required_u64(entry, "token_id")?;
        if token_id >= logits_len {
            return Err(anyhow!("sampling_policy.top_k_entries token_id outside logits range"));
        }
        require_number(entry, "value")?;
    }
    require_u64_eq(&top_k_entries[0], "token_id", selected_token)?;

    require_bool_eq(policy, "sampling_policy_claimed", true)?;
    require_bool_eq(policy, "sampling_integration_claimed", false)?;
    require_bool_eq(policy, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(policy, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(policy, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(policy, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(policy, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(policy, "speedup_claim", false)?;
    require_bool_eq(policy, "persistent_session_residency_claimed", false)?;
    require_bool_eq(policy, "full_cuda_residency_claimed", false)?;

    let remaining = object_field(receipt, "remaining_model_boundary_gaps")?;
    require_u64_eq(remaining, "schema", 1)?;
    let gaps = array_field(remaining, "gaps")?;
    if !gaps.is_empty() {
        return Err(anyhow!("sampling policy receipt must clear model-boundary policy gaps"));
    }
    require_bool_eq(remaining, "all_model_boundary_policies_governed", true)?;
    require_bool_eq(remaining, "kv_cache_policy_claimed", true)?;
    require_bool_eq(remaining, "sampling_policy_claimed", true)?;
    require_bool_eq(remaining, "sampling_integration_claimed", false)?;
    require_bool_eq(remaining, "qwen_one_token_cuda_blocked", false)?;
    require_bool_eq(remaining, "qwen_short_decode_cuda_blocked", true)?;
    require_bool_eq(remaining, "qwen_chat_cuda_blocked", true)?;
    require_string_eq(remaining, "next_required_proof", "qwen_one_token_strict_cuda_proof")?;
    require_bool_eq(remaining, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(remaining, "speedup_claim", false)?;
    require_bool_eq(remaining, "full_cuda_residency_claimed", false)?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_cpu_reference_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_cuda_integrated_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_all_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_model_boundary_fixtures_claimed", true)?;
    require_bool_eq(claim_boundary, "kv_cache_policy_claimed", true)?;
    require_bool_eq(claim_boundary, "kv_cache_cuda_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "sampling_policy_claimed", true)?;
    require_bool_eq(claim_boundary, "sampling_integration_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

fn validate_dense_qwen_transfer_timing(timing: &Value, transfer: &Value) -> Result<()> {
    let timing_status = required_string(timing, "transfer_timing_status")?;
    require_string_eq(transfer, "transfer_timing_status", timing_status)?;

    match timing_status {
        "device_to_host_measured_host_to_device_unmeasured" => {
            require_null(timing, "host_to_device_ms")?;
            require_string_eq(
                timing,
                "host_to_device_ms_source",
                "not_measured_by_dense_qwen_runtime",
            )?;
            require_null(transfer, "host_to_device_ms")?;
            require_string_eq(
                transfer,
                "host_to_device_ms_source",
                "not_measured_by_dense_qwen_runtime",
            )?;
        }
        "host_to_device_model_load_envelope_device_to_host_measured" => {
            require_non_negative_number(timing, "host_to_device_ms")?;
            require_string_eq(
                timing,
                "host_to_device_ms_source",
                "wall_clock_model_load_with_cuda_weight_upload",
            )?;
            require_string_eq(timing, "host_to_device_ms_scope", "model_load_wall_clock_envelope")?;
            require_bool_eq(timing, "host_to_device_ms_includes_non_transfer_overhead", true)?;

            require_non_negative_number(transfer, "host_to_device_ms")?;
            require_string_eq(
                transfer,
                "host_to_device_ms_source",
                "wall_clock_model_load_with_cuda_weight_upload",
            )?;
            require_string_eq(
                transfer,
                "host_to_device_ms_scope",
                "model_load_wall_clock_envelope",
            )?;
            require_bool_eq(transfer, "host_to_device_ms_includes_non_transfer_overhead", true)?;

            let timing_h2d = timing
                .get("host_to_device_ms")
                .and_then(Value::as_f64)
                .ok_or_else(|| anyhow!("timing.host_to_device_ms must be a number"))?;
            let transfer_h2d = transfer
                .get("host_to_device_ms")
                .and_then(Value::as_f64)
                .ok_or_else(|| anyhow!("transfer_accounting.host_to_device_ms must be a number"))?;
            if (timing_h2d - transfer_h2d).abs() > f64::EPSILON {
                return Err(anyhow!(
                    "timing.host_to_device_ms must match tensor_residency.transfer_accounting.host_to_device_ms"
                ));
            }
        }
        other => {
            return Err(anyhow!(
                "field `transfer_timing_status` must be a supported dense Qwen transfer timing status, got `{other}`"
            ));
        }
    }

    require_non_negative_number(timing, "device_to_host_ms")?;
    let d2h_source = required_string(timing, "device_to_host_ms_source")?;
    if !matches!(
        d2h_source,
        "wall_clock_extract_logits_2d_local" | "wall_clock_device_top_k_cuda_sampler"
    ) {
        return Err(anyhow!(
            "field `device_to_host_ms_source` must be a supported dense Qwen D2H timing source, got `{d2h_source}`"
        ));
    }
    require_non_negative_number(transfer, "device_to_host_ms")?;
    require_string_eq(transfer, "device_to_host_ms_source", d2h_source)?;

    let timing_d2h = timing
        .get("device_to_host_ms")
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow!("timing.device_to_host_ms must be a number"))?;
    let transfer_d2h = transfer
        .get("device_to_host_ms")
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow!("transfer_accounting.device_to_host_ms must be a number"))?;
    if (timing_d2h - transfer_d2h).abs() > f64::EPSILON {
        return Err(anyhow!(
            "timing.device_to_host_ms must match tensor_residency.transfer_accounting.device_to_host_ms"
        ));
    }

    Ok(())
}

fn dense_qwen_reduced_logits_transfer_requested(receipt: &Value) -> bool {
    receipt
        .get("logits_transfer_reduction")
        .and_then(|reduction| reduction.get("device_to_host_bytes_reduced"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn validate_dense_qwen_step_logits_sha256(step: &Value, reduced_cuda_transfer: bool) -> Result<()> {
    require_sha256(step, "cpu_logits_sha256")?;
    if step.get("cpu_logits_sha256_available").is_some() {
        require_bool_eq(step, "cpu_logits_sha256_available", true)?;
    }
    if step.get("cpu_logits_sha256_source").is_some() {
        require_string_eq(step, "cpu_logits_sha256_source", "full_logits_download")?;
    }

    if reduced_cuda_transfer {
        if let Some(value) = step.get("cuda_logits_sha256")
            && !value.is_null()
        {
            return Err(anyhow!(
                "cuda_logits_sha256 must be null or omitted when reduced device top-k transfer is claimed"
            ));
        }
        require_bool_eq(step, "cuda_logits_sha256_available", false)?;
        require_string_eq(
            step,
            "cuda_logits_sha256_source",
            "not_recorded_reduced_device_top_k_sampler",
        )?;
    } else {
        require_sha256(step, "cuda_logits_sha256")?;
        if step.get("cuda_logits_sha256_available").is_some() {
            require_bool_eq(step, "cuda_logits_sha256_available", true)?;
        }
        if step.get("cuda_logits_sha256_source").is_some() {
            require_string_eq(step, "cuda_logits_sha256_source", "full_logits_download")?;
        }
    }

    Ok(())
}

fn validate_dense_qwen_logits_transfer_reduction(
    receipt: &Value,
    stats_d2h: u64,
    generated_tokens: u64,
) -> Result<()> {
    let Some(reduction) = receipt.get("logits_transfer_reduction") else {
        return Ok(());
    };
    if !reduction.is_object() {
        return Err(anyhow!("logits_transfer_reduction must be an object when present"));
    }

    require_u64_eq(reduction, "schema", 1)?;
    require_string_eq(reduction, "scope", "dense_qwen_logits_top_k_transfer")?;
    let transfer_mode = required_string(reduction, "transfer_mode")?;
    if transfer_mode.trim().is_empty() {
        return Err(anyhow!("field `transfer_mode` must not be empty"));
    }
    let sampling_location = required_string(reduction, "sampling_location")?;
    if sampling_location.trim().is_empty() {
        return Err(anyhow!("field `sampling_location` must not be empty"));
    }
    let requested_top_k = required_u64(reduction, "requested_top_k")?;
    if requested_top_k == 0 {
        return Err(anyhow!("logits_transfer_reduction.requested_top_k must be positive"));
    }
    require_u64_eq(reduction, "generated_tokens_count", generated_tokens)?;
    let logits_vector_length = required_u64(reduction, "logits_vector_length")?;
    let logits_element_bytes = required_u64(reduction, "logits_element_bytes")?;
    if logits_vector_length == 0 || logits_element_bytes == 0 {
        return Err(anyhow!(
            "logits_transfer_reduction logits vector length and element bytes must be positive"
        ));
    }
    let full_logits_bytes_per_step = required_u64(reduction, "full_logits_bytes_per_step")?;
    let full_logits_download_bytes = required_u64(reduction, "full_logits_download_bytes")?;
    if full_logits_bytes_per_step == 0 || full_logits_download_bytes == 0 {
        return Err(anyhow!("logits_transfer_reduction full logits byte counts must be positive"));
    }
    let expected_full_logits_bytes_per_step = logits_vector_length
        .checked_mul(logits_element_bytes)
        .ok_or_else(|| anyhow!("logits_transfer_reduction logits byte count overflows"))?;
    if full_logits_bytes_per_step != expected_full_logits_bytes_per_step {
        return Err(anyhow!(
            "logits_transfer_reduction.full_logits_bytes_per_step must equal logits_vector_length * logits_element_bytes"
        ));
    }
    let expected_full_bytes = full_logits_bytes_per_step
        .checked_mul(generated_tokens)
        .ok_or_else(|| anyhow!("logits_transfer_reduction full logits byte count overflows"))?;
    if full_logits_download_bytes != expected_full_bytes {
        return Err(anyhow!(
            "logits_transfer_reduction.full_logits_download_bytes must equal full_logits_bytes_per_step * generated_tokens_count"
        ));
    }
    let actual_device_to_host_bytes = required_u64(reduction, "actual_device_to_host_bytes")?;
    if actual_device_to_host_bytes != stats_d2h {
        return Err(anyhow!(
            "logits_transfer_reduction.actual_device_to_host_bytes must match measured device_to_host_bytes"
        ));
    }
    let top_k_result_bytes_per_step_floor =
        required_u64(reduction, "top_k_result_bytes_per_step_floor")?;
    let top_k_result_bytes_total_floor = required_u64(reduction, "top_k_result_bytes_total_floor")?;
    let selected_token_bytes_total_floor =
        required_u64(reduction, "selected_token_bytes_total_floor")?;
    if top_k_result_bytes_per_step_floor == 0
        || top_k_result_bytes_total_floor == 0
        || selected_token_bytes_total_floor == 0
    {
        return Err(anyhow!(
            "logits_transfer_reduction preserved-evidence byte floors must be positive"
        ));
    }
    let expected_top_k_bytes_per_step = requested_top_k
        .checked_mul(12)
        .ok_or_else(|| anyhow!("logits_transfer_reduction top-k byte floor overflows"))?;
    if top_k_result_bytes_per_step_floor != expected_top_k_bytes_per_step {
        return Err(anyhow!(
            "logits_transfer_reduction.top_k_result_bytes_per_step_floor must equal requested_top_k * 12"
        ));
    }
    let expected_top_k_bytes_total = top_k_result_bytes_per_step_floor
        .checked_mul(generated_tokens)
        .ok_or_else(|| anyhow!("logits_transfer_reduction top-k byte total overflows"))?;
    if top_k_result_bytes_total_floor != expected_top_k_bytes_total {
        return Err(anyhow!(
            "logits_transfer_reduction.top_k_result_bytes_total_floor must equal top_k_result_bytes_per_step_floor * generated_tokens_count"
        ));
    }
    let expected_selected_token_bytes_total = 4_u64
        .checked_mul(generated_tokens)
        .ok_or_else(|| anyhow!("logits_transfer_reduction selected-token byte floor overflows"))?;
    if selected_token_bytes_total_floor != expected_selected_token_bytes_total {
        return Err(anyhow!(
            "logits_transfer_reduction.selected_token_bytes_total_floor must equal 4 * generated_tokens_count"
        ));
    }
    require_bool_eq(reduction, "selected_token_equality_preserved", true)?;
    require_bool_eq(reduction, "top_k_evidence_preserved", true)?;
    require_bool_eq(reduction, "quality_receipts_unchanged", true)?;

    let reduced = object_field(reduction, "device_to_host_bytes_reduced")?
        .as_bool()
        .ok_or_else(|| anyhow!("field `device_to_host_bytes_reduced` must be a bool"))?;
    let bytes_saved = required_u64(reduction, "bytes_saved_vs_full_logits")?;
    if reduced {
        if !matches!(transfer_mode, "device_top_k_cuda_sampler" | "device_greedy_cuda_sampler") {
            return Err(anyhow!(
                "logits_transfer_reduction.device_to_host_bytes_reduced requires a device_top_k_cuda_sampler or device_greedy_cuda_sampler transfer_mode"
            ));
        }
        if sampling_location != "cuda_device" {
            return Err(anyhow!(
                "logits_transfer_reduction.device_to_host_bytes_reduced requires sampling_location=cuda_device"
            ));
        }
        if let Some(blocker) = reduction.get("reduction_blocker")
            && !blocker.is_null()
        {
            return Err(anyhow!(
                "logits_transfer_reduction.reduction_blocker must be omitted or null when device_to_host_bytes_reduced is true"
            ));
        }
        if actual_device_to_host_bytes >= full_logits_download_bytes {
            return Err(anyhow!(
                "logits_transfer_reduction.device_to_host_bytes_reduced requires actual_device_to_host_bytes < full_logits_download_bytes"
            ));
        }
        if bytes_saved != full_logits_download_bytes - actual_device_to_host_bytes {
            return Err(anyhow!(
                "logits_transfer_reduction.bytes_saved_vs_full_logits must equal the measured D2H reduction"
            ));
        }
        let timing = object_field(receipt, "timing")?;
        let residency = object_field(receipt, "tensor_residency")?;
        let transfer = object_field(residency, "transfer_accounting")?;
        require_string_eq(
            timing,
            "device_to_host_ms_source",
            "wall_clock_device_top_k_cuda_sampler",
        )?;
        require_string_eq(
            transfer,
            "device_to_host_ms_source",
            "wall_clock_device_top_k_cuda_sampler",
        )?;
    } else {
        require_string_eq(reduction, "transfer_mode", "full_logits_download_cpu_sampler")?;
        require_string_eq(reduction, "sampling_location", "cpu")?;
        require_string_non_empty(reduction, "reduction_blocker")?;
        if actual_device_to_host_bytes != full_logits_download_bytes {
            return Err(anyhow!(
                "logits_transfer_reduction non-reduced receipts must account for full logits D2H bytes"
            ));
        }
        if bytes_saved != 0 {
            return Err(anyhow!(
                "logits_transfer_reduction.bytes_saved_vs_full_logits must be 0 when device_to_host_bytes_reduced is false"
            ));
        }
        let timing = object_field(receipt, "timing")?;
        let residency = object_field(receipt, "tensor_residency")?;
        let transfer = object_field(residency, "transfer_accounting")?;
        require_string_eq(
            timing,
            "device_to_host_ms_source",
            "wall_clock_extract_logits_2d_local",
        )?;
        require_string_eq(
            transfer,
            "device_to_host_ms_source",
            "wall_clock_extract_logits_2d_local",
        )?;
    }

    Ok(())
}

/// Validate the exact dense Qwen model identities allowed in runtime CUDA proofs.
fn require_verified_dense_qwen_runtime_model(model: &Value) -> Result<()> {
    let id = required_string(model, "id")?;
    match id {
        QWEN25_05B_INSTRUCT_Q8_0_MODEL_ID => {
            require_string_eq(model, "file", QWEN25_05B_INSTRUCT_Q8_0_MODEL_FILE)?;
            require_string_eq(model, "architecture", "qwen2")?;
            require_sha256(model, "sha256")?;
            require_string_eq(model, "sha256", QWEN25_05B_INSTRUCT_Q8_0_MODEL_SHA256)?;
        }
        QWEN3_06B_INSTRUCT_Q8_0_MODEL_ID => {
            require_string_eq(model, "file", QWEN3_06B_INSTRUCT_Q8_0_MODEL_FILE)?;
            require_string_eq(model, "architecture", "qwen3")?;
            require_sha256(model, "sha256")?;
            require_string_eq(model, "sha256", QWEN3_06B_INSTRUCT_Q8_0_MODEL_SHA256)?;
        }
        other => {
            return Err(anyhow!("dense Qwen runtime proof has unverified model id `{other}`"));
        }
    }

    Ok(())
}

pub fn validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(
    receipt: &Value,
) -> Result<()> {
    validate_cuda_receipt_common(
        receipt,
        DENSE_GGUF_QWEN_ONE_TOKEN_STRICT_CUDA_PROOF_ARTIFACT_KIND,
        "dense_gguf_qwen_one_token_strict_cuda_proof_recorded",
    )?;
    require_bool_eq(receipt, "speedup_claim", false)?;

    let model = object_field(receipt, "model")?;
    require_string_eq(model, "model_family", "qwen")?;
    require_verified_dense_qwen_runtime_model(model)?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "dense_qwen_one_token_strict_cuda")?;
    require_string_non_empty(execution_path, "quantization_family")?;
    reject_bitnet_packed_marker(
        required_string(execution_path, "quantization_family")?,
        "execution_path.quantization_family",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    validate_dense_one_layer_gap_execution_plan(receipt)?;

    let prerequisites = object_field(receipt, "prerequisite_receipts")?;
    require_u64_eq(prerequisites, "schema", 1)?;
    require_string_eq(
        prerequisites,
        "all_layer_execution_plan_artifact_kind",
        DENSE_GGUF_ALL_LAYER_EXECUTION_PLAN_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "all_layer_execution_plan_receipt_sha256")?;
    require_string_eq(
        prerequisites,
        "model_boundary_fixtures_artifact_kind",
        DENSE_GGUF_MODEL_BOUNDARY_FIXTURES_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "model_boundary_fixtures_receipt_sha256")?;
    require_string_eq(
        prerequisites,
        "kv_cache_policy_artifact_kind",
        DENSE_GGUF_KV_CACHE_POLICY_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "kv_cache_policy_receipt_sha256")?;
    require_string_eq(
        prerequisites,
        "sampling_policy_artifact_kind",
        DENSE_GGUF_SAMPLING_POLICY_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "sampling_policy_receipt_sha256")?;
    require_bool_eq(prerequisites, "all_required_receipts_verified", true)?;
    require_bool_eq(prerequisites, "sampling_policy_claimed", true)?;
    require_bool_eq(prerequisites, "kv_cache_policy_claimed", true)?;
    require_bool_eq(prerequisites, "model_boundary_fixtures_claimed", true)?;
    require_bool_eq(prerequisites, "all_layer_execution_plan_claimed", true)?;

    let authority = object_field(receipt, "tokenizer_prompt_authority")?;
    require_u64_eq(authority, "schema", 1)?;
    require_string_eq(authority, "tokenizer_authority", "contract_authoritative")?;
    require_string_eq(authority, "prompt_authority", "contract_authoritative")?;
    require_string_non_empty(authority, "prompt_template")?;
    require_string_non_empty(authority, "bos_policy")?;
    require_bool_eq(authority, "deterministic_prompt", true)?;
    require_positive_u64(authority, "prompt_token_count")?;
    require_sha256(authority, "prompt_token_ids_sha256")?;
    require_sha256(authority, "rendered_prompt_sha256")?;
    let authority_prompt_token_ids_sha256 = required_string(authority, "prompt_token_ids_sha256")?;

    let proof = object_field(receipt, "one_token_proof")?;
    require_u64_eq(proof, "schema", 1)?;
    require_string_eq(proof, "proof_scope", "qwen_strict_one_token_greedy_decode")?;
    require_string_eq(proof, "model_family", "qwen")?;
    require_u64_eq(proof, "requested_new_tokens", 1)?;
    require_u64_eq(proof, "generated_tokens_count", 1)?;
    require_string_eq(proof, "generation_policy", "greedy")?;
    require_bool_eq(proof, "deterministic", true)?;
    require_bool_eq(proof, "fallback_used", false)?;
    require_string_eq(proof, "cpu_reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(proof, "cuda_target_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_positive_u64(proof, "prompt_token_count")?;
    require_sha256(proof, "prompt_token_ids_sha256")?;
    let proof_prompt_token_ids_sha256 = required_string(proof, "prompt_token_ids_sha256")?;
    if proof_prompt_token_ids_sha256 != authority_prompt_token_ids_sha256 {
        return Err(anyhow!(
            "one_token_proof.prompt_token_ids_sha256 must match tokenizer_prompt_authority.prompt_token_ids_sha256"
        ));
    }
    require_sha256(proof, "cpu_logits_top_k_sha256")?;
    require_sha256(proof, "cuda_logits_top_k_sha256")?;
    let cpu_top_k_sha256 = required_string(proof, "cpu_logits_top_k_sha256")?;
    let cuda_top_k_sha256 = required_string(proof, "cuda_logits_top_k_sha256")?;
    if cpu_top_k_sha256 != cuda_top_k_sha256 {
        return Err(anyhow!(
            "one_token_proof.cpu_logits_top_k_sha256 must match one_token_proof.cuda_logits_top_k_sha256"
        ));
    }
    require_bool_eq(proof, "top_k_evidence_recorded", true)?;
    require_bool_eq(proof, "top_k_compared", true)?;
    require_bool_eq(proof, "top_k_match", true)?;
    require_bool_eq(proof, "selected_token_match", true)?;
    let cpu_token = required_u64(proof, "cpu_selected_token_id")?;
    let cuda_token = required_u64(proof, "cuda_selected_token_id")?;
    if cpu_token != cuda_token {
        return Err(anyhow!(
            "one_token_proof cpu_selected_token_id must match cuda_selected_token_id"
        ));
    }
    require_string_non_empty(proof, "decoded_token_text")?;
    require_bool_eq(proof, "qwen_one_token_cuda_claimed", true)?;
    require_bool_eq(proof, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(proof, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(proof, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(proof, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(proof, "speedup_claim", false)?;
    require_bool_eq(proof, "server_ready_claimed", false)?;
    require_bool_eq(proof, "full_cuda_residency_claimed", false)?;

    let quality = object_field(receipt, "quality_gate")?;
    require_u64_eq(quality, "schema", 1)?;
    require_string_eq(quality, "gate", "qwen_one_token_cuda_parity")?;
    require_bool_eq(quality, "passed", true)?;
    require_bool_eq(quality, "answer_ready_claimed", false)?;
    require_bool_eq(quality, "short_decode_claimed", false)?;
    require_bool_eq(quality, "chat_claimed", false)?;

    let stats = array_field(receipt, "kernel_stats")?;
    if stats.is_empty() {
        return Err(anyhow!("kernel_stats must contain dense CUDA token-generation entries"));
    }
    let mut stats_h2d = 0_u64;
    let mut stats_d2h = 0_u64;
    let mut stats_invocations = 0_u64;
    let mut stats_launches = 0_u64;
    for stat in stats {
        require_string_non_empty(stat, "kernel_id")?;
        reject_bitnet_packed_marker(required_string(stat, "kernel_id")?, "kernel_stats.kernel_id")?;
        require_positive_u64(stat, "invocations")?;
        require_u64_eq(stat, "fallback_invocations", 0)?;
        require_u64_eq(stat, "cpu_fallback_invocations", 0)?;
        required_u64(stat, "host_to_device_bytes")?;
        required_u64(stat, "device_to_host_bytes")?;
        require_positive_u64(stat, "kernel_launches")?;
        require_optional_non_negative_number(stat, "kernel_time_ms")?;
        stats_h2d += required_u64(stat, "host_to_device_bytes")?;
        stats_d2h += required_u64(stat, "device_to_host_bytes")?;
        stats_invocations += required_u64(stat, "invocations")?;
        stats_launches += required_u64(stat, "kernel_launches")?;
    }

    let kernel_coverage = object_field(receipt, "kernel_coverage")?;
    require_u64_eq(kernel_coverage, "schema", 1)?;
    require_string_eq(kernel_coverage, "route", DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND)?;
    require_bool_eq(kernel_coverage, "all_required_dense_kernels_executed", true)?;
    require_u64_eq(kernel_coverage, "bitnet_qk256_kernel_invocations", 0)?;
    require_u64_eq(kernel_coverage, "cpu_fallback_kernel_invocations", 0)?;
    require_u64_eq(kernel_coverage, "dense_kernel_invocations", stats_invocations)?;
    require_u64_eq(kernel_coverage, "dense_kernel_launches", stats_launches)?;
    require_bool_eq(kernel_coverage, "fallback_used", false)?;
    let kernels = array_field(kernel_coverage, "kernels_executed")?;
    if kernels.is_empty() {
        return Err(anyhow!("kernel_coverage.kernels_executed must not be empty"));
    }
    for kernel in kernels {
        let kernel = kernel
            .as_str()
            .ok_or_else(|| anyhow!("kernel_coverage.kernels_executed entries must be strings"))?;
        reject_bitnet_packed_marker(kernel, "kernel_coverage.kernels_executed")?;
    }

    let timing = object_field(receipt, "timing")?;
    require_non_negative_number(timing, "total_ms")?;
    require_non_negative_number(timing, "first_token_ms")?;
    require_non_negative_number(timing, "logits_download_ms")?;
    require_non_negative_number(timing, "kernel_time_ms")?;
    require_u64_eq(timing, "host_to_device_bytes", stats_h2d)?;
    require_u64_eq(timing, "device_to_host_bytes", stats_d2h)?;
    require_u64_eq(timing, "kernel_invocations", stats_invocations)?;
    require_u64_eq(timing, "kernel_launches", stats_launches)?;

    let residency = object_field(receipt, "tensor_residency")?;
    require_u64_eq(residency, "schema", 1)?;
    require_string_eq(residency, "scope", "qwen_one_token_strict_cuda")?;
    require_string_eq(residency, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_bool_eq(residency, "residency_accounting_recorded", true)?;
    require_bool_eq(residency, "kv_cache_policy_recorded", true)?;
    require_bool_eq(residency, "sampling_policy_recorded", true)?;
    require_bool_eq(residency, "per_token_weight_upload", false)?;
    require_bool_eq(residency, "fallback_used", false)?;
    require_bool_eq(residency, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(residency, "persistent_session_residency_claimed", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    let weights_uploaded_once = object_field(residency, "weights_uploaded_once")?
        .as_bool()
        .ok_or_else(|| anyhow!("field `weights_uploaded_once` must be a bool"))?;
    let weights_resident = object_field(residency, "weights_resident_on_cuda")?
        .as_bool()
        .ok_or_else(|| anyhow!("field `weights_resident_on_cuda` must be a bool"))?;
    if !weights_uploaded_once && !weights_resident {
        return Err(anyhow!(
            "tensor_residency must record either uploaded-once weights or CUDA-resident weights"
        ));
    }
    let transfer = object_field(residency, "transfer_accounting")?;
    require_string_eq(transfer, "status", "measured")?;
    require_u64_eq(transfer, "host_to_device_bytes", stats_h2d)?;
    require_u64_eq(transfer, "device_to_host_bytes", stats_d2h)?;
    require_u64_eq(transfer, "kernel_invocations", stats_invocations)?;
    require_u64_eq(transfer, "kernel_launches", stats_launches)?;
    validate_dense_qwen_transfer_timing(timing, transfer)?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_norm_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_rope_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_score_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_softmax_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_v_mix_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_mlp_activation_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_cpu_reference_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_cuda_integrated_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_all_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_model_boundary_fixtures_claimed", true)?;
    require_bool_eq(claim_boundary, "kv_cache_policy_claimed", true)?;
    require_bool_eq(claim_boundary, "sampling_policy_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "server_ready_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

/// Validate dense Qwen short-decode strict CUDA runtime proof evidence.
///
/// This artifact proves a bounded deterministic greedy short decode through the
/// dense regular-LLM CUDA route. It must consume the one-token proof and earlier
/// prerequisite receipts, reject hidden CPU fallback, and keep chat/server,
/// speedup, full-residency, and BitNet packed I2_S/QK256 proof claims false.
pub fn validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(
    receipt: &Value,
) -> Result<()> {
    validate_cuda_receipt_common(
        receipt,
        DENSE_GGUF_QWEN_SHORT_DECODE_STRICT_CUDA_PROOF_ARTIFACT_KIND,
        "dense_gguf_qwen_short_decode_strict_cuda_proof_recorded",
    )?;
    require_bool_eq(receipt, "speedup_claim", false)?;

    let model = object_field(receipt, "model")?;
    require_string_eq(model, "model_family", "qwen")?;
    require_verified_dense_qwen_runtime_model(model)?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "dense_qwen_short_decode_strict_cuda")?;
    require_string_non_empty(execution_path, "quantization_family")?;
    reject_bitnet_packed_marker(
        required_string(execution_path, "quantization_family")?,
        "execution_path.quantization_family",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    validate_dense_one_layer_gap_execution_plan(receipt)?;

    let prerequisites = object_field(receipt, "prerequisite_receipts")?;
    require_u64_eq(prerequisites, "schema", 1)?;
    require_string_eq(
        prerequisites,
        "all_layer_execution_plan_artifact_kind",
        DENSE_GGUF_ALL_LAYER_EXECUTION_PLAN_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "all_layer_execution_plan_receipt_sha256")?;
    require_string_eq(
        prerequisites,
        "model_boundary_fixtures_artifact_kind",
        DENSE_GGUF_MODEL_BOUNDARY_FIXTURES_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "model_boundary_fixtures_receipt_sha256")?;
    require_string_eq(
        prerequisites,
        "kv_cache_policy_artifact_kind",
        DENSE_GGUF_KV_CACHE_POLICY_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "kv_cache_policy_receipt_sha256")?;
    require_string_eq(
        prerequisites,
        "sampling_policy_artifact_kind",
        DENSE_GGUF_SAMPLING_POLICY_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "sampling_policy_receipt_sha256")?;
    require_string_eq(
        prerequisites,
        "one_token_proof_artifact_kind",
        DENSE_GGUF_QWEN_ONE_TOKEN_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "one_token_proof_receipt_sha256")?;
    require_bool_eq(prerequisites, "all_required_receipts_verified", true)?;
    require_bool_eq(prerequisites, "sampling_policy_claimed", true)?;
    require_bool_eq(prerequisites, "kv_cache_policy_claimed", true)?;
    require_bool_eq(prerequisites, "model_boundary_fixtures_claimed", true)?;
    require_bool_eq(prerequisites, "all_layer_execution_plan_claimed", true)?;
    require_bool_eq(prerequisites, "one_token_proof_claimed", true)?;

    let authority = object_field(receipt, "tokenizer_prompt_authority")?;
    require_u64_eq(authority, "schema", 1)?;
    require_string_eq(authority, "tokenizer_authority", "contract_authoritative")?;
    require_string_eq(authority, "prompt_authority", "contract_authoritative")?;
    require_string_non_empty(authority, "prompt_template")?;
    require_string_non_empty(authority, "bos_policy")?;
    require_bool_eq(authority, "deterministic_prompt", true)?;
    require_positive_u64(authority, "prompt_token_count")?;
    require_sha256(authority, "prompt_token_ids_sha256")?;
    require_sha256(authority, "rendered_prompt_sha256")?;
    let authority_prompt_token_ids_sha256 = required_string(authority, "prompt_token_ids_sha256")?;

    let proof = object_field(receipt, "short_decode_proof")?;
    require_u64_eq(proof, "schema", 1)?;
    let proof_scope = required_string(proof, "proof_scope")?;
    if proof_scope != "qwen_strict_short_decode_greedy"
        && proof_scope != "qwen3_strict_short_decode_32_greedy"
        && proof_scope != "qwen3_strict_warm_decode_128_greedy"
    {
        return Err(anyhow!(
            "short_decode_proof.proof_scope must be a governed Qwen short/warm decode scope"
        ));
    }
    require_string_eq(proof, "model_family", "qwen")?;
    let requested = required_u64(proof, "requested_new_tokens")?;
    let model_id = required_string(model, "id")?;
    let profile_id = proof.get("profile_id").and_then(Value::as_str).unwrap_or("short_decode");
    let valid_requested = if (5..=16).contains(&requested) {
        profile_id == "short_decode" || profile_id == "short_decode_8"
    } else {
        model_id == QWEN3_06B_INSTRUCT_Q8_0_MODEL_ID
            && ((requested == 32 && profile_id == "qwen3_short_decode_32")
                || (requested == 128 && profile_id == "qwen3_warm_decode_128"))
    };
    if !valid_requested {
        return Err(anyhow!(
            "short_decode_proof.requested_new_tokens must be 5-16 for standard short decode, 32 for qwen3_short_decode_32, or 128 for qwen3_warm_decode_128"
        ));
    }
    require_u64_eq(proof, "generated_tokens_count", requested)?;
    require_string_eq(proof, "generation_policy", "greedy")?;
    require_bool_eq(proof, "deterministic", true)?;
    require_bool_eq(proof, "fallback_used", false)?;
    require_string_eq(proof, "cpu_reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(proof, "cuda_target_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_positive_u64(proof, "prompt_token_count")?;
    require_sha256(proof, "prompt_token_ids_sha256")?;
    let proof_prompt_token_ids_sha256 = required_string(proof, "prompt_token_ids_sha256")?;
    if proof_prompt_token_ids_sha256 != authority_prompt_token_ids_sha256 {
        return Err(anyhow!(
            "short_decode_proof.prompt_token_ids_sha256 must match tokenizer_prompt_authority.prompt_token_ids_sha256"
        ));
    }
    require_sha256(proof, "cpu_generated_token_ids_sha256")?;
    require_sha256(proof, "cuda_generated_token_ids_sha256")?;
    let cpu_generated_sha = required_string(proof, "cpu_generated_token_ids_sha256")?;
    let cuda_generated_sha = required_string(proof, "cuda_generated_token_ids_sha256")?;
    if cpu_generated_sha != cuda_generated_sha {
        return Err(anyhow!(
            "short_decode_proof.cpu_generated_token_ids_sha256 must match cuda_generated_token_ids_sha256"
        ));
    }
    require_sha256(proof, "cpu_logits_top_k_steps_sha256")?;
    require_sha256(proof, "cuda_logits_top_k_steps_sha256")?;
    require_bool_eq(proof, "top_k_evidence_recorded", true)?;
    require_bool_eq(proof, "top_k_compared", true)?;
    let top_k_all_match = object_field(proof, "top_k_all_match")?
        .as_bool()
        .ok_or_else(|| anyhow!("field `top_k_all_match` must be a bool"))?;
    match (top_k_all_match, proof.get("first_top_k_divergence_index")) {
        (true, Some(value)) if !value.is_null() => {
            return Err(anyhow!(
                "short_decode_proof.first_top_k_divergence_index must be null when top_k_all_match is true"
            ));
        }
        (false, Some(value)) => {
            value.as_u64().ok_or_else(|| {
                anyhow!(
                    "short_decode_proof.first_top_k_divergence_index must be an unsigned integer when top_k_all_match is false"
                )
            })?;
        }
        (false, None) => {
            return Err(anyhow!(
                "short_decode_proof.first_top_k_divergence_index is required when top_k_all_match is false"
            ));
        }
        (true, Some(_)) => {}
        (true, None) => {}
    }
    require_bool_eq(proof, "generated_token_ids_match", true)?;
    if proof.get("first_token_divergence_index").is_some_and(|value| !value.is_null()) {
        return Err(anyhow!(
            "short_decode_proof.first_token_divergence_index must be null for a passing proof"
        ));
    }
    let cpu_tokens = array_field(proof, "cpu_generated_token_ids")?;
    let cuda_tokens = array_field(proof, "cuda_generated_token_ids")?;
    if cpu_tokens.len() != requested as usize || cuda_tokens.len() != requested as usize {
        return Err(anyhow!(
            "short_decode_proof generated token arrays must match generated_tokens_count"
        ));
    }
    if cpu_tokens != cuda_tokens {
        return Err(anyhow!(
            "short_decode_proof cpu_generated_token_ids must match cuda_generated_token_ids"
        ));
    }
    let steps = array_field(proof, "steps")?;
    if steps.len() != requested as usize {
        return Err(anyhow!("short_decode_proof.steps length must match generated_tokens_count"));
    }
    let reduced_cuda_transfer = dense_qwen_reduced_logits_transfer_requested(receipt);
    for (idx, step) in steps.iter().enumerate() {
        require_u64_eq(step, "index", idx as u64)?;
        let cpu_token = required_u64(step, "cpu_selected_token_id")?;
        let cuda_token = required_u64(step, "cuda_selected_token_id")?;
        if cpu_token != cuda_token {
            return Err(anyhow!("short_decode_proof step {idx} selected token mismatch"));
        }
        require_bool_eq(step, "selected_token_match", true)?;
        require_sha256(step, "cpu_logits_top_k_sha256")?;
        require_sha256(step, "cuda_logits_top_k_sha256")?;
        validate_dense_qwen_step_logits_sha256(step, reduced_cuda_transfer)?;
        object_field(step, "top_k_match")?
            .as_bool()
            .ok_or_else(|| anyhow!("field `top_k_match` must be a bool"))?;
        let step_timing = object_field(step, "cuda_step_timing")?;
        require_non_negative_number(step_timing, "logits_download_ms")?;
        require_non_negative_number(step, "top_k_max_abs_error")?;
        require_non_negative_number(step, "top_k_mean_abs_error")?;
    }
    require_string_non_empty(proof, "decoded_text")?;
    require_bool_eq(proof, "qwen_one_token_cuda_claimed", true)?;
    require_bool_eq(proof, "qwen_short_decode_cuda_claimed", true)?;
    require_bool_eq(proof, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(proof, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(proof, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(proof, "speedup_claim", false)?;
    require_bool_eq(proof, "server_ready_claimed", false)?;
    require_bool_eq(proof, "full_cuda_residency_claimed", false)?;

    let quality = object_field(receipt, "quality_gate")?;
    require_u64_eq(quality, "schema", 1)?;
    require_string_eq(quality, "gate", "qwen_short_decode_cuda_parity")?;
    require_bool_eq(quality, "passed", true)?;
    require_bool_eq(quality, "answer_ready_claimed", false)?;
    require_bool_eq(quality, "short_decode_claimed", true)?;
    require_bool_eq(quality, "chat_claimed", false)?;

    let stats = array_field(receipt, "kernel_stats")?;
    if stats.is_empty() {
        return Err(anyhow!("kernel_stats must contain dense CUDA short-decode entries"));
    }
    let mut stats_h2d = 0_u64;
    let mut stats_d2h = 0_u64;
    let mut stats_invocations = 0_u64;
    let mut stats_launches = 0_u64;
    for stat in stats {
        require_string_non_empty(stat, "kernel_id")?;
        reject_bitnet_packed_marker(required_string(stat, "kernel_id")?, "kernel_stats.kernel_id")?;
        require_positive_u64(stat, "invocations")?;
        require_u64_eq(stat, "fallback_invocations", 0)?;
        require_u64_eq(stat, "cpu_fallback_invocations", 0)?;
        required_u64(stat, "host_to_device_bytes")?;
        required_u64(stat, "device_to_host_bytes")?;
        require_positive_u64(stat, "kernel_launches")?;
        require_optional_non_negative_number(stat, "kernel_time_ms")?;
        stats_h2d += required_u64(stat, "host_to_device_bytes")?;
        stats_d2h += required_u64(stat, "device_to_host_bytes")?;
        stats_invocations += required_u64(stat, "invocations")?;
        stats_launches += required_u64(stat, "kernel_launches")?;
    }

    let kernel_coverage = object_field(receipt, "kernel_coverage")?;
    require_u64_eq(kernel_coverage, "schema", 1)?;
    require_string_eq(kernel_coverage, "route", DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND)?;
    require_bool_eq(kernel_coverage, "all_required_dense_kernels_executed", true)?;
    require_u64_eq(kernel_coverage, "bitnet_qk256_kernel_invocations", 0)?;
    require_u64_eq(kernel_coverage, "cpu_fallback_kernel_invocations", 0)?;
    require_u64_eq(kernel_coverage, "dense_kernel_invocations", stats_invocations)?;
    require_u64_eq(kernel_coverage, "dense_kernel_launches", stats_launches)?;
    require_bool_eq(kernel_coverage, "fallback_used", false)?;
    let kernels = array_field(kernel_coverage, "kernels_executed")?;
    if kernels.is_empty() {
        return Err(anyhow!("kernel_coverage.kernels_executed must not be empty"));
    }
    for kernel in kernels {
        let kernel = kernel
            .as_str()
            .ok_or_else(|| anyhow!("kernel_coverage.kernels_executed entries must be strings"))?;
        reject_bitnet_packed_marker(kernel, "kernel_coverage.kernels_executed")?;
    }

    let timing = object_field(receipt, "timing")?;
    require_non_negative_number(timing, "total_ms")?;
    require_non_negative_number(timing, "first_token_ms")?;
    require_non_negative_number(timing, "decode_total_ms")?;
    require_non_negative_number(timing, "logits_download_ms_total")?;
    require_non_negative_number(timing, "kernel_time_ms")?;
    require_u64_eq(timing, "generated_tokens_count", requested)?;
    require_u64_eq(timing, "host_to_device_bytes", stats_h2d)?;
    require_u64_eq(timing, "device_to_host_bytes", stats_d2h)?;
    require_u64_eq(timing, "kernel_invocations", stats_invocations)?;
    require_u64_eq(timing, "kernel_launches", stats_launches)?;

    let residency = object_field(receipt, "tensor_residency")?;
    require_u64_eq(residency, "schema", 1)?;
    require_string_eq(residency, "scope", "qwen_short_decode_strict_cuda")?;
    require_string_eq(residency, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_bool_eq(residency, "residency_accounting_recorded", true)?;
    require_bool_eq(residency, "kv_cache_policy_recorded", true)?;
    require_bool_eq(residency, "sampling_policy_recorded", true)?;
    require_bool_eq(residency, "per_token_weight_upload", false)?;
    require_bool_eq(residency, "fallback_used", false)?;
    require_bool_eq(residency, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(residency, "persistent_session_residency_claimed", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    let weights_uploaded_once = object_field(residency, "weights_uploaded_once")?
        .as_bool()
        .ok_or_else(|| anyhow!("field `weights_uploaded_once` must be a bool"))?;
    let weights_resident = object_field(residency, "weights_resident_on_cuda")?
        .as_bool()
        .ok_or_else(|| anyhow!("field `weights_resident_on_cuda` must be a bool"))?;
    if !weights_uploaded_once && !weights_resident {
        return Err(anyhow!(
            "tensor_residency must record either uploaded-once weights or CUDA-resident weights"
        ));
    }
    let transfer = object_field(residency, "transfer_accounting")?;
    require_string_eq(transfer, "status", "measured")?;
    require_u64_eq(transfer, "host_to_device_bytes", stats_h2d)?;
    require_u64_eq(transfer, "device_to_host_bytes", stats_d2h)?;
    require_u64_eq(transfer, "kernel_invocations", stats_invocations)?;
    require_u64_eq(transfer, "kernel_launches", stats_launches)?;
    validate_dense_qwen_transfer_timing(timing, transfer)?;
    validate_dense_qwen_logits_transfer_reduction(receipt, stats_d2h, requested)?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_norm_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_rope_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_score_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_softmax_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_v_mix_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_mlp_activation_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_cpu_reference_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_cuda_integrated_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_all_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_model_boundary_fixtures_claimed", true)?;
    require_bool_eq(claim_boundary, "kv_cache_policy_claimed", true)?;
    require_bool_eq(claim_boundary, "sampling_policy_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "server_ready_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

/// Validate Qwen3 warm-context 128-token strict CUDA decode source-capture evidence.
///
/// This validator is intentionally Qwen3-only and profile-specific. It reuses
/// the single-decode proof invariants while requiring an explicit warm-context
/// proof block so repeated-comparator captures cannot pass as ambiguous
/// short-decode receipts.
pub fn validate_dense_gguf_qwen_warm_decode_strict_cuda_proof_receipt_json(
    receipt: &Value,
) -> Result<()> {
    validate_cuda_receipt_common(
        receipt,
        DENSE_GGUF_QWEN_WARM_DECODE_STRICT_CUDA_PROOF_ARTIFACT_KIND,
        "dense_gguf_qwen_warm_decode_strict_cuda_proof_recorded",
    )?;

    let model = object_field(receipt, "model")?;
    require_string_eq(model, "model_family", "qwen")?;
    require_string_eq(model, "id", QWEN3_06B_INSTRUCT_Q8_0_MODEL_ID)?;
    require_string_eq(model, "file", QWEN3_06B_INSTRUCT_Q8_0_MODEL_FILE)?;
    require_string_eq(model, "sha256", QWEN3_06B_INSTRUCT_Q8_0_MODEL_SHA256)?;
    require_string_eq(model, "architecture", "qwen3")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "kernel_family", "dense_qwen_warm_decode_strict_cuda")?;
    require_string_eq(
        execution_path,
        "quantization_family",
        "dense_gguf_q8_0_f16_qwen_warm_decode_contract",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    let proof = object_field(receipt, "warm_decode_proof")?;
    require_u64_eq(proof, "schema", 1)?;
    require_string_eq(proof, "proof_scope", "qwen3_strict_warm_decode_128_greedy")?;
    require_string_eq(proof, "profile_id", "qwen3_warm_decode_128")?;
    require_u64_eq(proof, "requested_new_tokens", 128)?;
    require_u64_eq(proof, "generated_tokens_count", 128)?;
    require_bool_eq(proof, "warm_context_reused", true)?;
    require_bool_eq(proof, "decode_started_from_prefilled_context", true)?;
    require_positive_u64(proof, "warm_context_prompt_token_count")?;
    require_bool_eq(proof, "fallback_used", false)?;
    require_bool_eq(proof, "generated_token_ids_match", true)?;
    require_bool_eq(proof, "top_k_compared", true)?;
    require_bool_eq(proof, "qwen_warm_decode_cuda_claimed", true)?;
    require_bool_eq(proof, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(proof, "server_ready_claimed", false)?;
    require_bool_eq(proof, "speedup_claim", false)?;
    require_bool_eq(proof, "full_cuda_residency_claimed", false)?;

    let warm_context = object_field(receipt, "warm_context_proof")?;
    require_u64_eq(warm_context, "schema", 1)?;
    require_string_eq(warm_context, "proof_scope", "qwen3_decode_128_from_warm_context")?;
    require_string_eq(warm_context, "profile_id", "decode_128_from_warm_context")?;
    require_bool_eq(warm_context, "warm_context_reused", true)?;
    require_bool_eq(warm_context, "decode_started_from_prefilled_context", true)?;
    require_positive_u64(warm_context, "warm_context_prompt_token_count")?;
    require_sha256(warm_context, "prompt_token_ids_sha256")?;
    require_sha256(warm_context, "rendered_prompt_sha256")?;
    require_u64_eq(warm_context, "requested_new_tokens", 128)?;
    require_u64_eq(warm_context, "generated_tokens_count", 128)?;
    require_bool_eq(warm_context, "model_loaded_once", true)?;
    require_bool_eq(warm_context, "cuda_context_initialized_once", true)?;
    require_bool_eq(warm_context, "weights_uploaded_once", true)?;
    require_bool_eq(warm_context, "per_request_model_load", false)?;
    require_bool_eq(warm_context, "fallback_used", false)?;
    require_bool_eq(warm_context, "speedup_claim", false)?;
    require_bool_eq(warm_context, "server_ready_claimed", false)?;
    require_bool_eq(warm_context, "full_cuda_residency_claimed", false)?;

    let lifecycle = object_field(receipt, "session_lifecycle")?;
    require_u64_eq(lifecycle, "schema", 1)?;
    require_string_eq(lifecycle, "proof_scope", "qwen3_warm_decode_strict_cuda")?;
    require_bool_eq(lifecycle, "model_loaded_once", true)?;
    require_bool_eq(lifecycle, "tokenizer_loaded_once", true)?;
    require_bool_eq(lifecycle, "cuda_context_initialized_once", true)?;
    require_bool_eq(lifecycle, "cuda_context_once", true)?;
    require_bool_eq(lifecycle, "weights_uploaded_once", true)?;
    require_bool_eq(lifecycle, "per_request_model_load", false)?;
    require_bool_eq(lifecycle, "per_token_weight_upload", false)?;
    require_bool_eq(lifecycle, "workspace_reused", true)?;
    require_bool_eq(lifecycle, "runtime_buffers_reused", true)?;
    require_bool_eq(lifecycle, "warm_context_reused", true)?;
    require_bool_eq(lifecycle, "decode_started_from_prefilled_context", true)?;
    require_bool_eq(lifecycle, "fallback_used", false)?;
    require_bool_eq(lifecycle, "scoped_warm_context_residency_claimed", true)?;
    require_bool_eq(lifecycle, "persistent_session_residency_claimed", false)?;
    require_bool_eq(lifecycle, "full_cuda_residency_claimed", false)?;

    let quality = object_field(receipt, "quality_gate")?;
    require_string_eq(quality, "gate", "qwen_warm_decode_cuda_parity")?;
    require_bool_eq(quality, "passed", true)?;
    require_bool_eq(quality, "warm_context_decode_claimed", true)?;
    require_bool_eq(quality, "ask_claimed", false)?;
    require_bool_eq(quality, "chat_claimed", false)?;
    require_bool_eq(quality, "server_ready_claimed", false)?;

    let residency = object_field(receipt, "tensor_residency")?;
    require_string_eq(residency, "scope", "qwen_warm_decode_strict_cuda")?;
    require_bool_eq(residency, "warm_context_reused", true)?;
    require_bool_eq(residency, "scoped_warm_context_residency_claimed", true)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "qwen_warm_decode_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "server_ready_claimed", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;

    let mut surrogate = receipt.clone();
    surrogate["artifact_kind"] =
        Value::String(DENSE_GGUF_QWEN_SHORT_DECODE_STRICT_CUDA_PROOF_ARTIFACT_KIND.to_owned());
    surrogate["claim"] =
        Value::String("dense_gguf_qwen_short_decode_strict_cuda_proof_recorded".to_owned());
    surrogate["execution_path"]["kernel_family"] =
        Value::String("dense_qwen_short_decode_strict_cuda".to_owned());
    surrogate["execution_path"]["quantization_family"] =
        Value::String("dense_gguf_q8_0_f16_qwen_short_decode_contract".to_owned());
    surrogate["execution_plan"]["quantization"] =
        Value::String("dense_gguf_q8_0_f16_qwen_short_decode_contract".to_owned());
    let warm_decode_proof = surrogate
        .as_object_mut()
        .and_then(|object| object.remove("warm_decode_proof"))
        .ok_or_else(|| anyhow!("warm_decode_proof must be present"))?;
    surrogate["short_decode_proof"] = warm_decode_proof;
    surrogate["quality_gate"] = Value::Object(Map::from_iter([
        ("schema".to_owned(), Value::from(1_u64)),
        ("gate".to_owned(), Value::String("qwen_short_decode_cuda_parity".to_owned())),
        ("passed".to_owned(), Value::Bool(true)),
        ("answer_ready_claimed".to_owned(), Value::Bool(false)),
        ("short_decode_claimed".to_owned(), Value::Bool(true)),
        ("chat_claimed".to_owned(), Value::Bool(false)),
    ]));
    surrogate["tensor_residency"]["scope"] =
        Value::String("qwen_short_decode_strict_cuda".to_owned());
    validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&surrogate)?;
    Ok(())
}

/// Validate dense Qwen warm-session strict CUDA runtime proof evidence.
///
/// This artifact proves a bounded deterministic multi-turn warm session through
/// the dense regular-LLM CUDA route. It must consume the short-decode proof and
/// earlier prerequisite receipts, reject hidden CPU fallback, and keep ask/chat,
/// server, speedup, full-residency, and BitNet packed I2_S/QK256 proof claims
/// false.
pub fn validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(
    receipt: &Value,
) -> Result<()> {
    validate_cuda_receipt_common(
        receipt,
        DENSE_GGUF_QWEN_WARM_SESSION_STRICT_CUDA_PROOF_ARTIFACT_KIND,
        "dense_gguf_qwen_warm_session_strict_cuda_proof_recorded",
    )?;
    require_bool_eq(receipt, "speedup_claim", false)?;

    let model = object_field(receipt, "model")?;
    require_string_eq(model, "model_family", "qwen")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_verified_dense_qwen_runtime_model(model)?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "dense_qwen_warm_session_strict_cuda")?;
    require_string_non_empty(execution_path, "quantization_family")?;
    reject_bitnet_packed_marker(
        required_string(execution_path, "quantization_family")?,
        "execution_path.quantization_family",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    validate_dense_one_layer_gap_execution_plan(receipt)?;

    let prerequisites = object_field(receipt, "prerequisite_receipts")?;
    require_u64_eq(prerequisites, "schema", 1)?;
    require_string_eq(
        prerequisites,
        "all_layer_execution_plan_artifact_kind",
        DENSE_GGUF_ALL_LAYER_EXECUTION_PLAN_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "all_layer_execution_plan_receipt_sha256")?;
    require_string_eq(
        prerequisites,
        "model_boundary_fixtures_artifact_kind",
        DENSE_GGUF_MODEL_BOUNDARY_FIXTURES_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "model_boundary_fixtures_receipt_sha256")?;
    require_string_eq(
        prerequisites,
        "kv_cache_policy_artifact_kind",
        DENSE_GGUF_KV_CACHE_POLICY_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "kv_cache_policy_receipt_sha256")?;
    require_string_eq(
        prerequisites,
        "sampling_policy_artifact_kind",
        DENSE_GGUF_SAMPLING_POLICY_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "sampling_policy_receipt_sha256")?;
    require_string_eq(
        prerequisites,
        "one_token_proof_artifact_kind",
        DENSE_GGUF_QWEN_ONE_TOKEN_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "one_token_proof_receipt_sha256")?;
    require_string_eq(
        prerequisites,
        "short_decode_proof_artifact_kind",
        DENSE_GGUF_QWEN_SHORT_DECODE_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "short_decode_proof_receipt_sha256")?;
    require_bool_eq(prerequisites, "all_required_receipts_verified", true)?;
    require_bool_eq(prerequisites, "all_layer_execution_plan_claimed", true)?;
    require_bool_eq(prerequisites, "model_boundary_fixtures_claimed", true)?;
    require_bool_eq(prerequisites, "kv_cache_policy_claimed", true)?;
    require_bool_eq(prerequisites, "sampling_policy_claimed", true)?;
    require_bool_eq(prerequisites, "one_token_proof_claimed", true)?;
    require_bool_eq(prerequisites, "short_decode_proof_claimed", true)?;

    let authority = object_field(receipt, "tokenizer_prompt_authority")?;
    require_u64_eq(authority, "schema", 1)?;
    require_string_eq(authority, "tokenizer_authority", "contract_authoritative")?;
    require_string_eq(authority, "prompt_authority", "contract_authoritative")?;
    require_string_non_empty(authority, "prompt_template")?;
    require_string_non_empty(authority, "bos_policy")?;
    require_bool_eq(authority, "deterministic_prompt", true)?;
    let turns_count = required_u64(authority, "turns_count")?;
    if !(2..=4).contains(&turns_count) {
        return Err(anyhow!("tokenizer_prompt_authority.turns_count must be between 2 and 4"));
    }
    require_positive_u64(authority, "prompt_token_count_total")?;
    require_sha256(authority, "prompt_token_ids_sha256")?;
    require_sha256(authority, "rendered_prompt_sha256")?;
    let authority_turns = array_field(authority, "turns")?;
    if authority_turns.len() != turns_count as usize {
        return Err(anyhow!("tokenizer_prompt_authority.turns length must match turns_count"));
    }
    for (idx, turn) in authority_turns.iter().enumerate() {
        require_u64_eq(turn, "index", idx as u64)?;
        require_positive_u64(turn, "prompt_token_count")?;
        require_sha256(turn, "prompt_token_ids_sha256")?;
        require_sha256(turn, "rendered_prompt_sha256")?;
        required_u64(turn, "rendered_prompt_bytes")?;
    }

    let lifecycle = object_field(receipt, "session_lifecycle")?;
    require_u64_eq(lifecycle, "schema", 1)?;
    require_string_eq(lifecycle, "proof_scope", "qwen_warm_session_strict_cuda")?;
    require_u64_eq(lifecycle, "turns_count", turns_count)?;
    require_bool_eq(lifecycle, "model_loaded_once", true)?;
    require_bool_eq(lifecycle, "tokenizer_loaded_once", true)?;
    require_bool_eq(lifecycle, "cuda_context_initialized_once", true)?;
    require_bool_alias_eq(
        lifecycle,
        &["cuda_context_once", "cuda_context_initialized_once"],
        true,
        "cuda_context_once",
    )?;
    require_bool_eq(lifecycle, "weights_uploaded_once", true)?;
    require_bool_alias_eq(
        lifecycle,
        &["per_request_model_load", "per_turn_weight_upload"],
        false,
        "per_request_model_load",
    )?;
    require_bool_eq(lifecycle, "per_turn_weight_upload", false)?;
    require_bool_eq(lifecycle, "runtime_buffers_reused", true)?;
    require_bool_alias_eq(
        lifecycle,
        &["workspace_reused", "runtime_buffers_reused"],
        true,
        "workspace_reused",
    )?;
    require_bool_eq(lifecycle, "kv_cache_policy_recorded", true)?;
    require_bool_eq(lifecycle, "kv_cache_reinitialized_per_turn", true)?;
    require_bool_eq(lifecycle, "sampling_policy_recorded", true)?;
    require_bool_eq(lifecycle, "fallback_used", false)?;
    require_bool_eq(lifecycle, "scoped_warm_session_residency_claimed", true)?;
    require_bool_eq(lifecycle, "persistent_session_residency_claimed", false)?;
    require_bool_eq(lifecycle, "full_cuda_residency_claimed", false)?;

    let proof = object_field(receipt, "warm_session_proof")?;
    require_u64_eq(proof, "schema", 1)?;
    require_string_eq(proof, "proof_scope", "qwen_strict_warm_session_greedy")?;
    require_string_eq(proof, "model_family", "qwen")?;
    require_u64_eq(proof, "turns_count", turns_count)?;
    let requested = required_u64(proof, "requested_new_tokens_per_turn")?;
    if !(5..=16).contains(&requested) {
        return Err(anyhow!(
            "warm_session_proof.requested_new_tokens_per_turn must be between 5 and 16"
        ));
    }
    require_u64_eq(proof, "generated_tokens_total", turns_count * requested)?;
    require_string_eq(proof, "generation_policy", "greedy")?;
    require_bool_eq(proof, "deterministic", true)?;
    require_bool_eq(proof, "fallback_used", false)?;
    require_string_eq(proof, "cpu_reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(proof, "cuda_target_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_sha256(proof, "cpu_generated_token_ids_sha256")?;
    require_sha256(proof, "cuda_generated_token_ids_sha256")?;
    if required_string(proof, "cpu_generated_token_ids_sha256")?
        != required_string(proof, "cuda_generated_token_ids_sha256")?
    {
        return Err(anyhow!(
            "warm_session_proof.cpu_generated_token_ids_sha256 must match cuda_generated_token_ids_sha256"
        ));
    }
    require_bool_eq(proof, "generated_token_ids_match", true)?;
    require_null(proof, "first_token_divergence")?;
    require_sha256(proof, "cuda_logits_top_k_session_sha256")?;
    require_bool_eq(proof, "top_k_evidence_recorded", true)?;
    require_bool_eq(proof, "top_k_compared", true)?;
    let top_k_all_match = object_field(proof, "top_k_all_match")?
        .as_bool()
        .ok_or_else(|| anyhow!("field `top_k_all_match` must be a bool"))?;
    if top_k_all_match {
        require_null(proof, "first_top_k_divergence")?;
    }
    require_non_negative_number(proof, "top_k_max_abs_error")?;
    require_non_negative_number(proof, "top_k_mean_abs_error")?;
    let turns = array_field(proof, "turns")?;
    if turns.len() != turns_count as usize {
        return Err(anyhow!("warm_session_proof.turns length must match turns_count"));
    }
    let reduced_cuda_transfer = dense_qwen_reduced_logits_transfer_requested(receipt);
    for (turn_idx, turn) in turns.iter().enumerate() {
        require_u64_eq(turn, "index", turn_idx as u64)?;
        require_positive_u64(turn, "prompt_token_count")?;
        require_sha256(turn, "prompt_token_ids_sha256")?;
        require_sha256(turn, "rendered_prompt_sha256")?;
        require_u64_eq(turn, "requested_new_tokens", requested)?;
        require_u64_eq(turn, "generated_tokens_count", requested)?;
        require_sha256(turn, "cpu_generated_token_ids_sha256")?;
        require_sha256(turn, "cuda_generated_token_ids_sha256")?;
        if required_string(turn, "cpu_generated_token_ids_sha256")?
            != required_string(turn, "cuda_generated_token_ids_sha256")?
        {
            return Err(anyhow!(
                "warm_session_proof.turns[{turn_idx}] generated token SHA mismatch"
            ));
        }
        require_bool_eq(turn, "generated_token_ids_match", true)?;
        require_null(turn, "first_token_divergence_index")?;
        let cpu_tokens = array_field(turn, "cpu_generated_token_ids")?;
        let cuda_tokens = array_field(turn, "cuda_generated_token_ids")?;
        if cpu_tokens.len() != requested as usize || cuda_tokens.len() != requested as usize {
            return Err(anyhow!(
                "warm_session_proof.turns[{turn_idx}] generated token arrays must match generated_tokens_count"
            ));
        }
        if cpu_tokens != cuda_tokens {
            return Err(anyhow!(
                "warm_session_proof.turns[{turn_idx}] cpu_generated_token_ids must match cuda_generated_token_ids"
            ));
        }
        let steps = array_field(turn, "steps")?;
        if steps.len() != requested as usize {
            return Err(anyhow!(
                "warm_session_proof.turns[{turn_idx}].steps length must match generated_tokens_count"
            ));
        }
        for (idx, step) in steps.iter().enumerate() {
            require_u64_eq(step, "index", idx as u64)?;
            let cpu_token = required_u64(step, "cpu_selected_token_id")?;
            let cuda_token = required_u64(step, "cuda_selected_token_id")?;
            if cpu_token != cuda_token {
                return Err(anyhow!(
                    "warm_session_proof turn {turn_idx} step {idx} selected token mismatch"
                ));
            }
            require_bool_eq(step, "selected_token_match", true)?;
            require_sha256(step, "cpu_logits_top_k_sha256")?;
            require_sha256(step, "cuda_logits_top_k_sha256")?;
            validate_dense_qwen_step_logits_sha256(step, reduced_cuda_transfer)?;
            let step_timing = object_field(step, "cuda_step_timing")?;
            require_non_negative_number(step_timing, "logits_download_ms")?;
            require_non_negative_number(step, "top_k_max_abs_error")?;
            require_non_negative_number(step, "top_k_mean_abs_error")?;
        }
        require_string_non_empty(turn, "decoded_text")?;
        let turn_timing = object_field(turn, "cuda_turn_timing")?;
        require_non_negative_number(turn_timing, "logits_download_ms_total")?;
    }
    require_bool_eq(proof, "qwen_one_token_cuda_claimed", true)?;
    require_bool_eq(proof, "qwen_short_decode_cuda_claimed", true)?;
    require_bool_eq(proof, "qwen_warm_session_cuda_claimed", true)?;
    require_bool_eq(proof, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(proof, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(proof, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(proof, "speedup_claim", false)?;
    require_bool_eq(proof, "server_ready_claimed", false)?;
    require_bool_eq(proof, "full_cuda_residency_claimed", false)?;

    let quality = object_field(receipt, "quality_gate")?;
    require_u64_eq(quality, "schema", 1)?;
    require_string_eq(quality, "gate", "qwen_warm_session_cuda_parity")?;
    require_bool_eq(quality, "passed", true)?;
    require_bool_eq(quality, "answer_ready_claimed", false)?;
    require_bool_eq(quality, "short_decode_claimed", true)?;
    require_bool_eq(quality, "warm_session_claimed", true)?;
    require_bool_eq(quality, "chat_claimed", false)?;

    let stats = array_field(receipt, "kernel_stats")?;
    if stats.is_empty() {
        return Err(anyhow!("kernel_stats must contain dense CUDA warm-session entries"));
    }
    let mut stats_h2d = 0_u64;
    let mut stats_d2h = 0_u64;
    let mut stats_invocations = 0_u64;
    let mut stats_launches = 0_u64;
    for stat in stats {
        require_string_non_empty(stat, "kernel_id")?;
        reject_bitnet_packed_marker(required_string(stat, "kernel_id")?, "kernel_stats.kernel_id")?;
        require_positive_u64(stat, "invocations")?;
        require_u64_eq(stat, "fallback_invocations", 0)?;
        require_u64_eq(stat, "cpu_fallback_invocations", 0)?;
        required_u64(stat, "host_to_device_bytes")?;
        required_u64(stat, "device_to_host_bytes")?;
        require_positive_u64(stat, "kernel_launches")?;
        require_optional_non_negative_number(stat, "kernel_time_ms")?;
        stats_h2d += required_u64(stat, "host_to_device_bytes")?;
        stats_d2h += required_u64(stat, "device_to_host_bytes")?;
        stats_invocations += required_u64(stat, "invocations")?;
        stats_launches += required_u64(stat, "kernel_launches")?;
    }

    let kernel_coverage = object_field(receipt, "kernel_coverage")?;
    require_u64_eq(kernel_coverage, "schema", 1)?;
    require_string_eq(kernel_coverage, "route", DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND)?;
    require_bool_eq(kernel_coverage, "all_required_dense_kernels_executed", true)?;
    require_u64_eq(kernel_coverage, "bitnet_qk256_kernel_invocations", 0)?;
    require_u64_eq(kernel_coverage, "cpu_fallback_kernel_invocations", 0)?;
    require_u64_eq(kernel_coverage, "dense_kernel_invocations", stats_invocations)?;
    require_u64_eq(kernel_coverage, "dense_kernel_launches", stats_launches)?;
    require_bool_eq(kernel_coverage, "fallback_used", false)?;
    let kernels = array_field(kernel_coverage, "kernels_executed")?;
    if kernels.is_empty() {
        return Err(anyhow!("kernel_coverage.kernels_executed must not be empty"));
    }
    for kernel in kernels {
        let kernel = kernel
            .as_str()
            .ok_or_else(|| anyhow!("kernel_coverage.kernels_executed entries must be strings"))?;
        reject_bitnet_packed_marker(kernel, "kernel_coverage.kernels_executed")?;
    }

    let timing = object_field(receipt, "timing")?;
    require_non_negative_number(timing, "total_ms")?;
    require_non_negative_number(timing, "cpu_reference_total_ms")?;
    require_non_negative_number(timing, "cuda_context_init_ms")?;
    require_non_negative_number(timing, "tokenizer_load_ms")?;
    require_non_negative_number(timing, "model_load_ms")?;
    require_non_negative_number(timing, "cpu_reference_model_load_ms")?;
    require_non_negative_number(timing, "first_token_ms")?;
    require_non_negative_number(timing, "decode_total_ms")?;
    require_non_negative_number(timing, "logits_download_ms_total")?;
    require_non_negative_number(timing, "kernel_time_ms")?;
    require_u64_eq(timing, "host_to_device_bytes", stats_h2d)?;
    require_u64_eq(timing, "device_to_host_bytes", stats_d2h)?;
    require_u64_eq(timing, "kernel_invocations", stats_invocations)?;
    require_u64_eq(timing, "kernel_launches", stats_launches)?;
    require_u64_eq(timing, "turns_count", turns_count)?;
    require_u64_eq(timing, "generated_tokens_total", turns_count * requested)?;

    let residency = object_field(receipt, "tensor_residency")?;
    require_u64_eq(residency, "schema", 1)?;
    require_string_eq(residency, "scope", "qwen_warm_session_strict_cuda")?;
    require_string_eq(residency, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_bool_eq(residency, "residency_accounting_recorded", true)?;
    require_bool_eq(residency, "model_loaded_once", true)?;
    require_bool_eq(residency, "tokenizer_loaded_once", true)?;
    require_bool_eq(residency, "cuda_context_initialized_once", true)?;
    require_bool_alias_eq(
        residency,
        &["cuda_context_once", "cuda_context_initialized_once"],
        true,
        "cuda_context_once",
    )?;
    require_bool_eq(residency, "weights_uploaded_once", true)?;
    require_bool_eq(residency, "weights_resident_on_cuda", true)?;
    require_bool_alias_eq(
        residency,
        &["per_request_model_load", "per_turn_weight_upload"],
        false,
        "per_request_model_load",
    )?;
    require_bool_eq(residency, "per_turn_weight_upload", false)?;
    require_bool_eq(residency, "per_token_weight_upload", false)?;
    require_bool_eq(residency, "runtime_buffers_reused", true)?;
    require_bool_alias_eq(
        residency,
        &["workspace_reused", "runtime_buffers_reused"],
        true,
        "workspace_reused",
    )?;
    require_bool_eq(residency, "kv_cache_policy_recorded", true)?;
    require_bool_eq(residency, "kv_cache_reinitialized_per_turn", true)?;
    require_bool_eq(residency, "sampling_policy_recorded", true)?;
    require_bool_eq(residency, "runtime_logits_cuda_resident_before_download", true)?;
    require_bool_eq(residency, "fallback_used", false)?;
    require_bool_eq(residency, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(residency, "scoped_warm_session_residency_claimed", true)?;
    require_bool_eq(residency, "persistent_session_residency_claimed", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    let transfer = object_field(residency, "transfer_accounting")?;
    require_string_eq(transfer, "status", "measured")?;
    require_u64_eq(transfer, "host_to_device_bytes", stats_h2d)?;
    require_u64_eq(transfer, "device_to_host_bytes", stats_d2h)?;
    require_u64_eq(transfer, "kernel_invocations", stats_invocations)?;
    require_u64_eq(transfer, "kernel_launches", stats_launches)?;
    validate_dense_qwen_transfer_timing(timing, transfer)?;
    validate_dense_qwen_logits_transfer_reduction(receipt, stats_d2h, turns_count * requested)?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_norm_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_rope_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_score_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_softmax_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_v_mix_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_mlp_activation_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_cpu_reference_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_cuda_integrated_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_all_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_model_boundary_fixtures_claimed", true)?;
    require_bool_eq(claim_boundary, "kv_cache_policy_claimed", true)?;
    require_bool_eq(claim_boundary, "sampling_policy_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_warm_session_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "scoped_warm_session_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_ask_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "server_ready_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

/// Validate dense Qwen CUDA ask UX receipts.
///
/// This artifact is the first user-facing `bitnet ask --device cuda` wrapper
/// for the dense Qwen lane. It must embed a valid bounded short-decode proof,
/// record the warm-session proof prerequisite, reject hidden CPU fallback, and
/// keep chat/server, speedup, full-residency, and BitNet packed I2_S/QK256 proof
/// claims false.
pub fn validate_dense_gguf_qwen_ask_strict_cuda_proof_receipt_json(receipt: &Value) -> Result<()> {
    validate_cuda_receipt_common(
        receipt,
        DENSE_GGUF_QWEN_ASK_STRICT_CUDA_PROOF_ARTIFACT_KIND,
        "dense_gguf_qwen_ask_strict_cuda_proof_recorded",
    )?;
    require_bool_eq(receipt, "speedup_claim", false)?;

    let source = object_field(receipt, "source_short_decode_receipt")?;
    validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(source)?;

    let model = object_field(receipt, "model")?;
    require_string_eq(model, "model_family", "qwen")?;
    require_verified_dense_qwen_runtime_model(model)?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "dense_qwen_ask_strict_cuda")?;
    require_string_non_empty(execution_path, "quantization_family")?;
    reject_bitnet_packed_marker(
        required_string(execution_path, "quantization_family")?,
        "execution_path.quantization_family",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    validate_dense_one_layer_gap_execution_plan(receipt)?;

    let prerequisites = object_field(receipt, "prerequisite_receipts")?;
    require_u64_eq(prerequisites, "schema", 1)?;
    require_string_eq(
        prerequisites,
        "short_decode_proof_artifact_kind",
        DENSE_GGUF_QWEN_SHORT_DECODE_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "short_decode_proof_receipt_sha256")?;
    require_string_eq(
        prerequisites,
        "warm_session_proof_artifact_kind",
        DENSE_GGUF_QWEN_WARM_SESSION_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "warm_session_proof_receipt_sha256")?;
    require_bool_eq(prerequisites, "short_decode_proof_claimed", true)?;
    require_bool_eq(prerequisites, "warm_session_proof_claimed", true)?;
    require_bool_eq(prerequisites, "all_required_receipts_verified", true)?;

    let source_prerequisites = object_field(source, "prerequisite_receipts")?;
    for field in [
        "all_layer_execution_plan_artifact_kind",
        "all_layer_execution_plan_receipt_sha256",
        "model_boundary_fixtures_artifact_kind",
        "model_boundary_fixtures_receipt_sha256",
        "kv_cache_policy_artifact_kind",
        "kv_cache_policy_receipt_sha256",
        "sampling_policy_artifact_kind",
        "sampling_policy_receipt_sha256",
        "one_token_proof_artifact_kind",
        "one_token_proof_receipt_sha256",
    ] {
        if prerequisites.get(field) != source_prerequisites.get(field) {
            return Err(anyhow!(
                "prerequisite_receipts.{field} must match the embedded short-decode source receipt"
            ));
        }
    }

    let authority = object_field(receipt, "tokenizer_prompt_authority")?;
    let source_authority = object_field(source, "tokenizer_prompt_authority")?;
    require_u64_eq(authority, "schema", 1)?;
    require_string_eq(authority, "tokenizer_authority", "contract_authoritative")?;
    require_string_eq(authority, "prompt_authority", "contract_authoritative")?;
    require_string_non_empty(authority, "prompt_template")?;
    require_string_non_empty(authority, "bos_policy")?;
    require_bool_eq(authority, "deterministic_prompt", true)?;
    require_positive_u64(authority, "prompt_token_count")?;
    require_sha256(authority, "prompt_token_ids_sha256")?;
    require_sha256(authority, "rendered_prompt_sha256")?;
    if required_string(authority, "prompt_token_ids_sha256")?
        != required_string(source_authority, "prompt_token_ids_sha256")?
    {
        return Err(anyhow!(
            "tokenizer_prompt_authority.prompt_token_ids_sha256 must match the embedded short-decode source receipt"
        ));
    }

    let source_proof = object_field(source, "short_decode_proof")?;
    let ask = object_field(receipt, "ask_proof")?;
    require_u64_eq(ask, "schema", 1)?;
    require_string_eq(ask, "proof_scope", "qwen_strict_cuda_ask_from_short_decode")?;
    require_string_eq(ask, "model_family", "qwen")?;
    let requested = required_u64(ask, "requested_new_tokens")?;
    if !(5..=16).contains(&requested) {
        return Err(anyhow!("ask_proof.requested_new_tokens must be between 5 and 16"));
    }
    require_u64_eq(ask, "generated_tokens_count", requested)?;
    require_string_eq(ask, "generation_policy", "greedy")?;
    require_bool_eq(ask, "deterministic", true)?;
    require_bool_eq(ask, "fallback_used", false)?;
    require_string_eq(ask, "cpu_reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(ask, "cuda_target_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_non_empty(ask, "question")?;
    require_string_non_empty(ask, "answer")?;
    require_sha256(ask, "prompt_token_ids_sha256")?;
    if required_string(ask, "prompt_token_ids_sha256")?
        != required_string(source_proof, "prompt_token_ids_sha256")?
    {
        return Err(anyhow!(
            "ask_proof.prompt_token_ids_sha256 must match short_decode_proof.prompt_token_ids_sha256"
        ));
    }
    require_sha256(ask, "cpu_generated_token_ids_sha256")?;
    require_sha256(ask, "cuda_generated_token_ids_sha256")?;
    if required_string(ask, "cpu_generated_token_ids_sha256")?
        != required_string(ask, "cuda_generated_token_ids_sha256")?
    {
        return Err(anyhow!(
            "ask_proof.cpu_generated_token_ids_sha256 must match cuda_generated_token_ids_sha256"
        ));
    }
    if required_string(ask, "cpu_generated_token_ids_sha256")?
        != required_string(source_proof, "cpu_generated_token_ids_sha256")?
        || required_string(ask, "cuda_generated_token_ids_sha256")?
            != required_string(source_proof, "cuda_generated_token_ids_sha256")?
    {
        return Err(anyhow!(
            "ask_proof generated-token hashes must match the embedded short-decode source receipt"
        ));
    }
    let ask_cpu_tokens = array_field(ask, "cpu_generated_token_ids")?;
    let ask_cuda_tokens = array_field(ask, "cuda_generated_token_ids")?;
    let source_cpu_tokens = array_field(source_proof, "cpu_generated_token_ids")?;
    let source_cuda_tokens = array_field(source_proof, "cuda_generated_token_ids")?;
    if ask_cpu_tokens != source_cpu_tokens || ask_cuda_tokens != source_cuda_tokens {
        return Err(anyhow!(
            "ask_proof generated-token arrays must match the embedded short-decode source receipt"
        ));
    }
    require_bool_eq(ask, "generated_token_ids_match", true)?;
    require_null(ask, "first_token_divergence_index")?;
    require_bool_eq(ask, "top_k_evidence_recorded", true)?;
    require_bool_eq(ask, "top_k_compared", true)?;
    object_field(ask, "top_k_all_match")?
        .as_bool()
        .ok_or_else(|| anyhow!("field `top_k_all_match` must be a bool"))?;
    require_bool_eq(ask, "qwen_one_token_cuda_claimed", true)?;
    require_bool_eq(ask, "qwen_short_decode_cuda_claimed", true)?;
    require_bool_eq(ask, "qwen_warm_session_cuda_claimed", true)?;
    require_bool_eq(ask, "qwen_ask_cuda_claimed", true)?;
    require_bool_eq(ask, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(ask, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(ask, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(ask, "speedup_claim", false)?;
    require_bool_eq(ask, "server_ready_claimed", false)?;
    require_bool_eq(ask, "full_cuda_residency_claimed", false)?;

    let quality = object_field(receipt, "quality_gate")?;
    require_u64_eq(quality, "schema", 1)?;
    require_string_eq(quality, "gate", "qwen_cuda_ask_answer")?;
    require_bool_eq(quality, "passed", true)?;
    require_bool_eq(quality, "ask_claimed", true)?;
    require_bool_eq(quality, "chat_claimed", false)?;

    let stats = array_field(receipt, "kernel_stats")?;
    if stats.is_empty() {
        return Err(anyhow!("kernel_stats must contain dense CUDA ask entries"));
    }
    let mut stats_h2d = 0_u64;
    let mut stats_d2h = 0_u64;
    let mut stats_invocations = 0_u64;
    let mut stats_launches = 0_u64;
    for stat in stats {
        require_string_non_empty(stat, "kernel_id")?;
        reject_bitnet_packed_marker(required_string(stat, "kernel_id")?, "kernel_stats.kernel_id")?;
        require_positive_u64(stat, "invocations")?;
        require_u64_eq(stat, "fallback_invocations", 0)?;
        require_u64_eq(stat, "cpu_fallback_invocations", 0)?;
        required_u64(stat, "host_to_device_bytes")?;
        required_u64(stat, "device_to_host_bytes")?;
        require_positive_u64(stat, "kernel_launches")?;
        require_optional_non_negative_number(stat, "kernel_time_ms")?;
        stats_h2d += required_u64(stat, "host_to_device_bytes")?;
        stats_d2h += required_u64(stat, "device_to_host_bytes")?;
        stats_invocations += required_u64(stat, "invocations")?;
        stats_launches += required_u64(stat, "kernel_launches")?;
    }

    let kernel_coverage = object_field(receipt, "kernel_coverage")?;
    require_u64_eq(kernel_coverage, "schema", 1)?;
    require_string_eq(kernel_coverage, "route", DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND)?;
    require_bool_eq(kernel_coverage, "all_required_dense_kernels_executed", true)?;
    require_u64_eq(kernel_coverage, "bitnet_qk256_kernel_invocations", 0)?;
    require_u64_eq(kernel_coverage, "cpu_fallback_kernel_invocations", 0)?;
    require_u64_eq(kernel_coverage, "dense_kernel_invocations", stats_invocations)?;
    require_u64_eq(kernel_coverage, "dense_kernel_launches", stats_launches)?;
    require_bool_eq(kernel_coverage, "fallback_used", false)?;

    let timing = object_field(receipt, "timing")?;
    require_non_negative_number(timing, "total_ms")?;
    require_non_negative_number(timing, "first_token_ms")?;
    require_non_negative_number(timing, "decode_total_ms")?;
    require_non_negative_number(timing, "kernel_time_ms")?;
    require_u64_eq(timing, "generated_tokens_count", requested)?;
    require_u64_eq(timing, "host_to_device_bytes", stats_h2d)?;
    require_u64_eq(timing, "device_to_host_bytes", stats_d2h)?;
    require_u64_eq(timing, "kernel_invocations", stats_invocations)?;
    require_u64_eq(timing, "kernel_launches", stats_launches)?;

    let residency = object_field(receipt, "tensor_residency")?;
    require_u64_eq(residency, "schema", 1)?;
    require_string_eq(residency, "scope", "qwen_ask_strict_cuda")?;
    require_string_eq(residency, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_bool_eq(residency, "residency_accounting_recorded", true)?;
    require_bool_eq(residency, "kv_cache_policy_recorded", true)?;
    require_bool_eq(residency, "sampling_policy_recorded", true)?;
    require_bool_eq(residency, "per_token_weight_upload", false)?;
    require_bool_eq(residency, "fallback_used", false)?;
    require_bool_eq(residency, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(residency, "persistent_session_residency_claimed", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    let transfer = object_field(residency, "transfer_accounting")?;
    require_string_eq(transfer, "status", "measured")?;
    require_u64_eq(transfer, "host_to_device_bytes", stats_h2d)?;
    require_u64_eq(transfer, "device_to_host_bytes", stats_d2h)?;
    require_u64_eq(transfer, "kernel_invocations", stats_invocations)?;
    require_u64_eq(transfer, "kernel_launches", stats_launches)?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_all_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_model_boundary_fixtures_claimed", true)?;
    require_bool_eq(claim_boundary, "kv_cache_policy_claimed", true)?;
    require_bool_eq(claim_boundary, "sampling_policy_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_warm_session_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_ask_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "server_ready_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

/// Validate dense Qwen CUDA chat UX receipts.
///
/// This artifact is the first user-facing `bitnet chat --device cuda` wrapper
/// for the dense Qwen lane. It must embed a valid bounded warm-session proof,
/// reject hidden CPU fallback, and keep server, speedup, full-residency, broad
/// dense GGUF inference, and BitNet packed I2_S/QK256 proof claims false.
pub fn validate_dense_gguf_qwen_chat_strict_cuda_proof_receipt_json(receipt: &Value) -> Result<()> {
    validate_cuda_receipt_common(
        receipt,
        DENSE_GGUF_QWEN_CHAT_STRICT_CUDA_PROOF_ARTIFACT_KIND,
        "dense_gguf_qwen_chat_strict_cuda_proof_recorded",
    )?;
    require_bool_eq(receipt, "speedup_claim", false)?;

    let source = object_field(receipt, "source_warm_session_receipt")?;
    validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(source)?;

    let model = object_field(receipt, "model")?;
    require_string_eq(model, "model_family", "qwen")?;
    require_verified_dense_qwen_runtime_model(model)?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "dense_qwen_chat_strict_cuda")?;
    require_string_non_empty(execution_path, "quantization_family")?;
    reject_bitnet_packed_marker(
        required_string(execution_path, "quantization_family")?,
        "execution_path.quantization_family",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    validate_dense_one_layer_gap_execution_plan(receipt)?;

    let prerequisites = object_field(receipt, "prerequisite_receipts")?;
    require_u64_eq(prerequisites, "schema", 1)?;
    require_string_eq(
        prerequisites,
        "warm_session_proof_artifact_kind",
        DENSE_GGUF_QWEN_WARM_SESSION_STRICT_CUDA_PROOF_ARTIFACT_KIND,
    )?;
    require_sha256(prerequisites, "warm_session_proof_receipt_sha256")?;
    require_bool_eq(prerequisites, "warm_session_proof_claimed", true)?;
    require_bool_eq(prerequisites, "all_required_receipts_verified", true)?;

    let source_prerequisites = object_field(source, "prerequisite_receipts")?;
    for field in [
        "all_layer_execution_plan_artifact_kind",
        "all_layer_execution_plan_receipt_sha256",
        "model_boundary_fixtures_artifact_kind",
        "model_boundary_fixtures_receipt_sha256",
        "kv_cache_policy_artifact_kind",
        "kv_cache_policy_receipt_sha256",
        "sampling_policy_artifact_kind",
        "sampling_policy_receipt_sha256",
        "one_token_proof_artifact_kind",
        "one_token_proof_receipt_sha256",
        "short_decode_proof_artifact_kind",
        "short_decode_proof_receipt_sha256",
    ] {
        if prerequisites.get(field) != source_prerequisites.get(field) {
            return Err(anyhow!(
                "prerequisite_receipts.{field} must match the embedded warm-session source receipt"
            ));
        }
    }

    let source_authority = object_field(source, "tokenizer_prompt_authority")?;
    let authority = object_field(receipt, "tokenizer_prompt_authority")?;
    require_u64_eq(authority, "schema", 1)?;
    require_string_eq(authority, "tokenizer_authority", "contract_authoritative")?;
    require_string_eq(authority, "prompt_authority", "contract_authoritative")?;
    require_string_non_empty(authority, "prompt_template")?;
    require_bool_eq(authority, "deterministic_prompt", true)?;
    require_positive_u64(authority, "turns_count")?;
    require_sha256(authority, "prompt_token_ids_sha256")?;
    require_sha256(authority, "rendered_prompt_sha256")?;
    if required_string(authority, "prompt_token_ids_sha256")?
        != required_string(source_authority, "prompt_token_ids_sha256")?
        || required_string(authority, "rendered_prompt_sha256")?
            != required_string(source_authority, "rendered_prompt_sha256")?
    {
        return Err(anyhow!(
            "tokenizer_prompt_authority prompt hashes must match the embedded warm-session source receipt"
        ));
    }

    let source_proof = object_field(source, "warm_session_proof")?;
    let chat = object_field(receipt, "chat_session")?;
    require_u64_eq(chat, "schema", 1)?;
    require_string_eq(chat, "proof_scope", "qwen_strict_cuda_chat_from_warm_session")?;
    require_string_eq(chat, "model_family", "qwen")?;
    let turns_count = required_u64(chat, "turns_count")?;
    if !(2..=4).contains(&turns_count) {
        return Err(anyhow!("chat_session.turns_count must be between 2 and 4"));
    }
    let requested = required_u64(chat, "requested_new_tokens_per_turn")?;
    if !(5..=16).contains(&requested) {
        return Err(anyhow!("chat_session.requested_new_tokens_per_turn must be between 5 and 16"));
    }
    require_u64_eq(chat, "generated_tokens_total", turns_count * requested)?;
    require_string_eq(chat, "generation_policy", "greedy")?;
    require_bool_eq(chat, "deterministic", true)?;
    require_bool_eq(chat, "fallback_used", false)?;
    require_string_eq(chat, "cpu_reference_backend", "amd-9950x3d-cpu-avx512")?;
    require_string_eq(chat, "cuda_target_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_sha256(chat, "cpu_generated_token_ids_sha256")?;
    require_sha256(chat, "cuda_generated_token_ids_sha256")?;
    if required_string(chat, "cpu_generated_token_ids_sha256")?
        != required_string(chat, "cuda_generated_token_ids_sha256")?
        || required_string(chat, "cpu_generated_token_ids_sha256")?
            != required_string(source_proof, "cpu_generated_token_ids_sha256")?
        || required_string(chat, "cuda_generated_token_ids_sha256")?
            != required_string(source_proof, "cuda_generated_token_ids_sha256")?
    {
        return Err(anyhow!(
            "chat_session generated-token hashes must match the embedded warm-session source receipt"
        ));
    }
    require_bool_eq(chat, "generated_token_ids_match", true)?;
    require_null(chat, "first_token_divergence")?;
    require_bool_eq(chat, "top_k_evidence_recorded", true)?;
    require_bool_eq(chat, "top_k_compared", true)?;
    object_field(chat, "top_k_all_match")?
        .as_bool()
        .ok_or_else(|| anyhow!("field `top_k_all_match` must be a bool"))?;
    require_bool_eq(chat, "qwen_one_token_cuda_claimed", true)?;
    require_bool_eq(chat, "qwen_short_decode_cuda_claimed", true)?;
    require_bool_eq(chat, "qwen_warm_session_cuda_claimed", true)?;
    require_bool_eq(chat, "qwen_ask_cuda_claimed", false)?;
    require_bool_eq(chat, "qwen_chat_cuda_claimed", true)?;
    require_bool_eq(chat, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(chat, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(chat, "speedup_claim", false)?;
    require_bool_eq(chat, "server_ready_claimed", false)?;
    require_bool_eq(chat, "full_cuda_residency_claimed", false)?;

    let chat_turns = array_field(chat, "turns")?;
    let source_turns = array_field(source_proof, "turns")?;
    if chat_turns.len() != turns_count as usize || source_turns.len() != turns_count as usize {
        return Err(anyhow!("chat_session.turns length must match turns_count"));
    }
    for (turn_idx, (turn, source_turn)) in chat_turns.iter().zip(source_turns.iter()).enumerate() {
        require_u64_eq(turn, "index", turn_idx as u64)?;
        require_string_non_empty(turn, "user_message")?;
        require_string_non_empty(turn, "assistant_answer")?;
        require_sha256(turn, "prompt_token_ids_sha256")?;
        if required_string(turn, "prompt_token_ids_sha256")?
            != required_string(source_turn, "prompt_token_ids_sha256")?
        {
            return Err(anyhow!(
                "chat_session.turns[{turn_idx}].prompt_token_ids_sha256 must match source turn"
            ));
        }
        if array_field(turn, "cpu_generated_token_ids")?
            != array_field(source_turn, "cpu_generated_token_ids")?
            || array_field(turn, "cuda_generated_token_ids")?
                != array_field(source_turn, "cuda_generated_token_ids")?
        {
            return Err(anyhow!(
                "chat_session.turns[{turn_idx}] generated token arrays must match source turn"
            ));
        }
        require_bool_eq(turn, "generated_token_ids_match", true)?;
        require_null(turn, "first_token_divergence_index")?;
    }

    let quality = object_field(receipt, "quality_gate")?;
    require_u64_eq(quality, "schema", 1)?;
    require_string_eq(quality, "gate", "qwen_cuda_chat_session")?;
    require_bool_eq(quality, "passed", true)?;
    require_bool_eq(quality, "chat_claimed", true)?;
    require_bool_eq(quality, "server_claimed", false)?;

    let stats = array_field(receipt, "kernel_stats")?;
    if stats.is_empty() {
        return Err(anyhow!("kernel_stats must contain dense CUDA chat entries"));
    }
    let mut stats_h2d = 0_u64;
    let mut stats_d2h = 0_u64;
    let mut stats_invocations = 0_u64;
    let mut stats_launches = 0_u64;
    for stat in stats {
        require_string_non_empty(stat, "kernel_id")?;
        reject_bitnet_packed_marker(required_string(stat, "kernel_id")?, "kernel_stats.kernel_id")?;
        require_positive_u64(stat, "invocations")?;
        require_u64_eq(stat, "fallback_invocations", 0)?;
        require_u64_eq(stat, "cpu_fallback_invocations", 0)?;
        required_u64(stat, "host_to_device_bytes")?;
        required_u64(stat, "device_to_host_bytes")?;
        require_positive_u64(stat, "kernel_launches")?;
        require_optional_non_negative_number(stat, "kernel_time_ms")?;
        stats_h2d += required_u64(stat, "host_to_device_bytes")?;
        stats_d2h += required_u64(stat, "device_to_host_bytes")?;
        stats_invocations += required_u64(stat, "invocations")?;
        stats_launches += required_u64(stat, "kernel_launches")?;
    }

    let timing = object_field(receipt, "timing")?;
    require_non_negative_number(timing, "total_ms")?;
    require_non_negative_number(timing, "first_token_ms")?;
    require_non_negative_number(timing, "decode_total_ms")?;
    require_non_negative_number(timing, "kernel_time_ms")?;
    require_u64_eq(timing, "turns_count", turns_count)?;
    require_u64_eq(timing, "generated_tokens_total", turns_count * requested)?;
    require_u64_eq(timing, "host_to_device_bytes", stats_h2d)?;
    require_u64_eq(timing, "device_to_host_bytes", stats_d2h)?;
    require_u64_eq(timing, "kernel_invocations", stats_invocations)?;
    require_u64_eq(timing, "kernel_launches", stats_launches)?;

    let residency = object_field(receipt, "tensor_residency")?;
    require_u64_eq(residency, "schema", 1)?;
    require_string_eq(residency, "scope", "qwen_chat_strict_cuda")?;
    require_string_eq(residency, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_bool_eq(residency, "residency_accounting_recorded", true)?;
    require_bool_eq(residency, "model_loaded_once", true)?;
    require_bool_eq(residency, "tokenizer_loaded_once", true)?;
    require_bool_eq(residency, "cuda_context_initialized_once", true)?;
    require_bool_eq(residency, "weights_uploaded_once", true)?;
    require_bool_eq(residency, "per_turn_weight_upload", false)?;
    require_bool_eq(residency, "per_token_weight_upload", false)?;
    require_bool_eq(residency, "runtime_buffers_reused", true)?;
    require_bool_eq(residency, "fallback_used", false)?;
    require_bool_eq(residency, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(residency, "persistent_session_residency_claimed", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    let transfer = object_field(residency, "transfer_accounting")?;
    require_string_eq(transfer, "status", "measured")?;
    require_u64_eq(transfer, "host_to_device_bytes", stats_h2d)?;
    require_u64_eq(transfer, "device_to_host_bytes", stats_d2h)?;
    require_u64_eq(transfer, "kernel_invocations", stats_invocations)?;
    require_u64_eq(transfer, "kernel_launches", stats_launches)?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_all_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_model_boundary_fixtures_claimed", true)?;
    require_bool_eq(claim_boundary, "kv_cache_policy_claimed", true)?;
    require_bool_eq(claim_boundary, "sampling_policy_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_warm_session_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "qwen_ask_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "server_ready_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

/// Validate dense GGUF one-layer CPU reference harness evidence.
///
/// This artifact records a CPU-only full layer-0 reference output. It is the
/// anchor for later integrated CUDA parity, not CUDA execution or dense GGUF
/// inference.
pub fn validate_dense_gguf_one_layer_cpu_reference_receipt_json(receipt: &Value) -> Result<()> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", DENSE_GGUF_ONE_LAYER_CPU_REFERENCE_ARTIFACT_KIND)?;
    require_string_eq(receipt, "claim", "dense_gguf_one_layer_cpu_reference_recorded")?;
    require_string_eq(receipt, "hardware_lane", "cpu-reference")?;
    require_string_eq(receipt, "requested_backend", "cpu_reference")?;
    require_string_eq(receipt, "selected_backend", "cpu_reference")?;
    require_string_eq(receipt, "runtime_api", "cpu")?;
    require_bool_eq(receipt, "fallback_used", false)?;
    require_null(receipt, "fallback_backend")?;
    require_null(receipt, "fallback_reason")?;
    require_bool_eq(receipt, "speedup_claim", false)?;
    require_null(receipt, "error")?;

    let model = object_field(receipt, "model")?;
    require_string_non_empty(model, "model_family")?;
    reject_bitnet_packed_marker(required_string(model, "model_family")?, "model.model_family")?;
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_sha256(model, "sha256")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "cpu_reference_dense_one_layer")?;
    require_string_eq(
        execution_path,
        "quantization_family",
        "dense_gguf_materialized_f32_reference",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_positive_u64(descriptor, "tensor_count")?;
    require_positive_u64(descriptor, "metadata_count")?;
    require_bool_eq(descriptor, "required_roles_present", true)?;
    require_bool_eq(descriptor, "strict_descriptor_complete", true)?;
    require_string_non_empty(descriptor, "dense_cuda_route_status")?;
    reject_bitnet_packed_marker(
        required_string(descriptor, "dense_cuda_route_status")?,
        "descriptor_coverage.dense_cuda_route_status",
    )?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;
    let quantization_families = array_field(descriptor, "quantization_families")?;
    if quantization_families.is_empty() {
        return Err(anyhow!("descriptor_coverage.quantization_families must not be empty"));
    }
    for family in quantization_families {
        let family = family.as_str().ok_or_else(|| {
            anyhow!("descriptor_coverage.quantization_families entries must be strings")
        })?;
        reject_bitnet_packed_marker(family, "descriptor_coverage.quantization_families")?;
    }

    let harness = object_field(receipt, "reference_harness")?;
    require_u64_eq(harness, "schema", 1)?;
    require_string_non_empty(harness, "fixture_id")?;
    reject_bitnet_packed_marker(
        required_string(harness, "fixture_id")?,
        "reference_harness.fixture_id",
    )?;
    require_u64_eq(harness, "layer_index", 0)?;
    require_positive_u64(harness, "seq_len")?;
    required_u64(harness, "position_offset")?;
    require_positive_u64(harness, "hidden_size")?;
    require_positive_u64(harness, "q_heads")?;
    require_positive_u64(harness, "kv_heads")?;
    require_positive_u64(harness, "heads_per_kv_group")?;
    require_positive_u64(harness, "head_dim")?;
    require_positive_u64(harness, "intermediate_size")?;
    require_positive_number(harness, "rmsnorm_eps")?;
    require_string_non_empty(harness, "epsilon_source")?;
    require_positive_number(harness, "rope_base")?;
    require_string_non_empty(harness, "rope_base_source")?;
    require_positive_number(harness, "rope_scaling_factor")?;
    require_positive_u64(harness, "deterministic_input_len")?;
    require_sha256(harness, "deterministic_input_sha256")?;
    require_positive_u64(harness, "phases_total")?;
    require_positive_u64(harness, "final_output_len")?;
    require_sha256(harness, "final_output_sha256")?;
    require_non_negative_number(harness, "final_output_max_abs")?;
    require_bool_eq(harness, "cpu_reference_only", true)?;
    require_bool_eq(harness, "cuda_execution_claimed", false)?;
    require_bool_eq(harness, "one_layer_inference_claimed", false)?;
    require_bool_eq(harness, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(harness, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(harness, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(harness, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(harness, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(harness, "speedup_claim", false)?;
    require_bool_eq(harness, "full_cuda_residency_claimed", false)?;
    require_string_eq(harness, "next_required_proof", "one_layer_cuda_integrated_parity")?;

    let phases = array_field(harness, "phases")?;
    let phases_total = required_u64(harness, "phases_total")?;
    if phases.len() != phases_total as usize {
        return Err(anyhow!("reference_harness.phases length must match phases_total"));
    }
    let mut names = BTreeSet::new();
    for (idx, phase) in phases.iter().enumerate() {
        require_u64_eq(phase, "index", idx as u64)?;
        require_string_non_empty(phase, "name")?;
        let name = required_string(phase, "name")?;
        reject_bitnet_packed_marker(name, "reference_harness.phases.name")?;
        names.insert(name.to_string());
        require_string_non_empty(phase, "role")?;
        reject_bitnet_packed_marker(
            required_string(phase, "role")?,
            "reference_harness.phases.role",
        )?;
        require_string_non_empty(phase, "op_type")?;
        require_positive_u64(phase, "output_len")?;
        require_sha256(phase, "output_sha256")?;
        require_non_negative_number(phase, "max_abs")?;
    }
    const REQUIRED_PHASES: &[&str] = &[
        "deterministic_input",
        "attention_norm",
        "attention_q",
        "attention_k",
        "attention_v",
        "rope",
        "attention_scores",
        "attention_softmax",
        "attention_v_mix",
        "attention_output",
        "first_residual",
        "ffn_norm",
        "mlp_gate",
        "mlp_up",
        "mlp_activation",
        "mlp_down",
        "second_residual",
    ];
    for required in REQUIRED_PHASES {
        if !names.contains(*required) {
            return Err(anyhow!("reference_harness.phases missing required phase `{required}`"));
        }
    }
    if phases_total != REQUIRED_PHASES.len() as u64 {
        return Err(anyhow!("reference_harness.phases_total must equal governed CPU phase count"));
    }
    let deterministic_input = phases
        .iter()
        .find(|phase| phase.get("name").and_then(Value::as_str) == Some("deterministic_input"))
        .ok_or_else(|| anyhow!("reference_harness.phases missing deterministic_input phase"))?;
    let deterministic_input_len = required_u64(harness, "deterministic_input_len")?;
    let deterministic_input_sha256 = required_string(harness, "deterministic_input_sha256")?;
    let deterministic_phase_len = required_u64(deterministic_input, "output_len")?;
    let deterministic_phase_sha256 = required_string(deterministic_input, "output_sha256")?;
    if deterministic_phase_len != deterministic_input_len {
        return Err(anyhow!(
            "reference_harness.deterministic_input_len must match deterministic_input phase output_len"
        ));
    }
    if deterministic_phase_sha256 != deterministic_input_sha256 {
        return Err(anyhow!(
            "reference_harness.deterministic_input_sha256 must match deterministic_input phase output_sha256"
        ));
    }

    let second_residual = phases
        .iter()
        .find(|phase| phase.get("name").and_then(Value::as_str) == Some("second_residual"))
        .ok_or_else(|| anyhow!("reference_harness.phases missing second_residual phase"))?;
    let final_output_len = required_u64(harness, "final_output_len")?;
    let final_output_sha256 = required_string(harness, "final_output_sha256")?;
    let final_output_max_abs = object_field(harness, "final_output_max_abs")?
        .as_f64()
        .ok_or_else(|| anyhow!("field `final_output_max_abs` must be a number"))?;
    let second_residual_len = required_u64(second_residual, "output_len")?;
    let second_residual_sha256 = required_string(second_residual, "output_sha256")?;
    let second_residual_max_abs = object_field(second_residual, "max_abs")?
        .as_f64()
        .ok_or_else(|| anyhow!("field `max_abs` must be a number"))?;
    if second_residual_len != final_output_len {
        return Err(anyhow!(
            "reference_harness.final_output_len must match second_residual phase output_len"
        ));
    }
    if second_residual_sha256 != final_output_sha256 {
        return Err(anyhow!(
            "reference_harness.final_output_sha256 must match second_residual phase output_sha256"
        ));
    }
    if second_residual_max_abs != final_output_max_abs {
        return Err(anyhow!(
            "reference_harness.final_output_max_abs must match second_residual phase max_abs"
        ));
    }

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_cpu_reference_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

/// Validate integrated dense GGUF one-layer CUDA parity evidence.
///
/// This artifact may claim only that one governed layer-0 pass matched the CPU
/// reference harness. It must still reject dense GGUF inference, Qwen token /
/// decode / chat, speedup, persistent residency, full CUDA residency, and
/// BitNet packed I2_S/QK256 proof claims.
pub fn validate_dense_gguf_one_layer_cuda_integrated_parity_receipt_json(
    receipt: &Value,
) -> Result<()> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(
        receipt,
        "artifact_kind",
        DENSE_GGUF_ONE_LAYER_CUDA_INTEGRATED_PARITY_ARTIFACT_KIND,
    )?;
    require_string_eq(receipt, "claim", "dense_gguf_one_layer_cuda_integrated_parity_recorded")?;
    require_string_eq(receipt, "machine_id", "windows-9950x3d-rtx5070ti")?;
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
    require_string_non_empty(model, "architecture")?;
    reject_bitnet_packed_marker(required_string(model, "architecture")?, "model.architecture")?;
    require_string_eq(model, "artifact_kind", "dense_gguf")?;
    require_sha256(model, "sha256")?;

    let execution_path = object_field(receipt, "execution_path")?;
    require_string_eq(execution_path, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_string_eq(execution_path, "kernel_family", "dense_cuda_integrated_one_layer")?;
    require_string_non_empty(execution_path, "quantization_family")?;
    reject_bitnet_packed_marker(
        required_string(execution_path, "quantization_family")?,
        "execution_path.quantization_family",
    )?;
    require_bool_eq(execution_path, "bitnet_packed_kernel_proof", false)?;
    require_bool_eq(execution_path, "qk256_proof", false)?;

    validate_dense_regular_llm_execution_plan(receipt)?;

    let descriptor = object_field(receipt, "descriptor_coverage")?;
    require_u64_eq(descriptor, "schema", 1)?;
    require_string_eq(
        descriptor,
        "source_artifact_kind",
        DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND,
    )?;
    require_positive_u64(descriptor, "tensor_count")?;
    require_positive_u64(descriptor, "metadata_count")?;
    require_bool_eq(descriptor, "required_roles_present", true)?;
    require_bool_eq(descriptor, "strict_descriptor_complete", true)?;
    require_string_non_empty(descriptor, "dense_cuda_route_status")?;
    require_bool_eq(descriptor, "bitnet_packed_marker_found", false)?;
    require_bool_eq(descriptor, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(descriptor, "speedup_claim", false)?;
    require_bool_eq(descriptor, "full_cuda_residency_claimed", false)?;

    let reference = object_field(receipt, "cpu_reference")?;
    require_u64_eq(reference, "schema", 1)?;
    require_string_eq(
        reference,
        "source_artifact_kind",
        DENSE_GGUF_ONE_LAYER_CPU_REFERENCE_ARTIFACT_KIND,
    )?;
    require_string_non_empty(reference, "fixture_id")?;
    reject_bitnet_packed_marker(
        required_string(reference, "fixture_id")?,
        "cpu_reference.fixture_id",
    )?;
    require_u64_eq(reference, "layer_index", 0)?;
    require_positive_u64(reference, "seq_len")?;
    required_u64(reference, "position_offset")?;
    require_positive_u64(reference, "final_output_len")?;
    require_sha256(reference, "final_output_sha256")?;
    require_bool_eq(reference, "cpu_reference_only", true)?;
    require_bool_eq(reference, "cuda_execution_claimed", false)?;
    require_bool_eq(reference, "dense_gguf_inference_claimed", false)?;

    let layer = object_field(receipt, "cuda_layer")?;
    require_u64_eq(layer, "schema", 1)?;
    require_string_non_empty(layer, "fixture_id")?;
    reject_bitnet_packed_marker(required_string(layer, "fixture_id")?, "cuda_layer.fixture_id")?;
    require_string_eq(
        layer,
        "source_cpu_reference_fixture_id",
        required_string(reference, "fixture_id")?,
    )?;
    require_u64_eq(layer, "layer_index", 0)?;
    require_u64_eq(layer, "seq_len", required_u64(reference, "seq_len")?)?;
    require_u64_eq(layer, "position_offset", required_u64(reference, "position_offset")?)?;
    require_u64_eq(layer, "governed_cuda_ops_total", 14)?;
    require_u64_eq(layer, "residual_host_ops_total", 2)?;
    require_u64_eq(layer, "host_deterministic_input_ops_total", 1)?;
    require_u64_eq(layer, "unsupported_ops_total", 0)?;
    require_u64_eq(layer, "cpu_fallback_ops_total", 0)?;
    require_bool_eq(layer, "strict_cuda_ready", true)?;
    require_bool_eq(layer, "fallback_used", false)?;
    require_bool_eq(layer, "one_layer_cuda_integrated_parity_claimed", true)?;
    require_bool_eq(layer, "one_layer_inference_claimed", false)?;
    require_bool_eq(layer, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(layer, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(layer, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(layer, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(layer, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(layer, "speedup_claim", false)?;
    require_bool_eq(layer, "persistent_session_residency_claimed", false)?;
    require_bool_eq(layer, "full_cuda_residency_claimed", false)?;
    require_u64_eq(layer, "final_output_len", required_u64(reference, "final_output_len")?)?;
    require_sha256(layer, "final_output_sha256")?;
    require_non_negative_number(layer, "final_output_max_abs")?;
    require_non_negative_number(layer, "final_output_max_abs_error")?;
    require_non_negative_number(layer, "final_output_mean_abs_error")?;
    require_positive_number(layer, "tolerance")?;
    require_bool_eq(layer, "passed", true)?;
    let max_abs_error = object_field(layer, "final_output_max_abs_error")?
        .as_f64()
        .ok_or_else(|| anyhow!("cuda_layer.final_output_max_abs_error must be a number"))?;
    let tolerance = object_field(layer, "tolerance")?
        .as_f64()
        .ok_or_else(|| anyhow!("cuda_layer.tolerance must be a number"))?;
    if max_abs_error > tolerance {
        return Err(anyhow!("cuda_layer final output max_abs_error exceeds tolerance"));
    }

    let phases = array_field(layer, "phases")?;
    require_u64_eq(layer, "phases_total", phases.len() as u64)?;
    const REQUIRED_PHASES: &[&str] = &[
        "deterministic_input",
        "attention_norm",
        "attention_q",
        "attention_k",
        "attention_v",
        "rope",
        "attention_scores",
        "attention_softmax",
        "attention_v_mix",
        "attention_output",
        "first_residual",
        "ffn_norm",
        "mlp_gate",
        "mlp_up",
        "mlp_activation",
        "mlp_down",
        "second_residual",
    ];
    if phases.len() != REQUIRED_PHASES.len() {
        return Err(anyhow!("cuda_layer.phases_total must equal integrated layer phase count"));
    }
    let final_phase =
        phases.last().ok_or_else(|| anyhow!("cuda_layer.phases must contain a terminal phase"))?;
    require_string_eq(final_phase, "name", "second_residual")?;
    require_u64_eq(layer, "final_output_len", required_u64(final_phase, "output_len")?)?;
    require_string_eq(
        layer,
        "final_output_sha256",
        required_string(final_phase, "output_sha256")?,
    )?;
    let mut cuda_phase_count = 0_u64;
    let mut host_residual_count = 0_u64;
    let mut cuda_phase_rows = Vec::new();
    for (idx, phase) in phases.iter().enumerate() {
        require_u64_eq(phase, "index", idx as u64)?;
        require_string_eq(phase, "name", REQUIRED_PHASES[idx])?;
        require_string_non_empty(phase, "role")?;
        reject_bitnet_packed_marker(required_string(phase, "role")?, "cuda_layer.phases.role")?;
        require_string_non_empty(phase, "op_type")?;
        require_positive_u64(phase, "output_len")?;
        require_sha256(phase, "output_sha256")?;
        require_non_negative_number(phase, "max_abs")?;
        require_non_negative_number(phase, "max_abs_error")?;
        require_non_negative_number(phase, "mean_abs_error")?;
        require_non_negative_number(phase, "tolerance")?;
        let phase_max_abs_error = object_field(phase, "max_abs_error")?
            .as_f64()
            .ok_or_else(|| anyhow!("cuda_layer phase max_abs_error must be a number"))?;
        let phase_tolerance = object_field(phase, "tolerance")?
            .as_f64()
            .ok_or_else(|| anyhow!("cuda_layer phase tolerance must be a number"))?;
        if phase_max_abs_error > phase_tolerance {
            return Err(anyhow!(
                "cuda_layer phase `{}` max_abs_error exceeds tolerance",
                required_string(phase, "name")?
            ));
        }
        require_bool_eq(phase, "passed", true)?;
        require_bool_eq(phase, "fallback_used", false)?;
        match required_string(phase, "route")? {
            DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND => {
                require_string_eq(phase, "status", "cuda_executed")?;
                require_string_non_empty(phase, "kernel_id")?;
                reject_bitnet_packed_marker(
                    required_string(phase, "kernel_id")?,
                    "cuda_layer.phases.kernel_id",
                )?;
                require_positive_u64(phase, "kernel_launches")?;
                require_positive_u64(phase, "invocations")?;
                require_u64_eq(phase, "fallback_invocations", 0)?;
                required_u64(phase, "host_to_device_bytes")?;
                required_u64(phase, "device_to_host_bytes")?;
                cuda_phase_count += 1;
                cuda_phase_rows.push(phase);
            }
            "host_measured_glue" => {
                let name = required_string(phase, "name")?;
                if !matches!(name, "first_residual" | "second_residual") {
                    return Err(anyhow!("host_measured_glue is only allowed for residual phases"));
                }
                require_string_eq(phase, "status", "host_measured_glue")?;
                require_null(phase, "kernel_id")?;
                require_u64_eq(phase, "kernel_launches", 0)?;
                require_u64_eq(phase, "invocations", 1)?;
                require_u64_eq(phase, "host_to_device_bytes", 0)?;
                require_u64_eq(phase, "device_to_host_bytes", 0)?;
                host_residual_count += 1;
            }
            "host_deterministic_input" => {
                require_string_eq(phase, "name", "deterministic_input")?;
                require_string_eq(phase, "status", "host_deterministic_input")?;
                require_null(phase, "kernel_id")?;
                require_u64_eq(phase, "kernel_launches", 0)?;
                require_u64_eq(phase, "invocations", 1)?;
                require_u64_eq(phase, "host_to_device_bytes", 0)?;
                require_u64_eq(phase, "device_to_host_bytes", 0)?;
            }
            other => {
                return Err(anyhow!("unsupported cuda_layer phase route `{other}`"));
            }
        }
    }
    require_u64_eq(layer, "governed_cuda_ops_total", cuda_phase_count)?;
    require_u64_eq(layer, "residual_host_ops_total", host_residual_count)?;

    let stats = array_field(receipt, "kernel_stats")?;
    if stats.len() != cuda_phase_count as usize {
        return Err(anyhow!("kernel_stats length must match governed_cuda_ops_total"));
    }
    let mut stats_h2d = 0_u64;
    let mut stats_d2h = 0_u64;
    let mut stats_invocations = 0_u64;
    let mut stats_launches = 0_u64;
    for (stat, phase) in stats.iter().zip(cuda_phase_rows.iter()) {
        require_string_non_empty(stat, "phase")?;
        require_string_eq(stat, "phase", required_string(phase, "name")?)?;
        require_string_non_empty(stat, "kernel_id")?;
        require_string_eq(stat, "kernel_id", required_string(phase, "kernel_id")?)?;
        reject_bitnet_packed_marker(required_string(stat, "kernel_id")?, "kernel_stats.kernel_id")?;
        require_u64_eq(stat, "invocations", required_u64(phase, "invocations")?)?;
        require_u64_eq(stat, "fallback_invocations", required_u64(phase, "fallback_invocations")?)?;
        require_u64_eq(stat, "host_to_device_bytes", required_u64(phase, "host_to_device_bytes")?)?;
        require_u64_eq(stat, "device_to_host_bytes", required_u64(phase, "device_to_host_bytes")?)?;
        require_u64_eq(stat, "kernel_launches", required_u64(phase, "kernel_launches")?)?;
        require_optional_non_negative_number(stat, "kernel_time_ms")?;
        if object_field(stat, "kernel_time_ms")? != object_field(phase, "kernel_time_ms")? {
            return Err(anyhow!(
                "kernel_stats phase `{}` kernel_time_ms must match cuda_layer phase",
                required_string(stat, "phase")?
            ));
        }
        stats_h2d += required_u64(stat, "host_to_device_bytes")?;
        stats_d2h += required_u64(stat, "device_to_host_bytes")?;
        stats_invocations += required_u64(stat, "invocations")?;
        stats_launches += required_u64(stat, "kernel_launches")?;
    }

    let timing = object_field(receipt, "timing")?;
    require_optional_non_negative_number(timing, "kernel_time_ms")?;
    require_u64_eq(timing, "host_to_device_bytes", stats_h2d)?;
    require_u64_eq(timing, "device_to_host_bytes", stats_d2h)?;
    require_u64_eq(timing, "kernel_invocations", stats_invocations)?;
    require_u64_eq(timing, "kernel_launches", stats_launches)?;

    let residency = object_field(receipt, "tensor_residency")?;
    require_string_eq(residency, "scope", "integrated_dense_gguf_one_layer")?;
    require_string_eq(residency, "model_class", DENSE_REGULAR_LLM_MODEL_CLASS)?;
    require_bool_eq(residency, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(residency, "integrated_one_layer_cuda_parity_claimed", true)?;
    require_bool_eq(residency, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(residency, "persistent_session_residency_claimed", false)?;
    require_bool_eq(residency, "full_cuda_residency_claimed", false)?;
    require_bool_eq(residency, "weights_uploaded_per_kernel", true)?;
    require_bool_eq(residency, "weights_uploaded_once", false)?;
    require_bool_eq(residency, "intermediate_downloads_for_phase_parity", true)?;
    require_bool_eq(residency, "host_device_transfer_accounting_matches_kernel_stats", true)?;
    let transfer = object_field(residency, "transfer_accounting")?;
    require_string_eq(transfer, "status", "measured")?;
    require_u64_eq(transfer, "host_to_device_bytes", stats_h2d)?;
    require_u64_eq(transfer, "device_to_host_bytes", stats_d2h)?;
    require_u64_eq(transfer, "kernel_invocations", stats_invocations)?;
    require_u64_eq(transfer, "kernel_launches", stats_launches)?;

    let claim_boundary = object_field(receipt, "claim_boundary")?;
    require_bool_eq(claim_boundary, "dense_regular_llm_cuda_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_tensor_residency_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_descriptor_inspection_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_fixture_extraction_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_linear_role_sweep_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_norm_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_rope_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_score_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_softmax_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_attention_v_mix_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_mlp_activation_cuda_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_execution_plan_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_cpu_reference_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_cuda_integrated_parity_claimed", true)?;
    require_bool_eq(claim_boundary, "dense_gguf_one_layer_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(claim_boundary, "server_ready_claimed", false)?;
    require_bool_eq(claim_boundary, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(claim_boundary, "speedup_claim", false)?;
    require_bool_eq(claim_boundary, "persistent_session_residency_claimed", false)?;
    require_bool_eq(claim_boundary, "full_cuda_residency_claimed", false)?;

    Ok(())
}

struct DenseOneLayerGapCounts {
    cuda_routable_ops: u64,
    linear_cuda_ops: u64,
    norm_cuda_ops: u64,
    rope_cuda_ops: u64,
    attention_score_cuda_ops: u64,
    attention_softmax_cuda_ops: u64,
    attention_v_mix_cuda_ops: u64,
    mlp_activation_cuda_ops: u64,
    unsupported_ops: u64,
}

fn validate_dense_one_layer_gap_audit(
    receipt: &Value,
    counts: &DenseOneLayerGapCounts,
    expected_unsupported_roles: &BTreeSet<String>,
) -> Result<()> {
    let audit = object_field(receipt, "gap_audit")?;
    require_u64_eq(audit, "schema", 1)?;
    require_string_eq(
        audit,
        "source_artifact_kind",
        DENSE_GGUF_ONE_LAYER_EXECUTION_PLAN_ARTIFACT_KIND,
    )?;
    require_u64_eq(audit, "cuda_routable_ops_total", counts.cuda_routable_ops)?;
    require_u64_eq(audit, "cuda_routable_linear_ops_total", counts.linear_cuda_ops)?;
    require_u64_eq(audit, "cuda_routable_norm_ops_total", counts.norm_cuda_ops)?;
    require_u64_eq(audit, "cuda_routable_rope_ops_total", counts.rope_cuda_ops)?;
    require_u64_eq(
        audit,
        "cuda_routable_attention_score_ops_total",
        counts.attention_score_cuda_ops,
    )?;
    require_u64_eq(
        audit,
        "cuda_routable_attention_softmax_ops_total",
        counts.attention_softmax_cuda_ops,
    )?;
    require_u64_eq(
        audit,
        "cuda_routable_attention_v_mix_ops_total",
        counts.attention_v_mix_cuda_ops,
    )?;
    require_u64_eq(
        audit,
        "cuda_routable_mlp_activation_ops_total",
        counts.mlp_activation_cuda_ops,
    )?;
    require_u64_eq(audit, "unsupported_ops_total", counts.unsupported_ops)?;
    require_u64_eq(audit, "cpu_fallback_ops_total", 0)?;
    require_bool_eq(audit, "strict_cuda_ready", true)?;
    require_bool_eq(audit, "unsupported_ops_have_dependency_notes", true)?;
    require_bool_eq(audit, "strict_cuda_rejects_cpu_fallback", true)?;
    require_bool_eq(audit, "dense_gguf_one_layer_execution_plan_claimed", true)?;
    require_bool_eq(audit, "dense_gguf_one_layer_inference_claimed", false)?;
    require_bool_eq(audit, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(audit, "qwen_one_token_cuda_claimed", false)?;
    require_bool_eq(audit, "qwen_short_decode_cuda_claimed", false)?;
    require_bool_eq(audit, "qwen_chat_cuda_claimed", false)?;
    require_bool_eq(audit, "bitnet_packed_i2s_qk256_proof", false)?;
    require_bool_eq(audit, "speedup_claim", false)?;
    require_bool_eq(audit, "full_cuda_residency_claimed", false)?;

    let cuda_roles = array_field(audit, "cuda_routable_roles")?;
    if cuda_roles.len() != counts.cuda_routable_ops as usize {
        return Err(anyhow!("gap_audit.cuda_routable_roles length must match CUDA op count"));
    }
    for role in cuda_roles {
        let role = role
            .as_str()
            .ok_or_else(|| anyhow!("gap_audit.cuda_routable_roles entries must be strings"))?;
        reject_bitnet_packed_marker(role, "gap_audit.cuda_routable_roles")?;
    }

    let linear_roles = array_field(audit, "linears_routable_roles")?;
    if linear_roles.len() != counts.linear_cuda_ops as usize {
        return Err(anyhow!(
            "gap_audit.linears_routable_roles length must match CUDA linear op count"
        ));
    }
    for role in linear_roles {
        let role = role
            .as_str()
            .ok_or_else(|| anyhow!("gap_audit.linears_routable_roles entries must be strings"))?;
        reject_bitnet_packed_marker(role, "gap_audit.linears_routable_roles")?;
    }

    let norm_roles = array_field(audit, "norms_routable_roles")?;
    if norm_roles.len() != counts.norm_cuda_ops as usize {
        return Err(anyhow!(
            "gap_audit.norms_routable_roles length must match CUDA RMSNorm op count"
        ));
    }
    for role in norm_roles {
        let role = role
            .as_str()
            .ok_or_else(|| anyhow!("gap_audit.norms_routable_roles entries must be strings"))?;
        reject_bitnet_packed_marker(role, "gap_audit.norms_routable_roles")?;
    }
    require_bool_eq(audit, "rmsnorm_cuda_parity_available", true)?;

    let rope_roles = array_field(audit, "rope_routable_roles")?;
    if rope_roles.len() != counts.rope_cuda_ops as usize {
        return Err(anyhow!("gap_audit.rope_routable_roles length must match CUDA RoPE op count"));
    }
    for role in rope_roles {
        let role = role
            .as_str()
            .ok_or_else(|| anyhow!("gap_audit.rope_routable_roles entries must be strings"))?;
        reject_bitnet_packed_marker(role, "gap_audit.rope_routable_roles")?;
    }
    require_bool_eq(audit, "rope_cuda_parity_available", true)?;

    let attention_score_roles = array_field(audit, "attention_scores_routable_roles")?;
    if attention_score_roles.len() != counts.attention_score_cuda_ops as usize {
        return Err(anyhow!(
            "gap_audit.attention_scores_routable_roles length must match CUDA attention-score op count"
        ));
    }
    for role in attention_score_roles {
        let role = role.as_str().ok_or_else(|| {
            anyhow!("gap_audit.attention_scores_routable_roles entries must be strings")
        })?;
        reject_bitnet_packed_marker(role, "gap_audit.attention_scores_routable_roles")?;
    }
    require_bool_eq(audit, "attention_score_cuda_parity_available", true)?;
    let attention_softmax_roles = array_field(audit, "attention_softmax_routable_roles")?;
    if attention_softmax_roles.len() != counts.attention_softmax_cuda_ops as usize {
        return Err(anyhow!(
            "gap_audit.attention_softmax_routable_roles length must match CUDA attention-softmax op count"
        ));
    }
    for role in attention_softmax_roles {
        let role = role.as_str().ok_or_else(|| {
            anyhow!("gap_audit.attention_softmax_routable_roles entries must be strings")
        })?;
        reject_bitnet_packed_marker(role, "gap_audit.attention_softmax_routable_roles")?;
    }
    require_bool_eq(audit, "attention_softmax_cuda_parity_available", true)?;
    let attention_v_mix_roles = array_field(audit, "attention_v_mix_routable_roles")?;
    if attention_v_mix_roles.len() != counts.attention_v_mix_cuda_ops as usize {
        return Err(anyhow!(
            "gap_audit.attention_v_mix_routable_roles length must match CUDA attention V-mix op count"
        ));
    }
    for role in attention_v_mix_roles {
        let role = role.as_str().ok_or_else(|| {
            anyhow!("gap_audit.attention_v_mix_routable_roles entries must be strings")
        })?;
        reject_bitnet_packed_marker(role, "gap_audit.attention_v_mix_routable_roles")?;
    }
    require_bool_eq(audit, "attention_v_mix_cuda_parity_available", true)?;
    let mlp_activation_roles = array_field(audit, "mlp_activation_routable_roles")?;
    if mlp_activation_roles.len() != counts.mlp_activation_cuda_ops as usize {
        return Err(anyhow!(
            "gap_audit.mlp_activation_routable_roles length must match CUDA MLP activation op count"
        ));
    }
    for role in mlp_activation_roles {
        let role = role.as_str().ok_or_else(|| {
            anyhow!("gap_audit.mlp_activation_routable_roles entries must be strings")
        })?;
        reject_bitnet_packed_marker(role, "gap_audit.mlp_activation_routable_roles")?;
    }
    require_bool_eq(audit, "mlp_activation_cuda_parity_available", true)?;
    require_string_eq(audit, "next_candidate_gap", "none")?;
    require_string_eq(audit, "next_required_proof", "one_layer_cpu_reference_harness")?;

    let unsupported_entries = array_field(audit, "unsupported_ops")?;
    if unsupported_entries.len() != counts.unsupported_ops as usize {
        return Err(anyhow!("gap_audit.unsupported_ops length must match unsupported op count"));
    }
    let mut audit_roles = BTreeSet::new();
    for op in unsupported_entries {
        require_string_non_empty(op, "name")?;
        reject_bitnet_packed_marker(
            required_string(op, "name")?,
            "gap_audit.unsupported_ops.name",
        )?;
        require_string_non_empty(op, "role")?;
        let role = required_string(op, "role")?;
        reject_bitnet_packed_marker(role, "gap_audit.unsupported_ops.role")?;
        audit_roles.insert(role.to_string());
        require_string_non_empty(op, "op_type")?;
        require_positive_u64(op, "size")?;
        require_string_eq(op, "cuda_kernel_status", "missing_cuda_kernel")?;
        require_bool_eq(op, "cpu_fallback_allowed", false)?;
        require_bool_eq(op, "blocks_strict_cuda_one_layer", true)?;
        require_string_eq(op, "input_residency", "not_executed")?;
        require_string_eq(op, "output_residency", "not_executed")?;
        require_string_eq(op, "transfer_timing_status", "not_measured_no_kernel")?;
        let deps = array_field(op, "input_dependencies")?;
        if deps.is_empty() {
            return Err(anyhow!("gap_audit unsupported ops must include input dependencies"));
        }
        for dep in deps {
            let dep = dep
                .as_str()
                .ok_or_else(|| anyhow!("gap_audit input_dependencies entries must be strings"))?;
            reject_bitnet_packed_marker(dep, "gap_audit.unsupported_ops.input_dependencies")?;
        }
    }
    if &audit_roles != expected_unsupported_roles {
        return Err(anyhow!("gap_audit unsupported roles must match one_layer_plan"));
    }

    let candidate_order = array_field(audit, "candidate_order")?;
    let candidate_order = candidate_order
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .ok_or_else(|| anyhow!("gap_audit.candidate_order entries must be strings"))
        })
        .collect::<Result<Vec<_>>>()?;
    if candidate_order != DENSE_ONE_LAYER_NO_REMAINING_GAP_CANDIDATE_ORDER {
        return Err(anyhow!(
            "gap_audit.candidate_order must be empty once strict CUDA one-layer routing is complete"
        ));
    }
    let candidate_set: BTreeSet<String> =
        candidate_order.iter().map(|role| (*role).to_string()).collect();
    if &candidate_set != expected_unsupported_roles {
        return Err(anyhow!("gap_audit.candidate_order roles must match unsupported roles"));
    }

    let op_type_counts = object_field(audit, "unsupported_op_type_counts")?
        .as_object()
        .ok_or_else(|| anyhow!("gap_audit.unsupported_op_type_counts must be an object"))?;
    let mut op_type_sum = 0_u64;
    for (op_type, count) in op_type_counts {
        if op_type.trim().is_empty() {
            return Err(anyhow!("gap_audit unsupported op type key must not be empty"));
        }
        reject_bitnet_packed_marker(op_type, "gap_audit.unsupported_op_type_counts")?;
        op_type_sum += count.as_u64().ok_or_else(|| {
            anyhow!("gap_audit.unsupported_op_type_counts values must be unsigned integers")
        })?;
    }
    if op_type_sum != counts.unsupported_ops {
        return Err(anyhow!("gap_audit.unsupported_op_type_counts must sum to unsupported_ops"));
    }

    let dependency_edges = array_field(audit, "dependency_edges")?;
    if dependency_edges.len() < counts.unsupported_ops as usize {
        return Err(anyhow!(
            "gap_audit.dependency_edges must describe unsupported op dependencies"
        ));
    }
    for edge in dependency_edges {
        require_string_non_empty(edge, "from")?;
        require_string_non_empty(edge, "to")?;
        reject_bitnet_packed_marker(
            required_string(edge, "from")?,
            "gap_audit.dependency_edges.from",
        )?;
        reject_bitnet_packed_marker(required_string(edge, "to")?, "gap_audit.dependency_edges.to")?;
    }

    Ok(())
}

fn validate_dense_one_layer_gap_execution_plan(receipt: &Value) -> Result<()> {
    let plan = object_field(receipt, "execution_plan")?;
    require_string_eq(plan, "planner_version", CUDA_PLANNER_RECEIPT_VERSION)?;
    require_string_non_empty(plan, "model_family")?;
    reject_bitnet_packed_marker(
        required_string(plan, "model_family")?,
        "execution_plan.model_family",
    )?;
    require_string_non_empty(plan, "quantization")?;
    reject_bitnet_packed_marker(
        required_string(plan, "quantization")?,
        "execution_plan.quantization",
    )?;
    require_string_eq(plan, "selected_route", DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND)?;
    require_string_eq(plan, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(plan, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(plan, "runtime_api", "cuda")?;
    require_string_eq(plan, "strict_fallback_policy", "reject")?;
    require_bool_eq(plan, "dense_regular_llm_cuda", true)?;
    require_bool_eq(plan, "bitnet_packed_qk256_cuda", false)?;
    require_u64_eq(plan, "cuda_bitnet_qk256_ops", 0)?;
    require_positive_u64(plan, "cuda_dense_regular_llm_ops")?;
    require_u64_eq(plan, "cpu_fallback_ops", 0)?;
    require_u64_eq(plan, "unsupported_ops", 0)?;
    let total_ops = object_field(plan, "total_ops")?
        .as_u64()
        .ok_or_else(|| anyhow!("execution_plan.total_ops must be an unsigned integer"))?;
    let cuda_ops = object_field(plan, "cuda_ops")?
        .as_u64()
        .ok_or_else(|| anyhow!("execution_plan.cuda_ops must be an unsigned integer"))?;
    let dense_ops =
        object_field(plan, "cuda_dense_regular_llm_ops")?.as_u64().ok_or_else(|| {
            anyhow!("execution_plan.cuda_dense_regular_llm_ops must be an unsigned integer")
        })?;
    let unsupported_ops = object_field(plan, "unsupported_ops")?
        .as_u64()
        .ok_or_else(|| anyhow!("execution_plan.unsupported_ops must be an unsigned integer"))?;
    if cuda_ops != dense_ops || total_ops != dense_ops + unsupported_ops {
        return Err(anyhow!(
            "execution_plan dense CUDA and unsupported op counts are inconsistent"
        ));
    }
    require_bool_eq(plan, "mixed_cuda_routes", false)?;
    require_bool_eq(plan, "fallback_used", false)?;
    require_bool_eq(plan, "strict_cuda_ready", true)?;
    require_bool_eq(plan, "speedup_claim", false)?;
    require_bool_eq(plan, "full_cuda_residency_claimed", false)?;

    Ok(())
}

const REQUIRED_DENSE_DESCRIPTOR_ROLES: &[&str] = &[
    "token_embedding",
    "output",
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_output",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
    "attention_norm",
    "ffn_norm",
];

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
    let descriptor_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| claim_boundary.get("dense_gguf_descriptor_inspection_claimed"))
        .and_then(Value::as_bool);
    let linear_fixture_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| {
            claim_boundary.get("dense_gguf_linear_fixture_extraction_claimed")
        })
        .and_then(Value::as_bool);
    let norm_fixture_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| claim_boundary.get("dense_gguf_norm_fixture_extraction_claimed"))
        .and_then(Value::as_bool);
    let norm_cuda_parity_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| claim_boundary.get("dense_gguf_norm_cuda_parity_claimed"))
        .and_then(Value::as_bool);
    let attention_score_fixture_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| {
            claim_boundary.get("dense_gguf_attention_score_fixture_extraction_claimed")
        })
        .and_then(Value::as_bool);
    let attention_score_cuda_parity_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| {
            claim_boundary.get("dense_gguf_attention_score_cuda_parity_claimed")
        })
        .and_then(Value::as_bool);
    let attention_softmax_fixture_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| {
            claim_boundary.get("dense_gguf_attention_softmax_fixture_extraction_claimed")
        })
        .and_then(Value::as_bool);
    let attention_softmax_cuda_parity_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| {
            claim_boundary.get("dense_gguf_attention_softmax_cuda_parity_claimed")
        })
        .and_then(Value::as_bool);
    let attention_v_mix_fixture_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| {
            claim_boundary.get("dense_gguf_attention_v_mix_fixture_extraction_claimed")
        })
        .and_then(Value::as_bool);
    let attention_v_mix_cuda_parity_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| {
            claim_boundary.get("dense_gguf_attention_v_mix_cuda_parity_claimed")
        })
        .and_then(Value::as_bool);
    let mlp_activation_fixture_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| {
            claim_boundary.get("dense_gguf_mlp_activation_fixture_extraction_claimed")
        })
        .and_then(Value::as_bool);
    let mlp_activation_cuda_parity_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| {
            claim_boundary.get("dense_gguf_mlp_activation_cuda_parity_claimed")
        })
        .and_then(Value::as_bool);
    let linear_cuda_parity_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| claim_boundary.get("dense_gguf_linear_cuda_parity_claimed"))
        .and_then(Value::as_bool);
    let linear_role_sweep_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| {
            claim_boundary.get("dense_gguf_linear_role_sweep_cuda_parity_claimed")
        })
        .and_then(Value::as_bool);
    let one_layer_plan_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| {
            claim_boundary.get("dense_gguf_one_layer_execution_plan_claimed")
        })
        .and_then(Value::as_bool);
    let one_layer_cpu_reference_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| claim_boundary.get("dense_gguf_one_layer_cpu_reference_claimed"))
        .and_then(Value::as_bool);
    let one_layer_cuda_integrated_parity_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| {
            claim_boundary.get("dense_gguf_one_layer_cuda_integrated_parity_claimed")
        })
        .and_then(Value::as_bool);
    let all_layer_execution_plan_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| {
            claim_boundary.get("dense_gguf_all_layer_execution_plan_claimed")
        })
        .and_then(Value::as_bool);
    let model_boundary_fixtures_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| claim_boundary.get("dense_gguf_model_boundary_fixtures_claimed"))
        .and_then(Value::as_bool);
    let kv_cache_policy_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| claim_boundary.get("kv_cache_policy_claimed"))
        .and_then(Value::as_bool);
    let sampling_policy_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| claim_boundary.get("sampling_policy_claimed"))
        .and_then(Value::as_bool);
    let qwen_one_token_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| claim_boundary.get("qwen_one_token_cuda_claimed"))
        .and_then(Value::as_bool);
    let qwen_short_decode_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| claim_boundary.get("qwen_short_decode_cuda_claimed"))
        .and_then(Value::as_bool);
    let qwen_warm_session_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| claim_boundary.get("qwen_warm_session_cuda_claimed"))
        .and_then(Value::as_bool);
    let qwen_ask_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| claim_boundary.get("qwen_ask_cuda_claimed"))
        .and_then(Value::as_bool);
    let qwen_chat_claim = receipt
        .get("claim_boundary")
        .and_then(|claim_boundary| claim_boundary.get("qwen_chat_cuda_claimed"))
        .and_then(Value::as_bool);

    if artifact_kind == Some(DENSE_REGULAR_LLM_CUDA_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_DESCRIPTOR_INSPECTION_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_LINEAR_FIXTURE_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_NORM_FIXTURE_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_NORM_CUDA_PARITY_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_ROPE_CUDA_PARITY_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_ATTENTION_SCORE_FIXTURE_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_ATTENTION_SCORE_CUDA_PARITY_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_ATTENTION_SOFTMAX_FIXTURE_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_ATTENTION_SOFTMAX_CUDA_PARITY_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_ATTENTION_V_MIX_FIXTURE_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_ATTENTION_V_MIX_CUDA_PARITY_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_MLP_ACTIVATION_FIXTURE_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_MLP_ACTIVATION_CUDA_PARITY_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_LINEAR_CUDA_PARITY_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_LINEAR_ROLE_SWEEP_CUDA_PARITY_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_ONE_LAYER_EXECUTION_PLAN_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_ONE_LAYER_CPU_REFERENCE_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_ONE_LAYER_CUDA_INTEGRATED_PARITY_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_ALL_LAYER_EXECUTION_PLAN_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_MODEL_BOUNDARY_FIXTURES_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_KV_CACHE_POLICY_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_SAMPLING_POLICY_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_QWEN_ONE_TOKEN_STRICT_CUDA_PROOF_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_QWEN_SHORT_DECODE_STRICT_CUDA_PROOF_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_QWEN_WARM_DECODE_STRICT_CUDA_PROOF_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_QWEN_WARM_SESSION_STRICT_CUDA_PROOF_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_QWEN_ASK_STRICT_CUDA_PROOF_ARTIFACT_KIND)
        || artifact_kind == Some(DENSE_GGUF_QWEN_CHAT_STRICT_CUDA_PROOF_ARTIFACT_KIND)
        || model_class == Some(DENSE_REGULAR_LLM_MODEL_CLASS)
        || dense_claim == Some(true)
        || descriptor_claim == Some(true)
        || linear_fixture_claim == Some(true)
        || norm_fixture_claim == Some(true)
        || norm_cuda_parity_claim == Some(true)
        || attention_score_fixture_claim == Some(true)
        || attention_score_cuda_parity_claim == Some(true)
        || attention_softmax_fixture_claim == Some(true)
        || attention_softmax_cuda_parity_claim == Some(true)
        || attention_v_mix_fixture_claim == Some(true)
        || attention_v_mix_cuda_parity_claim == Some(true)
        || mlp_activation_fixture_claim == Some(true)
        || mlp_activation_cuda_parity_claim == Some(true)
        || linear_cuda_parity_claim == Some(true)
        || linear_role_sweep_claim == Some(true)
        || one_layer_plan_claim == Some(true)
        || one_layer_cpu_reference_claim == Some(true)
        || one_layer_cuda_integrated_parity_claim == Some(true)
        || all_layer_execution_plan_claim == Some(true)
        || model_boundary_fixtures_claim == Some(true)
        || kv_cache_policy_claim == Some(true)
        || sampling_policy_claim == Some(true)
        || qwen_one_token_claim == Some(true)
        || qwen_short_decode_claim == Some(true)
        || qwen_warm_session_claim == Some(true)
        || qwen_ask_claim == Some(true)
        || qwen_chat_claim == Some(true)
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

/// Return the canonical SHA256 identity digest for an Apple M4 `run_identity`.
pub fn m4_run_identity_sha256(run_identity: &Value) -> Result<String> {
    let bytes = serde_json::to_vec(run_identity)?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Validate the reusable Apple M4 run-identity contract on a receipt.
///
/// This checks the top-level `run_identity` object and, when present, the
/// top-level `run_identity_sha256` digest. It also cross-checks the common
/// top-level backend and artifact fields so receipt validators can share one
/// identity gate without duplicating field-level checks.
pub fn validate_m4_run_identity_contract_json(receipt: &Value) -> Result<()> {
    let identity = object_field(receipt, "run_identity")?;
    require_string_eq(identity, "contract_version", M4_RUN_IDENTITY_CONTRACT_VERSION)?;
    require_string_non_empty_not_tbd(identity, "machine_id")?;
    require_string_non_empty_not_tbd(identity, "soc")?;
    require_string_non_empty_not_tbd(identity, "artifact_kind")?;
    require_string_non_empty_not_tbd(identity, "evidence_family")?;

    if let Some(receipt_artifact_kind) = receipt.get("artifact_kind").and_then(Value::as_str) {
        let identity_artifact_kind = required_string(identity, "artifact_kind")?;
        if identity_artifact_kind != receipt_artifact_kind {
            return Err(anyhow!("run_identity.artifact_kind must match receipt artifact_kind"));
        }
    }

    validate_m4_run_identity_os(object_field(identity, "os")?)?;
    validate_m4_run_identity_git(object_field(identity, "git")?)?;
    validate_m4_run_identity_binary(object_field(identity, "binary")?)?;
    validate_m4_run_identity_command(object_field(identity, "command")?)?;
    validate_m4_run_identity_model(object_field(identity, "model")?)?;
    validate_m4_run_identity_tokenizer(object_field(identity, "tokenizer")?)?;
    validate_m4_run_identity_prompt_template(object_field(identity, "prompt_template")?)?;
    validate_m4_run_identity_backend(receipt, object_field(identity, "backend")?)?;
    validate_m4_run_identity_evidence(object_field(identity, "evidence_identity")?)?;
    validate_m4_run_identity_timing(object_field(identity, "timing")?)?;

    let digest = object_field(receipt, "run_identity_sha256")?
        .as_str()
        .ok_or_else(|| anyhow!("field `run_identity_sha256` must be a string"))?;
    if digest.len() != 64 || !digest.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Err(anyhow!(
            "field `run_identity_sha256` must be a 64-character sha256 hex digest"
        ));
    }
    let expected = m4_run_identity_sha256(identity)?;
    if digest != expected {
        return Err(anyhow!("field `run_identity_sha256` does not match run_identity"));
    }

    Ok(())
}

/// Validate Lunar Lake OpenVINO route receipts against the shared proof boundary.
///
/// This is a claim-boundary validator, not a route promoter. It accepts the
/// existing OpenVINO dense-SLM, route-profile, route-ledger, and diagnosis
/// receipt families while rejecting hidden fallback, backend/device identity
/// drift, retokenized-token ambiguity, dense-SLM-to-BitNet claim leakage,
/// OpenVINO-GPU-to-native-OpenCL claim leakage, and premature NPU promotion
/// without cache plus warm/resident evidence.
pub fn validate_lunar_lake_openvino_receipt_json(receipt: &Value) -> Result<()> {
    let artifact_kind = required_string(receipt, "artifact_kind")?;
    if !is_lunar_lake_openvino_artifact_kind(artifact_kind) {
        return Err(anyhow!("unsupported Lunar Lake OpenVINO artifact_kind `{artifact_kind}`"));
    }

    if let Some(machine_id) = receipt.get("machine_id").and_then(Value::as_str)
        && machine_id != "intel-258v"
    {
        return Err(anyhow!(
            "Lunar Lake OpenVINO receipts must target machine_id `intel-258v`, got `{machine_id}`"
        ));
    }

    if receipt.get("fallback_used").is_some() {
        require_bool_eq(receipt, "fallback_used", false)?;
    }
    if receipt.get("runtime_api").is_some() {
        require_string_eq(receipt, "runtime_api", "openvino_genai")?;
    }

    validate_lunar_lake_openvino_value(receipt, "$")?;
    Ok(())
}

/// Validate a Lunar Lake OpenVINO route receipt file.
pub fn validate_lunar_lake_openvino_receipt_file(path: &Path) -> Result<()> {
    let receipt = load_json_receipt(path)?;
    validate_lunar_lake_openvino_receipt_json(&receipt)
}

fn is_lunar_lake_openvino_artifact_kind(artifact_kind: &str) -> bool {
    matches!(
        artifact_kind,
        "intel_258v_dense_slm_openvino_corpus_v2"
            | "intel_258v_dense_slm_openvino_generation_budget_sensitivity"
            | "intel_258v_dense_slm_openvino_phase_comparison"
            | "intel_258v_dense_slm_openvino_phase_runner"
            | "intel_258v_dense_slm_openvino_profile_run"
            | "lunar_lake_openvino_corpus_v2_diagnosis"
            | "lunar_lake_openvino_npu_cold_start_diagnosis"
            | "lunar_lake_openvino_npu_cache_experiment"
            | "lunar_lake_openvino_npu_resident_session"
            | "lunar_lake_openvino_operator_ask"
            | "lunar_lake_route_profile_comparison"
            | "lunar_lake_route_promotion_ledger"
    )
}

fn validate_lunar_lake_openvino_value(value: &Value, path: &str) -> Result<()> {
    match value {
        Value::Object(object) => {
            validate_lunar_lake_openvino_object(value, path)?;
            for (key, child) in object {
                let child_path = format!("{path}.{key}");
                validate_lunar_lake_openvino_forbidden_claim(key, child, &child_path)?;
                validate_lunar_lake_openvino_value(child, &child_path)?;
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                validate_lunar_lake_openvino_value(child, &format!("{path}[{index}]"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_lunar_lake_openvino_object(object: &Value, path: &str) -> Result<()> {
    validate_lunar_lake_openvino_backend_object(object, path)?;
    validate_lunar_lake_openvino_generated_token_marking(object, path)?;
    validate_lunar_lake_openvino_npu_promotion_evidence(object, path)
}

fn validate_lunar_lake_openvino_backend_object(object: &Value, path: &str) -> Result<()> {
    let selected_backend = object.get("selected_backend").and_then(Value::as_str);
    let route_id = object.get("route_id").and_then(Value::as_str);
    let openvino_route = route_id.is_some_and(|route| route.contains("openvino"));
    let Some(selected_backend) = selected_backend else {
        return Ok(());
    };

    if !selected_backend.starts_with("openvino") && !openvino_route {
        return Ok(());
    }

    if let Some(fallback_used) = object.get("fallback_used").and_then(Value::as_bool) {
        if fallback_used {
            return Err(anyhow!("{path} OpenVINO route must record fallback_used=false"));
        }
    } else if object.get("fallback_policy").and_then(Value::as_str) == Some("strict_no_fallback") {
        // Policy/ledger entries do not execute a route themselves, but they must
        // still fail closed by declaring strict no-fallback routing.
    } else {
        return Err(anyhow!(
            "{path} OpenVINO route must record fallback_used=false or fallback_policy=strict_no_fallback"
        ));
    }

    if selected_backend.starts_with("openvino") {
        require_string_eq(object, "runtime_api", "openvino_genai")
            .map_err(|err| anyhow!("{path}: {err}"))?;
    }

    if let Some(route_id) = route_id {
        match route_id {
            "dense_slm_openvino_gpu_candidate" if selected_backend != "openvino-gpu" => {
                return Err(anyhow!(
                    "{path} GPU OpenVINO route must select openvino-gpu, got `{selected_backend}`"
                ));
            }
            "dense_slm_openvino_npu_candidate" if selected_backend != "openvino-npu" => {
                return Err(anyhow!(
                    "{path} NPU OpenVINO route must select openvino-npu, got `{selected_backend}`"
                ));
            }
            route if route.contains("openvino_cpu") && selected_backend != "openvino-cpu" => {
                return Err(anyhow!(
                    "{path} CPU OpenVINO route must select openvino-cpu, got `{selected_backend}`"
                ));
            }
            _ => {}
        }
    }

    match selected_backend {
        "openvino-cpu" => require_runtime_device_prefix(object, "CPU", path)?,
        "openvino-gpu" => {
            require_runtime_device_prefix(object, "GPU", path)?;
            reject_openvino_gpu_opencl_claim(object, path)?;
        }
        "openvino-npu" => require_runtime_device_prefix(object, "NPU", path)?,
        "openvino-cpu-gpu-npu" => {}
        other if other.starts_with("openvino") => {
            return Err(anyhow!("{path} has unsupported OpenVINO backend `{other}`"));
        }
        _ => {}
    }

    Ok(())
}

fn require_runtime_device_prefix(object: &Value, expected_prefix: &str, path: &str) -> Result<()> {
    let Some(runtime_device) = object.get("runtime_device").and_then(Value::as_str) else {
        return Ok(());
    };
    if !runtime_device.starts_with(expected_prefix) {
        return Err(anyhow!(
            "{path} runtime_device `{runtime_device}` must start with `{expected_prefix}`"
        ));
    }
    Ok(())
}

fn reject_openvino_gpu_opencl_claim(object: &Value, path: &str) -> Result<()> {
    for field in ["runtime_api", "backend_lane", "selected_kernel_or_runtime"] {
        let Some(value) = object.get(field).and_then(Value::as_str) else {
            continue;
        };
        let normalized = value.replace(['-', '_', ' '], "").to_ascii_lowercase();
        if normalized.contains("opencl") {
            return Err(anyhow!(
                "{path}.{field} must not claim native OpenCL for an OpenVINO GPU route"
            ));
        }
    }
    Ok(())
}

fn validate_lunar_lake_openvino_generated_token_marking(object: &Value, path: &str) -> Result<()> {
    if object.get("generated_token_ids").is_none_or(|value| value.as_array().is_none()) {
        return Ok(());
    }

    let source = required_string(object, "generated_token_ids_source")
        .map_err(|err| anyhow!("{path}: generated_token_ids require source marking: {err}"))?;
    let available_from_pipeline = object
        .get("generated_token_ids_available_from_pipeline")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if !available_from_pipeline && !source.contains("retokenized") {
        return Err(anyhow!(
            "{path}.generated_token_ids_source must mark retokenized IDs when OpenVINO pipeline IDs are unavailable"
        ));
    }
    if available_from_pipeline && source.contains("retokenized") {
        return Err(anyhow!(
            "{path}.generated_token_ids_source must not claim retokenized IDs when pipeline IDs are marked available"
        ));
    }

    Ok(())
}

fn validate_lunar_lake_openvino_npu_promotion_evidence(object: &Value, path: &str) -> Result<()> {
    let route_id = object.get("route_id").and_then(Value::as_str).unwrap_or_default();
    let selected_backend =
        object.get("selected_backend").and_then(Value::as_str).unwrap_or_default();
    if !route_id.contains("openvino_npu") && selected_backend != "openvino-npu" {
        return Ok(());
    }

    let promotion_attempted = object
        .get("route_status")
        .or_else(|| object.get("status"))
        .or_else(|| object.get("promotion_status"))
        .and_then(Value::as_str)
        .is_some_and(|status| status == "promoted")
        || object.get("promotion_eligible_for_profile").and_then(Value::as_bool).unwrap_or(false);
    if !promotion_attempted {
        return Ok(());
    }

    if !openvino_object_has_cache_evidence(object) {
        return Err(anyhow!("{path} promoted OpenVINO NPU route must include cache evidence"));
    }
    if !openvino_object_has_warm_or_resident_evidence(object) {
        return Err(anyhow!(
            "{path} promoted OpenVINO NPU route must include warm or resident evidence"
        ));
    }

    Ok(())
}

fn openvino_object_has_cache_evidence(object: &Value) -> bool {
    object.get("npu_cache").is_some()
        || object.get("cache").is_some()
        || object.get("cache_identity").is_some()
        || object.get("cache_hit").is_some()
        || object
            .get("phase_coverage")
            .is_some_and(|value| value_contains_case_insensitive(value, "cache"))
        || object.get("timing").is_some_and(|value| value_contains_case_insensitive(value, "cache"))
}

fn openvino_object_has_warm_or_resident_evidence(object: &Value) -> bool {
    object.get("warm_session").is_some()
        || object.get("resident_session").is_some()
        || object
            .get("phase_coverage")
            .is_some_and(|value| value_contains_case_insensitive(value, "warm"))
        || object
            .get("phase_coverage")
            .is_some_and(|value| value_contains_case_insensitive(value, "resident"))
        || object.get("timing").is_some_and(|value| value_contains_case_insensitive(value, "warm"))
        || object
            .get("timing")
            .is_some_and(|value| value_contains_case_insensitive(value, "resident"))
}

fn validate_lunar_lake_openvino_forbidden_claim(
    key: &str,
    value: &Value,
    path: &str,
) -> Result<()> {
    let normalized_key = key.replace(['-', '_', ' '], "").to_ascii_lowercase();
    if value.as_bool() == Some(true)
        && (normalized_key.contains("qk256")
            || normalized_key.contains("i2s")
            || normalized_key.contains("bitnetpacked")
            || normalized_key.contains("nativeopencl")
            || normalized_key.contains("openclclaim"))
    {
        return Err(anyhow!("{path} must not be true for Lunar Lake OpenVINO receipts"));
    }

    if matches!(key, "claim" | "proof_family" | "claim_boundary" | "selected_kernel_or_runtime")
        && let Some(text) = value.as_str()
    {
        reject_bitnet_packed_marker(text, path)?;
        let normalized_text = text.replace(['-', '_', ' '], "").to_ascii_lowercase();
        if normalized_text.contains("nativeopencl") {
            return Err(anyhow!(
                "{path} must not claim native OpenCL for Lunar Lake OpenVINO receipts"
            ));
        }
    }

    Ok(())
}

fn value_contains_case_insensitive(value: &Value, needle: &str) -> bool {
    match value {
        Value::String(text) => text.to_ascii_lowercase().contains(needle),
        Value::Array(items) => {
            items.iter().any(|item| value_contains_case_insensitive(item, needle))
        }
        Value::Object(map) => {
            map.values().any(|item| value_contains_case_insensitive(item, needle))
        }
        _ => false,
    }
}

fn validate_m4_run_identity_os(os: &Value) -> Result<()> {
    require_string_non_empty_not_tbd(os, "name")?;
    require_string_non_empty_not_tbd(os, "version")?;
    require_string_non_empty_not_tbd(os, "version_source")
}

fn validate_m4_run_identity_git(git: &Value) -> Result<()> {
    let commit = required_string(git, "commit")?;
    if commit.trim().is_empty() || commit == "TBD" || commit == "unknown" {
        return Err(anyhow!("field `commit` must record a concrete git commit"));
    }
    require_string_non_empty_not_tbd(git, "commit_source")
}

fn validate_m4_run_identity_binary(binary: &Value) -> Result<()> {
    require_string_non_empty_not_tbd(binary, "crate_version")?;
    let build_profile = binary.get("build_profile").and_then(Value::as_str);
    let binary_sha256 = binary.get("binary_sha256").and_then(Value::as_str);
    if build_profile.is_none_or(|value| value.trim().is_empty())
        && binary_sha256.is_none_or(str::is_empty)
    {
        return Err(anyhow!("run_identity.binary must record build_profile or binary_sha256"));
    }
    if let Some(sha256) = binary_sha256
        && (sha256.len() != 64 || !sha256.chars().all(|ch| ch.is_ascii_hexdigit()))
    {
        return Err(anyhow!("field `binary_sha256` must be a 64-character sha256 hex digest"));
    }
    Ok(())
}

fn validate_m4_run_identity_command(command: &Value) -> Result<()> {
    require_string_non_empty_not_tbd(command, "class")?;
    object_field(command, "live_model_run")?
        .as_bool()
        .ok_or_else(|| anyhow!("field `live_model_run` must be a boolean"))?;
    Ok(())
}

fn validate_m4_run_identity_model(model: &Value) -> Result<()> {
    require_string_non_empty_not_tbd(model, "id")?;
    require_sha256_or_not_applicable(model, "sha256")
}

fn validate_m4_run_identity_tokenizer(tokenizer: &Value) -> Result<()> {
    require_string_non_empty_not_tbd(tokenizer, "authority")?;
    require_sha256_or_not_applicable(tokenizer, "sha256")?;
    if let Some(strict) = tokenizer.get("strict") {
        strict.as_bool().ok_or_else(|| anyhow!("field `strict` must be a boolean"))?;
    }
    Ok(())
}

fn validate_m4_run_identity_prompt_template(prompt_template: &Value) -> Result<()> {
    require_string_non_empty_not_tbd(prompt_template, "id")?;
    require_sha256(prompt_template, "sha256")
}

fn validate_m4_run_identity_backend(receipt: &Value, backend: &Value) -> Result<()> {
    require_string_non_empty_not_tbd(backend, "requested_backend")?;
    require_string_non_empty_not_tbd(backend, "selected_backend")?;
    require_string_non_empty_not_tbd(backend, "runtime_api")?;
    require_bool_eq(backend, "fallback_used", false)?;
    require_same_string(
        backend,
        "requested_backend",
        backend,
        "selected_backend",
        "run_identity backend selection",
    )?;
    for field in ["requested_backend", "selected_backend", "runtime_api"] {
        if let Some(top_level) = receipt.get(field).and_then(Value::as_str) {
            let identity_value = required_string(backend, field)?;
            if top_level != identity_value {
                return Err(anyhow!("run_identity.backend.{field} must match receipt {field}"));
            }
        }
    }
    if let Some(top_level_fallback) = receipt.get("fallback_used").and_then(Value::as_bool)
        && top_level_fallback != object_field(backend, "fallback_used")?.as_bool().unwrap_or(true)
    {
        return Err(anyhow!("run_identity.backend.fallback_used must match receipt fallback_used"));
    }
    Ok(())
}

fn validate_m4_run_identity_evidence(evidence: &Value) -> Result<()> {
    require_string_non_empty_not_tbd(evidence, "scope")?;
    require_string_non_empty_not_tbd(evidence, "seed")?;
    require_string_non_empty_not_tbd(evidence, "corpus_id")?;
    require_string_non_empty_not_tbd(evidence, "profile_id")
}

fn validate_m4_run_identity_timing(timing: &Value) -> Result<()> {
    require_string_non_empty_not_tbd(timing, "source")
}

fn require_sha256_or_not_applicable(object: &Value, field: &str) -> Result<()> {
    let value = required_string(object, field)?;
    if value == "not_applicable" {
        return Ok(());
    }
    require_sha256(object, field)
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

fn required_u64(object: &Value, field: &str) -> Result<u64> {
    object_field(object, field)?
        .as_u64()
        .ok_or_else(|| anyhow!("field `{field}` must be an unsigned integer"))
}

fn require_string_eq(object: &Value, field: &str, expected: &str) -> Result<()> {
    let actual = required_string(object, field)?;
    if actual != expected {
        return Err(anyhow!("field `{field}` must be `{expected}`, got `{actual}`"));
    }
    Ok(())
}

fn require_same_string(
    left: &Value,
    left_field: &str,
    right: &Value,
    right_field: &str,
    label: &str,
) -> Result<()> {
    let left = required_string(left, left_field)?;
    let right = required_string(right, right_field)?;
    if left != right {
        return Err(anyhow!("`{label}` must match between `{left_field}` and `{right_field}`"));
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

const DENSE_ALL_LAYER_OPERATION_SEQUENCE: [(&str, &str); 14] = [
    ("attention_norm", "rmsnorm"),
    ("attention_q", "matmul"),
    ("attention_k", "matmul"),
    ("attention_v", "matmul"),
    ("rope", "rope"),
    ("attention_scores", "attention"),
    ("attention_softmax", "softmax"),
    ("attention_v_mix", "attention"),
    ("attention_output", "matmul"),
    ("ffn_norm", "rmsnorm"),
    ("mlp_gate", "matmul"),
    ("mlp_up", "matmul"),
    ("mlp_activation", "activation"),
    ("mlp_down", "matmul"),
];

fn dense_all_layer_operation_signature_sha256(operations: &[Value]) -> Result<String> {
    let signature = operations
        .iter()
        .map(dense_all_layer_operation_signature_entry)
        .collect::<Result<Vec<_>>>()?;
    let bytes = serde_json::to_vec(&Value::Array(signature))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn dense_all_layer_operation_signature_entry(op: &Value) -> Result<Value> {
    let mut entry = Map::new();
    entry.insert("role".to_string(), Value::String(required_string(op, "role")?.to_string()));
    entry.insert("op_type".to_string(), Value::String(required_string(op, "op_type")?.to_string()));
    entry.insert("source".to_string(), Value::String(required_string(op, "source")?.to_string()));
    entry.insert(
        "source_tensor_type".to_string(),
        op.get("source_tensor_type").cloned().unwrap_or(Value::Null),
    );
    entry
        .insert("source_shape".to_string(), op.get("source_shape").cloned().unwrap_or(Value::Null));
    entry.insert(
        "is_quantized".to_string(),
        op.get("is_quantized").cloned().unwrap_or(Value::Bool(false)),
    );
    entry.insert("route".to_string(), Value::String(required_string(op, "route")?.to_string()));
    entry.insert("status".to_string(), Value::String(required_string(op, "status")?.to_string()));
    Ok(Value::Object(entry))
}

fn require_sha256(object: &Value, field: &str) -> Result<()> {
    let value = required_string(object, field)?;
    if value.len() != 64 || !value.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Err(anyhow!("field `{field}` must be a 64-character sha256 hex digest"));
    }
    Ok(())
}

fn require_extractable_dense_linear_role(role: &str) -> Result<()> {
    const EXTRACTABLE_ROLES: &[&str] = &[
        "output",
        "attention_q",
        "attention_k",
        "attention_v",
        "attention_output",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
    ];
    if !EXTRACTABLE_ROLES.contains(&role) {
        return Err(anyhow!(
            "linear_fixture.role must be an extractable dense linear role, got `{role}`"
        ));
    }
    Ok(())
}

fn require_extractable_dense_norm_role(role: &str) -> Result<()> {
    const EXTRACTABLE_ROLES: &[&str] = &["attention_norm", "ffn_norm"];
    if !EXTRACTABLE_ROLES.contains(&role) {
        return Err(anyhow!(
            "norm_fixtures.role must be an extractable dense norm role, got `{role}`"
        ));
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

fn require_bool_alias_eq(
    object: &Value,
    fields: &[&str],
    expected: bool,
    label: &str,
) -> Result<()> {
    let mut saw_field = false;
    for field in fields {
        if let Some(value) = object.get(*field) {
            saw_field = true;
            let actual =
                value.as_bool().ok_or_else(|| anyhow!("field `{field}` must be a bool"))?;
            if actual != expected {
                return Err(anyhow!("field `{field}` must be `{expected}`, got `{actual}`"));
            }
        }
    }
    if saw_field { Ok(()) } else { Err(anyhow!("field `{label}` must be `{expected}`")) }
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

fn require_optional_u64_field(object: &Value, field: &str) -> Result<()> {
    let value = object_field(object, field)?;
    if value.is_null() {
        return Ok(());
    }
    value.as_u64().ok_or_else(|| anyhow!("field `{field}` must be null or an unsigned integer"))?;
    Ok(())
}

fn reject_bitnet_packed_marker(value: &str, field: &str) -> Result<()> {
    let normalized = value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    const BITNET_PACKED_MARKERS: &[&str] = &["bitnet", "i2s", "iq2s", "qk256", "w158a8"];
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

fn require_positive_number(object: &Value, field: &str) -> Result<()> {
    let actual = object_field(object, field)?
        .as_f64()
        .ok_or_else(|| anyhow!("field `{field}` must be a number"))?;
    if actual <= 0.0 {
        return Err(anyhow!("field `{field}` must be positive"));
    }
    Ok(())
}

fn require_number(object: &Value, field: &str) -> Result<()> {
    object_field(object, field)?
        .as_f64()
        .ok_or_else(|| anyhow!("field `{field}` must be a number"))?;
    Ok(())
}

fn validate_dense_boundary_tensor_fixture(fixture: &Value, expected_role: &str) -> Result<()> {
    require_string_non_empty(fixture, "name")?;
    reject_bitnet_packed_marker(
        required_string(fixture, "name")?,
        "model_boundary_fixtures.fixture.name",
    )?;
    require_string_eq(fixture, "role", expected_role)?;
    require_string_non_empty(fixture, "tensor_name")?;
    reject_bitnet_packed_marker(
        required_string(fixture, "tensor_name")?,
        "model_boundary_fixtures.fixture.tensor_name",
    )?;
    require_string_non_empty(fixture, "tensor_type")?;
    reject_bitnet_packed_marker(
        required_string(fixture, "tensor_type")?,
        "model_boundary_fixtures.fixture.tensor_type",
    )?;
    if array_field(fixture, "source_shape")?.is_empty() {
        return Err(anyhow!("model_boundary_fixtures.fixture.source_shape must not be empty"));
    }
    required_u64(fixture, "source_offset")?;
    require_positive_u64(fixture, "source_size_bytes")?;
    require_positive_u64(fixture, "value_count")?;
    require_positive_u64(fixture, "output_len")?;
    require_sha256(fixture, "output_sha256")?;
    require_non_negative_number(fixture, "max_abs")?;
    require_bool_eq(fixture, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(fixture, "bitnet_packed_i2s_qk256_proof", false)?;
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
        if provenance.requested_backend != provenance.selected_backend {
            return Err(anyhow!(
                "strict CPU proof backend mismatch: requested={:?}, selected={:?}",
                provenance.requested_backend,
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
    matches!(label, "cpu" | "apple-m4-cpu-neon" | "apple-m3-air-cpu-neon")
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
    use serde_json::json;

    fn minimal_lunar_lake_openvino_gpu_receipt() -> Value {
        json!({
            "schema_version": "1.0.0",
            "artifact_kind": "lunar_lake_openvino_operator_ask",
            "machine_id": "intel-258v",
            "proof_stage": "operator_candidate_route_executed",
            "requested_backend": "openvino-gpu",
            "selected_backend": "openvino-gpu",
            "runtime_api": "openvino_genai",
            "runtime_device": "GPU.0",
            "fallback_used": false,
            "backend_lane": "dense_slm_openvino_gpu_arc140v",
            "selected_kernel_or_runtime": "openvino-genai-llmpipeline-gpu0",
            "route_id": "dense_slm_openvino_gpu_candidate",
            "model_family": "qwen",
            "model_architecture": "qwen2",
            "quantization": "INT4_SYM",
            "prompt_template": "qwen2.5",
            "tokenizer_source": "hf_tokenizer_export",
            "generation": {
                "decoded_text": "2+2 equals 4.",
                "generated_token_ids": [17, 488, 17],
                "generated_token_ids_available_from_pipeline": false,
                "generated_token_ids_source": "retokenized_generated_text_not_pipeline_internal_ids"
            }
        })
    }

    #[test]
    fn lunar_lake_openvino_validator_accepts_candidate_gpu_receipt() {
        let result =
            validate_lunar_lake_openvino_receipt_json(&minimal_lunar_lake_openvino_gpu_receipt());
        assert!(result.is_ok(), "got: {result:?}");
    }

    #[test]
    fn lunar_lake_openvino_validator_accepts_npu_resident_session_receipt() {
        let receipt = json!({
            "schema_version": "1.0.0",
            "artifact_kind": "lunar_lake_openvino_npu_resident_session",
            "machine_id": "intel-258v",
            "requested_backend": "openvino-npu",
            "selected_backend": "openvino-npu",
            "runtime_api": "openvino_genai",
            "runtime_device": "NPU",
            "fallback_used": false,
            "route_id": "dense_slm_openvino_npu_candidate",
            "resident_session": {
                "resident_session_ready": true,
                "warm_resident_asks": {
                    "ask_count": 10,
                    "passed": 10,
                    "failed": 0,
                    "fallback_used": false
                }
            },
            "asks": [{
                "generated_text": "2+2 equals 4.",
                "generated_token_ids": [17, 488, 17],
                "generated_token_ids_available_from_pipeline": false,
                "generated_token_ids_source": "retokenized_generated_text_not_pipeline_internal_ids"
            }],
            "claim_boundary": {
                "route_promotion_changed": false,
                "speedup_claim": false,
                "power_advantage_claim": false,
                "acceleration_claim": false,
                "native_npu_inference_claim": false,
                "bitnet_qk256_i2s_behavior_changed": false
            }
        });
        let result = validate_lunar_lake_openvino_receipt_json(&receipt);
        assert!(result.is_ok(), "got: {result:?}");
    }

    #[test]
    fn lunar_lake_openvino_validator_accepts_npu_cache_experiment_receipt() {
        let receipt = json!({
            "schema_version": "1.0.0",
            "artifact_kind": "lunar_lake_openvino_npu_cache_experiment",
            "machine_id": "intel-258v",
            "requested_backend": "openvino-npu",
            "selected_backend": "openvino-npu",
            "runtime_api": "openvino_genai",
            "runtime_device": "NPU",
            "fallback_used": false,
            "route_id": "dense_slm_openvino_npu_candidate",
            "cache": {
                "cache_dir": "target/openvino-cache/lnl258v-npu-cache-001",
                "cache_enabled": true,
                "cache_hit_runtime_metric_available": false,
                "cache_effective_by_timing": false
            },
            "process_runs": [{
                "child_receipt": {
                    "generated_text": "2+2 equals 4.",
                    "generated_token_ids": [17, 488, 17],
                    "generated_token_ids_available_from_pipeline": false,
                    "generated_token_ids_source": "retokenized_generated_text_not_pipeline_internal_ids"
                }
            }],
            "generated_token_visibility": {
                "direct_generated_token_ids_available": false,
                "generated_token_ids_source": "retokenized_generated_text_not_pipeline_internal_ids"
            },
            "claim_boundary": {
                "route_promotion_changed": false,
                "speedup_claim": false,
                "power_advantage_claim": false,
                "acceleration_claim": false,
                "native_npu_inference_claim": false,
                "bitnet_qk256_i2s_behavior_changed": false
            }
        });
        let result = validate_lunar_lake_openvino_receipt_json(&receipt);
        assert!(result.is_ok(), "got: {result:?}");
    }

    #[test]
    fn lunar_lake_openvino_validator_accepts_profile_run_receipt() {
        let receipt = json!({
            "schema_version": "1.0.0",
            "artifact_kind": "intel_258v_dense_slm_openvino_profile_run",
            "machine_id": "intel-258v",
            "proof_stage": "openvino_heavy_profile_timing_evidence",
            "generation": {
                "fallback_used": false,
                "devices": [
                    {
                        "runtime_device": "GPU.0",
                        "requested_backend": "openvino-gpu",
                        "selected_backend": "openvino-gpu",
                        "runtime_api": "openvino_genai",
                        "selected_kernel_or_runtime": "openvino-genai-llmpipeline-gpu0",
                        "fallback_used": false,
                        "cases": [
                            {
                                "id": "prefill_heavy_route_policy_long_context",
                                "profile": "prefill_heavy",
                                "requested_backend": "openvino-gpu",
                                "selected_backend": "openvino-gpu",
                                "runtime_api": "openvino_genai",
                                "runtime_device": "GPU.0",
                                "selected_kernel_or_runtime": "openvino-genai-llmpipeline-gpu0",
                                "fallback_used": false,
                                "prompt_token_count": 2731,
                                "generated_token_ids": [111, 222, 333],
                                "generated_token_ids_available_from_pipeline": true,
                                "generated_token_ids_source": "openvino_genai_encoded_results_tokens",
                                "generated_token_count": 3
                            }
                        ]
                    },
                    {
                        "runtime_device": "NPU",
                        "requested_backend": "openvino-npu",
                        "selected_backend": "openvino-npu",
                        "runtime_api": "openvino_genai",
                        "selected_kernel_or_runtime": "openvino-genai-llmpipeline-npu",
                        "fallback_used": false,
                        "cases": [
                            {
                                "id": "decode_heavy_route_policy_long_generation",
                                "profile": "decode_heavy",
                                "requested_backend": "openvino-npu",
                                "selected_backend": "openvino-npu",
                                "runtime_api": "openvino_genai",
                                "runtime_device": "NPU",
                                "selected_kernel_or_runtime": "openvino-genai-llmpipeline-npu",
                                "fallback_used": false,
                                "prompt_token_count": 66,
                                "generated_token_ids": [444, 555, 666],
                                "generated_token_ids_available_from_pipeline": true,
                                "generated_token_ids_source": "openvino_genai_encoded_results_tokens",
                                "generated_token_count": 3
                            }
                        ]
                    }
                ]
            },
            "verification": {
                "fallback_used": false,
                "route_promotion_changed": false,
                "candidate_routes_remain_unpromoted": true
            },
            "claim_boundary": {
                "speedup_claim": false,
                "power_advantage_claim": false,
                "acceleration_claim": false,
                "route_promotion_changed": false,
                "native_npu_inference_claim": false,
                "bitnet_qk256_i2s_behavior_changed": false
            }
        });
        let result = validate_lunar_lake_openvino_receipt_json(&receipt);
        assert!(result.is_ok(), "got: {result:?}");
    }

    #[test]
    fn lunar_lake_openvino_validator_rejects_hidden_fallback() {
        let mut receipt = minimal_lunar_lake_openvino_gpu_receipt();
        receipt["fallback_used"] = json!(true);
        let err = validate_lunar_lake_openvino_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("fallback_used"), "got: {err}");
    }

    #[test]
    fn lunar_lake_openvino_validator_rejects_backend_device_mismatch() {
        let mut receipt = minimal_lunar_lake_openvino_gpu_receipt();
        receipt["runtime_device"] = json!("CPU");
        let err = validate_lunar_lake_openvino_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("runtime_device"), "got: {err}");
    }

    #[test]
    fn lunar_lake_openvino_validator_rejects_route_backend_mismatch() {
        let mut receipt = minimal_lunar_lake_openvino_gpu_receipt();
        receipt["selected_backend"] = json!("openvino-npu");
        receipt["runtime_device"] = json!("NPU");
        let err = validate_lunar_lake_openvino_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("GPU OpenVINO route must select openvino-gpu"), "got: {err}");
    }

    #[test]
    fn lunar_lake_openvino_validator_rejects_opencl_claim_leak() {
        let mut receipt = minimal_lunar_lake_openvino_gpu_receipt();
        receipt["selected_kernel_or_runtime"] = json!("native-opencl-kernel");
        let err = validate_lunar_lake_openvino_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("native OpenCL"), "got: {err}");
    }

    #[test]
    fn lunar_lake_openvino_validator_requires_retokenized_id_marking() {
        let mut receipt = minimal_lunar_lake_openvino_gpu_receipt();
        let generation = receipt.get_mut("generation").and_then(serde_json::Value::as_object_mut);
        assert!(generation.is_some(), "test receipt must include generation object");
        if let Some(generation) = generation {
            generation.remove("generated_token_ids_source");
        }
        let err = validate_lunar_lake_openvino_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("generated_token_ids require source marking"), "got: {err}");
    }

    #[test]
    fn lunar_lake_openvino_validator_rejects_dense_slm_to_bitnet_claim_leak() {
        let mut receipt = minimal_lunar_lake_openvino_gpu_receipt();
        receipt["bitnet_packed_i2s_qk256_proof"] = json!(true);
        let err = validate_lunar_lake_openvino_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("qk256"), "got: {err}");
    }

    #[test]
    fn lunar_lake_openvino_validator_requires_npu_cache_and_warm_evidence_for_promotion() {
        let receipt = json!({
            "schema_version": "1.0.0",
            "artifact_kind": "lunar_lake_route_promotion_ledger",
            "machine_id": "intel-258v",
            "routes": [{
                "route_id": "dense_slm_openvino_npu_candidate",
                "status": "promoted",
                "selected_backend": "openvino-npu",
                "runtime_api": "openvino_genai",
                "runtime_device": "NPU",
                "fallback_policy": "strict_no_fallback"
            }]
        });
        let err = validate_lunar_lake_openvino_receipt_json(&receipt).unwrap_err().to_string();
        assert!(err.contains("cache evidence"), "got: {err}");
    }

    #[test]
    fn lunar_lake_openvino_validator_accepts_npu_promotion_with_cache_and_warm_evidence() {
        let receipt = json!({
            "schema_version": "1.0.0",
            "artifact_kind": "lunar_lake_route_promotion_ledger",
            "machine_id": "intel-258v",
            "routes": [{
                "route_id": "dense_slm_openvino_npu_candidate",
                "status": "promoted",
                "selected_backend": "openvino-npu",
                "runtime_api": "openvino_genai",
                "runtime_device": "NPU",
                "fallback_policy": "strict_no_fallback",
                "cache": {"mode": "openvino_model_cache", "cache_hit": true},
                "warm_session": {"mode": "resident", "attempts": 10}
            }]
        });
        let result = validate_lunar_lake_openvino_receipt_json(&receipt);
        assert!(result.is_ok(), "got: {result:?}");
    }

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
    fn test_validate_strict_cpu_proof_accepts_m3_air_cpu_neon_label() {
        let mut receipt = strict_cpu_proof_receipt();
        assert!(receipt.strict_provenance.is_some());
        if let Some(provenance) = receipt.strict_provenance.as_mut() {
            provenance.requested_backend = "apple-m3-air-cpu-neon".to_string();
            provenance.selected_backend = "apple-m3-air-cpu-neon".to_string();
            provenance.selected_kernel = Some("i2_s-scalar-reference".to_string());
            provenance.requested_kernel = Some("i2_s-scalar-reference".to_string());
            provenance.quant_format = Some("I2_S".to_string());
            provenance.cpu_features = vec!["neon".to_string()];
        }
        receipt.kernels = vec!["i2_s-scalar-reference".to_string()];

        assert!(receipt.validate_strict_cpu_proof().is_ok());
    }

    #[test]
    fn test_validate_strict_cpu_proof_rejects_apple_cpu_label_mismatch() {
        let mut receipt = strict_cpu_proof_receipt();
        assert!(receipt.strict_provenance.is_some());
        if let Some(provenance) = receipt.strict_provenance.as_mut() {
            provenance.requested_backend = "apple-m3-air-cpu-neon".to_string();
            provenance.selected_backend = "apple-m4-cpu-neon".to_string();
            provenance.selected_kernel = Some("i2_s-scalar-reference".to_string());
            provenance.requested_kernel = Some("i2_s-scalar-reference".to_string());
            provenance.quant_format = Some("I2_S".to_string());
            provenance.cpu_features = vec!["neon".to_string()];
        }
        receipt.kernels = vec!["i2_s-scalar-reference".to_string()];

        let err = match receipt.validate_strict_cpu_proof() {
            Ok(()) => String::from("strict CPU proof unexpectedly passed"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("backend mismatch"), "unexpected error: {err}");
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
