use super::*;

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
