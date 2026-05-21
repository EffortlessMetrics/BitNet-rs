//! RTX 5070 Ti CUDA receipt validation tests.
//!
//! These tests validate strict smoke/parity proof artifacts without claiming
//! benchmark or full BitNet inference readiness.
#![recursion_limit = "256"]

use bitnet_receipts::{
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof, validate_cuda_parity_receipt_json,
    validate_cuda_smoke_receipt_json, validate_dense_gguf_all_layer_execution_plan_receipt_json,
    validate_dense_gguf_attention_score_cuda_parity_receipt_json,
    validate_dense_gguf_attention_score_fixture_receipt_json,
    validate_dense_gguf_attention_softmax_cuda_parity_receipt_json,
    validate_dense_gguf_attention_softmax_fixture_receipt_json,
    validate_dense_gguf_attention_v_mix_cuda_parity_receipt_json,
    validate_dense_gguf_attention_v_mix_fixture_receipt_json,
    validate_dense_gguf_kv_cache_policy_receipt_json,
    validate_dense_gguf_linear_cuda_parity_receipt_json,
    validate_dense_gguf_linear_fixture_extraction_receipt_json,
    validate_dense_gguf_linear_role_sweep_cuda_parity_receipt_json,
    validate_dense_gguf_mlp_activation_cuda_parity_receipt_json,
    validate_dense_gguf_mlp_activation_fixture_receipt_json,
    validate_dense_gguf_model_boundary_fixtures_receipt_json,
    validate_dense_gguf_norm_cuda_parity_receipt_json,
    validate_dense_gguf_norm_fixture_extraction_receipt_json,
    validate_dense_gguf_one_layer_cpu_reference_receipt_json,
    validate_dense_gguf_one_layer_cuda_integrated_parity_receipt_json,
    validate_dense_gguf_one_layer_execution_plan_receipt_json,
    validate_dense_gguf_qwen_ask_strict_cuda_proof_receipt_json,
    validate_dense_gguf_qwen_chat_strict_cuda_proof_receipt_json,
    validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json,
    validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json,
    validate_dense_gguf_qwen_warm_decode_strict_cuda_proof_receipt_json,
    validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json,
    validate_dense_gguf_rope_cuda_parity_receipt_json,
    validate_dense_gguf_sampling_policy_receipt_json,
    validate_dense_gguf_tensor_descriptor_inspection_receipt_json,
    validate_dense_regular_llm_cuda_persistent_residency_receipt_json,
    validate_dense_regular_llm_cuda_receipt_json,
    validate_dense_regular_llm_cuda_tensor_residency_receipt_json,
    validate_server_shared_engine_chat_completion_receipt_json,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn server_shared_engine_chat_completion_receipt() -> Value {
    json!({
        "receipt_kind": "server_shared_engine_chat_completion",
        "request_id": "request-1",
        "runtime_path": "shared_local_inference_engine",
        "runtime_api": "cuda",
        "model_identity": {
            "model_id": "qwen2.5-0.5b-instruct-q8_0",
            "requested_model": "qwen2.5-0.5b-instruct-q8_0",
            "active_model_id": "model-1",
            "active_model_path": "models/qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf",
            "model_sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"
        },
        "endpoint_profile": {
            "endpoint": "/v1/chat/completions",
            "method": "POST",
            "request_profile": "non_streaming_chat_completion",
            "streaming": false,
            "message_count": 1
        },
        "generation_policy": {
            "max_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0,
            "decoding": "greedy"
        },
        "requested_model": "qwen2.5-0.5b-instruct-q8_0",
        "active_model_id": "model-1",
        "active_model_path": "models/qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf",
        "model_sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
        "model_coverage_row": "dense_qwen25_05b_q8_cuda",
        "model_coverage_tier": "product_cli_ready",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_route": "dense_regular_llm_cuda",
        "prompt_template": "chatml",
        "tokenizer_authority": "active_model_tokenizer",
        "prompt_authority": "server_chat_template",
        "fallback_used": false,
        "simulated_inference": false,
        "streaming": false,
        "generated_text_non_empty": true,
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_ms": 25,
        "quality_gate": {
            "gate": "server_non_empty_utf8_response",
            "passed": true,
            "generated_text_non_empty": true,
            "utf8_valid": true,
            "broad_chat_quality_claimed": false
        },
        "server_smoke_response_claimed": true,
        "server_ready_claimed": false,
        "speedup_claim": false,
        "full_cuda_residency_claimed": false,
        "dense_regular_llm_cuda_inference_claimed": true,
        "bitnet_packed_i2s_qk256_proof": false
    })
}

fn qwen3_server_shared_engine_chat_completion_receipt() -> Value {
    let mut receipt = server_shared_engine_chat_completion_receipt();
    receipt["model_identity"]["model_id"] = json!("qwen3-0.6b-instruct-q8_0");
    receipt["model_identity"]["requested_model"] = json!("qwen3-0.6b-instruct-q8_0");
    receipt["model_identity"]["active_model_path"] =
        json!("models/qwen3-0.6b-instruct-q8_0/Qwen3-0.6B-Q8_0.gguf");
    receipt["model_identity"]["model_sha256"] =
        json!("9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031");
    receipt["requested_model"] = json!("qwen3-0.6b-instruct-q8_0");
    receipt["active_model_path"] = json!("models/qwen3-0.6b-instruct-q8_0/Qwen3-0.6B-Q8_0.gguf");
    receipt["model_sha256"] =
        json!("9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031");
    receipt["model_coverage_row"] = json!("dense_qwen3_06b_q8_candidate");
    receipt
}

fn bitnet_qk256_server_smoke_receipt() -> Value {
    json!({
        "receipt_kind": "server_shared_engine_chat_completion",
        "request_id": "request-bitnet-1",
        "runtime_path": "shared_local_inference_engine",
        "runtime_api": "cuda",
        "model_identity": {
            "model_id": "microsoft-bitnet-b1.58-2B-4T-i2s",
            "requested_model": "microsoft-bitnet-b1.58-2B-4T-i2s",
            "active_model_id": "model-1",
            "active_model_path": "models/microsoft-bitnet-b1.58-2B-4T-i2s/ggml-model-i2_s.gguf",
            "model_sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162"
        },
        "endpoint_profile": {
            "endpoint": "/v1/chat/completions",
            "method": "POST",
            "request_profile": "non_streaming_chat_completion",
            "streaming": false,
            "message_count": 1
        },
        "generation_policy": {
            "max_tokens": 2,
            "temperature": 0.0,
            "top_p": 1.0,
            "decoding": "greedy"
        },
        "requested_model": "microsoft-bitnet-b1.58-2B-4T-i2s",
        "active_model_id": "model-1",
        "active_model_path": "models/microsoft-bitnet-b1.58-2B-4T-i2s/ggml-model-i2_s.gguf",
        "model_sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
        "model_coverage_row": "bitnet_official_2b_i2s_qk256",
        "model_coverage_tier": "product_cli_ready",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_route": "bitnet_qk256_cuda",
        "execution_plan": {
            "planner_version": "cuda-planner-004",
            "model_family": "bitnet_b1_58",
            "quantization": "i2_s_qk256",
            "selected_route": "bitnet_qk256_cuda",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "strict_fallback_policy": "reject",
            "dense_regular_llm_cuda": false,
            "bitnet_packed_qk256_cuda": true,
            "cuda_bitnet_qk256_ops": 420,
            "cuda_dense_regular_llm_ops": 0,
            "cpu_fallback_ops": 0,
            "unsupported_ops": 0,
            "total_ops": 420,
            "cuda_ops": 420,
            "mixed_cuda_routes": false,
            "fallback_used": false,
            "strict_cuda_ready": true,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "execution_coverage": {
            "execution_claim": "cuda_inference_contribution",
            "bitnet_linear_layers_total": 420,
            "bitnet_linear_layers_on_cuda": 420,
            "bitnet_linear_layers_cpu_fallback": 0,
            "unsupported_ops": [],
            "fallback_used": false
        },
        "kernel_stats": [{
            "kernel_id": "qk256_gemv_cuda",
            "invocations": 420,
            "fallback_invocations": 0,
            "cpu_fallback_invocations": 0,
            "host_to_device_bytes": 1024,
            "device_to_host_bytes": 2048,
            "kernel_launches": 420,
            "kernel_time_ms": 12.5,
            "kernel_time_samples": 420
        }],
        "prompt_template": "bitnetcpp-answer",
        "tokenizer_authority": "active_model_tokenizer",
        "prompt_authority": "server_chat_template",
        "fallback_used": false,
        "simulated_inference": false,
        "streaming": false,
        "generated_text_non_empty": true,
        "prompt_tokens": 14,
        "completion_tokens": 1,
        "total_ms": 83,
        "quality_gate": {
            "gate": "server_non_empty_utf8_response",
            "passed": true,
            "generated_text_non_empty": true,
            "utf8_valid": true,
            "broad_chat_quality_claimed": false
        },
        "server_smoke_response_claimed": true,
        "server_ready_claimed": false,
        "speedup_claim": false,
        "full_cuda_residency_claimed": false,
        "dense_regular_llm_cuda_inference_claimed": false,
        "bitnet_packed_i2s_qk256_proof": true
    })
}

fn remove_top_level_field(receipt: &mut Value, field: &str) {
    let removed = receipt.as_object_mut().and_then(|object| object.remove(field));
    assert!(removed.is_some(), "expected receipt field `{field}` to exist");
}

fn remove_nested_field(receipt: &mut Value, object_field: &str, field: &str) {
    let removed = receipt
        .get_mut(object_field)
        .and_then(Value::as_object_mut)
        .and_then(|object| object.remove(field));
    assert!(removed.is_some(), "expected receipt field `{object_field}.{field}` to exist");
}

#[test]
fn committed_cuda_smoke_receipt_validates() -> Result<(), Box<dyn std::error::Error>> {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-smoke.json"
    ))?;

    validate_cuda_smoke_receipt_json(&receipt)?;
    Ok(())
}

#[test]
fn committed_cuda_parity_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-parity.json"
    ))
    .unwrap();

    validate_cuda_parity_receipt_json(&receipt).unwrap();
}

#[test]
fn committed_dense_regular_llm_cuda_gemm_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-parity.json"
    ))
    .unwrap();

    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_regular_llm_cuda_residency_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-residency.json"
    ))
    .unwrap();

    validate_dense_regular_llm_cuda_tensor_residency_receipt_json(&receipt).unwrap();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_regular_llm_cuda_persistent_residency_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-persistent.json"
    ))
    .unwrap();

    validate_dense_regular_llm_cuda_persistent_residency_receipt_json(&receipt).unwrap();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_qwen3_chat_user_path_receipts_validate() -> Result<(), Box<dyn std::error::Error>> {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-18/qwen3-0_6b-chat-user-path-cuda.json"
    ))?;
    let source_receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-18/qwen3-0_6b-chat-user-path-cuda.source-warm-session.json"
    ))?;

    validate_dense_gguf_qwen_chat_strict_cuda_proof_receipt_json(&receipt)?;
    validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(&source_receipt)?;
    assert!(validate_dense_regular_llm_cuda_receipt_json(&receipt).is_err());
    assert!(reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).is_err());
    Ok(())
}

#[test]
fn server_shared_engine_chat_completion_receipt_validates() {
    let receipt = server_shared_engine_chat_completion_receipt();

    assert!(validate_server_shared_engine_chat_completion_receipt_json(&receipt).is_ok());
}

#[test]
fn qwen3_server_shared_engine_chat_completion_receipt_validates_without_readiness_claim() {
    let receipt = qwen3_server_shared_engine_chat_completion_receipt();

    assert!(validate_server_shared_engine_chat_completion_receipt_json(&receipt).is_ok());
    assert_eq!(receipt["model_coverage_row"], "dense_qwen3_06b_q8_candidate");
    assert_eq!(receipt["server_ready_claimed"], false);
    assert_eq!(receipt["speedup_claim"], false);
    assert_eq!(receipt["full_cuda_residency_claimed"], false);
    assert_eq!(receipt["dense_regular_llm_cuda_inference_claimed"], true);
    assert_eq!(receipt["bitnet_packed_i2s_qk256_proof"], false);
}

#[test]
fn bitnet_qk256_server_smoke_receipt_validates() {
    let receipt = bitnet_qk256_server_smoke_receipt();

    assert!(validate_server_shared_engine_chat_completion_receipt_json(&receipt).is_ok());
}

#[test]
fn bitnet_qk256_server_smoke_receipt_rejects_cpu_fallback() {
    let mut receipt = bitnet_qk256_server_smoke_receipt();
    receipt["execution_plan"]["cpu_fallback_ops"] = json!(1);
    receipt["execution_plan"]["strict_cuda_ready"] = json!(false);
    receipt["execution_coverage"]["bitnet_linear_layers_cpu_fallback"] = json!(1);
    receipt["execution_coverage"]["fallback_used"] = json!(true);
    receipt["kernel_stats"][0]["fallback_invocations"] = json!(1);
    receipt["kernel_stats"][0]["cpu_fallback_invocations"] = json!(1);

    let err = validate_server_shared_engine_chat_completion_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("cpu_fallback_ops"), "unexpected error: {err}");
}

#[test]
fn bitnet_qk256_server_smoke_receipt_rejects_dense_claim_leak() {
    let mut receipt = bitnet_qk256_server_smoke_receipt();
    receipt["dense_regular_llm_cuda_inference_claimed"] = json!(true);

    let err = validate_server_shared_engine_chat_completion_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("dense_regular_llm_cuda_inference_claimed"), "unexpected error: {err}");
}

#[test]
fn committed_stale_server_shared_engine_chat_completion_receipt_fails_hardened_validator()
-> Result<(), serde_json::Error> {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/server-strict-dense-qwen25-q8-smoke.json"
    ))?;

    assert!(validate_server_shared_engine_chat_completion_receipt_json(&receipt).is_err());
    Ok(())
}

#[test]
fn committed_refreshed_server_shared_engine_chat_completion_receipt_validates()
-> Result<(), serde_json::Error> {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-17/server-strict-dense-qwen25-q8-smoke.json"
    ))?;

    assert!(validate_server_shared_engine_chat_completion_receipt_json(&receipt).is_ok());
    Ok(())
}

#[test]
fn server_shared_engine_chat_completion_receipt_rejects_missing_profile_fields() {
    let mut missing_checksum = server_shared_engine_chat_completion_receipt();
    missing_checksum["model_identity"]["model_sha256"] = Value::Null;
    assert!(validate_server_shared_engine_chat_completion_receipt_json(&missing_checksum).is_err());

    let mut missing_endpoint = server_shared_engine_chat_completion_receipt();
    remove_top_level_field(&mut missing_endpoint, "endpoint_profile");
    assert!(validate_server_shared_engine_chat_completion_receipt_json(&missing_endpoint).is_err());

    let mut missing_policy = server_shared_engine_chat_completion_receipt();
    remove_top_level_field(&mut missing_policy, "generation_policy");
    assert!(validate_server_shared_engine_chat_completion_receipt_json(&missing_policy).is_err());
}

#[test]
fn server_shared_engine_chat_completion_receipt_rejects_missing_request_authority_or_usage_fields()
{
    let mut missing_request_id = server_shared_engine_chat_completion_receipt();
    remove_top_level_field(&mut missing_request_id, "request_id");
    assert!(
        validate_server_shared_engine_chat_completion_receipt_json(&missing_request_id).is_err()
    );

    let mut missing_prompt_template = server_shared_engine_chat_completion_receipt();
    remove_top_level_field(&mut missing_prompt_template, "prompt_template");
    assert!(
        validate_server_shared_engine_chat_completion_receipt_json(&missing_prompt_template)
            .is_err()
    );

    let mut missing_tokenizer_authority = server_shared_engine_chat_completion_receipt();
    remove_top_level_field(&mut missing_tokenizer_authority, "tokenizer_authority");
    assert!(
        validate_server_shared_engine_chat_completion_receipt_json(&missing_tokenizer_authority)
            .is_err()
    );

    let mut missing_prompt_authority = server_shared_engine_chat_completion_receipt();
    remove_top_level_field(&mut missing_prompt_authority, "prompt_authority");
    assert!(
        validate_server_shared_engine_chat_completion_receipt_json(&missing_prompt_authority)
            .is_err()
    );

    let mut missing_prompt_tokens = server_shared_engine_chat_completion_receipt();
    remove_top_level_field(&mut missing_prompt_tokens, "prompt_tokens");
    assert!(
        validate_server_shared_engine_chat_completion_receipt_json(&missing_prompt_tokens).is_err()
    );

    let mut missing_completion_tokens = server_shared_engine_chat_completion_receipt();
    remove_top_level_field(&mut missing_completion_tokens, "completion_tokens");
    assert!(
        validate_server_shared_engine_chat_completion_receipt_json(&missing_completion_tokens)
            .is_err()
    );

    let mut missing_total_ms = server_shared_engine_chat_completion_receipt();
    remove_top_level_field(&mut missing_total_ms, "total_ms");
    assert!(validate_server_shared_engine_chat_completion_receipt_json(&missing_total_ms).is_err());

    let mut missing_quality_gate_name = server_shared_engine_chat_completion_receipt();
    remove_nested_field(&mut missing_quality_gate_name, "quality_gate", "gate");
    assert!(
        validate_server_shared_engine_chat_completion_receipt_json(&missing_quality_gate_name)
            .is_err()
    );
}

#[test]
fn server_shared_engine_chat_completion_receipt_rejects_checksum_identity_mismatch() {
    let mut mismatch = server_shared_engine_chat_completion_receipt();
    mismatch["model_sha256"] =
        json!("9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031");
    assert!(validate_server_shared_engine_chat_completion_receipt_json(&mismatch).is_err());

    let mut missing_top_level = server_shared_engine_chat_completion_receipt();
    remove_top_level_field(&mut missing_top_level, "model_sha256");
    assert!(
        validate_server_shared_engine_chat_completion_receipt_json(&missing_top_level).is_err()
    );
}

#[test]
fn server_shared_engine_chat_completion_receipt_rejects_inconsistent_model_identity() {
    let mut receipt = server_shared_engine_chat_completion_receipt();
    receipt["model_identity"]["model_id"] = json!("qwen2.5-0.5b-instruct-q4_k_m");

    assert!(validate_server_shared_engine_chat_completion_receipt_json(&receipt).is_err());
}

#[test]
fn server_shared_engine_chat_completion_receipt_rejects_wrong_dense_model_scope() {
    let mut receipt = qwen3_server_shared_engine_chat_completion_receipt();
    receipt["model_coverage_row"] = json!("dense_qwen25_05b_q8_cuda");

    assert!(validate_server_shared_engine_chat_completion_receipt_json(&receipt).is_err());
}

#[test]
fn server_shared_engine_chat_completion_receipt_rejects_unknown_dense_model_scope() {
    let mut receipt = server_shared_engine_chat_completion_receipt();
    receipt["model_identity"]["model_id"] = json!("smollm2-360m-instruct");
    receipt["model_identity"]["requested_model"] = json!("smollm2-360m-instruct");
    receipt["requested_model"] = json!("smollm2-360m-instruct");
    receipt["model_coverage_row"] = json!("dense_qwen3_06b_q8_candidate");

    assert!(validate_server_shared_engine_chat_completion_receipt_json(&receipt).is_err());
}

#[test]
fn server_shared_engine_chat_completion_receipt_rejects_hidden_fallback_and_ready_claims() {
    let mut generic_backend = server_shared_engine_chat_completion_receipt();
    generic_backend["selected_backend"] = json!("cuda");
    assert!(validate_server_shared_engine_chat_completion_receipt_json(&generic_backend).is_err());

    let mut generic_requested_backend = server_shared_engine_chat_completion_receipt();
    generic_requested_backend["requested_backend"] = json!("cuda");
    assert!(
        validate_server_shared_engine_chat_completion_receipt_json(&generic_requested_backend)
            .is_err()
    );

    let mut non_cuda_runtime_api = server_shared_engine_chat_completion_receipt();
    non_cuda_runtime_api["runtime_api"] = json!("wgpu");
    assert!(
        validate_server_shared_engine_chat_completion_receipt_json(&non_cuda_runtime_api).is_err()
    );

    let mut fallback = server_shared_engine_chat_completion_receipt();
    fallback["fallback_used"] = json!(true);
    assert!(validate_server_shared_engine_chat_completion_receipt_json(&fallback).is_err());

    let mut not_smoke = server_shared_engine_chat_completion_receipt();
    not_smoke["server_smoke_response_claimed"] = json!(false);
    assert!(validate_server_shared_engine_chat_completion_receipt_json(&not_smoke).is_err());

    let mut promoted = server_shared_engine_chat_completion_receipt();
    promoted["server_ready_claimed"] = json!(true);
    assert!(validate_server_shared_engine_chat_completion_receipt_json(&promoted).is_err());

    let mut speedup = server_shared_engine_chat_completion_receipt();
    speedup["speedup_claim"] = json!(true);
    assert!(validate_server_shared_engine_chat_completion_receipt_json(&speedup).is_err());

    let mut full_residency = server_shared_engine_chat_completion_receipt();
    full_residency["full_cuda_residency_claimed"] = json!(true);
    assert!(validate_server_shared_engine_chat_completion_receipt_json(&full_residency).is_err());

    let mut broad_quality = server_shared_engine_chat_completion_receipt();
    broad_quality["quality_gate"]["broad_chat_quality_claimed"] = json!(true);
    assert!(validate_server_shared_engine_chat_completion_receipt_json(&broad_quality).is_err());
}

#[test]
fn server_shared_engine_chat_completion_receipt_rejects_simulated_or_low_quality_smoke() {
    let mut simulated = server_shared_engine_chat_completion_receipt();
    simulated["simulated_inference"] = json!(true);
    assert!(validate_server_shared_engine_chat_completion_receipt_json(&simulated).is_err());

    let mut empty_text = server_shared_engine_chat_completion_receipt();
    empty_text["generated_text_non_empty"] = json!(false);
    assert!(validate_server_shared_engine_chat_completion_receipt_json(&empty_text).is_err());

    let mut quality_failed = server_shared_engine_chat_completion_receipt();
    quality_failed["quality_gate"]["passed"] = json!(false);
    assert!(validate_server_shared_engine_chat_completion_receipt_json(&quality_failed).is_err());

    let mut invalid_utf8 = server_shared_engine_chat_completion_receipt();
    invalid_utf8["quality_gate"]["utf8_valid"] = json!(false);
    assert!(validate_server_shared_engine_chat_completion_receipt_json(&invalid_utf8).is_err());
}

#[test]
fn server_shared_engine_chat_completion_receipt_rejects_streaming_profile_mismatch() {
    let mut streaming_mismatch = server_shared_engine_chat_completion_receipt();
    streaming_mismatch["endpoint_profile"]["streaming"] = json!(true);

    assert!(
        validate_server_shared_engine_chat_completion_receipt_json(&streaming_mismatch).is_err()
    );
}

#[test]
fn server_shared_engine_chat_completion_receipt_rejects_bitnet_route_without_qk256_evidence() {
    let mut bitnet_route = server_shared_engine_chat_completion_receipt();
    bitnet_route["selected_route"] = json!("bitnet_qk256_cuda");
    bitnet_route["dense_regular_llm_cuda_inference_claimed"] = json!(false);
    bitnet_route["bitnet_packed_i2s_qk256_proof"] = json!(true);

    assert!(validate_server_shared_engine_chat_completion_receipt_json(&bitnet_route).is_err());
}

#[test]
fn committed_dense_gguf_descriptor_inspection_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-gguf-descriptor-inspection.json"
    ))
    .unwrap();

    validate_dense_gguf_tensor_descriptor_inspection_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_gguf_linear_fixture_extraction_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-gguf-linear-fixture-extraction.json"
    ))
    .unwrap();

    validate_dense_gguf_linear_fixture_extraction_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_gguf_one_layer_execution_plan_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-one-layer-plan-qwen25-q8.json"
    ))
    .unwrap();

    validate_dense_gguf_one_layer_execution_plan_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_gguf_norm_fixture_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-norm-fixture-qwen25-q8.json"
    ))
    .unwrap();

    validate_dense_gguf_norm_fixture_extraction_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_gguf_norm_cuda_parity_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-rmsnorm-cuda-parity-qwen25-q8.json"
    ))
    .unwrap();

    validate_dense_gguf_norm_cuda_parity_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_gguf_rope_cuda_parity_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-rope-cuda-parity-qwen25-q8.json"
    ))
    .unwrap();

    validate_dense_gguf_rope_cuda_parity_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_gguf_attention_score_fixture_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-score-fixture-qwen25-q8.json"
    ))
    .unwrap();

    validate_dense_gguf_attention_score_fixture_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_gguf_attention_score_cuda_parity_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-score-cuda-parity-qwen25-q8.json"
    ))
    .unwrap();

    validate_dense_gguf_attention_score_cuda_parity_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_gguf_attention_softmax_fixture_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-softmax-fixture-qwen25-q8.json"
    ))
    .unwrap();

    validate_dense_gguf_attention_softmax_fixture_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_gguf_attention_softmax_cuda_parity_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-softmax-cuda-parity-qwen25-q8.json"
    ))
    .unwrap();

    validate_dense_gguf_attention_softmax_cuda_parity_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_gguf_attention_v_mix_fixture_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-v-mix-fixture-qwen25-q8.json"
    ))
    .unwrap();

    validate_dense_gguf_attention_v_mix_fixture_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_gguf_attention_v_mix_cuda_parity_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-v-mix-cuda-parity-qwen25-q8.json"
    ))
    .unwrap();

    validate_dense_gguf_attention_v_mix_cuda_parity_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_gguf_mlp_activation_fixture_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-mlp-activation-fixture-qwen25-q8.json"
    ))
    .unwrap();

    validate_dense_gguf_mlp_activation_fixture_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_gguf_mlp_activation_cuda_parity_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-mlp-activation-cuda-parity-qwen25-q8.json"
    ))
    .unwrap();

    validate_dense_gguf_mlp_activation_cuda_parity_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_gguf_model_boundary_fixtures_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-model-boundary-fixtures-qwen25-q8.json"
    ))
    .unwrap();

    validate_dense_gguf_model_boundary_fixtures_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_gguf_kv_cache_policy_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-kv-cache-policy-qwen25-q8.json"
    ))
    .unwrap();

    validate_dense_gguf_kv_cache_policy_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn committed_dense_gguf_sampling_policy_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-sampling-policy-qwen25-q8.json"
    ))
    .unwrap();

    validate_dense_gguf_sampling_policy_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_attention_score_fixture_rejects_cuda_parity_claim() {
    let mut receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-score-fixture-qwen25-q8.json"
    ))
    .unwrap();
    receipt["claim_boundary"]["dense_regular_llm_cuda_claimed"] = json!(true);

    let err =
        validate_dense_gguf_attention_score_fixture_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("dense_regular_llm_cuda_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_attention_softmax_fixture_rejects_cuda_claim() {
    let mut receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-softmax-fixture-qwen25-q8.json"
    ))
    .unwrap();
    receipt["claim_boundary"]["dense_regular_llm_cuda_claimed"] = json!(true);

    let err = validate_dense_gguf_attention_softmax_fixture_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("dense_regular_llm_cuda_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_attention_v_mix_fixture_rejects_cuda_claim() {
    let mut receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-v-mix-fixture-qwen25-q8.json"
    ))
    .unwrap();
    receipt["claim_boundary"]["dense_regular_llm_cuda_claimed"] = json!(true);

    let err =
        validate_dense_gguf_attention_v_mix_fixture_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("dense_regular_llm_cuda_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_mlp_activation_fixture_rejects_cuda_claim() {
    let mut receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-mlp-activation-fixture-qwen25-q8.json"
    ))
    .unwrap();
    receipt["claim_boundary"]["dense_regular_llm_cuda_claimed"] = json!(true);

    let err =
        validate_dense_gguf_mlp_activation_fixture_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("dense_regular_llm_cuda_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_attention_score_cuda_parity_rejects_inference_claim() {
    let mut receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-score-cuda-parity-qwen25-q8.json"
    ))
    .unwrap();
    receipt["claim_boundary"]["dense_gguf_inference_claimed"] = json!(true);

    let err = validate_dense_gguf_attention_score_cuda_parity_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("dense_gguf_inference_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_attention_softmax_cuda_parity_rejects_inference_claim() {
    let mut receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-softmax-cuda-parity-qwen25-q8.json"
    ))
    .unwrap();
    receipt["claim_boundary"]["dense_gguf_inference_claimed"] = json!(true);

    let err = validate_dense_gguf_attention_softmax_cuda_parity_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("dense_gguf_inference_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_attention_v_mix_cuda_parity_rejects_inference_claim() {
    let mut receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-v-mix-cuda-parity-qwen25-q8.json"
    ))
    .unwrap();
    receipt["claim_boundary"]["dense_gguf_inference_claimed"] = json!(true);

    let err = validate_dense_gguf_attention_v_mix_cuda_parity_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("dense_gguf_inference_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_attention_v_mix_cuda_parity_rejects_bitnet_proof_claim() {
    let mut receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-v-mix-cuda-parity-qwen25-q8.json"
    ))
    .unwrap();
    receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"] = json!(true);

    let err = validate_dense_gguf_attention_v_mix_cuda_parity_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("bitnet_packed_i2s_qk256_proof"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_mlp_activation_cuda_parity_rejects_inference_claim() {
    let mut receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-mlp-activation-cuda-parity-qwen25-q8.json"
    ))
    .unwrap();
    receipt["claim_boundary"]["dense_gguf_inference_claimed"] = json!(true);

    let err = validate_dense_gguf_mlp_activation_cuda_parity_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("dense_gguf_inference_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_mlp_activation_cuda_parity_rejects_bitnet_proof_claim() {
    let mut receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-mlp-activation-cuda-parity-qwen25-q8.json"
    ))
    .unwrap();
    receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"] = json!(true);

    let err = validate_dense_gguf_mlp_activation_cuda_parity_receipt_json(&receipt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("bitnet_packed_i2s_qk256_proof"), "unexpected error: {err}");
}

#[test]
fn strict_cuda_smoke_rejects_top_level_fallback() {
    let mut receipt = valid_smoke_receipt();
    receipt["fallback_used"] = json!(true);
    receipt["fallback_reason"] = json!("selected CPU fallback");

    let err = validate_cuda_smoke_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("fallback_used"), "unexpected error: {err}");
}

#[test]
fn strict_cuda_smoke_rejects_fallback_invocations() {
    let mut receipt = valid_smoke_receipt();
    receipt["kernel_stats"][0]["fallback_invocations"] = json!(1);

    let err = validate_cuda_smoke_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("fallback_invocations"), "unexpected error: {err}");
}

#[test]
fn strict_cuda_smoke_rejects_zero_kernel_invocations() {
    let mut receipt = valid_smoke_receipt();
    receipt["kernel_stats"][0]["invocations"] = json!(0);

    let err = validate_cuda_smoke_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("invocations"), "unexpected error: {err}");
}

#[test]
fn strict_cuda_smoke_rejects_missing_transfer_bytes() {
    let mut receipt = valid_smoke_receipt();
    receipt["kernel_stats"][0]["host_to_device_bytes"] = Value::Null;

    let err = validate_cuda_smoke_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("host_to_device_bytes"), "unexpected error: {err}");
}

#[test]
fn strict_cuda_smoke_rejects_generic_cuda_backend() {
    let mut receipt = valid_smoke_receipt();
    receipt["selected_backend"] = json!("cuda");

    let err = validate_cuda_smoke_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("selected_backend"), "unexpected error: {err}");
}

#[test]
fn strict_cuda_parity_rejects_failed_parity() {
    let mut receipt = valid_parity_receipt();
    receipt["parity"]["passed"] = json!(false);
    receipt["result"] = json!("fail");

    let err = validate_cuda_parity_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("result"), "unexpected error: {err}");
}

#[test]
fn strict_cuda_parity_rejects_missing_runtime_identity() {
    let mut receipt = valid_parity_receipt();
    receipt["cuda"]["driver_version"] = Value::Null;

    let err = validate_cuda_parity_receipt_json(&receipt).unwrap_err().to_string();
    assert!(err.contains("driver_version"), "unexpected error: {err}");
}

#[test]
fn dense_regular_llm_cuda_receipt_validates_with_separate_label() {
    let receipt = valid_dense_regular_llm_cuda_receipt();

    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap();
}

#[test]
fn dense_regular_llm_cuda_receipt_requires_dense_execution_plan() {
    let mut receipt = valid_dense_regular_llm_cuda_receipt();
    receipt.as_object_mut().expect("receipt object").remove("execution_plan");

    let err = validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("execution_plan"), "unexpected error: {err}");
}

#[test]
fn dense_regular_llm_cuda_receipt_rejects_bitnet_execution_plan_route() {
    let mut receipt = valid_dense_regular_llm_cuda_receipt();
    receipt["execution_plan"]["selected_route"] = json!("bitnet_qk256_cuda");
    receipt["execution_plan"]["bitnet_packed_qk256_cuda"] = json!(true);
    receipt["execution_plan"]["dense_regular_llm_cuda"] = json!(false);
    receipt["execution_plan"]["cuda_bitnet_qk256_ops"] = json!(1);
    receipt["execution_plan"]["cuda_dense_regular_llm_ops"] = json!(0);

    let err = validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("selected_route"), "unexpected error: {err}");
}

#[test]
fn dense_regular_llm_cuda_receipt_rejects_bitnet_kernel_label() {
    let mut receipt = valid_dense_regular_llm_cuda_receipt();
    receipt["kernel_stats"][0]["kernel_id"] = json!("qk256_gemv_cuda");

    let err = validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("kernel_stats[0].kernel_id"), "unexpected error: {err}");
}

#[test]
fn dense_regular_llm_cuda_receipt_rejects_bitnet_model_family() {
    let mut receipt = valid_dense_regular_llm_cuda_receipt();
    receipt["model"]["model_family"] = json!("bitnet");

    let err = validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("model.model_family"), "unexpected error: {err}");
}

#[test]
fn dense_regular_llm_cuda_receipt_rejects_speedup_claim() {
    let mut receipt = valid_dense_regular_llm_cuda_receipt();
    receipt["speedup_claim"] = json!(true);
    receipt["claim_boundary"]["speedup_claim"] = json!(true);

    let err = validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("speedup_claim"), "unexpected error: {err}");
}

#[test]
fn dense_regular_llm_cuda_receipt_requires_passing_cpu_cuda_parity() {
    let mut receipt = valid_dense_regular_llm_cuda_receipt();
    receipt["parity"]["passed"] = json!(false);

    let err = validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("`passed`"), "unexpected error: {err}");
}

#[test]
fn dense_regular_llm_cuda_receipt_rejects_bitnet_parity_fixture() {
    let mut receipt = valid_dense_regular_llm_cuda_receipt();
    receipt["parity"]["fixture_id"] = json!("qk256_bitnet_fixture");

    let err = validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("parity.fixture_id"), "unexpected error: {err}");
}

#[test]
fn dense_regular_llm_cuda_receipt_cannot_satisfy_bitnet_packed_proof() {
    let receipt = valid_dense_regular_llm_cuda_receipt();

    let err =
        reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err().to_string();

    assert!(
        err.contains("cannot satisfy BitNet packed I2_S/QK256 proof"),
        "unexpected error: {err}"
    );
    validate_cuda_smoke_receipt_json(&receipt).unwrap_err();
    validate_cuda_parity_receipt_json(&receipt).unwrap_err();
}

#[test]
fn dense_regular_llm_cuda_residency_receipt_validates() {
    let receipt = valid_dense_regular_llm_cuda_residency_receipt();

    validate_dense_regular_llm_cuda_tensor_residency_receipt_json(&receipt).unwrap();
}

#[test]
fn dense_regular_llm_cuda_residency_rejects_missing_tensor_section() {
    let receipt = valid_dense_regular_llm_cuda_receipt();

    let err = validate_dense_regular_llm_cuda_tensor_residency_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("claim"), "unexpected error: {err}");
}

#[test]
fn dense_regular_llm_cuda_residency_rejects_persistent_handle_claim() {
    let mut receipt = valid_dense_regular_llm_cuda_residency_receipt();
    receipt["tensor_residency"]["allocation"]["persistent_handles_claimed"] = json!(true);
    receipt["claim_boundary"]["persistent_session_residency_claimed"] = json!(true);

    let err = validate_dense_regular_llm_cuda_tensor_residency_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("persistent_session_residency_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_regular_llm_cuda_residency_rejects_transfer_mismatch() {
    let mut receipt = valid_dense_regular_llm_cuda_residency_receipt();
    receipt["tensor_residency"]["transfer_accounting"]["host_to_device_bytes"] = json!(1);

    let err = validate_dense_regular_llm_cuda_tensor_residency_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("host_to_device_bytes"), "unexpected error: {err}");
}

#[test]
fn dense_regular_llm_cuda_persistent_residency_receipt_validates() {
    let receipt = valid_dense_regular_llm_cuda_persistent_residency_receipt();

    validate_dense_regular_llm_cuda_persistent_residency_receipt_json(&receipt).unwrap();
}

#[test]
fn dense_regular_llm_cuda_persistent_residency_rejects_single_launch() {
    let mut receipt = valid_dense_regular_llm_cuda_persistent_residency_receipt();
    receipt["kernel_stats"][0]["invocations"] = json!(1);
    receipt["kernel_stats"][0]["kernel_launches"] = json!(1);
    receipt["parity"]["runs"] = json!(1);
    receipt["persistent_session"]["repeated_runs"] = json!(1);
    receipt["persistent_session"]["kernel_launches"] = json!(1);

    let err = validate_dense_regular_llm_cuda_persistent_residency_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("at least two invocations"), "unexpected error: {err}");
}

#[test]
fn dense_regular_llm_cuda_persistent_residency_rejects_per_run_uploads() {
    let mut receipt = valid_dense_regular_llm_cuda_persistent_residency_receipt();
    receipt["persistent_session"]["per_run_host_to_device_bytes"] = json!(40);
    receipt["tensor_residency"]["per_run_host_to_device_bytes"] = json!(40);

    let err = validate_dense_regular_llm_cuda_persistent_residency_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("per_run_host_to_device_bytes"), "unexpected error: {err}");
}

#[test]
fn dense_regular_llm_cuda_persistent_residency_rejects_full_residency_claim() {
    let mut receipt = valid_dense_regular_llm_cuda_persistent_residency_receipt();
    receipt["claim_boundary"]["full_cuda_residency_claimed"] = json!(true);
    receipt["persistent_session"]["full_cuda_residency_claimed"] = json!(true);
    receipt["tensor_residency"]["full_cuda_residency_claimed"] = json!(true);

    let err = validate_dense_regular_llm_cuda_persistent_residency_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("full_cuda_residency_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_descriptor_inspection_receipt_validates() {
    let receipt = valid_dense_gguf_descriptor_inspection_receipt();

    validate_dense_gguf_tensor_descriptor_inspection_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_descriptor_inspection_rejects_missing_required_role() {
    let mut receipt = valid_dense_gguf_descriptor_inspection_receipt();
    receipt["descriptor_inspection"]["descriptors"]
        .as_array_mut()
        .expect("descriptors array")
        .retain(|descriptor| descriptor["role"] != json!("mlp_down"));

    let err = validate_dense_gguf_tensor_descriptor_inspection_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("mlp_down"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_descriptor_inspection_rejects_cuda_claim_leakage() {
    let mut receipt = valid_dense_gguf_descriptor_inspection_receipt();
    receipt["claim_boundary"]["dense_regular_llm_cuda_claimed"] = json!(true);
    receipt["descriptor_inspection"]["dense_regular_llm_cuda_claimed"] = json!(true);

    let err = validate_dense_gguf_tensor_descriptor_inspection_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("dense_regular_llm_cuda_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_descriptor_inspection_rejects_bitnet_tensor_marker() {
    let mut receipt = valid_dense_gguf_descriptor_inspection_receipt();
    receipt["descriptor_inspection"]["descriptors"][2]["tensor_type"] = json!("i2_s");
    receipt["descriptor_inspection"]["quantization_families"] = json!(["q8_0", "i2_s"]);

    let err = validate_dense_gguf_tensor_descriptor_inspection_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("BitNet packed I2_S/QK256"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_linear_fixture_extraction_receipt_validates() {
    let receipt = valid_dense_gguf_linear_fixture_extraction_receipt();

    validate_dense_gguf_linear_fixture_extraction_receipt_json(&receipt).unwrap();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_linear_fixture_rejects_non_linear_role() {
    let mut receipt = valid_dense_gguf_linear_fixture_extraction_receipt();
    receipt["linear_fixture"]["role"] = json!("attention_norm");

    let err = validate_dense_gguf_linear_fixture_extraction_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("extractable dense linear role"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_linear_fixture_rejects_cuda_parity_claim_leakage() {
    let mut receipt = valid_dense_gguf_linear_fixture_extraction_receipt();
    receipt["claim_boundary"]["cpu_cuda_parity_claimed"] = json!(true);

    let err = validate_dense_gguf_linear_fixture_extraction_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("cpu_cuda_parity_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_linear_fixture_rejects_bad_matrix_count() {
    let mut receipt = valid_dense_gguf_linear_fixture_extraction_receipt();
    receipt["linear_fixture"]["value_count"] = json!(11);

    let err = validate_dense_gguf_linear_fixture_extraction_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("value_count"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_linear_fixture_rejects_source_shape_mismatch() {
    let mut receipt = valid_dense_gguf_linear_fixture_extraction_receipt();
    receipt["linear_fixture"]["source_shape"] = json!([3, 4]);

    let err = validate_dense_gguf_linear_fixture_extraction_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("source_shape"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_linear_fixture_rejects_iq2s_tensor_type() {
    let mut receipt = valid_dense_gguf_linear_fixture_extraction_receipt();
    receipt["linear_fixture"]["tensor_type"] = json!("iq2_s");

    let err = validate_dense_gguf_linear_fixture_extraction_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("BitNet packed I2_S/QK256"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_linear_cuda_parity_receipt_validates() {
    let receipt = valid_dense_gguf_linear_cuda_parity_receipt();

    validate_dense_gguf_linear_cuda_parity_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_linear_cuda_parity_rejects_failed_parity() {
    let mut receipt = valid_dense_gguf_linear_cuda_parity_receipt();
    receipt["parity"]["passed"] = json!(false);

    let err =
        validate_dense_gguf_linear_cuda_parity_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("`passed`"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_linear_cuda_parity_rejects_dense_inference_claim() {
    let mut receipt = valid_dense_gguf_linear_cuda_parity_receipt();
    receipt["claim_boundary"]["dense_gguf_inference_claimed"] = json!(true);
    receipt["linear_fixture"]["dense_gguf_inference_claimed"] = json!(true);

    let err =
        validate_dense_gguf_linear_cuda_parity_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("dense_gguf_inference_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_linear_cuda_parity_rejects_bitnet_source_marker() {
    let mut receipt = valid_dense_gguf_linear_cuda_parity_receipt();
    receipt["linear_fixture"]["tensor_type"] = json!("i2_s");

    let err =
        validate_dense_gguf_linear_cuda_parity_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("BitNet packed I2_S/QK256"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_linear_cuda_parity_rejects_transfer_mismatch() {
    let mut receipt = valid_dense_gguf_linear_cuda_parity_receipt();
    receipt["tensor_residency"]["transfer_accounting"]["device_to_host_bytes"] = json!(4);

    let err =
        validate_dense_gguf_linear_cuda_parity_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("device_to_host_bytes"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_linear_role_sweep_cuda_parity_receipt_validates() {
    let receipt = valid_dense_gguf_linear_role_sweep_cuda_parity_receipt();

    validate_dense_gguf_linear_role_sweep_cuda_parity_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_linear_role_sweep_rejects_bitnet_proof_claim() {
    let mut receipt = valid_dense_gguf_linear_role_sweep_cuda_parity_receipt();
    receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"] = json!(true);
    receipt["linear_role_sweep"]["bitnet_packed_i2s_qk256_proof"] = json!(true);

    let err = validate_dense_gguf_linear_role_sweep_cuda_parity_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("bitnet_packed_i2s_qk256_proof"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_linear_role_sweep_rejects_count_mismatch() {
    let mut receipt = valid_dense_gguf_linear_role_sweep_cuda_parity_receipt();
    receipt["execution_plan"]["cuda_dense_regular_llm_ops"] = json!(1);

    let err = validate_dense_gguf_linear_role_sweep_cuda_parity_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("cuda_dense_regular_llm_ops"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_linear_role_sweep_rejects_duplicate_roles() {
    let mut receipt = valid_dense_gguf_linear_role_sweep_cuda_parity_receipt();
    receipt["linear_role_sweep"]["covered_roles"] = json!(["attention_q", "attention_q"]);

    let err = validate_dense_gguf_linear_role_sweep_cuda_parity_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("duplicate"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_one_layer_execution_plan_receipt_validates_gap() {
    let receipt = valid_dense_gguf_one_layer_execution_plan_receipt();

    validate_dense_gguf_one_layer_execution_plan_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_one_layer_plan_rejects_inference_claim() {
    let mut receipt = valid_dense_gguf_one_layer_execution_plan_receipt();
    receipt["claim_boundary"]["dense_gguf_one_layer_inference_claimed"] = json!(true);
    receipt["one_layer_plan"]["one_layer_inference_claimed"] = json!(true);

    let err = validate_dense_gguf_one_layer_execution_plan_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("one_layer_inference_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_one_layer_plan_rejects_unrouted_strict_ops_after_mlp_promotion() {
    let mut receipt = valid_dense_gguf_one_layer_execution_plan_receipt();
    receipt["execution_plan"]["unsupported_ops"] = json!(1);
    receipt["execution_plan"]["strict_cuda_ready"] = json!(false);

    let err = validate_dense_gguf_one_layer_execution_plan_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(
        err.contains("unsupported_ops") || err.contains("strict_cuda_ready"),
        "unexpected error: {err}"
    );
}

#[test]
fn dense_gguf_one_layer_plan_requires_gap_audit() {
    let mut receipt = valid_dense_gguf_one_layer_execution_plan_receipt();
    receipt.as_object_mut().expect("receipt object").remove("gap_audit");

    let err = validate_dense_gguf_one_layer_execution_plan_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("gap_audit"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_one_layer_gap_audit_rejects_cpu_fallback_policy_change() {
    let mut receipt = valid_dense_gguf_one_layer_execution_plan_receipt();
    receipt["gap_audit"]["strict_cuda_rejects_cpu_fallback"] = json!(false);

    let err = validate_dense_gguf_one_layer_execution_plan_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("strict_cuda_rejects_cpu_fallback"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_all_layer_execution_plan_receipt_validates_transformer_scope() {
    let receipt = valid_dense_gguf_all_layer_execution_plan_receipt();

    validate_dense_gguf_all_layer_execution_plan_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_all_layer_plan_rejects_inference_claim() {
    let mut receipt = valid_dense_gguf_all_layer_execution_plan_receipt();
    receipt["claim_boundary"]["dense_gguf_inference_claimed"] = json!(true);
    receipt["all_layer_plan"]["dense_gguf_inference_claimed"] = json!(true);

    let err = validate_dense_gguf_all_layer_execution_plan_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("dense_gguf_inference_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_all_layer_plan_rejects_missing_boundary_gap() {
    let mut receipt = valid_dense_gguf_all_layer_execution_plan_receipt();
    receipt["model_boundary_gaps"]["gaps"]
        .as_array_mut()
        .expect("gaps array")
        .retain(|gap| gap["gap"] != "kv_cache_policy");

    let err = validate_dense_gguf_all_layer_execution_plan_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("model_boundary_gaps missing required gaps"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_all_layer_plan_rejects_layer_difference() {
    let mut receipt = valid_dense_gguf_all_layer_execution_plan_receipt();
    receipt["all_layer_plan"]["layer_plan_matches_layer0"] = json!(false);
    receipt["all_layer_plan"]["layer_differences"] =
        json!([{ "layer_index": 1, "reason": "operation_signature_differs_from_layer0" }]);

    let err = validate_dense_gguf_all_layer_execution_plan_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("layer_plan_matches_layer0"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_all_layer_plan_rejects_unsupported_ops() {
    let mut receipt = valid_dense_gguf_all_layer_execution_plan_receipt();
    receipt["execution_plan"]["unsupported_ops"] = json!(1);
    receipt["execution_plan"]["strict_cuda_ready"] = json!(false);
    receipt["all_layer_plan"]["unsupported_strict_cuda_ops_total"] = json!(1);

    let err = validate_dense_gguf_all_layer_execution_plan_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("unsupported_ops"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_all_layer_plan_rejects_forged_operation_signature() {
    let mut receipt = valid_dense_gguf_all_layer_execution_plan_receipt();
    receipt["all_layer_plan"]["layers"][0]["operation_signature_sha256"] = json!("0".repeat(64));

    let err = validate_dense_gguf_all_layer_execution_plan_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(
        err.contains("operation_signature_sha256 must match operations"),
        "unexpected error: {err}"
    );
}

#[test]
fn dense_gguf_all_layer_plan_rejects_wrong_operation_role_sequence() {
    let mut receipt = valid_dense_gguf_all_layer_execution_plan_receipt();
    receipt["all_layer_plan"]["layers"][0]["operations"][1]["role"] = json!("attention_q_dup");
    let operations = receipt["all_layer_plan"]["layers"][0]["operations"].clone();
    receipt["all_layer_plan"]["layers"][0]["operation_signature_sha256"] =
        json!(dense_all_layer_operation_signature_sha256(&operations));

    let err = validate_dense_gguf_all_layer_execution_plan_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("role"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_model_boundary_fixtures_receipt_validates_without_inference_claims() {
    let receipt = valid_dense_gguf_model_boundary_fixtures_receipt();

    validate_dense_gguf_model_boundary_fixtures_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_model_boundary_fixtures_rejects_one_token_claim() {
    let mut receipt = valid_dense_gguf_model_boundary_fixtures_receipt();
    receipt["claim_boundary"]["qwen_one_token_cuda_claimed"] = json!(true);
    receipt["model_boundary_fixtures"]["qwen_one_token_cuda_claimed"] = json!(true);

    let err =
        validate_dense_gguf_model_boundary_fixtures_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("qwen_one_token_cuda_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_model_boundary_fixtures_rejects_missing_top_k() {
    let mut receipt = valid_dense_gguf_model_boundary_fixtures_receipt();
    receipt["model_boundary_fixtures"]["lm_head_logits"]["top_k_entries"] = json!([]);

    let err =
        validate_dense_gguf_model_boundary_fixtures_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("top_k_entries"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_model_boundary_fixtures_rejects_dimension_mismatch() {
    let mut receipt = valid_dense_gguf_model_boundary_fixtures_receipt();
    receipt["model_boundary_fixtures"]["token_embedding"]["output_len"] = json!(15);

    let err =
        validate_dense_gguf_model_boundary_fixtures_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("output_len"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_model_boundary_fixtures_rejects_logits_vocab_mismatch() {
    let mut receipt = valid_dense_gguf_model_boundary_fixtures_receipt();
    receipt["model_boundary_fixtures"]["lm_head_logits"]["logits_len"] = json!(5);

    let err =
        validate_dense_gguf_model_boundary_fixtures_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("vocab_size"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_model_boundary_fixtures_rejects_kv_or_sampling_claim() {
    let mut receipt = valid_dense_gguf_model_boundary_fixtures_receipt();
    receipt["claim_boundary"]["kv_cache_policy_claimed"] = json!(true);

    let err =
        validate_dense_gguf_model_boundary_fixtures_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("kv_cache_policy_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_model_boundary_fixtures_rejects_bitnet_proof_claim() {
    let mut receipt = valid_dense_gguf_model_boundary_fixtures_receipt();
    receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"] = json!(true);
    receipt["model_boundary_fixtures"]["bitnet_packed_i2s_qk256_proof"] = json!(true);

    let err =
        validate_dense_gguf_model_boundary_fixtures_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("bitnet_packed_i2s_qk256_proof"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_kv_cache_policy_receipt_validates_without_inference_claims() {
    let receipt = valid_dense_gguf_kv_cache_policy_receipt();

    validate_dense_gguf_kv_cache_policy_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_kv_cache_policy_rejects_runtime_residency_claim() {
    let mut receipt = valid_dense_gguf_kv_cache_policy_receipt();
    receipt["claim_boundary"]["kv_cache_cuda_residency_claimed"] = json!(true);
    receipt["kv_cache_policy"]["kv_cache_cuda_residency_claimed"] = json!(true);

    let err = validate_dense_gguf_kv_cache_policy_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("kv_cache_cuda_residency_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_kv_cache_policy_rejects_sampling_claim() {
    let mut receipt = valid_dense_gguf_kv_cache_policy_receipt();
    receipt["claim_boundary"]["sampling_integration_claimed"] = json!(true);
    receipt["kv_cache_policy"]["sampling_integration_claimed"] = json!(true);

    let err = validate_dense_gguf_kv_cache_policy_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("sampling_integration_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_kv_cache_policy_rejects_bad_byte_math() {
    let mut receipt = valid_dense_gguf_kv_cache_policy_receipt();
    receipt["kv_cache_policy"]["kv_bytes_per_token_all_layers"] = json!(1234);

    let err = validate_dense_gguf_kv_cache_policy_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("kv_bytes_per_token_all_layers"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_kv_cache_policy_rejects_bitnet_proof_claim() {
    let mut receipt = valid_dense_gguf_kv_cache_policy_receipt();
    receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"] = json!(true);
    receipt["kv_cache_policy"]["bitnet_packed_i2s_qk256_proof"] = json!(true);

    let err = validate_dense_gguf_kv_cache_policy_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("bitnet_packed_i2s_qk256_proof"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_sampling_policy_receipt_validates_without_inference_claims() {
    let receipt = valid_dense_gguf_sampling_policy_receipt();

    validate_dense_gguf_sampling_policy_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_sampling_policy_rejects_sampling_integration_claim() {
    let mut receipt = valid_dense_gguf_sampling_policy_receipt();
    receipt["claim_boundary"]["sampling_integration_claimed"] = json!(true);
    receipt["sampling_policy"]["sampling_integration_claimed"] = json!(true);

    let err = validate_dense_gguf_sampling_policy_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("sampling_integration_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_sampling_policy_rejects_qwen_one_token_claim() {
    let mut receipt = valid_dense_gguf_sampling_policy_receipt();
    receipt["claim_boundary"]["qwen_one_token_cuda_claimed"] = json!(true);
    receipt["sampling_policy"]["qwen_one_token_cuda_claimed"] = json!(true);

    let err = validate_dense_gguf_sampling_policy_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("qwen_one_token_cuda_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_sampling_policy_rejects_bad_logits_transfer_byte_math() {
    let mut receipt = valid_dense_gguf_sampling_policy_receipt();
    receipt["sampling_policy"]["logits_transfer_bytes_per_step_estimate"] = json!(1234);

    let err = validate_dense_gguf_sampling_policy_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("logits_transfer_bytes_per_step_estimate"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_sampling_policy_rejects_bitnet_proof_claim() {
    let mut receipt = valid_dense_gguf_sampling_policy_receipt();
    receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"] = json!(true);
    receipt["sampling_policy"]["bitnet_packed_i2s_qk256_proof"] = json!(true);

    let err = validate_dense_gguf_sampling_policy_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("bitnet_packed_i2s_qk256_proof"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_one_token_strict_cuda_proof_receipt_validates() {
    let receipt = valid_dense_gguf_qwen_one_token_strict_cuda_proof_receipt();

    validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_qwen_one_token_accepts_qwen3_06b_model_identity()
-> Result<(), Box<dyn std::error::Error>> {
    let mut receipt = valid_dense_gguf_qwen_one_token_strict_cuda_proof_receipt();
    receipt["model"]["id"] = json!("qwen3-0.6b-instruct-q8_0");
    receipt["model"]["file"] = json!("Qwen3-0.6B-Q8_0.gguf");
    receipt["model"]["architecture"] = json!("qwen3");
    receipt["model"]["sha256"] =
        json!("9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031");
    receipt["parity"]["fixture_id"] = json!("qwen3-0.6b-instruct-q8_0-one-token-greedy");

    validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(&receipt)?;
    Ok(())
}

#[test]
fn dense_gguf_qwen_one_token_rejects_sampling_policy_only_receipt() {
    let receipt = valid_dense_gguf_sampling_policy_receipt();

    let err = validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("artifact_kind"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_one_token_requires_prerequisite_receipts() {
    let mut receipt = valid_dense_gguf_qwen_one_token_strict_cuda_proof_receipt();
    receipt["prerequisite_receipts"]["all_required_receipts_verified"] = json!(false);

    let err = validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("all_required_receipts_verified"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_one_token_rejects_selected_token_mismatch() {
    let mut receipt = valid_dense_gguf_qwen_one_token_strict_cuda_proof_receipt();
    receipt["one_token_proof"]["cuda_selected_token_id"] = json!(7);

    let err = validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("cpu_selected_token_id"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_one_token_rejects_prompt_hash_mismatch() {
    let mut receipt = valid_dense_gguf_qwen_one_token_strict_cuda_proof_receipt();
    receipt["one_token_proof"]["prompt_token_ids_sha256"] = json!(format!("{:064x}", 90));

    let err = validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("prompt_token_ids_sha256"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_one_token_rejects_top_k_hash_mismatch() {
    let mut receipt = valid_dense_gguf_qwen_one_token_strict_cuda_proof_receipt();
    receipt["one_token_proof"]["cuda_logits_top_k_sha256"] = json!(format!("{:064x}", 91));

    let err = validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("cuda_logits_top_k_sha256"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_one_token_rejects_unverified_model_identity() {
    let mut receipt = valid_dense_gguf_qwen_one_token_strict_cuda_proof_receipt();
    receipt["model"]["sha256"] = json!(format!("{:064x}", 92));

    let err = validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("model.sha256") || err.contains("sha256"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_one_token_rejects_missing_timing_evidence() {
    let mut receipt = valid_dense_gguf_qwen_one_token_strict_cuda_proof_receipt();
    receipt["timing"]["total_ms"] = Value::Null;

    let err = validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("total_ms"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_one_token_rejects_missing_transfer_timing_source_fields() {
    let mut receipt = valid_dense_gguf_qwen_one_token_strict_cuda_proof_receipt();
    receipt["timing"]["device_to_host_ms"] = Value::Null;

    let err = validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("device_to_host_ms"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_one_token_rejects_missing_h2d_timing_envelope() {
    let mut receipt = valid_dense_gguf_qwen_one_token_strict_cuda_proof_receipt();
    receipt["timing"]["host_to_device_ms"] = Value::Null;

    let err = validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("host_to_device_ms"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_one_token_rejects_h2d_timing_accounting_mismatch() {
    let mut receipt = valid_dense_gguf_qwen_one_token_strict_cuda_proof_receipt();
    receipt["tensor_residency"]["transfer_accounting"]["host_to_device_ms"] = json!(99.0);

    let err = validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("host_to_device_ms"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_one_token_rejects_short_decode_claim() {
    let mut receipt = valid_dense_gguf_qwen_one_token_strict_cuda_proof_receipt();
    receipt["claim_boundary"]["qwen_short_decode_cuda_claimed"] = json!(true);

    let err = validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("qwen_short_decode_cuda_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_one_token_rejects_speedup_and_full_residency_claims() {
    let mut receipt = valid_dense_gguf_qwen_one_token_strict_cuda_proof_receipt();
    receipt["speedup_claim"] = json!(true);
    receipt["claim_boundary"]["full_cuda_residency_claimed"] = json!(true);

    let err = validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("speedup_claim"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_one_token_rejects_bitnet_proof_claim() {
    let mut receipt = valid_dense_gguf_qwen_one_token_strict_cuda_proof_receipt();
    receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"] = json!(true);

    let err = validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("bitnet_packed_i2s_qk256_proof"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_validates() {
    let receipt = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();

    validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_qwen_short_decode_accepts_qwen3_06b_model_identity()
-> Result<(), Box<dyn std::error::Error>> {
    let mut receipt = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();
    receipt["model"]["id"] = json!("qwen3-0.6b-instruct-q8_0");
    receipt["model"]["file"] = json!("Qwen3-0.6B-Q8_0.gguf");
    receipt["model"]["architecture"] = json!("qwen3");
    receipt["model"]["sha256"] =
        json!("9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031");
    receipt["parity"]["fixture_id"] = json!("qwen3-0.6b-instruct-q8_0-short-decode-greedy");

    validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&receipt)?;
    Ok(())
}

#[test]
fn dense_gguf_qwen_short_decode_accepts_qwen3_32_capture_profile()
-> Result<(), Box<dyn std::error::Error>> {
    let receipt = valid_dense_gguf_qwen_short_decode_32_capture_receipt();

    validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&receipt)?;
    Ok(())
}

#[test]
fn dense_gguf_qwen_short_decode_rejects_32_tokens_without_qwen3_capture_profile() {
    let mut receipt = valid_dense_gguf_qwen_short_decode_32_capture_receipt();
    receipt["short_decode_proof"]["profile_id"] = json!("short_decode");
    receipt["short_decode_proof"]["proof_scope"] = json!("qwen_strict_short_decode_greedy");

    let err = validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("qwen3_short_decode_32"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_warm_decode_128_receipt_validates() -> Result<(), Box<dyn std::error::Error>> {
    let receipt = valid_dense_gguf_qwen_warm_decode_strict_cuda_proof_receipt()?;

    validate_dense_gguf_qwen_warm_decode_strict_cuda_proof_receipt_json(&receipt)?;
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
    Ok(())
}

#[test]
fn dense_gguf_qwen_warm_decode_128_requires_warm_context_reuse()
-> Result<(), Box<dyn std::error::Error>> {
    let mut receipt = valid_dense_gguf_qwen_warm_decode_strict_cuda_proof_receipt()?;
    receipt["warm_context_proof"]["warm_context_reused"] = json!(false);

    let err = validate_dense_gguf_qwen_warm_decode_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("warm_context_reused"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn dense_gguf_qwen_short_decode_rejects_one_token_only_receipt() {
    let receipt = valid_dense_gguf_qwen_one_token_strict_cuda_proof_receipt();

    let err = validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("artifact_kind"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_short_decode_requires_one_token_prerequisite() {
    let mut receipt = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();
    receipt["prerequisite_receipts"]["one_token_proof_claimed"] = json!(false);

    let err = validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("one_token_proof_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_short_decode_rejects_generated_token_mismatch() {
    let mut receipt = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();
    receipt["short_decode_proof"]["cuda_generated_token_ids"][2] = json!(999);

    let err = validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("cpu_generated_token_ids"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_short_decode_rejects_missing_transfer_timing_source_fields() {
    let mut receipt = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();
    receipt["tensor_residency"]["transfer_accounting"]["device_to_host_ms"] = Value::Null;

    let err = validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("device_to_host_ms"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_short_decode_accepts_legacy_receipt_without_transfer_reduction_section()
-> Result<(), Box<dyn std::error::Error>> {
    let mut receipt = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();
    let receipt_object = receipt.as_object_mut().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "receipt must be an object")
    })?;
    receipt_object.remove("logits_transfer_reduction");

    validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&receipt)?;
    Ok(())
}

#[test]
fn dense_gguf_qwen_short_decode_rejects_unearned_transfer_reduction_claim()
-> Result<(), Box<dyn std::error::Error>> {
    let mut receipt = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();
    mark_short_decode_reduced_sampler_metadata(&mut receipt)?;
    receipt["logits_transfer_reduction"]["transfer_mode"] = json!("device_top_k_cuda_sampler");
    receipt["logits_transfer_reduction"]["sampling_location"] = json!("cuda_device");
    receipt["logits_transfer_reduction"]["reduction_blocker"] = Value::Null;
    receipt["logits_transfer_reduction"]["device_to_host_bytes_reduced"] = json!(true);

    let err = match validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&receipt) {
        Ok(()) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "validator accepted an unearned transfer-reduction claim",
            )
            .into());
        }
        Err(err) => err.to_string(),
    };

    assert!(err.contains("device_to_host_bytes_reduced"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn dense_gguf_qwen_short_decode_requires_reduction_blocker_when_full_logits_download_remains()
-> Result<(), Box<dyn std::error::Error>> {
    let mut receipt = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();
    receipt["logits_transfer_reduction"]["reduction_blocker"] = Value::Null;

    let err = match validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&receipt) {
        Ok(()) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "validator accepted a full-logits receipt without a reduction blocker",
            )
            .into());
        }
        Err(err) => err.to_string(),
    };

    assert!(err.contains("reduction_blocker"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn dense_gguf_qwen_short_decode_rejects_malformed_full_logits_byte_accounting()
-> Result<(), Box<dyn std::error::Error>> {
    let mut receipt = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();
    receipt["logits_transfer_reduction"]["full_logits_bytes_per_step"] = json!(64);
    receipt["logits_transfer_reduction"]["full_logits_download_bytes"] = json!(512);

    let err = match validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&receipt) {
        Ok(()) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "validator accepted inconsistent full-logits byte accounting",
            )
            .into());
        }
        Err(err) => err.to_string(),
    };

    assert!(err.contains("full_logits_bytes_per_step"), "unexpected error: {err}");
    Ok(())
}

fn mark_short_decode_reduced_sampler_metadata(
    receipt: &mut Value,
) -> Result<(), Box<dyn std::error::Error>> {
    receipt["timing"]["device_to_host_ms_source"] = json!("wall_clock_device_top_k_cuda_sampler");
    receipt["tensor_residency"]["transfer_accounting"]["device_to_host_ms_source"] =
        json!("wall_clock_device_top_k_cuda_sampler");
    for step in receipt["short_decode_proof"]["steps"].as_array_mut().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "test fixture short_decode_proof.steps must be an array",
        )
    })? {
        step["cuda_logits_sha256"] = Value::Null;
        step["cuda_logits_sha256_available"] = json!(false);
        step["cuda_logits_sha256_source"] = json!("not_recorded_reduced_device_top_k_sampler");
        step["cpu_logits_sha256_available"] = json!(true);
        step["cpu_logits_sha256_source"] = json!("full_logits_download");
    }
    Ok(())
}

fn mark_short_decode_transfer_reduced(
    receipt: &mut Value,
    actual_device_to_host_bytes: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let full_logits_download_bytes =
        receipt["logits_transfer_reduction"]["full_logits_download_bytes"].as_u64().ok_or_else(
            || {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "test fixture full logits bytes must be a u64",
                )
            },
        )?;
    if actual_device_to_host_bytes >= full_logits_download_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "test fixture must model a reduced D2H transfer",
        )
        .into());
    }
    receipt["kernel_stats"][0]["device_to_host_bytes"] = json!(actual_device_to_host_bytes);
    receipt["timing"]["device_to_host_bytes"] = json!(actual_device_to_host_bytes);
    receipt["tensor_residency"]["transfer_accounting"]["device_to_host_bytes"] =
        json!(actual_device_to_host_bytes);
    receipt["logits_transfer_reduction"]["actual_device_to_host_bytes"] =
        json!(actual_device_to_host_bytes);
    receipt["logits_transfer_reduction"]["device_to_host_bytes_reduced"] = json!(true);
    receipt["logits_transfer_reduction"]["bytes_saved_vs_full_logits"] =
        json!(full_logits_download_bytes - actual_device_to_host_bytes);
    mark_short_decode_reduced_sampler_metadata(receipt)?;
    Ok(())
}

#[test]
fn dense_gguf_qwen_short_decode_rejects_reduced_transfer_without_device_sampler()
-> Result<(), Box<dyn std::error::Error>> {
    let mut receipt = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();
    mark_short_decode_transfer_reduced(&mut receipt, 192)?;

    let err = match validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&receipt) {
        Ok(()) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "validator accepted reduced D2H bytes with the CPU full-logits sampler",
            )
            .into());
        }
        Err(err) => err.to_string(),
    };

    assert!(err.contains("transfer_mode"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn dense_gguf_qwen_short_decode_accepts_device_top_k_reduced_transfer_contract()
-> Result<(), Box<dyn std::error::Error>> {
    let mut receipt = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();
    mark_short_decode_transfer_reduced(&mut receipt, 192)?;
    receipt["logits_transfer_reduction"]["transfer_mode"] = json!("device_top_k_cuda_sampler");
    receipt["logits_transfer_reduction"]["sampling_location"] = json!("cuda_device");
    receipt["logits_transfer_reduction"]["reduction_blocker"] = Value::Null;

    validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&receipt)?;
    Ok(())
}

#[test]
fn dense_gguf_qwen_short_decode_rejects_reduced_transfer_with_cuda_full_logits_hash()
-> Result<(), Box<dyn std::error::Error>> {
    let mut receipt = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();
    mark_short_decode_transfer_reduced(&mut receipt, 192)?;
    receipt["logits_transfer_reduction"]["transfer_mode"] = json!("device_top_k_cuda_sampler");
    receipt["logits_transfer_reduction"]["sampling_location"] = json!("cuda_device");
    receipt["logits_transfer_reduction"]["reduction_blocker"] = Value::Null;
    receipt["short_decode_proof"]["steps"][0]["cuda_logits_sha256"] =
        json!(format!("{:064x}", 990));
    receipt["short_decode_proof"]["steps"][0]["cuda_logits_sha256_available"] = json!(true);
    receipt["short_decode_proof"]["steps"][0]["cuda_logits_sha256_source"] =
        json!("full_logits_download");

    let err = match validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&receipt) {
        Ok(()) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "validator accepted a reduced D2H receipt with a CUDA full-logits hash",
            )
            .into());
        }
        Err(err) => err.to_string(),
    };

    assert!(err.contains("cuda_logits_sha256"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn dense_gguf_qwen_short_decode_rejects_chat_speedup_full_residency_and_bitnet_claims() {
    let mut receipt = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();
    receipt["claim_boundary"]["qwen_chat_cuda_claimed"] = json!(true);
    receipt["claim_boundary"]["speedup_claim"] = json!(true);
    receipt["claim_boundary"]["full_cuda_residency_claimed"] = json!(true);
    receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"] = json!(true);

    let err = validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(
        err.contains("qwen_chat_cuda_claimed")
            || err.contains("speedup_claim")
            || err.contains("full_cuda_residency_claimed")
            || err.contains("bitnet_packed_i2s_qk256_proof"),
        "unexpected error: {err}"
    );
}

#[test]
fn dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_validates() {
    let receipt = valid_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt();

    validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();

    assert_eq!(receipt["session_lifecycle"]["model_loaded_once"], true);
    assert_eq!(receipt["session_lifecycle"]["cuda_context_once"], true);
    assert_eq!(receipt["session_lifecycle"]["weights_uploaded_once"], true);
    assert_eq!(receipt["session_lifecycle"]["per_request_model_load"], false);
    assert_eq!(receipt["session_lifecycle"]["workspace_reused"], true);
    assert_eq!(receipt["session_lifecycle"]["fallback_used"], false);
    assert_eq!(receipt["tensor_residency"]["per_request_model_load"], false);
    assert_eq!(receipt["tensor_residency"]["workspace_reused"], true);
}

#[test]
fn dense_gguf_qwen_warm_session_accepts_qwen3_06b_model_identity()
-> Result<(), Box<dyn std::error::Error>> {
    let mut receipt = valid_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt();
    receipt["model"]["id"] = json!("qwen3-0.6b-instruct-q8_0");
    receipt["model"]["file"] = json!("Qwen3-0.6B-Q8_0.gguf");
    receipt["model"]["architecture"] = json!("qwen3");
    receipt["model"]["sha256"] =
        json!("9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031");
    receipt["parity"]["fixture_id"] = json!("qwen3-0.6b-instruct-q8_0-warm-session-greedy");

    validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(&receipt)?;
    Ok(())
}

#[test]
fn dense_gguf_qwen_warm_session_rejects_short_decode_only_receipt() {
    let receipt = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();

    let err = validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("artifact_kind"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_warm_session_requires_short_decode_prerequisite() {
    let mut receipt = valid_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt();
    receipt["prerequisite_receipts"]["short_decode_proof_claimed"] = json!(false);

    let err = validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("short_decode_proof_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_warm_session_rejects_turn_token_mismatch() {
    let mut receipt = valid_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt();
    receipt["warm_session_proof"]["turns"][1]["cuda_generated_token_ids"][2] = json!(999);

    let err = validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("cpu_generated_token_ids"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_warm_session_rejects_missing_transfer_timing_source_fields() {
    let mut receipt = valid_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt();
    receipt["timing"]["transfer_timing_status"] = json!("bytes_measured_time_unmeasured");

    let err = validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("transfer_timing_status"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_warm_session_accepts_legacy_receipt_without_transfer_reduction_section()
-> Result<(), Box<dyn std::error::Error>> {
    let mut receipt = valid_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt();
    let receipt_object = receipt.as_object_mut().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "receipt must be an object")
    })?;
    receipt_object.remove("logits_transfer_reduction");

    validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(&receipt)?;
    Ok(())
}

#[test]
fn dense_gguf_qwen_warm_session_rejects_transfer_reduction_byte_mismatch()
-> Result<(), Box<dyn std::error::Error>> {
    let mut receipt = valid_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt();
    receipt["logits_transfer_reduction"]["actual_device_to_host_bytes"] = json!(128);

    let err = match validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(&receipt) {
        Ok(()) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "validator accepted mismatched transfer byte accounting",
            )
            .into());
        }
        Err(err) => err.to_string(),
    };

    assert!(err.contains("actual_device_to_host_bytes"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn dense_gguf_qwen_warm_session_rejects_transfer_reduction_without_top_k_evidence()
-> Result<(), Box<dyn std::error::Error>> {
    let mut receipt = valid_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt();
    receipt["logits_transfer_reduction"]["top_k_evidence_preserved"] = json!(false);

    let err = match validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(&receipt) {
        Ok(()) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "validator accepted transfer reduction without top-k evidence",
            )
            .into());
        }
        Err(err) => err.to_string(),
    };

    assert!(err.contains("top_k_evidence_preserved"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn dense_gguf_qwen_warm_session_rejects_malformed_top_k_floor_accounting()
-> Result<(), Box<dyn std::error::Error>> {
    let mut receipt = valid_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt();
    receipt["logits_transfer_reduction"]["top_k_result_bytes_total_floor"] = json!(24);

    let err = match validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(&receipt) {
        Ok(()) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "validator accepted inconsistent top-k floor accounting",
            )
            .into());
        }
        Err(err) => err.to_string(),
    };

    assert!(err.contains("top_k_result_bytes_total_floor"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn dense_gguf_qwen_warm_session_rejects_chat_speedup_full_residency_and_bitnet_claims() {
    let mut receipt = valid_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt();
    receipt["claim_boundary"]["qwen_chat_cuda_claimed"] = json!(true);
    receipt["claim_boundary"]["speedup_claim"] = json!(true);
    receipt["claim_boundary"]["full_cuda_residency_claimed"] = json!(true);
    receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"] = json!(true);

    let err = validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(
        err.contains("qwen_chat_cuda_claimed")
            || err.contains("speedup_claim")
            || err.contains("full_cuda_residency_claimed")
            || err.contains("bitnet_packed_i2s_qk256_proof"),
        "unexpected error: {err}"
    );
}

#[test]
fn dense_gguf_qwen_warm_session_rejects_broad_persistent_residency_claim() {
    let mut receipt = valid_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt();
    receipt["claim_boundary"]["persistent_session_residency_claimed"] = json!(true);

    let err = validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("persistent_session_residency_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_warm_session_rejects_incorrect_persistent_handle_aliases() {
    let mut receipt = valid_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt();
    receipt["session_lifecycle"]["per_request_model_load"] = json!(true);

    let err = validate_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("per_request_model_load"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_ask_strict_cuda_proof_receipt_validates() {
    let receipt = valid_dense_gguf_qwen_ask_strict_cuda_proof_receipt();

    validate_dense_gguf_qwen_ask_strict_cuda_proof_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_qwen_ask_rejects_short_decode_only_receipt() {
    let receipt = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();

    let err = validate_dense_gguf_qwen_ask_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("artifact_kind"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_ask_requires_warm_session_prerequisite() {
    let mut receipt = valid_dense_gguf_qwen_ask_strict_cuda_proof_receipt();
    receipt["prerequisite_receipts"]["warm_session_proof_claimed"] = json!(false);

    let err = validate_dense_gguf_qwen_ask_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("warm_session_proof_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_ask_rejects_generated_token_mismatch() {
    let mut receipt = valid_dense_gguf_qwen_ask_strict_cuda_proof_receipt();
    receipt["ask_proof"]["cuda_generated_token_ids"][0] = json!(999);

    let err = validate_dense_gguf_qwen_ask_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("generated-token arrays"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_ask_rejects_chat_speedup_full_residency_and_bitnet_claims() {
    let mut receipt = valid_dense_gguf_qwen_ask_strict_cuda_proof_receipt();
    receipt["claim_boundary"]["qwen_chat_cuda_claimed"] = json!(true);
    receipt["claim_boundary"]["speedup_claim"] = json!(true);
    receipt["claim_boundary"]["full_cuda_residency_claimed"] = json!(true);
    receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"] = json!(true);

    let err = validate_dense_gguf_qwen_ask_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(
        err.contains("qwen_chat_cuda_claimed")
            || err.contains("speedup_claim")
            || err.contains("full_cuda_residency_claimed")
            || err.contains("bitnet_packed_i2s_qk256_proof"),
        "unexpected error: {err}"
    );
}

#[test]
fn dense_gguf_qwen_chat_strict_cuda_proof_receipt_validates() {
    let receipt = valid_dense_gguf_qwen_chat_strict_cuda_proof_receipt();

    validate_dense_gguf_qwen_chat_strict_cuda_proof_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_qwen_chat_rejects_warm_session_only_receipt() {
    let receipt = valid_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt();

    let err = validate_dense_gguf_qwen_chat_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("artifact_kind"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_chat_requires_source_warm_session_receipt() {
    let mut receipt = valid_dense_gguf_qwen_chat_strict_cuda_proof_receipt();
    receipt.as_object_mut().unwrap().remove("source_warm_session_receipt");

    let err = validate_dense_gguf_qwen_chat_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("source_warm_session_receipt"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_chat_rejects_turn_token_mismatch() {
    let mut receipt = valid_dense_gguf_qwen_chat_strict_cuda_proof_receipt();
    receipt["chat_session"]["turns"][1]["cuda_generated_token_ids"][2] = json!(999);

    let err = validate_dense_gguf_qwen_chat_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("generated token arrays"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_qwen_chat_rejects_speedup_server_full_residency_and_bitnet_claims() {
    let mut receipt = valid_dense_gguf_qwen_chat_strict_cuda_proof_receipt();
    receipt["claim_boundary"]["speedup_claim"] = json!(true);
    receipt["claim_boundary"]["server_ready_claimed"] = json!(true);
    receipt["claim_boundary"]["full_cuda_residency_claimed"] = json!(true);
    receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"] = json!(true);

    let err = validate_dense_gguf_qwen_chat_strict_cuda_proof_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(
        err.contains("speedup_claim")
            || err.contains("server_ready_claimed")
            || err.contains("full_cuda_residency_claimed")
            || err.contains("bitnet_packed_i2s_qk256_proof"),
        "unexpected error: {err}"
    );
}

#[test]
fn dense_gguf_one_layer_cpu_reference_receipt_validates_without_cuda_claims() {
    let receipt = valid_dense_gguf_one_layer_cpu_reference_receipt();

    validate_dense_gguf_one_layer_cpu_reference_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_one_layer_cpu_reference_rejects_cuda_execution_claim() {
    let mut receipt = valid_dense_gguf_one_layer_cpu_reference_receipt();
    receipt["reference_harness"]["cuda_execution_claimed"] = json!(true);

    let err =
        validate_dense_gguf_one_layer_cpu_reference_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("cuda_execution_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_one_layer_cpu_reference_requires_final_output_hash() {
    let mut receipt = valid_dense_gguf_one_layer_cpu_reference_receipt();
    receipt["reference_harness"]["final_output_sha256"] = Value::Null;

    let err =
        validate_dense_gguf_one_layer_cpu_reference_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("final_output_sha256"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_one_layer_cpu_reference_rejects_unbound_final_output_hash() {
    let mut receipt = valid_dense_gguf_one_layer_cpu_reference_receipt();
    receipt["reference_harness"]["final_output_sha256"] = json!("2".repeat(64));

    let err =
        validate_dense_gguf_one_layer_cpu_reference_receipt_json(&receipt).unwrap_err().to_string();

    assert!(
        err.contains("final_output_sha256") && err.contains("second_residual phase output_sha256"),
        "unexpected error: {err}"
    );
}

#[test]
fn dense_gguf_one_layer_cpu_reference_rejects_unbound_deterministic_input_hash() {
    let mut receipt = valid_dense_gguf_one_layer_cpu_reference_receipt();
    receipt["reference_harness"]["deterministic_input_sha256"] = json!("1".repeat(64));

    let err =
        validate_dense_gguf_one_layer_cpu_reference_receipt_json(&receipt).unwrap_err().to_string();

    assert!(
        err.contains("deterministic_input_sha256")
            && err.contains("deterministic_input phase output_sha256"),
        "unexpected error: {err}"
    );
}

#[test]
fn dense_gguf_one_layer_cuda_integrated_parity_receipt_validates() {
    let receipt = valid_dense_gguf_one_layer_cuda_integrated_parity_receipt();

    validate_dense_gguf_one_layer_cuda_integrated_parity_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_one_layer_cuda_integrated_parity_rejects_inference_claim() {
    let mut receipt = valid_dense_gguf_one_layer_cuda_integrated_parity_receipt();
    receipt["cuda_layer"]["dense_gguf_inference_claimed"] = json!(true);
    receipt["claim_boundary"]["dense_gguf_inference_claimed"] = json!(true);

    let err = validate_dense_gguf_one_layer_cuda_integrated_parity_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("dense_gguf_inference_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_one_layer_cuda_integrated_parity_rejects_speedup_claim() {
    let mut receipt = valid_dense_gguf_one_layer_cuda_integrated_parity_receipt();
    receipt["speedup_claim"] = json!(true);
    receipt["claim_boundary"]["speedup_claim"] = json!(true);

    let err = validate_dense_gguf_one_layer_cuda_integrated_parity_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("speedup_claim"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_one_layer_cuda_integrated_parity_rejects_bitnet_proof_claim() {
    let mut receipt = valid_dense_gguf_one_layer_cuda_integrated_parity_receipt();
    receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"] = json!(true);

    let err = validate_dense_gguf_one_layer_cuda_integrated_parity_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("bitnet_packed_i2s_qk256_proof"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_one_layer_cuda_integrated_parity_rejects_phase_error_over_tolerance() {
    let mut receipt = valid_dense_gguf_one_layer_cuda_integrated_parity_receipt();
    receipt["cuda_layer"]["phases"][1]["max_abs_error"] = json!(0.25);
    receipt["cuda_layer"]["phases"][1]["tolerance"] = json!(0.125);

    let err = validate_dense_gguf_one_layer_cuda_integrated_parity_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("attention_norm"), "unexpected error: {err}");
    assert!(err.contains("max_abs_error exceeds tolerance"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_one_layer_cuda_integrated_parity_binds_final_hash_to_terminal_phase() {
    let mut receipt = valid_dense_gguf_one_layer_cuda_integrated_parity_receipt();
    receipt["cuda_layer"]["final_output_sha256"] = json!("f".repeat(64));

    let err = validate_dense_gguf_one_layer_cuda_integrated_parity_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("final_output_sha256"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_one_layer_cuda_integrated_parity_binds_kernel_stats_to_phase_rows() {
    let mut receipt = valid_dense_gguf_one_layer_cuda_integrated_parity_receipt();
    receipt["kernel_stats"][0]["phase"] = json!("attention_q");

    let err = validate_dense_gguf_one_layer_cuda_integrated_parity_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("phase"), "unexpected error: {err}");
    assert!(err.contains("attention_norm"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_one_layer_cuda_integrated_parity_requires_transfer_accounting() {
    let mut receipt = valid_dense_gguf_one_layer_cuda_integrated_parity_receipt();
    receipt["timing"]["host_to_device_bytes"] = json!(1);

    let err = validate_dense_gguf_one_layer_cuda_integrated_parity_receipt_json(&receipt)
        .unwrap_err()
        .to_string();

    assert!(err.contains("host_to_device_bytes"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_norm_fixture_receipt_validates_missing_cuda_kernel() {
    let receipt = valid_dense_gguf_norm_fixture_extraction_receipt();

    validate_dense_gguf_norm_fixture_extraction_receipt_json(&receipt).unwrap();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_norm_fixture_rejects_inference_claim() {
    let mut receipt = valid_dense_gguf_norm_fixture_extraction_receipt();
    receipt["claim_boundary"]["dense_gguf_inference_claimed"] = json!(true);
    receipt["norm_fixtures"][0]["dense_gguf_inference_claimed"] = json!(true);

    let err =
        validate_dense_gguf_norm_fixture_extraction_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("dense_gguf_inference_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_norm_fixture_requires_both_norm_roles() {
    let mut receipt = valid_dense_gguf_norm_fixture_extraction_receipt();
    receipt["norm_fixture_audit"]["covered_roles"] = json!(["attention_norm"]);
    receipt["norm_fixture_audit"]["roles_total"] = json!(1);
    receipt["norm_fixture_audit"]["roles_extracted"] = json!(1);
    receipt["norm_fixtures"].as_array_mut().unwrap().truncate(1);

    let err =
        validate_dense_gguf_norm_fixture_extraction_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("roles_total"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_norm_fixture_rejects_cuda_kernel_claim() {
    let mut receipt = valid_dense_gguf_norm_fixture_extraction_receipt();
    receipt["norm_fixture_audit"]["cuda_kernel_status"] = json!("cuda_kernel_passed");

    let err =
        validate_dense_gguf_norm_fixture_extraction_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("cuda_kernel_status"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_norm_cuda_parity_receipt_validates() {
    let receipt = valid_dense_gguf_norm_cuda_parity_receipt();

    validate_dense_gguf_norm_cuda_parity_receipt_json(&receipt).unwrap();
    validate_dense_regular_llm_cuda_receipt_json(&receipt).unwrap_err();
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof(&receipt).unwrap_err();
}

#[test]
fn dense_gguf_norm_cuda_parity_rejects_dense_inference_claim() {
    let mut receipt = valid_dense_gguf_norm_cuda_parity_receipt();
    receipt["claim_boundary"]["dense_gguf_inference_claimed"] = json!(true);

    let err = validate_dense_gguf_norm_cuda_parity_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("dense_gguf_inference_claimed"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_norm_cuda_parity_requires_both_norm_roles() {
    let mut receipt = valid_dense_gguf_norm_cuda_parity_receipt();
    receipt["parity"]["covered_roles"] = json!(["attention_norm", "attention_norm"]);
    receipt["norm_fixtures"][1]["role"] = json!("attention_norm");

    let err = validate_dense_gguf_norm_cuda_parity_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("duplicate") || err.contains("ffn_norm"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_norm_cuda_parity_rejects_cpu_fallback() {
    let mut receipt = valid_dense_gguf_norm_cuda_parity_receipt();
    receipt["kernel_stats"][0]["fallback_invocations"] = json!(1);

    let err = validate_dense_gguf_norm_cuda_parity_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("fallback_invocations"), "unexpected error: {err}");
}

#[test]
fn dense_gguf_norm_cuda_parity_rejects_bitnet_proof_claim() {
    let mut receipt = valid_dense_gguf_norm_cuda_parity_receipt();
    receipt["claim_boundary"]["bitnet_packed_i2s_qk256_proof"] = json!(true);

    let err = validate_dense_gguf_norm_cuda_parity_receipt_json(&receipt).unwrap_err().to_string();

    assert!(err.contains("bitnet_packed_i2s_qk256_proof"), "unexpected error: {err}");
}

fn valid_smoke_receipt() -> Value {
    json!({
        "schema": 1,
        "artifact_kind": "cuda_smoke",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia-rtx-5070-ti-cuda",
        "timestamp_utc": "2026-05-06T00:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "reference_backend": "amd-9950x3d-cpu-avx512",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "cuda": cuda_identity(),
        "kernel_stats": [kernel_stats()],
        "input_len": 1024,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "result": "pass",
        "claim": "kernel_smoke_tested",
        "artifact_path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-smoke.json",
        "error": null
    })
}

fn valid_parity_receipt() -> Value {
    let mut cuda = cuda_identity();
    cuda["selected_device_index"] = json!(0);
    json!({
        "schema": 1,
        "artifact_kind": "cuda_parity",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia-rtx-5070-ti-cuda",
        "timestamp_utc": "2026-05-06T00:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "reference_backend": "amd-9950x3d-cpu-avx512",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "cuda": cuda,
        "input_len": 1024,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "parity": {
            "reference_backend": "amd-9950x3d-cpu-avx512",
            "target_backend": "nvidia-rtx-5070-ti-cuda",
            "kernel_id": "cuda_tiny_vector_add",
            "fixture_id": "cuda_tiny_vector_add_1024",
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "passed": true,
            "tolerance": 1.1920928955078125e-7,
            "tolerance_source": "docs/bitnet/BITNET_PARITY_TOLERANCES.md",
            "debug_artifact_path": null
        },
        "kernel_stats": [kernel_stats()],
        "result": "pass",
        "claim": "cuda_cpu_parity_tested",
        "artifact_path": "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-parity.json",
        "error": null
    })
}

fn valid_dense_regular_llm_cuda_receipt() -> Value {
    json!({
        "schema": 1,
        "artifact_kind": "dense_regular_llm_cuda",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia-rtx-5070-ti-cuda",
        "timestamp_utc": "2026-05-08T00:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda_identity(),
        "model": {
            "model_family": "qwen",
            "artifact_kind": "dense_gguf",
            "file": "qwen3-0.6b-q4_k_m.gguf",
            "sha256": "0".repeat(64)
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "cublas_dense_gemm",
            "quantization_family": "fp16_bf16_dense",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": {
            "planner_version": "cuda-planner-004",
            "model_family": "qwen",
            "quantization": "dense_fp16",
            "selected_route": "dense_regular_llm_cuda",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "strict_fallback_policy": "reject",
            "dense_regular_llm_cuda": true,
            "bitnet_packed_qk256_cuda": false,
            "cuda_bitnet_qk256_ops": 0,
            "cuda_dense_regular_llm_ops": 1,
            "cpu_fallback_ops": 0,
            "unsupported_ops": 0,
            "total_ops": 1,
            "cuda_ops": 1,
            "mixed_cuda_routes": false,
            "fallback_used": false,
            "strict_cuda_ready": true,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "kernel_stats": [{
            "kernel_id": "dense_f16_gemm_cuda",
            "invocations": 1,
            "fallback_invocations": 0,
            "host_to_device_bytes": 40,
            "device_to_host_bytes": 24,
            "kernel_launches": 1,
            "kernel_time_ms": null
        }],
        "parity": {
            "reference_backend": "amd-9950x3d-cpu-avx512",
            "target_backend": "nvidia-rtx-5070-ti-cuda",
            "kernel_id": "dense_f16_gemm_cuda",
            "fixture_id": "dense_f16_gemm_m2_n3_k4",
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "passed": true,
            "tolerance": 0.002,
            "tolerance_source": "CUDA-DENSE-002 deterministic FP16 smoke fixture"
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "error": null
    })
}

fn valid_dense_regular_llm_cuda_residency_receipt() -> Value {
    let mut receipt = valid_dense_regular_llm_cuda_receipt();
    receipt["claim"] = json!("dense_regular_llm_cuda_tensor_residency_tested");
    receipt["artifact_path"] =
        json!("ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-residency.json");
    receipt["claim_boundary"]["dense_tensor_residency_claimed"] = json!(true);
    receipt["claim_boundary"]["dense_gguf_inference_claimed"] = json!(false);
    receipt["claim_boundary"]["persistent_session_residency_claimed"] = json!(false);
    receipt["tensor_residency"] = json!({
        "schema_version": "1.0.0",
        "scope": "single_dense_f16_gemm_fixture",
        "model_class": "dense_regular_llm",
        "fixture_id": "dense_f16_gemm_m2_n3_k4",
        "dense_tensor_residency_claimed": true,
        "dense_gguf_inference_claimed": false,
        "persistent_session_residency_claimed": false,
        "full_cuda_residency_claimed": false,
        "input_tensors_uploaded_once": true,
        "output_tensor_cuda_resident_during_kernel": true,
        "host_device_transfer_accounting_matches_kernel_stats": true,
        "inputs": [
            {
                "name": "a",
                "dtype": "f16",
                "shape": [2, 4],
                "host_bytes": 16,
                "device_residency": "cuda_device_buffer",
                "upload_count": 1,
                "reuse_scope": "single_fixture_launch"
            },
            {
                "name": "b",
                "dtype": "f16",
                "shape": [4, 3],
                "host_bytes": 24,
                "device_residency": "cuda_device_buffer",
                "upload_count": 1,
                "reuse_scope": "single_fixture_launch"
            }
        ],
        "outputs": [
            {
                "name": "c",
                "dtype": "f32",
                "shape": [2, 3],
                "device_residency": "cuda_device_buffer",
                "device_to_host_bytes": 24,
                "download_scope": "parity_check_only"
            }
        ],
        "allocation": {
            "device_buffer_count": 3,
            "temporary_workspace_bytes": 0,
            "persistent_handle_count": 0,
            "persistent_handles_claimed": false
        },
        "transfer_accounting": {
            "status": "measured",
            "host_to_device_bytes": 40,
            "device_to_host_bytes": 24
        }
    });
    receipt
}

fn valid_dense_regular_llm_cuda_persistent_residency_receipt() -> Value {
    let mut receipt = valid_dense_regular_llm_cuda_receipt();
    receipt["claim"] = json!("dense_regular_llm_cuda_persistent_fixture_residency_tested");
    receipt["artifact_path"] =
        json!("ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-persistent.json");
    receipt["kernel_stats"][0]["invocations"] = json!(3);
    receipt["kernel_stats"][0]["device_to_host_bytes"] = json!(72);
    receipt["kernel_stats"][0]["kernel_launches"] = json!(3);
    receipt["parity"]["runs"] = json!(3);
    receipt["claim_boundary"]["dense_tensor_residency_claimed"] = json!(true);
    receipt["claim_boundary"]["dense_gguf_inference_claimed"] = json!(false);
    receipt["claim_boundary"]["persistent_session_residency_claimed"] = json!(true);
    receipt["persistent_session"] = json!({
        "schema_version": "1.0.0",
        "scope": "persistent_dense_f16_gemm_fixture_session",
        "repeated_runs": 3,
        "context_creations": 1,
        "module_loads": 1,
        "kernel_launches": 3,
        "input_uploads": 2,
        "output_allocations": 1,
        "persistent_handle_count": 3,
        "per_run_host_to_device_bytes": 0,
        "dense_gguf_inference_claimed": false,
        "full_cuda_residency_claimed": false,
        "speedup_claim": false
    });
    receipt["tensor_residency"] = json!({
        "schema_version": "1.0.0",
        "scope": "persistent_dense_f16_gemm_fixture_session",
        "model_class": "dense_regular_llm",
        "fixture_id": "dense_f16_gemm_m2_n3_k4",
        "dense_tensor_residency_claimed": true,
        "dense_gguf_inference_claimed": false,
        "persistent_session_residency_claimed": true,
        "full_cuda_residency_claimed": false,
        "input_tensors_uploaded_once": true,
        "output_tensor_cuda_resident_during_kernel": true,
        "host_device_transfer_accounting_matches_kernel_stats": true,
        "per_run_host_to_device_bytes": 0,
        "inputs": [
            {
                "name": "a",
                "dtype": "f16",
                "shape": [2, 4],
                "host_bytes": 16,
                "device_residency": "cuda_device_buffer",
                "upload_count": 1,
                "reuse_scope": "persistent_fixture_session"
            },
            {
                "name": "b",
                "dtype": "f16",
                "shape": [4, 3],
                "host_bytes": 24,
                "device_residency": "cuda_device_buffer",
                "upload_count": 1,
                "reuse_scope": "persistent_fixture_session"
            }
        ],
        "outputs": [
            {
                "name": "c",
                "dtype": "f32",
                "shape": [2, 3],
                "device_residency": "cuda_device_buffer",
                "device_to_host_bytes": 72,
                "download_scope": "parity_check_each_run"
            }
        ],
        "allocation": {
            "device_buffer_count": 3,
            "temporary_workspace_bytes": 0,
            "persistent_handle_count": 3,
            "persistent_handles_claimed": true
        },
        "transfer_accounting": {
            "status": "measured",
            "host_to_device_bytes": 40,
            "device_to_host_bytes": 72
        }
    });
    receipt
}

fn valid_dense_gguf_descriptor_inspection_receipt() -> Value {
    json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_tensor_descriptor_inspection",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia-rtx-5070-ti-cuda",
        "timestamp_utc": "2026-05-08T23:45:00Z",
        "claim": "dense_gguf_tensor_descriptors_inspected",
        "inspection_source": "synthetic_gguf_reader_fixture",
        "model": {
            "model_family": "qwen",
            "architecture": "qwen3",
            "artifact_kind": "dense_gguf",
            "quantization_family": "q8_0_dense_gguf",
            "file": "synthetic-qwen3-q8_0-descriptor-fixture.gguf",
            "fixture": true
        },
        "descriptor_inspection": {
            "schema": 1,
            "artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "architecture": "qwen3",
            "model_family": "qwen",
            "tensor_count": 11,
            "metadata_count": 4,
            "quantization_families": ["f32", "q8_0"],
            "descriptors": dense_gguf_descriptor_entries(),
            "required_roles_present": true,
            "missing_required_roles": [],
            "strict_descriptor_complete": true,
            "dense_cuda_route_status": "descriptor_only_quant_bridge_required",
            "bitnet_packed_marker_found": false,
            "dense_gguf_inference_claimed": false,
            "dense_regular_llm_cuda_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "claim_boundary": {
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_regular_llm_cuda_claimed": false,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "notes": [
            "Descriptor-only GGUF reader fixture; no CUDA kernel or dense GGUF inference was executed.",
            "Q8_0 tensors require a future quant bridge before strict dense CUDA routing can be claimed."
        ],
        "error": null
    })
}

fn valid_dense_gguf_linear_fixture_extraction_receipt() -> Value {
    json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_linear_fixture_extraction",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia-rtx-5070-ti-cuda",
        "timestamp_utc": "2026-05-08T23:55:00Z",
        "claim": "dense_gguf_linear_fixture_extracted",
        "inspection_source": "synthetic_gguf_reader_fixture",
        "model": {
            "model_family": "qwen",
            "architecture": "qwen3",
            "artifact_kind": "dense_gguf",
            "quantization_family": "q8_0_dense_gguf",
            "file": "synthetic-qwen3-q8_0-linear-fixture.gguf",
            "fixture": true
        },
        "linear_fixture": {
            "schema": 1,
            "artifact_kind": "dense_gguf_linear_fixture_extraction",
            "architecture": "qwen3",
            "model_family": "qwen",
            "tensor_name": "blk.0.attn_q.weight",
            "role": "attention_q",
            "tensor_type": "q8_0",
            "source_shape": [4, 3],
            "source_offset": 0,
            "source_size_bytes": 34,
            "matrix_rows": 3,
            "matrix_cols": 4,
            "value_count": 12,
            "logical_layout": "gguf_in_out_reinterpreted_as_out_in",
            "values_materialized_as_f32": true,
            "weight_values_sha256": "f54b6160287bd214bbd21d91fdd4e8d0853f2d1d171dd44c62bf4a6387ef78d9",
            "cpu_reference_input_len": 4,
            "cpu_reference_output_len": 3,
            "cpu_reference_input_sha256": "ca10b81731aaa2cfc8af6f8331f18aee7ee9a8c656a52557dfaacd00eefd72c5",
            "cpu_reference_output_sha256": "d6ce8d6984e070a14053339ea7955453127c771e958c66de5e4a6d1d79423bef",
            "cpu_reference_computed": true,
            "dense_gguf_inference_claimed": false,
            "dense_regular_llm_cuda_claimed": false,
            "cpu_cuda_parity_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "claim_boundary": {
            "dense_gguf_linear_fixture_extraction_claimed": true,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_regular_llm_cuda_claimed": false,
            "dense_gguf_inference_claimed": false,
            "cpu_cuda_parity_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "notes": [
            "Synthetic GGUF reader fixture; one Q8_0 dense linear tensor was materialized as F32 for CPU reference matvec extraction.",
            "No CUDA kernel, dense GGUF inference, speedup, full residency, or BitNet packed-kernel proof is claimed."
        ],
        "error": null
    })
}

fn valid_dense_gguf_linear_cuda_parity_receipt() -> Value {
    json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_linear_cuda_parity",
        "artifact_path": "target/bitnet/receipts/dense-gguf-linear-cuda-parity.json",
        "claim": "dense_gguf_linear_cuda_parity_tested",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia-rtx-5070-ti-cuda",
        "timestamp_utc": "2026-05-09T00:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda_identity(),
        "model": {
            "model_family": "qwen",
            "architecture": "qwen3",
            "artifact_kind": "dense_gguf",
            "file": "synthetic-dense-gguf-linear-fixture",
            "sha256": "0".repeat(64)
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_fp16_gemm",
            "quantization_family": "q8_0_materialized_to_f16_bridge",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": {
            "planner_version": "cuda-planner-004",
            "model_family": "qwen",
            "quantization": "dense_fp16",
            "selected_route": "dense_regular_llm_cuda",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "strict_fallback_policy": "reject",
            "dense_regular_llm_cuda": true,
            "bitnet_packed_qk256_cuda": false,
            "cuda_bitnet_qk256_ops": 0,
            "cuda_dense_regular_llm_ops": 1,
            "cpu_fallback_ops": 0,
            "unsupported_ops": 0,
            "total_ops": 1,
            "cuda_ops": 1,
            "mixed_cuda_routes": false,
            "fallback_used": false,
            "strict_cuda_ready": true,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "linear_fixture": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_linear_fixture_extraction",
            "fixture_id": "dense_gguf_linear_qwen_attention_q_f16_bridge",
            "model_family": "qwen",
            "architecture": "qwen3",
            "tensor_name": "blk.0.attn_q.weight",
            "role": "attention_q",
            "tensor_type": "q8_0",
            "matrix_rows": 3,
            "matrix_cols": 4,
            "logical_layout": "gguf_in_out_reinterpreted_as_out_in",
            "gemm_layout": "input_1_by_in_times_weight_in_by_out",
            "values_materialized_as_f32": true,
            "gemm_input_dtype": "f16",
            "gemm_weight_dtype": "f16",
            "gemm_output_dtype": "f32",
            "weight_values_sha256": "1".repeat(64),
            "dense_gguf_inference_claimed": false,
            "dense_regular_llm_cuda_claimed": true,
            "cpu_cuda_parity_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "kernel_stats": [{
            "kernel_id": "dense_f16_gemm_cuda",
            "invocations": 1,
            "fallback_invocations": 0,
            "host_to_device_bytes": 32,
            "device_to_host_bytes": 12,
            "kernel_launches": 1,
            "kernel_time_ms": null
        }],
        "parity": {
            "reference_backend": "amd-9950x3d-cpu-avx512",
            "target_backend": "nvidia-rtx-5070-ti-cuda",
            "kernel_id": "dense_f16_gemm_cuda",
            "fixture_id": "dense_gguf_linear_qwen_attention_q_f16_bridge",
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "passed": true,
            "tolerance": 0.002,
            "tolerance_source": "CUDA-DENSE-008 dense GGUF linear FP16 bridge fixture"
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": true,
            "dense_gguf_linear_cuda_parity_claimed": true,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "tensor_residency": {
            "schema_version": "1.0.0",
            "scope": "single_dense_gguf_linear_fixture",
            "model_class": "dense_regular_llm",
            "fixture_id": "dense_gguf_linear_qwen_attention_q_f16_bridge",
            "dense_tensor_residency_claimed": true,
            "dense_gguf_inference_claimed": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false,
            "input_tensors_uploaded_once": true,
            "output_tensor_cuda_resident_during_kernel": true,
            "host_device_transfer_accounting_matches_kernel_stats": true,
            "inputs": [
                {
                    "name": "dense_gguf_linear_input",
                    "dtype": "f16",
                    "shape": [1, 4],
                    "host_bytes": 8,
                    "device_residency": "cuda_device_buffer",
                    "upload_count": 1,
                    "reuse_scope": "single_fixture_launch"
                },
                {
                    "name": "dense_gguf_linear_weight_transposed",
                    "dtype": "f16",
                    "shape": [4, 3],
                    "host_bytes": 24,
                    "device_residency": "cuda_device_buffer",
                    "upload_count": 1,
                    "reuse_scope": "single_fixture_launch"
                }
            ],
            "outputs": [
                {
                    "name": "dense_gguf_linear_output",
                    "dtype": "f32",
                    "shape": [1, 3],
                    "device_residency": "cuda_device_buffer",
                    "device_to_host_bytes": 12,
                    "download_scope": "parity_check_only"
                }
            ],
            "allocation": {
                "device_buffer_count": 3,
                "temporary_workspace_bytes": 0,
                "persistent_handle_count": 0,
                "persistent_handles_claimed": false
            },
            "transfer_accounting": {
                "status": "measured",
                "host_to_device_bytes": 32,
                "device_to_host_bytes": 12
            }
        },
        "error": null
    })
}

fn valid_dense_gguf_linear_role_sweep_cuda_parity_receipt() -> Value {
    let mut attention_q = valid_dense_gguf_linear_cuda_parity_receipt();
    let fixture_q = attention_q["linear_fixture"].take();
    let stat_q = json!({
        "role": "attention_q",
        "tensor_name": "blk.0.attn_q.weight",
        "fixture_id": "dense_gguf_linear_qwen_attention_q_f16_bridge",
        "kernel_id": "dense_f16_gemm_cuda",
        "invocations": 1,
        "fallback_invocations": 0,
        "host_to_device_bytes": 32,
        "device_to_host_bytes": 12,
        "kernel_launches": 1,
        "kernel_time_ms": null
    });

    let mut fixture_k = fixture_q.clone();
    fixture_k["fixture_id"] = json!("dense_gguf_linear_qwen_attention_k_f16_bridge");
    fixture_k["tensor_name"] = json!("blk.0.attn_k.weight");
    fixture_k["role"] = json!("attention_k");
    fixture_k["weight_values_sha256"] = json!("2".repeat(64));
    let stat_k = json!({
        "role": "attention_k",
        "tensor_name": "blk.0.attn_k.weight",
        "fixture_id": "dense_gguf_linear_qwen_attention_k_f16_bridge",
        "kernel_id": "dense_f16_gemm_cuda",
        "invocations": 1,
        "fallback_invocations": 0,
        "host_to_device_bytes": 32,
        "device_to_host_bytes": 12,
        "kernel_launches": 1,
        "kernel_time_ms": null
    });

    json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_linear_role_sweep_cuda_parity",
        "artifact_path": "target/bitnet/receipts/dense-gguf-linear-role-sweep-cuda-parity.json",
        "claim": "dense_gguf_linear_role_sweep_cuda_parity_tested",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia-rtx-5070-ti-cuda",
        "timestamp_utc": "2026-05-09T00:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda_identity(),
        "model": {
            "model_family": "qwen",
            "architecture": "qwen3",
            "artifact_kind": "dense_gguf",
            "file": "synthetic-dense-gguf-linear-role-sweep-fixture",
            "sha256": "0".repeat(64)
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_fp16_gemm",
            "quantization_family": "q8_0_materialized_to_f16_bridge",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": {
            "planner_version": "cuda-planner-004",
            "model_family": "qwen",
            "quantization": "dense_fp16",
            "selected_route": "dense_regular_llm_cuda",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "strict_fallback_policy": "reject",
            "dense_regular_llm_cuda": true,
            "bitnet_packed_qk256_cuda": false,
            "cuda_bitnet_qk256_ops": 0,
            "cuda_dense_regular_llm_ops": 2,
            "cpu_fallback_ops": 0,
            "unsupported_ops": 0,
            "total_ops": 2,
            "cuda_ops": 2,
            "mixed_cuda_routes": false,
            "fallback_used": false,
            "strict_cuda_ready": true,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "linear_role_sweep": {
            "schema": 1,
            "roles_total": 2,
            "roles_passed": 2,
            "roles_failed": 0,
            "covered_roles": ["attention_q", "attention_k"],
            "all_parity_passed": true,
            "max_abs_error": 0.0,
            "max_mean_abs_error": 0.0,
            "aggregate_kernel_time_ms": null,
            "host_to_device_bytes": 64,
            "device_to_host_bytes": 24,
            "kernel_invocations": 2,
            "kernel_launches": 2,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "linear_fixtures": [fixture_q, fixture_k],
        "kernel_stats": [stat_q, stat_k],
        "parity": {
            "reference_backend": "amd-9950x3d-cpu-avx512",
            "target_backend": "nvidia-rtx-5070-ti-cuda",
            "kernel_id": "dense_f16_gemm_cuda",
            "roles_total": 2,
            "roles_passed": 2,
            "roles_failed": 0,
            "max_abs_error": 0.0,
            "max_mean_abs_error": 0.0,
            "passed": true,
            "tolerance": 0.002,
            "tolerance_source": "CUDA-DENSE-012 extracted dense GGUF linear role-sweep FP16 bridge"
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": true,
            "dense_gguf_linear_cuda_parity_claimed": true,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": true,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "tensor_residency": {
            "schema_version": "1.0.0",
            "scope": "dense_gguf_linear_role_sweep_fixture",
            "model_class": "dense_regular_llm",
            "roles_total": 2,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_inference_claimed": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false,
            "input_tensors_uploaded_once_per_role": true,
            "output_tensor_cuda_resident_during_kernel": true,
            "host_device_transfer_accounting_matches_kernel_stats": true,
            "allocation": {
                "device_buffer_count": 6,
                "temporary_workspace_bytes": 0,
                "persistent_handle_count": 0,
                "persistent_handles_claimed": false
            },
            "transfer_accounting": {
                "status": "measured",
                "host_to_device_bytes": 64,
                "device_to_host_bytes": 24,
                "kernel_invocations": 2,
                "kernel_launches": 2
            }
        },
        "error": null
    })
}

fn valid_dense_gguf_one_layer_execution_plan_receipt() -> Value {
    json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_one_layer_execution_plan",
        "artifact_path": "target/bitnet/receipts/dense-gguf-one-layer-plan.json",
        "claim": "dense_gguf_one_layer_execution_plan_gap_recorded",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia-rtx-5070-ti-cuda",
        "timestamp_utc": "2026-05-09T00:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda_identity(),
        "model": {
            "model_family": "qwen",
            "architecture": "qwen3",
            "artifact_kind": "dense_gguf",
            "file": "synthetic-dense-gguf-one-layer-plan",
            "sha256": "0".repeat(64)
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_fp16_gemm_plus_f32_rmsnorm_plus_f32_rope_plus_f32_attention_plus_f32_mlp_activation",
            "quantization_family": "dense_fp16_bridge_from_gguf_descriptors_with_f32_rmsnorm_rope_attention_mlp_activation",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": {
            "planner_version": "cuda-planner-004",
            "model_family": "qwen",
            "quantization": "dense_fp16",
            "selected_route": "dense_regular_llm_cuda",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "strict_fallback_policy": "reject",
            "dense_regular_llm_cuda": true,
            "bitnet_packed_qk256_cuda": false,
            "cuda_bitnet_qk256_ops": 0,
            "cuda_dense_regular_llm_ops": 14,
            "cpu_fallback_ops": 0,
            "unsupported_ops": 0,
            "total_ops": 14,
            "cuda_ops": 14,
            "mixed_cuda_routes": false,
            "fallback_used": false,
            "strict_cuda_ready": true,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": 11,
            "metadata_count": 4,
            "required_roles_present": true,
            "strict_descriptor_complete": true,
            "dense_cuda_route_status": "descriptor_only_quant_bridge_required",
            "quantization_families": ["f32", "q8_0"],
            "bitnet_packed_marker_found": false,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "one_layer_plan": {
            "schema": 1,
            "layer_index": 0,
            "total_ops": 14,
            "cuda_routable_ops_total": 14,
            "linear_cuda_ops_total": 7,
            "norm_cuda_ops_total": 2,
            "rope_cuda_ops_total": 1,
            "attention_score_cuda_ops_total": 1,
            "attention_softmax_cuda_ops_total": 1,
            "attention_v_mix_cuda_ops_total": 1,
            "mlp_activation_cuda_ops_total": 1,
            "unsupported_strict_cuda_ops_total": 0,
            "cpu_fallback_ops_total": 0,
            "strict_cuda_ready": true,
            "unsupported_ops_explicitly_listed": true,
            "operations": dense_one_layer_operations(),
            "dense_gguf_one_layer_execution_plan_claimed": true,
            "one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "gap_audit": dense_one_layer_gap_audit(),
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": false,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": false,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": true,
            "dense_gguf_one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "error": null
    })
}

fn valid_dense_gguf_all_layer_execution_plan_receipt() -> Value {
    json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_all_layer_execution_plan",
        "artifact_path": "target/bitnet/receipts/dense-gguf-all-layer-plan.json",
        "claim": "dense_gguf_all_layer_execution_plan_recorded",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia-rtx-5070-ti-cuda",
        "timestamp_utc": "2026-05-09T00:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda_identity(),
        "model": {
            "model_family": "qwen",
            "architecture": "qwen3",
            "artifact_kind": "dense_gguf",
            "file": "synthetic-dense-gguf-all-layer-plan",
            "sha256": "0".repeat(64)
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_cuda_all_layer_execution_plan",
            "quantization_family": "dense_fp16_bridge_from_gguf_descriptors_with_q8_0_fixture_contracts",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": {
            "planner_version": "cuda-planner-004",
            "model_family": "qwen",
            "quantization": "dense_fp16",
            "selected_route": "dense_regular_llm_cuda",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "strict_fallback_policy": "reject",
            "dense_regular_llm_cuda": true,
            "bitnet_packed_qk256_cuda": false,
            "cuda_bitnet_qk256_ops": 0,
            "cuda_dense_regular_llm_ops": 28,
            "cpu_fallback_ops": 0,
            "unsupported_ops": 0,
            "total_ops": 28,
            "cuda_ops": 28,
            "mixed_cuda_routes": false,
            "fallback_used": false,
            "strict_cuda_ready": true,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": 20,
            "metadata_count": 4,
            "required_roles_present": true,
            "strict_descriptor_complete": true,
            "dense_cuda_route_status": "descriptor_only_quant_bridge_required",
            "quantization_families": ["f32", "q8_0"],
            "bitnet_packed_marker_found": false,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "all_layer_plan": {
            "schema": 1,
            "transformer_layers_total": 2,
            "layers_with_complete_cuda_block_plan": 2,
            "layer_plan_matches_layer0": true,
            "layer_differences": [],
            "missing_layer_indices": [],
            "total_ops": 28,
            "cuda_routable_ops_total": 28,
            "linear_cuda_ops_total": 14,
            "norm_cuda_ops_total": 4,
            "rope_cuda_ops_total": 2,
            "attention_score_cuda_ops_total": 2,
            "attention_softmax_cuda_ops_total": 2,
            "attention_v_mix_cuda_ops_total": 2,
            "mlp_activation_cuda_ops_total": 2,
            "unsupported_strict_cuda_ops_total": 0,
            "cpu_fallback_ops_total": 0,
            "strict_cuda_ready": true,
            "strict_cuda_ready_scope": "transformer_blocks_only",
            "all_layers_inspected": true,
            "operations_per_layer": 14,
            "layers": [
                dense_all_layer_plan_layer(0),
                dense_all_layer_plan_layer(1)
            ],
            "dense_gguf_all_layer_execution_plan_claimed": true,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "model_boundary_gaps": dense_all_layer_model_boundary_gaps(),
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": false,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": false,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": true,
            "dense_gguf_one_layer_cpu_reference_claimed": true,
            "dense_gguf_one_layer_cuda_integrated_parity_claimed": true,
            "dense_gguf_all_layer_execution_plan_claimed": true,
            "dense_gguf_one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "error": null
    })
}

fn valid_dense_gguf_model_boundary_fixtures_receipt() -> Value {
    json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_model_boundary_fixtures",
        "artifact_path": "target/bitnet/receipts/dense-gguf-model-boundary-fixtures.json",
        "claim": "dense_gguf_model_boundary_fixtures_recorded",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia-rtx-5070-ti-cuda",
        "timestamp_utc": "2026-05-09T00:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda_identity(),
        "model": {
            "model_family": "qwen",
            "architecture": "qwen3",
            "artifact_kind": "dense_gguf",
            "file": "synthetic-dense-gguf-model-boundary-fixtures",
            "sha256": "0".repeat(64)
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_cuda_model_boundary_fixture_route",
            "quantization_family": "dense_gguf_q8_0_f16_boundary_fixture_contract",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": {
            "planner_version": "cuda-planner-004",
            "model_family": "qwen",
            "quantization": "dense_fp16",
            "selected_route": "dense_regular_llm_cuda",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "strict_fallback_policy": "reject",
            "dense_regular_llm_cuda": true,
            "bitnet_packed_qk256_cuda": false,
            "cuda_bitnet_qk256_ops": 0,
            "cuda_dense_regular_llm_ops": 3,
            "cpu_fallback_ops": 0,
            "unsupported_ops": 0,
            "total_ops": 3,
            "cuda_ops": 3,
            "mixed_cuda_routes": false,
            "fallback_used": false,
            "strict_cuda_ready": true,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": 23,
            "metadata_count": 4,
            "required_roles_present": true,
            "strict_descriptor_complete": true,
            "dense_cuda_route_status": "descriptor_only_quant_bridge_required",
            "quantization_families": ["f32", "q8_0"],
            "bitnet_packed_marker_found": false,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "model_boundary_fixtures": {
            "schema": 1,
            "fixture_id": "dense_gguf_model_boundary_fixtures_qwen_s4_top4",
            "seq_len": 4,
            "hidden_size": 4,
            "vocab_size": 4,
            "token_ids": [0, 1, 2, 3],
            "token_ids_sha256": format!("{:064x}", 41),
            "fixtures_total": 3,
            "token_embedding": dense_boundary_fixture(
                "token_embedding_lookup",
                "token_embedding",
                "token_embd.weight",
                "q8_0",
                json!([4, 4]),
                16,
                16,
                format!("{:064x}", 42)
            ),
            "final_norm": {
                "rmsnorm_eps": 0.000001,
                "epsilon_source": "default_1e-6",
                "input_sha256": format!("{:064x}", 43),
                "output_sha256": format!("{:064x}", 44),
                "fixture": dense_boundary_fixture(
                    "final_model_norm",
                    "final_norm",
                    "output_norm.weight",
                    "f32",
                    json!([4]),
                    4,
                    4,
                    format!("{:064x}", 44)
                )
            },
            "lm_head_logits": {
                "logits_len": 4,
                "logits_sha256": format!("{:064x}", 45),
                "top_k": 4,
                "top_k_entries": [
                    { "rank": 0, "token_id": 3, "value": 0.4 },
                    { "rank": 1, "token_id": 2, "value": 0.3 },
                    { "rank": 2, "token_id": 1, "value": 0.2 },
                    { "rank": 3, "token_id": 0, "value": 0.1 }
                ],
                "fixture": dense_boundary_fixture(
                    "lm_head_logits",
                    "lm_head_logits",
                    "output.weight",
                    "q8_0",
                    json!([4, 4]),
                    16,
                    4,
                    format!("{:064x}", 45)
                )
            },
            "boundary_fixtures_claimed": true,
            "token_embedding_fixture_claimed": true,
            "final_norm_fixture_claimed": true,
            "lm_head_logits_fixture_claimed": true,
            "fixture_route_only": true,
            "cuda_kernel_execution_claimed": false,
            "kernel_invocations": 0,
            "fallback_used": false,
            "kv_cache_policy_claimed": false,
            "sampling_integration_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "remaining_model_boundary_gaps": {
            "schema": 1,
            "gaps": [
                {
                    "gap": "kv_cache_policy",
                    "status": "not_governed_by_model_boundary_fixtures",
                    "required_next_proof": "dense_gguf_kv_cache_policy_receipt",
                    "blocks_qwen_one_token": true,
                    "blocks_qwen_short_decode": true,
                    "blocks_qwen_chat": true
                },
                {
                    "gap": "sampling",
                    "status": "not_governed_by_model_boundary_fixtures",
                    "required_next_proof": "dense_gguf_sampling_policy_receipt",
                    "blocks_qwen_one_token": true,
                    "blocks_qwen_short_decode": true,
                    "blocks_qwen_chat": true
                }
            ],
            "qwen_one_token_cuda_blocked": true,
            "qwen_short_decode_cuda_blocked": true,
            "qwen_chat_cuda_blocked": true,
            "next_required_proof": "dense_gguf_kv_cache_policy_receipt",
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": false,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": true,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": true,
            "dense_gguf_one_layer_cpu_reference_claimed": true,
            "dense_gguf_one_layer_cuda_integrated_parity_claimed": true,
            "dense_gguf_all_layer_execution_plan_claimed": true,
            "dense_gguf_model_boundary_fixtures_claimed": true,
            "dense_gguf_one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "kv_cache_policy_claimed": false,
            "sampling_integration_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "error": null
    })
}

fn valid_dense_gguf_kv_cache_policy_receipt() -> Value {
    json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_kv_cache_policy",
        "artifact_path": "target/bitnet/receipts/dense-gguf-kv-cache-policy.json",
        "claim": "dense_gguf_kv_cache_policy_recorded",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia-rtx-5070-ti-cuda",
        "timestamp_utc": "2026-05-09T00:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda_identity(),
        "model": {
            "model_family": "qwen",
            "architecture": "qwen3",
            "artifact_kind": "dense_gguf",
            "file": "synthetic-dense-gguf-kv-cache-policy",
            "sha256": "0".repeat(64)
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_cuda_kv_cache_policy_route",
            "quantization_family": "dense_gguf_q8_0_f16_kv_cache_policy_contract",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": {
            "planner_version": "cuda-planner-004",
            "model_family": "qwen",
            "quantization": "dense_fp16",
            "selected_route": "dense_regular_llm_cuda",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "strict_fallback_policy": "reject",
            "dense_regular_llm_cuda": true,
            "bitnet_packed_qk256_cuda": false,
            "cuda_bitnet_qk256_ops": 0,
            "cuda_dense_regular_llm_ops": 1,
            "cpu_fallback_ops": 0,
            "unsupported_ops": 0,
            "total_ops": 1,
            "cuda_ops": 1,
            "mixed_cuda_routes": false,
            "fallback_used": false,
            "strict_cuda_ready": true,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": 23,
            "metadata_count": 4,
            "required_roles_present": true,
            "strict_descriptor_complete": true,
            "dense_cuda_route_status": "descriptor_only_quant_bridge_required",
            "quantization_families": ["f32", "q8_0"],
            "bitnet_packed_marker_found": false,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "kv_cache_policy": {
            "schema": 1,
            "policy_id": "dense_gguf_kv_cache_policy_qwen_layers1_ctx5_kv1_k2_v2",
            "policy_scope": "dense_qwen_prefill_decode_boundary",
            "planned_residency": "cuda_required_for_strict_dense_cuda",
            "observed_residency": "not_allocated_policy_only",
            "transformer_layers_total": 1,
            "context_length": 5,
            "seq_len": 4,
            "decode_steps": 1,
            "q_heads": 2,
            "kv_heads": 1,
            "heads_per_kv_group": 2,
            "key_head_dim": 2,
            "value_head_dim": 2,
            "kv_element_dtype": "f16",
            "kv_element_bytes": 2,
            "kv_values_per_token_per_layer": 4,
            "kv_bytes_per_token_per_layer": 8,
            "kv_bytes_per_token_all_layers": 8,
            "metadata_sources": {
                "transformer_layers": "inferred_from_dense_layer_descriptors",
                "context_length": "seq_len_plus_decode_steps",
                "q_heads": "qwen3.attention.head_count",
                "kv_heads": "qwen3.attention.head_count_kv",
                "key_head_dim": "qwen3.attention.key_length",
                "value_head_dim": "qwen3.attention.key_length_default_value_dim"
            },
            "prefill": {
                "write_tokens": 4,
                "writes_keys": true,
                "writes_values": true,
                "write_bytes_estimate": 32,
                "write_path": "qkv_projection_to_cuda_kv_cache",
                "measured": false
            },
            "decode": {
                "decode_steps": 1,
                "read_tokens_per_step": 4,
                "read_bytes_per_step_estimate": 32,
                "write_tokens_per_step": 1,
                "write_bytes_per_step_estimate": 8,
                "read_path": "cuda_kv_cache_to_attention",
                "write_path": "qkv_projection_to_cuda_kv_cache",
                "measured": false
            },
            "max_context": {
                "tokens": 5,
                "bytes_estimate": 40
            },
            "kv_cache_policy_claimed": true,
            "runtime_kv_cache_allocated": false,
            "kv_cache_cuda_residency_claimed": false,
            "estimated_bytes_only": true,
            "transfer_bytes_measured": false,
            "transfer_timing_measured": false,
            "fallback_used": false,
            "sampling_integration_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "remaining_model_boundary_gaps": {
            "schema": 1,
            "gaps": [
                {
                    "gap": "sampling",
                    "status": "not_governed_by_kv_cache_policy",
                    "required_next_proof": "dense_gguf_sampling_policy_receipt",
                    "blocks_qwen_one_token": true,
                    "blocks_qwen_short_decode": true,
                    "blocks_qwen_chat": true
                }
            ],
            "kv_cache_policy_claimed": true,
            "sampling_integration_claimed": false,
            "qwen_one_token_cuda_blocked": true,
            "qwen_short_decode_cuda_blocked": true,
            "qwen_chat_cuda_blocked": true,
            "next_required_proof": "dense_gguf_sampling_policy_receipt",
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": false,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": true,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": true,
            "dense_gguf_one_layer_cpu_reference_claimed": true,
            "dense_gguf_one_layer_cuda_integrated_parity_claimed": true,
            "dense_gguf_all_layer_execution_plan_claimed": true,
            "dense_gguf_model_boundary_fixtures_claimed": true,
            "kv_cache_policy_claimed": true,
            "kv_cache_cuda_residency_claimed": false,
            "dense_gguf_one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "sampling_integration_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "error": null
    })
}

fn valid_dense_gguf_sampling_policy_receipt() -> Value {
    let mut receipt = valid_dense_gguf_kv_cache_policy_receipt();
    receipt["artifact_kind"] = json!("dense_gguf_sampling_policy");
    receipt["artifact_path"] = json!("target/bitnet/receipts/dense-gguf-sampling-policy.json");
    receipt["claim"] = json!("dense_gguf_sampling_policy_recorded");
    receipt["model"]["file"] = json!("synthetic-dense-gguf-sampling-policy");
    receipt["execution_path"]["kernel_family"] = json!("dense_cuda_sampling_policy_route");
    receipt["execution_path"]["quantization_family"] =
        json!("dense_gguf_q8_0_f16_logits_sampling_policy_contract");
    receipt["sampling_policy"] = json!({
        "schema": 1,
        "policy_id": "dense_gguf_sampling_policy_qwen_vocab8_top3",
        "policy_scope": "dense_qwen_logits_to_sampler_boundary",
        "logits_source": "dense_gguf_model_boundary_lm_head_logits",
        "logits_sha256": format!("{:064x}", 22),
        "logits_len": 8,
        "vocab_size": 8,
        "seq_len": 4,
        "logits_dtype": "f32",
        "logits_element_bytes": 4,
        "logits_transfer_bytes_per_step_estimate": 32,
        "logits_transfer_path": "cuda_lm_head_logits_to_cpu_sampler",
        "logits_transfer_required_for_cpu_sampling": true,
        "logits_transfer_bytes_measured": false,
        "logits_transfer_timing_measured": false,
        "sampler_backend": "bitnet-sampling",
        "sampler_location": "cpu",
        "sampler_mode": "greedy",
        "temperature": 0.0,
        "top_k_filter": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "deterministic": true,
        "tie_break_policy": "lowest_token_id",
        "rng_required": false,
        "selected_token_id_from_fixture_logits": 2,
        "selected_token_scope": "fixture_logits_only_not_generation",
        "top_k": 3,
        "top_k_entries": [
            {"rank": 0, "token_id": 2, "value": 3.0},
            {"rank": 1, "token_id": 1, "value": 2.0},
            {"rank": 2, "token_id": 0, "value": 1.0}
        ],
        "sampling_policy_claimed": true,
        "sampling_integration_claimed": false,
        "qwen_one_token_cuda_claimed": false,
        "qwen_short_decode_cuda_claimed": false,
        "qwen_chat_cuda_claimed": false,
        "dense_gguf_inference_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "speedup_claim": false,
        "persistent_session_residency_claimed": false,
        "full_cuda_residency_claimed": false
    });
    receipt["remaining_model_boundary_gaps"] = json!({
        "schema": 1,
        "gaps": [],
        "all_model_boundary_policies_governed": true,
        "kv_cache_policy_claimed": true,
        "sampling_policy_claimed": true,
        "sampling_integration_claimed": false,
        "qwen_one_token_cuda_blocked": false,
        "qwen_short_decode_cuda_blocked": true,
        "qwen_chat_cuda_blocked": true,
        "next_required_proof": "qwen_one_token_strict_cuda_proof",
        "dense_gguf_inference_claimed": false,
        "speedup_claim": false,
        "full_cuda_residency_claimed": false
    });
    receipt["claim_boundary"]["sampling_policy_claimed"] = json!(true);
    receipt
}

fn valid_dense_gguf_qwen_one_token_strict_cuda_proof_receipt() -> Value {
    let mut receipt = valid_dense_gguf_sampling_policy_receipt();
    receipt["artifact_kind"] = json!("dense_gguf_qwen_one_token_strict_cuda_proof");
    receipt["artifact_path"] = json!("target/bitnet/receipts/dense-gguf-qwen-one-token.json");
    receipt["claim"] = json!("dense_gguf_qwen_one_token_strict_cuda_proof_recorded");
    receipt["model"]["id"] = json!("qwen2.5-0.5b-instruct-q8_0");
    receipt["model"]["file"] = json!("qwen2.5-0.5b-instruct-q8_0.gguf");
    receipt["model"]["architecture"] = json!("qwen2");
    receipt["model"]["sha256"] =
        json!("ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e");
    receipt["execution_path"]["kernel_family"] = json!("dense_qwen_one_token_strict_cuda");
    receipt["execution_path"]["quantization_family"] =
        json!("dense_gguf_q8_0_f16_qwen_one_token_contract");
    receipt["execution_plan"]["cuda_dense_regular_llm_ops"] = json!(31);
    receipt["execution_plan"]["total_ops"] = json!(31);
    receipt["execution_plan"]["cuda_ops"] = json!(31);
    receipt["prerequisite_receipts"] = json!({
        "schema": 1,
        "all_layer_execution_plan_artifact_kind": "dense_gguf_all_layer_execution_plan",
        "all_layer_execution_plan_receipt_sha256": format!("{:064x}", 41),
        "model_boundary_fixtures_artifact_kind": "dense_gguf_model_boundary_fixtures",
        "model_boundary_fixtures_receipt_sha256": format!("{:064x}", 42),
        "kv_cache_policy_artifact_kind": "dense_gguf_kv_cache_policy",
        "kv_cache_policy_receipt_sha256": format!("{:064x}", 43),
        "sampling_policy_artifact_kind": "dense_gguf_sampling_policy",
        "sampling_policy_receipt_sha256": format!("{:064x}", 44),
        "all_required_receipts_verified": true,
        "all_layer_execution_plan_claimed": true,
        "model_boundary_fixtures_claimed": true,
        "kv_cache_policy_claimed": true,
        "sampling_policy_claimed": true
    });
    receipt["tokenizer_prompt_authority"] = json!({
        "schema": 1,
        "tokenizer_authority": "contract_authoritative",
        "prompt_authority": "contract_authoritative",
        "prompt_template": "qwen-chat-raw-deterministic",
        "bos_policy": "contract_default",
        "deterministic_prompt": true,
        "prompt_token_count": 4,
        "prompt_token_ids_sha256": format!("{:064x}", 45),
        "rendered_prompt_sha256": format!("{:064x}", 46)
    });
    receipt["one_token_proof"] = json!({
        "schema": 1,
        "proof_scope": "qwen_strict_one_token_greedy_decode",
        "model_family": "qwen",
        "requested_new_tokens": 1,
        "generated_tokens_count": 1,
        "generation_policy": "greedy",
        "deterministic": true,
        "fallback_used": false,
        "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
        "cuda_target_backend": "nvidia-rtx-5070-ti-cuda",
        "prompt_token_count": 4,
        "prompt_token_ids_sha256": format!("{:064x}", 45),
        "cpu_selected_token_id": 2,
        "cuda_selected_token_id": 2,
        "selected_token_match": true,
        "decoded_token_text": "test",
        "cpu_logits_top_k_sha256": format!("{:064x}", 47),
        "cuda_logits_top_k_sha256": format!("{:064x}", 47),
        "top_k_evidence_recorded": true,
        "top_k_compared": true,
        "top_k_match": true,
        "qwen_one_token_cuda_claimed": true,
        "qwen_short_decode_cuda_claimed": false,
        "qwen_chat_cuda_claimed": false,
        "dense_gguf_inference_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "speedup_claim": false,
        "server_ready_claimed": false,
        "full_cuda_residency_claimed": false
    });
    receipt["quality_gate"] = json!({
        "schema": 1,
        "gate": "qwen_one_token_cuda_parity",
        "passed": true,
        "answer_ready_claimed": false,
        "short_decode_claimed": false,
        "chat_claimed": false
    });
    receipt["kernel_stats"] = json!([
        {
            "phase": "transformer_blocks",
            "kernel_id": "dense_f16_gemm_cuda",
            "invocations": 28,
            "fallback_invocations": 0,
            "cpu_fallback_invocations": 0,
            "host_to_device_bytes": 4096,
            "device_to_host_bytes": 1024,
            "kernel_launches": 28,
            "kernel_time_ms": null
        },
        {
            "phase": "lm_head",
            "kernel_id": "dense_f16_lm_head_cuda",
            "invocations": 1,
            "fallback_invocations": 0,
            "cpu_fallback_invocations": 0,
            "host_to_device_bytes": 512,
            "device_to_host_bytes": 128,
            "kernel_launches": 1,
            "kernel_time_ms": null
        },
        {
            "phase": "logits_transfer",
            "kernel_id": "dense_logits_transfer_cuda",
            "invocations": 2,
            "fallback_invocations": 0,
            "cpu_fallback_invocations": 0,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 256,
            "kernel_launches": 2,
            "kernel_time_ms": null
        }
    ]);
    receipt["kernel_coverage"] = json!({
        "schema": 1,
        "route": "dense_regular_llm_cuda",
        "kernels_executed": [
            "dense_f16_gemm_cuda",
            "dense_f16_lm_head_cuda",
            "dense_logits_transfer_cuda"
        ],
        "all_required_dense_kernels_executed": true,
        "dense_kernel_invocations": 31,
        "dense_kernel_launches": 31,
        "bitnet_qk256_kernel_invocations": 0,
        "cpu_fallback_kernel_invocations": 0,
        "fallback_used": false
    });
    receipt["timing"] = json!({
        "total_ms": 12.5,
        "first_token_ms": 12.5,
        "logits_download_ms": 0.4,
        "kernel_time_ms": 3.25,
        "host_to_device_bytes": 4608,
        "device_to_host_bytes": 1408,
        "host_to_device_ms": 100.0,
        "host_to_device_ms_source": "wall_clock_model_load_with_cuda_weight_upload",
        "host_to_device_ms_scope": "model_load_wall_clock_envelope",
        "host_to_device_ms_includes_non_transfer_overhead": true,
        "device_to_host_ms": 0.4,
        "device_to_host_ms_source": "wall_clock_extract_logits_2d_local",
        "transfer_timing_status": "host_to_device_model_load_envelope_device_to_host_measured",
        "kernel_invocations": 31,
        "kernel_launches": 31
    });
    receipt["tensor_residency"] = json!({
        "schema": 1,
        "scope": "qwen_one_token_strict_cuda",
        "model_class": "dense_regular_llm",
        "residency_accounting_recorded": true,
        "weights_uploaded_once": true,
        "weights_resident_on_cuda": true,
        "per_token_weight_upload": false,
        "kv_cache_policy_recorded": true,
        "sampling_policy_recorded": true,
        "fallback_used": false,
        "dense_gguf_inference_claimed": false,
        "persistent_session_residency_claimed": false,
        "full_cuda_residency_claimed": false,
        "transfer_accounting": {
            "status": "measured",
            "host_to_device_bytes": 4608,
            "device_to_host_bytes": 1408,
            "host_to_device_ms": 100.0,
            "host_to_device_ms_source": "wall_clock_model_load_with_cuda_weight_upload",
            "host_to_device_ms_scope": "model_load_wall_clock_envelope",
            "host_to_device_ms_includes_non_transfer_overhead": true,
            "device_to_host_ms": 0.4,
            "device_to_host_ms_source": "wall_clock_extract_logits_2d_local",
            "transfer_timing_status": "host_to_device_model_load_envelope_device_to_host_measured",
            "kernel_invocations": 31,
            "kernel_launches": 31
        }
    });
    receipt["claim_boundary"]["dense_tensor_residency_claimed"] = json!(true);
    receipt["claim_boundary"]["dense_gguf_linear_cuda_parity_claimed"] = json!(true);
    receipt["claim_boundary"]["dense_gguf_linear_role_sweep_cuda_parity_claimed"] = json!(true);
    receipt["claim_boundary"]["dense_gguf_norm_cuda_parity_claimed"] = json!(true);
    receipt["claim_boundary"]["dense_gguf_rope_cuda_parity_claimed"] = json!(true);
    receipt["claim_boundary"]["dense_gguf_attention_score_cuda_parity_claimed"] = json!(true);
    receipt["claim_boundary"]["dense_gguf_attention_softmax_cuda_parity_claimed"] = json!(true);
    receipt["claim_boundary"]["dense_gguf_attention_v_mix_cuda_parity_claimed"] = json!(true);
    receipt["claim_boundary"]["dense_gguf_mlp_activation_cuda_parity_claimed"] = json!(true);
    receipt["claim_boundary"]["qwen_one_token_cuda_claimed"] = json!(true);
    receipt["claim_boundary"]["server_ready_claimed"] = json!(false);
    receipt
}

fn valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt() -> Value {
    let mut receipt = valid_dense_gguf_qwen_one_token_strict_cuda_proof_receipt();
    receipt["artifact_kind"] = json!("dense_gguf_qwen_short_decode_strict_cuda_proof");
    receipt["artifact_path"] = json!("target/bitnet/receipts/dense-gguf-qwen-short-decode.json");
    receipt["claim"] = json!("dense_gguf_qwen_short_decode_strict_cuda_proof_recorded");
    receipt["execution_path"]["kernel_family"] = json!("dense_qwen_short_decode_strict_cuda");
    receipt["execution_path"]["quantization_family"] =
        json!("dense_gguf_q8_0_f16_qwen_short_decode_contract");
    receipt["execution_plan"]["cuda_dense_regular_llm_ops"] = json!(264);
    receipt["execution_plan"]["total_ops"] = json!(264);
    receipt["execution_plan"]["cuda_ops"] = json!(264);
    receipt["prerequisite_receipts"]["one_token_proof_artifact_kind"] =
        json!("dense_gguf_qwen_one_token_strict_cuda_proof");
    receipt["prerequisite_receipts"]["one_token_proof_receipt_sha256"] =
        json!(format!("{:064x}", 48));
    receipt["prerequisite_receipts"]["one_token_proof_claimed"] = json!(true);

    let token_ids = vec![576_u64, 2, 2, 2, 2, 2, 2, 2];
    let steps = token_ids
        .iter()
        .enumerate()
        .map(|(index, token)| {
            json!({
                "index": index as u64,
                "cpu_selected_token_id": *token,
                "cuda_selected_token_id": *token,
                "selected_token_match": true,
                "cpu_logits_top_k_sha256": format!("{:064x}", 50 + index),
                "cuda_logits_top_k_sha256": format!("{:064x}", 50 + index),
                "cpu_logits_sha256": format!("{:064x}", 70 + index),
                "cuda_logits_sha256": format!("{:064x}", 80 + index),
                "logits_vector_length": 32,
                "cpu_top_k": [
                    {"rank": 1, "token_id": *token, "value": 1.0},
                    {"rank": 2, "token_id": 3, "value": 0.5}
                ],
                "cuda_top_k": [
                    {"rank": 1, "token_id": *token, "value": 1.0},
                    {"rank": 2, "token_id": 3, "value": 0.5}
                ],
                "top_k_match": true,
                "top_k_max_abs_error": 0.0,
                "top_k_mean_abs_error": 0.0,
                "cuda_step_timing": {
                    "embed_ms": 0.5,
                    "forward_ms": 3.0,
                    "logits_ms": 0.75,
                    "logits_download_ms": 0.4,
                    "decode_ms": 4.25,
                    "logits_device_is_cuda": true
                }
            })
        })
        .collect::<Vec<_>>();
    receipt["short_decode_proof"] = json!({
        "schema": 1,
        "proof_scope": "qwen_strict_short_decode_greedy",
        "model_family": "qwen",
        "requested_new_tokens": 8,
        "generated_tokens_count": 8,
        "generation_policy": "greedy",
        "deterministic": true,
        "fallback_used": false,
        "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
        "cuda_target_backend": "nvidia-rtx-5070-ti-cuda",
        "prompt_token_count": 4,
        "prompt_token_ids_sha256": format!("{:064x}", 45),
        "cpu_generated_token_ids": token_ids,
        "cuda_generated_token_ids": token_ids,
        "cpu_generated_token_ids_sha256": format!("{:064x}", 49),
        "cuda_generated_token_ids_sha256": format!("{:064x}", 49),
        "generated_token_ids_match": true,
        "first_token_divergence_index": null,
        "cpu_logits_top_k_steps_sha256": format!("{:064x}", 90),
        "cuda_logits_top_k_steps_sha256": format!("{:064x}", 90),
        "top_k_evidence_recorded": true,
        "top_k_compared": true,
        "top_k_all_match": true,
        "first_top_k_divergence_index": null,
        "top_k_max_abs_error": 0.0,
        "top_k_mean_abs_error": 0.0,
        "steps": steps,
        "decoded_text": "test answer",
        "qwen_one_token_cuda_claimed": true,
        "qwen_short_decode_cuda_claimed": true,
        "qwen_chat_cuda_claimed": false,
        "dense_gguf_inference_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "speedup_claim": false,
        "server_ready_claimed": false,
        "full_cuda_residency_claimed": false
    });
    receipt["quality_gate"] = json!({
        "schema": 1,
        "gate": "qwen_short_decode_cuda_parity",
        "passed": true,
        "answer_ready_claimed": false,
        "short_decode_claimed": true,
        "chat_claimed": false
    });
    receipt["kernel_stats"] = json!([
        {
            "phase": "qwen_short_decode_runtime",
            "kernel_id": "dense_qwen_short_decode_cuda_runtime",
            "invocations": 248,
            "fallback_invocations": 0,
            "cpu_fallback_invocations": 0,
            "host_to_device_bytes": 4096,
            "device_to_host_bytes": 1024,
            "kernel_launches": 248,
            "kernel_time_ms": 25.0
        },
        {
            "phase": "lm_head",
            "kernel_id": "dense_qwen_lm_head_cuda",
            "invocations": 8,
            "fallback_invocations": 0,
            "cpu_fallback_invocations": 0,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
            "kernel_launches": 8,
            "kernel_time_ms": 6.0
        },
        {
            "phase": "logits_transfer",
            "kernel_id": "dense_qwen_logits_transfer_cuda",
            "invocations": 8,
            "fallback_invocations": 0,
            "cpu_fallback_invocations": 0,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
            "kernel_launches": 8,
            "kernel_time_ms": 0.0
        }
    ]);
    receipt["logits_transfer_reduction"] = json!({
        "schema": 1,
        "scope": "dense_qwen_logits_top_k_transfer",
        "transfer_mode": "full_logits_download_cpu_sampler",
        "sampling_location": "cpu",
        "requested_top_k": 2,
        "generated_tokens_count": 8,
        "logits_vector_length": 32,
        "logits_element_bytes": 4,
        "full_logits_bytes_per_step": 128,
        "full_logits_download_bytes": 1024,
        "actual_device_to_host_bytes": 1024,
        "top_k_result_bytes_per_step_floor": 24,
        "top_k_result_bytes_total_floor": 192,
        "selected_token_bytes_total_floor": 32,
        "device_to_host_bytes_reduced": false,
        "bytes_saved_vs_full_logits": 0,
        "selected_token_equality_preserved": true,
        "top_k_evidence_preserved": true,
        "quality_receipts_unchanged": true,
        "reduction_blocker": "cpu_sampler_requires_full_logits_until_device_top_k_sampler"
    });
    receipt["kernel_coverage"] = json!({
        "schema": 1,
        "route": "dense_regular_llm_cuda",
        "kernels_executed": [
            "dense_qwen_short_decode_cuda_runtime",
            "dense_qwen_lm_head_cuda",
            "dense_qwen_logits_transfer_cuda"
        ],
        "all_required_dense_kernels_executed": true,
        "dense_kernel_invocations": 264,
        "dense_kernel_launches": 264,
        "bitnet_qk256_kernel_invocations": 0,
        "cpu_fallback_kernel_invocations": 0,
        "fallback_used": false
    });
    receipt["timing"] = json!({
        "total_ms": 50.0,
        "first_token_ms": 8.0,
        "prefill_ms": 10.0,
        "decode_total_ms": 40.0,
        "logits_download_ms_total": 3.2,
        "kernel_time_ms": 31.0,
        "generated_tokens_count": 8,
        "host_to_device_bytes": 4096,
        "device_to_host_bytes": 1024,
        "host_to_device_ms": 100.0,
        "host_to_device_ms_source": "wall_clock_model_load_with_cuda_weight_upload",
        "host_to_device_ms_scope": "model_load_wall_clock_envelope",
        "host_to_device_ms_includes_non_transfer_overhead": true,
        "device_to_host_ms": 3.2,
        "device_to_host_ms_source": "wall_clock_extract_logits_2d_local",
        "transfer_timing_status": "host_to_device_model_load_envelope_device_to_host_measured",
        "kernel_invocations": 264,
        "kernel_launches": 264
    });
    receipt["tensor_residency"] = json!({
        "schema": 1,
        "scope": "qwen_short_decode_strict_cuda",
        "model_class": "dense_regular_llm",
        "residency_accounting_recorded": true,
        "weights_uploaded_once": true,
        "weights_resident_on_cuda": true,
        "per_token_weight_upload": false,
        "kv_cache_policy_recorded": true,
        "sampling_policy_recorded": true,
        "runtime_logits_cuda_resident_before_download": true,
        "fallback_used": false,
        "dense_gguf_inference_claimed": false,
        "persistent_session_residency_claimed": false,
        "full_cuda_residency_claimed": false,
        "transfer_accounting": {
            "status": "measured",
            "host_to_device_bytes": 4096,
            "device_to_host_bytes": 1024,
            "host_to_device_ms": 100.0,
            "host_to_device_ms_source": "wall_clock_model_load_with_cuda_weight_upload",
            "host_to_device_ms_scope": "model_load_wall_clock_envelope",
            "host_to_device_ms_includes_non_transfer_overhead": true,
            "device_to_host_ms": 3.2,
            "device_to_host_ms_source": "wall_clock_extract_logits_2d_local",
            "transfer_timing_status": "host_to_device_model_load_envelope_device_to_host_measured",
            "kernel_invocations": 264,
            "kernel_launches": 264
        }
    });
    receipt["claim_boundary"]["qwen_short_decode_cuda_claimed"] = json!(true);
    receipt
}

fn retarget_receipt_to_qwen3(receipt: &mut Value) {
    receipt["model"]["id"] = json!("qwen3-0.6b-instruct-q8_0");
    receipt["model"]["file"] = json!("Qwen3-0.6B-Q8_0.gguf");
    receipt["model"]["architecture"] = json!("qwen3");
    receipt["model"]["sha256"] =
        json!("9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031");
}

fn update_single_decode_token_count(receipt: &mut Value, count: usize) {
    let token_ids = (0..count).map(|idx| 1000_u64 + idx as u64).collect::<Vec<_>>();
    let steps = token_ids
        .iter()
        .enumerate()
        .map(|(index, token)| {
            json!({
                "index": index as u64,
                "cpu_selected_token_id": *token,
                "cuda_selected_token_id": *token,
                "selected_token_match": true,
                "cpu_logits_top_k_sha256": format!("{:064x}", 500 + index),
                "cuda_logits_top_k_sha256": format!("{:064x}", 500 + index),
                "cpu_logits_sha256": format!("{:064x}", 700 + index),
                "cuda_logits_sha256": format!("{:064x}", 900 + index),
                "logits_vector_length": 32,
                "cpu_top_k": [
                    {"rank": 1, "token_id": *token, "value": 1.0},
                    {"rank": 2, "token_id": 3, "value": 0.5}
                ],
                "cuda_top_k": [
                    {"rank": 1, "token_id": *token, "value": 1.0},
                    {"rank": 2, "token_id": 3, "value": 0.5}
                ],
                "top_k_match": true,
                "top_k_max_abs_error": 0.0,
                "top_k_mean_abs_error": 0.0,
                "cuda_step_timing": {
                    "embed_ms": 0.5,
                    "forward_ms": 3.0,
                    "logits_ms": 0.75,
                    "logits_download_ms": 0.4,
                    "decode_ms": 4.25,
                    "logits_device_is_cuda": true
                }
            })
        })
        .collect::<Vec<_>>();
    let count_u64 = count as u64;
    let transfer_bytes = count_u64 * 128;
    let transformer_invocations = count_u64 * 31;
    let total_invocations = transformer_invocations + count_u64 + count_u64;

    receipt["short_decode_proof"]["requested_new_tokens"] = json!(count_u64);
    receipt["short_decode_proof"]["generated_tokens_count"] = json!(count_u64);
    receipt["short_decode_proof"]["cpu_generated_token_ids"] = json!(token_ids.clone());
    receipt["short_decode_proof"]["cuda_generated_token_ids"] = json!(token_ids);
    receipt["short_decode_proof"]["cpu_generated_token_ids_sha256"] =
        json!(format!("{:064x}", 444));
    receipt["short_decode_proof"]["cuda_generated_token_ids_sha256"] =
        json!(format!("{:064x}", 444));
    receipt["short_decode_proof"]["steps"] = json!(steps);
    receipt["logits_transfer_reduction"]["generated_tokens_count"] = json!(count_u64);
    receipt["logits_transfer_reduction"]["full_logits_download_bytes"] = json!(transfer_bytes);
    receipt["logits_transfer_reduction"]["actual_device_to_host_bytes"] = json!(transfer_bytes);
    receipt["logits_transfer_reduction"]["top_k_result_bytes_total_floor"] = json!(count_u64 * 24);
    receipt["logits_transfer_reduction"]["selected_token_bytes_total_floor"] = json!(count_u64 * 4);
    receipt["kernel_stats"][0]["invocations"] = json!(transformer_invocations);
    receipt["kernel_stats"][0]["kernel_launches"] = json!(transformer_invocations);
    receipt["kernel_stats"][0]["device_to_host_bytes"] = json!(transfer_bytes);
    receipt["kernel_stats"][1]["invocations"] = json!(count_u64);
    receipt["kernel_stats"][1]["kernel_launches"] = json!(count_u64);
    receipt["kernel_stats"][2]["invocations"] = json!(count_u64);
    receipt["kernel_stats"][2]["kernel_launches"] = json!(count_u64);
    receipt["kernel_coverage"]["dense_kernel_invocations"] = json!(total_invocations);
    receipt["kernel_coverage"]["dense_kernel_launches"] = json!(total_invocations);
    receipt["timing"]["generated_tokens_count"] = json!(count_u64);
    receipt["timing"]["device_to_host_bytes"] = json!(transfer_bytes);
    receipt["timing"]["device_to_host_ms"] = json!(count_u64 as f64 * 0.4);
    receipt["timing"]["kernel_invocations"] = json!(total_invocations);
    receipt["timing"]["kernel_launches"] = json!(total_invocations);
    receipt["tensor_residency"]["transfer_accounting"]["device_to_host_bytes"] =
        json!(transfer_bytes);
    receipt["tensor_residency"]["transfer_accounting"]["device_to_host_ms"] =
        json!(count_u64 as f64 * 0.4);
    receipt["tensor_residency"]["transfer_accounting"]["kernel_invocations"] =
        json!(total_invocations);
    receipt["tensor_residency"]["transfer_accounting"]["kernel_launches"] =
        json!(total_invocations);
}

fn valid_dense_gguf_qwen_short_decode_32_capture_receipt() -> Value {
    let mut receipt = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();
    retarget_receipt_to_qwen3(&mut receipt);
    update_single_decode_token_count(&mut receipt, 32);
    receipt["short_decode_proof"]["profile_id"] = json!("qwen3_short_decode_32");
    receipt["short_decode_proof"]["proof_scope"] = json!("qwen3_strict_short_decode_32_greedy");
    receipt["parity"]["fixture_id"] = json!("qwen3-0.6b-instruct-q8_0-short-decode-32-greedy");
    receipt
}

fn valid_dense_gguf_qwen_warm_decode_strict_cuda_proof_receipt()
-> Result<Value, Box<dyn std::error::Error>> {
    let mut receipt = valid_dense_gguf_qwen_short_decode_32_capture_receipt();
    update_single_decode_token_count(&mut receipt, 128);
    receipt["artifact_kind"] = json!("dense_gguf_qwen_warm_decode_strict_cuda_proof");
    receipt["artifact_path"] = json!("target/bitnet/receipts/dense-gguf-qwen-warm-decode.json");
    receipt["claim"] = json!("dense_gguf_qwen_warm_decode_strict_cuda_proof_recorded");
    receipt["execution_path"]["kernel_family"] = json!("dense_qwen_warm_decode_strict_cuda");
    receipt["execution_path"]["quantization_family"] =
        json!("dense_gguf_q8_0_f16_qwen_warm_decode_contract");
    receipt["execution_plan"]["quantization"] =
        json!("dense_gguf_q8_0_f16_qwen_warm_decode_contract");
    let proof = receipt
        .as_object_mut()
        .ok_or("warm-decode fixture must be a JSON object")?
        .remove("short_decode_proof")
        .ok_or("warm-decode fixture must contain short_decode_proof")?;
    receipt["warm_decode_proof"] = proof;
    receipt["warm_decode_proof"]["proof_scope"] = json!("qwen3_strict_warm_decode_128_greedy");
    receipt["warm_decode_proof"]["profile_id"] = json!("qwen3_warm_decode_128");
    receipt["warm_decode_proof"]["warm_context_reused"] = json!(true);
    receipt["warm_decode_proof"]["decode_started_from_prefilled_context"] = json!(true);
    receipt["warm_decode_proof"]["warm_context_prompt_token_count"] = json!(4);
    receipt["warm_decode_proof"]["qwen_warm_decode_cuda_claimed"] = json!(true);
    receipt["warm_decode_proof"]["qwen_chat_cuda_claimed"] = json!(false);
    receipt["warm_decode_proof"]["server_ready_claimed"] = json!(false);
    receipt["warm_decode_proof"]["speedup_claim"] = json!(false);
    receipt["warm_decode_proof"]["full_cuda_residency_claimed"] = json!(false);
    receipt["quality_gate"] = json!({
        "schema": 1,
        "gate": "qwen_warm_decode_cuda_parity",
        "passed": true,
        "warm_context_decode_claimed": true,
        "ask_claimed": false,
        "chat_claimed": false,
        "server_ready_claimed": false
    });
    receipt["warm_context_proof"] = json!({
        "schema": 1,
        "proof_scope": "qwen3_decode_128_from_warm_context",
        "profile_id": "decode_128_from_warm_context",
        "warm_context_reused": true,
        "decode_started_from_prefilled_context": true,
        "warm_context_prompt_token_count": 4,
        "prompt_token_ids_sha256": receipt["tokenizer_prompt_authority"]["prompt_token_ids_sha256"].clone(),
        "rendered_prompt_sha256": receipt["tokenizer_prompt_authority"]["rendered_prompt_sha256"].clone(),
        "requested_new_tokens": 128,
        "generated_tokens_count": 128,
        "model_loaded_once": true,
        "cuda_context_initialized_once": true,
        "weights_uploaded_once": true,
        "per_request_model_load": false,
        "fallback_used": false,
        "speedup_claim": false,
        "server_ready_claimed": false,
        "full_cuda_residency_claimed": false
    });
    receipt["session_lifecycle"] = json!({
        "schema": 1,
        "proof_scope": "qwen3_warm_decode_strict_cuda",
        "model_loaded_once": true,
        "tokenizer_loaded_once": true,
        "cuda_context_initialized_once": true,
        "cuda_context_once": true,
        "weights_uploaded_once": true,
        "per_request_model_load": false,
        "per_token_weight_upload": false,
        "workspace_reused": true,
        "runtime_buffers_reused": true,
        "warm_context_reused": true,
        "decode_started_from_prefilled_context": true,
        "fallback_used": false,
        "scoped_warm_context_residency_claimed": true,
        "persistent_session_residency_claimed": false,
        "full_cuda_residency_claimed": false
    });
    receipt["tensor_residency"]["scope"] = json!("qwen_warm_decode_strict_cuda");
    receipt["tensor_residency"]["warm_context_reused"] = json!(true);
    receipt["tensor_residency"]["scoped_warm_context_residency_claimed"] = json!(true);
    receipt["claim_boundary"]["qwen_warm_decode_cuda_claimed"] = json!(true);
    receipt["claim_boundary"]["qwen_chat_cuda_claimed"] = json!(false);
    receipt["claim_boundary"]["server_ready_claimed"] = json!(false);
    receipt["claim_boundary"]["speedup_claim"] = json!(false);
    receipt["claim_boundary"]["persistent_session_residency_claimed"] = json!(false);
    receipt["claim_boundary"]["full_cuda_residency_claimed"] = json!(false);
    receipt["parity"]["fixture_id"] = json!("qwen3-0.6b-instruct-q8_0-warm-decode-128-greedy");
    Ok(receipt)
}

fn valid_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt() -> Value {
    let mut receipt = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();
    receipt["artifact_kind"] = json!("dense_gguf_qwen_warm_session_strict_cuda_proof");
    receipt["artifact_path"] = json!("target/bitnet/receipts/dense-gguf-qwen-warm-session.json");
    receipt["claim"] = json!("dense_gguf_qwen_warm_session_strict_cuda_proof_recorded");
    receipt["execution_path"]["kernel_family"] = json!("dense_qwen_warm_session_strict_cuda");
    receipt["execution_path"]["quantization_family"] =
        json!("dense_gguf_q8_0_f16_qwen_warm_session_contract");
    receipt["execution_plan"]["cuda_dense_regular_llm_ops"] = json!(792);
    receipt["execution_plan"]["total_ops"] = json!(792);
    receipt["execution_plan"]["cuda_ops"] = json!(792);
    receipt["prerequisite_receipts"]["short_decode_proof_artifact_kind"] =
        json!("dense_gguf_qwen_short_decode_strict_cuda_proof");
    receipt["prerequisite_receipts"]["short_decode_proof_receipt_sha256"] =
        json!(format!("{:064x}", 93));
    receipt["prerequisite_receipts"]["short_decode_proof_claimed"] = json!(true);

    let prompt_turns = (0..3)
        .map(|index| {
            json!({
                "index": index,
                "prompt_token_count": 4,
                "prompt_token_ids_sha256": format!("{:064x}", 100 + index),
                "rendered_prompt_sha256": format!("{:064x}", 110 + index),
                "rendered_prompt_bytes": 16
            })
        })
        .collect::<Vec<_>>();
    receipt["tokenizer_prompt_authority"] = json!({
        "schema": 1,
        "tokenizer_authority": "contract_authoritative",
        "prompt_authority": "contract_authoritative",
        "prompt_template": "qwen-chat-raw-deterministic",
        "bos_policy": "contract_default_add_bos",
        "deterministic_prompt": true,
        "turns_count": 3,
        "prompt_token_count_total": 12,
        "prompt_token_ids_sha256": format!("{:064x}", 120),
        "rendered_prompt_sha256": format!("{:064x}", 121),
        "turns": prompt_turns
    });
    receipt["session_lifecycle"] = json!({
        "schema": 1,
        "proof_scope": "qwen_warm_session_strict_cuda",
        "turns_count": 3,
        "model_loaded_once": true,
        "tokenizer_loaded_once": true,
        "cuda_context_initialized_once": true,
        "cuda_context_once": true,
        "weights_uploaded_once": true,
        "per_request_model_load": false,
        "per_turn_weight_upload": false,
        "runtime_buffers_reused": true,
        "workspace_reused": true,
        "kv_cache_policy_recorded": true,
        "kv_cache_reinitialized_per_turn": true,
        "sampling_policy_recorded": true,
        "fallback_used": false,
        "scoped_warm_session_residency_claimed": true,
        "persistent_session_residency_claimed": false,
        "full_cuda_residency_claimed": false
    });

    let turn_tokens = [
        vec![576_u64, 4226, 374, 220, 19, 13, 3555, 374],
        vec![421_u64, 279, 6437, 374, 6303, 13, 576, 2],
        vec![9707_u64, 0, 576, 374, 264, 2805, 13, 2],
    ];
    let turns = turn_tokens
        .iter()
        .enumerate()
        .map(|(turn_index, token_ids)| {
            let steps = token_ids
                .iter()
                .enumerate()
                .map(|(index, token)| {
                    json!({
                        "index": index as u64,
                        "cpu_selected_token_id": *token,
                        "cuda_selected_token_id": *token,
                        "selected_token_match": true,
                        "cpu_logits_top_k_sha256": format!("{:064x}", 130 + turn_index * 10 + index),
                        "cuda_logits_top_k_sha256": format!("{:064x}", 130 + turn_index * 10 + index),
                        "cpu_logits_sha256": format!("{:064x}", 170 + turn_index * 10 + index),
                        "cuda_logits_sha256": format!("{:064x}", 200 + turn_index * 10 + index),
                        "logits_vector_length": 32,
                        "cpu_top_k": [
                            {"rank": 1, "token_id": *token, "value": 1.0},
                            {"rank": 2, "token_id": 3, "value": 0.5}
                        ],
                        "cuda_top_k": [
                            {"rank": 1, "token_id": *token, "value": 1.0},
                            {"rank": 2, "token_id": 3, "value": 0.5}
                        ],
                        "top_k_match": true,
                        "top_k_max_abs_error": 0.0,
                        "top_k_mean_abs_error": 0.0,
                        "cuda_step_timing": {
                            "embed_ms": 0.5,
                            "forward_ms": 3.0,
                            "logits_ms": 0.75,
                            "logits_download_ms": 0.4,
                            "decode_ms": 4.25,
                            "logits_device_is_cuda": true
                        }
                    })
                })
                .collect::<Vec<_>>();
            json!({
                "index": turn_index as u64,
                "prompt_token_count": 4,
                "prompt_token_ids_sha256": format!("{:064x}", 100 + turn_index),
                "rendered_prompt_sha256": format!("{:064x}", 110 + turn_index),
                "requested_new_tokens": 8,
                "generated_tokens_count": 8,
                "cpu_generated_token_ids": token_ids,
                "cuda_generated_token_ids": token_ids,
                "cpu_generated_token_ids_sha256": format!("{:064x}", 220 + turn_index),
                "cuda_generated_token_ids_sha256": format!("{:064x}", 220 + turn_index),
                "generated_token_ids_match": true,
                "first_token_divergence_index": null,
                "cpu_logits_top_k_steps_sha256": format!("{:064x}", 230 + turn_index),
                "cuda_logits_top_k_steps_sha256": format!("{:064x}", 230 + turn_index),
                "top_k_all_match": true,
                "first_top_k_divergence_index": null,
                "steps": steps,
                "decoded_text": "test answer",
                "cuda_turn_timing": {
                    "total_ms": 50.0,
                    "first_token_ms": 8.0,
                    "prefill_ms": 10.0,
                    "decode_total_ms": 40.0,
                    "embed_ms_total": 4.0,
                    "forward_ms_total": 24.0,
                    "logits_ms_total": 6.0,
                    "logits_download_ms_total": 3.2,
                    "logits_device_all_cuda_resident": true
                }
            })
        })
        .collect::<Vec<_>>();
    receipt["warm_session_proof"] = json!({
        "schema": 1,
        "proof_scope": "qwen_strict_warm_session_greedy",
        "model_family": "qwen",
        "turns_count": 3,
        "requested_new_tokens_per_turn": 8,
        "generated_tokens_total": 24,
        "generation_policy": "greedy",
        "deterministic": true,
        "fallback_used": false,
        "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
        "cuda_target_backend": "nvidia-rtx-5070-ti-cuda",
        "cpu_generated_token_ids_sha256": format!("{:064x}", 240),
        "cuda_generated_token_ids_sha256": format!("{:064x}", 240),
        "generated_token_ids_match": true,
        "first_token_divergence": null,
        "cuda_logits_top_k_session_sha256": format!("{:064x}", 241),
        "top_k_evidence_recorded": true,
        "top_k_compared": true,
        "top_k_all_match": true,
        "first_top_k_divergence": null,
        "top_k_max_abs_error": 0.0,
        "top_k_mean_abs_error": 0.0,
        "turns": turns,
        "qwen_one_token_cuda_claimed": true,
        "qwen_short_decode_cuda_claimed": true,
        "qwen_warm_session_cuda_claimed": true,
        "qwen_chat_cuda_claimed": false,
        "dense_gguf_inference_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "speedup_claim": false,
        "server_ready_claimed": false,
        "full_cuda_residency_claimed": false
    });
    receipt["quality_gate"] = json!({
        "schema": 1,
        "gate": "qwen_warm_session_cuda_parity",
        "passed": true,
        "answer_ready_claimed": false,
        "short_decode_claimed": true,
        "warm_session_claimed": true,
        "chat_claimed": false
    });
    receipt["kernel_stats"] = json!([
        {
            "phase": "qwen_warm_session_runtime",
            "kernel_id": "dense_qwen_warm_session_cuda_runtime",
            "invocations": 744,
            "fallback_invocations": 0,
            "cpu_fallback_invocations": 0,
            "host_to_device_bytes": 4096,
            "device_to_host_bytes": 3072,
            "kernel_launches": 744,
            "kernel_time_ms": 75.0
        },
        {
            "phase": "lm_head",
            "kernel_id": "dense_qwen_lm_head_cuda",
            "invocations": 24,
            "fallback_invocations": 0,
            "cpu_fallback_invocations": 0,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
            "kernel_launches": 24,
            "kernel_time_ms": 18.0
        },
        {
            "phase": "logits_transfer",
            "kernel_id": "dense_qwen_logits_transfer_cuda",
            "invocations": 24,
            "fallback_invocations": 0,
            "cpu_fallback_invocations": 0,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
            "kernel_launches": 24,
            "kernel_time_ms": 0.0
        }
    ]);
    receipt["logits_transfer_reduction"] = json!({
        "schema": 1,
        "scope": "dense_qwen_logits_top_k_transfer",
        "transfer_mode": "full_logits_download_cpu_sampler",
        "sampling_location": "cpu",
        "requested_top_k": 2,
        "generated_tokens_count": 24,
        "logits_vector_length": 32,
        "logits_element_bytes": 4,
        "full_logits_bytes_per_step": 128,
        "full_logits_download_bytes": 3072,
        "actual_device_to_host_bytes": 3072,
        "top_k_result_bytes_per_step_floor": 24,
        "top_k_result_bytes_total_floor": 576,
        "selected_token_bytes_total_floor": 96,
        "device_to_host_bytes_reduced": false,
        "bytes_saved_vs_full_logits": 0,
        "selected_token_equality_preserved": true,
        "top_k_evidence_preserved": true,
        "quality_receipts_unchanged": true,
        "reduction_blocker": "cpu_sampler_requires_full_logits_until_device_top_k_sampler"
    });
    receipt["kernel_coverage"] = json!({
        "schema": 1,
        "route": "dense_regular_llm_cuda",
        "kernels_executed": [
            "dense_qwen_warm_session_cuda_runtime",
            "dense_qwen_lm_head_cuda",
            "dense_qwen_logits_transfer_cuda"
        ],
        "all_required_dense_kernels_executed": true,
        "dense_kernel_invocations": 792,
        "dense_kernel_launches": 792,
        "bitnet_qk256_kernel_invocations": 0,
        "cpu_fallback_kernel_invocations": 0,
        "fallback_used": false
    });
    receipt["timing"] = json!({
        "total_ms": 175.0,
        "cpu_reference_total_ms": 300.0,
        "cuda_context_init_ms": 3.0,
        "tokenizer_load_ms": 2.0,
        "model_load_ms": 100.0,
        "cpu_reference_model_load_ms": 100.0,
        "first_token_ms": 8.0,
        "prefill_ms": 30.0,
        "decode_total_ms": 120.0,
        "logits_download_ms_total": 9.6,
        "kernel_time_ms": 93.0,
        "host_to_device_bytes": 4096,
        "device_to_host_bytes": 3072,
        "host_to_device_ms": 100.0,
        "host_to_device_ms_source": "wall_clock_model_load_with_cuda_weight_upload",
        "host_to_device_ms_scope": "model_load_wall_clock_envelope",
        "host_to_device_ms_includes_non_transfer_overhead": true,
        "device_to_host_ms": 9.6,
        "device_to_host_ms_source": "wall_clock_extract_logits_2d_local",
        "transfer_timing_status": "host_to_device_model_load_envelope_device_to_host_measured",
        "kernel_invocations": 792,
        "kernel_launches": 792,
        "turns_count": 3,
        "generated_tokens_total": 24
    });
    receipt["tensor_residency"] = json!({
        "schema": 1,
        "scope": "qwen_warm_session_strict_cuda",
        "model_class": "dense_regular_llm",
        "residency_accounting_recorded": true,
        "model_loaded_once": true,
        "tokenizer_loaded_once": true,
        "cuda_context_initialized_once": true,
        "cuda_context_once": true,
        "weights_uploaded_once": true,
        "weights_resident_on_cuda": true,
        "per_request_model_load": false,
        "per_turn_weight_upload": false,
        "per_token_weight_upload": false,
        "runtime_buffers_reused": true,
        "workspace_reused": true,
        "kv_cache_policy_recorded": true,
        "kv_cache_reinitialized_per_turn": true,
        "sampling_policy_recorded": true,
        "runtime_logits_cuda_resident_before_download": true,
        "fallback_used": false,
        "dense_gguf_inference_claimed": false,
        "scoped_warm_session_residency_claimed": true,
        "persistent_session_residency_claimed": false,
        "full_cuda_residency_claimed": false,
        "transfer_accounting": {
            "status": "measured",
            "host_to_device_bytes": 4096,
            "device_to_host_bytes": 3072,
            "host_to_device_ms": 100.0,
            "host_to_device_ms_source": "wall_clock_model_load_with_cuda_weight_upload",
            "host_to_device_ms_scope": "model_load_wall_clock_envelope",
            "host_to_device_ms_includes_non_transfer_overhead": true,
            "device_to_host_ms": 9.6,
            "device_to_host_ms_source": "wall_clock_extract_logits_2d_local",
            "transfer_timing_status": "host_to_device_model_load_envelope_device_to_host_measured",
            "kernel_invocations": 792,
            "kernel_launches": 792
        }
    });
    receipt["claim_boundary"]["qwen_warm_session_cuda_claimed"] = json!(true);
    receipt["claim_boundary"]["scoped_warm_session_residency_claimed"] = json!(true);
    receipt["claim_boundary"]["qwen_ask_cuda_claimed"] = json!(false);
    receipt
}

fn valid_dense_gguf_qwen_ask_strict_cuda_proof_receipt() -> Value {
    let source = valid_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt();
    let source_proof = source["short_decode_proof"].clone();
    let source_prerequisites = source["prerequisite_receipts"].clone();
    let mut receipt = source.clone();

    receipt["artifact_kind"] = json!("dense_gguf_qwen_ask_strict_cuda_proof");
    receipt["artifact_path"] = json!("target/bitnet/receipts/dense-qwen-ask.json");
    receipt["claim"] = json!("dense_gguf_qwen_ask_strict_cuda_proof_recorded");
    receipt["execution_path"]["kernel_family"] = json!("dense_qwen_ask_strict_cuda");
    receipt["execution_path"]["quantization_family"] =
        json!("dense_gguf_q8_0_f16_qwen_ask_contract");
    receipt["question"] = json!("What is 2+2?");
    receipt["answer"] = json!("test answer");
    receipt["receipt"] = json!({
        "path": "target/bitnet/receipts/dense-qwen-ask.json",
        "defaulted_for_dense_cuda_ask": true
    });
    receipt["prerequisite_receipts"] = json!({
        "schema": 1,
        "all_layer_execution_plan_artifact_kind": source_prerequisites["all_layer_execution_plan_artifact_kind"].clone(),
        "all_layer_execution_plan_receipt_sha256": source_prerequisites["all_layer_execution_plan_receipt_sha256"].clone(),
        "model_boundary_fixtures_artifact_kind": source_prerequisites["model_boundary_fixtures_artifact_kind"].clone(),
        "model_boundary_fixtures_receipt_sha256": source_prerequisites["model_boundary_fixtures_receipt_sha256"].clone(),
        "kv_cache_policy_artifact_kind": source_prerequisites["kv_cache_policy_artifact_kind"].clone(),
        "kv_cache_policy_receipt_sha256": source_prerequisites["kv_cache_policy_receipt_sha256"].clone(),
        "sampling_policy_artifact_kind": source_prerequisites["sampling_policy_artifact_kind"].clone(),
        "sampling_policy_receipt_sha256": source_prerequisites["sampling_policy_receipt_sha256"].clone(),
        "one_token_proof_artifact_kind": source_prerequisites["one_token_proof_artifact_kind"].clone(),
        "one_token_proof_receipt_sha256": source_prerequisites["one_token_proof_receipt_sha256"].clone(),
        "short_decode_proof_artifact_kind": "dense_gguf_qwen_short_decode_strict_cuda_proof",
        "short_decode_proof_receipt_sha256": format!("{:064x}", 250),
        "warm_session_proof_artifact_kind": "dense_gguf_qwen_warm_session_strict_cuda_proof",
        "warm_session_proof_receipt_sha256": format!("{:064x}", 251),
        "all_required_receipts_verified": true,
        "all_layer_execution_plan_claimed": true,
        "model_boundary_fixtures_claimed": true,
        "kv_cache_policy_claimed": true,
        "sampling_policy_claimed": true,
        "one_token_proof_claimed": true,
        "short_decode_proof_claimed": true,
        "warm_session_proof_claimed": true
    });
    receipt["ask_proof"] = json!({
        "schema": 1,
        "proof_scope": "qwen_strict_cuda_ask_from_short_decode",
        "model_family": "qwen",
        "question": "What is 2+2?",
        "answer": "test answer",
        "requested_new_tokens": 8,
        "generated_tokens_count": 8,
        "generation_policy": "greedy",
        "deterministic": true,
        "fallback_used": false,
        "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
        "cuda_target_backend": "nvidia-rtx-5070-ti-cuda",
        "prompt_token_count": source_proof["prompt_token_count"].clone(),
        "prompt_token_ids_sha256": source_proof["prompt_token_ids_sha256"].clone(),
        "cpu_generated_token_ids": source_proof["cpu_generated_token_ids"].clone(),
        "cuda_generated_token_ids": source_proof["cuda_generated_token_ids"].clone(),
        "cpu_generated_token_ids_sha256": source_proof["cpu_generated_token_ids_sha256"].clone(),
        "cuda_generated_token_ids_sha256": source_proof["cuda_generated_token_ids_sha256"].clone(),
        "generated_token_ids_match": true,
        "first_token_divergence_index": null,
        "top_k_evidence_recorded": true,
        "top_k_compared": true,
        "top_k_all_match": true,
        "first_top_k_divergence_index": null,
        "top_k_max_abs_error": 0.0,
        "top_k_mean_abs_error": 0.0,
        "qwen_one_token_cuda_claimed": true,
        "qwen_short_decode_cuda_claimed": true,
        "qwen_warm_session_cuda_claimed": true,
        "qwen_ask_cuda_claimed": true,
        "qwen_chat_cuda_claimed": false,
        "dense_gguf_inference_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "speedup_claim": false,
        "server_ready_claimed": false,
        "full_cuda_residency_claimed": false
    });
    receipt["quality_gate"] = json!({
        "schema": 1,
        "gate": "qwen_cuda_ask_answer",
        "passed": true,
        "ask_claimed": true,
        "chat_claimed": false
    });
    receipt["quality"] = json!({
        "passed": true,
        "gate": "qwen_cuda_ask_answer",
        "ask_claimed": true,
        "chat_claimed": false
    });
    receipt["tensor_residency"]["scope"] = json!("qwen_ask_strict_cuda");
    receipt["residency"] = json!({
        "weights_uploaded_once": true,
        "per_token_weight_upload": false,
        "full_cuda_residency_claimed": false
    });
    receipt["claim_boundary"]["qwen_warm_session_cuda_claimed"] = json!(true);
    receipt["claim_boundary"]["qwen_ask_cuda_claimed"] = json!(true);
    receipt["claim_boundary"]["qwen_chat_cuda_claimed"] = json!(false);
    receipt["claim_boundary"]["server_ready_claimed"] = json!(false);
    receipt["claim_boundary"]["speedup_claim"] = json!(false);
    receipt["claim_boundary"]["persistent_session_residency_claimed"] = json!(false);
    receipt["claim_boundary"]["full_cuda_residency_claimed"] = json!(false);
    receipt["source_short_decode_receipt"] = source;
    receipt
}

fn valid_dense_gguf_qwen_chat_strict_cuda_proof_receipt() -> Value {
    let source = valid_dense_gguf_qwen_warm_session_strict_cuda_proof_receipt();
    let source_proof = source["warm_session_proof"].clone();
    let source_prerequisites = source["prerequisite_receipts"].clone();
    let source_turns = source_proof["turns"].as_array().unwrap();
    let turns = source_turns
        .iter()
        .enumerate()
        .map(|(index, source_turn)| {
            json!({
                "index": index as u64,
                "user_message": format!("turn {}", index + 1),
                "assistant_answer": source_turn["decoded_text"].clone(),
                "prompt_token_count": source_turn["prompt_token_count"].clone(),
                "prompt_token_ids_sha256": source_turn["prompt_token_ids_sha256"].clone(),
                "rendered_prompt_sha256": source_turn["rendered_prompt_sha256"].clone(),
                "requested_new_tokens": source_turn["requested_new_tokens"].clone(),
                "generated_tokens_count": source_turn["generated_tokens_count"].clone(),
                "cpu_generated_token_ids": source_turn["cpu_generated_token_ids"].clone(),
                "cuda_generated_token_ids": source_turn["cuda_generated_token_ids"].clone(),
                "cpu_generated_token_ids_sha256": source_turn["cpu_generated_token_ids_sha256"].clone(),
                "cuda_generated_token_ids_sha256": source_turn["cuda_generated_token_ids_sha256"].clone(),
                "generated_token_ids_match": true,
                "first_token_divergence_index": null,
                "top_k_all_match": source_turn["top_k_all_match"].clone(),
                "first_top_k_divergence_index": source_turn["first_top_k_divergence_index"].clone(),
            })
        })
        .collect::<Vec<_>>();

    let mut receipt = source.clone();
    receipt["artifact_kind"] = json!("dense_gguf_qwen_chat_strict_cuda_proof");
    receipt["artifact_path"] = json!("target/bitnet/receipts/dense-qwen-chat.json");
    receipt["claim"] = json!("dense_gguf_qwen_chat_strict_cuda_proof_recorded");
    receipt["execution_path"]["kernel_family"] = json!("dense_qwen_chat_strict_cuda");
    receipt["execution_path"]["quantization_family"] =
        json!("dense_gguf_q8_0_f16_qwen_chat_contract");
    receipt["execution_plan"]["quantization"] = json!("dense_gguf_q8_0_f16_qwen_chat_contract");
    receipt["answers"] = json!(["test answer", "test answer", "test answer"]);
    receipt["receipt"] = json!({
        "path": "target/bitnet/receipts/dense-qwen-chat.json",
        "defaulted_for_dense_cuda_chat": true
    });
    receipt["prerequisite_receipts"] = json!({
        "schema": 1,
        "all_layer_execution_plan_artifact_kind": source_prerequisites["all_layer_execution_plan_artifact_kind"].clone(),
        "all_layer_execution_plan_receipt_sha256": source_prerequisites["all_layer_execution_plan_receipt_sha256"].clone(),
        "model_boundary_fixtures_artifact_kind": source_prerequisites["model_boundary_fixtures_artifact_kind"].clone(),
        "model_boundary_fixtures_receipt_sha256": source_prerequisites["model_boundary_fixtures_receipt_sha256"].clone(),
        "kv_cache_policy_artifact_kind": source_prerequisites["kv_cache_policy_artifact_kind"].clone(),
        "kv_cache_policy_receipt_sha256": source_prerequisites["kv_cache_policy_receipt_sha256"].clone(),
        "sampling_policy_artifact_kind": source_prerequisites["sampling_policy_artifact_kind"].clone(),
        "sampling_policy_receipt_sha256": source_prerequisites["sampling_policy_receipt_sha256"].clone(),
        "one_token_proof_artifact_kind": source_prerequisites["one_token_proof_artifact_kind"].clone(),
        "one_token_proof_receipt_sha256": source_prerequisites["one_token_proof_receipt_sha256"].clone(),
        "short_decode_proof_artifact_kind": source_prerequisites["short_decode_proof_artifact_kind"].clone(),
        "short_decode_proof_receipt_sha256": source_prerequisites["short_decode_proof_receipt_sha256"].clone(),
        "warm_session_proof_artifact_kind": "dense_gguf_qwen_warm_session_strict_cuda_proof",
        "warm_session_proof_receipt_sha256": format!("{:064x}", 252),
        "all_required_receipts_verified": true,
        "all_layer_execution_plan_claimed": true,
        "model_boundary_fixtures_claimed": true,
        "kv_cache_policy_claimed": true,
        "sampling_policy_claimed": true,
        "one_token_proof_claimed": true,
        "short_decode_proof_claimed": true,
        "warm_session_proof_claimed": true
    });
    receipt["chat_session"] = json!({
        "schema": 1,
        "proof_scope": "qwen_strict_cuda_chat_from_warm_session",
        "model_family": "qwen",
        "turns_count": source_proof["turns_count"].clone(),
        "requested_new_tokens_per_turn": source_proof["requested_new_tokens_per_turn"].clone(),
        "generated_tokens_total": source_proof["generated_tokens_total"].clone(),
        "generation_policy": "greedy",
        "deterministic": true,
        "fallback_used": false,
        "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
        "cuda_target_backend": "nvidia-rtx-5070-ti-cuda",
        "cpu_generated_token_ids_sha256": source_proof["cpu_generated_token_ids_sha256"].clone(),
        "cuda_generated_token_ids_sha256": source_proof["cuda_generated_token_ids_sha256"].clone(),
        "generated_token_ids_match": true,
        "first_token_divergence": null,
        "top_k_evidence_recorded": true,
        "top_k_compared": true,
        "top_k_all_match": source_proof["top_k_all_match"].clone(),
        "first_top_k_divergence": source_proof["first_top_k_divergence"].clone(),
        "top_k_max_abs_error": source_proof["top_k_max_abs_error"].clone(),
        "top_k_mean_abs_error": source_proof["top_k_mean_abs_error"].clone(),
        "turns": turns,
        "qwen_one_token_cuda_claimed": true,
        "qwen_short_decode_cuda_claimed": true,
        "qwen_warm_session_cuda_claimed": true,
        "qwen_ask_cuda_claimed": false,
        "qwen_chat_cuda_claimed": true,
        "dense_gguf_inference_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "speedup_claim": false,
        "server_ready_claimed": false,
        "full_cuda_residency_claimed": false
    });
    receipt["quality_gate"] = json!({
        "schema": 1,
        "gate": "qwen_cuda_chat_session",
        "passed": true,
        "chat_claimed": true,
        "server_claimed": false
    });
    receipt["quality"] = json!({
        "passed": true,
        "gate": "qwen_cuda_chat_session",
        "chat_claimed": true,
        "server_claimed": false
    });
    receipt["tensor_residency"]["scope"] = json!("qwen_chat_strict_cuda");
    receipt["residency"] = json!({
        "weights_uploaded_once": true,
        "runtime_buffers_reused": true,
        "per_turn_weight_upload": false,
        "full_cuda_residency_claimed": false
    });
    receipt["claim_boundary"]["qwen_ask_cuda_claimed"] = json!(false);
    receipt["claim_boundary"]["qwen_chat_cuda_claimed"] = json!(true);
    receipt["claim_boundary"]["server_ready_claimed"] = json!(false);
    receipt["claim_boundary"]["speedup_claim"] = json!(false);
    receipt["claim_boundary"]["persistent_session_residency_claimed"] = json!(false);
    receipt["claim_boundary"]["full_cuda_residency_claimed"] = json!(false);
    receipt["source_warm_session_receipt"] = source;
    receipt
}

fn dense_boundary_fixture(
    name: &str,
    role: &str,
    tensor_name: &str,
    tensor_type: &str,
    source_shape: Value,
    value_count: u64,
    output_len: u64,
    output_sha256: String,
) -> Value {
    json!({
        "name": name,
        "role": role,
        "tensor_name": tensor_name,
        "tensor_type": tensor_type,
        "source_shape": source_shape,
        "source_offset": 0,
        "source_size_bytes": 128,
        "value_count": value_count,
        "output_len": output_len,
        "output_sha256": output_sha256,
        "max_abs": 1.0,
        "dense_gguf_inference_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false
    })
}

fn valid_dense_gguf_one_layer_cpu_reference_receipt() -> Value {
    json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_one_layer_cpu_reference",
        "artifact_path": "target/bitnet/receipts/dense-gguf-one-layer-cpu-reference.json",
        "claim": "dense_gguf_one_layer_cpu_reference_recorded",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "cpu-reference",
        "timestamp_utc": "2026-05-09T00:00:00Z",
        "requested_backend": "cpu_reference",
        "selected_backend": "cpu_reference",
        "runtime_api": "cpu",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "model": {
            "model_family": "qwen",
            "architecture": "qwen3",
            "artifact_kind": "dense_gguf",
            "file": "synthetic-dense-gguf-one-layer-cpu-reference",
            "sha256": "0".repeat(64)
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "cpu_reference_dense_one_layer",
            "quantization_family": "dense_gguf_materialized_f32_reference",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": 11,
            "metadata_count": 4,
            "required_roles_present": true,
            "strict_descriptor_complete": true,
            "dense_cuda_route_status": "descriptor_only_quant_bridge_required",
            "quantization_families": ["f32", "q8_0"],
            "bitnet_packed_marker_found": false,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "reference_harness": {
            "schema": 1,
            "fixture_id": "dense_gguf_one_layer_cpu_reference_qwen_layer0_s4",
            "layer_index": 0,
            "seq_len": 4,
            "position_offset": 1,
            "hidden_size": 4,
            "q_heads": 2,
            "kv_heads": 1,
            "heads_per_kv_group": 2,
            "head_dim": 2,
            "intermediate_size": 6,
            "rmsnorm_eps": 1e-6,
            "epsilon_source": "default_1e-6",
            "rope_base": 1000000.0,
            "rope_base_source": "qwen3.rope.freq_base",
            "rope_scaling_factor": 1.0,
            "deterministic_input_len": 16,
            "deterministic_input_sha256": format!("{:064x}", 3),
            "phases_total": 17,
            "phases": dense_one_layer_cpu_reference_phases(),
            "final_output_len": 16,
            "final_output_sha256": format!("{:064x}", 19),
            "final_output_max_abs": 1.0,
            "cpu_reference_only": true,
            "cuda_execution_claimed": false,
            "one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false,
            "next_required_proof": "one_layer_cuda_integrated_parity"
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": false,
            "dense_tensor_residency_claimed": false,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": true,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": false,
            "dense_gguf_one_layer_cpu_reference_claimed": true,
            "dense_gguf_one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "error": null
    })
}

fn valid_dense_gguf_one_layer_cuda_integrated_parity_receipt() -> Value {
    let phases = dense_one_layer_cuda_integrated_phases();
    let kernel_stats = dense_one_layer_cuda_kernel_stats();
    let h2d: u64 =
        kernel_stats.iter().map(|stat| stat["host_to_device_bytes"].as_u64().unwrap()).sum();
    let d2h: u64 =
        kernel_stats.iter().map(|stat| stat["device_to_host_bytes"].as_u64().unwrap()).sum();
    let invocations: u64 =
        kernel_stats.iter().map(|stat| stat["invocations"].as_u64().unwrap()).sum();
    let launches: u64 =
        kernel_stats.iter().map(|stat| stat["kernel_launches"].as_u64().unwrap()).sum();

    json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_one_layer_cuda_integrated_parity",
        "artifact_path": "target/bitnet/receipts/dense-gguf-one-layer-cuda-parity.json",
        "claim": "dense_gguf_one_layer_cuda_integrated_parity_recorded",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia-rtx-5070-ti-cuda",
        "timestamp_utc": "2026-05-09T00:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": {
            "available": true,
            "device_count": 1,
            "device_index": 0,
            "device_name": "NVIDIA GeForce RTX 5070 Ti",
            "compute_capability": "12.0",
            "driver_version": "591.86",
            "cuda_runtime_version": "12.9",
            "cuda_toolkit_version": "12.9",
            "nvrtc_version": "12.9",
            "nvml_available": true,
            "vram_bytes": 17094475776_u64
        },
        "model": {
            "model_family": "qwen",
            "architecture": "qwen3",
            "artifact_kind": "dense_gguf",
            "file": "synthetic-dense-gguf-one-layer-cuda-parity",
            "sha256": "0".repeat(64)
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_cuda_integrated_one_layer",
            "quantization_family": "dense_gguf_q8_0_f16_cuda_bridge",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": {
            "planner_version": "cuda-planner-004",
            "model_family": "qwen",
            "quantization": "dense_fp16",
            "selected_route": "dense_regular_llm_cuda",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "strict_fallback_policy": "reject",
            "dense_regular_llm_cuda": true,
            "bitnet_packed_qk256_cuda": false,
            "cuda_bitnet_qk256_ops": 0,
            "cuda_dense_regular_llm_ops": 14,
            "cpu_fallback_ops": 0,
            "unsupported_ops": 0,
            "total_ops": 14,
            "cuda_ops": 14,
            "mixed_cuda_routes": false,
            "fallback_used": false,
            "strict_cuda_ready": true,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": 11,
            "metadata_count": 4,
            "required_roles_present": true,
            "strict_descriptor_complete": true,
            "dense_cuda_route_status": "descriptor_only_quant_bridge_required",
            "quantization_families": ["f32", "q8_0"],
            "bitnet_packed_marker_found": false,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "cpu_reference": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_one_layer_cpu_reference",
            "fixture_id": "dense_gguf_one_layer_cpu_reference_qwen_layer0_s4",
            "layer_index": 0,
            "seq_len": 4,
            "position_offset": 1,
            "final_output_len": 16,
            "final_output_sha256": format!("{:064x}", 19),
            "final_output_max_abs": 1.0,
            "cpu_reference_only": true,
            "cuda_execution_claimed": false,
            "dense_gguf_inference_claimed": false
        },
        "cuda_layer": {
            "schema": 1,
            "fixture_id": "dense_gguf_one_layer_cuda_integrated_parity_qwen_layer0_s4",
            "source_cpu_reference_fixture_id": "dense_gguf_one_layer_cpu_reference_qwen_layer0_s4",
            "layer_index": 0,
            "seq_len": 4,
            "position_offset": 1,
            "hidden_size": 4,
            "q_heads": 2,
            "kv_heads": 1,
            "heads_per_kv_group": 2,
            "head_dim": 2,
            "intermediate_size": 6,
            "governed_cuda_ops_total": 14,
            "residual_host_ops_total": 2,
            "host_deterministic_input_ops_total": 1,
            "unsupported_ops_total": 0,
            "cpu_fallback_ops_total": 0,
            "strict_cuda_ready": true,
            "fallback_used": false,
            "phases_total": 17,
            "phases": phases,
            "final_output_len": 16,
            "final_output_sha256": format!("{:064x}", 19),
            "final_output_max_abs": 1.0,
            "final_output_max_abs_error": 0.0,
            "final_output_mean_abs_error": 0.0,
            "tolerance": 0.5,
            "passed": true,
            "one_layer_cuda_integrated_parity_claimed": true,
            "one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "kernel_stats": kernel_stats,
        "timing": {
            "kernel_time_ms": null,
            "host_to_device_bytes": h2d,
            "device_to_host_bytes": d2h,
            "kernel_invocations": invocations,
            "kernel_launches": launches
        },
        "tensor_residency": {
            "scope": "integrated_dense_gguf_one_layer",
            "model_class": "dense_regular_llm",
            "fixture_id": "dense_gguf_one_layer_cuda_integrated_parity_qwen_layer0_s4",
            "dense_tensor_residency_claimed": true,
            "integrated_one_layer_cuda_parity_claimed": true,
            "dense_gguf_inference_claimed": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false,
            "weights_uploaded_per_kernel": true,
            "weights_uploaded_once": false,
            "intermediate_downloads_for_phase_parity": true,
            "host_device_transfer_accounting_matches_kernel_stats": true,
            "transfer_accounting": {
                "status": "measured",
                "host_to_device_bytes": h2d,
                "device_to_host_bytes": d2h,
                "kernel_invocations": invocations,
                "kernel_launches": launches
            }
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": true,
            "dense_gguf_linear_cuda_parity_claimed": true,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": true,
            "dense_gguf_norm_cuda_parity_claimed": true,
            "dense_gguf_rope_cuda_parity_claimed": true,
            "dense_gguf_attention_score_cuda_parity_claimed": true,
            "dense_gguf_attention_softmax_cuda_parity_claimed": true,
            "dense_gguf_attention_v_mix_cuda_parity_claimed": true,
            "dense_gguf_mlp_activation_cuda_parity_claimed": true,
            "dense_gguf_one_layer_execution_plan_claimed": true,
            "dense_gguf_one_layer_cpu_reference_claimed": true,
            "dense_gguf_one_layer_cuda_integrated_parity_claimed": true,
            "dense_gguf_one_layer_inference_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "server_ready_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "error": null
    })
}

fn dense_one_layer_cpu_reference_phases() -> Vec<Value> {
    [
        ("deterministic_input", "hidden_state", "input"),
        ("attention_norm", "attention_norm", "rmsnorm"),
        ("attention_q", "attention_q", "matmul"),
        ("attention_k", "attention_k", "matmul"),
        ("attention_v", "attention_v", "matmul"),
        ("rope", "rope", "rope"),
        ("attention_scores", "attention_scores", "attention"),
        ("attention_softmax", "attention_softmax", "softmax"),
        ("attention_v_mix", "attention_v_mix", "attention"),
        ("attention_output", "attention_output", "matmul"),
        ("first_residual", "first_residual", "residual_add"),
        ("ffn_norm", "ffn_norm", "rmsnorm"),
        ("mlp_gate", "mlp_gate", "matmul"),
        ("mlp_up", "mlp_up", "matmul"),
        ("mlp_activation", "mlp_activation", "activation"),
        ("mlp_down", "mlp_down", "matmul"),
        ("second_residual", "second_residual", "residual_add"),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (name, role, op_type))| {
        json!({
            "index": index as u64,
            "name": name,
            "role": role,
            "op_type": op_type,
            "output_len": 16,
            "output_sha256": format!("{:064x}", index + 3),
            "max_abs": 1.0
        })
    })
    .collect()
}

fn dense_one_layer_cuda_integrated_phases() -> Vec<Value> {
    let phase_defs = [
        (
            "deterministic_input",
            "hidden_state",
            "input",
            "host_deterministic_input",
            "host_deterministic_input",
            None,
            1_u64,
            0_u64,
            0_u64,
            0_u64,
        ),
        (
            "attention_norm",
            "attention_norm",
            "rmsnorm",
            "dense_regular_llm_cuda",
            "cuda_executed",
            Some("dense_rmsnorm_f32_cuda"),
            1,
            64,
            64,
            1,
        ),
        (
            "attention_q",
            "attention_q",
            "matmul",
            "dense_regular_llm_cuda",
            "cuda_executed",
            Some("dense_f16_gemm_cuda"),
            4,
            384,
            64,
            4,
        ),
        (
            "attention_k",
            "attention_k",
            "matmul",
            "dense_regular_llm_cuda",
            "cuda_executed",
            Some("dense_f16_gemm_cuda"),
            4,
            384,
            32,
            4,
        ),
        (
            "attention_v",
            "attention_v",
            "matmul",
            "dense_regular_llm_cuda",
            "cuda_executed",
            Some("dense_f16_gemm_cuda"),
            4,
            384,
            32,
            4,
        ),
        (
            "rope",
            "rope",
            "rope",
            "dense_regular_llm_cuda",
            "cuda_executed",
            Some("dense_rope_f32_cuda"),
            2,
            96,
            96,
            2,
        ),
        (
            "attention_scores",
            "attention_scores",
            "attention",
            "dense_regular_llm_cuda",
            "cuda_executed",
            Some("dense_attention_scores_f32_cuda"),
            1,
            96,
            128,
            1,
        ),
        (
            "attention_softmax",
            "attention_softmax",
            "softmax",
            "dense_regular_llm_cuda",
            "cuda_executed",
            Some("dense_attention_softmax_f32_cuda"),
            1,
            128,
            128,
            1,
        ),
        (
            "attention_v_mix",
            "attention_v_mix",
            "attention",
            "dense_regular_llm_cuda",
            "cuda_executed",
            Some("dense_attention_v_mix_f32_cuda"),
            1,
            160,
            64,
            1,
        ),
        (
            "attention_output",
            "attention_output",
            "matmul",
            "dense_regular_llm_cuda",
            "cuda_executed",
            Some("dense_f16_gemm_cuda"),
            4,
            384,
            64,
            4,
        ),
        (
            "first_residual",
            "first_residual",
            "residual_add",
            "host_measured_glue",
            "host_measured_glue",
            None,
            1,
            0,
            0,
            0,
        ),
        (
            "ffn_norm",
            "ffn_norm",
            "rmsnorm",
            "dense_regular_llm_cuda",
            "cuda_executed",
            Some("dense_rmsnorm_f32_cuda"),
            1,
            64,
            64,
            1,
        ),
        (
            "mlp_gate",
            "mlp_gate",
            "matmul",
            "dense_regular_llm_cuda",
            "cuda_executed",
            Some("dense_f16_gemm_cuda"),
            4,
            512,
            96,
            4,
        ),
        (
            "mlp_up",
            "mlp_up",
            "matmul",
            "dense_regular_llm_cuda",
            "cuda_executed",
            Some("dense_f16_gemm_cuda"),
            4,
            512,
            96,
            4,
        ),
        (
            "mlp_activation",
            "mlp_activation",
            "activation",
            "dense_regular_llm_cuda",
            "cuda_executed",
            Some("dense_mlp_activation_f32_cuda"),
            1,
            192,
            96,
            1,
        ),
        (
            "mlp_down",
            "mlp_down",
            "matmul",
            "dense_regular_llm_cuda",
            "cuda_executed",
            Some("dense_f16_gemm_cuda"),
            4,
            432,
            64,
            4,
        ),
        (
            "second_residual",
            "second_residual",
            "residual_add",
            "host_measured_glue",
            "host_measured_glue",
            None,
            1,
            0,
            0,
            0,
        ),
    ];

    phase_defs
        .into_iter()
        .enumerate()
        .map(
            |(index, (name, role, op_type, route, status, kernel_id, invocations, h2d, d2h, launches))| {
                json!({
                    "index": index as u64,
                    "name": name,
                    "role": role,
                    "op_type": op_type,
                    "route": route,
                    "status": status,
                    "output_len": if name == "attention_scores" || name == "attention_softmax" { 32 } else if name == "mlp_gate" || name == "mlp_up" || name == "mlp_activation" { 24 } else { 16 },
                    "output_sha256": format!("{:064x}", index + 3),
                    "max_abs": 1.0,
                    "max_abs_error": 0.0,
                    "mean_abs_error": 0.0,
                    "tolerance": 0.5,
                    "passed": true,
                    "fallback_used": false,
                    "kernel_id": kernel_id,
                    "invocations": invocations,
                    "fallback_invocations": 0,
                    "host_to_device_bytes": h2d,
                    "device_to_host_bytes": d2h,
                    "kernel_launches": launches,
                    "kernel_time_ms": null,
                })
            },
        )
        .collect()
}

fn dense_one_layer_cuda_kernel_stats() -> Vec<Value> {
    dense_one_layer_cuda_integrated_phases()
        .into_iter()
        .filter(|phase| phase["kernel_id"].is_string())
        .map(|phase| {
            json!({
                "phase": phase["name"],
                "kernel_id": phase["kernel_id"],
                "invocations": phase["invocations"],
                "fallback_invocations": phase["fallback_invocations"],
                "host_to_device_bytes": phase["host_to_device_bytes"],
                "device_to_host_bytes": phase["device_to_host_bytes"],
                "kernel_launches": phase["kernel_launches"],
                "kernel_time_ms": null
            })
        })
        .collect()
}

fn valid_dense_gguf_norm_fixture_extraction_receipt() -> Value {
    json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_norm_fixture_extraction",
        "artifact_path": "target/bitnet/receipts/dense-gguf-norm-fixture.json",
        "claim": "dense_gguf_norm_fixture_extracted",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia-rtx-5070-ti-cuda",
        "timestamp_utc": "2026-05-09T00:00:00Z",
        "inspection_source": "gguf_reader_norm_fixture",
        "model": {
            "model_family": "qwen",
            "architecture": "qwen3",
            "artifact_kind": "dense_gguf",
            "quantization_families": ["f32", "q8_0"],
            "file": "synthetic-qwen3-q8_0-norm-fixture.gguf",
            "sha256": "0".repeat(64)
        },
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": 11,
            "metadata_count": 4,
            "required_roles_present": true,
            "strict_descriptor_complete": true,
            "dense_cuda_route_status": "descriptor_only_quant_bridge_required",
            "quantization_families": ["f32", "q8_0"],
            "bitnet_packed_marker_found": false,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "norm_fixture_audit": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_norm_fixture_extraction",
            "roles_total": 2,
            "roles_extracted": 2,
            "roles_failed": 0,
            "covered_roles": ["attention_norm", "ffn_norm"],
            "all_cpu_reference_computed": true,
            "cuda_kernel_status": "missing_cuda_kernel",
            "strict_cuda_ready": false,
            "cpu_fallback_allowed": false,
            "transfer_timing_status": "not_measured_no_kernel",
            "candidate_order": ["attention_norm", "ffn_norm"],
            "next_required_proof": "cuda_rmsnorm_kernel_parity",
            "dense_gguf_norm_fixture_extraction_claimed": true,
            "dense_gguf_inference_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "norm_fixtures": [
            dense_norm_fixture("blk.0.attn_norm.weight", "attention_norm"),
            dense_norm_fixture("blk.0.ffn_norm.weight", "ffn_norm")
        ],
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": false,
            "dense_tensor_residency_claimed": false,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_norm_fixture_extraction_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": false,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "cpu_cuda_parity_claimed": false,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "notes": [
            "Dense GGUF norm fixture extraction only; no CUDA norm kernel or dense GGUF inference was executed."
        ],
        "error": null
    })
}

fn valid_dense_gguf_norm_cuda_parity_receipt() -> Value {
    json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_norm_cuda_parity",
        "artifact_path": "target/bitnet/receipts/dense-gguf-norm-cuda-parity.json",
        "claim": "dense_gguf_norm_cuda_parity_tested",
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia-rtx-5070-ti-cuda",
        "timestamp_utc": "2026-05-09T00:00:00Z",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "cuda": cuda_identity(),
        "model": {
            "model_family": "qwen",
            "architecture": "qwen3",
            "artifact_kind": "dense_gguf",
            "quantization_families": ["f32", "q8_0"],
            "file": "synthetic-qwen3-q8_0-norm-cuda-parity.gguf",
            "sha256": "0".repeat(64)
        },
        "execution_path": {
            "model_class": "dense_regular_llm",
            "kernel_family": "dense_f32_rmsnorm",
            "quantization_family": "f32_norm_weights",
            "bitnet_packed_kernel_proof": false,
            "qk256_proof": false
        },
        "execution_plan": {
            "planner_version": "cuda-planner-004",
            "model_family": "qwen",
            "quantization": "dense_f32_rmsnorm",
            "selected_route": "dense_regular_llm_cuda",
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "strict_fallback_policy": "reject",
            "dense_regular_llm_cuda": true,
            "bitnet_packed_qk256_cuda": false,
            "cuda_bitnet_qk256_ops": 0,
            "cuda_dense_regular_llm_ops": 2,
            "cpu_fallback_ops": 0,
            "unsupported_ops": 0,
            "total_ops": 2,
            "cuda_ops": 2,
            "mixed_cuda_routes": false,
            "fallback_used": false,
            "strict_cuda_ready": true,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "descriptor_coverage": {
            "schema": 1,
            "source_artifact_kind": "dense_gguf_tensor_descriptor_inspection",
            "tensor_count": 11,
            "metadata_count": 4,
            "required_roles_present": true,
            "strict_descriptor_complete": true,
            "dense_cuda_route_status": "descriptor_only_quant_bridge_required",
            "quantization_families": ["f32", "q8_0"],
            "bitnet_packed_marker_found": false,
            "dense_gguf_inference_claimed": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "norm_fixtures": [
            dense_norm_cuda_fixture(
                "dense_gguf_rmsnorm_attention_norm",
                "blk.0.attn_norm.weight",
                "attention_norm"
            ),
            dense_norm_cuda_fixture(
                "dense_gguf_rmsnorm_ffn_norm",
                "blk.0.ffn_norm.weight",
                "ffn_norm"
            )
        ],
        "kernel_stats": [
            dense_norm_cuda_kernel_stat(
                "dense_gguf_rmsnorm_attention_norm",
                "blk.0.attn_norm.weight",
                "attention_norm"
            ),
            dense_norm_cuda_kernel_stat(
                "dense_gguf_rmsnorm_ffn_norm",
                "blk.0.ffn_norm.weight",
                "ffn_norm"
            )
        ],
        "parity_results": [
            dense_norm_cuda_parity_result("dense_gguf_rmsnorm_attention_norm", "attention_norm"),
            dense_norm_cuda_parity_result("dense_gguf_rmsnorm_ffn_norm", "ffn_norm")
        ],
        "parity": {
            "passed": true,
            "roles_total": 2,
            "covered_roles": ["attention_norm", "ffn_norm"],
            "first_divergence": null
        },
        "timing": {
            "kernel_time_ms": null,
            "host_to_device_bytes": 256,
            "device_to_host_bytes": 128
        },
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_descriptor_inspection_claimed": true,
            "dense_gguf_norm_fixture_extraction_claimed": true,
            "dense_gguf_norm_cuda_parity_claimed": true,
            "dense_gguf_linear_fixture_extraction_claimed": false,
            "dense_gguf_linear_cuda_parity_claimed": false,
            "dense_gguf_linear_role_sweep_cuda_parity_claimed": false,
            "dense_gguf_one_layer_execution_plan_claimed": false,
            "dense_gguf_inference_claimed": false,
            "qwen_one_token_cuda_claimed": false,
            "qwen_short_decode_cuda_claimed": false,
            "qwen_chat_cuda_claimed": false,
            "cpu_cuda_parity_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false
        },
        "tensor_residency": {
            "schema_version": "1.0.0",
            "scope": "single_dense_gguf_rmsnorm_fixture",
            "model_class": "dense_regular_llm",
            "roles_total": 2,
            "dense_tensor_residency_claimed": true,
            "dense_gguf_inference_claimed": false,
            "persistent_session_residency_claimed": false,
            "full_cuda_residency_claimed": false,
            "input_tensors_uploaded_once": true,
            "output_tensor_cuda_resident_during_kernel": true,
            "host_device_transfer_accounting_matches_kernel_stats": true,
            "allocation": {
                "device_buffer_count_per_role": 3,
                "temporary_workspace_bytes": 0,
                "persistent_handle_count": 0,
                "persistent_handles_claimed": false
            },
            "transfer_accounting": {
                "status": "measured",
                "host_to_device_bytes": 256,
                "device_to_host_bytes": 128
            },
            "kernel_launches": 2
        },
        "error": null
    })
}

fn dense_norm_fixture(tensor_name: &str, role: &str) -> Value {
    json!({
        "schema": 1,
        "artifact_kind": "dense_gguf_norm_fixture_extraction",
        "model_family": "qwen",
        "architecture": "qwen3",
        "tensor_name": tensor_name,
        "role": role,
        "tensor_type": "f32",
        "source_shape": [16],
        "source_offset": 1024,
        "source_size_bytes": 64,
        "hidden_dim": 16,
        "value_count": 16,
        "values_materialized_as_f32": true,
        "weight_values_sha256": "1".repeat(64),
        "rmsnorm_eps": 0.000001,
        "epsilon_source": "qwen3.attention.layer_norm_rms_epsilon",
        "cpu_reference_input_len": 16,
        "cpu_reference_output_len": 16,
        "cpu_reference_input_sha256": "2".repeat(64),
        "cpu_reference_output_sha256": "3".repeat(64),
        "cpu_reference_computed": true,
        "cuda_kernel_status": "missing_cuda_kernel",
        "dense_gguf_inference_claimed": false,
        "dense_regular_llm_cuda_claimed": false,
        "cpu_cuda_parity_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "speedup_claim": false,
        "full_cuda_residency_claimed": false
    })
}

fn dense_norm_cuda_fixture(fixture_id: &str, tensor_name: &str, role: &str) -> Value {
    json!({
        "schema": 1,
        "source_artifact_kind": "dense_gguf_norm_fixture_extraction",
        "fixture_id": fixture_id,
        "model_family": "qwen",
        "architecture": "qwen3",
        "tensor_name": tensor_name,
        "role": role,
        "tensor_type": "f32",
        "source_shape": [16],
        "hidden_dim": 16,
        "value_count": 16,
        "values_materialized_as_f32": true,
        "weight_values_sha256": "1".repeat(64),
        "rmsnorm_eps": 0.000001,
        "epsilon_source": "qwen3.attention.layer_norm_rms_epsilon",
        "cuda_input_dtype": "f32",
        "cuda_gamma_dtype": "f32",
        "cuda_output_dtype": "f32",
        "dense_gguf_inference_claimed": false,
        "dense_regular_llm_cuda_claimed": true,
        "cpu_cuda_parity_claimed": true,
        "bitnet_packed_i2s_qk256_proof": false,
        "speedup_claim": false,
        "full_cuda_residency_claimed": false
    })
}

fn dense_norm_cuda_kernel_stat(fixture_id: &str, tensor_name: &str, role: &str) -> Value {
    json!({
        "kernel_id": "dense_rmsnorm_f32_cuda",
        "role": role,
        "tensor_name": tensor_name,
        "fixture_id": fixture_id,
        "invocations": 1,
        "fallback_invocations": 0,
        "host_to_device_bytes": 128,
        "device_to_host_bytes": 64,
        "kernel_launches": 1,
        "kernel_time_ms": null
    })
}

fn dense_norm_cuda_parity_result(fixture_id: &str, role: &str) -> Value {
    json!({
        "reference_backend": "amd-9950x3d-cpu-avx512",
        "target_backend": "nvidia-rtx-5070-ti-cuda",
        "kernel_id": "dense_rmsnorm_f32_cuda",
        "fixture_id": fixture_id,
        "role": role,
        "hidden_dim": 16,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "passed": true,
        "tolerance": 0.00005,
        "tolerance_source": "CUDA-DENSE-016 dense GGUF RMSNorm F32 CUDA fixture"
    })
}

fn dense_one_layer_gap_audit() -> Value {
    json!({
        "schema": 1,
        "source_artifact_kind": "dense_gguf_one_layer_execution_plan",
        "layer_index": 0,
        "cuda_routable_ops_total": 14,
        "cuda_routable_linear_ops_total": 7,
        "cuda_routable_norm_ops_total": 2,
        "cuda_routable_rope_ops_total": 1,
        "cuda_routable_attention_score_ops_total": 1,
        "cuda_routable_attention_softmax_ops_total": 1,
        "cuda_routable_attention_v_mix_ops_total": 1,
        "cuda_routable_mlp_activation_ops_total": 1,
        "unsupported_ops_total": 0,
        "cpu_fallback_ops_total": 0,
        "strict_cuda_ready": true,
        "unsupported_ops_have_dependency_notes": true,
        "strict_cuda_rejects_cpu_fallback": true,
        "cuda_routable_roles": [
            "attention_norm",
            "attention_q",
            "attention_k",
            "attention_v",
            "rope",
            "attention_scores",
            "attention_softmax",
            "attention_v_mix",
            "attention_output",
            "ffn_norm",
            "mlp_gate",
            "mlp_up",
            "mlp_activation",
            "mlp_down"
        ],
        "linears_routable_roles": [
            "attention_q",
            "attention_k",
            "attention_v",
            "attention_output",
            "mlp_gate",
            "mlp_up",
            "mlp_down"
        ],
        "norms_routable_roles": [
            "attention_norm",
            "ffn_norm"
        ],
        "rope_routable_roles": [
            "rope"
        ],
        "attention_scores_routable_roles": [
            "attention_scores"
        ],
        "attention_softmax_routable_roles": [
            "attention_softmax"
        ],
        "attention_v_mix_routable_roles": [
            "attention_v_mix"
        ],
        "mlp_activation_routable_roles": [
            "mlp_activation"
        ],
        "rmsnorm_cuda_parity_available": true,
        "rope_cuda_parity_available": true,
        "attention_score_cuda_parity_available": true,
        "attention_softmax_cuda_parity_available": true,
        "attention_v_mix_cuda_parity_available": true,
        "mlp_activation_cuda_parity_available": true,
        "next_candidate_gap": "none",
        "next_required_proof": "one_layer_cpu_reference_harness",
        "unsupported_op_type_counts": {},
        "candidate_order": [],
        "dependency_edges": [
            { "from": "attention_norm", "to": "attention_q" },
            { "from": "attention_norm", "to": "attention_k" },
            { "from": "attention_norm", "to": "attention_v" },
            { "from": "attention_q", "to": "rope" },
            { "from": "attention_k", "to": "rope" },
            { "from": "rope", "to": "attention_scores" },
            { "from": "attention_scores", "to": "attention_softmax" },
            { "from": "attention_softmax", "to": "attention_v_mix" },
            { "from": "attention_v", "to": "attention_v_mix" },
            { "from": "attention_v_mix", "to": "attention_output" },
            { "from": "ffn_norm", "to": "mlp_gate" },
            { "from": "ffn_norm", "to": "mlp_up" },
            { "from": "mlp_gate", "to": "mlp_activation" },
            { "from": "mlp_up", "to": "mlp_activation" },
            { "from": "mlp_activation", "to": "mlp_down" }
        ],
        "unsupported_ops": [],
        "dense_gguf_one_layer_execution_plan_claimed": true,
        "dense_gguf_one_layer_inference_claimed": false,
        "dense_gguf_inference_claimed": false,
        "qwen_one_token_cuda_claimed": false,
        "qwen_short_decode_cuda_claimed": false,
        "qwen_chat_cuda_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "speedup_claim": false,
        "full_cuda_residency_claimed": false
    })
}

fn dense_all_layer_plan_layer(layer_index: u64) -> Value {
    let operations = dense_all_layer_operations(layer_index);
    let operation_signature_sha256 = dense_all_layer_operation_signature_sha256(&operations);
    json!({
        "layer_index": layer_index,
        "total_ops": 14,
        "cuda_routable_ops_total": 14,
        "linear_cuda_ops_total": 7,
        "norm_cuda_ops_total": 2,
        "rope_cuda_ops_total": 1,
        "attention_score_cuda_ops_total": 1,
        "attention_softmax_cuda_ops_total": 1,
        "attention_v_mix_cuda_ops_total": 1,
        "mlp_activation_cuda_ops_total": 1,
        "unsupported_strict_cuda_ops_total": 0,
        "cpu_fallback_ops_total": 0,
        "strict_cuda_ready": true,
        "matches_layer0": true,
        "operation_signature_sha256": operation_signature_sha256,
        "operations": operations
    })
}

fn dense_all_layer_operation_signature_sha256(operations: &Value) -> String {
    let signature = operations
        .as_array()
        .expect("operations array")
        .iter()
        .map(|op| {
            json!({
                "role": op["role"].as_str().expect("role"),
                "op_type": op["op_type"].as_str().expect("op_type"),
                "source": op["source"].as_str().expect("source"),
                "source_tensor_type": op.get("source_tensor_type").cloned().unwrap_or(Value::Null),
                "source_shape": op.get("source_shape").cloned().unwrap_or(Value::Null),
                "is_quantized": op.get("is_quantized").cloned().unwrap_or(Value::Bool(false)),
                "route": op["route"].as_str().expect("route"),
                "status": op["status"].as_str().expect("status"),
            })
        })
        .collect::<Vec<_>>();
    let bytes = serde_json::to_vec(&Value::Array(signature)).expect("signature json");
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn dense_all_layer_model_boundary_gaps() -> Value {
    json!({
        "schema": 1,
        "gaps": [
            dense_all_layer_boundary_gap("token_embedding", "embedding lookup fixture and route not yet governed", Some("token_embd.weight"), Some("q8_0"), "dense_gguf_embedding_fixture"),
            dense_all_layer_boundary_gap("final_norm", "final model normalization fixture not yet governed", Option::<&str>::None, Option::<&str>::None, "dense_gguf_final_norm_fixture"),
            dense_all_layer_boundary_gap("lm_head_logits", "LM head and logits fixture not yet governed", Some("output.weight"), Some("q8_0"), "dense_gguf_lm_head_logits_fixture"),
            dense_all_layer_boundary_gap("kv_cache_policy", "KV cache residency and transfer policy not yet recorded", Option::<&str>::None, Option::<&str>::None, "dense_gguf_kv_cache_policy_receipt"),
            dense_all_layer_boundary_gap("sampling", "sampler integration and logits transfer policy not yet governed", Option::<&str>::None, Option::<&str>::None, "dense_gguf_sampling_policy_receipt")
        ],
        "all_boundary_gaps_explicit": true,
        "qwen_one_token_cuda_blocked": true,
        "qwen_short_decode_cuda_blocked": true,
        "qwen_chat_cuda_blocked": true,
        "next_required_proof": "dense_gguf_model_boundary_fixtures",
        "dense_gguf_inference_claimed": false,
        "speedup_claim": false,
        "full_cuda_residency_claimed": false
    })
}

fn dense_all_layer_boundary_gap(
    gap: &str,
    disposition: &str,
    source_tensor: Option<&str>,
    source_tensor_type: Option<&str>,
    next_proof: &str,
) -> Value {
    json!({
        "gap": gap,
        "status": "not_governed_by_all_layer_block_plan",
        "disposition": disposition,
        "source_tensor": source_tensor,
        "source_tensor_type": source_tensor_type,
        "blocks_qwen_one_token": true,
        "blocks_qwen_short_decode": true,
        "blocks_qwen_chat": true,
        "required_next_proof": next_proof
    })
}

fn dense_all_layer_operations(layer_index: u64) -> Value {
    let mut operations = dense_one_layer_operations();
    for op in operations.as_array_mut().expect("operations array") {
        if let Some(name) = op.get("name").and_then(Value::as_str) {
            let replaced = name.replace("blk.0", &format!("blk.{layer_index}"));
            op["name"] = json!(replaced);
        }
        if let Some(source_tensor) = op.get("source_tensor").and_then(Value::as_str) {
            let replaced = source_tensor.replace("blk.0", &format!("blk.{layer_index}"));
            op["source_tensor"] = json!(replaced);
        }
    }
    operations
}

fn dense_one_layer_operations() -> Value {
    let mut operations = Vec::new();
    push_one_layer_cuda_rmsnorm_op(&mut operations, 0, "blk.0.attn_norm.weight", "attention_norm");
    push_one_layer_cuda_op(&mut operations, 1, "blk.0.attn_q.weight", "attention_q");
    push_one_layer_cuda_op(&mut operations, 2, "blk.0.attn_k.weight", "attention_k");
    push_one_layer_cuda_op(&mut operations, 3, "blk.0.attn_v.weight", "attention_v");
    push_one_layer_cuda_rope_op(&mut operations, 4, "blk.0.rope", "rope");
    push_one_layer_cuda_attention_score_op(
        &mut operations,
        5,
        "blk.0.attention_scores",
        "attention_scores",
    );
    push_one_layer_cuda_attention_softmax_op(
        &mut operations,
        6,
        "blk.0.attention_softmax",
        "attention_softmax",
    );
    push_one_layer_cuda_attention_score_op(
        &mut operations,
        7,
        "blk.0.attention_v_mix",
        "attention_v_mix",
    );
    push_one_layer_cuda_op(&mut operations, 8, "blk.0.attn_output.weight", "attention_output");
    push_one_layer_cuda_rmsnorm_op(&mut operations, 9, "blk.0.ffn_norm.weight", "ffn_norm");
    push_one_layer_cuda_op(&mut operations, 10, "blk.0.ffn_gate.weight", "mlp_gate");
    push_one_layer_cuda_op(&mut operations, 11, "blk.0.ffn_up.weight", "mlp_up");
    push_one_layer_cuda_mlp_activation_op(
        &mut operations,
        12,
        "blk.0.mlp_activation",
        "mlp_activation",
    );
    push_one_layer_cuda_op(&mut operations, 13, "blk.0.ffn_down.weight", "mlp_down");
    Value::Array(operations)
}

fn push_one_layer_cuda_mlp_activation_op(
    operations: &mut Vec<Value>,
    index: u64,
    name: &str,
    role: &str,
) {
    operations.push(json!({
        "index": index,
        "name": name,
        "role": role,
        "op_type": "activation",
        "size": 16,
        "source": "derived_transformer_op",
        "source_tensor": Value::Null,
        "source_tensor_type": Value::Null,
        "source_shape": Value::Null,
        "is_quantized": false,
        "route": "dense_regular_llm_cuda",
        "status": "cuda_routable",
        "fallback_used": false,
        "reason": format!("cuda_dense_regular_llm route selected for {name}")
    }));
}

fn push_one_layer_cuda_op(operations: &mut Vec<Value>, index: u64, name: &str, role: &str) {
    operations.push(json!({
        "index": index,
        "name": name,
        "role": role,
        "op_type": "matmul",
        "size": 256,
        "source": "gguf_tensor_descriptor",
        "source_tensor": name,
        "source_tensor_type": "q8_0",
        "source_shape": [16, 16],
        "is_quantized": false,
        "route": "dense_regular_llm_cuda",
        "status": "cuda_routable",
        "fallback_used": false,
        "reason": format!("cuda_dense_regular_llm route selected for {name}")
    }));
}

fn push_one_layer_cuda_rmsnorm_op(operations: &mut Vec<Value>, index: u64, name: &str, role: &str) {
    operations.push(json!({
        "index": index,
        "name": name,
        "role": role,
        "op_type": "rmsnorm",
        "size": 16,
        "source": "gguf_tensor_descriptor",
        "source_tensor": name,
        "source_tensor_type": "f32",
        "source_shape": [16],
        "is_quantized": false,
        "route": "dense_regular_llm_cuda",
        "status": "cuda_routable",
        "fallback_used": false,
        "reason": format!("cuda_dense_regular_llm route selected for {name}")
    }));
}

fn push_one_layer_cuda_rope_op(operations: &mut Vec<Value>, index: u64, name: &str, role: &str) {
    operations.push(json!({
        "index": index,
        "name": name,
        "role": role,
        "op_type": "rope",
        "size": 16,
        "source": "derived_transformer_op",
        "source_tensor": Value::Null,
        "source_tensor_type": Value::Null,
        "source_shape": Value::Null,
        "is_quantized": false,
        "route": "dense_regular_llm_cuda",
        "status": "cuda_routable",
        "fallback_used": false,
        "reason": format!("cuda_dense_regular_llm route selected for {name}")
    }));
}

fn push_one_layer_cuda_attention_score_op(
    operations: &mut Vec<Value>,
    index: u64,
    name: &str,
    role: &str,
) {
    operations.push(json!({
        "index": index,
        "name": name,
        "role": role,
        "op_type": "attention",
        "size": 16,
        "source": "derived_transformer_op",
        "source_tensor": Value::Null,
        "source_tensor_type": Value::Null,
        "source_shape": Value::Null,
        "is_quantized": false,
        "route": "dense_regular_llm_cuda",
        "status": "cuda_routable",
        "fallback_used": false,
        "reason": format!("cuda_dense_regular_llm route selected for {name}")
    }));
}

fn push_one_layer_cuda_attention_softmax_op(
    operations: &mut Vec<Value>,
    index: u64,
    name: &str,
    role: &str,
) {
    operations.push(json!({
        "index": index,
        "name": name,
        "role": role,
        "op_type": "softmax",
        "size": 16,
        "source": "derived_transformer_op",
        "source_tensor": Value::Null,
        "source_tensor_type": Value::Null,
        "source_shape": Value::Null,
        "is_quantized": false,
        "route": "dense_regular_llm_cuda",
        "status": "cuda_routable",
        "fallback_used": false,
        "reason": format!("cuda_dense_regular_llm route selected for {name}")
    }));
}

fn dense_gguf_descriptor_entries() -> Value {
    json!([
        dense_descriptor(
            "token_embd.weight",
            "token_embedding",
            json!([16, 16]),
            "q8_0",
            0,
            272,
            true,
            "dense_quant_descriptor_only"
        ),
        dense_descriptor(
            "output.weight",
            "output",
            json!([16, 16]),
            "q8_0",
            272,
            272,
            true,
            "dense_quant_descriptor_only"
        ),
        dense_descriptor(
            "blk.0.attn_q.weight",
            "attention_q",
            json!([16, 16]),
            "q8_0",
            544,
            272,
            true,
            "dense_quant_descriptor_only"
        ),
        dense_descriptor(
            "blk.0.attn_k.weight",
            "attention_k",
            json!([16, 16]),
            "q8_0",
            816,
            272,
            true,
            "dense_quant_descriptor_only"
        ),
        dense_descriptor(
            "blk.0.attn_v.weight",
            "attention_v",
            json!([16, 16]),
            "q8_0",
            1088,
            272,
            true,
            "dense_quant_descriptor_only"
        ),
        dense_descriptor(
            "blk.0.attn_output.weight",
            "attention_output",
            json!([16, 16]),
            "q8_0",
            1360,
            272,
            true,
            "dense_quant_descriptor_only"
        ),
        dense_descriptor(
            "blk.0.ffn_gate.weight",
            "mlp_gate",
            json!([16, 16]),
            "q8_0",
            1632,
            272,
            true,
            "dense_quant_descriptor_only"
        ),
        dense_descriptor(
            "blk.0.ffn_up.weight",
            "mlp_up",
            json!([16, 16]),
            "q8_0",
            1904,
            272,
            true,
            "dense_quant_descriptor_only"
        ),
        dense_descriptor(
            "blk.0.ffn_down.weight",
            "mlp_down",
            json!([16, 16]),
            "q8_0",
            2176,
            272,
            true,
            "dense_quant_descriptor_only"
        ),
        dense_descriptor(
            "blk.0.attn_norm.weight",
            "attention_norm",
            json!([16]),
            "f32",
            2448,
            64,
            false,
            "norm_or_metadata_descriptor_only"
        ),
        dense_descriptor(
            "blk.0.ffn_norm.weight",
            "ffn_norm",
            json!([16]),
            "f32",
            2512,
            64,
            false,
            "norm_or_metadata_descriptor_only"
        )
    ])
}

fn dense_descriptor(
    name: &str,
    role: &str,
    shape: Value,
    tensor_type: &str,
    offset: u64,
    size_bytes: u64,
    quantized: bool,
    descriptor_status: &str,
) -> Value {
    json!({
        "name": name,
        "role": role,
        "shape": shape,
        "tensor_type": tensor_type,
        "offset": offset,
        "size_bytes": size_bytes,
        "quantized": quantized,
        "descriptor_status": descriptor_status
    })
}

fn cuda_identity() -> Value {
    json!({
        "available": true,
        "device_count": 1,
        "device_index": 0,
        "device_name": "NVIDIA GeForce RTX 5070 Ti",
        "compute_capability": "12.0",
        "driver_version": "591.86",
        "cuda_runtime_version": "12.9",
        "cuda_toolkit_version": "12.9",
        "nvrtc_version": "12.9",
        "nvml_available": true,
        "vram_bytes": 17094475776_u64,
        "power_limit_watts": 300.0,
        "power_draw_watts": 34.97,
        "temperature_c": 38.0
    })
}

fn kernel_stats() -> Value {
    json!({
        "kernel_id": "cuda_tiny_vector_add",
        "invocations": 1,
        "fallback_invocations": 0,
        "host_to_device_bytes": 8192,
        "device_to_host_bytes": 4096,
        "kernel_launches": 1,
        "kernel_time_ms": null
    })
}
