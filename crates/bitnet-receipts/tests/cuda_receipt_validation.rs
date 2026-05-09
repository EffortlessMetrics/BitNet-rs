//! RTX 5070 Ti CUDA receipt validation tests.
//!
//! These tests validate strict smoke/parity proof artifacts without claiming
//! benchmark or full BitNet inference readiness.
#![recursion_limit = "256"]

use bitnet_receipts::{
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof, validate_cuda_parity_receipt_json,
    validate_cuda_smoke_receipt_json, validate_dense_gguf_attention_score_cuda_parity_receipt_json,
    validate_dense_gguf_attention_score_fixture_receipt_json,
    validate_dense_gguf_attention_softmax_cuda_parity_receipt_json,
    validate_dense_gguf_attention_softmax_fixture_receipt_json,
    validate_dense_gguf_attention_v_mix_cuda_parity_receipt_json,
    validate_dense_gguf_attention_v_mix_fixture_receipt_json,
    validate_dense_gguf_linear_cuda_parity_receipt_json,
    validate_dense_gguf_linear_fixture_extraction_receipt_json,
    validate_dense_gguf_linear_role_sweep_cuda_parity_receipt_json,
    validate_dense_gguf_mlp_activation_cuda_parity_receipt_json,
    validate_dense_gguf_mlp_activation_fixture_receipt_json,
    validate_dense_gguf_norm_cuda_parity_receipt_json,
    validate_dense_gguf_norm_fixture_extraction_receipt_json,
    validate_dense_gguf_one_layer_cpu_reference_receipt_json,
    validate_dense_gguf_one_layer_cuda_integrated_parity_receipt_json,
    validate_dense_gguf_one_layer_execution_plan_receipt_json,
    validate_dense_gguf_rope_cuda_parity_receipt_json,
    validate_dense_gguf_tensor_descriptor_inspection_receipt_json,
    validate_dense_regular_llm_cuda_persistent_residency_receipt_json,
    validate_dense_regular_llm_cuda_receipt_json,
    validate_dense_regular_llm_cuda_tensor_residency_receipt_json,
};
use serde_json::{Value, json};

#[test]
fn committed_cuda_smoke_receipt_validates() {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-smoke.json"
    ))
    .unwrap();

    validate_cuda_smoke_receipt_json(&receipt).unwrap();
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
