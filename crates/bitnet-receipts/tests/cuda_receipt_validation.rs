//! RTX 5070 Ti CUDA receipt validation tests.
//!
//! These tests validate strict smoke/parity proof artifacts without claiming
//! benchmark or full BitNet inference readiness.

use bitnet_receipts::{
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof, validate_cuda_parity_receipt_json,
    validate_cuda_smoke_receipt_json,
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
