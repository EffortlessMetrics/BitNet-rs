//! RTX 5070 Ti CUDA receipt validation tests.
//!
//! These tests validate strict smoke/parity proof artifacts without claiming
//! benchmark or full BitNet inference readiness.

use bitnet_receipts::{
    reject_dense_regular_llm_as_bitnet_packed_cuda_proof, validate_cuda_parity_receipt_json,
    validate_cuda_smoke_receipt_json, validate_dense_regular_llm_cuda_receipt_json,
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
        "kernel_stats": [{
            "kernel_id": "cublaslt_dense_gemm",
            "invocations": 1,
            "fallback_invocations": 0,
            "host_to_device_bytes": 8192,
            "device_to_host_bytes": 4096,
            "kernel_launches": 1,
            "kernel_time_ms": 0.25
        }],
        "claim_boundary": {
            "dense_regular_llm_cuda_claimed": true,
            "bitnet_packed_i2s_qk256_proof": false,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false
        },
        "error": null
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
