use anyhow::Result;
use std::path::PathBuf;

fn build_openvino_gpu_smoke_receipt(
    smoke: bitnet_device_probe::runtimes::OpenVinoGpuTinyGraphSmoke,
    strict: bool,
    timestamp_utc: String,
    artifact_path: Option<String>,
) -> serde_json::Value {
    let backend_runtime = serde_json::json!({
        "name": "openvino",
        "version": smoke.openvino_version.clone(),
        "device": smoke.runtime_device.clone(),
        "device_name": smoke.openvino_gpu_full_name.clone(),
    });
    let shape_contract = serde_json::json!({
        "shape_mode": smoke.shape_mode.clone(),
        "input_shape": smoke.input_shape.clone(),
        "output_shape": smoke.output_shape.clone(),
    });
    let fallback_policy = serde_json::json!({
        "fallback_used": smoke.fallback_used,
        "fallback_backend": null,
        "fallback_reason": null,
        "cpu_fallback_allowed": smoke.cpu_fallback_allowed,
    });
    let error = if strict && !smoke.passed {
        Some(smoke.error.clone().unwrap_or_else(|| {
            "strict Arc 140V OpenVINO GPU smoke requires tiny graph pass".to_owned()
        }))
    } else {
        smoke.error.clone()
    };
    let claim = if smoke.passed {
        "openvino_gpu_arc140v_tiny_graph_smoke_passed"
    } else if smoke.graph_execution {
        "openvino_gpu_tiny_graph_executed_without_arc140v_identity"
    } else {
        "openvino_gpu_arc140v_smoke_not_proven"
    };

    serde_json::json!({
        "schema": 1,
        "artifact_kind": "intel_arc_140v_openvino_gpu_smoke",
        "machine_id": "intel-258v",
        "hardware_lane": "intel-arc-140v-openvino-gpu",
        "proof_stage": smoke.proof_stage.clone(),
        "timestamp_utc": timestamp_utc,
        "requested_backend": smoke.requested_backend.clone(),
        "selected_backend": smoke.selected_backend.clone(),
        "runtime_api": smoke.runtime_api.clone(),
        "runtime_device": smoke.runtime_device.clone(),
        "backend_runtime": backend_runtime,
        "shape_contract": shape_contract,
        "shape_mode": smoke.shape_mode.clone(),
        "strict_mode": strict,
        "fallback_used": smoke.fallback_used,
        "fallback_backend": null,
        "fallback_reason": null,
        "fallback_policy": fallback_policy,
        "cpu_fallback_allowed": smoke.cpu_fallback_allowed,
        "kernel_execution": false,
        "graph_execution": smoke.graph_execution,
        "bitnet_inference": smoke.bitnet_inference,
        "qk256_decode": smoke.qk256_decode,
        "arc140v_identity_matched": smoke.arc140v_identity_matched,
        "graph": {
            "name": smoke.graph_name.clone(),
            "precision": smoke.precision.clone(),
            "cache_dir": null,
            "input_shape": smoke.input_shape.clone(),
            "output_shape": smoke.output_shape.clone(),
            "max_abs_error": smoke.max_abs_error,
            "mean_abs_error": smoke.mean_abs_error,
            "tolerance": smoke.tolerance,
            "result": if smoke.passed { "pass" } else { "fail" },
        },
        "timing": {
            "first_ever_compile_and_infer_ms": null,
            "cached_compile_ms": smoke.compile_ms,
            "steady_state_infer_ms": null,
            "compile_ms": smoke.compile_ms,
            "first_infer_ms": smoke.first_infer_ms,
        },
        "kernels_or_graphs": [
            "tiny_matmul_openvino_gpu"
        ],
        "openvino_gpu_smoke": smoke,
        "claim": claim,
        "must_not_claim": [
            "Native OpenCL kernels work on Arc 140V",
            "BitNet inference works on Arc 140V",
            "Arc 140V accelerates BitNet",
            "Packed BitNet QK256 decode works on Arc 140V",
            "CPU fallback satisfies Arc 140V proof"
        ],
        "artifact_path": artifact_path,
        "error": error,
    })
}

fn build_opencl_smoke_receipt(
    smoke: bitnet_device_probe::runtimes::OpenClTinyKernelSmoke,
    strict: bool,
    timestamp_utc: String,
    artifact_path: Option<String>,
) -> serde_json::Value {
    let backend_runtime = serde_json::json!({
        "name": "opencl",
        "device": smoke.runtime_device.clone(),
        "platform": smoke.platform_name.clone(),
        "vendor": smoke.vendor.clone(),
        "driver_version": smoke.driver_version.clone(),
    });
    let fallback_policy = serde_json::json!({
        "fallback_used": smoke.fallback_used,
        "fallback_backend": null,
        "fallback_reason": null,
        "cpu_fallback_allowed": smoke.cpu_fallback_allowed,
    });
    let error = if strict && !smoke.passed {
        Some(smoke.error.clone().unwrap_or_else(|| {
            "strict Arc 140V OpenCL smoke requires tiny kernel execution".to_owned()
        }))
    } else {
        smoke.error.clone()
    };
    let claim = if smoke.passed {
        "opencl_arc140v_tiny_kernel_smoke_passed"
    } else {
        "opencl_arc140v_smoke_not_proven"
    };

    serde_json::json!({
        "schema": 1,
        "artifact_kind": "intel_arc_140v_opencl_smoke",
        "machine_id": "intel-258v",
        "hardware_lane": "intel-arc-140v-opencl",
        "proof_stage": smoke.proof_stage.clone(),
        "timestamp_utc": timestamp_utc,
        "requested_backend": smoke.requested_backend.clone(),
        "selected_backend": smoke.selected_backend.clone(),
        "runtime_api": smoke.runtime_api.clone(),
        "runtime_device": smoke.runtime_device.clone(),
        "backend_runtime": backend_runtime,
        "strict_mode": strict,
        "fallback_used": smoke.fallback_used,
        "fallback_backend": null,
        "fallback_reason": null,
        "fallback_policy": fallback_policy,
        "cpu_fallback_allowed": smoke.cpu_fallback_allowed,
        "kernel_execution": smoke.kernel_execution,
        "graph_execution": false,
        "bitnet_inference": smoke.bitnet_inference,
        "qk256_decode": smoke.qk256_decode,
        "kernel": {
            "name": smoke.kernel_name.clone(),
            "operation": "vector_add",
            "precision": "F32",
            "input_len": smoke.input_len,
            "max_abs_error": smoke.max_abs_error,
            "mean_abs_error": smoke.mean_abs_error,
            "tolerance": smoke.tolerance,
            "result": if smoke.passed { "pass" } else { "fail" },
        },
        "timing": {
            "enqueue_ms": smoke.enqueue_ms,
            "readback_ms": smoke.readback_ms,
        },
        "kernels_or_graphs": [
            "tiny_vector_add_opencl"
        ],
        "opencl_smoke": smoke,
        "claim": claim,
        "must_not_claim": [
            "BitNet inference works on Arc 140V",
            "Arc 140V accelerates BitNet",
            "Packed BitNet QK256 decode works on Arc 140V",
            "OpenVINO GPU smoke proves native OpenCL kernels",
            "CPU fallback satisfies Arc 140V proof"
        ],
        "artifact_path": artifact_path,
        "error": error,
    })
}

fn build_opencl_parity_receipt(
    smoke: bitnet_device_probe::runtimes::OpenClTinyKernelSmoke,
    strict: bool,
    timestamp_utc: String,
    artifact_path: Option<String>,
    cpu_reference_artifact: Option<String>,
) -> serde_json::Value {
    let backend_runtime = serde_json::json!({
        "name": "opencl",
        "device": smoke.runtime_device.clone(),
        "platform": smoke.platform_name.clone(),
        "vendor": smoke.vendor.clone(),
        "driver_version": smoke.driver_version.clone(),
    });
    let fallback_policy = serde_json::json!({
        "fallback_used": smoke.fallback_used,
        "fallback_backend": null,
        "fallback_reason": null,
        "cpu_fallback_allowed": smoke.cpu_fallback_allowed,
    });
    let proof_stage = if smoke.passed { "parity_tested" } else { smoke.proof_stage.as_str() };
    let error = if strict && !smoke.passed {
        Some(smoke.error.clone().unwrap_or_else(|| {
            "strict Arc 140V OpenCL CPU parity requires native kernel pass".to_owned()
        }))
    } else {
        smoke.error.clone()
    };
    let claim = if smoke.passed {
        "opencl_arc140v_cpu_parity_passed"
    } else {
        "opencl_arc140v_cpu_parity_not_proven"
    };

    serde_json::json!({
        "schema": 1,
        "artifact_kind": "intel_arc_140v_opencl_cpu_parity",
        "machine_id": "intel-258v",
        "hardware_lane": "intel-arc-140v-opencl",
        "proof_stage": proof_stage,
        "timestamp_utc": timestamp_utc,
        "requested_backend": smoke.requested_backend.clone(),
        "selected_backend": smoke.selected_backend.clone(),
        "runtime_api": smoke.runtime_api.clone(),
        "runtime_device": smoke.runtime_device.clone(),
        "backend_runtime": backend_runtime,
        "strict_mode": strict,
        "fallback_used": smoke.fallback_used,
        "fallback_backend": null,
        "fallback_reason": null,
        "fallback_policy": fallback_policy,
        "cpu_fallback_allowed": smoke.cpu_fallback_allowed,
        "kernel_execution": smoke.kernel_execution,
        "graph_execution": false,
        "bitnet_inference": smoke.bitnet_inference,
        "qk256_decode": smoke.qk256_decode,
        "cpu_reference": {
            "artifact_path": cpu_reference_artifact,
            "reference_path": "cpu_vector_add_f32",
            "comparison": "opencl_arc140v_output_vs_cpu_reference",
        },
        "kernel": {
            "name": smoke.kernel_name.clone(),
            "operation": "vector_add",
            "precision": "F32",
            "input_len": smoke.input_len,
            "max_abs_error": smoke.max_abs_error,
            "mean_abs_error": smoke.mean_abs_error,
            "tolerance": smoke.tolerance,
            "result": if smoke.passed { "pass" } else { "fail" },
        },
        "timing": {
            "enqueue_ms": smoke.enqueue_ms,
            "readback_ms": smoke.readback_ms,
        },
        "kernels_or_graphs": [
            "tiny_vector_add_opencl_cpu_parity"
        ],
        "opencl_parity": smoke,
        "claim": claim,
        "must_not_claim": [
            "BitNet inference works on Arc 140V",
            "Arc 140V accelerates BitNet",
            "Packed BitNet QK256 decode works on Arc 140V",
            "OpenVINO GPU smoke proves native OpenCL parity",
            "CPU fallback satisfies Arc 140V proof"
        ],
        "artifact_path": artifact_path,
        "error": error,
    })
}

pub(crate) async fn handle_openvino_gpu_smoke_command(
    strict: bool,
    json_out: Option<PathBuf>,
) -> Result<()> {
    let smoke = bitnet_device_probe::runtimes::run_openvino_gpu_tiny_graph_smoke();
    let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let artifact_path = json_out.as_ref().map(|path| path.display().to_string());
    let receipt = build_openvino_gpu_smoke_receipt(smoke, strict, timestamp_utc, artifact_path);

    crate::write_json_output(json_out.as_ref(), &receipt)?;

    if strict && let Some(error) = receipt.get("error").and_then(serde_json::Value::as_str) {
        anyhow::bail!("{error}");
    }

    Ok(())
}

pub(crate) async fn handle_opencl_smoke_command(
    strict: bool,
    json_out: Option<PathBuf>,
) -> Result<()> {
    let smoke = bitnet_device_probe::runtimes::opencl::run_arc140v_opencl_tiny_kernel_smoke();
    let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let artifact_path = json_out.as_ref().map(|path| path.display().to_string());
    let receipt = build_opencl_smoke_receipt(smoke, strict, timestamp_utc, artifact_path);

    crate::write_json_output(json_out.as_ref(), &receipt)?;

    if strict && let Some(error) = receipt.get("error").and_then(serde_json::Value::as_str) {
        anyhow::bail!("{error}");
    }

    Ok(())
}

pub(crate) async fn handle_opencl_parity_command(
    strict: bool,
    cpu_reference: Option<PathBuf>,
    json_out: Option<PathBuf>,
) -> Result<()> {
    let smoke = bitnet_device_probe::runtimes::opencl::run_arc140v_opencl_tiny_kernel_smoke();
    let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let artifact_path = json_out.as_ref().map(|path| path.display().to_string());
    let cpu_reference_artifact = cpu_reference.as_ref().map(|path| path.display().to_string());
    let receipt = build_opencl_parity_receipt(
        smoke,
        strict,
        timestamp_utc,
        artifact_path,
        cpu_reference_artifact,
    );

    crate::write_json_output(json_out.as_ref(), &receipt)?;

    if strict && let Some(error) = receipt.get("error").and_then(serde_json::Value::as_str) {
        anyhow::bail!("{error}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        build_opencl_parity_receipt, build_opencl_smoke_receipt, build_openvino_gpu_smoke_receipt,
    };

    #[test]
    fn intel_arc_openvino_gpu_smoke_receipt_records_graph_execution_only() {
        let smoke = bitnet_device_probe::runtimes::OpenVinoGpuTinyGraphSmoke {
            passed: true,
            proof_stage: "kernel_smoke_tested".to_string(),
            requested_backend: "intel-arc-140v".to_string(),
            selected_backend: Some("intel-arc-140v-openvino-gpu".to_string()),
            runtime_api: Some("openvino".to_string()),
            runtime_device: Some("GPU.0".to_string()),
            openvino_gpu_full_name: Some("Intel(R) Arc(TM) 140V Graphics".to_string()),
            arc140v_identity_matched: true,
            openvino_version: Some("2026.1".to_string()),
            openvino_available_devices: vec!["CPU".to_string(), "GPU.0".to_string()],
            graph_name: "tiny_matmul_add_f16_1x16".to_string(),
            shape_mode: "static".to_string(),
            input_shape: vec![1, 16],
            output_shape: Some(vec![1, 16]),
            precision: "F16".to_string(),
            tolerance: 0.001,
            max_abs_error: Some(0.0),
            mean_abs_error: Some(0.0),
            compile_ms: Some(7.5),
            first_infer_ms: Some(0.75),
            fallback_used: false,
            cpu_fallback_allowed: false,
            graph_execution: true,
            bitnet_inference: false,
            qk256_decode: false,
            error: None,
        };

        let receipt = build_openvino_gpu_smoke_receipt(
            smoke,
            true,
            "2026-05-07T00:00:00Z".to_string(),
            Some("ci/hardware/intel-258v/2026-05-07/arc-140v-openvino-gpu-smoke.json".to_string()),
        );

        assert_eq!(receipt["artifact_kind"], "intel_arc_140v_openvino_gpu_smoke");
        assert_eq!(receipt["proof_stage"], "kernel_smoke_tested");
        assert_eq!(receipt["requested_backend"], "intel-arc-140v");
        assert_eq!(receipt["selected_backend"], "intel-arc-140v-openvino-gpu");
        assert_eq!(receipt["runtime_api"], "openvino");
        assert_eq!(receipt["runtime_device"], "GPU.0");
        assert_eq!(receipt["backend_runtime"]["name"], "openvino");
        assert_eq!(receipt["backend_runtime"]["version"], "2026.1");
        assert_eq!(receipt["backend_runtime"]["device"], "GPU.0");
        assert_eq!(receipt["backend_runtime"]["device_name"], "Intel(R) Arc(TM) 140V Graphics");
        assert_eq!(receipt["shape_mode"], "static");
        assert_eq!(receipt["shape_contract"]["input_shape"], serde_json::json!([1, 16]));
        assert_eq!(receipt["shape_contract"]["output_shape"], serde_json::json!([1, 16]));
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["fallback_policy"]["cpu_fallback_allowed"], false);
        assert_eq!(receipt["graph_execution"], true);
        assert_eq!(receipt["kernel_execution"], false);
        assert_eq!(receipt["bitnet_inference"], false);
        assert_eq!(receipt["qk256_decode"], false);
        assert_eq!(receipt["arc140v_identity_matched"], true);
        assert_eq!(receipt["graph"]["name"], "tiny_matmul_add_f16_1x16");
        assert_eq!(receipt["graph"]["result"], "pass");
        assert_eq!(receipt["timing"]["cached_compile_ms"], 7.5);
        assert_eq!(receipt["timing"]["first_infer_ms"], 0.75);
        assert_eq!(receipt["kernels_or_graphs"], serde_json::json!(["tiny_matmul_openvino_gpu"]));
        assert_eq!(receipt["claim"], "openvino_gpu_arc140v_tiny_graph_smoke_passed");
        assert!(receipt["error"].is_null());
    }

    #[test]
    fn strict_intel_arc_openvino_gpu_smoke_records_error_without_fallback() {
        let smoke = bitnet_device_probe::runtimes::OpenVinoGpuTinyGraphSmoke {
            passed: false,
            proof_stage: "runtime_detected".to_string(),
            requested_backend: "intel-arc-140v".to_string(),
            selected_backend: None,
            runtime_api: Some("openvino".to_string()),
            runtime_device: Some("GPU.0".to_string()),
            openvino_gpu_full_name: Some("Intel(R) UHD Graphics".to_string()),
            arc140v_identity_matched: false,
            openvino_version: Some("2026.1".to_string()),
            openvino_available_devices: vec!["CPU".to_string(), "GPU.0".to_string()],
            graph_name: "tiny_matmul_add_f16_1x16".to_string(),
            shape_mode: "static".to_string(),
            input_shape: vec![1, 16],
            output_shape: Some(vec![1, 16]),
            precision: "F16".to_string(),
            tolerance: 0.001,
            max_abs_error: Some(0.0),
            mean_abs_error: Some(0.0),
            compile_ms: Some(7.5),
            first_infer_ms: Some(0.75),
            fallback_used: false,
            cpu_fallback_allowed: false,
            graph_execution: true,
            bitnet_inference: false,
            qk256_decode: false,
            error: Some("OpenVINO GPU full name did not identify Arc 140V".to_string()),
        };

        let receipt =
            build_openvino_gpu_smoke_receipt(smoke, true, "2026-05-07T00:00:00Z".to_string(), None);

        assert_eq!(receipt["artifact_kind"], "intel_arc_140v_openvino_gpu_smoke");
        assert_eq!(receipt["proof_stage"], "runtime_detected");
        assert!(receipt["selected_backend"].is_null());
        assert_eq!(receipt["runtime_api"], "openvino");
        assert_eq!(receipt["runtime_device"], "GPU.0");
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["fallback_policy"]["fallback_used"], false);
        assert_eq!(receipt["fallback_policy"]["cpu_fallback_allowed"], false);
        assert_eq!(receipt["graph_execution"], true);
        assert_eq!(receipt["kernel_execution"], false);
        assert_eq!(receipt["bitnet_inference"], false);
        assert_eq!(receipt["qk256_decode"], false);
        assert_eq!(receipt["arc140v_identity_matched"], false);
        assert_eq!(receipt["claim"], "openvino_gpu_tiny_graph_executed_without_arc140v_identity");
        assert_eq!(receipt["error"], "OpenVINO GPU full name did not identify Arc 140V");
    }

    #[test]
    fn intel_arc_opencl_smoke_receipt_records_kernel_execution_only() {
        let smoke = bitnet_device_probe::runtimes::OpenClTinyKernelSmoke {
            passed: true,
            proof_stage: "kernel_smoke_tested".to_string(),
            requested_backend: "intel-arc-140v".to_string(),
            selected_backend: Some("intel-arc-140v-opencl".to_string()),
            runtime_api: Some("opencl".to_string()),
            runtime_device: Some("Intel(R) Arc(TM) 140V Graphics".to_string()),
            platform_index: Some(0),
            device_index: Some(0),
            platform_name: Some("Intel(R) OpenCL Graphics".to_string()),
            vendor: Some("Intel(R) Corporation".to_string()),
            driver_version: Some("test-driver".to_string()),
            kernel_name: "tiny_vector_add".to_string(),
            input_len: 16,
            tolerance: 0.000001,
            max_abs_error: Some(0.0),
            mean_abs_error: Some(0.0),
            enqueue_ms: Some(0.25),
            readback_ms: Some(0.05),
            kernel_execution: true,
            fallback_used: false,
            cpu_fallback_allowed: false,
            bitnet_inference: false,
            qk256_decode: false,
            error: None,
        };

        let receipt = build_opencl_smoke_receipt(
            smoke,
            true,
            "2026-05-07T00:00:00Z".to_string(),
            Some("ci/hardware/intel-258v/2026-05-07/arc-140v-opencl-smoke.json".to_string()),
        );

        assert_eq!(receipt["artifact_kind"], "intel_arc_140v_opencl_smoke");
        assert_eq!(receipt["proof_stage"], "kernel_smoke_tested");
        assert_eq!(receipt["requested_backend"], "intel-arc-140v");
        assert_eq!(receipt["selected_backend"], "intel-arc-140v-opencl");
        assert_eq!(receipt["runtime_api"], "opencl");
        assert_eq!(receipt["runtime_device"], "Intel(R) Arc(TM) 140V Graphics");
        assert_eq!(receipt["backend_runtime"]["name"], "opencl");
        assert_eq!(receipt["backend_runtime"]["platform"], "Intel(R) OpenCL Graphics");
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["fallback_policy"]["cpu_fallback_allowed"], false);
        assert_eq!(receipt["kernel_execution"], true);
        assert_eq!(receipt["graph_execution"], false);
        assert_eq!(receipt["bitnet_inference"], false);
        assert_eq!(receipt["qk256_decode"], false);
        assert_eq!(receipt["kernel"]["name"], "tiny_vector_add");
        assert_eq!(receipt["kernel"]["operation"], "vector_add");
        assert_eq!(receipt["kernel"]["result"], "pass");
        assert_eq!(receipt["timing"]["enqueue_ms"], 0.25);
        assert_eq!(receipt["timing"]["readback_ms"], 0.05);
        assert_eq!(receipt["kernels_or_graphs"], serde_json::json!(["tiny_vector_add_opencl"]));
        assert_eq!(receipt["claim"], "opencl_arc140v_tiny_kernel_smoke_passed");
        assert!(receipt["error"].is_null());
    }

    #[test]
    fn intel_arc_opencl_parity_receipt_records_cpu_reference_only() {
        let smoke = bitnet_device_probe::runtimes::OpenClTinyKernelSmoke {
            passed: true,
            proof_stage: "kernel_smoke_tested".to_string(),
            requested_backend: "intel-arc-140v".to_string(),
            selected_backend: Some("intel-arc-140v-opencl".to_string()),
            runtime_api: Some("opencl".to_string()),
            runtime_device: Some("Intel(R) Arc(TM) 140V Graphics".to_string()),
            platform_index: Some(0),
            device_index: Some(0),
            platform_name: Some("Intel(R) OpenCL Graphics".to_string()),
            vendor: Some("Intel(R) Corporation".to_string()),
            driver_version: Some("test-driver".to_string()),
            kernel_name: "tiny_vector_add".to_string(),
            input_len: 16,
            tolerance: 0.000001,
            max_abs_error: Some(0.0),
            mean_abs_error: Some(0.0),
            enqueue_ms: Some(0.25),
            readback_ms: Some(0.05),
            kernel_execution: true,
            fallback_used: false,
            cpu_fallback_allowed: false,
            bitnet_inference: false,
            qk256_decode: false,
            error: None,
        };

        let receipt = build_opencl_parity_receipt(
            smoke,
            true,
            "2026-05-08T00:00:00Z".to_string(),
            Some("ci/hardware/intel-258v/2026-05-08/arc-140v-opencl-parity.json".to_string()),
            Some(
                "ci/hardware/intel-258v/2026-05-08/cpu-reference-bundle-post-mechanics.json"
                    .to_string(),
            ),
        );

        assert_eq!(receipt["artifact_kind"], "intel_arc_140v_opencl_cpu_parity");
        assert_eq!(receipt["proof_stage"], "parity_tested");
        assert_eq!(receipt["requested_backend"], "intel-arc-140v");
        assert_eq!(receipt["selected_backend"], "intel-arc-140v-opencl");
        assert_eq!(receipt["runtime_api"], "opencl");
        assert_eq!(receipt["runtime_device"], "Intel(R) Arc(TM) 140V Graphics");
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["fallback_policy"]["cpu_fallback_allowed"], false);
        assert_eq!(receipt["kernel_execution"], true);
        assert_eq!(receipt["graph_execution"], false);
        assert_eq!(receipt["bitnet_inference"], false);
        assert_eq!(receipt["qk256_decode"], false);
        assert_eq!(receipt["kernel"]["name"], "tiny_vector_add");
        assert_eq!(receipt["kernel"]["operation"], "vector_add");
        assert_eq!(receipt["kernel"]["result"], "pass");
        assert_eq!(
            receipt["cpu_reference"]["artifact_path"],
            "ci/hardware/intel-258v/2026-05-08/cpu-reference-bundle-post-mechanics.json"
        );
        assert_eq!(
            receipt["kernels_or_graphs"],
            serde_json::json!(["tiny_vector_add_opencl_cpu_parity"])
        );
        assert_eq!(receipt["claim"], "opencl_arc140v_cpu_parity_passed");
        assert!(receipt["error"].is_null());
    }

    #[test]
    fn strict_intel_arc_opencl_smoke_records_error_without_fallback() {
        let smoke = bitnet_device_probe::runtimes::OpenClTinyKernelSmoke {
            passed: false,
            proof_stage: "runtime_detected".to_string(),
            requested_backend: "intel-arc-140v".to_string(),
            selected_backend: None,
            runtime_api: Some("opencl".to_string()),
            runtime_device: None,
            platform_index: None,
            device_index: None,
            platform_name: None,
            vendor: None,
            driver_version: None,
            kernel_name: "tiny_vector_add".to_string(),
            input_len: 16,
            tolerance: 0.000001,
            max_abs_error: None,
            mean_abs_error: None,
            enqueue_ms: None,
            readback_ms: None,
            kernel_execution: false,
            fallback_used: false,
            cpu_fallback_allowed: false,
            bitnet_inference: false,
            qk256_decode: false,
            error: Some("compiled without opencl feature".to_string()),
        };

        let receipt =
            build_opencl_smoke_receipt(smoke, true, "2026-05-07T00:00:00Z".to_string(), None);

        assert_eq!(receipt["artifact_kind"], "intel_arc_140v_opencl_smoke");
        assert_eq!(receipt["proof_stage"], "runtime_detected");
        assert!(receipt["selected_backend"].is_null());
        assert_eq!(receipt["runtime_api"], "opencl");
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["fallback_policy"]["fallback_used"], false);
        assert_eq!(receipt["fallback_policy"]["cpu_fallback_allowed"], false);
        assert_eq!(receipt["kernel_execution"], false);
        assert_eq!(receipt["graph_execution"], false);
        assert_eq!(receipt["bitnet_inference"], false);
        assert_eq!(receipt["qk256_decode"], false);
        assert_eq!(receipt["claim"], "opencl_arc140v_smoke_not_proven");
        assert_eq!(receipt["error"], "compiled without opencl feature");
    }
}
