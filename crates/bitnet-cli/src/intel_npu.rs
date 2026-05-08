use anyhow::Result;
use std::path::PathBuf;

fn build_probe_receipt(
    probe: bitnet_device_probe::IntelNpuProbe,
    strict: bool,
    timestamp_utc: String,
    artifact_path: Option<String>,
) -> serde_json::Value {
    let backend_runtime = serde_json::json!({
        "name": "openvino",
        "version": probe.openvino_version.clone(),
        "device": probe.runtime_device.clone(),
        "device_name": probe.openvino_npu_full_name.clone(),
        "driver_version": probe.driver_version.clone(),
        "compiler_version": probe.compiler_version.clone(),
        "max_tiles": probe.max_tiles,
    });
    let shape_contract = serde_json::json!({
        "shape_mode": "static",
        "input_shape": null,
        "output_shape": null,
        "note": "runtime_probe_only_no_graph_compiled",
    });
    let fallback_policy = serde_json::json!({
        "fallback_used": probe.fallback_used,
        "fallback_backend": null,
        "fallback_reason": null,
        "cpu_fallback_allowed": false,
    });
    let claim = if probe.openvino_npu_visible {
        "openvino_npu_runtime_visibility_recorded"
    } else if probe.available {
        "intel_npu_os_visibility_recorded"
    } else {
        "intel_npu_unavailable"
    };
    let error =
        if strict && !probe.openvino_npu_visible {
            Some(probe.failure_reason.clone().unwrap_or_else(|| {
                "strict Intel NPU probe requires OpenVINO to report NPU".to_owned()
            }))
        } else {
            None
        };

    serde_json::json!({
        "schema": 1,
        "artifact_kind": "intel_npu_runtime_probe",
        "machine_id": "intel-258v",
        "hardware_lane": "intel-npu-openvino",
        "proof_stage": probe.proof_stage.clone(),
        "timestamp_utc": timestamp_utc,
        "requested_backend": probe.requested_backend.clone(),
        "selected_backend": probe.selected_backend.clone(),
        "runtime_api": probe.runtime_api.clone(),
        "runtime_device": probe.runtime_device.clone(),
        "backend_runtime": backend_runtime,
        "shape_contract": shape_contract,
        "shape_mode": "static",
        "strict_mode": strict,
        "fallback_used": probe.fallback_used,
        "fallback_backend": null,
        "fallback_reason": null,
        "fallback_policy": fallback_policy,
        "kernel_execution": false,
        "graph_execution": false,
        "bitnet_inference": false,
        "kernels_or_graphs": [],
        "npu": probe,
        "claim": claim,
        "must_not_claim": [
            "OpenVINO NPU graph execution works",
            "Intel NPU accelerates BitNet",
            "BitNet inference works on Intel NPU",
            "CPU fallback satisfies NPU proof"
        ],
        "artifact_path": artifact_path,
        "error": error,
    })
}

pub(crate) async fn handle_probe_command(strict: bool, json_out: Option<PathBuf>) -> Result<()> {
    let openvino = bitnet_device_probe::runtimes::openvino::probe_openvino();
    let npu = bitnet_device_probe::intel::lunar_lake::probe_intel_npu(&openvino);
    let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let artifact_path = json_out.as_ref().map(|path| path.display().to_string());
    let receipt = build_probe_receipt(npu, strict, timestamp_utc, artifact_path);

    crate::write_json_output(json_out.as_ref(), &receipt)?;

    if let Some(error) = receipt.get("error").and_then(serde_json::Value::as_str) {
        anyhow::bail!("{error}");
    }

    Ok(())
}

fn build_smoke_receipt(
    probe: bitnet_device_probe::IntelNpuProbe,
    smoke: bitnet_device_probe::runtimes::OpenVinoNpuTinyGraphSmoke,
    strict: bool,
    timestamp_utc: String,
    artifact_path: Option<String>,
) -> serde_json::Value {
    let selected_backend =
        smoke.selected_backend.clone().or_else(|| probe.selected_backend.clone());
    let runtime_api = smoke.runtime_api.clone().or_else(|| probe.runtime_api.clone());
    let runtime_device = smoke.runtime_device.clone().or_else(|| probe.runtime_device.clone());
    let backend_runtime = serde_json::json!({
        "name": "openvino",
        "version": smoke.openvino_version.clone().or_else(|| probe.openvino_version.clone()),
        "device": runtime_device.clone(),
        "device_name": probe.openvino_npu_full_name.clone(),
        "driver_version": probe.driver_version.clone(),
        "compiler_version": probe.compiler_version.clone(),
        "max_tiles": probe.max_tiles,
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
        Some(
            smoke
                .error
                .clone()
                .unwrap_or_else(|| "strict Intel NPU smoke requires tiny graph pass".to_owned()),
        )
    } else {
        smoke.error.clone()
    };
    let claim = if smoke.passed {
        "openvino_npu_tiny_graph_smoke_passed"
    } else if probe.openvino_npu_visible {
        "openvino_npu_tiny_graph_smoke_failed"
    } else if probe.available {
        "intel_npu_runtime_visibility_recorded_without_openvino_graph"
    } else {
        "intel_npu_unavailable"
    };

    serde_json::json!({
        "schema": 1,
        "artifact_kind": "intel_npu_tiny_graph_smoke",
        "machine_id": "intel-258v",
        "hardware_lane": "intel-npu-openvino",
        "proof_stage": smoke.proof_stage.clone(),
        "timestamp_utc": timestamp_utc,
        "requested_backend": probe.requested_backend.clone(),
        "selected_backend": selected_backend,
        "runtime_api": runtime_api,
        "runtime_device": runtime_device,
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
            "tiny_matmul_openvino_npu"
        ],
        "npu": probe,
        "openvino_smoke": smoke,
        "claim": claim,
        "must_not_claim": [
            "BitNet inference works on Intel NPU",
            "Intel NPU accelerates BitNet",
            "Packed BitNet QK256 decode works on Intel NPU",
            "CPU fallback satisfies NPU proof"
        ],
        "artifact_path": artifact_path,
        "error": error,
    })
}

pub(crate) async fn handle_smoke_command(strict: bool, json_out: Option<PathBuf>) -> Result<()> {
    let openvino = bitnet_device_probe::runtimes::openvino::probe_openvino();
    let npu = bitnet_device_probe::intel::lunar_lake::probe_intel_npu(&openvino);
    let smoke = bitnet_device_probe::runtimes::run_openvino_npu_tiny_graph_smoke();
    let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let artifact_path = json_out.as_ref().map(|path| path.display().to_string());
    let receipt = build_smoke_receipt(npu, smoke, strict, timestamp_utc, artifact_path);

    crate::write_json_output(json_out.as_ref(), &receipt)?;

    if strict && let Some(error) = receipt.get("error").and_then(serde_json::Value::as_str) {
        anyhow::bail!("{error}");
    }

    Ok(())
}

fn build_bitnet_subgraph_receipt(
    probe: bitnet_device_probe::IntelNpuProbe,
    parity: bitnet_device_probe::runtimes::OpenVinoNpuBitnetSubgraphParity,
    strict: bool,
    timestamp_utc: String,
    artifact_path: Option<String>,
) -> serde_json::Value {
    build_bitnet_subgraph_receipt_with_cpu_reference(
        probe,
        parity,
        strict,
        timestamp_utc,
        artifact_path,
        None,
    )
}

fn build_bitnet_subgraph_receipt_with_cpu_reference(
    probe: bitnet_device_probe::IntelNpuProbe,
    parity: bitnet_device_probe::runtimes::OpenVinoNpuBitnetSubgraphParity,
    strict: bool,
    timestamp_utc: String,
    artifact_path: Option<String>,
    cpu_reference_artifact: Option<String>,
) -> serde_json::Value {
    let selected_backend =
        parity.selected_backend.clone().or_else(|| probe.selected_backend.clone());
    let runtime_api = parity.runtime_api.clone().or_else(|| probe.runtime_api.clone());
    let runtime_device = parity.runtime_device.clone().or_else(|| probe.runtime_device.clone());
    let backend_runtime = serde_json::json!({
        "name": "openvino",
        "version": parity.openvino_version.clone().or_else(|| probe.openvino_version.clone()),
        "device": runtime_device.clone(),
        "device_name": probe.openvino_npu_full_name.clone(),
        "driver_version": probe.driver_version.clone(),
        "compiler_version": probe.compiler_version.clone(),
        "max_tiles": probe.max_tiles,
    });
    let shape_contract = serde_json::json!({
        "shape_mode": parity.shape_mode.clone(),
        "input_shape": parity.input_shape.clone(),
        "output_shape": parity.output_shape.clone(),
    });
    let fallback_policy = serde_json::json!({
        "fallback_used": parity.fallback_used,
        "fallback_backend": null,
        "fallback_reason": null,
        "cpu_fallback_allowed": parity.cpu_fallback_allowed,
    });
    let error = if strict && !parity.passed {
        Some(parity.error.clone().unwrap_or_else(|| {
            "strict Intel NPU BitNet subgraph parity requires selected subgraph pass".to_owned()
        }))
    } else {
        parity.error.clone()
    };
    let claim = if parity.passed {
        "openvino_npu_bitnet_subgraph_parity_passed"
    } else if probe.openvino_npu_visible {
        "openvino_npu_bitnet_subgraph_parity_failed"
    } else if probe.available {
        "intel_npu_runtime_visibility_recorded_without_bitnet_subgraph"
    } else {
        "intel_npu_unavailable"
    };

    serde_json::json!({
        "schema": 1,
        "artifact_kind": "intel_npu_bitnet_subgraph_parity",
        "machine_id": "intel-258v",
        "hardware_lane": "intel-npu-openvino",
        "proof_stage": parity.proof_stage.clone(),
        "timestamp_utc": timestamp_utc,
        "requested_backend": probe.requested_backend.clone(),
        "selected_backend": selected_backend,
        "runtime_api": runtime_api,
        "runtime_device": runtime_device,
        "backend_runtime": backend_runtime,
        "shape_contract": shape_contract,
        "shape_mode": parity.shape_mode.clone(),
        "strict_mode": strict,
        "fallback_used": parity.fallback_used,
        "fallback_backend": null,
        "fallback_reason": null,
        "fallback_policy": fallback_policy,
        "cpu_fallback_allowed": parity.cpu_fallback_allowed,
        "kernel_execution": false,
        "graph_execution": parity.graph_execution,
        "bitnet_inference": parity.bitnet_inference,
        "qk256_decode": parity.qk256_decode,
        "subgraph": {
            "name": parity.subgraph_name.clone(),
            "bitnet_op": parity.bitnet_op.clone(),
            "precision": parity.precision.clone(),
            "reference_path": parity.reference_path.clone(),
            "shape_mode": parity.shape_mode.clone(),
            "input_shape": parity.input_shape.clone(),
            "output_shape": parity.output_shape.clone(),
            "epsilon": parity.epsilon,
            "max_abs_error": parity.max_abs_error,
            "mean_abs_error": parity.mean_abs_error,
            "tolerance": parity.tolerance,
            "result": if parity.passed { "pass" } else { "fail" },
        },
        "cpu_reference": {
            "artifact_path": cpu_reference_artifact,
            "reference_path": parity.reference_path.clone(),
            "comparison": "openvino_npu_output_vs_cpu_numpy_reference",
        },
        "timing": {
            "first_ever_compile_and_infer_ms": null,
            "cached_compile_ms": parity.compile_ms,
            "steady_state_infer_ms": null,
            "compile_ms": parity.compile_ms,
            "first_infer_ms": parity.first_infer_ms,
        },
        "kernels_or_graphs": [
            format!("bitnet_{}_openvino_npu", parity.bitnet_op.replace('-', "_"))
        ],
        "npu": probe,
        "openvino_subgraph": parity,
        "claim": claim,
        "must_not_claim": [
            "Full BitNet inference works on Intel NPU",
            "Native bitnet-rs NPU inference works",
            "Intel NPU accelerates BitNet",
            "Packed BitNet QK256 decode works on Intel NPU",
            "CPU fallback satisfies NPU proof"
        ],
        "artifact_path": artifact_path,
        "error": error,
    })
}

pub(crate) async fn handle_bitnet_subgraph_command(
    strict: bool,
    json_out: Option<PathBuf>,
) -> Result<()> {
    let openvino = bitnet_device_probe::runtimes::openvino::probe_openvino();
    let npu = bitnet_device_probe::intel::lunar_lake::probe_intel_npu(&openvino);
    let parity = bitnet_device_probe::runtimes::run_openvino_npu_bitnet_subgraph_parity();
    let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let artifact_path = json_out.as_ref().map(|path| path.display().to_string());
    let receipt = build_bitnet_subgraph_receipt(npu, parity, strict, timestamp_utc, artifact_path);

    crate::write_json_output(json_out.as_ref(), &receipt)?;

    if strict && let Some(error) = receipt.get("error").and_then(serde_json::Value::as_str) {
        anyhow::bail!("{error}");
    }

    Ok(())
}

pub(crate) async fn handle_bitnet_linear_subgraph_command(
    strict: bool,
    json_out: Option<PathBuf>,
) -> Result<()> {
    let openvino = bitnet_device_probe::runtimes::openvino::probe_openvino();
    let npu = bitnet_device_probe::intel::lunar_lake::probe_intel_npu(&openvino);
    let parity = bitnet_device_probe::runtimes::run_openvino_npu_bitnet_linear_projection_parity();
    let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let artifact_path = json_out.as_ref().map(|path| path.display().to_string());
    let receipt = build_bitnet_subgraph_receipt(npu, parity, strict, timestamp_utc, artifact_path);

    crate::write_json_output(json_out.as_ref(), &receipt)?;

    if strict && let Some(error) = receipt.get("error").and_then(serde_json::Value::as_str) {
        anyhow::bail!("{error}");
    }

    Ok(())
}

pub(crate) async fn handle_bitnet_ffn_subgraph_command(
    strict: bool,
    cpu_reference: Option<PathBuf>,
    json_out: Option<PathBuf>,
) -> Result<()> {
    let openvino = bitnet_device_probe::runtimes::openvino::probe_openvino();
    let npu = bitnet_device_probe::intel::lunar_lake::probe_intel_npu(&openvino);
    let parity = bitnet_device_probe::runtimes::run_openvino_npu_bitnet_ffn_relu2_parity();
    let timestamp_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let artifact_path = json_out.as_ref().map(|path| path.display().to_string());
    let cpu_reference_artifact = cpu_reference.as_ref().map(|path| path.display().to_string());
    let receipt = build_bitnet_subgraph_receipt_with_cpu_reference(
        npu,
        parity,
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
    use super::*;

    fn visible_npu_probe(
        openvino_available_devices: Vec<String>,
    ) -> bitnet_device_probe::IntelNpuProbe {
        bitnet_device_probe::IntelNpuProbe {
            proof_stage: "runtime_detected".to_string(),
            requested_backend: "intel-npu".to_string(),
            selected_backend: Some("intel-npu-openvino".to_string()),
            runtime_api: Some("openvino".to_string()),
            runtime_device: Some("NPU".to_string()),
            os: "windows".to_string(),
            arch: "x86_64".to_string(),
            available: true,
            accel_device_present: false,
            accel_devices: Vec::new(),
            intel_vpu_driver_seen: true,
            driver_hint: Some("intel_vpu/ivpu evidence".to_string()),
            openvino_runtime_available: true,
            openvino_version: Some("2026.1".to_string()),
            openvino_available_devices,
            openvino_npu_visible: true,
            openvino_npu_full_name: Some("Intel(R) AI Boost".to_string()),
            supported_properties: vec!["FULL_DEVICE_NAME".to_string()],
            driver_version: Some("1.2.3".to_string()),
            compiler_version: Some("4.5.6".to_string()),
            total_mem_size: Some(1024),
            alloc_mem_size: Some(128),
            max_tiles: Some(1),
            fallback_used: false,
            failure_reason: None,
        }
    }

    fn openvino_missing_npu_probe() -> bitnet_device_probe::IntelNpuProbe {
        bitnet_device_probe::IntelNpuProbe {
            proof_stage: "runtime_detected".to_string(),
            requested_backend: "intel-npu".to_string(),
            selected_backend: None,
            runtime_api: None,
            runtime_device: None,
            os: "windows".to_string(),
            arch: "x86_64".to_string(),
            available: true,
            accel_device_present: false,
            accel_devices: Vec::new(),
            intel_vpu_driver_seen: true,
            driver_hint: Some("intel_vpu/ivpu evidence".to_string()),
            openvino_runtime_available: false,
            openvino_version: None,
            openvino_available_devices: Vec::new(),
            openvino_npu_visible: false,
            openvino_npu_full_name: None,
            supported_properties: Vec::new(),
            driver_version: None,
            compiler_version: None,
            total_mem_size: None,
            alloc_mem_size: None,
            max_tiles: None,
            fallback_used: false,
            failure_reason: Some("OpenVINO NPU was not visible".to_string()),
        }
    }

    #[test]
    fn intel_npu_probe_receipt_is_visibility_only() {
        let receipt = build_probe_receipt(
            visible_npu_probe(vec!["CPU".to_string(), "GPU.0".to_string(), "NPU".to_string()]),
            true,
            "2026-05-06T00:00:00Z".to_string(),
            Some("ci/hardware/intel-258v/2026-05-06/npu-openvino-runtime-probe.json".to_string()),
        );

        assert_eq!(receipt["artifact_kind"], "intel_npu_runtime_probe");
        assert_eq!(receipt["requested_backend"], "intel-npu");
        assert_eq!(receipt["selected_backend"], "intel-npu-openvino");
        assert_eq!(receipt["runtime_api"], "openvino");
        assert_eq!(receipt["runtime_device"], "NPU");
        assert_eq!(receipt["backend_runtime"]["name"], "openvino");
        assert_eq!(receipt["backend_runtime"]["version"], "2026.1");
        assert_eq!(receipt["backend_runtime"]["device"], "NPU");
        assert_eq!(receipt["backend_runtime"]["device_name"], "Intel(R) AI Boost");
        assert_eq!(receipt["backend_runtime"]["driver_version"], "1.2.3");
        assert_eq!(receipt["backend_runtime"]["compiler_version"], "4.5.6");
        assert_eq!(receipt["backend_runtime"]["max_tiles"], 1);
        assert_eq!(receipt["shape_contract"]["shape_mode"], "static");
        assert!(receipt["shape_contract"]["input_shape"].is_null());
        assert!(receipt["shape_contract"]["output_shape"].is_null());
        assert_eq!(receipt["strict_mode"], true);
        assert_eq!(receipt["shape_mode"], "static");
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["fallback_policy"]["fallback_used"], false);
        assert_eq!(receipt["fallback_policy"]["cpu_fallback_allowed"], false);
        assert!(receipt["fallback_policy"]["fallback_backend"].is_null());
        assert_eq!(receipt["kernel_execution"], false);
        assert_eq!(receipt["graph_execution"], false);
        assert_eq!(receipt["bitnet_inference"], false);
        assert_eq!(receipt["kernels_or_graphs"], serde_json::json!([]));
        assert!(receipt["error"].is_null());
    }

    #[test]
    fn strict_intel_npu_probe_records_error_without_fallback() {
        let receipt = build_probe_receipt(
            openvino_missing_npu_probe(),
            true,
            "2026-05-06T00:00:00Z".to_string(),
            None,
        );

        assert_eq!(receipt["requested_backend"], "intel-npu");
        assert!(receipt["selected_backend"].is_null());
        assert!(receipt["runtime_api"].is_null());
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["graph_execution"], false);
        assert_eq!(receipt["claim"], "intel_npu_os_visibility_recorded");
        assert_eq!(receipt["error"], "OpenVINO NPU was not visible");
    }

    #[test]
    fn intel_npu_smoke_receipt_records_static_graph_execution_only() {
        let smoke = bitnet_device_probe::runtimes::OpenVinoNpuTinyGraphSmoke {
            passed: true,
            proof_stage: "kernel_smoke_tested".to_string(),
            requested_backend: "intel-npu".to_string(),
            selected_backend: Some("intel-npu-openvino".to_string()),
            runtime_api: Some("openvino".to_string()),
            runtime_device: Some("NPU".to_string()),
            openvino_version: Some("2026.1".to_string()),
            openvino_available_devices: vec!["CPU".to_string(), "NPU".to_string()],
            graph_name: "tiny_matmul_add_f16_1x16".to_string(),
            shape_mode: "static".to_string(),
            input_shape: vec![1, 16],
            output_shape: Some(vec![1, 16]),
            precision: "F16".to_string(),
            tolerance: 0.001,
            max_abs_error: Some(0.0),
            mean_abs_error: Some(0.0),
            compile_ms: Some(12.5),
            first_infer_ms: Some(1.25),
            fallback_used: false,
            cpu_fallback_allowed: false,
            graph_execution: true,
            bitnet_inference: false,
            error: None,
        };
        let receipt = build_smoke_receipt(
            visible_npu_probe(vec!["CPU".to_string(), "NPU".to_string()]),
            smoke,
            true,
            "2026-05-06T00:00:00Z".to_string(),
            Some("ci/hardware/intel-258v/2026-05-06/npu-tiny-graph-smoke.json".to_string()),
        );

        assert_eq!(receipt["artifact_kind"], "intel_npu_tiny_graph_smoke");
        assert_eq!(receipt["proof_stage"], "kernel_smoke_tested");
        assert_eq!(receipt["requested_backend"], "intel-npu");
        assert_eq!(receipt["selected_backend"], "intel-npu-openvino");
        assert_eq!(receipt["runtime_api"], "openvino");
        assert_eq!(receipt["runtime_device"], "NPU");
        assert_eq!(receipt["backend_runtime"]["name"], "openvino");
        assert_eq!(receipt["backend_runtime"]["version"], "2026.1");
        assert_eq!(receipt["backend_runtime"]["device"], "NPU");
        assert_eq!(receipt["backend_runtime"]["device_name"], "Intel(R) AI Boost");
        assert_eq!(receipt["backend_runtime"]["driver_version"], "1.2.3");
        assert_eq!(receipt["backend_runtime"]["compiler_version"], "4.5.6");
        assert_eq!(receipt["backend_runtime"]["max_tiles"], 1);
        assert_eq!(receipt["shape_mode"], "static");
        assert_eq!(receipt["shape_contract"]["shape_mode"], "static");
        assert_eq!(receipt["shape_contract"]["input_shape"], serde_json::json!([1, 16]));
        assert_eq!(receipt["shape_contract"]["output_shape"], serde_json::json!([1, 16]));
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["cpu_fallback_allowed"], false);
        assert_eq!(receipt["fallback_policy"]["fallback_used"], false);
        assert!(receipt["fallback_policy"]["fallback_backend"].is_null());
        assert!(receipt["fallback_policy"]["fallback_reason"].is_null());
        assert_eq!(receipt["fallback_policy"]["cpu_fallback_allowed"], false);
        assert_eq!(receipt["graph_execution"], true);
        assert_eq!(receipt["kernel_execution"], false);
        assert_eq!(receipt["bitnet_inference"], false);
        assert_eq!(receipt["graph"]["name"], "tiny_matmul_add_f16_1x16");
        assert_eq!(receipt["graph"]["precision"], "F16");
        assert!(receipt["graph"]["cache_dir"].is_null());
        assert_eq!(receipt["graph"]["input_shape"], serde_json::json!([1, 16]));
        assert_eq!(receipt["graph"]["output_shape"], serde_json::json!([1, 16]));
        assert_eq!(receipt["graph"]["result"], "pass");
        assert_eq!(receipt["timing"]["cached_compile_ms"], 12.5);
        assert_eq!(receipt["timing"]["first_infer_ms"], 1.25);
        assert!(receipt["timing"]["first_ever_compile_and_infer_ms"].is_null());
        assert!(receipt["timing"]["steady_state_infer_ms"].is_null());
        assert_eq!(receipt["kernels_or_graphs"], serde_json::json!(["tiny_matmul_openvino_npu"]));
        assert!(receipt["error"].is_null());
    }

    #[test]
    fn strict_intel_npu_smoke_records_error_without_fallback() {
        let smoke = bitnet_device_probe::runtimes::OpenVinoNpuTinyGraphSmoke {
            passed: false,
            proof_stage: "runtime_detected".to_string(),
            requested_backend: "intel-npu".to_string(),
            selected_backend: None,
            runtime_api: None,
            runtime_device: None,
            openvino_version: None,
            openvino_available_devices: Vec::new(),
            graph_name: "tiny_matmul_add_f16_1x16".to_string(),
            shape_mode: "static".to_string(),
            input_shape: vec![1, 16],
            output_shape: None,
            precision: "F16".to_string(),
            tolerance: 0.001,
            max_abs_error: None,
            mean_abs_error: None,
            compile_ms: None,
            first_infer_ms: None,
            fallback_used: false,
            cpu_fallback_allowed: false,
            graph_execution: false,
            bitnet_inference: false,
            error: Some("OpenVINO did not report NPU".to_string()),
        };
        let receipt = build_smoke_receipt(
            openvino_missing_npu_probe(),
            smoke,
            true,
            "2026-05-06T00:00:00Z".to_string(),
            None,
        );

        assert_eq!(receipt["artifact_kind"], "intel_npu_tiny_graph_smoke");
        assert_eq!(receipt["proof_stage"], "runtime_detected");
        assert!(receipt["selected_backend"].is_null());
        assert!(receipt["runtime_api"].is_null());
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["fallback_policy"]["fallback_used"], false);
        assert_eq!(receipt["fallback_policy"]["cpu_fallback_allowed"], false);
        assert_eq!(receipt["shape_contract"]["shape_mode"], "static");
        assert_eq!(receipt["shape_contract"]["input_shape"], serde_json::json!([1, 16]));
        assert!(receipt["shape_contract"]["output_shape"].is_null());
        assert_eq!(receipt["graph_execution"], false);
        assert_eq!(receipt["bitnet_inference"], false);
        assert_eq!(
            receipt["claim"],
            "intel_npu_runtime_visibility_recorded_without_openvino_graph"
        );
        assert_eq!(receipt["error"], "OpenVINO did not report NPU");
    }

    #[test]
    fn intel_npu_bitnet_subgraph_receipt_records_parity_only() {
        let parity = bitnet_device_probe::runtimes::OpenVinoNpuBitnetSubgraphParity {
            passed: true,
            proof_stage: "parity_tested".to_string(),
            requested_backend: "intel-npu".to_string(),
            selected_backend: Some("intel-npu-openvino".to_string()),
            runtime_api: Some("openvino".to_string()),
            runtime_device: Some("NPU".to_string()),
            openvino_version: Some("2026.1".to_string()),
            openvino_available_devices: vec!["CPU".to_string(), "NPU".to_string()],
            subgraph_name: "bitnet_rmsnorm_f16_1x16".to_string(),
            bitnet_op: "rmsnorm".to_string(),
            reference_path: "cpu_numpy_rmsnorm_f32".to_string(),
            shape_mode: "static".to_string(),
            input_shape: vec![1, 16],
            output_shape: Some(vec![1, 16]),
            precision: "F16".to_string(),
            epsilon: 0.00001,
            tolerance: 0.005,
            max_abs_error: Some(0.0009),
            mean_abs_error: Some(0.0002),
            compile_ms: Some(14.5),
            first_infer_ms: Some(1.5),
            fallback_used: false,
            cpu_fallback_allowed: false,
            graph_execution: true,
            bitnet_inference: false,
            qk256_decode: false,
            error: None,
        };
        let receipt = build_bitnet_subgraph_receipt(
            visible_npu_probe(vec!["CPU".to_string(), "NPU".to_string()]),
            parity,
            true,
            "2026-05-06T00:00:00Z".to_string(),
            Some("ci/hardware/intel-258v/2026-05-06/npu-bitnet-subgraph.json".to_string()),
        );

        assert_eq!(receipt["artifact_kind"], "intel_npu_bitnet_subgraph_parity");
        assert_eq!(receipt["proof_stage"], "parity_tested");
        assert_eq!(receipt["requested_backend"], "intel-npu");
        assert_eq!(receipt["selected_backend"], "intel-npu-openvino");
        assert_eq!(receipt["runtime_api"], "openvino");
        assert_eq!(receipt["runtime_device"], "NPU");
        assert_eq!(receipt["backend_runtime"]["name"], "openvino");
        assert_eq!(receipt["backend_runtime"]["version"], "2026.1");
        assert_eq!(receipt["shape_contract"]["shape_mode"], "static");
        assert_eq!(receipt["shape_contract"]["input_shape"], serde_json::json!([1, 16]));
        assert_eq!(receipt["shape_contract"]["output_shape"], serde_json::json!([1, 16]));
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["fallback_policy"]["fallback_used"], false);
        assert_eq!(receipt["fallback_policy"]["cpu_fallback_allowed"], false);
        assert_eq!(receipt["graph_execution"], true);
        assert_eq!(receipt["kernel_execution"], false);
        assert_eq!(receipt["bitnet_inference"], false);
        assert_eq!(receipt["qk256_decode"], false);
        assert_eq!(receipt["subgraph"]["name"], "bitnet_rmsnorm_f16_1x16");
        assert_eq!(receipt["subgraph"]["bitnet_op"], "rmsnorm");
        assert_eq!(receipt["subgraph"]["reference_path"], "cpu_numpy_rmsnorm_f32");
        assert_eq!(receipt["subgraph"]["result"], "pass");
        assert_eq!(receipt["timing"]["cached_compile_ms"], 14.5);
        assert_eq!(receipt["timing"]["first_infer_ms"], 1.5);
        assert_eq!(
            receipt["kernels_or_graphs"],
            serde_json::json!(["bitnet_rmsnorm_openvino_npu"])
        );
        assert_eq!(receipt["claim"], "openvino_npu_bitnet_subgraph_parity_passed");
        assert!(receipt["error"].is_null());
    }

    #[test]
    fn intel_npu_bitnet_linear_subgraph_receipt_records_parity_only() {
        let parity = bitnet_device_probe::runtimes::OpenVinoNpuBitnetSubgraphParity {
            passed: true,
            proof_stage: "parity_tested".to_string(),
            requested_backend: "intel-npu".to_string(),
            selected_backend: Some("intel-npu-openvino".to_string()),
            runtime_api: Some("openvino".to_string()),
            runtime_device: Some("NPU".to_string()),
            openvino_version: Some("2026.1".to_string()),
            openvino_available_devices: vec!["CPU".to_string(), "NPU".to_string()],
            subgraph_name: "bitnet_linear_projection_f16_1x16x16".to_string(),
            bitnet_op: "linear_projection".to_string(),
            reference_path: "cpu_numpy_linear_f32".to_string(),
            shape_mode: "static".to_string(),
            input_shape: vec![1, 16],
            output_shape: Some(vec![1, 16]),
            precision: "F16".to_string(),
            epsilon: 0.0,
            tolerance: 0.01,
            max_abs_error: Some(0.001),
            mean_abs_error: Some(0.0003),
            compile_ms: Some(15.5),
            first_infer_ms: Some(1.75),
            fallback_used: false,
            cpu_fallback_allowed: false,
            graph_execution: true,
            bitnet_inference: false,
            qk256_decode: false,
            error: None,
        };
        let receipt = build_bitnet_subgraph_receipt(
            visible_npu_probe(vec!["CPU".to_string(), "NPU".to_string()]),
            parity,
            true,
            "2026-05-06T00:00:00Z".to_string(),
            Some("ci/hardware/intel-258v/2026-05-06/npu-bitnet-linear-subgraph.json".to_string()),
        );

        assert_eq!(receipt["artifact_kind"], "intel_npu_bitnet_subgraph_parity");
        assert_eq!(receipt["proof_stage"], "parity_tested");
        assert_eq!(receipt["selected_backend"], "intel-npu-openvino");
        assert_eq!(receipt["runtime_api"], "openvino");
        assert_eq!(receipt["runtime_device"], "NPU");
        assert_eq!(receipt["shape_contract"]["input_shape"], serde_json::json!([1, 16]));
        assert_eq!(receipt["shape_contract"]["output_shape"], serde_json::json!([1, 16]));
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["fallback_policy"]["cpu_fallback_allowed"], false);
        assert_eq!(receipt["graph_execution"], true);
        assert_eq!(receipt["bitnet_inference"], false);
        assert_eq!(receipt["qk256_decode"], false);
        assert_eq!(receipt["subgraph"]["name"], "bitnet_linear_projection_f16_1x16x16");
        assert_eq!(receipt["subgraph"]["bitnet_op"], "linear_projection");
        assert_eq!(receipt["subgraph"]["reference_path"], "cpu_numpy_linear_f32");
        assert_eq!(receipt["subgraph"]["epsilon"], 0.0);
        assert_eq!(receipt["subgraph"]["result"], "pass");
        assert_eq!(
            receipt["kernels_or_graphs"],
            serde_json::json!(["bitnet_linear_projection_openvino_npu"])
        );
        assert_eq!(receipt["claim"], "openvino_npu_bitnet_subgraph_parity_passed");
        assert!(receipt["error"].is_null());
    }

    #[test]
    fn intel_npu_bitnet_ffn_subgraph_receipt_records_parity_only() {
        let parity = bitnet_device_probe::runtimes::OpenVinoNpuBitnetSubgraphParity {
            passed: true,
            proof_stage: "parity_tested".to_string(),
            requested_backend: "intel-npu".to_string(),
            selected_backend: Some("intel-npu-openvino".to_string()),
            runtime_api: Some("openvino".to_string()),
            runtime_device: Some("NPU".to_string()),
            openvino_version: Some("2026.1".to_string()),
            openvino_available_devices: vec!["CPU".to_string(), "NPU".to_string()],
            subgraph_name: "bitnet_ffn_relu2_f16_1x16x32".to_string(),
            bitnet_op: "ffn_relu2".to_string(),
            reference_path: "cpu_numpy_ffn_relu2_f32".to_string(),
            shape_mode: "static".to_string(),
            input_shape: vec![1, 16],
            output_shape: Some(vec![1, 16]),
            precision: "F16".to_string(),
            epsilon: 0.0,
            tolerance: 0.05,
            max_abs_error: Some(0.004),
            mean_abs_error: Some(0.0007),
            compile_ms: Some(18.5),
            first_infer_ms: Some(2.25),
            fallback_used: false,
            cpu_fallback_allowed: false,
            graph_execution: true,
            bitnet_inference: false,
            qk256_decode: false,
            error: None,
        };
        let receipt = build_bitnet_subgraph_receipt_with_cpu_reference(
            visible_npu_probe(vec!["CPU".to_string(), "NPU".to_string()]),
            parity,
            true,
            "2026-05-06T00:00:00Z".to_string(),
            Some("ci/hardware/intel-258v/2026-05-06/npu-bitnet-ffn-subgraph.json".to_string()),
            Some(
                "ci/hardware/intel-258v/2026-05-08/cpu-reference-bundle-post-mechanics.json"
                    .to_string(),
            ),
        );

        assert_eq!(receipt["artifact_kind"], "intel_npu_bitnet_subgraph_parity");
        assert_eq!(receipt["proof_stage"], "parity_tested");
        assert_eq!(receipt["selected_backend"], "intel-npu-openvino");
        assert_eq!(receipt["runtime_api"], "openvino");
        assert_eq!(receipt["runtime_device"], "NPU");
        assert_eq!(receipt["shape_contract"]["input_shape"], serde_json::json!([1, 16]));
        assert_eq!(receipt["shape_contract"]["output_shape"], serde_json::json!([1, 16]));
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["fallback_policy"]["cpu_fallback_allowed"], false);
        assert_eq!(receipt["graph_execution"], true);
        assert_eq!(receipt["bitnet_inference"], false);
        assert_eq!(receipt["qk256_decode"], false);
        assert_eq!(receipt["subgraph"]["name"], "bitnet_ffn_relu2_f16_1x16x32");
        assert_eq!(receipt["subgraph"]["bitnet_op"], "ffn_relu2");
        assert_eq!(receipt["subgraph"]["reference_path"], "cpu_numpy_ffn_relu2_f32");
        assert_eq!(receipt["subgraph"]["epsilon"], 0.0);
        assert!(
            (receipt["subgraph"]["tolerance"].as_f64().unwrap_or_default() - 0.05).abs() < 1e-6
        );
        assert_eq!(receipt["subgraph"]["result"], "pass");
        assert_eq!(
            receipt["cpu_reference"]["artifact_path"],
            "ci/hardware/intel-258v/2026-05-08/cpu-reference-bundle-post-mechanics.json"
        );
        assert_eq!(
            receipt["kernels_or_graphs"],
            serde_json::json!(["bitnet_ffn_relu2_openvino_npu"])
        );
        assert_eq!(receipt["claim"], "openvino_npu_bitnet_subgraph_parity_passed");
        assert!(receipt["error"].is_null());
    }

    #[test]
    fn strict_intel_npu_bitnet_subgraph_records_error_without_fallback() {
        let parity = bitnet_device_probe::runtimes::OpenVinoNpuBitnetSubgraphParity {
            passed: false,
            proof_stage: "runtime_detected".to_string(),
            requested_backend: "intel-npu".to_string(),
            selected_backend: None,
            runtime_api: None,
            runtime_device: None,
            openvino_version: None,
            openvino_available_devices: Vec::new(),
            subgraph_name: "bitnet_rmsnorm_f16_1x16".to_string(),
            bitnet_op: "rmsnorm".to_string(),
            reference_path: "cpu_numpy_rmsnorm_f32".to_string(),
            shape_mode: "static".to_string(),
            input_shape: vec![1, 16],
            output_shape: None,
            precision: "F16".to_string(),
            epsilon: 0.00001,
            tolerance: 0.005,
            max_abs_error: None,
            mean_abs_error: None,
            compile_ms: None,
            first_infer_ms: None,
            fallback_used: false,
            cpu_fallback_allowed: false,
            graph_execution: false,
            bitnet_inference: false,
            qk256_decode: false,
            error: Some("OpenVINO did not report NPU".to_string()),
        };
        let receipt = build_bitnet_subgraph_receipt(
            openvino_missing_npu_probe(),
            parity,
            true,
            "2026-05-06T00:00:00Z".to_string(),
            None,
        );

        assert_eq!(receipt["artifact_kind"], "intel_npu_bitnet_subgraph_parity");
        assert_eq!(receipt["proof_stage"], "runtime_detected");
        assert!(receipt["selected_backend"].is_null());
        assert!(receipt["runtime_api"].is_null());
        assert_eq!(receipt["fallback_used"], false);
        assert_eq!(receipt["fallback_policy"]["fallback_used"], false);
        assert_eq!(receipt["fallback_policy"]["cpu_fallback_allowed"], false);
        assert_eq!(receipt["shape_contract"]["shape_mode"], "static");
        assert_eq!(receipt["shape_contract"]["input_shape"], serde_json::json!([1, 16]));
        assert!(receipt["shape_contract"]["output_shape"].is_null());
        assert_eq!(receipt["graph_execution"], false);
        assert_eq!(receipt["bitnet_inference"], false);
        assert_eq!(receipt["qk256_decode"], false);
        assert_eq!(
            receipt["claim"],
            "intel_npu_runtime_visibility_recorded_without_bitnet_subgraph"
        );
        assert_eq!(receipt["error"], "OpenVINO did not report NPU");
    }
}
