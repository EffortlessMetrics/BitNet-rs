#![cfg(feature = "cuda")]

use bitnet_device_probe::{NvidiaCudaProbe, probe_nvidia_cuda};
use bitnet_kernels::gpu::{
    CUDA_TINY_VECTOR_ADD_FIXTURE_ID, CUDA_TINY_VECTOR_ADD_INPUT_LEN,
    CUDA_TINY_VECTOR_ADD_KERNEL_ID, CUDA_TINY_VECTOR_ADD_PARITY_TOLERANCE, CudaDeviceInfo,
    CudaTinyVectorAddComparison, CudaTinyVectorAddMismatch, CudaTinyVectorAddParity,
    compare_cuda_tiny_vector_add_outputs, cuda_tiny_vector_add_inputs,
    expected_cuda_tiny_vector_add, run_cuda_tiny_vector_add_parity,
};
use serde_json::{Value, json};
use std::{
    error::Error,
    io,
    path::{Path, PathBuf},
};

const RUN_ENV: &str = "BITNET_RUN_RTX5070TI_CUDA_PARITY";
const RECEIPT_ENV: &str = "BITNET_RTX5070TI_CUDA_PARITY_RECEIPT";
const ARTIFACT_PATH_ENV: &str = "BITNET_RTX5070TI_CUDA_PARITY_ARTIFACT_PATH";
const DEBUG_ARTIFACT_ENV: &str = "BITNET_RTX5070TI_CUDA_PARITY_DEBUG_ARTIFACT";
const DATE_ENV: &str = "BITNET_RTX5070TI_CUDA_PARITY_DATE";
const TIMESTAMP_ENV: &str = "BITNET_RTX5070TI_CUDA_PARITY_TIMESTAMP_UTC";
const DEVICE_INDEX_ENV: &str = "BITNET_RTX5070TI_CUDA_DEVICE_INDEX";

const MACHINE_ID: &str = "windows-9950x3d-rtx5070ti";
const ARTIFACT_KIND: &str = "cuda_parity";
const HARDWARE_LANE: &str = "nvidia-rtx-5070-ti-cuda";
const REQUESTED_BACKEND: &str = "nvidia-rtx-5070-ti-cuda";
const SELECTED_BACKEND: &str = "nvidia-rtx-5070-ti-cuda";
const RUNTIME_API: &str = "cuda";
const REFERENCE_BACKEND: &str = "amd-9950x3d-cpu-avx512";
const CLAIM: &str = "cuda_cpu_parity_tested";
const TOLERANCE_SOURCE: &str = "docs/bitnet/BITNET_PARITY_TOLERANCES.md";

#[test]
fn tiny_vector_add_expected_output_matches_cpu_reference() -> Result<(), Box<dyn Error>> {
    let (a, b) = cuda_tiny_vector_add_inputs();
    let expected = expected_cuda_tiny_vector_add(&a, &b)?;

    assert_eq!(expected.len(), CUDA_TINY_VECTOR_ADD_INPUT_LEN);
    for (index, value) in expected.iter().enumerate() {
        assert_eq!(*value, a[index] + b[index]);
    }

    Ok(())
}

#[test]
fn parity_comparison_records_mismatch_without_fallback() -> Result<(), Box<dyn Error>> {
    let expected = [1.0_f32, 2.0, 3.0];
    let actual = [1.0_f32, 20.0, 3.0];

    let comparison = compare_cuda_tiny_vector_add_outputs(&expected, &actual, 1e-6)?;

    assert!(!comparison.passed);
    assert_eq!(
        comparison.first_mismatch,
        Some(CudaTinyVectorAddMismatch { index: 1, expected: 2.0, actual: 20.0, abs_error: 18.0 })
    );

    Ok(())
}

#[test]
fn parity_receipt_contract_records_9950x3d_reference_and_cuda_target() {
    let receipt = cuda_parity_receipt_json(
        &synthetic_passed_parity(),
        &synthetic_probe(),
        "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-parity.json",
        "2026-05-06T00:00:00Z",
        None,
    );

    assert_eq!(receipt["machine_id"], MACHINE_ID);
    assert_eq!(receipt["artifact_kind"], ARTIFACT_KIND);
    assert_eq!(receipt["requested_backend"], REQUESTED_BACKEND);
    assert_eq!(receipt["selected_backend"], SELECTED_BACKEND);
    assert_eq!(receipt["runtime_api"], RUNTIME_API);
    assert_eq!(receipt["fallback_used"], false);
    assert_eq!(receipt["parity"]["reference_backend"], REFERENCE_BACKEND);
    assert_eq!(receipt["parity"]["target_backend"], SELECTED_BACKEND);
    assert_eq!(receipt["parity"]["kernel_id"], CUDA_TINY_VECTOR_ADD_KERNEL_ID);
    assert_eq!(receipt["parity"]["fixture_id"], CUDA_TINY_VECTOR_ADD_FIXTURE_ID);
    assert_eq!(receipt["parity"]["passed"], true);
    assert_eq!(receipt["kernel_stats"][0]["fallback_invocations"], 0);
    assert_eq!(receipt["claim"], CLAIM);
}

#[test]
fn rtx5070ti_device_name_detection_is_specific() {
    assert!(is_rtx5070ti_device_name("NVIDIA GeForce RTX 5070 Ti"));
    assert!(is_rtx5070ti_device_name("nvidia geforce rtx5070ti"));
    assert!(!is_rtx5070ti_device_name("NVIDIA GeForce RTX 5070"));
    assert!(!is_rtx5070ti_device_name("Generic CUDA Device"));
}

#[test]
fn tiny_rtx5070ti_cuda_add_matches_cpu_reference_when_enabled() -> Result<(), Box<dyn Error>> {
    if std::env::var(RUN_ENV).as_deref() != Ok("1") {
        eprintln!("skipping live RTX 5070 Ti CUDA parity; set {RUN_ENV}=1 to run it");
        return Ok(());
    }

    let device_index = selected_device_index()?;
    let probe = probe_nvidia_cuda(Some(device_index));
    if !probe.available {
        return Err(io_error(format!(
            "RTX5070TI-006 parity requires CUDA probe success: {:?}",
            probe.failure_reason
        )));
    }

    let parity =
        run_cuda_tiny_vector_add_parity(device_index, CUDA_TINY_VECTOR_ADD_PARITY_TOLERANCE)?;
    if !is_rtx5070ti_device_name(&parity.device_info.name) {
        return Err(io_error(format!(
            "RTX5070TI-006 parity requires NVIDIA GeForce RTX 5070 Ti; found '{}'",
            parity.device_info.name
        )));
    }

    let date = std::env::var(DATE_ENV).unwrap_or_else(|_| "2026-05-06".to_string());
    let artifact_path = std::env::var(ARTIFACT_PATH_ENV)
        .or_else(|_| std::env::var(RECEIPT_ENV))
        .unwrap_or_else(|_| cuda_parity_artifact_path(&date));
    let timestamp_utc = std::env::var(TIMESTAMP_ENV).unwrap_or_else(|_| "TBD".to_string());

    let debug_artifact_path = if parity.passed {
        None
    } else {
        Some(
            std::env::var(DEBUG_ARTIFACT_ENV)
                .unwrap_or_else(|_| cuda_parity_debug_artifact_path(&date)),
        )
    };

    if let Some(path) = debug_artifact_path.as_deref() {
        write_json_file(path, &cuda_parity_debug_artifact_json(&parity))?;
    }

    let receipt_json = cuda_parity_receipt_json(
        &parity,
        &probe,
        &artifact_path,
        &timestamp_utc,
        debug_artifact_path.as_deref(),
    );

    if let Ok(path) = std::env::var(RECEIPT_ENV) {
        write_json_file(&path, &receipt_json)?;
    }

    println!("{}", serde_json::to_string_pretty(&receipt_json)?);

    if !parity.passed {
        return Err(io_error(format!(
            "RTX5070TI-006 parity failed; debug artifact: {}",
            debug_artifact_path.as_deref().unwrap_or("not written")
        )));
    }

    Ok(())
}

fn selected_device_index() -> Result<usize, Box<dyn Error>> {
    match std::env::var(DEVICE_INDEX_ENV) {
        Ok(value) => value.parse::<usize>().map_err(|error| {
            io_error(format!("{DEVICE_INDEX_ENV} must be a non-negative integer: {error}"))
        }),
        Err(_) => Ok(0),
    }
}

fn is_rtx5070ti_device_name(name: &str) -> bool {
    let compact = name
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();

    compact.contains("nvidia") && compact.contains("rtx5070ti")
}

fn cuda_parity_artifact_path(date: &str) -> String {
    format!("ci/hardware/windows-9950x3d-rtx5070ti/{date}/cuda-parity.json")
}

fn cuda_parity_debug_artifact_path(date: &str) -> String {
    format!("ci/hardware/windows-9950x3d-rtx5070ti/{date}/cuda-parity-mismatch.json")
}

fn write_json_file(path: &str, value: &Value) -> Result<(), Box<dyn Error>> {
    let output_path = workspace_relative_path(path);
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output_path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn workspace_relative_path(path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        return path.to_path_buf();
    }

    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join(path)
}

fn cuda_parity_receipt_json(
    parity: &CudaTinyVectorAddParity,
    probe: &NvidiaCudaProbe,
    artifact_path: &str,
    timestamp_utc: &str,
    debug_artifact_path: Option<&str>,
) -> Value {
    json!({
        "schema": 1,
        "artifact_kind": ARTIFACT_KIND,
        "artifact_path": artifact_path,
        "claim": CLAIM,
        "cuda": {
            "available": probe.available,
            "compute_capability": probe.compute_capability.clone().unwrap_or_else(|| {
                format!(
                    "{}.{}",
                    parity.device_info.compute_capability.0,
                    parity.device_info.compute_capability.1
                )
            }),
            "cuda_runtime_version": probe.cuda_runtime_version,
            "cuda_toolkit_version": probe.cuda_toolkit_version,
            "device_count": probe.device_count,
            "device_index": parity.device_info.device_id,
            "device_name": parity.device_info.name,
            "driver_version": probe.driver_version,
            "nvml_available": probe.nvml_available,
            "nvrtc_version": probe.nvrtc_version,
            "power_draw_watts": probe.power_draw_watts,
            "power_limit_watts": probe.power_limit_watts,
            "selected_device_index": probe.selected_device_index.unwrap_or(parity.device_info.device_id),
            "temperature_c": probe.temperature_c,
            "vram_bytes": probe.vram_bytes.unwrap_or(parity.device_info.total_memory as u64)
        },
        "error": null,
        "fallback_backend": null,
        "fallback_reason": null,
        "fallback_used": false,
        "hardware_lane": HARDWARE_LANE,
        "input_len": parity.input_len,
        "kernel_stats": [
            {
                "device_to_host_bytes": parity.device_to_host_bytes,
                "fallback_invocations": 0,
                "host_to_device_bytes": parity.host_to_device_bytes,
                "invocations": 1,
                "kernel_id": parity.kernel_id,
                "kernel_launches": parity.kernel_launches,
                "kernel_time_ms": null
            }
        ],
        "machine_id": MACHINE_ID,
        "max_abs_error": parity.max_abs_error,
        "mean_abs_error": parity.mean_abs_error,
        "parity": {
            "debug_artifact_path": debug_artifact_path,
            "fixture_id": parity.fixture_id,
            "kernel_id": parity.kernel_id,
            "max_abs_error": parity.max_abs_error,
            "mean_abs_error": parity.mean_abs_error,
            "passed": parity.passed,
            "reference_backend": REFERENCE_BACKEND,
            "target_backend": SELECTED_BACKEND,
            "tolerance": parity.tolerance,
            "tolerance_source": TOLERANCE_SOURCE
        },
        "reference_backend": REFERENCE_BACKEND,
        "requested_backend": REQUESTED_BACKEND,
        "result": if parity.passed { "pass" } else { "fail" },
        "runtime_api": RUNTIME_API,
        "selected_backend": SELECTED_BACKEND,
        "timestamp_utc": timestamp_utc
    })
}

fn cuda_parity_debug_artifact_json(parity: &CudaTinyVectorAddParity) -> Value {
    json!({
        "artifact_kind": "cuda_parity_mismatch",
        "machine_id": MACHINE_ID,
        "requested_backend": REQUESTED_BACKEND,
        "selected_backend": SELECTED_BACKEND,
        "runtime_api": RUNTIME_API,
        "fallback_used": false,
        "kernel_id": parity.kernel_id,
        "fixture_id": parity.fixture_id,
        "input_len": parity.input_len,
        "tolerance": parity.tolerance,
        "max_abs_error": parity.max_abs_error,
        "mean_abs_error": parity.mean_abs_error,
        "first_mismatch": parity.first_mismatch.as_ref().map(first_mismatch_json)
    })
}

fn first_mismatch_json(mismatch: &CudaTinyVectorAddMismatch) -> Value {
    json!({
        "index": mismatch.index,
        "expected": mismatch.expected,
        "actual": mismatch.actual,
        "abs_error": mismatch.abs_error
    })
}

fn synthetic_passed_parity() -> CudaTinyVectorAddParity {
    let comparison = CudaTinyVectorAddComparison {
        passed: true,
        max_abs_error: 0.0,
        mean_abs_error: 0.0,
        first_mismatch: None,
    };
    CudaTinyVectorAddParity {
        kernel_id: CUDA_TINY_VECTOR_ADD_KERNEL_ID,
        fixture_id: CUDA_TINY_VECTOR_ADD_FIXTURE_ID,
        device_info: CudaDeviceInfo {
            device_id: 0,
            name: "NVIDIA GeForce RTX 5070 Ti".to_string(),
            compute_capability: (12, 0),
            total_memory: 17_094_475_776,
            multiprocessor_count: 70,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 99_328,
            supports_fp16: true,
            supports_bf16: true,
        },
        input_len: CUDA_TINY_VECTOR_ADD_INPUT_LEN,
        passed: comparison.passed,
        tolerance: CUDA_TINY_VECTOR_ADD_PARITY_TOLERANCE,
        max_abs_error: comparison.max_abs_error,
        mean_abs_error: comparison.mean_abs_error,
        first_mismatch: comparison.first_mismatch,
        host_to_device_bytes: 8192,
        device_to_host_bytes: 4096,
        kernel_launches: 1,
    }
}

fn synthetic_probe() -> NvidiaCudaProbe {
    NvidiaCudaProbe {
        available: true,
        device_count: 1,
        selected_device_index: Some(0),
        selected_device_name: Some("NVIDIA GeForce RTX 5070 Ti".to_string()),
        compute_capability: Some("12.0".to_string()),
        driver_version: Some("591.86".to_string()),
        cuda_runtime_version: Some("12.9".to_string()),
        cuda_toolkit_version: Some("12.9".to_string()),
        nvrtc_version: Some("12.9".to_string()),
        nvml_available: true,
        vram_bytes: Some(17_094_475_776),
        power_limit_watts: Some(300.0),
        power_draw_watts: Some(34.97),
        temperature_c: Some(38.0),
        failure_reason: None,
    }
}

fn io_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::other(message.into()))
}
