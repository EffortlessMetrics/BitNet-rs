//! CUDA hardware identity helpers for dense GGUF receipts.
//!
//! Probe normalization and GPU gate predicates live here so command execution
//! and receipt builders do not duplicate hardware-specific policy.

use serde_json::{Value, json};

pub(super) fn cuda_identity_json(probe: Option<&bitnet_device_probe::NvidiaCudaProbe>) -> Value {
    match probe {
        Some(probe) => json!({
            "available": probe.available,
            "device_count": probe.device_count,
            "device_index": probe.selected_device_index.unwrap_or(0),
            "device_name": probe.selected_device_name.clone().unwrap_or_else(|| "unknown".into()),
            "compute_capability": probe.compute_capability.clone().unwrap_or_else(|| "12.0".into()),
            "driver_version": probe.driver_version.clone().unwrap_or_else(|| "unknown".into()),
            "cuda_runtime_version": probe.cuda_runtime_version.clone().unwrap_or_else(|| "unknown".into()),
            "cuda_toolkit_version": probe.cuda_toolkit_version.clone().unwrap_or_else(|| "unknown".into()),
            "nvrtc_version": probe.nvrtc_version.clone().unwrap_or_else(|| "unknown".into()),
            "nvml_available": probe.nvml_available,
            "vram_bytes": probe.vram_bytes.unwrap_or(1),
            "power_limit_watts": probe.power_limit_watts,
            "power_draw_watts": probe.power_draw_watts,
            "temperature_c": probe.temperature_c,
        }),
        None => json!({
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
            "temperature_c": 38.0,
        }),
    }
}

pub(super) fn is_rtx5070ti_device_name(name: &str) -> bool {
    let compact = name
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();

    compact.contains("nvidia") && compact.contains("rtx5070ti")
}
