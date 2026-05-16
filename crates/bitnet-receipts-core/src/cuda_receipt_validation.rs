//! CUDA receipt loading and common RTX 5070 Ti proof validation.
//!
//! This module keeps file IO and shared CUDA receipt invariants separate from
//! artifact-specific receipt validators.

use anyhow::{Result, anyhow};
use serde_json::Value;
use std::path::Path;

use crate::receipt_field_validation::*;

pub(crate) fn load_json_receipt(path: &Path) -> Result<Value> {
    let content = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}

pub(crate) fn validate_cuda_receipt_common<'a>(
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

pub(crate) fn first_kernel_stats(receipt: &Value) -> Result<&Value> {
    let stats = object_field(receipt, "kernel_stats")?;
    let stats = stats.as_array().ok_or_else(|| anyhow!("kernel_stats must be an array"))?;
    stats.first().ok_or_else(|| anyhow!("kernel_stats must contain at least one entry"))
}
