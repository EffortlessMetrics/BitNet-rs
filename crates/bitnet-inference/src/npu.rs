//! NPU integration utilities for inference.
//!
//! This module centralizes environment-driven controls for the NPU path so the
//! engine and CLI can expose a stable "npu" target while backend wiring to
//! Qualcomm QNN/SNPE matures.

use bitnet_common::Device;

/// Environment variable used to enable NPU routing.
pub const BITNET_ENABLE_NPU: &str = "BITNET_ENABLE_NPU";

/// Return `true` when the runtime should prefer NPU execution.
pub fn npu_requested() -> bool {
    std::env::var(BITNET_ENABLE_NPU)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Map an external `--device` style token to an internal device preference.
pub fn map_device_token(token: &str) -> Option<Device> {
    match token {
        "cpu" => Some(Device::Cpu),
        "cuda" | "gpu" => Some(Device::Cuda(0)),
        "metal" | "npu" => Some(Device::Metal),
        "oneapi" | "opencl" | "intel-gpu" => Some(Device::OpenCL(0)),
        "vulkan" => Some(Device::Vulkan(0)),
        _ => None,
    }
}
