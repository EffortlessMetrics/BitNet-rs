//! NPU integration utilities for inference.
//!
//! This module centralizes environment-driven controls for the NPU path so the
//! engine and CLI can expose a stable "npu" target while backend wiring to
//! Intel OpenVINO NPU runtime integration matures.

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
    match token.trim().to_ascii_lowercase().as_str() {
        "cpu" => Some(Device::Cpu),
        "cuda" | "gpu" => Some(Device::Cuda(0)),
        "opencl" | "intel-gpu" | "intel-arc-140v" | "arc-140v" => Some(Device::OpenCL(0)),
        "npu" | "intel-npu" | "openvino-npu" => Some(Device::Npu),
        "metal" => Some(Device::Metal),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npu_tokens_preserve_npu_identity() {
        assert_eq!(map_device_token("npu"), Some(Device::Npu));
        assert_eq!(map_device_token("intel-npu"), Some(Device::Npu));
        assert_eq!(map_device_token("openvino-npu"), Some(Device::Npu));
    }

    #[test]
    fn metal_token_remains_distinct_from_npu() {
        assert_eq!(map_device_token("metal"), Some(Device::Metal));
        assert_ne!(map_device_token("npu"), Some(Device::Metal));
    }
}
