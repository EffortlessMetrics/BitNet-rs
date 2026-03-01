//! NPU integration utilities for inference.
//!
//! This module centralizes environment-driven controls for the NPU path so the
//! engine and CLI can expose a stable "npu" target while backend wiring to
//! Qualcomm QNN/SNPE matures.

use bitnet_common::Device;

pub use bitnet_qualcomm::BITNET_ENABLE_NPU;
use bitnet_qualcomm::npu_enabled;

/// Return `true` when the runtime should prefer NPU execution.
pub fn npu_requested() -> bool {
    npu_enabled()
}

/// Map an external `--device` style token to an internal device preference.
pub fn map_device_token(token: &str) -> Option<Device> {
    match token {
        "cpu" => Some(Device::Cpu),
        "cuda" | "gpu" => Some(Device::Cuda(0)),
        "metal" | "npu" => Some(Device::Metal),
        "oneapi" | "opencl" | "intel-gpu" => Some(Device::OpenCL(0)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_npu_alias() {
        assert_eq!(map_device_token("npu"), Some(Device::Metal));
    }
}
