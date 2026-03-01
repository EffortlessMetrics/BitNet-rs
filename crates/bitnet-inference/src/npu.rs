//! NPU integration utilities for inference.
//!
//! This module centralizes device-token mapping for the NPU path while runtime
//! enablement controls are sourced from the `bitnet-qualcomm` microcrate.

use bitnet_common::Device;

pub use bitnet_qualcomm::{BITNET_ENABLE_NPU, npu_requested};

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
    fn npu_alias_maps_to_metal_path() {
        assert_eq!(map_device_token("npu"), Some(Device::Metal));
    }
}
