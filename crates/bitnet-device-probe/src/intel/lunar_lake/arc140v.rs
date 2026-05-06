//! Intel Arc 140V visibility probe.
#![allow(clippy::doc_markdown)]

use serde::{Deserialize, Serialize};

use crate::runtimes::{LevelZeroProbe, OpenClRuntimeProbe, OpenVinoProbe};

use super::platform::{PlatformMemoryProbe, PlatformPowerProbe};

const ARC_140V_PCI_DEVICE_ID: &str = "0x64A0";

/// Visibility facts for the Lunar Lake integrated Arc 140V GPU.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct IntelArc140vProbe {
    /// Whether the probe found evidence that the Arc 140V is visible.
    pub available: bool,
    /// Expected PCI device ID when it can be resolved.
    pub pci_device_id: Option<String>,
    /// Whether OpenCL runtime visibility includes Arc 140V.
    pub opencl_available: bool,
    /// OpenCL platform name when available.
    pub opencl_platform_name: Option<String>,
    /// OpenCL device name when available.
    pub opencl_device_name: Option<String>,
    /// OpenCL driver version when available.
    pub opencl_driver_version: Option<String>,
    /// Whether Level Zero tooling sees a matching device.
    pub level_zero_available: bool,
    /// Level Zero device lines or names.
    pub level_zero_devices: Vec<String>,
    /// Whether OpenVINO exposes a GPU device on this platform.
    pub openvino_gpu_visible: bool,
    /// OpenVINO GPU token, usually `GPU.0` on an iGPU-only Lunar Lake system.
    pub openvino_gpu_device: Option<String>,
    /// OpenVINO GPU full device name when available.
    pub openvino_gpu_full_name: Option<String>,
    /// Shared-memory size context when available.
    pub shared_memory_bytes: Option<u64>,
    /// Power mode context when available.
    pub power_mode: Option<String>,
    /// Non-fatal reason explaining why exact Arc 140V identity was not found.
    pub failure_reason: Option<String>,
}

/// Build an Arc 140V visibility result from runtime probes.
pub fn probe_intel_arc_140v(
    opencl: &OpenClRuntimeProbe,
    level_zero: &LevelZeroProbe,
    openvino: &OpenVinoProbe,
    memory: &PlatformMemoryProbe,
    power: &PlatformPowerProbe,
) -> IntelArc140vProbe {
    let opencl_device =
        opencl.devices.iter().find(|device| name_matches_arc_140v(&device.device_name));

    let level_zero_devices: Vec<String> =
        level_zero.devices.iter().filter(|device| name_matches_arc_140v(device)).cloned().collect();

    let openvino_gpu_device = openvino.gpu_device_token();
    let openvino_gpu_full_name =
        openvino_gpu_device.as_deref().and_then(|token| openvino.full_name_for(token));
    let openvino_gpu_visible = openvino_gpu_device.is_some();
    let openvino_name_matches =
        openvino_gpu_full_name.as_deref().is_some_and(name_matches_arc_140v);

    let available =
        opencl_device.is_some() || !level_zero_devices.is_empty() || openvino_name_matches;
    let failure_reason = if available {
        None
    } else {
        Some("Arc 140V identity was not visible through OpenCL, Level Zero, or OpenVINO".to_owned())
    };

    IntelArc140vProbe {
        available,
        pci_device_id: available.then_some(ARC_140V_PCI_DEVICE_ID.to_owned()),
        opencl_available: opencl_device.is_some(),
        opencl_platform_name: opencl_device.and_then(|device| device.platform_name.clone()),
        opencl_device_name: opencl_device.map(|device| device.device_name.clone()),
        opencl_driver_version: opencl_device.and_then(|device| device.driver_version.clone()),
        level_zero_available: !level_zero_devices.is_empty(),
        level_zero_devices,
        openvino_gpu_visible,
        openvino_gpu_device,
        openvino_gpu_full_name,
        shared_memory_bytes: memory.shared_memory_bytes,
        power_mode: power.mode.clone(),
        failure_reason,
    }
}

fn name_matches_arc_140v(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    (lower.contains("arc") && lower.contains("140v")) || lower.contains("64a0")
}
