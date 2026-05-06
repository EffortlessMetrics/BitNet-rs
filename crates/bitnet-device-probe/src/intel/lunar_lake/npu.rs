//! Intel AI Boost NPU visibility probe for Lunar Lake.
#![allow(clippy::doc_markdown)]

use serde::{Deserialize, Serialize};

use crate::runtimes::OpenVinoProbe;

/// Visibility facts for the Intel AI Boost NPU lane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct IntelNpuProbe {
    /// Whether any OS or OpenVINO NPU evidence was visible.
    pub available: bool,
    /// Whether Linux `/dev/accel` devices are present.
    pub accel_device_present: bool,
    /// Linux `/dev/accel/*` entries when available.
    pub accel_devices: Vec<String>,
    /// Whether kernel logs or OS device evidence mention Intel VPU/NPU.
    pub intel_vpu_driver_seen: bool,
    /// Driver hint parsed from local evidence when available.
    pub driver_hint: Option<String>,
    /// Whether OpenVINO runtime itself was visible.
    pub openvino_runtime_available: bool,
    /// OpenVINO version when available.
    pub openvino_version: Option<String>,
    /// Whether OpenVINO exposes an NPU device token.
    pub openvino_npu_visible: bool,
    /// OpenVINO NPU full name when available.
    pub openvino_npu_full_name: Option<String>,
    /// OpenVINO supported property names for NPU when available.
    pub supported_properties: Vec<String>,
    /// NPU driver version when available.
    pub driver_version: Option<String>,
    /// NPU compiler version when available.
    pub compiler_version: Option<String>,
    /// NPU total memory size when available.
    pub total_mem_size: Option<u64>,
    /// Non-fatal reason explaining unavailable state.
    pub failure_reason: Option<String>,
}

/// Probe Intel NPU visibility without compiling an OpenVINO graph.
pub fn probe_intel_npu(openvino: &OpenVinoProbe) -> IntelNpuProbe {
    let accel_devices = accel_devices();
    let accel_device_present = !accel_devices.is_empty();
    let openvino_npu_visible = openvino.npu_visible();
    let npu_device = openvino
        .available_devices
        .iter()
        .find(|device| device.as_str() == "NPU" || device.starts_with("NPU."))
        .cloned();
    let openvino_npu_full_name =
        npu_device.as_deref().and_then(|token| openvino.full_name_for(token));
    let supported_properties = npu_device
        .as_deref()
        .and_then(|token| {
            openvino
                .devices
                .iter()
                .find(|device| device.device == token)
                .map(|device| device.supported_properties.clone())
        })
        .unwrap_or_default();
    let intel_vpu_driver_seen = intel_vpu_driver_seen();
    let available = accel_device_present || openvino_npu_visible || intel_vpu_driver_seen;
    let failure_reason = if available {
        None
    } else {
        Some("Intel NPU was not visible through OS accelerator devices or OpenVINO".to_owned())
    };

    IntelNpuProbe {
        available,
        accel_device_present,
        accel_devices,
        intel_vpu_driver_seen,
        driver_hint: intel_vpu_driver_seen.then_some("intel_vpu/ivpu evidence".to_owned()),
        openvino_runtime_available: openvino.runtime_available,
        openvino_version: openvino.version.clone(),
        openvino_npu_visible,
        openvino_npu_full_name,
        supported_properties,
        driver_version: None,
        compiler_version: None,
        total_mem_size: None,
        failure_reason,
    }
}

#[allow(clippy::missing_const_for_fn)]
fn accel_devices() -> Vec<String> {
    #[cfg(target_os = "linux")]
    {
        return std::fs::read_dir("/dev/accel")
            .map(|entries| {
                entries
                    .flatten()
                    .map(|entry| entry.path().display().to_string())
                    .filter(|path| path.contains("accel"))
                    .collect()
            })
            .unwrap_or_default();
    }

    #[cfg(not(target_os = "linux"))]
    {
        Vec::new()
    }
}

fn intel_vpu_driver_seen() -> bool {
    #[cfg(target_os = "linux")]
    {
        return std::fs::read_to_string("/sys/bus/pci/drivers/intel_vpu/module/drivers")
            .map(|content| content.to_ascii_lowercase().contains("vpu"))
            .unwrap_or(false);
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "Get-PnpDevice | Where-Object { $_.FriendlyName -match 'NPU|AI Boost|VPU' }",
            ])
            .output()
            .map(|output| output.status.success() && !output.stdout.is_empty())
            .unwrap_or(false)
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        false
    }
}
