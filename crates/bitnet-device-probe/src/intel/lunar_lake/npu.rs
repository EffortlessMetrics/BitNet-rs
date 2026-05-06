//! Intel AI Boost NPU visibility probe for Lunar Lake.
#![allow(clippy::doc_markdown)]

use serde::{Deserialize, Serialize};

use crate::runtimes::OpenVinoProbe;

/// Requested backend label for the Intel NPU lane.
pub const INTEL_NPU_REQUESTED_BACKEND: &str = "intel-npu";
/// Selected backend label once OpenVINO reports an NPU runtime device.
pub const INTEL_NPU_OPENVINO_BACKEND: &str = "intel-npu-openvino";
/// Runtime API used by the Intel NPU lane.
pub const INTEL_NPU_RUNTIME_API_OPENVINO: &str = "openvino";
/// Proof stage emitted by runtime visibility probes.
pub const INTEL_NPU_PROOF_STAGE_RUNTIME_DETECTED: &str = "runtime_detected";

/// Visibility facts for the Intel AI Boost NPU lane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct IntelNpuProbe {
    /// Universal proof stage for this visibility-only probe.
    pub proof_stage: String,
    /// Requested backend identity preserved before runtime selection.
    pub requested_backend: String,
    /// Selected backend identity when OpenVINO reports an NPU device.
    pub selected_backend: Option<String>,
    /// Runtime API used by the selected backend.
    pub runtime_api: Option<String>,
    /// Concrete OpenVINO runtime device token, normally `NPU`.
    pub runtime_device: Option<String>,
    /// OS family.
    pub os: String,
    /// CPU architecture.
    pub arch: String,
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
    /// OpenVINO available device tokens.
    pub openvino_available_devices: Vec<String>,
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
    /// NPU allocated memory size when available.
    pub alloc_mem_size: Option<u64>,
    /// OpenVINO NPU tile count when available.
    pub max_tiles: Option<u32>,
    /// Always false: this probe never substitutes CPU/GPU fallback.
    pub fallback_used: bool,
    /// Non-fatal reason explaining unavailable state.
    pub failure_reason: Option<String>,
}

/// Probe Intel NPU visibility without compiling an OpenVINO graph.
pub fn probe_intel_npu(openvino: &OpenVinoProbe) -> IntelNpuProbe {
    let accel_devices = accel_devices();
    let accel_device_present = !accel_devices.is_empty();
    let npu_device = openvino.npu_device_token();
    let openvino_npu_visible = npu_device.is_some();
    let openvino_npu_full_name =
        npu_device.as_deref().and_then(|token| openvino.full_name_for(token));
    let supported_properties = npu_device
        .as_deref()
        .and_then(|token| {
            openvino.device_for(token).map(|device| device.supported_properties.clone())
        })
        .unwrap_or_default();
    let driver_version = npu_device
        .as_deref()
        .and_then(|token| openvino.property_value_for(token, "NPU_DRIVER_VERSION"));
    let compiler_version = npu_device
        .as_deref()
        .and_then(|token| openvino.property_value_for(token, "NPU_COMPILER_VERSION"));
    let total_mem_size = npu_device
        .as_deref()
        .and_then(|token| openvino.property_value_for(token, "NPU_DEVICE_TOTAL_MEM_SIZE"))
        .and_then(|value| parse_u64_property(&value));
    let alloc_mem_size = npu_device
        .as_deref()
        .and_then(|token| openvino.property_value_for(token, "NPU_DEVICE_ALLOC_MEM_SIZE"))
        .and_then(|value| parse_u64_property(&value));
    let max_tiles = npu_device
        .as_deref()
        .and_then(|token| openvino.property_value_for(token, "NPU_MAX_TILES"))
        .and_then(|value| parse_u32_property(&value));
    let intel_vpu_driver_seen = intel_vpu_driver_seen();
    let available = accel_device_present || openvino_npu_visible || intel_vpu_driver_seen;
    let selected_backend = openvino_npu_visible.then_some(INTEL_NPU_OPENVINO_BACKEND.to_owned());
    let runtime_api = openvino_npu_visible.then_some(INTEL_NPU_RUNTIME_API_OPENVINO.to_owned());
    let failure_reason = if available {
        None
    } else {
        Some("Intel NPU was not visible through OS accelerator devices or OpenVINO".to_owned())
    };

    IntelNpuProbe {
        proof_stage: INTEL_NPU_PROOF_STAGE_RUNTIME_DETECTED.to_owned(),
        requested_backend: INTEL_NPU_REQUESTED_BACKEND.to_owned(),
        selected_backend,
        runtime_api,
        runtime_device: npu_device,
        os: std::env::consts::OS.to_owned(),
        arch: std::env::consts::ARCH.to_owned(),
        available,
        accel_device_present,
        accel_devices,
        intel_vpu_driver_seen,
        driver_hint: intel_vpu_driver_seen.then_some("intel_vpu/ivpu evidence".to_owned()),
        openvino_runtime_available: openvino.runtime_available,
        openvino_version: openvino.version.clone(),
        openvino_available_devices: openvino.available_devices.clone(),
        openvino_npu_visible,
        openvino_npu_full_name,
        supported_properties,
        driver_version,
        compiler_version,
        total_mem_size,
        alloc_mem_size,
        max_tiles,
        fallback_used: false,
        failure_reason,
    }
}

fn parse_u64_property(value: &str) -> Option<u64> {
    value.trim().parse::<u64>().ok().or_else(|| first_ascii_digit_run(value)?.parse().ok())
}

fn parse_u32_property(value: &str) -> Option<u32> {
    value.trim().parse::<u32>().ok().or_else(|| first_ascii_digit_run(value)?.parse().ok())
}

fn first_ascii_digit_run(value: &str) -> Option<&str> {
    let start = value.find(|ch: char| ch.is_ascii_digit())?;
    let len = value[start..].find(|ch: char| !ch.is_ascii_digit()).unwrap_or(value.len() - start);
    Some(&value[start..start + len])
}

#[allow(clippy::missing_const_for_fn)]
fn accel_devices() -> Vec<String> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_dir("/dev/accel")
            .map(|entries| {
                entries
                    .flatten()
                    .map(|entry| entry.path().display().to_string())
                    .filter(|path| path.contains("accel"))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[cfg(not(target_os = "linux"))]
    {
        Vec::new()
    }
}

#[allow(clippy::missing_const_for_fn)]
fn intel_vpu_driver_seen() -> bool {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/sys/bus/pci/drivers/intel_vpu/module/drivers")
            .map(|content| content.to_ascii_lowercase().contains("vpu"))
            .unwrap_or(false)
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

#[cfg(test)]
mod tests {
    use super::{
        INTEL_NPU_OPENVINO_BACKEND, INTEL_NPU_PROOF_STAGE_RUNTIME_DETECTED,
        INTEL_NPU_REQUESTED_BACKEND, INTEL_NPU_RUNTIME_API_OPENVINO, probe_intel_npu,
    };
    use crate::runtimes::{OpenVinoDeviceProbe, OpenVinoProbe, OpenVinoPropertyProbe};

    #[test]
    fn openvino_npu_identity_selects_openvino_backend_without_fallback() {
        let openvino = OpenVinoProbe {
            runtime_available: true,
            version: Some("2026.1".to_owned()),
            available_devices: vec!["CPU".to_owned(), "GPU.0".to_owned(), "NPU".to_owned()],
            devices: vec![OpenVinoDeviceProbe {
                device: "NPU".to_owned(),
                full_name: Some("Intel(R) AI Boost".to_owned()),
                supported_properties: vec![
                    "FULL_DEVICE_NAME".to_owned(),
                    "NPU_DRIVER_VERSION".to_owned(),
                    "NPU_COMPILER_VERSION".to_owned(),
                ],
                properties: vec![
                    OpenVinoPropertyProbe {
                        name: "NPU_DRIVER_VERSION".to_owned(),
                        value: "1.2.3".to_owned(),
                    },
                    OpenVinoPropertyProbe {
                        name: "NPU_COMPILER_VERSION".to_owned(),
                        value: "4.5.6".to_owned(),
                    },
                    OpenVinoPropertyProbe {
                        name: "NPU_DEVICE_TOTAL_MEM_SIZE".to_owned(),
                        value: "1048576".to_owned(),
                    },
                    OpenVinoPropertyProbe {
                        name: "NPU_DEVICE_ALLOC_MEM_SIZE".to_owned(),
                        value: "524288".to_owned(),
                    },
                    OpenVinoPropertyProbe {
                        name: "NPU_MAX_TILES".to_owned(),
                        value: "2".to_owned(),
                    },
                ],
            }],
            error: None,
        };

        let probe = probe_intel_npu(&openvino);

        assert_eq!(probe.proof_stage, INTEL_NPU_PROOF_STAGE_RUNTIME_DETECTED);
        assert_eq!(probe.requested_backend, INTEL_NPU_REQUESTED_BACKEND);
        assert_eq!(probe.selected_backend.as_deref(), Some(INTEL_NPU_OPENVINO_BACKEND));
        assert_eq!(probe.runtime_api.as_deref(), Some(INTEL_NPU_RUNTIME_API_OPENVINO));
        assert_eq!(probe.runtime_device.as_deref(), Some("NPU"));
        assert_eq!(probe.openvino_available_devices, ["CPU", "GPU.0", "NPU"]);
        assert_eq!(probe.openvino_npu_full_name.as_deref(), Some("Intel(R) AI Boost"));
        assert_eq!(probe.driver_version.as_deref(), Some("1.2.3"));
        assert_eq!(probe.compiler_version.as_deref(), Some("4.5.6"));
        assert_eq!(probe.total_mem_size, Some(1_048_576));
        assert_eq!(probe.alloc_mem_size, Some(524_288));
        assert_eq!(probe.max_tiles, Some(2));
        assert!(!probe.fallback_used);
        assert!(probe.failure_reason.is_none());
    }

    #[test]
    fn missing_openvino_npu_does_not_select_backend() {
        let openvino = OpenVinoProbe::unavailable("not installed");
        let probe = probe_intel_npu(&openvino);

        assert_eq!(probe.proof_stage, INTEL_NPU_PROOF_STAGE_RUNTIME_DETECTED);
        assert_eq!(probe.requested_backend, INTEL_NPU_REQUESTED_BACKEND);
        assert_eq!(probe.selected_backend, None);
        assert_eq!(probe.runtime_api, None);
        assert_eq!(probe.runtime_device, None);
        assert!(!probe.openvino_npu_visible);
        assert!(!probe.fallback_used);
    }
}
