//! Intel Arc 140V visibility probe.
#![allow(clippy::doc_markdown)]

use serde::{Deserialize, Serialize};

use crate::runtimes::{LevelZeroProbe, OpenClRuntimeProbe, OpenVinoProbe};

use super::platform::{PlatformMemoryProbe, PlatformPowerProbe};

/// Stable requested backend label for the Lunar Lake integrated GPU lane.
pub const INTEL_ARC_140V_REQUESTED_BACKEND: &str = "intel-arc-140v";
/// Native OpenCL proof label for Arc 140V.
pub const INTEL_ARC_140V_OPENCL_BACKEND: &str = "intel-arc-140v-opencl";
/// OpenVINO GPU reference proof label for Arc 140V.
pub const INTEL_ARC_140V_OPENVINO_GPU_BACKEND: &str = "intel-arc-140v-openvino-gpu";
/// Expected PCI device ID for Arc 140V.
pub const INTEL_ARC_140V_PCI_DEVICE_ID: &str = "0x64A0";
/// Universal proof stage for this probe.
pub const INTEL_ARC_140V_PROOF_STAGE_RUNTIME_DETECTED: &str = "runtime_detected";

/// Visibility facts for the Lunar Lake integrated Arc 140V GPU.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct IntelArc140vProbe {
    /// Universal proof stage for this visibility-only lane probe.
    pub proof_stage: String,
    /// Requested backend identity for receipts.
    pub requested_backend: String,
    /// Selected backend when an exact runtime identity is visible.
    pub selected_backend: Option<String>,
    /// Runtime API associated with the selected backend.
    pub runtime_api: Option<String>,
    /// Whether the probe found evidence that the Arc 140V is visible.
    pub available: bool,
    /// Expected PCI device ID when it can be resolved.
    pub pci_device_id: Option<String>,
    /// Identity evidence that matched Arc 140V by name or PCI ID.
    pub identity_evidence: Vec<String>,
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
    /// Always false: this probe never substitutes CPU or another GPU as Arc 140V.
    pub fallback_used: bool,
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
    let level_zero_pci_match =
        level_zero.device_ids.iter().any(|device_id| device_id_matches_arc_140v(device_id));

    let openvino_gpu_device = openvino.gpu_device_token();
    let openvino_gpu_full_name =
        openvino_gpu_device.as_deref().and_then(|token| openvino.full_name_for(token));
    let openvino_gpu_visible = openvino_gpu_device.is_some();
    let openvino_name_matches =
        openvino_gpu_full_name.as_deref().is_some_and(name_matches_arc_140v);

    let mut identity_evidence = Vec::new();
    if let Some(device) = opencl_device {
        identity_evidence.push(format!("opencl:{}", device.device_name));
    }
    identity_evidence
        .extend(level_zero_devices.iter().map(|device| format!("level_zero:{device}")));
    if level_zero_pci_match {
        identity_evidence.push(format!("level_zero_pci_device_id:{INTEL_ARC_140V_PCI_DEVICE_ID}"));
    }
    if let (Some(token), Some(full_name)) = (&openvino_gpu_device, &openvino_gpu_full_name)
        && openvino_name_matches
    {
        identity_evidence.push(format!("openvino:{token}:{full_name}"));
    }

    let opencl_available = opencl_device.is_some();
    let level_zero_available = !level_zero_devices.is_empty() || level_zero_pci_match;
    let available = opencl_available || level_zero_available || openvino_name_matches;
    let selected_backend = selected_backend(opencl_available, openvino_name_matches);
    let runtime_api =
        selected_runtime_api(opencl_available, level_zero_available, openvino_name_matches);
    let failure_reason = if available {
        None
    } else {
        Some("Arc 140V identity was not visible through OpenCL, Level Zero, or OpenVINO".to_owned())
    };

    IntelArc140vProbe {
        proof_stage: INTEL_ARC_140V_PROOF_STAGE_RUNTIME_DETECTED.to_owned(),
        requested_backend: INTEL_ARC_140V_REQUESTED_BACKEND.to_owned(),
        selected_backend,
        runtime_api,
        available,
        pci_device_id: available.then_some(INTEL_ARC_140V_PCI_DEVICE_ID.to_owned()),
        identity_evidence,
        opencl_available,
        opencl_platform_name: opencl_device.and_then(|device| device.platform_name.clone()),
        opencl_device_name: opencl_device.map(|device| device.device_name.clone()),
        opencl_driver_version: opencl_device.and_then(|device| device.driver_version.clone()),
        level_zero_available,
        level_zero_devices,
        openvino_gpu_visible,
        openvino_gpu_device,
        openvino_gpu_full_name,
        shared_memory_bytes: memory.shared_memory_bytes,
        power_mode: power.mode.clone(),
        fallback_used: false,
        failure_reason,
    }
}

fn name_matches_arc_140v(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    (lower.contains("arc") && lower.contains("140v")) || lower.contains("64a0")
}

fn device_id_matches_arc_140v(value: &str) -> bool {
    let normalized = value.trim().to_ascii_uppercase();
    normalized == INTEL_ARC_140V_PCI_DEVICE_ID
        || normalized.trim_start_matches("0X") == "64A0"
        || normalized.contains("64A0")
}

fn selected_backend(opencl_available: bool, openvino_name_matches: bool) -> Option<String> {
    if opencl_available {
        Some(INTEL_ARC_140V_OPENCL_BACKEND.to_owned())
    } else if openvino_name_matches {
        Some(INTEL_ARC_140V_OPENVINO_GPU_BACKEND.to_owned())
    } else {
        None
    }
}

fn selected_runtime_api(
    opencl_available: bool,
    level_zero_available: bool,
    openvino_name_matches: bool,
) -> Option<String> {
    if opencl_available {
        Some("opencl".to_owned())
    } else if openvino_name_matches {
        Some("openvino".to_owned())
    } else if level_zero_available {
        Some("level_zero".to_owned())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtimes::{
        LevelZeroProbe, OpenClRuntimeDevice, OpenClRuntimeProbe, OpenVinoDeviceProbe, OpenVinoProbe,
    };

    fn memory() -> PlatformMemoryProbe {
        PlatformMemoryProbe {
            total_bytes: Some(32 * 1024 * 1024 * 1024),
            shared_memory_bytes: Some(32 * 1024 * 1024 * 1024),
            shared_memory: true,
        }
    }

    fn power() -> PlatformPowerProbe {
        PlatformPowerProbe {
            mode: Some("balanced".to_owned()),
            thermal_profile: None,
            ac_power: Some(true),
        }
    }

    #[test]
    fn opencl_arc_140v_identity_selects_native_lane_without_fallback() {
        let opencl = OpenClRuntimeProbe {
            runtime_available: true,
            devices: vec![OpenClRuntimeDevice {
                platform_name: Some("Intel(R) OpenCL Graphics".to_owned()),
                device_name: "Intel(R) Arc(TM) 140V Graphics".to_owned(),
                vendor: "Intel(R) Corporation".to_owned(),
                driver_version: Some("test-driver".to_owned()),
                is_gpu: true,
            }],
            error: None,
        };
        let level_zero = LevelZeroProbe::unavailable("not installed");
        let openvino = OpenVinoProbe::unavailable("not installed");

        let probe = probe_intel_arc_140v(&opencl, &level_zero, &openvino, &memory(), &power());

        assert!(probe.available);
        assert!(probe.opencl_available);
        assert_eq!(probe.selected_backend.as_deref(), Some(INTEL_ARC_140V_OPENCL_BACKEND));
        assert_eq!(probe.runtime_api.as_deref(), Some("opencl"));
        assert_eq!(probe.pci_device_id.as_deref(), Some(INTEL_ARC_140V_PCI_DEVICE_ID));
        assert!(!probe.fallback_used);
        assert!(probe.identity_evidence.iter().any(|entry| entry.starts_with("opencl:")));
    }

    #[test]
    fn level_zero_pci_id_is_sufficient_for_runtime_detected_identity() {
        let opencl = OpenClRuntimeProbe::unavailable("not installed");
        let level_zero = LevelZeroProbe {
            runtime_available: true,
            devices: Vec::new(),
            device_ids: vec!["0x64A0".to_owned()],
            error: None,
        };
        let openvino = OpenVinoProbe::unavailable("not installed");

        let probe = probe_intel_arc_140v(&opencl, &level_zero, &openvino, &memory(), &power());

        assert!(probe.available);
        assert!(probe.level_zero_available);
        assert_eq!(probe.selected_backend, None);
        assert_eq!(probe.runtime_api.as_deref(), Some("level_zero"));
        assert!(probe.identity_evidence.iter().any(|entry| entry.contains("64A0")));
    }

    #[test]
    fn openvino_gpu_full_name_selects_reference_lane() {
        let opencl = OpenClRuntimeProbe::unavailable("not installed");
        let level_zero = LevelZeroProbe::unavailable("not installed");
        let openvino = OpenVinoProbe {
            runtime_available: true,
            version: Some("2026.1".to_owned()),
            available_devices: vec!["CPU".to_owned(), "GPU.0".to_owned()],
            devices: vec![OpenVinoDeviceProbe {
                device: "GPU.0".to_owned(),
                full_name: Some("Intel(R) Arc(TM) 140V Graphics".to_owned()),
                supported_properties: Vec::new(),
            }],
            error: None,
        };

        let probe = probe_intel_arc_140v(&opencl, &level_zero, &openvino, &memory(), &power());

        assert!(probe.available);
        assert!(probe.openvino_gpu_visible);
        assert_eq!(probe.selected_backend.as_deref(), Some(INTEL_ARC_140V_OPENVINO_GPU_BACKEND));
        assert_eq!(probe.runtime_api.as_deref(), Some("openvino"));
        assert!(probe.identity_evidence.iter().any(|entry| entry.starts_with("openvino:")));
    }

    #[test]
    fn generic_gpu_visibility_is_not_arc_140v_identity() {
        let opencl = OpenClRuntimeProbe {
            runtime_available: true,
            devices: vec![OpenClRuntimeDevice {
                platform_name: Some("Intel(R) OpenCL Graphics".to_owned()),
                device_name: "Intel(R) UHD Graphics".to_owned(),
                vendor: "Intel(R) Corporation".to_owned(),
                driver_version: Some("test-driver".to_owned()),
                is_gpu: true,
            }],
            error: None,
        };
        let level_zero = LevelZeroProbe {
            runtime_available: true,
            devices: vec!["Intel(R) UHD Graphics".to_owned()],
            device_ids: vec!["0x5917".to_owned()],
            error: None,
        };
        let openvino = OpenVinoProbe {
            runtime_available: true,
            version: Some("2026.1".to_owned()),
            available_devices: vec!["GPU.0".to_owned()],
            devices: vec![OpenVinoDeviceProbe {
                device: "GPU.0".to_owned(),
                full_name: Some("Intel(R) UHD Graphics 620".to_owned()),
                supported_properties: Vec::new(),
            }],
            error: None,
        };

        let probe = probe_intel_arc_140v(&opencl, &level_zero, &openvino, &memory(), &power());

        assert!(!probe.available);
        assert!(!probe.opencl_available);
        assert!(!probe.level_zero_available);
        assert!(probe.openvino_gpu_visible);
        assert_eq!(probe.selected_backend, None);
        assert_eq!(probe.runtime_api, None);
        assert!(probe.failure_reason.is_some());
    }
}
