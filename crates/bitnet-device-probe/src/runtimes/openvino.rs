//! OpenVINO runtime visibility probing without linking OpenVINO.

use serde::{Deserialize, Serialize};

use super::command_output;

/// OpenVINO device facts used in platform receipts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenVinoDeviceProbe {
    /// OpenVINO device token, such as `CPU`, `GPU.0`, or `NPU`.
    pub device: String,
    /// `FULL_DEVICE_NAME` when OpenVINO reports it.
    pub full_name: Option<String>,
    /// Supported property names when collected.
    pub supported_properties: Vec<String>,
}

/// OpenVINO runtime visibility result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenVinoProbe {
    /// Whether Python could import OpenVINO and instantiate `ov.Core`.
    pub runtime_available: bool,
    /// OpenVINO Python package version.
    pub version: Option<String>,
    /// Available OpenVINO device tokens.
    pub available_devices: Vec<String>,
    /// Per-device properties collected without compiling a model.
    pub devices: Vec<OpenVinoDeviceProbe>,
    /// Non-fatal probe error when OpenVINO was absent or unusable.
    pub error: Option<String>,
}

impl OpenVinoProbe {
    /// Build an unavailable OpenVINO probe result.
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            runtime_available: false,
            version: None,
            available_devices: Vec::new(),
            devices: Vec::new(),
            error: Some(reason.into()),
        }
    }

    /// Return the first OpenVINO GPU token, preferring `GPU.0`.
    pub fn gpu_device_token(&self) -> Option<String> {
        self.available_devices
            .iter()
            .find(|device| device.as_str() == "GPU.0")
            .or_else(|| self.available_devices.iter().find(|device| device.starts_with("GPU")))
            .cloned()
    }

    /// Return the full name for an OpenVINO device token.
    pub fn full_name_for(&self, token: &str) -> Option<String> {
        self.devices
            .iter()
            .find(|device| device.device == token)
            .and_then(|device| device.full_name.clone())
    }

    /// Return whether OpenVINO exposes an `NPU` device.
    pub fn npu_visible(&self) -> bool {
        self.available_devices.iter().any(|device| device == "NPU" || device.starts_with("NPU."))
    }
}

/// Probe OpenVINO runtime visibility through Python, without compiling a model.
pub fn probe_openvino() -> OpenVinoProbe {
    let script = r#"
import openvino as ov

core = ov.Core()
print("OPENVINO_VERSION=" + str(ov.__version__))
for dev in core.available_devices:
    print("DEVICE=" + str(dev))
    for prop in ["FULL_DEVICE_NAME", "SUPPORTED_PROPERTIES"]:
        try:
            print("PROP=" + str(dev) + "=" + prop + "=" + str(core.get_property(dev, prop)))
        except Exception as exc:
            print("PROP_ERR=" + str(dev) + "=" + prop + "=" + repr(exc))
"#;

    for python in ["python3", "python"] {
        match command_output(python, ["-c", script]) {
            Ok(stdout) => return parse_openvino_line_output(&stdout),
            Err(_) => continue,
        }
    }

    OpenVinoProbe::unavailable("python openvino import unavailable")
}

pub(crate) fn parse_openvino_line_output(output: &str) -> OpenVinoProbe {
    let mut probe = OpenVinoProbe {
        runtime_available: true,
        version: None,
        available_devices: Vec::new(),
        devices: Vec::new(),
        error: None,
    };

    for line in output.lines().map(str::trim).filter(|line| !line.is_empty()) {
        if let Some(value) = line.strip_prefix("OPENVINO_VERSION=") {
            probe.version = Some(value.to_owned());
        } else if let Some(value) = line.strip_prefix("DEVICE=") {
            if !probe.available_devices.iter().any(|device| device == value) {
                probe.available_devices.push(value.to_owned());
                probe.devices.push(OpenVinoDeviceProbe {
                    device: value.to_owned(),
                    full_name: None,
                    supported_properties: Vec::new(),
                });
            }
        } else if let Some(rest) = line.strip_prefix("PROP=") {
            apply_property_line(&mut probe, rest);
        }
    }

    probe
}

fn apply_property_line(probe: &mut OpenVinoProbe, rest: &str) {
    let mut parts = rest.splitn(3, '=');
    let Some(device_token) = parts.next() else {
        return;
    };
    let Some(property) = parts.next() else {
        return;
    };
    let Some(value) = parts.next() else {
        return;
    };

    if !probe.available_devices.iter().any(|device| device == device_token) {
        probe.available_devices.push(device_token.to_owned());
    }

    let idx = match probe.devices.iter().position(|device| device.device == device_token) {
        Some(idx) => idx,
        None => {
            probe.devices.push(OpenVinoDeviceProbe {
                device: device_token.to_owned(),
                full_name: None,
                supported_properties: Vec::new(),
            });
            probe.devices.len() - 1
        }
    };

    match property {
        "FULL_DEVICE_NAME" if !value.trim().is_empty() => {
            probe.devices[idx].full_name = Some(value.trim().to_owned());
        }
        "SUPPORTED_PROPERTIES" => {
            probe.devices[idx].supported_properties = value
                .trim_matches(['[', ']'])
                .split(',')
                .map(|entry| entry.trim().trim_matches(['\'', '"']).to_owned())
                .filter(|entry| !entry.is_empty())
                .collect();
        }
        _ => {}
    }
}
