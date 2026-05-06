//! OpenVINO runtime visibility probing without linking OpenVINO.
#![allow(clippy::doc_markdown)]

use serde::{Deserialize, Serialize};

use super::command_output;

/// OpenVINO property value captured without compiling a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenVinoPropertyProbe {
    /// OpenVINO property name.
    pub name: String,
    /// Stringified property value reported by OpenVINO.
    pub value: String,
}

/// OpenVINO device facts used in platform receipts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenVinoDeviceProbe {
    /// OpenVINO device token, such as `CPU`, `GPU.0`, or `NPU`.
    pub device: String,
    /// `FULL_DEVICE_NAME` when OpenVINO reports it.
    pub full_name: Option<String>,
    /// Supported property names when collected.
    pub supported_properties: Vec<String>,
    /// Selected property values collected for runtime identity receipts.
    pub properties: Vec<OpenVinoPropertyProbe>,
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
        self.device_for(token).and_then(|device| device.full_name.clone())
    }

    /// Return per-device facts for an OpenVINO device token.
    pub fn device_for(&self, token: &str) -> Option<&OpenVinoDeviceProbe> {
        self.devices.iter().find(|device| device.device == token)
    }

    /// Return whether OpenVINO exposes an `NPU` device.
    pub fn npu_visible(&self) -> bool {
        self.available_devices.iter().any(|device| device == "NPU" || device.starts_with("NPU."))
    }

    /// Return the first OpenVINO NPU token, preferring the unindexed `NPU` alias.
    pub fn npu_device_token(&self) -> Option<String> {
        self.available_devices
            .iter()
            .find(|device| device.as_str() == "NPU")
            .or_else(|| self.available_devices.iter().find(|device| device.starts_with("NPU.")))
            .cloned()
    }

    /// Return a stringified OpenVINO property value for a device.
    pub fn property_value_for(&self, token: &str, property: &str) -> Option<String> {
        self.device_for(token).and_then(|device| {
            device
                .properties
                .iter()
                .find(|entry| entry.name == property)
                .map(|entry| entry.value.clone())
        })
    }
}

/// Probe OpenVINO runtime visibility through Python, without compiling a model.
pub fn probe_openvino() -> OpenVinoProbe {
    let script = r#"
import openvino as ov

core = ov.Core()
print("OPENVINO_VERSION=" + str(ov.__version__))
common_props = [
    "FULL_DEVICE_NAME",
    "SUPPORTED_PROPERTIES",
]
npu_props = [
    "NPU_DRIVER_VERSION",
    "NPU_COMPILER_VERSION",
    "NPU_DEVICE_TOTAL_MEM_SIZE",
    "NPU_DEVICE_ALLOC_MEM_SIZE",
    "NPU_MAX_TILES",
]
for dev in core.available_devices:
    print("DEVICE=" + str(dev))
    props = list(common_props)
    if str(dev) == "NPU" or str(dev).startswith("NPU."):
        props.extend(npu_props)
    for prop in props:
        try:
            print("PROP=" + str(dev) + "=" + prop + "=" + str(core.get_property(dev, prop)))
        except Exception as exc:
            print("PROP_ERR=" + str(dev) + "=" + prop + "=" + repr(exc))
"#;

    for python in ["python3", "python"] {
        if let Ok(stdout) = command_output(python, ["-c", script]) {
            return parse_openvino_line_output(&stdout);
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
                    properties: Vec::new(),
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

    let idx =
        if let Some(idx) = probe.devices.iter().position(|device| device.device == device_token) {
            idx
        } else {
            probe.devices.push(OpenVinoDeviceProbe {
                device: device_token.to_owned(),
                full_name: None,
                supported_properties: Vec::new(),
                properties: Vec::new(),
            });
            probe.devices.len() - 1
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
        _ => upsert_property_value(&mut probe.devices[idx], property, value),
    }
}

fn upsert_property_value(device: &mut OpenVinoDeviceProbe, property: &str, value: &str) {
    let value = value.trim();
    if value.is_empty() {
        return;
    }

    if let Some(entry) = device.properties.iter_mut().find(|entry| entry.name == property) {
        value.clone_into(&mut entry.value);
    } else {
        device
            .properties
            .push(OpenVinoPropertyProbe { name: property.to_owned(), value: value.to_owned() });
    }
}

#[cfg(test)]
mod tests {
    use super::parse_openvino_line_output;

    #[test]
    fn parses_openvino_npu_runtime_properties() {
        let output = r"
OPENVINO_VERSION=2026.1
DEVICE=CPU
PROP=CPU=FULL_DEVICE_NAME=Intel CPU
DEVICE=NPU
PROP=NPU=FULL_DEVICE_NAME=Intel(R) AI Boost
PROP=NPU=SUPPORTED_PROPERTIES=['FULL_DEVICE_NAME', 'NPU_DRIVER_VERSION']
PROP=NPU=NPU_DRIVER_VERSION=1.2.3
PROP=NPU=NPU_COMPILER_VERSION=4.5.6
PROP=NPU=NPU_DEVICE_TOTAL_MEM_SIZE=123456
PROP=NPU=NPU_DEVICE_ALLOC_MEM_SIZE=65432
PROP=NPU=NPU_MAX_TILES=2
";

        let probe = parse_openvino_line_output(output);

        assert!(probe.runtime_available);
        assert_eq!(probe.version.as_deref(), Some("2026.1"));
        assert_eq!(probe.npu_device_token().as_deref(), Some("NPU"));
        assert_eq!(probe.full_name_for("NPU").as_deref(), Some("Intel(R) AI Boost"));
        assert_eq!(probe.property_value_for("NPU", "NPU_DRIVER_VERSION").as_deref(), Some("1.2.3"));
        assert_eq!(
            probe.property_value_for("NPU", "NPU_DEVICE_TOTAL_MEM_SIZE").as_deref(),
            Some("123456")
        );
    }
}
