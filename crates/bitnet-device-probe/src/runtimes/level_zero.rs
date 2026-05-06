//! Level Zero runtime visibility probing through installed command-line tools.

use serde::{Deserialize, Serialize};

use super::command_output;

/// Level Zero runtime visibility result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LevelZeroProbe {
    /// Whether Level Zero tooling was visible.
    pub runtime_available: bool,
    /// Best-effort device names parsed from `ze_info` or `sycl-ls`.
    pub devices: Vec<String>,
    /// Non-fatal probe error when the runtime tooling was absent or unusable.
    pub error: Option<String>,
}

impl LevelZeroProbe {
    /// Build an unavailable Level Zero probe result.
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self { runtime_available: false, devices: Vec::new(), error: Some(reason.into()) }
    }
}

/// Probe Level Zero visibility without compiling or dispatching kernels.
pub fn probe_level_zero() -> LevelZeroProbe {
    match command_output("ze_info", std::iter::empty::<&str>()) {
        Ok(stdout) => {
            let devices = parse_ze_info_devices(&stdout);
            LevelZeroProbe { runtime_available: true, devices, error: None }
        }
        Err(ze_error) => match command_output("sycl-ls", std::iter::empty::<&str>()) {
            Ok(stdout) => {
                let devices = parse_sycl_ls_level_zero_devices(&stdout);
                LevelZeroProbe { runtime_available: !devices.is_empty(), devices, error: None }
            }
            Err(sycl_error) => LevelZeroProbe::unavailable(format!("{ze_error}; {sycl_error}")),
        },
    }
}

pub(crate) fn parse_ze_info_devices(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            trimmed
                .strip_prefix("Device Name")
                .and_then(|rest| rest.split_once(':').map(|(_, value)| value.trim().to_owned()))
                .filter(|value| !value.is_empty())
        })
        .collect()
}

pub(crate) fn parse_sycl_ls_level_zero_devices(output: &str) -> Vec<String> {
    output
        .lines()
        .filter(|line| line.to_ascii_lowercase().contains("level_zero"))
        .map(|line| line.trim().to_owned())
        .filter(|line| !line.is_empty())
        .collect()
}
