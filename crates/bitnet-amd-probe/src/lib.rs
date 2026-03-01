//! AMD runtime probing helpers.
//!
//! This crate isolates ROCm/AMD driver detection from backend-specific crates.
//! It can be reused by OpenCL, ROCm, and dispatch crates without coupling them
//! to command execution details.

/// Result of AMD runtime probing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdDriverStatus {
    /// Whether an AMD runtime/driver was detected.
    pub found: bool,
    /// Human-readable description of what was detected.
    pub description: String,
}

impl AmdDriverStatus {
    /// Status representing no AMD runtime being found.
    #[must_use]
    pub fn not_found() -> Self {
        Self { found: false, description: "not found".to_owned() }
    }
}

/// Checks whether ROCm runtime tooling appears available.
///
/// Detection order:
/// 1. Optional test override (`BITNET_AMD_FAKE=1|true` or `0|false`)
/// 2. `rocm-smi --showid`
/// 3. `rocminfo`
#[must_use]
pub fn rocm_runtime_available() -> bool {
    if let Some(v) = fake_probe_override() {
        return v;
    }

    command_ok("rocm-smi", &["--showid"]) || command_ok("rocminfo", &[])
}

/// Probe for AMD runtime and return a display-ready status.
#[must_use]
pub fn detect_amd_driver() -> AmdDriverStatus {
    if let Some(v) = fake_probe_override() {
        return if v {
            AmdDriverStatus {
                found: true,
                description: "AMD ROCm driver detected (BITNET_AMD_FAKE)".to_owned(),
            }
        } else {
            AmdDriverStatus::not_found()
        };
    }

    if command_ok("rocm-smi", &["--showid"]) {
        return AmdDriverStatus {
            found: true,
            description: "AMD ROCm driver detected (rocm-smi)".to_owned(),
        };
    }

    if command_ok("rocminfo", &[]) {
        return AmdDriverStatus {
            found: true,
            description: "AMD ROCm runtime detected (rocminfo)".to_owned(),
        };
    }

    AmdDriverStatus::not_found()
}

fn fake_probe_override() -> Option<bool> {
    let raw = std::env::var("BITNET_AMD_FAKE").ok()?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" | "none" => Some(false),
        _ => None,
    }
}

fn command_ok(cmd: &str, args: &[&str]) -> bool {
    std::process::Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
