//! Apple Metal runtime visibility probing.
//!
//! This module records whether macOS exposes a Metal device and the machine
//! facts needed for later Apple M4 receipts. It does not compile or dispatch a
//! Metal pipeline.

/// Apple M4 native Metal backend label.
pub const APPLE_M4_METAL_BACKEND: &str = "apple-m4-metal";
/// Runtime API recorded by Apple Metal probe receipts.
pub const APPLE_M4_METAL_RUNTIME_API: &str = "metal";
/// Proof stage for a visible Apple M4-family Metal runtime probe.
pub const APPLE_M4_METAL_PROOF_STAGE_DETECTED: &str = "runtime_detected";
/// Proof stage for a Metal runtime probe where Apple M4-family Metal is not visible.
pub const APPLE_M4_METAL_PROOF_STAGE_UNAVAILABLE: &str = "runtime_unavailable";

/// Raw command text used to build an [`AppleMetalProbe`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AppleMetalProbeText {
    /// Host OS name, normally [`std::env::consts::OS`].
    pub host_os: String,
    /// Output of `sw_vers`.
    pub sw_vers: String,
    /// Output of `uname -a`.
    pub uname: String,
    /// Output of `system_profiler SPHardwareDataType`.
    pub hardware: String,
    /// Output of `system_profiler SPDisplaysDataType`.
    pub displays: String,
    /// Output of `system_profiler SPMetalDataType`.
    pub metal: String,
    /// Whether `system_profiler SPMetalDataType` exited successfully.
    pub metal_command_succeeded: bool,
    /// Output of `sysctl hw.memsize`.
    pub memsize: String,
    /// Output of `sysctl kern.hv_vmm_present`.
    pub virtualization: String,
}

/// Apple Metal runtime probe result.
///
/// `metal_visible=true` means macOS reported Metal runtime/device visibility.
/// It is not a Metal execution, `MPSGraph`, Neural Engine, or `BitNet` inference
/// proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppleMetalProbe {
    /// Requested backend identity.
    pub requested_backend: &'static str,
    /// Selected backend identity when Apple M4-family Metal is visible.
    pub selected_backend: Option<&'static str>,
    /// Runtime API identity.
    pub runtime_api: &'static str,
    /// Host operating system.
    pub host_os: String,
    /// macOS product version from `sw_vers`.
    pub macos_version: Option<String>,
    /// macOS build version from `sw_vers`.
    pub macos_build: Option<String>,
    /// Kernel version from `uname -a`.
    pub kernel_version: Option<String>,
    /// Apple chip name, such as `Apple M4` or `Apple M4 Pro`.
    pub chip: Option<String>,
    /// CPU core count reported by `system_profiler SPHardwareDataType`.
    pub cpu_cores: Option<usize>,
    /// GPU core count reported by display or Metal system profiler output.
    pub gpu_cores: Option<usize>,
    /// Unified memory size in bytes from `sysctl hw.memsize`.
    pub unified_memory_bytes: Option<u64>,
    /// Whether the machine uses unified memory.
    pub unified_memory: Option<bool>,
    /// Metal device name or chipset model when visible.
    pub metal_device_name: Option<String>,
    /// GPU family or Metal support string when visible.
    pub gpu_family: Option<String>,
    /// Native macOS, virtualized macOS, non-macOS, or unknown.
    pub native_or_virtualized: Option<String>,
    /// Whether macOS reports a Metal device/runtime.
    pub metal_visible: bool,
    /// Always false for this probe; no fallback execution occurs.
    pub fallback_used: bool,
    /// Runtime proof stage.
    pub proof_stage: &'static str,
}

/// Probe Apple Metal runtime visibility on the current machine.
#[must_use]
pub fn probe_apple_metal() -> AppleMetalProbe {
    let host_os = std::env::consts::OS.to_owned();
    if host_os != "macos" {
        return AppleMetalProbe::from_text(&AppleMetalProbeText {
            host_os,
            ..AppleMetalProbeText::default()
        });
    }

    let metal = command_stdout("system_profiler", &["SPMetalDataType"]);
    let text = AppleMetalProbeText {
        host_os,
        sw_vers: command_stdout("sw_vers", &[]).stdout,
        uname: command_stdout("uname", &["-a"]).stdout,
        hardware: command_stdout("system_profiler", &["SPHardwareDataType"]).stdout,
        displays: command_stdout("system_profiler", &["SPDisplaysDataType"]).stdout,
        metal: metal.stdout,
        metal_command_succeeded: metal.success,
        memsize: command_stdout("sysctl", &["hw.memsize"]).stdout,
        virtualization: command_stdout("sysctl", &["kern.hv_vmm_present"]).stdout,
    };

    AppleMetalProbe::from_text(&text)
}

/// Build a probe result from captured command text.
#[must_use]
pub fn parse_apple_metal_probe(text: &AppleMetalProbeText) -> AppleMetalProbe {
    AppleMetalProbe::from_text(text)
}

/// Return whether Apple Metal was visible at runtime.
///
/// This records runtime visibility only. It does not compile or execute Metal
/// kernels.
#[must_use]
pub fn apple_metal_available_runtime() -> bool {
    probe_apple_metal().metal_visible
}

/// Return the planned M4 Metal probe artifact path for an ISO date.
#[must_use]
pub fn apple_metal_probe_artifact_path(date: &str) -> String {
    format!("ci/hardware/apple-m4-mac-mini/{date}/metal-probe.json")
}

impl AppleMetalProbe {
    fn from_text(text: &AppleMetalProbeText) -> Self {
        let is_macos = text.host_os == "macos";
        let chip = parse_colon_value(&text.hardware, "Chip")
            .or_else(|| parse_colon_value(&text.metal, "Chipset Model"))
            .or_else(|| parse_colon_value(&text.displays, "Chipset Model"));
        let apple_m4_family_chip = chip.as_deref().is_some_and(is_apple_m4_family_chip);
        let metal_visible =
            is_macos && text.metal_command_succeeded && metal_text_reports_visibility(&text.metal);
        let apple_m4_metal_visible = metal_visible && apple_m4_family_chip;
        let unified_memory_bytes =
            parse_colon_value(&text.memsize, "hw.memsize").and_then(|value| {
                value.split_whitespace().next().and_then(|number| number.parse::<u64>().ok())
            });
        let unified_memory = if chip.as_deref().is_some_and(|value| value.starts_with("Apple M")) {
            Some(true)
        } else if is_macos && chip.is_some() {
            Some(false)
        } else {
            None
        };
        let native_or_virtualized = parse_virtualization_state(&text.virtualization, is_macos)
            .or_else(|| {
                if is_macos { Some("unknown".to_owned()) } else { Some("not-macos".to_owned()) }
            });

        Self {
            requested_backend: APPLE_M4_METAL_BACKEND,
            selected_backend: apple_m4_metal_visible.then_some(APPLE_M4_METAL_BACKEND),
            runtime_api: APPLE_M4_METAL_RUNTIME_API,
            host_os: text.host_os.clone(),
            macos_version: parse_colon_value(&text.sw_vers, "ProductVersion"),
            macos_build: parse_colon_value(&text.sw_vers, "BuildVersion"),
            kernel_version: first_nonempty_line(&text.uname),
            chip,
            cpu_cores: parse_colon_value(&text.hardware, "Total Number of Cores")
                .and_then(|value| parse_first_usize(&value)),
            gpu_cores: parse_colon_value(&text.metal, "Total Number of Cores")
                .or_else(|| parse_colon_value(&text.displays, "Total Number of Cores"))
                .and_then(|value| parse_first_usize(&value)),
            unified_memory_bytes,
            unified_memory,
            metal_device_name: parse_colon_value(&text.metal, "Chipset Model")
                .or_else(|| parse_colon_value(&text.displays, "Chipset Model")),
            gpu_family: parse_colon_value(&text.metal, "Metal Family")
                .or_else(|| parse_colon_value(&text.metal, "Metal Support")),
            native_or_virtualized,
            metal_visible,
            fallback_used: false,
            proof_stage: if apple_m4_metal_visible {
                APPLE_M4_METAL_PROOF_STAGE_DETECTED
            } else {
                APPLE_M4_METAL_PROOF_STAGE_UNAVAILABLE
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CommandText {
    success: bool,
    stdout: String,
}

fn command_stdout(command: &str, args: &[&str]) -> CommandText {
    std::process::Command::new(command)
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .map_or_else(
            |_| CommandText { success: false, stdout: String::new() },
            |output| CommandText {
                success: output.status.success(),
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            },
        )
}

fn parse_colon_value(output: &str, key: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let trimmed = line.trim();
        let value = trimmed.strip_prefix(key)?.trim_start().strip_prefix(':')?.trim();
        (!value.is_empty()).then(|| value.to_owned())
    })
}

fn first_nonempty_line(output: &str) -> Option<String> {
    output.lines().map(str::trim).find(|line| !line.is_empty()).map(ToOwned::to_owned)
}

fn parse_first_usize(value: &str) -> Option<usize> {
    let mut digits = String::new();
    for ch in value.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
        } else if !digits.is_empty() {
            break;
        }
    }
    digits.parse().ok()
}

fn parse_virtualization_state(output: &str, is_macos: bool) -> Option<String> {
    if !is_macos {
        return Some("not-macos".to_owned());
    }

    let value = parse_colon_value(output, "kern.hv_vmm_present")?;
    match value.split_whitespace().next() {
        Some("0") => Some("native-macos".to_owned()),
        Some("1") => Some("virtualized-macos".to_owned()),
        _ => Some("unknown".to_owned()),
    }
}

fn metal_text_reports_visibility(output: &str) -> bool {
    let lower = output.to_ascii_lowercase();
    lower.contains("metal")
        && (lower.contains("chipset model")
            || lower.contains("metal support")
            || lower.contains("metal family")
            || lower.contains("gpu"))
}

fn is_apple_m4_family_chip(chip: &str) -> bool {
    chip == "Apple M4" || chip.starts_with("Apple M4 ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_m4_text() -> AppleMetalProbeText {
        AppleMetalProbeText {
            host_os: "macos".to_owned(),
            sw_vers: "ProductName:\t\tmacOS\nProductVersion:\t\t15.4\nBuildVersion:\t\t24E248\n"
                .to_owned(),
            uname: "Darwin m4-mini 24.4.0 Darwin Kernel Version\n".to_owned(),
            hardware: "Hardware:\n\n    Chip: Apple M4\n    Total Number of Cores: 10 (4 performance and 6 efficiency)\n    Memory: 16 GB\n"
                .to_owned(),
            displays: "Graphics/Displays:\n\n    Apple M4:\n      Chipset Model: Apple M4\n      Total Number of Cores: 10\n"
                .to_owned(),
            metal: "Metal:\n\n    Apple M4:\n      Chipset Model: Apple M4\n      Total Number of Cores: 10\n      Metal Support: Metal 3\n"
                .to_owned(),
            metal_command_succeeded: true,
            memsize: "hw.memsize: 17179869184\n".to_owned(),
            virtualization: "kern.hv_vmm_present: 0\n".to_owned(),
        }
    }

    #[test]
    fn parses_base_m4_metal_visibility_without_execution_claims() {
        let probe = parse_apple_metal_probe(&base_m4_text());

        assert_eq!(probe.requested_backend, APPLE_M4_METAL_BACKEND);
        assert_eq!(probe.selected_backend, Some(APPLE_M4_METAL_BACKEND));
        assert_eq!(probe.runtime_api, APPLE_M4_METAL_RUNTIME_API);
        assert_eq!(probe.chip.as_deref(), Some("Apple M4"));
        assert_eq!(probe.cpu_cores, Some(10));
        assert_eq!(probe.gpu_cores, Some(10));
        assert_eq!(probe.unified_memory_bytes, Some(17_179_869_184));
        assert_eq!(probe.native_or_virtualized.as_deref(), Some("native-macos"));
        assert!(probe.metal_visible);
        assert!(!probe.fallback_used);
        assert_eq!(probe.proof_stage, APPLE_M4_METAL_PROOF_STAGE_DETECTED);
    }

    #[test]
    fn non_macos_does_not_select_metal_backend() {
        let text = AppleMetalProbeText {
            host_os: "linux".to_owned(),
            metal: "Metal Support: Metal 3\nChipset Model: Apple M4\n".to_owned(),
            metal_command_succeeded: true,
            ..AppleMetalProbeText::default()
        };

        let probe = parse_apple_metal_probe(&text);

        assert_eq!(probe.selected_backend, None);
        assert_eq!(probe.native_or_virtualized.as_deref(), Some("not-macos"));
        assert!(!probe.metal_visible);
        assert!(!probe.fallback_used);
        assert_eq!(probe.proof_stage, APPLE_M4_METAL_PROOF_STAGE_UNAVAILABLE);
    }

    #[test]
    fn artifact_path_uses_hardware_convention() {
        assert_eq!(
            apple_metal_probe_artifact_path("2026-05-05"),
            "ci/hardware/apple-m4-mac-mini/2026-05-05/metal-probe.json"
        );
    }

    #[test]
    fn metal_visible_on_non_m4_mac_does_not_select_m4_backend() {
        let text = AppleMetalProbeText {
            host_os: "macos".to_owned(),
            hardware: "Hardware:\n\n    Chip: Apple M3\n    Total Number of Cores: 8\n"
                .to_owned(),
            metal: "Metal:\n\n    Apple M3:\n      Chipset Model: Apple M3\n      Metal Support: Metal 3\n"
                .to_owned(),
            metal_command_succeeded: true,
            memsize: "hw.memsize: 17179869184\n".to_owned(),
            virtualization: "kern.hv_vmm_present: 0\n".to_owned(),
            ..AppleMetalProbeText::default()
        };

        let probe = parse_apple_metal_probe(&text);

        assert_eq!(probe.chip.as_deref(), Some("Apple M3"));
        assert!(probe.metal_visible);
        assert_eq!(probe.selected_backend, None);
        assert_eq!(probe.proof_stage, APPLE_M4_METAL_PROOF_STAGE_UNAVAILABLE);
    }
}
