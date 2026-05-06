//! JSON-ready Core Ultra 7 258V Lunar Lake platform probe.
#![allow(clippy::doc_markdown)]

use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::runtimes::{
    OpenVinoProbe, level_zero::probe_level_zero, opencl::probe_opencl_runtime,
    openvino::probe_openvino,
};

use super::{arc140v::probe_intel_arc_140v, cpu::probe_lnl258v_cpu, npu::probe_intel_npu};

/// Proof stage emitted by the platform probe.
pub const LNL258V_PROOF_STAGE_RUNTIME_DETECTED: &str = "runtime_detected";

/// Memory context for the shared-memory Lunar Lake platform.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlatformMemoryProbe {
    /// Total system memory in bytes when available.
    pub total_bytes: Option<u64>,
    /// Shared memory available to integrated devices when known.
    pub shared_memory_bytes: Option<u64>,
    /// Whether platform devices use shared memory.
    pub shared_memory: bool,
}

/// Power and thermal context for platform receipts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlatformPowerProbe {
    /// OS power mode or governor summary when available.
    pub mode: Option<String>,
    /// Thermal profile when available.
    pub thermal_profile: Option<String>,
    /// Whether AC power is visible as connected.
    pub ac_power: Option<bool>,
}

/// Visibility-only Lunar Lake platform probe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct Lnl258vPlatformProbe {
    /// Stable machine identifier used by receipts.
    pub machine_id: String,
    /// Universal proof stage for this visibility-only probe.
    pub proof_stage: String,
    /// OS family.
    pub os: String,
    /// OS version/build when available.
    pub os_build: Option<String>,
    /// CPU architecture.
    pub arch: String,
    /// CPU lane facts.
    pub cpu: super::cpu::Lnl258vCpuProbe,
    /// Arc 140V lane visibility facts.
    pub arc140v: super::arc140v::IntelArc140vProbe,
    /// Intel NPU lane visibility facts.
    pub npu: super::npu::IntelNpuProbe,
    /// OpenVINO runtime visibility facts.
    pub openvino: OpenVinoProbe,
    /// Memory context.
    pub memory: PlatformMemoryProbe,
    /// Power context.
    pub power: PlatformPowerProbe,
    /// Always false: this probe never substitutes a fallback device.
    pub fallback_used: bool,
    /// Human-readable status.
    pub status: String,
    /// Non-fatal failure reason when platform identity is incomplete.
    pub failure_reason: Option<String>,
}

/// Probe Lunar Lake platform visibility without running inference.
pub fn probe_lnl258v_platform() -> Lnl258vPlatformProbe {
    let cpu = probe_lnl258v_cpu();
    let memory = probe_platform_memory();
    let power = probe_platform_power();
    let opencl = probe_opencl_runtime();
    let level_zero = probe_level_zero();
    let openvino = probe_openvino();
    let arc140v = probe_intel_arc_140v(&opencl, &level_zero, &openvino, &memory, &power);
    let npu = probe_intel_npu(&openvino);

    Lnl258vPlatformProbe {
        machine_id: "intel-258v".to_owned(),
        proof_stage: LNL258V_PROOF_STAGE_RUNTIME_DETECTED.to_owned(),
        os: std::env::consts::OS.to_owned(),
        os_build: os_build(),
        arch: std::env::consts::ARCH.to_owned(),
        cpu,
        arc140v,
        npu,
        openvino,
        memory,
        power,
        fallback_used: false,
        status: LNL258V_PROOF_STAGE_RUNTIME_DETECTED.to_owned(),
        failure_reason: None,
    }
}

/// Probe system memory context.
pub fn probe_platform_memory() -> PlatformMemoryProbe {
    let total_bytes = total_memory_bytes();
    PlatformMemoryProbe { total_bytes, shared_memory_bytes: total_bytes, shared_memory: true }
}

/// Probe platform power context.
pub fn probe_platform_power() -> PlatformPowerProbe {
    PlatformPowerProbe {
        mode: power_mode(),
        thermal_profile: thermal_profile(),
        ac_power: ac_power_connected(),
    }
}

fn total_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/meminfo").ok().and_then(|meminfo| {
            meminfo.lines().find_map(|line| {
                let rest = line.strip_prefix("MemTotal:")?;
                let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
                Some(kb * 1024)
            })
        })
    }

    #[cfg(target_os = "windows")]
    {
        command_stdout(
            "powershell",
            &[
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ],
        )
        .and_then(|value| value.trim().parse::<u64>().ok())
    }

    #[cfg(target_os = "macos")]
    {
        command_stdout("sysctl", &["-n", "hw.memsize"])
            .and_then(|value| value.trim().parse::<u64>().ok())
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        None
    }
}

#[allow(clippy::missing_const_for_fn)]
fn power_mode() -> Option<String> {
    platform_power_mode()
}

#[allow(clippy::missing_const_for_fn)]
fn thermal_profile() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let zones = std::fs::read_dir("/sys/class/thermal").ok()?.flatten().count();
        (zones > 0).then(|| format!("{zones} thermal zones visible"))
    }

    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[allow(clippy::missing_const_for_fn)]
fn ac_power_connected() -> Option<bool> {
    #[cfg(target_os = "linux")]
    {
        let supplies = std::fs::read_dir("/sys/class/power_supply").ok()?;
        for entry in supplies.flatten() {
            let online_path = entry.path().join("online");
            if let Ok(value) = std::fs::read_to_string(online_path) {
                return Some(value.trim() == "1");
            }
        }
    }

    None
}

fn os_build() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let release = std::fs::read_to_string("/etc/os-release").ok();
        let pretty = release.as_deref().and_then(|content| {
            content.lines().find_map(|line| {
                let value = line.strip_prefix("PRETTY_NAME=")?;
                Some(value.trim_matches('"').to_owned())
            })
        });
        let kernel = command_stdout("uname", &["-r"]);
        return match (pretty, kernel) {
            (Some(pretty), Some(kernel)) => Some(format!("{pretty}; kernel {kernel}")),
            (Some(pretty), None) => Some(pretty),
            (None, Some(kernel)) => Some(format!("kernel {kernel}")),
            (None, None) => None,
        };
    }

    #[cfg(target_os = "windows")]
    {
        return command_stdout(
            "powershell",
            &[
                "-NoProfile",
                "-Command",
                "$ci = Get-ComputerInfo; \"$($ci.OsName) $($ci.OsVersion) $($ci.WindowsVersion)\"",
            ],
        );
    }

    #[cfg(target_os = "macos")]
    {
        return command_stdout("sw_vers", &["-productVersion"]);
    }

    #[allow(unreachable_code)]
    None
}

fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    Command::new(command)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|value| !value.is_empty())
}

#[cfg(target_os = "linux")]
fn platform_power_mode() -> Option<String> {
    let governors = std::fs::read_dir("/sys/devices/system/cpu")
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let path = entry.path().join("cpufreq/scaling_governor");
            std::fs::read_to_string(path).ok().map(|value| value.trim().to_owned())
        })
        .filter(|value| !value.is_empty())
        .collect::<std::collections::BTreeSet<_>>();
    (!governors.is_empty()).then(|| governors.into_iter().collect::<Vec<_>>().join(","))
}

#[cfg(target_os = "windows")]
fn platform_power_mode() -> Option<String> {
    command_stdout("powercfg", &["/GETACTIVESCHEME"])
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
const fn platform_power_mode() -> Option<String> {
    None
}
