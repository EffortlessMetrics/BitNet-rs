//! Portable CPU topology and brand probing.

#[cfg(any(target_os = "windows", target_os = "macos"))]
use std::process::Command;

use serde::{Deserialize, Serialize};

/// Basic CPU topology facts available without platform-specific dependencies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuTopologyProbe {
    /// Human-readable CPU brand string when available.
    pub brand: Option<String>,
    /// Best-effort physical core count.
    pub cores: usize,
    /// Logical thread count visible to the process.
    pub threads: usize,
}

/// Probe CPU brand and topology, falling back to logical thread count.
pub fn probe_cpu_topology() -> CpuTopologyProbe {
    let threads = std::thread::available_parallelism().map(std::num::NonZero::get).unwrap_or(1);
    CpuTopologyProbe { brand: cpu_brand(), cores: threads, threads }
}

fn cpu_brand() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo")
            && let Some(value) = cpuinfo.lines().find_map(|line| {
                line.strip_prefix("model name")
                    .and_then(|rest| rest.split_once(':').map(|(_, value)| value.trim().to_owned()))
            })
        {
            return non_empty(&value);
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Some(value) = command_stdout(
            "powershell",
            &[
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name",
            ],
        ) {
            return non_empty(&value);
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(value) = command_stdout("sysctl", &["-n", "machdep.cpu.brand_string"]) {
            return non_empty(&value);
        }
    }

    None
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    Command::new(command)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn non_empty(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() { None } else { Some(trimmed.to_owned()) }
}
