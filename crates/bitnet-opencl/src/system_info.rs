//! System information collection for GPU diagnostics.

use serde::{Deserialize, Serialize};

/// System-level information about the host machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Operating system name (e.g., "Windows", "Linux", "macOS").
    pub os_name: String,
    /// Operating system version string.
    pub os_version: String,
    /// Kernel version string (e.g., "6.5.0-generic").
    pub kernel_version: String,
    /// CPU model name.
    pub cpu_name: String,
    /// Number of logical CPU cores.
    pub cpu_cores: usize,
    /// Total system memory in megabytes.
    pub total_memory_mb: u64,
    /// Rust compiler version used to build this binary.
    pub rust_version: String,
    /// `BitNet` crate version.
    pub bitnet_version: String,
}

/// Collect system information from the current host.
///
/// Fields that cannot be determined are filled with `"unknown"`.
pub fn collect_system_info() -> SystemInfo {
    SystemInfo {
        os_name: std::env::consts::OS.to_owned(),
        os_version: os_version(),
        kernel_version: kernel_version(),
        cpu_name: cpu_name(),
        cpu_cores: std::thread::available_parallelism().map(std::num::NonZero::get).unwrap_or(1),
        total_memory_mb: total_memory_mb(),
        rust_version: rustc_version(),
        bitnet_version: env!("CARGO_PKG_VERSION").to_owned(),
    }
}

fn os_version() -> String {
    #[cfg(target_os = "windows")]
    {
        // Read Windows version from registry-style env or fallback.
        std::env::var("OS").unwrap_or_else(|_| "Windows".to_owned())
    }
    #[cfg(not(target_os = "windows"))]
    {
        read_command_output("uname", &["-r"])
    }
}

fn kernel_version() -> String {
    #[cfg(target_os = "windows")]
    {
        read_command_output("cmd", &["/C", "ver"])
    }
    #[cfg(not(target_os = "windows"))]
    {
        read_command_output("uname", &["-r"])
    }
}

fn cpu_name() -> String {
    #[cfg(target_os = "windows")]
    {
        std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".to_owned())
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|info| {
                info.lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|s| s.trim().to_owned())
            })
            .unwrap_or_else(|| "unknown".to_owned())
    }
    #[cfg(target_os = "macos")]
    {
        read_command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        "unknown".to_owned()
    }
}

#[allow(clippy::missing_const_for_fn)] // Conditional: non-const on Linux
fn total_memory_mb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|info| {
                info.lines()
                    .find(|l| l.starts_with("MemTotal"))
                    .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok()))
            })
            .map(|kb| kb / 1024)
            .unwrap_or(0)
    }
    #[cfg(not(target_os = "linux"))]
    {
        0 // Not easily available without a sys-info crate.
    }
}

fn rustc_version() -> String {
    read_command_output("rustc", &["--version"])
}

fn read_command_output(cmd: &str, args: &[&str]) -> String {
    std::process::Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .and_then(|o| if o.status.success() { String::from_utf8(o.stdout).ok() } else { None })
        .map_or_else(|| "unknown".to_owned(), |s| s.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_system_info_populates_fields() {
        let info = collect_system_info();
        assert!(!info.os_name.is_empty());
        assert!(info.cpu_cores >= 1);
        assert!(!info.bitnet_version.is_empty());
    }

    #[test]
    fn os_name_is_known_platform() {
        let info = collect_system_info();
        let known = ["windows", "linux", "macos"];
        assert!(known.contains(&info.os_name.as_str()), "unexpected OS: {}", info.os_name);
    }

    #[test]
    fn rust_version_contains_rustc() {
        let info = collect_system_info();
        assert!(
            info.rust_version.contains("rustc"),
            "rust_version should contain 'rustc': {}",
            info.rust_version
        );
    }
}
