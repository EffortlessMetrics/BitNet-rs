//! AMD ROCm runtime detection helpers.
//!
//! This crate centralizes the ROCm-specific detection policy so callers can
//! keep device-probe orchestration separate from AMD runtime probing details.

use std::process::{Command, Stdio};

/// Returns whether strict detection mode is enabled.
///
/// Strict mode is enabled when `BITNET_STRICT_MODE` is `1` or `true`
/// (case-insensitive).
#[must_use]
pub fn strict_mode_enabled() -> bool {
    std::env::var("BITNET_STRICT_MODE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Returns an env-forced ROCm availability decision, if present.
///
/// Behavior:
/// - Returns `None` when strict mode is enabled.
/// - Returns `Some(false)` when `BITNET_GPU_FAKE=none`.
/// - Returns `Some(true)` when `BITNET_GPU_FAKE` contains `rocm` or `gpu`.
/// - Returns `Some(false)` for any other `BITNET_GPU_FAKE` value.
#[must_use]
pub fn fake_rocm_available_from_env() -> Option<bool> {
    if strict_mode_enabled() {
        return None;
    }

    let fake = std::env::var("BITNET_GPU_FAKE").ok()?;
    let normalized = fake.trim().to_ascii_lowercase();

    if normalized == "none" {
        return Some(false);
    }

    let has_rocm = normalized
        .split([',', ';', '|', ' '])
        .filter(|part| !part.is_empty())
        .any(|part| matches!(part, "rocm" | "gpu"));

    Some(has_rocm)
}

/// Probe whether ROCm runtime appears to be available.
///
/// Detection strategy:
/// 1. `BITNET_GPU_FAKE` override when strict mode is disabled.
/// 2. `rocm-smi --showid` command success.
#[must_use]
pub fn rocm_available_runtime() -> bool {
    if let Some(fake) = fake_rocm_available_from_env() {
        return fake;
    }

    command_ok("rocm-smi", &["--showid"])
}

fn command_ok(cmd: &str, args: &[&str]) -> bool {
    Command::new(cmd)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
