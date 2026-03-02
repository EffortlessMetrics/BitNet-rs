//! System information collection for GPU diagnostics.

pub use bitnet_system_info_core::{SystemInfo, collect_system_info_with_version};

/// Collect system information using this crate's package version.
#[must_use]
pub fn collect_system_info() -> SystemInfo {
    collect_system_info_with_version(env!("CARGO_PKG_VERSION"))
}
