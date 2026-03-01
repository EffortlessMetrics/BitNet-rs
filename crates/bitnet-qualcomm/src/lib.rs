//! Qualcomm NPU integration helpers.
//!
//! This crate centralizes environment policy for Qualcomm-oriented NPU routing
//! so kernel and inference crates can share the same runtime behavior.

use bitnet_common::Device;

/// Environment variable used to enable NPU routing.
pub const BITNET_ENABLE_NPU: &str = "BITNET_ENABLE_NPU";

/// Environment variable selecting the Qualcomm NPU backend implementation.
pub const BITNET_NPU_BACKEND: &str = "BITNET_NPU_BACKEND";

/// Environment variable controlling CPU fallback behavior for NPU inference.
pub const BITNET_NPU_ALLOW_FALLBACK: &str = "BITNET_NPU_ALLOW_FALLBACK";

/// Qualcomm NPU backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QualcommNpuBackend {
    /// Qualcomm Neural Network SDK backend.
    #[default]
    Qnn,
    /// Snapdragon Neural Processing Engine backend.
    Snpe,
}

impl QualcommNpuBackend {
    /// Parse backend selection from `BITNET_NPU_BACKEND`.
    pub fn from_env() -> Self {
        match std::env::var(BITNET_NPU_BACKEND) {
            Ok(value) if value.eq_ignore_ascii_case("snpe") => Self::Snpe,
            _ => Self::Qnn,
        }
    }

    /// Canonical kernel provider name.
    pub fn kernel_name(self) -> &'static str {
        match self {
            Self::Qnn => "npu-qnn",
            Self::Snpe => "npu-snpe",
        }
    }

    /// Human-readable Qualcomm runtime name.
    pub fn runtime_name(self) -> &'static str {
        match self {
            Self::Qnn => "QNN",
            Self::Snpe => "SNPE",
        }
    }
}

/// Return `true` when NPU execution is explicitly enabled.
pub fn npu_requested() -> bool {
    std::env::var(BITNET_ENABLE_NPU)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Return `true` when CPU fallback should be allowed for NPU operations.
pub fn allow_cpu_fallback() -> bool {
    std::env::var(BITNET_NPU_ALLOW_FALLBACK)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true)
}

/// Map an external `--device` style token to an internal device preference.
pub fn map_device_token(token: &str) -> Option<Device> {
    match token {
        "cpu" => Some(Device::Cpu),
        "cuda" | "gpu" => Some(Device::Cuda(0)),
        "metal" | "npu" => Some(Device::Metal),
        "oneapi" | "opencl" | "intel-gpu" => Some(Device::OpenCL(0)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_defaults_to_qnn() {
        temp_env::with_var_unset(BITNET_NPU_BACKEND, || {
            assert_eq!(QualcommNpuBackend::from_env(), QualcommNpuBackend::Qnn);
        });
    }

    #[test]
    fn backend_parses_snpe() {
        temp_env::with_var(BITNET_NPU_BACKEND, Some("snpe"), || {
            assert_eq!(QualcommNpuBackend::from_env(), QualcommNpuBackend::Snpe);
        });
    }

    #[test]
    fn npu_requested_uses_enable_env() {
        temp_env::with_var(BITNET_ENABLE_NPU, Some("true"), || {
            assert!(npu_requested());
        });
    }

    #[test]
    fn fallback_defaults_enabled() {
        temp_env::with_var_unset(BITNET_NPU_ALLOW_FALLBACK, || {
            assert!(allow_cpu_fallback());
        });
    }
}
