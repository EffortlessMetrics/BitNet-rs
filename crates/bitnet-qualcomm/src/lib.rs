//! Qualcomm runtime configuration helpers.
//!
//! This crate centralizes environment-driven controls for Qualcomm NPU runtime
//! selection so kernel and inference crates can share one source of truth.

/// Environment variable used to enable NPU routing.
pub const BITNET_ENABLE_NPU: &str = "BITNET_ENABLE_NPU";

/// Environment variable used to pick the Qualcomm backend implementation.
pub const BITNET_NPU_BACKEND: &str = "BITNET_NPU_BACKEND";

/// Qualcomm NPU backend selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QualcommNpuBackend {
    /// Qualcomm AI Engine Direct (QNN).
    #[default]
    Qnn,
    /// Snapdragon Neural Processing Engine (SNPE).
    Snpe,
}

impl QualcommNpuBackend {
    /// Parse backend from runtime environment.
    #[must_use]
    pub fn from_env() -> Self {
        match std::env::var(BITNET_NPU_BACKEND) {
            Ok(value) if value.eq_ignore_ascii_case("snpe") => Self::Snpe,
            _ => Self::Qnn,
        }
    }

    /// Stable provider name used by kernel registration.
    #[must_use]
    pub const fn provider_name(self) -> &'static str {
        match self {
            Self::Qnn => "npu-qnn",
            Self::Snpe => "npu-snpe",
        }
    }

    /// Human-readable runtime name.
    #[must_use]
    pub const fn runtime_name(self) -> &'static str {
        match self {
            Self::Qnn => "QNN",
            Self::Snpe => "SNPE",
        }
    }
}

/// Return `true` when the runtime should prefer NPU execution.
#[must_use]
pub fn npu_requested() -> bool {
    std::env::var(BITNET_ENABLE_NPU)
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_qnn() {
        temp_env::with_var_unset(BITNET_NPU_BACKEND, || {
            assert_eq!(QualcommNpuBackend::from_env(), QualcommNpuBackend::Qnn);
        });
    }

    #[test]
    fn reads_snpe_backend_case_insensitive() {
        temp_env::with_var(BITNET_NPU_BACKEND, Some("SnPe"), || {
            assert_eq!(QualcommNpuBackend::from_env(), QualcommNpuBackend::Snpe);
        });
    }

    #[test]
    fn npu_requested_supports_true_and_one() {
        temp_env::with_var(BITNET_ENABLE_NPU, Some("true"), || {
            assert!(npu_requested());
        });
        temp_env::with_var(BITNET_ENABLE_NPU, Some("1"), || {
            assert!(npu_requested());
        });
    }

    #[test]
    fn provider_and_runtime_names_are_stable() {
        assert_eq!(QualcommNpuBackend::Qnn.provider_name(), "npu-qnn");
        assert_eq!(QualcommNpuBackend::Qnn.runtime_name(), "QNN");
        assert_eq!(QualcommNpuBackend::Snpe.provider_name(), "npu-snpe");
        assert_eq!(QualcommNpuBackend::Snpe.runtime_name(), "SNPE");
    }
}
