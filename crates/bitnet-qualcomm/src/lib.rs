//! Qualcomm NPU environment and backend selection helpers.
//!
//! This microcrate centralizes env-var parsing shared by inference and kernel
//! crates for the Qualcomm NPU integration path (QNN / SNPE).

/// Environment variable used to enable NPU routing.
pub const BITNET_ENABLE_NPU: &str = "BITNET_ENABLE_NPU";

/// Environment variable used to choose the Qualcomm backend implementation.
pub const BITNET_NPU_BACKEND: &str = "BITNET_NPU_BACKEND";

/// Qualcomm NPU runtime family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QualcommNpuBackend {
    /// Qualcomm AI Engine Direct (QNN).
    #[default]
    Qnn,
    /// Snapdragon Neural Processing Engine (SNPE).
    Snpe,
}

impl QualcommNpuBackend {
    /// Resolve backend from `BITNET_NPU_BACKEND`; defaults to [`Self::Qnn`].
    #[must_use]
    pub fn from_env() -> Self {
        match std::env::var(BITNET_NPU_BACKEND) {
            Ok(value) if value.eq_ignore_ascii_case("snpe") => Self::Snpe,
            _ => Self::Qnn,
        }
    }

    /// Provider name suffix used in kernel registration.
    #[must_use]
    pub const fn provider_name(self) -> &'static str {
        match self {
            Self::Qnn => "npu-qnn",
            Self::Snpe => "npu-snpe",
        }
    }

    /// Human-readable runtime family.
    #[must_use]
    pub const fn runtime_name(self) -> &'static str {
        match self {
            Self::Qnn => "QNN",
            Self::Snpe => "SNPE",
        }
    }
}

/// Return `true` when the runtime should prefer Qualcomm NPU execution.
#[must_use]
pub fn npu_enabled() -> bool {
    std::env::var(BITNET_ENABLE_NPU)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
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
    fn backend_reads_snpe_case_insensitive() {
        temp_env::with_var(BITNET_NPU_BACKEND, Some("SnPe"), || {
            assert_eq!(QualcommNpuBackend::from_env(), QualcommNpuBackend::Snpe);
        });
    }

    #[test]
    fn npu_enabled_accepts_true_and_one() {
        temp_env::with_var(BITNET_ENABLE_NPU, Some("true"), || {
            assert!(npu_enabled());
        });

        temp_env::with_var(BITNET_ENABLE_NPU, Some("1"), || {
            assert!(npu_enabled());
        });
    }
}
