use crate::DeviceConfig;

/// Thermal policy class that affects how performance evidence must be scoped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalPolicy {
    /// Mobile Apple Silicon host with no active fan.
    FanlessMobile,
}

/// Stable proof-lane label carried by a device profile contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceProfileLabel {
    pub backend_label: &'static str,
    pub runtime_api: &'static str,
    pub execution_available: bool,
    pub claim_scope: &'static str,
}

/// Storage and artifact-retention policy for a device profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceProfileStoragePolicy {
    pub cache_root_required: bool,
    pub large_artifact_sweep_allowed: bool,
    pub model_binaries_committed: bool,
}

/// Claims that are explicitly unsupported by a device profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceProfileUnsupportedClaim {
    MetalModelInference,
    MpsGraphModelInference,
    NeuralEngineExecution,
    Qk256AppleSilicon,
    M4MacMiniPerformance,
    BroadAppleSiliconPerformance,
    BitNetLocalAnswerQualityFromDenseSlm,
}

/// Structured host/profile contract for proof-lane routing and receipt wording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceProfileContract {
    pub profile_id: &'static str,
    pub machine_id: &'static str,
    pub soc_family: &'static str,
    pub thermal_policy: ThermalPolicy,
    pub core_split_required: bool,
    pub memory_tier_required: bool,
    pub storage: DeviceProfileStoragePolicy,
    pub labels: &'static [DeviceProfileLabel],
    pub unsupported_claims: &'static [DeviceProfileUnsupportedClaim],
}

impl DeviceProfileContract {
    #[must_use]
    pub fn apple_m3_air() -> Self {
        Self {
            profile_id: "apple-m3-macbook-air",
            machine_id: "apple-m3-macbook-air",
            soc_family: "Apple M3",
            thermal_policy: ThermalPolicy::FanlessMobile,
            core_split_required: true,
            memory_tier_required: true,
            storage: DeviceProfileStoragePolicy {
                cache_root_required: true,
                large_artifact_sweep_allowed: true,
                model_binaries_committed: false,
            },
            labels: &APPLE_M3_AIR_LABELS,
            unsupported_claims: &APPLE_M3_AIR_UNSUPPORTED_CLAIMS,
        }
    }

    #[must_use]
    pub fn label(&self, backend_label: &str) -> Option<DeviceProfileLabel> {
        self.labels.iter().copied().find(|label| label.backend_label == backend_label)
    }

    #[must_use]
    pub fn rejects(&self, claim: DeviceProfileUnsupportedClaim) -> bool {
        self.unsupported_claims.contains(&claim)
    }
}

impl DeviceConfig {
    /// Return the structured host/profile contract for proof-lane-specific labels.
    #[must_use]
    pub fn device_profile_contract(&self) -> Option<DeviceProfileContract> {
        match self {
            DeviceConfig::AppleM3AirMetal
            | DeviceConfig::AppleM3AirMpsGraph
            | DeviceConfig::AppleM3AirCpuNeon => Some(DeviceProfileContract::apple_m3_air()),
            _ => None,
        }
    }
}

const APPLE_M3_AIR_LABELS: [DeviceProfileLabel; 3] = [
    DeviceProfileLabel {
        backend_label: "apple-m3-air-cpu-neon",
        runtime_api: "cpu-neon",
        execution_available: true,
        claim_scope: "M3 Air Apple CPU/NEON dense SLM and receipt-checked host evidence only",
    },
    DeviceProfileLabel {
        backend_label: "apple-m3-air-metal",
        runtime_api: "metal",
        execution_available: false,
        claim_scope: "M3 Air Metal visibility/request identity only until receipt-backed runtime work lands",
    },
    DeviceProfileLabel {
        backend_label: "apple-m3-air-mpsgraph",
        runtime_api: "mpsgraph",
        execution_available: false,
        claim_scope: "M3 Air MPSGraph visibility/request identity only until receipt-backed runtime work lands",
    },
];

const APPLE_M3_AIR_UNSUPPORTED_CLAIMS: [DeviceProfileUnsupportedClaim; 7] = [
    DeviceProfileUnsupportedClaim::MetalModelInference,
    DeviceProfileUnsupportedClaim::MpsGraphModelInference,
    DeviceProfileUnsupportedClaim::NeuralEngineExecution,
    DeviceProfileUnsupportedClaim::Qk256AppleSilicon,
    DeviceProfileUnsupportedClaim::M4MacMiniPerformance,
    DeviceProfileUnsupportedClaim::BroadAppleSiliconPerformance,
    DeviceProfileUnsupportedClaim::BitNetLocalAnswerQualityFromDenseSlm,
];
