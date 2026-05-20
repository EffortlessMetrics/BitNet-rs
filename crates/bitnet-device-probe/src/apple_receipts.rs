//! Durable Apple backend receipt fields.
//!
//! These types record Apple proof identity without collapsing Metal, `MPSGraph`,
//! and CPU/NEON evidence. They do not prove `BitNet` inference on their own.

use serde::{Deserialize, Serialize};
use std::fmt;

pub const APPLE_M3_AIR_MACHINE_ID: &str = "apple-m3-macbook-air";
pub const APPLE_M3_AIR_METAL_BACKEND: &str = "apple-m3-air-metal";
pub const APPLE_M3_AIR_MPSGRAPH_BACKEND: &str = "apple-m3-air-mpsgraph";
pub const APPLE_M3_AIR_CPU_NEON_BACKEND: &str = "apple-m3-air-cpu-neon";
pub const APPLE_VISIBILITY_PREFLIGHT_KIND: &str = "backend_visibility_preflight";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppleM3AirProofLabel {
    pub backend_label: String,
    pub runtime_api: String,
    pub execution_available: bool,
    pub claim_scope: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppleM3AirStoragePolicy {
    pub cache_root_required: bool,
    pub large_artifact_sweep_allowed: bool,
    pub model_binaries_committed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppleM3AirUnsupportedClaim {
    MetalModelInference,
    MpsGraphModelInference,
    NeuralEngineExecution,
    Qk256AppleSilicon,
    M4MacMiniPerformance,
    BroadAppleSiliconPerformance,
    BitNetLocalAnswerQualityFromDenseSlm,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppleM3AirHostProfileContract {
    pub machine_id: String,
    pub soc_family: String,
    pub thermal_policy: String,
    pub core_split_required: bool,
    pub memory_tier_required: bool,
    pub storage: AppleM3AirStoragePolicy,
    pub proof_lane_labels: Vec<AppleM3AirProofLabel>,
    pub unsupported_claims: Vec<AppleM3AirUnsupportedClaim>,
}

impl AppleM3AirHostProfileContract {
    #[must_use]
    pub fn current() -> Self {
        Self {
            machine_id: APPLE_M3_AIR_MACHINE_ID.to_owned(),
            soc_family: "Apple M3".to_owned(),
            thermal_policy: "fanless_mobile".to_owned(),
            core_split_required: true,
            memory_tier_required: true,
            storage: AppleM3AirStoragePolicy {
                cache_root_required: true,
                large_artifact_sweep_allowed: true,
                model_binaries_committed: false,
            },
            proof_lane_labels: vec![
                AppleM3AirProofLabel {
                    backend_label: APPLE_M3_AIR_CPU_NEON_BACKEND.to_owned(),
                    runtime_api: "cpu-neon".to_owned(),
                    execution_available: true,
                    claim_scope: "M3 Air Apple CPU/NEON dense SLM and receipt-checked host evidence only"
                        .to_owned(),
                },
                AppleM3AirProofLabel {
                    backend_label: APPLE_M3_AIR_METAL_BACKEND.to_owned(),
                    runtime_api: "metal".to_owned(),
                    execution_available: false,
                    claim_scope: "M3 Air Metal visibility/request identity only until receipt-backed runtime work lands"
                        .to_owned(),
                },
                AppleM3AirProofLabel {
                    backend_label: APPLE_M3_AIR_MPSGRAPH_BACKEND.to_owned(),
                    runtime_api: "mpsgraph".to_owned(),
                    execution_available: false,
                    claim_scope: "M3 Air MPSGraph visibility/request identity only until receipt-backed runtime work lands"
                        .to_owned(),
                },
            ],
            unsupported_claims: vec![
                AppleM3AirUnsupportedClaim::MetalModelInference,
                AppleM3AirUnsupportedClaim::MpsGraphModelInference,
                AppleM3AirUnsupportedClaim::NeuralEngineExecution,
                AppleM3AirUnsupportedClaim::Qk256AppleSilicon,
                AppleM3AirUnsupportedClaim::M4MacMiniPerformance,
                AppleM3AirUnsupportedClaim::BroadAppleSiliconPerformance,
                AppleM3AirUnsupportedClaim::BitNetLocalAnswerQualityFromDenseSlm,
            ],
        }
    }

    pub fn validate(&self) -> Result<(), AppleReceiptError> {
        require_nonempty("machine_id", &self.machine_id)?;
        require_nonempty("soc_family", &self.soc_family)?;
        require_nonempty("thermal_policy", &self.thermal_policy)?;
        if self.machine_id != APPLE_M3_AIR_MACHINE_ID {
            return Err(AppleReceiptError::UnsupportedAppleMachine {
                machine_id: self.machine_id.clone(),
            });
        }
        if self.thermal_policy != "fanless_mobile" {
            return Err(AppleReceiptError::InvalidProfileField("thermal_policy"));
        }
        if !self.core_split_required {
            return Err(AppleReceiptError::InvalidProfileField("core_split_required"));
        }
        if !self.memory_tier_required {
            return Err(AppleReceiptError::InvalidProfileField("memory_tier_required"));
        }
        if !self.storage.cache_root_required || !self.storage.large_artifact_sweep_allowed {
            return Err(AppleReceiptError::InvalidProfileField("storage"));
        }
        if self.storage.model_binaries_committed {
            return Err(AppleReceiptError::ClaimBoundaryViolation("model_binaries_committed"));
        }
        for required in [
            APPLE_M3_AIR_CPU_NEON_BACKEND,
            APPLE_M3_AIR_METAL_BACKEND,
            APPLE_M3_AIR_MPSGRAPH_BACKEND,
        ] {
            if !self.proof_lane_labels.iter().any(|label| label.backend_label == required) {
                return Err(AppleReceiptError::InvalidProfileField("proof_lane_labels"));
            }
        }
        for label in &self.proof_lane_labels {
            require_nonempty("proof_lane_labels.backend_label", &label.backend_label)?;
            require_nonempty("proof_lane_labels.runtime_api", &label.runtime_api)?;
            require_nonempty("proof_lane_labels.claim_scope", &label.claim_scope)?;
        }
        for (backend_label, runtime_api, execution_available) in [
            (APPLE_M3_AIR_CPU_NEON_BACKEND, "cpu-neon", true),
            (APPLE_M3_AIR_METAL_BACKEND, "metal", false),
            (APPLE_M3_AIR_MPSGRAPH_BACKEND, "mpsgraph", false),
        ] {
            let label = self
                .proof_lane_labels
                .iter()
                .find(|label| label.backend_label == backend_label)
                .ok_or(AppleReceiptError::InvalidProfileField("proof_lane_labels"))?;
            if label.runtime_api != runtime_api {
                return Err(AppleReceiptError::InvalidProfileField(
                    "proof_lane_labels.runtime_api",
                ));
            }
            if label.execution_available != execution_available {
                if backend_label == APPLE_M3_AIR_CPU_NEON_BACKEND {
                    return Err(AppleReceiptError::InvalidProfileField("apple-m3-air-cpu-neon"));
                }
                return Err(AppleReceiptError::ClaimBoundaryViolation("accelerator_execution"));
            }
        }
        for claim in [
            AppleM3AirUnsupportedClaim::MetalModelInference,
            AppleM3AirUnsupportedClaim::MpsGraphModelInference,
            AppleM3AirUnsupportedClaim::NeuralEngineExecution,
            AppleM3AirUnsupportedClaim::Qk256AppleSilicon,
            AppleM3AirUnsupportedClaim::M4MacMiniPerformance,
            AppleM3AirUnsupportedClaim::BroadAppleSiliconPerformance,
            AppleM3AirUnsupportedClaim::BitNetLocalAnswerQualityFromDenseSlm,
        ] {
            if !self.unsupported_claims.contains(&claim) {
                return Err(AppleReceiptError::InvalidProfileField("unsupported_claims"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppleResolvedDevice {
    pub chip: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_cores: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unified_memory: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bandwidth_gbps: Option<u32>,
}

impl AppleResolvedDevice {
    #[must_use]
    pub fn new(chip: impl Into<String>) -> Self {
        Self {
            chip: chip.into(),
            gpu_cores: None,
            unified_memory: None,
            memory_bandwidth_gbps: None,
        }
    }

    #[must_use]
    pub const fn with_gpu_cores(mut self, gpu_cores: u32) -> Self {
        self.gpu_cores = Some(gpu_cores);
        self
    }

    #[must_use]
    pub const fn with_unified_memory(mut self, unified_memory: bool) -> Self {
        self.unified_memory = Some(unified_memory);
        self
    }

    #[must_use]
    pub const fn with_memory_bandwidth_gbps(mut self, memory_bandwidth_gbps: u32) -> Self {
        self.memory_bandwidth_gbps = Some(memory_bandwidth_gbps);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppleVisibilityClaimBoundary {
    pub model_downloaded: bool,
    pub model_loaded: bool,
    pub model_inference: bool,
    pub metal_inference_claimed: bool,
    pub mpsgraph_model_inference_claimed: bool,
    pub neural_engine_claimed: bool,
    pub performance_claimed: bool,
}

impl AppleVisibilityClaimBoundary {
    #[must_use]
    pub const fn bounded_preflight() -> Self {
        Self {
            model_downloaded: false,
            model_loaded: false,
            model_inference: false,
            metal_inference_claimed: false,
            mpsgraph_model_inference_claimed: false,
            neural_engine_claimed: false,
            performance_claimed: false,
        }
    }

    fn validate_bounded_preflight(&self) -> Result<(), AppleReceiptError> {
        if self.model_downloaded {
            return Err(AppleReceiptError::ClaimBoundaryViolation("model_downloaded"));
        }
        if self.model_loaded {
            return Err(AppleReceiptError::ClaimBoundaryViolation("model_loaded"));
        }
        if self.model_inference {
            return Err(AppleReceiptError::ClaimBoundaryViolation("model_inference"));
        }
        if self.metal_inference_claimed {
            return Err(AppleReceiptError::ClaimBoundaryViolation("metal_inference_claimed"));
        }
        if self.mpsgraph_model_inference_claimed {
            return Err(AppleReceiptError::ClaimBoundaryViolation(
                "mpsgraph_model_inference_claimed",
            ));
        }
        if self.neural_engine_claimed {
            return Err(AppleReceiptError::ClaimBoundaryViolation("neural_engine_claimed"));
        }
        if self.performance_claimed {
            return Err(AppleReceiptError::ClaimBoundaryViolation("performance_claimed"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppleBackendVisibilityPreflight {
    pub machine_id: String,
    pub artifact_kind: String,
    pub requested_backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_backend: Option<String>,
    pub runtime_api: String,
    pub resolved_device: AppleResolvedDevice,
    pub metal_visible: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mpsgraph_visible: Option<bool>,
    pub fallback_used: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    pub artifact_path: String,
    pub claim_boundary: AppleVisibilityClaimBoundary,
}

impl AppleBackendVisibilityPreflight {
    #[must_use]
    pub fn new(
        machine_id: impl Into<String>,
        requested_backend: impl Into<String>,
        selected_backend: Option<impl Into<String>>,
        runtime_api: impl Into<String>,
        resolved_device: AppleResolvedDevice,
        visibility: AppleRuntimeVisibility,
        fallback_used: bool,
        artifact_path: impl Into<String>,
    ) -> Self {
        Self {
            machine_id: machine_id.into(),
            artifact_kind: APPLE_VISIBILITY_PREFLIGHT_KIND.to_owned(),
            requested_backend: requested_backend.into(),
            selected_backend: selected_backend.map(Into::into),
            runtime_api: runtime_api.into(),
            resolved_device,
            metal_visible: visibility.metal_visible,
            mpsgraph_visible: visibility.mpsgraph_visible,
            fallback_used,
            fallback_reason: None,
            artifact_path: artifact_path.into(),
            claim_boundary: AppleVisibilityClaimBoundary::bounded_preflight(),
        }
    }

    #[must_use]
    pub fn m3_air_metal(
        selected_backend: Option<impl Into<String>>,
        resolved_device: AppleResolvedDevice,
        metal_visible: bool,
        fallback_used: bool,
        artifact_path: impl Into<String>,
    ) -> Self {
        Self::new(
            APPLE_M3_AIR_MACHINE_ID,
            APPLE_M3_AIR_METAL_BACKEND,
            selected_backend,
            "metal",
            resolved_device,
            AppleRuntimeVisibility { metal_visible, mpsgraph_visible: None },
            fallback_used,
            artifact_path,
        )
    }

    #[must_use]
    pub fn m3_air_mpsgraph(
        selected_backend: Option<impl Into<String>>,
        resolved_device: AppleResolvedDevice,
        metal_visible: bool,
        mpsgraph_visible: bool,
        fallback_used: bool,
        artifact_path: impl Into<String>,
    ) -> Self {
        Self::new(
            APPLE_M3_AIR_MACHINE_ID,
            APPLE_M3_AIR_MPSGRAPH_BACKEND,
            selected_backend,
            "mpsgraph",
            resolved_device,
            AppleRuntimeVisibility { metal_visible, mpsgraph_visible: Some(mpsgraph_visible) },
            fallback_used,
            artifact_path,
        )
    }

    #[must_use]
    pub fn with_fallback_reason(mut self, fallback_reason: impl Into<String>) -> Self {
        self.fallback_reason = Some(fallback_reason.into());
        self
    }

    pub fn validate(&self) -> Result<(), AppleReceiptError> {
        require_nonempty("machine_id", &self.machine_id)?;
        require_nonempty("artifact_kind", &self.artifact_kind)?;
        require_nonempty("requested_backend", &self.requested_backend)?;
        require_nonempty("runtime_api", &self.runtime_api)?;
        require_nonempty("resolved_device.chip", &self.resolved_device.chip)?;
        require_nonempty("artifact_path", &self.artifact_path)?;
        self.claim_boundary.validate_bounded_preflight()?;
        validate_fallback(self.fallback_used, self.fallback_reason.as_deref())?;

        if self.machine_id == APPLE_M3_AIR_MACHINE_ID
            && !matches!(
                self.requested_backend.as_str(),
                APPLE_M3_AIR_METAL_BACKEND | APPLE_M3_AIR_MPSGRAPH_BACKEND
            )
        {
            return Err(AppleReceiptError::UnsupportedAppleBackend {
                machine_id: APPLE_M3_AIR_MACHINE_ID,
                requested_backend: self.requested_backend.clone(),
            });
        }
        if self.machine_id == APPLE_M3_AIR_MACHINE_ID {
            if let Some(selected_backend) = self.selected_backend.as_deref()
                && selected_backend != self.requested_backend
            {
                return Err(AppleReceiptError::UnsupportedAppleSelectedBackend {
                    machine_id: APPLE_M3_AIR_MACHINE_ID,
                    selected_backend: selected_backend.to_owned(),
                });
            }
            match self.requested_backend.as_str() {
                APPLE_M3_AIR_METAL_BACKEND if self.runtime_api != "metal" => {
                    return Err(AppleReceiptError::RuntimeApiMismatch {
                        requested_backend: APPLE_M3_AIR_METAL_BACKEND,
                        runtime_api: self.runtime_api.clone(),
                    });
                }
                APPLE_M3_AIR_MPSGRAPH_BACKEND if self.runtime_api != "mpsgraph" => {
                    return Err(AppleReceiptError::RuntimeApiMismatch {
                        requested_backend: APPLE_M3_AIR_MPSGRAPH_BACKEND,
                        runtime_api: self.runtime_api.clone(),
                    });
                }
                _ => {}
            }
        }

        if self.requested_backend == APPLE_M3_AIR_MPSGRAPH_BACKEND
            && self.mpsgraph_visible.is_none()
        {
            return Err(AppleReceiptError::MissingField("mpsgraph_visible"));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppleRuntimeVisibility {
    pub metal_visible: bool,
    pub mpsgraph_visible: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppleBackendReceipt {
    pub machine_id: String,
    pub artifact_kind: String,
    pub requested_backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_backend: Option<String>,
    pub runtime_api: String,
    pub resolved_device: AppleResolvedDevice,
    pub fallback_used: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    pub artifact_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolved_target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
}

impl AppleBackendReceipt {
    #[must_use]
    pub fn new(
        machine_id: impl Into<String>,
        artifact_kind: impl Into<String>,
        requested_backend: impl Into<String>,
        selected_backend: Option<impl Into<String>>,
        runtime_api: impl Into<String>,
        resolved_device: AppleResolvedDevice,
        fallback_used: bool,
        artifact_path: impl Into<String>,
    ) -> Self {
        Self {
            machine_id: machine_id.into(),
            artifact_kind: artifact_kind.into(),
            requested_backend: requested_backend.into(),
            selected_backend: selected_backend.map(Into::into),
            runtime_api: runtime_api.into(),
            resolved_device,
            fallback_used,
            fallback_reason: None,
            artifact_path: artifact_path.into(),
            kernel_id: None,
            graph_id: None,
            resolved_target: None,
            result: None,
        }
    }

    #[must_use]
    pub fn with_kernel_id(mut self, kernel_id: impl Into<String>) -> Self {
        self.kernel_id = Some(kernel_id.into());
        self
    }

    #[must_use]
    pub fn with_graph_id(mut self, graph_id: impl Into<String>) -> Self {
        self.graph_id = Some(graph_id.into());
        self
    }

    #[must_use]
    pub fn with_resolved_target(mut self, resolved_target: impl Into<String>) -> Self {
        self.resolved_target = Some(resolved_target.into());
        self
    }

    #[must_use]
    pub fn with_result(mut self, result: impl Into<String>) -> Self {
        self.result = Some(result.into());
        self
    }

    #[must_use]
    pub fn with_fallback_reason(mut self, fallback_reason: impl Into<String>) -> Self {
        self.fallback_reason = Some(fallback_reason.into());
        self
    }

    pub fn validate(&self) -> Result<(), AppleReceiptError> {
        require_nonempty("machine_id", &self.machine_id)?;
        require_nonempty("artifact_kind", &self.artifact_kind)?;
        require_nonempty("requested_backend", &self.requested_backend)?;
        require_nonempty("runtime_api", &self.runtime_api)?;
        require_nonempty("resolved_device.chip", &self.resolved_device.chip)?;
        require_nonempty("artifact_path", &self.artifact_path)?;

        validate_fallback(self.fallback_used, self.fallback_reason.as_deref())?;
        if self.kernel_id.is_some() && self.graph_id.is_some() {
            return Err(AppleReceiptError::AmbiguousWorkId);
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppleReceiptError {
    MissingField(&'static str),
    MissingFallbackReason,
    UnexpectedFallbackReason,
    AmbiguousWorkId,
    ClaimBoundaryViolation(&'static str),
    UnsupportedAppleMachine { machine_id: String },
    InvalidProfileField(&'static str),
    UnsupportedAppleBackend { machine_id: &'static str, requested_backend: String },
    UnsupportedAppleSelectedBackend { machine_id: &'static str, selected_backend: String },
    RuntimeApiMismatch { requested_backend: &'static str, runtime_api: String },
}

impl fmt::Display for AppleReceiptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(field) => write!(f, "Apple backend receipt missing {field}"),
            Self::MissingFallbackReason => {
                write!(f, "Apple backend receipt fallback_used=true requires fallback_reason")
            }
            Self::UnexpectedFallbackReason => {
                write!(
                    f,
                    "Apple backend receipt fallback_reason must be absent when fallback_used=false"
                )
            }
            Self::AmbiguousWorkId => {
                write!(f, "Apple backend receipt must not record both kernel_id and graph_id")
            }
            Self::ClaimBoundaryViolation(field) => {
                write!(f, "Apple visibility preflight must not claim {field}")
            }
            Self::UnsupportedAppleMachine { machine_id } => {
                write!(f, "Apple profile contract does not support machine {machine_id}")
            }
            Self::InvalidProfileField(field) => {
                write!(f, "Apple profile contract has invalid field {field}")
            }
            Self::UnsupportedAppleBackend { machine_id, requested_backend } => write!(
                f,
                "Apple visibility preflight for {machine_id} must use an explicit M3 Air backend, got {requested_backend}"
            ),
            Self::UnsupportedAppleSelectedBackend { machine_id, selected_backend } => write!(
                f,
                "Apple visibility preflight for {machine_id} must not select generic or cross-lane backend {selected_backend}"
            ),
            Self::RuntimeApiMismatch { requested_backend, runtime_api } => write!(
                f,
                "Apple visibility preflight requested backend {requested_backend} does not match runtime API {runtime_api}"
            ),
        }
    }
}

impl std::error::Error for AppleReceiptError {}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), AppleReceiptError> {
    if value.trim().is_empty() { Err(AppleReceiptError::MissingField(field)) } else { Ok(()) }
}

fn validate_fallback(
    fallback_used: bool,
    fallback_reason: Option<&str>,
) -> Result<(), AppleReceiptError> {
    if fallback_used && fallback_reason.unwrap_or_default().trim().is_empty() {
        return Err(AppleReceiptError::MissingFallbackReason);
    }
    if !fallback_used && fallback_reason.is_some() {
        return Err(AppleReceiptError::UnexpectedFallbackReason);
    }
    Ok(())
}
