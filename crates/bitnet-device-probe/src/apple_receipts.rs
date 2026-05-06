//! Durable Apple backend receipt fields.
//!
//! These types record Apple proof identity without collapsing Metal, `MPSGraph`,
//! and CPU/NEON evidence. They do not prove `BitNet` inference on their own.

use serde::{Deserialize, Serialize};
use std::fmt;

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

        if self.fallback_used && self.fallback_reason.as_deref().unwrap_or_default().is_empty() {
            return Err(AppleReceiptError::MissingFallbackReason);
        }
        if !self.fallback_used && self.fallback_reason.is_some() {
            return Err(AppleReceiptError::UnexpectedFallbackReason);
        }
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
        }
    }
}

impl std::error::Error for AppleReceiptError {}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), AppleReceiptError> {
    if value.trim().is_empty() { Err(AppleReceiptError::MissingField(field)) } else { Ok(()) }
}
