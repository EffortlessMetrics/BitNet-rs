//! Claim evaluation and decision reporting.

use super::support::model_kernel_support;
use super::types::{BitnetKernel, HostArch, ModelKernelSupport};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityClaim {
    DiagnosticRun,
    ArtifactInspection,
    UnsupportedPathReceipt,
    AnswerReady,
    ReferenceAuthority,
    BackendParity,
    Speedup,
}

impl CompatibilityClaim {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DiagnosticRun => "diagnostic_run",
            Self::ArtifactInspection => "artifact_inspection",
            Self::UnsupportedPathReceipt => "unsupported_path_receipt",
            Self::AnswerReady => "answer_ready",
            Self::ReferenceAuthority => "reference_authority",
            Self::BackendParity => "backend_parity",
            Self::Speedup => "speedup",
        }
    }

    fn is_diagnostic_only(&self) -> bool {
        matches!(
            self,
            Self::DiagnosticRun | Self::ArtifactInspection | Self::UnsupportedPathReceipt
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatibilityDecision {
    pub allowed: bool,
    pub support: ModelKernelSupport,
    pub reason: String,
}

pub fn evaluate_model_kernel_claim(
    model_id: &str,
    arch: HostArch,
    kernel: BitnetKernel,
    claim: CompatibilityClaim,
) -> CompatibilityDecision {
    let support = model_kernel_support(model_id, arch, kernel);
    let allowed = match support {
        ModelKernelSupport::SupportedReference | ModelKernelSupport::Supported => true,
        ModelKernelSupport::ListedSupportedVerifyRunner => claim.is_diagnostic_only(),
        ModelKernelSupport::UnsupportedUpstream | ModelKernelSupport::Unknown => {
            claim.is_diagnostic_only()
        }
    };

    CompatibilityDecision {
        allowed,
        support,
        reason: decision_reason(model_id, arch, kernel, claim, support, allowed),
    }
}

fn decision_reason(
    model_id: &str,
    arch: HostArch,
    kernel: BitnetKernel,
    claim: CompatibilityClaim,
    support: ModelKernelSupport,
    allowed: bool,
) -> String {
    if allowed && claim.is_diagnostic_only() {
        return format!(
            "{} {} {} may be used for {} with claim=false",
            model_id,
            arch.as_str(),
            kernel.as_str(),
            claim.as_str()
        );
    }
    if allowed {
        return format!(
            "{} {} {} is {}; artifact, receipt, and benchmark gates still apply before {} can be claimed",
            model_id,
            arch.as_str(),
            kernel.as_str(),
            support.as_str(),
            claim.as_str()
        );
    }

    match support {
        ModelKernelSupport::UnsupportedUpstream => format!(
            "{} {} {} is unsupported upstream and cannot be used for {}",
            model_id,
            arch.as_str(),
            kernel.as_str(),
            claim.as_str()
        ),
        ModelKernelSupport::ListedSupportedVerifyRunner => format!(
            "{} {} {} is listed upstream but still needs runner-path verification before {}",
            model_id,
            arch.as_str(),
            kernel.as_str(),
            claim.as_str()
        ),
        ModelKernelSupport::Unknown => format!(
            "{} {} {} has no compatibility authority and cannot be used for {}",
            model_id,
            arch.as_str(),
            kernel.as_str(),
            claim.as_str()
        ),
        ModelKernelSupport::SupportedReference | ModelKernelSupport::Supported => format!(
            "{} {} {} is {}; artifact, receipt, and benchmark gates still apply before {} can be claimed",
            model_id,
            arch.as_str(),
            kernel.as_str(),
            support.as_str(),
            claim.as_str()
        ),
    }
}
