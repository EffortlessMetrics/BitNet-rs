//! Model/kernel compatibility claim policy.
//!
//! This module records upstream model/kernel support boundaries that are
//! independent from local Rust loader or kernel correctness. It prevents known
//! unsupported combinations from becoming answer, reference, parity, or
//! benchmark authorities while still allowing diagnostic receipts.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostArch {
    X86,
    Arm,
    Unknown,
}

impl HostArch {
    pub fn from_label(label: &str) -> Self {
        let label = normalize_label(label);
        if label.contains("x86") || label.contains("amd64") || label.contains("x64") {
            Self::X86
        } else if label.contains("arm") || label.contains("aarch64") {
            Self::Arm
        } else {
            Self::Unknown
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::X86 => "x86",
            Self::Arm => "arm",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitnetKernel {
    I2S,
    Tl1,
    Tl2,
    Unknown,
}

impl BitnetKernel {
    pub fn from_label(label: &str) -> Self {
        let label = normalize_label(label);
        if label.contains("i2_s") || label.contains("i2s") {
            Self::I2S
        } else if label.contains("tl1") {
            Self::Tl1
        } else if label.contains("tl2") {
            Self::Tl2
        } else {
            Self::Unknown
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::I2S => "i2_s",
            Self::Tl1 => "tl1",
            Self::Tl2 => "tl2",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKernelSupport {
    SupportedReference,
    Supported,
    ListedSupportedVerifyRunner,
    UnsupportedUpstream,
    Unknown,
}

impl ModelKernelSupport {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SupportedReference => "supported_reference",
            Self::Supported => "supported",
            Self::ListedSupportedVerifyRunner => "listed_supported_verify_runner",
            Self::UnsupportedUpstream => "unsupported_upstream",
            Self::Unknown => "unknown",
        }
    }
}

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

pub fn model_kernel_support(
    model_id: &str,
    arch: HostArch,
    kernel: BitnetKernel,
) -> ModelKernelSupport {
    let model = normalize_label(model_id);
    if is_official_microsoft_2b(&model) {
        return official_microsoft_2b_support(arch, kernel);
    }
    if is_1bitllm_3b(&model) {
        return bitnet_3b_support(arch, kernel);
    }
    ModelKernelSupport::Unknown
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

fn official_microsoft_2b_support(arch: HostArch, kernel: BitnetKernel) -> ModelKernelSupport {
    match (arch, kernel) {
        (HostArch::X86, BitnetKernel::I2S) => ModelKernelSupport::SupportedReference,
        (HostArch::X86, BitnetKernel::Tl2) => ModelKernelSupport::Supported,
        (HostArch::Arm, BitnetKernel::I2S | BitnetKernel::Tl1) => ModelKernelSupport::Supported,
        (HostArch::X86, BitnetKernel::Tl1)
        | (HostArch::Arm, BitnetKernel::Tl2)
        | (_, BitnetKernel::Unknown)
        | (HostArch::Unknown, _) => ModelKernelSupport::UnsupportedUpstream,
    }
}

fn bitnet_3b_support(arch: HostArch, kernel: BitnetKernel) -> ModelKernelSupport {
    match (arch, kernel) {
        (HostArch::X86, BitnetKernel::Tl2) | (HostArch::Arm, BitnetKernel::Tl1) => {
            ModelKernelSupport::ListedSupportedVerifyRunner
        }
        (HostArch::X86, BitnetKernel::I2S | BitnetKernel::Tl1)
        | (HostArch::Arm, BitnetKernel::I2S | BitnetKernel::Tl2)
        | (_, BitnetKernel::Unknown)
        | (HostArch::Unknown, _) => ModelKernelSupport::UnsupportedUpstream,
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
        ModelKernelSupport::SupportedReference | ModelKernelSupport::Supported => {
            unreachable!("supported routes are allowed at the compatibility layer")
        }
    }
}

fn is_official_microsoft_2b(model: &str) -> bool {
    model.contains("microsoft_bitnet_b1_58_2b_4t") || model.contains("microsoft_bitnet_b158_2b_4t")
}

fn is_1bitllm_3b(model: &str) -> bool {
    model.contains("1bitllm_bitnet_b1_58_3b") || model.contains("bitnet_b1_58_3b")
}

fn normalize_label(label: &str) -> String {
    label
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_2b_x86_i2s_is_reference_supported() {
        let support = model_kernel_support(
            "microsoft/BitNet-b1.58-2B-4T-gguf",
            HostArch::X86,
            BitnetKernel::I2S,
        );

        assert_eq!(support, ModelKernelSupport::SupportedReference);
    }

    #[test]
    fn three_b_x86_i2s_is_upstream_unsupported() {
        let support =
            model_kernel_support("1bitLLM/bitnet_b1_58-3B", HostArch::X86, BitnetKernel::I2S);

        assert_eq!(support, ModelKernelSupport::UnsupportedUpstream);
    }

    #[test]
    fn three_b_x86_i2s_rejects_proof_claims() {
        for claim in [
            CompatibilityClaim::AnswerReady,
            CompatibilityClaim::ReferenceAuthority,
            CompatibilityClaim::BackendParity,
            CompatibilityClaim::Speedup,
        ] {
            let decision = evaluate_model_kernel_claim(
                "1bitLLM/bitnet_b1_58-3B",
                HostArch::X86,
                BitnetKernel::I2S,
                claim,
            );

            assert!(!decision.allowed, "{claim:?} must be rejected");
            assert_eq!(decision.support, ModelKernelSupport::UnsupportedUpstream);
            assert!(decision.reason.contains("unsupported upstream"));
        }
    }

    #[test]
    fn three_b_x86_i2s_allows_diagnostic_claims_only() {
        for claim in [
            CompatibilityClaim::DiagnosticRun,
            CompatibilityClaim::ArtifactInspection,
            CompatibilityClaim::UnsupportedPathReceipt,
        ] {
            let decision = evaluate_model_kernel_claim(
                "1bitLLM/bitnet_b1_58-3B",
                HostArch::X86,
                BitnetKernel::I2S,
                claim,
            );

            assert!(decision.allowed, "{claim:?} should be allowed");
            assert_eq!(decision.support, ModelKernelSupport::UnsupportedUpstream);
        }
    }

    #[test]
    fn three_b_x86_tl2_requires_runner_verification_before_authority_claims() {
        let decision = evaluate_model_kernel_claim(
            "1bitLLM/bitnet_b1_58-3B",
            HostArch::X86,
            BitnetKernel::Tl2,
            CompatibilityClaim::ReferenceAuthority,
        );

        assert!(!decision.allowed);
        assert_eq!(decision.support, ModelKernelSupport::ListedSupportedVerifyRunner);
        assert!(decision.reason.contains("needs runner-path verification"));
    }

    #[test]
    fn label_parsing_accepts_common_arch_and_kernel_aliases() {
        assert_eq!(HostArch::from_label("x86_64"), HostArch::X86);
        assert_eq!(HostArch::from_label("amd64"), HostArch::X86);
        assert_eq!(HostArch::from_label("aarch64"), HostArch::Arm);
        assert_eq!(BitnetKernel::from_label("I2_S"), BitnetKernel::I2S);
        assert_eq!(BitnetKernel::from_label("TL2"), BitnetKernel::Tl2);
    }
}
