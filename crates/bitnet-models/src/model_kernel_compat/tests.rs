use super::*;

#[test]
fn official_2b_x86_i2s_is_reference_supported() {
    let support =
        model_kernel_support("microsoft/BitNet-b1.58-2B-4T-gguf", HostArch::X86, BitnetKernel::I2S);

    assert_eq!(support, ModelKernelSupport::SupportedReference);
}

#[test]
fn three_b_x86_i2s_is_upstream_unsupported() {
    let support = model_kernel_support("1bitLLM/bitnet_b1_58-3B", HostArch::X86, BitnetKernel::I2S);

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
