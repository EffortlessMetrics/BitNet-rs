//! BitNet-family model contract registry.
//!
//! The contract registry is an authority surface, not a runtime dispatcher. It
//! records which BitNet-family artifacts may support answer, parity, benchmark,
//! or diagnostic claims before a backend attempts to turn a loaded model into
//! proof evidence.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactFormat {
    Gguf,
    Unknown,
}

impl ArtifactFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitnetKernelFamily {
    I2sQk256,
    Tl1Lut,
    Tl2Lut,
    UnsupportedI2s,
}

impl BitnetKernelFamily {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::I2sQk256 => "i2_s_qk256",
            Self::Tl1Lut => "tl1_lut",
            Self::Tl2Lut => "tl2_lut",
            Self::UnsupportedI2s => "unsupported_i2_s",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractClaim {
    DiagnosticRun,
    ArtifactInspection,
    UnsupportedPathReceipt,
    AnswerReady,
    ReferenceAuthority,
    BackendParity,
    BenchmarkBaseline,
    SpeedupQualified,
    FullResidency,
}

impl ContractClaim {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DiagnosticRun => "diagnostic_run",
            Self::ArtifactInspection => "artifact_inspection",
            Self::UnsupportedPathReceipt => "unsupported_path_receipt",
            Self::AnswerReady => "answer_ready",
            Self::ReferenceAuthority => "reference_authority",
            Self::BackendParity => "backend_parity",
            Self::BenchmarkBaseline => "benchmark_baseline",
            Self::SpeedupQualified => "speedup_qualified",
            Self::FullResidency => "full_residency",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractStatus {
    ReferenceReady,
    PlannedProofRequired,
    UpstreamUnsupported,
    ListedVerifyRunner,
    AlternateControl,
}

impl ContractStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ReferenceReady => "reference_ready",
            Self::PlannedProofRequired => "planned_proof_required",
            Self::UpstreamUnsupported => "upstream_unsupported",
            Self::ListedVerifyRunner => "listed_verify_runner",
            Self::AlternateControl => "alternate_control",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchitectureSupport {
    pub arch: &'static str,
    pub kernel: &'static str,
    pub status: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceleratorRoute {
    pub backend: &'static str,
    pub route: &'static str,
    pub status: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitnetModelContract {
    pub id: &'static str,
    pub aliases: &'static [&'static str],
    pub model_family: &'static str,
    pub artifact_format: ArtifactFormat,
    pub artifact_id: Option<&'static str>,
    pub kernel_family: BitnetKernelFamily,
    pub status: ContractStatus,
    pub architecture_support: &'static [ArchitectureSupport],
    pub tokenizer_authority: &'static str,
    pub prompt_authority: &'static str,
    pub cpu_oracle: &'static str,
    pub accelerator_routes: &'static [AcceleratorRoute],
    pub permitted_claims: &'static [ContractClaim],
    pub required_receipts: &'static [&'static str],
    pub claim_boundary: &'static str,
}

impl BitnetModelContract {
    pub fn permits_claim(&self, claim: ContractClaim) -> bool {
        self.permitted_claims.contains(&claim)
    }

    pub fn requires_receipt(&self, receipt: &str) -> bool {
        self.required_receipts.contains(&receipt)
    }
}

const OFFICIAL_2B_I2S_ALIASES: &[&str] = &[
    "microsoft/BitNet-b1.58-2B-4T",
    "microsoft/bitnet-b1.58-2B-4T",
    "microsoft/bitnet-b1.58-2B-4T-gguf",
    "microsoft_bitnet_b158_2b_4t_gguf_i2s_current",
    "ggml-model-i2_s.gguf",
];

const OFFICIAL_2B_TL1_ALIASES: &[&str] = &["microsoft/BitNet-b1.58-2B-4T:tl1", "microsoft_2b_tl1"];
const OFFICIAL_2B_TL2_ALIASES: &[&str] = &["microsoft/BitNet-b1.58-2B-4T:tl2", "microsoft_2b_tl2"];
const BITNET_3B_I2S_X86_ALIASES: &[&str] =
    &["1bitLLM/bitnet_b1_58-3B:i2_s:x86", "bitnet_3b_i2s_x86"];
const BITNET_3B_TL2_X86_ALIASES: &[&str] =
    &["1bitLLM/bitnet_b1_58-3B:tl2:x86", "bitnet_3b_tl2_x86"];
const BITNET_3B_TL1_ARM_ALIASES: &[&str] =
    &["1bitLLM/bitnet_b1_58-3B:tl1:arm", "bitnet_3b_tl1_arm"];
const TDH111_IQ2_BN_R4_ALIASES: &[&str] =
    &["tdh111_bitnet_b158_2b_4t_iq2_bn_r4", "tdh111/bitnet-b1.58-2b-4t-iq2_bn_r4"];

const PROOF_CLAIMS: &[ContractClaim] = &[
    ContractClaim::DiagnosticRun,
    ContractClaim::ArtifactInspection,
    ContractClaim::AnswerReady,
    ContractClaim::ReferenceAuthority,
    ContractClaim::BackendParity,
    ContractClaim::BenchmarkBaseline,
];

const DIAGNOSTIC_CLAIMS: &[ContractClaim] = &[
    ContractClaim::DiagnosticRun,
    ContractClaim::ArtifactInspection,
    ContractClaim::UnsupportedPathReceipt,
];

const ALT_CONTROL_CLAIMS: &[ContractClaim] = &[
    ContractClaim::DiagnosticRun,
    ContractClaim::ArtifactInspection,
    ContractClaim::BackendParity,
];

const OFFICIAL_2B_I2S_ARCH: &[ArchitectureSupport] = &[
    ArchitectureSupport { arch: "x86", kernel: "i2_s", status: "supported_reference" },
    ArchitectureSupport { arch: "arm", kernel: "i2_s", status: "supported" },
];

const OFFICIAL_2B_TL1_ARCH: &[ArchitectureSupport] = &[
    ArchitectureSupport { arch: "arm", kernel: "tl1", status: "supported_proof_required" },
    ArchitectureSupport { arch: "x86", kernel: "tl1", status: "unsupported_upstream" },
];

const OFFICIAL_2B_TL2_ARCH: &[ArchitectureSupport] = &[
    ArchitectureSupport { arch: "x86", kernel: "tl2", status: "supported_proof_required" },
    ArchitectureSupport { arch: "arm", kernel: "tl2", status: "unsupported_upstream" },
];

const BITNET_3B_I2S_X86_ARCH: &[ArchitectureSupport] =
    &[ArchitectureSupport { arch: "x86", kernel: "i2_s", status: "unsupported_upstream" }];

const BITNET_3B_TL2_X86_ARCH: &[ArchitectureSupport] =
    &[ArchitectureSupport { arch: "x86", kernel: "tl2", status: "listed_supported_verify_runner" }];

const BITNET_3B_TL1_ARM_ARCH: &[ArchitectureSupport] =
    &[ArchitectureSupport { arch: "arm", kernel: "tl1", status: "listed_supported_verify_runner" }];

const OFFICIAL_2B_I2S_ACCEL: &[AcceleratorRoute] = &[
    AcceleratorRoute {
        backend: "cpu-scalar",
        route: "bitnet_i2s_qk256_cpu_scalar",
        status: "oracle",
    },
    AcceleratorRoute {
        backend: "cpu-avx2",
        route: "bitnet_i2s_qk256_cpu_avx2",
        status: "parity_lane",
    },
    AcceleratorRoute {
        backend: "cpu-avx512",
        route: "bitnet_i2s_qk256_cpu_avx512",
        status: "parity_lane",
    },
    AcceleratorRoute {
        backend: "nvidia-rtx-5070-ti-cuda",
        route: "bitnet_qk256_cuda",
        status: "strict_receipt_lane",
    },
    AcceleratorRoute {
        backend: "intel-arc-a770-opencl",
        route: "a770.bitnet.i2s.qk256",
        status: "diagnostic_qk256_route_receipt_only",
    },
];

const EMPTY_ACCEL: &[AcceleratorRoute] = &[];

const OFFICIAL_2B_I2S_RECEIPTS: &[&str] = &[
    "answer_artifact_gate",
    "prompt_authority_audit",
    "strict_answer_corpus",
    "cpu_cuda_answer_parity",
    "execution_plan",
    "fallback_free_backend_receipt",
    "benchmark_profile_receipt",
];

const PROOF_REQUIRED_RECEIPTS: &[&str] = &[
    "runner_path_verification",
    "prompt_authority_audit",
    "strict_answer_corpus",
    "backend_parity_receipt",
];

const UNSUPPORTED_RECEIPTS: &[&str] = &["unsupported_path_receipt"];

pub static BITNET_MODEL_CONTRACTS: &[BitnetModelContract] = &[
    BitnetModelContract {
        id: "microsoft_bitnet_b158_2b_4t_i2s",
        aliases: OFFICIAL_2B_I2S_ALIASES,
        model_family: "bitnet_b1_58",
        artifact_format: ArtifactFormat::Gguf,
        artifact_id: Some("microsoft_bitnet_b158_2b_4t_gguf_i2s_current"),
        kernel_family: BitnetKernelFamily::I2sQk256,
        status: ContractStatus::ReferenceReady,
        architecture_support: OFFICIAL_2B_I2S_ARCH,
        tokenizer_authority: "external_llama_bpe",
        prompt_authority: "bitnetcpp-answer",
        cpu_oracle: "x86_cpu_scalar_then_avx512_parity",
        accelerator_routes: OFFICIAL_2B_I2S_ACCEL,
        permitted_claims: PROOF_CLAIMS,
        required_receipts: OFFICIAL_2B_I2S_RECEIPTS,
        claim_boundary: "Reference lane for x86 CPU and RTX 5070 Ti CUDA, but speedup and full-residency claims still require profile-specific receipts.",
    },
    BitnetModelContract {
        id: "microsoft_bitnet_b158_2b_4t_tl1",
        aliases: OFFICIAL_2B_TL1_ALIASES,
        model_family: "bitnet_b1_58",
        artifact_format: ArtifactFormat::Gguf,
        artifact_id: None,
        kernel_family: BitnetKernelFamily::Tl1Lut,
        status: ContractStatus::PlannedProofRequired,
        architecture_support: OFFICIAL_2B_TL1_ARCH,
        tokenizer_authority: "pending_lane_audit",
        prompt_authority: "pending_lane_audit",
        cpu_oracle: "arm_neon_or_reference_runner_required",
        accelerator_routes: EMPTY_ACCEL,
        permitted_claims: DIAGNOSTIC_CLAIMS,
        required_receipts: PROOF_REQUIRED_RECEIPTS,
        claim_boundary: "ARM TL1 is upstream-supported for the official 2B family, but BitNet-rs needs parser, fixture, prompt authority, answer corpus, and backend receipts before proof claims.",
    },
    BitnetModelContract {
        id: "microsoft_bitnet_b158_2b_4t_tl2",
        aliases: OFFICIAL_2B_TL2_ALIASES,
        model_family: "bitnet_b1_58",
        artifact_format: ArtifactFormat::Gguf,
        artifact_id: None,
        kernel_family: BitnetKernelFamily::Tl2Lut,
        status: ContractStatus::PlannedProofRequired,
        architecture_support: OFFICIAL_2B_TL2_ARCH,
        tokenizer_authority: "pending_lane_audit",
        prompt_authority: "pending_lane_audit",
        cpu_oracle: "x86_scalar_then_avx_parity_required",
        accelerator_routes: EMPTY_ACCEL,
        permitted_claims: DIAGNOSTIC_CLAIMS,
        required_receipts: PROOF_REQUIRED_RECEIPTS,
        claim_boundary: "x86 TL2 is upstream-supported for the official 2B family, but it is not the current I2_S/QK256 CUDA target and needs its own proof lane.",
    },
    BitnetModelContract {
        id: "onebitllm_bitnet_b158_3b_i2s_x86",
        aliases: BITNET_3B_I2S_X86_ALIASES,
        model_family: "bitnet_b1_58",
        artifact_format: ArtifactFormat::Gguf,
        artifact_id: None,
        kernel_family: BitnetKernelFamily::UnsupportedI2s,
        status: ContractStatus::UpstreamUnsupported,
        architecture_support: BITNET_3B_I2S_X86_ARCH,
        tokenizer_authority: "not_authoritative_for_claims",
        prompt_authority: "not_authoritative_for_claims",
        cpu_oracle: "none",
        accelerator_routes: EMPTY_ACCEL,
        permitted_claims: DIAGNOSTIC_CLAIMS,
        required_receipts: UNSUPPORTED_RECEIPTS,
        claim_boundary: "3B x86 I2_S is upstream-unsupported and cannot be answer, reference, backend-parity, or speed authority.",
    },
    BitnetModelContract {
        id: "onebitllm_bitnet_b158_3b_tl2_x86",
        aliases: BITNET_3B_TL2_X86_ALIASES,
        model_family: "bitnet_b1_58",
        artifact_format: ArtifactFormat::Gguf,
        artifact_id: None,
        kernel_family: BitnetKernelFamily::Tl2Lut,
        status: ContractStatus::ListedVerifyRunner,
        architecture_support: BITNET_3B_TL2_X86_ARCH,
        tokenizer_authority: "pending_runner_verification",
        prompt_authority: "pending_runner_verification",
        cpu_oracle: "x86_tl2_runner_verification_required",
        accelerator_routes: EMPTY_ACCEL,
        permitted_claims: DIAGNOSTIC_CLAIMS,
        required_receipts: PROOF_REQUIRED_RECEIPTS,
        claim_boundary: "3B x86 TL2 is listed upstream but must verify the runner path before proof claims.",
    },
    BitnetModelContract {
        id: "onebitllm_bitnet_b158_3b_tl1_arm",
        aliases: BITNET_3B_TL1_ARM_ALIASES,
        model_family: "bitnet_b1_58",
        artifact_format: ArtifactFormat::Gguf,
        artifact_id: None,
        kernel_family: BitnetKernelFamily::Tl1Lut,
        status: ContractStatus::ListedVerifyRunner,
        architecture_support: BITNET_3B_TL1_ARM_ARCH,
        tokenizer_authority: "pending_runner_verification",
        prompt_authority: "pending_runner_verification",
        cpu_oracle: "arm_tl1_runner_verification_required",
        accelerator_routes: EMPTY_ACCEL,
        permitted_claims: DIAGNOSTIC_CLAIMS,
        required_receipts: PROOF_REQUIRED_RECEIPTS,
        claim_boundary: "3B ARM TL1 is listed upstream but must verify the runner path before proof claims.",
    },
    BitnetModelContract {
        id: "tdh111_bitnet_b158_2b_4t_iq2_bn_r4",
        aliases: TDH111_IQ2_BN_R4_ALIASES,
        model_family: "bitnet_b1_58",
        artifact_format: ArtifactFormat::Gguf,
        artifact_id: Some("tdh111_bitnet_b158_2b_4t_iq2_bn_r4"),
        kernel_family: BitnetKernelFamily::I2sQk256,
        status: ContractStatus::AlternateControl,
        architecture_support: &[],
        tokenizer_authority: "missing_pretokenizer_authority",
        prompt_authority: "ik_llama_intended_runner",
        cpu_oracle: "alternate_quant_control_only",
        accelerator_routes: EMPTY_ACCEL,
        permitted_claims: ALT_CONTROL_CLAIMS,
        required_receipts: PROOF_REQUIRED_RECEIPTS,
        claim_boundary: "tdh111 IQ2_BN_R4 is useful alternate-quant control evidence but cannot unblock official Microsoft I2_S CUDA readiness.",
    },
];

pub fn bitnet_model_contracts() -> &'static [BitnetModelContract] {
    BITNET_MODEL_CONTRACTS
}

pub fn find_bitnet_model_contract(label: &str) -> Option<&'static BitnetModelContract> {
    let label = normalize_label(label);
    BITNET_MODEL_CONTRACTS.iter().find(|contract| {
        normalize_label(contract.id) == label
            || contract.aliases.iter().any(|alias| normalize_label(alias) == label)
    })
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
    fn official_2b_i2s_contract_is_cuda_reference_lane() {
        let contract = find_bitnet_model_contract("microsoft/bitnet-b1.58-2B-4T-gguf")
            .expect("official 2B I2_S contract");

        assert_eq!(contract.status, ContractStatus::ReferenceReady);
        assert_eq!(contract.kernel_family, BitnetKernelFamily::I2sQk256);
        assert_eq!(contract.tokenizer_authority, "external_llama_bpe");
        assert!(contract.permits_claim(ContractClaim::AnswerReady));
        assert!(contract.permits_claim(ContractClaim::BackendParity));
        assert!(!contract.permits_claim(ContractClaim::SpeedupQualified));
        assert!(contract.accelerator_routes.iter().any(|route| route.route == "bitnet_qk256_cuda"));
        assert!(contract.accelerator_routes.iter().any(|route| {
            route.backend == "intel-arc-a770-opencl"
                && route.route == "a770.bitnet.i2s.qk256"
                && route.status == "diagnostic_qk256_route_receipt_only"
        }));
        assert!(contract.requires_receipt("execution_plan"));
    }

    #[test]
    fn official_tl_lanes_are_diagnostic_until_their_own_proofs_exist() {
        for label in ["microsoft/BitNet-b1.58-2B-4T:tl1", "microsoft/BitNet-b1.58-2B-4T:tl2"] {
            let contract = find_bitnet_model_contract(label).expect("official TL contract");
            assert_eq!(contract.status, ContractStatus::PlannedProofRequired);
            assert!(contract.permits_claim(ContractClaim::DiagnosticRun));
            assert!(!contract.permits_claim(ContractClaim::AnswerReady));
            assert!(!contract.permits_claim(ContractClaim::ReferenceAuthority));
        }
    }

    #[test]
    fn three_b_x86_i2s_contract_cannot_be_proof_authority() {
        let contract =
            find_bitnet_model_contract("1bitLLM/bitnet_b1_58-3B:i2_s:x86").expect("3B I2_S");

        assert_eq!(contract.status, ContractStatus::UpstreamUnsupported);
        assert_eq!(contract.cpu_oracle, "none");
        assert!(contract.permits_claim(ContractClaim::UnsupportedPathReceipt));
        assert!(!contract.permits_claim(ContractClaim::AnswerReady));
        assert!(!contract.permits_claim(ContractClaim::BackendParity));
        assert!(!contract.permits_claim(ContractClaim::SpeedupQualified));
    }

    #[test]
    fn listed_3b_routes_require_runner_verification_before_claims() {
        for label in ["1bitLLM/bitnet_b1_58-3B:tl2:x86", "1bitLLM/bitnet_b1_58-3B:tl1:arm"] {
            let contract = find_bitnet_model_contract(label).expect("listed 3B contract");
            assert_eq!(contract.status, ContractStatus::ListedVerifyRunner);
            assert!(contract.permits_claim(ContractClaim::DiagnosticRun));
            assert!(!contract.permits_claim(ContractClaim::ReferenceAuthority));
            assert!(contract.requires_receipt("runner_path_verification"));
        }
    }

    #[test]
    fn alternate_quant_control_cannot_unblock_official_cuda_target() {
        let contract =
            find_bitnet_model_contract("tdh111_bitnet_b158_2b_4t_iq2_bn_r4").expect("tdh111");

        assert_eq!(contract.status, ContractStatus::AlternateControl);
        assert!(contract.permits_claim(ContractClaim::BackendParity));
        assert!(!contract.permits_claim(ContractClaim::AnswerReady));
        assert!(contract.accelerator_routes.is_empty());
    }

    #[test]
    fn every_contract_has_claim_boundary_and_receipt_requirements() {
        for contract in bitnet_model_contracts() {
            assert!(
                !contract.claim_boundary.trim().is_empty(),
                "{} lacks a claim boundary",
                contract.id
            );
            assert!(
                !contract.required_receipts.is_empty(),
                "{} lacks receipt requirements",
                contract.id
            );
            assert!(!contract.permitted_claims.is_empty(), "{} lacks claim policy", contract.id);
        }
    }
}
