//! Upstream model/kernel support matrix.

use super::labels::{is_1bitllm_3b, is_official_microsoft_2b, normalize_label};
use super::types::{BitnetKernel, HostArch, ModelKernelSupport};

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
