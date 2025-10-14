//! Device Capability Mocks for Issue #453 Strict Quantization Guards
//!
//! Provides realistic device capability mocks for testing GPU/CPU kernel
//! availability, SIMD feature detection, and fallback scenarios.
//!
//! All device mocks support feature-gated compilation for CPU/GPU testing.

#![allow(dead_code)]

/// Mock GPU device with compute capability
#[derive(Debug, Clone)]
pub struct MockGpuDevice {
    pub device_id: u32,
    pub name: &'static str,
    pub compute_capability: (u32, u32), // (major, minor)
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub supports_tensor_cores: bool,
    pub max_threads_per_block: u32,
    pub max_shared_memory_per_block: usize,
    pub available: bool,
}

/// Mock CPU device with SIMD features
#[derive(Debug, Clone)]
pub struct MockCpuDevice {
    pub device_id: u32,
    pub name: &'static str,
    pub architecture: CpuArchitecture,
    pub supports_avx2: bool,
    pub supports_avx512: bool,
    pub supports_neon: bool,
    pub supports_sve: bool,
    pub num_cores: u32,
    pub available: bool,
}

/// CPU architecture enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuArchitecture {
    X86_64,
    Aarch64,
    Unknown,
}

/// Fallback trigger scenario
#[derive(Debug, Clone)]
pub struct FallbackScenario {
    pub scenario_name: &'static str,
    pub trigger_reason: FallbackTrigger,
    pub expected_strict_mode_behavior: StrictModeBehavior,
    pub description: &'static str,
}

/// Fallback trigger reason
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackTrigger {
    KernelUnavailable,
    DeviceMismatch,
    UnsupportedDimensions,
    InsufficientMemory,
    ComputeCapabilityTooLow,
    MissingSimdFeatures,
}

/// Expected strict mode behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrictModeBehavior {
    RejectWithError,
    AllowWithWarning,
    AllowSilently,
}

// ============================================================================
// NVIDIA GPU Device Mocks (Compute Capabilities)
// ============================================================================

/// NVIDIA GTX 1080 Ti (Pascal - Compute 6.1)
#[cfg(feature = "gpu")]
pub fn nvidia_gtx_1080ti() -> MockGpuDevice {
    MockGpuDevice {
        device_id: 0,
        name: "NVIDIA GeForce GTX 1080 Ti",
        compute_capability: (6, 1),
        supports_fp16: true,
        supports_bf16: false, // BF16 requires Ampere (8.0+)
        supports_tensor_cores: false,
        max_threads_per_block: 1024,
        max_shared_memory_per_block: 49152, // 48 KB
        available: true,
    }
}

/// NVIDIA RTX 2080 Ti (Turing - Compute 7.5)
#[cfg(feature = "gpu")]
pub fn nvidia_rtx_2080ti() -> MockGpuDevice {
    MockGpuDevice {
        device_id: 0,
        name: "NVIDIA GeForce RTX 2080 Ti",
        compute_capability: (7, 5),
        supports_fp16: true,
        supports_bf16: false,
        supports_tensor_cores: true,
        max_threads_per_block: 1024,
        max_shared_memory_per_block: 65536, // 64 KB
        available: true,
    }
}

/// NVIDIA A100 (Ampere - Compute 8.0)
#[cfg(feature = "gpu")]
pub fn nvidia_a100() -> MockGpuDevice {
    MockGpuDevice {
        device_id: 0,
        name: "NVIDIA A100-SXM4-40GB",
        compute_capability: (8, 0),
        supports_fp16: true,
        supports_bf16: true,
        supports_tensor_cores: true,
        max_threads_per_block: 1024,
        max_shared_memory_per_block: 163840, // 160 KB
        available: true,
    }
}

/// NVIDIA RTX 3090 (Ampere - Compute 8.6)
#[cfg(feature = "gpu")]
pub fn nvidia_rtx_3090() -> MockGpuDevice {
    MockGpuDevice {
        device_id: 0,
        name: "NVIDIA GeForce RTX 3090",
        compute_capability: (8, 6),
        supports_fp16: true,
        supports_bf16: true,
        supports_tensor_cores: true,
        max_threads_per_block: 1024,
        max_shared_memory_per_block: 102400, // 100 KB
        available: true,
    }
}

/// NVIDIA RTX 4090 (Ada Lovelace - Compute 8.9)
#[cfg(feature = "gpu")]
pub fn nvidia_rtx_4090() -> MockGpuDevice {
    MockGpuDevice {
        device_id: 0,
        name: "NVIDIA GeForce RTX 4090",
        compute_capability: (8, 9),
        supports_fp16: true,
        supports_bf16: true,
        supports_tensor_cores: true,
        max_threads_per_block: 1024,
        max_shared_memory_per_block: 102400, // 100 KB
        available: true,
    }
}

/// NVIDIA H100 (Hopper - Compute 9.0)
#[cfg(feature = "gpu")]
pub fn nvidia_h100() -> MockGpuDevice {
    MockGpuDevice {
        device_id: 0,
        name: "NVIDIA H100-SXM5-80GB",
        compute_capability: (9, 0),
        supports_fp16: true,
        supports_bf16: true,
        supports_tensor_cores: true,
        max_threads_per_block: 1024,
        max_shared_memory_per_block: 228352, // 223 KB
        available: true,
    }
}

/// GPU device with unavailable status (for fallback testing)
#[cfg(feature = "gpu")]
pub fn nvidia_gpu_unavailable() -> MockGpuDevice {
    MockGpuDevice {
        device_id: 0,
        name: "NVIDIA GeForce RTX 3090",
        compute_capability: (8, 6),
        supports_fp16: true,
        supports_bf16: true,
        supports_tensor_cores: true,
        max_threads_per_block: 1024,
        max_shared_memory_per_block: 102400,
        available: false, // Force unavailable for strict mode testing
    }
}

// ============================================================================
// CPU Device Mocks (SIMD Features)
// ============================================================================

/// Intel x86_64 CPU with AVX2 support
#[cfg(feature = "cpu")]
pub fn intel_cpu_avx2() -> MockCpuDevice {
    MockCpuDevice {
        device_id: 0,
        name: "Intel Core i7-9700K",
        architecture: CpuArchitecture::X86_64,
        supports_avx2: true,
        supports_avx512: false,
        supports_neon: false,
        supports_sve: false,
        num_cores: 8,
        available: true,
    }
}

/// Intel x86_64 CPU with AVX-512 support
#[cfg(feature = "cpu")]
pub fn intel_cpu_avx512() -> MockCpuDevice {
    MockCpuDevice {
        device_id: 0,
        name: "Intel Xeon Platinum 8280",
        architecture: CpuArchitecture::X86_64,
        supports_avx2: true,
        supports_avx512: true,
        supports_neon: false,
        supports_sve: false,
        num_cores: 28,
        available: true,
    }
}

/// AMD x86_64 CPU with AVX2 support (no AVX-512)
#[cfg(feature = "cpu")]
pub fn amd_cpu_avx2() -> MockCpuDevice {
    MockCpuDevice {
        device_id: 0,
        name: "AMD Ryzen 9 5950X",
        architecture: CpuArchitecture::X86_64,
        supports_avx2: true,
        supports_avx512: false, // AMD Zen 3 lacks AVX-512
        supports_neon: false,
        supports_sve: false,
        num_cores: 16,
        available: true,
    }
}

/// ARM Aarch64 CPU with NEON support
#[cfg(feature = "cpu")]
pub fn arm_cpu_neon() -> MockCpuDevice {
    MockCpuDevice {
        device_id: 0,
        name: "ARM Cortex-A72",
        architecture: CpuArchitecture::Aarch64,
        supports_avx2: false,
        supports_avx512: false,
        supports_neon: true,
        supports_sve: false,
        num_cores: 4,
        available: true,
    }
}

/// ARM Aarch64 CPU with NEON and SVE support
#[cfg(feature = "cpu")]
pub fn arm_cpu_neon_sve() -> MockCpuDevice {
    MockCpuDevice {
        device_id: 0,
        name: "ARM Neoverse V1",
        architecture: CpuArchitecture::Aarch64,
        supports_avx2: false,
        supports_avx512: false,
        supports_neon: true,
        supports_sve: true, // Scalable Vector Extension
        num_cores: 64,
        available: true,
    }
}

/// CPU with no SIMD features (for fallback testing)
#[cfg(feature = "cpu")]
pub fn cpu_no_simd() -> MockCpuDevice {
    MockCpuDevice {
        device_id: 0,
        name: "Generic x86_64 CPU",
        architecture: CpuArchitecture::X86_64,
        supports_avx2: false,
        supports_avx512: false,
        supports_neon: false,
        supports_sve: false,
        num_cores: 2,
        available: true,
    }
}

/// CPU device unavailable (for fallback testing)
#[cfg(feature = "cpu")]
pub fn cpu_unavailable() -> MockCpuDevice {
    MockCpuDevice {
        device_id: 0,
        name: "Intel Core i7-9700K",
        architecture: CpuArchitecture::X86_64,
        supports_avx2: true,
        supports_avx512: false,
        supports_neon: false,
        supports_sve: false,
        num_cores: 8,
        available: false, // Force unavailable for strict mode testing
    }
}

// ============================================================================
// Fallback Trigger Scenarios
// ============================================================================

/// I2S kernel unavailable on GPU
#[cfg(feature = "gpu")]
pub fn fallback_i2s_kernel_unavailable() -> FallbackScenario {
    FallbackScenario {
        scenario_name: "i2s_gpu_kernel_unavailable",
        trigger_reason: FallbackTrigger::KernelUnavailable,
        expected_strict_mode_behavior: StrictModeBehavior::RejectWithError,
        description: "I2S GPU kernel unavailable - strict mode should reject with error",
    }
}

/// TL1 NEON kernel unavailable on ARM
#[cfg(feature = "cpu")]
pub fn fallback_tl1_neon_unavailable() -> FallbackScenario {
    FallbackScenario {
        scenario_name: "tl1_neon_kernel_unavailable",
        trigger_reason: FallbackTrigger::KernelUnavailable,
        expected_strict_mode_behavior: StrictModeBehavior::RejectWithError,
        description: "TL1 NEON kernel unavailable - strict mode should reject with error",
    }
}

/// TL2 AVX kernel unavailable on x86
#[cfg(feature = "cpu")]
pub fn fallback_tl2_avx_unavailable() -> FallbackScenario {
    FallbackScenario {
        scenario_name: "tl2_avx_kernel_unavailable",
        trigger_reason: FallbackTrigger::KernelUnavailable,
        expected_strict_mode_behavior: StrictModeBehavior::RejectWithError,
        description: "TL2 AVX kernel unavailable - strict mode should reject with error",
    }
}

/// GPU compute capability too low for FP16
#[cfg(feature = "gpu")]
pub fn fallback_compute_capability_too_low() -> FallbackScenario {
    FallbackScenario {
        scenario_name: "gpu_compute_capability_too_low",
        trigger_reason: FallbackTrigger::ComputeCapabilityTooLow,
        expected_strict_mode_behavior: StrictModeBehavior::RejectWithError,
        description: "GPU compute capability < 7.0 - FP16 not fully supported",
    }
}

/// GPU memory insufficient for quantization buffer
#[cfg(feature = "gpu")]
pub fn fallback_insufficient_gpu_memory() -> FallbackScenario {
    FallbackScenario {
        scenario_name: "gpu_insufficient_memory",
        trigger_reason: FallbackTrigger::InsufficientMemory,
        expected_strict_mode_behavior: StrictModeBehavior::RejectWithError,
        description: "GPU memory insufficient for quantization buffer allocation",
    }
}

/// Tensor dimensions unsupported by quantized kernel
pub fn fallback_unsupported_dimensions() -> FallbackScenario {
    FallbackScenario {
        scenario_name: "unsupported_tensor_dimensions",
        trigger_reason: FallbackTrigger::UnsupportedDimensions,
        expected_strict_mode_behavior: StrictModeBehavior::RejectWithError,
        description: "Tensor dimensions not aligned to kernel requirements (e.g., not multiple of 32)",
    }
}

/// Device mismatch (model on GPU, kernel on CPU)
pub fn fallback_device_mismatch() -> FallbackScenario {
    FallbackScenario {
        scenario_name: "device_mismatch",
        trigger_reason: FallbackTrigger::DeviceMismatch,
        expected_strict_mode_behavior: StrictModeBehavior::RejectWithError,
        description: "Model loaded on GPU but quantized kernel only available on CPU",
    }
}

/// Missing SIMD features for CPU kernel
#[cfg(feature = "cpu")]
pub fn fallback_missing_simd_features() -> FallbackScenario {
    FallbackScenario {
        scenario_name: "missing_simd_features",
        trigger_reason: FallbackTrigger::MissingSimdFeatures,
        expected_strict_mode_behavior: StrictModeBehavior::RejectWithError,
        description: "Required SIMD features (AVX2/AVX-512/NEON) not available on CPU",
    }
}

// ============================================================================
// Device Capability Helpers
// ============================================================================

/// Check if GPU supports FP16 Tensor Cores
#[cfg(feature = "gpu")]
pub fn supports_fp16_tensor_cores(device: &MockGpuDevice) -> bool {
    device.supports_fp16 && device.supports_tensor_cores && device.compute_capability >= (7, 0)
}

/// Check if GPU supports BF16 Tensor Cores
#[cfg(feature = "gpu")]
pub fn supports_bf16_tensor_cores(device: &MockGpuDevice) -> bool {
    device.supports_bf16 && device.supports_tensor_cores && device.compute_capability >= (8, 0)
}

/// Check if CPU supports TL2 AVX-512 kernels
#[cfg(feature = "cpu")]
pub fn supports_tl2_avx512(device: &MockCpuDevice) -> bool {
    device.supports_avx512 && device.architecture == CpuArchitecture::X86_64
}

/// Check if CPU supports TL1 NEON kernels
#[cfg(feature = "cpu")]
pub fn supports_tl1_neon(device: &MockCpuDevice) -> bool {
    device.supports_neon && device.architecture == CpuArchitecture::Aarch64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_compute_capabilities() {
        let pascal = nvidia_gtx_1080ti();
        assert_eq!(pascal.compute_capability, (6, 1));
        assert!(!pascal.supports_tensor_cores);

        let turing = nvidia_rtx_2080ti();
        assert_eq!(turing.compute_capability, (7, 5));
        assert!(turing.supports_tensor_cores);
        assert!(!turing.supports_bf16);

        let ampere = nvidia_a100();
        assert_eq!(ampere.compute_capability, (8, 0));
        assert!(ampere.supports_bf16);
        assert!(ampere.supports_tensor_cores);
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_cpu_simd_features() {
        let avx2 = intel_cpu_avx2();
        assert!(avx2.supports_avx2);
        assert!(!avx2.supports_avx512);

        let avx512 = intel_cpu_avx512();
        assert!(avx512.supports_avx2);
        assert!(avx512.supports_avx512);

        let neon = arm_cpu_neon();
        assert!(neon.supports_neon);
        assert!(!neon.supports_avx2);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_tensor_core_support() {
        let pascal = nvidia_gtx_1080ti();
        assert!(!supports_fp16_tensor_cores(&pascal));

        let turing = nvidia_rtx_2080ti();
        assert!(supports_fp16_tensor_cores(&turing));
        assert!(!supports_bf16_tensor_cores(&turing));

        let ampere = nvidia_a100();
        assert!(supports_fp16_tensor_cores(&ampere));
        assert!(supports_bf16_tensor_cores(&ampere));
    }

    #[test]
    fn test_fallback_scenarios() {
        let unsupported_dims = fallback_unsupported_dimensions();
        assert_eq!(
            unsupported_dims.expected_strict_mode_behavior,
            StrictModeBehavior::RejectWithError
        );

        let device_mismatch = fallback_device_mismatch();
        assert_eq!(
            device_mismatch.trigger_reason,
            FallbackTrigger::DeviceMismatch
        );
    }
}
