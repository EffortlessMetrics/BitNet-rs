//! Kernel selection heuristics.
//!
//! Auto-select the best kernel implementation based on hardware
//! capabilities, problem size, and data format.

/// Available kernel backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelBackendType {
    ScalarCpu,
    Avx2,
    Avx512,
    Neon,
    Cuda,
    OpenCl,
    Metal,
}

impl KernelBackendType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ScalarCpu => "scalar_cpu",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Neon => "neon",
            Self::Cuda => "cuda",
            Self::OpenCl => "opencl",
            Self::Metal => "metal",
        }
    }

    pub fn is_simd(&self) -> bool {
        matches!(self, Self::Avx2 | Self::Avx512 | Self::Neon)
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Cuda | Self::OpenCl | Self::Metal)
    }
}

/// Problem characteristics for kernel selection.
#[derive(Debug, Clone)]
pub struct ProblemSpec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch_size: usize,
    pub is_quantized: bool,
    pub bits_per_weight: u8,
}

impl ProblemSpec {
    pub fn matmul(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k, batch_size: 1, is_quantized: false, bits_per_weight: 16 }
    }

    pub fn quantized(m: usize, n: usize, k: usize, bits: u8) -> Self {
        Self { m, n, k, batch_size: 1, is_quantized: true, bits_per_weight: bits }
    }

    /// Total FLOPs for this matmul (2*M*N*K * batch).
    pub fn flops(&self) -> u64 {
        2 * self.m as u64 * self.n as u64 * self.k as u64 * self.batch_size as u64
    }

    /// Whether this is a "large" problem that benefits from GPU.
    pub fn is_large(&self) -> bool {
        self.flops() > 100_000_000
    }

    /// Whether this is a "tiny" problem where kernel overhead dominates.
    pub fn is_tiny(&self) -> bool {
        self.flops() < 10_000
    }
}

/// Hardware capabilities detected at runtime.
#[derive(Debug, Clone)]
pub struct HardwareCaps {
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_neon: bool,
    pub has_cuda: bool,
    pub has_opencl: bool,
    pub has_metal: bool,
    pub cpu_cores: usize,
}

impl Default for HardwareCaps {
    fn default() -> Self {
        Self {
            has_avx2: false,
            has_avx512: false,
            has_neon: false,
            has_cuda: false,
            has_opencl: false,
            has_metal: false,
            cpu_cores: 1,
        }
    }
}

impl HardwareCaps {
    pub fn available_backends(&self) -> Vec<KernelBackendType> {
        let mut backends = vec![KernelBackendType::ScalarCpu];
        if self.has_avx2 {
            backends.push(KernelBackendType::Avx2);
        }
        if self.has_avx512 {
            backends.push(KernelBackendType::Avx512);
        }
        if self.has_neon {
            backends.push(KernelBackendType::Neon);
        }
        if self.has_cuda {
            backends.push(KernelBackendType::Cuda);
        }
        if self.has_opencl {
            backends.push(KernelBackendType::OpenCl);
        }
        if self.has_metal {
            backends.push(KernelBackendType::Metal);
        }
        backends
    }
}

/// Selection result with reasoning.
#[derive(Debug, Clone)]
pub struct KernelChoice {
    pub backend: KernelBackendType,
    pub reason: String,
    pub alternatives: Vec<KernelBackendType>,
}

/// Select the best kernel for a given problem and hardware.
pub fn select_kernel(spec: &ProblemSpec, caps: &HardwareCaps) -> KernelChoice {
    let available = caps.available_backends();

    // GPU for large problems
    if spec.is_large() {
        if caps.has_cuda {
            return KernelChoice {
                backend: KernelBackendType::Cuda,
                reason: "large problem; CUDA available".into(),
                alternatives: available
                    .into_iter()
                    .filter(|b| *b != KernelBackendType::Cuda)
                    .collect(),
            };
        }
        if caps.has_metal {
            return KernelChoice {
                backend: KernelBackendType::Metal,
                reason: "large problem; Metal available".into(),
                alternatives: available
                    .into_iter()
                    .filter(|b| *b != KernelBackendType::Metal)
                    .collect(),
            };
        }
    }

    // SIMD for medium/small CPU problems
    if caps.has_avx512 {
        return KernelChoice {
            backend: KernelBackendType::Avx512,
            reason: "AVX-512 available; best CPU SIMD".into(),
            alternatives: available
                .into_iter()
                .filter(|b| *b != KernelBackendType::Avx512)
                .collect(),
        };
    }
    if caps.has_avx2 {
        return KernelChoice {
            backend: KernelBackendType::Avx2,
            reason: "AVX2 available; good CPU SIMD".into(),
            alternatives: available.into_iter().filter(|b| *b != KernelBackendType::Avx2).collect(),
        };
    }
    if caps.has_neon {
        return KernelChoice {
            backend: KernelBackendType::Neon,
            reason: "NEON available; ARM SIMD".into(),
            alternatives: available.into_iter().filter(|b| *b != KernelBackendType::Neon).collect(),
        };
    }

    // Scalar fallback
    KernelChoice {
        backend: KernelBackendType::ScalarCpu,
        reason: "scalar fallback; no SIMD detected".into(),
        alternatives: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_type_properties() {
        assert!(KernelBackendType::Avx2.is_simd());
        assert!(!KernelBackendType::Avx2.is_gpu());
        assert!(KernelBackendType::Cuda.is_gpu());
        assert!(!KernelBackendType::Cuda.is_simd());
    }

    #[test]
    fn test_problem_spec_flops() {
        let spec = ProblemSpec::matmul(128, 256, 512);
        assert_eq!(spec.flops(), 2 * 128 * 256 * 512);
    }

    #[test]
    fn test_problem_size_classification() {
        let tiny = ProblemSpec::matmul(2, 2, 2);
        assert!(tiny.is_tiny());
        assert!(!tiny.is_large());

        let large = ProblemSpec::matmul(1024, 1024, 1024);
        assert!(large.is_large());
        assert!(!large.is_tiny());
    }

    #[test]
    fn test_select_scalar_fallback() {
        let spec = ProblemSpec::matmul(64, 64, 64);
        let caps = HardwareCaps::default();
        let choice = select_kernel(&spec, &caps);
        assert_eq!(choice.backend, KernelBackendType::ScalarCpu);
    }

    #[test]
    fn test_select_avx2() {
        let spec = ProblemSpec::matmul(64, 64, 64);
        let caps = HardwareCaps { has_avx2: true, ..Default::default() };
        let choice = select_kernel(&spec, &caps);
        assert_eq!(choice.backend, KernelBackendType::Avx2);
    }

    #[test]
    fn test_select_cuda_large() {
        let spec = ProblemSpec::matmul(1024, 1024, 1024);
        let caps = HardwareCaps { has_cuda: true, has_avx2: true, ..Default::default() };
        let choice = select_kernel(&spec, &caps);
        assert_eq!(choice.backend, KernelBackendType::Cuda);
    }

    #[test]
    fn test_select_avx2_when_cuda_but_small() {
        let spec = ProblemSpec::matmul(8, 8, 8);
        let caps = HardwareCaps { has_cuda: true, has_avx2: true, ..Default::default() };
        let choice = select_kernel(&spec, &caps);
        assert_eq!(choice.backend, KernelBackendType::Avx2);
    }

    #[test]
    fn test_available_backends() {
        let caps = HardwareCaps { has_avx2: true, has_neon: true, ..Default::default() };
        let backends = caps.available_backends();
        assert!(backends.contains(&KernelBackendType::ScalarCpu));
        assert!(backends.contains(&KernelBackendType::Avx2));
        assert!(backends.contains(&KernelBackendType::Neon));
        assert!(!backends.contains(&KernelBackendType::Cuda));
    }

    #[test]
    fn test_quantized_spec() {
        let spec = ProblemSpec::quantized(128, 256, 512, 2);
        assert!(spec.is_quantized);
        assert_eq!(spec.bits_per_weight, 2);
    }

    #[test]
    fn test_choice_has_alternatives() {
        let spec = ProblemSpec::matmul(64, 64, 64);
        let caps = HardwareCaps { has_avx2: true, ..Default::default() };
        let choice = select_kernel(&spec, &caps);
        assert!(!choice.alternatives.is_empty());
        assert!(choice.alternatives.contains(&KernelBackendType::ScalarCpu));
    }

    #[test]
    fn test_backend_as_str() {
        assert_eq!(KernelBackendType::ScalarCpu.as_str(), "scalar_cpu");
        assert_eq!(KernelBackendType::Avx2.as_str(), "avx2");
        assert_eq!(KernelBackendType::Cuda.as_str(), "cuda");
    }

    #[test]
    fn test_avx512_preferred_over_avx2() {
        let spec = ProblemSpec::matmul(64, 64, 64);
        let caps = HardwareCaps { has_avx2: true, has_avx512: true, ..Default::default() };
        let choice = select_kernel(&spec, &caps);
        assert_eq!(choice.backend, KernelBackendType::Avx512);
    }
}
