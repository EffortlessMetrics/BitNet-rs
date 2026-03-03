//! Kernel selection heuristic.
//!
//! Automatically select the best kernel implementation for a given operation.

/// Available kernel implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelImpl {
    Scalar,
    Avx2,
    Avx512,
    Neon,
    Cuda,
    OpenCL,
    Wasm,
}

impl KernelImpl {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Neon => "neon",
            Self::Cuda => "cuda",
            Self::OpenCL => "opencl",
            Self::Wasm => "wasm",
        }
    }

    pub fn is_simd(&self) -> bool {
        matches!(self, Self::Avx2 | Self::Avx512 | Self::Neon)
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Cuda | Self::OpenCL)
    }

    /// Relative throughput estimate (higher = faster).
    pub fn throughput_score(&self) -> u32 {
        match self {
            Self::Cuda => 100,
            Self::OpenCL => 60,
            Self::Avx512 => 40,
            Self::Avx2 => 20,
            Self::Neon => 18,
            Self::Wasm => 5,
            Self::Scalar => 1,
        }
    }
}

/// Operation characteristics for kernel selection.
#[derive(Debug, Clone)]
pub struct OpProfile {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub is_quantized: bool,
    pub batch_size: usize,
}

impl OpProfile {
    pub fn matmul(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k, is_quantized: false, batch_size: 1 }
    }

    pub fn flops(&self) -> u64 {
        2 * self.m as u64 * self.n as u64 * self.k as u64
    }

    pub fn is_large(&self) -> bool {
        self.flops() > 100_000_000
    }
    pub fn is_tiny(&self) -> bool {
        self.flops() < 10_000
    }
}

/// Hardware capabilities.
#[derive(Debug, Clone, Default)]
pub struct HardwareCaps {
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_neon: bool,
    pub has_cuda: bool,
    pub has_opencl: bool,
}

impl HardwareCaps {
    pub fn detect() -> Self {
        Self {
            has_avx2: cfg!(target_feature = "avx2") || detect_x86("avx2"),
            has_avx512: cfg!(target_feature = "avx512f") || detect_x86("avx512f"),
            has_neon: cfg!(target_arch = "aarch64"),
            has_cuda: false, // runtime detection needed
            has_opencl: false,
        }
    }

    pub fn available_impls(&self) -> Vec<KernelImpl> {
        let mut impls = vec![KernelImpl::Scalar];
        if self.has_avx2 {
            impls.push(KernelImpl::Avx2);
        }
        if self.has_avx512 {
            impls.push(KernelImpl::Avx512);
        }
        if self.has_neon {
            impls.push(KernelImpl::Neon);
        }
        if self.has_cuda {
            impls.push(KernelImpl::Cuda);
        }
        if self.has_opencl {
            impls.push(KernelImpl::OpenCL);
        }
        impls
    }
}

fn detect_x86(_feature: &str) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        match _feature {
            "avx2" => std::arch::is_x86_feature_detected!("avx2"),
            "avx512f" => std::arch::is_x86_feature_detected!("avx512f"),
            _ => false,
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Select the best kernel for an operation.
pub fn select_kernel(caps: &HardwareCaps, profile: &OpProfile) -> KernelImpl {
    let available = caps.available_impls();

    // For GPU ops, prefer GPU if the operation is large enough
    if profile.is_large() {
        if available.contains(&KernelImpl::Cuda) {
            return KernelImpl::Cuda;
        }
        if available.contains(&KernelImpl::OpenCL) {
            return KernelImpl::OpenCL;
        }
    }

    // For CPU, pick highest throughput SIMD
    available
        .iter()
        .filter(|k| !k.is_gpu())
        .max_by_key(|k| k.throughput_score())
        .copied()
        .unwrap_or(KernelImpl::Scalar)
}

/// Selection result with reasoning.
#[derive(Debug, Clone)]
pub struct SelectionResult {
    pub selected: KernelImpl,
    pub available: Vec<KernelImpl>,
    pub reason: String,
}

pub fn select_with_reason(caps: &HardwareCaps, profile: &OpProfile) -> SelectionResult {
    let available = caps.available_impls();
    let selected = select_kernel(caps, profile);
    let reason = if profile.is_large() && selected.is_gpu() {
        format!("Large op ({} GFLOPS) → GPU", profile.flops() as f64 / 1e9)
    } else {
        format!("Best CPU: {} (score {})", selected.as_str(), selected.throughput_score())
    };
    SelectionResult { selected, available, reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_fallback() {
        let caps = HardwareCaps::default();
        let profile = OpProfile::matmul(128, 128, 128);
        let selected = select_kernel(&caps, &profile);
        assert_eq!(selected, KernelImpl::Scalar);
    }

    #[test]
    fn test_avx2_preferred() {
        let caps = HardwareCaps { has_avx2: true, ..Default::default() };
        let profile = OpProfile::matmul(128, 128, 128);
        let selected = select_kernel(&caps, &profile);
        assert_eq!(selected, KernelImpl::Avx2);
    }

    #[test]
    fn test_avx512_over_avx2() {
        let caps = HardwareCaps { has_avx2: true, has_avx512: true, ..Default::default() };
        let profile = OpProfile::matmul(128, 128, 128);
        let selected = select_kernel(&caps, &profile);
        assert_eq!(selected, KernelImpl::Avx512);
    }

    #[test]
    fn test_gpu_for_large() {
        let caps = HardwareCaps { has_avx2: true, has_cuda: true, ..Default::default() };
        let profile = OpProfile::matmul(4096, 4096, 4096);
        let selected = select_kernel(&caps, &profile);
        assert_eq!(selected, KernelImpl::Cuda);
    }

    #[test]
    fn test_flops() {
        let p = OpProfile::matmul(100, 200, 300);
        assert_eq!(p.flops(), 12_000_000);
    }

    #[test]
    fn test_is_large() {
        assert!(OpProfile::matmul(1000, 1000, 1000).is_large());
        assert!(!OpProfile::matmul(10, 10, 10).is_large());
    }

    #[test]
    fn test_is_tiny() {
        assert!(OpProfile::matmul(5, 5, 5).is_tiny());
    }

    #[test]
    fn test_impl_types() {
        assert!(KernelImpl::Avx2.is_simd());
        assert!(KernelImpl::Cuda.is_gpu());
        assert!(!KernelImpl::Scalar.is_simd());
    }

    #[test]
    fn test_detect_caps() {
        let caps = HardwareCaps::detect();
        let impls = caps.available_impls();
        assert!(impls.contains(&KernelImpl::Scalar));
    }

    #[test]
    fn test_select_with_reason() {
        let caps = HardwareCaps { has_avx2: true, ..Default::default() };
        let profile = OpProfile::matmul(128, 128, 128);
        let result = select_with_reason(&caps, &profile);
        assert_eq!(result.selected, KernelImpl::Avx2);
        assert!(!result.reason.is_empty());
    }

    #[test]
    fn test_throughput_ordering() {
        assert!(KernelImpl::Cuda.throughput_score() > KernelImpl::Avx512.throughput_score());
        assert!(KernelImpl::Avx2.throughput_score() > KernelImpl::Scalar.throughput_score());
    }

    #[test]
    fn test_impl_str() {
        assert_eq!(KernelImpl::Scalar.as_str(), "scalar");
        assert_eq!(KernelImpl::Cuda.as_str(), "cuda");
    }
}
