//! Runtime diagnostics.
//!
//! System information and capability detection for inference.

/// CPU feature set.
#[derive(Debug, Clone, Default)]
pub struct CpuFeatures {
    pub sse2: bool,
    pub sse42: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub neon: bool,
    pub fma: bool,
}

impl CpuFeatures {
    /// Detect CPU features at runtime.
    pub fn detect() -> Self {
        Self {
            sse2: cfg!(target_feature = "sse2") || is_x86_feature("sse2"),
            sse42: cfg!(target_feature = "sse4.2") || is_x86_feature("sse4.2"),
            avx: cfg!(target_feature = "avx") || is_x86_feature("avx"),
            avx2: cfg!(target_feature = "avx2") || is_x86_feature("avx2"),
            avx512f: cfg!(target_feature = "avx512f") || is_x86_feature("avx512f"),
            neon: cfg!(target_arch = "aarch64"),
            fma: cfg!(target_feature = "fma") || is_x86_feature("fma"),
        }
    }

    pub fn best_simd(&self) -> &'static str {
        if self.avx512f {
            "avx512"
        } else if self.avx2 {
            "avx2"
        } else if self.avx {
            "avx"
        } else if self.neon {
            "neon"
        } else if self.sse42 {
            "sse4.2"
        } else if self.sse2 {
            "sse2"
        } else {
            "scalar"
        }
    }
}

fn is_x86_feature(_feature: &str) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        match _feature {
            "sse2" => std::arch::is_x86_feature_detected!("sse2"),
            "sse4.2" => std::arch::is_x86_feature_detected!("sse4.2"),
            "avx" => std::arch::is_x86_feature_detected!("avx"),
            "avx2" => std::arch::is_x86_feature_detected!("avx2"),
            "avx512f" => std::arch::is_x86_feature_detected!("avx512f"),
            "fma" => std::arch::is_x86_feature_detected!("fma"),
            _ => false,
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Memory information.
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

impl MemoryInfo {
    pub fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
    pub fn available_gb(&self) -> f64 {
        self.available_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn can_fit_model(&self, model_bytes: u64) -> bool {
        self.available_bytes > model_bytes
    }
}

/// System diagnostic snapshot.
#[derive(Debug, Clone)]
pub struct SystemDiag {
    pub arch: String,
    pub os: String,
    pub cpu_features: CpuFeatures,
    pub num_cpus: usize,
    pub pointer_width: u32,
}

impl SystemDiag {
    pub fn capture() -> Self {
        Self {
            arch: std::env::consts::ARCH.to_string(),
            os: std::env::consts::OS.to_string(),
            cpu_features: CpuFeatures::detect(),
            num_cpus: std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1),
            pointer_width: std::mem::size_of::<usize>() as u32 * 8,
        }
    }

    pub fn is_64bit(&self) -> bool {
        self.pointer_width == 64
    }

    pub fn recommended_threads(&self) -> usize {
        // Use physical cores (approx: total/2 on hyperthreaded)
        (self.num_cpus / 2).max(1)
    }
}

/// Inference readiness check.
#[derive(Debug, Clone)]
pub struct ReadinessCheck {
    pub simd_available: bool,
    pub sufficient_memory: bool,
    pub arch_supported: bool,
    pub ready: bool,
    pub warnings: Vec<String>,
}

pub fn check_readiness(model_bytes: u64, available_memory: u64) -> ReadinessCheck {
    let cpu = CpuFeatures::detect();
    let diag = SystemDiag::capture();

    let simd_available = cpu.avx2 || cpu.neon || cpu.avx;
    let sufficient_memory = available_memory > model_bytes;
    let arch_supported = diag.arch == "x86_64" || diag.arch == "aarch64";

    let mut warnings = Vec::new();
    if !simd_available {
        warnings.push("No SIMD detected; inference will be slow".into());
    }
    if !sufficient_memory {
        warnings.push(format!(
            "Model needs {} GB but only {} GB available",
            model_bytes as f64 / 1e9,
            available_memory as f64 / 1e9
        ));
    }
    if !arch_supported {
        warnings.push(format!("Untested architecture: {}", diag.arch));
    }

    let ready = sufficient_memory && arch_supported;

    ReadinessCheck { simd_available, sufficient_memory, arch_supported, ready, warnings }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_detect() {
        let cpu = CpuFeatures::detect();
        // On any test machine, at least scalar should work
        let best = cpu.best_simd();
        assert!(!best.is_empty());
    }

    #[test]
    fn test_system_diag() {
        let diag = SystemDiag::capture();
        assert!(!diag.arch.is_empty());
        assert!(!diag.os.is_empty());
        assert!(diag.num_cpus > 0);
    }

    #[test]
    fn test_is_64bit() {
        let diag = SystemDiag::capture();
        // Most test machines are 64-bit
        assert!(diag.is_64bit());
    }

    #[test]
    fn test_recommended_threads() {
        let diag = SystemDiag::capture();
        assert!(diag.recommended_threads() >= 1);
    }

    #[test]
    fn test_memory_info() {
        let mem = MemoryInfo {
            total_bytes: 16 * 1024 * 1024 * 1024,
            available_bytes: 8 * 1024 * 1024 * 1024,
        };
        assert!((mem.total_gb() - 16.0).abs() < 0.01);
        assert!(mem.can_fit_model(4 * 1024 * 1024 * 1024));
    }

    #[test]
    fn test_readiness_ok() {
        let r = check_readiness(1_000_000, 1_000_000_000);
        assert!(r.sufficient_memory);
        assert!(r.ready);
    }

    #[test]
    fn test_readiness_insufficient_memory() {
        let r = check_readiness(100_000_000_000, 1_000_000);
        assert!(!r.sufficient_memory);
        assert!(!r.ready);
    }

    #[test]
    fn test_best_simd_default() {
        let cpu = CpuFeatures::default();
        assert_eq!(cpu.best_simd(), "scalar");
    }

    #[test]
    fn test_readiness_warnings() {
        let r = check_readiness(100_000_000_000, 1_000_000);
        assert!(!r.warnings.is_empty());
    }

    #[test]
    fn test_memory_cant_fit() {
        let mem = MemoryInfo {
            total_bytes: 4 * 1024 * 1024 * 1024,
            available_bytes: 2 * 1024 * 1024 * 1024,
        };
        assert!(!mem.can_fit_model(3 * 1024 * 1024 * 1024));
    }
}
