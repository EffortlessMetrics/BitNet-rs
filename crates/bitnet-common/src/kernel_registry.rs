//! Kernel capability registry — single source of truth for available backends.
//!
//! Defines the canonical enumeration of kernel backends and the capabilities
//! snapshot that describes what a given build or runtime configuration provides.

use std::fmt;

/// SIMD instruction set level available at compile or runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum SimdLevel {
    /// No SIMD; scalar fallback only.
    Scalar,
    /// ARM NEON (128-bit).
    Neon,
    /// x86 SSE4.2 (128-bit).
    Sse42,
    /// x86 AVX2 (256-bit).
    Avx2,
    /// x86 AVX-512 (512-bit).
    Avx512,
}

impl fmt::Display for SimdLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimdLevel::Scalar => write!(f, "scalar"),
            SimdLevel::Neon => write!(f, "neon"),
            SimdLevel::Sse42 => write!(f, "sse4.2"),
            SimdLevel::Avx2 => write!(f, "avx2"),
            SimdLevel::Avx512 => write!(f, "avx512"),
        }
    }
}

/// The kernel backend variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum KernelBackend {
    /// Pure-Rust CPU kernels with optional SIMD.
    CpuRust,
    /// CUDA GPU kernels via `cudarc`.
    Cuda,
    /// C++ FFI bridge to bitnet.cpp / llama.cpp.
    CppFfi,
}

impl fmt::Display for KernelBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelBackend::CpuRust => write!(f, "cpu-rust"),
            KernelBackend::Cuda => write!(f, "cuda"),
            KernelBackend::CppFfi => write!(f, "cpp-ffi"),
        }
    }
}

impl KernelBackend {
    /// Returns true if this backend requires a GPU at runtime.
    pub fn requires_gpu(self) -> bool {
        matches!(self, KernelBackend::Cuda)
    }

    /// Returns true if this backend is compiled in the current build.
    pub fn is_compiled(self) -> bool {
        match self {
            KernelBackend::CpuRust => cfg!(feature = "cpu"),
            KernelBackend::Cuda => cfg!(feature = "cuda"),
            // FFI availability is determined by the consumer crate's feature flags
            KernelBackend::CppFfi => false,
        }
    }
}

/// Snapshot of what a build configuration provides.
///
/// Constructed from compile-time feature flags and optional runtime probing.
/// This is the "source of truth" for kernel selection logic.
#[derive(Debug, Clone)]
pub struct KernelCapabilities {
    /// CPU-Rust backend is compiled and available.
    pub cpu_rust: bool,
    /// CUDA backend is compiled (may still require runtime GPU).
    pub cuda_compiled: bool,
    /// CUDA runtime detected (GPU present and accessible).
    pub cuda_runtime: bool,
    /// C++ FFI bridge is compiled.
    pub cpp_ffi: bool,
    /// Best SIMD level available at compile time.
    pub simd_level: SimdLevel,
}

impl KernelCapabilities {
    /// Build from compile-time feature flags (no runtime probing).
    ///
    /// `cuda_runtime` will be `false`; use [`with_cuda_runtime`] to fill it in.
    pub fn from_compile_time() -> Self {
        KernelCapabilities {
            cpu_rust: cfg!(feature = "cpu"),
            cuda_compiled: cfg!(feature = "cuda"),
            cuda_runtime: false, // requires runtime check
            cpp_ffi: false,      // bitnet-common has no ffi feature; FFI detection is crate-local
            simd_level: compile_time_simd_level(),
        }
    }

    /// Fill in the `cuda_runtime` field from a live probe result.
    ///
    /// Callers in `bitnet-kernels` call
    /// `KernelCapabilities::from_compile_time().with_cuda_runtime(gpu_available_runtime())`
    /// to get a fully-populated snapshot.
    #[must_use]
    pub fn with_cuda_runtime(mut self, available: bool) -> Self {
        self.cuda_runtime = available;
        self
    }

    /// Fill in the `cpp_ffi` field.
    #[must_use]
    pub fn with_cpp_ffi(mut self, available: bool) -> Self {
        self.cpp_ffi = available;
        self
    }

    /// Returns backends that are compiled in, in priority order (best first).
    pub fn compiled_backends(&self) -> Vec<KernelBackend> {
        let mut backends = Vec::new();
        if self.cuda_compiled {
            backends.push(KernelBackend::Cuda);
        }
        if self.cpp_ffi {
            backends.push(KernelBackend::CppFfi);
        }
        if self.cpu_rust {
            backends.push(KernelBackend::CpuRust);
        }
        backends
    }

    /// Returns the best backend available (preferring GPU > FFI > CPU).
    pub fn best_available(&self) -> Option<KernelBackend> {
        if self.cuda_compiled && self.cuda_runtime {
            return Some(KernelBackend::Cuda);
        }
        if self.cpp_ffi {
            return Some(KernelBackend::CppFfi);
        }
        if self.cpu_rust {
            return Some(KernelBackend::CpuRust);
        }
        None
    }

    /// Returns a human-readable summary string for receipts/logs.
    pub fn summary(&self) -> String {
        let backends: Vec<String> =
            self.compiled_backends().iter().map(|b| b.to_string()).collect();
        format!("simd={} backends=[{}]", self.simd_level, backends.join(","))
    }
}

/// Detect the best SIMD level available at compile time.
const fn compile_time_simd_level() -> SimdLevel {
    #[cfg(target_feature = "avx512f")]
    return SimdLevel::Avx512;
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    return SimdLevel::Avx2;
    #[cfg(all(target_feature = "sse4.2", not(target_feature = "avx2")))]
    return SimdLevel::Sse42;
    #[cfg(all(target_arch = "aarch64", not(target_feature = "sse4.2")))]
    return SimdLevel::Neon;
    #[cfg(not(any(
        target_feature = "avx512f",
        target_feature = "avx2",
        target_feature = "sse4.2",
        target_arch = "aarch64",
    )))]
    SimdLevel::Scalar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_level_ordering() {
        assert!(SimdLevel::Scalar < SimdLevel::Neon);
        assert!(SimdLevel::Neon < SimdLevel::Sse42);
        assert!(SimdLevel::Sse42 < SimdLevel::Avx2);
        assert!(SimdLevel::Avx2 < SimdLevel::Avx512);
    }

    #[test]
    fn simd_level_display() {
        assert_eq!(SimdLevel::Avx2.to_string(), "avx2");
        assert_eq!(SimdLevel::Scalar.to_string(), "scalar");
    }

    #[test]
    fn kernel_backend_display() {
        assert_eq!(KernelBackend::CpuRust.to_string(), "cpu-rust");
        assert_eq!(KernelBackend::Cuda.to_string(), "cuda");
        assert_eq!(KernelBackend::CppFfi.to_string(), "cpp-ffi");
    }

    #[test]
    fn kernel_backend_gpu_requirement() {
        assert!(!KernelBackend::CpuRust.requires_gpu());
        assert!(KernelBackend::Cuda.requires_gpu());
        assert!(!KernelBackend::CppFfi.requires_gpu());
    }

    #[test]
    fn compile_time_capabilities_no_cuda() {
        let caps = KernelCapabilities::from_compile_time();
        // cpu_rust reflects feature="cpu"
        #[cfg(feature = "cpu")]
        assert!(caps.cpu_rust);
        // cuda_compiled reflects feature="cuda"
        #[cfg(not(feature = "cuda"))]
        assert!(!caps.cuda_compiled);
    }

    #[test]
    fn best_available_prefers_gpu() {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: true,
            cuda_runtime: true,
            cpp_ffi: false,
            simd_level: SimdLevel::Avx2,
        };
        assert_eq!(caps.best_available(), Some(KernelBackend::Cuda));
    }

    #[test]
    fn best_available_falls_back_to_cpu() {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: false,
            cuda_runtime: false,
            cpp_ffi: false,
            simd_level: SimdLevel::Avx2,
        };
        assert_eq!(caps.best_available(), Some(KernelBackend::CpuRust));
    }

    #[test]
    fn compiled_backends_order_gpu_first() {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: true,
            cuda_runtime: false,
            cpp_ffi: true,
            simd_level: SimdLevel::Avx2,
        };
        let backends = caps.compiled_backends();
        assert_eq!(backends[0], KernelBackend::Cuda);
        assert_eq!(backends[1], KernelBackend::CppFfi);
        assert_eq!(backends[2], KernelBackend::CpuRust);
    }

    #[test]
    fn summary_contains_simd_and_backends() {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: false,
            cuda_runtime: false,
            cpp_ffi: false,
            simd_level: SimdLevel::Avx2,
        };
        let s = caps.summary();
        assert!(s.contains("avx2"), "summary: {s}");
        assert!(s.contains("cpu-rust"), "summary: {s}");
    }

    #[test]
    fn with_cuda_runtime_sets_flag() {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: true,
            cuda_runtime: false,
            cpp_ffi: false,
            simd_level: SimdLevel::Scalar,
        };
        let caps = caps.with_cuda_runtime(true);
        assert!(caps.cuda_runtime);
        // cuda available → best_available is Cuda
        assert_eq!(caps.best_available(), Some(KernelBackend::Cuda));
    }

    #[test]
    fn with_cpp_ffi_sets_flag() {
        let caps = KernelCapabilities::from_compile_time().with_cpp_ffi(true);
        assert!(caps.cpp_ffi);
    }

    #[test]
    fn with_cuda_runtime_false_keeps_cpu_as_best() {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled: true,
            cuda_runtime: false, // compiled but no runtime
            cpp_ffi: false,
            simd_level: SimdLevel::Scalar,
        };
        assert_eq!(caps.best_available(), Some(KernelBackend::CpuRust));
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    fn arb_caps() -> impl Strategy<Value = KernelCapabilities> {
        (any::<bool>(), any::<bool>(), any::<bool>(), any::<bool>()).prop_map(
            |(cpu_rust, cuda_compiled, cuda_runtime, cpp_ffi)| KernelCapabilities {
                cpu_rust,
                cuda_compiled,
                cuda_runtime: cuda_compiled && cuda_runtime,
                cpp_ffi,
                simd_level: SimdLevel::Scalar,
            },
        )
    }

    /// compiled_backends never contains duplicates.
    proptest! {
        #[test]
        fn compiled_backends_no_duplicates(caps in arb_caps()) {
            let backends = caps.compiled_backends();
            let unique: std::collections::HashSet<_> = backends.iter().collect();
            prop_assert_eq!(backends.len(), unique.len(), "duplicates in {:?}", backends);
        }
    }

    /// best_available returns Some iff cpu_rust or (cuda_compiled && cuda_runtime) or cpp_ffi.
    proptest! {
        #[test]
        fn best_available_iff_any_backend_reachable(caps in arb_caps()) {
            let reachable = caps.cpu_rust
                || (caps.cuda_compiled && caps.cuda_runtime)
                || caps.cpp_ffi;
            let best = caps.best_available();
            prop_assert_eq!(
                best.is_some(),
                reachable,
                "reachable={} but best={:?} for cpu_rust={} cuda_compiled={} cuda_runtime={} cpp_ffi={}",
                reachable, best, caps.cpu_rust, caps.cuda_compiled, caps.cuda_runtime, caps.cpp_ffi
            );
        }
    }

    /// CUDA is preferred over CPU when both are available.
    proptest! {
        #[test]
        fn cuda_preferred_over_cpu_when_both_available(any_ffi in any::<bool>()) {
            let caps = KernelCapabilities {
                cpu_rust: true,
                cuda_compiled: true,
                cuda_runtime: true,
                cpp_ffi: any_ffi,
                simd_level: SimdLevel::Scalar,
            };
            prop_assert_eq!(caps.best_available(), Some(KernelBackend::Cuda));
        }
    }

    /// KernelBackend::requires_gpu is true only for Cuda.
    proptest! {
        #[test]
        fn requires_gpu_only_for_cuda(
            backend in prop_oneof![
                Just(KernelBackend::CpuRust),
                Just(KernelBackend::Cuda),
                Just(KernelBackend::CppFfi),
            ],
        ) {
            let requires = backend.requires_gpu();
            prop_assert_eq!(requires, backend == KernelBackend::Cuda);
        }
    }
}
