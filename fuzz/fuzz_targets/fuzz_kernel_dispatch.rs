#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::{KernelBackend, SimdLevel};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct KernelDispatchInput {
    /// Selects backend variant (modulo enum count).
    backend_byte: u8,
    /// Selects SIMD level (modulo enum count).
    simd_byte: u8,
    /// Whether to test requires_gpu.
    check_gpu: bool,
    /// Whether to test is_compiled.
    check_compiled: bool,
    /// Repeat count for backend selection stress.
    repeat: u8,
}

fuzz_target!(|input: KernelDispatchInput| {
    // Exercise KernelBackend enum methods with arbitrary selectors
    let backend = match input.backend_byte % 6 {
        0 => KernelBackend::CpuRust,
        1 => KernelBackend::Cuda,
        2 => KernelBackend::Hip,
        3 => KernelBackend::OneApi,
        4 => KernelBackend::OpenCL,
        _ => KernelBackend::CppFfi,
    };

    let simd = match input.simd_byte % 5 {
        0 => SimdLevel::Scalar,
        1 => SimdLevel::Neon,
        2 => SimdLevel::Sse42,
        3 => SimdLevel::Avx2,
        _ => SimdLevel::Avx512,
    };

    // Display and Debug must not panic
    let _ = format!("{backend}");
    let _ = format!("{backend:?}");
    let _ = format!("{simd}");
    let _ = format!("{simd:?}");

    // requires_gpu must not panic and must be consistent
    let gpu_required = backend.requires_gpu();
    if input.check_gpu {
        match backend {
            KernelBackend::CpuRust | KernelBackend::CppFfi => {
                assert!(!gpu_required, "{backend} should not require GPU");
            }
            KernelBackend::Cuda
            | KernelBackend::Hip
            | KernelBackend::OneApi
            | KernelBackend::OpenCL => {
                assert!(gpu_required, "{backend} should require GPU");
            }
            _ => {}
        }
    }

    // is_compiled must not panic
    if input.check_compiled {
        let _ = backend.is_compiled();
    }

    // SimdLevel ordering: Scalar < Neon < Sse42 < Avx2 < Avx512
    assert!(SimdLevel::Scalar <= simd, "Scalar should be <= any level");
    assert!(SimdLevel::Scalar < SimdLevel::Avx512, "Scalar < Avx512");

    // Stress: repeated backend selection must be idempotent
    for _ in 0..(input.repeat % 16) {
        let _ = backend.requires_gpu();
        let _ = backend.is_compiled();
        let _ = format!("{backend}");
    }
});
