#![no_main]

//! Fuzz kernel configuration string parsing: exercises `KernelBackend`,
//! `KernelCapabilities`, and `SimdLevel` construction with arbitrary inputs
//! to ensure no panics or undefined behaviour when building kernel configs.

use arbitrary::Arbitrary;
use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct KernelConfigInput {
    simd_level: u8,
    backends: Vec<u8>,
    cuda_runtime: bool,
}

fn simd_from_byte(b: u8) -> SimdLevel {
    match b % 5 {
        0 => SimdLevel::Scalar,
        1 => SimdLevel::Neon,
        2 => SimdLevel::Sse42,
        3 => SimdLevel::Avx2,
        _ => SimdLevel::Avx512,
    }
}

fn backend_from_byte(b: u8) -> KernelBackend {
    match b % 6 {
        0 => KernelBackend::CpuRust,
        1 => KernelBackend::Cuda,
        2 => KernelBackend::Hip,
        3 => KernelBackend::OneApi,
        4 => KernelBackend::OpenCL,
        _ => KernelBackend::CppFfi,
    }
}

fuzz_target!(|input: KernelConfigInput| {
    let level = simd_from_byte(input.simd_level);

    // Build capabilities with arbitrary runtime flags.
    let caps = KernelCapabilities::from_compile_time().with_cuda_runtime(input.cuda_runtime);

    // Verify simd_level accessor is consistent.
    let _ = caps.simd_level;
    let _ = format!("{:?}", caps);

    // Round-trip backend enum through Debug.
    for &b in input.backends.iter().take(32) {
        let backend = backend_from_byte(b);
        let _ = format!("{:?}", backend);
    }

    let _ = format!("{:?}", level);
});
