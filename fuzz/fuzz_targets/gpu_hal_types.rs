#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::{KernelBackend, KernelCapabilities, SimdLevel};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct HalInput {
    cpu_rust: bool,
    cuda_compiled: bool,
    cuda_runtime: bool,
    hip_compiled: bool,
    hip_runtime: bool,
    oneapi_compiled: bool,
    oneapi_runtime: bool,
    cpp_ffi: bool,
    simd_idx: u8,
    backend_idx: u8,
}

fuzz_target!(|input: HalInput| {
    let simd_levels =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];
    let simd = simd_levels[input.simd_idx as usize % simd_levels.len()];

    let backends = [
        KernelBackend::CpuRust,
        KernelBackend::Cuda,
        KernelBackend::Hip,
        KernelBackend::OneApi,
        KernelBackend::CppFfi,
    ];
    let backend = backends[input.backend_idx as usize % backends.len()];

    // KernelBackend methods must never panic.
    let _ = backend.requires_gpu();
    let _ = backend.is_compiled();
    let _ = format!("{backend}");

    // SimdLevel Display and ordering must never panic.
    let _ = format!("{simd}");
    let _ = simd.cmp(&SimdLevel::Avx2);

    // Build KernelCapabilities from arbitrary flags.
    let caps = KernelCapabilities {
        cpu_rust: input.cpu_rust,
        cuda_compiled: input.cuda_compiled,
        cuda_runtime: input.cuda_runtime,
        hip_compiled: input.hip_compiled,
        hip_runtime: input.hip_runtime,
        oneapi_compiled: input.oneapi_compiled,
        oneapi_runtime: input.oneapi_runtime,
        cpp_ffi: input.cpp_ffi,
        simd_level: simd,
    };

    // All capability queries must never panic.
    let compiled = caps.compiled_backends();
    let _ = caps.best_available();
    let _ = format!("{caps:?}");
    let _ = compiled.len();

    // Builder chain methods must never panic.
    let chained = KernelCapabilities::from_compile_time()
        .with_cuda_runtime(input.cuda_runtime)
        .with_hip_runtime(input.hip_runtime)
        .with_oneapi_runtime(input.oneapi_runtime)
        .with_cpp_ffi(input.cpp_ffi);
    let _ = chained.compiled_backends();
    let _ = chained.best_available();
    let _ = chained.summary();

    // Verify requires_gpu consistency for all backends.
    for b in &backends {
        let needs_gpu = b.requires_gpu();
        if needs_gpu {
            assert!(
                matches!(b, KernelBackend::Cuda | KernelBackend::Hip | KernelBackend::OneApi),
                "requires_gpu mismatch for {b:?}",
            );
        }
    }
});
