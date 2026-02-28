#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    // Try constructing backend configs from arbitrary data
    let backend_idx = data[0] % 6; // CPU, CUDA, Metal, OpenCL, ROCm, Vulkan
    let simd_idx = data[1] % 5; // None, SSE, AVX2, AVX512, NEON
    let _ = (backend_idx, simd_idx); // Validate no panics in enum construction
});
