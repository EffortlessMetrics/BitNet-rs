// Device-aware TL1/TL2 quantization backend selection
// Tests specification: docs/explanation/issue-439-spec.md#neural-network-context

use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};

/// Select TL1 quantization backend (ARM NEON optimized)
///
/// TL1 is optimized for ARM NEON, but can also run on GPU.
/// Demonstrates architecture-specific + device-aware selection.
pub fn select_tl1_backend() -> &'static str {
    // Prefer GPU if available
    if gpu_compiled() && gpu_available_runtime() {
        return "tl1_gpu";
    }

    // Fallback to CPU with architecture-specific SIMD
    #[cfg(target_arch = "aarch64")]
    {
        "tl1_cpu_neon"
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        "tl1_cpu_generic"
    }
}

/// Select TL2 quantization backend (x86 AVX optimized)
///
/// TL2 is optimized for x86 AVX2/AVX-512, but can also run on GPU.
pub fn select_tl2_backend() -> &'static str {
    // Prefer GPU if available
    if gpu_compiled() && gpu_available_runtime() {
        return "tl2_gpu";
    }

    // Fallback to CPU with architecture-specific SIMD
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            "tl2_cpu_avx512"
        } else if is_x86_feature_detected!("avx2") {
            "tl2_cpu_avx2"
        } else {
            "tl2_cpu_generic"
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        "tl2_cpu_generic"
    }
}

/// TL1 quantization with automatic device/architecture selection
pub fn tl1_quantize_auto(input: &[f32]) -> Vec<u8> {
    let backend = select_tl1_backend();

    match backend {
        "tl1_gpu" => tl1_quantize_gpu(input),
        "tl1_cpu_neon" => tl1_quantize_cpu_neon(input),
        "tl1_cpu_generic" => tl1_quantize_cpu_generic(input),
        _ => unreachable!("Unknown TL1 backend: {}", backend),
    }
}

/// TL2 quantization with automatic device/architecture selection
pub fn tl2_quantize_auto(input: &[f32]) -> Vec<u8> {
    let backend = select_tl2_backend();

    match backend {
        "tl2_gpu" => tl2_quantize_gpu(input),
        "tl2_cpu_avx512" => tl2_quantize_cpu_avx512(input),
        "tl2_cpu_avx2" => tl2_quantize_cpu_avx2(input),
        "tl2_cpu_generic" => tl2_quantize_cpu_generic(input),
        _ => unreachable!("Unknown TL2 backend: {}", backend),
    }
}

// GPU implementations (unified feature gate)
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn tl1_quantize_gpu(input: &[f32]) -> Vec<u8> {
    println!("[GPU] TL1 quantization: {} elements", input.len());
    vec![0; input.len()]
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
fn tl2_quantize_gpu(input: &[f32]) -> Vec<u8> {
    println!("[GPU] TL2 quantization: {} elements", input.len());
    vec![0; input.len()]
}

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn tl1_quantize_gpu(_input: &[f32]) -> Vec<u8> {
    panic!("GPU backend not compiled");
}

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn tl2_quantize_gpu(_input: &[f32]) -> Vec<u8> {
    panic!("GPU backend not compiled");
}

// CPU implementations
#[cfg(target_arch = "aarch64")]
fn tl1_quantize_cpu_neon(input: &[f32]) -> Vec<u8> {
    println!("[CPU/NEON] TL1 quantization: {} elements", input.len());
    vec![0; input.len()]
}

#[cfg(not(target_arch = "aarch64"))]
fn tl1_quantize_cpu_neon(_input: &[f32]) -> Vec<u8> {
    tl1_quantize_cpu_generic(_input)
}

#[cfg(target_arch = "x86_64")]
fn tl2_quantize_cpu_avx512(input: &[f32]) -> Vec<u8> {
    println!("[CPU/AVX-512] TL2 quantization: {} elements", input.len());
    vec![0; input.len()]
}

#[cfg(target_arch = "x86_64")]
fn tl2_quantize_cpu_avx2(input: &[f32]) -> Vec<u8> {
    println!("[CPU/AVX2] TL2 quantization: {} elements", input.len());
    vec![0; input.len()]
}

fn tl1_quantize_cpu_generic(input: &[f32]) -> Vec<u8> {
    println!("[CPU/Generic] TL1 quantization: {} elements", input.len());
    vec![0; input.len()]
}

fn tl2_quantize_cpu_generic(input: &[f32]) -> Vec<u8> {
    println!("[CPU/Generic] TL2 quantization: {} elements", input.len());
    vec![0; input.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tl1_backend_selection() {
        let backend = select_tl1_backend();
        println!("TL1 backend: {}", backend);

        // Verify backend is valid
        assert!(
            backend.starts_with("tl1_"),
            "TL1 backend should start with 'tl1_'"
        );
    }

    #[test]
    fn test_tl2_backend_selection() {
        let backend = select_tl2_backend();
        println!("TL2 backend: {}", backend);

        // Verify backend is valid
        assert!(
            backend.starts_with("tl2_"),
            "TL2 backend should start with 'tl2_'"
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_tl1_prefers_neon_on_arm() {
        let backend = select_tl1_backend();

        #[cfg(not(any(feature = "gpu", feature = "cuda")))]
        assert_eq!(backend, "tl1_cpu_neon", "ARM should use NEON backend");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_tl2_uses_avx_on_x86() {
        let backend = select_tl2_backend();

        #[cfg(not(any(feature = "gpu", feature = "cuda")))]
        assert!(
            backend.contains("avx") || backend == "tl2_cpu_generic",
            "x86 should use AVX or generic backend"
        );
    }
}
