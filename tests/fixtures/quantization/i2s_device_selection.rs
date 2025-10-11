// Device-aware I2S quantization backend selection
// Tests specification: docs/explanation/issue-439-spec.md#neural-network-context

use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};

/// Select I2S quantization backend based on compile-time and runtime GPU availability
///
/// This pattern demonstrates Issue #439 AC3 device feature helpers in action:
/// - Compile-time: Check if GPU was compiled with unified predicate
/// - Runtime: Check if GPU hardware is available (respects BITNET_GPU_FAKE)
pub fn select_i2s_backend() -> &'static str {
    if gpu_compiled() && gpu_available_runtime() {
        "i2s_gpu"
    } else {
        "i2s_cpu"
    }
}

/// I2S quantization with automatic device selection
pub fn i2s_quantize_auto(input: &[f32]) -> Vec<i8> {
    let backend = select_i2s_backend();

    match backend {
        "i2s_gpu" => i2s_quantize_gpu(input),
        "i2s_cpu" => i2s_quantize_cpu(input),
        _ => unreachable!("Unknown backend: {}", backend),
    }
}

/// GPU I2S quantization (only compiled when feature="gpu" or feature="cuda")
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn i2s_quantize_gpu(input: &[f32]) -> Vec<i8> {
    println!("[GPU] I2S quantization: {} elements", input.len());
    // Simplified: Real implementation would launch CUDA kernel
    vec![0; input.len()]
}

/// CPU I2S quantization (always available)
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn i2s_quantize_gpu(_input: &[f32]) -> Vec<i8> {
    panic!("GPU backend not compiled - use --features gpu");
}

/// CPU I2S quantization with SIMD optimization
fn i2s_quantize_cpu(input: &[f32]) -> Vec<i8> {
    #[cfg(target_arch = "x86_64")]
    {
        println!("[CPU/AVX2] I2S quantization: {} elements", input.len());
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("[CPU/NEON] I2S quantization: {} elements", input.len());
    }

    // Simplified: Real implementation would use SIMD
    vec![0; input.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_selection_respects_compilation() {
        let backend = select_i2s_backend();

        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            // GPU compiled - backend depends on runtime availability
            // (May be "i2s_gpu" or "i2s_cpu" depending on BITNET_GPU_FAKE)
            assert!(
                backend == "i2s_gpu" || backend == "i2s_cpu",
                "GPU compiled: backend should be gpu or cpu"
            );
        }

        #[cfg(not(any(feature = "gpu", feature = "cuda")))]
        {
            // GPU not compiled - must use CPU
            assert_eq!(backend, "i2s_cpu", "CPU-only build must use cpu backend");
        }
    }

    #[test]
    fn test_auto_quantization_works() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let quantized = i2s_quantize_auto(&input);

        assert_eq!(quantized.len(), input.len());
    }
}
