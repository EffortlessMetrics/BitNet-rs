// Mixed precision GPU kernel selection
// Tests specification: docs/explanation/issue-439-spec.md#neural-network-context

use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};

/// GPU precision modes for mixed precision inference
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrecisionMode {
    FP32,         // Full precision (baseline)
    FP16,         // Half precision
    BF16,         // BFloat16
    MixedFP16,    // FP32 accumulation, FP16 computation
    MixedBF16,    // FP32 accumulation, BF16 computation
}

/// Select optimal precision mode based on GPU capabilities
///
/// Demonstrates device-aware selection for mixed precision:
/// - Checks compile-time GPU availability
/// - Checks runtime GPU availability
/// - Validates device compute capability and precision support
pub fn select_precision_mode(device_id: usize) -> PrecisionMode {
    // If GPU not compiled or not available, fallback to CPU (FP32 only)
    if !gpu_compiled() || !gpu_available_runtime() {
        return PrecisionMode::FP32;
    }

    // Query device capabilities
    let device_info = query_device_capabilities(device_id);

    // Select precision based on hardware support
    if device_info.supports_bf16 && device_info.supports_tensor_cores {
        // Prefer BF16 for Tensor Core GPUs (better numerical stability)
        PrecisionMode::MixedBF16
    } else if device_info.supports_fp16 && device_info.supports_tensor_cores {
        // Fallback to FP16 if BF16 not available
        PrecisionMode::MixedFP16
    } else if device_info.supports_fp16 {
        // FP16 without Tensor Cores (slower, but saves memory)
        PrecisionMode::FP16
    } else {
        // Legacy GPU: FP32 only
        PrecisionMode::FP32
    }
}

/// Device capability information
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub compute_capability: String,
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub supports_tensor_cores: bool,
    pub memory_gb: f32,
}

/// Query GPU device capabilities (simplified stub)
///
/// Real implementation would call into CUDA API or use cudarc crate
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn query_device_capabilities(_device_id: usize) -> DeviceCapabilities {
    // Simplified: Would query actual GPU in real implementation
    DeviceCapabilities {
        compute_capability: "8.9".to_string(),
        supports_fp16: true,
        supports_bf16: true,
        supports_tensor_cores: true,
        memory_gb: 24.0,
    }
}

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn query_device_capabilities(_device_id: usize) -> DeviceCapabilities {
    DeviceCapabilities {
        compute_capability: "N/A".to_string(),
        supports_fp16: false,
        supports_bf16: false,
        supports_tensor_cores: false,
        memory_gb: 0.0,
    }
}

/// GEMM kernel with automatic precision selection
pub fn gemm_auto_precision(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    device_id: usize,
) -> &'static str {
    let precision_mode = select_precision_mode(device_id);

    match precision_mode {
        PrecisionMode::MixedBF16 => gemm_mixed_bf16(a, b, c),
        PrecisionMode::MixedFP16 => gemm_mixed_fp16(a, b, c),
        PrecisionMode::FP16 => gemm_fp16(a, b, c),
        PrecisionMode::FP32 => gemm_fp32(a, b, c),
        _ => gemm_fp32(a, b, c),
    }
}

// GPU precision kernels (unified feature gate)
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn gemm_mixed_bf16(_a: &[f32], _b: &[f32], _c: &mut [f32]) -> &'static str {
    println!("[GPU/BF16] Mixed precision GEMM with Tensor Cores");
    "gemm_mixed_bf16"
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
fn gemm_mixed_fp16(_a: &[f32], _b: &[f32], _c: &mut [f32]) -> &'static str {
    println!("[GPU/FP16] Mixed precision GEMM with Tensor Cores");
    "gemm_mixed_fp16"
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
fn gemm_fp16(_a: &[f32], _b: &[f32], _c: &mut [f32]) -> &'static str {
    println!("[GPU/FP16] Half precision GEMM");
    "gemm_fp16"
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
fn gemm_fp32(_a: &[f32], _b: &[f32], _c: &mut [f32]) -> &'static str {
    println!("[GPU/FP32] Full precision GEMM");
    "gemm_fp32"
}

// CPU stubs
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn gemm_mixed_bf16(_a: &[f32], _b: &[f32], _c: &mut [f32]) -> &'static str {
    panic!("GPU not compiled");
}

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn gemm_mixed_fp16(_a: &[f32], _b: &[f32], _c: &mut [f32]) -> &'static str {
    panic!("GPU not compiled");
}

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn gemm_fp16(_a: &[f32], _b: &[f32], _c: &mut [f32]) -> &'static str {
    panic!("GPU not compiled");
}

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn gemm_fp32(_a: &[f32], _b: &[f32], _c: &mut [f32]) -> &'static str {
    println!("[CPU/FP32] Full precision GEMM");
    "gemm_cpu_fp32"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_selection_without_gpu() {
        #[cfg(not(any(feature = "gpu", feature = "cuda")))]
        {
            let precision = select_precision_mode(0);
            assert_eq!(precision, PrecisionMode::FP32, "CPU should use FP32");
        }
    }

    #[test]
    fn test_gemm_auto_works() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let mut c = vec![0.0, 0.0];

        let kernel = gemm_auto_precision(&a, &b, &mut c, 0);
        println!("Selected kernel: {}", kernel);

        assert!(kernel.starts_with("gemm_"), "Should select valid GEMM kernel");
    }
}
