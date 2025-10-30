//! Mock Kernel Registry for Issue #453 Strict Quantization Guards
//!
//! Provides realistic kernel availability mocks with proper ADR-012 naming
//! conventions for testing GPU/CPU kernel selection and fallback scenarios.
//!
//! All kernel IDs follow the naming pattern:
//! - GPU: `gemm_*`, `wmma_*`, `cuda_*`, `i2s_gpu_*`, `tl1_gpu_*`, `tl2_gpu_*`
//! - CPU: `i2s_gemv`, `tl1_neon_*`, `tl2_avx_*`, `quantized_matmul_*`
#![allow(dead_code)]
/// Mock kernel with availability status
#[derive(Debug, Clone)]
pub struct MockKernel {
    pub kernel_id: &'static str,
    pub quantization_type: QuantizationType,
    pub device_type: DeviceType,
    pub precision: KernelPrecision,
    pub available: bool,
    pub min_compute_capability: Option<(u32, u32)>,
    pub required_simd_features: Vec<SimdFeature>,
    pub description: &'static str,
}
/// Quantization type for kernel selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    I2S,
    TL1,
    TL2,
    FP32,
}
/// Device type for kernel execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Gpu,
    Cpu,
}
/// Kernel precision mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelPrecision {
    FP32,
    FP16,
    BF16,
    INT8,
    INT2,
}
/// SIMD feature requirements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdFeature {
    Avx2,
    Avx512,
    Neon,
    Sve,
}
/// Kernel availability registry
pub struct MockKernelRegistry {
    pub kernels: Vec<MockKernel>,
}
impl MockKernelRegistry {
    /// Create a new kernel registry with all available kernels
    pub fn new() -> Self {
        Self {
            kernels: vec![
                gpu_kernel_gemm_fp16(), gpu_kernel_gemm_bf16(),
                gpu_kernel_wmma_matmul_fp16(), gpu_kernel_wmma_matmul_bf16(),
                gpu_kernel_i2s_quantize(), gpu_kernel_i2s_pack(),
                gpu_kernel_i2s_matmul(), gpu_kernel_tl1_pack(), gpu_kernel_tl2_pack(),
                gpu_kernel_cuda_sync(), cpu_kernel_i2s_gemv(),
                cpu_kernel_tl1_neon_pack(), cpu_kernel_tl1_neon_matmul(),
                cpu_kernel_tl2_avx_matmul(), cpu_kernel_tl2_avx512_pack(),
                cpu_kernel_quantized_matmul_i2s(), fallback_kernel_dequant_i2s(),
                fallback_kernel_fp32_matmul(), fallback_kernel_scalar_matmul(),
            ],
        }
    }
    /// Find kernel by ID
    pub fn find_kernel(&self, kernel_id: &str) -> Option<&MockKernel> {
        self.kernels.iter().find(|k| k.kernel_id == kernel_id)
    }
    /// Check if kernel is available for device
    pub fn is_kernel_available(&self, kernel_id: &str, device_type: DeviceType) -> bool {
        self.find_kernel(kernel_id)
            .map(|k| k.available && k.device_type == device_type)
            .unwrap_or(false)
    }
    /// Get all available GPU kernels
    pub fn available_gpu_kernels(&self) -> Vec<&MockKernel> {
        self.kernels
            .iter()
            .filter(|k| k.device_type == DeviceType::Gpu && k.available)
            .collect()
    }
    /// Get all available CPU kernels
    pub fn available_cpu_kernels(&self) -> Vec<&MockKernel> {
        self.kernels
            .iter()
            .filter(|k| k.device_type == DeviceType::Cpu && k.available)
            .collect()
    }
    /// Check if quantization type has native kernels
    pub fn has_native_quantized_kernel(
        &self,
        qtype: QuantizationType,
        device: DeviceType,
    ) -> bool {
        self.kernels
            .iter()
            .any(|k| {
                k.quantization_type == qtype && k.device_type == device && k.available
                    && !is_fallback_kernel(k.kernel_id)
            })
    }
}
/// GPU FP16 GEMM kernel
#[cfg(feature = "gpu")]
fn gpu_kernel_gemm_fp16() -> MockKernel {
    MockKernel {
        kernel_id: "gemm_fp16",
        quantization_type: QuantizationType::I2S,
        device_type: DeviceType::Gpu,
        precision: KernelPrecision::FP16,
        available: true,
        min_compute_capability: Some((7, 0)),
        required_simd_features: vec![],
        description: "FP16 GEMM with quantized I2S weights on GPU",
    }
}
/// GPU BF16 GEMM kernel
#[cfg(feature = "gpu")]
fn gpu_kernel_gemm_bf16() -> MockKernel {
    MockKernel {
        kernel_id: "gemm_bf16",
        quantization_type: QuantizationType::I2S,
        device_type: DeviceType::Gpu,
        precision: KernelPrecision::BF16,
        available: true,
        min_compute_capability: Some((8, 0)),
        required_simd_features: vec![],
        description: "BF16 GEMM with quantized I2S weights on GPU",
    }
}
/// GPU Tensor Core FP16 matmul
#[cfg(feature = "gpu")]
fn gpu_kernel_wmma_matmul_fp16() -> MockKernel {
    MockKernel {
        kernel_id: "wmma_matmul",
        quantization_type: QuantizationType::I2S,
        device_type: DeviceType::Gpu,
        precision: KernelPrecision::FP16,
        available: true,
        min_compute_capability: Some((7, 0)),
        required_simd_features: vec![],
        description: "Tensor Core mixed precision matmul with I2S quantization",
    }
}
/// GPU Tensor Core BF16 matmul
#[cfg(feature = "gpu")]
fn gpu_kernel_wmma_matmul_bf16() -> MockKernel {
    MockKernel {
        kernel_id: "wmma_bf16",
        quantization_type: QuantizationType::I2S,
        device_type: DeviceType::Gpu,
        precision: KernelPrecision::BF16,
        available: true,
        min_compute_capability: Some((8, 0)),
        required_simd_features: vec![],
        description: "BF16 Tensor Core matmul with I2S quantization",
    }
}
/// GPU I2S quantization kernel
#[cfg(feature = "gpu")]
fn gpu_kernel_i2s_quantize() -> MockKernel {
    MockKernel {
        kernel_id: "i2s_gpu_quantize",
        quantization_type: QuantizationType::I2S,
        device_type: DeviceType::Gpu,
        precision: KernelPrecision::INT2,
        available: true,
        min_compute_capability: Some((6, 1)),
        required_simd_features: vec![],
        description: "I2S GPU quantization operation (2-bit signed)",
    }
}
/// GPU I2S packing kernel
#[cfg(feature = "gpu")]
fn gpu_kernel_i2s_pack() -> MockKernel {
    MockKernel {
        kernel_id: "i2s_gpu_pack",
        quantization_type: QuantizationType::I2S,
        device_type: DeviceType::Gpu,
        precision: KernelPrecision::INT2,
        available: true,
        min_compute_capability: Some((6, 1)),
        required_simd_features: vec![],
        description: "I2S GPU bit-packing operation",
    }
}
/// GPU I2S matmul kernel
#[cfg(feature = "gpu")]
fn gpu_kernel_i2s_matmul() -> MockKernel {
    MockKernel {
        kernel_id: "i2s_gpu_matmul",
        quantization_type: QuantizationType::I2S,
        device_type: DeviceType::Gpu,
        precision: KernelPrecision::INT2,
        available: true,
        min_compute_capability: Some((7, 0)),
        required_simd_features: vec![],
        description: "I2S GPU matrix multiplication",
    }
}
/// GPU TL1 packing kernel
#[cfg(feature = "gpu")]
fn gpu_kernel_tl1_pack() -> MockKernel {
    MockKernel {
        kernel_id: "tl1_gpu_pack",
        quantization_type: QuantizationType::TL1,
        device_type: DeviceType::Gpu,
        precision: KernelPrecision::INT8,
        available: true,
        min_compute_capability: Some((6, 1)),
        required_simd_features: vec![],
        description: "TL1 GPU table lookup packing",
    }
}
/// GPU TL2 packing kernel
#[cfg(feature = "gpu")]
fn gpu_kernel_tl2_pack() -> MockKernel {
    MockKernel {
        kernel_id: "tl2_gpu_pack",
        quantization_type: QuantizationType::TL2,
        device_type: DeviceType::Gpu,
        precision: KernelPrecision::INT8,
        available: true,
        min_compute_capability: Some((6, 1)),
        required_simd_features: vec![],
        description: "TL2 GPU table lookup packing",
    }
}
/// GPU CUDA synchronization kernel
#[cfg(feature = "gpu")]
fn gpu_kernel_cuda_sync() -> MockKernel {
    MockKernel {
        kernel_id: "cuda_sync",
        quantization_type: QuantizationType::I2S,
        device_type: DeviceType::Gpu,
        precision: KernelPrecision::FP32,
        available: true,
        min_compute_capability: Some((6, 0)),
        required_simd_features: vec![],
        description: "CUDA stream synchronization",
    }
}
/// CPU I2S GEMV kernel
#[cfg(feature = "cpu")]
fn cpu_kernel_i2s_gemv() -> MockKernel {
    MockKernel {
        kernel_id: "i2s_gemv",
        quantization_type: QuantizationType::I2S,
        device_type: DeviceType::Cpu,
        precision: KernelPrecision::INT2,
        available: true,
        min_compute_capability: None,
        required_simd_features: vec![],
        description: "I2S CPU GEMV (general vector multiplication)",
    }
}
/// CPU TL1 NEON packing kernel
#[cfg(feature = "cpu")]
fn cpu_kernel_tl1_neon_pack() -> MockKernel {
    MockKernel {
        kernel_id: "tl1_neon_pack",
        quantization_type: QuantizationType::TL1,
        device_type: DeviceType::Cpu,
        precision: KernelPrecision::INT8,
        available: true,
        min_compute_capability: None,
        required_simd_features: vec![SimdFeature::Neon],
        description: "TL1 ARM NEON table lookup packing",
    }
}
/// CPU TL1 NEON matmul kernel
#[cfg(feature = "cpu")]
fn cpu_kernel_tl1_neon_matmul() -> MockKernel {
    MockKernel {
        kernel_id: "tl1_neon_matmul",
        quantization_type: QuantizationType::TL1,
        device_type: DeviceType::Cpu,
        precision: KernelPrecision::INT8,
        available: true,
        min_compute_capability: None,
        required_simd_features: vec![SimdFeature::Neon],
        description: "TL1 ARM NEON matrix multiplication",
    }
}
/// CPU TL2 AVX matmul kernel
#[cfg(feature = "cpu")]
fn cpu_kernel_tl2_avx_matmul() -> MockKernel {
    MockKernel {
        kernel_id: "tl2_avx_matmul",
        quantization_type: QuantizationType::TL2,
        device_type: DeviceType::Cpu,
        precision: KernelPrecision::INT8,
        available: true,
        min_compute_capability: None,
        required_simd_features: vec![SimdFeature::Avx2],
        description: "TL2 x86 AVX2 table lookup matmul",
    }
}
/// CPU TL2 AVX-512 packing kernel
#[cfg(feature = "cpu")]
fn cpu_kernel_tl2_avx512_pack() -> MockKernel {
    MockKernel {
        kernel_id: "tl2_avx512_pack",
        quantization_type: QuantizationType::TL2,
        device_type: DeviceType::Cpu,
        precision: KernelPrecision::INT8,
        available: true,
        min_compute_capability: None,
        required_simd_features: vec![SimdFeature::Avx512],
        description: "TL2 x86 AVX-512 table lookup packing (enhanced)",
    }
}
/// CPU quantized matmul (I2S)
#[cfg(feature = "cpu")]
fn cpu_kernel_quantized_matmul_i2s() -> MockKernel {
    MockKernel {
        kernel_id: "quantized_matmul_i2s",
        quantization_type: QuantizationType::I2S,
        device_type: DeviceType::Cpu,
        precision: KernelPrecision::INT2,
        available: true,
        min_compute_capability: None,
        required_simd_features: vec![],
        description: "CPU I2S quantized matrix multiplication",
    }
}
/// Fallback I2S dequantization
fn fallback_kernel_dequant_i2s() -> MockKernel {
    MockKernel {
        kernel_id: "dequant_i2s",
        quantization_type: QuantizationType::FP32,
        device_type: DeviceType::Cpu,
        precision: KernelPrecision::FP32,
        available: true,
        min_compute_capability: None,
        required_simd_features: vec![],
        description: "Fallback I2S dequantization (FP32 path)",
    }
}
/// Fallback FP32 matmul
fn fallback_kernel_fp32_matmul() -> MockKernel {
    MockKernel {
        kernel_id: "fp32_matmul",
        quantization_type: QuantizationType::FP32,
        device_type: DeviceType::Cpu,
        precision: KernelPrecision::FP32,
        available: true,
        min_compute_capability: None,
        required_simd_features: vec![],
        description: "Fallback FP32 matrix multiplication (no quantization)",
    }
}
/// Fallback scalar matmul (no SIMD)
fn fallback_kernel_scalar_matmul() -> MockKernel {
    MockKernel {
        kernel_id: "scalar_matmul",
        quantization_type: QuantizationType::FP32,
        device_type: DeviceType::Cpu,
        precision: KernelPrecision::FP32,
        available: true,
        min_compute_capability: None,
        required_simd_features: vec![],
        description: "Fallback scalar matmul (no SIMD, no quantization)",
    }
}
/// Check if kernel ID represents a GPU kernel
pub fn is_gpu_kernel(kernel_id: &str) -> bool {
    kernel_id.starts_with("gemm_") || kernel_id.starts_with("wmma_")
        || kernel_id.starts_with("cuda_") || kernel_id.starts_with("i2s_gpu_")
        || kernel_id.starts_with("tl1_gpu_") || kernel_id.starts_with("tl2_gpu_")
}
/// Check if kernel ID represents a quantized kernel
pub fn is_quantized_kernel(kernel_id: &str) -> bool {
    is_gpu_kernel(kernel_id) || kernel_id.starts_with("i2s_")
        || kernel_id.starts_with("tl1_") || kernel_id.starts_with("tl2_")
        || kernel_id.starts_with("quantized_")
}
/// Check if kernel ID represents a fallback kernel
pub fn is_fallback_kernel(kernel_id: &str) -> bool {
    kernel_id.starts_with("dequant_") || kernel_id.starts_with("fp32_")
        || kernel_id.starts_with("fallback_") || kernel_id == "scalar_matmul"
}
/// Generate realistic kernel ID for receipt
pub fn generate_kernel_id(
    qtype: QuantizationType,
    device: DeviceType,
    precision: KernelPrecision,
) -> String {
    match (qtype, device, precision) {
        (QuantizationType::I2S, DeviceType::Gpu, KernelPrecision::FP16) => {
            "gemm_fp16".to_string()
        }
        (QuantizationType::I2S, DeviceType::Gpu, KernelPrecision::BF16) => {
            "gemm_bf16".to_string()
        }
        (QuantizationType::I2S, DeviceType::Gpu, KernelPrecision::INT2) => {
            "i2s_gpu_quantize".to_string()
        }
        (QuantizationType::I2S, DeviceType::Cpu, _) => "i2s_gemv".to_string(),
        (QuantizationType::TL1, DeviceType::Cpu, _) => "tl1_neon_matmul".to_string(),
        (QuantizationType::TL2, DeviceType::Cpu, _) => "tl2_avx_matmul".to_string(),
        (QuantizationType::FP32, _, _) => "fp32_matmul".to_string(),
        _ => "scalar_matmul".to_string(),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_kernel_registry_creation() {
        let registry = MockKernelRegistry::new();
        assert!(registry.kernels.len() > 0);
    }
    #[test]
    fn test_kernel_availability_check() {
        let registry = MockKernelRegistry::new();
        #[cfg(feature = "gpu")]
        {
            assert!(registry.is_kernel_available("gemm_fp16", DeviceType::Gpu));
            assert!(registry.is_kernel_available("i2s_gpu_quantize", DeviceType::Gpu));
        }
        #[cfg(feature = "cpu")]
        {
            assert!(registry.is_kernel_available("i2s_gemv", DeviceType::Cpu));
            assert!(registry.is_kernel_available("tl2_avx_matmul", DeviceType::Cpu));
        }
    }
    #[test]
    fn test_kernel_id_pattern_matching() {
        assert!(is_gpu_kernel("gemm_fp16"));
        assert!(is_gpu_kernel("wmma_matmul"));
        assert!(is_gpu_kernel("i2s_gpu_quantize"));
        assert!(! is_gpu_kernel("i2s_gemv"));
        assert!(is_quantized_kernel("gemm_fp16"));
        assert!(is_quantized_kernel("i2s_gemv"));
        assert!(is_quantized_kernel("tl1_neon_pack"));
        assert!(! is_quantized_kernel("fp32_matmul"));
        assert!(is_fallback_kernel("dequant_i2s"));
        assert!(is_fallback_kernel("fp32_matmul"));
        assert!(! is_fallback_kernel("gemm_fp16"));
    }
    #[test]
    fn test_kernel_id_generation() {
        let gpu_i2s_fp16 = generate_kernel_id(
            QuantizationType::I2S,
            DeviceType::Gpu,
            KernelPrecision::FP16,
        );
        assert_eq!(gpu_i2s_fp16, "gemm_fp16");
        let cpu_i2s = generate_kernel_id(
            QuantizationType::I2S,
            DeviceType::Cpu,
            KernelPrecision::INT2,
        );
        assert_eq!(cpu_i2s, "i2s_gemv");
    }
    #[test]
    fn test_has_native_quantized_kernel() {
        let registry = MockKernelRegistry::new();
        #[cfg(feature = "gpu")]
        {
            assert!(
                registry.has_native_quantized_kernel(QuantizationType::I2S,
                DeviceType::Gpu)
            );
        }
        #[cfg(feature = "cpu")]
        {
            assert!(
                registry.has_native_quantized_kernel(QuantizationType::I2S,
                DeviceType::Cpu)
            );
            assert!(
                registry.has_native_quantized_kernel(QuantizationType::TL1,
                DeviceType::Cpu)
            );
            assert!(
                registry.has_native_quantized_kernel(QuantizationType::TL2,
                DeviceType::Cpu)
            );
        }
    }
}
