//! Test Fixtures for Issue #453: Strict Quantization Guards
//!
//! Comprehensive test fixtures for validating strict mode quantization enforcement
//! in BitNet-rs neural network inference.
//!
//! ## Module Organization
//!
//! - `quantization_test_data` - Realistic I2S/TL1/TL2 quantization matrices
//! - `device_capabilities` - GPU/CPU device mocks with compute capabilities
//! - `mock_kernels` - Kernel availability registry with ADR-012 naming
//! - `mock_quantized_model` - Mock layers and models for integration testing
//!
//! ## Usage Example
//!
//! ```rust
//! use crate::fixtures::quantization_test_data::*;
//! use crate::fixtures::device_capabilities::*;
//! use crate::fixtures::mock_kernels::*;
//!
//! // Load quantization test matrix
//! let matrix = i2s_matrix_medium();
//! assert_eq!(matrix.shape, (512, 512));
//!
//! // Check device capabilities
//! let gpu = nvidia_a100();
//! assert!(supports_bf16_tensor_cores(&gpu));
//!
//! // Verify kernel availability
//! let registry = MockKernelRegistry::new();
//! assert!(registry.has_native_quantized_kernel(
//!     QuantizationType::I2S,
//!     DeviceType::Gpu
//! ));
//! ```
pub mod quantization_test_data;
pub mod device_capabilities;
pub mod mock_kernels;
pub mod mock_quantized_model;
pub use quantization_test_data::{
    QuantizationTestMatrix, QuantizationType as QuantDataType,
    DeviceType as QuantDeviceType, I2S_ACCURACY_METRICS, TL1_ACCURACY_METRICS,
    TL2_ACCURACY_METRICS,
};
pub use device_capabilities::{
    MockGpuDevice, MockCpuDevice, FallbackScenario, FallbackTrigger, StrictModeBehavior,
};
pub use mock_kernels::{
    MockKernel, MockKernelRegistry, QuantizationType as KernelQuantType,
    DeviceType as KernelDeviceType, is_gpu_kernel, is_quantized_kernel,
    is_fallback_kernel, generate_kernel_id,
};
pub use mock_quantized_model::{
    MockQuantizedLinear, MockBitNetAttention, MockBitNetModel, MockTokenizer,
    MockReceipt, create_test_model_with_fallback, create_test_model_quantized,
};
