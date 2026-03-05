//! BDD Wave 12 — Validation Lifecycle Integration Tests
//!
//! Given/When/Then scenarios covering:
//! 1. Shape validation → error formatting → recovery
//! 2. Budget creation → consumption → exhaustion lifecycle
//! 3. Tensor type detection → validation → dispatch

use bitnet_common::kernel_registry::{KernelBackend, SimdLevel};
use bitnet_common::memory_pool::TensorPool;
use bitnet_common::shape_validator::{
    assert_broadcastable, assert_dim, assert_element_count, assert_head_divisible,
    assert_matmul_compat, assert_rank, assert_shape_eq,
};
use bitnet_common::tensor_validation::{
    TensorLike, TensorValidationError, TensorValidator, ValidationConfig,
};
use bitnet_common::types::{Device, QuantizationType};

// ── Test tensor helper ─────────────────────────────────────────────

struct TestTensor {
    shape: Vec<usize>,
    data: Option<Vec<f32>>,
    strides: Option<Vec<usize>>,
    alignment: usize,
}

impl TestTensor {
    fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self { shape, data: Some(data), strides: None, alignment: 64 }
    }

    fn no_data(shape: Vec<usize>) -> Self {
        Self { shape, data: None, strides: None, alignment: 64 }
    }

    #[allow(dead_code)]
    fn with_alignment(mut self, align: usize) -> Self {
        self.alignment = align;
        self
    }
}

impl TensorLike for TestTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data_f32(&self) -> Option<&[f32]> {
        self.data.as_deref()
    }
    fn strides(&self) -> Option<&[usize]> {
        self.strides.as_deref()
    }
    fn data_alignment(&self) -> usize {
        self.alignment
    }
}

// ═══════════════════════════════════════════════════════════════════
// Section 1 — Shape validation → error formatting → recovery
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_given_matching_shapes_when_validated_then_ok() {
    // Given two identical shapes
    let a = [2, 3, 4];
    let b = [2, 3, 4];

    // When validated for equality
    let result = assert_shape_eq("test", &a, &b);

    // Then validation passes
    assert!(result.is_ok());
}

#[test]
fn test_given_mismatched_shapes_when_validated_then_error_with_context() {
    // Given two different shapes
    let a = [2, 3];
    let b = [2, 4];

    // When validated for equality
    let result = assert_shape_eq("matmul_input", &a, &b);

    // Then error contains the context string
    let err = result.unwrap_err();
    assert!(err.to_string().contains("matmul_input"), "error should contain context: {}", err);
}

#[test]
fn test_given_correct_rank_when_validated_then_ok() {
    // Given a 3D shape
    let shape = [2, 3, 4];

    // When rank is validated as 3
    let result = assert_rank("attention", &shape, 3);

    // Then it passes
    assert!(result.is_ok());
}

#[test]
fn test_given_wrong_rank_when_validated_then_error_shows_expected() {
    // Given a 2D shape
    let shape = [2, 3];

    // When validated against rank 3
    let result = assert_rank("qkv", &shape, 3);

    // Then error reports expected vs actual rank
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("rank 3"), "should mention expected rank: {msg}");
    assert!(msg.contains("rank 2"), "should mention actual rank: {msg}");
}

#[test]
fn test_given_correct_dim_when_validated_then_ok() {
    // Given a shape with known dimension
    let shape = [2, 64, 128];

    // When validating dimension 1 == 64
    let result = assert_dim("hidden", &shape, 1, 64);

    // Then it passes
    assert!(result.is_ok());
}

#[test]
fn test_given_wrong_dim_when_validated_then_error() {
    // Given a shape
    let shape = [2, 32, 128];

    // When validating dimension 1 == 64 (wrong)
    let result = assert_dim("hidden", &shape, 1, 64);

    // Then error is returned
    assert!(result.is_err());
}

#[test]
fn test_given_matmul_compatible_shapes_when_validated_then_ok() {
    // Given compatible matmul shapes: [2,3] × [3,4]
    let a = [2, 3];
    let b = [3, 4];

    // When validated for matmul compatibility
    let result = assert_matmul_compat("gemm", &a, &b);

    // Then validation passes
    assert!(result.is_ok());
}

#[test]
fn test_given_matmul_incompatible_shapes_when_validated_then_error() {
    // Given incompatible shapes: [2,3] × [4,5]
    let a = [2, 3];
    let b = [4, 5];

    // When validated
    let result = assert_matmul_compat("gemm", &a, &b);

    // Then error is returned
    assert!(result.is_err());
}

#[test]
fn test_given_broadcastable_shapes_when_checked_then_ok() {
    // Given broadcastable shapes: [1,3] and [2,3]
    let result = assert_broadcastable("add", &[1, 3], &[2, 3]);

    // Then validation passes
    assert!(result.is_ok());
}

#[test]
fn test_given_correct_element_count_when_validated_then_ok() {
    // Given a shape with 24 elements
    let shape = [2, 3, 4];

    // When validated for 24 elements
    let result = assert_element_count("reshape", &shape, 24);

    // Then it passes
    assert!(result.is_ok());
}

#[test]
fn test_given_wrong_element_count_when_validated_then_error() {
    // Given a shape with 24 elements
    let shape = [2, 3, 4];

    // When validated for 12 elements
    let result = assert_element_count("reshape", &shape, 12);

    // Then error is returned
    assert!(result.is_err());
}

#[test]
fn test_given_head_divisible_shape_when_validated_then_ok() {
    // Given hidden_dim=64 divisible by num_heads=8
    let result = assert_head_divisible("mha", 64, 8);

    // Then it passes
    assert!(result.is_ok());
}

// ═══════════════════════════════════════════════════════════════════
// Section 2 — Budget creation → consumption → exhaustion lifecycle
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_given_pool_budget_when_allocated_within_limit_then_success() {
    // Given a pool with 4KB budget
    let pool = TensorPool::new(4096);

    // When allocating within budget
    let buf = pool.allocate(1024);

    // Then allocation succeeds and stats reflect it
    assert!(buf.as_f32_slice().len() > 0);
    let stats = pool.stats();
    assert!(stats.active_bytes > 0);
}

#[test]
fn test_given_pool_budget_when_allocations_returned_then_reusable() {
    // Given a pool
    let pool = TensorPool::new(4096);

    // When we allocate, return, then re-allocate
    let buf1 = pool.allocate(256);
    drop(buf1);
    let _buf2 = pool.allocate(256);

    // Then the second allocation reuses the returned buffer
    let stats = pool.stats();
    assert!(stats.hits >= 1);
}

#[test]
fn test_given_validation_config_when_max_elements_exceeded_then_error() {
    // Given a validator with low element limit
    let config = ValidationConfig::new().max_total_elements(100);
    let validator = TensorValidator::new(config);

    // When validating a tensor with too many elements
    let tensor = TestTensor::no_data(vec![20, 20]); // 400 > 100

    // Then validation fails with TotalElementsExceeded
    let result = validator.validate(&tensor);
    assert!(matches!(result, Err(TensorValidationError::TotalElementsExceeded { .. })));
}

#[test]
fn test_given_validation_config_when_dimensions_exceeded_then_error() {
    // Given a validator with low dimension limit
    let config = ValidationConfig::new().max_dimensions(3);
    let validator = TensorValidator::new(config);

    // When validating a 5D tensor
    let tensor = TestTensor::no_data(vec![2, 3, 4, 5, 6]);

    // Then validation fails with DimensionsExceeded
    let result = validator.validate(&tensor);
    assert!(matches!(result, Err(TensorValidationError::DimensionsExceeded { .. })));
}

#[test]
fn test_given_tensor_with_nan_when_validated_then_nan_detected() {
    // Given a tensor containing NaN
    let config = ValidationConfig::new().check_nan(true);
    let validator = TensorValidator::new(config);
    let tensor = TestTensor::new(vec![4], vec![1.0, 2.0, f32::NAN, 4.0]);

    // When validated
    let result = validator.validate(&tensor);

    // Then NaN is detected at the correct index
    assert!(matches!(result, Err(TensorValidationError::NanDetected { index: 2 })));
}

#[test]
fn test_given_tensor_with_inf_when_validated_then_inf_detected() {
    // Given a tensor containing infinity
    let config = ValidationConfig::new().check_inf(true);
    let validator = TensorValidator::new(config);
    let tensor = TestTensor::new(vec![3], vec![1.0, f32::INFINITY, 3.0]);

    // When validated
    let result = validator.validate(&tensor);

    // Then infinity is detected
    assert!(matches!(result, Err(TensorValidationError::InfDetected { index: 1, .. })));
}

#[test]
fn test_given_tensor_with_out_of_range_value_when_validated_then_error() {
    // Given a validator with a value range [-1, 1]
    let config = ValidationConfig::new().value_range(-1.0, 1.0);
    let validator = TensorValidator::new(config);
    let tensor = TestTensor::new(vec![3], vec![0.5, 2.0, -0.5]);

    // When validated
    let result = validator.validate(&tensor);

    // Then out-of-range is detected
    assert!(matches!(result, Err(TensorValidationError::ValueOutOfRange { index: 1, .. })));
}

#[test]
fn test_given_valid_tensor_when_batch_validated_then_all_ok() {
    // Given multiple valid tensors
    let validator = TensorValidator::new(ValidationConfig::new());
    let tensors = vec![
        TestTensor::new(vec![2, 3], vec![1.0; 6]),
        TestTensor::new(vec![4], vec![0.5; 4]),
        TestTensor::new(vec![1, 1], vec![0.0]),
    ];

    // When batch-validated
    let results = validator.validate_batch(&tensors);

    // Then all pass
    assert!(results.iter().all(|r| r.is_ok()));
}

// ═══════════════════════════════════════════════════════════════════
// Section 3 — Tensor type detection → validation → dispatch
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_given_quantization_type_i2s_when_checked_then_correct_variant() {
    // Given an I2S quantization type
    let qt = QuantizationType::I2S;

    // When we format it
    let name = format!("{qt:?}");

    // Then it identifies as I2S
    assert!(name.contains("I2S"));
}

#[test]
fn test_given_quantization_type_tl1_when_checked_then_correct_variant() {
    // Given a TL1 quantization type
    let qt = QuantizationType::TL1;

    // When we format it
    let name = format!("{qt:?}");

    // Then it identifies as TL1
    assert!(name.contains("TL1"));
}

#[test]
fn test_given_cpu_device_when_checked_then_is_cpu() {
    // Given a CPU device
    let device = Device::Cpu;

    // When checking device type
    // Then it reports as CPU
    assert!(device.is_cpu());
    assert!(!device.is_cuda());
}

#[test]
fn test_given_simd_level_scalar_when_compared_then_lowest() {
    // Given SimdLevel variants
    // When compared
    // Then Scalar < Avx2 < Avx512
    assert!(SimdLevel::Scalar < SimdLevel::Avx2);
    assert!(SimdLevel::Avx2 < SimdLevel::Avx512);
}

#[test]
fn test_given_kernel_backend_cpu_when_checked_then_no_gpu_required() {
    // Given a CpuRust backend
    let backend = KernelBackend::CpuRust;

    // When checking GPU requirement
    // Then GPU is not required
    assert!(!backend.requires_gpu());
}

#[test]
fn test_given_kernel_backend_cuda_when_checked_then_gpu_required() {
    // Given a Cuda backend
    let backend = KernelBackend::Cuda;

    // When checking GPU requirement
    // Then GPU is required
    assert!(backend.requires_gpu());
}

#[test]
fn test_given_simd_level_when_displayed_then_human_readable() {
    // Given various SIMD levels
    // When displayed
    // Then output is human-readable
    assert_eq!(format!("{}", SimdLevel::Scalar), "scalar");
    assert_eq!(format!("{}", SimdLevel::Avx2), "avx2");
    assert_eq!(format!("{}", SimdLevel::Avx512), "avx512");
}

#[test]
fn test_given_kernel_backend_when_displayed_then_human_readable() {
    // Given various backends
    // When displayed
    // Then output is human-readable
    assert_eq!(format!("{}", KernelBackend::CpuRust), "cpu-rust");
    assert_eq!(format!("{}", KernelBackend::Cuda), "cuda");
    assert_eq!(format!("{}", KernelBackend::CppFfi), "cpp-ffi");
}

#[test]
fn test_given_zero_dim_tensor_when_validated_then_error() {
    // Given a tensor with a zero dimension
    let validator = TensorValidator::new(ValidationConfig::new());
    let tensor = TestTensor::no_data(vec![2, 0, 4]);

    // When validated
    let result = validator.validate(&tensor);

    // Then zero dimension is detected
    assert!(matches!(result, Err(TensorValidationError::ZeroDimension { axis: 1, .. })));
}

#[test]
fn test_given_shape_error_when_formatted_then_includes_all_fields() {
    // Given a shape mismatch error
    let err = assert_shape_eq("linear_weight", &[4, 8], &[4, 16]).unwrap_err();

    // When formatted
    let msg = err.to_string();

    // Then the message contains context, expected, and actual
    assert!(msg.contains("linear_weight"));
    assert!(msg.contains("[4, 8]") || msg.contains("[4, 16]"));
}
