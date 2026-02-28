//! Tests for the mock GPU test harness and numerical validator.

use bitnet_opencl::reference_kernels;
use bitnet_opencl::test_fixtures;
use bitnet_opencl::testing::{BufferHandle, MockGpuContext, NumericalValidator};

// ── MockGpuContext tests ────────────────────────────────────────────

#[test]
fn test_mock_context_creation() {
    let ctx = MockGpuContext::new("Mock Intel Arc A770", 16 * 1024 * 1024 * 1024);
    assert_eq!(ctx.device_name(), "Mock Intel Arc A770");
    assert_eq!(ctx.total_memory(), 16 * 1024 * 1024 * 1024);
    assert_eq!(ctx.buffer_count(), 0);
}

#[test]
fn test_mock_buffer_alloc_and_count() {
    let mut ctx = MockGpuContext::new("test", 1024);
    let h1 = ctx.alloc_buffer(64);
    let h2 = ctx.alloc_buffer(128);
    assert_eq!(h1, BufferHandle(0));
    assert_eq!(h2, BufferHandle(1));
    assert_eq!(ctx.buffer_count(), 2);
}

#[test]
fn test_mock_buffer_read_write() {
    let mut ctx = MockGpuContext::new("test", 1024);
    let h = ctx.alloc_buffer(16);

    let data = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    ctx.write_buffer(h, &data).unwrap();

    let read_back = ctx.read_buffer(h).unwrap();
    assert_eq!(read_back, &data);
}

#[test]
fn test_mock_buffer_write_overflow() {
    let mut ctx = MockGpuContext::new("test", 1024);
    let h = ctx.alloc_buffer(4);
    let result = ctx.write_buffer(h, &[0u8; 8]);
    assert!(result.is_err());
}

#[test]
fn test_mock_buffer_invalid_handle() {
    let ctx = MockGpuContext::new("test", 1024);
    let result = ctx.read_buffer(BufferHandle(99));
    assert!(result.is_err());
}

#[test]
fn test_mock_kernel_execution() {
    let mut ctx = MockGpuContext::new("test", 4096);

    // Register a simple "double" kernel
    ctx.register_kernel("double", |inputs, output| {
        let input = inputs[0];
        if output.len() < input.len() {
            return Err("output too small".to_string());
        }
        for (i, chunk) in input.chunks_exact(4).enumerate() {
            let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let doubled = val * 2.0;
            output[i * 4..(i + 1) * 4].copy_from_slice(&doubled.to_le_bytes());
        }
        Ok(())
    });

    // Set up buffers
    let input_h = ctx.alloc_buffer(16); // 4 floats
    let output_h = ctx.alloc_buffer(16);

    let input_data: Vec<u8> =
        [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|f| f.to_le_bytes()).collect();
    ctx.write_buffer(input_h, &input_data).unwrap();

    ctx.execute_kernel("double", &[input_h], output_h).unwrap();

    let result_bytes = ctx.read_buffer(output_h).unwrap();
    let results: Vec<f32> = result_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(results, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_mock_kernel_not_found() {
    let mut ctx = MockGpuContext::new("test", 1024);
    let h = ctx.alloc_buffer(16);
    let result = ctx.execute_kernel("nonexistent", &[h], h);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not registered"));
}

// ── NumericalValidator tests ────────────────────────────────────────

#[test]
fn test_validator_strict_identical() {
    let v = NumericalValidator::strict();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = v.compare_f32(&data, &data);
    assert!(result.passed);
    assert_eq!(result.num_mismatches, 0);
    assert_eq!(result.max_abs_diff, 0.0);
}

#[test]
fn test_validator_strict_tiny_diff() {
    let v = NumericalValidator::strict();
    let expected = vec![1.0, 2.0, 3.0];
    let actual = vec![1.0 + 1e-7, 2.0 - 1e-7, 3.0 + 1e-7];
    let result = v.compare_f32(&expected, &actual);
    assert!(result.passed);
}

#[test]
fn test_validator_strict_large_diff_fails() {
    let v = NumericalValidator::strict();
    let expected = vec![1.0, 2.0, 3.0];
    let actual = vec![1.0, 2.0, 4.0]; // 3.0 vs 4.0
    let result = v.compare_f32(&expected, &actual);
    assert!(!result.passed);
    assert!(result.num_mismatches > 0);
}

#[test]
fn test_validator_relaxed_moderate_diff() {
    let v = NumericalValidator::relaxed();
    let expected = vec![1.0, 2.0, 3.0];
    let actual = vec![1.0005, 2.001, 3.0005];
    let result = v.compare_f32(&expected, &actual);
    assert!(result.passed);
}

#[test]
fn test_validator_f16_comparison() {
    let v = NumericalValidator::relaxed();
    // f16 1.0 = 0x3C00
    let expected = vec![0x3C00u16, 0x4000]; // 1.0, 2.0
    let actual = vec![0x3C00u16, 0x4000];
    let result = v.compare_f16(&expected, &actual);
    assert!(result.passed);
}

#[test]
fn test_validator_worst_index() {
    let v = NumericalValidator::strict();
    let expected = vec![1.0, 2.0, 3.0, 4.0];
    let actual = vec![1.0, 2.0, 3.5, 4.0]; // index 2 worst
    let result = v.compare_f32(&expected, &actual);
    assert_eq!(result.worst_index, 2);
}

// ── Golden fixture validation ────────────────────────────────────────

#[test]
fn test_golden_matmul_matches_reference() {
    let (a, b, expected) = test_fixtures::golden_matmul_4x4();
    let mut actual = vec![0.0f32; 16];
    reference_kernels::ref_matmul(&a, &b, &mut actual, 4, 4, 4);

    let v = NumericalValidator::strict();
    let result = v.compare_f32(&expected, &actual);
    assert!(
        result.passed,
        "golden matmul mismatch: max_abs={} at index {}",
        result.max_abs_diff, result.worst_index
    );
}

#[test]
fn test_golden_softmax_matches_reference() {
    let (input, expected) = test_fixtures::golden_softmax_8();
    let mut actual = vec![0.0f32; 8];
    reference_kernels::ref_softmax(&input, &mut actual, 8);

    let v = NumericalValidator::strict();
    let result = v.compare_f32(&expected, &actual);
    assert!(
        result.passed,
        "golden softmax mismatch: max_abs={} at index {}",
        result.max_abs_diff, result.worst_index
    );
}

#[test]
fn test_golden_rmsnorm_matches_reference() {
    let (input, weight, expected) = test_fixtures::golden_rmsnorm_4();
    let mut actual = vec![0.0f32; 4];
    reference_kernels::ref_rmsnorm(&input, &weight, &mut actual, 1e-6);

    let v = NumericalValidator::strict();
    let result = v.compare_f32(&expected, &actual);
    assert!(
        result.passed,
        "golden rmsnorm mismatch: max_abs={} at index {}",
        result.max_abs_diff, result.worst_index
    );
}

#[test]
fn test_golden_silu_matches_reference() {
    let (input, expected) = test_fixtures::golden_silu_values();
    let mut actual = vec![0.0f32; input.len()];
    reference_kernels::ref_silu(&input, &mut actual);

    let v = NumericalValidator::strict();
    let result = v.compare_f32(&expected, &actual);
    assert!(result.passed);
}

#[test]
fn test_golden_gelu_matches_reference() {
    let (input, expected) = test_fixtures::golden_gelu_values();
    let mut actual = vec![0.0f32; input.len()];
    reference_kernels::ref_gelu(&input, &mut actual);

    let v = NumericalValidator::strict();
    let result = v.compare_f32(&expected, &actual);
    assert!(result.passed);
}
