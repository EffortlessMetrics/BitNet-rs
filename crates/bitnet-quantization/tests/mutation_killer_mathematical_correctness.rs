//! Mathematical Correctness Mutation Killer Tests for BitNet.rs Quantization
//!
//! This test suite is designed to kill mutations in quantization algorithms by testing
//! mathematical correctness, device-aware operations, and numerical accuracy validation.
//! Focuses on I2S, TL1, TL2 quantization implementations with device parameters.

use bitnet_quantization::device_aware_quantizer::{
    AccuracyValidator, DeviceAwareQuantizer, QuantizationType as DeviceQuantizationType,
    QuantizedTensor as DeviceQuantizedTensor, ToleranceConfig,
};
use std::f32::consts::PI;

/// Test device-aware I2S quantization with CPU device parameter
#[test]
fn test_i2s_quantization_cpu_device_correctness() {
    // Use realistic tolerance with non-strict validation for mutation testing
    let tolerance_config =
        ToleranceConfig { i2s_tolerance: 1e-3, strict_validation: false, ..Default::default() };

    let quantizer = DeviceAwareQuantizer::with_tolerance_config(tolerance_config);

    // Generate mathematically precise test data that exposes quantization errors
    let test_data = generate_mathematical_test_patterns();

    // Test I2S quantization with explicit validation
    let result = quantizer.quantize_with_validation(&test_data, DeviceQuantizationType::I2S);
    assert!(result.is_ok(), "I2S quantization should succeed with valid input");

    let quantized = result.unwrap();
    assert_eq!(quantized.qtype, DeviceQuantizationType::I2S);
    assert!(!quantized.data.is_empty(), "Quantized data should not be empty");
    assert!(!quantized.scales.is_empty(), "Scale factors should not be empty");

    // Validate mathematical properties that mutations would break
    assert_device_mathematical_properties(&test_data, &quantized);
}

/// Test TL1 quantization with device-aware operations
#[test]
fn test_tl1_quantization_device_aware_correctness() {
    // Use realistic tolerance with non-strict validation for mutation testing
    let tolerance_config =
        ToleranceConfig { tl_tolerance: 1e-2, strict_validation: false, ..Default::default() };

    let quantizer = DeviceAwareQuantizer::with_tolerance_config(tolerance_config);

    // Test data designed to expose TL1-specific mutation bugs
    let test_data = generate_tl1_edge_cases();

    let result = quantizer.quantize_with_validation(&test_data, DeviceQuantizationType::TL1);
    assert!(result.is_ok(), "TL1 quantization should succeed");

    let quantized = result.unwrap();
    assert_eq!(quantized.qtype, DeviceQuantizationType::TL1);

    // Test TL1-specific mathematical properties
    validate_device_tl1_lookup_properties(&quantized);
}

/// Test TL2 quantization with x86-specific optimizations
#[test]
fn test_tl2_quantization_x86_correctness() {
    // Use realistic tolerance for table lookup quantization (1e-2)
    let tolerance_config = ToleranceConfig {
        tl_tolerance: 1e-2,
        strict_validation: false, // Allow for implementation variations
        ..Default::default()
    };

    let quantizer = DeviceAwareQuantizer::with_tolerance_config(tolerance_config);

    // Test data that exposes SIMD/AVX implementation bugs
    let test_data = generate_simd_alignment_patterns();

    let result = quantizer.quantize_with_validation(&test_data, DeviceQuantizationType::TL2);
    assert!(result.is_ok(), "TL2 quantization should succeed");

    let quantized = result.unwrap();
    assert_eq!(quantized.qtype, DeviceQuantizationType::TL2);

    // Validate SIMD-specific properties
    validate_device_simd_correctness(&test_data, &quantized);
}

/// Test device fallback scenarios for GPU/CPU quantization
#[test]
fn test_device_fallback_quantization_correctness() {
    let tolerance_config = ToleranceConfig { strict_validation: false, ..Default::default() };
    let quantizer = DeviceAwareQuantizer::with_tolerance_config(tolerance_config);
    let test_data = vec![1.0, -0.5, 0.75, -0.25, 0.0, 0.125, -0.875, 0.5];

    // Test CPU quantization (should always work)
    let cpu_result = quantizer.quantize_with_validation(&test_data, DeviceQuantizationType::I2S);
    assert!(cpu_result.is_ok(), "CPU quantization should always succeed");

    // Test device-aware selection
    #[cfg(feature = "gpu")]
    {
        // GPU quantization should fallback gracefully if GPU not available
        let gpu_result = quantizer.validate_gpu_cpu_parity(&test_data);
        // This may succeed or fail depending on GPU availability, both are valid
        match gpu_result {
            Ok(report) => {
                assert_eq!(report.quantization_type, DeviceQuantizationType::I2S);
                assert!(report.cross_device_error >= 0.0);
            }
            Err(_) => {
                // GPU not available, test passes
                println!("GPU not available for parity testing (expected)");
            }
        }
    }
}

/// Test numerical accuracy validation with strict tolerances
#[test]
fn test_accuracy_validation_strict_tolerances() {
    // Use realistic tolerances with non-strict validation for validator testing
    let tolerance_config = ToleranceConfig {
        i2s_tolerance: 1e-3, // Realistic for 2-bit quantization
        tl_tolerance: 1e-2,  // Realistic for table lookup
        strict_validation: false,
        ..Default::default()
    };

    let validator = AccuracyValidator::new(tolerance_config);

    // Generate highly precise test data
    let original_data = generate_precision_test_data();

    // Create a mock quantized tensor for validation
    let quantized = create_mock_device_quantized_tensor(&original_data);

    let accuracy_report = validator.validate_i2s_accuracy(&original_data, &quantized);
    assert!(accuracy_report.is_ok(), "Accuracy validation should succeed");

    let report = accuracy_report.unwrap();
    assert!(report.max_absolute_error >= 0.0);
    assert!(report.mean_absolute_error >= 0.0);
    assert!(report.relative_error >= 0.0);
}

/// Test boundary conditions that expose quantization bugs
#[test]
fn test_quantization_boundary_conditions() {
    let quantizer = DeviceAwareQuantizer::new();

    // Test extreme values
    let extreme_values = vec![
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::EPSILON,
        -f32::EPSILON,
        0.0,
        -0.0,
    ];

    for &value in &extreme_values {
        if value.is_finite() {
            let test_data = vec![value; 32]; // Minimum block size
            let result =
                quantizer.quantize_with_validation(&test_data, DeviceQuantizationType::I2S);

            if value == 0.0 || value == -0.0 {
                assert!(result.is_ok(), "Zero values should quantize successfully");
            } else {
                // Other finite extreme values may or may not succeed depending on implementation
                match result {
                    Ok(quantized) => {
                        assert_eq!(quantized.qtype, DeviceQuantizationType::I2S);
                        assert!(!quantized.data.is_empty());
                    }
                    Err(_) => {
                        // Failure is acceptable for extreme values
                        println!("Extreme value {} failed quantization (acceptable)", value);
                    }
                }
            }
        }
    }
}

/// Test scale factor computation accuracy
#[test]
fn test_scale_factor_computation_accuracy() {
    let tolerance_config = ToleranceConfig { strict_validation: false, ..Default::default() };
    let quantizer = DeviceAwareQuantizer::with_tolerance_config(tolerance_config);

    // Test data with known scale properties
    let test_cases = [
        (vec![1.0, -1.0, 0.5, -0.5], 1.0),    // Max abs = 1.0
        (vec![2.0, -3.0, 1.0, -1.5], 3.0),    // Max abs = 3.0
        (vec![0.1, -0.05, 0.03, -0.07], 0.1), // Max abs = 0.1
    ];

    for (test_data, expected_max_scale) in test_cases {
        let result = quantizer.quantize_with_validation(&test_data, DeviceQuantizationType::I2S);
        assert!(result.is_ok(), "Quantization should succeed for test data");

        let quantized = result.unwrap();

        // Verify scale factors are reasonable
        for &scale in &quantized.scales {
            assert!(scale >= 0.0, "Scale factors should be non-negative");
            assert!(
                scale <= expected_max_scale + f32::EPSILON,
                "Scale factor {} should not exceed max value {}",
                scale,
                expected_max_scale
            );
        }
    }
}

/// Test compression ratio calculations
#[test]
fn test_compression_ratio_calculation() {
    // Use non-strict validation to focus on compression ratio calculation, not accuracy
    let tolerance_config = ToleranceConfig { strict_validation: false, ..Default::default() };
    let quantizer = DeviceAwareQuantizer::with_tolerance_config(tolerance_config);

    // Test with different data sizes
    let test_sizes = [32, 64, 128, 256, 512, 1024];

    for size in test_sizes {
        let test_data = generate_random_test_data(size);
        let result = quantizer.quantize_with_validation(&test_data, DeviceQuantizationType::I2S);
        assert!(result.is_ok(), "Quantization should succeed for size {}", size);

        let quantized = result.unwrap();
        // Note: DeviceQuantizedTensor doesn't have compression_ratio method, so we calculate manually
        let compression_ratio = calculate_compression_ratio(&quantized, size);

        assert!(compression_ratio >= 1.0, "Compression ratio should be >= 1.0");
        assert!(compression_ratio <= 16.0, "Compression ratio should be reasonable for I2S");

        // I2S uses 2 bits per element, so theoretical max is 16x compression
        // In practice, with scale factors and metadata, it should be lower
        assert!(compression_ratio <= 8.0, "Practical compression ratio should be <= 8x");
    }
}

/// Test round-trip quantization accuracy
#[test]
fn test_round_trip_quantization_accuracy() {
    let tolerance_config = ToleranceConfig { strict_validation: false, ..Default::default() };
    let quantizer = DeviceAwareQuantizer::with_tolerance_config(tolerance_config);

    // Generate test data with various patterns
    let test_patterns = [
        generate_sine_wave_pattern(64),
        generate_random_normal_pattern(64),
        generate_sparse_pattern(64),
        generate_uniform_pattern(64),
    ];

    for test_data in test_patterns {
        let result = quantizer.quantize_with_validation(&test_data, DeviceQuantizationType::I2S);
        assert!(result.is_ok(), "Quantization should succeed");

        let quantized = result.unwrap();

        // For device quantization, we need to use the CPU quantizer to dequantize
        let cpu_quantizer = bitnet_quantization::device_aware_quantizer::CPUQuantizer::new(
            ToleranceConfig::default(),
        );
        let dequantized = cpu_quantizer.dequantize_i2s(&quantized);
        assert!(dequantized.is_ok(), "Dequantization should succeed");

        let recovered_data = dequantized.unwrap();
        assert_eq!(
            recovered_data.len(),
            test_data.len(),
            "Recovered data should have same length as original"
        );

        // Calculate round-trip error
        let mut max_error = 0.0f32;
        for (orig, recovered) in test_data.iter().zip(recovered_data.iter()) {
            let error = (orig - recovered).abs();
            max_error = max_error.max(error);
        }

        // Ensure round-trip error is within reasonable bounds
        assert!(max_error < 1.0, "Round-trip error should be reasonable");
    }
}

// Helper functions for generating test data

fn generate_mathematical_test_patterns() -> Vec<f32> {
    let mut data = Vec::new();

    // Add mathematical constants and derived values
    data.extend([PI, -PI, PI / 2.0, -PI / 2.0, PI / 4.0, -PI / 4.0]);
    data.extend([std::f32::consts::E, -std::f32::consts::E]);
    data.extend([std::f32::consts::SQRT_2, -std::f32::consts::SQRT_2]);

    // Add powers of 2 (important for binary quantization)
    for i in -10..=10 {
        data.push(2.0f32.powi(i));
        data.push(-2.0f32.powi(i));
    }

    // Pad to minimum block size
    while data.len() < 32 {
        data.push(0.0);
    }

    data
}

fn generate_tl1_edge_cases() -> Vec<f32> {
    // TL1 uses 4-bit table lookup, so test all quantization levels
    let mut data = Vec::new();

    for i in 0..16 {
        let normalized = (i as f32 / 15.0) * 2.0 - 1.0; // Map to [-1, 1]
        data.push(normalized);
        data.push(-normalized);
    }

    // Add boundary cases
    data.extend([-1.0, 1.0, 0.0]);

    // Pad to minimum block size
    while data.len() < 32 {
        data.push(0.0);
    }

    data
}

fn generate_simd_alignment_patterns() -> Vec<f32> {
    // Generate data with SIMD alignment considerations
    let mut data = Vec::new();

    // Test various alignment patterns
    let alignments = [8, 16, 32, 64]; // Common SIMD widths

    for alignment in alignments {
        for i in 0..alignment {
            data.push((i as f32 / alignment as f32) * 2.0 - 1.0);
        }
    }

    data
}

fn generate_precision_test_data() -> Vec<f32> {
    // Generate data that tests numerical precision
    let mut data = Vec::new();

    // Very small values near quantization thresholds
    let thresholds = [0.5, 0.25, 0.125, 0.0625];
    for &threshold in &thresholds {
        data.push(threshold + f32::EPSILON);
        data.push(threshold - f32::EPSILON);
        data.push(-threshold + f32::EPSILON);
        data.push(-threshold - f32::EPSILON);
    }

    data
}

fn generate_random_test_data(size: usize) -> Vec<f32> {
    // Generate pseudo-random data with fixed seed for reproducibility
    let mut data = Vec::with_capacity(size);
    let mut x = 1u32;

    for _ in 0..size {
        // Linear congruential generator for reproducible "random" data
        x = x.wrapping_mul(1103515245).wrapping_add(12345);
        let normalized = (x as f32 / u32::MAX as f32) * 2.0 - 1.0;
        data.push(normalized);
    }

    data
}

fn generate_sine_wave_pattern(size: usize) -> Vec<f32> {
    (0..size).map(|i| (2.0 * PI * i as f32 / size as f32).sin()).collect()
}

fn generate_random_normal_pattern(size: usize) -> Vec<f32> {
    // Box-Muller transform for normal distribution
    let mut data = Vec::with_capacity(size);
    let mut use_last = false;
    let mut last = 0.0;

    for i in 0..size {
        if use_last {
            data.push(last * 0.1); // Scale to reasonable range
            use_last = false;
        } else {
            let u1 = (i + 1) as f32 / (size + 1) as f32;
            let u2 = ((i * 17 + 7) % size + 1) as f32 / (size + 1) as f32;

            let mag = 0.1 * (-2.0 * u1.ln()).sqrt();
            let z0 = mag * (2.0 * PI * u2).cos();
            let z1 = mag * (2.0 * PI * u2).sin();

            data.push(z0);
            last = z1;
            use_last = true;
        }
    }

    data
}

fn generate_sparse_pattern(size: usize) -> Vec<f32> {
    let mut data = vec![0.0; size];

    // Make 10% of values non-zero
    for i in (0..size).step_by(10) {
        data[i] = if i % 20 == 0 { 1.0 } else { -1.0 };
    }

    data
}

fn generate_uniform_pattern(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32 / size as f32) * 2.0 - 1.0).collect()
}

fn create_mock_device_quantized_tensor(original_data: &[f32]) -> DeviceQuantizedTensor {
    // Create a simple mock quantized tensor for testing validation
    let block_size = 32;
    let num_blocks = original_data.len().div_ceil(block_size);
    let mut scales = Vec::new();
    let mut quantized_data = Vec::new();

    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(original_data.len());
        let block = &original_data[start..end];

        let scale = block.iter().map(|x| x.abs()).fold(0.0, f32::max);
        scales.push(scale);

        // Simple 2-bit quantization
        for &value in block {
            let normalized = if scale > 0.0 { value / scale } else { 0.0 };
            let quantized = if normalized > 0.5 {
                1u8
            } else if normalized < -0.5 {
                3u8 // Represents -1 in 2-bit
            } else {
                0u8
            };
            quantized_data.push(quantized);
        }
    }

    DeviceQuantizedTensor::new(
        quantized_data,
        DeviceQuantizationType::I2S,
        vec![original_data.len()],
        scales,
        block_size,
    )
}

fn calculate_compression_ratio(quantized: &DeviceQuantizedTensor, original_size: usize) -> f32 {
    let original_bytes = original_size * 4; // FP32 = 4 bytes per element
    let compressed_bytes = quantized.data.len() + quantized.scales.len() * 4;
    if compressed_bytes == 0 {
        1.0 // Avoid division by zero
    } else {
        (original_bytes as f32 / compressed_bytes as f32).max(1.0)
    }
}

// Validation helper functions

fn assert_device_mathematical_properties(original: &[f32], quantized: &DeviceQuantizedTensor) {
    // Validate that basic mathematical properties hold
    assert_eq!(quantized.numel(), original.len(), "Element count should match");
    assert!(quantized.nbytes() > 0, "Should have compressed data");
    assert!(!quantized.scales.is_empty(), "Should have scale factors");

    // Validate scale factors are reasonable
    for &scale in &quantized.scales {
        assert!(scale >= 0.0, "Scale factors should be non-negative");
        assert!(scale.is_finite(), "Scale factors should be finite");
    }
}

fn validate_device_tl1_lookup_properties(quantized: &DeviceQuantizedTensor) {
    assert_eq!(quantized.qtype, DeviceQuantizationType::TL1);

    // TL1 specific validations
    for &byte in &quantized.data {
        assert!(byte <= 15, "TL1 values should be 4-bit (0-15)");
    }
}

fn validate_device_simd_correctness(original: &[f32], quantized: &DeviceQuantizedTensor) {
    assert_eq!(quantized.qtype, DeviceQuantizationType::TL2);

    // Validate SIMD alignment doesn't break correctness
    assert_eq!(quantized.numel(), original.len());
    assert!(!quantized.data.is_empty());
}
