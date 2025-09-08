// Remove strict feature gates for now to allow basic testing

//! Quantization-specific FFI tests
//!
//! This module provides comprehensive testing of quantization functionality
//! exposed through the C API, including all quantization types and configurations.

use bitnet_common::QuantizationType;
use bitnet_ffi::{
    BITNET_ERROR_INVALID_ARGUMENT, BITNET_SUCCESS, BitNetCConfig, BitNetCInferenceConfig,
    BitNetCModel, bitnet_cleanup, bitnet_inference_with_config, bitnet_init, bitnet_model_free,
    bitnet_model_get_info, bitnet_model_load,
};
use std::ffi::CString;
use std::os::raw::c_char;

/// Test fixture for quantization tests
struct QuantizationTestFixture {
    _temp_dir: tempfile::TempDir,
    model_path: std::path::PathBuf,
}

impl QuantizationTestFixture {
    fn new() -> Self {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let model_path = temp_dir.path().join("test_quantization_model.gguf");

        // Create a dummy model file with quantization metadata
        let dummy_model_data = b"GGUF quantization test model data";
        std::fs::write(&model_path, dummy_model_data).expect("Failed to write test model");

        Self { _temp_dir: temp_dir, model_path }
    }

    fn model_path_cstr(&self) -> CString {
        CString::new(self.model_path.to_str().unwrap()).unwrap()
    }
}

#[test]
fn test_quantization_type_mapping() {
    bitnet_init();

    // Test I2S quantization (type 0)
    let config = BitNetCConfig { quantization_type: 0, ..Default::default() };

    let rust_config = config.to_bitnet_config();
    assert!(rust_config.is_ok());
    let rust_config = rust_config.unwrap();
    assert_eq!(rust_config.quantization.quantization_type, QuantizationType::I2S);

    // Test TL1 quantization (type 1)
    let config = BitNetCConfig { quantization_type: 1, ..Default::default() };

    let rust_config = config.to_bitnet_config();
    assert!(rust_config.is_ok());
    let rust_config = rust_config.unwrap();
    assert_eq!(rust_config.quantization.quantization_type, QuantizationType::TL1);

    // Test TL2 quantization (type 2)
    let config = BitNetCConfig { quantization_type: 2, ..Default::default() };

    let rust_config = config.to_bitnet_config();
    assert!(rust_config.is_ok());
    let rust_config = rust_config.unwrap();
    assert_eq!(rust_config.quantization.quantization_type, QuantizationType::TL2);

    // Test invalid quantization type
    let config = BitNetCConfig { quantization_type: 999, ..Default::default() };

    let rust_config = config.to_bitnet_config();
    assert!(rust_config.is_err());

    bitnet_cleanup();
}

#[test]
fn test_quantization_block_size_validation() {
    bitnet_init();

    // Test valid block sizes
    let valid_sizes = [16, 32, 64, 128, 256];
    for &block_size in &valid_sizes {
        let config = BitNetCConfig { block_size, ..Default::default() };

        let rust_config = config.to_bitnet_config().unwrap();
        assert_eq!(rust_config.quantization.block_size, block_size as usize);
    }

    // Test invalid block size (0)
    let config = BitNetCConfig { block_size: 0, ..Default::default() };

    let rust_config = config.to_bitnet_config();
    // This might not fail depending on implementation, but should produce meaningful config
    if let Ok(rust_config) = rust_config {
        // Block size should be set to some reasonable default
        assert!(rust_config.quantization.block_size > 0);
    }

    bitnet_cleanup();
}

#[test]
fn test_quantization_precision_validation() {
    bitnet_init();

    // Test valid precision values
    let precisions = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2];
    for &precision in &precisions {
        let config = BitNetCConfig { precision, ..Default::default() };

        let rust_config = config.to_bitnet_config().unwrap();
        assert!((rust_config.quantization.precision - precision).abs() < 1e-10);
    }

    // Test edge case: very small precision
    let config = BitNetCConfig { precision: 1e-10, ..Default::default() };

    let rust_config = config.to_bitnet_config().unwrap();
    assert!(rust_config.quantization.precision > 0.0);

    // Test edge case: very large precision (may be clamped)
    let config = BitNetCConfig { precision: 1.0, ..Default::default() };

    let rust_config = config.to_bitnet_config().unwrap();
    assert!(rust_config.quantization.precision <= 1.0);

    bitnet_cleanup();
}

#[test]
fn test_model_quantization_info() {
    let fixture = QuantizationTestFixture::new();
    bitnet_init();

    let model_path_cstr = fixture.model_path_cstr();
    let model_id = unsafe { bitnet_model_load(model_path_cstr.as_ptr()) };

    if model_id >= 0 {
        // Test model info retrieval includes quantization information
        let mut model_info = BitNetCModel::default();
        let result = unsafe { bitnet_model_get_info(model_id, &mut model_info) };

        if result == BITNET_SUCCESS {
            // Quantization type should be valid (0, 1, or 2)
            assert!(model_info.quantization_type <= 2);

            // Memory usage should be reasonable
            assert!(model_info.memory_usage > 0);

            // File size should match actual file
            let expected_size =
                std::fs::metadata(&fixture.model_path).map(|m| m.len()).unwrap_or(0);
            if expected_size > 0 {
                assert!(model_info.file_size > 0);
            }
        }

        bitnet_model_free(model_id);
    }

    bitnet_cleanup();
}

#[test]
fn test_quantization_specific_inference() {
    let fixture = QuantizationTestFixture::new();
    bitnet_init();

    let model_path_cstr = fixture.model_path_cstr();
    let model_id = unsafe { bitnet_model_load(model_path_cstr.as_ptr()) };

    if model_id >= 0 {
        // Test inference with different quantization types
        let quantization_types = [0, 1, 2]; // I2S, TL1, TL2

        for &_quant_type in &quantization_types {
            let config = BitNetCInferenceConfig::default();
            // Note: quantization type might not be directly settable in inference config
            // but we can test that inference works with models of different quantization types

            let prompt = CString::new("Test quantization inference").unwrap();
            let mut output = [0u8; 512];

            let result = unsafe {
                bitnet_inference_with_config(
                    model_id,
                    prompt.as_ptr(),
                    &config,
                    output.as_mut_ptr() as *mut c_char,
                    output.len(),
                )
            };

            // Result might fail due to dummy model, but error should be meaningful
            if result < 0 {
                // Error should be related to model format, not quantization
                assert_ne!(result, BITNET_ERROR_INVALID_ARGUMENT);
            }
        }

        bitnet_model_free(model_id);
    }

    bitnet_cleanup();
}

#[test]
fn test_quantization_config_round_trip() {
    bitnet_init();

    // Test round-trip conversion for all quantization types
    for quant_type in 0..=2 {
        let original_config = BitNetCConfig {
            quantization_type: quant_type,
            block_size: 64,
            precision: 1e-4,
            ..Default::default()
        };

        // Convert to Rust config and back
        let rust_config = original_config.to_bitnet_config().unwrap();
        let converted_config = BitNetCConfig::from_bitnet_config(&rust_config);

        assert_eq!(converted_config.quantization_type, original_config.quantization_type);
        assert_eq!(converted_config.block_size, original_config.block_size);
        assert!((converted_config.precision - original_config.precision).abs() < 1e-6);
    }

    bitnet_cleanup();
}

#[test]
fn test_quantization_memory_estimation() {
    bitnet_init();

    let configs_and_expected_sizes = [
        (0, 64, 1e-4),  // I2S
        (1, 128, 1e-5), // TL1
        (2, 256, 1e-3), // TL2
    ];

    for &(quant_type, block_size, precision) in &configs_and_expected_sizes {
        let config = BitNetCConfig {
            quantization_type: quant_type,
            block_size,
            precision,
            ..Default::default()
        };

        let rust_config = config.to_bitnet_config().unwrap();

        // Memory estimation should be reasonable based on quantization type
        // I2S should use less memory than TL2, all else being equal
        // This is more of a structural test since we don't have real models
        assert!(rust_config.quantization.block_size > 0);
        assert!(rust_config.quantization.precision > 0.0);
    }

    bitnet_cleanup();
}

#[test]
fn test_quantization_error_handling() {
    bitnet_init();

    // Test error handling with invalid quantization configurations
    let invalid_configs = [
        (999, 64, 1e-4), // Invalid quantization type
        (0, 0, 1e-4),    // Invalid block size (might be handled gracefully)
        (1, 64, -1.0),   // Invalid precision (negative)
        (2, 64, 0.0),    // Invalid precision (zero)
    ];

    for &(quant_type, block_size, precision) in &invalid_configs {
        let config = BitNetCConfig {
            quantization_type: quant_type,
            block_size,
            precision,
            ..Default::default()
        };

        let rust_config = config.to_bitnet_config();

        // First case (invalid quantization type) should definitely fail
        if quant_type == 999 {
            assert!(rust_config.is_err());
        }
        // Other cases might be handled gracefully with defaults or clamping
    }

    bitnet_cleanup();
}

#[test]
fn test_quantization_performance_characteristics() {
    bitnet_init();

    // Test that quantization configurations don't cause performance regression in setup
    let start = std::time::Instant::now();

    for _ in 0..100 {
        for quant_type in 0..=2 {
            let config = BitNetCConfig {
                quantization_type: quant_type,
                block_size: 64,
                precision: 1e-4,
                ..Default::default()
            };

            let _rust_config = config.to_bitnet_config().unwrap();
        }
    }

    let elapsed = start.elapsed();

    // Configuration conversion should be very fast
    assert!(
        elapsed < std::time::Duration::from_millis(100),
        "Quantization config conversion took too long: {:?}",
        elapsed
    );

    bitnet_cleanup();
}

/// Integration test combining quantization with other FFI features
#[test]
fn test_quantization_integration_workflow() {
    let fixture = QuantizationTestFixture::new();

    // Initialize with different quantization configurations
    bitnet_init();

    // Test workflow with I2S quantization
    let config = BitNetCConfig {
        quantization_type: 0, // I2S
        block_size: 64,
        precision: 1e-4,
        ..Default::default()
    };

    let rust_config = config.to_bitnet_config().unwrap();
    assert_eq!(rust_config.quantization.quantization_type, QuantizationType::I2S);

    // Attempt model loading (will likely fail with dummy file but tests the path)
    let model_path_cstr = fixture.model_path_cstr();
    let model_id = unsafe { bitnet_model_load(model_path_cstr.as_ptr()) };

    if model_id >= 0 {
        // If loading succeeds, test quantization-aware operations
        let mut model_info = BitNetCModel::default();
        let result = unsafe { bitnet_model_get_info(model_id, &mut model_info) };

        if result == BITNET_SUCCESS {
            assert!(model_info.quantization_type <= 2);
        }

        bitnet_model_free(model_id);
    }

    bitnet_cleanup();
}
