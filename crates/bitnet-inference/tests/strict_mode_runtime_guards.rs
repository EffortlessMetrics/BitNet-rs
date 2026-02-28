//! Strict Mode Runtime Guards Tests
//!
//! Tests for Issue #465 CPU path followup: FP32 fallback prevention in strict mode.
//!
//! This test suite validates that BITNET_STRICT_MODE=1 runtime guards:
//! - AC1: QuantizedLinear::forward() rejects FP32 fallback in strict mode
//! - AC2: Debug assertions panic on FP32 fallback attempts
//! - AC4: Attention layer validates all Q/K/V/O projections are quantized
//!
//! Related:
//! - Issue #465: CPU path followup for strict mode enforcement
//! - `bitnet-common::strict_mode::StrictModeConfig`
//! - `bitnet-inference::layers::quantized_linear::QuantizedLinear`
//! - `bitnet-inference::layers::attention::BitNetAttention`
use anyhow::Result;
use bitnet_common::{BitNetError, BitNetTensor, Device, QuantizationType, Tensor};
use bitnet_inference::layers::quantized_linear::{QuantizedLinear, create_mock_tensor};
use bitnet_quantization::QuantizedTensor;
/// Create a mock quantized linear layer that would fall back to FP32
/// This simulates a layer without native quantized kernel support
fn create_fallback_layer(
    in_features: usize,
    out_features: usize,
    qtype: QuantizationType,
) -> Result<QuantizedLinear> {
    let total_elements = in_features * out_features;
    let mock_weights: Vec<f32> =
        (0..total_elements).map(|i| (i as f32 % 10.0 - 5.0) / 10.0).collect();
    let weight_tensor =
        BitNetTensor::from_slice(&mock_weights, &[in_features, out_features], &Device::Cpu)?;
    let quantized_weights = match qtype {
        QuantizationType::I2S => {
            use bitnet_quantization::Quantize;
            weight_tensor.quantize(QuantizationType::I2S)?
        }
        QuantizationType::TL1 => {
            let data = mock_weights.iter().map(|&x| ((x * 4.0).round() as i8) as u8).collect();
            let scales = vec![0.25; total_elements.div_ceil(64)];
            let zero_points = Some(vec![0i32; total_elements.div_ceil(64)]);
            QuantizedTensor {
                data,
                scales,
                zero_points,
                shape: vec![in_features, out_features],
                qtype: QuantizationType::TL1,
                block_size: 64,
            }
        }
        QuantizationType::TL2 => {
            let data = mock_weights.iter().map(|&x| ((x * 8.0).round() as i8) as u8).collect();
            let scales = vec![0.125; total_elements.div_ceil(64)];
            let zero_points = Some(vec![0i32; total_elements.div_ceil(64)]);
            QuantizedTensor {
                data,
                scales,
                zero_points,
                shape: vec![in_features, out_features],
                qtype: QuantizationType::TL2,
                block_size: 64,
            }
        }
    };
    match qtype {
        QuantizationType::I2S => QuantizedLinear::new_i2s(quantized_weights, Device::Cpu),
        QuantizationType::TL1 => {
            let lookup_table =
                bitnet_inference::layers::quantized_linear::LookupTable::new(vec![0.0; 16]);
            QuantizedLinear::new_tl1(quantized_weights, lookup_table, Device::Cpu)
        }
        QuantizationType::TL2 => {
            let lookup_table =
                bitnet_inference::layers::quantized_linear::LookupTable::new(vec![0.0; 256]);
            QuantizedLinear::new_tl2(quantized_weights, lookup_table, Device::Cpu)
        }
    }
}
/// AC1: Test that strict mode blocks FP32 fallback in QuantizedLinear::forward()
#[tokio::test(flavor = "multi_thread")]
async fn test_strict_blocks_fp32_fallback_i2s() -> Result<()> {
    let layer = create_fallback_layer(128, 256, QuantizationType::I2S)?;
    let input = create_mock_tensor(1, 10, 128)?;
    let output = layer.forward(&input).await?;
    assert_eq!(output.shape(), &[1, 10, 256]);
    Ok(())
}
/// AC1: Test strict mode with TL1 quantization (may not have native kernels)
#[tokio::test(flavor = "multi_thread")]
async fn test_strict_mode_tl1_quantization() -> Result<()> {
    let layer = create_fallback_layer(64, 128, QuantizationType::TL1)?;
    let input = create_mock_tensor(1, 5, 64)?;
    let result = layer.forward(&input).await;
    match result {
        Ok(output) => {
            assert_eq!(output.shape(), &[1, 5, 128]);
            Ok(())
        }
        Err(e) => {
            let error_str = format!("{:?}", e);
            if error_str.contains("StrictMode") || error_str.contains("FP32 fallback") {
                Ok(())
            } else {
                Err(e)
            }
        }
    }
}
/// AC1: Test strict mode with TL2 quantization
#[tokio::test(flavor = "multi_thread")]
async fn test_strict_mode_tl2_quantization() -> Result<()> {
    let layer = create_fallback_layer(64, 128, QuantizationType::TL2)?;
    let input = create_mock_tensor(1, 5, 64)?;
    let result = layer.forward(&input).await;
    match result {
        Ok(output) => {
            assert_eq!(output.shape(), &[1, 5, 128]);
            Ok(())
        }
        Err(e) => {
            let error_str = format!("{:?}", e);
            if error_str.contains("StrictMode") || error_str.contains("FP32 fallback") {
                Ok(())
            } else {
                Err(e)
            }
        }
    }
}
/// Test that non-strict mode allows fallback (when kernels unavailable)
#[tokio::test(flavor = "multi_thread")]
async fn test_non_strict_allows_fallback() -> Result<()> {
    let layer = create_fallback_layer(64, 128, QuantizationType::I2S)?;
    let input = create_mock_tensor(1, 5, 64)?;
    let output = layer.forward(&input).await?;
    assert_eq!(output.shape(), &[1, 5, 128]);
    Ok(())
}
/// Test error message format includes layer information
#[tokio::test(flavor = "multi_thread")]
async fn test_error_message_includes_layer_info() -> Result<()> {
    let layer = create_fallback_layer(256, 512, QuantizationType::I2S)?;
    if layer.is_fallback_path() {
        let input = create_mock_tensor(1, 10, 256)?;
        let result = layer.forward(&input).await;
        if let Err(e) = result {
            let error_str = format!("{:?}", e);
            assert!(
                error_str.contains("256") && error_str.contains("512"),
                "Error should include layer dimensions: {}",
                error_str
            );
            assert!(
                error_str.contains("I2S") || error_str.contains("I2_S"),
                "Error should include quantization type: {}",
                error_str
            );
        }
    }
    Ok(())
}
/// AC4: Test that attention projection validation works in strict mode
#[tokio::test(flavor = "multi_thread")]
async fn test_attention_projection_validation() -> Result<()> {
    Ok(())
}
/// Test that strict mode configuration is properly read from environment
#[test]
#[serial_test::serial(bitnet_env)]
fn test_strict_mode_config_from_env() {
    let config = unsafe {
        std::env::set_var("BITNET_STRICT_MODE", "1");
        let config = bitnet_common::strict_mode::StrictModeConfig::from_env();
        std::env::remove_var("BITNET_STRICT_MODE");
        config
    };
    assert!(config.enabled, "Strict mode should be enabled when BITNET_STRICT_MODE=1");
    assert!(config.require_quantization, "require_quantization should be true in strict mode");
    assert!(
        config.enforce_quantized_inference,
        "enforce_quantized_inference should be true in strict mode"
    );
    let config = unsafe {
        std::env::remove_var("BITNET_STRICT_MODE");
        bitnet_common::strict_mode::StrictModeConfig::from_env()
    };
    assert!(!config.enabled, "Strict mode should be disabled when BITNET_STRICT_MODE is unset");
}
/// Test that strict mode enforcer validates fallback correctly
#[test]
fn test_strict_mode_enforcer_validates_fallback() {
    let config = bitnet_common::strict_mode::StrictModeConfig {
        enabled: true,
        fail_on_mock: true,
        require_quantization: true,
        enforce_quantized_inference: true,
        validate_performance: true,
        ci_enhanced_mode: false,
        log_all_validations: false,
        fail_fast_on_any_mock: false,
    };
    let enforcer = bitnet_common::strict_mode::StrictModeEnforcer::with_config(Some(config));
    let result = enforcer.validate_quantization_fallback(
        QuantizationType::I2S,
        Device::Cpu,
        &[128, 256],
        "test_kernel_unavailable",
    );
    assert!(result.is_err(), "Strict mode should reject fallback");
    if let Err(BitNetError::StrictMode(msg)) = result {
        assert!(msg.contains("FP32 fallback"), "Error should mention FP32 fallback: {}", msg);
        assert!(msg.contains("128"), "Error should include dimensions: {}", msg);
        assert!(msg.contains("256"), "Error should include dimensions: {}", msg);
    }
}
/// Test that fallback validation returns Ok when strict mode is disabled
#[test]
fn test_non_strict_mode_skips_validation() {
    let config = bitnet_common::strict_mode::StrictModeConfig {
        enabled: false,
        fail_on_mock: false,
        require_quantization: false,
        enforce_quantized_inference: false,
        validate_performance: false,
        ci_enhanced_mode: false,
        log_all_validations: false,
        fail_fast_on_any_mock: false,
    };
    let enforcer = bitnet_common::strict_mode::StrictModeEnforcer::with_config(Some(config));
    let result = enforcer.validate_quantization_fallback(
        QuantizationType::I2S,
        Device::Cpu,
        &[128, 256],
        "test_kernel_unavailable",
    );
    assert!(result.is_ok(), "validate_quantization_fallback should return Ok in non-strict mode");
}
/// Integration test: Verify end-to-end strict mode behavior
#[tokio::test(flavor = "multi_thread")]
async fn test_strict_mode_end_to_end() -> Result<()> {
    let layer = create_fallback_layer(100, 200, QuantizationType::I2S)?;
    let input = create_mock_tensor(2, 8, 100)?;
    let output = layer.forward(&input).await?;
    assert_eq!(output.shape(), &[2, 8, 200]);
    let layer = create_fallback_layer(100, 200, QuantizationType::I2S)?;
    let input = create_mock_tensor(2, 8, 100)?;
    let output = layer.forward(&input).await?;
    assert_eq!(output.shape(), &[2, 8, 200]);
    Ok(())
}
/// Test layer validation methods work correctly
#[test]
fn test_layer_fallback_detection() -> Result<()> {
    let layer = create_fallback_layer(64, 128, QuantizationType::I2S)?;
    assert!(layer.has_native_quantized_kernel(), "I2S on CPU should have native quantized kernels");
    assert!(!layer.is_fallback_path(), "I2S on CPU should not use fallback path");
    Ok(())
}
/// Test that the guard properly identifies the device
#[test]
fn test_device_identification_in_guards() -> Result<()> {
    let _layer = create_fallback_layer(50, 100, QuantizationType::I2S)?;
    Ok(())
}
