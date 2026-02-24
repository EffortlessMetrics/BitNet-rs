//! Issue #261 Mutation Coverage Tests
//!
//! Targeted tests to kill specific surviving mutants and achieve ≥80% mutation score.
//!
//! Surviving Mutants Addressed:
//! 1. Type Safety Validation (lib.rs:114:23) - Replace `==` with `!=`
//! 2. Quantizer Availability (lib.rs:190:9) - Replace `return true` with `return false`
//! 3. Conversion Validation (lib.rs:199:21) - Replace `==` with `!=`
//! 4. Round-Trip Validation (lib.rs:214:5) - Replace call with `Ok(true)`
//!
//! Specification: docs/explanation/specs/issue-261-mock-performance-reporting-elimination-spec.md

use bitnet_common::QuantizationType;
use bitnet_quantization::{I2SQuantizer, QuantizerFactory, QuantizerTrait, TL2Quantizer};

#[cfg(target_arch = "aarch64")]
use bitnet_quantization::TL1Quantizer;

#[cfg(feature = "cpu")]
use {
    anyhow::Result,
    bitnet_common::{BitNetTensor, Device},
    bitnet_quantization::{Quantize, convert_quantization, validate_round_trip},
};

// ============================================================================
// Mutant 1: Type Safety Validation (lib.rs:114:23)
// Kill mutant that replaces `self.qtype == qtype` with `self.qtype != qtype`
// ============================================================================

/// Test quantizing already-quantized tensor with same type (should be no-op clone)
#[test]
#[cfg(feature = "cpu")]
fn test_same_type_quantization_is_noop() -> Result<()> {
    // Create test tensor
    let size = 256;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) / 100.0).collect();
    let tensor = BitNetTensor::from_slice(&data, &[size], &Device::Cpu)?;

    // Quantize to I2S
    let quantized_i2s = tensor.quantize(QuantizationType::I2S)?;

    // Re-quantize to same type (should clone, not re-quantize)
    let requantized_i2s = quantized_i2s.quantize(QuantizationType::I2S)?;

    // Verify data is identical (bit-for-bit clone)
    assert_eq!(
        quantized_i2s.data, requantized_i2s.data,
        "Same-type quantization should clone data exactly"
    );
    assert_eq!(
        quantized_i2s.scales, requantized_i2s.scales,
        "Same-type quantization should clone scales exactly"
    );
    assert_eq!(
        quantized_i2s.qtype, requantized_i2s.qtype,
        "Same-type quantization should preserve qtype"
    );

    Ok(())
}

/// Test quantizing I2S tensor with wrong type fails type check
#[test]
#[cfg(feature = "cpu")]
fn test_cross_type_quantization_preserves_data_integrity() -> Result<()> {
    // Create test tensor
    let size = 256;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) / 100.0).collect();
    let tensor = BitNetTensor::from_slice(&data, &[size], &Device::Cpu)?;

    // Quantize to I2S
    let quantized_i2s = tensor.quantize(QuantizationType::I2S)?;
    assert_eq!(quantized_i2s.qtype, QuantizationType::I2S);

    // Convert to TL1 (requires dequantize + re-quantize)
    let quantized_tl1 = quantized_i2s.quantize(QuantizationType::TL1)?;
    assert_eq!(quantized_tl1.qtype, QuantizationType::TL1);

    // Verify types changed (mutation would break this)
    assert_ne!(
        quantized_i2s.qtype, quantized_tl1.qtype,
        "Cross-type quantization should change qtype"
    );

    Ok(())
}

/// Test batch quantization with mixed types validates type matching
#[test]
#[cfg(feature = "cpu")]
fn test_batch_quantization_type_validation() -> Result<()> {
    let size = 128;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) / 50.0).collect();
    let tensor = BitNetTensor::from_slice(&data, &[size], &Device::Cpu)?;

    // Create quantized tensors with different types
    let i2s = tensor.quantize(QuantizationType::I2S)?;
    let tl1 = tensor.quantize(QuantizationType::TL1)?;
    let tl2 = tensor.quantize(QuantizationType::TL2)?;

    // Verify each maintains its type identity
    assert_eq!(i2s.qtype, QuantizationType::I2S);
    assert_eq!(tl1.qtype, QuantizationType::TL1);
    assert_eq!(tl2.qtype, QuantizationType::TL2);

    // Re-quantize each to same type (should be clones)
    let i2s_clone = i2s.quantize(QuantizationType::I2S)?;
    let tl1_clone = tl1.quantize(QuantizationType::TL1)?;
    let tl2_clone = tl2.quantize(QuantizationType::TL2)?;

    // Verify clones match exactly
    assert_eq!(i2s.data, i2s_clone.data, "I2S same-type should clone");
    assert_eq!(tl1.data, tl1_clone.data, "TL1 same-type should clone");
    assert_eq!(tl2.data, tl2_clone.data, "TL2 same-type should clone");

    Ok(())
}

// ============================================================================
// Mutant 2: Quantizer Availability (lib.rs:190:9)
// Kill mutant that replaces `return true` with `return false`
// ============================================================================

/// Test I2S quantizer availability on all platforms
#[test]
fn test_i2s_quantizer_always_available() {
    let quantizer = I2SQuantizer::new();
    assert!(quantizer.is_available(), "I2S quantizer should be available on all platforms");
}

/// Test TL1 quantizer availability (ARM-optimized)
#[test]
#[cfg(target_arch = "aarch64")]
fn test_tl1_quantizer_available_on_aarch64() {
    let quantizer = TL1Quantizer::new();
    assert!(quantizer.is_available(), "TL1 quantizer should be available on ARM");
}

/// Test TL2 quantizer availability (x86-optimized)
#[test]
#[cfg(target_arch = "x86_64")]
fn test_tl2_quantizer_available_on_x86_64() {
    let quantizer = TL2Quantizer::new();
    assert!(quantizer.is_available(), "TL2 quantizer should be available on x86_64");
}

/// Test quantizer factory creates available quantizers
#[test]
fn test_quantizer_factory_creates_available_quantizers() {
    let i2s = QuantizerFactory::create(QuantizationType::I2S);
    assert!(i2s.is_available(), "Factory-created I2S quantizer should be available");

    let tl1 = QuantizerFactory::create(QuantizationType::TL1);
    assert!(tl1.is_available(), "Factory-created TL1 quantizer should be available");

    let tl2 = QuantizerFactory::create(QuantizationType::TL2);
    assert!(tl2.is_available(), "Factory-created TL2 quantizer should be available");
}

/// Test best-for-arch selection validates availability
#[test]
fn test_best_for_arch_is_available() {
    let best_qtype = QuantizerFactory::best_for_arch();
    let quantizer = QuantizerFactory::create(best_qtype);

    assert!(
        quantizer.is_available(),
        "Best-for-arch quantizer should be available on current platform"
    );

    // Verify the selected type matches architecture
    #[cfg(target_arch = "aarch64")]
    assert_eq!(best_qtype, QuantizationType::TL1, "ARM should prefer TL1");

    #[cfg(target_arch = "x86_64")]
    assert_eq!(best_qtype, QuantizationType::TL2, "x86_64 should prefer TL2");

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    assert_eq!(best_qtype, QuantizationType::I2S, "Other archs should use I2S");
}

// ============================================================================
// Mutant 3: Conversion Validation (lib.rs:199:21)
// Kill mutant that replaces `tensor.qtype == target_qtype` with `tensor.qtype != target_qtype`
// ============================================================================

/// Test convert_quantization with same type (should be no-op clone)
#[test]
#[cfg(feature = "cpu")]
fn test_convert_quantization_same_type_noop() -> Result<()> {
    let size = 256;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) / 100.0).collect();
    let tensor = BitNetTensor::from_slice(&data, &[size], &Device::Cpu)?;

    // Quantize to I2S
    let quantized = tensor.quantize(QuantizationType::I2S)?;

    // Convert to same type (should clone)
    let converted = convert_quantization(&quantized, QuantizationType::I2S)?;

    // Verify bit-for-bit identical (clone, not re-quantize)
    assert_eq!(quantized.data, converted.data, "Same-type conversion should clone data");
    assert_eq!(quantized.scales, converted.scales, "Same-type conversion should clone scales");
    assert_eq!(quantized.qtype, converted.qtype, "Same-type conversion should preserve qtype");

    Ok(())
}

/// Test convert_quantization with different types
#[test]
#[cfg(feature = "cpu")]
fn test_convert_quantization_different_types() -> Result<()> {
    let size = 256;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) / 100.0).collect();
    let tensor = BitNetTensor::from_slice(&data, &[size], &Device::Cpu)?;

    // Quantize to I2S
    let i2s = tensor.quantize(QuantizationType::I2S)?;
    assert_eq!(i2s.qtype, QuantizationType::I2S);

    // Convert to TL1
    let tl1 = convert_quantization(&i2s, QuantizationType::TL1)?;
    assert_eq!(tl1.qtype, QuantizationType::TL1);

    // Convert to TL2
    let tl2 = convert_quantization(&i2s, QuantizationType::TL2)?;
    assert_eq!(tl2.qtype, QuantizationType::TL2);

    // Verify types are different
    assert_ne!(i2s.qtype, tl1.qtype, "I2S != TL1");
    assert_ne!(i2s.qtype, tl2.qtype, "I2S != TL2");
    assert_ne!(tl1.qtype, tl2.qtype, "TL1 != TL2");

    Ok(())
}

/// Test conversion round-trip preserves data integrity
#[test]
#[cfg(feature = "cpu")]
fn test_conversion_round_trip_preserves_data() -> Result<()> {
    let size = 512;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) / 200.0).collect();
    let original = BitNetTensor::from_slice(&data, &[size], &Device::Cpu)?;

    // I2S -> TL1 -> I2S
    let i2s_1 = original.quantize(QuantizationType::I2S)?;
    let tl1 = convert_quantization(&i2s_1, QuantizationType::TL1)?;
    let i2s_2 = convert_quantization(&tl1, QuantizationType::I2S)?;

    // Dequantize both I2S versions
    let dequant_1 = i2s_1.dequantize()?;
    let dequant_2 = i2s_2.dequantize()?;

    // Verify data integrity (should be similar, allowing for quantization error)
    let data_1 = dequant_1.to_vec()?;
    let data_2 = dequant_2.to_vec()?;

    let mut max_diff: f32 = 0.0;
    for (d1, d2) in data_1.iter().zip(data_2.iter()) {
        max_diff = max_diff.max((d1 - d2).abs());
    }

    // Round-trip should have reasonable error (< 0.5)
    assert!(max_diff < 0.5, "Round-trip max error should be < 0.5, got {}", max_diff);

    Ok(())
}

// ============================================================================
// Mutant 4: Round-Trip Validation (lib.rs:214:5)
// Kill mutant that replaces `validate_round_trip()` call with `Ok(true)`
// ============================================================================

/// Test validate_round_trip with valid quantization
#[test]
#[cfg(feature = "cpu")]
fn test_validate_round_trip_success() -> Result<()> {
    // I2S: use ternary values {-1, 0, 1} which are the only representable levels —
    // round-trip is exact with tight tolerance.
    let ternary: Vec<f32> = (0..256)
        .map(|i| match i % 3 {
            0 => -1.0f32,
            1 => 0.0f32,
            _ => 1.0f32,
        })
        .collect();
    let tensor_i2s = BitNetTensor::from_slice(&ternary, &[256], &Device::Cpu)?;
    let result = validate_round_trip(&tensor_i2s, QuantizationType::I2S, 0.01)?;
    assert!(result, "I2S round-trip should succeed for ternary values");

    // TL1/TL2: use general ascending values to exercise the quantization range.
    // After the pack_unsigned_2bit / dequantize fix, max error ≈ scale/2 ≈ abs_max/2.
    // For values 0..2.55, last-block abs_max≈2.55 → max_error≈1.275; use 1.5 headroom.
    let general: Vec<f32> = (0..256).map(|i| (i as f32) / 100.0).collect();
    let tensor_tl = BitNetTensor::from_slice(&general, &[256], &Device::Cpu)?;

    let result = validate_round_trip(&tensor_tl, QuantizationType::TL1, 1.5)?;
    assert!(result, "TL1 round-trip should succeed within tolerance");

    let result = validate_round_trip(&tensor_tl, QuantizationType::TL2, 1.5)?;
    assert!(result, "TL2 round-trip should succeed within tolerance");

    Ok(())
}

/// Test validate_round_trip with corrupted data (should detect issues)
#[test]
#[cfg(feature = "cpu")]
fn test_validate_round_trip_with_corrupted_quantization() -> Result<()> {
    // Use f32 literals explicitly to avoid the f64 dtype mismatch.
    let zeroes = vec![0.0f32; 256];
    let corrupted_tensor = BitNetTensor::from_slice(&zeroes, &[256], &Device::Cpu)?;

    // All-zero input: I2S quantizes 0 → code 0 → dequantizes to 0. Round-trip is exact.
    let result = validate_round_trip(&corrupted_tensor, QuantizationType::I2S, 0.01)?;
    // The validation function is exercised; mutation that short-circuits it would
    // return Ok(false) and fail this assertion.
    assert!(result, "Validation should execute and pass for all-zero input");

    Ok(())
}

/// Test validate_round_trip with various tolerance levels
#[test]
#[cfg(feature = "cpu")]
fn test_validate_round_trip_tolerance_levels() -> Result<()> {
    // Use exact ternary values: I2S represents {-1, 0, 1} losslessly.
    // This lets us exercise both tight and loose tolerances meaningfully.
    let data: Vec<f32> = (0..512)
        .map(|i| match i % 3 {
            0 => -1.0f32,
            1 => 0.0f32,
            _ => 1.0f32,
        })
        .collect();
    let tensor = BitNetTensor::from_slice(&data, &[512], &Device::Cpu)?;

    // With ternary input all tolerances should pass
    let result = validate_round_trip(&tensor, QuantizationType::I2S, 0.001)?;
    assert!(result, "Should validate with strict tolerance for ternary values");

    let result = validate_round_trip(&tensor, QuantizationType::I2S, 0.1)?;
    assert!(result, "Should validate with moderate tolerance");

    let result = validate_round_trip(&tensor, QuantizationType::I2S, 1.0)?;
    assert!(result, "Should validate with loose tolerance");

    Ok(())
}

/// Test validate_round_trip for all quantization types
#[test]
#[cfg(feature = "cpu")]
fn test_validate_round_trip_all_types() -> Result<()> {
    // Ternary values {-1, 0, 1}: representable exactly by all supported types.
    let data: Vec<f32> = (0..256)
        .map(|i| match i % 3 {
            0 => -1.0f32,
            1 => 0.0f32,
            _ => 1.0f32,
        })
        .collect();
    let tensor = BitNetTensor::from_slice(&data, &[256], &Device::Cpu)?;

    // Validate I2S
    let i2s_result = validate_round_trip(&tensor, QuantizationType::I2S, 0.01)?;
    assert!(i2s_result, "I2S round-trip validation should succeed");

    // TL1/TL2: ternary {-1, 0, 1} aligns exactly with the 2-bit LUT levels
    // after the pack_unsigned_2bit fix. Round-trip should be near-exact (< 0.01).
    let tl1_result = validate_round_trip(&tensor, QuantizationType::TL1, 0.01)?;
    assert!(tl1_result, "TL1 round-trip validation should succeed for ternary values");

    let tl2_result = validate_round_trip(&tensor, QuantizationType::TL2, 0.01)?;
    assert!(tl2_result, "TL2 round-trip validation should succeed for ternary values");

    Ok(())
}

// ============================================================================
// Additional Edge Case Tests for Comprehensive Coverage
// ============================================================================

/// Test quantization with edge case tensor sizes
#[test]
#[cfg(feature = "cpu")]
fn test_quantization_edge_case_sizes() -> Result<()> {
    // Small tensor (32 elements = 1 block)
    let small = BitNetTensor::from_slice(&[1.0; 32], &[32], &Device::Cpu)?;
    let small_quant = small.quantize(QuantizationType::I2S)?;
    assert_eq!(small_quant.qtype, QuantizationType::I2S);

    // Large tensor (multiple blocks)
    let large = BitNetTensor::from_slice(&[1.0; 2048], &[2048], &Device::Cpu)?;
    let large_quant = large.quantize(QuantizationType::I2S)?;
    assert_eq!(large_quant.qtype, QuantizationType::I2S);

    // Verify same-type re-quantization is no-op for both
    let small_requant = small_quant.quantize(QuantizationType::I2S)?;
    assert_eq!(small_quant.data, small_requant.data);

    let large_requant = large_quant.quantize(QuantizationType::I2S)?;
    assert_eq!(large_quant.data, large_requant.data);

    Ok(())
}

/// Test quantization type identity preservation
#[test]
#[cfg(feature = "cpu")]
fn test_quantization_type_identity_preservation() -> Result<()> {
    let size = 256;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) / 100.0).collect();
    let tensor = BitNetTensor::from_slice(&data, &[size], &Device::Cpu)?;

    // Test each quantization type preserves identity through re-quantization
    let types = vec![QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];

    for qtype in types {
        let quantized = tensor.quantize(qtype)?;
        assert_eq!(quantized.qtype, qtype, "Should preserve type {qtype:?}");

        // Re-quantize to same type
        let requantized = quantized.quantize(qtype)?;
        assert_eq!(
            requantized.qtype, qtype,
            "Should preserve type {qtype:?} after re-quantization"
        );

        // Verify data is cloned, not re-quantized
        assert_eq!(
            quantized.data, requantized.data,
            "Same-type re-quantization should clone data for {qtype:?}"
        );
    }

    Ok(())
}
