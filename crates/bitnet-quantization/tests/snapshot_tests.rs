//! Snapshot tests for bitnet-quantization crate.
//!
//! These tests pin the string representation of core quantization types
//! (`QuantizationType` Display/Debug, `QuantizedTensor` metadata) so that
//! any accidental rename or serialization change is caught by CI.

use bitnet_common::QuantizationType;
use bitnet_quantization::QuantizerFactory;
use insta::assert_snapshot;

// -- QuantizationType Display and Debug --------------------------------------

#[test]
fn quantization_type_display_strings() {
    let types = [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
    let displays: Vec<String> = types.iter().map(|t| t.to_string()).collect();
    assert_snapshot!("quantization_type_display_strings", format!("{:?}", displays));
}

#[test]
fn quantization_type_debug_strings() {
    let types = [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
    let debugs: Vec<String> = types.iter().map(|t| format!("{:?}", t)).collect();
    assert_snapshot!("quantization_type_debug_strings", format!("{:?}", debugs));
}

// -- QuantizerFactory::best_for_arch -----------------------------------------

/// The best-for-arch selection is platform-specific, but it should always be
/// one of the valid types. This snapshot pins the value on the x86_64 CI runner.
#[cfg(target_arch = "x86_64")]
#[test]
fn best_for_arch_is_tl2_on_x86_64() {
    let best = QuantizerFactory::best_for_arch();
    assert_snapshot!("best_for_arch_x86_64", format!("{}", best));
}

#[cfg(target_arch = "aarch64")]
#[test]
fn best_for_arch_is_tl1_on_aarch64() {
    let best = QuantizerFactory::best_for_arch();
    assert_snapshot!("best_for_arch_aarch64", format!("{}", best));
}

// -- validate_round_trip on known inputs -------------------------------------

#[cfg(feature = "cpu")]
#[test]
fn validate_round_trip_i2s_small_tensor() {
    use bitnet_common::{BitNetTensor, Device};
    use bitnet_quantization::validate_round_trip;

    // A small 32-element (one I2S block) all-ones weight tensor.
    let data = vec![1.0f32; 32];
    let tensor = BitNetTensor::from_slice(&data, &[32], &Device::Cpu)
        .expect("BitNetTensor construction should succeed");
    let result = validate_round_trip(&tensor, QuantizationType::I2S, 1.0);
    assert!(result.is_ok(), "Round-trip validation should succeed: {:?}", result);
    let passed = result.unwrap();
    assert_snapshot!("round_trip_i2s_all_ones_passed", format!("{}", passed));
}

#[cfg(feature = "cpu")]
#[test]
fn validate_round_trip_tl2_small_tensor() {
    use bitnet_common::{BitNetTensor, Device};
    use bitnet_quantization::validate_round_trip;

    let data = vec![0.5f32; 32];
    let tensor = BitNetTensor::from_slice(&data, &[32], &Device::Cpu)
        .expect("BitNetTensor construction should succeed");
    let result = validate_round_trip(&tensor, QuantizationType::TL2, 1.0);
    assert!(result.is_ok(), "TL2 round-trip validation should succeed: {:?}", result);
    let passed = result.unwrap();
    assert_snapshot!("round_trip_tl2_half_ones_passed", format!("{}", passed));
}

// -- Wave 3: SIMD capabilities & quantization strategy -----------------------

use bitnet_quantization::simd_ops::{QuantizationStrategy, SimdCapabilities};

#[test]
fn simd_capabilities_detect_debug() {
    let caps = SimdCapabilities::detect();
    insta::assert_debug_snapshot!("simd_capabilities_detect", caps);
}

#[test]
fn simd_capabilities_best_strategy() {
    let caps = SimdCapabilities::detect();
    let strategy = caps.best_quantization_strategy();
    insta::assert_snapshot!("simd_best_strategy", format!("{strategy:?}"));
}

#[test]
fn quantization_strategy_variants_debug() {
    let strategies = [
        QuantizationStrategy::Scalar,
        QuantizationStrategy::SSE4_1,
        QuantizationStrategy::AVX2,
        QuantizationStrategy::AVX512,
        QuantizationStrategy::NEON,
    ];
    let debug: Vec<String> = strategies.iter().map(|s| format!("{s:?}")).collect();
    insta::assert_debug_snapshot!("quantization_strategy_variants", debug);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn simd_capabilities_x86_has_sse4_1() {
    let caps = SimdCapabilities::detect();
    insta::assert_snapshot!("simd_x86_sse4_1", format!("has_sse4_1={}", caps.has_sse4_1));
}

#[test]
fn simd_optimal_block_size() {
    let caps = SimdCapabilities::detect();
    let block_size = caps.optimal_block_size();
    assert!(block_size >= 32 && block_size <= 256, "block_size should be 32..256");
    insta::assert_snapshot!("simd_optimal_block_size", format!("block_size={block_size}"));
}

#[test]
fn quantization_strategy_scalar_is_fallback() {
    let caps =
        SimdCapabilities { has_avx512: false, has_avx2: false, has_neon: false, has_sse4_1: false };
    let strategy = caps.best_quantization_strategy();
    assert_eq!(strategy, QuantizationStrategy::Scalar);
    insta::assert_snapshot!("scalar_fallback_strategy", format!("{strategy:?}"));
}
