//! GGUF Weight Loading Property-Based Tests (Issue #159)
//!
//! Tests feature spec: gguf-weight-loading.md#quantization-accuracy-validation
//! API contract: gguf-weight-loading-api-contracts.md
//!
//! Property-based tests for GGUF weight loading quantization accuracy, numerical
//! stability, and edge case handling. Uses proptest framework to generate
//! comprehensive test cases and validate quantization properties.

#![allow(dead_code)] // Test utilities may be used by future tests

use anyhow::Result;
use bitnet_quantization::Quantize;
use proptest::prelude::*;
use proptest::prop_oneof;

/// Property-based test configuration
#[derive(Debug, Clone)]
pub struct PropertyTestConfig {
    pub accuracy_threshold: f32,
    pub numerical_tolerance: f64,
    pub max_tensor_size: usize,
    pub min_tensor_size: usize,
    pub test_iterations: u32,
}

impl Default for PropertyTestConfig {
    fn default() -> Self {
        Self {
            accuracy_threshold: 0.99,
            numerical_tolerance: 1e-5,
            max_tensor_size: 8192,
            min_tensor_size: 32,
            test_iterations: 100,
        }
    }
}

/// Generate arbitrary weight tensors for property testing
fn arbitrary_weight_tensor() -> impl Strategy<Value = Vec<f32>> {
    let config = PropertyTestConfig::default();
    prop::collection::vec(-10.0f32..10.0f32, config.min_tensor_size..config.max_tensor_size)
}

/// Generate arbitrary tensor shapes for testing
fn arbitrary_tensor_shape() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(1usize..128, 2..4)
}

/// Generate arbitrary quantization parameters
fn arbitrary_quantization_params() -> impl Strategy<Value = QuantizationTestParams> {
    (1usize..64, 0.01f32..10.0f32, -1.0f32..1.0f32).prop_map(|(block_size, scale, offset)| {
        QuantizationTestParams { block_size, scale, offset }
    })
}

// ============================================================================
// Enhanced Generators with Edge Cases
// ============================================================================

/// Generate weight tensors with edge case values (NaN, Inf, very small/large)
fn arbitrary_weight_tensor_with_edge_cases() -> impl Strategy<Value = Vec<f32>> {
    let config = PropertyTestConfig::default();
    prop::collection::vec(
        prop_oneof![
            // Normal values
            8 => -10.0f32..10.0f32,
            // Very small values (denormals)
            1 => f32::MIN_POSITIVE..f32::MIN_POSITIVE * 10.0,
            // Very large values
            1 => (f32::MAX / 2.0)..(f32::MAX / 1.5),
            // Values near zero
            2 => -1e-10f32..1e-10f32,
            // Edge values
            1 => prop::strategy::Just(0.0f32),
            1 => prop::strategy::Just(f32::NAN),
            1 => prop::strategy::Just(f32::INFINITY),
            1 => prop::strategy::Just(f32::NEG_INFINITY),
        ],
        config.min_tensor_size..config.max_tensor_size,
    )
}

/// Generate random model architectures
fn arbitrary_model_architecture() -> impl Strategy<Value = ModelArchitecture> {
    (
        1usize..64,                                   // num_layers
        256usize..4096,                               // hidden_size
        512usize..8192,                               // intermediate_size
        4usize..32,                                   // num_heads
        prop::bool::ANY,                              // use_bias
        prop::sample::select(vec![32, 64, 128, 256]), // block_size
    )
        .prop_map(
            |(num_layers, hidden_size, intermediate_size, num_heads, use_bias, block_size)| {
                ModelArchitecture {
                    num_layers,
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    use_bias,
                    block_size,
                }
            },
        )
}

/// Generate block-aligned tensor shapes (multiples of block_size)
fn arbitrary_block_aligned_shape(block_size: usize) -> impl Strategy<Value = Vec<usize>> {
    let bs = block_size.max(1);
    prop::collection::vec((1usize..16).prop_map(move |mult| mult * bs), 2..4)
}

/// Generate sparse weight tensors with controlled sparsity
fn arbitrary_sparse_weight_tensor(sparsity: f32) -> impl Strategy<Value = Vec<f32>> {
    let config = PropertyTestConfig::default();
    prop::collection::vec(
        prop_oneof![
            (sparsity * 100.0) as u32 => prop::strategy::Just(0.0f32),
            ((1.0 - sparsity) * 100.0) as u32 => -10.0f32..10.0f32,
        ],
        config.min_tensor_size..config.max_tensor_size,
    )
}

/// Generate tensors with extreme dynamic ranges
fn arbitrary_extreme_dynamic_range_tensor() -> impl Strategy<Value = Vec<f32>> {
    let config = PropertyTestConfig::default();
    prop::collection::vec(
        prop_oneof![
            5 => -1000.0f32..1000.0f32,
            2 => -1e-6f32..1e-6f32,
            1 => f32::MIN_POSITIVE..f32::MIN_POSITIVE * 100.0,
            1 => (f32::MAX / 10.0)..(f32::MAX / 5.0),
        ],
        config.min_tensor_size..config.max_tensor_size,
    )
}

/// Generate random scales for quantization
fn arbitrary_scales(num_blocks: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(0.001f32..100.0f32, num_blocks..num_blocks + 1)
}

/// Generate random zero points
fn arbitrary_zero_points(num_blocks: usize) -> impl Strategy<Value = Vec<i32>> {
    prop::collection::vec(-128i32..127i32, num_blocks..num_blocks + 1)
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct QuantizationTestParams {
    block_size: usize,
    scale: f32,
    offset: f32,
}

/// Model architecture for testing
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ModelArchitecture {
    num_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    use_bias: bool,
    block_size: usize,
}

// ============================================================================
// Property Tests for I2S Quantization
// ============================================================================

proptest! {
    /// Property: I2S quantization round-trip preserves distribution characteristics
    /// Tests feature spec: gguf-weight-loading.md#quantization-accuracy-validation
    /// AC2: Support Quantization Formats with ≥99% Accuracy
    #[test]
    #[ignore] // Issue #159: TDD placeholder - I2S quantization integration needed
    #[cfg(feature = "cpu")]
    fn prop_i2s_quantization_preserves_distribution(
        weight_data in arbitrary_weight_tensor(),
        shape in arbitrary_tensor_shape(),
        params in arbitrary_quantization_params()
    ) {
        let config = PropertyTestConfig::default();

        // Skip if tensor size doesn't match shape
        if weight_data.len() != shape.iter().product::<usize>() {
            return Ok(());
        }

        // TODO: This will initially fail until I2S quantization integration is complete
        let result = test_i2s_quantization_roundtrip(&weight_data, &shape, &params);

        match result {
            Ok(accuracy) => {
                prop_assert!(
                    accuracy >= config.accuracy_threshold,
                    "I2S quantization accuracy {:.4} below threshold {:.4}",
                    accuracy, config.accuracy_threshold
                );
            }
            Err(err) => {
                // Expected to fail in TDD Red phase
                eprintln!("Property test correctly failing (TDD Red): I2S quantization - {}", err);
                prop_assert!(false, "I2S quantization property test will pass once integration is complete");
            }
        }
    }

    /// Property: I2S quantization error bounds are consistent
    /// AC2: Support Quantization Formats with ≥99% Accuracy
    #[test]
    #[cfg(feature = "cpu")]
    fn prop_i2s_quantization_error_bounds(
        weight_data in arbitrary_weight_tensor(),
        shape in arbitrary_tensor_shape()
    ) {
        let _config = PropertyTestConfig::default();

        if weight_data.len() != shape.iter().product::<usize>() {
            return Ok(());
        }

        let result = test_i2s_quantization_error_bounds(&weight_data, &shape);

        match result {
            Ok((max_error, mean_error)) => {
                // Validate error bounds are reasonable
                prop_assert!(
                    max_error <= 1.0,
                    "I2S max quantization error {} too large",
                    max_error
                );
                prop_assert!(
                    mean_error <= 0.1,
                    "I2S mean quantization error {} too large",
                    mean_error
                );
            }
            Err(err) => {
                eprintln!("Property test correctly failing (TDD Red): I2S error bounds - {}", err);
                prop_assert!(false, "I2S error bounds test will pass once implementation is complete");
            }
        }
    }

    /// Property: I2S quantization is deterministic with same seed
    /// AC2: Support Quantization Formats with ≥99% Accuracy
    #[test]
    #[ignore] // Issue #159: TDD placeholder - I2S deterministic implementation needed
    #[cfg(feature = "cpu")]
    fn prop_i2s_quantization_deterministic(
        weight_data in arbitrary_weight_tensor(),
        shape in arbitrary_tensor_shape(),
        seed in 0u64..1000
    ) {
        if weight_data.len() != shape.iter().product::<usize>() {
            return Ok(());
        }

        // Set deterministic environment
        unsafe {
            std::env::set_var("BITNET_DETERMINISTIC", "1");
            std::env::set_var("BITNET_SEED", seed.to_string());
        }

        let result1 = test_i2s_quantization_deterministic(&weight_data, &shape, seed);
        let result2 = test_i2s_quantization_deterministic(&weight_data, &shape, seed);

        match (result1, result2) {
            (Ok(output1), Ok(output2)) => {
                prop_assert_eq!(
                    output1, output2,
                    "I2S quantization should be deterministic with same seed"
                );
            }
            _ => {
                eprintln!("Property test correctly failing (TDD Red): I2S deterministic");
                prop_assert!(false, "I2S deterministic test will pass once implementation is complete");
            }
        }
    }
}

// ============================================================================
// Property Tests for TL1 Quantization
// ============================================================================

proptest! {
    /// Property: TL1 quantization maintains numerical stability
    /// AC2: Support Quantization Formats with ≥99% Accuracy
    #[test]
    #[ignore] // Issue #159: TDD placeholder - TL1 quantization implementation needed
    #[cfg(feature = "cpu")]
    fn prop_tl1_quantization_numerical_stability(
        weight_data in arbitrary_weight_tensor(),
        shape in arbitrary_tensor_shape()
    ) {
        let config = PropertyTestConfig::default();

        if weight_data.len() != shape.iter().product::<usize>() {
            return Ok(());
        }

        let result = test_tl1_quantization_stability(&weight_data, &shape);

        match result {
            Ok((accuracy, stability_metric)) => {
                prop_assert!(
                    accuracy >= config.accuracy_threshold,
                    "TL1 quantization accuracy {:.4} below threshold {:.4}",
                    accuracy, config.accuracy_threshold
                );
                prop_assert!(
                    stability_metric.is_finite(),
                    "TL1 quantization stability metric should be finite"
                );
            }
            Err(err) => {
                eprintln!("Property test correctly failing (TDD Red): TL1 stability - {}", err);
                prop_assert!(false, "TL1 stability test will pass once implementation is complete");
            }
        }
    }

    /// Property: TL1 quantization preserves tensor sparsity patterns
    /// AC2: Support Quantization Formats with ≥99% Accuracy
    #[test]
    #[cfg(feature = "cpu")]
    fn prop_tl1_quantization_sparsity_preservation(
        weight_data in arbitrary_weight_tensor(),
        shape in arbitrary_tensor_shape(),
        sparsity_ratio in 0.0f32..0.9f32
    ) {
        if weight_data.len() != shape.iter().product::<usize>() {
            return Ok(());
        }

        // Create sparse weight data
        let sparse_weights = create_sparse_weights(&weight_data, sparsity_ratio);

        let result = test_tl1_sparsity_preservation(&sparse_weights, &shape, sparsity_ratio);

        match result {
            Ok(preserved_sparsity) => {
                let sparsity_error = (preserved_sparsity - sparsity_ratio).abs();
                prop_assert!(
                    sparsity_error <= 0.1,
                    "TL1 sparsity preservation error {:.3} too large, expected ~{:.3}, got {:.3}",
                    sparsity_error, sparsity_ratio, preserved_sparsity
                );
            }
            Err(err) => {
                eprintln!("Property test correctly failing (TDD Red): TL1 sparsity - {}", err);
                prop_assert!(false, "TL1 sparsity test will pass once implementation is complete");
            }
        }
    }
}

// ============================================================================
// Property Tests for TL2 Quantization
// ============================================================================

proptest! {
    /// Property: TL2 quantization handles extreme values gracefully
    /// AC2: Support Quantization Formats with ≥99% Accuracy
    #[test]
    #[ignore] // Issue #159: TDD placeholder - TL2 extreme value handling needed
    #[cfg(feature = "cpu")]
    fn prop_tl2_quantization_extreme_values(
        base_data in arbitrary_weight_tensor(),
        shape in arbitrary_tensor_shape(),
        extreme_multiplier in 1.0f32..1000.0f32
    ) {
        if base_data.len() != shape.iter().product::<usize>() {
            return Ok(());
        }

        // Create data with extreme values
        let extreme_data: Vec<f32> = base_data.iter()
            .map(|&x| if x.abs() > 1.0 { x * extreme_multiplier } else { x })
            .collect();

        let result = test_tl2_extreme_value_handling(&extreme_data, &shape);

        match result {
            Ok((accuracy, overflow_handled)) => {
                prop_assert!(
                    overflow_handled,
                    "TL2 should handle overflow gracefully"
                );
                prop_assert!(
                    accuracy >= 0.9, // Lower threshold for extreme values
                    "TL2 extreme value accuracy {:.4} below minimum 90%",
                    accuracy
                );
            }
            Err(err) => {
                eprintln!("Property test correctly failing (TDD Red): TL2 extreme values - {}", err);
                prop_assert!(false, "TL2 extreme values test will pass once implementation is complete");
            }
        }
    }

    /// Property: TL2 quantization block size affects accuracy predictably
    /// AC2: Support Quantization Formats with ≥99% Accuracy
    #[test]
    #[ignore] // Issue #159: TDD placeholder - TL2 block size scaling needed
    #[cfg(feature = "cpu")]
    fn prop_tl2_quantization_block_size_scaling(
        weight_data in arbitrary_weight_tensor(),
        shape in arbitrary_tensor_shape(),
        block_size in 8usize..128
    ) {
        if weight_data.len() != shape.iter().product::<usize>() {
            return Ok(());
        }

        // Ensure block size is compatible with tensor size
        let adjusted_block_size = std::cmp::min(block_size, weight_data.len());

        let result = test_tl2_block_size_effects(&weight_data, &shape, adjusted_block_size);

        match result {
            Ok(accuracy) => {
                // Larger block sizes should generally provide better accuracy
                let min_accuracy = if adjusted_block_size >= 32 { 0.95 } else { 0.9 };
                prop_assert!(
                    accuracy >= min_accuracy,
                    "TL2 accuracy {:.4} below expected {:.4} for block size {}",
                    accuracy, min_accuracy, adjusted_block_size
                );
            }
            Err(err) => {
                eprintln!("Property test correctly failing (TDD Red): TL2 block size - {}", err);
                prop_assert!(false, "TL2 block size test will pass once implementation is complete");
            }
        }
    }
}

// ============================================================================
// Property Tests for Cross-Quantization Consistency
// ============================================================================

proptest! {
    /// Property: Different quantization methods produce consistent relative orderings
    /// AC2: Support Quantization Formats with ≥99% Accuracy
    #[test]
    #[cfg(all(feature = "cpu", feature = "quantization"))]
    fn prop_cross_quantization_consistency(
        weight_data in arbitrary_weight_tensor(),
        shape in arbitrary_tensor_shape()
    ) {
        if weight_data.len() != shape.iter().product::<usize>() || weight_data.len() < 64 {
            return Ok(());
        }

        let result = test_cross_quantization_consistency(&weight_data, &shape);

        match result {
            Ok((i2s_output, tl1_output, tl2_output)) => {
                // Test that quantization methods preserve relative ordering
                let ordering_correlation = calculate_ordering_correlation(&i2s_output, &tl1_output);
                prop_assert!(
                    ordering_correlation >= 0.8,
                    "Cross-quantization ordering correlation {:.3} too low",
                    ordering_correlation
                );

                // Test that all methods maintain similar dynamic range
                let i2s_range = calculate_dynamic_range(&i2s_output);
                let tl1_range = calculate_dynamic_range(&tl1_output);
                let tl2_range = calculate_dynamic_range(&tl2_output);

                let max_range = i2s_range.max(tl1_range).max(tl2_range);
                let min_range = i2s_range.min(tl1_range).min(tl2_range);
                let range_ratio = min_range / max_range;

                prop_assert!(
                    range_ratio >= 0.5,
                    "Dynamic range variation too large: I2S={:.3}, TL1={:.3}, TL2={:.3}",
                    i2s_range, tl1_range, tl2_range
                );
            }
            Err(err) => {
                eprintln!("Property test correctly failing (TDD Red): Cross-quantization consistency - {}", err);
                prop_assert!(false, "Cross-quantization test will pass once all quantization methods are implemented");
            }
        }
    }
}

// ============================================================================
// Property Tests for Memory Efficiency
// ============================================================================

proptest! {
    /// Property: Memory usage scales linearly with tensor size
    /// AC7: Memory-Efficient Loading with Zero-Copy Operations
    #[test]
    #[ignore] // Issue #159: TDD placeholder - memory usage scaling implementation needed
    #[cfg(feature = "cpu")]
    fn prop_memory_usage_linear_scaling(
        base_size in (64usize..1024),
        scale_factor in (1usize..8)
    ) {
        let size1 = base_size;
        let size2 = base_size * scale_factor;

        let data1 = vec![1.0f32; size1];
        let data2 = vec![1.0f32; size2];

        let result = test_memory_usage_scaling(&data1, &data2, scale_factor);

        match result {
            Ok((_memory1, _memory2, actual_scale)) => {
                // Memory usage should scale approximately linearly
                let expected_scale = scale_factor as f32;
                let scale_error = (actual_scale - expected_scale).abs() / expected_scale;

                prop_assert!(
                    scale_error <= 0.2, // Allow 20% tolerance for overhead
                    "Memory scaling error {:.3} too large: expected {:.1}x, got {:.1}x",
                    scale_error, expected_scale, actual_scale
                );
            }
            Err(err) => {
                eprintln!("Property test correctly failing (TDD Red): Memory scaling - {}", err);
                prop_assert!(false, "Memory scaling test will pass once memory optimization is implemented");
            }
        }
    }

    /// Property: Zero-copy operations maintain memory efficiency
    /// AC7: Memory-Efficient Loading with Zero-Copy Operations
    #[test]
    #[cfg(feature = "cpu")]
    fn prop_zero_copy_memory_efficiency(
        tensor_size in (512usize..4096),
        alignment in prop::sample::select(vec![16usize, 32, 64])
    ) {
        let weight_data = vec![1.0f32; tensor_size];

        let result = test_zero_copy_efficiency(&weight_data, alignment);

        match result {
            Ok((copy_memory, zero_copy_memory, copy_saved)) => {
                // Zero-copy should use significantly less memory
                prop_assert!(
                    copy_saved,
                    "Zero-copy should save memory compared to copying"
                );

                let memory_ratio = zero_copy_memory as f32 / copy_memory as f32;
                prop_assert!(
                    memory_ratio <= 0.8,
                    "Zero-copy memory ratio {:.3} should be significantly lower",
                    memory_ratio
                );
            }
            Err(err) => {
                eprintln!("Property test correctly failing (TDD Red): Zero-copy efficiency - {}", err);
                prop_assert!(false, "Zero-copy test will pass once memory optimization is implemented");
            }
        }
    }
}

// ============================================================================
// Property Tests for Edge Cases and Numerical Stability
// ============================================================================

proptest! {
    /// Property: Quantization handles NaN and Inf gracefully
    /// AC2: Support Quantization Formats with ≥99% Accuracy
    #[test]
    #[ignore] // Issue #159: TDD placeholder - NaN/Inf handling implementation needed
    #[cfg(feature = "cpu")]
    fn prop_quantization_handles_nan_inf(
        weight_data in arbitrary_weight_tensor_with_edge_cases(),
        shape in arbitrary_tensor_shape()
    ) {
        if weight_data.len() != shape.iter().product::<usize>() {
            return Ok(());
        }

        let result = test_edge_case_handling(&weight_data, &shape);

        match result {
            Ok((nan_handled, inf_handled, finite_output)) => {
                prop_assert!(
                    nan_handled,
                    "NaN values should be handled without panicking"
                );
                prop_assert!(
                    inf_handled,
                    "Inf values should be handled without panicking"
                );
                prop_assert!(
                    finite_output.iter().all(|&x| x.is_finite()),
                    "Output should contain only finite values after quantization"
                );
            }
            Err(err) => {
                eprintln!("Property test correctly failing (TDD Red): Edge case handling - {}", err);
                prop_assert!(false, "Edge case handling test will pass once implementation is complete");
            }
        }
    }

    /// Property: Quantization preserves distribution characteristics
    /// AC2: Support Quantization Formats with ≥99% Accuracy
    #[test]
    #[ignore] // Issue #159: TDD placeholder - distribution preservation validation needed
    #[cfg(feature = "cpu")]
    fn prop_quantization_preserves_distribution(
        weight_data in arbitrary_weight_tensor(),
        shape in arbitrary_tensor_shape()
    ) {
        let config = PropertyTestConfig::default();

        if weight_data.len() != shape.iter().product::<usize>() {
            return Ok(());
        }

        let result = test_distribution_preservation(&weight_data, &shape);

        match result {
            Ok((mean_preserved, variance_preserved, correlation)) => {
                prop_assert!(
                    mean_preserved,
                    "Quantization should preserve mean within tolerance"
                );
                prop_assert!(
                    variance_preserved,
                    "Quantization should preserve variance within tolerance"
                );
                prop_assert!(
                    correlation >= config.accuracy_threshold,
                    "Quantization correlation {:.4} below threshold {:.4}",
                    correlation, config.accuracy_threshold
                );
            }
            Err(err) => {
                eprintln!("Property test correctly failing (TDD Red): Distribution preservation - {}", err);
                prop_assert!(false, "Distribution preservation test will pass once implementation is complete");
            }
        }
    }

    /// Property: Block-aligned tensors quantize efficiently
    /// AC2: Support Quantization Formats with ≥99% Accuracy
    #[test]
    #[ignore] // Issue #159: TDD placeholder - block alignment optimization needed
    #[cfg(feature = "cpu")]
    fn prop_block_aligned_quantization(
        block_size in prop::sample::select(vec![32usize, 64, 128, 256]),
        shape in arbitrary_tensor_shape()
    ) {
        let config = PropertyTestConfig::default();

        // Create block-aligned shape
        let aligned_shape: Vec<usize> = shape.iter()
            .map(|&dim| ((dim + block_size - 1) / block_size) * block_size)
            .collect();
        let size = aligned_shape.iter().product::<usize>();

        // Generate data matching the aligned shape
        let weight_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

        let result = test_block_aligned_efficiency(&weight_data, &aligned_shape, block_size);

        match result {
            Ok((accuracy, efficiency_gain)) => {
                prop_assert!(
                    accuracy >= config.accuracy_threshold,
                    "Block-aligned accuracy {:.4} below threshold {:.4}",
                    accuracy, config.accuracy_threshold
                );
                prop_assert!(
                    efficiency_gain >= 0.0,
                    "Block-aligned quantization should not degrade efficiency"
                );
            }
            Err(err) => {
                eprintln!("Property test correctly failing (TDD Red): Block alignment - {}", err);
                prop_assert!(false, "Block alignment test will pass once implementation is complete");
            }
        }
    }

    /// Property: Extreme dynamic range handling
    /// AC2: Support Quantization Formats with ≥99% Accuracy
    #[test]
    #[cfg(feature = "cpu")]
    fn prop_extreme_dynamic_range(
        weight_data in arbitrary_extreme_dynamic_range_tensor(),
        shape in arbitrary_tensor_shape()
    ) {
        if weight_data.len() != shape.iter().product::<usize>() {
            return Ok(());
        }

        let result = test_extreme_dynamic_range_handling(&weight_data, &shape);

        match result {
            Ok((dynamic_range, accuracy, clipping_handled)) => {
                prop_assert!(
                    clipping_handled,
                    "Extreme values should be clipped gracefully"
                );
                prop_assert!(
                    dynamic_range.is_finite(),
                    "Dynamic range should be finite after quantization"
                );
                // Lower accuracy threshold for extreme ranges
                prop_assert!(
                    accuracy >= 0.85,
                    "Extreme range accuracy {:.4} below minimum 85%",
                    accuracy
                );
            }
            Err(err) => {
                eprintln!("Property test correctly failing (TDD Red): Extreme range - {}", err);
                prop_assert!(false, "Extreme range test will pass once implementation is complete");
            }
        }
    }

    /// Property: Sparse tensors maintain sparsity after quantization
    /// AC2: Support Quantization Formats with ≥99% Accuracy
    #[test]
    #[ignore] // Issue #159: TDD placeholder - sparsity preservation validation needed
    #[cfg(feature = "cpu")]
    fn prop_sparse_tensor_handling(
        sparsity in 0.1f32..0.9f32,
        base_data in arbitrary_weight_tensor()
    ) {
        // Create sparse weight data by zeroing out elements
        let zero_count = (base_data.len() as f32 * sparsity) as usize;
        let mut weight_data = base_data.clone();
        for i in 0..zero_count.min(weight_data.len()) {
            weight_data[i] = 0.0;
        }

        // Create shape that matches size
        let size = weight_data.len();
        let rows = (size as f32).sqrt() as usize;
        let cols = size / rows;
        let shape = vec![rows, cols];

        // Skip if shape doesn't match
        if shape.iter().product::<usize>() != size {
            return Ok(());
        }

        let result = test_sparse_tensor_preservation(&weight_data, &shape, sparsity);

        match result {
            Ok((preserved_sparsity, compression_ratio)) => {
                let sparsity_error = (preserved_sparsity - sparsity).abs();
                prop_assert!(
                    sparsity_error <= 0.15,
                    "Sparsity preservation error {:.3} too large",
                    sparsity_error
                );
                prop_assert!(
                    compression_ratio >= 1.0,
                    "Sparse tensors should achieve compression ratio ≥ 1.0, got {:.2}",
                    compression_ratio
                );
            }
            Err(err) => {
                eprintln!("Property test correctly failing (TDD Red): Sparse tensor - {}", err);
                prop_assert!(false, "Sparse tensor test will pass once implementation is complete");
            }
        }
    }

    /// Property: Model architecture variations are supported
    /// AC2: Support Quantization Formats with ≥99% Accuracy
    #[test]
    #[ignore] // Issue #159: TDD placeholder - architecture validation needed
    #[cfg(feature = "cpu")]
    fn prop_model_architecture_support(
        arch in arbitrary_model_architecture()
    ) {
        let result = test_architecture_compatibility(&arch);

        match result {
            Ok((supported, accuracy)) => {
                prop_assert!(
                    supported,
                    "Architecture should be supported: {:?}",
                    arch
                );
                prop_assert!(
                    accuracy >= 0.99,
                    "Architecture accuracy {:.4} below threshold 99%",
                    accuracy
                );
            }
            Err(err) => {
                eprintln!("Property test correctly failing (TDD Red): Architecture support - {}", err);
                prop_assert!(false, "Architecture support test will pass once implementation is complete");
            }
        }
    }

    /// Property: Round-trip quantization with custom scales and zero points
    /// AC2: Support Quantization Formats with ≥99% Accuracy
    #[test]
    #[ignore] // Issue #159: TDD placeholder - custom quantization params implementation needed
    #[cfg(feature = "cpu")]
    fn prop_custom_quantization_params(
        weight_data in arbitrary_weight_tensor(),
        shape in arbitrary_tensor_shape()
    ) {
        let config = PropertyTestConfig::default();

        if weight_data.len() != shape.iter().product::<usize>() || weight_data.len() < 32 {
            return Ok(());
        }

        let num_blocks = (weight_data.len() + 31) / 32;
        // Generate simple scales and zero points inline
        let scales: Vec<f32> = (0..num_blocks).map(|i| 0.1 + (i as f32) * 0.01).collect();
        let zero_points: Vec<i32> = (0..num_blocks).map(|i| (i as i32) % 128).collect();

        let result = test_custom_quantization_params(&weight_data, &shape, &scales, &zero_points);

        match result {
            Ok(accuracy) => {
                prop_assert!(
                    accuracy >= config.accuracy_threshold,
                    "Custom param accuracy {:.4} below threshold {:.4}",
                    accuracy, config.accuracy_threshold
                );
            }
            Err(err) => {
                eprintln!("Property test correctly failing (TDD Red): Custom params - {}", err);
                prop_assert!(false, "Custom params test will pass once implementation is complete");
            }
        }
    }
}

// ============================================================================
// Helper Functions for Property Testing
// ============================================================================

/// Test I2S quantization round-trip accuracy
fn test_i2s_quantization_roundtrip(
    weight_data: &[f32],
    shape: &[usize],
    params: &QuantizationTestParams,
) -> Result<f32> {
    // TODO: Implement I2S quantization round-trip test
    let _ = (weight_data, shape, params);
    Err(anyhow::anyhow!("I2S quantization integration not implemented"))
}

/// Test I2S quantization error bounds
fn test_i2s_quantization_error_bounds(weight_data: &[f32], shape: &[usize]) -> Result<(f32, f32)> {
    // TODO: Implement I2S error bounds analysis
    let _ = (weight_data, shape);
    Err(anyhow::anyhow!("I2S error bounds analysis not implemented"))
}

/// Test I2S quantization deterministic behavior
fn test_i2s_quantization_deterministic(
    weight_data: &[f32],
    shape: &[usize],
    seed: u64,
) -> Result<Vec<f32>> {
    // TODO: Implement I2S deterministic quantization test
    let _ = (weight_data, shape, seed);
    Err(anyhow::anyhow!("I2S deterministic test not implemented"))
}

/// Test TL1 quantization numerical stability
fn test_tl1_quantization_stability(weight_data: &[f32], shape: &[usize]) -> Result<(f32, f32)> {
    use bitnet_quantization::TL1Quantizer;

    // Create tensor from weight data
    let bitnet_tensor = bitnet_quantization::utils::create_tensor_from_f32(
        weight_data.to_vec(),
        shape,
        &candle_core::Device::Cpu,
    )?;

    // Quantize using TL1 (4-bit table lookup quantization)
    let quantizer = TL1Quantizer::new();
    let quantized = quantizer.quantize_tensor(&bitnet_tensor)?;

    // Dequantize back to f32
    let dequantized = quantized.dequantize()?;

    // Extract data for validation
    let original_data = bitnet_quantization::utils::extract_f32_data(&bitnet_tensor)?;
    let dequantized_data = bitnet_quantization::utils::extract_f32_data(&dequantized)?;

    // Ensure same length
    if original_data.len() != dequantized_data.len() {
        return Err(anyhow::anyhow!(
            "Length mismatch: original={}, dequantized={}",
            original_data.len(),
            dequantized_data.len()
        ));
    }

    // Filter finite values for fair comparison
    let finite_pairs: Vec<(f32, f32)> = original_data
        .iter()
        .zip(dequantized_data.iter())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (*a, *b))
        .collect();

    if finite_pairs.is_empty() {
        return Err(anyhow::anyhow!("No finite values to compare"));
    }

    let finite_original: Vec<f32> = finite_pairs.iter().map(|(a, _)| *a).collect();
    let finite_dequantized: Vec<f32> = finite_pairs.iter().map(|(_, b)| *b).collect();

    // Calculate MSE (Mean Squared Error)
    let mse = calculate_mse(&finite_original, &finite_dequantized);
    let signal_power = calculate_signal_power(&finite_original);

    // Calculate accuracy: 1 - (MSE / signal_power)
    let accuracy = if signal_power > 1e-10 {
        1.0 - (mse / signal_power)
    } else {
        // For near-zero signals, check if MSE is also near-zero
        if mse < 1e-6 { 1.0 } else { 0.0 }
    };

    // Calculate stability metric (variance of errors)
    // Errors = original - dequantized
    let errors: Vec<f32> = finite_pairs.iter().map(|(orig, deq)| orig - deq).collect();

    // Calculate mean of errors
    let error_mean =
        if !errors.is_empty() { errors.iter().sum::<f32>() / errors.len() as f32 } else { 0.0 };

    // Calculate variance of errors (stability metric)
    let stability_metric = if !errors.is_empty() {
        errors.iter().map(|e| (e - error_mean).powi(2)).sum::<f32>() / errors.len() as f32
    } else {
        0.0
    };

    Ok((accuracy, stability_metric))
}

/// Test TL1 sparsity preservation
fn test_tl1_sparsity_preservation(
    weight_data: &[f32],
    shape: &[usize],
    target_sparsity: f32,
) -> Result<f32> {
    use bitnet_quantization::TL1Quantizer;

    // Create tensor from sparse weight data
    let bitnet_tensor = bitnet_quantization::utils::create_tensor_from_f32(
        weight_data.to_vec(),
        shape,
        &candle_core::Device::Cpu,
    )?;

    // Quantize using TL1
    let quantizer = TL1Quantizer::new();
    let quantized = quantizer.quantize_tensor(&bitnet_tensor)?;

    // Dequantize back
    let dequantized = quantizer.dequantize_tensor(&quantized)?;

    // Extract dequantized data
    let dequantized_data = bitnet_quantization::utils::extract_f32_data(&dequantized)?;

    // Count zeros in output (sparsity tolerance for floating point)
    let zero_threshold = 1e-6;
    let zero_count = dequantized_data.iter().filter(|&&x| x.abs() < zero_threshold).count();
    let total_count = dequantized_data.len();

    // Calculate preserved sparsity
    let preserved_sparsity =
        if total_count == 0 { 0.0 } else { zero_count as f32 / total_count as f32 };

    Ok(preserved_sparsity)
}

/// Test TL2 extreme value handling
fn test_tl2_extreme_value_handling(weight_data: &[f32], shape: &[usize]) -> Result<(f32, bool)> {
    // TODO: Implement TL2 extreme value handling test
    let _ = (weight_data, shape);
    Err(anyhow::anyhow!("TL2 extreme value handling not implemented"))
}

/// Test TL2 block size effects on accuracy
fn test_tl2_block_size_effects(
    weight_data: &[f32],
    shape: &[usize],
    block_size: usize,
) -> Result<f32> {
    // TODO: Implement TL2 block size analysis
    let _ = (weight_data, shape, block_size);
    Err(anyhow::anyhow!("TL2 block size analysis not implemented"))
}

/// Test cross-quantization consistency
#[allow(dead_code)]
fn test_cross_quantization_consistency(
    weight_data: &[f32],
    shape: &[usize],
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    // TODO: Implement cross-quantization consistency test
    let _ = (weight_data, shape);
    Err(anyhow::anyhow!("Cross-quantization consistency not implemented"))
}

/// Test memory usage scaling
fn test_memory_usage_scaling(
    data1: &[f32],
    data2: &[f32],
    expected_scale: usize,
) -> Result<(usize, usize, f32)> {
    // TODO: Implement memory usage scaling test
    let _ = (data1, data2, expected_scale);
    Err(anyhow::anyhow!("Memory usage scaling not implemented"))
}

/// Test zero-copy memory efficiency
fn test_zero_copy_efficiency(
    weight_data: &[f32],
    alignment: usize,
) -> Result<(usize, usize, bool)> {
    use bitnet_models::loader::MmapFile;
    use std::fs::File;
    use std::io::Write;
    use sysinfo::{MemoryRefreshKind, RefreshKind, System};

    // Create a temporary file with aligned weight data
    let temp_dir = tempfile::tempdir()?;
    let temp_path = temp_dir.path().join("test_weights.bin");

    // Write weight data as bytes (aligned)
    let mut file = File::create(&temp_path)?;
    let byte_data: Vec<u8> = weight_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    // Add padding for alignment if needed
    let padding = (alignment - (byte_data.len() % alignment)) % alignment;
    file.write_all(&byte_data)?;
    file.write_all(&vec![0u8; padding])?;
    file.sync_all()?;
    drop(file);

    // Initialize system info for memory tracking
    let mut sys = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
    );

    // Measure memory with copy-based loading
    sys.refresh_memory();
    let memory_before_copy = sys.used_memory();

    // Simulate copy-based loading: read entire file into Vec
    let copy_data = std::fs::read(&temp_path)?;
    let _copy_vec = copy_data.to_vec(); // Force allocation

    sys.refresh_memory();
    let memory_after_copy = sys.used_memory();
    let copy_memory = memory_after_copy.saturating_sub(memory_before_copy);

    // Drop copy data to free memory
    drop(_copy_vec);
    drop(copy_data);

    // Force garbage collection and wait a bit for memory to settle
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Measure memory with zero-copy (mmap) loading
    sys.refresh_memory();
    let memory_before_mmap = sys.used_memory();

    // Use memory-mapped file for zero-copy access
    let mmap_file = MmapFile::open(&temp_path)?;
    let _mmap_slice = mmap_file.as_slice(); // Access but don't copy

    sys.refresh_memory();
    let memory_after_mmap = sys.used_memory();
    let mmap_memory = memory_after_mmap.saturating_sub(memory_before_mmap);

    // Calculate memory savings
    let copy_saved = copy_memory > mmap_memory;

    // Clean up
    drop(mmap_file);
    drop(temp_dir);

    Ok((copy_memory as usize, mmap_memory as usize, copy_saved))
}

/// Create sparse weight data
fn create_sparse_weights(weight_data: &[f32], sparsity_ratio: f32) -> Vec<f32> {
    let zero_count = (weight_data.len() as f32 * sparsity_ratio) as usize;
    let mut sparse_data = weight_data.to_vec();

    // Set first zero_count elements to zero
    for i in 0..zero_count.min(sparse_data.len()) {
        sparse_data[i] = 0.0;
    }

    sparse_data
}

/// Calculate ordering correlation between two vectors
#[allow(dead_code)]
fn calculate_ordering_correlation(vec1: &[f32], vec2: &[f32]) -> f32 {
    if vec1.len() != vec2.len() || vec1.is_empty() {
        return 0.0;
    }

    // Simple rank correlation approximation
    let mut concordant_pairs = 0;
    let mut total_pairs = 0;

    for i in 0..vec1.len() {
        for j in (i + 1)..vec1.len() {
            total_pairs += 1;
            let order1 = vec1[i] < vec1[j];
            let order2 = vec2[i] < vec2[j];
            if order1 == order2 {
                concordant_pairs += 1;
            }
        }
    }

    if total_pairs > 0 { concordant_pairs as f32 / total_pairs as f32 } else { 1.0 }
}

/// Calculate dynamic range of values
#[allow(dead_code)]
fn calculate_dynamic_range(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));

    max_val - min_val
}

// ============================================================================
// Helper Functions for Edge Case Testing
// ============================================================================

/// Test edge case handling (NaN, Inf)
fn test_edge_case_handling(weight_data: &[f32], shape: &[usize]) -> Result<(bool, bool, Vec<f32>)> {
    // TODO: Implement edge case handling test
    let _ = (weight_data, shape);
    Err(anyhow::anyhow!("Edge case handling not implemented"))
}

/// Test distribution preservation after quantization
fn test_distribution_preservation(
    weight_data: &[f32],
    shape: &[usize],
) -> Result<(bool, bool, f32)> {
    // TODO: Implement distribution preservation test
    // Returns (mean_preserved, variance_preserved, correlation)
    let _ = (weight_data, shape);
    Err(anyhow::anyhow!("Distribution preservation not implemented"))
}

/// Test block-aligned quantization efficiency
fn test_block_aligned_efficiency(
    weight_data: &[f32],
    shape: &[usize],
    block_size: usize,
) -> Result<(f32, f32)> {
    // TODO: Implement block alignment efficiency test
    // Returns (accuracy, efficiency_gain)
    let _ = (weight_data, shape, block_size);
    Err(anyhow::anyhow!("Block alignment efficiency not implemented"))
}

/// Test extreme dynamic range handling
fn test_extreme_dynamic_range_handling(
    weight_data: &[f32],
    shape: &[usize],
) -> Result<(f32, f32, bool)> {
    use bitnet_quantization::{I2SQuantizer, Quantize};

    // 1. Find min/max in weight_data (extreme values)
    let finite_values: Vec<f32> = weight_data.iter().copied().filter(|x| x.is_finite()).collect();

    if finite_values.is_empty() {
        // All values are non-finite, return minimal valid result
        return Ok((0.0, 0.0, false));
    }

    let min_val = finite_values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = finite_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let dynamic_range = max_val - min_val;

    // 2. Quantize with I2S (handles clipping/saturation automatically)
    let bitnet_tensor = bitnet_quantization::utils::create_tensor_from_f32(
        weight_data.to_vec(),
        shape,
        &candle_core::Device::Cpu,
    )?;

    let quantizer = I2SQuantizer::new();
    let quantized = quantizer.quantize_tensor(&bitnet_tensor)?;

    // 3. Dequantize back
    let dequantized = quantized.dequantize()?;
    let dequantized_data = bitnet_quantization::utils::extract_f32_data(&dequantized)?;

    // 4. Calculate accuracy (MSE-based)
    let mse = calculate_mse(weight_data, &dequantized_data);
    let signal_power = calculate_signal_power(weight_data);

    let accuracy = if signal_power > 1e-10 {
        (1.0 - (mse / signal_power)).max(0.0)
    } else {
        // For near-zero signals, check if dequantized is also near-zero
        if mse < 1e-6 {
            1.0 // Perfect match for zero signal
        } else {
            0.0 // Mismatch
        }
    };

    // 5. Ensure clipping_handled = all outputs are finite
    let clipping_handled = dequantized_data.iter().all(|x| x.is_finite());

    // 6. Return (dynamic_range, accuracy, clipping_handled)
    Ok((dynamic_range, accuracy, clipping_handled))
}

/// Test sparse tensor preservation
fn test_sparse_tensor_preservation(
    weight_data: &[f32],
    shape: &[usize],
    target_sparsity: f32,
) -> Result<(f32, f32)> {
    use bitnet_quantization::{I2SQuantizer, Quantize};

    // 1. Count original zeros to validate input sparsity
    let original_zero_count = weight_data.iter().filter(|&&x| x == 0.0).count();
    let _original_sparsity = original_zero_count as f32 / weight_data.len() as f32;

    // 2. Quantize using I2S
    let bitnet_tensor = bitnet_quantization::utils::create_tensor_from_f32(
        weight_data.to_vec(),
        shape,
        &candle_core::Device::Cpu,
    )?;

    let quantizer = I2SQuantizer::new();
    let quantized = quantizer.quantize_tensor(&bitnet_tensor)?;

    // 3. Dequantize back
    let dequantized = quantized.dequantize()?;
    let dequantized_data = bitnet_quantization::utils::extract_f32_data(&dequantized)?;

    // 4. Calculate preserved sparsity (count zeros in dequantized output)
    // For I2S quantization, zeros may not be exactly preserved, so use a small threshold
    let threshold = 1e-5;
    let preserved_zero_count = dequantized_data.iter().filter(|&&x| x.abs() < threshold).count();
    let preserved_sparsity = preserved_zero_count as f32 / dequantized_data.len() as f32;

    // 5. Calculate compression ratio
    // Original size: weight_data.len() * sizeof(f32) = len * 4 bytes
    // Quantized size: I2S uses 2 bits per weight + scale factors
    // For I2S with block size 32: (2 bits * N) + (F16 scale per 32 elements)
    // Scale storage: N/32 * 2 bytes (F16)
    let original_size_bytes = weight_data.len() * 4; // f32 = 4 bytes
    let num_weights = weight_data.len();
    let block_size = 32; // I2S typical block size
    let num_blocks = (num_weights + block_size - 1) / block_size;

    // Quantized storage:
    // - 2 bits per weight = num_weights * 2 / 8 bytes
    // - F16 scale per block = num_blocks * 2 bytes
    let quantized_weights_bytes = (num_weights * 2 + 7) / 8; // Round up
    let quantized_scales_bytes = num_blocks * 2; // F16 = 2 bytes
    let quantized_size_bytes = quantized_weights_bytes + quantized_scales_bytes;

    let compression_ratio = if quantized_size_bytes > 0 {
        original_size_bytes as f32 / quantized_size_bytes as f32
    } else {
        1.0
    };

    // 6. Return (preserved_sparsity, compression_ratio)
    Ok((preserved_sparsity, compression_ratio))
}

/// Test architecture compatibility
fn test_architecture_compatibility(arch: &ModelArchitecture) -> Result<(bool, f32)> {
    // TODO: Implement architecture compatibility test
    // Returns (supported, accuracy)
    let _ = arch;
    Err(anyhow::anyhow!("Architecture compatibility not implemented"))
}

/// Test custom quantization parameters
fn test_custom_quantization_params(
    weight_data: &[f32],
    shape: &[usize],
    scales: &[f32],
    zero_points: &[i32],
) -> Result<f32> {
    // Implement custom quantization with provided scales and zero points
    let block_size = 32;
    let num_blocks = (weight_data.len() + block_size - 1) / block_size;

    // Validate scales and zero_points match number of blocks
    if scales.len() < num_blocks || zero_points.len() < num_blocks {
        return Err(anyhow::anyhow!(
            "Insufficient scales ({}) or zero_points ({}) for {} blocks",
            scales.len(),
            zero_points.len(),
            num_blocks
        ));
    }

    // Manual quantization with custom parameters
    let mut quantized = Vec::with_capacity(weight_data.len());
    for (block_idx, chunk) in weight_data.chunks(block_size).enumerate() {
        let scale = scales[block_idx];
        let zero_point = zero_points[block_idx];

        for &value in chunk {
            // Quantize to 2-bit signed [-2, 1]
            let quant_val = if scale > 0.0 && scale.is_finite() {
                let normalized = value / scale;
                let shifted = normalized - zero_point as f32;
                shifted.round().clamp(-2.0, 1.0) as i8
            } else {
                0i8
            };
            quantized.push(quant_val);
        }
    }

    // Dequantize back to f32
    let mut dequantized = Vec::with_capacity(weight_data.len());
    for (block_idx, chunk) in quantized.chunks(block_size).enumerate() {
        let scale = scales[block_idx];
        let zero_point = zero_points[block_idx];

        for &quant_val in chunk {
            let deq = (quant_val as f32 + zero_point as f32) * scale;
            dequantized.push(deq);
        }
    }

    // Calculate accuracy using MSE-based metric
    let mse =
        weight_data.iter().zip(&dequantized).map(|(orig, deq)| (orig - deq).powi(2)).sum::<f32>()
            / weight_data.len() as f32;

    let signal_power =
        weight_data.iter().map(|x| x.powi(2)).sum::<f32>() / weight_data.len() as f32;

    // Avoid division by zero
    let accuracy = if signal_power > 1e-10 {
        1.0 - (mse / signal_power).min(1.0)
    } else {
        // For near-zero signals, check if MSE is also near zero
        if mse < 1e-10 { 1.0 } else { 0.0 }
    };

    Ok(accuracy.max(0.0).min(1.0))
}

// ============================================================================
// Statistical Helper Functions
// ============================================================================

/// Calculate mean of f32 slice
#[allow(dead_code)]
fn calculate_mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let finite_values: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if finite_values.is_empty() {
        return 0.0;
    }
    finite_values.iter().sum::<f32>() / finite_values.len() as f32
}

/// Calculate variance of f32 slice
#[allow(dead_code)]
fn calculate_variance(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let finite_values: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if finite_values.is_empty() {
        return 0.0;
    }
    let mean = calculate_mean(&finite_values);
    finite_values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / finite_values.len() as f32
}

/// Calculate Pearson correlation coefficient
#[allow(dead_code)]
fn calculate_correlation(vec1: &[f32], vec2: &[f32]) -> f32 {
    if vec1.len() != vec2.len() || vec1.is_empty() {
        return 0.0;
    }

    let mean1 = calculate_mean(vec1);
    let mean2 = calculate_mean(vec2);

    let mut numerator = 0.0;
    let mut sum_sq1 = 0.0;
    let mut sum_sq2 = 0.0;

    for i in 0..vec1.len() {
        if vec1[i].is_finite() && vec2[i].is_finite() {
            let diff1 = vec1[i] - mean1;
            let diff2 = vec2[i] - mean2;
            numerator += diff1 * diff2;
            sum_sq1 += diff1 * diff1;
            sum_sq2 += diff2 * diff2;
        }
    }

    let denominator = (sum_sq1 * sum_sq2).sqrt();
    if denominator > 0.0 { numerator / denominator } else { 0.0 }
}

/// Calculate sparsity ratio (proportion of zeros)
#[allow(dead_code)]
fn calculate_sparsity(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let zero_count = values.iter().filter(|&&x| x.abs() < 1e-8).count();
    zero_count as f32 / values.len() as f32
}

/// Calculate Mean Squared Error between two vectors
#[allow(dead_code)]
fn calculate_mse(original: &[f32], dequantized: &[f32]) -> f32 {
    if original.len() != dequantized.len() || original.is_empty() {
        return f32::MAX;
    }

    let mse: f32 = original
        .iter()
        .zip(dequantized.iter())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / original.len() as f32;

    mse
}

/// Calculate signal power (variance of original signal)
#[allow(dead_code)]
fn calculate_signal_power(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }

    let finite_values: Vec<f32> = signal.iter().copied().filter(|x| x.is_finite()).collect();
    if finite_values.is_empty() {
        return 0.0;
    }

    let mean = calculate_mean(&finite_values);
    finite_values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / finite_values.len() as f32
}
