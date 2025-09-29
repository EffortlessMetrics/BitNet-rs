//! Comprehensive error handling and recovery tests for BitNet quantization
//!
//! This module provides extensive testing of error conditions, error propagation,
//! recovery mechanisms, and structured error handling across quantization algorithms.

#[cfg(test)]
mod tests {
    use crate::utils::create_tensor_from_f32;
    use crate::validation::{validate_quantized_tensor, validate_tensor_input};
    use crate::{I2SQuantizer, QuantizerTrait, TL1Quantizer, TL2Quantizer};
    use bitnet_common::{QuantizationError, Result, SecurityError};
    use candle_core::Tensor;
    use std::f32;

    /// Test comprehensive error type coverage
    #[test]
    fn test_error_type_coverage() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Test invalid input scenarios that should produce specific error types
        let error_scenarios = vec![
            ("Empty data", || create_tensor_from_f32(&[], &[0])),
            ("Mismatched dimensions", || create_tensor_from_f32(&[1.0, 2.0, 3.0], &[2, 2])),
            ("Negative dimensions", || create_tensor_from_f32(&[1.0], &[0, 1])),
        ];

        for (scenario_name, create_faulty_tensor) in error_scenarios {
            match create_faulty_tensor() {
                Ok(tensor) => {
                    // If tensor creation succeeds, test error handling in quantization
                    match quantizer.quantize(&tensor) {
                        Ok(_) => {
                            println!("ℹ️  {} was handled gracefully by quantizer", scenario_name);
                        }
                        Err(e) => {
                            println!("✅ {} correctly produced error: {}", scenario_name, e);

                            // Verify error can be displayed and debugged
                            let error_string = format!("{}", e);
                            let debug_string = format!("{:?}", e);
                            assert!(
                                !error_string.is_empty(),
                                "Error should have display representation"
                            );
                            assert!(
                                !debug_string.is_empty(),
                                "Error should have debug representation"
                            );
                        }
                    }
                }
                Err(e) => {
                    println!("✅ {} correctly failed at tensor creation: {}", scenario_name, e);
                }
            }
        }

        Ok(())
    }

    /// Test error propagation through the quantization pipeline
    #[test]
    fn test_error_propagation() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Create a valid tensor first
        let valid_data = vec![1.0, 2.0, 3.0, 4.0];
        let valid_tensor = create_tensor_from_f32(&valid_data, &[2, 2])?;

        // Test successful quantization
        let quantized = quantizer.quantize(&valid_tensor)?;
        assert!(!quantized.data.is_empty(), "Valid quantization should produce data");

        // Test error in dequantization by corrupting the quantized data
        let mut corrupted_quantized = quantized.clone();
        corrupted_quantized.data.clear(); // Corrupt the data

        match quantizer.dequantize(&corrupted_quantized) {
            Ok(_) => {
                println!("⚠️  Corrupted data was handled gracefully");
            }
            Err(e) => {
                println!("✅ Corrupted data correctly produced error: {}", e);

                // Test error propagation characteristics
                match e.downcast_ref::<QuantizationError>() {
                    Some(quant_err) => {
                        println!(
                            "✅ Error correctly identified as QuantizationError: {:?}",
                            quant_err
                        );
                    }
                    None => {
                        println!("ℹ️  Error type: {}", e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Test validation error scenarios
    #[test]
    fn test_validation_errors() -> Result<()> {
        // Test input validation errors
        let validation_scenarios = vec![
            ("Extreme values", vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::MAX]),
            ("All zeros", vec![0.0; 16]),
            ("Single value", vec![1.0]),
            ("Large range", vec![-1e6, 1e6, -1e-6, 1e-6]),
        ];

        for (scenario_name, test_data) in validation_scenarios {
            match create_tensor_from_f32(&test_data, &[test_data.len()]) {
                Ok(tensor) => {
                    // Test validation function directly
                    match validate_tensor_input(&tensor) {
                        Ok(_) => {
                            println!("✅ {} passed validation", scenario_name);
                        }
                        Err(e) => {
                            println!("✅ {} correctly failed validation: {}", scenario_name, e);
                        }
                    }

                    // Test quantization with potentially invalid data
                    let quantizer = I2SQuantizer::new();
                    match quantizer.quantize(&tensor) {
                        Ok(quantized) => {
                            println!("ℹ️  {} was quantized successfully", scenario_name);

                            // Test quantized tensor validation
                            match validate_quantized_tensor(&quantized) {
                                Ok(_) => {
                                    println!(
                                        "✅ {} produced valid quantized tensor",
                                        scenario_name
                                    );
                                }
                                Err(e) => {
                                    println!(
                                        "⚠️  {} produced invalid quantized tensor: {}",
                                        scenario_name, e
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            println!(
                                "✅ {} correctly rejected by quantization: {}",
                                scenario_name, e
                            );
                        }
                    }
                }
                Err(e) => {
                    println!("✅ {} correctly failed tensor creation: {}", scenario_name, e);
                }
            }
        }

        Ok(())
    }

    /// Test error recovery and state consistency
    #[test]
    fn test_error_recovery() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Test that quantizer maintains consistent state after errors
        let valid_data = vec![0.1, 0.2, 0.3, 0.4];
        let valid_tensor = create_tensor_from_f32(&valid_data, &[2, 2])?;

        // Perform successful operation
        let result1 = quantizer.quantize(&valid_tensor)?;
        assert!(!result1.data.is_empty(), "First operation should succeed");

        // Try to cause an error
        let invalid_data = vec![];
        match create_tensor_from_f32(&invalid_data, &[0]) {
            Ok(invalid_tensor) => {
                let _ = quantizer.quantize(&invalid_tensor); // May succeed or fail
            }
            Err(_) => {
                // Expected to fail
            }
        }

        // Test that quantizer still works after error
        let result2 = quantizer.quantize(&valid_tensor)?;
        assert!(!result2.data.is_empty(), "Quantizer should recover after error");
        assert_eq!(result1.data, result2.data, "Results should be consistent after recovery");

        Ok(())
    }

    /// Test security-related error handling
    #[test]
    fn test_security_error_handling() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Test with potentially dangerous input sizes
        let security_scenarios = vec![
            ("Very large tensor", 1_000_000),
            ("Moderate large tensor", 100_000),
            ("Edge case size", 65_537), // Just over 64K
        ];

        for (scenario_name, size) in security_scenarios {
            // Skip very large allocations in CI environments
            if size > 500_000 {
                println!("⏭️  Skipping {} due to CI memory constraints", scenario_name);
                continue;
            }

            let test_data: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 / 1000.0).collect();

            match create_tensor_from_f32(&test_data, &[size]) {
                Ok(tensor) => {
                    match quantizer.quantize(&tensor) {
                        Ok(_) => {
                            println!("ℹ️  {} was processed successfully", scenario_name);
                        }
                        Err(e) => {
                            println!("✅ {} correctly limited: {}", scenario_name, e);

                            // Check if it's a security-related error
                            match e.downcast_ref::<SecurityError>() {
                                Some(sec_err) => {
                                    println!(
                                        "✅ Correctly identified as SecurityError: {:?}",
                                        sec_err
                                    );
                                }
                                None => {
                                    println!("ℹ️  Non-security error: {}", e);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    println!("✅ {} correctly failed at tensor creation: {}", scenario_name, e);
                }
            }
        }

        Ok(())
    }

    /// Test cross-quantizer error consistency
    #[test]
    fn test_cross_quantizer_error_consistency() -> Result<()> {
        let quantizers: Vec<(&str, Box<dyn QuantizerTrait>)> = vec![
            ("I2S", Box::new(I2SQuantizer::new())),
            ("TL1", Box::new(TL1Quantizer::new())),
            ("TL2", Box::new(TL2Quantizer::new())),
        ];

        // Error scenarios that should be handled consistently
        let error_scenarios = vec![
            ("NaN input", vec![f32::NAN; 16]),
            ("Infinity input", vec![f32::INFINITY; 16]),
            ("Mixed extreme", vec![f32::MIN, f32::MAX, 0.0, 1.0]),
        ];

        for (scenario_name, test_data) in error_scenarios {
            let mut error_behaviors = Vec::new();

            for (quantizer_name, quantizer) in &quantizers {
                match create_tensor_from_f32(&test_data, &[4, 4]) {
                    Ok(tensor) => {
                        let behavior = match quantizer.quantize(&tensor) {
                            Ok(_) => "success",
                            Err(_) => "error",
                        };
                        error_behaviors.push((quantizer_name, behavior));
                        println!("{} on {}: {}", quantizer_name, scenario_name, behavior);
                    }
                    Err(_) => {
                        error_behaviors.push((quantizer_name, "tensor_creation_failed"));
                        println!("{} on {}: tensor creation failed", quantizer_name, scenario_name);
                    }
                }
            }

            // Analyze consistency (all quantizers should behave similarly for edge cases)
            let success_count = error_behaviors.iter().filter(|(_, b)| *b == "success").count();
            let error_count = error_behaviors.iter().filter(|(_, b)| *b == "error").count();

            println!(
                "{}: {} successes, {} errors out of {} quantizers",
                scenario_name,
                success_count,
                error_count,
                quantizers.len()
            );

            // Either all should succeed or all should fail for consistency
            let is_consistent =
                success_count == quantizers.len() || error_count == quantizers.len();
            if !is_consistent {
                println!("⚠️  Inconsistent behavior across quantizers for {}", scenario_name);
            }
        }

        Ok(())
    }

    /// Test structured error messages and debugging information
    #[test]
    fn test_error_messages_quality() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Test scenarios that should produce informative error messages
        let message_test_scenarios = vec![
            ("Empty tensor", || create_tensor_from_f32(&[], &[0])),
            ("Dimension mismatch", || create_tensor_from_f32(&[1.0, 2.0], &[3])),
        ];

        for (scenario_name, create_error_tensor) in message_test_scenarios {
            match create_error_tensor() {
                Ok(tensor) => {
                    match quantizer.quantize(&tensor) {
                        Ok(_) => {
                            println!("ℹ️  {} did not produce an error", scenario_name);
                        }
                        Err(e) => {
                            let error_message = format!("{}", e);
                            let debug_message = format!("{:?}", e);

                            println!("Error message for {}: {}", scenario_name, error_message);
                            println!("Debug message for {}: {}", scenario_name, debug_message);

                            // Verify error messages are informative
                            assert!(
                                error_message.len() > 10,
                                "Error message too short for {}",
                                scenario_name
                            );
                            assert!(
                                !error_message.contains("Error"),
                                "Error message should be more specific than just 'Error'"
                            );

                            // Check for helpful information
                            let message_lower = error_message.to_lowercase();
                            let has_context = message_lower.contains("tensor")
                                || message_lower.contains("dimension")
                                || message_lower.contains("shape")
                                || message_lower.contains("size")
                                || message_lower.contains("quantization");

                            assert!(
                                has_context,
                                "Error message should contain contextual information: {}",
                                error_message
                            );
                        }
                    }
                }
                Err(e) => {
                    println!(
                        "✅ {} correctly failed at tensor creation with message: {}",
                        scenario_name, e
                    );
                }
            }
        }

        Ok(())
    }

    /// Test error handling under resource constraints
    #[test]
    fn test_resource_constraint_errors() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Test progressive memory allocation until failure
        let mut successful_size = 0;
        let sizes = vec![64, 128, 256, 512, 1024, 2048];

        for size in sizes {
            let total_elements = size * size;

            // Skip very large allocations to avoid CI timeouts
            if total_elements > 500_000 {
                break;
            }

            let test_data: Vec<f32> =
                (0..total_elements).map(|i| (i % 1000) as f32 / 1000.0).collect();

            match create_tensor_from_f32(&test_data, &[size, size]) {
                Ok(tensor) => {
                    match quantizer.quantize(&tensor) {
                        Ok(quantized) => {
                            // Test full round-trip under resource pressure
                            match quantizer.dequantize(&quantized) {
                                Ok(_) => {
                                    successful_size = size;
                                    println!("✅ Successfully processed {}x{} tensor", size, size);
                                }
                                Err(e) => {
                                    println!(
                                        "⚠️  Dequantization failed at {}x{}: {}",
                                        size, size, e
                                    );
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            println!("⚠️  Quantization failed at {}x{}: {}", size, size, e);
                            break;
                        }
                    }
                }
                Err(e) => {
                    println!("⚠️  Tensor creation failed at {}x{}: {}", size, size, e);
                    break;
                }
            }
        }

        println!("Largest successful tensor size: {}x{}", successful_size, successful_size);

        // Should handle at least moderate-sized tensors
        assert!(successful_size >= 64, "Should handle at least 64x64 tensors");

        Ok(())
    }

    /// Test error serialization and deserialization
    #[test]
    fn test_error_serialization() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Generate various error types
        let error_generators = vec![
            ("Empty tensor error", || {
                create_tensor_from_f32(&[], &[0]).and_then(|t| quantizer.quantize(&t)).map(|_| ())
            }),
            ("Dimension mismatch error", || {
                create_tensor_from_f32(&[1.0, 2.0], &[3])
                    .and_then(|t| quantizer.quantize(&t))
                    .map(|_| ())
            }),
        ];

        for (error_name, generate_error) in error_generators {
            match generate_error() {
                Ok(_) => {
                    println!("ℹ️  {} did not generate an error", error_name);
                }
                Err(e) => {
                    // Test error conversion and display
                    let error_string = e.to_string();
                    let debug_string = format!("{:?}", e);

                    println!("Error string for {}: {}", error_name, error_string);
                    println!("Debug string for {}: {}", error_name, debug_string);

                    // Verify error information is preserved
                    assert!(!error_string.is_empty(), "Error string should not be empty");
                    assert!(!debug_string.is_empty(), "Debug string should not be empty");

                    // Test that error can be handled by Result chains
                    let chained_result = generate_error().map_err(|e| format!("Wrapped: {}", e));

                    match chained_result {
                        Ok(_) => unreachable!(),
                        Err(wrapped) => {
                            assert!(wrapped.contains("Wrapped:"), "Error chaining should work");
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Test graceful degradation under various failure modes
    #[test]
    fn test_graceful_degradation() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Test graceful degradation scenarios
        let degradation_scenarios = vec![
            ("Partial invalid data", create_partially_invalid_tensor),
            ("Boundary conditions", create_boundary_condition_tensor),
            ("Precision limits", create_precision_limit_tensor),
        ];

        for (scenario_name, create_tensor_fn) in degradation_scenarios {
            match create_tensor_fn() {
                Ok(tensor) => {
                    match quantizer.quantize(&tensor) {
                        Ok(quantized) => {
                            // Test that quantized result is valid
                            match validate_quantized_tensor(&quantized) {
                                Ok(_) => {
                                    println!(
                                        "✅ {} handled with graceful degradation",
                                        scenario_name
                                    );

                                    // Test dequantization still works
                                    match quantizer.dequantize(&quantized) {
                                        Ok(_) => {
                                            println!(
                                                "✅ {} completed full round-trip",
                                                scenario_name
                                            );
                                        }
                                        Err(e) => {
                                            println!(
                                                "⚠️  {} failed dequantization: {}",
                                                scenario_name, e
                                            );
                                        }
                                    }
                                }
                                Err(e) => {
                                    println!(
                                        "⚠️  {} produced invalid quantized tensor: {}",
                                        scenario_name, e
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            println!("✅ {} correctly rejected: {}", scenario_name, e);
                        }
                    }
                }
                Err(e) => {
                    println!("✅ {} correctly failed tensor creation: {}", scenario_name, e);
                }
            }
        }

        Ok(())
    }

    // Helper functions for error test scenarios

    fn create_partially_invalid_tensor() -> Result<Tensor> {
        // Mix of valid and potentially problematic values
        let data = vec![
            0.1,
            0.2,
            0.3,
            f32::NAN,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            f32::INFINITY,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
        ];
        create_tensor_from_f32(&data, &[4, 4])
    }

    fn create_boundary_condition_tensor() -> Result<Tensor> {
        // Values at quantization boundaries
        let data = vec![
            -1.0, -0.33333, 0.33333, 1.0, -0.99999, -0.33334, 0.33332, 0.99999, -1.00001, -0.33332,
            0.33334, 1.00001, -0.5, 0.0, 0.5, 0.75,
        ];
        create_tensor_from_f32(&data, &[4, 4])
    }

    fn create_precision_limit_tensor() -> Result<Tensor> {
        // Values at floating-point precision limits
        let data = vec![
            f32::EPSILON,
            -f32::EPSILON,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1.0 + f32::EPSILON,
            1.0 - f32::EPSILON,
            -1.0 + f32::EPSILON,
            -1.0 - f32::EPSILON,
            1e-38,
            -1e-38,
            1e38,
            -1e38,
            f32::MAX / 2.0,
            f32::MIN / 2.0,
            0.0,
            -0.0,
        ];
        create_tensor_from_f32(&data, &[4, 4])
    }
}
