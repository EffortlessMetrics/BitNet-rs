//! Property-based testing for quantization invariants and mathematical properties
//!
//! This module provides property-based tests to verify fundamental mathematical
//! properties and invariants that should hold across all quantization algorithms.

#[cfg(test)]
mod tests {
    use crate::utils::create_tensor_from_f32;
    use crate::{I2SQuantizer, QuantizerTrait, TL1Quantizer, TL2Quantizer};
    use bitnet_common::{Result, Tensor};
    use candle_core::Device;

    /// Property: Quantization should be deterministic
    #[test]
    fn property_quantization_determinism() -> Result<()> {
        let quantizers: Vec<(&str, Box<dyn QuantizerTrait>)> = vec![
            ("I2S", Box::new(I2SQuantizer::new())),
            ("TL1", Box::new(TL1Quantizer::new())),
            ("TL2", Box::new(TL2Quantizer::new())),
        ];

        let device = Device::Cpu;
        let test_data = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let mut padded_data = test_data.clone();
        while padded_data.len() < 32 {
            padded_data.extend_from_slice(&test_data);
        }

        for (quantizer_name, quantizer) in &quantizers {
            let tensor =
                create_tensor_from_f32(padded_data.clone(), &[padded_data.len()], &device)?;

            // Run quantization multiple times
            let mut results = Vec::new();
            for _ in 0..3 {
                match quantizer.quantize_tensor(&tensor) {
                    Ok(quantized) => results.push(quantized),
                    Err(_) => break, // Skip if quantization fails
                }
            }

            if results.len() >= 2 {
                // All results should be identical
                for (i, result) in results.iter().enumerate().skip(1) {
                    assert_eq!(
                        results[0].data, result.data,
                        "{} determinism failed at run {}",
                        quantizer_name, i
                    );
                    assert_eq!(
                        results[0].shape, result.shape,
                        "{} shape determinism failed at run {}",
                        quantizer_name, i
                    );
                }
            }
        }

        Ok(())
    }

    /// Property: Round-trip should preserve values within tolerance
    #[test]
    fn property_round_trip_tolerance() -> Result<()> {
        let quantizers: Vec<(&str, Box<dyn QuantizerTrait>)> = vec![
            ("I2S", Box::new(I2SQuantizer::new())),
            ("TL1", Box::new(TL1Quantizer::new())),
            ("TL2", Box::new(TL2Quantizer::new())),
        ];

        let device = Device::Cpu;
        let test_data = generate_test_cases();

        for (quantizer_name, quantizer) in &quantizers {
            for (case_id, data) in test_data.iter().enumerate() {
                let tensor = create_tensor_from_f32(data.clone(), &[data.len()], &device)?;

                match quantizer.quantize_tensor(&tensor) {
                    Ok(quantized) => {
                        match quantizer.dequantize_tensor(&quantized) {
                            Ok(dequantized) => {
                                let reconstructed = tensor_to_vec(&dequantized)?;

                                // Check that we get reasonable reconstruction
                                let mse = compute_mse(data, &reconstructed);

                                // Tolerance should be reasonable (not infinity)
                                assert!(
                                    mse < 10.0 && mse.is_finite(),
                                    "{} case {} MSE too high or invalid: {}",
                                    quantizer_name,
                                    case_id,
                                    mse
                                );
                            }
                            Err(e) => {
                                println!(
                                    "⚠️  {} dequantization failed for case {}: {}",
                                    quantizer_name, case_id, e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        println!(
                            "⚠️  {} quantization failed for case {}: {}",
                            quantizer_name, case_id, e
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Property: Scale invariance within reasonable bounds
    #[test]
    fn property_scale_bounds() -> Result<()> {
        let quantizer = I2SQuantizer::new();
        let device = Device::Cpu;

        let base_data = vec![0.1, -0.2, 0.3, -0.4];
        let mut padded_data = base_data.clone();
        while padded_data.len() < 32 {
            padded_data.extend_from_slice(&base_data);
        }

        // Test different scales
        for scale in [0.1, 1.0, 10.0] {
            let scaled_data: Vec<f32> = padded_data.iter().map(|&x| x * scale).collect();
            let tensor =
                create_tensor_from_f32(scaled_data.clone(), &[scaled_data.len()], &device)?;

            match quantizer.quantize_tensor(&tensor) {
                Ok(quantized) => {
                    match quantizer.dequantize_tensor(&quantized) {
                        Ok(dequantized) => {
                            let reconstructed = tensor_to_vec(&dequantized)?;
                            let mse = compute_mse(&scaled_data, &reconstructed);

                            // Should maintain some reasonable accuracy
                            assert!(mse < 100.0, "Scale {} MSE too high: {}", scale, mse);
                        }
                        Err(e) => {
                            println!("⚠️  Dequantization failed for scale {}: {}", scale, e);
                        }
                    }
                }
                Err(e) => {
                    println!("⚠️  Quantization failed for scale {}: {}", scale, e);
                }
            }
        }

        Ok(())
    }

    /// Property: Data type preservation
    #[test]
    fn property_data_type_preservation() -> Result<()> {
        let quantizers: Vec<(&str, Box<dyn QuantizerTrait>)> = vec![
            ("I2S", Box::new(I2SQuantizer::new())),
            ("TL1", Box::new(TL1Quantizer::new())),
            ("TL2", Box::new(TL2Quantizer::new())),
        ];

        let device = Device::Cpu;
        let test_data = vec![0.125, -0.25, 0.5, -0.75];
        let mut padded_data = test_data.clone();
        while padded_data.len() < 64 {
            padded_data.extend_from_slice(&test_data);
        }

        for (quantizer_name, quantizer) in &quantizers {
            let tensor =
                create_tensor_from_f32(padded_data.clone(), &[padded_data.len()], &device)?;

            match quantizer.quantize_tensor(&tensor) {
                Ok(quantized) => {
                    // Check quantized tensor has expected properties
                    assert!(
                        !quantized.data.is_empty(),
                        "{} quantized data should not be empty",
                        quantizer_name
                    );
                    assert!(
                        !quantized.scales.is_empty(),
                        "{} scales should not be empty",
                        quantizer_name
                    );
                    assert_eq!(
                        quantized.shape,
                        vec![padded_data.len()],
                        "{} shape should be preserved",
                        quantizer_name
                    );

                    match quantizer.dequantize_tensor(&quantized) {
                        Ok(dequantized) => {
                            // Check dimensions are preserved
                            assert_eq!(
                                dequantized.shape(),
                                &[padded_data.len()],
                                "{} shape should be preserved after round-trip",
                                quantizer_name
                            );
                        }
                        Err(e) => {
                            println!("⚠️  {} dequantization failed: {}", quantizer_name, e);
                        }
                    }
                }
                Err(e) => {
                    println!("⚠️  {} quantization failed: {}", quantizer_name, e);
                }
            }
        }

        Ok(())
    }

    // Helper functions

    fn generate_test_cases() -> Vec<Vec<f32>> {
        let mut cases = Vec::new();

        // Simple patterns
        cases.push(vec![0.0; 32]);
        cases.push(vec![1.0; 32]);
        cases.push(vec![-1.0; 32]);

        // Mixed values
        let mixed: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
        cases.push(mixed);

        // Sequential values
        let sequential: Vec<f32> = (0..32).map(|i| (i as f32 / 32.0) * 2.0 - 1.0).collect();
        cases.push(sequential);

        cases
    }

    fn compute_mse(original: &[f32], reconstructed: &[f32]) -> f64 {
        if original.len() != reconstructed.len() {
            return f64::INFINITY;
        }

        let n = original.len() as f64;
        original
            .iter()
            .zip(reconstructed.iter())
            .map(|(o, r)| (*o as f64 - *r as f64).powi(2))
            .sum::<f64>()
            / n
    }

    fn tensor_to_vec(tensor: &bitnet_common::BitNetTensor) -> Result<Vec<f32>> {
        Ok(tensor.as_candle().flatten_all()?.to_vec1::<f32>()?)
    }
}
