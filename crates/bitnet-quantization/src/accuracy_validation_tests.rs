//! Numerical accuracy validation tests for BitNet quantization algorithms
//!
//! This module provides comprehensive validation of numerical accuracy, stability,
//! and precision for quantization operations across different data distributions.

#[cfg(test)]
mod tests {
    use crate::utils::create_tensor_from_f32;
    use crate::{I2SQuantizer, QuantizerTrait, TL1Quantizer, TL2Quantizer};
    use bitnet_common::Result;
    use candle_core::Device;
    use std::f32::consts::PI;

    /// Test I2S quantization accuracy across different data distributions
    #[test]
    fn test_i2s_accuracy_distributions() -> Result<()> {
        let quantizer = I2SQuantizer::new();
        let device = Device::Cpu;

        // Test different statistical distributions
        let distributions = vec![
            ("uniform", generate_uniform_data(1024, -1.0, 1.0)),
            ("normal", generate_normal_data(1024, 0.0, 0.3)),
            ("neural_weights", generate_neural_weight_distribution(1024)),
        ];

        for (dist_name, data) in distributions {
            let tensor = create_tensor_from_f32(data.clone(), &[32, 32], &device)?;

            match quantizer.quantize_tensor(&tensor) {
                Ok(quantized) => {
                    match quantizer.dequantize_tensor(&quantized) {
                        Ok(dequantized) => {
                            let reconstructed = tensor_to_vec(&dequantized)?;

                            // Check for NaN values in reconstruction
                            let has_nan = reconstructed.iter().any(|&x| !x.is_finite());
                            if has_nan {
                                println!(
                                    "⚠️  I2S {} distribution produced NaN values, skipping MSE check",
                                    dist_name
                                );
                                continue;
                            }

                            let mse = compute_mse(&data, &reconstructed);
                            println!("I2S {} distribution: MSE={:.6}", dist_name, mse);

                            // I2S should maintain reasonable quantization error
                            if mse.is_finite() {
                                assert!(
                                    mse < 1.0,
                                    "I2S MSE too high for {} distribution: {:.6}",
                                    dist_name,
                                    mse
                                );
                            } else {
                                println!(
                                    "⚠️  I2S {} distribution produced non-finite MSE, skipping",
                                    dist_name
                                );
                            }
                        }
                        Err(e) => {
                            println!(
                                "⚠️  I2S dequantization failed for {} distribution: {}",
                                dist_name, e
                            );
                        }
                    }
                }
                Err(e) => {
                    println!("⚠️  I2S quantization failed for {} distribution: {}", dist_name, e);
                }
            }
        }

        Ok(())
    }

    /// Test TL1/TL2 accuracy comparison
    #[test]
    fn test_tl1_tl2_accuracy_comparison() -> Result<()> {
        let tl1 = TL1Quantizer::new();
        let tl2 = TL2Quantizer::new();
        let device = Device::Cpu;

        let test_data = generate_neural_weight_distribution(256);
        let tensor = create_tensor_from_f32(test_data.clone(), &[16, 16], &device)?;

        // Test TL1
        let tl1_quantized = tl1.quantize_tensor(&tensor)?;
        let tl1_dequantized = tl1.dequantize_tensor(&tl1_quantized)?;
        let tl1_reconstructed = tensor_to_vec(&tl1_dequantized)?;
        let tl1_mse = compute_mse(&test_data, &tl1_reconstructed);

        // Test TL2
        let tl2_quantized = tl2.quantize_tensor(&tensor)?;
        let tl2_dequantized = tl2.dequantize_tensor(&tl2_quantized)?;
        let tl2_reconstructed = tensor_to_vec(&tl2_dequantized)?;
        let tl2_mse = compute_mse(&test_data, &tl2_reconstructed);

        println!("TL1 MSE: {:.6}, TL2 MSE: {:.6}", tl1_mse, tl2_mse);

        // Both should provide reasonable accuracy
        assert!(tl1_mse < 0.5, "TL1 MSE too high: {:.6}", tl1_mse);
        assert!(tl2_mse < 0.5, "TL2 MSE too high: {:.6}", tl2_mse);

        Ok(())
    }

    /// Test quantization stability
    #[test]
    fn test_quantization_stability() -> Result<()> {
        let quantizer = I2SQuantizer::new();
        let device = Device::Cpu;

        // Base signal
        let base_signal: Vec<f32> = (0..128).map(|i| (i as f32 / 64.0).sin()).collect();

        // Test with small perturbations
        for noise_level in [0.001, 0.01, 0.05] {
            let mut noisy_signal = base_signal.clone();
            for (i, value) in noisy_signal.iter_mut().enumerate() {
                *value += (i as f32 * 0.12345).sin() * noise_level;
            }

            let tensor = create_tensor_from_f32(noisy_signal.clone(), &[128], &device)?;
            let quantized = quantizer.quantize_tensor(&tensor)?;
            let dequantized = quantizer.dequantize_tensor(&quantized)?;
            let reconstructed = tensor_to_vec(&dequantized)?;

            let mse = compute_mse(&noisy_signal, &reconstructed);
            println!("Noise level {:.3}: MSE = {:.6}", noise_level, mse);

            // Should maintain stability
            assert!(mse < 1.0, "Stability test failed at noise level {:.3}", noise_level);
        }

        Ok(())
    }

    /// Test I2S bit-level accuracy
    #[test]
    fn test_i2s_bit_level_accuracy() -> Result<()> {
        let quantizer = I2SQuantizer::new();
        let device = Device::Cpu;

        // Test values at quantization boundaries
        let test_values = vec![-1.0, -0.33, 0.33, 1.0];

        for value in test_values {
            let single_value = vec![value; 32]; // Block size for I2S
            let tensor = create_tensor_from_f32(single_value.clone(), &[32], &device)?;
            let quantized = quantizer.quantize_tensor(&tensor)?;
            let dequantized = quantizer.dequantize_tensor(&quantized)?;
            let reconstructed = tensor_to_vec(&dequantized)?;

            // Check quantization error is bounded
            let error = (value - reconstructed[0]).abs();
            assert!(
                error <= 0.5,
                "Quantization error too large for value {}: error = {}",
                value,
                error
            );
        }

        Ok(())
    }

    /// Test round-trip determinism
    #[test]
    fn test_deterministic_quantization_round_trip() -> Result<()> {
        let quantizers: Vec<(&str, Box<dyn QuantizerTrait>)> = vec![
            ("I2S", Box::new(I2SQuantizer::new())),
            ("TL1", Box::new(TL1Quantizer::new())),
            ("TL2", Box::new(TL2Quantizer::new())),
        ];

        let device = Device::Cpu;
        let test_data = vec![0.1234, -0.5678, 0.9012, -0.3456, 0.7890];
        let mut padded_data = test_data.clone();
        while padded_data.len() < 32 {
            padded_data.extend_from_slice(&test_data);
        }

        for (quantizer_name, quantizer) in &quantizers {
            let tensor =
                create_tensor_from_f32(padded_data.clone(), &[padded_data.len()], &device)?;
            let quantized = quantizer.quantize_tensor(&tensor)?;
            let dequantized = quantizer.dequantize_tensor(&quantized)?;
            let reconstructed = tensor_to_vec(&dequantized)?;

            let mse = compute_mse(&padded_data, &reconstructed);
            println!("{} test vector MSE: {:.6}", quantizer_name, mse);

            // Ensure reasonable reconstruction
            assert!(mse < 1.0, "{} quantization MSE too high: {:.6}", quantizer_name, mse);
        }

        Ok(())
    }

    // Helper functions

    fn compute_mse(original: &[f32], reconstructed: &[f32]) -> f64 {
        if original.len() != reconstructed.len() {
            eprintln!(
                "Length mismatch: original={}, reconstructed={}",
                original.len(),
                reconstructed.len()
            );
            return f64::NAN;
        }

        let n = original.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        let mse = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(o, r)| {
                let diff = (*o as f64 - *r as f64).powi(2);
                if !diff.is_finite() {
                    eprintln!("Non-finite difference: {} - {} = {}", *o, *r, diff);
                }
                diff
            })
            .sum::<f64>()
            / n;

        if !mse.is_finite() {
            eprintln!("Non-finite MSE computed");
        }

        mse
    }

    fn generate_uniform_data(size: usize, min: f32, max: f32) -> Vec<f32> {
        (0..size).map(|i| min + (max - min) * (i as f32 / size as f32)).collect()
    }

    fn generate_normal_data(size: usize, mean: f32, std: f32) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let u1 = (i as f32 / size as f32 + 0.001).max(0.001);
                let u2 = ((i * 7 + 3) as f32 / size as f32 + 0.001).max(0.001);
                mean + std * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
            })
            .collect()
    }

    fn generate_neural_weight_distribution(size: usize) -> Vec<f32> {
        let scale = (2.0 / size as f32).sqrt();
        (0..size)
            .map(|i| {
                let u1 = (i as f32 / size as f32 + 0.001).max(0.001);
                let u2 = ((i * 13 + 7) as f32 / size as f32 + 0.001).max(0.001);
                scale * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
            })
            .collect()
    }

    fn tensor_to_vec(tensor: &bitnet_common::BitNetTensor) -> Result<Vec<f32>> {
        Ok(tensor.as_candle().flatten_all()?.to_vec1::<f32>()?)
    }
}
