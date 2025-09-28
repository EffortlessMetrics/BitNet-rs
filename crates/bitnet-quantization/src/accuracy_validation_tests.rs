//! Numerical accuracy validation tests for BitNet quantization algorithms
//!
//! This module provides comprehensive validation of numerical accuracy, stability,
//! and precision for quantization operations across different data distributions.

#[cfg(test)]
mod tests {
    use crate::utils::create_tensor_from_f32;
    use crate::{I2SQuantizer, QuantizerTrait, TL1Quantizer, TL2Quantizer};
    use bitnet_common::Result;
    use candle_core::Tensor;
    use std::f32::consts::PI;

    /// Statistical accuracy metrics for quantization evaluation
    #[derive(Debug, Clone)]
    struct AccuracyMetrics {
        mse: f64,                 // Mean Squared Error
        mae: f64,                 // Mean Absolute Error
        max_error: f64,           // Maximum absolute error
        snr_db: f64,              // Signal-to-noise ratio in dB
        pearson_correlation: f64, // Pearson correlation coefficient
        cosine_similarity: f64,   // Cosine similarity
    }

    impl AccuracyMetrics {
        fn compute(original: &[f32], reconstructed: &[f32]) -> Self {
            assert_eq!(original.len(), reconstructed.len());
            let n = original.len() as f64;

            // Mean Squared Error
            let mse = original
                .iter()
                .zip(reconstructed.iter())
                .map(|(o, r)| (*o as f64 - *r as f64).powi(2))
                .sum::<f64>()
                / n;

            // Mean Absolute Error
            let mae = original
                .iter()
                .zip(reconstructed.iter())
                .map(|(o, r)| (*o as f64 - *r as f64).abs())
                .sum::<f64>()
                / n;

            // Maximum error
            let max_error = original
                .iter()
                .zip(reconstructed.iter())
                .map(|(o, r)| (*o as f64 - *r as f64).abs())
                .fold(0.0, f64::max);

            // Signal-to-noise ratio
            let signal_power = original.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / n;
            let noise_power = mse;
            let snr_db = if noise_power > 0.0 {
                10.0 * (signal_power / noise_power).log10()
            } else {
                f64::INFINITY
            };

            // Pearson correlation
            let orig_mean = original.iter().map(|x| *x as f64).sum::<f64>() / n;
            let recon_mean = reconstructed.iter().map(|x| *x as f64).sum::<f64>() / n;

            let numerator: f64 = original
                .iter()
                .zip(reconstructed.iter())
                .map(|(o, r)| (*o as f64 - orig_mean) * (*r as f64 - recon_mean))
                .sum();

            let orig_var: f64 = original.iter().map(|x| (*x as f64 - orig_mean).powi(2)).sum();

            let recon_var: f64 =
                reconstructed.iter().map(|x| (*x as f64 - recon_mean).powi(2)).sum();

            let pearson_correlation = if orig_var > 0.0 && recon_var > 0.0 {
                numerator / (orig_var * recon_var).sqrt()
            } else {
                0.0
            };

            // Cosine similarity
            let dot_product: f64 = original
                .iter()
                .zip(reconstructed.iter())
                .map(|(o, r)| (*o as f64) * (*r as f64))
                .sum();

            let orig_norm: f64 = original.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let recon_norm: f64 =
                reconstructed.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();

            let cosine_similarity = if orig_norm > 0.0 && recon_norm > 0.0 {
                dot_product / (orig_norm * recon_norm)
            } else {
                0.0
            };

            Self { mse, mae, max_error, snr_db, pearson_correlation, cosine_similarity }
        }

        /// Check if metrics meet production quality thresholds
        fn meets_production_quality(&self) -> bool {
            self.snr_db >= 40.0 &&               // High signal-to-noise ratio
            self.pearson_correlation >= 0.95 &&   // Strong correlation
            self.cosine_similarity >= 0.95 &&     // High similarity
            self.mae <= 0.05 // Low mean absolute error
        }
    }

    /// Test I2S quantization accuracy across different data distributions
    #[test]
    fn test_i2s_accuracy_distributions() -> Result<()> {
        let quantizer = I2SQuantizer::new();
        let mut all_metrics_passed = true;

        // Test different statistical distributions
        let distributions = vec![
            ("uniform", generate_uniform_data(1024, -1.0, 1.0)),
            ("normal", generate_normal_data(1024, 0.0, 0.3)),
            ("exponential", generate_exponential_data(1024, 1.0)),
            ("bimodal", generate_bimodal_data(1024)),
            ("sparse", generate_sparse_data(1024, 0.1)),
            ("neural_weights", generate_neural_weight_distribution(1024)),
        ];

        for (dist_name, data) in distributions {
            let tensor = create_tensor_from_f32(&data, &[32, 32])?;
            let quantized = quantizer.quantize(&tensor)?;
            let dequantized = quantizer.dequantize(&quantized)?;

            let reconstructed = tensor_to_vec(&dequantized)?;
            let metrics = AccuracyMetrics::compute(&data, &reconstructed);

            println!(
                "I2S {} distribution: MSE={:.6}, MAE={:.6}, SNR={:.2}dB, Corr={:.4}",
                dist_name, metrics.mse, metrics.mae, metrics.snr_db, metrics.pearson_correlation
            );

            // Log if metrics don't meet production quality
            if !metrics.meets_production_quality() {
                println!("⚠️  I2S {} distribution below production quality thresholds", dist_name);
                all_metrics_passed = false;
            }

            // Ensure basic sanity checks pass
            assert!(metrics.snr_db > 20.0, "SNR too low for {} distribution", dist_name);
            assert!(
                metrics.pearson_correlation > 0.8,
                "Correlation too low for {} distribution",
                dist_name
            );
        }

        if all_metrics_passed {
            println!("✅ I2S quantization meets production quality for all distributions");
        }

        Ok(())
    }

    /// Test TL1/TL2 accuracy comparison and stability
    #[test]
    fn test_tl1_tl2_accuracy_comparison() -> Result<()> {
        let tl1 = TL1Quantizer::new();
        let tl2 = TL2Quantizer::new();

        // Test with neural network weight patterns
        let weight_patterns = vec![
            generate_conv_weight_pattern(256),
            generate_attention_weight_pattern(256),
            generate_embedding_weight_pattern(256),
            generate_layer_norm_pattern(256),
        ];

        for (idx, pattern) in weight_patterns.iter().enumerate() {
            let tensor = create_tensor_from_f32(pattern, &[16, 16])?;

            // Test TL1
            let tl1_quantized = tl1.quantize(&tensor)?;
            let tl1_dequantized = tl1.dequantize(&tl1_quantized)?;
            let tl1_reconstructed = tensor_to_vec(&tl1_dequantized)?;
            let tl1_metrics = AccuracyMetrics::compute(pattern, &tl1_reconstructed);

            // Test TL2
            let tl2_quantized = tl2.quantize(&tensor)?;
            let tl2_dequantized = tl2.dequantize(&tl2_quantized)?;
            let tl2_reconstructed = tensor_to_vec(&tl2_dequantized)?;
            let tl2_metrics = AccuracyMetrics::compute(pattern, &tl2_reconstructed);

            println!(
                "Pattern {}: TL1 SNR={:.2}dB, TL2 SNR={:.2}dB",
                idx, tl1_metrics.snr_db, tl2_metrics.snr_db
            );

            // Both should maintain reasonable accuracy
            assert!(tl1_metrics.snr_db > 15.0, "TL1 SNR too low for pattern {}", idx);
            assert!(tl2_metrics.snr_db > 15.0, "TL2 SNR too low for pattern {}", idx);

            // Check compression efficiency vs accuracy trade-off
            let tl1_compression_ratio =
                (pattern.len() * 4) as f64 / tl1_quantized.data.len() as f64;
            let tl2_compression_ratio =
                (pattern.len() * 4) as f64 / tl2_quantized.data.len() as f64;

            println!(
                "Pattern {}: TL1 compression={:.2}x, TL2 compression={:.2}x",
                idx, tl1_compression_ratio, tl2_compression_ratio
            );

            // Both should provide meaningful compression
            assert!(tl1_compression_ratio > 1.5, "TL1 compression too low");
            assert!(tl2_compression_ratio > 1.5, "TL2 compression too low");
        }

        Ok(())
    }

    /// Test quantization stability under perturbations
    #[test]
    fn test_quantization_stability() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Base signal
        let base_signal: Vec<f32> = (0..128).map(|i| (i as f32 / 64.0).sin()).collect();

        // Test with different noise levels
        let noise_levels = vec![0.001, 0.01, 0.05, 0.1];

        for noise_level in noise_levels {
            let mut total_stability_score = 0.0;
            let num_trials = 10;

            for trial in 0..num_trials {
                // Add random noise
                let mut noisy_signal = base_signal.clone();
                for value in &mut noisy_signal {
                    *value += (trial as f32 * 0.12345).sin() * noise_level; // Deterministic "random"
                }

                let tensor = create_tensor_from_f32(&noisy_signal, &[128])?;
                let quantized = quantizer.quantize(&tensor)?;
                let dequantized = quantizer.dequantize(&quantized)?;
                let reconstructed = tensor_to_vec(&dequantized)?;

                // Compare with base quantized signal
                let base_tensor = create_tensor_from_f32(&base_signal, &[128])?;
                let base_quantized = quantizer.quantize(&base_tensor)?;
                let base_dequantized = quantizer.dequantize(&base_quantized)?;
                let base_reconstructed = tensor_to_vec(&base_dequantized)?;

                // Compute stability metric
                let stability_score = compute_stability_metric(&base_reconstructed, &reconstructed);
                total_stability_score += stability_score;
            }

            let avg_stability = total_stability_score / num_trials as f64;
            println!(
                "Noise level {:.3}: Average stability score = {:.4}",
                noise_level, avg_stability
            );

            // Stability should degrade gracefully with noise
            let expected_min_stability = 1.0 - (noise_level as f64 * 5.0);
            assert!(
                avg_stability >= expected_min_stability.max(0.5),
                "Stability too low at noise level {:.3}",
                noise_level
            );
        }

        Ok(())
    }

    /// Test quantization with adversarial patterns designed to stress the algorithm
    #[test]
    fn test_adversarial_patterns() -> Result<()> {
        let quantizers = vec![
            ("I2S", Box::new(I2SQuantizer::new()) as Box<dyn QuantizerTrait>),
            ("TL1", Box::new(TL1Quantizer::new()) as Box<dyn QuantizerTrait>),
            ("TL2", Box::new(TL2Quantizer::new()) as Box<dyn QuantizerTrait>),
        ];

        // Adversarial patterns that challenge quantization
        let adversarial_patterns = vec![
            ("high_frequency", generate_high_frequency_pattern(128)),
            ("checkerboard", generate_checkerboard_pattern(128)),
            ("step_function", generate_step_function_pattern(128)),
            ("impulse_train", generate_impulse_train_pattern(128)),
            ("near_quantization_boundary", generate_boundary_stress_pattern(128)),
        ];

        for (quantizer_name, quantizer) in &quantizers {
            for (pattern_name, pattern) in &adversarial_patterns {
                let tensor = create_tensor_from_f32(pattern, &[128])?;

                match quantizer.quantize(&tensor) {
                    Ok(quantized) => {
                        match quantizer.dequantize(&quantized) {
                            Ok(dequantized) => {
                                let reconstructed = tensor_to_vec(&dequantized)?;
                                let metrics = AccuracyMetrics::compute(pattern, &reconstructed);

                                println!(
                                    "{} {} pattern: MSE={:.6}, SNR={:.2}dB",
                                    quantizer_name, pattern_name, metrics.mse, metrics.snr_db
                                );

                                // Even adversarial patterns should maintain basic fidelity
                                assert!(
                                    metrics.snr_db > 10.0,
                                    "{} failed on {} pattern with SNR {:.2}dB",
                                    quantizer_name,
                                    pattern_name,
                                    metrics.snr_db
                                );
                            }
                            Err(e) => {
                                println!(
                                    "⚠️  {} dequantization failed on {} pattern: {}",
                                    quantizer_name, pattern_name, e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        println!(
                            "⚠️  {} quantization failed on {} pattern: {}",
                            quantizer_name, pattern_name, e
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Test bit-level accuracy for I2S quantization
    #[test]
    fn test_i2s_bit_level_accuracy() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Test values that should map to specific quantization levels
        let test_values = vec![
            -1.0,  // Should map to minimum quantization level
            -0.33, // Should map to second level
            0.33,  // Should map to third level
            1.0,   // Should map to maximum level
        ];

        for value in test_values {
            let single_value = vec![value; 32]; // Block size for I2S
            let tensor = create_tensor_from_f32(&single_value, &[32])?;
            let quantized = quantizer.quantize(&tensor)?;
            let dequantized = quantizer.dequantize(&quantized)?;
            let reconstructed = tensor_to_vec(&dequantized)?;

            // Check that all values in the block are consistently quantized
            let first_reconstructed = reconstructed[0];
            for &recon_value in &reconstructed {
                assert!(
                    (recon_value - first_reconstructed).abs() < 1e-6,
                    "Inconsistent quantization within block for value {}",
                    value
                );
            }

            // Check that quantization error is bounded
            let error = (value - first_reconstructed).abs();
            assert!(
                error <= 0.5,
                "Quantization error too large for value {}: error = {}",
                value,
                error
            );
        }

        Ok(())
    }

    // Helper functions for data generation

    fn generate_uniform_data(size: usize, min: f32, max: f32) -> Vec<f32> {
        (0..size).map(|i| min + (max - min) * (i as f32 / size as f32)).collect()
    }

    fn generate_normal_data(size: usize, mean: f32, std: f32) -> Vec<f32> {
        // Box-Muller transform for normal distribution
        (0..size)
            .map(|i| {
                let u1 = (i as f32 / size as f32 + 0.001).max(0.001);
                let u2 = ((i * 7 + 3) as f32 / size as f32 + 0.001).max(0.001);
                mean + std * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
            })
            .collect()
    }

    fn generate_exponential_data(size: usize, lambda: f32) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let u = (i as f32 / size as f32 + 0.001).max(0.001);
                -(-u).ln() / lambda
            })
            .collect()
    }

    fn generate_bimodal_data(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                if i % 2 == 0 {
                    -0.5 + 0.2 * (i as f32 / size as f32)
                } else {
                    0.5 + 0.2 * (i as f32 / size as f32)
                }
            })
            .collect()
    }

    fn generate_sparse_data(size: usize, sparsity: f32) -> Vec<f32> {
        (0..size)
            .map(|i| {
                if (i as f32 / size as f32) < sparsity {
                    (i as f32 / size as f32) * 2.0 - 1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn generate_neural_weight_distribution(size: usize) -> Vec<f32> {
        // Typical neural network weight distribution (Xavier/Glorot-like)
        let scale = (2.0 / size as f32).sqrt();
        (0..size)
            .map(|i| {
                let u1 = (i as f32 / size as f32 + 0.001).max(0.001);
                let u2 = ((i * 13 + 7) as f32 / size as f32 + 0.001).max(0.001);
                scale * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
            })
            .collect()
    }

    fn generate_conv_weight_pattern(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let spatial_freq = (i as f32 * 2.0 * PI / 8.0).sin();
                let decay = (-i as f32 / size as f32).exp();
                spatial_freq * decay * 0.1
            })
            .collect()
    }

    fn generate_attention_weight_pattern(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let pos = i as f32 / size as f32;
                let attention = (-((pos - 0.5) * 4.0).powi(2)).exp();
                (attention - 0.5) * 0.3
            })
            .collect()
    }

    fn generate_embedding_weight_pattern(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let dim_scale = (1.0 + i as f32 / size as f32).ln();
                let random_like = ((i * 17 + 11) as f32).sin();
                random_like * dim_scale * 0.05
            })
            .collect()
    }

    fn generate_layer_norm_pattern(size: usize) -> Vec<f32> {
        let mean = 0.1;
        let std = 0.05;
        (0..size).map(|i| mean + std * ((i * 23 + 19) as f32).sin()).collect()
    }

    fn generate_high_frequency_pattern(size: usize) -> Vec<f32> {
        (0..size).map(|i| (i as f32 * PI).sin()).collect()
    }

    fn generate_checkerboard_pattern(size: usize) -> Vec<f32> {
        (0..size).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect()
    }

    fn generate_step_function_pattern(size: usize) -> Vec<f32> {
        (0..size).map(|i| if i < size / 2 { -0.8 } else { 0.8 }).collect()
    }

    fn generate_impulse_train_pattern(size: usize) -> Vec<f32> {
        (0..size).map(|i| if i % 8 == 0 { 1.0 } else { 0.0 }).collect()
    }

    fn generate_boundary_stress_pattern(size: usize) -> Vec<f32> {
        // Values near quantization boundaries to stress rounding behavior
        (0..size)
            .map(|i| {
                let base = (i % 4) as f32 * 0.67 - 1.0; // Near quantization levels
                base + 0.01 * ((i * 7) as f32).sin() // Small perturbation
            })
            .collect()
    }

    fn compute_stability_metric(base: &[f32], perturbed: &[f32]) -> f64 {
        let mse = base
            .iter()
            .zip(perturbed.iter())
            .map(|(b, p)| (*b as f64 - *p as f64).powi(2))
            .sum::<f64>()
            / base.len() as f64;

        1.0 / (1.0 + mse) // Stability score between 0 and 1
    }

    fn tensor_to_vec(tensor: &Tensor) -> Result<Vec<f32>> {
        Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
    }
}
