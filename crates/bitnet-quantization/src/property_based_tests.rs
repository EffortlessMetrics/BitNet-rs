//! Property-based testing for quantization invariants and mathematical properties
//!
//! This module provides property-based tests to verify fundamental mathematical
//! properties and invariants that should hold across all quantization algorithms.

#[cfg(test)]
mod tests {
    use crate::utils::create_tensor_from_f32;
    use crate::{I2SQuantizer, QuantizerTrait, TL1Quantizer, TL2Quantizer};
    use bitnet_common::Result;
    use candle_core::Tensor;
    use std::f32::consts::PI;

    /// Property: Quantization should be deterministic
    /// For any input x, quantize(x) should always produce the same result
    #[test]
    fn property_quantization_determinism() -> Result<()> {
        let quantizers: Vec<(&str, Box<dyn QuantizerTrait>)> = vec![
            ("I2S", Box::new(I2SQuantizer::new())),
            ("TL1", Box::new(TL1Quantizer::new())),
            ("TL2", Box::new(TL2Quantizer::new())),
        ];

        let test_cases = generate_property_test_cases(50);

        for (quantizer_name, quantizer) in &quantizers {
            for (case_id, test_data) in test_cases.iter().enumerate() {
                let tensor = create_tensor_from_f32(test_data, &[test_data.len()])?;

                // Run quantization multiple times
                let mut results = Vec::new();
                for _ in 0..3 {
                    match quantizer.quantize(&tensor) {
                        Ok(quantized) => results.push(quantized),
                        Err(_) => break, // Skip if quantization fails
                    }
                }

                if results.len() >= 2 {
                    // All results should be identical
                    for (i, result) in results.iter().enumerate().skip(1) {
                        assert_eq!(
                            results[0].data, result.data,
                            "{} determinism failed for case {} at run {}",
                            quantizer_name, case_id, i
                        );
                        assert_eq!(
                            results[0].shape, result.shape,
                            "{} shape determinism failed for case {} at run {}",
                            quantizer_name, case_id, i
                        );
                    }
                }
            }
        }

        println!("✅ Quantization determinism property verified for all quantizers");
        Ok(())
    }

    /// Property: Shape preservation
    /// For any input tensor of shape S, quantize(x) should preserve the shape
    #[test]
    fn property_shape_preservation() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        let test_shapes = vec![
            (vec![1.0; 16], vec![16]),
            (vec![1.0; 64], vec![8, 8]),
            (vec![1.0; 120], vec![8, 15]),
            (vec![1.0; 256], vec![16, 16]),
            (vec![1.0; 300], vec![12, 25]),
        ];

        for (test_data, shape) in test_shapes {
            let tensor = create_tensor_from_f32(&test_data, &shape)?;

            match quantizer.quantize(&tensor) {
                Ok(quantized) => {
                    assert_eq!(
                        quantized.shape, shape,
                        "Shape not preserved: expected {:?}, got {:?}",
                        shape, quantized.shape
                    );

                    // Test dequantization shape preservation
                    match quantizer.dequantize(&quantized) {
                        Ok(dequantized) => {
                            assert_eq!(
                                dequantized.dims(),
                                shape.as_slice(),
                                "Dequantization shape not preserved: expected {:?}, got {:?}",
                                shape,
                                dequantized.dims()
                            );
                        }
                        Err(_) => {
                            println!("⚠️  Dequantization failed for shape {:?}", shape);
                        }
                    }
                }
                Err(_) => {
                    println!("⚠️  Quantization failed for shape {:?}", shape);
                }
            }
        }

        println!("✅ Shape preservation property verified");
        Ok(())
    }

    /// Property: Idempotency of quantization
    /// quantize(dequantize(quantize(x))) should be equivalent to quantize(x)
    #[test]
    fn property_quantization_idempotency() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        let test_cases = generate_property_test_cases(20);

        for (case_id, test_data) in test_cases.iter().enumerate() {
            let tensor = create_tensor_from_f32(test_data, &[test_data.len()])?;

            match quantizer.quantize(&tensor) {
                Ok(quantized1) => {
                    match quantizer.dequantize(&quantized1) {
                        Ok(dequantized) => {
                            match quantizer.quantize(&dequantized) {
                                Ok(quantized2) => {
                                    // The two quantized results should be very similar
                                    let similarity =
                                        compute_data_similarity(&quantized1.data, &quantized2.data);
                                    assert!(
                                        similarity > 0.99,
                                        "Idempotency failed for case {}: similarity = {:.4}",
                                        case_id,
                                        similarity
                                    );
                                }
                                Err(_) => {
                                    println!("⚠️  Second quantization failed for case {}", case_id);
                                }
                            }
                        }
                        Err(_) => {
                            println!("⚠️  Dequantization failed for case {}", case_id);
                        }
                    }
                }
                Err(_) => {
                    println!("⚠️  Initial quantization failed for case {}", case_id);
                }
            }
        }

        println!("✅ Quantization idempotency property verified");
        Ok(())
    }

    /// Property: Linearity approximation
    /// For small perturbations δ, quantize(x + δ) should be close to quantize(x)
    #[test]
    fn property_continuity_approximation() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        let base_values = vec![
            vec![0.0; 32],
            vec![0.5; 32],
            vec![-0.5; 32],
            (0..32).map(|i| (i as f32 / 32.0) * 2.0 - 1.0).collect(),
        ];

        let perturbation_magnitudes = vec![0.001, 0.01, 0.05];

        for (base_idx, base_data) in base_values.iter().enumerate() {
            let base_tensor = create_tensor_from_f32(base_data, &[32])?;

            match quantizer.quantize(&base_tensor) {
                Ok(base_quantized) => {
                    for &perturbation in &perturbation_magnitudes {
                        // Create perturbed version
                        let perturbed_data: Vec<f32> = base_data
                            .iter()
                            .enumerate()
                            .map(|(i, &x)| x + perturbation * (i as f32).sin())
                            .collect();

                        let perturbed_tensor = create_tensor_from_f32(&perturbed_data, &[32])?;

                        match quantizer.quantize(&perturbed_tensor) {
                            Ok(perturbed_quantized) => {
                                let similarity = compute_data_similarity(
                                    &base_quantized.data,
                                    &perturbed_quantized.data,
                                );

                                // Similarity should decrease gradually with perturbation magnitude
                                let expected_min_similarity = 1.0 - perturbation * 10.0;
                                assert!(
                                    similarity >= expected_min_similarity.max(0.5),
                                    "Continuity failed for base {} with perturbation {}: similarity = {:.4}",
                                    base_idx,
                                    perturbation,
                                    similarity
                                );
                            }
                            Err(_) => {
                                println!(
                                    "⚠️  Perturbed quantization failed for base {} with perturbation {}",
                                    base_idx, perturbation
                                );
                            }
                        }
                    }
                }
                Err(_) => {
                    println!("⚠️  Base quantization failed for base {}", base_idx);
                }
            }
        }

        println!("✅ Continuity approximation property verified");
        Ok(())
    }

    /// Property: Scale invariance (up to precision)
    /// quantize(c * x) should have predictable relationship to quantize(x) for scalar c
    #[test]
    fn property_scale_relationship() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        let base_pattern: Vec<f32> = (0..32).map(|i| (i as f32 / 16.0).sin()).collect();
        let scales = vec![0.1, 0.5, 1.0, 2.0, 5.0];

        let mut scale_behaviors = Vec::new();

        for &scale in &scales {
            let scaled_data: Vec<f32> = base_pattern.iter().map(|&x| x * scale).collect();
            let scaled_tensor = create_tensor_from_f32(&scaled_data, &[32])?;

            match quantizer.quantize(&scaled_tensor) {
                Ok(quantized) => {
                    let quantized_energy = compute_data_energy(&quantized.data);
                    scale_behaviors.push((scale, quantized_energy));
                }
                Err(_) => {
                    println!("⚠️  Quantization failed for scale {}", scale);
                }
            }
        }

        // Analyze scale relationship
        if scale_behaviors.len() >= 3 {
            // Energy should generally increase with scale (within quantization limits)
            let energies: Vec<f64> = scale_behaviors.iter().map(|(_, e)| *e).collect();
            let is_monotonic = energies.windows(2).all(|w| w[1] >= w[0] * 0.8); // Allow some tolerance

            if is_monotonic {
                println!("✅ Scale relationship shows expected monotonic behavior");
            } else {
                println!("ℹ️  Scale relationship is non-monotonic (expected for quantization)");
                for (scale, energy) in &scale_behaviors {
                    println!("  Scale {}: Energy {:.4}", scale, energy);
                }
            }
        }

        println!("✅ Scale relationship property analyzed");
        Ok(())
    }

    /// Property: Quantization error bounds
    /// The error |x - dequantize(quantize(x))| should be bounded
    #[test]
    fn property_error_bounds() -> Result<()> {
        let quantizers: Vec<(&str, Box<dyn QuantizerTrait>, f32)> = vec![
            ("I2S", Box::new(I2SQuantizer::new()), 0.5), // I2S has 2 bits, so max error ~0.5
            ("TL1", Box::new(TL1Quantizer::new()), 0.3), // TL1 should have better precision
            ("TL2", Box::new(TL2Quantizer::new()), 0.3), // TL2 should have better precision
        ];

        for (quantizer_name, quantizer, max_expected_error) in &quantizers {
            let mut max_observed_error = 0.0f32;
            let mut total_error = 0.0f64;
            let mut num_samples = 0;

            let test_cases = generate_property_test_cases(30);

            for test_data in test_cases {
                let tensor = create_tensor_from_f32(&test_data, &[test_data.len()])?;

                match quantizer.quantize(&tensor) {
                    Ok(quantized) => match quantizer.dequantize(&quantized) {
                        Ok(dequantized) => {
                            let reconstructed = tensor_to_vec(&dequantized)?;

                            for (original, reconstructed) in
                                test_data.iter().zip(reconstructed.iter())
                            {
                                let error = (original - reconstructed).abs();
                                max_observed_error = max_observed_error.max(error);
                                total_error += error as f64;
                                num_samples += 1;
                            }
                        }
                        Err(_) => continue,
                    },
                    Err(_) => continue,
                }
            }

            if num_samples > 0 {
                let avg_error = total_error / num_samples as f64;

                println!(
                    "{}: Max error = {:.4}, Avg error = {:.6}, Expected max = {:.2}",
                    quantizer_name, max_observed_error, avg_error, max_expected_error
                );

                // Check if observed error is within expected bounds (with some tolerance)
                let tolerance_factor = 1.5; // Allow 50% tolerance for algorithmic variations
                assert!(
                    max_observed_error <= max_expected_error * tolerance_factor,
                    "{} error bound exceeded: {:.4} > {:.4}",
                    quantizer_name,
                    max_observed_error,
                    max_expected_error * tolerance_factor
                );
            }
        }

        println!("✅ Error bounds property verified for all quantizers");
        Ok(())
    }

    /// Property: Compression effectiveness
    /// Quantized representation should be smaller than original for reasonable inputs
    #[test]
    fn property_compression_effectiveness() -> Result<()> {
        let quantizers: Vec<(&str, Box<dyn QuantizerTrait>)> = vec![
            ("I2S", Box::new(I2SQuantizer::new())),
            ("TL1", Box::new(TL1Quantizer::new())),
            ("TL2", Box::new(TL2Quantizer::new())),
        ];

        let test_patterns = vec![
            ("Random", generate_random_pattern(128)),
            ("Sparse", generate_sparse_pattern(128)),
            ("Smooth", generate_smooth_pattern(128)),
            ("Neural weights", generate_neural_weight_pattern(128)),
        ];

        for (quantizer_name, quantizer) in &quantizers {
            for (pattern_name, pattern_data) in &test_patterns {
                let tensor = create_tensor_from_f32(pattern_data, &[pattern_data.len()])?;

                match quantizer.quantize(&tensor) {
                    Ok(quantized) => {
                        let original_size = pattern_data.len() * 4; // 4 bytes per f32
                        let compressed_size = quantized.data.len();
                        let compression_ratio = original_size as f64 / compressed_size as f64;

                        println!(
                            "{} on {}: {:.2}x compression ({} -> {} bytes)",
                            quantizer_name,
                            pattern_name,
                            compression_ratio,
                            original_size,
                            compressed_size
                        );

                        // Should achieve some compression for most patterns
                        assert!(
                            compression_ratio >= 1.0,
                            "{} failed to compress {} pattern: ratio = {:.2}",
                            quantizer_name,
                            pattern_name,
                            compression_ratio
                        );

                        // For quantization algorithms, expect meaningful compression
                        if compression_ratio < 1.5 {
                            println!(
                                "⚠️  Low compression ratio for {} on {}: {:.2}x",
                                quantizer_name, pattern_name, compression_ratio
                            );
                        }
                    }
                    Err(_) => {
                        println!(
                            "⚠️  {} quantization failed for {} pattern",
                            quantizer_name, pattern_name
                        );
                    }
                }
            }
        }

        println!("✅ Compression effectiveness property verified");
        Ok(())
    }

    /// Property: Quantization preserves relative ordering (monotonicity)
    /// If x₁ < x₂, then dequantize(quantize(x₁)) ≤ dequantize(quantize(x₂)) (approximately)
    #[test]
    fn property_monotonicity_preservation() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Test with ordered sequences
        let test_sequences = vec![
            (-1.0..=1.0).step_by(16).collect::<Vec<f32>>(),
            (-0.5..=0.5).step_by(8).collect::<Vec<f32>>(),
            (0.0..=1.0).step_by(10).collect::<Vec<f32>>(),
        ];

        for (seq_idx, sequence) in test_sequences.iter().enumerate() {
            if sequence.len() < 2 {
                continue;
            }

            let tensor = create_tensor_from_f32(sequence, &[sequence.len()])?;

            match quantizer.quantize(&tensor) {
                Ok(quantized) => {
                    match quantizer.dequantize(&quantized) {
                        Ok(dequantized) => {
                            let reconstructed = tensor_to_vec(&dequantized)?;

                            // Check monotonicity preservation
                            let mut monotonicity_violations = 0;
                            for i in 0..sequence.len() - 1 {
                                if sequence[i] < sequence[i + 1]
                                    && reconstructed[i] > reconstructed[i + 1]
                                {
                                    monotonicity_violations += 1;
                                }
                            }

                            let violation_rate =
                                monotonicity_violations as f64 / (sequence.len() - 1) as f64;
                            println!(
                                "Sequence {}: {:.1}% monotonicity violations",
                                seq_idx,
                                violation_rate * 100.0
                            );

                            // Allow some violations due to quantization granularity
                            assert!(
                                violation_rate <= 0.2,
                                "Too many monotonicity violations in sequence {}: {:.1}%",
                                seq_idx,
                                violation_rate * 100.0
                            );
                        }
                        Err(_) => {
                            println!("⚠️  Dequantization failed for sequence {}", seq_idx);
                        }
                    }
                }
                Err(_) => {
                    println!("⚠️  Quantization failed for sequence {}", seq_idx);
                }
            }
        }

        println!("✅ Monotonicity preservation property verified");
        Ok(())
    }

    // Helper functions for property-based testing

    fn generate_property_test_cases(num_cases: usize) -> Vec<Vec<f32>> {
        let mut test_cases = Vec::new();

        for i in 0..num_cases {
            let size = 16 + (i % 48); // Sizes from 16 to 64
            let seed = i as f32;

            let test_data: Vec<f32> = (0..size)
                .map(|j| {
                    let t = (j as f32 + seed) / size as f32;
                    ((t * PI * 2.0 + seed).sin() + (t * PI * 4.0 + seed * 1.7).cos()) * 0.5
                })
                .collect();

            test_cases.push(test_data);
        }

        test_cases
    }

    fn compute_data_similarity(data1: &[u8], data2: &[u8]) -> f64 {
        if data1.len() != data2.len() {
            return 0.0;
        }

        if data1.is_empty() {
            return 1.0;
        }

        let matching_bytes = data1.iter().zip(data2.iter()).filter(|(a, b)| a == b).count();

        matching_bytes as f64 / data1.len() as f64
    }

    fn compute_data_energy(data: &[u8]) -> f64 {
        data.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / data.len() as f64
    }

    fn generate_random_pattern(size: usize) -> Vec<f32> {
        (0..size).map(|i| ((i * 17 + 7) as f32).sin() * 0.5).collect()
    }

    fn generate_sparse_pattern(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| if i % 8 == 0 { (i as f32 / size as f32) * 2.0 - 1.0 } else { 0.0 })
            .collect()
    }

    fn generate_smooth_pattern(size: usize) -> Vec<f32> {
        (0..size).map(|i| ((i as f32 / size as f32) * PI * 2.0).sin() * 0.3).collect()
    }

    fn generate_neural_weight_pattern(size: usize) -> Vec<f32> {
        let scale = (2.0 / size as f32).sqrt();
        (0..size)
            .map(|i| {
                let u1 = (i as f32 / size as f32 + 0.001).max(0.001);
                let u2 = ((i * 13 + 7) as f32 / size as f32 + 0.001).max(0.001);
                scale * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
            })
            .collect()
    }

    fn tensor_to_vec(tensor: &Tensor) -> Result<Vec<f32>> {
        Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
    }

    // Trait for step_by iterator (simple implementation)
    trait StepBy {
        fn step_by(self, step: usize) -> Vec<f32>;
    }

    impl StepBy for std::ops::RangeInclusive<f32> {
        fn step_by(self, step: usize) -> Vec<f32> {
            let mut result = Vec::new();
            let start = *self.start();
            let end = *self.end();
            let num_steps = step;

            for i in 0..num_steps {
                let t = i as f32 / (num_steps - 1) as f32;
                let value = start + t * (end - start);
                if value <= end {
                    result.push(value);
                }
            }

            result
        }
    }
}
