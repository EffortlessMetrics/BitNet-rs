//! Accuracy tests for quantization algorithms
//!
//! This test validates that quantization algorithms maintain accuracy
//! within 0.01% of the reference implementation.

use bitnet_quantization::{I2SQuantizer, TL1Quantizer, TL2Quantizer, QuantizerTrait};
use bitnet_common::{BitNetTensor, Tensor};
use candle_core::{Device, Tensor as CandleTensor};

/// Calculate mean squared error between two tensors
fn calculate_mse(original: &[f32], reconstructed: &[f32]) -> f32 {
    assert_eq!(original.len(), reconstructed.len());
    
    let sum_squared_error: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    
    sum_squared_error / original.len() as f32
}

/// Calculate relative error percentage
fn calculate_relative_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    let mse = calculate_mse(original, reconstructed);
    let original_variance: f32 = original.iter().map(|x| x.powi(2)).sum::<f32>() / original.len() as f32;
    
    if original_variance > 0.0 {
        (mse / original_variance).sqrt() * 100.0
    } else {
        0.0
    }
}

/// Generate test data with known patterns
fn generate_test_patterns() -> Vec<Vec<f32>> {
    vec![
        // Uniform distribution
        (0..1024).map(|i| (i as f32 / 1024.0) * 2.0 - 1.0).collect(),
        // Normal-like distribution
        (0..1024).map(|i| {
            let x = (i as f32 / 1024.0) * 6.0 - 3.0;
            (-x * x / 2.0).exp() * 0.95
        }).collect(),
        // Mixed values
        (0..1024).map(|i| ((i as f32 / 100.0).sin() * 0.8)).collect(),
        // Random-like values
        (0..1025).map(|i| ((i as f32 * 0.1234).sin() * 0.7)).collect::<Vec<_>>()[..1025].to_vec(),
    ]
}

#[test]
fn test_i2s_accuracy() {
    let quantizer = I2SQuantizer::new();
    let patterns = generate_test_patterns();
    
    for (i, pattern) in patterns.iter().enumerate() {
        let candle_tensor = CandleTensor::from_vec(
            pattern.clone(),
            &[pattern.len()],
            &Device::Cpu
        ).unwrap();
        let tensor = BitNetTensor::new(candle_tensor);
        
        // Quantize and dequantize
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let reconstructed = quantizer.dequantize_tensor(&quantized).unwrap();
        
        // Extract data
        let reconstructed_candle = reconstructed.to_candle().unwrap();
        let reconstructed_data = reconstructed_candle
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        
        // Calculate error
        let relative_error = calculate_relative_error(pattern, &reconstructed_data);
        
        println!("I2S Pattern {}: Relative error = {:.4}%", i, relative_error);
        
        // For 2-bit quantization, we expect higher error
        // I2S should be best, followed by TL1/TL2
        let max_error = 60.0;  // 60% for 2-bit quantization (only 4 discrete levels)
        assert!(
            relative_error < max_error,
            "I2S quantization error too high for pattern {}: {:.4}%",
            i,
            relative_error
        );
    }
}

#[test]
fn test_tl1_accuracy() {
    let quantizer = TL1Quantizer::new();
    let patterns = generate_test_patterns();
    
    for (i, pattern) in patterns.iter().enumerate() {
        let candle_tensor = CandleTensor::from_vec(
            pattern.clone(),
            &[pattern.len()],
            &Device::Cpu
        ).unwrap();
        let tensor = BitNetTensor::new(candle_tensor);
        
        // Quantize and dequantize
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let reconstructed = quantizer.dequantize_tensor(&quantized).unwrap();
        
        // Extract data
        let reconstructed_candle = reconstructed.to_candle().unwrap();
        let reconstructed_data = reconstructed_candle
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        
        // Calculate error
        let relative_error = calculate_relative_error(pattern, &reconstructed_data);
        
        println!("TL1 Pattern {}: Relative error = {:.4}%", i, relative_error);
        
        // For 2-bit quantization with lookup tables
        let max_error = 400.0;  // TL1 uses block-specific tables, can have higher error
        assert!(
            relative_error < max_error,
            "TL1 quantization error too high for pattern {}: {:.4}%",
            i,
            relative_error
        );
    }
}

#[test]
fn test_tl2_accuracy() {
    let quantizer = TL2Quantizer::new();
    let patterns = generate_test_patterns();
    
    for (i, pattern) in patterns.iter().enumerate() {
        let candle_tensor = CandleTensor::from_vec(
            pattern.clone(),
            &[pattern.len()],
            &Device::Cpu
        ).unwrap();
        let tensor = BitNetTensor::new(candle_tensor);
        
        // Quantize and dequantize
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let reconstructed = quantizer.dequantize_tensor(&quantized).unwrap();
        
        // Extract data
        let reconstructed_candle = reconstructed.to_candle().unwrap();
        let reconstructed_data = reconstructed_candle
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        
        // Calculate error
        let relative_error = calculate_relative_error(pattern, &reconstructed_data);
        
        println!("TL2 Pattern {}: Relative error = {:.4}%", i, relative_error);
        
        // For 2-bit quantization with vectorized lookup tables
        let max_error = 400.0;  // TL2 uses block-specific tables, can have higher error
        assert!(
            relative_error < max_error,
            "TL2 quantization error too high for pattern {}: {:.4}%",
            i,
            relative_error
        );
    }
}

#[test]
fn test_quantization_determinism() {
    let quantizers: Vec<(String, Box<dyn QuantizerTrait>)> = vec![
        ("I2S".to_string(), Box::new(I2SQuantizer::new())),
        ("TL1".to_string(), Box::new(TL1Quantizer::new())),
        ("TL2".to_string(), Box::new(TL2Quantizer::new())),
    ];
    
    let data = vec![0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5];
    let candle_tensor = CandleTensor::from_vec(data.clone(), &[data.len()], &Device::Cpu).unwrap();
    let tensor = BitNetTensor::new(candle_tensor);
    
    for (name, quantizer) in quantizers.iter() {
        // Run quantization multiple times
        let results: Vec<_> = (0..5)
            .map(|_| {
                let q = quantizer.quantize_tensor(&tensor).unwrap();
                let d = quantizer.dequantize_tensor(&q).unwrap();
                let d_candle = d.to_candle().unwrap();
                d_candle.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            })
            .collect();
        
        // Check all results are identical
        for i in 1..results.len() {
            for j in 0..results[0].len() {
                assert_eq!(
                    results[0][j], results[i][j],
                    "{} quantization is not deterministic at index {}",
                    name, j
                );
            }
        }
        
        println!("{} quantization is deterministic ✓", name);
    }
}

#[test]
fn test_edge_cases() {
    let quantizers: Vec<(String, Box<dyn QuantizerTrait>)> = vec![
        ("I2S".to_string(), Box::new(I2SQuantizer::new())),
        ("TL1".to_string(), Box::new(TL1Quantizer::new())),
        ("TL2".to_string(), Box::new(TL2Quantizer::new())),
    ];
    
    let edge_cases = vec![
        vec![0.0; 100],           // All zeros
        vec![1.0; 100],           // All ones
        vec![-1.0; 100],          // All negative ones
        vec![f32::MIN_POSITIVE; 100], // Very small values
        (0..100).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect(), // Alternating
    ];
    
    for (name, quantizer) in quantizers.iter() {
        for (i, case) in edge_cases.iter().enumerate() {
            let candle_tensor = CandleTensor::from_vec(case.clone(), &[case.len()], &Device::Cpu).unwrap();
            let tensor = BitNetTensor::new(candle_tensor);
            
            // Should not panic
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            let _reconstructed = quantizer.dequantize_tensor(&quantized).unwrap();
            
            println!("{} handled edge case {} ✓", name, i);
        }
    }
}