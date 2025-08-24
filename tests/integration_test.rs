#![cfg(feature = "integration-tests")]
//! Integration tests for the optimized quantization kernels
//!
//! This test suite validates that the optimized SIMD kernels work correctly
//! in the full inference pipeline, from model loading to generation.

use bitnet_common::{BitNetTensor, Device, QuantizationType};
use bitnet_models::minimal::{LoadMode, load_minimal};
use bitnet_quantization::{I2SQuantizer, QuantizerTrait, TL1Quantizer, TL2Quantizer};

#[test]
fn test_minimal_model_loading() {
    // Load minimal model with dummy weights
    let model = load_minimal(LoadMode::Dummy { vocab: 100, dim: 64 }).unwrap();

    // Verify model dimensions
    assert_eq!(model.vocab, 100);
    assert_eq!(model.dim, 64);
    assert_eq!(model.tok_embeddings.len(), 100 * 64);
    assert_eq!(model.lm_head.len(), 64 * 100);

    // Check that weights are in reasonable range
    for val in &model.tok_embeddings[..10] {
        assert!(val.is_finite(), "Weight is not finite: {}", val);
        assert!(val.abs() <= 0.01, "Weight out of range: {}", val);
    }
}

#[test]
fn test_quantization_roundtrip_integration() {
    // Test that quantization works end-to-end with tensor operations
    let quantizers: Vec<(&str, Box<dyn QuantizerTrait>)> = vec![
        ("I2S", Box::new(I2SQuantizer::new())),
        ("TL1", Box::new(TL1Quantizer::new())),
        ("TL2", Box::new(TL2Quantizer::new())),
    ];

    // Create test data that simulates model weights
    let test_size = 1024;
    let test_data: Vec<f32> = (0..test_size).map(|i| ((i as f32 / 100.0).sin() * 0.5)).collect();

    for (name, quantizer) in quantizers {
        println!("Testing {} quantizer integration", name);

        // Create tensor using BitNetTensor::from_slice
        let tensor = BitNetTensor::from_slice(&test_data, &[test_size], &Device::Cpu).unwrap();

        // Quantize
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        assert!(quantized.data.len() > 0, "{}: Quantized data is empty", name);

        // Dequantize
        let reconstructed = quantizer.dequantize_tensor(&quantized).unwrap();
        let reconstructed_data = reconstructed.to_vec().unwrap();

        assert_eq!(reconstructed_data.len(), test_data.len());

        // Check that values are in reasonable range
        for val in &reconstructed_data {
            assert!(val.is_finite(), "{}: Reconstructed value is not finite", name);
            assert!(val.abs() <= 2.0, "{}: Reconstructed value out of range: {}", name, val);
        }
    }
}

#[cfg(feature = "kernels")]
#[test]
fn test_simd_kernel_selection() {
    use bitnet_kernels::KernelManager;

    // Create kernel manager which auto-selects best kernel
    let manager = KernelManager::new();
    let kernel = manager.select_best().unwrap();

    println!("Selected kernel: {}", kernel.name());

    // Verify kernel is available
    assert!(kernel.is_available());

    // Test a small quantization operation
    let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3];
    let mut output = vec![0u8; input.len() / 4]; // 2-bit packing
    let mut scales = vec![0.0f32; 1];

    kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();

    // Verify output is non-zero
    assert!(output.iter().any(|&x| x != 0), "Quantized output is all zeros");
    assert!(scales[0] != 0.0, "Scale factor is zero");
}

#[test]
fn test_quantization_performance() {
    use std::time::Instant;

    // Create test data
    let size = 10240;
    let data: Vec<f32> = (0..size).map(|i| (i as f32 / 1000.0).sin()).collect();

    // Test I2S quantization performance
    let quantizer = I2SQuantizer::new();
    let tensor = BitNetTensor::from_slice(&data, &[size], &Device::Cpu).unwrap();

    // Warm up
    let _ = quantizer.quantize_tensor(&tensor).unwrap();

    // Measure performance
    let start = Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        let _ = quantizer.quantize_tensor(&tensor).unwrap();
    }
    let elapsed = start.elapsed();

    let throughput = (size * iterations) as f64 / elapsed.as_secs_f64();
    println!("I2S quantization throughput: {:.2} elements/sec", throughput);

    // Ensure reasonable performance (at least 1M elements/sec on modern CPU)
    assert!(throughput > 1_000_000.0, "SIMD performance seems too low");
}

#[test]
fn test_quantization_consistency() {
    // Ensure quantization produces consistent results across runs
    let quantizers: Vec<Box<dyn QuantizerTrait>> = vec![
        Box::new(I2SQuantizer::new()),
        Box::new(TL1Quantizer::new()),
        Box::new(TL2Quantizer::new()),
    ];

    let data = vec![0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3];

    for quantizer in quantizers {
        let tensor = BitNetTensor::from_slice(&data, &[data.len()], &Device::Cpu).unwrap();

        // Run multiple times
        let q1 = quantizer.quantize_tensor(&tensor).unwrap();
        let q2 = quantizer.quantize_tensor(&tensor).unwrap();

        // Compare results
        assert_eq!(q1.data, q2.data, "Quantization is not deterministic");
        assert_eq!(q1.scales, q2.scales, "Scales are not deterministic");
    }
}
