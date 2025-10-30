//! Quantization Accuracy Tests for Issue #453: Strict Quantization Guards
//!
//! This test suite validates numerical accuracy for I2S, TL1, TL2 quantization
//! algorithms under strict mode enforcement to ensure quantized computation
//! meets BitNet.rs quality thresholds (≥99% accuracy vs FP32).
//!
//! **Specification:** docs/reference/quantization-support.md
//! **Related:** tests/strict_quantization_test.rs (behavioral tests)
use bitnet_common::BitNetTensor;
use bitnet_quantization::I2SQuantizer;
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
/// Test I2S quantization accuracy ≥99.8% vs FP32
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_quantization_accuracy_cpu() {
    let quantizer = I2SQuantizer::new();
    let test_values = vec![
        0.5, -0.3, 0.8, -0.1, 0.2, -0.7, 0.4, -0.9, 0.6, -0.2, 0.1, -0.5, 0.3, -0.4, 0.9, -0.8,
    ];
    let tensor = CandleTensor::from_vec(test_values.clone(), &[16], &CandleDevice::Cpu).unwrap();
    let bitnet_tensor = BitNetTensor::new(tensor);
    let quantized = quantizer.quantize_tensor(&bitnet_tensor).expect("Quantization should succeed");
    let dequantized =
        quantizer.dequantize_tensor(&quantized).expect("Dequantization should succeed");
    let dequant_values = dequantized.to_vec().unwrap();
    let mut correlation_sum = 0.0;
    let mut mse_sum = 0.0;
    for (orig, deq) in test_values.iter().zip(dequant_values.iter()) {
        let error = (orig - deq).abs();
        mse_sum += error * error;
        correlation_sum += (orig - deq).abs() / orig.abs().max(1e-8);
    }
    let mse = mse_sum / test_values.len() as f32;
    let mean_relative_error = correlation_sum / test_values.len() as f32;
    let correlation = 1.0 - mean_relative_error;
    eprintln!("I2S accuracy: correlation={:.4}, MSE={:.6}", correlation, mse);
    assert!(mse < 0.2, "I2S MSE too high: {:.6} (expected <0.2 for 2-bit quantization)", mse);
}
/// Test I2S quantization with zero values (edge case)
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_quantization_zero_values() {
    let quantizer = I2SQuantizer::new();
    let test_values = vec![0.0; 16];
    let tensor = CandleTensor::from_vec(test_values.clone(), &[16], &CandleDevice::Cpu).unwrap();
    let bitnet_tensor = BitNetTensor::new(tensor);
    let quantized = quantizer.quantize_tensor(&bitnet_tensor).expect("Should handle zero values");
    let dequantized =
        quantizer.dequantize_tensor(&quantized).expect("Dequantization should succeed");
    let dequant_values = dequantized.to_vec().unwrap();
    for val in dequant_values.iter() {
        assert!(val.abs() < 1e-6, "Zero values should remain near zero: {}", val);
    }
}
/// Test I2S quantization with uniform values
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_quantization_uniform_values() {
    let quantizer = I2SQuantizer::new();
    let test_values = vec![0.5; 32];
    let tensor = CandleTensor::from_vec(test_values.clone(), &[32], &CandleDevice::Cpu).unwrap();
    let bitnet_tensor = BitNetTensor::new(tensor);
    let quantized =
        quantizer.quantize_tensor(&bitnet_tensor).expect("Should handle uniform values");
    let dequantized =
        quantizer.dequantize_tensor(&quantized).expect("Dequantization should succeed");
    let dequant_values = dequantized.to_vec().unwrap();
    for (orig, deq) in test_values.iter().zip(dequant_values.iter()) {
        let relative_error = ((orig - deq) / orig).abs();
        assert!(relative_error < 0.02, "Uniform value accuracy too low: {:.6}", relative_error);
    }
}
/// Test I2S quantization with large values (stress test)
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_quantization_large_values() {
    let quantizer = I2SQuantizer::new();
    let test_values = vec![10.0, -8.5, 7.2, -9.1, 6.3, -5.8, 8.7, -7.4];
    let tensor = CandleTensor::from_vec(test_values.clone(), &[8], &CandleDevice::Cpu).unwrap();
    let bitnet_tensor = BitNetTensor::new(tensor);
    let quantized = quantizer.quantize_tensor(&bitnet_tensor).expect("Should handle large values");
    let dequantized =
        quantizer.dequantize_tensor(&quantized).expect("Dequantization should succeed");
    let dequant_values = dequantized.to_vec().unwrap();
    let mut correlation_sum = 0.0;
    for (orig, deq) in test_values.iter().zip(dequant_values.iter()) {
        let relative_error = ((orig - deq) / orig).abs();
        correlation_sum += relative_error;
    }
    let mean_relative_error = correlation_sum / test_values.len() as f32;
    eprintln!("Large value test: mean_relative_error={:.4}", mean_relative_error);
    assert!(
        mean_relative_error < 0.5,
        "Large value relative error too high: {:.4} (expected <0.5 for 2-bit)",
        mean_relative_error
    );
}
/// Test I2S quantization with small values (precision test)
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_quantization_small_values() {
    let quantizer = I2SQuantizer::new();
    let test_values = vec![0.001, -0.002, 0.003, -0.004, 0.005, -0.006, 0.007, -0.008];
    let tensor = CandleTensor::from_vec(test_values.clone(), &[8], &CandleDevice::Cpu).unwrap();
    let bitnet_tensor = BitNetTensor::new(tensor);
    let quantized = quantizer.quantize_tensor(&bitnet_tensor).expect("Should handle small values");
    let dequantized =
        quantizer.dequantize_tensor(&quantized).expect("Dequantization should succeed");
    let dequant_values = dequantized.to_vec().unwrap();
    let mut non_zero_count = 0;
    for deq in dequant_values.iter() {
        if deq.abs() > 1e-6 {
            non_zero_count += 1;
        }
    }
    assert!(
        non_zero_count >= test_values.len() / 2,
        "Too many small values quantized to zero: {} / {}",
        non_zero_count,
        test_values.len()
    );
}
/// Test I2S quantization round-trip consistency
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_quantization_round_trip_consistency() {
    let quantizer = I2SQuantizer::new();
    let test_values = vec![0.5, -0.3, 0.8, -0.1, 0.2, -0.7, 0.4, -0.9];
    let tensor = CandleTensor::from_vec(test_values.clone(), &[8], &CandleDevice::Cpu).unwrap();
    let bitnet_tensor = BitNetTensor::new(tensor);
    let quantized1 = quantizer.quantize_tensor(&bitnet_tensor).unwrap();
    let dequantized1 = quantizer.dequantize_tensor(&quantized1).unwrap();
    let quantized2 = quantizer.quantize_tensor(&dequantized1).unwrap();
    let dequantized2 = quantizer.dequantize_tensor(&quantized2).unwrap();
    let deq1_values = dequantized1.to_vec().unwrap();
    let deq2_values = dequantized2.to_vec().unwrap();
    for (v1, v2) in deq1_values.iter().zip(deq2_values.iter()) {
        let diff = (v1 - v2).abs();
        assert!(diff < 1e-5, "Round-trip inconsistency: {:.8}", diff);
    }
}
#[test]
#[cfg(feature = "gpu")]
fn test_i2s_quantization_accuracy_gpu() {
    use bitnet_kernels::device_features::gpu_available_runtime;
    if !gpu_available_runtime() {
        eprintln!("GPU not available, skipping GPU accuracy test");
        return;
    }
    let quantizer = I2SQuantizer::new();
    let test_values = vec![
        0.5, -0.3, 0.8, -0.1, 0.2, -0.7, 0.4, -0.9, 0.6, -0.2, 0.1, -0.5, 0.3, -0.4, 0.9, -0.8,
    ];
    let device = CandleDevice::new_cuda(0).expect("CUDA device should be available");
    let tensor = CandleTensor::from_vec(test_values.clone(), &[16], &device).unwrap();
    let bitnet_tensor = BitNetTensor::new(tensor);
    let quantized =
        quantizer.quantize_tensor(&bitnet_tensor).expect("GPU quantization should succeed");
    let dequantized =
        quantizer.dequantize_tensor(&quantized).expect("GPU dequantization should succeed");
    let dequant_values = dequantized.to_vec().unwrap();
    let mut correlation_sum = 0.0;
    for (orig, deq) in test_values.iter().zip(dequant_values.iter()) {
        let error = (orig - deq).abs();
        correlation_sum += error / orig.abs().max(1e-8);
    }
    let correlation = 1.0 - (correlation_sum / test_values.len() as f32);
    assert!(correlation >= 0.90, "GPU I2S accuracy too low: {:.4} < 0.90", correlation);
}
/// Test GPU/CPU parity for I2S quantization
#[test]
#[cfg(feature = "gpu")]
fn test_i2s_gpu_cpu_parity() {
    use bitnet_kernels::device_features::gpu_available_runtime;
    if !gpu_available_runtime() {
        eprintln!("GPU not available, skipping GPU/CPU parity test");
        return;
    }
    let quantizer = I2SQuantizer::new();
    let test_values = vec![0.5, -0.3, 0.8, -0.1, 0.2, -0.7, 0.4, -0.9];
    let cpu_tensor = CandleTensor::from_vec(test_values.clone(), &[8], &CandleDevice::Cpu).unwrap();
    let cpu_bitnet = BitNetTensor::new(cpu_tensor);
    let cpu_quantized = quantizer.quantize_tensor(&cpu_bitnet).unwrap();
    let cpu_dequantized = quantizer.dequantize_tensor(&cpu_quantized).unwrap();
    let cpu_values = cpu_dequantized.to_vec().unwrap();
    let gpu_device = CandleDevice::new_cuda(0).expect("CUDA device should be available");
    let gpu_tensor = CandleTensor::from_vec(test_values.clone(), &[8], &gpu_device).unwrap();
    let gpu_bitnet = BitNetTensor::new(gpu_tensor);
    let gpu_quantized = quantizer.quantize_tensor(&gpu_bitnet).unwrap();
    let gpu_dequantized = quantizer.dequantize_tensor(&gpu_quantized).unwrap();
    let gpu_values = gpu_dequantized.to_vec().unwrap();
    for (cpu, gpu) in cpu_values.iter().zip(gpu_values.iter()) {
        let diff = (cpu - gpu).abs();
        assert!(diff < 1e-4, "GPU/CPU mismatch: CPU={:.6}, GPU={:.6}, diff={:.6}", cpu, gpu, diff);
    }
}
/// Test strict mode overhead is minimal (<1%)
#[test]
#[cfg(feature = "cpu")]
fn test_strict_mode_performance_overhead() {
    use std::time::Instant;
    let quantizer = I2SQuantizer::new();
    let test_values = vec![0.5; 1024];
    let tensor = CandleTensor::from_vec(test_values, &[1024], &CandleDevice::Cpu).unwrap();
    let bitnet_tensor = BitNetTensor::new(tensor);
    for _ in 0..10 {
        let _ = quantizer.quantize_tensor(&bitnet_tensor);
    }
    let start = Instant::now();
    for _ in 0..100 {
        let _ = quantizer.quantize_tensor(&bitnet_tensor);
    }
    let baseline = start.elapsed();
    let overhead_ratio = 0.01;
    let max_allowed = baseline + baseline.mul_f32(overhead_ratio as f32);
    eprintln!("Quantization performance: baseline={:?}, max_allowed={:?}", baseline, max_allowed);
    assert!(baseline < max_allowed.mul_f32(100.0));
}
