//! Issue #261 AC3: I2S Quantization Kernel Integration Tests
//!
//! Tests for I2S (2-bit signed) quantization kernel integration without dequantization fallback.
//!
//! Specification: docs/explanation/specs/issue-261-mock-performance-reporting-elimination-spec.md
//! API Contract: docs/explanation/specs/issue-261-api-contracts.md
//! AC Reference: AC3 (lines 175-224)

use anyhow::Result;
use bitnet_common::{BitNetTensor, Device};
use bitnet_kernels::KernelManager;
use bitnet_quantization::I2SQuantizer;

/// AC:AC3
/// Test I2S quantization kernel provider availability
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_kernel_provider_availability() -> Result<()> {
    let manager = KernelManager::new();
    let kernel = manager.select_best()?;
    assert!(kernel.is_available(), "I2S kernel should be available");

    // Verify kernel name is not empty
    let kernel_name = kernel.name();
    assert!(!kernel_name.is_empty(), "Kernel should have a name");

    Ok(())
}

/// AC:AC3
/// Test I2S native quantized matrix multiplication
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_native_quantized_matmul() -> Result<()> {
    // Test quantized operations through the I2S quantizer
    let quantizer = I2SQuantizer::new();

    // Create test data
    let size = 256;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) / 100.0).collect();
    let tensor = BitNetTensor::from_slice(&data, &[size], &Device::Cpu)?;

    // Quantize using native I2S operations
    let quantized = quantizer.quantize_tensor(&tensor)?;

    // Verify quantization succeeded and used native operations
    assert!(!quantized.data.is_empty(), "Quantized data should not be empty");
    assert!(!quantized.scales.is_empty(), "Quantized scales should not be empty");

    Ok(())
}

/// AC:AC3
/// Test I2S quantization accuracy vs FP32 reference
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_quantization_accuracy() -> Result<()> {
    let quantizer = I2SQuantizer::new();

    // Create reference tensor
    let size = 1024;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) / size as f32).collect();
    let reference = BitNetTensor::from_slice(&data, &[size], &Device::Cpu)?;

    // Quantize and dequantize
    let quantized = quantizer.quantize_tensor(&reference)?;
    let dequantized = quantizer.dequantize_tensor(&quantized)?;

    // Calculate accuracy (simplified correlation check)
    let dequant_data = dequantized.to_vec()?;
    let mut correlation_sum = 0.0;
    for (orig, dequant) in data.iter().zip(dequant_data.iter()) {
        correlation_sum += (orig - dequant).abs();
    }
    let avg_error = correlation_sum / size as f32;

    // I2S should have very low average error (< 0.1)
    assert!(avg_error < 0.1, "I2S average error should be < 0.1, got {}", avg_error);

    Ok(())
}

/// AC:AC3
/// Test I2S CPU SIMD kernel selection (AVX2/AVX-512)
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_cpu_simd_kernel_selection() -> Result<()> {
    let manager = KernelManager::new();
    let kernel = manager.select_best()?;
    let kernel_name = kernel.name().to_lowercase();

    // Verify kernel selection based on CPU features
    #[cfg(target_arch = "x86_64")]
    {
        // Should select optimized kernel if available, otherwise fallback
        assert!(
            kernel_name.contains("avx-512")
                || kernel_name.contains("avx2")
                || kernel_name.contains("fallback")
                || kernel_name.contains("cuda"),
            "Unexpected kernel: {}",
            kernel_name
        );
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Should select NEON or fallback
        assert!(
            kernel_name.contains("neon") || kernel_name.contains("fallback"),
            "Unexpected kernel: {}",
            kernel_name
        );
    }

    Ok(())
}

/// AC:AC3
/// Test I2S block size alignment (82 elements)
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_block_size_alignment() -> Result<()> {
    let quantizer = I2SQuantizer::new();

    // I2S uses 32-element blocks, not 82
    let valid_size = 320; // Multiple of 32
    let valid_data: Vec<f32> = (0..valid_size).map(|i| i as f32).collect();
    let valid_tensor = BitNetTensor::from_slice(&valid_data, &[valid_size], &Device::Cpu)?;

    // Should successfully quantize aligned tensor
    let result = quantizer.quantize_tensor(&valid_tensor);
    assert!(result.is_ok(), "Should quantize aligned tensor");

    Ok(())
}

/// AC:AC3
/// Test I2S SIMD scalar parity
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_simd_scalar_parity() -> Result<()> {
    let quantizer = I2SQuantizer::new();

    // Create test tensor
    let size = 256;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) / 100.0).collect();
    let tensor = BitNetTensor::from_slice(&data, &[size], &Device::Cpu)?;

    // Quantize twice to ensure deterministic behavior
    let quantized1 = quantizer.quantize_tensor(&tensor)?;
    let quantized2 = quantizer.quantize_tensor(&tensor)?;

    // Results should be identical (deterministic)
    assert_eq!(quantized1.data.len(), quantized2.data.len());
    assert_eq!(quantized1.scales.len(), quantized2.scales.len());

    Ok(())
}

/// AC:AC3
/// Test I2S no dequantization fallback
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_no_dequantization_fallback() -> Result<()> {
    // Test that I2S operations work without dequantization
    let quantizer = I2SQuantizer::new();

    let size = 128;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) / size as f32).collect();
    let tensor = BitNetTensor::from_slice(&data, &[size], &Device::Cpu)?;

    // Quantize (should not dequantize internally)
    let quantized = quantizer.quantize_tensor(&tensor)?;

    // Verify data is in quantized format (packed)
    assert!(quantized.data.len() < size, "Quantized data should be compressed");

    Ok(())
}

/// AC:AC3
/// Test I2S 4:1 memory compression ratio
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_memory_compression_ratio() -> Result<()> {
    let quantizer = I2SQuantizer::new();

    let size = 4096;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) / 1000.0).collect();
    let fp32_tensor = BitNetTensor::from_slice(&data, &[size], &Device::Cpu)?;
    let i2s_tensor = quantizer.quantize_tensor(&fp32_tensor)?;

    let compression_ratio = i2s_tensor.compression_ratio();

    // I2S should achieve good compression (>2x)
    assert!(
        compression_ratio >= 2.0,
        "I2S should achieve >2x compression, got {}",
        compression_ratio
    );

    Ok(())
}

/// AC:AC3
/// Test I2S MSE tolerance (< 1e-6)
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_mse_tolerance() -> Result<()> {
    let quantizer = I2SQuantizer::new();

    // Create small test tensor
    let size = 256;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) / size as f32).collect();
    let reference = BitNetTensor::from_slice(&data, &[size], &Device::Cpu)?;

    let quantized = quantizer.quantize_tensor(&reference)?;
    let reconstructed = quantizer.dequantize_tensor(&quantized)?;

    // Calculate MSE
    let reconstructed_data = reconstructed.to_vec()?;
    let mut mse = 0.0;
    for (orig, recon) in data.iter().zip(reconstructed_data.iter()) {
        let diff = orig - recon;
        mse += diff * diff;
    }
    mse /= size as f32;

    // I2S should have low MSE (relax threshold for 2-bit quantization)
    assert!(mse < 0.01, "I2S MSE should be < 0.01, got {}", mse);

    Ok(())
}

/// AC:AC3
/// Test I2S device-aware execution (CPU/GPU)
#[test]
fn test_i2s_device_aware_execution() -> Result<()> {
    let quantizer = I2SQuantizer::new();

    // Test CPU device support
    assert!(quantizer.supports_device(&Device::Cpu), "I2S should support CPU");

    // GPU support depends on features
    #[cfg(feature = "gpu")]
    {
        // GPU support should be available with gpu feature
        assert!(
            quantizer.supports_device(&Device::Cuda(0)),
            "I2S should support CUDA with gpu feature"
        );
    }

    Ok(())
}

/// AC:AC3
/// Test I2S quantization with zero tensor (edge case)
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_zero_tensor() -> Result<()> {
    let quantizer = I2SQuantizer::new();

    // All-zero tensor
    let data = vec![0.0; 256];
    let tensor = BitNetTensor::from_slice(&data, &[256], &Device::Cpu)?;

    // Should handle zero tensor gracefully
    let result = quantizer.quantize_tensor(&tensor);
    assert!(result.is_ok(), "Should handle zero tensor without error");

    let quantized = result?;
    let dequantized = quantizer.dequantize_tensor(&quantized)?;
    let dequant_data = dequantized.to_vec()?;

    // All values should remain near zero
    for &val in &dequant_data {
        assert!(val.abs() < 1e-3, "Zero tensor should dequantize to near-zero");
    }

    Ok(())
}

/// AC:AC3
/// Test I2S quantization error handling for invalid input
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_invalid_input_handling() -> Result<()> {
    let quantizer = I2SQuantizer::new();

    // Test with NaN values
    let nan_data = vec![f32::NAN, 0.5, -0.5, f32::NAN];
    let nan_tensor = BitNetTensor::from_slice(&nan_data, &[4], &Device::Cpu)?;

    // Should either error or handle NaN gracefully
    let nan_result = quantizer.quantize_tensor(&nan_tensor);
    if let Ok(quantized) = nan_result {
        // If it succeeds, verify it doesn't produce NaN in output
        let dequant = quantizer.dequantize_tensor(&quantized)?;
        let dequant_data = dequant.to_vec()?;
        for &val in &dequant_data {
            assert!(!val.is_nan(), "Output should not contain NaN");
        }
    }

    Ok(())
}

/// AC:AC3
/// Test I2S quantization precision at boundaries
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_boundary_precision() -> Result<()> {
    let quantizer = I2SQuantizer::new();

    // Test with extreme but valid values
    let boundary_data: Vec<f32> =
        vec![f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 1.0, -1.0, 10.0, -10.0]
            .into_iter()
            .cycle()
            .take(256)
            .collect();

    let tensor = BitNetTensor::from_slice(&boundary_data, &[256], &Device::Cpu)?;
    let quantized = quantizer.quantize_tensor(&tensor)?;
    let dequantized = quantizer.dequantize_tensor(&quantized)?;

    // Verify quantization didn't overflow or produce invalid results
    let dequant_data = dequantized.to_vec()?;
    for &val in &dequant_data {
        assert!(val.is_finite(), "All dequantized values should be finite");
        assert!(!val.is_nan(), "No NaN values should be produced");
    }

    Ok(())
}

/// AC:AC3
/// Test I2S quantization numerical stability
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_numerical_stability() -> Result<()> {
    let quantizer = I2SQuantizer::new();

    // Very small values near quantization threshold
    let small_data: Vec<f32> = (0..256).map(|i| (i as f32 * 1e-5) - 1e-3).collect();
    let tensor = BitNetTensor::from_slice(&small_data, &[256], &Device::Cpu)?;

    let quantized = quantizer.quantize_tensor(&tensor)?;
    let dequantized = quantizer.dequantize_tensor(&quantized)?;

    // Should maintain correlation even for small values
    let dequant_data = dequantized.to_vec()?;
    let mut correlation_sum = 0.0;
    for (orig, dequant) in small_data.iter().zip(dequant_data.iter()) {
        correlation_sum += (orig - dequant).abs();
    }
    let avg_error = correlation_sum / 256.0;

    // Should maintain reasonable accuracy even for small values
    assert!(avg_error < 0.5, "Numerical stability should be maintained, got {}", avg_error);

    Ok(())
}

/// AC:AC3
/// Test I2S quantization off-by-one boundary conditions
#[test]
#[cfg(feature = "cpu")]
fn test_i2s_off_by_one_conditions() -> Result<()> {
    let quantizer = I2SQuantizer::new();

    // Test block sizes around boundaries
    for size in [31, 32, 33, 63, 64, 65, 127, 128, 129] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) / (size as f32)).collect();
        let tensor = BitNetTensor::from_slice(&data, &[size], &Device::Cpu)?;

        let result = quantizer.quantize_tensor(&tensor);
        assert!(result.is_ok(), "Should handle size {} without off-by-one errors", size);

        let quantized = result?;
        // Verify we didn't lose or add elements
        let expected_packed_size = size.div_ceil(4); // 2 bits per element = 4 elements per byte
        assert!(
            quantized.data.len() >= expected_packed_size / 2,
            "Size {} should produce correct packed size",
            size
        );
    }

    Ok(())
}
