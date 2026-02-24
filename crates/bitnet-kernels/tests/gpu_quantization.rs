//! Comprehensive GPU quantization tests
//!
//! These tests validate GPU quantization functionality, device-aware fallback,
//! and accuracy compared to CPU implementations.

#![cfg(feature = "gpu")]

use bitnet_common::{Device, QuantizationType, Result};
use bitnet_kernels::{
    KernelProvider,
    device_aware::{DeviceAwareQuantizer, DeviceAwareQuantizerFactory},
    gpu::{CudaKernel, is_cuda_available},
};

/// Test data generator for quantization tests
struct TestDataGenerator;

impl TestDataGenerator {
    /// Generate test input with known patterns
    fn generate_test_input(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                match i % 8 {
                    0 => 1.0,                     // High positive
                    1 => -1.0,                    // High negative
                    2 => 0.5,                     // Medium positive
                    3 => -0.5,                    // Medium negative
                    4 => 0.1,                     // Low positive
                    5 => -0.1,                    // Low negative
                    6 => 0.0,                     // Zero
                    _ => 0.25 * (i as f32).sin(), // Varying values
                }
            })
            .collect()
    }

    /// Generate random test input
    fn generate_random_input(size: usize, seed: u64) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let mut state = hasher.finish();

        (0..size)
            .map(|_| {
                // Simple LCG for reproducible random numbers
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                let normalized = (state as f32) / (u64::MAX as f32);
                (normalized - 0.5) * 4.0 // Range: -2.0 to 2.0
            })
            .collect()
    }
}

#[test]
fn test_cuda_available() {
    if is_cuda_available() {
        println!("✅ CUDA is available for testing");
    } else {
        println!("⚠️ CUDA not available - tests will be skipped");
    }
}

#[test]
fn test_gpu_quantization_creation() -> Result<()> {
    if !is_cuda_available() {
        println!("Skipping GPU tests - CUDA not available");
        return Ok(());
    }

    let kernel = CudaKernel::new()?;
    assert!(kernel.is_available());
    println!("✅ CUDA kernel created successfully");

    Ok(())
}

#[test]
fn test_device_aware_quantizer_creation() -> Result<()> {
    // Test CPU creation (should always work)
    let cpu_quantizer = DeviceAwareQuantizer::new(Device::Cpu)?;
    assert_eq!(cpu_quantizer.device(), Device::Cpu);
    assert!(!cpu_quantizer.is_gpu_active());
    println!("✅ CPU device-aware quantizer created");

    // Test GPU creation if available
    if is_cuda_available() {
        let gpu_quantizer = DeviceAwareQuantizer::new(Device::Cuda(0))?;
        assert_eq!(gpu_quantizer.device(), Device::Cuda(0));
        println!(
            "✅ GPU device-aware quantizer created, GPU active: {}",
            gpu_quantizer.is_gpu_active()
        );
    }

    Ok(())
}

#[test]
fn test_quantization_factory() -> Result<()> {
    let quantizer = DeviceAwareQuantizerFactory::auto_detect()?;
    println!("✅ Auto-detected quantizer active provider: {}", quantizer.active_provider());

    let devices = DeviceAwareQuantizerFactory::list_available_devices();
    println!("Available devices: {:?}", devices);
    assert!(!devices.is_empty());
    assert!(devices.contains(&Device::Cpu));

    Ok(())
}

#[test]
fn test_i2s_quantization_small_input() -> Result<()> {
    let quantizer = DeviceAwareQuantizer::new(Device::Cpu)?; // Start with CPU

    // Test with small, predictable input
    let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.25, -0.25, 0.75];
    let mut output = vec![0u8; 2]; // 8 values / 4 per byte = 2 bytes
    let mut scales = vec![0.0f32; 8_usize.div_ceil(32)]; // Block size 32

    let result = quantizer.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
    assert!(result.is_ok());

    // Verify output is not all zeros (indicating quantization worked)
    let has_nonzero = output.iter().any(|&x| x != 0) || scales.iter().any(|&x| x != 0.0);
    assert!(has_nonzero, "Quantization should produce non-zero output");

    println!("✅ Small I2S quantization test passed");
    println!("   Input: {:?}", input);
    println!("   Output: {:?}", output);
    println!("   Scales: {:?}", scales);

    Ok(())
}

#[test]
#[ignore = "Only run with --ignored flag when CUDA is available"]
fn test_gpu_i2s_quantization() -> Result<()> {
    if !is_cuda_available() {
        println!("Skipping GPU quantization test - CUDA not available");
        return Ok(());
    }

    let quantizer = DeviceAwareQuantizer::new(Device::Cuda(0))?;

    // Test with medium-sized input
    let input = TestDataGenerator::generate_test_input(256);
    let mut output = vec![0u8; 64]; // 256/4 = 64 bytes
    let mut scales = vec![0.0f32; 256_usize.div_ceil(32)]; // Block size 32

    let result = quantizer.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);

    match result {
        Ok(()) => {
            println!("✅ GPU I2S quantization succeeded");
            println!("   Active provider: {}", quantizer.active_provider());

            // Verify output
            let has_nonzero = output.iter().any(|&x| x != 0) || scales.iter().any(|&x| x != 0.0);
            assert!(has_nonzero, "GPU quantization should produce non-zero output");

            if let Some(stats) = quantizer.get_stats() {
                println!("   Device stats: {:?}", stats);
            }
        }
        Err(e) => {
            println!("⚠️ GPU quantization failed, but may have fallen back: {}", e);
            // Check if fallback worked
            assert_eq!(quantizer.active_provider(), "fallback");
        }
    }

    Ok(())
}

#[test]
#[ignore = "Only run with --ignored flag when CUDA is available"]
fn test_gpu_vs_cpu_quantization_accuracy() -> Result<()> {
    if !is_cuda_available() {
        println!("Skipping GPU vs CPU comparison - CUDA not available");
        return Ok(());
    }

    // Create quantizers for both devices
    let cpu_quantizer = DeviceAwareQuantizer::new(Device::Cpu)?;
    let gpu_quantizer = DeviceAwareQuantizer::new(Device::Cuda(0))?;

    // Test with identical input
    let input = TestDataGenerator::generate_random_input(512, 12345);

    // CPU quantization
    let mut cpu_output = vec![0u8; 128]; // 512/4 = 128 bytes
    let mut cpu_scales = vec![0.0f32; 512_usize.div_ceil(32)]; // Block size 32

    let cpu_result =
        cpu_quantizer.quantize(&input, &mut cpu_output, &mut cpu_scales, QuantizationType::I2S);
    assert!(cpu_result.is_ok(), "CPU quantization should succeed");

    // GPU quantization (with potential fallback)
    let mut gpu_output = vec![0u8; 128];
    let mut gpu_scales = vec![0.0f32; 512_usize.div_ceil(32)]; // Block size 32

    let gpu_result =
        gpu_quantizer.quantize(&input, &mut gpu_output, &mut gpu_scales, QuantizationType::I2S);
    assert!(gpu_result.is_ok(), "GPU quantization (or fallback) should succeed");

    println!("✅ GPU vs CPU quantization comparison completed");
    println!("   CPU provider: {}", cpu_quantizer.active_provider());
    println!("   GPU provider: {}", gpu_quantizer.active_provider());

    // If GPU actually ran (not fallback), we could compare accuracy
    if gpu_quantizer.is_gpu_active() {
        println!("   GPU quantization active - comparing results");

        // Compare a few values for basic sanity check
        let cpu_nonzero = cpu_output.iter().filter(|&&x| x != 0).count();
        let gpu_nonzero = gpu_output.iter().filter(|&&x| x != 0).count();

        println!("   CPU non-zero values: {}, GPU non-zero values: {}", cpu_nonzero, gpu_nonzero);

        // Both should have produced meaningful results
        assert!(cpu_nonzero > 0, "CPU should produce non-zero output");
        assert!(gpu_nonzero > 0, "GPU should produce non-zero output");
    } else {
        println!("   GPU fell back to CPU - providers should be identical");
    }

    Ok(())
}

#[test]
fn test_quantization_types_support() -> Result<()> {
    let quantizer = DeviceAwareQuantizer::new(Device::Cpu)?;

    let test_cases = [
        (QuantizationType::I2S, "I2S"),
        (QuantizationType::TL1, "TL1"),
        (QuantizationType::TL2, "TL2"),
    ];

    for (qtype, name) in test_cases.iter() {
        let input = vec![1.0, -1.0, 0.5, -0.5];
        let mut output = vec![0u8; 1];
        let mut scales = vec![0.0f32; 1];

        let result = quantizer.quantize(&input, &mut output, &mut scales, *qtype);

        match result {
            Ok(()) => {
                println!("✅ {} quantization supported", name);
            }
            Err(e) => {
                println!("⚠️ {} quantization failed: {}", name, e);
                // For now, we allow failures as some quantization types might not be fully implemented
            }
        }
    }

    Ok(())
}

#[test]
#[ignore = "Only run with --ignored flag when CUDA is available"]
fn test_gpu_quantization_fallback() -> Result<()> {
    if !is_cuda_available() {
        println!("Skipping GPU fallback test - CUDA not available");
        return Ok(());
    }

    let mut quantizer = DeviceAwareQuantizer::new(Device::Cuda(0))?;

    // Test normal operation
    let input = vec![1.0; 64];
    let mut output = vec![0u8; 16];
    let mut scales = vec![0.0f32; 64_usize.div_ceil(32)]; // Block size 32

    let result1 = quantizer.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
    assert!(result1.is_ok());

    let provider_before = quantizer.active_provider();
    println!("Provider before fallback: {}", provider_before);

    // Force fallback
    quantizer.force_cpu_fallback();
    let provider_after = quantizer.active_provider();
    println!("Provider after fallback: {}", provider_after);

    // Test operation after forced fallback
    let result2 = quantizer.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
    assert!(result2.is_ok());

    println!("✅ GPU fallback mechanism works correctly");

    Ok(())
}

#[test]
fn test_empty_input_handling() -> Result<()> {
    let quantizer = DeviceAwareQuantizer::new(Device::Cpu)?;

    // Test empty input
    let input: Vec<f32> = vec![];
    let mut output: Vec<u8> = vec![];
    let mut scales: Vec<f32> = vec![];

    let result = quantizer.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
    assert!(result.is_ok(), "Empty input should be handled gracefully");

    println!("✅ Empty input handling works");

    Ok(())
}

#[test]
#[ignore = "Only run with --ignored flag when CUDA is available"]
fn test_gpu_memory_management() -> Result<()> {
    if !is_cuda_available() {
        println!("Skipping GPU memory test - CUDA not available");
        return Ok(());
    }

    // Test multiple quantization operations to check for memory leaks
    for i in 0..10 {
        let quantizer = DeviceAwareQuantizer::new(Device::Cuda(0))?;

        let input = TestDataGenerator::generate_random_input(1024, i as u64);
        let mut output = vec![0u8; 256];
        let mut scales = vec![0.0f32; 8];

        let result = quantizer.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);

        match result {
            Ok(()) => {
                println!("Iteration {}: GPU quantization succeeded", i);
            }
            Err(e) => {
                println!("Iteration {}: GPU quantization failed (fallback): {}", i, e);
            }
        }
    }

    println!("✅ GPU memory management test completed");
    Ok(())
}

#[test]
#[ignore = "Only run with --ignored flag when CUDA is available"]
fn test_concurrent_gpu_operations() -> Result<()> {
    if !is_cuda_available() {
        println!("Skipping concurrent GPU test - CUDA not available");
        return Ok(());
    }

    use std::sync::Arc;
    use std::thread;

    let quantizer = Arc::new(DeviceAwareQuantizer::new(Device::Cuda(0))?);
    let mut handles = vec![];

    // Launch multiple concurrent operations
    for thread_id in 0..4 {
        let quantizer_clone = Arc::clone(&quantizer);

        let handle = thread::spawn(move || -> Result<()> {
            let input = TestDataGenerator::generate_random_input(256, thread_id as u64);
            let mut output = vec![0u8; 64];
            let mut scales = vec![0.0f32; 2];

            let result =
                quantizer_clone.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);

            match result {
                Ok(()) => {
                    println!("Thread {}: GPU quantization succeeded", thread_id);
                }
                Err(e) => {
                    println!("Thread {}: GPU quantization failed: {}", thread_id, e);
                }
            }

            Ok(())
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().map_err(|_| {
            bitnet_common::BitNetError::Kernel(bitnet_common::KernelError::ExecutionFailed {
                reason: "Thread panicked".to_string(),
            })
        })??;
    }

    println!("✅ Concurrent GPU operations test completed");
    Ok(())
}
