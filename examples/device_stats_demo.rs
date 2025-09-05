#!/usr/bin/env rust-script
//! Demonstration of device-aware quantization with memory tracking and platform-specific kernels
//!
//! This example showcases the new memory statistics tracking and platform-specific kernel selection
//! introduced in PR #177.
//!
//! Run with: cargo run --example device_stats_demo --no-default-features --features cpu

use bitnet_kernels::device_aware::{DeviceAwareQuantizer, DeviceAwareQuantizerFactory};
use bitnet_common::{Device, QuantizationType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger to see kernel selection details
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("=== BitNet.rs Device-Aware Quantization with Memory Tracking ===\n");

    // Demonstrate auto-detection
    println!("🔍 Auto-detecting best device...");
    let auto_quantizer = DeviceAwareQuantizerFactory::auto_detect()?;
    println!("   Auto-detected device: {:?}", auto_quantizer.device());
    println!("   Active provider: {}", auto_quantizer.active_provider());

    // List all available devices
    println!("\n📱 Available devices:");
    let devices = DeviceAwareQuantizerFactory::list_available_devices();
    for device in &devices {
        println!("   - {:?}", device);
    }

    println!("\n=== Testing CPU Device with Memory Tracking ===\n");

    // Create a CPU-specific quantizer to demonstrate memory tracking
    let quantizer = DeviceAwareQuantizer::new(Device::Cpu)?;
    println!("✅ Created CPU quantizer");
    println!("   Active provider: {}", quantizer.active_provider());
    println!("   GPU active: {}", quantizer.is_gpu_active());

    // Get initial statistics
    if let Some(stats) = quantizer.get_stats() {
        println!("\n📊 Initial statistics:");
        println!("   {}", stats.summary());
        println!("   Memory used: {:.2} MB", stats.memory_used_bytes as f64 / (1024.0 * 1024.0));
        println!("   Memory total: {:.2} MB", stats.memory_total_bytes as f64 / (1024.0 * 1024.0));
        println!("   Memory utilization: {:.1}%", 
            (stats.memory_used_bytes as f64 / stats.memory_total_bytes as f64) * 100.0);
    }

    println!("\n🔧 Performing quantization operations...");

    // Perform several quantization operations
    for i in 1..=5 {
        let size = 1024 * i; // Increasing sizes
        let input: Vec<f32> = (0..size).map(|j| (j as f32).sin()).collect();
        let mut output = vec![0u8; size / 4]; // I2S packs 4 values per byte
        let mut scales = vec![0.0f32; (size + 127) / 128]; // 128-element blocks
        
        println!("   Operation {}: {} elements", i, size);
        quantizer.quantize(&input, &mut output, &mut scales, QuantizationType::I2S)?;
    }

    // Perform matrix multiplication operations
    println!("\n🧮 Performing matrix multiplication operations...");
    for i in 1..=3 {
        let size = 64 * i;
        let a = vec![1i8; size * size];
        let b = vec![255u8; size * size];
        let mut c = vec![0.0f32; size * size];
        
        println!("   MatMul {}: {}x{} matrix", i, size, size);
        quantizer.matmul_i2s(&a, &b, &mut c, size, size, size)?;
    }

    // Get final statistics
    if let Some(stats) = quantizer.get_stats() {
        println!("\n📈 Final statistics:");
        println!("   {}", stats.summary());
        
        println!("\n📊 Detailed breakdown:");
        println!("   Total operations: {}", stats.total_operations);
        println!("   - Quantization ops: {}", stats.quantization_operations);
        println!("   - MatMul ops: {}", stats.matmul_operations);
        println!("   Performance:");
        println!("   - Total time: {:.2}ms", stats.total_time_ms);
        println!("   - Avg quantization: {:.2}ms", stats.avg_quantization_time_ms());
        println!("   - Avg matmul: {:.2}ms", stats.avg_matmul_time_ms());
        println!("   Device usage:");
        println!("   - GPU operations: {}", stats.gpu_operations);
        println!("   - CPU operations: {}", stats.cpu_operations);
        println!("   - GPU efficiency: {:.1}%", stats.gpu_efficiency * 100.0);
        println!("   - Fallback count: {}", stats.fallback_count);
        println!("   Memory tracking:");
        println!("   - Memory used: {:.2} MB ({} bytes)", 
            stats.memory_used_bytes as f64 / (1024.0 * 1024.0), 
            stats.memory_used_bytes);
        println!("   - Memory total: {:.2} MB ({} bytes)", 
            stats.memory_total_bytes as f64 / (1024.0 * 1024.0),
            stats.memory_total_bytes);
        println!("   - Memory utilization: {:.1}%", 
            (stats.memory_used_bytes as f64 / stats.memory_total_bytes as f64) * 100.0);
        
        // Check GPU effectiveness
        if stats.is_gpu_effective() {
            println!("   ✅ GPU is being used effectively");
        } else if stats.gpu_operations == 0 {
            println!("   ℹ️  CPU-only operations (as expected)");
        } else {
            println!("   ⚠️  Low GPU efficiency detected");
        }
    }

    println!("\n🔄 Demonstrating statistics reset...");
    quantizer.reset_stats();
    
    if let Some(stats) = quantizer.get_stats() {
        println!("   After reset: {}", stats.summary());
        assert_eq!(stats.total_operations, 0);
        println!("   ✅ Statistics reset successfully");
    }

    // Test platform-specific features
    println!("\n🏗️  Platform-specific kernel information:");
    #[cfg(target_arch = "x86_64")]
    {
        println!("   Architecture: x86_64");
        #[cfg(feature = "avx2")]
        {
            if std::is_x86_feature_detected!("avx2") {
                println!("   ✅ AVX2 support detected and enabled");
            } else {
                println!("   ⚠️  AVX2 feature enabled but not detected at runtime");
            }
        }
        #[cfg(not(feature = "avx2"))]
        {
            println!("   ℹ️  AVX2 feature not enabled, using fallback kernel");
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        println!("   Architecture: aarch64");
        #[cfg(feature = "neon")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                println!("   ✅ NEON support detected and enabled");
            } else {
                println!("   ⚠️  NEON feature enabled but not detected at runtime");
            }
        }
        #[cfg(not(feature = "neon"))]
        {
            println!("   ℹ️  NEON feature not enabled, using fallback kernel");
        }
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        println!("   Architecture: {} (using fallback kernel)", std::env::consts::ARCH);
    }

    println!("\n=== Device Stats Demo Complete ===");
    Ok(())
}