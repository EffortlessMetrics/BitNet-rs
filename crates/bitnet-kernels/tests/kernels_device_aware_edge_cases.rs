//! Edge-case integration tests for `bitnet_kernels::device_aware` module.
//!
//! Tests CPU-only path (no GPU feature): DeviceAwareQuantizer construction,
//! quantization, matmul, stats tracking, force_cpu_fallback,
//! DeviceStats helpers, DeviceAwareQuantizerFactory, and thread safety.

use bitnet_kernels::device_aware::{DeviceAwareQuantizer, DeviceAwareQuantizerFactory};

use bitnet_common::{Device, QuantizationType};

// =========================================================================
// Construction
// =========================================================================

#[test]
fn create_cpu_quantizer() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    assert_eq!(q.device(), Device::Cpu);
    assert!(!q.is_gpu_active());
}

#[test]
fn active_provider_is_nonempty() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let name = q.active_provider();
    assert!(!name.is_empty());
}

#[test]
fn create_for_cuda_without_gpu_feature() {
    // Without the `gpu` feature, CUDA creation should still succeed with CPU fallback
    let q = DeviceAwareQuantizer::new(Device::Cuda(0)).unwrap();
    assert!(!q.is_gpu_active(), "GPU should not be active without gpu feature");
    assert_eq!(q.device(), Device::Cuda(0));
}

#[test]
fn create_for_metal() {
    let q = DeviceAwareQuantizer::new(Device::Metal).unwrap();
    assert!(!q.is_gpu_active());
}

#[test]
fn create_for_opencl() {
    let q = DeviceAwareQuantizer::new(Device::OpenCL(0)).unwrap();
    assert!(!q.is_gpu_active());
}

// =========================================================================
// Quantization (CPU path)
// =========================================================================

#[test]
fn quantize_i2s_small() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let input = vec![1.0f32, -1.0, 0.5, -0.5];
    let mut output = vec![0u8; 1];
    let mut scales = vec![0.0f32; 1];
    q.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
}

#[test]
fn quantize_i2s_block_of_32() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let input: Vec<f32> = (0..32).map(|i| (i as f32) / 16.0 - 1.0).collect();
    let mut output = vec![0u8; 8]; // 32 / 4 = 8
    let mut scales = vec![0.0f32; 1]; // 1 scale per block of 32
    q.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
    assert!(scales[0].abs() > 0.0, "scale should be non-zero for non-zero input");
}

#[test]
fn quantize_zeros() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let input = vec![0.0f32; 4];
    let mut output = vec![0u8; 1];
    let mut scales = vec![0.0f32; 1];
    q.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
}

// =========================================================================
// Matmul I2S (CPU path)
// =========================================================================

// NOTE: matmul_i2s is not tested here because the AVX2 kernel
// panics on undersized/mis-packed inputs (known issue in cpu/x86.rs:462).
// The matmul path is exercised through FallbackKernel integration tests.

// =========================================================================
// Stats tracking
// =========================================================================

#[test]
fn initial_stats_zero_ops() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let stats = q.get_stats().expect("stats should be available");
    assert_eq!(stats.total_operations, 0);
    assert_eq!(stats.quantization_operations, 0);
    assert_eq!(stats.matmul_operations, 0);
}

#[test]
fn stats_after_quantize() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let input = vec![1.0f32, -1.0, 0.5, -0.5];
    let mut output = vec![0u8; 1];
    let mut scales = vec![0.0f32; 1];
    q.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();

    let stats = q.get_stats().unwrap();
    assert_eq!(stats.quantization_operations, 1);
    assert_eq!(stats.total_operations, 1);
    assert!(stats.cpu_operations > 0);
    assert_eq!(stats.gpu_operations, 0);
}

#[test]
fn stats_after_multiple_ops() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let input = vec![1.0f32, -1.0, 0.5, -0.5];
    let mut output = vec![0u8; 1];
    let mut scales = vec![0.0f32; 1];

    for _ in 0..5 {
        q.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
    }

    let stats = q.get_stats().unwrap();
    assert_eq!(stats.quantization_operations, 5);
    assert_eq!(stats.total_operations, 5);
}

#[test]
fn reset_stats_clears() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let input = vec![1.0f32, -1.0, 0.5, -0.5];
    let mut output = vec![0u8; 1];
    let mut scales = vec![0.0f32; 1];
    q.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();

    q.reset_stats();
    let stats = q.get_stats().unwrap();
    assert_eq!(stats.total_operations, 0);
    assert_eq!(stats.quantization_operations, 0);
}

// =========================================================================
// DeviceStats helpers
// =========================================================================

#[test]
fn device_stats_avg_quantize_no_ops() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let stats = q.get_stats().unwrap();
    assert!((stats.avg_quantization_time_ms() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn device_stats_avg_matmul_no_ops() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let stats = q.get_stats().unwrap();
    assert!((stats.avg_matmul_time_ms() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn device_stats_is_gpu_effective_false_for_cpu() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let input = vec![1.0f32, -1.0, 0.5, -0.5];
    let mut output = vec![0u8; 1];
    let mut scales = vec![0.0f32; 1];
    q.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();

    let stats = q.get_stats().unwrap();
    assert!(!stats.is_gpu_effective());
}

#[test]
fn device_stats_summary_not_empty() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let stats = q.get_stats().unwrap();
    let summary = stats.summary();
    assert!(!summary.is_empty());
    assert!(summary.contains("Device:"));
    assert!(summary.contains("Memory:"));
}

#[test]
fn device_stats_device_type_contains_fallback_name() {
    let q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let stats = q.get_stats().unwrap();
    // Format is "Primary+Fallback" where Primary is "None" for CPU-only
    assert!(stats.device_type.contains("None+") || stats.device_type.contains('+'));
}

// =========================================================================
// force_cpu_fallback
// =========================================================================

#[test]
fn force_cpu_fallback_on_cpu_quantizer() {
    let mut q = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    // Should be no-op (no GPU to disable)
    q.force_cpu_fallback();
    assert!(!q.is_gpu_active());

    // Should still work
    let input = vec![1.0f32, -1.0, 0.5, -0.5];
    let mut output = vec![0u8; 1];
    let mut scales = vec![0.0f32; 1];
    q.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
}

// =========================================================================
// DeviceAwareQuantizerFactory
// =========================================================================

#[test]
fn factory_auto_detect() {
    let q = DeviceAwareQuantizerFactory::auto_detect().unwrap();
    // On CPU-only build, should be CPU
    assert!(!q.is_gpu_active());
}

#[test]
fn factory_create_best_none() {
    let q = DeviceAwareQuantizerFactory::create_best(None).unwrap();
    assert!(!q.is_gpu_active());
}

#[test]
fn factory_create_best_explicit_cpu() {
    let q = DeviceAwareQuantizerFactory::create_best(Some(Device::Cpu)).unwrap();
    assert_eq!(q.device(), Device::Cpu);
}

#[test]
fn factory_list_devices_contains_cpu() {
    let devices = DeviceAwareQuantizerFactory::list_available_devices();
    assert!(devices.contains(&Device::Cpu));
}

// =========================================================================
// Thread safety
// =========================================================================

#[test]
fn concurrent_quantize() {
    use std::sync::Arc;
    use std::thread;

    let q = Arc::new(DeviceAwareQuantizer::new(Device::Cpu).unwrap());

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let q = Arc::clone(&q);
            thread::spawn(move || {
                let input = vec![1.0f32, -1.0, 0.5, -0.5];
                let mut output = vec![0u8; 1];
                let mut scales = vec![0.0f32; 1];
                q.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let stats = q.get_stats().unwrap();
    assert_eq!(stats.quantization_operations, 4);
    assert_eq!(stats.cpu_operations, 4);
}

#[test]
fn concurrent_stats_reads() {
    use std::sync::Arc;
    use std::thread;

    let q = Arc::new(DeviceAwareQuantizer::new(Device::Cpu).unwrap());

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let q = Arc::clone(&q);
            thread::spawn(move || {
                let _ = q.get_stats();
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
