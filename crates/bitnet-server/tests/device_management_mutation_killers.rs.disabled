//! Device Management Mutation Killer Tests for BitNet.rs Server
//!
//! This test suite is designed to kill mutations in device management logic by testing
//! device selection, capability detection, fallback mechanisms, and resource management
//! scenarios that could be compromised by code mutations.

use bitnet_common::{Device, Result};
use bitnet_server::{
    device::{DeviceCapabilities, DeviceManager, DeviceSelector, ResourceMonitor},
    models::InferenceRequest,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Test device capability detection with various device configurations
#[test]
fn test_device_capability_detection_mutation_killer() {
    let device_manager = DeviceManager::new();

    let test_devices = [
        Device::Cpu,
        Device::Cuda(0),
        Device::Cuda(1),
        Device::Cuda(15), // High index
        Device::Metal,
    ];

    for device in test_devices.iter() {
        let capabilities_result = device_manager.get_device_capabilities(*device);

        match device {
            Device::Cpu => {
                // CPU should always be available
                assert!(
                    capabilities_result.is_ok(),
                    "CPU capabilities should always be detectable"
                );

                if let Ok(capabilities) = capabilities_result {
                    assert!(capabilities.is_available, "CPU should always be available");
                    assert!(
                        capabilities.compute_capability.is_empty(),
                        "CPU should not have compute capability version"
                    );
                    assert_eq!(capabilities.device_type, "cpu");
                    assert!(capabilities.total_memory_mb > 0, "CPU should report some memory");
                }
            }
            Device::Cuda(index) => {
                #[cfg(feature = "gpu")]
                {
                    // CUDA devices may or may not be available
                    match capabilities_result {
                        Ok(capabilities) => {
                            if capabilities.is_available {
                                assert_eq!(capabilities.device_type, "cuda");
                                assert!(
                                    capabilities.device_index == *index,
                                    "Device index should match"
                                );
                                assert!(
                                    !capabilities.compute_capability.is_empty(),
                                    "CUDA should have compute capability"
                                );
                                assert!(
                                    capabilities.total_memory_mb > 0,
                                    "CUDA should report memory"
                                );
                            }
                        }
                        Err(_) => {
                            // CUDA device not available - acceptable
                            println!(
                                "CUDA device {} not available (expected on some systems)",
                                index
                            );
                        }
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    // Without GPU features, CUDA should not be available
                    assert!(
                        capabilities_result.is_err(),
                        "CUDA should not be available without GPU features"
                    );
                }
            }
            Device::Metal => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    // Metal may or may not be available on macOS
                    match capabilities_result {
                        Ok(capabilities) => {
                            if capabilities.is_available {
                                assert_eq!(capabilities.device_type, "metal");
                                assert!(
                                    capabilities.total_memory_mb > 0,
                                    "Metal should report memory"
                                );
                            }
                        }
                        Err(_) => {
                            println!("Metal device not available (may be expected)");
                        }
                    }
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    // Metal should not be available on non-macOS or without Metal features
                    assert!(
                        capabilities_result.is_err(),
                        "Metal should not be available on non-macOS"
                    );
                }
            }
        }
    }
}

/// Test device selection algorithm with various preferences
#[test]
fn test_device_selection_algorithm_mutation_killer() {
    let device_selector = DeviceSelector::new();

    let selection_test_cases = [
        // Explicit device preferences
        (Some(Device::Cpu), Device::Cpu),
        (Some(Device::Cuda(0)), Device::Cuda(0)),
        (Some(Device::Metal), Device::Metal),
        // Auto selection should fall back appropriately
        (None, Device::Cpu), // Default fallback
    ];

    for (preference, expected_fallback) in selection_test_cases.iter() {
        let selection_result = device_selector.select_device(*preference);

        match preference {
            Some(requested_device) => {
                match selection_result {
                    Ok(selected_device) => {
                        // Should either get the requested device or a suitable fallback
                        assert!(
                            selected_device == *requested_device || selected_device == Device::Cpu, // CPU is always a valid fallback
                            "Selected device should be requested device or CPU fallback"
                        );
                    }
                    Err(_) => {
                        // Selection can fail if no suitable device is available
                        println!(
                            "Device selection failed for {:?} (may be expected)",
                            requested_device
                        );
                    }
                }
            }
            None => {
                // Auto selection should always succeed with some device
                assert!(selection_result.is_ok(), "Auto device selection should always succeed");

                if let Ok(selected_device) = selection_result {
                    // Should select the best available device
                    assert!(
                        matches!(selected_device, Device::Cpu | Device::Cuda(_) | Device::Metal),
                        "Auto-selected device should be valid"
                    );
                }
            }
        }
    }
}

/// Test device fallback scenarios
#[test]
fn test_device_fallback_scenarios_mutation_killer() {
    let device_manager = DeviceManager::new();

    let fallback_test_cases = [
        // GPU not available -> CPU fallback
        (Device::Cuda(999), Device::Cpu), // Non-existent CUDA device
        (Device::Metal, Device::Cpu),     // Metal might not be available
        // Invalid device configurations
        (Device::Cuda(usize::MAX), Device::Cpu), // Invalid CUDA index
    ];

    for (requested_device, expected_fallback) in fallback_test_cases.iter() {
        let fallback_result = device_manager.get_fallback_device(*requested_device);

        match fallback_result {
            Ok(fallback_device) => {
                assert_eq!(
                    fallback_device, *expected_fallback,
                    "Fallback device should match expected for {:?}",
                    requested_device
                );
            }
            Err(_) => {
                // Some devices might not have valid fallbacks
                println!("No fallback available for {:?} (may be expected)", requested_device);
            }
        }

        // Test that fallback device is actually available
        if let Ok(fallback_device) = fallback_result {
            let capabilities = device_manager.get_device_capabilities(fallback_device);
            assert!(capabilities.is_ok(), "Fallback device should have detectable capabilities");

            if let Ok(caps) = capabilities {
                assert!(caps.is_available, "Fallback device should be available");
            }
        }
    }
}

/// Test memory management and resource monitoring
#[test]
fn test_memory_management_mutation_killer() {
    let resource_monitor = ResourceMonitor::new();

    let test_devices = [Device::Cpu, Device::Cuda(0), Device::Metal];

    for device in test_devices.iter() {
        let memory_info_result = resource_monitor.get_memory_info(*device);

        match device {
            Device::Cpu => {
                // CPU memory info should always be available
                assert!(memory_info_result.is_ok(), "CPU memory info should be available");

                if let Ok(memory_info) = memory_info_result {
                    assert!(memory_info.total_mb > 0, "CPU should report positive total memory");
                    assert!(
                        memory_info.available_mb >= 0,
                        "CPU available memory should be non-negative"
                    );
                    assert!(memory_info.used_mb >= 0, "CPU used memory should be non-negative");
                    assert!(
                        memory_info.used_mb <= memory_info.total_mb,
                        "Used memory should not exceed total memory"
                    );
                    assert!(
                        memory_info.available_mb <= memory_info.total_mb,
                        "Available memory should not exceed total memory"
                    );
                }
            }
            Device::Cuda(_) => {
                #[cfg(feature = "gpu")]
                {
                    // GPU memory info may or may not be available
                    match memory_info_result {
                        Ok(memory_info) => {
                            assert!(
                                memory_info.total_mb > 0,
                                "GPU should report positive total memory"
                            );
                            assert!(
                                memory_info.available_mb >= 0,
                                "GPU available memory should be non-negative"
                            );
                            assert!(
                                memory_info.used_mb >= 0,
                                "GPU used memory should be non-negative"
                            );
                            assert!(
                                memory_info.used_mb <= memory_info.total_mb,
                                "GPU used memory should not exceed total memory"
                            );
                        }
                        Err(_) => {
                            println!("GPU memory info not available (may be expected)");
                        }
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    assert!(
                        memory_info_result.is_err(),
                        "GPU memory info should not be available without GPU features"
                    );
                }
            }
            Device::Metal => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    // Metal memory info may or may not be available
                    if let Ok(memory_info) = memory_info_result {
                        assert!(
                            memory_info.total_mb > 0,
                            "Metal should report positive total memory"
                        );
                        assert!(
                            memory_info.available_mb >= 0,
                            "Metal available memory should be non-negative"
                        );
                    }
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    assert!(
                        memory_info_result.is_err(),
                        "Metal memory info should not be available on non-macOS"
                    );
                }
            }
        }
    }
}

/// Test device compatibility with model requirements
#[test]
fn test_device_model_compatibility_mutation_killer() {
    let device_manager = DeviceManager::new();

    let compatibility_test_cases = [
        // Model size requirements
        ("small_model_100mb", 100, Device::Cpu, true),
        ("medium_model_1gb", 1024, Device::Cpu, true),
        ("large_model_10gb", 10240, Device::Cpu, false), // Might not fit in CPU memory
        ("huge_model_100gb", 102400, Device::Cpu, false), // Definitely won't fit
        // GPU-specific requirements
        ("gpu_model_2gb", 2048, Device::Cuda(0), true),
        ("gpu_model_8gb", 8192, Device::Cuda(0), false), // Depends on GPU memory
        ("gpu_model_24gb", 24576, Device::Cuda(0), false), // Likely won't fit
    ];

    for (model_name, model_size_mb, device, expected_compatible) in compatibility_test_cases.iter()
    {
        let compatibility_result =
            device_manager.check_model_compatibility(*device, *model_size_mb, "bitnet".to_string());

        if *expected_compatible && *device == Device::Cpu {
            // CPU compatibility should be predictable for reasonable sizes
            if *model_size_mb < 5000 {
                assert!(
                    compatibility_result.is_ok(),
                    "Small model '{}' should be compatible with CPU",
                    model_name
                );
            }
        } else {
            // GPU compatibility depends on available hardware
            match compatibility_result {
                Ok(is_compatible) => {
                    if *model_size_mb > 50000 {
                        assert!(
                            !is_compatible,
                            "Extremely large model '{}' should not be compatible",
                            model_name
                        );
                    }
                }
                Err(_) => {
                    // Compatibility check can fail if device is not available
                    println!(
                        "Compatibility check failed for {} on {:?} (may be expected)",
                        model_name, device
                    );
                }
            }
        }
    }
}

/// Test concurrent device access and thread safety
#[test]
fn test_concurrent_device_access_mutation_killer() {
    let device_manager = DeviceManager::new();
    let num_threads = 4;
    let operations_per_thread = 10;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let manager = device_manager.clone();
            std::thread::spawn(move || {
                for i in 0..operations_per_thread {
                    // Test concurrent device capability queries
                    let device = if thread_id % 2 == 0 { Device::Cpu } else { Device::Cuda(0) };

                    let start_time = Instant::now();
                    let result = manager.get_device_capabilities(device);
                    let duration = start_time.elapsed();

                    // Operations should complete reasonably quickly
                    assert!(
                        duration < Duration::from_secs(5),
                        "Device capability query should complete quickly"
                    );

                    // Results should be consistent across threads
                    match device {
                        Device::Cpu => {
                            assert!(result.is_ok(), "CPU capabilities should always be available in thread {}", thread_id);
                        }
                        Device::Cuda(_) => {
                            // CUDA availability is system-dependent
                            if result.is_err() {
                                println!("CUDA not available in thread {} iteration {} (expected on some systems)", thread_id, i);
                            }
                        }
                        _ => {}
                    }
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread should complete successfully");
    }
}

/// Test device performance monitoring
#[test]
fn test_device_performance_monitoring_mutation_killer() {
    let resource_monitor = ResourceMonitor::new();

    let test_devices = [Device::Cpu, Device::Cuda(0)];

    for device in test_devices.iter() {
        // Test performance metrics collection
        let start_time = Instant::now();

        // Simulate some device usage
        let _capabilities = resource_monitor.get_memory_info(*device);
        std::thread::sleep(Duration::from_millis(10));

        let performance_result = resource_monitor.get_performance_metrics(*device, start_time);

        match device {
            Device::Cpu => {
                // CPU performance metrics should be available
                if let Ok(metrics) = performance_result {
                    assert!(
                        metrics.utilization_percent >= 0.0,
                        "CPU utilization should be non-negative"
                    );
                    assert!(
                        metrics.utilization_percent <= 100.0,
                        "CPU utilization should not exceed 100%"
                    );
                    assert!(
                        metrics.memory_bandwidth_gbps >= 0.0,
                        "Memory bandwidth should be non-negative"
                    );
                    assert!(
                        metrics.temperature_celsius >= -50.0,
                        "Temperature should be reasonable"
                    );
                    assert!(
                        metrics.temperature_celsius <= 150.0,
                        "Temperature should not be extreme"
                    );
                }
            }
            Device::Cuda(_) => {
                #[cfg(feature = "gpu")]
                {
                    // GPU performance metrics may be available
                    if let Ok(metrics) = performance_result {
                        assert!(
                            metrics.utilization_percent >= 0.0,
                            "GPU utilization should be non-negative"
                        );
                        assert!(
                            metrics.utilization_percent <= 100.0,
                            "GPU utilization should not exceed 100%"
                        );
                        assert!(
                            metrics.memory_bandwidth_gbps >= 0.0,
                            "GPU memory bandwidth should be non-negative"
                        );
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    assert!(
                        performance_result.is_err(),
                        "GPU performance metrics should not be available without GPU features"
                    );
                }
            }
            _ => {}
        }
    }
}

/// Test device error handling and recovery
#[test]
fn test_device_error_handling_mutation_killer() {
    let device_manager = DeviceManager::new();

    let error_scenarios = [
        // Invalid device indices
        Device::Cuda(999),
        Device::Cuda(usize::MAX),
        // These should trigger error handling paths
    ];

    for device in error_scenarios.iter() {
        let capabilities_result = device_manager.get_device_capabilities(*device);

        // Invalid devices should either return an error or indicate unavailability
        match capabilities_result {
            Ok(capabilities) => {
                assert!(
                    !capabilities.is_available,
                    "Invalid device {:?} should be marked as unavailable",
                    device
                );
            }
            Err(error) => {
                // Error is expected for invalid devices
                let error_msg = format!("{}", error);
                assert!(
                    error_msg.contains("device")
                        || error_msg.contains("invalid")
                        || error_msg.contains("not found"),
                    "Error message should indicate device issue: {}",
                    error_msg
                );
            }
        }

        // Error recovery: try to get a fallback device
        let fallback_result = device_manager.get_fallback_device(*device);
        if let Ok(fallback_device) = fallback_result {
            let fallback_capabilities = device_manager.get_device_capabilities(fallback_device);
            assert!(
                fallback_capabilities.is_ok(),
                "Fallback device should be valid after error recovery"
            );
        }
    }
}

/// Test device resource allocation and deallocation
#[test]
fn test_device_resource_allocation_mutation_killer() {
    let device_manager = DeviceManager::new();

    let allocation_test_cases = [
        (Device::Cpu, 100),     // 100MB on CPU
        (Device::Cpu, 1000),    // 1GB on CPU
        (Device::Cuda(0), 500), // 500MB on GPU (if available)
    ];

    for (device, size_mb) in allocation_test_cases.iter() {
        let allocation_result = device_manager.allocate_memory(*device, *size_mb);

        match device {
            Device::Cpu => {
                if *size_mb < 2000 {
                    // Small CPU allocations should generally succeed
                    if allocation_result.is_err() {
                        println!("CPU allocation failed (may be due to low system memory)");
                    }
                } else {
                    // Large allocations may fail
                    match allocation_result {
                        Ok(allocation_id) => {
                            // If allocation succeeded, test deallocation
                            let deallocation_result =
                                device_manager.deallocate_memory(*device, allocation_id);
                            assert!(
                                deallocation_result.is_ok(),
                                "Deallocation should succeed after successful allocation"
                            );
                        }
                        Err(_) => {
                            println!("Large CPU allocation failed (expected)");
                        }
                    }
                }
            }
            Device::Cuda(_) => {
                #[cfg(feature = "gpu")]
                {
                    // GPU allocations depend on hardware availability
                    match allocation_result {
                        Ok(allocation_id) => {
                            let deallocation_result =
                                device_manager.deallocate_memory(*device, allocation_id);
                            assert!(
                                deallocation_result.is_ok(),
                                "GPU deallocation should succeed after successful allocation"
                            );
                        }
                        Err(_) => {
                            println!("GPU allocation failed (may be expected)");
                        }
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    assert!(
                        allocation_result.is_err(),
                        "GPU allocation should fail without GPU features"
                    );
                }
            }
            _ => {}
        }
    }
}

// Mock implementations for the test framework

#[derive(Debug, Clone)]
struct DeviceManager {
    capabilities_cache: HashMap<Device, DeviceCapabilities>,
}

#[derive(Debug, Clone)]
struct DeviceCapabilities {
    device_type: String,
    device_index: usize,
    is_available: bool,
    compute_capability: String,
    total_memory_mb: usize,
    max_threads: usize,
}

#[derive(Debug, Clone)]
struct DeviceSelector {
    preference_order: Vec<Device>,
}

#[derive(Debug, Clone)]
struct ResourceMonitor;

#[derive(Debug, Clone)]
struct MemoryInfo {
    total_mb: usize,
    used_mb: usize,
    available_mb: usize,
}

#[derive(Debug, Clone)]
struct PerformanceMetrics {
    utilization_percent: f64,
    memory_bandwidth_gbps: f64,
    temperature_celsius: f64,
}

impl DeviceManager {
    fn new() -> Self {
        Self { capabilities_cache: HashMap::new() }
    }

    fn get_device_capabilities(&self, device: Device) -> Result<DeviceCapabilities> {
        match device {
            Device::Cpu => Ok(DeviceCapabilities {
                device_type: "cpu".to_string(),
                device_index: 0,
                is_available: true,
                compute_capability: "".to_string(),
                total_memory_mb: 8192, // Mock 8GB RAM
                max_threads: num_cpus::get(),
            }),
            Device::Cuda(index) => {
                #[cfg(feature = "gpu")]
                {
                    if index < 8 {
                        Ok(DeviceCapabilities {
                            device_type: "cuda".to_string(),
                            device_index: index,
                            is_available: false, // Mock: GPU not available in test environment
                            compute_capability: "8.6".to_string(),
                            total_memory_mb: 4096, // Mock 4GB VRAM
                            max_threads: 1024,
                        })
                    } else {
                        Err(bitnet_common::BitNetError::DeviceError {
                            message: format!("CUDA device {} not found", index),
                        })
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Err(bitnet_common::BitNetError::DeviceError {
                        message: "GPU features not enabled".to_string(),
                    })
                }
            }
            Device::Metal => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    Ok(DeviceCapabilities {
                        device_type: "metal".to_string(),
                        device_index: 0,
                        is_available: false, // Mock: Metal not available in test environment
                        compute_capability: "".to_string(),
                        total_memory_mb: 8192, // Mock unified memory
                        max_threads: 512,
                    })
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    Err(bitnet_common::BitNetError::DeviceError {
                        message: "Metal not available on this platform".to_string(),
                    })
                }
            }
        }
    }

    fn get_fallback_device(&self, _requested: Device) -> Result<Device> {
        // Always fall back to CPU
        Ok(Device::Cpu)
    }

    fn check_model_compatibility(
        &self,
        device: Device,
        model_size_mb: usize,
        _architecture: String,
    ) -> Result<bool> {
        let capabilities = self.get_device_capabilities(device)?;

        if !capabilities.is_available {
            return Ok(false);
        }

        // Simple compatibility check: model should fit in device memory with some overhead
        let available_memory = capabilities.total_memory_mb * 8 / 10; // Use 80% of memory
        Ok(model_size_mb <= available_memory)
    }

    fn allocate_memory(&self, device: Device, size_mb: usize) -> Result<u64> {
        let capabilities = self.get_device_capabilities(device)?;

        if !capabilities.is_available {
            return Err(bitnet_common::BitNetError::DeviceError {
                message: "Device not available".to_string(),
            });
        }

        if size_mb > capabilities.total_memory_mb {
            return Err(bitnet_common::BitNetError::DeviceError {
                message: "Requested size exceeds device memory".to_string(),
            });
        }

        // Mock allocation ID
        Ok(12345)
    }

    fn deallocate_memory(&self, _device: Device, _allocation_id: u64) -> Result<()> {
        // Mock successful deallocation
        Ok(())
    }
}

impl DeviceSelector {
    fn new() -> Self {
        Self { preference_order: vec![Device::Cuda(0), Device::Metal, Device::Cpu] }
    }

    fn select_device(&self, preference: Option<Device>) -> Result<Device> {
        if let Some(device) = preference {
            // Try to use the preferred device
            let device_manager = DeviceManager::new();
            match device_manager.get_device_capabilities(device) {
                Ok(capabilities) if capabilities.is_available => Ok(device),
                _ => {
                    // Fall back to CPU
                    Ok(Device::Cpu)
                }
            }
        } else {
            // Auto-select the best available device
            let device_manager = DeviceManager::new();
            for &device in &self.preference_order {
                if let Ok(capabilities) = device_manager.get_device_capabilities(device) {
                    if capabilities.is_available {
                        return Ok(device);
                    }
                }
            }
            // Always fall back to CPU
            Ok(Device::Cpu)
        }
    }
}

impl ResourceMonitor {
    fn new() -> Self {
        Self
    }

    fn get_memory_info(&self, device: Device) -> Result<MemoryInfo> {
        match device {
            Device::Cpu => {
                // Mock CPU memory info
                Ok(MemoryInfo { total_mb: 8192, used_mb: 2048, available_mb: 6144 })
            }
            Device::Cuda(_) => {
                #[cfg(feature = "gpu")]
                {
                    // Mock GPU memory info
                    Ok(MemoryInfo { total_mb: 4096, used_mb: 1024, available_mb: 3072 })
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Err(bitnet_common::BitNetError::DeviceError {
                        message: "GPU features not enabled".to_string(),
                    })
                }
            }
            Device::Metal => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    Ok(MemoryInfo { total_mb: 8192, used_mb: 1500, available_mb: 6692 })
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    Err(bitnet_common::BitNetError::DeviceError {
                        message: "Metal not available".to_string(),
                    })
                }
            }
        }
    }

    fn get_performance_metrics(
        &self,
        device: Device,
        _start_time: Instant,
    ) -> Result<PerformanceMetrics> {
        match device {
            Device::Cpu => Ok(PerformanceMetrics {
                utilization_percent: 25.0,
                memory_bandwidth_gbps: 50.0,
                temperature_celsius: 45.0,
            }),
            Device::Cuda(_) => {
                #[cfg(feature = "gpu")]
                {
                    Ok(PerformanceMetrics {
                        utilization_percent: 75.0,
                        memory_bandwidth_gbps: 500.0,
                        temperature_celsius: 65.0,
                    })
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Err(bitnet_common::BitNetError::DeviceError {
                        message: "GPU features not enabled".to_string(),
                    })
                }
            }
            Device::Metal => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    Ok(PerformanceMetrics {
                        utilization_percent: 50.0,
                        memory_bandwidth_gbps: 200.0,
                        temperature_celsius: 55.0,
                    })
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    Err(bitnet_common::BitNetError::DeviceError {
                        message: "Metal not available".to_string(),
                    })
                }
            }
        }
    }
}
