# [SIMULATION] Replace Fake GPU Environment Variable with Real GPU Detection

## Problem Description

The `get_gpu_info` function in `crates/bitnet-kernels/src/gpu_utils.rs` uses the `BITNET_GPU_FAKE` environment variable to simulate GPU availability instead of implementing real GPU detection. This simulation approach prevents the system from accurately detecting actual GPU hardware and capabilities, leading to incorrect device selection and potential runtime failures when real GPU operations are attempted.

## Environment

- **File**: `crates/bitnet-kernels/src/gpu_utils.rs`
- **Function**: `get_gpu_info`
- **Crate**: `bitnet-kernels`
- **Affected Systems**: All GPU detection and device selection logic
- **Impact**: GPU availability detection, device management, kernel selection

## Current Implementation Issues

```rust
pub fn get_gpu_info() -> GpuInfo {
    if let Ok(fake) = env::var("BITNET_GPU_FAKE") {
        if env::var("BITNET_STRICT_NO_FAKE_GPU").as_deref() == Ok("1") {
            panic!("BITNET_GPU_FAKE is set but strict mode forbids fake GPU");
        }
        let lower = fake.to_lowercase();
        return GpuInfo {
            cuda: lower.contains("cuda"),
            cuda_version: None,
            metal: lower.contains("metal"),
            rocm: lower.contains("rocm"),
            rocm_version: None,
            wgpu: lower.contains("wgpu") || /* ... */,
        };
    }
    // ... rest of implementation
}
```

## Root Cause Analysis

### Simulation vs Real Detection
1. **Fake GPU Environment**: Relies on environment variable simulation instead of hardware detection
2. **No Hardware Validation**: Cannot verify actual GPU presence or capabilities
3. **Incorrect Runtime Behavior**: May report GPU availability when none exists
4. **Testing Confusion**: Simulation mixed with production code creates maintenance issues
5. **Version Information Missing**: Fake detection doesn't provide real version information

### Missing Production Features
- **CUDA Runtime Query**: No actual CUDA device enumeration
- **ROCm Detection**: Missing AMD GPU detection and version querying
- **Metal Capability Query**: Basic macOS detection without Metal capability validation
- **Device Memory Information**: No actual GPU memory detection
- **Multi-GPU Support**: Missing detection of multiple GPU devices

## Impact Assessment

- **Severity**: Medium-High - Affects core GPU functionality and device selection
- **Affected Components**: All GPU inference operations, device management, kernel selection
- **User Impact**: Incorrect GPU detection leading to runtime failures or suboptimal performance
- **Development Impact**: Confusion between test simulation and production behavior

## Proposed Solution

Implement comprehensive real GPU detection with proper hardware querying:

### 1. CUDA Detection and Enumeration
```rust
use std::process::Command;
use std::ffi::CString;

#[cfg(feature = "cuda")]
mod cuda_detection {
    use cuda_runtime_sys::*;

    pub fn detect_cuda_devices() -> Result<Vec<CudaDeviceInfo>> {
        unsafe {
            let mut device_count: i32 = 0;
            let result = cudaGetDeviceCount(&mut device_count);

            if result != cudaError_t::cudaSuccess {
                return Ok(Vec::new()); // No CUDA devices
            }

            let mut devices = Vec::new();

            for device_id in 0..device_count {
                let device_info = query_cuda_device_info(device_id)?;
                devices.push(device_info);
            }

            Ok(devices)
        }
    }

    unsafe fn query_cuda_device_info(device_id: i32) -> Result<CudaDeviceInfo> {
        let mut props: cudaDeviceProp = std::mem::zeroed();
        let result = cudaGetDeviceProperties(&mut props, device_id);

        if result != cudaError_t::cudaSuccess {
            return Err(BitNetError::CudaError {
                message: format!("Failed to query device {}", device_id),
            });
        }

        let name = std::ffi::CStr::from_ptr(props.name.as_ptr())
            .to_string_lossy()
            .to_string();

        Ok(CudaDeviceInfo {
            device_id,
            name,
            compute_capability: (props.major, props.minor),
            total_memory: props.totalGlobalMem,
            multiprocessor_count: props.multiProcessorCount,
            max_threads_per_block: props.maxThreadsPerBlock,
            max_threads_per_multiprocessor: props.maxThreadsPerMultiProcessor,
            memory_clock_rate: props.memoryClockRate,
            memory_bus_width: props.memoryBusWidth,
        })
    }

    pub fn get_cuda_runtime_version() -> Option<(i32, i32)> {
        unsafe {
            let mut version: i32 = 0;
            let result = cudaRuntimeGetVersion(&mut version);

            if result == cudaError_t::cudaSuccess {
                let major = version / 1000;
                let minor = (version % 1000) / 10;
                Some((major, minor))
            } else {
                None
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod cuda_detection {
    pub fn detect_cuda_devices() -> Result<Vec<CudaDeviceInfo>> {
        Ok(Vec::new())
    }

    pub fn get_cuda_runtime_version() -> Option<(i32, i32)> {
        None
    }
}
```

### 2. ROCm/HIP Detection
```rust
mod rocm_detection {
    use std::process::Command;

    pub fn detect_rocm_devices() -> Result<Vec<RocmDeviceInfo>> {
        // Try to use rocm-smi command
        let output = Command::new("rocm-smi")
            .arg("--showid")
            .arg("--csv")
            .output();

        match output {
            Ok(output) if output.status.success() => {
                parse_rocm_smi_output(&output.stdout)
            }
            _ => {
                // Fallback to HIP runtime if available
                #[cfg(feature = "rocm")]
                {
                    detect_rocm_devices_hip()
                }
                #[cfg(not(feature = "rocm"))]
                {
                    Ok(Vec::new())
                }
            }
        }
    }

    fn parse_rocm_smi_output(output: &[u8]) -> Result<Vec<RocmDeviceInfo>> {
        let output_str = String::from_utf8_lossy(output);
        let mut devices = Vec::new();

        for line in output_str.lines().skip(1) { // Skip header
            if let Some(device_info) = parse_rocm_device_line(line) {
                devices.push(device_info);
            }
        }

        Ok(devices)
    }

    fn parse_rocm_device_line(line: &str) -> Option<RocmDeviceInfo> {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            let device_id = parts[0].trim().parse().ok()?;
            let name = parts[1].trim().to_string();

            Some(RocmDeviceInfo {
                device_id,
                name,
                memory_info: query_rocm_memory_info(device_id).ok(),
                compute_units: query_rocm_compute_units(device_id).ok(),
            })
        } else {
            None
        }
    }

    #[cfg(feature = "rocm")]
    fn detect_rocm_devices_hip() -> Result<Vec<RocmDeviceInfo>> {
        // Use HIP runtime API for direct device detection
        // This would require HIP runtime bindings
        Ok(Vec::new()) // Placeholder for HIP implementation
    }

    pub fn get_rocm_version() -> Option<String> {
        Command::new("rocm-smi")
            .arg("--showversion")
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    String::from_utf8(output.stdout)
                        .ok()
                        .and_then(|s| parse_rocm_version(&s))
                } else {
                    None
                }
            })
    }

    fn parse_rocm_version(output: &str) -> Option<String> {
        // Parse ROCm version from rocm-smi output
        for line in output.lines() {
            if line.contains("Version:") {
                return line.split(':').nth(1).map(|s| s.trim().to_string());
            }
        }
        None
    }
}
```

### 3. Metal Detection (macOS)
```rust
#[cfg(target_os = "macos")]
mod metal_detection {
    use std::process::Command;

    pub fn detect_metal_devices() -> Result<Vec<MetalDeviceInfo>> {
        // Use system_profiler to get GPU information on macOS
        let output = Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .arg("-json")
            .output();

        match output {
            Ok(output) if output.status.success() => {
                parse_metal_system_profiler(&output.stdout)
            }
            _ => {
                // Fallback to basic detection
                if is_metal_available() {
                    Ok(vec![MetalDeviceInfo::default()])
                } else {
                    Ok(Vec::new())
                }
            }
        }
    }

    fn parse_metal_system_profiler(output: &[u8]) -> Result<Vec<MetalDeviceInfo>> {
        // Parse JSON output from system_profiler
        // This would require JSON parsing to extract GPU information
        Ok(Vec::new()) // Placeholder for JSON parsing implementation
    }

    fn is_metal_available() -> bool {
        // Basic check for Metal availability
        std::env::var("METAL_DEVICE_WRAPPER_TYPE").is_err() // Not in simulator
            && Command::new("metal")
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
    }

    pub fn get_metal_version() -> Option<String> {
        Command::new("metal")
            .arg("--version")
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    String::from_utf8(output.stdout)
                        .ok()
                        .and_then(|s| s.lines().next().map(|l| l.to_string()))
                } else {
                    None
                }
            })
    }
}

#[cfg(not(target_os = "macos"))]
mod metal_detection {
    pub fn detect_metal_devices() -> Result<Vec<MetalDeviceInfo>> {
        Ok(Vec::new())
    }

    pub fn get_metal_version() -> Option<String> {
        None
    }
}
```

### 4. Comprehensive GPU Information System
```rust
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub cuda_devices: Vec<CudaDeviceInfo>,
    pub rocm_devices: Vec<RocmDeviceInfo>,
    pub metal_devices: Vec<MetalDeviceInfo>,
    pub cuda_version: Option<(i32, i32)>,
    pub rocm_version: Option<String>,
    pub metal_version: Option<String>,
    pub detection_time: std::time::SystemTime,
}

#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub device_id: i32,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_memory: usize,
    pub multiprocessor_count: i32,
    pub max_threads_per_block: i32,
    pub max_threads_per_multiprocessor: i32,
    pub memory_clock_rate: i32,
    pub memory_bus_width: i32,
}

#[derive(Debug, Clone)]
pub struct RocmDeviceInfo {
    pub device_id: u32,
    pub name: String,
    pub memory_info: Option<RocmMemoryInfo>,
    pub compute_units: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct MetalDeviceInfo {
    pub name: String,
    pub device_type: MetalDeviceType,
    pub memory_size: Option<u64>,
    pub max_threads_per_threadgroup: Option<u32>,
}

pub fn get_gpu_info() -> GpuInfo {
    let detection_start = std::time::SystemTime::now();

    let cuda_devices = cuda_detection::detect_cuda_devices()
        .unwrap_or_else(|e| {
            log::debug!("CUDA detection failed: {}", e);
            Vec::new()
        });

    let rocm_devices = rocm_detection::detect_rocm_devices()
        .unwrap_or_else(|e| {
            log::debug!("ROCm detection failed: {}", e);
            Vec::new()
        });

    let metal_devices = metal_detection::detect_metal_devices()
        .unwrap_or_else(|e| {
            log::debug!("Metal detection failed: {}", e);
            Vec::new()
        });

    let cuda_version = cuda_detection::get_cuda_runtime_version();
    let rocm_version = rocm_detection::get_rocm_version();
    let metal_version = metal_detection::get_metal_version();

    log::info!(
        "GPU detection completed: {} CUDA, {} ROCm, {} Metal devices",
        cuda_devices.len(),
        rocm_devices.len(),
        metal_devices.len()
    );

    GpuInfo {
        cuda_devices,
        rocm_devices,
        metal_devices,
        cuda_version,
        rocm_version,
        metal_version,
        detection_time: detection_start,
    }
}
```

### 5. Testing Infrastructure Separation
```rust
#[cfg(test)]
pub mod test_utils {
    use super::*;

    pub fn create_fake_gpu_info(config: &str) -> GpuInfo {
        let lower = config.to_lowercase();

        let cuda_devices = if lower.contains("cuda") {
            vec![CudaDeviceInfo {
                device_id: 0,
                name: "Fake CUDA Device".to_string(),
                compute_capability: (7, 5),
                total_memory: 8 * 1024 * 1024 * 1024, // 8GB
                multiprocessor_count: 72,
                max_threads_per_block: 1024,
                max_threads_per_multiprocessor: 1024,
                memory_clock_rate: 7000,
                memory_bus_width: 256,
            }]
        } else {
            Vec::new()
        };

        let rocm_devices = if lower.contains("rocm") {
            vec![RocmDeviceInfo {
                device_id: 0,
                name: "Fake ROCm Device".to_string(),
                memory_info: Some(RocmMemoryInfo { total: 8192, used: 0 }),
                compute_units: Some(64),
            }]
        } else {
            Vec::new()
        };

        let metal_devices = if lower.contains("metal") {
            vec![MetalDeviceInfo {
                name: "Fake Metal Device".to_string(),
                device_type: MetalDeviceType::DiscreteGpu,
                memory_size: Some(8 * 1024 * 1024 * 1024),
                max_threads_per_threadgroup: Some(1024),
            }]
        } else {
            Vec::new()
        };

        GpuInfo {
            cuda_devices,
            rocm_devices,
            metal_devices,
            cuda_version: if lower.contains("cuda") { Some((11, 0)) } else { None },
            rocm_version: if lower.contains("rocm") { Some("5.0.0".to_string()) } else { None },
            metal_version: if lower.contains("metal") { Some("3.0".to_string()) } else { None },
            detection_time: std::time::SystemTime::now(),
        }
    }
}
```

## Implementation Plan

### Phase 1: CUDA Detection Infrastructure
- [ ] Add CUDA runtime device enumeration
- [ ] Implement device capability querying
- [ ] Add CUDA version detection
- [ ] Create comprehensive device information structures

### Phase 2: ROCm Detection System
- [ ] Implement rocm-smi command parsing
- [ ] Add HIP runtime device detection (optional)
- [ ] Create ROCm device information structures
- [ ] Add ROCm version detection

### Phase 3: Metal Detection (macOS)
- [ ] Implement system_profiler parsing for GPU information
- [ ] Add Metal capability detection
- [ ] Create Metal device information structures
- [ ] Add Metal version detection

### Phase 4: Testing Infrastructure
- [ ] Move simulation logic to dedicated test utilities
- [ ] Create comprehensive test GPU configurations
- [ ] Add unit tests for detection logic
- [ ] Separate production and test code paths

### Phase 5: Integration and Optimization
- [ ] Integrate with existing device management
- [ ] Add caching for expensive detection operations
- [ ] Optimize detection performance and error handling
- [ ] Add comprehensive logging and debugging

## Testing Strategy

### Real Hardware Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_gpu_detection() {
        let gpu_info = get_gpu_info();

        // Basic validation of detection results
        assert!(gpu_info.detection_time <= std::time::SystemTime::now());

        // Log detected devices for manual verification
        println!("Detected {} CUDA devices", gpu_info.cuda_devices.len());
        println!("Detected {} ROCm devices", gpu_info.rocm_devices.len());
        println!("Detected {} Metal devices", gpu_info.metal_devices.len());
    }

    #[test]
    fn test_fake_gpu_for_testing() {
        let fake_info = test_utils::create_fake_gpu_info("cuda,rocm");

        assert_eq!(fake_info.cuda_devices.len(), 1);
        assert_eq!(fake_info.rocm_devices.len(), 1);
        assert_eq!(fake_info.metal_devices.len(), 0);
    }

    #[test]
    fn test_no_gpu_scenario() {
        // Test behavior when no GPUs are detected
        // This would run on systems without GPU hardware
    }
}
```

## BitNet.rs Integration Notes

### Feature Flag Integration
```rust
#[cfg(feature = "cuda")]
// CUDA-specific detection code

#[cfg(feature = "rocm")]
// ROCm-specific detection code

#[cfg(target_os = "macos")]
// Metal-specific detection code
```

### Device Management Integration
- Replace fake detection with real hardware enumeration
- Integrate with existing device selection and management
- Maintain backward compatibility with existing device APIs

## Dependencies

```toml
[dependencies]
# Optional CUDA runtime for direct device querying
cuda-runtime-sys = { version = "0.3", optional = true }

# Optional HIP runtime for ROCm detection
hip-runtime-sys = { version = "0.3", optional = true }

[features]
cuda = ["cuda-runtime-sys"]
rocm = ["hip-runtime-sys"]
```

## Acceptance Criteria

- [ ] Complete removal of `BITNET_GPU_FAKE` environment variable simulation
- [ ] Real CUDA device detection and enumeration
- [ ] ROCm device detection with version information
- [ ] Metal device detection on macOS systems
- [ ] Comprehensive device information including memory and capabilities
- [ ] Proper error handling for systems without GPU support
- [ ] Separated testing infrastructure for simulation scenarios
- [ ] Performance-optimized detection with result caching
- [ ] Integration with existing device management systems
- [ ] Full test coverage including real hardware scenarios

## Related Issues

- Device management and selection optimization
- GPU capability detection and utilization
- Multi-GPU support and device enumeration
- Hardware compatibility and feature detection

## Priority

**Medium-High** - Critical for accurate GPU detection and proper device utilization. Essential for production deployments where real GPU hardware detection is required.
