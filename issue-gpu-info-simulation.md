# [Simulation] get_gpu_info function uses BITNET_GPU_FAKE environment variable instead of real GPU detection

## Problem Description

The `get_gpu_info` function in `crates/bitnet-kernels/src/gpu_utils.rs` uses the `BITNET_GPU_FAKE` environment variable to simulate GPU availability instead of performing actual runtime GPU detection. This simulation code can cause incorrect behavior in production environments and makes testing inconsistent with real deployment scenarios.

## Environment

- **File**: `crates/bitnet-kernels/src/gpu_utils.rs`
- **Function**: `get_gpu_info`
- **Environment Variables**: `BITNET_GPU_FAKE`, `BITNET_STRICT_NO_FAKE_GPU`
- **Crate**: `bitnet-kernels`
- **Related Components**: GPU device detection, CUDA/ROCm/Metal availability

## Current Implementation Analysis

The function prioritizes environment variable simulation over real detection:

```rust
pub fn get_gpu_info() -> GpuInfo {
    if let Ok(fake) = env::var("BITNET_GPU_FAKE") {
        if env::var("BITNET_STRICT_NO_FAKE_GPU").as_deref() == Ok("1") {
            panic!(
                "BITNET_GPU_FAKE is set but strict mode forbids fake GPU (BITNET_STRICT_NO_FAKE_GPU=1)"
            );
        }
        let lower = fake.to_lowercase();
        return GpuInfo {
            cuda: lower.contains("cuda"),
            cuda_version: None,
            metal: lower.contains("metal"),
            rocm: lower.contains("rocm"),
            rocm_version: None,
            wgpu: lower.contains("wgpu")
                || lower.contains("cuda")
                || lower.contains("rocm")
                || lower.contains("metal"),
        };
    }

    // Real GPU detection code follows...
}
```

## Root Cause Analysis

1. **Development Convenience Over Production Safety**: Environment variable simulation takes precedence over real detection
2. **Inconsistent Testing**: Tests may pass with fake GPU but fail in real deployment
3. **Production Risk**: Accidental environment variable setting can cause incorrect GPU detection
4. **No Real Hardware Validation**: Simulation doesn't validate actual GPU capabilities or availability
5. **Debugging Confusion**: Developers may unknowingly rely on fake GPU info when diagnosing issues

## Impact Assessment

**Severity**: Medium-High - Production Reliability & Testing Accuracy
**Affected Components**:
- GPU device selection and initialization
- CUDA/ROCm kernel availability detection
- Performance optimization based on GPU capabilities
- Production deployment reliability

**Production Risks**:
- Incorrect GPU selection leading to runtime failures
- Performance degradation from wrong device detection
- False positive GPU availability in CI/testing environments
- Inconsistent behavior between development and production

## Proposed Solution

### Primary Solution: Move Simulation to Test Utilities

Remove simulation from production code and move to test-specific utilities:

```rust
// In crates/bitnet-kernels/src/gpu_utils.rs
pub fn get_gpu_info() -> GpuInfo {
    // Remove environment variable simulation entirely
    // Only perform real hardware detection

    detect_real_gpu_info()
}

fn detect_real_gpu_info() -> GpuInfo {
    let mut gpu_info = GpuInfo::default();

    // CUDA detection
    #[cfg(feature = "cuda")]
    {
        gpu_info.cuda = detect_cuda_availability();
        if gpu_info.cuda {
            gpu_info.cuda_version = get_cuda_version();
        }
    }

    // ROCm detection
    #[cfg(feature = "rocm")]
    {
        gpu_info.rocm = detect_rocm_availability();
        if gpu_info.rocm {
            gpu_info.rocm_version = get_rocm_version();
        }
    }

    // Metal detection (macOS)
    #[cfg(target_os = "macos")]
    {
        gpu_info.metal = detect_metal_availability();
    }

    // WGPU detection
    gpu_info.wgpu = gpu_info.cuda || gpu_info.rocm || gpu_info.metal || detect_wgpu_availability();

    gpu_info
}

fn detect_cuda_availability() -> bool {
    // Query CUDA runtime directly
    use std::process::Command;

    Command::new("nvidia-smi")
        .arg("--query-gpu=gpu_name")
        .arg("--format=csv,noheader")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn detect_rocm_availability() -> bool {
    use std::process::Command;

    Command::new("rocm-smi")
        .arg("--showid")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn detect_metal_availability() -> bool {
    // Use Metal framework APIs to detect availability
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .output()
            .map(|output| output.status.success() &&
                String::from_utf8_lossy(&output.stdout).contains("Metal"))
            .unwrap_or(false)
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

fn get_cuda_version() -> Option<String> {
    use std::process::Command;

    Command::new("nvcc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .and_then(|s| extract_cuda_version(&s))
            } else {
                None
            }
        })
}

fn get_rocm_version() -> Option<String> {
    use std::process::Command;

    Command::new("hipcc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .and_then(|s| extract_rocm_version(&s))
            } else {
                None
            }
        })
}
```

### Test-Specific GPU Simulation

Move simulation to test utilities:

```rust
// In crates/bitnet-test-utils/src/gpu.rs
pub struct MockGpuInfo {
    cuda: bool,
    cuda_version: Option<String>,
    metal: bool,
    rocm: bool,
    rocm_version: Option<String>,
    wgpu: bool,
}

impl MockGpuInfo {
    pub fn cuda_only() -> Self {
        Self {
            cuda: true,
            cuda_version: Some("12.0".to_string()),
            metal: false,
            rocm: false,
            rocm_version: None,
            wgpu: true,
        }
    }

    pub fn cpu_only() -> Self {
        Self {
            cuda: false,
            cuda_version: None,
            metal: false,
            rocm: false,
            rocm_version: None,
            wgpu: false,
        }
    }

    pub fn into_gpu_info(self) -> GpuInfo {
        GpuInfo {
            cuda: self.cuda,
            cuda_version: self.cuda_version,
            metal: self.metal,
            rocm: self.rocm,
            rocm_version: self.rocm_version,
            wgpu: self.wgpu,
        }
    }
}

// For testing with dependency injection
pub trait GpuInfoProvider {
    fn get_gpu_info(&self) -> GpuInfo;
}

pub struct RealGpuInfoProvider;

impl GpuInfoProvider for RealGpuInfoProvider {
    fn get_gpu_info(&self) -> GpuInfo {
        bitnet_kernels::gpu_utils::get_gpu_info()
    }
}

pub struct MockGpuInfoProvider {
    info: GpuInfo,
}

impl MockGpuInfoProvider {
    pub fn new(info: GpuInfo) -> Self {
        Self { info }
    }
}

impl GpuInfoProvider for MockGpuInfoProvider {
    fn get_gpu_info(&self) -> GpuInfo {
        self.info.clone()
    }
}
```

### Alternative: Configuration-Based Detection

If some level of override is needed for specific deployment scenarios:

```rust
pub fn get_gpu_info() -> GpuInfo {
    // Check for explicit configuration override (not environment variable)
    if let Some(config_file) = get_gpu_config_file() {
        if let Ok(gpu_info) = load_gpu_info_from_config(&config_file) {
            log::info!("Using GPU info from configuration file: {:?}", config_file);
            return gpu_info;
        }
    }

    // Always perform real detection as default
    detect_real_gpu_info()
}

fn get_gpu_config_file() -> Option<PathBuf> {
    // Look for explicit configuration file, not environment variables
    let config_paths = [
        "/etc/bitnet/gpu_config.toml",
        "~/.config/bitnet/gpu_config.toml",
        "./gpu_config.toml",
    ];

    config_paths
        .iter()
        .map(|p| PathBuf::from(p))
        .find(|path| path.exists())
}
```

## Implementation Plan

### Phase 1: Real GPU Detection Implementation
- [ ] Implement robust CUDA detection using nvidia-ml-py or CUDA runtime APIs
- [ ] Add ROCm detection using ROCm SMI or HIP runtime
- [ ] Implement Metal detection for macOS using Metal Performance Shaders
- [ ] Add WGPU detection using wgpu-core device enumeration

### Phase 2: Remove Simulation from Production Code
- [ ] Remove `BITNET_GPU_FAKE` environment variable handling
- [ ] Remove `BITNET_STRICT_NO_FAKE_GPU` checks
- [ ] Update function to only perform real hardware detection
- [ ] Add comprehensive error handling for detection failures

### Phase 3: Test Utilities Development
- [ ] Create test-specific GPU simulation utilities
- [ ] Implement dependency injection patterns for testable GPU detection
- [ ] Add mock GPU providers with configurable capabilities
- [ ] Update existing tests to use test utilities instead of environment variables

### Phase 4: Integration & Validation
- [ ] Test real GPU detection on various hardware configurations
- [ ] Validate behavior on systems without GPUs
- [ ] Performance benchmark detection overhead
- [ ] Add comprehensive integration tests

## Testing Strategy

### Real Hardware Testing
```rust
#[test]
fn test_real_gpu_detection() {
    let gpu_info = get_gpu_info();

    // Test should work regardless of actual hardware
    // but should never return simulation data
    assert!(gpu_info.cuda || !gpu_info.cuda); // Always valid

    if gpu_info.cuda {
        // If CUDA detected, validate it's actually available
        assert!(validate_cuda_actually_works());
    }
}

#[test]
fn test_no_environment_variable_override() {
    // Ensure environment variables don't affect detection
    std::env::set_var("BITNET_GPU_FAKE", "cuda");

    let gpu_info = get_gpu_info();

    // Should detect real hardware, not fake
    assert_eq!(gpu_info.cuda, actual_cuda_available());

    std::env::remove_var("BITNET_GPU_FAKE");
}
```

### Mock Testing with Test Utilities
```rust
#[test]
fn test_with_mock_gpu() {
    use bitnet_test_utils::gpu::{MockGpuInfoProvider, MockGpuInfo};

    let mock_provider = MockGpuInfoProvider::new(
        MockGpuInfo::cuda_only().into_gpu_info()
    );

    let gpu_info = mock_provider.get_gpu_info();
    assert!(gpu_info.cuda);
    assert!(!gpu_info.rocm);
}
```

### Performance Testing
```rust
#[test]
fn test_gpu_detection_performance() {
    let start = Instant::now();
    let _gpu_info = get_gpu_info();
    let duration = start.elapsed();

    // GPU detection should be reasonably fast
    assert!(duration < Duration::from_millis(500));
}
```

## Related Issues/PRs

- GPU device selection and initialization reliability
- Testing framework improvements for GPU-dependent code
- Production deployment GPU availability validation
- CUDA/ROCm/Metal runtime integration

## Acceptance Criteria

- [ ] `get_gpu_info` performs only real hardware detection
- [ ] No environment variable simulation in production code paths
- [ ] Robust detection for CUDA, ROCm, Metal, and WGPU
- [ ] Test utilities provide controllable GPU simulation for testing
- [ ] Performance impact of real detection is acceptable (<500ms)
- [ ] Function works correctly on systems without GPUs
- [ ] Error handling gracefully manages detection failures
- [ ] Integration tests validate detection accuracy
- [ ] All existing functionality preserved through dependency injection
- [ ] Documentation updated to reflect real detection behavior

## Notes

This change improves production reliability by ensuring GPU detection reflects actual hardware capabilities. The simulation functionality should be moved to test utilities where it belongs, allowing developers to write robust tests while ensuring production deployments use real hardware detection.

Consider implementing caching for GPU detection results since hardware configuration doesn't change during runtime, which can improve performance for repeated calls.