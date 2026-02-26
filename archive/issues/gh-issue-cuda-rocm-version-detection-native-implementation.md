# [GPU] Native CUDA/ROCm Version Detection via Runtime APIs

## Problem Description

The current GPU version detection in BitNet-rs relies on external command execution (`nvcc --version` and `rocm-smi --version`) which introduces fragility, security concerns, and deployment complexity. This approach fails in containerized environments, CI/CD pipelines, and systems where command-line tools are unavailable while GPU runtimes are present.

## Environment

- **Component**: `bitnet-kernels` crate
- **File**: `crates/bitnet-kernels/src/gpu_utils.rs`
- **Rust Version**: 1.90.0+ (2024 edition)
- **GPU Runtimes**: CUDA 11.8+, ROCm 5.0+
- **Target Platforms**: Linux, Windows (CUDA), Linux (ROCm)

## Current Implementation Analysis

### Problematic External Command Dependencies
```rust
/// Get CUDA version if available
fn get_cuda_version() -> Option<String> {
    Command::new("nvcc")  // External dependency - fragile
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() { String::from_utf8(output.stdout).ok() } else { None }
        })
        .and_then(|output| {
            // Brittle string parsing
            output.lines().find(|line| line.contains("release")).and_then(|line| {
                line.split("release")
                    .nth(1)
                    .and_then(|s| s.split(',').next())
                    .map(|s| s.trim().to_string())
            })
        })
}

/// Get ROCm version if available
fn get_rocm_version() -> Option<String> {
    Command::new("rocm-smi")  // External dependency - fragile
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() { String::from_utf8(output.stdout).ok() } else { None }
        })
        .and_then(|output| {
            // Brittle string parsing
            output
                .lines()
                .find(|line| line.contains("Version"))
                .and_then(|line| line.split(':').nth(1).map(|s| s.trim().to_string()))
        })
}
```

## Root Cause Analysis

1. **External Command Dependency**: Relies on `nvcc` and `rocm-smi` being in PATH
2. **Fragile String Parsing**: Output format changes can break detection
3. **Environment Assumptions**: Commands may be unavailable in production
4. **Security Concerns**: Executing external processes in production environments
5. **Performance Overhead**: Process spawning for simple version queries
6. **Deployment Complexity**: Requires additional tools beyond GPU drivers/runtimes

## Impact Assessment

**Severity**: High - GPU capability detection critical for device selection

**Affected Systems**:
- Docker containers without dev tools
- CI/CD environments
- Edge deployment scenarios
- Systems with runtime-only GPU installations

**User Impact**:
- GPU acceleration unavailable despite hardware support
- Fallback to CPU inference unnecessarily
- Deployment failures in production environments
- Inconsistent behavior across environments

## Proposed Solution

### Primary Approach: Native Runtime API Integration

Replace command execution with direct GPU runtime API calls:

```rust
use std::ffi::CStr;

// CUDA version detection via CUDA Runtime API
#[cfg(feature = "gpu")]
mod cuda_version {
    use std::os::raw::c_int;

    // CUDA Runtime API bindings
    extern "C" {
        fn cudaRuntimeGetVersion(runtime_version: *mut c_int) -> c_int;
        fn cudaDriverGetVersion(driver_version: *mut c_int) -> c_int;
    }

    /// Get CUDA version from runtime API
    pub fn get_cuda_version() -> Option<CudaVersion> {
        unsafe {
            let mut runtime_version: c_int = 0;
            let mut driver_version: c_int = 0;

            // Query CUDA Runtime version
            let runtime_result = cudaRuntimeGetVersion(&mut runtime_version);
            if runtime_result != 0 {
                return None; // CUDA not available
            }

            // Query CUDA Driver version
            let driver_result = cudaDriverGetVersion(&mut driver_version);
            if driver_result != 0 {
                return None; // Driver not available
            }

            Some(CudaVersion {
                runtime: parse_cuda_version(runtime_version),
                driver: parse_cuda_version(driver_version),
            })
        }
    }

    fn parse_cuda_version(version: c_int) -> (u32, u32) {
        let major = (version / 1000) as u32;
        let minor = ((version % 1000) / 10) as u32;
        (major, minor)
    }
}

// ROCm version detection via HIP Runtime API
#[cfg(feature = "gpu")]
mod rocm_version {
    use std::os::raw::c_int;

    extern "C" {
        fn hipRuntimeGetVersion(runtime_version: *mut c_int) -> c_int;
        fn hipDriverGetVersion(driver_version: *mut c_int) -> c_int;
    }

    /// Get ROCm version from HIP runtime API
    pub fn get_rocm_version() -> Option<RocmVersion> {
        unsafe {
            let mut runtime_version: c_int = 0;
            let mut driver_version: c_int = 0;

            let runtime_result = hipRuntimeGetVersion(&mut runtime_version);
            if runtime_result != 0 {
                return None;
            }

            let driver_result = hipDriverGetVersion(&mut driver_version);
            if driver_result != 0 {
                return None;
            }

            Some(RocmVersion {
                runtime: parse_rocm_version(runtime_version),
                driver: parse_rocm_version(driver_version),
            })
        }
    }

    fn parse_rocm_version(version: c_int) -> (u32, u32, u32) {
        let major = (version / 10000000) as u32;
        let minor = ((version / 100000) % 100) as u32;
        let patch = ((version / 1000) % 100) as u32;
        (major, minor, patch)
    }
}

/// Unified GPU version information
#[derive(Debug, Clone, PartialEq)]
pub enum GpuVersion {
    Cuda(CudaVersion),
    Rocm(RocmVersion),
    None,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CudaVersion {
    pub runtime: (u32, u32),  // (major, minor)
    pub driver: (u32, u32),   // (major, minor)
}

#[derive(Debug, Clone, PartialEq)]
pub struct RocmVersion {
    pub runtime: (u32, u32, u32),  // (major, minor, patch)
    pub driver: (u32, u32, u32),   // (major, minor, patch)
}

/// Get available GPU version information
pub fn get_gpu_version() -> GpuVersion {
    // Try CUDA first
    #[cfg(feature = "gpu")]
    if let Some(cuda_version) = cuda_version::get_cuda_version() {
        return GpuVersion::Cuda(cuda_version);
    }

    // Try ROCm if CUDA unavailable
    #[cfg(feature = "gpu")]
    if let Some(rocm_version) = rocm_version::get_rocm_version() {
        return GpuVersion::Rocm(rocm_version);
    }

    GpuVersion::None
}

/// Check minimum GPU version requirements
pub fn check_gpu_compatibility() -> Result<GpuCompatibility, GpuError> {
    match get_gpu_version() {
        GpuVersion::Cuda(cuda_version) => {
            let min_required = (11, 8);  // Minimum CUDA version
            if cuda_version.runtime >= min_required {
                Ok(GpuCompatibility::Cuda {
                    version: cuda_version,
                    features: query_cuda_capabilities()?,
                })
            } else {
                Err(GpuError::IncompatibleVersion {
                    found: format!("{}.{}", cuda_version.runtime.0, cuda_version.runtime.1),
                    required: format!("{}.{}", min_required.0, min_required.1),
                })
            }
        }
        GpuVersion::Rocm(rocm_version) => {
            let min_required = (5, 0, 0);  // Minimum ROCm version
            if rocm_version.runtime >= min_required {
                Ok(GpuCompatibility::Rocm {
                    version: rocm_version,
                    features: query_rocm_capabilities()?,
                })
            } else {
                Err(GpuError::IncompatibleVersion {
                    found: format!("{}.{}.{}", rocm_version.runtime.0, rocm_version.runtime.1, rocm_version.runtime.2),
                    required: format!("{}.{}.{}", min_required.0, min_required.1, min_required.2),
                })
            }
        }
        GpuVersion::None => {
            Err(GpuError::NoGpuAvailable)
        }
    }
}

#[derive(Debug, Clone)]
pub enum GpuCompatibility {
    Cuda {
        version: CudaVersion,
        features: CudaCapabilities,
    },
    Rocm {
        version: RocmVersion,
        features: RocmCapabilities,
    },
}

#[derive(Debug, Clone)]
pub struct CudaCapabilities {
    pub compute_capability: (u32, u32),  // e.g., (8, 6) for RTX 4090
    pub tensor_cores: bool,
    pub mixed_precision: bool,
    pub memory_bandwidth: u64,  // GB/s
}

#[derive(Debug, Clone)]
pub struct RocmCapabilities {
    pub gfx_version: String,  // e.g., "gfx1030"
    pub mixed_precision: bool,
    pub matrix_cores: bool,
    pub memory_bandwidth: u64,  // GB/s
}

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("No GPU runtime available")]
    NoGpuAvailable,

    #[error("Incompatible GPU version: found {found}, required {required}")]
    IncompatibleVersion { found: String, required: String },

    #[error("GPU capability query failed: {0}")]
    CapabilityQueryFailed(String),
}

fn query_cuda_capabilities() -> Result<CudaCapabilities, GpuError> {
    // Query CUDA device capabilities
    // This would use CUDA Device API to get compute capability, memory info, etc.
    Ok(CudaCapabilities {
        compute_capability: (8, 6), // Placeholder - actual implementation would query
        tensor_cores: true,
        mixed_precision: true,
        memory_bandwidth: 1008, // GB/s for RTX 4090
    })
}

fn query_rocm_capabilities() -> Result<RocmCapabilities, GpuError> {
    // Query ROCm device capabilities
    Ok(RocmCapabilities {
        gfx_version: "gfx1030".to_string(), // Placeholder
        mixed_precision: true,
        matrix_cores: true,
        memory_bandwidth: 512, // Placeholder
    })
}
```

### Alternative Approaches

1. **Crate-based Detection**: Use existing crates like `cudarc` or `hip-rs`
2. **Dynamic Library Loading**: Load GPU libraries at runtime without compile-time dependencies
3. **Hybrid Approach**: Runtime APIs with command fallback for edge cases

## Implementation Plan

### Phase 1: Foundation (Week 1)
- [ ] Define GPU version detection API and data structures
- [ ] Implement CUDA runtime API bindings for version detection
- [ ] Implement ROCm/HIP runtime API bindings for version detection
- [ ] Add comprehensive error handling and fallback strategies

### Phase 2: Capability Detection (Week 2)
- [ ] Extend CUDA detection to include compute capabilities
- [ ] Add memory bandwidth and tensor core detection
- [ ] Implement ROCm GFX version and feature detection
- [ ] Create unified capability assessment framework

### Phase 3: Integration & Testing (Week 3)
- [ ] Replace command-based detection in existing code
- [ ] Add comprehensive testing across GPU configurations
- [ ] Validate behavior in various deployment environments
- [ ] Performance benchmarking vs. command-based approach

### Phase 4: Production Hardening (Week 4)
- [ ] Error recovery and graceful degradation
- [ ] Logging and diagnostics for deployment debugging
- [ ] Documentation and integration examples
- [ ] Backwards compatibility testing

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_version_detection() {
        // Test on systems with CUDA available
        if cfg!(feature = "gpu") {
            let version = cuda_version::get_cuda_version();
            if version.is_some() {
                let cuda_ver = version.unwrap();
                assert!(cuda_ver.runtime.0 >= 11); // Minimum major version
                assert!(cuda_ver.driver.0 >= 11);
            }
        }
    }

    #[test]
    fn test_rocm_version_detection() {
        // Test on systems with ROCm available
        if cfg!(feature = "gpu") {
            let version = rocm_version::get_rocm_version();
            if version.is_some() {
                let rocm_ver = version.unwrap();
                assert!(rocm_ver.runtime.0 >= 5); // Minimum major version
            }
        }
    }

    #[test]
    fn test_gpu_compatibility_check() {
        let compatibility = check_gpu_compatibility();
        match compatibility {
            Ok(compat) => {
                match compat {
                    GpuCompatibility::Cuda { version, features } => {
                        assert!(version.runtime >= (11, 8));
                        // Validate capability detection
                    }
                    GpuCompatibility::Rocm { version, features } => {
                        assert!(version.runtime >= (5, 0, 0));
                        // Validate capability detection
                    }
                }
            }
            Err(GpuError::NoGpuAvailable) => {
                // Expected on systems without GPU
            }
            Err(e) => panic!("Unexpected GPU error: {:?}", e),
        }
    }

    #[test]
    fn test_version_parsing() {
        // Test CUDA version parsing
        assert_eq!(cuda_version::parse_cuda_version(11080), (11, 8));
        assert_eq!(cuda_version::parse_cuda_version(12010), (12, 1));

        // Test ROCm version parsing
        assert_eq!(rocm_version::parse_rocm_version(50200100), (5, 2, 1));
    }

    #[test]
    #[ignore] // Integration test
    fn test_no_external_commands() {
        // Ensure no external command execution
        // This could use process monitoring or static analysis
        // to verify no Command::new() calls in GPU detection paths
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_docker_environment() {
        // Test that detection works in Docker without dev tools
        let version = get_gpu_version();
        // Should work even without nvcc/rocm-smi in PATH
    }

    #[test]
    fn test_ci_environment() {
        // Test behavior in CI environment
        // Should gracefully handle lack of GPU hardware
    }
}
```

## Risk Assessment

### Technical Risks
- **ABI Compatibility**: GPU runtime API changes across versions
- **Dynamic Loading**: Runtime libraries may not be available
- **Platform Differences**: Windows vs. Linux API variations
- **Error Handling**: Graceful degradation when APIs fail

### Mitigation Strategies
- Comprehensive testing across GPU runtime versions
- Fallback mechanisms for edge cases
- Clear error messages for deployment debugging
- Feature flags for gradual rollout

## Dependencies

```toml
[dependencies]
# For enhanced GPU runtime bindings
cudarc = { version = "0.11", optional = true, features = ["cuda-12-0"] }
hip-rs = { version = "0.1", optional = true }
thiserror = "1.0"

[features]
gpu = ["cudarc", "hip-rs"]
cuda = ["cudarc"]
rocm = ["hip-rs"]
```

## Success Criteria

- [ ] **Reliability**: 99%+ success rate in detecting available GPU runtimes
- [ ] **Performance**: < 1ms version detection time
- [ ] **Portability**: Works in containers, CI/CD, and edge environments
- [ ] **Accuracy**: Correctly identifies GPU capabilities for optimization
- [ ] **Maintainability**: No external command dependencies
- [ ] **Debugging**: Clear error messages for deployment issues

## Related Issues

- #XXX: GPU memory manager capability requirements
- #XXX: Mixed precision detection and utilization
- #XXX: Dynamic kernel selection based on GPU capabilities
- #XXX: Production deployment environment validation

## Implementation Notes

This native approach eliminates external dependencies while providing more detailed GPU capability information. The implementation focuses on reliability, performance, and deployment simplicity - critical requirements for production inference systems.
