# [IMPLEMENTATION] Replace configuration-based tensor core support check with actual GPU capability detection

## Problem Description
The `GpuBackend::supports_tensor_cores` function in `crates/bitnet-inference/src/gpu.rs` returns configuration values instead of querying actual GPU hardware capabilities for tensor core support.

## Environment
- **File**: `crates/bitnet-inference/src/gpu.rs`
- **Function**: `GpuBackend::supports_tensor_cores`
- **Current State**: Configuration-based placeholder

## Root Cause Analysis
```rust
pub fn supports_tensor_cores(&self) -> bool {
    // Would check actual GPU capabilities (Volta+)
    self.performance_config.enable_tensor_cores
}
```

**Issues:**
1. Returns configuration setting, not hardware capability
2. No actual GPU capability detection
3. Could enable tensor cores on unsupported hardware
4. Prevents automatic optimization based on hardware

## Proposed Solution
```rust
impl GpuBackend {
    pub fn supports_tensor_cores(&self) -> bool {
        // First check if enabled in config
        if !self.performance_config.enable_tensor_cores {
            return false;
        }

        // Then check actual hardware capability
        self.query_tensor_core_support()
    }

    fn query_tensor_core_support(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            match self.get_compute_capability() {
                Ok((major, minor)) => {
                    // Tensor cores available on Volta+ (7.0+), Turing (7.5+), Ampere (8.0+)
                    major > 7 || (major == 7 && minor >= 0)
                }
                Err(_) => false,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    fn get_compute_capability(&self) -> Result<(i32, i32)> {
        use cudarc::driver::CudaDevice;

        let device = CudaDevice::new(self.device_id)?;
        let major = device.get_attribute(cudarc::driver::DeviceAttribute::ComputeCapabilityMajor)?;
        let minor = device.get_attribute(cudarc::driver::DeviceAttribute::ComputeCapabilityMinor)?;

        Ok((major, minor))
    }
}
```

## Implementation Plan
### Phase 1: Hardware Detection (1 day)
- [ ] Implement CUDA compute capability querying
- [ ] Add specific tensor core generation detection
- [ ] Create fallback for non-CUDA builds

### Phase 2: Validation & Testing (1 day)
- [ ] Test on various GPU generations
- [ ] Add comprehensive error handling
- [ ] Create mock testing infrastructure

## Acceptance Criteria
- [ ] Accurate hardware capability detection
- [ ] Configuration override still respected
- [ ] Proper fallback for unsupported hardware
- [ ] Comprehensive test coverage

**Labels**: `implementation`, `gpu`, `hardware-detection`, `P2-medium`
**Effort**: 2 days
