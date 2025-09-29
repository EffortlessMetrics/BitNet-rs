# [GPU] Implement Real Tensor Core Capability Detection in GpuBackend

## Problem Description

The `GpuBackend::supports_tensor_cores` function in `crates/bitnet-inference/src/gpu.rs` currently returns a configuration value instead of performing actual hardware capability detection. This prevents the system from automatically optimizing for tensor core acceleration when available and may lead to performance degradation or runtime errors when tensor core operations are enabled on incompatible hardware.

## Environment

- **Component**: `crates/bitnet-inference/src/gpu.rs`
- **Function**: `GpuBackend::supports_tensor_cores`
- **Feature Context**: `gpu` feature flag with CUDA backend
- **Hardware Requirements**: NVIDIA GPUs with Tensor Core support (Volta+ architecture)
- **CUDA Version**: CUDA 10.0+ for tensor core APIs

## Current Implementation Analysis

```rust
pub fn supports_tensor_cores(&self) -> bool {
    // Would check actual GPU capabilities (Volta+)
    self.performance_config.enable_tensor_cores
}
```

**Issues Identified:**
1. **No hardware detection**: Returns configuration value rather than querying actual GPU capabilities
2. **Missing architecture validation**: Doesn't verify if GPU supports tensor cores (Volta, Turing, Ampere, Ada Lovelace, Hopper)
3. **No CUDA version checking**: Tensor core support requires specific CUDA versions
4. **No compute capability validation**: Tensor cores require compute capability 7.0+
5. **Configuration override ignored**: User configuration should influence but not replace capability detection

## Impact Assessment

**Severity**: Medium-High
**Affected Users**: All users with NVIDIA GPUs, especially those with tensor core capable hardware
**Performance Impact**:
- Missed optimization opportunities on capable hardware
- Potential runtime failures when tensor core operations are enabled on incompatible GPUs
- Suboptimal kernel selection for mixed-precision inference

## Root Cause Analysis

The current implementation is a placeholder that doesn't integrate with CUDA device query APIs. Proper tensor core detection requires:

1. **Hardware capability query**: CUDA device properties and compute capability
2. **Architecture detection**: Identifying tensor core capable architectures
3. **Runtime validation**: Ensuring CUDA runtime supports tensor core operations
4. **Mixed precision compatibility**: Verifying support for FP16/BF16 tensor core operations

## Proposed Solution

### 1. Comprehensive Tensor Core Detection System

Implement multi-layered detection that combines hardware queries, runtime validation, and user configuration:

```rust
impl GpuBackend {
    pub fn supports_tensor_cores(&self) -> bool {
        // User can force disable tensor cores
        if !self.performance_config.enable_tensor_cores {
            return false;
        }

        // Check hardware and runtime capabilities
        self.detect_tensor_core_capability()
            .unwrap_or_else(|err| {
                warn!("Failed to detect tensor core capability: {}", err);
                false
            })
    }

    fn detect_tensor_core_capability(&self) -> Result<bool> {
        #[cfg(feature = "cuda")]
        {
            self.cuda_detect_tensor_cores()
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(false)
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_detect_tensor_cores(&self) -> Result<bool> {
        use crate::cuda::{self, DeviceAttribute};

        // Get device properties
        let device_props = cuda::get_device_properties(self.device_id)?;

        // Check compute capability (tensor cores require 7.0+)
        let major = device_props.major;
        let minor = device_props.minor;
        let compute_capability = major * 10 + minor;

        if compute_capability < 70 {
            debug!("Device {} has compute capability {}.{}, tensor cores require 7.0+",
                   self.device_id, major, minor);
            return Ok(false);
        }

        // Verify tensor core specific features
        let tensor_core_support = self.verify_tensor_core_features(&device_props)?;
        if !tensor_core_support {
            return Ok(false);
        }

        // Check CUDA runtime version
        let runtime_version = cuda::get_runtime_version()?;
        if runtime_version < 10000 { // CUDA 10.0+
            warn!("CUDA runtime version {} does not support tensor cores (requires 10.0+)",
                  runtime_version);
            return Ok(false);
        }

        // Verify mixed precision support
        self.verify_mixed_precision_support()?;

        info!("Tensor core support detected on device {} (compute {}.{})",
              self.device_id, major, minor);
        Ok(true)
    }

    #[cfg(feature = "cuda")]
    fn verify_tensor_core_features(&self, props: &cuda::DeviceProperties) -> Result<bool> {
        // Check architecture-specific tensor core features
        match (props.major, props.minor) {
            // Volta (V100, Titan V)
            (7, 0) => self.verify_volta_tensor_cores(props),

            // Turing (RTX 20 series, T4)
            (7, 5) => self.verify_turing_tensor_cores(props),

            // Ampere (RTX 30 series, A100)
            (8, 0) | (8, 6) => self.verify_ampere_tensor_cores(props),

            // Ada Lovelace (RTX 40 series)
            (8, 9) => self.verify_ada_tensor_cores(props),

            // Hopper (H100)
            (9, 0) => self.verify_hopper_tensor_cores(props),

            _ => {
                debug!("Unknown architecture {}.{}, attempting generic tensor core detection",
                       props.major, props.minor);
                self.verify_generic_tensor_cores(props)
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn verify_volta_tensor_cores(&self, props: &cuda::DeviceProperties) -> Result<bool> {
        // Volta supports FP16 tensor cores
        let has_fp16_support = props.supports_half_precision();

        if !has_fp16_support {
            debug!("Volta device lacks FP16 support required for tensor cores");
            return Ok(false);
        }

        // Check for minimum memory bandwidth (V100 specific)
        if props.memory_bus_width < 4096 {
            warn!("Volta device has insufficient memory bandwidth for optimal tensor core performance");
        }

        Ok(true)
    }

    #[cfg(feature = "cuda")]
    fn verify_turing_tensor_cores(&self, props: &cuda::DeviceProperties) -> Result<bool> {
        // Turing supports FP16, INT8, and INT4 tensor cores
        let has_fp16_support = props.supports_half_precision();
        let has_int8_support = props.supports_int8_tensor_cores();

        if !has_fp16_support || !has_int8_support {
            debug!("Turing device lacks required precision support for tensor cores");
            return Ok(false);
        }

        Ok(true)
    }

    #[cfg(feature = "cuda")]
    fn verify_ampere_tensor_cores(&self, props: &cuda::DeviceProperties) -> Result<bool> {
        // Ampere supports FP16, BF16, INT8, INT4, and sparsity
        let has_bf16_support = props.supports_bfloat16();
        let has_sparsity_support = props.supports_structured_sparsity();

        if !has_bf16_support {
            debug!("Ampere device lacks BF16 support");
            // BF16 is preferred but not required
        }

        // Check for 3rd generation tensor cores
        let tensor_core_version = props.get_tensor_core_version()?;
        if tensor_core_version < 3 {
            warn!("Ampere device has older tensor core generation: {}", tensor_core_version);
        }

        Ok(true)
    }

    #[cfg(feature = "cuda")]
    fn verify_ada_tensor_cores(&self, props: &cuda::DeviceProperties) -> Result<bool> {
        // Ada Lovelace supports enhanced FP16, BF16, and FP8 tensor cores
        let has_fp8_support = props.supports_fp8_tensor_cores();

        if has_fp8_support {
            info!("Ada Lovelace FP8 tensor core support detected");
        }

        Ok(true)
    }

    #[cfg(feature = "cuda")]
    fn verify_hopper_tensor_cores(&self, props: &cuda::DeviceProperties) -> Result<bool> {
        // Hopper supports advanced tensor core features
        let has_transformer_engine = props.supports_transformer_engine();

        if has_transformer_engine {
            info!("Hopper Transformer Engine support detected");
        }

        Ok(true)
    }

    #[cfg(feature = "cuda")]
    fn verify_generic_tensor_cores(&self, props: &cuda::DeviceProperties) -> Result<bool> {
        // Generic detection for unknown architectures
        let has_basic_requirements = props.supports_half_precision()
            && props.multiprocessor_count >= 80; // Minimum SM count

        Ok(has_basic_requirements)
    }

    #[cfg(feature = "cuda")]
    fn verify_mixed_precision_support(&self) -> Result<bool> {
        // Test actual mixed precision operations
        let test_result = cuda::test_mixed_precision_operation(self.device_id)?;

        if !test_result {
            warn!("Mixed precision test failed on device {}", self.device_id);
            return Ok(false);
        }

        Ok(true)
    }

    /// Get detailed tensor core capabilities for optimization
    pub fn get_tensor_core_capabilities(&self) -> Option<TensorCoreCapabilities> {
        if !self.supports_tensor_cores() {
            return None;
        }

        #[cfg(feature = "cuda")]
        {
            self.cuda_get_tensor_core_capabilities().ok()
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_get_tensor_core_capabilities(&self) -> Result<TensorCoreCapabilities> {
        let props = cuda::get_device_properties(self.device_id)?;

        let capabilities = TensorCoreCapabilities {
            architecture: self.detect_architecture(&props)?,
            supported_precisions: self.detect_supported_precisions(&props)?,
            max_tensor_core_frequency: props.tensor_core_frequency_mhz,
            supports_sparsity: props.supports_structured_sparsity(),
            supports_mma: props.supports_matrix_multiply_accumulate(),
            optimal_tile_sizes: self.get_optimal_tile_sizes(&props)?,
        };

        Ok(capabilities)
    }
}

#[derive(Debug, Clone)]
pub struct TensorCoreCapabilities {
    pub architecture: GpuArchitecture,
    pub supported_precisions: Vec<TensorCorePrecision>,
    pub max_tensor_core_frequency: u32,
    pub supports_sparsity: bool,
    pub supports_mma: bool,
    pub optimal_tile_sizes: Vec<(usize, usize, usize)>, // M, N, K dimensions
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuArchitecture {
    Volta,
    Turing,
    Ampere,
    AdaLovelace,
    Hopper,
    Unknown(u32, u32), // major, minor
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorCorePrecision {
    FP16,
    BF16,
    FP8,
    INT8,
    INT4,
    Binary,
}
```

### 2. CUDA Integration Layer

```rust
// crates/bitnet-inference/src/cuda/device_query.rs
pub mod device_query {
    use super::*;

    pub fn get_device_properties(device_id: u32) -> Result<DeviceProperties> {
        unsafe {
            let mut props = std::mem::zeroed::<cuda_sys::cudaDeviceProp>();
            let result = cuda_sys::cudaGetDeviceProperties(&mut props, device_id as i32);

            if result != cuda_sys::cudaError_t::cudaSuccess {
                bail!("Failed to get device properties for device {}: {:?}", device_id, result);
            }

            Ok(DeviceProperties::from_cuda_props(props))
        }
    }

    pub fn test_mixed_precision_operation(device_id: u32) -> Result<bool> {
        // Perform a small tensor core operation to verify functionality
        let test_size = 16; // Minimum tensor core tile size

        // Allocate test matrices
        let mut a_host = vec![1.0f16; test_size * test_size];
        let mut b_host = vec![1.0f16; test_size * test_size];
        let mut c_host = vec![0.0f32; test_size * test_size];

        // Initialize test data
        for i in 0..test_size {
            a_host[i * test_size + i] = 2.0f16;
            b_host[i * test_size + i] = 2.0f16;
        }

        // Perform tensor core GEMM
        let result = unsafe {
            cuda_tensor_core_gemm(
                device_id,
                test_size, test_size, test_size,
                a_host.as_ptr(), b_host.as_ptr(), c_host.as_mut_ptr(),
                1.0f32, 0.0f32
            )
        };

        if result.is_err() {
            return Ok(false);
        }

        // Verify results (should be 4.0 on diagonal)
        for i in 0..test_size {
            let expected = 4.0f32;
            let actual = c_host[i * test_size + i];
            if (actual - expected).abs() > 0.01 {
                warn!("Tensor core test failed: expected {}, got {}", expected, actual);
                return Ok(false);
            }
        }

        Ok(true)
    }
}

#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub major: u32,
    pub minor: u32,
    pub memory_bus_width: u32,
    pub multiprocessor_count: u32,
    pub tensor_core_frequency_mhz: u32,
    // ... other properties
}

impl DeviceProperties {
    fn from_cuda_props(props: cuda_sys::cudaDeviceProp) -> Self {
        Self {
            name: unsafe {
                std::ffi::CStr::from_ptr(props.name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            },
            major: props.major as u32,
            minor: props.minor as u32,
            memory_bus_width: props.memoryBusWidth as u32,
            multiprocessor_count: props.multiProcessorCount as u32,
            tensor_core_frequency_mhz: props.clockRate / 1000, // Convert to MHz
        }
    }

    pub fn supports_half_precision(&self) -> bool {
        // Check device attributes for FP16 support
        unsafe {
            let mut value = 0i32;
            let result = cuda_sys::cudaDeviceGetAttribute(
                &mut value,
                cuda_sys::cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor,
                self.get_device_id() as i32
            );

            result == cuda_sys::cudaError_t::cudaSuccess && value >= 7
        }
    }

    pub fn supports_bfloat16(&self) -> bool {
        // BF16 support requires Ampere+ (8.0+)
        self.major >= 8
    }

    pub fn supports_structured_sparsity(&self) -> bool {
        // Structured sparsity requires Ampere+ (8.0+)
        self.major >= 8
    }

    pub fn get_tensor_core_version(&self) -> Result<u32> {
        match (self.major, self.minor) {
            (7, 0) => Ok(1), // Volta - 1st gen
            (7, 5) => Ok(2), // Turing - 2nd gen
            (8, 0) | (8, 6) => Ok(3), // Ampere - 3rd gen
            (8, 9) => Ok(4), // Ada Lovelace - 4th gen
            (9, 0) => Ok(5), // Hopper - 5th gen
            _ => bail!("Unknown tensor core version for compute {}.{}", self.major, self.minor)
        }
    }
}
```

## Implementation Breakdown

### Phase 1: Core Detection Infrastructure
- [ ] Implement `DeviceProperties` structure and CUDA integration
- [ ] Add basic tensor core capability detection
- [ ] Implement architecture-specific detection logic
- [ ] Add unit tests for detection logic

### Phase 2: Advanced Capability Detection
- [ ] Implement precision support detection (FP16, BF16, INT8, etc.)
- [ ] Add tensor core version identification
- [ ] Implement mixed precision validation testing
- [ ] Add comprehensive error handling

### Phase 3: Performance Optimization
- [ ] Implement optimal tile size detection
- [ ] Add tensor core frequency querying
- [ ] Implement capability caching for repeated queries
- [ ] Add performance benchmarking

### Phase 4: Integration and Testing
- [ ] Update performance configuration system
- [ ] Add integration tests with real GPU hardware
- [ ] Implement capability-based kernel selection
- [ ] Add comprehensive documentation

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_capability_detection() {
        let backend = create_test_gpu_backend();

        // Mock device with Volta capabilities
        mock_device_properties(7, 0);
        assert!(backend.detect_tensor_core_capability().unwrap());

        // Mock device without tensor cores
        mock_device_properties(6, 1);
        assert!(!backend.detect_tensor_core_capability().unwrap());
    }

    #[test]
    fn test_architecture_specific_features() {
        let backend = create_test_gpu_backend();

        // Test Ampere BF16 support
        mock_ampere_device();
        let caps = backend.get_tensor_core_capabilities().unwrap();
        assert!(caps.supported_precisions.contains(&TensorCorePrecision::BF16));

        // Test Volta lacks BF16
        mock_volta_device();
        let caps = backend.get_tensor_core_capabilities().unwrap();
        assert!(!caps.supported_precisions.contains(&TensorCorePrecision::BF16));
    }

    #[test]
    fn test_user_configuration_override() {
        let mut backend = create_test_gpu_backend();

        // Even with capable hardware, user can disable
        mock_ampere_device();
        backend.performance_config.enable_tensor_cores = false;
        assert!(!backend.supports_tensor_cores());
    }
}
```

### Integration Tests
```rust
#[cfg(test)]
mod integration_tests {
    #[test]
    #[ignore] // Requires actual GPU hardware
    fn test_real_hardware_detection() {
        let backend = GpuBackend::new(0).unwrap();

        // Test should pass on any system with or without tensor cores
        let supports_tensor_cores = backend.supports_tensor_cores();

        if supports_tensor_cores {
            let caps = backend.get_tensor_core_capabilities().unwrap();
            assert!(!caps.supported_precisions.is_empty());
            println!("Detected tensor core capabilities: {:?}", caps);
        }
    }

    #[test]
    #[ignore] // Requires CUDA capable GPU
    fn test_mixed_precision_validation() {
        let backend = GpuBackend::new(0).unwrap();

        if backend.supports_tensor_cores() {
            // Should be able to perform actual tensor core operations
            assert!(backend.verify_mixed_precision_support().unwrap());
        }
    }
}
```

## Performance Considerations

1. **Capability Caching**: Cache detection results to avoid repeated GPU queries
2. **Lazy Initialization**: Only perform detailed capability detection when needed
3. **Fallback Gracefully**: Ensure system works even if detection fails
4. **Minimal Overhead**: Keep detection fast for startup performance

## Risk Assessment

**Low Risk Changes:**
- Adding capability detection infrastructure
- Implementing architecture identification

**Medium Risk Changes:**
- Changing tensor core usage based on detection
- Modifying kernel selection logic

**High Risk Changes:**
- Altering core GPU initialization flow

**Mitigation Strategies:**
- Comprehensive testing on multiple GPU architectures
- Fallback mechanisms for detection failures
- Feature flag for gradual rollout
- Detailed logging for debugging

## Acceptance Criteria

- [ ] Accurate tensor core detection on all supported NVIDIA architectures (Volta+)
- [ ] Proper fallback when tensor cores not available
- [ ] User configuration properly overrides hardware capabilities
- [ ] Performance overhead < 10ms for capability detection
- [ ] Comprehensive test coverage across GPU generations
- [ ] Integration with existing performance configuration system
- [ ] Detailed capability reporting for debugging and optimization

## Related Issues/PRs

- **Related to**: GPU memory allocation optimization
- **Depends on**: CUDA integration infrastructure
- **Blocks**: Mixed precision inference optimization
- **References**: Performance configuration system improvements

## Additional Context

This enhancement is crucial for automatically optimizing BitNet.rs performance on tensor core capable hardware while maintaining compatibility with older GPU architectures. The implementation should provide detailed capability information for further optimization decisions throughout the inference pipeline.