# [STUB] Replace Fallback Logic with Accurate Kernel Availability Detection

## Problem Description

The `QuantizedLinear::can_use_native_quantized_matmul` function in `crates/bitnet-inference/src/layers/quantized_linear.rs` contains a `_ => true` fallback that assumes native quantized matrix multiplication is always available for unspecified device/quantization type combinations. This approach masks cases where native kernels are not actually available, potentially leading to runtime failures or fallback to inefficient implementations.

## Environment

- **File**: `crates/bitnet-inference/src/layers/quantized_linear.rs`
- **Function**: `QuantizedLinear::can_use_native_quantized_matmul`
- **Crate**: `bitnet-inference`
- **Impact**: Kernel selection, performance optimization, error handling

## Current Implementation Issues

```rust
fn can_use_native_quantized_matmul(&self) -> bool {
    match (&self.device, &self.qtype) {
        (Device::Cuda(_), QuantizationType::I2S) => true, // GPU I2S kernel available
        (Device::Cpu, QuantizationType::I2S) => true, // CPU I2S kernel always available via fallback
        (Device::Cpu, QuantizationType::TL1) => true, // CPU TL1 kernel available
        (Device::Cpu, QuantizationType::TL2) => true, // CPU TL2 kernel available
        _ => true, // Default to native quantized operations - PROBLEMATIC!
    }
}
```

## Root Cause Analysis

### Overly Permissive Fallback
1. **Blanket Assumption**: `_ => true` assumes all combinations are supported
2. **No Kernel Validation**: Doesn't verify actual kernel implementation existence
3. **Runtime Failures**: May lead to panics or errors when unsupported kernels are called
4. **Missing Device Types**: Doesn't account for new device types or quantization methods
5. **No Version Checking**: Ignores kernel version compatibility and feature availability

### Missing Integration Points
- **No KernelManager Query**: Should query `KernelManager` for actual kernel availability
- **Static Analysis**: Hard-coded logic instead of dynamic kernel registration
- **Architecture Awareness**: Missing CPU architecture-specific kernel availability
- **Feature Flag Consideration**: Doesn't account for compile-time feature availability

## Impact Assessment

- **Severity**: Medium - Can cause runtime failures and performance degradation
- **Affected Components**: All quantized layer operations, kernel selection, device optimization
- **User Impact**: Unexpected failures, suboptimal performance, confusing error messages
- **Development Impact**: Hidden bugs, difficult debugging, maintenance overhead

## Proposed Solution

Implement comprehensive kernel availability checking with dynamic kernel registry:

### 1. Kernel Registry Infrastructure
```rust
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct KernelKey {
    pub device_type: DeviceType,
    pub operation: KernelOperation,
    pub quantization_type: QuantizationType,
    pub data_type: DataType,
}

#[derive(Debug, Clone)]
pub struct KernelInfo {
    pub key: KernelKey,
    pub implementation: KernelImplementation,
    pub performance_tier: PerformanceTier,
    pub memory_requirements: MemoryRequirements,
    pub supported_shapes: Vec<ShapeConstraint>,
    pub feature_requirements: Vec<FeatureRequirement>,
}

#[derive(Debug)]
pub struct KernelManager {
    registered_kernels: Arc<RwLock<HashMap<KernelKey, KernelInfo>>>,
    device_capabilities: Arc<RwLock<HashMap<DeviceId, DeviceCapabilities>>>,
    feature_flags: FeatureFlags,
}

impl KernelManager {
    pub fn new() -> Self {
        let mut manager = Self {
            registered_kernels: Arc::new(RwLock::new(HashMap::new())),
            device_capabilities: Arc::new(RwLock::new(HashMap::new())),
            feature_flags: FeatureFlags::detect(),
        };

        manager.register_default_kernels();
        manager
    }

    pub fn register_kernel(&self, kernel_info: KernelInfo) -> Result<()> {
        let mut kernels = self.registered_kernels.write().unwrap();
        kernels.insert(kernel_info.key.clone(), kernel_info);
        Ok(())
    }

    pub fn has_kernel_for(
        &self,
        device: &Device,
        qtype: QuantizationType,
        operation: KernelOperation,
    ) -> bool {
        let key = KernelKey {
            device_type: device.device_type(),
            operation,
            quantization_type: qtype,
            data_type: DataType::F32, // Default, could be parameterized
        };

        let kernels = self.registered_kernels.read().unwrap();
        if let Some(kernel_info) = kernels.get(&key) {
            self.validate_kernel_compatibility(device, kernel_info)
        } else {
            false
        }
    }

    fn validate_kernel_compatibility(&self, device: &Device, kernel_info: &KernelInfo) -> bool {
        // Check device capabilities
        if !self.device_supports_kernel(device, kernel_info) {
            return false;
        }

        // Check feature requirements
        if !self.feature_flags.satisfies(&kernel_info.feature_requirements) {
            return false;
        }

        // Check runtime requirements
        self.validate_runtime_requirements(device, kernel_info)
    }

    fn device_supports_kernel(&self, device: &Device, kernel_info: &KernelInfo) -> bool {
        let capabilities = self.device_capabilities.read().unwrap();
        if let Some(device_caps) = capabilities.get(&device.id()) {
            device_caps.supports_kernel(kernel_info)
        } else {
            // If we don't know device capabilities, assume supported
            // This could be made more conservative
            true
        }
    }
}
```

### 2. Enhanced QuantizedLinear with Kernel Manager Integration
```rust
impl QuantizedLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        qtype: QuantizationType,
        device: Device,
        kernel_manager: Arc<KernelManager>,
    ) -> Result<Self> {
        // Validate kernel availability during construction
        if !kernel_manager.has_kernel_for(&device, qtype, KernelOperation::MatMul) {
            return Err(BitNetError::UnsupportedKernelConfiguration {
                device: device.clone(),
                quantization_type: qtype,
                operation: "quantized_matmul".to_string(),
            });
        }

        Ok(Self {
            in_features,
            out_features,
            qtype,
            device,
            kernel_manager,
            // ... other fields
        })
    }

    fn can_use_native_quantized_matmul(&self) -> bool {
        self.kernel_manager.has_kernel_for(
            &self.device,
            self.qtype,
            KernelOperation::MatMul,
        )
    }

    fn get_best_kernel(&self, input_shape: &[usize]) -> Result<KernelHandle> {
        let requirements = KernelRequirements {
            device: self.device.clone(),
            quantization_type: self.qtype,
            operation: KernelOperation::MatMul,
            input_shape: input_shape.to_vec(),
            output_shape: vec![input_shape[0], self.out_features],
            performance_priority: self.performance_config.priority,
        };

        self.kernel_manager.select_best_kernel(&requirements)
    }

    pub fn forward(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        if self.can_use_native_quantized_matmul() {
            let kernel = self.get_best_kernel(input.shape())?;
            kernel.execute_quantized_matmul(input, &self.weight, &self.bias)
        } else {
            // Fallback to dequantized computation
            log::warn!(
                "Native quantized kernel not available for {:?} on {:?}, using fallback",
                self.qtype,
                self.device
            );
            self.forward_dequantized(input)
        }
    }
}
```

### 3. Comprehensive Kernel Registration
```rust
impl KernelManager {
    fn register_default_kernels(&mut self) {
        // CPU kernels
        self.register_cpu_kernels();

        // GPU kernels (feature-gated)
        #[cfg(feature = "cuda")]
        self.register_cuda_kernels();

        #[cfg(feature = "rocm")]
        self.register_rocm_kernels();

        #[cfg(target_os = "macos")]
        self.register_metal_kernels();
    }

    fn register_cpu_kernels(&mut self) {
        // I2S CPU kernels
        self.register_kernel(KernelInfo {
            key: KernelKey {
                device_type: DeviceType::Cpu,
                operation: KernelOperation::MatMul,
                quantization_type: QuantizationType::I2S,
                data_type: DataType::F32,
            },
            implementation: KernelImplementation::CpuI2SMatMul,
            performance_tier: PerformanceTier::Optimized,
            memory_requirements: MemoryRequirements {
                temp_memory: 0,
                alignment: 32,
            },
            supported_shapes: vec![
                ShapeConstraint::MinSize { dimension: 0, min_size: 1 },
                ShapeConstraint::MaxSize { dimension: 1, max_size: 16384 },
            ],
            feature_requirements: vec![],
        }).unwrap();

        // TL1 CPU kernels
        self.register_kernel(KernelInfo {
            key: KernelKey {
                device_type: DeviceType::Cpu,
                operation: KernelOperation::MatMul,
                quantization_type: QuantizationType::TL1,
                data_type: DataType::F32,
            },
            implementation: KernelImplementation::CpuTL1MatMul,
            performance_tier: PerformanceTier::Optimized,
            memory_requirements: MemoryRequirements {
                temp_memory: 1024, // Lookup table storage
                alignment: 32,
            },
            supported_shapes: vec![],
            feature_requirements: vec![],
        }).unwrap();

        // Architecture-specific optimized kernels
        #[cfg(target_arch = "x86_64")]
        self.register_x86_kernels();

        #[cfg(target_arch = "aarch64")]
        self.register_arm_kernels();
    }

    #[cfg(target_arch = "x86_64")]
    fn register_x86_kernels(&mut self) {
        // AVX2 optimized kernels
        if self.feature_flags.has_avx2() {
            self.register_kernel(KernelInfo {
                key: KernelKey {
                    device_type: DeviceType::Cpu,
                    operation: KernelOperation::MatMul,
                    quantization_type: QuantizationType::I2S,
                    data_type: DataType::F32,
                },
                implementation: KernelImplementation::CpuI2SMatMulAVX2,
                performance_tier: PerformanceTier::HighlyOptimized,
                memory_requirements: MemoryRequirements {
                    temp_memory: 256,
                    alignment: 32,
                },
                supported_shapes: vec![],
                feature_requirements: vec![
                    FeatureRequirement::CpuFeature(CpuFeature::AVX2),
                ],
            }).unwrap();
        }

        // AVX-512 optimized kernels
        if self.feature_flags.has_avx512() {
            self.register_kernel(KernelInfo {
                key: KernelKey {
                    device_type: DeviceType::Cpu,
                    operation: KernelOperation::MatMul,
                    quantization_type: QuantizationType::I2S,
                    data_type: DataType::F32,
                },
                implementation: KernelImplementation::CpuI2SMatMulAVX512,
                performance_tier: PerformanceTier::ExtremelyOptimized,
                memory_requirements: MemoryRequirements {
                    temp_memory: 512,
                    alignment: 64,
                },
                supported_shapes: vec![],
                feature_requirements: vec![
                    FeatureRequirement::CpuFeature(CpuFeature::AVX512),
                ],
            }).unwrap();
        }
    }

    #[cfg(feature = "cuda")]
    fn register_cuda_kernels(&mut self) {
        // CUDA I2S kernels
        self.register_kernel(KernelInfo {
            key: KernelKey {
                device_type: DeviceType::Cuda,
                operation: KernelOperation::MatMul,
                quantization_type: QuantizationType::I2S,
                data_type: DataType::F32,
            },
            implementation: KernelImplementation::CudaI2SMatMul,
            performance_tier: PerformanceTier::HighlyOptimized,
            memory_requirements: MemoryRequirements {
                temp_memory: 8192, // GPU workspace
                alignment: 256,
            },
            supported_shapes: vec![
                ShapeConstraint::MinSize { dimension: 0, min_size: 32 },
            ],
            feature_requirements: vec![
                FeatureRequirement::CudaComputeCapability { min_major: 5, min_minor: 0 },
            ],
        }).unwrap();

        // Tensor Core optimized kernels
        self.register_kernel(KernelInfo {
            key: KernelKey {
                device_type: DeviceType::Cuda,
                operation: KernelOperation::MatMul,
                quantization_type: QuantizationType::I2S,
                data_type: DataType::F16,
            },
            implementation: KernelImplementation::CudaI2SMatMulTensorCore,
            performance_tier: PerformanceTier::ExtremelyOptimized,
            memory_requirements: MemoryRequirements {
                temp_memory: 16384,
                alignment: 256,
            },
            supported_shapes: vec![
                ShapeConstraint::MultipleOf { dimension: 0, multiple: 8 },
                ShapeConstraint::MultipleOf { dimension: 1, multiple: 8 },
            ],
            feature_requirements: vec![
                FeatureRequirement::CudaComputeCapability { min_major: 7, min_minor: 0 },
                FeatureRequirement::TensorCores,
            ],
        }).unwrap();
    }
}
```

### 4. Runtime Kernel Selection and Validation
```rust
impl KernelManager {
    pub fn select_best_kernel(&self, requirements: &KernelRequirements) -> Result<KernelHandle> {
        let available_kernels = self.find_compatible_kernels(requirements)?;

        if available_kernels.is_empty() {
            return Err(BitNetError::NoCompatibleKernel {
                requirements: requirements.clone(),
            });
        }

        // Sort by performance tier and select best
        let best_kernel = available_kernels
            .iter()
            .max_by_key(|k| k.performance_tier as u8)
            .unwrap();

        log::debug!(
            "Selected kernel {:?} for operation {:?}",
            best_kernel.implementation,
            requirements.operation
        );

        Ok(KernelHandle::new(best_kernel.clone()))
    }

    fn find_compatible_kernels(&self, requirements: &KernelRequirements) -> Result<Vec<KernelInfo>> {
        let kernels = self.registered_kernels.read().unwrap();
        let mut compatible = Vec::new();

        for kernel_info in kernels.values() {
            if self.is_kernel_compatible(kernel_info, requirements) {
                compatible.push(kernel_info.clone());
            }
        }

        Ok(compatible)
    }

    fn is_kernel_compatible(&self, kernel: &KernelInfo, requirements: &KernelRequirements) -> bool {
        // Check basic compatibility
        if kernel.key.device_type != requirements.device.device_type()
            || kernel.key.quantization_type != requirements.quantization_type
            || kernel.key.operation != requirements.operation {
            return false;
        }

        // Check shape constraints
        if !self.validate_shape_constraints(kernel, requirements) {
            return false;
        }

        // Check feature requirements
        if !self.feature_flags.satisfies(&kernel.feature_requirements) {
            return false;
        }

        // Check device capabilities
        self.device_supports_kernel(&requirements.device, kernel)
    }
}
```

## Implementation Plan

### Phase 1: Kernel Registry Infrastructure
- [ ] Implement `KernelManager` and kernel registration system
- [ ] Add `KernelInfo` and `KernelKey` data structures
- [ ] Create feature detection and capability checking
- [ ] Add basic kernel compatibility validation

### Phase 2: Default Kernel Registration
- [ ] Register all existing CPU kernels with proper metadata
- [ ] Add CUDA kernel registration with compute capability requirements
- [ ] Create architecture-specific kernel registration (AVX2, NEON, etc.)
- [ ] Add feature requirement validation

### Phase 3: QuantizedLinear Integration
- [ ] Replace hardcoded logic with kernel manager queries
- [ ] Add kernel selection based on input shape and performance requirements
- [ ] Implement graceful fallback to dequantized operations
- [ ] Add comprehensive error handling and logging

### Phase 4: Advanced Features
- [ ] Add runtime performance monitoring and kernel benchmarking
- [ ] Implement adaptive kernel selection based on actual performance
- [ ] Add kernel warmup and caching for improved startup performance
- [ ] Create kernel configuration and tuning interfaces

### Phase 5: Testing and Validation
- [ ] Add comprehensive unit tests for kernel registration and selection
- [ ] Create integration tests with real kernel implementations
- [ ] Add performance regression testing
- [ ] Test error handling and edge cases

## Testing Strategy

### Kernel Availability Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_registration() {
        let manager = KernelManager::new();

        // Test basic CPU kernel availability
        assert!(manager.has_kernel_for(
            &Device::Cpu,
            QuantizationType::I2S,
            KernelOperation::MatMul
        ));

        assert!(manager.has_kernel_for(
            &Device::Cpu,
            QuantizationType::TL1,
            KernelOperation::MatMul
        ));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_kernel_availability() {
        let manager = KernelManager::new();

        // Test CUDA kernel availability with capability check
        let cuda_device = Device::Cuda(0);
        let has_cuda_i2s = manager.has_kernel_for(
            &cuda_device,
            QuantizationType::I2S,
            KernelOperation::MatMul
        );

        // Result depends on actual hardware capabilities
        println!("CUDA I2S kernel available: {}", has_cuda_i2s);
    }

    #[test]
    fn test_unsupported_combination() {
        let manager = KernelManager::new();

        // Test with unsupported combination
        let has_unsupported = manager.has_kernel_for(
            &Device::Cpu,
            QuantizationType::Unknown,
            KernelOperation::MatMul
        );

        assert!(!has_unsupported);
    }

    #[test]
    fn test_quantized_linear_construction() {
        let manager = Arc::new(KernelManager::new());

        // Should succeed with supported combination
        let layer = QuantizedLinear::new(
            128, 256,
            QuantizationType::I2S,
            Device::Cpu,
            manager.clone()
        );
        assert!(layer.is_ok());

        // Should fail with unsupported combination
        let layer = QuantizedLinear::new(
            128, 256,
            QuantizationType::Unknown,
            Device::Cpu,
            manager
        );
        assert!(layer.is_err());
    }
}
```

## BitNet-rs Integration Notes

### Feature Flag Integration
- Kernel registration respects compile-time feature flags
- Runtime availability checking for optional features
- Graceful degradation when advanced kernels unavailable

### Performance Considerations
- Kernel manager initialization should be fast and cached
- Kernel selection optimized for hot path performance
- Minimal runtime overhead for availability checking

## Dependencies

No new dependencies required - uses existing kernel and device infrastructure.

## Acceptance Criteria

- [ ] Complete replacement of `_ => true` fallback with accurate kernel checking
- [ ] Comprehensive kernel registry with device and feature capability validation
- [ ] Integration with existing `KernelManager` or creation of new kernel management system
- [ ] Proper error handling when unsupported kernel combinations are requested
- [ ] Architecture-specific kernel availability detection (AVX2, NEON, CUDA compute capability)
- [ ] Performance-optimized kernel selection based on input characteristics
- [ ] Graceful fallback to alternative implementations when preferred kernels unavailable
- [ ] Full test coverage including edge cases and unsupported configurations
- [ ] Clear error messages for unsupported kernel configurations
- [ ] Integration with existing device management and feature detection systems

## Related Issues

- Kernel optimization and performance tuning
- Device capability detection and management
- Feature flag architecture and conditional compilation
- Error handling standardization and user experience

## Priority

**Medium** - Important for correctness and proper error handling. Prevents runtime failures and provides better user experience when unsupported operations are attempted.
