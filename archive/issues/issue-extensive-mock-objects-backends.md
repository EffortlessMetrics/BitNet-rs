# [Refactoring] Replace extensive mock objects with real backend implementations

## Problem Statement

The inference backend system in `crates/bitnet-inference/src/backends.rs` currently relies heavily on mock implementations and incomplete backend functionality. While the basic architecture for CPU and GPU backends exists, several critical components are using placeholder implementations or overly extensive mock objects that mask missing real functionality.

## Current State Analysis

### Issues Identified

1. **Incomplete GPU Tensor Operations**: The `GpuBackend::ensure_gpu_tensor()` method simply creates a mock GPU tensor instead of performing actual device transfer
2. **Extensive Test Mock Objects**: The `MockModel` in the test module implements the full `Model` trait but only provides mock responses
3. **Missing Kernel Integration**: The backends don't properly integrate with the existing kernel infrastructure in `bitnet-kernels`
4. **Placeholder Forward Passes**: Both CPU and GPU backends perform simplified forward passes without proper quantization and kernel utilization
5. **Absent Device Memory Management**: No real GPU memory allocation or management
6. **Missing Mixed Precision Support**: GPU backend reports mixed precision capability but doesn't implement it

### Architecture Overview

**Current Infrastructure Available:**
- ✅ Kernel infrastructure in `bitnet-kernels` with CPU (AVX2, NEON) and GPU (CUDA) implementations
- ✅ Quantization algorithms (I2_S, TL1, TL2) in `bitnet-quantization`
- ✅ Device-aware abstractions in `bitnet-common`
- ✅ Model loading and weight management in `bitnet-models`
- ❌ Real backend integration with kernels
- ❌ Proper device memory management
- ❌ Actual quantized inference execution

## Root Cause Analysis

The mock implementations were likely introduced to:
1. Enable rapid prototyping of the inference engine API
2. Provide test scaffolding without complex dependencies
3. Allow development to proceed while kernel implementations were being built

However, the extensive mock objects now mask critical missing functionality:
- **Backend-Kernel Disconnection**: Backends don't use the sophisticated kernel infrastructure
- **Missing Quantization Integration**: Forward passes don't utilize the production quantization algorithms
- **Incomplete Device Abstraction**: GPU backends don't perform real device operations
- **Test Coverage Gaps**: Mock objects hide integration failures that would occur with real implementations

## Impact Assessment

**Severity**: High
**Affected Components**:
- Inference engine reliability and performance
- GPU acceleration capabilities
- Production deployment readiness
- Integration testing validity

**Business Impact**:
- Inference performance significantly below potential
- GPU acceleration completely non-functional
- Production deployments limited to mock-quality responses
- Development velocity reduced due to late-stage integration issues

## Technical Implementation Plan

### Phase 1: Real Kernel Integration (Priority: Critical)

#### 1.1 CPU Backend Real Implementation
```rust
// In CpuBackend::forward()
async fn forward(&self, input: &ConcreteTensor, cache: &mut KVCache) -> Result<ConcreteTensor> {
    debug!("CPU forward pass with input shape: {:?}", input.shape());

    // Get the best available CPU kernel
    let kernel_manager = KernelManager::new();
    let kernel = kernel_manager.select_best()?;

    // Perform model forward pass with real kernels
    let output = tokio::task::spawn_blocking({
        let model = self.model.clone();
        let input_tensor = input.clone();
        let kernel = kernel;

        move || {
            // Convert KVCache to proper format for model
            let mut model_cache = cache.to_model_cache();

            // Perform forward pass using model with kernel integration
            model.forward_with_kernel(&input_tensor, &mut model_cache, kernel)
        }
    }).await.context("CPU forward pass task failed")??;

    debug!("CPU forward pass completed");
    Ok(output)
}
```

#### 1.2 GPU Backend Real Implementation
```rust
// In GpuBackend::forward()
async fn forward(&self, input: &ConcreteTensor, cache: &mut KVCache) -> Result<ConcreteTensor> {
    debug!("GPU forward pass with input shape: {:?}", input.shape());

    // Ensure CUDA is available
    if !Self::is_available() {
        return Err(anyhow::anyhow!("CUDA not available for GPU backend"));
    }

    // Get GPU kernel
    #[cfg(feature = "gpu")]
    let kernel = crate::kernels::select_gpu_kernel(self.device.cuda_device_id()?)?;

    // Transfer input to GPU device
    let gpu_input = self.transfer_to_gpu(input).await?;

    // Perform model forward pass with GPU kernels
    let output = tokio::task::spawn_blocking({
        let model = self.model.clone();
        let input_tensor = gpu_input;
        let mixed_precision = self.mixed_precision;

        move || {
            let mut model_cache = cache.to_gpu_cache()?;

            // Configure precision based on backend settings
            if mixed_precision {
                model.forward_mixed_precision(&input_tensor, &mut model_cache, kernel)
            } else {
                model.forward_with_kernel(&input_tensor, &mut model_cache, kernel)
            }
        }
    }).await.context("GPU forward pass task failed")??;

    debug!("GPU forward pass completed");
    Ok(output)
}
```

### Phase 2: Device Memory Management (Priority: High)

#### 2.1 GPU Memory Allocation
```rust
impl GpuBackend {
    /// Transfer tensor to GPU device with proper memory management
    async fn transfer_to_gpu(&self, input: &ConcreteTensor) -> Result<ConcreteTensor> {
        match &self.device {
            Device::Cuda(device_id) => {
                // Set CUDA device context
                cuda_runtime::set_device(*device_id)?;

                // Allocate GPU memory
                let gpu_tensor = input.to_device(&candle_core::Device::new_cuda(*device_id)?)?;

                // Wrap in our tensor abstraction
                Ok(ConcreteTensor::from_candle(gpu_tensor))
            }
            _ => Err(anyhow::anyhow!("Invalid device for GPU backend: {:?}", self.device))
        }
    }

    /// Check actual CUDA availability
    fn is_available() -> bool {
        #[cfg(feature = "gpu")]
        {
            use candle_core::Device;
            match Device::new_cuda(0) {
                Ok(_) => true,
                Err(_) => false,
            }
        }
        #[cfg(not(feature = "gpu"))]
        false
    }
}
```

#### 2.2 Memory Pool Integration
```rust
// Integrate with existing memory pool from inference engine
pub struct GpuMemoryManager {
    device_id: usize,
    allocated_tensors: Vec<CudaPtr>,
    memory_pool_size: usize,
}

impl GpuMemoryManager {
    pub fn new(device_id: usize, pool_size: usize) -> Result<Self> {
        cuda_runtime::set_device(device_id)?;
        Ok(Self {
            device_id,
            allocated_tensors: Vec::new(),
            memory_pool_size: pool_size,
        })
    }

    pub fn allocate_tensor(&mut self, shape: &[usize], dtype: DType) -> Result<CudaPtr> {
        let size = shape.iter().product::<usize>() * dtype.size_in_bytes();
        let ptr = cuda_runtime::malloc(size)?;
        self.allocated_tensors.push(ptr);
        Ok(ptr)
    }
}
```

### Phase 3: Model Integration (Priority: High)

#### 3.1 Extend Model Trait for Kernel Integration
```rust
// In bitnet-models/src/bitnet.rs
pub trait Model: Send + Sync {
    fn config(&self) -> &BitNetConfig;

    // Existing methods
    fn forward(&self, input: &ConcreteTensor, cache: &mut dyn std::any::Any) -> Result<ConcreteTensor>;
    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor>;
    fn logits(&self, hidden: &ConcreteTensor) -> Result<ConcreteTensor>;

    // New kernel-aware methods
    fn forward_with_kernel(
        &self,
        input: &ConcreteTensor,
        cache: &mut dyn std::any::Any,
        kernel: &dyn KernelProvider,
    ) -> Result<ConcreteTensor>;

    #[cfg(feature = "gpu")]
    fn forward_mixed_precision(
        &self,
        input: &ConcreteTensor,
        cache: &mut dyn std::any::Any,
        kernel: &dyn KernelProvider,
    ) -> Result<ConcreteTensor>;
}
```

#### 3.2 BitNetModel Kernel Integration
```rust
impl Model for BitNetModel {
    fn forward_with_kernel(
        &self,
        input: &ConcreteTensor,
        cache: &mut dyn std::any::Any,
        kernel: &dyn KernelProvider,
    ) -> Result<ConcreteTensor> {
        // Use transformer with kernel-aware operations
        if let Some(transformer) = &self.transformer {
            transformer.forward_with_kernel(input, cache, kernel)
        } else {
            // Fallback to direct computation with kernels
            self.compute_forward_with_kernel(input, cache, kernel)
        }
    }

    #[cfg(feature = "gpu")]
    fn forward_mixed_precision(
        &self,
        input: &ConcreteTensor,
        cache: &mut dyn std::any::Any,
        kernel: &dyn KernelProvider,
    ) -> Result<ConcreteTensor> {
        // Convert to mixed precision tensors
        let fp16_input = input.to_dtype(DType::F16)?;

        // Perform computation with mixed precision
        let output = self.forward_with_kernel(&fp16_input, cache, kernel)?;

        // Convert back to F32 for consistency
        output.to_dtype(DType::F32)
    }
}
```

### Phase 4: Test Infrastructure Refactoring (Priority: Medium)

#### 4.1 Move Mock Objects to Test Utilities
```rust
// Create: crates/bitnet-inference/tests/utils/mod.rs
pub mod mock_model;
pub mod mock_tokenizer;
pub mod test_fixtures;

// Re-export commonly used mocks
pub use mock_model::MockModel;
pub use mock_tokenizer::MockTokenizer;
```

#### 4.2 Simplified Mock Model
```rust
// In crates/bitnet-inference/tests/utils/mock_model.rs
pub struct MockModel {
    config: BitNetConfig,
    behavior: MockBehavior,
}

pub enum MockBehavior {
    SuccessWithShape(Vec<usize>),
    DelayedSuccess(Duration, Vec<usize>),
    Failure(BitNetError),
}

impl MockModel {
    pub fn new() -> Self {
        Self {
            config: BitNetConfig::default(),
            behavior: MockBehavior::SuccessWithShape(vec![1, 50257]),
        }
    }

    pub fn with_behavior(mut self, behavior: MockBehavior) -> Self {
        self.behavior = behavior;
        self
    }
}

impl Model for MockModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(&self, _input: &ConcreteTensor, _cache: &mut dyn std::any::Any) -> Result<ConcreteTensor> {
        match &self.behavior {
            MockBehavior::SuccessWithShape(shape) => Ok(ConcreteTensor::mock(shape.clone())),
            MockBehavior::DelayedSuccess(delay, shape) => {
                std::thread::sleep(*delay);
                Ok(ConcreteTensor::mock(shape.clone()))
            }
            MockBehavior::Failure(error) => Err(error.clone()),
        }
    }

    // Minimal implementations for other required methods
    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![tokens.len(), self.config.model.hidden_size]))
    }

    fn logits(&self, _hidden: &ConcreteTensor) -> Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, self.config.model.vocab_size]))
    }
}
```

#### 4.3 Integration Tests with Real Implementations
```rust
// Create: crates/bitnet-inference/tests/integration/backend_kernel_integration.rs
#[cfg(feature = "integration-tests")]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_cpu_backend_with_real_kernels() {
        // Load a minimal real model
        let model = load_test_model().await.expect("Failed to load test model");
        let backend = CpuBackend::new(model).expect("Failed to create CPU backend");

        // Test with real tensor data
        let input = create_test_tensor();
        let mut cache = KVCache::new(Default::default()).unwrap();

        let output = backend.forward(&input, &mut cache).await
            .expect("Real CPU backend forward should succeed");

        // Validate output properties
        assert!(output.shape().len() > 0, "Output should have valid shape");
        assert!(output.dtype() == DType::F32, "Output should be F32");
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_gpu_backend_with_cuda_kernels() {
        if !GpuBackend::is_available() {
            return; // Skip if CUDA not available
        }

        let model = load_test_model().await.expect("Failed to load test model");
        let backend = GpuBackend::new(model, Device::Cuda(0))
            .expect("Failed to create GPU backend");

        let input = create_test_tensor();
        let mut cache = KVCache::new(Default::default()).unwrap();

        let output = backend.forward(&input, &mut cache).await
            .expect("Real GPU backend forward should succeed");

        // Validate GPU-specific properties
        assert_eq!(output.device(), &Device::Cuda(0));
        assert!(output.shape().len() > 0);
    }
}
```

## Implementation Tasks

### Phase 1: Core Backend Integration (Week 1-2)
- [ ] **Task 1.1**: Integrate `KernelManager` into `CpuBackend::forward()` method
- [ ] **Task 1.2**: Implement real GPU tensor transfer in `GpuBackend::ensure_gpu_tensor()`
- [ ] **Task 1.3**: Add kernel-aware methods to `Model` trait
- [ ] **Task 1.4**: Update `BitNetModel` to use kernels in forward passes
- [ ] **Task 1.5**: Implement proper `GpuBackend::is_available()` with CUDA detection

### Phase 2: Memory Management (Week 2-3)
- [ ] **Task 2.1**: Create `GpuMemoryManager` for device memory allocation
- [ ] **Task 2.2**: Implement tensor device transfer methods
- [ ] **Task 2.3**: Add memory pool integration for GPU operations
- [ ] **Task 2.4**: Implement proper cleanup and memory deallocation

### Phase 3: Advanced Features (Week 3-4)
- [ ] **Task 3.1**: Implement mixed precision support in GPU backend
- [ ] **Task 3.2**: Add quantization integration to forward passes
- [ ] **Task 3.3**: Optimize memory usage patterns for large models
- [ ] **Task 3.4**: Add performance monitoring and profiling hooks

### Phase 4: Testing and Validation (Week 4-5)
- [ ] **Task 4.1**: Move `MockModel` to `tests/utils/mock_model.rs`
- [ ] **Task 4.2**: Create simplified mock implementations
- [ ] **Task 4.3**: Add integration tests with real kernels
- [ ] **Task 4.4**: Validate performance improvements with benchmarks
- [ ] **Task 4.5**: Cross-validate against C++ reference implementation

## Acceptance Criteria

### AC1: Real Backend Functionality
- [ ] CPU backend uses actual SIMD kernels (AVX2, NEON) for computations
- [ ] GPU backend performs actual CUDA operations with device memory management
- [ ] Forward passes integrate with quantization algorithms (I2_S, TL1, TL2)
- [ ] Backend selection correctly identifies and utilizes available hardware

### AC2: Performance Requirements
- [ ] CPU backend achieves >80% of theoretical SIMD performance
- [ ] GPU backend provides >5x speedup over CPU for large models (>1B parameters)
- [ ] Memory usage remains within configured limits
- [ ] Inference latency meets production requirements (<100ms for 2K context)

### AC3: Test Coverage and Quality
- [ ] Unit tests cover real backend implementations (not just mocks)
- [ ] Integration tests validate kernel-backend interaction
- [ ] Mock objects are simplified and moved to test utilities
- [ ] Cross-validation tests pass with C++ reference implementation

### AC4: Production Readiness
- [ ] Error handling for CUDA initialization failures and memory allocation
- [ ] Graceful fallback from GPU to CPU when hardware unavailable
- [ ] Memory leak prevention in long-running inference sessions
- [ ] Proper resource cleanup on backend destruction

## Testing Strategy

### Unit Testing
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_integration() {
        let kernel_manager = KernelManager::new();
        let kernel = kernel_manager.select_best().unwrap();
        assert!(kernel.is_available());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_memory_management() {
        if !GpuBackend::is_available() { return; }

        let mut memory_manager = GpuMemoryManager::new(0, 1024 * 1024).unwrap();
        let tensor_ptr = memory_manager.allocate_tensor(&[128, 128], DType::F32).unwrap();
        assert!(!tensor_ptr.is_null());
    }
}
```

### Integration Testing
```rust
#[tokio::test]
async fn test_real_model_inference() {
    let model_path = std::env::var("BITNET_TEST_MODEL")
        .unwrap_or_else(|_| "tests/fixtures/small_model.gguf".to_string());

    let model = load_gguf_model(&model_path).await.unwrap();
    let backend = CpuBackend::new(model).unwrap();

    let tokens = vec![1, 2, 3, 4]; // Actual token IDs
    let result = backend.process_tokens(&tokens).await.unwrap();

    assert!(result.len() > tokens.len()); // Should generate additional tokens
}
```

### Performance Benchmarking
```rust
#[bench]
fn bench_cpu_backend_forward(b: &mut Bencher) {
    let model = create_benchmark_model();
    let backend = CpuBackend::new(model).unwrap();
    let input = create_benchmark_input();

    b.iter(|| {
        let mut cache = KVCache::new(Default::default()).unwrap();
        black_box(backend.forward(&input, &mut cache))
    });
}
```

## Risk Assessment and Mitigation

### High Risk: CUDA Integration Complexity
**Risk**: GPU backend implementation may face complex CUDA integration issues
**Mitigation**:
- Start with basic CUDA operations and gradually add complexity
- Maintain fallback to CPU backend for all GPU operations
- Use existing `bitnet-kernels` GPU implementations as reference

### Medium Risk: Performance Regression
**Risk**: Real implementations may initially perform worse than mocks
**Mitigation**:
- Establish performance baselines before implementation
- Use feature flags to enable/disable real implementations during development
- Benchmark each component individually to identify bottlenecks

### Medium Risk: Test Suite Disruption
**Risk**: Replacing mocks may break existing tests
**Mitigation**:
- Move mocks to dedicated test utilities before replacement
- Maintain backward compatibility during transition
- Add feature flags for test modes (mock vs. real)

## Dependencies and Prerequisites

### Required Features
- `cpu` feature for SIMD kernel access
- `gpu` feature for CUDA kernel access
- `full-engine` feature for integration testing

### External Dependencies
- CUDA Toolkit (11.0+) for GPU backend functionality
- AVX2/NEON support for optimal CPU performance

### Related Issues and PRs
- Issue #251: Production-Ready Inference Server
- Issue [AC1 GPU Acceleration]: Missing GPU acceleration cross-validation
- PR [Kernel Infrastructure]: SIMD and CUDA kernel implementations

## Labels
- `priority:high`
- `type:refactoring`
- `component:inference`
- `component:backends`
- `component:kernels`
- `effort:large`
- `gpu`
- `performance`

## Estimated Effort
**Story Points**: 13 (Large)
**Duration**: 4-5 weeks
**Team Size**: 2-3 developers (backend + GPU specialist)

This comprehensive refactoring will transform the BitNet.rs inference system from a prototype using extensive mocks to a production-ready implementation with real backend functionality, setting the foundation for high-performance inference deployments.
