# [CPU] Implement Parallel Forward Pass in CpuInferenceEngine

## Problem Description

The `CpuInferenceEngine::forward_parallel` function in `crates/bitnet-inference/src/cpu.rs` currently returns a placeholder result instead of performing actual parallel inference. This prevents the system from utilizing multi-core CPU architectures effectively and results in significantly degraded performance compared to what could be achieved with proper parallelization.

## Environment

- **Component**: `crates/bitnet-inference/src/cpu.rs`
- **Function**: `CpuInferenceEngine::forward_parallel`
- **Feature Context**: `cpu` feature flag with SIMD optimization
- **Dependencies**: Rayon for parallel processing, Candle for tensor operations
- **Target Architectures**: x86_64 (AVX2/AVX-512), ARM64 (NEON)

## Current Implementation Analysis

```rust
fn forward_parallel(&self, input: &BitNetTensor, _step: usize) -> Result<BitNetTensor> {
    // This is a simplified synchronous version
    // In a full async implementation, we would use model.read().await

    // For now, create a placeholder result
    // In practice, this would require async model access
    let result = BitNetTensor::zeros(&[1, 32000], candle_core::DType::F32, &candle_core::Device::Cpu)?;

    Ok(result)
}
```

**Issues Identified:**
1. **No actual computation**: Returns placeholder tensor instead of performing inference
2. **Missing parallelization**: Doesn't utilize multi-core CPU capabilities
3. **Hardcoded output dimensions**: Uses fixed shape `[1, 32000]` regardless of model or input
4. **Ignores input tensor**: Input parameter is unused
5. **No layer processing**: Doesn't iterate through model layers
6. **Missing optimization opportunities**: No SIMD utilization or cache-friendly memory access

## Impact Assessment

**Severity**: High
**Affected Users**: All users running CPU inference, especially on multi-core systems
**Performance Impact**:
- Orders of magnitude slower than possible with proper parallelization
- Poor CPU utilization on multi-core systems
- Incorrect inference results due to placeholder implementation

## Root Cause Analysis

The current implementation is a placeholder that doesn't integrate with the actual model architecture or leverage CPU parallelization capabilities. A proper implementation requires:

1. **Layer-level parallelization**: Parallel execution of independent operations within layers
2. **Batch processing**: Efficient processing of multiple tokens/sequences
3. **SIMD optimization**: Vectorized operations for quantized computations
4. **Memory locality**: Cache-friendly memory access patterns
5. **Work distribution**: Optimal thread pool utilization

## Proposed Solution

### 1. Comprehensive Parallel Inference Architecture

Implement multi-level parallelization strategy that maximizes CPU utilization:

```rust
use rayon::prelude::*;
use std::sync::Arc;

impl CpuInferenceEngine {
    fn forward_parallel(&self, input: &BitNetTensor, step: usize) -> Result<BitNetTensor> {
        // Get model with appropriate locking strategy
        let model = self.get_model_for_inference()?;

        // Configure parallel execution context
        let parallel_config = self.get_parallel_config(input)?;

        // Execute parallel forward pass
        self.execute_parallel_forward(&model, input, step, &parallel_config)
    }

    fn execute_parallel_forward(
        &self,
        model: &BitNetModel,
        input: &BitNetTensor,
        step: usize,
        config: &ParallelConfig,
    ) -> Result<BitNetTensor> {
        let mut current_tensor = input.clone();

        // Process through model layers with parallelization
        for (layer_idx, layer) in model.layers().iter().enumerate() {
            current_tensor = self.process_layer_parallel(
                layer,
                &current_tensor,
                layer_idx,
                step,
                config,
            )?;
        }

        // Apply final transformations
        self.apply_output_transformations(current_tensor, model)
    }

    fn process_layer_parallel(
        &self,
        layer: &BitNetLayer,
        input: &BitNetTensor,
        layer_idx: usize,
        step: usize,
        config: &ParallelConfig,
    ) -> Result<BitNetTensor> {
        match layer.layer_type() {
            LayerType::Attention => {
                self.process_attention_parallel(layer, input, step, config)
            }
            LayerType::FeedForward => {
                self.process_feedforward_parallel(layer, input, config)
            }
            LayerType::Embedding => {
                self.process_embedding_parallel(layer, input, config)
            }
            LayerType::LayerNorm => {
                self.process_layernorm_parallel(layer, input, config)
            }
        }
    }

    fn process_attention_parallel(
        &self,
        layer: &BitNetLayer,
        input: &BitNetTensor,
        step: usize,
        config: &ParallelConfig,
    ) -> Result<BitNetTensor> {
        let attention = layer.as_attention()?;
        let batch_size = input.dims()[0];
        let seq_len = input.dims()[1];
        let head_dim = attention.head_dim();
        let num_heads = attention.num_heads();

        // Parallel QKV computation
        let (queries, keys, values) = rayon::join3(
            || self.compute_queries_parallel(attention, input, config),
            || self.compute_keys_parallel(attention, input, config),
            || self.compute_values_parallel(attention, input, config),
        );

        let (queries, keys, values) = (queries?, keys?, values?);

        // Parallel attention computation across heads
        let attention_outputs: Result<Vec<_>> = (0..num_heads)
            .into_par_iter()
            .map(|head_idx| {
                self.compute_single_head_attention(
                    &queries, &keys, &values,
                    head_idx, step, config
                )
            })
            .collect();

        let attention_outputs = attention_outputs?;

        // Concatenate and project outputs
        self.concat_and_project_attention_outputs(attention_outputs, attention)
    }

    fn compute_queries_parallel(
        &self,
        attention: &BitNetAttention,
        input: &BitNetTensor,
        config: &ParallelConfig,
    ) -> Result<BitNetTensor> {
        let weight = attention.query_weight();
        self.parallel_quantized_matmul(input, weight, config)
    }

    fn parallel_quantized_matmul(
        &self,
        input: &BitNetTensor,
        weight: &QuantizedWeight,
        config: &ParallelConfig,
    ) -> Result<BitNetTensor> {
        let input_dims = input.dims();
        let weight_dims = weight.dims();

        let batch_size = input_dims[0];
        let seq_len = input_dims[1];
        let input_features = input_dims[2];
        let output_features = weight_dims[0];

        // Determine optimal parallelization strategy
        let strategy = self.determine_parallel_strategy(
            batch_size, seq_len, input_features, output_features, config
        );

        match strategy {
            ParallelStrategy::BatchParallel => {
                self.batch_parallel_matmul(input, weight, config)
            }
            ParallelStrategy::FeatureParallel => {
                self.feature_parallel_matmul(input, weight, config)
            }
            ParallelStrategy::HybridParallel => {
                self.hybrid_parallel_matmul(input, weight, config)
            }
        }
    }

    fn batch_parallel_matmul(
        &self,
        input: &BitNetTensor,
        weight: &QuantizedWeight,
        config: &ParallelConfig,
    ) -> Result<BitNetTensor> {
        let batch_size = input.dims()[0];
        let chunk_size = (batch_size + config.num_threads - 1) / config.num_threads;

        let results: Result<Vec<_>> = (0..batch_size)
            .into_par_iter()
            .chunks(chunk_size)
            .map(|batch_chunk| {
                let batch_indices: Vec<usize> = batch_chunk.collect();
                self.process_batch_chunk(input, weight, &batch_indices, config)
            })
            .collect();

        let results = results?;
        self.concatenate_batch_results(results)
    }

    fn feature_parallel_matmul(
        &self,
        input: &BitNetTensor,
        weight: &QuantizedWeight,
        config: &ParallelConfig,
    ) -> Result<BitNetTensor> {
        let output_features = weight.dims()[0];
        let chunk_size = (output_features + config.num_threads - 1) / config.num_threads;

        let results: Result<Vec<_>> = (0..output_features)
            .into_par_iter()
            .chunks(chunk_size)
            .map(|feature_chunk| {
                let feature_indices: Vec<usize> = feature_chunk.collect();
                self.process_feature_chunk(input, weight, &feature_indices, config)
            })
            .collect();

        let results = results?;
        self.concatenate_feature_results(results)
    }

    fn process_batch_chunk(
        &self,
        input: &BitNetTensor,
        weight: &QuantizedWeight,
        batch_indices: &[usize],
        config: &ParallelConfig,
    ) -> Result<BitNetTensor> {
        // Extract batch slice
        let batch_input = input.slice_batch(batch_indices)?;

        // Perform quantized matrix multiplication with SIMD optimization
        let result = match weight.quantization_type() {
            QuantizationType::I2S => {
                self.i2s_matmul_simd(&batch_input, weight, config)
            }
            QuantizationType::TL1 => {
                self.tl1_matmul_simd(&batch_input, weight, config)
            }
            QuantizationType::TL2 => {
                self.tl2_matmul_simd(&batch_input, weight, config)
            }
        }?;

        Ok(result)
    }

    fn i2s_matmul_simd(
        &self,
        input: &BitNetTensor,
        weight: &QuantizedWeight,
        config: &ParallelConfig,
    ) -> Result<BitNetTensor> {
        // Get optimized kernel based on CPU capabilities
        let kernel = self.get_optimized_i2s_kernel(config.cpu_features)?;

        // Execute SIMD-optimized I2S matrix multiplication
        kernel.execute(input, weight)
    }

    fn get_optimized_i2s_kernel(&self, cpu_features: &CpuFeatures) -> Result<Box<dyn I2SKernel>> {
        #[cfg(target_arch = "x86_64")]
        {
            if cpu_features.has_avx512 {
                Ok(Box::new(I2SAvx512Kernel::new()))
            } else if cpu_features.has_avx2 {
                Ok(Box::new(I2SAvx2Kernel::new()))
            } else {
                Ok(Box::new(I2SScalarKernel::new()))
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if cpu_features.has_neon {
                Ok(Box::new(I2SNeonKernel::new()))
            } else {
                Ok(Box::new(I2SScalarKernel::new()))
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Ok(Box::new(I2SScalarKernel::new()))
        }
    }

    fn determine_parallel_strategy(
        &self,
        batch_size: usize,
        seq_len: usize,
        input_features: usize,
        output_features: usize,
        config: &ParallelConfig,
    ) -> ParallelStrategy {
        let total_work = batch_size * seq_len * input_features * output_features;
        let work_per_thread = total_work / config.num_threads;

        // Heuristics for parallel strategy selection
        if batch_size >= config.num_threads && work_per_thread > config.min_work_per_thread {
            ParallelStrategy::BatchParallel
        } else if output_features >= config.num_threads * 8 {
            ParallelStrategy::FeatureParallel
        } else if batch_size >= 2 && output_features >= config.num_threads * 2 {
            ParallelStrategy::HybridParallel
        } else {
            ParallelStrategy::BatchParallel
        }
    }

    fn get_model_for_inference(&self) -> Result<Arc<BitNetModel>> {
        // Use read-write lock or atomic reference counting for safe concurrent access
        match &self.model_storage {
            ModelStorage::SharedRead(model) => {
                Ok(model.clone())
            }
            ModelStorage::Locked(model_lock) => {
                let model = model_lock.read()
                    .map_err(|_| anyhow!("Failed to acquire model read lock"))?;
                Ok(model.clone())
            }
        }
    }

    fn get_parallel_config(&self, input: &BitNetTensor) -> Result<ParallelConfig> {
        let cpu_info = self.cpu_info.as_ref()
            .ok_or_else(|| anyhow!("CPU information not available"))?;

        let config = ParallelConfig {
            num_threads: self.calculate_optimal_thread_count(input, cpu_info)?,
            cpu_features: cpu_info.features.clone(),
            cache_size: cpu_info.l3_cache_size,
            min_work_per_thread: self.performance_config.min_work_per_thread,
            prefer_simd: cpu_info.features.has_simd(),
        };

        Ok(config)
    }

    fn calculate_optimal_thread_count(&self, input: &BitNetTensor, cpu_info: &CpuInfo) -> Result<usize> {
        let available_threads = rayon::current_num_threads();
        let logical_cores = cpu_info.logical_cores;
        let physical_cores = cpu_info.physical_cores;

        // Calculate work size
        let work_size = input.numel() * std::mem::size_of::<f32>();

        // Heuristics for optimal thread count
        let optimal_threads = if work_size < 1024 * 1024 { // < 1MB
            std::cmp::min(2, physical_cores)
        } else if work_size < 16 * 1024 * 1024 { // < 16MB
            std::cmp::min(physical_cores, available_threads)
        } else {
            std::cmp::min(logical_cores, available_threads)
        };

        Ok(optimal_threads.max(1))
    }
}

#[derive(Debug, Clone)]
struct ParallelConfig {
    num_threads: usize,
    cpu_features: CpuFeatures,
    cache_size: usize,
    min_work_per_thread: usize,
    prefer_simd: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ParallelStrategy {
    BatchParallel,   // Parallelize across batch dimension
    FeatureParallel, // Parallelize across feature dimension
    HybridParallel,  // Mix of batch and feature parallelization
}

enum ModelStorage {
    SharedRead(Arc<BitNetModel>),
    Locked(Arc<RwLock<BitNetModel>>),
}
```

### 2. SIMD Optimized Kernels

```rust
// crates/bitnet-kernels/src/cpu/i2s_avx2.rs
#[cfg(target_arch = "x86_64")]
pub struct I2SAvx2Kernel {
    // Kernel state and configuration
}

#[cfg(target_arch = "x86_64")]
impl I2SKernel for I2SAvx2Kernel {
    fn execute(&self, input: &BitNetTensor, weight: &QuantizedWeight) -> Result<BitNetTensor> {
        unsafe {
            self.execute_avx2_optimized(input, weight)
        }
    }

    unsafe fn execute_avx2_optimized(
        &self,
        input: &BitNetTensor,
        weight: &QuantizedWeight,
    ) -> Result<BitNetTensor> {
        use std::arch::x86_64::*;

        let input_data = input.as_slice::<f32>()?;
        let weight_data = weight.as_i2_slice()?;
        let scale = weight.scale();
        let zero_point = weight.zero_point();

        // Implementation using AVX2 intrinsics for I2S quantization
        // Process 8 elements at a time with AVX2

        // ... detailed SIMD implementation

        todo!("Implement AVX2 optimized I2S kernel")
    }
}
```

## Implementation Breakdown

### Phase 1: Core Parallel Infrastructure
- [ ] Implement `ParallelConfig` and strategy selection
- [ ] Add basic batch-parallel matrix multiplication
- [ ] Implement thread pool management
- [ ] Add CPU feature detection

### Phase 2: Layer-Specific Parallelization
- [ ] Implement parallel attention computation
- [ ] Add parallel feed-forward processing
- [ ] Implement parallel embedding lookups
- [ ] Add parallel layer normalization

### Phase 3: SIMD Optimization
- [ ] Implement AVX2 optimized kernels for x86_64
- [ ] Add NEON optimized kernels for ARM64
- [ ] Implement scalar fallback kernels
- [ ] Add runtime kernel selection

### Phase 4: Advanced Optimizations
- [ ] Implement cache-aware memory access patterns
- [ ] Add work stealing for dynamic load balancing
- [ ] Implement prefetching for memory-bound operations
- [ ] Add performance profiling and tuning

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_forward_correctness() {
        let engine = create_test_cpu_engine();
        let input = create_test_input();

        let sequential_result = engine.forward_sequential(&input, 0).unwrap();
        let parallel_result = engine.forward_parallel(&input, 0).unwrap();

        assert_tensors_approximately_equal(&sequential_result, &parallel_result, 1e-5);
    }

    #[test]
    fn test_parallel_strategy_selection() {
        let engine = create_test_cpu_engine();

        // Small batch should prefer feature parallelism
        let small_input = BitNetTensor::zeros(&[1, 128, 512], DType::F32, &Device::Cpu).unwrap();
        let config = engine.get_parallel_config(&small_input).unwrap();
        let strategy = engine.determine_parallel_strategy(1, 128, 512, 2048, &config);
        assert_eq!(strategy, ParallelStrategy::FeatureParallel);

        // Large batch should prefer batch parallelism
        let large_input = BitNetTensor::zeros(&[32, 128, 512], DType::F32, &Device::Cpu).unwrap();
        let config = engine.get_parallel_config(&large_input).unwrap();
        let strategy = engine.determine_parallel_strategy(32, 128, 512, 2048, &config);
        assert_eq!(strategy, ParallelStrategy::BatchParallel);
    }

    #[test]
    fn test_simd_kernel_selection() {
        let engine = create_test_cpu_engine();
        let cpu_features = detect_cpu_features();

        let kernel = engine.get_optimized_i2s_kernel(&cpu_features).unwrap();

        #[cfg(target_arch = "x86_64")]
        {
            if cpu_features.has_avx2 {
                assert!(kernel.name().contains("AVX2"));
            }
        }
    }
}
```

### Performance Tests
```rust
#[cfg(test)]
mod performance_tests {
    #[test]
    fn benchmark_parallel_vs_sequential() {
        let engine = create_test_cpu_engine();
        let input = create_large_test_input();

        let sequential_time = time_execution(|| {
            engine.forward_sequential(&input, 0).unwrap()
        });

        let parallel_time = time_execution(|| {
            engine.forward_parallel(&input, 0).unwrap()
        });

        let speedup = sequential_time.as_secs_f64() / parallel_time.as_secs_f64();
        let expected_min_speedup = (rayon::current_num_threads() as f64 * 0.5).max(1.5);

        assert!(speedup >= expected_min_speedup,
                "Parallel speedup {} below expected minimum {}",
                speedup, expected_min_speedup);
    }
}
```

## Performance Considerations

1. **Thread Pool Reuse**: Avoid creating new threads for each inference
2. **Memory Locality**: Optimize data layout for cache efficiency
3. **Work Distribution**: Balance load across available cores
4. **NUMA Awareness**: Consider NUMA topology for large systems

## Risk Assessment

**Low Risk Changes:**
- Adding parallel configuration infrastructure
- Implementing basic batch parallelization

**Medium Risk Changes:**
- Changing model access patterns
- Implementing SIMD kernels

**High Risk Changes:**
- Modifying core inference flow
- Changing tensor layout assumptions

**Mitigation Strategies:**
- Comprehensive correctness testing against sequential implementation
- Performance regression testing
- Gradual rollout with feature flags
- Fallback to sequential implementation on errors

## Acceptance Criteria

- [ ] Parallel implementation produces identical results to sequential (within floating point precision)
- [ ] Achieves minimum 50% of theoretical speedup on multi-core systems
- [ ] Automatic selection of optimal parallelization strategy
- [ ] SIMD optimizations active on compatible hardware
- [ ] Performance regression < 5% on single-core systems
- [ ] Comprehensive test coverage (>95% line coverage)
- [ ] Memory usage within 20% of sequential implementation

## Related Issues/PRs

- **Related to**: SIMD kernel optimization
- **Depends on**: CPU feature detection infrastructure
- **Blocks**: Performance optimization for production inference
- **References**: Quantization algorithm performance improvements

## Additional Context

This implementation is critical for making BitNet.rs competitive with other inference frameworks on CPU hardware. The parallel execution should maintain numerical accuracy while maximizing utilization of available CPU resources, especially on modern multi-core processors commonly used for AI inference workloads.