# [Backend] Implement production-ready CPU and GPU backend functions

## Problem Description

Multiple backend functions in both CPU and GPU implementations are currently placeholder stubs that don't perform actual operations. These need to be replaced with production-ready implementations to enable real neural network inference.

## Environment
- **Affected Files**:
  - `crates/bitnet-inference/src/cpu.rs` - CPU backend functions
  - `crates/bitnet-inference/src/gpu.rs` - GPU backend functions
- **Affected Functions**: tokenize, detokenize, is_eos_token, forward_parallel, forward_gpu, etc.
- **Impact**: Core inference functionality, tokenization, GPU acceleration

## Issues Identified

### 1. Tokenization Stubs (CPU & GPU Backends)

**Current Implementation** (Both CPU and GPU):
```rust
fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
    // Placeholder implementation - in practice would use a proper tokenizer
    Ok(text.chars().map(|c| c as u32).collect())
}
```

**Problem**: Character-to-ASCII conversion instead of proper tokenization.

### 2. Detokenization Stubs (CPU & GPU Backends)

**Current Implementation**:
```rust
fn detokenize(&self, tokens: &[u32]) -> Result<String> {
    // Placeholder implementation
    Ok(tokens.iter().map(|&t| (t as u8) as char).collect())
}
```

**Problem**: ASCII-to-character conversion instead of proper detokenization.

### 3. EOS Token Detection Stubs

**Current Implementation**:
```rust
fn is_eos_token(&self, token: u32) -> bool {
    token == 0 // Placeholder - actual EOS token ID would come from tokenizer
}
```

**Problem**: Hardcoded EOS token ID instead of tokenizer-aware detection.

### 4. Forward Pass Stubs

**CPU Forward Parallel**:
```rust
fn forward_parallel(&self, input: &BitNetTensor, _step: usize) -> Result<BitNetTensor> {
    // For now, create a placeholder result
    let result = BitNetTensor::zeros(&[1, 32000], candle_core::DType::F32, &candle_core::Device::Cpu)?;
    Ok(result)
}
```

**GPU Forward Pass**:
```rust
fn forward_gpu(&self, input: &BitNetTensor, _step: usize) -> Result<BitNetTensor> {
    // For now, create a placeholder result
    let result = BitNetTensor::zeros(&[1, 32000], candle_core::DType::F32, &self.backend.device)?;
    Ok(result)
}
```

**Problem**: Return placeholder tensors instead of performing actual inference.

## Root Cause Analysis

1. **Development Phase**: Stubs were created to enable basic testing and integration
2. **Tokenizer Integration**: Requires proper integration with `bitnet-tokenizers` crate
3. **Model Integration**: Forward passes need integration with actual model implementations
4. **Device Management**: GPU functions need proper CUDA memory and compute management

## Impact Assessment
- **Severity**: Critical (for production use)
- **Impact**:
  - No actual text processing possible
  - No real neural network inference
  - Invalid inference results
  - Cannot process real-world inputs
- **Affected Components**: All inference pipelines, text processing, model execution

## Proposed Solution

Implement production-ready backend functions with proper tokenizer integration and model execution.

### Implementation Plan

#### 1. Tokenization Integration

**A. CPU Backend Tokenization**:
```rust
impl CpuBackend {
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        // Use the configured tokenizer from the engine
        let tokenizer = &self.tokenizer;
        let encoding = tokenizer.encode(text)?;
        Ok(encoding.get_ids().to_vec())
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        let tokenizer = &self.tokenizer;
        let decoded = tokenizer.decode(tokens)?;
        Ok(decoded)
    }

    fn is_eos_token(&self, token: u32) -> bool {
        let tokenizer = &self.tokenizer;
        token == tokenizer.eos_token_id()
    }
}
```

**B. GPU Backend Tokenization** (similar implementation with GPU-aware optimizations):
```rust
impl GpuBackend {
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        // Use the same tokenizer as CPU but prepare for GPU tensor transfer
        let tokenizer = &self.tokenizer;
        let encoding = tokenizer.encode(text)?;
        let tokens = encoding.get_ids().to_vec();

        // Pre-allocate GPU memory for efficient transfer if needed
        self.memory_manager.prefetch_tokens(&tokens)?;
        Ok(tokens)
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        // GPU tokens might need to be transferred to CPU first
        let cpu_tokens = self.transfer_tokens_to_cpu(tokens)?;
        let tokenizer = &self.tokenizer;
        let decoded = tokenizer.decode(&cpu_tokens)?;
        Ok(decoded)
    }
}
```

#### 2. Forward Pass Implementation

**A. CPU Parallel Forward Pass**:
```rust
impl CpuInferenceEngine {
    fn forward_parallel(&self, input: &BitNetTensor, step: usize) -> Result<BitNetTensor> {
        let model = self.model.read();

        // Parallel layer processing using rayon
        let mut hidden_states = input.clone();

        // Process transformer layers in parallel where possible
        for layer_idx in 0..model.config().num_layers {
            hidden_states = self.process_layer_parallel(&hidden_states, layer_idx, step)?;
        }

        // Final output projection
        let logits = self.apply_output_projection(&hidden_states)?;
        Ok(logits)
    }

    fn process_layer_parallel(&self, input: &BitNetTensor, layer_idx: usize, step: usize) -> Result<BitNetTensor> {
        let model = self.model.read();
        let layer = model.get_layer(layer_idx);

        // Pre-layer normalization
        let normed_input = layer.layer_norm_pre().forward(input)?;

        // Multi-head attention with parallel head processing
        let attention_output = self.parallel_attention(&normed_input, layer, step)?;

        // Residual connection
        let attention_residual = input.add(&attention_output)?;

        // Feed-forward network
        let ff_input = layer.layer_norm_post().forward(&attention_residual)?;
        let ff_output = layer.feed_forward().forward(&ff_input)?;

        // Final residual connection
        let output = attention_residual.add(&ff_output)?;
        Ok(output)
    }

    fn parallel_attention(&self, input: &BitNetTensor, layer: &TransformerLayer, step: usize) -> Result<BitNetTensor> {
        use rayon::prelude::*;

        let config = self.model.read().config();
        let num_heads = config.num_heads;
        let head_dim = config.hidden_size / num_heads;

        // Compute Q, K, V projections
        let q = layer.query_projection().forward(input)?;
        let k = layer.key_projection().forward(input)?;
        let v = layer.value_projection().forward(input)?;

        // Process attention heads in parallel
        let head_outputs: Result<Vec<_>> = (0..num_heads)
            .into_par_iter()
            .map(|head_idx| {
                self.compute_attention_head(&q, &k, &v, head_idx, head_dim, step)
            })
            .collect();

        let head_outputs = head_outputs?;

        // Concatenate head outputs
        let concatenated = BitNetTensor::cat(&head_outputs, -1)?;

        // Final output projection
        let output = layer.output_projection().forward(&concatenated)?;
        Ok(output)
    }
}
```

**B. GPU Forward Pass**:
```rust
impl GpuInferenceEngine {
    fn forward_gpu(&self, input: &BitNetTensor, step: usize) -> Result<BitNetTensor> {
        let compute_start = Instant::now();

        // Ensure input is on GPU
        let gpu_input = self.backend.ensure_gpu_tensor(input)?;

        let model = self.model.read();

        // GPU-optimized forward pass
        let mut hidden_states = gpu_input;

        // Process layers with GPU kernels
        for layer_idx in 0..model.config().num_layers {
            hidden_states = self.process_layer_gpu(&hidden_states, layer_idx, step)?;
        }

        // Final output projection
        let logits = self.apply_output_projection_gpu(&hidden_states)?;

        // Update compute metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.compute_time_ms = compute_start.elapsed().as_millis() as f64;
            metrics.gpu_utilization = self.backend.get_gpu_utilization()?;
        }

        Ok(logits)
    }

    fn process_layer_gpu(&self, input: &BitNetTensor, layer_idx: usize, step: usize) -> Result<BitNetTensor> {
        let model = self.model.read();
        let layer = model.get_layer(layer_idx);

        // Use GPU kernels for operations
        let normed_input = self.layer_norm_gpu(input, layer.layer_norm_pre())?;
        let attention_output = self.gpu_attention(&normed_input, layer, step)?;
        let attention_residual = self.add_tensors_gpu(input, &attention_output)?;
        let ff_input = self.layer_norm_gpu(&attention_residual, layer.layer_norm_post())?;
        let ff_output = self.feed_forward_gpu(&ff_input, layer)?;
        let output = self.add_tensors_gpu(&attention_residual, &ff_output)?;

        Ok(output)
    }

    fn gpu_attention(&self, input: &BitNetTensor, layer: &TransformerLayer, step: usize) -> Result<BitNetTensor> {
        // Launch CUDA kernels for attention computation
        let q = self.gpu_linear_projection(input, layer.query_projection())?;
        let k = self.gpu_linear_projection(input, layer.key_projection())?;
        let v = self.gpu_linear_projection(input, layer.value_projection())?;

        // Use optimized attention kernel
        let attention_output = self.backend.launch_attention_kernel(&q, &k, &v, step)?;

        // Output projection
        let output = self.gpu_linear_projection(&attention_output, layer.output_projection())?;
        Ok(output)
    }
}
```

#### 3. Backend Structure Updates

**A. Enhanced Backend Traits**:
```rust
pub trait Backend: Send + Sync {
    fn device_type(&self) -> DeviceType;
    fn tokenize(&self, text: &str) -> Result<Vec<u32>>;
    fn detokenize(&self, tokens: &[u32]) -> Result<String>;
    fn is_eos_token(&self, token: u32) -> bool;

    // Model execution
    fn forward(&self, model: &dyn Model, input: &BitNetTensor) -> Result<BitNetTensor>;

    // Device-specific operations
    fn prepare_input(&self, tokens: &[u32]) -> Result<BitNetTensor>;
    fn extract_logits(&self, output: &BitNetTensor) -> Result<Vec<f32>>;
}
```

**B. Tokenizer Integration**:
```rust
pub struct CpuBackend {
    device: candle_core::Device,
    tokenizer: Arc<dyn Tokenizer>,
    thread_pool: rayon::ThreadPool,
}

pub struct GpuBackend {
    device: candle_core::Device,
    tokenizer: Arc<dyn Tokenizer>,
    memory_manager: GpuMemoryManager,
    cuda_context: CudaContext,
}
```

## Testing Strategy
- **Unit Tests**: Test each backend function individually
- **Integration Tests**: Test complete inference pipelines
- **Tokenization Tests**: Verify tokenize/detokenize round-trip consistency
- **Performance Tests**: Benchmark CPU vs GPU performance
- **Memory Tests**: Verify GPU memory management
- **Cross-validation Tests**: Compare results with reference implementations

## Implementation Tasks

### Phase 1: Tokenization Integration
- [ ] Integrate tokenizer into CpuBackend constructor
- [ ] Implement CpuBackend::tokenize with proper tokenizer
- [ ] Implement CpuBackend::detokenize with proper tokenizer
- [ ] Implement CpuBackend::is_eos_token with tokenizer
- [ ] Replicate tokenization functions for GpuBackend
- [ ] Add GPU memory management for token transfers

### Phase 2: CPU Forward Pass
- [ ] Implement CpuInferenceEngine::forward_parallel
- [ ] Add parallel layer processing with rayon
- [ ] Implement parallel attention computation
- [ ] Add SIMD optimizations for CPU operations
- [ ] Implement layer normalization and feed-forward networks

### Phase 3: GPU Forward Pass
- [ ] Implement GpuInferenceEngine::forward_gpu
- [ ] Add CUDA kernel launches for layer operations
- [ ] Implement GPU attention kernels
- [ ] Add mixed precision support (FP16/BF16)
- [ ] Implement tensor core utilization

### Phase 4: Performance Optimization
- [ ] Add batch processing support
- [ ] Implement KV-cache for attention
- [ ] Add dynamic memory allocation
- [ ] Optimize GPU memory transfers
- [ ] Add performance profiling

## Acceptance Criteria
- [ ] All backend functions perform actual operations (no placeholders)
- [ ] Tokenization uses proper tokenizer from bitnet-tokenizers crate
- [ ] Forward passes produce valid neural network outputs
- [ ] CPU parallel processing shows performance improvements
- [ ] GPU forward pass utilizes CUDA acceleration
- [ ] Memory management is efficient for large models
- [ ] Cross-validation tests pass with reference implementations
- [ ] Performance meets target metrics (see below)

## Performance Targets
- **CPU Tokenization**: >1000 tokens/second
- **GPU Tokenization**: >5000 tokens/second with batching
- **CPU Forward Pass**: >10 tokens/second (1B parameter model)
- **GPU Forward Pass**: >50 tokens/second (1B parameter model)
- **Memory Usage**: <2x model size for inference

## Dependencies
- Integration with `bitnet-tokenizers` crate
- CUDA toolkit for GPU implementations
- Rayon for CPU parallelization
- Candle tensor operations
- Model loading from `bitnet-models`

## Labels
- `backend`
- `core-functionality`
- `cpu`
- `gpu`
- `priority-critical`
- `performance`

## Related Issues
- Tokenizer integration
- GPU acceleration
- Performance optimization
- Production readiness
