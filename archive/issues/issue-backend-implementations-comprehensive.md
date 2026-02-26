# [Backend] Implement production-ready CPU and GPU backend functions with comprehensive tokenization and inference

## Problem Description

Multiple backend functions in both CPU and GPU implementations are currently placeholder stubs that don't perform actual operations. These placeholders prevent real neural network inference, proper tokenization, and GPU acceleration. The current implementations return hardcoded values or create empty tensors instead of executing the intended operations, making the inference system non-functional for production use.

## Environment
- **Affected Files**:
  - `crates/bitnet-inference/src/cpu.rs` - CPU backend functions
  - `crates/bitnet-inference/src/gpu.rs` - GPU backend functions
- **Critical Functions**:
  - Tokenization: `tokenize`, `detokenize`, `is_eos_token`
  - Inference: `forward_parallel`, `forward_gpu`, `forward_mixed_precision`
  - Memory Management: `ensure_gpu_tensor`, `process_batch_gpu`
- **Hardware**: Multi-core CPUs (x86_64, ARM64), NVIDIA GPUs (RTX, Tesla, H100)
- **Feature Flags**: `--no-default-features --features cpu` or `--features gpu`

## Reproduction Steps

1. Build BitNet-rs with backend features:
   ```bash
   cargo build --no-default-features --features cpu
   cargo build --no-default-features --features gpu  # If CUDA available
   ```

2. Attempt inference with text input:
   ```bash
   cargo run -p xtask -- infer --model test-model.gguf --prompt "Hello world" --backend cpu
   cargo run -p xtask -- infer --model test-model.gguf --prompt "Hello world" --backend gpu
   ```

3. Observe the placeholder behavior in logs and outputs

**Expected Results**:
- Text should be properly tokenized using model-specific tokenizer
- Forward passes should produce valid logits for next token prediction
- GPU backend should utilize CUDA acceleration
- Generated text should be coherent and model-appropriate

**Actual Results**:
- Tokenization converts characters to ASCII values (invalid)
- Forward passes return zero tensors (no actual computation)
- GPU operations don't utilize GPU hardware
- Generated output is meaningless due to placeholder operations

## Root Cause Analysis

### Current Placeholder Implementations

#### 1. Tokenization Stubs (Both CPU and GPU)

```rust
fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
    // Placeholder implementation - in practice would use a proper tokenizer
    Ok(text.chars().map(|c| c as u32).collect())
}

fn detokenize(&self, tokens: &[u32]) -> Result<String> {
    // Placeholder implementation
    Ok(tokens.iter().map(|&t| (t as u8) as char).collect())
}

fn is_eos_token(&self, token: u32) -> bool {
    token == 0 // Placeholder - actual EOS token ID would come from tokenizer
}
```

**Problems**:
- Character-to-ASCII conversion breaks Unicode and subword tokenization
- No vocabulary awareness or model-specific token handling
- Hardcoded EOS token ID incompatible with real tokenizers

#### 2. Forward Pass Stubs

```rust
// CPU Backend
fn forward_parallel(&self, input: &BitNetTensor, _step: usize) -> Result<BitNetTensor> {
    // For now, create a placeholder result
    let result = BitNetTensor::zeros(&[1, 32000], candle_core::DType::F32, &candle_core::Device::Cpu)?;
    Ok(result)
}

// GPU Backend
fn forward_gpu(&self, input: &BitNetTensor, _step: usize) -> Result<BitNetTensor> {
    // For now, create a placeholder result
    let result = BitNetTensor::zeros(&[1, 32000], candle_core::DType::F32, &self.backend.device)?;
    Ok(result)
}
```

**Problems**:
- Returns zero tensors instead of computing logits
- No actual model computation or weight usage
- Hardcoded output dimensions don't match input or model requirements

#### 3. GPU-Specific Stubs

```rust
fn ensure_gpu_tensor(&self, tensor: &BitNetTensor) -> Result<BitNetTensor> {
    // Placeholder - would transfer to GPU if needed
    Ok(tensor.clone())
}

fn process_batch_gpu(&self, batch: &[BitNetTensor]) -> Result<Vec<BitNetTensor>> {
    // Placeholder implementation
    Ok(batch.to_vec())
}
```

**Problems**:
- No actual GPU memory transfer
- No batch processing optimization
- Missing CUDA kernel utilization

## Impact Assessment

- **Severity**: Critical (complete inference system failure)
- **Functional Impact**:
  - No real text processing possible
  - No neural network inference
  - GPU acceleration completely non-functional
  - Cannot process real-world inputs or generate meaningful outputs

- **Production Impact**:
  - System cannot be deployed for actual use
  - Validation and benchmarking impossible
  - Cross-validation fails due to invalid outputs
  - Demo and evaluation scenarios fail

- **Development Impact**:
  - Cannot test real inference workflows
  - Performance optimization blocked
  - Model accuracy validation impossible
  - Integration testing meaningless

## Proposed Solution

Implement comprehensive, production-ready backend functions with proper tokenizer integration, real model execution, and efficient GPU utilization.

### Technical Implementation

#### 1. Tokenizer Integration Architecture

```rust
use bitnet_tokenizers::{Tokenizer, UniversalTokenizer, TokenizerError};
use std::sync::Arc;

pub trait Backend: Send + Sync {
    fn device_type(&self) -> DeviceType;

    // Core tokenization operations
    fn tokenize(&self, text: &str) -> Result<Vec<u32>>;
    fn detokenize(&self, tokens: &[u32]) -> Result<String>;
    fn is_eos_token(&self, token: u32) -> bool;
    fn get_vocab_size(&self) -> usize;

    // Model execution
    fn forward(&self, model: &dyn Model, input: &BitNetTensor, step: usize) -> Result<BitNetTensor>;

    // Device-specific operations
    fn prepare_input(&self, tokens: &[u32]) -> Result<BitNetTensor>;
    fn extract_logits(&self, output: &BitNetTensor) -> Result<Vec<f32>>;

    // Batch processing
    fn process_batch(&self, model: &dyn Model, inputs: &[BitNetTensor]) -> Result<Vec<BitNetTensor>>;
}

pub struct CpuBackend {
    device: candle_core::Device,
    tokenizer: Arc<dyn Tokenizer>,
    thread_pool: rayon::ThreadPool,
    cpu_features: CpuFeatures,
}

pub struct GpuBackend {
    device: candle_core::Device,
    tokenizer: Arc<dyn Tokenizer>,
    memory_manager: GpuMemoryManager,
    cuda_context: CudaContext,
    stream_manager: CudaStreamManager,
}

#[derive(Debug, Clone)]
pub struct CpuFeatures {
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_neon: bool,
    pub num_cores: usize,
    pub cache_sizes: CacheSizes,
}
```

#### 2. CPU Backend Implementation

```rust
impl CpuBackend {
    pub fn new(tokenizer: Arc<dyn Tokenizer>) -> Result<Self> {
        let device = candle_core::Device::Cpu;
        let num_threads = std::thread::available_parallelism()?.get();

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()?;

        let cpu_features = Self::detect_cpu_features()?;

        log::info!("Initialized CPU backend with {} threads, features: {:?}",
                  num_threads, cpu_features);

        Ok(Self {
            device,
            tokenizer,
            thread_pool,
            cpu_features,
        })
    }

    fn detect_cpu_features() -> Result<CpuFeatures> {
        Ok(CpuFeatures {
            has_avx2: is_x86_feature_detected!("avx2"),
            has_avx512: is_x86_feature_detected!("avx512f"),
            has_neon: cfg!(target_arch = "aarch64"),
            num_cores: std::thread::available_parallelism()?.get(),
            cache_sizes: CacheSizes::detect()?,
        })
    }
}

impl Backend for CpuBackend {
    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        // Use actual tokenizer implementation
        let encoding = self.tokenizer.encode(text)?;
        let tokens = encoding.get_ids().to_vec();

        log::debug!("Tokenized '{}' -> {} tokens",
                   text.chars().take(50).collect::<String>(),
                   tokens.len());

        Ok(tokens)
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        // Use actual tokenizer implementation
        let text = self.tokenizer.decode(tokens)?;

        log::debug!("Detokenized {} tokens -> '{}'",
                   tokens.len(),
                   text.chars().take(50).collect::<String>());

        Ok(text)
    }

    fn is_eos_token(&self, token: u32) -> bool {
        self.tokenizer.eos_token_id() == token
    }

    fn get_vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    fn forward(&self, model: &dyn Model, input: &BitNetTensor, step: usize) -> Result<BitNetTensor> {
        // Delegate to parallel implementation
        self.forward_parallel(model, input, step)
    }

    fn forward_parallel(&self, model: &dyn Model, input: &BitNetTensor, step: usize) -> Result<BitNetTensor> {
        let forward_start = std::time::Instant::now();

        // Ensure input is on CPU device
        let cpu_input = input.to_device(&self.device)?;

        // Execute model forward pass with parallel processing
        let output = self.thread_pool.install(|| {
            self.execute_model_forward(model, &cpu_input, step)
        })?;

        log::debug!("CPU forward pass completed in {:?}", forward_start.elapsed());

        Ok(output)
    }

    fn execute_model_forward(&self, model: &dyn Model, input: &BitNetTensor, step: usize) -> Result<BitNetTensor> {
        use rayon::prelude::*;

        let config = model.config();
        let batch_size = input.shape()[0];
        let sequence_length = input.shape()[1];

        // Input embeddings
        let mut hidden_states = model.embed_tokens(input)?;

        // Add positional embeddings
        hidden_states = model.add_positional_embeddings(&hidden_states, step)?;

        // Process transformer layers with potential parallelization
        for layer_idx in 0..config.num_layers {
            hidden_states = self.process_transformer_layer(
                model,
                &hidden_states,
                layer_idx,
                step
            )?;
        }

        // Final layer normalization
        hidden_states = model.final_layer_norm(&hidden_states)?;

        // Output projection to vocabulary
        let logits = model.lm_head(&hidden_states)?;

        // Extract logits for the last token in each sequence
        let output_logits = if sequence_length > 1 {
            // Take last token logits for generation
            logits.i((.., sequence_length - 1, ..))?
        } else {
            logits.squeeze(1)?
        };

        Ok(output_logits)
    }

    fn process_transformer_layer(
        &self,
        model: &dyn Model,
        hidden_states: &BitNetTensor,
        layer_idx: usize,
        step: usize,
    ) -> Result<BitNetTensor> {
        let layer = model.get_layer(layer_idx);

        // Pre-layer normalization
        let normed_input = layer.layer_norm_pre(hidden_states)?;

        // Self-attention with parallel head processing
        let attention_output = self.parallel_self_attention(&layer, &normed_input, step)?;

        // Residual connection
        let attention_residual = hidden_states.add(&attention_output)?;

        // Post-attention layer normalization
        let ff_input = layer.layer_norm_post(&attention_residual)?;

        // Feed-forward network
        let ff_output = self.parallel_feed_forward(&layer, &ff_input)?;

        // Final residual connection
        let output = attention_residual.add(&ff_output)?;

        Ok(output)
    }

    fn parallel_self_attention(
        &self,
        layer: &TransformerLayer,
        input: &BitNetTensor,
        step: usize,
    ) -> Result<BitNetTensor> {
        use rayon::prelude::*;

        let config = layer.config();
        let num_heads = config.num_heads;
        let head_dim = config.hidden_size / num_heads;

        // Compute Q, K, V projections with BitNet quantization
        let q = layer.query_projection().forward(input)?;
        let k = layer.key_projection().forward(input)?;
        let v = layer.value_projection().forward(input)?;

        // Reshape for multi-head attention
        let q = self.reshape_for_attention(&q, num_heads, head_dim)?;
        let k = self.reshape_for_attention(&k, num_heads, head_dim)?;
        let v = self.reshape_for_attention(&v, num_heads, head_dim)?;

        // Process attention heads in parallel
        let head_outputs: Result<Vec<BitNetTensor>> = (0..num_heads)
            .into_par_iter()
            .map(|head_idx| {
                let q_head = q.i((.., head_idx, .., ..))?;
                let k_head = k.i((.., head_idx, .., ..))?;
                let v_head = v.i((.., head_idx, .., ..))?;

                self.compute_attention_head(&q_head, &k_head, &v_head, head_dim, step)
            })
            .collect();

        let head_outputs = head_outputs?;

        // Concatenate head outputs
        let concatenated = BitNetTensor::cat(&head_outputs, -1)?;

        // Final output projection
        let output = layer.output_projection().forward(&concatenated)?;

        Ok(output)
    }

    fn compute_attention_head(
        &self,
        q: &BitNetTensor,
        k: &BitNetTensor,
        v: &BitNetTensor,
        head_dim: usize,
        step: usize,
    ) -> Result<BitNetTensor> {
        // Scaled dot-product attention
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Compute attention scores
        let scores = q.matmul(&k.transpose(-2, -1)?)?;
        let scaled_scores = (scores * scale)?;

        // Apply causal mask for autoregressive generation
        let masked_scores = self.apply_causal_mask(&scaled_scores, step)?;

        // Softmax to get attention weights
        let attention_weights = masked_scores.softmax(-1)?;

        // Apply attention to values
        let output = attention_weights.matmul(v)?;

        Ok(output)
    }

    fn parallel_feed_forward(&self, layer: &TransformerLayer, input: &BitNetTensor) -> Result<BitNetTensor> {
        // Use CPU-optimized kernels for feed-forward computation
        let intermediate = layer.feed_forward_up(input)?;
        let activated = self.apply_activation(&intermediate, layer.activation_type())?;
        let output = layer.feed_forward_down(&activated)?;

        Ok(output)
    }

    fn prepare_input(&self, tokens: &[u32]) -> Result<BitNetTensor> {
        let tensor = BitNetTensor::new(
            tokens.to_vec(),
            &[1, tokens.len()],
            candle_core::DType::U32,
            &self.device,
        )?;

        Ok(tensor)
    }

    fn extract_logits(&self, output: &BitNetTensor) -> Result<Vec<f32>> {
        // Extract logits as Vec<f32> for token sampling
        let logits_flat = output.flatten_all()?;
        let logits_vec = logits_flat.to_vec1::<f32>()?;

        Ok(logits_vec)
    }

    fn process_batch(&self, model: &dyn Model, inputs: &[BitNetTensor]) -> Result<Vec<BitNetTensor>> {
        use rayon::prelude::*;

        // Process batch in parallel on CPU
        let outputs: Result<Vec<BitNetTensor>> = inputs
            .par_iter()
            .enumerate()
            .map(|(i, input)| {
                log::debug!("Processing batch item {} of {}", i + 1, inputs.len());
                self.forward(model, input, 0)
            })
            .collect();

        outputs
    }
}
```

#### 3. GPU Backend Implementation

```rust
impl GpuBackend {
    pub fn new(tokenizer: Arc<dyn Tokenizer>, device_id: usize) -> Result<Self> {
        // Initialize CUDA device
        let device = candle_core::Device::cuda_if_available(device_id)?;

        // Initialize GPU memory manager
        let memory_manager = GpuMemoryManager::new(device_id, true)?;

        // Create CUDA context and streams
        let cuda_context = CudaContext::new(device_id)?;
        let stream_manager = CudaStreamManager::new(&cuda_context, 4)?; // 4 streams

        log::info!("Initialized GPU backend on device {}", device_id);

        Ok(Self {
            device,
            tokenizer,
            memory_manager,
            cuda_context,
            stream_manager,
        })
    }
}

impl Backend for GpuBackend {
    fn device_type(&self) -> DeviceType {
        DeviceType::Gpu
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        // Use same tokenizer as CPU, but prepare for GPU transfer
        let encoding = self.tokenizer.encode(text)?;
        let tokens = encoding.get_ids().to_vec();

        // Pre-allocate GPU memory for efficient transfer if batch processing
        if tokens.len() > 1000 {
            self.memory_manager.prefetch_tokens(&tokens)?;
        }

        log::debug!("GPU tokenized '{}' -> {} tokens",
                   text.chars().take(50).collect::<String>(),
                   tokens.len());

        Ok(tokens)
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        // Note: Tokenization is typically done on CPU even in GPU backends
        let text = self.tokenizer.decode(tokens)?;

        log::debug!("GPU detokenized {} tokens -> '{}'",
                   tokens.len(),
                   text.chars().take(50).collect::<String>());

        Ok(text)
    }

    fn is_eos_token(&self, token: u32) -> bool {
        self.tokenizer.eos_token_id() == token
    }

    fn get_vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    fn forward(&self, model: &dyn Model, input: &BitNetTensor, step: usize) -> Result<BitNetTensor> {
        self.forward_gpu(model, input, step)
    }

    fn forward_gpu(&self, model: &dyn Model, input: &BitNetTensor, step: usize) -> Result<BitNetTensor> {
        let compute_start = std::time::Instant::now();

        // Ensure input is on GPU
        let gpu_input = self.ensure_gpu_tensor(input)?;

        // Select appropriate precision based on model and hardware
        let output = if self.supports_mixed_precision()? {
            self.forward_mixed_precision(model, &gpu_input, step)?
        } else {
            self.forward_full_precision(model, &gpu_input, step)?
        };

        // Update performance metrics
        let compute_time = compute_start.elapsed();
        log::debug!("GPU forward pass completed in {:?}", compute_time);

        Ok(output)
    }

    fn forward_mixed_precision(&self, model: &dyn Model, input: &BitNetTensor, step: usize) -> Result<BitNetTensor> {
        // Use FP16 for most operations, FP32 for critical operations

        // Convert input to FP16 for efficiency
        let fp16_input = input.to_dtype(candle_core::DType::F16)?;

        // Execute model with mixed precision
        let mut hidden_states = model.embed_tokens(&fp16_input)?;

        // Keep embeddings in FP16
        hidden_states = model.add_positional_embeddings(&hidden_states, step)?;

        // Process transformer layers
        for layer_idx in 0..model.config().num_layers {
            hidden_states = self.process_gpu_layer(model, &hidden_states, layer_idx, step)?;
        }

        // Convert to FP32 for final layer norm (stability)
        let fp32_hidden = hidden_states.to_dtype(candle_core::DType::F32)?;
        let normalized = model.final_layer_norm(&fp32_hidden)?;

        // Output projection in FP32 for precision
        let logits = model.lm_head(&normalized)?;

        Ok(logits)
    }

    fn process_gpu_layer(
        &self,
        model: &dyn Model,
        hidden_states: &BitNetTensor,
        layer_idx: usize,
        step: usize,
    ) -> Result<BitNetTensor> {
        let layer = model.get_layer(layer_idx);

        // Use CUDA kernels for layer operations
        let normed_input = self.layer_norm_cuda(hidden_states, layer.layer_norm_pre())?;
        let attention_output = self.gpu_multi_head_attention(&layer, &normed_input, step)?;
        let attention_residual = self.add_tensors_cuda(hidden_states, &attention_output)?;
        let ff_input = self.layer_norm_cuda(&attention_residual, layer.layer_norm_post())?;
        let ff_output = self.feed_forward_cuda(&layer, &ff_input)?;
        let output = self.add_tensors_cuda(&attention_residual, &ff_output)?;

        Ok(output)
    }

    fn gpu_multi_head_attention(
        &self,
        layer: &TransformerLayer,
        input: &BitNetTensor,
        step: usize,
    ) -> Result<BitNetTensor> {
        // Launch optimized CUDA attention kernel
        let config = layer.config();

        // Compute QKV projections on GPU
        let q = self.gpu_linear_projection(input, layer.query_projection())?;
        let k = self.gpu_linear_projection(input, layer.key_projection())?;
        let v = self.gpu_linear_projection(input, layer.value_projection())?;

        // Use fused attention kernel if available
        let attention_output = if self.has_tensor_cores()? {
            self.launch_tensor_core_attention(&q, &k, &v, step)?
        } else {
            self.launch_standard_attention(&q, &k, &v, step)?
        };

        // Output projection
        let output = self.gpu_linear_projection(&attention_output, layer.output_projection())?;

        Ok(output)
    }

    fn launch_tensor_core_attention(
        &self,
        q: &BitNetTensor,
        k: &BitNetTensor,
        v: &BitNetTensor,
        step: usize,
    ) -> Result<BitNetTensor> {
        // Use tensor core optimized attention
        self.cuda_context.launch_attention_kernel_tc(q, k, v, step)
    }

    fn launch_standard_attention(
        &self,
        q: &BitNetTensor,
        k: &BitNetTensor,
        v: &BitNetTensor,
        step: usize,
    ) -> Result<BitNetTensor> {
        // Use standard CUDA attention kernel
        self.cuda_context.launch_attention_kernel(q, k, v, step)
    }

    fn ensure_gpu_tensor(&self, tensor: &BitNetTensor) -> Result<BitNetTensor> {
        if tensor.device() == &self.device {
            Ok(tensor.clone())
        } else {
            // Transfer tensor to GPU with memory management
            let gpu_tensor = tensor.to_device(&self.device)?;
            log::debug!("Transferred tensor {:?} to GPU", tensor.shape());
            Ok(gpu_tensor)
        }
    }

    fn process_batch(&self, model: &dyn Model, inputs: &[BitNetTensor]) -> Result<Vec<BitNetTensor>> {
        self.process_batch_gpu(model, inputs)
    }

    fn process_batch_gpu(&self, model: &dyn Model, inputs: &[BitNetTensor]) -> Result<Vec<BitNetTensor>> {
        // Optimize batch processing with CUDA streams
        let batch_size = inputs.len();
        let stream_count = self.stream_manager.stream_count().min(batch_size);

        if batch_size <= 1 {
            // Single input, use main stream
            return Ok(vec![self.forward_gpu(model, &inputs[0], 0)?]);
        }

        // Distribute work across CUDA streams
        let chunks: Vec<&[BitNetTensor]> = inputs.chunks(
            (batch_size + stream_count - 1) / stream_count
        ).collect();

        let mut results = Vec::with_capacity(batch_size);

        // Launch work on each stream
        for (stream_id, chunk) in chunks.iter().enumerate() {
            let stream = self.stream_manager.get_stream(stream_id)?;

            for (i, input) in chunk.iter().enumerate() {
                // Transfer input to GPU asynchronously
                let gpu_input = self.async_transfer_to_gpu(input, &stream)?;

                // Execute forward pass on this stream
                let output = self.forward_gpu_stream(model, &gpu_input, i, &stream)?;

                results.push(output);
            }
        }

        // Synchronize all streams
        self.stream_manager.synchronize_all()?;

        Ok(results)
    }

    fn supports_mixed_precision(&self) -> Result<bool> {
        // Check if GPU supports FP16 operations efficiently
        Ok(self.cuda_context.compute_capability()?.0 >= 7) // Volta and newer
    }

    fn has_tensor_cores(&self) -> Result<bool> {
        // Check if GPU has tensor cores for accelerated operations
        Ok(self.cuda_context.compute_capability()?.0 >= 7)
    }
}
```

#### 4. Error Handling and Validation

```rust
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("Tokenization failed: {0}")]
    TokenizationError(#[from] TokenizerError),

    #[error("Model execution failed: {0}")]
    ModelError(String),

    #[error("GPU operation failed: {0}")]
    GpuError(String),

    #[error("Memory allocation failed: {0}")]
    MemoryError(String),

    #[error("Invalid input shape: expected {expected:?}, got {actual:?}")]
    InvalidShape { expected: Vec<usize>, actual: Vec<usize> },

    #[error("Device mismatch: tensor on {tensor_device:?}, expected {expected_device:?}")]
    DeviceMismatch { tensor_device: String, expected_device: String },
}

impl Backend for CpuBackend {
    fn forward(&self, model: &dyn Model, input: &BitNetTensor, step: usize) -> Result<BitNetTensor> {
        // Validate input
        self.validate_input(input, model)?;

        // Execute with error handling
        let result = self.forward_parallel(model, input, step)
            .map_err(|e| BackendError::ModelError(format!("CPU forward pass failed: {}", e)))?;

        // Validate output
        self.validate_output(&result, model)?;

        Ok(result)
    }

    fn validate_input(&self, input: &BitNetTensor, model: &dyn Model) -> Result<()> {
        let input_shape = input.shape();

        // Check dimensions
        if input_shape.len() != 2 {
            return Err(BackendError::InvalidShape {
                expected: vec![1, 0], // [batch_size, sequence_length]
                actual: input_shape.to_vec(),
            });
        }

        // Check sequence length
        let max_seq_len = model.config().max_position_embeddings;
        if input_shape[1] > max_seq_len {
            return Err(BackendError::InvalidShape {
                expected: vec![input_shape[0], max_seq_len],
                actual: input_shape.to_vec(),
            });
        }

        // Check device compatibility
        if input.device() != &self.device {
            return Err(BackendError::DeviceMismatch {
                tensor_device: format!("{:?}", input.device()),
                expected_device: format!("{:?}", self.device),
            });
        }

        Ok(())
    }

    fn validate_output(&self, output: &BitNetTensor, model: &dyn Model) -> Result<()> {
        let output_shape = output.shape();
        let expected_vocab_size = model.config().vocab_size;

        // Check output dimensions
        if output_shape.len() < 2 || output_shape[output_shape.len() - 1] != expected_vocab_size {
            return Err(BackendError::InvalidShape {
                expected: vec![1, expected_vocab_size],
                actual: output_shape.to_vec(),
            });
        }

        // Check for invalid values (NaN, Inf)
        let logits = self.extract_logits(output)?;
        for (i, &value) in logits.iter().enumerate() {
            if !value.is_finite() {
                return Err(BackendError::ModelError(
                    format!("Invalid logit value at position {}: {}", i, value)
                ));
            }
        }

        Ok(())
    }
}
```

## Implementation Plan

### Phase 1: Tokenizer Integration (Week 1-2)
- [ ] Integrate `bitnet-tokenizers` crate into backend constructors
- [ ] Implement proper `tokenize`, `detokenize`, and `is_eos_token` methods
- [ ] Add tokenizer validation and error handling
- [ ] Test tokenization with various text inputs and model types

### Phase 2: CPU Forward Pass Implementation (Week 3-4)
- [ ] Implement `forward_parallel` with real model execution
- [ ] Add transformer layer processing with attention and feed-forward
- [ ] Implement parallel multi-head attention computation
- [ ] Add SIMD optimizations for CPU operations
- [ ] Integrate with existing BitNet quantization kernels

### Phase 3: GPU Forward Pass Implementation (Week 5-6)
- [ ] Implement `forward_gpu` with CUDA kernel utilization
- [ ] Add mixed precision support (FP16/BF16)
- [ ] Implement tensor core optimizations
- [ ] Add efficient GPU memory management
- [ ] Create CUDA stream-based batch processing

### Phase 4: Performance Optimization & Testing (Week 7-8)
- [ ] Optimize memory allocation patterns
- [ ] Add comprehensive error handling and validation
- [ ] Create extensive test suite for both backends
- [ ] Performance benchmarking and optimization
- [ ] Cross-validation with reference implementations

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod backend_tests {
    use super::*;

    #[test]
    fn test_cpu_tokenization() {
        let tokenizer = create_test_tokenizer();
        let backend = CpuBackend::new(Arc::new(tokenizer)).unwrap();

        let text = "Hello, world! This is a test.";
        let tokens = backend.tokenize(text).unwrap();
        let decoded = backend.detokenize(&tokens).unwrap();

        // Tokenization should be reversible for most text
        assert!(!tokens.is_empty());
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_cpu_forward_pass() {
        let tokenizer = create_test_tokenizer();
        let backend = CpuBackend::new(Arc::new(tokenizer)).unwrap();
        let model = create_test_model();

        let input_tokens = vec![1, 2, 3, 4, 5]; // Example token sequence
        let input_tensor = backend.prepare_input(&input_tokens).unwrap();

        let output = backend.forward(&*model, &input_tensor, 0).unwrap();

        // Verify output shape and validity
        assert_eq!(output.shape()[0], 1); // batch size
        assert_eq!(output.shape()[1], model.config().vocab_size);

        let logits = backend.extract_logits(&output).unwrap();
        assert!(logits.iter().all(|&x| x.is_finite()));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_memory_transfer() {
        let tokenizer = create_test_tokenizer();
        let backend = GpuBackend::new(Arc::new(tokenizer), 0).unwrap();

        let cpu_tensor = BitNetTensor::zeros(&[1, 10], candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        let gpu_tensor = backend.ensure_gpu_tensor(&cpu_tensor).unwrap();

        assert_eq!(gpu_tensor.device().location(), candle_core::DeviceLocation::Cuda);
        assert_eq!(gpu_tensor.shape(), cpu_tensor.shape());
    }
}
```

### Integration Tests
```rust
#[cfg(test)]
mod integration_tests {
    #[test]
    fn test_end_to_end_inference() {
        // Test complete inference pipeline
        let tokenizer = UniversalTokenizer::from_model_path("test-model.gguf").unwrap();
        let model = BitNetModel::from_gguf("test-model.gguf").unwrap();
        let backend = CpuBackend::new(Arc::new(tokenizer)).unwrap();

        let prompt = "The quick brown fox";
        let max_tokens = 10;

        let mut generated_tokens = Vec::new();
        let mut current_tokens = backend.tokenize(prompt).unwrap();

        for step in 0..max_tokens {
            let input_tensor = backend.prepare_input(&current_tokens).unwrap();
            let output = backend.forward(&*model, &input_tensor, step).unwrap();
            let logits = backend.extract_logits(&output).unwrap();

            // Simple greedy sampling
            let next_token = argmax(&logits);

            if backend.is_eos_token(next_token) {
                break;
            }

            generated_tokens.push(next_token);
            current_tokens.push(next_token);
        }

        let generated_text = backend.detokenize(&generated_tokens).unwrap();

        assert!(!generated_text.is_empty());
        println!("Generated: {}", generated_text);
    }
}
```

## Performance Targets

### CPU Backend
- **Tokenization**: >1000 tokens/second
- **Forward Pass**: >10 tokens/second (1B parameter model)
- **Memory Usage**: <2x model size
- **Parallel Efficiency**: >70% scaling up to 8 cores

### GPU Backend
- **Tokenization**: >5000 tokens/second with batching
- **Forward Pass**: >50 tokens/second (1B parameter model)
- **Mixed Precision**: 30-50% speedup over FP32
- **Batch Processing**: >200 tokens/second for batch size 8+

## Acceptance Criteria

- [ ] All backend functions perform actual operations (no placeholders)
- [ ] Tokenization uses proper tokenizer from `bitnet-tokenizers` crate
- [ ] Forward passes produce valid neural network outputs with correct shapes
- [ ] CPU parallel processing shows measurable performance improvements
- [ ] GPU forward pass utilizes CUDA acceleration effectively
- [ ] Mixed precision support works on compatible hardware
- [ ] Memory management is efficient and prevents OOM errors
- [ ] Error handling provides clear, actionable error messages
- [ ] Cross-validation tests pass with reference implementations
- [ ] Performance meets or exceeds target metrics
- [ ] Integration tests demonstrate end-to-end functionality
- [ ] Documentation explains backend usage and configuration

## Dependencies

- `bitnet-tokenizers` crate for tokenization
- `rayon` for CPU parallelization
- `cudarc` or `candle-cuda` for GPU operations
- CUDA toolkit (11.8+ or 12.x) for GPU backend
- Existing BitNet model and tensor infrastructure

## Related Issues

- Tokenizer integration and universal tokenizer
- GPU memory management implementation
- CUDA kernel optimization
- Performance benchmarking and validation
- Cross-validation framework

## Labels
- `backend`
- `core-functionality`
- `cpu`
- `gpu`
- `priority-critical`
- `performance`
- `tokenization`
- `inference`
