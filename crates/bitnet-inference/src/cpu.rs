//! CPU backend implementation with Rayon parallelism

use crate::{Backend, DeviceInfo, DeviceType, KVCache, SamplingStrategy, StreamingConfig};
use bitnet_common::{
    BitNetConfig, BitNetTensor, GenerationConfig, InferenceError,
    PerformanceMetrics, Result
};
use bitnet_kernels::KernelProvider;
use bitnet_models::Model;
use candle_core::Device;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
use std::time::Instant;

/// CPU backend for inference with Rayon parallelism
pub struct CpuBackend {
    kernel_provider: Box<dyn KernelProvider>,
    device: Device,
    thread_pool: rayon::ThreadPool,
    performance_config: CpuPerformanceConfig,
}

/// CPU-specific performance configuration
#[derive(Debug, Clone)]
pub struct CpuPerformanceConfig {
    pub num_threads: usize,
    pub enable_parallel_layers: bool,
    pub enable_parallel_attention: bool,
    pub batch_size_threshold: usize,
    pub memory_pool_size_mb: usize,
}

impl Default for CpuPerformanceConfig {
    fn default() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
            enable_parallel_layers: true,
            enable_parallel_attention: true,
            batch_size_threshold: 4,
            memory_pool_size_mb: 256,
        }
    }
}

impl CpuBackend {
    /// Create a new CPU backend
    pub fn new() -> Result<Self> {
        Self::with_config(CpuPerformanceConfig::default())
    }

    /// Create CPU backend with custom configuration
    pub fn with_config(config: CpuPerformanceConfig) -> Result<Self> {
        let kernel_provider = bitnet_kernels::select_cpu_kernel()?;
        let device = Device::Cpu;

        // Create custom thread pool
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .thread_name(|i| format!("bitnet-cpu-{}", i))
            .build()
            .map_err(|e| InferenceError::GenerationFailed {
                reason: format!("Failed to create thread pool: {}", e)
            })?;

        Ok(Self {
            kernel_provider,
            device,
            thread_pool,
            performance_config: config,
        })
    }

    /// Get performance configuration
    pub fn performance_config(&self) -> &CpuPerformanceConfig {
        &self.performance_config
    }

    /// Update performance configuration
    pub fn update_performance_config(&mut self, config: CpuPerformanceConfig) -> Result<()> {
        // Recreate thread pool if thread count changed
        if config.num_threads != self.performance_config.num_threads {
            self.thread_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(config.num_threads)
                .thread_name(|i| format!("bitnet-cpu-{}", i))
                .build()
                .map_err(|e| InferenceError::GenerationFailed {
                    reason: format!("Failed to recreate thread pool: {}", e)
                })?;
        }

        self.performance_config = config;
        Ok(())
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "CPU-Rayon"
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        // Placeholder implementation - in practice would use a proper tokenizer
        Ok(text.chars().map(|c| c as u32).collect())
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        // Placeholder implementation - in practice would use a proper tokenizer
        Ok(tokens.iter().map(|&t| char::from(t as u8)).collect())
    }

    fn tokens_to_tensor(&self, tokens: &[u32]) -> Result<BitNetTensor> {
        BitNetTensor::from_slice(tokens, &[tokens.len()], &self.device)
    }

    fn is_eos_token(&self, token: u32) -> bool {
        token == 2 // Placeholder EOS token ID
    }

    fn clone_backend(&self) -> Box<dyn Backend> {
        Box::new(Self::with_config(self.performance_config.clone()).unwrap())
    }

    fn kernel_provider(&self) -> &dyn KernelProvider {
        self.kernel_provider.as_ref()
    }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            device_type: DeviceType::Cpu,
            memory_total: None, // Could query system memory
            memory_available: None, // Could query available memory
            compute_capability: Some(format!("CPU-{}-threads", self.performance_config.num_threads)),
        }
    }
}

/// CPU-specific inference engine with optimized parallel processing
pub struct CpuInferenceEngine {
    model: Arc<RwLock<Box<dyn Model<Config = BitNetConfig>>>>,
    backend: CpuBackend,
    cache: Arc<Mutex<KVCache>>,
    sampling: Arc<Mutex<SamplingStrategy>>,
    metrics: Arc<Mutex<PerformanceMetrics>>,
    config: CpuInferenceConfig,
}

/// CPU inference configuration
#[derive(Debug, Clone)]
pub struct CpuInferenceConfig {
    pub max_sequence_length: usize,
    pub enable_kv_cache: bool,
    pub enable_memory_pooling: bool,
    pub parallel_layer_threshold: usize,
    pub batch_processing: bool,
}

impl Default for CpuInferenceConfig {
    fn default() -> Self {
        Self {
            max_sequence_length: 2048,
            enable_kv_cache: true,
            enable_memory_pooling: true,
            parallel_layer_threshold: 8,
            batch_processing: true,
        }
    }
}

impl CpuInferenceEngine {
    /// Create a new CPU inference engine
    pub fn new(
        model: Box<dyn Model<Config = BitNetConfig>>,
        backend: CpuBackend,
        config: CpuInferenceConfig,
    ) -> Result<Self> {
        let model_config = model.config().clone();

        // Create KV cache with memory pooling if enabled
        let cache = if config.enable_memory_pooling {
            KVCache::with_memory_pool(
                &model_config,
                config.max_sequence_length,
                backend.performance_config().memory_pool_size_mb,
            )?
        } else {
            KVCache::new(&model_config, config.max_sequence_length)?
        };

        // Create sampling strategy with deterministic seeding
        let sampling_config = crate::SamplingConfig::default();
        let sampling = SamplingStrategy::new(sampling_config)?;

        Ok(Self {
            model: Arc::new(RwLock::new(model)),
            backend,
            cache: Arc::new(Mutex::new(cache)),
            sampling: Arc::new(Mutex::new(sampling)),
            metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
            config,
        })
    }

    /// Generate tokens with parallel processing
    pub fn generate_tokens_parallel(
        &self,
        input_tokens: &[u32],
        generation_config: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        let start_time = Instant::now();
        let mut generated_tokens = Vec::new();
        let mut current_tokens = input_tokens.to_vec();

        // Reset cache
        {
            let mut cache = self.cache.lock().unwrap();
            cache.reset();
        }

        for step in 0..generation_config.max_new_tokens {
            // Check sequence length limit
            if current_tokens.len() >= self.config.max_sequence_length {
                break;
            }

            // Prepare input tensor
            let input_tensor = self.backend.tokens_to_tensor(&current_tokens)?;

            // Forward pass with parallel processing
            let logits = self.forward_parallel(&input_tensor, step)?;

            // Sample next token
            let next_token = {
                let mut sampling = self.sampling.lock().unwrap();
                sampling.sample(&logits, &current_tokens, step, generation_config)?
            };

            // 3-tier stop check (partial - CPU backend lacks tokenizer for string checks)
            // 1) ID-based stops (fast path - O(1) using HashSet)
            // CRITICAL: Check token IDs BEFORE EOS for performance and correctness
            // For LLaMA-3 <|eot_id|> and other models with token-ID stop sequences
            if generation_config.is_stop_token(next_token) {
                break;
            }

            // 2) EOS token check (explicit or backend default)
            // NOTE: Backend is_eos_token() typically checks tokenizer's EOS token ID
            if self.backend.is_eos_token(next_token) {
                break;
            }

            // 3) String-based stop sequences - NOT IMPLEMENTED in CPU backend
            // CPU backend lacks tokenizer access for string-based checks
            // String-based stops are handled at higher level (InferenceEngine)
            // TODO: Consider refactoring to pass tokenizer to backends for full 3-tier support

            generated_tokens.push(next_token);
            current_tokens.push(next_token);
        }

        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            let elapsed = start_time.elapsed();
            metrics.latency_ms = elapsed.as_millis() as f64;
            metrics.tokens_per_second = generated_tokens.len() as f64 / elapsed.as_secs_f64();
        }

        Ok(generated_tokens)
    }

    /// Forward pass with parallel layer processing
    fn forward_parallel(&self, input: &BitNetTensor, _step: usize) -> Result<BitNetTensor> {
        // This is a simplified synchronous version
        // In a full async implementation, we would use model.read().await

        // For now, create a placeholder result
        // In practice, this would require async model access
        let result = BitNetTensor::zeros(&[1, 32000], candle_core::DType::F32, &candle_core::Device::Cpu)?;

        Ok(result)
    }

    /// Process multiple requests in parallel (batch processing)
    pub fn process_batch_parallel(
        &self,
        requests: &[(Vec<u32>, GenerationConfig)],
    ) -> Result<Vec<Vec<u32>>> {
        if !self.config.batch_processing || requests.len() == 1 {
            // Process sequentially
            return requests
                .iter()
                .map(|(tokens, config)| self.generate_tokens_parallel(tokens, config))
                .collect();
        }

        // Use Rayon for parallel batch processing
        let results: Result<Vec<_>> = self.backend.thread_pool.install(|| {
            requests
                .par_iter()
                .map(|(tokens, config)| {
                    // Each request gets its own temporary engine state
                    self.generate_tokens_parallel(tokens, config)
                })
                .collect()
        });

        results
    }

    /// Get current performance metrics
    pub fn metrics(&self) -> PerformanceMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Reset metrics
    pub fn reset_metrics(&self) {
        let mut metrics = self.metrics.lock().unwrap();
        *metrics = PerformanceMetrics::default();
    }

    /// Get configuration
    pub fn config(&self) -> &CpuInferenceConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: CpuInferenceConfig) -> Result<()> {
        // Resize cache if needed
        if config.max_sequence_length != self.config.max_sequence_length {
            // For now, skip the model config check since it requires async
            let mut cache = self.cache.lock().unwrap();
            cache.resize(config.max_sequence_length)?;
        }

        self.config = config;
        Ok(())
    }

    /// Generate tokens asynchronously
    pub async fn generate_tokens_async(
        &self,
        input_tokens: &[u32],
        generation_config: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        let input_tokens = input_tokens.to_vec();
        let generation_config = generation_config.clone();
        let engine = self.clone_for_async();

        tokio::task::spawn_blocking(move || {
            engine.generate_tokens_parallel(&input_tokens, &generation_config)
        }).await.map_err(|e| bitnet_common::BitNetError::Validation(e.to_string()))?
    }

    /// Create streaming generation asynchronously
    pub async fn generate_stream_async(
        &self,
        input_tokens: Vec<u32>,
        generation_config: GenerationConfig,
        stream_config: StreamingConfig,
    ) -> Result<crate::streaming::TokenGenerationStream> {
        crate::streaming::TokenGenerationStream::create_and_start(
            self.model.clone(),
            self.backend.clone_backend(),
            input_tokens,
            generation_config,
            stream_config,
        ).await
    }

    /// Clone engine for async operations (simplified)
    fn clone_for_async(&self) -> Self {
        // This is a simplified clone - in practice would need proper cloning
        Self {
            model: self.model.clone(),
            backend: CpuBackend::with_config(self.backend.performance_config().clone()).unwrap(),
            cache: Arc::new(Mutex::new(
                KVCache::new(
                    &bitnet_common::BitNetConfig::default(),
                    self.config.max_sequence_length
                ).unwrap()
            )),
            sampling: Arc::new(Mutex::new(
                SamplingStrategy::new(crate::SamplingConfig::default()).unwrap()
            )),
            metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
            config: self.config.clone(),
        }
    }
}

/// CPU-specific optimizations and utilities
pub mod cpu_optimizations {
    use super::*;

    /// Parallel matrix multiplication using Rayon
    pub fn parallel_matmul(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        num_threads: usize,
    ) -> Result<()> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(InferenceError::InvalidInput {
                reason: "Matrix dimension mismatch".to_string()
            }.into());
        }

        // Parallel processing by rows
        let chunk_size = (m + num_threads - 1) / num_threads;

        c.par_chunks_mut(chunk_size * n)
            .enumerate()
            .for_each(|(chunk_idx, c_chunk)| {
                let start_row = chunk_idx * chunk_size;
                let end_row = (start_row + chunk_size).min(m);

                for i in 0..(end_row - start_row) {
                    let global_i = start_row + i;
                    for j in 0..n {
                        let mut sum = 0.0f32;
                        for l in 0..k {
                            sum += a[global_i * k + l] * b[l * n + j];
                        }
                        c_chunk[i * n + j] = sum;
                    }
                }
            });

        Ok(())
    }

    /// Parallel attention computation with numerically-stable softmax.
    ///
    /// Computes scaled dot-product attention per query position:
    /// ```text
    /// scores[i, j] = (Q[i] · K[j]) / sqrt(head_dim)
    /// attn[i, j]   = softmax(scores[i, :])
    ///     = exp(scores[i,j] − max_j scores[i,j])
    ///       / Σ_k exp(scores[i,k] − max_k scores[i,k])
    /// output[i]    = Σ_j attn[i,j] · V[j]
    /// ```
    pub fn parallel_attention(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        output: &mut [f32],
        seq_len: usize,
        head_dim: usize,
        num_heads: usize,
    ) -> Result<()> {
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Parallel processing by attention heads
        output
            .par_chunks_mut(seq_len * head_dim)
            .enumerate()
            .try_for_each(|(head_idx, head_output)| -> Result<()> {
                if head_idx >= num_heads {
                    return Ok(());
                }

                let q_offset = head_idx * seq_len * head_dim;
                let k_offset = head_idx * seq_len * head_dim;
                let v_offset = head_idx * seq_len * head_dim;

                // Pre-compute raw attention scores for each query position.
                let mut scores = vec![0.0f32; seq_len];

                for i in 0..seq_len {
                    // Compute dot products Q[i] · K[j] for all j.
                    for j in 0..seq_len {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += query[q_offset + i * head_dim + d]
                                * key[k_offset + j * head_dim + d];
                        }
                        scores[j] = dot * scale;
                    }

                    // Numerically-stable softmax: subtract max before exp.
                    let max_score = scores[..seq_len].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for j in 0..seq_len {
                        scores[j] = (scores[j] - max_score).exp();
                        sum_exp += scores[j];
                    }

                    // Normalize and accumulate weighted values.
                    let out_base = i * head_dim;
                    for d in 0..head_dim {
                        head_output[out_base + d] = 0.0; // explicit zero (no stale data)
                    }
                    if sum_exp > 0.0 {
                        for j in 0..seq_len {
                            let attn_weight = scores[j] / sum_exp;
                            for d in 0..head_dim {
                                head_output[out_base + d] +=
                                    attn_weight * value[v_offset + j * head_dim + d];
                            }
                        }
                    }
                }

                Ok(())
            })?;

        Ok(())
    }

    /// Memory-efficient tensor operations
    pub fn efficient_tensor_copy(src: &[f32], dst: &mut [f32], chunk_size: usize) {
        src.par_chunks(chunk_size)
            .zip(dst.par_chunks_mut(chunk_size))
            .for_each(|(src_chunk, dst_chunk)| {
                dst_chunk.copy_from_slice(src_chunk);
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_common::BitNetConfig;

    #[test]
    fn test_cpu_backend_creation() {
        let backend = CpuBackend::new();
        assert!(backend.is_ok());

        let backend = backend.unwrap();
        assert_eq!(backend.name(), "CPU-Rayon");
        assert!(backend.is_available());
    }

    #[test]
    fn test_cpu_performance_config() {
        let config = CpuPerformanceConfig {
            num_threads: 4,
            enable_parallel_layers: true,
            enable_parallel_attention: true,
            batch_size_threshold: 2,
            memory_pool_size_mb: 128,
        };

        let backend = CpuBackend::with_config(config.clone());
        assert!(backend.is_ok());

        let backend = backend.unwrap();
        assert_eq!(backend.performance_config().num_threads, 4);
        assert_eq!(backend.performance_config().memory_pool_size_mb, 128);
    }

    #[test]
    fn test_parallel_matmul() {
        use cpu_optimizations::parallel_matmul;

        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let mut c = vec![0.0; 4]; // 2x2

        let result = parallel_matmul(&a, &b, &mut c, 2, 2, 2, 2);
        assert!(result.is_ok());

        // A * I = A
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_cpu_inference_config() {
        let config = CpuInferenceConfig {
            max_sequence_length: 1024,
            enable_kv_cache: true,
            enable_memory_pooling: false,
            parallel_layer_threshold: 4,
            batch_processing: true,
        };

        assert_eq!(config.max_sequence_length, 1024);
        assert!(config.enable_kv_cache);
        assert!(!config.enable_memory_pooling);
    }

    /// Attention weights for a single query position must sum to 1.0 (proper softmax).
    #[test]
    fn test_parallel_attention_softmax_normalization() {
        use cpu_optimizations::parallel_attention;

        let seq_len = 4;
        let head_dim = 2;
        let num_heads = 1;
        // Q, K, V all 1 × seq_len × head_dim (one head)
        let query = vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]; // seq_len=4 rows
        let key = query.clone();
        let value = vec![1.0f32; seq_len * head_dim];
        let mut output = vec![0.0f32; seq_len * head_dim];

        parallel_attention(&query, &key, &value, &mut output, seq_len, head_dim, num_heads)
            .expect("attention should not fail");

        // When V is all-ones, output for each query should also be all-ones
        // (regardless of attention weights, since every value is 1.0).
        for i in 0..seq_len {
            for d in 0..head_dim {
                let got = output[i * head_dim + d];
                assert!(
                    (got - 1.0).abs() < 1e-5,
                    "output[{i},{d}] = {got}, expected ~1.0 (weights sum to 1.0)"
                );
            }
        }
    }

    /// With a single key/value position, attention weight must be exactly 1.0.
    #[test]
    fn test_parallel_attention_single_token() {
        use cpu_optimizations::parallel_attention;

        let seq_len = 1;
        let head_dim = 4;
        let num_heads = 1;
        let query = vec![0.5f32; head_dim];
        let key = vec![0.5f32; head_dim];
        let value = vec![2.0f32; head_dim];
        let mut output = vec![0.0f32; head_dim];

        cpu_optimizations::parallel_attention(
            &query, &key, &value, &mut output, seq_len, head_dim, num_heads,
        )
        .expect("single-token attention should not fail");

        // Single token → softmax gives weight 1.0 → output == value.
        for d in 0..head_dim {
            assert!(
                (output[d] - 2.0).abs() < 1e-5,
                "output[{d}] = {}, expected ~2.0",
                output[d]
            );
        }
    }
}
