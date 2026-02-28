//! Autoregressive Generator Implementation
//!
//! Provides autoregressive text generation with temperature, top-k, and nucleus sampling.

use anyhow::{Context, Result};
use bitnet_common::{BitNetTensor, Device, GenerationConfig as CommonGenConfig};
use candle_core::DType;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use super::deterministic::DeterministicGenerator;
use super::sampling::{SamplingConfig, SamplingStrategy};

/// Token generation optimization constants
const TOKEN_BUFFER_SIZE: usize = 64; // Buffer size for batched token processing
const LATENCY_TARGET_MS: f64 = 100.0; // Target latency per token (ms)
const PREFETCH_WINDOW: usize = 8; // Number of tokens to prefetch
const MIN_BATCH_SIZE: usize = 1; // Minimum batch size for efficient processing
const MAX_BATCH_SIZE: usize = 32; // Maximum batch size to prevent OOM

/// Generation configuration specific to autoregressive generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub do_sample: bool,
    pub seed: Option<u64>,
    pub eos_token_id: usize,
    pub pad_token_id: usize,
    pub min_length: usize,
    pub max_length: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(0.9),
            repetition_penalty: 1.1,
            do_sample: true,
            seed: None,
            eos_token_id: 2, // Common EOS token
            pad_token_id: 0, // Common PAD token
            min_length: 1,
            max_length: 2048,
        }
    }
}

impl From<CommonGenConfig> for GenerationConfig {
    fn from(config: CommonGenConfig) -> Self {
        Self {
            max_new_tokens: config.max_new_tokens,
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            repetition_penalty: config.repetition_penalty,
            do_sample: config.do_sample,
            seed: config.seed,
            eos_token_id: 2,
            pad_token_id: 0,
            min_length: 1,
            max_length: 2048,
        }
    }
}

/// Comprehensive generation statistics for performance monitoring
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    pub tokens_generated: usize,
    pub total_time_ms: f64,
    pub tokens_per_second: f64,
    pub repetitions_detected: usize,
    pub early_stopping: bool,

    // Detailed performance metrics
    pub average_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub cache_hit_rate: f64,
    pub memory_usage_mb: f64,

    // Sampling statistics
    pub temperature_adjustments: usize,
    pub fallback_to_greedy: usize,
    pub batched_generations: usize,

    // Quality metrics
    pub average_entropy: f64,
    pub diversity_score: f64,
}

/// Streaming generation result
#[derive(Debug, Clone)]
pub struct GenerationStep {
    pub token_id: usize,
    pub logits: BitNetTensor,
    pub probability: f32,
    pub is_finished: bool,
    pub cumulative_text: String,
}

/// High-performance autoregressive text generator with latency optimization
pub struct AutoregressiveGenerator {
    config: GenerationConfig,
    device: Device,
    rng: ChaCha8Rng,
    sampling_strategy: SamplingStrategy,
    deterministic_gen: Option<DeterministicGenerator>,

    // Generation state with performance tracking
    generated_tokens: VecDeque<usize>,
    repetition_window: VecDeque<usize>,
    current_length: usize,

    // Performance optimization state
    token_buffer: Vec<usize>,
    generation_times: VecDeque<f64>,
    #[allow(dead_code)]
    batch_processor: Option<Arc<BatchProcessor>>,

    // Memory management
    tensor_cache: Option<BitNetTensor>,
    cache_hit_count: usize,
    cache_miss_count: usize,

    // Adaptive optimization
    adaptive_batch_size: usize,
    latency_window: VecDeque<f64>,
    performance_mode: PerformanceMode,

    // Sampling statistics (reported in GenerationStats)
    temperature_adjustments: usize,
    fallback_to_greedy_count: usize,
    batched_generations_count: usize,
}

/// Performance mode for different optimization strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PerformanceMode {
    Latency,      // Optimize for minimal latency
    Throughput,   // Optimize for maximum throughput
    Balanced,     // Balance between latency and throughput
    Conservative, // Conservative mode for resource-constrained environments
}

/// Batch processor for efficient token generation
#[derive(Debug)]
struct BatchProcessor {
    #[allow(dead_code)]
    batch_size: usize,
    #[allow(dead_code)]
    max_batch_size: usize,
    #[allow(dead_code)]
    pending_requests: Vec<BatchRequest>,
}

impl BatchProcessor {
    fn new(initial_batch_size: usize, max_batch_size: usize) -> Self {
        Self { batch_size: initial_batch_size, max_batch_size, pending_requests: Vec::new() }
    }
}

#[derive(Debug)]
struct BatchRequest {
    #[allow(dead_code)]
    input_tokens: Vec<usize>,
    #[allow(dead_code)]
    generation_config: GenerationConfig,
    #[allow(dead_code)]
    start_time: Instant,
}

impl AutoregressiveGenerator {
    /// Create optimized autoregressive generator with performance tuning
    pub fn new(config: GenerationConfig, device: Device) -> Result<Self> {
        let seed = config.seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
        });

        let rng = ChaCha8Rng::seed_from_u64(seed);

        let sampling_config = SamplingConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            repetition_penalty: config.repetition_penalty,
            do_sample: config.do_sample,
        };
        let sampling_strategy = SamplingStrategy::new(sampling_config);

        // Check for deterministic generation
        let deterministic_gen = if std::env::var("BITNET_DETERMINISTIC").is_ok() {
            // Prefer config.seed over environment variable for deterministic generation
            let det_seed = config.seed.unwrap_or_else(|| {
                std::env::var("BITNET_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(42)
            });
            Some(DeterministicGenerator::new(det_seed)?)
        } else {
            None
        };

        // Determine optimal performance mode based on device and config
        let performance_mode = Self::select_performance_mode(&device, &config);
        let initial_batch_size = Self::calculate_initial_batch_size(&device, performance_mode);

        // Initialize batch processor for throughput optimization
        let batch_processor = if performance_mode == PerformanceMode::Throughput {
            Some(Arc::new(BatchProcessor::new(initial_batch_size, MAX_BATCH_SIZE)))
        } else {
            None
        };

        log::debug!(
            "AutoregressiveGenerator initialized: mode={:?}, batch_size={}, device={:?}",
            performance_mode,
            initial_batch_size,
            device
        );

        Ok(Self {
            config,
            device,
            rng,
            sampling_strategy,
            deterministic_gen,
            generated_tokens: VecDeque::new(),
            repetition_window: VecDeque::new(),
            current_length: 0,

            // Performance optimization
            token_buffer: Vec::with_capacity(TOKEN_BUFFER_SIZE),
            generation_times: VecDeque::with_capacity(100), // Keep last 100 timings
            batch_processor,

            // Memory management
            tensor_cache: None,
            cache_hit_count: 0,
            cache_miss_count: 0,

            // Adaptive optimization
            adaptive_batch_size: initial_batch_size,
            latency_window: VecDeque::with_capacity(50),
            performance_mode,

            // Sampling statistics
            temperature_adjustments: 0,
            fallback_to_greedy_count: 0,
            batched_generations_count: 0,
        })
    }

    /// Select optimal performance mode based on device capabilities
    fn select_performance_mode(device: &Device, config: &GenerationConfig) -> PerformanceMode {
        match device {
            Device::Cuda(_) => {
                // GPU: prioritize throughput for larger models, latency for smaller
                if config.max_new_tokens > 512 {
                    PerformanceMode::Throughput
                } else {
                    PerformanceMode::Latency
                }
            }
            Device::Cpu => {
                // CPU: balance between latency and throughput
                if config.do_sample {
                    PerformanceMode::Balanced
                } else {
                    PerformanceMode::Conservative
                }
            }
            Device::Metal => PerformanceMode::Balanced,
            Device::Hip(_) | Device::Npu => PerformanceMode::Balanced,
            Device::OpenCL(_) => PerformanceMode::Balanced,
        }
    }

    /// Calculate initial batch size based on device and performance mode
    fn calculate_initial_batch_size(device: &Device, mode: PerformanceMode) -> usize {
        match (device, mode) {
            (Device::Cuda(_), PerformanceMode::Throughput) => 16,
            (Device::Cuda(_), PerformanceMode::Latency) => 4,
            (Device::Cuda(_), _) => 8,
            (Device::Cpu, PerformanceMode::Conservative) => 1,
            (Device::Cpu, _) => 2,
            (Device::Metal, _) => 4,
            (Device::Hip(_), _) | (Device::Npu, _) => 2,
            (Device::OpenCL(_), _) => 4,
        }
    }

    /// Generate tokens with optimized latency and adaptive batching
    pub async fn generate<F, Fut>(
        &mut self,
        input_ids: &[usize],
        forward_fn: F,
    ) -> Result<Vec<usize>>
    where
        F: Fn(BitNetTensor) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<BitNetTensor>> + Send,
    {
        let start_time = Instant::now();
        self.reset_state();
        self.prefetch_tensor_cache_if_needed(input_ids)?;

        let mut current_tokens = input_ids.to_vec();
        let mut generated = Vec::new();

        for step in 0..self.config.max_new_tokens {
            // Use consolidated should_stop check (includes max_length, EOS, repetition)
            if self.should_stop(&current_tokens) {
                break;
            }

            let (next_token, step_time) =
                self.generate_next_token(&current_tokens, &forward_fn, step).await?;

            current_tokens.push(next_token);
            generated.push(next_token);
            self.update_generation_state(next_token, step_time);

            if step % 10 == 0 {
                self.adapt_generation_strategy();
            }
        }

        self.finalize_generation(&generated, start_time.elapsed().as_millis() as f64);
        Ok(generated)
    }

    /// Prefetch tensor cache if not already allocated
    fn prefetch_tensor_cache_if_needed(&mut self, tokens: &[usize]) -> Result<()> {
        if self.tensor_cache.is_none() {
            self.prefetch_tensor_cache(tokens)?;
        }
        Ok(())
    }

    /// Generate next token with adaptive optimization
    async fn generate_next_token<F, Fut>(
        &mut self,
        current_tokens: &[usize],
        forward_fn: &F,
        step: usize,
    ) -> Result<(usize, f64)>
    where
        F: Fn(BitNetTensor) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<BitNetTensor>> + Send,
    {
        let step_start = Instant::now();

        let next_token = if self.should_use_batching(step) {
            self.generate_token_batched(current_tokens, forward_fn, step).await?
        } else {
            self.generate_token_single(current_tokens, forward_fn, step).await?
        };

        let step_time = step_start.elapsed().as_millis() as f64;
        Ok((next_token, step_time))
    }

    /// Finalize generation and update statistics
    fn finalize_generation(&mut self, generated: &[usize], total_time: f64) {
        self.update_generation_statistics(generated.len(), total_time);
    }

    /// Generate single token with optimized caching
    async fn generate_token_single<F, Fut>(
        &mut self,
        current_tokens: &[usize],
        forward_fn: &F,
        step: usize,
    ) -> Result<usize>
    where
        F: Fn(BitNetTensor) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<BitNetTensor>> + Send,
    {
        // Check tensor cache first
        let input_tensor = if let Some(cached) = self.try_get_cached_tensor(current_tokens)? {
            self.cache_hit_count += 1;
            cached
        } else {
            self.cache_miss_count += 1;
            let tensor = self.tokens_to_tensor(current_tokens)?;
            self.update_tensor_cache(&tensor)?;
            tensor
        };

        // Forward pass with error recovery
        let logits = forward_fn(input_tensor).await.with_context(|| {
            format!("Forward pass failed at step {} with {} tokens", step, current_tokens.len())
        })?;

        // Sample next token with performance tracking
        let next_token = self.sample_next_token(&logits, step).await?;

        Ok(next_token)
    }

    /// Generate token using batched processing for throughput
    async fn generate_token_batched<F, Fut>(
        &mut self,
        current_tokens: &[usize],
        forward_fn: &F,
        step: usize,
    ) -> Result<usize>
    where
        F: Fn(BitNetTensor) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<BitNetTensor>> + Send,
    {
        // For now, fallback to single token generation
        // In a full implementation, this would batch multiple sequences
        self.generate_token_single(current_tokens, forward_fn, step).await
    }

    /// Check if batching should be used for current step
    fn should_use_batching(&self, step: usize) -> bool {
        // Use batching for throughput mode after warmup period
        self.performance_mode == PerformanceMode::Throughput
            && step > 5
            && self.adaptive_batch_size > 1
    }

    /// Generate tokens with streaming support
    pub async fn generate_stream<F, Fut>(
        &mut self,
        input_ids: &[usize],
        forward_fn: F,
    ) -> Result<impl futures_util::Stream<Item = Result<GenerationStep>>>
    where
        F: Fn(BitNetTensor) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<BitNetTensor>> + Send,
    {
        self.reset_state();

        let mut current_tokens = input_ids.to_vec();
        let _initial_length = current_tokens.len();

        Ok(async_stream::stream! {
            for step in 0..self.config.max_new_tokens {
                // Check length constraints
                if current_tokens.len() >= self.config.max_length {
                    yield Ok(GenerationStep {
                        token_id: self.config.eos_token_id,
                        logits: BitNetTensor::zeros(&[1], DType::F32, &self.device)?,
                        probability: 1.0,
                        is_finished: true,
                        cumulative_text: String::new(), // Would be populated by tokenizer
                    });
                    break;
                }

                // Create input tensor
                let input_tensor = match self.tokens_to_tensor(&current_tokens) {
                    Ok(tensor) => tensor,
                    Err(e) => {
                        yield Err(e);
                        break;
                    }
                };

                // Forward pass
                let logits = match forward_fn(input_tensor).await {
                    Ok(logits) => logits,
                    Err(e) => {
                        yield Err(e.context("Forward pass failed during streaming generation"));
                        break;
                    }
                };

                // Sample next token
                let (next_token, probability) = match self.sample_next_token_with_prob(&logits, step).await {
                    Ok(result) => result,
                    Err(e) => {
                        yield Err(e);
                        break;
                    }
                };

                let is_eos = next_token == self.config.eos_token_id;
                let is_finished = is_eos && current_tokens.len() >= self.config.min_length;

                // Yield generation step
                yield Ok(GenerationStep {
                    token_id: next_token,
                    logits: logits.clone(),
                    probability,
                    is_finished,
                    cumulative_text: String::new(), // Would be populated by tokenizer
                });

                if is_finished {
                    break;
                }

                // Add token to sequence
                current_tokens.push(next_token);
                self.update_generation_state(next_token, 0.0);

                // Check for repetition penalty trigger
                if self.should_apply_repetition_penalty(next_token) {
                    self.sampling_strategy.increase_repetition_penalty();
                }
            }
        })
    }

    /// Sample next token with latency optimization and fallback strategies
    async fn sample_next_token(&mut self, logits: &BitNetTensor, step: usize) -> Result<usize> {
        let sampling_start = Instant::now();
        let (token, _prob) = self.sample_next_token_with_prob(logits, step).await?;

        self.track_sampling_performance(sampling_start.elapsed().as_millis() as f64);
        Ok(token)
    }

    /// Track sampling performance and consider fallback if needed
    fn track_sampling_performance(&mut self, sampling_time: f64) {
        if sampling_time > LATENCY_TARGET_MS {
            self.consider_sampling_fallback();
        }
    }

    /// Sample next token with probability and performance monitoring
    async fn sample_next_token_with_prob(
        &mut self,
        logits: &BitNetTensor,
        step: usize,
    ) -> Result<(usize, f32)> {
        // Debug probe: log top-5 logits for first token only (Issue #XXX parity debugging)
        if std::env::var("BITNET_DEBUG_LOGITS").as_deref() == Ok("1") {
            eprintln!("DEBUG: logits probe enabled, step={}", step);
            if step == 0 {
                self.log_top5_logits(logits)?;
            }
        }

        // Use deterministic generation if enabled
        if let Some(ref mut det_gen) = self.deterministic_gen {
            return det_gen.sample_deterministic(logits, step).await;
        }

        // Try fast sampling for latency-critical scenarios
        if self.should_use_fast_sampling()
            && let Ok(result) = self.try_fast_sampling(logits).await
        {
            return Ok(result);
        }

        // Apply full sampling strategy
        self.sampling_strategy.sample(logits, &mut self.rng).await
    }

    /// Check if fast sampling should be used
    fn should_use_fast_sampling(&self) -> bool {
        self.performance_mode == PerformanceMode::Latency && self.config.do_sample
    }

    /// Fast sampling for latency-critical scenarios
    async fn try_fast_sampling(&mut self, logits: &BitNetTensor) -> Result<(usize, f32)> {
        // Simplified sampling with reduced computation
        if self.config.temperature < 0.1 || !self.config.do_sample {
            // Use greedy sampling for very low temperature
            self.fallback_to_greedy_count += 1;
            return self.sampling_strategy.sample(logits, &mut self.rng).await;
        }

        // Use top-k with reduced k for faster sampling
        let fast_config = SamplingConfig {
            temperature: self.config.temperature,
            top_k: Some(self.config.top_k.unwrap_or(50).min(20)), // Reduce top-k for speed
            top_p: None,                                          // Skip nucleus sampling for speed
            repetition_penalty: self.config.repetition_penalty,
            do_sample: true,
        };

        let mut fast_strategy = SamplingStrategy::new(fast_config);
        fast_strategy.sample(logits, &mut self.rng).await
    }

    /// Consider fallback to simpler sampling for performance
    fn consider_sampling_fallback(&mut self) {
        match self.performance_mode {
            PerformanceMode::Latency => {
                // Reduce sampling complexity
                if self.config.top_k.unwrap_or(50) > 20 {
                    log::debug!("Reducing top-k for latency optimization");
                    self.temperature_adjustments += 1;
                    // Would update config in practice
                }
            }
            PerformanceMode::Conservative => {
                // Switch to greedy sampling if needed
                log::debug!("Considering greedy sampling for conservative mode");
                self.fallback_to_greedy_count += 1;
            }
            _ => {}
        }
    }

    /// Convert token sequence to tensor with caching optimization
    fn tokens_to_tensor(&self, tokens: &[usize]) -> Result<BitNetTensor> {
        let token_data: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
        BitNetTensor::from_slice(&token_data, &[1, tokens.len()], &self.device)
            .context("Failed to create tensor from tokens")
    }

    /// Try to get cached tensor for recent token sequences
    fn try_get_cached_tensor(&self, tokens: &[usize]) -> Result<Option<BitNetTensor>> {
        // Simple cache implementation - in practice would use more sophisticated caching
        if tokens.len() <= 1 {
            return Ok(None);
        }

        // For now, return None to indicate no cache hit
        // A full implementation would maintain a tensor cache
        Ok(None)
    }

    /// Update tensor cache with new tensor
    fn update_tensor_cache(&mut self, tensor: &BitNetTensor) -> Result<()> {
        // Simple cache update - store last tensor
        self.tensor_cache = Some(tensor.clone());
        Ok(())
    }

    /// Prefetch tensor cache for better memory access patterns
    fn prefetch_tensor_cache(&mut self, tokens: &[usize]) -> Result<()> {
        // Pre-allocate tensor for expected size
        let expected_size = tokens.len() + self.config.max_new_tokens.min(PREFETCH_WINDOW);
        let dummy_data = vec![0.0f32; expected_size];
        self.tensor_cache =
            Some(BitNetTensor::from_slice(&dummy_data, &[1, expected_size], &self.device)?);
        Ok(())
    }

    /// Reset generation state with performance tracking preservation
    fn reset_state(&mut self) {
        self.generated_tokens.clear();
        self.repetition_window.clear();
        self.current_length = 0;

        // Clear token buffer but preserve performance metrics
        self.token_buffer.clear();

        // Keep some performance history for adaptive optimization
        if self.generation_times.len() > 50 {
            // Keep last 25 timings for trend analysis
            let mut kept_times = VecDeque::new();
            for _ in 0..25 {
                if let Some(time) = self.generation_times.pop_back() {
                    kept_times.push_front(time);
                }
            }
            self.generation_times = kept_times;
        }

        log::debug!(
            "Reset generation state, cache stats: hits={}, misses={}",
            self.cache_hit_count,
            self.cache_miss_count
        );
    }

    /// Update generation state with performance tracking
    fn update_generation_state(&mut self, token: usize, generation_time_ms: f64) {
        // Update token tracking
        self.generated_tokens.push_back(token);
        self.repetition_window.push_back(token);
        self.token_buffer.push(token);

        // Update performance tracking
        self.generation_times.push_back(generation_time_ms);
        self.latency_window.push_back(generation_time_ms);

        // Keep windows to reasonable size
        while self.repetition_window.len() > 50 {
            self.repetition_window.pop_front();
        }

        while self.generation_times.len() > 100 {
            self.generation_times.pop_front();
        }

        while self.latency_window.len() > 50 {
            self.latency_window.pop_front();
        }

        // Flush token buffer if full
        if self.token_buffer.len() >= TOKEN_BUFFER_SIZE {
            self.flush_token_buffer();
        }

        // Track repetition and apply penalties
        if self.should_apply_repetition_penalty(token) {
            self.sampling_strategy.increase_repetition_penalty();
        }

        self.current_length += 1;
    }

    /// Flush token buffer for batch processing
    fn flush_token_buffer(&mut self) {
        // In a full implementation, this would process buffered tokens
        log::debug!("Flushing token buffer with {} tokens", self.token_buffer.len());
        self.batched_generations_count += 1;
        self.token_buffer.clear();
    }

    /// Check if repetition penalty should be applied
    fn should_apply_repetition_penalty(&self, token: usize) -> bool {
        // Count occurrences of token in recent window
        let count = self.repetition_window.iter().filter(|&&t| t == token).count();
        count > 2 // Apply penalty if token appears more than twice
    }

    /// Get comprehensive generation statistics with performance metrics
    pub fn get_stats(&self) -> GenerationStats {
        let total_time = self.generation_times.iter().sum::<f64>();
        let avg_latency = if !self.generation_times.is_empty() {
            total_time / self.generation_times.len() as f64
        } else {
            0.0
        };

        let min_latency = self.generation_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_latency: f32 = self.generation_times.iter().fold(0.0f32, |a, &b| a.max(b as f32));

        let tokens_per_second = if total_time > 0.0 {
            (self.generated_tokens.len() as f64 * 1000.0) / total_time
        } else {
            0.0
        };

        let cache_hit_rate = if self.cache_hit_count + self.cache_miss_count > 0 {
            self.cache_hit_count as f64 / (self.cache_hit_count + self.cache_miss_count) as f64
        } else {
            0.0
        };

        // Calculate diversity metrics
        let unique_tokens =
            self.generated_tokens.iter().collect::<std::collections::HashSet<_>>().len();
        let diversity_score = if !self.generated_tokens.is_empty() {
            unique_tokens as f64 / self.generated_tokens.len() as f64
        } else {
            0.0
        };

        GenerationStats {
            tokens_generated: self.generated_tokens.len(),
            total_time_ms: total_time,
            tokens_per_second,
            repetitions_detected: self.count_repetitions(),
            early_stopping: false,

            // Detailed performance metrics
            average_latency_ms: avg_latency,
            min_latency_ms: if min_latency == f64::INFINITY { 0.0 } else { min_latency },
            max_latency_ms: max_latency as f64,
            cache_hit_rate,
            memory_usage_mb: self.estimate_memory_usage(),

            // Sampling statistics
            temperature_adjustments: self.temperature_adjustments,
            fallback_to_greedy: self.fallback_to_greedy_count,
            batched_generations: self.batched_generations_count,

            // Quality metrics
            average_entropy: 0.0, // Would calculate from logits in full implementation
            diversity_score,
        }
    }

    /// Estimate current memory usage in MB
    fn estimate_memory_usage(&self) -> f64 {
        let tensor_memory = self
            .tensor_cache
            .as_ref()
            .map(|t| {
                use bitnet_common::Tensor;
                t.shape().iter().product::<usize>() * std::mem::size_of::<f32>()
            })
            .unwrap_or(0);

        let buffer_memory = self.token_buffer.capacity() * std::mem::size_of::<usize>();
        let state_memory = (self.generated_tokens.len() + self.repetition_window.len())
            * std::mem::size_of::<usize>();

        (tensor_memory + buffer_memory + state_memory) as f64 / (1024.0 * 1024.0)
    }

    /// Adapt generation strategy based on recent performance
    fn adapt_generation_strategy(&mut self) {
        if self.latency_window.len() < 10 {
            return; // Need more data points
        }

        let avg_latency =
            self.latency_window.iter().sum::<f64>() / self.latency_window.len() as f64;

        match self.performance_mode {
            PerformanceMode::Latency => {
                if avg_latency > LATENCY_TARGET_MS {
                    // Reduce batch size or switch to simpler sampling
                    self.adaptive_batch_size = (self.adaptive_batch_size - 1).max(MIN_BATCH_SIZE);
                    log::debug!(
                        "Reduced batch size to {} for latency optimization",
                        self.adaptive_batch_size
                    );
                }
            }
            PerformanceMode::Throughput => {
                if avg_latency < LATENCY_TARGET_MS * 0.5 {
                    // Can increase batch size for better throughput
                    self.adaptive_batch_size = (self.adaptive_batch_size + 1).min(MAX_BATCH_SIZE);
                    log::debug!(
                        "Increased batch size to {} for throughput optimization",
                        self.adaptive_batch_size
                    );
                }
            }
            _ => {} // No adaptation for other modes
        }
    }

    /// Update final generation statistics
    fn update_generation_statistics(&mut self, tokens_generated: usize, total_time_ms: f64) {
        log::info!(
            "Generation completed: {} tokens in {:.2}ms ({:.2} tok/sec), cache hit rate: {:.2}%",
            tokens_generated,
            total_time_ms,
            (tokens_generated as f64 * 1000.0) / total_time_ms,
            if self.cache_hit_count + self.cache_miss_count > 0 {
                (self.cache_hit_count as f64
                    / (self.cache_hit_count + self.cache_miss_count) as f64)
                    * 100.0
            } else {
                0.0
            }
        );
    }

    /// Count repetitions in generated sequence
    fn count_repetitions(&self) -> usize {
        let mut repetitions = 0;
        let tokens: Vec<usize> = self.generated_tokens.iter().cloned().collect();

        // Simple n-gram repetition detection
        for window_size in 2..=4 {
            for i in 0..tokens.len().saturating_sub(window_size * 2) {
                let window1 = &tokens[i..i + window_size];
                let window2 = &tokens[i + window_size..i + window_size * 2];
                if window1 == window2 {
                    repetitions += 1;
                }
            }
        }

        repetitions
    }

    /// Set seed for reproducible generation with state preservation
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = ChaCha8Rng::seed_from_u64(seed);

        // Update deterministic generator if present
        if let Some(ref mut det_gen) = self.deterministic_gen {
            det_gen.set_seed(seed);
        }

        log::debug!("Updated generation seed to {}", seed);
    }

    /// Enable performance mode switching during generation
    pub fn set_performance_mode(&mut self, mode: PerformanceMode) {
        if mode != self.performance_mode {
            log::info!("Switching performance mode from {:?} to {:?}", self.performance_mode, mode);
            self.performance_mode = mode;

            // Adjust batch size for new mode
            self.adaptive_batch_size = Self::calculate_initial_batch_size(&self.device, mode);

            // Clear performance history for clean adaptation
            self.latency_window.clear();
        }
    }

    /// Update generation config
    pub fn update_config(&mut self, config: GenerationConfig) {
        self.config = config.clone();

        // Update sampling strategy
        let sampling_config = SamplingConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            repetition_penalty: config.repetition_penalty,
            do_sample: config.do_sample,
        };
        self.sampling_strategy = SamplingStrategy::new(sampling_config);
    }

    /// Check if generation should stop early
    pub fn should_stop(&self, current_tokens: &[usize]) -> bool {
        // Check max length
        if current_tokens.len() >= self.config.max_length {
            return true;
        }

        // Check for EOS token (if min length is met)
        if let Some(&last_token) = current_tokens.last()
            && last_token == self.config.eos_token_id
            && current_tokens.len() >= self.config.min_length
        {
            return true;
        }

        // Check for excessive repetition
        if self.count_repetitions() > 10 {
            return true;
        }

        false
    }

    /// Debug helper: log top-5 logits with indices and values
    ///
    /// This is a surgical debug probe for diagnosing math issues in greedy generation.
    /// Enable with `BITNET_DEBUG_LOGITS=1`.
    fn log_top5_logits(&self, logits: &BitNetTensor) -> Result<()> {
        // Convert logits to Vec<f32> for sorting
        let logits_vec = logits.to_vec()?;

        // Create (index, value) pairs and sort by value descending
        let mut indexed_logits: Vec<(usize, f32)> =
            logits_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top 5
        let top5: Vec<_> = indexed_logits.iter().take(5).collect();

        let top5_idx: Vec<usize> = top5.iter().map(|(i, _)| *i).collect();
        let top5_val: Vec<f32> = top5.iter().map(|(_, v)| *v).collect();

        eprintln!("top5_idx={:?}", top5_idx);
        eprintln!("top5_val={:?}", top5_val);

        // TODO: decode tokens if tokenizer is available (requires access to tokenizer)
        // For now, just log indices and values

        Ok(())
    }
}
