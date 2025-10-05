//! # Streaming Generation Support
//!
//! Provides streaming token generation with async/await support, backpressure handling,
//! and cancellation support for real-time applications.

use anyhow::{Context, Result};
use bitnet_common::ConcreteTensor;
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;
use futures_util::Stream;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context as TaskContext, Poll};
use tokio::sync::{RwLock, mpsc};
use tracing::{debug, error, instrument, trace, warn};

use crate::{
    backends::Backend,
    cache::KVCache,
    config::GenerationConfig,
    sampling::{SamplingConfig, SamplingStrategy},
};

/// Configuration for streaming generation
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Size of the internal buffer for tokens
    pub buffer_size: usize,
    /// Interval between token flushes in milliseconds
    pub flush_interval_ms: u64,
    /// Maximum number of retries on transient errors
    pub max_retries: usize,
    /// Timeout for individual token generation (milliseconds)
    pub token_timeout_ms: u64,
    /// Enable cancellation support
    pub cancellable: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10,
            flush_interval_ms: 50,
            max_retries: 3,
            token_timeout_ms: 5000, // 5 seconds
            cancellable: true,
        }
    }
}

impl StreamingConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.buffer_size == 0 {
            return Err(anyhow::anyhow!("Buffer size must be greater than 0"));
        }
        if self.flush_interval_ms == 0 {
            return Err(anyhow::anyhow!("Flush interval must be greater than 0"));
        }
        if self.token_timeout_ms == 0 {
            return Err(anyhow::anyhow!("Token timeout must be greater than 0"));
        }
        Ok(())
    }

    /// Create a configuration optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            buffer_size: 1,
            flush_interval_ms: 10,
            max_retries: 1,
            token_timeout_ms: 1000,
            cancellable: true,
        }
    }

    /// Create a configuration optimized for high throughput
    pub fn high_throughput() -> Self {
        Self {
            buffer_size: 50,
            flush_interval_ms: 200,
            max_retries: 5,
            token_timeout_ms: 10000,
            cancellable: false,
        }
    }
}

/// A stream of generated tokens
pub struct GenerationStream {
    receiver: mpsc::Receiver<Result<StreamResponse>>,
    _handle: tokio::task::JoinHandle<()>,
    cancellation_token: Arc<AtomicBool>,
    generation_stats: Arc<GenerationStats>,
}

/// Statistics for generation streaming
#[derive(Debug, Default)]
pub struct GenerationStats {
    pub tokens_generated: AtomicUsize,
    pub errors_encountered: AtomicUsize,
    pub retries_attempted: AtomicUsize,
    pub cancelled: AtomicBool,
}

impl GenerationStats {
    pub fn tokens_generated(&self) -> usize {
        self.tokens_generated.load(Ordering::Relaxed)
    }

    pub fn errors_encountered(&self) -> usize {
        self.errors_encountered.load(Ordering::Relaxed)
    }

    pub fn retries_attempted(&self) -> usize {
        self.retries_attempted.load(Ordering::Relaxed)
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

/// Parameters for internal streaming generation
struct StreamingParams {
    _model: Arc<dyn Model>,
    tokenizer: Arc<dyn Tokenizer>,
    backend: Box<dyn Backend>,
    cache: Arc<RwLock<KVCache>>,
    prompt: String,
    config: GenerationConfig,
    streaming_config: StreamingConfig,
    sender: mpsc::Sender<Result<StreamResponse>>,
    cancellation_token: Arc<AtomicBool>,
    stats: Arc<GenerationStats>,
}

impl GenerationStream {
    /// Create a new generation stream
    #[instrument(skip(model, tokenizer, backend, cache))]
    pub fn new(
        model: Arc<dyn Model>,
        tokenizer: Arc<dyn Tokenizer>,
        backend: Box<dyn Backend>,
        cache: Arc<RwLock<KVCache>>,
        prompt: String,
        generation_config: GenerationConfig,
        streaming_config: StreamingConfig,
    ) -> Result<Self> {
        // Validate streaming configuration
        streaming_config.validate().context("Invalid streaming configuration")?;

        let (sender, receiver) =
            mpsc::channel::<Result<StreamResponse>>(streaming_config.buffer_size);
        let cancellation_token = Arc::new(AtomicBool::new(false));
        let generation_stats = Arc::new(GenerationStats::default());

        let cancel_token_clone = cancellation_token.clone();
        let stats_clone = generation_stats.clone();

        let handle = tokio::spawn(async move {
            if let Err(e) = Self::generate_stream_internal(StreamingParams {
                _model: model,
                tokenizer,
                backend,
                cache,
                prompt,
                config: generation_config,
                streaming_config,
                sender: sender.clone(),
                cancellation_token: cancel_token_clone,
                stats: stats_clone,
            })
            .await
            {
                error!("Stream generation failed with error: {}", e);

                // Create detailed error context
                let detailed_error = anyhow::anyhow!(
                    "Streaming generation failed: {}. This may be due to model errors, \
                     tokenization issues, or resource constraints. Check logs for details.",
                    e
                )
                .context("GenerationStream internal error");

                // Send detailed error to consumer
                if let Err(send_err) = sender.send(Err(detailed_error)).await {
                    error!("Failed to send error to stream consumer: {}", send_err);
                }
            }
        });

        Ok(Self { receiver, _handle: handle, cancellation_token, generation_stats })
    }

    /// Internal streaming generation implementation
    async fn generate_stream_internal(params: StreamingParams) -> Result<()> {
        let StreamingParams {
            _model,
            tokenizer,
            backend,
            cache,
            prompt,
            config,
            streaming_config,
            sender,
            cancellation_token,
            stats,
        } = params;
        debug!("Starting streaming generation for prompt");

        // Tokenize input with comprehensive error handling
        let input_tokens = tokenizer.encode(&prompt, true, true).with_context(|| {
            format!(
                "Failed to tokenize input prompt of length {}. \
                     The prompt may contain unsupported characters or exceed \
                     tokenizer limits. Prompt preview: '{}...'",
                prompt.len(),
                &prompt[..50.min(prompt.len())]
            )
        })?;

        if input_tokens.is_empty() {
            warn!(
                "Tokenization resulted in empty token sequence for prompt: '{}...'. \
                 This may cause generation issues.",
                &prompt[..20.min(prompt.len())]
            );
        }

        debug!(
            "Successfully tokenized prompt of length {} into {} tokens",
            prompt.len(),
            input_tokens.len()
        );

        let mut current_tokens = input_tokens.clone();
        let mut generated_count = 0;

        let sampling_config = SamplingConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            repetition_penalty: config.repetition_penalty,
            seed: config.seed,
        };

        let mut sampling_strategy = SamplingStrategy::new(sampling_config);
        let mut token_buffer = Vec::new();
        let mut token_ids_buffer = Vec::new();
        let mut last_flush = std::time::Instant::now();

        for _ in 0..config.max_new_tokens {
            // Check for cancellation
            if streaming_config.cancellable && cancellation_token.load(Ordering::Relaxed) {
                debug!("Generation cancelled by user");
                stats.cancelled.store(true, Ordering::Relaxed);
                break;
            }

            // Check if receiver is closed (client disconnected)
            if sender.is_closed() {
                debug!("Client disconnected, stopping generation");
                break;
            }

            // Forward pass through model with retries
            let mut retry_count = 0;
            let logits = loop {
                match tokio::time::timeout(
                    std::time::Duration::from_millis(streaming_config.token_timeout_ms),
                    Self::forward_pass(&*backend, &current_tokens, &cache, &tokenizer),
                )
                .await
                {
                    Ok(Ok(logits)) => break logits,
                    Ok(Err(e)) => {
                        stats.errors_encountered.fetch_add(1, Ordering::Relaxed);

                        if retry_count >= streaming_config.max_retries {
                            error!(
                                "Forward pass failed after {} retries for token position {}. \
                                 Final error: {}. Consider reducing generation complexity or \
                                 increasing timeout/retry limits.",
                                retry_count,
                                generated_count + 1,
                                e
                            );

                            let context_error = e.context(format!(
                                "Forward pass exhausted {} retries at token position {}",
                                streaming_config.max_retries,
                                generated_count + 1
                            ));

                            return Err(context_error);
                        }

                        retry_count += 1;
                        stats.retries_attempted.fetch_add(1, Ordering::Relaxed);
                        warn!(
                            "Forward pass failed at token position {}, retrying ({}/{}): {}. \
                             Applying exponential backoff.",
                            generated_count + 1,
                            retry_count,
                            streaming_config.max_retries,
                            e
                        );

                        // Exponential backoff
                        let backoff_ms = (50 * retry_count as u64).min(1000);
                        tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                    }
                    Err(_) => {
                        stats.errors_encountered.fetch_add(1, Ordering::Relaxed);
                        error!(
                            "Forward pass timed out after {}ms at token position {}. \
                             This may indicate model complexity exceeds available resources.",
                            streaming_config.token_timeout_ms,
                            generated_count + 1
                        );

                        return Err(anyhow::anyhow!(
                            "Forward pass timed out after {}ms at token position {}. \
                             Consider increasing token_timeout_ms or reducing model complexity.",
                            streaming_config.token_timeout_ms,
                            generated_count + 1
                        ));
                    }
                }
            };

            // Sample next token with error context
            let next_token =
                sampling_strategy.sample(&logits, &current_tokens).with_context(|| {
                    format!(
                        "Sampling failed at token position {} with {} current tokens. \
                         This may indicate invalid logits or sampling configuration issues.",
                        generated_count + 1,
                        current_tokens.len()
                    )
                })?;

            // Check for stop conditions
            if Self::should_stop(next_token, &current_tokens, &config, &tokenizer) {
                break;
            }

            // Decode token to text with detailed error context
            let token_text = tokenizer.decode(&[next_token]).with_context(|| {
                format!(
                    "Failed to decode token {} at position {}. \
                         The token ID may be out of vocabulary range or the tokenizer \
                         may be corrupted. Vocab size: {}",
                    next_token,
                    generated_count + 1,
                    tokenizer.vocab_size()
                )
            })?;

            if token_text.is_empty() {
                debug!(
                    "Token {} decoded to empty string at position {}. \
                     This may be expected for special tokens.",
                    next_token,
                    generated_count + 1
                );
            }

            token_buffer.push(token_text);
            token_ids_buffer.push(next_token);
            current_tokens.push(next_token);
            generated_count += 1;
            stats.tokens_generated.fetch_add(1, Ordering::Relaxed);

            trace!(
                "Generated token {}: {} (token_id: {})",
                generated_count,
                token_buffer.last().unwrap_or(&"<empty>".to_string()),
                next_token
            );

            // Flush buffer based on size or time
            let should_flush = token_buffer.len() >= streaming_config.buffer_size
                || last_flush.elapsed().as_millis() >= streaming_config.flush_interval_ms as u128;

            if should_flush && !token_buffer.is_empty() {
                let buffered_text = token_buffer.join("");
                let buffered_token_ids = token_ids_buffer.clone();
                if sender
                    .send(Ok(StreamResponse { text: buffered_text, token_ids: buffered_token_ids }))
                    .await
                    .is_err()
                {
                    debug!("Client disconnected during send");
                    break;
                }
                token_buffer.clear();
                token_ids_buffer.clear();
                last_flush = std::time::Instant::now();
            }

            // Limit context length
            if current_tokens.len() > 2048 {
                let keep_length = 1024;
                current_tokens = current_tokens[current_tokens.len() - keep_length..].to_vec();
            }
        }

        // Flush remaining tokens
        if !token_buffer.is_empty() {
            let remaining_text = token_buffer.join("");
            let remaining_token_ids = token_ids_buffer.clone();
            let _ = sender
                .send(Ok(StreamResponse { text: remaining_text, token_ids: remaining_token_ids }))
                .await;
        }

        debug!("Streaming generation completed: {} tokens", generated_count);
        Ok(())
    }

    /// Perform forward pass through the model
    async fn forward_pass(
        backend: &dyn Backend,
        tokens: &[u32],
        cache: &Arc<RwLock<KVCache>>,
        tokenizer: &Arc<dyn Tokenizer>,
    ) -> Result<Vec<f32>> {
        // Convert tokens to tensor
        let input_tensor = Self::tokens_to_tensor(tokens)?;

        // Get cache for this sequence
        let mut cache_guard = cache.write().await;

        // Forward pass through backend
        let output_tensor = backend.forward(&input_tensor, &mut cache_guard).await?;

        // Extract logits from output tensor
        Self::tensor_to_logits(&output_tensor, tokenizer.vocab_size())
    }

    /// Convert tokens to input tensor (mock implementation)
    fn tokens_to_tensor(tokens: &[u32]) -> Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, tokens.len()]))
    }

    /// Extract logits from output tensor (mock implementation)
    fn tensor_to_logits(_tensor: &ConcreteTensor, vocab_size: usize) -> Result<Vec<f32>> {
        // In a real implementation, this would extract logits from the tensor
        Ok(vec![0.1; vocab_size])
    }

    /// Check if generation should stop
    fn should_stop(
        token: u32,
        current_tokens: &[u32],
        config: &GenerationConfig,
        tokenizer: &Arc<dyn Tokenizer>,
    ) -> bool {
        // Check for EOS token from config, fallback to tokenizer default
        let eos_token = config.eos_token_id.or_else(|| tokenizer.eos_token_id());
        if let Some(eos) = eos_token
            && token == eos
        {
            return true;
        }

        // Check for stop sequences
        if !config.stop_sequences.is_empty()
            && let Ok(current_text) = tokenizer.decode(current_tokens)
        {
            for stop_seq in &config.stop_sequences {
                if current_text.ends_with(stop_seq) {
                    return true;
                }
            }
        }

        false
    }

    /// Cancel the stream generation
    pub fn cancel(&self) {
        if self.cancellation_token.load(Ordering::Relaxed) {
            debug!("Stream already cancelled");
            return;
        }

        self.cancellation_token.store(true, Ordering::Relaxed);
        debug!("Stream cancellation requested");
    }

    /// Check if the stream is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancellation_token.load(Ordering::Relaxed)
    }

    /// Get generation statistics
    pub fn stats(&self) -> &GenerationStats {
        &self.generation_stats
    }

    /// Check if the stream is still active (not finished and not cancelled)
    pub fn is_active(&self) -> bool {
        !self.receiver.is_closed() && !self.is_cancelled()
    }
}

/// Enhanced streaming response with token text and token ID
#[derive(Debug)]
pub struct StreamResponse {
    pub text: String,
    pub token_ids: Vec<u32>,
}

impl Stream for GenerationStream {
    type Item = Result<StreamResponse>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

// MockTensor is now defined in bitnet_common

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use std::sync::Arc;

    struct MockModel {
        config: bitnet_common::BitNetConfig,
    }

    impl MockModel {
        fn new() -> Self {
            Self { config: bitnet_common::BitNetConfig::default() }
        }
    }

    impl Model for MockModel {
        fn config(&self) -> &bitnet_common::BitNetConfig {
            &self.config
        }

        fn forward(
            &self,
            _input: &ConcreteTensor,
            _cache: &mut dyn std::any::Any,
        ) -> bitnet_common::Result<ConcreteTensor> {
            Ok(ConcreteTensor::mock(vec![1, 50257]))
        }

        fn embed(&self, _tokens: &[u32]) -> bitnet_common::Result<ConcreteTensor> {
            Ok(ConcreteTensor::mock(vec![1, 10, 768]))
        }

        fn logits(&self, _hidden: &ConcreteTensor) -> bitnet_common::Result<ConcreteTensor> {
            Ok(ConcreteTensor::mock(vec![1, 50257]))
        }
    }

    struct MockTokenizer;

    impl Tokenizer for MockTokenizer {
        fn encode(
            &self,
            _text: &str,
            _add_bos: bool,
            _add_special: bool,
        ) -> bitnet_common::Result<Vec<u32>> {
            Ok(vec![1, 2, 3])
        }

        fn decode(&self, tokens: &[u32]) -> bitnet_common::Result<String> {
            Ok(format!("token_{}", tokens.len()))
        }

        fn vocab_size(&self) -> usize {
            50257
        }

        fn token_to_piece(&self, _token: u32) -> Option<String> {
            Some("<token>".to_string())
        }
    }

    struct MockBackend;

    #[async_trait::async_trait]
    impl Backend for MockBackend {
        fn backend_type(&self) -> String {
            "mock".to_string()
        }

        fn clone_backend(&self) -> Box<dyn Backend> {
            Box::new(MockBackend)
        }

        async fn forward(
            &self,
            _input: &ConcreteTensor,
            _cache: &mut KVCache,
        ) -> Result<ConcreteTensor> {
            Ok(ConcreteTensor::mock(vec![1, 50257]))
        }
    }

    #[tokio::test]
    async fn test_streaming_generation() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer);
        let backend = Box::new(MockBackend);
        let cache = Arc::new(RwLock::new(KVCache::new(Default::default()).unwrap()));

        let config = GenerationConfig { max_new_tokens: 5, ..Default::default() };

        let streaming_config = StreamingConfig::default();

        let mut stream = GenerationStream::new(
            model,
            tokenizer,
            backend,
            cache,
            "Hello".to_string(),
            config,
            streaming_config,
        )
        .unwrap();

        let mut token_count = 0;
        while let Some(result) = stream.next().await {
            let stream_response = result.expect("Stream should not error");
            assert!(!stream_response.text.is_empty());
            assert!(!stream_response.token_ids.is_empty());
            token_count += 1;

            // Prevent infinite loop in test
            if token_count > 10 {
                break;
            }
        }

        assert!(token_count > 0);
    }

    #[tokio::test]
    async fn test_streaming_config() {
        let config = StreamingConfig {
            buffer_size: 5,
            flush_interval_ms: 100,
            max_retries: 3,
            token_timeout_ms: 5000,
            cancellable: true,
        };

        assert_eq!(config.buffer_size, 5);
        assert_eq!(config.flush_interval_ms, 100);
    }

    #[tokio::test]
    async fn test_token_id_streaming() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer);
        let backend = Box::new(MockBackend);
        let cache = Arc::new(RwLock::new(KVCache::new(Default::default()).unwrap()));

        let config = GenerationConfig {
            max_new_tokens: 5,
            seed: Some(42), // Ensure reproducible token generation
            ..Default::default()
        };

        let streaming_config = StreamingConfig::default();

        let mut stream = GenerationStream::new(
            model,
            tokenizer,
            backend,
            cache,
            "Hello".to_string(),
            config,
            streaming_config,
        )
        .unwrap();

        let mut total_responses = 0;
        let mut total_token_ids = Vec::new();

        while let Some(result) = stream.next().await {
            let stream_response = result.expect("Stream should not error");
            assert!(!stream_response.text.is_empty());
            assert!(!stream_response.token_ids.is_empty());
            total_responses += 1;
            total_token_ids.extend(stream_response.token_ids);

            // Prevent infinite loop
            if total_responses > 10 {
                break;
            }
        }

        assert!(total_responses > 0, "No streaming responses generated");
        assert!(!total_token_ids.is_empty(), "No token IDs generated");
        // Each streaming response should have at least one token ID
        assert!(
            total_token_ids.len() >= total_responses,
            "Should have at least one token ID per response"
        );
    }
}
