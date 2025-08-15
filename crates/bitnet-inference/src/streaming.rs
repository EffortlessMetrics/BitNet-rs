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
use std::task::{Context as TaskContext, Poll};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, instrument, warn};

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
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self { buffer_size: 10, flush_interval_ms: 50 }
    }
}

/// A stream of generated tokens
pub struct GenerationStream {
    receiver: mpsc::Receiver<Result<String>>,
    _handle: tokio::task::JoinHandle<()>,
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
    ) -> Self {
        let (sender, receiver) = mpsc::channel(streaming_config.buffer_size);

        let handle = tokio::spawn(async move {
            if let Err(e) = Self::generate_stream_internal(
                model,
                tokenizer,
                backend,
                cache,
                prompt,
                generation_config,
                streaming_config,
                sender.clone(),
            )
            .await
            {
                warn!("Stream generation failed: {}", e);
                let _ = sender.send(Err(e)).await;
            }
        });

        Self { receiver, _handle: handle }
    }

    /// Internal streaming generation implementation
    async fn generate_stream_internal(
        _model: Arc<dyn Model>,
        tokenizer: Arc<dyn Tokenizer>,
        backend: Box<dyn Backend>,
        cache: Arc<RwLock<KVCache>>,
        prompt: String,
        config: GenerationConfig,
        streaming_config: StreamingConfig,
        sender: mpsc::Sender<Result<String>>,
    ) -> Result<()> {
        debug!("Starting streaming generation for prompt");

        // Tokenize input
        let input_tokens =
            tokenizer.encode(&prompt, true).context("Failed to tokenize input prompt")?;

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
        let mut last_flush = std::time::Instant::now();

        for _ in 0..config.max_new_tokens {
            // Check if receiver is closed (client disconnected)
            if sender.is_closed() {
                debug!("Client disconnected, stopping generation");
                break;
            }

            // Forward pass through model
            let logits = Self::forward_pass(&*backend, &current_tokens, &cache, &tokenizer).await?;

            // Sample next token
            let next_token = sampling_strategy.sample(&logits, &current_tokens)?;

            // Check for stop conditions
            if Self::should_stop(next_token, &current_tokens, &config, &tokenizer) {
                break;
            }

            // Decode token to text
            let token_text =
                tokenizer.decode(&[next_token], true).context("Failed to decode token")?;

            token_buffer.push(token_text);
            current_tokens.push(next_token);
            generated_count += 1;

            // Flush buffer based on size or time
            let should_flush = token_buffer.len() >= streaming_config.buffer_size
                || last_flush.elapsed().as_millis() >= streaming_config.flush_interval_ms as u128;

            if should_flush && !token_buffer.is_empty() {
                let buffered_text = token_buffer.join("");
                if sender.send(Ok(buffered_text)).await.is_err() {
                    debug!("Client disconnected during send");
                    break;
                }
                token_buffer.clear();
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
            let _ = sender.send(Ok(remaining_text)).await;
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
        let output_tensor = backend.forward(&input_tensor, &mut *cache_guard).await?;

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
        // Check for EOS token
        if let Some(eos_token) = tokenizer.eos_token_id() {
            if token == eos_token {
                return true;
            }
        }

        // Check for stop sequences
        if !config.stop_sequences.is_empty() {
            if let Ok(current_text) = tokenizer.decode(current_tokens, true) {
                for stop_seq in &config.stop_sequences {
                    if current_text.ends_with(stop_seq) {
                        return true;
                    }
                }
            }
        }

        false
    }
}

impl Stream for GenerationStream {
    type Item = Result<String>;

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
        fn encode(&self, _text: &str, _add_special_tokens: bool) -> bitnet_common::Result<Vec<u32>> {
            Ok(vec![1, 2, 3])
        }

        fn decode(&self, tokens: &[u32], _skip_special_tokens: bool) -> bitnet_common::Result<String> {
            Ok(format!("token_{}", tokens.len()))
        }

        fn vocab_size(&self) -> usize {
            50257
        }

        fn eos_token_id(&self) -> Option<u32> {
            Some(50256)
        }

        fn pad_token_id(&self) -> Option<u32> {
            None
        }
    }

    struct MockBackend;

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
        );

        let mut token_count = 0;
        while let Some(result) = stream.next().await {
            match result {
                Ok(token_text) => {
                    assert!(!token_text.is_empty());
                    token_count += 1;
                }
                Err(e) => {
                    panic!("Stream error: {}", e);
                }
            }

            // Prevent infinite loop in test
            if token_count > 10 {
                break;
            }
        }

        assert!(token_count > 0);
    }

    #[tokio::test]
    async fn test_streaming_config() {
        let config = StreamingConfig { buffer_size: 5, flush_interval_ms: 100 };

        assert_eq!(config.buffer_size, 5);
        assert_eq!(config.flush_interval_ms, 100);
    }
}
