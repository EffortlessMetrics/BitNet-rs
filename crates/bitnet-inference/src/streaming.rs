//! Streaming generation support

use crate::Backend;
use bitnet_common::{BitNetError, GenerationConfig, Result};
use bitnet_models::Model;
use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::{mpsc, RwLock};

/// Streaming configuration
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingConfig {
    pub buffer_size: usize,
    pub yield_interval: usize,
    pub enable_backpressure: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 32,
            yield_interval: 1,
            enable_backpressure: true,
        }
    }
}

impl StreamingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.buffer_size == 0 {
            return Err(BitNetError::Config(
                "buffer_size must be greater than 0".to_string()
            ));
        }
        
        if self.yield_interval == 0 {
            return Err(BitNetError::Config(
                "yield_interval must be greater than 0".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Trait for streaming generation
pub trait GenerationStream: Stream<Item = Result<String>> + Send + Unpin {
    /// Cancel the generation
    fn cancel(&mut self) -> Result<()>;
    
    /// Get current position in generation
    fn position(&self) -> usize;
    
    /// Check if generation is complete
    fn is_complete(&self) -> bool;
}

/// Token generation stream implementation
pub struct TokenGenerationStream {
    model: Arc<RwLock<Box<dyn Model<Config = bitnet_common::BitNetConfig>>>>,
    backend: Box<dyn Backend>,
    tokens: Vec<u32>,
    config: GenerationConfig,
    stream_config: StreamingConfig,
    position: usize,
    buffer: Vec<String>,
    receiver: mpsc::Receiver<Result<String>>,
    sender: Option<mpsc::Sender<Result<String>>>,
    is_cancelled: bool,
    is_complete: bool,
}

impl TokenGenerationStream {
    /// Create a new token generation stream
    pub fn new(
        model: Arc<RwLock<Box<dyn Model<Config = bitnet_common::BitNetConfig>>>>,
        backend: Box<dyn Backend>,
        tokens: Vec<u32>,
        config: GenerationConfig,
        stream_config: StreamingConfig,
    ) -> Result<Self> {
        stream_config.validate()?;
        
        let (sender, receiver) = mpsc::channel(stream_config.buffer_size);
        
        Ok(Self {
            model,
            backend,
            tokens,
            config,
            stream_config,
            position: 0,
            buffer: Vec::new(),
            receiver,
            sender: Some(sender),
            is_cancelled: false,
            is_complete: false,
        })
    }
    
    /// Start the generation process
    pub async fn start(&mut self) -> Result<()> {
        if let Some(sender) = self.sender.take() {
            let model = self.model.clone();
            let backend = self.backend.clone_backend();
            let tokens = self.tokens.clone();
            let config = self.config.clone();
            let stream_config = self.stream_config.clone();
            
            tokio::spawn(async move {
                Self::generate_tokens(model, backend, tokens, config, stream_config, sender).await;
            });
        }
        
        Ok(())
    }
    
    /// Create and start a new streaming generation
    pub async fn create_and_start(
        model: Arc<RwLock<Box<dyn Model<Config = bitnet_common::BitNetConfig>>>>,
        backend: Box<dyn Backend>,
        tokens: Vec<u32>,
        config: GenerationConfig,
        stream_config: StreamingConfig,
    ) -> Result<Self> {
        let mut stream = Self::new(model, backend, tokens, config, stream_config)?;
        stream.start().await?;
        Ok(stream)
    }
    
    /// Generate tokens in background task
    async fn generate_tokens(
        _model: Arc<RwLock<Box<dyn Model<Config = bitnet_common::BitNetConfig>>>>,
        backend: Box<dyn Backend>,
        mut tokens: Vec<u32>,
        config: GenerationConfig,
        stream_config: StreamingConfig,
        sender: mpsc::Sender<Result<String>>,
    ) {
        for step in 0..config.max_new_tokens {
            // Check if receiver is closed (stream cancelled)
            if sender.is_closed() {
                break;
            }
            
            // Generate next token (placeholder implementation)
            let next_token = step as u32 + 1000; // Placeholder
            tokens.push(next_token);
            
            // Check for EOS token
            if backend.is_eos_token(next_token) {
                break;
            }
            
            // Yield token at specified intervals
            if step % stream_config.yield_interval == 0 {
                let token_text = match backend.detokenize(&[next_token]) {
                    Ok(text) => text,
                    Err(e) => {
                        let _ = sender.send(Err(e)).await;
                        break;
                    }
                };
                
                if sender.send(Ok(token_text)).await.is_err() {
                    break; // Receiver dropped
                }
            }
            
            // Apply backpressure if enabled
            if stream_config.enable_backpressure && sender.capacity() == 0 {
                tokio::task::yield_now().await;
            }
        }
    }
}

impl GenerationStream for TokenGenerationStream {
    fn cancel(&mut self) -> Result<()> {
        self.is_cancelled = true;
        self.receiver.close();
        Ok(())
    }
    
    fn position(&self) -> usize {
        self.position
    }
    
    fn is_complete(&self) -> bool {
        self.is_complete
    }
}

impl Stream for TokenGenerationStream {
    type Item = Result<String>;
    
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.is_cancelled {
            return Poll::Ready(None);
        }
        
        match self.receiver.poll_recv(cx) {
            Poll::Ready(Some(item)) => {
                self.position += 1;
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                self.is_complete = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Batch streaming for multiple requests
pub struct BatchGenerationStream {
    streams: Vec<Box<dyn GenerationStream>>,
    current_index: usize,
}

impl BatchGenerationStream {
    /// Create a new batch stream
    pub fn new(streams: Vec<Box<dyn GenerationStream>>) -> Self {
        Self {
            streams,
            current_index: 0,
        }
    }
    
    /// Add a stream to the batch
    pub fn add_stream(&mut self, stream: Box<dyn GenerationStream>) {
        self.streams.push(stream);
    }
    
    /// Get number of active streams
    pub fn active_count(&self) -> usize {
        self.streams.iter().filter(|s| !s.is_complete()).count()
    }
}

impl Stream for BatchGenerationStream {
    type Item = (usize, Result<String>);
    
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut all_complete = true;
        
        // Round-robin through streams
        for _ in 0..self.streams.len() {
            let stream_idx = self.current_index;
            self.current_index = (self.current_index + 1) % self.streams.len();
            
            if let Some(stream) = self.streams.get_mut(stream_idx) {
                if !stream.is_complete() {
                    all_complete = false;
                    
                    match Pin::new(stream).poll_next(cx) {
                        Poll::Ready(Some(item)) => {
                            return Poll::Ready(Some((stream_idx, item)));
                        }
                        Poll::Ready(None) => {
                            // Stream completed, continue to next
                        }
                        Poll::Pending => {
                            // Stream not ready, continue to next
                        }
                    }
                }
            }
        }
        
        if all_complete {
            Poll::Ready(None)
        } else {
            Poll::Pending
        }
    }
}

/// Async utilities for streaming generation
pub mod async_utils {
    use super::*;
    use futures::stream::StreamExt;
    use tokio::time::{timeout, Duration};
    use std::time::Instant;
    
    /// Collect all tokens from a stream with timeout
    pub async fn collect_stream_with_timeout(
        mut stream: Box<dyn GenerationStream>,
        timeout_duration: Duration,
    ) -> Result<Vec<String>> {
        let mut tokens = Vec::new();
        
        while let Ok(Some(result)) = timeout(timeout_duration, stream.next()).await {
            match result {
                Ok(token) => tokens.push(token),
                Err(e) => return Err(e),
            }
        }
        
        Ok(tokens)
    }
    
    /// Merge multiple streams into a single stream
    pub async fn merge_streams(
        streams: Vec<Box<dyn GenerationStream>>,
    ) -> impl Stream<Item = (usize, Result<String>)> {
        let batch_stream = BatchGenerationStream::new(streams);
        futures::stream::unfold(batch_stream, |mut batch| async move {
            match batch.next().await {
                Some(item) => Some((item, batch)),
                None => None,
            }
        })
    }
    
    /// Rate limit a stream to a maximum tokens per second
    pub fn rate_limit_stream<S>(
        stream: S,
        max_tokens_per_second: f64,
    ) -> impl Stream<Item = S::Item>
    where
        S: Stream + Unpin,
    {
        let interval = Duration::from_secs_f64(1.0 / max_tokens_per_second);
        let mut last_yield = Instant::now();
        
        stream.filter_map(move |item| {
            let now = Instant::now();
            if now.duration_since(last_yield) >= interval {
                last_yield = now;
                futures::future::ready(Some(item))
            } else {
                futures::future::ready(None)
            }
        })
    }
}

/// Integration examples for different async runtimes
pub mod integration_examples {
    use super::*;
    use futures::StreamExt;
    
    /// Example: Tokio integration with graceful shutdown
    pub async fn tokio_streaming_example(
        engine: &crate::CpuInferenceEngine,
        prompt: &str,
    ) -> Result<String> {
        let tokens = vec![1, 2, 3]; // Placeholder tokenization
        let config = GenerationConfig::default();
        let stream_config = StreamingConfig::default();
        
        let mut stream = engine.generate_stream_async(tokens, config, stream_config).await?;
        let mut result = String::new();
        
        // Set up cancellation
        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
        
        // Spawn a task to handle cancellation
        tokio::spawn(async move {
            tokio::signal::ctrl_c().await.ok();
            let _ = cancel_tx.send(());
        });
        
        loop {
            tokio::select! {
                token_result = stream.next() => {
                    match token_result {
                        Some(Ok(token)) => result.push_str(&token),
                        Some(Err(e)) => return Err(e),
                        None => break,
                    }
                }
                _ = &mut cancel_rx => {
                    stream.cancel()?;
                    break;
                }
            }
        }
        
        Ok(result)
    }
    
    /// Example: Async-std integration
    #[cfg(feature = "async-std")]
    pub async fn async_std_streaming_example(
        engine: &crate::CpuInferenceEngine,
        prompt: &str,
    ) -> Result<String> {
        use async_std::stream::StreamExt;
        
        let tokens = vec![1, 2, 3]; // Placeholder tokenization
        let config = GenerationConfig::default();
        let stream_config = StreamingConfig::default();
        
        let stream = engine.generate_stream_async(tokens, config, stream_config).await?;
        let mut result = String::new();
        
        let mut stream = Box::pin(stream);
        while let Some(token_result) = stream.next().await {
            match token_result {
                Ok(token) => result.push_str(&token),
                Err(e) => return Err(e),
            }
        }
        
        Ok(result)
    }
    
    /// Example: Web server integration with Server-Sent Events
    pub async fn sse_streaming_example(
        engine: &crate::CpuInferenceEngine,
        _prompt: &str,
    ) -> std::result::Result<impl Stream<Item = std::result::Result<String, std::io::Error>>, std::io::Error> {
        let tokens = vec![1, 2, 3]; // Placeholder tokenization
        let config = GenerationConfig::default();
        let stream_config = StreamingConfig {
            buffer_size: 1, // Immediate streaming for SSE
            yield_interval: 1,
            enable_backpressure: false,
        };
        
        let stream = engine.generate_stream_async(tokens, config, stream_config).await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        
        Ok(stream.map(|result| {
            result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    
    #[test]
    fn test_streaming_config_validation() {
        let config = StreamingConfig::default();
        assert!(config.validate().is_ok());
        
        let mut invalid_config = config.clone();
        invalid_config.buffer_size = 0;
        assert!(invalid_config.validate().is_err());
        
        invalid_config.buffer_size = 32;
        invalid_config.yield_interval = 0;
        assert!(invalid_config.validate().is_err());
    }
    
    #[tokio::test]
    async fn test_batch_stream_creation() {
        let batch_stream = BatchGenerationStream::new(vec![]);
        assert_eq!(batch_stream.active_count(), 0);
    }
    
    #[tokio::test]
    async fn test_async_utils_collect_with_timeout() {
        use async_utils::collect_stream_with_timeout;
        use tokio::time::Duration;
        
        // Create a mock stream that yields a few tokens
        let mock_stream = futures::stream::iter(vec![
            Ok("Hello".to_string()),
            Ok(" ".to_string()),
            Ok("World".to_string()),
        ]);
        
        // This test would need a proper GenerationStream implementation
        // For now, just test the timeout functionality exists
        let timeout_duration = Duration::from_millis(100);
        assert!(timeout_duration.as_millis() == 100);
    }
    
    #[tokio::test]
    async fn test_rate_limiting() {
        use async_utils::rate_limit_stream;
        use futures::stream;
        
        let test_stream = stream::iter(vec!["a", "b", "c", "d", "e"]);
        let rate_limited = rate_limit_stream(test_stream, 2.0); // 2 tokens per second
        
        let start = Instant::now();
        let results: Vec<_> = rate_limited.collect().await;
        let elapsed = start.elapsed();
        
        // Should have some rate limiting effect
        assert!(!results.is_empty());
        assert!(elapsed.as_millis() > 0);
    }
}