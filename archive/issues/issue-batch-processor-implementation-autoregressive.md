# [Inference] Implement BatchProcessor for Efficient Multi-Sequence Generation

## Problem Description

The `BatchProcessor` struct and `generate_token_batched` method in `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/generation/autoregressive.rs` are placeholders that fall back to single token generation, missing the critical performance benefits of batched inference for concurrent requests. This significantly limits throughput in production scenarios where multiple inference requests should be processed efficiently.

## Environment

- **File**: `crates/bitnet-inference/src/generation/autoregressive.rs`
- **Struct**: `BatchProcessor`
- **Method**: `AutoregressiveGenerator::generate_token_batched`
- **MSRV**: Rust 1.90.0
- **Feature flags**: Both `cpu` and `gpu` benefit from batched operations

## Current Implementation Analysis

### Existing Placeholder Code
```rust
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
```

### Performance Impact
- **CPU**: 3-5x throughput improvement with proper batching
- **GPU**: 5-10x throughput improvement due to parallel processing
- **Memory**: More efficient memory utilization through shared computations
- **Latency**: Better amortization of model loading and computation overhead

## Root Cause Analysis

1. **Incomplete Implementation**: Batching infrastructure was planned but not implemented
2. **Complex Coordination**: Batching requires sophisticated request queuing and result distribution
3. **Memory Management**: Efficient batch tensor creation and management needs careful design
4. **Synchronization**: Multiple concurrent requests need proper coordination

## Impact Assessment

### Severity: High
### Affected Components: Production inference throughput, concurrent request handling

**Performance Impact:**
- 5-10x throughput loss for concurrent inference requests
- Inefficient GPU utilization in multi-user scenarios
- Higher latency per request due to sequential processing
- Suboptimal memory usage patterns

**Scalability Impact:**
- Cannot handle high-concurrency inference workloads
- Poor resource utilization in production deployments
- Increased infrastructure costs due to inefficiency

## Proposed Solution

### Primary Approach: Dynamic Batch Processing System

Implement a sophisticated batch processing system that dynamically collects inference requests, batches them efficiently, and distributes results back to individual requests.

#### Implementation Plan

**1. Batch Request Management**

```rust
use std::collections::VecDeque;
use tokio::sync::{mpsc, oneshot, Mutex};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Request for batched token generation
#[derive(Debug)]
pub struct BatchRequest {
    /// Current token sequence for this request
    pub tokens: Vec<usize>,
    /// Generation step
    pub step: usize,
    /// Response channel for sending result back
    pub response: oneshot::Sender<Result<usize>>,
    /// Request timestamp for timeout handling
    pub timestamp: Instant,
    /// Unique request ID for tracking
    pub request_id: u64,
}

/// Configuration for batch processing behavior
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size (number of sequences)
    pub max_batch_size: usize,
    /// Maximum wait time before processing partial batch
    pub max_wait_time: Duration,
    /// Maximum sequence length in batch
    pub max_sequence_length: usize,
    /// Memory limit for batch tensors
    pub memory_limit_mb: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_wait_time: Duration::from_millis(10),
            max_sequence_length: 2048,
            memory_limit_mb: 1024,
        }
    }
}

/// Dynamic batch processor for efficient multi-sequence generation
pub struct BatchProcessor {
    /// Configuration for batch behavior
    config: BatchConfig,
    /// Queue of pending requests
    request_queue: Arc<Mutex<VecDeque<BatchRequest>>>,
    /// Channel for receiving new requests
    request_sender: mpsc::UnboundedSender<BatchRequest>,
    request_receiver: Arc<Mutex<mpsc::UnboundedReceiver<BatchRequest>>>,
    /// Next request ID for tracking
    next_request_id: Arc<std::sync::atomic::AtomicU64>,
    /// Performance metrics
    metrics: BatchMetrics,
}

#[derive(Debug, Default)]
pub struct BatchMetrics {
    pub total_requests: u64,
    pub total_batches: u64,
    pub average_batch_size: f64,
    pub average_wait_time_ms: f64,
    pub throughput_requests_per_sec: f64,
}
```

**2. Batch Assembly and Processing**

```rust
impl BatchProcessor {
    pub fn new(config: BatchConfig) -> Self {
        let (request_sender, request_receiver) = mpsc::unbounded_channel();

        Self {
            config,
            request_queue: Arc::new(Mutex::new(VecDeque::new())),
            request_sender,
            request_receiver: Arc::new(Mutex::new(request_receiver)),
            next_request_id: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            metrics: BatchMetrics::default(),
        }
    }

    /// Submit a request for batched processing
    pub async fn submit_request(
        &self,
        tokens: Vec<usize>,
        step: usize,
    ) -> Result<usize> {
        let (response_tx, response_rx) = oneshot::channel();
        let request_id = self.next_request_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let request = BatchRequest {
            tokens,
            step,
            response: response_tx,
            timestamp: Instant::now(),
            request_id,
        };

        self.request_sender.send(request)
            .map_err(|_| bitnet_common::BitNetError::InferenceError("Batch processor stopped".to_string()))?;

        // Wait for response with timeout
        let result = tokio::time::timeout(Duration::from_secs(30), response_rx).await
            .map_err(|_| bitnet_common::BitNetError::InferenceError("Request timeout".to_string()))?
            .map_err(|_| bitnet_common::BitNetError::InferenceError("Request cancelled".to_string()))?;

        result
    }

    /// Start the batch processing loop
    pub async fn start_processing<F, Fut>(&mut self, forward_fn: F) -> Result<()>
    where
        F: Fn(BitNetTensor) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<BitNetTensor>> + Send,
    {
        let mut receiver = self.request_receiver.lock().await;
        let mut batch_timer = tokio::time::interval(self.config.max_wait_time);

        loop {
            let mut current_batch = Vec::new();
            let batch_start = Instant::now();

            // Collect requests for current batch
            'collect: loop {
                tokio::select! {
                    // New request available
                    request = receiver.recv() => {
                        match request {
                            Some(req) => {
                                current_batch.push(req);

                                // Check if batch is full
                                if current_batch.len() >= self.config.max_batch_size {
                                    break 'collect;
                                }
                            }
                            None => return Ok(()), // Channel closed
                        }
                    }

                    // Timeout reached - process partial batch
                    _ = batch_timer.tick() => {
                        if !current_batch.is_empty() {
                            break 'collect;
                        }
                    }
                }
            }

            if !current_batch.is_empty() {
                self.process_batch(current_batch, &forward_fn).await;
                self.metrics.total_batches += 1;
            }
        }
    }

    /// Process a batch of requests
    async fn process_batch<F, Fut>(&mut self, batch: Vec<BatchRequest>, forward_fn: &F)
    where
        F: Fn(BitNetTensor) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<BitNetTensor>> + Send,
    {
        let batch_size = batch.len();
        debug!("Processing batch of {} requests", batch_size);

        // Measure batch processing time
        let batch_start = Instant::now();

        // Assemble batch tensor
        let batch_result = self.assemble_and_process_batch(&batch, forward_fn).await;

        // Distribute results back to requests
        match batch_result {
            Ok(results) => {
                assert_eq!(results.len(), batch.len());
                for (request, result) in batch.into_iter().zip(results.into_iter()) {
                    let _ = request.response.send(Ok(result));
                }
            }
            Err(e) => {
                // Broadcast error to all requests in batch
                for request in batch {
                    let _ = request.response.send(Err(e.clone()));
                }
            }
        }

        // Update metrics
        let batch_duration = batch_start.elapsed();
        self.update_metrics(batch_size, batch_duration);
    }

    /// Assemble requests into batch tensor and process
    async fn assemble_and_process_batch<F, Fut>(
        &self,
        batch: &[BatchRequest],
        forward_fn: &F,
    ) -> Result<Vec<usize>>
    where
        F: Fn(BitNetTensor) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<BitNetTensor>> + Send,
    {
        // Find maximum sequence length in batch
        let max_len = batch.iter().map(|req| req.tokens.len()).max().unwrap_or(0);
        let batch_size = batch.len();

        debug!("Assembling batch: {} sequences, max length: {}", batch_size, max_len);

        // Create padded batch tensor [batch_size, max_len]
        let mut batch_tokens = vec![0usize; batch_size * max_len];
        let mut attention_mask = vec![0.0f32; batch_size * max_len];

        for (i, request) in batch.iter().enumerate() {
            let seq_len = request.tokens.len();
            let start_idx = i * max_len;

            // Copy tokens to batch tensor
            for (j, &token) in request.tokens.iter().enumerate() {
                batch_tokens[start_idx + j] = token;
                attention_mask[start_idx + j] = 1.0; // Mark as valid token
            }

            // Pad remaining positions (already initialized to 0)
        }

        // Convert to BitNetTensor
        let batch_tensor = self.create_batch_tensor(batch_tokens, batch_size, max_len)?;
        let mask_tensor = self.create_attention_mask(attention_mask, batch_size, max_len)?;

        // Execute forward pass on batch
        let logits = forward_fn(batch_tensor).await?;

        // Extract next tokens for each sequence
        self.extract_next_tokens(logits, batch, mask_tensor)
    }

    fn create_batch_tensor(&self, tokens: Vec<usize>, batch_size: usize, seq_len: usize) -> Result<BitNetTensor> {
        // Convert to appropriate tensor format
        // This depends on the specific BitNetTensor implementation
        let data: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
        BitNetTensor::from_vec(data, &[batch_size, seq_len])
    }

    fn create_attention_mask(&self, mask: Vec<f32>, batch_size: usize, seq_len: usize) -> Result<BitNetTensor> {
        BitNetTensor::from_vec(mask, &[batch_size, seq_len])
    }

    fn extract_next_tokens(
        &self,
        logits: BitNetTensor,
        batch: &[BatchRequest],
        attention_mask: BitNetTensor,
    ) -> Result<Vec<usize>> {
        // Extract logits for last valid position of each sequence
        let batch_size = batch.len();
        let vocab_size = logits.shape().last().copied().unwrap_or(50257);

        let mut next_tokens = Vec::with_capacity(batch_size);

        for (i, request) in batch.iter().enumerate() {
            let last_pos = request.tokens.len().saturating_sub(1);

            // Extract logits for this sequence at last position
            let logit_start = i * vocab_size * logits.shape()[1] + last_pos * vocab_size;
            let sequence_logits = logits.data_slice(logit_start, vocab_size)?;

            // Apply sampling (simple argmax for now)
            let next_token = self.sample_next_token(sequence_logits)?;
            next_tokens.push(next_token);
        }

        Ok(next_tokens)
    }

    fn sample_next_token(&self, logits: &[f32]) -> Result<usize> {
        // Simple argmax sampling
        let max_idx = logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| bitnet_common::BitNetError::InferenceError("Empty logits".to_string()))?;

        Ok(max_idx)
    }

    fn update_metrics(&mut self, batch_size: usize, duration: Duration) {
        self.metrics.total_requests += batch_size as u64;

        // Update running averages
        let alpha = 0.1; // Exponential moving average factor
        self.metrics.average_batch_size =
            alpha * batch_size as f64 + (1.0 - alpha) * self.metrics.average_batch_size;

        let duration_ms = duration.as_millis() as f64;
        self.metrics.average_wait_time_ms =
            alpha * duration_ms + (1.0 - alpha) * self.metrics.average_wait_time_ms;

        let requests_per_sec = batch_size as f64 / duration.as_secs_f64();
        self.metrics.throughput_requests_per_sec =
            alpha * requests_per_sec + (1.0 - alpha) * self.metrics.throughput_requests_per_sec;
    }

    pub fn get_metrics(&self) -> &BatchMetrics {
        &self.metrics
    }
}
```

**3. Integration with AutoregressiveGenerator**

```rust
impl AutoregressiveGenerator {
    /// Enhanced batch token generation with proper batching
    async fn generate_token_batched<F, Fut>(
        &mut self,
        current_tokens: &[usize],
        forward_fn: &F,
        step: usize,
    ) -> Result<usize>
    where
        F: Fn(BitNetTensor) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<BitNetTensor>> + Send,
    {
        // Use the batch processor for efficient multi-sequence generation
        if let Some(batch_processor) = &self.batch_processor {
            debug!("Using batch processor for token generation at step {}", step);

            // Submit request to batch processor
            batch_processor.submit_request(current_tokens.to_vec(), step).await
        } else {
            // Fallback to single token generation if batch processor not available
            debug!("Batch processor not available, falling back to single token generation");
            self.generate_token_single(current_tokens, forward_fn, step).await
        }
    }

    /// Initialize batch processor with configuration
    pub fn enable_batching(&mut self, config: BatchConfig) -> Result<()> {
        let batch_processor = Arc::new(BatchProcessor::new(config));

        // Start background task for batch processing
        let processor_clone = batch_processor.clone();
        let forward_fn = self.create_forward_function()?;

        tokio::spawn(async move {
            let mut processor = processor_clone.as_ref().clone();
            if let Err(e) = processor.start_processing(forward_fn).await {
                error!("Batch processor error: {}", e);
            }
        });

        self.batch_processor = Some(batch_processor);
        info!("Batch processing enabled");

        Ok(())
    }

    /// Create forward function for batch processor
    fn create_forward_function(&self) -> Result<impl Fn(BitNetTensor) -> impl std::future::Future<Output = Result<BitNetTensor>> + Send + Sync + Clone> {
        let model = self.model.clone();
        let cache = self.cache.clone();

        Ok(move |input: BitNetTensor| {
            let model = model.clone();
            let cache = cache.clone();

            async move {
                let mut cache_guard = cache.lock().await;
                model.forward(&input, &mut *cache_guard).await
            }
        })
    }
}
```

**4. Memory-Efficient Tensor Operations**

```rust
impl BatchProcessor {
    /// Optimized batch tensor creation with memory pooling
    fn create_optimized_batch_tensor(&self, requests: &[BatchRequest]) -> Result<BatchTensorInfo> {
        let batch_size = requests.len();
        let max_len = requests.iter().map(|r| r.tokens.len()).max().unwrap_or(0);

        // Estimate memory requirements
        let estimated_memory_mb = (batch_size * max_len * 4) / (1024 * 1024); // 4 bytes per token

        if estimated_memory_mb > self.config.memory_limit_mb {
            return Err(bitnet_common::BitNetError::InferenceError(
                format!("Batch memory requirement ({} MB) exceeds limit ({} MB)",
                    estimated_memory_mb, self.config.memory_limit_mb)
            ));
        }

        // Use memory pool for efficient allocation
        let tensor_data = self.memory_pool.allocate_batch_tensor(batch_size, max_len)?;

        Ok(BatchTensorInfo {
            data: tensor_data,
            batch_size,
            sequence_length: max_len,
            memory_usage_mb: estimated_memory_mb,
        })
    }

    /// Adaptive batching based on available memory
    fn calculate_optimal_batch_size(&self, pending_requests: &[BatchRequest]) -> usize {
        let available_memory_mb = self.get_available_memory_mb();
        let average_seq_len = pending_requests.iter()
            .map(|r| r.tokens.len())
            .sum::<usize>() / pending_requests.len().max(1);

        let max_batch_by_memory = (available_memory_mb * 1024 * 1024) / (average_seq_len * 4);

        max_batch_by_memory.min(self.config.max_batch_size).min(pending_requests.len())
    }
}

#[derive(Debug)]
struct BatchTensorInfo {
    data: BitNetTensor,
    batch_size: usize,
    sequence_length: usize,
    memory_usage_mb: usize,
}
```

**5. Performance Monitoring and Optimization**

```rust
impl BatchProcessor {
    /// Dynamic configuration adjustment based on performance
    pub fn optimize_config(&mut self) {
        let metrics = &self.metrics;

        // Adjust batch size based on throughput
        if metrics.throughput_requests_per_sec < 10.0 && self.config.max_batch_size < 16 {
            self.config.max_batch_size += 2;
            info!("Increased batch size to {}", self.config.max_batch_size);
        } else if metrics.average_wait_time_ms > 50.0 && self.config.max_batch_size > 2 {
            self.config.max_batch_size -= 1;
            info!("Decreased batch size to {}", self.config.max_batch_size);
        }

        // Adjust wait time based on request patterns
        if metrics.average_batch_size < 2.0 {
            self.config.max_wait_time = Duration::from_millis(5); // Reduce wait for sparse requests
        } else {
            self.config.max_wait_time = Duration::from_millis(15); // Allow more time for fuller batches
        }
    }

    /// Get detailed performance report
    pub fn get_performance_report(&self) -> PerformanceReport {
        PerformanceReport {
            total_requests: self.metrics.total_requests,
            total_batches: self.metrics.total_batches,
            average_batch_size: self.metrics.average_batch_size,
            batch_utilization: self.metrics.average_batch_size / self.config.max_batch_size as f64,
            throughput_requests_per_sec: self.metrics.throughput_requests_per_sec,
            average_latency_ms: self.metrics.average_wait_time_ms,
            efficiency_score: self.calculate_efficiency_score(),
        }
    }

    fn calculate_efficiency_score(&self) -> f64 {
        let batch_efficiency = self.metrics.average_batch_size / self.config.max_batch_size as f64;
        let latency_efficiency = (50.0 - self.metrics.average_wait_time_ms.min(50.0)) / 50.0;

        (batch_efficiency + latency_efficiency) / 2.0
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub total_requests: u64,
    pub total_batches: u64,
    pub average_batch_size: f64,
    pub batch_utilization: f64,
    pub throughput_requests_per_sec: f64,
    pub average_latency_ms: f64,
    pub efficiency_score: f64,
}
```

## Testing Strategy

### Performance Testing
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_batch_processing_throughput() {
        let config = BatchConfig {
            max_batch_size: 4,
            max_wait_time: Duration::from_millis(10),
            ..Default::default()
        };

        let mut processor = BatchProcessor::new(config);

        // Simulate concurrent requests
        let mut handles = Vec::new();
        for i in 0..10 {
            let tokens = vec![1, 2, 3, i];
            let handle = processor.submit_request(tokens, 0);
            handles.push(handle);
        }

        // All requests should complete
        for handle in handles {
            assert!(handle.await.is_ok());
        }

        let report = processor.get_performance_report();
        assert!(report.efficiency_score > 0.5);
    }

    #[tokio::test]
    async fn test_memory_limit_enforcement() {
        let config = BatchConfig {
            max_batch_size: 1000,
            memory_limit_mb: 10, // Very low limit
            ..Default::default()
        };

        let processor = BatchProcessor::new(config);

        // Large requests should be rejected or batched appropriately
        let large_tokens = vec![1; 10000]; // Large sequence
        let result = processor.submit_request(large_tokens, 0).await;

        // Should either succeed with smaller batch or fail gracefully
        assert!(result.is_ok() || result.unwrap_err().to_string().contains("memory"));
    }
}
```

### Integration Testing
```bash
# Test batch processing with different batch sizes
cargo test --release batch_processing -- --nocapture

# Performance benchmarks
cargo bench batch_throughput

# Memory usage testing
cargo test --release memory_efficiency
```

## Acceptance Criteria

- [ ] Functional batch processing that handles multiple concurrent requests
- [ ] 3-5x throughput improvement for CPU inference with batching
- [ ] 5-10x throughput improvement for GPU inference with batching
- [ ] Dynamic batch size adaptation based on memory and performance
- [ ] Comprehensive error handling and timeout management
- [ ] Memory-efficient tensor operations with pooling
- [ ] Performance monitoring and optimization capabilities
- [ ] Zero regression in single-request latency
- [ ] Graceful fallback to single-token generation when batching unavailable
- [ ] Complete test coverage for batch processing scenarios

## Dependencies

### New Dependencies
```toml
[dependencies]
tokio = { version = "1.0", features = ["sync", "time"] }
```

## Related Issues

- Memory pool implementation for efficient tensor allocation
- Performance monitoring integration
- GPU batch processing optimization
- Request queuing and prioritization

## BitNet-Specific Considerations

- **Quantization Efficiency**: Batched operations with I2S, TL1, TL2 quantization
- **Memory Patterns**: Efficient memory usage with 1-bit quantized weights
- **Device Optimization**: Different batching strategies for CPU vs GPU backends
- **KV Cache**: Efficient KV cache management for batched sequences

This implementation will transform BitNet-rs from a single-request system into a production-ready multi-user inference engine capable of handling concurrent requests efficiently.
