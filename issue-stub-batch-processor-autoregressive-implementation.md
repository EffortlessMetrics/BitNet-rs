# [STUB] BatchProcessor in autoregressive.rs falls back to single token generation

## Problem Description

The `BatchProcessor` struct exists in `autoregressive.rs` but the `generate_token_batched` method falls back to single token generation, preventing efficient batch processing of multiple inference requests and significantly limiting throughput in production scenarios.

## Environment

**File**: `crates/bitnet-inference/src/generation/autoregressive.rs`
**Component**: Autoregressive Generation Engine with Batch Processing
**Issue Type**: Stub Implementation / Missing Batch Processing Logic

## Root Cause Analysis

**Current Implementation:**
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

**Analysis:**
1. **Stub Implementation**: Method exists but delegates to single token generation
2. **Performance Impact**: No batching benefits for multiple concurrent requests
3. **Missing Infrastructure**: Batch collection, tensor combination, and result distribution not implemented
4. **Scalability Limitation**: System cannot efficiently handle concurrent inference requests

## Impact Assessment

**Severity**: High
**Affected Areas**:
- Inference throughput and scalability
- GPU resource utilization efficiency
- Production deployment performance
- Concurrent request handling

**Performance Impact**:
- Missing 5-50x throughput improvements from batching
- Inefficient GPU utilization with single-request processing
- Higher latency for concurrent requests due to serialization
- Suboptimal resource usage in production environments

**Business Impact**:
- Reduced competitive advantage in high-throughput scenarios
- Higher infrastructure costs due to inefficient resource usage
- Poor scaling characteristics for production deployments

## Proposed Solution

### Complete Batch Processing Implementation

```rust
#[derive(Debug)]
pub struct BatchProcessor {
    max_batch_size: usize,
    batch_timeout_ms: u64,
    pending_requests: VecDeque<BatchRequest>,
    active_batch: Option<ActiveBatch>,
}

#[derive(Debug)]
struct BatchRequest {
    request_id: RequestId,
    tokens: Vec<usize>,
    step: usize,
    response_sender: oneshot::Sender<Result<usize>>,
    queued_at: Instant,
}

#[derive(Debug)]
struct ActiveBatch {
    requests: Vec<BatchRequest>,
    combined_tensor: BitNetTensor,
    batch_id: BatchId,
    started_at: Instant,
}

impl BatchProcessor {
    pub fn new(max_batch_size: usize, batch_timeout_ms: u64) -> Self {
        Self {
            max_batch_size,
            batch_timeout_ms,
            pending_requests: VecDeque::new(),
            active_batch: None,
        }
    }

    pub async fn process_request(
        &mut self,
        tokens: Vec<usize>,
        step: usize,
    ) -> Result<usize> {
        let (tx, rx) = oneshot::channel();
        let request = BatchRequest {
            request_id: RequestId::new(),
            tokens,
            step,
            response_sender: tx,
            queued_at: Instant::now(),
        };

        self.pending_requests.push_back(request);

        // Try to form a batch
        if self.should_create_batch() {
            self.create_and_process_batch().await?;
        }

        // Wait for response
        rx.await.map_err(|_| anyhow::anyhow!("Request cancelled"))?
    }

    fn should_create_batch(&self) -> bool {
        // Create batch if we have enough requests or timeout is reached
        self.pending_requests.len() >= self.max_batch_size ||
        (self.pending_requests.len() > 0 &&
         self.pending_requests.front().unwrap().queued_at.elapsed().as_millis() >= self.batch_timeout_ms as u128)
    }

    async fn create_and_process_batch(&mut self) -> Result<()> {
        if self.pending_requests.is_empty() {
            return Ok(());
        }

        // Collect requests for this batch
        let batch_size = self.max_batch_size.min(self.pending_requests.len());
        let mut batch_requests = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            if let Some(request) = self.pending_requests.pop_front() {
                batch_requests.push(request);
            }
        }

        if batch_requests.is_empty() {
            return Ok(());
        }

        // Combine tokens into batch tensor
        let combined_tensor = self.combine_tokens_to_batch(&batch_requests)?;

        let active_batch = ActiveBatch {
            requests: batch_requests,
            combined_tensor,
            batch_id: BatchId::new(),
            started_at: Instant::now(),
        };

        self.active_batch = Some(active_batch);
        Ok(())
    }

    fn combine_tokens_to_batch(&self, requests: &[BatchRequest]) -> Result<BitNetTensor> {
        // Find the maximum sequence length in the batch
        let max_seq_len = requests.iter()
            .map(|req| req.tokens.len())
            .max()
            .unwrap_or(0);

        if max_seq_len == 0 {
            return Err(anyhow::anyhow!("Cannot create batch with empty sequences"));
        }

        let batch_size = requests.len();
        let mut batch_data = vec![0usize; batch_size * max_seq_len];

        // Pad sequences and combine into batch
        for (batch_idx, request) in requests.iter().enumerate() {
            let seq_len = request.tokens.len();
            let start_idx = batch_idx * max_seq_len;

            // Copy tokens
            for (token_idx, &token) in request.tokens.iter().enumerate() {
                batch_data[start_idx + token_idx] = token;
            }

            // Pad with special padding token if needed
            for pad_idx in seq_len..max_seq_len {
                batch_data[start_idx + pad_idx] = self.get_padding_token();
            }
        }

        // Convert to f32 tensor (assuming model expects f32 input)
        let float_data: Vec<f32> = batch_data.iter()
            .map(|&token| token as f32)
            .collect();

        BitNetTensor::from_data(
            float_data,
            vec![batch_size, max_seq_len],
        )
    }

    fn get_padding_token(&self) -> usize {
        // Return appropriate padding token ID
        0 // Assuming 0 is padding token, should be configurable
    }
}

impl AutoregressiveGenerator {
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
        // Use the batch processor to handle the request
        if let Some(ref mut batch_processor) = self.batch_processor {
            return batch_processor.process_request(current_tokens.to_vec(), step).await;
        }

        // Fallback to single token generation if batch processor not available
        self.generate_token_single(current_tokens, forward_fn, step).await
    }

    pub async fn process_active_batch<F, Fut>(
        &mut self,
        forward_fn: &F,
    ) -> Result<()>
    where
        F: Fn(BitNetTensor) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<BitNetTensor>> + Send,
    {
        if let Some(ref mut batch_processor) = self.batch_processor {
            if let Some(active_batch) = batch_processor.active_batch.take() {
                // Execute forward pass on the batched tensor
                let batch_output = forward_fn(active_batch.combined_tensor).await?;

                // Extract results for each request in the batch
                let batch_results = self.extract_batch_results(&batch_output, &active_batch)?;

                // Send results back to individual requests
                for (request, result) in active_batch.requests.into_iter().zip(batch_results) {
                    let _ = request.response_sender.send(Ok(result));
                }
            }
        }

        Ok(())
    }

    fn extract_batch_results(
        &self,
        batch_output: &BitNetTensor,
        active_batch: &ActiveBatch,
    ) -> Result<Vec<usize>> {
        let batch_size = active_batch.requests.len();
        let output_data = batch_output.data();

        // Assuming output is [batch_size, vocab_size]
        if batch_output.shape().len() != 2 || batch_output.shape()[0] != batch_size {
            return Err(anyhow::anyhow!(
                "Unexpected batch output shape: {:?}, expected [batch_size={}, vocab_size]",
                batch_output.shape(), batch_size
            ));
        }

        let vocab_size = batch_output.shape()[1];
        let mut results = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let start_idx = batch_idx * vocab_size;
            let end_idx = start_idx + vocab_size;

            if end_idx > output_data.len() {
                return Err(anyhow::anyhow!("Batch output data insufficient for batch index {}", batch_idx));
            }

            // Find the token with the highest probability
            let logits = &output_data[start_idx..end_idx];
            let best_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            results.push(best_token);
        }

        Ok(results)
    }
}
```

## Implementation Plan

### Task 1: Core Batch Processing Infrastructure
- [ ] Implement `BatchProcessor` struct with request queuing
- [ ] Add batch formation logic with size and timeout thresholds
- [ ] Implement tensor combination for batched inference
- [ ] Add request routing and response distribution

### Task 2: Batch Execution Engine
- [ ] Integrate batch processor with autoregressive generator
- [ ] Implement batched forward pass execution
- [ ] Add result extraction and distribution logic
- [ ] Handle variable sequence lengths with padding

### Task 3: Configuration and Optimization
- [ ] Add configurable batch size and timeout parameters
- [ ] Implement adaptive batching based on load
- [ ] Add batch processing metrics and monitoring
- [ ] Optimize memory usage for large batches

### Task 4: Error Handling and Recovery
- [ ] Add comprehensive error handling for batch failures
- [ ] Implement request cancellation and timeout handling
- [ ] Add graceful degradation to single-request processing
- [ ] Handle partial batch failures

## Testing Strategy

### Batch Processing Tests
```rust
#[tokio::test]
async fn test_batch_processor_basic_functionality() {
    let mut batch_processor = BatchProcessor::new(4, 100); // batch_size=4, timeout=100ms

    // Create multiple concurrent requests
    let handles: Vec<_> = (0..4).map(|i| {
        let tokens = vec![i, i + 1, i + 2];
        batch_processor.process_request(tokens, 0)
    }).collect();

    // All requests should complete
    let results: Vec<Result<usize>> = join_all(handles).await;
    for result in results {
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_batch_timeout_behavior() {
    let mut batch_processor = BatchProcessor::new(10, 50); // Large batch size, short timeout

    // Submit single request
    let start = Instant::now();
    let result = batch_processor.process_request(vec![1, 2, 3], 0).await;

    // Should complete within timeout period
    assert!(start.elapsed().as_millis() <= 100);
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_variable_sequence_lengths() {
    let mut batch_processor = BatchProcessor::new(3, 100);

    // Create requests with different sequence lengths
    let handles = vec![
        batch_processor.process_request(vec![1], 0),
        batch_processor.process_request(vec![2, 3, 4, 5], 0),
        batch_processor.process_request(vec![6, 7], 0),
    ];

    let results: Vec<Result<usize>> = join_all(handles).await;
    for result in results {
        assert!(result.is_ok());
    }
}
```

### Performance Benchmarks
```rust
#[tokio::test]
async fn benchmark_batch_vs_single_processing() {
    let num_requests = 100;

    // Benchmark single request processing
    let start = Instant::now();
    for i in 0..num_requests {
        let _ = process_single_request(vec![i, i + 1]).await;
    }
    let single_duration = start.elapsed();

    // Benchmark batch processing
    let start = Instant::now();
    let handles: Vec<_> = (0..num_requests).map(|i| {
        process_batch_request(vec![i, i + 1])
    }).collect();
    let _results: Vec<_> = join_all(handles).await;
    let batch_duration = start.elapsed();

    // Batch processing should be significantly faster
    assert!(batch_duration < single_duration / 2);
}
```

## Related Issues/PRs

- Part of high-performance inference system
- Related to concurrent request handling architecture
- Connected to GPU utilization optimization

## Acceptance Criteria

- [ ] Batch processor collects multiple requests into efficient batches
- [ ] Batched forward passes execute correctly with proper tensor combination
- [ ] Results are correctly distributed back to individual requests
- [ ] Variable sequence lengths are handled with appropriate padding
- [ ] Batch timeout mechanism prevents excessive request latency
- [ ] Performance benchmarks show significant throughput improvements
- [ ] Error handling gracefully manages batch and individual request failures

## Risk Assessment

**Medium-High Risk**: Batch processing is complex and affects core inference performance.

**Mitigation Strategies**:
- Implement comprehensive testing for all batch scenarios
- Add graceful fallback to single-request processing when batching fails
- Implement incremental rollout with feature flags
- Monitor performance metrics to detect regressions
- Provide configuration options to tune batch behavior for different workloads