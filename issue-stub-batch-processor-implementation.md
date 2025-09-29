# [IMPLEMENTATION] Complete BatchProcessor implementation for efficient autoregressive generation

## Problem Description
The `BatchProcessor` in `crates/bitnet-inference/src/generation/autoregressive.rs` is defined but not implemented, with `generate_token_batched` falling back to single token generation instead of true batching.

## Environment
- **File**: `crates/bitnet-inference/src/generation/autoregressive.rs`
- **Component**: BatchProcessor and generate_token_batched method
- **Current State**: Fallback to single token generation

## Root Cause Analysis
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

**Issues:**
1. No actual batch processing implementation
2. Inefficient single-sequence processing
3. Poor GPU utilization for multiple concurrent requests
4. Missing request queuing and batching logic

## Proposed Solution
```rust
pub struct BatchProcessor {
    max_batch_size: usize,
    max_wait_time: Duration,
    pending_requests: VecDeque<BatchRequest>,
    request_sender: mpsc::Sender<BatchRequest>,
    request_receiver: mpsc::Receiver<BatchRequest>,
}

#[derive(Debug)]
struct BatchRequest {
    tokens: Vec<usize>,
    step: usize,
    response_sender: oneshot::Sender<Result<usize>>,
    submitted_at: Instant,
}

impl BatchProcessor {
    pub fn new(max_batch_size: usize, max_wait_time: Duration) -> Self {
        let (sender, receiver) = mpsc::channel(1000);

        Self {
            max_batch_size,
            max_wait_time,
            pending_requests: VecDeque::new(),
            request_sender: sender,
            request_receiver: receiver,
        }
    }

    pub async fn start_processing<F, Fut>(&mut self, forward_fn: F)
    where
        F: Fn(BitNetTensor) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<BitNetTensor>> + Send,
    {
        loop {
            // Collect batch of requests
            let batch = self.collect_batch().await;

            if !batch.is_empty() {
                self.process_batch(batch, &forward_fn).await;
            }
        }
    }

    async fn collect_batch(&mut self) -> Vec<BatchRequest> {
        let mut batch = Vec::new();
        let deadline = Instant::now() + self.max_wait_time;

        // Collect requests until batch is full or timeout
        while batch.len() < self.max_batch_size && Instant::now() < deadline {
            // Try to get a request with timeout
            match timeout(deadline - Instant::now(), self.request_receiver.recv()).await {
                Ok(Some(request)) => batch.push(request),
                Ok(None) => break, // Channel closed
                Err(_) => break,   // Timeout
            }

            // Add any pending requests from previous iterations
            while let Some(pending) = self.pending_requests.pop_front() {
                batch.push(pending);
                if batch.len() >= self.max_batch_size {
                    break;
                }
            }
        }

        batch
    }

    async fn process_batch<F, Fut>(&self, batch: Vec<BatchRequest>, forward_fn: &F)
    where
        F: Fn(BitNetTensor) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<BitNetTensor>> + Send,
    {
        let batch_size = batch.len();

        // Combine all sequences into batch tensor
        let (batch_tensor, sequence_lengths) = self.create_batch_tensor(&batch)?;

        // Single forward pass for entire batch
        let batch_output = match forward_fn(batch_tensor).await {
            Ok(output) => output,
            Err(e) => {
                // Send error to all requests in batch
                for request in batch {
                    let _ = request.response_sender.send(Err(e.clone()));
                }
                return;
            }
        };

        // Distribute results back to individual requests
        self.distribute_batch_results(batch, batch_output, sequence_lengths).await;

        tracing::debug!("Processed batch of {} requests", batch_size);
    }

    fn create_batch_tensor(&self, batch: &[BatchRequest]) -> Result<(BitNetTensor, Vec<usize>)> {
        let max_seq_len = batch.iter().map(|req| req.tokens.len()).max().unwrap_or(0);
        let batch_size = batch.len();
        let vocab_size = 50257; // TODO: Get from model config

        // Create padded batch tensor
        let mut batch_data = vec![0u32; batch_size * max_seq_len];
        let mut sequence_lengths = Vec::with_capacity(batch_size);

        for (batch_idx, request) in batch.iter().enumerate() {
            let seq_len = request.tokens.len();
            sequence_lengths.push(seq_len);

            // Copy tokens to batch tensor
            for (token_idx, &token) in request.tokens.iter().enumerate() {
                batch_data[batch_idx * max_seq_len + token_idx] = token as u32;
            }

            // Pad remaining positions if needed
            for token_idx in seq_len..max_seq_len {
                batch_data[batch_idx * max_seq_len + token_idx] = 0; // Pad token
            }
        }

        let tensor = BitNetTensor::from_data(
            batch_data,
            &[batch_size, max_seq_len],
            DType::U32,
            &Device::Cpu, // TODO: Use appropriate device
        )?;

        Ok((tensor, sequence_lengths))
    }

    async fn distribute_batch_results(
        &self,
        batch: Vec<BatchRequest>,
        batch_output: BitNetTensor,
        sequence_lengths: Vec<usize>,
    ) {
        // Extract logits for each sequence and sample tokens
        for (batch_idx, (request, &seq_len)) in batch.into_iter().zip(sequence_lengths.iter()).enumerate() {
            // Extract logits for this sequence (last token position)
            let logits_result = self.extract_sequence_logits(&batch_output, batch_idx, seq_len - 1);

            let token_result = match logits_result {
                Ok(logits) => {
                    // Sample next token from logits
                    self.sample_token_from_logits(logits)
                }
                Err(e) => Err(e),
            };

            // Send result back to requester
            let _ = request.response_sender.send(token_result);
        }
    }

    fn sample_token_from_logits(&self, logits: Vec<f32>) -> Result<usize> {
        // Simple greedy sampling - could be enhanced with top-k, top-p, etc.
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok(max_idx)
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
        let (response_sender, response_receiver) = oneshot::channel();

        let request = BatchRequest {
            tokens: current_tokens.to_vec(),
            step,
            response_sender,
            submitted_at: Instant::now(),
        };

        // Submit request to batch processor
        self.batch_processor.request_sender.send(request).await
            .map_err(|_| BitNetError::BatchProcessing("Failed to submit batch request".into()))?;

        // Wait for response
        response_receiver.await
            .map_err(|_| BitNetError::BatchProcessing("Failed to receive batch response".into()))?
    }
}
```

## Implementation Plan
### Phase 1: Core Batching Infrastructure (2 days)
- [ ] Implement BatchProcessor struct and request queuing
- [ ] Create batch tensor construction and padding logic
- [ ] Add request-response coordination with channels

### Phase 2: Batch Processing Logic (2 days)
- [ ] Implement batch collection with timeouts
- [ ] Add single forward pass execution for batches
- [ ] Create result distribution back to individual requests

### Phase 3: Integration & Optimization (1 day)
- [ ] Integrate with AutoregressiveGenerator
- [ ] Add performance monitoring and metrics
- [ ] Optimize for different batch sizes and scenarios

## Acceptance Criteria
- [ ] True batch processing of multiple sequences
- [ ] Efficient GPU utilization through batching
- [ ] Configurable batch size and timeout parameters
- [ ] Significant performance improvement for concurrent requests

**Labels**: `implementation`, `performance`, `batching`, `P1-high`
**Effort**: 5 days