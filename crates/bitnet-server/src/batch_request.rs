//! Batch inference request builder.
//!
//! Assemble, validate, and prioritize batch inference requests
//! for efficient server-side processing.

use std::time::{Duration, Instant};

/// Priority level for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Low,
    Normal,
    High,
    Critical,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

/// A single inference request in a batch.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub id: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub priority: Priority,
    pub created_at: Instant,
}

impl InferenceRequest {
    pub fn new(id: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            prompt: prompt.into(),
            max_tokens: 256,
            temperature: 1.0,
            priority: Priority::Normal,
            created_at: Instant::now(),
        }
    }

    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    pub fn with_priority(mut self, p: Priority) -> Self {
        self.priority = p;
        self
    }

    pub fn prompt_len(&self) -> usize {
        self.prompt.len()
    }

    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Batch processing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchStrategy {
    /// Process in FIFO order.
    Fifo,
    /// Process highest priority first.
    PriorityFirst,
    /// Group by similar prompt lengths for efficiency.
    LengthBucketed,
}

/// Assembled batch ready for processing.
#[derive(Debug)]
pub struct RequestBatch {
    pub requests: Vec<InferenceRequest>,
    pub strategy: BatchStrategy,
    pub max_batch_size: usize,
}

impl RequestBatch {
    pub fn new(strategy: BatchStrategy, max_batch_size: usize) -> Self {
        Self { requests: Vec::new(), strategy, max_batch_size }
    }

    pub fn add(&mut self, req: InferenceRequest) -> bool {
        if self.requests.len() >= self.max_batch_size {
            return false;
        }
        self.requests.push(req);
        true
    }

    pub fn is_full(&self) -> bool {
        self.requests.len() >= self.max_batch_size
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Sort the batch according to the strategy.
    pub fn sort(&mut self) {
        match self.strategy {
            BatchStrategy::Fifo => {} // already in order
            BatchStrategy::PriorityFirst => {
                self.requests.sort_by(|a, b| b.priority.cmp(&a.priority));
            }
            BatchStrategy::LengthBucketed => {
                self.requests.sort_by_key(|r| r.prompt_len());
            }
        }
    }

    pub fn total_prompt_tokens_est(&self) -> usize {
        // Rough estimate: 4 chars per token
        self.requests.iter().map(|r| r.prompt_len() / 4 + 1).sum()
    }

    pub fn total_max_output_tokens(&self) -> usize {
        self.requests.iter().map(|r| r.max_tokens).sum()
    }

    pub fn max_prompt_len(&self) -> usize {
        self.requests.iter().map(|r| r.prompt_len()).max().unwrap_or(0)
    }

    pub fn drain_batch(&mut self) -> Vec<InferenceRequest> {
        std::mem::take(&mut self.requests)
    }
}

/// Builder for constructing batches.
#[derive(Debug)]
pub struct BatchBuilder {
    pending: Vec<InferenceRequest>,
    strategy: BatchStrategy,
    max_batch_size: usize,
    max_total_tokens: Option<usize>,
}

impl BatchBuilder {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            strategy: BatchStrategy::Fifo,
            max_batch_size: 32,
            max_total_tokens: None,
        }
    }

    pub fn with_strategy(mut self, s: BatchStrategy) -> Self {
        self.strategy = s;
        self
    }

    pub fn with_max_batch_size(mut self, n: usize) -> Self {
        self.max_batch_size = n;
        self
    }

    pub fn with_max_total_tokens(mut self, n: usize) -> Self {
        self.max_total_tokens = Some(n);
        self
    }

    pub fn enqueue(&mut self, req: InferenceRequest) {
        self.pending.push(req);
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Build the next batch from pending requests.
    pub fn build_next(&mut self) -> Option<RequestBatch> {
        if self.pending.is_empty() {
            return None;
        }

        let mut batch = RequestBatch::new(self.strategy, self.max_batch_size);
        let mut token_budget = self.max_total_tokens.unwrap_or(usize::MAX);

        // Sort pending by priority for PriorityFirst
        if self.strategy == BatchStrategy::PriorityFirst {
            self.pending.sort_by(|a, b| b.priority.cmp(&a.priority));
        }

        let mut remaining = Vec::new();
        for req in self.pending.drain(..) {
            let est = req.prompt_len() / 4 + 1 + req.max_tokens;
            if batch.len() < self.max_batch_size && est <= token_budget {
                token_budget = token_budget.saturating_sub(est);
                batch.requests.push(req);
            } else {
                remaining.push(req);
            }
        }
        self.pending = remaining;

        batch.sort();
        if batch.is_empty() { None } else { Some(batch) }
    }
}

impl Default for BatchBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_creation() {
        let req = InferenceRequest::new("r1", "Hello world");
        assert_eq!(req.id, "r1");
        assert_eq!(req.max_tokens, 256);
    }

    #[test]
    fn test_request_builder() {
        let req = InferenceRequest::new("r1", "test")
            .with_max_tokens(100)
            .with_temperature(0.7)
            .with_priority(Priority::High);
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.priority, Priority::High);
    }

    #[test]
    fn test_batch_add() {
        let mut batch = RequestBatch::new(BatchStrategy::Fifo, 2);
        assert!(batch.add(InferenceRequest::new("1", "a")));
        assert!(batch.add(InferenceRequest::new("2", "b")));
        assert!(!batch.add(InferenceRequest::new("3", "c")));
        assert!(batch.is_full());
    }

    #[test]
    fn test_priority_sort() {
        let mut batch = RequestBatch::new(BatchStrategy::PriorityFirst, 10);
        batch.add(InferenceRequest::new("low", "a").with_priority(Priority::Low));
        batch.add(InferenceRequest::new("high", "b").with_priority(Priority::High));
        batch.sort();
        assert_eq!(batch.requests[0].id, "high");
    }

    #[test]
    fn test_length_bucket_sort() {
        let mut batch = RequestBatch::new(BatchStrategy::LengthBucketed, 10);
        batch.add(InferenceRequest::new("long", "hello world this is long"));
        batch.add(InferenceRequest::new("short", "hi"));
        batch.sort();
        assert_eq!(batch.requests[0].id, "short");
    }

    #[test]
    fn test_batch_stats() {
        let mut batch = RequestBatch::new(BatchStrategy::Fifo, 10);
        batch.add(InferenceRequest::new("1", "hello").with_max_tokens(10));
        batch.add(InferenceRequest::new("2", "world foo").with_max_tokens(20));
        assert_eq!(batch.total_max_output_tokens(), 30);
        assert!(batch.max_prompt_len() > 0);
    }

    #[test]
    fn test_drain_batch() {
        let mut batch = RequestBatch::new(BatchStrategy::Fifo, 10);
        batch.add(InferenceRequest::new("1", "a"));
        let drained = batch.drain_batch();
        assert_eq!(drained.len(), 1);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_builder_basic() {
        let mut builder = BatchBuilder::new().with_max_batch_size(2);
        builder.enqueue(InferenceRequest::new("1", "a"));
        builder.enqueue(InferenceRequest::new("2", "b"));
        builder.enqueue(InferenceRequest::new("3", "c"));
        let batch = builder.build_next().unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(builder.pending_count(), 1);
    }

    #[test]
    fn test_builder_priority() {
        let mut builder =
            BatchBuilder::new().with_strategy(BatchStrategy::PriorityFirst).with_max_batch_size(1);
        builder.enqueue(InferenceRequest::new("low", "a").with_priority(Priority::Low));
        builder.enqueue(InferenceRequest::new("high", "b").with_priority(Priority::High));
        let batch = builder.build_next().unwrap();
        assert_eq!(batch.requests[0].id, "high");
    }

    #[test]
    fn test_builder_token_budget() {
        let mut builder = BatchBuilder::new().with_max_total_tokens(50);
        builder.enqueue(InferenceRequest::new("1", "short").with_max_tokens(10));
        builder.enqueue(InferenceRequest::new("2", "short").with_max_tokens(10));
        builder.enqueue(
            InferenceRequest::new("3", "a very long prompt that takes many tokens")
                .with_max_tokens(500),
        );
        let batch = builder.build_next().unwrap();
        // Should include the small ones, skip the huge one
        assert!(batch.len() >= 2);
    }

    #[test]
    fn test_builder_empty() {
        let mut builder = BatchBuilder::new();
        assert!(builder.build_next().is_none());
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }
}
