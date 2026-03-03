//! Batch inference coordination.
//!
//! Manage batched request scheduling: request queuing, batch formation,
//! priority ordering, and batch execution tracking.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Priority level for inference requests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    #[default]
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// An inference request in the batch queue.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    pub id: u64,
    pub token_ids: Vec<u32>,
    pub max_tokens: usize,
    pub priority: Priority,
    pub enqueued_at: Instant,
}

impl BatchRequest {
    pub fn new(id: u64, token_ids: Vec<u32>, max_tokens: usize) -> Self {
        Self { id, token_ids, max_tokens, priority: Priority::Normal, enqueued_at: Instant::now() }
    }

    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    pub fn wait_time(&self) -> Duration {
        self.enqueued_at.elapsed()
    }

    pub fn input_len(&self) -> usize {
        self.token_ids.len()
    }
}

/// Configuration for batch formation.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub max_total_tokens: usize,
    pub max_wait: Duration,
    pub padding_strategy: PaddingStrategy,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_total_tokens: 4096,
            max_wait: Duration::from_millis(50),
            padding_strategy: PaddingStrategy::MaxInBatch,
        }
    }
}

/// How to pad sequences within a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// Pad to the longest sequence in the batch.
    MaxInBatch,
    /// Pad to a fixed length.
    Fixed(usize),
    /// No padding (dynamic batching).
    None,
}

/// A formed batch ready for execution.
#[derive(Debug)]
pub struct Batch {
    pub requests: Vec<BatchRequest>,
    pub formed_at: Instant,
}

impl Batch {
    pub fn size(&self) -> usize {
        self.requests.len()
    }

    pub fn total_tokens(&self) -> usize {
        self.requests.iter().map(|r| r.input_len()).sum()
    }

    pub fn max_input_len(&self) -> usize {
        self.requests.iter().map(|r| r.input_len()).max().unwrap_or(0)
    }

    pub fn padded_tokens(&self, strategy: PaddingStrategy) -> usize {
        match strategy {
            PaddingStrategy::MaxInBatch => self.max_input_len() * self.size(),
            PaddingStrategy::Fixed(len) => len * self.size(),
            PaddingStrategy::None => self.total_tokens(),
        }
    }
}

/// Batch coordinator: queues requests and forms batches.
#[derive(Debug)]
pub struct BatchCoordinator {
    queue: VecDeque<BatchRequest>,
    config: BatchConfig,
    next_id: u64,
    batches_formed: u64,
}

impl BatchCoordinator {
    pub fn new(config: BatchConfig) -> Self {
        Self { queue: VecDeque::new(), config, next_id: 0, batches_formed: 0 }
    }

    pub fn with_defaults() -> Self {
        Self::new(BatchConfig::default())
    }

    /// Enqueue a request and return its assigned ID.
    pub fn enqueue(&mut self, token_ids: Vec<u32>, max_tokens: usize) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.queue.push_back(BatchRequest::new(id, token_ids, max_tokens));
        id
    }

    /// Enqueue with priority.
    pub fn enqueue_priority(
        &mut self,
        token_ids: Vec<u32>,
        max_tokens: usize,
        priority: Priority,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let req = BatchRequest::new(id, token_ids, max_tokens).with_priority(priority);
        // Insert based on priority: higher priority goes earlier
        let pos = self.queue.iter().position(|r| r.priority < priority).unwrap_or(self.queue.len());
        self.queue.insert(pos, req);
        id
    }

    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Try to form a batch from queued requests.
    pub fn form_batch(&mut self) -> Option<Batch> {
        if self.queue.is_empty() {
            return None;
        }

        let mut requests = Vec::new();
        let mut total_tokens = 0;

        while let Some(req) = self.queue.front() {
            if requests.len() >= self.config.max_batch_size {
                break;
            }
            let new_total = total_tokens + req.input_len();
            if new_total > self.config.max_total_tokens && !requests.is_empty() {
                break;
            }
            total_tokens = new_total;
            requests.push(self.queue.pop_front().unwrap());
        }

        if requests.is_empty() {
            return None;
        }

        self.batches_formed += 1;
        Some(Batch { requests, formed_at: Instant::now() })
    }

    pub fn batches_formed(&self) -> u64 {
        self.batches_formed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_enqueue_and_form_batch() {
        let mut coord = BatchCoordinator::with_defaults();
        coord.enqueue(vec![1, 2, 3], 10);
        coord.enqueue(vec![4, 5], 10);
        let batch = coord.form_batch().unwrap();
        assert_eq!(batch.size(), 2);
        assert_eq!(batch.total_tokens(), 5);
    }

    #[test]
    fn test_max_batch_size() {
        let config = BatchConfig { max_batch_size: 2, ..Default::default() };
        let mut coord = BatchCoordinator::new(config);
        coord.enqueue(vec![1], 10);
        coord.enqueue(vec![2], 10);
        coord.enqueue(vec![3], 10);
        let batch = coord.form_batch().unwrap();
        assert_eq!(batch.size(), 2);
        assert_eq!(coord.queue_len(), 1);
    }

    #[test]
    fn test_max_total_tokens() {
        let config = BatchConfig { max_batch_size: 100, max_total_tokens: 5, ..Default::default() };
        let mut coord = BatchCoordinator::new(config);
        coord.enqueue(vec![1, 2, 3], 10);
        coord.enqueue(vec![4, 5, 6], 10);
        let batch = coord.form_batch().unwrap();
        assert_eq!(batch.size(), 1); // second would exceed 5 tokens
    }

    #[test]
    fn test_empty_queue() {
        let mut coord = BatchCoordinator::with_defaults();
        assert!(coord.form_batch().is_none());
        assert!(coord.is_empty());
    }

    #[test]
    fn test_priority_enqueue() {
        let mut coord = BatchCoordinator::with_defaults();
        coord.enqueue(vec![1], 10); // normal
        coord.enqueue(vec![2], 10); // normal
        coord.enqueue_priority(vec![99], 10, Priority::Critical);
        let batch = coord.form_batch().unwrap();
        assert_eq!(batch.requests[0].token_ids, vec![99]);
    }

    #[test]
    fn test_batch_padded_tokens() {
        let mut coord = BatchCoordinator::with_defaults();
        coord.enqueue(vec![1, 2], 10);
        coord.enqueue(vec![3, 4, 5, 6], 10);
        let batch = coord.form_batch().unwrap();
        assert_eq!(batch.padded_tokens(PaddingStrategy::MaxInBatch), 8); // 4 * 2
        assert_eq!(batch.padded_tokens(PaddingStrategy::Fixed(10)), 20); // 10 * 2
        assert_eq!(batch.padded_tokens(PaddingStrategy::None), 6);
    }

    #[test]
    fn test_batches_formed_counter() {
        let mut coord = BatchCoordinator::with_defaults();
        coord.enqueue(vec![1], 10);
        coord.form_batch();
        coord.enqueue(vec![2], 10);
        coord.form_batch();
        assert_eq!(coord.batches_formed(), 2);
    }

    #[test]
    fn test_request_wait_time() {
        let req = BatchRequest::new(0, vec![1], 10);
        std::thread::sleep(Duration::from_millis(10));
        assert!(req.wait_time() >= Duration::from_millis(5));
    }

    #[test]
    fn test_default_config() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.max_total_tokens, 4096);
    }

    #[test]
    fn test_sequential_ids() {
        let mut coord = BatchCoordinator::with_defaults();
        let id1 = coord.enqueue(vec![1], 10);
        let id2 = coord.enqueue(vec![2], 10);
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
    }

    #[test]
    fn test_batch_max_input_len() {
        let mut coord = BatchCoordinator::with_defaults();
        coord.enqueue(vec![1, 2], 10);
        coord.enqueue(vec![3, 4, 5], 10);
        let batch = coord.form_batch().unwrap();
        assert_eq!(batch.max_input_len(), 3);
    }
}
