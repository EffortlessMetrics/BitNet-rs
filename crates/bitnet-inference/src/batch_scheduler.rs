//! Batch inference scheduler.
//!
//! Groups and schedules inference requests for efficient processing.

use std::collections::VecDeque;

/// Priority level for batch items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// A request to be scheduled.
#[derive(Debug, Clone)]
pub struct BatchItem {
    pub id: u64,
    pub priority: Priority,
    pub token_count: usize,
    pub max_tokens: usize,
}

/// Scheduling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleStrategy {
    Fifo,
    PriorityFirst,
    ShortestFirst,
}

/// A batch of items to process together.
#[derive(Debug, Clone)]
pub struct Batch {
    pub items: Vec<BatchItem>,
    pub total_input_tokens: usize,
    pub total_max_output: usize,
}

impl Batch {
    pub fn new() -> Self {
        Self { items: Vec::new(), total_input_tokens: 0, total_max_output: 0 }
    }

    pub fn add(&mut self, item: BatchItem) {
        self.total_input_tokens += item.token_count;
        self.total_max_output += item.max_tokens;
        self.items.push(item);
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl Default for Batch {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch scheduler.
#[derive(Debug)]
pub struct BatchScheduler {
    queue: VecDeque<BatchItem>,
    strategy: ScheduleStrategy,
    max_batch_size: usize,
    max_batch_tokens: usize,
    next_id: u64,
}

impl BatchScheduler {
    pub fn new(strategy: ScheduleStrategy, max_batch_size: usize, max_batch_tokens: usize) -> Self {
        Self { queue: VecDeque::new(), strategy, max_batch_size, max_batch_tokens, next_id: 1 }
    }

    pub fn submit(&mut self, token_count: usize, max_tokens: usize, priority: Priority) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.queue.push_back(BatchItem { id, priority, token_count, max_tokens });
        id
    }

    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    /// Form the next batch from the queue.
    pub fn next_batch(&mut self) -> Option<Batch> {
        if self.queue.is_empty() {
            return None;
        }

        // Sort queue based on strategy
        let mut items: Vec<BatchItem> = self.queue.drain(..).collect();
        match self.strategy {
            ScheduleStrategy::PriorityFirst => {
                items.sort_by(|a, b| b.priority.cmp(&a.priority));
            }
            ScheduleStrategy::ShortestFirst => items.sort_by_key(|i| i.token_count),
            ScheduleStrategy::Fifo => {} // already in order
        }

        let mut batch = Batch::new();
        let mut remaining = VecDeque::new();

        for item in items {
            if batch.len() < self.max_batch_size
                && batch.total_input_tokens + item.token_count <= self.max_batch_tokens
            {
                batch.add(item);
            } else {
                remaining.push_back(item);
            }
        }

        self.queue = remaining;

        if batch.is_empty() { None } else { Some(batch) }
    }

    pub fn cancel(&mut self, id: u64) -> bool {
        if let Some(pos) = self.queue.iter().position(|i| i.id == id) {
            self.queue.remove(pos);
            true
        } else {
            false
        }
    }

    pub fn clear(&mut self) -> usize {
        let count = self.queue.len();
        self.queue.clear();
        count
    }
}

impl Default for BatchScheduler {
    fn default() -> Self {
        Self::new(ScheduleStrategy::Fifo, 32, 8192)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_submit() {
        let mut s = BatchScheduler::default();
        let id = s.submit(10, 50, Priority::Normal);
        assert_eq!(id, 1);
        assert_eq!(s.pending(), 1);
    }

    #[test]
    fn test_fifo_batch() {
        let mut s = BatchScheduler::new(ScheduleStrategy::Fifo, 2, 1000);
        s.submit(10, 50, Priority::Normal);
        s.submit(20, 50, Priority::Normal);
        s.submit(30, 50, Priority::Normal);
        let batch = s.next_batch().unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(s.pending(), 1);
    }

    #[test]
    fn test_priority_batch() {
        let mut s = BatchScheduler::new(ScheduleStrategy::PriorityFirst, 2, 1000);
        s.submit(10, 50, Priority::Low);
        s.submit(10, 50, Priority::Critical);
        s.submit(10, 50, Priority::High);
        let batch = s.next_batch().unwrap();
        assert_eq!(batch.items[0].priority, Priority::Critical);
        assert_eq!(batch.items[1].priority, Priority::High);
    }

    #[test]
    fn test_shortest_first() {
        let mut s = BatchScheduler::new(ScheduleStrategy::ShortestFirst, 2, 1000);
        s.submit(100, 50, Priority::Normal);
        s.submit(5, 50, Priority::Normal);
        s.submit(50, 50, Priority::Normal);
        let batch = s.next_batch().unwrap();
        assert_eq!(batch.items[0].token_count, 5);
    }

    #[test]
    fn test_token_limit() {
        let mut s = BatchScheduler::new(ScheduleStrategy::Fifo, 10, 30);
        s.submit(15, 50, Priority::Normal);
        s.submit(15, 50, Priority::Normal);
        s.submit(15, 50, Priority::Normal);
        let batch = s.next_batch().unwrap();
        assert_eq!(batch.len(), 2); // only 2 fit in 30 tokens
        assert_eq!(s.pending(), 1);
    }

    #[test]
    fn test_empty_batch() {
        let mut s = BatchScheduler::default();
        assert!(s.next_batch().is_none());
    }

    #[test]
    fn test_cancel() {
        let mut s = BatchScheduler::default();
        let id = s.submit(10, 50, Priority::Normal);
        assert!(s.cancel(id));
        assert_eq!(s.pending(), 0);
        assert!(!s.cancel(999));
    }

    #[test]
    fn test_clear() {
        let mut s = BatchScheduler::default();
        s.submit(10, 50, Priority::Normal);
        s.submit(20, 50, Priority::Normal);
        assert_eq!(s.clear(), 2);
        assert_eq!(s.pending(), 0);
    }

    #[test]
    fn test_batch_totals() {
        let mut b = Batch::new();
        b.add(BatchItem { id: 1, priority: Priority::Normal, token_count: 10, max_tokens: 50 });
        b.add(BatchItem { id: 2, priority: Priority::Normal, token_count: 20, max_tokens: 100 });
        assert_eq!(b.total_input_tokens, 30);
        assert_eq!(b.total_max_output, 150);
    }

    #[test]
    fn test_default_scheduler() {
        let s = BatchScheduler::default();
        assert_eq!(s.pending(), 0);
    }

    #[test]
    fn test_id_increment() {
        let mut s = BatchScheduler::default();
        let id1 = s.submit(10, 50, Priority::Normal);
        let id2 = s.submit(10, 50, Priority::Normal);
        assert_eq!(id2, id1 + 1);
    }

    #[test]
    fn test_multiple_batches() {
        let mut s = BatchScheduler::new(ScheduleStrategy::Fifo, 1, 1000);
        s.submit(10, 50, Priority::Normal);
        s.submit(20, 50, Priority::Normal);
        let b1 = s.next_batch().unwrap();
        let b2 = s.next_batch().unwrap();
        assert_eq!(b1.len(), 1);
        assert_eq!(b2.len(), 1);
        assert!(s.next_batch().is_none());
    }
}
