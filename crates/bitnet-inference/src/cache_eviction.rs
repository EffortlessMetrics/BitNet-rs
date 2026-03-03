//! KV cache eviction policies.
//!
//! Manage cache memory under pressure: LRU, sliding window,
//! attention-score-based eviction, and budget management.

use std::collections::VecDeque;

/// Eviction policy type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used.
    Lru,
    /// Fixed sliding window — keep most recent N positions.
    SlidingWindow,
    /// Evict positions with lowest cumulative attention scores.
    AttentionBased,
    /// No eviction — fail when full.
    NoEviction,
}

/// A cache entry tracking metadata.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub position: usize,
    pub last_accessed: u64,
    pub attention_score: f64,
}

/// Cache budget constraints.
#[derive(Debug, Clone)]
pub struct CacheBudget {
    pub max_entries: usize,
    pub current_entries: usize,
}

impl CacheBudget {
    pub fn new(max_entries: usize) -> Self {
        Self { max_entries, current_entries: 0 }
    }

    pub fn utilization(&self) -> f64 {
        if self.max_entries == 0 {
            return 1.0;
        }
        self.current_entries as f64 / self.max_entries as f64
    }

    pub fn is_full(&self) -> bool {
        self.current_entries >= self.max_entries
    }

    pub fn remaining(&self) -> usize {
        self.max_entries.saturating_sub(self.current_entries)
    }
}

/// LRU eviction tracker.
#[derive(Debug)]
pub struct LruTracker {
    order: VecDeque<usize>,
    max_size: usize,
}

impl LruTracker {
    pub fn new(max_size: usize) -> Self {
        Self { order: VecDeque::new(), max_size }
    }

    pub fn access(&mut self, position: usize) {
        self.order.retain(|&p| p != position);
        self.order.push_back(position);
    }

    /// Returns positions to evict (oldest first) to free `count` slots.
    pub fn evict(&mut self, count: usize) -> Vec<usize> {
        let n = count.min(self.order.len());
        let evicted: Vec<usize> = self.order.drain(..n).collect();
        evicted
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_full(&self) -> bool {
        self.order.len() >= self.max_size
    }

    pub fn should_evict(&self) -> bool {
        self.order.len() > self.max_size
    }
}

/// Sliding window eviction.
#[derive(Debug)]
pub struct SlidingWindowTracker {
    window_size: usize,
    current_pos: usize,
}

impl SlidingWindowTracker {
    pub fn new(window_size: usize) -> Self {
        Self { window_size, current_pos: 0 }
    }

    pub fn advance(&mut self) {
        self.current_pos += 1;
    }

    pub fn advance_to(&mut self, pos: usize) {
        self.current_pos = pos;
    }

    /// Range of positions to keep: [start, end).
    pub fn active_range(&self) -> (usize, usize) {
        let start = self.current_pos.saturating_sub(self.window_size);
        (start, self.current_pos)
    }

    /// Positions that should be evicted (before the window start).
    pub fn evictable_before(&self) -> usize {
        self.current_pos.saturating_sub(self.window_size)
    }
}

/// Attention-score-based eviction.
pub fn evict_by_attention(entries: &mut Vec<CacheEntry>, count: usize) -> Vec<usize> {
    if count == 0 || entries.is_empty() {
        return vec![];
    }
    entries.sort_by(|a, b| a.attention_score.partial_cmp(&b.attention_score).unwrap());
    let n = count.min(entries.len());
    let evicted: Vec<usize> = entries.drain(..n).map(|e| e.position).collect();
    evicted
}

/// Select eviction targets based on policy.
pub fn select_evictions(
    policy: EvictionPolicy,
    entries: &[CacheEntry],
    count: usize,
) -> Vec<usize> {
    if count == 0 || entries.is_empty() {
        return vec![];
    }
    let n = count.min(entries.len());

    match policy {
        EvictionPolicy::Lru => {
            let mut sorted = entries.to_vec();
            sorted.sort_by_key(|e| e.last_accessed);
            sorted[..n].iter().map(|e| e.position).collect()
        }
        EvictionPolicy::AttentionBased => {
            let mut sorted = entries.to_vec();
            sorted.sort_by(|a, b| a.attention_score.partial_cmp(&b.attention_score).unwrap());
            sorted[..n].iter().map(|e| e.position).collect()
        }
        EvictionPolicy::SlidingWindow => {
            // Evict oldest positions
            let mut sorted = entries.to_vec();
            sorted.sort_by_key(|e| e.position);
            sorted[..n].iter().map(|e| e.position).collect()
        }
        EvictionPolicy::NoEviction => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_budget() {
        let mut budget = CacheBudget::new(100);
        assert!(!budget.is_full());
        assert_eq!(budget.remaining(), 100);
        budget.current_entries = 100;
        assert!(budget.is_full());
        assert!((budget.utilization() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lru_tracker() {
        let mut lru = LruTracker::new(3);
        lru.access(0);
        lru.access(1);
        lru.access(2);
        assert_eq!(lru.len(), 3);
        lru.access(0); // refresh 0
        let evicted = lru.evict(1);
        assert_eq!(evicted, vec![1]); // 1 is oldest now
    }

    #[test]
    fn test_lru_evict_multiple() {
        let mut lru = LruTracker::new(5);
        for i in 0..5 {
            lru.access(i);
        }
        let evicted = lru.evict(3);
        assert_eq!(evicted, vec![0, 1, 2]);
        assert_eq!(lru.len(), 2);
    }

    #[test]
    fn test_sliding_window() {
        let mut sw = SlidingWindowTracker::new(4);
        sw.advance_to(10);
        let (start, end) = sw.active_range();
        assert_eq!(start, 6);
        assert_eq!(end, 10);
    }

    #[test]
    fn test_sliding_window_early() {
        let mut sw = SlidingWindowTracker::new(10);
        sw.advance_to(3);
        let (start, _) = sw.active_range();
        assert_eq!(start, 0); // can't go negative
    }

    #[test]
    fn test_attention_eviction() {
        let mut entries = vec![
            CacheEntry { position: 0, last_accessed: 10, attention_score: 0.9 },
            CacheEntry { position: 1, last_accessed: 5, attention_score: 0.1 },
            CacheEntry { position: 2, last_accessed: 8, attention_score: 0.5 },
        ];
        let evicted = evict_by_attention(&mut entries, 1);
        assert_eq!(evicted, vec![1]); // lowest attention score
    }

    #[test]
    fn test_select_lru() {
        let entries = vec![
            CacheEntry { position: 0, last_accessed: 10, attention_score: 0.5 },
            CacheEntry { position: 1, last_accessed: 1, attention_score: 0.5 },
            CacheEntry { position: 2, last_accessed: 5, attention_score: 0.5 },
        ];
        let evicted = select_evictions(EvictionPolicy::Lru, &entries, 1);
        assert_eq!(evicted, vec![1]); // oldest access
    }

    #[test]
    fn test_select_no_eviction() {
        let entries = vec![CacheEntry { position: 0, last_accessed: 1, attention_score: 0.5 }];
        let evicted = select_evictions(EvictionPolicy::NoEviction, &entries, 1);
        assert!(evicted.is_empty());
    }

    #[test]
    fn test_lru_is_full() {
        let mut lru = LruTracker::new(2);
        lru.access(0);
        assert!(!lru.is_full());
        lru.access(1);
        assert!(lru.is_full());
    }

    #[test]
    fn test_budget_zero() {
        let budget = CacheBudget::new(0);
        assert!(budget.is_full());
        assert!((budget.utilization() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_eviction() {
        let evicted = select_evictions(EvictionPolicy::Lru, &[], 5);
        assert!(evicted.is_empty());
    }

    #[test]
    fn test_evictable_before() {
        let mut sw = SlidingWindowTracker::new(100);
        sw.advance_to(150);
        assert_eq!(sw.evictable_before(), 50);
    }
}
