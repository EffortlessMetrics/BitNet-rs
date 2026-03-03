//! Cache eviction policies for KV cache management.
//!
//! Implements LRU, FIFO, and priority-based eviction strategies
//! for managing memory-constrained KV caches.

use std::collections::VecDeque;

/// Eviction policy type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used: evict the entry accessed longest ago.
    Lru,
    /// First In First Out: evict the oldest entry.
    Fifo,
    /// Evict the entry with the lowest priority score.
    Priority,
}

/// A tracked cache entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub key: u64,
    pub access_count: u64,
    pub priority: f32,
    insert_order: u64,
    last_access: u64,
}

/// Eviction tracker that determines which entries to evict.
#[derive(Debug)]
pub struct EvictionTracker {
    policy: EvictionPolicy,
    entries: Vec<CacheEntry>,
    capacity: usize,
    clock: u64,
    insert_counter: u64,
    eviction_count: u64,
}

impl EvictionTracker {
    pub fn new(policy: EvictionPolicy, capacity: usize) -> Self {
        Self {
            policy,
            entries: Vec::with_capacity(capacity),
            capacity,
            clock: 0,
            insert_counter: 0,
            eviction_count: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn eviction_count(&self) -> u64 {
        self.eviction_count
    }

    /// Record an access to a key (for LRU tracking).
    pub fn access(&mut self, key: u64) {
        self.clock += 1;
        if let Some(entry) = self.entries.iter_mut().find(|e| e.key == key) {
            entry.last_access = self.clock;
            entry.access_count += 1;
        }
    }

    /// Insert a new entry. Returns the evicted key if cache was full.
    pub fn insert(&mut self, key: u64, priority: f32) -> Option<u64> {
        // Check if key already exists
        if self.entries.iter().any(|e| e.key == key) {
            self.access(key);
            return None;
        }

        let evicted = if self.is_full() {
            let victim = self.select_victim();
            victim.map(|idx| {
                let evicted_key = self.entries[idx].key;
                self.entries.swap_remove(idx);
                self.eviction_count += 1;
                evicted_key
            })
        } else {
            None
        };

        self.clock += 1;
        self.insert_counter += 1;
        self.entries.push(CacheEntry {
            key,
            access_count: 1,
            priority,
            insert_order: self.insert_counter,
            last_access: self.clock,
        });

        evicted
    }

    /// Select the index of the victim to evict.
    fn select_victim(&self) -> Option<usize> {
        if self.entries.is_empty() {
            return None;
        }
        match self.policy {
            EvictionPolicy::Lru => {
                self.entries.iter().enumerate().min_by_key(|(_, e)| e.last_access).map(|(i, _)| i)
            }
            EvictionPolicy::Fifo => {
                self.entries.iter().enumerate().min_by_key(|(_, e)| e.insert_order).map(|(i, _)| i)
            }
            EvictionPolicy::Priority => self
                .entries
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.priority.partial_cmp(&b.priority).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i),
        }
    }

    /// Remove a specific key.
    pub fn remove(&mut self, key: u64) -> bool {
        if let Some(pos) = self.entries.iter().position(|e| e.key == key) {
            self.entries.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Check whether a key is present.
    pub fn contains(&self, key: u64) -> bool {
        self.entries.iter().any(|e| e.key == key)
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.clock = 0;
        self.insert_counter = 0;
    }

    /// Get keys in eviction order (first = next to be evicted).
    pub fn eviction_order(&self) -> Vec<u64> {
        let mut indices: Vec<usize> = (0..self.entries.len()).collect();
        match self.policy {
            EvictionPolicy::Lru => {
                indices.sort_by_key(|&i| self.entries[i].last_access);
            }
            EvictionPolicy::Fifo => {
                indices.sort_by_key(|&i| self.entries[i].insert_order);
            }
            EvictionPolicy::Priority => {
                indices.sort_by(|&a, &b| {
                    self.entries[a]
                        .priority
                        .partial_cmp(&self.entries[b].priority)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
        indices.iter().map(|&i| self.entries[i].key).collect()
    }
}

/// Simple bounded FIFO queue for sequence IDs.
#[derive(Debug)]
pub struct EvictionQueue {
    queue: VecDeque<u64>,
    max_size: usize,
}

impl EvictionQueue {
    pub fn new(max_size: usize) -> Self {
        Self { queue: VecDeque::with_capacity(max_size), max_size }
    }

    pub fn push(&mut self, id: u64) -> Option<u64> {
        if self.queue.len() >= self.max_size {
            let evicted = self.queue.pop_front();
            self.queue.push_back(id);
            evicted
        } else {
            self.queue.push_back(id);
            None
        }
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_eviction() {
        let mut t = EvictionTracker::new(EvictionPolicy::Lru, 3);
        t.insert(1, 0.0);
        t.insert(2, 0.0);
        t.insert(3, 0.0);
        t.access(1); // 1 is most recent now
        let evicted = t.insert(4, 0.0);
        assert_eq!(evicted, Some(2)); // 2 was least recently used
    }

    #[test]
    fn test_fifo_eviction() {
        let mut t = EvictionTracker::new(EvictionPolicy::Fifo, 3);
        t.insert(1, 0.0);
        t.insert(2, 0.0);
        t.insert(3, 0.0);
        t.access(1); // access doesn't matter for FIFO
        let evicted = t.insert(4, 0.0);
        assert_eq!(evicted, Some(1)); // 1 was first in
    }

    #[test]
    fn test_priority_eviction() {
        let mut t = EvictionTracker::new(EvictionPolicy::Priority, 3);
        t.insert(1, 5.0);
        t.insert(2, 1.0); // lowest priority
        t.insert(3, 3.0);
        let evicted = t.insert(4, 10.0);
        assert_eq!(evicted, Some(2)); // lowest priority evicted
    }

    #[test]
    fn test_no_eviction_under_capacity() {
        let mut t = EvictionTracker::new(EvictionPolicy::Lru, 5);
        assert_eq!(t.insert(1, 0.0), None);
        assert_eq!(t.insert(2, 0.0), None);
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn test_duplicate_insert() {
        let mut t = EvictionTracker::new(EvictionPolicy::Lru, 3);
        t.insert(1, 0.0);
        t.insert(1, 0.0); // duplicate - just access
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn test_remove() {
        let mut t = EvictionTracker::new(EvictionPolicy::Lru, 5);
        t.insert(1, 0.0);
        t.insert(2, 0.0);
        assert!(t.remove(1));
        assert!(!t.remove(99));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn test_contains() {
        let mut t = EvictionTracker::new(EvictionPolicy::Lru, 5);
        t.insert(42, 0.0);
        assert!(t.contains(42));
        assert!(!t.contains(99));
    }

    #[test]
    fn test_eviction_count() {
        let mut t = EvictionTracker::new(EvictionPolicy::Fifo, 2);
        t.insert(1, 0.0);
        t.insert(2, 0.0);
        t.insert(3, 0.0);
        t.insert(4, 0.0);
        assert_eq!(t.eviction_count(), 2);
    }

    #[test]
    fn test_clear() {
        let mut t = EvictionTracker::new(EvictionPolicy::Lru, 5);
        t.insert(1, 0.0);
        t.insert(2, 0.0);
        t.clear();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn test_eviction_order() {
        let mut t = EvictionTracker::new(EvictionPolicy::Priority, 5);
        t.insert(1, 5.0);
        t.insert(2, 1.0);
        t.insert(3, 3.0);
        let order = t.eviction_order();
        assert_eq!(order[0], 2); // lowest priority first
    }

    #[test]
    fn test_eviction_queue() {
        let mut q = EvictionQueue::new(3);
        assert_eq!(q.push(1), None);
        assert_eq!(q.push(2), None);
        assert_eq!(q.push(3), None);
        assert_eq!(q.push(4), Some(1)); // evicts oldest
        assert_eq!(q.len(), 3);
    }

    #[test]
    fn test_is_full() {
        let mut t = EvictionTracker::new(EvictionPolicy::Lru, 2);
        assert!(!t.is_full());
        t.insert(1, 0.0);
        t.insert(2, 0.0);
        assert!(t.is_full());
    }
}
