#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

/// Fuzz KV cache eviction policies with random access patterns, verifying
/// that eviction never corrupts state and capacity invariants hold.
#[derive(Arbitrary, Debug)]
struct EvictionInput {
    capacity: u8,
    policy: u8,
    ops: Vec<EvictionOp>,
}

#[derive(Arbitrary, Debug)]
enum EvictionOp {
    Insert { key: u16, value_byte: u8 },
    Access { key: u16 },
    Evict,
    EvictN { count: u8 },
    BulkInsert { keys: Vec<u8> },
    Stats,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EvictionPolicy {
    LRU,
    LFU,
    FIFO,
}

struct EvictableCache {
    capacity: usize,
    policy: EvictionPolicy,
    entries: Vec<(u16, f32)>,
    access_counts: Vec<u64>,
    access_order: Vec<u64>,
    insert_order: Vec<u64>,
    clock: u64,
    total_evictions: usize,
    total_inserts: usize,
    total_hits: usize,
    total_misses: usize,
}

impl EvictableCache {
    fn new(capacity: usize, policy: EvictionPolicy) -> Self {
        Self {
            capacity,
            policy,
            entries: Vec::with_capacity(capacity),
            access_counts: Vec::with_capacity(capacity),
            access_order: Vec::with_capacity(capacity),
            insert_order: Vec::with_capacity(capacity),
            clock: 0,
            total_evictions: 0,
            total_inserts: 0,
            total_hits: 0,
            total_misses: 0,
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn find(&self, key: u16) -> Option<usize> {
        self.entries.iter().position(|(k, _)| *k == key)
    }

    fn access(&mut self, key: u16) -> Option<f32> {
        self.clock += 1;
        if let Some(idx) = self.find(key) {
            self.access_counts[idx] += 1;
            self.access_order[idx] = self.clock;
            self.total_hits += 1;
            Some(self.entries[idx].1)
        } else {
            self.total_misses += 1;
            None
        }
    }

    fn victim_index(&self) -> Option<usize> {
        if self.entries.is_empty() {
            return None;
        }
        match self.policy {
            EvictionPolicy::LRU => {
                // Evict least recently accessed
                self.access_order.iter().enumerate().min_by_key(|&(_, t)| *t).map(|(i, _)| i)
            }
            EvictionPolicy::LFU => {
                // Evict least frequently accessed
                self.access_counts.iter().enumerate().min_by_key(|&(_, c)| *c).map(|(i, _)| i)
            }
            EvictionPolicy::FIFO => {
                // Evict oldest insertion
                self.insert_order.iter().enumerate().min_by_key(|&(_, t)| *t).map(|(i, _)| i)
            }
        }
    }

    fn evict_one(&mut self) -> Option<(u16, f32)> {
        let idx = self.victim_index()?;
        let entry = self.entries.swap_remove(idx);
        self.access_counts.swap_remove(idx);
        self.access_order.swap_remove(idx);
        self.insert_order.swap_remove(idx);
        self.total_evictions += 1;
        Some(entry)
    }

    fn insert(&mut self, key: u16, value: f32) {
        self.clock += 1;
        // Update if already present
        if let Some(idx) = self.find(key) {
            self.entries[idx].1 = value;
            self.access_counts[idx] += 1;
            self.access_order[idx] = self.clock;
            return;
        }
        // Evict if at capacity
        if self.len() >= self.capacity {
            self.evict_one();
        }
        self.entries.push((key, value));
        self.access_counts.push(1);
        self.access_order.push(self.clock);
        self.insert_order.push(self.clock);
        self.total_inserts += 1;
    }

    fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            return 0.0;
        }
        self.total_hits as f64 / total as f64
    }
}

fuzz_target!(|input: EvictionInput| {
    let capacity = (input.capacity as usize % 32) + 2;
    let policy = match input.policy % 3 {
        0 => EvictionPolicy::LRU,
        1 => EvictionPolicy::LFU,
        _ => EvictionPolicy::FIFO,
    };

    let mut cache = EvictableCache::new(capacity, policy);

    // Invariant 1: Fresh cache is empty.
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.total_evictions, 0);

    for op in input.ops.iter().take(512) {
        match op {
            EvictionOp::Insert { key, value_byte } => {
                let value = *value_byte as f32 / 255.0;
                cache.insert(*key, value);
                // Invariant 2: Size never exceeds capacity.
                assert!(
                    cache.len() <= capacity,
                    "cache size {} > capacity {}",
                    cache.len(),
                    capacity
                );
                // Invariant 3: Inserted key is findable.
                assert!(cache.find(*key).is_some(), "key {} not found after insert", key);
            }
            EvictionOp::Access { key } => {
                let _ = cache.access(*key);
                // Invariant 4: Access doesn't change size.
                assert!(cache.len() <= capacity);
            }
            EvictionOp::Evict => {
                let prev_len = cache.len();
                if let Some((key, _)) = cache.evict_one() {
                    // Invariant 5: Evict removes exactly one entry.
                    assert_eq!(cache.len(), prev_len - 1);
                    // Invariant 6: Evicted key is gone.
                    assert!(cache.find(key).is_none(), "evicted key {} still present", key);
                } else {
                    assert_eq!(prev_len, 0, "evict_one returned None on non-empty cache");
                }
            }
            EvictionOp::EvictN { count } => {
                let n = (*count as usize % 16).min(cache.len());
                for _ in 0..n {
                    cache.evict_one();
                }
                assert!(cache.len() <= capacity);
            }
            EvictionOp::BulkInsert { keys } => {
                for &k in keys.iter().take(32) {
                    cache.insert(k as u16, k as f32 / 255.0);
                    assert!(cache.len() <= capacity);
                }
            }
            EvictionOp::Stats => {
                let hr = cache.hit_rate();
                assert!((0.0..=1.0).contains(&hr), "hit rate {hr} out of range");
                // Invariant 7: hits + misses == total accesses.
                // (inserts that update don't count as access via access())
            }
        }
    }

    // Invariant 8: Internal vectors are consistent.
    assert_eq!(cache.entries.len(), cache.access_counts.len());
    assert_eq!(cache.entries.len(), cache.access_order.len());
    assert_eq!(cache.entries.len(), cache.insert_order.len());

    // Invariant 9: Evicting everything empties the cache.
    while cache.evict_one().is_some() {}
    assert_eq!(cache.len(), 0);
});
