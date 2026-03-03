//! KV cache statistics and monitoring.
//!
//! Track hit rates, memory usage, evictions, and performance metrics
//! for the key-value cache used during inference.

use std::time::{Duration, Instant};

/// Snapshot of KV cache state.
#[derive(Debug, Clone)]
pub struct KvCacheSnapshot {
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub current_seq_len: usize,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    pub dtype_bytes: usize,
}

impl KvCacheSnapshot {
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        current_seq_len: usize,
        head_dim: usize,
        num_kv_heads: usize,
        dtype_bytes: usize,
    ) -> Self {
        Self { num_layers, max_seq_len, current_seq_len, head_dim, num_kv_heads, dtype_bytes }
    }

    /// Bytes used by current sequence length.
    pub fn used_bytes(&self) -> usize {
        // K + V for each layer: 2 * layers * seq * heads * dim * dtype_bytes
        2 * self.num_layers
            * self.current_seq_len
            * self.num_kv_heads
            * self.head_dim
            * self.dtype_bytes
    }

    /// Bytes allocated for max sequence length.
    pub fn allocated_bytes(&self) -> usize {
        2 * self.num_layers
            * self.max_seq_len
            * self.num_kv_heads
            * self.head_dim
            * self.dtype_bytes
    }

    /// Utilization ratio (0.0 to 1.0).
    pub fn utilization(&self) -> f64 {
        if self.max_seq_len == 0 {
            return 0.0;
        }
        self.current_seq_len as f64 / self.max_seq_len as f64
    }

    /// Remaining capacity in tokens.
    pub fn remaining_tokens(&self) -> usize {
        self.max_seq_len.saturating_sub(self.current_seq_len)
    }
}

/// Eviction event record.
#[derive(Debug, Clone)]
pub struct EvictionEvent {
    pub tokens_evicted: usize,
    pub reason: EvictionReason,
    pub timestamp: Instant,
}

/// Reason for cache eviction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionReason {
    CapacityFull,
    SlidingWindow,
    Manual,
    NewRequest,
}

/// Collected cache statistics.
#[derive(Debug, Clone)]
pub struct KvCacheStats {
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub tokens_evicted: u64,
    pub total_append_time: Duration,
    pub append_count: u64,
    pub peak_seq_len: usize,
    eviction_log: Vec<EvictionEvent>,
}

impl Default for KvCacheStats {
    fn default() -> Self {
        Self::new()
    }
}

impl KvCacheStats {
    pub fn new() -> Self {
        Self {
            lookups: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            tokens_evicted: 0,
            total_append_time: Duration::ZERO,
            append_count: 0,
            peak_seq_len: 0,
            eviction_log: Vec::new(),
        }
    }

    pub fn record_hit(&mut self) {
        self.lookups += 1;
        self.hits += 1;
    }

    pub fn record_miss(&mut self) {
        self.lookups += 1;
        self.misses += 1;
    }

    pub fn record_append(&mut self, duration: Duration, new_seq_len: usize) {
        self.append_count += 1;
        self.total_append_time += duration;
        if new_seq_len > self.peak_seq_len {
            self.peak_seq_len = new_seq_len;
        }
    }

    pub fn record_eviction(&mut self, tokens: usize, reason: EvictionReason) {
        self.evictions += 1;
        self.tokens_evicted += tokens as u64;
        self.eviction_log.push(EvictionEvent {
            tokens_evicted: tokens,
            reason,
            timestamp: Instant::now(),
        });
    }

    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            return 0.0;
        }
        self.hits as f64 / self.lookups as f64
    }

    pub fn avg_append_time(&self) -> Duration {
        if self.append_count == 0 {
            return Duration::ZERO;
        }
        self.total_append_time / self.append_count as u32
    }

    pub fn eviction_log(&self) -> &[EvictionEvent] {
        &self.eviction_log
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Merge another stats into this one.
    pub fn merge(&mut self, other: &KvCacheStats) {
        self.lookups += other.lookups;
        self.hits += other.hits;
        self.misses += other.misses;
        self.evictions += other.evictions;
        self.tokens_evicted += other.tokens_evicted;
        self.total_append_time += other.total_append_time;
        self.append_count += other.append_count;
        if other.peak_seq_len > self.peak_seq_len {
            self.peak_seq_len = other.peak_seq_len;
        }
    }
}

/// Report summary.
#[derive(Debug)]
pub struct CacheReport {
    pub snapshot: KvCacheSnapshot,
    pub stats: KvCacheStats,
}

impl CacheReport {
    pub fn new(snapshot: KvCacheSnapshot, stats: KvCacheStats) -> Self {
        Self { snapshot, stats }
    }

    pub fn memory_efficiency(&self) -> f64 {
        let alloc = self.snapshot.allocated_bytes();
        if alloc == 0 {
            return 0.0;
        }
        self.snapshot.used_bytes() as f64 / alloc as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> KvCacheSnapshot {
        KvCacheSnapshot::new(32, 4096, 1024, 128, 8, 2)
    }

    #[test]
    fn test_snapshot_used_bytes() {
        let snap = sample_snapshot();
        // 2 * 32 * 1024 * 8 * 128 * 2 = 134,217,728
        assert_eq!(snap.used_bytes(), 2 * 32 * 1024 * 8 * 128 * 2);
    }

    #[test]
    fn test_snapshot_allocated_bytes() {
        let snap = sample_snapshot();
        assert_eq!(snap.allocated_bytes(), 2 * 32 * 4096 * 8 * 128 * 2);
    }

    #[test]
    fn test_utilization() {
        let snap = sample_snapshot();
        assert!((snap.utilization() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_remaining_tokens() {
        let snap = sample_snapshot();
        assert_eq!(snap.remaining_tokens(), 3072);
    }

    #[test]
    fn test_hit_rate() {
        let mut stats = KvCacheStats::new();
        stats.record_hit();
        stats.record_hit();
        stats.record_miss();
        assert!((stats.hit_rate() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_hit_rate_empty() {
        let stats = KvCacheStats::new();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_eviction() {
        let mut stats = KvCacheStats::new();
        stats.record_eviction(100, EvictionReason::CapacityFull);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.tokens_evicted, 100);
        assert_eq!(stats.eviction_log().len(), 1);
    }

    #[test]
    fn test_append_stats() {
        let mut stats = KvCacheStats::new();
        stats.record_append(Duration::from_millis(10), 100);
        stats.record_append(Duration::from_millis(20), 200);
        assert_eq!(stats.append_count, 2);
        assert_eq!(stats.peak_seq_len, 200);
        assert_eq!(stats.avg_append_time(), Duration::from_millis(15));
    }

    #[test]
    fn test_merge() {
        let mut a = KvCacheStats::new();
        a.record_hit();
        a.record_miss();
        let mut b = KvCacheStats::new();
        b.record_hit();
        b.record_hit();
        b.record_append(Duration::from_millis(10), 500);
        a.merge(&b);
        assert_eq!(a.lookups, 4);
        assert_eq!(a.hits, 3);
        assert_eq!(a.peak_seq_len, 500);
    }

    #[test]
    fn test_reset() {
        let mut stats = KvCacheStats::new();
        stats.record_hit();
        stats.reset();
        assert_eq!(stats.lookups, 0);
    }

    #[test]
    fn test_cache_report() {
        let snap = sample_snapshot();
        let stats = KvCacheStats::new();
        let report = CacheReport::new(snap, stats);
        assert!((report.memory_efficiency() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_zero_capacity() {
        let snap = KvCacheSnapshot::new(0, 0, 0, 0, 0, 0);
        assert_eq!(snap.utilization(), 0.0);
        assert_eq!(snap.remaining_tokens(), 0);
    }
}
