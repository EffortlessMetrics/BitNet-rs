#![allow(dead_code, unused_imports, unused_variables, unused_unsafe, unsafe_op_in_unsafe_fn)]
#![cfg(all(feature = "cpu", target_arch = "aarch64"))]
#![allow(clippy::float_cmp)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::approx_constant)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::collections::VecDeque;

/// Tolerance for floating-point comparisons
const TOLERANCE: f32 = 1e-5;

/// Simple KV Cache implementation for testing
struct KVCache {
    /// Cached keys: (position, head_id, key_values)
    keys: VecDeque<(usize, usize, Vec<f32>)>,
    /// Cached values: (position, head_id, key_values)
    values: VecDeque<(usize, usize, Vec<f32>)>,
    /// Maximum capacity
    max_capacity: usize,
    /// Number of heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Current sequence position
    seq_pos: usize,
}

impl KVCache {
    /// Create a new KV cache
    fn new(max_capacity: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            keys: VecDeque::new(),
            values: VecDeque::new(),
            max_capacity,
            num_heads,
            head_dim,
            seq_pos: 0,
        }
    }

    /// Append a key-value pair
    fn append_kv(&mut self, head_id: usize, key: Vec<f32>, value: Vec<f32>) {
        // Check head_id is valid
        assert!(head_id < self.num_heads, "Invalid head_id");

        // Check dimensions
        assert_eq!(key.len(), self.head_dim, "Key dimension mismatch");
        assert_eq!(value.len(), self.head_dim, "Value dimension mismatch");

        // Store with position
        self.keys.push_back((self.seq_pos, head_id, key));
        self.values.push_back((self.seq_pos, head_id, value));

        // Evict oldest if exceeds capacity
        if self.keys.len() > self.max_capacity {
            self.keys.pop_front();
            self.values.pop_front();
        }

        self.seq_pos += 1;
    }

    /// Batch append multiple key-value pairs
    fn batch_append(&mut self, batch: Vec<(usize, Vec<f32>, Vec<f32>)>) {
        for (head_id, key, value) in batch {
            self.append_kv(head_id, key, value);
        }
    }

    /// Lookup key at specific position
    fn lookup_key(&self, pos: usize, head_id: usize) -> Option<Vec<f32>> {
        self.keys.iter().find(|(p, h, _)| *p == pos && *h == head_id).map(|(_, _, k)| k.clone())
    }

    /// Lookup value at specific position
    fn lookup_value(&self, pos: usize, head_id: usize) -> Option<Vec<f32>> {
        self.values.iter().find(|(p, h, _)| *p == pos && *h == head_id).map(|(_, _, v)| v.clone())
    }

    /// Range query: get all keys in position range
    fn range_query_keys(&self, start_pos: usize, end_pos: usize) -> Vec<Vec<f32>> {
        self.keys
            .iter()
            .filter(|(p, _, _)| *p >= start_pos && *p <= end_pos)
            .map(|(_, _, k)| k.clone())
            .collect()
    }

    /// Range query: get all values in position range
    fn range_query_values(&self, start_pos: usize, end_pos: usize) -> Vec<Vec<f32>> {
        self.values
            .iter()
            .filter(|(p, _, _)| *p >= start_pos && *p <= end_pos)
            .map(|(_, _, v)| v.clone())
            .collect()
    }

    /// Get current capacity usage
    fn capacity_usage(&self) -> usize {
        self.keys.len()
    }

    /// Check if cache is full
    fn is_full(&self) -> bool {
        self.keys.len() >= self.max_capacity
    }

    /// Get minimum position in cache (for eviction policy)
    fn min_position(&self) -> Option<usize> {
        self.keys.front().map(|(p, _, _)| *p)
    }

    /// Get maximum position in cache
    fn max_position(&self) -> Option<usize> {
        self.keys.back().map(|(p, _, _)| *p)
    }

    /// Clear all entries
    fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.seq_pos = 0;
    }

    /// Get sliding window of recent entries
    fn recent_window(&self, window_size: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let keys: Vec<Vec<f32>> =
            self.keys.iter().rev().take(window_size).map(|(_, _, k)| k.clone()).collect();
        let values: Vec<Vec<f32>> =
            self.values.iter().rev().take(window_size).map(|(_, _, v)| v.clone()).collect();
        (keys, values)
    }

    /// Get multi-head layout view
    fn get_head_layout(&self, head_id: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let keys: Vec<Vec<f32>> =
            self.keys.iter().filter(|(_, h, _)| *h == head_id).map(|(_, _, k)| k.clone()).collect();
        let values: Vec<Vec<f32>> = self
            .values
            .iter()
            .filter(|(_, h, _)| *h == head_id)
            .map(|(_, _, v)| v.clone())
            .collect();
        (keys, values)
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Create a test vector with a pattern
fn create_test_vector(size: usize, seed: u32) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let val = ((seed.wrapping_mul(31) + i as u32) % 1000) as f32 / 1000.0;
            val - 0.5
        })
        .collect()
}

/// Assert vectors are approximately equal
fn assert_vec_approx_eq(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "Vector lengths differ");
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(
            (x - y).abs() < TOLERANCE,
            "Values differ: {} vs {}, diff = {}",
            x,
            y,
            (x - y).abs()
        );
    }
}

/// Sum of absolute values in a vector
fn vec_sum_abs(v: &[f32]) -> f32 {
    v.iter().map(|x| x.abs()).sum()
}

// ============================================================================
// CACHE APPEND TESTS (4 tests)
// ============================================================================

#[test]
fn test_single_token_append() {
    let mut cache = KVCache::new(128, 8, 64);
    let key = create_test_vector(64, 42);
    let value = create_test_vector(64, 43);

    cache.append_kv(0, key.clone(), value.clone());

    assert_eq!(cache.capacity_usage(), 1);
    assert_eq!(cache.seq_pos, 1);

    let retrieved_key = cache.lookup_key(0, 0).expect("Key not found");
    let retrieved_value = cache.lookup_value(0, 0).expect("Value not found");

    assert_vec_approx_eq(&key, &retrieved_key);
    assert_vec_approx_eq(&value, &retrieved_value);
}

#[test]
fn test_batch_append() {
    let mut cache = KVCache::new(256, 8, 64);
    let batch_size = 32;

    let mut batch = Vec::new();
    for i in 0..batch_size {
        let key = create_test_vector(64, i as u32);
        let value = create_test_vector(64, i as u32 + 100);
        batch.push((i % 8, key, value));
    }

    cache.batch_append(batch);

    assert_eq!(cache.capacity_usage(), batch_size);
    assert_eq!(cache.seq_pos, batch_size);
}

#[test]
fn test_append_preserves_existing() {
    let mut cache = KVCache::new(128, 8, 64);

    // Append first token
    let key1 = create_test_vector(64, 1);
    let value1 = create_test_vector(64, 101);
    cache.append_kv(0, key1.clone(), value1.clone());

    // Append second token
    let key2 = create_test_vector(64, 2);
    let value2 = create_test_vector(64, 102);
    cache.append_kv(1, key2.clone(), value2.clone());

    // Check both exist
    assert_eq!(cache.capacity_usage(), 2);

    let retrieved_key1 = cache.lookup_key(0, 0).expect("First key not found");
    let retrieved_value1 = cache.lookup_value(0, 0).expect("First value not found");
    assert_vec_approx_eq(&key1, &retrieved_key1);
    assert_vec_approx_eq(&value1, &retrieved_value1);

    let retrieved_key2 = cache.lookup_key(1, 1).expect("Second key not found");
    let retrieved_value2 = cache.lookup_value(1, 1).expect("Second value not found");
    assert_vec_approx_eq(&key2, &retrieved_key2);
    assert_vec_approx_eq(&value2, &retrieved_value2);
}

#[test]
fn test_sequential_append_ordering() {
    let mut cache = KVCache::new(512, 8, 32);
    let num_tokens = 100;

    for token_id in 0..num_tokens {
        let head_id = token_id % 8;
        let key = create_test_vector(32, token_id as u32);
        let value = create_test_vector(32, token_id as u32 + 1000);
        cache.append_kv(head_id, key, value);
    }

    assert_eq!(cache.capacity_usage(), num_tokens);

    // Verify ordering: earliest should be at position 0
    let min_pos = cache.min_position().expect("No minimum position");
    assert_eq!(min_pos, 0);

    let max_pos = cache.max_position().expect("No maximum position");
    assert_eq!(max_pos, num_tokens - 1);

    // Verify positions are sequential
    let mut positions: Vec<usize> = cache.keys.iter().map(|(p, _, _)| *p).collect();
    positions.sort_unstable();
    for (i, pos) in positions.iter().enumerate() {
        assert_eq!(*pos, i, "Position ordering broken");
    }
}

// ============================================================================
// CACHE LOOKUP TESTS (4 tests)
// ============================================================================

#[test]
fn test_key_lookup_by_position() {
    let mut cache = KVCache::new(256, 8, 64);

    // Append 40 entries (5 positions * 8 heads each)
    for i in 0..40 {
        let head_id = i % 8;
        let key = create_test_vector(64, i as u32);
        cache.append_kv(head_id, key, create_test_vector(64, 999));
    }

    // Lookup and verify
    for pos in 0..40 {
        let head_id = pos % 8;
        let retrieved = cache.lookup_key(pos, head_id).expect("Key not found at position");
        assert_eq!(retrieved.len(), 64, "Key dimension mismatch");
    }
}

#[test]
fn test_value_retrieval() {
    let mut cache = KVCache::new(256, 8, 64);
    let num_entries = 50;

    let mut stored_values = Vec::new();
    for i in 0..num_entries {
        let head_id = i % 8;
        let value = create_test_vector(64, i as u32 + 2000);
        stored_values.push((i, head_id, value.clone()));
        cache.append_kv(head_id, create_test_vector(64, i as u32), value);
    }

    // Verify all values are retrievable
    for (pos, head_id, expected_value) in stored_values {
        let retrieved = cache.lookup_value(pos, head_id).expect("Value not found");
        assert_vec_approx_eq(&expected_value, &retrieved);
    }
}

#[test]
fn test_range_query() {
    let mut cache = KVCache::new(512, 8, 32);

    // Populate cache
    for i in 0..100 {
        let head_id = i % 8;
        let key = create_test_vector(32, i as u32);
        let value = create_test_vector(32, i as u32 + 500);
        cache.append_kv(head_id, key, value);
    }

    // Query range [30, 60]
    let keys_in_range = cache.range_query_keys(30, 60);
    let values_in_range = cache.range_query_values(30, 60);

    // Since we append 8 heads per position, we get multiple entries per position
    assert!(!keys_in_range.is_empty(), "Range query returned no keys");
    assert!(!values_in_range.is_empty(), "Range query returned no values");
}

#[test]
fn test_out_of_bounds_handling() {
    let mut cache = KVCache::new(128, 8, 64);

    let key = create_test_vector(64, 42);
    let value = create_test_vector(64, 43);
    cache.append_kv(0, key, value);

    // Query positions that don't exist
    assert_eq!(cache.lookup_key(999, 0), None, "Should not find out-of-bounds key");
    assert_eq!(cache.lookup_value(999, 0), None, "Should not find out-of-bounds value");

    // Query invalid head_id
    assert_eq!(cache.lookup_key(0, 99), None, "Should not find invalid head_id");

    // Query before first position
    assert_eq!(cache.lookup_key(0, 0), Some(create_test_vector(64, 42)));
}

// ============================================================================
// CACHE MANAGEMENT TESTS (4 tests)
// ============================================================================

#[test]
fn test_capacity_tracking() {
    let mut cache = KVCache::new(64, 8, 32);

    assert_eq!(cache.capacity_usage(), 0, "Initial capacity should be 0");
    assert!(!cache.is_full(), "Cache should not be full initially");

    // Fill to half capacity
    for i in 0..32 {
        let head_id = i % 8;
        let key = create_test_vector(32, i as u32);
        let value = create_test_vector(32, i as u32 + 100);
        cache.append_kv(head_id, key, value);
    }

    assert_eq!(cache.capacity_usage(), 32);
    assert!(!cache.is_full(), "Cache should not be full at half");

    // Fill to capacity
    for i in 32..64 {
        let head_id = i % 8;
        let key = create_test_vector(32, i as u32);
        let value = create_test_vector(32, i as u32 + 100);
        cache.append_kv(head_id, key, value);
    }

    assert_eq!(cache.capacity_usage(), 64);
    assert!(cache.is_full(), "Cache should be full now");
}

#[test]
fn test_eviction_oldest_first() {
    let mut cache = KVCache::new(10, 1, 32);

    // Add 15 entries (should trigger eviction of first 5)
    for i in 0..15 {
        let key = create_test_vector(32, i as u32);
        let value = create_test_vector(32, i as u32 + 100);
        cache.append_kv(0, key, value);
    }

    assert_eq!(cache.capacity_usage(), 10, "Cache should maintain max capacity");

    // Oldest entries (0-4) should be evicted, newest (5-14) should remain
    let min_pos = cache.min_position().expect("No min position");
    assert_eq!(min_pos, 5, "Oldest remaining position should be 5");

    // Verify entry at position 5 exists
    let key_at_5 = cache.lookup_key(5, 0).expect("Entry at position 5 should exist");
    assert_eq!(key_at_5, create_test_vector(32, 5));

    // Verify entry at position 4 is gone
    assert_eq!(cache.lookup_key(4, 0), None, "Entry at position 4 should be evicted");
}

#[test]
fn test_sliding_window() {
    let mut cache = KVCache::new(256, 8, 32);

    // Populate cache with 100 entries
    for i in 0..100 {
        let head_id = i % 8;
        let key = create_test_vector(32, i as u32);
        let value = create_test_vector(32, i as u32 + 200);
        cache.append_kv(head_id, key, value);
    }

    // Get recent window of size 10
    let (window_keys, window_values) = cache.recent_window(10);

    assert!(window_keys.len() <= 10, "Window should be at most 10");
    assert_eq!(
        window_keys.len(),
        window_values.len(),
        "Keys and values window should have same length"
    );

    // Get recent window of size 256 (more than available)
    let (large_window_keys, _large_window_values) = cache.recent_window(256);
    assert!(large_window_keys.len() <= 100, "Window should not exceed cache size");
}

#[test]
fn test_clear_and_reset() {
    let mut cache = KVCache::new(128, 8, 64);

    // Populate cache
    for i in 0..50 {
        let head_id = i % 8;
        let key = create_test_vector(64, i as u32);
        let value = create_test_vector(64, i as u32 + 100);
        cache.append_kv(head_id, key, value);
    }

    assert_eq!(cache.capacity_usage(), 50);
    assert_eq!(cache.seq_pos, 50);

    // Clear
    cache.clear();

    assert_eq!(cache.capacity_usage(), 0, "Cache should be empty after clear");
    assert_eq!(cache.seq_pos, 0, "Sequence position should be reset");

    // Verify lookups fail
    assert_eq!(cache.lookup_key(0, 0), None, "No keys should exist after clear");
    assert_eq!(cache.lookup_value(0, 0), None, "No values should exist after clear");

    // Verify we can add new entries
    let key = create_test_vector(64, 42);
    let value = create_test_vector(64, 43);
    cache.append_kv(0, key, value);
    assert_eq!(cache.capacity_usage(), 1, "Should be able to add after clear");
}

// ============================================================================
// BITNET-SPECIFIC TESTS (3+ tests)
// ============================================================================

#[test]
fn test_multi_head_cache_layout() {
    let num_heads = 8;
    let head_dim = 64;
    let mut cache = KVCache::new(512, num_heads, head_dim);

    // We need a structure where we have multiple heads at the same position.
    // The KVCache model stores one entry per append, each getting a sequential position.
    // So we append num_heads entries for each token position.
    // Positions 0-7 are token 0 (heads 0-7), 8-15 are token 1, etc.

    for token_id in 0..10 {
        for head_id in 0..num_heads {
            let global_pos = token_id * num_heads + head_id;
            let key = create_test_vector(head_dim, global_pos as u32);
            let value = create_test_vector(head_dim, (global_pos + 1000) as u32);
            cache.append_kv(head_id, key, value);
        }
    }

    // Verify multi-head layout per token
    // For token_id T, the positions are [T*8, T*8+1, ..., T*8+7] for heads [0, 1, ..., 7]
    for token_id in 0..10 {
        for head_id in 0..num_heads {
            let expected_pos = token_id * num_heads + head_id;
            let key_opt = cache.lookup_key(expected_pos, head_id);
            let value_opt = cache.lookup_value(expected_pos, head_id);
            assert!(
                key_opt.is_some(),
                "Key missing for token {}, head {} at pos {}",
                token_id,
                head_id,
                expected_pos
            );
            assert!(
                value_opt.is_some(),
                "Value missing for token {}, head {} at pos {}",
                token_id,
                head_id,
                expected_pos
            );
        }
    }
}

#[test]
fn test_head_independent_storage() {
    let mut cache = KVCache::new(256, 8, 32);

    // Store different data for each head at different positions
    let mut stored_values = Vec::new();

    for head_id in 0..8 {
        let key = create_test_vector(32, head_id as u32);
        let value = create_test_vector(32, head_id as u32 + 100);
        stored_values.push((head_id, value.clone()));
        cache.append_kv(head_id, key, value);
    }

    // Verify each head's storage is independent
    for (head_id, expected_value) in stored_values {
        let retrieved = cache.lookup_value(head_id, head_id).expect("Value not found");
        assert_vec_approx_eq(&expected_value, &retrieved);

        // Verify this value is different from other heads
        if head_id < 7 {
            let other_value =
                cache.lookup_value(head_id + 1, head_id + 1).expect("Other value not found");
            // Values should differ because they use different seeds
            let diff: f32 =
                retrieved.iter().zip(other_value.iter()).map(|(a, b)| (a - b).abs()).sum();
            assert!(diff > 0.01, "Head-independent storage may be broken");
        }
    }
}

#[test]
fn test_typical_sequence_length_2048() {
    let mut cache = KVCache::new(2048, 32, 64);
    let seq_len = 2048;

    // Simulate filling cache with typical sequence
    for pos in 0..seq_len {
        let head_id = pos % 32;
        let key = create_test_vector(64, pos as u32);
        let value = create_test_vector(64, pos as u32 + 1000);
        cache.append_kv(head_id, key, value);
    }

    assert_eq!(cache.capacity_usage(), seq_len);
    assert_eq!(cache.seq_pos, seq_len);

    // Verify random access throughout sequence
    for sample_pos in &[0, 256, 512, 1024, 1536, 2047] {
        let head_id = sample_pos % 32;
        assert!(
            cache.lookup_key(*sample_pos, head_id).is_some(),
            "Key should exist at position {}",
            sample_pos
        );
        assert!(
            cache.lookup_value(*sample_pos, head_id).is_some(),
            "Value should exist at position {}",
            sample_pos
        );
    }

    // Verify head layout is consistent
    for head_id in 0..32 {
        let (head_keys, head_values) = cache.get_head_layout(head_id);
        assert!(!head_keys.is_empty(), "Head {} should have cached keys", head_id);
        assert_eq!(head_keys.len(), head_values.len(), "Head {} key/value count mismatch", head_id);
    }
}

#[test]
fn test_typical_sequence_length_4096() {
    let mut cache = KVCache::new(4096, 32, 64);
    let seq_len = 4096;

    // Simulate filling cache with extended sequence
    for pos in 0..seq_len {
        let head_id = pos % 32;
        let key = create_test_vector(64, pos as u32);
        let value = create_test_vector(64, pos as u32 + 1000);
        cache.append_kv(head_id, key, value);
    }

    assert_eq!(cache.capacity_usage(), seq_len);

    // Verify windowed access pattern (common in transformers)
    let window_start = 3000;
    let window_end = 4000;
    let keys_in_window = cache.range_query_keys(window_start, window_end);
    let values_in_window = cache.range_query_values(window_start, window_end);

    assert!(!keys_in_window.is_empty(), "Should find keys in window");
    assert_eq!(keys_in_window.len(), values_in_window.len(), "Key/value count should match");
}

#[test]
fn test_multi_head_aggregate_operations() {
    let num_heads = 16;
    let head_dim = 32;
    let mut cache = KVCache::new(512, num_heads, head_dim);

    // Populate cache with multi-head structure
    for token_id in 0..100 {
        for head_id in 0..num_heads {
            let seed = (token_id * num_heads + head_id) as u32;
            let key = create_test_vector(head_dim, seed);
            let value = create_test_vector(head_dim, seed + 500);
            cache.append_kv(head_id, key, value);
        }
    }

    // Aggregate: compute statistics per head
    for head_id in 0..num_heads {
        let (head_keys, head_values) = cache.get_head_layout(head_id);

        // Verify we have consistent number of entries per head
        assert!(!head_keys.is_empty(), "Head {} should have entries", head_id);
        assert_eq!(head_keys.len(), head_values.len(), "Head {} key/value mismatch", head_id);

        // Compute sum of norms (simple aggregate)
        let key_norm: f32 = head_keys.iter().map(|k| vec_sum_abs(k)).sum();
        let value_norm: f32 = head_values.iter().map(|v| vec_sum_abs(v)).sum();

        assert!(key_norm > 0.0, "Key norm should be positive for head {}", head_id);
        assert!(value_norm > 0.0, "Value norm should be positive for head {}", head_id);
    }
}

#[test]
fn test_kv_cache_stress_mixed_operations() {
    let mut cache = KVCache::new(1024, 8, 64);

    // Phase 1: Append 100 entries
    for i in 0..100 {
        let head_id = i % 8;
        let key = create_test_vector(64, i as u32);
        let value = create_test_vector(64, i as u32 + 100);
        cache.append_kv(head_id, key, value);
    }

    // Phase 2: Range queries
    let _keys = cache.range_query_keys(10, 50);
    let _values = cache.range_query_values(10, 50);

    // Phase 3: Random lookups
    for idx in &[0, 25, 50, 75, 99] {
        let head_id = idx % 8;
        assert!(cache.lookup_key(*idx, head_id).is_some());
        assert!(cache.lookup_value(*idx, head_id).is_some());
    }

    // Phase 4: More appends to trigger eviction
    for i in 100..500 {
        let head_id = i % 8;
        let key = create_test_vector(64, i as u32);
        let value = create_test_vector(64, i as u32 + 100);
        cache.append_kv(head_id, key, value);
    }

    assert!(cache.capacity_usage() <= 1024, "Capacity should be respected");

    // Phase 5: Window queries on remaining data
    let (window_keys, window_values) = cache.recent_window(50);
    assert!(window_keys.len() <= 50);
    assert_eq!(window_keys.len(), window_values.len());

    // Phase 6: Clear and verify reset
    cache.clear();
    assert_eq!(cache.capacity_usage(), 0);
    assert_eq!(cache.seq_pos, 0);
}
