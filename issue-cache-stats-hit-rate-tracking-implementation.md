# [Performance] Cache Statistics Hit Rate Tracking Implementation

## Problem Description

The `CacheStats::hit_rate` field in the KV cache system currently returns a hardcoded value of `0.0` instead of tracking actual cache performance metrics. This prevents proper monitoring of cache efficiency and optimization of memory usage patterns during neural network inference.

## Environment

- **File**: `crates/bitnet-inference/src/cache.rs`
- **Component**: KV Cache Statistics System
- **Rust Version**: 1.90.0+ (2024 edition)
- **Features Affected**: `cpu`, `gpu` (cache used in both backends)

## Root Cause Analysis

The `CacheStats` structure includes a `hit_rate` field for tracking cache performance, but the current implementation lacks the infrastructure to measure actual cache hits and misses:

### **Current Implementation:**
```rust
pub struct CacheStats {
    pub total_entries: usize,
    pub compressed_entries: usize,
    pub current_size_bytes: usize,
    pub max_size_bytes: usize,
    pub hit_rate: f64,           // ← Always returns 0.0
    pub memory_efficiency: f64,
    pub cache_size: usize,
}

// In KVCache::stats method:
CacheStats {
    // ... other fields populated correctly ...
    hit_rate: 0.0, // Would need to track hits/misses
    // ...
}
```

### **Missing Components:**
1. **Hit/Miss Counters**: No tracking of cache access patterns
2. **Metrics Collection**: No infrastructure for gathering performance data
3. **Rate Calculation**: No logic to compute hit rates from collected metrics
4. **Performance Analysis**: No ability to optimize cache behavior based on real usage

## Impact Assessment

### **Severity**: Medium
### **Affected Operations**: Cache performance monitoring and optimization
### **Business Impact**: Degraded observability and optimization capabilities

**Current Limitations:**
- Cannot monitor cache efficiency during inference
- Unable to detect cache performance issues
- Missing data for memory usage optimization
- No metrics for capacity planning decisions

## Proposed Solution

### **Primary Approach**: Comprehensive Cache Metrics Tracking

Implement a complete cache metrics tracking system that accurately measures hit rates and provides detailed performance insights.

### **Implementation Strategy:**

#### **1. Enhanced KVCache Structure**
```rust
pub struct KVCache {
    cache: HashMap<CacheKey, CacheEntry>,
    capacity: usize,
    device: Device,
    // New metrics tracking fields
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    total_accesses: AtomicU64,
    hit_rate_window: VecDeque<bool>, // For windowed hit rate calculation
    window_size: usize,
}
```

#### **2. Metrics Collection Infrastructure**
```rust
impl KVCache {
    pub fn new(capacity: usize, device: Device) -> Self {
        Self {
            cache: HashMap::new(),
            capacity,
            device,
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            total_accesses: AtomicU64::new(0),
            hit_rate_window: VecDeque::with_capacity(1000), // Track last 1000 accesses
            window_size: 1000,
        }
    }

    fn record_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
        self.total_accesses.fetch_add(1, Ordering::Relaxed);
        // Update windowed metrics if needed
    }

    fn record_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        self.total_accesses.fetch_add(1, Ordering::Relaxed);
    }
}
```

#### **3. Accurate Hit Rate Calculation**
```rust
impl KVCache {
    pub fn stats(&self) -> CacheStats {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;

        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };

        CacheStats {
            total_entries: self.cache.len(),
            compressed_entries: self.count_compressed_entries(),
            current_size_bytes: self.calculate_memory_usage(),
            max_size_bytes: self.capacity * std::mem::size_of::<CacheEntry>(),
            hit_rate, // Now accurately calculated
            memory_efficiency: self.calculate_memory_efficiency(),
            cache_size: self.cache.len(),
        }
    }
}
```

#### **4. Enhanced Cache Operations**
```rust
impl KVCache {
    pub fn get(&mut self, key: &CacheKey) -> Option<&CacheEntry> {
        if let Some(entry) = self.cache.get(key) {
            self.record_hit();
            Some(entry)
        } else {
            self.record_miss();
            None
        }
    }

    pub fn update(&mut self, key: CacheKey, k_cache: BitNetTensor, v_cache: BitNetTensor) -> Result<()> {
        // Check if this is a hit (existing key) or miss (new key)
        let is_hit = self.cache.contains_key(&key);

        if is_hit {
            self.record_hit();
        } else {
            self.record_miss();

            // Handle capacity management
            if self.cache.len() >= self.capacity {
                self.evict_lru_entry()?;
            }
        }

        self.cache.insert(key, CacheEntry {
            k_cache,
            v_cache,
            last_used: Instant::now(),
        });

        Ok(())
    }
}
```

## Implementation Plan

### **Phase 1: Core Metrics Infrastructure (Week 1)**

#### **Task 1.1: Add Metrics Fields**
```rust
// Add to KVCache struct
struct KVCache {
    // Existing fields...
    metrics: CacheMetrics,
}

#[derive(Debug, Default)]
struct CacheMetrics {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    memory_pressure_events: AtomicU64,
}
```

#### **Task 1.2: Implement Metrics Collection**
```rust
impl CacheMetrics {
    fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    fn get_hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;

        if total > 0 { hits as f64 / total as f64 } else { 0.0 }
    }
}
```

### **Phase 2: Integration with Cache Operations (Week 2)**

#### **Task 2.1: Update Cache Access Methods**
```rust
impl KVCache {
    pub fn get_cached_kv(&mut self, layer_idx: usize, seq_pos: usize) -> Option<(BitNetTensor, BitNetTensor)> {
        let key = CacheKey { layer_idx, seq_pos };

        if let Some(entry) = self.cache.get_mut(&key) {
            self.metrics.record_hit();
            entry.last_used = Instant::now();
            Some((entry.k_cache.clone(), entry.v_cache.clone()))
        } else {
            self.metrics.record_miss();
            None
        }
    }
}
```

#### **Task 2.2: Enhanced Statistics Reporting**
```rust
impl KVCache {
    pub fn detailed_stats(&self) -> DetailedCacheStats {
        DetailedCacheStats {
            basic_stats: self.stats(),
            hit_rate: self.metrics.get_hit_rate(),
            total_hits: self.metrics.hits.load(Ordering::Relaxed),
            total_misses: self.metrics.misses.load(Ordering::Relaxed),
            eviction_count: self.metrics.evictions.load(Ordering::Relaxed),
            average_entry_size: self.calculate_average_entry_size(),
            memory_pressure_events: self.metrics.memory_pressure_events.load(Ordering::Relaxed),
        }
    }
}
```

### **Phase 3: Advanced Analytics (Week 3)**

#### **Task 3.1: Windowed Hit Rate Tracking**
```rust
struct WindowedMetrics {
    recent_accesses: VecDeque<CacheAccess>,
    window_duration: Duration,
}

#[derive(Debug, Clone)]
struct CacheAccess {
    timestamp: Instant,
    was_hit: bool,
    layer_idx: usize,
}

impl WindowedMetrics {
    fn record_access(&mut self, was_hit: bool, layer_idx: usize) {
        let access = CacheAccess {
            timestamp: Instant::now(),
            was_hit,
            layer_idx,
        };

        self.recent_accesses.push_back(access);
        self.cleanup_old_accesses();
    }

    fn get_windowed_hit_rate(&self) -> f64 {
        if self.recent_accesses.is_empty() { return 0.0; }

        let hits = self.recent_accesses.iter()
            .filter(|access| access.was_hit)
            .count();

        hits as f64 / self.recent_accesses.len() as f64
    }
}
```

#### **Task 3.2: Performance Analysis Tools**
```rust
impl KVCache {
    pub fn analyze_performance(&self) -> CachePerformanceAnalysis {
        CachePerformanceAnalysis {
            overall_hit_rate: self.metrics.get_hit_rate(),
            recent_hit_rate: self.windowed_metrics.get_windowed_hit_rate(),
            hit_rate_by_layer: self.calculate_hit_rate_by_layer(),
            memory_efficiency: self.calculate_memory_efficiency(),
            recommended_capacity: self.suggest_optimal_capacity(),
            performance_issues: self.detect_performance_issues(),
        }
    }
}
```

## Testing Strategy

### **Unit Tests:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hit_rate_calculation() {
        let mut cache = KVCache::new(100, Device::Cpu);

        // Initially no accesses, hit rate should be 0.0
        assert_eq!(cache.stats().hit_rate, 0.0);

        // Add some entries and test hit/miss recording
        let key1 = CacheKey { layer_idx: 0, seq_pos: 0 };
        cache.update(key1, create_test_tensor(), create_test_tensor()).unwrap();

        // First access should be a miss (insertion)
        cache.get_cached_kv(0, 0);

        // Second access should be a hit
        cache.get_cached_kv(0, 0);

        let stats = cache.stats();
        assert!(stats.hit_rate > 0.0);
        assert!(stats.hit_rate <= 1.0);
    }

    #[test]
    fn test_cache_metrics_accuracy() {
        let mut cache = KVCache::new(10, Device::Cpu);

        // Test hit rate with known pattern
        for i in 0..5 {
            let key = CacheKey { layer_idx: i, seq_pos: 0 };
            cache.update(key, create_test_tensor(), create_test_tensor()).unwrap();
        }

        // Access each entry twice (should be 5 hits out of 10 total accesses)
        for i in 0..5 {
            cache.get_cached_kv(i, 0); // Miss (first access)
            cache.get_cached_kv(i, 0); // Hit (second access)
        }

        let stats = cache.stats();
        assert!((stats.hit_rate - 0.5).abs() < 0.1); // Should be around 50%
    }
}
```

### **Integration Tests:**
```rust
#[test]
fn test_cache_performance_during_inference() {
    let mut engine = create_test_inference_engine();
    let test_sequence = create_test_token_sequence(100);

    // Run inference and collect cache statistics
    let result = engine.generate(&test_sequence);
    let cache_stats = engine.get_cache_stats();

    // Verify cache is being used effectively
    assert!(cache_stats.hit_rate > 0.3); // Should have reasonable hit rate
    assert!(cache_stats.total_entries > 0); // Should have cached some data
}
```

## Alternative Approaches

### **Alternative 1: Simple Counter-Based Tracking**
**Approach**: Use basic atomic counters without windowed analysis
**Pros**: Simpler implementation, lower overhead
**Cons**: Less detailed analytics, no temporal analysis

### **Alternative 2: External Metrics Collection**
**Approach**: Use external monitoring system (e.g., Prometheus metrics)
**Pros**: Better integration with monitoring infrastructure
**Cons**: Additional dependencies, more complex setup

### **Alternative 3: Sampling-Based Metrics**
**Approach**: Only track metrics for a sample of cache operations
**Pros**: Lower performance overhead
**Cons**: Less accurate metrics, potential sampling bias

**Selected Approach**: Primary comprehensive tracking provides the best balance of accuracy and performance insight.

## Performance Considerations

### **Memory Overhead:**
- Atomic counters: ~32 bytes per cache instance
- Windowed metrics: ~8KB for 1000-entry window
- Total overhead: <0.1% of typical cache memory usage

### **CPU Overhead:**
- Atomic operations: ~1-2 CPU cycles per cache access
- Windowed updates: ~10-20 cycles per access
- Total overhead: <1% of cache operation time

### **Optimization Opportunities:**
- Use relaxed memory ordering for non-critical metrics
- Batch windowed metric updates
- Optional detailed tracking for production deployments

## Success Metrics

### **Functionality:**
- [ ] Accurate hit rate calculation for all cache operations
- [ ] Real-time metrics available through stats API
- [ ] Windowed hit rate tracking for temporal analysis
- [ ] Per-layer cache performance analysis

### **Performance:**
- [ ] Metrics collection overhead <1% of cache operation time
- [ ] Memory overhead <0.1% of cache memory usage
- [ ] No measurable impact on inference throughput

### **Quality:**
- [ ] Unit test coverage >95% for metrics code
- [ ] Integration tests validate metrics accuracy during inference
- [ ] Performance benchmarks show acceptable overhead

## Acceptance Criteria

- [ ] `CacheStats::hit_rate` returns accurate hit rate based on real access patterns
- [ ] Cache hit/miss events are properly recorded for all cache operations
- [ ] Statistics API provides both instantaneous and windowed hit rates
- [ ] Performance overhead of metrics collection is <1% of cache operation time
- [ ] Unit tests validate metrics accuracy under various access patterns
- [ ] Integration tests demonstrate proper metrics collection during inference
- [ ] Documentation updated to describe cache performance monitoring capabilities

## Labels

- `performance`
- `cache-optimization`
- `observability`
- `metrics`
- `cpu-gpu-common`

## Related Issues

- **Dependencies**: None (standalone enhancement)
- **Related**: Issue #XXX (KV Cache Optimization), Issue #XXX (Memory Management)
- **Blocks**: Production deployment monitoring requirements
