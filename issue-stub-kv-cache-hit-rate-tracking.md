# [STUB] CacheStats::hit_rate hardcoded to 0.0 instead of tracking actual cache performance

## Problem Description

The `CacheStats::hit_rate` field in `cache.rs` is hardcoded to `0.0` and doesn't track actual cache hits and misses, preventing performance monitoring and optimization of the KV cache system.

## Environment

**File**: `crates/bitnet-inference/src/cache.rs`
**Component**: KV Cache Performance Monitoring
**Issue Type**: Stub Implementation / Missing Metrics

## Root Cause Analysis

**Current Implementation:**
```rust
pub struct CacheStats {
    pub total_entries: usize,
    pub compressed_entries: usize,
    pub current_size_bytes: usize,
    pub max_size_bytes: usize,
    pub hit_rate: f64,
    pub memory_efficiency: f64,
    pub cache_size: usize,
}

// In KVCache::stats():
hit_rate: 0.0, // Would need to track hits/misses
```

**Analysis:**
1. **No Hit/Miss Tracking**: Cache access patterns are not monitored
2. **Missing Performance Insights**: Cannot assess cache effectiveness
3. **Optimization Impediment**: No data to guide cache tuning
4. **Stub Implementation**: Placeholder value provides no useful information

## Impact Assessment

**Severity**: Medium
**Affected Areas**:
- Cache performance monitoring
- System optimization guidance
- Production performance debugging
- Resource allocation decisions

**Performance Impact**:
- Cannot identify cache inefficiencies
- Missing optimization opportunities
- Reduced visibility into system bottlenecks
- Inability to validate cache configuration changes

## Proposed Solution

### Complete Cache Hit Rate Tracking Implementation

```rust
use std::sync::atomic::{AtomicU64, Ordering};

pub struct KVCache {
    cache: HashMap<CacheKey, CacheEntry>,
    max_entries: usize,
    current_size: usize,
    max_size: usize,

    // Hit/miss tracking
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,

    // Detailed performance tracking
    hit_rate_window: RollingWindow<bool>,
    access_patterns: AccessPatternTracker,
    performance_history: PerformanceHistory,
}

impl KVCache {
    pub fn new(max_entries: usize, max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_entries,
            current_size: 0,
            max_size,
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            hit_rate_window: RollingWindow::new(1000), // Last 1000 accesses
            access_patterns: AccessPatternTracker::new(),
            performance_history: PerformanceHistory::new(),
        }
    }

    pub fn get(&mut self, layer_idx: usize, sequence_pos: usize) -> Option<&CacheEntry> {
        let key = self.create_cache_key(layer_idx, sequence_pos);
        let start_time = Instant::now();

        if let Some(entry) = self.cache.get(&key) {
            // Cache hit
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            self.hit_rate_window.add(true);
            self.access_patterns.record_hit(layer_idx, sequence_pos);

            // Update entry access time for LRU
            self.update_access_time(&key);

            let access_time = start_time.elapsed();
            self.performance_history.record_hit(access_time);

            Some(entry)
        } else {
            // Cache miss
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
            self.hit_rate_window.add(false);
            self.access_patterns.record_miss(layer_idx, sequence_pos);

            let access_time = start_time.elapsed();
            self.performance_history.record_miss(access_time);

            None
        }
    }

    pub fn put(&mut self, layer_idx: usize, sequence_pos: usize, entry: CacheEntry) -> Result<()> {
        let key = self.create_cache_key(layer_idx, sequence_pos);
        let entry_size = entry.size_bytes();

        // Check if we need to evict entries
        self.ensure_capacity(entry_size)?;

        // Insert new entry
        self.cache.insert(key, entry);
        self.current_size += entry_size;

        // Update access patterns
        self.access_patterns.record_insertion(layer_idx, sequence_pos);

        Ok(())
    }

    pub fn stats(&self) -> CacheStats {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total_accesses = hits + misses;

        let hit_rate = if total_accesses > 0 {
            hits as f64 / total_accesses as f64
        } else {
            0.0
        };

        let windowed_hit_rate = self.hit_rate_window.average();
        let memory_efficiency = self.calculate_memory_efficiency();

        CacheStats {
            total_entries: self.cache.len(),
            compressed_entries: self.count_compressed_entries(),
            current_size_bytes: self.current_size,
            max_size_bytes: self.max_size,
            hit_rate,
            windowed_hit_rate,
            memory_efficiency,
            cache_size: self.current_size,

            // Additional detailed metrics
            total_hits: hits,
            total_misses: misses,
            total_accesses,
            average_hit_time_ns: self.performance_history.average_hit_time(),
            average_miss_time_ns: self.performance_history.average_miss_time(),
            access_patterns: self.access_patterns.summary(),
        }
    }

    fn calculate_memory_efficiency(&self) -> f64 {
        if self.max_size == 0 {
            return 0.0;
        }

        let used_ratio = self.current_size as f64 / self.max_size as f64;
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let total = hits + self.cache_misses.load(Ordering::Relaxed);

        if total == 0 {
            return 0.0;
        }

        let hit_rate = hits as f64 / total as f64;

        // Efficiency combines memory usage and hit rate
        (hit_rate * 0.7) + ((1.0 - used_ratio) * 0.3)
    }

    fn ensure_capacity(&mut self, required_size: usize) -> Result<()> {
        while self.current_size + required_size > self.max_size && !self.cache.is_empty() {
            self.evict_lru_entry()?;
        }

        if self.current_size + required_size > self.max_size {
            return Err(anyhow::anyhow!(
                "Cannot fit entry of size {} bytes (current: {}, max: {})",
                required_size, self.current_size, self.max_size
            ));
        }

        Ok(())
    }

    pub fn detailed_performance_report(&self) -> PerformanceReport {
        let stats = self.stats();

        PerformanceReport {
            overall_stats: stats,
            layer_performance: self.access_patterns.layer_performance(),
            temporal_patterns: self.access_patterns.temporal_analysis(),
            memory_usage_history: self.performance_history.memory_usage_over_time(),
            recommendations: self.generate_optimization_recommendations(),
        }
    }

    fn generate_optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        let stats = self.stats();

        if stats.hit_rate < 0.7 {
            recommendations.push(OptimizationRecommendation {
                priority: Priority::High,
                category: Category::CacheSize,
                description: "Low hit rate detected. Consider increasing cache size.".to_string(),
                suggested_action: format!(
                    "Increase max_size from {} to {}",
                    self.max_size,
                    self.max_size * 2
                ),
            });
        }

        if stats.memory_efficiency < 0.5 {
            recommendations.push(OptimizationRecommendation {
                priority: Priority::Medium,
                category: Category::MemoryUsage,
                description: "Poor memory efficiency. Cache may be underutilized.".to_string(),
                suggested_action: "Consider reducing cache size or improving eviction policy.".to_string(),
            });
        }

        // Analyze access patterns for hotspots
        let layer_stats = self.access_patterns.layer_performance();
        for (layer_idx, layer_perf) in layer_stats.iter() {
            if layer_perf.hit_rate < 0.5 && layer_perf.access_count > 100 {
                recommendations.push(OptimizationRecommendation {
                    priority: Priority::Medium,
                    category: Category::AccessPattern,
                    description: format!("Layer {} has poor cache performance", layer_idx),
                    suggested_action: format!(
                        "Consider specialized caching strategy for layer {}",
                        layer_idx
                    ),
                });
            }
        }

        recommendations
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    // Basic stats
    pub total_entries: usize,
    pub compressed_entries: usize,
    pub current_size_bytes: usize,
    pub max_size_bytes: usize,
    pub cache_size: usize,

    // Hit rate metrics
    pub hit_rate: f64,
    pub windowed_hit_rate: f64,
    pub memory_efficiency: f64,

    // Detailed metrics
    pub total_hits: u64,
    pub total_misses: u64,
    pub total_accesses: u64,
    pub average_hit_time_ns: u64,
    pub average_miss_time_ns: u64,
    pub access_patterns: AccessPatternSummary,
}

struct RollingWindow<T> {
    data: VecDeque<T>,
    capacity: usize,
}

impl RollingWindow<bool> {
    fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn add(&mut self, value: bool) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(value);
    }

    fn average(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }

        let hits = self.data.iter().filter(|&&x| x).count();
        hits as f64 / self.data.len() as f64
    }
}
```

## Implementation Plan

### Task 1: Basic Hit/Miss Tracking
- [ ] Add atomic counters for cache hits and misses
- [ ] Update get/put methods to track access patterns
- [ ] Calculate accurate hit rate in stats method
- [ ] Add thread-safe access to counters

### Task 2: Advanced Metrics
- [ ] Implement rolling window for recent hit rate
- [ ] Add timing measurements for cache operations
- [ ] Track memory efficiency metrics
- [ ] Create detailed performance reporting

### Task 3: Access Pattern Analysis
- [ ] Implement layer-specific performance tracking
- [ ] Add temporal access pattern analysis
- [ ] Create optimization recommendations system
- [ ] Add cache effectiveness scoring

### Task 4: Performance Optimization
- [ ] Optimize counter updates for minimal overhead
- [ ] Add configurable metrics collection levels
- [ ] Implement lazy metric calculation
- [ ] Add performance monitoring integration

## Testing Strategy

### Hit Rate Accuracy Tests
```rust
#[test]
fn test_cache_hit_rate_tracking() {
    let mut cache = KVCache::new(100, 1024 * 1024);

    // Populate cache
    for i in 0..10 {
        let entry = create_test_entry(i);
        cache.put(0, i, entry).unwrap();
    }

    // Access existing entries (hits)
    for i in 0..10 {
        assert!(cache.get(0, i).is_some());
    }

    // Access non-existing entries (misses)
    for i in 10..15 {
        assert!(cache.get(0, i).is_none());
    }

    let stats = cache.stats();
    assert_eq!(stats.total_hits, 10);
    assert_eq!(stats.total_misses, 5);
    assert_eq!(stats.total_accesses, 15);
    assert!((stats.hit_rate - 10.0/15.0).abs() < 0.001);
}

#[test]
fn test_rolling_window_hit_rate() {
    let mut cache = KVCache::new(10, 1024);

    // Fill rolling window with known pattern
    for i in 0..1000 {
        if i % 2 == 0 {
            cache.put(0, i, create_test_entry(i)).unwrap();
            cache.get(0, i); // Hit
        } else {
            cache.get(0, i + 1000); // Miss
        }
    }

    let stats = cache.stats();
    assert!((stats.windowed_hit_rate - 0.5).abs() < 0.01);
}
```

## Related Issues/PRs

- Part of comprehensive cache performance monitoring
- Related to system optimization and tuning
- Connected to production performance visibility

## Acceptance Criteria

- [ ] Cache hit rate accurately reflects actual cache performance
- [ ] Hit/miss tracking has minimal performance overhead
- [ ] Rolling window provides recent performance visibility
- [ ] Detailed metrics support optimization decisions
- [ ] Thread-safe operation in concurrent scenarios
- [ ] Performance reports guide cache tuning

## Risk Assessment

**Low Risk**: Adding metrics tracking should not affect cache correctness.

**Mitigation Strategies**:
- Use atomic operations for thread-safe counter updates
- Minimize overhead by using efficient data structures
- Add configuration to disable detailed tracking if needed
- Implement lazy calculation of expensive metrics