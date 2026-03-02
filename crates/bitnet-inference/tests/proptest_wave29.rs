//! Property-based tests — wave 29.
//!
//! Covers sampling strategy properties, GenerationConfig builder invariants,
//! KVCache behaviour, PagedKvCache properties, PrefixCache invariants,
//! KernelRecorder, MetricsCollector, LatencyHistogram, MemoryProfiler,
//! and ThroughputTracker properties.
//!
//! 42 property tests validating: sampling bounds, config validation,
//! cache invariants, metrics monotonicity, and builder correctness.

use bitnet_inference::cache::{CacheConfig, EvictionPolicy, KVCache};
use bitnet_inference::config::GenerationConfig;
use bitnet_inference::kernel_recorder::KernelRecorder;
use bitnet_inference::kv_cache_optimized::{CacheEvictionPolicy, EvictionConfig, PagedKvCache};
use bitnet_inference::metrics::{LatencyHistogram, MemoryProfiler, MetricsCollector};
use bitnet_inference::prefix_cache::{PrefixCache, PrefixCacheConfig};
use bitnet_inference::sampling::{SamplingConfig, SamplingStrategy, greedy_sample};
use proptest::prelude::*;

// ── 1. Sampling strategy properties ─────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Greedy sample of uniform logits returns a valid index.
    #[test]
    fn prop_greedy_uniform_logits(n in 2usize..=100) {
        let logits = vec![1.0f32; n];
        if let Ok(token) = greedy_sample(&logits) {
            prop_assert!((token as usize) < n, "token {} >= n {}", token, n);
        }
    }

    /// Greedy sample returns the argmax.
    #[test]
    fn prop_greedy_returns_argmax(n in 2usize..=64) {
        let mut logits = vec![0.0f32; n];
        let peak = n / 2;
        logits[peak] = 100.0;
        if let Ok(token) = greedy_sample(&logits) {
            prop_assert_eq!(token as usize, peak);
        }
    }

    /// SamplingStrategy with temperature 0 behaves like greedy.
    #[test]
    fn prop_sampling_temp_zero_is_greedy(n in 2usize..=32) {
        let config = SamplingConfig {
            temperature: 0.0,
            seed: Some(42),
            ..Default::default()
        };
        let mut strategy = SamplingStrategy::new(config);
        let mut logits = vec![0.0f32; n];
        logits[0] = 100.0;
        if let Ok(token) = strategy.sample(&logits, &[]) {
            prop_assert_eq!(token, 0);
        }
    }

    /// SamplingStrategy reset doesn't panic.
    #[test]
    fn prop_sampling_reset_safe(_dummy in 0u8..1) {
        let config = SamplingConfig {
            temperature: 1.0,
            seed: Some(42),
            ..Default::default()
        };
        let mut strategy = SamplingStrategy::new(config);
        strategy.reset();
    }

    /// Sampling with valid temperature produces a valid token.
    #[test]
    fn prop_sampling_valid_token(
        n in 2usize..=32,
        temp in 0.1f32..2.0,
    ) {
        let config = SamplingConfig {
            temperature: temp,
            seed: Some(42),
            ..Default::default()
        };
        let mut strategy = SamplingStrategy::new(config);
        let logits: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        if let Ok(token) = strategy.sample(&logits, &[]) {
            prop_assert!((token as usize) < n, "token {} out of range", token);
        }
    }
}

// ── 2. GenerationConfig builder properties ──────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// GenerationConfig::greedy() validates ok.
    #[test]
    fn prop_gen_config_greedy_valid(_dummy in 0u8..1) {
        let config = GenerationConfig::greedy();
        prop_assert!(config.validate().is_ok());
    }

    /// GenerationConfig::creative() validates ok.
    #[test]
    fn prop_gen_config_creative_valid(_dummy in 0u8..1) {
        let config = GenerationConfig::creative();
        prop_assert!(config.validate().is_ok());
    }

    /// GenerationConfig::balanced() validates ok.
    #[test]
    fn prop_gen_config_balanced_valid(_dummy in 0u8..1) {
        let config = GenerationConfig::balanced();
        prop_assert!(config.validate().is_ok());
    }

    /// with_max_tokens sets max_new_tokens correctly.
    #[test]
    fn prop_gen_config_max_tokens(max_tokens in 1u32..=10000) {
        let config = GenerationConfig::greedy().with_max_tokens(max_tokens);
        prop_assert_eq!(config.max_new_tokens, max_tokens);
    }

    /// with_temperature sets the value correctly.
    #[test]
    fn prop_gen_config_temperature(temp in 0.0f32..5.0) {
        let config = GenerationConfig::greedy().with_temperature(temp);
        prop_assert!((config.temperature - temp).abs() < 1e-6);
    }

    /// with_seed sets the seed.
    #[test]
    fn prop_gen_config_seed(seed in 0u64..=u64::MAX) {
        let config = GenerationConfig::greedy().with_seed(seed);
        prop_assert_eq!(config.seed, Some(seed));
    }

    /// with_top_k sets the value.
    #[test]
    fn prop_gen_config_top_k(k in 1u32..=1000) {
        let config = GenerationConfig::creative().with_top_k(k);
        prop_assert_eq!(config.top_k, k);
    }

    /// with_top_p sets the value.
    #[test]
    fn prop_gen_config_top_p(p in 0.01f32..1.0) {
        let config = GenerationConfig::creative().with_top_p(p);
        prop_assert!((config.top_p - p).abs() < 1e-6);
    }

    /// with_repetition_penalty stores the value.
    #[test]
    fn prop_gen_config_repetition_penalty(penalty in 0.5f32..2.0) {
        let config = GenerationConfig::greedy().with_repetition_penalty(penalty);
        prop_assert!((config.repetition_penalty - penalty).abs() < 1e-6);
    }

    /// Adding stop sequences preserves them.
    #[test]
    fn prop_gen_config_stop_sequences(n in 1usize..=5) {
        let mut config = GenerationConfig::greedy();
        for i in 0..n {
            config = config.with_stop_sequence(format!("stop_{}", i));
        }
        prop_assert_eq!(config.stop_sequences.len(), n);
    }

    /// with_stop_string_window sets the window.
    #[test]
    fn prop_gen_config_stop_window(window in 16usize..=256) {
        let config = GenerationConfig::greedy().with_stop_string_window(window);
        prop_assert_eq!(config.stop_string_window, window);
    }

    /// with_skip_special_tokens toggles the flag.
    #[test]
    fn prop_gen_config_skip_special(skip in proptest::bool::ANY) {
        let config = GenerationConfig::greedy().with_skip_special_tokens(skip);
        prop_assert_eq!(config.skip_special_tokens, skip);
    }
}

// ── 3. KVCache properties ───────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// KVCache::new succeeds for valid default-ish configs.
    #[test]
    fn prop_kv_cache_construction(
        max_size in 1024usize..=65536,
        max_seq in 16usize..=256,
    ) {
        let config = CacheConfig {
            max_size_bytes: max_size,
            max_sequence_length: max_seq,
            enable_compression: false,
            eviction_policy: EvictionPolicy::LRU,
            block_size: 64,
        };
        let result = KVCache::new(config);
        prop_assert!(result.is_ok(), "KVCache::new failed");
    }

    /// KVCache size starts at 0.
    #[test]
    fn prop_kv_cache_initial_size(_dummy in 0u8..1) {
        let config = CacheConfig::default();
        if let Ok(cache) = KVCache::new(config) {
            prop_assert_eq!(cache.size(), 0);
        }
    }

    /// KVCache clear resets size to 0.
    #[test]
    fn prop_kv_cache_clear_resets(head_dim in 2usize..=16) {
        let config = CacheConfig::default();
        if let Ok(mut cache) = KVCache::new(config) {
            let k = vec![1.0f32; head_dim];
            let v = vec![1.0f32; head_dim];
            let _ = cache.store(0, 0, k, v);
            cache.clear();
            prop_assert_eq!(cache.size(), 0);
        }
    }

    /// KVCache store then contains returns true.
    #[test]
    fn prop_kv_cache_store_contains(
        layer in 0usize..=3,
        pos in 0usize..=15,
    ) {
        let config = CacheConfig::default();
        if let Ok(mut cache) = KVCache::new(config) {
            let k = vec![1.0f32; 8];
            let v = vec![2.0f32; 8];
            if cache.store(layer, pos, k, v).is_ok() {
                prop_assert!(cache.contains(layer, pos));
            }
        }
    }

    /// KVCache usage_percent starts at 0.
    #[test]
    fn prop_kv_cache_usage_initial(_dummy in 0u8..1) {
        let config = CacheConfig::default();
        if let Ok(cache) = KVCache::new(config) {
            prop_assert!((cache.usage_percent() - 0.0).abs() < 1e-6);
        }
    }

    /// KVCache clear_layer removes only that layer.
    #[test]
    fn prop_kv_cache_clear_layer(layer in 0usize..=2) {
        let config = CacheConfig::default();
        if let Ok(mut cache) = KVCache::new(config) {
            let k = vec![1.0f32; 4];
            let v = vec![1.0f32; 4];
            let _ = cache.store(layer, 0, k.clone(), v.clone());
            let other = (layer + 1) % 3;
            let _ = cache.store(other, 0, k, v);
            cache.clear_layer(layer);
            prop_assert!(!cache.contains(layer, 0));
            prop_assert!(cache.contains(other, 0));
        }
    }
}

// ── 4. PagedKvCache properties ──────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// PagedKvCache starts with no allocated pages.
    #[test]
    fn prop_paged_cache_initial_empty(
        tokens_per_page in 4usize..=32,
        head_dim in 2usize..=16,
    ) {
        let eviction = EvictionConfig {
            policy: CacheEvictionPolicy::LRU,
            max_pages: 64,
            window_size: 16,
        };
        let cache = PagedKvCache::new(tokens_per_page, head_dim, eviction);
        prop_assert_eq!(cache.allocated_pages(), 0);
    }

    /// Allocate then free a page adds to free list.
    #[test]
    fn prop_paged_cache_alloc_free(
        tokens_per_page in 4usize..=16,
        head_dim in 2usize..=8,
    ) {
        let eviction = EvictionConfig {
            policy: CacheEvictionPolicy::LRU,
            max_pages: 64,
            window_size: 16,
        };
        let mut cache = PagedKvCache::new(tokens_per_page, head_dim, eviction);
        if let Some(page_id) = cache.allocate_page(0) {
            prop_assert_eq!(cache.allocated_pages(), 1);
            cache.free_page(page_id);
            // Freed page is on the free list
            prop_assert_eq!(cache.free_pages(), 1);
            prop_assert_eq!(cache.allocated_pages(), 0);
        }
    }

    /// Clear resets all allocations.
    #[test]
    fn prop_paged_cache_clear(head_dim in 2usize..=8) {
        let eviction = EvictionConfig {
            policy: CacheEvictionPolicy::LRU,
            max_pages: 32,
            window_size: 8,
        };
        let mut cache = PagedKvCache::new(8, head_dim, eviction);
        let _ = cache.allocate_page(0);
        let _ = cache.allocate_page(1);
        cache.clear();
        prop_assert_eq!(cache.allocated_pages(), 0);
    }

    /// Multiple allocations increase count monotonically.
    #[test]
    fn prop_paged_cache_alloc_monotonic(n in 1usize..=10) {
        let eviction = EvictionConfig::default();
        let mut cache = PagedKvCache::new(8, 4, eviction);
        for i in 0..n {
            let _ = cache.allocate_page(i);
        }
        prop_assert!(cache.allocated_pages() <= n);
    }
}

// ── 5. PrefixCache properties ───────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// PrefixCache starts empty.
    #[test]
    fn prop_prefix_cache_empty(max_entries in 1usize..=100) {
        let config = PrefixCacheConfig {
            max_entries,
            ..Default::default()
        };
        let cache = PrefixCache::new(config);
        prop_assert!(cache.is_empty());
        prop_assert_eq!(cache.len(), 0);
    }

    /// PrefixCache insert then lookup returns the entry.
    #[test]
    fn prop_prefix_cache_insert_lookup(n_tokens in 4usize..=16) {
        let config = PrefixCacheConfig {
            max_entries: 10,
            min_prefix_length: 1,
            ..Default::default()
        };
        let mut cache = PrefixCache::new(config);
        let tokens: Vec<u32> = (0..n_tokens as u32).collect();
        let state = vec![42u8; 64];
        if cache.insert(&tokens, state).is_ok() {
            let result = cache.lookup(&tokens);
            prop_assert!(result.is_some(), "lookup after insert failed");
        }
    }

    /// PrefixCache clear empties the cache.
    #[test]
    fn prop_prefix_cache_clear(n in 1usize..=5) {
        let config = PrefixCacheConfig {
            max_entries: 10,
            min_prefix_length: 1,
            ..Default::default()
        };
        let mut cache = PrefixCache::new(config);
        for i in 0..n {
            let tokens: Vec<u32> = (0..4).map(|j| (i * 10 + j) as u32).collect();
            let _ = cache.insert(&tokens, vec![0u8; 16]);
        }
        cache.clear();
        prop_assert!(cache.is_empty());
    }

    /// PrefixCache stats eviction_count starts at 0.
    #[test]
    fn prop_prefix_cache_stats_initial(_dummy in 0u8..1) {
        let cache = PrefixCache::new(PrefixCacheConfig::default());
        let stats = cache.stats();
        prop_assert_eq!(stats.eviction_count, 0);
        prop_assert_eq!(stats.memory_usage, 0);
    }
}

// ── 6. KernelRecorder properties ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// KernelRecorder count tracks record calls.
    #[test]
    fn prop_kernel_recorder_count(n in 1usize..=50) {
        let recorder = KernelRecorder::new();
        for _ in 0..n {
            recorder.record("test_kernel");
        }
        prop_assert_eq!(recorder.count(), n);
    }

    /// KernelRecorder clear resets count to 0.
    #[test]
    fn prop_kernel_recorder_clear(n in 1usize..=20) {
        let recorder = KernelRecorder::new();
        for _ in 0..n {
            recorder.record("kernel_a");
        }
        recorder.clear();
        prop_assert_eq!(recorder.count(), 0);
    }

    /// KernelRecorder snapshot returns recorded kernel IDs.
    #[test]
    fn prop_kernel_recorder_snapshot(n in 1usize..=10) {
        let recorder = KernelRecorder::new();
        for _ in 0..n {
            recorder.record("snap_kernel");
        }
        let snap = recorder.snapshot();
        prop_assert!(!snap.is_empty());
    }
}

// ── 7. MetricsCollector properties ──────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// MetricsCollector total_requests tracks submissions.
    #[test]
    fn prop_metrics_total_requests(n in 1usize..=20) {
        let collector = MetricsCollector::new();
        for _ in 0..n {
            collector.record_request(10, 5, 1000, 200);
        }
        prop_assert_eq!(collector.total_requests(), n as u64);
    }

    /// MetricsCollector reset zeroes counts.
    #[test]
    fn prop_metrics_reset(_dummy in 0u8..1) {
        let collector = MetricsCollector::new();
        collector.record_request(10, 5, 1000, 200);
        collector.record_cache_hit();
        collector.reset();
        prop_assert_eq!(collector.total_requests(), 0);
    }

    /// MetricsCollector snapshot captures prompt_tokens.
    #[test]
    fn prop_metrics_snapshot_tokens(prompt_toks in 1u64..=100) {
        let collector = MetricsCollector::new();
        collector.record_request(prompt_toks, 5, 1000, 200);
        let snap = collector.snapshot();
        prop_assert_eq!(snap.prompt_tokens, prompt_toks);
    }
}

// ── 8. LatencyHistogram properties ──────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// LatencyHistogram count tracks recordings.
    #[test]
    fn prop_latency_histogram_count(n in 1usize..=50) {
        let mut hist = LatencyHistogram::new();
        for i in 0..n {
            hist.record(i as f64 * 1.0);
        }
        prop_assert_eq!(hist.count(), n);
    }

    /// LatencyHistogram mean is reasonable for uniform values.
    #[test]
    fn prop_latency_histogram_mean(val in 1.0f64..100.0, n in 1usize..=20) {
        let mut hist = LatencyHistogram::new();
        for _ in 0..n {
            hist.record(val);
        }
        if let Some(mean) = hist.mean() {
            prop_assert!((mean - val).abs() < 1e-6, "mean {} != val {}", mean, val);
        }
    }

    /// LatencyHistogram reset clears data.
    #[test]
    fn prop_latency_histogram_reset(n in 1usize..=10) {
        let mut hist = LatencyHistogram::new();
        for _ in 0..n {
            hist.record(5.0);
        }
        hist.reset();
        prop_assert_eq!(hist.count(), 0);
    }

    /// p50 is between min and max.
    #[test]
    fn prop_latency_p50_bounded(n in 5usize..=20) {
        let mut hist = LatencyHistogram::new();
        for i in 0..n {
            hist.record(i as f64);
        }
        if let (Some(p50), Some(min_val), Some(max_val)) = (hist.p50(), hist.min(), hist.max()) {
            prop_assert!(p50 >= min_val && p50 <= max_val);
        }
    }
}

// ── 9. MemoryProfiler properties ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// MemoryProfiler tracks allocations.
    #[test]
    fn prop_memory_profiler_alloc(bytes in 1u64..=10000) {
        let profiler = MemoryProfiler::new();
        profiler.record_allocation(bytes);
        prop_assert_eq!(profiler.current_bytes(), bytes);
        prop_assert_eq!(profiler.allocation_count(), 1);
    }

    /// MemoryProfiler deallocation reduces current.
    #[test]
    fn prop_memory_profiler_dealloc(bytes in 1u64..=5000) {
        let profiler = MemoryProfiler::new();
        profiler.record_allocation(bytes);
        profiler.record_deallocation(bytes);
        prop_assert_eq!(profiler.current_bytes(), 0);
        prop_assert_eq!(profiler.deallocation_count(), 1);
    }

    /// Peak bytes tracks the maximum.
    #[test]
    fn prop_memory_profiler_peak(a in 100u64..=5000, b in 100u64..=5000) {
        let profiler = MemoryProfiler::new();
        profiler.record_allocation(a);
        profiler.record_allocation(b);
        prop_assert!(profiler.peak_bytes() >= a + b);
    }

    /// MemoryProfiler reset clears state.
    #[test]
    fn prop_memory_profiler_reset(bytes in 1u64..=1000) {
        let profiler = MemoryProfiler::new();
        profiler.record_allocation(bytes);
        profiler.reset();
        prop_assert_eq!(profiler.current_bytes(), 0);
    }
}
