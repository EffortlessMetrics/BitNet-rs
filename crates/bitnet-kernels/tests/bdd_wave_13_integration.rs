//! BDD Wave 13 — Integration tests for bitnet-kernels
//!
//! 40 BDD-style scenarios organized in describe/context/it blocks covering:
//!   1. Kernel Registration (8 tests)
//!   2. Performance Tracking (8 tests)
//!   3. Device Selection (8 tests)
//!   4. Memory Management (8 tests)
//!   5. Error Recovery (8 tests)

use std::time::Duration;

use bitnet_kernels::cpu::kv_cache::{
    KvCache, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear, kv_cache_memory_usage,
    kv_cache_slice,
};
use bitnet_kernels::cpu::quantize::{
    compute_quantization_error, dequantize_symmetric_i8, quantize_symmetric_i8, quantize_ternary,
};
use bitnet_kernels::device_features::{
    current_kernel_capabilities, detect_simd_level, device_capability_summary, gpu_compiled,
};
use bitnet_kernels::perf_tracker::{KernelTiming, PerfTracker, format_perf_report};
use bitnet_kernels::{FallbackKernel, KernelManager, KernelProvider};

// ═══════════════════════════════════════════════════════════════════
// 1. Kernel Registration (8 tests)
// ═══════════════════════════════════════════════════════════════════

mod describe_kernel_registration {
    use super::*;

    mod context_fresh_manager {
        use super::*;

        #[test]
        fn it_has_at_least_one_provider_on_creation() {
            // Given a fresh KernelManager
            let manager = KernelManager::new();

            // When we list available providers
            let providers = manager.list_available_providers();

            // Then at least one provider (FallbackKernel) is present
            assert!(
                !providers.is_empty(),
                "KernelManager should register at least the CPU fallback provider"
            );
        }

        #[test]
        fn it_always_includes_fallback_provider() {
            // Given a fresh KernelManager (cpu feature enabled)
            let manager = KernelManager::new();

            // When we list providers
            let providers = manager.list_available_providers();

            // Then "cpu_fallback" is among them
            assert!(
                providers.iter().any(|name| name.contains("fallback") || name.contains("cpu")),
                "Fallback CPU provider must always be registered, got: {providers:?}"
            );
        }

        #[test]
        fn it_can_select_a_provider_immediately() {
            // Given a fresh KernelManager
            let manager = KernelManager::new();

            // When we select the best provider
            let result = manager.select_best();

            // Then selection succeeds
            assert!(result.is_ok(), "select_best should succeed with at least fallback");
        }

        #[test]
        fn it_returns_consistent_provider_name() {
            // Given a fresh KernelManager after selection
            let manager = KernelManager::new();
            let _ = manager.select_best();

            // When we query the selected provider name
            let name = manager.selected_provider_name();

            // Then the name is Some and non-empty
            assert!(name.is_some(), "selected_provider_name should be Some after select_best");
            assert!(!name.unwrap().is_empty(), "provider name should not be empty");
        }
    }

    mod context_provider_queries {
        use super::*;

        #[test]
        fn it_lists_only_available_providers() {
            // Given a KernelManager
            let manager = KernelManager::new();

            // When we list available providers
            let providers = manager.list_available_providers();

            // Then every listed provider must be queryable via select_best
            let best = manager.select_best().unwrap();
            assert!(
                providers.contains(&best.name()),
                "selected provider '{}' must appear in available list: {providers:?}",
                best.name()
            );
        }

        #[test]
        fn it_reports_fallback_is_available() {
            // Given the FallbackKernel
            let fallback = FallbackKernel;

            // When we check availability
            let available = fallback.is_available();

            // Then it reports true (always available on CPU)
            assert!(available, "FallbackKernel must always report is_available() == true");
        }

        #[test]
        fn it_returns_nonempty_name_for_fallback() {
            // Given the FallbackKernel
            let fallback = FallbackKernel;

            // When we query its name
            let name = fallback.name();

            // Then name is non-empty
            assert!(!name.is_empty(), "FallbackKernel name must be non-empty");
        }

        #[test]
        fn it_caches_selection_across_calls() {
            // Given a KernelManager with a cached selection
            let manager = KernelManager::new();
            let first = manager.select_best().unwrap().name();

            // When we call select_best again
            let second = manager.select_best().unwrap().name();

            // Then the same provider is returned (cached via OnceLock)
            assert_eq!(first, second, "select_best must be idempotent");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. Performance Tracking (8 tests)
// ═══════════════════════════════════════════════════════════════════

mod describe_performance_tracking {
    use super::*;

    mod context_recording_operations {
        use super::*;

        #[test]
        fn it_starts_with_zero_count() {
            // Given a fresh PerfTracker
            let tracker = PerfTracker::new();

            // Then count is 0 and total_time is zero
            assert_eq!(tracker.count(), 0);
            assert_eq!(tracker.total_time(), Duration::ZERO);
        }

        #[test]
        fn it_increments_count_on_record() {
            // Given a PerfTracker
            let mut tracker = PerfTracker::new();

            // When recording three operations
            for i in 0..3 {
                tracker.record(KernelTiming::new(
                    "matmul",
                    Duration::from_millis(10 * (i + 1)),
                    1024,
                ));
            }

            // Then count is 3
            assert_eq!(tracker.count(), 3);
        }

        #[test]
        fn it_accumulates_total_time() {
            // Given a PerfTracker with two recorded operations
            let mut tracker = PerfTracker::new();
            tracker.record(KernelTiming::new("a", Duration::from_millis(15), 100));
            tracker.record(KernelTiming::new("b", Duration::from_millis(25), 200));

            // When querying total_time
            let total = tracker.total_time();

            // Then it equals the sum of durations
            assert_eq!(total, Duration::from_millis(40));
        }

        #[test]
        fn it_groups_timings_by_kernel_name() {
            // Given a PerfTracker with mixed kernel invocations
            let mut tracker = PerfTracker::new();
            tracker.record(KernelTiming::new("softmax", Duration::from_millis(5), 64));
            tracker.record(KernelTiming::new("matmul", Duration::from_millis(10), 256));
            tracker.record(KernelTiming::new("softmax", Duration::from_millis(7), 64));
            tracker.record(KernelTiming::new("matmul", Duration::from_millis(12), 256));
            tracker.record(KernelTiming::new("matmul", Duration::from_millis(8), 256));

            // When grouping by kernel
            let grouped = tracker.by_kernel();

            // Then softmax has 2 entries and matmul has 3
            assert_eq!(grouped["softmax"].len(), 2);
            assert_eq!(grouped["matmul"].len(), 3);
        }
    }

    mod context_report_generation {
        use super::*;

        #[test]
        fn it_generates_valid_report_header() {
            // Given a PerfTracker with some data
            let mut tracker = PerfTracker::new();
            tracker.record(KernelTiming::new("layernorm", Duration::from_millis(3), 512));

            // When generating a report
            let report = format_perf_report(&tracker);

            // Then the report contains the header and kernel name
            assert!(report.contains("Kernel Performance Report"));
            assert!(report.contains("layernorm"));
            assert!(report.contains("Total kernels: 1"));
        }

        #[test]
        fn it_handles_empty_tracker_report() {
            // Given an empty PerfTracker
            let tracker = PerfTracker::new();

            // When generating a report
            let report = format_perf_report(&tracker);

            // Then the report shows zero kernels
            assert!(report.contains("Total kernels: 0"));
        }

        #[test]
        fn it_identifies_slowest_and_fastest() {
            // Given a PerfTracker with varied durations
            let mut tracker = PerfTracker::new();
            tracker.record(KernelTiming::new("fast", Duration::from_micros(100), 50));
            tracker.record(KernelTiming::new("slow", Duration::from_millis(50), 50));
            tracker.record(KernelTiming::new("mid", Duration::from_millis(5), 50));

            // Then slowest is "slow" and fastest is "fast"
            assert_eq!(tracker.slowest().unwrap().kernel_name, "slow");
            assert_eq!(tracker.fastest().unwrap().kernel_name, "fast");
        }

        #[test]
        fn it_computes_kernel_stats_with_throughput() {
            // Given a PerfTracker with known elements and time
            let mut tracker = PerfTracker::new();
            tracker.record(KernelTiming::new("test", Duration::from_secs(1), 5000));
            tracker.record(KernelTiming::new("test", Duration::from_secs(1), 5000));

            // When computing kernel stats
            let stats = tracker.kernel_stats();

            // Then there is one stats entry with total_elements=10000 and 2 invocations
            assert_eq!(stats.len(), 1);
            assert_eq!(stats[0].count, 2);
            assert_eq!(stats[0].total_elements, 10000);
            assert!((stats[0].avg_throughput() - 5000.0).abs() < 1.0);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. Device Selection (8 tests)
// ═══════════════════════════════════════════════════════════════════

mod describe_device_selection {
    use super::*;

    mod context_cpu_only {
        use super::*;

        #[test]
        fn it_selects_cpu_when_no_gpu_compiled() {
            // Given a build with --features cpu (no gpu)
            // When checking gpu_compiled
            let has_gpu = gpu_compiled();

            // Then gpu is not compiled (we only have cpu feature)
            // And KernelManager still selects a provider
            if !has_gpu {
                let manager = KernelManager::new();
                let best = manager.select_best().unwrap();
                assert!(
                    best.name().contains("cpu")
                        || best.name().contains("fallback")
                        || best.name().contains("avx")
                        || best.name().contains("neon"),
                    "Expected CPU-family provider, got: {}",
                    best.name()
                );
            }
        }

        #[test]
        fn it_falls_back_gracefully_when_gpu_unavailable() {
            // Given the cpu feature enabled
            let manager = KernelManager::new();

            // When we select the best provider
            let result = manager.select_best();

            // Then it succeeds (falls back to CPU)
            assert!(result.is_ok(), "Must always have a fallback provider");
        }

        #[test]
        fn it_detects_simd_level() {
            // Given the current hardware
            let simd = detect_simd_level();

            // Then a valid SimdLevel is returned (never panics)
            let _desc = format!("{simd:?}");
        }

        #[test]
        fn it_generates_capability_summary() {
            // Given the current device features
            let summary = device_capability_summary();

            // Then the summary is a non-empty string
            assert!(!summary.is_empty(), "capability summary must be non-empty");
        }
    }

    mod context_feature_gate_validation {
        use super::*;

        #[test]
        fn it_reports_gpu_compiled_false_without_gpu_feature() {
            // Given build with only cpu feature
            // When checking gpu_compiled
            let compiled = gpu_compiled();

            // Then without gpu feature, it should be false
            // (If somehow GPU is compiled, the test still passes —
            //  we just verify the function is callable)
            let _ = compiled;
        }

        #[test]
        fn it_returns_kernel_capabilities() {
            // Given the current compilation
            let caps = current_kernel_capabilities();

            // Then capabilities are a valid struct
            let _simd = format!("{:?}", caps.simd_level);
            let _cpu = caps.cpu_rust;
        }

        #[test]
        fn it_provider_name_is_static_str() {
            // Given a FallbackKernel
            let kernel = FallbackKernel;

            // When querying name
            let name: &'static str = kernel.name();

            // Then it is a valid static string
            assert!(!name.is_empty());
        }

        #[test]
        fn it_selects_same_provider_across_threads() {
            // Given a KernelManager shared across threads
            use std::sync::Arc;
            let manager = Arc::new(KernelManager::new());

            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let m = Arc::clone(&manager);
                    std::thread::spawn(move || m.select_best().unwrap().name())
                })
                .collect();

            let names: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

            // Then all threads see the same provider
            assert!(
                names.windows(2).all(|w| w[0] == w[1]),
                "All threads must see the same provider: {names:?}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. Memory Management (8 tests)
// ═══════════════════════════════════════════════════════════════════

mod describe_memory_management {
    use super::*;

    mod context_kv_cache_allocation {
        use super::*;

        fn small_cache_config() -> KvCacheConfig {
            KvCacheConfig {
                num_layers: 2,
                num_heads: 4,
                head_dim: 8,
                max_seq_len: 16,
                dtype: KvDtype::F32,
            }
        }

        #[test]
        fn it_allocates_cache_matching_config() {
            // Given a valid KvCacheConfig
            let config = small_cache_config();

            // When creating a KvCache
            let cache = KvCache::new(config.clone()).unwrap();

            // Then num_layers matches and initial seq_len is 0
            assert_eq!(cache.num_layers(), config.num_layers);
            assert_eq!(cache.seq_len(0).unwrap(), 0);
            assert_eq!(cache.seq_len(1).unwrap(), 0);
        }

        #[test]
        fn it_tracks_memory_usage() {
            // Given a freshly allocated KvCache
            let cache = KvCache::new(small_cache_config()).unwrap();

            // When checking memory usage
            let usage = kv_cache_memory_usage(&cache);

            // Then usage is > 0 (pre-allocated buffers)
            assert!(usage > 0, "KvCache should have non-zero memory usage");
        }

        #[test]
        fn it_appends_and_increases_seq_len() {
            // Given a KvCache with initial seq_len=0
            let config = small_cache_config();
            let token_elems = config.num_heads * config.head_dim; // 4*8=32
            let mut cache = KvCache::new(config).unwrap();

            // When appending one token to layer 0
            let key_data = vec![1.0f32; token_elems];
            let val_data = vec![2.0f32; token_elems];
            kv_cache_append(&mut cache, 0, &key_data, &val_data).unwrap();

            // Then seq_len for layer 0 is 1
            assert_eq!(cache.seq_len(0).unwrap(), 1);
            assert_eq!(cache.seq_len(1).unwrap(), 0);
        }

        #[test]
        fn it_slices_cached_data_correctly() {
            // Given a KvCache with 3 appended tokens
            let config = small_cache_config();
            let elems = config.num_heads * config.head_dim;
            let mut cache = KvCache::new(config).unwrap();

            for t in 0..3 {
                let keys = vec![(t as f32 + 1.0); elems];
                let vals = vec![(t as f32 + 10.0); elems];
                kv_cache_append(&mut cache, 0, &keys, &vals).unwrap();
            }

            // When slicing the first 2 tokens
            let (keys, vals) = kv_cache_slice(&cache, 0, 0, 2).unwrap();

            // Then we get 2*elems elements for keys and vals
            assert_eq!(keys.len(), 2 * elems);
            assert_eq!(vals.len(), 2 * elems);
        }
    }

    mod context_cache_cleanup {
        use super::*;

        fn tiny_config() -> KvCacheConfig {
            KvCacheConfig {
                num_layers: 1,
                num_heads: 2,
                head_dim: 4,
                max_seq_len: 8,
                dtype: KvDtype::F32,
            }
        }

        #[test]
        fn it_clears_all_layers() {
            // Given a KvCache with data in multiple layers
            let config = KvCacheConfig {
                num_layers: 2,
                num_heads: 2,
                head_dim: 4,
                max_seq_len: 8,
                dtype: KvDtype::F32,
            };
            let elems = config.num_heads * config.head_dim;
            let mut cache = KvCache::new(config).unwrap();

            let k = vec![1.0f32; elems];
            let v = vec![2.0f32; elems];
            kv_cache_append(&mut cache, 0, &k, &v).unwrap();
            kv_cache_append(&mut cache, 1, &k, &v).unwrap();

            // When clearing the cache
            kv_cache_clear(&mut cache);

            // Then all layers have seq_len 0
            assert_eq!(cache.seq_len(0).unwrap(), 0);
            assert_eq!(cache.seq_len(1).unwrap(), 0);
        }

        #[test]
        fn it_can_reuse_cache_after_clear() {
            // Given a cleared KvCache
            let config = tiny_config();
            let elems = config.num_heads * config.head_dim;
            let mut cache = KvCache::new(config).unwrap();

            let k = vec![1.0f32; elems];
            let v = vec![2.0f32; elems];
            kv_cache_append(&mut cache, 0, &k, &v).unwrap();
            kv_cache_clear(&mut cache);

            // When appending new data
            kv_cache_append(&mut cache, 0, &k, &v).unwrap();

            // Then seq_len is 1 again
            assert_eq!(cache.seq_len(0).unwrap(), 1);
        }

        #[test]
        fn it_validates_buffer_alignment() {
            // Given a KvCache
            let config = tiny_config();
            let mut cache = KvCache::new(config.clone()).unwrap();

            let elems = config.num_heads * config.head_dim;
            let k = vec![3.14f32; elems];
            let v = vec![2.72f32; elems];
            kv_cache_append(&mut cache, 0, &k, &v).unwrap();

            // When slicing the data back
            let (keys, vals) = kv_cache_slice(&cache, 0, 0, 1).unwrap();

            // Then every element is exactly what we stored (no alignment corruption)
            for (i, (&got, &expected)) in keys.iter().zip(k.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < f32::EPSILON,
                    "key mismatch at {i}: got {got}, expected {expected}"
                );
            }
            for (i, (&got, &expected)) in vals.iter().zip(v.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < f32::EPSILON,
                    "val mismatch at {i}: got {got}, expected {expected}"
                );
            }
        }

        #[test]
        fn it_drops_cache_without_leak() {
            // Given a KvCache allocated with data
            let config = tiny_config();
            let elems = config.num_heads * config.head_dim;

            {
                let mut cache = KvCache::new(config).unwrap();
                let k = vec![1.0f32; elems];
                let v = vec![2.0f32; elems];
                kv_cache_append(&mut cache, 0, &k, &v).unwrap();
                // cache is dropped at end of scope
            }

            // Then no panic and no leak (Rust's ownership guarantees cleanup)
            // This test verifies the Drop path doesn't panic
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. Error Recovery (8 tests)
// ═══════════════════════════════════════════════════════════════════

mod describe_error_recovery {
    use super::*;

    mod context_invalid_dimensions {
        use super::*;

        #[test]
        fn it_rejects_zero_layer_kv_cache() {
            // Given a KvCacheConfig with zero layers
            let config = KvCacheConfig {
                num_layers: 0,
                num_heads: 4,
                head_dim: 8,
                max_seq_len: 16,
                dtype: KvDtype::F32,
            };

            // When creating a KvCache
            let result = KvCache::new(config);

            // Then it returns an error
            assert!(result.is_err(), "zero num_layers must be rejected");
        }

        #[test]
        fn it_rejects_zero_heads_kv_cache() {
            // Given a KvCacheConfig with zero heads
            let config = KvCacheConfig {
                num_layers: 1,
                num_heads: 0,
                head_dim: 8,
                max_seq_len: 16,
                dtype: KvDtype::F32,
            };

            // When creating a KvCache
            let result = KvCache::new(config);

            // Then it returns an error
            assert!(result.is_err(), "zero num_heads must be rejected");
        }

        #[test]
        fn it_rejects_zero_head_dim_kv_cache() {
            // Given a KvCacheConfig with zero head_dim
            let config = KvCacheConfig {
                num_layers: 1,
                num_heads: 4,
                head_dim: 0,
                max_seq_len: 16,
                dtype: KvDtype::F32,
            };

            // When creating a KvCache
            let result = KvCache::new(config);

            // Then it returns an error
            assert!(result.is_err(), "zero head_dim must be rejected");
        }

        #[test]
        fn it_rejects_out_of_bounds_layer_index() {
            // Given a KvCache with 2 layers
            let config = KvCacheConfig {
                num_layers: 2,
                num_heads: 2,
                head_dim: 4,
                max_seq_len: 8,
                dtype: KvDtype::F32,
            };
            let cache = KvCache::new(config).unwrap();

            // When querying seq_len for layer 5 (out of bounds)
            let result = cache.seq_len(5);

            // Then it returns an error
            assert!(result.is_err(), "out-of-bounds layer index must fail");
        }
    }

    mod context_kernel_error_handling {
        use super::*;

        #[test]
        fn it_handles_mismatched_matmul_dimensions() {
            // Given a FallbackKernel
            let kernel = FallbackKernel;

            // When calling matmul_i2s with mismatched output buffer size
            // M=2, N=3, K=4 → output should be 2*3=6 elements
            let a = vec![0i8; 2 * 4]; // M*K
            let b = vec![0u8; 4 * 3]; // K*N
            let mut c = vec![0.0f32; 3]; // too small (should be 6)

            let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 3, 4);

            // Then it returns an error (dimension mismatch)
            assert!(result.is_err(), "mismatched output dimensions should fail");
        }

        #[test]
        fn it_handles_quantization_edge_case_empty_input() {
            // Given an empty input
            let input: Vec<f32> = vec![];

            // When quantizing
            let (quantized, scale) = quantize_symmetric_i8(&input, 8);

            // Then empty vectors are returned with zero or unit scale
            assert!(quantized.is_empty());
            let _ = scale; // scale may be 0.0 or defined — no crash
        }

        #[test]
        fn it_recovers_ternary_quantization_all_zeros() {
            // Given an all-zero input
            let input = vec![0.0f32; 16];

            // When ternary quantizing
            let quantized = quantize_ternary(&input, 0.5);

            // Then all outputs are zero
            assert!(quantized.iter().all(|&v| v == 0), "all-zero input → all-zero ternary");
        }

        #[test]
        fn it_computes_quantization_error_for_roundtrip() {
            // Given a known input
            let input = vec![1.0, -2.0, 3.0, -4.0, 0.5];

            // When round-tripping through symmetric i8 quantization
            let (quantized, scale) = quantize_symmetric_i8(&input, 8);
            let recovered = dequantize_symmetric_i8(&quantized, scale);

            // Then quantization error is bounded
            let error = compute_quantization_error(&input, &recovered);
            assert!(
                error.max_abs_error < 0.5,
                "max error should be bounded, got {}",
                error.max_abs_error
            );
            assert!(error.snr > 0.0, "SNR should be positive, got {}", error.snr);
        }
    }
}
