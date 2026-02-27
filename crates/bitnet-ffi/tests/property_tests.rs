//! Property-based tests for bitnet-ffi safe wrapper invariants.
//!
//! Covers:
//! - `BitNetCConfig` model-format / quant-type encoding and round-trip through Rust config
//! - `BitNetCInferenceConfig` validation boundary invariants
//! - `BitNetCError` Display / Debug invariants (non-empty, contains message)
//! - `MemoryStats` arithmetic invariants (has_leaks, leaked_allocations, MB conversions)
//! - Thread-local error-state consistency

use bitnet_ffi::{
    BitNetCConfig, BitNetCError, BitNetCInferenceConfig, BitNetCPerformanceMetrics, MemoryStats,
    ThreadPoolConfig, clear_last_error, get_last_error, set_last_error,
};
use proptest::prelude::*;

// ─── BitNetCConfig round-trip helpers ────────────────────────────────────────

/// Build a minimal valid `BitNetCConfig` with a null model path and controlled
/// format / quant-type fields so `to_bitnet_config()` succeeds.
fn make_cconfig(model_format: u32, quantization_type: u32) -> BitNetCConfig {
    BitNetCConfig { model_format, quantization_type, ..BitNetCConfig::default() }
}

// ─── BitNetCConfig tests ──────────────────────────────────────────────────────

proptest! {
    /// Any valid model-format code (0=GGUF, 1=SafeTensors, 2=HuggingFace) combined with
    /// any valid quant-type code (0=I2S, 1=TL1, 2=TL2) must convert without error and
    /// round-trip the format/type fields back through `from_bitnet_config`.
    #[test]
    fn prop_valid_format_qt_roundtrip(fmt in 0u32..3u32, qt in 0u32..3u32) {
        let c_cfg = make_cconfig(fmt, qt);
        let rust_cfg = c_cfg.to_bitnet_config();
        prop_assert!(
            rust_cfg.is_ok(),
            "Expected success for format={fmt} qt={qt}, got: {:?}", rust_cfg
        );
        let c_cfg2 = BitNetCConfig::from_bitnet_config(&rust_cfg.unwrap());
        prop_assert_eq!(c_cfg2.model_format, fmt);
        prop_assert_eq!(c_cfg2.quantization_type, qt);
    }

    /// Any model-format code ≥ 3 must produce an `InvalidArgument` error.
    #[test]
    fn prop_invalid_model_format_always_errors(fmt in 3u32..=u32::MAX) {
        let c_cfg = make_cconfig(fmt, 0);
        prop_assert!(
            c_cfg.to_bitnet_config().is_err(),
            "Expected error for model_format={fmt}"
        );
    }

    /// Any quant-type code ≥ 3 must produce an `InvalidArgument` error.
    #[test]
    fn prop_invalid_quant_type_always_errors(qt in 3u32..=u32::MAX) {
        let c_cfg = make_cconfig(0, qt);
        prop_assert!(
            c_cfg.to_bitnet_config().is_err(),
            "Expected error for quantization_type={qt}"
        );
    }

    /// Numeric fields (vocab_size, hidden_size, batch_size) round-trip exactly.
    #[test]
    fn prop_numeric_fields_roundtrip(
        vocab      in 1u32..65536u32,
        hidden     in 1u32..65536u32,
        batch_size in 1u32..=256u32,
    ) {
        let c_cfg = BitNetCConfig {
            vocab_size: vocab,
            hidden_size: hidden,
            batch_size,
            ..BitNetCConfig::default()
        };
        let rust_cfg = c_cfg.to_bitnet_config().expect("valid config");
        let c_cfg2 = BitNetCConfig::from_bitnet_config(&rust_cfg);
        prop_assert_eq!(c_cfg2.vocab_size, vocab);
        prop_assert_eq!(c_cfg2.hidden_size, hidden);
        prop_assert_eq!(c_cfg2.batch_size, batch_size);
    }

    /// `use_gpu` round-trips as a boolean: 0→false→0, 1→true→1.
    #[test]
    fn prop_use_gpu_roundtrip(use_gpu in 0u32..=1u32) {
        let c_cfg = BitNetCConfig { use_gpu, ..BitNetCConfig::default() };
        let rust_cfg = c_cfg.to_bitnet_config().unwrap();
        let c_cfg2 = BitNetCConfig::from_bitnet_config(&rust_cfg);
        prop_assert_eq!(c_cfg2.use_gpu, use_gpu);
    }

    /// Non-zero `block_size` round-trips exactly through Rust config.
    #[test]
    fn prop_nonzero_block_size_roundtrip(block in 1u32..=1024u32) {
        let c_cfg = BitNetCConfig { block_size: block, ..BitNetCConfig::default() };
        let rust_cfg = c_cfg.to_bitnet_config().unwrap();
        prop_assert_eq!(rust_cfg.quantization.block_size, block as usize);
    }

    /// `memory_limit=0` maps to `None` (no limit); any non-zero value round-trips.
    #[test]
    fn prop_memory_limit_roundtrip(mem_limit in 1u64..=(4 * 1024 * 1024 * 1024u64)) {
        let c_cfg = BitNetCConfig {
            memory_limit: mem_limit,
            ..BitNetCConfig::default()
        };
        let rust_cfg = c_cfg.to_bitnet_config().unwrap();
        prop_assert_eq!(rust_cfg.performance.memory_limit, Some(mem_limit as usize));
        // round-trip back
        let c_cfg2 = BitNetCConfig::from_bitnet_config(&rust_cfg);
        prop_assert_eq!(c_cfg2.memory_limit, mem_limit);
    }
}

// ─── BitNetCInferenceConfig validation invariants ────────────────────────────

proptest! {
    /// Configs within all valid ranges must pass `validate()`.
    #[test]
    fn prop_valid_inference_config_passes(
        temperature        in 1e-4f32..100.0f32,
        top_p              in 0.0f32..=1.0f32,
        repetition_penalty in 1e-4f32..10.0f32,
        backend            in 0u32..=2u32,
        max_length         in 1u32..4096u32,
        max_new_tokens     in 1u32..512u32,
        stream_buffer_size in 1u32..=256u32,
    ) {
        let cfg = BitNetCInferenceConfig {
            temperature,
            top_p,
            repetition_penalty,
            backend_preference: backend,
            max_length,
            max_new_tokens,
            stream_buffer_size,
            ..BitNetCInferenceConfig::default()
        };
        prop_assert!(
            cfg.validate().is_ok(),
            "Expected valid config to pass, got: {:?}", cfg.validate()
        );
    }

    /// Temperature ≤ 0 must always fail validation.
    #[test]
    fn prop_non_positive_temperature_fails(temp in -1000.0f32..=0.0f32) {
        let cfg = BitNetCInferenceConfig { temperature: temp, ..BitNetCInferenceConfig::default() };
        prop_assert!(
            cfg.validate().is_err(),
            "Expected error for temperature={temp}"
        );
    }

    /// `top_p` > 1.0 must always fail validation.
    #[test]
    fn prop_top_p_above_one_fails(excess in 0.001f32..100.0f32) {
        let cfg = BitNetCInferenceConfig {
            top_p: 1.0 + excess,
            ..BitNetCInferenceConfig::default()
        };
        prop_assert!(cfg.validate().is_err(), "Expected error for top_p={}", 1.0 + excess);
    }

    /// `top_p` < 0.0 must always fail validation.
    #[test]
    fn prop_top_p_below_zero_fails(deficit in 0.001f32..100.0f32) {
        let cfg = BitNetCInferenceConfig {
            top_p: -deficit,
            ..BitNetCInferenceConfig::default()
        };
        prop_assert!(cfg.validate().is_err(), "Expected error for top_p={}", -deficit);
    }

    /// `repetition_penalty` ≤ 0 must always fail validation.
    #[test]
    fn prop_non_positive_repetition_penalty_fails(rp in -1000.0f32..=0.0f32) {
        let cfg = BitNetCInferenceConfig {
            repetition_penalty: rp,
            ..BitNetCInferenceConfig::default()
        };
        prop_assert!(
            cfg.validate().is_err(),
            "Expected error for repetition_penalty={rp}"
        );
    }

    /// `backend_preference` > 2 must always fail validation.
    #[test]
    fn prop_invalid_backend_preference_fails(backend in 3u32..=u32::MAX) {
        let cfg = BitNetCInferenceConfig {
            backend_preference: backend,
            ..BitNetCInferenceConfig::default()
        };
        prop_assert!(cfg.validate().is_err(), "Expected error for backend_preference={backend}");
    }

    /// `stream_buffer_size` = 0 must fail; any positive value combined with otherwise-valid
    /// fields must pass.
    #[test]
    fn prop_stream_buffer_size_must_be_positive(size in 1u32..=1024u32) {
        // zero fails
        let cfg_zero = BitNetCInferenceConfig {
            stream_buffer_size: 0,
            ..BitNetCInferenceConfig::default()
        };
        assert!(cfg_zero.validate().is_err());

        // positive passes (default everything else is valid)
        let cfg_ok = BitNetCInferenceConfig {
            stream_buffer_size: size,
            ..BitNetCInferenceConfig::default()
        };
        prop_assert!(cfg_ok.validate().is_ok());
    }
}

// ─── BitNetCError Display / Debug invariants ─────────────────────────────────

fn all_variants_with(msg: &str) -> Vec<BitNetCError> {
    vec![
        BitNetCError::InvalidArgument(msg.to_owned()),
        BitNetCError::ModelNotFound(msg.to_owned()),
        BitNetCError::ModelLoadFailed(msg.to_owned()),
        BitNetCError::InferenceFailed(msg.to_owned()),
        BitNetCError::OutOfMemory(msg.to_owned()),
        BitNetCError::ThreadSafety(msg.to_owned()),
        BitNetCError::InvalidModelId(msg.to_owned()),
        BitNetCError::ContextLengthExceeded(msg.to_owned()),
        BitNetCError::UnsupportedOperation(msg.to_owned()),
        BitNetCError::Internal(msg.to_owned()),
    ]
}

proptest! {
    /// `Display` output is non-empty for every variant regardless of inner message.
    #[test]
    fn prop_error_display_non_empty(msg in ".*") {
        for err in all_variants_with(&msg) {
            let display = format!("{err}");
            prop_assert!(!display.is_empty(), "Display was empty for {:?}", err);
        }
    }

    /// `Display` always contains the inner message when the message is non-empty ASCII.
    #[test]
    fn prop_error_display_contains_message(msg in "[a-zA-Z0-9_]{1,50}") {
        for err in all_variants_with(&msg) {
            let display = format!("{err}");
            prop_assert!(
                display.contains(&msg),
                "Display '{display}' does not contain message '{msg}'"
            );
        }
    }

    /// `Debug` output is non-empty for every variant.
    #[test]
    fn prop_error_debug_non_empty(msg in ".*") {
        for err in all_variants_with(&msg) {
            let debug = format!("{err:?}");
            prop_assert!(!debug.is_empty());
        }
    }

    /// `Display` output contains a human-readable prefix (the variant label word).
    #[test]
    fn prop_error_display_has_prefix(msg in "[a-zA-Z0-9]{1,20}") {
        let expected_prefixes: &[(&BitNetCError, &str)] = &[
            (&BitNetCError::InvalidArgument(msg.clone()), "Invalid argument"),
            (&BitNetCError::ModelNotFound(msg.clone()),   "Model not found"),
            (&BitNetCError::ModelLoadFailed(msg.clone()), "Model loading failed"),
            (&BitNetCError::InferenceFailed(msg.clone()), "Inference failed"),
            (&BitNetCError::OutOfMemory(msg.clone()),     "Out of memory"),
            (&BitNetCError::ThreadSafety(msg.clone()),    "Thread safety"),
            (&BitNetCError::InvalidModelId(msg.clone()),  "Invalid model ID"),
            (&BitNetCError::Internal(msg.clone()),        "Internal error"),
        ];
        for (err, prefix) in expected_prefixes {
            let display = format!("{err}");
            prop_assert!(
                display.starts_with(prefix),
                "Expected prefix '{prefix}' in '{display}'"
            );
        }
    }
}

// ─── MemoryStats invariants ───────────────────────────────────────────────────

proptest! {
    /// `has_leaks()` is true iff allocation_count ≠ deallocation_count.
    #[test]
    fn prop_has_leaks_iff_counts_differ(alloc in 0usize..10_000usize, dealloc in 0usize..10_000usize) {
        let stats = MemoryStats {
            allocation_count: alloc,
            deallocation_count: dealloc,
            ..MemoryStats::default()
        };
        prop_assert_eq!(stats.has_leaks(), alloc != dealloc);
    }

    /// `leaked_allocations()` equals `alloc.saturating_sub(dealloc)`.
    #[test]
    fn prop_leaked_allocations_is_saturating_diff(
        alloc   in 0usize..10_000usize,
        dealloc in 0usize..10_000usize,
    ) {
        let stats = MemoryStats {
            allocation_count: alloc,
            deallocation_count: dealloc,
            ..MemoryStats::default()
        };
        prop_assert_eq!(stats.leaked_allocations(), alloc.saturating_sub(dealloc));
    }

    /// `current_usage_mb()` is consistent with `current_usage` bytes.
    #[test]
    fn prop_current_usage_mb_consistent(current_usage in 0usize..=(1024 * 1024 * 1024)) {
        let stats = MemoryStats { current_usage, ..MemoryStats::default() };
        let expected = current_usage as f64 / (1024.0 * 1024.0);
        prop_assert!(
            (stats.current_usage_mb() - expected).abs() < 1e-9,
            "current_usage_mb mismatch: got {}, expected {expected}", stats.current_usage_mb()
        );
    }

    /// `peak_usage_mb()` is consistent with `peak_usage` bytes.
    #[test]
    fn prop_peak_usage_mb_consistent(peak_usage in 0usize..=(1024 * 1024 * 1024)) {
        let stats = MemoryStats { peak_usage, ..MemoryStats::default() };
        let expected = peak_usage as f64 / (1024.0 * 1024.0);
        prop_assert!(
            (stats.peak_usage_mb() - expected).abs() < 1e-9,
            "peak_usage_mb mismatch: got {}, expected {expected}", stats.peak_usage_mb()
        );
    }

    /// A fresh `MemoryStats::default()` has no leaks and zero MB usage.
    #[test]
    fn prop_default_stats_clean(_dummy in 0u8..1u8) {
        let stats = MemoryStats::default();
        assert!(!stats.has_leaks());
        assert_eq!(stats.leaked_allocations(), 0);
        assert_eq!(stats.current_usage_mb(), 0.0);
        assert_eq!(stats.peak_usage_mb(), 0.0);
    }
}

// ─── Thread-local error-state invariants ─────────────────────────────────────

proptest! {
    /// set → get → clear cycle is self-consistent for every error message.
    #[test]
    fn prop_error_state_set_get_clear(msg in "[a-zA-Z0-9_ ]{1,50}") {
        // Start clean (thread-local, safe to mutate in place)
        clear_last_error();
        assert!(get_last_error().is_none(), "Expected no error after clear");

        set_last_error(BitNetCError::InvalidArgument(msg.clone()));
        let retrieved = get_last_error();
        prop_assert!(retrieved.is_some(), "Expected error after set");

        let display = format!("{}", retrieved.unwrap());
        prop_assert!(
            display.contains(&msg),
            "Retrieved error display '{display}' should contain '{msg}'"
        );

        clear_last_error();
        prop_assert!(get_last_error().is_none(), "Expected no error after second clear");
    }

    /// After clearing, a second `set_last_error` always overwrites the previous value.
    #[test]
    fn prop_error_state_overwrite(msg1 in "[a-z]{3,20}", msg2 in "[A-Z]{3,20}") {
        prop_assume!(msg1 != msg2); // ensure they're distinguishable

        clear_last_error();
        set_last_error(BitNetCError::Internal(msg1.clone()));
        set_last_error(BitNetCError::Internal(msg2.clone()));

        let display = format!("{}", get_last_error().expect("error should be set"));
        // The second set should have replaced the first
        prop_assert!(display.contains(&msg2), "Expected msg2 '{msg2}' in '{display}'");
        clear_last_error();
    }
}

// ─── Fixed unit tests for zero-boundary edge cases ───────────────────────────

#[test]
fn test_block_size_zero_becomes_default_64() {
    let c_cfg = BitNetCConfig { block_size: 0, ..BitNetCConfig::default() };
    let rust_cfg = c_cfg.to_bitnet_config().expect("should succeed with block_size=0");
    assert_eq!(rust_cfg.quantization.block_size, 64, "block_size=0 should coerce to 64");
}

#[test]
fn test_zero_max_length_fails_validation() {
    let cfg = BitNetCInferenceConfig { max_length: 0, ..BitNetCInferenceConfig::default() };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_zero_max_new_tokens_fails_validation() {
    let cfg = BitNetCInferenceConfig { max_new_tokens: 0, ..BitNetCInferenceConfig::default() };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_null_model_path_becomes_none() {
    let c_cfg = BitNetCConfig { model_path: std::ptr::null(), ..BitNetCConfig::default() };
    let rust_cfg = c_cfg.to_bitnet_config().expect("null path should be accepted");
    assert!(rust_cfg.model.path.is_none());
}

#[test]
fn test_memory_limit_zero_becomes_none() {
    let c_cfg = BitNetCConfig { memory_limit: 0, ..BitNetCConfig::default() };
    let rust_cfg = c_cfg.to_bitnet_config().unwrap();
    assert!(rust_cfg.performance.memory_limit.is_none());
}

#[test]
fn test_num_threads_zero_becomes_none() {
    let c_cfg = BitNetCConfig { num_threads: 0, ..BitNetCConfig::default() };
    let rust_cfg = c_cfg.to_bitnet_config().unwrap();
    assert!(rust_cfg.performance.num_threads.is_none());
}

// ── BitNetCPerformanceMetrics conversion invariants ───────────────────────────

proptest! {
    /// tokens_per_second, latency_ms and memory_usage_mb round-trip through
    /// `from_performance_metrics` (within f32 precision).
    #[test]
    fn prop_perf_metrics_scalar_fields_preserved(
        tps    in 0.0f64..100_000.0f64,
        lat    in 0.0f64..100_000.0f64,
        mem_mb in 0.0f64..100_000.0f64,
    ) {
        let rust = bitnet_common::PerformanceMetrics {
            tokens_per_second: tps,
            latency_ms: lat,
            memory_usage_mb: mem_mb,
            gpu_utilization: None,
            computation_type: bitnet_common::ComputationType::Real,
        };
        let c = BitNetCPerformanceMetrics::from_performance_metrics(&rust);
        // f64 → f32 cast: allow generous epsilon to avoid spurious failures near f32 max
        let eps = |v: f64| (v.abs() as f32) * 1e-5 + 1e-5;
        prop_assert!(
            (c.tokens_per_second - tps as f32).abs() <= eps(tps),
            "tokens_per_second mismatch: C={}, Rust={tps}", c.tokens_per_second
        );
        prop_assert!(
            (c.latency_ms - lat as f32).abs() <= eps(lat),
            "latency_ms mismatch: C={}, Rust={lat}", c.latency_ms
        );
        prop_assert!(
            (c.memory_usage_mb - mem_mb as f32).abs() <= eps(mem_mb),
            "memory_usage_mb mismatch: C={}, Rust={mem_mb}", c.memory_usage_mb
        );
    }

    /// When gpu_utilization is `None` the C field must be exactly -1.0 (sentinel).
    #[test]
    fn prop_perf_metrics_no_gpu_yields_sentinel(tps in 0.0f64..1.0f64) {
        let rust = bitnet_common::PerformanceMetrics {
            tokens_per_second: tps,
            latency_ms: 1.0,
            memory_usage_mb: 1.0,
            gpu_utilization: None,
            computation_type: bitnet_common::ComputationType::Real,
        };
        let c = BitNetCPerformanceMetrics::from_performance_metrics(&rust);
        prop_assert_eq!(
            c.gpu_utilization, -1.0_f32,
            "None GPU utilization must map to -1.0 sentinel"
        );
    }

    /// When gpu_utilization is `Some(x)` the C field is not -1.0 and preserves the value.
    #[test]
    fn prop_perf_metrics_gpu_some_preserved(pct in 0.0f64..100.0f64) {
        let rust = bitnet_common::PerformanceMetrics {
            tokens_per_second: 1.0,
            latency_ms: 1.0,
            memory_usage_mb: 1.0,
            gpu_utilization: Some(pct),
            computation_type: bitnet_common::ComputationType::Real,
        };
        let c = BitNetCPerformanceMetrics::from_performance_metrics(&rust);
        prop_assert_ne!(c.gpu_utilization, -1.0_f32,
            "Some GPU utilization must not produce the -1.0 sentinel");
        let eps = (pct.abs() as f32) * 1e-5 + 1e-5;
        prop_assert!(
            (c.gpu_utilization - pct as f32).abs() <= eps,
            "gpu_utilization mismatch: C={}, Rust={pct}", c.gpu_utilization
        );
    }
}

// ── ThreadPoolConfig field preservation ──────────────────────────────────────

proptest! {
    /// Any positive num_threads is preserved in the config struct.
    #[test]
    fn prop_thread_pool_config_num_threads_preserved(n in 1usize..=256usize) {
        let cfg = ThreadPoolConfig { num_threads: n, ..ThreadPoolConfig::default() };
        prop_assert_eq!(cfg.num_threads, n);
        prop_assert!(cfg.num_threads > 0);
    }

    /// Any positive max_queue_size is preserved in the config struct.
    #[test]
    fn prop_thread_pool_config_queue_size_preserved(q in 1usize..=100_000usize) {
        let cfg = ThreadPoolConfig { max_queue_size: q, ..ThreadPoolConfig::default() };
        prop_assert_eq!(cfg.max_queue_size, q);
        prop_assert!(cfg.max_queue_size > 0);
    }
}

/// Default ThreadPoolConfig must have at least one worker thread.
#[test]
fn test_thread_pool_default_positive_threads() {
    let cfg = ThreadPoolConfig::default();
    assert!(cfg.num_threads > 0, "default num_threads must be > 0");
    assert!(cfg.max_queue_size > 0, "default max_queue_size must be > 0");
}
