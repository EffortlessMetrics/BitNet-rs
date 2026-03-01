//! Edge-case tests for the bitnet-ffi C API surface.
//!
//! Tests cover version queries, init/cleanup lifecycle, error handling,
//! config construction, performance metrics, and model operations.

use bitnet_ffi::config::{BitNetCConfig, BitNetCInferenceConfig, BitNetCPerformanceMetrics};
use bitnet_ffi::error::{BitNetCError, clear_last_error, get_last_error, set_last_error};
use bitnet_ffi::{
    BITNET_SUCCESS, bitnet_abi_version, bitnet_cleanup, bitnet_ffi_api_version,
    bitnet_garbage_collect, bitnet_get_memory_usage, bitnet_get_num_threads, bitnet_init,
    bitnet_is_gpu_available, bitnet_model_is_loaded, bitnet_set_num_threads, bitnet_version,
};
use std::ffi::CStr;

// ── Version queries ──────────────────────────────────────────────────

#[test]
fn abi_version_is_nonzero() {
    let v = bitnet_abi_version();
    assert!(v > 0, "ABI version should be > 0, got {v}");
}

#[test]
fn ffi_api_version_is_nonzero() {
    let v = bitnet_ffi_api_version();
    assert!(v > 0, "FFI API version should be > 0, got {v}");
}

#[test]
fn version_string_is_valid_utf8() {
    let ptr = bitnet_version();
    assert!(!ptr.is_null(), "version pointer should not be null");
    let cstr = unsafe { CStr::from_ptr(ptr) };
    let s = cstr.to_str().expect("version should be valid UTF-8");
    assert!(!s.is_empty(), "version string should not be empty");
}

#[test]
fn version_contains_semver_pattern() {
    let ptr = bitnet_version();
    let cstr = unsafe { CStr::from_ptr(ptr) };
    let s = cstr.to_str().unwrap();
    // Should contain at least one dot (X.Y.Z pattern)
    assert!(s.contains('.'), "version '{s}' should contain '.'");
}

// ── Init / Cleanup lifecycle ─────────────────────────────────────────

#[test]
fn init_returns_success() {
    let rc = bitnet_init();
    assert_eq!(rc, BITNET_SUCCESS as i32);
    bitnet_cleanup();
}

#[test]
fn double_init_is_safe() {
    let rc1 = bitnet_init();
    let rc2 = bitnet_init();
    assert_eq!(rc1, BITNET_SUCCESS as i32);
    assert_eq!(rc2, BITNET_SUCCESS as i32);
    bitnet_cleanup();
}

// ── Error handling ───────────────────────────────────────────────────

#[test]
fn error_clear_then_get_returns_none() {
    clear_last_error();
    let err = get_last_error();
    assert!(err.is_none(), "should be None after clear");
}

#[test]
fn error_set_then_get_returns_message() {
    set_last_error(BitNetCError::InvalidArgument("test arg".into()));
    let err = get_last_error();
    assert!(err.is_some(), "should have error after set");
    clear_last_error();
}

#[test]
fn error_clear_is_idempotent() {
    clear_last_error();
    clear_last_error();
    clear_last_error();
    assert!(get_last_error().is_none());
}

#[test]
fn error_variants_display() {
    let errors = vec![
        BitNetCError::InvalidArgument("bad arg".into()),
        BitNetCError::ModelNotFound("missing.gguf".into()),
        BitNetCError::ModelLoadFailed("corrupt".into()),
        BitNetCError::InferenceFailed("oom".into()),
        BitNetCError::OutOfMemory("12GB needed".into()),
        BitNetCError::Internal("unknown".into()),
    ];
    for err in errors {
        let msg = format!("{err:?}");
        assert!(!msg.is_empty());
    }
}

// ── Config construction ──────────────────────────────────────────────

#[test]
fn config_default_has_sane_values() {
    let cfg = BitNetCConfig::default();
    assert!(cfg.model_path.is_null());
    assert_eq!(cfg.num_threads, 0);
}

#[test]
fn inference_config_default_values() {
    let cfg = BitNetCInferenceConfig::default();
    // Default inference config should have sensible defaults
    let _ = format!("{cfg:?}");
}

#[test]
fn performance_metrics_default_zeroed() {
    let m = BitNetCPerformanceMetrics::default();
    assert_eq!(m.tokens_generated, 0);
    assert_eq!(m.prompt_tokens, 0);
    assert_eq!(m.tokens_per_second, 0.0);
}

#[test]
fn performance_metrics_debug_display() {
    let m = BitNetCPerformanceMetrics::default();
    let s = format!("{m:?}");
    assert!(s.contains("BitNetCPerformanceMetrics"));
}

// ── Thread management ────────────────────────────────────────────────

#[test]
fn get_num_threads_returns_positive() {
    let n = bitnet_get_num_threads();
    assert!(n > 0, "thread count should be > 0, got {n}");
}

#[test]
fn set_num_threads_roundtrip() {
    let rc = bitnet_set_num_threads(2);
    assert_eq!(rc, BITNET_SUCCESS as i32);
    let n = bitnet_get_num_threads();
    assert_eq!(n, 2, "thread count should be 2 after set");
}

// ── Memory & GPU ─────────────────────────────────────────────────────

#[test]
fn memory_usage_is_sane() {
    let usage = bitnet_get_memory_usage();
    assert!(usage < 100_000_000_000, "usage too large: {usage}");
}

#[test]
fn garbage_collect_returns_success() {
    let rc = bitnet_garbage_collect();
    assert_eq!(rc, BITNET_SUCCESS as i32);
}

#[test]
fn gpu_available_returns_valid_bool() {
    let rc = bitnet_is_gpu_available();
    assert!(rc == 0 || rc == 1, "expected 0 or 1, got {rc}");
}

// ── Model operations (no model loaded) ───────────────────────────────

#[test]
fn model_not_loaded_by_default() {
    // Model ID 0 should not be loaded
    let loaded = bitnet_model_is_loaded(0);
    assert_eq!(loaded, 0, "no model should be loaded for ID 0");
}

#[test]
fn model_not_loaded_negative_id() {
    let loaded = bitnet_model_is_loaded(-1);
    // Negative ID returns -1 (error) or 0 (not loaded)
    assert!(loaded <= 0, "negative model ID should not return 'loaded'");
}

#[test]
fn model_not_loaded_large_id() {
    let loaded = bitnet_model_is_loaded(999999);
    assert_eq!(loaded, 0, "large model ID should not be loaded");
}
