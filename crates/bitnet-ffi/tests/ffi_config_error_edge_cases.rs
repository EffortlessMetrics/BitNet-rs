//! Edge-case tests for bitnet-ffi config and error modules.
//!
//! Coverage:
//! - BitNetCConfig default values and to_bitnet_config roundtrip
//! - BitNetCConfig error cases (invalid model_format, quant type)
//! - BitNetCConfig from_bitnet_config reverse mapping
//! - BitNetCInferenceConfig validation edge cases
//! - BitNetCInferenceConfig to_generation_config
//! - BitNetCError Display impl for all variants
//! - BitNetCError From<BitNetError> conversion
//! - Thread-local error state (set/get/clear)
//! - BitNetCPerformanceMetrics defaults

use bitnet_ffi::config::{BitNetCConfig, BitNetCInferenceConfig, BitNetCPerformanceMetrics};
use bitnet_ffi::error::{BitNetCError, clear_last_error, get_last_error, set_last_error};

// ---------------------------------------------------------------------------
// BitNetCConfig — defaults
// ---------------------------------------------------------------------------

#[test]
fn config_default_model_format_is_gguf() {
    let cfg = BitNetCConfig::default();
    assert_eq!(cfg.model_format, 0);
}

#[test]
fn config_default_vocab_size() {
    let cfg = BitNetCConfig::default();
    assert_eq!(cfg.vocab_size, 32000);
}

#[test]
fn config_default_hidden_size() {
    let cfg = BitNetCConfig::default();
    assert_eq!(cfg.hidden_size, 4096);
}

#[test]
fn config_default_num_layers() {
    let cfg = BitNetCConfig::default();
    assert_eq!(cfg.num_layers, 32);
}

#[test]
fn config_default_num_heads() {
    let cfg = BitNetCConfig::default();
    assert_eq!(cfg.num_heads, 32);
}

#[test]
fn config_default_model_path_is_null() {
    let cfg = BitNetCConfig::default();
    assert!(cfg.model_path.is_null());
}

#[test]
fn config_default_num_threads_is_auto() {
    let cfg = BitNetCConfig::default();
    assert_eq!(cfg.num_threads, 0);
}

#[test]
fn config_default_use_gpu_is_false() {
    let cfg = BitNetCConfig::default();
    assert_eq!(cfg.use_gpu, 0);
}

#[test]
fn config_default_memory_limit_is_unlimited() {
    let cfg = BitNetCConfig::default();
    assert_eq!(cfg.memory_limit, 0);
}

// ---------------------------------------------------------------------------
// BitNetCConfig — to_bitnet_config
// ---------------------------------------------------------------------------

#[test]
fn config_to_bitnet_config_default_succeeds() {
    let cfg = BitNetCConfig::default();
    let result = cfg.to_bitnet_config();
    assert!(result.is_ok());
    let bc = result.unwrap();
    assert_eq!(bc.model.vocab_size, 32000);
    assert_eq!(bc.model.hidden_size, 4096);
    assert_eq!(bc.model.num_layers, 32);
    assert_eq!(bc.model.num_heads, 32);
}

#[test]
fn config_to_bitnet_config_safetensors_format() {
    let mut cfg = BitNetCConfig::default();
    cfg.model_format = 1; // SafeTensors
    let bc = cfg.to_bitnet_config().unwrap();
    assert_eq!(bc.model.format, bitnet_common::ModelFormat::SafeTensors);
}

#[test]
fn config_to_bitnet_config_huggingface_format() {
    let mut cfg = BitNetCConfig::default();
    cfg.model_format = 2; // HuggingFace
    let bc = cfg.to_bitnet_config().unwrap();
    assert_eq!(bc.model.format, bitnet_common::ModelFormat::HuggingFace);
}

#[test]
fn config_to_bitnet_config_invalid_format_fails() {
    let mut cfg = BitNetCConfig::default();
    cfg.model_format = 99;
    let result = cfg.to_bitnet_config();
    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("Invalid model format"));
}

#[test]
fn config_to_bitnet_config_invalid_quant_type_fails() {
    let mut cfg = BitNetCConfig::default();
    cfg.quantization_type = 42;
    let result = cfg.to_bitnet_config();
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("Invalid quantization type"));
}

#[test]
fn config_to_bitnet_config_tl1_quant() {
    let mut cfg = BitNetCConfig::default();
    cfg.quantization_type = 1; // TL1
    let bc = cfg.to_bitnet_config().unwrap();
    assert_eq!(bc.quantization.quantization_type, bitnet_common::QuantizationType::TL1);
}

#[test]
fn config_to_bitnet_config_tl2_quant() {
    let mut cfg = BitNetCConfig::default();
    cfg.quantization_type = 2; // TL2
    let bc = cfg.to_bitnet_config().unwrap();
    assert_eq!(bc.quantization.quantization_type, bitnet_common::QuantizationType::TL2);
}

#[test]
fn config_to_bitnet_config_null_model_path() {
    let cfg = BitNetCConfig::default();
    let bc = cfg.to_bitnet_config().unwrap();
    assert!(bc.model.path.is_none());
}

#[test]
fn config_to_bitnet_config_with_model_path() {
    let path = std::ffi::CString::new("/tmp/model.gguf").unwrap();
    let mut cfg = BitNetCConfig::default();
    cfg.model_path = path.as_ptr();
    let bc = cfg.to_bitnet_config().unwrap();
    assert_eq!(bc.model.path.as_ref().unwrap().to_str().unwrap(), "/tmp/model.gguf");
}

#[test]
fn config_to_bitnet_config_auto_threads() {
    let cfg = BitNetCConfig::default();
    let bc = cfg.to_bitnet_config().unwrap();
    assert!(bc.performance.num_threads.is_none());
}

#[test]
fn config_to_bitnet_config_explicit_threads() {
    let mut cfg = BitNetCConfig::default();
    cfg.num_threads = 4;
    let bc = cfg.to_bitnet_config().unwrap();
    assert_eq!(bc.performance.num_threads, Some(4));
}

#[test]
fn config_to_bitnet_config_gpu_enabled() {
    let mut cfg = BitNetCConfig::default();
    cfg.use_gpu = 1;
    let bc = cfg.to_bitnet_config().unwrap();
    assert!(bc.performance.use_gpu);
}

#[test]
fn config_to_bitnet_config_zero_block_size_defaults() {
    let mut cfg = BitNetCConfig::default();
    cfg.block_size = 0;
    let bc = cfg.to_bitnet_config().unwrap();
    assert_eq!(bc.quantization.block_size, 64);
}

// ---------------------------------------------------------------------------
// BitNetCConfig — from_bitnet_config roundtrip
// ---------------------------------------------------------------------------

#[test]
fn config_roundtrip() {
    let original = BitNetCConfig::default();
    let bc = original.to_bitnet_config().unwrap();
    let roundtrip = BitNetCConfig::from_bitnet_config(&bc);
    assert_eq!(roundtrip.model_format, original.model_format);
    assert_eq!(roundtrip.vocab_size, original.vocab_size);
    assert_eq!(roundtrip.hidden_size, original.hidden_size);
    assert_eq!(roundtrip.num_layers, original.num_layers);
    assert_eq!(roundtrip.num_heads, original.num_heads);
    assert_eq!(roundtrip.quantization_type, original.quantization_type);
}

#[test]
fn config_debug_impl() {
    let cfg = BitNetCConfig::default();
    let dbg = format!("{cfg:?}");
    assert!(dbg.contains("BitNetCConfig"));
}

// ---------------------------------------------------------------------------
// BitNetCInferenceConfig — defaults
// ---------------------------------------------------------------------------

#[test]
fn inference_config_default_max_length() {
    let cfg = BitNetCInferenceConfig::default();
    assert_eq!(cfg.max_length, 2048);
}

#[test]
fn inference_config_default_temperature() {
    let cfg = BitNetCInferenceConfig::default();
    assert!((cfg.temperature - 1.0).abs() < f32::EPSILON);
}

#[test]
fn inference_config_default_top_k() {
    let cfg = BitNetCInferenceConfig::default();
    assert_eq!(cfg.top_k, 50);
}

#[test]
fn inference_config_default_validates() {
    let cfg = BitNetCInferenceConfig::default();
    assert!(cfg.validate().is_ok());
}

// ---------------------------------------------------------------------------
// BitNetCInferenceConfig — validation edge cases
// ---------------------------------------------------------------------------

#[test]
fn inference_config_zero_max_length_fails() {
    let mut cfg = BitNetCInferenceConfig::default();
    cfg.max_length = 0;
    let err = cfg.validate().unwrap_err();
    assert!(format!("{err}").contains("max_length"));
}

#[test]
fn inference_config_zero_max_new_tokens_fails() {
    let mut cfg = BitNetCInferenceConfig::default();
    cfg.max_new_tokens = 0;
    let err = cfg.validate().unwrap_err();
    assert!(format!("{err}").contains("max_new_tokens"));
}

#[test]
fn inference_config_zero_temperature_fails() {
    let mut cfg = BitNetCInferenceConfig::default();
    cfg.temperature = 0.0;
    let err = cfg.validate().unwrap_err();
    assert!(format!("{err}").contains("temperature"));
}

#[test]
fn inference_config_negative_temperature_fails() {
    let mut cfg = BitNetCInferenceConfig::default();
    cfg.temperature = -1.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn inference_config_top_p_out_of_range_fails() {
    let mut cfg = BitNetCInferenceConfig::default();
    cfg.top_p = 1.5;
    assert!(cfg.validate().is_err());

    cfg.top_p = -0.1;
    assert!(cfg.validate().is_err());
}

#[test]
fn inference_config_top_p_boundary_values() {
    let mut cfg = BitNetCInferenceConfig::default();
    cfg.top_p = 0.0;
    assert!(cfg.validate().is_ok());

    cfg.top_p = 1.0;
    assert!(cfg.validate().is_ok());
}

#[test]
fn inference_config_zero_repetition_penalty_fails() {
    let mut cfg = BitNetCInferenceConfig::default();
    cfg.repetition_penalty = 0.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn inference_config_invalid_backend_fails() {
    let mut cfg = BitNetCInferenceConfig::default();
    cfg.backend_preference = 3;
    assert!(cfg.validate().is_err());
}

#[test]
fn inference_config_valid_backends() {
    for backend in 0..=2 {
        let mut cfg = BitNetCInferenceConfig::default();
        cfg.backend_preference = backend;
        assert!(cfg.validate().is_ok(), "backend {backend} should be valid");
    }
}

#[test]
fn inference_config_zero_stream_buffer_fails() {
    let mut cfg = BitNetCInferenceConfig::default();
    cfg.stream_buffer_size = 0;
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// BitNetCInferenceConfig — to_generation_config
// ---------------------------------------------------------------------------

#[test]
fn inference_config_to_generation_config() {
    let cfg = BitNetCInferenceConfig::default();
    let gc = cfg.to_generation_config();
    assert_eq!(gc.max_new_tokens, 512);
}

#[test]
fn inference_config_to_generation_config_with_seed() {
    let mut cfg = BitNetCInferenceConfig::default();
    cfg.seed = 42;
    let gc = cfg.to_generation_config();
    assert_eq!(gc.seed, Some(42));
}

#[test]
fn inference_config_to_generation_config_without_seed() {
    let cfg = BitNetCInferenceConfig::default();
    let gc = cfg.to_generation_config();
    assert!(gc.seed.is_none());
}

// ---------------------------------------------------------------------------
// BitNetCError — Display
// ---------------------------------------------------------------------------

#[test]
fn error_display_all_variants() {
    let cases: Vec<(BitNetCError, &str)> = vec![
        (BitNetCError::InvalidArgument("test".into()), "Invalid argument: test"),
        (BitNetCError::ModelNotFound("path".into()), "Model not found: path"),
        (BitNetCError::ModelLoadFailed("reason".into()), "Model loading failed: reason"),
        (BitNetCError::InferenceFailed("msg".into()), "Inference failed: msg"),
        (BitNetCError::OutOfMemory("oom".into()), "Out of memory: oom"),
        (BitNetCError::ThreadSafety("race".into()), "Thread safety violation: race"),
        (BitNetCError::InvalidModelId("id".into()), "Invalid model ID: id"),
        (BitNetCError::ContextLengthExceeded("16384".into()), "Context length exceeded: 16384"),
        (BitNetCError::UnsupportedOperation("op".into()), "Unsupported operation: op"),
        (BitNetCError::Internal("err".into()), "Internal error: err"),
    ];
    for (error, expected) in cases {
        assert_eq!(format!("{error}"), expected, "Display mismatch for {error:?}");
    }
}

#[test]
fn error_debug_impl() {
    let err = BitNetCError::InvalidArgument("test".into());
    let dbg = format!("{err:?}");
    assert!(dbg.contains("InvalidArgument"));
}

#[test]
fn error_clone() {
    let err = BitNetCError::OutOfMemory("oom".into());
    let err2 = err.clone();
    assert_eq!(format!("{err}"), format!("{err2}"));
}

// ---------------------------------------------------------------------------
// Thread-local error state
// ---------------------------------------------------------------------------

#[test]
fn error_state_starts_empty() {
    clear_last_error();
    assert!(get_last_error().is_none());
}

#[test]
fn error_state_set_and_get() {
    set_last_error(BitNetCError::InvalidArgument("test".into()));
    let err = get_last_error().unwrap();
    assert!(format!("{err}").contains("test"));
    clear_last_error();
}

#[test]
fn error_state_clear() {
    set_last_error(BitNetCError::Internal("err".into()));
    clear_last_error();
    assert!(get_last_error().is_none());
}

#[test]
fn error_state_overwrite() {
    set_last_error(BitNetCError::Internal("first".into()));
    set_last_error(BitNetCError::OutOfMemory("second".into()));
    let err = get_last_error().unwrap();
    assert!(format!("{err}").contains("second"));
    clear_last_error();
}

// ---------------------------------------------------------------------------
// BitNetCPerformanceMetrics — defaults
// ---------------------------------------------------------------------------

#[test]
fn perf_metrics_default() {
    let m = BitNetCPerformanceMetrics::default();
    assert_eq!(m.tokens_per_second, 0.0);
    assert_eq!(m.latency_ms, 0.0);
    assert_eq!(m.memory_usage_mb, 0.0);
    assert_eq!(m.gpu_utilization, -1.0); // -1 means not available
    assert_eq!(m.tokens_generated, 0);
    assert_eq!(m.prompt_tokens, 0);
}

#[test]
fn perf_metrics_debug_impl() {
    let m = BitNetCPerformanceMetrics::default();
    let dbg = format!("{m:?}");
    assert!(dbg.contains("BitNetCPerformanceMetrics"));
}

#[test]
fn perf_metrics_clone() {
    let mut m = BitNetCPerformanceMetrics::default();
    m.tokens_per_second = 42.0;
    m.tokens_generated = 100;
    let m2 = m.clone();
    assert_eq!(m2.tokens_per_second, 42.0);
    assert_eq!(m2.tokens_generated, 100);
}
