//! Edge-case tests for server model management, streaming types, and
//! async model manager operations.
//!
//! Tests cover: ModelManagerConfig (defaults, serde, boundary), ModelMetadata
//! construction, ModelLoadStatus variants, ModelMemoryStats, ModelManagerHealth,
//! ModelManager async ops, StreamingToken/Complete/Error serde.

use bitnet_server::model_manager::{
    ModelLoadStatus, ModelManagerConfig, ModelManagerHealth, ModelMemoryStats, ModelMetadata,
};
use bitnet_server::streaming::{StreamingComplete, StreamingError, StreamingToken};
use serde_json;
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// ModelManagerConfig
// ---------------------------------------------------------------------------

#[test]
fn model_manager_config_defaults() {
    let cfg = ModelManagerConfig::default();
    assert_eq!(cfg.max_concurrent_loads, 2);
    assert_eq!(cfg.model_cache_size, 3);
    assert_eq!(cfg.load_timeout, Duration::from_secs(300));
    assert!(cfg.validation_enabled);
    assert_eq!(cfg.memory_limit_gb, Some(16.0));
}

#[test]
fn model_manager_config_serde_roundtrip() {
    let cfg = ModelManagerConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: ModelManagerConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg2.max_concurrent_loads, cfg.max_concurrent_loads);
    assert_eq!(cfg2.model_cache_size, cfg.model_cache_size);
    assert_eq!(cfg2.validation_enabled, cfg.validation_enabled);
    assert_eq!(cfg2.memory_limit_gb, cfg.memory_limit_gb);
}

#[test]
fn model_manager_config_custom_values() {
    let json = r#"{
        "max_concurrent_loads": 5,
        "model_cache_size": 10,
        "load_timeout": {"secs": 60, "nanos": 0},
        "validation_enabled": false,
        "memory_limit_gb": 64.0
    }"#;
    let cfg: ModelManagerConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.max_concurrent_loads, 5);
    assert_eq!(cfg.model_cache_size, 10);
    assert!(!cfg.validation_enabled);
    assert_eq!(cfg.memory_limit_gb, Some(64.0));
}

#[test]
fn model_manager_config_null_memory_limit() {
    let json = r#"{
        "max_concurrent_loads": 1,
        "model_cache_size": 1,
        "load_timeout": {"secs": 10, "nanos": 0},
        "validation_enabled": true,
        "memory_limit_gb": null
    }"#;
    let cfg: ModelManagerConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.memory_limit_gb, None);
}

#[test]
fn model_manager_config_debug() {
    let cfg = ModelManagerConfig::default();
    let dbg = format!("{cfg:?}");
    assert!(dbg.contains("ModelManagerConfig"));
    assert!(dbg.contains("max_concurrent_loads"));
}

#[test]
fn model_manager_config_clone() {
    let cfg = ModelManagerConfig::default();
    let cloned = cfg.clone();
    assert_eq!(cloned.max_concurrent_loads, cfg.max_concurrent_loads);
}

// ---------------------------------------------------------------------------
// ModelMetadata
// ---------------------------------------------------------------------------

#[test]
fn model_metadata_construction() {
    let meta = ModelMetadata {
        model_id: "model-1".into(),
        model_path: "/tmp/model.gguf".into(),
        device: "cpu".into(),
        quantization_type: "I2_S".into(),
        loaded_at: SystemTime::now(),
        size_mb: 1024,
        parameters: 2_000_000_000,
        context_length: 4096,
        inference_count: 0,
        avg_tokens_per_second: 0.0,
    };
    assert_eq!(meta.model_id, "model-1");
    assert_eq!(meta.parameters, 2_000_000_000);
}

#[test]
fn model_metadata_serde_roundtrip() {
    let meta = ModelMetadata {
        model_id: "test".into(),
        model_path: "/path".into(),
        device: "cuda:0".into(),
        quantization_type: "FP16".into(),
        loaded_at: SystemTime::UNIX_EPOCH,
        size_mb: 512,
        parameters: 14_000_000_000,
        context_length: 16384,
        inference_count: 42,
        avg_tokens_per_second: 15.3,
    };
    let json = serde_json::to_string(&meta).unwrap();
    let meta2: ModelMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(meta2.model_id, "test");
    assert_eq!(meta2.context_length, 16384);
    assert_eq!(meta2.inference_count, 42);
}

#[test]
fn model_metadata_clone() {
    let meta = ModelMetadata {
        model_id: "clone-test".into(),
        model_path: "p".into(),
        device: "cpu".into(),
        quantization_type: "Q4".into(),
        loaded_at: SystemTime::now(),
        size_mb: 0,
        parameters: 0,
        context_length: 2048,
        inference_count: 0,
        avg_tokens_per_second: 0.0,
    };
    let c = meta.clone();
    assert_eq!(c.model_id, "clone-test");
}

// ---------------------------------------------------------------------------
// ModelLoadStatus
// ---------------------------------------------------------------------------

#[test]
fn model_load_status_loading() {
    let status = ModelLoadStatus::Loading { progress: 0.5, stage: "Downloading".into() };
    let dbg = format!("{status:?}");
    assert!(dbg.contains("Loading"));
    assert!(dbg.contains("0.5"));
}

#[test]
fn model_load_status_failed() {
    let status = ModelLoadStatus::Failed { error: "Out of memory".into() };
    let dbg = format!("{status:?}");
    assert!(dbg.contains("Failed"));
    assert!(dbg.contains("Out of memory"));
}

#[test]
fn model_load_status_unloading() {
    let status = ModelLoadStatus::Unloading;
    let dbg = format!("{status:?}");
    assert!(dbg.contains("Unloading"));
}

// ---------------------------------------------------------------------------
// ModelMemoryStats
// ---------------------------------------------------------------------------

#[test]
fn model_memory_stats_serialize() {
    let stats = ModelMemoryStats {
        total_models: 2,
        total_size_mb: 4096,
        active_model_id: Some("model-1".into()),
        cache_size_limit: 3,
        memory_limit_gb: Some(16.0),
    };
    let json = serde_json::to_string(&stats).unwrap();
    assert!(json.contains("\"total_models\":2"));
    assert!(json.contains("model-1"));
}

#[test]
fn model_memory_stats_no_active_model() {
    let stats = ModelMemoryStats {
        total_models: 0,
        total_size_mb: 0,
        active_model_id: None,
        cache_size_limit: 3,
        memory_limit_gb: None,
    };
    let json = serde_json::to_string(&stats).unwrap();
    assert!(json.contains("\"active_model_id\":null"));
}

// ---------------------------------------------------------------------------
// ModelManagerHealth
// ---------------------------------------------------------------------------

#[test]
fn model_manager_health_serialize() {
    let health = ModelManagerHealth {
        active_model_healthy: true,
        cached_models: 2,
        loading_operations: 0,
        last_error: None,
    };
    let json = serde_json::to_string(&health).unwrap();
    assert!(json.contains("\"active_model_healthy\":true"));
}

#[test]
fn model_manager_health_with_error() {
    let health = ModelManagerHealth {
        active_model_healthy: false,
        cached_models: 0,
        loading_operations: 1,
        last_error: Some("Load failed".into()),
    };
    let dbg = format!("{health:?}");
    assert!(dbg.contains("Load failed"));
}

// ---------------------------------------------------------------------------
// ModelManager
// ---------------------------------------------------------------------------

#[tokio::test]
async fn model_manager_new_no_active_model() {
    let mm = bitnet_server::model_manager::ModelManager::new(ModelManagerConfig::default());
    assert!(mm.get_active_model().await.is_none());
}

#[tokio::test]
async fn model_manager_get_status_empty() {
    let mm = bitnet_server::model_manager::ModelManager::new(ModelManagerConfig::default());
    let status = mm.get_loading_status("nonexistent").await;
    assert!(status.is_none());
}

// ---------------------------------------------------------------------------
// StreamingToken
// ---------------------------------------------------------------------------

#[test]
fn streaming_token_serde_roundtrip() {
    let tok = StreamingToken {
        token: "Hello".into(),
        token_id: 42,
        cumulative_time_ms: 100,
        position: 1,
    };
    let json = serde_json::to_string(&tok).unwrap();
    let tok2: StreamingToken = serde_json::from_str(&json).unwrap();
    assert_eq!(tok2.token, "Hello");
    assert_eq!(tok2.token_id, 42);
    assert_eq!(tok2.position, 1);
}

#[test]
fn streaming_token_empty_string() {
    let tok = StreamingToken { token: "".into(), token_id: 0, cumulative_time_ms: 0, position: 0 };
    let json = serde_json::to_string(&tok).unwrap();
    assert!(json.contains("\"token\":\"\""));
}

// ---------------------------------------------------------------------------
// StreamingComplete
// ---------------------------------------------------------------------------

#[test]
fn streaming_complete_serialize() {
    let complete = StreamingComplete {
        total_tokens: 100,
        total_time_ms: 5000,
        tokens_per_second: 20.0,
        completed_normally: true,
        completion_reason: Some("Done".into()),
    };
    let json = serde_json::to_string(&complete).unwrap();
    assert!(json.contains("\"total_tokens\":100"));
    assert!(json.contains("\"completed_normally\":true"));
}

#[test]
fn streaming_complete_cancelled() {
    let complete = StreamingComplete {
        total_tokens: 5,
        total_time_ms: 30000,
        tokens_per_second: 0.17,
        completed_normally: false,
        completion_reason: Some("Timeout".into()),
    };
    let json = serde_json::to_string(&complete).unwrap();
    assert!(json.contains("\"completed_normally\":false"));
}

#[test]
fn streaming_complete_no_reason() {
    let complete = StreamingComplete {
        total_tokens: 0,
        total_time_ms: 0,
        tokens_per_second: 0.0,
        completed_normally: true,
        completion_reason: None,
    };
    let json = serde_json::to_string(&complete).unwrap();
    assert!(json.contains("\"completion_reason\":null"));
}

// ---------------------------------------------------------------------------
// StreamingError
// ---------------------------------------------------------------------------

#[test]
fn streaming_error_serialize() {
    let err = StreamingError {
        error_type: "generation".into(),
        message: "Model not loaded".into(),
        recovery_hints: Some(vec!["Load a model first".into()]),
        tokens_before_error: 0,
    };
    let json = serde_json::to_string(&err).unwrap();
    assert!(json.contains("\"error_type\":\"generation\""));
    assert!(json.contains("Load a model first"));
}

#[test]
fn streaming_error_no_hints() {
    let err = StreamingError {
        error_type: "internal".into(),
        message: "Unknown error".into(),
        recovery_hints: None,
        tokens_before_error: 42,
    };
    let json = serde_json::to_string(&err).unwrap();
    assert!(json.contains("\"tokens_before_error\":42"));
    assert!(json.contains("\"recovery_hints\":null"));
}
