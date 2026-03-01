//! Edge-case tests for model_manager.rs public types and configuration.
//!
//! Tests cover `ModelManagerConfig`, `ModelMetadata`, `ModelLoadStatus`,
//! `ModelManager` construction and basic async operations.

use bitnet_server::model_manager::{
    ModelLoadStatus, ModelManager, ModelManagerConfig, ModelMemoryStats, ModelMetadata,
};
use std::time::{Duration, SystemTime};

// ── ModelManagerConfig ───────────────────────────────────────────────────────

#[test]
fn model_manager_config_default() {
    let cfg = ModelManagerConfig::default();
    assert_eq!(cfg.max_concurrent_loads, 2);
    assert_eq!(cfg.model_cache_size, 3);
    assert_eq!(cfg.load_timeout, Duration::from_secs(300));
    assert!(cfg.validation_enabled);
    assert_eq!(cfg.memory_limit_gb, Some(16.0));
}

#[test]
fn model_manager_config_clone() {
    let cfg = ModelManagerConfig::default();
    let cfg2 = cfg.clone();
    assert_eq!(cfg.max_concurrent_loads, cfg2.max_concurrent_loads);
    assert_eq!(cfg.memory_limit_gb, cfg2.memory_limit_gb);
}

#[test]
fn model_manager_config_debug() {
    let cfg = ModelManagerConfig::default();
    let dbg = format!("{cfg:?}");
    assert!(dbg.contains("ModelManagerConfig"));
}

#[test]
fn model_manager_config_serde_roundtrip() {
    let cfg = ModelManagerConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: ModelManagerConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg.max_concurrent_loads, cfg2.max_concurrent_loads);
    assert_eq!(cfg.model_cache_size, cfg2.model_cache_size);
}

#[test]
fn model_manager_config_custom() {
    let cfg = ModelManagerConfig {
        max_concurrent_loads: 8,
        model_cache_size: 10,
        load_timeout: Duration::from_secs(60),
        validation_enabled: false,
        memory_limit_gb: None,
    };
    assert_eq!(cfg.max_concurrent_loads, 8);
    assert!(cfg.memory_limit_gb.is_none());
}

// ── ModelMetadata ────────────────────────────────────────────────────────────

#[test]
fn model_metadata_construction() {
    let meta = ModelMetadata {
        model_id: "phi-4".to_string(),
        model_path: "/models/phi4.gguf".to_string(),
        device: "cpu".to_string(),
        quantization_type: "I2S".to_string(),
        loaded_at: SystemTime::now(),
        size_mb: 5000,
        parameters: 14_000_000_000,
        context_length: 16384,
        inference_count: 0,
        avg_tokens_per_second: 0.0,
    };
    assert_eq!(meta.model_id, "phi-4");
    assert_eq!(meta.parameters, 14_000_000_000);
    assert_eq!(meta.context_length, 16384);
}

#[test]
fn model_metadata_clone() {
    let meta = ModelMetadata {
        model_id: "test".to_string(),
        model_path: "/path".to_string(),
        device: "cpu".to_string(),
        quantization_type: "TL1".to_string(),
        loaded_at: SystemTime::UNIX_EPOCH,
        size_mb: 100,
        parameters: 2_000_000,
        context_length: 4096,
        inference_count: 42,
        avg_tokens_per_second: 15.5,
    };
    let meta2 = meta.clone();
    assert_eq!(meta.model_id, meta2.model_id);
    assert_eq!(meta.inference_count, meta2.inference_count);
}

#[test]
fn model_metadata_serde_roundtrip() {
    let meta = ModelMetadata {
        model_id: "model-1".to_string(),
        model_path: "/some/path".to_string(),
        device: "cuda:0".to_string(),
        quantization_type: "I2S".to_string(),
        loaded_at: SystemTime::UNIX_EPOCH,
        size_mb: 200,
        parameters: 100_000,
        context_length: 2048,
        inference_count: 0,
        avg_tokens_per_second: 0.0,
    };
    let json = serde_json::to_string(&meta).unwrap();
    let meta2: ModelMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(meta.model_id, meta2.model_id);
    assert_eq!(meta.size_mb, meta2.size_mb);
}

#[test]
fn model_metadata_debug() {
    let meta = ModelMetadata {
        model_id: "dbg-test".to_string(),
        model_path: "p".to_string(),
        device: "cpu".to_string(),
        quantization_type: "I2S".to_string(),
        loaded_at: SystemTime::UNIX_EPOCH,
        size_mb: 0,
        parameters: 0,
        context_length: 0,
        inference_count: 0,
        avg_tokens_per_second: 0.0,
    };
    let dbg = format!("{meta:?}");
    assert!(dbg.contains("ModelMetadata"));
    assert!(dbg.contains("dbg-test"));
}

// ── ModelLoadStatus ──────────────────────────────────────────────────────────

#[test]
fn model_load_status_loading() {
    let status = ModelLoadStatus::Loading { progress: 0.5, stage: "downloading".to_string() };
    if let ModelLoadStatus::Loading { progress, stage } = &status {
        assert!((progress - 0.5).abs() < 0.001);
        assert_eq!(stage, "downloading");
    } else {
        panic!("expected Loading");
    }
}

#[test]
fn model_load_status_ready() {
    let meta = ModelMetadata {
        model_id: "m".to_string(),
        model_path: "p".to_string(),
        device: "cpu".to_string(),
        quantization_type: "I2S".to_string(),
        loaded_at: SystemTime::UNIX_EPOCH,
        size_mb: 0,
        parameters: 0,
        context_length: 0,
        inference_count: 0,
        avg_tokens_per_second: 0.0,
    };
    let status = ModelLoadStatus::Ready { metadata: meta };
    if let ModelLoadStatus::Ready { metadata } = &status {
        assert_eq!(metadata.model_id, "m");
    } else {
        panic!("expected Ready");
    }
}

#[test]
fn model_load_status_failed() {
    let status = ModelLoadStatus::Failed { error: "out of memory".to_string() };
    if let ModelLoadStatus::Failed { error } = &status {
        assert!(error.contains("out of memory"));
    } else {
        panic!("expected Failed");
    }
}

#[test]
fn model_load_status_unloading() {
    let status = ModelLoadStatus::Unloading;
    assert!(format!("{status:?}").contains("Unloading"));
}

#[test]
fn model_load_status_clone() {
    let status = ModelLoadStatus::Loading { progress: 0.75, stage: "loading".to_string() };
    let status2 = status.clone();
    if let ModelLoadStatus::Loading { progress, .. } = status2 {
        assert!((progress - 0.75).abs() < 0.001);
    }
}

// ── ModelManager construction ────────────────────────────────────────────────

#[test]
fn model_manager_new_default() {
    let mgr = ModelManager::new(ModelManagerConfig::default());
    let _ = format!("{:p}", &mgr);
}

#[test]
fn model_manager_new_custom() {
    let cfg = ModelManagerConfig {
        max_concurrent_loads: 1,
        model_cache_size: 1,
        load_timeout: Duration::from_secs(10),
        validation_enabled: false,
        memory_limit_gb: Some(4.0),
    };
    let mgr = ModelManager::new(cfg);
    let _ = format!("{:p}", &mgr);
}

// ── ModelManager async ───────────────────────────────────────────────────────

#[tokio::test]
async fn model_manager_no_active_model() {
    let mgr = ModelManager::new(ModelManagerConfig::default());
    let active = mgr.get_active_model().await;
    assert!(active.is_none());
}

// ── ModelMemoryStats ─────────────────────────────────────────────────────────

#[test]
fn model_memory_stats_construction() {
    let stats = ModelMemoryStats {
        total_models: 3,
        total_size_mb: 5000,
        active_model_id: Some("phi-4".to_string()),
        cache_size_limit: 5,
        memory_limit_gb: Some(16.0),
    };
    assert_eq!(stats.total_models, 3);
    assert_eq!(stats.total_size_mb, 5000);
}

#[test]
fn model_memory_stats_clone() {
    let stats = ModelMemoryStats {
        total_models: 1,
        total_size_mb: 100,
        active_model_id: None,
        cache_size_limit: 3,
        memory_limit_gb: Some(8.0),
    };
    let stats2 = stats.clone();
    assert_eq!(stats.total_models, stats2.total_models);
}

#[test]
fn model_memory_stats_debug() {
    let stats = ModelMemoryStats {
        total_models: 0,
        total_size_mb: 0,
        active_model_id: None,
        cache_size_limit: 0,
        memory_limit_gb: None,
    };
    let dbg = format!("{stats:?}");
    assert!(dbg.contains("ModelMemoryStats"));
}
