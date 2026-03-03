//! Edge-case tests for batch_engine.rs public types and configuration.
//!
//! Tests cover `BatchEngineConfig`, `RequestPriority`, `BatchRequest`,
//! `BatchResult`, `QuantizationOptimization`, `BatchEngineStats`,
//! `BatchEngine` construction, and async submit/health operations.

use bitnet_common::Device;
use bitnet_inference::GenerationConfig;
use bitnet_server::batch_engine::{
    BatchEngine, BatchEngineConfig, BatchEngineStats, BatchRequest, BatchResult,
    QuantizationOptimization, RequestPriority,
};
use std::time::Duration;

// ── BatchEngineConfig ────────────────────────────────────────────────────────

#[test]
fn batch_engine_config_default() {
    let cfg = BatchEngineConfig::default();
    assert_eq!(cfg.max_batch_size, 16);
    assert_eq!(cfg.batch_timeout, Duration::from_millis(100));
    assert_eq!(cfg.max_concurrent_batches, 4);
    assert!(cfg.priority_queue_enabled);
    assert!(cfg.adaptive_batching);
    assert!(cfg.quantization_aware);
    assert!(cfg.simd_optimization);
}

#[test]
fn batch_engine_config_clone() {
    let cfg = BatchEngineConfig::default();
    let cfg2 = cfg.clone();
    assert_eq!(cfg.max_batch_size, cfg2.max_batch_size);
    assert_eq!(cfg.batch_timeout, cfg2.batch_timeout);
}

#[test]
fn batch_engine_config_debug() {
    let cfg = BatchEngineConfig::default();
    let dbg = format!("{cfg:?}");
    assert!(dbg.contains("BatchEngineConfig"));
    assert!(dbg.contains("16")); // max_batch_size
}

#[test]
fn batch_engine_config_serde_roundtrip() {
    let cfg = BatchEngineConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: BatchEngineConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg.max_batch_size, cfg2.max_batch_size);
    assert_eq!(cfg.max_concurrent_batches, cfg2.max_concurrent_batches);
}

#[test]
fn batch_engine_config_custom_values() {
    let cfg = BatchEngineConfig {
        max_batch_size: 32,
        batch_timeout: Duration::from_millis(500),
        max_concurrent_batches: 8,
        priority_queue_enabled: false,
        adaptive_batching: false,
        quantization_aware: false,
        simd_optimization: false,
    };
    assert_eq!(cfg.max_batch_size, 32);
    assert!(!cfg.priority_queue_enabled);
}

// ── RequestPriority ──────────────────────────────────────────────────────────

#[test]
fn request_priority_ordering() {
    assert!(RequestPriority::Low < RequestPriority::Normal);
    assert!(RequestPriority::Normal < RequestPriority::High);
    assert!(RequestPriority::High < RequestPriority::Critical);
}

#[test]
fn request_priority_equality() {
    assert_eq!(RequestPriority::Normal, RequestPriority::Normal);
    assert_ne!(RequestPriority::Low, RequestPriority::High);
}

#[test]
fn request_priority_clone_copy() {
    let p = RequestPriority::High;
    let p2 = p; // copy
    let p3 = p.clone();
    assert_eq!(p, p2);
    assert_eq!(p, p3);
}

#[test]
fn request_priority_debug() {
    assert!(format!("{:?}", RequestPriority::Low).contains("Low"));
    assert!(format!("{:?}", RequestPriority::Normal).contains("Normal"));
    assert!(format!("{:?}", RequestPriority::High).contains("High"));
    assert!(format!("{:?}", RequestPriority::Critical).contains("Critical"));
}

// ── BatchRequest ─────────────────────────────────────────────────────────────

#[test]
fn batch_request_new_defaults() {
    let config = GenerationConfig::default();
    let max_tokens_expected = config.max_new_tokens;
    let req = BatchRequest::new("hello world".to_string(), config);
    assert_eq!(req.prompt, "hello world");
    assert_eq!(req.priority, RequestPriority::Normal);
    assert!(req.device_preference.is_none());
    assert_eq!(req.max_tokens, max_tokens_expected);
    assert!(req.quantization_hint.is_none());
    assert!(req.timeout.is_none());
    assert!(!req.id.is_empty()); // UUID generated
}

#[test]
fn batch_request_with_priority() {
    let req = BatchRequest::new("test".to_string(), GenerationConfig::default())
        .with_priority(RequestPriority::Critical);
    assert_eq!(req.priority, RequestPriority::Critical);
}

#[test]
fn batch_request_with_device_preference() {
    let req = BatchRequest::new("test".to_string(), GenerationConfig::default())
        .with_device_preference(Device::Cpu);
    assert_eq!(req.device_preference, Some(Device::Cpu));
}

#[test]
fn batch_request_with_timeout() {
    let req = BatchRequest::new("test".to_string(), GenerationConfig::default())
        .with_timeout(Duration::from_secs(30));
    assert_eq!(req.timeout, Some(Duration::from_secs(30)));
}

#[test]
fn batch_request_with_quantization_hint() {
    let req = BatchRequest::new("test".to_string(), GenerationConfig::default())
        .with_quantization_hint("int8".to_string());
    assert_eq!(req.quantization_hint, Some("int8".to_string()));
}

#[test]
fn batch_request_chained_builders() {
    let req = BatchRequest::new("prompt".to_string(), GenerationConfig::default())
        .with_priority(RequestPriority::High)
        .with_device_preference(Device::Cpu)
        .with_timeout(Duration::from_secs(60))
        .with_quantization_hint("i2s".to_string());
    assert_eq!(req.priority, RequestPriority::High);
    assert_eq!(req.device_preference, Some(Device::Cpu));
    assert_eq!(req.timeout, Some(Duration::from_secs(60)));
    assert_eq!(req.quantization_hint, Some("i2s".to_string()));
}

#[test]
fn batch_request_unique_ids() {
    let req1 = BatchRequest::new("a".to_string(), GenerationConfig::default());
    let req2 = BatchRequest::new("b".to_string(), GenerationConfig::default());
    assert_ne!(req1.id, req2.id);
}

#[test]
fn batch_request_clone() {
    let req = BatchRequest::new("test".to_string(), GenerationConfig::default());
    let req2 = req.clone();
    assert_eq!(req.id, req2.id);
    assert_eq!(req.prompt, req2.prompt);
}

#[test]
fn batch_request_debug() {
    let req = BatchRequest::new("test".to_string(), GenerationConfig::default());
    let dbg = format!("{req:?}");
    assert!(dbg.contains("BatchRequest"));
    assert!(dbg.contains("test"));
}

// ── BatchResult ──────────────────────────────────────────────────────────────

#[test]
fn batch_result_construction() {
    let result = BatchResult {
        request_id: "req-1".to_string(),
        generated_text: "Hello!".to_string(),
        tokens_generated: 5,
        execution_time: Duration::from_millis(100),
        device_used: Device::Cpu,
        quantization_type: "I2S".to_string(),
        batch_id: "batch-1".to_string(),
        batch_size: 4,
    };
    assert_eq!(result.request_id, "req-1");
    assert_eq!(result.tokens_generated, 5);
    assert_eq!(result.batch_size, 4);
}

#[test]
fn batch_result_clone() {
    let result = BatchResult {
        request_id: "req-1".to_string(),
        generated_text: "text".to_string(),
        tokens_generated: 10,
        execution_time: Duration::from_millis(50),
        device_used: Device::Cpu,
        quantization_type: "TL1".to_string(),
        batch_id: "b1".to_string(),
        batch_size: 1,
    };
    let result2 = result.clone();
    assert_eq!(result.request_id, result2.request_id);
    assert_eq!(result.tokens_generated, result2.tokens_generated);
}

#[test]
fn batch_result_debug() {
    let result = BatchResult {
        request_id: "r".to_string(),
        generated_text: "t".to_string(),
        tokens_generated: 0,
        execution_time: Duration::ZERO,
        device_used: Device::Cpu,
        quantization_type: "I2S".to_string(),
        batch_id: "b".to_string(),
        batch_size: 0,
    };
    let dbg = format!("{result:?}");
    assert!(dbg.contains("BatchResult"));
}

// ── QuantizationOptimization ─────────────────────────────────────────────────

#[test]
fn quantization_optimization_construction() {
    let opt = QuantizationOptimization {
        batch_compatible_requests: vec![0, 1, 2],
        recommended_device: Device::Cpu,
        quantization_type: "I2S".to_string(),
        simd_instruction_set: Some("AVX2".to_string()),
        memory_requirement_mb: 512,
    };
    assert_eq!(opt.batch_compatible_requests.len(), 3);
    assert_eq!(opt.memory_requirement_mb, 512);
}

#[test]
fn quantization_optimization_no_simd() {
    let opt = QuantizationOptimization {
        batch_compatible_requests: vec![],
        recommended_device: Device::Cpu,
        quantization_type: "TL1".to_string(),
        simd_instruction_set: None,
        memory_requirement_mb: 0,
    };
    assert!(opt.simd_instruction_set.is_none());
    assert!(opt.batch_compatible_requests.is_empty());
}

#[test]
fn quantization_optimization_clone() {
    let opt = QuantizationOptimization {
        batch_compatible_requests: vec![0],
        recommended_device: Device::Cpu,
        quantization_type: "I2S".to_string(),
        simd_instruction_set: None,
        memory_requirement_mb: 100,
    };
    let opt2 = opt.clone();
    assert_eq!(opt.batch_compatible_requests, opt2.batch_compatible_requests);
}

// ── BatchEngineStats ─────────────────────────────────────────────────────────

#[test]
fn batch_engine_stats_debug() {
    let stats = BatchEngineStats {
        total_requests_processed: 100,
        total_batches_processed: 25,
        average_batch_size: 4.0,
        average_batch_time_ms: 50.0,
        queue_depth: 3,
        active_batches: 2,
        throughput_tokens_per_second: 100.0,
        cache_hit_rate: 0.85,
    };
    let dbg = format!("{stats:?}");
    assert!(dbg.contains("BatchEngineStats"));
    assert!(dbg.contains("100")); // total_requests_processed or throughput
}

#[test]
fn batch_engine_stats_serde() {
    let stats = BatchEngineStats {
        total_requests_processed: 50,
        total_batches_processed: 10,
        average_batch_size: 5.0,
        average_batch_time_ms: 25.0,
        queue_depth: 0,
        active_batches: 0,
        throughput_tokens_per_second: 200.0,
        cache_hit_rate: 0.0,
    };
    let json = serde_json::to_string(&stats).unwrap();
    assert!(json.contains("total_requests_processed"));
    assert!(json.contains("50"));
}

#[test]
fn batch_engine_stats_clone() {
    let stats = BatchEngineStats {
        total_requests_processed: 1,
        total_batches_processed: 1,
        average_batch_size: 1.0,
        average_batch_time_ms: 1.0,
        queue_depth: 0,
        active_batches: 0,
        throughput_tokens_per_second: 0.0,
        cache_hit_rate: 0.0,
    };
    let stats2 = stats.clone();
    assert_eq!(stats.total_requests_processed, stats2.total_requests_processed);
}

// ── BatchEngine construction ─────────────────────────────────────────────────

#[test]
fn batch_engine_new_default() {
    let engine = BatchEngine::new(BatchEngineConfig::default());
    // Just verify it constructs without panic
    let _ = format!("{:p}", &engine); // verify it exists
}

#[test]
fn batch_engine_new_custom_config() {
    let cfg = BatchEngineConfig {
        max_batch_size: 1,
        batch_timeout: Duration::from_millis(10),
        max_concurrent_batches: 1,
        priority_queue_enabled: false,
        adaptive_batching: false,
        quantization_aware: false,
        simd_optimization: false,
    };
    let engine = BatchEngine::new(cfg);
    let _ = format!("{:p}", &engine);
}

// ── BatchEngine async operations ─────────────────────────────────────────────

#[tokio::test]
async fn batch_engine_health_check() {
    let engine = BatchEngine::new(BatchEngineConfig::default());
    let health = engine.get_health().await;
    // Health check should succeed on a fresh engine
    assert!(health.healthy);
    assert_eq!(health.queue_depth, 0);
    assert_eq!(health.active_batches, 0);
}

#[tokio::test]
async fn batch_engine_stats() {
    let engine = BatchEngine::new(BatchEngineConfig::default());
    let stats = engine.get_stats().await;
    assert_eq!(stats.total_requests_processed, 0);
    assert_eq!(stats.total_batches_processed, 0);
    assert_eq!(stats.queue_depth, 0);
}
