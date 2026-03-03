//! Snapshot wave 32 — generation config defaults, sampling strategy
//! descriptions, batch config display, request/response types.

use bitnet_inference::batch_engine::{
    BatchConfig, BatchRequest, BatchResponse, Priority, SchedulingPolicy,
};
use bitnet_inference::cache::{CacheConfig, EvictionPolicy};
use bitnet_inference::config::{GenerationConfig, InferenceConfig};
use bitnet_inference::generation_output::StopReason;
use bitnet_inference::request_types::{
    FinishReason, InferenceRequest, InferenceResponse, TimingInfo, UsageInfo,
};
use bitnet_inference::streaming::StreamingConfig;

// ── GenerationConfig defaults ───────────────────────────────────────

#[test]
fn generation_config_default_debug() {
    let cfg = GenerationConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn generation_config_default_yaml() {
    let cfg = GenerationConfig::default();
    insta::assert_yaml_snapshot!(cfg);
}

#[test]
fn generation_config_custom_debug() {
    let cfg = GenerationConfig::default()
        .with_max_tokens(256)
        .with_temperature(0.0)
        .with_top_k(1)
        .with_top_p(1.0)
        .with_repetition_penalty(1.05)
        .with_stop_sequences(vec!["</s>".to_string(), "\n\nQ:".to_string()])
        .with_stop_token_ids(vec![128009])
        .with_seed(42);
    insta::assert_debug_snapshot!(cfg);
}

// ── InferenceConfig ─────────────────────────────────────────────────

#[test]
fn inference_config_default_debug() {
    let cfg = InferenceConfig::default();
    // num_threads is host-dependent so we format manually
    let output = format!(
        "InferenceConfig {{ max_context_length: {}, batch_size: {}, mixed_precision: {}, memory_pool_size: {} }}",
        cfg.max_context_length, cfg.batch_size, cfg.mixed_precision, cfg.memory_pool_size
    );
    insta::assert_snapshot!(output);
}

// ── BatchConfig ─────────────────────────────────────────────────────

#[test]
fn batch_config_default_debug() {
    let cfg = BatchConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn batch_config_custom_debug() {
    let cfg = BatchConfig {
        max_batch_size: 32,
        max_queue_depth: 500,
        scheduling_policy: SchedulingPolicy::PriorityBased,
        timeout_ms: 60_000,
    };
    insta::assert_debug_snapshot!(cfg);
}

// ── Priority / SchedulingPolicy ─────────────────────────────────────

#[test]
fn priority_debug_all() {
    let priorities = vec![Priority::Low, Priority::Normal, Priority::High, Priority::Critical];
    insta::assert_debug_snapshot!(priorities);
}

#[test]
fn scheduling_policy_debug_all() {
    let policies = vec![
        SchedulingPolicy::FIFO,
        SchedulingPolicy::PriorityBased,
        SchedulingPolicy::ShortestJobFirst,
        SchedulingPolicy::RoundRobin,
    ];
    insta::assert_debug_snapshot!(policies);
}

// ── BatchRequest / BatchResponse ────────────────────────────────────

#[test]
fn batch_request_debug() {
    let req = BatchRequest {
        id: "req-001".to_string(),
        prompt: "What is 2+2?".to_string(),
        max_tokens: 32,
        temperature: 0.7,
        priority: Priority::Normal,
    };
    insta::assert_debug_snapshot!(req);
}

#[test]
fn batch_response_debug() {
    let resp = BatchResponse {
        request_id: "req-001".to_string(),
        text: "4".to_string(),
        tokens_generated: 1,
        finish_reason: "stop".to_string(),
        time_ms: 42,
    };
    insta::assert_debug_snapshot!(resp);
}

// ── CacheConfig ─────────────────────────────────────────────────────

#[test]
fn cache_config_default_debug() {
    let cfg = CacheConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn eviction_policy_debug_all() {
    let policies = vec![EvictionPolicy::LRU, EvictionPolicy::FIFO, EvictionPolicy::LFU];
    insta::assert_debug_snapshot!(policies);
}

// ── StreamingConfig ─────────────────────────────────────────────────

#[test]
fn streaming_config_default_debug() {
    let cfg = StreamingConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

// ── Request / Response types ────────────────────────────────────────

#[test]
fn inference_request_default_debug() {
    let req = InferenceRequest::default();
    insta::assert_debug_snapshot!(req);
}

#[test]
fn inference_response_completed_debug() {
    let resp = InferenceResponse {
        id: "resp-001".to_string(),
        text: "The capital of France is Paris.".to_string(),
        token_ids: vec![450, 6864, 315, 9822, 374, 12366, 13],
        token_count: 7,
        prompt_tokens: 5,
        finish_reason: FinishReason::EosToken,
        timing: TimingInfo::default(),
        usage: UsageInfo::default(),
    };
    insta::assert_debug_snapshot!(resp);
}

#[test]
fn finish_reason_debug_all() {
    let reasons = vec![
        FinishReason::MaxTokens,
        FinishReason::StopSequence,
        FinishReason::EosToken,
        FinishReason::Error("timeout".to_string()),
    ];
    insta::assert_debug_snapshot!(reasons);
}

// ── StopReason ──────────────────────────────────────────────────────

#[test]
fn stop_reason_debug_all() {
    let reasons = vec![
        StopReason::MaxTokens,
        StopReason::EosToken,
        StopReason::StopSequence("</s>".to_string()),
        StopReason::UserAbort,
        StopReason::Error("OOM".to_string()),
    ];
    insta::assert_debug_snapshot!(reasons);
}

// ── TimingInfo / UsageInfo ──────────────────────────────────────────

#[test]
fn timing_info_default_debug() {
    let t = TimingInfo::default();
    insta::assert_debug_snapshot!(t);
}

#[test]
fn usage_info_default_debug() {
    let u = UsageInfo::default();
    insta::assert_debug_snapshot!(u);
}
