//! Wave 26 snapshot tests for `bitnet-inference` — generation budget,
//! stop reasons, profiler configs, memory estimation, layer profiles,
//! memory snapshots, and performance mode.
//!
//! Pins Debug/Display output so unintentional changes are caught at review.

use std::time::Duration;

// =========================================================================
// Section 1 — GenerationBudget
// =========================================================================

use bitnet_inference::generation_budget::{BudgetTracker, GenerationBudget, StopReason};

#[test]
fn w26_generation_budget_default() {
    let budget = GenerationBudget::default();
    insta::assert_debug_snapshot!(budget);
}

#[test]
fn w26_generation_budget_with_token_limit() {
    let budget = GenerationBudget::new(128);
    insta::assert_debug_snapshot!(budget);
}

#[test]
fn w26_generation_budget_with_time_limit() {
    let budget = GenerationBudget::new(64).with_time_limit(Duration::from_secs(30));
    insta::assert_debug_snapshot!(budget);
}

#[test]
fn w26_generation_budget_with_memory_limit() {
    let budget = GenerationBudget::new(256).with_memory_limit(1_073_741_824);
    insta::assert_debug_snapshot!(budget);
}

#[test]
fn w26_generation_budget_full_limits() {
    let budget = GenerationBudget::new(512)
        .with_time_limit(Duration::from_secs(60))
        .with_memory_limit(2_147_483_648);
    insta::assert_debug_snapshot!(budget);
}

#[test]
fn w26_generation_budget_unlimited() {
    let budget = GenerationBudget::unlimited();
    insta::assert_debug_snapshot!(budget);
}

// =========================================================================
// Section 2 — StopReason Display
// =========================================================================

#[test]
fn w26_stop_reason_all_display() {
    let reasons = vec![
        StopReason::MaxTokens,
        StopReason::TimeLimit,
        StopReason::MemoryLimit,
        StopReason::EndOfSequence,
        StopReason::UserStop,
    ];
    let displays: Vec<String> = reasons.iter().map(|r| r.to_string()).collect();
    insta::assert_debug_snapshot!(displays);
}

#[test]
fn w26_stop_reason_max_tokens_debug() {
    insta::assert_debug_snapshot!(StopReason::MaxTokens);
}

#[test]
fn w26_stop_reason_eos_debug() {
    insta::assert_debug_snapshot!(StopReason::EndOfSequence);
}

// =========================================================================
// Section 3 — BudgetTracker
// =========================================================================

#[test]
fn w26_budget_tracker_fresh() {
    let budget = GenerationBudget::new(32);
    let tracker = BudgetTracker::new(budget);
    // Only check tokens_remaining; start_time is non-deterministic
    insta::assert_snapshot!(format!("remaining={}", tracker.tokens_remaining()));
}

#[test]
fn w26_budget_tracker_after_tokens() {
    let budget = GenerationBudget::new(100);
    let mut tracker = BudgetTracker::new(budget);
    for _ in 0..25 {
        tracker.record_token();
    }
    insta::assert_snapshot!(format!(
        "generated={} remaining={}",
        tracker.tokens_generated(),
        tracker.tokens_remaining()
    ));
}

// =========================================================================
// Section 4 — ProfilerConfig
// =========================================================================

use bitnet_inference::profiler::ProfilerConfig;

#[test]
fn w26_profiler_config_default() {
    let cfg = ProfilerConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w26_profiler_config_disabled() {
    let cfg = ProfilerConfig::disabled();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w26_profiler_config_with_memory() {
    let cfg = ProfilerConfig {
        enabled: true,
        record_memory: true,
        warmup_iterations: 3,
        sample_size: 10,
    };
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 5 — LayerProfile
// =========================================================================

use bitnet_inference::profiler::LayerProfile;

#[test]
fn w26_layer_profile_attention() {
    let lp = LayerProfile {
        layer_name: "layer_0.attention".into(),
        layer_type: "attention".into(),
        forward_time_us: 1250.5,
        backward_time_us: 0.0,
        memory_bytes: 8_388_608,
        flops_estimate: 2_147_483_648,
    };
    insta::assert_debug_snapshot!(lp);
}

#[test]
fn w26_layer_profile_ffn() {
    let lp = LayerProfile {
        layer_name: "layer_3.ffn".into(),
        layer_type: "ffn".into(),
        forward_time_us: 890.2,
        backward_time_us: 0.0,
        memory_bytes: 16_777_216,
        flops_estimate: 4_294_967_296,
    };
    insta::assert_debug_snapshot!(lp);
}

#[test]
fn w26_layer_profile_norm() {
    let lp = LayerProfile {
        layer_name: "layer_7.norm".into(),
        layer_type: "norm".into(),
        forward_time_us: 12.3,
        backward_time_us: 0.0,
        memory_bytes: 16384,
        flops_estimate: 8192,
    };
    insta::assert_debug_snapshot!(lp);
}

// =========================================================================
// Section 6 — MemorySnapshot
// =========================================================================

use bitnet_inference::profiler::MemorySnapshot;

#[test]
fn w26_memory_snapshot_init() {
    let snap = MemorySnapshot {
        label: "model_loaded".into(),
        timestamp_us: 0.0,
        memory_bytes: 536_870_912,
    };
    insta::assert_debug_snapshot!(snap);
}

#[test]
fn w26_memory_snapshot_peak() {
    let snap = MemorySnapshot {
        label: "peak_forward_pass".into(),
        timestamp_us: 15000.5,
        memory_bytes: 1_073_741_824,
    };
    insta::assert_debug_snapshot!(snap);
}

// =========================================================================
// Section 7 — MemoryEstimation
// =========================================================================

use bitnet_inference::memory_estimation::{
    KvCacheEstimation, MemoryEstimation, ModelMemoryProfile,
};

#[test]
fn w26_memory_estimation_small_model() {
    let est = MemoryEstimation {
        model_params_bytes: 268_435_456,
        kv_cache_bytes: 67_108_864,
        activation_bytes: 33_554_432,
        total_bytes: 369_098_752,
        recommended_gpu_vram_gb: 0.41,
        recommended_system_ram_gb: 0.52,
    };
    insta::assert_debug_snapshot!(est);
}

#[test]
fn w26_memory_estimation_large_model() {
    let est = MemoryEstimation {
        model_params_bytes: 4_294_967_296,
        kv_cache_bytes: 1_073_741_824,
        activation_bytes: 536_870_912,
        total_bytes: 5_905_580_032,
        recommended_gpu_vram_gb: 6.6,
        recommended_system_ram_gb: 8.25,
    };
    insta::assert_debug_snapshot!(est);
}

#[test]
fn w26_kv_cache_estimation_f16() {
    let est = KvCacheEstimation {
        per_layer_bytes: 2_097_152,
        total_bytes: 67_108_864,
        num_layers: 32,
        num_kv_heads: 8,
        head_dim: 128,
        max_seq_len: 2048,
        dtype_bytes: 2,
    };
    insta::assert_debug_snapshot!(est);
}

#[test]
fn w26_kv_cache_estimation_f32() {
    let est = KvCacheEstimation {
        per_layer_bytes: 4_194_304,
        total_bytes: 100_663_296,
        num_layers: 24,
        num_kv_heads: 4,
        head_dim: 64,
        max_seq_len: 8192,
        dtype_bytes: 4,
    };
    insta::assert_debug_snapshot!(est);
}

#[test]
fn w26_model_memory_profile_empty() {
    let profile = ModelMemoryProfile {
        model_name: "bitnet-2b".into(),
        architecture: "bitnet".into(),
        known_profiles: vec![],
    };
    insta::assert_debug_snapshot!(profile);
}

#[test]
fn w26_model_memory_profile_with_entries() {
    let profile = ModelMemoryProfile {
        model_name: "bitnet-2b-4t".into(),
        architecture: "bitnet".into(),
        known_profiles: vec![
            (
                512,
                MemoryEstimation {
                    model_params_bytes: 500_000_000,
                    kv_cache_bytes: 16_777_216,
                    activation_bytes: 8_388_608,
                    total_bytes: 525_165_824,
                    recommended_gpu_vram_gb: 0.59,
                    recommended_system_ram_gb: 0.73,
                },
            ),
            (
                2048,
                MemoryEstimation {
                    model_params_bytes: 500_000_000,
                    kv_cache_bytes: 67_108_864,
                    activation_bytes: 33_554_432,
                    total_bytes: 600_663_296,
                    recommended_gpu_vram_gb: 0.67,
                    recommended_system_ram_gb: 0.84,
                },
            ),
        ],
    };
    insta::assert_debug_snapshot!(profile);
}

// =========================================================================
// Section 8 — PerformanceMode
// =========================================================================

use bitnet_inference::generation::autoregressive::PerformanceMode;

#[test]
fn w26_performance_mode_all_debug() {
    let modes = vec![
        PerformanceMode::Latency,
        PerformanceMode::Throughput,
        PerformanceMode::Balanced,
        PerformanceMode::Conservative,
    ];
    insta::assert_debug_snapshot!(modes);
}

#[test]
fn w26_performance_mode_latency() {
    insta::assert_debug_snapshot!(PerformanceMode::Latency);
}

#[test]
fn w26_performance_mode_conservative() {
    insta::assert_debug_snapshot!(PerformanceMode::Conservative);
}

// =========================================================================
// Section 9 — Profiler config JSON serialization
// =========================================================================

#[test]
fn w26_profiler_config_json_default() {
    let cfg = ProfilerConfig::default();
    let json = serde_json::to_string_pretty(&cfg).unwrap();
    insta::assert_snapshot!(json);
}

#[test]
fn w26_profiler_config_json_custom() {
    let cfg = ProfilerConfig {
        enabled: true,
        record_memory: true,
        warmup_iterations: 5,
        sample_size: 20,
    };
    let json = serde_json::to_string_pretty(&cfg).unwrap();
    insta::assert_snapshot!(json);
}

#[test]
fn w26_layer_profile_json() {
    let lp = LayerProfile {
        layer_name: "layer_0.attention".into(),
        layer_type: "attention".into(),
        forward_time_us: 1250.5,
        backward_time_us: 0.0,
        memory_bytes: 8_388_608,
        flops_estimate: 2_147_483_648,
    };
    let json = serde_json::to_string_pretty(&lp).unwrap();
    insta::assert_snapshot!(json);
}

#[test]
fn w26_memory_snapshot_json() {
    let snap = MemorySnapshot {
        label: "after_layer_0".into(),
        timestamp_us: 1250.5,
        memory_bytes: 67_108_864,
    };
    let json = serde_json::to_string_pretty(&snap).unwrap();
    insta::assert_snapshot!(json);
}

// =========================================================================
// Section 10 — estimate_kv_cache helper
// =========================================================================

use bitnet_inference::memory_estimation::estimate_kv_cache;

#[test]
fn w26_estimate_kv_cache_bitnet_2b() {
    let est = estimate_kv_cache(24, 8, 128, 2048, 2);
    insta::assert_debug_snapshot!(est);
}

#[test]
fn w26_estimate_kv_cache_llama_7b() {
    let est = estimate_kv_cache(32, 32, 128, 4096, 2);
    insta::assert_debug_snapshot!(est);
}

#[test]
fn w26_estimate_kv_cache_tiny() {
    let est = estimate_kv_cache(4, 2, 64, 512, 4);
    insta::assert_debug_snapshot!(est);
}

#[test]
fn w26_budget_tracker_exhausted() {
    let budget = GenerationBudget::new(5);
    let mut tracker = BudgetTracker::new(budget);
    for _ in 0..5 {
        tracker.record_token();
    }
    insta::assert_snapshot!(format!(
        "exhausted={} remaining={}",
        !tracker.can_continue(),
        tracker.tokens_remaining()
    ));
}
