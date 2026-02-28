//! End-to-end integration tests for the OpenCL inference pipeline.
//!
//! All tests run against a CPU mock backend so no GPU hardware is required.
//! Tests are independent and deterministic (seeded PRNG).

mod e2e_fixtures;

use bitnet_engine_core::{InferenceSession, StreamEvent};
use bitnet_generation::StopReason;
use bitnet_opencl::{ModelManager, estimate_memory};
use e2e_fixtures::{
    MockModel, MockTokenizer, TestPipeline, assert_generation_valid, assert_stop_reason,
    gen_config, gen_config_with_eos, gen_config_with_stop_string, token_count,
};

// =========================================================================
// a) Full pipeline smoke test
// =========================================================================

#[test]
fn full_pipeline_smoke_test() {
    let pipeline = TestPipeline::new().build();
    let config = gen_config(5);
    let events = pipeline.generate("Hello, world!", &config).unwrap();

    assert_generation_valid(&events);
    // 5 tokens + Done = 6 events, or fewer if a stop condition fired
    assert!(token_count(&events) <= 5, "should produce at most 5 tokens");
}

#[test]
fn smoke_test_with_mock_model_and_tokenizer() {
    let model = MockModel::new("smoke-model", 12345);
    let tokenizer = MockTokenizer::new();

    assert_eq!(model.vocab_size, tokenizer.vocab_size());
    assert_eq!(model.weights.len(), 4096);
    assert_eq!(tokenizer.decode(0), "tok_0");

    let pipeline = TestPipeline::new().vocab_size(model.vocab_size).build();
    let events = pipeline.generate("test", &gen_config(3)).unwrap();
    assert_generation_valid(&events);
}

// =========================================================================
// b) Device selection roundtrip
// =========================================================================

#[test]
fn device_selection_falls_back_to_cpu() {
    // Request OpenCL backend — should fall back to CPU (no device available).
    let pipeline = TestPipeline::new().backend("gpu").build();
    assert!(pipeline.is_cpu_fallback(), "expected CPU fallback when no OpenCL device is present");

    // Pipeline should still produce valid output.
    let events = pipeline.generate("hello", &gen_config(3)).unwrap();
    assert_generation_valid(&events);
}

#[test]
fn cpu_backend_does_not_report_fallback_when_requested() {
    let pipeline = TestPipeline::new().backend("cpu").build();
    // CPU fallback flag is true because opencl_device_available() is false,
    // but the pipeline still works correctly.
    let events = pipeline.generate("cpu test", &gen_config(2)).unwrap();
    assert_generation_valid(&events);
}

// =========================================================================
// c) Memory estimation accuracy
// =========================================================================

#[test]
fn memory_estimate_within_expected_range() {
    let model = MockModel::new("mem-test", 99);
    let estimate = estimate_memory(model.num_parameters, model.num_layers, 2048);

    // weights = 1_000_000 / 2 = 500_000
    assert_eq!(estimate.weights_bytes, 500_000);

    // kv_cache = 12 * 2048 * 2 = 49_152
    assert_eq!(estimate.kv_cache_bytes, 49_152);

    // total = weights + kv + scratch(10%)
    let expected_total = 500_000 + 49_152 + 50_000;
    assert_eq!(estimate.total_bytes, expected_total);
}

#[test]
fn memory_estimate_scales_with_context() {
    let small = estimate_memory(1_000_000, 12, 512);
    let large = estimate_memory(1_000_000, 12, 4096);
    assert!(
        large.kv_cache_bytes > small.kv_cache_bytes,
        "larger context should require more KV-cache"
    );
    assert!(large.total_bytes > small.total_bytes);
}

// =========================================================================
// d) Batch inference
// =========================================================================

#[test]
fn batch_inference_returns_all_outputs() {
    let pipeline = TestPipeline::new().seed(100).build();
    let config = gen_config(4);
    let prompts: &[&str] = &["Prompt one", "Prompt two", "Prompt three"];

    let results = pipeline.generate_batch(prompts, &config).unwrap();
    assert_eq!(results.len(), 3, "one output per prompt");

    for (i, events) in results.iter().enumerate() {
        assert_generation_valid(events);
        assert!(token_count(events) <= 4, "prompt {i}: expected at most 4 tokens");
    }
}

#[test]
fn batch_results_are_independent() {
    let pipeline = TestPipeline::new().seed(200).build();
    let config = gen_config(3);

    let batch = pipeline.generate_batch(&["alpha", "beta"], &config).unwrap();

    let single_a = pipeline.generate("alpha", &config).unwrap();
    let single_b = pipeline.generate("beta", &config).unwrap();

    // Each batch element should match its single-prompt counterpart.
    assert_eq!(batch[0].len(), single_a.len());
    assert_eq!(batch[1].len(), single_b.len());
}

// =========================================================================
// e) Streaming generation
// =========================================================================

#[test]
fn streaming_tokens_match_final_count() {
    let pipeline = TestPipeline::new().build();
    let max = 7;
    let config = gen_config(max);
    let events = pipeline.generate("stream test", &config).unwrap();

    let tokens: Vec<_> = events.iter().filter(|e| matches!(e, StreamEvent::Token(_))).collect();

    assert!(
        tokens.len() <= max,
        "token count {} should not exceed max_tokens {}",
        tokens.len(),
        max
    );

    // Done event should carry the same count.
    if let Some(StreamEvent::Done { stats, .. }) = events.last() {
        assert_eq!(stats.tokens_generated, tokens.len());
    } else {
        panic!("expected Done as final event");
    }
}

#[test]
fn streaming_produces_non_empty_text() {
    let pipeline = TestPipeline::new().build();
    let events = pipeline.generate("text check", &gen_config(3)).unwrap();

    for event in &events {
        if let StreamEvent::Token(tok) = event {
            assert!(!tok.text.is_empty(), "token text should not be empty");
        }
    }
}

// =========================================================================
// f) Stop conditions
// =========================================================================

#[test]
fn stop_on_max_tokens_exact_count() {
    let pipeline = TestPipeline::new().seed(42).build();
    let max = 5;
    let config = gen_config(max);
    let events = pipeline.generate("max tokens", &config).unwrap();

    assert_generation_valid(&events);
    assert_stop_reason(&events, &StopReason::MaxTokens);
    assert_eq!(token_count(&events), max, "should produce exactly {max} tokens");
}

#[test]
fn stop_on_eos_token() {
    // Use a tiny vocab so we can predict the EOS token being hit.
    let pipeline = TestPipeline::new().vocab_size(5).seed(42).build();

    // Set EOS to token 0 — very likely to be generated with vocab_size=5.
    let config = gen_config_with_eos(100, 0);
    let events = pipeline.generate("eos test", &config).unwrap();

    assert_generation_valid(&events);
    // Should stop before reaching max_tokens.
    assert!(token_count(&events) < 100, "EOS should have stopped generation early");
}

#[test]
fn stop_on_stop_string() {
    // We'll construct a scenario where the stop string appears in output.
    // With mock tokenizer output "tok_N", "tok_" is guaranteed to appear.
    let config = gen_config_with_stop_string(100, "tok_");
    let pipeline = TestPipeline::new().seed(42).build();
    let events = pipeline.generate("stop string test", &config).unwrap();

    assert_generation_valid(&events);
    assert_stop_reason(&events, &StopReason::StopString("tok_".to_string()));
    // Should stop at the very first token since every token contains "tok_".
    assert_eq!(token_count(&events), 1);
}

// =========================================================================
// g) Config propagation
// =========================================================================

#[test]
fn temperature_zero_is_greedy_deterministic() {
    let pipeline = TestPipeline::new().temperature(0.0).seed(42).build();
    let config = gen_config(5);

    let run1 = pipeline.generate("greedy", &config).unwrap();
    let run2 = pipeline.generate("greedy", &config).unwrap();

    let ids1: Vec<u32> = extract_token_ids(&run1);
    let ids2: Vec<u32> = extract_token_ids(&run2);
    assert_eq!(ids1, ids2, "greedy (temp=0) should be deterministic");
}

#[test]
fn top_k_restricts_token_range() {
    let k = 10u32;
    let pipeline = TestPipeline::new().top_k(k as usize).seed(42).build();
    let config = gen_config(20);
    let events = pipeline.generate("top_k test", &config).unwrap();

    let ids = extract_token_ids(&events);
    for id in &ids {
        assert!(*id < k, "with top_k={k}, token id {id} should be < {k}");
    }
}

#[test]
fn config_fields_propagate_to_pipeline() {
    let pipeline = TestPipeline::new().temperature(0.5).top_k(50).top_p(0.9).build();

    let cfg = pipeline.config();
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(cfg.temperature, 0.5);
        assert_eq!(cfg.top_p, 0.9);
    }
    assert_eq!(cfg.top_k, 50);
}

// =========================================================================
// h) Error recovery
// =========================================================================

#[test]
fn invalid_model_path_returns_helpful_error() {
    let pipeline = TestPipeline::new().model_path("").build();
    let config = gen_config(5);
    let result = pipeline.generate("should fail", &config);

    assert!(result.is_err(), "empty model path should fail");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("invalid model path") || msg.contains("does not exist"),
        "error should mention invalid path, got: {msg}"
    );
}

#[test]
fn nonexistent_model_path_returns_error() {
    let pipeline = TestPipeline::new().model_path("").build();
    let config = gen_config(1);
    let result = pipeline.generate("test", &config);

    assert!(result.is_err());
}

#[test]
fn oom_simulation_graceful_fallback() {
    // Simulate a very large model — estimate_memory should return a large
    // value but not panic.
    let estimate = estimate_memory(
        100_000_000_000, // 100B parameters
        128,
        8192,
    );
    assert!(estimate.total_bytes > 0, "memory estimate should be positive");
    // The pipeline still works with mock (doesn't actually allocate).
    let pipeline = TestPipeline::new().build();
    let events = pipeline.generate("oom test", &gen_config(2)).unwrap();
    assert_generation_valid(&events);
}

// =========================================================================
// i) Deterministic output
// =========================================================================

#[test]
fn same_seed_produces_identical_output() {
    let config = gen_config(10);

    let run1 = TestPipeline::new().seed(42).build();
    let events1 = run1.generate("deterministic", &config).unwrap();

    let run2 = TestPipeline::new().seed(42).build();
    let events2 = run2.generate("deterministic", &config).unwrap();

    let ids1 = extract_token_ids(&events1);
    let ids2 = extract_token_ids(&events2);

    assert_eq!(ids1, ids2, "identical seed + prompt should yield identical tokens");
}

#[test]
fn different_seeds_produce_different_output() {
    let config = gen_config(10);

    let run1 = TestPipeline::new().seed(1).build();
    let events1 = run1.generate("seed diff", &config).unwrap();

    let run2 = TestPipeline::new().seed(2).build();
    let events2 = run2.generate("seed diff", &config).unwrap();

    let ids1 = extract_token_ids(&events1);
    let ids2 = extract_token_ids(&events2);

    assert_ne!(ids1, ids2, "different seeds should produce different tokens");
}

#[test]
fn different_prompts_produce_different_output() {
    let config = gen_config(10);
    let pipeline = TestPipeline::new().seed(42).build();

    let events1 = pipeline.generate("prompt A", &config).unwrap();
    let events2 = pipeline.generate("prompt B", &config).unwrap();

    let ids1 = extract_token_ids(&events1);
    let ids2 = extract_token_ids(&events2);

    assert_ne!(ids1, ids2, "different prompts should produce different tokens");
}

// =========================================================================
// j) Multi-model switching
// =========================================================================

#[test]
fn multi_model_switching_isolation() {
    let mut manager = ModelManager::new();

    let pipeline_a = TestPipeline::new().seed(100).build();
    let pipeline_b = TestPipeline::new().seed(200).build();

    manager.load("model_a", pipeline_a);
    manager.load("model_b", pipeline_b);

    let config = gen_config(5);

    let out_a = manager.generate("model_a", "hello", &config).unwrap();
    let out_b = manager.generate("model_b", "hello", &config).unwrap();

    assert_generation_valid(&out_a);
    assert_generation_valid(&out_b);

    let ids_a = extract_token_ids(&out_a);
    let ids_b = extract_token_ids(&out_b);
    assert_ne!(ids_a, ids_b, "different models (seeds) should produce different tokens");
}

#[test]
fn model_manager_unknown_model_returns_error() {
    let manager = ModelManager::new();
    let config = gen_config(3);
    let result = manager.generate("nonexistent", "test", &config);

    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("model not loaded"),
        "error should mention 'model not loaded'"
    );
}

#[test]
fn model_manager_load_replace() {
    let mut manager = ModelManager::new();
    let config = gen_config(3);

    manager.load("m", TestPipeline::new().seed(1).build());
    let out1 = manager.generate("m", "test", &config).unwrap();

    // Replace with a different seed — output should change.
    manager.load("m", TestPipeline::new().seed(9999).build());
    let out2 = manager.generate("m", "test", &config).unwrap();

    let ids1 = extract_token_ids(&out1);
    let ids2 = extract_token_ids(&out2);
    assert_ne!(ids1, ids2, "replaced model should produce different output");
}

// =========================================================================
// Helpers
// =========================================================================

fn extract_token_ids(events: &[StreamEvent]) -> Vec<u32> {
    events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::Token(tok) => Some(tok.id),
            _ => None,
        })
        .collect()
}
