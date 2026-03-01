//! Cross-crate integration tests combining bitnet-engine-core, bitnet-generation,
//! and bitnet-logits to verify the full inference pipeline contract.
//!
//! These tests ensure that logits processing → stop-criteria checking → session
//! configuration all work together coherently.

use bitnet_engine_core::{ConfigError, EngineState, EngineStateTracker, SessionConfig};
use bitnet_generation::{GenerationConfig, StopCriteria, StopReason};
use bitnet_logits::{
    apply_repetition_penalty, apply_temperature, apply_top_k, argmax, softmax_in_place,
};

// --- Logits → Generation pipeline tests ---

#[test]
fn logits_pipeline_feeds_into_stop_criteria() {
    // Simulate a real inference step: process logits, pick a token, check stop
    let mut logits = vec![1.0f32, 5.0, 2.0, 0.5, 3.0];

    // Apply temperature and sampling
    apply_temperature(&mut logits, 0.7);
    softmax_in_place(&mut logits);
    let selected = argmax(&logits);
    assert_eq!(selected, 1); // Token 1 had highest logit

    // Check stop criteria
    let criteria = StopCriteria {
        stop_token_ids: vec![4],
        stop_strings: vec![],
        max_tokens: 100,
        eos_token_id: Some(99),
    };
    let result = bitnet_generation::check_stop(&criteria, selected as u32, &[], "");
    assert!(result.is_none()); // Token 1 is not a stop token
}

#[test]
fn logits_pipeline_detects_stop_token() {
    let mut logits = vec![0.1f32, 0.1, 0.1, 0.1, 10.0]; // Token 4 dominant

    apply_temperature(&mut logits, 1.0);
    softmax_in_place(&mut logits);
    let selected = argmax(&logits);
    assert_eq!(selected, 4);

    let criteria = StopCriteria {
        stop_token_ids: vec![4],
        stop_strings: vec![],
        max_tokens: 100,
        eos_token_id: None,
    };
    let result = bitnet_generation::check_stop(&criteria, selected as u32, &[], "");
    assert!(matches!(result, Some(StopReason::StopTokenId(4))));
}

#[test]
fn logits_pipeline_detects_eos_token() {
    let eos_id = 99u32;
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec![],
        max_tokens: 100,
        eos_token_id: Some(eos_id),
    };
    let result = bitnet_generation::check_stop(&criteria, eos_id, &[], "");
    assert!(matches!(result, Some(StopReason::EosToken)));
}

#[test]
fn repetition_penalty_changes_selection() {
    // Without penalty, token 1 wins
    let base = vec![3.0f32, 5.0, 4.0, 1.0];
    assert_eq!(argmax(&base), 1);

    // With penalty on token 1 (already generated), another token should win
    let mut penalized = base.clone();
    apply_repetition_penalty(&mut penalized, &[1], 5.0);
    let new_selection = argmax(&penalized);
    assert_ne!(new_selection, 1, "Repetition penalty should steer away from token 1");
}

#[test]
fn top_k_then_argmax_restricts_candidates() {
    let mut logits = vec![1.0f32, 5.0, 3.0, 4.0, 2.0];
    let kept = apply_top_k(&mut logits, 2);
    assert_eq!(kept, 2);
    // Only top 2 should be non-negative-infinity
    let selected = argmax(&logits);
    assert!(selected == 1 || selected == 3); // Tokens with logits 5.0 and 4.0
}

// --- Generation config tests ---

#[test]
fn generation_config_with_stop_criteria() {
    let config = GenerationConfig {
        max_new_tokens: 512,
        seed: Some(42),
        stop_criteria: StopCriteria {
            stop_token_ids: vec![0, 1, 2],
            stop_strings: vec!["<|end|>".to_string()],
            max_tokens: 512,
            eos_token_id: Some(2),
        },
    };
    assert_eq!(config.max_new_tokens, 512);
    assert_eq!(config.seed, Some(42));
    assert_eq!(config.stop_criteria.stop_token_ids.len(), 3);
}

#[test]
fn max_tokens_stop_after_limit() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec![],
        max_tokens: 5,
        eos_token_id: None,
    };

    // Generate 5 tokens - should trigger max_tokens stop
    let generated: Vec<u32> = vec![10, 20, 30, 40, 50];
    let result = bitnet_generation::check_stop(&criteria, 60, &generated, "hello world");
    assert!(matches!(result, Some(StopReason::MaxTokens)));
}

// --- SessionConfig validation tests ---

#[test]
fn session_config_valid() {
    let config = SessionConfig {
        model_path: "model.gguf".to_string(),
        tokenizer_path: "tokenizer.json".to_string(),
        backend: "cpu".to_string(),
        max_context: 4096,
        seed: Some(42),
    };
    assert!(config.validate().is_ok());
}

#[test]
fn session_config_empty_model_path() {
    let config = SessionConfig {
        model_path: "".to_string(),
        tokenizer_path: "tokenizer.json".to_string(),
        backend: "cpu".to_string(),
        max_context: 4096,
        seed: None,
    };
    assert!(matches!(config.validate(), Err(ConfigError::EmptyModelPath)));
}

#[test]
fn session_config_empty_tokenizer_path() {
    let config = SessionConfig {
        model_path: "model.gguf".to_string(),
        tokenizer_path: "".to_string(),
        backend: "cpu".to_string(),
        max_context: 4096,
        seed: None,
    };
    assert!(matches!(config.validate(), Err(ConfigError::EmptyTokenizerPath)));
}

#[test]
fn session_config_zero_context() {
    let config = SessionConfig {
        model_path: "model.gguf".to_string(),
        tokenizer_path: "tokenizer.json".to_string(),
        backend: "cpu".to_string(),
        max_context: 0,
        seed: None,
    };
    assert!(matches!(config.validate(), Err(ConfigError::ZeroContextWindow)));
}

// --- Engine state tracker tests ---

#[test]
fn engine_state_tracker_transitions() {
    let tracker = EngineStateTracker::new();
    assert_eq!(*tracker.state(), EngineState::Idle);
}

// --- Full pipeline simulation tests ---

#[test]
fn simulated_phi4_inference_step() {
    // Simulate a single Phi-4 inference step:
    // 1. Config with 16K context
    let config = SessionConfig {
        model_path: "phi-4.gguf".to_string(),
        tokenizer_path: "phi-4-tokenizer.json".to_string(),
        backend: "cpu".to_string(),
        max_context: 16384,
        seed: Some(42),
    };
    assert!(config.validate().is_ok());

    // 2. Process logits (simulate 100K vocab)
    let vocab_size = 100352;
    let mut logits: Vec<f32> = vec![0.0; vocab_size];
    // Make a specific token dominant
    logits[42] = 100.0;

    apply_temperature(&mut logits, 0.8);
    softmax_in_place(&mut logits);
    let selected = argmax(&logits);
    assert_eq!(selected, 42);

    // 3. Check generation stop
    let stop = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec![],
        max_tokens: 32,
        eos_token_id: Some(100265), // Phi-4 EOS
    };
    let result = bitnet_generation::check_stop(&stop, selected as u32, &[], "");
    assert!(result.is_none()); // Token 42 is not EOS
}

#[test]
fn simulated_llama_inference_step() {
    // Simulate LLaMA inference with 32K vocab
    let config = SessionConfig {
        model_path: "llama.gguf".to_string(),
        tokenizer_path: "llama-tokenizer.json".to_string(),
        backend: "cpu".to_string(),
        max_context: 4096,
        seed: Some(123),
    };
    assert!(config.validate().is_ok());

    let mut logits = vec![0.0f32; 32000];
    logits[1] = 10.0; // BOS token dominant

    apply_temperature(&mut logits, 1.0);
    let kept = apply_top_k(&mut logits, 50);
    assert!(kept <= 50);
    softmax_in_place(&mut logits);

    let selected = argmax(&logits);
    assert_eq!(selected, 1); // BOS should still be dominant
}

#[test]
fn simulated_multi_turn_generation() {
    let stop = StopCriteria {
        stop_token_ids: vec![100],
        stop_strings: vec!["<|end|>".to_string()],
        max_tokens: 10,
        eos_token_id: Some(2),
    };

    // Simulate 4 tokens generated
    let mut generated = vec![];
    let tokens = [5, 10, 15, 100]; // Last token is a stop token

    for &tok_id in &tokens {
        generated.push(tok_id);
        let result =
            bitnet_generation::check_stop(&stop, tok_id, &generated[..generated.len() - 1], "");
        if tok_id == 100 {
            assert!(matches!(result, Some(StopReason::StopTokenId(100))));
        } else {
            assert!(result.is_none());
        }
    }
}

#[test]
fn logits_pipeline_deterministic_with_greedy() {
    // Greedy decoding (temperature=0 or argmax) should be deterministic
    let logits = vec![1.0f32, 3.0, 2.0, 5.0, 4.0];

    let r1 = argmax(&logits);
    let r2 = argmax(&logits);
    let r3 = argmax(&logits);

    assert_eq!(r1, r2);
    assert_eq!(r2, r3);
    assert_eq!(r1, 3); // Token 3 has logit 5.0
}

#[test]
fn stop_string_detection() {
    let stop = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec!["```".to_string()],
        max_tokens: 100,
        eos_token_id: None,
    };

    // Token doesn't trigger stop, but decoded tail contains stop string
    let result = bitnet_generation::check_stop(&stop, 42, &[1, 2, 3], "some code```");
    assert!(matches!(result, Some(StopReason::StopString(_))));
}

#[test]
fn combined_temperature_penalty_topk_pipeline() {
    let mut logits = vec![2.0f32, 8.0, 3.0, 7.0, 1.0, 6.0, 4.0, 5.0];

    // Full pipeline: temperature → repetition penalty → top-k → softmax → argmax
    apply_temperature(&mut logits, 0.5);
    apply_repetition_penalty(&mut logits, &[1], 2.0); // Penalize token 1 (was highest)
    let _kept = apply_top_k(&mut logits, 3);
    softmax_in_place(&mut logits);
    let selected = argmax(&logits);

    // Token 1 was penalized, so token 3 (logit 7.0) should be selected
    assert_eq!(selected, 3);
}
