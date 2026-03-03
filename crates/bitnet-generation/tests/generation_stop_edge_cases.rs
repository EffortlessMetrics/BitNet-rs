//! Edge-case tests for bitnet-generation stopping logic, config, and streaming types.
//!
//! These tests exercise boundary conditions in `check_stop`, priority ordering
//! of stop reasons, `GenerationConfig` defaults, serde roundtrips for all types,
//! and multi-SLM stop token configurations.

use bitnet_generation::*;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn criteria(stop_ids: &[u32], stop_strings: &[&str], max: usize, eos: Option<u32>) -> StopCriteria {
    StopCriteria {
        stop_token_ids: stop_ids.to_vec(),
        stop_strings: stop_strings.iter().map(|s| s.to_string()).collect(),
        max_tokens: max,
        eos_token_id: eos,
    }
}

// ---------------------------------------------------------------------------
// Priority ordering
// ---------------------------------------------------------------------------

#[test]
fn priority_stop_token_over_eos() {
    let c = criteria(&[2], &[], 0, Some(2));
    assert_eq!(check_stop(&c, 2, &[], ""), Some(StopReason::StopTokenId(2)));
}

#[test]
fn priority_stop_token_over_max_tokens() {
    let c = criteria(&[7], &[], 1, None);
    assert_eq!(check_stop(&c, 7, &[1], ""), Some(StopReason::StopTokenId(7)));
}

#[test]
fn priority_stop_token_over_stop_string() {
    let c = criteria(&[7], &["hello"], 0, None);
    assert_eq!(check_stop(&c, 7, &[], "hello"), Some(StopReason::StopTokenId(7)));
}

#[test]
fn priority_eos_over_max_tokens() {
    let c = criteria(&[], &[], 1, Some(42));
    assert_eq!(check_stop(&c, 42, &[1], ""), Some(StopReason::EosToken));
}

#[test]
fn priority_eos_over_stop_string() {
    let c = criteria(&[], &["hello"], 0, Some(42));
    assert_eq!(check_stop(&c, 42, &[], "hello"), Some(StopReason::EosToken));
}

#[test]
fn priority_max_tokens_over_stop_string() {
    let c = criteria(&[], &["hello"], 2, None);
    assert_eq!(check_stop(&c, 5, &[1, 2], "hello"), Some(StopReason::MaxTokens));
}

// ---------------------------------------------------------------------------
// StopCriteria edge cases
// ---------------------------------------------------------------------------

#[test]
fn default_criteria_never_stops() {
    let c = StopCriteria::default();
    assert!(check_stop(&c, 42, &[1, 2, 3], "hello world").is_none());
}

#[test]
fn empty_stop_ids_and_strings_no_stop() {
    let c = criteria(&[], &[], 0, None);
    assert!(check_stop(&c, 999, &vec![0u32; 1000], "any text").is_none());
}

#[test]
fn multiple_stop_ids_first_match_wins() {
    let c = criteria(&[10, 20, 30], &[], 0, None);
    assert_eq!(check_stop(&c, 20, &[], ""), Some(StopReason::StopTokenId(20)));
}

#[test]
fn multiple_stop_strings_first_match_wins() {
    let c = criteria(&[], &["<end>", "</s>", "[STOP]"], 0, None);
    let result = check_stop(&c, 5, &[], "text<end>more</s>");
    assert_eq!(result, Some(StopReason::StopString("<end>".to_string())));
}

#[test]
fn stop_string_at_start_of_tail() {
    let c = criteria(&[], &["STOP"], 0, None);
    assert_eq!(check_stop(&c, 5, &[], "STOP"), Some(StopReason::StopString("STOP".to_string())));
}

#[test]
fn stop_string_at_end_of_tail() {
    let c = criteria(&[], &["STOP"], 0, None);
    assert_eq!(
        check_stop(&c, 5, &[], "text text STOP"),
        Some(StopReason::StopString("STOP".to_string()))
    );
}

#[test]
fn stop_string_empty_string_always_matches() {
    let c = criteria(&[], &[""], 0, None);
    // "" is contained in any string
    assert_eq!(check_stop(&c, 5, &[], "anything"), Some(StopReason::StopString(String::new())));
}

#[test]
fn stop_string_unicode() {
    let c = criteria(&[], &["🛑"], 0, None);
    assert_eq!(
        check_stop(&c, 5, &[], "stop here 🛑 done"),
        Some(StopReason::StopString("🛑".to_string()))
    );
}

#[test]
fn max_tokens_boundary_one_below() {
    let c = criteria(&[], &[], 5, None);
    // generated has 4 tokens, max is 5 → not yet
    assert!(check_stop(&c, 5, &[1, 2, 3, 4], "").is_none());
}

#[test]
fn max_tokens_boundary_exact() {
    let c = criteria(&[], &[], 5, None);
    // generated has 5 tokens, max is 5 → stop
    assert_eq!(check_stop(&c, 5, &[1, 2, 3, 4, 5], ""), Some(StopReason::MaxTokens));
}

#[test]
fn max_tokens_boundary_one_above() {
    let c = criteria(&[], &[], 5, None);
    assert_eq!(check_stop(&c, 5, &[1, 2, 3, 4, 5, 6], ""), Some(StopReason::MaxTokens));
}

#[test]
fn max_tokens_one() {
    let c = criteria(&[], &[], 1, None);
    assert_eq!(check_stop(&c, 5, &[1], ""), Some(StopReason::MaxTokens));
}

#[test]
fn max_tokens_zero_means_unlimited() {
    let c = criteria(&[], &[], 0, None);
    assert!(check_stop(&c, 5, &vec![0u32; 100_000], "").is_none());
}

// ---------------------------------------------------------------------------
// Multi-SLM stop token configurations
// ---------------------------------------------------------------------------

#[test]
fn phi4_stop_tokens() {
    // Phi-4 uses EOS=100257, EOT=100265
    let c = criteria(&[100265], &[], 0, Some(100257));
    assert_eq!(check_stop(&c, 100257, &[], ""), Some(StopReason::EosToken));
    assert_eq!(check_stop(&c, 100265, &[], ""), Some(StopReason::StopTokenId(100265)));
}

#[test]
fn llama3_stop_tokens() {
    // LLaMA-3 uses EOS=128001, EOT=128009
    let c = criteria(&[128009], &[], 0, Some(128001));
    assert_eq!(check_stop(&c, 128001, &[], ""), Some(StopReason::EosToken));
    assert_eq!(check_stop(&c, 128009, &[], ""), Some(StopReason::StopTokenId(128009)));
}

#[test]
fn gemma_stop_tokens() {
    // Gemma uses EOS=1, <end_of_turn>=107
    let c = criteria(&[107], &["<end_of_turn>"], 0, Some(1));
    assert_eq!(check_stop(&c, 1, &[], ""), Some(StopReason::EosToken));
    assert_eq!(check_stop(&c, 107, &[], ""), Some(StopReason::StopTokenId(107)));
    assert_eq!(
        check_stop(&c, 5, &[], "text<end_of_turn>"),
        Some(StopReason::StopString("<end_of_turn>".to_string()))
    );
}

#[test]
fn qwen_stop_tokens() {
    // Qwen2.5 uses EOS=151645 <|im_end|>=151645, <|endoftext|>=151643
    let c = criteria(&[151643], &["<|im_end|>"], 0, Some(151645));
    assert_eq!(check_stop(&c, 151645, &[], ""), Some(StopReason::EosToken));
}

#[test]
fn mistral_stop_tokens() {
    // Mistral uses EOS=2, </s>=2
    let c = criteria(&[], &["</s>"], 0, Some(2));
    assert_eq!(check_stop(&c, 2, &[], ""), Some(StopReason::EosToken));
    assert_eq!(
        check_stop(&c, 5, &[], "output</s>"),
        Some(StopReason::StopString("</s>".to_string()))
    );
}

// ---------------------------------------------------------------------------
// GenerationConfig
// ---------------------------------------------------------------------------

#[test]
fn generation_config_default_values() {
    let cfg = GenerationConfig::default();
    assert_eq!(cfg.max_new_tokens, 128);
    assert!(cfg.seed.is_none());
    assert_eq!(cfg.stop_criteria.max_tokens, 0);
    assert!(cfg.stop_criteria.stop_token_ids.is_empty());
    assert!(cfg.stop_criteria.stop_strings.is_empty());
    assert!(cfg.stop_criteria.eos_token_id.is_none());
}

#[test]
fn generation_config_custom() {
    let cfg = GenerationConfig {
        max_new_tokens: 512,
        seed: Some(42),
        stop_criteria: criteria(&[128009], &["</s>"], 512, Some(128001)),
    };
    assert_eq!(cfg.max_new_tokens, 512);
    assert_eq!(cfg.seed, Some(42));
    assert_eq!(cfg.stop_criteria.max_tokens, 512);
}

// ---------------------------------------------------------------------------
// Serde roundtrips
// ---------------------------------------------------------------------------

#[test]
fn stop_criteria_serde_roundtrip() {
    let c = criteria(&[42, 100], &["end", "stop"], 256, Some(2));
    let json = serde_json::to_string(&c).unwrap();
    let c2: StopCriteria = serde_json::from_str(&json).unwrap();
    assert_eq!(c.stop_token_ids, c2.stop_token_ids);
    assert_eq!(c.stop_strings, c2.stop_strings);
    assert_eq!(c.max_tokens, c2.max_tokens);
    assert_eq!(c.eos_token_id, c2.eos_token_id);
}

#[test]
fn stop_reason_serde_roundtrip() {
    let reasons = vec![
        StopReason::MaxTokens,
        StopReason::StopTokenId(42),
        StopReason::StopString("</s>".to_string()),
        StopReason::EosToken,
    ];
    for reason in &reasons {
        let json = serde_json::to_string(reason).unwrap();
        let r2: StopReason = serde_json::from_str(&json).unwrap();
        assert_eq!(*reason, r2);
    }
}

#[test]
fn generation_config_serde_roundtrip() {
    let cfg = GenerationConfig {
        max_new_tokens: 64,
        seed: Some(42),
        stop_criteria: criteria(&[128009], &["</s>"], 64, Some(128001)),
    };
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: GenerationConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg.max_new_tokens, cfg2.max_new_tokens);
    assert_eq!(cfg.seed, cfg2.seed);
}

#[test]
fn token_event_serde_roundtrip() {
    let ev = TokenEvent { id: 42, text: "hello".to_string() };
    let json = serde_json::to_string(&ev).unwrap();
    let ev2: TokenEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(ev.id, ev2.id);
    assert_eq!(ev.text, ev2.text);
}

#[test]
fn generation_stats_serde_roundtrip() {
    let stats = GenerationStats { tokens_generated: 100, tokens_per_second: 42.5 };
    let json = serde_json::to_string(&stats).unwrap();
    let stats2: GenerationStats = serde_json::from_str(&json).unwrap();
    assert_eq!(stats.tokens_generated, stats2.tokens_generated);
    assert!((stats.tokens_per_second - stats2.tokens_per_second).abs() < 1e-10);
}

#[test]
fn stream_event_token_serde_roundtrip() {
    let ev = StreamEvent::Token(TokenEvent { id: 7, text: "world".to_string() });
    let json = serde_json::to_string(&ev).unwrap();
    let ev2: StreamEvent = serde_json::from_str(&json).unwrap();
    match ev2 {
        StreamEvent::Token(t) => {
            assert_eq!(t.id, 7);
            assert_eq!(t.text, "world");
        }
        _ => panic!("expected Token"),
    }
}

#[test]
fn stream_event_done_serde_roundtrip() {
    let ev = StreamEvent::Done {
        reason: StopReason::EosToken,
        stats: GenerationStats { tokens_generated: 10, tokens_per_second: 5.0 },
    };
    let json = serde_json::to_string(&ev).unwrap();
    let ev2: StreamEvent = serde_json::from_str(&json).unwrap();
    match ev2 {
        StreamEvent::Done { reason, stats } => {
            assert_eq!(reason, StopReason::EosToken);
            assert_eq!(stats.tokens_generated, 10);
        }
        _ => panic!("expected Done"),
    }
}

// ---------------------------------------------------------------------------
// Type defaults and construction
// ---------------------------------------------------------------------------

#[test]
fn stop_criteria_default_all_empty() {
    let c = StopCriteria::default();
    assert!(c.stop_token_ids.is_empty());
    assert!(c.stop_strings.is_empty());
    assert_eq!(c.max_tokens, 0);
    assert!(c.eos_token_id.is_none());
}

#[test]
fn generation_stats_default() {
    let s = GenerationStats::default();
    assert_eq!(s.tokens_generated, 0);
    assert_eq!(s.tokens_per_second, 0.0);
}

#[test]
fn token_event_clone() {
    let ev = TokenEvent { id: 42, text: "hello".to_string() };
    let ev2 = ev.clone();
    assert_eq!(ev.id, ev2.id);
    assert_eq!(ev.text, ev2.text);
}

#[test]
fn stop_reason_clone() {
    let r = StopReason::StopString("test".to_string());
    let r2 = r.clone();
    assert_eq!(r, r2);
}

// ---------------------------------------------------------------------------
// Stress: many stop conditions simultaneously
// ---------------------------------------------------------------------------

#[test]
fn many_stop_ids() {
    let ids: Vec<u32> = (0..1000).collect();
    let c = StopCriteria {
        stop_token_ids: ids,
        stop_strings: vec![],
        max_tokens: 0,
        eos_token_id: None,
    };
    // Token 500 should match
    assert_eq!(check_stop(&c, 500, &[], ""), Some(StopReason::StopTokenId(500)));
    // Token 1001 should not match
    assert!(check_stop(&c, 1001, &[], "").is_none());
}

#[test]
fn many_stop_strings() {
    let strings: Vec<String> = (0..100).map(|i| format!("STOP{i}")).collect();
    let c = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: strings,
        max_tokens: 0,
        eos_token_id: None,
    };
    // "STOP4" substring-matches "STOP42", so STOP4 fires first (index 4 < 42)
    assert_eq!(
        check_stop(&c, 5, &[], "text STOP42 more"),
        Some(StopReason::StopString("STOP4".to_string()))
    );
}

#[test]
fn long_decoded_tail() {
    let tail = "x".repeat(100_000) + "</s>";
    let c = criteria(&[], &["</s>"], 0, None);
    assert_eq!(check_stop(&c, 5, &[], &tail), Some(StopReason::StopString("</s>".to_string())));
}

#[test]
fn all_conditions_met_simultaneously() {
    // Token is both in stop_ids AND is eos AND max reached AND stop string present
    let c = criteria(&[42], &["hello"], 1, Some(42));
    // stop_token_id has highest priority
    assert_eq!(check_stop(&c, 42, &[1], "hello"), Some(StopReason::StopTokenId(42)));
}
