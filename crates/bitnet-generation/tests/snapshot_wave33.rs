//! Wave 33 snapshot tests for bitnet-generation.
//!
//! Covers: stop condition display, generation event formatting,
//! budget state display, StopCriteria variants, GenerationConfig serialization.

use bitnet_generation::{
    GenerationConfig, GenerationStats, StopCriteria, StopReason, StreamEvent, TokenEvent,
    check_stop,
};

// ── StopReason display variants ─────────────────────────────────────────────

#[test]
fn w33_stop_reason_max_tokens_debug() {
    insta::assert_debug_snapshot!(StopReason::MaxTokens);
}

#[test]
fn w33_stop_reason_stop_token_id_debug() {
    insta::assert_debug_snapshot!(StopReason::StopTokenId(128_009));
}

#[test]
fn w33_stop_reason_stop_string_debug() {
    insta::assert_debug_snapshot!(StopReason::StopString("</s>".to_string()));
}

#[test]
fn w33_stop_reason_eos_token_debug() {
    insta::assert_debug_snapshot!(StopReason::EosToken);
}

#[test]
fn w33_stop_reason_all_variants_json() {
    let reasons = vec![
        StopReason::MaxTokens,
        StopReason::StopTokenId(128_009),
        StopReason::StopString("\n\nQ:".to_string()),
        StopReason::EosToken,
    ];
    let json = serde_json::to_string_pretty(&reasons).unwrap();
    insta::assert_snapshot!(json);
}

// ── StopCriteria ────────────────────────────────────────────────────────────

#[test]
fn w33_stop_criteria_empty_json() {
    let c = StopCriteria::default();
    let json = serde_json::to_string_pretty(&c).unwrap();
    insta::assert_snapshot!(json);
}

#[test]
fn w33_stop_criteria_full_json() {
    let c = StopCriteria {
        stop_token_ids: vec![128_009, 2],
        stop_strings: vec!["</s>".to_string(), "\n\nQ:".to_string()],
        max_tokens: 256,
        eos_token_id: Some(2),
    };
    let json = serde_json::to_string_pretty(&c).unwrap();
    insta::assert_snapshot!(json);
}

#[test]
fn w33_stop_criteria_single_stop_id_debug() {
    let c = StopCriteria {
        stop_token_ids: vec![42],
        stop_strings: vec![],
        max_tokens: 0,
        eos_token_id: None,
    };
    insta::assert_debug_snapshot!(c);
}

// ── check_stop outcomes ─────────────────────────────────────────────────────

#[test]
fn w33_check_stop_no_trigger_debug() {
    let c = StopCriteria {
        stop_token_ids: vec![999],
        stop_strings: vec![],
        max_tokens: 100,
        eos_token_id: Some(998),
    };
    let result = check_stop(&c, 42, &[1, 2, 3], "hello world");
    insta::assert_debug_snapshot!(result);
}

#[test]
fn w33_check_stop_stop_token_hit_debug() {
    let c = StopCriteria {
        stop_token_ids: vec![42],
        stop_strings: vec![],
        max_tokens: 100,
        eos_token_id: None,
    };
    let result = check_stop(&c, 42, &[], "");
    insta::assert_debug_snapshot!(result);
}

#[test]
fn w33_check_stop_eos_hit_debug() {
    let c = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec![],
        max_tokens: 100,
        eos_token_id: Some(2),
    };
    let result = check_stop(&c, 2, &[], "");
    insta::assert_debug_snapshot!(result);
}

#[test]
fn w33_check_stop_max_tokens_hit_debug() {
    let c = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec![],
        max_tokens: 3,
        eos_token_id: None,
    };
    let result = check_stop(&c, 99, &[1, 2, 3], "");
    insta::assert_debug_snapshot!(result);
}

#[test]
fn w33_check_stop_string_hit_debug() {
    let c = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec!["</s>".to_string()],
        max_tokens: 100,
        eos_token_id: None,
    };
    let result = check_stop(&c, 99, &[], "some text</s>rest");
    insta::assert_debug_snapshot!(result);
}

// ── GenerationConfig ────────────────────────────────────────────────────────

#[test]
fn w33_generation_config_default_json() {
    let cfg = GenerationConfig::default();
    let json = serde_json::to_string_pretty(&cfg).unwrap();
    insta::assert_snapshot!(json);
}

#[test]
fn w33_generation_config_custom_json() {
    let cfg = GenerationConfig {
        max_new_tokens: 512,
        seed: Some(42),
        stop_criteria: StopCriteria {
            stop_token_ids: vec![128_009],
            stop_strings: vec!["<|eot_id|>".to_string()],
            max_tokens: 512,
            eos_token_id: Some(2),
        },
    };
    let json = serde_json::to_string_pretty(&cfg).unwrap();
    insta::assert_snapshot!(json);
}

#[test]
fn w33_generation_config_default_debug() {
    let cfg = GenerationConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

// ── GenerationStats ─────────────────────────────────────────────────────────

#[test]
fn w33_generation_stats_zero_json() {
    let stats = GenerationStats::default();
    let json = serde_json::to_string_pretty(&stats).unwrap();
    insta::assert_snapshot!(json);
}

#[test]
fn w33_generation_stats_populated_json() {
    let stats = GenerationStats { tokens_generated: 128, tokens_per_second: 23.7 };
    let json = serde_json::to_string_pretty(&stats).unwrap();
    insta::assert_snapshot!(json);
}

#[test]
fn w33_generation_stats_populated_debug() {
    let stats = GenerationStats { tokens_generated: 128, tokens_per_second: 23.7 };
    insta::assert_debug_snapshot!(stats);
}

// ── StreamEvent ─────────────────────────────────────────────────────────────

#[test]
fn w33_stream_event_token_json() {
    let event = StreamEvent::Token(TokenEvent { id: 4567, text: "world".to_string() });
    let json = serde_json::to_string_pretty(&event).unwrap();
    insta::assert_snapshot!(json);
}

#[test]
fn w33_stream_event_done_max_tokens_json() {
    let event = StreamEvent::Done {
        reason: StopReason::MaxTokens,
        stats: GenerationStats { tokens_generated: 64, tokens_per_second: 12.5 },
    };
    let json = serde_json::to_string_pretty(&event).unwrap();
    insta::assert_snapshot!(json);
}

#[test]
fn w33_stream_event_done_eos_json() {
    let event = StreamEvent::Done {
        reason: StopReason::EosToken,
        stats: GenerationStats { tokens_generated: 32, tokens_per_second: 8.0 },
    };
    let json = serde_json::to_string_pretty(&event).unwrap();
    insta::assert_snapshot!(json);
}

#[test]
fn w33_stream_event_done_stop_string_debug() {
    let event = StreamEvent::Done {
        reason: StopReason::StopString("\n\nHuman:".to_string()),
        stats: GenerationStats { tokens_generated: 15, tokens_per_second: 3.0 },
    };
    insta::assert_debug_snapshot!(event);
}

#[test]
fn w33_token_event_debug() {
    let event = TokenEvent { id: 42, text: "hello".to_string() };
    insta::assert_debug_snapshot!(event);
}
