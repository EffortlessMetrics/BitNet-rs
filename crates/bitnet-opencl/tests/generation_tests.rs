//! Integration tests for the GPU / CPU generation loop.

use bitnet_generation::StopReason;
use bitnet_opencl::{
    GenerationConfig, GenerationEngine, GenerationError, GenerationStats, MockModelBackend,
    MockTokenizer, StatsCollector, StoppingCriteria, StreamToken,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_engine(max_tokens: usize) -> GenerationEngine<MockModelBackend, MockTokenizer> {
    GenerationEngine::new(
        MockModelBackend::new(256),
        MockTokenizer,
        GenerationConfig { max_tokens, temperature: 0.0, ..Default::default() },
        StoppingCriteria { max_length: max_tokens, ..Default::default() },
    )
}

// ---------------------------------------------------------------------------
// Basic generation
// ---------------------------------------------------------------------------

#[test]
fn simple_generation_produces_tokens() {
    let mut e = make_engine(5);
    let res = e.generate("hi").unwrap();
    assert!(!res.tokens.is_empty());
    assert!(!res.output_text.is_empty());
}

#[test]
fn generation_returns_correct_token_count() {
    let mut e = make_engine(4);
    let res = e.generate("ab").unwrap();
    assert_eq!(res.tokens.len(), 4);
}

// ---------------------------------------------------------------------------
// Stopping criteria
// ---------------------------------------------------------------------------

#[test]
fn max_tokens_enforced() {
    let mut e = make_engine(3);
    let res = e.generate("hello").unwrap();
    assert!(res.tokens.len() <= 3);
    assert_eq!(res.stop_reason, StopReason::MaxTokens);
}

#[test]
fn stop_token_ends_generation() {
    let mut e = GenerationEngine::new(
        MockModelBackend::new(256),
        MockTokenizer,
        GenerationConfig { max_tokens: 100, temperature: 0.0, ..Default::default() },
        StoppingCriteria { max_length: 100, stop_tokens: vec![2], ..Default::default() },
    );
    let res = e.generate("ab").unwrap();
    assert_eq!(res.stop_reason, StopReason::StopTokenId(2));
}

#[test]
fn stop_string_ends_generation() {
    let mut e = GenerationEngine::new(
        MockModelBackend::new(128),
        MockTokenizer,
        GenerationConfig { max_tokens: 200, temperature: 0.0, ..Default::default() },
        StoppingCriteria {
            max_length: 200,
            stop_strings: vec!["\n".to_string()],
            ..Default::default()
        },
    );
    let res = e.generate("a").unwrap();
    assert!(matches!(res.stop_reason, StopReason::StopString(_)));
}

#[test]
fn eos_token_ends_generation() {
    let mut e = GenerationEngine::new(
        MockModelBackend::new(256),
        MockTokenizer,
        GenerationConfig { max_tokens: 100, temperature: 0.0, ..Default::default() },
        StoppingCriteria { max_length: 100, eos_token: Some(1), ..Default::default() },
    );
    let res = e.generate("x").unwrap();
    assert_eq!(res.stop_reason, StopReason::EosToken);
}

#[test]
fn multiple_stop_tokens_checked() {
    let mut e = GenerationEngine::new(
        MockModelBackend::new(256),
        MockTokenizer,
        GenerationConfig { max_tokens: 100, temperature: 0.0, ..Default::default() },
        StoppingCriteria { max_length: 100, stop_tokens: vec![99, 3], ..Default::default() },
    );
    let res = e.generate("abc").unwrap();
    assert_eq!(res.stop_reason, StopReason::StopTokenId(3));
}

#[test]
fn config_stop_sequences_are_honoured() {
    let mut e = GenerationEngine::new(
        MockModelBackend::new(128),
        MockTokenizer,
        GenerationConfig {
            max_tokens: 200,
            temperature: 0.0,
            stop_sequences: vec!["\n".to_string()],
            ..Default::default()
        },
        StoppingCriteria { max_length: 200, ..Default::default() },
    );
    let res = e.generate("a").unwrap();
    assert!(matches!(res.stop_reason, StopReason::StopString(_)));
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

#[test]
fn streaming_yields_tokens_incrementally() {
    let mut e = make_engine(5);
    let stream = e.generate_stream("hi").unwrap();
    let tokens: Vec<StreamToken> = stream.collect();
    assert_eq!(tokens.len(), 5);
    for (i, tok) in tokens.iter().enumerate() {
        assert_eq!(tok.index, i);
    }
}

#[test]
fn streaming_respects_max_tokens() {
    let mut e = make_engine(3);
    let stream = e.generate_stream("a").unwrap();
    let tokens: Vec<StreamToken> = stream.collect();
    assert!(tokens.len() <= 3);
}

#[test]
fn streaming_stop_token() {
    let mut e = GenerationEngine::new(
        MockModelBackend::new(256),
        MockTokenizer,
        GenerationConfig { max_tokens: 100, temperature: 0.0, ..Default::default() },
        StoppingCriteria { max_length: 100, stop_tokens: vec![2], ..Default::default() },
    );
    let stream = e.generate_stream("ab").unwrap();
    let tokens: Vec<StreamToken> = stream.collect();
    // Should stop at or before token 2.
    assert!(tokens.len() <= 3);
}

#[test]
fn streaming_empty_prompt_is_error() {
    let mut e = make_engine(5);
    let res = e.generate_stream("");
    assert!(res.is_err());
}

#[test]
fn stream_tokens_have_text() {
    let mut e = make_engine(3);
    let stream = e.generate_stream("a").unwrap();
    for tok in stream {
        assert!(!tok.text.is_empty());
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

#[test]
fn stats_recorded_correctly() {
    let mut e = make_engine(5);
    let res = e.generate("hi").unwrap();
    assert_eq!(res.stats.tokens_generated, res.tokens.len());
    assert!(res.stats.total_time_ms >= 0.0);
    assert!(res.stats.prefill_time_ms >= 0.0);
    assert!(res.stats.decode_time_ms >= 0.0);
}

#[test]
fn stats_collector_records_phases() {
    let mut c = StatsCollector::new();
    c.begin_prefill();
    c.end_prefill();
    c.record_token();
    c.record_token();
    let s = c.finish();
    assert_eq!(s.tokens_generated, 2);
    assert!(s.prefill_time_ms >= 0.0);
    assert!(s.decode_time_ms >= 0.0);
}

#[test]
fn stats_json_round_trip() {
    let s = GenerationStats {
        total_time_ms: 42.0,
        prefill_time_ms: 10.0,
        decode_time_ms: 32.0,
        tokens_generated: 8,
        tokens_per_second: 250.0,
        first_token_latency_ms: 5.0,
        peak_memory_bytes: 2048,
    };
    let json = serde_json::to_string(&s).unwrap();
    let de: GenerationStats = serde_json::from_str(&json).unwrap();
    assert_eq!(de.tokens_generated, 8);
    assert!((de.total_time_ms - 42.0).abs() < f64::EPSILON);
}

#[test]
fn stats_format_is_human_readable() {
    let s = GenerationStats {
        total_time_ms: 100.0,
        tokens_generated: 10,
        tokens_per_second: 100.0,
        ..Default::default()
    };
    let text = s.format_stats();
    assert!(text.contains("tokens generated"));
    assert!(text.contains("100.00 tok/s"));
}

#[test]
fn stats_display_trait() {
    let s = GenerationStats {
        tokens_generated: 3,
        total_time_ms: 30.0,
        tokens_per_second: 100.0,
        ..Default::default()
    };
    let text = format!("{s}");
    assert!(text.contains("3 tokens"));
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn empty_prompt_returns_error() {
    let mut e = make_engine(10);
    let err = e.generate("").unwrap_err();
    assert!(matches!(err, GenerationError::EmptyPrompt));
}

#[test]
fn error_display_is_descriptive() {
    let err = GenerationError::EmptyPrompt;
    let msg = format!("{err}");
    assert!(msg.contains("empty"));
}

// ---------------------------------------------------------------------------
// Determinism & sampling
// ---------------------------------------------------------------------------

#[test]
fn greedy_is_deterministic() {
    let mut e1 = make_engine(5);
    let mut e2 = make_engine(5);
    let r1 = e1.generate("ab").unwrap();
    let r2 = e2.generate("ab").unwrap();
    assert_eq!(r1.tokens, r2.tokens);
}

#[test]
fn seed_produces_same_output() {
    let cfg =
        GenerationConfig { max_tokens: 5, temperature: 0.7, seed: Some(42), ..Default::default() };
    let stopping = StoppingCriteria { max_length: 5, ..Default::default() };

    let mut e1 = GenerationEngine::new(
        MockModelBackend::new(256),
        MockTokenizer,
        cfg.clone(),
        stopping.clone(),
    );
    let mut e2 = GenerationEngine::new(MockModelBackend::new(256), MockTokenizer, cfg, stopping);

    let r1 = e1.generate("test").unwrap();
    let r2 = e2.generate("test").unwrap();
    assert_eq!(r1.tokens, r2.tokens);
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

#[test]
fn generation_result_json_round_trip() {
    let mut e = make_engine(3);
    let res = e.generate("hi").unwrap();
    let json = serde_json::to_string(&res).unwrap();
    assert!(json.contains("output_text"));
    assert!(json.contains("stop_reason"));
    assert!(json.contains("tokens"));
}

#[test]
fn generation_config_json_round_trip() {
    let cfg = GenerationConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let de: GenerationConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(de.max_tokens, cfg.max_tokens);
}

#[test]
fn stopping_criteria_json_round_trip() {
    let sc = StoppingCriteria {
        max_length: 42,
        stop_tokens: vec![1, 2],
        stop_strings: vec!["end".to_string()],
        eos_token: Some(99),
    };
    let json = serde_json::to_string(&sc).unwrap();
    let de: StoppingCriteria = serde_json::from_str(&json).unwrap();
    assert_eq!(de.max_length, 42);
    assert_eq!(de.stop_tokens, vec![1, 2]);
}
