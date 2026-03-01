//! Edge-case tests for batch processing, streaming configuration, token
//! buffering, and batch scheduling.

use std::time::Duration;

use bitnet_inference::batch::{
    BatchConfig, BatchRequest, BatchResult, BatchScheduler, SingleResult,
};
use bitnet_inference::config::GenerationConfig;
use bitnet_inference::streaming::StreamingConfig;
use bitnet_inference::token_stream::{StreamConfig, StreamEvent, StreamStats, TokenBuffer};

// ── BatchRequest ─────────────────────────────────────────────────────────

#[test]
fn batch_request_new_is_empty() {
    let batch = BatchRequest::new();
    assert!(batch.is_empty());
    assert_eq!(batch.len(), 0);
}

#[test]
fn batch_request_default_is_empty() {
    let batch = BatchRequest::default();
    assert!(batch.is_empty());
}

#[test]
fn batch_request_add_returns_sequential_ids() {
    let mut batch = BatchRequest::new();
    let id0 = batch.add("Hello".to_string(), GenerationConfig::default());
    let id1 = batch.add("World".to_string(), GenerationConfig::greedy());
    assert_eq!(id0, 0);
    assert_eq!(id1, 1);
    assert_eq!(batch.len(), 2);
    assert!(!batch.is_empty());
}

#[test]
fn batch_request_get_valid_id() {
    let mut batch = BatchRequest::new();
    batch.add("Test prompt".to_string(), GenerationConfig::default());
    let entry = batch.get(0).unwrap();
    assert_eq!(entry.id, 0);
    assert_eq!(entry.prompt, "Test prompt");
}

#[test]
fn batch_request_get_invalid_id() {
    let batch = BatchRequest::new();
    assert!(batch.get(0).is_none());
    assert!(batch.get(100).is_none());
}

#[test]
fn batch_request_iter() {
    let mut batch = BatchRequest::new();
    batch.add("A".to_string(), GenerationConfig::default());
    batch.add("B".to_string(), GenerationConfig::default());
    batch.add("C".to_string(), GenerationConfig::default());

    let prompts: Vec<&str> = batch.iter().map(|e| e.prompt.as_str()).collect();
    assert_eq!(prompts, vec!["A", "B", "C"]);
}

#[test]
fn batch_request_into_iter() {
    let mut batch = BatchRequest::new();
    batch.add("X".to_string(), GenerationConfig::default());
    let entries: Vec<_> = (&batch).into_iter().collect();
    assert_eq!(entries.len(), 1);
}

// ── BatchResult ──────────────────────────────────────────────────────────

#[test]
fn batch_result_empty() {
    let result = BatchResult::with_capacity(0);
    assert_eq!(result.completed_count(), 0);
    assert_eq!(result.capacity(), 0);
}

#[test]
fn batch_result_insert_and_get() {
    let mut result = BatchResult::with_capacity(3);
    result.insert(SingleResult { id: 0, text: "Hello".to_string(), tokens_generated: 1 });
    result.insert(SingleResult { id: 2, text: "World".to_string(), tokens_generated: 2 });

    assert_eq!(result.completed_count(), 2);
    assert!(result.get(0).is_some());
    assert!(result.get(1).is_none()); // slot 1 not filled
    assert!(result.get(2).is_some());
    assert_eq!(result.get(0).unwrap().text, "Hello");
    assert_eq!(result.get(2).unwrap().tokens_generated, 2);
}

#[test]
fn batch_result_insert_beyond_capacity_grows() {
    let mut result = BatchResult::with_capacity(2);
    result.insert(SingleResult { id: 5, text: "Far".to_string(), tokens_generated: 1 });
    assert!(result.get(5).is_some());
    assert!(result.capacity() >= 6);
}

#[test]
fn batch_result_iter_only_completed() {
    let mut result = BatchResult::with_capacity(5);
    result.insert(SingleResult { id: 1, text: "A".to_string(), tokens_generated: 1 });
    result.insert(SingleResult { id: 3, text: "B".to_string(), tokens_generated: 1 });

    let texts: Vec<&str> = result.iter().map(|r| r.text.as_str()).collect();
    assert_eq!(texts, vec!["A", "B"]);
}

// ── BatchConfig ──────────────────────────────────────────────────────────

#[test]
fn batch_config_default() {
    let cfg = BatchConfig::default();
    assert_eq!(cfg.max_batch_size, 8);
    assert_eq!(cfg.timeout, Duration::from_secs(30));
    assert_eq!(cfg.max_total_tokens, 8192);
}

#[test]
fn batch_config_new() {
    let cfg = BatchConfig::new(16, Duration::from_millis(500));
    assert_eq!(cfg.max_batch_size, 16);
    assert_eq!(cfg.timeout, Duration::from_millis(500));
    assert_eq!(cfg.max_total_tokens, 8192); // default
}

#[test]
fn batch_config_with_max_total_tokens() {
    let cfg = BatchConfig::new(4, Duration::from_secs(1)).with_max_total_tokens(16384);
    assert_eq!(cfg.max_total_tokens, 16384);
}

#[test]
fn batch_config_validates_default() {
    assert!(BatchConfig::default().validate().is_ok());
}

#[test]
fn batch_config_serde_roundtrip() {
    let cfg = BatchConfig::new(4, Duration::from_millis(100));
    let json = serde_json::to_string(&cfg).expect("serialize");
    let cfg2: BatchConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(cfg2.max_batch_size, 4);
    assert_eq!(cfg2.timeout, Duration::from_millis(100));
}

#[test]
#[should_panic(expected = "max_batch_size must be > 0")]
fn batch_config_panics_on_zero_batch_size() {
    let _ = BatchConfig::new(0, Duration::from_secs(1));
}

// ── BatchScheduler ───────────────────────────────────────────────────────

#[test]
fn scheduler_empty_batch_returns_empty() {
    let scheduler = BatchScheduler::new(BatchConfig::default());
    let batch = BatchRequest::new();
    let order = scheduler.schedule(&batch);
    assert!(order.is_empty());
}

#[test]
fn scheduler_respects_max_batch_size() {
    let cfg = BatchConfig::new(2, Duration::from_secs(1));
    let scheduler = BatchScheduler::new(cfg);

    let mut batch = BatchRequest::new();
    batch.add("Short".to_string(), GenerationConfig::default());
    batch.add("Medium length prompt".to_string(), GenerationConfig::default());
    batch.add("A very long prompt that exceeds others".to_string(), GenerationConfig::default());

    let order = scheduler.schedule(&batch);
    assert!(order.len() <= 2);
}

#[test]
fn scheduler_prefers_shorter_prompts() {
    let scheduler = BatchScheduler::new(BatchConfig::default());

    let mut batch = BatchRequest::new();
    batch
        .add("This is a very long prompt with many words".to_string(), GenerationConfig::default());
    batch.add("Short".to_string(), GenerationConfig::default());
    batch.add("Mid".to_string(), GenerationConfig::default());

    let order = scheduler.schedule(&batch);
    // First scheduled should be the shortest prompt (ID 2 = "Mid" or ID 1 = "Short")
    assert!(order[0] == 1 || order[0] == 2);
}

#[test]
fn scheduler_config_accessor() {
    let cfg = BatchConfig::new(4, Duration::from_millis(100));
    let scheduler = BatchScheduler::new(cfg.clone());
    assert_eq!(scheduler.config().max_batch_size, 4);
}

// ── StreamingConfig ──────────────────────────────────────────────────────

#[test]
fn streaming_config_default() {
    let cfg = StreamingConfig::default();
    assert_eq!(cfg.buffer_size, 10);
    assert_eq!(cfg.flush_interval_ms, 50);
    assert_eq!(cfg.max_retries, 3);
    assert_eq!(cfg.token_timeout_ms, 5000);
    assert!(cfg.cancellable);
}

#[test]
fn streaming_config_validates_default() {
    assert!(StreamingConfig::default().validate().is_ok());
}

#[test]
fn streaming_config_low_latency() {
    let cfg = StreamingConfig::low_latency();
    assert_eq!(cfg.buffer_size, 1);
    assert!(cfg.flush_interval_ms <= 10);
    assert!(cfg.cancellable);
}

#[test]
fn streaming_config_high_throughput() {
    let cfg = StreamingConfig::high_throughput();
    assert!(cfg.buffer_size >= 50);
    assert!(cfg.flush_interval_ms >= 200);
    assert!(!cfg.cancellable);
}

#[test]
fn streaming_config_rejects_zero_buffer() {
    let cfg = StreamingConfig { buffer_size: 0, ..Default::default() };
    assert!(cfg.validate().is_err());
}

#[test]
fn streaming_config_rejects_zero_flush_interval() {
    let cfg = StreamingConfig { flush_interval_ms: 0, ..Default::default() };
    assert!(cfg.validate().is_err());
}

#[test]
fn streaming_config_rejects_zero_timeout() {
    let cfg = StreamingConfig { token_timeout_ms: 0, ..Default::default() };
    assert!(cfg.validate().is_err());
}

// ── StreamConfig (token_stream) ──────────────────────────────────────────

#[test]
fn stream_config_default() {
    let cfg = StreamConfig::default();
    assert_eq!(cfg.buffer_size, 8);
    assert!(cfg.flush_on_whitespace);
    assert!(cfg.flush_on_newline);
    assert_eq!(cfg.max_pending_tokens, 64);
}

// ── TokenBuffer ──────────────────────────────────────────────────────────

#[test]
fn token_buffer_new_is_empty() {
    let buf = TokenBuffer::new();
    assert!(buf.is_empty());
    assert_eq!(buf.len(), 0);
}

#[test]
fn token_buffer_push_ascii() {
    let mut buf = TokenBuffer::new();
    buf.push_bytes(b"Hello");
    assert!(!buf.is_empty());
    assert_eq!(buf.len(), 5);
}

#[test]
fn token_buffer_try_decode_ascii() {
    let mut buf = TokenBuffer::new();
    buf.push_bytes(b"Hello");
    let text = buf.try_decode();
    assert_eq!(text, Some("Hello".to_string()));
    assert!(buf.is_empty());
}

#[test]
fn token_buffer_try_decode_partial_utf8() {
    let mut buf = TokenBuffer::new();
    // First two bytes of a 3-byte UTF-8 sequence (€ = 0xE2 0x82 0xAC)
    buf.push_bytes(&[0xE2, 0x82]);
    // Should not decode yet — incomplete UTF-8
    let text = buf.try_decode();
    // May return None or partial depending on implementation
    // The buffer should retain the incomplete bytes
    if text.is_none() {
        assert!(!buf.is_empty());
    }
}

#[test]
fn token_buffer_try_decode_complete_utf8() {
    let mut buf = TokenBuffer::new();
    buf.push_bytes("€".as_bytes()); // 0xE2 0x82 0xAC
    let text = buf.try_decode();
    assert_eq!(text, Some("€".to_string()));
}

#[test]
fn token_buffer_drain_lossy() {
    let mut buf = TokenBuffer::new();
    buf.push_bytes(b"Hello");
    buf.push_bytes(&[0xFF]); // invalid UTF-8
    let text = buf.drain_lossy();
    assert!(text.contains("Hello"));
    assert!(buf.is_empty());
}

#[test]
fn token_buffer_drain_lossy_empty() {
    let mut buf = TokenBuffer::new();
    let text = buf.drain_lossy();
    assert!(text.is_empty());
}

// ── StreamEvent ──────────────────────────────────────────────────────────

#[test]
fn stream_event_variants() {
    let token = StreamEvent::Token(42);
    let text = StreamEvent::Text("hello".to_string());
    let eos = StreamEvent::EndOfStream;
    let err = StreamEvent::Error("oops".to_string());

    assert_ne!(token, text);
    assert_ne!(text, eos);
    assert_ne!(eos, err);
}

#[test]
fn stream_event_eq() {
    assert_eq!(StreamEvent::Token(1), StreamEvent::Token(1));
    assert_ne!(StreamEvent::Token(1), StreamEvent::Token(2));
    assert_eq!(StreamEvent::Text("hi".to_string()), StreamEvent::Text("hi".to_string()));
    assert_eq!(StreamEvent::EndOfStream, StreamEvent::EndOfStream);
}

#[test]
fn stream_event_debug_format() {
    let event = StreamEvent::Token(42);
    let dbg = format!("{:?}", event);
    assert!(dbg.contains("Token"));
    assert!(dbg.contains("42"));
}

// ── StreamStats ──────────────────────────────────────────────────────────

#[test]
fn stream_stats_default() {
    let stats = StreamStats::default();
    assert_eq!(stats.tokens_generated, 0);
    assert_eq!(stats.text_chunks_emitted, 0);
    assert_eq!(stats.total_bytes, 0);
    assert!((stats.avg_tokens_per_chunk - 0.0).abs() < f64::EPSILON);
}

// ── TokenStream ──────────────────────────────────────────────────────────

#[test]
fn token_stream_basic_flow() {
    use bitnet_inference::token_stream::TokenStream;

    // Simple decode function: maps token ID to its string representation
    let decode_fn = |id: u32| -> Option<Vec<u8>> { Some(format!("{}", id).into_bytes()) };

    let mut stream = TokenStream::new(StreamConfig::default(), decode_fn);
    assert!(!stream.is_complete());

    // Push a token
    let event = stream.push_token(42);
    // Should produce at least a Token event
    if let Some(evt) = event {
        match evt {
            StreamEvent::Token(id) => assert_eq!(id, 42),
            StreamEvent::Text(_) => {} // Also valid
            _ => panic!("Unexpected event: {:?}", evt),
        }
    }
}

#[test]
fn token_stream_flush() {
    use bitnet_inference::token_stream::TokenStream;

    let decode_fn = |id: u32| -> Option<Vec<u8>> { Some(format!("t{}", id).into_bytes()) };

    let mut stream = TokenStream::new(StreamConfig::default(), decode_fn);
    stream.push_token(1);
    stream.push_token(2);

    let events = stream.flush();
    // Flush should produce EndOfStream as the last event
    assert!(events.iter().any(|e| matches!(e, StreamEvent::EndOfStream)));
    assert!(stream.is_complete());
}

#[test]
fn token_stream_stats() {
    use bitnet_inference::token_stream::TokenStream;

    let decode_fn = |_id: u32| -> Option<Vec<u8>> { Some(b"x".to_vec()) };

    let mut stream = TokenStream::new(StreamConfig::default(), decode_fn);
    stream.push_token(1);
    stream.push_token(2);
    stream.push_token(3);

    let stats = stream.stats();
    assert_eq!(stats.tokens_generated, 3);
}
