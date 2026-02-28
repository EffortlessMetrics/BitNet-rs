//! Integration tests for WebSocket streaming and SSE fallback modules.

use bitnet_server::sse::{SseConfig, build_sse_token, format_sse_event};
use bitnet_server::websocket::{
    WsConfig, WsConnectionManager, WsMessage, should_apply_backpressure,
};

// ── WsMessage serialization round-trips ─────────────────────────────

#[test]
fn round_trip_request_message() {
    let msg = WsMessage::Request {
        prompt: "Explain gravity".into(),
        max_tokens: Some(64),
        temperature: Some(0.7),
        top_k: Some(50),
        top_p: Some(0.9),
        stream: Some(true),
    };
    let json = serde_json::to_string(&msg).unwrap();
    let decoded: WsMessage = serde_json::from_str(&json).unwrap();
    if let WsMessage::Request { prompt, max_tokens, .. } = decoded {
        assert_eq!(prompt, "Explain gravity");
        assert_eq!(max_tokens, Some(64));
    } else {
        panic!("expected Request variant");
    }
}

#[test]
fn round_trip_token_message() {
    let msg = WsMessage::Token {
        text: "hello".into(),
        token_id: 42,
        logprob: Some(-1.5),
        finish_reason: None,
    };
    let json = serde_json::to_string(&msg).unwrap();
    let decoded: WsMessage = serde_json::from_str(&json).unwrap();
    if let WsMessage::Token { text, token_id, logprob, finish_reason } = decoded {
        assert_eq!(text, "hello");
        assert_eq!(token_id, 42);
        assert!((logprob.unwrap() - (-1.5)).abs() < f32::EPSILON);
        assert!(finish_reason.is_none());
    } else {
        panic!("expected Token variant");
    }
}

#[test]
fn round_trip_metadata_message() {
    let msg = WsMessage::Metadata {
        model: "bitnet-2B".into(),
        backend: "cuda".into(),
        tokens_per_second: 12.5,
        total_tokens: 100,
        prompt_tokens: 10,
    };
    let json = serde_json::to_string(&msg).unwrap();
    let decoded: WsMessage = serde_json::from_str(&json).unwrap();
    if let WsMessage::Metadata { model, backend, tokens_per_second, total_tokens, prompt_tokens } =
        decoded
    {
        assert_eq!(model, "bitnet-2B");
        assert_eq!(backend, "cuda");
        assert!((tokens_per_second - 12.5).abs() < f64::EPSILON);
        assert_eq!(total_tokens, 100);
        assert_eq!(prompt_tokens, 10);
    } else {
        panic!("expected Metadata variant");
    }
}

#[test]
fn round_trip_error_message() {
    let msg = WsMessage::Error { message: "out of memory".into(), code: 503 };
    let json = serde_json::to_string(&msg).unwrap();
    let decoded: WsMessage = serde_json::from_str(&json).unwrap();
    if let WsMessage::Error { message, code } = decoded {
        assert_eq!(message, "out of memory");
        assert_eq!(code, 503);
    } else {
        panic!("expected Error variant");
    }
}

#[test]
fn round_trip_ping_pong() {
    for msg in [WsMessage::Ping, WsMessage::Pong] {
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: WsMessage = serde_json::from_str(&json).unwrap();
        // Verify the tag survives the round-trip.
        let re_json = serde_json::to_string(&decoded).unwrap();
        assert_eq!(json, re_json);
    }
}

// ── WsConfig defaults ───────────────────────────────────────────────

#[test]
fn ws_config_default_values() {
    let cfg = WsConfig::default();
    assert_eq!(cfg.max_connections, 100);
    assert_eq!(cfg.idle_timeout_secs, 300);
    assert_eq!(cfg.max_message_size, 64 * 1024);
    assert_eq!(cfg.heartbeat_interval_secs, 30);
    assert_eq!(cfg.backpressure_limit, 1024);
}

// ── Connection manager limits ───────────────────────────────────────

#[test]
fn connection_manager_respects_capacity() {
    let mgr = WsConnectionManager::new(WsConfig { max_connections: 3, ..WsConfig::default() });
    assert!(mgr.can_accept());
    assert!(mgr.register());
    assert!(mgr.register());
    assert!(mgr.register());
    assert!(!mgr.can_accept());
    assert!(!mgr.register());

    mgr.unregister();
    assert!(mgr.can_accept());
    assert!(mgr.register());
}

#[test]
fn connection_manager_accessors() {
    let mgr = WsConnectionManager::new(WsConfig {
        max_connections: 10,
        idle_timeout_secs: 120,
        max_message_size: 8192,
        heartbeat_interval_secs: 15,
        backpressure_limit: 512,
    });
    assert_eq!(mgr.max_connections(), 10);
    assert_eq!(mgr.idle_timeout_secs(), 120);
    assert_eq!(mgr.max_message_size(), 8192);
    assert_eq!(mgr.heartbeat_interval_secs(), 15);
    assert_eq!(mgr.backpressure_limit(), 512);
}

// ── Backpressure ────────────────────────────────────────────────────

#[test]
fn backpressure_boundary() {
    assert!(!should_apply_backpressure(0, 1));
    assert!(should_apply_backpressure(1, 1));
    assert!(!should_apply_backpressure(1023, 1024));
    assert!(should_apply_backpressure(1024, 1024));
    assert!(should_apply_backpressure(2000, 1024));
}

// ── SSE event formatting ────────────────────────────────────────────

#[test]
fn sse_format_token_event() {
    let msg = WsMessage::Token {
        text: "world".into(),
        token_id: 7,
        logprob: None,
        finish_reason: Some("stop".into()),
    };
    let sse = format_sse_event(&msg);
    assert!(sse.starts_with("event: token\n"));
    assert!(sse.contains("\"token_id\":7"));
    assert!(sse.contains("\"finish_reason\":\"stop\""));
    assert!(sse.ends_with("\n\n"));
}

#[test]
fn sse_format_metadata_event() {
    let msg = WsMessage::Metadata {
        model: "m".into(),
        backend: "cpu".into(),
        tokens_per_second: 1.0,
        total_tokens: 1,
        prompt_tokens: 0,
    };
    let sse = format_sse_event(&msg);
    assert!(sse.starts_with("event: metadata\n"));
    assert!(sse.ends_with("\n\n"));
}

#[test]
fn sse_format_error_event() {
    let msg = WsMessage::Error { message: "bad".into(), code: 400 };
    let sse = format_sse_event(&msg);
    assert!(sse.starts_with("event: error\n"));
    assert!(sse.contains("\"code\":400"));
}

#[test]
fn sse_format_ping_pong_events() {
    assert_eq!(format_sse_event(&WsMessage::Ping), "event: ping\ndata: \n\n");
    assert_eq!(format_sse_event(&WsMessage::Pong), "event: pong\ndata: \n\n");
}

// ── SseConfig defaults ──────────────────────────────────────────────

#[test]
fn sse_config_defaults() {
    let cfg = SseConfig::default();
    assert_eq!(cfg.retry_ms, 3000);
    assert_eq!(cfg.keep_alive_secs, 15);
}

// ── build_sse_token ─────────────────────────────────────────────────

#[test]
fn build_sse_token_with_id_and_retry() {
    let msg =
        WsMessage::Token { text: "hi".into(), token_id: 1, logprob: None, finish_reason: None };
    let tok = build_sse_token(&msg, Some("ev-1".into()), Some(2000));
    assert_eq!(tok.event, "token");
    assert_eq!(tok.id.as_deref(), Some("ev-1"));
    assert_eq!(tok.retry, Some(2000));
    assert!(tok.data.contains("\"text\":\"hi\""));
}

#[test]
fn build_sse_token_no_optional_fields() {
    let msg = WsMessage::Ping;
    let tok = build_sse_token(&msg, None, None);
    assert_eq!(tok.event, "ping");
    assert!(tok.id.is_none());
    assert!(tok.retry.is_none());
}

// ── Message type variant coverage ───────────────────────────────────

#[test]
fn all_message_variants_serialize() {
    let variants: Vec<WsMessage> = vec![
        WsMessage::Request {
            prompt: "x".into(),
            max_tokens: None,
            temperature: None,
            top_k: None,
            top_p: None,
            stream: None,
        },
        WsMessage::Token { text: "t".into(), token_id: 0, logprob: None, finish_reason: None },
        WsMessage::Metadata {
            model: "m".into(),
            backend: "b".into(),
            tokens_per_second: 0.0,
            total_tokens: 0,
            prompt_tokens: 0,
        },
        WsMessage::Error { message: "e".into(), code: 0 },
        WsMessage::Ping,
        WsMessage::Pong,
    ];
    for v in &variants {
        let json = serde_json::to_string(v).unwrap();
        let _: WsMessage = serde_json::from_str(&json).unwrap();
    }
}

// ── Property tests for WsMessage serialization ─────────────────────

mod prop {
    use super::*;
    use proptest::prelude::*;

    fn arb_ws_message() -> impl Strategy<Value = WsMessage> {
        prop_oneof![
            (
                ".*",
                proptest::option::of(0usize..4096),
                proptest::option::of(0.0f32..2.0),
                proptest::option::of(0usize..200),
                proptest::option::of(0.0f32..1.0),
                proptest::option::of(proptest::bool::ANY),
            )
                .prop_map(|(prompt, max_tokens, temperature, top_k, top_p, stream)| {
                    WsMessage::Request { prompt, max_tokens, temperature, top_k, top_p, stream }
                }),
            (".*", 0u32..100_000, proptest::option::of(-10.0f32..0.0f32)).prop_map(
                |(text, token_id, logprob)| {
                    WsMessage::Token { text, token_id, logprob, finish_reason: None }
                }
            ),
            Just(WsMessage::Ping),
            Just(WsMessage::Pong),
        ]
    }

    proptest! {
        #[test]
        fn ws_message_round_trip(msg in arb_ws_message()) {
            let json = serde_json::to_string(&msg).unwrap();
            let decoded: WsMessage = serde_json::from_str(&json).unwrap();
            let re_json = serde_json::to_string(&decoded).unwrap();
            prop_assert_eq!(json, re_json);
        }
    }
}
