use bitnet_generation_events_core::{GenerationStats, StreamEvent, TokenEvent};
use bitnet_generation_stop_core::StopReason;

#[test]
fn token_event_round_trips_through_json() {
    let ev = TokenEvent { id: 42, text: "hello".to_string() };
    let json = serde_json::to_string(&ev).unwrap();
    let back: TokenEvent = serde_json::from_str(&json).unwrap();

    assert_eq!(back.id, 42);
    assert_eq!(back.text, "hello");
}

#[test]
fn done_event_contains_stats_and_reason() {
    let ev = StreamEvent::Done {
        reason: StopReason::MaxTokens,
        stats: GenerationStats { tokens_generated: 4, tokens_per_second: 2.0 },
    };

    match ev {
        StreamEvent::Done { reason, stats } => {
            assert_eq!(reason, StopReason::MaxTokens);
            assert_eq!(stats.tokens_generated, 4);
            assert!((stats.tokens_per_second - 2.0).abs() < f64::EPSILON);
        }
        StreamEvent::Token(_) => panic!("expected done event"),
    }
}
