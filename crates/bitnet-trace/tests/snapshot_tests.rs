//! Snapshot tests for bitnet-trace crate.
//!
//! These tests pin the JSON serialization format of `TraceRecord`,
//! catching any accidental breaking changes to the trace file format
//! that cross-validation tools depend on.

use bitnet_trace::TraceRecord;
use insta::assert_json_snapshot;

fn make_record() -> TraceRecord {
    TraceRecord {
        name: "layer0/q_proj".to_string(),
        shape: vec![1, 32, 128],
        dtype: "F32".to_string(),
        blake3: "a".repeat(64),
        rms: 0.123_456_789,
        num_elements: 4096,
        seq: None,
        layer: None,
        stage: None,
    }
}

#[test]
fn trace_record_minimal() {
    let record = make_record();
    assert_json_snapshot!("trace_record_minimal", record);
}

#[test]
fn trace_record_with_optional_fields() {
    let record = TraceRecord {
        name: "embeddings".to_string(),
        shape: vec![1, 512],
        dtype: "F32".to_string(),
        blake3: "b".repeat(64),
        rms: 1.0,
        num_elements: 512,
        seq: Some(0),
        layer: Some(-1),
        stage: Some("embeddings".to_string()),
    };
    assert_json_snapshot!("trace_record_with_optional_fields", record);
}

#[test]
fn trace_record_decode_step() {
    let record = TraceRecord {
        name: "layer3/ffn_out".to_string(),
        shape: vec![1, 1, 2048],
        dtype: "F32".to_string(),
        blake3: "c".repeat(64),
        rms: 0.5,
        num_elements: 2048,
        seq: Some(5),
        layer: Some(3),
        stage: Some("ffn_out".to_string()),
    };
    assert_json_snapshot!("trace_record_decode_step", record);
}

#[test]
fn trace_record_logits() {
    let record = TraceRecord {
        name: "logits".to_string(),
        shape: vec![1, 1, 32000],
        dtype: "F32".to_string(),
        blake3: "d".repeat(64),
        rms: 2.345,
        num_elements: 32000,
        seq: Some(1),
        layer: Some(-1),
        stage: Some("logits".to_string()),
    };
    assert_json_snapshot!("trace_record_logits", record);
}
