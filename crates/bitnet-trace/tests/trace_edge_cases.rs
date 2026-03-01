//! Edge-case tests for bitnet-trace: TraceRecord, TraceComparison, TraceSink,
//! compare_records, and serialization/deserialization edge cases.

use bitnet_trace::{TraceComparison, TraceRecord, TraceSink, compare_records};

// ---------------------------------------------------------------------------
// Helper to build a minimal TraceRecord
// ---------------------------------------------------------------------------

fn record(name: &str, shape: Vec<usize>, rms: f64) -> TraceRecord {
    TraceRecord {
        name: name.to_string(),
        shape,
        dtype: "F32".to_string(),
        blake3: "deadbeef".to_string(),
        rms,
        num_elements: 1,
        seq: None,
        layer: None,
        stage: None,
    }
}

// ---------------------------------------------------------------------------
// TraceRecord — serialization
// ---------------------------------------------------------------------------

#[test]
fn trace_record_json_roundtrip() {
    let r = record("layer0", vec![2, 3], 1.5);
    let json = serde_json::to_string(&r).unwrap();
    let deser: TraceRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(r.name, deser.name);
    assert_eq!(r.shape, deser.shape);
    assert_eq!(r.rms, deser.rms);
}

#[test]
fn trace_record_optional_fields_omitted_when_none() {
    let r = record("t", vec![1], 0.0);
    let json = serde_json::to_string(&r).unwrap();
    assert!(!json.contains("\"seq\""));
    assert!(!json.contains("\"layer\""));
    assert!(!json.contains("\"stage\""));
}

#[test]
fn trace_record_optional_fields_present_when_some() {
    let mut r = record("t", vec![1], 0.0);
    r.seq = Some(5);
    r.layer = Some(-1);
    r.stage = Some("q_proj".into());
    let json = serde_json::to_string(&r).unwrap();
    assert!(json.contains("\"seq\":5"));
    assert!(json.contains("\"layer\":-1"));
    assert!(json.contains("\"q_proj\""));
}

#[test]
fn trace_record_debug() {
    let r = record("x", vec![4], 2.0);
    let dbg = format!("{r:?}");
    assert!(dbg.contains("TraceRecord"));
}

#[test]
fn trace_record_clone() {
    let r = record("a", vec![10, 20], 3.14);
    let c = r.clone();
    assert_eq!(r.name, c.name);
    assert_eq!(r.shape, c.shape);
    assert_eq!(r.rms, c.rms);
}

#[test]
fn trace_record_empty_shape() {
    let r = record("scalar", vec![], 0.0);
    let json = serde_json::to_string(&r).unwrap();
    let deser: TraceRecord = serde_json::from_str(&json).unwrap();
    assert!(deser.shape.is_empty());
}

#[test]
fn trace_record_large_shape() {
    let r = record("big", vec![128, 256, 512, 1024], 42.0);
    assert_eq!(r.shape.len(), 4);
}

// ---------------------------------------------------------------------------
// TraceComparison — compare_records
// ---------------------------------------------------------------------------

#[test]
fn compare_identical_records() {
    let a = record("x", vec![2, 3], 1.0);
    let b = record("x", vec![2, 3], 1.0);
    let cmp = compare_records(&a, &b, 0.01);
    assert!(cmp.is_ok());
    assert!(cmp.shapes_match);
    assert!(cmp.dtypes_match);
    assert!(cmp.hashes_match);
    assert!(cmp.rms_diff < f64::EPSILON);
    assert!(cmp.rms_within_tolerance);
}

#[test]
fn compare_different_shapes() {
    let a = record("x", vec![2, 3], 1.0);
    let mut b = record("x", vec![3, 2], 1.0);
    b.blake3 = a.blake3.clone();
    let cmp = compare_records(&a, &b, 0.01);
    assert!(!cmp.shapes_match);
    assert!(!cmp.is_ok());
}

#[test]
fn compare_different_dtypes() {
    let a = record("x", vec![2], 1.0);
    let mut b = record("x", vec![2], 1.0);
    b.dtype = "F16".to_string();
    let cmp = compare_records(&a, &b, 0.01);
    assert!(!cmp.dtypes_match);
    assert!(!cmp.is_ok());
}

#[test]
fn compare_different_hashes() {
    let a = record("x", vec![2], 1.0);
    let mut b = record("x", vec![2], 1.0);
    b.blake3 = "cafebabe".to_string();
    let cmp = compare_records(&a, &b, 0.01);
    assert!(!cmp.hashes_match);
    // is_ok() doesn't depend on hash match, only shape/dtype/rms
    assert!(cmp.is_ok());
}

#[test]
fn compare_rms_within_tolerance() {
    let a = record("x", vec![2], 1.0);
    let b = record("x", vec![2], 1.005);
    let cmp = compare_records(&a, &b, 0.01);
    assert!(cmp.rms_within_tolerance);
    assert!(cmp.is_ok());
}

#[test]
fn compare_rms_outside_tolerance() {
    let a = record("x", vec![2], 1.0);
    let b = record("x", vec![2], 2.0);
    let cmp = compare_records(&a, &b, 0.01);
    assert!(!cmp.rms_within_tolerance);
    assert!(!cmp.is_ok());
}

#[test]
fn compare_zero_tolerance() {
    let a = record("x", vec![2], 1.0);
    let b = record("x", vec![2], 1.0);
    let cmp = compare_records(&a, &b, 0.0);
    assert!(cmp.rms_within_tolerance);
}

#[test]
fn compare_rms_exactly_at_tolerance() {
    let a = record("x", vec![2], 1.0);
    let b = record("x", vec![2], 1.5);
    let cmp = compare_records(&a, &b, 0.5);
    // rms_diff == 0.5, tolerance == 0.5 → within (<=)
    assert!(cmp.rms_within_tolerance);
}

#[test]
fn comparison_debug_and_clone() {
    let cmp = TraceComparison {
        shapes_match: true,
        dtypes_match: true,
        hashes_match: false,
        rms_diff: 0.001,
        rms_within_tolerance: true,
    };
    let c = cmp.clone();
    assert_eq!(cmp, c);
    let dbg = format!("{cmp:?}");
    assert!(dbg.contains("TraceComparison"));
}

#[test]
fn comparison_partial_eq() {
    let a = TraceComparison {
        shapes_match: true,
        dtypes_match: true,
        hashes_match: true,
        rms_diff: 0.0,
        rms_within_tolerance: true,
    };
    let b = a.clone();
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------------
// TraceSink
// ---------------------------------------------------------------------------

#[test]
fn sink_new_is_empty() {
    let sink = TraceSink::new();
    assert!(sink.is_empty());
    assert_eq!(sink.len(), 0);
}

#[test]
fn sink_default_is_empty() {
    let sink = TraceSink::default();
    assert!(sink.is_empty());
}

#[test]
fn sink_append_increments_len() {
    let mut sink = TraceSink::new();
    sink.append(record("a", vec![1], 1.0));
    assert_eq!(sink.len(), 1);
    assert!(!sink.is_empty());
    sink.append(record("b", vec![2], 2.0));
    assert_eq!(sink.len(), 2);
}

#[test]
fn sink_iter_order() {
    let mut sink = TraceSink::new();
    sink.append(record("first", vec![1], 1.0));
    sink.append(record("second", vec![2], 2.0));
    sink.append(record("third", vec![3], 3.0));
    let names: Vec<&str> = sink.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(names, vec!["first", "second", "third"]);
}

#[test]
fn sink_filter_by_name() {
    let mut sink = TraceSink::new();
    sink.append(record("blk0/attn_q", vec![1], 1.0));
    sink.append(record("blk0/attn_k", vec![1], 1.0));
    sink.append(record("blk1/attn_q", vec![1], 1.0));
    sink.append(record("embeddings", vec![1], 1.0));

    let attn = sink.filter_by_name("attn");
    assert_eq!(attn.len(), 3);

    let blk0 = sink.filter_by_name("blk0");
    assert_eq!(blk0.len(), 2);

    let emb = sink.filter_by_name("embeddings");
    assert_eq!(emb.len(), 1);

    let none = sink.filter_by_name("nonexistent");
    assert!(none.is_empty());
}

#[test]
fn sink_clear() {
    let mut sink = TraceSink::new();
    sink.append(record("a", vec![1], 1.0));
    sink.append(record("b", vec![2], 2.0));
    assert_eq!(sink.len(), 2);
    sink.clear();
    assert!(sink.is_empty());
    assert_eq!(sink.len(), 0);
}

#[test]
fn sink_debug() {
    let sink = TraceSink::new();
    let dbg = format!("{sink:?}");
    assert!(dbg.contains("TraceSink"));
}

#[test]
fn sink_filter_empty_substr() {
    let mut sink = TraceSink::new();
    sink.append(record("a", vec![1], 1.0));
    sink.append(record("b", vec![2], 2.0));
    // Empty string matches everything
    let all = sink.filter_by_name("");
    assert_eq!(all.len(), 2);
}

#[test]
fn sink_many_records() {
    let mut sink = TraceSink::new();
    for i in 0..100 {
        sink.append(record(&format!("layer_{i}"), vec![i], i as f64));
    }
    assert_eq!(sink.len(), 100);
    let layer_5 = sink.filter_by_name("layer_5");
    // matches "layer_5", "layer_50", "layer_51", ... "layer_59"
    assert_eq!(layer_5.len(), 11);
}
