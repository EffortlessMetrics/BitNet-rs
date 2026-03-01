//! Edge-case tests for bitnet-trace: TraceRecord, TraceComparison,
//! TraceSink, and compare_records.

use bitnet_trace::{TraceComparison, TraceRecord, TraceSink, compare_records};

// ===========================================================================
// TraceRecord
// ===========================================================================

fn sample_record(name: &str, rms: f64) -> TraceRecord {
    TraceRecord {
        name: name.to_string(),
        shape: vec![1, 2560],
        dtype: "F32".to_string(),
        blake3: "abcdef0123456789".to_string(),
        rms,
        num_elements: 2560,
        seq: None,
        layer: None,
        stage: None,
    }
}

#[test]
fn trace_record_serde_roundtrip() {
    let rec = sample_record("layer_0", 0.998);
    let json = serde_json::to_string(&rec).unwrap();
    let back: TraceRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(rec.name, back.name);
    assert_eq!(rec.shape, back.shape);
    assert_eq!(rec.rms, back.rms);
}

#[test]
fn trace_record_optional_fields_omitted() {
    let rec = sample_record("test", 1.0);
    let json = serde_json::to_string(&rec).unwrap();
    assert!(!json.contains("seq"));
    assert!(!json.contains("layer"));
    assert!(!json.contains("stage"));
}

#[test]
fn trace_record_optional_fields_present() {
    let rec = TraceRecord {
        name: "attn_q".into(),
        shape: vec![4, 128],
        dtype: "F32".into(),
        blake3: "hash123".into(),
        rms: 0.5,
        num_elements: 512,
        seq: Some(0),
        layer: Some(3),
        stage: Some("q_proj".into()),
    };
    let json = serde_json::to_string(&rec).unwrap();
    assert!(json.contains("\"seq\":0"));
    assert!(json.contains("\"layer\":3"));
    assert!(json.contains("q_proj"));
}

// ===========================================================================
// compare_records
// ===========================================================================

#[test]
fn compare_identical_records() {
    let a = sample_record("layer", 0.998);
    let b = sample_record("layer", 0.998);
    let cmp = compare_records(&a, &b, 0.001);
    assert!(cmp.is_ok());
    assert!(cmp.shapes_match);
    assert!(cmp.dtypes_match);
    assert!(cmp.hashes_match);
    assert!(cmp.rms_diff < f64::EPSILON);
}

#[test]
fn compare_different_shapes() {
    let a = sample_record("layer", 0.998);
    let mut b = sample_record("layer", 0.998);
    b.shape = vec![2, 1280];
    let cmp = compare_records(&a, &b, 0.001);
    assert!(!cmp.shapes_match);
    assert!(!cmp.is_ok());
}

#[test]
fn compare_different_dtypes() {
    let a = sample_record("layer", 0.998);
    let mut b = sample_record("layer", 0.998);
    b.dtype = "F16".into();
    let cmp = compare_records(&a, &b, 0.001);
    assert!(!cmp.dtypes_match);
    assert!(!cmp.is_ok());
}

#[test]
fn compare_different_hashes() {
    let a = sample_record("layer", 0.998);
    let mut b = sample_record("layer", 0.998);
    b.blake3 = "different_hash".into();
    let cmp = compare_records(&a, &b, 0.001);
    assert!(!cmp.hashes_match);
    // is_ok doesn't require hash match—only shapes/dtypes/rms
    assert!(cmp.is_ok());
}

#[test]
fn compare_rms_within_tolerance() {
    let a = sample_record("layer", 1.000);
    let b = sample_record("layer", 1.0005);
    let cmp = compare_records(&a, &b, 0.001);
    assert!(cmp.rms_within_tolerance);
    assert!(cmp.is_ok());
}

#[test]
fn compare_rms_outside_tolerance() {
    let a = sample_record("layer", 1.0);
    let b = sample_record("layer", 1.1);
    let cmp = compare_records(&a, &b, 0.01);
    assert!(!cmp.rms_within_tolerance);
    assert!(!cmp.is_ok());
}

// ===========================================================================
// TraceComparison
// ===========================================================================

#[test]
fn trace_comparison_is_ok_all_match() {
    let cmp = TraceComparison {
        shapes_match: true,
        dtypes_match: true,
        hashes_match: true,
        rms_diff: 0.0,
        rms_within_tolerance: true,
    };
    assert!(cmp.is_ok());
}

#[test]
fn trace_comparison_is_ok_requires_shapes() {
    let cmp = TraceComparison {
        shapes_match: false,
        dtypes_match: true,
        hashes_match: true,
        rms_diff: 0.0,
        rms_within_tolerance: true,
    };
    assert!(!cmp.is_ok());
}

#[test]
fn trace_comparison_clone_eq() {
    let cmp = TraceComparison {
        shapes_match: true,
        dtypes_match: true,
        hashes_match: false,
        rms_diff: 0.05,
        rms_within_tolerance: true,
    };
    let cloned = cmp.clone();
    assert_eq!(cmp, cloned);
}

// ===========================================================================
// TraceSink
// ===========================================================================

#[test]
fn trace_sink_new_empty() {
    let sink = TraceSink::new();
    assert!(sink.is_empty());
    assert_eq!(sink.len(), 0);
}

#[test]
fn trace_sink_append_and_len() {
    let mut sink = TraceSink::new();
    sink.append(sample_record("a", 1.0));
    sink.append(sample_record("b", 2.0));
    assert_eq!(sink.len(), 2);
    assert!(!sink.is_empty());
}

#[test]
fn trace_sink_iter() {
    let mut sink = TraceSink::new();
    sink.append(sample_record("first", 1.0));
    sink.append(sample_record("second", 2.0));
    let names: Vec<_> = sink.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(names, &["first", "second"]);
}

#[test]
fn trace_sink_filter_by_name() {
    let mut sink = TraceSink::new();
    sink.append(sample_record("blk0_attn", 1.0));
    sink.append(sample_record("blk0_ffn", 2.0));
    sink.append(sample_record("blk1_attn", 3.0));
    let attn = sink.filter_by_name("attn");
    assert_eq!(attn.len(), 2);
}

#[test]
fn trace_sink_filter_no_match() {
    let mut sink = TraceSink::new();
    sink.append(sample_record("layer_0", 1.0));
    let result = sink.filter_by_name("nonexistent");
    assert!(result.is_empty());
}

#[test]
fn trace_sink_clear() {
    let mut sink = TraceSink::new();
    sink.append(sample_record("a", 1.0));
    sink.append(sample_record("b", 2.0));
    sink.clear();
    assert!(sink.is_empty());
}

#[test]
fn trace_sink_default() {
    let sink = TraceSink::default();
    assert!(sink.is_empty());
}
