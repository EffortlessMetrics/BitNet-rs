//! Comprehensive tests for `bitnet-trace`:
//!
//! * [`TraceRecord`] construction and serialization
//! * [`TraceSink`] collection semantics (append / len / iter / filter / clear)
//! * [`compare_records`] comparison logic (exact match, tolerance, shape/dtype mismatch)
//! * Trace export / import via JSON files
//! * Property: random `TraceRecord`s survive a JSON round-trip

use bitnet_trace::{TraceRecord, TraceSink, compare_records};
use std::fs;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_record(name: &str) -> TraceRecord {
    TraceRecord {
        name: name.to_string(),
        shape: vec![2, 4],
        dtype: "F32".to_string(),
        blake3: "a".repeat(64),
        rms: 1.0,
        num_elements: 8,
        seq: None,
        layer: None,
        stage: None,
    }
}

fn make_record_full(
    name: &str,
    shape: Vec<usize>,
    rms: f64,
    seq: usize,
    layer: isize,
) -> TraceRecord {
    let num_elements = shape.iter().product();
    TraceRecord {
        name: name.to_string(),
        shape,
        dtype: "F32".to_string(),
        blake3: "b".repeat(64),
        rms,
        num_elements,
        seq: Some(seq),
        layer: Some(layer),
        stage: Some("q_proj".to_string()),
    }
}

// ---------------------------------------------------------------------------
// 1. TraceRecord construction and serialization
// ---------------------------------------------------------------------------

#[test]
fn test_record_construction_minimal_fields() {
    let r = make_record("blk0/attn_norm");
    assert_eq!(r.name, "blk0/attn_norm");
    assert_eq!(r.shape, vec![2, 4]);
    assert_eq!(r.dtype, "F32");
    assert_eq!(r.num_elements, 8);
    assert_eq!(r.rms, 1.0);
}

#[test]
fn test_record_num_elements_matches_shape_product() {
    let shapes: &[&[usize]] = &[&[1], &[4, 8], &[2, 3, 4], &[2, 2, 2, 2]];
    for &shape in shapes {
        let expected: usize = shape.iter().product();
        let r = TraceRecord {
            name: "test".into(),
            shape: shape.to_vec(),
            dtype: "F32".into(),
            blake3: "c".repeat(64),
            rms: 0.5,
            num_elements: expected,
            seq: None,
            layer: None,
            stage: None,
        };
        assert_eq!(r.num_elements, expected);
    }
}

#[test]
fn test_record_json_roundtrip_required_fields() {
    let original = make_record("layer/output");
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: TraceRecord = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(original.name, restored.name);
    assert_eq!(original.shape, restored.shape);
    assert_eq!(original.dtype, restored.dtype);
    assert_eq!(original.blake3, restored.blake3);
    assert_eq!(original.num_elements, restored.num_elements);
    assert!((original.rms - restored.rms).abs() < f64::EPSILON);
}

#[test]
fn test_record_json_roundtrip_optional_fields_some() {
    let original = make_record_full("embed", vec![1, 512], 0.8, 3, 2);
    let json = serde_json::to_string(&original).unwrap();
    let restored: TraceRecord = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.seq, Some(3));
    assert_eq!(restored.layer, Some(2));
    assert_eq!(restored.stage.as_deref(), Some("q_proj"));
}

#[test]
fn test_record_optional_fields_omitted_when_none() {
    let r = make_record("simple");
    let json = serde_json::to_string(&r).unwrap();
    assert!(!json.contains("\"seq\""));
    assert!(!json.contains("\"layer\""));
    assert!(!json.contains("\"stage\""));
}

#[test]
fn test_record_optional_fields_present_when_some() {
    let r = make_record_full("logits", vec![1, 32000], 2.5, 1, -1);
    let json = serde_json::to_string(&r).unwrap();
    assert!(json.contains("\"seq\""));
    assert!(json.contains("\"layer\""));
    assert!(json.contains("\"stage\""));
}

#[test]
fn test_record_blake3_field_is_valid_hex_length() {
    let r = make_record("x");
    assert_eq!(r.blake3.len(), 64);
    assert!(r.blake3.chars().all(|c| c.is_ascii_hexdigit()));
}

// ---------------------------------------------------------------------------
// 2. TraceSink collection semantics
// ---------------------------------------------------------------------------

#[test]
fn test_sink_new_is_empty() {
    let sink = TraceSink::new();
    assert!(sink.is_empty());
    assert_eq!(sink.len(), 0);
}

#[test]
fn test_sink_append_increments_len() {
    let mut sink = TraceSink::new();
    sink.append(make_record("a"));
    assert_eq!(sink.len(), 1);
    assert!(!sink.is_empty());
    sink.append(make_record("b"));
    assert_eq!(sink.len(), 2);
}

#[test]
fn test_sink_append_multiple_records() {
    let mut sink = TraceSink::new();
    for i in 0..5 {
        sink.append(make_record(&format!("layer{i}")));
    }
    assert_eq!(sink.len(), 5);
}

#[test]
fn test_sink_records_all_activations_it_receives() {
    // Core feature: every record appended must be retrievable.
    let mut sink = TraceSink::new();
    let names = ["blk0/q", "blk0/k", "blk0/v", "blk1/q", "logits"];
    for name in names {
        sink.append(make_record(name));
    }
    let recorded: Vec<&str> = sink.iter().map(|r| r.name.as_str()).collect();
    for name in names {
        assert!(recorded.contains(&name), "sink missing record for {name}");
    }
}

#[test]
fn test_sink_iter_preserves_insertion_order() {
    let mut sink = TraceSink::new();
    let names = ["first", "second", "third"];
    for name in names {
        sink.append(make_record(name));
    }
    let order: Vec<&str> = sink.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(order, names);
}

#[test]
fn test_sink_iter_yields_correct_record_data() {
    let mut sink = TraceSink::new();
    let r = make_record_full("blk3/ffn", vec![1, 128], 0.75, 0, 3);
    sink.append(r.clone());
    let retrieved = sink.iter().next().unwrap();
    assert_eq!(retrieved.name, "blk3/ffn");
    assert_eq!(retrieved.shape, vec![1, 128]);
    assert_eq!(retrieved.layer, Some(3));
}

#[test]
fn test_sink_filter_by_name_prefix() {
    let mut sink = TraceSink::new();
    sink.append(make_record("blk0/attn_out"));
    sink.append(make_record("blk1/attn_out"));
    sink.append(make_record("blk0/ffn_out"));
    sink.append(make_record("logits"));

    let blk0 = sink.filter_by_name("blk0");
    assert_eq!(blk0.len(), 2, "expected 2 blk0 records");
    assert!(blk0.iter().all(|r| r.name.contains("blk0")));
}

#[test]
fn test_sink_filter_by_name_exact_substring() {
    let mut sink = TraceSink::new();
    sink.append(make_record("layer/attn_norm"));
    sink.append(make_record("layer/ffn_norm"));
    sink.append(make_record("layer/output"));

    let norm = sink.filter_by_name("norm");
    assert_eq!(norm.len(), 2);
}

#[test]
fn test_sink_filter_by_name_no_match_returns_empty() {
    let mut sink = TraceSink::new();
    sink.append(make_record("alpha"));
    sink.append(make_record("beta"));

    let result = sink.filter_by_name("gamma");
    assert!(result.is_empty());
}

#[test]
fn test_sink_clear_resets_to_empty() {
    let mut sink = TraceSink::new();
    sink.append(make_record("a"));
    sink.append(make_record("b"));
    assert_eq!(sink.len(), 2);
    sink.clear();
    assert!(sink.is_empty());
    assert_eq!(sink.len(), 0);
}

#[test]
fn test_sink_clear_then_reuse() {
    let mut sink = TraceSink::new();
    sink.append(make_record("old"));
    sink.clear();
    sink.append(make_record("new"));
    assert_eq!(sink.len(), 1);
    assert_eq!(sink.iter().next().unwrap().name, "new");
}

// ---------------------------------------------------------------------------
// 3. compare_records — trace comparison logic
// ---------------------------------------------------------------------------

#[test]
fn test_compare_identical_records_is_ok() {
    let r = make_record("x");
    let cmp = compare_records(&r, &r, 1e-6);
    assert!(cmp.shapes_match);
    assert!(cmp.dtypes_match);
    assert!(cmp.hashes_match);
    assert!(cmp.rms_within_tolerance);
    assert!(cmp.is_ok());
}

#[test]
fn test_compare_shape_mismatch_not_ok() {
    let a = TraceRecord { shape: vec![1, 4], num_elements: 4, ..make_record("a") };
    let b = TraceRecord { shape: vec![2, 2], num_elements: 4, ..make_record("b") };
    let cmp = compare_records(&a, &b, 1.0);
    assert!(!cmp.shapes_match);
    assert!(!cmp.is_ok());
}

#[test]
fn test_compare_dtype_mismatch_not_ok() {
    let a = make_record("a");
    let b = TraceRecord { dtype: "F16".into(), ..make_record("b") };
    let cmp = compare_records(&a, &b, 1.0);
    assert!(!cmp.dtypes_match);
    assert!(!cmp.is_ok());
}

#[test]
fn test_compare_rms_within_tolerance_passes() {
    let a = TraceRecord { rms: 1.000, ..make_record("a") };
    let b = TraceRecord { rms: 1.001, ..make_record("b") };
    let cmp = compare_records(&a, &b, 0.01);
    assert!(cmp.rms_within_tolerance);
    assert!(cmp.is_ok());
}

#[test]
fn test_compare_rms_exceeds_tolerance_fails() {
    let a = TraceRecord { rms: 1.0, ..make_record("a") };
    let b = TraceRecord { rms: 1.5, ..make_record("b") };
    let cmp = compare_records(&a, &b, 0.1);
    assert!(!cmp.rms_within_tolerance);
    assert!(!cmp.is_ok());
    assert!((cmp.rms_diff - 0.5).abs() < 1e-10);
}

#[test]
fn test_compare_hash_mismatch_does_not_block_is_ok() {
    // is_ok() checks shapes, dtypes, rms — NOT hash. Hash is informational.
    let a = TraceRecord { blake3: "a".repeat(64), rms: 0.5, ..make_record("a") };
    let b = TraceRecord { blake3: "b".repeat(64), rms: 0.5, ..make_record("b") };
    let cmp = compare_records(&a, &b, 0.01);
    assert!(!cmp.hashes_match);
    assert!(cmp.is_ok(), "is_ok should pass even with hash mismatch");
}

#[test]
fn test_compare_zero_tolerance_requires_exact_rms() {
    let a = TraceRecord { rms: 1.0, ..make_record("a") };
    let b = TraceRecord { rms: 1.0 + f64::EPSILON, ..make_record("b") };
    // With zero tolerance, even epsilon-level difference fails.
    let cmp = compare_records(&a, &b, 0.0);
    assert!(!cmp.rms_within_tolerance);
}

// ---------------------------------------------------------------------------
// 4. Trace export / import (file-based JSON)
// ---------------------------------------------------------------------------

#[test]
fn test_export_trace_record_to_file() {
    let dir = TempDir::new().unwrap();
    let record = make_record_full("blk5/out", vec![1, 256], 0.62, 2, 5);
    let path = dir.path().join("blk5_out.trace");
    let json = serde_json::to_string_pretty(&record).unwrap();
    fs::write(&path, &json).unwrap();
    assert!(path.exists());
    assert!(!json.is_empty());
}

#[test]
fn test_import_trace_record_from_file() {
    let dir = TempDir::new().unwrap();
    let original = make_record("embeddings");
    let path = dir.path().join("embeddings.trace");
    fs::write(&path, serde_json::to_string_pretty(&original).unwrap()).unwrap();

    let contents = fs::read_to_string(&path).unwrap();
    let restored: TraceRecord = serde_json::from_str(&contents).unwrap();
    assert_eq!(restored.name, "embeddings");
    assert_eq!(restored.shape, vec![2, 4]);
}

#[test]
fn test_export_import_roundtrip_preserves_all_fields() {
    let dir = TempDir::new().unwrap();
    let original = make_record_full("blk7/k_proj", vec![4, 4, 8], 1.23, 7, 7);
    let path = dir.path().join("record.trace");
    fs::write(&path, serde_json::to_string_pretty(&original).unwrap()).unwrap();

    let restored: TraceRecord = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

    assert_eq!(restored.name, original.name);
    assert_eq!(restored.shape, original.shape);
    assert_eq!(restored.dtype, original.dtype);
    assert_eq!(restored.blake3, original.blake3);
    assert_eq!(restored.num_elements, original.num_elements);
    assert_eq!(restored.seq, original.seq);
    assert_eq!(restored.layer, original.layer);
    assert_eq!(restored.stage, original.stage);
    assert!((restored.rms - original.rms).abs() < 1e-10);
}

#[test]
fn test_import_invalid_json_returns_error() {
    let result: Result<TraceRecord, _> = serde_json::from_str("not valid json {{");
    assert!(result.is_err());
}

#[test]
fn test_import_missing_required_field_returns_error() {
    // Missing `blake3` is a required field — deserialization must fail.
    let bad = r#"{"name":"x","shape":[1],"dtype":"F32","rms":1.0,"num_elements":1}"#;
    let result: Result<TraceRecord, _> = serde_json::from_str(bad);
    assert!(result.is_err());
}
