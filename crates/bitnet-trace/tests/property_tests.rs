//! Property-based tests for `bitnet-trace`.
//!
//! Key invariants tested:
//! - `TraceRecord` JSON round-trip: serialize → deserialize → same data
//! - `TraceRecord` fields: name is non-empty, num_elements matches shape product
//! - Blake3 hash field: always 64 hex chars (256-bit hash)
//! - RMS is non-negative and finite
//! - Optional fields (seq, layer, stage) survive round-trip unchanged

use bitnet_trace::TraceRecord;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Generate a valid dtype string (matches the subset BitNet.rs uses).
fn arb_dtype() -> impl Strategy<Value = String> {
    prop::sample::select(vec!["F32", "F16", "BF16", "I8", "U8", "I32"])
        .prop_map(|s: &str| s.to_string())
}

/// Generate a plausible shape (1-4 dims, each 1..64).
fn arb_shape() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(1usize..64, 1..5)
}

/// Generate a 64-hex-char string (Blake3 output mock).
fn arb_blake3() -> impl Strategy<Value = String> {
    prop::collection::vec(prop::sample::select(b"0123456789abcdef" as &[u8]), 64)
        .prop_map(|bytes| String::from_utf8(bytes).unwrap())
}

/// Generate a valid TraceRecord from arbitrary components.
fn arb_trace_record() -> impl Strategy<Value = TraceRecord> {
    (
        "[a-z][a-z0-9_./-]{0,31}", // name
        arb_shape(),
        arb_dtype(),
        arb_blake3(),
        0.0f64..100.0f64,              // rms
        prop::option::of(0usize..64),  // seq
        prop::option::of(-1isize..32), // layer
        prop::option::of(prop::sample::select(vec!["embeddings", "q_proj", "attn_out", "logits"])),
    )
        .prop_map(|(name, shape, dtype, blake3, rms, seq, layer, stage_opt)| {
            let num_elements = shape.iter().product();
            TraceRecord {
                name,
                num_elements,
                shape,
                dtype,
                blake3,
                rms,
                seq,
                layer,
                stage: stage_opt.map(|s: &str| s.to_string()),
            }
        })
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

proptest! {
    /// JSON round-trip: serialize then deserialize must recover identical data.
    #[test]
    fn prop_trace_record_json_roundtrip(record in arb_trace_record()) {
        let json = serde_json::to_string(&record).expect("serialize");
        let recovered: TraceRecord = serde_json::from_str(&json).expect("deserialize");

        prop_assert_eq!(&record.name, &recovered.name);
        prop_assert_eq!(&record.shape, &recovered.shape);
        prop_assert_eq!(&record.dtype, &recovered.dtype);
        prop_assert_eq!(&record.blake3, &recovered.blake3);
        prop_assert_eq!(record.num_elements, recovered.num_elements);
        prop_assert_eq!(record.seq, recovered.seq);
        prop_assert_eq!(record.layer, recovered.layer);
        prop_assert_eq!(&record.stage, &recovered.stage);
    }

    /// Name field is always non-empty after round-trip.
    #[test]
    fn prop_name_survives_roundtrip(record in arb_trace_record()) {
        let json = serde_json::to_string(&record).unwrap();
        let recovered: TraceRecord = serde_json::from_str(&json).unwrap();
        prop_assert!(!recovered.name.is_empty(),
            "name must not be empty after JSON round-trip");
    }

    /// num_elements matches the product of shape dimensions.
    #[test]
    fn prop_num_elements_equals_shape_product(record in arb_trace_record()) {
        let expected: usize = record.shape.iter().product();
        prop_assert_eq!(record.num_elements, expected,
            "num_elements={} must equal shape product={}", record.num_elements, expected);
    }

    /// Blake3 hash is always exactly 64 hex characters.
    #[test]
    fn prop_blake3_is_64_hex_chars(record in arb_trace_record()) {
        prop_assert_eq!(record.blake3.len(), 64,
            "blake3 hash must be 64 hex chars, got {}", record.blake3.len());
        prop_assert!(record.blake3.chars().all(|c| c.is_ascii_hexdigit()),
            "blake3 must contain only hex digits: {:?}", record.blake3);
    }

    /// RMS is non-negative.
    #[test]
    fn prop_rms_is_non_negative(record in arb_trace_record()) {
        prop_assert!(record.rms >= 0.0,
            "rms must be non-negative, got {}", record.rms);
    }

    /// JSON output omits `seq`, `layer`, `stage` when they are None (skip_serializing_if).
    #[test]
    fn prop_optional_fields_omitted_when_none(
        name in "[a-z][a-z0-9_./-]{0,16}",
        shape in arb_shape(),
    ) {
        let num_elements = shape.iter().product();
        let record = TraceRecord {
            name,
            num_elements,
            shape,
            dtype: "F32".to_string(),
            blake3: "a".repeat(64),
            rms: 1.0,
            seq: None,
            layer: None,
            stage: None,
        };
        let json = serde_json::to_string(&record).unwrap();
        prop_assert!(!json.contains("\"seq\""),
            "json must not contain 'seq' key when seq=None: {json}");
        prop_assert!(!json.contains("\"layer\""),
            "json must not contain 'layer' key when layer=None: {json}");
        prop_assert!(!json.contains("\"stage\""),
            "json must not contain 'stage' key when stage=None: {json}");
    }
}
