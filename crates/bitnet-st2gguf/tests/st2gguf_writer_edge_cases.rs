//! Edge-case tests for bitnet-st2gguf: GgufWriter, TensorEntry,
//! TensorDType, MetadataValue, and LayerNorm detection utilities.

use bitnet_st2gguf::layernorm::{count_layernorm_tensors, is_layernorm_tensor};
use bitnet_st2gguf::writer::{GgufWriter, MetadataValue, TensorDType, TensorEntry};
use tempfile::tempdir;

// ── TensorDType ──────────────────────────────────────────────────────

#[test]
fn tensor_dtype_f32_type_and_size() {
    assert_eq!(TensorDType::F32.element_size(), 4);
    let gguf_type = TensorDType::F32.as_gguf_type();
    assert!(gguf_type < 100, "GGUF type should be a small enum value");
}

#[test]
fn tensor_dtype_f16_type_and_size() {
    assert_eq!(TensorDType::F16.element_size(), 2);
    let gguf_type = TensorDType::F16.as_gguf_type();
    assert!(gguf_type < 100);
}

#[test]
fn tensor_dtype_eq() {
    assert_eq!(TensorDType::F32, TensorDType::F32);
    assert_ne!(TensorDType::F32, TensorDType::F16);
}

#[test]
fn tensor_dtype_debug() {
    let s = format!("{:?}", TensorDType::F32);
    assert!(s.contains("F32"));
}

// ── TensorEntry ──────────────────────────────────────────────────────

#[test]
fn tensor_entry_construction() {
    let entry = TensorEntry::new(
        "blk.0.attn_q.weight".into(),
        vec![128, 128],
        TensorDType::F32,
        vec![0u8; 128 * 128 * 4],
    );
    assert_eq!(entry.name, "blk.0.attn_q.weight");
    assert_eq!(entry.shape, vec![128, 128]);
    assert_eq!(entry.dtype, TensorDType::F32);
    assert_eq!(entry.data.len(), 128 * 128 * 4);
}

#[test]
fn tensor_entry_empty_data() {
    let entry = TensorEntry::new("empty".into(), vec![0], TensorDType::F16, vec![]);
    assert_eq!(entry.data.len(), 0);
}

#[test]
fn tensor_entry_scalar() {
    let entry = TensorEntry::new(
        "scalar".into(),
        vec![1],
        TensorDType::F32,
        vec![0, 0, 128, 63], // 1.0f32 in little-endian
    );
    assert_eq!(entry.shape, vec![1]);
}

// ── MetadataValue ────────────────────────────────────────────────────

#[test]
fn metadata_value_bool() {
    let v = MetadataValue::Bool(true);
    let s = format!("{v:?}");
    assert!(s.contains("true"));
}

#[test]
fn metadata_value_u32() {
    let v = MetadataValue::U32(42);
    let s = format!("{v:?}");
    assert!(s.contains("42"));
}

#[test]
fn metadata_value_string() {
    let v = MetadataValue::String("hello".into());
    let s = format!("{v:?}");
    assert!(s.contains("hello"));
}

#[test]
fn metadata_value_f32() {
    let v = MetadataValue::F32(3.14);
    let s = format!("{v:?}");
    assert!(s.contains("3.14"));
}

// ── GgufWriter ───────────────────────────────────────────────────────

#[test]
fn writer_new_is_empty() {
    let _w = GgufWriter::new();
    // Just verify construction succeeds
}

#[test]
fn writer_add_metadata() {
    let mut w = GgufWriter::new();
    w.add_metadata("general.architecture", MetadataValue::String("llama".into()));
    w.add_metadata("general.name", MetadataValue::String("test".into()));
    w.add_metadata("llama.context_length", MetadataValue::U32(4096));
}

#[test]
fn writer_add_tensor() {
    let mut w = GgufWriter::new();
    let data = vec![0u8; 16 * 4]; // 16 floats
    let entry = TensorEntry::new("test.weight".into(), vec![4, 4], TensorDType::F32, data);
    w.add_tensor(entry);
}

#[test]
fn writer_write_minimal_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.gguf");

    let mut w = GgufWriter::new();
    w.add_metadata("general.architecture", MetadataValue::String("test".into()));
    let data = vec![0u8; 4]; // 1 float
    w.add_tensor(TensorEntry::new("weight".into(), vec![1], TensorDType::F32, data));
    w.write_to_file(&path).unwrap();

    assert!(path.exists());
    let size = std::fs::metadata(&path).unwrap().len();
    assert!(size > 0, "GGUF file should not be empty");
}

#[test]
fn writer_write_multiple_tensors() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("multi.gguf");

    let mut w = GgufWriter::new();
    w.add_metadata("general.architecture", MetadataValue::String("test".into()));
    for i in 0..5 {
        let data = vec![0u8; 32 * 4]; // 32 floats each
        w.add_tensor(TensorEntry::new(
            format!("layer.{i}.weight"),
            vec![32],
            TensorDType::F32,
            data,
        ));
    }
    w.write_to_file(&path).unwrap();

    assert!(path.exists());
}

// ── LayerNorm detection ──────────────────────────────────────────────

#[test]
fn layernorm_tensor_matches_norms() {
    assert!(is_layernorm_tensor("blk.0.attn_norm.weight"));
    assert!(is_layernorm_tensor("blk.5.ffn_norm.weight"));
    // output_norm.weight is NOT classified as a layernorm tensor
    assert!(!is_layernorm_tensor("output_norm.weight"));
}

#[test]
fn layernorm_tensor_rejects_non_norms() {
    assert!(!is_layernorm_tensor("blk.0.attn_q.weight"));
    assert!(!is_layernorm_tensor("token_embd.weight"));
    assert!(!is_layernorm_tensor("output.weight"));
}

#[test]
fn count_layernorm_tensors_basic() {
    let names = vec![
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.attn_q.weight",
        "output_norm.weight",
        "token_embd.weight",
    ];
    let count = count_layernorm_tensors(names.into_iter());
    assert_eq!(count, 2, "should find 2 LN tensors (blk.X norms)");
}

#[test]
fn count_layernorm_tensors_empty() {
    let names: Vec<&str> = vec![];
    assert_eq!(count_layernorm_tensors(names.into_iter()), 0);
}

#[test]
fn count_layernorm_tensors_all_ln() {
    let names = vec![
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.1.attn_norm.weight",
        "blk.1.ffn_norm.weight",
    ];
    assert_eq!(count_layernorm_tensors(names.into_iter()), 4);
}
