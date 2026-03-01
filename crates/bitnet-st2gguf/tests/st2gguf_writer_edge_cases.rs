//! Edge-case and boundary tests for the bitnet-st2gguf GGUF writer and
//! LayerNorm detection utilities.

use bitnet_st2gguf::layernorm::{count_layernorm_tensors, is_layernorm_tensor};
use bitnet_st2gguf::writer::{GgufWriter, MetadataValue, TensorDType, TensorEntry};
use tempfile::NamedTempFile;

// ===========================================================================
// TensorDType
// ===========================================================================

#[test]
fn tensor_dtype_f32_gguf_type() {
    assert_eq!(TensorDType::F32.as_gguf_type(), 0);
}

#[test]
fn tensor_dtype_f16_gguf_type() {
    assert_eq!(TensorDType::F16.as_gguf_type(), 1);
}

#[test]
fn tensor_dtype_f32_element_size() {
    assert_eq!(TensorDType::F32.element_size(), 4);
}

#[test]
fn tensor_dtype_f16_element_size() {
    assert_eq!(TensorDType::F16.element_size(), 2);
}

#[test]
fn tensor_dtype_eq() {
    assert_eq!(TensorDType::F32, TensorDType::F32);
    assert_eq!(TensorDType::F16, TensorDType::F16);
    assert_ne!(TensorDType::F32, TensorDType::F16);
}

#[test]
fn tensor_dtype_debug() {
    let s = format!("{:?}", TensorDType::F32);
    assert!(s.contains("F32"));
}

#[test]
fn tensor_dtype_clone() {
    let d = TensorDType::F16;
    let d2 = d;
    assert_eq!(d, d2);
}

// ===========================================================================
// MetadataValue
// ===========================================================================

#[test]
fn metadata_value_bool_true() {
    let v = MetadataValue::Bool(true);
    let s = format!("{:?}", v);
    assert!(s.contains("true"));
}

#[test]
fn metadata_value_bool_false() {
    let v = MetadataValue::Bool(false);
    let s = format!("{:?}", v);
    assert!(s.contains("false"));
}

#[test]
fn metadata_value_u32() {
    let v = MetadataValue::U32(42);
    let s = format!("{:?}", v);
    assert!(s.contains("42"));
}

#[test]
fn metadata_value_u32_max() {
    let v = MetadataValue::U32(u32::MAX);
    let s = format!("{:?}", v);
    assert!(s.contains(&u32::MAX.to_string()));
}

#[test]
fn metadata_value_i32_negative() {
    let v = MetadataValue::I32(-1);
    let s = format!("{:?}", v);
    assert!(s.contains("-1"));
}

#[test]
fn metadata_value_f32_pi() {
    let v = MetadataValue::F32(std::f32::consts::PI);
    let s = format!("{:?}", v);
    assert!(s.contains("3.14"));
}

#[test]
fn metadata_value_string_empty() {
    let v = MetadataValue::String(String::new());
    let s = format!("{:?}", v);
    assert!(s.contains("String"));
}

#[test]
fn metadata_value_string_unicode() {
    let v = MetadataValue::String("こんにちは🌸".to_string());
    let s = format!("{:?}", v);
    assert!(s.contains("こんにちは"));
}

#[test]
fn metadata_value_clone() {
    let v = MetadataValue::String("hello".to_string());
    let v2 = v.clone();
    let s1 = format!("{:?}", v);
    let s2 = format!("{:?}", v2);
    assert_eq!(s1, s2);
}

// ===========================================================================
// TensorEntry
// ===========================================================================

#[test]
fn tensor_entry_new_basic() {
    let data = vec![0u8; 8];
    let t = TensorEntry::new("test".to_string(), vec![2, 2], TensorDType::F16, data.clone());
    assert_eq!(t.name, "test");
    assert_eq!(t.shape, vec![2, 2]);
    assert_eq!(t.dtype, TensorDType::F16);
    assert_eq!(t.data, data);
}

#[test]
fn tensor_entry_empty_data() {
    let t = TensorEntry::new("empty".to_string(), vec![0], TensorDType::F32, vec![]);
    assert!(t.data.is_empty());
    assert_eq!(t.shape, vec![0]);
}

#[test]
fn tensor_entry_scalar_shape() {
    let data = vec![0u8; 4];
    let t = TensorEntry::new("scalar".to_string(), vec![], TensorDType::F32, data);
    assert!(t.shape.is_empty());
}

#[test]
fn tensor_entry_large_shape() {
    let data = vec![0u8; 16];
    let t = TensorEntry::new("high_dim".to_string(), vec![2, 2, 2, 2], TensorDType::F16, data);
    assert_eq!(t.shape.len(), 4);
}

#[test]
fn tensor_entry_long_name() {
    let name: String = (0..500).map(|_| 'x').collect();
    let t = TensorEntry::new(name.clone(), vec![1], TensorDType::F32, vec![0; 4]);
    assert_eq!(t.name.len(), 500);
}

// ===========================================================================
// GgufWriter: metadata
// ===========================================================================

#[test]
fn writer_new_creates_instance() {
    let w = GgufWriter::new();
    // Just verify it constructs; metadata is pub(crate)
    let tmp = NamedTempFile::new().unwrap();
    w.write_to_file(tmp.path()).unwrap();
}

#[test]
fn writer_add_metadata_string() {
    let mut w = GgufWriter::new();
    w.add_metadata("key", MetadataValue::String("value".into()));
    // Verify via write — kv_count should be 1
    let tmp = NamedTempFile::new().unwrap();
    w.write_to_file(tmp.path()).unwrap();
    let bytes = std::fs::read(tmp.path()).unwrap();
    assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 1);
}

#[test]
fn writer_add_metadata_multiple_types() {
    let mut w = GgufWriter::new();
    w.add_metadata("bool_key", MetadataValue::Bool(true));
    w.add_metadata("u32_key", MetadataValue::U32(100));
    w.add_metadata("i32_key", MetadataValue::I32(-50));
    w.add_metadata("f32_key", MetadataValue::F32(3.14));
    w.add_metadata("str_key", MetadataValue::String("hello".into()));
    let tmp = NamedTempFile::new().unwrap();
    w.write_to_file(tmp.path()).unwrap();
    let bytes = std::fs::read(tmp.path()).unwrap();
    assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 5);
}

#[test]
fn writer_add_metadata_duplicate_keys() {
    let mut w = GgufWriter::new();
    w.add_metadata("dup", MetadataValue::U32(1));
    w.add_metadata("dup", MetadataValue::U32(2));
    // Both entries are kept (Vec-based storage) → kv_count = 2
    let tmp = NamedTempFile::new().unwrap();
    w.write_to_file(tmp.path()).unwrap();
    let bytes = std::fs::read(tmp.path()).unwrap();
    assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 2);
}

// ===========================================================================
// GgufWriter: write_to_file
// ===========================================================================

#[test]
fn writer_empty_writes_valid_file() {
    let tmp = NamedTempFile::new().unwrap();
    let w = GgufWriter::new();
    w.write_to_file(tmp.path()).unwrap();

    let bytes = std::fs::read(tmp.path()).unwrap();
    // Check GGUF magic
    assert_eq!(&bytes[0..4], b"GGUF");
    // Check version = 3
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 3);
    // tensor_count = 0
    assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 0);
    // kv_count = 0
    assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 0);
}

#[test]
fn writer_with_metadata_only() {
    let tmp = NamedTempFile::new().unwrap();
    let mut w = GgufWriter::new();
    w.add_metadata("general.name", MetadataValue::String("test_model".into()));
    w.add_metadata("general.file_type", MetadataValue::U32(1));
    w.write_to_file(tmp.path()).unwrap();

    let bytes = std::fs::read(tmp.path()).unwrap();
    assert_eq!(&bytes[0..4], b"GGUF");
    // kv_count = 2
    assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 2);
}

#[test]
fn writer_with_single_tensor() {
    let tmp = NamedTempFile::new().unwrap();
    let mut w = GgufWriter::new();

    let data = vec![0u8; 16]; // 8 f16 values
    w.add_tensor(TensorEntry::new("test.weight".to_string(), vec![4, 2], TensorDType::F16, data));

    w.write_to_file(tmp.path()).unwrap();

    let bytes = std::fs::read(tmp.path()).unwrap();
    assert_eq!(&bytes[0..4], b"GGUF");
    // tensor_count = 1
    assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 1);
}

#[test]
fn writer_with_multiple_tensors() {
    let tmp = NamedTempFile::new().unwrap();
    let mut w = GgufWriter::new();

    for i in 0..5 {
        let data = vec![0u8; 32];
        w.add_tensor(TensorEntry::new(
            format!("layer.{}.weight", i),
            vec![4, 4],
            TensorDType::F16,
            data,
        ));
    }

    w.write_to_file(tmp.path()).unwrap();

    let bytes = std::fs::read(tmp.path()).unwrap();
    assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 5);
}

#[test]
fn writer_with_metadata_and_tensors() {
    let tmp = NamedTempFile::new().unwrap();
    let mut w = GgufWriter::new();

    w.add_metadata("general.name", MetadataValue::String("combined".into()));
    w.add_metadata("general.layers", MetadataValue::U32(2));

    let data = vec![0u8; 8];
    w.add_tensor(TensorEntry::new("weight".to_string(), vec![4], TensorDType::F16, data));

    w.write_to_file(tmp.path()).unwrap();

    let bytes = std::fs::read(tmp.path()).unwrap();
    assert_eq!(&bytes[0..4], b"GGUF");
    assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 1); // 1 tensor
    assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 2); // 2 metadata
}

#[test]
fn writer_file_not_empty() {
    let tmp = NamedTempFile::new().unwrap();
    let w = GgufWriter::new();
    w.write_to_file(tmp.path()).unwrap();

    let size = std::fs::metadata(tmp.path()).unwrap().len();
    assert!(size > 0, "GGUF file should not be empty");
    // At minimum: header is 4+4+8+8+4+8 = 36 bytes
    assert!(size >= 36, "file too small: {size}");
}

#[test]
fn writer_alignment_32_bytes() {
    let tmp = NamedTempFile::new().unwrap();
    let mut w = GgufWriter::new();

    // Small tensor — data section should still be 32-byte aligned
    let data = vec![1u8; 6]; // odd size
    w.add_tensor(TensorEntry::new("small".to_string(), vec![3], TensorDType::F16, data));

    w.write_to_file(tmp.path()).unwrap();

    let bytes = std::fs::read(tmp.path()).unwrap();
    // data_offset should be 32-byte aligned
    let data_offset = u64::from_le_bytes(bytes[28..36].try_into().unwrap());
    assert_eq!(data_offset % 32, 0, "data_offset {data_offset} should be 32-byte aligned");
}

// ===========================================================================
// LayerNorm detection: is_layernorm_tensor
// ===========================================================================

#[test]
fn layernorm_positive_attn_norm() {
    assert!(is_layernorm_tensor("model.layers.0.attn_norm.weight"));
}

#[test]
fn layernorm_positive_ffn_norm() {
    assert!(is_layernorm_tensor("model.layers.15.ffn_norm.weight"));
}

#[test]
fn layernorm_positive_final_norm() {
    assert!(is_layernorm_tensor("model.norm.weight"));
}

#[test]
fn layernorm_positive_input_layernorm() {
    assert!(is_layernorm_tensor("model.layers.0.input_layernorm.weight"));
}

#[test]
fn layernorm_positive_post_attention() {
    assert!(is_layernorm_tensor("model.layers.31.post_attention_layernorm.weight"));
}

#[test]
fn layernorm_negative_projection() {
    assert!(!is_layernorm_tensor("model.layers.0.attn.q_proj.weight"));
}

#[test]
fn layernorm_negative_embedding() {
    assert!(!is_layernorm_tensor("model.embed_tokens.weight"));
}

#[test]
fn layernorm_negative_lm_head() {
    assert!(!is_layernorm_tensor("lm_head.weight"));
}

#[test]
fn layernorm_negative_empty() {
    assert!(!is_layernorm_tensor(""));
}

#[test]
fn layernorm_negative_bias() {
    assert!(!is_layernorm_tensor("model.layers.0.attn.k_proj.bias"));
}

// ===========================================================================
// LayerNorm detection: count_layernorm_tensors
// ===========================================================================

#[test]
fn count_layernorm_empty_list() {
    let names: Vec<&str> = vec![];
    assert_eq!(count_layernorm_tensors(names), 0);
}

#[test]
fn count_layernorm_all_match() {
    let names = vec![
        "model.layers.0.attn_norm.weight",
        "model.layers.0.ffn_norm.weight",
        "model.norm.weight",
    ];
    assert_eq!(count_layernorm_tensors(names), 3);
}

#[test]
fn count_layernorm_none_match() {
    let names =
        vec!["model.layers.0.attn.q_proj.weight", "model.embed_tokens.weight", "lm_head.weight"];
    assert_eq!(count_layernorm_tensors(names), 0);
}

#[test]
fn count_layernorm_mixed() {
    let names = vec![
        "model.layers.0.attn_norm.weight",
        "model.layers.0.attn.q_proj.weight",
        "model.layers.0.ffn_norm.weight",
        "model.layers.0.attn.k_proj.weight",
        "model.norm.weight",
    ];
    assert_eq!(count_layernorm_tensors(names), 3);
}

#[test]
fn count_layernorm_realistic_model() {
    // Simulate a 4-layer model
    let mut names = Vec::new();
    for i in 0..4 {
        names.push(format!("model.layers.{i}.attn_norm.weight"));
        names.push(format!("model.layers.{i}.ffn_norm.weight"));
        names.push(format!("model.layers.{i}.attn.q_proj.weight"));
        names.push(format!("model.layers.{i}.attn.k_proj.weight"));
        names.push(format!("model.layers.{i}.attn.v_proj.weight"));
        names.push(format!("model.layers.{i}.attn.o_proj.weight"));
        names.push(format!("model.layers.{i}.mlp.gate_proj.weight"));
        names.push(format!("model.layers.{i}.mlp.up_proj.weight"));
        names.push(format!("model.layers.{i}.mlp.down_proj.weight"));
    }
    names.push("model.norm.weight".to_string());
    names.push("model.embed_tokens.weight".to_string());
    names.push("lm_head.weight".to_string());

    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    // 4 layers × 2 norms + 1 final norm = 9
    assert_eq!(count_layernorm_tensors(name_refs), 9);
}

// ===========================================================================
// Writer: overwrite existing file
// ===========================================================================

#[test]
fn writer_overwrites_existing_file() {
    let tmp = NamedTempFile::new().unwrap();

    // Write first file
    let mut w1 = GgufWriter::new();
    w1.add_metadata("v", MetadataValue::U32(1));
    w1.write_to_file(tmp.path()).unwrap();
    let size1 = std::fs::metadata(tmp.path()).unwrap().len();

    // Overwrite with more data
    let mut w2 = GgufWriter::new();
    w2.add_metadata("v", MetadataValue::U32(2));
    w2.add_metadata("extra", MetadataValue::String("more data here".into()));
    for i in 0..3 {
        w2.add_tensor(TensorEntry::new(format!("t{i}"), vec![8], TensorDType::F16, vec![0u8; 16]));
    }
    w2.write_to_file(tmp.path()).unwrap();
    let size2 = std::fs::metadata(tmp.path()).unwrap().len();

    assert!(size2 > size1, "second write should produce larger file");

    // Verify it's valid GGUF
    let bytes = std::fs::read(tmp.path()).unwrap();
    assert_eq!(&bytes[0..4], b"GGUF");
}

// ===========================================================================
// Writer: F32 tensor data
// ===========================================================================

#[test]
fn writer_f32_tensor() {
    let tmp = NamedTempFile::new().unwrap();
    let mut w = GgufWriter::new();

    let f32_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let data: Vec<u8> = bytemuck::cast_slice(&f32_data).to_vec();

    w.add_tensor(TensorEntry::new("f32_weight".to_string(), vec![4], TensorDType::F32, data));

    w.write_to_file(tmp.path()).unwrap();

    let bytes = std::fs::read(tmp.path()).unwrap();
    assert_eq!(&bytes[0..4], b"GGUF");
}

// ===========================================================================
// Writer: stress with many tensors
// ===========================================================================

#[test]
fn writer_many_tensors_100() {
    let tmp = NamedTempFile::new().unwrap();
    let mut w = GgufWriter::new();

    for i in 0..100 {
        w.add_tensor(TensorEntry::new(
            format!("layer.{i}.weight"),
            vec![16],
            TensorDType::F16,
            vec![0u8; 32],
        ));
    }

    w.write_to_file(tmp.path()).unwrap();

    let bytes = std::fs::read(tmp.path()).unwrap();
    assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 100);
}

// ===========================================================================
// Writer: metadata value edge cases
// ===========================================================================

#[test]
fn writer_metadata_long_string() {
    let tmp = NamedTempFile::new().unwrap();
    let mut w = GgufWriter::new();
    let long_str: String = (0..10_000).map(|_| 'a').collect();
    w.add_metadata("long", MetadataValue::String(long_str));
    w.write_to_file(tmp.path()).unwrap();

    let bytes = std::fs::read(tmp.path()).unwrap();
    assert_eq!(&bytes[0..4], b"GGUF");
}

#[test]
fn writer_metadata_empty_string() {
    let tmp = NamedTempFile::new().unwrap();
    let mut w = GgufWriter::new();
    w.add_metadata("empty", MetadataValue::String(String::new()));
    w.write_to_file(tmp.path()).unwrap();

    let bytes = std::fs::read(tmp.path()).unwrap();
    assert_eq!(&bytes[0..4], b"GGUF");
}

#[test]
fn writer_metadata_f32_special_values() {
    let tmp = NamedTempFile::new().unwrap();
    let mut w = GgufWriter::new();
    w.add_metadata("zero", MetadataValue::F32(0.0));
    w.add_metadata("neg_zero", MetadataValue::F32(-0.0));
    w.add_metadata("inf", MetadataValue::F32(f32::INFINITY));
    w.add_metadata("neg_inf", MetadataValue::F32(f32::NEG_INFINITY));
    w.add_metadata("nan", MetadataValue::F32(f32::NAN));
    w.write_to_file(tmp.path()).unwrap();

    let bytes = std::fs::read(tmp.path()).unwrap();
    assert_eq!(&bytes[0..4], b"GGUF");
}

#[test]
fn writer_metadata_i32_extremes() {
    let tmp = NamedTempFile::new().unwrap();
    let mut w = GgufWriter::new();
    w.add_metadata("max", MetadataValue::I32(i32::MAX));
    w.add_metadata("min", MetadataValue::I32(i32::MIN));
    w.add_metadata("zero", MetadataValue::I32(0));
    w.write_to_file(tmp.path()).unwrap();

    let bytes = std::fs::read(tmp.path()).unwrap();
    assert_eq!(&bytes[0..4], b"GGUF");
}
